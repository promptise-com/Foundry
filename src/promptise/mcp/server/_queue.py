"""MCP-native job queue for asynchronous background work.

Allows server authors to define long-running job types that agents
submit and poll for results, instead of blocking on synchronous
tool calls.

Example::

    from promptise.mcp.server import MCPServer
    from promptise.mcp.server._queue import MCPQueue

    server = MCPServer(name="analytics")
    queue = MCPQueue(server, max_workers=4)

    @queue.job(name="generate_report", timeout=60)
    async def generate_report(department: str) -> dict:
        await asyncio.sleep(10)  # long-running work
        return {"department": department, "rows": 500}

    # MCPQueue auto-registers 5 tools on the server:
    #   queue_submit, queue_status, queue_result, queue_cancel, queue_list

Jobs belong to the client (and tenant) that submitted them: the status,
result, cancel and list tools only show a caller its own jobs, unless the
caller holds the queue's admin role.  The default backend keeps jobs in
process memory — they are lost on restart and not shared across replicas.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import secrets
import time
import typing
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol, runtime_checkable

from ._cancellation import CancellationToken, CancelledError
from ._errors import ToolError, ValidationError
from ._progress import ProgressReporter

logger = logging.getLogger("promptise.server")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class JobStatus(str, Enum):
    """Lifecycle states of a queue job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class JobPriority(str, Enum):
    """Job priority levels. Higher priority jobs are dequeued first."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Job:
    """Internal mutable record tracking a single job through its lifecycle.

    Attributes:
        id: Unique job identifier (hex token).
        job_type: Registered job type name.
        args: Arguments passed at submission time.
        status: Current lifecycle state.
        priority: Scheduling priority.
        result: Return value on completion.
        error: Error message on failure.
        progress: Current progress value (0.0 to 1.0).
        progress_message: Human-readable progress status.
        created_at: Submission timestamp (monotonic).
        started_at: Execution start timestamp.
        completed_at: Completion timestamp.
        attempts: Number of execution attempts so far.
        max_retries: Maximum retry count for this job.
        timeout: Per-job timeout in seconds.
        owner_client_id: ``client_id`` of the caller that submitted the
            job (``None`` for unauthenticated callers and jobs submitted
            from Python).
        owner_tenant_id: Tenant of the submitting caller, if any.
    """

    id: str
    job_type: str
    args: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    priority: JobPriority = JobPriority.NORMAL
    result: Any = None
    error: str | None = None
    progress: float = 0.0
    progress_message: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    completed_at: float | None = None
    attempts: int = 0
    max_retries: int = 0
    timeout: float | None = None
    owner_client_id: str | None = None
    owner_tenant_id: str | None = None


@dataclass(frozen=True)
class JobDef:
    """Definition of a registered job type (immutable).

    Created at decoration time by ``@queue.job()``.

    Attributes:
        name: Unique job type name.
        handler: The async callable that executes the job.
        description: Human-readable description.
        timeout: Default timeout in seconds.
        max_retries: Default max retry count.
        backoff_base: Exponential backoff base in seconds.
        input_model: Pydantic model validating the job's arguments
            (built from the handler signature, injected parameters
            excluded).
        input_schema: JSON Schema of the job's arguments, as shown to
            clients in the ``queue_submit`` tool description.
        injections: ``(parameter, kind)`` pairs for framework-injected
            parameters; *kind* is ``"progress"`` or ``"cancel"``.
    """

    name: str
    handler: Any  # Callable
    description: str
    timeout: float = 300.0
    max_retries: int = 0
    backoff_base: float = 1.0
    input_model: Any = None
    input_schema: dict[str, Any] = field(default_factory=dict)
    injections: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class QueueCaller:
    """Identity a queue operation is performed for.

    The queue's MCP tools build one from the request context, so each
    caller only sees and controls its own jobs.  Python code calling
    :class:`MCPQueue` methods directly passes ``caller=None`` (the
    default) and is not restricted.

    Attributes:
        client_id: Authenticated ``client_id``, or ``None`` when the
            request was not authenticated.
        tenant_id: The caller's tenant, if any.
        is_admin: Holds the queue's admin role — may see and cancel every
            job in its own tenant, not just its own jobs.
    """

    client_id: str | None = None
    tenant_id: str | None = None
    is_admin: bool = False

    def can_access(self, job: Job) -> bool:
        """Whether this caller may see or cancel *job*.

        Tenants never cross: the admin role widens access to every
        client of the caller's own tenant, not to other tenants.
        """
        if job.owner_tenant_id != self.tenant_id:
            return False
        return self.is_admin or job.owner_client_id == self.client_id


# ---------------------------------------------------------------------------
# Queue backend protocol + in-memory implementation
# ---------------------------------------------------------------------------

_PRIORITY_MAP: dict[JobPriority, int] = {
    JobPriority.CRITICAL: 0,
    JobPriority.HIGH: 1,
    JobPriority.NORMAL: 2,
    JobPriority.LOW: 3,
}


@runtime_checkable
class QueueBackend(Protocol):
    """Protocol for pluggable queue storage backends.

    The default implementation is ``InMemoryQueueBackend``.
    """

    async def enqueue(self, job: Job) -> None:
        """Add a job to the queue."""
        ...

    async def dequeue(self) -> Job | None:
        """Remove and return the highest-priority pending job, or ``None``."""
        ...

    async def get(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        ...

    async def update(self, job: Job) -> None:
        """Update a job's state."""
        ...

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[Job]:
        """List jobs, optionally filtered by status."""
        ...

    async def remove(self, job_id: str) -> bool:
        """Remove a job record. Returns ``True`` if found."""
        ...

    async def count(self, status: JobStatus | None = None) -> int:
        """Count jobs, optionally filtered by status."""
        ...


class InMemoryQueueBackend:
    """In-process queue backend using asyncio primitives.

    Uses ``asyncio.PriorityQueue`` for the pending queue and a dict
    for job storage. Suitable for single-process deployments and testing:
    jobs live in this process only, so they are lost when it restarts and
    are not visible to other replicas of the server.

    Args:
        max_size: Maximum number of pending jobs (0 = unlimited).
    """

    def __init__(self, *, max_size: int = 0) -> None:
        self._jobs: dict[str, Job] = {}
        self._queue: asyncio.PriorityQueue[tuple[int, float, str]] = asyncio.PriorityQueue(
            maxsize=max_size
        )

    async def enqueue(self, job: Job) -> None:
        """Add a job to the queue."""
        self._jobs[job.id] = job
        pri = _PRIORITY_MAP.get(job.priority, 2)
        await self._queue.put((pri, job.created_at, job.id))

    async def dequeue(self) -> Job | None:
        """Remove and return the highest-priority pending job."""
        try:
            _, _, job_id = self._queue.get_nowait()
            job = self._jobs.get(job_id)
            if job is not None and job.status == JobStatus.PENDING:
                return job
            return None  # Job was cancelled while pending
        except asyncio.QueueEmpty:
            return None

    async def get(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    async def update(self, job: Job) -> None:
        """Update a job's state."""
        self._jobs[job.id] = job

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 50,
    ) -> list[Job]:
        """List jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status is not None:
            jobs = [j for j in jobs if j.status == status]
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs[:limit]

    async def remove(self, job_id: str) -> bool:
        """Remove a job record."""
        return self._jobs.pop(job_id, None) is not None

    async def count(self, status: JobStatus | None = None) -> int:
        """Count jobs, optionally filtered by status."""
        if status is None:
            return len(self._jobs)
        return sum(1 for j in self._jobs.values() if j.status == status)


# ---------------------------------------------------------------------------
# Job progress reporter
# ---------------------------------------------------------------------------


class _JobProgressReporter(ProgressReporter):
    """Writes progress updates into a Job record.

    This is what a job handler receives for a ``ProgressReporter``
    parameter: instead of MCP progress notifications, progress is stored
    on the job so agents polling ``queue_status`` see it.  Reports made
    after the job was cancelled are ignored.
    """

    def __init__(
        self,
        job: Job,
        backend: QueueBackend,
        cancel_token: CancellationToken | None = None,
    ) -> None:
        super().__init__()
        self._job = job
        self._backend = backend
        self._cancel_token = cancel_token

    async def report(
        self,
        progress: float,
        *,
        total: float | None = None,
        message: str | None = None,
    ) -> None:
        """Update job progress (a fraction from 0.0 to 1.0) and persist it."""
        if self._cancel_token is not None and self._cancel_token.is_cancelled:
            return
        if total and total > 0:
            self._job.progress = min(progress / total, 1.0)
        else:
            self._job.progress = min(progress, 1.0)
        self._job.progress_message = message
        await self._backend.update(self._job)


def _injection_kind(annotation: Any, default: Any) -> str | None:
    """Classify a job handler parameter as injected (``"progress"``/``"cancel"``) or not."""
    dependency = getattr(default, "dependency", None)
    for candidate in (annotation, dependency):
        if isinstance(candidate, type):
            if issubclass(candidate, ProgressReporter):
                return "progress"
            if issubclass(candidate, CancellationToken):
                return "cancel"
    return None


def _build_job_input(func: Any) -> tuple[Any, dict[str, Any], tuple[tuple[str, str], ...]]:
    """Build the argument model, its JSON Schema and the injections for a job handler.

    Unknown arguments are rejected (``additionalProperties: false``)
    unless the handler takes ``**kwargs``.
    """
    from pydantic import ConfigDict

    from ._validation import build_input_model

    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}
    sig = inspect.signature(func)
    injections: list[tuple[str, str]] = []
    exclude: set[str] = set()
    accepts_extra = False
    for pname, param in sig.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_extra = True
            exclude.add(pname)
            continue
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            exclude.add(pname)
            continue
        kind = _injection_kind(hints.get(pname, param.annotation), param.default)
        if kind is not None:
            injections.append((pname, kind))
            exclude.add(pname)
        elif hasattr(param.default, "dependency"):
            # Other Depends() markers are not resolved for jobs; keep them
            # out of the client-facing schema.
            exclude.add(pname)

    # Untyped parameters accept anything, as they did before validation.
    base, schema = build_input_model(func, exclude=exclude, untyped=Any)
    extra: Literal["allow", "forbid"] = "allow" if accepts_extra else "forbid"
    model = type(base.__name__, (base,), {"model_config": ConfigDict(extra=extra)})
    if not accepts_extra:
        schema["additionalProperties"] = False
    return model, schema, tuple(injections)


def _compact_schema(schema: dict[str, Any]) -> str:
    """One-line JSON of an argument schema, without pydantic's ``title`` noise."""

    def strip(node: Any, *, names: bool = False) -> Any:
        # ``names``: the keys are parameter names (a ``properties`` map),
        # so a parameter called "title" must survive.
        if isinstance(node, dict):
            return {
                k: strip(v, names=k == "properties" and not names)
                for k, v in node.items()
                if names or k != "title"
            }
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    return json.dumps(strip(schema), separators=(",", ":"), sort_keys=True)


# ---------------------------------------------------------------------------
# MCPQueue
# ---------------------------------------------------------------------------

_SUBMIT_DESCRIPTION = (
    "Submit a job to the background queue for async processing. "
    "Returns a job_id for tracking. Use {prefix}_status to poll progress "
    "and {prefix}_result to retrieve the output when complete. "
    "Arguments are validated on submission against the job type's schema."
)

_TERMINAL = (
    JobStatus.COMPLETED,
    JobStatus.FAILED,
    JobStatus.CANCELLED,
    JobStatus.TIMEOUT,
)


class MCPQueue:
    """MCP-native job queue for asynchronous background work.

    Allows server authors to define long-running job types that agents
    submit and poll for results, rather than blocking on synchronous
    tool calls.

    When attached to an ``MCPServer``, the queue auto-registers 5 MCP
    tools (``queue_submit``, ``queue_status``, ``queue_result``,
    ``queue_cancel``, ``queue_list``) and hooks into the server lifecycle
    for worker management.

    Jobs are owned by the client and tenant that submitted them.  The
    status, result, cancel and list tools only expose a caller's own jobs;
    a caller holding ``admin_role`` sees every job of its own tenant.
    Ownership relies on authenticated tool calls — on a server without
    ``require_auth=True``, pass ``auth=True`` so the queue tools
    authenticate; unauthenticated callers all share one anonymous owner.

    Args:
        server: The MCPServer to attach to. When provided, tools and
            lifecycle hooks are registered immediately.
        backend: Queue storage backend (default: InMemoryQueueBackend,
            which is process-local: jobs are lost on restart and not
            shared across replicas).
        max_workers: Maximum concurrent job workers.
        default_timeout: Default per-job timeout in seconds.
        result_ttl: How long to keep completed job results before
            auto-cleanup (seconds).
        cleanup_interval: Seconds between cleanup sweeps.
        tool_prefix: Prefix for auto-registered tool names.
        auth: Require authentication on the queue tools (in addition to
            the server's ``require_auth``).
        admin_role: Role that may see and cancel other clients' jobs
            within its own tenant.  ``None`` disables the override.
        cancel_grace_period: Seconds a cancelled running job gets to stop
            on its own (its ``CancellationToken`` is set immediately)
            before its task is cancelled.  Default ``0``: the task is
            cancelled right away.

    Example::

        server = MCPServer(name="analytics")
        queue = MCPQueue(server, max_workers=4)

        @queue.job(name="generate_report", timeout=60)
        async def generate_report(department: str) -> dict:
            await asyncio.sleep(10)
            return {"department": department, "rows": 500}

        server.run(transport="http", port=8080)
    """

    def __init__(
        self,
        server: Any | None = None,
        *,
        backend: QueueBackend | None = None,
        max_workers: int = 4,
        default_timeout: float = 300.0,
        result_ttl: float = 3600.0,
        cleanup_interval: float = 60.0,
        tool_prefix: str = "queue",
        auth: bool = False,
        admin_role: str | None = "admin",
        cancel_grace_period: float = 0.0,
    ) -> None:
        self._backend = backend or InMemoryQueueBackend()
        self._max_workers = max_workers
        self._default_timeout = default_timeout
        self._result_ttl = result_ttl
        self._cleanup_interval = cleanup_interval
        self._tool_prefix = tool_prefix
        self._auth = auth
        self._admin_role = admin_role
        self._cancel_grace_period = cancel_grace_period
        self._job_defs: dict[str, JobDef] = {}
        self._workers: list[asyncio.Task[None]] = []
        self._cleanup_task: asyncio.Task[None] | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._cancellation_tokens: dict[str, CancellationToken] = {}
        # Handler tasks of running jobs, so queue_cancel can stop a job
        # that never checks its CancellationToken.
        self._job_tasks: dict[str, asyncio.Task[Any]] = {}
        # Jobs waiting out a retry backoff (job id -> timer task).  Workers
        # never sleep through a backoff; a timer re-enqueues the job.
        self._retry_timers: dict[str, asyncio.Task[None]] = {}
        self._servers: list[Any] = []

        if server is not None:
            self.register(server)

    # ------------------------------------------------------------------
    # Decorator: @queue.job()
    # ------------------------------------------------------------------

    def job(
        self,
        name: str | None = None,
        *,
        timeout: float | None = None,
        max_retries: int = 0,
        backoff_base: float = 1.0,
    ) -> Any:
        """Register an async function as a queue job type.

        The decorated function runs in background workers, not inline
        with the tool call. It receives its arguments as keyword args,
        and may optionally accept ``ProgressReporter`` or
        ``CancellationToken`` parameters (detected by type annotation or
        a ``Depends(...)`` default).  Arguments are validated against the
        handler signature when the job is submitted, and the resulting
        schema is listed in the ``queue_submit`` tool description.

        Args:
            name: Job type name (defaults to function name).
            timeout: Per-job timeout (overrides queue default).
            max_retries: Max retry attempts on failure.
            backoff_base: Exponential backoff base in seconds.

        Example::

            @queue.job(name="generate_report", timeout=60)
            async def generate_report(department: str) -> dict:
                return {"department": department, "rows": 500}
        """

        def decorator(func: Any) -> Any:
            job_name = name or func.__name__
            if job_name in self._job_defs:
                raise ValueError(f"Job type '{job_name}' is already registered")
            description = (func.__doc__ or "").strip().split("\n")[0] or job_name
            input_model, input_schema, injections = _build_job_input(func)
            job_def = JobDef(
                name=job_name,
                handler=func,
                description=description,
                timeout=timeout or self._default_timeout,
                max_retries=max_retries,
                backoff_base=backoff_base,
                input_model=input_model,
                input_schema=input_schema,
                injections=injections,
            )
            self._job_defs[job_name] = job_def
            for server in self._servers:
                self._refresh_submit_tool(server)
            return func

        return decorator

    # ------------------------------------------------------------------
    # Registration on MCPServer
    # ------------------------------------------------------------------

    def register(self, server: Any) -> None:
        """Register queue tools and lifecycle hooks on an MCPServer.

        Called automatically when ``server`` is passed to the
        constructor. Call manually when constructing the queue
        separately.

        Args:
            server: The MCPServer instance.
        """
        self._register_tools(server)
        self._register_lifecycle(server)
        self._servers.append(server)
        self._refresh_submit_tool(server)

    def _caller_from_context(self) -> QueueCaller:
        """Build the caller identity of the current tool call."""
        from ._context import get_context

        try:
            ctx = get_context()
        except RuntimeError:
            # Only reachable when a tool handler is invoked outside the
            # server pipeline: treat it as an anonymous, non-admin caller.
            return QueueCaller()
        client = ctx.client
        roles = client.roles if client and client.roles else ctx.state.get("roles", set())
        return QueueCaller(
            client_id=ctx.client_id,
            tenant_id=getattr(client, "tenant_id", None),
            is_admin=self._admin_role is not None and self._admin_role in roles,
        )

    def _register_tools(self, server: Any) -> None:
        """Auto-register the 5 queue management MCP tools."""
        prefix = self._tool_prefix
        queue_ref = self
        auth = self._auth

        @server.tool(
            name=f"{prefix}_submit",
            description=_SUBMIT_DESCRIPTION.format(prefix=prefix),
            tags=["queue"],
            auth=auth,
        )
        async def queue_submit(
            job_type: str,
            args: dict[str, Any] | None = None,
            priority: Literal["low", "normal", "high", "critical"] = "normal",
        ) -> dict[str, Any]:
            """Submit a job for background processing.

            Args:
                job_type: The registered job type name.
                args: Arguments to pass to the job handler.
                priority: Job priority (low, normal, high, critical).
            """
            return await queue_ref.submit(
                job_type,
                args or {},
                priority=JobPriority(priority),
                caller=queue_ref._caller_from_context(),
            )

        @server.tool(
            name=f"{prefix}_status",
            description=(
                "Check the current status and progress of a queued job. "
                "Returns the status, progress as a fraction from 0.0 to 1.0, "
                "and the latest progress message."
            ),
            tags=["queue"],
            auth=auth,
        )
        async def queue_status(job_id: str) -> dict[str, Any]:
            """Get the status of a job.

            Args:
                job_id: The job identifier returned by queue_submit.
            """
            return await queue_ref.status(job_id, caller=queue_ref._caller_from_context())

        @server.tool(
            name=f"{prefix}_result",
            description=(
                "Get the result of a completed job. If the job is still "
                "running, returns the current status instead."
            ),
            tags=["queue"],
            auth=auth,
        )
        async def queue_result(job_id: str) -> dict[str, Any]:
            """Get the result of a completed job.

            Args:
                job_id: The job identifier.
            """
            return await queue_ref.get_result(job_id, caller=queue_ref._caller_from_context())

        @server.tool(
            name=f"{prefix}_cancel",
            description="Cancel a pending or running job.",
            tags=["queue"],
            auth=auth,
        )
        async def queue_cancel(job_id: str) -> dict[str, Any]:
            """Cancel a job.

            Args:
                job_id: The job identifier to cancel.
            """
            return await queue_ref.cancel(job_id, caller=queue_ref._caller_from_context())

        @server.tool(
            name=f"{prefix}_list",
            description=(
                "List your jobs in the queue, optionally filtered by status. "
                "Returns job summaries with status and progress."
            ),
            tags=["queue"],
            auth=auth,
        )
        async def queue_list(
            status: Literal["pending", "running", "completed", "failed", "cancelled", "timeout"]
            | None = None,
            limit: int = 20,
            offset: int = 0,
        ) -> dict[str, Any]:
            """List jobs in the queue.

            Args:
                status: Filter by status (pending, running, completed, failed, cancelled, timeout).
                limit: Maximum number of jobs to return (default 20).
                offset: Number of jobs to skip for pagination (default 0).
            """
            return await queue_ref.list_jobs(
                status=JobStatus(status) if status else None,
                limit=limit,
                offset=offset,
                caller=queue_ref._caller_from_context(),
            )

    def _refresh_submit_tool(self, server: Any) -> None:
        """Advertise the registered job types and their argument schemas.

        ``queue_submit`` is registered before any ``@queue.job`` runs, so
        its description and ``job_type`` enum are rebuilt whenever a job
        type is added.
        """
        import copy
        import dataclasses

        from ._types import ToolDef

        registry = getattr(server, "_tool_registry", None)
        if registry is None:
            return
        tdef = registry.get(f"{self._tool_prefix}_submit")
        if not isinstance(tdef, ToolDef):
            return

        description = _SUBMIT_DESCRIPTION.format(prefix=self._tool_prefix)
        schema = copy.deepcopy(tdef.input_schema)
        if self._job_defs:
            lines = [description, "", "Job types (pass the arguments in `args`):"]
            for jd in self._job_defs.values():
                args_schema = _compact_schema(jd.input_schema)
                lines.append(f"- {jd.name}: {jd.description} args schema: {args_schema}")
            description = "\n".join(lines)
            job_type_prop = schema.get("properties", {}).get("job_type")
            if isinstance(job_type_prop, dict):
                job_type_prop["enum"] = list(self._job_defs)
        registry.replace(dataclasses.replace(tdef, description=description, input_schema=schema))

    def _register_lifecycle(self, server: Any) -> None:
        """Hook into server startup/shutdown to manage workers."""
        queue_ref = self

        @server.on_startup
        async def _start_queue_workers() -> None:
            await queue_ref.start()

        @server.on_shutdown
        async def _stop_queue_workers() -> None:
            await queue_ref.stop()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    async def submit(
        self,
        job_type: str,
        args: dict[str, Any],
        *,
        priority: JobPriority = JobPriority.NORMAL,
        caller: QueueCaller | None = None,
    ) -> dict[str, Any]:
        """Submit a job for background execution.

        Args:
            job_type: Registered job type name.
            args: Arguments for the job handler.
            priority: Scheduling priority.
            caller: Identity that will own the job (``None`` for jobs
                submitted from server code).

        Returns:
            Dict with job_id, status, and job_type.

        Raises:
            ToolError: If the job type is not registered
                (``UNKNOWN_JOB_TYPE``) or the arguments do not match the
                handler signature (``INVALID_JOB_ARGUMENTS``).  Neither
                is retryable.
        """
        job_def = self._job_defs.get(job_type)
        if job_def is None:
            available = list(self._job_defs.keys())
            raise ToolError(
                f"Unknown job type: {job_type}",
                code="UNKNOWN_JOB_TYPE",
                retryable=False,
                suggestion=f"Available job types: {available}",
            )

        self._validate_args(job_def, args)

        job = Job(
            id=secrets.token_hex(8),
            job_type=job_type,
            args=args,
            priority=priority,
            max_retries=job_def.max_retries,
            timeout=job_def.timeout,
            owner_client_id=caller.client_id if caller else None,
            owner_tenant_id=caller.tenant_id if caller else None,
        )
        await self._backend.enqueue(job)
        logger.info(
            "Job %s submitted (type=%s, priority=%s)",
            job.id,
            job_type,
            priority.value,
        )
        return {"job_id": job.id, "status": job.status.value, "job_type": job_type}

    @staticmethod
    def _validate_args(job_def: JobDef, args: dict[str, Any]) -> dict[str, Any]:
        """Validate *args* against the job's signature; return the coerced kwargs."""
        if job_def.input_model is None:
            return dict(args)
        from ._validation import validate_arguments

        try:
            validated = validate_arguments(job_def.input_model, args)
        except ValidationError as exc:
            schema = _compact_schema(job_def.input_schema)
            raise ToolError(
                f"Invalid arguments for job type '{job_def.name}': "
                f"{str(exc).removeprefix('Invalid input: ')}",
                code="INVALID_JOB_ARGUMENTS",
                retryable=False,
                suggestion=f"Pass `args` matching this schema: {schema}",
                details={"field_errors": exc.details.get("field_errors", {})},
            ) from exc
        # A handler taking **kwargs keeps the arguments it does not name.
        for key, value in args.items():
            validated.setdefault(key, value)
        return validated

    async def _get_visible(self, job_id: str, caller: QueueCaller | None) -> Job:
        """Fetch a job the caller may access; otherwise raise ``JOB_NOT_FOUND``.

        Another caller's job is reported exactly like a missing one, so
        job ids cannot be probed.
        """
        job = await self._backend.get(job_id)
        if job is None or (caller is not None and not caller.can_access(job)):
            raise ToolError(
                f"Job not found: {job_id}",
                code="JOB_NOT_FOUND",
                retryable=False,
            )
        return job

    async def status(self, job_id: str, *, caller: QueueCaller | None = None) -> dict[str, Any]:
        """Get job status.

        Args:
            job_id: Job identifier.
            caller: Restrict to jobs this caller may access (``None`` =
                unrestricted).

        Returns:
            Dict with job status information.

        Raises:
            ToolError: If the job is not found (or not visible to *caller*).
        """
        job = await self._get_visible(job_id, caller)
        return self._job_to_dict(job)

    async def get_result(self, job_id: str, *, caller: QueueCaller | None = None) -> dict[str, Any]:
        """Get job result.

        If the job is still in progress, returns current status
        instead of a result.

        Args:
            job_id: Job identifier.
            caller: Restrict to jobs this caller may access (``None`` =
                unrestricted).

        Returns:
            Dict with job result or status.

        Raises:
            ToolError: If the job is not found (or not visible to *caller*).
        """
        job = await self._get_visible(job_id, caller)
        if job.status in (JobStatus.RUNNING, JobStatus.PENDING):
            return {
                "job_id": job.id,
                "status": job.status.value,
                "message": "Job is still in progress. Poll again later.",
                "progress": job.progress,
            }
        return self._job_to_dict(job, include_result=True)

    async def cancel(self, job_id: str, *, caller: QueueCaller | None = None) -> dict[str, Any]:
        """Cancel a job.

        A pending job is never started.  A running job's
        ``CancellationToken`` is set and its handler task is cancelled
        (after ``cancel_grace_period``, if set), so even a handler that
        never calls ``cancel.check()`` stops; anything it returns
        afterwards is discarded and the job stays ``cancelled``.

        Args:
            job_id: Job identifier.
            caller: Restrict to jobs this caller may access (``None`` =
                unrestricted).

        Returns:
            Dict with cancellation result.

        Raises:
            ToolError: If the job is not found (or not visible to *caller*).
        """
        job = await self._get_visible(job_id, caller)
        if job.status in _TERMINAL:
            return {
                "job_id": job.id,
                "status": job.status.value,
                "message": "Job already finished.",
            }
        job.status = JobStatus.CANCELLED
        job.completed_at = time.monotonic()
        job.result = None
        await self._backend.update(job)

        # Stop the running handler: signal the token for cooperative
        # handlers, then cancel the task for those that never check it.
        token = self._cancellation_tokens.get(job_id)
        if token is not None:
            token.cancel(reason="Cancelled by user")
        task = self._job_tasks.get(job_id)
        if task is not None and not task.done():
            if self._cancel_grace_period > 0:
                asyncio.get_running_loop().call_later(self._cancel_grace_period, task.cancel)
            else:
                task.cancel()
        timer = self._retry_timers.pop(job_id, None)
        if timer is not None:
            timer.cancel()
        logger.info("Job %s cancelled", job_id)
        return {"job_id": job.id, "status": "cancelled"}

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 20,
        *,
        offset: int = 0,
        caller: QueueCaller | None = None,
    ) -> dict[str, Any]:
        """List jobs, newest first.

        Args:
            status: Optional status filter.
            limit: Maximum number of jobs to return.
            offset: Number of jobs to skip (pagination).
            caller: Only list jobs this caller may access (``None`` =
                every job).

        Returns:
            Dict with the page of jobs and the total count of matching
            jobs visible to *caller*.
        """
        limit = max(limit, 0)
        offset = max(offset, 0)
        if caller is None:
            jobs = await self._backend.list_jobs(status=status, limit=offset + limit)
            total = await self._backend.count(status)
        else:
            everything = await self._backend.list_jobs(
                status=status, limit=max(await self._backend.count(status), 1)
            )
            jobs = [j for j in everything if caller.can_access(j)]
            total = len(jobs)
        return {
            "jobs": [self._job_to_dict(j) for j in jobs[offset : offset + limit]],
            "total": total,
        }

    # ------------------------------------------------------------------
    # Worker management
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the worker loop and cleanup task."""
        self._shutdown_event = asyncio.Event()
        for i in range(self._max_workers):
            task = asyncio.create_task(
                self._worker_loop(worker_id=i),
                name=f"queue-worker-{i}",
            )
            self._workers.append(task)
        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(),
            name="queue-cleanup",
        )
        logger.info(
            "Queue started: %d workers, %d job types",
            self._max_workers,
            len(self._job_defs),
        )

    async def stop(self) -> None:
        """Gracefully stop all workers.

        Jobs still running are cancelled.  Jobs waiting out a retry
        backoff are put back on the queue, so a later :meth:`start` with
        the same backend picks them up.
        """
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        for task in self._workers:
            task.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        timers, self._retry_timers = self._retry_timers, {}
        for timer in timers.values():
            timer.cancel()
        if timers:
            await asyncio.gather(*timers.values(), return_exceptions=True)
        for job_id in timers:
            await self._requeue(job_id)
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
        logger.info("Queue stopped")

    async def _worker_loop(self, *, worker_id: int) -> None:
        """Single worker: dequeue jobs and execute them."""
        while True:
            try:
                if self._shutdown_event and self._shutdown_event.is_set():
                    break

                job = await self._backend.dequeue()
                if job is None:
                    await asyncio.sleep(0.1)
                    continue

                await self._execute_job(job, worker_id=worker_id)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.debug(
                    "Worker %d encountered unexpected error",
                    worker_id,
                    exc_info=True,
                )
                await asyncio.sleep(1.0)

    async def _execute_job(self, job: Job, *, worker_id: int) -> None:
        """Execute a single job with timeout, cancellation, and retry."""
        if job.status is not JobStatus.PENDING:
            return  # cancelled between dequeue and now
        job_def = self._job_defs.get(job.job_type)
        if job_def is None:
            job.status = JobStatus.FAILED
            job.error = f"Job type '{job.job_type}' no longer registered"
            job.completed_at = time.monotonic()
            await self._backend.update(job)
            return

        cancel_token = CancellationToken()
        try:
            handler_kwargs = self._validate_args(job_def, job.args)
        except ToolError as exc:
            # Validated on submit; only reachable if the job record was
            # altered in the backend.  Not transient, so never retried.
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.completed_at = time.monotonic()
            await self._backend.update(job)
            return
        progress = _JobProgressReporter(job, self._backend, cancel_token)
        for pname, kind in job_def.injections:
            handler_kwargs[pname] = progress if kind == "progress" else cancel_token

        # Mark running
        job.status = JobStatus.RUNNING
        job.started_at = time.monotonic()
        job.attempts += 1
        self._cancellation_tokens[job.id] = cancel_token
        await self._backend.update(job)
        if cancel_token.is_cancelled:
            # queue_cancel ran while the RUNNING state was being saved
            self._cancellation_tokens.pop(job.id, None)
            return

        # The handler runs in its own task so cancellation can stop it
        # even if it never checks the token, without stopping the worker.
        timeout = job.timeout or job_def.timeout
        task = asyncio.create_task(
            self._invoke_handler(job_def.handler, handler_kwargs),
            name=f"queue-job-{job.id}",
        )
        self._job_tasks[job.id] = task
        try:
            try:
                done, _ = await asyncio.wait({task}, timeout=timeout)
            except asyncio.CancelledError:
                # The worker itself is being stopped (queue.stop()).
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                if not cancel_token.is_cancelled:
                    job.status = JobStatus.CANCELLED
                    job.error = "Queue stopped before the job finished"
                    job.completed_at = time.monotonic()
                    await self._backend.update(job)
                raise

            if cancel_token.is_cancelled:
                # Cancelled while running: cancel() already recorded the
                # CANCELLED state.  Whatever the handler produced is dropped.
                if not done:
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                logger.info("Job %s stopped after cancellation; result discarded", job.id)
                return

            if not done:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                job.status = JobStatus.TIMEOUT
                job.error = f"Job timed out after {timeout}s"
                job.completed_at = time.monotonic()
                logger.warning("Job %s timed out (worker %d)", job.id, worker_id)
            elif task.cancelled() or isinstance(task.exception(), CancelledError):
                job.status = JobStatus.CANCELLED
                job.completed_at = time.monotonic()
                logger.info("Job %s cancelled during execution", job.id)
            elif task.exception() is not None:
                failure = task.exception()
                if job.attempts < job_def.max_retries + 1:
                    if not await self._cancelled_in_backend(job):
                        await self._schedule_retry(job, job_def, failure)
                    return
                job.status = JobStatus.FAILED
                job.error = str(failure)
                job.completed_at = time.monotonic()
                logger.error("Job %s failed (worker %d): %s", job.id, worker_id, failure)
            else:
                job.status = JobStatus.COMPLETED
                job.result = task.result()
                job.error = None  # earlier failed attempts no longer apply
                job.progress = 1.0
                job.completed_at = time.monotonic()
        finally:
            self._cancellation_tokens.pop(job.id, None)
            self._job_tasks.pop(job.id, None)

        if await self._cancelled_in_backend(job):
            return
        await self._backend.update(job)

    async def _cancelled_in_backend(self, job: Job) -> bool:
        """Whether the stored record was cancelled behind this worker's back.

        With a shared backend another process can cancel a job this one is
        running; its outcome must not overwrite that.  (With the in-memory
        backend the record is the worker's own object, and the local
        cancellation token already covers it.)
        """
        stored = await self._backend.get(job.id)
        if stored is None or stored is job or stored.status is not JobStatus.CANCELLED:
            return False
        logger.info("Job %s was cancelled elsewhere; outcome discarded", job.id)
        return True

    async def _schedule_retry(self, job: Job, job_def: JobDef, exc: BaseException | None) -> None:
        """Put a failed job back to PENDING and re-enqueue it after the backoff.

        The worker returns immediately; a timer task waits out the backoff.
        """
        backoff = job_def.backoff_base * (2 ** (job.attempts - 1))
        job.status = JobStatus.PENDING
        job.error = f"Attempt {job.attempts} failed: {exc}. Retrying in {backoff}s."
        await self._backend.update(job)
        logger.info(
            "Job %s retry %d/%d after %.1fs",
            job.id,
            job.attempts,
            job_def.max_retries,
            backoff,
        )

        async def _requeue_after_backoff() -> None:
            await asyncio.sleep(backoff)
            self._retry_timers.pop(job.id, None)
            await self._requeue(job.id)

        self._retry_timers[job.id] = asyncio.create_task(
            _requeue_after_backoff(), name=f"queue-retry-{job.id}"
        )

    async def _requeue(self, job_id: str) -> None:
        """Enqueue a job again if it is still pending (not cancelled meanwhile)."""
        job = await self._backend.get(job_id)
        if job is not None and job.status is JobStatus.PENDING:
            await self._backend.enqueue(job)

    async def _invoke_handler(
        self,
        handler: Any,
        kwargs: dict[str, Any],
    ) -> Any:
        """Call the job handler, supporting both sync and async."""
        result = handler(**kwargs)
        if asyncio.iscoroutine(result):
            result = await result
        return result

    # ------------------------------------------------------------------
    # Cleanup loop
    # ------------------------------------------------------------------

    async def _cleanup_loop(self) -> None:
        """Periodically remove completed/failed jobs past their TTL."""
        try:
            while True:
                await asyncio.sleep(self._cleanup_interval)
                await self._cleanup_expired()
        except asyncio.CancelledError:
            pass

    async def _cleanup_expired(self) -> None:
        """Remove terminal jobs older than result_ttl."""
        now = time.monotonic()
        for status in _TERMINAL:
            jobs = await self._backend.list_jobs(status=status, limit=1000)
            for job in jobs:
                if job.completed_at and (now - job.completed_at) > self._result_ttl:
                    await self._backend.remove(job.id)

    # ------------------------------------------------------------------
    # Health integration
    # ------------------------------------------------------------------

    def register_health(self, health: Any) -> None:
        """Add queue health checks to a HealthCheck instance.

        Args:
            health: The HealthCheck to register on.
        """
        backend = self._backend

        async def _queue_health() -> bool:
            pending = await backend.count(JobStatus.PENDING)
            return pending < 1000

        health.add_check("queue", _queue_health, required_for_ready=True)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def job_types(self) -> list[str]:
        """List registered job type names."""
        return list(self._job_defs.keys())

    @property
    def backend(self) -> QueueBackend:
        """The queue storage backend."""
        return self._backend

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _job_to_dict(
        job: Job,
        *,
        include_result: bool = False,
    ) -> dict[str, Any]:
        """Serialize a Job to a dict for MCP tool responses."""
        d: dict[str, Any] = {
            "job_id": job.id,
            "job_type": job.job_type,
            "status": job.status.value,
            "priority": job.priority.value,
            "progress": job.progress,
            "attempts": job.attempts,
        }
        if job.progress_message:
            d["progress_message"] = job.progress_message
        if job.error:
            d["error"] = job.error
        if include_result and job.result is not None:
            d["result"] = job.result
        return d
