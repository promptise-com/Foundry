"""AgentProcess — lifecycle container for a long-running agent.

Wraps a :class:`~promptise.agent.PromptiseAgent` with:

* State machine lifecycle (CREATED → RUNNING → STOPPED)
* Trigger queue: events from triggers are enqueued, then processed
* Heartbeat loop with configurable interval
* Concurrent invocation control via ``asyncio.Semaphore``
* :class:`~promptise.runtime.context.AgentContext` as the unified
  context layer
* Short-term memory via :class:`ConversationBuffer`
* Long-term memory via :class:`~promptise.memory.MemoryProvider`
* Open mode: dynamic self-modification via meta-tools and hot-reload

Example::

    from promptise.runtime import AgentProcess, ProcessConfig, TriggerConfig

    process = AgentProcess(
        name="data-watcher",
        config=ProcessConfig(
            model="openai:gpt-5-mini",
            instructions="You monitor data pipelines.",
            triggers=[
                TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
            ],
        ),
    )
    await process.start()
    # Process runs until stopped …
    await process.stop()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from .config import ExecutionMode, ProcessConfig
from .context import AgentContext
from .conversation import ConversationBuffer
from .lifecycle import ProcessLifecycle, ProcessState
from .triggers import create_trigger
from .triggers.base import BaseTrigger, TriggerEvent

if TYPE_CHECKING:
    from promptise.config import HTTPServerSpec, StdioServerSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory provider factory
# ---------------------------------------------------------------------------


def _create_memory_provider(ctx_config: Any) -> Any:
    """Create a MemoryProvider from ContextConfig settings.

    Args:
        ctx_config: A :class:`ContextConfig` instance.

    Returns:
        A :class:`~promptise.memory.MemoryProvider` instance, or ``None``
        if no provider is configured.
    """
    provider_type = ctx_config.memory_provider
    if provider_type is None:
        return None

    if provider_type == "in_memory":
        from promptise.memory import InMemoryProvider

        return InMemoryProvider()

    if provider_type == "chroma":
        from promptise.memory import ChromaProvider

        return ChromaProvider(
            collection_name=ctx_config.memory_collection,
            persist_directory=ctx_config.memory_persist_directory,
        )

    if provider_type == "mem0":
        from promptise.memory import Mem0Provider

        return Mem0Provider(user_id=ctx_config.memory_user_id)

    logger.warning("Unknown memory provider type: %s", provider_type)
    return None


def _resolve_server_specs(
    servers: dict[str, Any],
) -> dict[str, HTTPServerSpec | StdioServerSpec]:
    """Coerce raw server specs into typed :class:`ServerSpec` objects.

    Entries that are already :class:`HTTPServerSpec` / :class:`StdioServerSpec`
    pass through unchanged. Dict entries are normalized, carrying through
    **every** supported field so a dict-based config behaves identically to
    passing the spec object directly — an HTTP dict keeps ``headers``,
    ``auth``, ``bearer_token``, ``api_key`` and ``audience``; a stdio dict
    keeps ``args``, ``env``, ``cwd`` and ``keep_alive``. Unrecognized entries
    (neither a spec object nor a dict with a ``url``/``command``) are skipped.

    Args:
        servers: Mapping of server name to spec object or raw dict.

    Returns:
        Mapping of server name to a typed spec object.
    """
    from promptise.config import HTTPServerSpec, StdioServerSpec

    resolved: dict[str, HTTPServerSpec | StdioServerSpec] = {}
    for name, spec in servers.items():
        if isinstance(spec, (HTTPServerSpec, StdioServerSpec)):
            resolved[name] = spec
        elif isinstance(spec, dict):
            transport = spec.get("type", spec.get("transport", ""))
            if transport in ("http", "streamable-http", "sse") or "url" in spec:
                kw: dict[str, Any] = {"url": spec["url"]}
                if "transport" in spec:
                    kw["transport"] = spec["transport"]
                elif "type" in spec:
                    kw["transport"] = spec["type"]
                for opt in ("headers", "auth", "bearer_token", "api_key", "audience"):
                    if opt in spec:
                        kw[opt] = spec[opt]
                resolved[name] = HTTPServerSpec(**kw)
            elif "command" in spec:
                resolved[name] = StdioServerSpec(
                    command=spec["command"],
                    args=spec.get("args", []),
                    env=spec.get("env", {}),
                    cwd=spec.get("cwd"),
                    keep_alive=spec.get("keep_alive", True),
                )
    return resolved


class AgentProcess:
    """Lifecycle container for a long-running agent process.

    Args:
        name: Unique process name.
        config: Process configuration.
        process_id: Unique ID (auto-generated if not provided).
        event_bus: Optional shared EventBus for inter-process events.
        broker: Optional shared MessageBroker for message triggers.
        runtime: Optional parent :class:`AgentRuntime` reference
            (enables the ``spawn_process`` meta-tool in open mode).
    """

    def __init__(
        self,
        name: str,
        config: ProcessConfig,
        *,
        process_id: str | None = None,
        event_bus: Any | None = None,
        broker: Any | None = None,
        runtime: Any | None = None,
        event_notifier: Any | None = None,
    ) -> None:
        self.name = name
        self.process_id = process_id or str(uuid4())
        self.config = config
        self._event_notifier = event_notifier

        # Message inbox (human-to-agent communication)
        self._inbox = None
        if config.inbox.enabled:
            from .inbox import MessageInbox

            self._inbox = MessageInbox(
                max_messages=config.inbox.max_messages,
                max_message_length=config.inbox.max_message_length,
                default_ttl=config.inbox.default_ttl,
                max_ttl=config.inbox.max_ttl,
                rate_limit_per_sender=config.inbox.rate_limit_per_sender,
            )

        # Lifecycle
        self._lifecycle = ProcessLifecycle()

        # Memory (long-term)
        self._long_term_memory: Any | None = _create_memory_provider(config.context)

        # Context (with memory wired in)
        self._context = AgentContext(
            writable_keys=config.context.writable_keys or None,
            memory_provider=self._long_term_memory,
            file_mounts=config.context.file_mounts,
            env_prefix=config.context.env_prefix,
            initial_state=config.context.initial_state,
        )

        # Short-term memory (conversation buffer)
        self._conversation_buffer = ConversationBuffer(
            max_messages=config.context.conversation_max_messages,
        )

        # Agent (built lazily in start())
        self._agent: Any | None = None

        # MCP multi-client (native Promptise client for tool access)
        self._mcp_multi: Any | None = None
        self._mcp_adapter: Any | None = None

        # Triggers (static, from config)
        self._event_bus = event_bus
        self._broker = broker
        self._triggers: list[BaseTrigger] = []
        self._trigger_queue: asyncio.Queue[TriggerEvent] = asyncio.Queue(maxsize=1000)

        # Runtime reference (for spawn_process meta-tool)
        self._runtime = runtime

        # Dynamic state (open mode only)
        self._dynamic_instructions: str | None = None
        self._dynamic_servers: dict[str, Any] = {}
        self._custom_tools: list[Any] = []
        self._dynamic_triggers: list[BaseTrigger] = []
        self._spawned_processes: list[str] = []
        self._rebuild_lock = asyncio.Lock()
        self._rebuild_count = 0

        # Background tasks
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._trigger_listener_tasks: list[asyncio.Task[None]] = []
        self._heartbeat_task: asyncio.Task[None] | None = None

        # Concurrency
        self._semaphore = asyncio.Semaphore(config.concurrency)
        self._lock = asyncio.Lock()

        # -- Governance subsystems (zero overhead when disabled) --
        self._secrets: Any | None = None
        self._budget: Any | None = None
        self._budget_enforcer: Any | None = None
        self._health: Any | None = None
        self._mission: Any | None = None
        self._runtime_callback: Any | None = None
        self._init_governance()

        # Counters
        self._invocation_count = 0
        self._consecutive_failures = 0
        self._start_time: float | None = None
        self._last_activity: float | None = None

    # ------------------------------------------------------------------
    # Governance init
    # ------------------------------------------------------------------

    def _init_governance(self) -> None:
        """Initialise governance subsystems based on config.

        Only creates instances when the feature is explicitly enabled,
        so there is zero overhead by default.
        """
        cfg = self.config

        if cfg.secrets.enabled:
            from .secrets import SecretScope

            self._secrets = SecretScope(
                config=cfg.secrets,
                process_id=self.process_id,
            )

        if cfg.budget.enabled:
            from .budget import BudgetEnforcer, BudgetState

            self._budget = BudgetState(cfg.budget)
            self._budget_enforcer = BudgetEnforcer(cfg.budget)

        if cfg.health.enabled:
            from .health import HealthMonitor

            self._health = HealthMonitor(cfg.health, self.process_id)

        if cfg.mission.enabled:
            from .mission import MissionTracker

            self._mission = MissionTracker(
                config=cfg.mission,
                process_id=self.process_id,
            )

        # Create callback handler when budget or health is active
        if self._budget is not None or self._health is not None:
            from .callbacks import RuntimeCallbackHandler

            self._runtime_callback = RuntimeCallbackHandler(
                budget=self._budget,
                health=self._health,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self) -> ProcessState:
        """Current process state."""
        return self._lifecycle.state

    @property
    def context(self) -> AgentContext:
        """The unified agent context."""
        return self._context

    @property
    def lifecycle(self) -> ProcessLifecycle:
        """The lifecycle state machine (for inspection)."""
        return self._lifecycle

    async def start(self) -> None:
        """Build agent, start triggers, begin processing.

        Transitions: ``CREATED → STARTING → RUNNING``
        (or ``STOPPED/FAILED → STARTING → RUNNING`` on restart).

        Raises:
            ProcessStateError: If the transition is invalid.
        """
        async with self._lock:
            await self._lifecycle.transition(ProcessState.STARTING, reason="start() called")

            try:
                # 0. Resolve secrets before agent build
                if self._secrets is not None:
                    await self._secrets.resolve_initial()

                # 1. Build the agent
                await self._build_agent()

                # 2. Create and start triggers
                self._triggers = self._create_triggers()
                for trigger in self._triggers:
                    await trigger.start()

                # 3. Start trigger listener tasks
                for trigger in self._triggers:
                    task = asyncio.create_task(
                        self._trigger_listener(trigger),
                        name=f"{self.name}-trigger-{trigger.trigger_id}",
                    )
                    self._trigger_listener_tasks.append(task)

                # 4. Start worker tasks
                for i in range(self.config.concurrency):
                    task = asyncio.create_task(
                        self._worker_loop(),
                        name=f"{self.name}-worker-{i}",
                    )
                    self._worker_tasks.append(task)

                # 5. Start heartbeat
                self._heartbeat_task = asyncio.create_task(
                    self._heartbeat_loop(),
                    name=f"{self.name}-heartbeat",
                )

                self._start_time = time.monotonic()
                self._last_activity = time.monotonic()

                await self._lifecycle.transition(ProcessState.RUNNING, reason="startup complete")
                logger.info("AgentProcess %s started", self.name)
                if self._event_notifier is not None:
                    from promptise.events import emit_event

                    emit_event(
                        self._event_notifier,
                        "process.started",
                        "info",
                        {"process_name": self.name, "process_id": self.process_id},
                        agent_id=self.name,
                    )

            except Exception as exc:
                await self._lifecycle.transition(
                    ProcessState.FAILED,
                    reason=f"startup failed: {exc}",
                    metadata={"error": str(exc)},
                )
                if self._event_notifier is not None:
                    from promptise.events import emit_event

                    emit_event(
                        self._event_notifier,
                        "process.failed",
                        "critical",
                        {"process_name": self.name, "error": str(exc)[:200]},
                        agent_id=self.name,
                    )
                raise

    async def stop(self) -> None:
        """Gracefully stop: cancel workers, stop triggers, shutdown agent.

        Transitions: ``* → STOPPING → STOPPED``

        If the process is in FAILED state, cleanup is performed and the
        state transitions to ``FAILED → STARTING`` is **not** attempted;
        instead we go straight to ``STOPPED`` via internal reset.
        """
        async with self._lock:
            if self.state in (ProcessState.STOPPED, ProcessState.STOPPING):
                return

            # FAILED state can't transition to STOPPING, so handle cleanup
            # without state machine for already-failed processes
            is_failed = self.state == ProcessState.FAILED
            if not is_failed:
                await self._lifecycle.transition(ProcessState.STOPPING, reason="stop() called")

            # 1. Cancel worker tasks
            for task in self._worker_tasks:
                task.cancel()
            for task in self._worker_tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            # 2. Cancel trigger listeners
            for task in self._trigger_listener_tasks:
                task.cancel()
            for task in self._trigger_listener_tasks:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            # 3. Cancel heartbeat
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._heartbeat_task

            # 4. Stop triggers (static + dynamic)
            for trigger in self._triggers:
                with contextlib.suppress(Exception):
                    await trigger.stop()
            for trigger in self._dynamic_triggers:
                with contextlib.suppress(Exception):
                    await trigger.stop()

            # 5. Shutdown agent
            if self._agent is not None:
                with contextlib.suppress(Exception):
                    await self._agent.shutdown()

            # 6. Close long-term memory provider
            if self._long_term_memory is not None:
                with contextlib.suppress(Exception):
                    result = self._long_term_memory.close()
                    if hasattr(result, "__await__"):
                        await result

            # 7. Revoke secrets
            if self._secrets is not None:
                with contextlib.suppress(Exception):
                    await self._secrets.revoke_all()

            # Clear
            self._worker_tasks.clear()
            self._trigger_listener_tasks.clear()
            self._heartbeat_task = None
            self._triggers.clear()
            self._dynamic_triggers.clear()
            self._conversation_buffer.clear()

            if not is_failed:
                await self._lifecycle.transition(ProcessState.STOPPED, reason="shutdown complete")
                if self._event_notifier is not None:
                    from promptise.events import emit_event

                    emit_event(
                        self._event_notifier,
                        "process.stopped",
                        "info",
                        {"process_name": self.name, "process_id": self.process_id},
                        agent_id=self.name,
                    )
            # For failed processes, we leave them in FAILED state
            # (they can be restarted via start())
            logger.info("AgentProcess %s stopped", self.name)

    async def suspend(self) -> None:
        """Pause processing without tearing down the agent.

        Triggers continue to fire but events are queued, not processed.
        """
        await self._lifecycle.transition(ProcessState.SUSPENDED, reason="suspend() called")
        logger.info("AgentProcess %s suspended", self.name)

    async def resume(self) -> None:
        """Resume from SUSPENDED or AWAITING state."""
        await self._lifecycle.transition(ProcessState.RUNNING, reason="resume() called")
        logger.info("AgentProcess %s resumed", self.name)

    async def inject(self, event: TriggerEvent) -> None:
        """Manually inject a trigger event into the queue.

        Args:
            event: The trigger event to inject.
        """
        try:
            self._trigger_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning(
                "AgentProcess %s: trigger queue full, dropping event",
                self.name,
            )

    async def send_message(
        self,
        content: str,
        *,
        message_type: str = "context",
        priority: str = "normal",
        sender_id: str | None = None,
        ttl: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Send a human message to the running agent.

        The agent sees the message as additional context on its next
        invocation cycle.

        Args:
            content: Message text.
            message_type: One of ``"directive"``, ``"context"``,
                ``"question"``, ``"correction"``.
            priority: ``"low"``, ``"normal"``, ``"high"``, ``"critical"``.
            sender_id: Who sent it (for audit trail).
            ttl: Time-to-live in seconds.
            metadata: Custom data.

        Returns:
            The message ID.

        Raises:
            RuntimeError: If inbox is not enabled.
            ValueError: If rate limit exceeded.
        """
        if self._inbox is None:
            raise RuntimeError(
                f"Message inbox not enabled for process '{self.name}'. "
                "Set inbox.enabled=true in ProcessConfig."
            )
        from .lifecycle import ProcessState

        if self.state in (ProcessState.STOPPED, ProcessState.FAILED):
            raise RuntimeError(
                f"Cannot send messages to process '{self.name}' in {self.state.value} state"
            )
        from .inbox import InboxMessage, MessageType

        expires_at = None
        if ttl and ttl > 0:
            expires_at = time.time() + ttl

        message = InboxMessage(
            content=content,
            message_type=MessageType(message_type),
            sender_id=sender_id,
            priority=priority,
            expires_at=expires_at,
            metadata=metadata or {},
        )
        return await self._inbox.add(message)

    async def ask(
        self,
        content: str,
        *,
        sender_id: str | None = None,
        timeout: float = 120,
    ) -> Any:
        """Ask the agent a question and wait for the response.

        The question is delivered to the agent on its next invocation.
        This method blocks until the agent responds or the timeout
        expires.

        Args:
            content: The question text.
            sender_id: Who is asking.
            timeout: Maximum seconds to wait for an answer.

        Returns:
            An :class:`InboxResponse` with the agent's answer.

        Raises:
            RuntimeError: If inbox is not enabled.
            asyncio.TimeoutError: If timeout expires.
        """
        if self._inbox is None:
            raise RuntimeError(f"Message inbox not enabled for process '{self.name}'.")
        from .lifecycle import ProcessState

        if self.state in (ProcessState.STOPPED, ProcessState.FAILED):
            raise RuntimeError(
                f"Cannot ask questions to process '{self.name}' in {self.state.value} state"
            )
        from .inbox import InboxMessage, MessageType

        message = InboxMessage(
            content=content,
            message_type=MessageType.QUESTION,
            sender_id=sender_id,
            expires_at=time.time() + timeout + 60,
        )
        msg_id = await self._inbox.add(message)
        return await self._inbox.wait_for_response(msg_id, timeout=timeout)

    def status(self) -> dict[str, Any]:
        """Serializable status snapshot.

        Returns:
            Dict with process name, state, counters, uptime, etc.
        """
        uptime = None
        if self._start_time is not None:
            uptime = time.monotonic() - self._start_time

        status: dict[str, Any] = {
            "name": self.name,
            "process_id": self.process_id,
            "state": self.state.value,
            "execution_mode": self.config.execution_mode.value,
            "invocation_count": self._invocation_count,
            "consecutive_failures": self._consecutive_failures,
            "trigger_count": len(self._triggers),
            "dynamic_trigger_count": len(self._dynamic_triggers),
            "custom_tool_count": len(self._custom_tools),
            "rebuild_count": self._rebuild_count,
            "spawned_process_count": len(self._spawned_processes),
            "conversation_messages": len(self._conversation_buffer),
            "has_memory": self._long_term_memory is not None,
            "queue_size": self._trigger_queue.qsize(),
            "uptime_seconds": uptime,
        }

        # Governance status
        if self._budget is not None:
            status["budget"] = self._budget.remaining()
        if self._health is not None:
            status["health_anomalies"] = len(self._health.anomalies)
        if self._mission is not None:
            status["mission_state"] = self._mission.state.value
        if self._secrets is not None:
            status["active_secrets"] = self._secrets.active_secret_count

        return status

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> AgentProcess:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Internal: Agent Building
    # ------------------------------------------------------------------

    async def _build_agent(self) -> None:
        """Build the PromptiseAgent via build_agent().

        Merges static config with any dynamic overrides (open mode).

        Pre-discovers MCP tools via MCPMultiClient and passes them as
        extra_tools so the agent is built without holding server
        connections open across asyncio task boundaries.
        """
        from promptise.agent import build_agent as _build

        # Merge servers: static + dynamic
        servers = dict(self.config.servers or {})
        servers.update(self._dynamic_servers)

        # Use dynamic instructions if set, otherwise config instructions
        instructions = self._dynamic_instructions or self.config.instructions or ""

        # Collect extra tools: custom tools + meta-tools (open mode)
        extra_tools: list[Any] = list(self._custom_tools)
        if self.config.execution_mode == ExecutionMode.OPEN:
            from .meta_tools import create_meta_tools

            extra_tools.extend(create_meta_tools(self, runtime=self._runtime))

        # Pre-discover MCP tools via native client (cross-task safe),
        # then build the agent WITHOUT servers (tools come via extra_tools).
        if servers:
            from promptise.config import HTTPServerSpec
            from promptise.mcp.client import MCPClient, MCPMultiClient, MCPToolAdapter

            # Ensure all server specs are proper ServerSpec objects,
            # carrying through every supported field (see
            # :func:`_resolve_server_specs`).
            resolved = _resolve_server_specs(servers)

            # When the process carries a verifiable identity, present its
            # credential to MCP servers that have no bearer of their own —
            # scoped to each server's audience (best-effort; an unreachable
            # IdP must not fail the build).
            def _identity_bearer(spec: HTTPServerSpec) -> str | None:
                identity = self.config.identity
                if identity is None or not getattr(identity, "is_verifiable", False):
                    return None
                from promptise.identity import IdentityError

                try:
                    return str(identity.get_credential(spec.audience))
                except IdentityError as exc:
                    logger.warning(
                        "AgentProcess %s: identity could not acquire a "
                        "credential for MCP server audience %r (%s); "
                        "connecting without it.",
                        self.name,
                        spec.audience,
                        exc,
                    )
                    return None

            # Build native MCP clients from resolved specs
            clients: dict[str, MCPClient] = {}
            for name, spec in resolved.items():
                if isinstance(spec, HTTPServerSpec):
                    clients[name] = MCPClient(
                        url=spec.url,
                        transport=spec.transport,
                        headers=spec.headers,
                        bearer_token=spec.bearer_token.get_secret_value()
                        if spec.bearer_token
                        else _identity_bearer(spec),
                        api_key=spec.api_key.get_secret_value() if spec.api_key else None,
                    )
                else:
                    clients[name] = MCPClient(
                        transport="stdio",
                        command=spec.command,
                        args=spec.args,
                        env=spec.env,
                        cwd=spec.cwd,
                    )

            self._mcp_multi = MCPMultiClient(clients)
            await self._mcp_multi.__aenter__()
            self._mcp_adapter = MCPToolAdapter(self._mcp_multi)
            mcp_tools = await self._mcp_adapter.as_langchain_tools()
            extra_tools.extend(mcp_tools)

        # Build kwargs — add optional capabilities from config
        build_kwargs: dict[str, Any] = {
            "servers": None,  # Tools already added via extra_tools
            "model": self.config.model,
            "instructions": instructions,
            "memory": self._long_term_memory,
            "memory_auto_store": self.config.context.memory_auto_store,
            "memory_max_results": self.config.context.memory_max,
            "memory_min_score": self.config.context.memory_min_score,
            "memory_timeout": self.config.context.memory_timeout,
            "extra_tools": extra_tools or None,
        }

        # Wire optional capabilities from ProcessConfig
        if self.config.identity is not None:
            build_kwargs["identity"] = self.config.identity
        if self.config.approval is not None:
            build_kwargs["approval"] = self.config.approval
        if self._event_notifier is not None:
            build_kwargs["events"] = self._event_notifier
        if hasattr(self.config, "observe") and self.config.observe:
            build_kwargs["observe"] = self.config.observe
        if hasattr(self.config, "guardrails") and self.config.guardrails:
            build_kwargs["guardrails"] = self.config.guardrails
        if hasattr(self.config, "cache") and self.config.cache:
            build_kwargs["cache"] = self.config.cache
        if hasattr(self.config, "optimize_tools") and self.config.optimize_tools:
            build_kwargs["optimize_tools"] = self.config.optimize_tools
        if hasattr(self.config, "adaptive") and self.config.adaptive:
            build_kwargs["adaptive"] = self.config.adaptive
        if hasattr(self.config, "max_invocation_time") and self.config.max_invocation_time:
            build_kwargs["max_invocation_time"] = self.config.max_invocation_time

        self._agent = await _build(**build_kwargs)

    def _create_triggers(self) -> list[BaseTrigger]:
        """Instantiate triggers from config."""
        triggers: list[BaseTrigger] = []
        for trigger_config in self.config.triggers:
            trigger = create_trigger(
                trigger_config,
                event_bus=self._event_bus,
                broker=self._broker,
            )
            triggers.append(trigger)
        return triggers

    # ------------------------------------------------------------------
    # Internal: Background Loops
    # ------------------------------------------------------------------

    async def _trigger_listener(self, trigger: BaseTrigger) -> None:
        """Background task: listen for trigger events and enqueue them.

        Runs continuously until cancelled or the trigger errors.
        """
        try:
            while self.state not in (
                ProcessState.STOPPED,
                ProcessState.STOPPING,
            ):
                try:
                    event = await trigger.wait_for_next()
                    try:
                        self._trigger_queue.put_nowait(event)
                    except asyncio.QueueFull:
                        logger.warning("AgentProcess %s: trigger queue full", self.name)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception(
                        "AgentProcess %s: trigger %s error",
                        self.name,
                        trigger.trigger_id,
                    )
                    await asyncio.sleep(1)  # back off on error
        except asyncio.CancelledError:
            return

    async def _worker_loop(self) -> None:
        """Background task: dequeue trigger events and invoke the agent.

        Respects the concurrency semaphore and tracks consecutive
        failures for automatic FAILED state transition.
        """
        try:
            while True:
                event = await self._trigger_queue.get()

                # Don't process if suspended/awaiting
                if self.state in (
                    ProcessState.SUSPENDED,
                    ProcessState.AWAITING,
                ):
                    # Re-queue the event
                    try:
                        self._trigger_queue.put_nowait(event)
                    except asyncio.QueueFull:
                        pass
                    await asyncio.sleep(0.5)
                    continue

                if self.state not in (ProcessState.RUNNING,):
                    continue

                async with self._semaphore:
                    try:
                        await self._invoke_agent(event)
                        self._consecutive_failures = 0
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        self._consecutive_failures += 1
                        # Record error for health error-rate tracking
                        if self._health is not None:
                            with contextlib.suppress(Exception):
                                await self._health.record_error()
                        logger.exception(
                            "AgentProcess %s: invocation failed (consecutive=%d/%d)",
                            self.name,
                            self._consecutive_failures,
                            self.config.max_consecutive_failures,
                        )
                        if self._consecutive_failures >= self.config.max_consecutive_failures:
                            logger.error(
                                "AgentProcess %s: max failures reached, transitioning to FAILED",
                                self.name,
                            )
                            with contextlib.suppress(Exception):
                                await self._lifecycle.transition(
                                    ProcessState.FAILED,
                                    reason="max consecutive failures",
                                )
                            return
        except asyncio.CancelledError:
            return

    async def _invoke_agent(self, event: TriggerEvent) -> Any:
        """Single agent invocation with context and memory injection.

        1. Inject context state as system message
        2. Inject conversation history (short-term memory)
        3. Format trigger event as user message
        4. Invoke agent (long-term memory auto-injected by PromptiseAgent)
        5. Update conversation buffer with exchange
        6. Update counters
        """
        if self._agent is None:
            raise RuntimeError("Agent not built")

        # -- Pre-invoke: reset callback handler for this invocation --
        if self._runtime_callback is not None:
            self._runtime_callback.reset()

        # -- Pre-invoke: check daily budget reset + record run + reset per-run --
        if self._budget is not None:
            did_reset = await self._budget.check_daily_reset()
            if did_reset and self._event_notifier is not None:
                from promptise.events import emit_event

                emit_event(
                    self._event_notifier,
                    "budget.daily_reset",
                    "info",
                    {"process_name": self.name},
                    agent_id=self.name,
                )
            run_violation = await self._budget.record_run_start()
            if run_violation is not None and self._budget_enforcer is not None:
                await self._budget_enforcer.handle_violation(run_violation, self)
                return None
            await self._budget.reset_run()

        # -- Pre-invoke: check mission timeout / invocation limits --
        if self._mission is not None:
            if self._mission.state.value in ("completed", "failed"):
                logger.info(
                    "AgentProcess %s: mission already %s, skipping",
                    self.name,
                    self._mission.state.value,
                )
                return None
            if self._mission.is_timed_out():
                self._mission.fail("Mission timeout reached")
                logger.info("AgentProcess %s: mission timed out", self.name)
                # Emit mission.failed event
                if self._event_notifier is not None:
                    from promptise.events import emit_event

                    emit_event(
                        self._event_notifier,
                        "mission.failed",
                        "critical",
                        {"process_name": self.name, "reason": "timeout"},
                        agent_id=self.name,
                    )
                return None

        # Format the trigger event as a user message
        message = f"[Trigger: {event.trigger_type}] Payload: {event.payload}"
        user_msg: dict[str, Any] = {"role": "user", "content": message}

        # Build messages list with full context
        messages: list[dict[str, Any]] = []

        # 1. Inject context state if available
        state = self._context.state_snapshot()
        if state:
            messages.append({"role": "system", "content": f"[Context State] {state}"})

        # 2. Inject budget remaining into context
        if self._budget is not None and self.config.budget.inject_remaining:
            remaining = self._budget.remaining()
            messages.append({"role": "system", "content": f"[Budget Remaining] {remaining}"})

        # 3. Inject mission context
        if self._mission is not None:
            mission_ctx = self._mission.context_summary()
            messages.append({"role": "system", "content": f"[Mission] {mission_ctx}"})

        # 3.5. Inject inbox messages (human operator communication)
        _inbox_questions: list[Any] = []
        if self._inbox is not None:
            pending = await self._inbox.get_pending()
            if pending:
                from .inbox import format_inbox_for_prompt

                inbox_block = format_inbox_for_prompt(pending)
                if inbox_block:
                    messages.append({"role": "system", "content": inbox_block})
                # Track questions for answer extraction later
                _inbox_questions = [
                    m
                    for m in pending
                    if hasattr(m, "message_type") and m.message_type.value == "question"
                ]

        # 4. Inject conversation history (short-term memory)
        history = await self._conversation_buffer.async_snapshot()
        messages.extend(history)

        # 5. Append current user message
        messages.append(user_msg)

        # 6. Invoke (long-term memory auto-injected by PromptiseAgent)
        invoke_config: dict[str, Any] = {}
        if self._runtime_callback is not None:
            invoke_config["callbacks"] = [self._runtime_callback]

        result = await self._agent.ainvoke(
            {"messages": messages},
            config=invoke_config if invoke_config else None,
        )

        # 7. Update conversation buffer with this exchange
        await self._conversation_buffer.async_append(user_msg)
        if isinstance(result, dict) and "messages" in result:
            for msg in result["messages"]:
                role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", None)
                if role in ("assistant", "ai"):
                    content = (
                        msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                    )
                    await self._conversation_buffer.async_append(
                        {"role": "assistant", "content": str(content or "")}
                    )
                    break

        # 7.5. Extract answers to inbox questions from agent response
        if _inbox_questions and self._inbox is not None:
            import re as _re

            # Get the agent's response text
            _response_text = ""
            if isinstance(result, dict):
                for msg in result.get("messages", []):
                    content = (
                        msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
                    )
                    role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
                    if role in ("assistant", "ai") and content:
                        _response_text = str(content)

            if _response_text:
                # Parse "ANSWER Q1: ..." patterns
                answer_pattern = _re.compile(
                    r"ANSWER Q(\d+):\s*(.*?)(?=ANSWER Q\d+:|$)", _re.DOTALL
                )
                answers = answer_pattern.findall(_response_text)
                for q_num_str, answer_text in answers:
                    q_idx = int(q_num_str) - 1  # 0-based
                    if 0 <= q_idx < len(_inbox_questions):
                        q_msg = _inbox_questions[q_idx]
                        from .inbox import InboxResponse

                        try:
                            await self._inbox.submit_response(
                                q_msg.message_id,
                                InboxResponse(
                                    question_id=q_msg.message_id,
                                    content=answer_text.strip(),
                                    invocation_id=str(self._invocation_count),
                                ),
                            )
                        except (KeyError, Exception):
                            pass  # Question may have expired

            # Mark processed messages
            for msg in await self._inbox.get_pending():
                if msg.message_type.value != "question":
                    await self._inbox.mark_processed(msg.message_id)

        # 8. Update counters
        self._invocation_count += 1
        self._last_activity = time.monotonic()

        # 9. Increment mission invocation counter
        if self._mission is not None:
            self._mission.increment_invocation()

        # 10. Record success for health error-rate tracking
        if self._health is not None:
            recovered = await self._health.record_success()
            if recovered and self._event_notifier is not None:
                from promptise.events import emit_event

                emit_event(
                    self._event_notifier,
                    "health.recovered",
                    "info",
                    {"process_name": self.name},
                    agent_id=self.name,
                )

        # -- Post-invoke: handle budget violations --
        if (
            self._runtime_callback is not None
            and self._runtime_callback.pending_violations
            and self._budget_enforcer is not None
        ):
            violation = self._runtime_callback.pending_violations[0]
            # Emit budget.exceeded event
            if self._event_notifier is not None:
                from promptise.events import emit_event

                emit_event(
                    self._event_notifier,
                    "budget.exceeded",
                    "critical",
                    {
                        "process_name": self.name,
                        "limit_type": violation.limit_name,
                        "current": violation.current_value,
                        "limit": violation.limit_value,
                    },
                    agent_id=self.name,
                )
            await self._budget_enforcer.handle_violation(violation, self)

        # -- Post-invoke: emit budget warnings (approaching limits) --
        if self._budget is not None and self._event_notifier is not None:
            warnings = getattr(self._budget, "pending_warnings", [])
            for bw in warnings:
                from promptise.events import emit_event

                emit_event(
                    self._event_notifier,
                    "budget.warning",
                    "warning",
                    {
                        "process_name": self.name,
                        "limit_type": bw.limit_name,
                        "current": bw.current_value,
                        "limit": bw.limit_value,
                        "percentage": bw.percentage,
                    },
                    agent_id=self.name,
                )

        # -- Post-invoke: handle health anomalies --
        if self._health is not None and self._health.latest_anomaly is not None:
            latest = self._health.latest_anomaly
            # Only act on anomalies from this invocation (within last 5s)
            if latest.timestamp >= datetime.now(timezone.utc) - timedelta(seconds=5):
                from .escalation import escalate as _escalate

                # Emit health.anomaly event
                if self._event_notifier is not None:
                    from promptise.events import emit_event

                    emit_event(
                        self._event_notifier,
                        "health.anomaly",
                        "warning",
                        {
                            "process_name": self.name,
                            "anomaly_type": latest.anomaly_type.value,
                            "details": latest.details,
                        },
                        agent_id=self.name,
                    )

                action = self.config.health.on_anomaly
                if action == "pause":
                    with contextlib.suppress(Exception):
                        await self.suspend()
                elif action == "stop":
                    with contextlib.suppress(Exception):
                        await self.stop()
                elif action == "escalate" and self.config.health.escalation:
                    with contextlib.suppress(Exception):
                        await _escalate(
                            self.config.health.escalation,
                            {
                                "type": "health_anomaly",
                                "process_id": self.process_id,
                                "anomaly_type": latest.anomaly_type.value,
                                "details": latest.details,
                            },
                            event_bus=self._event_bus,
                        )
                    with contextlib.suppress(Exception):
                        await self.suspend()

        # -- Post-invoke: evaluate mission --
        if self._mission is not None and self._mission.should_evaluate():
            from .mission import MissionEvidence

            evidence = MissionEvidence(
                conversation=await self._conversation_buffer.async_snapshot(),
                state=self._context.state_snapshot(),
                tool_calls=(self._health.tool_history if self._health is not None else []),
                trigger_event={
                    "type": event.trigger_type,
                    "payload": event.payload,
                },
                invocation_count=self._invocation_count,
            )
            evaluation = await self._mission.evaluate(
                evidence,
                self.config.model,
            )
            # Emit mission events
            if self._event_notifier is not None:
                from promptise.events import emit_event

                if evaluation.achieved:
                    emit_event(
                        self._event_notifier,
                        "mission.complete",
                        "info",
                        {
                            "process_name": self.name,
                            "confidence": evaluation.confidence,
                            "invocations": self._invocation_count,
                        },
                        agent_id=self.name,
                    )
                else:
                    emit_event(
                        self._event_notifier,
                        "mission.progress",
                        "info",
                        {
                            "process_name": self.name,
                            "confidence": evaluation.confidence,
                            "achieved": False,
                            "invocations": self._invocation_count,
                        },
                        agent_id=self.name,
                    )

            if evaluation.achieved:
                logger.info(
                    "AgentProcess %s: mission achieved!",
                    self.name,
                )
                if self.config.mission.auto_complete:
                    with contextlib.suppress(Exception):
                        await self.stop()
            elif evaluation.confidence < self.config.mission.confidence_threshold:
                logger.info(
                    "AgentProcess %s: low confidence (%.2f), escalating",
                    self.name,
                    evaluation.confidence,
                )
                if self.config.mission.escalation:
                    from .escalation import escalate as _escalate

                    with contextlib.suppress(Exception):
                        await _escalate(
                            self.config.mission.escalation,
                            {
                                "type": "low_confidence",
                                "process_id": self.process_id,
                                "confidence": evaluation.confidence,
                                "reasoning": evaluation.reasoning,
                            },
                            event_bus=self._event_bus,
                        )

        logger.debug(
            "AgentProcess %s: invocation #%d complete",
            self.name,
            self._invocation_count,
        )
        return result

    async def _heartbeat_loop(self) -> None:
        """Periodic health check and idle timeout monitoring."""
        try:
            while True:
                await asyncio.sleep(self.config.heartbeat_interval)

                if self.state not in (
                    ProcessState.RUNNING,
                    ProcessState.AWAITING,
                ):
                    continue

                # Check idle timeout
                if self.config.idle_timeout > 0 and self._last_activity is not None:
                    idle = time.monotonic() - self._last_activity
                    if idle > self.config.idle_timeout:
                        logger.info(
                            "AgentProcess %s: idle timeout (%.0fs), suspending",
                            self.name,
                            idle,
                        )
                        with contextlib.suppress(Exception):
                            await self._lifecycle.transition(
                                ProcessState.SUSPENDED,
                                reason=f"idle timeout ({idle:.0f}s)",
                            )

                # Check max lifetime
                if self.config.max_lifetime > 0 and self._start_time is not None:
                    lifetime = time.monotonic() - self._start_time
                    if lifetime > self.config.max_lifetime:
                        logger.info(
                            "AgentProcess %s: max lifetime reached (%.0fs), stopping",
                            self.name,
                            lifetime,
                        )
                        # Schedule stop outside the heartbeat loop
                        asyncio.create_task(self.stop())
                        return

                logger.debug(
                    "AgentProcess %s: heartbeat (state=%s, invocations=%d)",
                    self.name,
                    self.state.value,
                    self._invocation_count,
                )
        except asyncio.CancelledError:
            return

    # ------------------------------------------------------------------
    # Open mode: Hot-reload + Rollback
    # ------------------------------------------------------------------

    async def _hot_reload(self, *, reason: str = "") -> str:
        """Rebuild the agent graph with current dynamic state.

        Thread-safe via ``_rebuild_lock``.  Preserves conversation
        history across the rebuild.  Only available in open mode.

        Args:
            reason: Human-readable reason for the rebuild (logged).

        Returns:
            Status message indicating success or failure.

        Raises:
            RuntimeError: If called in strict mode.
        """
        async with self._rebuild_lock:
            if self.config.execution_mode != ExecutionMode.OPEN:
                raise RuntimeError("Hot reload is only available in open mode")

            # Check rebuild limits
            max_rebuilds = self.config.open_mode.max_rebuilds
            if max_rebuilds is not None and self._rebuild_count >= max_rebuilds:
                return f"Error: max rebuilds ({max_rebuilds}) reached"

            # 1. Preserve conversation history (async-safe)
            history = await self._conversation_buffer.async_snapshot()

            # 2. Shutdown old agent (closes MCP connections)
            if self._agent is not None:
                with contextlib.suppress(Exception):
                    await self._agent.shutdown()

            # 3. Rebuild with merged config (static + dynamic)
            await self._build_agent()

            # 4. Restore conversation history (async-safe)
            await self._conversation_buffer.async_replace(history)

            self._rebuild_count += 1
            logger.info(
                "AgentProcess %s: hot reload #%d (%s) — "
                "%d custom tools, %d dynamic servers, %d dynamic triggers",
                self.name,
                self._rebuild_count,
                reason,
                len(self._custom_tools),
                len(self._dynamic_servers),
                len(self._dynamic_triggers),
            )
            return "Rebuild successful"

    async def rollback(self) -> str:
        """Revert to the original configuration.

        Clears all dynamic state (instructions, tools, servers,
        triggers) and rebuilds the agent from the original config.

        Returns:
            Status message.

        Raises:
            RuntimeError: If called in strict mode.
        """
        if self.config.execution_mode != ExecutionMode.OPEN:
            raise RuntimeError("Rollback is only available in open mode")

        # Clear dynamic instructions
        self._dynamic_instructions = None

        # Clear custom tools
        self._custom_tools.clear()

        # Clear dynamic servers
        self._dynamic_servers.clear()

        # Stop and clear dynamic triggers
        for trigger in self._dynamic_triggers:
            with contextlib.suppress(Exception):
                await trigger.stop()
        self._dynamic_triggers.clear()

        return await self._hot_reload(reason="rollback to original configuration")

    def __repr__(self) -> str:
        return (
            f"AgentProcess(name={self.name!r}, state={self.state.value!r}, "
            f"invocations={self._invocation_count})"
        )
