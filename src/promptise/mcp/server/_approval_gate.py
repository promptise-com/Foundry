"""Server-side approval gates — human-in-the-loop enforced where the tool lives.

Any MCP client calling a tool declared with ``requires_approval=True`` is
subject to the gate: the call blocks until a human (or a policy on their
behalf) approves it, and is **denied by default** on timeout.  Unlike
agent-side approval (``promptise.approval.ApprovalPolicy``), which only
governs Promptise agents, the gate lives in the server's middleware chain —
governance does not depend on trusting the caller.

Composes from existing parts: the :class:`~promptise.approval.ApprovalRequest`
/ :class:`~promptise.approval.ApprovalDecision` data model and the
:class:`~promptise.approval.ApprovalHandler` protocol are shared with the
agent-side system, so any existing handler (callback, HMAC-signed webhook,
queue) plugs straight in.

Example::

    from promptise.mcp.server import (
        ApprovalGateMiddleware, MCPServer, PendingApprover,
    )

    server = MCPServer(name="billing")
    approver = PendingApprover(server, approver_role="approver")
    server.add_middleware(AuthMiddleware(auth))          # identity first
    server.add_middleware(ApprovalGateMiddleware(approver, timeout=300))

    @server.tool(auth=True, requires_approval=True)
    async def refund(order_id: str, amount: float) -> dict:
        ...

A human with the ``approver`` role then uses the auto-registered
``approvals_list`` / ``approvals_decide`` tools to release or deny pending
calls.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from collections.abc import Callable
from typing import Any, Literal

from promptise.approval import (
    ApprovalDecision,
    ApprovalRequest,
    CallbackApprovalHandler,
)

from ._context import RequestContext
from ._errors import ApprovalDeniedError

logger = logging.getLogger("promptise.server.approval")


def _annotation_hints(tool_def: Any) -> dict[str, Any]:
    """A tool's annotations in MCP's wire names, for ``ApprovalRequest.tool_annotations``."""
    annotations = getattr(tool_def, "annotations", None)
    if annotations is None:
        return {}
    hints = {
        "title": annotations.title,
        "readOnlyHint": annotations.read_only_hint,
        "destructiveHint": annotations.destructive_hint,
        "idempotentHint": annotations.idempotent_hint,
        "openWorldHint": annotations.open_world_hint,
    }
    return {key: value for key, value in hints.items() if value is not None}


class ApprovalGateMiddleware:
    """Enforces human approval on tools declared with ``requires_approval=True``.

    Runs in the middleware chain (install it **after** ``AuthMiddleware`` so
    the approval request carries the verified caller identity).  For gated
    tools it builds an :class:`ApprovalRequest`, awaits the configured
    handler's decision, and either lets the call proceed or raises
    :class:`ApprovalDeniedError`.  Non-gated tools pass through untouched.

    Fail-closed semantics throughout: a timeout denies by default
    (``on_timeout="deny"``), a handler crash denies, and a decision carrying
    ``modified_arguments`` denies — the server-side gate cannot rewrite the
    call's arguments (they are already bound), and silently ignoring a
    reviewer's modification would execute something the reviewer did not
    approve.

    Args:
        handler: An :class:`~promptise.approval.ApprovalHandler` (anything
            with ``async request_approval(request) -> ApprovalDecision``),
            or a bare callable (sync/async, returning a decision or bool) —
            wrapped in a ``CallbackApprovalHandler`` automatically.
            Handlers may additionally implement
            ``request_approval_ctx(request, ctx)`` to receive the server
            :class:`RequestContext` (used by :class:`ElicitationApprover`).
        timeout: Seconds to wait for a decision before applying
            ``on_timeout`` (default 300).
        on_timeout: ``"deny"`` (default) or ``"allow"``.
        include_arguments: Include the validated tool arguments in the
            approval request (default ``True``).  Disable for tools whose
            arguments are too sensitive to surface to reviewers.
    """

    def __init__(
        self,
        handler: Any,
        *,
        timeout: float = 300.0,
        on_timeout: Literal["deny", "allow"] = "deny",
        include_arguments: bool = True,
    ) -> None:
        if on_timeout not in ("deny", "allow"):
            raise ValueError(f"on_timeout must be 'deny' or 'allow', got {on_timeout!r}")
        if not (0 < timeout <= 86400):
            raise ValueError(f"timeout must be in (0, 86400], got {timeout}")
        if callable(handler) and not hasattr(handler, "request_approval"):
            handler = CallbackApprovalHandler(handler)
        self._handler = handler
        self._timeout = timeout
        self._on_timeout = on_timeout
        self._include_arguments = include_arguments

    @staticmethod
    async def _check_guards(guards: list[Any], ctx: RequestContext) -> None:
        """Deny (raise ``AuthenticationError``) if any guard rejects the caller.

        Mirrors the pipeline's own guard evaluation so the gate's early
        check is identical to the authorization that would run later — a
        caller the tool's guards reject never reaches the approval step.
        """
        from ._errors import AuthenticationError

        for guard in guards:
            if not await guard.check(ctx):
                guard_name = type(guard).__name__
                detail = (
                    guard.describe_denial(ctx)
                    if hasattr(guard, "describe_denial")
                    else f"Access denied by {guard_name}"
                )
                raise AuthenticationError(
                    detail,
                    code="ACCESS_DENIED",
                    details={"guard": guard_name, "tool": ctx.tool_name},
                )

    def _build_request(self, ctx: RequestContext) -> ApprovalRequest:
        client = getattr(ctx, "client", None)
        raw_arguments = dict(ctx.state.get("_tool_arguments") or {})
        arguments: dict[str, Any] = dict(raw_arguments) if self._include_arguments else {}
        return ApprovalRequest(
            request_id=secrets.token_hex(16),
            tool_name=ctx.tool_name,
            arguments=arguments,
            agent_id=None,
            caller_user_id=ctx.client_id,
            context_summary=f"server-side approval gate on {ctx.server_name!r}",
            timeout=self._timeout,
            metadata={
                "server": ctx.server_name,
                "mcp_request_id": ctx.request_id,
                "client_id": ctx.client_id,
                "tenant_id": getattr(client, "tenant_id", None),
                "subject": getattr(client, "subject", None),
                "issuer": getattr(client, "issuer", None),
            },
            tool_annotations=_annotation_hints(ctx.state.get("tool_def")),
            # In-process only (never serialized): lets an
            # ``AutoApprovalClassifier`` handler match its rules even with
            # ``include_arguments=False``.
            raw_arguments=raw_arguments,
        )

    async def __call__(self, ctx: RequestContext, call_next: Callable[..., Any]) -> Any:
        tool_def = ctx.state.get("tool_def")
        if not getattr(tool_def, "requires_approval", False):
            return await call_next(ctx)

        # Evaluate the tool's own guards BEFORE requesting approval. Guards
        # normally run innermost (after the whole middleware chain), so
        # without this an unauthorized or unauthenticated caller would
        # trigger a real human approval — spamming reviewers with
        # attacker-chosen arguments and filling a bounded pending queue
        # (a fail-closed DoS on legitimate approvals). Deny first, then ask.
        guards = getattr(tool_def, "guards", None)
        if guards:
            await self._check_guards(guards, ctx)

        request = self._build_request(ctx)
        logger.info(
            "Approval requested: tool=%s client=%s request_id=%s",
            ctx.tool_name,
            ctx.client_id,
            request.request_id,
        )

        try:
            if hasattr(self._handler, "request_approval_ctx"):
                decision = await asyncio.wait_for(
                    self._handler.request_approval_ctx(request, ctx),
                    timeout=self._timeout,
                )
            else:
                decision = await asyncio.wait_for(
                    self._handler.request_approval(request),
                    timeout=self._timeout,
                )
        except (asyncio.TimeoutError, TimeoutError):
            if self._on_timeout == "allow":
                logger.warning(
                    "Approval timed out after %.0fs for tool=%s — proceeding (on_timeout='allow')",
                    self._timeout,
                    ctx.tool_name,
                )
                return await call_next(ctx)
            raise ApprovalDeniedError(
                f"Approval for tool {ctx.tool_name!r} timed out after "
                f"{self._timeout:.0f}s and was denied by default",
                request_id=request.request_id,
            ) from None

        if decision.modified_arguments is not None:
            # Arguments are already bound in the handler closure — the gate
            # cannot apply a reviewer's modification, and executing the
            # ORIGINAL arguments after the reviewer changed them would run
            # something the reviewer did not approve. Fail closed.
            raise ApprovalDeniedError(
                f"Approval for tool {ctx.tool_name!r} returned modified "
                "arguments, which the server-side gate does not support — "
                "denied (approve or deny only)",
                request_id=request.request_id,
                reviewer_id=decision.reviewer_id,
            )

        if not decision.approved:
            reason = f": {decision.reason}" if decision.reason else ""
            logger.info(
                "Approval DENIED: tool=%s client=%s reviewer=%s request_id=%s",
                ctx.tool_name,
                ctx.client_id,
                decision.reviewer_id,
                request.request_id,
            )
            raise ApprovalDeniedError(
                f"Approval denied for tool {ctx.tool_name!r}{reason}",
                request_id=request.request_id,
                reviewer_id=decision.reviewer_id,
            )

        logger.info(
            "Approval GRANTED: tool=%s client=%s reviewer=%s request_id=%s",
            ctx.tool_name,
            ctx.client_id,
            decision.reviewer_id,
            request.request_id,
        )
        return await call_next(ctx)


class ElicitationApprover:
    """Approves by asking the human behind the *calling* MCP client.

    Uses MCP elicitation: the connected client is asked to confirm the call
    (``{"approve": bool, "reason": str}``).  This is confirm-your-own-action
    dual control — right for destructive-but-personal operations ("really
    delete this?").  For independent four-eyes review, use
    :class:`PendingApprover` instead.

    Fail-closed: if the transport has no live MCP session (``TestClient``,
    stdio clients without elicitation support), the request is **denied**
    with a clear reason — never silently allowed.
    """

    def __init__(self, *, message_template: str | None = None) -> None:
        self._template = message_template or (
            "Approval required: call tool '{tool_name}' with arguments {arguments}. Approve?"
        )

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        # Protocol-compat path (no server context available): fail closed.
        return ApprovalDecision(
            approved=False,
            reviewer_id="elicitation",
            reason=(
                "elicitation approver requires the server request context; "
                "install it via ApprovalGateMiddleware"
            ),
        )

    async def request_approval_ctx(
        self, request: ApprovalRequest, ctx: RequestContext
    ) -> ApprovalDecision:
        session = ctx.state.get("_mcp_session")
        if session is None:
            return ApprovalDecision(
                approved=False,
                reviewer_id="elicitation",
                reason=(
                    "no live MCP session for elicitation (client or transport "
                    "does not support it) — denied fail-closed"
                ),
            )

        from ._elicitation import Elicitor

        elicitor = Elicitor(timeout=request.timeout)
        elicitor._bind(session, None)
        answer = await elicitor.ask(
            self._template.format(tool_name=request.tool_name, arguments=request.arguments),
            schema={
                "type": "object",
                "properties": {
                    "approve": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["approve"],
            },
        )
        if not isinstance(answer, dict) or not isinstance(answer.get("approve"), bool):
            return ApprovalDecision(
                approved=False,
                reviewer_id="elicitation",
                reason="client declined or returned an invalid elicitation response",
            )
        return ApprovalDecision(
            approved=answer["approve"],
            reviewer_id="elicitation:client-user",
            reason=answer.get("reason"),
        )


class PendingApprover:
    """Blocks gated calls in a pending store until a human decides.

    Independent four-eyes review: the gated call waits while a *different*
    human — anyone holding ``approver_role`` — lists pending requests and
    approves or denies them through two auto-registered, role-guarded admin
    tools:

    - ``approvals_list()`` — pending requests with tool, arguments, caller
      identity, and age.
    - ``approvals_decide(request_id, approve, reason)`` — release or deny;
      the reviewer's ``client_id`` is recorded on the decision.

    The store is process-local (like the in-memory job queue); calls that
    time out before a decision are denied by the gate's default.

    Args:
        server: Server to auto-register the admin tools on.  Pass ``None``
            and call :meth:`register_tools` later to register on a different
            server than the one being gated.
        approver_role: Role required to list/decide (default ``"approver"``).
        max_pending: Deny immediately beyond this many waiting requests
            (default 100) — a full queue must not become an unbounded pile
            of blocked calls.
        max_pending_per_client: Deny immediately once a single caller has
            this many requests waiting (default ``None`` — no per-client
            cap, only the global one).  Without it the store is one shared
            queue: a single client or tenant that fills ``max_pending``
            makes every other caller's gated call fail closed.  A caller is
            identified by the request's ``caller_user_id`` (the verified
            ``client_id`` on the server-side gate), falling back to the
            ``client_id`` / ``tenant_id`` metadata and then ``agent_id``;
            requests carrying no identity at all share one anonymous bucket
            — an unauthenticated flood cannot escape the cap by being
            anonymous.  Must not exceed ``max_pending`` (nor
            ``max_pending_per_tenant`` when that is set).
        max_pending_per_tenant: Deny immediately once one tenant has this
            many requests waiting, across all of its clients (default
            ``None`` — no per-tenant cap).  The per-client cap counts by
            client id, so a tenant holding several API keys could still take
            the whole store; this cap counts by the ``tenant_id`` metadata
            the server-side gate records (requests without one share one
            bucket).  Must not exceed ``max_pending``.  The caps nest:
            client ≤ tenant ≤ store.
    """

    def __init__(
        self,
        server: Any = None,
        *,
        approver_role: str = "approver",
        max_pending: int = 100,
        max_pending_per_client: int | None = None,
        max_pending_per_tenant: int | None = None,
    ) -> None:
        if max_pending <= 0:
            raise ValueError(f"max_pending must be positive, got {max_pending}")
        if max_pending_per_tenant is not None and not (0 < max_pending_per_tenant <= max_pending):
            raise ValueError(
                "max_pending_per_tenant must be between 1 and max_pending "
                f"({max_pending}), got {max_pending_per_tenant}"
            )
        ceiling_name, ceiling = "max_pending", max_pending
        if max_pending_per_tenant is not None:
            ceiling_name, ceiling = "max_pending_per_tenant", max_pending_per_tenant
        if max_pending_per_client is not None and not (0 < max_pending_per_client <= ceiling):
            raise ValueError(
                f"max_pending_per_client must be between 1 and {ceiling_name} "
                f"({ceiling}), got {max_pending_per_client}"
            )
        self._approver_role = approver_role
        self._max_pending = max_pending
        self._max_pending_per_client = max_pending_per_client
        self._max_pending_per_tenant = max_pending_per_tenant
        self._pending: dict[str, tuple[ApprovalRequest, asyncio.Future[ApprovalDecision]]] = {}
        self._lock = asyncio.Lock()
        if server is not None:
            self.register_tools(server)

    @staticmethod
    def client_key(request: ApprovalRequest) -> str:
        """The identity a request is counted under for ``max_pending_per_client``.

        Prefers the verified caller (``caller_user_id``), then the
        ``client_id`` and ``tenant_id`` metadata the server-side gate
        records, then the agent that raised the request.  Returns ``""``
        for a request carrying no identity — all such requests share one
        bucket.
        """
        metadata = request.metadata or {}
        for candidate in (
            request.caller_user_id,
            metadata.get("client_id"),
            metadata.get("tenant_id"),
            request.agent_id,
        ):
            if candidate:
                return str(candidate)
        return ""

    def pending_for(self, client_key: str) -> int:
        """Number of waiting requests counted under *client_key*."""
        return sum(1 for req, _ in self._pending.values() if self.client_key(req) == client_key)

    @staticmethod
    def tenant_key(request: ApprovalRequest) -> str:
        """The tenant a request is counted under for ``max_pending_per_tenant``.

        The ``tenant_id`` the server-side gate records in the request
        metadata; ``""`` for a request without one — all such requests
        share one bucket.
        """
        tenant = (request.metadata or {}).get("tenant_id")
        return str(tenant) if tenant else ""

    def pending_for_tenant(self, tenant_key: str) -> int:
        """Number of waiting requests counted under *tenant_key*."""
        return sum(1 for req, _ in self._pending.values() if self.tenant_key(req) == tenant_key)

    # -- ApprovalHandler protocol ---------------------------------------

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        async with self._lock:
            if len(self._pending) >= self._max_pending:
                return ApprovalDecision(
                    approved=False,
                    reviewer_id="pending-approver",
                    reason=f"pending approval queue is full ({self._max_pending})",
                )
            if self._max_pending_per_tenant is not None:
                tenant = self.tenant_key(request)
                if self.pending_for_tenant(tenant) >= self._max_pending_per_tenant:
                    whose = f"tenant {tenant!r}" if tenant else "callers without a tenant"
                    return ApprovalDecision(
                        approved=False,
                        reviewer_id="pending-approver",
                        reason=(
                            f"per-tenant pending approval cap reached for {whose} "
                            f"({self._max_pending_per_tenant} per tenant)"
                        ),
                    )
            if self._max_pending_per_client is not None:
                key = self.client_key(request)
                if self.pending_for(key) >= self._max_pending_per_client:
                    who = repr(key) if key else "anonymous callers"
                    return ApprovalDecision(
                        approved=False,
                        reviewer_id="pending-approver",
                        reason=(
                            f"per-client pending approval cap reached for {who} "
                            f"({self._max_pending_per_client} per client)"
                        ),
                    )
            future: asyncio.Future[ApprovalDecision] = asyncio.get_running_loop().create_future()
            self._pending[request.request_id] = (request, future)
        try:
            # The gate applies the timeout via wait_for; cancellation on
            # timeout lands here and the finally clause cleans up.
            return await future
        finally:
            self._pending.pop(request.request_id, None)

    # -- Reviewer API -----------------------------------------------------

    def pending(self) -> list[dict[str, Any]]:
        """Summaries of requests currently awaiting a decision."""
        now = time.time()
        return [
            {
                "request_id": req.request_id,
                "tool": req.tool_name,
                "arguments": req.arguments,
                "client_id": req.caller_user_id,
                "tenant_id": (req.metadata or {}).get("tenant_id"),
                "age_seconds": round(now - req.timestamp, 1),
            }
            for req, _ in self._pending.values()
        ]

    def caller_of(self, request_id: str) -> str | None:
        """The ``client_id`` that triggered a pending request, or ``None``."""
        entry = self._pending.get(request_id)
        return entry[0].caller_user_id if entry is not None else None

    def decide(
        self,
        request_id: str,
        approved: bool,
        *,
        reviewer_id: str | None = None,
        reason: str | None = None,
    ) -> bool:
        """Resolve a pending request. Returns ``False`` if it is unknown
        (already decided, timed out, or never existed)."""
        entry = self._pending.get(request_id)
        if entry is None:
            return False
        _, future = entry
        if future.done():
            return False
        future.set_result(
            ApprovalDecision(approved=approved, reviewer_id=reviewer_id, reason=reason)
        )
        return True

    # -- Admin tool registration -------------------------------------------

    def register_tools(self, server: Any) -> None:
        """Register the role-guarded ``approvals_list`` / ``approvals_decide``
        admin tools on *server*."""
        from ._guards import HasRole

        approver = self

        @server.tool(
            name="approvals_list",
            description="List tool calls awaiting human approval.",
            auth=True,
            guards=[HasRole(self._approver_role)],
            tags=["approvals"],
        )
        async def approvals_list() -> list[dict]:
            """List pending approval requests."""
            return approver.pending()

        @server.tool(
            name="approvals_decide",
            description="Approve or deny a pending tool call by request id.",
            auth=True,
            guards=[HasRole(self._approver_role)],
            tags=["approvals"],
        )
        async def approvals_decide(
            request_id: str,
            approve: bool,
            reason: str = "",
        ) -> dict:
            """Decide a pending approval request."""
            # Read identity from the active request context (populated on every
            # transport via set_context) — NOT from an injected parameter, which
            # would leave the reviewer unknown on stdio/http/sse and silently
            # defeat separation-of-duties + audit accountability.
            from ._context import get_context

            try:
                reviewer = get_context().client_id or "unknown-reviewer"
            except RuntimeError:
                reviewer = "unknown-reviewer"
            # Separation of duties: a caller may not approve their own request
            # (four-eyes). Denying your own is allowed — anyone can cancel.
            if approve and reviewer == approver.caller_of(request_id):
                return {
                    "resolved": False,
                    "error": "cannot approve your own request — four-eyes "
                    "separation of duties requires a different reviewer",
                }
            resolved = approver.decide(
                request_id, approve, reviewer_id=reviewer, reason=reason or None
            )
            if not resolved:
                return {
                    "resolved": False,
                    "error": f"no pending approval with request_id {request_id!r} "
                    "(already decided, timed out, or unknown)",
                }
            return {"resolved": True, "request_id": request_id, "approved": approve}
