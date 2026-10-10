"""Human-in-the-Loop approval for agent tool calls.

Intercepts tool calls that match configurable patterns, sends approval
requests to a human reviewer (via webhook, callback, or async queue),
and either proceeds or denies based on the decision.

Example::

    from promptise import build_agent, ApprovalPolicy, CallbackApprovalHandler

    async def my_handler(request):
        print(f"Approve {request.tool_name}({request.arguments})? [y/n]")
        # ... collect decision ...
        return ApprovalDecision(approved=True, timestamp=time.time())

    agent = await build_agent(
        ...,
        approval=ApprovalPolicy(
            tools=["send_email", "delete_*"],
            handler=CallbackApprovalHandler(my_handler),
            timeout=300,
        ),
    )
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac as _hmac_mod
import json
import logging
import secrets
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Literal, Protocol, runtime_checkable

from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

logger = logging.getLogger("promptise.approval")

__all__ = [
    "ApprovalRequest",
    "ApprovalDecision",
    "ApprovalHandler",
    "ApprovalPolicy",
    "CallbackApprovalHandler",
    "WebhookApprovalHandler",
    "QueueApprovalHandler",
    "wrap_tools_with_approval",
]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ApprovalRequest:
    """A request for human approval of a tool call.

    Attributes:
        request_id: Unique cryptographic ID for this request.
        tool_name: Name of the tool requiring approval.
        arguments: Tool arguments (redacted if configured).
        agent_id: Agent or process identifier.
        caller_user_id: User who triggered the agent (from CallerContext).
        context_summary: Last few messages for reviewer context.
        timestamp: When the request was created (``time.time()``).
        timeout: Seconds until auto-deny/allow.
        metadata: Developer-provided custom data.
    """

    request_id: str
    tool_name: str
    arguments: dict[str, Any]
    agent_id: str | None = None
    caller_user_id: str | None = None
    context_summary: str = ""
    timestamp: float = field(default_factory=time.time)
    timeout: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (for webhook payloads)."""
        return {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "agent_id": self.agent_id,
            "caller_user_id": self.caller_user_id,
            "context_summary": self.context_summary,
            "timestamp": self.timestamp,
            "timeout": self.timeout,
            "metadata": self.metadata,
        }

    def compute_hmac(self, secret: str) -> str:
        """Compute HMAC-SHA256 signature covering all request fields."""
        payload = json.dumps(
            {
                "request_id": self.request_id,
                "tool_name": self.tool_name,
                "arguments": self.arguments,
                "agent_id": self.agent_id,
                "caller_user_id": self.caller_user_id,
                "timestamp": self.timestamp,
            },
            sort_keys=True,
            default=str,
        )
        return _hmac_mod.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


@dataclass
class ApprovalDecision:
    """A human's decision on an approval request.

    Attributes:
        approved: Whether the tool call is approved.
        modified_arguments: If the reviewer edited arguments.
        reviewer_id: Who made the decision.
        reason: Optional explanation.
        timestamp: When the decision was made.
    """

    approved: bool
    modified_arguments: dict[str, Any] | None = None
    reviewer_id: str | None = None
    reason: str | None = None
    timestamp: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Handler protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ApprovalHandler(Protocol):
    """Protocol for approval handlers.

    Implementations receive an :class:`ApprovalRequest` and must return
    an :class:`ApprovalDecision`.  The handler is async — it can await
    webhooks, poll APIs, or wait on queues.
    """

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision: ...


# ---------------------------------------------------------------------------
# Built-in handlers
# ---------------------------------------------------------------------------


class CallbackApprovalHandler:
    """Approval handler that delegates to an async Python callable.

    The simplest handler — pass any ``async def handler(request) -> decision``
    function and it will be called for each approval request.

    Args:
        callback: Async callable that receives an :class:`ApprovalRequest`
            and returns an :class:`ApprovalDecision`.
    """

    def __init__(self, callback: Callable[..., Any]) -> None:
        self._callback = callback

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Delegate to the user-provided callback."""
        result = self._callback(request)
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            result = await result
        if isinstance(result, ApprovalDecision):
            return result
        # Allow returning a plain bool for convenience
        if isinstance(result, bool):
            return ApprovalDecision(approved=result)
        raise TypeError(
            f"Approval callback must return ApprovalDecision or bool, got {type(result).__name__}"
        )


class WebhookApprovalHandler:
    """Approval handler that POSTs to a webhook URL and polls for a decision.

    Sends the :class:`ApprovalRequest` as a JSON POST to ``url``.  Then
    polls ``poll_url`` (or ``url + "/" + request_id``) for the decision.

    Args:
        url: Webhook URL to POST the approval request to.
        secret: HMAC secret for signing requests.  If not provided,
            a random secret is generated (ephemeral — not useful for
            cross-process verification).
        poll_url: URL to poll for the decision.  Defaults to
            ``{url}/{request_id}``.
        poll_interval: Seconds between poll attempts.
        headers: Custom HTTP headers (e.g., auth tokens).
    """

    def __init__(
        self,
        url: str,
        *,
        secret: str | None = None,
        poll_url: str | None = None,
        poll_interval: float = 2.0,
        headers: dict[str, str] | None = None,
        http_client: Any | None = None,
    ) -> None:
        # SSRF protection — reject private/internal URLs
        try:
            from promptise.mcp.server._openapi import _validate_url_not_private

            _validate_url_not_private(url)
        except (ImportError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            # MCP server module not available — skip validation

        self._url = url
        self._secret = secret or secrets.token_hex(32)
        self._poll_url = poll_url
        self._poll_interval = max(0.5, poll_interval)
        self._headers = headers or {}
        self._http_client = http_client  # Optional pre-configured httpx.AsyncClient

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """POST request to webhook, poll for decision."""
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for WebhookApprovalHandler. Install with: pip install httpx"
            )

        signature = request.compute_hmac(self._secret)
        headers = {
            "Content-Type": "application/json",
            "X-Promptise-Signature": signature,
            "X-Promptise-Request-Id": request.request_id,
            **self._headers,
        }

        # Use developer-provided client (for proxy, mTLS, custom auth) or create one.
        # Pre-built clients must NOT be entered as context managers — that would
        # close them after the first request.
        if self._http_client is not None:
            client = self._http_client
            should_close = False
        else:
            client = httpx.AsyncClient(timeout=30)
            should_close = True

        try:
            # POST the approval request
            resp = await client.post(
                self._url,
                json=request.to_dict(),
                headers=headers,
            )
            resp.raise_for_status()

            # Poll for decision
            poll_target = self._poll_url or f"{self._url}/{request.request_id}"
            deadline = time.monotonic() + request.timeout

            while time.monotonic() < deadline:
                await asyncio.sleep(self._poll_interval)
                try:
                    poll_resp = await client.get(
                        poll_target,
                        headers={
                            "X-Promptise-Request-Id": request.request_id,
                            **self._headers,
                        },
                    )
                    if poll_resp.status_code == 200:
                        data = poll_resp.json()
                        if "approved" in data:
                            return ApprovalDecision(
                                approved=data["approved"],
                                modified_arguments=data.get("modified_arguments"),
                                reviewer_id=data.get("reviewer_id"),
                                reason=data.get("reason"),
                            )
                    # 202 = still pending, continue polling
                except httpx.HTTPError:
                    logger.warning(
                        "Approval poll failed for %s, retrying",
                        request.request_id,
                    )

        finally:
            if should_close:
                await client.aclose()

        # Timeout — no decision received
        raise asyncio.TimeoutError(f"No approval decision within {request.timeout}s")


class QueueApprovalHandler:
    """Approval handler using async queues for in-process UIs.

    For Gradio, Streamlit, or other in-process UIs where the human
    reviewer is in the same Python process.  The UI reads from
    :attr:`request_queue` and writes decisions to the handler
    via :meth:`submit_decision`.

    Example::

        handler = QueueApprovalHandler()

        # UI thread reads approval requests
        request = await handler.request_queue.get()
        # Show to user, collect decision
        handler.submit_decision(request.request_id, ApprovalDecision(approved=True))

    Attributes:
        request_queue: Queue of pending :class:`ApprovalRequest` objects.
    """

    def __init__(self, maxsize: int = 100) -> None:
        self.request_queue: asyncio.Queue[ApprovalRequest] = asyncio.Queue(maxsize=maxsize)
        self._pending: dict[str, asyncio.Future[ApprovalDecision]] = {}
        self._lock = asyncio.Lock()

    def submit_decision(self, request_id: str, decision: ApprovalDecision) -> None:
        """Submit a decision for a pending request.

        Called by the UI after the human reviewer makes a choice.

        Args:
            request_id: The ``request_id`` from the :class:`ApprovalRequest`.
            decision: The reviewer's decision.

        Raises:
            KeyError: If no pending request with this ID exists.
        """
        future = self._pending.get(request_id)
        if future is None:
            raise KeyError(f"No pending approval request with id {request_id!r}")
        if not future.done():
            future.set_result(decision)

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Enqueue request and wait for decision from the UI."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ApprovalDecision] = loop.create_future()

        async with self._lock:
            self._pending[request.request_id] = future

        try:
            await self.request_queue.put(request)
            # Wait for the UI to call submit_decision()
            return await asyncio.wait_for(future, timeout=request.timeout)
        except asyncio.TimeoutError:
            raise
        finally:
            async with self._lock:
                self._pending.pop(request.request_id, None)


# ---------------------------------------------------------------------------
# Approval policy
# ---------------------------------------------------------------------------


class ApprovalPolicy:
    """Configuration for human-in-the-loop approval.

    Defines which tools require approval, how to request it, and
    what happens on timeout or repeated denial.

    Args:
        tools: Glob patterns for tool names that require approval.
            Examples: ``["send_email"]``, ``["delete_*", "payment_*"]``.
        handler: An :class:`ApprovalHandler` implementation or an async
            callable ``(ApprovalRequest) -> ApprovalDecision``.
        timeout: Seconds to wait for a decision before applying
            ``on_timeout``.  Default: 300 (5 minutes).
        on_timeout: What to do when timeout expires.
            ``"deny"`` (default) rejects the tool call.
            ``"allow"`` permits it.
        include_arguments: Include tool arguments in the approval
            request.  Set to ``False`` to hide arguments from reviewers.
        redact_sensitive: Run arguments through PII/credential detection
            before sending to the reviewer.  Requires guardrails.
        max_pending: Maximum concurrent pending approvals per agent.
            Additional tool calls are auto-denied.
        max_retries_after_deny: If the agent retries a denied tool
            this many times, return a permanent denial message.
    """

    def __init__(
        self,
        *,
        tools: list[str],
        handler: ApprovalHandler | Callable[..., Any],
        timeout: float = 300.0,
        on_timeout: Literal["deny", "allow"] = "deny",
        include_arguments: bool = True,
        redact_sensitive: bool = True,
        max_pending: int = 10,
        max_retries_after_deny: int = 3,
    ) -> None:
        if not tools:
            raise ValueError("ApprovalPolicy requires at least one tool pattern")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if timeout > 86400:
            raise ValueError("timeout cannot exceed 86400 seconds (24 hours)")
        if max_pending < 1:
            raise ValueError("max_pending must be at least 1")

        self.tools = tools
        self.timeout = timeout
        self.on_timeout = on_timeout
        self.include_arguments = include_arguments
        self.redact_sensitive = redact_sensitive
        self.max_pending = max_pending
        self.max_retries_after_deny = max_retries_after_deny

        # Normalize handler — wrap callable in CallbackApprovalHandler
        if isinstance(handler, ApprovalHandler):
            self.handler = handler
        elif callable(handler):
            self.handler = CallbackApprovalHandler(handler)
        else:
            raise TypeError(
                f"handler must be an ApprovalHandler or callable, got {type(handler).__name__}"
            )

    def requires_approval(self, tool_name: str) -> bool:
        """Check if a tool name matches any approval pattern.

        Uses ``fnmatch`` glob matching — supports ``*`` and ``?``
        wildcards.
        """
        return any(fnmatch(tool_name, pattern) for pattern in self.tools)

    async def redact_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive data from arguments before sending to reviewer.

        Uses the guardrails scanner's ``check_output`` if available.
        Falls back to returning arguments as-is if guardrails are not
        installed.
        """
        if not self.redact_sensitive:
            return dict(arguments)

        try:
            from .guardrails import PromptiseSecurityScanner

            text = json.dumps(arguments, default=str)
            scanner = PromptiseSecurityScanner(
                detect_injection=False,
                detect_toxicity=False,
            )
            redacted = await scanner.check_output(text)
            if isinstance(redacted, str) and redacted != text:
                try:
                    return json.loads(redacted)
                except json.JSONDecodeError:
                    return {"_redacted": redacted}
            return dict(arguments)
        except (ImportError, Exception):
            return dict(arguments)


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------


class _ApprovalToolWrapper(BaseTool):
    """Wraps a tool with an approval gate.

    Transparent to the LLM — same name, description, and schema as the
    inner tool.  When ``_arun()`` is called, sends an approval request
    and waits for a human decision before executing.
    """

    _inner: BaseTool = PrivateAttr()
    _policy: ApprovalPolicy = PrivateAttr()
    _pending_count: list[int] = PrivateAttr()  # Shared mutable counter [count]
    _deny_counts: dict[str, int] = PrivateAttr()  # Shared deny tracker
    _used_request_ids: set[str] = PrivateAttr()  # Replay protection
    _event_notifier: Any = PrivateAttr(default=None)  # EventNotifier for events

    def __init__(
        self,
        inner: BaseTool,
        policy: ApprovalPolicy,
        pending_count: list[int],
        deny_counts: dict[str, int],
        used_request_ids: set[str],
        event_notifier: Any = None,
    ) -> None:
        # Copy name, description, args_schema from inner tool
        super().__init__(
            name=inner.name,
            description=inner.description,
            args_schema=getattr(inner, "args_schema", None),
        )
        self._inner = inner
        self._policy = policy
        self._pending_count = pending_count
        self._deny_counts = deny_counts
        self._event_notifier = event_notifier
        self._used_request_ids = used_request_ids

    async def _arun(self, **kwargs: Any) -> Any:
        """Intercept tool call, request approval, then execute or deny."""
        tool_name = self._inner.name

        # Check max pending
        if self._pending_count[0] >= self._policy.max_pending:
            logger.warning(
                "Approval: max_pending=%d reached, auto-denying %s",
                self._policy.max_pending,
                tool_name,
            )
            return (
                f"DENIED: Too many pending approval requests "
                f"(max {self._policy.max_pending}). Try again later."
            )

        # Check retry limit
        deny_key = tool_name
        if self._deny_counts.get(deny_key, 0) >= self._policy.max_retries_after_deny:
            return (
                "DENIED: This action was permanently denied after "
                f"{self._policy.max_retries_after_deny} attempts. "
                "Do not retry this tool."
            )

        # Build approval request
        request_id = secrets.token_hex(16)
        arguments = (
            await self._policy.redact_arguments(kwargs) if self._policy.include_arguments else {}
        )

        # Get caller context
        caller_user_id: str | None = None
        try:
            from .agent import get_current_caller

            caller = get_current_caller()
            if caller is not None:
                caller_user_id = getattr(caller, "user_id", None)
        except ImportError:
            pass

        request = ApprovalRequest(
            request_id=request_id,
            tool_name=tool_name,
            arguments=arguments,
            caller_user_id=caller_user_id,
            timeout=self._policy.timeout,
        )

        logger.info(
            "Approval: requesting approval for %s (request_id=%s)",
            tool_name,
            request_id,
        )
        if self._event_notifier is not None:
            from .events import emit_event

            emit_event(
                self._event_notifier,
                "approval.requested",
                "info",
                {
                    "tool_name": tool_name,
                    "request_id": request_id,
                    "timeout": self._policy.timeout,
                    # As redacted by the policy ({} when include_arguments=False)
                    "arguments": arguments,
                },
            )

        # Send approval request
        self._pending_count[0] += 1
        try:
            decision = await asyncio.wait_for(
                self._policy.handler.request_approval(request),
                timeout=self._policy.timeout,
            )
        except asyncio.TimeoutError:
            decision = ApprovalDecision(
                approved=(self._policy.on_timeout == "allow"),
                reason=f"Approval timed out after {self._policy.timeout}s",
            )
            logger.warning(
                "Approval: timeout for %s (request_id=%s), on_timeout=%s",
                tool_name,
                request_id,
                self._policy.on_timeout,
            )
        except Exception as exc:
            logger.error(
                "Approval: handler error for %s: %s",
                tool_name,
                exc,
            )
            decision = ApprovalDecision(
                approved=False,
                reason=f"Approval handler error: {type(exc).__name__}",
            )
        finally:
            self._pending_count[0] = max(0, self._pending_count[0] - 1)

        # Replay protection — mark request_id as used
        self._used_request_ids.add(request_id)

        if not decision.approved:
            self._deny_counts[deny_key] = self._deny_counts.get(deny_key, 0) + 1
            reason = decision.reason or "Action denied by reviewer."
            logger.info(
                "Approval: DENIED %s (request_id=%s): %s",
                tool_name,
                request_id,
                reason,
            )
            if self._event_notifier is not None:
                from .events import emit_event

                emit_event(
                    self._event_notifier,
                    "approval.denied",
                    "warning",
                    {"tool_name": tool_name, "request_id": request_id, "reason": reason},
                )
            return f"DENIED: {reason}"

        # Approved — execute with original or modified arguments
        final_args = (
            decision.modified_arguments if decision.modified_arguments is not None else kwargs
        )
        logger.info(
            "Approval: APPROVED %s (request_id=%s, reviewer=%s)",
            tool_name,
            request_id,
            decision.reviewer_id or "unknown",
        )
        if self._event_notifier is not None:
            from .events import emit_event

            emit_event(
                self._event_notifier,
                "approval.granted",
                "info",
                {
                    "tool_name": tool_name,
                    "request_id": request_id,
                    "reviewer": decision.reviewer_id,
                },
            )
        return await self._inner._arun(**final_args)

    def _run(self, **kwargs: Any) -> Any:  # pragma: no cover
        import anyio

        return anyio.run(lambda: self._arun(**kwargs))


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def wrap_tools_with_approval(
    tools: list[BaseTool],
    policy: ApprovalPolicy,
    *,
    event_notifier: Any = None,
) -> list[BaseTool]:
    """Wrap tools that match the approval policy's patterns.

    Tools that don't match any pattern are returned as-is (zero overhead).
    Tools that match are wrapped in :class:`_ApprovalToolWrapper`.

    Args:
        tools: List of LangChain tools (from MCP, extra_tools, etc.).
        policy: The approval policy defining which tools need approval.

    Returns:
        New list with matching tools wrapped. Order preserved.
    """
    # Shared state across all wrappers for this agent
    pending_count: list[int] = [0]  # Mutable counter
    deny_counts: dict[str, int] = {}
    used_request_ids: set[str] = set()

    wrapped: list[BaseTool] = []
    for tool in tools:
        if policy.requires_approval(tool.name):
            logger.debug("Approval: wrapping tool %r", tool.name)
            wrapped.append(
                _ApprovalToolWrapper(
                    inner=tool,
                    policy=policy,
                    pending_count=pending_count,
                    deny_counts=deny_counts,
                    used_request_ids=used_request_ids,
                    event_notifier=event_notifier,
                )
            )
        else:
            wrapped.append(tool)

    approval_count = sum(1 for t in wrapped if isinstance(t, _ApprovalToolWrapper))
    logger.info(
        "Approval: %d/%d tools require approval",
        approval_count,
        len(tools),
    )
    return wrapped
