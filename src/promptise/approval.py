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
import contextlib
import hashlib
import hmac as _hmac_mod
import inspect
import json
import logging
import secrets
import time
import typing
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Literal, Protocol, runtime_checkable

# Imported at runtime, not under TYPE_CHECKING: LangChain reads the
# ``_arun`` annotations to decide which of these to inject.
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables import RunnableConfig
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
    "verify_webhook_signature",
    "SIGNED_FIELDS",
]

#: The :class:`ApprovalRequest` fields the webhook signature covers.
SIGNED_FIELDS: tuple[str, ...] = (
    "request_id",
    "tool_name",
    "arguments",
    "agent_id",
    "caller_user_id",
    "timestamp",
)


def _signature_payload(fields: Mapping[str, Any]) -> bytes:
    """The bytes ``X-Promptise-Signature`` is the HMAC-SHA256 of."""
    return json.dumps(
        {name: fields.get(name) for name in SIGNED_FIELDS}, sort_keys=True, default=str
    ).encode()


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
        agent_id: Agent or process identifier: the agent identity's id or
            ``build_agent(observer_agent_id=...)``; in the Agent Runtime,
            the process name.  ``None`` when the agent has neither.
        caller_user_id: User who triggered the agent (from CallerContext).
        context_summary: The last few user/assistant messages of the
            conversation, for reviewer context (redacted like the
            arguments).  See ``ApprovalPolicy(context_messages=...)``.
        timestamp: When the request was created (``time.time()``).
        timeout: Seconds until auto-deny/allow.
        metadata: ``source`` (``"agent"`` for the agent's own gate),
            ``session_id`` and ``tenant_id`` when known, plus whatever
            ``ApprovalPolicy(metadata=...)`` adds.
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
        """Compute the HMAC-SHA256 signature sent as ``X-Promptise-Signature``.

        Covers the fields in :data:`SIGNED_FIELDS` (``request_id``,
        ``tool_name``, ``arguments``, ``agent_id``, ``caller_user_id``,
        ``timestamp``), serialized as sorted-key JSON.  Verify it on the
        receiving side with :func:`verify_webhook_signature`.
        """
        payload = _signature_payload(
            {name: getattr(self, name) for name in SIGNED_FIELDS},
        )
        return _hmac_mod.new(secret.encode(), payload, hashlib.sha256).hexdigest()


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
        http_client: Pre-configured ``httpx.AsyncClient`` (proxy, mTLS,
            custom CA).  Not closed by the handler.
        allow_private_networks: Allow ``url`` and ``poll_url`` to point at
            localhost, private, link-local or reserved addresses.  Off by
            default (SSRF protection); turn it on when the approval service
            runs on your own network, e.g. ``https://ops.internal/approvals``.

    Raises:
        ValueError: ``url`` or ``poll_url`` targets a private or internal
            address and ``allow_private_networks`` is not set.
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
        allow_private_networks: bool = False,
    ) -> None:
        # SSRF protection — reject private/internal URLs unless opted in
        if not allow_private_networks:
            from promptise.mcp.server._openapi import _validate_url_not_private

            hint = (
                "Pass WebhookApprovalHandler(..., allow_private_networks=True) "
                "if your approval service runs on a private network."
            )
            _validate_url_not_private(url, hint=hint)
            if poll_url is not None:
                _validate_url_not_private(poll_url, hint=hint)

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


def verify_webhook_signature(
    body: bytes | str | Mapping[str, Any],
    signature: str | None,
    secret: str,
    *,
    max_age: float | None = None,
) -> bool:
    """Check the ``X-Promptise-Signature`` of a webhook approval request.

    For the approval service that receives :class:`WebhookApprovalHandler`'s
    POST.  Recomputes the HMAC-SHA256 over the fields in
    :data:`SIGNED_FIELDS` and compares it in constant time::

        from promptise.approval import verify_webhook_signature

        @app.post("/approvals")
        async def receive(request: Request):
            body = await request.body()
            if not verify_webhook_signature(
                body,
                request.headers.get("X-Promptise-Signature"),
                os.environ["APPROVAL_WEBHOOK_SECRET"],
                max_age=300,
            ):
                raise HTTPException(401)
            ...

    The signature covers ``request_id``, ``tool_name``, ``arguments``,
    ``agent_id``, ``caller_user_id`` and ``timestamp`` — not
    ``context_summary``, ``timeout`` or ``metadata``.

    Args:
        body: The raw request body (``bytes`` / ``str``) or the parsed JSON
            object.
        signature: The ``X-Promptise-Signature`` header value.
        secret: The handler's ``secret``.
        max_age: Also reject requests whose ``timestamp`` is more than this
            many seconds away from now (replay protection).  ``None`` skips
            the check.

    Returns:
        ``True`` only when the body is a JSON object with every signed field,
        the signature matches, and (with ``max_age``) the timestamp is fresh.
    """
    if not signature:
        return False
    if isinstance(body, (bytes, bytearray, str)):
        try:
            data = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
    else:
        data = body
    if not isinstance(data, Mapping) or any(name not in data for name in SIGNED_FIELDS):
        return False
    expected = _hmac_mod.new(secret.encode(), _signature_payload(data), hashlib.sha256).hexdigest()
    if not _hmac_mod.compare_digest(expected, signature):
        return False
    if max_age is not None:
        timestamp = data.get("timestamp")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            return False
        if abs(time.time() - timestamp) > max_age:
            return False
    return True


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
        redact_sensitive: Run the arguments and ``context_summary``
            through PII/credential detection before sending them to the
            reviewer.  Each string value is redacted on its own, so the
            arguments keep their structure.
        max_pending: Maximum concurrent pending approvals per agent.
            Additional tool calls are auto-denied.
        max_retries_after_deny: After this many denials of the same tool
            within ``deny_window`` (and the same ``deny_scope``), later
            calls are denied without asking the reviewer.  ``None``
            disables the limit.
        deny_window: Seconds a denial counts towards
            ``max_retries_after_deny``.  Default: 600 (10 minutes).
            ``None`` keeps denials until the agent is rebuilt.
        deny_scope: Whose denials count together: ``"session"`` (default;
            the same user and ``chat()`` session), ``"user"`` (the same
            ``CallerContext`` user, across sessions) or ``"agent"`` (every
            caller of the agent).  Without a ``CallerContext``, callers
            share one count.
        sequential: Ask the reviewer one call at a time.  When the model
            requests several gated calls in one turn they run concurrently,
            so by default their approval requests arrive together; with
            ``True`` each waits until the previous one is decided.
        context_messages: How many of the conversation's last
            user/assistant messages go into ``context_summary``.  ``0``
            leaves it empty.
        metadata: Extra :attr:`ApprovalRequest.metadata` for every request:
            a dict, or a callable ``(tool_name, arguments) -> dict`` (sync
            or async) that receives the call's unredacted arguments.
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
        max_retries_after_deny: int | None = 3,
        deny_window: float | None = 600.0,
        deny_scope: Literal["session", "user", "agent"] = "session",
        sequential: bool = False,
        context_messages: int = 3,
        metadata: dict[str, Any] | Callable[[str, dict[str, Any]], Any] | None = None,
    ) -> None:
        if not tools:
            raise ValueError("ApprovalPolicy requires at least one tool pattern")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if timeout > 86400:
            raise ValueError("timeout cannot exceed 86400 seconds (24 hours)")
        if max_pending < 1:
            raise ValueError("max_pending must be at least 1")
        if max_retries_after_deny is not None and max_retries_after_deny < 1:
            raise ValueError("max_retries_after_deny must be at least 1, or None to disable it")
        if deny_window is not None and deny_window <= 0:
            raise ValueError("deny_window must be positive, or None to keep denials")
        if deny_scope not in ("session", "user", "agent"):
            raise ValueError(f"deny_scope must be 'session', 'user' or 'agent', got {deny_scope!r}")
        if context_messages < 0:
            raise ValueError("context_messages must be 0 or more")
        if metadata is not None and not (isinstance(metadata, dict) or callable(metadata)):
            raise TypeError(f"metadata must be a dict or a callable, got {type(metadata).__name__}")

        self.tools = tools
        self.timeout = timeout
        self.on_timeout = on_timeout
        self.include_arguments = include_arguments
        self.redact_sensitive = redact_sensitive
        self.max_pending = max_pending
        self.max_retries_after_deny = max_retries_after_deny
        self.deny_window = deny_window
        self.deny_scope = deny_scope
        self.sequential = sequential
        self.context_messages = context_messages
        self.metadata = metadata
        self._scanner: Any = None

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

    def _get_scanner(self) -> Any:
        """The PII/credential scanner used for redaction (built once)."""
        if self._scanner is None:
            from .guardrails import PromptiseSecurityScanner

            self._scanner = PromptiseSecurityScanner(
                detect_injection=False,
                detect_toxicity=False,
            )
        return self._scanner

    async def _redact(self, value: Any) -> Any:
        """Redact every string inside *value*, keeping its structure."""
        if isinstance(value, str):
            if not value:
                return value
            report = await self._get_scanner().scan_text(value, direction="output")
            return report.redacted_text if report.redacted_text is not None else value
        if isinstance(value, Mapping):
            return {key: await self._redact(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [await self._redact(item) for item in value]
        return value

    async def redact_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive data from arguments before sending to reviewer.

        Each string value — including strings nested in dicts and lists —
        is scanned on its own with the guardrails PII/credential detectors,
        and detected spans are replaced by labels such as ``[EMAIL]``.  Keys,
        numbers and the structure are kept: ``{"to": "dana@example.com",
        "amount": 18.5}`` becomes ``{"to": "[EMAIL]", "amount": 18.5}``.

        Returns the arguments unchanged when ``redact_sensitive`` is off.  If
        the guardrails module cannot be imported or the scan fails, a warning
        is logged and the arguments are returned unredacted.
        """
        if not self.redact_sensitive:
            return dict(arguments)
        try:
            return dict(await self._redact(arguments))
        except Exception as exc:
            logger.warning(
                "Approval: argument redaction failed (%s: %s); the reviewer sees "
                "the arguments unredacted",
                type(exc).__name__,
                exc,
            )
            return dict(arguments)

    async def summarize_context(self, messages: Sequence[Any]) -> str:
        """Build ``context_summary`` from a conversation's messages.

        Takes the last ``context_messages`` user/assistant messages that have
        text, one per line as ``user: ...`` / ``assistant: ...`` (each cut to
        500 characters), redacted like the arguments when
        ``redact_sensitive`` is on.  System and tool messages are skipped.
        """
        if self.context_messages == 0 or not messages:
            return ""
        lines: list[str] = []
        for message in reversed(messages):
            role, text = _message_role_and_text(message)
            if role is None or not text:
                continue
            lines.append(f"{role}: {_shorten(text, 500)}")
            if len(lines) == self.context_messages:
                break
        summary = "\n".join(reversed(lines))
        if summary and self.redact_sensitive:
            try:
                summary = await self._redact(summary)
            except Exception as exc:
                logger.warning(
                    "Approval: context redaction failed (%s: %s); context_summary left empty",
                    type(exc).__name__,
                    exc,
                )
                return ""
        return summary

    async def request_metadata(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """The developer-supplied part of :attr:`ApprovalRequest.metadata`."""
        if self.metadata is None:
            return {}
        if isinstance(self.metadata, dict):
            return dict(self.metadata)
        result = self.metadata(tool_name, dict(arguments))
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, dict):
            raise TypeError(
                f"ApprovalPolicy metadata callable must return a dict, got {type(result).__name__}"
            )
        return result


def _shorten(text: str, limit: int) -> str:
    """*text* cut to *limit* characters, marked with an ellipsis when cut."""
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _message_role_and_text(message: Any) -> tuple[str | None, str]:
    """``("user" | "assistant" | None, text)`` for a LangChain or dict message."""
    if isinstance(message, Mapping):
        role_raw, content = message.get("role") or message.get("type"), message.get("content")
    elif isinstance(message, (tuple, list)) and len(message) == 2:
        role_raw, content = message
    else:
        role_raw, content = getattr(message, "type", None), getattr(message, "content", None)
    role = {"user": "user", "human": "user", "assistant": "assistant", "ai": "assistant"}.get(
        str(role_raw).lower()
    )
    if isinstance(content, list):
        content = " ".join(
            block if isinstance(block, str) else str(block.get("text", ""))
            for block in content
            if isinstance(block, str)
            or (isinstance(block, Mapping) and block.get("type") == "text")
        )
    return role, content.strip() if isinstance(content, str) else ""


# ---------------------------------------------------------------------------
# Tool wrapper
# ---------------------------------------------------------------------------


class _DenialTracker:
    """Recent denials per key (scope + tool name), forgotten after a window."""

    #: Keys kept at most; the least recently denied are dropped first.
    MAX_KEYS = 10_000

    def __init__(self, window: float | None) -> None:
        self._window = window
        self._denials: OrderedDict[Hashable, list[float]] = OrderedDict()

    def count(self, key: Hashable) -> int:
        """Denials recorded for *key* within the window."""
        times = self._denials.get(key)
        if not times:
            return 0
        if self._window is not None:
            cutoff = time.monotonic() - self._window
            times[:] = [t for t in times if t > cutoff]
            if not times:
                del self._denials[key]
                return 0
        return len(times)

    def record(self, key: Hashable) -> None:
        """Record a denial for *key*."""
        self._denials.setdefault(key, []).append(time.monotonic())
        self._denials.move_to_end(key)
        while len(self._denials) > self.MAX_KEYS:
            self._denials.popitem(last=False)

    def clear(self, key: Hashable) -> None:
        """Forget *key*'s denials (the reviewer approved the tool)."""
        self._denials.pop(key, None)


class _GateState:
    """State shared by every gated tool of one agent."""

    def __init__(self, policy: ApprovalPolicy, *, pending: int = 0) -> None:
        self.pending = pending
        self.denials = _DenialTracker(policy.deny_window)
        self.used_request_ids: set[str] = set()  # Replay protection
        self._locks: dict[Hashable, tuple[asyncio.Lock, list[int]]] = {}

    @contextlib.asynccontextmanager
    async def one_at_a_time(self, key: Hashable) -> AsyncIterator[None]:
        """Serialize the block per *key* (``ApprovalPolicy(sequential=True)``)."""
        lock, users = self._locks.setdefault(key, (asyncio.Lock(), [0]))
        users[0] += 1
        try:
            async with lock:
                yield
        finally:
            users[0] -= 1
            if users[0] == 0:
                self._locks.pop(key, None)


def _current_scope(scope: str) -> tuple[Any, ...]:
    """The caller/session a denial counts against, per ``deny_scope``."""
    if scope == "agent":
        return ()
    from .agent import get_current_caller

    caller = get_current_caller()
    principal = caller.isolation_key if caller is not None else None
    if scope == "user":
        return (principal,)
    return (principal, _current_session_id())


def _current_session_id() -> str | None:
    """The ``chat()`` session id, or ``CallerContext.metadata["session_id"]``."""
    from .agent import _session_ctx_var, get_current_caller

    session_id = _session_ctx_var.get()
    if session_id is None:
        caller = get_current_caller()
        meta = getattr(caller, "metadata", None) or {}
        value = meta.get("session_id")
        session_id = str(value) if value is not None else None
    return session_id


def _format_window(seconds: float) -> str:
    """``600`` → ``"10 minutes"``."""
    for unit, size in (("hour", 3600), ("minute", 60)):
        if seconds >= size and seconds % size == 0:
            n = int(seconds // size)
            return f"{n} {unit}{'s' if n != 1 else ''}"
    return f"{seconds:g} seconds"


_UNSET = object()


def _apply_modified_arguments(
    original: dict[str, Any],
    shown: dict[str, Any],
    modified: dict[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, Any, Any]]]:
    """Merge a reviewer's ``modified_arguments`` onto the original call.

    Top-level fields in *modified* replace the original's; fields it leaves
    out keep their original values.  A field the reviewer sent back exactly
    as shown in the request (e.g. the redacted ``"[EMAIL]"``) is not a
    change and keeps the original value.

    Returns:
        The final arguments and the ``(name, old, new)`` changes.
    """
    final = dict(original)
    changes: list[tuple[str, Any, Any]] = []
    for name, value in modified.items():
        old = original.get(name, _UNSET)
        if old == value:
            continue
        if name in shown and shown[name] == value:
            continue  # echoed back unchanged (possibly redacted)
        final[name] = value
        changes.append((name, old, value))
    return final, changes


def _modification_note(changes: list[tuple[str, Any, Any]]) -> str:
    """The note that tells the model a reviewer changed its call."""

    def show(value: Any) -> str:
        if value is _UNSET:
            return "(not set)"
        return _shorten(json.dumps(value, default=str, ensure_ascii=False), 200)

    edits = "; ".join(f"{name}: {show(old)} -> {show(new)}" for name, old, new in changes)
    return (
        f"[Approved with modified arguments] A reviewer changed {edits}. "
        "The tool ran with the modified arguments and the result below reflects "
        "them; treat the reviewer's values as final."
    )


def _with_note(result: Any, note: str, response_format: str) -> Any:
    """Put *note* in front of a tool result, keeping its shape where possible."""
    if response_format == "content_and_artifact" and isinstance(result, tuple) and len(result) == 2:
        content, artifact = result
        return (_with_note(content, note, "content"), artifact)
    if isinstance(result, str):
        return f"{note}\n\n{result}"
    if isinstance(result, list) and all(
        isinstance(block, Mapping) and "type" in block for block in result
    ):
        return [{"type": "text", "text": note}, *result]
    if result is None:
        return note
    return f"{note}\n\n{json.dumps(result, default=str, ensure_ascii=False)}"


def _runnable_config_param(func: Callable[..., Any]) -> str | None:
    """The parameter of *func* annotated ``RunnableConfig``, if any.

    LangChain injects the run's config into such a parameter; tools built
    with ``@tool`` / ``StructuredTool`` require it.
    """
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        return None
    for name, hint in hints.items():
        if hint is RunnableConfig:
            return name
    return None


class _ApprovalToolWrapper(BaseTool):
    """Wraps a tool with an approval gate.

    Same name, description, and schema as the inner tool.  When
    ``_arun()`` is called, sends an approval request and waits for a human
    decision before executing.
    """

    _inner: Any = PrivateAttr()
    _policy: ApprovalPolicy = PrivateAttr()
    _state: _GateState = PrivateAttr()
    _event_notifier: Any = PrivateAttr(default=None)  # EventNotifier for events
    _agent_id: str | None = PrivateAttr(default=None)

    def __init__(
        self,
        inner: BaseTool,
        policy: ApprovalPolicy,
        state: _GateState | None = None,
        event_notifier: Any = None,
        agent_id: str | None = None,
    ) -> None:
        # Copy name, description, schema and result handling from inner tool
        super().__init__(
            name=inner.name,
            description=inner.description,
            args_schema=getattr(inner, "args_schema", None),
            response_format=getattr(inner, "response_format", "content"),
            return_direct=getattr(inner, "return_direct", False),
            handle_tool_error=getattr(inner, "handle_tool_error", False),
            handle_validation_error=getattr(inner, "handle_validation_error", False),
        )
        self._inner = inner
        self._policy = policy
        self._state = state if state is not None else _GateState(policy)
        self._event_notifier = event_notifier
        self._agent_id = agent_id

    def _emit(self, event_type: str, severity: str, data: dict[str, Any]) -> None:
        if self._event_notifier is None:
            return
        from .events import emit_event

        emit_event(self._event_notifier, event_type, severity, data, agent_id=self._agent_id)

    async def _arun(
        self,
        *,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> Any:
        """Intercept tool call, request approval, then execute or deny."""
        tool_name = self._inner.name
        policy = self._policy

        # Check max pending
        if self._state.pending >= policy.max_pending:
            logger.warning(
                "Approval: max_pending=%d reached, auto-denying %s",
                policy.max_pending,
                tool_name,
            )
            return (
                f"DENIED: Too many pending approval requests "
                f"(max {policy.max_pending}). Try again later."
            )

        # Check repeated denials (per caller/session, within the window)
        deny_key = (*_current_scope(policy.deny_scope), tool_name)
        limit = policy.max_retries_after_deny
        if limit is not None:
            denied = self._state.denials.count(deny_key)
            if denied >= limit:
                within = (
                    f" in the last {_format_window(policy.deny_window)}"
                    if policy.deny_window is not None
                    else ""
                )
                logger.info(
                    "Approval: %s denied %d times%s for this caller; not asking again",
                    tool_name,
                    denied,
                    within,
                )
                return (
                    f"DENIED: This action was already denied {denied} times{within}, "
                    "so the reviewer was not asked again. Do not retry this tool."
                )

        from .agent import _invocation_ctx_var, get_current_caller

        caller = get_current_caller()
        invocation = _invocation_ctx_var.get()
        session_id = _current_session_id()

        # Build approval request
        request_id = secrets.token_hex(16)
        arguments = await policy.redact_arguments(kwargs) if policy.include_arguments else {}
        metadata: dict[str, Any] = {"source": "agent"}
        if session_id is not None:
            metadata["session_id"] = session_id
        if caller is not None and caller.tenant_id:
            metadata["tenant_id"] = caller.tenant_id
        metadata.update(await policy.request_metadata(tool_name, kwargs))
        request = ApprovalRequest(
            request_id=request_id,
            tool_name=tool_name,
            arguments=arguments,
            agent_id=self._agent_id,
            caller_user_id=caller.user_id if caller is not None else None,
            context_summary=await policy.summarize_context(
                invocation.messages if invocation is not None else ()
            ),
            timeout=policy.timeout,
            metadata=metadata,
        )

        # Send approval request
        self._state.pending += 1
        try:
            async with contextlib.AsyncExitStack() as stack:
                if policy.sequential:
                    # One request at a time per invocation (or caller, outside one)
                    order_key = (
                        ("invocation", invocation.invocation_id)
                        if invocation is not None
                        else ("scope", *_current_scope("session"))
                    )
                    await stack.enter_async_context(self._state.one_at_a_time(order_key))
                logger.info(
                    "Approval: requesting approval for %s (request_id=%s)",
                    tool_name,
                    request_id,
                )
                self._emit(
                    "approval.requested",
                    "info",
                    {"tool_name": tool_name, "request_id": request_id, "timeout": policy.timeout},
                )
                decision = await asyncio.wait_for(
                    policy.handler.request_approval(request),
                    timeout=policy.timeout,
                )
        except asyncio.TimeoutError:
            decision = ApprovalDecision(
                approved=(policy.on_timeout == "allow"),
                reason=f"Approval timed out after {policy.timeout}s",
            )
            logger.warning(
                "Approval: timeout for %s (request_id=%s), on_timeout=%s",
                tool_name,
                request_id,
                policy.on_timeout,
            )
            if not decision.approved:
                self._state.denials.record(deny_key)
        except Exception as exc:
            logger.error(
                "Approval: handler error for %s: %s",
                tool_name,
                exc,
            )
            # Not counted as a denial: no reviewer said no.
            decision = ApprovalDecision(
                approved=False,
                reason=f"Approval handler error: {type(exc).__name__}",
            )
        else:
            if decision.approved:
                self._state.denials.clear(deny_key)
            else:
                self._state.denials.record(deny_key)
        finally:
            self._state.pending = max(0, self._state.pending - 1)

        # Replay protection — mark request_id as used
        self._state.used_request_ids.add(request_id)

        if not decision.approved:
            reason = decision.reason or "Action denied by reviewer."
            logger.info(
                "Approval: DENIED %s (request_id=%s): %s",
                tool_name,
                request_id,
                reason,
            )
            self._emit(
                "approval.denied",
                "warning",
                {"tool_name": tool_name, "request_id": request_id, "reason": reason},
            )
            return f"DENIED: {reason}"

        # Approved — execute with the original arguments, or the reviewer's edits
        final_args: dict[str, Any] = kwargs
        changes: list[tuple[str, Any, Any]] = []
        if decision.modified_arguments is not None:
            final_args, changes = _apply_modified_arguments(
                kwargs, request.arguments, decision.modified_arguments
            )
        logger.info(
            "Approval: APPROVED %s (request_id=%s, reviewer=%s%s)",
            tool_name,
            request_id,
            decision.reviewer_id or "unknown",
            f", modified: {[name for name, _, _ in changes]}" if changes else "",
        )
        self._emit(
            "approval.granted",
            "info",
            {
                "tool_name": tool_name,
                "request_id": request_id,
                "reviewer": decision.reviewer_id,
                "modified_arguments": [name for name, _, _ in changes],
            },
        )
        result = await self._run_inner(final_args, bool(changes), config, run_manager)
        if changes:
            return _with_note(result, _modification_note(changes), self.response_format)
        return result

    async def _run_inner(
        self,
        arguments: dict[str, Any],
        modified: bool,
        config: RunnableConfig,
        run_manager: AsyncCallbackManagerForToolRun | None,
    ) -> Any:
        """Run the inner tool the way ``BaseTool.arun`` would.

        Passes ``config`` and ``run_manager`` on when the inner tool takes
        them (``@tool`` / ``StructuredTool`` require ``config``), without
        opening a second tool run.  Arguments a reviewer changed are
        validated against the inner tool's schema first.
        """
        inner = self._inner
        call_args: tuple[Any, ...] = ()
        call_kwargs: dict[str, Any] = dict(arguments)
        if modified and isinstance(inner, BaseTool):
            call_args, call_kwargs = inner._to_args_and_kwargs(dict(arguments), None)
            call_kwargs = dict(call_kwargs)

        # A tool without its own _arun runs its _run in an executor; that is
        # the signature that decides what gets injected (as in BaseTool.arun).
        func = inner._arun
        if isinstance(inner, BaseTool) and type(inner)._arun is BaseTool._arun:
            func = inner._run
        if run_manager is not None and "run_manager" in inspect.signature(func).parameters:
            call_kwargs["run_manager"] = run_manager
        config_param = _runnable_config_param(func)
        if config_param is not None:
            call_kwargs[config_param] = config
        return await inner._arun(*call_args, **call_kwargs)

    def _run(
        self,
        *,
        config: RunnableConfig,
        run_manager: CallbackManagerForToolRun | None = None,
        **kwargs: Any,
    ) -> Any:  # pragma: no cover
        import anyio

        return anyio.run(lambda: self._arun(config=config, **kwargs))


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def wrap_tools_with_approval(
    tools: list[BaseTool],
    policy: ApprovalPolicy,
    *,
    event_notifier: Any = None,
    agent_id: str | None = None,
) -> list[BaseTool]:
    """Wrap tools that match the approval policy's patterns.

    Tools that don't match any pattern are returned as-is (zero overhead).
    Tools that match are wrapped in an approval gate.  All gates returned
    by one call share their state (pending count, denial counts), so call
    this once per agent.

    Args:
        tools: List of LangChain tools (from MCP, extra_tools, etc.).
        policy: The approval policy defining which tools need approval.
        event_notifier: Optional :class:`~promptise.events.EventNotifier`
            for ``approval.*`` events.
        agent_id: Sent as :attr:`ApprovalRequest.agent_id` and attached to
            events.  ``build_agent()`` passes the agent's identifier.

    Returns:
        New list with matching tools wrapped. Order preserved.
    """
    state = _GateState(policy)

    wrapped: list[BaseTool] = []
    for tool in tools:
        if policy.requires_approval(tool.name):
            logger.debug("Approval: wrapping tool %r", tool.name)
            wrapped.append(
                _ApprovalToolWrapper(
                    inner=tool,
                    policy=policy,
                    state=state,
                    event_notifier=event_notifier,
                    agent_id=agent_id,
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
