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
import contextvars
import copy
import hashlib
import hmac as _hmac_mod
import inspect
import json
import logging
import secrets
import threading
import time
import typing
from collections import OrderedDict
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Hashable,
    Iterator,
    Mapping,
    Sequence,
)
from dataclasses import replace as _dc_replace
from fnmatch import fnmatch
from typing import Any, Literal

# Imported at runtime, not under TYPE_CHECKING: LangChain reads the
# ``_arun`` annotations to decide which of these to inject.
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from pydantic import PrivateAttr

from ._approval_types import (
    SIGNED_FIELDS,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    CallbackApprovalHandler,
    _signature_payload,
)
from ._outbound import pin_target
from .tools import WrappingTool, plain_arguments

logger = logging.getLogger("promptise.approval")

#: Listeners told the tool name when an approval gate denies a call, for the
#: run in progress (see :func:`on_approval_denied`).
_denial_listeners: contextvars.ContextVar[tuple[Callable[[str], None], ...]] = (
    contextvars.ContextVar("promptise_approval_denial_listeners", default=())
)


@contextlib.contextmanager
def on_approval_denied(listener: Callable[[str], None]) -> Iterator[None]:
    """Call *listener* with the tool name of every call a gate denies in this context.

    The agent runtime uses it to give a denied call's irreversible-action
    budget back: the budget is counted when the call starts, before the
    gate asks, and a call that never ran must not use up
    ``max_irreversible_per_run``.
    """
    token = _denial_listeners.set((*_denial_listeners.get(), listener))
    try:
        yield
    finally:
        _denial_listeners.reset(token)


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
    "approval_elicitation_callback",
    "CONFIRMATION_FIELDS",
    "RequestIdCache",
    "ApprovalReplayError",
]

# ---------------------------------------------------------------------------
# Single-use request ids
# ---------------------------------------------------------------------------

#: The longest approval timeout ``ApprovalPolicy`` accepts (24 hours).
_MAX_APPROVAL_TIMEOUT = 86400.0


class ApprovalReplayError(KeyError):
    """A second decision arrived for a request id that already has one.

    Every approval request id takes exactly one decision.  Raised by
    :meth:`QueueApprovalHandler.submit_decision` for a second decision, and
    by the built-in handlers when they are asked to decide a request id they
    already decided.  A subclass of :class:`KeyError`, so code that catches
    ``KeyError`` for "no such pending request" keeps working.

    Attributes:
        request_id: The request id that was replayed.
    """

    def __init__(self, request_id: str, message: str | None = None) -> None:
        self.request_id = request_id
        self.message = message or (
            f"Approval request {request_id!r} was already decided; a request id takes one decision"
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class RequestIdCache:
    """Request ids that have had their one decision, remembered for a while.

    The approval gate, :class:`QueueApprovalHandler`,
    :class:`WebhookApprovalHandler` and the MCP server's ``PendingApprover``
    use one to make every request id decidable exactly once.  Pass one to
    :func:`verify_webhook_signature` (``replay_cache=``) on your approval
    service to refuse a replayed signed request.

    Memory is bounded twice: an id is forgotten ``ttl`` seconds after it was
    recorded (each :meth:`claim` can set its own), and at most ``max_size``
    ids are kept, the oldest dropped first.  An id is only useful for as long
    as a decision for it could still arrive, so the built-in users set the
    TTL to the request's ``timeout`` (or ``max_age`` for signatures).
    Thread-safe.

    Args:
        ttl: Default seconds an id is remembered.  Default 86400, the longest
            approval timeout.
        max_size: Most ids kept.  Default 10,000.  Size it above the number
            of requests you expect within the TTL: an id dropped early can be
            decided again.
    """

    def __init__(self, ttl: float = _MAX_APPROVAL_TIMEOUT, *, max_size: int = 10_000) -> None:
        if ttl <= 0:
            raise ValueError(f"ttl must be positive, got {ttl}")
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1, got {max_size}")
        self.ttl = ttl
        self.max_size = max_size
        self._expires: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        # Ids are kept in insertion order; drop expired ones from the front
        # (an entry with a longer TTL stops the sweep, which max_size bounds).
        while self._expires:
            request_id, expires = next(iter(self._expires.items()))
            if expires > now:
                break
            del self._expires[request_id]

    def claim(self, request_id: str, *, ttl: float | None = None) -> bool:
        """Record *request_id*; ``False`` when it is already recorded.

        ``True`` means this is the id's first decision and it may be used.
        ``False`` means the id was decided before (within its TTL): refuse
        the decision.

        Args:
            request_id: The id to record.
            ttl: Seconds to remember it; defaults to :attr:`ttl`.
        """
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            expires = self._expires.get(request_id)
            if expires is not None and expires > now:
                return False
            self._expires[request_id] = now + (ttl if ttl and ttl > 0 else self.ttl)
            self._expires.move_to_end(request_id)
            while len(self._expires) > self.max_size:
                self._expires.popitem(last=False)
            return True

    def __contains__(self, request_id: object) -> bool:
        with self._lock:
            expires = self._expires.get(request_id) if isinstance(request_id, str) else None
            return expires is not None and expires > time.monotonic()

    def __len__(self) -> int:
        with self._lock:
            self._prune(time.monotonic())
            return len(self._expires)


def _log_replay(where: str, request_id: str) -> None:
    logger.warning(
        "Approval: refused a second decision for request_id=%s (%s); a request id "
        "takes one decision",
        request_id,
        where,
    )


class _DecisionTimeout(Exception):
    """No answer came within the timeout :func:`_wait_for_decision` was given.

    Raised instead of ``TimeoutError`` so that a ``TimeoutError`` the handler
    raises itself is not mistaken for it: on Python 3.11+
    ``asyncio.TimeoutError`` *is* ``TimeoutError``, so an
    ``except asyncio.TimeoutError`` around ``asyncio.wait_for`` caught both.
    """


class _HandlerTimeoutError(Exception):
    """Carries a ``TimeoutError`` the awaited handler raised past ``wait_for``."""

    def __init__(self, error: BaseException) -> None:
        super().__init__(error)
        self.error = error


async def _wait_for_decision(awaitable: Awaitable[Any], timeout: float | None) -> Any:
    """``asyncio.wait_for`` for an approval handler, an elicitation, an agent
    invocation or a memory search: code that may raise ``TimeoutError`` itself.

    Raises :class:`_DecisionTimeout` when *timeout* passes first.  An error the
    handler raises propagates unchanged, a ``TimeoutError`` included, so the
    caller handles it as the handler's error, not as the gate's timeout.
    """

    async def _call() -> Any:
        try:
            return await awaitable
        except asyncio.TimeoutError as exc:
            raise _HandlerTimeoutError(exc) from exc

    try:
        return await asyncio.wait_for(_call(), timeout=timeout)
    except _HandlerTimeoutError as exc:
        raise exc.error from None
    except asyncio.TimeoutError:
        raise _DecisionTimeout(f"no decision within {timeout}s") from None


# ---------------------------------------------------------------------------
# Built-in handlers
# ---------------------------------------------------------------------------


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

    Unless ``allow_private_networks`` is set, the host is resolved again
    before the POST and before every poll, the request is refused if it
    resolves to a non-public address, and it is sent to the address that
    was checked (keeping the ``Host`` header and TLS server name), so a DNS
    answer that changes after construction (DNS rebinding) cannot redirect
    it to an internal service.  Redirects are never followed.

    Each request id takes one decision.  A poll response whose JSON carries
    a ``request_id`` must carry this request's id: a response for another
    request (a stale or replayed decision, e.g. from a fixed ``poll_url``) is
    logged and ignored, and polling continues.  Asking the handler again
    about a request id it already decided raises
    :class:`ApprovalReplayError`, which the agent's gate turns into a
    denial.  Decided ids are remembered for the request's ``timeout``.

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
        self._allow_private_networks = allow_private_networks
        self._secret = secret or secrets.token_hex(32)
        self._poll_url = poll_url
        self._poll_interval = max(0.5, poll_interval)
        self._headers = headers or {}
        self._http_client = http_client  # Optional pre-configured httpx.AsyncClient
        self._decided = RequestIdCache()

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """POST request to webhook, poll for decision.

        Raises:
            ApprovalReplayError: This handler already decided *request*'s id.
        """
        try:
            import httpx
        except ImportError:
            raise ImportError(
                "httpx is required for WebhookApprovalHandler. Install with: pip install httpx"
            )
        if request.request_id in self._decided:
            _log_replay("WebhookApprovalHandler", request.request_id)
            raise ApprovalReplayError(request.request_id)

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
            # POST the approval request.  Every request resolves and checks
            # the host again and connects to the checked address (DNS
            # rebinding); a private answer raises BlockedTarget (fail closed).
            target = await pin_target(
                self._url, allow_private_networks=self._allow_private_networks
            )
            resp = await client.post(
                target.url,
                json=request.to_dict(),
                headers={**headers, **target.headers},
                extensions=target.extensions,
                follow_redirects=False,
            )
            resp.raise_for_status()

            # Poll for decision
            poll_url = self._poll_url or f"{self._url}/{request.request_id}"
            deadline = time.monotonic() + request.timeout

            while time.monotonic() < deadline:
                await asyncio.sleep(self._poll_interval)
                try:
                    target = await pin_target(
                        poll_url, allow_private_networks=self._allow_private_networks
                    )
                    poll_resp = await client.get(
                        target.url,
                        headers={
                            "X-Promptise-Request-Id": request.request_id,
                            **self._headers,
                            **target.headers,
                        },
                        extensions=target.extensions,
                        follow_redirects=False,
                    )
                    if poll_resp.status_code == 200:
                        data = poll_resp.json()
                        if (
                            isinstance(data, dict)
                            and "request_id" in data
                            and data["request_id"] != request.request_id
                        ):
                            logger.warning(
                                "Approval: ignored a poll response for request_id=%r while "
                                "waiting for request_id=%s (stale or replayed decision)",
                                data["request_id"],
                                request.request_id,
                            )
                            continue
                        if isinstance(data, dict) and "approved" in data:
                            if not self._decided.claim(request.request_id, ttl=request.timeout):
                                _log_replay("WebhookApprovalHandler", request.request_id)
                                raise ApprovalReplayError(request.request_id)
                            return ApprovalDecision(
                                approved=data["approved"],
                                modified_arguments=data.get("modified_arguments"),
                                reviewer_id=data.get("reviewer_id"),
                                reason=data.get("reason"),
                            )
                    # 202 = still pending, continue polling
                except (httpx.HTTPError, OSError):
                    # Transient (connection, DNS); a BlockedTarget is not
                    # caught here, so a private answer ends the request.
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
    replay_cache: RequestIdCache | None = None,
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
                replay_cache=seen_requests,  # RequestIdCache(), created once
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
            many seconds away from now.  ``None`` skips the check.
        replay_cache: A :class:`RequestIdCache` shared by every call: a
            request whose ``request_id`` already passed is rejected (and
            logged) as a replay.  Ids are remembered for twice ``max_age``
            when it is set (the longest a timestamp can stay within
            ``max_age`` of now), else for the cache's ``ttl``.  Without it, a captured request
            verifies again until ``max_age`` runs out.

    Returns:
        ``True`` only when the body is a JSON object with every signed field,
        the signature matches, (with ``max_age``) the timestamp is fresh, and
        (with ``replay_cache``) its ``request_id`` was not seen before.
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
    if replay_cache is not None:
        request_id = data.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            return False
        # A timestamp up to max_age in the future verifies until max_age past
        # it, so remember the id for twice max_age.
        ttl = 2 * max_age if max_age is not None else None
        if not replay_cache.claim(request_id, ttl=ttl):
            _log_replay("verify_webhook_signature", request_id)
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

    Each request id takes one decision: a second :meth:`submit_decision`
    for the same id raises :class:`ApprovalReplayError` (a ``KeyError``) and
    is logged, and the first decision stands.  Decided ids are remembered for
    the request's ``timeout``.

    Attributes:
        request_queue: Queue of pending :class:`ApprovalRequest` objects.
    """

    def __init__(self, maxsize: int = 100) -> None:
        self.request_queue: asyncio.Queue[ApprovalRequest] = asyncio.Queue(maxsize=maxsize)
        self._pending: dict[str, asyncio.Future[ApprovalDecision]] = {}
        self._timeouts: dict[str, float] = {}
        self._decided = RequestIdCache()
        self._lock = asyncio.Lock()

    def submit_decision(self, request_id: str, decision: ApprovalDecision) -> None:
        """Submit a decision for a pending request.

        Called by the UI after the human reviewer makes a choice.

        Args:
            request_id: The ``request_id`` from the :class:`ApprovalRequest`.
            decision: The reviewer's decision.

        Raises:
            ApprovalReplayError: The request already has a decision (a
                subclass of ``KeyError``).
            KeyError: If no pending request with this ID exists.
        """
        future = self._pending.get(request_id)
        if future is None:
            if request_id in self._decided:
                _log_replay("QueueApprovalHandler", request_id)
                raise ApprovalReplayError(request_id)
            raise KeyError(f"No pending approval request with id {request_id!r}")
        if future.done() or not self._decided.claim(request_id, ttl=self._timeouts.get(request_id)):
            _log_replay("QueueApprovalHandler", request_id)
            raise ApprovalReplayError(request_id)
        future.set_result(decision)

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Enqueue request and wait for decision from the UI.

        Raises:
            ApprovalReplayError: *request*'s id is already pending here or
                was already decided.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[ApprovalDecision] = loop.create_future()

        async with self._lock:
            if request.request_id in self._pending or request.request_id in self._decided:
                _log_replay("QueueApprovalHandler", request.request_id)
                raise ApprovalReplayError(request.request_id)
            self._pending[request.request_id] = future
            self._timeouts[request.request_id] = request.timeout

        try:
            await self.request_queue.put(request)
            # Wait for the UI to call submit_decision()
            return await asyncio.wait_for(future, timeout=request.timeout)
        except asyncio.TimeoutError:
            raise
        finally:
            async with self._lock:
                self._pending.pop(request.request_id, None)
                self._timeouts.pop(request.request_id, None)


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
            Classifier rules still see them (``raw_arguments``).
        redact_sensitive: Run the arguments and ``context_summary``
            through PII/credential detection before sending them to the
            reviewer.  Each string value is redacted on its own, so the
            arguments keep their structure.  Only the reviewer's copy is
            redacted: an
            :class:`~promptise.approval_classifier.AutoApprovalClassifier`
            matches its rules against the real arguments.
        max_pending: Maximum concurrent pending approvals per agent.
            Additional tool calls are denied without asking the handler
            (``decided_by="gate"``).
        max_retries_after_deny: After this many reviewer denials of the
            same tool within ``deny_window`` (and the same
            ``deny_scope``), later calls are denied without asking the
            handler (``decided_by="gate"``).  Denials by classifier rules
            don't count.  ``None`` disables the limit.
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
        on_decision: Called with ``(request, decision)`` for every decision
            the agent's gate reaches — the handler's, a classifier rule's,
            and the gate's own (``max_pending``, the retry limit, a
            timeout, a handler error) — before the tool runs, and for
            server-side approval gates answered through MCP elicitation
            (``request.metadata["source"] == "mcp_elicitation"``).  Sync or
            async.  Use it for the audit log: ``decision.decided_by`` and
            ``decision.trace`` say what decided.  ``request.arguments`` is
            the redacted copy.  Errors it raises are logged and ignored.
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
        on_decision: Callable[[ApprovalRequest, ApprovalDecision], Any] | None = None,
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
        if on_decision is not None and not callable(on_decision):
            raise TypeError(f"on_decision must be callable, got {type(on_decision).__name__}")

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
        self.on_decision = on_decision
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

    async def record_decision(self, request: ApprovalRequest, decision: ApprovalDecision) -> None:
        """Pass a decision to ``on_decision``, logging (not raising) its errors."""
        if self.on_decision is None:
            return
        try:
            result = self.on_decision(request, decision)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception(
                "Approval: on_decision raised for %s (request_id=%s); ignored",
                request.tool_name,
                request.request_id,
            )


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
        # Every request id takes one decision.  Remembered for the policy's
        # timeout: a decision can only arrive while its request is pending.
        self.used_request_ids = RequestIdCache(ttl=policy.timeout)
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


#: MCP tool annotation keys copied into :attr:`ApprovalRequest.tool_annotations`.
_ANNOTATION_KEYS = ("title", "readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint")


def _tool_annotations(tool: Any) -> dict[str, Any]:
    """The MCP annotations a LangChain tool carries in its ``metadata``.

    Promptise's MCP tools and ``langchain-mcp-adapters`` both put them there
    as flat keys (``{"readOnlyHint": True, ...}``).
    """
    meta = getattr(tool, "metadata", None)
    if not isinstance(meta, Mapping):
        return {}
    return {key: meta[key] for key in _ANNOTATION_KEYS if meta.get(key) is not None}


def _record_approval_timeline(event_type: str, data: dict[str, Any]) -> None:
    """Record an ``approval.*`` event on the observed agent run's timeline.

    The event notifier gets these events too; this puts them on the trace
    (``observe=``, its transporters, ``promptise run --devtools``) next to
    the tool call they belong to.  A no-op outside an observed run.
    """
    from .observability import (
        TimelineEventType,
        _json_safe_arguments,
        _record_on_current_run,
        get_current_run,
    )

    run = get_current_run()
    if run is None or run.collector is None:
        return
    try:
        event = TimelineEventType(event_type)
    except ValueError:
        return
    metadata = dict(data)
    if "arguments" in metadata:
        if run.record_tool_io:
            metadata["arguments"] = _json_safe_arguments(metadata["arguments"])
        else:
            metadata.pop("arguments")
    tool = metadata.get("tool_name") or "?"
    details = f"{event_type}: {tool}"
    if metadata.get("reason"):
        details += f": {metadata['reason']}"
    _record_on_current_run(event, details=details, metadata=metadata)


def _decision_event_fields(decision: ApprovalDecision) -> dict[str, Any]:
    """``decided_by`` (and the classifier layer, if any) for ``approval.*`` events."""
    fields: dict[str, Any] = {"decided_by": decision.decided_by}
    if decision.trace is not None:
        fields["classifier_layer"] = decision.trace.layer
    return fields


class _ApprovalToolWrapper(WrappingTool):
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
            metadata=getattr(inner, "metadata", None),
        )
        self._inner = inner
        self._policy = policy
        self._state = state if state is not None else _GateState(policy)
        self._event_notifier = event_notifier
        self._agent_id = agent_id

    def _emit(self, event_type: str, severity: str, data: dict[str, Any]) -> None:
        _record_approval_timeline(event_type, data)
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

        from .agent import _invocation_ctx_var, get_current_caller

        caller = get_current_caller()
        invocation = _invocation_ctx_var.get()
        session_id = _current_session_id()

        # Build approval request.  ``arguments`` is the reviewer's (redacted)
        # copy; ``raw_arguments`` is what classifier rules match against.
        # Both are plain JSON (nested objects as dicts, not model instances).
        request_id = secrets.token_hex(16)
        plain: dict[str, Any] = plain_arguments(kwargs)
        arguments = (
            await policy.redact_arguments(copy.deepcopy(plain)) if policy.include_arguments else {}
        )
        metadata: dict[str, Any] = {"source": "agent"}
        if session_id is not None:
            metadata["session_id"] = session_id
        if caller is not None and caller.tenant_id:
            metadata["tenant_id"] = caller.tenant_id
        metadata.update(await policy.request_metadata(tool_name, dict(plain)))
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
            tool_annotations=_tool_annotations(self._inner),
            raw_arguments=plain,
        )

        deny_key = (*_current_scope(policy.deny_scope), tool_name)
        decision = self._check_limits(request, deny_key)
        if decision is None:
            decision = await self._ask(request, deny_key, invocation)
        await policy.record_decision(request, decision)

        if not decision.approved:
            reason = decision.reason or "Action denied by reviewer."
            logger.info(
                "Approval: DENIED %s (request_id=%s, decided_by=%s): %s",
                tool_name,
                request_id,
                decision.decided_by,
                reason,
            )
            self._emit(
                "approval.denied",
                "warning",
                {
                    "tool_name": tool_name,
                    "request_id": request_id,
                    "reason": reason,
                    **_decision_event_fields(decision),
                },
            )
            for listener in _denial_listeners.get():
                try:
                    listener(tool_name)
                except Exception:
                    logger.debug("Approval denial listener failed", exc_info=True)
            from .observability import _mark_tool_run_denied

            # The trace's tool.result says the call was denied, not "ok".
            _mark_tool_run_denied(
                run_manager.run_id if run_manager is not None else None,
                reason=reason,
                denied_by="approval",
                decided_by=decision.decided_by,
                request_id=request_id,
            )
            return f"DENIED: {reason}"

        # Approved — execute with the original arguments, or the reviewer's edits
        final_args: dict[str, Any] = kwargs
        changes: list[tuple[str, Any, Any]] = []
        if decision.modified_arguments is not None:
            # Merged in plain JSON; _run_inner validates the result against
            # the inner tool's schema.
            modified_args, changes = _apply_modified_arguments(
                plain, request.arguments, decision.modified_arguments
            )
            if changes:
                final_args = modified_args
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
                **_decision_event_fields(decision),
            },
        )
        result = await self._run_inner(final_args, bool(changes), config, run_manager)
        if changes:
            return _with_note(result, _modification_note(changes), self.response_format)
        return result

    def _check_limits(
        self, request: ApprovalRequest, deny_key: Hashable
    ) -> ApprovalDecision | None:
        """The gate's own denial when ``max_pending`` or the retry limit applies."""
        policy = self._policy
        tool_name = request.tool_name
        if self._state.pending >= policy.max_pending:
            logger.warning(
                "Approval: max_pending=%d reached, auto-denying %s",
                policy.max_pending,
                tool_name,
            )
            return ApprovalDecision(
                approved=False,
                reason=(
                    f"Too many pending approval requests (max {policy.max_pending}). "
                    "Try again later."
                ),
                decided_by="gate",
            )

        # Repeated reviewer denials (per caller/session, within the window)
        limit = policy.max_retries_after_deny
        if limit is None:
            return None
        denied = self._state.denials.count(deny_key)
        if denied < limit:
            return None
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
        return ApprovalDecision(
            approved=False,
            reason=(
                f"This action was already denied {denied} times{within}, "
                "so the reviewer was not asked again. Do not retry this tool."
            ),
            decided_by="gate",
        )

    async def _ask(
        self, request: ApprovalRequest, deny_key: Hashable, invocation: Any
    ) -> ApprovalDecision:
        """Send *request* to the handler and track the reviewer's denials."""
        policy = self._policy
        tool_name = request.tool_name
        request_id = request.request_id
        if request_id in self._state.used_request_ids:
            _log_replay("agent approval gate", request_id)
            return self._replay_denial()
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
                    {
                        "tool_name": tool_name,
                        "request_id": request_id,
                        "timeout": policy.timeout,
                        # The reviewer's redacted copy ({} when include_arguments=False),
                        # never raw_arguments
                        "arguments": request.arguments,
                    },
                )
                decision = await _wait_for_decision(
                    policy.handler.request_approval(request), policy.timeout
                )
        except _DecisionTimeout:
            decision = ApprovalDecision(
                approved=(policy.on_timeout == "allow"),
                reason=f"Approval timed out after {policy.timeout}s",
                decided_by="gate",
            )
            _record_approval_timeline(
                "approval.timeout",
                {
                    "tool_name": tool_name,
                    "request_id": request_id,
                    "timeout": policy.timeout,
                    "on_timeout": policy.on_timeout,
                },
            )
            logger.warning(
                "Approval: timeout for %s (request_id=%s), on_timeout=%s",
                tool_name,
                request_id,
                policy.on_timeout,
            )
            if not decision.approved:
                # A reviewer was asked and didn't answer: counts like a denial.
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
                decided_by="gate",
            )
        else:
            # Only a reviewer's decisions move the retry count: a classifier
            # rule's denial doesn't use up the reviewer's patience, and its
            # approval doesn't reset what the reviewer denied.
            if decision.decided_by == "reviewer":
                if decision.approved:
                    self._state.denials.clear(deny_key)
                else:
                    self._state.denials.record(deny_key)
        finally:
            self._state.pending = max(0, self._state.pending - 1)

        # A request id takes one decision: one that was already decided
        # (a replay) is refused, whatever this decision says.
        if not self._state.used_request_ids.claim(request_id):
            _log_replay("agent approval gate", request_id)
            return self._replay_denial()
        return decision

    @staticmethod
    def _replay_denial() -> ApprovalDecision:
        return ApprovalDecision(
            approved=False,
            reason="This approval request id was already decided; a request id takes one decision.",
            decided_by="gate",
        )

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


# ---------------------------------------------------------------------------
# Server-side approval gates: answering MCP elicitation with a handler
# ---------------------------------------------------------------------------

#: Boolean field names read as "the reviewer approves" in an elicitation
#: form.  A form answered through :func:`approval_elicitation_callback` must
#: have no fields (a bare confirmation) or exactly one of these as its
#: boolean decision field, with nothing else required.
CONFIRMATION_FIELDS: tuple[str, ...] = (
    "approve",
    "approved",
    "confirm",
    "confirmed",
    "accept",
    "accepted",
    "proceed",
    "allow",
)


def _confirmation_content(schema: Any, reason: str | None) -> dict[str, Any] | None:
    """The form content that answers a confirmation-shaped request "yes".

    Returns ``None`` when *schema* is not a confirmation — the request asks
    for input an approve/deny decision cannot supply, so it must be declined
    rather than filled in.
    """
    if schema is None:
        return {}
    if not isinstance(schema, dict):
        return None
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        return None
    if not properties:
        return {}
    decision_fields = [
        name
        for name, prop in properties.items()
        if name in CONFIRMATION_FIELDS and isinstance(prop, dict) and prop.get("type") == "boolean"
    ]
    if len(decision_fields) != 1:
        return None
    decision_field = decision_fields[0]
    required = set(schema.get("required") or [])
    if required - {decision_field}:
        return None
    content: dict[str, Any] = {decision_field: True}
    reason_prop = properties.get("reason")
    if reason and isinstance(reason_prop, dict) and reason_prop.get("type") == "string":
        content["reason"] = reason
    return content


def _caller_user_id() -> str | None:
    """``user_id`` of the current :class:`~promptise.agent.CallerContext`, if any."""
    try:
        from .agent import get_current_caller
    except ImportError:  # pragma: no cover - agent module always ships
        return None
    caller = get_current_caller()
    return getattr(caller, "user_id", None) if caller is not None else None


def approval_elicitation_callback(
    handler: ApprovalPolicy | ApprovalHandler | Callable[..., Any],
    *,
    server_name: str | None = None,
    in_flight: Callable[[], Sequence[Any]] | None = None,
    timeout: float | None = None,
    event_notifier: Any = None,
) -> Callable[[Any, Any], Awaitable[Any]]:
    """Build an MCP ``elicitation_callback`` that asks an approval handler.

    Lets the human behind a Promptise client approve **server-side**
    approval gates — e.g. ``ApprovalGateMiddleware`` with
    ``ElicitationApprover`` on an MCPcast-generated server, which asks the
    calling client to confirm a gated tool call through MCP elicitation.
    ``build_agent(approval=...)`` installs one per server automatically;
    use this directly with :class:`~promptise.mcp.client.MCPClient`::

        client = MCPClient(
            transport="stdio",
            command="python",
            args=["petstore-mcp/server.py"],
            elicitation_callback=approval_elicitation_callback(
                CallbackApprovalHandler(ask_human),
                server_name="petstore",
                in_flight=lambda: client.in_flight_calls,
            ),
        )

    Each request becomes an :class:`ApprovalRequest`: ``context_summary``
    carries the server's message, ``metadata`` the message, the requested
    schema and the server name (``source="mcp_elicitation"``).  When exactly
    one tool call is in flight on the connection, ``tool_name`` and
    ``arguments`` are that call's — the arguments this client actually
    sent — and the handler runs in the caller's context (so
    ``get_current_caller()`` works and ``caller_user_id`` is set).  When no
    call or several calls are in flight, the request cannot be tied to one,
    so ``tool_name`` is ``""``, ``arguments`` is empty and
    ``metadata["in_flight_tools"]`` lists the candidates; the reviewer
    decides from the server's message alone.

    The decision maps back as follows, failing closed at every step:

    - approved → ``accept``, with the form's decision field set to ``True``
      (and ``reason`` filled in when the form has one);
    - denied → ``decline``;
    - a timeout or a handler error → ``decline`` — never ``accept``, even
      when the policy says ``on_timeout="allow"``: that setting governs the
      agent's own gate, not one the server asked a human to clear;
    - ``modified_arguments`` → ``decline`` (the server has already bound the
      arguments; executing the originals after a reviewer changed them
      would run something nobody approved);
    - a request that is not a confirmation — URL mode, or a form whose
      fields are not one boolean from :data:`CONFIRMATION_FIELDS` plus an
      optional ``reason`` — is declined without asking the handler.

    Args:
        handler: An :class:`ApprovalPolicy` (its handler, ``timeout``,
            ``include_arguments`` and ``redact_sensitive`` apply; its tool
            patterns do not — the server decided the call needs approval),
            an :class:`ApprovalHandler`, or a callable accepted by
            :class:`CallbackApprovalHandler`.  An
            :class:`~promptise.approval_classifier.AutoApprovalClassifier`
            is bypassed in favour of its human ``fallback``: its auto-allow
            rules were written for the agent's own gate, and must not clear
            a gate the server put in front of a human.
        server_name: Name shown to the reviewer and recorded in metadata.
        in_flight: Returns the connection's in-flight calls — pass
            ``lambda: client.in_flight_calls``.  Without it, requests are
            never tied to a tool call.
        timeout: Seconds to wait for the handler.  Defaults to the
            policy's ``timeout``, or 300.
        event_notifier: Optional :class:`~promptise.events.EventNotifier`;
            ``approval.requested`` / ``approval.granted`` /
            ``approval.denied`` are emitted with ``source="mcp_elicitation"``.

    Returns:
        An async ``(context, params) -> ElicitResult`` callable for
        ``MCPClient(elicitation_callback=...)``.
    """
    policy = handler if isinstance(handler, ApprovalPolicy) else None
    target: ApprovalHandler
    if policy is not None:
        target = policy.handler
    elif isinstance(handler, ApprovalHandler):
        target = handler
    elif callable(handler):
        target = CallbackApprovalHandler(handler)
    else:
        raise TypeError(
            "handler must be an ApprovalPolicy, an ApprovalHandler or a callable, "
            f"got {type(handler).__name__}"
        )

    from .approval_classifier import AutoApprovalClassifier

    if isinstance(target, AutoApprovalClassifier):
        target = target.fallback

    if timeout is None:
        timeout = policy.timeout if policy is not None else 300.0
    if not (0 < timeout <= 86400):
        raise ValueError(f"timeout must be in (0, 86400], got {timeout}")
    effective_timeout: float = timeout
    include_arguments = policy.include_arguments if policy is not None else True
    where = f"Server {server_name!r}" if server_name else "The MCP server"

    async def _callback(context: Any, params: Any) -> Any:
        from mcp import types

        decline = types.ElicitResult(action="decline")
        message = str(getattr(params, "message", "") or "")
        mode = getattr(params, "mode", None) or "form"
        if mode != "form":
            logger.warning(
                "Approval: %s sent a %s-mode elicitation; an approval handler only "
                "answers confirmation forms; declined",
                where,
                mode,
            )
            return decline
        from .mcp._compat import field as mcp_field

        schema = mcp_field(params, "requested_schema")
        if _confirmation_content(schema, None) is None:
            logger.warning(
                "Approval: %s asked for input that is not a confirmation (%r); declined",
                where,
                message,
            )
            return decline

        calls = list(in_flight()) if in_flight is not None else []
        call = calls[0] if len(calls) == 1 else None
        # Run the handler and events in the caller's context when the request
        # belongs to a known call: the elicitation arrives on the session's own
        # task, where the agent's CallerContext is not set.
        run_in_caller: Callable[..., Any] = (
            call.context.run if call is not None else lambda fn, *a, **kw: fn(*a, **kw)
        )

        arguments: dict[str, Any] = {}
        if call is not None and include_arguments:
            arguments = (
                await policy.redact_arguments(call.arguments)
                if policy is not None
                else dict(call.arguments)
            )
        request = ApprovalRequest(
            request_id=secrets.token_hex(16),
            tool_name=call.name if call is not None else "",
            arguments=arguments,
            caller_user_id=run_in_caller(_caller_user_id),
            context_summary=f"{where} asks: {message}",
            timeout=effective_timeout,
            metadata={
                "source": "mcp_elicitation",
                "server": server_name,
                "elicitation_message": message,
                "requested_schema": schema,
                "in_flight_tools": [c.name for c in calls],
            },
        )

        def _emit(event_type: str, severity: str, data: dict[str, Any]) -> None:
            payload = {
                "tool_name": request.tool_name,
                "request_id": request.request_id,
                "source": "mcp_elicitation",
                "server": server_name,
                **data,
            }
            # In the caller's context: that is where the agent run is current.
            run_in_caller(_record_approval_timeline, event_type, payload)
            if event_notifier is None:
                return
            from .events import emit_event

            run_in_caller(emit_event, event_notifier, event_type, severity, payload)

        logger.info(
            "Approval: %s requests approval%s (request_id=%s)",
            where,
            f" for {request.tool_name}" if request.tool_name else "",
            request.request_id,
        )
        _emit("approval.requested", "info", {"timeout": effective_timeout})

        async def _record(decision: ApprovalDecision) -> None:
            # ``ApprovalPolicy(on_decision=...)`` sees these decisions too.
            if policy is not None:
                await run_in_caller(
                    asyncio.ensure_future, policy.record_decision(request, decision)
                )

        try:
            task = run_in_caller(asyncio.ensure_future, target.request_approval(request))
            decision = await _wait_for_decision(task, effective_timeout)
        except _DecisionTimeout:
            logger.warning(
                "Approval: no decision on %s's request within %.0fs (request_id=%s); declined",
                where,
                effective_timeout,
                request.request_id,
            )
            _emit("approval.denied", "warning", {"reason": "timeout"})
            await _record(
                ApprovalDecision(
                    approved=False,
                    reason=f"Approval timed out after {effective_timeout}s",
                    decided_by="gate",
                )
            )
            return decline
        except Exception as exc:
            logger.error(
                "Approval: handler error on %s's request (%s: %s); declined",
                where,
                type(exc).__name__,
                exc,
            )
            _emit("approval.denied", "warning", {"reason": f"handler error: {type(exc).__name__}"})
            await _record(
                ApprovalDecision(
                    approved=False,
                    reason=f"Approval handler error: {type(exc).__name__}",
                    decided_by="gate",
                )
            )
            return decline

        if not isinstance(decision, ApprovalDecision) or not decision.approved:
            reason = getattr(decision, "reason", None) or "denied by reviewer"
            await _record(
                decision
                if isinstance(decision, ApprovalDecision)
                else ApprovalDecision(approved=False, reason=reason, decided_by="gate")
            )
            logger.info(
                "Approval: DENIED %s's request (request_id=%s): %s",
                where,
                request.request_id,
                reason,
            )
            _emit("approval.denied", "warning", {"reason": reason})
            return decline
        if decision.modified_arguments is not None:
            logger.warning(
                "Approval: reviewer modified the arguments of %s's request "
                "(request_id=%s); a server-side gate cannot apply that; declined",
                where,
                request.request_id,
            )
            _emit("approval.denied", "warning", {"reason": "modified arguments"})
            await _record(
                _dc_replace(
                    decision,
                    approved=False,
                    reason="The reviewer modified the arguments; a server-side gate "
                    "cannot apply that, so the request was declined",
                    decided_by="gate",
                )
            )
            return decline

        logger.info(
            "Approval: APPROVED %s's request (request_id=%s, reviewer=%s)",
            where,
            request.request_id,
            decision.reviewer_id or "unknown",
        )
        _emit("approval.granted", "info", {"reviewer": decision.reviewer_id})
        await _record(decision)
        return types.ElicitResult(
            action="accept", content=_confirmation_content(schema, decision.reason)
        )

    return _callback


def _human_node_handler(
    approval: ApprovalPolicy | ApprovalHandler | Callable[..., Any],
) -> Callable[[str, Any], Awaitable[bool]]:
    """A Reasoning Graph ``HumanNode`` handler that asks an approval handler.

    ``build_agent(approval=...)`` passes it to the engine, so a ``HumanNode``
    in the agent's graph asks the same reviewer as the agent's tool gate.
    The request carries the node's prompt as ``context_summary``, the node
    name as ``tool_name`` and ``metadata["source"] = "human_node"``.  An
    :class:`~promptise.approval_classifier.AutoApprovalClassifier` is
    bypassed in favour of its human ``fallback`` (its rules are written for
    tool calls).  A handler error is a denial.

    Args:
        approval: An :class:`ApprovalPolicy` (its handler and ``timeout``
            apply), an :class:`ApprovalHandler`, or a callable accepted by
            :class:`CallbackApprovalHandler`.

    Returns:
        An async ``(prompt, state) -> bool``.
    """
    policy = approval if isinstance(approval, ApprovalPolicy) else None
    target: ApprovalHandler
    if policy is not None:
        target = policy.handler
    elif isinstance(approval, ApprovalHandler):
        target = approval
    elif callable(approval):
        target = CallbackApprovalHandler(approval)
    else:
        raise TypeError(
            "approval must be an ApprovalPolicy, an ApprovalHandler or a callable, "
            f"got {type(approval).__name__}"
        )

    from .approval_classifier import AutoApprovalClassifier

    if isinstance(target, AutoApprovalClassifier):
        target = target.fallback
    timeout = policy.timeout if policy is not None else 300.0

    async def _ask(prompt: str, state: Any) -> bool:
        node_name = str(getattr(state, "current_node", "") or "")
        request = ApprovalRequest(
            request_id=secrets.token_hex(16),
            tool_name=node_name,
            arguments={},
            caller_user_id=_caller_user_id(),
            context_summary=prompt,
            timeout=timeout,
            metadata={"source": "human_node", "node": node_name},
        )
        try:
            decision = await target.request_approval(request)
        except Exception as exc:
            logger.warning(
                "Approval: the handler failed for HumanNode %r, denied: %s", node_name, exc
            )
            return False
        return bool(getattr(decision, "approved", False))

    return _ask
