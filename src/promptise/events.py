"""Webhook and event notification system for Promptise agents.

Emits structured notifications when significant things happen during
agent execution — invocation complete, tool failure, guardrail block,
budget exceeded, process failed.  Events are delivered to configurable
sinks (webhooks, callbacks, logs).  Every sink has its own queue and
delivery task, so a slow or failing sink never delays the others.

Example::

    from promptise import build_agent, EventNotifier, WebhookSink, CallbackSink

    notifier = EventNotifier(sinks=[
        WebhookSink(
            url="https://hooks.slack.com/services/...",
            events=["invocation.error", "budget.exceeded"],
        ),
        CallbackSink(lambda event: print(event.event_type)),
    ])

    agent = await build_agent(..., events=notifier)
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import hmac as _hmac_mod
import json
import logging
import re as _re
import secrets
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler

from ._outbound import BlockedTarget, pin_target

logger = logging.getLogger("promptise.events")

__all__ = [
    "AgentEvent",
    "EventSink",
    "EventNotifier",
    "WebhookSink",
    "CallbackSink",
    "LogSink",
    "EventBusSink",
    "default_pii_sanitizer",
    "verify_event_signature",
]


# ---------------------------------------------------------------------------
# PII / credential redaction (shared by all sinks + observability)
# ---------------------------------------------------------------------------

_PII_PATTERNS: list[tuple[_re.Pattern[str], str]] = [
    # URL credentials first: the email pattern would otherwise consume
    # ``password@host`` and leave the user name behind.  User info never
    # contains whitespace, quotes, ``/`` or (unencoded) ``@``, so a match
    # cannot run on into the next field of a serialised payload.
    (_re.compile(r"://[^\s:/@\"']+:[^\s/@\"']+@"), "://[REDACTED]@"),
    (_re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[CARD]"),
    (_re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (_re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    # OpenAI (``sk-proj-…``, ``sk-svcacct-…``), Anthropic (``sk-ant-…``) and
    # legacy ``sk-…`` keys; the newer formats contain ``-`` and ``_``.
    (_re.compile(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9][A-Za-z0-9_-]{19,}"), "[API_KEY]"),
    (_re.compile(r"\b(AKIA[A-Z0-9]{16})\b"), "[AWS_KEY]"),
    (_re.compile(r"\b(ghp_[a-zA-Z0-9]{36})\b"), "[GITHUB_TOKEN]"),
    (_re.compile(r"Bearer\s+[A-Za-z0-9\-._~+/]+=*"), "Bearer [REDACTED]"),
]


def default_pii_sanitizer(data: dict[str, Any]) -> dict[str, Any]:
    """Redact PII and credentials from a data dictionary.

    Serialises the dict to JSON, applies regex patterns, and
    deserialises back.  Safe to call on any dict — returns the
    original on serialization failure.

    Args:
        data: Arbitrary dict (event payload, observability metadata, etc.).

    Returns:
        A new dict with sensitive values replaced by placeholders.
    """
    try:
        text = json.dumps(data, default=str)
        for pattern, replacement in _PII_PATTERNS:
            text = pattern.sub(replacement, text)
        return json.loads(text)
    except Exception:
        return data


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class AgentEvent:
    """A structured notification event from the Promptise framework.

    Attributes:
        event_type: Dotted event name (e.g. ``"invocation.complete"``).
        severity: One of ``"info"``, ``"warning"``, ``"error"``, ``"critical"``.
        timestamp: When the event occurred (``time.time()``).
        agent_id: Agent or process identifier.
        user_id: User who triggered the action (from CallerContext).
        session_id: Conversation session ID if applicable.
        data: Event-specific payload (tool name, error message, etc.).
        metadata: Agent configuration, model ID, etc.
    """

    event_type: str
    severity: str = "info"
    timestamp: float = field(default_factory=time.time)
    agent_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "event_type": self.event_type,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "agent_id": self.agent_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "data": self.data,
            "metadata": self.metadata,
        }

    def compute_hmac(self, secret: str) -> str:
        """Compute an HMAC-SHA256 over this event's sorted-key JSON.

        This is *not* what :class:`WebhookSink` sends: webhook deliveries
        are signed over the exact body bytes plus a timestamp.  Verify
        those with :func:`verify_event_signature`.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, default=str)
        return _hmac_mod.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


# ---------------------------------------------------------------------------
# Severity levels
# ---------------------------------------------------------------------------

SEVERITY_ORDER = {"info": 0, "warning": 1, "error": 2, "critical": 3}


# ---------------------------------------------------------------------------
# Sink protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EventSink(Protocol):
    """Protocol for event notification sinks.

    Sinks receive events from the :class:`EventNotifier` and deliver
    them to external systems (webhooks, logs, callbacks, etc.).
    """

    async def emit(self, event: AgentEvent) -> None:
        """Deliver a single event."""
        ...


# ---------------------------------------------------------------------------
# Webhook signatures
# ---------------------------------------------------------------------------

SIGNATURE_HEADER = "X-Promptise-Signature"
TIMESTAMP_HEADER = "X-Promptise-Timestamp"
DEFAULT_SIGNATURE_TOLERANCE = 300.0


def _sign(body: bytes, secret: str, timestamp: int) -> str:
    """HMAC-SHA256 over ``b"<timestamp>." + body`` (hex digest)."""
    signed = str(timestamp).encode() + b"." + body
    return _hmac_mod.new(secret.encode(), signed, hashlib.sha256).hexdigest()


def verify_event_signature(
    body: bytes | str,
    signature: str | None,
    secret: str | Iterable[str],
    *,
    tolerance: float | None = DEFAULT_SIGNATURE_TOLERANCE,
    now: float | None = None,
) -> bool:
    """Check the ``X-Promptise-Signature`` header of a webhook delivery.

    :class:`WebhookSink` signs every request Stripe-style: the header is
    ``t=<unix seconds>,v1=<hex>``, where ``<hex>`` is the HMAC-SHA256 of
    ``b"<t>." + body`` keyed with the sink's ``secret``.  Pass the **raw**
    request body, exactly as received — not re-serialised JSON.

    Args:
        body: The raw request body bytes (a ``str`` is UTF-8 encoded).
        signature: The ``X-Promptise-Signature`` header value.
        secret: The sink's secret, or several secrets while rotating.
        tolerance: Maximum age (and clock skew) in seconds before a
            signature is rejected as a possible replay.  ``None`` or ``0``
            disables the check.
        now: The current time (``time.time()`` when omitted); for tests.

    Returns:
        ``True`` only when the signature matches and is fresh enough.
    """
    if not signature:
        return False
    raw = body.encode() if isinstance(body, str) else bytes(body)
    secrets_ = [secret] if isinstance(secret, str) else list(secret)

    timestamp: int | None = None
    candidates: list[str] = []
    for part in signature.split(","):
        key, sep, value = part.strip().partition("=")
        if not sep:
            continue
        if key == "t":
            try:
                timestamp = int(value)
            except ValueError:
                return False
        elif key == "v1":
            candidates.append(value)
    if timestamp is None or not candidates:
        return False

    if tolerance:
        current = time.time() if now is None else now
        if abs(current - timestamp) > tolerance:
            return False

    for key_ in secrets_:
        if not key_:
            continue
        expected = _sign(raw, key_, timestamp)
        if any(_hmac_mod.compare_digest(expected, c) for c in candidates):
            return True
    return False


# ---------------------------------------------------------------------------
# Built-in sinks
# ---------------------------------------------------------------------------


_PRIVATE_NETWORK_HINT = (
    "Pass WebhookSink(..., allow_private_networks=True) to deliver to "
    "localhost or a private network."
)


class WebhookSink:
    """Deliver events via HTTP POST to a webhook URL.

    Features: HMAC-SHA256 signing with a timestamp (replay protection),
    retry with exponential backoff, SSRF protection, per-event filtering,
    payload redaction.

    Every request carries:

    - ``X-Promptise-Signature: t=<unix seconds>,v1=<hex>`` — HMAC-SHA256
      over ``b"<t>." + body`` with ``secret``.  Check it with
      :func:`verify_event_signature` against the raw body.
    - ``X-Promptise-Timestamp`` — the same ``t``, for convenience.
    - ``X-Promptise-Event`` — the event type.
    - ``X-Promptise-Delivery`` — an id that stays the same across retries
      of one event, so receivers can de-duplicate.

    Args:
        url: Webhook URL to POST events to.
        events: Event types to subscribe to (``None`` = all events).
        headers: Custom HTTP headers (e.g. auth tokens).
        secret: HMAC secret for signing payloads.  If not provided,
            a random secret is generated (readable as :attr:`secret`).
        max_retries: Maximum retry attempts on failure.
        retry_delay: Initial retry delay in seconds (doubles each retry).
        redact_sensitive: Scan the whole payload (``data``, ``user_id``,
            ``session_id``, ``metadata`` …) for PII/credentials before
            sending.
        min_severity: Minimum severity level to emit.
        transform: Reshape the (redacted) payload before it is sent,
            e.g. into a Slack or PagerDuty message.
        allow_private_networks: Allow ``localhost``, loopback and private
            IP ranges.  Off by default as SSRF protection; turn it on for
            a receiver on your own machine or private network.
    """

    def __init__(
        self,
        url: str,
        *,
        events: list[str] | None = None,
        headers: dict[str, str] | None = None,
        secret: str | None = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        redact_sensitive: bool = True,
        min_severity: str | None = None,
        transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        allow_private_networks: bool = False,
    ) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ValueError(f"WebhookSink needs an http(s) URL, got {url!r}")
        if not allow_private_networks:
            # SSRF protection.  Checked again on every delivery, against the
            # address actually connected to (see promptise._outbound).
            from promptise.mcp.server._openapi import _validate_url_not_private

            _validate_url_not_private(url, hint=_PRIVATE_NETWORK_HINT)

        self._url = url
        self._events = set(events) if events else None
        self._headers = headers or {}
        self._secret = secret or secrets.token_hex(32)
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._redact_sensitive = redact_sensitive
        self._min_severity = min_severity
        self._transform = transform
        self._allow_private_networks = allow_private_networks
        self._client: Any = None  # Lazy httpx.AsyncClient

    @property
    def secret(self) -> str:
        """The signing secret (generated when none was passed)."""
        return self._secret

    def _should_emit(self, event: AgentEvent) -> bool:
        """Check if this sink should process the event."""
        if self._events and event.event_type not in self._events:
            return False
        if self._min_severity:
            event_level = SEVERITY_ORDER.get(event.severity, 0)
            min_level = SEVERITY_ORDER.get(self._min_severity, 0)
            if event_level < min_level:
                return False
        return True

    def _redact_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Redact sensitive data from the whole payload before sending.

        Delegates to :func:`default_pii_sanitizer`, so ``user_id``,
        ``session_id``, ``agent_id`` and ``metadata`` are covered as well
        as ``data``.
        """
        if not self._redact_sensitive:
            return payload
        return default_pii_sanitizer(dict(payload))

    async def emit(self, event: AgentEvent) -> None:
        """POST the event to the webhook URL with retries."""
        if not self._should_emit(event):
            return

        try:
            import httpx
        except ImportError:
            logger.warning("httpx not installed — WebhookSink cannot deliver events")
            return

        payload = event.to_dict()
        payload = self._redact_payload(payload)

        # Apply custom transform (e.g., PagerDuty/Slack format)
        if self._transform is not None:
            try:
                payload = self._transform(payload)
            except Exception as exc:
                logger.warning("WebhookSink: transform failed: %s", exc)
                return

        # Serialise once: the signature covers these exact bytes, which are
        # also exactly what goes on the wire.
        body = json.dumps(payload, separators=(",", ":"), default=str).encode()
        delivery_id = secrets.token_hex(16)

        delay = self._retry_delay
        if self._client is None:
            # Never follow redirects: a 3xx could point at an internal host.
            self._client = httpx.AsyncClient(timeout=10, follow_redirects=False)
        client = self._client
        for attempt in range(self._max_retries + 1):
            # Re-sign each attempt so a retry after a long backoff is not
            # rejected as stale by the receiver's tolerance window.
            timestamp = int(time.time())
            headers = {
                "Content-Type": "application/json",
                SIGNATURE_HEADER: f"t={timestamp},v1={_sign(body, self._secret, timestamp)}",
                TIMESTAMP_HEADER: str(timestamp),
                "X-Promptise-Event": event.event_type,
                "X-Promptise-Delivery": delivery_id,
                **self._headers,
            }
            try:
                # Resolved and checked on every attempt (DNS rebinding).
                target = await pin_target(
                    self._url, allow_private_networks=self._allow_private_networks
                )
                resp = await client.post(
                    target.url,
                    content=body,
                    headers={**headers, **target.headers},
                    extensions=target.extensions,
                    follow_redirects=False,
                )
                resp.raise_for_status()
                return  # Success
            except BlockedTarget as exc:
                logger.warning(
                    "WebhookSink: not delivering %s: %s. %s",
                    event.event_type,
                    exc,
                    _PRIVATE_NETWORK_HINT,
                )
                return
            except Exception as exc:
                if attempt < self._max_retries:
                    logger.debug(
                        "WebhookSink: attempt %d failed for %s: %s, retrying in %.1fs",
                        attempt + 1,
                        event.event_type,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    logger.warning(
                        "WebhookSink: failed to deliver %s after %d attempts: %s",
                        event.event_type,
                        self._max_retries + 1,
                        exc,
                    )

    async def close(self) -> None:
        """Release the persistent HTTP connection pool."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class CallbackSink:
    """Deliver events to a Python callable.

    Args:
        callback: Async or sync callable that receives an :class:`AgentEvent`.
        events: Event types to subscribe to (``None`` = all events).
        min_severity: Minimum severity level to emit.
    """

    def __init__(
        self,
        callback: Callable[..., Any],
        *,
        events: list[str] | None = None,
        min_severity: str | None = None,
    ) -> None:
        self._callback = callback
        self._events = set(events) if events else None
        self._min_severity = min_severity

    async def emit(self, event: AgentEvent) -> None:
        """Call the callback with the event."""
        if self._events and event.event_type not in self._events:
            return
        if self._min_severity:
            if SEVERITY_ORDER.get(event.severity, 0) < SEVERITY_ORDER.get(self._min_severity, 0):
                return

        try:
            result = self._callback(event)
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                await result
        except Exception as exc:
            logger.warning("CallbackSink: handler error: %s", exc)


class LogSink:
    """Deliver events to Python's logging system.

    Args:
        events: Event types to subscribe to (``None`` = all events).
        logger_name: Logger name (default: ``"promptise.events"``).
        min_severity: Minimum severity level to emit.
    """

    _SEVERITY_TO_LEVEL = {
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL,
    }

    def __init__(
        self,
        *,
        events: list[str] | None = None,
        logger_name: str = "promptise.events",
        min_severity: str | None = None,
    ) -> None:
        self._events = set(events) if events else None
        self._logger = logging.getLogger(logger_name)
        self._min_severity = min_severity

    async def emit(self, event: AgentEvent) -> None:
        """Log the event as a structured JSON line."""
        if self._events and event.event_type not in self._events:
            return
        if self._min_severity:
            if SEVERITY_ORDER.get(event.severity, 0) < SEVERITY_ORDER.get(self._min_severity, 0):
                return

        level = self._SEVERITY_TO_LEVEL.get(event.severity, logging.INFO)
        self._logger.log(
            level,
            "%s [%s] %s",
            event.event_type,
            event.severity,
            json.dumps(event.data, default=str),
        )


class EventBusSink:
    """Bridge events to the runtime's EventBus for inter-process notifications.

    Args:
        event_bus: Any object with an ``emit(event_type, data)`` method.
        events: Event types to subscribe to (``None`` = all events).
    """

    def __init__(
        self,
        event_bus: Any,
        *,
        events: list[str] | None = None,
    ) -> None:
        self._bus = event_bus
        self._events = set(events) if events else None

    async def emit(self, event: AgentEvent) -> None:
        """Publish the event to the EventBus."""
        if self._events and event.event_type not in self._events:
            return

        try:
            emit_fn = getattr(self._bus, "emit", None)
            if emit_fn is None:
                return
            result = emit_fn(event.event_type, event.to_dict())
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                await result
        except Exception as exc:
            logger.warning("EventBusSink: delivery error: %s", exc)


# ---------------------------------------------------------------------------
# EventNotifier — the central coordinator
# ---------------------------------------------------------------------------


def _running_loop() -> asyncio.AbstractEventLoop | None:
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


class EventNotifier:
    """Central event coordinator that routes events to configured sinks.

    Each sink gets its own queue and background delivery task, so a slow
    or retrying sink (a webhook whose receiver is down) never delays the
    other sinks, and a failing sink never affects them.  Within one sink,
    events are delivered in order.  The agent never blocks waiting for
    delivery.

    Args:
        sinks: List of :class:`EventSink` implementations.
        max_queue_size: Maximum undelivered events per sink.  When a
            sink's queue is full, new events are dropped *for that sink*
            and counted in :attr:`dropped_count`, with a warning log.
        shutdown_timeout: Seconds :meth:`stop` waits for queued events to
            be delivered before giving up.  Events still undelivered then
            are dropped and logged (per sink, with their types).
        slow_tool_threshold: Seconds after which a tool call emits a
            ``tool.slow`` event.  ``None`` disables ``tool.slow``.

    Example::

        notifier = EventNotifier(sinks=[
            WebhookSink("https://hooks.slack.com/...", events=["invocation.error"]),
            CallbackSink(my_handler),
        ])
        await notifier.start()
        notifier.emit_sync(AgentEvent(event_type="invocation.start", severity="info"))
        await notifier.stop()
    """

    def __init__(
        self,
        sinks: list[EventSink],
        *,
        max_queue_size: int = 1000,
        shutdown_timeout: float = 10.0,
        slow_tool_threshold: float | None = 5.0,
    ) -> None:
        if not sinks:
            raise ValueError("EventNotifier requires at least one sink")
        if shutdown_timeout < 0:
            raise ValueError("shutdown_timeout must be >= 0")
        if slow_tool_threshold is not None and slow_tool_threshold < 0:
            raise ValueError("slow_tool_threshold must be >= 0 or None")
        self._sinks = list(sinks)
        self._max_queue_size = max_queue_size
        self.shutdown_timeout = shutdown_timeout
        self.slow_tool_threshold = slow_tool_threshold
        self._queues: list[asyncio.Queue[AgentEvent]] = [
            asyncio.Queue(maxsize=max_queue_size) for _ in self._sinks
        ]
        self._in_flight: list[AgentEvent | None] = [None] * len(self._sinks)
        self._workers: list[asyncio.Task[None]] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self._started = False
        self._stopping = False
        #: Events dropped so far (full queues and shutdown timeouts).
        self.dropped_count = 0

    @property
    def sinks(self) -> list[EventSink]:
        """The configured sinks."""
        return list(self._sinks)

    @property
    def is_running(self) -> bool:
        """Whether delivery tasks are running."""
        return self._started

    # -- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        """Start the background delivery tasks (idempotent)."""
        self._ensure_started(asyncio.get_running_loop())

    def _ensure_started(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._stopping:
            return
        if self._started and self._loop is not loop:
            owner = self._loop
            if owner is None or owner.is_closed() or not owner.is_running():
                # The loop that ran our tasks is gone (e.g. an earlier
                # asyncio.run()); start over on this one.
                self._started = False
                self._workers = []
        if self._started:
            return
        if self._loop is not loop:
            # Queues bind to the loop they are first awaited on; move any
            # pending events into fresh queues for this loop.
            old = self._queues
            self._queues = [asyncio.Queue(maxsize=self._max_queue_size) for _ in self._sinks]
            for src, dst in zip(old, self._queues, strict=True):
                while True:
                    try:
                        dst.put_nowait(src.get_nowait())
                    except (asyncio.QueueEmpty, asyncio.QueueFull):
                        break
            self._loop = loop
        self._started = True
        self._workers = [
            loop.create_task(self._worker(i), name=f"promptise-events-{type(sink).__name__}-{i}")
            for i, sink in enumerate(self._sinks)
        ]
        logger.info("EventNotifier started with %d sink(s)", len(self._sinks))

    async def flush(self, timeout: float | None = None) -> bool:
        """Wait until every queued event has been delivered.

        Args:
            timeout: Maximum seconds to wait (``None`` = no limit).

        Returns:
            ``True`` when everything was delivered, ``False`` on timeout
            (or when the notifier is not running and events are queued).
        """
        if self._idle():
            return True
        if not self._started:
            return False
        joins = asyncio.gather(*(q.join() for q in self._queues))
        try:
            if timeout is None:
                await joins
            else:
                await asyncio.wait_for(joins, timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def _idle(self) -> bool:
        return all(q.empty() for q in self._queues) and all(e is None for e in self._in_flight)

    async def stop(self, timeout: float | None = None) -> None:
        """Deliver what is queued, then stop the delivery tasks.

        Waits up to ``timeout`` seconds (default: ``shutdown_timeout``).
        Events that are still undelivered after that are dropped, counted
        in :attr:`dropped_count` and logged with their sink and types.
        Sinks with a ``close()`` method (like :class:`WebhookSink`) are
        closed afterwards.  The notifier can be started again.
        """
        if not self._started or self._stopping:
            return
        limit = self.shutdown_timeout if timeout is None else timeout
        self._stopping = True
        try:
            drained = await self.flush(limit)
            if not drained:
                self._drop_undelivered(limit)
            for task in self._workers:
                task.cancel()
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers = []
            for sink in self._sinks:
                close = getattr(sink, "close", None)
                if callable(close):
                    try:
                        result = close()
                        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                            await result
                    except Exception:
                        logger.debug("EventNotifier: closing %s failed", sink, exc_info=True)
        finally:
            self._started = False
            self._stopping = False
        logger.info("EventNotifier stopped")

    def _drop_undelivered(self, limit: float) -> None:
        # Synchronous on purpose: no delivery task runs between the
        # snapshot of in-flight events and their cancellation.
        for i, sink in enumerate(self._sinks):
            lost: list[AgentEvent] = []
            current = self._in_flight[i]
            if current is not None:
                lost.append(current)
            queue = self._queues[i]
            while True:
                try:
                    lost.append(queue.get_nowait())
                    queue.task_done()
                except asyncio.QueueEmpty:
                    break
            if lost:
                self.dropped_count += len(lost)
                types = ", ".join(e.event_type for e in lost[:10])
                if len(lost) > 10:
                    types += ", …"
                logger.warning(
                    "EventNotifier: %s did not finish within the %.1fs shutdown timeout; "
                    "dropped %d event(s): %s",
                    type(sink).__name__,
                    limit,
                    len(lost),
                    types,
                )

    # -- emitting -------------------------------------------------------

    async def emit(self, event: AgentEvent) -> None:
        """Queue an event for delivery (non-blocking, starts the notifier).

        If a sink's queue is full, the event is dropped for that sink
        with a warning log.
        """
        self.emit_sync(event)

    def emit_sync(self, event: AgentEvent) -> None:
        """Queue an event from a synchronous context.

        Safe to call from any thread: from outside the notifier's event
        loop the event is handed over thread-safely.  Inside a running
        loop, the notifier starts itself if needed.  Never raises.
        """
        try:
            loop = _running_loop()
            owner = self._loop
            if (
                self._started
                and owner is not None
                and owner is not loop
                and owner.is_running()
                and not owner.is_closed()
            ):
                owner.call_soon_threadsafe(self._enqueue, event)
                return
            if loop is not None:
                self._ensure_started(loop)
            self._enqueue(event)
        except Exception:
            logger.debug("EventNotifier: could not queue %s", event.event_type, exc_info=True)

    def _enqueue(self, event: AgentEvent) -> None:
        for sink, queue in zip(self._sinks, self._queues, strict=True):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                self.dropped_count += 1
                if self.dropped_count == 1 or self.dropped_count % 100 == 0:
                    logger.warning(
                        "EventNotifier: queue for %s is full (%d), dropping %s "
                        "(%d event(s) dropped so far)",
                        type(sink).__name__,
                        self._max_queue_size,
                        event.event_type,
                        self.dropped_count,
                    )

    async def _worker(self, index: int) -> None:
        """Deliver one sink's queue, in order, isolated from other sinks."""
        sink = self._sinks[index]
        queue = self._queues[index]
        while True:
            event = await queue.get()
            self._in_flight[index] = event
            try:
                await sink.emit(event)
            except Exception as exc:
                logger.warning(
                    "EventNotifier: sink %s failed for %s: %s",
                    type(sink).__name__,
                    event.event_type,
                    exc,
                )
            finally:
                self._in_flight[index] = None
                queue.task_done()


# ---------------------------------------------------------------------------
# Event scope (who/where an event comes from)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _EventScope:
    agent_id: str | None = None
    session_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


_scope_var: contextvars.ContextVar[_EventScope | None] = contextvars.ContextVar(
    "promptise_event_scope", default=None
)


def _push_scope(
    *,
    agent_id: str | None,
    session_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> contextvars.Token[_EventScope | None]:
    """Attribute events emitted in this context to an agent run.

    ``agent_id`` replaces the outer scope's (so a peer agent delegated to
    mid-run is attributed to itself); ``session_id`` is inherited when not
    given; ``metadata`` is merged over the outer scope's.
    """
    outer = _scope_var.get()
    return _scope_var.set(
        _EventScope(
            agent_id=agent_id,
            session_id=session_id or (outer.session_id if outer else None),
            metadata={**(outer.metadata if outer else {}), **(metadata or {})},
        )
    )


@contextmanager
def _event_scope(
    *,
    agent_id: str | None = None,
    session_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Like :func:`_push_scope`, but ``agent_id`` is inherited when not given."""
    outer = _scope_var.get()
    token = _push_scope(
        agent_id=agent_id or (outer.agent_id if outer else None),
        session_id=session_id,
        metadata=metadata,
    )
    try:
        yield
    finally:
        _scope_var.reset(token)


# ---------------------------------------------------------------------------
# Helper for emission points
# ---------------------------------------------------------------------------


def emit_event(
    notifier: EventNotifier | None,
    event_type: str,
    severity: str = "info",
    data: dict[str, Any] | None = None,
    *,
    agent_id: str | None = None,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    user_id: str | None = None,
) -> None:
    """Emit an event from any code path (null-safe, sync-safe).

    Fields not given are filled from context: ``user_id`` from the
    current :class:`CallerContext`; ``agent_id``, ``session_id`` and
    ``metadata`` from the running agent invocation (``session_id`` falls
    back to ``caller.metadata["session_id"]``).  Does nothing if
    ``notifier`` is None.

    Args:
        notifier: The :class:`EventNotifier` instance (or None to no-op).
        event_type: Dotted event name.
        severity: Event severity level.
        data: Event-specific payload.
        agent_id: Agent or process identifier.
        session_id: Conversation session ID.
        metadata: Additional context metadata (merged over the scope's).
        user_id: The user the event concerns.
    """
    if notifier is None:
        return

    caller = None
    try:
        from .agent import get_current_caller

        caller = get_current_caller()
    except Exception:
        pass

    if user_id is None and caller is not None:
        user_id = getattr(caller, "user_id", None)

    scope = _scope_var.get()
    if scope is not None:
        agent_id = agent_id if agent_id is not None else scope.agent_id
        session_id = session_id if session_id is not None else scope.session_id
    if session_id is None and caller is not None:
        caller_meta = getattr(caller, "metadata", None)
        if isinstance(caller_meta, Mapping) and caller_meta.get("session_id") is not None:
            session_id = str(caller_meta["session_id"])

    event = AgentEvent(
        event_type=event_type,
        severity=severity,
        agent_id=agent_id,
        user_id=user_id,
        session_id=session_id,
        data=data or {},
        metadata={**(scope.metadata if scope else {}), **(metadata or {})},
    )
    notifier.emit_sync(event)


# ---------------------------------------------------------------------------
# Tool events (tool.error / tool.slow)
# ---------------------------------------------------------------------------

_MAX_ENVELOPE_BYTES = 64 * 1024


def _error_envelope(text: Any) -> dict[str, Any] | None:
    """Parse a ``{"error": {"code": ..., "message": ...}}`` tool result.

    This is how Promptise MCP servers (``ToolError`` and friends) report a
    failed call, in a result that is otherwise a normal success.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped.startswith("{") or len(stripped) > _MAX_ENVELOPE_BYTES:
        return None
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return None
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if not (isinstance(error, dict) and "code" in error and isinstance(error.get("message"), str)):
        return None
    info: dict[str, Any] = {
        "error": error["message"][:200],
        "error_type": "ToolError",
        "code": error["code"],
    }
    if isinstance(error.get("retryable"), bool):
        info["retryable"] = error["retryable"]
    return info


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            str(c.get("text", "")) if isinstance(c, dict) else str(getattr(c, "text", c))
            for c in content
        ]
        return "\n".join(p for p in parts if p)
    return str(content or "")


def _tool_result_error(output: Any) -> dict[str, Any] | None:
    """Return error details when a tool *returned* (not raised) a failure."""
    if isinstance(output, str):
        return _error_envelope(output)
    if isinstance(output, dict):
        return _error_envelope(json.dumps(output, default=str))
    # A raw MCP CallToolResult, or a ToolMessage from a tool that handled
    # its own exception (status="error").
    if getattr(output, "isError", False) is True or getattr(output, "status", None) == "error":
        text = _content_text(getattr(output, "content", ""))
        return _error_envelope(text) or {"error": text[:200], "error_type": "ToolError"}
    return _error_envelope(_content_text(getattr(output, "content", None)))


class _ToolEventCallback(AsyncCallbackHandler):
    """Emit ``tool.error`` and ``tool.slow`` from LangChain tool callbacks.

    Attached to every invocation of an agent built with ``events=``,
    independent of observability.  It covers tools that raise (with the
    real tool name) and MCP tools whose result is an error.
    """

    run_inline = True

    def __init__(self, notifier: EventNotifier) -> None:
        super().__init__()
        self._notifier = notifier
        self._runs: dict[UUID, tuple[str, float]] = {}

    def _finish(self, run_id: UUID, kwargs: dict[str, Any]) -> tuple[str, float | None]:
        name, started = self._runs.pop(run_id, (kwargs.get("name") or "unknown", None))
        duration_ms = round((time.perf_counter() - started) * 1000, 1) if started else None
        return name, duration_ms

    def _check_slow(self, name: str, duration_ms: float | None) -> None:
        threshold = self._notifier.slow_tool_threshold
        if threshold is None or duration_ms is None or duration_ms <= threshold * 1000:
            return
        emit_event(
            self._notifier,
            "tool.slow",
            "warning",
            {"tool_name": name, "latency_ms": duration_ms, "threshold_ms": threshold * 1000},
        )

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = (serialized or {}).get("name") or kwargs.get("name") or "unknown"
        self._runs[run_id] = (str(name), time.perf_counter())

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        name, duration_ms = self._finish(run_id, kwargs)
        error = _tool_result_error(output)
        if error is not None:
            emit_event(
                self._notifier,
                "tool.error",
                "error",
                {"tool_name": name, **error, "duration_ms": duration_ms},
            )
        self._check_slow(name, duration_ms)

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        name, duration_ms = self._finish(run_id, kwargs)
        if name == "unknown":
            name = str(getattr(error, "tool_name", None) or name)
        message = getattr(error, "message", None)
        data: dict[str, Any] = {
            "tool_name": name,
            "error": (message if isinstance(message, str) else str(error))[:200],
            "error_type": type(error).__name__,
        }
        code = getattr(error, "code", None)
        if isinstance(code, (str, int)) and not isinstance(code, bool):
            data["code"] = code
        retryable = getattr(error, "retryable", None)
        if isinstance(retryable, bool):
            data["retryable"] = retryable
        data["duration_ms"] = duration_ms
        emit_event(self._notifier, "tool.error", "error", data)
        self._check_slow(name, duration_ms)
