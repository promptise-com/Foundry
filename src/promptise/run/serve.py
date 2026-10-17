"""``promptise run --serve``: an agent as a production HTTP API.

Routes:

``GET /healthz``
    ``200 {"status": "ok"}`` while the process is up (liveness probe).
``GET /readyz``
    ``200 {"status": "ready"}`` when the agent is built and the server is not
    draining, otherwise ``503`` (readiness probe).
``GET /v1/info``
    The project's info (name, version, model, ...) and the agent's tool names.
``POST /v1/invoke``
    ``{"input": str | [messages], "session_id"?: str}`` →
    ``{"output", "duration_ms", "usage", "request_id"}``.
``POST /v1/stream``
    Same body; answers ``text/event-stream`` with one ``data: <json>`` line per
    :class:`~promptise.streaming.StreamEvent` dict (``token``, ``tool_start``,
    ``tool_end``, then ``done`` or ``error``, which ends the stream).

Security and operations:

* ``/v1/*`` requires ``Authorization: Bearer <token>`` when a token is set
  (compared in constant time).  :func:`run_serve` refuses to bind a
  non-loopback address without one.
* Without a token on a loopback address, ``/v1/*`` refuses requests whose
  ``Host`` is not a loopback name (``421``, DNS rebinding) or that carry a
  cross-site ``Origin`` (``403``), so a web page open on the same machine
  can't drive the agent.
* At most ``max_concurrency`` turns run at once; more get ``429 {"error":
  "busy"}`` with ``Retry-After``.
* On SIGTERM / SIGINT the server drains: ``/readyz`` and new ``/v1/*``
  requests answer ``503``, in-flight turns finish (up to the shutdown
  timeout), then :func:`run_serve` returns ``0``.
* A failure answers ``500 {"error": "internal error", "error_id": ...}``; the
  exception and its traceback are only logged (with the same ``error_id``),
  never sent to the client.
* Every response carries ``X-Request-ID`` (the client's, when it sent a
  well-formed one).  The request id is also the run events' ``turn_id``.

Run events published (when an :class:`~promptise.run.events.EventBus` is
given): ``http`` for every request, and ``turn.start`` / ``stream`` /
``turn.end`` for every invoke and stream turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import logging
import math
import os
import re
import signal
import sys
import threading
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterator
from types import FrameType
from typing import TYPE_CHECKING, Any, Protocol

import uvicorn
from langchain_core.callbacks import BaseCallbackHandler
from starlette.applications import Starlette
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect, Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ..runtime._http_guard import bearer_token_matches, is_loopback, loopback_request_problem
from .events import EventBus
from .manifest import ServeSettings

if TYPE_CHECKING:
    from langchain_core.outputs import LLMResult

    from ..agent import CallerContext, PromptiseAgent

__all__ = [
    "MAX_BODY_BYTES",
    "AgentHost",
    "HostError",
    "PromptiseAgentHost",
    "create_app",
    "run_serve",
]

logger = logging.getLogger("promptise.run.serve")

MAX_BODY_BYTES = 8 * 1024 * 1024
"""Largest request body ``/v1/invoke`` and ``/v1/stream`` accept (``413`` above)."""

SSE_PING_SECONDS = 15.0
"""Seconds without a stream event after which an SSE comment (``: ping``) is
sent, so proxies don't close a stream that waits on a slow tool."""

_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")
_SESSION_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,256}")
_ROLES = ("system", "user", "assistant")
_MESSAGE_KEYS = frozenset({"role", "content", "name"})
_BODY_KEYS = frozenset({"input", "session_id"})
_TERMINAL = frozenset({"done", "error"})
_STREAM_FINISH_SECONDS = 5.0  # how long a stream may run on after its final event

TurnInput = str | list[dict[str, Any]]
"""A turn's input: one user message, or a list of ``{"role", "content"}`` messages."""


# ---------------------------------------------------------------------------
# The agent behind the API
# ---------------------------------------------------------------------------


class AgentHost(Protocol):
    """What :func:`create_app` serves: an agent that answers turns.

    :class:`PromptiseAgentHost` adapts a :class:`~promptise.agent.PromptiseAgent`.
    A host may also have a ``tool_names`` attribute (a list of names), which
    ``/v1/info`` reports.
    """

    @property
    def ready(self) -> bool:
        """Whether the host can answer turns (``/readyz`` answers ``503`` while not)."""
        ...

    async def invoke(
        self, input: TurnInput, *, session_id: str | None, request_id: str
    ) -> dict[str, Any]:
        """Answer one turn: ``{"output": str, "usage": {...}}``.

        Raise :class:`HostError` for a failure the client may be told about;
        any other exception answers a generic ``500``.
        """
        ...

    def stream(
        self, input: TurnInput, *, session_id: str | None, request_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Answer one turn as :meth:`~promptise.streaming.StreamEvent.to_dict` dicts,
        ending with a ``done`` or ``error`` event."""
        ...


class HostError(Exception):
    """A turn failure whose status and message may be sent to the client.

    Args:
        status: The HTTP status to answer (e.g. ``403``, ``422``).
        message: The client-facing message.  It must not contain internal
            details; the cause is logged, not sent.
    """

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class _UsageCounter(BaseCallbackHandler):
    """Sums the token usage of every model call made in one request."""

    run_inline = True  # count in the event loop thread, in call order

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        prompt = completion = total = 0
        usage = (response.llm_output or {}).get("token_usage") or {}
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens") or 0)
            completion = int(usage.get("completion_tokens") or 0)
            total = int(usage.get("total_tokens") or 0)
        for generations in response.generations:
            for generation in generations:
                meta = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if isinstance(meta, dict):
                    prompt = prompt or int(meta.get("input_tokens") or 0)
                    completion = completion or int(meta.get("output_tokens") or 0)
                    total = total or int(meta.get("total_tokens") or 0)
        with self._lock:
            self.llm_calls += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += total or (prompt + completion)

    def as_dict(self) -> dict[str, int]:
        """``{"llm_calls", "prompt_tokens", "completion_tokens", "total_tokens"}``."""
        with self._lock:
            return {
                "llm_calls": self.llm_calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            }


# The usage counter of the request running in this context.  A LangChain
# configure hook adds it to every callback manager created while it is set, so
# each model call of the request (including those made by ``chat()``, which
# takes no ``config``) is counted, and concurrent requests never mix.
_usage_var: contextvars.ContextVar[_UsageCounter | None] = contextvars.ContextVar(
    "promptise_run_serve_usage", default=None
)
_hook_lock = threading.Lock()
_hook_installed = False


def _install_usage_hook() -> None:
    global _hook_installed
    with _hook_lock:
        if _hook_installed:
            return
        from langchain_core.tracers.context import register_configure_hook

        register_configure_hook(_usage_var, inheritable=True)
        _hook_installed = True


@contextlib.contextmanager
def _counting_usage() -> Iterator[_UsageCounter]:
    counter = _UsageCounter()
    token = _usage_var.set(counter)
    try:
        yield counter
    finally:
        try:
            _usage_var.reset(token)
        except ValueError:  # closed from another context (a finalised generator)
            _usage_var.set(None)


def _agent_input(input: TurnInput) -> dict[str, Any]:
    if isinstance(input, str):
        return {"messages": [{"role": "user", "content": input}]}
    return {"messages": [dict(m) for m in input]}


class PromptiseAgentHost:
    """Serves a :class:`~promptise.agent.PromptiseAgent` through :func:`create_app`.

    * A string input is one user message; a list is passed as the messages.
    * ``session_id`` with a string input, on an agent built with a
      ``conversation_store``, goes through :meth:`PromptiseAgent.chat`, so the
      history is loaded and saved per session.  Otherwise it is passed to
      ``ainvoke(session_id=...)`` (it scopes conversation-flow state) and
      nothing is persisted: with a message list the client owns the history.
    * ``usage`` counts the tokens of every model call the turn made, as the
      model reported them (zeros when it reports none).
    * Streaming a persisted session runs :meth:`PromptiseAgent.chat` (the
      streaming API has no session persistence), so it yields the whole
      answer as one ``token`` event, then ``done``.

    Each turn runs with a :class:`~promptise.agent.CallerContext` whose
    ``metadata`` holds ``request_id`` and ``transport="http"``.

    Args:
        agent: The agent to serve.
    """

    def __init__(self, agent: PromptiseAgent) -> None:
        self.agent = agent
        self.ready = True
        _install_usage_hook()

    @property
    def tool_names(self) -> list[str]:
        """The names of the agent's tools."""
        return list(self.agent.tool_names)

    @property
    def persists_sessions(self) -> bool:
        """Whether the agent has a conversation store (``session_id`` keeps history)."""
        return getattr(self.agent, "_conversation_store", None) is not None

    def _caller(self, request_id: str) -> CallerContext:
        from ..agent import CallerContext

        return CallerContext(metadata={"request_id": request_id, "transport": "http"})

    def _use_chat(self, input: TurnInput, session_id: str | None) -> bool:
        return isinstance(input, str) and session_id is not None and self.persists_sessions

    async def invoke(
        self, input: TurnInput, *, session_id: str | None, request_id: str
    ) -> dict[str, Any]:
        """Answer one turn: ``{"output": str, "usage": {...}}``.

        Raises:
            HostError: ``422`` when a guardrail blocked the input or output,
                ``403`` when the session belongs to another user.
        """
        from ..agent import _extract_response_text

        caller = self._caller(request_id)
        with _counting_usage() as usage, _client_errors():
            if self._use_chat(input, session_id):
                assert isinstance(input, str) and session_id is not None
                output = await self.agent.chat(input, session_id=session_id, caller=caller)
            else:
                result = await self.agent.ainvoke(
                    _agent_input(input), caller=caller, session_id=session_id
                )
                output = _extract_response_text(result)
        return {"output": output, "usage": usage.as_dict()}

    async def stream(
        self, input: TurnInput, *, session_id: str | None, request_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        """Answer one turn as stream event dicts; the ``done`` event also has ``usage``."""
        from ..streaming import DoneEvent, ErrorEvent, TokenEvent

        caller = self._caller(request_id)
        with _counting_usage() as usage:
            if self._use_chat(input, session_id):
                assert isinstance(input, str) and session_id is not None
                start = time.perf_counter()
                try:
                    with _client_errors():
                        text = await self.agent.chat(input, session_id=session_id, caller=caller)
                except HostError as exc:
                    yield ErrorEvent(message=exc.message).to_dict()
                    return
                yield TokenEvent(text=text, cumulative_text=text).to_dict()
                done = DoneEvent(
                    full_response=text,
                    duration_ms=round((time.perf_counter() - start) * 1000, 1),
                ).to_dict()
                yield {**done, "usage": usage.as_dict()}
                return
            events = self.agent.astream_with_tools(_agent_input(input), caller=caller)
            try:
                async for event in events:
                    data = event.to_dict()
                    if data.get("type") == "done":
                        data["usage"] = usage.as_dict()
                    yield data
            finally:
                # Close the agent's stream in this task: it resets context
                # variables it set, which only works in the same context.
                aclose = getattr(events, "aclose", None)
                if aclose is not None:
                    await aclose()


@contextlib.contextmanager
def _client_errors() -> Iterator[None]:
    """Turn the agent's client-facing failures into :class:`HostError`."""
    from ..conversations import SessionAccessDenied
    from ..guardrails import GuardrailViolation

    try:
        yield
    except GuardrailViolation as exc:
        logger.info("Turn blocked by a guardrail (%s)", exc.direction)
        raise HostError(422, "blocked by safety policy") from exc
    except SessionAccessDenied as exc:
        raise HostError(403, "session access denied") from exc


# ---------------------------------------------------------------------------
# The ASGI app
# ---------------------------------------------------------------------------


class _JSON(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(content, ensure_ascii=False, default=str, separators=(",", ":")).encode(
            "utf-8"
        )


def _error(status: int, message: str, **extra: Any) -> Response:
    headers = extra.pop("headers", None)
    return _JSON({"error": message, **extra}, status_code=status, headers=headers)


class _RequestError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def _new_error_id() -> str:
    return uuid.uuid4().hex[:16]


def _redacted(exc: BaseException) -> str:
    """``Type: message`` with credentials and PII scrubbed, for run events."""
    text = f"{type(exc).__name__}: {exc}"[:500]
    try:
        from ..observability import redact_sensitive

        return str(redact_sensitive({"error": text})["error"])
    except Exception:
        return type(exc).__name__


class _ServeState:
    def __init__(
        self,
        host: AgentHost,
        *,
        token: str | None,
        max_concurrency: int,
        bus: EventBus | None,
        info: dict[str, Any],
        draining: Callable[[], bool] | None,
        local_only: bool,
        max_body_bytes: int,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if token is not None and not token:
            raise ValueError("token must be a non-empty string (or None)")
        self.host = host
        self.token = token
        self.max_concurrency = max_concurrency
        self.bus = bus
        self.info = info
        self._draining = draining
        self.local_only = local_only
        self.max_body_bytes = max_body_bytes
        self.active = 0

    def is_draining(self) -> bool:
        return bool(self._draining is not None and self._draining())

    def try_acquire(self) -> bool:
        # Handlers run on one event loop and nothing awaits in between, so a
        # plain counter is an atomic try-acquire.
        if self.active >= self.max_concurrency:
            return False
        self.active += 1
        return True

    def release(self) -> None:
        self.active = max(0, self.active - 1)

    def publish(self, kind: str, **payload: Any) -> None:
        if self.bus is None:
            return
        try:
            self.bus.publish(kind, **payload)
        except Exception:
            logger.debug("Could not publish a %s run event", kind, exc_info=True)


class _ServeMiddleware:
    """Request ids, ``http`` events, and the ``/v1/*`` checks (host, auth, draining)."""

    def __init__(self, app: ASGIApp, state: _ServeState) -> None:
        self.app = app
        self.state = state

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        start = time.perf_counter()
        headers = Headers(scope=scope)
        incoming = headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID_RE.fullmatch(incoming) else uuid.uuid4().hex
        scope["promptise.request_id"] = request_id
        status = 0

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message["status"])
                MutableHeaders(scope=message)["X-Request-ID"] = request_id
            await send(message)

        try:
            refusal = self._check(scope, headers)
            if refusal is not None:
                await refusal(scope, receive, send_with_id)
            else:
                await self.app(scope, receive, send_with_id)
        except ClientDisconnect:
            if not status:
                status = 499  # the client went away before an answer
        except Exception as exc:
            error_id = _new_error_id()
            logger.error(
                "Unhandled error in %s %s (request_id=%s, error_id=%s)",
                scope.get("method"),
                scope.get("path"),
                request_id,
                error_id,
                exc_info=exc,
            )
            if status:
                raise  # the answer has started: let the server close the connection
            await _error(500, "internal error", error_id=error_id)(scope, receive, send_with_id)
        finally:
            duration_ms = round((time.perf_counter() - start) * 1000, 1)
            self.state.publish(
                "http",
                method=scope.get("method", ""),
                path=scope.get("path", ""),
                status=status or 500,
                duration_ms=duration_ms,
                request_id=request_id,
            )
            logger.debug(
                "%s %s %s %.1fms (request_id=%s)",
                scope.get("method"),
                scope.get("path"),
                status,
                duration_ms,
                request_id,
            )

    def _check(self, scope: Scope, headers: Headers) -> Response | None:
        path: str = scope.get("path", "")
        if path != "/v1" and not path.startswith("/v1/"):
            return None
        state = self.state
        if state.local_only:
            problem = loopback_request_problem(headers.get("host", ""), headers.get("origin"))
            if problem is not None:
                code, message = problem
                return _error(code, message.lower())
        if state.token is not None and not bearer_token_matches(
            headers.get("authorization", ""), state.token
        ):
            return _error(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})
        if state.is_draining():
            return _error(503, "draining", headers={"Connection": "close", "Retry-After": "1"})
        return None


async def _read_json(request: Request, limit: int) -> dict[str, Any]:
    length = request.headers.get("content-length")
    if length is not None:
        try:
            declared = int(length)
        except ValueError:
            raise _RequestError(400, "invalid Content-Length header") from None
        if declared > limit:
            raise _RequestError(413, f"request body is larger than {limit} bytes")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise _RequestError(413, f"request body is larger than {limit} bytes")
    if not body.strip():
        raise _RequestError(400, "request body must be a JSON object")
    try:
        data = json.loads(bytes(body))
    except ValueError:
        raise _RequestError(400, "request body is not valid JSON") from None
    if not isinstance(data, dict):
        raise _RequestError(400, "request body must be a JSON object")
    return data


def _check_message(index: int, message: Any) -> None:
    where = f"input[{index}]"
    if not isinstance(message, dict):
        raise _RequestError(400, f"{where} must be an object with 'role' and 'content'")
    unknown = sorted(str(k)[:64] for k in set(message) - _MESSAGE_KEYS)
    if unknown:
        raise _RequestError(400, f"{where} has unknown field(s): {', '.join(unknown[:5])}")
    if message.get("role") not in _ROLES:
        raise _RequestError(400, f"{where}.role must be one of: {', '.join(_ROLES)}")
    content = message.get("content")
    if isinstance(content, list):
        if not all(isinstance(block, dict) for block in content):
            raise _RequestError(400, f"{where}.content blocks must be objects")
    elif not isinstance(content, str):
        raise _RequestError(400, f"{where}.content must be a string or a list of content blocks")
    if "name" in message and not isinstance(message["name"], str):
        raise _RequestError(400, f"{where}.name must be a string")


def _parse_turn(data: dict[str, Any]) -> tuple[TurnInput, str | None]:
    unknown = sorted(str(k)[:64] for k in set(data) - _BODY_KEYS)
    if unknown:
        raise _RequestError(400, f"unknown field(s): {', '.join(unknown[:5])}")
    if "input" not in data:
        raise _RequestError(400, "'input' is required")
    value = data["input"]
    turn_input: TurnInput
    if isinstance(value, str):
        if not value.strip():
            raise _RequestError(400, "'input' must not be empty")
        turn_input = value
    elif isinstance(value, list):
        if not value:
            raise _RequestError(400, "'input' must not be an empty list")
        for index, message in enumerate(value):
            _check_message(index, message)
        turn_input = value
    else:
        raise _RequestError(400, "'input' must be a string or a list of messages")
    session_id = data.get("session_id")
    if session_id is not None and not (
        isinstance(session_id, str) and _SESSION_ID_RE.fullmatch(session_id)
    ):
        raise _RequestError(
            400,
            "'session_id' must be a string of 1-256 letters, digits, '.', '_', ':' or '-'",
        )
    return turn_input, session_id


def _sse(event: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n".encode()


class _StreamTurn:
    """One ``/v1/stream`` turn: runs the host's stream in its own task and
    relays its events as SSE, with pings while it is quiet.

    The host's stream runs in one task from start to end (and is closed
    there), because agents set and reset context variables across its steps.
    :meth:`close` runs exactly once, after the response, whatever happened.
    """

    def __init__(
        self,
        state: _ServeState,
        turn_input: TurnInput,
        session_id: str | None,
        request_id: str,
        ping_seconds: float,
    ) -> None:
        self.state = state
        self.input = turn_input
        self.session_id = session_id
        self.request_id = request_id
        self.ping_seconds = ping_seconds
        self.start = time.perf_counter()
        self.output = ""
        self.error: str | None = "stream ended before a final event"
        self._queue: asyncio.Queue[tuple[str, Any]] = asyncio.Queue(maxsize=64)
        self._producer: asyncio.Task[None] | None = None
        self._answered = False  # a done/error event was relayed
        self._closed = False

    async def _produce(self) -> None:
        stream = self.state.host.stream(
            self.input, session_id=self.session_id, request_id=self.request_id
        )
        terminal = False
        try:
            async for event in stream:
                # After done/error the stream is only finishing its own
                # bookkeeping: let it, but relay nothing more.
                if not terminal:
                    await self._queue.put(("event", event))
                    terminal = isinstance(event, dict) and event.get("type") in _TERMINAL
            if not terminal:
                await self._queue.put(("end", None))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not terminal:
                await self._queue.put(("raise", exc))
            else:
                logger.warning(
                    "Stream failed after its final event (request_id=%s)",
                    self.request_id,
                    exc_info=exc,
                )
        finally:
            aclose = getattr(stream, "aclose", None)
            if aclose is not None:
                with contextlib.suppress(Exception):
                    await aclose()

    async def events(self) -> AsyncGenerator[bytes, None]:
        self._producer = producer = asyncio.create_task(self._produce())
        while True:
            try:
                kind, item = await asyncio.wait_for(self._queue.get(), self.ping_seconds)
            except asyncio.TimeoutError:
                if producer.done() and self._queue.empty():
                    kind, item = "end", None  # the producer died without a word
                else:
                    yield b": ping\n\n"
                    continue
            if kind == "event":
                event = item if isinstance(item, dict) else {"type": "unknown", "value": item}
                etype = event.get("type")
                if etype == "done":
                    self.output = str(event.get("full_response") or "")
                    self.error = None
                    self._answered = True
                elif etype == "error":
                    self.error = str(event.get("message") or "error")
                    self._answered = True
                self.state.publish("stream", turn_id=self.request_id, event=event)
                yield _sse(event)
                if self._answered:
                    return
                continue
            if kind == "raise":
                error_id = _new_error_id()
                logger.error(
                    "Stream failed (request_id=%s, error_id=%s)",
                    self.request_id,
                    error_id,
                    exc_info=item,
                )
                self.error = f"{_redacted(item)} (error_id={error_id})"
                event = {
                    "type": "error",
                    "message": "An error occurred during processing.",
                    "recoverable": False,
                    "error_id": error_id,
                }
            else:  # the stream ended without done/error: still end with one
                logger.warning(
                    "Stream ended without a final event (request_id=%s)", self.request_id
                )
                event = {"type": "error", "message": self.error, "recoverable": False}
            self._answered = True
            self.state.publish("stream", turn_id=self.request_id, event=event)
            yield _sse(event)
            return

    async def close(self) -> None:
        """Stop the host's stream, free the slot and publish ``turn.end`` (once)."""
        if self._closed:
            return
        self._closed = True
        producer = self._producer
        try:
            if producer is not None and not producer.done():
                if self._answered:
                    # Give the stream a moment to finish its own bookkeeping.
                    await asyncio.wait({producer}, timeout=_STREAM_FINISH_SECONDS)
                if not producer.done():
                    producer.cancel()
                await asyncio.gather(producer, return_exceptions=True)
        finally:
            if not self._answered:
                self.error = "cancelled"  # the client went away, or the server stopped
            self.state.release()
            self.state.publish(
                "turn.end",
                turn_id=self.request_id,
                output=self.output,
                duration_ms=round((time.perf_counter() - self.start) * 1000, 1),
                error=self.error,
            )


class _EventStreamResponse(StreamingResponse):
    """An SSE response that closes its turn when it ends, however it ends."""

    def __init__(self, turn: _StreamTurn) -> None:
        self._body = turn.events()
        super().__init__(
            self._body,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        self._turn = turn

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with contextlib.suppress(RuntimeError):
                await self._body.aclose()
            await self._turn.close()


def create_app(
    host: AgentHost,
    *,
    token: str | None,
    max_concurrency: int,
    bus: EventBus | None,
    info: dict[str, Any],
    draining: Callable[[], bool] | None = None,
    local_only: bool = False,
    max_body_bytes: int = MAX_BODY_BYTES,
    ping_seconds: float = SSE_PING_SECONDS,
) -> Starlette:
    """The HTTP API serving *host* (see the module docstring for the routes).

    Args:
        host: The agent to serve.
        token: Bearer token ``/v1/*`` requires, or ``None`` for no auth.
        max_concurrency: Turns answered at once; more get ``429``.
        bus: Where ``http``, ``turn.*`` and ``stream`` run events go (optional).
        info: What ``/v1/info`` reports (project, version, model, ...); the
            host's ``tool_names`` are added as ``tools``.
        draining: Returns ``True`` once shutdown has started: ``/readyz`` and
            new ``/v1/*`` requests then answer ``503``.
        local_only: Refuse ``/v1/*`` requests with a non-loopback ``Host``
            (``421``) or a cross-site ``Origin`` (``403``).  For a server on a
            loopback address without a token.
        max_body_bytes: Largest request body accepted (``413`` above).
        ping_seconds: Quiet seconds after which a stream sends ``: ping``.

    Raises:
        ValueError: ``max_concurrency`` is below 1, or ``token`` is empty.
    """
    state = _ServeState(
        host,
        token=token,
        max_concurrency=max_concurrency,
        bus=bus,
        info=info,
        draining=draining,
        local_only=local_only,
        max_body_bytes=max_body_bytes,
    )

    async def healthz(request: Request) -> Response:
        return _JSON({"status": "ok"})

    async def readyz(request: Request) -> Response:
        if state.is_draining():
            return _JSON({"status": "draining"}, status_code=503)
        if not state.host.ready:
            return _JSON({"status": "not_ready"}, status_code=503)
        return _JSON({"status": "ready"})

    async def v1_info(request: Request) -> Response:
        data = dict(state.info)
        names = getattr(state.host, "tool_names", None)
        if names is not None and "tools" not in data:
            data["tools"] = list(names)
        return _JSON(data)

    async def parse(request: Request) -> tuple[TurnInput, str | None]:
        return _parse_turn(await _read_json(request, state.max_body_bytes))

    async def v1_invoke(request: Request) -> Response:
        request_id: str = request.scope["promptise.request_id"]
        try:
            turn_input, session_id = await parse(request)
        except _RequestError as exc:
            return _error(exc.status, exc.message)
        if not state.try_acquire():
            return _error(429, "busy", headers={"Retry-After": "1"})
        start = time.perf_counter()
        output = ""
        error: str | None = "cancelled"  # unless the turn gets an answer or a failure
        state.publish(
            "turn.start", turn_id=request_id, input=turn_input, conversation_id=session_id
        )
        try:
            try:
                result = await state.host.invoke(
                    turn_input, session_id=session_id, request_id=request_id
                )
                output = str(result.get("output") or "")
                usage = result.get("usage")
            except HostError as exc:
                error = exc.message
                return _error(exc.status, exc.message)
            except Exception as exc:
                error_id = _new_error_id()
                logger.error(
                    "Invoke failed (request_id=%s, error_id=%s)",
                    request_id,
                    error_id,
                    exc_info=exc,
                )
                error = f"{_redacted(exc)} (error_id={error_id})"
                return _error(500, "internal error", error_id=error_id)
            error = None
            return _JSON(
                {
                    "output": output,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 1),
                    "usage": usage if isinstance(usage, dict) else {},
                    "request_id": request_id,
                }
            )
        finally:
            state.release()
            state.publish(
                "turn.end",
                turn_id=request_id,
                output=output,
                duration_ms=round((time.perf_counter() - start) * 1000, 1),
                error=error,
            )

    async def v1_stream(request: Request) -> Response:
        request_id: str = request.scope["promptise.request_id"]
        try:
            turn_input, session_id = await parse(request)
        except _RequestError as exc:
            return _error(exc.status, exc.message)
        if not state.try_acquire():
            return _error(429, "busy", headers={"Retry-After": "1"})
        state.publish(
            "turn.start", turn_id=request_id, input=turn_input, conversation_id=session_id
        )
        return _EventStreamResponse(
            _StreamTurn(state, turn_input, session_id, request_id, ping_seconds)
        )

    async def http_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, HTTPException)
        return _error(exc.status_code, str(exc.detail).lower(), headers=exc.headers)

    from starlette.middleware import Middleware

    return Starlette(
        routes=[
            Route("/healthz", healthz, methods=["GET"]),
            Route("/readyz", readyz, methods=["GET"]),
            Route("/v1/info", v1_info, methods=["GET"]),
            Route("/v1/invoke", v1_invoke, methods=["POST"]),
            Route("/v1/stream", v1_stream, methods=["POST"]),
        ],
        middleware=[Middleware(_ServeMiddleware, state=state)],
        exception_handlers={HTTPException: http_error},
    )


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


class _Server(uvicorn.Server):
    """uvicorn's server, plus: the bound port reported after startup, a
    callback when shutdown starts, and signals that end in a clean return
    (uvicorn re-raises SIGTERM/SIGINT after draining, which would kill the
    process with 143 or raise ``KeyboardInterrupt``)."""

    def __init__(
        self,
        config: uvicorn.Config,
        *,
        on_started: Callable[[int], None],
        on_shutdown: Callable[[], None],
    ) -> None:
        super().__init__(config)
        self._on_started = on_started
        self._on_shutdown = on_shutdown

    async def startup(self, sockets: list[Any] | None = None) -> None:
        await super().startup(sockets=sockets)
        if self.started:
            port = int(self.config.port)
            with contextlib.suppress(Exception):
                port = int(self.servers[0].sockets[0].getsockname()[1])
            self._on_started(port)

    async def shutdown(self, sockets: list[Any] | None = None) -> None:
        self._on_shutdown()
        await super().shutdown(sockets=sockets)

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        from uvicorn.server import HANDLED_SIGNALS

        previous: dict[signal.Signals, Any] = {}
        for sig in HANDLED_SIGNALS:
            with contextlib.suppress(ValueError, OSError):
                previous[sig] = signal.signal(sig, self._signal)
        try:
            yield
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def _signal(self, sig: int, frame: FrameType | None) -> None:
        if self.should_exit and sig == signal.SIGINT:
            self.force_exit = True  # a second Ctrl+C: stop waiting for in-flight turns
        else:
            if not self.should_exit:
                logger.info("Received %s: draining", signal.Signals(sig).name)
            self.should_exit = True


def _read_token(env_name: str | None) -> str | None:
    if not env_name:
        return None
    value = os.environ.get(env_name, "").strip()
    return value or None


async def run_serve(
    agent: PromptiseAgent | AgentHost,
    *,
    settings: ServeSettings,
    bus: EventBus | None,
    info: dict[str, Any],
    shutdown_timeout: float = 30.0,
    host: str | None = None,
    port: int | None = None,
    log_config: dict[str, Any] | None = None,
    on_started: Callable[[str, int], None] | None = None,
    stop: asyncio.Event | None = None,
) -> int:
    """Serve *agent* over HTTP until SIGTERM / SIGINT (or *stop*), then drain.

    The bearer token is read from ``os.environ[settings.auth_token_env]``.
    On shutdown ``/readyz`` and new ``/v1/*`` requests answer ``503`` while
    in-flight turns finish, for up to *shutdown_timeout* seconds (rounded up);
    turns still running then are cancelled.  A second Ctrl+C stops waiting.

    Args:
        agent: A :class:`~promptise.agent.PromptiseAgent` (served through
            :class:`PromptiseAgentHost`) or any :class:`AgentHost`.
        settings: The manifest's ``[serve]`` section.
        bus: Where run events go (optional).
        info: What ``/v1/info`` reports.
        shutdown_timeout: Seconds in-flight turns may take after shutdown starts.
        host: Bind address (default ``settings.host``).
        port: Port (default ``settings.port``; ``0`` picks a free one).
        log_config: uvicorn logging config (``None``: leave logging as it is).
            uvicorn's access log is off: the ``http`` run events replace it.
        on_started: Called with the host and the bound port once listening.
        stop: Setting this event shuts the server down like SIGTERM does.

    Returns:
        ``0`` after a clean shutdown, ``1`` when the server failed (e.g. the
        port is in use), ``2`` when the configuration is refused (a
        non-loopback host without a token).
    """
    bind_host = host if host is not None else settings.host
    bind_port = port if port is not None else settings.port
    token = _read_token(settings.auth_token_env)
    loopback = is_loopback(bind_host)
    if not loopback and token is None:
        if settings.auth_token_env:
            how = f"Set {settings.auth_token_env} to a secret token"
        else:
            how = "Set [serve] auth_token_env, and that variable to a secret token"
        print(
            f"promptise run: refusing to serve on {bind_host} without an auth token: "
            f"anyone who can reach it could use the agent. {how}, or bind to 127.0.0.1.",
            file=sys.stderr,
        )
        return 2
    if token is not None and len(token) < 16:
        logger.warning("The serve auth token is short (%d characters); use 32 or more", len(token))

    from ..agent import PromptiseAgent

    agent_host: AgentHost = (
        PromptiseAgentHost(agent) if isinstance(agent, PromptiseAgent) else agent
    )
    draining = threading.Event()
    app = create_app(
        agent_host,
        token=token,
        max_concurrency=settings.max_concurrency,
        bus=bus,
        info=info,
        draining=draining.is_set,
        local_only=loopback and token is None,
    )
    config = uvicorn.Config(
        app,
        host=bind_host,
        port=bind_port,
        log_config=log_config,
        access_log=False,
        lifespan="off",
        ws="none",
        server_header=False,
        timeout_graceful_shutdown=max(0, math.ceil(shutdown_timeout)),
    )

    def started(bound_port: int) -> None:
        logger.info("Serving on http://%s:%d", bind_host, bound_port)
        if on_started is not None:
            try:
                on_started(bind_host, bound_port)
            except Exception:
                logger.exception("on_started callback failed")

    def shutting_down() -> None:
        draining.set()

    server = _Server(config, on_started=started, on_shutdown=shutting_down)

    async def watch_stop(event: asyncio.Event) -> None:
        await event.wait()
        server.should_exit = True

    watcher = asyncio.create_task(watch_stop(stop)) if stop is not None else None
    try:
        await server.serve()
    except SystemExit as exc:  # uvicorn exits on a failed startup (e.g. port in use)
        print(
            f"promptise run: could not serve on {bind_host}:{bind_port} "
            f"(exit {exc.code}); see the error above.",
            file=sys.stderr,
        )
        return 1
    except Exception:
        logger.exception("The HTTP server failed")
        return 1
    finally:
        draining.set()
        if watcher is not None:
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
    return 0
