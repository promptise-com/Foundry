"""``promptise run --serve``: the HTTP API (``promptise.run.serve``).

The app is driven in-process through ``httpx.ASGITransport`` with a fake
:class:`AgentHost` (no model, no network); one test serves a real
``PromptiseAgent`` on the scripted chat model, and the ``run_serve`` tests
bind 127.0.0.1 on a free port.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import signal
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from promptise import build_agent
from promptise.conversations import InMemoryConversationStore, SessionAccessDenied
from promptise.guardrails import GuardrailViolation, ScanReport
from promptise.run.events import EventBus
from promptise.run.manifest import ServeSettings
from promptise.run.serve import HostError, PromptiseAgentHost, create_app, run_serve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _scripted_model import Scripted  # noqa: E402

TOKEN = "test-token-0123456789abcdef"
USAGE = {"llm_calls": 1, "prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
INFO = {"project": "demo", "version": "0.1.0", "model": "scripted"}

DONE = {"type": "done", "full_response": "Hi there", "tool_calls": [], "duration_ms": 1.0}
EVENTS: list[dict[str, Any]] = [
    {"type": "tool_start", "tool_name": "search", "tool_display_name": "Search", "arguments": {}},
    {"type": "tool_end", "tool_name": "search", "tool_summary": "1 hit", "success": True},
    {"type": "token", "text": "Hi ", "cumulative_text": "Hi "},
    {"type": "token", "text": "there", "cumulative_text": "Hi there"},
    DONE,
]


class FakeHost:
    """An :class:`AgentHost` with scripted answers."""

    def __init__(
        self,
        *,
        ready: bool = True,
        output: str = "hello",
        events: list[dict[str, Any]] | None = None,
        fail: BaseException | None = None,
        gate: asyncio.Event | None = None,
        delay: float = 0.0,
    ) -> None:
        self.ready = ready
        self.tool_names = ["search", "lookup"]
        self.output = output
        self.events = EVENTS if events is None else events
        self.fail = fail
        self.gate = gate
        self.delay = delay
        self.calls: list[tuple[Any, str | None, str]] = []
        self.entered = asyncio.Event()

    async def invoke(
        self, input: Any, *, session_id: str | None, request_id: str
    ) -> dict[str, Any]:
        self.calls.append((input, session_id, request_id))
        self.entered.set()
        if self.gate is not None:
            await self.gate.wait()
        if self.fail is not None:
            raise self.fail
        return {"output": self.output, "usage": USAGE}

    async def stream(
        self, input: Any, *, session_id: str | None, request_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        self.calls.append((input, session_id, request_id))
        self.entered.set()
        for event in self.events:
            if self.delay:
                await asyncio.sleep(self.delay)
            yield event
        if self.fail is not None:
            raise self.fail


def make_app(host: Any = None, **kwargs: Any) -> Any:
    options: dict[str, Any] = {"token": None, "max_concurrency": 4, "bus": None, "info": INFO}
    options.update(kwargs)
    return create_app(host if host is not None else FakeHost(), **options)


def client(app: Any, base_url: str = "http://testserver", **kwargs: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=base_url, **kwargs)


def sse_events(text: str) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in text.split("\n") if line.startswith("data: ")]


def kinds(bus: EventBus) -> list[str]:
    return [e["kind"] for e in bus.history()]


# ---------------------------------------------------------------------------
# Probes and info
# ---------------------------------------------------------------------------


async def test_healthz_and_readyz():
    async with client(make_app()) as c:
        health = await c.get("/healthz")
        ready = await c.get("/readyz")
    assert (health.status_code, health.json()) == (200, {"status": "ok"})
    assert (ready.status_code, ready.json()) == (200, {"status": "ready"})
    assert health.headers["x-request-id"]


async def test_readyz_is_503_while_the_host_is_not_ready():
    async with client(make_app(FakeHost(ready=False))) as c:
        ready = await c.get("/readyz")
        health = await c.get("/healthz")
    assert ready.status_code == 503
    assert ready.json() == {"status": "not_ready"}
    assert health.status_code == 200


async def test_info_reports_the_info_and_the_tool_names():
    async with client(make_app()) as c:
        resp = await c.get("/v1/info")
    assert resp.status_code == 200
    assert resp.json() == {**INFO, "tools": ["search", "lookup"]}


async def test_unknown_routes_and_methods_answer_json():
    async with client(make_app()) as c:
        missing = await c.get("/nope")
        wrong = await c.get("/v1/invoke")
    assert (missing.status_code, missing.json()) == (404, {"error": "not found"})
    assert (wrong.status_code, wrong.json()) == (405, {"error": "method not allowed"})


# ---------------------------------------------------------------------------
# /v1/invoke
# ---------------------------------------------------------------------------


async def test_invoke_answers_and_publishes_run_events():
    bus = EventBus()
    host = FakeHost(output="The answer")
    async with client(make_app(host, bus=bus)) as c:
        resp = await c.post("/v1/invoke", json={"input": "What is it?", "session_id": "s-1"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["output"] == "The answer"
    assert body["usage"] == USAGE
    assert isinstance(body["duration_ms"], float)
    request_id = body["request_id"]
    assert resp.headers["x-request-id"] == request_id
    assert host.calls == [("What is it?", "s-1", request_id)]

    assert kinds(bus) == ["turn.start", "turn.end", "http"]
    start, end, http = bus.history()
    assert start["turn_id"] == request_id
    assert start["input"] == "What is it?"
    assert start["conversation_id"] == "s-1"
    assert end["turn_id"] == request_id
    assert end["output"] == "The answer"
    assert end["error"] is None
    assert end["duration_ms"] >= 0
    assert http["method"] == "POST"
    assert http["path"] == "/v1/invoke"
    assert http["status"] == 200
    assert http["request_id"] == request_id
    assert http["duration_ms"] >= 0


async def test_invoke_accepts_a_message_list():
    host = FakeHost()
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": [{"type": "text", "text": "And now?"}], "name": "ann"},
    ]
    async with client(make_app(host)) as c:
        resp = await c.post("/v1/invoke", json={"input": messages})
    assert resp.status_code == 200
    assert host.calls[0][0] == messages
    assert host.calls[0][1] is None


async def test_a_well_formed_client_request_id_is_kept():
    host = FakeHost()
    async with client(make_app(host)) as c:
        kept = await c.post("/v1/invoke", json={"input": "hi"}, headers={"X-Request-ID": "abc-123"})
        replaced = await c.get("/healthz", headers={"X-Request-ID": "bad id\twith spaces"})
    assert kept.headers["x-request-id"] == "abc-123"
    assert kept.json()["request_id"] == "abc-123"
    assert host.calls[0][2] == "abc-123"
    assert replaced.headers["x-request-id"] != "bad id\twith spaces"
    assert len(replaced.headers["x-request-id"]) == 32


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (b"not json", "not valid JSON"),
        (b"", "JSON object"),
        (b"[1, 2]", "JSON object"),
        (b"{}", "'input' is required"),
        (b'{"input": ""}', "must not be empty"),
        (b'{"input": "   "}', "must not be empty"),
        (b'{"input": 3}', "string or a list of messages"),
        (b'{"input": []}', "empty list"),
        (b'{"input": ["hi"]}', "input[0] must be an object"),
        (b'{"input": [{"role": "bot", "content": "x"}]}', "input[0].role"),
        (b'{"input": [{"role": "user"}]}', "input[0].content"),
        (b'{"input": [{"role": "user", "content": ["x"]}]}', "content blocks must be objects"),
        (b'{"input": [{"role": "user", "content": "x", "id": 1}]}', "unknown field(s): id"),
        (b'{"input": [{"role": "user", "content": "x", "name": 2}]}', "name must be a string"),
        (b'{"input": "hi", "stream": true}', "unknown field(s): stream"),
        (b'{"input": "hi", "session_id": "../../etc/passwd"}', "'session_id'"),
        (b'{"input": "hi", "session_id": 5}', "'session_id'"),
        (b'{"input": "hi", "session_id": ""}', "'session_id'"),
    ],
)
async def test_invoke_rejects_invalid_bodies(body: bytes, message: str):
    host = FakeHost()
    async with client(make_app(host)) as c:
        resp = await c.post(
            "/v1/invoke", content=body, headers={"Content-Type": "application/json"}
        )
    assert resp.status_code == 400
    assert message in resp.json()["error"]
    assert host.calls == []


async def test_invoke_rejects_a_body_over_the_limit():
    host = FakeHost()
    async with client(make_app(host, max_body_bytes=64)) as c:
        resp = await c.post("/v1/invoke", json={"input": "x" * 100})
    assert resp.status_code == 413
    assert "larger than 64 bytes" in resp.json()["error"]
    assert host.calls == []


async def test_a_failing_host_answers_500_with_an_error_id_and_no_details(caplog):
    caplog.set_level(logging.ERROR, logger="promptise.run.serve")
    bus = EventBus()
    host = FakeHost(fail=RuntimeError("db password is hunter2"))
    async with client(make_app(host, bus=bus)) as c:
        resp = await c.post("/v1/invoke", json={"input": "hi"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "internal error"
    error_id = body["error_id"]
    assert error_id
    assert "hunter2" not in resp.text
    assert "RuntimeError" not in resp.text
    # The details are in the log, under the same error id.
    record = next(r for r in caplog.records if error_id in r.getMessage())
    assert record.exc_info is not None
    assert "hunter2" in str(record.exc_info[1])
    end = bus.history(["turn.end"])[0]
    assert end["output"] == ""
    assert error_id in end["error"]
    assert bus.history(["http"])[0]["status"] == 500


async def test_a_host_error_answers_its_status_and_message():
    host = FakeHost(fail=HostError(422, "blocked by safety policy"))
    async with client(make_app(host)) as c:
        resp = await c.post("/v1/invoke", json={"input": "hi"})
    assert (resp.status_code, resp.json()) == (422, {"error": "blocked by safety policy"})


# ---------------------------------------------------------------------------
# Auth, host checks, concurrency, draining
# ---------------------------------------------------------------------------


async def test_bearer_auth_on_v1_routes(monkeypatch):
    compared: list[tuple[bytes, bytes]] = []
    real_compare = hmac.compare_digest

    def spy(a: Any, b: Any) -> bool:
        compared.append((a, b))
        return real_compare(a, b)

    monkeypatch.setattr(hmac, "compare_digest", spy)
    host = FakeHost()
    async with client(make_app(host, token=TOKEN)) as c:
        missing = await c.post("/v1/invoke", json={"input": "hi"})
        wrong = await c.post(
            "/v1/invoke", json={"input": "hi"}, headers={"Authorization": "Bearer nope"}
        )
        basic = await c.post(
            "/v1/invoke", json={"input": "hi"}, headers={"Authorization": f"Basic {TOKEN}"}
        )
        info = await c.get("/v1/info")
        unknown = await c.get("/v1/nothing-here")
        right = await c.post(
            "/v1/invoke", json={"input": "hi"}, headers={"Authorization": f"Bearer {TOKEN}"}
        )
        health = await c.get("/healthz")
        ready = await c.get("/readyz")
    for resp in (missing, wrong, basic, info, unknown):
        assert resp.status_code == 401
        assert resp.json() == {"error": "unauthorized"}
        assert resp.headers["www-authenticate"] == "Bearer"
    assert right.status_code == 200
    assert (health.status_code, ready.status_code) == (200, 200)
    assert len(host.calls) == 1  # only the authorised request reached the agent
    # The token is checked with a constant-time comparison.
    assert (b"nope", TOKEN.encode()) in compared
    assert (TOKEN.encode(), TOKEN.encode()) in compared


def test_an_empty_token_is_refused():
    with pytest.raises(ValueError, match="token"):
        make_app(token="")
    with pytest.raises(ValueError, match="max_concurrency"):
        make_app(max_concurrency=0)


async def test_local_only_refuses_foreign_hosts_and_cross_site_origins():
    app = make_app(local_only=True)
    async with client(app, base_url="http://evil.example") as c:
        rebound = await c.post("/v1/invoke", json={"input": "hi"})
        health = await c.get("/healthz")
    async with client(app, base_url="http://127.0.0.1:8080") as c:
        cross_site = await c.post(
            "/v1/invoke", json={"input": "hi"}, headers={"Origin": "https://evil.example"}
        )
        same_site = await c.post(
            "/v1/invoke", json={"input": "hi"}, headers={"Origin": "http://localhost:3000"}
        )
        plain = await c.post("/v1/invoke", json={"input": "hi"})
    assert rebound.status_code == 421
    assert health.status_code == 200
    assert cross_site.status_code == 403
    assert same_site.status_code == 200
    assert plain.status_code == 200


async def test_requests_over_max_concurrency_get_429():
    gate = asyncio.Event()
    host = FakeHost(gate=gate)
    async with client(make_app(host, max_concurrency=1)) as c:
        first = asyncio.create_task(c.post("/v1/invoke", json={"input": "one"}))
        await asyncio.wait_for(host.entered.wait(), 5)
        busy = await c.post("/v1/invoke", json={"input": "two"})
        busy_stream = await c.post("/v1/stream", json={"input": "two"})
        gate.set()
        done = await first
        after = await c.post("/v1/invoke", json={"input": "three"})
    assert (busy.status_code, busy.json()) == (429, {"error": "busy"})
    assert busy.headers["retry-after"] == "1"
    assert busy_stream.status_code == 429
    assert done.status_code == 200
    assert after.status_code == 200  # the slot was released
    assert [call[0] for call in host.calls] == ["one", "three"]


async def test_a_stream_holds_its_slot_until_it_ends():
    host = FakeHost()
    async with client(make_app(host, max_concurrency=1)) as c:
        stream = await c.post("/v1/stream", json={"input": "one"})
        after = await c.post("/v1/invoke", json={"input": "two"})
    assert stream.status_code == 200
    assert after.status_code == 200


async def test_draining_refuses_new_turns_and_fails_readiness():
    draining = False
    host = FakeHost()
    async with client(make_app(host, draining=lambda: draining)) as c:
        before = await c.post("/v1/invoke", json={"input": "hi"})
        draining = True
        invoke = await c.post("/v1/invoke", json={"input": "hi"})
        stream = await c.post("/v1/stream", json={"input": "hi"})
        ready = await c.get("/readyz")
        health = await c.get("/healthz")
    assert before.status_code == 200
    assert (invoke.status_code, invoke.json()) == (503, {"error": "draining"})
    assert stream.status_code == 503
    assert (ready.status_code, ready.json()) == (503, {"status": "draining"})
    assert health.status_code == 200
    assert len(host.calls) == 1


# ---------------------------------------------------------------------------
# /v1/stream
# ---------------------------------------------------------------------------


async def test_stream_relays_events_as_sse_in_order():
    bus = EventBus()
    extra = {"type": "token", "text": "late", "cumulative_text": "late"}
    host = FakeHost(events=[*EVENTS, extra])
    async with client(make_app(host, bus=bus)) as c:
        resp = await c.post("/v1/stream", json={"input": "hi", "session_id": "s-2"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    assert resp.headers["cache-control"] == "no-cache"
    request_id = resp.headers["x-request-id"]
    events = sse_events(resp.text)
    assert events == EVENTS  # in order, ending with done; nothing after it
    assert resp.text.endswith("\n\n")

    assert kinds(bus) == ["turn.start", *["stream"] * len(EVENTS), "turn.end", "http"]
    assert bus.history(["turn.start"])[0]["conversation_id"] == "s-2"
    assert [e["event"] for e in bus.history(["stream"])] == EVENTS
    assert {e["turn_id"] for e in bus.history(["stream", "turn.start", "turn.end"])} == {request_id}
    end = bus.history(["turn.end"])[0]
    assert (end["output"], end["error"]) == ("Hi there", None)
    assert bus.history(["http"])[0]["status"] == 200


async def test_stream_ends_after_an_error_event():
    bus = EventBus()
    error = {"type": "error", "message": "Input blocked by safety policy.", "recoverable": False}
    host = FakeHost(events=[error, DONE])
    async with client(make_app(host, bus=bus)) as c:
        resp = await c.post("/v1/stream", json={"input": "hi"})
    assert sse_events(resp.text) == [error]
    assert bus.history(["turn.end"])[0]["error"] == "Input blocked by safety policy."


async def test_a_failing_stream_ends_with_a_generic_error_event(caplog):
    caplog.set_level(logging.ERROR, logger="promptise.run.serve")
    host = FakeHost(events=EVENTS[:1], fail=RuntimeError("token sk-secret leaked"))
    async with client(make_app(host)) as c:
        resp = await c.post("/v1/stream", json={"input": "hi"})
    assert resp.status_code == 200
    events = sse_events(resp.text)
    assert events[0] == EVENTS[0]
    last = events[-1]
    assert last["type"] == "error"
    assert last["message"] == "An error occurred during processing."
    assert last["error_id"]
    assert "sk-secret" not in resp.text
    assert any(last["error_id"] in r.getMessage() for r in caplog.records)


async def test_a_stream_without_a_final_event_still_ends_with_one():
    host = FakeHost(events=EVENTS[:2])
    async with client(make_app(host)) as c:
        resp = await c.post("/v1/stream", json={"input": "hi"})
    events = sse_events(resp.text)
    assert events[:2] == EVENTS[:2]
    assert events[-1]["type"] == "error"


async def test_a_quiet_stream_sends_pings():
    host = FakeHost(events=[DONE], delay=0.2)
    async with client(make_app(host, ping_seconds=0.02)) as c:
        resp = await c.post("/v1/stream", json={"input": "hi"})
    assert ": ping\n\n" in resp.text
    assert sse_events(resp.text) == [DONE]


async def test_stream_validates_like_invoke():
    async with client(make_app()) as c:
        resp = await c.post("/v1/stream", json={"input": ""})
    assert resp.status_code == 400
    assert "must not be empty" in resp.json()["error"]


class _HangingHost(FakeHost):
    """Streams one token, then waits (until cancelled)."""

    def __init__(self) -> None:
        super().__init__()
        self.closed = asyncio.Event()

    async def stream(
        self, input: Any, *, session_id: str | None, request_id: str
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            yield EVENTS[2]
            await asyncio.sleep(30)
            yield DONE
        finally:
            self.closed.set()


@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
async def test_a_client_that_goes_away_cancels_its_stream(spec_version: str):
    """ASGI < 2.4 reports the disconnect as a message; 2.4 servers fail the send."""
    bus = EventBus()
    host = _HangingHost()
    app = make_app(host, bus=bus, max_concurrency=1, ping_seconds=0.05)
    pending = [{"type": "http.request", "body": b'{"input": "hi"}', "more_body": False}]
    first_chunk = asyncio.Event()

    async def receive() -> dict[str, Any]:
        if pending:
            return pending.pop(0)
        await first_chunk.wait()
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        if message["type"] == "http.response.body" and message.get("body"):
            if first_chunk.is_set() and spec_version == "2.4":
                raise OSError("connection reset")
            first_chunk.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": spec_version},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/stream",
        "raw_path": b"/v1/stream",
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"testserver"), (b"content-type", b"application/json")],
        "client": ("127.0.0.1", 50000),
        "server": ("testserver", 80),
    }
    await asyncio.wait_for(app(scope, receive, send), 5)
    assert host.closed.is_set()  # the agent's stream was stopped
    end = bus.history(["turn.end"])[0]
    assert end["error"] == "cancelled"
    assert bus.history(["http"])[0]["status"] == 200
    async with client(app) as c:  # the slot was released
        after = await c.post("/v1/invoke", json={"input": "next"})
    assert after.status_code == 200


# ---------------------------------------------------------------------------
# PromptiseAgentHost (a real agent on the scripted model)
# ---------------------------------------------------------------------------


class _Metered(Scripted):
    """Answers "ok" and reports token usage like a real provider."""

    async def _agenerate(self, messages: Any, stop: Any = None, **kw: Any) -> ChatResult:
        usage = {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}
        message = AIMessage(content="ok", usage_metadata=usage)  # type: ignore[arg-type]
        return ChatResult(generations=[ChatGeneration(message=message)])


async def test_promptise_agent_host_invokes_and_streams_a_real_agent():
    agent = await build_agent(model=Scripted(mode="answer"))
    try:
        host = PromptiseAgentHost(agent)
        assert host.ready
        assert host.tool_names == []
        result = await host.invoke("hi", session_id=None, request_id="r1")
        assert result["output"] == "ok"
        assert result["usage"]["llm_calls"] == 1

        events = [e async for e in host.stream("hi", session_id=None, request_id="r2")]
        assert [e["type"] for e in events] == ["token", "done"]
        assert events[0]["text"] == "ok"
        assert events[-1]["full_response"] == "ok"
        assert events[-1]["usage"]["llm_calls"] == 1

        messages = [{"role": "user", "content": "hi"}]
        result = await host.invoke(messages, session_id="s1", request_id="r3")
        assert result["output"] == "ok"

        async with client(make_app(host)) as c:
            invoke = await c.post("/v1/invoke", json={"input": "hi"})
            stream = await c.post("/v1/stream", json={"input": "hi"})
        assert invoke.json()["output"] == "ok"
        assert [e["type"] for e in sse_events(stream.text)] == ["token", "done"]
    finally:
        await agent.shutdown()


async def test_promptise_agent_host_reports_token_usage_per_turn():
    agent = await build_agent(model=_Metered(mode="answer"))
    try:
        host = PromptiseAgentHost(agent)
        first, second = await asyncio.gather(
            host.invoke("a", session_id=None, request_id="r1"),
            host.invoke("b", session_id=None, request_id="r2"),
        )
        expected = {"llm_calls": 1, "prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9}
        assert first["usage"] == expected
        assert second["usage"] == expected  # concurrent turns are counted apart
        events = [e async for e in host.stream("c", session_id=None, request_id="r3")]
        assert events[-1]["usage"] == expected
    finally:
        await agent.shutdown()


async def test_promptise_agent_host_keeps_session_history_in_the_store():
    store = InMemoryConversationStore()
    agent = await build_agent(model=Scripted(mode="answer"), conversation_store=store)
    try:
        host = PromptiseAgentHost(agent)
        assert host.persists_sessions
        result = await host.invoke("first", session_id="s1", request_id="r1")
        assert result["output"] == "ok"
        assert [m.content for m in await store.load_messages("s1")] == ["first", "ok"]

        events = [e async for e in host.stream("second", session_id="s1", request_id="r2")]
        assert [e["type"] for e in events] == ["token", "done"]
        assert events[-1]["full_response"] == "ok"
        history = [m.content for m in await store.load_messages("s1")]
        assert history == ["first", "ok", "second", "ok"]
    finally:
        await agent.shutdown()


class _RefusingAgent:
    """Stands in for an agent whose guardrail or session check refuses the turn."""

    tool_names: list[str] = []
    _conversation_store = object()

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        raise self.exc

    async def chat(self, *args: Any, **kwargs: Any) -> str:
        raise self.exc


@pytest.mark.parametrize(
    ("exc", "status", "message"),
    [
        (
            GuardrailViolation(
                ScanReport(passed=False, findings=[], duration_ms=0, scanners_run=[], text_length=0)
            ),
            422,
            "blocked by safety policy",
        ),
        (SessionAccessDenied("s1", "mallory", "alice"), 403, "session access denied"),
    ],
)
async def test_promptise_agent_host_maps_refusals_to_client_errors(exc, status, message):
    host = PromptiseAgentHost(_RefusingAgent(exc))  # type: ignore[arg-type]
    async with client(make_app(host)) as c:
        invoke = await c.post("/v1/invoke", json={"input": "hi"})
        chat = await c.post("/v1/invoke", json={"input": "hi", "session_id": "s1"})
        stream = await c.post("/v1/stream", json={"input": "hi", "session_id": "s1"})
    assert (invoke.status_code, invoke.json()) == (status, {"error": message})
    assert (chat.status_code, chat.json()) == (status, {"error": message})
    assert [e["type"] for e in sse_events(stream.text)] == ["error"]
    assert sse_events(stream.text)[0]["message"] == message
    for resp in (invoke, chat, stream):
        assert "mallory" not in resp.text


# ---------------------------------------------------------------------------
# run_serve
# ---------------------------------------------------------------------------


async def test_run_serve_refuses_a_public_host_without_a_token(monkeypatch, capsys):
    monkeypatch.delenv("PROMPTISE_SERVE_TOKEN", raising=False)
    settings = ServeSettings(host="0.0.0.0", port=0)
    code = await run_serve(FakeHost(), settings=settings, bus=None, info=INFO)
    assert code == 2
    err = capsys.readouterr().err
    assert "refusing to serve on 0.0.0.0 without an auth token" in err
    assert "PROMPTISE_SERVE_TOKEN" in err


async def _serve_in_background(
    host: Any, monkeypatch: pytest.MonkeyPatch, **kwargs: Any
) -> tuple[asyncio.Task[int], int, asyncio.Event]:
    monkeypatch.delenv("PROMPTISE_SERVE_TOKEN", raising=False)
    started: asyncio.Future[tuple[str, int]] = asyncio.get_running_loop().create_future()
    stop = asyncio.Event()
    task = asyncio.create_task(
        run_serve(
            host,
            settings=ServeSettings(host="127.0.0.1", port=0),
            bus=kwargs.pop("bus", None),
            info=INFO,
            on_started=lambda h, p: started.set_result((h, p)),
            stop=stop,
            **kwargs,
        )
    )
    bound_host, port = await asyncio.wait_for(started, 10)
    assert bound_host == "127.0.0.1"
    assert port > 0
    return task, port, stop


async def test_run_serve_listens_on_a_free_port_and_stops_cleanly(monkeypatch):
    bus = EventBus()
    task, port, stop = await _serve_in_background(FakeHost(), monkeypatch, bus=bus)
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False) as c:
            health = await c.get("/healthz")
            invoke = await c.post("/v1/invoke", json={"input": "hi"})
            stream = await c.post("/v1/stream", json={"input": "hi"})
    finally:
        stop.set()
    assert await asyncio.wait_for(task, 10) == 0
    assert (health.status_code, health.json()) == (200, {"status": "ok"})
    assert "server" not in health.headers
    assert invoke.json()["output"] == "hello"
    assert sse_events(stream.text) == EVENTS
    paths = [e["path"] for e in bus.history(["http"])]
    assert paths == ["/healthz", "/v1/invoke", "/v1/stream"]


async def test_run_serve_lets_in_flight_turns_finish_on_shutdown(monkeypatch):
    gate = asyncio.Event()
    host = FakeHost(gate=gate)
    task, port, stop = await _serve_in_background(host, monkeypatch, shutdown_timeout=10)
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", trust_env=False) as c:
        request = asyncio.create_task(c.post("/v1/invoke", json={"input": "slow"}))
        await asyncio.wait_for(host.entered.wait(), 5)
        stop.set()
        await asyncio.sleep(0.3)  # shutdown has started and is waiting for the turn
        assert not task.done()
        gate.set()
        resp = await asyncio.wait_for(request, 10)
    assert resp.status_code == 200
    assert resp.json()["output"] == "hello"
    assert await asyncio.wait_for(task, 10) == 0


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
async def test_run_serve_drains_on_sigterm_and_returns_0(monkeypatch):
    previous = signal.getsignal(signal.SIGTERM)
    task, port, _stop = await _serve_in_background(FakeHost(), monkeypatch)
    os.kill(os.getpid(), signal.SIGTERM)
    assert await asyncio.wait_for(task, 10) == 0
    # The previous handler is back, and the signal was not re-raised.
    assert signal.getsignal(signal.SIGTERM) == previous
