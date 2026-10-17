"""Tests for the ``promptise run`` terminal UI's state (``promptise.run.tui.state``).

:class:`RunState` is plain Python: every run event kind is folded in and read
back here without Textual.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any

import pytest

from promptise.run.tui.state import (
    CAP,
    ChatError,
    ChatNote,
    ChatText,
    ChatTurn,
    Ring,
    RunState,
    ToolCard,
    format_cost,
    format_duration,
    preview,
)

_seq = iter(range(1, 10**9))


def ev(kind: str, **payload: Any) -> dict[str, Any]:
    """A run event as the bus makes it."""
    event: dict[str, Any] = {"v": 1, "seq": next(_seq), "ts": time.time(), "kind": kind}
    event.update(payload)
    return event


def run_start(mode: str = "chat", **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "project": "support-bot",
        "project_version": "0.1.0",
        "main": "main.py",
        "mode": mode,
        "entry_kind": "script" if mode == "script" else "factory",
        "pid": 4242,
        "python": "3.12.4",
        "promptise_version": "1.4.0",
        "cwd": "/work",
        "root": "/work",
        "argv": ["main.py"],
        "flags": {"devtools": True, "prod": False, "tui": True, "watch": False},
        "model_override": None,
        "config": {"name": "support-bot", "main": "main.py"},
        "env": [{"name": "OPENAI_API_KEY", "set": True, "required": True}],
        "scripts": {"dev": "promptise run --devtools"},
    }
    payload.update(extra)
    return ev("run.start", **payload)


def timeline(agent: str, event_type: str, **entry: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "entry_id": f"e{next(_seq)}",
        "timestamp": time.time(),
        "event_type": event_type,
        "category": "x",
        "agent_id": agent,
        "details": entry.pop("details", f"{event_type} happened"),
        "duration": entry.pop("duration", None),
        "parent_id": entry.pop("parent_id", None),
        "metadata": entry.pop("metadata", {}),
    }
    body.update(entry)
    return ev("timeline", agent=agent, entry=body)


def llm_end(agent: str = "bot", tokens: int = 30, model: str = "gpt-test") -> dict[str, Any]:
    return timeline(
        agent,
        "llm.end",
        duration=0.5,
        metadata={
            "model": model,
            "prompt_tokens": tokens - 10,
            "completion_tokens": 10,
            "total_tokens": tokens,
        },
    )


def agent_built(agent: str = "bot") -> dict[str, Any]:
    return ev(
        "agent.built",
        agent=agent,
        model="openai:gpt-test",
        observe_level="full",
        tools=[
            {"name": "search", "description": "Search docs", "server": "docs", "args_schema": {}},
            {"name": "lookup", "description": "Look up", "server": "docs", "args_schema": {}},
            {"name": "send", "description": "Send mail", "server": "mail", "args_schema": {}},
        ],
        servers=[
            {"name": "docs", "transport": "http", "target": "http://localhost:8000/mcp"},
            {"name": "mail", "transport": "stdio", "target": "python mail.py"},
        ],
        features={"memory": True, "cache": False, "guardrails": "pii", "approval": True},
        instructions="Be helpful.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class Item:
    value: int
    id: int = 0


class TestRing:
    def test_ids_increase_and_the_oldest_is_dropped(self) -> None:
        ring: Ring[Item] = Ring(3)
        for value in range(5):
            ring.append(Item(value))
        assert [i.value for i in ring] == [2, 3, 4]
        assert [i.id for i in ring] == [3, 4, 5]
        assert ring.first_id == 3 and ring.last_id == 5
        assert ring.get(2) is None and ring.get(4).value == 3  # type: ignore[union-attr]
        assert ring.get(6) is None

    def test_since_returns_only_new_items(self) -> None:
        ring: Ring[Item] = Ring(10)
        assert ring.since(0) == [] and ring.last_id == 0 and ring.last() is None
        for value in range(6):
            ring.append(Item(value))
        assert [i.value for i in ring.since(4)] == [4, 5]
        assert [i.value for i in ring.since(0)] == list(range(6))
        assert [i.value for i in ring.since(1)] == [1, 2, 3, 4, 5]
        assert ring.since(6) == []
        assert ring.last().value == 5  # type: ignore[union-attr]

    def test_since_after_eviction(self) -> None:
        ring: Ring[Item] = Ring(3)
        for value in range(10):
            ring.append(Item(value))
        assert [i.value for i in ring.since(2)] == [7, 8, 9]


def test_formatting() -> None:
    assert format_cost(None) == "—"
    assert format_cost(0.0123) == "$0.0123"
    assert format_cost(0) == "$0.0000"
    assert format_duration(None) == "—"
    assert format_duration(0.42) == "0.4s"
    assert format_duration(12) == "12s"
    assert format_duration(184) == "3m 04s"
    assert format_duration(3720) == "1h 02m"
    assert preview(None) == ""
    assert preview({"a": 1}) == '{"a": 1}'
    assert preview("x" * 200, 10) == "x" * 9 + "…"
    assert preview("a\n  b") == "a b"


def test_importing_the_package_does_not_import_textual() -> None:
    code = (
        "import sys; import promptise.run.tui; "
        "print('textual' in sys.modules, 'promptise.run.tui.app' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.split() == ["False", "False"]


# ---------------------------------------------------------------------------
# Run lifecycle and header
# ---------------------------------------------------------------------------


class TestRunLifecycle:
    def test_before_anything_arrived(self) -> None:
        state = RunState()
        fields = state.header_fields()
        assert fields["state"] == "starting"
        assert fields["project"] == "promptise run"
        assert fields["mode"] == "—"
        assert fields["cost"] == "—"
        assert fields["approvals"] == "0"
        assert state.exit_code is None
        assert not state.chat_mode

    def test_run_start_fills_the_header(self) -> None:
        state = RunState()
        sections = state.apply(run_start("chat"))
        assert {"header", "run", "agents"} <= sections
        fields = state.header_fields()
        assert fields["project"] == "support-bot"
        assert fields["version"] == "0.1.0"
        assert fields["mode"] == "chat"
        assert fields["state"] == "running"
        assert fields["run"] == ""
        assert state.chat_mode
        assert state.current_run is not None and state.current_run.info["pid"] == 4242

    def test_project_falls_back_to_the_main_file(self) -> None:
        state = RunState()
        state.apply(run_start("script", project="", main="/work/app/main.py"))
        assert state.project == "main.py"
        assert not state.chat_mode

    def test_state_events_and_exit(self) -> None:
        state = RunState()
        state.apply(run_start())
        state.apply(ev("state", state="busy", detail="answering"))
        assert state.state_label() == "busy" and state.state_detail == "answering"
        state.apply(ev("run.exit", code=3, error="boom", traceback="Traceback...", duration_s=2.5))
        assert state.state_label() == "exited code 3"
        assert state.exit_code == 3
        assert state.current_run is not None and state.current_run.duration(0) == 2.5
        state.apply(ev("state", state="failed", detail=""))
        assert state.state_label() == "failed code 3"
        notes = [item for item in state.chat if isinstance(item, ChatNote)]
        assert "exited with code 3 after 2.5s · boom" in notes[-1].text
        assert notes[-1].style == "error"

    def test_exit_without_a_start(self) -> None:
        state = RunState()
        state.apply(ev("run.exit", code=0, error=None, traceback=None, duration_s=None))
        assert state.exit_code == 0
        assert state.run_number == 1

    def test_restart_keeps_history_and_marks_the_new_run(self) -> None:
        state = RunState()
        state.apply(run_start("script"))
        state.apply(agent_built("bot"))
        state.apply(llm_end())
        state.apply(ev("output", stream="stdout", text="first run\n"))
        state.apply(ev("run.exit", code=1, error="x", traceback=None, duration_s=1.0))
        state.pop_changes()
        state.apply(run_start("script"))
        changes = state.pop_changes()
        assert changes.new_run
        assert state.run_number == 2
        assert state.header_fields()["run"] == "run #2"
        assert state.header_fields()["state"] == "running"
        assert state.exit_code == 1  # the last exit is remembered across the restart
        assert state.agents == {}  # per run
        assert state.stats["totals"].get("llm_calls") == 0  # stats are per run
        assert len(state.timeline) == 1  # history is kept
        texts = [line.text for line in state.console]
        assert texts[0] == "first run"
        assert texts[-1] == "run #2"
        assert isinstance(state.chat.last(), ChatNote)
        assert [r.number for r in state.runs] == [1, 2]

    def test_exit_aborts_open_tool_calls_and_turns(self) -> None:
        state = RunState()
        state.apply(run_start())
        state.apply(timeline("bot", "tool.call", metadata={"run_id": "r1", "tool_name": "search"}))
        state.apply(ev("turn.start", turn_id="t1", input="hi", conversation_id="c"))
        state.apply(ev("stream", turn_id="t1", event={"type": "tool_start", "tool_name": "search"}))
        state.pop_changes()
        state.apply(ev("run.exit", code=130, error=None, traceback=None, duration_s=1.0))
        changes = state.pop_changes()
        call = state.tool_calls.last()
        assert call is not None and call.status == "aborted"
        assert call.id in changes.tool_calls
        turn = state.chat.get(1)
        assert isinstance(turn, ChatTurn) and turn.done and turn.error == "the program exited"
        card = turn.parts[0]
        assert isinstance(card, ToolCard) and card.success is False


# ---------------------------------------------------------------------------
# Agents, timeline, tools, events, stats
# ---------------------------------------------------------------------------


class TestAgents:
    def test_agent_built_and_closed(self) -> None:
        state = RunState()
        state.apply(run_start())
        assert state.apply(agent_built("bot")) >= {"agents"}
        agent = state.agents["bot"]
        assert agent.model == "openai:gpt-test"
        assert agent.observe_level == "full"
        assert [t.name for t in agent.tools] == ["search", "lookup", "send"]
        assert [(s.name, s.transport, s.tool_count) for s in agent.servers] == [
            ("docs", "http", 2),
            ("mail", "stdio", 1),
        ]
        assert agent.features["guardrails"] == "pii"
        assert agent.run == 1 and not agent.closed
        state.apply(ev("agent.closed", agent="bot"))
        assert state.agents["bot"].closed
        state.apply(ev("agent.closed", agent="nobody"))  # unknown: ignored

    def test_the_configured_model_shows_in_local_stats(self) -> None:
        state = RunState()
        state.apply(agent_built("bot"))
        assert state.stats["agents"]["bot"]["model"] == "openai:gpt-test"


class TestTimeline:
    def test_rows_tokens_and_sparkline(self) -> None:
        state = RunState()
        state.apply(run_start())
        state.apply(timeline("bot", "agent.input", details="hello"))
        sections = state.apply(llm_end("bot", tokens=42))
        assert {"timeline", "metrics"} <= sections
        state.apply(llm_end("helper", tokens=7))
        rows = list(state.timeline)
        assert [r.event_type for r in rows] == ["agent.input", "llm.end", "llm.end"]
        assert rows[0].details == "hello" and rows[0].tokens is None
        assert rows[1].tokens == 42 and rows[1].duration_ms == 500.0
        assert list(state.token_series) == [42, 7]
        assert state.timeline_agents == ["bot", "helper"]
        # Without a stats event the totals are folded locally.
        totals = state.stats["totals"]
        assert totals["llm_calls"] == 2 and totals["total_tokens"] == 49
        assert state.header_fields()["tokens"] == "49"

    def test_cap_drops_the_oldest(self) -> None:
        state = RunState()
        for _ in range(CAP + 100):
            state.apply(timeline("bot", "agent.input"))
        assert len(state.timeline) == CAP
        assert state.timeline.first_id == 101

    def test_small_cap(self) -> None:
        state = RunState(cap=10)
        for _ in range(25):
            state.apply(timeline("bot", "tool.call", metadata={"run_id": "x"}))
            state.apply(ev("log", logger="l", level="INFO", message="m", exc_text=None))
        assert len(state.timeline) == 10
        assert len(state.tool_calls) == 10
        assert len(state.logs) == 10


class TestToolCalls:
    def test_call_and_result_are_paired_by_run_id(self) -> None:
        state = RunState()
        state.apply(run_start())
        state.apply(
            timeline(
                "bot",
                "tool.call",
                metadata={"run_id": "r1", "tool_name": "search", "arguments": '{"q": "refund"}'},
            )
        )
        state.apply(timeline("bot", "tool.call", metadata={"run_id": "r2", "tool_name": "send"}))
        state.pop_changes()
        state.apply(
            timeline(
                "bot",
                "tool.result",
                duration=0.12,
                metadata={
                    "run_id": "r1",
                    "tool_name": "search",
                    "result_preview": "3 articles",
                    "latency_ms": 120.0,
                },
            )
        )
        changes = state.pop_changes()
        calls = list(state.tool_calls)
        assert len(calls) == 2
        first, second = calls
        assert first.status == "ok" and first.latency_ms == 120.0
        assert first.arguments == '{"q": "refund"}'
        assert first.result_preview == "3 articles"
        assert first.call_entry is not None and first.result_entry is not None
        assert changes.tool_calls == {first.id}
        assert second.status == "running"

    def test_errors_and_tool_reported_errors(self) -> None:
        state = RunState()
        state.apply(timeline("bot", "tool.call", metadata={"run_id": "a", "tool_name": "t"}))
        state.apply(timeline("bot", "tool.call", metadata={"run_id": "b", "tool_name": "t"}))
        state.apply(
            timeline(
                "bot",
                "tool.error",
                duration=0.05,
                metadata={
                    "run_id": "a",
                    "tool_name": "t",
                    "error_type": "ValueError",
                    "error": "bad input",
                    "traceback": "Traceback (most recent call last): ...",
                },
            )
        )
        state.apply(
            timeline(
                "bot",
                "tool.result",
                metadata={"run_id": "b", "status": "error", "result_preview": "denied"},
            )
        )
        a, b = list(state.tool_calls)
        assert a.status == "error" and a.error == "bad input" and a.error_type == "ValueError"
        assert a.latency_ms == 50.0  # from the entry's duration
        assert a.traceback and a.traceback.startswith("Traceback")
        assert b.status == "error" and b.result_preview == "denied"

    def test_a_result_without_its_call_still_shows(self) -> None:
        state = RunState()
        state.apply(timeline("bot", "tool.result", metadata={"run_id": "zz", "tool_name": "lost"}))
        (call,) = list(state.tool_calls)
        assert call.tool == "lost" and call.status == "ok" and call.arguments == ""

    def test_arguments_not_recorded(self) -> None:
        state = RunState()
        state.apply(
            timeline(
                "bot",
                "tool.call",
                metadata={"run_id": "r", "tool_name": "t", "arguments_length": 9},
            )
        )
        assert state.tool_calls.last().arguments == "<9 chars, not recorded>"  # type: ignore[union-attr]


class TestEventsAndStats:
    def test_agent_events(self) -> None:
        state = RunState()
        state.apply(
            ev(
                "event",
                agent="bot",
                event={
                    "event_type": "guardrail.blocked",
                    "severity": "warning",
                    "timestamp": 1_700_000_000.0,
                    "agent_id": "bot",
                    "data": {"rule": "pii"},
                },
            )
        )
        (row,) = list(state.events)
        assert (row.agent, row.event_type, row.severity) == ("bot", "guardrail.blocked", "warning")
        assert row.ts == 1_700_000_000.0 and row.data == {"rule": "pii"}

    def test_stats_event_wins_over_local_totals(self) -> None:
        state = RunState()
        state.apply(llm_end(tokens=10))
        state.apply(
            ev(
                "stats",
                totals={
                    "llm_calls": 3,
                    "total_tokens": 1234,
                    "cost_usd": 0.0123,
                    "errors": 2,
                },
                agents={"bot": {"model": "m", "llm_calls": 3}},
                models={"m": {"llm_calls": 3}},
                tools={"search": {"calls": 1, "errors": 0, "total_ms": 5.0, "avg_ms": 5.0}},
            )
        )
        fields = state.header_fields()
        assert fields["llm_calls"] == "3"
        assert fields["tokens"] == "1,234"
        assert fields["cost"] == "$0.0123"
        assert fields["errors"] == "2"
        state.apply(llm_end(tokens=99))  # local folding no longer shows
        assert state.header_fields()["tokens"] == "1,234"
        assert state.stats["tools"]["search"]["calls"] == 1

    def test_malformed_stats(self) -> None:
        state = RunState()
        state.apply(ev("stats", totals="nope", agents=[1], models=None))
        assert state.header_fields()["cost"] == "—"
        assert state.stats["agents"] == {}


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


class TestChat:
    def test_a_streamed_turn(self) -> None:
        state = RunState()
        state.apply(run_start("chat"))
        turn = state.add_user_turn("t1", "find my order")
        state.pop_changes()
        state.apply(ev("turn.start", turn_id="t1", input="find my order", conversation_id="c1"))
        assert len([i for i in state.chat if isinstance(i, ChatTurn)]) == 1  # merged
        assert turn.conversation_id == "c1"
        for word in ("Let ", "me ", "check."):
            state.apply(
                ev(
                    "stream",
                    turn_id="t1",
                    event={"type": "token", "text": word, "cumulative_text": "..."},
                )
            )
        state.apply(
            ev(
                "stream",
                turn_id="t1",
                event={
                    "type": "tool_start",
                    "tool_name": "lookup_order",
                    "tool_display_name": "Looking up order",
                    "arguments": {"id": 7},
                    "tool_index": 0,
                },
            )
        )
        state.apply(
            ev(
                "stream",
                turn_id="t1",
                event={
                    "type": "tool_end",
                    "tool_name": "lookup_order",
                    "tool_summary": "shipped",
                    "duration_ms": 31.0,
                    "success": True,
                    "tool_index": 0,
                    "reused": False,
                },
            )
        )
        state.apply(ev("stream", turn_id="t1", event={"type": "token", "text": " It shipped."}))
        state.apply(
            ev(
                "stream",
                turn_id="t1",
                event={"type": "done", "full_response": "It shipped.", "tool_calls": []},
            )
        )
        changes = state.pop_changes()
        assert turn.id in changes.chat
        state.apply(
            ev("turn.end", turn_id="t1", output="It shipped.", duration_ms=900.0, error=None)
        )
        assert turn.done and turn.duration_ms == 900.0 and turn.output == "It shipped."
        assert [type(p) for p in turn.parts] == [ChatText, ToolCard, ChatText]
        assert turn.text == "Let me check. It shipped."
        card = turn.parts[1]
        assert isinstance(card, ToolCard)
        assert card.display_name == "Looking up order" and card.arguments == {"id": 7}
        assert card.success is True and card.summary == "shipped" and card.duration_ms == 31.0

    def test_done_or_turn_end_fill_in_an_unstreamed_answer(self) -> None:
        state = RunState()
        state.apply(ev("turn.start", turn_id="a", input="q1"))
        state.apply(ev("stream", turn_id="a", event={"type": "done", "full_response": "A1"}))
        state.apply(ev("turn.start", turn_id="b", input="q2"))
        state.apply(ev("turn.end", turn_id="b", output="A2", duration_ms=1.0, error=None))
        turns = [i for i in state.chat if isinstance(i, ChatTurn)]
        assert [t.text for t in turns] == ["A1", "A2"]

    def test_errors_and_failed_tools(self) -> None:
        state = RunState()
        state.apply(
            ev(
                "stream",
                turn_id="x",
                event={"type": "tool_start", "tool_name": "t", "tool_index": 0},
            )
        )
        state.apply(
            ev(
                "stream",
                turn_id="x",
                event={"type": "tool_end", "tool_name": "t", "success": False, "tool_index": 0},
            )
        )
        state.apply(ev("stream", turn_id="x", event={"type": "error", "message": "rate limited"}))
        state.apply(ev("turn.end", turn_id="x", output=None, duration_ms=5.0, error="failed"))
        (turn,) = [i for i in state.chat if isinstance(i, ChatTurn)]
        assert turn.input == ""  # no turn.start seen: still shown
        card, error = turn.parts
        assert isinstance(card, ToolCard) and card.success is False
        assert isinstance(error, ChatError) and error.message == "rate limited"
        assert turn.error == "failed"

    def test_tool_end_matching_without_index(self) -> None:
        state = RunState()
        for name in ("a", "b"):
            state.apply(ev("stream", turn_id="t", event={"type": "tool_start", "tool_name": name}))
        state.apply(ev("stream", turn_id="t", event={"type": "tool_end", "tool_name": "a"}))
        turn = state.chat.last()
        assert isinstance(turn, ChatTurn)
        a, b = turn.parts
        assert isinstance(a, ToolCard) and a.success is True
        assert isinstance(b, ToolCard) and b.success is None
        # A tool_end with no matching start gets a card of its own.
        state.apply(ev("stream", turn_id="t", event={"type": "tool_end", "tool_name": "zzz"}))
        assert len(turn.parts) == 3

    def test_unknown_stream_types_are_ignored(self) -> None:
        state = RunState()
        state.apply(ev("turn.start", turn_id="t", input="hi"))
        state.pop_changes()
        state.apply(ev("stream", turn_id="t", event={"type": "thinking", "text": "hmm"}))
        state.apply(ev("stream", turn_id="t", event={"type": "token", "text": ""}))
        assert not state.pop_changes().chat


# ---------------------------------------------------------------------------
# Console, logs, http
# ---------------------------------------------------------------------------


class TestConsoleAndLogs:
    def test_script_mode_output(self) -> None:
        state = RunState()
        state.apply(run_start("script"))
        state.apply(ev("output", stream="stdout", text="hello\n"))
        state.apply(ev("output", stream="stderr", text="careful"))
        state.add_console_input("42")
        assert [(c.stream, c.text) for c in state.console] == [
            ("stdout", "hello"),
            ("stderr", "careful"),
            ("stdin", "42"),
        ]
        # stderr goes to the logs too; stdout does not (it is in the console).
        assert [(log.source, log.message, log.level) for log in state.logs] == [
            ("stderr", "careful", "WARNING")
        ]

    def test_chat_mode_prints_go_to_the_logs(self) -> None:
        state = RunState()
        state.apply(run_start("chat"))
        state.apply(ev("output", stream="stdout", text="debug print"))
        assert [(log.source, log.message) for log in state.logs] == [("stdout", "debug print")]

    def test_log_records(self) -> None:
        state = RunState()
        state.apply(ev("log", logger="app", level="error", message="db down", exc_text="Trace"))
        state.apply(ev("log", logger="app", level=None, message=None, exc_text=None))
        error, default = list(state.logs)
        assert (error.level, error.levelno, error.exc_text) == ("ERROR", 40, "Trace")
        assert (default.level, default.levelno, default.message) == ("INFO", 20, "")

    def test_http_requests(self) -> None:
        state = RunState()
        state.apply(
            ev(
                "http",
                method="POST",
                path="/v1/invoke",
                status=200,
                duration_ms=12.5,
                request_id="r",
            )
        )
        state.apply(ev("http", method=None, path=None, status="bad", duration_ms=None))
        ok, bad = list(state.http)
        assert (ok.method, ok.path, ok.status, ok.duration_ms) == ("POST", "/v1/invoke", 200, 12.5)
        assert (bad.method, bad.status) == ("?", None)


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------


def approval_pending(request_id: str = "req1", deadline: float | None = None) -> dict[str, Any]:
    now = time.time()
    return ev(
        "approval.pending",
        agent="bot",
        request={
            "request_id": request_id,
            "tool_name": "send_email",
            "arguments": {"to": "a@example.com", "body": "hi"},
            "agent_id": "bot",
            "caller_user_id": "u1",
            "context_summary": "user: please email Ann",
            "timestamp": now,
            "timeout": 300.0,
            "metadata": {"risk": "write"},
        },
        deadline=deadline if deadline is not None else now + 120,
    )


class TestApprovals:
    def test_pending_then_resolved(self) -> None:
        state = RunState()
        state.apply(run_start())
        sections = state.apply(approval_pending())
        assert {"approvals", "header"} <= sections
        changes = state.pop_changes()
        (row,) = state.pending_approvals
        assert changes.new_approvals == [row.id]
        assert (row.agent, row.tool, row.caller) == ("bot", "send_email", "u1")
        assert row.arguments == {"to": "a@example.com", "body": "hi"}
        assert row.context_summary == "user: please email Ann"
        left = row.time_left(time.time())
        assert left is not None and 100 < left <= 120
        assert state.approvals_awaiting == 1
        assert state.header_fields()["approvals"] == "1"
        assert state.approval("req1") is row

        assert state.mark_approval_answered("req1", True) is row
        assert row.answered is True and row.pending
        assert state.approvals_awaiting == 0  # answered here, awaiting the program
        assert len(state.pending_approvals) == 1

        state.apply(
            ev(
                "approval.resolved",
                request_id="req1",
                approved=True,
                reason=None,
                outcome="approved",
            )
        )
        assert row.status == "approved" and row.approved is True and not row.pending
        assert row.answered is None and row.resolved_at is not None
        assert state.pending_approvals == []
        assert state.mark_approval_answered("req1", False) is None  # already resolved

    def test_outcomes(self) -> None:
        state = RunState()
        state.apply(approval_pending("a"))
        state.apply(approval_pending("b"))
        state.apply(approval_pending("c"))
        state.apply(
            ev("approval.resolved", request_id="a", approved=False, reason="no", outcome="denied")
        )
        state.apply(ev("approval.resolved", request_id="b", approved=None, reason=None))
        state.apply(ev("approval.resolved", request_id="c", approved=False))
        assert [(r.status, r.reason) for r in state.approvals] == [
            ("denied", "no"),
            ("expired", None),
            ("denied", None),
        ]

    def test_deadline_from_the_timeout_and_duplicates(self) -> None:
        state = RunState()
        event = approval_pending("x")
        del event["deadline"]
        state.apply(event)
        state.apply(approval_pending("x"))  # a replay of the same request
        (row,) = list(state.approvals)
        assert row.deadline == pytest.approx(row.requested_at + 300.0)

    def test_resolved_without_a_request(self) -> None:
        state = RunState()
        state.apply(ev("approval.resolved", request_id="ghost", approved=True, outcome="approved"))
        (row,) = list(state.approvals)
        assert row.tool == "?" and row.status == "approved"

    def test_exit_expires_pending_requests(self) -> None:
        state = RunState()
        state.apply(run_start())
        state.apply(approval_pending())
        state.apply(ev("run.exit", code=0, error=None, traceback=None, duration_s=1.0))
        (row,) = list(state.approvals)
        assert row.status == "expired" and row.reason == "the program exited"
        assert state.approvals_awaiting == 0

    def test_malformed_approvals(self) -> None:
        state = RunState()
        assert state.apply(ev("approval.pending", agent="bot", request="nope")) == set()
        assert state.apply(ev("approval.resolved")) == set()
        state.apply(ev("approval.pending", request={"request_id": "r"}, deadline="soon"))
        (row,) = list(state.approvals)
        assert row.tool == "?" and row.deadline is None and row.time_left(0) is None


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


class TestRobustness:
    @pytest.mark.parametrize(
        "event",
        [
            None,
            "text",
            42,
            [],
            {},
            {"kind": None},
            {"kind": "something.new", "x": 1},
        ],
    )
    def test_ignored(self, event: Any) -> None:
        state = RunState()
        assert state.apply(event) == set()
        assert state.ignored_count == 1 and state.event_count == 0

    @pytest.mark.parametrize(
        "event",
        [
            {"kind": "run.start", "mode": 5, "argv": "x", "flags": [1], "env": "x"},
            {"kind": "run.exit", "code": "x", "duration_s": "y"},
            {"kind": "state", "state": None},
            {"kind": "agent.built", "agent": None, "tools": "x", "servers": [1, None, {}]},
            {"kind": "agent.built", "tools": [{"name": None}, "x"], "features": [1]},
            {"kind": "timeline", "entry": 5},
            {"kind": "timeline", "entry": {"event_type": "tool.call", "metadata": [1, 2]}},
            {
                "kind": "timeline",
                "entry": {
                    "event_type": "llm.end",
                    "duration": "fast",
                    "metadata": {"total_tokens": "x"},
                },
            },
            {"kind": "timeline", "entry": {"event_type": "tool.result", "metadata": {"run_id": 7}}},
            {"kind": "event", "event": "x"},
            {"kind": "stats", "totals": None},
            {"kind": "turn.start", "turn_id": None, "input": 5},
            {"kind": "stream", "turn_id": 1, "event": None},
            {"kind": "stream", "turn_id": "t", "event": {"type": "tool_end", "success": "yes"}},
            {"kind": "turn.end", "turn_id": "t", "duration_ms": "x", "output": 5},
            {"kind": "http", "status": float("nan"), "duration_ms": float("inf")},
            {"kind": "log", "level": 10, "message": {"a": 1}},
            {"kind": "output", "stream": None, "text": None},
            {"kind": "approval.pending", "request": {"request_id": 1, "timestamp": "x"}},
            {"kind": "approval.resolved", "request_id": 1, "approved": "maybe", "outcome": 3},
            {"kind": "run.start", "seq": "x", "ts": "y"},
        ],
    )
    def test_malformed_events_never_raise(self, event: dict[str, Any]) -> None:
        state = RunState()
        state.apply(event)
        state.header_fields()
        _ = state.stats
        _ = state.pending_approvals

    def test_pop_changes_resets(self) -> None:
        state = RunState()
        state.apply(run_start())
        assert state.pop_changes()
        assert not state.pop_changes()
        assert state.last_seq > 0
