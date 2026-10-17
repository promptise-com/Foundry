"""Tests for the ``promptise run --tui`` terminal UI (``promptise.run.tui``).

The Textual app is driven headlessly with ``App.run_test`` over a scripted
:class:`~promptise.run.testing.FakeRunSource`: events are pushed with
``fake.emit`` and what the UI sent back is read from ``fake.sent``.
"""

from __future__ import annotations

import asyncio
import io
import time
from typing import Any

import pytest
from rich.console import Console

from promptise.run.testing import FakeRunSource
from promptise.run.tui import run_tui
from promptise.run.tui.app import TAB_IDS, RunTUI
from promptise.run.tui.panes import ChatPane, LogsPane, RunPane
from promptise.run.tui.state import CAP
from promptise.run.tui.widgets import FeedTable, HelpScreen, ReasonScreen

SIZE = (140, 45)


@pytest.fixture(autouse=True)
def _patient_pilot(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a bare ``pilot.pause()`` sturdy on slow CI runners (as in the mcpcast tests).

    Textual's ``pause()`` waits for one idle cycle; a message posted from a
    worker or a timer can land just after it returns.  A bare pause also gives
    late messages 50 ms and idles again.  An explicit ``pause(delay)`` is
    unchanged.
    """
    from textual.pilot import Pilot

    original = Pilot.pause

    async def pause(self: Pilot, delay: float | None = None) -> None:
        if delay is not None:
            await original(self, delay)
            return
        await original(self)
        await asyncio.sleep(0.05)
        await original(self)

    monkeypatch.setattr(Pilot, "pause", pause)


@pytest.fixture(autouse=True)
def _no_animations(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switch tabs without the underline animation, which ``pilot.press`` waits for."""
    from textual import constants

    monkeypatch.setattr(constants, "TEXTUAL_ANIMATIONS", "none")


async def settle(pilot: Any) -> None:
    """Let the workers move commands and events, apply them now, and let the UI draw."""
    app: RunTUI = pilot.app
    for _ in range(3):
        await asyncio.sleep(0.01)
        app.flush_events()
    await pilot.pause()


async def submit(pilot: Any, text: str) -> None:
    """Put *text* in the input and press enter (typing it key by key is slow)."""
    pilot.app.query_one("#prompt").value = text
    await pilot.press("enter")
    await settle(pilot)


def plain(renderable: Any, width: int = 160) -> str:
    """A Rich renderable (a Static's content) as plain text."""
    if hasattr(renderable, "plain"):
        return str(renderable.plain)
    console = Console(width=width, record=True, file=io.StringIO(), color_system=None)
    console.print(renderable)
    return console.export_text()


def static_text(app: RunTUI, selector: str) -> str:
    from textual.widgets import Static

    return plain(app.screen.query_one(selector, Static).content)


def cells(table: Any, row: int) -> list[str]:
    return [str(c) for c in table.get_row_at(row)]


def notifications(app: RunTUI) -> list[str]:
    return [n.message for n in app._notifications]


def start(fake: FakeRunSource, mode: str = "chat", **extra: Any) -> None:
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
        "argv": ["main.py", "--fast"],
        "flags": {"devtools": True, "prod": False, "tui": True, "watch": False},
        "model_override": None,
        "config": {"name": "support-bot", "main": "main.py", "env": {"files": [".env"]}},
        "env": [
            {"name": "OPENAI_API_KEY", "set": True, "required": True},
            {"name": "SLACK_TOKEN", "set": False, "required": False},
        ],
        "scripts": {"dev": "promptise run --devtools --watch"},
    }
    payload.update(extra)
    fake.emit("run.start", **payload)


def built(fake: FakeRunSource, agent: str = "bot") -> None:
    fake.emit(
        "agent.built",
        agent=agent,
        model="openai:gpt-test",
        observe_level="full",
        tools=[
            {"name": "search_docs", "description": "Search the docs", "server": "docs"},
            {"name": "send_email", "description": "Send an email", "server": "mail"},
        ],
        servers=[
            {"name": "docs", "transport": "http", "target": "http://localhost:8000/mcp"},
            {"name": "mail", "transport": "stdio", "target": "python mail.py"},
        ],
        features={"memory": True, "cache": False, "guardrails": "pii", "approval": True},
        instructions="Answer support tickets politely.",
    )


def entry(fake: FakeRunSource, agent: str, event_type: str, **fields: Any) -> None:
    fake.emit(
        "timeline",
        agent=agent,
        entry={
            "entry_id": f"e{time.monotonic_ns()}",
            "timestamp": time.time(),
            "event_type": event_type,
            "category": "x",
            "agent_id": agent,
            "details": fields.pop("details", event_type),
            "duration": fields.pop("duration", None),
            "parent_id": None,
            "metadata": fields.pop("metadata", {}),
        },
    )


async def press_tab(pilot: Any, digit: str) -> None:
    """Show tab *digit* (the digit keys themselves are tested once, below)."""
    await pilot.pause()  # the start-up focus of the input must land first
    pilot.app.action_tab(int(digit) or 10)
    await settle(pilot)
    assert pilot.app.active_tab == TAB_IDS[(int(digit) or 10) - 1]


# ---------------------------------------------------------------------------
# Header, tabs, keys
# ---------------------------------------------------------------------------


class TestHeaderAndTabs:
    async def test_header_shows_the_run(self) -> None:
        fake = FakeRunSource()
        start(fake, "chat")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            header = app.header.plain
            assert "support-bot v0.1.0" in header
            assert "mode chat" in header
            assert "running" in header
            assert "cost —" in header
            assert "LLM calls 0" in header
            fake.emit(
                "stats",
                totals={
                    "llm_calls": 3,
                    "prompt_tokens": 1000,
                    "completion_tokens": 234,
                    "total_tokens": 1234,
                    "cost_usd": 0.0123,
                    "errors": 2,
                },
                agents={},
                models={},
                tools={},
            )
            fake.emit("state", state="busy", detail="")
            await settle(pilot)
            header = app.header.plain
            assert "LLM calls 3" in header
            assert "tokens 1,234" in header
            assert "cost $0.0123" in header
            assert "errors 2" in header
            assert "busy" in header
            assert app.ui_errors == []

    async def test_digits_switch_tabs_when_the_input_is_not_focused(self) -> None:
        fake = FakeRunSource()
        start(fake, "script")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            prompt = app.query_one("#prompt")
            assert app.focused is prompt  # tab 1 starts in the input
            await pilot.press("2")
            assert app.active_tab == "tab-chat"  # typed into the input instead
            assert prompt.value == "2"  # type: ignore[attr-defined]
            await pilot.press("escape")
            assert app.focused is None
            for digit, pane_id in zip("234567890", TAB_IDS[1:], strict=True):
                await pilot.press(digit)
                await pilot.pause()
                assert app.active_tab == pane_id, digit
            await pilot.press("1")
            await pilot.pause()
            assert app.active_tab == "tab-chat"
            assert app.focused is prompt

    async def test_tab_labels_follow_the_mode(self) -> None:
        fake = FakeRunSource()
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            tabs = app.query_one("#tabs")
            assert tabs.get_tab("tab-chat").label.plain == "1 Console"  # type: ignore[attr-defined]
            assert tabs.get_tab("tab-run").label.plain == "0 Run"  # type: ignore[attr-defined]
            start(fake, "chat")
            await settle(pilot)
            assert tabs.get_tab("tab-chat").label.plain == "1 Chat"  # type: ignore[attr-defined]

    async def test_restart_and_stop_keys(self) -> None:
        fake = FakeRunSource()
        start(fake, "script")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            await pilot.press("r")  # typed: the input has the focus
            await pilot.pause()
            assert fake.restarts == 0
            await pilot.press("escape", "r")
            await pilot.pause()
            assert fake.restarts == 1
            await pilot.press("s")
            await pilot.pause()
            assert fake.stops == 1
            assert not fake.closed

    async def test_help_overlay(self) -> None:
        fake = FakeRunSource()
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await pilot.press("escape", "question_mark")
            await pilot.pause()
            assert isinstance(app.screen, HelpScreen)
            assert "approve the highlighted request" in static_text(app, "#help-body")
            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, HelpScreen)

    async def test_quit_closes_the_source_and_returns_the_exit_code(self) -> None:
        fake = FakeRunSource()
        start(fake, "script")
        fake.emit("run.exit", code=4, error=None, traceback=None, duration_s=1.0)
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            await pilot.press("escape", "q")
            await pilot.pause()
        assert fake.closed
        assert app.return_value == 4

    async def test_redraw_key(self) -> None:
        fake = FakeRunSource()
        start(fake, "script")
        fake.emit("output", stream="stdout", text="one")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            await pilot.press("ctrl+l")
            await settle(pilot)
            assert list(app.chat_pane.console_lines) == ["one"]
            assert app.ui_errors == []


async def test_run_tui_returns_the_last_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeRunSource()
    seen: list[RunTUI] = []

    async def fake_run_async(self: RunTUI, **_: Any) -> None:
        seen.append(self)
        assert self.title == "my project"
        self.state.apply({"kind": "run.exit", "code": 7})

    monkeypatch.setattr(RunTUI, "run_async", fake_run_async)
    assert await run_tui(fake, title="my project") == 7
    assert fake.closed and len(seen) == 1


async def test_run_tui_returns_zero_when_no_run_ended(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_async(self: RunTUI, **_: Any) -> None:
        return None

    monkeypatch.setattr(RunTUI, "run_async", fake_run_async)
    fake = FakeRunSource()
    assert await run_tui(fake) == 0
    assert fake.closed


# ---------------------------------------------------------------------------
# 1 Chat / Console
# ---------------------------------------------------------------------------


class TestChat:
    async def test_a_message_is_sent_and_the_reply_streams_in(self) -> None:
        fake = FakeRunSource(echo=True)
        start(fake, "chat")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            pane = app.query_one(ChatPane)
            assert pane.chat_mode
            await pilot.press(*"hello world", "enter")
            await settle(pilot)
            (command,) = fake.sent
            assert command["cmd"] == "chat" and command["text"] == "hello world"
            turn_id = command["turn_id"]
            assert isinstance(turn_id, str) and len(turn_id) == 32
            int(turn_id, 16)  # uuid4 hex
            text = pane.chat_text()
            assert "❯ hello world" in text
            assert text.count("hello world") == 2  # the question and the echoed answer
            assert "thinking" not in text  # the turn ended
            assert app.query_one("#prompt").value == ""  # type: ignore[attr-defined]

    async def test_tool_cards_and_errors_show_inline(self) -> None:
        fake = FakeRunSource()
        start(fake, "chat")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            fake.emit("turn.start", turn_id="t1", input="where is order 7?", conversation_id="c")
            fake.emit("stream", turn_id="t1", event={"type": "token", "text": "Checking. "})
            fake.emit(
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
            await settle(pilot)
            text = app.chat_pane.chat_text()
            assert "⚙ Looking up order (lookup_order)" in text
            assert '{"id": 7}' in text and "running" in text
            fake.emit(
                "stream",
                turn_id="t1",
                event={
                    "type": "tool_end",
                    "tool_name": "lookup_order",
                    "tool_summary": "shipped yesterday",
                    "duration_ms": 31.0,
                    "success": True,
                    "tool_index": 0,
                    "reused": False,
                },
            )
            fake.emit("stream", turn_id="t1", event={"type": "error", "message": "rate limited"})
            fake.emit("turn.end", turn_id="t1", output=None, duration_ms=1500.0, error="failed")
            await settle(pilot)
            text = app.chat_pane.chat_text()
            assert "✓ 31 ms" in text
            assert "↳ shipped yesterday" in text
            assert "✗ rate limited" in text
            assert "✗ failed" in text
            assert "1.5s" in text

    async def test_slash_commands(self) -> None:
        fake = FakeRunSource()
        start(fake, "chat")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            await submit(pilot, "/reset")
            assert fake.sent == [{"cmd": "reset"}]
            assert "new conversation" in app.chat_pane.chat_text()
            await submit(pilot, "slow question")
            turn_id = fake.sent[-1]["turn_id"]
            await submit(pilot, "/cancel")
            assert fake.sent[-1] == {"cmd": "cancel", "turn_id": turn_id}
            assert "cancel requested" in app.chat_pane.chat_text()
            await submit(pilot, "   ")  # blank: nothing sent
            await submit(pilot, "/help")
            assert len(fake.sent) == 3
            assert "/reset starts a new conversation" in app.chat_pane.chat_text()


class TestConsole:
    async def test_output_and_stdin_in_script_mode(self) -> None:
        fake = FakeRunSource()
        start(fake, "script")
        fake.emit("output", stream="stdout", text="What is your name?")
        fake.emit("output", stream="stderr", text="DeprecationWarning: old")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            pane = app.chat_pane
            assert not pane.chat_mode
            assert list(pane.console_lines) == ["What is your name?", "DeprecationWarning: old"]
            await submit(pilot, "Ann")
            assert fake.sent == [{"cmd": "stdin", "text": "Ann"}]
            assert list(pane.console_lines)[-1] == "❯ Ann"
            await pilot.pause(0.1)
            lines = [strip.text for strip in app.query_one("#console").lines]  # type: ignore[attr-defined]
            assert any("What is your name?" in line for line in lines)

    async def test_restart_marks_the_new_run(self) -> None:
        fake = FakeRunSource()
        start(fake, "script")
        built(fake, "first")
        fake.emit("output", stream="stdout", text="run one")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            await pilot.press("escape", "r")
            await pilot.pause()
            assert fake.restarts == 1
            # What the supervisor sends on a restart:
            fake.emit("state", state="restarting", detail="")
            fake.emit("run.exit", code=0, error=None, traceback=None, duration_s=2.0)
            await settle(pilot)
            start(fake, "script")
            built(fake, "second")
            fake.emit("output", stream="stdout", text="run two")
            await settle(pilot)
            assert "run #2" in app.header.plain
            assert "running" in app.header.plain
            lines = list(app.chat_pane.console_lines)
            assert lines[0] == "run one"
            assert any("exited with code 0" in line for line in lines)
            assert "── run #2 ──" in lines
            assert lines[-1] == "run two"
            assert list(app.state.agents) == ["second"]
            await press_tab(pilot, "7")
            agents = static_text(app, "#agents-body")
            assert "second" in agents and "first" not in agents
            await press_tab(pilot, "0")
            run_text = static_text(app, "#run-body")
            assert "Run #2" in run_text
            assert "#1" in run_text and "exited code 0" in run_text

    async def test_exit_is_shown(self) -> None:
        fake = FakeRunSource()
        start(fake, "script")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            fake.emit(
                "run.exit",
                code=3,
                error="ValueError: boom",
                traceback='Traceback (most recent call last):\n  File "main.py"\nValueError: boom',
                duration_s=1.5,
            )
            await settle(pilot)
            assert "exited code 3" in app.header.plain
            assert any(
                "exited with code 3 after 1.5s · ValueError: boom" in line
                for line in app.chat_pane.console_lines
            )
            await press_tab(pilot, "0")
            text = static_text(app, "#run-body")
            assert "ValueError: boom" in text
            assert "Traceback (most recent call last)" in text
            assert "code" in text and "3" in text


# ---------------------------------------------------------------------------
# 2-5 Trace, Tools, Metrics, Events
# ---------------------------------------------------------------------------


class TestTables:
    async def test_trace_rows_detail_and_agent_filter(self) -> None:
        fake = FakeRunSource()
        start(fake, "chat")
        entry(fake, "bot", "agent.input", details="hello")
        entry(
            fake,
            "bot",
            "llm.end",
            duration=0.25,
            metadata={
                "model": "m",
                "prompt_tokens": 20,
                "completion_tokens": 5,
                "total_tokens": 25,
            },
        )
        entry(fake, "helper", "agent.output", details="done")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await press_tab(pilot, "2")
            table = app.query_one("#trace-table", FeedTable)
            assert table.row_count == 3
            row = cells(table, 1)
            assert row[1:] == ["bot", "llm.end", "250", "25", "llm.end"]
            assert app.focused is table
            await pilot.press("down")
            await pilot.pause()
            detail = static_text(app, "#trace-detail .detail-body")
            assert '"event_type": "llm.end"' in detail
            assert '"total_tokens": 25' in detail
            await pilot.press("a")  # agent filter: bot
            await pilot.pause()
            assert table.row_count == 2
            await pilot.press("a")  # helper
            await pilot.pause()
            assert table.row_count == 1
            entry(fake, "bot", "agent.input")
            entry(fake, "helper", "agent.input")
            await settle(pilot)
            assert table.row_count == 2  # only helper's rows are added
            await pilot.press("a")  # all again
            await pilot.pause()
            assert table.row_count == 5

    async def test_hidden_tabs_catch_up_when_shown(self) -> None:
        fake = FakeRunSource()
        start(fake, "chat")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            for _ in range(5):
                entry(fake, "bot", "agent.input")
            await settle(pilot)
            table = app.query_one("#trace-table", FeedTable)
            assert table.row_count == 0  # not drawn while hidden
            await press_tab(pilot, "2")
            assert table.row_count == 5

    async def test_tool_calls_are_paired_with_results(self) -> None:
        fake = FakeRunSource()
        start(fake, "chat")
        built(fake)
        entry(
            fake,
            "bot",
            "tool.call",
            metadata={"run_id": "r1", "tool_name": "search_docs", "arguments": '{"q": "refund"}'},
        )
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await press_tab(pilot, "3")
            table = app.query_one("#tool-calls", FeedTable)
            assert table.row_count == 1
            assert cells(table, 0)[2:] == ["search_docs", "running", "", '{"q": "refund"}']
            detail = static_text(app, "#tool-detail .detail-body")
            assert '"q": "refund"' in detail and "running" in detail
            entry(
                fake,
                "bot",
                "tool.result",
                duration=0.12,
                metadata={
                    "run_id": "r1",
                    "tool_name": "search_docs",
                    "result_preview": "3 articles found",
                    "latency_ms": 120.0,
                },
            )
            entry(
                fake,
                "bot",
                "tool.error",
                metadata={
                    "run_id": "r9",
                    "tool_name": "send_email",
                    "error_type": "PermissionError",
                    "error": "no",
                },
            )
            await settle(pilot)
            assert table.row_count == 2  # the result updated its call; the orphan error got a row
            assert cells(table, 0)[2:5] == ["search_docs", "ok", "120"]
            assert cells(table, 1)[2:4] == ["send_email", "error"]
            detail = static_text(app, "#tool-detail .detail-body")
            assert "3 articles found" in detail  # the highlighted call's detail followed
            available = app.query_one("#tools-available")
            assert available.row_count == 2  # type: ignore[attr-defined]
            assert cells(available, 0)[:3] == ["bot", "search_docs", "docs"]

    async def test_metrics(self) -> None:
        fake = FakeRunSource()
        start(fake, "chat")
        for tokens in (10, 30, 20):
            entry(
                fake,
                "bot",
                "llm.end",
                duration=0.1,
                metadata={"model": "m", "total_tokens": tokens},
            )
        fake.emit(
            "stats",
            totals={
                "llm_calls": 3,
                "prompt_tokens": 40,
                "completion_tokens": 20,
                "total_tokens": 60,
                "tool_calls": 2,
                "tool_errors": 1,
                "errors": 1,
                "retries": 0,
                "cost_usd": 0.5,
                "unpriced_models": ["local-llm"],
                "latency_p50_ms": 100.0,
                "latency_p95_ms": 180.0,
            },
            agents={
                "bot": {"model": "m", "llm_calls": 3, "total_tokens": 60, "tool_calls": 2},
                "helper": {"model": None, "llm_calls": 0},
            },
            models={"m": {"llm_calls": 3, "prompt_tokens": 40, "completion_tokens": 20}},
            tools={"search": {"calls": 2, "errors": 1, "total_ms": 30.0, "avg_ms": 15.0}},
        )
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await press_tab(pilot, "4")
            totals = static_text(app, "#metrics-totals")
            assert "$0.5000" in totals and "100 / 180 ms" in totals
            assert "No [pricing] for: local-llm" in totals
            assert app.query_one("#metrics-agents").row_count == 2  # type: ignore[attr-defined]
            assert app.query_one("#metrics-models").row_count == 1  # type: ignore[attr-defined]
            assert app.query_one("#metrics-tools").row_count == 1  # type: ignore[attr-defined]
            assert cells(app.query_one("#metrics-tools"), 0) == ["search", "2", "1", "15", "30"]
            spark = app.query_one("#metrics-spark")
            assert list(spark.data) == [10.0, 30.0, 20.0]  # type: ignore[attr-defined]

    async def test_events(self) -> None:
        fake = FakeRunSource()
        start(fake, "chat")
        fake.emit(
            "event",
            agent="bot",
            event={
                "event_type": "tool.slow",
                "severity": "warning",
                "timestamp": time.time(),
                "agent_id": "bot",
                "data": {"tool": "search", "ms": 9000},
            },
        )
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await press_tab(pilot, "5")
            table = app.query_one("#events-table", FeedTable)
            assert table.row_count == 1
            assert cells(table, 0)[1:4] == ["bot", "tool.slow", "warning"]
            assert '"ms": 9000' in static_text(app, "#events-detail .detail-body")

    async def test_rows_are_capped(self) -> None:
        fake = FakeRunSource()
        start(fake, "chat")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await press_tab(pilot, "2")
            for _ in range(CAP + 1_000):
                entry(fake, "bot", "agent.input")
            await settle(pilot)
            table = app.query_one("#trace-table", FeedTable)
            assert len(app.state.timeline) == CAP
            assert 0 < table.row_count <= CAP
            assert app.ui_errors == []


# ---------------------------------------------------------------------------
# 6 Approvals
# ---------------------------------------------------------------------------


def approval(fake: FakeRunSource, request_id: str, tool: str = "send_email") -> None:
    now = time.time()
    fake.emit(
        "approval.pending",
        agent="bot",
        request={
            "request_id": request_id,
            "tool_name": tool,
            "arguments": {"to": "ann@example.com", "subject": "Refund"},
            "agent_id": "bot",
            "caller_user_id": "u1",
            "context_summary": "user: please email Ann about the refund",
            "timestamp": now,
            "timeout": 300.0,
            "metadata": {},
        },
        deadline=now + 90,
    )


class TestApprovals:
    async def test_approve(self) -> None:
        fake = FakeRunSource()
        start(fake, "chat")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await settle(pilot)
            approval(fake, "req-1")
            await settle(pilot)
            assert "✋ 1 approval" in app.header.plain
            assert any("bot wants to call send_email" in m for m in notifications(app))
            await press_tab(pilot, "6")
            pending = app.query_one("#approvals-pending")
            assert pending.row_count == 1  # type: ignore[attr-defined]
            row = cells(pending, 0)
            assert row[1:3] == ["bot", "send_email"]
            assert row[3].endswith("s")  # time left, e.g. "1m 29s"
            assert "ann@example.com" in row[4]
            detail = static_text(app, "#approvals-detail .detail-body")
            assert "please email Ann about the refund" in detail
            assert '"subject": "Refund"' in detail
            await pilot.press("a")
            await settle(pilot)
            assert fake.sent == [
                {"cmd": "approval", "request_id": "req-1", "approved": True, "reason": None}
            ]
            assert "✋" not in app.header.plain
            assert cells(pending, 0)[3] == "approving…"
            fake.emit(
                "approval.resolved",
                request_id="req-1",
                approved=True,
                reason=None,
                outcome="approved",
            )
            await settle(pilot)
            assert pending.row_count == 0  # type: ignore[attr-defined]
            history = app.query_one("#approvals-history")
            assert history.row_count == 1  # type: ignore[attr-defined]
            assert cells(history, 0)[1:4] == ["bot", "send_email", "approved"]
            await pilot.press("a")  # nothing pending: nothing sent
            await settle(pilot)
            assert len(fake.sent) == 1

    async def test_deny_with_a_reason(self) -> None:
        fake = FakeRunSource()
        start(fake, "chat")
        approval(fake, "req-1")
        approval(fake, "req-2", tool="delete_account")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await press_tab(pilot, "6")
            assert "✋ 2 approvals" in app.header.plain
            await pilot.press("down")  # the second request
            await pilot.press("d")
            await pilot.pause()
            assert isinstance(app.screen, ReasonScreen)
            await pilot.press("escape")  # cancel: nothing sent
            await pilot.pause()
            assert not isinstance(app.screen, ReasonScreen)
            assert fake.sent == []
            await pilot.press("d")
            await pilot.pause()
            app.screen.query_one("#reason-input").value = "too risky"  # type: ignore[attr-defined]
            await pilot.press("enter")
            await settle(pilot)
            assert fake.sent == [
                {
                    "cmd": "approval",
                    "request_id": "req-2",
                    "approved": False,
                    "reason": "too risky",
                }
            ]
            assert "✋ 1 approval" in app.header.plain
            await pilot.press("up", "d")
            await pilot.pause()
            await pilot.press("enter")  # an empty reason is fine
            await settle(pilot)
            assert fake.sent[-1] == {
                "cmd": "approval",
                "request_id": "req-1",
                "approved": False,
                "reason": None,
            }

    async def test_the_countdown_ticks(self) -> None:
        fake = FakeRunSource()
        start(fake, "chat")
        approval(fake, "req-1")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await press_tab(pilot, "6")
            row = app.state.pending_approvals[0]
            row.deadline = time.time() + 5  # about to expire
            app.query_one("#tab-approvals").tick()  # type: ignore[attr-defined]
            await pilot.pause()
            assert cells(app.query_one("#approvals-pending"), 0)[3] in ("5s", "4s")
            row.deadline = time.time() - 1
            app.query_one("#tab-approvals").tick()  # type: ignore[attr-defined]
            await pilot.pause()
            assert cells(app.query_one("#approvals-pending"), 0)[3] == "expiring"


# ---------------------------------------------------------------------------
# 7-0 Agents, MCP, Logs, Run
# ---------------------------------------------------------------------------


class TestInfoTabs:
    async def test_agents_and_mcp(self) -> None:
        fake = FakeRunSource()
        start(fake, "chat")
        built(fake)
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await press_tab(pilot, "7")
            text = static_text(app, "#agents-body")
            assert "openai:gpt-test" in text
            assert "✓ memory" in text and "✗ cache" in text and "✓ guardrails (pii)" in text
            assert "docs (http)" in text
            assert "Answer support tickets politely." in text
            await press_tab(pilot, "8")
            table = app.query_one("#mcp-table")
            assert table.row_count == 2  # type: ignore[attr-defined]
            assert cells(table, 0) == ["bot", "docs", "http", "http://localhost:8000/mcp", "1"]
            assert not app.query_one("#mcp-empty").display

    async def test_logs_and_the_level_filter(self) -> None:
        fake = FakeRunSource()
        start(fake, "script")
        fake.emit("log", logger="app", level="DEBUG", message="details", exc_text=None)
        fake.emit("log", logger="app", level="INFO", message="started", exc_text=None)
        fake.emit("log", logger="app", level="ERROR", message="db down", exc_text="Traceback x")
        fake.emit("output", stream="stderr", text="a warning line")
        fake.emit("output", stream="stdout", text="not a log line")
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await press_tab(pilot, "9")
            pane = app.query_one(LogsPane)
            assert pane.shown == 4
            await pilot.press("l")  # info+
            await pilot.pause()
            assert pane.shown == 3
            await pilot.press("l")  # warning+
            await pilot.pause()
            assert pane.shown == 2
            await pilot.press("l")  # error+
            await pilot.pause()
            assert pane.shown == 1
            assert "Level: error+" in str(app.query_one("#logs-level").label)  # type: ignore[attr-defined]
            await pilot.press("l")  # all
            await pilot.pause()
            assert pane.shown == 4

    async def test_run_tab(self) -> None:
        fake = FakeRunSource()
        start(fake, "serve", model_override="openai:gpt-5-mini")
        fake.emit(
            "http",
            method="POST",
            path="/v1/invoke",
            status=200,
            duration_ms=12.0,
            request_id="req-a",
        )
        fake.emit(
            "http",
            method="GET",
            path="/readyz",
            status=503,
            duration_ms=1.0,
            request_id="req-b",
        )
        app = RunTUI(fake)
        async with app.run_test(size=SIZE) as pilot:
            await press_tab(pilot, "0")
            text = static_text(app, "#run-body")
            for expected in (
                "support-bot v0.1.0",
                "4242",
                "3.12.4",
                "1.4.0",
                "main.py --fast",
                "devtools, tui",
                "openai:gpt-5-mini",
                "OPENAI_API_KEY",
                "SLACK_TOKEN",
                "promptise run --devtools --watch",
                '"files"',
            ):
                assert expected in text, expected
            http = app.query_one("#run-http", FeedTable)
            assert http.display and http.row_count == 2
            assert cells(http, 1)[1:4] == ["GET", "/readyz", "503"]
            assert isinstance(app.query_one("#tab-run"), RunPane)


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


class _BrokenSource(FakeRunSource):
    async def send(self, command: dict[str, Any]) -> None:
        raise ConnectionError("pipe closed")

    async def restart(self) -> None:
        raise RuntimeError("cannot restart")


async def test_bad_events_do_not_crash_the_ui() -> None:
    fake = FakeRunSource()
    fake._history.append("not a dict")  # type: ignore[arg-type]
    start(fake, "chat")
    app = RunTUI(fake)
    async with app.run_test(size=SIZE) as pilot:
        for payload in (
            {"entry": 5},
            {"entry": {"event_type": None, "metadata": "x"}},
        ):
            fake.emit("timeline", **payload)
        fake.emit("brand.new.kind", x=1)
        fake.emit("agent.built", agent=None, tools="nope", servers=[None])
        fake.emit("stream", turn_id=None, event="x")
        fake.emit("stats", totals=[1])
        fake.emit("approval.pending", request={"request_id": "r", "arguments": object()})
        fake.emit("output", stream=None, text=None)
        await pilot.pause()
        for number in range(1, len(TAB_IDS) + 1):  # every pane draws the bad data
            app.action_tab(number)
            await pilot.pause()
            app.flush_events()
            assert app.active_tab == TAB_IDS[number - 1]
        assert app.ui_errors == []
        assert app.is_running


async def test_failing_commands_are_reported_not_raised() -> None:
    fake = _BrokenSource()
    start(fake, "chat")
    app = RunTUI(fake)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        await submit(pilot, "hi")
        await pilot.press("escape", "r")
        await settle(pilot)
        assert app.is_running
        assert any("pipe closed" in e for e in app.ui_errors)
        assert any("cannot restart" in e for e in app.ui_errors)
        assert any("Could not" in m for m in notifications(app))


async def test_switching_tabs_at_start_up_does_not_ping_pong(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale tab activation must not focus its pane (that re-activates it, forever)."""
    calls: list[str] = []
    original = RunTUI._activate

    def counting(self: RunTUI, pane: Any, *, retry: bool = True) -> None:
        calls.append(str(pane.id))
        original(self, pane, retry=retry)

    monkeypatch.setattr(RunTUI, "_activate", counting)
    fake = FakeRunSource()
    start(fake, "chat")
    app = RunTUI(fake)
    async with app.run_test(size=SIZE) as pilot:
        app.action_tab(7)  # before the start-up focus of the input has landed
        await pilot.pause(0.3)
        settled = len(calls)
        await pilot.pause(0.3)
        assert len(calls) == settled <= 4


async def test_a_burst_of_output_is_drawn_over_several_ticks() -> None:
    fake = FakeRunSource()
    start(fake, "script")
    for i in range(2_500):
        fake.emit("output", stream="stdout", text=f"line {i}")
    app = RunTUI(fake)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        pane = app.chat_pane
        assert 0 < len(pane.console_lines) <= 2_500
        for _ in range(10):
            if not pane.behind():
                break
            app.flush_events()
            await pilot.pause()
        assert len(pane.console_lines) == 2_500
        assert pane.console_lines[-1] == "line 2499"
        assert not pane.dirty


async def test_untrusted_text_is_never_markup() -> None:
    """Tool output, approval arguments and HTTP paths are shown as text, not parsed as markup."""
    fake = FakeRunSource()
    fake.emit("run.start", project="p", mode="chat", main="m.py", pid=1)
    fake.emit("agent.built", agent="a", model="m", tools=[], servers=[], features={})
    evil = "[/INST] [@click=app.quit]click me[/] [link=https://evil.example]x[/link] \x1b]0;t\x07"
    fake.emit(
        "timeline",
        agent="a",
        entry={
            "entry_id": "e1",
            "timestamp": time.time(),
            "event_type": "tool.result",
            "details": evil,
            "metadata": {"tool_name": "t", "result_preview": evil, "run_id": "r1"},
        },
    )
    fake.emit(
        "http",
        method="GET",
        path="/[/x][@click=app.quit]y[/]",
        status=404,
        duration_ms=1.0,
        request_id="q",
    )
    fake.emit(
        "approval.pending",
        agent="a",
        request={"request_id": "r1", "tool_name": "t", "arguments": {"u": evil}},
        deadline=time.time() + 60,
    )
    app = RunTUI(fake)
    async with app.run_test(size=SIZE) as pilot:
        await settle(pilot)
        for key in ("escape", "2", "3", "5", "6", "0"):
            await pilot.press(key)
            await settle(pilot)
        # Clicking a row must not run an action hidden in its text.
        await pilot.press("6")
        await settle(pilot)
        await pilot.click("#approvals-pending")
        await settle(pilot)
        assert app.is_running
        assert not any(c.get("cmd") == "approval" for c in fake.sent)
