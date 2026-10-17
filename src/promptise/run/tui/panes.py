"""The ten tabs of the ``promptise run`` terminal UI.

Every pane reads from the app's :class:`~promptise.run.tui.state.RunState`.
The app tells each pane what changed (:meth:`Pane.note`) after every batch
of events, and only the visible pane redraws (:meth:`Pane.sync`); a hidden
pane catches up when it is shown, so a busy program costs only the work of
the tab you look at.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, ClassVar

from rich.console import Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import (
    Button,
    ContentSwitcher,
    DataTable,
    Input,
    Label,
    RichLog,
    Sparkline,
    Static,
    TabPane,
)

from . import state as S
from .state import (
    AgentEventRow,
    AgentInfo,
    ApprovalRow,
    Changes,
    ChatError,
    ChatNote,
    ChatText,
    ChatTurn,
    ConsoleLine,
    HttpRequest,
    LogLine,
    RunState,
    TimelineRow,
    ToolCall,
    ToolCard,
    format_cost,
    format_duration,
    preview,
)
from .theme import (
    ACCENT,
    DIM,
    ERROR,
    MUTED,
    PRIMARY,
    SUCCESS,
    TEXT,
    WARNING,
    level_style,
    severity_style,
    status_style,
)
from .widgets import DetailView, FeedTable, FitTable, clock, json_view, pretty_arguments

__all__ = [
    "AgentsPane",
    "ApprovalsPane",
    "ChatPane",
    "EventsPane",
    "LogsPane",
    "McpPane",
    "MetricsPane",
    "Pane",
    "RunPane",
    "ToolsPane",
    "TracePane",
    "render_chat_item",
]


WRITE_BUDGET = 1_000
"""The most lines a log view writes per batch; a burst is drawn over the next ticks."""

MOUNT_BUDGET = 200
"""The most chat items mounted per batch."""


def _ms(value: float | None) -> str:
    if value is None:
        return ""
    if value >= 10_000:
        return f"{value / 1000:.1f}s"
    return f"{value:.0f}"


def _num(value: Any) -> str:
    if isinstance(value, bool) or value is None:
        return "0"
    if isinstance(value, float):
        return f"{value:,.1f}"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _cost(value: Any) -> str:
    return format_cost(value if isinstance(value, (int, float)) else None)


class Pane(TabPane):
    """A tab that redraws from the run state when it is visible.

    Subclasses list the state sections they show in :attr:`SECTIONS` and
    implement :meth:`update_view` (incremental) and, when a full redraw
    differs, :meth:`rebuild`.
    """

    SECTIONS: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, title: str, state: RunState, *, id: str) -> None:
        super().__init__(title, id=id)
        self.state = state
        self.dirty = True
        """Data this pane shows changed since it last drew."""
        self.needs_rebuild = True
        """The next :meth:`sync` redraws everything (first draw, ``ctrl+l``)."""

    def note(self, changes: Changes) -> None:
        """Remember what changed (called after every batch, visible or not)."""
        if changes.sections & self.SECTIONS:
            self.dirty = True

    def sync(self) -> bool:
        """Bring the pane up to date with the state.

        Returns ``False`` when the pane's widgets are not mounted yet (a tab
        shown right at startup); it stays dirty and is drawn on the next try.
        """
        if not self.is_mounted:
            return False
        try:
            if self.needs_rebuild:
                self.rebuild()
                self.needs_rebuild = False
            else:
                self.update_view()
        except NoMatches:
            return False
        self.dirty = self.behind()
        return True

    def behind(self) -> bool:
        """Whether drawing stopped short (a write budget ran out) and must go on."""
        return False

    def rebuild(self) -> None:
        """Redraw everything (defaults to :meth:`update_view`)."""
        self.update_view()

    def update_view(self) -> None:
        """Draw what changed."""

    def focus_main(self) -> None:
        """Focus the pane's main widget (called when the tab is shown)."""


# ---------------------------------------------------------------------------
# 1 Chat / Console
# ---------------------------------------------------------------------------


def _card_text(card: ToolCard) -> Text:
    t = Text("  ⚙ ", style=ACCENT)
    t.append(card.display_name or card.name, style=f"bold {ACCENT}")
    if card.display_name and card.display_name != card.name:
        t.append(f" ({card.name})", style=MUTED)
    args = preview(card.arguments, 100)
    if args and args != "{}":
        t.append(f"  {args}", style=DIM)
    if card.success is None:
        t.append("  … running", style=ACCENT)
    elif card.success:
        t.append("  ✓", style=SUCCESS)
    else:
        t.append("  ✗ failed", style=ERROR)
    if card.duration_ms is not None:
        t.append(f" {_ms(card.duration_ms)} ms", style=MUTED)
    if card.reused:
        t.append(" (reused)", style=MUTED)
    if card.summary:
        t.append(f"\n    ↳ {preview(card.summary, 300)}", style=DIM)
    return t


def render_chat_item(item: ChatTurn | ChatNote) -> Text:
    """One chat item as styled text: the question, streamed answer, tool cards, errors."""
    if isinstance(item, ChatNote):
        style = {"run": f"bold {WARNING}", "error": f"bold {ERROR}", "exit": MUTED}.get(
            item.style, MUTED
        )
        return Text(f"── {item.text} ──", style=style, justify="center")
    t = Text()
    t.append("❯ ", style=f"bold {PRIMARY}")
    t.append(item.input or "…", style=f"bold {TEXT}")
    for part in item.parts:
        t.append("\n")
        if isinstance(part, ChatText):
            t.append(part.text.strip("\n"), style=TEXT)
        elif isinstance(part, ToolCard):
            t.append_text(_card_text(part))
        elif isinstance(part, ChatError):
            t.append(f"✗ {part.message}", style=f"bold {ERROR}")
    if not item.done:
        t.append("\n… thinking" if not item.parts else " …", style=MUTED)
    else:
        footer = []
        if item.duration_ms is not None:
            footer.append(format_duration(item.duration_ms / 1000))
        if footer or item.error:
            t.append("\n")
            if footer:
                t.append(" · ".join(footer), style=MUTED)
            if item.error:
                t.append(("  " if footer else "") + f"✗ {item.error}", style=f"bold {ERROR}")
    return t


def render_console_line(line: ConsoleLine) -> Text:
    """One console line: stdout plain, stderr in the warning colour, typed input, notes."""
    if line.stream == "stderr":
        return Text(line.text, style=WARNING)
    if line.stream == "stdin":
        return Text(f"❯ {line.text}", style=ACCENT)
    if line.stream == "note":
        return Text(f"── {line.text} ──", style=f"bold {MUTED}")
    return Text(line.text)


class ChatItem(Static):
    """One turn or note in the chat log."""

    DEFAULT_CSS = """
    ChatItem { width: 1fr; height: auto; margin: 0 0 1 0; }
    """

    def __init__(self, item: ChatTurn | ChatNote) -> None:
        self.text = render_chat_item(item)
        super().__init__(self.text)
        self.item_id = item.id

    def redraw(self, item: ChatTurn | ChatNote) -> None:
        """Draw *item* again (tokens streamed in)."""
        self.text = render_chat_item(item)
        self.update(self.text)


class ChatPane(Pane):
    """Tab 1: the chat (agent modes) or the program's console (script mode), and the input."""

    SECTIONS = frozenset({S.CHAT, S.CONSOLE, S.RUN})

    DEFAULT_CSS = """
    ChatPane #chat-switcher { height: 1fr; }
    ChatPane #chat { height: 1fr; padding: 0 1; }
    ChatPane #console { height: 1fr; padding: 0 1; background: $background; }
    ChatPane #prompt { margin: 0 0; }
    """

    def __init__(self, title: str, state: RunState, *, id: str) -> None:
        super().__init__(title, state, id=id)
        self.chat_mode = False
        self._chat_last = 0
        self._chat_widgets: dict[int, ChatItem] = {}
        self._chat_order: deque[int] = deque()
        self._chat_updates: set[int] = set()
        self._console_last = 0
        self.console_lines: deque[str] = deque(maxlen=state.cap)
        """The console's lines as plain text (what the log shows)."""

    def compose(self) -> ComposeResult:
        """Chat log and console (one shown at a time) above the input."""
        with ContentSwitcher(initial="console", id="chat-switcher"):
            yield VerticalScroll(id="chat")
            yield RichLog(
                id="console",
                max_lines=self.state.cap,
                wrap=True,
                markup=False,
                highlight=False,
                auto_scroll=True,
            )
        yield Input(placeholder=self._placeholder(), id="prompt")

    def on_mount(self) -> None:
        """Keep the chat scrolled to the newest message."""
        self.query_one("#chat", VerticalScroll).anchor()

    def _placeholder(self) -> str:
        if self.chat_mode:
            return "Message the agent — enter to send · /reset · /cancel · esc for shortcuts"
        return "A line for the program's stdin — enter to send · esc for shortcuts"

    def note(self, changes: Changes) -> None:
        """Remember the turns that changed in place."""
        super().note(changes)
        self._chat_updates |= changes.chat

    def set_mode(self, chat_mode: bool) -> None:
        """Show the chat (agent modes) or the console (script mode)."""
        if chat_mode == self.chat_mode and self.is_mounted:
            self.query_one("#chat-switcher", ContentSwitcher).current = (
                "chat" if chat_mode else "console"
            )
            return
        self.chat_mode = chat_mode
        if not self.is_mounted:
            return
        self.query_one("#chat-switcher", ContentSwitcher).current = (
            "chat" if chat_mode else "console"
        )
        self.query_one("#prompt", Input).placeholder = self._placeholder()
        self.needs_rebuild = True
        self.dirty = True

    def rebuild(self) -> None:
        """Redraw the visible view from scratch."""
        if self.chat_mode:
            for widget in self._chat_widgets.values():
                widget.remove()
            self._chat_widgets.clear()
            self._chat_order.clear()
            self._chat_last = 0
            self._chat_updates.clear()
            self._sync_chat()
        else:
            self.query_one("#console", RichLog).clear()
            self.console_lines.clear()
            self._console_last = 0
            self._sync_console()

    def update_view(self) -> None:
        """Draw new and changed items of the visible view."""
        if self.chat_mode:
            self._sync_chat()
        else:
            self._sync_console()

    def _sync_chat(self) -> None:
        ring = self.state.chat
        log = self.query_one("#chat", VerticalScroll)
        while self._chat_order and self._chat_order[0] < ring.first_id:
            widget = self._chat_widgets.pop(self._chat_order.popleft(), None)
            if widget is not None:
                widget.remove()
        for item_id in sorted(self._chat_updates):
            widget = self._chat_widgets.get(item_id)
            item = ring.get(item_id)
            if widget is not None and item is not None:
                widget.redraw(item)
        self._chat_updates.clear()
        new = ring.since(self._chat_last)[:MOUNT_BUDGET]
        if new:
            self._chat_last = new[-1].id
            widgets = [ChatItem(item) for item in new]
            for widget in widgets:
                self._chat_widgets[widget.item_id] = widget
                self._chat_order.append(widget.item_id)
            log.mount(*widgets)

    def _sync_console(self) -> None:
        ring = self.state.console
        log = self.query_one("#console", RichLog)
        new = ring.since(self._console_last)[:WRITE_BUDGET]
        for line in new:
            text = render_console_line(line)
            log.write(text)
            self.console_lines.append(text.plain)
        if new:
            self._console_last = new[-1].id

    def behind(self) -> bool:
        """The visible view has items it did not draw yet."""
        if self.chat_mode:
            return self._chat_last < self.state.chat.last_id
        return self._console_last < self.state.console.last_id

    def focus_main(self) -> None:
        """Focus the input."""
        self.query_one("#prompt", Input).focus()

    def chat_text(self) -> str:
        """The chat log's text, without styles (for tests)."""
        return "\n".join(
            self._chat_widgets[i].text.plain for i in self._chat_order if i in self._chat_widgets
        )


# ---------------------------------------------------------------------------
# 2 Trace
# ---------------------------------------------------------------------------


def _timeline_cells(row: TimelineRow) -> tuple[Any, ...]:
    etype = row.event_type
    style = (
        ERROR
        if etype.endswith(".error") or etype == "task.failed"
        else ACCENT
        if etype.startswith("tool.")
        else PRIMARY
        if etype.startswith("llm.")
        else DIM
    )
    return (
        Text(clock(row.ts), style=MUTED),
        row.agent,
        Text(etype, style=style),
        _ms(row.duration_ms),
        _num(row.tokens) if row.tokens is not None else "",
        preview(row.details, 160),
    )


class TracePane(Pane):
    """Tab 2: every timeline entry, filterable by agent, with the full entry beside it."""

    SECTIONS = frozenset({S.TIMELINE})

    DEFAULT_CSS = """
    TracePane .bar { height: 1; }
    TracePane .bar Button { margin-right: 2; }
    TracePane .hint { color: $text-muted; width: 1fr; }
    TracePane .split { height: 1fr; }
    TracePane FeedTable { width: 2fr; }
    """

    COLUMNS = [
        ("time", "Time", 12),
        ("agent", "Agent", 14),
        ("event", "Event", 16),
        ("duration", "ms", 7),
        ("tokens", "Tokens", 7),
        ("details", "Details", 12),
    ]

    def __init__(self, title: str, state: RunState, *, id: str) -> None:
        super().__init__(title, state, id=id)
        self.agent_filter: str | None = None

    def compose(self) -> ComposeResult:
        """Filter bar, table and detail."""
        with Horizontal(classes="bar"):
            yield Button("Agent: all", id="trace-agent", compact=True)
            yield Static("a: cycle agents", classes="hint", id="trace-hint")
        with Horizontal(classes="split"):
            yield FeedTable(self.COLUMNS, _timeline_cells, accept=self._accept, id="trace-table")
            yield DetailView(id="trace-detail")

    def _accept(self, row: TimelineRow) -> bool:
        return self.agent_filter is None or row.agent == self.agent_filter

    @property
    def table(self) -> FeedTable:
        """The timeline table."""
        return self.query_one("#trace-table", FeedTable)

    def rebuild(self) -> None:
        """Show the ring again (filter changed)."""
        self.table.rebuild(self.state.timeline)
        self._update_hint()

    def update_view(self) -> None:
        """Append new entries."""
        self.table.sync(self.state.timeline)
        self._update_hint()

    def _update_hint(self) -> None:
        shown = self.table.row_count
        total = len(self.state.timeline)
        self.query_one("#trace-hint", Static).update(
            f"a: cycle agents · {shown:,} shown of {total:,} kept"
        )

    def cycle_agent(self) -> None:
        """Show the next agent's entries only (then all again)."""
        options: list[str | None] = [None, *self.state.timeline_agents]
        try:
            index = options.index(self.agent_filter)
        except ValueError:
            index = 0
        self.agent_filter = options[(index + 1) % len(options)]
        self.query_one("#trace-agent", Button).label = f"Agent: {self.agent_filter or 'all'}"
        self.query_one("#trace-detail", DetailView).reset()
        self.needs_rebuild = True
        self.sync()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """The filter button cycles the agent."""
        if event.button.id == "trace-agent":
            event.stop()
            self.cycle_agent()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show the highlighted entry in full."""
        row = _ring_item(self.state.timeline, event.row_key.value)
        if row is not None:
            body = dict(row.entry)
            body.setdefault("agent", row.agent)
            self.query_one("#trace-detail", DetailView).show(json_view(body), row.id)

    def focus_main(self) -> None:
        """Focus the table."""
        self.table.focus()


def _ring_item(ring: S.Ring[Any], key: str | None) -> Any:
    try:
        return ring.get(int(key)) if key is not None else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 3 Tools
# ---------------------------------------------------------------------------


def _call_cells(call: ToolCall) -> tuple[Any, ...]:
    status = call.status + (" ↺" if call.reused else "")  # ↺: answered from an earlier result
    return (
        Text(clock(call.ts), style=MUTED),
        call.agent,
        Text(call.tool, style=f"bold {ACCENT}"),
        Text(status, style=status_style(call.status)),
        _ms(call.latency_ms),
        preview(call.arguments, 160),
    )


def render_tool_call(call: ToolCall) -> RenderableType:
    """Everything known about one tool call."""
    head = Table.grid(padding=(0, 2))
    head.add_column(style=MUTED)
    head.add_column()
    head.add_row("tool", Text(call.tool, style=f"bold {ACCENT}"))
    head.add_row("agent", call.agent)
    head.add_row("status", Text(call.status, style=status_style(call.status)))
    head.add_row("latency", f"{_ms(call.latency_ms)} ms" if call.latency_ms is not None else "—")
    head.add_row("time", clock(call.ts))
    if call.run_id:
        head.add_row("run id", Text(call.run_id, style=MUTED))
    if call.reused:
        head.add_row("reused", "answered from an earlier result")
    parts: list[RenderableType] = [head, Text("\nArguments", style=f"bold {PRIMARY}")]
    parts.append(pretty_arguments(call.arguments) if call.arguments else Text("—", style=MUTED))
    parts.append(Text("\nResult", style=f"bold {PRIMARY}"))
    if call.result_preview is not None:
        parts.append(Text(call.result_preview))
    elif call.result_length is not None:
        parts.append(Text(f"<{call.result_length} chars, not recorded>", style=MUTED))
    else:
        parts.append(Text("—" if call.status != "running" else "… running", style=MUTED))
    if call.error or call.error_type:
        parts.append(Text("\nError", style=f"bold {ERROR}"))
        label = ": ".join(x for x in (call.error_type, call.error) if x)
        parts.append(Text(label, style=ERROR))
    if call.traceback:
        parts.append(Text(call.traceback, style=MUTED))
    return Group(*parts)


class ToolsPane(Pane):
    """Tab 3: tool calls (arguments, result, latency) and every tool each agent has."""

    SECTIONS = frozenset({S.TOOL_CALLS, S.AGENTS})

    DEFAULT_CSS = """
    ToolsPane .split { height: 1fr; }
    ToolsPane #tool-calls { width: 2fr; }
    ToolsPane .section { color: $primary; text-style: bold; padding: 0 1; height: 1; }
    ToolsPane #tools-available { height: 12; }
    """

    COLUMNS = [
        ("time", "Time", 12),
        ("agent", "Agent", 14),
        ("tool", "Tool", 20),
        ("status", "Status", 9),
        ("latency", "ms", 7),
        ("args", "Arguments", 12),
    ]

    def __init__(self, title: str, state: RunState, *, id: str) -> None:
        super().__init__(title, state, id=id)
        self._updates: set[int] = set()
        self._agents_dirty = True

    def compose(self) -> ComposeResult:
        """Calls table and detail, then the available tools."""
        with Horizontal(classes="split"):
            yield FeedTable(self.COLUMNS, _call_cells, id="tool-calls")
            yield DetailView(id="tool-detail")
        yield Label("Available tools", classes="section", id="tools-available-label")
        yield FitTable(
            [
                ("agent", "Agent", 14),
                ("tool", "Tool", 26),
                ("server", "Server", 14),
                ("description", "Description", 12),
            ],
            id="tools-available",
        )

    def note(self, changes: Changes) -> None:
        """Remember the calls that finished and whether the agents changed."""
        super().note(changes)
        self._updates |= changes.tool_calls
        if S.AGENTS in changes.sections:
            self._agents_dirty = True

    @property
    def table(self) -> FeedTable:
        """The tool calls table."""
        return self.query_one("#tool-calls", FeedTable)

    def rebuild(self) -> None:
        """Redraw everything."""
        self.table.rebuild(self.state.tool_calls)
        self._updates.clear()
        self._agents_dirty = True
        self.update_view()

    def update_view(self) -> None:
        """Append new calls, redraw finished ones, refresh the tools list."""
        ring = self.state.tool_calls
        table = self.table
        table.sync(ring)
        detail = self.query_one("#tool-detail", DetailView)
        for call_id in sorted(self._updates):
            call = ring.get(call_id)
            if call is None:
                continue
            table.refresh_item(call)
            if detail.shown_id == call_id:
                detail.show(render_tool_call(call), call_id)
        self._updates.clear()
        if self._agents_dirty:
            self._agents_dirty = False
            available = self.query_one("#tools-available", FitTable)
            available.ensure_columns()
            available.clear()
            for agent in self.state.agents.values():
                for tool in agent.tools:
                    available.add_row(
                        agent.id,
                        Text(tool.name, style=ACCENT),
                        tool.server or "—",
                        preview(tool.description, 200),
                    )
            count = available.row_count
            self.query_one("#tools-available-label", Label).update(
                f"Available tools ({count})" if count else "Available tools (none yet)"
            )

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show the highlighted call in full."""
        if event.data_table.id != "tool-calls":
            return
        call = _ring_item(self.state.tool_calls, event.row_key.value)
        if call is not None:
            self.query_one("#tool-detail", DetailView).show(render_tool_call(call), call.id)

    def focus_main(self) -> None:
        """Focus the calls table."""
        self.table.focus()


# ---------------------------------------------------------------------------
# 4 Metrics
# ---------------------------------------------------------------------------


class MetricsPane(Pane):
    """Tab 4: totals, per agent / model / tool, and tokens per LLM call."""

    SECTIONS = frozenset({S.METRICS})

    DEFAULT_CSS = """
    MetricsPane #metrics-scroll { height: 1fr; padding: 0 1; }
    MetricsPane .section { color: $primary; text-style: bold; height: 1; margin-top: 1; }
    MetricsPane #metrics-spark { height: 3; margin: 0 0; }
    MetricsPane DataTable { height: auto; max-height: 14; }
    """

    def compose(self) -> ComposeResult:
        """Totals, sparkline and the three tables."""
        with VerticalScroll(id="metrics-scroll"):
            yield Static(id="metrics-totals")
            yield Label("Tokens per LLM call", classes="section", id="metrics-spark-label")
            yield Sparkline([], id="metrics-spark")
            yield Label("Per agent", classes="section")
            yield DataTable(id="metrics-agents", zebra_stripes=True, cursor_type="none")
            yield Label("Per model", classes="section")
            yield DataTable(id="metrics-models", zebra_stripes=True, cursor_type="none")
            yield Label("Per tool", classes="section")
            yield DataTable(id="metrics-tools", zebra_stripes=True, cursor_type="none")

    def on_mount(self) -> None:
        """Table columns."""
        self.query_one("#metrics-agents", DataTable).add_columns(
            "Agent", "Model", "LLM calls", "Tokens", "Tool calls", "Errors", "Cost"
        )
        self.query_one("#metrics-models", DataTable).add_columns(
            "Model", "LLM calls", "Prompt", "Completion", "Cost"
        )
        self.query_one("#metrics-tools", DataTable).add_columns(
            "Tool", "Calls", "Errors", "Avg ms", "Total ms"
        )

    def update_view(self) -> None:
        """Redraw from the latest stats."""
        stats = self.state.stats
        totals = stats["totals"]
        grid = Table.grid(padding=(0, 3))
        for _ in range(5):
            grid.add_column()

        def cell(label: str, value: str, style: str = TEXT) -> Text:
            t = Text(f"{label}\n", style=MUTED)
            t.append(value, style=f"bold {style}")
            return t

        errors = totals.get("errors") or 0
        grid.add_row(
            cell("LLM calls", _num(totals.get("llm_calls"))),
            cell("prompt tokens", _num(totals.get("prompt_tokens"))),
            cell("completion tokens", _num(totals.get("completion_tokens"))),
            cell("total tokens", _num(totals.get("total_tokens"))),
            cell("cost", _cost(totals.get("cost_usd")), SUCCESS),
        )
        grid.add_row(
            cell("tool calls", _num(totals.get("tool_calls"))),
            cell("tool errors", _num(totals.get("tool_errors")), ERROR if errors else TEXT),
            cell("errors", _num(errors), ERROR if errors else TEXT),
            cell("retries", _num(totals.get("retries"))),
            cell(
                "latency p50 / p95",
                f"{_ms(totals.get('latency_p50_ms') or 0.0)} / "
                f"{_ms(totals.get('latency_p95_ms') or 0.0)} ms",
            ),
        )
        parts: list[RenderableType] = [grid]
        unpriced = totals.get("unpriced_models") or []
        if isinstance(unpriced, list) and unpriced:
            parts.append(
                Text(
                    "No [pricing] for: " + ", ".join(str(m) for m in unpriced),
                    style=MUTED,
                )
            )
        self.query_one("#metrics-totals", Static).update(Group(*parts))

        series = list(self.state.token_series)
        self.query_one("#metrics-spark", Sparkline).data = [float(v) for v in series]
        label = "Tokens per LLM call"
        if series:
            label += f" (last {len(series)} · max {max(series):,} · latest {series[-1]:,})"
        self.query_one("#metrics-spark-label", Label).update(label)

        agents = self.query_one("#metrics-agents", DataTable)
        agents.clear()
        for name, row in stats["agents"].items():
            agents.add_row(
                name,
                str(row.get("model") or "—"),
                _num(row.get("llm_calls")),
                _num(row.get("total_tokens")),
                _num(row.get("tool_calls")),
                _num(row.get("errors")),
                _cost(row.get("cost_usd")),
            )
        models = self.query_one("#metrics-models", DataTable)
        models.clear()
        for name, row in stats["models"].items():
            models.add_row(
                name,
                _num(row.get("llm_calls")),
                _num(row.get("prompt_tokens")),
                _num(row.get("completion_tokens")),
                _cost(row.get("cost_usd")),
            )
        tools = self.query_one("#metrics-tools", DataTable)
        tools.clear()
        for name, row in stats["tools"].items():
            tools.add_row(
                name,
                _num(row.get("calls")),
                _num(row.get("errors")),
                _ms(row.get("avg_ms") if isinstance(row.get("avg_ms"), (int, float)) else None),
                _ms(row.get("total_ms") if isinstance(row.get("total_ms"), (int, float)) else None),
            )

    def focus_main(self) -> None:
        """Focus the scroll area."""
        self.query_one("#metrics-scroll", VerticalScroll).focus()


# ---------------------------------------------------------------------------
# 5 Events
# ---------------------------------------------------------------------------


def _event_cells(row: AgentEventRow) -> tuple[Any, ...]:
    return (
        Text(clock(row.ts), style=MUTED),
        row.agent,
        Text(row.event_type, style=ACCENT),
        Text(row.severity, style=severity_style(row.severity)),
        preview(row.data, 160),
    )


class EventsPane(Pane):
    """Tab 5: agent events (invocations, slow tools, guardrails, approvals, …)."""

    SECTIONS = frozenset({S.EVENTS})

    DEFAULT_CSS = """
    EventsPane .split { height: 1fr; }
    EventsPane FeedTable { width: 2fr; }
    """

    COLUMNS = [
        ("time", "Time", 12),
        ("agent", "Agent", 14),
        ("event", "Event", 24),
        ("severity", "Severity", 8),
        ("data", "Data", 12),
    ]

    def compose(self) -> ComposeResult:
        """Table and detail."""
        with Horizontal(classes="split"):
            yield FeedTable(self.COLUMNS, _event_cells, id="events-table")
            yield DetailView(id="events-detail")

    @property
    def table(self) -> FeedTable:
        """The events table."""
        return self.query_one("#events-table", FeedTable)

    def rebuild(self) -> None:
        """Redraw everything."""
        self.table.rebuild(self.state.events)

    def update_view(self) -> None:
        """Append new events."""
        self.table.sync(self.state.events)

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show the highlighted event in full."""
        row = _ring_item(self.state.events, event.row_key.value)
        if row is not None:
            body = dict(row.event)
            body.setdefault("agent", row.agent)
            self.query_one("#events-detail", DetailView).show(json_view(body), row.id)

    def focus_main(self) -> None:
        """Focus the table."""
        self.table.focus()


# ---------------------------------------------------------------------------
# 6 Approvals
# ---------------------------------------------------------------------------


def time_left_text(row: ApprovalRow, now: float) -> Text:
    """The countdown of a pending request (``42s``, ``2m 05s``, ``expiring``)."""
    if row.answered is not None:
        return Text("approving…" if row.answered else "denying…", style=ACCENT)
    left = row.time_left(now)
    if left is None:
        return Text("—", style=MUTED)
    if left <= 0:
        return Text("expiring", style=f"bold {ERROR}")
    style = f"bold {ERROR}" if left < 30 else WARNING if left < 120 else TEXT
    return Text(format_duration(left) if left >= 10 else f"{left:.0f}s", style=style)


_OUTCOME_STYLES = {"approved": SUCCESS, "denied": f"bold {ERROR}", "expired": WARNING}


def render_approval(row: ApprovalRow, now: float) -> RenderableType:
    """Everything about one approval request: tool, arguments, conversation, outcome."""
    head = Table.grid(padding=(0, 2))
    head.add_column(style=MUTED, no_wrap=True)
    head.add_column()
    head.add_row("tool", Text(row.tool, style=f"bold {ACCENT}"))
    head.add_row("agent", row.agent)
    if row.caller:
        head.add_row("caller", row.caller)
    head.add_row("requested", clock(row.requested_at))
    if row.pending:
        head.add_row("time left", time_left_text(row, now))
    else:
        head.add_row("outcome", Text(row.status, style=_OUTCOME_STYLES.get(row.status, DIM)))
        if row.reason:
            head.add_row("reason", row.reason)
    head.add_row("request id", Text(row.request_id, style=MUTED))
    parts: list[RenderableType] = [head, Text("\nArguments", style=f"bold {PRIMARY}")]
    parts.append(json_view(row.arguments) if row.arguments is not None else Text("—", style=MUTED))
    if row.context_summary:
        parts.append(Text("\nConversation", style=f"bold {PRIMARY}"))
        parts.append(Text(row.context_summary, style=DIM))
    metadata = row.request.get("metadata")
    if metadata:
        parts.append(Text("\nMetadata", style=f"bold {PRIMARY}"))
        parts.append(json_view(metadata))
    if row.pending and row.answered is None:
        parts.append(Text("\na approve · d deny", style=f"bold {WARNING}"))
    return Group(*parts)


class ApprovalsPane(Pane):
    """Tab 6: tool calls waiting for a human decision, and the decisions made."""

    SECTIONS = frozenset({S.APPROVALS})

    DEFAULT_CSS = """
    ApprovalsPane .split { height: 1fr; }
    ApprovalsPane #approvals-left { width: 2fr; }
    ApprovalsPane .section { color: $primary; text-style: bold; padding: 0 1; height: 1; }
    ApprovalsPane #approvals-pending { height: 1fr; min-height: 5; }
    ApprovalsPane #approvals-history { height: 1fr; min-height: 5; }
    """

    def compose(self) -> ComposeResult:
        """Pending and history tables beside the detail."""
        with Horizontal(classes="split"):
            with Vertical(id="approvals-left"):
                yield Label("Pending", classes="section", id="approvals-pending-label")
                yield FitTable(
                    [
                        ("time", "Time", 12),
                        ("agent", "Agent", 14),
                        ("tool", "Tool", 22),
                        ("left", "Time left", 11),
                        ("args", "Arguments", 12),
                    ],
                    id="approvals-pending",
                )
                yield Label("History", classes="section", id="approvals-history-label")
                yield FitTable(
                    [
                        ("time", "Time", 12),
                        ("agent", "Agent", 14),
                        ("tool", "Tool", 22),
                        ("outcome", "Outcome", 9),
                        ("reason", "Reason", 12),
                    ],
                    id="approvals-history",
                )
            yield DetailView(
                "A tool call that needs approval shows up here. Highlight it, then press "
                "a to approve or d to deny.",
                id="approvals-detail",
            )

    @property
    def pending_table(self) -> FitTable:
        """The pending requests."""
        return self.query_one("#approvals-pending", FitTable)

    @property
    def history_table(self) -> FitTable:
        """The resolved requests."""
        return self.query_one("#approvals-history", FitTable)

    def update_view(self) -> None:
        """Redraw both tables (approvals come at human speed: a full redraw is cheap)."""
        now = time.time()
        pending = self.pending_table
        pending.ensure_columns()
        selected = _cursor_key(pending)
        pending.clear()
        rows = self.state.pending_approvals
        for row in rows:
            pending.add_row(
                Text(clock(row.requested_at), style=MUTED),
                row.agent,
                Text(row.tool, style=f"bold {ACCENT}"),
                time_left_text(row, now),
                preview(row.arguments, 160),
                key=str(row.id),
            )
        if selected is not None and selected in pending.rows:
            pending.move_cursor(row=pending.get_row_index(selected), animate=False)
        history = self.history_table
        history.ensure_columns()
        history.clear()
        done = [row for row in self.state.approvals if not row.pending]
        for row in done[-self.state.cap :]:
            history.add_row(
                Text(clock(row.resolved_at or row.requested_at), style=MUTED),
                row.agent,
                Text(row.tool, style=ACCENT),
                Text(row.status, style=_OUTCOME_STYLES.get(row.status, DIM)),
                preview(row.reason or "", 160),
                key=str(row.id),
            )
        if done:
            history.scroll_end(animate=False)
        self.query_one("#approvals-pending-label", Label).update(
            f"Pending ({len(rows)})" if rows else "Pending (none)"
        )
        self.query_one("#approvals-history-label", Label).update(f"History ({len(done)})")
        self._show_detail()

    def tick(self) -> None:
        """Count the pending requests' time left down (every second)."""
        now = time.time()
        table = self.pending_table
        for row in self.state.pending_approvals:
            key = str(row.id)
            if key in table.rows:
                try:
                    table.update_cell(key, "left", time_left_text(row, now))
                except Exception:
                    pass
        detail = self.query_one("#approvals-detail", DetailView)
        if detail.shown_id is not None:
            shown = self.state.approvals.get(detail.shown_id)
            if shown is not None and shown.pending:
                detail.show(render_approval(shown, now), shown.id)

    def highlighted(self) -> ApprovalRow | None:
        """The pending request under the cursor (or the only one)."""
        key = _cursor_key(self.pending_table)
        row = _ring_item(self.state.approvals, key) if key is not None else None
        if isinstance(row, ApprovalRow) and row.pending:
            return row
        return None

    def _show_detail(self, table: DataTable[Any] | None = None) -> None:
        table = table or (
            self.pending_table if self.pending_table.row_count else self.history_table
        )
        key = _cursor_key(table)
        row = _ring_item(self.state.approvals, key) if key is not None else None
        detail = self.query_one("#approvals-detail", DetailView)
        if isinstance(row, ApprovalRow):
            detail.show(render_approval(row, time.time()), row.id)
        elif detail.shown_id is not None and self.state.approvals.get(detail.shown_id) is None:
            detail.reset()

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Show the highlighted request in full."""
        if event.data_table.id in ("approvals-pending", "approvals-history"):
            self._show_detail(event.data_table)

    def focus_main(self) -> None:
        """Focus the pending table."""
        self.pending_table.focus()


def _cursor_key(table: DataTable[Any]) -> str | None:
    if table.row_count == 0:
        return None
    try:
        return table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 7 Agents
# ---------------------------------------------------------------------------


def _feature_text(features: dict[str, Any]) -> Text:
    t = Text()
    for name, value in features.items():
        if t:
            t.append("  ")
        if value is True:
            t.append(f"✓ {name}", style=SUCCESS)
        elif value in (False, None, "", 0):
            t.append(f"✗ {name}", style=MUTED)
        else:
            t.append(f"✓ {name}", style=SUCCESS)
            t.append(f" ({preview(value, 40)})", style=DIM)
    return t or Text("—", style=MUTED)


def render_agent(agent: AgentInfo) -> RenderableType:
    """One agent's model, observe level, features, servers, tools and instructions."""
    title = Text(agent.id, style=f"bold {ACCENT}")
    if agent.closed:
        title.append("  (closed)", style=MUTED)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=MUTED, no_wrap=True)
    grid.add_column()
    grid.add_row("model", Text(agent.model or "—", style=TEXT))
    grid.add_row("observe", agent.observe_level or "—")
    grid.add_row("features", _feature_text(agent.features))
    servers = Text()
    for server in agent.servers:
        if servers:
            servers.append(", ")
        servers.append(server.name, style=ACCENT)
        if server.transport:
            servers.append(f" ({server.transport})", style=MUTED)
    grid.add_row("servers", servers or Text("—", style=MUTED))
    grid.add_row("tools", f"{len(agent.tools)}")
    if agent.instructions:
        grid.add_row("instructions", Text(agent.instructions, style=DIM))
    return Panel(grid, title=title, title_align="left", border_style=MUTED, padding=(0, 1))


class AgentsPane(Pane):
    """Tab 7: every agent the program built."""

    SECTIONS = frozenset({S.AGENTS})

    DEFAULT_CSS = """
    AgentsPane #agents-scroll { height: 1fr; padding: 0 1; }
    """

    def compose(self) -> ComposeResult:
        """A scrollable body."""
        with VerticalScroll(id="agents-scroll"):
            yield Static(id="agents-body")

    def update_view(self) -> None:
        """Redraw every agent."""
        agents = list(self.state.agents.values())
        body: RenderableType
        if agents:
            body = Group(*(render_agent(a) for a in agents))
        else:
            body = Text(
                "No agent built yet. Agents appear here when the program builds them.",
                style=MUTED,
            )
        self.query_one("#agents-body", Static).update(body)

    def focus_main(self) -> None:
        """Focus the scroll area."""
        self.query_one("#agents-scroll", VerticalScroll).focus()


# ---------------------------------------------------------------------------
# 8 MCP
# ---------------------------------------------------------------------------


class McpPane(Pane):
    """Tab 8: the MCP servers every agent is connected to."""

    SECTIONS = frozenset({S.AGENTS})

    DEFAULT_CSS = """
    McpPane #mcp-table { height: 1fr; }
    McpPane #mcp-empty { color: $text-muted; padding: 0 1; height: auto; }
    """

    def compose(self) -> ComposeResult:
        """The servers table."""
        yield Static("No MCP server yet.", id="mcp-empty")
        yield DataTable(id="mcp-table", cursor_type="row", zebra_stripes=True)

    def on_mount(self) -> None:
        """Columns."""
        self.query_one("#mcp-table", DataTable).add_columns(
            "Agent", "Server", "Transport", "Target", "Tools"
        )

    def update_view(self) -> None:
        """Redraw the servers."""
        table = self.query_one("#mcp-table", DataTable)
        table.clear()
        for agent in self.state.agents.values():
            for server in agent.servers:
                table.add_row(
                    agent.id,
                    Text(server.name, style=f"bold {ACCENT}"),
                    server.transport or "—",
                    preview(server.target, 80) or "—",
                    str(server.tool_count),
                )
        self.query_one("#mcp-empty", Static).display = table.row_count == 0

    def focus_main(self) -> None:
        """Focus the table."""
        self.query_one("#mcp-table", DataTable).focus()


# ---------------------------------------------------------------------------
# 9 Logs
# ---------------------------------------------------------------------------

LEVEL_FILTERS: list[tuple[int, str]] = [
    (0, "all"),
    (20, "info+"),
    (30, "warning+"),
    (40, "error+"),
]


def render_log_line(line: LogLine) -> Text:
    """One log record (or output line) with its level coloured."""
    label = line.source if line.source in ("stderr", "stdout") else line.level
    t = Text(no_wrap=False)
    t.append(clock(line.ts), style=MUTED)
    t.append(f" {label:<8} ", style=level_style(label))
    if line.source == "log" and line.logger:
        t.append(f"{line.logger}: ", style=ACCENT)
    message_style = level_style(line.level) if line.levelno >= 40 else TEXT
    if line.source == "stderr":
        message_style = level_style("stderr")
    t.append(line.message, style=message_style)
    if line.exc_text:
        t.append("\n" + line.exc_text.rstrip("\n"), style=ERROR)
    return t


class LogsPane(Pane):
    """Tab 9: log records and the program's stderr, filterable by level."""

    SECTIONS = frozenset({S.LOGS})

    DEFAULT_CSS = """
    LogsPane .bar { height: 1; }
    LogsPane .bar Button { margin-right: 2; }
    LogsPane .hint { color: $text-muted; width: 1fr; }
    LogsPane #logs-log { height: 1fr; padding: 0 1; background: $background; }
    """

    def __init__(self, title: str, state: RunState, *, id: str) -> None:
        super().__init__(title, state, id=id)
        self.level_index = 0
        self._last = 0
        self.shown = 0
        """How many lines the log shows (after the filter)."""

    @property
    def min_level(self) -> int:
        """The lowest level shown (0 = everything)."""
        return LEVEL_FILTERS[self.level_index][0]

    def compose(self) -> ComposeResult:
        """Filter bar and the log."""
        with Horizontal(classes="bar"):
            yield Button(self._label(), id="logs-level", compact=True)
            yield Static("l: cycle levels", classes="hint", id="logs-hint")
        yield RichLog(
            id="logs-log",
            max_lines=self.state.cap,
            wrap=True,
            markup=False,
            highlight=False,
            auto_scroll=True,
        )

    def _label(self) -> str:
        return f"Level: {LEVEL_FILTERS[self.level_index][1]}"

    def rebuild(self) -> None:
        """Write the kept lines again (the filter changed)."""
        self.query_one("#logs-log", RichLog).clear()
        self.shown = 0
        self._last = 0
        self.update_view()

    def update_view(self) -> None:
        """Write new lines."""
        ring = self.state.logs
        log = self.query_one("#logs-log", RichLog)
        floor = self.min_level
        written = 0
        for line in ring.since(self._last):
            if written >= WRITE_BUDGET:
                break
            self._last = line.id
            if line.levelno >= floor:
                log.write(render_log_line(line))
                written += 1
                self.shown = min(self.shown + 1, self.state.cap)
        self.query_one("#logs-hint", Static).update(
            f"l: cycle levels · {self.shown:,} shown of {len(ring):,} kept"
        )

    def behind(self) -> bool:
        """The log has lines it did not look at yet."""
        return self._last < self.state.logs.last_id

    def cycle_level(self) -> None:
        """Show only more severe lines (then everything again)."""
        self.level_index = (self.level_index + 1) % len(LEVEL_FILTERS)
        self.query_one("#logs-level", Button).label = self._label()
        self.needs_rebuild = True
        self.sync()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """The filter button cycles the level."""
        if event.button.id == "logs-level":
            event.stop()
            self.cycle_level()

    def focus_main(self) -> None:
        """Focus the log."""
        self.query_one("#logs-log", RichLog).focus()


# ---------------------------------------------------------------------------
# 0 Run
# ---------------------------------------------------------------------------


def _http_cells(req: HttpRequest) -> tuple[Any, ...]:
    status = req.status
    style = SUCCESS if status is not None and status < 400 else ERROR
    return (
        Text(clock(req.ts), style=MUTED),
        Text(req.method, style=f"bold {ACCENT}"),
        preview(req.path, 60),
        Text(str(status) if status is not None else "?", style=style),
        _ms(req.duration_ms),
        Text(req.request_id, style=MUTED),
    )


def render_run(state: RunState) -> RenderableType:
    """The current run: process, manifest config, env vars, scripts, exit and traceback."""
    run = state.current_run
    if run is None:
        return Text("Waiting for the program to start…", style=MUTED)
    info = run.info
    parts: list[RenderableType] = []
    title = Text(f"Run #{run.number}", style=f"bold {PRIMARY}")
    title.append(f"  {state.state_label()}", style=DIM)
    parts.append(title)

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=MUTED, no_wrap=True)
    grid.add_column()

    def row(label: str, value: Any) -> None:
        if value in (None, "", [], {}):
            return
        if isinstance(value, (list, dict)):
            grid.add_row(label, preview(value, 400))
        else:
            grid.add_row(label, str(value))

    project = str(info.get("project") or "")
    if info.get("project_version"):
        project += f" v{info.get('project_version')}"
    row("project", project)
    row("main", info.get("main"))
    row("mode", info.get("mode"))
    row("entry", info.get("entry_kind"))
    row("pid", info.get("pid"))
    row("python", info.get("python"))
    row("promptise", info.get("promptise_version"))
    row("cwd", info.get("cwd"))
    row("root", info.get("root"))
    argv = info.get("argv")
    row("argv", " ".join(str(a) for a in argv) if isinstance(argv, list) else argv)
    flags = info.get("flags")
    if isinstance(flags, dict):
        on = [name for name, value in flags.items() if value]
        row("flags", ", ".join(on) if on else "none")
    row("model override", info.get("model_override"))
    parts.append(grid)

    if run.exit is not None:
        parts.append(Text("\nExit", style=f"bold {PRIMARY}"))
        code = run.exit_code
        exit_grid = Table.grid(padding=(0, 2))
        exit_grid.add_column(style=MUTED, no_wrap=True)
        exit_grid.add_column()
        exit_grid.add_row(
            "code",
            Text(str(code) if code is not None else "?", style=SUCCESS if not code else ERROR),
        )
        exit_grid.add_row("duration", format_duration(run.duration(run.ended_at or 0)))
        error = run.exit.get("error")
        if error:
            exit_grid.add_row("error", Text(str(error), style=f"bold {ERROR}"))
        parts.append(exit_grid)
        traceback = run.exit.get("traceback")
        if traceback:
            parts.append(Text(str(traceback).rstrip("\n"), style=ERROR))

    if len(state.runs) > 1:
        parts.append(Text("\nRuns", style=f"bold {PRIMARY}"))
        history = Table.grid(padding=(0, 2))
        history.add_column(style=MUTED)
        history.add_column()
        for past in state.runs:
            if past.exit is None:
                status = "running"
            else:
                status = f"exited code {past.exit_code} after " + format_duration(
                    past.duration(past.ended_at or 0)
                )
            history.add_row(f"#{past.number}", status)
        parts.append(history)

    env = info.get("env")
    if isinstance(env, list) and env:
        parts.append(Text("\nEnvironment", style=f"bold {PRIMARY}"))
        table = Table(box=None, padding=(0, 2), show_edge=False, header_style=MUTED)
        table.add_column("Name")
        table.add_column("Set")
        table.add_column("Required")
        for item in env:
            if not isinstance(item, dict):
                continue
            is_set = bool(item.get("set"))
            required = bool(item.get("required"))
            table.add_row(
                Text(str(item.get("name") or "?"), style=TEXT),
                Text("✓", style=SUCCESS)
                if is_set
                else Text("✗", style=ERROR if required else MUTED),
                "required" if required else "",
            )
        parts.append(table)

    scripts = info.get("scripts")
    if isinstance(scripts, dict) and scripts:
        parts.append(Text("\nScripts", style=f"bold {PRIMARY}"))
        table = Table(box=None, padding=(0, 2), show_edge=False, show_header=False)
        table.add_column(style=ACCENT)
        table.add_column()
        for name, command in scripts.items():
            table.add_row(str(name), str(command))
        parts.append(table)

    config = info.get("config")
    if config:
        parts.append(Text("\nManifest", style=f"bold {PRIMARY}"))
        parts.append(json_view(config))
    return Group(*parts)


class RunPane(Pane):
    """Tab 0: the run itself, its config and environment, and (serve mode) HTTP requests."""

    SECTIONS = frozenset({S.RUN, S.HTTP, S.HEADER})

    DEFAULT_CSS = """
    RunPane #run-scroll { height: 1fr; padding: 0 1; }
    RunPane .section { color: $primary; text-style: bold; height: 1; margin-top: 1; }
    RunPane #run-http { height: 14; }
    """

    COLUMNS = [
        ("time", "Time", 12),
        ("method", "Method", 7),
        ("path", "Path", 40),
        ("status", "Status", 6),
        ("ms", "ms", 7),
        ("request", "Request id", 34),
    ]

    def __init__(self, title: str, state: RunState, *, id: str) -> None:
        super().__init__(title, state, id=id)
        self._run_dirty = True

    def compose(self) -> ComposeResult:
        """The run body and the HTTP requests (hidden until one arrives)."""
        with VerticalScroll(id="run-scroll"):
            yield Static(id="run-body")
            yield Label("HTTP requests", classes="section", id="run-http-label")
            yield FeedTable(self.COLUMNS, _http_cells, id="run-http")

    def on_mount(self) -> None:
        """Hide the HTTP section until there is a request."""
        self._show_http(False)

    def _show_http(self, show: bool) -> None:
        self.query_one("#run-http-label", Label).display = show
        self.query_one("#run-http", FeedTable).display = show

    def note(self, changes: Changes) -> None:
        """Remember whether the run itself changed."""
        super().note(changes)
        if changes.sections & {S.RUN, S.HEADER}:
            self._run_dirty = True

    def rebuild(self) -> None:
        """Redraw everything."""
        self._run_dirty = True
        self.query_one("#run-http", FeedTable).rebuild(self.state.http)
        self.update_view()

    def update_view(self) -> None:
        """Redraw the run and append requests."""
        if self._run_dirty:
            self._run_dirty = False
            self.query_one("#run-body", Static).update(render_run(self.state))
        table = self.query_one("#run-http", FeedTable)
        table.sync(self.state.http)
        if len(self.state.http):
            self._show_http(True)
            self.query_one("#run-http-label", Label).update(
                f"HTTP requests ({len(self.state.http):,})"
            )

    def focus_main(self) -> None:
        """Focus the scroll area."""
        self.query_one("#run-scroll", VerticalScroll).focus()


PANES: list[tuple[str, str, type[Pane]]] = [
    ("tab-chat", "Console", ChatPane),
    ("tab-trace", "Trace", TracePane),
    ("tab-tools", "Tools", ToolsPane),
    ("tab-metrics", "Metrics", MetricsPane),
    ("tab-events", "Events", EventsPane),
    ("tab-approvals", "Approvals", ApprovalsPane),
    ("tab-agents", "Agents", AgentsPane),
    ("tab-mcp", "MCP", McpPane),
    ("tab-logs", "Logs", LogsPane),
    ("tab-run", "Run", RunPane),
]
"""``(pane id, title, class)`` of every tab, in digit order (1-9, then 0)."""
