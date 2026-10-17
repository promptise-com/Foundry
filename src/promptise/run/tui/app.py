"""The ``promptise run --tui`` terminal UI: a Textual app over a :class:`RunSource`.

The app never touches an agent.  A worker reads run events from the source
into a queue; every :attr:`RunTUI.batch_interval` seconds the queued events
are folded into a :class:`~promptise.run.tui.state.RunState` and only the
visible tab redraws what changed, so a program producing thousands of events
a second does not freeze the UI.  Keys and the input send commands back
through the source (``chat``, ``stdin``, ``reset``, ``cancel``, restart, stop).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any, ClassVar

from textual.app import App, ComposeResult, ScreenStackError
from textual.binding import Binding, BindingType
from textual.css.query import NoMatches
from textual.widgets import Footer, Input, TabbedContent

from ..events import RunEvent
from ..source import RunSource
from .panes import PANES, ApprovalsPane, ChatPane, LogsPane, Pane, TracePane
from .state import ApprovalRow, ChatTurn, RunState
from .theme import THEME
from .widgets import HelpScreen, ReasonScreen, RunHeader

__all__ = ["RunTUI"]

FLUSH_LIMIT = 10_000
"""The most events folded in per batch (the rest wait for the next tick)."""

TAB_IDS = [pane_id for pane_id, _title, _cls in PANES]

_TYPING_BLOCKED = {
    "tab",
    "restart",
    "stop",
    "quit",
    "help",
    "cycle_agent",
    "cycle_level",
    "approve",
    "deny",
}

_TAB_ONLY = {
    "cycle_agent": "tab-trace",
    "cycle_level": "tab-logs",
    "approve": "tab-approvals",
    "deny": "tab-approvals",
}


class RunTUI(App[int]):
    """The terminal UI of ``promptise run``.

    Args:
        source: Where the run events come from and commands go to.
        title: The window title.
        batch_interval: Seconds between two UI updates (queued events are
            applied in one batch).

    The app's result is the program's last exit code (0 when unknown).
    """

    TITLE = "promptise run"
    ENABLE_COMMAND_PALETTE = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("1", "tab(1)", "Tabs", key_display="0-9"),
        *(Binding(str(n), f"tab({n})", show=False) for n in range(2, 10)),
        Binding("0", "tab(10)", show=False),
        Binding("a", "approve", "Approve"),
        Binding("d", "deny", "Deny"),
        Binding("r", "restart", "Restart"),
        Binding("s", "stop", "Stop"),
        Binding("a", "cycle_agent", "Agent filter"),
        Binding("l", "cycle_level", "Level filter"),
        Binding("question_mark", "help", "Help", key_display="?"),
        Binding("q", "quit", "Quit"),
        Binding("escape", "blur", "Shortcuts", key_display="esc"),
        Binding("ctrl+l", "refresh_all", "Redraw", show=False),
        Binding("ctrl+c", "interrupt", "Quit", show=False, priority=True),
    ]

    CSS = """
    Screen { layout: vertical; }
    #tabs { height: 1fr; }
    #tabs > ContentSwitcher { height: 1fr; }
    TabPane { padding: 0; height: 1fr; }
    """

    def __init__(
        self,
        source: RunSource,
        *,
        title: str = "promptise run",
        batch_interval: float = 0.1,
    ) -> None:
        super().__init__()
        self.source = source
        self.title = title
        self.batch_interval = batch_interval
        self.state = RunState()
        self.ui_errors: list[str] = []
        """Errors the UI caught and survived (a pane that failed to draw, a command that failed)."""
        self.stream_ended = False
        """The source's event stream ended (the source was closed)."""
        self._queue: list[RunEvent] = []
        self._outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._quitting = False
        self._source_closed = False

    # -- layout -------------------------------------------------------------

    def compose(self) -> ComposeResult:
        """Header, the ten tabs (digits 1-9, then 0), footer."""
        yield RunHeader(id="header")
        with TabbedContent(initial=TAB_IDS[0], id="tabs"):
            for number, (pane_id, title, cls) in enumerate(PANES, start=1):
                yield cls(f"{number % 10} {title}", self.state, id=pane_id)
        yield Footer()

    def on_mount(self) -> None:
        """Apply the theme and start reading events."""
        self.register_theme(THEME)
        self.theme = THEME.name
        self.run_worker(self._consume(), name="run-events", group="events", exit_on_error=False)
        self.run_worker(self._send_loop(), name="run-send", group="send", exit_on_error=False)
        self.set_interval(self.batch_interval, self.flush_events, name="flush")
        self.set_interval(1.0, self._tick, name="uptime")
        self._redraw_header()
        self.chat_pane.focus_main()

    # -- accessors ----------------------------------------------------------

    @property
    def panes(self) -> list[Pane]:
        """Every tab, in digit order."""
        return [self.query_one(f"#{pane_id}", Pane) for pane_id in TAB_IDS]

    @property
    def chat_pane(self) -> ChatPane:
        """Tab 1."""
        return self.query_one("#tab-chat", ChatPane)

    @property
    def header(self) -> RunHeader:
        """The top bar."""
        return self.query_one("#header", RunHeader)

    @property
    def active_tab(self) -> str:
        """The id of the visible tab."""
        return self.query_one("#tabs", TabbedContent).active

    def _active_pane(self) -> Pane | None:
        try:
            return self.query_one(f"#{self.active_tab}", Pane)
        except Exception:
            return None

    # -- events in ----------------------------------------------------------

    async def _consume(self) -> None:
        count = 0
        try:
            async for event in self.source.events():
                self._queue.append(event)
                count += 1
                if count % 500 == 0:
                    await asyncio.sleep(0)  # a burst must not starve the UI
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._ui_error(f"the event stream failed: {exc!r}")
            self.notify(f"The event stream failed: {exc}", severity="error", markup=False)
        finally:
            self.stream_ended = True
        if not self._source_closed:
            self.notify(
                "The run's event stream ended: nothing new will show. Press q to quit.",
                severity="warning",
                timeout=10,
            )

    @property
    def pending_events(self) -> int:
        """Events received but not applied yet."""
        return len(self._queue)

    def flush_events(self) -> None:
        """Apply the queued events now and redraw what changed (runs every batch interval)."""
        if self._queue:
            batch = self._queue[:FLUSH_LIMIT]
            del self._queue[:FLUSH_LIMIT]
            for event in batch:
                try:
                    self.state.apply(event)
                except Exception as exc:  # RunState never raises; belt and braces
                    self._ui_error(f"bad event: {exc!r}")
        changes = self.state.pop_changes()
        if not changes:
            active = self._active_pane()
            if active is not None and active.dirty:
                self._sync(active)  # a draw that could not happen yet
            return
        if changes.new_run:
            self._apply_mode()
        if changes.new_approvals:
            self._announce_approvals(changes.new_approvals)
        self._redraw_header()
        for pane in self.panes:
            pane.note(changes)
        active = self._active_pane()
        if active is not None and active.dirty:
            self._sync(active)

    def _apply_mode(self) -> None:
        chat = self.state.chat_mode
        pane = self.chat_pane
        pane.set_mode(chat)
        try:
            self.query_one("#tabs", TabbedContent).get_tab("tab-chat").label = (
                "1 Chat" if chat else "1 Console"
            )
        except Exception as exc:
            self._ui_error(f"tab label: {exc!r}")

    def _announce_approvals(self, ids: list[int]) -> None:
        rows = [self.state.approvals.get(i) for i in ids]
        pending = [row for row in rows if row is not None and row.pending]
        if not pending:
            return
        if len(pending) == 1:
            row = pending[0]
            message = f"{row.agent} wants to call {row.tool}."
        else:
            message = f"{len(pending)} tool calls are waiting for approval."
        hint = " Press 6 to review." if self.active_tab != "tab-approvals" else ""
        self.notify(
            message + hint,
            title="Approval needed",
            severity="warning",
            timeout=8,
            markup=False,
        )

    def _sync(self, pane: Pane) -> None:
        try:
            pane.sync()
        except Exception as exc:
            pane.dirty = False
            self._ui_error(f"{pane.id}: {exc!r}")

    def _ui_error(self, message: str) -> None:
        self.ui_errors.append(message)
        del self.ui_errors[:-100]
        self.log.error(message)

    def _redraw_header(self) -> None:
        try:
            self.header.show_state(self.state, closing=self._quitting)
        except Exception as exc:
            self._ui_error(f"header: {exc!r}")

    def _tick(self) -> None:
        self._redraw_header()
        if self.active_tab == "tab-approvals" and self.state.pending_approvals:
            try:
                self.query_one("#tab-approvals", ApprovalsPane).tick()
            except Exception as exc:
                self._ui_error(f"approvals tick: {exc!r}")

    # -- commands out -------------------------------------------------------

    async def _send_loop(self) -> None:
        while True:
            command = await self._outbox.get()
            try:
                await self.source.send(command)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._ui_error(f"send {command.get('cmd')}: {exc!r}")
                self.notify(f"Could not send: {exc}", severity="error", markup=False)

    def send_command(self, command: dict[str, Any]) -> None:
        """Queue *command* for the source (sent in order, without blocking the UI)."""
        self._outbox.put_nowait(command)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in the input: a chat message (agent modes) or a stdin line (script mode)."""
        if event.input.id != "prompt":
            return
        event.stop()
        raw = event.value
        event.input.value = ""
        if self.state.chat_mode:
            self._submit_chat(raw)
        else:
            self.state.add_console_input(raw)
            self.send_command({"cmd": "stdin", "text": raw})
        self.flush_events()

    def _submit_chat(self, raw: str) -> None:
        text = raw.strip()
        if not text:
            return
        if text == "/reset":
            self.send_command({"cmd": "reset"})
            self.state.add_note("new conversation")
            return
        if text == "/cancel":
            running = [
                item for item in self.state.chat if isinstance(item, ChatTurn) and not item.done
            ]
            if running:
                self.send_command({"cmd": "cancel", "turn_id": running[-1].turn_id})
                self.state.add_note("cancel requested")
            else:
                self.notify("Nothing to cancel.")
            return
        if text == "/help":
            self.state.add_note("/reset starts a new conversation · /cancel stops the answer")
            return
        turn_id = uuid.uuid4().hex
        self.state.add_user_turn(turn_id, text)
        self.send_command({"cmd": "chat", "text": text, "turn_id": turn_id})

    # -- actions ------------------------------------------------------------

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Hide single-key shortcuts while typing, and filters outside their tab."""
        typing = isinstance(self.focused, Input)
        if action == "blur":
            return typing
        if action in _TYPING_BLOCKED and typing:
            return False
        tab = _TAB_ONLY.get(action)
        if tab is not None:
            return self.active_tab == tab
        return True

    def action_tab(self, number: int) -> None:
        """Show tab *number* (1-10; the ``0`` key is 10)."""
        if 1 <= number <= len(TAB_IDS):
            self.query_one("#tabs", TabbedContent).active = TAB_IDS[number - 1]

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        """Bring the shown tab up to date and focus its main widget."""
        pane = event.pane
        # A stale activation (another tab was shown since) must not focus its
        # pane: focusing a widget in a pane activates that pane's tab again, and
        # two queued activations would hand the focus back and forth forever.
        if isinstance(pane, Pane) and pane.id == event.tabbed_content.active:
            self._activate(pane)
        try:
            self.refresh_bindings()
        except ScreenStackError:  # the app is shutting down
            pass

    def _activate(self, pane: Pane, *, retry: bool = True) -> None:
        if pane.dirty or pane.needs_rebuild:
            self._sync(pane)
        try:
            pane.focus_main()
        except NoMatches:
            if retry:  # shown before its widgets were mounted: once more after a refresh
                self.call_after_refresh(self._activate, pane, retry=False)
        except ScreenStackError:  # the app is shutting down
            pass

    def action_blur(self) -> None:
        """Leave the input, so the single-key shortcuts work."""
        self.screen.set_focus(None)

    def action_restart(self) -> None:
        """Restart the program."""
        self.notify("Restarting the program…", timeout=2)
        self.run_worker(self._control("restart"), group="control", exit_on_error=False)

    def action_stop(self) -> None:
        """Stop the program (the UI stays open)."""
        self.notify("Stopping the program…", timeout=2)
        self.run_worker(self._control("stop"), group="control", exit_on_error=False)

    async def _control(self, what: str) -> None:
        try:
            await getattr(self.source, what)()
        except Exception as exc:
            self._ui_error(f"{what}: {exc!r}")
            self.notify(f"Could not {what} the program: {exc}", severity="error", markup=False)

    def action_cycle_agent(self) -> None:
        """Trace: show the next agent only."""
        self.query_one("#tab-trace", TracePane).cycle_agent()

    def action_cycle_level(self) -> None:
        """Logs: show more severe lines only."""
        self.query_one("#tab-logs", LogsPane).cycle_level()

    def _approval_target(self) -> ApprovalRow | None:
        row = self.query_one("#tab-approvals", ApprovalsPane).highlighted()
        if row is None:
            self.notify("No pending approval request is highlighted.", timeout=3)
        elif row.answered is not None:
            self.notify("Already answered; waiting for the program.", timeout=3)
            return None
        return row

    def action_approve(self) -> None:
        """Approvals: approve the highlighted request."""
        row = self._approval_target()
        if row is not None:
            self.answer_approval(row.request_id, True)

    def action_deny(self) -> None:
        """Approvals: deny the highlighted request (asks for an optional reason)."""
        row = self._approval_target()
        if row is None:
            return
        request_id = row.request_id

        def deny(reason: str | None) -> None:
            if reason is not None:
                self.answer_approval(request_id, False, reason or None)

        self.push_screen(ReasonScreen(row.tool), deny)

    def answer_approval(self, request_id: str, approved: bool, reason: str | None = None) -> None:
        """Send the decision on *request_id* to the program."""
        self.send_command(
            {"cmd": "approval", "request_id": request_id, "approved": approved, "reason": reason}
        )
        self.state.mark_approval_answered(request_id, approved)
        self.flush_events()

    def action_help(self) -> None:
        """Show the key overview."""
        if not isinstance(self.screen, HelpScreen):
            self.push_screen(HelpScreen())

    def action_refresh_all(self) -> None:
        """Redraw everything from the state."""
        self.flush_events()
        for pane in self.panes:
            pane.needs_rebuild = True
            pane.dirty = True
        active = self._active_pane()
        if active is not None:
            self._sync(active)
        self._redraw_header()
        self.refresh(layout=True)

    async def action_quit(self) -> None:
        """Stop the program and leave; a second ``q`` leaves without waiting."""
        if self._quitting:
            self.exit(self.exit_code)
            return
        self._quitting = True
        self._redraw_header()
        self.run_worker(self._quit(), group="quit", exit_on_error=False)

    async def action_interrupt(self) -> None:
        """``ctrl+c``: like ``q``."""
        await self.action_quit()

    async def _quit(self) -> None:
        await self.close_source()
        self.exit(self.exit_code)

    async def close_source(self) -> None:
        """Close the source once (stops the program and ends the event stream)."""
        if self._source_closed:
            return
        self._source_closed = True
        try:
            await self.source.close()
        except Exception as exc:
            self._ui_error(f"close: {exc!r}")

    @property
    def exit_code(self) -> int:
        """The program's last exit code (0 when unknown)."""
        code = self.state.exit_code
        return code if code is not None else 0
