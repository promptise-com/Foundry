"""Widgets shared by the ``promptise run`` terminal UI panes."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from typing import Any

from rich.console import RenderableType
from rich.json import JSON
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Input, Label, Static
from textual.widgets.data_table import CellDoesNotExist, RowDoesNotExist

from .state import CAP, Ring, RunState
from .theme import ACCENT, DIM, ERROR, MUTED, PRIMARY, SECONDARY, SUCCESS, TEXT, WARNING

__all__ = [
    "Column",
    "DetailView",
    "FeedTable",
    "FitTable",
    "HelpScreen",
    "ReasonScreen",
    "RunHeader",
    "clock",
    "json_view",
    "pretty_arguments",
]

Column = tuple[str, str, int | None]
"""A table column: ``(key, label, width)``; ``width`` ``None`` fits the content."""


def clock(ts: float) -> str:
    """*ts* (epoch seconds) as local ``HH:MM:SS.mmm``."""
    try:
        local = time.localtime(ts)
    except (OverflowError, OSError, ValueError):
        return "??:??:??"
    return time.strftime("%H:%M:%S", local) + f".{int((ts % 1) * 1000):03d}"


def json_view(data: Any) -> RenderableType:
    """*data* as highlighted, indented JSON (plain text when it cannot be encoded)."""
    try:
        return JSON.from_data(data, indent=2, default=str)
    except Exception:
        return Text(repr(data))


def plain(value: Any) -> Any:
    """A table cell that is never read as markup: a ``str`` becomes plain :class:`Text`.

    Textual and Rich parse a ``str`` cell as markup.  The cells here hold what
    the program, its tools and HTTP clients produced: ``[/x]`` would crash the
    UI and ``[@click=app.quit]`` would run an action when clicked.
    """
    return Text(value) if isinstance(value, str) else value


class SafeDataTable(DataTable[Any]):
    """A :class:`DataTable` whose string cells are plain text (see :func:`plain`)."""

    def add_row(self, *cells: Any, **kwargs: Any) -> Any:
        return super().add_row(*(plain(c) for c in cells), **kwargs)

    def update_cell(self, row_key: Any, column_key: Any, value: Any, **kwargs: Any) -> None:
        super().update_cell(row_key, column_key, plain(value), **kwargs)


class PlainTable(Table):
    """A Rich :class:`Table` whose string cells are plain text (see :func:`plain`)."""

    def add_row(self, *renderables: Any, **kwargs: Any) -> None:
        super().add_row(*(plain(r) for r in renderables), **kwargs)


class FitTable(SafeDataTable):
    """A row-cursor table whose last column takes the width the others leave.

    Short columns come first and the long preview (details, arguments, …)
    last, so it is the preview that gets cut on a narrow terminal, never a
    status or a number.

    Args:
        columns: The columns, see :data:`Column`; the last one's width is its minimum.
    """

    def __init__(self, columns: Sequence[Column], *, id: str | None = None) -> None:
        super().__init__(id=id, cursor_type="row", zebra_stripes=True)
        self._spec = list(columns)

    def on_mount(self) -> None:
        """Add the columns."""
        self.ensure_columns()

    def ensure_columns(self) -> None:
        """Add the columns if they are not there yet."""
        if not self.columns:
            for key, label, width in self._spec:
                self.add_column(label, key=key, width=width)
            self.fit_last_column()

    def on_resize(self) -> None:
        """Stretch the last column again."""
        self.fit_last_column()

    def fit_last_column(self) -> None:
        """Give the last column the width the other columns leave (at least its own)."""
        columns = self.ordered_columns
        if not columns or not self.size.width:
            return
        last = columns[-1]
        others = sum(column.get_render_width(self) for column in columns[:-1])
        room = self.size.width - others - 2 * self.cell_padding - 2  # 2: a vertical scrollbar
        width = max(room, self._spec[-1][2] or 10)
        if width != last.width or last.auto_width:
            last.width = width
            last.auto_width = False
            self._require_update_dimensions = True
            self.refresh()


class FeedTable(FitTable):
    """A row-cursor table fed from a :class:`~promptise.run.tui.state.Ring`.

    :meth:`sync` appends only the items added since the last call; when the
    table would grow past *cap* rows the oldest are dropped (the table is
    rebuilt from the ring with a tenth of headroom, so trimming is not paid
    on every new row).  A table scrolled to its end follows new rows.

    Args:
        columns: The columns, see :data:`Column`.
        row: Turns an item into the cells of its row (same order as *columns*).
        accept: Which items are shown (a filter); all when ``None``.
        cap: The most rows shown.
    """

    DEFAULT_CSS = """
    FeedTable { height: 1fr; }
    """

    def __init__(
        self,
        columns: Sequence[Column],
        row: Callable[[Any], Sequence[Any]],
        *,
        accept: Callable[[Any], bool] | None = None,
        cap: int = CAP,
        id: str | None = None,
    ) -> None:
        super().__init__(columns, id=id)
        self._row = row
        self.accept: Callable[[Any], bool] = accept or (lambda _item: True)
        self.cap = max(cap, 10)
        self.last_id = 0

    def _following(self) -> bool:
        return self.row_count == 0 or self.is_vertical_scroll_end

    def _add(self, item: Any) -> None:
        self.add_row(*self._row(item), key=str(item.id))

    def sync(self, ring: Ring[Any]) -> None:
        """Append the items of *ring* added since the last call."""
        self.ensure_columns()
        new = ring.since(self.last_id)
        self.last_id = ring.last_id
        rows = [item for item in new if self.accept(item)]
        if not rows:
            return
        if self.row_count + len(rows) > self.cap:
            self.rebuild(ring, keep=self.cap - self.cap // 10)
            return
        follow = self._following()
        for item in rows:
            self._add(item)
        if follow:
            self.scroll_end(animate=False)

    def rebuild(self, ring: Ring[Any], *, keep: int | None = None) -> None:
        """Show the newest accepted items of *ring* again (after a filter change)."""
        self.ensure_columns()
        follow = self._following()
        self.clear()
        items = [item for item in ring if self.accept(item)]
        for item in items[-(keep or self.cap) :]:
            self._add(item)
        self.last_id = ring.last_id
        if follow:
            self.scroll_end(animate=False)

    def refresh_item(self, item: Any) -> None:
        """Redraw *item*'s row (it changed in place); no-op when it is not shown."""
        key = str(item.id)
        if key not in self.rows:
            return
        for (column, _label, _width), value in zip(self._spec, self._row(item), strict=False):
            try:
                self.update_cell(key, column, value)
            except (CellDoesNotExist, RowDoesNotExist):
                return

    def highlighted_id(self) -> int | None:
        """The id of the item under the cursor, or ``None``."""
        if self.row_count == 0:
            return None
        try:
            key = self.coordinate_to_cell_key(self.cursor_coordinate).row_key
        except (CellDoesNotExist, RowDoesNotExist):
            return None
        try:
            return int(key.value) if key.value is not None else None
        except ValueError:
            return None


class DetailView(VerticalScroll):
    """A scrollable panel showing the full record of the highlighted row."""

    DEFAULT_CSS = """
    DetailView {
        width: 1fr;
        height: 1fr;
        border-left: tall $panel;
        padding: 0 1;
    }
    DetailView > Static { width: 1fr; }
    """

    def __init__(self, placeholder: str = "Highlight a row to see it in full.", **kwargs: Any):
        super().__init__(**kwargs)
        self._placeholder = placeholder
        self.shown_id: int | None = None

    def compose(self) -> ComposeResult:
        """One static body."""
        yield Static(Text(self._placeholder, style=MUTED), classes="detail-body")

    @property
    def body(self) -> Static:
        """The static widget holding the detail."""
        return self.query_one(".detail-body", Static)

    def show(self, renderable: RenderableType, item_id: int | None = None) -> None:
        """Show *renderable* (for item *item_id*)."""
        if item_id != self.shown_id:
            self.scroll_home(animate=False)
        self.shown_id = item_id
        self.body.update(renderable)

    def reset(self) -> None:
        """Show the placeholder again."""
        self.shown_id = None
        self.body.update(Text(self._placeholder, style=MUTED))


_STATE_STYLES = {
    "running": SUCCESS,
    "idle": SUCCESS,
    "busy": ACCENT,
    "starting": WARNING,
    "restarting": WARNING,
    "failed": ERROR,
}


class RunHeader(Static):
    """The top bar: project, mode, state, run number, uptime and the running totals."""

    DEFAULT_CSS = """
    RunHeader {
        height: 2;
        padding: 0 1;
        background: $panel;
    }
    """

    def show_state(self, state: RunState, *, closing: bool = False) -> None:
        """Redraw from *state*."""
        fields = state.header_fields()
        label = "stopping…" if closing else fields["state"]
        if label.startswith("exited"):
            style = DIM if label.endswith(" 0") else ERROR
        else:
            style = _STATE_STYLES.get(label.split(" ")[0], WARNING if closing else DIM)
        t = Text(no_wrap=True, overflow="ellipsis")
        t.append("█▀▀▄ ", style=f"bold {PRIMARY}")
        t.append("PROMPTISE", style=f"bold {ACCENT}")
        t.append(" run  ", style=DIM)
        t.append(fields["project"], style=f"bold {TEXT}")
        if fields["version"]:
            t.append(f" v{fields['version']}", style=MUTED)
        t.append("   mode ", style=MUTED)
        t.append(fields["mode"], style=TEXT)
        t.append("   ● ", style=style)
        t.append(label, style=f"bold {style}")
        if fields["run"]:
            t.append(f"   {fields['run']}", style=f"bold {WARNING}")
        if fields["approvals"] != "0":
            t.append(f"   ✋ {fields['approvals']}", style=f"bold {WARNING}")
            t.append(" approval" + ("s" if fields["approvals"] != "1" else ""), style=WARNING)
        t.append("   uptime ", style=MUTED)
        t.append(fields["uptime"], style=TEXT)
        t.append("\n")
        t.append("█▄▄▀ ", style=f"bold {SECONDARY}")
        t.append("LLM calls ", style=MUTED)
        t.append(fields["llm_calls"], style=TEXT)
        t.append("   tokens ", style=MUTED)
        t.append(fields["tokens"], style=TEXT)
        t.append("   cost ", style=MUTED)
        t.append(fields["cost"], style=TEXT)
        t.append("   errors ", style=MUTED)
        errors = fields["errors"]
        t.append(errors, style=f"bold {ERROR}" if errors != "0" else TEXT)
        if state.state_detail and not closing:
            t.append(f"   {state.state_detail}", style=MUTED)
        self.update(t)

    @property
    def plain(self) -> str:
        """The header's text, without styles (for tests and logs)."""
        content = self.content
        return content.plain if isinstance(content, Text) else str(content)


HELP_TEXT = """\
[b]Tabs[/b]
  1  Chat / Console     2  Trace     3  Tools     4  Metrics     5  Events
  6  Approvals          7  Agents    8  MCP       9  Logs        0  Run
  The digits work when the input box does not have the focus; [b]esc[/b] leaves it.

[b]Program[/b]
  r        restart the program
  s        stop the program (the UI stays open)
  q        quit (stops the program), ctrl+c too
  ctrl+l   redraw everything

[b]Approvals[/b] (tab 6; ✋ in the header counts the pending ones)
  a        approve the highlighted request
  d        deny it (asks for an optional reason)

[b]Filters[/b]
  a        Trace: cycle the agent filter
  l        Logs: cycle the level filter

[b]Chat[/b]
  enter    send the message (script mode: send the line to the program's stdin)
  /reset   start a new conversation
  /cancel  cancel the running answer

Arrow keys move in tables; the panel on the right shows the highlighted row.
Press [b]esc[/b], [b]?[/b] or [b]q[/b] to close this help."""


class HelpScreen(ModalScreen[None]):
    """The key overview (``?``)."""

    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    HelpScreen > Static {
        width: 88;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: tall $primary;
    }
    """

    BINDINGS = [
        Binding("escape,question_mark,q", "dismiss_help", "Close", key_display="esc"),
    ]

    def compose(self) -> ComposeResult:
        """The help text."""
        yield Static(HELP_TEXT, id="help-body")

    def action_dismiss_help(self) -> None:
        """Close the help."""
        self.dismiss(None)


def pretty_arguments(raw: str) -> RenderableType:
    """Tool arguments as indented JSON when they parse, else as recorded."""
    text = raw.strip()
    if text[:1] in ("{", "["):
        try:
            return json_view(json.loads(text))
        except ValueError:
            pass
    return Text(raw)


class ReasonScreen(ModalScreen[str | None]):
    """Asks for an optional reason before denying an approval request.

    Dismisses with the reason (``""`` when none was typed), or ``None`` when
    the user cancels with ``esc``.
    """

    DEFAULT_CSS = """
    ReasonScreen { align: center middle; }
    ReasonScreen > #reason-box {
        width: 80;
        height: auto;
        padding: 1 2;
        background: $surface;
        border: tall $error;
    }
    ReasonScreen Label { width: 1fr; margin-bottom: 1; }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, tool: str) -> None:
        super().__init__()
        self.tool = tool

    def compose(self) -> ComposeResult:
        """A question and the reason input."""
        with VerticalScroll(id="reason-box"):
            yield Label(
                Text.assemble(
                    ("Deny ", f"bold {ERROR}"),
                    (self.tool, f"bold {ACCENT}"),
                    ("?  Enter a reason (optional), enter to deny, esc to cancel.", DIM),
                )
            )
            yield Input(placeholder="Reason (optional)", id="reason-input")

    def on_mount(self) -> None:
        """Focus the input."""
        self.query_one("#reason-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter: deny with the typed reason."""
        event.stop()
        self.dismiss(event.value.strip())

    def action_cancel(self) -> None:
        """Esc: do not deny."""
        self.dismiss(None)
