"""What the ``promptise run`` terminal UI shows, folded from run events.

:class:`RunState` is plain Python (no Textual): feed it run events with
:meth:`RunState.apply` and read what each pane shows from its attributes.
It never raises on bad data: unknown kinds are ignored and missing or
mistyped fields fall back to empty values, because the events come from
another process running arbitrary user code.

Long-lived collections are :class:`Ring` buffers: capped (5,000 items by
default, oldest dropped first) with ids that keep increasing, so a view can
ask for "everything after id N" and append only what is new.  What changed
since the last look is collected in a :class:`Changes` (see
:meth:`RunState.pop_changes`), so the UI refreshes only the panes whose data
moved.
"""

from __future__ import annotations

import json
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import islice
from typing import Any, Generic, Protocol, TypeVar

from ..events import strip_controls
from ..stats import RunStats

__all__ = [
    "CAP",
    "AgentEventRow",
    "ApprovalRow",
    "AgentInfo",
    "Changes",
    "ChatError",
    "ChatNote",
    "ChatText",
    "ChatTurn",
    "ConsoleLine",
    "HttpRequest",
    "LogLine",
    "Ring",
    "RunRecord",
    "RunState",
    "ServerInfo",
    "TimelineRow",
    "ToolCall",
    "ToolCard",
    "ToolSpec",
    "format_cost",
    "format_duration",
    "preview",
]

CAP = 5_000
"""Default cap of every list the UI keeps (timeline, tool calls, events, logs, …)."""

CHAT_CAP = 1_000
"""How many chat items (turns and notes) are kept."""

SPARK_CAP = 200
"""How many recent LLM calls the token sparkline shows."""

RUN_CAP = 100
"""How many runs (restarts) are remembered."""

# Sections a change can touch; panes subscribe to the ones they show.
HEADER = "header"
RUN = "run"
AGENTS = "agents"
TIMELINE = "timeline"
TOOL_CALLS = "tool_calls"
EVENTS = "events"
METRICS = "metrics"
CHAT = "chat"
CONSOLE = "console"
LOGS = "logs"
HTTP = "http"
APPROVALS = "approvals"

LEVELS: dict[str, int] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "WARN": 30,
    "ERROR": 40,
    "CRITICAL": 50,
    "FATAL": 50,
}

_ENVELOPE = ("v", "seq", "ts", "kind")


# ---------------------------------------------------------------------------
# Defensive coercion: events come from user programs in another process.
# ---------------------------------------------------------------------------


def _str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, str):
        return value
    try:
        return str(value)
    except Exception:
        return default


def _opt_str(value: Any) -> str | None:
    return None if value is None else _str(value)


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if result == result else None  # NaN → None


def _int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        number = _float(value)
        return int(number) if number is not None else None


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def preview(value: Any, limit: int = 120) -> str:
    """*value* as one short line: JSON for containers, ``…`` when cut at *limit*."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            text = _str(value)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: max(limit - 1, 0)] + "…"
    return text


def format_cost(cost: float | None) -> str:
    """``$0.0123``, or ``—`` when no priced call has been seen."""
    if cost is None:
        return "—"
    return f"${cost:.4f}"


def format_duration(seconds: float | None) -> str:
    """A compact duration: ``0.4s``, ``12s``, ``3m 04s``, ``1h 02m``."""
    if seconds is None or seconds < 0:
        return "—"
    if seconds < 10:
        return f"{seconds:.1f}s"
    total = int(seconds)
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


# ---------------------------------------------------------------------------
# Ring buffer
# ---------------------------------------------------------------------------


class _HasId(Protocol):
    id: int


T = TypeVar("T", bound=_HasId)


class Ring(Generic[T]):
    """A capped list whose items get increasing ids (``item.id``), oldest dropped first.

    Ids stay contiguous among the kept items, so :meth:`get` is O(1) and a
    view can fetch only what is new with :meth:`since`.

    Args:
        cap: How many items are kept.
    """

    def __init__(self, cap: int = CAP) -> None:
        self.cap = max(int(cap), 1)
        self._items: deque[T] = deque()
        self._next = 1

    def append(self, item: T) -> T:
        """Give *item* the next id, keep it, and drop the oldest item when full."""
        item.id = self._next
        self._next += 1
        self._items.append(item)
        if len(self._items) > self.cap:
            self._items.popleft()
        return item

    @property
    def first_id(self) -> int:
        """The id of the oldest kept item (the next id when empty)."""
        return self._items[0].id if self._items else self._next

    @property
    def last_id(self) -> int:
        """The id of the newest item ever added (0 before the first)."""
        return self._next - 1

    def get(self, item_id: int) -> T | None:
        """The kept item with *item_id*, or ``None``."""
        index = item_id - self.first_id
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def since(self, item_id: int) -> list[T]:
        """The kept items whose id is greater than *item_id*, oldest first."""
        start = max(item_id + 1 - self.first_id, 0)
        if start >= len(self._items):
            return []
        if start == 0:
            return list(self._items)
        if start > len(self._items) // 2:
            # Walk from the right: cheaper for the common "a few new items" case.
            count = len(self._items) - start
            tail = list(islice(reversed(self._items), count))
            tail.reverse()
            return tail
        return list(islice(self._items, start, None))

    def last(self) -> T | None:
        """The newest kept item, or ``None``."""
        return self._items[-1] if self._items else None

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class RunRecord:
    """One run of the program (the first, or one after a restart)."""

    number: int
    started_at: float
    info: dict[str, Any]
    """The ``run.start`` payload."""
    exit: dict[str, Any] | None = None
    """The ``run.exit`` payload, once the run ended."""
    ended_at: float | None = None

    @property
    def mode(self) -> str:
        """``script``, ``chat``, ``once`` or ``serve`` (``""`` when unknown)."""
        return _str(self.info.get("mode"))

    @property
    def exit_code(self) -> int | None:
        """The exit code, once the run ended."""
        return _int(self.exit.get("code")) if self.exit is not None else None

    def duration(self, now: float) -> float:
        """Seconds the run has been running (until its exit, once it ended)."""
        if self.exit is not None:
            reported = _float(self.exit.get("duration_s"))
            if reported is not None:
                return reported
        end = self.ended_at if self.ended_at is not None else now
        return max(end - self.started_at, 0.0)


@dataclass(eq=False)
class ToolSpec:
    """One tool an agent can call."""

    name: str
    description: str = ""
    server: str = ""
    args_schema: Any = None


@dataclass(eq=False)
class ServerInfo:
    """One MCP server an agent is connected to."""

    agent: str
    name: str
    transport: str = ""
    target: str = ""
    tool_count: int = 0


@dataclass(eq=False)
class AgentInfo:
    """An agent the program built (``agent.built``)."""

    id: str
    model: str = ""
    observe_level: str = ""
    tools: list[ToolSpec] = field(default_factory=list)
    servers: list[ServerInfo] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    instructions: str = ""
    run: int = 0
    closed: bool = False


@dataclass(eq=False, kw_only=True)
class TimelineRow:
    """One timeline entry (``timeline`` event)."""

    id: int = 0
    ts: float
    agent: str
    event_type: str
    details: str
    duration_ms: float | None
    tokens: int | None
    entry: dict[str, Any]
    run: int


@dataclass(eq=False, kw_only=True)
class ToolCall:
    """One tool call: its ``tool.call`` entry paired with its result or error."""

    id: int = 0
    ts: float
    agent: str
    tool: str
    run_id: str | None
    arguments: str
    status: str = "running"
    """``running``, ``ok``, ``error`` or ``aborted`` (the program exited first)."""
    latency_ms: float | None = None
    result_preview: str | None = None
    result_length: int | None = None
    error: str | None = None
    error_type: str | None = None
    traceback: str | None = None
    reused: bool = False
    run: int = 0
    call_entry: dict[str, Any] | None = None
    result_entry: dict[str, Any] | None = None


@dataclass(eq=False, kw_only=True)
class AgentEventRow:
    """One agent event (``event``): invocations, slow tools, guardrails, approvals, …"""

    id: int = 0
    ts: float
    agent: str
    event_type: str
    severity: str
    data: Any
    event: dict[str, Any]
    run: int


@dataclass(eq=False)
class ChatText:
    """Streamed assistant text."""

    text: str


@dataclass(eq=False)
class ChatError:
    """An error shown inside a turn."""

    message: str


@dataclass(eq=False)
class ToolCard:
    """A tool call inside a chat turn (``tool_start`` / ``tool_end`` stream events)."""

    name: str
    display_name: str = ""
    arguments: Any = None
    index: int | None = None
    call_id: str | None = None
    summary: str | None = None
    success: bool | None = None
    """``None`` while the tool runs."""
    duration_ms: float | None = None
    reused: bool = False


ChatPart = ChatText | ChatError | ToolCard


@dataclass(eq=False, kw_only=True)
class ChatTurn:
    """One question and its answer."""

    id: int = 0
    turn_id: str
    input: str
    run: int
    started_at: float
    conversation_id: str | None = None
    parts: list[ChatPart] = field(default_factory=list)
    output: str | None = None
    error: str | None = None
    duration_ms: float | None = None
    done: bool = False

    @property
    def text(self) -> str:
        """All streamed assistant text so far."""
        return "".join(p.text for p in self.parts if isinstance(p, ChatText))


@dataclass(eq=False, kw_only=True)
class ChatNote:
    """A line between turns: a new run, an exit, a reset."""

    id: int = 0
    text: str
    style: str = "info"
    """``info``, ``run``, ``exit`` or ``error``."""
    ts: float = 0.0


ChatItem = ChatTurn | ChatNote


@dataclass(eq=False, kw_only=True)
class ConsoleLine:
    """One console line (script mode)."""

    id: int = 0
    stream: str
    """``stdout``, ``stderr``, ``stdin`` (what the user typed) or ``note``."""
    text: str
    ts: float
    run: int


@dataclass(eq=False, kw_only=True)
class LogLine:
    """One log record, or one line the program wrote to stderr (or stdout outside script mode)."""

    id: int = 0
    ts: float
    level: str
    levelno: int
    logger: str
    message: str
    exc_text: str | None
    source: str
    """``log``, ``stderr`` or ``stdout``."""
    run: int


@dataclass(eq=False, kw_only=True)
class HttpRequest:
    """One HTTP request (serve mode)."""

    id: int = 0
    ts: float
    method: str
    path: str
    status: int | None
    duration_ms: float | None
    request_id: str
    run: int


@dataclass(eq=False, kw_only=True)
class ApprovalRow:
    """One human-approval request (``approval.pending``) and its outcome (``approval.resolved``)."""

    id: int = 0
    request_id: str
    agent: str
    tool: str
    arguments: Any
    context_summary: str
    caller: str
    requested_at: float
    deadline: float | None
    request: dict[str, Any]
    run: int
    status: str = "pending"
    """``pending``, ``approved``, ``denied`` or ``expired``."""
    approved: bool | None = None
    reason: str | None = None
    resolved_at: float | None = None
    answered: bool | None = None
    """What the user answered here (``True`` approve, ``False`` deny), while
    the program has not confirmed it yet; ``None`` when not answered here."""

    @property
    def pending(self) -> bool:
        """Still waiting for the program to resolve it."""
        return self.status == "pending"

    @property
    def awaiting_user(self) -> bool:
        """Pending and not answered here yet (what the header badge counts)."""
        return self.status == "pending" and self.answered is None

    def time_left(self, now: float) -> float | None:
        """Seconds until the deadline (negative once passed; ``None`` without one)."""
        return None if self.deadline is None else self.deadline - now


@dataclass
class Changes:
    """What changed since the last :meth:`RunState.pop_changes`."""

    sections: set[str] = field(default_factory=set)
    """Names of the sections whose data changed (``timeline``, ``chat``, …)."""
    tool_calls: set[int] = field(default_factory=set)
    """Ids of tool calls updated in place (a result arrived)."""
    chat: set[int] = field(default_factory=set)
    """Ids of chat items updated in place (tokens streamed in)."""
    new_approvals: list[int] = field(default_factory=list)
    """Ids of approval requests that arrived."""
    new_run: bool = False
    """A ``run.start`` arrived."""

    def __bool__(self) -> bool:
        return bool(
            self.sections or self.tool_calls or self.chat or self.new_approvals or self.new_run
        )


# ---------------------------------------------------------------------------
# The state
# ---------------------------------------------------------------------------


class RunState:
    """Everything the terminal UI shows, folded from run events.

    Args:
        cap: Cap of the timeline, tool calls, agent events, logs, console
            and HTTP lists (oldest dropped first).
    """

    def __init__(self, *, cap: int = CAP) -> None:
        self.cap = cap
        self.runs: deque[RunRecord] = deque(maxlen=RUN_CAP)
        self.run_count = 0
        """How many runs started (``runs`` keeps the last :data:`RUN_CAP`)."""
        self.state: str | None = None
        """The latest ``state`` event's state (``None`` before one arrived)."""
        self.state_detail: str = ""
        self.last_exit: dict[str, Any] | None = None
        """The most recent ``run.exit`` payload, across restarts."""

        self.agents: dict[str, AgentInfo] = {}
        self.timeline: Ring[TimelineRow] = Ring(cap)
        self.timeline_agents: list[str] = []
        """Every agent seen in the timeline, in order of appearance (for the filter)."""
        self.tool_calls: Ring[ToolCall] = Ring(cap)
        self._open_calls: dict[str, ToolCall] = {}
        self.events: Ring[AgentEventRow] = Ring(cap)
        self.chat: Ring[ChatTurn | ChatNote] = Ring(CHAT_CAP)
        self._turns: dict[str, ChatTurn] = {}
        self.console: Ring[ConsoleLine] = Ring(cap)
        self.logs: Ring[LogLine] = Ring(cap)
        self.http: Ring[HttpRequest] = Ring(cap)
        self.approvals: Ring[ApprovalRow] = Ring(cap)
        """Every approval request, oldest first (pending and resolved)."""
        self._approvals: dict[str, ApprovalRow] = {}
        self.token_series: deque[int] = deque(maxlen=SPARK_CAP)
        """Total tokens of each recent LLM call, oldest first (the Metrics sparkline)."""

        self._stats_event: dict[str, Any] | None = None
        self._local_stats = RunStats()
        self._stats_cache: dict[str, Any] | None = None

        self.event_count = 0
        """Events applied (known kinds only)."""
        self.ignored_count = 0
        """Events ignored: not a dict, or an unknown kind."""
        self.last_seq = 0
        self._changes = Changes()
        self._touched: set[str] = set()

    # -- reading ------------------------------------------------------------

    @property
    def current_run(self) -> RunRecord | None:
        """The latest run, or ``None`` before the first ``run.start``."""
        return self.runs[-1] if self.runs else None

    @property
    def run_number(self) -> int:
        """1 for the first run, 2 after the first restart, … (0 before any)."""
        return self.run_count

    @property
    def mode(self) -> str:
        """The current run's mode (``""`` before the first ``run.start``)."""
        run = self.current_run
        return run.mode if run is not None else ""

    @property
    def chat_mode(self) -> bool:
        """Whether the first tab is a chat (an agent mode) rather than a console (script)."""
        return self.mode not in ("", "script")

    @property
    def project(self) -> str:
        """The project name (or the main file's name when the project has none)."""
        run = self.current_run
        if run is None:
            return ""
        name = _str(run.info.get("project"))
        if not name:
            main = _str(run.info.get("main"))
            name = main.replace("\\", "/").rsplit("/", 1)[-1]
        return name

    @property
    def exit_code(self) -> int | None:
        """The program's last exit code (``None`` until a run exited)."""
        return _int(self.last_exit.get("code")) if self.last_exit is not None else None

    @property
    def pending_approvals(self) -> list[ApprovalRow]:
        """Requests still pending, oldest first (answered here or not)."""
        return [row for row in self.approvals if row.pending]

    @property
    def approvals_awaiting(self) -> int:
        """Pending requests nobody answered yet (the header badge)."""
        return sum(1 for row in self.approvals if row.awaiting_user)

    def approval(self, request_id: str) -> ApprovalRow | None:
        """The request *request_id*, if it is kept."""
        return self._approvals.get(request_id)

    @property
    def stats(self) -> dict[str, Any]:
        """The current run's stats (the latest ``stats`` event; folded locally before one).

        Always has ``totals``, ``agents``, ``models`` and ``tools`` dicts.
        """
        if self._stats_cache is None:
            raw = (
                self._stats_event if self._stats_event is not None else self._local_stats.snapshot()
            )
            self._stats_cache = {
                "totals": _dict(raw.get("totals")),
                "agents": {k: _dict(v) for k, v in _dict(raw.get("agents")).items()},
                "models": {k: _dict(v) for k, v in _dict(raw.get("models")).items()},
                "tools": {k: _dict(v) for k, v in _dict(raw.get("tools")).items()},
            }
        return self._stats_cache

    def state_label(self) -> str:
        """``starting``, ``running``, ``idle``, ``busy``, ``exited code N``, ``restarting``, ``failed``."""
        run = self.current_run
        state = self.state or ("running" if run is not None else "starting")
        code = run.exit_code if run is not None else None
        if state in ("exited", "failed") and code is not None:
            return f"{state} code {code}"
        return state

    def header_fields(self, now: float | None = None) -> dict[str, str]:
        """What the header bar shows, as text: project, mode, state, run, uptime, totals."""
        now = time.time() if now is None else now
        run = self.current_run
        totals = self.stats["totals"]
        return {
            "project": self.project or "promptise run",
            "version": _str(run.info.get("project_version")) if run is not None else "",
            "mode": self.mode or "—",
            "state": self.state_label(),
            "run": f"run #{self.run_number}" if self.run_number > 1 else "",
            "uptime": format_duration(run.duration(now)) if run is not None else "—",
            "llm_calls": f"{_int(totals.get('llm_calls')) or 0:,}",
            "tokens": f"{_int(totals.get('total_tokens')) or 0:,}",
            "cost": format_cost(_float(totals.get("cost_usd"))),
            "errors": f"{_int(totals.get('errors')) or 0:,}",
            "approvals": str(self.approvals_awaiting),
        }

    def pop_changes(self) -> Changes:
        """What changed since the last call (and start collecting anew)."""
        changes, self._changes = self._changes, Changes()
        return changes

    # -- local input (what the user typed) -----------------------------------

    def add_user_turn(self, turn_id: str, text: str) -> ChatTurn:
        """Show a question at once, before its ``turn.start`` arrives."""
        turn = self._turn(turn_id, text)
        return turn

    def add_console_input(self, text: str) -> None:
        """Echo a line the user sent to the program's stdin."""
        self._console("stdin", text)

    def add_note(self, text: str, style: str = "info") -> None:
        """Add a note line to the chat and the console."""
        self._note(text, style)

    def mark_approval_answered(self, request_id: str, approved: bool) -> ApprovalRow | None:
        """Remember that the user answered *request_id* here (until it is resolved)."""
        row = self._approvals.get(request_id)
        if row is None or not row.pending:
            return None
        row.answered = approved
        self._touch(APPROVALS, HEADER)
        return row

    # -- applying events ------------------------------------------------------

    def apply(self, event: Any) -> set[str]:
        """Fold one run event in; return the sections it changed (empty when ignored).

        Never raises: a malformed event is ignored or applied as far as it makes sense.
        """
        if not isinstance(event, dict):
            self.ignored_count += 1
            return set()
        handler = self._HANDLERS.get(_str(event.get("kind")))
        if handler is None:
            self.ignored_count += 1
            return set()
        # Everything shown comes from the program, its tools and HTTP clients:
        # no terminal control sequence gets through to the screen.
        event = strip_controls(event)
        seq = _int(event.get("seq"))
        if seq is not None:
            self.last_seq = max(self.last_seq, seq)
        self.event_count += 1
        self._touched = set()
        try:
            handler(self, event)
        except Exception:
            # Defensive: a field shape nobody anticipated.  Keep going.
            self.ignored_count += 1
        return self._touched

    def _touch(self, *sections: str) -> None:
        self._changes.sections.update(sections)
        self._touched.update(sections)

    def _ts(self, event: dict[str, Any], *candidates: Any) -> float:
        for value in candidates:
            number = _float(value)
            if number is not None and number > 1_000_000_000:  # an epoch time, not monotonic
                return number
        number = _float(event.get("ts"))
        return number if number is not None else time.time()

    # run lifecycle

    def _on_run_start(self, event: dict[str, Any]) -> None:
        info = {k: v for k, v in event.items() if k not in _ENVELOPE}
        self.run_count += 1
        record = RunRecord(number=self.run_count, started_at=self._ts(event), info=info)
        previous = self.current_run
        self.runs.append(record)
        self.state = None
        self.state_detail = ""
        if previous is not None:
            self._abort_open(previous.number)
            self._note(f"run #{record.number}", "run")
        # Per-run views start empty: the new process builds its agents again.
        self.agents.clear()
        self._open_calls.clear()
        self._stats_event = None
        self._local_stats = RunStats()
        self._stats_cache = None
        self.token_series.clear()
        self._changes.new_run = True
        self._touch(HEADER, RUN, AGENTS, METRICS, CHAT, CONSOLE)

    def _on_run_exit(self, event: dict[str, Any]) -> None:
        payload = {k: v for k, v in event.items() if k not in _ENVELOPE}
        self.last_exit = payload
        run = self.current_run
        if run is None:
            self.run_count += 1
            run = RunRecord(number=self.run_count, started_at=self._ts(event), info={})
            self.runs.append(run)
        run.exit = payload
        run.ended_at = self._ts(event)
        if self.state not in ("failed", "restarting"):
            self.state = "exited"
        self._abort_open(run.number)
        code = _int(payload.get("code"))
        took = format_duration(run.duration(run.ended_at))
        error = _str(payload.get("error"))
        text = f"exited with code {code if code is not None else '?'} after {took}"
        if error:
            text += f" · {preview(error, 200)}"
        self._note(text, "exit" if not code else "error")
        self._touch(HEADER, RUN)

    def _abort_open(self, run_number: int) -> None:
        for row in self.approvals:
            if row.run == run_number and row.pending:
                row.status = "expired"
                row.reason = row.reason or "the program exited"
                row.resolved_at = time.time()
                self._touch(APPROVALS, HEADER)
        for call in list(self._open_calls.values()):
            if call.run == run_number and call.status == "running":
                call.status = "aborted"
                self._changes.tool_calls.add(call.id)
                self._touch(TOOL_CALLS)
        self._open_calls = {k: v for k, v in self._open_calls.items() if v.status == "running"}
        for turn in list(self._turns.values()):
            if turn.run == run_number and not turn.done:
                turn.done = True
                if turn.error is None:
                    turn.error = "the program exited"
                for part in turn.parts:
                    if isinstance(part, ToolCard) and part.success is None:
                        part.success = False
                self._changes.chat.add(turn.id)
                self._touch(CHAT)

    def _on_state(self, event: dict[str, Any]) -> None:
        state = _str(event.get("state"))
        if state:
            self.state = state
            self.state_detail = _str(event.get("detail"))
            self._touch(HEADER)

    # agents

    def _on_agent_built(self, event: dict[str, Any]) -> None:
        agent_id = _str(event.get("agent")) or f"agent-{len(self.agents) + 1}"
        tools = []
        for raw in _list(event.get("tools")):
            item = _dict(raw)
            name = _str(item.get("name"))
            if not name:
                continue
            tools.append(
                ToolSpec(
                    name=name,
                    description=_str(item.get("description")),
                    server=_str(item.get("server")),
                    args_schema=item.get("args_schema"),
                )
            )
        servers = []
        for raw in _list(event.get("servers")):
            item = _dict(raw)
            name = _str(item.get("name"))
            if not name:
                continue
            servers.append(
                ServerInfo(
                    agent=agent_id,
                    name=name,
                    transport=_str(item.get("transport")),
                    target=_str(item.get("target")),
                    tool_count=sum(1 for t in tools if t.server == name),
                )
            )
        model = _str(event.get("model"))
        run = self.current_run
        self.agents[agent_id] = AgentInfo(
            id=agent_id,
            model=model,
            observe_level=_str(event.get("observe_level")),
            tools=tools,
            servers=servers,
            features=_dict(event.get("features")),
            instructions=_str(event.get("instructions")),
            run=run.number if run is not None else 0,
        )
        self._local_stats.set_model(agent_id, model or None)
        if self._stats_event is None:
            self._stats_cache = None
            self._touch(METRICS)
        self._touch(AGENTS)

    def _on_agent_closed(self, event: dict[str, Any]) -> None:
        agent = self.agents.get(_str(event.get("agent")))
        if agent is not None:
            agent.closed = True
            self._touch(AGENTS)

    # observability

    def _on_timeline(self, event: dict[str, Any]) -> None:
        entry = _dict(event.get("entry"))
        meta = _dict(entry.get("metadata"))
        agent = _str(event.get("agent")) or _str(entry.get("agent_id")) or "?"
        etype = _str(entry.get("event_type")) or "?"
        duration = _float(entry.get("duration"))
        tokens = None
        if etype == "llm.end" or "total_tokens" in meta:
            pt = _int(meta.get("prompt_tokens")) or 0
            ct = _int(meta.get("completion_tokens")) or 0
            tokens = _int(meta.get("total_tokens"))
            if tokens is None and (pt or ct):
                tokens = pt + ct
        ts = self._ts(event, entry.get("timestamp"))
        run = self.current_run
        run_number = run.number if run is not None else 0
        self.timeline.append(
            TimelineRow(
                ts=ts,
                agent=agent,
                event_type=etype,
                details=_str(entry.get("details")),
                duration_ms=duration * 1000 if duration is not None else None,
                tokens=tokens,
                entry=entry,
                run=run_number,
            )
        )
        if agent not in self.timeline_agents:
            self.timeline_agents.append(agent)
        self._touch(TIMELINE)

        if etype == "llm.end":
            self.token_series.append(max(tokens or 0, 0))
            self._touch(METRICS)
        if etype == "tool.call":
            self._tool_call(agent, entry, meta, ts, run_number)
        elif etype in ("tool.result", "tool.error"):
            self._tool_result(agent, etype, entry, meta, ts, run_number)

        try:
            self._local_stats.add_entry(agent, entry)
        except Exception:
            pass
        if self._stats_event is None:
            self._stats_cache = None
            self._touch(METRICS, HEADER)

    def _tool_call(
        self, agent: str, entry: dict[str, Any], meta: dict[str, Any], ts: float, run: int
    ) -> None:
        if "arguments" in meta:
            arguments = _str(meta.get("arguments"))
        elif "arguments_length" in meta:
            arguments = f"<{_int(meta.get('arguments_length')) or 0} chars, not recorded>"
        else:
            arguments = ""
        call = self.tool_calls.append(
            ToolCall(
                ts=ts,
                agent=agent,
                tool=_str(meta.get("tool_name")) or "unknown",
                run_id=_opt_str(meta.get("run_id")),
                arguments=arguments,
                reused=bool(meta.get("reused")),
                run=run,
                call_entry=entry,
            )
        )
        if call.run_id:
            self._open_calls[call.run_id] = call
            if len(self._open_calls) > self.cap:
                self._open_calls.pop(next(iter(self._open_calls)))
        self._touch(TOOL_CALLS)

    def _tool_result(
        self,
        agent: str,
        etype: str,
        entry: dict[str, Any],
        meta: dict[str, Any],
        ts: float,
        run: int,
    ) -> None:
        run_id = _opt_str(meta.get("run_id"))
        call = self._open_calls.pop(run_id, None) if run_id else None
        if call is None:
            # No matching call (not recorded, or long dropped): show the result alone.
            call = self.tool_calls.append(
                ToolCall(
                    ts=ts,
                    agent=agent,
                    tool=_str(meta.get("tool_name")) or "unknown",
                    run_id=run_id,
                    arguments="",
                    run=run,
                )
            )
        else:
            self._changes.tool_calls.add(call.id)
        failed = etype == "tool.error" or _str(meta.get("status")) == "error"
        call.status = "error" if failed else "ok"
        latency = _float(meta.get("latency_ms"))
        if latency is None:
            duration = _float(entry.get("duration"))
            latency = round(duration * 1000, 1) if duration is not None else None
        call.latency_ms = latency
        if meta.get("tool_name") and call.tool == "unknown":
            call.tool = _str(meta.get("tool_name"))
        call.result_preview = _opt_str(meta.get("result_preview"))
        call.result_length = _int(meta.get("result_length"))
        call.error = _opt_str(meta.get("error"))
        call.error_type = _opt_str(meta.get("error_type"))
        call.traceback = _opt_str(meta.get("traceback"))
        if meta.get("reused"):
            call.reused = True
        call.result_entry = entry
        self._touch(TOOL_CALLS)

    def _on_event(self, event: dict[str, Any]) -> None:
        payload = _dict(event.get("event"))
        run = self.current_run
        self.events.append(
            AgentEventRow(
                ts=self._ts(event, payload.get("timestamp")),
                agent=_str(event.get("agent")) or _str(payload.get("agent_id")) or "?",
                event_type=_str(payload.get("event_type")) or "?",
                severity=_str(payload.get("severity")) or "info",
                data=payload.get("data"),
                event=payload,
                run=run.number if run is not None else 0,
            )
        )
        self._touch(EVENTS)

    def _on_stats(self, event: dict[str, Any]) -> None:
        self._stats_event = {k: v for k, v in event.items() if k not in _ENVELOPE}
        self._stats_cache = None
        self._touch(METRICS, HEADER)

    # chat

    def _turn(self, turn_id: str, text: str = "") -> ChatTurn:
        turn = self._turns.get(turn_id)
        if turn is not None:
            if text and not turn.input:
                turn.input = text
                self._changes.chat.add(turn.id)
                self._touch(CHAT)
            return turn
        run = self.current_run
        turn = ChatTurn(
            turn_id=turn_id,
            input=text,
            run=run.number if run is not None else 0,
            started_at=time.time(),
        )
        self.chat.append(turn)
        self._turns[turn_id] = turn
        while len(self._turns) > CHAT_CAP:
            self._turns.pop(next(iter(self._turns)))
        self._touch(CHAT)
        return turn

    def _on_turn_start(self, event: dict[str, Any]) -> None:
        turn = self._turn(_str(event.get("turn_id")) or "?", _str(event.get("input")))
        turn.conversation_id = _opt_str(event.get("conversation_id"))

    def _on_stream(self, event: dict[str, Any]) -> None:
        turn = self._turn(_str(event.get("turn_id")) or "?")
        stream = _dict(event.get("event"))
        kind = _str(stream.get("type"))
        changed = True
        if kind == "token":
            text = _str(stream.get("text", stream.get("content")))
            if not text:
                changed = False
            elif turn.parts and isinstance(turn.parts[-1], ChatText):
                turn.parts[-1].text += text
            else:
                turn.parts.append(ChatText(text))
        elif kind == "tool_start":
            name = _str(stream.get("tool_name")) or "tool"
            turn.parts.append(
                ToolCard(
                    name=name,
                    display_name=_str(stream.get("tool_display_name", stream.get("display_name"))),
                    arguments=stream.get("arguments"),
                    index=_int(stream.get("tool_index")),
                    call_id=_opt_str(stream.get("tool_call_id")),
                )
            )
        elif kind == "tool_end":
            card = self._find_card(turn, stream)
            if card is None:
                card = ToolCard(name=_str(stream.get("tool_name")) or "tool")
                turn.parts.append(card)
            success = stream.get("success")
            card.success = bool(success) if success is not None else True
            summary = stream.get("tool_summary", stream.get("result"))
            card.summary = _opt_str(summary)
            card.duration_ms = _float(stream.get("duration_ms"))
            card.reused = bool(stream.get("reused"))
        elif kind == "done":
            full = _str(stream.get("full_response"))
            if full and not turn.text:
                turn.parts.append(ChatText(full))
            if turn.output is None and full:
                turn.output = full
        elif kind == "error":
            message = _str(stream.get("message")) or "error"
            turn.parts.append(ChatError(message))
        else:
            changed = False
        if changed:
            self._changes.chat.add(turn.id)
            self._touch(CHAT)

    @staticmethod
    def _find_card(turn: ChatTurn, stream: dict[str, Any]) -> ToolCard | None:
        cards = [p for p in turn.parts if isinstance(p, ToolCard)]
        call_id = _opt_str(stream.get("tool_call_id"))
        if call_id:
            for card in cards:
                if card.call_id == call_id:
                    return card
        index = _int(stream.get("tool_index"))
        name = _str(stream.get("tool_name"))
        if index is not None:
            for card in cards:
                if card.index == index and card.success is None and (not name or card.name == name):
                    return card
        running = [c for c in cards if c.success is None]
        for card in reversed(running):
            if not name or card.name == name:
                return card
        return None

    def _on_turn_end(self, event: dict[str, Any]) -> None:
        turn = self._turn(_str(event.get("turn_id")) or "?")
        output = _opt_str(event.get("output"))
        if output is not None:
            turn.output = output
            if output and not turn.text:
                turn.parts.append(ChatText(output))
        turn.duration_ms = _float(event.get("duration_ms"))
        error = _opt_str(event.get("error"))
        if error:
            turn.error = error
        for part in turn.parts:
            if isinstance(part, ToolCard) and part.success is None:
                part.success = error is None
        turn.done = True
        self._changes.chat.add(turn.id)
        self._touch(CHAT)

    def _note(self, text: str, style: str) -> None:
        self.chat.append(ChatNote(text=text, style=style, ts=time.time()))
        self._console("note", text)
        self._touch(CHAT)

    # serve, logs, output

    def _on_http(self, event: dict[str, Any]) -> None:
        run = self.current_run
        self.http.append(
            HttpRequest(
                ts=self._ts(event),
                method=_str(event.get("method")) or "?",
                path=_str(event.get("path")),
                status=_int(event.get("status")),
                duration_ms=_float(event.get("duration_ms")),
                request_id=_str(event.get("request_id")),
                run=run.number if run is not None else 0,
            )
        )
        self._touch(HTTP)

    def _log(
        self,
        ts: float,
        level: str,
        logger: str,
        message: str,
        exc_text: str | None,
        source: str,
    ) -> None:
        run = self.current_run
        self.logs.append(
            LogLine(
                ts=ts,
                level=level,
                levelno=LEVELS.get(level, 20),
                logger=logger,
                message=message,
                exc_text=exc_text,
                source=source,
                run=run.number if run is not None else 0,
            )
        )
        self._touch(LOGS)

    def _on_log(self, event: dict[str, Any]) -> None:
        level = _str(event.get("level")).upper() or "INFO"
        self._log(
            self._ts(event),
            level,
            _str(event.get("logger")),
            _str(event.get("message")),
            _opt_str(event.get("exc_text")) or None,
            "log",
        )

    def _console(self, stream: str, text: str) -> None:
        run = self.current_run
        self.console.append(
            ConsoleLine(
                stream=stream,
                text=text,
                ts=time.time(),
                run=run.number if run is not None else 0,
            )
        )
        self._touch(CONSOLE)

    def _on_output(self, event: dict[str, Any]) -> None:
        stream = "stderr" if _str(event.get("stream")) == "stderr" else "stdout"
        text = _str(event.get("text")).rstrip("\r\n")
        self._console(stream, text)
        # Outside script mode the console tab is a chat: printed lines go to the logs.
        if stream == "stderr" or self.chat_mode:
            self._log(
                self._ts(event),
                "WARNING" if stream == "stderr" else "INFO",
                stream,
                text,
                None,
                stream,
            )

    # approvals

    def _on_approval_pending(self, event: dict[str, Any]) -> None:
        request = _dict(event.get("request"))
        request_id = _str(request.get("request_id")) or _str(event.get("request_id"))
        if not request_id:
            return
        existing = self._approvals.get(request_id)
        if existing is not None and existing.pending:
            return  # sent twice (a replay): keep the first
        requested_at = self._ts(event, request.get("timestamp"))
        deadline = _float(event.get("deadline"))
        if deadline is None:
            timeout = _float(request.get("timeout"))
            deadline = requested_at + timeout if timeout is not None else None
        run = self.current_run
        row = self.approvals.append(
            ApprovalRow(
                request_id=request_id,
                agent=_str(event.get("agent")) or _str(request.get("agent_id")) or "?",
                tool=_str(request.get("tool_name")) or "?",
                arguments=request.get("arguments"),
                context_summary=_str(request.get("context_summary")),
                caller=_str(request.get("caller_user_id")),
                requested_at=requested_at,
                deadline=deadline,
                request=request,
                run=run.number if run is not None else 0,
            )
        )
        self._approvals[request_id] = row
        while len(self._approvals) > self.cap:
            self._approvals.pop(next(iter(self._approvals)))
        self._changes.new_approvals.append(row.id)
        self._touch(APPROVALS, HEADER)

    def _on_approval_resolved(self, event: dict[str, Any]) -> None:
        request_id = _str(event.get("request_id"))
        if not request_id:
            return
        row = self._approvals.get(request_id)
        if row is None:
            # Resolved without (or long after) its request: keep the outcome.
            run = self.current_run
            row = self.approvals.append(
                ApprovalRow(
                    request_id=request_id,
                    agent="?",
                    tool="?",
                    arguments=None,
                    context_summary="",
                    caller="",
                    requested_at=self._ts(event),
                    deadline=None,
                    request={},
                    run=run.number if run is not None else 0,
                )
            )
            self._approvals[request_id] = row
        approved = event.get("approved")
        row.approved = approved if isinstance(approved, bool) else None
        outcome = _str(event.get("outcome"))
        if outcome not in ("approved", "denied", "expired"):
            outcome = (
                "approved" if row.approved else "denied" if row.approved is False else "expired"
            )
        row.status = outcome
        row.reason = _opt_str(event.get("reason")) or None
        row.resolved_at = self._ts(event)
        row.answered = None
        self._touch(APPROVALS, HEADER)

    _HANDLERS: dict[str, Any] = {
        "run.start": _on_run_start,
        "run.exit": _on_run_exit,
        "state": _on_state,
        "agent.built": _on_agent_built,
        "agent.closed": _on_agent_closed,
        "timeline": _on_timeline,
        "event": _on_event,
        "stats": _on_stats,
        "turn.start": _on_turn_start,
        "stream": _on_stream,
        "turn.end": _on_turn_end,
        "http": _on_http,
        "log": _on_log,
        "output": _on_output,
        "approval.pending": _on_approval_pending,
        "approval.resolved": _on_approval_resolved,
    }
