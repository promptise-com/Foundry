"""Context compaction for long tool loops.

A tool loop sends the model its whole transcript on every call, so the
request grows with every tool result.  Compaction replaces that transcript
with a bounded view once the loop gets long:

- **Pinned, always sent:** every system message from the input (instructions
  a caller or the runtime added, such as ``[Context State]``) and the current
  user question.
- **Earlier conversation:** chat history before the current question becomes
  a short note, so the model answers the *current* question.
- **Latest exchange, verbatim:** the model's last tool call(s) and their
  results, so it sees the outcome of its last action in flow.
- **Ledger of older results:** one entry per earlier tool call (deduplicated,
  last result wins).  A result longer than ``keep_result_chars`` is cut to an
  excerpt that names the call, so the model can call it again for the full
  text.  When a token budget is set, the oldest entries shrink to a bare
  reference until the view fits.

There is no LLM summarization: compaction is deterministic and costs no
extra model calls.  Results that must stay verbatim need a higher
``keep_result_chars`` or ``context_compaction=False``.

Example::

    from promptise import build_agent
    from promptise.engine import ContextCompaction

    agent = await build_agent(
        ...,
        context_compaction=ContextCompaction(after_tool_results=10, keep_result_chars=8000),
    )
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    convert_to_messages,
)

logger = logging.getLogger("promptise.engine")

__all__ = [
    "ContextCompaction",
    "LEDGER_HEADER",
    "build_compacted_view",
    "estimate_tokens",
    "normalize_messages",
]

LEDGER_HEADER = "Facts already gathered"
HISTORY_HEADER = "Earlier in this conversation"


def estimate_tokens(text: str) -> int:
    """Rough token count (4 characters per token)."""
    return math.ceil(len(text) / 4) if text else 0


@dataclass(frozen=True)
class ContextCompaction:
    """Settings for compacting a long tool loop.

    Pass one to ``build_agent(context_compaction=...)`` (or ``True`` /
    ``False`` / an ``int`` for ``after_tool_results``), or to
    ``PromptNode(compaction=...)`` for a single node.

    Attributes:
        enabled: ``False`` keeps the full transcript: nodes with
            ``context_scope="auto"`` never compact.
        after_tool_results: ``"auto"`` nodes compact once the run has this
            many tool results.
        max_tokens: Also compact once the full transcript is estimated to
            pass this many tokens, and shrink the ledger until the compacted
            view fits.  ``None`` means no token budget.  ``build_agent``
            sets it from a ``ContextEngine``'s budget.
        keep_result_chars: Older tool results longer than this are cut to
            an excerpt of this many characters in the ledger.
        history_messages: How many earlier conversation messages the
            history note keeps.
        history_chars: Each earlier message is cut to this many characters
            in the history note.
        count_tokens: Token counter for ``max_tokens``.  Defaults to a
            4-characters-per-token estimate.
    """

    enabled: bool = True
    after_tool_results: int = 6
    max_tokens: int | None = None
    keep_result_chars: int = 2000
    history_messages: int = 6
    history_chars: int = 400
    count_tokens: Callable[[str], int] | None = None

    @classmethod
    def coerce(cls, value: Any) -> ContextCompaction:
        """Turn a ``context_compaction`` argument into settings.

        ``None`` / ``True`` give the defaults, ``False`` turns compaction
        off, an ``int`` sets ``after_tool_results``.
        """
        if value is None or value is True:
            return cls()
        if value is False:
            return cls(enabled=False)
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            if value < 1:
                raise ValueError("context_compaction as an int must be at least 1 tool result")
            return cls(after_tool_results=value)
        raise TypeError(
            "context_compaction must be a bool, an int (tool results before compacting) "
            f"or a ContextCompaction, not {type(value).__name__}"
        )

    def with_budget(self, max_tokens: int, count_tokens: Callable[[str], int]) -> ContextCompaction:
        """These settings with a token budget, unless one is already set."""
        if self.max_tokens is not None:
            return self
        return replace(self, max_tokens=max_tokens, count_tokens=self.count_tokens or count_tokens)

    def tokens(self, messages: Iterable[Any]) -> int:
        """Token count of *messages* (content plus tool-call arguments)."""
        count = self.count_tokens or estimate_tokens
        return sum(count(_message_text(m)) for m in messages)


# ---------------------------------------------------------------------------
# Input normalization and the current turn
# ---------------------------------------------------------------------------


def normalize_messages(messages: Iterable[Any]) -> list[Any]:
    """Convert dict / tuple / str messages to LangChain message objects.

    ``{"role": "user", "content": ...}`` becomes a ``HumanMessage`` and so
    on.  Message objects pass through unchanged; anything that does not
    convert is kept as given.
    """
    out: list[Any] = []
    for m in messages:
        if isinstance(m, BaseMessage) or not isinstance(m, (dict, tuple, str)):
            out.append(m)
            continue
        try:
            out.extend(convert_to_messages([m]))
        except Exception:  # not a message shape we know; keep it as given
            out.append(m)
    return out


def split_input(messages: Sequence[Any]) -> tuple[Any | None, list[Any]]:
    """The current question and the input up to and including it.

    The question is the last ``HumanMessage`` in *messages*; the head is
    every message up to it (system messages and history).  Without a
    ``HumanMessage`` the question is ``None`` and the head is every
    system message.
    """
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            return messages[i], list(messages[: i + 1])
    return None, [m for m in messages if isinstance(m, SystemMessage)]


# ---------------------------------------------------------------------------
# The compacted view
# ---------------------------------------------------------------------------


def _message_text(m: Any) -> str:
    content = getattr(m, "content", m)
    text = content if isinstance(content, str) else str(content or "")
    calls = getattr(m, "tool_calls", None)
    if calls:
        text += json.dumps([{"name": c.get("name"), "args": c.get("args")} for c in calls])
    return text


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _run_messages(messages: Sequence[Any], question: Any | None, head: Sequence[Any]) -> list[Any]:
    """Messages this run produced: everything after the current question."""
    if question is not None:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i] is question:
                return list(messages[i + 1 :])
    head_ids = {id(m) for m in head}
    return [m for m in messages if id(m) not in head_ids and not isinstance(m, SystemMessage)]


def _last_exchange(run: Sequence[Any]) -> list[Any]:
    """The trailing tool exchange (assistant tool call + its results), verbatim.

    When the run ends on an assistant message without tool calls (another
    node's answer), that message is the last exchange.
    """
    if not run:
        return []
    if isinstance(run[-1], AIMessage) and not run[-1].tool_calls:
        return [run[-1]]
    tail: list[Any] = []
    for m in reversed(run):
        if isinstance(m, ToolMessage):
            tail.append(m)
            continue
        if isinstance(m, AIMessage) and m.tool_calls:
            tail.append(m)
        break
    tail.reverse()
    # Tool results without the assistant message that asked for them are
    # rejected by providers; drop such orphans.
    return tail if tail and isinstance(tail[0], AIMessage) else []


def _history_note(history: Sequence[Any], settings: ContextCompaction) -> SystemMessage | None:
    turns: list[str] = []
    for m in history:
        if isinstance(m, HumanMessage):
            role = "User"
        elif isinstance(m, AIMessage):
            role = "Assistant"
        else:
            continue
        text = _message_text(m) if not isinstance(m, AIMessage) else str(m.content or "")
        if text.strip():
            turns.append(f"{role}: {_clip(' '.join(text.split()), settings.history_chars)}")
    if not turns:
        return None
    kept = turns[-settings.history_messages :] if settings.history_messages > 0 else []
    lines = [
        f"{HISTORY_HEADER} (shortened). Answer the user's current request, "
        "which is the next user message:"
    ]
    if len(turns) > len(kept):
        lines.append(f"({len(turns) - len(kept)} earlier messages not shown)")
    lines.extend(kept)
    return SystemMessage(content="\n".join(lines))


@dataclass
class _Entry:
    ref: int
    call: str
    tool: str
    result: str
    shown: str


def _ledger_entries(older: Sequence[Any], settings: ContextCompaction) -> list[_Entry]:
    """One entry per earlier tool call, deduplicated by (tool, args)."""
    calls: dict[str, tuple[str, str]] = {}
    for m in older:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls:
                args = json.dumps(tc.get("args", {}), sort_keys=True, default=str)
                calls[str(tc.get("id"))] = (str(tc.get("name", "?")), f"{tc.get('name')}({args})")
    entries: dict[str, _Entry] = {}
    for m in older:
        if not isinstance(m, ToolMessage):
            continue
        tool, call = calls.get(str(m.tool_call_id), ("?", f"tool call {m.tool_call_id}"))
        result = m.content if isinstance(m.content, str) else str(m.content)
        entries.pop(call, None)  # last result wins, listed in call order
        entries[call] = _Entry(0, call, tool, result, _excerpt(tool, result, settings))
    for i, e in enumerate(entries.values(), 1):
        e.ref = i
    return list(entries.values())


def _excerpt(tool: str, result: str, settings: ContextCompaction) -> str:
    limit = max(0, settings.keep_result_chars)
    if len(result) <= limit:
        return result
    return (
        f"{result[:limit].rstrip()} … [shortened: {len(result) - limit:,} of "
        f"{len(result):,} characters not shown; call {tool} with the same "
        "arguments to see the full result (several in one step if you need them "
        "together)]"
    )


def _reference(e: _Entry) -> str:
    return (
        f"[result not shown: {len(e.result):,} characters; call {e.tool} with "
        "the same arguments to see it]"
    )


def _ledger_message(entries: Sequence[_Entry]) -> SystemMessage:
    return SystemMessage(
        content=(
            f"{LEDGER_HEADER} by your earlier tool calls (do not call a tool "
            "again for these unless you need a result marked as not shown in "
            "full):\n" + "\n".join(f"- [{e.ref}] {e.call} = {e.shown}" for e in entries)
        )
    )


def build_compacted_view(
    messages: Sequence[Any],
    *,
    question: Any | None,
    head: Sequence[Any],
    settings: ContextCompaction,
    scoped: bool = False,
) -> list[Any]:
    """The bounded message list a compacting node sends instead of *messages*.

    Order: pinned input system messages, the history note, the current
    question, the latest exchange, then the ledger of older tool results.
    With ``scoped=True`` (``context_scope="scoped"``) the run's trailing
    tool loop is kept verbatim and no ledger is built.

    The node's own system prompt is not included; the caller inserts it.
    """
    view: list[Any] = []
    history = [m for m in head if m is not question and not isinstance(m, SystemMessage)]
    for m in head:
        if isinstance(m, SystemMessage):
            view.append(m)
        elif m is question:
            note = _history_note(history, settings)
            if note is not None:
                view.append(note)
            view.append(m)

    run = _run_messages(messages, question, head)
    if scoped:
        loop: list[Any] = []
        for m in reversed(run):
            if isinstance(m, ToolMessage) or (isinstance(m, AIMessage) and m.tool_calls):
                loop.append(m)
            else:
                break
        loop.reverse()
        while loop and isinstance(loop[0], ToolMessage):
            loop.pop(0)
        return view + loop

    last = _last_exchange(run)
    last_ids = {id(m) for m in last}
    entries = _ledger_entries([m for m in run if id(m) not in last_ids], settings)
    if not entries:
        return view + last

    compacted = view + last + [_ledger_message(entries)]
    if settings.max_tokens is not None:
        # Over budget: shrink the oldest entries to references until it fits.
        for e in entries:
            if settings.tokens(compacted) <= settings.max_tokens:
                break
            e.shown = _reference(e)
            compacted[-1] = _ledger_message(entries)
        used = settings.tokens(compacted)
        if used > settings.max_tokens:
            logger.warning(
                "Context compaction: %d tokens after compacting, over the %d-token "
                "budget (the pinned messages and the latest exchange are never cut).",
                used,
                settings.max_tokens,
            )
    return compacted
