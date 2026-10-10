"""Conversation flow state machine — turn-aware prompt evolution.

The system prompt is a living document that adapts to conversation state.
Phases control which blocks are active, and the prompt evolves across turns.

Example::

    from promptise.prompts.flows import ConversationFlow, phase
    from promptise.prompts.blocks import Identity, Rules, Section, OutputFormat

    class SupportFlow(ConversationFlow):
        base_blocks = [
            Identity("Customer support agent"),
            Rules(["Be empathetic", "Ask before escalating"]),
        ]

        @phase("greeting", initial=True)
        async def greet(self, ctx):
            ctx.activate(Section("greet", "Ask how you can help today."))

        @phase("investigate")
        async def investigate(self, ctx):
            ctx.deactivate("greet")
            ctx.activate(Section("investigate", "Ask clarifying questions."))

        @phase("resolve")
        async def resolve(self, ctx):
            ctx.activate(OutputFormat(format="markdown"))

    flow = SupportFlow()
    prompt = await flow.start("Hi")                   # Enter greeting phase
    prompt = await flow.next_turn("My app crashed")   # Still in greeting
    await flow.transition("investigate")              # Move to investigate
    prompt = await flow.next_turn("It crashes on login")

A flow instance holds one conversation's state.  ``build_agent(flow=...)``
treats the flow you pass as a template and keeps a separate copy per
session or caller; see :class:`FlowSessions`.
"""

from __future__ import annotations

import asyncio
import copy
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Union

from .blocks import (
    AssembledPrompt,
    Block,
    BlockContext,
    PromptAssembler,
)
from .inspector import PromptInspector

__all__ = [
    "Phase",
    "TurnContext",
    "ConversationFlow",
    "FlowSessions",
    "FlowSource",
    "phase",
]


# ---------------------------------------------------------------------------
# Phase
# ---------------------------------------------------------------------------


@dataclass
class Phase:
    """A named phase in a conversation flow.

    Phases can carry blocks that are automatically activated on entry
    and deactivated on exit, plus optional lifecycle hooks.
    """

    name: str
    blocks: list[Block] = field(default_factory=list)
    on_enter: Callable[..., Any] | None = None
    on_exit: Callable[..., Any] | None = None


# ---------------------------------------------------------------------------
# TurnContext — mutable view given to phase handlers
# ---------------------------------------------------------------------------


class TurnContext:
    """Mutable context passed to ``@phase`` handlers each turn.

    Phase handlers use this to activate/deactivate blocks, fill
    context slots, and trigger phase transitions.
    """

    def __init__(
        self,
        flow: ConversationFlow,
        turn: int,
        phase_name: str,
        state: dict[str, Any],
        history: list[dict[str, str]],
    ) -> None:
        self._flow = flow
        self._turn = turn
        self._phase_name = phase_name
        self._state = state
        self._history = history
        self._pending_transition: str | None = None

    # -- Read-only properties -----------------------------------------------

    @property
    def turn(self) -> int:
        """Current turn number (0-based)."""
        return self._turn

    @property
    def phase(self) -> str:
        """Current phase name."""
        return self._phase_name

    @property
    def state(self) -> dict[str, Any]:
        """Arbitrary flow state dict.  Mutate freely."""
        return self._state

    @property
    def history(self) -> list[dict[str, str]]:
        """Conversation message history (read-only view)."""
        return list(self._history)

    # -- Block manipulation -------------------------------------------------

    def activate(self, block: Block) -> None:
        """Add a block to the active prompt composition."""
        self._flow._activate_block(block)

    def deactivate(self, name: str) -> None:
        """Remove a block by name from the active composition."""
        self._flow._deactivate_block(name)

    # -- Slot filling -------------------------------------------------------

    def fill_slot(self, name: str, content: str) -> None:
        """Fill a :class:`ContextSlot` block by name."""
        self._flow._fill_slot(name, content)

    # -- Phase transitions --------------------------------------------------

    def transition(self, phase_name: str) -> None:
        """Request a transition to another phase.

        The transition happens after the current handler completes.
        """
        self._pending_transition = phase_name

    # -- Prompt access ------------------------------------------------------

    def get_prompt(self) -> AssembledPrompt:
        """Assemble the current prompt (base + active blocks)."""
        return self._flow._assemble(
            turn=self._turn,
            phase=self._phase_name,
        )


# ---------------------------------------------------------------------------
# ConversationFlow
# ---------------------------------------------------------------------------


class ConversationFlow:
    """Base class for conversation flow state machines.

    Subclass this and use the ``@phase`` decorator to define phase
    handlers.  Set ``base_blocks`` for blocks that are always active.

    One instance holds the state of one conversation.

    Args:
        token_budget: Cap on the assembled prompt, in estimated tokens.
            Over budget, blocks are dropped lowest-priority first (see
            :meth:`PromptAssembler.assemble`).  Defaults to the class
            attribute, ``None`` (unlimited).
        inspector: :class:`~promptise.prompts.inspector.PromptInspector`
            that records a trace, with phase and turn, each time
            :meth:`start`, :meth:`next_turn` or :meth:`transition`
            assembles the prompt.  Defaults to the class attribute.

    Attributes:
        base_blocks: Blocks always included in the prompt (class-level).
        token_budget: Default token budget for every instance.
        inspector: Default inspector for every instance.
    """

    base_blocks: ClassVar[list[Block]] = []
    token_budget: int | None = None
    inspector: PromptInspector | None = None

    def __init__(
        self,
        *,
        token_budget: int | None = None,
        inspector: PromptInspector | None = None,
    ) -> None:
        if token_budget is not None:
            self.token_budget = token_budget
        if inspector is not None:
            self.inspector = inspector

        # Discover @phase-decorated methods
        self._phases: dict[str, _PhaseSpec] = {}
        self._initial_phase: str | None = None
        self._current_phase: str | None = None

        # Active blocks beyond base_blocks
        self._active_blocks: dict[str, Block] = {}

        # Conversation state
        self._state: dict[str, Any] = {}
        self._history: list[dict[str, str]] = []
        self._turn: int = 0

        # Slot fills carried across turns
        self._slot_fills: dict[str, str] = {}

        self._discover_phases()

    # -- Phase discovery ----------------------------------------------------

    def _discover_phases(self) -> None:
        """Scan for methods decorated with ``@phase``."""
        for attr_name in dir(self):
            try:
                method = getattr(self, attr_name)
            except AttributeError:
                continue
            spec = getattr(method, "_phase_spec", None)
            if spec is not None:
                self._phases[spec.name] = _PhaseSpec(
                    name=spec.name,
                    handler=method,
                    blocks=list(spec.blocks),
                    on_enter=spec.on_enter,
                    on_exit=spec.on_exit,
                    initial=spec.initial,
                )
                if spec.initial:
                    if self._initial_phase is not None:
                        raise ValueError(
                            f"Multiple initial phases: {self._initial_phase!r} and {spec.name!r}"
                        )
                    self._initial_phase = spec.name

    # -- Internal block management ------------------------------------------

    def _activate_block(self, block: Block) -> None:
        self._active_blocks[block.name] = block

    def _deactivate_block(self, name: str) -> None:
        self._active_blocks.pop(name, None)

    def _fill_slot(self, name: str, content: str) -> None:
        self._slot_fills[name] = content

    # -- Assembly -----------------------------------------------------------

    def _assemble(
        self,
        turn: int = 0,
        phase: str = "",
    ) -> AssembledPrompt:
        """Build the current prompt from base + active blocks."""
        all_blocks: list[Block] = list(self.base_blocks) + list(self._active_blocks.values())

        assembler = PromptAssembler(*all_blocks, token_budget=self.token_budget)

        # Apply slot fills
        for slot_name, content in self._slot_fills.items():
            assembler = assembler.fill_slot(slot_name, content)

        ctx = BlockContext(
            state=self._state,
            turn=turn,
            phase=phase,
            active_tools=[],
            metadata={},
        )
        return assembler.assemble(ctx)

    def _emit(self) -> AssembledPrompt:
        """Assemble the prompt for the current turn and record it if inspected."""
        phase = self._current_phase or ""
        assembled = self._assemble(turn=self._turn, phase=phase)
        if self.inspector is not None:
            trace = self.inspector.record_assembly(
                assembled, prompt_name=type(self).__name__, model=""
            )
            trace.flow_phase = phase
            trace.flow_turn = self._turn
        return assembled

    # -- Phase transitions --------------------------------------------------

    _MAX_TRANSITION_DEPTH = 5

    async def _enter_phase(
        self,
        phase_name: str,
        *,
        _depth: int = 0,
    ) -> None:
        """Enter a phase: run on_exit for old, on_enter for new, activate blocks.

        Recursive transitions (when an on_enter hook triggers another
        transition) are limited to ``_MAX_TRANSITION_DEPTH`` to prevent
        infinite loops.
        """
        if _depth >= self._MAX_TRANSITION_DEPTH:
            raise RecursionError(
                f"Phase transition depth limit ({self._MAX_TRANSITION_DEPTH}) "
                f"exceeded.  Check for circular transitions."
            )
        if phase_name not in self._phases:
            raise ValueError(
                f"Unknown phase {phase_name!r}. Available: {list(self._phases.keys())}"
            )

        # Exit current phase
        if self._current_phase is not None:
            old_spec = self._phases.get(self._current_phase)
            if old_spec is not None:
                # Deactivate phase-specific blocks
                for block in old_spec.blocks:
                    self._active_blocks.pop(block.name, None)
                # Run on_exit hook
                if old_spec.on_exit is not None:
                    result = old_spec.on_exit()
                    if hasattr(result, "__await__"):
                        await result

        # Enter new phase
        new_spec = self._phases[phase_name]
        self._current_phase = phase_name

        # Activate phase-specific blocks
        for block in new_spec.blocks:
            self._active_blocks[block.name] = block

        # Run on_enter hook
        if new_spec.on_enter is not None:
            result = new_spec.on_enter()
            if hasattr(result, "__await__"):
                await result

    # -- Public API ---------------------------------------------------------

    async def start(self, user_message: str | None = None) -> AssembledPrompt:
        """Initialize the flow and enter the initial phase.

        Args:
            user_message: The conversation's first user message.  It is
                recorded in the history before the initial phase handler
                runs, so the handler can act on it (turn 0).

        Returns the first assembled prompt.

        Raises:
            ValueError: If no initial phase is defined.
        """
        if user_message:
            self._history.append({"role": "user", "content": user_message})

        if self._initial_phase is None:
            if not self._phases:
                # No phases defined — just return base blocks
                return self._emit()
            raise ValueError(
                "No initial phase defined. Use @phase('name', initial=True) "
                "on one of your phase handlers."
            )

        await self._enter_phase(self._initial_phase)

        # Run the initial phase handler
        ctx = TurnContext(
            flow=self,
            turn=0,
            phase_name=self._current_phase or "",
            state=self._state,
            history=self._history,
        )
        spec = self._phases[self._initial_phase]
        result = spec.handler(ctx)
        if hasattr(result, "__await__"):
            await result

        # Handle any transition requested by the handler
        if ctx._pending_transition is not None:
            await self._enter_phase(ctx._pending_transition)
            # Run the new phase's handler
            new_spec = self._phases.get(ctx._pending_transition)
            if new_spec is not None:
                new_ctx = TurnContext(
                    flow=self,
                    turn=0,
                    phase_name=self._current_phase or "",
                    state=self._state,
                    history=self._history,
                )
                new_result = new_spec.handler(new_ctx)
                if hasattr(new_result, "__await__"):
                    await new_result

        return self._emit()

    async def next_turn(
        self,
        user_message: str,
        *,
        assistant_message: str = "",
        transition_to: str | None = None,
    ) -> AssembledPrompt:
        """Process a conversation turn.

        Records the message in history, runs the current phase handler,
        and returns the updated prompt.

        Args:
            user_message: The user's message this turn.
            assistant_message: Optional assistant reply to record.
            transition_to: Force a phase transition before processing.

        Returns:
            The assembled prompt for this turn.
        """
        self._turn += 1

        # Record messages
        self._history.append({"role": "user", "content": user_message})
        if assistant_message:
            self._history.append({"role": "assistant", "content": assistant_message})

        # Handle explicit transition
        if transition_to is not None:
            await self._enter_phase(transition_to)

        # Run current phase handler
        if self._current_phase is not None:
            spec = self._phases.get(self._current_phase)
            if spec is not None:
                ctx = TurnContext(
                    flow=self,
                    turn=self._turn,
                    phase_name=self._current_phase,
                    state=self._state,
                    history=self._history,
                )
                result = spec.handler(ctx)
                if hasattr(result, "__await__"):
                    await result

                # Handle transition requested by handler
                if ctx._pending_transition is not None:
                    await self._enter_phase(ctx._pending_transition)
                    # Run the new phase's handler
                    new_spec = self._phases.get(ctx._pending_transition)
                    if new_spec is not None:
                        new_ctx = TurnContext(
                            flow=self,
                            turn=self._turn,
                            phase_name=self._current_phase or "",
                            state=self._state,
                            history=self._history,
                        )
                        new_result = new_spec.handler(new_ctx)
                        if hasattr(new_result, "__await__"):
                            await new_result

        return self._emit()

    async def transition(self, phase_name: str) -> AssembledPrompt:
        """Explicitly transition to a new phase.

        Returns the updated prompt after the transition.
        """
        await self._enter_phase(phase_name)

        # Run the new phase handler
        spec = self._phases.get(phase_name)
        if spec is not None:
            ctx = TurnContext(
                flow=self,
                turn=self._turn,
                phase_name=phase_name,
                state=self._state,
                history=self._history,
            )
            result = spec.handler(ctx)
            if hasattr(result, "__await__"):
                await result

            if ctx._pending_transition is not None:
                await self._enter_phase(ctx._pending_transition)

        return self._emit()

    @property
    def current_phase(self) -> str | None:
        """Name of the current phase, or ``None`` before :meth:`start`."""
        return self._current_phase

    def get_prompt(self) -> AssembledPrompt:
        """Get the current prompt without advancing the turn counter."""
        return self._assemble(
            turn=self._turn,
            phase=self._current_phase or "",
        )

    def reset(self) -> None:
        """Reset the flow to its initial state."""
        self._current_phase = None
        self._active_blocks.clear()
        self._state.clear()
        self._history.clear()
        self._slot_fills.clear()
        self._turn = 0

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"phase={self._current_phase!r} "
            f"turn={self._turn} "
            f"blocks={len(self._active_blocks)}>"
        )


# ---------------------------------------------------------------------------
# Internal phase spec
# ---------------------------------------------------------------------------


@dataclass
class _PhaseSpec:
    """Internal storage for phase metadata."""

    name: str
    handler: Callable[..., Any]
    blocks: list[Block] = field(default_factory=list)
    on_enter: Callable[..., Any] | None = None
    on_exit: Callable[..., Any] | None = None
    initial: bool = False


# ---------------------------------------------------------------------------
# @phase decorator
# ---------------------------------------------------------------------------


def phase(
    name: str,
    *,
    initial: bool = False,
    blocks: list[Block] | None = None,
    on_enter: Callable[..., Any] | None = None,
    on_exit: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    """Decorator that marks a method as a phase handler.

    Usage::

        class MyFlow(ConversationFlow):
            @phase("greeting", initial=True)
            async def greet(self, ctx: TurnContext):
                ctx.activate(Section("greet", "Say hello."))

            @phase("working", blocks=[OutputFormat(format="json")])
            async def work(self, ctx: TurnContext):
                ...

    Args:
        name: Phase name (used for transitions).
        initial: Whether this is the starting phase.
        blocks: Blocks auto-activated when entering this phase.
        on_enter: Callback invoked on phase entry.
        on_exit: Callback invoked on phase exit.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn._phase_spec = _PhaseSpec(  # type: ignore[attr-defined]
            name=name,
            handler=fn,  # temporary: overwritten when the flow collects phases
            blocks=list(blocks or []),
            on_enter=on_enter,
            on_exit=on_exit,
            initial=initial,
        )
        return fn

    return decorator


# ---------------------------------------------------------------------------
# FlowSessions — one flow per conversation
# ---------------------------------------------------------------------------

FlowSource = Union[ConversationFlow, Callable[[], ConversationFlow]]
"""What ``build_agent(flow=...)`` accepts: a flow instance used as a
template, a :class:`ConversationFlow` subclass, or a zero-argument
factory returning a new flow."""


def _copy_flow(template: ConversationFlow) -> ConversationFlow:
    """Deep-copy *template* into a fresh, unstarted flow.

    The inspector is shared, not copied, so every copy records into the
    inspector the template was given.
    """
    memo: dict[int, Any] = {}
    if template.inspector is not None:
        memo[id(template.inspector)] = template.inspector
    clone = copy.deepcopy(template, memo)
    if clone._current_phase is not None or clone._history or clone._turn:
        clone.reset()
    return clone


@dataclass
class _FlowEntry:
    flow: ConversationFlow
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    started: bool = False


class FlowSessions:
    """Keeps one :class:`ConversationFlow` per conversation.

    A flow instance carries one conversation's phase, history, state and
    slot fills, so an agent must never share one between conversations.
    ``build_agent(flow=...)`` wraps the flow it is given in this class.

    * A flow **instance** is a template: each conversation gets a deep
      copy, reset if the template was already started.
    * A :class:`ConversationFlow` **subclass** or any zero-argument
      **factory** is called once per conversation.

    Conversations are identified by a hashable key (the agent uses the
    session ID and caller).  A new flow is started from the conversation's
    user messages, so the first message reaches the initial phase and a
    conversation whose flow was evicted, or that predates a restart,
    resumes in the right phase.  Without a key, :meth:`advance` builds a
    throwaway flow from the messages it is given.

    Args:
        source: A flow instance (template), subclass, or factory.
        max_sessions: Most flows kept in memory.  The least recently used
            is evicted beyond that and rebuilt from its messages when the
            conversation continues.

    Raises:
        TypeError: If *source* is not a flow or callable, or a template
            cannot be deep-copied (pass a factory instead).
    """

    def __init__(self, source: FlowSource, *, max_sessions: int = 10_000) -> None:
        if isinstance(source, ConversationFlow):
            template = source
            try:
                _copy_flow(template)
            except Exception as exc:
                raise TypeError(
                    f"Cannot copy {type(template).__name__} for each conversation: {exc}. "
                    f"Pass a factory instead, e.g. flow=lambda: {type(template).__name__}(...)."
                ) from exc
            self._factory: Callable[[], ConversationFlow] = lambda: _copy_flow(template)
        elif callable(source):
            self._factory = source
        else:
            raise TypeError(
                "flow must be a ConversationFlow instance, subclass, or a factory "
                f"returning one; got {type(source).__name__}."
            )
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        self._max_sessions = max_sessions
        self._entries: OrderedDict[Hashable, _FlowEntry] = OrderedDict()

    def new_flow(self) -> ConversationFlow:
        """Build a fresh, unstarted flow from the template or factory."""
        flow = self._factory()
        if not isinstance(flow, ConversationFlow):
            raise TypeError(
                f"The flow factory returned {type(flow).__name__}, not a ConversationFlow."
            )
        return flow

    def get(self, key: Hashable) -> ConversationFlow | None:
        """Return the flow kept for *key*, or ``None``."""
        entry = self._entries.get(key)
        return entry.flow if entry is not None else None

    def discard(self, predicate: Callable[[Hashable], bool]) -> int:
        """Drop every flow whose key matches *predicate*.  Returns the count."""
        doomed = [key for key in self._entries if predicate(key)]
        for key in doomed:
            del self._entries[key]
        return len(doomed)

    def __len__(self) -> int:
        return len(self._entries)

    async def advance(
        self, key: Hashable | None, user_messages: list[str]
    ) -> tuple[ConversationFlow, AssembledPrompt]:
        """Feed a turn to the conversation's flow and return its prompt.

        Args:
            key: The conversation.  ``None`` builds a throwaway flow.
            user_messages: The conversation's user messages so far, the
                current one last.  A new flow replays all of them; a
                running flow takes only the last.

        Returns:
            The flow and the prompt it assembled for this turn.
        """
        if key is None:
            flow = self.new_flow()
            return flow, await _replay(flow, user_messages)

        entry = self._entries.get(key)
        if entry is None:
            entry = _FlowEntry(self.new_flow())
            self._entries[key] = entry
            while len(self._entries) > self._max_sessions:
                self._entries.popitem(last=False)
        else:
            self._entries.move_to_end(key)

        async with entry.lock:
            if not entry.started:
                try:
                    prompt = await _replay(entry.flow, user_messages)
                except BaseException:
                    # Don't keep a half-started flow: the next turn rebuilds it.
                    if self._entries.get(key) is entry:
                        del self._entries[key]
                    raise
                entry.started = True
            elif user_messages:
                prompt = await entry.flow.next_turn(user_messages[-1])
            else:
                prompt = entry.flow.get_prompt()
        return entry.flow, prompt


async def _replay(flow: ConversationFlow, user_messages: list[str]) -> AssembledPrompt:
    """Start *flow* and feed it every user message in order."""
    if not user_messages:
        return await flow.start()
    prompt = await flow.start(user_messages[0])
    for message in user_messages[1:]:
        prompt = await flow.next_turn(message)
    return prompt
