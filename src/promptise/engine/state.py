"""Graph state, node results, and event types for the PromptGraph engine.

This module defines the data structures that flow through the graph
during execution. ``GraphState`` is the primary state object that
carries messages, context, observations, plan, reflections, and
timing data between nodes. ``NodeResult`` captures everything that
happened during a single node execution for observability.

All dataclasses use ``field(default_factory=...)`` for mutable defaults
to avoid shared-state bugs across invocations.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("promptise.engine")

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Node Flags — typed enum for node capabilities and roles
# ---------------------------------------------------------------------------


class NodeFlag(str, Enum):
    """Typed flags that declare a node's role and capabilities.

    Use these instead of bare strings for type safety and IDE support::

        PlanNode("plan", flags={NodeFlag.ENTRY, NodeFlag.INJECT_TOOLS})
    """

    ENTRY = "entry"
    """This node starts the reasoning graph."""

    TERMINAL = "terminal"
    """Reaching this node can end the graph."""

    INJECT_TOOLS = "inject_tools"
    """Receives MCP tools at runtime from build_agent()."""

    READONLY = "readonly"
    """This node only reads state, never writes. Safe for parallel execution."""

    REQUIRES_HUMAN = "requires_human"
    """This node pauses for human input before proceeding."""

    CACHEABLE = "cacheable"
    """This node's output can be cached for identical inputs."""

    RETRYABLE = "retryable"
    """This node can be retried on failure."""

    CRITICAL = "critical"
    """This node must succeed — graph aborts on failure."""

    OBSERVABLE = "observable"
    """Emit extra observability events for this node."""

    STATEFUL = "stateful"
    """This node modifies state.context (for dependency tracking)."""

    PARALLEL_SAFE = "parallel_safe"
    """Safe to run concurrently with other parallel_safe nodes."""

    SKIP_ON_ERROR = "skip_on_error"
    """Skip this node if a previous node errored (don't abort)."""

    VERBOSE = "verbose"
    """Include full prompt and response in observability logs."""

    LIGHTWEIGHT = "lightweight"
    """Use a smaller/faster model for this node (if model_override not set)."""

    NO_HISTORY = "no_history"
    """Don't inject conversation history into this node's context."""

    ISOLATED_CONTEXT = "isolated_context"
    """Don't inherit context from previous nodes — start fresh."""

    SUMMARIZE_OUTPUT = "summarize_output"
    """Auto-summarize long outputs before passing to next node."""

    VALIDATE_OUTPUT = "validate_output"
    """Auto-validate output against output_schema before proceeding."""


# ---------------------------------------------------------------------------
# Graph Mutation — requested by nodes or the LLM at runtime
# ---------------------------------------------------------------------------


@dataclass
class GraphMutation:
    """A single graph modification requested during execution.

    Mutations are applied to the live graph copy (not the original).
    The engine validates mutations before applying them.

    Attributes:
        action: One of ``"add_node"``, ``"remove_node"``,
            ``"add_edge"``, ``"remove_edge"``.
        node_name: Target node name (for add/remove node).
        node_config: Configuration dict for new nodes (used with
            ``add_node``).  Passed to ``node_from_config()``.
        from_node: Source node for edge operations.
        to_node: Target node for edge operations.
        condition: Optional condition string for conditional edges.
    """

    action: str
    node_name: str = ""
    node_config: dict[str, Any] = field(default_factory=dict)
    from_node: str = ""
    to_node: str = ""
    condition: str = ""


# ---------------------------------------------------------------------------
# Node Result — observability unit per node execution
# ---------------------------------------------------------------------------


@dataclass
class NodeResult:
    """Everything that happened during a single node execution.

    This is the richest observability unit in the engine.  Every field
    is populated by the node's ``execute()`` method and the engine's
    post-processing.  Stored in ``GraphState.node_history`` for full
    traceability.
    """

    # Identity
    node_name: str = ""
    node_type: str = ""
    iteration: int = 0

    # Timing
    duration_ms: float = 0.0
    llm_duration_ms: float = 0.0
    tool_duration_ms: float = 0.0

    # Tokens
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    # Tool activity
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_deduplicated: int = 0
    tool_calls_failed: int = 0

    # LLM output
    output: Any = None
    raw_output: str = ""
    next_node: str | None = None
    transition_reason: str = ""

    # Messages appended to state
    messages_added: list[Any] = field(default_factory=list)

    # Prompt assembly trace (from PromptBlocks)
    blocks_used: list[str] = field(default_factory=list)
    blocks_dropped: list[str] = field(default_factory=list)
    strategy_applied: str | None = None
    perspective_applied: str | None = None

    # Guards
    guards_passed: list[str] = field(default_factory=list)
    guards_failed: list[str] = field(default_factory=list)
    guard_retries: int = 0

    # Graph mutations requested by this node
    graph_mutations: list[GraphMutation] = field(default_factory=list)

    # Error
    error: str | None = None
    error_recovered: bool = False


# ---------------------------------------------------------------------------
# Node Event — emitted during streaming
# ---------------------------------------------------------------------------


@dataclass
class NodeEvent:
    """An event emitted during streaming execution.

    These are engine-level events (``on_node_start``, ``on_node_end``,
    ``on_graph_mutation``) that complement the standard LangChain events
    (``on_tool_start``, ``on_chat_model_stream``, etc.).
    """

    event: str
    node_name: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Graph State — flows through the graph during execution
# ---------------------------------------------------------------------------


@dataclass
class GraphState:
    """The complete state of a graph execution.

    Passed to every node's ``execute()`` method.  Nodes read from
    and write to this state.  The engine updates it after each node
    execution.

    The ``graph`` field carries the live (mutable) copy of the
    ``PromptGraph`` so nodes can inspect and modify the graph
    topology at runtime.

    Attributes:
        messages: Full LangChain message history.
        context: Key-value state that persists across nodes.
        current_node: Name of the node currently being executed.
        visited: Ordered list of node names visited (for cycle
            detection).
        iteration: Global iteration counter (incremented each
            time any node executes).
        node_iterations: Per-node execution counter (for detecting
            stuck nodes).
        graph: The live ``PromptGraph`` copy (mutable during
            execution).  ``None`` if not yet set by the engine.
        plan: Current subgoals (set by planning nodes).
        completed: Completed subgoals.
        observations: Tool results accumulated during execution.
        reflections: Past learnings/mistakes from reflection nodes.
        tool_calls_made: Total tool calls across all nodes.
        node_timings: Cumulative milliseconds per node name.
        total_tokens: Cumulative token usage across all nodes.
        node_history: Ordered list of ``NodeResult`` objects for
            full execution trace.
        turn_question: The current user question: the last
            ``HumanMessage`` of the input.  Context compaction always
            sends it.
        turn_input: The input up to and including ``turn_question``
            (system messages and chat history).

    Dict messages (``{"role": "user", "content": ...}``) in ``messages``
    are converted to LangChain message objects when the state is created.
    """

    messages: list[Any] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)
    current_node: str = ""
    visited: list[str] = field(default_factory=list)
    iteration: int = 0
    node_iterations: dict[str, int] = field(default_factory=dict)
    max_messages: int = 200

    # The live graph (mutable during execution)
    graph: Any = None  # PromptGraph — forward ref to avoid circular import

    # Agentic state
    plan: list[str] = field(default_factory=list)
    completed: list[str] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    reflections: list[dict[str, Any]] = field(default_factory=list)
    tool_calls_made: int = 0

    # Timing & observability
    node_timings: dict[str, float] = field(default_factory=dict)
    total_tokens: int = 0
    node_history: list[NodeResult] = field(default_factory=list)

    # The current turn, captured from the input (see ``__post_init__``)
    turn_question: Any = field(default=None, repr=False)
    turn_input: list[Any] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        from .compaction import normalize_messages, split_input

        self.messages = normalize_messages(self.messages)
        if self.turn_question is None and not self.turn_input:
            self.turn_question, self.turn_input = split_input(self.messages)

    # --- helpers ---

    def record_node_timing(self, node_name: str, ms: float) -> None:
        """Add *ms* to the cumulative timing for *node_name*."""
        self.node_timings[node_name] = self.node_timings.get(node_name, 0.0) + ms

    def increment_node_iteration(self, node_name: str) -> int:
        """Increment and return the per-node iteration count."""
        count = self.node_iterations.get(node_name, 0) + 1
        self.node_iterations[node_name] = count
        return count

    def add_observation(
        self,
        tool_name: str,
        result: str,
        args: dict[str, Any] | None = None,
        success: bool = True,
        duration_ms: float = 0.0,
    ) -> None:
        """Record a tool observation.

        Observations are capped at 50 entries. When the cap is reached,
        the oldest entries are discarded.
        """
        self.observations.append(
            {
                "tool": tool_name,
                "result": result,
                "args": args or {},
                "success": success,
                "duration_ms": duration_ms,
            }
        )
        self.tool_calls_made += 1
        # Cap observations to prevent unbounded memory growth
        if len(self.observations) > 50:
            self.observations = self.observations[-50:]

    def add_reflection(
        self,
        iteration: int,
        mistake: str,
        correction: str,
        confidence: float = 0.5,
        stage: str = "",
    ) -> None:
        """Record a reflection from a reflect/evaluate node."""
        self.reflections.append(
            {
                "iteration": iteration,
                "mistake": mistake,
                "correction": correction,
                "confidence": confidence,
                "stage": stage,
            }
        )
        # Keep only last 5 reflections to bound context size
        if len(self.reflections) > 5:
            self.reflections = self.reflections[-5:]

    def complete_subgoal(self, subgoal: str) -> None:
        """Mark a subgoal as completed."""
        if not subgoal:
            return
        if subgoal not in self.plan:
            logger.debug(
                "complete_subgoal(%r) called but subgoal not in plan %s", subgoal, self.plan
            )
        if subgoal not in self.completed:
            self.completed.append(subgoal)

    @property
    def active_subgoal(self) -> str | None:
        """Return the first uncompleted subgoal, or ``None``."""
        for sg in self.plan:
            if sg not in self.completed:
                return sg
        return None

    @property
    def all_subgoals_complete(self) -> bool:
        """Return ``True`` if every planned subgoal is completed."""
        return bool(self.plan) and all(sg in self.completed for sg in self.plan)

    def trim_messages(self) -> None:
        """Trim message history to ``max_messages`` if exceeded.

        Keeps all system messages (essential context), the current user
        question, plus the most recent non-system messages to stay within
        the configured cap.
        """
        if self.max_messages <= 0 or len(self.messages) <= self.max_messages:
            return
        system_msgs = [
            m
            for m in self.messages
            if getattr(m, "type", None) == "system" or type(m).__name__ == "SystemMessage"
        ]
        non_system = [m for m in self.messages if m not in system_msgs]
        keep = self.max_messages - len(system_msgs)
        if keep < 1:
            keep = 1
        question = self.turn_question
        pin = question is not None and any(m is question for m in non_system[:-keep])
        recent = non_system[-(keep - 1) :] if pin and keep > 1 else non_system[-keep:]
        # A tool result whose assistant tool call was cut is rejected by
        # providers; start the window after any such orphans.
        while recent and getattr(recent[0], "type", None) == "tool":
            recent.pop(0)
        self.messages = system_msgs + ([question] if pin else []) + recent


# ---------------------------------------------------------------------------
# Execution Report — summary after graph completion
# ---------------------------------------------------------------------------


@dataclass
class ExecutionReport:
    """Summary of a complete graph execution.

    Produced by ``PromptGraphEngine`` after ``ainvoke()`` completes.
    """

    total_iterations: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    nodes_visited: list[str] = field(default_factory=list)
    tool_calls: int = 0
    graph_mutations: int = 0
    guards_passed: int = 0
    guards_failed: int = 0
    error: str | None = None

    def summary(self) -> str:
        """Human-readable execution summary."""
        path = " → ".join(self.nodes_visited) if self.nodes_visited else "(none)"
        lines = [
            "Execution Report",
            "════════════════",
            f"Total iterations: {self.total_iterations}",
            f"Total tokens: {self.total_tokens:,}",
            f"Total duration: {self.total_duration_ms / 1000:.1f}s",
            f"Nodes visited: {path}",
            f"Tool calls: {self.tool_calls}",
            f"Graph mutations: {self.graph_mutations}",
            f"Guards: {self.guards_passed} passed, {self.guards_failed} failed",
        ]
        if self.error:
            lines.append(f"Error: {self.error}")
        return "\n".join(lines)
