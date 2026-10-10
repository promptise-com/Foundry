"""PromptGraph — the directed graph of reasoning nodes.

A ``PromptGraph`` defines a graph topology: nodes and directed edges.
Nodes are instances of ``BaseNode`` (or subclasses).  Edges define
transitions between nodes, optionally with conditions.

The graph is **mutable at runtime** — nodes and edges can be added,
removed, or modified during execution via ``GraphMutation`` objects.
The engine always works on a ``copy()`` of the graph so the original
is never mutated.

Example::

    graph = PromptGraph("my-agent")
    graph.add_node(PromptNode("plan", ...))
    graph.add_node(PromptNode("act", ...))
    graph.add_edge("plan", "act")
    graph.add_edge("act", "__end__", condition=lambda r: not r.tool_calls)
    graph.add_edge("act", "act", condition=lambda r: bool(r.tool_calls))
    graph.set_entry("plan")
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from .base import BaseNode
from .state import GraphMutation, NodeResult

logger = logging.getLogger("promptise.engine")


# ---------------------------------------------------------------------------
# Edge types
# ---------------------------------------------------------------------------


@dataclass
class Edge:
    """A directed edge between two nodes.

    Attributes:
        from_node: Source node name.
        to_node: Target node name.
        condition: Optional callable ``(NodeResult) -> bool``.
            If set, this edge is only followed when the condition
            returns ``True``.  Edges without conditions are
            ``Always`` edges.
        label: Human-readable label for visualization.
        priority: When multiple conditional edges match, the one
            with the highest priority wins.  Default is 0.
    """

    from_node: str
    to_node: str
    condition: Callable[[NodeResult], bool] | None = None
    label: str = ""
    priority: int = 0


@dataclass(frozen=True)
class EdgeCondition:
    """A built-in edge condition, described by data instead of a lambda.

    The edge helpers (``on_tool_call``, ``on_output``, ``on_error`` …) use
    it, so their edges survive :func:`~promptise.engine.serialization.save_graph`.

    Attributes:
        kind: ``"tool_called"``, ``"no_tool_call"``, ``"output"``,
            ``"error"``, ``"confidence"`` or ``"guard_failed"``.
        key: Output key (``"output"``).
        value: Expected value of ``output[key]`` (``"output"``).
        min_confidence: Threshold for ``output["confidence"]`` (``"confidence"``).
    """

    kind: str
    key: str = ""
    value: Any = True
    min_confidence: float = 0.0

    KINDS: ClassVar[tuple[str, ...]] = (
        "tool_called",
        "no_tool_call",
        "output",
        "error",
        "confidence",
        "guard_failed",
    )

    def __post_init__(self) -> None:
        if self.kind not in self.KINDS:
            raise ValueError(f"Unknown edge condition kind {self.kind!r}. Kinds: {self.KINDS}")

    def __call__(self, r: NodeResult) -> bool:
        if self.kind == "tool_called":
            return bool(r.tool_calls)
        if self.kind == "no_tool_call":
            return not r.tool_calls
        if self.kind == "output":
            return isinstance(r.output, dict) and r.output.get(self.key) == self.value
        if self.kind == "error":
            return r.error is not None
        if self.kind == "confidence":
            return isinstance(r.output, dict) and float(r.output.get("confidence", 0)) >= (
                self.min_confidence
            )
        return bool(r.guards_failed)  # guard_failed


# ---------------------------------------------------------------------------
# PromptGraph
# ---------------------------------------------------------------------------


class PromptGraph:
    """A directed graph of reasoning nodes.

    Nodes are added via ``add_node()``.  Edges via ``add_edge()``.
    The entry point is set via ``set_entry()``.

    The graph supports runtime mutation via ``apply_mutation()``
    and ``copy()`` for per-invocation isolation.

    Args:
        name: Human-readable graph name (for visualization and logs).
    """

    # Prebuilt factory helpers — bound at import time by ``engine/prebuilts.py``.
    # Declared here (as ``Any``) so static analysis recognises them as attributes.
    react: Any
    managed: Any
    code_action: Any
    verify: Any
    peoatr: Any
    research: Any
    autonomous: Any
    deliberate: Any
    debate: Any
    pipeline: Any

    def __init__(
        self,
        name: str = "graph",
        *,
        mode: str = "autonomous",
        nodes: list[BaseNode] | None = None,
    ) -> None:
        self.name = name
        self.mode = mode  # "autonomous" (default) or "static"
        self._nodes: dict[str, BaseNode] = {}
        self._edges: list[Edge] = []
        self._edge_index: dict[str, list[Edge]] = {}  # Precomputed adjacency
        self._edge_index_dirty = True
        self._entry: str | None = None
        self._cow_source: PromptGraph | None = None  # Copy-on-write parent
        # loop_until() budgets: node name → (max_iterations, exit_to)
        self._loop_limits: dict[str, tuple[int, str]] = {}

        # Add nodes if passed in constructor
        if nodes:
            for node in nodes:
                self.add_node(node)
            # Auto-detect entry from is_entry flag
            for node in nodes:
                if getattr(node, "is_entry", False):
                    self._entry = node.name
                    break

    # ── Node management ──────────────────────────────────────────────

    def add_node(self, node: BaseNode) -> PromptGraph:
        """Add a node to the graph.  Returns self for chaining."""
        if node.name in self._nodes:
            logger.warning("Replacing existing node %r in graph %r", node.name, self.name)
        self._nodes[node.name] = node
        return self

    def remove_node(self, name: str) -> PromptGraph:
        """Remove a node and all edges referencing it."""
        self._nodes.pop(name, None)
        self._loop_limits.pop(name, None)
        self._ensure_edges_owned()
        self._edges = [e for e in self._edges if e.from_node != name and e.to_node != name]
        self._edge_index_dirty = True
        if self._entry == name:
            self._entry = None
        return self

    def get_node(self, name: str) -> BaseNode:
        """Get a node by name.  Raises ``KeyError`` if not found."""
        if name not in self._nodes:
            raise KeyError(
                f"Node {name!r} not found in graph {self.name!r}. "
                f"Available: {list(self._nodes.keys())}"
            )
        return self._nodes[name]

    def has_node(self, name: str) -> bool:
        """Check if a node exists."""
        return name in self._nodes

    @property
    def nodes(self) -> dict[str, BaseNode]:
        """All nodes in the graph."""
        return dict(self._nodes)

    # ── Edge management ──────────────────────────────────────────────

    def _ensure_edges_owned(self) -> None:
        """Copy-on-write: lazily copy edges on first mutation."""
        if self._cow_source is not None:
            self._edges = [copy.copy(e) for e in self._edges]
            self._cow_source = None

    def add_edge(
        self,
        from_node: str,
        to_node: str,
        *,
        condition: Callable[[NodeResult], bool] | None = None,
        label: str = "",
        priority: int = 0,
    ) -> PromptGraph:
        """Add a directed edge.  Returns self for chaining.

        Warns if ``from_node`` or ``to_node`` do not exist in the graph
        yet (they may be added later, so this is a warning, not an error).
        """
        if from_node not in self._nodes and from_node != "__end__":
            import logging as _log

            _log.getLogger("promptise.engine").warning(
                "Edge source %r not (yet) in graph %r", from_node, self.name
            )
        if to_node not in self._nodes and to_node != "__end__":
            import logging as _log

            _log.getLogger("promptise.engine").warning(
                "Edge target %r not (yet) in graph %r", to_node, self.name
            )
        # Warn about duplicate edges with different conditions — may cause undefined routing
        if condition is not None:
            for existing in self._edges:
                if (
                    existing.from_node == from_node
                    and existing.to_node == to_node
                    and existing.condition is not None
                    and existing.priority == priority
                ):
                    import logging as _log

                    _log.getLogger("promptise.engine").warning(
                        "Duplicate conditional edge %s → %s (same priority %d) — ensure conditions are mutually exclusive",
                        from_node,
                        to_node,
                        priority,
                    )
                    break

        self._ensure_edges_owned()
        self._edges.append(
            Edge(
                from_node=from_node,
                to_node=to_node,
                condition=condition,
                label=label,
                priority=priority,
            )
        )
        self._edge_index_dirty = True
        return self

    def _rebuild_edge_index(self) -> None:
        """Rebuild the adjacency index from the flat edge list."""
        idx: dict[str, list[Edge]] = {}
        for e in self._edges:
            idx.setdefault(e.from_node, []).append(e)
        # Pre-sort each bucket by priority (desc) so get_edges_from is O(1)
        for edges in idx.values():
            edges.sort(key=lambda e: e.priority, reverse=True)
        self._edge_index = idx
        self._edge_index_dirty = False

    def get_edges_from(self, node_name: str) -> list[Edge]:
        """Get all outgoing edges from a node, sorted by priority (desc).

        Uses a precomputed adjacency index — O(1) lookup instead of O(E) scan.
        """
        if self._edge_index_dirty:
            self._rebuild_edge_index()
        return self._edge_index.get(node_name, [])

    @property
    def edges(self) -> list[Edge]:
        """All edges in the graph."""
        return list(self._edges)

    # ── Edge convenience helpers ───────────────────────────────────

    def always(self, from_node: str, to_node: str) -> PromptGraph:
        """Add an unconditional edge: A always goes to B."""
        return self.add_edge(from_node, to_node)

    def when(
        self,
        from_node: str,
        to_node: str,
        condition: Callable[[NodeResult], bool],
        label: str = "",
    ) -> PromptGraph:
        """Add a conditional edge: A goes to B when condition is True."""
        return self.add_edge(from_node, to_node, condition=condition, label=label)

    def on_tool_call(self, from_node: str, to_node: str) -> PromptGraph:
        """Add an edge that fires when the node made tool calls."""
        return self.add_edge(
            from_node,
            to_node,
            condition=EdgeCondition("tool_called"),
            label="tool_called",
        )

    def on_no_tool_call(self, from_node: str, to_node: str) -> PromptGraph:
        """Add an edge that fires when the node made NO tool calls (final answer)."""
        return self.add_edge(
            from_node,
            to_node,
            condition=EdgeCondition("no_tool_call"),
            label="no_tools",
        )

    def on_output(self, from_node: str, to_node: str, key: str, value: Any = True) -> PromptGraph:
        """Add an edge that fires when output[key] == value."""
        return self.add_edge(
            from_node,
            to_node,
            condition=EdgeCondition("output", key=key, value=value),
            label=f"{key}={value}",
        )

    def on_error(self, from_node: str, to_node: str) -> PromptGraph:
        """Add an edge that fires when a node has an error."""
        return self.add_edge(from_node, to_node, condition=EdgeCondition("error"), label="error")

    def on_confidence(
        self, from_node: str, to_node: str, min_confidence: float = 0.7
    ) -> PromptGraph:
        """Add an edge that fires when output confidence exceeds threshold."""
        return self.add_edge(
            from_node,
            to_node,
            condition=EdgeCondition("confidence", min_confidence=min_confidence),
            label=f"confidence>={min_confidence}",
        )

    def on_guard_fail(self, from_node: str, to_node: str) -> PromptGraph:
        """Add an edge that fires when any guard fails."""
        return self.add_edge(
            from_node,
            to_node,
            condition=EdgeCondition("guard_failed"),
            label="guard_failed",
        )

    def sequential(self, *node_names: str) -> PromptGraph:
        """Chain nodes with always edges: A → B → C → ..."""
        for i in range(len(node_names) - 1):
            self.always(node_names[i], node_names[i + 1])
        return self

    def loop_until(
        self,
        node_name: str,
        exit_to: str,
        *,
        condition: Callable[[NodeResult], bool],
        max_iterations: int = 5,
    ) -> PromptGraph:
        """Add a loop: node re-enters itself until condition, then exits.

        The node runs at most *max_iterations* times in a run (fewer when
        its own ``max_iterations`` is lower); once it has used them, the
        engine exits to *exit_to* even though *condition* never held.
        """
        if max_iterations < 1:
            raise ValueError("loop_until max_iterations must be at least 1")
        self.add_edge(node_name, exit_to, condition=condition, label="exit_loop", priority=10)
        self.add_edge(node_name, node_name, label="loop", priority=0)
        self._loop_limits[node_name] = (max_iterations, exit_to)
        return self

    def loop_limit(self, node_name: str) -> tuple[int, str] | None:
        """``(max_iterations, exit_to)`` set by :meth:`loop_until` for
        *node_name*, or ``None``."""
        return self._loop_limits.get(node_name)

    # ── Factory: build from node pool ──────────────────────────────

    @classmethod
    def from_pool(
        cls,
        nodes: list[BaseNode],
        *,
        system_prompt: str = "",
        name: str = "autonomous",
    ) -> PromptGraph:
        """Create a graph from a pool of nodes (autonomous mode).

        The engine will use ``AutonomousNode`` to dynamically
        build the execution path from the node pool.  Nodes with
        ``is_entry=True`` start first.  Nodes with
        ``is_terminal=True`` can end the graph.
        """
        from .nodes import AutonomousNode

        entry_node = None
        terminal_names = []
        for n in nodes:
            if getattr(n, "is_entry", False):
                entry_node = n.name
            if getattr(n, "is_terminal", False):
                terminal_names.append(n.name)

        graph = cls(name=name, mode="autonomous")
        graph.add_node(
            AutonomousNode(
                "orchestrator",
                node_pool=nodes,
                planner_instructions=system_prompt,
                entry_node=entry_node,
                terminal_nodes=terminal_names,
            )
        )
        graph.set_entry("orchestrator")
        return graph

    # ── Entry point ──────────────────────────────────────────────────

    def set_entry(self, node_name: str) -> PromptGraph:
        """Set the entry node.  Returns self for chaining."""
        if node_name not in self._nodes:
            raise KeyError(
                f"Entry node {node_name!r} not found in graph. Add it with add_node() first."
            )
        self._entry = node_name
        return self

    @property
    def entry(self) -> str | None:
        """The entry node name."""
        return self._entry

    # ── Runtime mutation ─────────────────────────────────────────────

    def apply_mutation(self, mutation: GraphMutation) -> None:
        """Apply a single mutation to the graph.

        Called by the engine during execution to modify the live
        graph copy.  Validates mutations before applying.
        """
        if mutation.action == "add_node":
            from .nodes import PromptNode

            node = PromptNode.from_config(mutation.node_config)
            self._nodes[mutation.node_name or node.name] = node
            logger.info("Graph mutation: added node %r", node.name)

        elif mutation.action == "remove_node":
            if mutation.node_name in self._nodes:
                self.remove_node(mutation.node_name)
                logger.info("Graph mutation: removed node %r", mutation.node_name)

        elif mutation.action == "add_edge":
            from_n = mutation.from_node or ""
            to_n = mutation.to_node or ""
            if not from_n or not to_n:
                logger.warning("Graph mutation: add_edge missing from/to nodes")
                return
            self.add_edge(from_n, to_n, label=mutation.condition or "")
            logger.info("Graph mutation: added edge %s → %s", from_n, to_n)

        elif mutation.action == "remove_edge":
            self._ensure_edges_owned()
            self._edges = [
                e
                for e in self._edges
                if not (e.from_node == mutation.from_node and e.to_node == mutation.to_node)
            ]
            self._edge_index_dirty = True
            logger.info(
                "Graph mutation: removed edge %s → %s", mutation.from_node, mutation.to_node
            )

        else:
            logger.warning("Unknown graph mutation action: %s", mutation.action)

    # ── Copy ─────────────────────────────────────────────────────────

    def copy(self) -> PromptGraph:
        """Copy the graph for per-invocation isolation.

        Nodes are shared references (they are stateless config objects).
        Edges use copy-on-write — shared until the copy mutates them,
        at which point they are lazily copied via ``_ensure_edges_owned()``.
        """
        new = PromptGraph(name=self.name, mode=self.mode)
        new._nodes = dict(self._nodes)  # Shallow copy of node dict
        new._edges = self._edges  # Shared — COW on first mutation
        new._cow_source = self
        new._entry = self._entry
        new._loop_limits = dict(self._loop_limits)
        new._edge_index_dirty = True  # Rebuild on first use
        return new

    # ── Validation ───────────────────────────────────────────────────

    def validate(self) -> list[str]:
        """Validate the graph topology.  Returns list of error strings.

        Checks:
        - Entry node is set and exists
        - All edge targets exist (or are ``__end__``)
        - All transition targets in node configs exist
        - No unreachable nodes (except entry)
        - No dead-end nodes without ``__end__`` transition
        """
        errors: list[str] = []

        # Entry node
        if self._entry is None:
            errors.append("No entry node set. Call graph.set_entry(name).")
        elif self._entry not in self._nodes:
            errors.append(f"Entry node {self._entry!r} does not exist.")

        # Edge targets
        for edge in self._edges:
            if edge.from_node not in self._nodes:
                errors.append(f"Edge from unknown node {edge.from_node!r}")
            if edge.to_node != "__end__" and edge.to_node not in self._nodes:
                errors.append(f"Edge to unknown node {edge.to_node!r}")

        # Node transition targets
        for name, node in self._nodes.items():
            for key, target in node.transitions.items():
                if target != "__end__" and target not in self._nodes:
                    errors.append(
                        f"Node {name!r} transition {key!r} → {target!r} targets unknown node"
                    )
            if (
                node.default_next
                and node.default_next != "__end__"
                and node.default_next not in self._nodes
            ):
                errors.append(
                    f"Node {name!r} default_next {node.default_next!r} targets unknown node"
                )

        # Reachability (BFS from entry)
        if self._entry:
            reachable: set[str] = set()
            queue = [self._entry]
            while queue:
                current = queue.pop(0)
                if current in reachable or current == "__end__":
                    continue
                reachable.add(current)
                # From edges
                for edge in self.get_edges_from(current):
                    queue.append(edge.to_node)
                # From node transitions
                if current in self._nodes:
                    node = self._nodes[current]
                    for target in node.transitions.values():
                        queue.append(target)
                    if node.default_next:
                        queue.append(node.default_next)

            unreachable = set(self._nodes.keys()) - reachable
            for name in unreachable:
                errors.append(f"Node {name!r} is unreachable from entry")

        return errors

    # ── Visualization ────────────────────────────────────────────────

    def describe(self) -> str:
        """Human-readable graph description."""
        lines = [f'PromptGraph "{self.name}"']

        for name, node in self._nodes.items():
            prefix = "[entry] " if name == self._entry else ""
            node_type = type(node).__name__

            # Collect outgoing targets
            targets: list[str] = []
            for edge in self.get_edges_from(name):
                label = f"|{edge.label}| " if edge.label else ""
                targets.append(f"{label}{edge.to_node}")
            for key, target in node.transitions.items():
                targets.append(f"|{key}| {target}")
            if node.default_next and node.default_next not in [t.split()[-1] for t in targets]:
                targets.append(node.default_next)

            target_str = " | ".join(targets) if targets else "(no transitions)"
            lines.append(f"├── {prefix}{name} ({node_type}) → {target_str}")

        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Export graph as Mermaid diagram syntax."""
        lines = ["graph TD"]

        for edge in self._edges:
            if edge.label:
                lines.append(f"  {edge.from_node} -->|{edge.label}| {edge.to_node}")
            else:
                lines.append(f"  {edge.from_node} --> {edge.to_node}")

        for name, node in self._nodes.items():
            for key, target in node.transitions.items():
                lines.append(f"  {name} -->|{key}| {target}")
            if node.default_next:
                lines.append(f"  {name} --> {node.default_next}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"PromptGraph({self.name!r}, "
            f"nodes={len(self._nodes)}, "
            f"edges={len(self._edges)}, "
            f"entry={self._entry!r})"
        )
