# Edges & Transitions

Edges define how the graph flows from one node to another. The engine resolves transitions dynamically at runtime using a 7-step priority system.

## Edge Dataclass

```python
from promptise.engine import Edge, NodeResult


def my_fn(result: NodeResult) -> bool:
    return result.error is None


Edge(
    from_node="plan",       # Source node name
    to_node="act",          # Target node name
    condition=my_fn,        # Optional: (NodeResult) -> bool
    label="ready",          # Optional: label for visualization
    priority=0,             # Optional: higher = checked first
)
```

## Edge Helpers

All helpers return `self` for chaining: `graph.always("a", "b").always("b", "c")`.

### always — Unconditional

```python
graph.always("plan", "execute")   # plan always goes to execute
```

### when — Conditional

```python
graph.when("review", "fix",
    condition=lambda result: result.output.get("quality", 0) < 3,
    label="low_quality",
)
```

### on_tool_call / on_no_tool_call — Tool-based routing

```python
graph.on_tool_call("agent", "agent")       # Loop back when tools called
graph.on_no_tool_call("agent", "__end__")   # End when no tools (final answer)
```

### on_output — Output key matching

```python
graph.on_output("review", "publish", key="approved", value=True)
graph.on_output("review", "revise", key="approved", value=False)
```

### on_error — Error routing

```python
graph.on_error("risky_step", "fallback")   # Route to fallback if node errors
```

### on_confidence — Confidence threshold

```python
graph.on_confidence("analyze", "conclude", min_confidence=0.7)
# Routes to "conclude" when output["confidence"] >= 0.7
```

### on_guard_fail — Guard failure routing

```python
graph.on_guard_fail("validate", "revise")
# Routes to "revise" when any guard fails
```

### sequential — Chain multiple nodes

```python
graph.sequential("step1", "step2", "step3", "step4")
# Creates: step1 → step2 → step3 → step4 (always edges)
```

### loop_until — Conditional loop with exit

```python
graph.loop_until("refine", "deliver",
    condition=lambda result: result.output.get("quality", 0) >= 4,
    max_iterations=5,
)
# Loops "refine" until quality >= 4, then exits to "deliver"
# Exit edge gets priority=10, loop edge gets priority=0
```

`refine` runs at most `max_iterations` times in a run (fewer if its own `max_iterations` is lower). Once it has used them, the engine exits to `deliver` even though the condition never held.

### Low-level add_edge

```python
graph.add_edge("a", "b", condition=my_fn, label="custom", priority=10)
```

!!! note "Performance"
    The engine maintains a precomputed adjacency index for **O(1) edge lookup** per transition (instead of scanning all edges). The index is rebuilt lazily when edges are added or removed.

## Transition Resolution

When a node finishes, the engine resolves the next node in this order:

```mermaid
graph TD
    NE[Node Executed] --> TC{Tool calls?}
    TC -->|yes| SAME[Re-enter same node]
    TC -->|no| NR{NodeResult<br/>.next_node?}
    NR -->|yes| TARGET2[Go to next_node]
    NR -->|no| LLM{output.route<br/>set?}
    LLM -->|yes| TARGET[Follow that transition<br/>or go to that node]
    LLM -->|no| EDGE{Conditional<br/>edges?}
    EDGE -->|match| TARGET3[Follow edge]
    EDGE -->|no match| TR{Node transitions<br/>match output?}
    TR -->|yes| TARGET4[Follow transition]
    TR -->|no| DEF{default_next?}
    DEF -->|yes| TARGET5[Go to default]
    DEF -->|no| END[__end__]

    style NE fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style SAME fill:#1a2e1a,stroke:#4ade80,color:#fff
    style END fill:#1a1a1a,stroke:#666,color:#aaa
```

1. **Tool loop** — If tools were called, re-enter the same node so the LLM sees tool results
2. **NodeResult.next_node** — If the node explicitly set the next node in its execute() method (`ValidateNode`'s pass/fail, `PlanNode`'s re-plan)
3. **LLM routing** — If output contains `route`, `_next`, `next_step`, or `goto` naming one of the node's transition keys (follows that transition) or a node in the graph
4. **Graph edges** — Conditional edges checked in priority order (highest first)
5. **Node transitions** — Output keys matched against the node's `transitions` dict
6. **default_next** — Fallback node from the node's configuration
7. **__end__** — Graph terminates

## Edge Priority

When multiple conditional edges exist from the same node, they are checked in **priority order** (highest first). Use this to ensure exit conditions are checked before loop conditions:

```python
# Exit condition checked first (priority 10)
graph.add_edge("refine", "deliver",
    condition=lambda r: r.output.get("done"),
    label="done", priority=10)

# Loop condition checked second (priority 0)
graph.always("refine", "refine")  # default priority=0
```

## Dynamic LLM Routing

The LLM can choose the next node by including a routing field in its output. Routing reads a structured output, so give the node an `output_schema` (a Pydantic model or a `TypedDict`) with a `route` field:

```python
from typing import Literal

from pydantic import BaseModel


class Decision(BaseModel):
    route: Literal["search", "answer"]
    reason: str


graph.add_node(PromptNode(
    "decide",
    output_schema=Decision,
    transitions={"search": "web_search", "answer": "write"},
))
# The LLM outputs {"route": "search", "reason": "Need more data"}
# → the engine follows the "search" transition to "web_search"

# Supported field names: route, _next, next_step, goto
```

A route value that is a transition key follows that transition; otherwise it must name a node in the graph (or `__end__`). When a node without tools has two or more transitions, the engine lists them in the prompt. The `"error"` transition is reserved: the engine follows it when the node has used its `max_iterations`, and never offers it to the model.

## Runtime Graph Mutation

The LLM can modify the graph during execution via structured output:

| Action | Output Fields | Description |
|--------|--------------|-------------|
| `add_node` | `_graph_action: "add_node"`, `_node_config: {...}` | Add a new node to the live graph |
| `skip_to` | `_graph_action: "skip_to"`, `_target: "node_name"` | Jump directly to another node |
| `add_edge` | `_graph_action: "add_edge"`, `_from: "a"`, `_to: "b"` | Add a new edge between existing nodes |
| `retry_with` | `_graph_action: "retry_with"`, `_context_update: {...}` | Update context and retry current node |

Mutations are applied to a per-invocation copy of the graph — the original is never modified. The engine caps mutations at `max_mutations_per_run` (default: 10).
