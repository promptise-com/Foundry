# Node Flags

Every node in a Reasoning Graph declares its capabilities and behaviors through **typed flags**. Flags are checked by the engine at runtime to control execution flow, caching, error handling, and observability.

## Using Flags

```python
from promptise.engine import PromptNode, NodeFlag

# Set flags on any node
PromptNode("analyze",
    instructions="Analyze the data.",
    flags={NodeFlag.CRITICAL, NodeFlag.OBSERVABLE},
)

# Functional nodes support flags too
@node("validate", flags={NodeFlag.VALIDATE_OUTPUT, NodeFlag.RETRYABLE})
async def validate(state: GraphState) -> NodeResult:
    ...

# Check flags at runtime
if node.has_flag(NodeFlag.CRITICAL):
    ...
```

## Flag Reference

### Execution Control

| Flag | Engine Behavior |
|------|----------------|
| `ENTRY` | Marks the graph's starting node. Set via `is_entry=True`. |
| `TERMINAL` | Reaching this node can end the graph. Set via `is_terminal=True`. |
| `CRITICAL` | If this node errors, the engine **aborts the entire graph** immediately. The error is captured in `ExecutionReport.error` and `ainvoke()` raises `GraphExecutionError` (the node's exception is its `__cause__`). |
| `SKIP_ON_ERROR` | If the **previous** node produced an error, this node is skipped entirely. Useful for optional enrichment steps. |
| `RETRYABLE` | On failure, the engine retries this node with exponential backoff (0.5s, 1s, 2s...) up to `max_iterations` times. Between retries, `state.context["_retry_error"]` contains the last error so the node can adapt. |
| `REQUIRES_HUMAN` | Flags `state.context["_awaiting_human"]` with the node name. The node still executes, but downstream consumers can check this flag to pause for human input. |

### Context & Memory

| Flag | Engine Behavior |
|------|----------------|
| `NO_HISTORY` | The engine strips all conversation messages before execution, keeping only the system message. After the node runs, original messages are restored (plus any new messages the node added). |
| `ISOLATED_CONTEXT` | The node runs with a clean `state.context` containing only its `input_keys` data. After execution, the original context is restored and only `output_key` is merged back. Prevents context pollution between nodes. |
| `CACHEABLE` | The engine caches the node's `NodeResult` keyed on `node.name + input_keys values`. On subsequent visits with the same inputs, the cached result is returned without re-executing. |

### Model Selection

| Flag | Engine Behavior |
|------|----------------|
| `INJECT_TOOLS` | The node receives all MCP tools discovered by `build_agent()` at runtime. Set via `inject_tools=True` on PromptNode. |
| `LIGHTWEIGHT` | If the node has no `model_override` and the engine was created with a `lightweight_model`, the engine swaps to that model for this node. Use for routing, classification, or simple analysis where a smaller model suffices. |

### Observability

| Flag | Engine Behavior |
|------|----------------|
| `OBSERVABLE` | Emits an additional `on_node_metrics` event to hooks with timing, token count, and tool call stats. |
| `VERBOSE` | Logs the full raw output at DEBUG level. Use for audit-critical nodes. |

### Output Processing

| Flag | Engine Behavior |
|------|----------------|
| `SUMMARIZE_OUTPUT` | If `raw_output` exceeds 1000 characters, the engine calls the LLM to produce a concise summary. Preserves key facts, numbers, and conclusions while removing redundancy. |
| `VALIDATE_OUTPUT` | Validates `result.output` against the node's `output_schema` (Pydantic model or dict). Validation errors are added to `result.guards_failed`. |

### Concurrency & State

| Flag | Engine Behavior |
|------|----------------|
| `READONLY` | Declares that this node only reads state, never writes. Used by ParallelNode and the engine to determine safe concurrent execution. |
| `STATEFUL` | Declares that this node modifies `state.context`. Used for dependency tracking. |
| `PARALLEL_SAFE` | Declares that this node can run concurrently with other `PARALLEL_SAFE` nodes. |

## Pre-built Reasoning Node Flags

Each reasoning node ships with sensible default flags:

| Node | Default Flags | Rationale |
|------|--------------|-----------|
| ThinkNode | `READONLY`, `LIGHTWEIGHT` | Pure reasoning, no state writes, smaller model sufficient |
| ReflectNode | `STATEFUL`, `OBSERVABLE` | Writes reflections to state, worth logging |
| ObserveNode | `STATEFUL` | Writes entities/facts to state.context |
| JustifyNode | `READONLY`, `OBSERVABLE`, `VERBOSE` | Audit trail — full logging matters |
| CritiqueNode | `READONLY`, `OBSERVABLE` | Pure analysis, severity worth logging |
| PlanNode | `STATEFUL`, `OBSERVABLE` | Writes to state.plan |
| SynthesizeNode | `OBSERVABLE` | Final output, important to capture |
| ValidateNode | `READONLY`, `VALIDATE_OUTPUT` | Quality gate, validates own output |
| RetryNode | `RETRYABLE` | Built for retry behavior |
| FanOutNode | `PARALLEL_SAFE` | Designed for concurrent execution |

## Custom Flags

The flag set accepts any hashable value. You can define domain-specific flags:

```python
class MyFlags:
    AUDIT_REQUIRED = "audit_required"
    PII_SENSITIVE = "pii_sensitive"

PromptNode("handle_user_data",
    flags={MyFlags.PII_SENSITIVE, NodeFlag.OBSERVABLE},
)

# Check in hooks
class AuditHook:
    async def post_node(self, node, result, state):
        if node.has_flag(MyFlags.AUDIT_REQUIRED):
            await audit_log.record(node.name, result)
        return result
```

## Engine Configuration

```python
from promptise.engine import PromptGraphEngine

engine = PromptGraphEngine(
    graph=graph,
    model=main_model,
    lightweight_model=small_model,  # Used for LIGHTWEIGHT nodes
)
```
