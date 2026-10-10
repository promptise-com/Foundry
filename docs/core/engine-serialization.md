# Serialization & YAML

Save and load graphs as YAML files or Python dicts. A saved graph loads back as the same graph: every node setting round-trips, or saving refuses with an error that names the node and the setting.

## Save a Graph

```python
from promptise.engine.serialization import save_graph

save_graph(graph, "my-agent.yaml")
```

The file is written only if the whole graph can be represented (see [What can't be saved](#what-cant-be-saved)).

## Load a Graph

```python
from promptise.engine.serialization import load_graph

graph = load_graph("my-agent.yaml", tools=agent.tools, refs=[Verdict])
```

| Argument | What it's for |
|----------|---------------|
| `tools` | Tools are saved by name. Pass the tools to resolve them against — a list (typically `agent.tools`) or a `{name: tool}` mapping. A name that isn't provided raises. |
| `refs` | Python objects the file references by import path — `output_schema` models, processors, edge conditions, `@node` functions. Pass them as a list (or a `{"module:QualName": obj}` mapping). References into `promptise` itself resolve without it. |
| `allow_imports` | Import any other reference instead of requiring `refs`. Importing a module runs its code — only for files you trust. Default `False`. |

The file may hold the graph at the top level or under a `graph` key.

## What Round-Trips

| Setting | Stored as |
|---------|-----------|
| Instructions, transitions, `default_next`, `max_iterations`, metadata | Plain data |
| Every node type's own settings — `input_keys`, `output_key`, `inherit_context_from`, `context_scope`, `temperature`, `on_pass`/`on_fail`, `max_subgoals`, … | Plain data |
| Flags (including `inject_tools`, `is_entry`, `is_terminal`) | `flags: [inject_tools, ...]` |
| Tools | Tool names, resolved from `tools=` on load |
| `output_schema`, `preprocessor`, `postprocessor`, `transform`, `merge_fn`, `tool_selector`, a `LoopNode` condition | Import reference `"module:QualName"` |
| `model_override` given as a string | The model id |
| Nested nodes (`ParallelNode`, `LoopNode`, `RetryNode`, `FanOutNode`, `AutonomousNode` pool) and subgraphs | Nested configs |
| An `@node` function | A reference to the module-level node |
| Graph name, `mode`, entry, edges | Plain data |
| Edges from `on_tool_call`, `on_no_tool_call`, `on_output`, `on_error`, `on_confidence`, `on_guard_fail` | A `condition:` mapping |
| Edges from `when(...)` / `loop_until(...)` with a module-level function | `condition: {ref: "module:function"}` |

Values a node would get by default are left out, so a saved graph stays short and picks up improved defaults (for example a reasoning node's built-in instructions).

## What Can't Be Saved

These are Python objects with state of their own. Saving a graph that uses them raises `GraphSerializationError`; remove them before saving and set them in code after loading:

- A lambda, or a function or class defined inside another function (as an `output_schema`, processor or edge condition) — move it to module level. An object defined in the script you run is saved as `__main__:Name`; pass it in `refs=` when loading.
- `blocks`, `strategy`, `perspective`, `guards`, a `RouterNode`'s `context_blocks`.
- A `model_override` that is a model object rather than a model id string (it may carry credentials).
- A node class that isn't registered (see [Custom Node Types](#custom-node-types)).

```python
from promptise.engine.serialization import GraphSerializationError, save_graph

try:
    save_graph(graph, "agent.yaml")
except GraphSerializationError as exc:
    print(exc)
    # Edge act → verify condition: can't be saved — it is a lambda or defined
    # inside a function. Define it at module level in an importable module, ...
```

Loading is just as strict: an unknown node type, an unknown field, a tool that wasn't provided or a reference that can't be resolved raises `GraphSerializationError` (a `ValueError`).

## YAML Format

```yaml
version: 2
name: plan-act-verify
mode: static
entry: plan
nodes:
  plan:
    type: plan
    instructions: Plan how to answer with the sales tools.
    output_schema: my_agent.schemas:Plan
    flags: [observable, stateful]
  act:
    type: prompt
    instructions: Work through the plan with your tools, then write the answer.
    flags: [inject_tools]          # receives the agent's tools at run time
    max_iterations: 8
  verify:
    type: validate
    output_schema: my_agent.schemas:Verdict
    on_pass: __end__
    on_fail: reflect
    max_iterations: 3
    flags: [readonly, validate_output]
  reflect:
    type: reflect
    input_keys: [validation]
    flags: [observable, stateful]
edges:
  - from: plan
    to: act
  - from: act
    to: verify
  - from: reflect
    to: act
  - from: verify
    to: reflect
    condition: {kind: output, key: passes, value: false}
```

A node with `flags: [inject_tools]` (or `inject_tools: true`) gets the agent's tools when the graph runs, so the file doesn't need to name them. Hand-written files may also use `is_entry: true` / `is_terminal: true`.

### Edge Config Format

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `from` | `str` | Yes | Source node name |
| `to` | `str` | Yes | Target node name |
| `label` | `str` | No | Label for visualization |
| `priority` | `int` | No | Priority for condition checking (higher = checked first) |
| `condition` | mapping | No | `{kind: tool_called}`, `{kind: no_tool_call}`, `{kind: error}`, `{kind: guard_failed}`, `{kind: output, key: ..., value: ...}`, `{kind: confidence, min_confidence: 0.7}`, or `{ref: "module:function"}` |

The built-in conditions are `EdgeCondition` objects (`from promptise.engine import EdgeCondition`), so `graph.when("a", "b", condition=EdgeCondition("output", key="done", value=True))` saves too.

## Python Dict Format

```python
from promptise.engine.serialization import graph_from_config, graph_to_config

config = graph_to_config(graph)                    # plain data, with "version": 2
graph = graph_from_config(config, tools=agent.tools)
```

`node_to_config(node)` and `node_from_config(config, tools=..., refs=..., allow_imports=...)` do the same for a single node.

## Node Type Registry

All built-in node types are registered:

| Type | Node Class | Category |
|------|-----------|----------|
| `prompt` | PromptNode | Standard |
| `tool` | ToolNode | Standard |
| `router` | RouterNode | Standard |
| `guard` | GuardNode | Standard |
| `parallel` | ParallelNode | Standard |
| `loop` | LoopNode | Standard |
| `human` | HumanNode | Standard |
| `transform` | TransformNode | Standard |
| `subgraph` | SubgraphNode | Standard |
| `autonomous` | AutonomousNode | Standard |
| `code_action` | CodeActionNode | Standard |
| `think` | ThinkNode | Reasoning |
| `reflect` | ReflectNode | Reasoning |
| `observe` | ObserveNode | Reasoning |
| `justify` | JustifyNode | Reasoning |
| `critique` | CritiqueNode | Reasoning |
| `plan` | PlanNode | Reasoning |
| `synthesize` | SynthesizeNode | Reasoning |
| `validate` | ValidateNode | Reasoning |
| `retry` | RetryNode | Reasoning |
| `fan_out` | FanOutNode | Reasoning |

A `CodeActionNode` built by `build_agent(agent_pattern="code-action")` holds a sandbox factory created at run time, so it can't be saved; one you build with a module-level factory can.

## Custom Node Types

Register custom node types so they can be saved and loaded:

```python
from promptise.engine.serialization import register_node_type

class AuditedNode(PromptNode):
    """A PromptNode subclass with no constructor parameters of its own."""

register_node_type("audited", AuditedNode)
```

A registered subclass of a built-in node keeps all of the built-in's settings. If its constructor takes parameters of its own, define `to_config()` and `from_config()` so they are saved too — otherwise saving raises rather than dropping them. `from_config()` receives the node's whole config, base fields such as `instructions` and `default_next` included:

```python
class DatabaseNode(BaseNode):
    def __init__(self, name: str, *, table: str, **kwargs):
        super().__init__(name, **kwargs)
        self.table = table

    def to_config(self) -> dict:
        return {"table": self.table, "default_next": self.default_next}

    @classmethod
    def from_config(cls, config: dict) -> "DatabaseNode":
        return cls(config["name"], table=config["table"], default_next=config.get("default_next"))

register_node_type("database", DatabaseNode)
```

```yaml
nodes:
  fetch_data:
    type: database
    table: users
    default_next: analyze
```
