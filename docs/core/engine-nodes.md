# Nodes

Every node in a Reasoning Graph is a complete processing pipeline. Promptise ships with **20 built-in node types** — plus three ways to create custom ones.

## Reasoning Nodes (Pre-built Building Bricks)

These come fully configured with instructions, context management, and state updates. Just pick the ones you need. All reasoning nodes inherit from `PromptNode` (except RetryNode and FanOutNode which inherit from `BaseNode`).

### ThinkNode

Gap analysis and next-step reasoning. No tools — pure analytical reasoning.

```python
ThinkNode("think")
ThinkNode("think", focus_areas=["data quality", "missing sources"])
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"think"` | Node identifier |
| `focus_areas` | `list[str] \| None` | `None` | Optional areas to focus analysis on |
| `**kwargs` | | | All PromptNode parameters |

**Pre-configured behavior:**

- Analyzes current state, identifies information gaps, rates confidence (1-5), recommends next step
- Auto-injects observations and plan from state
- Output key: `think_output`
- Default flags: `READONLY`, `LIGHTWEIGHT`

### ReflectNode

Self-evaluation and mistake correction. Auto-stores reflections in state.

```python
ReflectNode("reflect")
ReflectNode("reflect", review_depth=5)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"reflect"` | Node identifier |
| `review_depth` | `int` | `3` | Number of past steps to review |
| `**kwargs` | | | All PromptNode parameters |

**Pre-configured behavior:**

- Assesses progress, identifies mistakes, suggests corrections, rates confidence
- Decides route: `continue`, `replan`, or `answer`
- Auto-stores reflections in `state.reflections` (capped at 5)
- Output key: `reflection`
- Default flags: `STATEFUL`, `OBSERVABLE`

### ObserveNode

Interprets tool results into structured data. Enriches state with extracted entities and facts.

```python
ObserveNode("observe")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"observe"` | Node identifier |
| `**kwargs` | | | All PromptNode parameters |

**Pre-configured behavior:**

- Summarizes findings, extracts named entities, extracts factual claims, identifies key findings
- Merges extracted entities into `state.context["extracted_entities"]`
- Merges extracted facts into `state.context["extracted_facts"]`
- Output key: `observation`
- Default flags: `STATEFUL`

### PlanNode

Structured planning with subgoal management and quality self-evaluation.

```python
PlanNode("plan", max_subgoals=4, quality_threshold=3)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"plan"` | Node identifier |
| `max_subgoals` | `int` | `4` | Maximum number of subgoals |
| `quality_threshold` | `int` | `3` | Plans below this quality score (1-5) trigger re-planning |
| `**kwargs` | | | All PromptNode parameters |

**Pre-configured behavior:**

- Creates prioritized subgoals, identifies first to tackle, self-evaluates quality
- Auto-manages `state.plan` and `state.completed`
- Re-plans if quality score < threshold (loops back to self or follows `replan` transition)
- Output key: `plan_output`
- Default flags: `STATEFUL`, `OBSERVABLE`

### SynthesizeNode

Combines all gathered data into a comprehensive final answer.

```python
SynthesizeNode("answer", is_terminal=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"synthesize"` | Node identifier |
| `**kwargs` | | | All PromptNode parameters |

**Pre-configured behavior:**

- Reads ALL observations, reflections, and plan progress
- Addresses original question, supports claims with evidence, cites sources
- Default `default_next="__end__"` (terminal by default)
- Output key: `synthesis`
- Default flags: `OBSERVABLE`

### CritiqueNode

Adversarial self-review with severity-based routing.

```python
CritiqueNode("critique", severity_threshold=0.5)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"critique"` | Node identifier |
| `severity_threshold` | `float` | `0.5` | Severity above this routes to revision (0.0 = perfect, 1.0 = flawed) |
| `**kwargs` | | | All PromptNode parameters |

**Pre-configured behavior:**

- Identifies weaknesses, presents counter-arguments, suggests improvements, rates severity
- If severity > threshold, routes to first matching transition: `revise`, `improve`, `retry`, or `replan`
- Output key: `critique`
- Default flags: `READONLY`, `OBSERVABLE`

### JustifyNode

Chain-of-thought justification for audit trail.

```python
JustifyNode("justify")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"justify"` | Node identifier |
| `**kwargs` | | | All PromptNode parameters |

**Pre-configured behavior:**

- Explains WHY the last action was taken with step-by-step reasoning chain
- Cites specific evidence, states conclusion, rates confidence
- Auto-stores justification chain in `state.context["justifications"]` (capped at 5)
- Output key: `justification`
- Default flags: `READONLY`, `OBSERVABLE`, `VERBOSE`

### ValidateNode

LLM-powered quality validation with pass/fail routing.

```python
ValidateNode("validate",
    criteria=["Be accurate", "Cite sources"],
    on_pass="deliver",
    on_fail="revise",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"validate"` | Node identifier |
| `criteria` | `list[str] \| None` | `None` | Validation criteria (defaults to accuracy, evidence, no factual errors) |
| `on_pass` | `str` | `"__end__"` | Node to transition to on pass |
| `on_fail` | `str \| None` | `None` | Node to transition to on fail |
| `**kwargs` | | | All PromptNode parameters |

**Pre-configured behavior:**

- Evaluates output against criteria, lists issues, suggests improvements
- Routes to `on_pass` or `on_fail` based on result
- Output key: `validation`
- Default flags: `READONLY`, `VALIDATE_OUTPUT`

### RetryNode

Wraps another node with retry logic and error enrichment. Extends BaseNode (not PromptNode).

```python
RetryNode("safe_search", wrapped_node=search_node, max_retries=3, backoff_factor=0.5)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Node identifier |
| `wrapped_node` | `BaseNode` | *required* | The node to wrap with retry logic |
| `max_retries` | `int` | `3` | Maximum retry attempts |
| `backoff_factor` | `float` | `0.5` | Base delay multiplier for exponential backoff |
| `**kwargs` | | | All BaseNode parameters |

**Behavior:**

- Retries `wrapped_node` on failure with exponential backoff
- Between retries, enriches `state.context` with `_retry_attempt`, `_retry_error`, `_retry_node`
- Default flags: `RETRYABLE`

### FanOutNode

Sends different sub-questions to parallel children with state overrides. Extends BaseNode.

```python
FanOutNode("gather", branches=[
    (PromptNode("web", tools=[web_search]), {"focus": "news"}),
    (PromptNode("docs", tools=[doc_search]), {"focus": "technical"}),
])
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Node identifier |
| `branches` | `list[tuple[BaseNode, dict]]` | `None` | List of (node, state_overrides) tuples |
| `merge_strategy` | `str` | `"dict"` | How to merge results |
| `**kwargs` | | | All BaseNode parameters |

**Behavior:**

- Each child node gets a separate GraphState with its own context overrides
- All branches run concurrently via `asyncio.gather`
- Results merged into a dict keyed by child node name
- Default flags: `PARALLEL_SAFE`

---

## Standard Nodes

### PromptNode

The core LLM reasoning node. Full prompt-assembly pipeline with tool calling, guards, strategies, and context management.

```python
PromptNode("analyze",
    instructions="Analyze the data.",
    blocks=[Identity("Analyst"), Rules(["Cite sources"])],
    strategy=chain_of_thought,
    tools=my_tools,
    inject_tools=True,
    output_key="analysis",
    input_keys=["raw_data"],
    inherit_context_from="search",
    preprocessor=enrich_fn,
    postprocessor=format_fn,
    guards=[SchemaStrictGuard(AnalysisOutput)],
    output_schema=AnalysisOutput,
    transitions={"complete": "report", "need_data": "search"},
    model_override="openai:gpt-4o-mini",
    temperature=0.2,
    max_tokens=4096,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Node identifier |
| `instructions` | `str` | `""` | Natural-language instructions injected into the system prompt |
| `blocks` | `list[Any]` | `None` | PromptBlock objects (Identity, Rules, OutputFormat, etc.) appended to system prompt |
| `strategy` | `Any` | `None` | Reasoning strategy (ChainOfThought, SelfCritique, etc.) wraps the prompt |
| `perspective` | `Any` | `None` | Perspective framing (Analyst, Critic, etc.) prepended to prompt |
| `tools` | `list[BaseTool]` | `None` | LangChain tools bound to the model for this node |
| `tool_choice` | `str` | `"auto"` | Tool calling mode (`"auto"`, `"required"`, `"none"`) |
| `inject_tools` | `bool` | `False` | If True, receives all MCP tools discovered by `build_agent()` at runtime |
| `output_schema` | `type` | `None` | Pydantic model for structured output via `with_structured_output()` |
| `guards` | `list[Any]` | `None` | Output guards (ContentFilterGuard, SchemaStrictGuard, etc.) |
| `model_override` | `Any` | `None` | Per-node model — a `BaseChatModel` instance or string like `"openai:gpt-4o-mini"` |
| `context_layers` | `dict[str, int]` | `None` | Extra context keys to inject from state, with priority values |
| `max_tokens` | `int` | `4096` | Max tokens for the LLM call |
| `temperature` | `float` | `0.0` | LLM temperature |
| `input_keys` | `list[str]` | `None` | Keys from `state.context` to inject as "Input data" in the prompt |
| `output_key` | `str` | `None` | Write `result.output` to `state.context[output_key]` after execution |
| `inherit_context_from` | `str` | `None` | Inject the output of another node (reads `state.context["{name}_output"]`) |
| `context_scope` | `str` | `"full"` | Context lifecycle mode: `"auto"` (full→ledger automatically; the ReAct default), `"full"` (whole transcript), `"scoped"` (bounded working set), or `"ledger"` (deduplicated facts ledger) — see [Context scope](#context-scope) |
| `auto_ledger_after` | `int` | `None` | For `context_scope="auto"`: number of accumulated tool results after which the node switches from the full transcript to the compacted view (default 6, or the agent's `context_compaction` setting) |
| `compaction` | `ContextCompaction \| bool \| int` | `None` | This node's [compaction settings](../guides/context-lifecycle.md#tune-it-or-turn-it-off); wins over `build_agent(context_compaction=...)`. `False` keeps an `"auto"` node on the full transcript |
| `preprocessor` | `Callable` | `None` | Runs before the LLM call: `fn(state, config) -> None` |
| `postprocessor` | `Callable` | `None` | Runs after: `fn(output, state, config) -> Any` |
| `include_observations` | `bool` | `True` | Auto-inject recent tool results from `state.observations` |
| `include_plan` | `bool` | `True` | Auto-inject current plan/subgoals from `state.plan` |
| `include_reflections` | `bool` | `True` | Auto-inject past learnings from `state.reflections` |
| `transitions` | `dict[str, str]` | `None` | Map output keys to next-node names (e.g. `{"proceed": "act"}`) |
| `default_next` | `str` | `None` | Fallback node if no transition matches |
| `max_iterations` | `int` | `10` | Max times this node can execute in one graph run |
| `flags` | `set[NodeFlag]` | `None` | Typed flags controlling engine behavior |
| `is_entry` | `bool` | `False` | Shorthand for adding `NodeFlag.ENTRY` |
| `is_terminal` | `bool` | `False` | Shorthand for adding `NodeFlag.TERMINAL` |

**Execution pipeline** (9 steps):

1. Preprocessor (custom data transformation)
2. Context assembly (instructions + blocks + input_keys + inherited context + observations/plan/reflections)
3. Strategy wrapping + perspective framing
4. Tool binding (`bind_tools()`) and structured output (`with_structured_output()`)
5. LLM call (`model.ainvoke(messages)`)
6. Response processing (tool calls → tool execution → loop back; or parse output)
7. Guard checking (validate output — supports both sync and async guards)
8. Strategy parsing (extract answer from strategy markers)
9. Postprocessor + write to `state.context[output_key]`

**Performance notes:**

- **Auto schema injection** — tool-using nodes automatically inject tool names, parameter types, and descriptions into the system prompt so the LLM knows exact parameter names and types
- **Parallel tool execution** — when the LLM requests 2+ tool calls in one response, they execute concurrently via `asyncio.gather`
- **Cached on re-entry** — on tool-loop re-entries, the system prompt, tool bindings, and model wrapper are reused from the first call
- **No observation duplication** — observation injection is skipped for tool-using nodes (data already in conversation as ToolMessages)
- **Separate SystemMessage** — the node's prompt is stored separately from the agent's SystemMessage to prevent inflation

#### Context scope

`context_scope` controls **how much accumulated history the node sees on each LLM
call** — the single most important lever for context-lifecycle management on long
or multi-stage tasks. The full transcript grows on every tool call; left
unbounded it inflates token cost super-linearly and the model starts losing track
of facts it already gathered.

The **default ReAct pattern uses `"auto"`** — context handling is automatic, with
no pattern to choose. The four modes:

| Mode | What the node sees | Use it for |
|------|--------------------|------------|
| `"auto"` *(ReAct default)* | `"full"` while the tool loop is short, then `"ledger"` once it grows past `auto_ledger_after` (default 6) tool results | The smart default — simple tasks unchanged, deep tool loops bounded automatically |
| `"full"` | The whole accumulated transcript | When every prior message must always be visible |
| `"scoped"` | The node's system prompt (carrying any inherited/injected distilled state) + the input's system messages + the **current** user question + **only this node's own in-progress tool loop** | Multi-stage reasoning graphs — drops the verbose intermediate messages produced by *other* stages so tokens don't grow across stages |
| `"ledger"` | System prompt + the input's system messages + a short note on earlier conversation + the **current** user question + the **most recent** assistant turn and its tool results + a compact **deduplicated "facts gathered" ledger** of older results | Long single-node tool loops over an interconnected dataset, where the model otherwise re-queries the same facts dozens of times |

```python
# Default agents already get this — no configuration needed:
PromptNode("reason", inject_tools=True, context_scope="auto", auto_ledger_after=6)

# Multi-stage graph: each stage only sees its own working set.
PromptNode("analyze", instructions="...", context_scope="scoped")

# Force the ledger from turn 0 (deep tool loop you know will be long):
PromptNode("reason", inject_tools=True, context_scope="ledger")
```

!!! note "Honest scope of `auto`"
    `auto` is an **efficiency / context-bounding** default: it keeps tokens and
    context manageable on deep loops at **zero change to short tasks** (below the
    threshold it is byte-for-byte `"full"`). It does **not** by itself make the
    model aggregate gathered facts more accurately — for that, change the action
    space with [code-action](engine-prebuilts.md#code-action-one-sandboxed-program).

**How `"ledger"` works:**

- The input's system messages (yours and runtime-injected ones such as `[Context State]`) and the **current** user question are always sent. Dict messages (`{"role": "user", ...}`) are converted to LangChain messages first, so every input form works; with chat history, the question is the *last* user message and earlier turns become a short note.
- The ledger is built from the tool calls in the transcript older than the latest exchange — one line per `tool(args) = result`, **last value wins** per `(tool, args)` so duplicates collapse. Results longer than `keep_result_chars` (2,000) are cut to an excerpt that names the call; results shown in the latest exchange are not repeated.
- It is placed **last**, immediately before the model's turn, where it is most salient — so the model consults it instead of re-calling a tool.
- The most recent assistant turn and its tool results (a whole parallel batch) are kept *in-flow* so the model doesn't lose continuity and loop.
- Tool execution is **cache-served**: if the node requests a `(tool, args)` pair already present in `state.observations`, the cached result is returned instead of re-executing — deep tasks otherwise re-fetch identical facts repeatedly.

This is the mechanism behind the [`managed` prebuilt pattern](engine-prebuilts.md#managed-context-managed-tool-loop). It is an **efficiency primitive** — it bounds token growth and removes redundant tool calls without changing the answer the model produces.

### ToolNode

Explicit tool execution with input validation and deduplication.

```python
ToolNode("search",
    tools=[search_tool, wiki_tool],
    validate_inputs=True,
    deduplicate=True,
    max_result_chars=4000,
    tool_selector=my_selector_fn,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Node identifier |
| `tools` | `list[BaseTool]` | `None` | Available tools |
| `validate_inputs` | `bool` | `True` | Validate tool arguments before calling |
| `deduplicate` | `bool` | `True` | Skip duplicate tool calls with same args |
| `max_result_chars` | `int` | `4000` | Truncate tool results to this length |
| `tool_selector` | `Callable` | `None` | Custom function `(state) -> (tool_name, args)` to pick which tool to call |

### RouterNode

LLM decides which path to take from a set of named routes.

```python
RouterNode("route",
    routes={"search": "search_node", "answer": "synthesize", "clarify": "ask_user"},
    model_override="openai:gpt-4o-mini",
    context_blocks=[Rules(["Choose the most relevant path"])],
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Node identifier |
| `routes` | `dict[str, str]` | `None` | Map of route labels to target node names |
| `context_blocks` | `list[Any]` | `None` | Blocks to include in the routing prompt |
| `model_override` | `BaseChatModel` | `None` | Use a different (typically lighter) model for routing |

### GuardNode

Programmatic validation and gating with pass/fail routing.

```python
GuardNode("check_quality",
    guards=[LengthGuard(min_chars=100), ContentFilterGuard(blocked=["todo"])],
    target_key="draft",
    on_pass="publish",
    on_fail="revise",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Node identifier |
| `guards` | `list[Any]` | `None` | Guard instances to run |
| `target_key` | `str` | `None` | Key in `state.context` to validate |
| `on_pass` | `str` | `"__end__"` | Transition on all guards passing |
| `on_fail` | `str \| None` | `None` | Transition on any guard failing |

### ParallelNode

Run multiple child nodes concurrently and merge results.

```python
ParallelNode("parallel_search",
    nodes=[search_node, wiki_node, docs_node],
    merge_strategy="concatenate",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Node identifier |
| `nodes` | `list[BaseNode]` | `None` | Child nodes to run concurrently |
| `merge_strategy` | `str` | `"concatenate"` | How to combine results (`"concatenate"`, `"dict"`, `"first"`) |
| `merge_fn` | `Callable` | `None` | Custom merge function overriding `merge_strategy` |

### LoopNode

Repeat a body node until a condition is met.

```python
LoopNode("refine",
    body_node=edit_node,
    condition=lambda state: state.context.get("quality_score", 0) >= 4,
    max_loop_iterations=5,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Node identifier |
| `body_node` | `BaseNode` | `None` | The node to execute on each iteration |
| `condition` | `Callable` | `None` | `(state) -> bool` — loop exits when True |
| `max_loop_iterations` | `int` | `5` | Maximum loop iterations |

### HumanNode

Pause for human approval with timeout.

```python
HumanNode("approve",
    prompt_template="Approve sending this email?",
    timeout=300.0,
    on_approve="send",
    on_deny="revise",
    on_timeout="cancel",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Node identifier |
| `prompt_template` | `str` | `"Approve this action?"` | Prompt shown to the human |
| `timeout` | `float` | `300.0` | Seconds to wait for human input |
| `on_approve` | `str` | `"__end__"` | Transition on approval |
| `on_deny` | `str \| None` | `None` | Transition on denial |
| `on_timeout` | `str` | `"__end__"` | Transition on timeout |

### TransformNode

Pure data transformation without an LLM call.

```python
TransformNode("format",
    transform=lambda state, config: {"formatted": state.context["raw"].upper()},
    output_key="formatted_data",
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Node identifier |
| `transform` | `Callable` | `None` | `(state, config) -> Any` transformation function |
| `output_key` | `str` | `"transform_result"` | Where to store result in `state.context` |

### SubgraphNode

Embed a complete sub-graph as a single node.

```python
SubgraphNode("research_pipeline",
    subgraph=research_graph,
    inherit_state=True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Node identifier |
| `subgraph` | `PromptGraph` | `None` | The sub-graph to execute |
| `inherit_state` | `bool` | `True` | Whether the sub-graph inherits the parent's state |

### AutonomousNode

Agent dynamically picks which node to execute from a pool.

```python
AutonomousNode("agent",
    node_pool=[think_node, search_node, answer_node],
    planner_instructions="Choose the most productive next step.",
    max_steps=15,
    entry_node="think",
    terminal_nodes=["answer"],
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"autonomous"` | Node identifier |
| `node_pool` | `list[BaseNode]` | `None` | Available nodes the agent can choose from |
| `planner_instructions` | `str` | `""` | Extra instructions for the planning LLM |
| `allow_repeat` | `bool` | `True` | Allow the agent to pick the same node multiple times |
| `max_steps` | `int` | `15` | Maximum autonomous steps |
| `entry_node` | `str \| None` | `None` | Force a specific first node |
| `terminal_nodes` | `list[str]` | `None` | Nodes that can end the autonomous loop |

---

## Custom Nodes

### @node decorator

Turn any async function into a graph node.

```python
from promptise.engine import node, GraphState, NodeResult, NodeFlag

@node("fetch_weather", default_next="respond", flags={NodeFlag.RETRYABLE})
async def fetch_weather(state: GraphState) -> NodeResult:
    weather = await api.get(state.context.get("city", "Berlin"))
    state.context["weather"] = weather
    return NodeResult(node_name="fetch_weather", output=weather)

graph.add_node(fetch_weather)
```

The decorator supports all BaseNode parameters: `instructions`, `transitions`, `default_next`, `max_iterations`, `metadata`, `flags`, `is_entry`, `is_terminal`.

The function can accept `(state)`, `(state, config)`, or `()`.

### BaseNode subclass

For full control, subclass BaseNode directly.

```python
from promptise.engine import BaseNode, NodeResult, NodeFlag

class DatabaseNode(BaseNode):
    def __init__(self, name, *, query_template, **kwargs):
        super().__init__(name, flags={NodeFlag.RETRYABLE, NodeFlag.STATEFUL}, **kwargs)
        self.template = query_template

    async def execute(self, state, config):
        result = await db.query(self.template.format(**state.context))
        state.context[f"{self.name}_output"] = result
        return NodeResult(node_name=self.name, output=result)
```

### BaseNode Parameters

All nodes inherit these parameters from BaseNode:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | *required* | Unique identifier within the graph |
| `instructions` | `str` | `""` | Description of what the node does |
| `description` | `str` | `""` | Short description for visualization (defaults to first 80 chars of instructions) |
| `transitions` | `dict[str, str]` | `None` | Output key to next-node mapping |
| `default_next` | `str` | `None` | Fallback transition |
| `max_iterations` | `int` | `10` | Max executions per graph run |
| `metadata` | `dict` | `None` | Arbitrary metadata for hooks and observability |
| `is_entry` | `bool` | `False` | Adds `NodeFlag.ENTRY` |
| `is_terminal` | `bool` | `False` | Adds `NodeFlag.TERMINAL` |
| `flags` | `set[NodeFlag]` | `None` | Typed flags controlling engine behavior |

---

## Node Flags

Nodes declare their capabilities through typed `NodeFlag` enums. The engine processes these flags at runtime to control execution, caching, error handling, and observability.

```python
from promptise.engine import NodeFlag

PlanNode("plan", is_entry=True)                                    # ENTRY flag
PromptNode("act", inject_tools=True)                               # INJECT_TOOLS flag
SynthesizeNode("answer", is_terminal=True)                         # TERMINAL flag
PromptNode("critical_step", flags={NodeFlag.CRITICAL})             # Abort on error
PromptNode("optional", flags={NodeFlag.SKIP_ON_ERROR, NodeFlag.LIGHTWEIGHT})
```

16 built-in flags cover execution control, context isolation, model selection, observability, and output processing. See [Node Flags](engine-flags.md) for the full reference.

---

## Skills Library

Pre-configured node factories for common tasks:

```python
from promptise.engine.skills import (
    # Standard skills
    web_researcher, code_reviewer, data_analyst, fact_checker,
    summarizer, planner, decision_router, formatter,
    # Reasoning skills
    thinker, reflector, critic, justifier, synthesizer,
    validator_node, observer_node,
)

graph.add_node(planner("plan", is_entry=True))
graph.add_node(web_researcher("search", tools=search_tools))
graph.add_node(reflector("reflect"))
graph.add_node(synthesizer("conclude", is_terminal=True))
```

See [Skills Library](engine-skills.md) for all 15 skill factories with parameters.
