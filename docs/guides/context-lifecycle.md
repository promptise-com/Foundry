# Context Lifecycle Management

The single biggest reason long-running agents get slow, expensive, and *wrong*
is **context bloat**. Every tool call appends its request and result to the
transcript. On a deep task the model ends up re-reading a growing wall of its
own past calls — it loses the thread, re-queries facts it already has, and pays
for thousands of redundant tokens on every turn.

Promptise bounds this for you: the default agent compacts a tool loop once it
gets long, and you can tune or turn that off, or choose per node how much
history it sees. This guide shows the problem, how compaction works, the modes
of `context_scope`, the two ready-made patterns built on them, and a decision
table for picking the right one.

!!! info "Runnable example"
    Everything here is demonstrated end-to-end in
    [`examples/reasoning/verify_and_managed.py`](https://github.com/promptise-com/foundry/blob/main/examples/reasoning/verify_and_managed.py)
    — real LLM calls, just set `OPENAI_API_KEY`.

## The problem: transcripts grow, models drown

A naive tool-calling loop feeds the model the **entire** conversation on every
turn:

```
turn 1:  [system, user]
turn 2:  [system, user, ai→tool, tool_result]
turn 3:  [system, user, ai→tool, tool_result, ai→tool, tool_result]
...
turn 12: [system, user, + 22 more messages]   ← the model re-reads ALL of this
```

For a task that needs ~13 distinct facts, a naive loop can make **dozens** of
tool calls — repeatedly looking up the same employee or record because the
relevant result is buried far back in the transcript. Tokens grow
super-linearly, latency climbs, and accuracy can *drop* as the signal gets lost
in the middle.

## The lever: `context_scope` on `PromptNode`

Every [`PromptNode`](../core/engine-nodes.md#promptnode) accepts a
`context_scope` argument controlling what it sees on each LLM call. A node you
create yourself defaults to `"full"`; the default ReAct agent that
`build_agent()` builds uses `"auto"`.

| Mode | What the node sees | Use it for |
|------|--------------------|------------|
| `"full"` *(`PromptNode` default)* | The whole accumulated transcript | Short tasks, or when every prior message matters |
| `"auto"` *(the ReAct default)* | `"full"` while the tool loop is short, then the compacted view once the run has 6 tool results (or passes a token budget) | The default: simple tasks are unchanged, deep tool loops stay bounded |
| `"ledger"` | The compacted view on every call | Long single-node tool loops that gather many facts then aggregate |
| `"scoped"` | Pinned messages + the current question + **only its own in-progress tool loop** | Multi-stage reasoning graphs: drops the verbose output of *other* stages so tokens don't grow across stages |

```python
from promptise.engine import PromptNode

# Multi-stage graph: each stage only sees its own working set.
PromptNode("analyze", instructions="...", context_scope="scoped")

# Deep tool loop: compact from the first call.
PromptNode("reason", inject_tools=True, context_scope="ledger")
```

### How compaction works

Once a node compacts (`"ledger"`, or `"auto"` past its threshold), each model
call gets a bounded view instead of the transcript:

1. **Pinned, always sent:** every system message in the input, such as
   instructions you or the runtime added (`[Context State]`, mission, budget),
   and the node's own system prompt.
2. **Earlier conversation:** chat history before the current question (what
   `agent.chat()` loads from the session) becomes a short note: the last 6
   messages, 400 characters each.
3. **The current question**, exactly as you passed it. A
   `{"role": "user", ...}` dict, a `HumanMessage` and a `("user", ...)` tuple
   all work: input messages are converted to LangChain messages first.
4. **The latest exchange, verbatim:** the model's last tool call(s) and their
   results, so it sees the outcome of its last action in flow. A parallel
   batch is kept whole.
5. **A ledger of older results**, last: one line per earlier `tool(args)`,
   **last value wins** per `(tool, args)`. A result longer than 2,000
   characters is cut to its first 2,000 with a note naming the call, so the
   model can call it again for the full text. Results already shown in the
   latest exchange are not repeated.

In ledger mode a repeated `(tool, args)` call is **served from cache**
instead of re-executing, so fetching a cut result in full again costs no tool
call.

!!! warning "No LLM summarization"
    Compaction is deterministic: it cuts and drops, it never asks a model to
    summarize, and it costs no extra calls. A fact in the part of an old result
    that was cut is only back in view if the model calls the tool again. If your
    tools return large results that must stay verbatim, raise
    `keep_result_chars`, return leaner results (a summary tool instead of a raw
    dump), or turn compaction off.

### Tune it, or turn it off

`build_agent(context_compaction=...)` sets compaction for the whole agent:

```python
from promptise import build_agent
from promptise.engine import ContextCompaction

# Never compact: the model always sees the full transcript.
agent = await build_agent(..., context_compaction=False)

# Compact after 10 tool results instead of 6.
agent = await build_agent(..., context_compaction=10)

# Full control.
agent = await build_agent(
    ...,
    context_compaction=ContextCompaction(
        after_tool_results=10,     # "auto" threshold
        keep_result_chars=8_000,   # older results longer than this are cut
        max_tokens=60_000,         # also compact past this many tokens, and
                                   # shrink the ledger until the view fits
        history_messages=6,        # earlier messages in the history note
        history_chars=400,         # characters per earlier message
    ),
)
```

`False` keeps `"auto"` nodes on the full transcript; nodes you set to
`"ledger"` or `"scoped"` keep that choice. A node's own
`PromptNode(compaction=...)` wins over the agent's setting.

With a [`ContextEngine`](../core/context-engine.md), the budget the engine has
left after your instructions and tool definitions becomes `max_tokens`, so the
engine keeps bounding the context on every call of the tool loop, not only the
first.

See [Context scope](../core/engine-nodes.md#context-scope) for the node-level
reference.

## Two ready-made patterns

You rarely need to wire a node by hand — two built-in `agent_pattern` values
package these levers for the common cases.

### `verify` — accuracy via a one-turn self-check

A single node that must **plan, solve, and re-check its own answer** within one
generation. You get the accuracy benefit of an explicit verification step at
one-turn latency — no multi-call pipeline.

```python
from promptise import build_agent

agent = await build_agent(
    servers={},                      # no tools needed for pure reasoning
    model="openai:gpt-5-mini",
    agent_pattern="verify",
    instructions="Give only the final answer at the end.",
)

result = await agent.ainvoke({"messages": [
    {"role": "user", "content":
     "A bat and a ball cost $1.10. The bat costs $1.00 more than the ball. "
     "How much is the ball?"}
]})
# The VERIFY step catches the intuitive-but-wrong $0.10 and corrects to $0.05.
```

!!! note "Honest scope"
    `verify` lifts accuracy on **weak and mainstream** models where a forced
    self-check recovers careless errors. A frontier model that already reasons
    internally is usually at its ceiling with a plain prompt, so `verify` there
    is a cheap safety net, not a step change. On a capable model it is
    *comparable to* a well-prompted single pass.

### `managed` — efficiency for deep tool chains

A single tool-using node run with `context_scope="ledger"`. Best for traversing
a database or graph: gather many facts, then aggregate.

```python
from promptise import build_agent

agent = await build_agent(
    servers={"company": my_server_spec},  # or pass extra_tools=[...]
    model="openai:gpt-5-mini",
    agent_pattern="managed",
    instructions=(
        "Answer by calling tools. A ledger of facts you already gathered is "
        "provided each turn — consult it and never re-fetch a fact you have."
    ),
    max_agent_iterations=30,          # deep chains make many calls
)
```

!!! note "Honest scope"
    `managed` is an **efficiency primitive**. On long chains it cuts redundant
    tool calls and bounds token growth at **equal accuracy** — a real cost and
    latency win. It does **not** by itself make the model's final answer more
    correct; if your bottleneck is the model mis-aggregating gathered facts,
    that is a model-capability limit, not a context one.

## Which one should I use?

| Situation | Reach for |
|---|---|
| Short Q&A, short tool loops | Default `react` (`context_scope="auto"`: full until the loop gets long) |
| Every tool result must stay verbatim | `build_agent(..., context_compaction=False)` |
| One question that's easy to get *subtly* wrong | `verify` |
| A long tool chain over a dataset (gather → aggregate) | `managed` |
| A multi-stage custom graph where stages pile up tokens | A custom graph with `context_scope="scoped"` on each stage |
| You need both bounded context *and* a custom topology | Build a [custom graph](../core/agents/reasoning-patterns.md#building-custom-graphs) and set `context_scope` per node |

## Composing it yourself

`context_scope` is a node-level primitive — drop it onto any node in a custom
graph, mixing modes per stage:

```python
from promptise.engine import PromptGraph, PromptNode

graph = PromptGraph("research", mode="static")
graph.add_node(PromptNode("gather", inject_tools=True, context_scope="ledger"))
graph.add_node(PromptNode("write", context_scope="scoped",
                          inherit_context_from="gather"))
graph.sequential("gather", "write")
graph.set_entry("gather")

agent = await build_agent(servers=my_servers, model="openai:gpt-5-mini",
                          agent_pattern=graph)
```

Here `gather` runs a bounded tool loop (ledger), then `write` sees only the
distilled output it inherits plus the task (scoped) — neither stage drowns in
the other's raw messages.

## Key takeaways

- **Context is a resource to manage, not a side effect.** On deep tasks it is
  the deciding factor for cost, latency, and reliability.
- **The default agent compacts long tool loops** (`"auto"`): short tasks see
  the full transcript; past 6 tool results the model sees the pinned messages,
  the current question, its latest exchange and a ledger. Tune it with
  `context_compaction`, or set it to `False`.
- **Compaction never drops the question or your system messages**, and it
  never summarizes with a model: older results are cut, not rewritten.
- **`verify` is the accuracy lever; `managed` is the efficiency lever.** Be
  honest about which problem you have — they solve different ones.

## See also

- [Reasoning Patterns](../core/agents/reasoning-patterns.md) — all 10 built-in patterns
- [Code-Action](code-action.md) — the most radical context move: collapse a long tool loop into one program
- [Nodes reference: Context scope](../core/engine-nodes.md#context-scope) — the full mechanism
- [Prebuilt Patterns](../core/engine-prebuilts.md) — `verify` and `managed` factories
- [`examples/reasoning/verify_and_managed.py`](https://github.com/promptise-com/foundry/blob/main/examples/reasoning/verify_and_managed.py) — runnable demo
