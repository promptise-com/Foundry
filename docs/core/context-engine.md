# Context Engine

Unified context assembly with token budgeting. Controls exactly what the LLM sees, in what order, with how many tokens — and trims gracefully when the context window is tight.

```python
from promptise import build_agent, ContextEngine

engine = ContextEngine(model="openai:gpt-5-mini")
engine.add_layer("company_policy", priority=7, content="We follow GDPR strictly.")

agent = await build_agent(
    servers=servers, model="openai:gpt-5-mini", context_engine=engine,
)
```

---

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## Why

Without the Context Engine, context is assembled ad-hoc — memory, strategies, prompt blocks, conversation history, and runtime context all inject independently with no coordination. If the total exceeds the model's context window, the LLM either errors out or silently truncates (often losing the user's message at the end).

The Context Engine:
- **Knows the model's context window** (auto-detected or explicit)
- **Counts tokens exactly** (tiktoken for OpenAI, estimation fallback)
- **Assigns priorities** to every context layer
- **Trims by priority** when budget exceeded — conversation history first, then memory, then strategies
- **Reports** per-layer token usage for debugging and optimization

---

## Built-in Layers

The engine registers 13 standard layers with default priorities:

| Priority | Layer | Required | Description |
|----------|-------|----------|-------------|
| 10 | `identity` | ✅ | Agent identity/instructions (never dropped) |
| 10 | `user_message` | ✅ | Current user query (never dropped) |
| 9 | `tools` | ✅ | Tool definitions (never dropped) |
| 9 | `flow` | | Conversation flow phase context |
| 8 | `prompt_blocks` | | PromptBlocks assembly |
| 8 | `output_format` | | Expected output structure |
| 7 | Custom layers | | Developer-added layers |
| 6 | `context_state` | | Runtime: AgentContext state |
| 6 | `mission` | | Runtime: mission objective |
| 5 | `budget` | | Runtime: budget remaining |
| 4 | `inbox` | | Runtime: human operator messages |
| 3 | `memory` | | Long-term memory recall |
| 2 | `strategies` | | Learned strategies from failures |
| 1 | `conversation` | | Conversation history (oldest dropped first) |

Required layers (priority ≥ 9 + `required=True`) are never trimmed, even if they exceed the budget.

---

## Token Counting

The engine uses **exact token counting** when possible:

| Model Provider | Method | Accuracy |
|---------------|--------|----------|
| OpenAI (GPT-*) | `tiktoken` library | Exact |
| Others | Character estimation (chars ÷ 3.5) | ~90% |
| Custom | Your own `Tokenizer` implementation | Developer-defined |

Install tiktoken for exact OpenAI counting: `pip install tiktoken`

### Custom tokenizer

```python
class MyTokenizer:
    def count(self, text: str) -> int:
        return len(my_custom_tokenize(text))

engine = ContextEngine(tokenizer=MyTokenizer())
```

---

## Custom Layers

Add domain-specific context at any priority level:

```python
engine = ContextEngine(model="openai:gpt-5-mini")

# Company knowledge — high priority, always included
engine.add_layer("company_policy", priority=7, content="We follow GDPR...", required=True)

# Seasonal context — medium priority, may be trimmed if window is tight
engine.add_layer("seasonal", priority=4, content="It's Q4 — focus on year-end reports.")

# Background info — low priority, first to be trimmed
engine.add_layer("background", priority=1, content="Historical context about the project...")
```

---

## Trimming

When total context exceeds the token budget, the engine trims lowest-priority non-required layers first:

1. **Conversation history** (priority 1) — oldest messages removed first, preserving user/assistant pairs
2. **Strategies** (priority 2) — truncated from the end
3. **Memory** (priority 3) — truncated from the end
4. **Custom layers** — by their configured priority
5. **Required layers** — NEVER trimmed

```python
report = engine.get_report()
print(f"Budget: {report.budget} tokens")
print(f"Used: {report.total_tokens} tokens ({report.utilization:.0%})")
print(f"Trimmed: {report.trimmed_layers}")
for layer in report.layers:
    print(f"  {layer['name']}: {layer['tokens']} tokens {'(trimmed)' if layer['trimmed'] else ''}")
```

---

## Configuration

```python
ContextEngine(
    model="openai:gpt-5-mini",         # Auto-detect the window (400K, input capped at 272K)
    model_context_window=128_000,       # Or set explicitly (no input cap applied)
    response_reserve=4096,              # Reserve tokens for the response
    tokenizer=my_tokenizer,             # Custom token counter
    auto_register_builtins=True,        # Register standard layers
)
```

### Model context windows (auto-detected)

With `build_agent(context_engine=engine)`, the window comes from the chat
model's own profile metadata (`max_input_tokens` / `max_output_tokens`, which
LangChain chat models carry) when it has one. Otherwise the engine looks the
model name up in this table; unknown models get 128,000. An explicit
`model_context_window` always wins.

| Model | Window | Budget cap |
|-------|--------|------------|
| GPT-4 | 8,192 | |
| GPT-4 Turbo / GPT-4o / GPT-4o mini | 128,000 | |
| GPT-4.1 (all sizes) | 1,047,576 | |
| GPT-5 / GPT-5 mini / GPT-5 nano | 400,000 | 272,000 input tokens |
| o4-mini | 200,000 | |
| Claude 3/4 (all variants) | 200,000 | |
| Llama 3 | 8,192 | |
| Gemini 2.0 / 2.5 Flash, Gemini 2.5 Pro | 1,048,576 | |
| Mistral Large | 128,000 | |

The budget is the window minus `response_reserve`, capped at the model's input
limit when one is known (OpenAI's GPT-5 models accept at most 272,000 input
tokens of their 400,000-token window).

---

## With `build_agent`

On every call, the agent fills the engine's layers for that call only and
assembles them with budgeting:

- **Your layers persist.** Content you set with `add_layer(..., content=...)`
  or `set_content()` is sent on every call; the engine is never cleared.
  Assembly works on a copy, so trimming never changes what you registered.
- **Sent once.** The instructions (`identity`) and the tool definitions
  (`tools`) count toward the budget and appear in the report with
  `"sent": False`: the agent's graph sends them itself, so they are not
  repeated.
- **The question stays as you passed it.** The current user message is the
  last message of the input, unchanged (a `HumanMessage`), and system messages
  in the input (yours, or runtime-injected ones such as `[Context State]`)
  pass through ahead of the engine's layers. Earlier turns go into the
  `conversation` layer and are trimmed oldest first.
- **The budget bounds the tool loop too.** What the budget leaves after the
  instructions and tool definitions becomes the token budget for
  [context compaction](../guides/context-lifecycle.md#how-compaction-works), so
  a long tool loop is compacted once the transcript passes it, on every model
  call, not only the first.

A layer the engine does not have (an engine built with
`auto_register_builtins=False`) is skipped; without a `conversation` layer the
chat history passes through unchanged.

---

## Opt-In Design

The Context Engine is **completely optional**. Without it, the agent uses the existing context injection system (which works fine for most use cases). Enable it when you need:

- Precise token budgeting for small-context models (8K-32K)
- Custom context layers with controlled priority
- Assembly reports for debugging context usage
- Guaranteed no context window overflow

---

## API Reference

### ContextEngine

| Method | Description |
|---|---|
| `add_layer(name, *, priority=5, content="", required=False, trim_strategy="truncate", metadata=None)` | Register a named context layer. Higher priority = kept longer during trimming. |
| `set_content(name, content)` | Set or update a layer's content. |
| `get_content(name) -> str` | Get a layer's current content. |
| `clear_content(name)` | Clear a single layer's content (keeps the registration). |
| `clear_all()` | Clear all layer contents (keeps registrations). |
| `remove_layer(name)` | Remove a layer entirely (unregister). |
| `assemble(content=None, *, budget_only=()) -> list[dict]` | Assemble all layers into a message list with token budgeting. `content` overrides layer content for this assembly only (by layer name); layers in `budget_only` count toward the budget but are not returned. Never changes the registered layers. |
| `get_report() -> ContextReport \| None` | The report of the most recent assembly (also `engine.last_report`). |
| `count_tokens(text) -> int` | Count tokens using the configured tokenizer. |
| `get_layer_info() -> list[dict]` | Introspect all registered layers (name, priority, token estimate). |

| Property | Description |
|---|---|
| `window` | The model's context window. |
| `budget` | Tokens available for the context: the window minus `response_reserve`, capped at `max_input_tokens`. |
| `max_input_tokens` | The model's input limit, when known. |

### ContextLayer

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Layer identifier |
| `priority` | `int` | Higher = more important (10 = identity, 1 = conversation) |
| `content` | `str` | Current content |
| `required` | `bool` | If True, never trimmed |
| `trim_strategy` | `str` | `"truncate"` (default) or `"oldest_first"` (pair-preserving, for the `conversation` layer) |

### ContextReport

Returned by `assemble()` via `engine.last_report`:

| Field | Type | Description |
|---|---|---|
| `total_tokens` | `int` | Total tokens across all included layers |
| `budget` | `int` | Configured token budget |
| `utilization` | `float` | `total_tokens / budget` (0.0 to 1.0) |
| `layers` | `list[dict]` | Per-layer breakdown (name, priority, tokens, required, trimmed, sent) |
| `trimmed_layers` | `list[str]` | Names of layers that were trimmed |

### Built-in Layer Names

When used with `build_agent(context_engine=engine)`, these layers are filled on every call:

| Name | Priority | Content source |
|---|---|---|
| `identity` | 10 | Agent instructions (budgeted, not sent twice) |
| `user_message` | 10 | Current user input (budgeted; the question itself is sent as given) |
| `tools` | 9 | Tool definitions (budgeted, not sent twice) |
| `memory` | 3 | Memory recall results |
| `strategies` | 2 | Adaptive strategy learnings |
| `conversation` | 1 | Chat history before the current question |

Register custom layers for domain-specific context:

```python
engine.add_layer("company_policy", priority=7, required=True)
engine.set_content("company_policy", "Never discuss competitor pricing.")
```

---

## What's Next?

- [Memory](memory.md) — the recall layer that feeds into the engine
- [Adaptive Strategy](adaptive-strategy.md) — the strategies layer
- [Observability](observability.md) — track context usage alongside token costs
