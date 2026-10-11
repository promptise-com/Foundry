# Tool Optimization

Reduce the token cost of MCP tool definitions sent to the LLM with every invocation.

```python
from promptise import build_agent
from promptise.config import HTTPServerSpec

agent = await build_agent(
    servers={"tools": HTTPServerSpec(url="http://localhost:8000/mcp")},
    model="openai:gpt-5-mini",
    optimize_tools=True,  # "minimal": shorter tool definitions, same tools
)
```

---

## The Problem

When an agent connects to MCP servers, every tool's full name, description, and JSON Schema is sent to the LLM on every invocation via function-calling. With 20-50+ tools, this costs 5,000-15,000+ tokens per call — just for tool definitions. This is the single largest token cost after the conversation itself.

## How It Works

Tool optimization operates at two layers:

**Layer 1: Static optimization** (applied once at build time) — reduces the per-tool token cost without changing which tools are available:

- **Schema minification** — strips `description` metadata from Pydantic Field schemas. The LLM still sees field names, types, and required status — but not verbose per-field descriptions.
- **Description truncation** — caps tool-level descriptions at N characters, ending on a whole sentence when that keeps most of the budget, otherwise at a word boundary with a single `...`.
- **Depth flattening** — replaces deeply nested objects with `dict` beyond a configurable depth.

**Layer 2: Semantic tool selection** (applied before every model call) — the biggest optimization. Instead of sending all 50 tools, only the most relevant ones are offered:

1. At build time, every tool's name and description is embedded with a small local model.
2. Before **each model call** (not just once per `ainvoke()`), a query is built from the recent conversation — the latest user message, the assistant reply it answers, the tool calls of the previous and current turn, and earlier user turns — and compared against the tool embeddings. A follow-up like "Yes, go ahead." therefore keeps the tools of the request it confirms, and the selection follows the agent as it calls tools.
3. The model is offered the `semantic_top_k` most relevant tools, plus every tool in `preserve_tools`, tools already called in the previous or current turn, and the `request_more_tools` fallback.
4. If the right tool is missing, the model calls `request_more_tools`; the tools it returns are offered from the next model call on.

---

## Quick Start

### One-liner

```python
agent = await build_agent(
    servers=servers, model="openai:gpt-5-mini",
    optimize_tools=True,  # Uses "minimal" preset
)
```

### Install

Static optimization (`True`, `"minimal"`, `"standard"`) needs nothing extra. Semantic selection embeds tools with `sentence-transformers`:

```bash
pip install "promptise[tool-optimization]"   # also included in promptise[all]
```

Without it, `build_agent(optimize_tools="semantic")` raises an `ImportError` that names this extra, before connecting to any server.

### Preset levels

```python
# Static optimization only — safe, no behavioral change
agent = await build_agent(servers=servers, model="openai:gpt-5-mini", optimize_tools="minimal")

# Deeper minification + nested description stripping
agent = await build_agent(servers=servers, model="openai:gpt-5-mini", optimize_tools="standard")

# Full semantic selection — biggest savings, per-invocation tool filtering
agent = await build_agent(servers=servers, model="openai:gpt-5-mini", optimize_tools="semantic")
```

### Fine-grained control

```python
from promptise import ToolOptimizationConfig, OptimizationLevel

agent = await build_agent(
    servers=servers,
    model="openai:gpt-5-mini",
    optimize_tools=ToolOptimizationConfig(
        level=OptimizationLevel.SEMANTIC,
        max_description_length=100,
        semantic_top_k=5,
        preserve_tools={"critical_payment_tool", "auth_tool"},
    ),
)
```

---

## Three Preset Levels

| Setting | `minimal` | `standard` | `semantic` |
|---|---|---|---|
| Schema minification | Yes | Yes | Yes |
| Max description length | 200 chars | 150 chars | 100 chars |
| Strip nested descriptions | No | Yes | Yes |
| Max schema depth | No limit | 3 | 2 |
| Semantic selection | No | No | Yes |
| Semantic top-K | — | — | 8 |
| Fallback tool | — | — | Yes |
| **Measured savings** (see below) | **14%** | **14%** | **90%** |

The savings are tool-definition tokens per model call, measured on one server: 90 tools with flat schemas (string parameters with one-line descriptions, tool descriptions under 200 characters). Static optimization saves what your schemas spend on parameter descriptions, long tool descriptions and nesting, so expect more on verbose or deeply nested schemas and less on terse ones; `standard` only pulls ahead of `minimal` when schemas are nested. Semantic selection's saving grows with the number of tools, since the model sees about `semantic_top_k` of them whatever the total. Measure your own payload with [observability](observability.md) before relying on a number.

---

## Configuration Reference

### ToolOptimizationConfig

| Field | Type | Default | Description |
|---|---|---|---|
| `level` | `OptimizationLevel \| None` | `None` | Preset level. Any explicit field overrides the preset. |
| `minify_schema` | `bool \| None` | from preset | Strip `description` from Pydantic Field metadata |
| `max_description_length` | `int \| None` | from preset | Truncate tool descriptions at N chars |
| `strip_nested_descriptions` | `bool \| None` | from preset | Remove descriptions from nested model fields |
| `max_schema_depth` | `int \| None` | from preset | Flatten nested objects beyond this depth to `dict` |
| `semantic_selection` | `bool \| None` | from preset | Select tools semantically before every model call (needs `promptise[tool-optimization]`) |
| `semantic_top_k` | `int \| None` | from preset (8) | Most-relevant tools offered per model call; preserved, recently called and unlocked tools and the fallback come on top |
| `semantic_context_turns` | `int \| None` | from preset (3) | Recent user turns that make up the selection query |
| `always_include_fallback` | `bool \| None` | from preset | Include the `request_more_tools` fallback |
| `embedding_model` | `str \| None` | `"all-MiniLM-L6-v2"` | Model name or **local path** for sentence-transformers |
| `preserve_tools` | `set[str] \| None` | `None` | Tool names that are never optimized and always offered |

### OptimizationLevel

| Level | Value |
|---|---|
| `OptimizationLevel.MINIMAL` | `"minimal"` |
| `OptimizationLevel.STANDARD` | `"standard"` |
| `OptimizationLevel.SEMANTIC` | `"semantic"` |

---

## Semantic Selection Details

### Embedding model

Semantic selection embeds tool descriptions using `sentence-transformers`. The default model is `all-MiniLM-L6-v2` (384 dimensions, no API key needed). On first use it downloads from HuggingFace Hub and caches locally at `~/.cache/huggingface/`. After that, it runs fully offline.

#### Local / air-gapped deployments

For enterprises that cannot access external networks, download the model files once and point to a local directory:

```bash
# Download the model on a machine with internet
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2').save('/models/all-MiniLM-L6-v2')"
```

```python
# Use the local model — zero network calls
agent = await build_agent(
    servers=servers,
    model="openai:gpt-5-mini",
    optimize_tools=ToolOptimizationConfig(
        level=OptimizationLevel.SEMANTIC,
        embedding_model="/models/all-MiniLM-L6-v2",
    ),
)
```

You can also use any other `sentence-transformers`-compatible model:

```python
optimize_tools=ToolOptimizationConfig(
    level=OptimizationLevel.SEMANTIC,
    embedding_model="BAAI/bge-small-en-v1.5",  # or any local path
)
```

### The `request_more_tools` fallback

When semantic selection is active, every model call is also offered a fallback tool (turn it off with `always_include_fallback=False`):

```
Tool: request_more_tools(query?: str, tool_names?: list[str])
"Call this when none of your current tools can do what is needed. Describe the
 capability in `query` (or give exact `tool_names`); the matching tools are
 returned and you can call them on your next step. Without arguments it returns
 the next few tools most relevant to the conversation; call it again for more."
```

- `query` searches the index and returns the `semantic_top_k` best matches.
- `tool_names` returns exactly those tools (unknown names are reported), at most 32 per call; the reply says how many were left out.
- No arguments returns the next `semantic_top_k` tools most relevant to the conversation that the model was not offered yet. Each further argument-less call pages on to the next `semantic_top_k`. It never enables the whole catalogue: on a server with hundreds of tools that would break the next model call (OpenAI accepts at most 128 tools per request).

One call returns at most 32 tools, and a model call is offered at most 100 of the indexed tools (`preserve_tools` first, then tools already called, then tools `request_more_tools` returned, then the most relevant ones). Tools the index doesn't manage, such as the fallback itself, come on top, so a request stays under OpenAI's 128-tool limit.

Every tool the call returns is offered on the agent's next model call, and stays offered for the rest of that turn and the next one. Nothing is stored on the agent: the selection is recomputed from the `request_more_tools` call in the conversation, so concurrent requests never see each other's tools.

A tool call the model makes for a tool that exists but wasn't offered on that step still runs — selection decides what the model is shown, it is not access control. Use [approval](approval.md) or server-side guards to restrict what may run.

### `preserve_tools`

Tools listed in `preserve_tools` are:

1. Never optimized: the full tool description and every parameter description are kept
2. Offered on every model call by semantic selection, on top of the `semantic_top_k` relevant tools

Use this for critical tools that the agent must always have access to:

```python
optimize_tools=ToolOptimizationConfig(
    level=OptimizationLevel.SEMANTIC,
    preserve_tools={"process_payment", "verify_identity"},
)
```

---

## Combining with Other Features

Tool optimization composes with all other agent features:

```python
agent = await build_agent(
    servers=servers,
    model="openai:gpt-5-mini",
    optimize_tools="standard",
    observe=True,           # observability still tracks all tool calls
    memory=provider,        # memory injection happens before tool selection
    sandbox=True,           # sandbox tools keep full schemas
)
```

Static optimization applies to tools discovered from MCP servers. Semantic selection covers every tool the agent has — MCP, `extra_tools`, sandbox and cross-agent tools — and applies to `ainvoke()`, `chat()`, `astream()` and `astream_with_tools()`.

### Checking selection offline with `ToolIndex`

`ToolIndex` (in `promptise.tool_optimization`) is the index semantic selection uses. Build one over an agent's tools to check, before paying for any model call, whether each of your typical requests would be offered the right tool — and to compare embedding models or description lengths:

```python
from promptise.tool_optimization import ToolIndex, build_selection_query

tools = [t for t in agent.tools if t.name != "request_more_tools"]
index = ToolIndex(tools, model_name_or_path="all-MiniLM-L6-v2")

for request, expected in [("Suspend user u_42, she left.", "suspend_user")]:
    offered = [t.name for t in index.select(request, top_k=8)]
    print(request, "->", "ok" if expected in offered else f"missed (got {offered[:3]})")

# The query the agent would use for a whole conversation:
query = build_selection_query(messages, user_turns=3)
```

`index.select(query, top_k=8, preserve=None)` returns the `top_k` most relevant tools followed by the preserved ones (which don't take a relevance slot). Embeddings of repeated queries are cached.

---

## FAQ

**Does optimization affect tool call quality?**

Schema minification removes per-field descriptions but keeps field names, types, and required status. Most LLMs infer field purpose from well-named parameters (e.g., `user_id`, `email`, `start_date`). For tools with ambiguous parameter names, use `preserve_tools` to exempt them.

**What if semantic selection picks the wrong tools?**

The model can call `request_more_tools`, and the tools it returns are offered on the next model call. To catch misses before they cost anything, check your typical requests with [`ToolIndex`](#checking-selection-offline-with-toolindex). The two most effective fixes are longer tool descriptions in the index (raise `max_description_length` — the semantic preset truncates descriptions to 100 characters, and those truncated descriptions are what gets embedded) and adding must-have tools to `preserve_tools`.

**Does this work with all LLM providers?**

Yes. Tool optimization modifies the tool definitions before they reach LangChain — it works with OpenAI, Anthropic, Ollama, and any other provider.

---

## What's Next?

- [Building Agents](agents/building-agents.md) — the `build_agent()` function and all its parameters
- [Memory](memory.md) — persistent memory with vector search
- [Observability](observability.md) — track token usage and see exactly what the LLM receives
