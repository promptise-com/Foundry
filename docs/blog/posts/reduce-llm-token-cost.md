---
title: "Cut LLM Token Costs with Semantic Tool Selection"
description: "The biggest hidden token cost usually isn't the conversation — it's 20-50 tool schemas re-sent on every single call. This shows semantic top-K tool selection…"
keywords: "reduce LLM token cost, tool selection to cut tokens, reduce agent token usage, MCP tool token cost, fewer tokens per LLM call"
date: 2026-07-16
slug: reduce-llm-token-cost
categories:
  - Memory & RAG
---

# Cut LLM Token Costs with Semantic Tool Selection

If you want to reduce LLM token cost on an agent that calls tools, the first place to look is almost never the conversation — it's the tool schemas you re-send on every single request. Connect an agent to a few MCP servers and you can easily be shipping 20–50 tool definitions, each with a name, description, and full JSON Schema, on every turn — thousands of tokens the model reads before it even sees the user's question. This post shows why that cost is hidden, and how per-query semantic tool selection trims it, with a `request_more_tools` fallback so the agent self-recovers when selection misses. By the end you'll have a runnable agent that only pays for the tools each query actually needs.

<!-- more -->

## Your biggest hidden token cost is tool schemas, not chat history

Most cost-optimization advice fixates on conversation history: summarize old turns, cap the buffer, use a cheaper model. That's real, but it misses the line item that scales with your *integration surface* rather than your chat length.

Every tool an agent can call is serialized into the prompt on every invocation via function-calling. The model needs the full name, description, and parameter schema to decide whether and how to call it. With 20–50+ tools connected, that's commonly 5,000–15,000 tokens of tool definitions per call — sent whether the user asked to refund an order or just said "hi."

Three things make this the quietest cost in an agent stack:

- **It's constant per call, not per session.** A one-line question pays the same tool tax as a twenty-turn debug session.
- **It grows every time you mount a server.** Adding a CRM integration or a billing MCP server silently raises the floor on every request across the whole agent.
- **It's invisible in your app code.** You never wrote the schema payload — the framework assembled it — so it doesn't show up where you'd think to look.

That's the MCP tool token cost, and it's exactly the kind of overhead that tool selection is built to remove.

## How tool selection cuts tokens per LLM call

The insight behind tool selection to cut tokens is simple: a given query rarely needs more than a handful of your tools. A refund request needs the billing tools; it does not need the calendar, the search index, or the analytics exporter. So instead of sending all 50 definitions every time, send only the ones relevant to *this* query.

Promptise Foundry's tool optimization does this in two layers, described in full on the [Tool Optimization guide](../../core/tool-optimization.md):

- **Static optimization** runs once at build time and shrinks each tool's footprint without changing which tools are available — schema minification (drop verbose per-field descriptions), description truncation, and depth flattening for deeply nested objects.
- **Semantic selection** runs per invocation and is the bigger win. At build time every tool description is embedded with a lightweight local model. Before each call, the user's query is embedded and compared against those descriptions, and only the top-K most relevant tools are included.

Static optimization makes each tool cheaper; semantic selection sends fewer tools. Together they're how you get fewer tokens per LLM call without dropping capabilities.

## Turn on SEMANTIC tool optimization

Here's the part the brief is really about: the **SEMANTIC** optimization level. One string flips it on. The agent connects to three MCP servers, but any single query only pays for the tools that match it.

```python
import asyncio
from promptise import build_agent, CallerContext
from promptise.config import HTTPServerSpec

async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={
            "crm":     HTTPServerSpec(url="http://localhost:8001/mcp"),
            "billing": HTTPServerSpec(url="http://localhost:8002/mcp"),
            "search":  HTTPServerSpec(url="http://localhost:8003/mcp"),
        },
        instructions="You are a support agent. Use tools to resolve tickets.",
        optimize_tools="semantic",   # per-query top-K selection + request_more_tools fallback
    )

    # Only the billing/CRM tools relevant to *this* query are embedded in the call —
    # not the search index or anything else the servers expose.
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Refund order 8891 and email the customer."}]},
        caller=CallerContext(user_id="alice", roles=["support"]),
    )
    print(result["messages"][-1].content)

    await agent.shutdown()

asyncio.run(main())
```

Semantic tool optimization reports **40–70% fewer tokens** on tool definitions, because a query that touches three tools no longer drags fifty schemas along with it. The embedding runs locally with `all-MiniLM-L6-v2` (384 dimensions, no API key), so selection itself adds no per-token API cost — the model downloads once, then runs offline.

The one flag maps to a preset. If you want to see the layers without behavioral change first, `optimize_tools="minimal"` and `optimize_tools="standard"` apply progressively deeper static optimization while keeping all tools available; `optimize_tools=True` is shorthand for the minimal preset. Move to `"semantic"` when you're ready for per-query filtering.

## Make top-K safe: request_more_tools, preserved tools, and local embeddings

The obvious objection to top-K selection is: what if the embedding search misses the one tool the query actually needed? Promptise handles that so the agent is never stuck.

**The `request_more_tools` fallback.** Whenever semantic selection is active, one extra tool is always included:

```
Tool: request_more_tools(query?: str, tool_names?: list[str])
Description: "Call this when none of your current tools can do what is needed.
             Describe the capability in `query` (or give exact `tool_names`);
             the matching tools are returned and you can call them on your
             next step. Without arguments it returns the next few tools most
             relevant to the conversation; call it again for more."
```

If the top-K set was wrong, the model calls `request_more_tools` with a description of what it needs (or pages through the next most relevant tools), gets the matching tools, and retries with the right one. It never unlocks the whole catalogue at once, so a large server can't push a request past a provider's tool limit. You trade one cheap recovery round-trip for the large, constant savings on every well-matched call — and the agent self-heals instead of failing.

**Never gamble on your critical tools.** For tools that must always be present regardless of the similarity score — payment, identity verification, anything irreversible — pin them with `preserve_tools`. Pinned tools skip optimization entirely and are always selected. You can also tune `semantic_top_k` and point `embedding_model` at a local directory for air-gapped deployments:

```python
from promptise import build_agent, ToolOptimizationConfig, OptimizationLevel

agent = await build_agent(
    model="openai:gpt-5-mini",
    servers=servers,
    optimize_tools=ToolOptimizationConfig(
        level=OptimizationLevel.SEMANTIC,
        semantic_top_k=8,
        preserve_tools={"process_payment", "verify_identity"},
        embedding_model="/models/all-MiniLM-L6-v2",  # local path → zero network calls
    ),
)
```

That combination — top-K plus a fallback plus preserved-tool pinning — is what lets you reduce agent token usage aggressively without praying the search is perfect on every query.

## Where the savings compound: cache and context budget

Tool selection is one lever. It composes cleanly with the other two token levers, and reaching for all three is how the numbers really move.

- **Semantic cache.** When a new query is close enough to one you already answered, serve the stored response and skip the model call entirely — reported at **30–50% cost reduction** on repetitive workloads. Because it's scoped per user by default and re-scans cached output through guardrails, it saves money without leaking answers across tenants. See the [Semantic Cache guide](../../core/cache.md).
- **Context budget.** Even with fewer tools, a long chat can overflow a small window. The [Context Engine](../../core/context-engine.md) counts tokens exactly, assigns a priority to every layer, and trims from the bottom up — conversation history first, never the tool definitions or the user's current question. It's what stops the silent-truncation bug where a long session quietly drops the newest turn.

Tool selection, the cache, and the budget attack different token costs — the tool tax, recomputed answers, and window overflow. For how these fit alongside conversation history and long-term recall, the hub post [AI Agent Memory: The Complete Guide for Python Devs](ai-agent-memory.md) maps the whole stack.

## When lighter optimization is the better fit

Semantic selection is not free, and it isn't always the right call. Be honest about your setup before you turn it on:

- **You have only a handful of tools.** With five or six tools, the definitions are already cheap and top-K selection adds embedding overhead and a small risk of a `request_more_tools` detour for no meaningful saving. Static `"minimal"` or `"standard"` optimization is the better fit — you shave per-tool bytes with zero behavioral change.
- **Every tool is plausibly relevant to every query.** If your agent is a tight, single-domain workflow where the model genuinely may need any tool at any step, filtering fights the model. Keep all tools and lean on the cache and context budget instead.
- **You can't run a local embedding model.** Semantic selection depends on `sentence-transformers` running locally. In an environment where you truly cannot ship those weights, stick to static optimization, which needs no model.

The rule of thumb: reach for `"semantic"` when you have many tools across several servers and any given query uses a small slice of them. That's precisely where the 40–70% figure comes from — and precisely where static-only optimization leaves money on the table.

## Frequently asked questions

### Does semantic tool selection change my agent's answers?

It changes which tool *definitions* the model sees per call, not the tools' behavior. When the top-K set is right — which top-K=8 covers for most workloads — the answer is identical to sending all tools, just cheaper. When it's wrong, the `request_more_tools` fallback lets the agent search the catalog for the missing tool and retry, so correctness is preserved at the cost of one extra round-trip.

### How much can I actually save on token cost?

Promptise reports **40–70% fewer tokens** on tool definitions with semantic selection, because a query that needs three tools stops paying for all fifty. Your exact figure depends on how many tools you connect and how narrowly each query uses them — the more tools and the smaller each slice, the larger the win. Layer the [semantic cache](../../core/cache.md) on top for another 30–50% on repeated queries.

### Do I need an API key or network access for the embeddings?

No. Selection embeds locally with `all-MiniLM-L6-v2` by default — it downloads once, then runs fully offline with no per-token API charge. For air-gapped deployments, download the model on a connected machine and point `embedding_model` at the local directory, and selection makes zero network calls.

## Next steps

Turn on `optimize_tools=True`, wire up a local embedding model, and watch your per-call tool tokens drop immediately — then move to `"semantic"` once you're connecting many tools across servers. Start from the [Quick Start](../../getting-started/quickstart.md), then read the [Tool Optimization guide](../../core/tool-optimization.md) for the full config reference, preset table, and `preserve_tools` patterns.
