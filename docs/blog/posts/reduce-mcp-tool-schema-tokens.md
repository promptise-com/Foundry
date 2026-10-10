---
title: "Cut Tool-Schema Tokens Without Per-Query Selection"
description: "Semantic top-K tool selection is the headline token saver, but it needs per-query embedding and introduces per-request variance -- a non-starter for…"
keywords: "reduce mcp tool schema tokens, tool schema minification, tool optimization levels, static schema optimization air-gapped, minify mcp tool definitions, shrink tool json schema tokens"
date: 2026-07-16
slug: reduce-mcp-tool-schema-tokens
categories:
  - Cost & Efficiency
---

# Cut Tool-Schema Tokens Without Per-Query Selection

The reliable way to **reduce MCP tool schema tokens** — the JSON definitions your agent ships to the model on every single call — is usually not the one teams reach for first. The headline move is semantic top-K selection: embed the query, embed each tool description, and send only the handful of tools that look relevant. It is the biggest single saver, and Promptise Foundry ships it. But it has two properties that rule it out for a lot of real deployments: it needs a per-query embedding pass, and it changes *which tools the model sees from one request to the next*. If your deployment is air-gapped, or your tool set has to be deterministic — every call presents the same tools, auditable and reproducible — per-query selection is off the table. This post is about the other half of `optimize_tools`: the graded static levels that shrink each tool's JSON schema deterministically, cutting tokens without ever changing tool availability.

<!-- more -->

## Where the tokens go: every tool's full schema, every call

When an agent connects to MCP servers, every tool's name, full description, and complete JSON Schema is serialized into the function-calling payload on **every** invocation. Twenty to fifty tools, each with a paragraph of description and a nested parameter schema, easily runs 5,000–15,000 tokens before the conversation even starts. It is the single largest fixed cost after the transcript, and unlike the transcript it does not shrink — you pay the full definition block on turn one and on turn thirty, identically.

Static [tool schema minification](../../core/tool-optimization.md) attacks that fixed cost directly. It does three things, once, at build time, and never touches which tools are available:

- **Schema minification** strips the `description` metadata out of each field's schema. The model still sees field names, types, and required status — it just stops paying for a verbose sentence explaining what `account_id` means when the name already says it.
- **Description truncation** caps each tool-level description at N characters, cut on a word boundary.
- **Depth flattening** replaces deeply nested objects with `dict` beyond a configured depth, so a five-level-deep parameter tree stops serializing its entire subtree.

None of that is per-query. It runs at build, produces the same minified definitions on every call, and needs no embedding model at all.

## What CrewAI, AutoGen, Pydantic AI, and LangGraph ship today

Be precise about the gap, because the honest version is the persuasive one.

- **CrewAI, AutoGen, and Pydantic AI** send the full JSON schema of every registered tool to the model by default. That is the correct, expected behavior — the model needs the schema to call the tool. What none of them ships is a built-in, graded flag that *minifies* that schema: strips per-field descriptions, truncates the tool description, and flattens nested objects. Your levers there are to register fewer tools, or to hand-edit the Pydantic models and descriptions until the serialized payload is smaller. Both are manual and both couple schema size to your source of truth.
- **LangGraph** does offer semantic tool retrieval — through the separate `langgraph-bigtool` add-on. It stores your tools and retrieves the relevant ones per query, which is the same *category* of optimization as Promptise's semantic top layer. But retrieval selects **which** tools to pass; it does not **minify** the JSON schema of the tools it selects. Each retrieved tool still carries its full description and full nested schema into the call. And, like any per-query selection, retrieval is non-deterministic across requests — exactly the property that disqualifies it for air-gapped or reproducible tool sets.

So the accurate delta is not "competitors can't cut tool tokens." It is that reducing per-tool schema footprint — [minify MCP tool definitions](../../core/tool-optimization.md) without changing availability — is a manual, per-framework chore everywhere else, and a **first-class, graded flag** in Promptise. That is the structural difference: `optimize_tools` makes static minification a built-in transform you turn up or down, not something you assemble by editing schemas.

## The four-level ladder on one flag

Promptise exposes [tool optimization levels](../../core/tool-optimization.md) as a single ladder — `optimize_tools` — with four rungs. Three of them are pure static minification with no per-query variance; only the top rung adds semantic selection.

| Level | How you set it | Static minify | Strip nested descriptions | Max schema depth | Per-query selection | Measured savings (90 flat tools) |
|---|---|---|---|---|---|---|
| **NONE** | unset / `False` | — | — | — | — | 0% (baseline) |
| **MINIMAL** | `True` or `"minimal"` | Yes (desc ≤ 200 chars) | No | No limit | No | 14% |
| **STANDARD** | `"standard"` | Yes (desc ≤ 150 chars) | Yes | 3 | No | 14% |
| **SEMANTIC** | `"semantic"` | Yes (desc ≤ 100 chars) | Yes | 2 | Yes (top-K 8) | 90% |

The savings were measured on one server: 90 tools with flat schemas (string parameters, one-line parameter descriptions, short tool descriptions). The static levels save more when your schemas carry long parameter descriptions, long tool descriptions or deep nesting, and less when they don't; STANDARD only pulls ahead of MINIMAL on nested schemas. The point of the table is the middle columns: MINIMAL and STANDARD cut tool-definition tokens **deterministically**. Every call presents the same tools, with the same minified schemas. No embedding runs. Nothing about the request changes what the model can call — only how many tokens each tool's definition costs. SEMANTIC keeps all of that minification *and* adds top-K selection on top, with a `request_more_tools` fallback so the agent can self-recover if the search missed something. It is the option you reach for when per-query variance is acceptable and you want the deepest cut.

## Cut schema tokens deterministically, no per-query variance (runnable)

This is the whole thing. Set `optimize_tools="standard"`, point at any MCP server, set `OPENAI_API_KEY`, and run it. Every tool your server exposes arrives at the model with a minified schema, and the tool set is identical on every call.

```python
import asyncio

from promptise import build_agent
from promptise.config import HTTPServerSpec


async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={"tools": HTTPServerSpec(url="http://localhost:8000/mcp")},
        optimize_tools="standard",  # static minification only — deterministic, no embedding
        instructions="Answer the user's question by calling the available tools.",
    )

    result = await agent.ainvoke({"messages": [
        {"role": "user", "content": "List the open orders for account 4821 and total them."}
    ]})
    print(result["messages"][-1].content)
    await agent.shutdown()


asyncio.run(main())
```

When you want to tune the cut instead of accepting a preset, drop the level into a `ToolOptimizationConfig` and override individual fields. Here STANDARD stays static (no semantic selection, so no embedding model is ever loaded), the description cap is tightened below the preset's 150, and one critical tool is exempted from minification so it always keeps its full schema:

```python
from promptise import build_agent, ToolOptimizationConfig, OptimizationLevel

agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={"tools": HTTPServerSpec(url="http://localhost:8000/mcp")},
    optimize_tools=ToolOptimizationConfig(
        level=OptimizationLevel.STANDARD,     # static-only: no embedding, no per-query variance
        max_description_length=120,           # tighter than the preset's 150-char cap
        preserve_tools={"process_payment"},   # never minified — keeps its full description + schema
    ),
)
```

`preserve_tools` is the escape hatch for tools whose parameters are genuinely ambiguous from their names alone. Everything else gets shrunk; the named tools keep full fidelity. That is [static schema optimization for air-gapped](../../core/tool-optimization.md) and deterministic fleets in one config — the model download that semantic selection needs never happens, because the static levels have zero ML dependency.

## Where schema minification fits your other token levers

Static minification cuts the *tool-definition* block. It composes with the other levers rather than replacing them, and stacking them is where a real deployment gets its numbers down:

- **Semantic selection**, the SEMANTIC rung, is the top layer when per-query variance is acceptable — it minifies *and* sends only the top-K tools. Start static, measure, and only pay the embedding pass and the per-request variance if the deterministic cut is not enough.
- **The transcript.** Minification does nothing for a bloated message history — that is a separate axis, managed by `context_scope` and the deduplicated facts ledger described in the [context lifecycle guide](../../guides/context-lifecycle.md).
- **The response cache.** For repeated or paraphrased queries across runs, the [semantic cache](../../core/cache.md) serves a whole cached answer instead of re-running the agent. If you run multi-tenant, mind the isolation boundary — we walk through exactly how a shared cache stays leak-proof in [Can a Paraphrase Leak Another Tenant's Cached Answer?](semantic-cache-cross-tenant-leak.md).
- **The whole budget.** For the end-to-end picture — tool definitions, memory, and transcript together across a live deployment — see [How to Cut Token Cost for a Multi-Tenant AI Agent](cut-token-cost-multi-tenant-ai-agent.md).

One honest caveat: [shrinking tool JSON schema tokens](../../core/tool-optimization.md) removes per-field *descriptions*, not the field names, types, or required status. Most models infer purpose from a well-named parameter (`user_id`, `start_date`, `email`). For a tool whose parameters really do need prose to be usable, add it to `preserve_tools`. Minification is a cost lever, not a correctness one — it should not change what your agent can do, only what it pays to be told about it.

## Frequently asked questions

### Does static minification change which tools my agent can call?

No. That is the entire reason to prefer it over per-query selection. MINIMAL and STANDARD only shrink each tool's serialized definition — strip field descriptions, truncate the tool description, flatten deep nesting. Every tool your MCP servers expose is still present, on every call, unchanged in availability. Only SEMANTIC filters the tool set, and only when you explicitly opt into it.

### Is this deterministic enough for an air-gapped deployment?

Yes, and it is actually the cleanest air-gapped option in the ladder. The static levels (MINIMAL, STANDARD) run entirely at build time with no embedding model — no `sentence-transformers`, no HuggingFace download, no network call ever. Semantic selection needs a local embedding model (which you can pre-stage on an offline path), but if you want zero ML dependency and byte-identical tool definitions on every request, `optimize_tools="standard"` gives you that.

### How much does STANDARD actually save versus SEMANTIC?

It depends on your schemas. On a 90-tool server with flat schemas, STANDARD trimmed 14% of tool-definition tokens deterministically; SEMANTIC cut 90% by additionally sending only the top-K relevant tools per model call. The extra points from SEMANTIC come at the cost of a per-query embedding pass and a tool set that varies request to request. The recommended path is to set STANDARD, measure your real payload, and only adopt semantic selection if the deterministic cut leaves you short.

### How is this different from LangGraph's langgraph-bigtool?

`langgraph-bigtool` retrieves relevant tools from a store per query — it decides *which* tools to pass. It does not minify the JSON schema of the tools it hands over; each selected tool arrives with its full description and nested schema. Promptise's static levels do the opposite job: they shrink *every* tool's schema without touching availability, and can layer semantic selection on top. The two are complementary ideas; Promptise ships both behind one flag.

## Next steps

If your tool set is large and your deployment is air-gapped or has to stay deterministic, set `optimize_tools="standard"` and measure the drop in your tool-definition tokens before you consider anything per-query. Read the full [tool optimization](../../core/tool-optimization.md) reference for every field and preset, then compose it with the [context lifecycle guide](../../guides/context-lifecycle.md) for transcript bounding and the [semantic cache](../../core/cache.md) for cross-run response reuse. Only once the deterministic cut is in place should you decide whether semantic selection's per-query variance is a trade you want to make.
