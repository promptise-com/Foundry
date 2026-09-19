---
title: Why Promptise Foundry — Production agent framework for Python
description: Honest, factual comparison of Promptise Foundry against other Python agent frameworks. What ships in the box, what design choices we made, and when each framework is the right fit.
keywords: Promptise Foundry vs LangChain, LangGraph alternative, CrewAI alternative, Python agent framework comparison, production agent framework, MCP framework, agentic AI framework
schema: TechArticle
---

# Why Promptise Foundry?

There are now dozens of Python frameworks for building LLM agents. Promptise Foundry is not the right answer for every team. This page is the honest version — what ships in the box, what design choices we made, and when other frameworks are a better fit.

## What ships in Promptise Foundry

One Python install (`pip install promptise`) gives you:

| Layer | What's included |
|---|---|
| **Core agent** | `build_agent()` factory, model-agnostic, MCP tool discovery, four levels of tool optimization with local embeddings |
| **Memory** | Three providers — in-memory, Chroma (local persistent), Mem0 (graph) — auto-retrieved and injected before every call |
| **Conversation persistence** | Four backends — in-memory, SQLite, Postgres, Redis |
| **Semantic cache** | In-memory or Redis with per-user / per-session / shared scope isolation |
| **Guardrails** | `PromptiseSecurityScanner` with 6 local detection heads (DeBERTa, GLiNER, Llama Guard, regex × 165 patterns, custom) |
| **Sandbox** | Docker with seccomp + capability dropping + read-only rootfs + resource limits + network isolation; optional gVisor |
| **Observability** | 4 levels, 8 transporters (HTML, JSON, log, console, Prometheus, OpenTelemetry, webhook, callback) |
| **MCP server SDK** | Decorators, middleware (logging, timeout, rate limit, circuit breaker, audit), JWT auth, capability guards, caching, health probes, queue, DI, versioning, OpenAPI ingestion, in-process testing |
| **MCPcast** | `promptise mcpcast` turns an API you already have (OpenAPI 3.x / Swagger 2) into a curated, risk-classified, approval-gated MCP server emitted as editable code — see [MCPcast an Existing API](../guides/mcpcast-existing-api.md) |
| **Enterprise MCP** | First-class multi-tenancy (`tenant_id` isolation invariant across cache/memory/limits/audit), server-side human-in-the-loop approval gates (`requires_approval` + four-eyes), tenant-qualified rate limits |
| **MCP client** | Native, no third-party deps. Single, multi-server, and LangChain adapter variants. Three transports (stdio, HTTP, SSE) |
| **Prompt engineering** | 8 prompt blocks, conversation flows, 5 strategies, 4 perspectives, 14 context providers, schema-strict guards, registry with versioning, inspector for debugging |
| **Agent runtime** | Process lifecycle, journals, replay engine, 5 trigger types (cron, event, message, webhook, file watch), 4 governance subsystems (budget, health, mission, secrets), distributed coordinator, dashboard |
| **Cross-agent** | `ask_peer` and `broadcast` over HTTP+JWT for multi-agent systems |
| **Self-modifying** | Open Mode with 14 meta-tools and guardrails for agent-written code |
| **Config** | `.superagent` and `.agent` YAML manifests with `${VAR}` resolution and cycle detection |
| **CLI** | `agent`, `validate`, `list-tools`, `run`, `serve`, `mcpcast` |

Every backend listed as a parameter option works. No `NotImplementedError`. No "planned." If it's documented, it's shipped.

## Design choices

These are the principles we wrote down on day one and have stuck to:

### 1. MCP-first

Tools are discovered, not wired. You give the agent an MCP server URL; the agent calls `tools/list` and uses what's there. This means the same tool you build for Promptise works with Claude Desktop, Cursor, ChatGPT, and any other MCP-compatible agent.

### 2. No silent fallbacks

If the model isn't reachable, we raise. If a tool errors, we raise. If a memory backend isn't configured but is requested, we raise. **Every silent fallback is a future production incident.**

### 3. Async-first

Every public API is async. No sync wrappers that pretend to be async. No mixed sync/async surfaces. `asyncio.run()` at the top, await everywhere else.

### 4. Capability-based security, not RBAC

Roles are blunt. Capabilities are sharp. A user with `tickets:read` can't accidentally get `tickets:delete` because of a role inheritance mistake. The MCP server SDK lets you express exactly what each token is allowed to do.

### 5. Production primitives included, not optional

Audit logs, rate limits, circuit breakers, health probes, sandboxing, observability — these are built into the framework. You don't bolt them on; you turn them on.

### 6. No cost / pricing tracking

LLM provider prices change weekly. We don't shadow-track them. The Budget governance subsystem counts tool calls, LLM turns, and abstract cost units that you map to your own pricing model. `ToolCostAnnotations` let you weight individual tools.

### 7. Local-first when possible

The guardrail models (DeBERTa, GLiNER, Llama Guard) run locally. Embeddings can be local. The vector store can be local. **You can run the full stack air-gapped.**

## When other frameworks are a better fit

We don't think Promptise Foundry is right for every team. Here's where you'd reasonably pick something else:

| If you need… | Consider… |
|---|---|
| Hundreds of pre-built integrations across model providers, document loaders, and vector stores | LangChain |
| A pure stateful-graph orchestration layer with checkpointing for conversational workflows | LangGraph |
| A multi-agent role-playing pattern with minimal code | CrewAI |
| A typed Python-only agent framework with structured output as the default | Pydantic AI |
| A hosted, fully-managed agent platform | A SaaS product, not a framework |

## When Promptise Foundry is the right fit

Pick Promptise Foundry when:

- You're shipping agents to **production**, not prototyping
- You need **multi-user / multi-tenant** access control out of the box
- You want **MCP-native** tool discovery instead of bespoke tool wiring
- You need **autonomous, long-running** agents that survive restarts
- You want **governance** (budget, health, mission, secrets) without building it yourself
- You need **local / air-gapped** deployment as a first-class option
- You want **one coherent stack** instead of five libraries glued together
- You value **type hints, async-first, and explicit errors** over magic and convenience

## A 30-second example

```python
from promptise import build_agent

agent = build_agent(
    model="openai:gpt-5-mini",
    servers=["https://your-mcp-server.example.com"],
    instructions="You are a careful research assistant.",
)

answer = await agent.run("What changed in the API last week?")
```

!!! info "You need an API key first"
    Every model provider needs a credential. Put it in a `.env` file in your project — Promptise loads it automatically, in scripts and in the CLI — or pass it in code with `Model(...)` — explained in [**Models & Providers → In code: Model**](../core/agents/models.md#in-code-model): `Model("gpt-4o-mini", provider="openai", api_key="...")`. The **exact variable name for every provider** (Azure AI Foundry, Bedrock, Gemini, Ollama, …) is in [**Configuration & Secrets → Every provider's variables**](../getting-started/configuration.md#every-providers-variables); self-hosted and inference endpoints are in [**Models & Providers → Custom, self-hosted and inference endpoints**](../core/agents/models.md#custom-self-hosted-and-inference-endpoints). `promptise models env <provider>` prints the lines a provider needs and `promptise models check <model>` tells you what is still missing.

That's the whole API for the simple case. From there:

- Add `memory=ChromaProvider(...)` and the agent remembers across calls
- Add `cache=SemanticCache(...)` and similar queries hit cache
- Add `guardrails=PromptiseSecurityScanner(...)` and prompt injection is filtered
- Add `sandbox=Sandbox(...)` and code execution is isolated
- Wrap in an `AgentProcess` and the agent gets cron triggers, journals, and crash recovery

## Read more

- [Quick Start →](quickstart.md)
- [Key Concepts →](concepts.md)
- [What is MCP? →](what-is-mcp.md)
- [FAQ →](../faq.md)
- [Building Production MCP Servers →](../guides/production-mcp-servers.md)
- [Agent Runtime →](../runtime/index.md)
