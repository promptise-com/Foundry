---
title: Promptise Foundry — Production Python framework for AI agents and MCP servers
description: Promptise Foundry is the production-grade Python framework for building AI agents, MCP servers, autonomous runtimes, and prompt engineering systems. Secure by default, MCP-native, model-agnostic. The modern alternative to LangChain, LangGraph, and CrewAI.
keywords: Python AI agent framework, MCP server framework, agentic AI, autonomous agents, LangChain alternative, LangGraph alternative, CrewAI alternative, Model Context Protocol, prompt engineering, multi-agent systems
schema: SoftwareApplication
---

# Promptise Foundry

The production framework for agentic AI systems. Build agents that discover tools, reason with custom patterns, remember context, enforce security, and run autonomously.

```bash
pip install promptise
```

## Quick Start

```python
import asyncio
from promptise import build_agent

async def main():
    agent = await build_agent(
        model="openai:gpt-4o-mini",
        instructions="You are a helpful assistant.",
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Hello!"}]}
    )
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

!!! info "You need an API key first"
    Put it in a `.env` file in your project (`OPENAI_API_KEY=sk-...`) — Promptise loads it automatically — or pass it in code with `Model(...)` — explained in [**Models & Providers → In code: Model**](core/agents/models.md#in-code-model): `Model("gpt-4o-mini", provider="openai", api_key="...")`. The **exact variable name for every provider** — Azure AI Foundry, Bedrock, Gemini, Ollama, … — is in [**Configuration & Secrets → Every provider's variables**](getting-started/configuration.md#every-providers-variables); self-hosted and inference endpoints are in [**Models & Providers → Custom, self-hosted and inference endpoints**](core/agents/models.md#custom-self-hosted-and-inference-endpoints). `promptise models env <provider>` prints the lines a provider needs.

See the full [Quick Start](getting-started/quickstart.md) for tools, reasoning patterns, and production features.

### Already have an API? MCPcast it

Don't hand-write a server for an API that already exists — generate one from its OpenAPI spec:

```bash
promptise mcpcast openapi.yaml --name myapi --no-curate --auth env-token  # → myapi-mcp/
export MCPCAST_UPSTREAM_TOKEN="Bearer <your API token>"
claude mcp add myapi -- python myapi-mcp/server.py
```

Read-only by default; writes, deletes and money-moving calls are only generated when you opt in, and every one of them is approval-gated **server-side** — so any MCP client can drive your product safely. Drop `--no-curate` to have a model design a smaller intent-tool surface. Start with [**MCPcast, end to end**](mcpcast/index.md) — how MCP works, a real API turned into a real server, and the review checklist — then the [step-by-step guide](guides/mcpcast-existing-api.md) and the [reference](mcp/server/mcpcast.md).

## Five Subsystems

| Subsystem | What it does | Start here |
|-----------|-------------|------------|
| **[Agent](core/index.md)** | Turn any LLM into a production agent. MCP tool discovery, memory, guardrails, semantic cache, streaming, approval workflows. | [Building Agents](guides/building-agents.md) |
| **[Reasoning Engine](core/engine.md)** | Design how your agent thinks. 20 composable nodes, 7 prebuilt patterns, 18 typed flags, 0.02ms overhead. Custom reasoning patterns for any task. | [Custom Reasoning](guides/custom-reasoning.md) |
| **[MCP Server](mcp/index.md)** | Build tool APIs that agents call. JWT auth, guards, middleware, rate limiting, audit logs, TestClient. The FastAPI of MCP. | [Building Servers](guides/production-mcp-servers.md) |
| **[Agent Runtime](runtime/index.md)** | Run agents autonomously. Cron triggers, crash recovery, budget enforcement, health monitoring, mission tracking, distributed coordination. | [Runtime Systems](guides/agentic-runtime.md) |
| **[Prompting](prompting/index.md)** | Prompts as software. Typed blocks with token budgeting, conversation flows, composable strategies, guards, version control, testing. | [Prompt Engineering](guides/prompt-engineering.md) |

## Hands-On Labs

Complete, runnable tutorials for real-world use cases:

- [Lab: Customer Support Agent](guides/lab-customer-support.md) — KB search, conversation phases, escalation, guardrails
- [Lab: Data Analysis Agent](guides/lab-data-analysis.md) — SQL queries, cross-table analysis, specialized reasoning pattern
- [Lab: Code Review Agent](guides/lab-code-review.md) — Adversarial critique, per-node models, security scanning
- [Lab: MCPcast Your SaaS API](guides/lab-mcpcast-storefront.md) — Generated tools, an agent doing real work, a refund held by the approval gate, an Agent Readiness Score

## Install

**Python 3.10+** required.

```bash
pip install promptise
```

Put your LLM key in a `.env` file in your project — scripts and the CLI load it automatically (or pass it in code with `Model(...)`; see [Configuration & Secrets](getting-started/configuration.md)):

```bash title=".env"
OPENAI_API_KEY=sk-...
```

Any other provider — Azure AI Foundry, Bedrock, Gemini, Ollama, … — is one string or one `Model(...)` away; `promptise models env <provider>` prints the lines it needs. Verify:

```bash
promptise models check openai:gpt-5-mini --ping
```

## Next Steps

- [Quick Start](getting-started/quickstart.md) — Build your first agent in 5 minutes
- [Configuration & Secrets](getting-started/configuration.md) — Where API keys live: `.env`, environment, in code, config files
- [Model Setup](getting-started/model-setup.md) — Azure AI Foundry, OpenAI, Bedrock, Gemini, Ollama, or any provider
- [Key Concepts](getting-started/concepts.md) — Architecture and design principles
- [Reasoning Engine](core/engine.md) — The custom execution runtime that powers every agent
