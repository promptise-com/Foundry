---
title: Guides — Build production AI agents and MCP servers with Promptise Foundry
description: Practical, end-to-end guides for building real systems with Promptise Foundry — AI agents, production MCP servers, autonomous runtimes, prompt engineering, multi-user systems, and multi-agent coordination. From concept to working code.
keywords: AI agent tutorials, MCP server tutorial, build AI agent guide, agent runtime tutorial, multi-agent Python, multi-user AI system
---

# Guides

Practical, end-to-end guides for building real systems with Promptise Foundry.
Each takes you from concept to working code — architecture decisions,
implementation patterns, and production considerations included.

New here? Follow the **learning path** below in order. Already building? Jump
straight to the guide for the problem you're solving.

## A learning path

```
1. Build an agent  →  2. Build the tools it uses  →  3. Secure & isolate it
        │                        │                            │
   Building AI Agents     Production MCP Servers      Multi-User / Multi-Tenant
        │                 MCPcast an Existing API              │
        │                        │                            │
        └──────────── 4. Scale & automate ───────────────────┘
                   Runtime · Multi-Agent Coordination
```

---

## 1 · Foundations — build and shape an agent

### [Building AI Agents](building-agents.md)
Build a production-ready agent from scratch with MCP tool discovery, persistent
memory, observability, sandboxed code execution, and cross-agent delegation.
One function call creates it; every capability is opt-in.
**You'll learn:** `build_agent()`, model independence, MCP auto-discovery, memory, observability, sandbox, cross-agent delegation, SuperAgent files.

### [Prompt Engineering](prompt-engineering.md)
Build reliable, testable system prompts with typed blocks, token budgeting,
composable reasoning strategies, runtime guards, and dynamic context injection.
**You'll learn:** PromptBlocks, strategies, perspectives, guards, context providers, ConversationFlow, registry, testing.

### [Context Lifecycle Management](context-lifecycle.md)
Keep deep tool loops token-efficient. The default agent handles context
automatically (`context_scope="auto"`); go further with `scoped` and `ledger`
for multi-stage graphs.
**You'll learn:** `context_scope` (`auto`/`full`/`scoped`/`ledger`), the facts-ledger loop, bounding token growth.

### [Code-Action: Agents that Write Programs](code-action.md)
For aggregation and data-traversal, have the model write **one Python program**
over your tools — a single LLM turn instead of dozens of tool calls — run in a
hardened Docker sandbox.
**You'll learn:** `agent_pattern="code-action"`, the sandbox tool-bridge, `max_tool_calls`, when to reach for it.

---

## 2 · Build the tools your agents use

### [Building Production MCP Servers](production-mcp-servers.md)
Build a production-grade MCP server: tool registration, Pydantic validation, JWT
auth with structured client context, scope-based authorization, routers,
middleware, caching, and request tracing.
**You'll learn:** `MCPServer`, tool/resource/prompt decorators, `AuthMiddleware`, `ClientContext`, guards, `on_authenticate`, `MCPRouter`, tracing.

### [MCPcast an Existing API](mcpcast-existing-api.md)
Already have an API? Don't hand-write the server. `promptise mcpcast` turns an
OpenAPI 3.x or Swagger 2 spec into a curated, safe, agent-ready MCP server
emitted as **editable code** — read-only by default, with every write gated by
human approval the server enforces, so any MCP client can drive your product.
**You'll learn:** `promptise mcpcast`, risk classification, safety profiles, LLM curation with code-enforced post-conditions, the plan file, auth modes, server-side approval, Agent Readiness Score.

### [Make Your Python API MCP-Ready](mcpcast-python-app.md)
You have an app, not a spec. Find the OpenAPI URL your framework already serves
(FastAPI, Django Ninja, DRF, Flask, Litestar), rewrite the spec so a model can
read it, generate the MCP server from the running app, map auth, wire up Claude
and Cursor, gate writes behind approval, and ship with a CI guard and a Dockerfile.
**You'll learn:** `operation_id`/`summary`/`Field` descriptions as the tool surface, `env-token` vs `passthrough` vs `api-key` auth, server-side approval, the readiness score as a quality bar.

---

## 3 · Secure, isolate, and govern

### [Building Multi-User Systems](multi-user-systems.md)
End-to-end identity: a user's JWT flows from your backend through the agent to
the MCP server; conversation ownership, per-user cache/memory isolation,
guardrails, and tamper-evident audit are all wired to that identity.
**You'll learn:** `CallerContext`, JWT/OAuth flow, guards, conversation ownership, per-user isolation, audit.

### [CallerContext: Agent → MCP Identity](multi-user-identity.md)
The focused reference for *how* identity crosses the wire — what the bearer
token carries, what the server extracts, and how guards see it.
**You'll learn:** `CallerContext` fields, JWT propagation, server-side extraction, guard evaluation.

### [Secure Multi-Tenant Platform](secure-multi-tenant-platform.md)
The enterprise capstone: one server serving many customer orgs with **provable
tenant isolation**, role-based access, **server-side human approval** for
destructive tools (four-eyes), fair per-tenant usage, and tamper-evident audit.
**You'll learn:** `tenant_id` isolation invariant, `require_tenant`, `RequireTenant`/`HasTenant`, `requires_approval` + `ApprovalGateMiddleware`, `PendingApprover`, tenant-stamped audit.

---

## 4 · Scale and automate

### [Building Agentic Runtime Systems](agentic-runtime.md)
Autonomous, long-running agents that react to events, persist state, recover
from crashes, enforce governance, and scale across machines.
**You'll learn:** `AgentProcess`, triggers, journals, governance (budget/health/mission/secrets), `AgentRuntime`, distributed coordination.

### [Multi-Agent Coordination](multi-agent-teams.md)
Systems where agents collaborate — sharing tools, delegating, communicating
through events, and coordinating through shared state.
**You'll learn:** shared servers with per-agent roles, `ask_peer()`/`broadcast()`, EventBus, shared context, supervisor/pipeline patterns.

---

## Hands-On Labs

Domain-specific, copy-paste-ready tutorials. Each includes a pre-built MCP
server, a specialized reasoning pattern, and runnable code.

- **[Customer Support Agent](lab-customer-support.md)** — issue classification, KB search, policy validation, human escalation. *(Classify → Investigate → Draft → Validate → Respond)*
- **[Data Analysis Agent](lab-data-analysis.md)** — questions → SQL, cross-table joins, accurate reports. *(Plan → Execute → Observe → Verify → Report)*
- **[Code Review Agent](lab-code-review.md)** — security review via adversarial self-critique with line-referenced claims. *(Read → Analyze → Critique → Justify → Synthesize)*
- **[Pipeline Observer Agent](lab-pipeline-observer.md)** — an autonomous runtime agent that watches a pipeline, reacts to events, and escalates.
- **[MCPcast Your SaaS API](lab-mcpcast-storefront.md)** — a storefront API turned into agent tools end to end: risk-classified tools, a real agent completing a business task, a refund held by the server-side approval gate, and an Agent Readiness Score before and after fixing the plan.

---

## Guide Structure

Every guide follows the same progression:

1. **What You'll Build** -- A concrete description of the end result
2. **Concepts** -- The key ideas before any code
3. **Step-by-Step** -- Progressive implementation with working code at each step
4. **Complete Example** -- Full working code you can copy and run
5. **What's Next** -- Links to reference docs for deeper exploration

## Prerequisites

All guides assume:

- Python 3.10+
- `pip install promptise` (or `pip install "promptise[all]"` for all extras)
- An `OPENAI_API_KEY` environment variable set (or another LLM provider)

See [Installation](../index.md) and [Model Setup](../getting-started/model-setup.md) if you need help getting started.
