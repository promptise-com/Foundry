---
title: Cookbook — Recipes for common Promptise Foundry tasks
description: Copy-paste recipes for the most common Promptise Foundry tasks — add memory, cache, guardrails, streaming, per-user identity, multi-tenancy, human approval, rate limits, sandboxed code, and more. Each recipe links to the full guide.
keywords: Promptise recipes, AI agent cookbook, build_agent examples, MCP server recipes, how to add memory AI agent, human approval AI tool
---

# Cookbook

Short, copy-paste recipes for the tasks you'll reach for most. Each is the
minimal correct form; follow the link for the full guide.

Every agent capability is **one parameter** on `build_agent()` — enable only
what you need; the rest has zero overhead.

---

## Agent recipes

### Add persistent memory
Remember context across invocations; relevant past context is auto-injected.
```python
from promptise.memory import ChromaProvider
agent = await build_agent(model="openai:gpt-5-mini", servers=srv,
    memory=ChromaProvider(persist_directory="./memory"))
```

!!! info "You need an API key first"
    Every model provider needs a credential. Put it in a `.env` file in your project — Promptise loads it automatically, in scripts and in the CLI — or pass it in code with `Model(...)` — explained in [**Models & Providers → In code: Model**](../core/agents/models.md#in-code-model): `Model("gpt-4o-mini", provider="openai", api_key="...")`. The **exact variable name for every provider** (Azure AI Foundry, Bedrock, Gemini, Ollama, …) is in [**Configuration & Secrets → Every provider's variables**](../getting-started/configuration.md#every-providers-variables); self-hosted and inference endpoints are in [**Models & Providers → Custom, self-hosted and inference endpoints**](../core/agents/models.md#custom-self-hosted-and-inference-endpoints). `promptise models env <provider>` prints the lines a provider needs and `promptise models check <model>` tells you what is still missing.
→ [Memory](../core/memory.md)

### Persist conversations (with ownership)
`chat()` loads history, invokes, and saves — and enforces session ownership.
```python
from promptise.conversations import SQLiteConversationStore
agent = await build_agent(model="openai:gpt-5-mini", servers=srv,
    conversation_store=SQLiteConversationStore("chat.db"))
reply = await agent.chat("hi", session_id="s1", caller=caller)
```
→ [Conversations](../core/conversations.md)

### Cache similar queries
Serve semantically-similar queries from cache (30–50% cost reduction).
```python
from promptise.cache import SemanticCache
agent = await build_agent(model="openai:gpt-5-mini", servers=srv, cache=SemanticCache())
```
→ [Semantic Cache](../core/cache.md)

### Add security guardrails
Block prompt injection on input; redact PII/credentials on output. Models run locally (needs `promptise[all]`; without the model the guardrails block every message rather than pass it unchecked).
```python
agent = await build_agent(model="openai:gpt-5-mini", servers=srv, guardrails=True)
```
→ [Guardrails](../core/guardrails.md)

### Attach per-request identity
Carry the user through the whole invocation (cache/memory/conversation scope, audit).
```python
from promptise import CallerContext
await agent.ainvoke(input, caller=CallerContext(user_id="alice", roles=["analyst"]))
```
→ [Multi-User Systems](../guides/multi-user-systems.md)

### Isolate by tenant (multi-tenant SaaS)
Add `tenant_id` — every isolation surface (cache, memory, conversations) scopes to it.
```python
caller = CallerContext(user_id="alice", tenant_id="acme")  # disjoint from any other tenant
```
→ [Secure Multi-Tenant Platform](../guides/secure-multi-tenant-platform.md)

### Compute over data with one program
For sums/joins/multi-hop aggregation, the model writes one sandboxed Python program.
```python
agent = await build_agent(model="openai:gpt-5-mini", servers=srv, agent_pattern="code-action")
```
→ [Code-Action](../guides/code-action.md)

### Sandbox code execution
Run agent-written code in a hardened Docker sandbox (seccomp, dropped caps, no network).
```python
agent = await build_agent(model="openai:gpt-5-mini", servers=srv, sandbox=True)
```
→ [Sandbox](../core/sandbox.md)

### Trace everything
Record every LLM turn, tool call, token, and latency.
```python
agent = await build_agent(model="openai:gpt-5-mini", servers=srv, observe=True)
```
→ [Observability](../core/observability.md)

---

## MCP server recipes

### Turn my OpenAPI spec into an MCP server
One command, no model. You get an editable `server.py`, a reviewable tool plan, and a README.
```bash
promptise mcpcast openapi.yaml --name myapi --no-curate   # → myapi-mcp/, read-only by default
```
→ [MCPcast an Existing API](../mcp/server/mcpcast.md)

### Let Claude Desktop use my internal API safely
`env-token` sends one upstream credential; `standard` adds writes, each approval-gated server-side.
```bash
promptise mcpcast openapi.yaml --name myapi --no-curate --profile standard --auth env-token
export MCPCAST_UPSTREAM_TOKEN="Bearer <your API token>"
claude mcp add myapi -- python myapi-mcp/server.py
```
→ [MCPcast guide](../guides/mcpcast-existing-api.md)

### Find out whether an agent can actually drive my tools
A real agent runs generated tasks against the server in-process. Reads may hit the live API; everything else hits spec-derived mocks, so no real data changes.
```bash
promptise mcpcast openapi.yaml --no-curate --eval --eval-tasks 20   # → eval/report.md, graded A–F
```
→ [Agent Readiness Score](../mcp/server/mcpcast.md#agent-readiness-score)

### Define a tool
Schema is generated from type hints.
```python
from promptise.mcp.server import MCPServer
server = MCPServer(name="api")

@server.tool()
async def search(query: str, limit: int = 10) -> list:
    """Search records."""
    return await db.search(query, limit)
```
→ [Server Fundamentals](../mcp/server/building-servers.md)

### Require authentication + a role
```python
from promptise.mcp.server import JWTAuth, AuthMiddleware
server.add_middleware(AuthMiddleware(JWTAuth(secret="...")))

@server.tool(auth=True, roles=["admin"])
async def delete_all() -> str: ...
```
→ [Auth & Security](../mcp/server/auth-security.md)

### Rate-limit a tool
Declared limits are enforced automatically (per client, tenant-qualified).
```python
@server.tool(rate_limit="100/min")
async def expensive() -> dict: ...
```
→ [Caching & Performance](../mcp/server/caching-performance.md)

### Require human approval for a tool
Server-side, for any MCP client. Fail-closed; four-eyes enforced.
```python
from promptise.mcp.server import ApprovalGateMiddleware, PendingApprover
approver = PendingApprover(server, approver_role="approver")
server.add_middleware(ApprovalGateMiddleware(approver, timeout=300))

@server.tool(auth=True, requires_approval=True)
async def refund(order_id: str, amount: float) -> dict: ...
```
→ [Approval Gates](../mcp/server/approval-gates.md)

### Make tenancy a server-wide invariant
Every tool authenticates and must carry a tenant, or the call is denied.
```python
server = MCPServer(name="api", require_tenant=True)
server.add_middleware(AuthMiddleware(JWTAuth(secret="..."), tenant_claim="org"))
```
→ [Multi-Tenancy](../mcp/server/multi-tenancy.md)

### Serve it
```bash
promptise serve myapp:server --transport http --port 8080
```
→ [Deployment](../mcp/server/deployment.md)

### Test it without a network
Full pipeline (validation → guards → middleware → handler) in-process.
```python
from promptise.mcp.server import TestClient
result = await TestClient(server).call_tool("search", {"query": "revenue"})
```
→ [Testing](../mcp/server/testing.md)

---

## See also

- [Quick Start](quickstart.md) — build your first agent in 5 minutes
- [Guides](../guides/index.md) — end-to-end, build-along tutorials
- [Why Promptise](why-promptise.md) — what ships and when to use it
