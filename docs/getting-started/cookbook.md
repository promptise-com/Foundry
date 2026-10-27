---
title: Cookbook, recipes for common Promptise Foundry tasks
seo_title: "AI Agent and MCP Server Recipes"
description: "Copy-paste recipes: @tool functions, MCP servers, streaming, fallback, approval, memory, cache, guardrails and tenancy for agents, plus auth and approval for MCP servers."
keywords: Promptise recipes, AI agent cookbook, build_agent examples, MCP server recipes, how to add memory AI agent, human approval AI tool
---

# Agent and MCP Server Cookbook

Short, copy-paste recipes for the tasks you'll reach for most. Each is the
minimal correct form; follow the link for the full guide.

Every agent capability is **one parameter** on `build_agent()`. Enable only
what you need; the rest costs nothing.

---

## Agent recipes

### Give the agent a Python function as a tool
The schema comes from the type hints and docstring; `ToolError` sends the model a clear message.
```python
from promptise import ToolError, build_agent, tool

@tool
def get_order(order_id: str) -> dict:
    """Look up an order by its ID."""
    if order_id != "A-1001":
        raise ToolError(f"No order {order_id}.")
    return {"order_id": order_id, "status": "shipped"}

agent = await build_agent(model="openai:gpt-5-mini", extra_tools=[get_order])
```
→ [Tools (@tool)](../core/tools.md)

### Connect an MCP server
```python
from promptise.config import HTTPServerSpec, StdioServerSpec
srv = {
    "crm": HTTPServerSpec(url="https://crm.internal/mcp", bearer_token="..."),
    "files": StdioServerSpec(command="python", args=["-m", "file_server"]),
}
agent = await build_agent(model="openai:gpt-5-mini", servers=srv)
```
→ [Server Configuration](../core/agents/server-specs.md)

### Stream the answer and the tool calls
```python
async for event in agent.astream_with_tools({"messages": [{"role": "user", "content": "Status of A-1001?"}]}):
    if event.type == "tool_start":
        print("calling", event.tool_display_name)
    elif event.type == "token":
        print(event.text, end="", flush=True)
```
→ [Streaming](../core/streaming.md)

### Fall back to another model when one fails
Needs only the primary's key (`OPENAI_API_KEY` here): a fallback is set up the first time it is tried, so one without its key (`ANTHROPIC_API_KEY`) is skipped with its setup error, and the chain still builds and answers.
```python
from promptise import FallbackChain
agent = await build_agent(model=FallbackChain(["openai:gpt-5-mini", "anthropic:claude-sonnet-5-5"]), servers=srv)
```
→ [Model Fallback](../core/fallback.md)

### Ask a human before a sensitive tool runs
```python
from promptise import ApprovalDecision, ApprovalPolicy

async def review(request):
    return ApprovalDecision(approved=input(f"Run {request.tool_name}? [y/N] ") == "y")

agent = await build_agent(model="openai:gpt-5-mini", servers=srv,
    approval=ApprovalPolicy(tools=["refund_*", "delete_*"], handler=review))
```
→ [Approval (HITL)](../core/approval.md)

### Add persistent memory
Remember context across invocations; relevant past context is auto-injected.
```python
from promptise.memory import ChromaProvider
agent = await build_agent(model="openai:gpt-5-mini", servers=srv,
    memory=ChromaProvider(persist_directory="./memory"))
```

!!! info "You need an API key first"
    Every model provider needs a credential. Put it in a `.env` file in your project (Promptise loads it automatically, in scripts and in the CLI) or pass it in code with `Model(...)`, explained in [**Models & Providers → In code: Model**](../core/agents/models.md#in-code-model): `Model("gpt-5-mini", provider="openai", api_key="...")`. The **exact variable name for every provider** (Azure AI Foundry, Bedrock, Gemini, Ollama, …) is in [**Configuration & Secrets → Every provider's variables**](../getting-started/configuration.md#every-providers-variables); self-hosted and inference endpoints are in [**Models & Providers → Custom, self-hosted and inference endpoints**](../core/agents/models.md#custom-self-hosted-and-inference-endpoints). `promptise models env <provider>` prints the lines a provider needs and `promptise models check <model>` tells you what is still missing.
→ [Memory](../core/memory.md)

### Persist conversations (with ownership)
`chat()` loads history, invokes, and saves, and enforces session ownership.
```python
from promptise import CallerContext
from promptise.conversations import SQLiteConversationStore
agent = await build_agent(model="openai:gpt-5-mini", servers=srv,
    conversation_store=SQLiteConversationStore("chat.db"))
caller = CallerContext(user_id="alice")  # the session belongs to alice
reply = await agent.chat("hi", session_id="s1", caller=caller)
```
→ [Conversations](../core/conversations.md)

### Cache similar queries
Answer repeated, semantically similar questions from the cache instead of the model. The cache is per user by default, so pass a `CallerContext` with a `user_id`: without one nothing is cached. Answers that called a tool are not cached unless you set `cache_tool_turns=True` (and never after a write tool). Needs `promptise[all]` (local embeddings).
```python
from promptise import CallerContext
from promptise.cache import SemanticCache
agent = await build_agent(model="openai:gpt-5-mini", servers=srv, cache=SemanticCache())
question = {"messages": [{"role": "user", "content": "What are your opening hours?"}]}
await agent.ainvoke(question, caller=CallerContext(user_id="alice"))  # model call, stored
await agent.ainvoke(question, caller=CallerContext(user_id="alice"))  # cache hit
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
request = {"messages": [{"role": "user", "content": "Status of order A-1001?"}]}
await agent.ainvoke(request, caller=CallerContext(user_id="alice", roles={"analyst"}))
```
→ [Multi-User Systems](../guides/multi-user-systems.md)

### Isolate by tenant (multi-tenant SaaS)
Add `tenant_id`: every isolation surface (cache, memory, conversations) scopes to it.
```python
from promptise import CallerContext
caller = CallerContext(user_id="alice", tenant_id="acme")  # disjoint from any other tenant
```
→ [Secure Multi-Tenant Platform](../guides/secure-multi-tenant-platform.md)

### Compute over data with one program
For sums/joins/multi-hop aggregation, the model writes one sandboxed Python program. Needs `pip install "promptise[sandbox]"` and Docker running.
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

- [Quick Start](quickstart.md): build your first agent in 5 minutes
- [Guides](../guides/index.md): end-to-end, build-along tutorials
- [Why Promptise](why-promptise.md): what ships and when to use it
