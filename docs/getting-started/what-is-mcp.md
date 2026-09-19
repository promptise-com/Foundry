---
title: What is MCP? — Model Context Protocol explained for Python developers
description: A practical guide to the Model Context Protocol (MCP) — what it is, how it works, why agent frameworks use it, and how Promptise Foundry implements both client and server. Covers transports, tools, resources, prompts, sampling, and elicitation.
keywords: Model Context Protocol, MCP, MCP server, MCP client, MCP Python, what is MCP, MCP tutorial, MCP framework, agentic AI tools
schema: TechArticle
---

# What is the Model Context Protocol (MCP)?

The **Model Context Protocol** (MCP) is the open standard for connecting Large Language Models to **tools**, **resources**, and **prompts** in a uniform way. It was introduced in late 2024 and is now supported by Claude, ChatGPT, Cursor, and a growing ecosystem of agent frameworks.

If you've built agentic systems before, you know the pain: every tool integration is bespoke. Every framework re-invents tool wiring. MCP solves that. **Build a tool once, and any MCP-compatible agent can call it.**

## Why MCP exists

Before MCP, the agent ecosystem had four problems:

1. **Bespoke tool wiring per framework** — LangChain tools, CrewAI tools, OpenAI function specs, Anthropic tool blocks. Each had its own format.
2. **No transport standard** — some frameworks shipped HTTP, others stdio, others process-spawning. Hard to mix and match.
3. **No security primitives** — auth, rate limits, audit trails were left to the developer.
4. **No interactive primitives** — agents that need to ask the user for input mid-execution had no standard way to do it.

MCP standardizes all of this. One protocol. One transport contract. One schema format. One auth pattern.

## The four MCP primitives

| Primitive | What it is | Example |
|---|---|---|
| **Tool** | A function the LLM can call | `search_database(query)`, `send_email(to, body)` |
| **Resource** | A read-only piece of context | A document, a database row, an image |
| **Prompt** | A reusable, parameterized prompt template | `summarize_in_style(style="executive")` |
| **Sampling** | The server asks the client's LLM for a completion | A subagent calls back to the orchestrator's model |

Plus two interactive primitives added in later spec revisions:

- **Elicitation** — the server requests structured input from the user mid-execution
- **Roots** — the server tells the client which filesystem paths it has access to

## How it works

An MCP **client** (your agent) and an MCP **server** (your tool provider) speak JSON-RPC over one of three transports:

- **stdio** — server runs as a subprocess; messages flow over stdin/stdout
- **streamable HTTP** — single HTTP endpoint with bidirectional streaming
- **SSE** — server-sent events for one-way streaming responses

The handshake:

1. Client connects, sends `initialize`
2. Server replies with capabilities (tools, resources, prompts)
3. Client calls `tools/list` to discover what's available
4. Client calls `tools/call` with arguments
5. Server returns the result

That's it. The same flow works whether the server is a 50-line Python script or a 50,000-line enterprise gateway.

## Why Promptise Foundry is MCP-native

Promptise Foundry is built on MCP from day one. **Tools are discovered, not wired.**

```python
from promptise import build_agent

agent = build_agent(
    model="openai:gpt-5-mini",
    servers=[
        "https://your-mcp-server.example.com",
        "stdio:./local-server.py",
    ],
)

answer = await agent.run("What's in the customer database?")
```

!!! info "You need an API key first"
    Every model provider needs a credential. Put it in a `.env` file in your project — Promptise loads it automatically, in scripts and in the CLI — or pass it in code with `Model(...)` — explained in [**Models & Providers → In code: Model**](../core/agents/models.md#in-code-model): `Model("gpt-4o-mini", provider="openai", api_key="...")`. The **exact variable name for every provider** (Azure AI Foundry, Bedrock, Gemini, Ollama, …) is in [**Configuration & Secrets → Every provider's variables**](../getting-started/configuration.md#every-providers-variables); self-hosted and inference endpoints are in [**Models & Providers → Custom, self-hosted and inference endpoints**](../core/agents/models.md#custom-self-hosted-and-inference-endpoints). `promptise models env <provider>` prints the lines a provider needs and `promptise models check <model>` tells you what is still missing.

The agent connects to both servers, calls `tools/list`, and starts using the tools. No manual schema translation, no per-tool registration, no boilerplate.

## Promptise Foundry's MCP capabilities

### As a client

- **`MCPClient`** — single-server connection
- **`MCPMultiClient`** — N servers with unified tool list and auto-routing
- **`MCPToolAdapter`** — converts MCP tools to LangChain `BaseTool` for interop
- All three transports: stdio, HTTP, SSE
- Bearer token + API key authentication
- **No third-party MCP client dependencies** — built from scratch

### As a server

The MCP server SDK has the same relationship to MCP that FastAPI has to REST:

```python
from promptise.mcp.server import MCPServer

server = MCPServer("my-tools")

@server.tool()
async def search_database(query: str) -> list[dict]:
    """Search the customer database."""
    return await db.search(query)
```

Type hints become the JSON schema. Docstrings become the description. Decorators handle the protocol plumbing.

And when the API already exists, you don't hand-write the server at all. `promptise mcpcast` reads its OpenAPI 3.x or Swagger 2 spec, classifies every operation's risk, curates a tool surface an agent can pick from correctly, and emits an editable MCP server — read-only unless you opt in, with writes, deletes and money-moving calls gated by human approval the server enforces. See [MCPcast an Existing API](../guides/mcpcast-existing-api.md).

Production features built in:

| Feature | What it does |
|---|---|
| **Authentication** | JWTAuth (HS256), AsymmetricJWTAuth (RS256/ES256), APIKeyAuth |
| **Authorization** | Per-tool guards: `HasRole`, `HasAllRoles`, `RequireAuth`, `RequireClientId` |
| **Middleware** | Logging, timeout, rate limit, circuit breaker, concurrency, audit |
| **Caching** | In-memory or Redis, `@cached` decorator, cache middleware |
| **Health checks** | Kubernetes-native liveness, readiness, startup probes |
| **Metrics** | Per-tool counts, latencies, error rates; Prometheus + OpenTelemetry |
| **Audit logging** | HMAC-chained entries for tamper detection, JSONL output |
| **Job queue** | Priority scheduling, retry with exponential backoff, cancellation |
| **DI** | FastAPI-style `Depends()` for per-request dependencies |
| **Versioning** | Multiple tool versions coexist (`search` vs `search@1.0`) |
| **Composition** | Mount sub-servers with namespace isolation |
| **OpenAPI** | Auto-generate MCP tools from OpenAPI specs |
| **Testing** | `TestClient` runs the full pipeline in-process |

## When to use MCP (and when not to)

**Use MCP when:**

- You want your tools to work with multiple LLM providers and agent frameworks
- You're building a multi-tenant tool service
- You want to compose tools from internal teams or third-party providers
- You need a standard place to put auth, rate limits, and audit
- You want your tools to outlive any specific agent framework

**Skip MCP when:**

- You have one agent calling one private function — direct invocation is simpler
- You're prototyping and the JSON-RPC overhead doesn't pay off yet

For everything else: MCP gives you a separation of concerns the LLM ecosystem has been missing.

## Learn more

- [Build your first MCP server](../guides/production-mcp-servers.md)
- [MCPcast an API you already have](../guides/mcpcast-existing-api.md)
- [MCP client deep dive](../mcp/client/index.md)
- [MCP server reference](../mcp/index.md)
- [Official MCP spec](https://modelcontextprotocol.io)

---

**Next:** [Build your first agent in 5 minutes →](quickstart.md)
