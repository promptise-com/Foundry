---
title: Key Concepts — How Promptise Foundry works
description: Learn the core concepts of Promptise Foundry — agents, MCP servers, autonomous runtime, prompt engineering, memory, governance, and security. Understand the production agentic stack in one place.
keywords: Promptise concepts, agentic AI architecture, MCP architecture, AI agent design, agent runtime, prompt engineering concepts
---

# What is Promptise Foundry?

Promptise Foundry is the production framework for building agentic AI systems. It covers the entire stack — from creating intelligent agents to deploying autonomous operations — in one framework.

---

## The Four Pillars

Promptise is built on four independent, composable pillars. Use one or all four. Each adds capabilities without requiring changes to the layers below.

```
┌─────────────────────────────────────────────────┐
│  Agent Runtime                                   │
│  Deploy autonomous agents with triggers,         │
│  governance, crash recovery, and orchestration    │
├─────────────────────────────────────────────────┤
│  Prompt & Context Engineering                    │
│  Typed blocks, conversation flows, strategies,   │
│  composable context providers, testing           │
├─────────────────────────────────────────────────┤
│  Agent                                           │
│  build_agent() — one function, production-ready  │
│  Memory, guardrails, cache, events, streaming    │
├─────────────────────────────────────────────────┤
│  MCP Server SDK                                  │
│  Build tool APIs that AI understands             │
│  Auth, middleware, resilience, observability      │
└─────────────────────────────────────────────────┘
```

### Pillar 1: Agent

`build_agent()` takes a model, MCP servers, and instructions — returns a production-ready agent. Everything else is opt-in:

| Capability | What it does | Enable with |
|---|---|---|
| **MCP Tool Discovery** | Agent finds and uses tools automatically | `servers={"api": HTTPServerSpec(...)}` |
| **Semantic Tool Optimization** | Only relevant tools per query (40-70% fewer tokens) | `optimize_tools="semantic"` |
| **Memory** | Auto-inject relevant context before every invocation | `memory=ChromaProvider(...)` |
| **Conversations** | Multi-user session persistence (Postgres, SQLite, Redis) | `conversation_store=PostgresConversationStore(...)` |
| **Guardrails** | Block injection attacks, redact PII, detect toxicity | `guardrails=PromptiseSecurityScanner.default()` |
| **Semantic Cache** | Serve similar queries from cache (30-50% cost savings) | `cache=SemanticCache()` |
| **HITL Approval** | Require human approval for sensitive tool calls | `approval=ApprovalPolicy(tools=["send_*"])` |
| **Event Notifications** | Webhook/Slack alerts on errors, budget limits, blocks | `events=EventNotifier(sinks=[...])` |
| **Streaming** | Real-time tool visibility in chat UIs | `agent.astream_with_tools(...)` |
| **Model Fallback** | Automatic failover across LLM providers | `model=FallbackChain(["openai:...", "anthropic:..."])` |
| **Adaptive Strategy** | Learn from failures, adjust approach over time | `adaptive=True` |
| **Context Engine** | Token-budgeted context assembly with exact counting | `context_engine=ContextEngine(budget=100000)` |
| **Observability** | Track every LLM turn, tool call, and token | `observe=True` |
| **Sandbox** | Execute agent-generated code in isolated Docker containers | `sandbox=True` |

### Pillar 2: MCP Server SDK

Build production MCP servers — the APIs that agents call. Same relationship FastAPI has with REST APIs.

- **`@server.tool()`** decorator with auto-generated schemas from type hints
- **Authentication**: JWT (symmetric + asymmetric), API key, OAuth 2.0
- **Guards**: Per-tool role-based access control (`HasRole`, `RequireAuth`)
- **Middleware**: Rate limiting, circuit breaker, timeout, concurrency, audit logging
- **Job Queue**: Long-running tasks with progress reporting and cancellation
- **Dashboard**: Live terminal UI with tool stats, agents, request log
- **Testing**: In-process `TestClient` — no network required

### Pillar 3: Prompt & Context Engineering

Prompts as software — typed, versioned, tested, debuggable.

- **8 block types** with priority-based token budgeting (Identity, Rules, OutputFormat, ContextSlot, Section, Examples, Conditional, Composite)
- **ConversationFlow** — system prompt evolves across conversation phases
- **5 reasoning strategies** that compose: `chain_of_thought + self_critique`
- **11 context providers** — auto-inject tools, memory, user data, environment, tasks
- **PromptInspector** — trace every assembly decision step-by-step
- **YAML loader** — define prompts in `.prompt` files, version control them
- **Testing** — `mock_llm()`, `assert_schema()`, `assert_contains()`

### Pillar 4: Agent Runtime

The operating system for autonomous agents.

- **AgentProcess** — lifecycle container (CREATED → RUNNING → SUSPENDED → STOPPED / FAILED)
- **5 trigger types**: cron schedules, webhooks, file watch, events, messages
- **Crash recovery** — journal-based checkpoint + replay
- **Governance**: autonomy budget, behavioral health monitoring, mission tracking, secret scoping
- **Open Mode** — 14 meta-tools for self-modifying agents (with guardrails)
- **Orchestration API** — 37 REST endpoints for managing deployed agents without code changes
- **Live Agent Conversation** — send messages to running agents, ask questions, get answers
- **Distributed** — multi-node coordination over HTTP with auth

---

## How They Fit Together

**Simple — chatbot with memory:**
```python
agent = await build_agent(
    servers={"tools": HTTPServerSpec(url="http://localhost:8000/mcp")},
    model="openai:gpt-5-mini",
    memory=ChromaProvider(persist_directory="./memory"),
)
result = await agent.ainvoke({"messages": [{"role": "user", "content": "Hello"}]})
```

!!! info "You need an API key first"
    Every model provider needs a credential. Put it in a `.env` file in your project — Promptise loads it automatically, in scripts and in the CLI — or pass it in code with `Model(...)` — explained in [**Models & Providers → In code: Model**](../core/agents/models.md#in-code-model): `Model("gpt-4o-mini", provider="openai", api_key="...")`. The **exact variable name for every provider** (Azure AI Foundry, Bedrock, Gemini, Ollama, …) is in [**Configuration & Secrets → Every provider's variables**](../getting-started/configuration.md#every-providers-variables); self-hosted and inference endpoints are in [**Models & Providers → Custom, self-hosted and inference endpoints**](../core/agents/models.md#custom-self-hosted-and-inference-endpoints). `promptise models env <provider>` prints the lines a provider needs and `promptise models check <model>` tells you what is still missing.

**Production — secured agent with full protections:**
```python
agent = await build_agent(
    servers=servers,
    model=FallbackChain(["openai:gpt-5-mini", "anthropic:claude-sonnet-4-20250514"]),
    guardrails=PromptiseSecurityScanner.default(),
    cache=SemanticCache(),
    events=EventNotifier(sinks=[WebhookSink(url="https://hooks.slack.com/...")]),
    approval=ApprovalPolicy(tools=["delete_*", "payment_*"], handler=webhook_handler),
    observe=True,
    max_invocation_time=30,
)
```

**Autonomous — runtime with governance:**
```python
async with AgentRuntime() as runtime:
    await runtime.add_process("monitor", ProcessConfig(
        model="openai:gpt-5-mini",
        triggers=[TriggerConfig(type="cron", cron_expression="*/5 * * * *")],
        budget=BudgetConfig(max_tool_calls_per_day=500),
        health=HealthConfig(stuck_threshold=5),
        inbox=InboxConfig(enabled=True),
    ))
    await runtime.start_all()
```

---

## Key Design Principles

### MCP-first

All tool integration uses the Model Context Protocol. Agents discover tools from MCP servers automatically — no manual wiring. The MCP Server SDK builds the servers. The MCP Client connects agents to them.

### Opt-in everything

Every capability is disabled by default. Enable what you need with one parameter. Disabled features have zero overhead — no extra memory, no extra latency, no extra dependencies.

### Async-first

All public APIs are async. Use `asyncio.run()` at the top level. This enables concurrent tool calls, parallel agent execution, and non-blocking I/O.

### Protocol-based extensibility

Memory providers, conversation stores, cache backends, approval handlers, event sinks — all defined as Python protocols. Implement the interface, plug it in. No base classes, no framework lock-in.

### Security by default

Guardrails scan for injection attacks and redact PII. HMAC-signed webhooks. SSRF-protected URL validation. Per-user cache isolation. Session ownership enforcement. Tamper-evident audit logging. Secrets that zero-fill on revocation.

---

## Core Abstractions

### PromptiseAgent

The unified agent object returned by `build_agent()`:

| Method | What it does |
|---|---|
| `ainvoke(input)` | Full pipeline: guardrails → cache → memory → LLM → guardrails → cache |
| `astream_with_tools(input)` | Stream with real-time tool visibility |
| `chat(message, session_id)` | Conversation-aware invocation with persistence |
| `shutdown()` | Close connections and flush transporters |

### ServerSpec

How agents connect to MCP servers:

```python
HTTPServerSpec(url="http://localhost:8000/mcp", bearer_token="...")
StdioServerSpec(command="python", args=["-m", "my_server"])
```

### CallerContext

Per-request user identity — flows through the entire pipeline:

```python
result = await agent.ainvoke(input, caller=CallerContext(user_id="user-42"))
# Used for: cache isolation, conversation ownership, approval attribution,
# event metadata, memory scoping, guardrail logging
```

---

## Environment Variables

Promptise resolves `${VAR}` and `${VAR:-default}` in all configuration:

```yaml
model: ${MODEL:-openai:gpt-5-mini}
servers:
  api:
    url: ${API_URL}
    bearer_token: ${API_TOKEN}
```

---

## Exception Hierarchy

All Promptise exceptions inherit from a common base for easy catching:

| Exception | When |
|---|---|
| `GuardrailViolation` | Input blocked by security guardrails |
| `SessionAccessDenied` | User tried to access another user's session |
| `FeedbackRateLimited` | Too many feedback submissions per hour |
| `MCPClientError` | MCP tool call failed |
| `SuperAgentValidationError` | Invalid `.superagent` YAML configuration |

---

## What's Next?

- [Quick Start](quickstart.md) — build your first agent in 5 minutes
- [Model Setup](model-setup.md) — configure OpenAI, Anthropic, Ollama, or any provider
- [Building Agents](../guides/building-agents.md) — step-by-step from simple to production
- [Building MCP Servers](../guides/production-mcp-servers.md) — create tool APIs
- [Building Runtime Systems](../guides/agentic-runtime.md) — deploy autonomous agents
