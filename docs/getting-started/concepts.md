---
title: "Key Concepts: How Promptise Foundry works"
seo_title: "Key Concepts: How Promptise Foundry Works"
description: "The core concepts of Promptise Foundry: agents, MCP servers, prompt and context engineering, the agent runtime, design principles and core abstractions."
keywords: Promptise concepts, agentic AI architecture, MCP architecture, AI agent design, agent runtime, prompt engineering concepts
---

# What is Promptise Foundry?

Promptise Foundry is the production framework for building agentic AI systems. It covers the entire stack (from creating intelligent agents to deploying autonomous operations) in one framework.

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

`build_agent()` takes a model, tools and instructions and returns a `PromptiseAgent`. Everything else is opt-in:

| Capability | What it does | Enable with |
|---|---|---|
| **Python tools** | Your functions become tools, schemas from type hints | `extra_tools=[my_tool]` with `@tool` |
| **MCP Tool Discovery** | Agent finds and uses MCP server tools automatically | `servers={"api": HTTPServerSpec(...)}` |
| **Semantic Tool Optimization** | Offers each model call only the relevant tools | `optimize_tools="semantic"` |
| **Memory** | Auto-inject relevant context before every invocation | `memory=ChromaProvider(...)` |
| **Conversations** | Multi-user session persistence (Postgres, SQLite, Redis) | `conversation_store=PostgresConversationStore(...)` |
| **Guardrails** | Block injection attacks, redact PII, detect toxicity | `guardrails=PromptiseSecurityScanner.default()` |
| **Semantic Cache** | Answer repeated, similar questions without a model call | `cache=SemanticCache()` |
| **HITL Approval** | Require human approval for sensitive tool calls | `approval=ApprovalPolicy(tools=["send_*"], handler=...)` |
| **Event Notifications** | Webhook/Slack alerts on errors, budget limits, blocks | `events=EventNotifier(sinks=[...])` |
| **Streaming** | Real-time tool visibility in chat UIs | `agent.astream_with_tools(...)` |
| **Model Fallback** | Automatic failover across LLM providers | `model=FallbackChain(["openai:...", "anthropic:..."])` |
| **Adaptive Strategy** | Learn from failures, adjust approach over time | `adaptive=True` |
| **Context Engine** | Token-budgeted context assembly with exact counting | `context_engine=ContextEngine(model_context_window=128000)` |
| **Cross-Agent Delegation** | Peer agents become `ask_agent_<name>` tools | `cross_agents={"peer": CrossAgent(agent=peer)}` |
| **Reasoning Patterns** | Change how the agent thinks (9 built-in patterns or your own graph) | `agent_pattern="verify"` |
| **Observability** | Track every LLM turn, tool call, and token | `observe=True` |
| **Sandbox** | Execute agent-generated code in isolated Docker containers | `sandbox=True` |

### Pillar 2: MCP Server SDK

Build production MCP servers: the APIs that agents call. Same relationship FastAPI has with REST APIs.

- **`@server.tool()`** decorator with auto-generated schemas from type hints
- **Authentication**: JWT (symmetric + asymmetric), API key, OAuth 2.0
- **Guards**: Per-tool role-based access control (`HasRole`, `RequireAuth`)
- **Middleware**: Rate limiting, circuit breaker, timeout, concurrency, audit logging
- **Job Queue**: Long-running tasks with progress reporting and cancellation
- **Dashboard**: Live terminal UI with tool stats, agents, request log
- **Testing**: In-process `TestClient`: no network required

### Pillar 3: Prompt & Context Engineering

Prompts as software: typed, versioned, tested, debuggable.

- **8 core block types** with priority-based token budgeting (Identity, Rules, OutputFormat, ContextSlot, Section, Examples, Conditional, Composite), plus blocks for tools, plans, observations, reflections and phases
- **ConversationFlow**: the system prompt evolves across conversation phases
- **5 reasoning strategies** that compose: `chain_of_thought + self_critique`
- **14 context providers**: inject tools, memory, user data, environment, tasks and more
- **PromptInspector**: trace every assembly decision step by step
- **YAML loader**: define prompts in `.prompt` files and version them
- **Testing**: `PromptTestCase` with `mock_llm()`, `assert_schema()` and `assert_contains()`

### Pillar 4: Agent Runtime

The operating system for autonomous agents.

- **AgentProcess**: lifecycle container (CREATED → RUNNING → SUSPENDED → STOPPED / FAILED)
- **5 trigger types**: cron schedules, webhooks, file watch, events, messages
- **Crash recovery**: journal-based checkpoint + replay
- **Governance**: autonomy budget, behavioral health monitoring, mission tracking, secret scoping
- **Open Mode**: 14 meta-tools for self-modifying agents (with guardrails)
- **Orchestration API**: a REST API for managing deployed agents without code changes
- **Live Agent Conversation**: send messages to running agents, ask questions, get answers
- **Distributed**: multi-node coordination over HTTP with auth

---

## How They Fit Together

**Simple, chatbot with memory:**
```python
agent = await build_agent(
    servers={"tools": HTTPServerSpec(url="http://localhost:8000/mcp")},
    model="openai:gpt-5-mini",
    memory=ChromaProvider(persist_directory="./memory"),
)
result = await agent.ainvoke({"messages": [{"role": "user", "content": "Hello"}]})
```

!!! info "You need an API key first"
    Every model provider needs a credential. Put it in a `.env` file in your project (Promptise loads it automatically, in scripts and in the CLI) or pass it in code with `Model(...)`, explained in [**Models & Providers → In code: Model**](../core/agents/models.md#in-code-model): `Model("gpt-5-mini", provider="openai", api_key="...")`. The **exact variable name for every provider** (Azure AI Foundry, Bedrock, Gemini, Ollama, …) is in [**Configuration & Secrets → Every provider's variables**](../getting-started/configuration.md#every-providers-variables); self-hosted and inference endpoints are in [**Models & Providers → Custom, self-hosted and inference endpoints**](../core/agents/models.md#custom-self-hosted-and-inference-endpoints). `promptise models env <provider>` prints the lines a provider needs and `promptise models check <model>` tells you what is still missing.

**Production, secured agent with full protections:**
```python
agent = await build_agent(
    servers=servers,
    model=FallbackChain(["openai:gpt-5-mini", "anthropic:claude-sonnet-5-5"]),
    guardrails=PromptiseSecurityScanner.default(),
    cache=SemanticCache(),
    events=EventNotifier(sinks=[WebhookSink(url="https://hooks.slack.com/...")]),
    approval=ApprovalPolicy(tools=["delete_*", "payment_*"], handler=webhook_handler),
    observe=True,
    max_invocation_time=30,
)
```

**Autonomous, runtime with governance:**
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

All tool integration uses the Model Context Protocol. Agents discover tools from MCP servers automatically: no manual wiring. The MCP Server SDK builds the servers. The MCP Client connects agents to them.

### Opt-in everything

Every capability is disabled by default. Enable what you need with one parameter. Disabled features have zero overhead: no extra memory, no extra latency, no extra dependencies.

### Async-first

The APIs are async (`await build_agent(...)`, `await agent.ainvoke(...)`); use `asyncio.run()` at the top level. This enables concurrent tool calls, parallel agent execution, and non-blocking I/O. `agent.invoke()` is a blocking wrapper for scripts that are not async.

### Protocol-based extensibility

Memory providers, conversation stores, cache backends, approval handlers, event sinks: all defined as Python protocols. Implement the interface, plug it in. No base classes, no framework lock-in.

### Security by default

Guardrails scan for injection attacks and redact PII. HMAC-signed webhooks. SSRF-protected URL validation. Per-user cache isolation. Session ownership enforcement. Tamper-evident audit logging. Secrets that zero-fill on revocation.

---

## Core Abstractions

### PromptiseAgent

The unified agent object returned by `build_agent()`:

| Method | What it does |
|---|---|
| `ainvoke(input, caller=...)` | Full pipeline: input guardrails → memory → cache → reasoning graph → output guardrails → cache store |
| `invoke(input)` | The same, blocking |
| `astream(input)` / `astream_with_tools(input)` | Stream graph events, or chat-UI events with tool progress |
| `chat(message, session_id=..., user_id=...)` | Conversation-aware invocation with persistence and session ownership |
| `list_sessions()`, `get_session()`, `update_session()`, `delete_session()` | Manage stored conversations |
| `get_stats()`, `generate_report(path)` | Observability totals and an HTML report (with `observe=`) |
| `tool_names`, `tools` | The tools the agent has |
| `shutdown()` | Close connections and flush transporters |

### ServerSpec

How agents connect to MCP servers:

```python
HTTPServerSpec(url="http://localhost:8000/mcp", bearer_token="...")
StdioServerSpec(command="python", args=["-m", "my_server"])
```

### CallerContext

Per-request user identity, flows through the entire pipeline:

```python
result = await agent.ainvoke(input, caller=CallerContext(user_id="user-42"))
# Used for: cache isolation, conversation ownership, approval attribution,
# event metadata, memory scoping, guardrail logging
```

---

## Environment Variables

`.superagent` files and `.agent` manifests resolve `${VAR}` and `${VAR:-default}`
(every variable Promptise itself reads is in the
[Environment Variables reference](../reference/environment-variables.md)):

```yaml
model: ${MODEL:-openai:gpt-5-mini}
servers:
  api:
    url: ${API_URL}
    bearer_token: ${API_TOKEN}
```

---

## Exceptions you will meet

There is no single base class; catch the specific ones:

| Exception | Base | When |
|---|---|---|
| `ModelSetupError` | `RuntimeError` | The model string or `Model` cannot be used (for example a missing key) |
| `MCPClientError` | `RuntimeError` | An MCP server cannot be reached, or a call to it failed |
| `GuardrailViolation` | `Exception` | Input or output blocked by guardrails |
| `SessionAccessDenied` | `PermissionError` | A user tried to use another user's session |
| `GraphIterationLimitError` | `GraphExecutionError` | A run exceeded `max_agent_iterations` |
| `SuperAgentValidationError` | `SuperAgentError` | Invalid `.superagent` file |
| `ToolError` | MCP `MCPError` | Raise it in a tool to send the model a clear error message |

See [Exceptions](../core/exceptions.md) for the full list.

---

## What's Next?

- [Quick Start](quickstart.md): build your first agent in 5 minutes
- [Model Setup](model-setup.md): configure OpenAI, Anthropic, Ollama, or any provider
- [Building Agents](../guides/building-agents.md): step-by-step from simple to production
- [Building MCP Servers](../guides/production-mcp-servers.md): create tool APIs
- [Building Runtime Systems](../guides/agentic-runtime.md): deploy autonomous agents

!!! tip "Learn the concept"

    - [What Is an Agent Harness? Parts and How to Build One](https://www.promptise.com/guides/view/agent-harness-python): What surrounds the model: tool loop, context compaction, budgets and loop guards.
    - [What Is MCP (Model Context Protocol)? A Developer's Guide](https://www.promptise.com/guides/view/what-is-model-context-protocol): Hosts, clients, servers and transports, the vocabulary behind MCP.
