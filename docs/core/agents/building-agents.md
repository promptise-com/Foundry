# Building Agents

Create intelligent agents that connect to MCP servers, discover tools automatically, and execute tasks with full observability.

## Quick Example

```python
import asyncio
from promptise import build_agent
from promptise.config import HTTPServerSpec

async def main():
    agent = await build_agent(
        servers={
            "weather": HTTPServerSpec(url="http://localhost:8000/mcp"),
        },
        model="openai:gpt-5-mini",
    )

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What is the weather in Zurich?"}]
    })
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

!!! info "You need an API key first"
    Every model provider needs a credential. Put it in a `.env` file in your project — Promptise loads it automatically, in scripts and in the CLI — or pass it in code with `Model(...)` — explained in [**Models & Providers → In code: Model**](../../core/agents/models.md#in-code-model): `Model("gpt-4o-mini", provider="openai", api_key="...")`. The **exact variable name for every provider** (Azure AI Foundry, Bedrock, Gemini, Ollama, …) is in [**Configuration & Secrets → Every provider's variables**](../../getting-started/configuration.md#every-providers-variables); self-hosted and inference endpoints are in [**Models & Providers → Custom, self-hosted and inference endpoints**](../../core/agents/models.md#custom-self-hosted-and-inference-endpoints). `promptise models env <provider>` prints the lines a provider needs and `promptise models check <model>` tells you what is still missing.

## Concepts

Promptise agents are built around three ideas:

1. **MCP-first tool discovery** -- You point the agent at one or more MCP servers via the `servers` dict. On startup it connects to every server, lists all available tools, and converts them into LangChain-compatible tools automatically.
2. **Opt-in capabilities** -- Observability, memory, sandbox execution, cross-agent delegation, and prompt flows are all disabled by default. Enable each one with a single parameter and the agent wires everything together.
3. **Unified return type** -- `build_agent()` always returns a `PromptiseAgent`. It wraps the underlying LangGraph ReAct agent and exposes a consistent interface regardless of which capabilities are active.

## Walkthrough

### The `build_agent()` Function

`build_agent()` is the primary entry point for creating agents. It is an `async` function that connects to MCP servers, discovers tools, and returns a ready-to-use agent.

```python
from promptise import build_agent
from promptise.config import StdioServerSpec, HTTPServerSpec

agent = await build_agent(
    # Required -----------------------------------------------
    servers={
        "files": StdioServerSpec(command="python", args=["-m", "file_server"]),
        "api":   HTTPServerSpec(url="https://api.example.com/mcp"),
    },
    model="openai:gpt-5-mini",

    # Optional -----------------------------------------------
    instructions="You are a helpful data analyst.",
    trace_tools=True,                   # print every tool call to stdout
    observe=True,                       # enable observability
    memory=None,                        # MemoryProvider instance
    memory_auto_store=False,            # auto-persist exchanges
    sandbox=True,                       # sandboxed code execution
    observer=None,                      # shared ObservabilityCollector
    observer_agent_id=None,             # agent id for shared observer
    cross_agents=None,                  # peer agents for delegation
    extra_tools=[],                     # additional BaseTool instances
    flow=None,                          # ConversationFlow for multi-turn prompts
)
```

#### Parameter Reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `servers` | `Mapping[str, ServerSpec]` | **required** | Named MCP server connections. See [Server Configuration](server-specs.md). |
| `model` | `str \| BaseChatModel \| Runnable` | **required** | LangChain model string (e.g. `"openai:gpt-5-mini"`), a chat model instance, or any Runnable. |
| `instructions` | `str \| Prompt \| PromptSuite \| None` | Built-in prompt | System prompt. Accepts a plain string, a `Prompt`, or a `PromptSuite`. |
| `trace_tools` | `bool` | `False` | Print each tool invocation and result to stdout. |
| `observe` | `bool \| ObservabilityConfig \| None` | `None` | Enable observability. Pass `True` for defaults or an `ObservabilityConfig` for full control. |
| `memory` | `MemoryProvider \| dict \| None` | `None` | Memory backend. Automatically searches and injects relevant context before each invocation. |
| `memory_auto_store` | `bool` | `False` | When `True`, automatically stores each user/assistant exchange in memory. |
| `sandbox` | `bool \| dict \| None` | `None` | Enable sandboxed code execution. `True` uses defaults; a dict provides custom config. |
| `observer` | `Any \| None` | `None` | Pass an existing `ObservabilityCollector` to reuse across multiple agents. Mutually exclusive with `observe`. |
| `observer_agent_id` | `str \| None` | `None` | Agent identifier for the shared observer's timeline entries. |
| `cross_agents` | `Mapping[str, CrossAgent] \| None` | `None` | Peer agents exposed as `ask_agent_<name>` tools. See [Cross-Agent Delegation](cross-agent.md). |
| `extra_tools` | `list[BaseTool] \| None` | `None` | Additional LangChain tools appended alongside MCP-discovered tools. |
| `flow` | `ConversationFlow \| type \| Callable \| None` | `None` | A conversation flow that evolves the system prompt across turns: an instance (used as a template), a subclass, or a factory. Each session or caller gets its own copy. See [ConversationFlow](../../prompting/flows.md#integration-with-build_agent). |
| `guardrails` | `PromptiseSecurityScanner \| None` | `None` | Security scanner for input/output. Blocks injection attacks, redacts PII and credentials. See [Guardrails](../guardrails.md). |
| `optimize_tools` | `str \| ToolOptimizationConfig \| None` | `None` | Semantic tool selection to reduce token costs. Pass `"semantic"` for defaults. See [Tool Optimization](../tool-optimization.md). |
| `conversation_store` | `ConversationStore \| None` | `None` | Persistent conversation history. See [Conversations](../conversations.md). |
| `conversation_max_messages` | `int` | `0` | Max messages to retain per session (0 = unlimited). |
| `cache` | `SemanticCache \| None` | `None` | Semantic response cache. Serves similar queries from cache, reducing LLM costs by 30-50%. See [Cache](../cache.md). |
| `approval` | `ApprovalPolicy \| None` | `None` | Human-in-the-loop approval for sensitive tools. See [Approval](../approval.md). |
| `events` | `EventNotifier \| None` | `None` | Webhook/event notifications. Emits structured events on invocation, tool, guardrail, budget, and process events. See [Events](../events.md). |
| `max_invocation_time` | `float` | `0` | Maximum seconds per invocation. Raises `TimeoutError` and emits `invocation.timeout` event when exceeded. `0` = unlimited. |

### The `PromptiseAgent` Class

`build_agent()` returns a `PromptiseAgent`. This is the unified agent object with opt-in capabilities -- disabled features no-op or return sensible defaults, so you never need to check what is active.

#### Invocation Methods

```python
# Async invocation (recommended)
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Summarize sales.csv"}]
})

# Async streaming
async for chunk in agent.astream({
    "messages": [{"role": "user", "content": "Explain quantum computing"}]
}):
    print(chunk)

# Synchronous invocation (convenience wrapper)
result = agent.invoke({
    "messages": [{"role": "user", "content": "Hello"}]
})
```

!!! warning "Sync invocation and memory"
    `invoke()` delegates to `ainvoke()` internally when memory is enabled because memory search requires async I/O. If a running event loop is already active (e.g. inside Jupyter), memory injection is skipped for the sync path. Use `ainvoke()` in async contexts to ensure memory always works.

#### Observability Methods

When `observe=True` is passed to `build_agent()`, the agent records every LLM turn, tool call, and token count.

```python
agent = await build_agent(
    servers=servers, model="openai:gpt-5-mini", observe=True
)

result = await agent.ainvoke({"messages": [{"role": "user", "content": "Hello"}]})

# Retrieve aggregate statistics
stats = agent.get_stats()
print(stats)

# Generate an interactive HTML report
report_path = agent.generate_report("report.html", title="My Agent Report")
```

`get_stats()` returns an empty dict when observability is disabled. `generate_report()` raises `RuntimeError` if observability is not enabled.

#### Programmatic Metrics

When `observe=True`, the agent internally creates a `PromptiseCallbackHandler` that captures all LLM and tool events. Use `get_stats()` to access aggregate metrics:

```python
agent = await build_agent(
    servers=servers, model="openai:gpt-5-mini", observe=True
)

result = await agent.ainvoke({"messages": [{"role": "user", "content": "Hello"}]})

stats = agent.get_stats()
stats["total_tokens"]           # Total tokens used (prompt + completion)
stats["total_prompt_tokens"]    # Total input tokens
stats["total_completion_tokens"] # Total output tokens
stats["llm_call_count"]         # Number of LLM calls
stats["tool_call_count"]        # Number of tool calls
stats["error_count"]            # Number of errors
stats["retry_count"]            # Number of retries
```

See [Observability](../observability.md) for full details on the callback handler, transporter configuration, and HTML report generation.

#### Lifecycle: `shutdown()`

Always call `shutdown()` when the agent is no longer needed. It closes MCP connections, flushes observability transporters, and releases the memory provider.

```python
try:
    result = await agent.ainvoke({"messages": [...]})
finally:
    await agent.shutdown()
```

`shutdown()` is always safe to call -- it no-ops for features that are not enabled.

### Combining Capabilities

Capabilities compose naturally. Here is an agent with observability, memory, and cross-agent delegation all enabled at once:

```python
from promptise import build_agent
from promptise.config import HTTPServerSpec
from promptise.cross_agent import CrossAgent
from promptise.memory import InMemoryProvider

# Build a peer agent first
researcher = await build_agent(
    servers={"search": HTTPServerSpec(url="http://localhost:8001/mcp")},
    model="openai:gpt-5-mini",
)

# Build the main agent with all capabilities
agent = await build_agent(
    servers={"files": HTTPServerSpec(url="http://localhost:8002/mcp")},
    model="openai:gpt-5-mini",
    observe=True,
    memory=InMemoryProvider(),
    memory_auto_store=True,
    cross_agents={
        "researcher": CrossAgent(agent=researcher, description="Web research"),
    },
)

result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Research and summarize recent AI papers"}]
})
await agent.shutdown()
await researcher.shutdown()
```

### Security Guardrails

Protect your agent from prompt injection attacks on input and PII/credential leakage on output. The `guardrails` parameter accepts a `PromptiseSecurityScanner` — a composable scanner built from detection heads.

#### Basic — block injections, redact PII and credentials

```python
from promptise import build_agent, PromptiseSecurityScanner

scanner = PromptiseSecurityScanner.default()
scanner.warmup()  # Pre-load models at startup, not on first message

agent = await build_agent(
    servers=servers, model="openai:gpt-5-mini", guardrails=scanner,
)
```

With this configuration:

- **Input**: Prompt injection attempts are blocked before reaching the LLM. `GuardrailViolation` is raised.
- **Output**: Credit card numbers, SSNs, API keys, and other sensitive data are replaced with labels like `[CREDIT_CARD_VISA]`, `[AWS_ACCESS_KEY]`.

#### Custom — pick specific detectors

```python
from promptise import (
    PromptiseSecurityScanner,
    InjectionDetector, PIIDetector, CredentialDetector, CustomRule,
    PIICategory, CredentialCategory,
)

scanner = PromptiseSecurityScanner(
    detectors=[
        InjectionDetector(threshold=0.9),
        PIIDetector(categories={PIICategory.CREDIT_CARDS, PIICategory.SSN, PIICategory.EMAIL}),
        CredentialDetector(categories={CredentialCategory.AWS, CredentialCategory.OPENAI}),
    ],
    custom_rules=[
        CustomRule(name="internal_id", pattern=r"EMP-\d{6}", description="Employee ID"),
    ],
)

agent = await build_agent(
    servers=servers, model="openai:gpt-5-mini", guardrails=scanner,
)
```

#### Handling blocked input

```python
from promptise.guardrails import GuardrailViolation

try:
    result = await agent.ainvoke({"messages": [{"role": "user", "content": user_input}]})
except GuardrailViolation as v:
    print(f"Blocked: {v.report.blocked[0].description}")
```

See [Guardrails](../guardrails.md) for the full reference — all 6 detector types, 165+ built-in patterns, model configuration, air-gapped deployments, and content safety classification.

---

## CallerContext — Per-Request Identity

Every `ainvoke()` and `chat()` call can carry a `CallerContext` that identifies the user making the request. This is the foundation for multi-user systems.

```python
from promptise.agent import CallerContext

caller = CallerContext(
    user_id="user-alice-001",           # Scopes memory, cache, conversations
    tenant_id="acme",                   # Tenant-qualifies ALL isolation keys (multi-tenant SaaS)
    bearer_token="eyJhbGciOi...",       # Forwarded to MCP servers as Authorization header
    roles={"analyst", "viewer"},         # Agent-side role info
    scopes={"read", "write"},            # Agent-side scope info
    metadata={"team": "finance"},        # Custom metadata
)

result = await agent.ainvoke(input, caller=caller)
```

### What CallerContext Does

| Field | Agent side | MCP server side |
|-------|-----------|-----------------|
| `user_id` | Scopes memory search, conversation history, semantic cache to this user | Not sent (stays on agent) |
| `bearer_token` | Forwarded as `Authorization: Bearer <token>` to every MCP server | Validated by `AuthMiddleware`, extracted into `ClientContext` with roles/scopes/claims |
| `roles` | Available for agent-side logic via `get_current_caller()` | Not sent — server extracts roles from the JWT |
| `scopes` | Available for agent-side logic | Not sent — server extracts scopes from the JWT |
| `metadata` | Custom data available to guardrails, hooks, events | Not sent |

### Identity Flow: Agent → MCP Server

```
CallerContext(bearer_token="eyJ...")
    → MCPClient sets Authorization header
        → HTTP request to MCP server
            → AuthMiddleware validates JWT
                → Extracts roles, scopes, claims
                → Builds ClientContext
                    → Guards check HasRole, HasScope
                        → Handler receives ctx.client
```

Only the `bearer_token` crosses the wire. Everything else is extracted server-side from the JWT payload.

### Per-User Isolation

When `CallerContext` is passed, these features automatically isolate per user:

```python
agent = await build_agent(
    ...,
    memory=ChromaProvider(...),          # Memory scoped to caller.user_id
    conversation_store=SQLiteStore(...), # Sessions owned by caller.user_id
    cache=SemanticCache(),               # Cache keyed by caller.user_id
    observe=True,                        # Traces tagged with caller.user_id
)

# Alice and Bob have completely separate state
await agent.chat("question", session_id="s1", caller=alice_ctx)
await agent.chat("question", session_id="s2", caller=bob_ctx)
```

### Accessing CallerContext in Custom Code

```python
from promptise.agent import get_current_caller

# Inside guardrails, hooks, or custom tools:
caller = get_current_caller()
if caller:
    print(caller.user_id)    # "user-alice-001"
    print(caller.roles)      # {"analyst", "viewer"}
```

See [CallerContext: Agent to MCP Identity](../../guides/multi-user-identity.md) for the complete end-to-end guide with JWT structure, guard reference, and server-side handler examples.

---

## From Agent to Production: Governance & Runtime

`build_agent()` creates a request-response agent — you call it, it replies. That's perfect for chatbots and synchronous workflows. For **autonomous agents** that run continuously, respond to triggers, and need safety rails, you wrap the agent in an **AgentProcess** managed by the **AgentRuntime**.

The runtime adds three capability layers on top of any agent you build with `build_agent()`:

### Triggers — how the agent wakes up

Instead of calling `ainvoke()` manually, you configure triggers that fire automatically:

- **Cron** — run on a schedule (every 5 minutes, hourly, daily at midnight)
- **Webhook** — listen for HTTP POSTs from monitoring systems, CI/CD pipelines, or APIs
- **File Watch** — react to filesystem changes
- **Event** — subscribe to an in-process EventBus shared with other agents
- **Message** — receive topic-based messages from a MessageBroker

Multiple triggers can fire on the same agent. See [Triggers](../../runtime/triggers/index.md).

### Governance — guardrails against runaway behavior

Four opt-in subsystems, each disabled by default with zero overhead:

| Subsystem | What it prevents | Docs |
|-----------|------------------|------|
| **Autonomy Budget** | Tool-call runaway loops, unexpected costs, unbounded irreversible actions | [Autonomy Budget](../../runtime/governance/budget.md) |
| **Behavioral Health** | Stuck agents, repeating loops, empty responses, elevated error rates | [Behavioral Health](../../runtime/governance/health.md) |
| **Mission Model** | Long-horizon trajectory drift from the agent's objective | [Mission Model](../../runtime/governance/mission.md) |
| **Secret Scoping** | Credential leakage, secrets in logs, keys stored as plaintext in memory | [Secret Scoping](../../runtime/governance/secrets.md) |

```python
from promptise.runtime import (
    AgentRuntime, ProcessConfig, TriggerConfig,
    BudgetConfig, HealthConfig, MissionConfig,
)

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Monitor the data pipeline.",
    servers={"tools": {"type": "http", "url": "http://localhost:8000/mcp"}},
    triggers=[
        TriggerConfig(type="webhook", webhook_path="/alerts", webhook_port=9090),
    ],

    # Budget — prevent runaway tool calls
    budget=BudgetConfig(
        enabled=True,
        max_tool_calls_per_run=20,
        max_cost_per_day=50.0,
        on_exceeded="pause",
    ),

    # Health — catch stuck agents
    health=HealthConfig(
        enabled=True,
        stuck_threshold=3,
        on_anomaly="escalate",
    ),

    # Mission — evaluate progress every 10 invocations
    mission=MissionConfig(
        enabled=True,
        objective="Keep all services above 99.9% uptime.",
        success_criteria="No unresolved P1 incidents for >15 minutes.",
        eval_every=10,
    ),
)

async with AgentRuntime() as runtime:
    await runtime.add_process("pipeline-observer", config)
    await runtime.start_all()
    # Process runs autonomously until stopped.
```

### Persistence — journal, crash recovery, audit trail

The **FileJournal** records every state transition, trigger event, invocation, and tool call to an append-only JSONL file on disk. On crash, the **ReplayEngine** reconstructs the last known state so the agent can pick up where it left off.

The journal is also your compliance audit trail. Every decision the agent made is timestamped, structured, and greppable.

### When to use the runtime

| If you're building... | Use |
|----------------------|-----|
| A chatbot or Q&A interface | `build_agent()` alone + `chat()` with `conversation_store` |
| A one-shot automation script | `build_agent()` alone + `ainvoke()` |
| A background worker that wakes on a schedule | `AgentProcess` with cron trigger |
| An incident responder that reacts to webhooks | `AgentProcess` with webhook trigger |
| A file processor that reacts to new files | `AgentProcess` with file_watch trigger |
| A multi-agent pipeline | `AgentRuntime` with multiple processes + event triggers |
| Any agent that spends real money autonomously | Always add `BudgetConfig` with `on_exceeded="pause"` |

See the full [Agent Runtime](../../runtime/index.md) docs for architecture, state machine, and deployment patterns.

---

## API Summary

| Symbol | Import | Description |
|---|---|---|
| `build_agent()` | `from promptise import build_agent` | Async factory that discovers tools and returns a `PromptiseAgent`. |
| `PromptiseAgent` | `from promptise.agent import PromptiseAgent` | Unified agent with `ainvoke()`, `astream()`, `invoke()`, `shutdown()`, `get_stats()`, `generate_report()`. |
| `ServerSpec` | `from promptise.config import ServerSpec` | Union type: `StdioServerSpec \| HTTPServerSpec`. |
| `CrossAgent` | `from promptise.cross_agent import CrossAgent` | Peer agent wrapper with `agent` and `description` fields. |
| `CallerContext` | `from promptise.agent import CallerContext` | Per-request identity: user_id, bearer_token, roles, scopes, metadata. |
| `get_current_caller()` | `from promptise.agent import get_current_caller` | Get the CallerContext for the current invocation (from contextvar). |
| `PromptiseSecurityScanner` | `from promptise import PromptiseSecurityScanner` | Composable security scanner for input/output guardrails. |

!!! tip "Model strings"
    Promptise uses LangChain's `init_chat_model()` under the hood, so any provider string it accepts works here: `"openai:gpt-5-mini"`, `"anthropic:claude-sonnet-4-20250514"`, `"ollama:llama3"`, etc.

!!! tip "Empty server dict"
    Passing an empty `servers={}` is valid. The agent runs without MCP tools -- useful when you only need `extra_tools` or `cross_agents`.

!!! warning "Always await shutdown"
    Forgetting `await agent.shutdown()` can leak open connections to MCP servers. Use a `try/finally` block or wrap the agent lifecycle in an async context manager pattern.

## What's Next?

**Core agent capabilities**

- [Server Configuration](server-specs.md) -- `StdioServerSpec` and `HTTPServerSpec` in detail.
- [SuperAgent Files](superagent-files.md) -- define agents declaratively in YAML.
- [Cross-Agent Delegation](cross-agent.md) -- multi-agent collaboration.
- [Guardrails](../guardrails.md) -- injection blocking, PII redaction, content safety, custom rules.
- [Memory](../memory.md) -- persistent memory with vector search.
- [Conversations](../conversations.md) -- multi-user session persistence.
- [Tool Optimization](../tool-optimization.md) -- semantic tool selection for token savings.
- [Observability](../observability.md) -- track token usage and agent behavior.

**Autonomous agents (Runtime)**

- [Agent Runtime](../../runtime/index.md) -- architecture and concepts.
- [Agent Processes](../../runtime/processes.md) -- `ProcessConfig` reference, lifecycle methods.
- [Triggers](../../runtime/triggers/index.md) -- cron, webhook, event, file watch, message.
- [Autonomy Budget](../../runtime/governance/budget.md) -- limit tool calls, cost, irreversible actions.
- [Behavioral Health](../../runtime/governance/health.md) -- detect stuck loops and anomalies.
- [Mission Model](../../runtime/governance/mission.md) -- mission-oriented execution.
- [Journal and Recovery](../../runtime/journal/index.md) -- crash recovery via replay.
