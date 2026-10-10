---
title: Build a Production AI Agent in Python — 9-step guide with Promptise Foundry
description: Build a production-ready Python AI agent in 9 incremental steps with Promptise Foundry. Covers MCP tool discovery, memory, semantic cache, guardrails, sandboxed code execution, observability, and cross-agent delegation. Each step adds one capability.
keywords: build AI agent Python, AI agent tutorial, production AI agent, MCP agent guide, Python agent example, agentic AI tutorial
---

# Building AI Agents

Build a production-ready AI agent in 9 incremental steps. Each step adds one capability -- from a bare agent that calls tools, to a fully observable agent with memory, sandboxed code execution, and cross-agent delegation.

!!! tip "This is the recommended starting point for building agents"
    This guide walks you through building a complete agent step by step. For deep reference on individual features, see the [Building Agents](../core/agents/building-agents.md), [Memory](../core/memory.md), [Sandbox](../core/sandbox.md), and [Observability](../core/observability.md) pages.

## What You'll Build

An AI agent that connects to MCP servers, discovers tools automatically, remembers context across conversations, executes code safely in a sandbox, observes every action for debugging and compliance, and delegates work to peer agents. One function call creates it. Every capability is opt-in.

## Concepts

Promptise agents are built around three ideas:

1. **MCP-first tool discovery** -- You point the agent at one or more MCP servers. On startup it connects to every server, lists all available tools, and converts them into typed tools automatically. No manual wiring.
2. **Opt-in capabilities** -- Observability, memory, sandbox execution, cross-agent delegation, and prompt flows are all disabled by default. Enable each one with a single parameter.
3. **Model independence** -- Any LLM model string (`"openai:gpt-5-mini"`, `"anthropic:claude-sonnet-4.5"`, `"ollama:llama3"`), any LangChain `BaseChatModel`, or any `Runnable`. Change one string, nothing else moves.

---

## Step 1: Minimal Agent

Start with the simplest possible agent -- a model connected to an MCP server:

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
    Every model provider needs a credential. Put it in a `.env` file in your project — Promptise loads it automatically, in scripts and in the CLI — or pass it in code with `Model(...)` — explained in [**Models & Providers → In code: Model**](../core/agents/models.md#in-code-model): `Model("gpt-4o-mini", provider="openai", api_key="...")`. The **exact variable name for every provider** (Azure AI Foundry, Bedrock, Gemini, Ollama, …) is in [**Configuration & Secrets → Every provider's variables**](../getting-started/configuration.md#every-providers-variables); self-hosted and inference endpoints are in [**Models & Providers → Custom, self-hosted and inference endpoints**](../core/agents/models.md#custom-self-hosted-and-inference-endpoints). `promptise models env <provider>` prints the lines a provider needs and `promptise models check <model>` tells you what is still missing.

`build_agent()` connects to the MCP server, discovers all available tools, converts their schemas, and returns a ready-to-use `PromptiseAgent`. The agent decides which tools to call based on the user's message.

---

## Step 2: Multiple Servers and Instructions

Connect to multiple MCP servers and provide a system prompt:

```python
from promptise.config import HTTPServerSpec, StdioServerSpec

agent = await build_agent(
    servers={
        "weather": HTTPServerSpec(url="http://localhost:8000/mcp"),
        "files": StdioServerSpec(command="python", args=["-m", "file_server"]),
        "database": HTTPServerSpec(
            url="https://db-api.internal/mcp",
            bearer_token="your-jwt-token",
        ),
    },
    model="openai:gpt-5-mini",
    instructions=(
        "You are a data analyst. Use the weather API for forecasts, "
        "the file server for reading reports, and the database for queries. "
        "Always cite your data sources."
    ),
)
```

The agent discovers tools from all three servers and has them available simultaneously. Tool names are automatically namespaced to avoid conflicts.

**Switch models with one line:**

```python
# OpenAI
agent = await build_agent(servers=servers, model="openai:gpt-5-mini")

# Anthropic
agent = await build_agent(servers=servers, model="anthropic:claude-sonnet-4.5")

# Google
agent = await build_agent(servers=servers, model="google_genai:gemini-2.0-flash")

# Local Ollama
agent = await build_agent(servers=servers, model="ollama:llama3")

# Any LangChain model instance
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-5-mini", temperature=0)
agent = await build_agent(servers=servers, model=model)
```

---

## Step 3: Memory

Give your agent persistent memory. Before every invocation, the agent automatically searches for relevant context and injects it into the system prompt.

```python
from promptise.memory import ChromaProvider

# ChromaDB for local persistent vector search
memory = ChromaProvider(
    collection_name="analyst_memory",
    persist_directory=".promptise/chroma",
)

agent = await build_agent(
    servers=servers,
    model="openai:gpt-5-mini",
    memory=memory,
    memory_auto_store=True,  # Automatically store each exchange
)

# First conversation
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "The Q3 report shows 15% revenue growth."}]
})

# Later conversation -- agent remembers Q3 data
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "How did Q3 compare to projections?"}]
})
# Agent automatically retrieves the Q3 context from memory
```

Three providers ship with the framework:

| Provider | Use case | Persistence |
|----------|----------|-------------|
| `InMemoryProvider` | Testing and prototyping | In-process only |
| `ChromaProvider` | Local vector search | Disk (configurable) |
| `Mem0Provider` | Enterprise graph search | External service |

```python
# In-memory (testing)
from promptise.memory import InMemoryProvider
memory = InMemoryProvider()

# ChromaDB (local, persistent)
from promptise.memory import ChromaProvider
memory = ChromaProvider(collection_name="my_agent", persist_directory="./data")

# Mem0 (enterprise)
from promptise.memory import Mem0Provider
memory = Mem0Provider(api_key="...", user_id="analyst-1")
```

---

## Step 4: Observability

Enable full observability with a single flag. Every LLM turn, tool call, token count, latency, retry, and error is captured automatically.

```python
agent = await build_agent(
    servers=servers,
    model="openai:gpt-5-mini",
    observe=True,  # That's it
)

result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Analyze the sales data"}]
})

# Get aggregate statistics
stats = agent.get_stats()
print(f"Total tokens: {stats['total_tokens']}")
print(f"LLM calls: {stats['llm_call_count']}, tool calls: {stats['tool_call_count']}")
print(f"Errors: {stats['error_count']}")

# Generate an interactive HTML report
agent.generate_report("report.html", title="Sales Analysis")
```

For full control, pass an `ObservabilityConfig`:

```python
from promptise.observability_config import ObservabilityConfig, ObserveLevel, TransporterType

agent = await build_agent(
    servers=servers,
    model="openai:gpt-5-mini",
    observe=ObservabilityConfig(
        level=ObserveLevel.FULL,           # OFF, BASIC, STANDARD, FULL
        session_name="production-audit",
        record_prompts=True,
        transporters=[
            TransporterType.HTML,           # Interactive HTML report
            TransporterType.STRUCTURED_LOG, # JSONL file
            TransporterType.CONSOLE,        # Live terminal output
            TransporterType.PROMETHEUS,     # Prometheus /metrics
            TransporterType.OTLP,           # OpenTelemetry traces
            TransporterType.WEBHOOK,        # HTTP POST on events
        ],
        output_dir="./reports",
        log_file="./logs/agent.jsonl",
    ),
)
```

Eight transporters available:

| Transporter | Output | Use case |
|-------------|--------|----------|
| `HTML` | Interactive HTML report | Post-analysis, sharing |
| `JSON` | JSON file | Programmatic analysis |
| `STRUCTURED_LOG` | JSONL file | Log aggregation (ELK, Datadog) |
| `CONSOLE` | Live terminal output | Development debugging |
| `PROMETHEUS` | Prometheus metrics | Infrastructure monitoring |
| `OTEL` | OpenTelemetry spans | Distributed tracing |
| `WEBHOOK` | HTTP POST | Real-time notifications |
| `CALLBACK` | Python callable | Custom processing |

---

## Step 5: Sandboxed Code Execution

When agents write and execute code, run it inside a multi-layer security sandbox:

```python
# Enable with defaults
agent = await build_agent(
    servers=servers,
    model="openai:gpt-5-mini",
    sandbox=True,
)

# Or configure resource limits and network mode
agent = await build_agent(
    servers=servers,
    model="openai:gpt-5-mini",
    sandbox={
        "network_mode": "restricted",  # "none", "restricted", "full"
        "memory_limit": "512M",
        "cpu_limit": 2,
        "timeout": 120,
    },
)
```

When sandbox is enabled, 5 tools are automatically injected into the agent:

| Tool | Description |
|------|-------------|
| `execute_code` | Run Python code in the sandbox |
| `read_file` | Read a file from the sandbox workspace |
| `write_file` | Write a file to the sandbox workspace |
| `list_files` | List files in the sandbox workspace |
| `install_package` | Install a pip package in the sandbox |

Security layers applied automatically:

- **Docker isolation** -- code runs in a container, not on your host
- **Seccomp filtering** -- blocks dangerous syscalls
- **Capability dropping** -- removes ~40 Linux capabilities
- **Read-only rootfs** -- only the workspace directory is writable
- **Resource limits** -- CPU, memory, and time constraints
- **Network isolation** -- configurable per agent (none/restricted/full)
- **Optional gVisor** -- userspace kernel for additional isolation

---

## Step 6: Cross-Agent Delegation

Let agents ask questions to peer agents via HTTP with JWT authentication:

```python
from promptise.cross_agent import CrossAgent

agent = await build_agent(
    servers=servers,
    model="openai:gpt-5-mini",
    cross_agents={
        "researcher": CrossAgent(
            url="http://research-agent:8001",
            jwt_secret="shared-secret",
            description="Expert at finding and summarizing research papers.",
        ),
        "coder": CrossAgent(
            url="http://code-agent:8002",
            jwt_secret="shared-secret",
            description="Expert at writing and reviewing code.",
        ),
    },
)
```

This injects `ask_agent_researcher` and `ask_agent_coder` tools. The agent decides when to delegate:

```
User: "Find recent papers on transformer architectures and write a summary script"

Agent thinks: "I need the researcher for papers and the coder for the script"
→ calls ask_agent_researcher("Find recent papers on transformer architectures")
→ gets research results
→ calls ask_agent_coder("Write a Python script that summarizes these papers: ...")
→ combines results and responds
```

**Broadcast to multiple peers:**

```python
# From code, send a question to all peers simultaneously
results = await agent.broadcast(
    "What is the current status of your subsystem?",
    timeout=30.0,
)
# Returns dict of agent_name → response, with graceful degradation on timeout
```

---

## Step 7: SuperAgent Files

Define an entire agent declaratively in a `.superagent` YAML file:

```yaml
# analyst.superagent
name: data-analyst
model: openai:gpt-5-mini
instructions: |
  You are a senior data analyst. Use available tools to query databases,
  generate visualizations, and produce reports. Always cite data sources.

servers:
  database:
    type: http
    url: http://localhost:8080/mcp
    bearer_token: "${DB_TOKEN}"
  files:
    type: http
    url: http://localhost:8081/mcp

memory:
  provider: chroma
  collection: analyst_memory
  persist_directory: .promptise/chroma
  auto_store: true

observability:
  level: standard
  transporters: [html, structured_log]
  output_dir: ./reports

sandbox:
  enabled: true
  network_mode: restricted
  memory_limit: 1G

cross_agents:
  researcher:
    url: http://research-agent:8001
    jwt_secret: "${CROSS_AGENT_SECRET}"
    description: Expert at finding research papers.
```

Load and run:

```python
from promptise.superagent import load_superagent

agent = await load_superagent("analyst.superagent")
result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "Analyze Q3 revenue trends"}]
})
```

Environment variables resolve automatically with `${VAR}` and `${VAR:-default}` syntax.

---

## Step 8: Conversation Flows

Evolve the system prompt across conversation phases for sophisticated multi-turn agents:

```python
from promptise.prompts import ConversationFlow, Phase

flow = ConversationFlow(
    phases={
        "greeting": Phase(
            blocks=["identity", "greeting_instructions"],
            transitions={"investigation": lambda ctx: ctx.turn > 1},
        ),
        "investigation": Phase(
            blocks=["identity", "investigation_rules", "tool_context"],
            transitions={"resolution": lambda ctx: ctx.state.get("has_diagnosis")},
        ),
        "resolution": Phase(
            blocks=["identity", "resolution_rules", "output_format"],
        ),
    },
    initial_phase="greeting",
)

agent = await build_agent(
    servers=servers,
    model="openai:gpt-5-mini",
    flow=flow,
)
```

The system prompt the agent sees on turn 1 is different from turn 5 and turn 10. Blocks appear and disappear as the conversation progresses. See the [Prompt Engineering guide](prompt-engineering.md) for the full prompt system.

---

## Step 9: Conversation Persistence

Every chat application needs to persist conversations. The conversation store handles loading history, saving new exchanges, and managing sessions -- one parameter:

```python
from promptise import build_agent
from promptise.config import HTTPServerSpec
from promptise.conversations import SQLiteConversationStore, generate_session_id

# Pick a backend: SQLite for dev, Postgres for production, Redis for ephemeral
store = SQLiteConversationStore("conversations.db")

agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={
        "tasks": HTTPServerSpec(url="http://localhost:8080/mcp"),
    },
    conversation_store=store,
    conversation_max_messages=200,  # Rolling window (0 = unlimited)
)

# Use secure session IDs — never user-controlled or predictable
sid = generate_session_id()  # "sess_a1b2c3d4e5f6..."

# chat() handles everything: ownership check → load history → invoke → persist
response = await agent.chat(
    "Create a task to review the PR",
    session_id=sid,
    user_id="user-42",  # Locks this session to user-42
)
response = await agent.chat(
    "What task did I just create?",
    session_id=sid,
    user_id="user-42",  # Same user — allowed
)

# Session management (all operations verify ownership)
sessions = await agent.list_sessions(user_id="user-42")
await agent.update_session(sid, calling_user_id="user-42", title="PR Review")
await agent.delete_session(sid, user_id="user-42")

await agent.shutdown()  # Closes store connections
```

Four built-in stores, or implement the `ConversationStore` protocol for any backend:

| Store | Backend | Use case |
|-------|---------|----------|
| `InMemoryConversationStore` | Dict | Testing |
| `SQLiteConversationStore` | aiosqlite | Local dev |
| `PostgresConversationStore` | asyncpg | Production |
| `RedisConversationStore` | redis.asyncio | Ephemeral sessions |

Conversation persistence works alongside memory (Step 3). Memory provides semantic search across all sessions ("what do I know about this user?"). The conversation store provides exact message replay within a session ("what did they say 3 messages ago?"). See [Conversation Persistence](../core/conversations.md) for the full reference.

---

## Complete Example

A fully-featured agent with MCP tools, memory, observability, sandbox, and cross-agent delegation:

```python
import asyncio
from promptise import build_agent
from promptise.config import HTTPServerSpec
from promptise.memory import ChromaProvider
from promptise.cross_agent import CrossAgent
from promptise.observability_config import ObservabilityConfig, ObserveLevel, TransporterType

async def main():
    # Memory provider
    memory = ChromaProvider(
        collection_name="analyst",
        persist_directory=".promptise/chroma",
    )

    # Build the agent
    agent = await build_agent(
        servers={
            "database": HTTPServerSpec(
                url="http://localhost:8080/mcp",
                bearer_token="your-jwt-token",
            ),
            "files": HTTPServerSpec(url="http://localhost:8081/mcp"),
        },
        model="openai:gpt-5-mini",
        instructions=(
            "You are a senior data analyst. Query databases, analyze data, "
            "write scripts when needed, and produce clear reports. "
            "Always cite your data sources."
        ),
        memory=memory,
        memory_auto_store=True,
        observe=ObservabilityConfig(
            level=ObserveLevel.STANDARD,
            transporters=[TransporterType.HTML, TransporterType.STRUCTURED_LOG],
            output_dir="./reports",
        ),
        sandbox={"network_mode": "restricted", "memory_limit": "1G"},
        cross_agents={
            "researcher": CrossAgent(
                url="http://research-agent:8001",
                jwt_secret="shared-secret",
                description="Expert at finding research papers.",
            ),
        },
    )

    # Run a conversation
    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Analyze Q3 revenue by region"}]
    })
    print(result["messages"][-1].content)

    # Check stats
    stats = agent.get_stats()
    print(f"\nTokens used: {stats['total_tokens']}")
    print(f"Tool calls: {stats['tool_calls']}")

    # Generate report
    agent.generate_report("analysis_report.html")

    await agent.shutdown()

asyncio.run(main())
```

---

## CLI

The Promptise CLI provides quick access to common agent operations:

```bash
# Run a SuperAgent file
promptise agent analyst.superagent

# Validate a SuperAgent file
promptise validate analyst.superagent

# List tools discovered from MCP servers
promptise list-tools --server http://localhost:8080/mcp

# Run an agent interactively
promptise run --model openai:gpt-5-mini --server http://localhost:8080/mcp

# Serve an agent over HTTP
promptise serve analyst.superagent --port 8001
```

---

## Step 10 — Custom Reasoning Patterns

Replace the default ReAct loop with a specialized reasoning pattern:

```python
from promptise.engine import PromptGraph, PromptNode, NodeFlag
from promptise.engine.reasoning_nodes import PlanNode, ThinkNode, SynthesizeNode

agent = await build_agent(
    model="openai:gpt-5-mini",
    servers=my_servers,
    # Use a built-in pattern
    agent_pattern="deliberate",  # Think → Plan → Act → Observe → Reflect
)

# Or build a custom graph with reasoning nodes
graph = PromptGraph("my-agent", nodes=[
    PlanNode("plan", is_entry=True),
    PromptNode("act", inject_tools=True, flags={NodeFlag.RETRYABLE}),
    ThinkNode("think"),
    SynthesizeNode("answer", is_terminal=True),
])

agent = await build_agent(
    model="openai:gpt-5-mini",
    servers=my_servers,
    agent_pattern=graph,
)
```

10 built-in patterns: `react` (default), `verify`, `managed`, `code-action`, `peoatr`, `research`, `autonomous`, `deliberate`, `debate`, `pipeline`. See [Reasoning Patterns](../core/agents/reasoning-patterns.md).

---

## Error Handling & Troubleshooting

### Agent Invocation Errors

```python
try:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Do something complex"}]}
    )
except Exception as exc:
    print(f"Agent failed: {exc}")
    # Common causes: LLM API down, all MCP servers unreachable, timeout
```

### MCP Server Connection Failures

```python
# Check which tools were discovered
print(f"Tools available: {[t.name for t in agent.tools]}")

# If 0 tools — server connection failed. Check:
# 1. Is the server running? (stdio: is the command correct? http: is the URL reachable?)
# 2. Are credentials valid? (bearer_token, api_key)
# 3. Is the server healthy? (check server logs)
```

### Timeout Handling

```python
agent = await build_agent(
    model="openai:gpt-5-mini",
    servers=my_servers,
    max_agent_iterations=25,   # Limit total reasoning steps
    max_invocation_time=120.0, # Hard timeout per invocation, in seconds
)
```

### Guardrail Rejections

```python
from promptise.guardrails import GuardrailViolation

try:
    result = await agent.ainvoke(input)
except GuardrailViolation as exc:
    # The input was blocked before the model was called.
    print(f"Blocked: {exc}")

# With observe=True the blocked run is recorded as an agent.error event:
for event in agent.collector.query(event_types=["agent.error"]):
    print(event.metadata["error_type"], event.metadata["error"])
agent.generate_report("guardrails-report.html")
```

---

## What's Next

**Go deeper on each feature:**

| Feature used in this guide | Deep reference |
|---|---|
| `build_agent()`, `PromptiseAgent` | [Building Agents](../core/agents/building-agents.md) |
| Server specs and connections | [Server Configuration](../core/agents/server-specs.md) |
| Reasoning patterns and custom graphs | [Reasoning Patterns](../core/agents/reasoning-patterns.md) |
| Reasoning Graph engine | [Reasoning Graph](../core/engine.md) |
| Memory providers and auto-injection | [Memory](../core/memory.md) |
| Observability levels and transporters | [Observability](../core/observability.md) |
| Sandbox configuration and security | [Sandbox](../core/sandbox.md) |
| Cross-agent delegation | [Cross-Agent Delegation](../core/agents/cross-agent.md) |
| SuperAgent YAML files | [SuperAgent Files](../core/agents/superagent-files.md) |
| CLI commands | [CLI Reference](../core/cli.md) |

**Other guides:**

- [Building Production MCP Servers](production-mcp-servers.md) -- Build the tool servers your agents connect to
- [Building Agentic Runtime Systems](agentic-runtime.md) -- Make agents autonomous with triggers and governance
- [Prompt Engineering](prompt-engineering.md) -- Build reliable, testable system prompts
