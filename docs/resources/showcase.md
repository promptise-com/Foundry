# What you can build

Promptise Foundry is not a chatbot wrapper. It is the infrastructure for building AI systems that do real work — systems that discover tools, remember context, guard against attacks, recover from crashes, and operate autonomously. This page exists to show you the full surface area of what is possible. Use it as a jumping-off point, not a ceiling.

---

## How to read this page

Each of the five modules gets its own section with build ideas at three complexity levels: **quick build** (30 minutes), **real-world build** (a day), and **ambitious build** (a week). Every idea includes what the module specifically contributes, a scenario, and a working code sketch using the actual Promptise API.

---

## Agent

With the Agent module, you give your application the ability to understand natural language, discover and call tools automatically, remember past interactions, and protect itself from misuse — all through a single function call. The agent is the bridge between a language model and the real world.

### What the Agent module makes possible

Before Promptise, connecting an LLM to tools meant manual schema definitions, custom API adapters, and fragile integration code. The Agent module eliminates that: point it at MCP servers, and the agent discovers every available tool, understands their parameters, and starts using them. Add memory, guardrails, caching, and streaming with one parameter each.

### Build ideas

---

#### 1. Internal knowledge base assistant with guardrails

**The scenario**
Your company has an internal wiki, a Confluence space, and a customer database. New employees spend hours searching for answers. You want an assistant that finds information across all three sources — but never leaks customer PII in its responses.

**What the Agent module contributes**
The agent auto-discovers tools from three MCP servers (wiki, Confluence, CRM), searches all of them on each question, and the guardrails scanner automatically redacts credit card numbers, SSNs, and emails from every response before it reaches the user.

**How it works**
The agent receives a question, searches all three knowledge bases via MCP tools, synthesizes an answer from the results, and passes the response through output guardrails that redact any PII patterns before returning.

```python
import asyncio
from promptise import build_agent, PromptiseSecurityScanner
from promptise.config import HTTPServerSpec
from promptise.memory import ChromaProvider

async def main():
    scanner = PromptiseSecurityScanner.default()
    scanner.warmup()

    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={
            "wiki": HTTPServerSpec(url="http://wiki-mcp:8001/mcp"),
            "confluence": HTTPServerSpec(url="http://confluence-mcp:8002/mcp"),
            "crm": HTTPServerSpec(url="http://crm-mcp:8003/mcp", bearer_token="..."),
        },
        instructions="You are an internal knowledge assistant. Answer questions using available tools.",
        memory=ChromaProvider(persist_directory="./assistant_memory"),
        guardrails=scanner,
        observe=True,
    )

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What's our refund policy for enterprise customers?"}]
    })
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

**What to explore next**
- Add `cache=SemanticCache()` to avoid repeated LLM calls for the same questions
- Add `conversation_store=SQLiteConversationStore("./chat.db")` for persistent chat sessions
- Add `optimize_tools="semantic"` to reduce token costs when connecting to many servers

---

#### 2. Multi-user customer support agent with session persistence

**The scenario**
You are building a SaaS product with in-app chat support. Each customer has their own conversation history that persists across sessions. A customer who started a conversation yesterday should pick up where they left off today. Different customers must never see each other's data.

**What the Agent module contributes**
The `chat()` method handles everything: load history from PostgreSQL, invoke the agent with full context, persist the new messages, and enforce session ownership so User A cannot access User B's conversations.

```python
from promptise import build_agent, CallerContext
from promptise.conversations import PostgresConversationStore
from promptise.config import HTTPServerSpec

agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={"support": HTTPServerSpec(url="http://support-tools:8000/mcp")},
    instructions="You are a customer support agent for Acme Corp.",
    conversation_store=PostgresConversationStore(dsn="postgresql://..."),
    conversation_max_messages=50,
    guardrails=PromptiseSecurityScanner.default(),
)

# Each API request carries the user's identity
response = await agent.chat(
    "I want to upgrade my plan",
    session_id="sess_abc123",
    user_id="user-42",
)
```

**What to explore next**
- Add `events=EventNotifier(sinks=[WebhookSink(url="https://slack.com/...")])` to alert your team on errors
- Add `approval=ApprovalPolicy(tools=["change_plan", "process_refund"])` for human review on sensitive actions

---

#### 3. Cost-optimized agent with semantic caching and token reduction

**The scenario**
Your agent handles 10,000 queries per day across 40 MCP tools. At $3 per million tokens, the monthly bill is growing. You need to cut costs without degrading quality.

**What the Agent module contributes**
Semantic tool optimization sends only the 5-8 relevant tools per query instead of all 40 (40-70% token savings). Semantic caching serves identical or similar queries from cache (30-50% additional savings). Combined: up to 80% cost reduction.

```python
from promptise import build_agent, SemanticCache, CallerContext

agent = await build_agent(
    model="openai:gpt-5-mini",
    servers=servers,  # 40 tools across multiple servers
    optimize_tools="semantic",  # Only relevant tools per query
    cache=SemanticCache(
        similarity_threshold=0.92,
        default_ttl=3600,
        scope="per_user",
    ),
)

# First call — full LLM invocation
result = await agent.ainvoke(input, caller=CallerContext(user_id="user-42"))

# Similar question 5 minutes later — served from cache, zero LLM cost
result = await agent.ainvoke(similar_input, caller=CallerContext(user_id="user-42"))

stats = await agent._cache.stats()
print(f"Cache hit rate: {stats.hit_rate:.0%}")  # ~35% after a few hours
```

---

#### 4. Streaming agent for real-time chat UIs

**The scenario**
Your chat interface shows "..." for 10 seconds while the agent thinks. Users think it is broken. You want them to see "Searching database...", "Found 3 results", then the answer appearing token by token.

**What the Agent module contributes**
`astream_with_tools()` yields structured events — tool starts, tool results, tokens, and completion — so your frontend can show exactly what the agent is doing in real time.

```python
from promptise import build_agent
from promptise.streaming import ToolStartEvent, ToolEndEvent, TokenEvent, DoneEvent

agent = await build_agent(model="openai:gpt-5-mini", servers=servers)

async for event in agent.astream_with_tools(
    {"messages": [{"role": "user", "content": "Show me last month's revenue"}]}
):
    if isinstance(event, ToolStartEvent):
        print(f"🔧 {event.tool_display_name}...")
    elif isinstance(event, ToolEndEvent):
        print(f"   → {event.tool_summary} ({event.duration_ms:.0f}ms)")
    elif isinstance(event, TokenEvent):
        print(event.text, end="", flush=True)
    elif isinstance(event, DoneEvent):
        print(f"\n\n✅ Done in {event.duration_ms:.0f}ms, {len(event.tool_calls)} tool calls")
```

### Agent module: quick reference for builders

| What you want to do | How to do it |
|---------------------|--------------|
| Connect to MCP tools | `servers={"name": HTTPServerSpec(url="...")}` |
| Add persistent memory | `memory=ChromaProvider(persist_directory="./mem")` |
| Block injection attacks | `guardrails=PromptiseSecurityScanner.default()` |
| Cache similar queries | `cache=SemanticCache()` |
| Persist conversations | `conversation_store=PostgresConversationStore(dsn="...")` |
| Require human approval | `approval=ApprovalPolicy(tools=["delete_*"])` |
| Get webhook alerts | `events=EventNotifier(sinks=[WebhookSink(url="...")])` |
| Stream with tool visibility | `agent.astream_with_tools(input)` |
| Reduce token costs | `optimize_tools="semantic"` |
| Auto-failover models | `model=FallbackChain(["openai:...", "anthropic:..."])` |

---

## MCP Server

With the MCP Server SDK, you build the tool APIs that agents call. Every `@server.tool()` becomes a capability the agent can discover and use. Authentication, rate limiting, audit logging, and circuit breakers come built in — so your tool server is production-ready from the first line.

### What the MCP Server module makes possible

MCP is how AI agents discover and call tools. Without a production server SDK, exposing your business logic to agents means building custom HTTP endpoints, writing JSON schemas by hand, and hoping agents call them correctly. The MCP Server SDK automates all of that: type hints become schemas, decorators become tools, and middleware handles everything agents should never worry about.

### Build ideas

---

#### 1. Database query tool with role-based access

**The scenario**
Your data team wants agents to query the analytics database directly. But not all agents should have the same access — the reporting agent can read, the admin agent can write, and no agent should run `DROP TABLE`.

```python
from promptise.mcp.server import MCPServer, JWTAuth, HasRole, AuthMiddleware

server = MCPServer(name="analytics-db", auth=JWTAuth(secret="${JWT_SECRET}"))
server.add_middleware(AuthMiddleware())

@server.tool(guards=[HasRole("analyst")])
async def query_analytics(sql: str, limit: int = 100) -> list[dict]:
    """Run a read-only SQL query against the analytics database."""
    if any(kw in sql.upper() for kw in ["DROP", "DELETE", "UPDATE", "INSERT"]):
        raise ValueError("Only SELECT queries are allowed")
    return await db.fetch(f"{sql} LIMIT {limit}")

@server.tool(guards=[HasRole("admin")])
async def run_migration(migration_name: str) -> str:
    """Apply a named database migration."""
    return await migrations.apply(migration_name)
```

---

#### 2. Payment processing server with audit logging

**The scenario**
Your agent needs to process Stripe payments. Every charge must be logged with tamper-evident audit records for compliance. Rate limiting prevents runaway agents from charging thousands of times.

```python
from promptise.mcp.server import (
    MCPServer, JWTAuth, AuthMiddleware, AuditMiddleware,
    RateLimitMiddleware, TimeoutMiddleware,
)

server = MCPServer(name="payments", auth=JWTAuth(secret="${JWT_SECRET}"))
server.add_middleware(AuthMiddleware())
server.add_middleware(AuditMiddleware(secret="${AUDIT_SECRET}"))  # HMAC-chained entries
server.add_middleware(RateLimitMiddleware(requests_per_minute=10))
server.add_middleware(TimeoutMiddleware(default_timeout=30.0))

@server.tool()
async def charge_customer(customer_id: str, amount_cents: int, description: str) -> dict:
    """Charge a customer's default payment method via Stripe."""
    return await stripe.charges.create(
        customer=customer_id, amount=amount_cents,
        currency="usd", description=description,
    )
```

---

#### 3. Multi-API gateway with circuit breakers

**The scenario**
Your agent integrates with 5 external APIs (weather, maps, flights, hotels, restaurants). When one API goes down, the agent should fail fast for that tool instead of waiting 30 seconds.

```python
from promptise.mcp.server import MCPServer, MCPRouter, CircuitBreakerMiddleware

server = MCPServer(name="travel-gateway")
server.add_middleware(CircuitBreakerMiddleware(
    failure_threshold=3, recovery_timeout=60,
))

weather = MCPRouter(prefix="weather")
flights = MCPRouter(prefix="flights")

@weather.tool()
async def get_forecast(city: str, days: int = 5) -> dict:
    """Get weather forecast for a city."""
    return await weather_api.forecast(city, days)

@flights.tool()
async def search_flights(origin: str, destination: str, date: str) -> list[dict]:
    """Search available flights."""
    return await flight_api.search(origin, destination, date)

server.mount(weather)
server.mount(flights)
```

---

#### 4. REST API bridge — turn any OpenAPI spec into MCP tools

**The scenario**
You have an existing REST API with an OpenAPI spec. You want to expose it to agents without rewriting anything.

```python
from promptise.mcp.server import MCPServer, OpenAPIProvider

server = MCPServer(name="api-bridge")

# Auto-generate MCP tools from your existing OpenAPI spec
provider = OpenAPIProvider(
    "https://api.yourcompany.com/openapi.json",
    prefix="api_",  # Tools named: api_getUser, api_listOrders, etc.
    auth_header=("Authorization", "Bearer ${API_TOKEN}"),
)
provider.register(server)
# Every endpoint in the spec is now an MCP tool agents can discover and call
```

### MCP Server: quick reference for builders

| What you want to do | How to do it |
|---------------------|--------------|
| Define a tool | `@server.tool()` |
| Add JWT auth | `MCPServer(auth=JWTAuth(secret="..."))` |
| Role-based access per tool | `@server.tool(guards=[HasRole("admin")])` |
| Rate limit | `server.add_middleware(RateLimitMiddleware(...))` |
| Circuit breaker | `server.add_middleware(CircuitBreakerMiddleware(...))` |
| Audit logging | `server.add_middleware(AuditMiddleware(secret="..."))` |
| Group tools by namespace | `MCPRouter(prefix="billing")` |
| Test without network | `TestClient(server)` |
| Import from OpenAPI spec | `OpenAPIProvider("https://...").register(server)` |

---

## Agent Runtime

With the Agent Runtime, you deploy agents as persistent processes — they wake on schedules, react to events, recover from crashes, enforce their own budgets, and accept messages from humans while running. This is not another LLM wrapper with a loop. It is an operating system for autonomous AI.

### What the Agent Runtime makes possible

Without a runtime, agents exist only for the duration of a single request. The Runtime turns them into persistent processes with state machines, trigger systems, crash recovery, and governance. Deploy an agent that monitors your infrastructure every 5 minutes, escalates when it detects anomalies, stays within budget, and lets your team talk to it without stopping it.

### Build ideas

---

#### 1. Infrastructure monitor that escalates to Slack

**The scenario**
Your SRE team wants an agent that checks infrastructure health every 5 minutes. When it detects an anomaly, it should investigate, and if it cannot resolve it, post to Slack with its findings.

```python
import asyncio
from promptise.runtime import AgentRuntime, ProcessConfig, TriggerConfig, BudgetConfig, HealthConfig
from promptise import EventNotifier, WebhookSink

notifier = EventNotifier(sinks=[
    WebhookSink(url="https://hooks.slack.com/services/...", events=["health.anomaly", "invocation.error"]),
])

async with AgentRuntime(event_notifier=notifier) as runtime:
    await runtime.add_process("infra-monitor", ProcessConfig(
        model="openai:gpt-5-mini",
        instructions="You monitor infrastructure. Check health, investigate anomalies, escalate if needed.",
        servers={"infra": {"url": "http://infra-tools:8000/mcp", "transport": "streamable-http"}},
        triggers=[TriggerConfig(type="cron", cron_expression="*/5 * * * *")],
        budget=BudgetConfig(enabled=True, max_tool_calls_per_day=500, on_exceeded="pause"),
        health=HealthConfig(enabled=True, stuck_threshold=5, on_anomaly="escalate"),
    ))
    await runtime.start_all()
    await asyncio.sleep(86400)  # Run for 24 hours
```

---

#### 2. Mission-driven data migration agent

**The scenario**
You need to migrate 50 database tables to a new schema. The agent works through them one by one, evaluating its own progress every 3 tables, and stops automatically when the migration is complete.

```python
from promptise.runtime import ProcessConfig, TriggerConfig, MissionConfig

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Migrate all database tables to v2 schema. Work through each table systematically.",
    servers={"db": {"url": "http://db-tools:8000/mcp"}},
    triggers=[TriggerConfig(type="cron", cron_expression="*/10 * * * *")],
    mission=MissionConfig(
        enabled=True,
        objective="Migrate all 50 database tables to v2 schema",
        success_criteria="All tables pass v2 validation with zero errors",
        eval_every=3,              # LLM-as-judge every 3 invocations
        confidence_threshold=0.7,
        timeout_hours=24,
        auto_complete=True,        # Stop when mission succeeds
    ),
)
```

---

#### 3. Agent fleet managed via REST API

**The scenario**
You deploy 10 agents across your organization. Operations team needs to start, stop, update instructions, and talk to agents without touching code — all through a REST API.

```python
from promptise.runtime import AgentRuntime
from promptise.runtime.api import OrchestrationAPI

runtime = AgentRuntime()
# ... add processes ...

api = OrchestrationAPI(
    runtime, host="0.0.0.0", port=9100,
    auth_token="${ADMIN_TOKEN}",  # All endpoints require auth
)
await api.start()

# Now ops team can manage agents via HTTP:
# POST /api/v1/processes — deploy a new agent
# PATCH /api/v1/processes/monitor/instructions — update live instructions
# POST /api/v1/processes/monitor/messages — send a message to the running agent
# POST /api/v1/processes/monitor/ask — ask a question and get an answer
# GET /api/v1/processes/monitor/inbox — see pending messages
```

---

#### 4. Human-in-the-loop autonomous agent

**The scenario**
Your agent processes support tickets autonomously but must get human approval before sending emails or issuing refunds. The team communicates with the agent via its inbox without stopping it.

```python
from promptise.runtime import ProcessConfig, TriggerConfig
from promptise.runtime.config import InboxConfig
from promptise import ApprovalPolicy, WebhookApprovalHandler

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Process support tickets. Investigate, draft responses, escalate when unsure.",
    triggers=[TriggerConfig(type="webhook", webhook_path="/tickets", webhook_port=9090)],
    inbox=InboxConfig(enabled=True, max_messages=50, default_ttl=3600),
    approval=ApprovalPolicy(
        tools=["send_email", "issue_refund"],
        handler=WebhookApprovalHandler(url="https://approval-service.internal/review"),
        timeout=300,
    ),
)
# The team can send messages: "Focus on enterprise tickets first"
# The agent checks its inbox every invocation cycle and adapts
```

### Agent Runtime: quick reference for builders

| What you want to do | How to do it |
|---------------------|--------------|
| Schedule an agent | `TriggerConfig(type="cron", cron_expression="*/5 * * * *")` |
| React to webhooks | `TriggerConfig(type="webhook", webhook_path="/events")` |
| Watch for file changes | `TriggerConfig(type="file_watch", watch_path="/data")` |
| Set a daily budget | `BudgetConfig(max_tool_calls_per_day=500)` |
| Detect stuck agents | `HealthConfig(stuck_threshold=5)` |
| Define a mission | `MissionConfig(objective="...", success_criteria="...")` |
| Talk to a running agent | `await process.send_message("Focus on X")` |
| Ask an agent a question | `response = await process.ask("What have you found?")` |
| Manage via REST API | `OrchestrationAPI(runtime, port=9100)` |
| Deploy from YAML | `.agent` manifest files |

---

## Prompt engineering

With the Prompt Engineering module, you treat prompts as software — typed, versioned, tested, and debuggable. Instead of tweaking strings in comments, you compose prompts from independent blocks, evolve them across conversation phases, and inspect every assembly decision.

### Build ideas

---

#### 1. Multi-phase customer intake flow

**The scenario**
A customer onboarding chatbot needs different behavior at each stage: friendly greeting first, then structured data collection, then confirmation. Each phase uses different rules, different examples, and a different tone.

```python
from promptise.prompts import prompt, ConversationFlow, Phase
from promptise.prompts.blocks import Identity, Rules, OutputFormat, Examples

onboarding = ConversationFlow(phases=[
    Phase(
        name="greeting",
        blocks=[Identity("Friendly onboarding assistant"), Rules(["Be warm and welcoming"])],
        initial=True,
    ),
    Phase(
        name="collection",
        blocks=[
            Identity("Structured data collector"),
            Rules(["Ask one question at a time", "Validate each answer"]),
            OutputFormat("JSON with field: {current_question, validated_answers}"),
        ],
    ),
    Phase(
        name="confirmation",
        blocks=[
            Identity("Confirmation specialist"),
            Rules(["Summarize all collected data", "Ask for final confirmation"]),
        ],
    ),
])
```

---

#### 2. Chain-of-thought analysis with self-critique

```python
from promptise.prompts import prompt, ChainOfThought, SelfCritique

@prompt(model="openai:gpt-5-mini", strategy=ChainOfThought() + SelfCritique())
async def analyze_risk(scenario: str) -> str:
    """Analyze the business risk of: {scenario}

    Consider financial, operational, and reputational factors.
    Provide a risk score from 1-10 with justification."""

# The agent thinks step-by-step, then critiques its own reasoning
result = await analyze_risk("Expanding into the Japanese market without local partners")
```

---

## Context engineering

With the Context Engine, you control exactly what the LLM sees — with token-level precision. Instead of hoping your context fits, you set a budget, register layers by priority, and the engine drops the least important content first when space is tight.

### Build ideas

---

#### 1. Token-budgeted agent for long conversations

**The scenario**
Your agent handles conversations that grow to 200+ messages. Without management, the context window overflows and the agent crashes or loses early context.

```python
from promptise import build_agent, ContextEngine

engine = ContextEngine(model_context_window=128_000, response_reserve=4_000)
# identity, user_message and conversation (oldest pairs trimmed first) are built in.
engine.add_layer("rules", priority=9, required=True, content="Never share account data.")

agent = await build_agent(
    model="openai:gpt-5-mini",
    servers=servers,
    context_engine=engine,
)
# The engine trims oldest conversation pairs first, preserving identity and rules
# The agent never overflows — it gracefully drops old context
```

---

## Combining modules: what becomes possible

### Full-stack autonomous customer support system

**Modules involved:** Agent, MCP Server, Runtime, Prompt Engineering

**The scenario:** A support system where an MCP server exposes your ticketing system and knowledge base; the Agent connects to both with guardrails and caching; the Runtime deploys it as an autonomous process that monitors the ticket queue via webhook triggers; and Prompt Engineering evolves the system prompt across conversation phases (greeting → investigation → resolution → follow-up).

```python
import asyncio
from promptise.runtime import AgentRuntime, ProcessConfig, TriggerConfig, BudgetConfig, MissionConfig
from promptise.runtime.config import InboxConfig
from promptise import (
    EventNotifier, WebhookSink, ApprovalPolicy, WebhookApprovalHandler,
    PromptiseSecurityScanner,
)

async def main():
    scanner = PromptiseSecurityScanner.default()
    scanner.warmup()

    notifier = EventNotifier(sinks=[
        WebhookSink(url="https://hooks.slack.com/services/...",
                     events=["invocation.error", "health.anomaly", "approval.requested"]),
    ])

    async with AgentRuntime(event_notifier=notifier) as runtime:
        await runtime.add_process("support-agent", ProcessConfig(
            model="openai:gpt-5-mini",
            instructions=(
                "You are an autonomous customer support agent. "
                "Monitor the ticket queue, investigate issues, draft responses, "
                "and escalate when you cannot resolve within 3 attempts."
            ),
            servers={"tickets": {"url": "http://ticket-mcp:8000/mcp"}},
            triggers=[
                TriggerConfig(type="webhook", webhook_path="/new-ticket", webhook_port=9090),
                TriggerConfig(type="cron", cron_expression="*/15 * * * *"),
            ],
            budget=BudgetConfig(
                enabled=True,
                max_tool_calls_per_day=1000,
                max_cost_per_day=200.0,
                on_exceeded="pause",
            ),
            health=HealthConfig(enabled=True, stuck_threshold=5, on_anomaly="escalate"),
            inbox=InboxConfig(enabled=True),
            approval=ApprovalPolicy(
                tools=["send_customer_email", "issue_refund"],
                handler=WebhookApprovalHandler(url="https://approvals.internal/review"),
            ),
            guardrails=scanner,
        ))

        await runtime.start_all()
        print("Support agent deployed. Listening for tickets...")
        await asyncio.sleep(86400)

asyncio.run(main())
```

**Why each module is essential:**
- **Agent**: Discovers and calls ticketing tools, searches knowledge base, manages memory
- **MCP Server**: Exposes the ticket system as discoverable tools with auth and rate limiting
- **Runtime**: Keeps the agent running 24/7, handles triggers, enforces budget, detects stuck behavior
- **Prompt Engineering**: Adapts the agent's approach across conversation phases

---

## Start building

Pick the module that matches your next project. The [Quick Start](../getting-started/quickstart.md) gets you running in 5 minutes. Each module has its own reference: [Agent](../core/agents/building-agents.md), [MCP Server](../guides/production-mcp-servers.md), [Runtime](../guides/agentic-runtime.md), [Prompts](../guides/prompt-engineering.md). The examples on this page are starting points — the framework handles combinations and edge cases we have not shown here.
