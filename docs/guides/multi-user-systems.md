---
title: Build a Multi-User AI Agent System — Multi-tenant Python tutorial
description: Build multi-tenant AI agent systems in Python with Promptise Foundry. JWT authentication, capability-based access control, per-user audit trails, isolated semantic cache and conversation persistence, and CallerContext propagation across the entire stack.
keywords: multi-user AI agent, multi-tenant AI agent, AI agent authentication, JWT AI agent, AI agent access control, secure AI agent Python
---

# Building Agentic Multi-User Systems

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## What You'll Build

A production-ready multi-user AI application where authenticated users each get isolated conversations, personalized memory, scoped cache, and guardrailed responses — backed by MCP servers with JWT authentication, role-based access control, tamper-evident audit logging, and per-user session state. The full stack: user authenticates → agent processes their request with their identity → MCP tools enforce their permissions → response is guardrailed → conversation persists to their session.

## Concepts

A **multi-user agentic system** has three layers of identity:

1. **User identity** — who the human is (JWT claims, user_id, roles, scopes). Carried by `CallerContext` on every agent invocation.
2. **Agent-to-server identity** — who the MCP server sees. The agent's own credential (`HTTPServerSpec.bearer_token`) opens the connection and discovers tools; when an invocation's `CallerContext` carries the user's `bearer_token`, every tool call during it is made with that token instead.
3. **Session identity** — which conversation this message belongs to. Carried by `session_id` and `user_id` on every `chat()` call.

These three layers work together. The user's JWT flows from your backend through the agent to the MCP server. The MCP server validates the token, extracts roles/scopes, and applies per-tool guards. The conversation store enforces ownership. The cache isolates per-user. Guardrails scan every input and output. Observability records every decision with the authenticated identity attached.

```
User (JWT) → Your Backend → agent.ainvoke(..., caller=CallerContext(...))
                                    ↓
                            CallerContext(user_id, bearer_token=jwt, roles, scopes)
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              Guardrails      Semantic Cache   Conversation Store
              (scan I/O)    (per_user scope)  (ownership enforced)
                    ↓               ↓               ↓
                    └───────────────┼───────────────┘
                                    ↓
                            MCP Server (JWT validated)
                                    ↓
                            Guards (HasRole, HasScope)
                                    ↓
                            Tool Handler (ctx.client has identity)
                                    ↓
                            Audit Log (client_id + tool + args + result)
```

---

## Step 1: Authenticate Users on Your MCP Server

Start with the MCP server — the tool layer that enforces permissions. Every tool call arrives with the caller's identity.

```python
from promptise.mcp.server import MCPServer, JWTAuth, AuthMiddleware, HasRole, HasScope

auth = JWTAuth(secret="${JWT_SECRET}")

server = MCPServer(name="customer-api", auth=auth)
```

The `JWTAuth` provider validates Bearer tokens on every request. It extracts `sub` (client_id), `roles`, and `scope` claims from the JWT payload. Verified tokens are cached in an LRU (256 entries by default) so repeated calls with the same token skip cryptographic validation.

### A real MCP server with 30 tools

Production servers have many tools across domains. Here's a customer operations server with 30 tools organized into routers — each domain has its own namespace, guards, and middleware:

```python
from promptise.mcp.server import (
    MCPServer, MCPRouter, JWTAuth, AuthMiddleware,
    HasRole, HasScope, HasAllRoles, RequireAuth,
    LoggingMiddleware, TimeoutMiddleware, RateLimitMiddleware,
    AuditMiddleware, RequestContext,
)

auth = JWTAuth(secret="${JWT_SECRET}")
server = MCPServer(name="customer-ops", auth=auth)

server.add_middleware(LoggingMiddleware())
server.add_middleware(TimeoutMiddleware(default_timeout=30.0))
server.add_middleware(RateLimitMiddleware(requests_per_second=20))
server.add_middleware(AuditMiddleware(secret="${AUDIT_SECRET}"))

# ── Customer Management (7 tools) ────────────────────────────
customers = MCPRouter(prefix="customers")

@customers.tool(guards=[HasScope("read")])
async def get_customer(customer_id: str, ctx: RequestContext) -> dict:
    """Retrieve customer profile."""
    return await db.get_customer(customer_id, tenant=ctx.client_id)

@customers.tool(guards=[HasScope("read")])
async def search_customers(query: str, limit: int = 10) -> list[dict]:
    """Search customers by name, email, or phone."""
    return await db.search_customers(query, limit=limit)

@customers.tool(guards=[HasScope("write")])
async def update_customer(customer_id: str, updates: dict) -> dict:
    """Update customer profile fields."""
    return await db.update_customer(customer_id, updates)

@customers.tool(guards=[HasAllRoles("admin", "support")])
async def delete_customer(customer_id: str) -> str:
    """Permanently delete a customer. Requires admin + support roles."""
    await db.delete_customer(customer_id)
    return f"Customer {customer_id} deleted"

@customers.tool(guards=[HasScope("read")])
async def get_customer_history(customer_id: str) -> list[dict]:
    """Get interaction history for a customer."""
    return await db.get_interactions(customer_id)

@customers.tool(guards=[HasScope("write")])
async def add_customer_note(customer_id: str, note: str) -> dict:
    """Add an internal note to a customer profile."""
    return await db.add_note(customer_id, note)

@customers.tool(guards=[HasScope("read")])
async def get_customer_segments(customer_id: str) -> list[str]:
    """Get marketing segments for a customer."""
    return await db.get_segments(customer_id)

server.mount(customers)

# ── Orders & Billing (8 tools) ───────────────────────────────
orders = MCPRouter(prefix="orders")

@orders.tool(guards=[HasScope("read")])
async def get_order(order_id: str) -> dict:
    """Retrieve order details."""
    return await db.get_order(order_id)

@orders.tool(guards=[HasScope("read")])
async def list_orders(customer_id: str, status: str = "all") -> list[dict]:
    """List orders for a customer, optionally filtered by status."""
    return await db.list_orders(customer_id, status=status)

@orders.tool(guards=[HasScope("read")])
async def track_shipment(order_id: str) -> dict:
    """Get real-time shipment tracking for an order."""
    return await shipping.track(order_id)

@orders.tool(guards=[HasScope("billing:read")])
async def get_invoice(invoice_id: str) -> dict:
    """Retrieve invoice details."""
    return await billing.get_invoice(invoice_id)

@orders.tool(guards=[HasScope("billing:write")])
async def process_refund(order_id: str, amount: float, reason: str) -> dict:
    """Process a refund. Requires billing:write scope."""
    return await billing.refund(order_id, amount, reason)

@orders.tool(guards=[HasScope("billing:write")])
async def apply_discount(order_id: str, code: str) -> dict:
    """Apply a discount code to an order."""
    return await billing.apply_discount(order_id, code)

@orders.tool(guards=[HasScope("write")])
async def cancel_order(order_id: str, reason: str) -> str:
    """Cancel a pending order."""
    return await db.cancel_order(order_id, reason)

@orders.tool(guards=[HasScope("billing:read")])
async def get_payment_methods(customer_id: str) -> list[dict]:
    """List saved payment methods for a customer."""
    return await billing.get_payment_methods(customer_id)

server.mount(orders)

# ── Support Tickets (8 tools) ────────────────────────────────
tickets = MCPRouter(prefix="tickets")

@tickets.tool(guards=[HasScope("read")])
async def get_ticket(ticket_id: str) -> dict:
    """Retrieve a support ticket."""
    return await db.get_ticket(ticket_id)

@tickets.tool(guards=[HasScope("read")])
async def list_tickets(customer_id: str = None, status: str = "open") -> list[dict]:
    """List support tickets, optionally filtered by customer and status."""
    return await db.list_tickets(customer_id=customer_id, status=status)

@tickets.tool(guards=[HasScope("write")])
async def create_ticket(customer_id: str, subject: str, description: str, priority: str = "medium") -> dict:
    """Create a new support ticket."""
    return await db.create_ticket(customer_id, subject, description, priority)

@tickets.tool(guards=[HasScope("write")])
async def update_ticket(ticket_id: str, updates: dict) -> dict:
    """Update ticket fields (status, priority, assignee)."""
    return await db.update_ticket(ticket_id, updates)

@tickets.tool(guards=[HasScope("write")])
async def add_ticket_comment(ticket_id: str, comment: str, internal: bool = False) -> dict:
    """Add a comment to a ticket. Set internal=True for agent-only notes."""
    return await db.add_ticket_comment(ticket_id, comment, internal)

@tickets.tool(guards=[HasRole("support")])
async def assign_ticket(ticket_id: str, agent_id: str) -> str:
    """Assign a ticket to a support agent."""
    return await db.assign_ticket(ticket_id, agent_id)

@tickets.tool(guards=[HasRole("support")])
async def escalate_ticket(ticket_id: str, reason: str) -> str:
    """Escalate a ticket to tier-2 support."""
    return await db.escalate_ticket(ticket_id, reason)

@tickets.tool(guards=[HasRole("support")])
async def close_ticket(ticket_id: str, resolution: str) -> str:
    """Close a resolved ticket."""
    return await db.close_ticket(ticket_id, resolution)

server.mount(tickets)

# ── Analytics & Reporting (4 tools) ──────────────────────────
analytics = MCPRouter(prefix="analytics")

@analytics.tool(guards=[HasRole("analyst")])
async def customer_lifetime_value(customer_id: str) -> dict:
    """Calculate customer lifetime value."""
    return await analytics_db.clv(customer_id)

@analytics.tool(guards=[HasRole("analyst")])
async def churn_risk_score(customer_id: str) -> dict:
    """Get churn risk prediction for a customer."""
    return await ml.predict_churn(customer_id)

@analytics.tool(guards=[HasRole("analyst")])
async def revenue_report(period: str = "monthly") -> dict:
    """Generate revenue summary report."""
    return await analytics_db.revenue(period)

@analytics.tool(guards=[HasRole("analyst")])
async def support_metrics(period: str = "weekly") -> dict:
    """Get support team performance metrics."""
    return await analytics_db.support_metrics(period)

server.mount(analytics)

# ── Internal Operations (3 tools) ────────────────────────────
ops = MCPRouter(prefix="ops")

@ops.tool(guards=[HasRole("admin")])
async def system_health() -> dict:
    """Check system health across all services."""
    return await monitoring.health_check()

@ops.tool(guards=[HasAllRoles("admin", "ops")])
async def feature_toggle(feature: str, enabled: bool) -> str:
    """Enable or disable a feature flag. Requires admin + ops."""
    await config_db.set_feature(feature, enabled)
    return f"Feature {feature} {'enabled' if enabled else 'disabled'}"

@ops.tool(guards=[HasRole("admin")])
async def audit_log(hours: int = 24) -> list[dict]:
    """Retrieve recent audit log entries."""
    return await db.get_audit_log(hours=hours)

server.mount(ops)
```

That's 30 tools across 5 domains. Each domain has its own router with appropriate guards. The agent discovers all 30 automatically — but sending all 30 tool descriptions to the LLM on every message wastes thousands of tokens. That's where semantic tool optimization comes in.

For asymmetric tokens from identity providers (Auth0, Keycloak, Okta):

```python
import os

from promptise.mcp.server import AsymmetricJWTAuth

auth = AsymmetricJWTAuth(
    public_key=os.environ["IDP_PUBLIC_KEY"],
    algorithm="RS256",
    issuer="https://auth.example.com",
    audience="my-api",
)
```

For simpler deployments, API key authentication with role mappings:

```python
from promptise.mcp.server import APIKeyAuth

auth = APIKeyAuth(keys={
    "key-admin-abc": {"client_id": "admin-agent", "roles": ["admin"]},
    "key-readonly-xyz": {"client_id": "reader-agent", "roles": ["reader"]},
})
```

---

## Step 2: Protect Tools with Guards

Guards enforce *what* an authenticated client can do. Authentication tells you *who*. Guards decide *whether they're allowed*.

```python
@server.tool(guards=[HasRole("analyst")])
async def get_customer_data(customer_id: str) -> dict:
    """Retrieve customer information."""
    return await db.get_customer(customer_id)

@server.tool(guards=[HasRole("admin")])
async def delete_customer(customer_id: str) -> str:
    """Delete a customer record. Requires admin role."""
    await db.delete_customer(customer_id)
    return f"Customer {customer_id} deleted"

@server.tool(guards=[HasScope("billing:write")])
async def process_refund(order_id: str, amount: float) -> dict:
    """Process a refund. Requires billing:write scope."""
    return await billing.refund(order_id, amount)
```

Six built-in guards:

| Guard | What it checks |
|---|---|
| `RequireAuth()` | Client is authenticated (any identity) |
| `HasRole("admin")` | Client has at least one of the given roles |
| `HasAllRoles("admin", "billing")` | Client has every given role |
| `HasScope("read")` | Client has at least one of the given OAuth2 scopes |
| `HasAllScopes("read", "write")` | Client has every given scope |
| `RequireClientId("agent-007")` | Client ID matches exactly |

Custom guards implement the `Guard` protocol — any callable that takes a `RequestContext` and returns allow/deny:

```python
class RequirePremiumPlan:
    async def check(self, ctx):
        plan = ctx.client.extra.get("plan")
        return plan in ("pro", "enterprise")

    def describe_denial(self, ctx):
        return "This tool requires a Pro or Enterprise plan."

@server.tool(guards=[RequirePremiumPlan()])
async def advanced_analytics(query: str) -> dict: ...
```

---

## Step 3: Access Client Identity in Tool Handlers

Inside any tool handler, `ctx.client` carries the full authenticated identity:

```python
from promptise.mcp.server import RequestContext

@server.tool(auth=True)
async def my_tool(query: str, ctx: RequestContext) -> dict:
    user_id = ctx.client_id                  # "user-42"
    roles = ctx.client.roles                 # {"analyst", "data-team"}
    scopes = ctx.client.scopes              # {"read", "write"}
    issuer = ctx.client.issuer              # "https://auth.example.com"
    ip = ctx.client.ip_address              # "192.168.1.100"
    claims = ctx.client.claims              # Full JWT payload
    org = ctx.client.extra.get("org_id")    # Custom enrichment

    # Use identity for data isolation
    return await db.query(query, tenant_id=user_id)
```

The `ClientContext` carries everything your handler needs to scope data access, log actions, and enforce tenant isolation.

### Enrich client context with custom data

Use the `on_authenticate` hook to add application-specific fields (plan, org, tenant) from your database:

```python
async def enrich_client(ctx):
    user = await db.get_user(ctx.client_id)
    ctx.client.extra["org_id"] = user.org_id
    ctx.client.extra["plan"] = user.plan
    return ctx

middleware = AuthMiddleware(auth, on_authenticate=enrich_client)
server.add_middleware(middleware)
```

---

## Step 4: Enable Tamper-Evident Audit Logging

Every tool call is recorded with the authenticated identity — who called what, when, with what arguments, and what happened:

```python
from promptise.mcp.server import AuditMiddleware

audit = AuditMiddleware(secret="${AUDIT_HMAC_SECRET}")
server.add_middleware(audit)
```

Each audit entry includes `client_id`, `tool_name`, `request_id`, `timestamp`, `duration`, `status`, and optionally `args` and `result`. Every entry is HMAC-chained — each entry includes a hash of the previous one. If anyone tampers with the log (edit, delete, reorder), the chain breaks and the tampering is detectable.

---

## Step 5: Per-Client Session State on the Server

MCP session state lets tools store per-client data that persists across tool calls within the same session:

```python
from promptise.mcp.server import SessionState

@server.tool()
async def track_search(query: str, state: SessionState) -> str:
    """Track user searches within this session."""
    history = state.get("search_history", [])
    history.append(query)
    state.set("search_history", history)
    return f"Tracked. {len(history)} searches this session."
```

Each client session gets its own `SessionState` — isolated automatically. No cross-client leakage.

---

## Step 6: Connect the Agent with User Identity

Now connect the agent to your authenticated MCP server. The user's JWT flows through:

```python
from promptise import build_agent
from promptise.config import HTTPServerSpec

# Your backend receives the user's JWT from their request
user_jwt = request.headers["Authorization"].split(" ")[1]

agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={
        "customer_api": HTTPServerSpec(
            url="http://localhost:8000/mcp",
            bearer_token=user_jwt,  # User's JWT flows to MCP server
        ),
    },
    instructions="You are a customer support agent.",
)
```

Every tool call the agent makes includes this JWT in the `Authorization` header. The MCP server validates it, extracts roles/scopes, and applies guards. If the user has the `analyst` role, they can call `get_customer_data`. If they don't have `admin`, `delete_customer` is blocked.

!!! tip "One agent for every user"
    Building an agent per request, as here, works but repeats tool discovery
    each time. You can instead build one agent with a service credential and
    pass each user's JWT on the invocation:
    `agent.ainvoke(..., caller=CallerContext(user_id=..., bearer_token=user_jwt))`.
    Every tool call during that invocation is then made with the user's
    token, over an MCP session opened for it, so concurrent users never
    share credentials. See
    [The user's token reaches the MCP server](../mcp/server/multi-tenancy.md#the-users-token-reaches-the-mcp-server).
    Never build a shared agent with one *user's* token: every user would
    then act as that user.

### Semantic tool optimization — 30 tools without the token cost

With 30 tools, every LLM call sends ~3,000-5,000 tokens of tool descriptions. Most messages only need 3-5 tools. Semantic optimization analyzes the user's message, matches it against tool descriptions using a local embedding model, and sends only the relevant tools:

```python
agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={
        "customer_ops": HTTPServerSpec(
            url="http://customer-ops:8000/mcp",
            bearer_token=user_jwt,
        ),
    },
    optimize_tools="semantic",  # 40-70% fewer tokens per request
)
```

"What's the status of order #12345?" → the LLM sees `get_order`, `track_shipment`, `list_orders` — not the 27 other tools about tickets, analytics, and admin operations. "Show me the churn risk for customer 42" → the LLM sees `churn_risk_score`, `customer_lifetime_value`, `get_customer` — not billing or ticket tools.

The embedding model runs locally (no API calls, no data leaving your infrastructure). For custom models or air-gapped deployments:

```python
from promptise.tools import ToolOptimizationConfig, OptimizationLevel

agent = await build_agent(
    ...,
    optimize_tools=ToolOptimizationConfig(
        level=OptimizationLevel.SEMANTIC,
        max_tools=8,                                    # Send at most 8 tools per request
        embedding_model="/models/local/all-MiniLM-L6-v2",  # Fully offline
    ),
)
```

---

## Step 7: Carry User Identity Through the Agent

`CallerContext` carries the user's identity through every layer of the agent — guardrails, cache, conversation store, observability:

```python
from promptise.agent import CallerContext

caller = CallerContext(
    user_id="user-42",
    bearer_token=user_jwt,
    roles={"analyst", "support"},
    scopes={"read", "write"},
    metadata={"ip": request.remote_addr, "session_id": "sess_abc123"},
)

result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Show me customer 99's data"}]},
    caller=caller,
)
```

Every downstream system sees this caller:

- **Guardrails** — `get_current_caller()` accesses the caller for audit enrichment
- **Semantic cache** — `_build_scope_key(caller)` isolates cache per user
- **Observability** — timeline events include the authenticated identity
- **Conversation store** — ownership enforcement uses `caller.user_id`

---

## Step 8: Add Conversation Persistence with Ownership

Users expect to resume conversations. The conversation store persists messages across sessions with ownership enforcement — User A cannot access User B's conversations:

```python
from promptise import build_agent
from promptise.conversations import PostgresConversationStore

store = PostgresConversationStore(dsn="postgresql://localhost/myapp")

agent = await build_agent(
    ...,
    conversation_store=store,
    conversation_max_messages=100,
)

# Each user's conversations are isolated
response = await agent.chat(
    "What's my order status?",
    session_id="sess_abc123",
    user_id="user-42",
)

# Different user trying to access this session → SessionAccessDenied
try:
    await agent.chat("Hello", session_id="sess_abc123", user_id="user-99")
except SessionAccessDenied as e:
    print(f"Blocked: user-99 tried to access user-42's session")
```

Four built-in stores:

| Store | Use case |
|---|---|
| `InMemoryConversationStore` | Testing, development |
| `SQLiteConversationStore` | Single-node local apps |
| `PostgresConversationStore` | Production, distributed |
| `RedisConversationStore` | Ephemeral, caching layer |

Session IDs are generated with `generate_session_id()` — cryptographically random, non-enumerable (`secrets.token_hex(16)`). Ownership is assigned on the first message and enforced on every subsequent access.

### Session management

```python
# List a user's sessions
sessions = await agent.list_sessions(user_id="user-42")

# Get session metadata (enforces ownership)
session = await agent.get_session("sess_abc123", user_id="user-42")

# Delete a session (enforces ownership)
await agent.delete_session("sess_abc123", user_id="user-42")
```

---

## Step 9: Add Semantic Caching with User Isolation

Identical questions from the same user get instant responses from cache. Different users never see each other's cached responses:

```python
from promptise.cache import SemanticCache

cache = SemanticCache(
    scope="per_user",           # Default — each user gets isolated cache
    similarity_threshold=0.92,
    default_ttl=3600,
)

agent = await build_agent(..., cache=cache)

# User 42 asks a question → LLM call → response cached under user:user-42
await agent.ainvoke(input, caller=CallerContext(user_id="user-42"))

# User 42 asks a similar question → cache hit (instant response, zero tokens)
await agent.ainvoke(similar_input, caller=CallerContext(user_id="user-42"))

# User 99 asks the same question → cache miss (different user scope)
await agent.ainvoke(same_input, caller=CallerContext(user_id="user-99"))
```

Three isolation levels:

| Scope | Isolation | Use case |
|---|---|---|
| `per_user` (default) | Each user has own cache | Any personalized agent |
| `per_session` | Each session has own cache | Context-heavy conversations |
| `shared` | Everyone shares cache | Public FAQ, documentation bots |

GDPR compliance: `await cache.purge_user("user-42")` removes all cached entries for that user.

---

## Step 10: Add Guardrails for Input/Output Security

Protect every user's input from injection attacks and every output from PII/credential leakage:

```python
from promptise import PromptiseSecurityScanner

scanner = PromptiseSecurityScanner.default()
scanner.warmup()

agent = await build_agent(
    ...,
    guardrails=scanner,
)
```

- **Input**: Prompt injection attempts are blocked before reaching the LLM
- **Output**: Credit card numbers, SSNs, API keys, and 160+ sensitive patterns are redacted before reaching the user
- **Cached responses**: Output guardrails run on every cache serve — new rules catch old cached PII

Guardrails work on every invocation path: `ainvoke()`, `astream()`, and `chat()`.

---

## Step 11: Add Memory with Auto-Injection

Give the agent persistent memory that's automatically searched and injected before every invocation. Every provider — `InMemoryProvider`, `ChromaProvider`, and `Mem0Provider` — supports two isolation modes via the `scope` parameter:

| Scope | Isolation | Use case |
|---|---|---|
| `MemoryScope.SHARED` (default) | All users share one pool | Knowledge-base agents, public FAQ, shared org memory |
| `MemoryScope.PER_USER` | Each user has isolated memory | Personal assistants, multi-tenant SaaS |

### Shared memory (default)

```python
from promptise.memory import ChromaProvider

memory = ChromaProvider(
    collection_name="support_memory",
    persist_directory=".promptise/memory",
)

agent = await build_agent(
    ...,
    memory=memory,
    memory_auto_store=True,  # Auto-persist exchanges
)
```

All users of this agent read and write the same pool — useful when the memory represents shared knowledge.

### Per-user memory

```python
from promptise.memory import ChromaProvider, MemoryScope

memory = ChromaProvider(
    collection_name="support_memory",
    persist_directory=".promptise/memory",
    scope=MemoryScope.PER_USER,
)

agent = await build_agent(..., memory=memory, memory_auto_store=True)
```

The agent automatically reads `caller.user_id` from `CallerContext` and scopes every `search`/`add`/`delete` to that tenant. Users cannot read or delete each other's entries. If a per-user operation runs without a `user_id`, it raises `MemoryIsolationError`; the `MemoryAgent` wrapper catches this during auto-search so background tasks degrade gracefully (skip memory injection) rather than crash.

GDPR compliance is built in: `await memory.purge_user("user-42")` removes every entry owned by that user across all three providers.

---

## Step 12: Full Observability with Identity Tracking

Every invocation is traced with the authenticated identity:

```python
from promptise.observability_config import ObservabilityConfig, ObserveLevel, TransporterType

agent = await build_agent(
    ...,
    observe=ObservabilityConfig(
        level=ObserveLevel.STANDARD,
        transporters=[
            TransporterType.STRUCTURED_LOG,  # ELK/Splunk/Datadog
            TransporterType.PROMETHEUS,       # Grafana dashboards
        ],
        log_file="./logs/agent.jsonl",
        correlation_id=request_id,  # Tie to your HTTP request
    ),
)
```

Timeline events include tool calls, token counts, latencies, cache hits/misses, and guardrail results — all tagged with the correlation ID for end-to-end tracing.

---

## Complete Example

A multi-user customer support system with every feature integrated:

```python
import asyncio
from promptise import build_agent, PromptiseSecurityScanner
from promptise.agent import CallerContext
from promptise.config import HTTPServerSpec
from promptise.cache import SemanticCache
from promptise.conversations import PostgresConversationStore
from promptise.memory import ChromaProvider

# Shared across all requests (stateless, thread-safe)
scanner = PromptiseSecurityScanner.default()
scanner.warmup()
conversation_store = PostgresConversationStore(dsn="${DATABASE_URL}")

async def handle_user_message(user_jwt: str, user_id: str, session_id: str, message: str):
    """Handle a single user message with full multi-user isolation.

    This is what your FastAPI/Django/Flask endpoint calls.
    The user's JWT flows end-to-end: backend → agent → MCP servers.
    """

    # Build agent with user's JWT for MCP auth
    # The agent discovers all 30 tools automatically
    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={
            # All 5 routers from one server, or split across services
            "customer_ops": HTTPServerSpec(
                url="http://customer-ops:8000/mcp",
                bearer_token=user_jwt,  # User's JWT → MCP server validates → guards enforce
            ),
        },
        instructions=(
            "You are a customer support agent for Acme Corp. "
            "Help users with their accounts, orders, billing, and support tickets. "
            "Be concise and professional. Never share data from other customers."
        ),
        # 30 tools discovered, but only 5-8 sent per request (40-70% token savings)
        optimize_tools="semantic",
        # Each user gets isolated conversation history
        conversation_store=conversation_store,
        conversation_max_messages=50,
        # Each user gets isolated cache (same question, different user = cache miss)
        cache=SemanticCache(scope="per_user", default_ttl=1800),
        # Shared memory across all users (agent learns from all interactions)
        memory=ChromaProvider(persist_directory=".promptise/memory"),
        # Block injection attacks on input, redact PII/credentials on output
        guardrails=scanner,
        observe=True,
    )

    try:
        # CallerContext carries identity through every layer:
        # guardrails, cache scoping, observability, conversation ownership
        caller = CallerContext(
            user_id=user_id,
            bearer_token=user_jwt,
            roles={"support_user"},
            metadata={"session_id": session_id},
        )

        # chat() handles the full pipeline:
        # 1. Ownership check (is this user_id allowed to access this session?)
        # 2. Load conversation history from PostgreSQL
        # 3. Input guardrails → memory search → cache check
        # 4. Semantic tool selection (pick 5-8 of 30 tools)
        # 5. LLM invocation with enriched context
        # 6. MCP tool calls (JWT validated, guards enforced, audit logged)
        # 7. Output guardrails (PII/credential redaction)
        # 8. Cache store (under user:user-42 scope)
        # 9. Persist messages to PostgreSQL with ownership
        response = await agent.chat(
            message,
            session_id=session_id,
            user_id=user_id,
            caller=caller,
        )
        return response
    finally:
        await agent.shutdown()
```

**What happens on each message:**

1. User's JWT is attached to MCP server connections
2. `CallerContext` is stored in async context for all downstream access
3. Input guardrails scan for injection attacks → block if detected
4. Memory searches for relevant context → injected into system prompt
5. Semantic cache checks for similar query from this user → instant return if hit
6. Conversation history loaded from PostgreSQL → ownership verified
7. **Semantic tool selection** picks 5-8 relevant tools from 30 → saves 40-70% tokens
8. LLM invoked with enriched context (memory + history + selected tools)
9. Agent calls MCP tools → JWT validated → guards enforced → audit logged
10. Output guardrails scan response → PII/credentials redacted
11. Response cached under `user:{user_id}` scope
12. Messages persisted to PostgreSQL with ownership stamp
13. Observability records everything with correlation ID

---

## Multi-tenancy and approval gates

For SaaS platforms serving many organizations — or any tool that needs a human
in the loop — two features turn the per-user isolation above into per-tenant
invariants and add server-enforced approval.

**First-class multi-tenancy.** Give each API key (or JWT claim) a tenant, and
tenant identity flows into every isolation surface — cache, sessions, rate
limits, audit, and guards — so two tenants with the *same* user id can never
see each other's data:

```python
auth = APIKeyAuth(keys={
    "sk-acme":   {"client_id": "svc", "roles": ["analyst"], "tenant_id": "acme"},
    "sk-globex": {"client_id": "svc", "roles": ["analyst"], "tenant_id": "globex"},
})
server = MCPServer("api", require_tenant=True)   # every tool must carry a tenant
server.add_middleware(AuthMiddleware(auth))       # or AuthMiddleware(JWTAuth(...), tenant_claim="org")

@server.tool(auth=True, guards=[HasTenant("acme")])   # tenant-scoped access
async def acme_report(ctx: RequestContext) -> dict: ...
```

Agent side, `CallerContext(user_id="alice", tenant_id="acme")` scopes cache,
memory, and conversations per tenant automatically. Full guide:
[Multi-Tenancy](../mcp/server/multi-tenancy.md).

**Server-side approval gates.** Declare a destructive tool `requires_approval`
and install an `ApprovalGateMiddleware` — the call blocks until a human
approves, for any MCP client, denied by default on timeout:

```python
from promptise.mcp.server import ApprovalGateMiddleware, PendingApprover

approver = PendingApprover(server, approver_role="approver")
server.add_middleware(ApprovalGateMiddleware(approver, timeout=300))

@server.tool(auth=True, guards=[HasRole("billing")], requires_approval=True)
async def issue_refund(order_id: str, amount: float) -> dict: ...
```

Guards run before a reviewer is bothered, and a caller can't approve their own
request (four-eyes). Full guide:
[Approval Gates](../mcp/server/approval-gates.md). Runnable end-to-end demo:
`examples/mcp/tenancy_and_approval.py`.

## Security Architecture Summary

| Layer | Feature | What it protects |
|---|---|---|
| **Transport** | JWT/OAuth/API key auth | Identifies every caller |
| **Authorization** | Guards (role, scope, client ID, tenant) | Controls who can call which tools |
| **Multi-tenancy** | `tenant_id` across cache/sessions/limits/audit | Prevents cross-tenant leakage (even for identical user ids) |
| **Human-in-the-loop** | `requires_approval` + `ApprovalGateMiddleware` | Blocks destructive calls until a human approves (four-eyes) |
| **Session isolation** | SessionState per client | Prevents cross-client state leakage |
| **Conversation ownership** | `_enforce_ownership()` | Prevents cross-user conversation access |
| **Cache isolation** | `per_user` scope key (tenant-qualified) | Prevents cross-user / cross-tenant cache hits |
| **Input protection** | Prompt injection detection | Blocks LLM manipulation attempts |
| **Output protection** | PII/credential redaction | Prevents sensitive data leakage |
| **Audit trail** | HMAC-chained audit log (tenant-stamped) | Tamper-evident record of every action |
| **Secret management** | Per-process SecretScope | Credentials isolated, TTL-expired, zero-filled |

---

## What's Next

**Reference documentation:**

- [Guardrails](../core/guardrails.md) — all detector types, 165+ patterns, ML models, content safety
- [Semantic Cache](../core/cache.md) — backends, embedding providers, scope modes, GDPR purge
- [Conversations](../core/conversations.md) — all 4 stores, ownership model, custom stores
- [Memory](../core/memory.md) — 3 providers, auto-injection, sanitization

**MCP server reference:**

- [Auth & Security](../mcp/server/auth-security.md) — JWT, OAuth, API keys, guards, audit logging
- [Multi-Tenancy](../mcp/server/multi-tenancy.md) — tenant isolation across the whole stack
- [Approval Gates](../mcp/server/approval-gates.md) — server-side human-in-the-loop
- [Routers & Middleware](../mcp/server/routers-middleware.md) — middleware chain, DI, session state

**Other guides:**

- [Building AI Agents](building-agents.md) — the core agent that powers every process
- [Building Production MCP Servers](production-mcp-servers.md) — the tool servers your agents connect to
- [Building Agentic Runtime Systems](agentic-runtime.md) — autonomous agents with triggers, governance, distributed coordination
