# Multi-User Identity: CallerContext to MCP Server

How user identity flows from your application through the agent to MCP tool servers — end to end. Covers CallerContext, JWT token propagation, server-side auth, role guards, and per-user data isolation.

## The Flow

```
Your App                    Agent                     MCP Server
────────                    ─────                     ──────────
CallerContext        →  bearer_token in      →  AuthMiddleware
  user_id                 HTTP header              validates JWT
  bearer_token                                     extracts roles
  roles                                            builds ClientContext
  scopes                                        →  Guards check roles
                                                →  Handler receives ctx.client
```

Only the `bearer_token` crosses the wire. Roles, scopes, and user metadata are extracted server-side from the JWT — never trusted from the client.

## Step 1: Create CallerContext (Your Application)

```python
from promptise.agent import CallerContext

# Each user gets their own CallerContext
alice = CallerContext(
    user_id="user-alice-001",
    bearer_token="eyJhbGciOiJIUzI1NiIs...",  # JWT token
    roles={"analyst", "viewer"},               # Used for agent-side decisions
    scopes={"read", "write"},                  # OAuth2 scopes
    metadata={"team": "finance", "plan": "pro"},
)

bob = CallerContext(
    user_id="user-bob-002",
    bearer_token="eyJhbGciOiJIUzI1NiIs...",  # Different JWT
    roles={"admin"},
    scopes={"read", "write", "admin"},
)
```

| Field | Purpose | Crosses the wire? |
|-------|---------|-------------------|
| `user_id` | Agent-side identification (memory, cache, conversations) | No |
| `bearer_token` | Sent as `Authorization: Bearer <token>` to MCP servers | **Yes** |
| `roles` | Agent-side role checking (for agent logic, not server auth) | No |
| `scopes` | Agent-side scope checking | No |
| `metadata` | Custom data (team, plan, etc.) | No |

## Step 2: Pass CallerContext to Agent

```python
from promptise import build_agent

agent = await build_agent(
    model="openai:gpt-4o-mini",
    servers=my_servers,
    instructions="You are a helpful assistant.",
)

# Every invocation carries identity
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Show my account"}]},
    caller=alice,  # ← Identity for this request
)

# Or with conversation persistence
response = await agent.chat(
    user_message="What's my balance?",
    session_id="session-alice-001",
    caller=alice,  # ← Isolated memory, cache, conversations
)
```

The `caller` parameter:

- Stores the CallerContext in an async-safe contextvar for the duration of the invocation
- Sends `bearer_token` as `Authorization: Bearer <token>` on every tool call to an HTTP or SSE MCP server during the invocation
- Scopes memory search, conversation history, and semantic cache to `user_id`
- Available to guardrails, observability, and event handlers via `get_current_caller()`

### How the token travels

The agent is built once and keeps one connection per server, opened with
the credentials in its `HTTPServerSpec` — those are what tool discovery
uses. When an invocation carries a `bearer_token`, its tool calls go over a
separate MCP session opened with that token instead. Each distinct token
gets its own session, so concurrent invocations for different users never
share headers, and one user's token is never sent on another user's
request. Sessions are reused by later calls with the same token and closed
after five idle minutes.

- An invocation **without** a `bearer_token` uses the spec's credential
  (or the agent's identity). The server then sees the agent, not the user.
- A server whose spec sets `forward_caller_token=False` always gets the
  spec's credential. Use it for third-party servers that must not receive
  your users' tokens.
- stdio servers have no request headers and never receive the token; the
  agent logs a warning the first time it cannot send one. A stdio tool runs
  with the agent's own privileges for every caller.
- If the server rejects the user's token, that tool call fails with
  `401 Unauthorized` (reported to the model like any tool error); other
  users' calls are unaffected.

## Step 3: Build the MCP Server with Auth

```python
from promptise.mcp.server import (
    MCPServer, MCPRouter, JWTAuth, AuthMiddleware,
    HasRole, HasScope, RequireAuth, RequestContext,
)

server = MCPServer("my-api")

# Configure JWT validation (same secret as token issuer)
jwt_auth = JWTAuth(secret="your-jwt-secret")
server.add_middleware(AuthMiddleware(jwt_auth))

# Public tool — no auth required
@server.tool()
async def get_status() -> str:
    """Get system status. No auth needed."""
    return "OK"

# Authenticated tool — any valid JWT
@server.tool(auth=True)
async def get_profile(ctx: RequestContext) -> str:
    """Get the authenticated user's profile."""
    return f"User: {ctx.client.client_id}, Roles: {ctx.client.roles}"

# Role-guarded tool — only analysts
@server.tool(auth=True, roles=["analyst"])
async def run_query(sql: str, ctx: RequestContext) -> str:
    """Run a SQL query. Analyst role required."""
    return f"Query by {ctx.client.client_id}: {sql}"

# Scope-guarded tool — requires 'admin' scope
@server.tool(auth=True, scopes=["admin"])
async def delete_user(user_id: str, ctx: RequestContext) -> str:
    """Delete a user. Admin scope required."""
    return f"Deleted {user_id} by {ctx.client.client_id}"
```

## Step 4: Access Identity in Tool Handlers

When a request reaches your tool handler, `ctx.client` contains the full authenticated identity:

```python
@server.tool(auth=True)
async def my_tool(query: str, ctx: RequestContext) -> str:
    """A tool that uses caller identity."""

    # Who is calling?
    client_id = ctx.client.client_id      # "user-alice-001" (JWT 'sub' claim)
    roles = ctx.client.roles              # {"analyst", "viewer"}
    scopes = ctx.client.scopes            # {"read", "write"}

    # Full JWT payload
    claims = ctx.client.claims            # {"sub": "user-alice-001", "roles": [...], ...}
    issuer = ctx.client.issuer            # "https://auth.example.com"
    expires = ctx.client.expires_at       # 1714000000

    # Transport info
    ip = ctx.client.ip_address            # "192.168.1.42"
    ua = ctx.client.user_agent            # "promptise-mcp-client/2.0"

    # Custom enrichment (from on_authenticate hook)
    org = ctx.client.extra.get("org")     # "Acme Corp"
    plan = ctx.client.extra.get("plan")   # "enterprise"

    # Use identity for per-user data access
    return await db.query(query, user_id=client_id)
```

## Step 5: Enrich Identity with on_authenticate Hook

Add custom data to the client context after JWT validation:

```python
async def enrich_client(ctx: RequestContext):
    """Called after JWT is validated. Add org, plan, permissions."""
    if ctx.client:
        # Look up user in your database
        user = await db.get_user(ctx.client.client_id)
        if user:
            ctx.client.extra["org"] = user.organization
            ctx.client.extra["plan"] = user.plan
            ctx.client.extra["quota_remaining"] = user.quota

server.add_middleware(AuthMiddleware(jwt_auth, on_authenticate=enrich_client))
```

## Step 6: Per-User Data Isolation (Agent Side)

CallerContext automatically isolates these agent-side features:

```python
from promptise.conversations import SQLiteConversationStore
from promptise.memory import ChromaProvider
from promptise.cache import SemanticCache

agent = await build_agent(
    ...,
    conversation_store=SQLiteConversationStore("chat.db"),
    memory=ChromaProvider(persist_directory="./memory"),
    cache=SemanticCache(),
)

# Alice's request — isolated memory, cache, conversations
await agent.chat("What did I ask last time?",
    session_id="session-alice", caller=alice)

# Bob's request — completely separate
await agent.chat("What did I ask last time?",
    session_id="session-bob", caller=bob)
```

| Feature | Isolation method |
|---------|-----------------|
| Conversations | Session ownership enforcement — Alice can't read Bob's sessions |
| Memory | Search scoped to `caller.user_id` |
| Semantic Cache | Cache entries keyed by `caller.user_id` |
| Guardrails | Scan results tagged with `caller.user_id` for audit |
| Observability | Traces include `caller.user_id` for per-user debugging |

### Continuity across cross-agent delegation

The caller survives delegation hops. When an agent handling Alice's request
delegates to a peer via `ask_peer` / `broadcast`, the peer's `ainvoke()`
**inherits the ambient `CallerContext`** (unless an explicit `caller=` is
passed), so the peer's cache scoping, memory search, guardrail tagging, and
conversation ownership all remain attributed to Alice — the original human
principal — rather than running unattributed:

```python
# Outer agent runs as Alice ...
await orchestrator.ainvoke({"messages": [...]}, caller=alice)
# ... and any peer it delegates to sees the same principal:
# get_current_caller() inside the peer's run returns `alice`.
```

Passing an explicit `caller=` to a peer still takes precedence — inheritance
only fills the gap when no caller is supplied.

### Tenant isolation

Add `tenant_id` to make every isolation surface tenant-scoped — the table
above then keys on `tenant::user` instead of the bare `user_id`, so two
tenants with the same `user_id` are fully isolated:

```python
acme_alice   = CallerContext(user_id="alice", tenant_id="acme")
globex_alice = CallerContext(user_id="alice", tenant_id="globex")  # sees NOTHING of acme
```

Server-side, `AuthMiddleware(auth, tenant_claim="tenant_id")` extracts the
tenant from the JWT into `ctx.client.tenant_id`, where rate limits, audit
entries, and the `RequireTenant`/`HasTenant` guards read it. Full guide:
[Multi-Tenancy](../mcp/server/multi-tenancy.md).

## JWT Token Structure

The JWT must contain at minimum a `sub` claim (subject = user ID). Optional claims:

```json
{
  "sub": "user-alice-001",
  "iss": "https://auth.example.com",
  "aud": "promptise-api",
  "exp": 1714000000,
  "iat": 1713913600,
  "roles": ["analyst", "viewer"],
  "scope": "read write",
  "org": "Acme Corp"
}
```

| Claim | Maps to | Used by |
|-------|---------|---------|
| `sub` | `ctx.client.client_id` | Guards, handlers, audit |
| `roles` | `ctx.client.roles` | `HasRole` guard |
| `scope` | `ctx.client.scopes` (space-separated) | `HasScope` guard |
| `iss` | `ctx.client.issuer` | Validation |
| `exp` | `ctx.client.expires_at` | Token expiry check |
| Custom claims | `ctx.client.claims[key]` | Handler logic |

## Guard Reference

| Guard | What it checks | Usage |
|-------|---------------|-------|
| `RequireAuth()` | Any valid JWT present | `@server.tool(auth=True)` |
| `HasRole("admin")` | Role in JWT `roles` array | `@server.tool(roles=["admin"])` |
| `HasAllRoles(["admin", "write"])` | ALL roles present | Manual guard |
| `HasScope("read")` | Scope in JWT `scope` string | `@server.tool(scopes=["read"])` |
| `HasAllScopes(["read", "write"])` | ALL scopes present | Manual guard |
| `RequireClientId("agent-1")` | Specific client ID | Manual guard |

## Complete Example

```python
# server.py
from promptise.mcp.server import MCPServer, JWTAuth, AuthMiddleware

server = MCPServer("multi-user-api")
server.add_middleware(AuthMiddleware(JWTAuth(secret="secret")))

@server.tool(auth=True, roles=["analyst"])
async def get_revenue(quarter: str, ctx: RequestContext) -> str:
    """Get revenue data. Analyst role required."""
    user = ctx.client.client_id
    return f"Revenue for {quarter} (queried by {user}): $4.2M"
```

```python
# agent.py
from promptise import build_agent
from promptise.agent import CallerContext

agent = await build_agent(
    model="openai:gpt-4o-mini",
    servers={"api": HTTPServerSpec(url="http://localhost:8080/mcp")},
)

# Analyst can access revenue
analyst = CallerContext(
    user_id="analyst-1",
    bearer_token=generate_jwt(roles=["analyst"]),
)
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "What was Q1 revenue?"}]},
    caller=analyst,
)

# Viewer cannot access revenue (role guard blocks)
viewer = CallerContext(
    user_id="viewer-1",
    bearer_token=generate_jwt(roles=["viewer"]),
)
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "What was Q1 revenue?"}]},
    caller=viewer,
)
# → Tool call denied: insufficient role
```

## See Also

- [Building Multi-User Systems](multi-user-systems.md) — Full 12-step guide with 30-tool server
- [Secure Multi-Tenant Platform](secure-multi-tenant-platform.md) — one server, many customer orgs, provably isolated
- [MCP Authentication](../mcp/server/auth-security.md) — JWT, OAuth, API key auth deep dive
- [CallerContext API](../api/agent.md#callercontext) — API reference
