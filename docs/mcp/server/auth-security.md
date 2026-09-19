# Authentication & Security

Secure your MCP servers with JWT tokens, API keys, role-based guards, and dependency injection.

## Quick Start

```python
from promptise.mcp.server import (
    MCPServer, AuthMiddleware, JWTAuth, HasRole, Depends, LoggingMiddleware,
)

jwt_auth = JWTAuth(secret="my-secret-key")

server = MCPServer(name="secure-api")
server.add_middleware(AuthMiddleware(jwt_auth))
server.add_middleware(LoggingMiddleware())

@server.tool(auth=True, roles=["admin"])
async def delete_user(user_id: str) -> str:
    """Delete a user (admin only)."""
    return f"Deleted {user_id}"

@server.tool(auth=True)
async def list_users() -> list[str]:
    """List users (any authenticated client)."""
    return ["alice", "bob"]

@server.tool()
async def health() -> str:
    """Public health check (no auth required)."""
    return "ok"

server.run(transport="http", host="127.0.0.1", port=8080)
```

Only tools with `auth=True` require authentication. Unauthenticated clients can still call `health`.

## Concepts

The auth system has three layers:

1. **Auth providers** (`JWTAuth`, `APIKeyAuth`, `AsymmetricJWTAuth`) verify credentials and extract client identity.
2. **AuthMiddleware** runs in the middleware chain, calling the provider for tools that require auth. On success, it populates `ctx.client` with a structured `ClientContext` containing identity, roles, scopes, JWT claims, IP address, and user-agent.
3. **Guards** (`RequireAuth`, `HasRole`, `HasAllRoles`, `HasScope`, `HasAllScopes`, `RequireClientId`, `RequireTenant`, `HasTenant`) enforce fine-grained permissions after authentication. When a guard denies access, the error message explains *why* — which roles or scopes were required vs. what the client has.

After authentication, everything you need is on `ctx.client`:

```python
@server.tool(auth=True)
async def my_tool(ctx: RequestContext) -> str:
    ctx.client.client_id   # "agent-007"
    ctx.client.roles       # {"admin", "analyst"}
    ctx.client.scopes      # {"read", "write"} (from JWT "scope" claim)
    ctx.client.issuer      # "https://auth.example.com" (JWT "iss")
    ctx.client.audience    # "my-api" (JWT "aud")
    ctx.client.subject     # "agent-007" (JWT "sub")
    ctx.client.ip_address  # "192.168.1.42"
    ctx.client.user_agent  # "MCP-Client/1.0"
    ctx.client.extra       # {} (populated by on_authenticate hook)
    ctx.client.claims      # Full JWT payload dict
    return "ok"
```

## Authentication Providers

### JWTAuth

Validates HS256 JWT tokens from request metadata. Verified tokens are cached in an LRU to avoid repeated crypto operations on the hot path.

```python
from promptise.mcp.server import JWTAuth

jwt_auth = JWTAuth(
    secret="my-secret-key",   # Shared secret for HS256 verification
    meta_key="authorization", # Key in request metadata (default)
    cache_size=256,           # Max cached tokens (0 to disable)
)
```

Clients send tokens via the `Authorization: Bearer <token>` header. The provider strips the `Bearer ` prefix automatically.

**Token payload format:**

```json
{
  "sub": "agent-admin",
  "roles": ["admin", "finance"],
  "exp": 1717200000
}
```

The `sub` (or `client_id`) claim becomes `ctx.client_id`. The `roles` array is extracted by `AuthMiddleware` into `ctx.state["roles"]`.

**Utility methods:**

```python
# Create a token (for testing)
token = jwt_auth.create_token(
    {"sub": "test-agent", "roles": ["admin"]},
    expires_in=3600,  # 1 hour
)

# Verify a token without a request context
is_valid = jwt_auth.verify_token(token)
```

### APIKeyAuth

Pre-shared key authentication with optional role support. Maps API keys to client identifiers.

**Simple format** — key maps to a client ID string:

```python
from promptise.mcp.server import APIKeyAuth

api_auth = APIKeyAuth(
    keys={
        "sk-abc123": "frontend-app",
        "sk-xyz789": "backend-service",
    },
    header="x-api-key",  # Header name (default)
)
```

**Rich format** — key maps to a config dict with roles for guard compatibility:

```python
api_auth = APIKeyAuth(
    keys={
        "sk-admin-abc": {"client_id": "admin-agent", "roles": ["admin", "write"]},
        "sk-read-xyz":  {"client_id": "viewer-agent", "roles": ["read"]},
    },
)
```

Rich keys automatically populate `ctx.state["roles"]`, so role-based guards (`HasRole`, `HasAllRoles`) work out of the box — no JWT required.

Clients send their key via the `x-api-key` header (configurable via `header`). The matching client ID becomes `ctx.client_id`.

### AsymmetricJWTAuth

For production deployments using RS256 or ES256 (asymmetric) JWT tokens — common with Auth0, Keycloak, and other identity providers:

```python
from promptise.mcp.server import AsymmetricJWTAuth, AuthMiddleware

# RS256 with a PEM public key
auth = AsymmetricJWTAuth(
    public_key="-----BEGIN PUBLIC KEY-----\nMIIBI...\n-----END PUBLIC KEY-----",
    algorithm="RS256",  # or "ES256" for ECDSA
)

server.add_middleware(AuthMiddleware(auth))
```

Load the key from a file:

```python
auth = AsymmetricJWTAuth(
    public_key=open("/path/to/public.pem").read(),
    algorithm="RS256",
)
```

`AsymmetricJWTAuth` uses the same interface as `JWTAuth` — it works with `AuthMiddleware`, guards, and `ctx.client_id`. Requires the `PyJWT` and `cryptography` packages (optional dependencies).

### JwksAuth

For tokens issued by an identity provider whose signing keys rotate and are published at a **JWKS** endpoint — Microsoft Entra, Okta, Auth0, Keycloak, any OIDC provider. This is the server-side counterpart to [Agent Identity](../../identity/overview.md): an agent presents an IdP-issued identity credential, and `JwksAuth` verifies it against the IdP's public keys (fetched on demand and cached, so key rotation needs no reconfiguration):

```python
from promptise.mcp.server import JwksAuth, AuthMiddleware

auth = JwksAuth(
    jwks_url="https://login.microsoftonline.com/<tenant>/discovery/v2.0/keys",
    issuer="https://login.microsoftonline.com/<tenant>/v2.0",
    audience="api://my-mcp-server",   # the resource agents target
)
server.add_middleware(AuthMiddleware(auth))
```

The validated token's `sub` and claims are surfaced on `ctx.client` (subject, issuer, audience, claims), so guards (`RequireClientId`, `HasRole`) and audit logs can see **which agent** called. Requires the `PyJWT` and `cryptography` packages.

!!! note "`audience` is required"
    `JwksAuth` verifies the `aud` claim on every token. This is what stops an agent from replaying a token it was legitimately issued for a *different* resource of the same IdP. Set `audience` to the resource this server represents.

You can let the JWKS endpoint be **discovered** from the issuer's OIDC metadata (`{issuer}/.well-known/openid-configuration`) instead of passing `jwks_url`:

```python
auth = JwksAuth.from_discovery(
    issuer="https://login.microsoftonline.com/<tenant>/v2.0",
    audience="api://my-mcp-server",
)
```

The discovery document is fetched once (lazily, on the first verified call) and its `issuer` is checked against the configured one before its `jwks_uri` is trusted.

### Custom auth provider

Implement the `AuthProvider` protocol for custom authentication (OAuth2 introspection, mTLS, etc.):

```python
from promptise.mcp.server import AuthProvider, RequestContext

class OAuth2Introspection:
    async def authenticate(self, ctx: RequestContext) -> str:
        token = ctx.meta.get("authorization", "").removeprefix("Bearer ")
        # Call your IdP's introspection endpoint
        claims = await introspect_token(token)
        ctx.state["jwt_payload"] = claims  # For role extraction
        return claims["sub"]
```

## AuthMiddleware

`AuthMiddleware` wraps an auth provider and runs it for tools marked with `auth=True`:

```python
from promptise.mcp.server import AuthMiddleware

server.add_middleware(AuthMiddleware(jwt_auth))
```

When a tool has `auth=True`:

1. The middleware calls `provider.authenticate(ctx)`.
2. On success, `ctx.client_id` is set and `ctx.client` is populated with a full `ClientContext` — identity, roles, scopes, JWT claims (`iss`, `aud`, `sub`, `iat`, `exp`), client IP address, and user-agent.
3. On failure, an `AuthenticationError` is returned to the client.

Tools without `auth=True` pass through the middleware without authentication checks.

### Identity is resolved per request

On the HTTP and SSE transports an MCP session spans many HTTP requests. The credential the provider verifies, and everything `ctx.client` reports (client id, tenant, roles, IP address, user-agent), come from the HTTP request that carries **this** `tools/call` — never from the request that opened the session:

- A call that presents a different valid credential than the one that opened the session runs as *that* credential's principal, with its own tenant and roles.
- A call with no credential is unauthenticated, even inside a session opened with a valid one — `auth=True` tools answer `AuthenticationError`.
- With `MCPServer(require_auth=True)` the transport gate goes one step further and binds the session to the credential that created it: a different credential on a known `mcp-session-id` is answered `404 Session not found`. See [Transport-level auth gate](deployment.md#transport-level-auth-gate).

`TestClient` behaves the same way: headers passed to one `call_tool` apply to that call only.

### Client enrichment hook

Use `on_authenticate` to load additional client metadata (org, tenant, plan tier) from your database after authentication:

```python
async def enrich_client(client: ClientContext, ctx: RequestContext):
    """Called after successful authentication."""
    org = await db.get_org_for_client(client.client_id)
    client.extra["org_id"] = org.id
    client.extra["plan"] = org.plan
    client.extra["rate_limit_tier"] = org.rate_limit_tier

server.add_middleware(AuthMiddleware(jwt_auth, on_authenticate=enrich_client))
```

The hook receives `(ClientContext, RequestContext)` and can be sync or async. Mutate `client.extra` to attach custom metadata that's available to all handlers and middleware for the rest of the request.

```python
@server.tool(auth=True)
async def my_tool(ctx: RequestContext) -> str:
    org_id = ctx.client.extra["org_id"]
    plan = ctx.client.extra["plan"]
    return f"Handling request for org {org_id} on {plan} plan"
```

### ClientContext

After successful authentication, `ctx.client` is a `ClientContext` dataclass with typed fields:

| Field | Type | Source |
|---|---|---|
| `client_id` | `str` | JWT `sub` or API key mapping |
| `roles` | `set[str]` | JWT `roles` claim + provider roles |
| `scopes` | `set[str]` | JWT `scope` claim (space-separated per RFC 8693) |
| `claims` | `dict` | Full JWT payload (empty for API key auth) |
| `issuer` | `str \| None` | JWT `iss` claim |
| `audience` | `str \| list \| None` | JWT `aud` claim |
| `subject` | `str \| None` | JWT `sub` claim |
| `issued_at` | `float \| None` | JWT `iat` claim (Unix timestamp) |
| `expires_at` | `float \| None` | JWT `exp` claim (Unix timestamp) |
| `ip_address` | `str \| None` | Client IP from transport layer |
| `user_agent` | `str \| None` | `User-Agent` header |
| `extra` | `dict` | Custom metadata from `on_authenticate` hook |

Helper methods: `has_role()`, `has_any_role()`, `has_all_roles()`, `has_scope()`, `has_any_scope()`.

### Force auth on all tools

Set `require_auth=True` on the server constructor to force `auth=True` on every tool, including those registered without the flag:

```python
server = MCPServer(name="locked-down", require_auth=True)
```

## Guards

Guards check fine-grained permissions after authentication. They run **after** middleware (so `AuthMiddleware` has already populated `ctx.client_id` and `ctx.state["roles"]`).

### RequireAuth

Denies access if `ctx.client_id` is not set:

```python
from promptise.mcp.server import RequireAuth

@server.tool(auth=True, guards=[RequireAuth()])
async def protected() -> str:
    return "authenticated"
```

### HasRole

Grants access if the client has **any** of the specified roles:

```python
from promptise.mcp.server import HasRole

@server.tool(auth=True, guards=[HasRole("admin", "manager")])
async def manage_team() -> str:
    return "team managed"
```

The `roles=["admin", "manager"]` shorthand on `@server.tool()` creates a `HasRole` guard automatically:

```python
# These two are equivalent:
@server.tool(auth=True, roles=["admin", "manager"])
@server.tool(auth=True, guards=[HasRole("admin", "manager")])
```

### HasAllRoles

Grants access only if the client has **all** of the specified roles:

```python
from promptise.mcp.server import HasAllRoles

@server.tool(auth=True, guards=[HasAllRoles("admin", "finance")])
async def approve_budget() -> str:
    return "budget approved"
```

### RequireClientId

Grants access only to specific client identifiers:

```python
from promptise.mcp.server import RequireClientId

@server.tool(auth=True, guards=[RequireClientId("cron-service", "admin-agent")])
async def run_migration() -> str:
    return "migration complete"
```

### HasScope

Grants access if the client has **any** of the specified OAuth2 scopes (from the JWT `scope` claim):

```python
from promptise.mcp.server import HasScope

@server.tool(auth=True, guards=[HasScope("read", "admin")])
async def get_data() -> str:
    return "data"
```

!!! note "JWT only"
    Scopes are extracted from the JWT `scope` claim. When using `APIKeyAuth` without JWT, `ctx.client.scopes` will be empty and scope guards will always deny. Use role-based guards (`HasRole`) for API key auth.

### HasAllScopes

Grants access only if the client has **all** of the specified scopes:

```python
from promptise.mcp.server import HasAllScopes

@server.tool(auth=True, guards=[HasAllScopes("read", "write")])
async def update_data() -> str:
    return "updated"
```

### Descriptive guard errors

When a guard denies access, the error message explains *why*:

```
# HasRole denial:
"Requires any of roles [admin, manager], but client has [viewer]"

# HasAllRoles denial:
"Requires all roles [admin, analyst, ops], client has [admin], missing [analyst, ops]"

# HasScope denial:
"Requires any of scopes [read, write], but client has [(none)]"
```

This makes debugging auth issues straightforward — you see exactly which roles or scopes are required, which the client has, and what's missing.

### Custom guards

Implement the `Guard` protocol. Override `describe_denial()` to provide helpful error messages:

```python
from promptise.mcp.server import Guard, RequestContext

class IPAllowlist(Guard):
    def __init__(self, allowed_ips: set[str]):
        self._allowed = allowed_ips

    async def check(self, ctx: RequestContext) -> bool:
        return ctx.client.ip_address in self._allowed

    def describe_denial(self, ctx: RequestContext) -> str:
        return (
            f"Client IP '{ctx.client.ip_address}' is not in the "
            f"allowlist [{', '.join(sorted(self._allowed))}]"
        )
```

## Request Tracing

Every request gets a unique `request_id`. If the client sends an `X-Request-ID` header, that value is used; otherwise one is generated automatically. The header is read from the HTTP request carrying each call, so every `tools/call` in a session carries its own id. This ID is available to all middleware, handlers, and audit logging:

```python
@server.tool()
async def my_tool(ctx: RequestContext) -> str:
    ctx.logger.info("Processing", extra={"request_id": ctx.request_id})
    return "done"
```

Propagate the `X-Request-ID` from your upstream services to enable end-to-end distributed tracing through your MCP tools.

## ToolResponse — response metadata

Handlers can return a `ToolResponse` to attach metadata for audit, observability, and middleware:

```python
from promptise.mcp.server import ToolResponse

@server.tool()
async def search(query: str, ctx: RequestContext) -> ToolResponse:
    results = await db.search(query)
    return ToolResponse(
        content=results,
        metadata={
            "source": "primary_db",
            "result_count": len(results),
            "cache": "miss",
        },
    )
```

The `content` is serialized normally (the agent sees the results). The `metadata` is stored on `ctx.state["response_metadata"]` for downstream use by audit middleware, webhook middleware, or custom logging. The agent never sees the metadata — it's for your infrastructure.

## Dependency Injection

Use `Depends()` to inject shared resources into tool handlers -- database connections, HTTP clients, configuration, or any callable.

### Basic injection

```python
from promptise.mcp.server import Depends

async def get_db():
    db = await Database.connect()
    try:
        yield db       # Yielded value is injected
    finally:
        await db.close()  # Cleanup runs after the handler

@server.tool()
async def query(sql: str, db: Database = Depends(get_db)) -> list:
    """Run a SQL query."""
    return await db.execute(sql)
```

### Supported dependency types

| Type | Example | Lifecycle |
|---|---|---|
| Async generator | `async def dep(): yield val` | Cleanup after handler |
| Sync generator | `def dep(): yield val` | Cleanup after handler |
| Async callable | `async def dep(): return val` | Called once per request |
| Sync callable | `def dep(): return val` | Called once per request |
| Class | `Depends(MySettings)` | Constructor called per request |

### Request-scoped caching

By default, the same dependency is resolved once per request (cached by identity). Disable with `use_cache=False`:

```python
# Resolved once per request (default):
db: Database = Depends(get_db, use_cache=True)

# Resolved fresh each time (e.g., for unique IDs):
request_id: str = Depends(generate_id, use_cache=False)
```

### Injecting RequestContext

Handler parameters typed as `RequestContext` are injected automatically:

```python
from promptise.mcp.server import RequestContext

@server.tool(auth=True)
async def whoami(ctx: RequestContext) -> dict:
    """Return the authenticated client's identity."""
    return {
        "client_id": ctx.client.client_id,
        "roles": sorted(ctx.client.roles),
        "scopes": sorted(ctx.client.scopes),
        "issuer": ctx.client.issuer,
        "ip": ctx.client.ip_address,
    }
```

## Token Endpoint (Dev/Testing)

For development and testing, enable a built-in `/auth/token` endpoint that issues JWT tokens:

```python
jwt_auth = JWTAuth(secret="dev-secret")
server.add_middleware(AuthMiddleware(jwt_auth))

server.enable_token_endpoint(
    jwt_auth=jwt_auth,
    clients={
        "agent-admin":  {"secret": "admin-secret",  "roles": ["admin", "finance"]},
        "agent-viewer": {"secret": "viewer-secret", "roles": ["viewer"]},
    },
    path="/auth/token",           # HTTP path (default)
    default_expires_in=86400,     # Token lifetime in seconds (default: 24h)
)
```

Clients request tokens via HTTP POST:

```bash
curl -X POST http://localhost:8080/auth/token \
  -H "Content-Type: application/json" \
  -d '{"client_id": "agent-admin", "client_secret": "admin-secret"}'
```

Response:

```json
{"access_token": "eyJhbGciOiJIUzI1NiIs...", "token_type": "bearer", "expires_in": 86400}
```

!!! warning "Not for production"
    The built-in token endpoint is a convenience for development. In production, use a proper Identity Provider (Auth0, Keycloak, Okta) and pass tokens to clients via `bearer_token`.

## Complete Auth Setup

A full example combining JWT auth, guards, dependency injection, and the token endpoint:

```python
from promptise.mcp.server import (
    MCPServer, MCPRouter, AuthMiddleware, JWTAuth,
    LoggingMiddleware, TimeoutMiddleware, ConcurrencyLimiter,
    BackgroundTasks, Depends, RequestContext, get_context,
)

server = MCPServer(name="hr-api", version="1.0.0")

# Auth
jwt_auth = JWTAuth(secret="production-secret")
server.add_middleware(AuthMiddleware(jwt_auth))
server.add_middleware(LoggingMiddleware())
server.add_middleware(TimeoutMiddleware(default_timeout=30.0))
server.add_middleware(ConcurrencyLimiter(max_concurrent=50))

# Dev token endpoint
server.enable_token_endpoint(
    jwt_auth=jwt_auth,
    clients={
        "agent-admin":  {"secret": "s3cret", "roles": ["admin"]},
        "agent-viewer": {"secret": "v1ewer", "roles": ["viewer"]},
    },
)

# Public tool (no auth)
@server.tool()
async def ping() -> str:
    """Health check."""
    return "pong"

# Read tools (any authenticated user)
@server.tool(auth=True, roles=["admin", "viewer"])
async def list_items() -> list[str]:
    """List all items."""
    return ["item-1", "item-2"]

# Write tools (admin only, with background tasks)
@server.tool(auth=True, roles=["admin"])
async def delete_item(
    item_id: str,
    bg: BackgroundTasks = Depends(BackgroundTasks),
) -> dict:
    """Delete an item (admin only)."""
    ctx = get_context()
    bg.add(log_audit, "DELETE", ctx.client_id, item_id)
    return {"deleted": item_id}

async def log_audit(action: str, client: str, target: str) -> None:
    print(f"AUDIT: {action} by {client} on {target}")

if __name__ == "__main__":
    server.run(transport="http", host="127.0.0.1", port=8080)
```

## Connecting from Promptise Agents

When a Promptise agent connects to your MCP server, identity flows automatically:

```python
# Agent side — CallerContext carries the JWT
from promptise.agent import CallerContext

caller = CallerContext(
    user_id="user-42",
    bearer_token=jwt_token,  # This gets sent to your server
)
result = await agent.ainvoke(input, caller=caller)
```

The agent's `MCPClient` sends the `bearer_token` as an `Authorization: Bearer <token>` header. Your server's `AuthMiddleware` validates it and populates `ctx.client` — the handler never sees the raw token.

```python
# Server side — handler receives validated identity
@server.tool(auth=True, roles=["analyst"])
async def query_data(sql: str, ctx: RequestContext) -> str:
    # ctx.client.client_id = "user-42" (from JWT 'sub' claim)
    # ctx.client.roles = {"analyst"} (from JWT 'roles' claim)
    # The raw bearer_token is NOT accessible here — only parsed claims
    return await db.query(sql, user_id=ctx.client.client_id)
```

| Agent side (CallerContext) | Wire | Server side (ClientContext) |
|---|---|---|
| `bearer_token` | → `Authorization: Bearer` header → | JWT validated, claims extracted |
| `user_id` | *(not sent)* | Extracted from JWT `sub` claim |
| `roles` | *(not sent)* | Extracted from JWT `roles` claim |
| `scopes` | *(not sent)* | Extracted from JWT `scope` claim |

See [CallerContext: Agent to MCP Identity](../../guides/multi-user-identity.md) for the complete flow with examples.

## API Summary

| Symbol | Type | Description |
|---|---|---|
| `JWTAuth(secret, meta_key, cache_size)` | Class | HS256 JWT authentication provider |
| `JWTAuth.create_token(payload, expires_in)` | Method | Create a signed JWT (testing utility) |
| `JWTAuth.verify_token(token)` | Method | Check token validity without context |
| `APIKeyAuth(keys, header)` | Class | Pre-shared API key authentication (simple or rich format) |
| `AsymmetricJWTAuth(public_key, algorithm)` | Class | RS256/ES256 asymmetric JWT authentication |
| `AuthMiddleware(provider, on_authenticate)` | Class | Middleware that enforces auth and populates `ClientContext` |
| `ClientContext` | Dataclass | Structured client info: identity, roles, scopes, JWT claims, IP, user-agent |
| `ToolResponse(content, metadata)` | Dataclass | Response wrapper with metadata for audit/observability |
| `RequireAuth()` | Guard | Require any authenticated client |
| `HasRole(*roles)` | Guard | Require **any** of the given roles |
| `HasAllRoles(*roles)` | Guard | Require **all** of the given roles |
| `HasScope(*scopes)` | Guard | Require **any** of the given OAuth2 scopes (JWT only) |
| `HasAllScopes(*scopes)` | Guard | Require **all** of the given OAuth2 scopes (JWT only) |
| `RequireClientId(*ids)` | Guard | Require specific client identifiers |
| `Depends(dependency, use_cache)` | Function | Dependency injection marker |
| `server.enable_token_endpoint(...)` | Method | Enable built-in dev token endpoint |

!!! tip "Guard composition"
    Pass multiple guards to a tool's `guards` list. They are checked in order; the first failure short-circuits and returns an `ACCESS_DENIED` error.

!!! warning "Secret management"
    Never hardcode JWT secrets in source code. Use environment variables or a secrets manager. The `ServerSettings` class supports loading secrets from env vars with a configurable prefix.

!!! tip "Token caching"
    `JWTAuth` caches verified tokens in a thread-safe LRU (default: 256 entries). This avoids repeated HMAC-SHA256 computation when the same token is reused across requests, which is the common case for agent sessions.

## What's Next

- [Caching & Performance](caching-performance.md) — Cache tool results, rate limit agents, control concurrency
- [Observability & Monitoring](observability.md) — Metrics, tracing, audit trails, structured logging
- [Resilience Patterns](resilience-patterns.md) — Circuit breakers, health checks, background tasks
- [Deployment](deployment.md) — HTTP transport, CORS, transport-level auth, Docker
- [Testing](testing.md) — Test authenticated endpoints with `TestClient`
- [Client Guide](../client/index.md) — Connect to your server with authentication
