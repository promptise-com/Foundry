# Routers & Middleware

Organize tools into modular routers and add cross-cutting concerns like logging and timeouts via middleware.

## Quick Start

```python
from promptise.mcp.server import MCPServer, MCPRouter, LoggingMiddleware, TimeoutMiddleware

# Create a router for database tools
db_router = MCPRouter(prefix="db", tags=["database"])

@db_router.tool()
async def query(sql: str) -> list:
    """Execute a read-only SQL query."""
    return await run_query(sql)

@db_router.tool()
async def tables() -> list[str]:
    """List all table names."""
    return ["users", "orders", "products"]

# Assemble the server
server = MCPServer(name="api")
server.add_middleware(LoggingMiddleware())
server.add_middleware(TimeoutMiddleware(default_timeout=30.0))
server.include_router(db_router)

server.run()
```

After mounting, the tools are registered as `db_query` and `db_tables` (prefix + underscore + function name).

## Concepts

**MCPRouter** groups related tools, resources, and prompts into a reusable module -- analogous to FastAPI's `APIRouter`. Routers carry their own prefix, tags, auth defaults, middleware, and guards. When mounted on a server via `include_router()`, all registrations are merged into the server's registries.

**Middleware** intercepts every tool call, enabling cross-cutting concerns like logging, timeouts, authentication, and rate limiting. Middleware runs in registration order (first added = outermost), wrapping the handler like layers of an onion.

## Routers

### Creating a router

```python
from promptise.mcp.server import MCPRouter

analytics_router = MCPRouter(
    prefix="analytics",    # Prepended to tool names: "analytics_<name>"
    tags=["reporting"],    # Default tags merged with per-tool tags
    auth=True,             # Override: force auth on all tools in this router
    guards=[RequireAuth()],  # Router-level guards applied to all tools
)
```

### Registering tools on a router

Routers expose the same decorator API as `MCPServer`:

```python
@analytics_router.tool()
async def daily_report(date: str) -> dict:
    """Generate a daily analytics report."""
    return {"date": date, "views": 1500}

@analytics_router.tool(tags=["admin"], roles=["admin"])
async def reset_counters() -> str:
    """Reset all analytics counters (admin only)."""
    return "Counters reset"
```

### Resources and prompts on routers

```python
@analytics_router.resource("analytics://config", mime_type="application/json")
async def analytics_config() -> str:
    return '{"retention_days": 90}'

@analytics_router.prompt()
async def report_prompt(metric: str) -> str:
    """Generate a prompt for analyzing a metric."""
    return f"Analyze the trend for {metric} over the last 30 days."
```

URIs and prompt names are not prefixed. The router's `auth`, `guards`, `middleware` and `tags` apply to its resources and prompts exactly as they do to its tools.

### Mounting routers on the server

```python
server = MCPServer(name="platform")
server.include_router(analytics_router)

# Override prefix or add extra tags at mount time
server.include_router(analytics_router, prefix="v2_analytics", tags=["v2"])
```

### Nesting routers

Routers can include other routers for deeper module hierarchies:

```python
reports_router = MCPRouter(prefix="reports")
finance_router = MCPRouter(prefix="finance")

@finance_router.tool()
async def revenue() -> dict:
    return {"total": 1_000_000}

reports_router.include_router(finance_router)
server.include_router(reports_router)
# Tool name: "reports_finance_revenue"
```

### Prefix mechanics

Prefixes are joined with underscores. Given:

- Server `include_router(router, prefix="v1")`
- Router `MCPRouter(prefix="users")`
- Tool function name `search`

The final tool name is `v1_users_search`.

Resource URIs are **not** prefixed -- they use the URI you specify.

## Middleware

### How middleware works

A middleware is any async callable matching this signature:

```python
async def my_middleware(ctx: RequestContext, call_next) -> Any:
    # Pre-processing (before the handler runs)
    print(f"Calling {ctx.tool_name}")
    result = await call_next(ctx)
    # Post-processing (after the handler returns)
    print(f"Result: {result}")
    return result
```

`call_next` invokes the next middleware in the chain, or the handler itself at the innermost layer.

### Adding middleware

```python
server.add_middleware(LoggingMiddleware())
server.add_middleware(TimeoutMiddleware(default_timeout=15.0))
```

Or use the decorator form:

```python
@server.middleware
async def track_calls(ctx, call_next):
    ctx.state["start"] = time.time()
    result = await call_next(ctx)
    elapsed = time.time() - ctx.state["start"]
    print(f"{ctx.tool_name} took {elapsed:.3f}s")
    return result
```

The chain runs for tool calls, resource reads and prompt requests alike. `ctx.request_type` says which (`"tool"`, `"resource"` or `"prompt"`), `ctx.tool_name` holds the tool, resource or prompt name, and a resource read sets `ctx.state["resource_uri"]`. Branch on `ctx.request_type` in a middleware that only makes sense for tools.

### Built-in middleware

#### LoggingMiddleware

Logs every tool call with timing information.

```python
from promptise.mcp.server import LoggingMiddleware

server.add_middleware(LoggingMiddleware(log_level=logging.INFO))
```

Output: `[req-abc123] search completed in 0.045s`

#### TimeoutMiddleware

Enforces per-call timeouts. Uses the tool's `timeout` setting if present, otherwise falls back to `default_timeout`.

```python
from promptise.mcp.server import TimeoutMiddleware

server.add_middleware(TimeoutMiddleware(default_timeout=30.0))
```

If a tool exceeds the timeout, a retryable `ToolError` with code `TIMEOUT` is returned to the client.

#### ConcurrencyLimiter

Prevents resource exhaustion by capping the number of in-flight tool calls. Excess requests receive a retryable `RateLimitError`.

```python
from promptise.mcp.server import ConcurrencyLimiter

limiter = ConcurrencyLimiter(max_concurrent=50)
server.add_middleware(limiter)

# Inspect stats at runtime
print(limiter.active_requests)   # Currently in-flight
print(limiter.peak_concurrent)   # Highest observed
print(limiter.total_requests)    # Total since server start
```

### Writing custom middleware

Implement the middleware protocol as a class:

```python
class AuditMiddleware:
    """Log every tool call to an audit trail."""

    def __init__(self, audit_log: list):
        self._log = audit_log

    async def __call__(self, ctx, call_next):
        self._log.append({
            "tool": ctx.tool_name,
            "client": ctx.client_id,
            "request_id": ctx.request_id,
        })
        return await call_next(ctx)

audit_trail = []
server.add_middleware(AuditMiddleware(audit_trail))
```

### Middleware ordering

Middleware runs in **registration order** -- the first middleware added is the outermost layer:

```python
server.add_middleware(AuthMiddleware(jwt_auth))   # 1. Runs first (outermost)
server.add_middleware(LoggingMiddleware())         # 2. Runs second
server.add_middleware(TimeoutMiddleware())         # 3. Runs third (innermost)
```

The execution flow for a tool call:

```
AuthMiddleware  ->  LoggingMiddleware  ->  TimeoutMiddleware  ->  handler
     <-                  <-                    <-                  <-
```

### Router-level middleware

Routers can carry their own middleware that runs **after** server-level middleware:

```python
db_router = MCPRouter(prefix="db", middleware=[TimeoutMiddleware(default_timeout=5.0)])
```

The combined chain for tools on this router is: server middleware, then router middleware, then the handler.

## RequestContext

Every middleware and handler receives a `RequestContext` with:

| Attribute | Type | Description |
|---|---|---|
| `server_name` | `str` | Name of the MCP server |
| `tool_name` | `str` | Name of the tool being called |
| `request_id` | `str` | Unique ID for this request |
| `client_id` | `str \| None` | Authenticated client identifier |
| `meta` | `dict` | Request metadata (headers, tokens) |
| `state` | `dict` | Mutable state shared across the middleware chain |
| `logger` | `Logger` | Pre-configured logger for this request |

## API Summary

| Symbol | Type | Description |
|---|---|---|
| `MCPRouter(prefix, tags, auth, middleware, guards)` | Class | Modular tool/resource/prompt grouping |
| `router.tool(...)` | Decorator | Register a tool on the router |
| `router.resource(uri, ...)` | Decorator | Register a resource on the router |
| `router.resource_template(uri, ...)` | Decorator | Register a resource template on the router |
| `router.prompt(...)` | Decorator | Register a prompt on the router |
| `router.include_router(sub)` | Method | Nest a sub-router |
| `server.include_router(router)` | Method | Mount a router on the server |
| `server.add_middleware(mw)` | Method | Append middleware to the chain |
| `@server.middleware` | Decorator | Register a middleware function |
| `LoggingMiddleware(log_level)` | Class | Log tool calls with timing |
| `TimeoutMiddleware(default_timeout)` | Class | Enforce per-call timeouts |
| `ConcurrencyLimiter(max_concurrent)` | Class | Cap in-flight tool calls |

!!! tip "Pre-compiled middleware chains"
    Middleware chains are compiled once at server startup into frozen closures. This eliminates per-request overhead from chain construction, making middleware essentially zero-cost in the hot path.

!!! warning "Middleware order matters"
    Authentication middleware must run **before** any middleware that reads `ctx.client_id` or `ctx.state["roles"]`. Place `AuthMiddleware` first in your chain.

!!! tip "Router prefix convention"
    Use short, lowercase prefixes that describe the domain: `db`, `users`, `reports`. The underscore separator is added automatically.

## What's Next?

- [Authentication & Security](auth-security.md) -- Add JWT, API keys, and role-based guards
- [Production Features](production-features.md) -- Add caching, rate limiting, health checks, and metrics
- [Testing](testing.md) -- Test servers in-process with `TestClient`
- [Client Guide](../client/index.md) -- Connect to your server from Python code
