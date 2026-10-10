# Production Features

Everything you need to run MCP servers in production — caching, rate limiting, health checks, metrics, observability, resilience, and advanced patterns. Each feature has a dedicated deep-dive page with real-world examples and full configuration reference.

## Quick Start

```python
from promptise.mcp.server import (
    MCPServer, AuthMiddleware, JWTAuth,
    LoggingMiddleware, TimeoutMiddleware, ConcurrencyLimiter,
    RateLimitMiddleware, MetricsMiddleware, MetricsCollector,
    CircuitBreakerMiddleware, AuditMiddleware,
    HealthCheck, InMemoryCache, cached,
)

server = MCPServer(name="production-api", version="1.0.0")

# Auth
jwt = JWTAuth(secret="your-secret-key")
server.add_middleware(AuthMiddleware(jwt))

# Observability
metrics = MetricsCollector()
server.add_middleware(MetricsMiddleware(metrics))
server.add_middleware(AuditMiddleware(log_path="audit.jsonl", signed=True))  # key: PROMPTISE_AUDIT_SECRET

# Protection
server.add_middleware(LoggingMiddleware())
server.add_middleware(RateLimitMiddleware(rate_per_minute=120))
server.add_middleware(TimeoutMiddleware(default_timeout=30.0))
server.add_middleware(ConcurrencyLimiter(max_concurrent=50))
server.add_middleware(CircuitBreakerMiddleware(failure_threshold=5))

# Health checks
health = HealthCheck()
health.add_check("database", check_db, required_for_ready=True)
health.register_resources(server)
metrics.register_resource(server)

@server.tool(auth=True)
@cached(ttl=300)
async def expensive_query(query: str) -> dict:
    """Run an expensive query (cached for 5 minutes)."""
    return await db.search(query)

# Listen on every interface (the default is 127.0.0.1) and name the public host
server.run(
    transport="http",
    host="0.0.0.0",
    port=8080,
    allowed_hosts=["mcp.example.com"],
    dashboard=True,
)
```

`run()` binds `127.0.0.1` unless you pass `host`, so a server is reachable from the same machine only until you choose otherwise. Bind `0.0.0.0` deliberately — in a container, or behind a load balancer — together with `AuthMiddleware` (or an authenticating gateway) and `allowed_hosts`; a non-loopback bind without them logs a warning at startup. See [Deployment](deployment.md#host-and-origin-validation).

## Feature Guide

Choose the right page for what you need:

| I want to... | Page |
|---|---|
| Cache tool results, rate limit agents, control concurrency | [Caching & Performance](caching-performance.md) |
| Add metrics, tracing, Prometheus, structured logging, audit trails | [Observability & Monitoring](observability.md) |
| Protect against failures, add health checks, webhooks, background tasks | [Resilience Patterns](resilience-patterns.md) |
| Version tools, transform tool lists, compose servers, bridge OpenAPI | [Advanced Patterns](advanced-patterns.md) |
| Deploy with HTTP/CORS, Host/Origin validation, Docker, reverse proxy, CLI serve | [Deployment](deployment.md) |
| Add JWT auth, API keys, guards, roles | [Authentication & Security](auth-security.md) |
| Test with TestClient, pytest fixtures, CI | [Testing](testing.md) |

## Middleware Ordering

For production servers, apply middleware in this order (outermost to innermost):

```python
# 1. Dashboard (auto-inserted when dashboard=True)
# 2. Audit logging — capture everything
server.add_middleware(AuditMiddleware(log_path="audit.jsonl"))  # key: PROMPTISE_AUDIT_SECRET
# 3. Webhooks — alert on errors
server.add_middleware(WebhookMiddleware(url="https://hooks.slack.com/..."))
# 4. Circuit breaker — fail fast for broken dependencies
server.add_middleware(CircuitBreakerMiddleware(failure_threshold=5))
# 5. Auth — reject unauthenticated requests
server.add_middleware(AuthMiddleware(jwt))
# 6. Metrics — track call counts and latency
server.add_middleware(MetricsMiddleware(metrics))
# 7. Logging — log every request
server.add_middleware(LoggingMiddleware())
# 8. Rate limiting — prevent abuse
server.add_middleware(RateLimitMiddleware(rate_per_minute=100))
# 9. Timeouts — prevent runaway calls
server.add_middleware(TimeoutMiddleware(default_timeout=30.0))
# 10. Concurrency — protect server capacity
server.add_middleware(ConcurrencyLimiter(max_concurrent=50))
```

!!! tip "Auth must be before guards"
    `AuthMiddleware` must run before any tool guards (`HasRole`, `RequireAuth`) because guards read `ctx.client_id` and `ctx.state["roles"]`, which are set by the auth middleware.

## Deep-Dive Pages

### [Caching & Performance](caching-performance.md)

`@cached` decorator, `InMemoryCache`, `RedisCache`, `CacheMiddleware`, `RateLimitMiddleware`, `TokenBucketLimiter`, `ConcurrencyLimiter`, `PerToolConcurrencyLimiter`, `TimeoutMiddleware`.

### [Observability & Monitoring](observability.md)

`MetricsCollector`, `MetricsMiddleware`, `Dashboard`, `OTelMiddleware`, `PrometheusMiddleware`, `StructuredLoggingMiddleware`, `AuditMiddleware`, `ServerLogger`.

### [Resilience Patterns](resilience-patterns.md)

`CircuitBreakerMiddleware`, `HealthCheck`, `WebhookMiddleware`, `BackgroundTasks`, `ExceptionHandlerRegistry`, `ProgressReporter`, `CancellationToken`.

### [Advanced Patterns](advanced-patterns.md)

`VersionedToolRegistry`, `NamespaceTransform`, `VisibilityTransform`, `TagFilterTransform`, `mount()`, `OpenAPIProvider`, `register_batch_tool()`, `StreamingResult`, `Elicitor`, `Sampler`, `register_manifest()`, `hot_reload()`, CLI serve.

### [Deployment](deployment.md)

Transport selection (stdio/HTTP/SSE), Host/Origin validation (`allowed_hosts` / `allowed_origins`, on by default for loopback binds), per-request caller identity, `CORSConfig`, reverse proxy, Docker, Kubernetes health probes, `TokenEndpointConfig`, `hot_reload()`, CLI `serve` command.

## What's Next

- [Caching & Performance](caching-performance.md) — start here for the most commonly needed production features
- [Authentication & Security](auth-security.md) — if you haven't set up auth yet
- [Testing](testing.md) — test your production server with `TestClient`
