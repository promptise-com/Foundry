# Caching & Performance

Control costs, protect downstream services, and keep your server responsive with caching, rate limiting, concurrency limits, and timeouts.

## Caching

### The problem

Your MCP server wraps an external API that charges per call. An agent asks the same question five times in a conversation — without caching, you pay five times and add five round trips of latency.

### `@cached` decorator

Cache individual tool results by argument hash. The first call executes the function; subsequent calls with the same arguments return the cached result until the TTL expires.

```python
from promptise.mcp.server import MCPServer, cached, InMemoryCache

server = MCPServer(name="weather-api")
cache = InMemoryCache(max_size=500)

@server.tool(read_only_hint=True)
@cached(ttl=300, backend=cache, scope="shared")  # same answer for every caller
async def get_weather(city: str, units: str = "metric") -> dict:
    """Get current weather for a city (cached for 5 minutes).

    Calls the OpenWeatherMap API. Each uncached call costs ~$0.001.
    Caching saves money and reduces latency from ~200ms to <1ms.
    """
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "units": units, "appid": API_KEY},
        )
        resp.raise_for_status()
        data = resp.json()
    return {
        "city": city,
        "temp": data["main"]["temp"],
        "description": data["weather"][0]["description"],
        "humidity": data["main"]["humidity"],
    }
```

**How cache keys work**: The key is the server, the function, the tool's arguments and the caller (see [Who shares a cached result](#who-shares-a-cached-result)). So `get_weather("London", "metric")` and `get_weather("London", "imperial")` are separate cache entries. Parameters the framework injects — `ctx: RequestContext`, `Depends(...)` values, `BackgroundTasks` — are not part of the key.

**Custom key functions**: Override how the arguments map to a key when you need control. The function receives every keyword argument the handler gets, and its key is still prefixed with the server and the scope, so it cannot widen sharing beyond `scope`:

```python
def ignore_page(func_name: str, args: dict) -> str:
    """One entry per query, whatever the page."""
    return f"{func_name}:{args.get('query', '')}"

@server.tool()
@cached(ttl=60, backend=cache, key_func=ignore_page)
async def search_documents(query: str, page: int = 1) -> list[dict]:
    """Search docs — cached per caller, ignoring page number."""
    return await doc_store.search(query, page=page)
```

### Who shares a cached result

A tool's result often depends on who asks: the same `list_accounts()` call
returns Acme's accounts to Acme and Globex's to Globex. Both `@cached` and
`CacheMiddleware` therefore key every entry on the authenticated caller by
default. The `scope` argument widens that, and you opt in explicitly:

| `scope` | Entries shared by | Use for |
|---------|-------------------|---------|
| `"client"` *(default)* | The same principal: issuer, tenant and client id | Anything that depends on the caller. Always safe |
| `"tenant"` | Every client of the same tenant | Tenant-wide data that is the same for every user in the tenant |
| `"shared"` | Everyone | Data that is the same for every caller: public reference data, weather, exchange rates |

```python
@server.tool(auth=True)
@cached(ttl=300)                       # per caller
async def my_open_tickets(ctx: RequestContext) -> list[dict]:
    return await db.tickets(owner=ctx.client.client_id)

@server.tool(auth=True)
@cached(ttl=300, scope="tenant")       # one entry per tenant
async def org_holidays(ctx: RequestContext) -> list[str]:
    return await db.holidays(org=ctx.client.tenant_id)
```

The scope comes from the request, not from the handler's parameters, so a
handler that reads the caller with `get_context()` is isolated just like
one that takes `ctx`. Unauthenticated callers all share the `anonymous`
identity. Outside a request (a handler called directly in a test), nothing
is shared with request traffic.

### `InMemoryCache`

In-process cache with TTL expiry and optional LRU eviction.

```python
from promptise.mcp.server import InMemoryCache

cache = InMemoryCache(
    max_size=1000,          # Evict oldest when full (0 = unlimited)
    cleanup_interval=60.0,  # Background sweep for expired entries (seconds)
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_size` | `0` | Maximum entries before LRU eviction (0 = unlimited) |
| `cleanup_interval` | `60.0` | Background cleanup interval in seconds |

**Limitation**: `InMemoryCache` is per-process. If you run multiple workers (e.g., `uvicorn --workers 4`), each has its own cache. Use `RedisCache` for shared caching.

### `RedisCache`

Distributed cache for multi-instance deployments. Values are JSON-serialized.

```python
from promptise.mcp.server import RedisCache, cached

# Connect to Redis
cache = RedisCache(
    url="redis://localhost:6379/0",
    prefix="weather:",  # Key namespace
)

@server.tool()
@cached(ttl=300, backend=cache, scope="shared")
async def get_forecast(city: str, days: int = 5) -> dict:
    """5-day forecast — cached in Redis, shared across all server instances."""
    return await weather_api.forecast(city, days)

# Cleanup on shutdown
@server.on_shutdown
async def cleanup():
    await cache.close()
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `url` | `"redis://localhost:6379/0"` | Redis connection URL |
| `prefix` | `"promptise:"` | Key namespace prefix |
| `client` | `None` | Pre-configured `redis.asyncio.Redis` client |

Requires `pip install redis`.

### `CacheMiddleware`

Apply caching to all tools server-wide (instead of per-tool with `@cached`):

```python
from promptise.mcp.server import AuthMiddleware, CacheMiddleware, InMemoryCache

cache = InMemoryCache(max_size=200)
server.add_middleware(AuthMiddleware(auth))          # identify the caller first
server.add_middleware(CacheMiddleware(backend=cache, ttl=120))
```

Entries are keyed on the server, the tool, its arguments and the caller,
with the same `scope` choices as `@cached` (default `"client"`).

- **Add `AuthMiddleware` first.** Middleware runs in the order added. If
  `CacheMiddleware` runs before the caller is identified, calls to
  authenticated tools skip the cache (and a warning is logged once) rather
  than share entries between callers.
- **Guards run on cache hits.** A hit still evaluates the tool's guards, so
  a cached result is never returned to a caller the tool would refuse.
- **It caches every tool**, including ones that change data: a second
  identical call within the TTL returns the first result without running
  the handler. If the server has write tools, cache the read tools with
  `@cached` instead.

### Custom cache backends

Implement the `CacheBackend` protocol for any storage (Memcached, DynamoDB, etc.):

```python
from promptise.mcp.server import CacheBackend

class DynamoCache:
    """Cache backed by DynamoDB."""

    async def get(self, key: str) -> Any | None:
        item = await dynamo.get_item(Key={"pk": key})
        return item.get("value")

    async def set(self, key: str, value: Any, ttl: float) -> None:
        await dynamo.put_item(Item={
            "pk": key,
            "value": value,
            "ttl": int(time.time() + ttl),
        })

    async def delete(self, key: str) -> None:
        await dynamo.delete_item(Key={"pk": key})

    async def clear(self) -> None:
        # Scan and delete all items with our prefix
        ...
```

---

## Rate Limiting

### The problem

An agent enters a retry loop and hammers your server with 1000 requests per second. Without rate limiting, your database connection pool exhausts and the server crashes for everyone.

### `RateLimitMiddleware`

Apply token-bucket rate limiting globally or per-tool:

```python
from promptise.mcp.server import (
    MCPServer, RateLimitMiddleware, TokenBucketLimiter,
)

server = MCPServer(name="api")

# Global rate limit: 120 requests/minute per client
server.add_middleware(RateLimitMiddleware(
    limiter=TokenBucketLimiter(rate_per_minute=120, burst=20),
))
```

### Per-tool rate limits

Different tools have different cost profiles. Limit expensive tools more aggressively:

```python
@server.tool(rate_limit="10/min")
async def generate_report(department: str) -> dict:
    """Generate a comprehensive department report.

    This queries multiple databases and takes 5-10 seconds.
    Limited to 10 calls/minute to prevent database overload.
    """
    return await build_report(department)

@server.tool(rate_limit="1000/min")
async def get_employee(employee_id: str) -> dict:
    """Look up a single employee. Fast, cheap, high limit."""
    return await db.get_employee(employee_id)
```

Declared limits are enforced automatically — no middleware wiring needed. The
spec is parsed at registration (`"N/sec"`, `"N/min"`, or `"N/hour"`; a typo
raises `ValueError` immediately instead of silently never limiting), and each
declaring tool gets its own token bucket sized to the declared count. Buckets
are keyed **per client** when authentication populates `client_id`, and fall
back to a single shared bucket for unauthenticated tools. A declared limit
composes with a server-wide `RateLimitMiddleware` if you install one: the
middleware enforces the global policy, the declaration enforces each tool's
contract.

### `TokenBucketLimiter`

The built-in rate limiter uses the token bucket algorithm — steady rate with burst capacity:

```python
limiter = TokenBucketLimiter(
    rate_per_minute=60,  # Sustained rate
    burst=10,            # Allow short bursts above the rate
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rate_per_minute` | `60` | Sustained request rate |
| `burst` | `None` | Max burst (defaults to `rate_per_minute`) |

When a client exceeds the limit, the server returns a `RateLimitError` with `retryable=True` and a `retry_after` hint.

### Tiered rate limiting

Real-world pattern: free agents get lower limits than premium agents.

```python
class TieredLimiter:
    """Different rates for free vs premium agents."""

    def __init__(self):
        self._free = TokenBucketLimiter(rate_per_minute=10, burst=3)
        self._premium = TokenBucketLimiter(rate_per_minute=1000, burst=50)

    async def check(self, key: str, ctx) -> tuple[bool, float]:
        roles = ctx.state.get("roles", set())
        limiter = self._premium if "premium" in roles else self._free
        return limiter.consume(key)
```

---

## Concurrency Limits

### The problem

Your database allows 10 concurrent connections. Without concurrency limits, 50 simultaneous agent calls overwhelm the connection pool and every call fails.

### Server-wide limit

```python
from promptise.mcp.server import ConcurrencyLimiter

# Max 50 concurrent tool calls across all tools
server.add_middleware(ConcurrencyLimiter(max_concurrent=50))
```

### Per-tool limits

Tools that access constrained resources get their own limits:

```python
@server.tool(max_concurrent=5)
async def query_database(sql: str) -> list[dict]:
    """Run a SQL query. Limited to 5 concurrent calls to protect the DB pool."""
    async with db_pool.acquire() as conn:
        return await conn.fetch(sql)

@server.tool(max_concurrent=2)
async def generate_pdf(report_id: str) -> dict:
    """Generate a PDF report. CPU-intensive — limited to 2 concurrent."""
    return await pdf_engine.render(report_id)

@server.tool()  # No limit — this is fast and stateless
async def ping() -> str:
    """Health check."""
    return "pong"
```

When the limit is reached, additional calls receive a `RateLimitError` with `retryable=True`.

---

## Timeouts

### The problem

An external API hangs indefinitely. Without a timeout, the tool call blocks forever, consuming a concurrency slot and leaving the agent waiting.

### `TimeoutMiddleware`

Apply default timeouts to all tools:

```python
from promptise.mcp.server import TimeoutMiddleware

server.add_middleware(TimeoutMiddleware(default_timeout=30.0))
```

### Per-tool timeouts

Override for specific tools:

```python
@server.tool(timeout=5.0)
async def quick_lookup(key: str) -> dict:
    """Fast key-value lookup. Should complete in <1s; timeout at 5s."""
    return await cache.get(key)

@server.tool(timeout=120.0)
async def run_analysis(dataset: str) -> dict:
    """Heavy data analysis. May take up to 2 minutes."""
    return await analytics.process(dataset)
```

When a timeout fires, the tool call raises `asyncio.TimeoutError`, which the framework converts to a `ToolError` with `retryable=True`.

---

## Combining Features

A real production server uses multiple performance features together:

```python
from promptise.mcp.server import (
    MCPServer, InMemoryCache, cached,
    RateLimitMiddleware, TokenBucketLimiter,
    ConcurrencyLimiter, TimeoutMiddleware,
    LoggingMiddleware,
)

server = MCPServer(name="production-api")
cache = InMemoryCache(max_size=1000)

# Middleware stack (outermost → innermost)
server.add_middleware(LoggingMiddleware())                         # Log all calls
server.add_middleware(RateLimitMiddleware(TokenBucketLimiter(120)))  # 120/min/client
server.add_middleware(TimeoutMiddleware(default_timeout=30.0))      # 30s default
server.add_middleware(ConcurrencyLimiter(max_concurrent=100))       # 100 total

@server.tool(
    rate_limit="20/min",      # Stricter per-tool limit
    timeout=60.0,             # This tool needs more time
    max_concurrent=10,        # Only 10 concurrent DB queries
    read_only_hint=True,
)
@cached(ttl=300, backend=cache, scope="shared")  # the catalog is the same for everyone
async def search_products(query: str, category: str = "all") -> list[dict]:
    """Search the product catalog.

    Hits Elasticsearch — cached 5 min, rate limited, concurrency capped.
    """
    return await elasticsearch.search(index="products", query=query, category=category)
```

---

## API Summary

| Symbol | Type | Description |
|--------|------|-------------|
| `@cached(ttl, key_func, backend, scope)` | Decorator | Cache tool results with TTL, per caller by default |
| `InMemoryCache(max_size, cleanup_interval)` | Class | In-process LRU cache with expiry |
| `RedisCache(url, prefix, client)` | Class | Redis-backed distributed cache |
| `CacheMiddleware(backend, ttl, scope)` | Class | Server-wide caching middleware, per caller by default |
| `CacheScope` | Type | `"client"`, `"tenant"` or `"shared"` |
| `CacheBackend` | Protocol | Interface for custom cache backends |
| `RateLimitMiddleware(limiter, per_tool)` | Class | Token-bucket rate limiting |
| `TokenBucketLimiter(rate_per_minute, burst)` | Class | Token bucket rate limiter |
| `ConcurrencyLimiter(max_concurrent)` | Class | Server-wide concurrency cap |
| `PerToolConcurrencyLimiter` | Class | Per-tool concurrency limits (auto-inserted) |
| `TimeoutMiddleware(default_timeout)` | Class | Per-call timeout enforcement |

## What's Next

- [Observability & Monitoring](observability.md) -- Metrics, tracing, Prometheus, structured logging
- [Resilience Patterns](resilience-patterns.md) -- Circuit breaker, health checks, background tasks
- [Authentication & Security](auth-security.md) -- JWT, API keys, guards
