---
title: "Cache MCP Tool Results to Cut Redundant API Calls"
description: "An agent-side semantic cache saves LLM tokens, but the other repeated cost is the same expensive tool hitting your database or a paid API five times in one…"
keywords: "cache mcp tool results, mcp server response caching, cached decorator mcp tool, per-client tool result cache, reduce redundant tool calls, redis cache mcp server"
date: 2026-07-16
slug: cache-mcp-tool-results
categories:
  - Cost & Efficiency
---

# Cache MCP Tool Results to Cut Redundant API Calls

You can **cache MCP tool results** at the server so that the same expensive lookup — a database query, a paid pricing API, an Elasticsearch scan — runs once per TTL window instead of five times in one conversation. An agent-side semantic cache is a great lever, but it only saves *LLM tokens*: it stops the model re-reasoning over a paraphrased question. It does nothing for the *other* repeated cost, which is your tool handler re-hitting the backend every time the model decides it needs that fact again. Those are two different bills, and they need two different caches.

<!-- more -->

This post is about the second one: **MCP server response caching**. In Promptise Foundry it's a first-class server primitive — a `@cached` decorator, an `InMemoryCache` and a distributed `RedisCache`, a `CacheMiddleware`, and a custom `key_func` so one tenant's cached result never answers another tenant's call. No `functools.lru_cache` in every handler, no bespoke Redis wrapper you rewrite per tool.

## Why the same tool call costs you five times

A single agent turn can call the same tool repeatedly. The model asks for a price, uses it, later re-derives that it needs the same price, and asks again. Multi-step reasoning, retries after a validation error, and self-critique loops all amplify this. Each call is an independent execution of your handler — so if that handler charges you $0.001 per request or holds a database connection for 200ms, you pay and wait every single time.

The agent-side [semantic cache](../../core/cache.md) sits in front of the *model* and can't help here: by the time the LLM has emitted a tool call, the decision to hit your backend is already made. The only place to **reduce redundant tool calls** at the backend is where the tool actually runs — on the MCP server. That's the layer this cache lives in, and the two caches compose: the semantic cache trims token cost, the server cache trims backend cost, and together they cover both halves of a multi-tenant deployment's bill (the full picture is in [How to Cut Token Cost for a Multi-Tenant AI Agent](cut-token-cost-multi-tenant-ai-agent.md)).

## Cache MCP tool results with the @cached decorator

The unit of caching is one tool. Stack the `@cached` decorator under `@server.tool()` and the first call executes the handler; every later call with the same key returns the stored result until the TTL expires. Here is a complete, runnable server — it needs nothing but `pip install promptise`, no API key and no network, because it uses the in-process `TestClient` to drive the full auth-plus-middleware pipeline:

```python
import asyncio

from promptise.mcp.server import (
    APIKeyAuth,
    AuthMiddleware,
    InMemoryCache,
    MCPServer,
    TestClient,
    cached,
)

server = MCPServer(name="pricing")
server.add_middleware(
    AuthMiddleware(
        APIKeyAuth(
            keys={
                "sk-acme": {"client_id": "acme", "roles": ["analyst"]},
                "sk-globex": {"client_id": "globex", "roles": ["analyst"]},
            }
        )
    )
)

cache = InMemoryCache(max_size=500)
backend_hits: dict[str, int] = {}


@server.tool(auth=True, read_only_hint=True)
@cached(ttl=300, backend=cache)  # entries are scoped to the caller by default
async def get_price(sku: str) -> dict:
    """Expensive priced lookup — hits a paid pricing API on a miss."""
    backend_hits[sku] = backend_hits.get(sku, 0) + 1
    return {"sku": sku, "price": 42.0}


async def main():
    client = TestClient(server)
    acme = {"x-api-key": "sk-acme"}
    globex = {"x-api-key": "sk-globex"}

    # acme asks for the same SKU three times in one conversation.
    for _ in range(3):
        await client.call_tool("get_price", {"sku": "A-1"}, headers=acme)

    # globex — a different tenant — asks for the same SKU string.
    await client.call_tool("get_price", {"sku": "A-1"}, headers=globex)

    print("backend executions for A-1 :", backend_hits["A-1"])
    print("acme sees      :", (await client.call_tool(
        "get_price", {"sku": "A-1"}, headers=acme))[0].text)


asyncio.run(main())
```

Running it prints exactly:

```text
backend executions for A-1 : 2
acme sees      : {"sku": "A-1", "price": 42.0}
```

Read that count carefully, because it is the whole point. `acme` called `get_price("A-1")` **three** times and the handler ran **once** — the two repeats were served from cache. `globex` called the identical SKU string and the handler ran **again**, giving a total of two backend executions rather than four. That second execution is not a bug; it is the per-client scoping doing its job, which the next section unpacks.

The `@cached` decorator takes `ttl` (seconds), an optional `backend`, an optional `key_func`, and a `scope`. The default key is the server, the function, a hash of the JSON-serialised arguments and the authenticated caller, so `get_price("A-1")` and `get_price("A-2")` are naturally separate entries, and so are acme's and globex's. The [caching guide](../../mcp/server/caching-performance.md) documents the full decorator surface, including combining it with per-tool rate limits, timeouts, and concurrency caps on the same tool.

## Per-client keys: one tenant's cache never answers another's

A cache keyed **only** on arguments has a sharp edge on any shared server: `get_price("A-1")` from tenant *acme* and `get_price("A-1")` from tenant *globex* collide on the same entry — and whoever misses first fills a cache the other then reads. On a multi-tenant server that is a cross-tenant data leak dressed up as a performance win.

So Promptise keys every entry on the caller by default: the token's issuer, the tenant and the client id, taken from the authenticated request rather than from a tool argument, so it can't be spoofed. That holds even when the handler reads the caller through `get_context()` instead of taking a `ctx` parameter. Widening the sharing is an explicit opt-out with `scope`:

```python
@server.tool(auth=True, read_only_hint=True)
@cached(ttl=300, backend=cache, scope="tenant")   # one entry per tenant
async def org_price_list(ctx: RequestContext) -> list[dict]:
    return await pricing.list_for(ctx.client.tenant_id)

@server.tool(read_only_hint=True)
@cached(ttl=300, backend=cache, scope="shared")   # same answer for everyone
async def exchange_rate(currency: str) -> float:
    return await fx.rate(currency)
```

A custom `key_func` (`(func_name: str, args: dict) -> str`) still lets you decide which arguments matter, for example ignoring a page number; its key is prefixed with the scope, so it can't widen sharing by accident.

Its one limitation is honest and unavoidable: it is per-process. Run `uvicorn --workers 4` and you have four independent caches, so a hit rate that looked great on one worker dilutes across the pool, and there's no shared invalidation. When that matters, switch the backend — the decorator and your handler don't change. A **Redis cache for your MCP server** gives every instance one shared store:

```python
from promptise.mcp.server import RedisCache, cached

cache = RedisCache(url="redis://localhost:6379/0", prefix="pricing:")

@server.tool(read_only_hint=True)
@cached(ttl=300, backend=cache, scope="shared")  # a forecast is the same for everyone
async def get_forecast(city: str, days: int = 5) -> dict:
    """5-day forecast — cached in Redis, shared across all server instances."""
    return await weather_api.forecast(city, days)
```

`RedisCache` JSON-serialises values and namespaces keys under `prefix`, so several services can share one Redis without colliding. It needs `pip install redis`. If you'd rather not decorate tools one at a time, `CacheMiddleware(backend=cache, ttl=120)` applies caching server-wide to every tool at once — the same backends, TTL semantics and per-caller scoping, applied as a middleware instead of a decorator. Add it after `AuthMiddleware`, and only on servers whose tools are all reads: it caches every tool. And if your storage is neither in-memory nor Redis, implement the `CacheBackend` protocol (`get`, `set`, `delete`, `clear`) and pass your own — DynamoDB, Memcached, whatever your platform already runs. The [core cache concepts](../../core/cache.md) page contrasts this server-side result cache with the agent-side semantic cache so you can reason about both layers together.

## What other frameworks do today

Caching a tool's output is not a novel idea — the honest gap is that most MCP server stacks make it *your* job to assemble, per handler, instead of handing you a primitive.

- **The official MCP Python SDK / FastMCP** gives you a real middleware pipeline, so you genuinely *can* build a caching layer — that's a partial capability, and it would be wrong to say the seam isn't there. What it doesn't ship is a result cache with argument-hash keys, TTL, LRU eviction, and per-client key derivation ready to attach. The exact delta: you still supply the store, the key function, the serialization, and the tenant scoping yourself, and the per-client dimension is the one teams most often forget.
- **`functools.lru_cache`** is the reflexive reach, and it's the sharpest trap. It has no TTL (entries never expire, so you serve stale prices indefinitely), no eviction by age, no per-client scoping (it keys purely on arguments, so tenant A's result answers tenant B), and it's process-local with no cross-worker sharing. It also can't see async coroutine identity cleanly. It solves the easy 20% and silently owns the risky 80%.
- **A hand-rolled Redis wrapper inside each handler** does get you cross-worker sharing, but you re-implement key derivation, TTL, JSON serialization, and — critically — the per-client scoping in *every tool*, and there's no shared invalidation contract across them. Miss the client dimension in one handler and that tool leaks across tenants while its neighbors don't.
- **LangChain / LangGraph** ship an *LLM* response cache (`set_llm_cache`) that memoizes model calls, which is real and useful — but it caches the LLM layer, not arbitrary tool outputs, so it doesn't stop a tool from re-hitting your database. Tool-result caching there is back to a wrapper you write.

None of this means those stacks *can't* cache tool results — with enough glue any of them will. Promptise's edge is that it makes the capability **structural**: caching, its backends, and per-client key scoping are shipped server primitives, so the safe thing (bounded, expiring, tenant-partitioned) is the default thing, and the leak-prone shortcut isn't the path of least resistance.

## Frequently asked questions

### Does caching apply before or after authentication and guards?

After. Auth middleware runs first and populates the request context, which is where the per-caller key comes from. With `@cached`, a client that isn't authorized for a tool is rejected by its guard before the cache is ever consulted. `CacheMiddleware` runs the tool's guards on every cache hit, so neither serves a result to a caller who couldn't have called the tool.

### Which tools should I cache, and which should I never cache?

Cache **read-only** tools whose output is stable for a while — lookups, prices, forecasts, catalog searches. Mark them `read_only_hint=True` so the model knows they're safe to repeat. Never wrap a tool with side effects (a write, a payment, a refund) in `@cached`: a cache hit would skip the side effect entirely. Caching is for reads.

### How do I stop one tenant's cached result from answering another tenant?

You don't have to: entries are scoped to the authenticated caller by default. `scope="tenant"` lets two clients in the same org share a partition while different orgs stay isolated; `scope="shared"` is for data that is the same for everyone. Because the identity comes from the authenticated context and not a tool argument, it can't be spoofed. The runnable example above proves it: `globex` re-executes the handler for a SKU string `acme` had already cached.

### Do InMemoryCache and RedisCache use the same TTL and eviction semantics?

TTL is identical — `@cached(ttl=...)` governs both. Eviction differs by nature: `InMemoryCache` does LRU eviction at `max_size` plus a background sweep of expired entries; `RedisCache` relies on Redis to expire keys and on your Redis `maxmemory-policy` for eviction. Swap one for the other without touching the decorator or your handler.

### Is there a shared cache across multiple workers?

Not with `InMemoryCache` — it's per-process, so `uvicorn --workers 4` means four independent caches. Use `RedisCache` (or a custom `CacheBackend`) for a store shared across every worker and instance, with a single invalidation point.

## Next steps

Wrap your single most expensive read-only tool in `@cached` with an `InMemoryCache` and run it under `TestClient` the way the example above does — watch redundant backend calls collapse to one per TTL window while a second client still gets its own execution. When one process stops being enough, change one line to `RedisCache` and re-run to confirm the hit path is identical across workers. From there, read the [Caching & Performance guide](../../mcp/server/caching-performance.md) to compose caching with rate limiting, concurrency caps, and timeouts on the same tool, the [core cache concepts](../../core/cache.md) to pair it with the agent-side semantic cache, and [Multi-Tenancy](../../mcp/server/multi-tenancy.md) to make per-tenant isolation a server-wide invariant rather than a per-handler habit.
