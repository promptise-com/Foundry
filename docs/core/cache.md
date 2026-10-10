# Semantic Cache

Cache LLM responses by query similarity. Save 30-50% on API costs by serving cached responses for semantically similar queries. All embedding runs locally by default.

```python
from promptise import build_agent, SemanticCache, CallerContext
from promptise.config import HTTPServerSpec

cache = SemanticCache()
cache.warmup()

agent = await build_agent(
    servers={"tools": HTTPServerSpec(url="http://localhost:8000/mcp")},
    model="openai:gpt-5-mini",
    cache=cache,
)

# First call → LLM, result cached
result = await agent.ainvoke(input, caller=CallerContext(user_id="user-42"))

# Second similar call → cache hit, no LLM call, instant response
result = await agent.ainvoke(input, caller=CallerContext(user_id="user-42"))
```

---

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## How It Works

1. User sends a message
2. Input guardrails scan the message (block injection attacks)
3. Memory search runs (so the cache key includes memory context)
4. **Cache check** — embed the query, search for similar cached queries with matching context
5. **Cache hit** → run output guardrails on cached response → return instantly. The result holds *this* request's messages followed by the cached answer; the cache stores only the final answer, never the messages, injected context or tool results of the request that produced it
6. **Cache miss** → continue to tools, LLM → output guardrails → **cache the post-guardrail response** → return

Cache check runs **after** input guardrails and memory search (so the cache key reflects current memory state) but **before** tool selection and LLM call. Cached responses are stored **after** output guardrails — only safe, redacted content is ever persisted in the cache.

---

## What Gets Cached

By default the cache only holds answers that are safe to replay:

| Request | Cached by default | Option |
|---------|-------------------|--------|
| One user message, no tool calls | Yes | — |
| Earlier turns in `messages` (a follow-up) | No — not looked up, not stored | `cache_multi_turn=True` |
| The agent called read-only tools | No | `cache_tool_turns=True` |
| The agent called a write tool | **Never** | — |
| The agent called a tool behind an [approval gate](approval.md) | **Never** | — |

**Follow-ups.** "What river runs through it?" means the Seine in a conversation about Paris and the Thames in one about London. With the default `cache_multi_turn=False`, a request whose `messages` carry earlier user, assistant or tool messages bypasses the cache (system messages don't count). With `cache_multi_turn=True` it is cached, and the earlier messages — their role, content and tool calls — are hashed into the cache key, so a follow-up only hits for the same conversation history.

**Tool turns.** A cached answer replaces the whole turn, tool calls included: the tool does not run. That is never acceptable for a write — a repeated "open a ticket" would answer "T-103 opened" without opening anything — so a turn that called a write tool is never stored. Answers built from read-only tools go stale when the data changes, so they are only cached with `cache_tool_turns=True`; [write invalidation](#write-invalidation) then evicts them when the agent writes, and `ttl_patterns` bound how long they live otherwise.

**Approval.** An approval is a decision about one call. A turn that called a tool gated by `build_agent(approval=ApprovalPolicy(...))` — approved or denied — is never cached, so a repeated question asks the reviewer again instead of replaying the earlier outcome. Gates enforced by an MCP server (approval through elicitation) are invisible to the cache; don't list such tools in `read_only_tools` or annotate them `readOnlyHint=True`.

```python
cache = SemanticCache(
    cache_tool_turns=True,                 # cache answers from read-only tools
    read_only_tools=["search_docs"],       # extra_tools without MCP annotations
    write_tools=["sync_*"],                # never trust these as read-only
)
```

---

## Security: Per-User Isolation

**Default scope is `per_user`** — every user gets an isolated cache partition. User A's cached responses are invisible to User B.

**No CallerContext = no caching.** If you don't pass `caller=CallerContext(user_id=...)`, caching is silently disabled for that request. This prevents accidental cross-user data leakage.

```python
# ✅ Cached (user isolated)
result = await agent.ainvoke(input, caller=CallerContext(user_id="user-42"))

# ❌ Not cached (no identity)
result = await agent.ainvoke(input)
```

Three scopes:

| Scope | Behavior | Use case |
|-------|----------|----------|
| `per_user` (default) | Each user has their own cache | Any personalized agent |
| `per_session` | Each session has its own cache | Conversation-specific |
| `shared` | All users share one cache | Public knowledge (weather, docs, FAQ) |

```python
# Shared scope — requires explicit acknowledgment
cache = SemanticCache(scope="shared", shared_data_acknowledged=True)
```

---

## Standalone / Shared Mode (No Multi-User)

If you're building a single-user app, internal tool, or public FAQ agent where there's no concept of "users," use `scope="shared"`:

```python
cache = SemanticCache(scope="shared", shared_data_acknowledged=True)

agent = await build_agent(
    servers={...}, model="openai:gpt-5-mini", cache=cache,
)

# No CallerContext needed — works immediately
result = await agent.ainvoke({"messages": [{"role": "user", "content": "What is Python?"}]})
```

With `scope="shared"`, caching works without `CallerContext`. Everyone shares the same cache. Use this when:

- Your agent answers public knowledge questions (docs, FAQ, weather)
- There's only one user (CLI tools, internal scripts)
- Responses never contain personalized data

!!! warning "Shared scope = no isolation"
    Every user sees everyone else's cached responses. Never use shared scope when the agent accesses user-specific data (accounts, orders, personal info).

---

## Multi-User Mode

For apps with multiple users (SaaS, customer support, multi-tenant), the default `per_user` scope isolates each user's cache automatically:

```python
cache = SemanticCache()  # scope="per_user" is the default

agent = await build_agent(..., cache=cache)

# Each user has their own cache partition
await agent.ainvoke(input, caller=CallerContext(user_id="alice"))  # Alice's cache
await agent.ainvoke(input, caller=CallerContext(user_id="bob"))    # Bob's cache (separate)
```

**How it works internally:**
- Cache key prefix is `user:{user_id}` — Alice's entries are keyed `user:alice`, Bob's are `user:bob`
- With `CallerContext(tenant_id="acme")` the key is tenant-qualified — an injective, colon-prefixed hash (`user:t:<sha256>`) disjoint from the untenanted namespace, so two tenants with the same `user_id` can never share a cache partition
- Similarity search only runs within a user's own partition — no cross-user matching possible
- If no `CallerContext` is provided, caching is silently disabled for that request (with a debug log: `"Cache: no CallerContext or user_id provided"`)
- `purge_user("alice")` removes all of Alice's cached entries (GDPR compliance)

**Per-session mode** isolates even further — each of a user's sessions has its own cache:

```python
cache = SemanticCache(scope="per_session")

caller = CallerContext(
    user_id="alice",
    metadata={"session_id": "sess_abc123"},
)
await agent.ainvoke(input, caller=caller)
```

A session partition belongs to the user and session together, so two users who pass the same `session_id` never share entries. A caller without a `user_id` is not cached, as with `per_user`; a caller without a `session_id` uses the user's partition. `purge_user()` removes the user partition only — per-session entries expire with their TTL — and a write evicts only the session it happened in.

---

## Configuration

```python
cache = SemanticCache(
    backend="memory",                # "memory" or "redis"
    similarity_threshold=0.92,       # 0.0-1.0 (higher = stricter matching)
    default_ttl=3600,                # seconds
    scope="per_user",                # "per_user", "per_session", "shared"
    max_entries_per_user=1000,
    max_total_entries=100_000,
    invalidate_on_write=True,        # evict cache when write tools fire
    cache_multi_turn=False,          # follow-ups bypass the cache
    cache_tool_turns=False,          # turns that called tools are not cached
    ttl_patterns={                   # regex → TTL for time-sensitive queries
        r"current|now|today|latest": 60,
        r"price|stock|rate": 30,
    },
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `backend` | `str` | `"memory"` | `"memory"` or `"redis"` |
| `redis_url` | `str` | `None` | Redis connection URL |
| `embedding` | `EmbeddingProvider \| str` | Local model | Embedding provider or model name |
| `similarity_threshold` | `float` | `0.92` | Min cosine similarity for cache hit |
| `default_ttl` | `int` | `3600` | Default time-to-live in seconds |
| `scope` | `str` | `"per_user"` | Cache isolation scope |
| `max_entries_per_user` | `int` | `1000` | Max entries per scope partition |
| `max_total_entries` | `int` | `100_000` | Max entries across all scopes |
| `encrypt_values` | `bool` | `False` | AES encryption at rest (Redis) |
| `ttl_patterns` | `dict` | `None` | Regex → TTL overrides |
| `invalidate_on_write` | `bool` | `True` | Evict the caller's cached answers when a write tool runs |
| `cache_multi_turn` | `bool` | `False` | Cache requests with earlier turns; the history is part of the key |
| `cache_tool_turns` | `bool` | `False` | Cache answers from turns that called only read-only tools |
| `write_tools` | `list[str]` | `None` | Tool names (`*` wildcards) always treated as writes |
| `read_only_tools` | `list[str]` | `None` | Tool names (`*` wildcards) treated as read-only |

---

## Cache Backends

### In-Memory (default)

Zero dependencies. Sub-millisecond lookups. Lost on restart. Single-process only.

```python
cache = SemanticCache(backend="memory")
```

### Redis

Shared across workers and servers. Survives restarts. Optional AES encryption at rest.

```python
cache = SemanticCache(
    backend="redis",
    redis_url="redis://localhost:6379",
    encrypt_values=True,  # AES encryption — set PROMPTISE_CACHE_KEY env var
)
```

Requires `pip install redis`. Encryption requires `pip install cryptography`.

Set `PROMPTISE_CACHE_KEY` to a Fernet key for persistent encryption across restarts. If not set, a key is auto-generated per process (cache won't survive restart).

**Graceful degradation:** If Redis is unreachable, cache operations fail silently (logged as warnings) and the agent continues normally — LLM is called directly.

---

## Embedding Providers

### Local (default)

Uses `sentence-transformers` — the same model used for semantic tool optimization. Zero API calls, runs locally. It needs `sentence-transformers` and `numpy` (`pip install "promptise[all]"`); if either is missing, `build_agent(cache=...)` raises `ImportError` naming the package, rather than building an agent that never caches. Call `cache.warmup()` at startup to load the model before the first request.

```python
cache = SemanticCache()  # all-MiniLM-L6-v2
cache = SemanticCache(embedding="BAAI/bge-small-en-v1.5")
cache = SemanticCache(embedding="/models/local/custom")
```

### OpenAI / Azure OpenAI

```python
from promptise import OpenAIEmbeddingProvider

cache = SemanticCache(
    embedding=OpenAIEmbeddingProvider(
        model="text-embedding-3-small",
        api_key="${OPENAI_API_KEY}",
    ),
)
```

### Custom provider

Any object implementing the `EmbeddingProvider` protocol:

```python
class MyProvider:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return my_model.encode(texts)

cache = SemanticCache(embedding=MyProvider())
```

---

## Cache Key

The cache key determines when a hit occurs. It includes:

| Component | What it prevents |
|-----------|-----------------|
| Scope prefix (`user:42`) | Cross-user data leakage |
| Query embedding | Semantic similarity matching |
| Context fingerprint | Stale answers after memory changes; one conversation's follow-up answering another's |
| Model ID | Serving GPT responses as Claude responses |
| Instruction hash | Stale responses after prompt updates |

The context fingerprint hashes the injected memory content and every message before the query — system messages, and with `cache_multi_turn=True` the earlier turns — not just how many there are.

With a [`FallbackChain`](fallback.md) the model ID is the chain member that is serving: answers are stored under the model that wrote them, and looked up under the first model whose circuit is not open. An answer from a fallback model is never served as the primary's.

If any component changes, the cache misses and a fresh LLM call is made. A similar entry stored under a different context does not hide one that matches: the closest entry *with the same context, model and instructions* is served.

---

## Write Invalidation

When the agent calls a write tool, every cached answer in the caller's scope is evicted, so nothing computed before the write is served after it:

```
"How many tickets are open?" → cached "47"        (cache_tool_turns=True)
create_ticket() runs → cache evicted
"How many tickets are open?" → fresh LLM call → "48"
```

**What counts as a write.** MCP servers describe their tools with annotations; a tool is read-only when it declares `readOnlyHint: true` (`@server.tool(read_only_hint=True)` in Promptise's server SDK). Any other tool — including one with no annotations, the MCP default — counts as a write. Override per tool name:

- `read_only_tools=[...]` — read-only whatever the annotations say; use it for `extra_tools` and other tools without annotations.
- `write_tools=[...]` — always a write; wins over `read_only_tools` and annotations. Annotations come from the server, so list a tool here if you don't trust its server.

**When.** The eviction happens as soon as the write tool finishes or fails — before the agent writes its answer, and even if the run fails afterwards. It also happens for runs made with `astream()` and `astream_with_tools()`. A request in the same scope that was already running when the write landed does not store its answer, since it may have read the old data. Writes made outside the agent are not seen: bound them with `default_ttl` and `ttl_patterns`.

To evict by hand, for example after a webhook reports a change: `await cache.invalidate_for_write("orders_updated", caller=caller)`.

Disable with `invalidate_on_write=False` if your tools don't affect query results. Turns that called a write tool are still never cached.

---

## GDPR Compliance

```python
# Delete all cached data for a user
count = await cache.purge_user("user-42")

# Tenant-scoped callers: purge exactly that tenant's scope
count = await cache.purge_user("user-42", tenant_id="acme")
```

---

## Observability

With `observe=True`, cache events appear in the observability timeline:

- `cache.hit` — response served from cache; `metadata` carries `similarity` (cosine similarity to the stored query), `age_seconds`, `scope` and `ttl`
- `cache.miss` — no cache hit, proceeding to LLM
- `cache.store` — new response stored in cache

A cache failure (Redis unreachable, embedding API down) never fails the request: it is logged as a warning on the `promptise.cache` or `promptise.agent` logger and the request goes to the LLM.

`await cache.stats()` returns a `CacheStats` with `hits`, `misses`, `stores`, `evictions` and `hit_rate`. Each looked-up request counts once, as a hit or a miss; a request that bypasses the cache (a follow-up with `cache_multi_turn=False`) counts as neither.

---

## What's Next?

- [Guardrails](guardrails.md) — output guardrails always run on cached responses
- [Tool Optimization](tool-optimization.md) — shares the same embedding model
- [Building Agents](agents/building-agents.md) — the `cache` parameter on `build_agent()`
