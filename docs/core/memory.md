# Memory

Give agents persistent memory with vector search, enabling context-aware responses across conversations.

```python
from promptise import build_agent
from promptise.config import HTTPServerSpec
from promptise.memory import ChromaProvider

provider = ChromaProvider(
    collection_name="agent_memory",
    persist_directory=".promptise/chroma",
)

agent = await build_agent(
    servers={"tools": HTTPServerSpec(url="http://localhost:8000/mcp")},
    model="openai:gpt-5-mini",
    memory=provider,
)
# Every ainvoke() now automatically searches memory and injects relevant context.
```

---

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## Concepts

Promptise memory has two layers:

- **Auto-injection** -- before every agent invocation, relevant memories are searched and injected as a `SystemMessage`. The agent sees contextual history without needing explicit memory tools. `build_agent(memory=...)` does this inside the agent; `MemoryAgent` does the same for a graph you built yourself.
- **Provider protocol** -- a simple async interface (`search`, `add`, `delete`, `close`) that any backend can implement.

Three providers ship with the framework, covering development through production use cases.

---

## MemoryProvider Protocol

All memory providers implement this async protocol:

```python
class MemoryProvider(Protocol):
    scope: MemoryScope  # SHARED (default) or PER_USER

    async def search(
        self, query: str, *, limit: int = 5, user_id: str | None = None
    ) -> list[MemoryResult]: ...
    async def add(
        self, content: str, *, metadata: dict | None = None, user_id: str | None = None
    ) -> str: ...
    async def delete(self, memory_id: str, *, user_id: str | None = None) -> bool: ...
    async def purge_user(self, user_id: str) -> int: ...
    async def close(self) -> None: ...
```

| Method | Returns | Description |
|---|---|---|
| `search(query, limit=5, user_id=None)` | `list[MemoryResult]` | Search for memories relevant to the query. In `PER_USER` scope, results are filtered to that user. |
| `add(content, metadata=None, user_id=None)` | `str` | Store a new memory, returns its ID. In `PER_USER` scope, stamps ownership. |
| `delete(memory_id, user_id=None)` | `bool` | Delete a memory by ID. In `PER_USER` scope, only the owner can delete. |
| `purge_user(user_id)` | `int` | Delete every entry owned by `user_id`. Returns the count removed. GDPR "right to erasure". |
| `close()` | `None` | Release resources (connections, file handles) |

---

## Memory Scopes — Shared vs Per-User

Every built-in provider supports two isolation modes controlled by the `scope` parameter:

| Scope | Behavior | Use case |
|---|---|---|
| `MemoryScope.SHARED` (default) | Legacy global pool. `user_id` is ignored. All callers read/write the same entries. | Knowledge-base bots, shared org memory, public FAQ agents |
| `MemoryScope.PER_USER` | Every operation is scoped to a `user_id`. Users cannot read or delete each other's entries. | Personal assistants, multi-tenant SaaS, compliance-sensitive workloads |

```python
from promptise.memory import InMemoryProvider, MemoryScope, MemoryIsolationError

# Per-tenant isolation
p = InMemoryProvider(scope=MemoryScope.PER_USER)

await p.add("alice's note", user_id="alice")
await p.add("bob's note", user_id="bob")

assert (await p.search("note", user_id="alice"))[0].content == "alice's note"
assert await p.delete("some-id", user_id="bob") is False  # not her entry
```

If a `PER_USER` provider is called without a `user_id`, it raises `MemoryIsolationError` — a fail-closed guarantee that nothing is ever stored or read without explicit ownership.

### Auto-propagation from CallerContext

When a memory provider is attached to an agent via `build_agent(memory=...)`, the agent reads the current `CallerContext` from the async contextvar and passes its isolation key (`user_id`, or `tenant_id::user_id` when a tenant is set) into every memory `search` and auto-store `add`. Your handler code never has to thread `user_id` manually:

```python
from promptise.agent import CallerContext

# In your request handler
caller = CallerContext(user_id="alice", metadata={"session_id": "sess-1"})
await agent.ainvoke(input, caller=caller)
# → memory provider sees user_id="alice" on every call

# chat(user_id=...) is shorthand for caller=CallerContext(user_id=...):
# it scopes memory exactly like the caller above, not only session ownership.
await agent.chat("What do I like to eat?", session_id=sid, user_id="alice")
```

If no caller is set (e.g., background tasks), a `PER_USER` provider raises `MemoryIsolationError` during the auto-search; the agent logs it as `Memory search failed` and runs without memory context.

### GDPR purge

```python
removed = await provider.purge_user("alice")   # → int count of deleted entries
```

Works on every provider. `InMemoryProvider` drops in-process entries. `ChromaProvider` deletes every entry whose metadata contains `_promptise_user_id: alice`. `Mem0Provider` counts the user's entries with `get_all` and removes them with Mem0's `delete_all(user_id=…)`.

---

## MemoryResult

Search results are returned as `MemoryResult` dataclasses:

| Field | Type | Description |
|---|---|---|
| `content` | `str` | The stored text |
| `score` | `float` | Relevance score (0.0 = no match, 1.0 = perfect) |
| `memory_id` | `str` | Unique identifier for this entry |
| `metadata` | `dict` | Provider-specific metadata |

Scores are clamped to `[0.0, 1.0]` on construction.

---

## Providers

### InMemoryProvider

Substring-search provider for testing and development. No persistence, no embeddings.

```python
from promptise.memory import InMemoryProvider, MemoryScope

# Shared pool (default, legacy behavior)
provider = InMemoryProvider(max_entries=1_000)

# Or per-user isolated
provider = InMemoryProvider(scope=MemoryScope.PER_USER)

await provider.add("Pipeline had 5% error rate at 07:30")
await provider.add("User prefers dark mode")

results = await provider.search("error rate")
# Matches entries containing the substring "error rate"
```

| Feature | Value |
|---|---|
| Search method | Case-insensitive substring matching |
| Persistence | None (in-memory only) |
| Isolation | `SHARED` or `PER_USER` via `scope=` |
| Dependencies | None |
| Best for | Testing, development, ephemeral agents |

!!! warning "Not for production"
    `InMemoryProvider` has no semantic understanding. The query `"deployment issues"` will **not** match content containing `"deploy"`. Use `ChromaProvider` or `Mem0Provider` for production workloads.

### ChromaProvider

Local vector similarity search with automatic embedding generation. Wraps [ChromaDB](https://www.trychroma.com/).

```python
from promptise.memory import ChromaProvider

# Ephemeral (no persistence)
provider = ChromaProvider(collection_name="agent_memory")

# Persistent (survives restarts)
provider = ChromaProvider(
    collection_name="agent_memory",
    persist_directory=".promptise/chroma",
)

await provider.add(
    "Pipeline had 5% error rate at 07:30",
    metadata={"source": "health-check", "severity": "warning"},
)

# Semantic search -- finds related content even without exact matches
results = await provider.search("deployment issues")
```

| Feature | Value |
|---|---|
| Search method | Vector similarity (cosine distance for collections it creates) |
| Score | Cosine similarity, `0.0`–`1.0` |
| Default embedding model | `all-MiniLM-L6-v2` (runs locally, no API key) |
| Persistence | Optional (`persist_directory` parameter) |
| Isolation | `SHARED` or `PER_USER` via `scope=` |
| Dependencies | `pip install "promptise[all]"` |
| Best for | Production agents needing semantic recall |

!!! warning "Use Chroma embedded, not as a shared server"
    `ChromaProvider` talks to an **embedded** Chroma: your process, your
    directory. Chroma's own HTTP server carries advisories that have no fixed
    release at the time of writing — code injection through the collection
    endpoints and an authorization provider that does not check which tenant a
    permission applies to ([GHSA-36p7-vc44-83pf](https://github.com/advisories/GHSA-36p7-vc44-83pf),
    [GHSA-f4j7-r4q5-qw2c](https://github.com/advisories/GHSA-f4j7-r4q5-qw2c),
    [GHSA-2wm9-hf6c-p5cr](https://github.com/advisories/GHSA-2wm9-hf6c-p5cr),
    [GHSA-xph7-9rjv-w5fr](https://github.com/advisories/GHSA-xph7-9rjv-w5fr)).
    Embedded use does not reach them. If several processes need one store, put
    it behind your own authenticated service, or use `Mem0Provider`.

Constructor parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `collection_name` | `str` | `"agent_memory"` | ChromaDB collection name |
| `persist_directory` | `str \| None` | `None` | Path for persistent storage (None = ephemeral) |
| `embedding_function` | `Any` | `None` | Custom ChromaDB embedding function |
| `scope` | `MemoryScope` | `SHARED` | `SHARED` (global pool) or `PER_USER` (metadata-filtered per tenant) |

In `PER_USER` mode the provider stamps every stored document with a `_promptise_user_id` metadata field and every `search`/`delete` uses ChromaDB's `where=` filter to restrict results to that owner.

#### Scores and distance functions

`ChromaProvider` creates a missing collection with cosine distance (`metadata={"hnsw:space": "cosine"}`), and `MemoryResult.score` is the cosine similarity: `1.0` for the same meaning, around `0.3`–`0.6` for related text, near `0.0` for unrelated text. That makes `min_score` filters meaningful — `memory_min_score=0.3` on `build_agent()` keeps related memories and drops noise.

A collection that already exists keeps the distance function it was created with; Chroma cannot change it. The provider reads it and scores accordingly:

| Collection space | Score |
|---|---|
| `cosine` | `1 - distance` |
| `ip` | `1 - distance` |
| `l2` (Chroma's default) | `1 - distance / 2` — equal to cosine similarity for unit-length embeddings, which Chroma's default model produces |

Promptise 1.2.1 and earlier created collections with Chroma's default `l2` space and scored them as `1 - distance`; typical distances are 1.2–1.5, so every score was `0.00` and any `min_score` above zero dropped every memory. Those collections now score correctly, and the provider logs a warning when it opens one. To move one to cosine distance, copy it into a new collection (the embeddings are reused, nothing is re-embedded):

```python
import chromadb

client = chromadb.PersistentClient(path=".promptise/chroma")
old = client.get_collection("agent_memory")
rows = old.get(include=["documents", "metadatas", "embeddings"])
new = client.create_collection("agent_memory_v2", metadata={"hnsw:space": "cosine"})
if rows["ids"]:
    new.add(
        ids=rows["ids"],
        documents=rows["documents"],
        metadatas=rows["metadatas"],
        embeddings=rows["embeddings"],
    )

provider = ChromaProvider(collection_name="agent_memory_v2", persist_directory=".promptise/chroma")
```

The `_promptise_user_id` ownership metadata is copied with the rows, so `PER_USER` isolation carries over. Delete the old collection (`client.delete_collection("agent_memory")`) once you have checked the new one.

### Mem0Provider

Wraps [Mem0](https://github.com/mem0ai/mem0) for hybrid vector + graph search. Can run fully local (with Ollama) or via the Mem0 cloud platform.

Supported releases: `mem0ai>=0.1.118,<3` — the 0.1, 1.x and 2.x lines (the `[all]` extra installs a release in that range). Mem0 2.0 changed `Memory.search()` and `Memory.get_all()` to take the user and agent ids in a `filters` dict and the result count as `top_k`, and rejects the old `user_id=` / `limit=` arguments; the provider reads the installed client's signature once and calls the form it accepts. Search errors from Mem0 are raised, not turned into an empty result, so an incompatible release or an unreachable vector store is visible (the agent still catches them and answers without memory).

```python
from promptise.memory import Mem0Provider, MemoryScope

# Per-user (recommended for multi-tenant deployments)
provider = Mem0Provider(scope=MemoryScope.PER_USER)

# user_id flows through from CallerContext on every call
await provider.add("User prefers dark mode", user_id="alice")
results = await provider.search("theme preferences", user_id="alice")
```

| Feature | Value |
|---|---|
| Search method | Hybrid vector + optional graph search |
| Persistence | Managed by Mem0 |
| Isolation | `SHARED` (default user) or `PER_USER` (per-call `user_id`; `delete` checks the entry's owner first) |
| Dependencies | `pip install "promptise[all]"` |
| Best for | Multi-user agents, knowledge graphs, cloud deployments |

Constructor parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `user_id` | `str` | `"default"` | Fallback owner when `scope=SHARED` or no per-call override given |
| `agent_id` | `str \| None` | `None` | Optional agent identifier for multi-agent scoping |
| `config` | `dict \| None` | `None` | Mem0 configuration dict, passed to `mem0.Memory.from_config()` (vector store, LLM, embedder) |
| `scope` | `MemoryScope` | `SHARED` | `PER_USER` makes per-call `user_id=` override the tenant for each operation |

!!! note "Mem0 runs its own LLM"
    `add()` asks Mem0's configured LLM to extract facts from the text, so Mem0 needs an LLM (and an embedder) of its own, set in `config`. If Mem0's default OpenAI model rejects a parameter Mem0 sends (for example `temperature` on a reasoning model), name the model in `config` — e.g. `{"llm": {"provider": "openai", "config": {"model": "gpt-5-mini", "is_reasoning_model": True}}}` on mem0ai 2.x. That error comes from Mem0's request, not from Promptise.

---

## MemoryAgent

`MemoryAgent` wraps any LangGraph agent with automatic memory context injection. Before every `ainvoke()`, it:

1. Extracts the user query from the input
2. Searches the memory provider for relevant content
3. Injects matching results as a `SystemMessage`
4. Invokes the inner agent
5. Optionally stores the exchange in memory (if `auto_store=True`)

```python
from promptise.memory import MemoryAgent, ChromaProvider

provider = ChromaProvider(persist_directory=".promptise/chroma")
memory_agent = MemoryAgent(
    inner=agent_graph,
    provider=provider,
    max_memories=5,
    min_score=0.3,
    timeout=5.0,
    auto_store=True,
)

result = await memory_agent.ainvoke({"messages": [{"role": "user", "content": "Hello"}]})
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `inner` | `Any` | required | The wrapped LangGraph agent |
| `provider` | `MemoryProvider` | required | Memory provider instance |
| `max_memories` | `int` | `5` | Max results to inject per invocation |
| `min_score` | `float` | `0.0` | Min relevance score threshold |
| `timeout` | `float` | `5.0` | Max seconds to wait for the memory search, and for an auto-store write before returning |
| `auto_store` | `bool` | `False` | Auto-store each exchange after invocation |

!!! tip "Graceful degradation"
    If the memory provider fails (timeout, connection error), the agent continues normally without memory context. Memory never blocks execution.

---

## Integration with build_agent

The simplest way to add memory is through `build_agent()`:

```python
from promptise import build_agent
from promptise.memory import ChromaProvider

provider = ChromaProvider(persist_directory=".promptise/chroma")

agent = await build_agent(
    servers={"tools": HTTPServerSpec(url="http://localhost:8000/mcp")},
    model="openai:gpt-5-mini",
    memory=provider,
)
```

The agent searches memory before every invocation and injects the results; no `MemoryAgent` wrapper is involved. These `build_agent()` parameters control it:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `memory` | `MemoryProvider` | `None` | The provider to search (and store into) |
| `memory_auto_store` | `bool` | `False` | Store each exchange (`User: ...` / `Assistant: ...`) after the invocation, scoped to the caller |
| `memory_max_results` | `int` | `5` | Maximum memories injected per invocation |
| `memory_min_score` | `float` | `0.0` | Drop results scoring below this (`0.0`–`1.0`) |
| `memory_timeout` | `float` | `5.0` | Seconds to wait for the search, and for an auto-store write before returning |

```python
from promptise.memory import ChromaProvider, MemoryScope

agent = await build_agent(
    servers=servers,
    model="openai:gpt-5-mini",
    memory=ChromaProvider(persist_directory=".promptise/chroma", scope=MemoryScope.PER_USER),
    memory_auto_store=True,
    memory_max_results=3,
    memory_min_score=0.3,   # cosine similarity — keeps related memories, drops noise
    memory_timeout=15.0,    # the first Chroma call loads the embedding model
)
```

### Timeouts

- **Search.** If the search takes longer than `memory_timeout`, the agent answers without memory context and logs `Memory search timed out after 5.0s; continuing without memory context`.
- **Auto-store.** The agent waits up to `memory_timeout` for the write. Provider writes run in a worker thread that cannot be cancelled, so a slow write is **not** abandoned: it keeps running in the background, the agent logs `Memory auto-store has not finished after 5.0s; continuing without waiting`, and when the write ends it logs the real outcome — `Memory auto-store completed after 7.2s` (INFO) or `Memory auto-store failed` with the error (WARNING). `agent.shutdown()` waits up to 30 seconds (or `memory_timeout`, if longer) for writes still in flight before closing the provider.

A cold `ChromaProvider` loads its embedding model on the first search or write, which can take longer than the 5-second default on a busy machine. Raise `memory_timeout`, or warm the provider up before the first request with `await provider.search("warm-up", limit=1, user_id=...)`.

---

## Integration with Agent Runtime

In the Agent Runtime, memory is configured through `ContextConfig`:

```python
from promptise.runtime import ProcessConfig, ContextConfig

config = ProcessConfig(
    model="openai:gpt-5-mini",
    context=ContextConfig(
        memory_provider="chroma",             # "in_memory", "chroma", or "mem0"
        memory_auto_store=True,               # Auto-store exchanges
        memory_max=5,                         # Max memories per invocation
        memory_min_score=0.3,                 # Min relevance score
        memory_timeout=5.0,                   # Seconds for search / auto-store
        memory_collection="agent_memory",     # ChromaDB collection name
        memory_persist_directory=".promptise/chroma",
        conversation_max_messages=50,         # Short-term buffer size
    ),
)
```

Or through a `.agent` manifest:

```yaml
memory:
  provider: chroma
  auto_store: true
  max: 5
  min_score: 0.3
  timeout: 5.0
  collection: agent_memory
  persist_directory: .promptise/chroma
```

---

## Security: Memory Sanitization

Injected memory content is sanitized before reaching the agent to mitigate prompt injection attacks. The `sanitize_memory_content()` function:

- Truncates content to a safe injection length (2,000 characters)
- Strips known prompt-injection patterns (`SYSTEM:`, `[INST]`, `<<SYS>>`, etc.)
- Removes memory fence markers to prevent content from escaping the context block

The injected memory block is wrapped in `<memory_context>` fences with explicit instructions that the agent should treat the content as factual context only and not follow any instructions within it.

---

## What's Next?

- [Sandbox](sandbox.md) -- execute untrusted code safely in isolated containers
- [Observability](observability.md) -- track token usage
- [Context & State](../runtime/context.md) -- AgentContext and state management
- [Conversation Management](../runtime/conversation.md) -- ConversationBuffer
