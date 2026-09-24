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

- **Auto-injection** via `MemoryAgent` -- before every agent invocation, relevant memories are searched and injected as a `SystemMessage`. The agent sees contextual history without needing explicit memory tools.
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

When a memory provider is attached to an agent via `build_agent(memory=...)`, the wrapping `MemoryAgent` reads the current `CallerContext.user_id` from the async contextvar and passes it into every `search`/`add`/`delete` call automatically. Your handler code never has to thread `user_id` manually:

```python
from promptise.agent import CallerContext

# In your request handler
caller = CallerContext(user_id="alice", metadata={"session_id": "sess-1"})
await agent.ainvoke(input, caller=caller)
# → memory provider sees user_id="alice" on every call
```

If no caller is set (e.g., background tasks), the `MemoryAgent` catches `MemoryIsolationError` during auto-search and simply skips memory injection — the agent still runs.

### GDPR purge

```python
removed = await provider.purge_user("alice")   # → int count of deleted entries
```

Works on every provider. `InMemoryProvider` drops in-process entries. `ChromaProvider` deletes every entry whose metadata contains `_promptise_user_id: alice`. `Mem0Provider` delegates to Mem0's `delete_all(user_id=…)`.

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
| Search method | Vector similarity (cosine distance) |
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

### Mem0Provider

Wraps [Mem0](https://github.com/mem0ai/mem0) for hybrid vector + graph search. Can run fully local (with Ollama) or via the Mem0 cloud platform.

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
| Isolation | `SHARED` (default user) or `PER_USER` (per-call `user_id` override) |
| Dependencies | `pip install "promptise[all]"` |
| Best for | Multi-user agents, knowledge graphs, cloud deployments |

Constructor parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `user_id` | `str` | `"default"` | Fallback owner when `scope=SHARED` or no per-call override given |
| `agent_id` | `str \| None` | `None` | Optional agent identifier for multi-agent scoping |
| `config` | `dict \| None` | `None` | Mem0 configuration dict (for self-hosted setups) |
| `scope` | `MemoryScope` | `SHARED` | `PER_USER` makes per-call `user_id=` override the tenant for each operation |

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
| `timeout` | `float` | `5.0` | Max seconds to wait for memory search |
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

This automatically wraps the agent graph in a `MemoryAgent`.

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
