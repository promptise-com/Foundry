# Conversation Persistence

Persist conversation history across sessions with pluggable database backends. Every chat application needs this — Promptise makes it one line of configuration.

## Concepts

The **conversation store** separates chat history from the agent. The agent remains stateless — the store handles loading, saving, and managing conversations. This means:

- **Sessions are isolated** — each `session_id` has its own message history
- **History survives restarts** — conversations persist in your database
- **Backends are swappable** — switch from SQLite to PostgreSQL without changing application code
- **Custom stores are trivial** — implement 4 async methods and you're done

**Conversation store vs. memory**: These serve different purposes and work together.

| | Conversation Store | Memory Provider |
|---|---|---|
| **What it stores** | Exact chat messages, in order | Semantic facts and knowledge |
| **How it retrieves** | By session ID (exact lookup) | By similarity search (fuzzy) |
| **Scope** | Per-session | Cross-session |
| **Purpose** | "What did they say 3 messages ago?" | "What do I know about this user?" |

## Quick Start

```python
from promptise import build_agent
from promptise.conversations import SQLiteConversationStore

store = SQLiteConversationStore("conversations.db")

agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={...},
    conversation_store=store,
)

# Each call to chat() automatically loads and saves history
response = await agent.chat("Hello!", session_id="conv-123")
response = await agent.chat("What did I say?", session_id="conv-123")  # Has full context

# Different session = fresh conversation
response = await agent.chat("New topic", session_id="conv-456")
```

## Built-in Stores

### InMemoryConversationStore

Dict-backed, no persistence. For testing and prototyping.

```python
from promptise.conversations import InMemoryConversationStore

store = InMemoryConversationStore(
    max_sessions=1000,              # Evict oldest when full (0 = unlimited)
    max_messages_per_session=200,   # Rolling window per session (0 = unlimited)
)
```

### PostgresConversationStore

Production store using `asyncpg`. Auto-creates tables and indexes on first use.

```python
from promptise.conversations import PostgresConversationStore

store = PostgresConversationStore(
    "postgresql://user:pass@localhost/mydb",
    table_prefix="promptise_",     # Safe for multi-app databases
    pool_min=2,                     # Connection pool sizing
    pool_max=10,
    max_messages_per_session=0,    # 0 = unlimited
)
```

Requires: `pip install asyncpg`

**Auto-created schema:**

- `{prefix}sessions` — session metadata (id, user, title, timestamps)
- `{prefix}messages` — messages with role, content, metadata, timestamps
- Indexes on `(session_id, created_at)` and `(user_id, updated_at)`

### SQLiteConversationStore

Local development store using `aiosqlite`. WAL mode enabled for concurrent reads.

```python
from promptise.conversations import SQLiteConversationStore

store = SQLiteConversationStore(
    "conversations.db",            # Or ":memory:" for testing
    table_prefix="promptise_",
    max_messages_per_session=0,
)
```

Requires: `pip install aiosqlite`

### RedisConversationStore

Fast ephemeral sessions using `redis.asyncio`. Ideal for caching or high-throughput scenarios with optional TTL-based expiry.

```python
from promptise.conversations import RedisConversationStore

store = RedisConversationStore(
    "redis://localhost:6379",
    key_prefix="promptise:",       # Namespace all keys
    ttl=86400,                      # Sessions expire after 24h (0 = no expiry)
    max_messages_per_session=100,
)
```

Requires: `pip install redis`

## The chat() API

`agent.chat()` is the high-level interface. It handles the full cycle:

1. **Ownership check** — verify session belongs to caller (when `user_id` or `caller` is provided)
2. Load history from store (by `session_id`)
3. Build LangChain message list with history + new user message
4. Invoke the agent (with memory injection, prompt context, etc.)
5. Extract assistant response
6. Persist updated history to store + assign ownership on new sessions

```python
from promptise.conversations import generate_session_id

# Always use generate_session_id() — never user-controlled or predictable IDs
sid = generate_session_id()  # "sess_a1b2c3d4e5f6..."

# Multi-user: user_id enables ownership enforcement. It is shorthand for
# caller=CallerContext(user_id="user-42"): per-user memory, the semantic cache,
# guardrails and observability see the same user for this call.
response = await agent.chat(
    "Hello",
    session_id=sid,
    user_id="user-42",
    metadata={"source": "web", "ip": "1.2.3.4"},
)

# Same user, same session — works
response = await agent.chat("Follow up", session_id=sid, user_id="user-42")

# Different user, same session — raises SessionAccessDenied
response = await agent.chat("Hack", session_id=sid, user_id="attacker")

# With a per-call system prompt override
response = await agent.chat(
    "Summarize this",
    session_id=sid,
    user_id="user-42",
    system_prompt="You are a concise summarizer. Max 2 sentences.",
)
```

## Multi-User Ownership Model

When `user_id` is provided, every operation enforces ownership:

```python
from promptise.conversations import SessionAccessDenied, generate_session_id

sid = generate_session_id()

# First chat assigns ownership
await agent.chat("Hello", session_id=sid, user_id="alice")

# Alice can continue — same user
await agent.chat("More", session_id=sid, user_id="alice")

# Bob cannot access Alice's session
try:
    await agent.chat("Hello", session_id=sid, user_id="bob")
except SessionAccessDenied as e:
    print(e)  # "User 'bob' denied access to session 'sess_...' (owned by 'alice')"
```

**Ownership rules:**

| Scenario | Behavior |
|----------|----------|
| `user_id` provided, session is new | Session assigned to that user |
| `user_id` provided, session belongs to same user | Access granted |
| `user_id` provided, session belongs to different user | `SessionAccessDenied` raised |
| `user_id=None`, session exists | Access granted (no enforcement) |
| `user_id` provided, session has no owner | Access granted (unowned) |

Ownership is enforced on `chat()`, `delete_session()`, `update_session()`, and `get_session()`. The `list_sessions()` method filters by `user_id` at the database level — users only see their own sessions.

## Session Management

```python
# List sessions for a specific user
sessions = await agent.list_sessions(user_id="user-42")

# Paginate
page2 = await agent.list_sessions(user_id="user-42", limit=20, offset=20)

# Get a specific session (with ownership check)
info = await agent.get_session("sess_abc", user_id="user-42")

# Update session metadata (with ownership check)
await agent.update_session(
    "sess_abc",
    calling_user_id="user-42",  # Verify ownership
    title="Support Chat #42",
)

# Delete a session (with ownership check)
await agent.delete_session("sess_abc", user_id="user-42")
```

## Custom Stores

Implement 5 data methods plus `close()`. The protocol is intentionally small — no auth logic in the store, just data:

```python
from promptise.conversations import ConversationStore, Message, SessionInfo

class MongoConversationStore:
    """MongoDB-backed conversation store."""

    def __init__(self, db):
        self._db = db

    async def get_session(self, session_id: str) -> SessionInfo | None:
        doc = await self._db.sessions.find_one({"_id": session_id})
        if not doc:
            return None
        return SessionInfo(session_id=doc["_id"], **doc)

    async def load_messages(self, session_id: str) -> list[Message]:
        docs = await self._db.messages.find(
            {"session_id": session_id}
        ).sort("created_at").to_list(None)
        return [Message(role=d["role"], content=d["content"]) for d in docs]

    async def save_messages(self, session_id: str, messages: list[Message]) -> None:
        async with await self._db.client.start_session() as session:
            async with session.start_transaction():
                await self._db.messages.delete_many({"session_id": session_id})
                if messages:
                    await self._db.messages.insert_many([
                        {"session_id": session_id, **m.to_dict()}
                        for m in messages
                    ])
                await self._db.sessions.update_one(
                    {"_id": session_id},
                    {"$set": {"message_count": len(messages)}},
                    upsert=True,
                )

    async def delete_session(self, session_id: str) -> bool:
        result = await self._db.sessions.delete_one({"_id": session_id})
        await self._db.messages.delete_many({"session_id": session_id})
        return result.deleted_count > 0

    async def list_sessions(
        self, *, user_id=None, limit=50, offset=0
    ) -> list[SessionInfo]:
        query = {"user_id": user_id} if user_id else {}
        docs = await self._db.sessions.find(query).sort(
            "updated_at", -1
        ).skip(offset).limit(limit).to_list(None)
        return [SessionInfo(session_id=d["_id"], **d) for d in docs]

    async def close(self) -> None:
        pass  # MongoDB client manages its own lifecycle
```

The agent handles all authorization on top of your store. Your store just reads and writes data — it never needs to know about users or permissions.

## Combining with Memory

Conversation store and memory provider are orthogonal — use both:

```python
from promptise import build_agent
from promptise.memory import ChromaProvider
from promptise.conversations import PostgresConversationStore

agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={...},

    # Long-term semantic memory (cross-session knowledge)
    memory=ChromaProvider(persist_directory=".promptise/chroma"),
    memory_auto_store=True,

    # Per-session chat history (exact message replay)
    conversation_store=PostgresConversationStore("postgresql://localhost/mydb"),
    conversation_max_messages=200,
)

# chat() uses both: memory injects relevant context, store provides exact history
response = await agent.chat("Continue where we left off", session_id="s1", user_id="user-42")
```

## Security

**Session isolation:**

- **Ownership enforcement** — when `user_id` is provided, the agent verifies session ownership before every read, write, delete, and update. `SessionAccessDenied` (a `PermissionError`) is raised on mismatch.
- **Secure session IDs** — `generate_session_id()` creates 32-character hex tokens using `secrets.token_hex()`. Non-enumerable, unpredictable.
- **Scoped listing** — `list_sessions(user_id=...)` filters at the database level, never exposing other users' sessions.

**Data protection:**

- **Table prefix validation** — SQL stores reject prefixes containing anything other than `[a-zA-Z0-9_]` to prevent SQL injection.
- **Parameterized queries** — all user data passes through parameterized queries, never interpolated into SQL.
- **Connection pooling** — PostgreSQL store uses `asyncpg` connection pools with configurable min/max.
- **Lazy initialization** — database connections are created on first use, not at construction time.
- **Store failures** — see [When the store fails](#when-the-store-fails).

## When the store fails

`chat()` treats a store failure differently depending on what it was doing. Verifying ownership fails closed; reading and writing history degrade gracefully:

| Step | If the store raises (DB down, network error, …) |
|---|---|
| Ownership check (`get_session`, before anything else) | `chat()` raises `RuntimeError("Cannot verify session ownership (store unreachable): ...")`. The model is **not** called: an agent that cannot tell whose session this is must not answer in it. |
| Loading history (`load_messages`) | Logged (`Failed to load conversation history`); the agent answers without the earlier messages. |
| Saving history (`save_messages`) | Logged (`Failed to save conversation history`); the reply is returned but this turn is not persisted. |
| Assigning ownership of a new session (`update_session`) | Logged as a warning (`Failed to assign session ... to its user`); the reply is returned, and the session has no owner until one is assigned, so ownership is not enforced for it. |

Handle the `RuntimeError` the way you handle any other unavailable dependency — for example, return HTTP 503 and let the client retry:

```python
from fastapi import HTTPException
from promptise.conversations import SessionAccessDenied

try:
    reply = await agent.chat(text, session_id=sid, user_id=user.id)
except SessionAccessDenied:
    raise HTTPException(status_code=403)
except RuntimeError:
    raise HTTPException(status_code=503, detail="Conversation store unavailable")
```

## API Summary

| Symbol | Description |
|--------|-------------|
| `ConversationStore` | Protocol — implement this for custom backends |
| `Message` | A single conversation message (role, content, metadata, timestamp) |
| `SessionInfo` | Session metadata (id, user, title, counts, timestamps) |
| `SessionAccessDenied` | Raised when a user accesses a session they don't own |
| `generate_session_id()` | Create a cryptographically secure session ID |
| `InMemoryConversationStore` | Dict-backed store for testing |
| `PostgresConversationStore` | Production store via asyncpg |
| `SQLiteConversationStore` | Local dev store via aiosqlite |
| `RedisConversationStore` | Ephemeral store via redis.asyncio |
| `agent.chat()` | High-level chat with ownership enforcement and session persistence |
| `agent.get_session()` | Get session metadata with ownership check |
| `agent.list_sessions()` | List sessions with pagination and user filtering |
| `agent.delete_session()` | Delete a session with ownership check |
| `agent.update_session()` | Update session metadata with ownership check |
