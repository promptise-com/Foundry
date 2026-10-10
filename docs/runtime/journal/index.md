# Journal System

The journal system provides durable audit logging for agent processes. Every state transition, trigger firing, invocation result, and checkpoint is recorded as a `JournalEntry`, enabling crash recovery, observability, and post-mortem analysis.

```python
from promptise.runtime.journal import FileJournal, JournalEntry

journal = FileJournal(base_path=".promptise/journal")

await journal.append(JournalEntry(
    process_id="data-watcher",
    entry_type="trigger_event",
    data={"trigger_type": "cron", "scheduled_time": "2026-03-04T10:05:00"},
))

entries = await journal.read("data-watcher", limit=10)
```

---

## Concepts

The journal sits between the agent runtime and durable storage, capturing a chronological record of everything that happens in a process. It serves two primary purposes:

1. **Observability** -- view what happened, when, and why. The CLI `promptise runtime logs` command reads from the journal.
2. **Crash recovery** -- after a crash, the `ReplayEngine` reads the journal, finds the last checkpoint, and replays subsequent entries to reconstruct the process state.

Three detail levels control how much is recorded:

| Level | What is recorded | When to use |
|---|---|---|
| `"none"` | Nothing | Fire-and-forget processes where history does not matter |
| `"checkpoint"` | State transitions, invocation results and errors, restarts, and a state snapshot after each trigger-invoke-result cycle | Default for a `JournalConfig`. Good balance of observability and storage efficiency |
| `"full"` | Everything above, plus each trigger event, invocation start, and every tool call and tool result | Debugging, audit trails, compliance requirements |

### Journaling an agent process

An `AgentProcess` writes its journal when `ProcessConfig.journal` sets a level. Journaling is **off by default** for a `ProcessConfig`; turn it on per process:

```python
from promptise.runtime import ProcessConfig
from promptise.runtime.config import JournalConfig

config = ProcessConfig(
    model="openai:gpt-5-mini",
    journal=JournalConfig(level="checkpoint", backend="file", path=".promptise/journal"),
)
```

Entries are filed under the **process name** (not its internal ID), so `promptise runtime logs <name>` and recovery after a restart find them. Secret-access and mission-evaluation entries from the governance subsystems go to the same journal. Checkpoints hold `context_state`, `lifecycle_state`, `invocation_count`, the conversation buffer, and the budget and mission state when those are enabled. A failing journal backend is logged and never stops the process.

!!! note "Changed in 1.3.0"
    Earlier versions accepted `ProcessConfig.journal` but never wrote anything. Because a process journal is now real, the `ProcessConfig` default became `level="none"` so upgrading does not start writing files to `.promptise/journal`.

---

## JournalEntry

Every journal record is a `JournalEntry` dataclass:

```python
from promptise.runtime.journal import JournalEntry

entry = JournalEntry(
    process_id="data-watcher",
    entry_type="state_transition",
    data={"from_state": "created", "to_state": "running"},
)
```

| Field | Type | Description |
|---|---|---|
| `entry_id` | `str` | Unique entry ID (auto-generated UUID) |
| `process_id` | `str` | Owning process identifier |
| `timestamp` | `datetime` | When the entry was recorded (UTC) |
| `entry_type` | `str` | Type of entry |
| `data` | `dict[str, Any]` | Entry-specific payload |

### Entry types

| Entry Type | Description |
|---|---|
| `state_transition` | Process state change (from/to state) |
| `trigger_event` | A trigger fired (trigger type, payload) |
| `invocation_start` | Agent invocation began |
| `invocation_result` | Agent invocation completed (final reply, duration) |
| `checkpoint` | Full state snapshot |
| `context_update` | Context state key changed (written by your code; replayed by `ReplayEngine`) |
| `tool_call` / `tool_result` | A tool call and its result (`"full"` level) |
| `restart` / `restart_exhausted` | Automatic restart attempt / `max_restarts` reached |
| `error` | An invocation failed |

### Serialization

```python
data = entry.to_dict()
restored = JournalEntry.from_dict(data)
```

---

## JournalLevel

Controls the detail level of journaling:

```python
from promptise.runtime.journal import JournalLevel

JournalLevel.NONE        # No journaling
JournalLevel.CHECKPOINT  # State snapshots per cycle (default)
JournalLevel.FULL        # Every side effect
```

Configure via `JournalConfig`:

```python
from promptise.runtime.config import JournalConfig

# Checkpoint level with file backend
JournalConfig(level="checkpoint", backend="file", path=".promptise/journal")

# Full level for debugging
JournalConfig(level="full", backend="file")

# Disabled
JournalConfig(level="none")
```

---

## JournalProvider Protocol

All journal backends implement the `JournalProvider` protocol:

```python
from promptise.runtime.journal import JournalProvider

class JournalProvider(Protocol):
    async def append(self, entry: JournalEntry) -> None:
        """Append an entry to the journal."""
        ...

    async def read(
        self,
        process_id: str,
        *,
        since: datetime | None = None,
        entry_type: str | None = None,
        limit: int | None = None,
    ) -> list[JournalEntry]:
        """Read entries with optional filters."""
        ...

    async def checkpoint(
        self, process_id: str, state: dict[str, Any]
    ) -> None:
        """Store a full state checkpoint."""
        ...

    async def last_checkpoint(
        self, process_id: str
    ) -> dict[str, Any] | None:
        """Return the most recent checkpoint, or None."""
        ...

    async def close(self) -> None:
        """Release any resources."""
        ...
```

---

## Available Backends

| Backend | Class | Persistence | Use Case |
|---|---|---|---|
| File | `FileJournal` | Append-only JSONL files on disk | Production |
| Memory | `InMemoryJournal` | Python list in memory | Testing and development |

See [Journal Backends](backends.md) for detailed documentation of each backend.

---

## Reading Entries

```python
from promptise.runtime.journal import FileJournal

journal = FileJournal()

# Read all entries for a process
entries = await journal.read("data-watcher")

# Read with filters
from datetime import datetime, UTC
entries = await journal.read(
    "data-watcher",
    since=datetime(2026, 3, 4, tzinfo=UTC),
    entry_type="trigger_event",
    limit=20,
)

# Read the last checkpoint
checkpoint = await journal.last_checkpoint("data-watcher")
```

---

## Checkpointing

Checkpoints store a full state snapshot that the `ReplayEngine` uses as a recovery point:

```python
await journal.checkpoint("data-watcher", {
    "context_state": {"pipeline_status": "healthy", "check_count": 42},
    "lifecycle_state": "running",
    "invocation_count": 42,
})
```

The checkpoint is stored separately from the regular journal entries (as a dedicated file in the file backend) and also recorded as a journal entry with `entry_type="checkpoint"`.

---

## API Summary

| Class / Enum | Description |
|---|---|
| `JournalEntry` | Single journal record dataclass |
| `JournalLevel` | Detail level enum: `NONE`, `CHECKPOINT`, `FULL` |
| `JournalProvider` | Protocol for journal backends |
| `FileJournal` | Append-only JSONL file backend |
| `InMemoryJournal` | In-memory backend for testing |
| `ReplayEngine` | Crash recovery from journal entries |

---

## Tips and Gotchas

!!! tip "Use checkpoint level for production"
    The `"checkpoint"` level captures enough information for crash recovery without the storage overhead of `"full"`. Reserve `"full"` for debugging specific issues.

!!! tip "View logs via CLI"
    Use `promptise runtime logs <process-name>` to view journal entries in a formatted table. Add `--lines 50` to see more entries.

!!! warning "Journal entries are append-only"
    Neither backend supports deleting individual entries. To clear a process's journal, delete the JSONL file manually (file backend) or create a new `InMemoryJournal` instance.

!!! warning "Checkpoint state must be JSON-serializable"
    The checkpoint data dict is serialized to JSON. Ensure all values are JSON-compatible (strings, numbers, booleans, lists, dicts, None).

---

## What's Next

- [Journal Backends](backends.md) -- `FileJournal` and `InMemoryJournal` in detail
- [Replay Engine](replay.md) -- crash recovery from journal entries
- [Configuration](../configuration.md) -- `JournalConfig` reference
- [CLI Commands](../cli.md) -- `promptise runtime logs` command
