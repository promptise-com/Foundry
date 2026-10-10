# Replay Engine

The `ReplayEngine` reconstructs agent process state from journal entries after a crash. It reads the journal, finds the last checkpoint, and replays subsequent entries to rebuild the context state and lifecycle state -- enabling the process to resume where it left off.

```python
from promptise.runtime.journal import ReplayEngine, FileJournal

journal = FileJournal()
engine = ReplayEngine(journal)

recovered = await engine.recover("data-watcher")
# {
#     "context_state": {"pipeline_status": "degraded", "check_count": 42},
#     "lifecycle_state": "running",
#     "last_entry_type": "invocation_result",
#     "entries_replayed": 7,
# }
```

---

## Concepts

Long-running agent processes may crash due to hardware failures, OOM kills, or bugs. Without recovery, the process would restart from scratch, losing all accumulated state. The replay engine solves this by reconstructing state from the durable journal:

1. **Load the last checkpoint** -- the most recent full state snapshot.
2. **Read entries after the checkpoint** -- incremental changes since the snapshot.
3. **Replay entries sequentially** -- apply state transitions and context updates to rebuild the current state.

This is a classic **event sourcing** recovery pattern: the checkpoint is the snapshot, and journal entries are the event log.

---

## How Recovery Works

### Step 1: Load the last checkpoint

The engine calls `journal.last_checkpoint(process_id)`. If a checkpoint exists, it provides the starting state:

```python
{
    "context_state": {"pipeline_status": "healthy", "check_count": 40},
    "lifecycle_state": "running",
}
```

If no checkpoint exists, recovery starts from an empty state (`context_state={}`, `lifecycle_state="created"`).

### Step 2: Find entries after the checkpoint

The engine reads all journal entries for the process and locates the **last** `checkpoint` entry. Only the entries after that point are collected for replay; entries before it (older checkpoints included) are already part of the snapshot. The checkpoint returned by the backend is copied, so replay never modifies the stored snapshot.

!!! note "Fixed in 1.3.0"
    Earlier versions replayed everything after the *first* checkpoint, so with several checkpoints stale `context_update` values overwrote the newer snapshot.

### Step 3: Replay entries

Each entry is applied in chronological order:

- **`state_transition`** entries update the `lifecycle_state` from the `to_state` field.
- **`context_update`** entries update `context_state` by setting the `key` to `value`.
- Other entry types are noted (the last entry type is tracked) but do not modify the recovered state.

---

## Using the ReplayEngine

### Basic recovery

```python
from promptise.runtime.journal import ReplayEngine, FileJournal

journal = FileJournal(base_path=".promptise/journal")
engine = ReplayEngine(journal)

# Recover state for a process
recovered = await engine.recover("data-watcher")

print(f"Lifecycle state: {recovered['lifecycle_state']}")
print(f"Context state:   {recovered['context_state']}")
print(f"Last entry type: {recovered['last_entry_type']}")
print(f"Entries replayed: {recovered['entries_replayed']}")
```

### Recovery result

The `recover` method returns a dict with:

| Key | Type | Description |
|---|---|---|
| `context_state` | `dict[str, Any]` | Reconstructed context key-value state |
| `lifecycle_state` | `str` | Final lifecycle state (`"running"`, `"suspended"`, etc.) |
| `last_entry_type` | `str` | Type of the last replayed entry |
| `entries_replayed` | `int` | Number of entries replayed after the checkpoint |

### Applying recovered state

After recovery, apply the state to a new `AgentContext` and `ProcessLifecycle`:

```python
from promptise.runtime.context import AgentContext
from promptise.runtime.lifecycle import ProcessLifecycle, ProcessState

recovered = await engine.recover("data-watcher")

# Restore context
ctx = AgentContext(initial_state=recovered["context_state"])

# Restore lifecycle
lc = ProcessLifecycle(initial=ProcessState(recovered["lifecycle_state"]))
```

---

## Recovery Scenarios

### Scenario 1: Crash after several invocations

```
Journal entries:
  1. state_transition: created -> starting
  2. state_transition: starting -> running
  3. trigger_event: cron fired
  4. invocation_result: success
  5. checkpoint: {check_count: 10, lifecycle: "running"}
  6. trigger_event: cron fired
  7. context_update: check_count = 11
  8. invocation_result: success
  --- CRASH ---
```

Recovery:

1. Loads checkpoint from entry 5: `check_count=10`, `lifecycle="running"`
2. Replays entries 6-8: `context_update` sets `check_count=11`
3. Returns `context_state={"check_count": 11}`, `lifecycle_state="running"`, `entries_replayed=3`

### Scenario 2: Crash with no checkpoint

```
Journal entries:
  1. state_transition: created -> starting
  2. state_transition: starting -> running
  3. context_update: pipeline_status = "degraded"
  --- CRASH ---
```

Recovery:

1. No checkpoint found. Start from empty state.
2. Replay all 3 entries.
3. Returns `context_state={"pipeline_status": "degraded"}`, `lifecycle_state="running"`, `entries_replayed=3`

### Scenario 3: No journal entries

```python
recovered = await engine.recover("new-process")
# {"context_state": {}, "lifecycle_state": "created", "last_entry_type": "", "entries_replayed": 0}
```

---

## API Summary

| Method | Description |
|---|---|
| `ReplayEngine(journal)` | Create with a `JournalProvider` |
| `await recover(process_id)` | Recover state from journal entries |

---

## Tips and Gotchas

!!! tip "Frequent checkpoints reduce replay time"
    The replay engine only needs to process entries after the last checkpoint. More frequent checkpoints mean fewer entries to replay, but more disk writes. The default `"checkpoint"` journal level creates a checkpoint after each trigger-invoke-result cycle.

!!! tip "Use full journal level for detailed recovery"
    An `AgentProcess` checkpoints its context state after every invocation (both levels), and `level="full"` adds trigger events and tool calls for auditing. The process does not write `context_update` entries itself; append them from your own code when you need state reconstruction between checkpoints.

!!! warning "Replay does not restore memory providers"
    The replay engine reconstructs `context_state` (key-value store) but does not restore the `MemoryProvider`. Long-term memory is managed by its own persistence layer (ChromaDB, Mem0, etc.) and does not need journal-based recovery.

!!! warning "Replay does not restore conversation buffer"
    The conversation buffer (short-term message history) is not replayed from journal entries. If you need conversation persistence across crashes, include it in checkpoint data.

!!! warning "Idempotency matters"
    The replay engine assumes entries can be applied sequentially. If your journal contains entries with side effects beyond state mutations (e.g., external API calls), those side effects are not replayed -- only the state changes recorded in the journal.

---

## What's Next

- [Journal Overview](index.md) -- journal concepts and entry types
- [Journal Backends](backends.md) -- `FileJournal` and `InMemoryJournal`
- [Lifecycle](../lifecycle.md) -- the state machine that produces `state_transition` entries
- [Context](../context.md) -- the state layer that produces `context_update` entries
