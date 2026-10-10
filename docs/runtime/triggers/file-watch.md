# File Watch Trigger

The `FileWatchTrigger` monitors a directory for file changes and fires trigger events when files are created, modified, deleted, or moved. It uses `watchdog` for native OS filesystem notifications when available, with an automatic polling fallback.

```python
from promptise.runtime.triggers.file_watch import FileWatchTrigger

trigger = FileWatchTrigger(
    watch_path="/data/inbox",
    patterns=["*.csv", "*.json"],
)
await trigger.start()

event = await trigger.wait_for_next()
print(event.payload)
# {"path": "/data/inbox/new_data.csv", "filename": "new_data.csv",
#  "event_type": "created", "event_types": ["created", "modified"]}

await trigger.stop()
```

---

## Concepts

The `FileWatchTrigger` bridges the filesystem and the agent runtime. When files appear in or change within a watched directory, the trigger produces `TriggerEvent` objects that wake the agent. This is ideal for data ingestion pipelines, file-based workflows, and monitoring drop folders.

Two backends are supported:

- **Watchdog** (default) -- uses native OS filesystem notifications (inotify on Linux, FSEvents on macOS, ReadDirectoryChangesW on Windows). `watchdog` ships with `pip install promptise`.
- **Polling fallback** -- scans the directory at regular intervals, comparing file modification times. Works everywhere but uses more CPU and has higher latency.

---

## Configuration

### Via TriggerConfig

```python
from promptise.runtime import ProcessConfig, TriggerConfig

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Process new data files as they arrive.",
    triggers=[
        TriggerConfig(
            type="file_watch",
            watch_path="/data/inbox",
            watch_patterns=["*.csv", "*.json"],
            watch_events=["created", "modified"],
            watch_debounce_seconds=0.5,
        ),
    ],
)
```

| Field | Default | Description |
|---|---|---|
| `watch_path` | required | Directory to monitor |
| `watch_patterns` | `["*"]` | Glob patterns matched against the filename |
| `watch_events` | `["created", "modified"]` | Which (merged) events run the agent: any of `created`, `modified`, `deleted`, `moved` |
| `watch_debounce_seconds` | `0.5` | Window in which events for the same file are merged into one |
| `filter_expression` | `None` | Extra filter on the event (see [Filtering](index.md#filtering-events-before-the-agent-runs)) |

Unknown event names in `watch_events` are rejected when the config is created.

### Direct instantiation

```python
from promptise.runtime.triggers.file_watch import FileWatchTrigger

trigger = FileWatchTrigger(
    watch_path="/data/inbox",
    patterns=["*.csv", "*.json"],
    events=["created", "modified", "deleted"],
    recursive=True,
    debounce_seconds=0.5,
    poll_interval=1.0,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `watch_path` | `str` | required | Directory to monitor |
| `patterns` | `list[str]` | `["*"]` | Glob patterns to match filenames |
| `events` | `list[str]` | `["created", "modified"]` | Merged events that fire the trigger |
| `recursive` | `bool` | `True` | Watch subdirectories |
| `debounce_seconds` | `float` | `0.5` | Window in which events for one file are merged |
| `poll_interval` | `float` | `1.0` | Polling interval in seconds (fallback only) |

---

## Supported Events

| Event | Description |
|---|---|
| `created` | A new file was created in the watched directory |
| `modified` | An existing file's content was changed |
| `deleted` | A file was removed |
| `moved` | A file was moved or renamed (watchdog backend only) |

Configure which events to react to via the `events` parameter or `watch_events` in `TriggerConfig`. The check runs on the **merged** event for each file (see [Debouncing](#debouncing)), so `watch_events=["deleted"]` fires only when a file is gone, never for a new or changed one.

Editors and many tools save atomically: they write a temporary file and rename it over the target. Depending on the platform that shows up as `moved` (to the target name) rather than `modified`. Add `"moved"` to `watch_events` if you need to catch those saves.

---

## Pattern Matching

Patterns use standard glob syntax and are matched against the **filename** (not the full path):

```python
# Match CSV and JSON files
patterns=["*.csv", "*.json"]

# Match all Python files
patterns=["*.py"]

# Match everything (default)
patterns=["*"]

# Match specific prefixes
patterns=["report_*.xlsx"]
```

---

## Event Payload

When the trigger fires, the `TriggerEvent.payload` contains:

| Field | Description |
|---|---|
| `path` | Full filesystem path of the changed file |
| `filename` | Just the filename (basename) |
| `event_type` | The merged change: `"created"`, `"modified"`, `"deleted"`, or `"moved"` |
| `event_types` | The raw events merged into this one, in arrival order (e.g. `["created", "modified"]`) |

The `metadata` includes:

| Field | Description |
|---|---|
| `watch_path` | The configured watch directory |
| `patterns` | The configured glob patterns |

Example:

```python
event = await trigger.wait_for_next()
print(event.payload)
# {
#     "path": "/data/inbox/report_2026.csv",
#     "filename": "report_2026.csv",
#     "event_type": "created",
#     "event_types": ["created", "modified"],
# }
print(event.metadata)
# {
#     "watch_path": "/data/inbox",
#     "patterns": ["*.csv", "*.json"],
# }
```

---

## Debouncing

The operating system often reports one logical change as several events: writing a new file usually produces `created` **and** `modified`, and copying a large file can produce several `modified`. Without merging, the agent would run once per raw event.

The trigger therefore collects every event for the same path during a `debounce_seconds` window (starting with the first event) and then emits **one** trigger event describing the net change:

| What happened in the window | `event_type` |
|---|---|
| File no longer exists | `deleted` (nothing at all if it was also created in the window) |
| File was created (or deleted and recreated) | `created` |
| File was moved/renamed into place | `moved` |
| Anything else | `modified` |

That merged type is then checked against `events` / `watch_events`. One `write_text()` to a new file therefore runs the agent once, with `event_type="created"` and `event_types=["created", "modified"]`.

```python
# Merge everything that happens to a file within 1 second
trigger = FileWatchTrigger(
    watch_path="/data/inbox",
    debounce_seconds=1.0,
)
```

A larger window coalesces tools that write in several steps; the trade-off is that events arrive that much later.

---

## Directory Creation

If the `watch_path` does not exist when `start()` is called, the trigger creates it automatically:

```python
trigger = FileWatchTrigger(watch_path="/data/new_inbox")
await trigger.start()  # Creates /data/new_inbox if missing
```

---

## Backend Selection

The backend is selected automatically based on available dependencies:

```python
# With watchdog installed
trigger = FileWatchTrigger(watch_path="/data/inbox")
print(trigger)
# FileWatchTrigger(path='/data/inbox', patterns=['*'], backend='watchdog')

# Without watchdog
# FileWatchTrigger(path='/data/inbox', patterns=['*'], backend='polling')
```

### Watchdog backend

- Uses native OS notifications for near-instant detection.
- Handles `created`, `modified`, `deleted`, and `moved` events.
- Runs an `Observer` thread that dispatches events to the async queue via `call_soon_threadsafe`.

### Polling backend

- Scans the directory at `poll_interval` intervals.
- Compares file modification times to detect changes.
- Detects `created`, `modified`, and `deleted` events (not `moved`).
- Uses `asyncio.wait_for` with a stop event for cancellable sleeping.

---

## Lifecycle

```python
trigger = FileWatchTrigger(
    watch_path="/data/inbox",
    patterns=["*.csv"],
)

await trigger.start()

# Process events in a loop
while True:
    try:
        event = await trigger.wait_for_next()
        print(f"File: {event.payload['filename']} ({event.payload['event_type']})")
    except asyncio.CancelledError:
        break

await trigger.stop()
```

When `stop()` is called:

1. The watchdog observer is stopped and joined (or the polling task is cancelled).
2. A sentinel event is enqueued to unblock any waiting `wait_for_next()`.
3. The sentinel causes `wait_for_next()` to raise `asyncio.CancelledError`.

---

## API Summary

| Method / Property | Description |
|---|---|
| `FileWatchTrigger(watch_path, patterns, events, recursive, debounce_seconds, poll_interval)` | Create a file watch trigger (raises `ValueError` on unknown event names) |
| `trigger_id` | Unique identifier: `file_watch-{path}` |
| `await start()` | Start watching (creates directory if needed) |
| `await stop()` | Stop watching and release resources |
| `await wait_for_next()` | Block until a matching file change occurs |

---

## Tips and Gotchas

!!! tip "watchdog is already installed"
    Polling works but introduces latency equal to the `poll_interval`. Native filesystem notifications via `watchdog` detect changes nearly instantly. `watchdog` ships with the base `pip install promptise`.

!!! tip "Narrow your patterns"
    Use specific glob patterns to avoid processing temporary files, swap files, and other noise. For example, `["*.csv"]` is better than `["*"]` for a data ingestion pipeline.

!!! tip "Increase debounce for noisy directories"
    Some tools write files in multiple steps (create, write, flush) spread over more than half a second. Increase `debounce_seconds` to 1.0 or higher so they still merge into a single event.

!!! warning "Recursive watching can be expensive"
    Watching a large directory tree recursively may consume significant resources, especially with the polling backend. Monitor the queue size and consider watching specific subdirectories instead.

!!! warning "moved events are watchdog-only"
    The polling backend detects moves as a `deleted` event for the old path and a `created` event for the new path. Only the watchdog backend produces a single `moved` event.

!!! warning "Queue overflow"
    The file watch queue has a capacity of 1000 events. In directories with very high file churn, events may be dropped. Consider increasing `debounce_seconds` or narrowing `patterns` to reduce event volume.

!!! warning "File contents are untrusted input"
    The trigger passes only the path and file name, but an agent that then reads the file reads whatever someone dropped into the folder. Treat that content like a webhook body: gate side-effecting tools with approval (see [Trigger payloads are untrusted input](index.md#trigger-payloads-are-untrusted-input)).

---

## What's Next

- [Triggers Overview](index.md) -- all trigger types and the base protocol
- [Cron Trigger](cron.md) -- time-based scheduling
- [Event and Webhook Triggers](event-webhook.md) -- event-driven activation
