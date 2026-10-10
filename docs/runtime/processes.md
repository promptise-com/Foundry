# Agent Processes

`AgentProcess` wraps an agent with lifecycle management, event triggers, and configurable behavior. It is the core unit of the Agent Runtime.

```python
from promptise.runtime import AgentProcess, ProcessConfig, TriggerConfig

process = AgentProcess(
    name="health-checker",
    config=ProcessConfig(
        model="openai:gpt-5-mini",
        instructions="You monitor infrastructure health.",
        servers={"tools": {"type": "http", "url": "http://localhost:8000/mcp"}},
        triggers=[
            TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
        ],
        concurrency=2,
    ),
)

await process.start()
# Process runs autonomously on the cron schedule...
print(process.status())
await process.stop()
```

---

## ProcessConfig

`ProcessConfig` composes all configuration for a single agent process. Every field has a sensible default.

### Core fields

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"openai:gpt-5-mini"` | LLM model identifier (e.g. `openai:gpt-5-mini`) |
| `instructions` | `str \| None` | `None` | System prompt for the agent |
| `execution_mode` | `ExecutionMode` | `STRICT` | `STRICT` (immutable) or `OPEN` (self-modifying) |
| `open_mode` | `OpenModeConfig` | defaults | Guardrails for open mode (ignored in strict) |
| `servers` | `dict[str, Any]` | `{}` | MCP server specifications |
| `triggers` | `list[TriggerConfig]` | `[]` | Trigger configurations |
| `journal` | `JournalConfig` | off (`level="none"`) | Journal (audit log). Pass `JournalConfig(...)` to record transitions, invocations and checkpoints. See [Journal](journal/index.md). |
| `context` | `ContextConfig` | defaults | AgentContext configuration |
| `concurrency` | `int` | `1` | Max concurrent trigger invocations |
| `heartbeat_interval` | `float` | `10.0` | Heartbeat period in seconds |
| `idle_timeout` | `float` | `0.0` | Seconds of inactivity before suspending (0 = never) |
| `max_lifetime` | `float` | `0.0` | Max process lifetime in seconds (0 = unlimited) |
| `max_consecutive_failures` | `int` | `3` | Consecutive failures before FAILED state |
| `restart_policy` | `str` | `"never"` | `"always"`, `"on_failure"`, or `"never"` |
| `max_restarts` | `int` | `3` | Max consecutive restart attempts after failures |
| `restart_backoff` | `float` | `1.0` | Seconds before the first restart attempt; doubles per attempt, capped at 60 s |

### Governance fields

All governance is opt-in and zero-overhead when disabled. Each subsystem has its own config object with `enabled=False` by default.

| Field | Type | Default | Description |
|---|---|---|---|
| `secrets` | `SecretScopeConfig` | disabled | Per-process encrypted secret scoping with TTL, rotation, and access logging. See [Secret Scoping](governance/secrets.md). |
| `budget` | `BudgetConfig` | disabled | Per-run and daily limits on tool calls, LLM turns, cost, and irreversible actions. See [Autonomy Budget](governance/budget.md). |
| `health` | `HealthConfig` | disabled | Behavioral anomaly detection (stuck loops, empty responses, error rate). See [Behavioral Health](governance/health.md). |
| `mission` | `MissionConfig` | disabled | Mission-oriented execution with LLM-as-judge evaluation and success criteria. See [Mission Model](governance/mission.md). |
| `inbox` | `InboxConfig` | disabled | Human-in-the-loop message inbox for approvals and ad-hoc questions. |

### Full example

```python
from promptise.runtime import (
    ProcessConfig, TriggerConfig,
    BudgetConfig, HealthConfig, MissionConfig, JournalConfig,
    SecretScopeConfig, ToolCostAnnotation,
)

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Monitor the data pipeline and respond to incidents.",
    servers={"tools": {"type": "http", "url": "http://localhost:8000/mcp"}},
    triggers=[
        TriggerConfig(type="webhook", webhook_path="/alerts", webhook_port=9090),
        TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
    ],
    concurrency=2,

    # Governance — all opt-in
    budget=BudgetConfig(
        enabled=True,
        max_tool_calls_per_run=20,
        max_cost_per_day=50.0,
        on_exceeded="pause",
        tool_costs={
            "restart_service": ToolCostAnnotation(cost_weight=5.0, irreversible=True),
        },
    ),
    health=HealthConfig(
        enabled=True,
        stuck_threshold=3,
        empty_threshold=3,
        on_anomaly="escalate",
    ),
    mission=MissionConfig(
        enabled=True,
        objective="Keep all services above 99.9% uptime.",
        success_criteria="No unresolved P1 incidents for >15 minutes.",
        eval_every=10,
    ),
    secrets=SecretScopeConfig(
        secrets={"api_key": "${MONITORING_API_KEY}"},
        default_ttl=3600,
    ),
    journal=JournalConfig(backend="file", path="./pipeline-journal"),
)
```

!!! tip "Start with governance disabled"
    For local development, leave all governance disabled. Enable it one subsystem at a time as you move toward production. Budget first (to prevent runaway costs), then health (to catch stuck agents), then mission (to track long-horizon progress).

---

## Lifecycle Methods

`AgentProcess` exposes async methods for controlling the process lifecycle.

| Method | Transition | Description |
|---|---|---|
| `await process.start()` | CREATED -> STARTING -> RUNNING | Build agent, start triggers, enter main loop |
| `await process.stop()` | RUNNING -> STOPPING -> STOPPED | Stop triggers, cancel tasks, clean up |
| `await process.inject(event)` | -- | Push a `TriggerEvent` into the process queue |
| `process.status()` | -- | Return a status dict (state, invocations, queue, etc.) |
| `process.state` | -- | Current `ProcessState` enum value |
| `process.context` | -- | Access the `AgentContext` instance |

```python
# Manual event injection (useful for testing)
from promptise.runtime.triggers.base import TriggerEvent

await process.inject(TriggerEvent(
    trigger_id="manual",
    trigger_type="manual",
    payload={"reason": "Ad-hoc health check"},
))
```

The `status()` method returns a dict with fields including `state`, `invocation_count`, `queue_size`, `execution_mode`, `has_memory`, `conversation_messages`, `custom_tool_count`, and `rebuild_count`.

---

## Triggers

Triggers produce `TriggerEvent` objects that wake the process and cause it to invoke its agent. Five trigger types are available.

### CronTrigger

Fires on a cron schedule. Requires the `cron_expression` field.

```python
TriggerConfig(type="cron", cron_expression="*/5 * * * *")
```

### WebhookTrigger

Listens on an HTTP endpoint. Uses `aiohttp` (included in the base install).

```python
TriggerConfig(type="webhook", webhook_path="/webhook", webhook_port=9090)
```

### FileWatchTrigger

Watches a directory for filesystem changes. Uses `watchdog` (included in the base install).

```python
TriggerConfig(
    type="file_watch",
    watch_path="/data/inbox",
    watch_patterns=["*.csv", "*.json"],
    watch_events=["created", "modified"],
)
```

### EventTrigger

Fires on EventBus events (inter-process communication). Requires a shared `EventBus` instance.

```python
TriggerConfig(type="event", event_type="pipeline.degraded", event_source="monitor")
```

### MessageTrigger

Fires on MessageBroker messages. Requires a shared broker instance.

```python
TriggerConfig(type="message", topic="alerts")
```

### TriggerConfig Reference

| Field | Type | Applies to | Description |
|---|---|---|---|
| `type` | `str` | all | `"cron"`, `"webhook"`, `"file_watch"`, `"event"`, `"message"` |
| `cron_expression` | `str` | cron | Cron schedule (e.g. `"*/5 * * * *"`) |
| `webhook_path` | `str` | webhook | URL path (default `"/webhook"`) |
| `webhook_port` | `int` | webhook | Listen port (default `9090`) |
| `watch_path` | `str` | file_watch | Directory to watch |
| `watch_patterns` | `list[str]` | file_watch | Glob patterns (default `["*"]`) |
| `watch_events` | `list[str]` | file_watch | Events: `"created"`, `"modified"` |
| `event_type` | `str` | event | EventBus event type string |
| `event_source` | `str \| None` | event | Optional source filter |
| `topic` | `str` | message | MessageBroker topic |
| `filter_expression` | `str \| None` | all | Pre-filter before LLM invocation |

### Trigger Class Constructors

Each trigger type maps to a concrete class. These are typically created via `create_trigger()`, but you can also instantiate them directly.

**`CronTrigger`**

```python
from promptise.runtime.triggers.cron import CronTrigger

trigger = CronTrigger(
    cron_expression="*/5 * * * *",   # Standard cron expression (required)
    trigger_id="my-cron",            # Optional unique ID (auto-generated if omitted)
)
```

Uses `croniter` for full cron support if installed; falls back to simple `*/N * * * *` parsing otherwise.

**`EventTrigger`**

```python
from promptise.runtime.triggers.event import EventTrigger

trigger = EventTrigger(
    event_bus,                       # EventBus instance (required)
    "pipeline.degraded",             # Event type string to listen for (required)
    source_filter="monitor",         # Optional: only events from this source
    trigger_id="my-event",           # Optional unique ID
)
```

**`MessageTrigger`**

```python
from promptise.runtime.triggers.event import MessageTrigger

trigger = MessageTrigger(
    broker,                          # MessageBroker instance (required)
    "alerts",                        # Topic to subscribe to (required, supports wildcards)
    trigger_id="my-message",         # Optional unique ID
)
```

**`FileWatchTrigger`**

```python
from promptise.runtime.triggers.file_watch import FileWatchTrigger

trigger = FileWatchTrigger(
    watch_path="/data/inbox",        # Directory to monitor (required)
    patterns=["*.csv", "*.json"],    # Glob patterns to match (default: ["*"])
    events=["created", "modified"],  # Event types to react to (default: ["created", "modified"])
    recursive=True,                  # Watch subdirectories (default: True)
    debounce_seconds=0.5,            # Debounce interval to avoid duplicates (default: 0.5)
    poll_interval=1.0,               # Polling interval in seconds when watchdog unavailable (default: 1.0)
)
```

Uses `watchdog` for native OS notifications if installed; falls back to polling otherwise.

**`WebhookTrigger`**

```python
from promptise.runtime.triggers.webhook import WebhookTrigger

trigger = WebhookTrigger(
    path="/webhook",                 # URL path to listen on (default: "/webhook")
    port=9090,                       # TCP port to bind to (default: 9090)
    host="0.0.0.0",                  # Host/IP to bind to (default: "0.0.0.0")
)
```

Starts an `aiohttp` server. Includes a `/health` endpoint for liveness checks.

All trigger classes expose the same lifecycle interface: `await trigger.start()`, `await trigger.stop()`, and `await trigger.wait_for_next()` (returns a `TriggerEvent`).

### The `create_trigger()` Factory

The `create_trigger()` function converts a `TriggerConfig` into a trigger instance:

```python
from promptise.runtime import create_trigger, TriggerConfig

config = TriggerConfig(type="cron", cron_expression="0 * * * *")
trigger = create_trigger(config)  # Returns a CronTrigger
```

For `event` and `message` triggers, pass the shared `event_bus` or `broker` as keyword arguments.

---

## Concurrency Control

The `concurrency` field on `ProcessConfig` controls how many trigger events can be processed simultaneously. Internally, this is enforced with an `asyncio.Semaphore`.

```python
ProcessConfig(
    model="openai:gpt-5-mini",
    concurrency=3,  # Up to 3 invocations in parallel
)
```

---

## Restart Policies

When a process enters the `FAILED` state while running (for example after `max_consecutive_failures` failed invocations), the restart policy determines what happens next.

| Policy | Behavior |
|---|---|
| `"never"` | Process stays in FAILED state (default) |
| `"on_failure"` | Tear the process down and start it again, up to `max_restarts` consecutive attempts |
| `"always"` | Same as `"on_failure"`, and a process that reaches `max_lifetime` is recycled (stopped and started again) instead of staying stopped |

```python
ProcessConfig(
    model="openai:gpt-5-mini",
    restart_policy="on_failure",
    max_restarts=5,
    restart_backoff=2.0,   # 2s, 4s, 8s, ... (max 60s)
    max_consecutive_failures=3,
)
```

How restarts behave:

- The first attempt waits `restart_backoff` seconds; each further consecutive attempt doubles the wait (capped at 60 s). A restart that fails to start counts as an attempt.
- The attempt count resets after a successful invocation, so `max_restarts` limits a *streak* of failures, not the process's lifetime. A restarted process also starts a fresh `max_consecutive_failures` streak. `max_lifetime` recycling does not count as an attempt.
- When `max_restarts` is reached the process stays `FAILED` (a `restart_exhausted` journal entry is written).
- An explicit `stop()` never triggers a restart and cancels a pending one. Stops the runtime performs on purpose (budget or health `"stop"`, mission completion) are not restarted either.
- A failure of your own `start()` call raises to you and is not retried.
- Each attempt emits a `process.restarted` event and the `FAILED` transition emits `process.failed`. `status()` reports `restart_count`.

!!! note "Changed in 1.3.0"
    `restart_policy` and `max_restarts` were accepted but had no effect in earlier versions: a failed process always stayed `FAILED`.

---

## What's Next?

- [Agent Manifests](manifests.md) -- declare processes in YAML instead of code
- [Runtime Manager](runtime-manager.md) -- multi-process management with `AgentRuntime`
- [Agent Runtime Overview](index.md) -- architecture and lifecycle state diagram
- [Autonomy Budget](governance/budget.md) -- limit tool calls, cost, and irreversible actions
- [Behavioral Health](governance/health.md) -- detect stuck loops and anomalies
- [Mission Model](governance/mission.md) -- mission-oriented execution with LLM-as-judge
- [Secret Scoping](governance/secrets.md) -- per-process encrypted secrets
