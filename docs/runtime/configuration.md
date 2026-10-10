# Configuration

All runtime configuration is expressed as Pydantic `BaseModel` subclasses with sensible defaults. Atomic configs (`TriggerConfig`, etc.) are composed into `ProcessConfig` (single process) and `RuntimeConfig` (multi-process manager).

```python
from promptise.runtime import (
    RuntimeConfig, ProcessConfig,
    TriggerConfig, JournalConfig, ContextConfig,
    DistributedConfig, ExecutionMode, OpenModeConfig,
)

config = RuntimeConfig(
    processes={
        "watcher": ProcessConfig(
            model="openai:gpt-5-mini",
            instructions="Monitor data pipelines.",
            triggers=[TriggerConfig(type="cron", cron_expression="*/5 * * * *")],
        ),
    },
)
```

---

## Concepts

The configuration system follows a **composition** pattern. Small, focused config objects are nested inside larger ones:

```
RuntimeConfig
  +-- processes: dict[str, ProcessConfig]
  |     +-- TriggerConfig (list)
  |     +-- JournalConfig
  |     +-- ContextConfig
  |     +-- OpenModeConfig
  +-- DistributedConfig
```

Every config field has a documented default value. You only need to specify what you want to override.

---

## RuntimeConfig

Top-level configuration that aggregates per-process configs and global settings.

```python
from promptise.runtime import RuntimeConfig, ProcessConfig

config = RuntimeConfig(
    processes={
        "agent-a": ProcessConfig(model="openai:gpt-5-mini"),
        "agent-b": ProcessConfig(model="openai:gpt-5-mini"),
    },
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `processes` | `dict[str, ProcessConfig]` | `{}` | Named process configurations |
| `distributed` | `DistributedConfig` | defaults | Distributed coordination settings |

### Serialization

```python
data = config.to_dict()      # JSON-compatible dict
restored = RuntimeConfig.from_dict(data)
```

### Presets

Two preset configurations ship with the framework:

| Preset | Description |
|---|---|
| `DEFAULT_DEVELOPMENT_CONFIG` | Local-only, distributed disabled |
| `DEFAULT_PRODUCTION_CONFIG` | Distributed enabled |

```python
from promptise.runtime.config import (
    DEFAULT_DEVELOPMENT_CONFIG,
    DEFAULT_PRODUCTION_CONFIG,
)
```

---

## ProcessConfig

Configuration for a single agent process. Composes all atomic configs and adds process-level settings.

```python
from promptise.runtime import ProcessConfig, TriggerConfig

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="You are a data monitoring agent.",
    triggers=[
        TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
    ],
    concurrency=1,
    heartbeat_interval=30.0,
    restart_policy="on_failure",
    max_restarts=3,
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | `"openai:gpt-5-mini"` | LLM model identifier |
| `instructions` | `str \| None` | `None` | System prompt for the agent |
| `execution_mode` | `ExecutionMode` | `STRICT` | `"strict"` (immutable) or `"open"` (self-modifying) |
| `open_mode` | `OpenModeConfig` | defaults | Guardrails for open mode (ignored in strict) |
| `servers` | `dict[str, Any]` | `{}` | MCP server specifications |
| `triggers` | `list[TriggerConfig]` | `[]` | Trigger configurations |
| `journal` | `JournalConfig` | defaults | Journal configuration |
| `context` | `ContextConfig` | defaults | AgentContext configuration |
| `concurrency` | `int` | `1` | Max concurrent trigger invocations (1-100) |
| `heartbeat_interval` | `float` | `10.0` | Heartbeat period in seconds |
| `idle_timeout` | `float` | `0.0` | Seconds idle before suspending (0 = never) |
| `max_lifetime` | `float` | `0.0` | Max process lifetime in seconds (0 = unlimited) |
| `max_consecutive_failures` | `int` | `3` | Consecutive failures before FAILED state |
| `restart_policy` | `str` | `"never"` | `"always"`, `"on_failure"`, or `"never"` |
| `max_restarts` | `int` | `3` | Max restart attempts |

---

## ExecutionMode

Controls whether the agent can modify itself at runtime.

| Mode | Description |
|---|---|
| `ExecutionMode.STRICT` | Agent cannot modify itself. This is the default and recommended for production. |
| `ExecutionMode.OPEN` | Agent can adapt its identity, memory, tools, and triggers at runtime via meta-tools. |

```python
from promptise.runtime.config import ExecutionMode

config = ProcessConfig(
    model="openai:gpt-5-mini",
    execution_mode=ExecutionMode.OPEN,
)
```

---

## OpenModeConfig

Guardrails for open execution mode. Only takes effect when `execution_mode` is `OPEN`.

| Field | Type | Default | Description |
|---|---|---|---|
| `allow_identity_change` | `bool` | `True` | Agent can modify its own instructions |
| `allow_tool_creation` | `bool` | `True` | Agent can define new Python tools |
| `allow_mcp_connect` | `bool` | `True` | Agent can connect to new MCP servers |
| `allow_trigger_management` | `bool` | `True` | Agent can add/remove triggers |
| `allow_memory_management` | `bool` | `True` | Agent can store/search/forget memories |
| `max_custom_tools` | `int` | `20` | Max number of agent-created tools |
| `max_dynamic_triggers` | `int` | `10` | Max dynamically added triggers |
| `max_instruction_length` | `int` | `10000` | Max character length for modified instructions |
| `max_rebuilds` | `int \| None` | `None` | Max agent rebuilds per lifetime (`None` = unlimited) |
| `allowed_mcp_urls` | `list[str]` | `[]` | Whitelist of MCP server URLs (empty = any) |
| `sandbox_custom_tools` | `bool` | `True` | Execute agent-written tools in a sandbox |
| `allow_process_spawn` | `bool` | `False` | Agent can spawn new processes in the runtime |
| `max_spawned_processes` | `int` | `3` | Max number of agent-spawned processes |

---

## TriggerConfig

Configuration for a single trigger. Each trigger `type` requires a different subset of fields. The `type` field accepts any string, including custom trigger types registered via `register_trigger_type()`.

```python
from promptise.runtime import TriggerConfig

# Cron trigger
TriggerConfig(type="cron", cron_expression="*/5 * * * *")

# Webhook trigger
TriggerConfig(type="webhook", webhook_path="/events", webhook_port=9090)

# File watch trigger
TriggerConfig(type="file_watch", watch_path="/data/inbox", watch_patterns=["*.csv"])

# Event trigger
TriggerConfig(type="event", event_type="task.completed")

# Message trigger
TriggerConfig(type="message", topic="reports.daily")

# Custom trigger type (must be registered first)
TriggerConfig(type="sqs", custom_config={"queue_url": "https://sqs..."})
```

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `str` | required | Any registered type: `"cron"`, `"webhook"`, `"file_watch"`, `"event"`, `"message"`, or custom |
| `cron_expression` | `str \| None` | `None` | Cron expression (required for `cron`); 5 fields, or 6 with trailing seconds |
| `cron_timezone` | `str \| None` | `None` | IANA time zone for the schedule (`None` = UTC) |
| `webhook_path` | `str` | `"/webhook"` | URL path (for `webhook`) |
| `webhook_port` | `int` | `9090` | Listen port (for `webhook`, 1025-65535) |
| `webhook_host` | `str` | `"127.0.0.1"` | Bind address (for `webhook`; `"0.0.0.0"` = all interfaces) |
| `hmac_secret` | `SecretStr \| None` | `None` | Webhook signature secret; unsigned requests get 401 |
| `signature_scheme` | `str` | `"generic"` | `"generic"`, `"github"` or `"stripe"` |
| `signature_header` | `str \| None` | `None` | Override the signature header |
| `signature_tolerance` | `int` | `300` | Max age (s) of a `stripe` signature timestamp |
| `allowed_sources` | `list[str]` | `[]` | Client IPs / CIDRs allowed to call the webhook (empty = any) |
| `watch_path` | `str \| None` | `None` | Directory to watch (required for `file_watch`) |
| `watch_patterns` | `list[str]` | `["*"]` | Glob patterns (for `file_watch`) |
| `watch_events` | `list[str]` | `["created", "modified"]` | Merged FS events to react to: `created`, `modified`, `deleted`, `moved` |
| `watch_debounce_seconds` | `float` | `0.5` | Window in which events for one file are merged |
| `event_type` | `str \| None` | `None` | EventBus event type (required for `event`) |
| `event_source` | `str \| None` | `None` | Optional source filter (for `event`) |
| `topic` | `str \| None` | `None` | Broker topic (required for `message`) |
| `filter_expression` | `str \| Callable \| None` | `None` | Safe expression or callable; non-matching events skip the agent ([syntax](triggers/index.md#filtering-events-before-the-agent-runs)) |
| `custom_config` | `dict[str, Any]` | `{}` | Arbitrary config for custom trigger types |

!!! warning "Validation for built-in types"
    A `cron` trigger without a valid `cron_expression`, a `file_watch` without `watch_path` or with unknown `watch_events`, an invalid `filter_expression`, time zone or `allowed_sources` entry, and any unknown key raise a `ValidationError` at construction time. Custom trigger types defer the rest of their validation to their factory function.

---

## JournalConfig

Controls the durable audit log for process events.

| Field | Type | Default | Description |
|---|---|---|---|
| `level` | `str` | `"checkpoint"` | `"none"` (disabled), `"checkpoint"` (per cycle), `"full"` (every side effect) |
| `backend` | `str` | `"file"` | `"file"` or `"memory"` |
| `path` | `str` | `".promptise/journal"` | Base directory for journal files |

---

## ContextConfig

Configures the `AgentContext` layer: state management, memory, environment, and file mounts.

| Field | Type | Default | Description |
|---|---|---|---|
| `writable_keys` | `list[str]` | `[]` | State keys the agent can write (empty = all writable) |
| `memory_provider` | `str \| None` | `None` | `"in_memory"`, `"chroma"`, `"mem0"`, or `None` |
| `memory_max` | `int` | `5` | Max memories injected per invocation |
| `memory_min_score` | `float` | `0.0` | Min relevance score for memory injection |
| `memory_auto_store` | `bool` | `False` | Auto-store exchanges in long-term memory |
| `memory_collection` | `str` | `"agent_memory"` | Collection name for ChromaDB |
| `memory_persist_directory` | `str \| None` | `None` | Persist directory for ChromaDB |
| `memory_user_id` | `str` | `"default"` | User ID for Mem0 scoping |
| `conversation_max_messages` | `int` | `100` | Max messages in conversation buffer (**0 = unlimited**) |
| `conversation_history` | `bool` | `True` | `False` turns short-term memory off: each run sees only its trigger event |
| `file_mounts` | `dict[str, str]` | `{}` | Logical name to filesystem path mapping |
| `env_prefix` | `str` | `"AGENT_"` | Prefix for exposed environment variables |
| `initial_state` | `dict[str, Any]` | `{}` | Pre-populated key-value state |

---

## DistributedConfig

Distributed runtime coordination settings.

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `False` | Enable distributed mode |
| `coordinator_url` | `str \| None` | `None` | Coordinator node URL |
| `transport_port` | `int` | `9100` | Management HTTP transport port (1025-65535) |
| `discovery_method` | `str` | `"registry"` | `"registry"` or `"multicast"` |
| `heartbeat_interval` | `float` | `15.0` | Node health check interval in seconds |

---

## Tips and Gotchas

!!! tip "Use presets as a starting point"
    `DEFAULT_DEVELOPMENT_CONFIG` and `DEFAULT_PRODUCTION_CONFIG` provide sensible defaults. Copy and customize rather than building from scratch.

!!! tip "Compose configs incrementally"
    Start with `ProcessConfig(model="openai:gpt-5-mini")` and add fields as you need them. Every sub-config has safe defaults.

!!! warning "TriggerConfig validation"
    Each trigger type requires specific fields. The Pydantic model validator enforces this at construction time, not at runtime. Always validate your config before deploying.

---

## What's Next

- [Runtime Manager](runtime-manager.md) -- using `AgentRuntime` to manage processes
- [Context](context.md) -- the unified state layer (`AgentContext`)
- [Triggers Overview](triggers/index.md) -- all trigger types and configuration
