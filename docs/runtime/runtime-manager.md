# Runtime Manager

The `AgentRuntime` is the central orchestrator for long-running agent processes. Think of it as the "Docker daemon" equivalent for AI agents: it manages a registry of `AgentProcess` instances, coordinates their lifecycles, and provides both programmatic and CLI-driven control surfaces.

```python
from promptise.runtime import AgentRuntime, ProcessConfig, TriggerConfig

async with AgentRuntime() as runtime:
    await runtime.add_process(
        "data-watcher",
        ProcessConfig(
            model="openai:gpt-5-mini",
            instructions="You monitor data pipelines and alert on anomalies.",
            triggers=[
                TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
            ],
        ),
    )
    await runtime.start_all()
    print(runtime.status())
    await runtime.stop_all()
```

---

## Concepts

**AgentRuntime** holds a dictionary of named `AgentProcess` instances. Each process wraps a fully configured agent (LLM, tools, triggers, context) and runs independently. The runtime provides shared resources -- an optional event bus for inter-process events and a message broker for message-based triggers. Both accept duck-typed objects, so you can plug in any implementation (Redis, Kafka, in-memory, etc.).

Key responsibilities:

- **Process registry** -- add, remove, and look up processes by name.
- **Manifest loading** -- create processes from declarative `.agent` YAML files.
- **Lifecycle coordination** -- start, stop, and restart individual or all processes.
- **Status monitoring** -- query per-process metrics.

---

## Creating a Runtime

### Default configuration

```python
from promptise.runtime import AgentRuntime

runtime = AgentRuntime()
```

### With shared event bus and message broker

The `event_bus` and `broker` parameters accept any object that implements the expected subscribe/publish interface. This lets you plug in your own event infrastructure (Redis Pub/Sub, Kafka, or a simple in-memory implementation):

```python
from promptise.runtime import AgentRuntime

# Any object with subscribe(event_type, callback) and emit(event_type, data)
event_bus = MyEventBus()

# Any object with subscribe(topic, callback) and publish(topic, message)
broker = MyMessageBroker()

runtime = AgentRuntime(event_bus=event_bus, broker=broker)
```

Event triggers subscribe via `event_bus.subscribe()` and message triggers subscribe via `broker.subscribe()`. The runtime passes these objects through to each `AgentProcess`.

### From a full RuntimeConfig with pre-defined processes

```python
from promptise.runtime import AgentRuntime, RuntimeConfig, ProcessConfig

config = RuntimeConfig(
    processes={
        "watcher": ProcessConfig(
            model="openai:gpt-5-mini",
            instructions="Monitor pipelines.",
        ),
        "responder": ProcessConfig(
            model="openai:gpt-5-mini",
            instructions="Handle alerts.",
        ),
    },
)

runtime = await AgentRuntime.from_config(config)
```

---

## Registering Processes

```python
from promptise.runtime import ProcessConfig, TriggerConfig

process = await runtime.add_process(
    "data-watcher",
    ProcessConfig(
        model="openai:gpt-5-mini",
        instructions="You monitor data pipelines.",
        triggers=[
            TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
        ],
    ),
)
```

Each name must be unique. Attempting to register a duplicate raises `RuntimeBaseError`. To replace a process, remove it first:

```python
await runtime.remove_process("data-watcher")
```

`remove_process` stops the process automatically if it is still running before removing it from the registry.

---

## Loading Manifests

Instead of constructing `ProcessConfig` objects in code, you can load processes from `.agent` YAML manifest files:

```python
# Load a single manifest
process = await runtime.load_manifest("agents/watcher.agent")

# Override the process name from the manifest
process = await runtime.load_manifest(
    "agents/watcher.agent", name_override="my-watcher"
)

# Load all .agent files from a directory
names = await runtime.load_directory("agents/")
# names = ["data-watcher", "alert-handler", ...]
```

---

## Lifecycle Control

### Starting and stopping

```python
# Start all registered processes
await runtime.start_all()

# Stop all running processes
await runtime.stop_all()

# Start a single process by name
await runtime.start_process("data-watcher")

# Stop a single process
await runtime.stop_process("data-watcher")

# Restart (stop then start)
await runtime.restart_process("data-watcher")
```

`start_all()` only starts processes in `CREATED`, `STOPPED`, or `FAILED` states. `stop_all()` skips processes that are already stopped.

When the runtime has an `event_notifier`, it is shared by every process. A process that stops no longer shuts the shared notifier down; `stop_all()` drains it after the last process has stopped, so every `process.stopped` event is delivered before `stop_all()` returns.

### Context manager

`AgentRuntime` supports `async with` for automatic cleanup. When the context exits, `stop_all()` is called:

```python
async with AgentRuntime() as runtime:
    await runtime.add_process("watcher", config)
    await runtime.start_all()
    # runtime manages processes until the block exits
# stop_all() is called automatically
```

---

## Status and Monitoring

### Global status

```python
status = runtime.status()
# {
#     "process_count": 2,
#     "processes": {
#         "watcher": {"state": "running", "invocation_count": 5, ...},
#         "responder": {"state": "running", "invocation_count": 3, ...},
#     },
# }
```

### Single process status

```python
info = runtime.process_status("data-watcher")
```

### Listing processes

```python
processes = runtime.list_processes()
# [
#     {"name": "watcher", "state": "running", "process_id": "abc123..."},
#     {"name": "responder", "state": "stopped", "process_id": "def456..."},
# ]
```

### Accessing processes directly

```python
process = runtime.get_process("data-watcher")

# Read-only dict of all processes
all_procs = runtime.processes
```

---

## API Summary

| Method / Property | Description |
|---|---|
| `AgentRuntime(config, event_bus, broker)` | Create a runtime instance |
| `AgentRuntime.from_config(config)` | Class method: create with pre-registered processes |
| `await add_process(name, config)` | Register a new process |
| `await remove_process(name)` | Remove (and stop) a process |
| `get_process(name)` | Get a process by name |
| `processes` | Read-only dict of all processes |
| `list_processes()` | Summary list of all processes |
| `await load_manifest(path)` | Load a process from a `.agent` file |
| `await load_directory(path)` | Load all `.agent` files from a directory |
| `await start_all()` | Start all stopped/created processes |
| `await stop_all()` | Stop all running processes, then drain the shared event notifier |
| `await start_process(name)` | Start a single process |
| `await stop_process(name)` | Stop a single process |
| `await restart_process(name)` | Stop then start a process |
| `status()` | Global status dict with usage |
| `process_status(name)` | Status dict for one process |

---

## Tips and Gotchas

!!! tip "Use the context manager in scripts"
    Always prefer `async with AgentRuntime() as runtime:` to ensure processes are cleanly stopped on exit, even if an exception is raised.

!!! warning "Process names must be unique"
    Calling `add_process` with a name that already exists raises `RuntimeBaseError`. Remove the existing process first or choose a different name.

!!! warning "start_all does not raise on individual failures"
    If one process fails to start, the error is logged but other processes continue starting. Check `status()` after `start_all()` to verify all processes reached the `running` state.

---

## What's Next

- [Configuration](configuration.md) -- deep dive into `RuntimeConfig`, `ProcessConfig`, and all atomic configs
- [Lifecycle](lifecycle.md) -- the process state machine and transition rules
- [CLI Commands](cli.md) -- manage runtimes from the command line
