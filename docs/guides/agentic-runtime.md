---
title: Build an Autonomous Agent Runtime — Python tutorial with Promptise Foundry
description: Build an autonomous agent system in Python with Promptise Foundry's Agent Runtime. Covers process lifecycle, cron and event triggers, journaled state, crash recovery, governance (budget, health, mission, secrets), and multi-node distribution.
keywords: autonomous agent tutorial, agent runtime guide, build autonomous AI agent, cron AI agent, AI agent crash recovery, agent governance Python
---

# Building Agentic Runtime Systems

## What You'll Build

A production-grade autonomous agent system that monitors data pipelines, reacts to events, persists state across invocations, recovers from crashes, enforces governance policies, and scales across multiple machines. You'll start with a single agent and progressively add capabilities until you have a governed, mission-driven, distributed multi-agent operations center.

## Concepts

An **agentic runtime** is the infrastructure that turns a stateless LLM into a persistent, autonomous process. Without a runtime, an agent exists only for the duration of a single request -- you call it, get a response, and it disappears. A runtime wraps that same agent in a long-running process with scheduling, event handling, persistent memory, crash recovery, and governance.

Think of the relationship like a web application versus a web framework: you could handle HTTP requests manually, but a framework gives you routing, middleware, sessions, and error handling. The Agent Runtime does the same for AI agents -- it provides the infrastructure so you can focus on what the agent should do, not how to keep it running.

The runtime is built in layers. Each layer adds capabilities without requiring changes to the layer below:

```
Layer 5: Governance                  Budget, health, mission, secrets
         |
Layer 4: Distributed Coordination    Multi-node clusters with transport & discovery
         |
Layer 3: AgentRuntime                Multi-process manager with lifecycle control
         |
Layer 2: AgentProcess                Single autonomous agent with triggers & state
         |
Layer 1: build_agent()          Stateless LLM agent with MCP tool access
```

You can stop at any layer. Most projects need Layer 2 (a single autonomous agent). Add Layer 3 when you need multiple agents working together. Layer 4 is for production deployments across multiple machines. Layer 5 is for autonomous agents that need guardrails.

---

## Step 1: Single Autonomous Agent

Start with an `AgentProcess` wrapping a stateless agent in a persistent process.

```python
from promptise.runtime import AgentProcess, ProcessConfig, TriggerConfig

process = AgentProcess(
    name="pipeline-monitor",
    config=ProcessConfig(
        model="openai:gpt-5-mini",
        instructions=(
            "You monitor data pipelines. When triggered, check pipeline "
            "health and report any anomalies."
        ),
        servers={
            "pipeline_api": {
                "url": "http://localhost:8080/mcp",
                "transport": "streamable-http",
            },
        },
        triggers=[
            TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
        ],
    ),
)

async with process:
    # Process runs autonomously, waking on every cron tick
    await asyncio.sleep(3600)  # Run for 1 hour
```

The process handles the full lifecycle: build the agent, connect MCP servers, discover tools, start triggers, and invoke the agent on each trigger event.

---

## Step 2: Add Triggers

Triggers wake the agent in response to events. Compose multiple triggers to create responsive agents.

```python
config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="You respond to pipeline events and scheduled checks.",
    triggers=[
        # Wake every 5 minutes for health checks
        TriggerConfig(type="cron", cron_expression="*/5 * * * *"),

        # Wake on incoming webhooks (e.g., from CI/CD)
        TriggerConfig(
            type="webhook",
            webhook_path="/pipeline-events",
            webhook_port=9090,
        ),

        # Wake when config files change
        TriggerConfig(
            type="file_watch",
            watch_path="/etc/pipeline/config",
            watch_patterns=["*.yaml", "*.json"],
            watch_events=["modified"],
        ),

        # Wake on events from other agents
        TriggerConfig(
            type="event",
            event_type="alert.critical",
        ),

        # Wake on inter-agent messages
        TriggerConfig(
            type="message",
            topic="pipeline.alerts.*",
        ),
    ],
)
```

Each trigger type fires a `TriggerEvent` with type-specific payload data that the agent receives as context.

| Trigger | Wakes On | Payload |
|---------|----------|---------|
| `cron` | Schedule expression | `scheduled_time`, `cron_expression` |
| `webhook` | HTTP POST request | Request body (JSON or text) |
| `file_watch` | File system changes | `path`, `filename`, `event_type` |
| `event` | EventBus events | `event_type`, `source`, `data` |
| `message` | MessageBroker messages | `topic`, `sender`, `content` |

---

## Step 3: Persistent State and Context

`AgentContext` gives each process a persistent key-value state with audit trail, long-term memory, environment variables, and file mounts.

```python
from promptise.runtime.config import ContextConfig

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="You track pipeline health over time.",
    context=ContextConfig(
        # Pre-populate state
        initial_state={
            "alert_count": 0,
            "last_healthy": None,
            "known_issues": [],
        },

        # Restrict which keys the agent can modify
        writable_keys=["alert_count", "last_healthy", "known_issues"],

        # Long-term memory (persists across restarts)
        memory_provider="chroma",
        memory_collection="pipeline_memory",
        memory_persist_directory=".promptise/memory",
        memory_auto_store=True,

        # Short-term conversation buffer
        conversation_max_messages=50,

        # Expose environment variables matching prefix
        env_prefix="PIPELINE_",

        # Mount files the agent can reference
        file_mounts={
            "config": "/etc/pipeline/config.yaml",
            "runbook": "/docs/runbook.md",
        },
    ),
    triggers=[
        TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
    ],
)
```

The context state is injected into every agent invocation as a system message. The agent can read and write state keys during execution.

**State access in code:**

```python
process = AgentProcess(name="monitor", config=config)
await process.start()

# Read state programmatically
ctx = process.context
alert_count = ctx.get("alert_count", 0)
snapshot = ctx.state_snapshot()

# Write state with audit trail
ctx.put("alert_count", alert_count + 1, source="external")

# View audit history for a key
history = ctx.state_history("alert_count")
for entry in history:
    print(f"  {entry.timestamp}: {entry.value} (by {entry.source})")
```

---

## Step 4: Crash Recovery with Journals

The journal system records every state transition, trigger event, and invocation. When a process crashes, the `ReplayEngine` reconstructs the last known state.

```python
from promptise.runtime.config import JournalConfig

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Critical monitoring agent.",
    journal=JournalConfig(
        level="full",                          # "none", "checkpoint", or "full"
        backend="file",                        # "file" or "memory"
        path=".promptise/journal",             # Persist to disk
    ),
    max_consecutive_failures=3,              # FAILED state after 3 crashes
    restart_policy="on_failure",             # Auto-restart on failure
    max_restarts=5,                          # Max restart attempts
    triggers=[
        TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
    ],
)
```

**Recovery flow:**

```python
from promptise.runtime.journal import FileJournal, ReplayEngine

# After a crash, recover state from journal
journal = FileJournal(base_path=".promptise/journal")
engine = ReplayEngine(journal)

recovered = await engine.recover(process_id="proc-123")
print(f"Recovered state: {recovered['context_state']}")
print(f"Last lifecycle state: {recovered['lifecycle_state']}")
print(f"Entries replayed: {recovered['entries_replayed']}")
```

With `restart_policy="on_failure"`, the runtime automatically attempts to restart failed processes up to `max_restarts` times.

---

## Step 5: Governance -- Autonomy Budget

Without a budget, an autonomous agent can loop indefinitely, call expensive APIs without limit, or take irreversible actions at 3am. The autonomy budget defines the envelope within which the agent operates freely.

```python
from promptise.runtime import ProcessConfig, BudgetConfig, EscalationTarget, ToolCostAnnotation

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Process customer support tickets.",
    budget=BudgetConfig(
        enabled=True,

        # Per-invocation limits
        max_tool_calls_per_run=20,
        max_llm_turns_per_run=10,
        max_cost_per_run=25.0,
        max_irreversible_per_run=2,

        # Daily limits (reset at midnight UTC)
        max_tool_calls_per_day=500,
        max_cost_per_day=100.0,

        # Per-tool cost annotations
        tool_costs={
            "stripe_charge": ToolCostAnnotation(cost_weight=5.0, irreversible=True),
            "send_email": ToolCostAnnotation(cost_weight=2.0, irreversible=True),
            "search": ToolCostAnnotation(cost_weight=0.5),
        },

        # What happens when a limit is hit
        on_exceeded="escalate",   # "pause", "stop", or "escalate"
        inject_remaining=True,     # Show budget in agent context

        escalation=EscalationTarget(
            webhook_url="https://hooks.slack.com/services/...",
        ),
    ),
    triggers=[
        TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
    ],
)
```

When `inject_remaining=True`, the agent sees its remaining budget before every turn -- so it can prioritize actions and avoid hitting limits.

!!! warning "cost_weight is abstract, not dollars"
    `cost_weight` values are **budget units you define**, not real money. The framework does not track LLM API costs (token pricing). `max_cost_per_day=100.0` with `cost_weight=5.0` on `stripe_charge` means "the agent can make 20 Stripe charges per day" — not "$100/day." Use `max_llm_turns_per_run` to limit LLM usage. See [Autonomy Budget](../runtime/governance/budget.md) for full details on designing your cost scale.

---

## Step 6: Governance -- Behavioral Health

System monitoring watches CPU and memory. Nobody watches whether the agent is actually doing what it should be doing. Behavioral health catches stuck agents, infinite loops, empty responses, and high error rates -- without making any LLM calls.

```python
from promptise.runtime import ProcessConfig, HealthConfig, EscalationTarget

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Monitor data pipelines.",
    health=HealthConfig(
        enabled=True,
        stuck_threshold=3,         # 3 identical calls = stuck
        loop_window=20,            # Check last 20 tool calls for loops
        loop_min_repeats=2,        # 2+ repeats of a pattern = loop
        empty_threshold=3,         # 3 consecutive short responses = anomaly
        empty_max_chars=10,        # Below 10 chars = trivial response
        error_rate_threshold=0.5,  # 50%+ error rate = anomaly
        on_anomaly="pause",        # "log", "pause", or "escalate"
        cooldown=300,              # 5 min between same anomaly type
        escalation=EscalationTarget(
            webhook_url="https://hooks.slack.com/services/...",
        ),
    ),
    triggers=[
        TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
    ],
)
```

Four anomaly detectors run automatically:

| Anomaly | Detection | Example |
|---------|-----------|---------|
| **Stuck** | Last N calls identical (same tool, same args) | Agent calling `get_status("pipeline-3")` 5 times in a row |
| **Loop** | Repeating subsequence in tool call history | Agent cycling through check → fix → check → fix endlessly |
| **Empty response** | N consecutive responses below char threshold | Agent returning `""` or `"ok"` repeatedly |
| **High error rate** | Error rate above threshold in sliding window | 6 out of last 10 tool calls failing |

---

## Step 7: Governance -- Mission-Oriented Execution

Standard agents run on a trigger, do something, stop. A mission-oriented agent runs until a goal is achieved -- accumulating context across invocations, with LLM-as-judge evaluation and automatic completion.

```python
from promptise.runtime import ProcessConfig, MissionConfig, EscalationTarget

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Migrate all database tables to v2 schema.",
    mission=MissionConfig(
        enabled=True,
        objective="Migrate all database tables to v2 schema",
        success_criteria="All tables pass v2 schema validation with zero errors",
        eval_every=3,               # Evaluate every 3 invocations
        confidence_threshold=0.7,   # Escalate if confidence drops below this
        timeout_hours=24,           # Fail after 24 hours
        max_invocations=50,         # Fail after 50 invocations
        auto_complete=True,         # Stop when mission succeeds
        eval_model="openai:gpt-5-mini",  # Separate model for evaluation
        on_complete="stop",         # "stop", "continue", or "suspend"
        escalation=EscalationTarget(
            webhook_url="https://hooks.slack.com/services/...",
        ),
    ),
    triggers=[
        TriggerConfig(type="cron", cron_expression="*/10 * * * *"),
    ],
)
```

The mission tracker:

1. Prepends the objective and progress into every agent invocation context
2. Every N invocations, runs an LLM-as-judge evaluation against the success criteria
3. If `achieved=True` and `auto_complete=True`, stops the process
4. If `confidence < threshold`, pauses the process and fires escalation
5. If timed out or over invocation limit, fails the mission

---

## Step 8: Governance -- Scoped Secrets

Environment variables are shared across all agents on the same host. Secret scoping gives each process its own isolated credential context with automatic expiry and access logging.

```python
from promptise.runtime import ProcessConfig, SecretScopeConfig

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Process payments.",
    secrets=SecretScopeConfig(
        enabled=True,
        secrets={
            "stripe_key": "${STRIPE_API_KEY}",    # Resolved from env at startup
            "db_password": "${DB_PASSWORD}",
            "static_token": "tok-abc123",          # Literal value
        },
        default_ttl=3600,           # 1 hour default
        ttls={
            "stripe_key": 1800,     # 30 min for payment credentials
        },
        revoke_on_stop=True,        # Zero-fill on process stop
    ),
    triggers=[
        TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
    ],
)
```

Key security properties:

- **Values live only in memory** -- never serialized to journal, checkpoint, or status output
- **TTL-based expiry** -- secrets become inaccessible after their TTL expires
- **Access logging** -- every `get_secret()` call is logged in the journal (name only, never the value)
- **Zero-fill revocation** -- on process stop, all values are overwritten with zeros and removed
- **Crash recovery** -- secrets are re-resolved from environment variables on restart, never from journal

---

## Step 9: Multi-Agent Runtime

`AgentRuntime` manages multiple `AgentProcess` instances with centralized lifecycle control.

```python
from promptise.runtime import AgentRuntime, ProcessConfig, TriggerConfig

# Shared communication channels — any objects implementing
# subscribe/emit (event bus) and subscribe/publish (broker) interfaces
event_bus = MyEventBus()
broker = MyMessageBroker()

async with AgentRuntime(event_bus=event_bus, broker=broker) as runtime:
    # Register agents
    await runtime.add_process("monitor", ProcessConfig(
        model="openai:gpt-5-mini",
        instructions="Monitor pipelines. Emit 'alert.critical' events on failures.",
        triggers=[
            TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
        ],
    ))

    await runtime.add_process("analyst", ProcessConfig(
        model="openai:gpt-5-mini",
        instructions="Analyze alerts and determine root cause.",
        triggers=[
            TriggerConfig(type="event", event_type="alert.critical"),
        ],
    ))

    await runtime.add_process("responder", ProcessConfig(
        model="openai:gpt-5-mini",
        instructions="Execute remediation based on analysis.",
        triggers=[
            TriggerConfig(type="message", topic="analysis.complete"),
        ],
    ))

    # Start all agents
    await runtime.start_all()

    # Monitor status
    status = runtime.status()
    for name, proc_status in status["processes"].items():
        print(f"  {name}: {proc_status['state']}")

    # Run until interrupted
    await asyncio.sleep(3600)
```

**Agent-to-agent communication** happens through two channels:

- **EventBus**: Broadcast events that any listening agent can receive. Use `TriggerConfig(type="event", event_type="...")` to subscribe.
- **MessageBroker**: Point-to-point or topic-based messaging with wildcard support. Use `TriggerConfig(type="message", topic="...")` to subscribe.

**Loading from manifests:**

```python
# Load all .agent files from a directory
loaded = await runtime.load_directory("agents/")
print(f"Loaded: {loaded}")  # ["monitor", "analyst", "responder"]

await runtime.start_all()
```

---

## Step 10: Distributed Coordination

For multi-node deployments, `RuntimeTransport` exposes each node's runtime over HTTP, and `RuntimeCoordinator` manages cluster membership.

```python
import os

from promptise.runtime import AgentRuntime
from promptise.runtime.distributed import RuntimeTransport, RuntimeCoordinator

TOKEN = os.environ["PROMPTISE_NODE_TOKEN"]  # required for a non-loopback bind

# Node 1 -- primary
runtime_1 = AgentRuntime()
transport_1 = RuntimeTransport(
    runtime_1,
    host="0.0.0.0",
    port=9100,
    node_id="node-1",
    auth_token=TOKEN,
)
await transport_1.start()

# Node 2 -- secondary
runtime_2 = AgentRuntime()
transport_2 = RuntimeTransport(
    runtime_2,
    host="0.0.0.0",
    port=9101,
    node_id="node-2",
    auth_token=TOKEN,
)
await transport_2.start()

# Coordinator tracks all nodes
coordinator = RuntimeCoordinator(
    health_check_interval=15.0,
    node_timeout=45.0,
    auth_token=TOKEN,  # sent as a bearer token to every node
)
coordinator.register_node("node-1", "http://node-1:9100")
coordinator.register_node("node-2", "http://node-2:9101")
```

Each node exposes a REST API for remote management. Every endpoint except `/health` requires `Authorization: Bearer <token>`; binding a non-loopback address without `auth_token` raises `ValueError` (see [Discovery and Transport](../runtime/distributed/discovery-transport.md#security)).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with process count |
| `/status` | GET | Full runtime status |
| `/processes` | GET | List all processes |
| `/processes/{name}/start` | POST | Start a process |
| `/processes/{name}/stop` | POST | Stop a process |
| `/processes/{name}/event` | POST | Inject a trigger event |

---

## Step 11: Production Hardening

Combine governance, failure handling, and resource limits for a production deployment:

```python
config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Production agent.",

    # Concurrency control
    concurrency=3,                    # Up to 3 parallel invocations

    # Health monitoring
    heartbeat_interval=10.0,          # Heartbeat every 10s
    idle_timeout=300.0,               # Suspend after 5min idle
    max_lifetime=86400.0,             # Restart after 24h

    # Failure handling
    max_consecutive_failures=3,       # FAILED after 3 consecutive errors
    restart_policy="on_failure",      # Auto-restart on failure
    max_restarts=5,                   # Max 5 restart attempts

    # Governance
    budget=BudgetConfig(
        enabled=True,
        max_tool_calls_per_day=1000,
        max_cost_per_day=200.0,
        on_exceeded="pause",
    ),
    health=HealthConfig(
        enabled=True,
        on_anomaly="escalate",
        escalation=EscalationTarget(webhook_url="https://hooks.slack.com/..."),
    ),
    mission=MissionConfig(
        enabled=True,
        objective="Keep all pipelines healthy",
        success_criteria="Zero critical alerts for 24 consecutive hours",
        eval_every=10,
        timeout_hours=48,
    ),
    secrets=SecretScopeConfig(
        enabled=True,
        secrets={"api_key": "${PIPELINE_API_KEY}"},
        default_ttl=3600,
        revoke_on_stop=True,
    ),
)
```

**Monitoring:**

```python
# Global runtime status
status = runtime.status()
print(f"Processes: {status['process_count']}")

# Per-process status (includes governance info)
for info in runtime.list_processes():
    print(f"  {info['name']}: {info['state']}")

# Detailed process status
detail = runtime.process_status("monitor")
print(f"  Invocations: {detail['invocation_count']}")
print(f"  Failures: {detail['consecutive_failures']}")
print(f"  Uptime: {detail['uptime_seconds']:.0f}s")
```

---

## Complete Example

A three-agent system with full governance: a monitor watches pipelines with a budget and health checks, an analyst diagnoses issues, and a responder executes fixes with a mission objective.

```python
import asyncio
from promptise.runtime import (
    AgentRuntime, ProcessConfig, TriggerConfig,
    BudgetConfig, HealthConfig, MissionConfig, SecretScopeConfig,
    EscalationTarget, ToolCostAnnotation,
)
from promptise.runtime.config import ContextConfig, JournalConfig

async def main():
    # Plug in your own event bus and message broker implementations
    event_bus = MyEventBus()
    broker = MyMessageBroker()

    slack = EscalationTarget(webhook_url="https://hooks.slack.com/services/...")

    async with AgentRuntime(event_bus=event_bus, broker=broker) as runtime:
        await runtime.add_process("monitor", ProcessConfig(
            model="openai:gpt-5-mini",
            instructions=(
                "You monitor data pipelines. Check health every 5 minutes. "
                "If a pipeline is unhealthy, emit an 'alert.critical' event."
            ),
            servers={
                "pipeline": {"url": "http://localhost:8080/mcp", "transport": "streamable-http"},
            },
            triggers=[TriggerConfig(type="cron", cron_expression="*/5 * * * *")],
            context=ContextConfig(
                initial_state={"total_checks": 0, "alerts_raised": 0},
                memory_provider="chroma",
                memory_auto_store=True,
            ),
            journal=JournalConfig(level="full", backend="file"),
            budget=BudgetConfig(
                enabled=True,
                max_tool_calls_per_day=500,
                max_cost_per_day=50.0,
                on_exceeded="pause",
            ),
            health=HealthConfig(
                enabled=True,
                stuck_threshold=5,
                on_anomaly="escalate",
                escalation=slack,
            ),
        ))

        await runtime.add_process("analyst", ProcessConfig(
            model="openai:gpt-5-mini",
            instructions="Analyze pipeline alerts. Determine root cause.",
            triggers=[TriggerConfig(type="event", event_type="alert.critical")],
        ))

        await runtime.add_process("responder", ProcessConfig(
            model="openai:gpt-5-mini",
            instructions="Execute pipeline remediation.",
            servers={
                "pipeline": {"url": "http://localhost:8080/mcp", "transport": "streamable-http"},
            },
            triggers=[TriggerConfig(type="message", topic="analysis.complete")],
            restart_policy="on_failure",
            mission=MissionConfig(
                enabled=True,
                objective="Restore all pipelines to healthy status",
                success_criteria="All pipeline stages pass health checks for 3 consecutive runs",
                eval_every=3,
                confidence_threshold=0.6,
                timeout_hours=4,
                auto_complete=True,
                escalation=slack,
            ),
            secrets=SecretScopeConfig(
                enabled=True,
                secrets={"pipeline_admin_key": "${PIPELINE_ADMIN_KEY}"},
                default_ttl=1800,
                revoke_on_stop=True,
            ),
        ))

        await runtime.start_all()

        while True:
            status = runtime.status()
            running = sum(
                1 for p in status["processes"].values()
                if p["state"] in ("running", "awaiting")
            )
            print(f"[Runtime] {running}/{status['process_count']} agents active")
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## What's Next

**Reference documentation:**

- [Agent Processes](../runtime/processes.md) -- Lifecycle methods, triggers, and `ProcessConfig`
- [Runtime Manager](../runtime/runtime-manager.md) -- Multi-process runtimes with `AgentRuntime`
- [Context & State](../runtime/context.md) -- Persistent state, memory, and environment
- [Triggers](../runtime/triggers/index.md) -- All five trigger types

**Governance:**

- [Autonomy Budget](../runtime/governance/budget.md) -- Per-run and daily limits, tool cost annotations
- [Behavioral Health](../runtime/governance/health.md) -- Stuck, loop, empty, and error rate detection
- [Mission Model](../runtime/governance/mission.md) -- LLM-as-judge evaluation, confidence thresholds
- [Secret Scoping](../runtime/governance/secrets.md) -- Per-process credentials with TTL and rotation

**Infrastructure:**

- [Journal & Recovery](../runtime/journal/index.md) -- Crash recovery with journals and replay
- [Distributed](../runtime/distributed/coordinator.md) -- Multi-node coordination
- [Agent Manifests](../runtime/manifests.md) -- `.agent` YAML files for declarative deployment

**Other guides:**

- [Building Production MCP Servers](production-mcp-servers.md) -- Build the tool servers your agents connect to
- [Building AI Agents](building-agents.md) -- The core agent that powers every process
- [Prompt Engineering](prompt-engineering.md) -- Build reliable, testable system prompts
