# Behavioral Health Monitoring

Lightweight anomaly detection that identifies stuck agents, infinite loops, empty responses, and high error rates — without making any LLM calls. All detection is pure pattern matching on tool call and response history.

## When you need it

System monitoring watches CPU and memory. Nobody watches whether the agent is actually doing what it should be doing. A stuck agent calling the same tool in a loop burns compute silently until it hits context limits. Behavioral health catches that.

## Configuration

```python
from promptise.runtime import ProcessConfig, HealthConfig, EscalationTarget

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Monitor data pipelines.",
    health=HealthConfig(
        enabled=True,
        stuck_threshold=3,       # 3 identical calls = stuck
        loop_window=20,          # Check last 20 tool calls for loops
        loop_min_repeats=2,      # 2+ repeats of a pattern = loop
        empty_threshold=3,       # 3 consecutive short responses = anomaly
        empty_max_chars=10,      # Below 10 chars = trivial response
        error_rate_threshold=0.5, # 50%+ error rate = anomaly
        on_anomaly="escalate",   # "pause", "stop", "escalate", "log"
        cooldown=300,            # 5 min between same anomaly type
        escalation=EscalationTarget(
            webhook_url="https://hooks.slack.com/...",
        ),
    ),
)
```

Or in a `.agent` manifest:

```yaml
name: pipeline-monitor
health:
  enabled: true
  stuck_threshold: 3
  loop_window: 20
  on_anomaly: escalate
  cooldown: 300
  escalation:
    webhook_url: "https://hooks.slack.com/..."
```

## Detectors

### Stuck detection

Tracks `(tool_name, hash(arguments))` tuples. When the agent calls the same tool with the same arguments `stuck_threshold` times consecutively **within one invocation**, it's stuck.

Tool history starts fresh at every invocation (stuck and loop detection look for repetition inside a run). A scheduled agent that checks the same queue once per cron tick is behaving normally and is not reported as stuck.

### Loop detection

Examines the last `loop_window` tool calls for repeating patterns of any length. If a sequence like `search → read → analyze → search → read → analyze` repeats `loop_min_repeats` times, it's a loop.

### Empty response detection

Counts consecutive **final replies** (the agent's answer at the end of an invocation) shorter than `empty_max_chars`. After `empty_threshold` consecutive trivial replies, the agent likely isn't making progress. Tool results are not replies: a tool that returns `[]` because a queue is empty does not count. Reply lengths carry over between invocations.

### Recovery

After an anomaly, the next successful invocation emits a single `health.recovered` event. Further successes do not emit it again until another anomaly fires.

### Error rate detection

Sliding window error rate. When the fraction of failed tool calls exceeds `error_rate_threshold`, something is systematically wrong.

## Enforcement actions

| Action | Behavior |
|--------|----------|
| `"log"` | Log the anomaly, continue execution |
| `"pause"` | Suspend the process |
| `"stop"` | Stop the process entirely |
| `"escalate"` | Fire escalation notification, then suspend |

## API reference

### `HealthConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable health monitoring |
| `stuck_threshold` | `int` | `3` | Consecutive identical calls to trigger |
| `loop_window` | `int` | `20` | Tool calls to examine for loops |
| `loop_min_repeats` | `int` | `2` | Min pattern repeats to trigger |
| `empty_threshold` | `int` | `3` | Consecutive short responses to trigger |
| `empty_max_chars` | `int` | `10` | Max chars for a "trivial" response |
| `error_rate_threshold` | `float` | `0.5` | Error rate to trigger |
| `on_anomaly` | `str` | `"log"` | Action when anomaly detected |
| `cooldown` | `float` | `300` | Seconds between same anomaly type |
| `escalation` | `EscalationTarget \| None` | `None` | Where to notify |

### `HealthMonitor`

| Method | Description |
|--------|-------------|
| `.record_tool_call(tool_name, args)` | Record a call, return anomaly if detected |
| `.record_response(text)` | Record the agent's final reply for empty detection (not tool results) |
| `.record_success()` | Record a successful invocation; returns `True` once, on the first success after an anomaly |
| `.begin_invocation()` | Start a new invocation: clears the tool history used by stuck/loop detection (the runtime calls it) |
| `.anomalies` | List of all detected anomalies |
| `.latest_anomaly` | Most recent anomaly (or None) |
| `.clear()` | Reset all history and anomalies |

### `AnomalyType`

| Value | Description |
|-------|-------------|
| `STUCK` | Same tool + args repeated consecutively |
| `LOOP` | Repeating pattern in tool call sequence |
| `EMPTY_RESPONSE` | Consecutive trivial responses |
| `HIGH_ERROR_RATE` | Error rate above threshold |
