# Events & Notifications

Emit structured notifications when significant things happen during agent execution. Route events to webhooks, callbacks, logs, or the runtime's event bus.

```python
from promptise import build_agent, EventNotifier, WebhookSink, CallbackSink

notifier = EventNotifier(sinks=[
    WebhookSink(
        url="https://hooks.slack.com/services/...",
        events=["invocation.error", "budget.exceeded", "guardrail.blocked"],
    ),
    CallbackSink(lambda event: print(f"[{event.severity}] {event.event_type}")),
])

agent = await build_agent(
    servers=servers, model="openai:gpt-5-mini", events=notifier,
)
```

---

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## Event Taxonomy

21 event types across 9 categories:

| Category | Event Type | Severity | When it fires |
|----------|-----------|----------|---------------|
| **Invocation** | `invocation.start` | info | Agent begins processing |
| | `invocation.complete` | info | Agent finishes successfully |
| | `invocation.error` | error | Unhandled exception during invocation |
| | `invocation.timeout` | error | Invocation exceeded `max_invocation_time` |
| **Tools** | `tool.error` | error | A tool call fails |
| | `tool.slow` | warning | Tool call exceeds latency threshold (default 5s) |
| | `tool.progress` | info | An MCP tool reports progress (`data`: `tool_name`, `progress`, `total`, `message`) |
| **Guardrails** | `guardrail.blocked` | warning | Input blocked by guardrails |
| | `guardrail.redacted` | info | Output had PII/credentials redacted |
| **Budget** | `budget.exceeded` | critical | Budget limit reached |
| **Approval** | `approval.requested` | info | Human approval requested |
| | `approval.granted` | info | Approval granted |
| | `approval.denied` | warning | Approval denied or timed out |
| **Mission** | `mission.progress` | info | Mission evaluation completed |
| | `mission.complete` | info | Mission objective achieved |
| | `mission.failed` | critical | Mission timed out or exceeded limits |
| **Health** | `health.anomaly` | warning | Behavioral anomaly detected |
| **Process** | `process.started` | info | Agent process started |
| | `process.stopped` | info | Agent process stopped |
| | `process.failed` | critical | Agent process entered FAILED state (failed start, or `max_consecutive_failures` reached) |
| | `process.restarted` | warning | Restart policy is restarting the process (`attempt`, `max_restarts`, `reason`) |
| **Cache** | `cache.purged` | info | User cache purged (GDPR) |

---

## Sinks

### WebhookSink

HTTP POST to any URL. HMAC-SHA256 signed. Retry with exponential backoff. SSRF protection.

```python
WebhookSink(
    url="https://hooks.slack.com/services/...",
    events=["invocation.error", "budget.exceeded"],  # Only these events
    secret="my-hmac-secret",                          # HMAC signing
    headers={"Authorization": "Bearer tok-123"},      # Custom headers
    max_retries=3,                                    # Retry on failure
    min_severity="warning",                           # Skip info events
    redact_sensitive=True,                            # Scan payloads for PII
    transform=lambda p: {"text": f"[{p['severity']}] {p['event_type']}"},  # Custom format
)
```

Each request includes:
- `X-Promptise-Signature`: HMAC-SHA256 of the JSON payload
- `X-Promptise-Event`: The event type (e.g. `invocation.error`)

### CallbackSink

Python callable (sync or async). Full control.

```python
# Simple
CallbackSink(lambda event: print(event.event_type))

# Async with filtering
async def handle_errors(event):
    await alert_team(event.data)

CallbackSink(handle_errors, events=["invocation.error", "process.failed"])
```

### LogSink

Structured logging via Python's `logging` module.

```python
LogSink(events=["invocation.complete", "tool.error"], min_severity="warning")
```

Events appear as structured log lines compatible with ELK, Datadog, Splunk.

### EventBusSink

Bridge to the Agent Runtime's event bus for inter-process notifications.

```python
EventBusSink(event_bus, events=["health.anomaly", "mission.complete"])
```

---

## EventNotifier

The central coordinator. Routes events to configured sinks via an async queue.

```python
notifier = EventNotifier(
    sinks=[sink_a, sink_b, sink_c],
    max_queue_size=1000,  # Drop events when queue is full (never block)
)
```

**Fire-and-forget**: `emit()` puts the event on a queue and returns immediately. A background task delivers to sinks. The agent never blocks waiting for event delivery.

**Sink isolation**: If one sink fails (webhook returns 500), other sinks still receive the event. Failures are logged, never propagated.

**Graceful shutdown**: `agent.shutdown()` automatically drains remaining events before stopping.

---

## Filtering

Each sink can filter by event type and/or minimum severity:

```python
# Only errors and critical events
WebhookSink(url="...", min_severity="error")

# Only specific event types
CallbackSink(handler, events=["budget.exceeded", "process.failed"])

# Combined: only critical process events
WebhookSink(url="...", events=["process.failed"], min_severity="critical")
```

---

## AgentEvent Structure

Every event is an `AgentEvent` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `event_type` | `str` | Dotted event name (e.g. `invocation.complete`) |
| `severity` | `str` | `info`, `warning`, `error`, or `critical` |
| `timestamp` | `float` | When the event occurred (`time.time()`) |
| `agent_id` | `str \| None` | Agent or process identifier |
| `user_id` | `str \| None` | From CallerContext (multi-user) |
| `session_id` | `str \| None` | Conversation session ID |
| `data` | `dict` | Event-specific payload |
| `metadata` | `dict` | Agent config, model ID, etc. |

---

## Security

- **SSRF protection**: WebhookSink validates URLs at construction — rejects private IPs, loopback, cloud metadata endpoints
- **HMAC signing**: Every webhook includes `X-Promptise-Signature` for tamper verification
- **Payload redaction**: WebhookSink scans payloads for PII/credentials before sending (uses guardrails regex patterns, no ML models)
- **Queue bounds**: `max_queue_size` prevents memory exhaustion. When full, events are dropped with a warning log
- **Sink isolation**: One failing sink never affects others

---

## SuperAgent YAML

```yaml
events:
  sinks:
    - type: webhook
      url: https://hooks.slack.com/services/...
      events: [invocation.error, budget.exceeded]
      min_severity: warning
    - type: log
      events: [invocation.complete]
```

---

## EventNotifier Lifecycle

The notifier must be started before events can be delivered, and stopped to drain remaining events:

```python
notifier = EventNotifier(sinks=[...])
await notifier.start()   # Start background drain task

# ... agent runs, events are emitted ...

await notifier.stop()    # Drain remaining events, stop background task
```

When passed to `build_agent(events=notifier)`, `start()` is called automatically. `shutdown()` calls `stop()` automatically.

| Method | Description |
|---|---|
| `await start()` | Start the background event delivery task. Auto-called by `build_agent()`. |
| `await stop()` | Drain remaining events and stop. Called by `agent.shutdown()`. |
| `await emit(event)` | Queue an event for delivery (non-blocking). Auto-starts if not started. |
| `emit_sync(event)` | Queue from synchronous code (used by LangChain callbacks). Silent on queue full. |

---

## Emitting Custom Events

Use `emit_event()` to emit events from your own code:

```python
from promptise.events import emit_event

# Inside an async function where you have access to the notifier:
emit_event(
    notifier,
    event_type="custom.my_event",
    severity="info",
    data={"key": "value"},
    agent_id="my-agent",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `notifier` | `EventNotifier \| None` | required | The notifier (None = no-op) |
| `event_type` | `str` | required | Dotted event name |
| `severity` | `str` | `"info"` | `info`, `warning`, `error`, `critical` |
| `data` | `dict \| None` | `None` | Event payload |
| `agent_id` | `str \| None` | `None` | Agent identifier |
| `session_id` | `str \| None` | `None` | Session ID |
| `metadata` | `dict \| None` | `None` | Additional metadata |

`emit_event()` is null-safe — passing `None` as the notifier does nothing. It automatically reads `user_id` from the current `CallerContext` if available.

---

## Sink Parameter Reference

### WebhookSink

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | required | Webhook URL (SSRF-protected) |
| `events` | `list[str] \| None` | `None` | Event types to subscribe to (None = all) |
| `headers` | `dict[str, str]` | `{}` | Custom HTTP headers |
| `secret` | `str \| None` | auto-generated | HMAC signing secret |
| `max_retries` | `int` | `3` | Retry attempts on failure |
| `retry_delay` | `float` | `1.0` | Initial retry delay (doubles each retry) |
| `redact_sensitive` | `bool` | `True` | Scan payloads for PII/credentials |
| `min_severity` | `str \| None` | `None` | Minimum severity to emit |
| `transform` | `Callable \| None` | `None` | Custom payload transformation |

### CallbackSink

| Parameter | Type | Default | Description |
|---|---|---|---|
| `callback` | `Callable` | required | Async or sync callable |
| `events` | `list[str] \| None` | `None` | Event filter |
| `min_severity` | `str \| None` | `None` | Minimum severity |

### LogSink

| Parameter | Type | Default | Description |
|---|---|---|---|
| `events` | `list[str] \| None` | `None` | Event filter |
| `logger_name` | `str` | `"promptise.events"` | Python logger name |
| `min_severity` | `str \| None` | `None` | Minimum severity |

### EventBusSink

| Parameter | Type | Default | Description |
|---|---|---|---|
| `event_bus` | `Any` | required | Object with `emit(event_type, data)` method |
| `events` | `list[str] \| None` | `None` | Event filter |

---

## What's Next?

- [Observability](observability.md) -- detailed execution traces (events are high-level alerts, observability is full traces)
- [Approval](approval.md) -- human-in-the-loop approval that emits `approval.*` events
- [Guardrails](guardrails.md) -- security scanning that emits `guardrail.*` events
