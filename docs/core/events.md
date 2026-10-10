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

23 event types across 9 categories. Events marked *runtime* come from an [`AgentProcess`](../runtime/index.md); the rest come from any agent built with `events=`.

| Category | Event Type | Severity | When it fires |
|----------|-----------|----------|---------------|
| **Invocation** | `invocation.start` | info | Agent begins processing |
| | `invocation.complete` | info | Agent finishes successfully |
| | `invocation.error` | error | Unhandled exception during invocation |
| | `invocation.timeout` | error | Invocation exceeded `max_invocation_time` |
| **Tools** | `tool.error` | error | A tool call raised, or an MCP tool returned an error result |
| | `tool.slow` | warning | A tool call took longer than `slow_tool_threshold` (default 5s) |
| **Guardrails** | `guardrail.blocked` | warning | Input (or streamed output) blocked by guardrails |
| | `guardrail.redacted` | info | Output had PII/credentials redacted |
| **Approval** | `approval.requested` | info | Human approval requested |
| | `approval.granted` | info | Approval granted |
| | `approval.denied` | warning | Approval denied or timed out |
| **Budget** *(runtime)* | `budget.warning` | warning | A budget limit is close to being reached |
| | `budget.exceeded` | critical | Budget limit reached |
| | `budget.daily_reset` | info | Daily budget counters were reset |
| **Mission** *(runtime)* | `mission.progress` | info | Mission evaluation completed |
| | `mission.complete` | info | Mission objective achieved |
| | `mission.failed` | critical | Mission timed out or exceeded limits |
| **Health** *(runtime)* | `health.anomaly` | warning | Behavioral anomaly detected |
| | `health.recovered` | info | The process is healthy again after an anomaly |
| **Process** *(runtime)* | `process.started` | info | Agent process started |
| | `process.stopped` | info | Agent process stopped |
| | `process.failed` | critical | Agent process entered FAILED state (including at startup) |
| **Cache** | `cache.purged` | info | User cache purged (GDPR) |

`tool.error` and `tool.slow` are emitted whenever `events=` is set — they don't need `observe=True`.

### What counts as a tool error

`tool.error` fires when a tool call:

- **raises** — any tool: MCP tools, `extra_tools`, sandbox and cross-agent tools;
- **returns an MCP error result** — a result with `isError: true`, or the structured error a Promptise MCP server sends when a tool raises `ToolError` (`{"error": {"code": ..., "message": ..., "retryable": ...}}`). The model still sees the error text and can recover; the event tells *you* it happened.

A tool that returns its own domain answer such as `{"error": "No invoice INV-404"}` (no `code`) is not a tool error.

### Payloads

The `data` of each event:

| Event Type | `data` fields |
|---|---|
| `invocation.start` | `model`; `streaming` when streamed |
| `invocation.complete` | `duration_ms`; `streaming` |
| `invocation.error` | `error`, `error_type`; `streaming` |
| `invocation.timeout` | `timeout_seconds` |
| `tool.error` | `tool_name`, `error` (message), `error_type`, `duration_ms`; `code` and `retryable` when the tool reported them |
| `tool.slow` | `tool_name`, `latency_ms`, `threshold_ms` |
| `guardrail.blocked` | `direction` (`input`/`output`), `error` (exception type), `reason`, `findings` (`detector`, `category`, `severity`, `description` per finding — never the matched text); `streaming` |
| `guardrail.redacted` | `direction`; `streaming` |
| `approval.requested` | `tool_name`, `request_id`, `timeout`, `arguments` (as the policy redacts them; `{}` with `include_arguments=False`) |
| `approval.granted` | `tool_name`, `request_id`, `reviewer` |
| `approval.denied` | `tool_name`, `request_id`, `reason` |
| `budget.warning` | `process_name`, `limit_type`, `current`, `limit`, `percentage` |
| `budget.exceeded` | `process_name`, `limit_type`, `current`, `limit` |
| `budget.daily_reset` | `process_name` |
| `mission.progress` | `process_name`, `confidence`, `achieved`, `invocations` |
| `mission.complete` | `process_name`, `confidence`, `invocations` |
| `mission.failed` | `process_name`, `reason` |
| `health.anomaly` | `process_name`, `anomaly_type`, `details` |
| `health.recovered` | `process_name` |
| `process.started` / `process.stopped` | `process_name`, `process_id` |
| `process.failed` | `process_name`, `error` |
| `cache.purged` | `user_id`, `entries_removed` |

---

## Sinks

### WebhookSink

HTTP POST to any URL. Signed with HMAC-SHA256 and a timestamp. Retry with exponential backoff. SSRF protection.

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

- `X-Promptise-Signature`: `t=<unix seconds>,v1=<hex>` — HMAC-SHA256 over `<t>.` followed by the exact body bytes
- `X-Promptise-Timestamp`: the same `t`
- `X-Promptise-Event`: the event type (e.g. `invocation.error`)
- `X-Promptise-Delivery`: an id that stays the same across retries of one event, for de-duplication

#### Verifying a delivery

Verify against the **raw** request body, before parsing it. `verify_event_signature()` checks the HMAC in constant time and rejects signatures older than five minutes (replays):

```python
from promptise import verify_event_signature

# In your web framework's handler (Flask shown):
raw = request.get_data()                      # bytes, exactly as received
if not verify_event_signature(raw, request.headers.get("X-Promptise-Signature"), SECRET):
    abort(401)
event = json.loads(raw)
```

| Parameter | Default | Description |
|---|---|---|
| `body` | required | Raw body (`bytes`, or `str` encoded as UTF-8) |
| `signature` | required | The `X-Promptise-Signature` header |
| `secret` | required | The sink's secret, or a list of secrets while rotating |
| `tolerance` | `300.0` | Maximum age in seconds; `None` or `0` turns the replay check off |

Without Promptise on the receiving side, compute `hex(HMAC_SHA256(secret, f"{t}." + raw_body))` and compare it to `v1` with a constant-time comparison, then check that `t` is recent. If you don't pass a `secret`, the sink generates one; read it from `sink.secret`.

!!! note "Changed in the next release"
    Earlier versions sent `X-Promptise-Signature` as a bare hex HMAC over `json.dumps(payload, sort_keys=True)`, with no timestamp. Receivers built for that format must switch to `verify_event_signature()` (or the recipe above).

#### Private networks

WebhookSink refuses `localhost` and every address that is not public unicast by default (SSRF protection): loopback, private ranges, link-local (cloud metadata at `169.254.169.254`), shared/carrier-grade NAT space (`100.64.0.0/10`), `0.0.0.0`, multicast and reserved ranges. The check runs when the sink is created and again before every delivery: the host is resolved, every address is checked, and the request goes to the checked address, so a DNS record that later points at an internal address (DNS rebinding) is not followed. Redirects are never followed. For a receiver on your own machine or private network, opt in:

```python
WebhookSink(url="http://127.0.0.1:8390/promptise", allow_private_networks=True)
```

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

The central coordinator. Routes events to configured sinks.

```python
notifier = EventNotifier(
    sinks=[sink_a, sink_b, sink_c],
    max_queue_size=1000,      # Per sink: drop events when its queue is full (never block)
    shutdown_timeout=10.0,    # How long stop() waits for queued events
    slow_tool_threshold=5.0,  # Seconds before tool.slow fires (None = off)
)
```

**Fire-and-forget**: `emit()` queues the event and returns immediately. The agent never blocks waiting for event delivery.

**Sink isolation**: every sink has its own queue and delivery task. A webhook that is down and retrying with backoff delays only its own deliveries — other sinks get every event immediately. Within one sink, events arrive in order. Sink failures are logged, never propagated.

**Graceful shutdown**: `agent.shutdown()` stops the notifier, which waits up to `shutdown_timeout` seconds for queued events to be delivered. Whatever is still undelivered then is dropped and **logged** (sink, count and event types) and counted in `notifier.dropped_count`. Size the timeout to your webhook retries: with the defaults, a retrying webhook needs about 7 seconds of backoff plus request time.

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
| `agent_id` | `str \| None` | Who emitted it — see below |
| `user_id` | `str \| None` | From `CallerContext` (multi-user) |
| `session_id` | `str \| None` | The `chat()` session, else `caller.metadata["session_id"]` |
| `data` | `dict` | Event-specific payload ([Payloads](#payloads)) |
| `metadata` | `dict` | Context: `model` and `invocation_id` on agent events; `process_name`, `process_id` (and `trigger_type`) on runtime events |

**`agent_id`** is the same on every event an agent emits: `observer_agent_id` when you pass it to `build_agent()`, else the [identity's](../identity/overview.md) `agent_id` (or IdP subject). Agents run by an `AgentProcess` use the process name. Without any of these it is the model name. The model is also in `metadata["model"]`.

```python
agent = await build_agent(..., events=notifier, observer_agent_id="billing-agent")
```

**`metadata["invocation_id"]`** is the same for every event of one `ainvoke()`/`chat()`/stream call, so you can group `invocation.start`, the `tool.error`s and `invocation.complete` of one run.

---

## Security

- **SSRF protection**: WebhookSink validates URLs at construction and again, against the address it connects to, before every delivery — rejects private, loopback, link-local (cloud metadata), CGNAT and other non-public addresses unless `allow_private_networks=True`
- **Signing with replay protection**: every webhook carries `X-Promptise-Signature` (HMAC over timestamp + raw body); verify with `verify_event_signature()`
- **Payload redaction**: WebhookSink scans the whole payload — `data`, `user_id`, `session_id`, `metadata` — for PII/credentials before sending (regex patterns, no ML models). A `user_id` that is an email address arrives as `[EMAIL]`; pass `redact_sensitive=False` if your receiver needs it
- **Queue bounds**: `max_queue_size` prevents memory exhaustion. When a sink's queue is full, events are dropped for that sink with a warning log
- **Sink isolation**: One failing or slow sink never affects others

---

## SuperAgent YAML

```yaml
events:
  shutdown_timeout: 10      # optional
  slow_tool_threshold: 5    # optional; null turns tool.slow off
  sinks:
    - type: webhook
      url: https://hooks.slack.com/services/...
      events: [invocation.error, budget.exceeded]
      min_severity: warning
    - type: webhook
      url: http://127.0.0.1:8390/promptise
      allow_private_networks: true
    - type: log
      events: [invocation.complete]
```

---

## EventNotifier Lifecycle

The notifier starts itself when the first event is emitted, and must be stopped to deliver what is still queued:

```python
notifier = EventNotifier(sinks=[...])
await notifier.start()   # Start the delivery tasks (optional — emitting starts them)

# ... agent runs, events are emitted ...

await notifier.stop()    # Deliver queued events (up to shutdown_timeout), then stop
```

When passed to `build_agent(events=notifier)`, `start()` is called automatically and `agent.shutdown()` calls `stop()`.

In the runtime, the process owns the notifier instead: `AgentProcess.stop()` emits `process.stopped` and then stops the notifier, so the last event is delivered too. A notifier shared by an `AgentRuntime` keeps running while single processes stop and is stopped by `runtime.stop_all()`.

| Method / attribute | Description |
|---|---|
| `await start()` | Start the delivery tasks. Auto-called by `build_agent()` and `AgentProcess.start()`. |
| `await stop(timeout=None)` | Deliver queued events (up to `timeout`, default `shutdown_timeout`), log anything dropped, close sinks, stop. |
| `await flush(timeout=None)` | Wait until everything queued is delivered, without stopping. Returns `False` on timeout. |
| `await emit(event)` | Queue an event for delivery (non-blocking). |
| `emit_sync(event)` | Queue from synchronous code, from any thread. Never raises. |
| `dropped_count` | Events dropped so far (full queues, shutdown timeout). |
| `is_running` | Whether the delivery tasks are running. |

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
| `agent_id` | `str \| None` | from context | Agent identifier |
| `session_id` | `str \| None` | from context | Session ID |
| `metadata` | `dict \| None` | `None` | Additional metadata, merged over the context's |
| `user_id` | `str \| None` | from context | The user the event concerns |

`emit_event()` is null-safe — passing `None` as the notifier does nothing. Called during an agent invocation (in a tool, for example), it fills `user_id` from the current `CallerContext` and `agent_id`, `session_id` and `metadata` from the running invocation.

---

## Sink Parameter Reference

### WebhookSink

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | required | Webhook URL (SSRF-protected) |
| `events` | `list[str] \| None` | `None` | Event types to subscribe to (None = all) |
| `headers` | `dict[str, str]` | `{}` | Custom HTTP headers |
| `secret` | `str \| None` | auto-generated | HMAC signing secret (readable as `sink.secret`) |
| `max_retries` | `int` | `3` | Retry attempts on failure |
| `retry_delay` | `float` | `1.0` | Initial retry delay (doubles each retry) |
| `redact_sensitive` | `bool` | `True` | Scan the whole payload for PII/credentials |
| `min_severity` | `str \| None` | `None` | Minimum severity to emit |
| `transform` | `Callable \| None` | `None` | Custom payload transformation (the signature covers the transformed body) |
| `allow_private_networks` | `bool` | `False` | Allow `localhost` and private-network URLs |

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

### EventNotifier

| Parameter | Type | Default | Description |
|---|---|---|---|
| `sinks` | `list[EventSink]` | required | At least one sink |
| `max_queue_size` | `int` | `1000` | Undelivered events kept per sink |
| `shutdown_timeout` | `float` | `10.0` | Seconds `stop()` waits for queued events |
| `slow_tool_threshold` | `float \| None` | `5.0` | Seconds before a tool call emits `tool.slow`; `None` disables it |

---

## What's Next?

- [Observability](observability.md) -- detailed execution traces (events are high-level alerts, observability is full traces)
- [Approval](approval.md) -- human-in-the-loop approval that emits `approval.*` events
- [Guardrails](guardrails.md) -- security scanning that emits `guardrail.*` events
