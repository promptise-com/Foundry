# Event and Webhook Triggers

The `EventTrigger`, `MessageTrigger`, and `WebhookTrigger` provide event-driven activation for agent processes. EventTrigger listens to the framework's internal EventBus, MessageTrigger subscribes to MessageBroker topics, and WebhookTrigger exposes an HTTP endpoint for external systems.

```python
from promptise.runtime import ProcessConfig, TriggerConfig

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Handle incoming events and webhooks.",
    triggers=[
        TriggerConfig(type="event", event_type="pipeline.error"),
        TriggerConfig(type="webhook", webhook_path="/events", webhook_port=9090),
        TriggerConfig(type="message", topic="reports.daily"),
    ],
)
```

---

## Concepts

All three trigger types use an internal `asyncio.Queue` to decouple event delivery from consumption. Events arriving faster than the agent can process them are buffered (up to the queue capacity). If the queue fills, events are dropped with a warning.

| Trigger | Source | Use Case |
|---|---|---|
| `EventTrigger` | Internal `EventBus` | React to events from other agent processes |
| `MessageTrigger` | Internal `MessageBroker` | Subscribe to topic-based message streams |
| `WebhookTrigger` | External HTTP POST | Receive events from external systems (CI/CD, monitoring, APIs) |

---

## EventTrigger

Wraps an `EventBus` subscription and converts matching events into `TriggerEvent` objects.

### Configuration

```python
from promptise.runtime import TriggerConfig

# Listen for pipeline errors
TriggerConfig(type="event", event_type="pipeline.error")

# With source filtering
TriggerConfig(
    type="event",
    event_type="task.completed",
    event_source="data-pipeline",
)
```

### Direct instantiation

```python
from promptise.runtime.triggers.event import EventTrigger
from promptise.runtime.events import EventBus

bus = EventBus()

trigger = EventTrigger(
    bus,
    "pipeline.error",
    source_filter="data-pipeline",
    trigger_id="pipe-err-listener",
)

await trigger.start()
event = await trigger.wait_for_next()
print(event.payload)
# {
#     "event_type": "pipeline.error",
#     "event_id": "...",
#     "source": "data-pipeline",
#     "data": {...},
# }
await trigger.stop()
```

### Source filtering

When `source_filter` is set, only events from that specific source fire the trigger. Events from other sources are silently ignored. This is useful when multiple producers publish to the same event type.

### Event payload

| Field | Description |
|---|---|
| `event_type` | The EventBus event type string |
| `event_id` | Unique event identifier |
| `source` | Event source |
| `data` | Event data dict |

---

## MessageTrigger

Wraps a `MessageBroker` subscription for topic-based messaging.

### Configuration

```python
from promptise.runtime import TriggerConfig

# Subscribe to a specific topic
TriggerConfig(type="message", topic="reports.daily")

# Wildcard topics are supported by the broker
TriggerConfig(type="message", topic="reports.*")
```

### Direct instantiation

```python
from promptise.runtime.triggers.event import MessageTrigger
from promptise.runtime.broker import MessageBroker

broker = MessageBroker()

trigger = MessageTrigger(
    broker,
    "reports.daily",
    trigger_id="daily-report-sub",
)

await trigger.start()
event = await trigger.wait_for_next()
print(event.payload)
# {
#     "topic": "reports.daily",
#     "message_id": "...",
#     "sender": "report-generator",
#     "content": "...",
# }
await trigger.stop()
```

### Message payload

| Field | Description |
|---|---|
| `topic` | The topic the message arrived on |
| `message_id` | Unique message identifier |
| `sender` | Message sender |
| `content` | Message content |

---

## WebhookTrigger

An `aiohttp` HTTP server that listens for incoming POST requests and converts them to `TriggerEvent` objects.

### Configuration

```python
import os
from promptise.runtime import TriggerConfig

TriggerConfig(
    type="webhook",
    webhook_path="/github",
    webhook_port=9090,
    webhook_host="127.0.0.1",          # default: loopback only
    hmac_secret=os.environ["GITHUB_WEBHOOK_SECRET"],
    signature_scheme="github",         # generic | github | stripe
    allowed_sources=["203.0.113.0/24"],  # the sender's published IP ranges
    filter_expression="payload['action'] == 'opened'",
)
```

| Field | Default | Description |
|---|---|---|
| `webhook_path` | `"/webhook"` | URL path that accepts POST requests |
| `webhook_port` | `9090` | TCP port (1025–65535) |
| `webhook_host` | `"127.0.0.1"` | Interface to bind. The default only accepts local connections; use `"0.0.0.0"` to accept requests from other machines |
| `hmac_secret` | `None` | Shared secret; when set, requests without a valid signature get `401` |
| `signature_scheme` | `"generic"` | How the signature is computed (table below) |
| `signature_header` | scheme default | Override the header the signature is read from |
| `signature_tolerance` | `300` | Max age in seconds of a `stripe` signature timestamp; inside that window each signed request is accepted once (replay protection) |
| `allowed_sources` | `[]` (any) | Client IPs or CIDR ranges allowed to call the webhook; others get `403` |
| `filter_expression` | `None` | Skip events before the agent runs (see [Filtering](index.md#filtering-events-before-the-agent-runs)) |

In a manifest, keep the secret out of the file with an environment reference: `hmac_secret: ${GITHUB_WEBHOOK_SECRET}`.

### Authentication: HMAC signatures

With `hmac_secret` set, every request must carry an HMAC-SHA256 signature of the raw body, made with the shared secret. The comparison is timing-safe.

| `signature_scheme` | Header | Value |
|---|---|---|
| `generic` | `X-Webhook-Signature` | `sha256=<hex HMAC of body>` |
| `github` | `X-Hub-Signature-256` | `sha256=<hex HMAC of body>`, as GitHub sends it |
| `stripe` | `Stripe-Signature` | `t=<unix ts>,v1=<hex HMAC of "<ts>.<body>">`, as Stripe sends it; rejected when the timestamp is older than `signature_tolerance`, or when the same signed request was already accepted |

`generic` and `github` signatures cover only the body, with no timestamp, so anyone who captures a signed request can send it again. Serve the webhook over TLS (a reverse proxy in front of it) so requests can't be captured, and make the agent's actions safe to repeat where you can.

Signing a request for the `generic` scheme:

```python
import hashlib, hmac, json, urllib.request

body = json.dumps({"order_id": 42}).encode()
signature = "sha256=" + hmac.new(b"my-secret", body, hashlib.sha256).hexdigest()
urllib.request.urlopen(urllib.request.Request(
    "http://127.0.0.1:9090/webhook",
    data=body,
    headers={"Content-Type": "application/json", "X-Webhook-Signature": signature},
))
```

For other providers that sign the body with HMAC-SHA256 and a `sha256=` prefix, keep `signature_scheme="generic"` and set `signature_header` to their header name.

!!! warning "Set a secret before you expose the port"
    Without `hmac_secret`, anyone who can reach the port can make the agent run, and pick what it reads. The trigger logs a warning at start-up in that case. Always set a secret (and ideally `allowed_sources`) when `webhook_host` isn't loopback.

`allowed_sources` is checked against the TCP peer address. Behind a reverse proxy that is the proxy's address, so list the proxy and do the provider IP check there.

### Direct instantiation

```python
from promptise.runtime.triggers.webhook import WebhookTrigger

trigger = WebhookTrigger(
    path="/webhook",
    port=9090,
    host="127.0.0.1",
    hmac_secret="my-secret",
    signature_scheme="generic",
)

await trigger.start()
# HTTP server now listening at http://127.0.0.1:9090/webhook

event = await trigger.wait_for_next()
print(event.payload)   # POST body (JSON or text)
print(event.metadata)  # {"method": "POST", "path": "/webhook", ...}

await trigger.stop()
```

### How requests are handled

1. A POST request arrives at the configured path.
2. The client address is checked against `allowed_sources` → `403` if not allowed.
3. With `hmac_secret` set, the signature is verified → `401` if missing or wrong.
4. If the owning process can't take the event (it is `FAILED` or stopping, or its queue is full) → `503` with `Retry-After: 30`.
5. The body is parsed as JSON (falling back to plain text) and a `TriggerEvent` is created with the body as `payload` and request details in `metadata`.
6. If a `filter_expression` is set and doesn't match → `200 {"status": "ignored", "event_id": ...}`; the agent doesn't run.
7. Otherwise the event is queued and `202 {"status": "accepted", "event_id": ...}` is returned.

| Status | Body `status` | Meaning |
|---|---|---|
| `202` | `accepted` | Queued for the agent |
| `200` | `ignored` | Didn't match `filter_expression` |
| `401` | `error` | Missing or invalid signature |
| `403` | `error` | Source address not in `allowed_sources` |
| `503` | `unavailable` | Process failed/stopping or queue full; retry later |

Senders such as GitHub and Stripe retry on `5xx`, so events sent while the process is down are redelivered once it is running again, instead of piling up in a process that can't run them.

Sensitive headers (`Authorization`, `Proxy-Authorization`, `Cookie`, `Set-Cookie`) and the signature header are stripped from the metadata.

### Health check endpoint

The webhook server also exposes `GET /health`. It returns `200` while events are accepted and `503` when they aren't:

```json
{
    "status": "healthy",
    "reason": null,
    "trigger_id": "webhook-9090/events",
    "queue_size": 0
}
```

### Webhook payload

The `payload` field contains the raw POST body:

- If the body is valid JSON, it is stored as a dict.
- Otherwise, it is stored as a string.

The agent sees it inside a delimited untrusted-data block; see [Trigger payloads are untrusted input](index.md#trigger-payloads-are-untrusted-input).

### Webhook metadata

| Field | Description |
|---|---|
| `method` | HTTP method (always `POST`) |
| `path` | Request path |
| `query` | Query parameters dict |
| `headers` | Safe headers (auth and signature headers excluded) |
| `remote` | Client IP address |
| `signature_verified` | `True` when the request passed the HMAC check |

---

## Shared Patterns

### Queue-based decoupling

All three triggers use `asyncio.Queue` to buffer events:

- `EventTrigger`: maxsize=100
- `MessageTrigger`: maxsize=100
- `WebhookTrigger`: maxsize=1000

### Graceful stop

When `stop()` is called, each trigger:

1. Unsubscribes from its event source (EventBus, MessageBroker, or shuts down the HTTP server).
2. Enqueues a sentinel event with `metadata={"_stop": True}`.
3. Any `wait_for_next()` call receiving the sentinel raises `asyncio.CancelledError`.

---

## API Summary

| Class | Description |
|---|---|
| `EventTrigger(event_bus, event_type, source_filter, trigger_id)` | EventBus trigger |
| `MessageTrigger(broker, topic, trigger_id)` | MessageBroker trigger |
| `WebhookTrigger(path, port, host, hmac_secret, *, signature_scheme, signature_header, signature_tolerance, allowed_sources, event_filter)` | HTTP webhook trigger |

All implement the `BaseTrigger` protocol: `start()`, `stop()`, `wait_for_next()`.

---

## Tips and Gotchas

!!! tip "Use EventTrigger for inter-process coordination"
    When one process needs to trigger another, use `EventTrigger` with a shared `EventBus`. This is more efficient than webhooks for in-process communication.

!!! tip "Webhook security"
    The webhook server binds to `127.0.0.1` by default, so only local clients can reach it. When you bind `0.0.0.0`, set `hmac_secret` (and `allowed_sources` where the sender publishes its IP ranges), and put a reverse proxy (nginx, Caddy) in front for TLS. The trigger doesn't terminate TLS itself.

!!! info "aiohttp shipped with base install"
    `WebhookTrigger` uses `aiohttp`, which is included in the base `pip install promptise`.

!!! warning "EventBus/Broker must be shared"
    For `EventTrigger` and `MessageTrigger` to work, the same `EventBus` or `MessageBroker` instance must be shared between the event producer and the trigger. Pass them to the `AgentRuntime` constructor.

!!! warning "Queue capacity is finite"
    If events arrive faster than the agent processes them, the queue fills and the webhook answers `503` until there is room again. Monitor queue sizes via the dashboard or `status()` API. Increase `ProcessConfig.concurrency` for high-throughput scenarios.

---

## What's Next

- [Triggers Overview](index.md) -- all trigger types and the base protocol
- [Cron Trigger](cron.md) -- time-based scheduling
- [File Watch Trigger](file-watch.md) -- filesystem monitoring
- [Configuration](../configuration.md) -- full `TriggerConfig` reference
