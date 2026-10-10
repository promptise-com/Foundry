# Orchestration API

A complete REST API for managing your Agent Runtime — deploy agents, update configurations, send messages, and monitor health without touching code.

```python
from promptise.runtime import AgentRuntime, OrchestrationAPI

runtime = AgentRuntime()
api = OrchestrationAPI(
    runtime,
    host="0.0.0.0",
    port=9100,
    auth_token="${ORCHESTRATION_API_TOKEN}",
)
await api.start()
```

---

## Authentication

When `auth_token` is set, every request except `GET /api/v1/health` needs a Bearer token, including requests to unknown paths:

```
Authorization: Bearer <your-token>
```

- **Loopback** (`127.0.0.1`, `localhost`, `::1`): auth token is optional
- **Anything else**: auth token is **required** — the API refuses to start without one (an empty token is refused too)
- Comparison is timing-safe (`hmac.compare_digest`)
- Token supports env var resolution: `${ORCHESTRATION_API_TOKEN}`

### Browsers can't drive a local API

A loopback bind is reachable by any web page open in a browser on the same machine. On a loopback bind the API therefore refuses:

| Request | Status | Why |
|---|---|---|
| `Host` header that isn't a loopback name | `421` `MISDIRECTED_REQUEST` | DNS rebinding: a page whose own host name resolves to `127.0.0.1` |
| `Origin` header from a non-loopback site | `403` `CROSS_ORIGIN_REFUSED` | A cross-site `fetch` or form post from a page you visit |

`curl`, scripts and the Promptise clients send a loopback `Host` and no `Origin`, so they are unaffected. A local web UI served from `http://localhost:<port>` is allowed. Set an `auth_token` even on loopback if other users share the machine.

Request bodies must be JSON objects. Anything else (a list, a string, invalid JSON) gets `400` `INVALID_JSON`; routes that take no body accept an empty one.

---

## Endpoints

### Process Lifecycle

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/processes` | List all processes |
| `GET` | `/api/v1/processes/{name}` | Get process details (state, budget, health, mission) |
| `POST` | `/api/v1/processes` | Deploy a new process from JSON config |
| `POST` | `/api/v1/processes/{name}/start` | Start a process |
| `POST` | `/api/v1/processes/{name}/stop` | Stop a process |
| `POST` | `/api/v1/processes/{name}/restart` | Restart a process |
| `DELETE` | `/api/v1/processes/{name}` | Remove a process |

#### Deploy a new agent (no code required)

```bash
curl -X POST http://localhost:9100/api/v1/processes \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "incident-handler",
    "config": {
        "model": "openai:gpt-5-mini",
        "instructions": "Handle infrastructure incidents.",
        "triggers": [{"type": "webhook", "webhook_path": "/incidents", "webhook_port": 9091}],
        "budget": {"enabled": true, "max_tool_calls_per_day": 500}
    },
    "start": true
  }'
```

### Configuration Updates (hot-reload)

| Method | Path | Description | Takes effect |
|--------|------|-------------|-------------|
| `PATCH` | `/api/v1/processes/{name}/instructions` | Update system prompt | Next invocation |
| `PATCH` | `/api/v1/processes/{name}/budget` | Update budget limits | Immediately |
| `PATCH` | `/api/v1/processes/{name}/health` | Update health thresholds | Immediately |
| `PATCH` | `/api/v1/processes/{name}/mission` | Update mission objective | Immediately |

```bash
# Increase budget mid-operation
curl -X PATCH http://localhost:9100/api/v1/processes/monitor/budget \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"max_cost_per_day": 500.0}'
```

The body of a `budget`, `health` or `mission` patch holds the fields to change, named as in [`BudgetConfig`, `HealthConfig` and `MissionConfig`](governance/budget.md). The patched section is validated against its model before anything changes, with the same limits as at deploy time and strict JSON types (`"5"` is not a number, `1` is not `true`). A field the model does not declare, including inside `escalation` or `tool_costs`, is answered with `422 UNKNOWN_FIELD`; an invalid value with `422 INVALID_CONFIG`. In both cases nothing is changed, even when other fields in the same patch are valid. `null` clears an optional field such as `max_cost_per_day`.

### Trigger Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/processes/{name}/triggers` | List all triggers (static + dynamic) |
| `POST` | `/api/v1/processes/{name}/triggers` | Add a trigger |
| `DELETE` | `/api/v1/processes/{name}/triggers/{trigger_id}` | Remove a trigger |

```bash
# List triggers
curl http://localhost:9100/api/v1/processes/monitor/triggers \
  -H "Authorization: Bearer $TOKEN"

# Add a cron trigger
curl -X POST http://localhost:9100/api/v1/processes/monitor/triggers \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"type": "cron", "cron_expression": "0 9 * * 1-5"}'

# Remove a trigger
curl -X DELETE http://localhost:9100/api/v1/processes/monitor/triggers/static_0 \
  -H "Authorization: Bearer $TOKEN"
```

### Secret Management

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/processes/{name}/secrets` | List secret names and TTL (never values) |
| `PATCH` | `/api/v1/processes/{name}/secrets/{secret_name}` | Rotate a secret value |
| `DELETE` | `/api/v1/processes/{name}/secrets` | Revoke all secrets (zero-fill) |

```bash
# List secrets (names only — values are never exposed)
curl http://localhost:9100/api/v1/processes/payment-bot/secrets \
  -H "Authorization: Bearer $TOKEN"

# Rotate a secret
curl -X PATCH http://localhost:9100/api/v1/processes/payment-bot/secrets/stripe_key \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"value": "sk_live_new_key_here", "ttl": 3600}'

# Emergency revoke all secrets
curl -X DELETE http://localhost:9100/api/v1/processes/payment-bot/secrets \
  -H "Authorization: Bearer $TOKEN"
```

### Journal

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/processes/{name}/journal` | Read journal entries (supports `?limit=50&type=invocation_result`) |

```bash
# Read last 20 journal entries
curl "http://localhost:9100/api/v1/processes/monitor/journal?limit=20" \
  -H "Authorization: Bearer $TOKEN"

# Filter by entry type
curl "http://localhost:9100/api/v1/processes/monitor/journal?type=state_transition" \
  -H "Authorization: Bearer $TOKEN"
```

### Mission Control

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/processes/{name}/mission` | Get mission state + evaluation history |
| `POST` | `/api/v1/processes/{name}/mission/fail` | Manually fail the mission |
| `POST` | `/api/v1/processes/{name}/mission/pause` | Pause mission evaluation |
| `POST` | `/api/v1/processes/{name}/mission/resume` | Resume mission evaluation |

```bash
# Check mission progress
curl http://localhost:9100/api/v1/processes/migrator/mission \
  -H "Authorization: Bearer $TOKEN"

# Manually fail (e.g., requirements changed)
curl -X POST http://localhost:9100/api/v1/processes/migrator/mission/fail \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"reason": "Requirements changed, migration cancelled"}'
```

### Health Monitoring

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/processes/{name}/health/anomalies` | Get anomaly history |
| `DELETE` | `/api/v1/processes/{name}/health/anomalies` | Clear anomaly history |

```bash
# Check anomalies
curl http://localhost:9100/api/v1/processes/monitor/health/anomalies \
  -H "Authorization: Bearer $TOKEN"

# Clear after fixing the issue
curl -X DELETE http://localhost:9100/api/v1/processes/monitor/health/anomalies \
  -H "Authorization: Bearer $TOKEN"
```

### Process Suspend/Resume

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/processes/{name}/suspend` | Pause trigger processing |
| `POST` | `/api/v1/processes/{name}/resume` | Resume trigger processing |

```bash
# Pause during maintenance
curl -X POST http://localhost:9100/api/v1/processes/monitor/suspend \
  -H "Authorization: Bearer $TOKEN"

# Resume after maintenance
curl -X POST http://localhost:9100/api/v1/processes/monitor/resume \
  -H "Authorization: Bearer $TOKEN"
```

### Human Communication

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/processes/{name}/messages` | Send a message to the agent |
| `POST` | `/api/v1/processes/{name}/ask` | Ask a question (long-poll for response) |
| `GET` | `/api/v1/processes/{name}/inbox` | Get inbox status |
| `DELETE` | `/api/v1/processes/{name}/inbox` | Clear inbox |

Requires `inbox.enabled: true` in the process config.

```bash
# Send a directive
curl -X POST http://localhost:9100/api/v1/processes/monitor/messages \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"content": "Ignore staging alerts for 1 hour", "message_type": "directive", "priority": "high", "ttl": 3600}'

# Ask a question (waits for agent to respond)
curl -X POST http://localhost:9100/api/v1/processes/monitor/ask \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"content": "What anomalies have you detected?", "timeout": 120}'
```

### Observability

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/processes/{name}/metrics` | Process metrics (invocations, tools, budget) |
| `GET` | `/api/v1/processes/{name}/context` | Agent context state |
| `PATCH` | `/api/v1/processes/{name}/context` | Update context state (writable keys only) |

### Runtime Management

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/health` | No | Health probe for load balancers |
| `GET` | `/api/v1/runtime/status` | Yes | Global runtime status |
| `POST` | `/api/v1/runtime/start-all` | Yes | Start all processes |
| `POST` | `/api/v1/runtime/stop-all` | Yes | Stop all processes |

---

## Message Types

Messages sent via `/messages` support 4 types:

| Type | Use case | Example |
|------|----------|---------|
| `directive` | Tell the agent to do or stop doing something | "Do not restart any services today" |
| `context` | Provide information the agent should know | "Traffic spike is from marketing campaign" |
| `question` | Ask the agent something (expects response) | "What's the current health status?" |
| `correction` | Tell the agent it was wrong | "That alert was a false positive" |

Messages have **priority** (`low`, `normal`, `high`, `critical`) and **TTL** (auto-expire after N seconds).

---

## Process Config (for deployment)

The `config` object in `POST /api/v1/processes` follows the same schema as `ProcessConfig`:

```json
{
    "model": "openai:gpt-5-mini",
    "instructions": "Your agent instructions here.",
    "servers": {
        "tools": {"url": "http://tools:8080/mcp", "transport": "streamable-http"}
    },
    "triggers": [
        {"type": "cron", "cron_expression": "*/5 * * * *"}
    ],
    "inbox": {"enabled": true, "max_messages": 50},
    "budget": {"enabled": true, "max_tool_calls_per_day": 500},
    "health": {"enabled": true, "stuck_threshold": 5},
    "mission": {
        "enabled": true,
        "objective": "Keep pipelines healthy",
        "success_criteria": "Zero critical alerts for 24h"
    }
}
```

---

## Python SDK Client

Use `OrchestrationClient` to manage agents programmatically — same capabilities as the REST API but with typed methods, error handling, and async/await.

```python
from promptise.runtime import OrchestrationClient

async with OrchestrationClient("http://localhost:9100", auth_token="my-token") as client:
    # Deploy a new agent
    await client.deploy(
        name="monitor",
        config={"model": "openai:gpt-5-mini", "instructions": "Monitor pipelines."},
        start=True,
    )

    # Check status
    status = await client.get_process("monitor")
    print(status["state"])  # "running"

    # Send a message to the running agent
    await client.send_message("monitor", "Ignore staging alerts for 1 hour",
                              message_type="directive", priority="high", ttl=3600)

    # Ask a question and wait for the agent to respond
    response = await client.ask("monitor", "What anomalies have you found?", timeout=60)
    print(response["response"]["content"])

    # Update budget on the fly
    await client.update_budget("monitor", max_cost_per_day=500.0)

    # Rotate a secret without restart
    await client.rotate_secret("monitor", "api_key", "sk-new-key-here")

    # Read journal entries
    journal = await client.get_journal("monitor", limit=20)

    # Stop the agent
    await client.stop_process("monitor")
```

Every endpoint has a matching typed method:

| REST Endpoint | Client Method |
|---------------|---------------|
| `GET /processes` | `client.list_processes()` |
| `POST /processes` | `client.deploy(name, config)` |
| `POST /processes/{name}/start` | `client.start_process(name)` |
| `POST /processes/{name}/stop` | `client.stop_process(name)` |
| `POST /processes/{name}/restart` | `client.restart_process(name)` |
| `POST /processes/{name}/suspend` | `client.suspend_process(name)` |
| `POST /processes/{name}/resume` | `client.resume_process(name)` |
| `DELETE /processes/{name}` | `client.remove_process(name)` |
| `PATCH /instructions` | `client.update_instructions(name, text)` |
| `PATCH /budget` | `client.update_budget(name, **fields)` |
| `PATCH /health` | `client.update_health(name, **fields)` |
| `PATCH /mission` | `client.update_mission(name, **fields)` |
| `GET /triggers` | `client.list_triggers(name)` |
| `POST /triggers` | `client.add_trigger(name, config)` |
| `DELETE /triggers/{id}` | `client.remove_trigger(name, id)` |
| `GET /secrets` | `client.list_secrets(name)` |
| `PATCH /secrets/{name}` | `client.rotate_secret(name, secret, value)` |
| `DELETE /secrets` | `client.revoke_secrets(name)` |
| `GET /journal` | `client.get_journal(name, limit, type)` |
| `GET /mission` | `client.get_mission(name)` |
| `POST /mission/fail` | `client.fail_mission(name, reason)` |
| `POST /mission/pause` | `client.pause_mission(name)` |
| `POST /mission/resume` | `client.resume_mission(name)` |
| `GET /health/anomalies` | `client.get_anomalies(name)` |
| `DELETE /health/anomalies` | `client.clear_anomalies(name)` |
| `POST /messages` | `client.send_message(name, content, ...)` |
| `POST /ask` | `client.ask(name, content, timeout)` |
| `GET /inbox` | `client.get_inbox(name)` |
| `DELETE /inbox` | `client.clear_inbox(name)` |
| `GET /context` | `client.get_context(name)` |
| `PATCH /context` | `client.update_context(name, state)` |
| `GET /metrics` | `client.get_metrics(name)` |

---

## How Inbox Messages Reach the Agent

When you send a message via `/messages` or `/ask`, here's what happens:

1. **Message stored** — added to the process's `MessageInbox` with priority, TTL, and sender_id
2. **Next invocation** — before the agent is called, the runtime checks the inbox:
   - Expired messages are purged
   - Pending messages are formatted into a system prompt block
   - Injected between mission context and conversation history
3. **Agent sees the messages** — formatted as:
   ```
   ## Operator Messages
   [DIRECTIVE] (high priority): Ignore staging alerts for 1 hour
   [QUESTION Q1]: What anomalies have you found?
   If any questions are marked above, include "ANSWER Q1: <your response>" in your reply.
   ```
4. **Answer extraction** — after the agent responds, the runtime parses for `ANSWER Q1:` patterns and resolves the waiting future from the `/ask` call
5. **Cleanup** — processed directives and answered questions are marked as done

---

## Inbox Configuration

Enable the inbox on a process via `ProcessConfig`:

```python
ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="...",
    inbox=InboxConfig(
        enabled=True,
        max_messages=50,            # Max messages in inbox
        max_message_length=2000,    # Truncate long messages
        default_ttl=3600,           # Messages expire after 1 hour
        max_ttl=86400,              # Maximum TTL: 24 hours
        rate_limit_per_sender=100,  # Max 100 messages per sender per hour
    ),
)
```

Or via JSON in the deploy endpoint:

```json
{"inbox": {"enabled": true, "max_messages": 50, "default_ttl": 3600}}
```

---

## Error Responses

All errors follow a consistent format:

```json
{
    "error": {
        "code": "PROCESS_NOT_FOUND",
        "message": "Process 'unknown-agent' not found"
    }
}
```

| HTTP Code | Error Code | When |
|-----------|-----------|------|
| 400 | `INVALID_JSON` | Request body is not valid JSON |
| 400 | `SECRETS_DISABLED` | Secret operations on process without secrets |
| 401 | `UNAUTHORIZED` | Missing or invalid Bearer token |
| 403 | `WRITE_DENIED` | Writing to a non-writable context key |
| 404 | `PROCESS_NOT_FOUND` | Process name doesn't exist |
| 404 | `TRIGGER_NOT_FOUND` | Trigger ID doesn't exist |
| 409 | `PROCESS_EXISTS` | Deploy with a name that's already registered |
| 422 | `MISSING_FIELD` | Required field missing from request |
| 422 | `INVALID_CONFIG` | Config fails validation (deploy, add trigger, `PATCH` budget/health/mission) |
| 422 | `UNKNOWN_FIELD` | `PATCH` budget/health/mission names a field the config model does not declare |
| 429 | `RATE_LIMITED` | Inbox rate limit exceeded |
| 500 | `INTERNAL_ERROR` | Unexpected server error |

---

## Security

- Bearer token required on non-localhost (timing-safe comparison)
- Process names validated: alphanumeric + hyphens, max 64 chars
- Config validated via Pydantic before process creation
- Secret values are never returned in API responses (only names and TTL status)
- Health endpoint has no auth (for Kubernetes probes)
- Config updates (`PATCH` budget/health/mission) are validated against the config models before they are applied; unknown fields, out-of-range values and wrong types are refused with `422` and change nothing
- All mutating operations (POST/PATCH/DELETE) are logged

---

## What's Next?

- [Agent Processes](processes.md) — process lifecycle and configuration
- [Governance](governance/budget.md) — budget, health, and mission configuration
- [Triggers](triggers/index.md) — cron, webhook, file watch, event, message triggers
