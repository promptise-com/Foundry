# Observability & Monitoring

Track what your MCP server is doing in production with built-in metrics, distributed tracing, Prometheus export, structured logging, audit trails, and a live terminal dashboard.

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## Built-in Metrics

### `MetricsCollector`

Track per-tool call counts, error rates, and latency out of the box:

```python
from promptise.mcp.server import (
    MCPServer, MetricsCollector, MetricsMiddleware,
)

server = MCPServer(name="hr-api")
metrics = MetricsCollector()

server.add_middleware(MetricsMiddleware(metrics))

# Expose metrics as an MCP resource (agents can read them)
metrics.register_resource(server)
```

Now every tool call is tracked. Agents can read `metrics://server` to see:

```json
{
  "uptime_seconds": 3600,
  "tools": {
    "search_employees": {
      "calls": 142,
      "errors": 3,
      "avg_latency_ms": 45.2
    },
    "create_employee": {
      "calls": 12,
      "errors": 0,
      "avg_latency_ms": 128.7
    }
  }
}
```

### Live Dashboard

For development and debugging, enable the terminal dashboard:

```python
server.run(transport="http", port=8080, dashboard=True)
```

The dashboard shows 6 tabs (switch with 1-6 keys):

1. **Overview** -- Server info, uptime, key metrics
2. **Tools** -- Registered tools with per-tool call stats
3. **Agents** -- Connected clients and session details
4. **Logs** -- Scrolling request log
5. **Metrics** -- Performance data and error breakdown
6. **Raw Logs** -- Python logger output

---

## OpenTelemetry

### When you need it

Your MCP server is one piece of a larger system -- the agent calls your server, which calls a database, which calls a cache. When something is slow, you need to trace the entire request path across services.

### `OTelMiddleware`

```python
from promptise.mcp.server import MCPServer, OTelMiddleware

server = MCPServer(name="order-api")
server.add_middleware(OTelMiddleware(
    service_name="order-mcp-server",
    endpoint="http://jaeger:4317",  # OTLP collector endpoint
))
```

Each tool call becomes a span with these attributes:

| Attribute | Example |
|-----------|---------|
| `mcp.tool.name` | `"create_order"` |
| `mcp.request.id` | `"a3f2b1"` |
| `mcp.client.id` | `"agent-checkout"` |
| `mcp.tool.status` | `"ok"` or `"error"` |

The middleware also records:

- **Histogram**: `mcp.tool.duration` -- call duration distribution
- **Counter**: `mcp.tool.errors` -- error count by tool

**No-op when not installed**: If `opentelemetry-api` is not installed, the middleware passes through without overhead. Install with `pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp`.

### Real-world setup with Jaeger

```python
from promptise.mcp.server import MCPServer, OTelMiddleware

# In production, configure via env vars:
# OTEL_SERVICE_NAME=order-mcp-server
# OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4317

server = MCPServer(name="order-api")
server.add_middleware(OTelMiddleware(service_name="order-mcp-server"))

@server.tool()
async def create_order(customer_id: str, items: list[dict]) -> dict:
    """Create an order.

    The OTel middleware creates a parent span. Your code can add child spans:
    """
    from opentelemetry import trace
    tracer = trace.get_tracer("order-api")

    with tracer.start_as_current_span("validate_inventory"):
        await check_inventory(items)

    with tracer.start_as_current_span("charge_payment"):
        await charge_customer(customer_id, items)

    return {"order_id": "ord-123", "status": "confirmed"}
```

---

## Prometheus Metrics

### When you need it

Your ops team uses Grafana dashboards and Prometheus alerting. You need standard `/metrics` endpoint that Prometheus can scrape.

### `PrometheusMiddleware`

```python
from promptise.mcp.server import MCPServer, PrometheusMiddleware

server = MCPServer(name="api")
prom = PrometheusMiddleware(namespace="myapp")
server.add_middleware(prom)
```

Records three standard metrics:

| Metric | Type | Labels |
|--------|------|--------|
| `myapp_tool_calls_total` | Counter | `tool`, `status` |
| `myapp_tool_duration_seconds` | Histogram | `tool` |
| `myapp_tool_in_flight` | Gauge | `tool` |

### Exposing `/metrics` endpoint

Use `get_metrics_text()` to serve Prometheus format:

```python
# In your HTTP handler or resource
@server.resource("metrics://prometheus", mime_type="text/plain")
async def prometheus_metrics() -> str:
    return prom.get_metrics_text()
```

Output (Prometheus text exposition format):

```
# HELP myapp_tool_calls_total Total MCP tool calls
# TYPE myapp_tool_calls_total counter
myapp_tool_calls_total{tool="search_employees",status="ok"} 142
myapp_tool_calls_total{tool="search_employees",status="error"} 3
myapp_tool_calls_total{tool="create_employee",status="ok"} 12

# HELP myapp_tool_duration_seconds MCP tool call duration in seconds
# TYPE myapp_tool_duration_seconds histogram
myapp_tool_duration_seconds_bucket{tool="search_employees",le="0.5"} 138
myapp_tool_duration_seconds_bucket{tool="search_employees",le="1.0"} 141
```

**No-op when not installed**: If `prometheus-client` is not installed, the middleware passes through. Install with `pip install prometheus-client`.

### Custom Prometheus registry

For testing or multi-component setups, use a custom registry:

```python
from prometheus_client import CollectorRegistry

registry = CollectorRegistry()
prom = PrometheusMiddleware(namespace="myapp", registry=registry)
```

---

## Structured Logging

### When you need it

Your log aggregator (ELK, Datadog, CloudWatch) expects JSON-formatted log entries. Python's default logging produces unstructured text that's hard to parse and alert on.

### `StructuredLoggingMiddleware`

```python
from promptise.mcp.server import MCPServer, StructuredLoggingMiddleware

server = MCPServer(name="api")
server.add_middleware(StructuredLoggingMiddleware())
```

Every tool call emits a JSON log entry:

```json
{
  "event": "tool_call",
  "tool": "search_employees",
  "request_id": "a3f2b1",
  "duration_ms": 45.2,
  "status": "ok",
  "timestamp": "2026-03-07T10:30:00Z"
}
```

On error:

```json
{
  "event": "tool_call",
  "tool": "search_employees",
  "request_id": "b4c3d2",
  "duration_ms": 12.1,
  "status": "error",
  "error": "ConnectionError: database unreachable",
  "timestamp": "2026-03-07T10:30:05Z"
}
```

---

## Audit Logging

### When you need it

Your compliance team requires a tamper-evident record of every tool call -- who called what, when, with what arguments, and what happened. HIPAA, SOC 2, and GDPR compliance often require audit trails.

### `AuditMiddleware`

```python
from promptise.mcp.server import MCPServer, AuditMiddleware

server = MCPServer(name="medical-records-api")
server.add_middleware(AuditMiddleware(
    log_path="audit.jsonl",    # Write to file
    signed=True,               # HMAC chain for tamper detection
    hmac_secret="your-secret", # Or set PROMPTISE_AUDIT_SECRET env var
    include_args=True,         # Log tool arguments
    include_result=False,      # Don't log results (may contain PHI)
))
```

Each entry in `audit.jsonl`:

```json
{
  "timestamp": 1709812200.0,
  "tool": "view_patient_record",
  "client_id": "dr-smith-agent",
  "request_id": "c5d4e3",
  "status": "ok",
  "duration_s": 0.045,
  "args": {"patient_id": "P-12345"},
  "identity": {
    "subject": "dr-smith-agent",
    "issuer": "https://login.microsoftonline.com/<tenant>/v2.0",
    "audience": "api://medical-records-api",
    "roles": ["records.read"]
  },
  "seq": 0,
  "prev_hash": "0000...0000",
  "hmac": "a1b2c3d4..."
}
```

When the caller authenticated with a JWT (e.g. [`JwksAuth`](auth-security.md#jwksauth) for an agent presenting an [IdP identity](../../identity/overview.md)), the entry carries an `identity` block with the verified `subject` / `issuer` / `audience` / `roles` — inside the HMAC chain, so *which agent did what* is both attributable and tamper-evident. Only these descriptors are recorded; the token and full claim set are never written to the log.

### HMAC chain integrity

Each entry's `hmac` is an HMAC-SHA256, under your key, over the entry's fields including `prev_hash` — the previous entry's `hmac` (64 zeros for the first entry) — and `seq`, the entry's position in the chain. Editing, deleting, inserting or reordering entries breaks the chain, and without the key nobody can sign a replacement.

The key must be one you keep: `hmac_secret=` or `PROMPTISE_AUDIT_SECRET`. A signed log file without either raises `ValueError` when the middleware is created, because a random per-process key would make the file impossible to verify. (Without `log_path`, the in-memory chain still falls back to a random key, with a warning.)

**Restarts.** A server that starts on an existing log file continues the chain from the file's last entry, so one key verifies the whole file across restarts. The first entry after a restart carries `"resumed": true`.

**Crashes.** Each entry is written with a single append, in chain order. If the process dies during a write, the last line is left incomplete. On the next start, the server closes that line and continues the chain from the entry before it, and the verifier reports the fragment as a warning, not as tampering. If the chain advanced but the write failed (a full disk, for example), the entry is not recorded: the error is logged and the chain stays where it was, so the file never shows a gap.

**Key rotation.** Start a new log file with the new key. If you keep the same file, the server warns at startup and continues the chain, and you verify the file with both keys (`verify_audit_log(path, [new_key, old_key])`, or `--key-env` twice).

**Rotating the file.** If a log rotator moves the file while the server runs, the next file continues the chain. Verify the files together, oldest first: `verify_audit_log([rotated, current], key)`.

Run one writer per file. Two processes appending to the same file fork the chain, so give each worker its own `log_path`.

`audit.verify_chain()` checks the entries the middleware keeps in memory (the most recent `max_memory_entries`). To check a file, use the verifier.

### Verifying a log file

`verify_audit_log(path, key)` reads the file, checks every entry's signature and its link to the entry before it, and reports where the first break is and what kind of change it is:

```python
import asyncio
import os
from pathlib import Path

from promptise.mcp.server import AuditMiddleware, MCPServer, TestClient, verify_audit_log

os.environ.setdefault("PROMPTISE_AUDIT_SECRET", "change-me")  # in production: from your secrets manager
key = os.environ["PROMPTISE_AUDIT_SECRET"]
log = Path("audit.jsonl")
log.unlink(missing_ok=True)

server = MCPServer(name="records")
server.add_middleware(AuditMiddleware(log_path=str(log), include_args=True))


@server.tool()
async def view_record(patient_id: str) -> str:
    return f"record {patient_id}"


async def main() -> None:
    client = TestClient(server)
    for patient in ("P-1", "P-2", "P-3"):
        await client.call_tool("view_record", {"patient_id": patient})

    report = verify_audit_log(log, key)
    print(report.ok, report.entries)  # True 3

    # Someone deletes the second entry
    lines = log.read_text().splitlines(keepends=True)
    log.write_text(lines[0] + lines[2])

    report = verify_audit_log(log, key)
    print(report.ok, report.first_problem)
    # False audit.jsonl:2: deleted: 1 entry deleted between line 1 and this line (seq 0 -> 2)


asyncio.run(main())
```

`AuditVerification` has:

| Field | Meaning |
|-------|---------|
| `ok` | `False` if the log was tampered with |
| `entries` | Number of entries read |
| `first_problem` / `problems` | Each an `AuditIssue` with `kind`, `path`, `line` and `message` |
| `warnings` | Findings that are not tampering (below) |
| `truncated` | The last line is incomplete: the process stopped during a write |
| `continuous` | One unbroken chain from the first entry to the last |
| `restarts` | `(path, line)` of each entry where a restarted server continued the chain |
| `last_hash` / `last_seq` | The last entry's `hmac` and `seq` |

Problem kinds: `modified` (an entry edited after signing), `inserted` (a forged entry, or one copied from another log signed with the same key), `duplicate` (a second copy of an entry), `reordered`, `deleted` (with the number of missing entries), `missing_start` (the first entries are gone), `malformed` (a line that is not JSON), `unsigned`, `bad_signature` (modified or forged, when the two can't be told apart), `wrong_key` (no entry verifies) and `anchor_missing`.

Warnings: `truncated` (an incomplete last line with no newline, which is what a crash leaves), `crash_fragment` (such a line inside the log, followed by the restarted server's entry) and `chain_reset` (a new chain starts inside the file, as servers before 1.3.0 wrote on every restart).

A hash chain can't show entries cut from the **end** of the log. To detect that, store `last_hash` somewhere the log writer can't reach, such as a ticket or a separate system, and pass it later as `verify_audit_log(path, key, anchor=saved_hash)`. The check fails with `anchor_missing` if that entry is gone.

From the command line, the key comes from an environment variable and is never printed:

```bash
promptise audit verify audit.jsonl                        # key from PROMPTISE_AUDIT_SECRET
promptise audit verify audit.jsonl --key-env AUDIT_KEY    # key from another variable
promptise audit verify audit.1.jsonl audit.jsonl          # rotated files, oldest first
promptise audit verify audit.jsonl --anchor <last_hash> --strict --json
```

It exits `0` when the log is intact, `1` when it was tampered with (also on warnings with `--strict`) and `2` when the file or the key can't be read. See the [CLI reference](../../core/cli.md#promptise-audit-verify-audit-log-integrity).

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `log_path` | `None` | File path for the JSONL audit log |
| `signed` | `True` | Enable the HMAC chain |
| `hmac_secret` | `PROMPTISE_AUDIT_SECRET` | Key for the HMAC chain. Required with `log_path` when `signed=True` |
| `include_args` | `False` | Log the tool's arguments (may contain PII) |
| `include_result` | `False` | Log tool results (first 1000 characters) |
| `max_memory_entries` | `10000` | Entries kept in memory for `audit.entries` / `verify_chain()`; `None` keeps all. The file keeps every entry |

The log never contains the key, the caller's token or its full claim set. With `include_args` or `include_result`, it contains whatever the tool receives or returns. An entry's `error` field holds the exception text, which the client never sees.

---

## Server-to-Client Logging

### When you need it

Your tool runs a multi-step process and you want the agent (or human watching) to see progress messages -- not just the final result.

### `ServerLogger`

```python
from promptise.mcp.server import MCPServer, ServerLogger, Depends

server = MCPServer(name="data-pipeline")

@server.tool()
async def import_csv(
    file_url: str,
    logger: ServerLogger = Depends(ServerLogger),
) -> dict:
    """Import a CSV file into the database."""
    await logger.info("Downloading CSV...")
    data = await download(file_url)

    await logger.info(f"Parsing {len(data)} rows...")
    rows = parse_csv(data)

    await logger.warning(f"Skipped {skipped} invalid rows")

    await logger.info("Inserting into database...")
    await db.bulk_insert(rows)

    return {"imported": len(rows), "skipped": skipped}
```

Log messages are sent to the client via MCP's `notifications/message` protocol. The client can display them in real-time.

Available log levels: `debug`, `info`, `notice`, `warning`, `error`, `critical`, `alert`, `emergency`.

---

## Combining Observability Features

A production server typically layers multiple observability tools:

```python
from promptise.mcp.server import (
    MCPServer, AuthMiddleware, JWTAuth,
    MetricsCollector, MetricsMiddleware,
    OTelMiddleware, PrometheusMiddleware,
    StructuredLoggingMiddleware, AuditMiddleware,
)

server = MCPServer(name="production-api")
metrics = MetricsCollector()

# Observability stack (order matters)
server.add_middleware(StructuredLoggingMiddleware())    # JSON logs for every call
server.add_middleware(AuditMiddleware(                  # Compliance audit trail
    log_path="/var/log/mcp-audit.jsonl",
    signed=True,
))
server.add_middleware(OTelMiddleware(                   # Distributed tracing
    service_name="production-api",
))
server.add_middleware(PrometheusMiddleware())           # Prometheus metrics
server.add_middleware(MetricsMiddleware(metrics))       # Built-in metrics
server.add_middleware(AuthMiddleware(JWTAuth(...)))     # Auth (before tools)

metrics.register_resource(server)
```

---

## API Summary

| Symbol | Type | Description |
|--------|------|-------------|
| `MetricsCollector()` | Class | Per-tool call count, latency, error tracking |
| `MetricsMiddleware(collector)` | Class | Record metrics for every tool call |
| `MetricsCollector.register_resource(server)` | Method | Expose `metrics://server` resource |
| `OTelMiddleware(service_name, endpoint)` | Class | OpenTelemetry tracing middleware |
| `PrometheusMiddleware(namespace, registry)` | Class | Prometheus metrics middleware |
| `StructuredLoggingMiddleware()` | Class | JSON structured logging middleware |
| `AuditMiddleware(log_path, signed, ...)` | Class | HMAC-chained audit log middleware |
| `verify_audit_log(path, key, anchor=None)` | Function | Verify an audit log file; returns `AuditVerification` |
| `ServerLogger` | Class | Send log messages to MCP client (via DI) |
| `Dashboard` | Class | Live terminal monitoring dashboard |

## What's Next

- [Caching & Performance](caching-performance.md) -- Cache, rate limit, concurrency control
- [Resilience Patterns](resilience-patterns.md) -- Circuit breaker, health checks, webhooks
- [Deployment](deployment.md) -- HTTP deployment, CORS, containers
