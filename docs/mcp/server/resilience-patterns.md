# Resilience Patterns

Keep your MCP server running reliably when things go wrong -- circuit breakers for flaky dependencies, health checks for monitoring, webhooks for alerting, background tasks for deferred work, and exception handlers for clean error responses.

## Circuit Breaker

### When you need it

Your MCP server wraps Stripe's payment API. When Stripe has an outage, every tool call to your server fails after a 30-second timeout. Without a circuit breaker, agents keep retrying, your server accumulates blocked connections, and everything slows to a crawl. With a circuit breaker, the server immediately tells agents "try again later" instead of waiting for Stripe to time out.

### `CircuitBreakerMiddleware`

```python
from promptise.mcp.server import MCPServer, CircuitBreakerMiddleware

server = MCPServer(name="payment-api")
server.add_middleware(CircuitBreakerMiddleware(
    failure_threshold=5,           # Open after 5 consecutive failures
    recovery_timeout=60.0,         # Wait 60s before testing recovery
    excluded_tools={"health"},     # Never circuit-break health checks
))

@server.tool()
async def charge_card(
    customer_id: str,
    amount_cents: int,
    currency: str = "usd",
) -> dict:
    """Charge a customer's card via Stripe.

    If Stripe is down, the circuit breaker trips after 5 failures
    and immediately rejects calls for 60 seconds instead of
    waiting for timeouts.
    """
    import stripe
    charge = await stripe.Charge.create(
        customer=customer_id,
        amount=amount_cents,
        currency=currency,
    )
    return {"charge_id": charge.id, "status": charge.status}
```

### How it works

```mermaid
stateDiagram-v2
    [*] --> Closed
    Closed --> Open: failure_threshold reached
    Open --> HalfOpen: recovery_timeout elapsed
    HalfOpen --> Closed: probe call succeeds
    HalfOpen --> Open: probe call fails
```

| State | Behavior |
|-------|----------|
| **Closed** | Normal operation. Each [failure](#what-counts-as-a-failure) increments a counter; a success resets it. |
| **Open** | All calls rejected immediately with `CircuitOpenError`. |
| **Half-Open** | Exactly one probe call is let through; calls arriving while it runs are rejected with `CircuitOpenError`. Success → Closed. Failure → Open for another `recovery_timeout`. A probe that ends without a verdict (the tool rejected bad input, or the call was cancelled) leaves the circuit half-open, and the next call probes. |

State is per tool: one failing tool doesn't pause the others.

### What counts as a failure

A circuit breaker should pause a tool when the tool or what it depends on is
unhealthy, not when one caller sends a bad request. By default only these
count towards `failure_threshold`:

- **Unexpected exceptions**: anything that isn't an `MCPError` (a crash, a
  `ConnectionError`, an upstream SDK raising).
- **Retryable `MCPError`s** that signal a transient failure: a `TIMEOUT`
  (from `@server.tool(timeout=...)` or `TimeoutMiddleware`), or your own
  `ToolError("Upstream unavailable", retryable=True)`.

These never count:

- A non-retryable `ToolError`: the tool worked and rejected this input
  (`ToolError(f"No product {sku}")`).
- Authentication and access denials, validation errors, approval denials.
- Rate-limit and concurrency refusals (`RATE_LIMIT_EXCEEDED`,
  `CONCURRENCY_LIMIT_EXCEEDED`), so a burst that hits `max_concurrent` doesn't
  open the circuit.
- `CircuitOpenError` itself.

An error that doesn't count leaves the failure streak as it is. To decide
differently, pass `is_failure`, a function from the exception to `bool`. The
default is exported as `is_upstream_failure`, so you can extend it:

```python
from promptise.mcp.server import CircuitBreakerMiddleware, is_upstream_failure

def is_failure(exc: BaseException) -> bool:
    if isinstance(exc, PaymentDeclinedError):   # the customer's problem, not Stripe's
        return False
    return is_upstream_failure(exc)

server.add_middleware(CircuitBreakerMiddleware(is_failure=is_failure))
```

### Handling `CircuitOpenError`

With no extra code, a call rejected by an open circuit reaches the agent as a
retryable error that says how long to wait:

```json
{
  "error": {
    "code": "CIRCUIT_OPEN",
    "message": "Circuit open for tool 'charge_card'. Retry after 42.0s.",
    "retryable": true,
    "suggestion": "'charge_card' is paused after repeated failures. Wait about 42s before calling it again, or continue without it.",
    "details": {"tool": "charge_card", "retry_after_seconds": 42.0}
  }
}
```

`CircuitOpenError` is an `MCPError` with `tool` and `retry_after` attributes.
Register an exception handler to reshape it:

```python
from promptise.mcp.server import CircuitOpenError

@server.exception_handler(CircuitOpenError)
async def handle_circuit_open(ctx, exc):
    from promptise.mcp.server import ToolError
    return ToolError(
        message=f"Service temporarily unavailable. Retry in {exc.retry_after:.0f}s.",
        code="SERVICE_UNAVAILABLE",
        retryable=True,
    )
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `failure_threshold` | `5` | Consecutive failures before opening |
| `recovery_timeout` | `60.0` | Seconds before probing recovery |
| `excluded_tools` | `set()` | Tools exempt from circuit breaking |
| `is_failure` | `is_upstream_failure` | `(exc) -> bool`: whether an exception counts towards opening the circuit |

### Programmatic control

```python
cb = CircuitBreakerMiddleware(failure_threshold=5)
server.add_middleware(cb)

# Check state
cb.get_state("charge_card")  # CircuitState.CLOSED / OPEN / HALF_OPEN

# Manual reset (e.g., after fixing the dependency)
cb.reset("charge_card")  # Reset one tool
cb.reset()               # Reset all
```

---

## Health Checks

### When you need it

Your MCP server runs in Kubernetes. Kubernetes needs to know if the server is alive (liveness probe) and ready to accept traffic (readiness probe). If your database is down, the server should report "not ready" so Kubernetes routes traffic elsewhere.

### `HealthCheck`

```python
from promptise.mcp.server import MCPServer, HealthCheck

server = MCPServer(name="api")
health = HealthCheck()

# Required check — server is "not ready" if this fails
async def check_database() -> bool:
    try:
        await db.execute("SELECT 1")
        return True
    except Exception:
        return False

health.add_check("database", check_database, required_for_ready=True)

# Optional check — logged but doesn't affect readiness
async def check_cache() -> bool:
    return cache.is_connected()

health.add_check("cache", check_cache, required_for_ready=False)

# Register as MCP resources
health.register_resources(server)
```

This exposes two resources:

| Resource URI | Purpose |
|-------------|---------|
| `health://liveness` | Liveness: is the server process running? |
| `health://readiness` | Readiness: are all required dependencies available? |

Agents and MCP clients can read these resources to check server health.

Container and Kubernetes probes cannot speak MCP, so over HTTP and SSE the same checks back two plain routes: `GET /health` (liveness, always `200`) and `GET /health/ready` (`200` when every required check passes, `503` otherwise). They skip the auth gate and never include the text of an exception a check raised. See [Deployment — Health probes](deployment.md#health-probes).

---

## Webhook Notifications

### When you need it

Your ops team uses Slack for alerts. When a tool call fails, you want a Slack message immediately -- not waiting for someone to check logs.

### `WebhookMiddleware`

```python
from promptise.mcp.server import MCPServer, WebhookMiddleware

server = MCPServer(name="api")
server.add_middleware(WebhookMiddleware(
    url="https://hooks.slack.com/services/T.../B.../xxx",
    events={"tool.error"},              # Only fire on errors
    headers={"Authorization": "Bearer slack-token"},
    timeout=5.0,                        # Don't block tool calls
))
```

### Events

| Event | When fired |
|-------|-----------|
| `tool.call` | Before every tool execution |
| `tool.success` | After successful execution |
| `tool.error` | After an exception |

### Webhook payload

```json
{
  "event": "tool.error",
  "tool": "charge_card",
  "client_id": "checkout-agent",
  "request_id": "f6e5d4",
  "timestamp": 1709812345.6,
  "error": "stripe.error.CardDeclinedError: Card declined"
}
```

Webhooks are fire-and-forget -- they never block or fail the tool call. If the webhook endpoint is unreachable, the failure is logged and the tool call proceeds normally.

---

## Background Tasks

### When you need it

After creating a user, you want to send a welcome email and log an analytics event. These are important but shouldn't delay the tool response.

### `BackgroundTasks`

```python
from promptise.mcp.server import MCPServer, BackgroundTasks, Depends

server = MCPServer(name="hr-api")

async def send_welcome_email(employee_id: str, email: str):
    """Send welcome email (runs after response is sent)."""
    await email_service.send(
        to=email,
        subject="Welcome to the team!",
        template="welcome",
        data={"employee_id": employee_id},
    )

async def log_audit_event(action: str, actor: str, details: dict):
    """Log to external audit system."""
    await audit_api.log(action=action, actor=actor, details=details)

@server.tool(auth=True)
async def create_employee(
    name: str,
    email: str,
    department: str,
    bg: BackgroundTasks = Depends(BackgroundTasks),
) -> dict:
    """Create an employee.

    Welcome email and audit log run in the background after the
    response is returned. If they fail, it's logged but doesn't
    affect the response.
    """
    from promptise.mcp.server import get_context
    ctx = get_context()

    emp_id = await db.create_employee(name=name, email=email, dept=department)

    bg.add(send_welcome_email, emp_id, email)
    bg.add(log_audit_event, "CREATE_EMPLOYEE", ctx.client_id, {"id": emp_id})

    return {"id": emp_id, "name": name, "status": "created"}
```

Background tasks start once the handler has returned, in a task of their own, so the client gets its response right away -- it never waits for the email. They run sequentially; if one raises, the error is logged (never sent to the client) and the remaining tasks still run.

- **Request context:** `get_context()` still works inside a background task and returns the request that scheduled it.
- **Dependencies:** dependencies with cleanup (`yield`) are closed when the call returns, before background tasks run. Pass background tasks plain values, not a request-scoped database session.
- **Shutdown:** on graceful shutdown the server waits for background tasks still running (bounded by `shutdown_timeout`) before running your shutdown hooks.
- **Not durable:** they live in the server process; a crash or restart drops them. Use [MCPQueue](queue.md) for work that must be tracked, retried or polled.
- **Tests:** `TestClient` runs background tasks before `call_tool` returns, so a test can assert on their effects directly.

---

## Exception Handlers

### When you need it

Your tools raise domain-specific exceptions (`EmployeeNotFoundError`, `InsufficientFundsError`). Without exception handlers, agents see generic "Internal error" messages. With handlers, they get structured, actionable error responses.

### Custom exception mapping

```python
from promptise.mcp.server import MCPServer, ToolError

class EmployeeNotFoundError(Exception):
    def __init__(self, employee_id: str):
        self.employee_id = employee_id
        super().__init__(f"Employee {employee_id} not found")

class InsufficientPermissionsError(Exception):
    pass

server = MCPServer(name="hr-api")

@server.exception_handler(EmployeeNotFoundError)
async def handle_not_found(ctx, exc):
    return ToolError(
        message=f"Employee '{exc.employee_id}' does not exist.",
        code="EMPLOYEE_NOT_FOUND",
        retryable=False,
    )

@server.exception_handler(InsufficientPermissionsError)
async def handle_permissions(ctx, exc):
    return ToolError(
        message="You don't have permission for this action.",
        code="FORBIDDEN",
        retryable=False,
    )

@server.tool()
async def get_employee(employee_id: str) -> dict:
    """Get an employee by ID."""
    record = await db.get_employee(employee_id)
    if record is None:
        raise EmployeeNotFoundError(employee_id)
    return record
```

The handler receives the `RequestContext` and the exception. It returns a `ToolError` that's sent to the client as a structured error response.

**MRO-based matching**: If you register a handler for `ValueError` and throw a `SpecificValueError(ValueError)`, the `ValueError` handler catches it. The most specific handler in the MRO wins.

**Structured errors**: an `MCPError` (`ToolError`, `RateLimitError`,
`CircuitOpenError`, ...) is already a structured response, so it is only
passed to a handler registered for its own class or another `MCPError`
subclass, such as `@server.exception_handler(CircuitOpenError)`. A catch-all
`@server.exception_handler(Exception)` doesn't swallow it.

**Unhandled exceptions** reach the client as
`{"code": "INTERNAL_ERROR", "message": "An internal error occurred."}`; the
exception text (which may hold connection strings or file paths) goes to the
server log only. `TestClient` returns the same response.

---

## Progress Reporting

### When you need it

Your tool processes a large dataset and takes 30+ seconds. Without progress, the agent (or human watching) has no idea if it's stuck or working.

### `ProgressReporter`

```python
from promptise.mcp.server import MCPServer, ProgressReporter, Depends

server = MCPServer(name="data-pipeline")

@server.tool()
async def process_dataset(
    dataset_url: str,
    progress: ProgressReporter = Depends(ProgressReporter),
) -> dict:
    """Process a large dataset with progress reporting."""
    await progress.report(0, total=100, message="Downloading dataset...")
    data = await download(dataset_url)

    rows = parse_csv(data)
    processed = 0
    for i, row in enumerate(rows):
        await transform_row(row)
        processed += 1
        if i % 100 == 0:
            pct = int((i / len(rows)) * 100)
            await progress.report(pct, total=100, message=f"Processed {i}/{len(rows)} rows")

    await progress.report(100, total=100, message="Complete")
    return {"processed": processed, "total": len(rows)}
```

Progress notifications are sent via MCP's `notifications/progress`. The client receives them in real-time and can display a progress bar or status message.

If the client doesn't support progress (no `progressToken` in the request), the `report()` calls are silently ignored.

### Receiving progress in a Promptise client or agent

A client asks for progress per call. With `MCPClient`, pass a `progress_callback`; it is awaited for every notification the server sends for that call:

```python
from promptise.mcp.client import MCPClient

async def on_progress(progress: float, total: float | None, message: str | None) -> None:
    print(f"{progress}/{total}: {message}")

async with MCPClient(url="http://localhost:8080/mcp") as client:
    result = await client.call_tool(
        "process_dataset",
        {"dataset_url": "s3://bucket/sales.csv"},
        progress_callback=on_progress,
    )
```

`MCPMultiClient.call_tool()` takes the same argument. Agents get it through `build_agent`:

```python
from promptise import build_agent
from promptise.config import HTTPServerSpec

def on_tool_progress(tool_name: str, progress: float, total: float | None, message: str | None):
    print(f"[{tool_name}] {progress}/{total} {message or ''}")

agent = await build_agent(
    servers={"pipeline": HTTPServerSpec(url="http://localhost:8080/mcp")},
    model="openai:gpt-5-mini",
    on_tool_progress=on_tool_progress,  # sync or async
)
```

With `events=EventNotifier(...)`, each notification is also emitted as a `tool.progress` event (`data`: `tool_name`, `progress`, `total`, `message`), and `trace_tools=True` prints it. A client only sends a progress token when one of these is set, so servers don't send progress nobody reads.

---

## Cancellation

### When you need it

An agent starts a 5-minute data processing job, then the user decides they don't need it anymore. Without cancellation support, the job runs to completion, wasting resources.

### `CancellationToken`

```python
from promptise.mcp.server import MCPServer, CancellationToken, Depends

server = MCPServer(name="data-pipeline")

@server.tool()
async def long_running_task(
    dataset: str,
    cancel: CancellationToken = Depends(CancellationToken),
) -> dict:
    """Process a dataset. Can be cancelled by the client."""
    results = []
    for chunk in load_chunks(dataset):
        cancel.check()  # Raises CancelledError if cancelled
        results.extend(await process_chunk(chunk))
    return {"processed": len(results)}
```

When the client cancels the call (an MCP `notifications/cancelled` for its request id), the client is answered `Request cancelled` straight away, and the server:

1. sets the token (`cancel.is_cancelled` is `True`, `cancel.reason` is `"Request cancelled by the client"` -- the MCP SDK doesn't pass on the client's own reason),
2. gives the handler `cancel_grace_period` seconds (default 5) to stop on its own -- `cancel.check()` raises, `cancel.wait()` returns `True`,
3. then cancels the handler's task if it is still running.

Whatever the handler returns after the cancellation is discarded. Tools that don't take a `CancellationToken` are cancelled immediately, as before. Tune or disable the grace period per server:

```python
server = MCPServer(name="data-pipeline", cancel_grace_period=1.0)  # 0 = cancel the task right away
```

You can also wait for cancellation with a timeout:

```python
@server.tool()
async def poll_for_updates(
    topic: str,
    cancel: CancellationToken = Depends(CancellationToken),
) -> dict:
    """Poll until cancelled or timeout."""
    updates = []
    while True:
        cancelled = await cancel.wait(timeout=5.0)
        if cancelled:
            break
        new = await fetch_updates(topic)
        updates.extend(new)
    return {"updates": updates}
```

---

## Combining Resilience Features

A real production server uses multiple resilience patterns together:

```python
from promptise.mcp.server import (
    MCPServer, AuthMiddleware, JWTAuth,
    CircuitBreakerMiddleware, WebhookMiddleware,
    AuditMiddleware, HealthCheck,
)

server = MCPServer(name="payment-api")
health = HealthCheck()

# Middleware stack
server.add_middleware(AuditMiddleware(log_path="audit.jsonl", signed=True))
server.add_middleware(WebhookMiddleware(
    url="https://hooks.slack.com/services/...",
    events={"tool.error"},
))
server.add_middleware(CircuitBreakerMiddleware(
    failure_threshold=5,
    recovery_timeout=60.0,
    excluded_tools={"health"},
))
server.add_middleware(AuthMiddleware(JWTAuth(secret="...")))

# Health checks
async def check_stripe():
    return await stripe_client.is_available()

health.add_check("stripe", check_stripe, required_for_ready=True)
health.register_resources(server)
```

---

## API Summary

| Symbol | Type | Description |
|--------|------|-------------|
| `CircuitBreakerMiddleware(...)` | Class | Circuit breaker for downstream protection |
| `CircuitOpenError` | Exception | Raised when circuit is open; reaches the client as retryable `CIRCUIT_OPEN` with `retry_after_seconds` |
| `is_upstream_failure(exc)` | Function | Default breaker failure classifier |
| `CircuitState` | Enum | `CLOSED`, `OPEN`, `HALF_OPEN` |
| `HealthCheck()` | Class | Health and readiness probe manager |
| `WebhookMiddleware(url, events, ...)` | Class | Fire webhooks on tool events |
| `BackgroundTasks()` | Class | Fire-and-forget task scheduler |
| `ExceptionHandlerRegistry` | Class | Map exceptions to MCP error responses |
| `ProgressReporter` | Class | Report progress during long tools (via DI) |
| `CancellationToken` | Class | Check/wait for client cancellation (via DI) |

## What's Next

- [Caching & Performance](caching-performance.md) -- Cache, rate limit, concurrency control
- [Observability & Monitoring](observability.md) -- Metrics, tracing, Prometheus, logging
- [Advanced Patterns](advanced-patterns.md) -- Composition, versioning, transforms, OpenAPI
