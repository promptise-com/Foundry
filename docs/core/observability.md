# Observability

Track token usage and export execution traces across your agent systems.

```python
from promptise import build_agent
from promptise.config import HTTPServerSpec

# Simple: turn on observability with defaults
agent = await build_agent(
    servers={"tools": HTTPServerSpec(url="http://localhost:8000/mcp")},
    model="openai:gpt-5-mini",
    observe=True,
)

result = await agent.ainvoke({"messages": [{"role": "user", "content": "Hello"}]})

print(agent.get_stats()["total_tokens"])
agent.generate_report("report.html")  # interactive HTML report, written to exactly this path
await agent.shutdown()                 # also writes ./reports/promptise-report-<timestamp>.html
```

---

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## Concepts

Promptise observability is plug-and-play. Set `observe=True` and every agent invocation, LLM turn, tool call, token count, latency, retry, cache hit, and error is captured automatically: the agent records each invocation itself, and a LangChain callback handler records what happens inside it. Events are routed to one or more **transporters** -- HTML reports, structured logs, console output, Prometheus metrics, OpenTelemetry traces, or custom webhooks.

### What gets recorded

Every invocation (`ainvoke`, `chat`, `astream`, `astream_with_tools`) is an **agent run**:

| Event | When | Metadata |
|---|---|---|
| `agent.input` | The invocation starts | `input_length`, `model` (+ `input_preview` when text is recorded) |
| `llm.start` / `llm.end` | Each LLM turn (STANDARD and FULL) | `model`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `tool_calls` (+ `prompt_preview` / `response_preview`) |
| `tool.call` | A tool is called | `tool_name`, `arguments` |
| `tool.result` | The tool returned | `tool_name`, `latency_ms`, `result_preview` |
| `tool.error` | The tool failed — it raised, or an MCP server answered `isError: true` (e.g. a `ToolError`) | `tool_name`, `latency_ms`, `error_type`, `error` |
| `llm.error`, `llm.retry` | An LLM call failed or was retried | `error_type`, `error`, `attempt` |
| `cache.miss` / `cache.store` / `cache.hit` | Semantic cache activity | `scope`, `ttl` |
| `agent.output` | The invocation finished | `duration_ms`, the run's `prompt_tokens` / `completion_tokens` / `total_tokens`, `llm_call_count`, `tool_call_count`, `error_count`, `cache_hit`, `output_length` (+ `output_preview`) |
| `agent.error` | The invocation raised | `error_type`, `error`, and the same run totals |

Every event recorded during a run has `parent_id` set to the run's `agent.input` entry, so you can pull one request out of a busy timeline (`collector.query(...)` then filter on `parent_id`), and the OpenTelemetry transporter turns each run into one trace. Events also carry the caller's `user_id` and `session_id` (see [Multi-User Attribution](#multi-user-attribution)).

---

## ObservabilityConfig

For full control, pass an `ObservabilityConfig` instead of `True`:

```python
from promptise import build_agent
from promptise.observability_config import ObservabilityConfig, ObserveLevel, TransporterType

config = ObservabilityConfig(
    level=ObserveLevel.FULL,
    session_name="production-audit",
    record_prompts=True,
    transporters=[
        TransporterType.HTML,
        TransporterType.STRUCTURED_LOG,
        TransporterType.CONSOLE,
    ],
    output_dir="./reports",
    log_file="./logs/agent.jsonl",
    console_live=True,
)

agent = await build_agent(
    servers={"tools": HTTPServerSpec(url="http://localhost:8000/mcp")},
    model="openai:gpt-5-mini",
    observe=config,
)
```

### ObserveLevel

Controls the detail level of captured events:

| Level | Captures |
|---|---|
| `ObserveLevel.OFF` | Nothing. The agent keeps an empty `collector` (so `get_stats()` still works) and creates no transporters. |
| `ObserveLevel.BASIC` | `agent.input` / `agent.output` / `agent.error`, tool calls and results, LLM and tool errors, cache events. No per-turn LLM events; `agent.output` carries the run's token totals, and `get_stats()` counts them. |
| `ObserveLevel.STANDARD` | Everything in BASIC + `llm.start` / `llm.end` for every LLM turn, with token usage, latency, and the tools the model asked for. Default. |
| `ObserveLevel.FULL` | Everything in STANDARD + prompt, response, agent input and output text (truncated to 2,000 characters), and streamed-token counts. |

### What text is recorded

Two switches decide whether user and model text lands in your traces:

| Setting | Default | Records |
|---|---|---|
| `record_prompts` | `None` — follow the level: on at `FULL`, off below | `prompt_preview` / `response_preview` on LLM events, `input_preview` / `output_preview` on agent events. `True` records them at any level; `False` never does, even at `FULL`. |
| `record_tool_io` | `True` | `arguments` on `tool.call`, `result_preview` on `tool.result`, `error` and `traceback` on `tool.error`. With `False` the events keep the tool name, latency and error type, plus `arguments_length` / `result_length`. |

Tool arguments are on by default because they are what makes a trace debuggable — but they often carry user data (names, emails, account numbers). Set `record_tool_io=False` when traces leave your trust boundary (a hosted APM, a shared log pipeline). Cache events never copy the user's query into their text.

Whatever is recorded is scrubbed first: with `redact_sensitive=True` (the default) every event's metadata and details pass through `promptise.observability.redact_sensitive` before they are stored or reach a transporter. API keys (`sk-…`), AWS access keys, GitHub tokens, `Bearer` tokens, passwords in URLs (`postgres://user:pw@host`), card numbers, US social security numbers and email addresses become `[API_KEY]`, `[AWS_KEY]`, `[GITHUB_TOKEN]`, `Bearer [REDACTED]`, `://[REDACTED]@`, `[CARD]`, `[SSN]` and `[EMAIL]` — in the HTML report, the NDJSON stream, log files, webhooks and OpenTelemetry spans alike. Pattern matching cannot recognise names or addresses, so it complements `record_tool_io=False` rather than replacing it. When an agent run fails on a guardrail, `agent.error` records the direction, the number of blocked findings and their categories, never the matched text.

The redaction applies to the collector `build_agent()` creates. A collector you pass as `observer=` keeps its own `sanitizer` — pass `ObservabilityCollector(sanitizer=redact_sensitive)` to get the same behavior.

### TransporterType

Available backends for receiving observability events:

| Transporter | Description |
|---|---|
| `TransporterType.HTML` | Self-contained interactive HTML report (default) |
| `TransporterType.JSON` | JSON file export (full session dump + NDJSON streaming) |
| `TransporterType.STRUCTURED_LOG` | JSON log lines for ELK, Datadog, Splunk, CloudWatch |
| `TransporterType.CONSOLE` | Real-time Rich console output with color-coded events |
| `TransporterType.PROMETHEUS` | Prometheus metrics (counters, histograms) for Grafana |
| `TransporterType.OTLP` | OpenTelemetry span export via OTLP gRPC |
| `TransporterType.WEBHOOK` | HTTP POST each event to a configurable URL |
| `TransporterType.CALLBACK` | Invoke a user-provided Python callable for each event |

### Transporter Classes

Each `TransporterType` maps to a concrete class in `promptise.observability_transporters`:

```python
from promptise.observability_transporters import (
    HTMLReportTransporter,
    JSONFileTransporter,
    StructuredLogTransporter,
    ConsoleTransporter,
    PrometheusTransporter,
    OTLPTransporter,
    WebhookTransporter,
    CallbackTransporter,
)
```

| Class | Constructor | Description |
|-------|------------|-------------|
| `HTMLReportTransporter` | `(output_dir="./reports", session_name="promptise")` | Self-contained HTML report |
| `JSONFileTransporter` | `(output_dir="./reports", session_name="promptise", stream=True)` | NDJSON streaming or full JSON dump |
| `StructuredLogTransporter` | `(log_file=None, session_name="promptise", service_name="promptise", correlation_id=None)` | ELK/Datadog/Splunk-compatible structured logs |
| `ConsoleTransporter` | `(live=True, verbose=False)` | Rich-powered real-time terminal output |
| `PrometheusTransporter` | `(port=0)` | Prometheus metrics endpoint (counters, histograms) |
| `OTLPTransporter` | `(endpoint="http://localhost:4317", service_name="promptise")` | OpenTelemetry spans export |
| `WebhookTransporter` | `(url, headers=None, batch_size=0, max_retries=3, timeout=10.0)` | HTTP POST to external endpoint |
| `CallbackTransporter` | `(callback)` | Invoke custom callable per event |

All implement `on_event(entry)`, `flush()`, and `close()`.

**Custom transporter selection:**

```python
config = ObservabilityConfig(
    level=ObserveLevel.FULL,
    transporters=[TransporterType.HTML, TransporterType.PROMETHEUS],
)
agent = await build_agent(
    servers=servers, model="openai:gpt-5-mini", observe=config,
)
```

**Custom callback example:**

```python
def my_handler(entry):  # entry is a TimelineEntry
    print(f"[{entry.event_type.value}] {entry.agent_id}: {entry.details}")

config = ObservabilityConfig(transporters=[TransporterType.CALLBACK], on_event=my_handler)
# or attach one to an existing collector:
agent.collector.add_transporter(CallbackTransporter(callback=my_handler))
```

`TimelineEntry` fields: `entry_id`, `timestamp`, `event_type`, `category`, `agent_id`, `phase`, `details` (the human-readable text), `duration` (seconds), `parent_id`, `user_id`, `session_id`, `metadata`.

### Configuration Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `level` | `ObserveLevel` | `STANDARD` | Detail level |
| `session_name` | `str` | `"promptise"` | Human-readable session identifier (also the OTLP service name and the report/NDJSON file prefix) |
| `record_prompts` | `bool \| None` | `None` | Store prompt/response and agent input/output text. `None` = only at `FULL` |
| `record_tool_io` | `bool` | `True` | Store tool arguments, result previews and tool error messages |
| `redact_sensitive` | `bool` | `True` | Replace credentials and common PII in every event with placeholders before it is stored or exported |
| `max_entries` | `int` | `100_000` | Max timeline entries before eviction |
| `transporters` | `list[TransporterType]` | `[HTML]` | Active transporters |
| `output_dir` | `str \| None` | `None` | Directory for HTML and JSON output (`./reports` when unset) |
| `log_file` | `str \| None` | `None` | File path for STRUCTURED_LOG transporter |
| `console_live` | `bool` | `False` | Real-time console printing |
| `webhook_url` | `str \| None` | `None` | Target URL for WEBHOOK transporter |
| `webhook_headers` | `dict[str, str]` | `{}` | Custom HTTP headers for webhooks |
| `otlp_endpoint` | `str` | `"http://localhost:4317"` | gRPC endpoint for OTLP |
| `prometheus_port` | `int` | `9090` | Port for Prometheus metrics |
| `on_event` | `Callable \| None` | `None` | User callback for CALLBACK transporter |
| `correlation_id` | `str \| None` | `None` | Added to every structured-log line and OpenTelemetry span |

---

## PromptiseCallbackHandler

`PromptiseCallbackHandler` is the LangChain callback that bridges LLM events into the observability collector. It is instantiated once per agent and reused across multiple `ainvoke()` calls.

### Constructor

```python
from promptise.callback_handler import PromptiseCallbackHandler
from promptise.observability_config import ObserveLevel

handler = PromptiseCallbackHandler(
    collector,                          # ObservabilityCollector instance
    agent_id="my-agent",                # Optional agent identifier for timeline entries
    record_prompts=None,                # None = follow level (text at FULL only)
    level=ObserveLevel.STANDARD,        # Detail level (default: STANDARD)
    record_tool_io=True,                # Tool arguments / result previews (default: True)
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `collector` | `ObservabilityCollector` | **required** | The collector that receives timeline events |
| `agent_id` | `str \| None` | `None` | Agent identifier for the collector timeline |
| `record_prompts` | `bool \| None` | `None` | Include prompt/response text; `None` follows `level` |
| `level` | `ObserveLevel` | `STANDARD` | `OFF` (nothing), `BASIC` (tools + errors), `STANDARD` (+ LLM turns), `FULL` (+ text and streamed tokens) |
| `record_tool_io` | `bool` | `True` | Include tool arguments, result previews and error messages |

### Auto-Tracked Events

The handler automatically captures the following without any additional code:

- **LLM turns**: Start/end with latency timing, model name extraction
- **Token counts**: Prompt tokens, completion tokens, total tokens (from `LLMResult` and `usage_metadata`), also added to the current agent run
- **Tool calls**: Start/end with tool name, arguments, results, and latency
- **Errors**: LLM errors and tool errors (including MCP `isError` results) with traceback
- **Retries**: Retry attempts with attempt number and triggering error
- **Chain events**: Top-level input/output when the handler is used on its own (inside a `PromptiseAgent`, the agent records `agent.input` / `agent.output` itself)
- **Streaming tokens**: Token-by-token accumulation at `FULL` level

### Session Totals

After running the agent, cumulative totals are available on the handler:

| Attribute | Type | Description |
|---|---|---|
| `total_prompt_tokens` | `int` | Total input tokens |
| `total_completion_tokens` | `int` | Total output tokens |
| `total_tokens` | `int` | Total tokens (prompt + completion) |
| `llm_call_count` | `int` | Number of LLM calls |
| `tool_call_count` | `int` | Number of tool calls |
| `error_count` | `int` | Number of errors |
| `retry_count` | `int` | Number of retries |

Use `handler.get_summary()` to retrieve all metrics as a dict.

---

## Post-Run Analysis

After running an agent with observability enabled, use the built-in reporting methods:

```python
# Get runtime statistics
stats = agent.get_stats()

# Write an interactive HTML report
path = agent.generate_report("reports/shop.html", title="Shop assistant — Oct 10")
```

`generate_report(path, title="Agent Observability Report")` writes the report to exactly `path` (creating missing directories, overwriting an existing file) and returns it as a string. It raises `RuntimeError` when observability is off and `OSError` when the file cannot be written. The page is self-contained (no network access): stat cards with the same numbers as `get_stats()`, a timeline you can filter by Agent / LLM / Tool / Error / Cache, and each event's metadata on click.

Separately, the `HTML` transporter writes `<output_dir>/<session_name>-report-<timestamp>.html` when the agent shuts down.

### NDJSON event stream

The `JSON` transporter appends every event as one JSON line to `<output_dir>/<session_name>-events.ndjson`, and writes a full `<session_name>-session-<timestamp>.json` dump on shutdown. The NDJSON file is opened in append mode, so **every run with the same `session_name` adds to the same file** — useful for tailing and for `explain`-style scripts, but it grows without limit. Rotate it externally, for example with `logrotate` and `copytruncate`.

### OpenTelemetry traces

The `OTLP` transporter exports each agent run as **one trace**:

```text
invoke_agent shop                    3.7 s   enduser.id=customer-42  session.id=q1
├── chat gpt-5-mini                  1.4 s   gen_ai.usage.input_tokens=288
├── execute_tool check_stock         0.8 s   gen_ai.tool.name=check_stock
└── chat gpt-5-mini                  1.5 s
```

- The run span lasts from `agent.input` to `agent.output` (or `agent.error`); each LLM call (`llm.start` → `llm.end`) and tool call (`tool.call` → `tool.result` / `tool.error`) is a child span with its real start and end time, so span durations are the latencies.
- Failed tool calls, LLM calls and runs get an `ERROR` status. Other events of the run (cache hits, retries) are span events on the run span.
- If an OpenTelemetry span is active when you invoke the agent — say, the server span of an instrumented FastAPI app — the run span becomes its child, so the agent joins the request's trace.
- Attributes follow the GenAI semantic conventions (`gen_ai.operation.name`, `gen_ai.agent.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens` / `output_tokens`, `gen_ai.tool.name`), plus `enduser.id` and `session.id` from the `CallerContext`, `promptise.correlation_id` from the config, and every scalar metadata field as `promptise.<key>`.
- LLM spans need `STANDARD` or `FULL`; at `BASIC` a run has tool spans only.

`OTLPTransporter` creates its own `TracerProvider` and never replaces the global one. To send spans through your application's provider (and its exporters, sampling and resource attributes) instead, attach the transporter yourself:

```python
from opentelemetry import trace
from promptise.observability_transporters import OTLPTransporter

agent.collector.add_transporter(
    OTLPTransporter(tracer_provider=trace.get_tracer_provider(), correlation_id="req-7f3a")
)
```

---

## Enterprise Configuration Example

A production setup with multiple transporters:

```python
from promptise.observability_config import ObservabilityConfig, ObserveLevel, TransporterType

config = ObservabilityConfig(
    level=ObserveLevel.FULL,
    session_name="production-audit",
    record_prompts=True,
    transporters=[
        TransporterType.HTML,
        TransporterType.STRUCTURED_LOG,
        TransporterType.CONSOLE,
        TransporterType.PROMETHEUS,
    ],
    output_dir="./observability",
    log_file="./logs/events.jsonl",
    console_live=True,
    prometheus_port=9090,
    correlation_id="req-abc-123",
)
```

This configuration:

- Generates an interactive HTML report in `./observability/`
- Writes structured JSON log lines to `./logs/events.jsonl` (compatible with ELK, Datadog, Splunk)
- Prints color-coded events to the console in real time
- Exposes Prometheus metrics on port 9090
- Tags every structured-log line (and, with the `OTLP` transporter, every span) with the correlation ID `req-abc-123`

---

!!! tip "Privacy"
    Prompt and response text is recorded only at `ObserveLevel.FULL` or with `record_prompts=True`; tool arguments and results are recorded unless `record_tool_io=False`. See [What text is recorded](#what-text-is-recorded). This is particularly important in production environments handling sensitive data.

---

## Multi-User Attribution

Every `TimelineEntry` carries the authenticated caller's `user_id` and `session_id`. The `ObservabilityCollector` reads these automatically from the `CallerContext` contextvar at record time — there is nothing to wire manually.

```python
from promptise.agent import CallerContext, _caller_ctx_var
from promptise.observability import ObservabilityCollector, TimelineEventType

collector = ObservabilityCollector()
token = _caller_ctx_var.set(CallerContext(user_id="alice", metadata={"session_id": "sess-1"}))
try:
    entry = collector.record(TimelineEventType.TOOL_CALL, details="search(q=...)")
finally:
    _caller_ctx_var.reset(token)

assert entry.user_id == "alice"
assert entry.session_id == "sess-1"
```

### Per-tenant queries

| Method | Returns | Description |
|---|---|---|
| `collector.query(user_ids=[...], session_ids=[...])` | `list[TimelineEntry]` | Filter the ring buffer by user, session, event type, or time window |
| `collector.for_user(user_id)` | `list[TimelineEntry]` | Shorthand for a single tenant's events |
| `collector.get_users()` | `list[str]` | Unique sorted `user_id`s present in the buffer |
| `collector.purge_user(user_id)` | `int` | GDPR "right to erasure" — drops every entry owned by that user |

Entries without a caller (system events like `HEALTH_CHECK`) keep `user_id=None` and `session_id=None`; `purge_user` never touches them.

`TimelineEntry.to_dict()` includes both fields, so `STRUCTURED_LOG` / `JSON` / `WEBHOOK` / `OTLP` transporters all emit per-tenant attribution automatically — no custom transporter code required.

---

## What's Next?

- [CLI Reference](cli.md) -- `--observe` flag and CLI commands
- [Memory](memory.md) -- persistent memory with vector search
