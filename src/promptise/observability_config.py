"""Configuration for Promptise's plug-and-play observability system.

Provides :class:`ObservabilityConfig` — the single dataclass that controls
what gets observed, how much detail is captured, and where events are sent.

Usage::

    from promptise import build_agent, ObservabilityConfig, ObserveLevel

    # Minimal — just turn it on (defaults to STANDARD level + HTML report)
    agent = await build_agent(servers=..., model=..., observe=True)

    # Full enterprise configuration
    config = ObservabilityConfig(
        level=ObserveLevel.FULL,
        session_name="production-audit",
        record_prompts=True,
        transporters=[TransporterType.HTML, TransporterType.STRUCTURED_LOG, TransporterType.CONSOLE],
        output_dir="./reports",
        log_file="./logs/agent.jsonl",
        console_live=True,
    )
    agent = await build_agent(servers=..., model=..., observe=config)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ObserveLevel(str, Enum):
    """Controls how much detail the observability system captures.

    - ``OFF``: nothing.  The agent still has a (permanently empty)
      ``collector``, so ``agent.get_stats()`` and timeline queries keep
      working, but no events are recorded and no transporters are created.
    - ``BASIC``: ``agent.input`` / ``agent.output`` / ``agent.error`` per
      invocation (``agent.output`` carries the run's token totals), tool
      calls and results, LLM and tool errors, cache events.
    - ``STANDARD``: BASIC plus ``llm.start`` / ``llm.end`` for every LLM
      turn, with token usage, latency and the tools the model asked for.
    - ``FULL``: STANDARD plus prompt, response, agent input and output text
      (unless ``record_prompts=False``) and streamed-token counts.

    Tool arguments and results are controlled separately by
    :attr:`ObservabilityConfig.record_tool_io`.
    """

    OFF = "off"
    """Observability disabled: nothing is recorded."""

    BASIC = "basic"
    """Agent input/output, tool calls, errors and cache events; no per-LLM-turn events."""

    STANDARD = "standard"
    """Everything in BASIC plus every LLM turn with token usage and latency."""

    FULL = "full"
    """Everything in STANDARD plus prompt/response content and streaming
    tokens."""


class TransporterType(str, Enum):
    """Available backends for receiving observability events."""

    HTML = "html"
    """Self-contained interactive HTML report (default)."""

    JSON = "json"
    """JSON file export (full session dump + NDJSON streaming)."""

    STRUCTURED_LOG = "log"
    """JSON log lines, one per event.  Compatible with ELK, Datadog,
    Splunk, CloudWatch, and other enterprise logging pipelines."""

    CONSOLE = "console"
    """Real-time Rich console output with color-coded events."""

    PROMETHEUS = "prometheus"
    """Prometheus metrics (counters, histograms) for Grafana dashboards."""

    OTLP = "otlp"
    """OpenTelemetry span export via OTLP gRPC.  Requires the
    ``[all]`` extra: ``pip install "promptise[all]"``."""

    WEBHOOK = "webhook"
    """HTTP POST each event (or batch) to a configurable URL."""

    CALLBACK = "callback"
    """Invoke a user-provided Python callable for each event."""


# Backward-compatible alias
ExportFormat = TransporterType


@dataclass
class ObservabilityConfig:
    """Configuration for the observability system.

    Pass as ``observe=config`` to :func:`build_agent` or use the
    shorthand ``observe=True`` for sensible defaults.

    Examples::

        # Defaults: STANDARD level, HTML transporter
        ObservabilityConfig()

        # Enterprise: full detail, multiple transporters
        ObservabilityConfig(
            level=ObserveLevel.FULL,
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
            correlation_id="req-abc-123",
        )
    """

    # --- Capture level -------------------------------------------------------

    level: ObserveLevel = ObserveLevel.STANDARD
    """How much detail to capture."""

    session_name: str = "promptise"
    """Human-readable session identifier embedded in reports/logs."""

    record_prompts: bool | None = None
    """Whether to store prompt, response, agent input and output text
    (truncated to 2,000 characters) in event metadata.

    ``None`` (default) follows :attr:`level`: on at ``FULL``, off below it.
    ``True`` records text at any level; ``False`` never records it, even at
    ``FULL``."""

    record_tool_io: bool = True
    """Whether to store tool arguments (``tool.call``) and result previews
    (``tool.result``) in event metadata, truncated to 2,000 characters.

    On by default — arguments and results are what make a trace debuggable.
    Tool arguments often carry user data (names, emails, account numbers),
    so set ``False`` when traces leave your trust boundary: events then
    keep the tool name, latency and error status, plus ``arguments_length``
    and ``result_length``.  Independent of :attr:`record_prompts`."""

    redact_sensitive: bool = True
    """Replace credentials and common PII in every recorded event with
    placeholders before it is stored or exported (see
    :func:`promptise.observability.redact_sensitive`): API keys, AWS and
    GitHub tokens, ``Bearer`` tokens, passwords in URLs, card numbers, US
    social security numbers and email addresses become ``[API_KEY]``,
    ``Bearer [REDACTED]``, ``[EMAIL]`` and so on.  Applies to the collector
    :func:`~promptise.agent.build_agent` creates; a collector passed as
    ``observer=`` keeps its own ``sanitizer``.  On by default; set
    ``False`` only when traces stay inside your trust boundary."""

    max_entries: int = 100_000
    """Maximum timeline entries before oldest are evicted (ring buffer)."""

    # --- Transporters --------------------------------------------------------

    transporters: list[TransporterType] = field(
        default_factory=lambda: [TransporterType.HTML],
    )
    """Which transporter backends receive events."""

    # --- Transporter-specific config -----------------------------------------

    output_dir: str | None = None
    """Directory for HTML and JSON output files."""

    log_file: str | None = None
    """File path for the STRUCTURED_LOG transporter."""

    console_live: bool = False
    """When True with CONSOLE transporter, start a background thread that
    prints events in real-time."""

    webhook_url: str | None = None
    """Target URL for the WEBHOOK transporter."""

    webhook_headers: dict[str, str] = field(default_factory=dict)
    """Custom HTTP headers for the WEBHOOK transporter (e.g. auth tokens)."""

    otlp_endpoint: str = "http://localhost:4317"
    """gRPC endpoint for the OTLP transporter."""

    prometheus_port: int = 9090
    """Port for the Prometheus metrics endpoint."""

    on_event: Callable[..., Any] | None = None
    """User callback for the CALLBACK transporter.  Receives a single
    :class:`TimelineEntry` argument."""

    # --- Correlation ---------------------------------------------------------

    correlation_id: str | None = None
    """Optional correlation ID that ties all events to an external
    request or trace."""
