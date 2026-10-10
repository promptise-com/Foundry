"""Pluggable transporter backends for the observability system.

Each transporter receives :class:`TimelineEntry` events in real time from
the :class:`ObservabilityCollector` and delivers them to a specific backend.

Available transporters:

- **HTMLReportTransporter** — self-contained interactive HTML report
- **JSONFileTransporter** — NDJSON streaming + full JSON session dump
- **StructuredLogTransporter** — JSON log lines for ELK / Datadog / Splunk
- **ConsoleTransporter** — Rich-powered real-time terminal output
- **PrometheusTransporter** — Prometheus metrics (counters + histograms)
- **OTLPTransporter** — OpenTelemetry span export via OTLP gRPC
- **WebhookTransporter** — HTTP POST per event (or batched)
- **CallbackTransporter** — invoke a user-provided Python callable

Usage::

    from promptise.observability_transporters import (
        HTMLReportTransporter,
        StructuredLogTransporter,
        ConsoleTransporter,
    )
    from promptise.observability import ObservabilityCollector

    collector = ObservabilityCollector("my-session")
    collector.add_transporter(HTMLReportTransporter(output_dir="./reports"))
    collector.add_transporter(StructuredLogTransporter(log_file="./logs/events.jsonl"))
    collector.add_transporter(ConsoleTransporter(live=True))
"""

from __future__ import annotations

import html
import json
import logging
import os
import re
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("promptise.transporters")


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseTransporter(ABC):
    """Abstract base class for all observability transporters.

    Subclasses must implement:
    - ``on_event(entry)`` — called for each timeline event in real time.
    - ``flush()`` — finalize / export pending data (may be async).

    Optionally override ``close()`` for cleanup.
    """

    @abstractmethod
    def on_event(self, entry: Any) -> None:
        """Process a single :class:`TimelineEntry` event."""
        ...

    @abstractmethod
    def flush(self) -> None:
        """Finalize and export any buffered data."""
        ...

    def close(self) -> None:
        """Release resources.  Called on shutdown."""
        pass


# ---------------------------------------------------------------------------
# 1) HTML Report Transporter
# ---------------------------------------------------------------------------


class HTMLReportTransporter(BaseTransporter):
    """Generates a self-contained interactive HTML report.

    Events are read from the collector when the report is written, so
    nothing is buffered here.  :meth:`flush` (called on agent shutdown)
    writes ``<output_dir>/<session_name>-report-<timestamp>.html``;
    :meth:`write` writes to an exact path, which is what
    :meth:`PromptiseAgent.generate_report` uses.

    The page shows the session's stats (taken from
    :meth:`ObservabilityCollector.get_stats`, so they match
    ``agent.get_stats()``), a filterable timeline, and each event's
    metadata on click.  All data is embedded; the page loads nothing.

    Args:
        output_dir: Directory for the report file.  Defaults to ``"./reports"``.
        session_name: Embedded in the filename.
        title: Page title and heading.
    """

    def __init__(
        self,
        output_dir: str = "./reports",
        session_name: str = "promptise",
        title: str = "Promptise Agent Report",
    ) -> None:
        self.output_dir = output_dir
        self.session_name = session_name
        self.title = title
        self._collector: Any | None = None  # Set externally when auto-created

    def on_event(self, entry: Any) -> None:
        # Events are already stored in the collector; nothing to buffer.
        pass

    def write(self, path: str | os.PathLike[str]) -> Path:
        """Write the report to exactly *path* and return it.

        Creates missing parent directories and overwrites an existing file.

        Raises:
            RuntimeError: When no collector is attached.
            OSError: When the file cannot be written.
        """
        if self._collector is None:
            raise RuntimeError("HTMLReportTransporter has no collector to report on")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        html_text = self._render_html(self._collector.to_json(), self.title)
        target.write_text(html_text, encoding="utf-8")
        logger.info("HTML observability report written: %s", target)
        return target

    def flush(self) -> None:
        """Write a timestamped report into :attr:`output_dir`."""
        if self._collector is None:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            self.write(Path(self.output_dir) / f"{self.session_name}-report-{ts}.html")
        except Exception as exc:
            logger.error("HTMLReportTransporter flush error: %s", exc)

    @staticmethod
    def _render_html(data_json: str, title: str = "Promptise Agent Report") -> str:
        """Render a self-contained HTML report with timeline visualization.

        ``data_json`` is embedded in a ``<script>`` element.  Every ``<``,
        ``>`` and ``&`` in it is written as a JSON unicode escape, so event
        text (tool output, user input) can never close the element or
        inject markup; the page renders all event text with
        ``textContent``.
        """
        safe_json = (
            data_json.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
        )
        values = {
            "__TITLE__": html.escape(title),
            "__TITLE_JSON__": json.dumps(title).replace("<", "\\u003c"),
            "__DATA__": safe_json,
        }
        # One pass, so a placeholder inside a substituted value stays literal.
        return re.sub(
            r"__TITLE_JSON__|__TITLE__|__DATA__",
            lambda m: values[m.group(0)],
            _HTML_REPORT_TEMPLATE,
        )


_HTML_REPORT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }
  h1 { color: #58a6ff; margin-bottom: 8px; }
  .meta { color: #8b949e; margin-bottom: 24px; font-size: 14px; }
  .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
  .stat { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .stat-value { font-size: 28px; font-weight: 700; color: #58a6ff; }
  .stat-label { font-size: 12px; color: #8b949e; margin-top: 4px; }
  .filter { margin-bottom: 16px; display: flex; gap: 8px; flex-wrap: wrap; }
  .filter button { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; padding: 6px 12px; border-radius: 16px; cursor: pointer; font-size: 12px; }
  .filter button.active { background: #1f6feb; border-color: #1f6feb; }
  .event { background: #161b22; border: 1px solid #30363d; border-radius: 6px; margin-bottom: 8px; }
  .event summary { padding: 10px 16px; display: flex; align-items: center; gap: 12px; cursor: pointer; list-style: none; }
  .event summary::-webkit-details-marker { display: none; }
  .event-icon { font-size: 18px; min-width: 24px; text-align: center; }
  .event-type { font-weight: 600; color: #f0f6fc; min-width: 120px; font-family: monospace; }
  .event-desc { flex: 1; color: #8b949e; font-size: 13px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .event-dur { color: #8b949e; font-size: 12px; font-family: monospace; min-width: 70px; text-align: right; }
  .event-time { color: #6e7681; font-size: 12px; font-family: monospace; }
  .event pre { margin: 0 16px 12px 52px; padding: 10px; background: #0d1117; border-radius: 6px; font-size: 12px; white-space: pre-wrap; word-break: break-word; }
  .tool { color: #d2a8ff; } .llm { color: #7ee787; } .error { color: #f85149; } .cache { color: #ffa657; } .agent { color: #79c0ff; }
</style>
</head>
<body>
<h1 id="title"></h1>
<p class="meta" id="meta"></p>
<div class="stats" id="stats"></div>
<div class="filter" id="filter"></div>
<div id="timeline"></div>
<script>
const title = __TITLE_JSON__;
const data = __DATA__;
const entries = data.entries || [];
const stats = data.stats || {};
const icons = {
  'agent.input': '▶️', 'agent.output': '⏹️', 'agent.error': '❌',
  'llm.start': '🧠', 'llm.end': '💬', 'llm.error': '💥', 'llm.retry': '🔄',
  'tool.call': '🔧', 'tool.result': '✅', 'tool.error': '❌',
  'cache.hit': '💨', 'cache.miss': '🔍', 'cache.store': '💾', 'cache.error': '⚠️'
};
function category(type) {
  if (type.endsWith('.error') || type.endsWith('.failed') || type.endsWith('.timeout')) return 'error';
  const prefix = type.split('.')[0];
  return ['agent', 'llm', 'tool', 'cache'].includes(prefix) ? prefix : 'other';
}
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

document.title = title;
document.getElementById('title').textContent = title;
const started = entries.length ? new Date(entries[0].timestamp * 1000).toLocaleString() : '';
document.getElementById('meta').textContent =
  'Session ' + (data.session_name || '') + (started ? ' · started ' + started : '') +
  ' · generated by Promptise Observability';

const byType = stats.events_by_type || {};
const cards = [
  [entries.length, 'Total Events'],
  [stats.total_tokens || 0, 'Total Tokens'],
  [stats.llm_call_count || 0, 'LLM Calls'],
  [stats.tool_call_count || 0, 'Tool Calls'],
  [stats.error_count || 0, 'Errors'],
  [byType['cache.hit'] || 0, 'Cache Hits']
];
const statsEl = document.getElementById('stats');
cards.forEach(([value, label]) => {
  const card = el('div', 'stat');
  card.appendChild(el('div', 'stat-value', Number(value).toLocaleString()));
  card.appendChild(el('div', 'stat-label', label));
  statsEl.appendChild(card);
});

const filterEl = document.getElementById('filter');
let activeFilter = 'all';
const filters = [['all', 'All'], ['agent', 'Agent'], ['llm', 'LLM'], ['tool', 'Tool'], ['error', 'Error'], ['cache', 'Cache']];
filters.forEach(([key, label]) => {
  const btn = el('button', key === 'all' ? 'active' : '', label);
  btn.dataset.filter = key;
  btn.onclick = () => {
    activeFilter = key;
    filterEl.querySelectorAll('button').forEach(b => b.className = b === btn ? 'active' : '');
    render();
  };
  filterEl.appendChild(btn);
});

const timeline = document.getElementById('timeline');
function render() {
  timeline.replaceChildren();
  entries.forEach(e => {
    const cat = category(e.event_type);
    if (activeFilter !== 'all' && cat !== activeFilter) return;
    const row = el('details', 'event');
    row.dataset.category = cat;
    const summary = el('summary');
    summary.appendChild(el('span', 'event-icon', icons[e.event_type] || '📌'));
    summary.appendChild(el('span', 'event-type ' + cat, e.event_type));
    summary.appendChild(el('span', 'event-desc', e.details || ''));
    const ms = e.duration != null ? e.duration * 1000 : (e.metadata || {}).latency_ms;
    summary.appendChild(el('span', 'event-dur', ms != null ? Math.round(ms) + ' ms' : ''));
    summary.appendChild(el('span', 'event-time', e.timestamp ? new Date(e.timestamp * 1000).toLocaleTimeString() : ''));
    row.appendChild(summary);
    const extra = Object.assign({}, e.metadata || {});
    ['user_id', 'session_id', 'agent_id', 'parent_id'].forEach(k => { if (e[k]) extra[k] = e[k]; });
    row.appendChild(el('pre', '', JSON.stringify(extra, null, 2)));
    timeline.appendChild(row);
  });
}
render();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# 2) JSON File Transporter
# ---------------------------------------------------------------------------


class JSONFileTransporter(BaseTransporter):
    """Writes NDJSON lines (one per event) and a full session dump on flush.

    Supports two output modes:
    - **Streaming**: Each event is appended as a single JSON line to
      ``<output_dir>/<session_name>-events.ndjson``.
    - **Dump**: On ``flush()``, a full session JSON is written to a separate,
      timestamped file.

    The NDJSON file is opened in append mode, so events from every run with
    the same ``session_name`` accumulate in it.

    Args:
        output_dir: Directory for output files.  Defaults to ``"./reports"``.
        session_name: Base name for output files.
        stream: If True, write NDJSON lines in real time.  Default: True.
    """

    def __init__(
        self,
        output_dir: str = "./reports",
        session_name: str = "promptise",
        stream: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.session_name = session_name
        self.stream = stream
        self._collector: Any | None = None
        self._stream_file: Any | None = None
        self._lock = threading.Lock()

        if self.stream:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            self._stream_path = os.path.join(self.output_dir, f"{session_name}-events.ndjson")
            # Open with explicit close in close() — can't use context manager
            # because the file must stay open across on_event calls.
            self._stream_file = open(  # noqa: SIM115
                self._stream_path, "a", encoding="utf-8"
            )

    def on_event(self, entry: Any) -> None:
        if self.stream and self._stream_file is not None:
            line = json.dumps(entry.to_dict(), default=str)
            with self._lock:
                if self._stream_file is None:
                    return
                self._stream_file.write(line + "\n")
                self._stream_file.flush()

    def flush(self) -> None:
        """Write full session JSON dump."""
        if self._collector is None:
            return
        try:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            dump_path = os.path.join(self.output_dir, f"{self.session_name}-session-{ts}.json")
            with open(dump_path, "w", encoding="utf-8") as f:
                json.dump(self._collector.to_dict(), f, indent=2, default=str)
            logger.info("JSON session dump: %s", dump_path)
        except Exception as exc:
            logger.error("JSONFileTransporter flush error: %s", exc)

    def close(self) -> None:
        if self._stream_file is not None:
            with self._lock:
                self._stream_file.close()
                self._stream_file = None


# ---------------------------------------------------------------------------
# 3) Structured Log Transporter
# ---------------------------------------------------------------------------


class StructuredLogTransporter(BaseTransporter):
    """Enterprise-grade structured logging — one JSON line per event.

    Each event becomes a log line like::

        {"timestamp":"2026-02-20T19:47:31Z", "level":"INFO", "service":"promptise",
         "session":"my-run", "agent_id":"code-reviewer", "event_type":"llm.end",
         "duration_ms":1250, "tokens":450, "message":"LLM call completed", ...}

    Compatible with ELK stack, Datadog, Splunk, CloudWatch, and any JSON
    log ingestion pipeline.

    Args:
        log_file: File path for structured logs.  If ``None``, writes to stdout.
        session_name: Embedded in each log line.
        service_name: Service identifier for log aggregation.
        correlation_id: Optional trace/request correlation ID.
    """

    def __init__(
        self,
        log_file: str | None = None,
        session_name: str = "promptise",
        service_name: str = "promptise",
        correlation_id: str | None = None,
    ) -> None:
        self.session_name = session_name
        self.service_name = service_name
        self.correlation_id = correlation_id
        self._lock = threading.Lock()

        self._logger = logging.getLogger(f"promptise.structured.{session_name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.propagate = False

        # Remove existing handlers to avoid duplicates
        self._logger.handlers.clear()

        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            handler: logging.Handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        else:
            handler = logging.StreamHandler(sys.stdout)

        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        self._handler = handler

    def on_event(self, entry: Any) -> None:
        """Format entry as structured JSON log line."""
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(entry.timestamp, tz=timezone.utc).isoformat(),
            "level": self._level_for(entry.event_type.value),
            "service": self.service_name,
            "session": self.session_name,
            "entry_id": entry.entry_id,
            "event_type": entry.event_type.value,
            "category": entry.category.value,
            "message": entry.details,
        }

        if entry.agent_id:
            log_entry["agent_id"] = entry.agent_id
        if entry.phase:
            log_entry["phase"] = entry.phase
        if entry.duration is not None:
            log_entry["duration_ms"] = round(entry.duration * 1000, 1)
        if entry.parent_id:
            log_entry["parent_id"] = entry.parent_id
        if self.correlation_id:
            log_entry["correlation_id"] = self.correlation_id

        # Promote key metadata fields to top level
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "latency_ms",
            "model",
            "tool_name",
            "error",
            "error_type",
        ):
            if key in entry.metadata:
                log_entry[key] = entry.metadata[key]

        # Include remaining metadata under "metadata" key
        remaining = {k: v for k, v in entry.metadata.items() if k not in log_entry}
        if remaining:
            log_entry["metadata"] = remaining

        with self._lock:
            self._logger.info(json.dumps(log_entry, default=str))

    def flush(self) -> None:
        if self._handler:
            self._handler.flush()

    def close(self) -> None:
        if self._handler:
            self._handler.close()

    @staticmethod
    def _level_for(event_type: str) -> str:
        """Map event type to log level."""
        if "error" in event_type or "failed" in event_type:
            return "ERROR"
        if "retry" in event_type:
            return "WARN"
        if event_type.startswith("session."):
            return "INFO"
        return "INFO"


# ---------------------------------------------------------------------------
# 4) Console Transporter
# ---------------------------------------------------------------------------


class ConsoleTransporter(BaseTransporter):
    """Real-time color-coded console output.

    Uses Rich if available for beautiful formatting, otherwise falls back
    to plain ``print()``.

    Color coding:
    - **Green**: LLM events (start, end, stream)
    - **Yellow**: Tool events (call, result)
    - **Red**: Errors (LLM error, tool error, task failed)
    - **Blue**: Agent events (input, output)
    - **Magenta**: Phase / orchestration events
    - **Cyan**: Session events

    Args:
        live: If True, enable real-time console output for each event.
            Default: True.
        verbose: If True, include metadata in output.  Default: False.
    """

    _COLOR_MAP: dict[str, str] = {
        "llm": "green",
        "tool": "yellow",
        "agent.input": "blue",
        "agent.output": "blue",
        "agent": "white",
        "task": "white",
        "phase": "magenta",
        "session": "cyan",
        "auth": "red",
        "rbac": "red",
        "health": "dim",
        "circuit_breaker": "dim",
    }

    _ICON_MAP: dict[str, str] = {
        "llm.start": "🧠",
        "llm.end": "✅",
        "llm.error": "💥",
        "llm.retry": "🔄",
        "llm.stream_chunk": "💬",
        "llm.turn": "🤖",
        "tool.call": "🔧",
        "tool.result": "📦",
        "tool.error": "❌",
        "agent.input": "📥",
        "agent.output": "📤",
        "phase.start": "🚀",
        "phase.end": "🏁",
        "session.start": "▶️",
        "session.end": "⏹️",
        "task.started": "📋",
        "task.completed": "✔️",
        "task.failed": "💀",
    }

    def __init__(self, live: bool = True, verbose: bool = False) -> None:
        self.live = live
        self.verbose = verbose
        self._has_rich = False
        self._console: Any = None
        self._event_count = 0
        self._total_tokens = 0

        try:
            from rich.console import Console

            self._console = Console(stderr=True)
            self._has_rich = True
        except ImportError:
            pass

    def on_event(self, entry: Any) -> None:
        if not self.live:
            return

        self._event_count += 1

        # Track running totals for display
        tokens = entry.metadata.get("total_tokens", 0)
        self._total_tokens += tokens

        etype = entry.event_type.value
        icon = self._ICON_MAP.get(etype, "•")
        ts = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S.%f")[:-3]
        agent = f" [{entry.agent_id}]" if entry.agent_id else ""
        duration_str = f" ({entry.duration * 1000:.0f}ms)" if entry.duration else ""
        latency = entry.metadata.get("latency_ms")
        latency_str = f" ({latency:.0f}ms)" if latency else ""

        # Build message
        msg = f"{icon} {ts}{agent} {entry.details}{duration_str}{latency_str}"

        # Add token info for LLM events
        if tokens:
            msg += f"  [{tokens} tok]"

        if self._has_rich and self._console:
            color = self._get_color(etype)
            self._console.print(f"[{color}]{msg}[/{color}]")
        else:
            print(msg)

        if self.verbose and entry.metadata:
            # Print key metadata
            meta_keys = ["model", "tool_name", "error", "prompt_tokens", "completion_tokens"]
            meta_parts = []
            for k in meta_keys:
                if k in entry.metadata:
                    meta_parts.append(f"{k}={entry.metadata[k]}")
            if meta_parts:
                meta_str = "    " + ", ".join(meta_parts)
                if self._has_rich and self._console:
                    self._console.print(f"[dim]{meta_str}[/dim]")
                else:
                    print(meta_str)

    def flush(self) -> None:
        """Print session summary."""
        if not self.live:
            return
        summary = f"\n📊 Session Summary: {self._event_count} events, {self._total_tokens} tokens"
        if self._has_rich and self._console:
            self._console.print(f"[bold cyan]{summary}[/bold cyan]")
        else:
            print(summary)

    def _get_color(self, event_type: str) -> str:
        """Get Rich color for an event type."""
        if event_type in self._COLOR_MAP:
            return self._COLOR_MAP[event_type]
        prefix = event_type.split(".")[0]
        return self._COLOR_MAP.get(prefix, "white")


# ---------------------------------------------------------------------------
# 5) Prometheus Transporter
# ---------------------------------------------------------------------------


class PrometheusTransporter(BaseTransporter):
    """Bridges timeline events to Prometheus metrics.

    Auto-creates and increments counters and histograms:

    - ``promptise_llm_calls_total`` (counter: agent_id, model)
    - ``promptise_llm_tokens_total`` (counter: agent_id, token_type)
    - ``promptise_llm_duration_seconds`` (histogram: agent_id, model)
    - ``promptise_tool_calls_total`` (counter: agent_id, tool_name)
    - ``promptise_tool_duration_seconds`` (histogram: agent_id, tool_name)
    - ``promptise_tool_errors_total`` (counter: agent_id, tool_name)
    - ``promptise_events_total`` (counter: event_type, category)

    If the ``prometheus_client`` package is not installed, metrics are
    tracked internally as plain Python dicts (accessible via :attr:`metrics`).

    Args:
        port: Port for the Prometheus ``/metrics`` HTTP endpoint.
            If set to 0, no HTTP server is started.  Default: 0 (no server).
    """

    def __init__(self, port: int = 0) -> None:
        self.port = port
        self._has_prometheus = False
        self.metrics: dict[str, Any] = {
            "llm_calls_total": {},
            "llm_tokens_total": {},
            "llm_duration_seconds": [],
            "tool_calls_total": {},
            "tool_errors_total": {},
            "tool_duration_seconds": [],
            "events_total": {},
        }
        self._lock = threading.Lock()

        try:
            from prometheus_client import Counter, Histogram, start_http_server  # type: ignore

            self._has_prometheus = True
            self._prom_llm_calls = Counter(
                "promptise_llm_calls_total",
                "Total LLM calls",
                ["agent_id", "model"],
            )
            self._prom_llm_tokens = Counter(
                "promptise_llm_tokens_total",
                "Total tokens used",
                ["agent_id", "token_type"],
            )
            self._prom_llm_duration = Histogram(
                "promptise_llm_duration_seconds",
                "LLM call duration in seconds",
                ["agent_id", "model"],
            )
            self._prom_tool_calls = Counter(
                "promptise_tool_calls_total",
                "Total tool calls",
                ["agent_id", "tool_name"],
            )
            self._prom_tool_errors = Counter(
                "promptise_tool_errors_total",
                "Total tool errors",
                ["agent_id", "tool_name"],
            )
            self._prom_tool_duration = Histogram(
                "promptise_tool_duration_seconds",
                "Tool call duration in seconds",
                ["agent_id", "tool_name"],
            )
            self._prom_events = Counter(
                "promptise_events_total",
                "Total observability events",
                ["event_type", "category"],
            )

            if port > 0:
                start_http_server(port)
                logger.info("Prometheus metrics server started on port %d", port)

        except ImportError:
            logger.warning(
                "prometheus_client not installed — PrometheusTransporter will "
                "track metrics internally only (not exported to Prometheus). "
                "Install with: pip install prometheus_client"
            )

    def on_event(self, entry: Any) -> None:
        etype = entry.event_type.value
        cat = entry.category.value
        agent = entry.agent_id or "unknown"

        # Always track event counts
        with self._lock:
            key = f"{etype}:{cat}"
            self.metrics["events_total"][key] = self.metrics["events_total"].get(key, 0) + 1

        if self._has_prometheus:
            self._prom_events.labels(event_type=etype, category=cat).inc()

        # LLM end events
        if etype in ("llm.end", "llm.turn"):
            model = entry.metadata.get("model", "unknown")
            tokens_prompt = entry.metadata.get("prompt_tokens", 0)
            tokens_completion = entry.metadata.get("completion_tokens", 0)
            latency = entry.metadata.get("latency_ms")

            with self._lock:
                mk = f"{agent}:{model}"
                self.metrics["llm_calls_total"][mk] = self.metrics["llm_calls_total"].get(mk, 0) + 1
                self.metrics["llm_tokens_total"][f"{agent}:prompt"] = (
                    self.metrics["llm_tokens_total"].get(f"{agent}:prompt", 0) + tokens_prompt
                )
                self.metrics["llm_tokens_total"][f"{agent}:completion"] = (
                    self.metrics["llm_tokens_total"].get(f"{agent}:completion", 0)
                    + tokens_completion
                )
                if latency is not None:
                    self.metrics["llm_duration_seconds"].append(latency / 1000.0)

            if self._has_prometheus:
                self._prom_llm_calls.labels(agent_id=agent, model=model).inc()
                self._prom_llm_tokens.labels(agent_id=agent, token_type="prompt").inc(tokens_prompt)
                self._prom_llm_tokens.labels(agent_id=agent, token_type="completion").inc(
                    tokens_completion
                )
                if latency is not None:
                    self._prom_llm_duration.labels(agent_id=agent, model=model).observe(
                        latency / 1000.0
                    )

        # Tool events
        elif etype == "tool.call":
            tool_name = entry.metadata.get("tool_name", "unknown")
            with self._lock:
                tk = f"{agent}:{tool_name}"
                self.metrics["tool_calls_total"][tk] = (
                    self.metrics["tool_calls_total"].get(tk, 0) + 1
                )

            if self._has_prometheus:
                self._prom_tool_calls.labels(agent_id=agent, tool_name=tool_name).inc()

        elif etype == "tool.result":
            tool_name = entry.metadata.get("tool_name", "unknown")
            latency = entry.metadata.get("latency_ms")
            if latency is not None:
                with self._lock:
                    self.metrics["tool_duration_seconds"].append(latency / 1000.0)
                if self._has_prometheus:
                    self._prom_tool_duration.labels(agent_id=agent, tool_name=tool_name).observe(
                        latency / 1000.0
                    )

        elif etype == "tool.error":
            tool_name = entry.metadata.get("tool_name", "unknown")
            with self._lock:
                tk = f"{agent}:{tool_name}"
                self.metrics["tool_errors_total"][tk] = (
                    self.metrics["tool_errors_total"].get(tk, 0) + 1
                )
            if self._has_prometheus:
                self._prom_tool_errors.labels(agent_id=agent, tool_name=tool_name).inc()

    def flush(self) -> None:
        pass  # Prometheus is push-based; nothing to flush.


# ---------------------------------------------------------------------------
# 6) OTLP Transporter (OpenTelemetry)
# ---------------------------------------------------------------------------


class OTLPTransporter(BaseTransporter):
    """Exports agent runs to OpenTelemetry as traces, via OTLP gRPC.

    Each agent invocation becomes one trace: an ``invoke_agent`` span from
    ``agent.input`` to ``agent.output`` (or ``agent.error``), with a
    ``chat <model>`` child span per LLM call (``llm.start`` → ``llm.end``)
    and an ``execute_tool <name>`` child span per tool call (``tool.call``
    → ``tool.result`` / ``tool.error``).  Spans carry the real start and
    end times of the work, so their duration is the latency.  Other events
    recorded during the run (cache hits, retries, approvals …) become span
    events on the run span; events outside a run become standalone spans.

    If an OpenTelemetry span is active when the agent is invoked (for
    example the server span of an instrumented web framework), the run
    span becomes its child, so the agent run joins the request's trace.

    Attributes follow the OpenTelemetry GenAI semantic conventions where
    one exists (``gen_ai.operation.name``, ``gen_ai.agent.name``,
    ``gen_ai.request.model``, ``gen_ai.usage.input_tokens`` /
    ``output_tokens``, ``gen_ai.tool.name``) plus ``enduser.id`` and
    ``session.id`` from the :class:`~promptise.agent.CallerContext`,
    ``promptise.correlation_id``, and each event's scalar metadata as
    ``promptise.<key>``.  Failed LLM calls, tool calls and runs get an
    ``ERROR`` status.

    Requires the ``[all]`` extra::

        pip install "promptise[all]"

    Compatible with Jaeger, Datadog, Honeycomb, Grafana Tempo, and any
    OTLP-compatible backend.

    Args:
        endpoint: OTLP gRPC endpoint.  Default: ``"http://localhost:4317"``.
        service_name: OpenTelemetry service name.  Default: ``"promptise"``.
        correlation_id: Optional ID added to every span as
            ``promptise.correlation_id``.
        tracer_provider: An existing ``TracerProvider`` to emit spans
            through (for example your application's).  When given,
            ``endpoint`` and ``service_name`` are not used and the provider
            is flushed but not shut down by :meth:`close`.  When omitted, a
            private provider with an OTLP exporter is created; the global
            tracer provider is never replaced.
    """

    #: Upper bound on spans kept open while waiting for their end event.
    _MAX_OPEN_SPANS = 10_000

    def __init__(
        self,
        endpoint: str = "http://localhost:4317",
        service_name: str = "promptise",
        *,
        correlation_id: str | None = None,
        tracer_provider: Any | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.service_name = service_name
        self.correlation_id = correlation_id
        self._tracer: Any = None
        self._provider: Any = None
        self._owns_provider = tracer_provider is None
        self._lock = threading.Lock()
        # Open spans waiting for their end event.
        self._runs: dict[str, Any] = {}  # agent.input entry_id → run span
        self._open: dict[tuple[str, str], Any] = {}  # (kind, key) → LLM/tool span

        try:
            from opentelemetry import trace  # type: ignore  # noqa: F401
            from opentelemetry.sdk.trace import TracerProvider  # type: ignore

            if tracer_provider is None:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # type: ignore
                    OTLPSpanExporter,
                )
                from opentelemetry.sdk.resources import Resource  # type: ignore
                from opentelemetry.sdk.trace.export import (  # type: ignore
                    BatchSpanProcessor,
                )

                resource = Resource.create({"service.name": service_name})
                tracer_provider = TracerProvider(resource=resource)
                tracer_provider.add_span_processor(
                    BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
                )
                logger.info("OTLPTransporter exporting to %s", endpoint)
            self._provider = tracer_provider
            self._tracer = tracer_provider.get_tracer("promptise")

        except ImportError:
            logger.warning(
                'OpenTelemetry packages not installed.  Install with: pip install "promptise[all]"'
            )

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _ns(seconds: float) -> int:
        return int(seconds * 1_000_000_000)

    def _context_for(self, parent_entry_id: str | None) -> Any:
        """Context whose current span is the open run span *parent_entry_id*."""
        if parent_entry_id is None:
            return None
        parent = self._runs.get(parent_entry_id)
        if parent is None:
            return None
        from opentelemetry import trace  # type: ignore

        return trace.set_span_in_context(parent)

    def _attributes(self, entry: Any) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            "promptise.event_type": entry.event_type.value,
            "promptise.category": entry.category.value,
            "promptise.entry_id": entry.entry_id,
        }
        if entry.agent_id:
            attrs["promptise.agent_id"] = entry.agent_id
            attrs["gen_ai.agent.name"] = entry.agent_id
        if entry.phase:
            attrs["promptise.phase"] = entry.phase
        if entry.details:
            attrs["promptise.details"] = entry.details
        if entry.user_id:
            attrs["enduser.id"] = entry.user_id
        if entry.session_id:
            attrs["session.id"] = entry.session_id
        if self.correlation_id:
            attrs["promptise.correlation_id"] = self.correlation_id
        for k, v in entry.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                attrs[f"promptise.{k}"] = v
            elif isinstance(v, (list, tuple)) and v and all(isinstance(i, str) for i in v):
                attrs[f"promptise.{k}"] = list(v)
        return attrs

    def _start(self, name: str, entry: Any, parent_entry_id: str | None, **attrs: Any) -> Any:
        from opentelemetry.trace import SpanKind  # type: ignore

        all_attrs = self._attributes(entry)
        all_attrs.update({k: v for k, v in attrs.items() if v is not None})
        return self._tracer.start_span(
            name,
            context=self._context_for(parent_entry_id),
            kind=SpanKind.INTERNAL,
            attributes=all_attrs,
            start_time=self._ns(entry.timestamp),
        )

    def _finish(self, span: Any, entry: Any, *, error: bool = False, **attrs: Any) -> None:
        for k, v in self._attributes(entry).items():
            if k not in ("promptise.event_type", "promptise.entry_id", "promptise.details"):
                span.set_attribute(k, v)
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(k, v)
        if error:
            from opentelemetry.trace import Status, StatusCode  # type: ignore

            message = entry.metadata.get("error_type") or entry.details or "error"
            span.set_status(Status(StatusCode.ERROR, str(message)))
        span.end(end_time=self._ns(entry.timestamp))

    def _remember(self, store: dict[Any, Any], key: Any, span: Any) -> None:
        if len(store) >= self._MAX_OPEN_SPANS:
            # Drop the oldest open span rather than grow without bound.
            oldest = next(iter(store))
            store.pop(oldest).end()
        store[key] = span

    def _pop_or_backdated(self, kind: str, key: str, name: str, entry: Any, **attrs: Any) -> Any:
        """The open span for *key*, or a new one back-dated by the event's duration."""
        span = self._open.pop((kind, key), None)
        if span is not None:
            return span
        from opentelemetry.trace import SpanKind  # type: ignore

        duration = entry.duration
        if duration is None and entry.metadata.get("latency_ms") is not None:
            duration = float(entry.metadata["latency_ms"]) / 1000.0
        start = entry.timestamp - (duration or 0.0)
        all_attrs = self._attributes(entry)
        all_attrs.update({k: v for k, v in attrs.items() if v is not None})
        return self._tracer.start_span(
            name,
            context=self._context_for(entry.parent_id),
            kind=SpanKind.INTERNAL,
            attributes=all_attrs,
            start_time=self._ns(start),
        )

    # -- event handling ---------------------------------------------------

    def on_event(self, entry: Any) -> None:
        if self._tracer is None:
            return
        with self._lock:
            self._handle(entry)

    def _handle(self, entry: Any) -> None:
        etype = entry.event_type.value
        meta = entry.metadata
        agent = entry.agent_id or "agent"

        if etype == "agent.input":
            span = self._start(
                f"invoke_agent {agent}",
                entry,
                entry.parent_id,
                **{
                    "gen_ai.operation.name": "invoke_agent",
                    "gen_ai.request.model": meta.get("model"),
                },
            )
            self._remember(self._runs, entry.entry_id, span)
            return

        if etype in ("agent.output", "agent.error"):
            span = self._runs.pop(entry.parent_id, None) if entry.parent_id else None
            if span is None:
                span = self._pop_or_backdated("run", entry.entry_id, f"invoke_agent {agent}", entry)
            self._finish(
                span,
                entry,
                error=etype == "agent.error",
                **{
                    "gen_ai.usage.input_tokens": meta.get("prompt_tokens"),
                    "gen_ai.usage.output_tokens": meta.get("completion_tokens"),
                },
            )
            return

        if etype == "llm.start":
            model = meta.get("model")
            span = self._start(
                f"chat {model}" if model else "chat",
                entry,
                entry.parent_id,
                **{"gen_ai.operation.name": "chat", "gen_ai.request.model": model},
            )
            self._remember(self._open, ("llm", str(meta.get("run_id", entry.entry_id))), span)
            return

        if etype in ("llm.end", "llm.error"):
            model = meta.get("model")
            span = self._pop_or_backdated(
                "llm",
                str(meta.get("run_id", entry.entry_id)),
                f"chat {model}" if model else "chat",
                entry,
                **{"gen_ai.operation.name": "chat"},
            )
            self._finish(
                span,
                entry,
                error=etype == "llm.error",
                **{
                    "gen_ai.response.model": model,
                    "gen_ai.usage.input_tokens": meta.get("prompt_tokens"),
                    "gen_ai.usage.output_tokens": meta.get("completion_tokens"),
                },
            )
            return

        if etype == "tool.call":
            tool = meta.get("tool_name", "tool")
            span = self._start(
                f"execute_tool {tool}",
                entry,
                entry.parent_id,
                **{"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool},
            )
            self._remember(self._open, ("tool", str(meta.get("run_id", tool))), span)
            return

        if etype in ("tool.result", "tool.error"):
            tool = meta.get("tool_name", "tool")
            span = self._pop_or_backdated(
                "tool",
                str(meta.get("run_id", tool)),
                f"execute_tool {tool}",
                entry,
                **{"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": tool},
            )
            self._finish(span, entry, error=etype == "tool.error" or meta.get("status") == "error")
            return

        # Anything else: an event on the run it belongs to, or its own span.
        run_span = self._runs.get(entry.parent_id) if entry.parent_id else None
        if run_span is not None:
            run_span.add_event(
                f"promptise.{etype}",
                attributes=self._attributes(entry),
                timestamp=self._ns(entry.timestamp),
            )
            return
        span = self._pop_or_backdated("event", entry.entry_id, f"promptise.{etype}", entry)
        self._finish(span, entry, error=etype.endswith((".error", ".failed")))

    # -- lifecycle --------------------------------------------------------

    def _end_open_spans(self) -> None:
        with self._lock:
            for store in (self._open, self._runs):
                for span in store.values():
                    span.end()
                store.clear()

    def flush(self) -> None:
        """Export finished spans now."""
        if self._provider is not None:
            try:
                self._provider.force_flush()
            except Exception as exc:
                logger.error("OTLPTransporter flush error: %s", exc)

    def close(self) -> None:
        """End spans still open, export them, and shut down the private provider."""
        if self._provider is None:
            return
        self._end_open_spans()
        try:
            if self._owns_provider:
                self._provider.shutdown()
            else:
                self._provider.force_flush()
        except Exception:
            logger.debug("OTLPTransporter shutdown error", exc_info=True)


# ---------------------------------------------------------------------------
# 7) Webhook Transporter
# ---------------------------------------------------------------------------


class WebhookTransporter(BaseTransporter):
    """HTTP POST each event (or batch) to a configurable URL.

    Supports:
    - Single mode (POST per event)
    - Batch mode (buffer N events, flush periodically)
    - Custom HTTP headers (for auth tokens)
    - Retry with exponential backoff
    - Async non-blocking delivery via background thread

    Useful for Slack, Discord, PagerDuty, or custom webhook integrations.

    Args:
        url: Target URL for the webhook.
        headers: Custom HTTP headers (e.g. ``{"Authorization": "Bearer ..."}``)
        batch_size: Buffer events and send in batches.  0 = send per event.
        max_retries: Maximum retry attempts for failed deliveries.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        batch_size: int = 0,
        max_retries: int = 3,
        timeout: float = 10.0,
    ) -> None:
        if not url.startswith(("http://", "https://")):
            raise ValueError(
                f"WebhookTransporter url must start with http:// or https:// (got {url!r})"
            )
        self.url = url
        self.headers = {"Content-Type": "application/json", **(headers or {})}
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.timeout = timeout
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def on_event(self, entry: Any) -> None:
        payload = entry.to_dict()

        if self.batch_size <= 0:
            # Immediate delivery
            self._send([payload])
        else:
            batch_to_send: list[dict[str, Any]] | None = None
            with self._lock:
                self._buffer.append(payload)
                if len(self._buffer) >= self.batch_size:
                    batch_to_send = self._buffer[:]
                    self._buffer.clear()
            if batch_to_send is not None:
                self._send(batch_to_send)

    def flush(self) -> None:
        """Send any buffered events."""
        with self._lock:
            if self._buffer:
                batch = self._buffer[:]
                self._buffer.clear()
            else:
                return
        self._send(batch)

    def _send(self, events: list[dict[str, Any]]) -> None:
        """Send events with retry logic.  Runs in a background thread."""
        thread = threading.Thread(
            target=self._send_sync,
            args=(events,),
            daemon=True,
        )
        thread.start()

    def _send_sync(self, events: list[dict[str, Any]]) -> None:
        """Synchronous send with exponential backoff retry."""
        import urllib.error
        import urllib.request

        payload = json.dumps(
            {"events": events} if len(events) > 1 else events[0],
            default=str,
        ).encode("utf-8")

        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(
                    self.url,
                    data=payload,
                    headers=self.headers,
                    method="POST",
                )
                with urllib.request.urlopen(  # noqa: S310  # nosec B310 - url scheme validated as http/https in __init__
                    req, timeout=self.timeout
                ) as resp:
                    if resp.status < 300:
                        return
                    logger.warning(
                        "Webhook returned %d on attempt %d",
                        resp.status,
                        attempt + 1,
                    )
            except Exception as exc:
                logger.debug("Webhook attempt %d failed: %s", attempt + 1, exc)
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 5))

        logger.error(
            "Webhook delivery failed after %d attempts to %s",
            self.max_retries + 1,
            self.url,
        )


# ---------------------------------------------------------------------------
# 8) Callback Transporter
# ---------------------------------------------------------------------------


class CallbackTransporter(BaseTransporter):
    """Invoke a user-provided Python callable for each event.

    The simplest extensibility point — users can do anything they want
    with each event.

    Args:
        callback: A callable that receives a single :class:`TimelineEntry`.
    """

    def __init__(self, callback: Callable[..., Any]) -> None:
        if not callable(callback):
            raise TypeError(f"callback must be callable, got {type(callback).__name__}")
        self._callback = callback

    def on_event(self, entry: Any) -> None:
        try:
            self._callback(entry)
        except Exception as exc:
            logger.error("CallbackTransporter error: %s", exc)

    def flush(self) -> None:
        pass  # Nothing to flush — events are delivered immediately.


# ---------------------------------------------------------------------------
# Factory: create transporters from ObservabilityConfig
# ---------------------------------------------------------------------------


def create_transporters(
    config: Any,  # ObservabilityConfig
    collector: Any,  # ObservabilityCollector
) -> list[BaseTransporter]:
    """Create and register transporter instances from an ObservabilityConfig.

    This is called automatically by ``build_agent()`` when the
    ``observe`` parameter is enabled.

    Args:
        config: An :class:`ObservabilityConfig` instance.
        collector: The :class:`ObservabilityCollector` to register with.

    Returns:
        List of created transporter instances.
    """
    from .observability_config import TransporterType

    transporters: list[BaseTransporter] = []

    for t_type in config.transporters:
        try:
            t: BaseTransporter | None = None

            if t_type == TransporterType.HTML:
                t = HTMLReportTransporter(
                    output_dir=config.output_dir or "./reports",
                    session_name=config.session_name,
                )
                t._collector = collector  # type: ignore[attr-defined]

            elif t_type == TransporterType.JSON:
                t = JSONFileTransporter(
                    output_dir=config.output_dir or "./reports",
                    session_name=config.session_name,
                )
                t._collector = collector  # type: ignore[attr-defined]

            elif t_type == TransporterType.STRUCTURED_LOG:
                t = StructuredLogTransporter(
                    log_file=config.log_file,
                    session_name=config.session_name,
                    correlation_id=config.correlation_id,
                )

            elif t_type == TransporterType.CONSOLE:
                t = ConsoleTransporter(
                    live=config.console_live,
                    verbose=(config.level.value == "full"),
                )

            elif t_type == TransporterType.PROMETHEUS:
                t = PrometheusTransporter(port=config.prometheus_port)

            elif t_type == TransporterType.OTLP:
                t = OTLPTransporter(
                    endpoint=config.otlp_endpoint,
                    service_name=config.session_name,
                    correlation_id=config.correlation_id,
                )

            elif t_type == TransporterType.WEBHOOK:
                if config.webhook_url:
                    t = WebhookTransporter(
                        url=config.webhook_url,
                        headers=config.webhook_headers,
                    )
                else:
                    logger.warning("WebhookTransporter requested but no webhook_url set.")

            elif t_type == TransporterType.CALLBACK:
                if config.on_event:
                    t = CallbackTransporter(callback=config.on_event)
                else:
                    logger.warning("CallbackTransporter requested but no on_event callable set.")

            if t is not None:
                collector.add_transporter(t)
                transporters.append(t)

        except Exception as exc:
            logger.error("Failed to create transporter %s: %s", t_type.value, exc)

    return transporters
