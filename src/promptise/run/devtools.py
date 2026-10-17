"""``--devtools``: a live trace of every agent on stderr, and a summary at the end.

The program's own output stays on stdout, untouched.  The trace shows each
agent as it is built, every LLM turn (model, latency, tokens, cost, the tools
it asked for), every tool call with its arguments and result, errors and
notable agent events.  When the run ends, a summary table lists the totals per
agent, model and tool.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import IO, Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .events import EventBus, RunEvent
from .pricing import PriceTable

__all__ = ["DevTracer", "format_cost", "print_summary", "write_report"]

_QUIET_EVENTS = {"invocation.start", "invocation.complete", "tool.start", "tool.end"}


def format_cost(cost: float | None) -> str:
    """``$0.0123``, or ``—`` when there is no estimate."""
    if cost is None:
        return "—"
    if cost == 0:
        return "$0"
    return f"${cost:.4f}" if cost < 1 else f"${cost:.2f}"


def _short(value: Any, limit: int = 120) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class DevTracer:
    """Prints run events as a readable live trace (a bus subscriber).

    Args:
        prices: For the cost of each LLM turn.
        file: Where to print (default: stderr).
        verbose: Also print ``llm.start`` and every agent event.
    """

    def __init__(
        self,
        prices: PriceTable | None = None,
        *,
        file: IO[str] | None = None,
        verbose: bool = False,
    ) -> None:
        self.console = Console(file=file, stderr=file is None, highlight=False, soft_wrap=True)
        self.prices = prices or PriceTable()
        self.verbose = verbose
        self._lock = threading.Lock()

    def attach(self, bus: EventBus) -> None:
        bus.subscribe(self)

    def _print(self, text: str) -> None:
        with self._lock:
            self.console.print(text)

    def __call__(self, event: RunEvent) -> None:
        kind = event.get("kind")
        try:
            if kind == "agent.built":
                self._agent_built(event)
            elif kind == "timeline":
                self._timeline(event)
            elif kind == "event":
                self._agent_event(event)
            elif kind == "agent.closed":
                self._print(f"[dim]■ {escape(str(event.get('agent')))} shut down[/dim]")
        except Exception:  # a display problem must never break the program
            pass

    def _agent_built(self, event: RunEvent) -> None:
        tools = event.get("tools") or []
        servers = event.get("servers") or []
        model = event.get("model") or "?"
        features = [k for k, v in (event.get("features") or {}).items() if v]
        extra = f"  [dim]{', '.join(features)}[/dim]" if features else ""
        self._print(
            f"[bold cyan]● {escape(str(event.get('agent')))}[/bold cyan]  "
            f"model [bold]{escape(str(model))}[/bold]  "
            f"{len(tools)} tool(s)  {len(servers)} MCP server(s)  "
            f"observe {escape(str(event.get('observe_level')))}{extra}"
        )
        for server in servers:
            self._print(
                f"  [dim]⇄ {escape(str(server.get('name')))} ({escape(str(server.get('transport')))}) "
                f"{escape(_short(server.get('target', ''), 80))} · {server.get('tools', 0)} tool(s)[/dim]"
            )

    def _timeline(self, event: RunEvent) -> None:
        agent = escape(str(event.get("agent")))
        entry = event.get("entry") or {}
        etype = entry.get("event_type")
        meta = entry.get("metadata") or {}
        duration = entry.get("duration")
        took = f"{duration:.2f}s" if isinstance(duration, (int, float)) else ""
        if etype == "agent.input":
            preview = meta.get("input_preview") or meta.get("input") or ""
            line = f"[bold]▸ {agent}[/bold]"
            if preview:
                line += f"  [italic]{escape(_short(preview))}[/italic]"
            self._print(line)
        elif etype == "llm.start" and self.verbose:
            self._print(f"  [dim]… {agent} thinking[/dim]")
        elif etype == "llm.end":
            model = meta.get("model") or ""
            pt, ct = int(meta.get("prompt_tokens") or 0), int(meta.get("completion_tokens") or 0)
            cost = self.prices.cost(model, pt, ct)
            asked = meta.get("tool_calls") or []
            wants = f"  → {escape(', '.join(map(str, asked)))}" if asked else ""
            cost_text = f"  {format_cost(cost)}" if cost is not None else ""
            self._print(
                f"  [magenta]◆ llm[/magenta] {escape(str(model))}  {took}  "
                f"[dim]in[/dim] {pt} [dim]out[/dim] {ct} tok{cost_text}{wants}"
            )
        elif etype == "tool.call":
            name = escape(str(meta.get("tool_name") or "?"))
            args = meta.get("arguments")
            shown = f"({escape(_short(args, 160))})" if args not in (None, "") else ""
            reused = " [dim](reused)[/dim]" if meta.get("reused") else ""
            self._print(f"  [yellow]⚙ {name}[/yellow]{shown}{reused}")
        elif etype == "tool.result":
            name = escape(str(meta.get("tool_name") or "?"))
            latency = meta.get("latency_ms")
            took = f"{latency:.0f}ms" if isinstance(latency, (int, float)) else took
            preview = meta.get("result_preview")
            shown = f"  → {escape(_short(preview, 160))}" if preview else ""
            if meta.get("status") == "error":
                self._print(f"  [red]✗ {name}[/red] {took}{shown}")
            else:
                self._print(f"  [green]✓ {name}[/green] [dim]{took}[/dim]{shown}")
        elif etype == "tool.error":
            name = escape(str(meta.get("tool_name") or "?"))
            self._print(
                f"  [red]✗ {name} failed[/red] {escape(_short(meta.get('error') or entry.get('details') or ''))}"
            )
        elif etype in ("llm.error", "agent.error"):
            detail = meta.get("error") or entry.get("details") or ""
            self._print(f"[bold red]✗ {agent} {etype}[/bold red] {escape(_short(detail, 300))}")
        elif etype == "llm.retry":
            self._print(
                f"  [yellow]↻ llm retry[/yellow] {escape(_short(entry.get('details') or ''))}"
            )
        elif etype == "agent.output":
            tokens = meta.get("total_tokens")
            calls = meta.get("llm_call_count")
            tools = meta.get("tool_call_count")
            parts = [took]
            if calls is not None:
                parts.append(f"{calls} llm call(s)")
            if tools:
                parts.append(f"{tools} tool call(s)")
            if tokens:
                parts.append(f"{tokens} tok")
            self._print(
                f"[bold]◂ {agent}[/bold] [dim]done · {' · '.join(p for p in parts if p)}[/dim]"
            )
        elif etype in ("cache.hit",):
            self._print(
                f"  [blue]⚡ cache hit[/blue] [dim]{escape(_short(entry.get('details') or ''))}[/dim]"
            )
        elif etype and etype.startswith("approval."):
            self._print(
                f"  [yellow]✋ {escape(etype)}[/yellow] {escape(_short(entry.get('details') or ''))}"
            )

    def _agent_event(self, event: RunEvent) -> None:
        data = event.get("event") or {}
        etype = str(data.get("event_type") or "")
        if etype in _QUIET_EVENTS and not self.verbose:
            return
        severity = str(data.get("severity") or "info")
        color = {"critical": "bold red", "error": "red", "warning": "yellow"}.get(severity, "blue")
        if (
            severity == "info"
            and not self.verbose
            and not etype.startswith(("guardrail.", "approval.", "budget.", "tool.slow", "cache."))
        ):
            return
        payload = data.get("data") or {}
        self._print(
            f"  [{color}]! {escape(etype)}[/{color}] [dim]{escape(_short(payload, 200))}[/dim]"
        )


def print_summary(
    stats: dict[str, Any], *, duration_s: float, console: Console | None = None, code: int = 0
) -> None:
    """Print the end-of-run summary for a ``stats`` payload."""
    console = console or Console(stderr=True, highlight=False)
    totals = stats.get("totals") or {}
    status = "[green]exit 0[/green]" if code == 0 else f"[red]exit {code}[/red]"
    console.print()
    console.print(
        f"[bold]promptise run[/bold] · {status} · {duration_s:.1f}s · "
        f"{totals.get('llm_calls', 0)} llm call(s) · {totals.get('total_tokens', 0)} tok "
        f"([dim]in[/dim] {totals.get('prompt_tokens', 0)} [dim]out[/dim] {totals.get('completion_tokens', 0)}) · "
        f"{totals.get('tool_calls', 0)} tool call(s) · {totals.get('errors', 0)} error(s) · "
        f"cost {format_cost(totals.get('cost_usd'))}"
    )
    agents = stats.get("agents") or {}
    if agents:
        table = Table(show_edge=False, header_style="bold", pad_edge=False, box=None)
        for col in ("agent", "model", "llm calls", "tokens", "tool calls", "errors", "cost"):
            table.add_column(col, justify="left" if col in ("agent", "model") else "right")
        for name, row in agents.items():
            table.add_row(
                escape(name),
                escape(str(row.get("model") or "—")),
                str(row.get("llm_calls", 0)),
                str(row.get("total_tokens", 0)),
                str(row.get("tool_calls", 0)),
                str(row.get("errors", 0)),
                format_cost(row.get("cost_usd")),
            )
        console.print(table)
    tools = stats.get("tools") or {}
    if tools:
        table = Table(show_edge=False, header_style="bold", pad_edge=False, box=None)
        for col in ("tool", "calls", "errors", "avg ms"):
            table.add_column(col, justify="left" if col == "tool" else "right")
        for name, row in sorted(tools.items(), key=lambda kv: -kv[1].get("calls", 0)):
            table.add_row(
                escape(name),
                str(row.get("calls", 0)),
                str(row.get("errors", 0)),
                f"{row.get('avg_ms', 0):.0f}",
            )
        console.print(table)
    unpriced = totals.get("unpriced_models") or []
    if unpriced:
        console.print(
            f"[dim]no price for {escape(', '.join(unpriced))}: add it to [pricing] in "
            "promptise.toml for a cost estimate[/dim]"
        )


def write_report(path: Path, bus: EventBus, *, extra: dict[str, Any] | None = None) -> Path:
    """Write every kept run event and the final totals to *path* as JSON."""
    events = bus.history()
    stats = next((e for e in reversed(events) if e["kind"] == "stats"), None)
    report = {
        "version": 1,
        "stats": {k: v for k, v in (stats or {}).items() if k not in ("v", "seq", "ts", "kind")},
        "events": events,
        **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, default=str, indent=1), encoding="utf-8")
    return path
