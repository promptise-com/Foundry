"""Running totals for a ``promptise run`` session, folded from timeline entries.

:class:`RunStats` is fed every timeline entry of every agent the program
builds and keeps the totals the dev tools, the terminal UI and the run report
show: LLM calls, tokens, estimated cost, tool calls and errors, latencies,
per agent, per model and per tool.  :meth:`RunStats.snapshot` is the payload
of a ``stats`` run event::

    {
      "totals": {"llm_calls", "prompt_tokens", "completion_tokens", "total_tokens",
                 "tool_calls", "tool_errors", "errors", "retries", "cost_usd",
                 "unpriced_models", "latency_p50_ms", "latency_p95_ms"},
      "agents": {id: {"model", "llm_calls", "total_tokens", "tool_calls", "errors", "cost_usd"}},
      "models": {name: {"llm_calls", "prompt_tokens", "completion_tokens", "cost_usd"}},
      "tools":  {name: {"calls", "errors", "total_ms", "avg_ms"}},
    }

``cost_usd`` is ``None`` until a call of a priced model has been seen;
``unpriced_models`` lists the models with calls but no ``[pricing]`` entry.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Any

from .pricing import PriceTable

__all__ = ["RunStats"]

_ERROR_TYPES = {"llm.error", "tool.error", "agent.error", "task.failed"}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * pct / 100), len(ordered) - 1)]


class RunStats:
    """Thread-safe running totals (see the module docstring).

    Args:
        prices: The price table for cost estimates.
    """

    def __init__(self, prices: PriceTable | None = None) -> None:
        self._prices = prices or PriceTable()
        self._lock = threading.Lock()
        self._totals: dict[str, Any] = {
            "llm_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "tool_calls": 0,
            "tool_errors": 0,
            "errors": 0,
            "retries": 0,
            "cost_usd": None,
        }
        self._unpriced: set[str] = set()
        self._agents: dict[str, dict[str, Any]] = {}
        self._models: dict[str, dict[str, Any]] = {}
        self._tools: dict[str, dict[str, Any]] = {}
        self._latencies: deque[float] = deque(maxlen=5_000)
        # Runs that recorded llm.end: their agent.output totals are not added again.
        self._runs_with_llm_end: set[str] = set()
        self.version = 0
        """Incremented on every change, so a publisher can tell when to send."""

    def _agent(self, agent: str) -> dict[str, Any]:
        row = self._agents.get(agent)
        if row is None:
            row = {
                "model": None,
                "llm_calls": 0,
                "total_tokens": 0,
                "tool_calls": 0,
                "errors": 0,
                "cost_usd": None,
            }
            self._agents[agent] = row
        return row

    def set_model(self, agent: str, model: str | None) -> None:
        """Record *agent*'s configured model (shown before its first call)."""
        with self._lock:
            row = self._agent(agent)
            if model and not row["model"]:
                row["model"] = model
                self.version += 1

    def _add_llm(
        self,
        agent: str,
        model: str | None,
        prompt: int,
        completion: int,
        total: int,
        calls: int,
    ) -> None:
        totals = self._totals
        totals["llm_calls"] += calls
        totals["prompt_tokens"] += prompt
        totals["completion_tokens"] += completion
        totals["total_tokens"] += total
        row = self._agent(agent)
        row["llm_calls"] += calls
        row["total_tokens"] += total
        name = model or row["model"] or "unknown"
        m = self._models.setdefault(
            name, {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": None}
        )
        m["llm_calls"] += calls
        m["prompt_tokens"] += prompt
        m["completion_tokens"] += completion
        cost = self._prices.cost(name, prompt, completion)
        if cost is None:
            if prompt or completion:
                self._unpriced.add(name)
            return
        totals["cost_usd"] = (totals["cost_usd"] or 0.0) + cost
        row["cost_usd"] = (row["cost_usd"] or 0.0) + cost
        m["cost_usd"] = (m["cost_usd"] or 0.0) + cost

    def add_entry(self, agent: str, entry: dict[str, Any]) -> None:
        """Fold one timeline entry (``TimelineEntry.to_dict()``) of *agent* into the totals."""
        etype = entry.get("event_type")
        meta = entry.get("metadata") or {}
        with self._lock:
            changed = True
            if etype == "llm.end":
                parent = entry.get("parent_id")
                if parent:
                    self._runs_with_llm_end.add(parent)
                pt = int(meta.get("prompt_tokens") or 0)
                ct = int(meta.get("completion_tokens") or 0)
                tt = int(meta.get("total_tokens") or pt + ct)
                self._add_llm(agent, meta.get("model"), pt, ct, tt, 1)
                if entry.get("duration") is not None:
                    self._latencies.append(float(entry["duration"]) * 1000)
            elif etype == "agent.output" and "llm_call_count" in meta:
                # BASIC level: no llm.end, the run's totals ride on agent.output.
                if entry.get("parent_id") not in self._runs_with_llm_end:
                    pt = int(meta.get("prompt_tokens") or 0)
                    ct = int(meta.get("completion_tokens") or 0)
                    tt = int(meta.get("total_tokens") or pt + ct)
                    self._add_llm(agent, None, pt, ct, tt, int(meta.get("llm_call_count") or 0))
                else:
                    changed = False
            elif etype == "tool.call":
                self._totals["tool_calls"] += 1
                self._agent(agent)["tool_calls"] += 1
                name = str(meta.get("tool_name") or "unknown")
                tool = self._tools.setdefault(
                    name, {"calls": 0, "errors": 0, "total_ms": 0.0, "avg_ms": 0.0}
                )
                tool["calls"] += 1
            elif etype in ("tool.result", "tool.error"):
                name = str(meta.get("tool_name") or "unknown")
                tool = self._tools.setdefault(
                    name, {"calls": 0, "errors": 0, "total_ms": 0.0, "avg_ms": 0.0}
                )
                latency = meta.get("latency_ms")
                if latency is not None:
                    tool["total_ms"] += float(latency)
                    tool["avg_ms"] = tool["total_ms"] / max(tool["calls"], 1)
                failed = etype == "tool.error" or meta.get("status") == "error"
                if failed:
                    tool["errors"] += 1
                    self._totals["tool_errors"] += 1
                    self._totals["errors"] += 1
                    self._agent(agent)["errors"] += 1
            elif etype in _ERROR_TYPES:
                self._totals["errors"] += 1
                self._agent(agent)["errors"] += 1
            elif etype == "llm.retry":
                self._totals["retries"] += 1
            else:
                changed = False
            if changed:
                self.version += 1

    def snapshot(self) -> dict[str, Any]:
        """The payload of a ``stats`` run event (see the module docstring)."""
        with self._lock:
            latencies = list(self._latencies)
            totals = dict(self._totals)
            totals["unpriced_models"] = sorted(self._unpriced)
            totals["latency_p50_ms"] = round(_percentile(latencies, 50), 1)
            totals["latency_p95_ms"] = round(_percentile(latencies, 95), 1)
            return {
                "totals": totals,
                "agents": {k: dict(v) for k, v in self._agents.items()},
                "models": {k: dict(v) for k, v in self._models.items()},
                "tools": {k: dict(v) for k, v in self._tools.items()},
            }
