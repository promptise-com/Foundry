"""Autonomy budget system for the agent runtime.

Provides per-run and daily limits on tool calls, LLM turns, cost,
and irreversible actions.  Agents are aware of their remaining budget
and the runtime enforces limits with configurable behaviour (pause,
stop, or escalate).

Example::

    from promptise.runtime.budget import BudgetState
    from promptise.runtime.config import BudgetConfig

    config = BudgetConfig(
        enabled=True,
        max_tool_calls_per_run=20,
        max_cost_per_day=50.0,
    )
    state = BudgetState(config)
    violation = await state.record_tool_call("search")
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import BudgetConfig

logger = logging.getLogger("promptise.runtime.budget")


@dataclass
class BudgetWarning:
    """Describes a budget limit approaching its threshold (default 80%).

    Attributes:
        limit_name: Which limit is approaching (e.g. ``"max_tool_calls_per_day"``).
        limit_value: The configured maximum.
        current_value: The current counter value.
        percentage: How close to the limit (0.0-1.0).
    """

    limit_name: str
    limit_value: int | float
    current_value: int | float
    percentage: float


@dataclass
class BudgetViolation:
    """Describes a budget limit that was exceeded.

    Attributes:
        limit_name: Which limit was hit (e.g. ``"max_tool_calls_per_run"``).
        limit_value: The configured maximum.
        current_value: The counter value that exceeded the limit (for a
            blocked call: the value the call *would* have produced).
        tool_name: Tool that triggered the violation (if applicable).
        blocked: ``True`` when the tool call was refused before it ran
            (per-call enforcement) rather than detected after the fact.
    """

    limit_name: str
    limit_value: float
    current_value: float
    tool_name: str | None = None
    blocked: bool = False


class BudgetState:
    """Thread-safe budget counters with per-run and daily tracking.

    Args:
        config: Budget configuration with limits and tool cost annotations.
    """

    def __init__(self, config: BudgetConfig) -> None:
        self._config = config
        # A thread lock, not an asyncio.Lock: no critical section awaits,
        # and synchronous tools report their calls from executor threads
        # (see record_tool_call_sync).
        self._lock = threading.Lock()

        # Per-run counters (reset each invocation)
        self.run_tool_calls: int = 0
        self.run_llm_turns: int = 0
        self.run_cost: float = 0.0
        self.run_irreversible: int = 0

        # Daily counters
        self.daily_tool_calls: int = 0
        self.daily_runs: int = 0
        self.daily_cost: float = 0.0
        self.last_daily_reset: datetime = datetime.now(timezone.utc)

        # Limits at >= 80% after the latest check (see _check_limits)
        self.pending_warnings: list[BudgetWarning] = []

    async def record_tool_call(
        self, tool_name: str, *, enforce: bool = False
    ) -> BudgetViolation | None:
        """Record a tool call and check per-run + daily limits.

        Args:
            tool_name: The tool about to run.
            enforce: Check the call *before* counting it.  If running it
                would exceed a limit (or a limit is already exceeded, such
                as ``max_llm_turns_per_run``), the call is not counted and
                the returned violation has ``blocked=True`` — the caller
                must not execute the tool.

        Returns the first :class:`BudgetViolation` found, or ``None``.
        """
        return self.record_tool_call_sync(tool_name, enforce=enforce)

    def record_tool_call_sync(
        self, tool_name: str, *, enforce: bool = False
    ) -> BudgetViolation | None:
        """Synchronous :meth:`record_tool_call` (thread-safe).

        Used by the budget guard callback, which LangChain may run on a
        worker thread for synchronous tools.
        """
        with self._lock:
            # Look up cost annotation
            annotation = self._config.tool_costs.get(tool_name)
            cost_weight = 1.0
            irreversible = False
            if annotation is not None:
                cost_weight = annotation.cost_weight
                irreversible = annotation.irreversible

            if enforce:
                violation = self._check_limits(
                    tool_name,
                    extra_calls=1,
                    extra_cost=cost_weight,
                    extra_irreversible=1 if irreversible else 0,
                )
                if violation is not None:
                    violation.blocked = True
                    return violation

            self.run_tool_calls += 1
            self.daily_tool_calls += 1
            self.run_cost += cost_weight
            self.daily_cost += cost_weight
            if irreversible:
                self.run_irreversible += 1

            return self._check_limits(tool_name)

    async def record_llm_turn(self) -> BudgetViolation | None:
        """Record an LLM turn and check limits."""
        with self._lock:
            self.run_llm_turns += 1
            return self._check_limits()

    async def record_run_start(self) -> BudgetViolation | None:
        """Record start of a new invocation and check daily run limit."""
        with self._lock:
            self.daily_runs += 1
            cfg = self._config
            if cfg.max_runs_per_day is not None and self.daily_runs > cfg.max_runs_per_day:
                return BudgetViolation(
                    limit_name="max_runs_per_day",
                    limit_value=cfg.max_runs_per_day,
                    current_value=self.daily_runs,
                )
            return None

    async def reset_run(self) -> None:
        """Reset per-run counters at the start of each invocation."""
        with self._lock:
            self.run_tool_calls = 0
            self.run_llm_turns = 0
            self.run_cost = 0.0
            self.run_irreversible = 0

    async def check_daily_reset(self) -> bool:
        """Reset daily counters if the reset hour has passed.

        Returns:
            ``True`` if counters were reset, ``False`` otherwise.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            reset_hour = self._config.daily_reset_hour_utc
            # Check if we've crossed the reset hour since last reset
            if now.date() > self.last_daily_reset.date() or (
                now.date() == self.last_daily_reset.date()
                and now.hour >= reset_hour
                and self.last_daily_reset.hour < reset_hour
            ):
                self.daily_tool_calls = 0
                self.daily_runs = 0
                self.daily_cost = 0.0
                self.last_daily_reset = now
                logger.info("Daily budget counters reset")
                return True
            return False

    def remaining(self) -> dict[str, int | float | None]:
        """Return remaining budget per category.

        ``None`` means unlimited.  Non-negative values indicate
        remaining capacity.  Negative values mean the limit was
        exceeded.
        """
        cfg = self._config
        result: dict[str, int | float | None] = {}

        def _remaining(limit: int | float | None, current: int | float) -> int | float | None:
            if limit is None:
                return None
            return limit - current

        result["tool_calls_run"] = _remaining(cfg.max_tool_calls_per_run, self.run_tool_calls)
        result["llm_turns_run"] = _remaining(cfg.max_llm_turns_per_run, self.run_llm_turns)
        result["cost_run"] = _remaining(cfg.max_cost_per_run, self.run_cost)
        result["irreversible_run"] = _remaining(cfg.max_irreversible_per_run, self.run_irreversible)
        result["tool_calls_day"] = _remaining(cfg.max_tool_calls_per_day, self.daily_tool_calls)
        result["runs_day"] = _remaining(cfg.max_runs_per_day, self.daily_runs)
        result["cost_day"] = _remaining(cfg.max_cost_per_day, self.daily_cost)
        return result

    def budget_context(self) -> str:
        """Format remaining budget as a context string for injection."""
        r = self.remaining()
        parts: list[str] = []
        for key, value in r.items():
            if value is not None:
                label = key.replace("_", " ").title()
                parts.append(f"{label}: {value}")
        if not parts:
            return ""
        return "[Budget] " + " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for journal checkpointing."""
        return {
            "run_tool_calls": self.run_tool_calls,
            "run_llm_turns": self.run_llm_turns,
            "run_cost": self.run_cost,
            "run_irreversible": self.run_irreversible,
            "daily_tool_calls": self.daily_tool_calls,
            "daily_runs": self.daily_runs,
            "daily_cost": self.daily_cost,
            "last_daily_reset": self.last_daily_reset.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: Any) -> BudgetState:
        """Reconstruct from a journal checkpoint."""
        state = cls(config)
        state.run_tool_calls = data.get("run_tool_calls", 0)
        state.run_llm_turns = data.get("run_llm_turns", 0)
        state.run_cost = data.get("run_cost", 0.0)
        state.run_irreversible = data.get("run_irreversible", 0)
        state.daily_tool_calls = data.get("daily_tool_calls", 0)
        state.daily_runs = data.get("daily_runs", 0)
        state.daily_cost = data.get("daily_cost", 0.0)
        reset_str = data.get("last_daily_reset")
        if reset_str:
            state.last_daily_reset = datetime.fromisoformat(reset_str)
        return state

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _check_limits(
        self,
        tool_name: str | None = None,
        *,
        extra_calls: int = 0,
        extra_cost: float = 0.0,
        extra_irreversible: int = 0,
    ) -> BudgetViolation | None:
        """Check all limits and return the first violation found.

        The ``extra_*`` deltas are added to the current counters without
        committing them, to ask "would one more call exceed a limit?".

        Also populates :attr:`pending_warnings` for limits at ≥80% usage.
        """
        cfg = self._config
        checks: list[tuple[str, int | float | None, int | float]] = [
            (
                "max_tool_calls_per_run",
                cfg.max_tool_calls_per_run,
                self.run_tool_calls + extra_calls,
            ),
            ("max_llm_turns_per_run", cfg.max_llm_turns_per_run, self.run_llm_turns),
            ("max_cost_per_run", cfg.max_cost_per_run, self.run_cost + extra_cost),
            (
                "max_irreversible_per_run",
                cfg.max_irreversible_per_run,
                self.run_irreversible + extra_irreversible,
            ),
            (
                "max_tool_calls_per_day",
                cfg.max_tool_calls_per_day,
                self.daily_tool_calls + extra_calls,
            ),
            ("max_cost_per_day", cfg.max_cost_per_day, self.daily_cost + extra_cost),
        ]

        # Check for warnings (≥80% of limit) — populate pending_warnings
        warning_threshold = getattr(cfg, "warning_threshold", 0.8)
        self.pending_warnings = []
        for limit_name, limit_value, current in checks:
            if limit_value is not None and limit_value > 0:
                pct = current / limit_value
                if pct >= warning_threshold and current <= limit_value:
                    self.pending_warnings.append(
                        BudgetWarning(
                            limit_name=limit_name,
                            limit_value=limit_value,
                            current_value=current,
                            percentage=round(pct, 3),
                        )
                    )

        for limit_name, limit_value, current in checks:
            if limit_value is not None and current > limit_value:
                return BudgetViolation(
                    limit_name=limit_name,
                    limit_value=limit_value,
                    current_value=current,
                    tool_name=tool_name,
                )
        return None


class BudgetEnforcer:
    """Handles budget violations according to the configured policy.

    Args:
        config: Budget configuration with ``on_exceeded`` policy.
    """

    def __init__(self, config: BudgetConfig) -> None:
        self._config = config

    async def handle_violation(
        self,
        violation: BudgetViolation,
        process: Any,  # AgentProcess (avoids circular import)
    ) -> None:
        """Execute the configured response to a budget violation.

        Actions:
        - ``"pause"``: Suspend the process.
        - ``"stop"``: Stop the process entirely.
        - ``"escalate"``: Fire escalation notification, then suspend.

        Args:
            violation: The budget violation that occurred.
            process: The owning AgentProcess.
        """
        action = self._config.on_exceeded
        logger.warning(
            "Budget violation: %s (limit=%s, current=%s) — action=%s",
            violation.limit_name,
            violation.limit_value,
            violation.current_value,
            action,
        )

        if action == "pause":
            try:
                await process.suspend()
            except Exception as exc:
                logger.debug("Budget pause failed: %s", exc)

        elif action == "stop":
            try:
                await process.stop()
            except Exception as exc:
                logger.debug("Budget stop failed: %s", exc)

        elif action == "escalate":
            if self._config.escalation is not None:
                from .escalation import escalate

                payload = {
                    "type": "budget_violation",
                    "process_id": getattr(process, "process_id", "unknown"),
                    "process_name": getattr(process, "name", "unknown"),
                    "violation": {
                        "limit_name": violation.limit_name,
                        "limit_value": violation.limit_value,
                        "current_value": violation.current_value,
                        "tool_name": violation.tool_name,
                    },
                }
                event_bus = getattr(process, "_event_bus", None)
                await escalate(self._config.escalation, payload, event_bus=event_bus)
            # Also pause the process after escalating
            try:
                await process.suspend()
            except Exception:
                pass
