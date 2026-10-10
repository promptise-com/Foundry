"""Behavioral health monitoring for the agent runtime.

Lightweight anomaly detection that identifies stuck agents, infinite
loops, empty responses, and high error rates — without making any
LLM calls.  All detection is pure pattern matching on tool call and
response history.

Example::

    from promptise.runtime.health import HealthMonitor
    from promptise.runtime.config import HealthConfig

    monitor = HealthMonitor(HealthConfig(enabled=True), process_id="p1")
    anomaly = await monitor.record_tool_call("search", {"q": "test"})
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import HealthConfig

logger = logging.getLogger("promptise.runtime.health")


class AnomalyType(str, Enum):
    """Types of behavioral anomaly."""

    STUCK = "stuck"
    LOOP = "loop"
    EMPTY_RESPONSE = "empty_response"
    HIGH_ERROR_RATE = "high_error_rate"


@dataclass
class Anomaly:
    """A detected behavioral anomaly.

    Attributes:
        anomaly_type: Kind of anomaly.
        description: Human-readable explanation.
        timestamp: When the anomaly was detected.
        details: Additional evidence (tool sequences, counts, etc.).
    """

    anomaly_type: AnomalyType
    description: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """Lightweight behavioral health analysis.

    All detection methods are pure logic — no I/O, no LLM calls.
    Anomalies respect a configurable cooldown to avoid spam.

    Args:
        config: Health monitoring configuration.
        process_id: Owning process ID.
    """

    def __init__(self, config: HealthConfig, process_id: str) -> None:
        self._config = config
        self._process_id = process_id

        max_window = max(
            config.loop_window,
            config.stuck_threshold * 2,
            config.empty_threshold * 2,
            config.error_window * 2,
        )
        self._tool_history: deque[tuple[str, str]] = deque(maxlen=max_window)
        self._response_lengths: deque[int] = deque(maxlen=max_window)
        self._error_window: deque[bool] = deque(maxlen=config.error_window)
        self._last_anomaly_time: dict[AnomalyType, float] = {}
        self._anomalies: list[Anomaly] = []
        self._anomaly_count: dict[AnomalyType, int] = dict.fromkeys(AnomalyType, 0)
        # Set when an anomaly fires; cleared by the next clean success.
        self._unrecovered = False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def anomalies(self) -> list[Anomaly]:
        """All detected anomalies."""
        return list(self._anomalies)

    @property
    def latest_anomaly(self) -> Anomaly | None:
        """Most recent anomaly, or ``None``."""
        return self._anomalies[-1] if self._anomalies else None

    @property
    def tool_history(self) -> list[tuple[str, str]]:
        """Recent tool call history ``[(name, args_hash)]``."""
        return list(self._tool_history)

    # ------------------------------------------------------------------
    # Recording API
    # ------------------------------------------------------------------

    def begin_invocation(self) -> None:
        """Start a new invocation: forget the previous run's tool calls.

        Stuck and loop detection look for repetition *within* a run.  A
        scheduled agent that calls the same tool once per run (e.g. a
        queue check every minute) is behaving normally, so tool history
        does not carry over between invocations.  Response lengths and
        the error window do carry over: "three empty replies in a row"
        and "half the recent runs failed" are cross-run signals.
        """
        self._tool_history.clear()

    async def record_tool_call(self, tool_name: str, args: dict[str, Any]) -> Anomaly | None:
        """Record a tool call and check for stuck/loop anomalies."""
        args_hash = self._hash_args(args)
        self._tool_history.append((tool_name, args_hash))

        # Run detectors
        anomaly = self._detect_stuck() or self._detect_loop()
        if anomaly and self._is_cooled_down(anomaly.anomaly_type):
            self._register_anomaly(anomaly)
            return anomaly
        return None

    async def record_response(self, content: str) -> Anomaly | None:
        """Record the agent's final reply and check for empty responses.

        Pass the reply the agent gave at the end of an invocation, not
        tool results: a tool returning ``[]`` is not an empty response.
        """
        self._response_lengths.append(len(content.strip()))
        anomaly = self._detect_empty()
        if anomaly and self._is_cooled_down(anomaly.anomaly_type):
            self._register_anomaly(anomaly)
            return anomaly
        return None

    async def record_success(self) -> bool:
        """Record a successful invocation.

        Returns:
            ``True`` if this success clears a previous anomaly (recovery).
        """
        self._error_window.append(False)
        # Recovery: the first clean success after an anomaly (once, so
        # process.py emits a single health.recovered per incident).
        if self._unrecovered:
            self._unrecovered = False
            return True
        return False

    async def record_error(self) -> Anomaly | None:
        """Record a failed invocation and check error rate."""
        self._error_window.append(True)
        anomaly = self._detect_error_rate()
        if anomaly and self._is_cooled_down(anomaly.anomaly_type):
            self._register_anomaly(anomaly)
            return anomaly
        return None

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------

    def _detect_stuck(self) -> Anomaly | None:
        """Detect: last N tool calls are identical (same name + same args)."""
        threshold = self._config.stuck_threshold
        if len(self._tool_history) < threshold:
            return None

        last_n = list(self._tool_history)[-threshold:]
        if len(set(last_n)) == 1:
            tool_name, args_hash = last_n[0]
            return Anomaly(
                anomaly_type=AnomalyType.STUCK,
                description=(
                    f"Agent stuck: tool '{tool_name}' called {threshold} times "
                    f"consecutively with identical arguments"
                ),
                details={
                    "tool_name": tool_name,
                    "consecutive_count": threshold,
                },
            )
        return None

    def _detect_loop(self) -> Anomaly | None:
        """Detect: repeating subsequence in tool call history."""
        window = self._config.loop_window
        min_repeats = self._config.loop_min_repeats
        history = list(self._tool_history)[-window:]

        if len(history) < 4:
            return None

        # Check for patterns of length 2 up to half the window
        for pattern_len in range(2, len(history) // min_repeats + 1):
            pattern = history[-pattern_len:]
            repeat_count = 0
            idx = len(history) - pattern_len

            while idx >= pattern_len:
                candidate = history[idx - pattern_len : idx]
                if candidate == pattern:
                    repeat_count += 1
                    idx -= pattern_len
                else:
                    break

            if (
                repeat_count >= min_repeats - 1
            ):  # -1 because the last occurrence is the pattern itself
                tool_names = [t[0] for t in pattern]
                return Anomaly(
                    anomaly_type=AnomalyType.LOOP,
                    description=(
                        f"Agent in loop: sequence {tool_names} repeating {repeat_count + 1} times"
                    ),
                    details={
                        "pattern": tool_names,
                        "repeat_count": repeat_count + 1,
                        "pattern_length": pattern_len,
                    },
                )
        return None

    def _detect_empty(self) -> Anomaly | None:
        """Detect: N consecutive responses below the character threshold."""
        threshold = self._config.empty_threshold
        max_chars = self._config.empty_max_chars

        if len(self._response_lengths) < threshold:
            return None

        last_n = list(self._response_lengths)[-threshold:]
        if all(length <= max_chars for length in last_n):
            return Anomaly(
                anomaly_type=AnomalyType.EMPTY_RESPONSE,
                description=(
                    f"Agent producing empty responses: {threshold} consecutive "
                    f"responses under {max_chars} characters"
                ),
                details={
                    "consecutive_count": threshold,
                    "max_chars": max_chars,
                    "lengths": last_n,
                },
            )
        return None

    def _detect_error_rate(self) -> Anomaly | None:
        """Detect: error rate above threshold in the sliding window."""
        if len(self._error_window) < self._config.error_window:
            return None

        error_count = sum(1 for e in self._error_window if e)
        rate = error_count / len(self._error_window)

        if rate >= self._config.error_rate_threshold:
            return Anomaly(
                anomaly_type=AnomalyType.HIGH_ERROR_RATE,
                description=(
                    f"High error rate: {rate:.0%} errors in last "
                    f"{len(self._error_window)} invocations "
                    f"(threshold: {self._config.error_rate_threshold:.0%})"
                ),
                details={
                    "error_rate": round(rate, 3),
                    "error_count": error_count,
                    "window_size": len(self._error_window),
                    "threshold": self._config.error_rate_threshold,
                },
            )
        return None

    # ------------------------------------------------------------------
    # Status & serialisation
    # ------------------------------------------------------------------

    def health_status(self) -> dict[str, Any]:
        """Return health summary for ``process.status()``."""
        return {
            "is_healthy": len(self._anomalies) == 0
            or (
                time.monotonic()
                - max(
                    (self._last_anomaly_time.get(t, 0.0) for t in AnomalyType),
                    default=0.0,
                )
                > self._config.cooldown
            ),
            "anomaly_count": sum(self._anomaly_count.values()),
            "anomalies_by_type": dict(self._anomaly_count),
            "latest_anomaly": (
                {
                    "type": self._anomalies[-1].anomaly_type.value,
                    "description": self._anomalies[-1].description,
                    "timestamp": self._anomalies[-1].timestamp.isoformat(),
                }
                if self._anomalies
                else None
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialise for journal checkpointing."""
        return {
            "tool_history": list(self._tool_history),
            "response_lengths": list(self._response_lengths),
            "error_window": list(self._error_window),
            "anomaly_count": dict(self._anomaly_count),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], config: Any) -> HealthMonitor:
        """Reconstruct from a journal checkpoint."""
        monitor = cls(config, process_id="recovered")
        for item in data.get("tool_history", []):
            if isinstance(item, (list, tuple)) and len(item) == 2:
                monitor._tool_history.append((item[0], item[1]))
        for length in data.get("response_lengths", []):
            monitor._response_lengths.append(length)
        for err in data.get("error_window", []):
            monitor._error_window.append(bool(err))
        counts = data.get("anomaly_count", {})
        for t in AnomalyType:
            monitor._anomaly_count[t] = counts.get(t.value, 0)
        return monitor

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _hash_args(self, args: dict[str, Any]) -> str:
        """Stable hash of tool arguments for comparison."""
        try:
            raw = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            raw = str(args)
        # MD5 used as a non-cryptographic fingerprint for anomaly detection
        # (detecting identical repeated tool calls). Not security-sensitive.
        return hashlib.md5(raw.encode(), usedforsecurity=False).hexdigest()[:12]

    def _is_cooled_down(self, anomaly_type: AnomalyType) -> bool:
        """Check if enough time has passed since the last anomaly of this type."""
        last_time = self._last_anomaly_time.get(anomaly_type)
        if last_time is None:
            return True
        return (time.monotonic() - last_time) >= self._config.cooldown

    def _register_anomaly(self, anomaly: Anomaly) -> None:
        """Record an anomaly and update cooldown tracking."""
        self._anomalies.append(anomaly)
        self._anomaly_count[anomaly.anomaly_type] += 1
        self._last_anomaly_time[anomaly.anomaly_type] = time.monotonic()
        self._unrecovered = True
        logger.warning(
            "Behavioral anomaly detected: %s — %s",
            anomaly.anomaly_type.value,
            anomaly.description,
        )
