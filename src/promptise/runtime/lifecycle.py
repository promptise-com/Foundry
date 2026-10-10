"""Process lifecycle state machine.

Defines the valid states an :class:`AgentProcess` can be in and the
legal transitions between them.  The :class:`ProcessLifecycle` class
provides a thread-safe, auditable state machine that records every
transition with timestamps and reasons.

State diagram::

    CREATED ──► STARTING ──► RUNNING ──► STOPPING ──► STOPPED
                   │            │  ▲          ▲
                   │            │  │          │
                   ▼            ▼  │          │
                 FAILED     SUSPENDED     FAILED ─► STARTING (restart)
                               │
                               ▼
                           AWAITING ──► RUNNING (trigger fires)
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

#: Callback invoked after every successful transition (sync or async).
TransitionListener = Callable[["ProcessTransition"], "Awaitable[None] | None"]


class ProcessState(str, Enum):
    """Agent process lifecycle states."""

    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    SUSPENDED = "suspended"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"
    AWAITING = "awaiting"


#: Valid transitions: current state → set of allowed target states.
VALID_TRANSITIONS: dict[ProcessState, frozenset[ProcessState]] = {
    ProcessState.CREATED: frozenset({ProcessState.STARTING, ProcessState.STOPPED}),
    ProcessState.STARTING: frozenset(
        {ProcessState.RUNNING, ProcessState.FAILED, ProcessState.STOPPING}
    ),
    ProcessState.RUNNING: frozenset(
        {
            ProcessState.SUSPENDED,
            ProcessState.STOPPING,
            ProcessState.FAILED,
            ProcessState.AWAITING,
        }
    ),
    ProcessState.SUSPENDED: frozenset(
        {ProcessState.RUNNING, ProcessState.STOPPING, ProcessState.FAILED}
    ),
    ProcessState.AWAITING: frozenset(
        {ProcessState.RUNNING, ProcessState.STOPPING, ProcessState.FAILED}
    ),
    ProcessState.STOPPING: frozenset({ProcessState.STOPPED, ProcessState.FAILED}),
    ProcessState.STOPPED: frozenset({ProcessState.STARTING}),
    ProcessState.FAILED: frozenset({ProcessState.STARTING}),
}


class StateError(ValueError):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, current: ProcessState, target: ProcessState) -> None:
        self.current = current
        self.target = target
        allowed = sorted(s.value for s in VALID_TRANSITIONS.get(current, set()))
        super().__init__(
            f"Cannot transition from {current.value!r} to {target.value!r}. "
            f"Allowed targets: {allowed}"
        )


@dataclass
class ProcessTransition:
    """Record of a single state transition.

    Attributes:
        from_state: Previous state.
        to_state: New state.
        timestamp: When the transition occurred (timezone.utc).
        reason: Human-readable reason for the transition.
        metadata: Additional context (e.g. error details).
    """

    from_state: ProcessState
    to_state: ProcessState
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "timestamp": self.timestamp.isoformat(),
            "reason": self.reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProcessTransition:
        """Deserialize from a dict."""
        return cls(
            from_state=ProcessState(data["from_state"]),
            to_state=ProcessState(data["to_state"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            reason=data.get("reason", ""),
            metadata=data.get("metadata", {}),
        )


class ProcessLifecycle:
    """Thread-safe state machine for process lifecycle.

    Tracks the current state, validates transitions against
    :data:`VALID_TRANSITIONS`, and records an audit trail of all
    transitions.

    Args:
        initial: Starting state (default :attr:`ProcessState.CREATED`).

    Example::

        lc = ProcessLifecycle()
        assert lc.state == ProcessState.CREATED

        await lc.transition(ProcessState.STARTING, reason="user requested")
        await lc.transition(ProcessState.RUNNING)

        assert len(lc.history) == 2
    """

    def __init__(self, initial: ProcessState = ProcessState.CREATED) -> None:
        self._state = initial
        self._history: list[ProcessTransition] = []
        self._lock = asyncio.Lock()
        self._listeners: list[TransitionListener] = []

    def add_listener(self, listener: TransitionListener) -> None:
        """Register a callback invoked after every successful transition.

        Listeners run in registration order, outside the state lock, so
        they may inspect :attr:`state` (and even trigger further
        transitions).  A listener that raises is logged and skipped; it
        never undoes the transition.

        Args:
            listener: Sync or async callable receiving the
                :class:`ProcessTransition`.
        """
        self._listeners.append(listener)

    @property
    def state(self) -> ProcessState:
        """Current process state."""
        return self._state

    @property
    def history(self) -> list[ProcessTransition]:
        """Chronological list of all transitions."""
        return list(self._history)

    def can_transition(self, target: ProcessState) -> bool:
        """Check whether transitioning to *target* is valid."""
        return target in VALID_TRANSITIONS.get(self._state, frozenset())

    async def transition(
        self,
        target: ProcessState,
        *,
        reason: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ProcessTransition:
        """Attempt a state transition.

        Args:
            target: Desired next state.
            reason: Human-readable explanation.
            metadata: Extra context to store with the transition.

        Returns:
            The recorded :class:`ProcessTransition`.

        Raises:
            StateError: If the transition is not valid from the current state.
        """
        async with self._lock:
            if not self.can_transition(target):
                raise StateError(self._state, target)

            transition = ProcessTransition(
                from_state=self._state,
                to_state=target,
                reason=reason,
                metadata=metadata or {},
            )
            self._state = target
            self._history.append(transition)

        for listener in list(self._listeners):
            try:
                result = listener(transition)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.exception(
                    "Lifecycle listener failed for %s -> %s",
                    transition.from_state.value,
                    transition.to_state.value,
                )
        return transition

    def snapshot(self) -> dict[str, Any]:
        """Serializable snapshot of current state and full history."""
        return {
            "state": self._state.value,
            "history": [t.to_dict() for t in self._history],
        }

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> ProcessLifecycle:
        """Reconstruct a :class:`ProcessLifecycle` from a snapshot dict."""
        lc = cls(initial=ProcessState(data["state"]))
        lc._history = [ProcessTransition.from_dict(t) for t in data.get("history", [])]
        return lc

    def __repr__(self) -> str:
        return f"ProcessLifecycle(state={self._state.value!r}, transitions={len(self._history)})"
