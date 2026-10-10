"""Custom exceptions for the agent runtime module.

All runtime exceptions inherit from :class:`RuntimeBaseError` so callers
can catch the whole family with a single ``except RuntimeBaseError``.
"""

from __future__ import annotations

from typing import Any


class RuntimeBaseError(RuntimeError):
    """Base exception for all runtime module errors."""


class ProcessStateError(RuntimeBaseError):
    """Raised when a lifecycle state transition is invalid.

    Attributes:
        process_id: The process that attempted the transition.
        current_state: The process's current state.
        attempted_state: The target state that was rejected.
    """

    def __init__(
        self,
        process_id: str,
        current_state: str,
        attempted_state: str,
    ) -> None:
        self.process_id = process_id
        self.current_state = current_state
        self.attempted_state = attempted_state
        super().__init__(
            f"Process {process_id!r}: invalid transition {current_state!r} -> {attempted_state!r}"
        )


class ManifestError(RuntimeBaseError):
    """Raised when ``.agent`` manifest loading fails (I/O or parse error)."""


class ManifestValidationError(ManifestError):
    """Raised when ``.agent`` manifest schema validation fails.

    Attributes:
        errors: Pydantic validation error details (if available).
        file_path: Path to the manifest file that failed validation.
    """

    def __init__(
        self,
        message: str,
        *,
        errors: list[dict[str, Any]] | None = None,
        file_path: str | None = None,
    ) -> None:
        self.errors = errors or []
        self.file_path = file_path
        super().__init__(message)


class TriggerError(RuntimeBaseError):
    """Raised when a trigger fails to start, stop, or produce events."""


class JournalError(RuntimeBaseError):
    """Raised when journal read, write, or replay fails."""


class BudgetExceededError(RuntimeBaseError):
    """Raised before a tool call that would exceed an autonomy budget limit.

    The runtime raises it from the tool-start callback, so the tool never
    executes; the agent sees the message as the tool's error result and
    can wind the run down.

    Attributes:
        violation: The :class:`~promptise.runtime.budget.BudgetViolation`
            that blocked the call.
    """

    def __init__(self, violation: Any) -> None:
        self.violation = violation
        tool = f" '{violation.tool_name}'" if violation.tool_name else ""
        super().__init__(
            f"Budget limit {violation.limit_name} reached "
            f"(limit {violation.limit_value:g}); tool call{tool} was not executed. "
            "Stop calling tools and give your final answer."
        )
