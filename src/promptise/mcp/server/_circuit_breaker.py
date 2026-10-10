"""Circuit breaker middleware for MCP servers.

Protects downstream services from cascading failures by tracking
error rates and short-circuiting tool calls when a threshold is
exceeded.

Example::

    from promptise.mcp.server import MCPServer, CircuitBreakerMiddleware

    server = MCPServer(name="api")
    server.add_middleware(CircuitBreakerMiddleware(
        failure_threshold=5,
        recovery_timeout=60.0,
        excluded_tools={"health_check"},
    ))
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from ._context import RequestContext
from ._errors import (
    ApprovalDeniedError,
    AuthenticationError,
    MCPError,
    RateLimitError,
    ValidationError,
)


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing — reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class _CircuitStats:
    """Per-tool circuit state and counters."""

    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: float = 0.0
    success_count: int = 0
    probe_in_flight: bool = False


class CircuitOpenError(MCPError):
    """Raised when a tool call is rejected by an open circuit.

    Reaches the client as a retryable ``CIRCUIT_OPEN`` error with
    ``details.retry_after_seconds``, so an agent knows to wait (or use
    another tool) instead of treating the tool as broken.  Register
    ``@server.exception_handler(CircuitOpenError)`` to customise it.

    Attributes:
        tool: The tool whose circuit is open.
        retry_after: Seconds until the breaker lets a probe call through.
    """

    def __init__(self, tool: str, retry_after: float) -> None:
        self.tool = tool
        self.retry_after = retry_after
        super().__init__(
            f"Circuit open for tool '{tool}'. Retry after {retry_after:.1f}s.",
            code="CIRCUIT_OPEN",
            retryable=True,
            suggestion=(
                f"'{tool}' is paused after repeated failures. Wait about "
                f"{max(1, round(retry_after))}s before calling it again, "
                "or continue without it."
            ),
            details={"tool": tool, "retry_after_seconds": round(retry_after, 1)},
        )


# Errors that say the *request* was refused or wrong, not that the tool or
# its upstream is unhealthy.  They never trip a breaker by default.
_CALLER_ERRORS: tuple[type[MCPError], ...] = (
    AuthenticationError,
    ValidationError,
    RateLimitError,
    ApprovalDeniedError,
    CircuitOpenError,
)


def is_upstream_failure(exc: BaseException) -> bool:
    """Default failure classifier for :class:`CircuitBreakerMiddleware`.

    Counts:

    * any exception that is not an ``MCPError`` — a crash, a connection
      error, an upstream SDK raising: the tool is unhealthy;
    * a *retryable* ``MCPError`` other than the caller-side ones below —
      e.g. a ``TIMEOUT``, or ``ToolError("upstream unavailable",
      retryable=True)``.

    Does not count:

    * a non-retryable ``ToolError`` — the tool worked and rejected this
      input (``ToolError("No product BAD")``);
    * authentication / access denials, validation errors, rate-limit and
      concurrency refusals, approval denials, and ``CircuitOpenError``.
    """
    if not isinstance(exc, Exception):
        return False  # cancellation, KeyboardInterrupt, ...
    if not isinstance(exc, MCPError):
        return True
    if isinstance(exc, _CALLER_ERRORS):
        return False
    return exc.retryable


class CircuitBreakerMiddleware:
    """Circuit breaker middleware.

    Tracks consecutive failures per tool. When ``failure_threshold``
    consecutive failures occur, the circuit opens and subsequent calls
    are rejected immediately with :class:`CircuitOpenError` (a retryable
    ``CIRCUIT_OPEN`` error).  After ``recovery_timeout`` seconds the
    circuit enters half-open state and lets exactly **one** probe call
    through; other calls are rejected until the probe finishes.  A
    successful probe closes the circuit, a failed one re-opens it.

    Only upstream and unexpected errors count as failures (see
    :func:`is_upstream_failure`): a tool rejecting bad input, an access
    denial, or a rate-limit / capacity refusal never pauses the tool for
    everyone.  Errors that don't count leave the failure streak as it is.

    Args:
        failure_threshold: Consecutive failures before opening the
            circuit (default ``5``).
        recovery_timeout: Seconds to wait before probing recovery
            (default ``60.0``).
        excluded_tools: Tool names exempt from circuit breaking.
        is_failure: ``(exc) -> bool`` deciding whether an exception
            counts towards opening the circuit.  Defaults to
            :func:`is_upstream_failure`.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        *,
        excluded_tools: set[str] | None = None,
        is_failure: Callable[[BaseException], bool] | None = None,
    ) -> None:
        self._threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._excluded = excluded_tools or set()
        self._is_failure = is_failure or is_upstream_failure
        self._circuits: dict[str, _CircuitStats] = {}

    def _get_circuit(self, tool: str) -> _CircuitStats:
        if tool not in self._circuits:
            self._circuits[tool] = _CircuitStats()
        return self._circuits[tool]

    def get_state(self, tool: str) -> CircuitState:
        """Get the current circuit state for a tool."""
        return self._get_circuit(tool).state

    async def __call__(self, ctx: RequestContext, call_next: Callable[..., Any]) -> Any:
        tool = ctx.tool_name
        if tool in self._excluded:
            return await call_next(ctx)

        circuit = self._get_circuit(tool)

        # Open circuit — reject unless recovery timeout has elapsed
        if circuit.state == CircuitState.OPEN:
            elapsed = time.monotonic() - circuit.last_failure_time
            if elapsed < self._recovery_timeout:
                raise CircuitOpenError(tool, self._recovery_timeout - elapsed)
            circuit.state = CircuitState.HALF_OPEN

        # Half-open — exactly one probe at a time; the rest wait it out
        is_probe = False
        if circuit.state == CircuitState.HALF_OPEN:
            if circuit.probe_in_flight:
                raise CircuitOpenError(tool, min(self._recovery_timeout, 1.0))
            circuit.probe_in_flight = True
            is_probe = True

        try:
            result = await call_next(ctx)
        except BaseException as exc:
            if self._is_failure(exc):
                circuit.failure_count += 1
                circuit.last_failure_time = time.monotonic()
                circuit.success_count = 0
                if is_probe or circuit.failure_count >= self._threshold:
                    circuit.state = CircuitState.OPEN
            # A probe that ended without a verdict (bad input, cancelled)
            # leaves the circuit half-open: the next call probes again.
            raise
        else:
            circuit.failure_count = 0
            circuit.success_count += 1
            if circuit.state == CircuitState.HALF_OPEN:
                circuit.state = CircuitState.CLOSED
            return result
        finally:
            if is_probe:
                circuit.probe_in_flight = False

    def reset(self, tool: str | None = None) -> None:
        """Reset circuit state for a tool (or all tools)."""
        if tool is None:
            self._circuits.clear()
        else:
            self._circuits.pop(tool, None)
