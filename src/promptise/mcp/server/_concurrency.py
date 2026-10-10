"""Concurrent request tracking and limiting.

Prevents resource exhaustion under load spikes by enforcing a
configurable maximum number of in-flight tool calls.

Example::

    from promptise.mcp.server import MCPServer, ConcurrencyLimiter

    limiter = ConcurrencyLimiter(max_concurrent=50)
    server.add_middleware(limiter)

Per-tool concurrency limits can also be set via the ``@server.tool()``
decorator::

    @server.tool(max_concurrent=5)
    async def db_query(sql: str) -> list[dict]:
        return await db.execute(sql)
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from ._context import RequestContext
from ._errors import ConcurrencyLimitError

logger = logging.getLogger("promptise.server")


class ConcurrencyLimiter:
    """Middleware that limits concurrent in-flight tool executions.

    When the limit is reached, incoming requests receive a retryable
    ``ConcurrencyLimitError`` (code ``CONCURRENCY_LIMIT_EXCEEDED``)
    instead of queueing indefinitely.

    Args:
        max_concurrent: Maximum number of concurrent tool calls.
            ``0`` means unlimited (tracking only).
    """

    def __init__(self, max_concurrent: int = 100) -> None:
        self._max = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent > 0 else None
        self._active: int = 0
        self._total: int = 0
        self._peak: int = 0

    async def __call__(self, ctx: RequestContext, call_next: Callable[..., Any]) -> Any:
        if self._semaphore is not None:
            if self._semaphore.locked():
                raise ConcurrencyLimitError(
                    f"Server at capacity ({self._max} concurrent requests)",
                    suggestion="The server is busy. Retry in a moment.",
                )
            async with self._semaphore:
                return await self._execute(ctx, call_next)
        else:
            return await self._execute(ctx, call_next)

    async def _execute(self, ctx: RequestContext, call_next: Callable[..., Any]) -> Any:
        self._active += 1
        self._total += 1
        if self._active > self._peak:
            self._peak = self._active
        try:
            return await call_next(ctx)
        finally:
            self._active -= 1

    @property
    def active_requests(self) -> int:
        """Number of currently in-flight requests."""
        return self._active

    @property
    def total_requests(self) -> int:
        """Total requests processed since server start."""
        return self._total

    @property
    def peak_concurrent(self) -> int:
        """Highest concurrent request count observed."""
        return self._peak

    def stats(self) -> dict[str, int]:
        """Return concurrency statistics."""
        return {
            "active": self._active,
            "total": self._total,
            "peak": self._peak,
            "max_concurrent": self._max,
        }


class PerToolConcurrencyLimiter:
    """Middleware that enforces per-tool concurrency limits.

    Reads ``max_concurrent`` from the ``ToolDef`` stored in
    ``ctx.state["tool_def"]``.  When a tool has ``max_concurrent``
    set and the limit is reached, additional calls receive a
    retryable ``ConcurrencyLimitError`` (code ``CONCURRENCY_LIMIT_EXCEEDED``).

    Added automatically — by the server and by ``TestClient`` — when any
    tool declares ``max_concurrent``.  It is placed just outside the first
    ``CircuitBreakerMiddleware``, so a call refused for capacity never
    reaches the breaker.

    Semaphores are lazily created per tool name on first use.
    """

    def __init__(self) -> None:
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    async def __call__(self, ctx: RequestContext, call_next: Callable[..., Any]) -> Any:
        tool_def = ctx.state.get("tool_def")
        if tool_def is None or not getattr(tool_def, "max_concurrent", None):
            return await call_next(ctx)

        limit = tool_def.max_concurrent
        tool_name = tool_def.name

        # Lazily create semaphore for this tool
        if tool_name not in self._semaphores:
            self._semaphores[tool_name] = asyncio.Semaphore(limit)

        sem = self._semaphores[tool_name]
        if sem.locked():
            raise ConcurrencyLimitError(
                f"Tool '{tool_name}' at capacity ({limit} concurrent calls)"
            )

        async with sem:
            return await call_next(ctx)


def insert_per_tool_limiter(middlewares: list[Any], limiter: PerToolConcurrencyLimiter) -> None:
    """Add *limiter* to *middlewares* (in place) unless one is already there.

    It goes just before the first ``CircuitBreakerMiddleware`` — so a call
    refused for capacity never reaches a breaker — or at the end.
    """
    if any(isinstance(m, PerToolConcurrencyLimiter) for m in middlewares):
        return
    from ._circuit_breaker import CircuitBreakerMiddleware

    index = next(
        (i for i, m in enumerate(middlewares) if isinstance(m, CircuitBreakerMiddleware)),
        len(middlewares),
    )
    middlewares.insert(index, limiter)
