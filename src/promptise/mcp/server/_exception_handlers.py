"""Custom exception handlers for MCP server tools.

Map application exceptions to structured MCP error responses.
Uses MRO-based lookup to find the most specific handler.

Example::

    from promptise.mcp.server import MCPServer

    server = MCPServer(name="api")

    class DatabaseError(Exception):
        pass

    @server.exception_handler(DatabaseError)
    async def handle_db_error(ctx, exc):
        from promptise.mcp.server import ToolError
        return ToolError("Database unavailable", code="DB_ERROR", retryable=True)

    @server.tool()
    async def query(sql: str) -> list:
        raise DatabaseError("connection refused")
        # → returns structured error instead of generic INTERNAL_ERROR
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ._context import RequestContext
from ._errors import MCPError

logger = logging.getLogger("promptise.server")


class ExceptionHandlerRegistry:
    """Registry mapping exception types to handler functions.

    Handlers are matched using the exception's MRO (most specific first).

    A handler receives ``(ctx, exc)`` and must return an ``MCPError``
    instance, or ``None`` to fall through to the generic handler.
    """

    def __init__(self) -> None:
        self._handlers: dict[type[Exception], Callable[..., Any]] = {}

    def register(
        self,
        exc_type: type[Exception],
        handler: Callable[..., Any],
    ) -> None:
        """Register a handler for *exc_type*.

        Args:
            exc_type: The exception class to handle.
            handler: Async callable ``(ctx, exc) -> MCPError | None``.
        """
        self._handlers[exc_type] = handler

    def find_handler(
        self,
        exc: Exception,
    ) -> Callable[..., Any] | None:
        """Find the best handler for *exc* by walking its MRO.

        Returns ``None`` if no handler is registered for any type
        in the exception's MRO.

        An ``MCPError`` already serialises to a structured error, so for one
        only handlers registered for an ``MCPError`` *subclass* (e.g.
        ``CircuitOpenError``, ``RateLimitError``) apply — a catch-all
        ``Exception`` handler does not swallow it.
        """
        structured = isinstance(exc, MCPError)
        for cls in type(exc).__mro__:
            if structured and (cls is MCPError or not issubclass(cls, MCPError)):
                return None
            if cls in self._handlers:
                return self._handlers[cls]
        return None

    async def handle(
        self,
        ctx: RequestContext,
        exc: Exception,
    ) -> MCPError | None:
        """Attempt to handle *exc*.

        Returns the ``MCPError`` produced by the matched handler,
        or ``None`` if no handler matched.
        """
        handler = self.find_handler(exc)
        if handler is None:
            return None

        import asyncio

        result = handler(ctx, exc)
        if asyncio.iscoroutine(result):
            result = await result

        if isinstance(result, MCPError):
            return result

        logger.warning(
            "Exception handler for %s returned %s instead of MCPError",
            type(exc).__name__,
            type(result).__name__,
        )
        return None

    def __len__(self) -> int:
        return len(self._handlers)

    def __contains__(self, exc_type: type[Exception]) -> bool:
        return exc_type in self._handlers
