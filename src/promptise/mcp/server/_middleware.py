"""Middleware protocol and chain for MCP server request processing.

Middleware functions intercept tool/resource/prompt calls, enabling
cross-cutting concerns like auth, rate limiting, logging, and timeouts.

A middleware is any async callable with signature::

    async def my_middleware(ctx: RequestContext, call_next) -> Any:
        # pre-processing
        result = await call_next(ctx)
        # post-processing
        return result
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ._context import RequestContext

logger = logging.getLogger("promptise.server")


@runtime_checkable
class Middleware(Protocol):
    """Protocol for middleware callables."""

    async def __call__(self, ctx: RequestContext, call_next: Callable[..., Any]) -> Any: ...


class MiddlewareChain:
    """Executes a stack of middleware around a handler.

    Middleware runs in registration order (first added = outermost).
    The innermost function is the actual handler.
    """

    def __init__(self, middlewares: list[Any] | None = None) -> None:
        self._middlewares: list[Any] = list(middlewares or [])

    def add(self, middleware: Any) -> None:
        """Append a middleware to the chain."""
        self._middlewares.append(middleware)

    async def run(
        self,
        ctx: RequestContext,
        handler: Callable[..., Any],
        handler_kwargs: dict[str, Any],
    ) -> Any:
        """Execute the middleware chain, finishing with *handler*."""

        async def _final(ctx: RequestContext) -> Any:
            result = handler(**handler_kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        # Build chain from inside out
        next_fn = _final
        for mw in reversed(self._middlewares):
            next_fn = _make_link(mw, next_fn)

        return await next_fn(ctx)

    def __len__(self) -> int:
        return len(self._middlewares)


def compile_middleware_chain(
    middlewares: list[Any],
) -> Callable[[RequestContext, Callable[..., Any], dict[str, Any]], Any]:
    """Pre-compile a middleware stack into a single callable.

    Returns a coroutine function ``(ctx, handler, kwargs) -> result``
    that avoids re-building the closure chain on every request.  The
    middleware list is captured once at compile time.

    When there are no middlewares the returned function calls the handler
    directly (zero overhead).
    """
    if not middlewares:

        async def _direct(
            ctx: RequestContext,
            handler: Callable[..., Any],
            kwargs: dict[str, Any],
        ) -> Any:
            result = handler(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        return _direct

    # Freeze the middleware list so mutations don't affect the compiled chain
    frozen = tuple(middlewares)

    async def _compiled(
        ctx: RequestContext,
        handler: Callable[..., Any],
        kwargs: dict[str, Any],
    ) -> Any:
        async def _final(ctx: RequestContext) -> Any:
            result = handler(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        next_fn = _final
        for mw in reversed(frozen):
            next_fn = _make_link(mw, next_fn)
        return await next_fn(ctx)

    return _compiled


def _make_link(
    mw: Any,
    next_fn: Callable[..., Any],
) -> Callable[..., Any]:
    """Create a chain link that calls *mw* with *next_fn* as ``call_next``."""

    async def link(ctx: RequestContext) -> Any:
        return await mw(ctx, next_fn)

    return link


# =====================================================================
# Built-in middleware
# =====================================================================


class LoggingMiddleware:
    """Log every tool call with timing.

    Args:
        log_level: Logging level (default ``logging.INFO``).
    """

    def __init__(self, log_level: int = logging.INFO) -> None:
        self._level = log_level

    async def __call__(self, ctx: RequestContext, call_next: Callable[..., Any]) -> Any:
        start = time.perf_counter()
        try:
            result = await call_next(ctx)
            elapsed = time.perf_counter() - start
            ctx.logger.log(
                self._level,
                "[%s] %s completed in %.3fs",
                ctx.request_id,
                ctx.tool_name,
                elapsed,
            )
            return result
        except Exception:
            elapsed = time.perf_counter() - start
            ctx.logger.log(
                self._level,
                "[%s] %s failed after %.3fs",
                ctx.request_id,
                ctx.tool_name,
                elapsed,
            )
            raise


class TimeoutMiddleware:
    """Enforce per-call timeout.

    Uses the tool's ``timeout`` setting if present, otherwise falls
    back to *default_timeout*.

    Args:
        default_timeout: Default timeout in seconds.
    """

    def __init__(self, default_timeout: float = 30.0) -> None:
        self._default_timeout = default_timeout

    async def __call__(self, ctx: RequestContext, call_next: Callable[..., Any]) -> Any:
        tool_def = ctx.state.get("tool_def")
        timeout = tool_def.timeout if tool_def and tool_def.timeout else self._default_timeout
        try:
            return await asyncio.wait_for(call_next(ctx), timeout=timeout)
        except asyncio.TimeoutError:
            raise tool_timeout_error(ctx.tool_name, timeout) from None


def tool_timeout_error(tool_name: str, timeout: float) -> Any:
    """The retryable ``TIMEOUT`` error for a call that ran past *timeout*."""
    from ._errors import ToolError

    return ToolError(
        f"Tool '{tool_name}' timed out after {timeout}s",
        code="TIMEOUT",
        retryable=True,
        suggestion=(
            f"The tool did not finish within {timeout}s. It may depend on a slow "
            "or unavailable service: retry later, or continue without it."
        ),
    )


def with_tool_timeout(
    handler: Callable[..., Any], timeout: float, tool_name: str
) -> Callable[..., Any]:
    """Bound *handler* by a ``@server.tool(timeout=...)`` limit.

    Applied by the server (and ``TestClient``) to every tool that declares
    a timeout, with or without :class:`TimeoutMiddleware`.  Only the tool's
    own run is timed — not middleware in front of it, such as an approval
    gate waiting for a human.  A synchronous handler blocks the event loop
    and cannot be interrupted, so the limit only applies to ``async`` tools.
    """

    async def _timed(**kwargs: Any) -> Any:
        async def _run() -> Any:
            result = handler(**kwargs)
            if asyncio.iscoroutine(result):
                result = await result
            return result

        try:
            return await asyncio.wait_for(_run(), timeout=timeout)
        except asyncio.TimeoutError:
            raise tool_timeout_error(tool_name, timeout) from None

    return _timed
