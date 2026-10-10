"""Caching layer for MCP server tool results.

Provides an in-memory cache backend, a ``@cached`` decorator for tools,
and a ``CacheMiddleware`` for server-wide caching.  Both scope entries to
the authenticated caller by default (see :data:`CacheScope`).

Example::

    from promptise.mcp.server import MCPServer
    from promptise.mcp.server._cache import InMemoryCache, cached

    cache = InMemoryCache()

    @server.tool()
    @cached(ttl=300, backend=cache)
    async def expensive_query(query: str) -> dict:
        return await db.slow_search(query)
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, runtime_checkable

logger = logging.getLogger("promptise.server")


@runtime_checkable
class CacheBackend(Protocol):
    """Protocol for cache backends."""

    async def get(self, key: str) -> Any | None:
        """Get a cached value, or ``None`` if not found / expired."""
        ...

    async def set(self, key: str, value: Any, ttl: float) -> None:
        """Store a value with a TTL in seconds."""
        ...

    async def delete(self, key: str) -> None:
        """Delete a cached value."""
        ...

    async def clear(self) -> None:
        """Clear all cached values."""
        ...


class InMemoryCache:
    """In-process cache with TTL-based expiry and background cleanup.

    Thread-safe for asyncio (single-threaded event loop).

    Args:
        max_size: Maximum number of entries.  When the limit is reached
            the least recently used entry (the one read or written longest
            ago) is evicted.  ``0`` means unlimited.
        cleanup_interval: Seconds between background sweeps of expired
            entries.  ``0`` disables background cleanup (expired entries
            are still removed on access).
    """

    def __init__(
        self,
        *,
        max_size: int = 0,
        cleanup_interval: float = 60.0,
    ) -> None:
        self._store: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._max_size = max_size
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: asyncio.Task[None] | None = None
        self._evicted_count: int = 0

    async def get(self, key: str) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        self._store.move_to_end(key)  # most recently used
        return value

    async def set(self, key: str, value: Any, ttl: float) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        elif self._max_size > 0 and len(self._store) >= self._max_size:
            # Evict the least recently used entry
            self._store.popitem(last=False)
        self._store[key] = (value, time.monotonic() + ttl)
        # Lazily start the background cleanup on first write
        self._ensure_cleanup_running()

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def clear(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        """Current number of entries (including expired)."""
        return len(self._store)

    @property
    def evicted_count(self) -> int:
        """Total number of entries removed by background cleanup."""
        return self._evicted_count

    # ------------------------------------------------------------------
    # Background cleanup
    # ------------------------------------------------------------------

    def _ensure_cleanup_running(self) -> None:
        """Start the background sweep task if not already running."""
        if self._cleanup_interval <= 0:
            return
        if self._cleanup_task is not None and not self._cleanup_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
            self._cleanup_task = loop.create_task(self._cleanup_loop())
        except RuntimeError:
            # No running loop — skip (e.g. sync tests).
            pass

    async def _cleanup_loop(self) -> None:
        """Periodically sweep expired entries from the store."""
        try:
            while True:
                await asyncio.sleep(self._cleanup_interval)
                self._sweep_expired()
        except asyncio.CancelledError:
            pass

    def _sweep_expired(self) -> int:
        """Remove all expired entries.  Returns number removed."""
        now = time.monotonic()
        expired_keys = [k for k, (_, expires_at) in self._store.items() if now > expires_at]
        for k in expired_keys:
            del self._store[k]
        if expired_keys:
            self._evicted_count += len(expired_keys)
            logger.debug("Cache cleanup: removed %d expired entries", len(expired_keys))
        return len(expired_keys)

    async def stop_cleanup(self) -> None:
        """Cancel the background cleanup task (for graceful shutdown)."""
        if self._cleanup_task is not None and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None


# Default shared cache instance
_default_cache = InMemoryCache()


class _LeaderGone(Exception):
    """The call computing a coalesced result was cancelled."""


class _SingleFlight:
    """Collapses concurrent computations of the same key into one.

    The first caller for a key computes; callers arriving while it runs
    await its result (or its exception) instead of starting their own.
    If the computing call is cancelled, a waiting caller takes over.
    """

    def __init__(self) -> None:
        self._inflight: dict[str, asyncio.Future[Any]] = {}

    async def run(self, key: str, compute: Callable[[], Awaitable[Any]]) -> Any:
        while (pending := self._inflight.get(key)) is not None:
            try:
                return await asyncio.shield(pending)
            except _LeaderGone:
                continue

        flight: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._inflight[key] = flight
        try:
            result = await compute()
        except asyncio.CancelledError:
            flight.set_exception(_LeaderGone())
            raise
        except BaseException as exc:
            flight.set_exception(exc)
            raise
        else:
            flight.set_result(result)
            return result
        finally:
            del self._inflight[key]
            if flight.done() and not flight.cancelled():
                flight.exception()  # mark retrieved: there may be no waiters


CacheScope = Literal["client", "tenant", "shared"]
"""Who may share a cached result.

* ``"client"`` *(default)* — only the same authenticated principal
  (issuer + tenant + client id).  Safe for any tool, including ones whose
  result depends on who is asking.
* ``"tenant"`` — every client of the same tenant.  Use for data that is
  tenant-wide and the same for every user in the tenant.
* ``"shared"`` — everyone.  Use only for data that is the same for every
  caller (public reference data, weather, exchange rates).
"""

_SCOPES = ("client", "tenant", "shared")


def _validate_scope(scope: str) -> None:
    if scope not in _SCOPES:
        raise ValueError(f"cache scope must be one of {_SCOPES}, got {scope!r}")


def _digest(value: Any) -> str:
    """SHA-256 of *value*'s canonical JSON form."""
    serialised = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(serialised.encode()).hexdigest()


def _scope_part(scope: str, ctx: Any | None) -> str:
    """The key component that keeps one caller's entries from another's."""
    if scope == "shared":
        return "shared"
    client = getattr(ctx, "client", None)
    if client is None:
        # Outside a request (a handler called directly): nobody to share with.
        return "no-request"
    if scope == "tenant":
        return "tenant:" + _digest([client.tenant_id])
    client_id = getattr(ctx, "client_id", None) or client.client_id
    return "client:" + _digest([client.issuer, client.tenant_id, client_id])


def _is_authenticated(ctx: Any) -> bool:
    """Whether auth middleware has already identified the caller."""
    return getattr(ctx, "client_id", None) is not None


def _make_cache_key(name: str, arguments: dict[str, Any], scope_part: str = "shared") -> str:
    """Build a deterministic cache key from a name, the arguments and the scope."""
    return f"cache:{name}:{scope_part}:{_digest(arguments)}"


def _user_arguments(kwargs: dict[str, Any], ctx: Any | None) -> dict[str, Any]:
    """The tool's own arguments, without objects the framework injected.

    ``RequestContext``, ``BackgroundTasks``, ``Depends`` values and the like
    differ on every request, so keying on them means the cache never hits.
    Inside a tool call the validated user arguments are known exactly; a
    direct call falls back to dropping the request context.
    """
    from ._context import RequestContext

    tool_args = ctx.state.get("_tool_arguments") if ctx is not None else None
    if isinstance(tool_args, dict):
        return {k: v for k, v in kwargs.items() if k in tool_args}
    return {k: v for k, v in kwargs.items() if not isinstance(v, RequestContext)}


def cached(
    ttl: float = 60.0,
    *,
    key_func: Callable[..., str] | None = None,
    backend: CacheBackend | None = None,
    scope: CacheScope = "client",
    coalesce: bool = True,
) -> Callable[..., Any]:
    """Decorator that caches tool handler results.

    Entries are scoped to the calling principal by default, so a result
    computed for one client is never returned to another — even when the
    handler reads the caller with :func:`get_context` rather than taking
    a ``ctx`` parameter.  Injected parameters (``ctx: RequestContext``,
    ``Depends(...)`` values, ``BackgroundTasks`` ...) are not part of the
    key.

    Identical calls that arrive while the first one is still computing
    wait for its result instead of each calling the upstream service
    (request coalescing, on by default), so a burst against a cold or
    just-expired entry costs one upstream call, not one per caller.
    Coalescing is per process; a ``RedisCache`` shared by several
    replicas still sees one upstream call per replica.

    Args:
        ttl: Time-to-live in seconds.
        key_func: Custom key function ``(func_name, kwargs) -> str``.
            Receives every keyword argument the handler gets (injected ones
            included).  Its key is still prefixed with the *scope*, so a
            custom key cannot widen sharing beyond it.
        backend: Cache backend.  Defaults to the module-level
            ``InMemoryCache`` singleton.
        scope: Who may share an entry: ``"client"`` (default), ``"tenant"``
            or ``"shared"`` — see :data:`CacheScope`.  Widen it only for
            data that does not depend on the caller.
        coalesce: Share one in-flight computation between identical
            concurrent calls (default ``True``).  Concurrent callers also
            share its error, if it raises.

    Example::

        @server.tool()
        @cached(ttl=300)
        async def my_open_tickets(ctx: RequestContext) -> list[dict]:
            return await db.tickets(owner=ctx.client.client_id)

        @server.tool()
        @cached(ttl=300, scope="shared")
        async def exchange_rate(currency: str) -> float:
            return await fx.rate(currency)
    """
    _validate_scope(scope)
    cache = backend or _default_cache

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        name = f"{func.__module__}.{func.__qualname__}"
        flights = _SingleFlight()

        @functools.wraps(func)
        async def wrapper(**kwargs: Any) -> Any:
            from ._context import _current_context

            ctx = _current_context.get()
            scope_part = _scope_part(scope, ctx)
            # The shared default backend serves every server in the process.
            server = ctx.server_name if ctx is not None else ""
            if key_func is not None:
                cache_key = f"cache:{server}:{scope_part}:{key_func(func.__name__, kwargs)}"
            else:
                cache_key = _make_cache_key(
                    f"{server}:{name}", _user_arguments(kwargs, ctx), scope_part
                )

            # Try cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value

            async def compute() -> Any:
                result = func(**kwargs)
                if asyncio.iscoroutine(result):
                    result = await result
                await cache.set(cache_key, result, ttl)
                return result

            if coalesce:
                return await flights.run(cache_key, compute)
            return await compute()

        # Attach cache reference for testing
        wrapper.cache = cache  # type: ignore[attr-defined]
        return wrapper

    return decorator


def _cacheable(tool_def: Any) -> bool:
    """Whether ``CacheMiddleware`` may cache *tool_def*'s results."""
    if tool_def is None:
        return True
    if getattr(tool_def, "cache", True) is False:
        return False
    annotations = getattr(tool_def, "annotations", None)
    return getattr(annotations, "destructive_hint", None) is not True


class CacheMiddleware:
    """Server-wide caching middleware.

    Caches tool results, resource reads and prompt results, keyed on the
    server, request type, name, resource URI, arguments and — by default —
    the authenticated caller, so one client never receives a result
    computed for another.  Note that it caches *every* tool that doesn't
    opt out: mark tools that change data or must always be fresh with
    ``@server.tool(cache=False)`` (tools annotated
    ``destructive_hint=True`` are skipped automatically), or use
    :func:`cached` on just the handlers you want cached.

    A cache hit still runs the definition's guards (``HasRole``,
    ``HasTenant``, ...), so a cached result is never returned to a caller
    the tool, resource or prompt would refuse.  When it requires
    authentication but the caller has not been identified yet
    (``CacheMiddleware`` added *before* ``AuthMiddleware``), the request
    bypasses the cache — whatever the scope — so a hit can never skip
    authentication: add ``AuthMiddleware`` first.

    Args:
        backend: Cache backend.
        ttl: Default TTL in seconds.
        scope: Who may share an entry: ``"client"`` (default), ``"tenant"``
            or ``"shared"`` — see :data:`CacheScope`.
    """

    def __init__(
        self,
        backend: CacheBackend | None = None,
        *,
        ttl: float = 60.0,
        scope: CacheScope = "client",
    ) -> None:
        _validate_scope(scope)
        self.cache = backend or InMemoryCache()
        self.ttl = ttl
        self.scope = scope
        self._warned_order = False

    async def __call__(
        self,
        ctx: Any,
        call_next: Callable[..., Any],
    ) -> Any:
        tool_def = ctx.state.get("tool_def")
        if not _cacheable(tool_def):
            return await call_next(ctx)
        guards = getattr(tool_def, "guards", None)
        if getattr(tool_def, "auth", False) and not _is_authenticated(ctx):
            # The caller is not identified yet, so a hit here would skip
            # authentication (and, for a non-shared scope, could not tell
            # callers apart).  Bypass the cache.
            if not self._warned_order:
                self._warned_order = True
                logger.warning(
                    "CacheMiddleware runs before authentication, so it cannot tell "
                    "callers apart; caching is skipped for authenticated tools, "
                    "resources and prompts. Add AuthMiddleware before CacheMiddleware."
                )
            return await call_next(ctx)

        # The validated arguments (or a resource template's parameters).
        # Resource reads and prompt requests are namespaced by request type
        # and keyed on the URI too, so a tool and a prompt sharing a name,
        # or two URIs of one template, never share an entry.
        arguments = ctx.state.get("_tool_arguments", ctx.state.get("arguments", {}))
        kind = getattr(ctx, "request_type", "tool")
        if kind == "tool":
            name = f"mw:{ctx.server_name}:{ctx.tool_name}"
            keyed: Any = arguments
        else:
            name = f"mw:{ctx.server_name}:{kind}:{ctx.tool_name}"
            keyed = {"uri": ctx.state.get("resource_uri"), "args": arguments}
        final_key = _make_cache_key(name, keyed, _scope_part(self.scope, ctx))

        cached_value = await self.cache.get(final_key)
        if cached_value is not None:
            if guards:
                from ._testing import check_guards

                await check_guards(guards, ctx)
            return cached_value

        result = await call_next(ctx)
        await self.cache.set(final_key, result, self.ttl)
        return result
