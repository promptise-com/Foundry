"""Token bucket rate limiter for MCP server tools.

Example::

    from promptise.mcp.server import MCPServer, RateLimitMiddleware, TokenBucketLimiter

    limiter = TokenBucketLimiter(rate_per_minute=100, burst=20)
    server = MCPServer(name="api")
    server.add_middleware(RateLimitMiddleware(limiter))
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from ._context import RequestContext
from ._errors import RateLimitError


def _compose_key(*parts: str) -> str:
    """Join key parts injectively.

    Length-prefixes each part (``"<len>:<part>"``) so a ``:`` *inside* a
    part (e.g. a URN tenant ``"org:acme"`` or provider-prefixed client id
    ``"okta:bob"`` — server-side ids come straight from JWT claims / API-key
    config and are not colon-validated) can never merge with the joiner and
    collide two distinct ``(tenant, client, tool)`` triples onto one bucket.
    """
    return "|".join(f"{len(p)}:{p}" for p in parts)


@runtime_checkable
class RateLimitStrategy(Protocol):
    """Protocol for rate limiting strategies."""

    def consume(self, key: str) -> tuple[bool, float]:
        """Try to consume a token.

        Returns:
            ``(allowed, retry_after_seconds)``
        """
        ...


#: Time units accepted in declared rate-limit strings (``"100/min"``).
_UNIT_SECONDS: dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "m": 60.0,
    "min": 60.0,
    "minute": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
}


def parse_rate_limit(spec: str) -> tuple[float, int]:
    """Parse a declared rate-limit string like ``"100/min"``.

    Accepted formats: ``"<count>/<unit>"`` where unit is one of
    ``s|sec|second``, ``m|min|minute``, ``h|hr|hour``.

    Args:
        spec: The declared limit, e.g. ``"100/min"``, ``"10/sec"``, ``"500/hour"``.

    Returns:
        ``(rate_per_minute, burst)`` — the sustained refill rate normalised
        to tokens-per-minute, and the bucket capacity (the declared count).

    Raises:
        ValueError: If the spec is malformed, so a typo fails at tool
            registration instead of silently never limiting.
    """
    raw = spec.strip().lower()
    count_part, sep, unit_part = raw.partition("/")
    if not sep:
        raise ValueError(f"Invalid rate_limit {spec!r}: expected '<count>/<unit>' (e.g. '100/min')")
    try:
        count = int(count_part.strip())
    except ValueError:
        raise ValueError(f"Invalid rate_limit {spec!r}: count must be an integer") from None
    if count <= 0:
        raise ValueError(f"Invalid rate_limit {spec!r}: count must be positive")
    unit = unit_part.strip()
    if unit not in _UNIT_SECONDS:
        raise ValueError(
            f"Invalid rate_limit {spec!r}: unknown unit {unit!r} "
            f"(use s/sec/second, m/min/minute, or h/hr/hour)"
        )
    rate_per_minute = count * (60.0 / _UNIT_SECONDS[unit])
    return rate_per_minute, count


class TokenBucketLimiter:
    """Token bucket rate limiter.

    Args:
        rate_per_minute: Sustained rate (tokens refilled per minute).
        burst: Maximum burst size (bucket capacity).
    """

    _MAX_BUCKETS = 10_000  # Evict stale entries beyond this count

    def __init__(self, rate_per_minute: float = 60, burst: int | None = None) -> None:
        self._rate = rate_per_minute / 60.0  # tokens per second
        self._burst = float(burst or rate_per_minute)
        self._buckets: dict[str, tuple[float, float]] = {}  # key → (tokens, last_time)
        self._lock = threading.Lock()

    def consume(self, key: str) -> tuple[bool, float]:
        """Try to consume one token from the bucket for *key*."""
        now = time.monotonic()
        with self._lock:
            tokens, last_time = self._buckets.get(key, (self._burst, now))

            # Refill tokens based on elapsed time
            elapsed = now - last_time
            tokens = min(self._burst, tokens + elapsed * self._rate)

            if tokens >= 1.0:
                self._buckets[key] = (tokens - 1.0, now)
                allowed, retry = True, 0.0
            else:
                deficit = 1.0 - tokens
                retry = deficit / self._rate if self._rate > 0 else 60.0
                self._buckets[key] = (tokens, now)
                allowed, retry = False, retry

            # Evict stale buckets periodically or when the map grows too large.
            # Remove entries not accessed in the last 5 minutes.
            _last_cleanup = getattr(self, "_last_cleanup", 0.0)
            if len(self._buckets) > self._MAX_BUCKETS or (
                now - _last_cleanup > 60.0 and len(self._buckets) > 100
            ):
                stale_cutoff = now - 300.0
                stale_keys = [k for k, (_, lt) in self._buckets.items() if lt < stale_cutoff]
                for k in stale_keys:
                    del self._buckets[k]
                self._last_cleanup = now  # type: ignore[attr-defined]

            return allowed, retry


class RateLimitMiddleware:
    """Middleware that enforces rate limits.

    Args:
        limiter: A ``RateLimitStrategy`` implementation. If ``None``,
            a ``TokenBucketLimiter`` is created with the given params.
        per_tool: If ``True``, rate limit per tool name (in addition
            to per client).
        key_func: Custom function to extract the rate limit key from
            context. Overrides ``per_tool`` behaviour.
        rate_per_minute: Default rate if creating a new limiter.
        burst: Default burst if creating a new limiter.
    """

    def __init__(
        self,
        limiter: Any | None = None,
        *,
        per_tool: bool = False,
        key_func: Callable[[RequestContext], str] | None = None,
        rate_per_minute: int = 60,
        burst: int | None = None,
    ) -> None:
        self._limiter = limiter or TokenBucketLimiter(rate_per_minute, burst)
        self._per_tool = per_tool
        self._key_func = key_func

    def _get_key(self, ctx: RequestContext) -> str:
        if self._key_func:
            return self._key_func(ctx)
        parts: list[str] = []
        # Tenant is part of the key when present — buckets never span
        # tenants, so one tenant's traffic cannot exhaust another's quota.
        tenant = getattr(getattr(ctx, "client", None), "tenant_id", None)
        if tenant:
            parts.append(f"tenant={tenant}")
        parts.append(ctx.client_id or "global")
        if self._per_tool:
            parts.append(ctx.tool_name)
        return _compose_key(*parts)

    async def __call__(self, ctx: RequestContext, call_next: Callable[..., Any]) -> Any:
        key = self._get_key(ctx)
        allowed, retry_after = self._limiter.consume(key)
        if not allowed:
            raise RateLimitError(retry_after=retry_after)
        return await call_next(ctx)


class DeclaredRateLimitMiddleware:
    """Enforces per-tool rate limits declared via ``@server.tool(rate_limit=...)``.

    Also enforces ``rate_limit=`` on ``@server.resource`` /
    ``@server.resource_template`` / ``@server.prompt``.

    Auto-inserted into the middleware chain at build time when any registered
    tool declares a ``rate_limit`` — no manual wiring required (mirrors
    ``PerToolConcurrencyLimiter``).  Each declaring tool gets its own token
    bucket sized from its spec; buckets are keyed per client when
    authentication populates ``ctx.client_id``, falling back to a single
    shared bucket for unauthenticated tools.

    Coexists with a user-installed :class:`RateLimitMiddleware`: that one
    enforces a server-wide policy, this one enforces each tool's declared
    contract.
    """

    def __init__(self) -> None:
        self._limiters: dict[str, TokenBucketLimiter] = {}
        self._lock = threading.Lock()

    def _limiter_for(self, tool_name: str, spec: str) -> TokenBucketLimiter:
        limiter = self._limiters.get(tool_name)
        if limiter is None:
            with self._lock:
                limiter = self._limiters.get(tool_name)
                if limiter is None:
                    rate_per_minute, burst = parse_rate_limit(spec)
                    limiter = TokenBucketLimiter(rate_per_minute=rate_per_minute, burst=burst)
                    self._limiters[tool_name] = limiter
        return limiter

    async def __call__(self, ctx: RequestContext, call_next: Callable[..., Any]) -> Any:
        tool_def = ctx.state.get("tool_def")
        spec = getattr(tool_def, "rate_limit", None)
        if not spec:
            return await call_next(ctx)
        # Tools keep their name as the bucket key; resources and prompts are
        # namespaced so a prompt and a tool with one name never share a bucket.
        kind = getattr(ctx, "request_type", "tool")
        limiter = self._limiter_for(
            ctx.tool_name if kind == "tool" else f"{kind}:{ctx.tool_name}", spec
        )
        # Buckets never span tenants: tenant-qualify the per-client key, with
        # an injective (length-prefixed) join so a colon/pipe inside a tenant
        # or client id cannot collide two distinct tenants onto one bucket.
        # BOTH branches go through _compose_key — it is uniquely decodable
        # across arities, so the 1-part untenanted keyspace is disjoint from
        # the 2-part tenanted one (an untenanted client_id can never forge a
        # tenanted key).
        tenant = getattr(getattr(ctx, "client", None), "tenant_id", None)
        client = ctx.client_id or "global"
        key = _compose_key(f"tenant={tenant}", client) if tenant else _compose_key(client)
        allowed, retry_after = limiter.consume(key)
        if not allowed:
            raise RateLimitError(
                f"Rate limit exceeded for {kind} {ctx.tool_name!r} (declared limit: {spec})",
                retry_after=retry_after,
            )
        return await call_next(ctx)
