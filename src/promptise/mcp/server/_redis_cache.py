"""Redis cache backend for MCP servers.

Provides a ``RedisCache`` implementation of ``CacheBackend`` for
distributed caching across multiple server instances.

Requires the ``redis`` package (optional dependency).

Example::

    from promptise.mcp.server import MCPServer, cached
    from promptise.mcp.server import RedisCache

    cache = RedisCache(url="redis://localhost:6379/0")
    server = MCPServer(name="api")

    @server.tool()
    @cached(ttl=300, backend=cache)
    async def expensive_query(q: str) -> dict:
        return await db.search(q)
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("promptise.server")


class RedisCache:
    """Redis-backed cache implementing the ``CacheBackend`` protocol.

    Uses ``redis.asyncio`` for async Redis operations.  Values are
    JSON-serialised for storage.

    If ``redis`` is not installed, all operations raise ``ImportError``
    on first use.

    Args:
        url: Redis connection URL (e.g. ``"redis://localhost:6379/0"``).
        prefix: Key prefix to namespace cache entries
            (default ``"promptise:"``).
        client: Pre-configured ``redis.asyncio.Redis`` client.
            If provided, ``url`` is ignored.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        *,
        prefix: str = "promptise:",
        client: Any = None,
    ) -> None:
        self._url = url
        self._prefix = prefix
        self._client = client
        self._own_client = client is None

    async def _get_client(self) -> Any:
        """Lazy-init the Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as aioredis
            except ImportError:
                raise ImportError(
                    "redis is required for RedisCache. Install with: pip install redis"
                )
            self._client = aioredis.from_url(self._url)
        return self._client

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def get(self, key: str) -> Any | None:
        """Get a cached value by key."""
        client = await self._get_client()
        raw = await client.get(self._key(key))
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return None

    async def set(self, key: str, value: Any, ttl: float) -> None:
        """Set a cached value with TTL (millisecond precision)."""
        if ttl <= 0:
            return  # already expired; Redis rejects a non-positive expiry
        client = await self._get_client()
        raw = json.dumps(value, default=str)
        await client.set(self._key(key), raw, px=max(1, round(ttl * 1000)))

    async def delete(self, key: str) -> None:
        """Delete a cached value."""
        client = await self._get_client()
        await client.delete(self._key(key))

    async def clear(self) -> None:
        """Clear all cached values with this prefix."""
        client = await self._get_client()
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match=f"{self._prefix}*", count=100)
            if keys:
                await client.delete(*keys)
            if cursor == 0:
                break

    async def close(self) -> None:
        """Close the Redis connection (if we own it)."""
        if self._client and self._own_client:
            await self._client.aclose()
            self._client = None
