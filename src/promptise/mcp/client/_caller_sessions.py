"""Per-caller MCP sessions for HTTP servers.

An agent serves many users, but each ``MCPClient`` opens one session with
fixed headers.  To send *each caller's* bearer token, the pool keeps one
session per ``(server, token)`` pair: concurrent invocations by different
users never share a session or a header dict, so one user's token can never
ride on another user's request.

Sessions are opened on first use, shared by concurrent calls that carry the
same token, and closed when they have been idle for ``idle_timeout`` seconds
or when the pool holds more than ``max_sessions`` (least recently used
first).  A session that is still serving a call is never closed under it.
Tokens are never stored as keys: entries are keyed on a SHA-256 digest.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from ._client import MCPClient, MCPClientError

logger = logging.getLogger("promptise.mcp.client")


class _Entry:
    """One pooled session and the calls currently using it."""

    __slots__ = ("client", "connected", "lock", "refs", "last_used", "stale")

    def __init__(self, client: MCPClient) -> None:
        self.client = client
        self.connected = False
        self.lock = asyncio.Lock()
        self.refs = 0
        self.last_used = time.monotonic()
        # Set when the session failed mid-call; the next lease opens a new one.
        self.stale = False


class CallerSessionPool:
    """Bounded pool of per-caller sessions, keyed by server and token.

    Args:
        max_sessions: Most idle sessions kept open across all servers.
            Sessions serving a call are never evicted, so the pool can
            briefly exceed this under load.
        idle_timeout: Seconds after which an unused session is closed.
    """

    def __init__(self, *, max_sessions: int = 256, idle_timeout: float = 300.0) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least 1")
        self._max_sessions = max_sessions
        self._idle_timeout = idle_timeout
        self._entries: OrderedDict[tuple[str, str], _Entry] = OrderedDict()
        self._closing: set[asyncio.Task[None]] = set()
        self._closed = False

    def __len__(self) -> int:
        return len(self._entries)

    @staticmethod
    def _key(server_name: str, bearer_token: str) -> tuple[str, str]:
        return server_name, hashlib.sha256(bearer_token.encode()).hexdigest()

    @asynccontextmanager
    async def lease(
        self, server_name: str, base: MCPClient, bearer_token: str
    ) -> AsyncIterator[MCPClient]:
        """Yield a connected client that authenticates as *bearer_token*.

        Raises:
            MCPConnectionRejectedError: The server refused the token.
            MCPClientError: The pool is closed or the connection failed.
        """
        if self._closed:
            raise MCPClientError("The MCP client is closed")
        key = self._key(server_name, bearer_token)
        entry = self._entries.get(key)
        if entry is None or entry.stale:
            entry = _Entry(base.with_bearer_token(bearer_token))
            self._entries[key] = entry
        self._entries.move_to_end(key)
        entry.refs += 1
        try:
            self._evict()
            async with entry.lock:
                if not entry.connected:
                    try:
                        await entry.client.__aenter__()
                    except BaseException:
                        entry.stale = True
                        raise
                    entry.connected = True
            try:
                yield entry.client
            except MCPClientError:
                entry.stale = True
                raise
        finally:
            entry.refs -= 1
            entry.last_used = time.monotonic()
            if entry.refs == 0 and (entry.stale or self._entries.get(key) is not entry):
                if self._entries.get(key) is entry:
                    del self._entries[key]
                self._close_later(entry)

    def _evict(self) -> None:
        """Close idle-expired sessions, then the oldest idle ones over the cap."""
        now = time.monotonic()
        for key, entry in list(self._entries.items()):
            if entry.refs == 0 and now - entry.last_used > self._idle_timeout:
                del self._entries[key]
                self._close_later(entry)
        if len(self._entries) <= self._max_sessions:
            return
        for key, entry in list(self._entries.items()):
            if len(self._entries) <= self._max_sessions:
                break
            if entry.refs == 0:
                del self._entries[key]
                self._close_later(entry)

    def _close_later(self, entry: _Entry) -> None:
        """Close *entry*'s session in the background (close can take a while)."""
        if not entry.connected:
            return
        entry.connected = False
        task = asyncio.ensure_future(self._close(entry.client))
        self._closing.add(task)
        task.add_done_callback(self._closing.discard)

    @staticmethod
    async def _close(client: MCPClient) -> None:
        try:
            await client.__aexit__(None, None, None)
        except BaseException:  # noqa: BLE001 — best-effort cleanup
            logger.debug("Error closing a per-caller MCP session", exc_info=True)

    async def aclose(self) -> None:
        """Close every session and wait for the closes to finish."""
        self._closed = True
        entries = list(self._entries.values())
        self._entries.clear()
        for entry in entries:
            self._close_later(entry)
        if self._closing:
            await asyncio.gather(*list(self._closing), return_exceptions=True)
