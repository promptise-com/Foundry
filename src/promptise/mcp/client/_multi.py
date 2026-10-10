"""Multi-server MCP client that aggregates tools from multiple servers.

Connects to N servers, discovers tools from each, and presents a
unified tool list.  Routes ``call_tool`` to the correct server based
on tool discovery.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, Tool

from ._caller_sessions import CallerSessionPool
from ._client import (
    MCPClient,
    MCPClientError,
    MCPConnectionRejectedError,
    _sdk_supports_elicitation,
)

if TYPE_CHECKING:
    from mcp.client.session import ElicitationFnT
    from mcp.shared.session import ProgressFnT

logger = logging.getLogger("promptise.mcp.client")


class MCPMultiClient:
    """Connect to multiple MCP servers and aggregate their tools.

    Each server gets its own ``MCPClient`` with independent auth / headers.
    Tools are tracked per-server so ``call_tool`` routes to the correct one.

    **Tool name collisions**: If two servers expose a tool with the same
    name, the last-discovered server wins and a warning is logged.
    Consider using server-specific prefixes on your MCP servers to avoid
    collisions.

    **Per-caller tokens**: ``call_tool(..., bearer_token=...)`` sends that
    token instead of the client's own credentials.  Each distinct token
    gets its own session to an HTTP/SSE server (opened on first use,
    reused by later calls with the same token), so concurrent calls for
    different users never share headers.  stdio servers have no request
    headers: the token is ignored there and a warning is logged once.

    **Server restarts**: when a server loses the session (restart,
    redeploy), its ``MCPClient`` opens a new one and retries the call once;
    this client then re-lists that server's tools so routing follows the
    tools the new deployment serves.

    Args:
        clients: Mapping of server name → ``MCPClient`` instance.
        elicitation_callback: Default handler for MCP elicitation requests,
            installed on every client that was not given its own
            ``elicitation_callback``.  Same signature as
            :class:`MCPClient`'s.  The SDK callback does not say which
            server asked, so set a callback per ``MCPClient`` instead when
            the handler needs the server name.  Per-caller sessions
            inherit their server's callback, so a call made with
            ``bearer_token=...`` can still answer elicitation requests.
        max_caller_sessions: Most idle per-caller sessions kept open.
            Least recently used sessions are closed first.
        caller_session_idle_timeout: Seconds after which an unused
            per-caller session is closed.

    Example::

        multi = MCPMultiClient({
            "hr": MCPClient(url="http://localhost:8080/mcp", bearer_token="..."),
            "docs": MCPClient(url="http://localhost:9090/mcp", api_key="secret"),
        })
        async with multi:
            tools = await multi.list_tools()
            result = await multi.call_tool("search_employees", {"query": "python"})
    """

    def __init__(
        self,
        clients: dict[str, MCPClient],
        *,
        elicitation_callback: ElicitationFnT | None = None,
        max_caller_sessions: int = 256,
        caller_session_idle_timeout: float = 300.0,
    ) -> None:
        self._clients = clients
        if elicitation_callback is not None:
            if not _sdk_supports_elicitation():
                raise MCPClientError(
                    "elicitation_callback requires mcp>=1.10 (the installed MCP SDK "
                    "has no client elicitation support)"
                )
            for client in clients.values():
                if client._elicitation_callback is None:
                    client._elicitation_callback = elicitation_callback
        # tool_name → server_name mapping (populated on connect)
        self._tool_to_server: dict[str, str] = {}
        self._connected = False
        self._max_caller_sessions = max_caller_sessions
        self._caller_session_idle_timeout = caller_session_idle_timeout
        self._caller_sessions = CallerSessionPool(
            max_sessions=max_caller_sessions, idle_timeout=caller_session_idle_timeout
        )
        self._warned_no_headers: set[str] = set()

    async def __aenter__(self) -> MCPMultiClient:
        """Connect to all servers."""
        for name, client in self._clients.items():
            try:
                await client.__aenter__()
            except Exception as exc:
                # Clean up already-connected clients
                for prev_name, prev_client in self._clients.items():
                    if prev_name == name:
                        break
                    try:
                        await prev_client.__aexit__(None, None, None)
                    except Exception:
                        logger.debug(
                            "Error cleaning up previously connected client '%s'",
                            prev_name,
                            exc_info=True,
                        )
                if isinstance(exc, MCPConnectionRejectedError):
                    raise exc.for_server(name) from exc.__cause__
                raise MCPClientError(f"Failed to connect to server '{name}': {exc}") from exc
        self._connected = True
        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Disconnect from all servers."""
        errors: list[str] = []
        try:
            await self._caller_sessions.aclose()
        except BaseException as e:
            errors.append(f"per-caller sessions: {e}")
        # A fresh pool, so the multi-client can be entered again.
        self._caller_sessions = CallerSessionPool(
            max_sessions=self._max_caller_sessions,
            idle_timeout=self._caller_session_idle_timeout,
        )
        for name, client in self._clients.items():
            try:
                await client.__aexit__(*exc)
            except BaseException as e:
                # Catch BaseException (not just Exception) so that
                # asyncio.CancelledError during session teardown
                # doesn't propagate and kill the cleanup loop.
                errors.append(f"{name}: {e}")
        self._connected = False
        self._tool_to_server.clear()
        if errors:
            logger.warning(
                "Errors during MCPMultiClient shutdown: %s",
                "; ".join(errors),
            )

    async def list_tools(self) -> list[Tool]:
        """Discover tools from all connected servers.

        Returns:
            Combined list of tools from all servers.  The
            ``_tool_to_server`` mapping is updated so ``call_tool``
            routes correctly.
        """
        if not self._connected:
            raise MCPClientError("Not connected. Use 'async with multi:'")

        all_tools: list[Tool] = []
        self._tool_to_server.clear()

        for server_name, client in self._clients.items():
            tools = await client.list_tools()
            for tool in tools:
                if tool.name in self._tool_to_server:
                    prev = self._tool_to_server[tool.name]
                    logger.warning(
                        "Tool name collision: '%s' exists on servers '%s' "
                        "and '%s'. The version from '%s' will be used.",
                        tool.name,
                        prev,
                        server_name,
                        server_name,
                    )
                self._tool_to_server[tool.name] = server_name
                all_tools.append(tool)

        return all_tools

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        bearer_token: str | None = None,
        progress_callback: ProgressFnT | None = None,
    ) -> CallToolResult:
        """Call a tool, automatically routing to the correct server.

        Args:
            name: Tool name (as discovered via ``list_tools``).
            arguments: Tool arguments dict.
            bearer_token: Send this token as ``Authorization: Bearer ...``
                instead of the server's configured credentials, over a
                session dedicated to this token.  Ignored (with a one-time
                warning) for stdio servers.
            progress_callback: Receives the call's progress notifications;
                see :meth:`MCPClient.call_tool`.

        Returns:
            MCP ``CallToolResult``.

        Raises:
            MCPClientError: If the tool name is unknown or the call fails.
            MCPConnectionRejectedError: The server refused *bearer_token*.
        """
        server_name = self._tool_to_server.get(name)
        if server_name is None:
            raise MCPClientError(
                f"Unknown tool '{name}'. Call list_tools() first to discover tools."
            )
        client = self._clients[server_name]
        if bearer_token:
            if client.supports_bearer_token:
                return await self._call_as(
                    server_name, client, bearer_token, name, arguments, progress_callback
                )
            if server_name not in self._warned_no_headers:
                self._warned_no_headers.add(server_name)
                logger.warning(
                    "Server '%s' uses the %s transport, which has no request "
                    "headers: the caller's bearer token is not sent to it, and "
                    "its tools run with the agent's own privileges for every caller.",
                    server_name,
                    client.transport,
                )
        generation = client.session_generation
        try:
            if progress_callback is not None:
                return await client.call_tool(name, arguments, progress_callback=progress_callback)
            return await client.call_tool(name, arguments)
        finally:
            if client.session_generation != generation:
                await self._refresh_server_tools(server_name)

    async def _refresh_server_tools(self, server_name: str) -> None:
        """Re-list *server_name*'s tools after its session was re-opened.

        A restarted server may serve a different tool set; routing is
        updated for that server only.  Best-effort: a failure keeps the
        previous routing and is logged.
        """
        try:
            tools = await self._clients[server_name].list_tools()
        except Exception as exc:
            logger.warning("Could not re-list tools from server '%s': %s", server_name, exc)
            return
        for tool_name, owner in list(self._tool_to_server.items()):
            if owner == server_name:
                del self._tool_to_server[tool_name]
        for tool in tools:
            self._tool_to_server[tool.name] = server_name
        logger.info(
            "Server '%s' opened a new session; %d tool(s) re-discovered", server_name, len(tools)
        )

    async def _call_as(
        self,
        server_name: str,
        client: MCPClient,
        bearer_token: str,
        name: str,
        arguments: dict[str, Any] | None,
        progress_callback: ProgressFnT | None = None,
    ) -> CallToolResult:
        """Call *name* over the session that authenticates as *bearer_token*."""
        if not self._connected:
            raise MCPClientError("Not connected. Use 'async with multi:'")
        try:
            async with self._caller_sessions.lease(server_name, client, bearer_token) as session:
                if progress_callback is not None:
                    return await session.call_tool(
                        name, arguments, progress_callback=progress_callback
                    )
                return await session.call_tool(name, arguments)
        except MCPConnectionRejectedError as exc:
            raise exc.for_server(server_name) from exc.__cause__

    @property
    def servers(self) -> dict[str, MCPClient]:
        """Read-only view of server name → client mapping."""
        return dict(self._clients)

    @property
    def tool_to_server(self) -> dict[str, str]:
        """Read-only view of tool name → server name mapping."""
        return dict(self._tool_to_server)
