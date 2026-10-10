"""Multi-server MCP client that aggregates tools from multiple servers.

Connects to N servers, discovers tools from each, and presents a
unified tool list.  Routes ``call_tool`` to the correct server based
on tool discovery.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from mcp.types import CallToolResult, Tool

if TYPE_CHECKING:
    from mcp.shared.session import ProgressFnT

from ._client import MCPClient, MCPClientError, MCPConnectionRejectedError

logger = logging.getLogger("promptise.mcp.client")


class MCPMultiClient:
    """Connect to multiple MCP servers and aggregate their tools.

    Each server gets its own ``MCPClient`` with independent auth / headers.
    Tools are tracked per-server so ``call_tool`` routes to the correct one.

    **Tool name collisions**: If two servers expose a tool with the same
    name, the last-discovered server wins and a warning is logged.
    Consider using server-specific prefixes on your MCP servers to avoid
    collisions.

    Args:
        clients: Mapping of server name → ``MCPClient`` instance.

    Example::

        multi = MCPMultiClient({
            "hr": MCPClient(url="http://localhost:8080/mcp", bearer_token="..."),
            "docs": MCPClient(url="http://localhost:9090/mcp", api_key="secret"),
        })
        async with multi:
            tools = await multi.list_tools()
            result = await multi.call_tool("search_employees", {"query": "python"})
    """

    def __init__(self, clients: dict[str, MCPClient]) -> None:
        self._clients = clients
        # tool_name → server_name mapping (populated on connect)
        self._tool_to_server: dict[str, str] = {}
        self._connected = False

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
        progress_callback: ProgressFnT | None = None,
    ) -> CallToolResult:
        """Call a tool, automatically routing to the correct server.

        Args:
            name: Tool name (as discovered via ``list_tools``).
            arguments: Tool arguments dict.
            progress_callback: Receives the call's progress notifications;
                see :meth:`MCPClient.call_tool`.

        Returns:
            MCP ``CallToolResult``.

        Raises:
            MCPClientError: If the tool name is unknown or the call fails.
        """
        server_name = self._tool_to_server.get(name)
        if server_name is None:
            raise MCPClientError(
                f"Unknown tool '{name}'. Call list_tools() first to discover tools."
            )
        client = self._clients[server_name]
        try:
            if progress_callback is not None:
                return await client.call_tool(name, arguments, progress_callback=progress_callback)
            return await client.call_tool(name, arguments)
        except MCPClientError:
            # Invalidate stale tool mapping on connection failure —
            # the server may have restarted with different tools
            self._tool_to_server.pop(name, None)
            raise

    @property
    def servers(self) -> dict[str, MCPClient]:
        """Read-only view of server name → client mapping."""
        return dict(self._clients)

    @property
    def tool_to_server(self) -> dict[str, str]:
        """Read-only view of tool name → server name mapping."""
        return dict(self._tool_to_server)
