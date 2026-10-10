"""Multi-server MCP client that aggregates tools from multiple servers.

Connects to N servers, discovers tools from each, and presents a
unified tool list.  Routes ``call_tool`` to the correct server based
on tool discovery, and ``read_resource`` / ``get_prompt`` based on
resource and prompt discovery.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from mcp.types import (
    CallToolResult,
    GetPromptResult,
    Prompt,
    ReadResourceResult,
    Resource,
    ResourceTemplate,
    Tool,
)

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
        # Resource / prompt routing (populated by the list_* methods, or
        # lazily by the first read_resource / get_prompt that needs it)
        self._resource_to_server: dict[str, str] = {}
        self._templates: list[tuple[re.Pattern[str], str, str]] = []  # (pattern, template, server)
        self._prompt_to_server: dict[str, str] = {}
        self._resources_discovered = False
        self._prompts_discovered = False
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
        self._forget_resources()
        self._prompt_to_server.clear()
        self._prompts_discovered = False
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
    ) -> CallToolResult:
        """Call a tool, automatically routing to the correct server.

        Args:
            name: Tool name (as discovered via ``list_tools``).
            arguments: Tool arguments dict.

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

    # ------------------------------------------------------------------
    # Resources and prompts
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self._connected:
            raise MCPClientError("Not connected. Use 'async with multi:'")

    def _client_for(self, server: str) -> MCPClient:
        client = self._clients.get(server)
        if client is None:
            raise MCPClientError(
                f"Unknown server '{server}'. Known servers: {', '.join(self._clients)}"
            )
        return client

    def _forget_resources(self) -> None:
        self._resource_to_server.clear()
        self._templates.clear()
        self._resources_discovered = False

    async def list_resources(self) -> list[Resource]:
        """Discover static resources from every connected server.

        A server that does not serve resources is skipped.  Updates the
        URI → server routing used by :meth:`read_resource`.
        """
        self._require_connected()
        found: list[Resource] = []
        self._resource_to_server.clear()
        for server_name, client in self._clients.items():
            try:
                resources = await client.list_resources()
            except MCPClientError as exc:
                logger.debug("Server '%s' lists no resources: %s", server_name, exc)
                continue
            for resource in resources:
                self._resource_to_server[str(resource.uri)] = server_name
                found.append(resource)
        return found

    async def list_resource_templates(self) -> list[ResourceTemplate]:
        """Discover resource templates from every connected server.

        A server that does not serve resources is skipped.  Updates the
        template routing used by :meth:`read_resource`.
        """
        self._require_connected()
        found: list[ResourceTemplate] = []
        self._templates.clear()
        for server_name, client in self._clients.items():
            try:
                templates = await client.list_resource_templates()
            except MCPClientError as exc:
                logger.debug("Server '%s' lists no resource templates: %s", server_name, exc)
                continue
            for template in templates:
                pattern = _template_pattern(template.uriTemplate)
                if pattern is not None:
                    self._templates.append((pattern, template.uriTemplate, server_name))
                found.append(template)
        return found

    @property
    def resource_to_server(self) -> dict[str, str]:
        """Read-only view of static resource URI → server name mapping."""
        return dict(self._resource_to_server)

    @property
    def template_to_server(self) -> dict[str, str]:
        """Read-only view of resource URI template → server name mapping."""
        return {template: server for _, template, server in self._templates}

    def _route_resource(self, uri: str) -> str | None:
        if uri in self._resource_to_server:
            return self._resource_to_server[uri]
        for pattern, _template, server_name in self._templates:
            if pattern.match(uri):
                return server_name
        return None

    async def read_resource(self, uri: str, *, server: str | None = None) -> ReadResourceResult:
        """Read a resource, routing to the server that serves *uri*.

        Args:
            uri: Resource URI (static, or matching a server's template).
            server: Read from this server instead of routing by URI.

        Raises:
            MCPClientError: No server serves *uri*, or the read failed.
        """
        self._require_connected()
        if server is not None:
            return await self._client_for(server).read_resource(uri)
        if len(self._clients) == 1:
            return await next(iter(self._clients.values())).read_resource(uri)
        target = self._route_resource(uri)
        if target is None and not self._resources_discovered:
            await self.list_resources()
            await self.list_resource_templates()
            self._resources_discovered = True
            target = self._route_resource(uri)
        if target is None:
            raise MCPClientError(
                f"No connected server serves the resource '{uri}'. "
                "List resources and templates to see what is available, or pass server=."
            )
        return await self._clients[target].read_resource(uri)

    async def list_prompts(self) -> list[Prompt]:
        """Discover prompts from every connected server.

        A server that does not serve prompts is skipped.  If two servers
        serve a prompt with the same name, the last one wins (a warning is
        logged) — pass ``server=`` to :meth:`get_prompt` to pick one.
        """
        self._require_connected()
        found: list[Prompt] = []
        self._prompt_to_server.clear()
        for server_name, client in self._clients.items():
            try:
                prompts = await client.list_prompts()
            except MCPClientError as exc:
                logger.debug("Server '%s' lists no prompts: %s", server_name, exc)
                continue
            for prompt in prompts:
                if prompt.name in self._prompt_to_server:
                    logger.warning(
                        "Prompt name collision: '%s' exists on servers '%s' and '%s'. "
                        "The version from '%s' will be used.",
                        prompt.name,
                        self._prompt_to_server[prompt.name],
                        server_name,
                        server_name,
                    )
                self._prompt_to_server[prompt.name] = server_name
                found.append(prompt)
        self._prompts_discovered = True
        return found

    @property
    def prompt_to_server(self) -> dict[str, str]:
        """Read-only view of prompt name → server name mapping."""
        return dict(self._prompt_to_server)

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        server: str | None = None,
    ) -> GetPromptResult:
        """Get a prompt, routing to the server that serves *name*.

        Args:
            name: Prompt name.
            arguments: Prompt arguments (see :meth:`MCPClient.get_prompt`).
            server: Get the prompt from this server instead of routing by name.

        Raises:
            MCPClientError: No server serves *name*, or the request failed.
        """
        self._require_connected()
        if server is not None:
            return await self._client_for(server).get_prompt(name, arguments)
        if len(self._clients) == 1:
            return await next(iter(self._clients.values())).get_prompt(name, arguments)
        if name not in self._prompt_to_server and not self._prompts_discovered:
            await self.list_prompts()
        target = self._prompt_to_server.get(name)
        if target is None:
            raise MCPClientError(
                f"No connected server serves the prompt '{name}'. "
                "List prompts to see what is available, or pass server=."
            )
        return await self._clients[target].get_prompt(name, arguments)


_TEMPLATE_PLACEHOLDER = re.compile(r"\{([^}]*)\}")


def _template_pattern(template: str) -> re.Pattern[str] | None:
    """Compile an RFC 6570 URI template to a regex for routing.

    Handles ``{name}`` (one path segment) and ``{name*}`` / ``{+name}``
    (the rest of the URI, ``/`` included) — what servers use for resource
    URIs.  Returns ``None`` for other operators (``{?q}``, ``{/x}``,
    ``{#x}`` ...); such templates are still listed but cannot be routed by
    URI (pass ``server=``).
    """
    parts = ["^"]
    last = 0
    for m in _TEMPLATE_PLACEHOLDER.finditer(template):
        expr = m.group(1)
        if re.fullmatch(r"\w+", expr):
            segment = "[^/]+"
        elif re.fullmatch(r"\+\w+|\w+\*", expr):
            segment = ".+"
        else:
            return None
        parts.append(re.escape(template[last : m.start()]))
        parts.append(segment)
        last = m.end()
    parts.append(re.escape(template[last:]))
    parts.append("$")
    return re.compile("".join(parts))
