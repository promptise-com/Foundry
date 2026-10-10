"""Multi-server MCP client that aggregates tools from multiple servers.

Connects to N servers, discovers tools from each, and presents a
unified tool list.  Routes ``call_tool`` to the correct server based
on tool discovery, and ``read_resource`` / ``get_prompt`` based on
resource and prompt discovery.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, TypeVar

from mcp.types import (
    CallToolResult,
    GetPromptResult,
    Prompt,
    ReadResourceResult,
    Resource,
    ResourceTemplate,
    Tool,
)

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

_T = TypeVar("_T")


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
        # Resource / prompt routing (populated by the list_* methods, or
        # lazily by the first read_resource / get_prompt that needs it)
        self._resource_to_server: dict[str, str] = {}
        self._templates: list[tuple[re.Pattern[str], str, str]] = []  # (pattern, template, server)
        self._prompt_to_server: dict[str, str] = {}
        self._resources_discovered = False
        self._prompts_discovered = False
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
            self._warn_no_headers(server_name, client)
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

    async def server_for_resource(self, uri: str) -> str:
        """Name of the server that serves *uri* (discovering resources if needed).

        Raises:
            MCPClientError: No connected server serves *uri*.
        """
        self._require_connected()
        if len(self._clients) == 1:
            return next(iter(self._clients))
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
        return target

    async def read_resource(
        self,
        uri: str,
        *,
        server: str | None = None,
        bearer_token: str | None = None,
    ) -> ReadResourceResult:
        """Read a resource, routing to the server that serves *uri*.

        Args:
            uri: Resource URI (static, or matching a server's template).
            server: Read from this server instead of routing by URI.
            bearer_token: Read as this caller: the token is sent as
                ``Authorization: Bearer ...`` over the caller's own session,
                exactly as for :meth:`call_tool`.  Ignored (with a one-time
                warning) for stdio servers.

        Raises:
            MCPClientError: No server serves *uri*, or the read failed.
            MCPConnectionRejectedError: The server refused *bearer_token*.
        """
        self._require_connected()
        target = server if server is not None else await self.server_for_resource(uri)
        self._client_for(target)
        return await self._on_server(target, bearer_token, lambda c: c.read_resource(uri))

    async def _on_server(
        self,
        server_name: str,
        bearer_token: str | None,
        operation: Callable[[MCPClient], Awaitable[_T]],
    ) -> _T:
        """Run *operation* on *server_name*'s client, or as *bearer_token*'s caller."""
        client = self._client_for(server_name)
        if bearer_token:
            if client.supports_bearer_token:
                try:
                    async with self._caller_sessions.lease(
                        server_name, client, bearer_token
                    ) as session:
                        return await operation(session)
                except MCPConnectionRejectedError as exc:
                    raise exc.for_server(server_name) from exc.__cause__
            self._warn_no_headers(server_name, client)
        return await operation(client)

    def _warn_no_headers(self, server_name: str, client: MCPClient) -> None:
        if server_name not in self._warned_no_headers:
            self._warned_no_headers.add(server_name)
            logger.warning(
                "Server '%s' uses the %s transport, which has no request "
                "headers: the caller's bearer token is not sent to it, and "
                "its tools run with the agent's own privileges for every caller.",
                server_name,
                client.transport,
            )

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

    async def server_for_prompt(self, name: str) -> str:
        """Name of the server that serves prompt *name* (discovering prompts if needed).

        Raises:
            MCPClientError: No connected server serves *name*.
        """
        self._require_connected()
        if len(self._clients) == 1:
            return next(iter(self._clients))
        if name not in self._prompt_to_server and not self._prompts_discovered:
            await self.list_prompts()
        target = self._prompt_to_server.get(name)
        if target is None:
            raise MCPClientError(
                f"No connected server serves the prompt '{name}'. "
                "List prompts to see what is available, or pass server=."
            )
        return target

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        server: str | None = None,
        bearer_token: str | None = None,
    ) -> GetPromptResult:
        """Get a prompt, routing to the server that serves *name*.

        Args:
            name: Prompt name.
            arguments: Prompt arguments (see :meth:`MCPClient.get_prompt`).
            server: Get the prompt from this server instead of routing by name.
            bearer_token: Get it as this caller, over the caller's own
                session (see :meth:`read_resource`).

        Raises:
            MCPClientError: No server serves *name*, or the request failed.
            MCPConnectionRejectedError: The server refused *bearer_token*.
        """
        self._require_connected()
        target = server if server is not None else await self.server_for_prompt(name)
        return await self._on_server(target, bearer_token, lambda c: c.get_prompt(name, arguments))


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
