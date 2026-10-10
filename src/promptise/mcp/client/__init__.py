"""Promptise MCP Client — production-grade MCP client with token auth.

A production MCP client with:

- **Bearer token auth**: Pass pre-issued tokens from your IdP or the
  server's built-in token endpoint
- **API key auth**: Simple pre-shared secret for lightweight security
- **Header injection**: Custom headers sent on every request
- **Multi-server**: Connect to multiple MCP servers simultaneously
- **Transport auto-detection**: HTTP, SSE, and stdio support
- **LangChain integration**: Converts MCP tools to LangChain ``BaseTool`` with
  recursive schema handling (nested Pydantic models, ``$ref``/``$defs``, etc.)

The client **never** generates tokens.  Tokens are obtained externally
(from an Identity Provider or the server's token endpoint) and passed in.

Example::

    from promptise.mcp.client import MCPClient

    async with MCPClient(url="http://localhost:8080/mcp") as client:
        tools = await client.list_tools()
        result = await client.call_tool("search", {"query": "python"})

With Bearer token auth::

    # Get token from server's built-in endpoint (dev/testing)
    token = await MCPClient.fetch_token(
        "http://localhost:8080/auth/token",
        client_id="my-agent",
        client_secret="agent-secret",
    )

    # Or from your Identity Provider (production)
    # token = await my_idp.get_access_token()

    async with MCPClient(
        url="http://localhost:8080/mcp",
        bearer_token=token,
    ) as client:
        tools = await client.list_tools()

With API key auth::

    async with MCPClient(
        url="http://localhost:8080/mcp",
        api_key="my-secret-key",
    ) as client:
        tools = await client.list_tools()
"""

from ._client import MCPClient, MCPClientError, MCPConnectionRejectedError
from ._multi import MCPMultiClient
from ._tool_adapter import MCPToolAdapter, MCPToolError

__all__ = [
    "MCPClient",
    "MCPClientError",
    "MCPConnectionRejectedError",
    "MCPMultiClient",
    "MCPToolAdapter",
    "MCPToolError",
]
