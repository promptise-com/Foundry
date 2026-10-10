---
title: MCP Client — Native Python client for the Model Context Protocol
description: Promptise Foundry's native MCP client connects to one or many MCP servers, auto-discovers tools, and invokes them with full schema typing. Single-server, multi-server, and LangChain BaseTool adapter variants. HTTP, SSE, and stdio transports. Bearer token and API key authentication. No third-party MCP dependencies.
keywords: MCP client Python, Model Context Protocol client, MCP client SDK, MCPMultiClient, MCPClient, MCP tool adapter, LangChain MCP
---

# MCP Client

Connect to MCP servers as a client -- authenticate, discover tools, and invoke them programmatically.

## Quick Start

```python
import asyncio
from promptise import MCPClient

async def main():
    async with MCPClient(url="http://localhost:8080/mcp") as client:
        tools = await client.list_tools()
        print(f"Discovered {len(tools)} tools")

        result = await client.call_tool("add", {"a": 1, "b": 2})
        print(result.content[0].text)  # "3"

asyncio.run(main())
```

## Concepts

The client library has three classes:

- **MCPClient** connects to a single MCP server. It handles transport selection (HTTP, SSE, stdio), auth header injection, and session lifecycle.
- **MCPMultiClient** connects to multiple servers simultaneously and routes `call_tool` to the correct server based on tool discovery.
- **MCPToolAdapter** converts MCP tools into LangChain `BaseTool` instances for use with LangGraph agents.

All three use async context managers to manage connection lifecycle.

## MCPClient

### Unauthenticated connection

```python
async with MCPClient(url="http://localhost:8080/mcp") as client:
    tools = await client.list_tools()
```

### Bearer token authentication

```python
async with MCPClient(
    url="http://localhost:8080/mcp",
    bearer_token="eyJhbGciOiJIUzI1NiIs...",
) as client:
    result = await client.call_tool("search", {"query": "python"})
```

The `bearer_token` is injected as an `Authorization: Bearer <token>` header on every request.

### API key authentication

```python
async with MCPClient(
    url="http://localhost:8080/mcp",
    api_key="sk-my-secret-key",
) as client:
    tools = await client.list_tools()
```

The `api_key` is injected as an `x-api-key` header on every request.

### Custom headers

```python
async with MCPClient(
    url="http://localhost:8080/mcp",
    headers={"x-org-id": "acme-corp", "x-trace-id": "abc123"},
) as client:
    tools = await client.list_tools()
```

### Transports

MCPClient supports three transports:

| Transport | Use case | Required params |
|---|---|---|
| `"http"` (default) | Streamable HTTP for remote servers | `url` |
| `"sse"` | Server-Sent Events for streaming | `url` |
| `"stdio"` | Local process communication | `command`, `args` |

**HTTP (default):**

```python
async with MCPClient(url="http://localhost:8080/mcp") as client:
    ...
```

**SSE:**

```python
async with MCPClient(url="http://localhost:8080/sse", transport="sse") as client:
    ...
```

**stdio:**

```python
async with MCPClient(
    transport="stdio",
    command="python",
    args=["my_server.py"],
    env={"MY_VAR": "value"},
) as client:
    ...
```

### Constructor parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url` | `str \| None` | `None` | Server endpoint URL (HTTP/SSE) |
| `transport` | `str` | `"http"` | Transport type: `"http"`, `"sse"`, `"stdio"` |
| `headers` | `dict[str, str]` | `{}` | Extra HTTP headers |
| `bearer_token` | `str \| None` | `None` | Bearer token (auto-injected as `Authorization` header) |
| `api_key` | `str \| None` | `None` | API key (auto-injected as `x-api-key` header) |
| `command` | `str \| None` | `None` | Executable for stdio transport |
| `args` | `list[str]` | `[]` | Arguments for the stdio command |
| `env` | `dict[str, str]` | `{}` | Environment variables for the stdio process |
| `cwd` | `str \| None` | `None` | Working directory for the stdio process |
| `timeout` | `float` | `30.0` | HTTP request timeout in seconds |
| `elicitation_callback` | SDK `ElicitationFnT \| None` | `None` | Answers the server's MCP elicitation requests; the elicitation capability is declared only when set. See [Answering elicitation](#answering-elicitation-server-side-approval-gates) |

### Fetching tokens

Use the static `fetch_token` helper to acquire a JWT from a server's built-in token endpoint:

```python
token = await MCPClient.fetch_token(
    "http://localhost:8080/auth/token",
    client_id="agent-admin",
    client_secret="admin-secret",
)

async with MCPClient(url="http://localhost:8080/mcp", bearer_token=token) as client:
    tools = await client.list_tools()
```

`fetch_token` sends a POST with `{"client_id": ..., "client_secret": ...}` and returns the `access_token` string from the JSON response.

!!! warning "Production token acquisition"
    `fetch_token` is a convenience for development when the server has a built-in token endpoint. In production, obtain tokens from your Identity Provider (Auth0, Keycloak, Okta) and pass them via `bearer_token`.

### Calling tools

```python
async with MCPClient(url="http://localhost:8080/mcp", bearer_token=token) as client:
    # List available tools
    tools = await client.list_tools()
    for tool in tools:
        print(f"{tool.name}: {tool.description}")

    # Call a tool with arguments
    result = await client.call_tool("search_employees", {
        "query": "python",
        "remote_only": True,
        "limit": 5,
    })

    # Extract text from result
    for item in result.content:
        if hasattr(item, "text"):
            print(item.text)
```

### Accessing the session

For advanced use cases, access the underlying MCP `ClientSession`:

```python
async with MCPClient(url="http://localhost:8080/mcp") as client:
    session = client.session  # mcp.client.session.ClientSession
    headers = client.headers  # Read-only copy of HTTP headers
```

### Answering elicitation (server-side approval gates)

A server can ask the human behind the client for input mid-call through MCP
elicitation. The common case is a server-side
[approval gate](../server/approval-gates.md): `ElicitationApprover` (the
default on [MCPcast](../server/mcpcast.md#human-approval)-generated servers
with `env-token`, `passthrough` or `none` auth) asks the client to confirm each
gated tool call.

By default `MCPClient` declares **no** elicitation support, so such a server
denies the call with `APPROVAL_DENIED` and nothing runs. That is fail-closed and
stays the behaviour until you pass `elicitation_callback`. The callback has the
MCP SDK's signature, `async (context, params) -> ElicitResult | ErrorData`. To
send the requests to a human through any
[approval handler](../../core/approval.md), use
`approval_elicitation_callback`:

```python
from promptise.approval import CallbackApprovalHandler, approval_elicitation_callback
from promptise.mcp.client import MCPClient

async def ask_human(request):
    print(request.context_summary)                     # the server's own message
    print(f"{request.tool_name}({request.arguments})") # the call this client sent
    return input("Approve? [y/N] ").strip().lower() == "y"

client = MCPClient(
    transport="stdio",
    command="python",
    args=["petstore-mcp/server.py"],
    env={"MCPCAST_UPSTREAM_TOKEN": "Bearer <your API token>"},
    elicitation_callback=approval_elicitation_callback(
        CallbackApprovalHandler(ask_human),
        server_name="petstore",
        in_flight=lambda: client.in_flight_calls,
    ),
)
async with client:
    result = await client.call_tool("add_pet", {"name": "Rex"})
```

`build_agent(approval=...)` does this for every server automatically (see
[Server-side approval gates](../../core/approval.md#server-side-approval-gates)).
`examples/mcp/approve_server_gates.py` runs both cases, with and without a
handler, against a gated server and needs no LLM.

A request carries no tool name on the wire, so the client relates it to a call
itself. `client.in_flight_calls` lists the `call_tool` requests still awaiting a
result (`InFlightToolCall`: `name`, `arguments` and the caller's `contextvars`
snapshot). With exactly one call in flight, the approval request gets that
call's tool name and arguments. Otherwise the reviewer decides from the
server's message alone.

What the client guarantees:

- **No acceptance without a handler decision.** A callback that raises is answered with a JSON-RPC error, which a fail-closed server reads as a denial, and the session stays up.
- **Only confirmations are answered.** `approval_elicitation_callback` accepts only an empty form or a form whose one decision field is a boolean from `CONFIRMATION_FIELDS` (`approve`, `confirm`, `proceed`, ...). URL-mode requests, and forms that ask for anything else, are declined without asking the handler.
- **Elicitation is sequential per connection.** The MCP SDK handles server requests on the session's receive loop, so while a reviewer decides, other responses from that server wait. Keep handler timeouts below the server's approval timeout.

`MCPMultiClient(clients, elicitation_callback=...)` installs a default callback
on every client that has none of its own. The SDK callback does not say which
server asked, so give each `MCPClient` its own callback when the handler needs
the server name.

## MCPMultiClient

Connect to multiple servers and aggregate their tools. Routes `call_tool` to the correct server automatically.

```python
from promptise import MCPClient, MCPMultiClient

multi = MCPMultiClient({
    "hr": MCPClient(url="http://hr-server:8080/mcp", bearer_token=hr_token),
    "docs": MCPClient(url="http://docs-server:9090/mcp", api_key="sk-docs"),
})

async with multi:
    # Discover tools from all servers
    tools = await multi.list_tools()
    print(f"Total tools: {len(tools)}")

    # Call a tool -- routed to the correct server
    result = await multi.call_tool("search_employees", {"query": "python"})

    # Inspect routing
    print(multi.tool_to_server)  # {"search_employees": "hr", "search_docs": "docs"}
    print(multi.servers)         # {"hr": <MCPClient>, "docs": <MCPClient>}
```

### Calling a tool as a specific user

`call_tool(..., bearer_token=...)` sends that token as
`Authorization: Bearer <token>` instead of the server's configured
credentials. It is what `build_agent()` uses to make each invocation's tool
calls as the invoking user (`CallerContext.bearer_token`):

```python
async with multi:
    await multi.list_tools()
    alice_view = await multi.call_tool("my_tickets", {}, bearer_token=alice_jwt)
    bob_view = await multi.call_tool("my_tickets", {}, bearer_token=bob_jwt)
```

Each distinct token gets its own MCP session to the server: opened on first
use, reused by later calls with the same token, and closed after
`caller_session_idle_timeout` seconds unused (default 300). At most
`max_caller_sessions` idle sessions stay open (default 256; least recently
used first). Concurrent calls with different tokens never share a session
or a header, and a session in use is never closed. The token replaces any
`Authorization` header the client was configured with; other headers,
including `x-api-key`, are kept. A server that rejects the token raises
`MCPConnectionRejectedError` for that call only.

stdio servers have no request headers: the token is ignored for them and a
warning is logged once per server. `MCPClient.with_bearer_token(token)`
returns an unconnected copy of a client that authenticates with *token*, if
you manage sessions yourself.

### Tool name collisions

If two servers expose a tool with the same name, the last-discovered server wins and a warning is logged. Use server-specific prefixes on your MCP servers to avoid collisions.

### Error handling

```python
from promptise import MCPClientError

try:
    result = await multi.call_tool("unknown_tool", {})
except MCPClientError as e:
    print(f"Error: {e}")
    # "Unknown tool 'unknown_tool'. Call list_tools() first to discover tools."
```

### Rejected connections

When an HTTP server answers the `initialize` handshake with a 4xx status, connecting raises `MCPConnectionRejectedError` (a subclass of `MCPClientError`) immediately. The most common case is a server built with `require_auth=True` receiving no credentials, or the wrong ones:

```python
from promptise import MCPClient, MCPConnectionRejectedError

try:
    async with MCPClient(url="http://localhost:8080/mcp") as client:
        tools = await client.list_tools()
except MCPConnectionRejectedError as e:
    print(e.status_code)  # 401
    print(e)
    # "Server at http://localhost:8080/mcp rejected the connection:
    #  401 Unauthorized. Check the bearer_token/api_key configured for it."
```

Through `MCPMultiClient` and `build_agent()` the message names the server instead (`Server 'orders' rejected the connection: 401 Unauthorized. ...`) and `e.server_name` is set. A `404` points you at the URL instead of the credentials. Retrying with the same configuration fails the same way, so fix the credentials or URL instead of retrying.

Other connection failures (server down, connection refused) raise a plain `MCPClientError` such as `Failed to connect to http://localhost:8080/mcp (ConnectError: ...)`. If an established connection drops later, the next call raises `MCPClientError` (`Connection to ... was lost`). The failure never cancels the task that opened the client, because the client owns its transport in a task of its own. That also makes it safe to close a client from a different task than the one that opened it.

## MCPToolAdapter

Convert MCP tools into LangChain `BaseTool` instances for use with LangGraph or any LangChain-compatible agent.

```python
from promptise import MCPClient, MCPMultiClient, MCPToolAdapter

multi = MCPMultiClient({
    "hr": MCPClient(url="http://localhost:8080/mcp", bearer_token=token),
})

async with multi:
    adapter = MCPToolAdapter(multi)
    lc_tools = await adapter.as_langchain_tools()

    # Each tool is a LangChain BaseTool with a Pydantic args_schema
    for tool in lc_tools:
        print(f"{tool.name}: {tool.description}")
        print(f"  Schema: {tool.args_schema.model_json_schema()}")

    # Invoke directly
    result = await lc_tools[0].ainvoke({"query": "revenue"})
```

### Recursive schema handling

`MCPToolAdapter` builds fully-typed Pydantic models from MCP JSON Schemas, including:

- Nested objects (e.g., `Address` inside `CreateEmployeeRequest`)
- Arrays of objects
- `$ref` / `$defs` references
- Union types
- Field constraints (`minLength`, `pattern`, `ge`, `le`, etc.)
- Default values and descriptions

### Tracing callbacks

Attach callbacks for observability. Integrate with Datadog, Prometheus, OpenTelemetry, or any logging backend:

```python
def on_before(tool_name: str, kwargs: dict) -> None:
    print(f"Calling {tool_name} with {kwargs}")

def on_after(tool_name: str, result) -> None:
    print(f"{tool_name} returned successfully")

def on_error(tool_name: str, exc: Exception) -> None:
    print(f"{tool_name} failed: {exc}")

adapter = MCPToolAdapter(
    multi,
    on_before=on_before,
    on_after=on_after,
    on_error=on_error,
)
lc_tools = await adapter.as_langchain_tools()
```

### Tool introspection

Get metadata about discovered tools without converting to LangChain:

```python
tool_infos = await adapter.list_tool_info()
for info in tool_infos:
    print(f"{info.name} (server: {info.server_guess})")
    print(f"  {info.description}")
    print(f"  Schema: {info.input_schema}")
```

## Complete Example

A client that acquires a token, connects to a server, discovers tools, and calls them:

```python
import asyncio
import json
from promptise import MCPClient, MCPMultiClient, MCPToolAdapter

SERVER_URL = "http://127.0.0.1:8080"

async def main():
    # 1. Acquire a token
    token = await MCPClient.fetch_token(
        f"{SERVER_URL}/auth/token",
        client_id="agent-admin",
        client_secret="admin-secret",
    )

    # 2. Single-server connection
    async with MCPClient(url=f"{SERVER_URL}/mcp", bearer_token=token) as client:
        tools = await client.list_tools()
        print(f"Discovered {len(tools)} tools")

        result = await client.call_tool("list_employees")
        for item in result.content:
            if hasattr(item, "text"):
                employees = json.loads(item.text)
                print(f"Found {len(employees)} employees")

    # 3. Multi-server with LangChain adapter
    multi = MCPMultiClient({
        "hr": MCPClient(url=f"{SERVER_URL}/mcp", bearer_token=token),
    })

    async with multi:
        adapter = MCPToolAdapter(multi)
        lc_tools = await adapter.as_langchain_tools()
        print(f"Converted {len(lc_tools)} tools for LangChain")

        # These tools are ready for build_agent(extra_tools=lc_tools)

asyncio.run(main())
```

## API Summary

| Symbol | Type | Description |
|---|---|---|
| `MCPClient(url, transport, bearer_token, ...)` | Class | Single-server MCP client |
| `MCPClient.fetch_token(url, client_id, secret)` | Static method | Acquire a JWT from a token endpoint |
| `client.list_tools()` | Method | Discover all tools on the server |
| `client.call_tool(name, arguments)` | Method | Call a tool and get a `CallToolResult` |
| `client.in_flight_calls` | Property | `InFlightToolCall`s awaiting a result (for relating elicitation to a call) |
| `client.session` | Property | Underlying MCP `ClientSession` |
| `client.headers` | Property | Read-only copy of HTTP headers |
| `MCPMultiClient(clients, elicitation_callback=None)` | Class | Multi-server aggregating client |
| `approval_elicitation_callback(handler, ...)` | Function | Elicitation callback that asks an approval handler (`promptise.approval`) |
| `multi.list_tools()` | Method | Discover tools from all servers |
| `multi.call_tool(name, arguments)` | Method | Call a tool, auto-routed to the correct server |
| `multi.tool_to_server` | Property | Tool name to server name mapping |
| `multi.servers` | Property | Server name to `MCPClient` mapping |
| `MCPToolAdapter(multi, on_before, on_after, on_error)` | Class | MCP-to-LangChain tool converter |
| `adapter.as_langchain_tools()` | Method | Convert MCP tools to `BaseTool` instances |
| `adapter.list_tool_info()` | Method | Get tool metadata for introspection |
| `MCPClientError` | Exception | Raised on client operation failures |
| `MCPConnectionRejectedError` | Exception | The server refused the handshake with a 4xx (`status_code`, `reason`, `url`, `server_name`) |

!!! tip "Persistent connections"
    `MCPClient` and `MCPMultiClient` maintain persistent connections for the lifetime of the context manager. Avoid creating a new client per tool call -- instead, keep the client alive for the duration of your agent session.

!!! warning "Call list_tools() before call_tool()"
    With `MCPMultiClient`, you must call `list_tools()` at least once before `call_tool()` so the client can discover which server owns each tool. Without discovery, `call_tool()` raises `MCPClientError`.

!!! tip "LangChain integration"
    The tools returned by `MCPToolAdapter.as_langchain_tools()` are standard LangChain `BaseTool` instances. Pass them directly to `build_agent(extra_tools=lc_tools)` or any LangChain-compatible workflow.

## What's Next?

- [Tool Adapter](tool-adapter.md) -- Convert MCP tools to LangChain `BaseTool` instances
- [Step-by-Step Guide](../../guides/production-mcp-servers.md) -- Build the server your client connects to
- [Server Fundamentals](../server/building-servers.md) -- Full reference for tools, resources, and config
- [Auth & Security](../server/auth-security.md) -- Set up JWT auth on the server side
