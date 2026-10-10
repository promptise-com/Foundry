# Server Configuration

Configure how agents connect to MCP servers using `StdioServerSpec` for local servers and `HTTPServerSpec` for remote ones.

## Quick Example

=== "Stdio"

    ```python
    from promptise.config import StdioServerSpec

    servers = {
        "math": StdioServerSpec(
            command="python",
            args=["-m", "math_server"],
        ),
    }
    ```

=== "HTTP"

    ```python
    from promptise.config import HTTPServerSpec

    servers = {
        "api": HTTPServerSpec(
            url="http://localhost:8000/mcp",
            bearer_token="eyJhbGciOiJIUzI1NiIs...",
        ),
    }
    ```

## Concepts

Every Promptise agent receives a `servers` dict that maps a human-readable name to a server specification. The agent connects to each server at startup, discovers its tools, and makes them available to the LLM.

There are two spec types:

- **`StdioServerSpec`** -- launches a local process and communicates over stdin/stdout. Best for local development and bundled servers.
- **`HTTPServerSpec`** -- connects to a remote server over HTTP, Streamable HTTP, or SSE. Best for production deployments and shared services.

Both are Pydantic models with strict validation (`extra="forbid"`), so typos in field names are caught immediately.

## StdioServerSpec

Use `StdioServerSpec` when the MCP server is a local process that the agent should launch and manage.

```python
from promptise.config import StdioServerSpec

spec = StdioServerSpec(
    command="python",
    args=["-m", "mypackage.server", "--port", "0"],
    env={"API_KEY": "sk-..."},
    cwd="/path/to/project",
    keep_alive=True,
)
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `command` | `str` | **required** | Executable to launch (e.g. `"python"`, `"node"`, `"npx"`). |
| `args` | `list[str]` | `[]` | Positional arguments for the process. |
| `env` | `dict[str, str]` | `{}` | Environment variables set for the child process. |
| `cwd` | `str \| None` | `None` | Working directory for the process. |
| `keep_alive` | `bool` | `True` | Whether the client should maintain a persistent session. |

### Examples

=== "Python server"

    ```python
    StdioServerSpec(
        command="python",
        args=["-m", "my_mcp_server"],
        env={"DATABASE_URL": "sqlite:///data.db"},
    )
    ```

=== "Node.js server"

    ```python
    StdioServerSpec(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/data"],
    )
    ```

=== "Custom binary"

    ```python
    StdioServerSpec(
        command="/usr/local/bin/my-server",
        args=["--verbose"],
        cwd="/opt/servers",
    )
    ```

## HTTPServerSpec

Use `HTTPServerSpec` for remote MCP servers accessible over the network. Supports three transport protocols and multiple authentication methods.

```python
from promptise.config import HTTPServerSpec

spec = HTTPServerSpec(
    url="https://mcp.example.com/mcp",
    bearer_token="eyJhbGciOiJIUzI1NiIs...",
)
```

### Fields

| Field | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | **required** | Full endpoint URL (e.g. `"http://127.0.0.1:8080/mcp"`). |
| `transport` | `"http" \| "streamable-http" \| "sse"` | `"http"` | Transport protocol. |
| `headers` | `dict[str, str]` | `{}` | Extra HTTP headers sent on every request. |
| `bearer_token` | `str \| None` | `None` | Pre-issued Bearer token. Automatically injected as `Authorization: Bearer <token>`. |
| `api_key` | `str \| None` | `None` | Pre-shared API key. Automatically injected as `x-api-key: <key>`. |
| `audience` | `str \| None` | `None` | Resource audience this server expects; with an agent identity and no `bearer_token`, a credential for this audience is presented. |
| `forward_caller_token` | `bool` | `True` | When an invocation's `CallerContext` has a `bearer_token`, call this server's tools with that token (one MCP session per caller) instead of `bearer_token` / the agent identity. Set `False` for servers that must not receive your users' tokens. See [Per-user tokens](#per-user-tokens). |
| `auth` | `str \| None` | `None` | Legacy auth hint (kept for backward compatibility). |

### Per-user tokens

The credentials above authenticate **the agent**: they open its connection
to the server and are used to discover the server's tools. When an
invocation carries a user's token, the tool calls are made as that user:

```python
agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={
        "crm": HTTPServerSpec(url="https://crm.internal/mcp", bearer_token=SERVICE_TOKEN),
        "search": HTTPServerSpec(
            url="https://search.vendor.example/mcp",
            api_key=VENDOR_KEY,
            forward_caller_token=False,  # third party: never send our users' tokens
        ),
    },
)

caller = CallerContext(user_id="alice", tenant_id="acme", bearer_token=alice_jwt)
await agent.ainvoke({"messages": [...]}, caller=caller)
# crm tools:    Authorization: Bearer <alice_jwt>   (a session opened for Alice)
# search tools: x-api-key: <VENDOR_KEY>             (the agent's own session)
```

Each distinct caller token gets its own MCP session, opened on first use,
reused by that caller's later calls, and closed after five idle minutes;
concurrent invocations never share a session or a header. The forwarded
token replaces any `Authorization` header from `bearer_token` or
`headers`; other headers, including `x-api-key`, are kept. Invocations
without a caller token use the spec's credentials. `StdioServerSpec`
servers have no request headers and never receive a caller token.

### Transport Protocols

| Transport | When to Use |
|---|---|
| `"http"` | Default. MCP Streamable HTTP — the current MCP transport, served by Promptise servers at `/mcp`. Responses to a request can stream (progress, incremental results) and the server can send notifications. |
| `"streamable-http"` | An alias of `"http"`: the same Streamable HTTP client, accepted because other MCP tools spell the transport this way. |
| `"sse"` | The legacy HTTP+SSE transport (an event stream at `/sse`, posts to `/messages/`). Use it only for servers that do not speak Streamable HTTP. |

Point the URL at the MCP endpoint itself — `http://host:8080/mcp` for a Promptise server, not `http://host:8080`. A URL without the endpoint fails at connect time with `MCPConnectionRejectedError` (`404 Not Found`, with a hint to check the `/mcp` path).

### Authentication

`HTTPServerSpec` supports three authentication approaches. Use whichever matches your server's requirements.

=== "Bearer Token (JWT)"

    ```python
    HTTPServerSpec(
        url="http://localhost:8080/mcp",
        bearer_token="eyJhbGciOiJIUzI1NiIs...",
    )
    # Sends: Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
    ```

=== "API Key"

    ```python
    HTTPServerSpec(
        url="http://localhost:8080/mcp",
        api_key="my-secret-key",
    )
    # Sends: x-api-key: my-secret-key
    ```

=== "Custom Headers"

    ```python
    HTTPServerSpec(
        url="http://localhost:8080/mcp",
        headers={
            "Authorization": "Bearer <token>",
            "X-Custom-Header": "value",
        },
    )
    ```

!!! tip "Token source"
    Tokens should be obtained from your Identity Provider (Auth0, Keycloak, Okta, etc.) or from the MCP server's built-in token endpoint. The agent never generates tokens itself.

## Combining Multiple Servers

An agent can connect to any number of servers simultaneously. Each server's tools are discovered independently and merged into a single toolset.

```python
from promptise import build_agent
from promptise.config import StdioServerSpec, HTTPServerSpec

agent = await build_agent(
    servers={
        "filesystem": StdioServerSpec(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/data"],
        ),
        "database": HTTPServerSpec(
            url="http://localhost:9000/mcp",
            api_key="db-secret",
        ),
        "search": HTTPServerSpec(
            url="https://search.example.com/mcp",
            bearer_token="eyJ...",
        ),
    },
    model="openai:gpt-5-mini",
)
```

Tools from all three servers are available to the agent as a flat list. If two servers expose a tool with the same name, both are included -- the LLM decides which to call based on their descriptions.

## API Summary

| Symbol | Import | Description |
|---|---|---|
| `StdioServerSpec` | `from promptise.config import StdioServerSpec` | Local process server launched via stdin/stdout. |
| `HTTPServerSpec` | `from promptise.config import HTTPServerSpec` | Remote server over HTTP, Streamable HTTP, or SSE. |
| `ServerSpec` | `from promptise.config import ServerSpec` | Union type: `StdioServerSpec \| HTTPServerSpec`. |

!!! tip "Validation catches typos"
    Both specs use `extra="forbid"` in their Pydantic config. Passing an unknown field like `commnad` instead of `command` raises a validation error immediately rather than silently ignoring it.

!!! warning "Empty servers dict"
    Passing `servers={}` is valid but the agent will have no MCP tools. This is useful when you rely solely on `extra_tools` or `cross_agents`, but if unintentional, double-check your configuration.

## What's Next?

- [Building Agents](building-agents.md) -- full `build_agent()` parameter reference.
- [SuperAgent Files](superagent-files.md) -- define server configs declaratively in YAML.
- [Authentication & Security](../../mcp/server/auth-security.md) -- securing your MCP servers with JWT and API keys.
