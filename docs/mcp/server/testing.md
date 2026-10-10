# MCP Server Testing

The `TestClient` class lets you exercise the **full** MCP server pipeline --
input validation, dependency injection, guard checks, middleware chain, handler
invocation, and error serialisation -- without starting a network transport.
Tests run entirely in-process and are as fast as plain function calls.

**Source:** `src/promptise/mcp/server/testing.py` and `src/promptise/mcp/server/_testing.py`

## Quick example

```python
import pytest
from promptise.mcp.server import MCPServer
from promptise.mcp.server.testing import TestClient

server = MCPServer(name="test")

@server.tool()
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@pytest.mark.asyncio
async def test_add():
    client = TestClient(server)
    result = await client.call_tool("add", {"a": 1, "b": 2})
    assert result[0].text == "3"
```

## Concepts

### What TestClient exercises

The `TestClient` replicates the exact same call pipeline as the real MCP
transport.  Every tool call goes through:

1. **Input validation** -- arguments are validated against the tool's Pydantic
   input model.
2. **Dependency injection** -- `Depends(...)` parameters are resolved.
3. **Context injection** -- parameters typed as `RequestContext` receive the
   current request context.
4. **Guard checks** -- registered guards (`RequireAuth`, `HasRole`, etc.) are
   evaluated.
5. **Middleware chain** -- server-level and router-level middleware run in order.
6. **Handler invocation** -- the actual tool function is called.
7. **Result serialisation** -- the return value is converted to MCP
   `TextContent` list.
8. **Background tasks** -- any `BackgroundTasks` scheduled during the call are
   executed before `call_tool` returns, so tests can assert on their effects.
   (A running server sends the response first and runs them afterwards.)
9. **Error handling** -- `MCPError` subclasses are serialised to structured JSON.

### Creating a TestClient

```python
from promptise.mcp.server import MCPServer
from promptise.mcp.server.testing import TestClient

server = MCPServer(name="my-server")

# Basic client (no auth)
client = TestClient(server)

# Client with simulated auth metadata
client = TestClient(server, meta={"authorization": "Bearer my-test-token"})
```

The `meta` dict is copied into every `RequestContext.meta` the client creates,
simulating HTTP request headers without an actual transport.

### Calling tools

`call_tool` returns a `list[TextContent]`, exactly like the real MCP server:

```python
from promptise.mcp.server import MCPServer
from promptise.mcp.server.testing import TestClient

server = MCPServer(name="test")

@server.tool()
async def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

async def test_greet():
    client = TestClient(server)
    result = await client.call_tool("greet", {"name": "World"})
    assert len(result) == 1
    assert result[0].text == "Hello, World!"
```

### Handling errors

When a tool is not found, or an exception occurs, the client returns structured
error JSON rather than raising -- matching the real server's behaviour:

```python
import json

async def test_unknown_tool():
    client = TestClient(server)
    result = await client.call_tool("nonexistent", {})
    error = json.loads(result[0].text)
    assert error["error"]["code"] == "TOOL_NOT_FOUND"

async def test_internal_error():
    @server.tool()
    async def fail() -> str:
        raise ValueError("something broke")

    client = TestClient(server)
    result = await client.call_tool("fail", {})
    error = json.loads(result[0].text)
    assert error["error"]["code"] == "INTERNAL_ERROR"
```

`MCPError` subclasses (like `ToolError`, `AuthenticationError`) are serialised
using their own `to_text()` method, preserving the error code and retryable
flag.

### Testing with authentication

Pass a `meta` dict to simulate authenticated requests:

```python
from promptise.mcp.server import MCPServer, AuthMiddleware, JWTAuth, RequireAuth
from promptise.mcp.server.testing import TestClient

server = MCPServer(name="secure")

jwt_auth = JWTAuth(secret="test-secret", algorithm="HS256")
server.add_middleware(AuthMiddleware(jwt_auth))

@server.tool(guards=[RequireAuth()])
async def secret_data() -> str:
    """Return sensitive data."""
    return "top-secret-info"

async def test_unauthenticated():
    client = TestClient(server)
    result = await client.call_tool("secret_data", {})
    error_text = result[0].text
    assert "ACCESS_DENIED" in error_text

async def test_authenticated():
    # Generate a test token
    import jwt as pyjwt
    token = pyjwt.encode({"sub": "test-user"}, "test-secret", algorithm="HS256")

    client = TestClient(server, meta={"authorization": f"Bearer {token}"})
    result = await client.call_tool("secret_data", {})
    assert result[0].text == "top-secret-info"
```

### Testing with middleware

Middleware runs in the same order as on the real server:

```python
from promptise.mcp.server import MCPServer, LoggingMiddleware, TimeoutMiddleware
from promptise.mcp.server.testing import TestClient

server = MCPServer(name="test")
server.add_middleware(LoggingMiddleware())
server.add_middleware(TimeoutMiddleware(default_timeout=5.0))

@server.tool()
async def slow_task() -> str:
    import asyncio
    await asyncio.sleep(0.1)
    return "done"

async def test_with_middleware():
    client = TestClient(server)
    result = await client.call_tool("slow_task", {})
    assert result[0].text == "done"
```

### Listing tools

Retrieve all registered tools as MCP `Tool` objects:

```python
async def test_list_tools():
    client = TestClient(server)
    tools = await client.list_tools()
    names = [t.name for t in tools]
    assert "add" in names
    assert "greet" in names
```

### Reading resources

Test static resources and URI templates. Reads run the full pipeline (middleware, authentication, guards, conversion of template parameters), as on the live server:

```python
from promptise.mcp.server import MCPServer
from promptise.mcp.server.testing import TestClient

server = MCPServer(name="test")

@server.resource("config://app")
async def app_config() -> dict:
    return {"version": "1.0"}

@server.resource_template("users://{user_id}/profile")
async def user_profile(user_id: int) -> dict:
    return {"user_id": user_id}

async def test_resources():
    client = TestClient(server)

    # Static resource: a dict arrives as JSON text
    text = await client.read_resource("config://app")
    assert json.loads(text) == {"version": "1.0"}

    # URI template: "42" is converted to the int the handler declares
    text = await client.read_resource("users://42/profile")
    assert json.loads(text) == {"user_id": 42}

    # The MIME type and raw MCP contents, as a client receives them
    (item,) = await client.read_resource_contents("config://app")
    assert item.mimeType == "application/json"

    # List resources (includes the auto-registered docs://manifest)
    resources = await client.list_resources()
    assert {str(r.uri) for r in resources} == {"config://app", "docs://manifest"}

    # List templates
    templates = await client.list_resource_templates()
    assert len(templates) == 1
```

`read_resource()` returns the text of the first content item, or `bytes` for a binary resource.

### Testing prompts

Test registered prompt templates:

```python
from promptise.mcp.server import MCPServer
from promptise.mcp.server.testing import TestClient

server = MCPServer(name="test")

@server.prompt()
async def summarize(text: str, style: str = "concise") -> str:
    return f"Summarize the following text in a {style} style:\n\n{text}"

async def test_prompt():
    client = TestClient(server)

    result = await client.get_prompt("summarize", {"text": "Hello world"})
    assert "Hello world" in result.messages[0].content.text

    # List all prompts
    prompts = await client.list_prompts()
    assert any(p.name == "summarize" for p in prompts)
```

### Testing access to resources and prompts

Pass credentials with `meta=` (or per call with `headers=`). A denied read or prompt request raises the error the live server sends back as an MCP error:

```python
from promptise.mcp.server import APIKeyAuth, AuthenticationError, AuthMiddleware

server = MCPServer(name="test")
server.add_middleware(AuthMiddleware(APIKeyAuth(keys={
    "sk-admin": {"client_id": "alice", "roles": ["admin"]},
    "sk-user": {"client_id": "bob", "roles": ["user"]},
})))

@server.resource("hr://payroll", roles=["admin"])
async def payroll() -> dict:
    return {"alice": 100}

async def test_payroll_needs_admin():
    with pytest.raises(AuthenticationError):
        await TestClient(server, meta={"x-api-key": "sk-user"}).read_resource("hr://payroll")
    admin = TestClient(server, meta={"x-api-key": "sk-admin"})
    assert json.loads(await admin.read_resource("hr://payroll")) == {"alice": 100}
```

## API summary

### TestClient

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `server` | `MCPServer` | *required* | The server instance to test |
| `meta` | `dict[str, Any] \| None` | `None` | Simulated request metadata (e.g. auth headers) |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `call_tool(name, arguments)` | `list[TextContent]` | Call a tool through the full pipeline |
| `list_tools()` | `list[Tool]` | List all registered tools |
| `read_resource(uri, *, headers=None)` | `str \| bytes` | Read a resource through the full pipeline |
| `read_resource_contents(uri, *, headers=None)` | `list[TextResourceContents \| BlobResourceContents]` | Read a resource and return the MCP contents (with `mimeType`) |
| `list_resources()` | `list[Resource]` | List all static resources, including `docs://manifest` |
| `list_resource_templates()` | `list[ResourceTemplate]` | List all resource URI templates |
| `get_prompt(name, arguments, *, headers=None)` | `GetPromptResult` | Get a prompt through the full pipeline |
| `list_prompts()` | `list[Prompt]` | List all registered prompts |

## Tips and gotchas

!!! tip
    `TestClient` is marked with `__test__ = False` so pytest does not try to
    collect it as a test class.  Import it normally from
    `promptise.mcp.server.testing`.

!!! warning
    Errors from `MCPError` subclasses are serialised to JSON text (not raised).
    If you expect a tool call to fail, parse `result[0].text` as JSON and check
    the `error.code` field.

!!! tip
    The `meta` dict in `TestClient` merges with any ambient HTTP request
    headers (from context variables).  Explicit `meta` keys take precedence,
    so test code always wins over ambient state.

!!! warning
    `call_tool` returns `list[TextContent]`.  A successful call with a `None`
    return value from the handler produces `[TextContent(text="OK")]`.  A
    `dict` or `list` return value is JSON-serialised.  A `str` is returned
    as-is.

!!! tip
    Combine `TestClient` with pytest fixtures for clean, reusable test setups:

    ```python
    import pytest
    from promptise.mcp.server.testing import TestClient

    @pytest.fixture
    def client():
        return TestClient(server, meta={"authorization": "Bearer test-token"})

    @pytest.mark.asyncio
    async def test_tool(client):
        result = await client.call_tool("add", {"a": 1, "b": 2})
        assert result[0].text == "3"
    ```

## What's next

- [Building Servers](building-servers.md) — create the server you are testing
- [Auth & Security](auth-security.md) — set up authentication for guard testing
- [Routers & Middleware](routers-middleware.md) — modular routing that TestClient exercises
- [Caching & Performance](caching-performance.md) — test caching and rate limiting behavior
- [Deployment](deployment.md) — deploy your tested server to production
- [MCP Client](../client/index.md) — connect to a real running server
