# Server Fundamentals

Full reference for building MCP servers with decorator-based tool registration, Pydantic validation, and lifecycle hooks.

!!! tip "New to MCP servers?"
    Start with the [Step-by-Step Guide](../../guides/production-mcp-servers.md) for a hands-on walkthrough. This page is the deep reference.

## Quick Start

```python
from promptise.mcp.server import MCPServer

server = MCPServer(name="my-tools", version="1.0.0")

@server.tool()
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@server.resource("config://app", mime_type="application/json")
async def app_config() -> str:
    """Return application configuration."""
    return '{"version": "1.0.0"}'

server.run()  # stdio transport by default
```

Run this file and any MCP-compatible client can discover and call the `add` tool. Parameters are validated automatically from the function signature and type hints.

## Concepts

**MCPServer** is the central object. It manages tool, resource, and prompt registries, a middleware chain, lifecycle hooks, and the transport layer. You register handlers with decorators and call `server.run()` to start serving.

**Tools** are the primary capability. Each tool is a Python function exposed to MCP clients. The framework automatically generates a JSON Schema from the function signature, including Pydantic model parameters.

**Resources** expose read-only data at static URIs (e.g. `config://app`). **Resource templates** use `{param}` placeholders for dynamic URIs.

**Prompts** expose reusable prompt templates that clients can discover and render with arguments.

## Registering Tools

### Basic tools

Decorate any sync or async function with `@server.tool()`. The function name becomes the tool name, and the docstring's summary paragraph becomes the description.

```python
@server.tool()
async def search(query: str, limit: int = 10) -> list[dict]:
    """Search records by keyword."""
    return await db.search(query, limit)
```

### Describing parameters

The model only sees what is in the tool's `inputSchema`, so describe every parameter whose meaning is not obvious from its name. There are two ways to do it.

**`Annotated` with `Field`.** Put the description and any constraints on the type. They go into the schema and are enforced when the tool is called:

```python
from typing import Annotated

from pydantic import Field

@server.tool()
async def get_order_status(
    order_id: Annotated[str, Field(description='The order ID, for example "A-1001".')],
    limit: Annotated[int, Field(ge=1, le=50, description="Max shipping events.")] = 10,
) -> dict:
    """Get the status of an order and its shipping events."""
    ...
```

```json
{
  "type": "object",
  "properties": {
    "order_id": {"type": "string", "title": "Order Id", "description": "The order ID, for example \"A-1001\"."},
    "limit": {"type": "integer", "title": "Limit", "default": 10, "minimum": 1, "maximum": 50, "description": "Max shipping events."}
  },
  "required": ["order_id"]
}
```

A `Field(...)` as the default works the same way: `order_id: str = Field(description="...")` is a required parameter, and `limit: int = Field(default=10, ge=1)` is an optional one.

**The docstring `Args:` section.** Any parameter without a `Field` description takes its description from the docstring. Google style (`name: text` or `name (type): text` under `Args:`, `Arguments:` or `Parameters:`) and Sphinx style (`:param name: text`) are both read. A description that wraps onto indented continuation lines is joined into one line:

```python
@server.tool()
async def get_order_status(order_id: str, include_items: bool = False) -> dict:
    """Get the status of an order.

    Args:
        order_id: The order ID, for example "A-1001". Case
            sensitive.
        include_items: Also return the line items.
    """
    ...
```

If both are given, the `Field` description wins. Constraints from `Field` are kept when the description comes from the docstring. Injected parameters (`RequestContext`, `Depends(...)`) never appear in the schema, even if the docstring documents them. NumPy-style `Parameters` / `----------` blocks are not parsed. Use `Field` for those.

### How the description is chosen

The tool description is the first of these that applies:

1. `description=` passed to `@server.tool()`.
2. The docstring's **summary paragraph**: every line up to the first blank line, section header (`Args:`, `Returns:`, `Raises:`, `Example:`, ...) or Sphinx field (`:param ...:`). The lines are joined with single spaces, so a summary that wraps over two lines comes through as one sentence.
3. The function name.

Later paragraphs are **not** sent. In Python they are usually notes for maintainers (implementation details, caveats, examples), and every client gets the description with each `tools/list`. The `Args:` section goes into the parameter schema instead. To send more guidance to the model, such as when to use the tool or how it relates to others, put it in the summary paragraph or pass `description=`. Resources and prompts use the same rule, and prompt arguments read the `Args:` section the same way.

```python
@server.tool()
async def get_order_status(order_id: str) -> dict:
    """Get the current status of an order, including the carrier and
    the expected delivery date.

    Reads from the replica, so it can lag the primary by a few seconds.
    """
    ...

# description: "Get the current status of an order, including the carrier and the expected delivery date."
```

### Custom name and description

```python
@server.tool(name="find_users", description="Find users by name or email.")
async def user_search(query: str) -> list[dict]:
    return await db.find_users(query)
```

### Pydantic model parameters

For complex inputs, use Pydantic models. The framework generates a nested JSON Schema that MCP clients can use to construct valid requests.

```python
from pydantic import BaseModel, Field

class Address(BaseModel):
    street: str = Field(description="Street name and number")
    city: str = Field(description="City name")
    zip_code: str = Field(pattern=r"^\d{5}(-\d{4})?$")

class CreateUserRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    email: str
    address: Address
    tags: list[str] = Field(default_factory=list)

@server.tool(tags=["users", "write"])
async def create_user(user: CreateUserRequest) -> dict:
    """Create a new user with validated address."""
    return {"id": "u-123", "name": user.name}
```

The `tags` parameter is metadata for categorisation and does not affect behaviour.

### Tool-level options

The `@server.tool()` decorator accepts several options:

```python
@server.tool(
    name="delete_user",          # Override tool name
    description="Remove a user", # Override description
    tags=["users", "admin"],     # Categorisation tags
    auth=True,                   # Require authentication
    roles=["admin"],             # Require specific roles (shorthand for HasRole guard)
    guards=[RequireAuth()],      # Access control guards
    rate_limit="100/min",        # Per-tool rate limit string
    timeout=10.0,                # Per-call timeout in seconds
    max_concurrent=5,            # Per-tool concurrency limit
    # MCP spec annotations (hints for clients):
    title="Delete User",         # Human-readable title
    read_only_hint=False,        # Tool modifies state
    destructive_hint=True,       # Tool performs destructive action
    idempotent_hint=True,        # Safe to retry
    open_world_hint=False,       # No external system interaction
)
async def delete_user(user_id: str) -> dict:
    """Delete a user by ID."""
    return {"deleted": user_id}
```

Two more options carry server-wide invariants: `requires_approval=True`
requires a human decision before every call — the server **refuses to build**
unless an `ApprovalGateMiddleware` is installed (see
[Approval Gates](approval-gates.md)) — and tenant access is enforced with the
tenant guards or `MCPServer(require_tenant=True)` (see
[Multi-Tenancy](multi-tenancy.md)).

### Structured tool outputs

By default, tool return values are serialised to JSON text. For richer responses, return MCP content types directly:

```python
from promptise.mcp.server import MCPServer, ImageContent

server = MCPServer(name="charts")

@server.tool()
async def generate_chart(data: list[float]) -> ImageContent:
    """Generate a bar chart as PNG."""
    png_bytes = render_chart(data)
    return ImageContent(data=png_bytes, mime_type="image/png")
```

You can also return mixed content lists:

```python
from mcp.types import TextContent

@server.tool()
async def analyze(query: str) -> list:
    """Return analysis text with a chart."""
    chart_png = await render_chart(query)
    return [
        TextContent(type="text", text="Analysis complete:"),
        ImageContent(data=chart_png, mime_type="image/png"),
    ]
```

Supported return types:

| Return type | Behaviour |
|---|---|
| `str`, `int`, `float`, etc. | Wrapped in `TextContent` |
| `dict`, `list` (plain data) | JSON-serialised into `TextContent` |
| `ImageContent(data, mime_type)` | Converted to MCP `ImageContent` (base64-encoded) |
| `TextContent(type="text", text=...)` | Passed through as-is |
| `EmbeddedResource(...)` | Passed through as-is |
| Mixed `list` of the above | Each item serialised individually |

## Registering Resources

Resources expose static, read-only data at a fixed URI.

```python
@server.resource("config://app", mime_type="application/json")
async def app_config() -> str:
    """Server configuration."""
    return '{"version": "1.0.0", "environment": "production"}'
```

### Resource templates

For dynamic URIs, use `@server.resource_template()` with `{param}` placeholders:

```python
@server.resource_template(
    "users://{user_id}/profile",
    mime_type="application/json",
)
async def user_profile(user_id: str) -> str:
    """Fetch a user profile by ID."""
    return json.dumps({"user_id": user_id, "name": "Alice"})
```

## Registering Prompts

Prompts are reusable templates that clients discover and render with arguments.

```python
@server.prompt()
async def summarize(text: str, style: str = "concise") -> str:
    """Summarize the given text."""
    return f"Please summarize the following text in a {style} style:\n\n{text}"
```

## Lifecycle Hooks

Run setup and teardown logic with `@server.on_startup` and `@server.on_shutdown`:

```python
@server.on_startup
async def startup():
    """Initialize database pool on server start."""
    await db.connect()
    print("Database connected")

@server.on_shutdown
async def shutdown():
    """Clean up resources on server stop."""
    await db.disconnect()
    print("Database disconnected")
```

Startup hooks run before the server begins accepting requests. Shutdown hooks run during graceful shutdown (configurable via `shutdown_timeout`).

## Server Configuration

### Constructor options

```python
server = MCPServer(
    name="hr-platform",           # Server name advertised to clients
    version="2.1.0",              # Version string
    instructions="Use the HR tools to manage employees.",  # Sent to clients on init
    auto_manifest=True,           # Register a manifest resource (default: True)
    shutdown_timeout=30.0,        # Graceful shutdown timeout in seconds
    require_auth=False,           # Force auth=True on all tools
)
```

### ServerSettings for environment variables

Subclass `ServerSettings` to load configuration from environment variables using Pydantic's `BaseSettings`:

```python
from promptise.mcp.server import ServerSettings, Depends

class AppSettings(ServerSettings):
    database_url: str = "sqlite:///local.db"
    max_results: int = 100

    model_config = {"env_prefix": "MY_APP_", "extra": "ignore"}

@server.tool()
async def info(settings: AppSettings = Depends(AppSettings)) -> dict:
    """Return current configuration."""
    return {"db": settings.database_url, "max": settings.max_results}
```

Set `MY_APP_DATABASE_URL` and `MY_APP_MAX_RESULTS` in the environment to override defaults.

## Running the Server

`server.run()` is a blocking call that starts the transport and event loop:

```python
# stdio (default) -- for Claude Desktop, Cursor, etc.
server.run()

# HTTP (Streamable HTTP) -- for remote agents and web clients
server.run(transport="http", host="127.0.0.1", port=8080)

# SSE (Server-Sent Events)
server.run(transport="sse", host="0.0.0.0", port=9090)

# HTTP with live terminal dashboard
server.run(transport="http", host="127.0.0.1", port=8080, dashboard=True)
```

For async code, use `await server.run_async(...)` instead.

## Complete Example

A full server with Pydantic validation, lifecycle hooks, and HTTP transport:

```python
import json
from pydantic import BaseModel, Field
from promptise.mcp.server import MCPServer

server = MCPServer(name="bookstore", version="1.0.0")

_books: dict[str, dict] = {}

class Book(BaseModel):
    title: str = Field(min_length=1, description="Book title")
    author: str = Field(description="Author name")
    year: int = Field(ge=1450, description="Publication year")

@server.on_startup
async def seed():
    _books["b-1"] = {"id": "b-1", "title": "Dune", "author": "Frank Herbert", "year": 1965}

@server.tool(tags=["books", "read"])
async def list_books() -> list[dict]:
    """List all books in the store."""
    return list(_books.values())

@server.tool(tags=["books", "write"])
async def add_book(book: Book) -> dict:
    """Add a book to the store."""
    book_id = f"b-{len(_books) + 1}"
    record = {"id": book_id, **book.model_dump()}
    _books[book_id] = record
    return record

@server.resource("catalog://stats", mime_type="application/json")
async def catalog_stats() -> str:
    return json.dumps({"total_books": len(_books)})

if __name__ == "__main__":
    server.run(transport="http", host="127.0.0.1", port=8080)
```

## API Summary

| Symbol | Type | Description |
|---|---|---|
| `MCPServer(name, version, ...)` | Class | Main server class |
| `@server.tool(...)` | Decorator | Register a tool handler |
| `@server.resource(uri, ...)` | Decorator | Register a static resource |
| `@server.resource_template(uri, ...)` | Decorator | Register a resource template |
| `@server.prompt(...)` | Decorator | Register a prompt |
| `@server.on_startup` | Decorator | Register a startup hook |
| `@server.on_shutdown` | Decorator | Register a shutdown hook |
| `server.add_middleware(mw)` | Method | Add middleware to the chain |
| `server.include_router(router)` | Method | Merge a router into the server |
| `server.run(transport, host, port)` | Method | Start the server (blocking) |
| `server.run_async(...)` | Method | Start the server (async) |
| `ServerSettings` | Class | Base class for env-var settings |

!!! tip "Sync and async handlers"
    Both sync and async handler functions are supported. The framework automatically awaits coroutines. Use async handlers for I/O-bound work and sync handlers for simple computations.

!!! warning "Pydantic v2 required"
    The server uses Pydantic v2 for input validation and schema generation. Pydantic v1 models are not supported.

!!! tip "Auto-generated manifest"
    By default, `MCPServer` registers a `manifest://server` resource containing a JSON summary of all tools, resources, and prompts. Disable with `auto_manifest=False`.

## What's Next?

- [Routers & Middleware](routers-middleware.md) -- Organize tools into modules and add cross-cutting concerns
- [Authentication & Security](auth-security.md) -- Secure your server with JWT, API keys, and guards
- [Production Features](production-features.md) -- Add caching, rate limiting, health checks, and metrics
- [Testing](testing.md) -- Test servers in-process with `TestClient`
- [Step-by-Step Guide](../../guides/production-mcp-servers.md) -- Build a complete server incrementally
