# Advanced Patterns

Patterns for evolving, composing, and extending MCP servers — tool versioning for backwards compatibility, transforms for dynamic tool visibility, server composition for microservice gateways, OpenAPI bridging, batch execution, streaming results, elicitation, and sampling.

## Tool Versioning

### When you need it

You ship `search` v1 to production. Agents hard-code `search` in their prompts. Now you want to add a `filters` parameter — but you can't change the signature without breaking every agent that already uses v1. Versioned tools let you ship v2 while keeping v1 available.

### `VersionedToolRegistry`

```python
from promptise.mcp.server import MCPServer, VersionedToolRegistry

server = MCPServer(name="search-api")

@server.tool(version="1.0")
async def search(query: str) -> list[dict]:
    """Full-text search across documents.

    v1: Basic query string matching.
    """
    return await db.full_text_search(query)

@server.tool(version="2.0")
async def search(
    query: str,
    filters: dict | None = None,
    sort_by: str = "relevance",
) -> list[dict]:
    """Full-text search across documents.

    v2: Adds filtering and sorting.
    """
    results = await db.full_text_search(query, filters=filters)
    return sorted(results, key=lambda r: r.get(sort_by, 0), reverse=True)
```

What agents see:

| Tool name | Description |
|-----------|-------------|
| `search` | Points to the latest version (v2) |
| `search@1.0` | Pinned to v1 (basic search) |
| `search@2.0` | Pinned to v2 (with filters) |

### Working with versions programmatically

```python
vr = VersionedToolRegistry()
vr.register("search", "1.0", tool_def_v1)
vr.register("search", "2.0", tool_def_v2)

# Get latest
vr.get("search")          # → v2.0 ToolDef

# Get specific version
vr.get("search@1.0")      # → v1.0 ToolDef

# List all versions
vr.list_versions("search")  # → ["1.0", "2.0"]

# Check if versioned
vr.has("search")           # → True
```

### Version comparison

Versions are compared semantically: `"2.0"` > `"1.0"`, `"1.10"` > `"1.9"`. The `latest` property always returns the highest version.

---

## Tool Transforms

### When you need it

You have 50 tools registered. Some are admin-only, some are experimental. You don't want every agent to see all 50 — you want to control what each agent discovers based on context.

### `NamespaceTransform`

Prefix tool names when composing multiple servers or separating concerns:

```python
from promptise.mcp.server import MCPServer, NamespaceTransform

server = MCPServer(name="analytics-api")

@server.tool()
async def query(sql: str) -> list[dict]:
    """Run an analytics query."""
    return await analytics_db.execute(sql)

@server.tool()
async def export(format: str = "csv") -> str:
    """Export latest report."""
    return await reports.export(format)

# Agents see: analytics_query, analytics_export
server.add_transform(NamespaceTransform(prefix="analytics"))
```

### `VisibilityTransform`

Hide tools based on who's asking:

```python
from promptise.mcp.server import MCPServer, VisibilityTransform

server = MCPServer(name="admin-api")

@server.tool()
async def list_users() -> list[dict]:
    """List all users."""
    return await db.list_users()

@server.tool()
async def delete_user(user_id: str) -> dict:
    """Delete a user permanently."""
    return await db.delete_user(user_id)

@server.tool()
async def reset_database() -> str:
    """Reset the entire database."""
    await db.reset()
    return "Database reset complete"

# Hide destructive tools from non-admin agents
server.add_transform(VisibilityTransform(
    hidden={
        "delete_user": lambda ctx: "admin" not in (ctx.state.get("roles") if ctx else set()),
        "reset_database": lambda ctx: "superadmin" not in (ctx.state.get("roles") if ctx else set()),
    }
))
```

Hidden tools are removed from `list_tools` results but remain callable if an agent already has the tool name cached. This is a soft visibility control, not a security boundary — use guards for that.

### `TagFilterTransform`

Only expose tools that match required tags:

```python
from promptise.mcp.server import MCPServer, TagFilterTransform

server = MCPServer(name="multi-tier-api")

@server.tool(tags=["public", "read"])
async def get_product(product_id: str) -> dict:
    """Get product details."""
    return await catalog.get(product_id)

@server.tool(tags=["internal", "write"])
async def update_inventory(product_id: str, quantity: int) -> dict:
    """Update inventory count."""
    return await inventory.update(product_id, quantity)

@server.tool(tags=["admin", "write"])
async def set_pricing(product_id: str, price_cents: int) -> dict:
    """Set product pricing."""
    return await pricing.set(product_id, price_cents)

# External agents only see tools tagged "public"
server.add_transform(TagFilterTransform(required_tags={"public"}))
```

### Custom transforms

Transforms implement the `ToolTransform` protocol:

```python
from promptise.mcp.server import ToolTransform, ToolDef, RequestContext
from dataclasses import replace

class DescriptionRewriter:
    """Append usage hints to tool descriptions."""

    def apply(
        self,
        tools: list[ToolDef],
        ctx: RequestContext | None = None,
    ) -> list[ToolDef]:
        result = []
        for t in tools:
            new_desc = f"{t.description}\n\nUsage: call with JSON arguments."
            result.append(replace(t, description=new_desc))
        return result
```

Transforms are applied in order — each sees the output of the previous one. This lets you compose them:

```python
server.add_transform(NamespaceTransform(prefix="myapp"))
server.add_transform(TagFilterTransform(required_tags={"public"}))
# Result: only public tools, prefixed with "myapp_"
```

---

## Server Composition

### When you need it

Your company has separate teams building separate MCP servers — `payments`, `users`, `analytics`. You want to expose them all through a single gateway server so agents only connect to one endpoint.

### `mount()`

```python
from promptise.mcp.server import MCPServer, mount

# Team 1: Payment tools
payments = MCPServer(name="payments")

@payments.tool()
async def charge(customer_id: str, amount_cents: int) -> dict:
    """Charge a customer's payment method."""
    return await stripe.charge(customer_id, amount_cents)

@payments.tool()
async def refund(charge_id: str) -> dict:
    """Refund a charge."""
    return await stripe.refund(charge_id)

# Team 2: User tools
users = MCPServer(name="users")

@users.tool()
async def get_user(user_id: str) -> dict:
    """Get user profile."""
    return await db.get_user(user_id)

@users.tool()
async def update_user(user_id: str, name: str | None = None) -> dict:
    """Update user profile."""
    return await db.update_user(user_id, name=name)

# Gateway: compose into one server
gateway = MCPServer(name="api-gateway", version="1.0.0")
mount(gateway, payments, prefix="pay")
mount(gateway, users, prefix="usr")

# Agents see: pay_charge, pay_refund, usr_get_user, usr_update_user
gateway.run(transport="http", port=8080)
```

### What gets mounted

`mount()` copies everything from the child into the parent:

| Registry | Behavior |
|----------|----------|
| **Tools** | Names prefixed with `{prefix}_` |
| **Resources** | Copied as-is (no prefix) |
| **Prompts** | Copied as-is |
| **Exception handlers** | Merged into parent |
| **Input models** | Re-mapped to prefixed tool names |

### Adding tags during mount

```python
# Tag all payment tools for filtering
mount(gateway, payments, prefix="pay", tags=["payment", "billing"])
mount(gateway, users, prefix="usr", tags=["user-management"])
```

---

## OpenAPI Bridge

### When you need it

You have an existing REST API with an OpenAPI spec (Swagger). You want agents to call it through MCP without writing individual tool wrappers for each endpoint.

### `OpenAPIProvider`

```python
from promptise.mcp.server import MCPServer, OpenAPIProvider

server = MCPServer(name="github-bridge")

# Load from URL
provider = OpenAPIProvider(
    "https://raw.githubusercontent.com/github/rest-api-description/main/descriptions/api.github.com/api.github.com.json",
    prefix="gh_",
    include={"repos_list_for_user", "repos_get", "issues_list"},
    auth_header=("Authorization", f"Bearer {GITHUB_TOKEN}"),
    tags=["github"],
)
provider.register(server)

server.run()
```

Each OpenAPI operation becomes an MCP tool that makes HTTP requests to the target API.

### Spec loading options

```python
# From a URL (requires httpx)
provider = OpenAPIProvider("https://api.example.com/openapi.json")

# From a local file (JSON or YAML)
provider = OpenAPIProvider("./specs/api.yaml")

# From a pre-parsed dict
spec = {"openapi": "3.0.0", "paths": {...}}
provider = OpenAPIProvider(spec)
```

### Filtering operations

```python
# Only include specific operations
provider = OpenAPIProvider(spec, include={"getUser", "listUsers"})

# Exclude specific operations
provider = OpenAPIProvider(spec, exclude={"deleteUser", "resetDatabase"})
```

### Tool annotations

The provider automatically sets MCP tool annotations based on the HTTP method:

| HTTP method | `read_only_hint` | `destructive_hint` | `idempotent_hint` |
|-------------|------------------|---------------------|-------------------|
| `GET` | `True` | `False` | `True` |
| `POST` | `False` | `False` | `False` |
| `PUT` | `False` | `False` | `True` |
| `DELETE` | `False` | `True` | `True` |

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spec` | (required) | URL, file path, or dict |
| `base_url` | From spec | Override API base URL |
| `prefix` | `""` | Prefix for tool names |
| `include` | All | Only these operation IDs |
| `exclude` | None | Skip these operation IDs |
| `auth_header` | None | `(header_name, value)` tuple |
| `tags` | `[]` | Tags for all generated tools |

---

## Batch Tool Calls

### When you need it

An agent needs to fetch 20 user profiles. Without batching, that's 20 sequential MCP round-trips. With a batch tool, it's one request that executes all 20 in parallel.

### `register_batch_tool()`

```python
from promptise.mcp.server import MCPServer, register_batch_tool

server = MCPServer(name="user-api")

@server.tool()
async def get_user(user_id: str) -> dict:
    """Get a user by ID."""
    return await db.get_user(user_id)

@server.tool()
async def get_team(team_id: str) -> dict:
    """Get a team by ID."""
    return await db.get_team(team_id)

# Register the batch meta-tool
register_batch_tool(server, name="batch_call", max_parallel=10)
```

Agents call it like this:

```json
{
  "tool": "batch_call",
  "arguments": {
    "calls": [
      {"tool": "get_user", "args": {"user_id": "u-001"}},
      {"tool": "get_user", "args": {"user_id": "u-002"}},
      {"tool": "get_team", "args": {"team_id": "t-100"}}
    ]
  }
}
```

Response:

```json
[
  {"tool": "get_user", "status": "ok", "result": ["..."]},
  {"tool": "get_user", "status": "ok", "result": ["..."]},
  {"tool": "get_team", "status": "ok", "result": ["..."]}
]
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `name` | `"batch_call"` | Name of the batch tool |
| `description` | `"Execute multiple tool calls in parallel."` | Description |
| `max_parallel` | `10` | Maximum concurrent executions |

The total number of calls in a single batch is capped at `max_parallel * 2`. Failed individual calls return `{"status": "error", "error": "..."}` without failing the batch.

---

## Streaming Results

### When you need it

Your tool processes a large dataset or streams search results. Instead of accumulating everything in memory and returning a flat dict, you build up results incrementally and return them with metadata.

### `StreamingResult`

```python
from promptise.mcp.server import MCPServer, StreamingResult

server = MCPServer(name="search-api")

@server.tool()
async def search_products(
    query: str,
    max_results: int = 50,
) -> StreamingResult:
    """Search the product catalog.

    Returns results as they're found, with search metadata.
    """
    result = StreamingResult()
    result.set_metadata("query", query)

    async for hit in catalog.stream_search(query):
        result.add({
            "id": hit.id,
            "name": hit.name,
            "price": hit.price,
            "score": hit.relevance_score,
        })
        if len(result) >= max_results:
            break

    result.set_metadata("total_available", await catalog.count(query))
    return result
```

The framework serializes `StreamingResult` as JSON:

```json
{
  "items": [
    {"id": "p-001", "name": "Widget", "price": 999, "score": 0.95},
    {"id": "p-002", "name": "Gadget", "price": 1499, "score": 0.87}
  ],
  "count": 2,
  "metadata": {
    "query": "electronics",
    "total_available": 1250
  }
}
```

### Batch adding

```python
result = StreamingResult()

# Add one at a time
result.add({"id": 1, "name": "Alice"})

# Add many at once
result.add_many([
    {"id": 2, "name": "Bob"},
    {"id": 3, "name": "Charlie"},
])

# Attach metadata
result.set_metadata("source", "users_table")
result.set_metadata("cached", False)

len(result)         # → 3
result.items        # → [{"id": 1, ...}, {"id": 2, ...}, {"id": 3, ...}]
result.metadata     # → {"source": "users_table", "cached": False}
```

---

## Elicitation

### When you need it

Your tool is about to deploy to production or delete data. You want to ask the user for confirmation **during** tool execution, not before.

### `Elicitor`

```python
from promptise.mcp.server import MCPServer, Elicitor, Depends

server = MCPServer(name="deploy-api")

@server.tool()
async def deploy_to_production(
    service: str,
    version: str,
    elicit: Elicitor = Depends(Elicitor),
) -> dict:
    """Deploy a service version to production.

    Asks the user for confirmation before proceeding.
    """
    # Show what's about to happen and ask for confirmation
    answer = await elicit.ask(
        message=(
            f"About to deploy {service} v{version} to production.\n"
            f"Current production version: {await get_current_version(service)}\n"
            f"Confirm deployment?"
        ),
        schema={
            "type": "object",
            "properties": {
                "confirm": {
                    "type": "boolean",
                    "description": "Set to true to proceed with deployment",
                },
                "reason": {
                    "type": "string",
                    "description": "Optional: reason for deployment",
                },
            },
            "required": ["confirm"],
        },
    )

    if not answer or not answer.get("confirm"):
        return {"status": "cancelled", "reason": answer.get("reason", "User declined")}

    deploy_id = await deploy(service, version)
    return {"status": "deployed", "deploy_id": deploy_id, "version": version}
```

### Graceful degradation

`ask()` sends the request through the SDK's `ServerSession.elicit`, waits up to its `timeout` (default from the `Elicitor`), and returns the response `content` only when the user **accepts**. If the client does not support elicitation (not all MCP clients do), declines, cancels, or the request times out, `ask()` returns `None` and logs the reason at `WARNING`. Design your tools to handle this:

```python
answer = await elicit.ask("Confirm deletion?", schema={...})
if answer is None:
    # Client doesn't support elicitation — proceed with caution
    # or return an error asking the user to confirm manually
    return {"error": "This action requires confirmation. Please confirm and retry."}
```

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeout` | `60.0` | Seconds to wait for user response |

---

## Sampling

### When you need it

Your tool needs to call an LLM to process data — summarize text, classify content, extract entities — but the **client** controls which LLM to use. MCP sampling lets the server request completions through the client's model.

### `Sampler`

```python
from promptise.mcp.server import MCPServer, Sampler, Depends

server = MCPServer(name="content-api")

@server.tool()
async def summarize_document(
    document_url: str,
    sampler: Sampler = Depends(Sampler),
) -> dict:
    """Summarize a document using the client's LLM.

    The server fetches the document; the client provides the model.
    """
    content = await fetch_document(document_url)

    summary = await sampler.create_message(
        messages=[
            {"role": "user", "content": f"Summarize this document in 3 bullet points:\n\n{content[:5000]}"},
        ],
        max_tokens=500,
        system="You are a professional document summarizer. Be concise and factual.",
        temperature=0.3,
    )

    if summary is None:
        return {"error": "Sampling not supported by client"}

    return {
        "url": document_url,
        "summary": summary,
        "model": "client-provided",
    }
```

### Parameters

```python
result = await sampler.create_message(
    messages=[{"role": "user", "content": "..."}],
    max_tokens=1024,          # Maximum tokens to generate
    model="claude-sonnet-4-20250514",      # Model hint (client may ignore)
    system="You are...",      # System prompt
    temperature=0.7,          # Sampling temperature
    stop_sequences=["\n\n"],  # Stop sequences
)
```

The client controls which model actually runs. The `model` parameter is a hint — the client can override it based on cost, availability, or policy.

---

## Server Manifest

### When you need it

You want agents to introspect your server — discover all tools, their schemas, tags, auth requirements, and rate limits — without calling each tool individually.

### `register_manifest()`

```python
from promptise.mcp.server import MCPServer, register_manifest

server = MCPServer(name="my-api", version="2.0.0")

@server.tool(tags=["math"], roles=["user"])
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@server.tool(tags=["math"], auth=True)
async def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b

# Auto-registers a docs://manifest resource
register_manifest(server)
```

Agents read the manifest via MCP's resource protocol:

```json
{
  "server": {"name": "my-api", "version": "2.0.0"},
  "tools": [
    {
      "name": "add",
      "description": "Add two numbers.",
      "input_schema": {"type": "object", "properties": {"a": {...}, "b": {...}}},
      "tags": ["math"],
      "roles": ["user"]
    },
    {
      "name": "divide",
      "description": "Divide two numbers.",
      "input_schema": {...},
      "tags": ["math"],
      "auth_required": true
    }
  ],
  "resources": [...],
  "prompts": [...]
}
```

### Programmatic access

```python
from promptise.mcp.server import build_manifest

manifest = build_manifest(server)
print(manifest["tools"])  # All registered tools with metadata
```

---

## Hot Reload

### When you need it

You're developing an MCP server and want to see changes immediately without restarting manually. Hot reload watches your Python files and restarts the server when you save.

### `hot_reload()`

```python
from promptise.mcp.server import MCPServer, hot_reload

server = MCPServer(name="dev-server")

@server.tool()
async def greet(name: str) -> str:
    """Greet someone."""
    return f"Hello, {name}!"

if __name__ == "__main__":
    hot_reload(
        server,
        transport="http",
        port=8080,
        watch_dirs=["src/"],       # Directories to watch
        poll_interval=1.0,          # Check every second
    )
```

```
--- Starting server (pid will follow) ---
    Watching: src/
    Poll interval: 1.0s

MCP server running on http://127.0.0.1:8080/mcp

--- Detected changes in 1 file(s) ---
    src/tools.py
--- Restarting server ---
```

### How it works

1. The parent process watches `*.py` files for modification time changes
2. When a change is detected, the child server process is terminated
3. A new child process is started with the updated code
4. If the server crashes on startup (syntax error), it waits for the next file change

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `transport` | `"http"` | Transport for the child server |
| `host` | `"127.0.0.1"` | Bind host |
| `port` | `8080` | Bind port |
| `watch_dirs` | `["."]` | Directories to watch |
| `poll_interval` | `1.0` | Seconds between checks |
| `dashboard` | `False` | Enable dashboard in child |

!!! warning "Development only"
    Hot reload is for development. In production, use a process manager like systemd, supervisord, or Kubernetes.

---

## CLI Serve

### When you need it

You've built an MCP server in a Python module but don't want to add `if __name__ == "__main__"` boilerplate. The CLI `serve` command runs any server from the command line.

### Usage

```bash
# Run with stdio transport (default)
promptise serve myapp.server:server

# Run with HTTP transport
promptise serve myapp.server:server --transport http --port 8080

# With hot reload (development)
promptise serve myapp.server:server --transport http --reload

# With live dashboard
promptise serve myapp.server:server --transport http --dashboard
```

### Target format

The target uses `module:attribute` format:

```bash
# module.path:attribute_name
promptise serve myapp.server:server     # myapp/server.py → server variable
promptise serve myapp.api:app           # myapp/api.py → app variable
promptise serve tools:tool_server       # tools.py → tool_server variable
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--transport`, `-t` | `stdio` | Transport: `stdio`, `http`, or `sse` |
| `--host` | `127.0.0.1` | Bind host |
| `--port`, `-p` | `8080` | Bind port |
| `--dashboard` | off | Enable live dashboard |
| `--reload` | off | Enable hot reload |

---

## API Summary

| Symbol | Type | Description |
|--------|------|-------------|
| `VersionedToolRegistry` | Class | Manage multiple tool versions |
| `ToolTransform` | Protocol | Interface for tool list transforms |
| `NamespaceTransform(prefix)` | Class | Prefix tool names |
| `VisibilityTransform(hidden)` | Class | Hide tools by predicate |
| `TagFilterTransform(required_tags)` | Class | Filter tools by tags |
| `mount(parent, child, prefix, tags)` | Function | Compose servers |
| `OpenAPIProvider(spec, ...)` | Class | Generate tools from OpenAPI |
| `register_batch_tool(server, ...)` | Function | Parallel batch execution |
| `StreamingResult` | Class | Incremental result collection |
| `Elicitor` | Class | Ask user for input mid-execution (via DI) |
| `Sampler` | Class | Request LLM completions from client (via DI) |
| `build_manifest(server)` | Function | Build server manifest dict |
| `register_manifest(server)` | Function | Register `docs://manifest` resource |
| `hot_reload(server, ...)` | Function | File-watching server restarter |
| `build_serve_parser(...)` | Function | CLI argument parser |
| `resolve_server(target)` | Function | Import server from `module:attr` |
| `run_serve(args)` | Function | Execute CLI serve command |

## What's Next

- [Deployment](deployment.md) — HTTP transport, CORS, containers, reverse proxy
- [Caching & Performance](caching-performance.md) — Cache, rate limit, concurrency control
- [Observability & Monitoring](observability.md) — Metrics, tracing, Prometheus, logging
- [Resilience Patterns](resilience-patterns.md) — Circuit breakers, health checks, webhooks
