---
seo_title: "MCP Tool Versioning, Transforms, Composition"
description: "Advanced MCP server patterns: tool versioning, transforms, server composition, an OpenAPI bridge, batch calls, streaming results, elicitation and sampling."
---

# Advanced MCP Server Patterns

Patterns for evolving, composing, and extending MCP servers: tool versioning for backwards compatibility, controlling which tools each caller sees, server composition for gateways, OpenAPI bridging, batch execution, streaming results, elicitation, and sampling.

## Tool Versioning

### When you need it

You ship `search` to production and agents depend on its arguments. Now you want to add a `filters` parameter without breaking the agents that use the old shape.

### `@server.tool(version=...)`

Register each version of the tool with `version=`. Every version is served under a pinned name, `search_v1_0` and `search_v2_0`, and the bare name `search` runs the highest version:

```python
from promptise.mcp.server import MCPServer

server = MCPServer(name="search-api")

DOCS = [{"id": 1, "title": "Refund policy", "score": 0.9}, {"id": 2, "title": "Shipping", "score": 0.4}]

@server.tool(version="1.0")
async def search(query: str) -> list[dict]:
    """Full-text search."""
    return [d for d in DOCS if query.lower() in d["title"].lower()]

@server.tool(version="2.0")
async def search(query: str, min_score: float = 0.0) -> list[dict]:
    """Full-text search with an optional minimum score."""
    return [d for d in DOCS if query.lower() in d["title"].lower() and d["score"] >= min_score]

# tools/list: search (runs 2.0), search_v1_0, search_v2_0
```

An agent that binds to `search_v1_0` keeps the old contract when you ship `3.0`; agents that use `search` move to each new version. How it behaves:

- **Pinned names** are `<name>_v<version>`, with every run of characters other than letters and digits replaced by `_` (`"2.1-beta"` gives `search_v2_1_beta`). They use only characters LLM providers accept in function names, so agents can call them. MCP has no per-tool version field, so the version lives in the name.
- **The bare name** is a copy of the highest version: same description, schema, guards and settings. Versions are compared numerically per dot-separated part: `"2.0"` > `"1.0"`, `"1.10"` > `"1.9"` (a non-numeric part counts as 0), not by registration order.
- **Each version is its own tool**: its own input schema and validation, `auth`, `roles`, `rate_limit` and other options, as given on its decorator.
- **Every registration of a versioned name needs `version=`.** Registering the same version twice, a version next to an unversioned tool of the same name (in either order), or a version with no letter or digit raises `ValueError`.
- **Routers**: `@router.tool(version=...)` works the same; the router's prefix applies to all names (`billing_charge`, `billing_charge_v1_0`).

New optional parameters with defaults (like `min_score` above) are backwards compatible on their own: an agent that doesn't send them gets the old behaviour, so a version is only needed for a breaking change (a renamed or removed parameter, a different result shape).

### `VersionedToolRegistry`

`VersionedToolRegistry` is the registry behind `version=` (`server._tool_versions`). It also works on its own for code that keeps several `ToolDef` versions under one base name, for example a gateway that builds its own tool list, and resolves `name@version` there:

```python
from promptise.mcp.server import VersionedToolRegistry, ToolDef

async def handler(query: str) -> list:
    return []

v1 = ToolDef(name="search_v1_0", description="v1", handler=handler, input_schema={"type": "object"})
v2 = ToolDef(name="search_v2_0", description="v2", handler=handler, input_schema={"type": "object"})

vr = VersionedToolRegistry()
vr.register("search", "1.0", v1)
vr.register("search", "2.0", v2)

print(vr.get("search").description)      # "v2": the highest version
print(vr.get("search@1.0").description)  # "v1"
print(vr.list_versions("search"))        # ["1.0", "2.0"]
print(vr.has("search"))                  # True
print(len(vr.list_all()))                # 2: every registered version
```

Registering the same version twice raises `ValueError`; `get()` returns `None` for an unknown name or version. Registering a `ToolDef` in a registry you created yourself does not serve it; use `@server.tool(version=...)` for that.

---

## Tool Transforms

### When you need it

You want different agents to discover different tools: prefixed names, hidden admin tools, only public tools, descriptions rewritten for one audience.

### `server.add_transform()`

A transform changes the tool list clients discover. `server.add_transform(transform)` adds one; transforms run in the order they were added, on every `tools/list` and on the `docs://manifest` resource, each on the output of the previous one (after `hide_unauthorized_tools` filtering).

```python
from promptise.mcp.server import MCPServer, NamespaceTransform, TagFilterTransform

server = MCPServer(name="analytics")

@server.tool(tags=["public"])
async def query(sql: str) -> list[dict]:
    """Run a read-only query."""
    return []

@server.tool(tags=["admin"], roles=["admin"])
async def reset() -> str:
    """Reset the warehouse."""
    return "reset"

server.add_transform(TagFilterTransform(required_tags={"public"}))
server.add_transform(NamespaceTransform(prefix="analytics"))
# tools/list: analytics_query
```

| Transform | What it does |
|-----------|--------------|
| `NamespaceTransform(prefix)` | Lists every tool as `<prefix>_<name>`. A call by the prefixed name runs the tool; the registered name keeps working too. |
| `TagFilterTransform(required_tags)` | Lists only tools with at least one of the tags. |
| `VisibilityTransform(hidden={name: predicate})` | Leaves out a tool when `predicate(ctx)` returns true. A predicate that raises hides its tool. |

Each transform receives the listing request's context. When the server has an `AuthMiddleware`, the context is authenticated from the request's headers first, so `ctx.client_id`, `ctx.client` and `ctx.state["roles"]` describe the caller (they are empty for a caller whose credentials don't verify):

```python
from promptise.mcp.server import APIKeyAuth, AuthMiddleware, VisibilityTransform

server.add_middleware(AuthMiddleware(APIKeyAuth({"sk-ops": "ops", "sk-bi": "bi-team"})))
server.add_transform(VisibilityTransform(hidden={"analytics_reset": lambda ctx: ctx.client_id != "ops"}))
```

!!! warning "Transforms change discovery, not access"
    A tool a transform leaves out can still be called by its name: a client with an old tool list, or one that guesses the name, reaches it. Put access decisions in guards (`roles=`, `guards=`), as `reset` above does with `roles=["admin"]`, and use transforms to keep each caller's list short and relevant. `MCPServer(hide_unauthorized_tools=True)` lists only the tools a caller's guards allow, without a transform.

### Writing a transform

A transform is any object with `apply(tools: list[ToolDef], ctx: RequestContext | None = None) -> list[ToolDef]` (the `ToolTransform` protocol); `apply` may be `async`. Return copies made with `dataclasses.replace`, never change the definitions you receive. A transform that renames tools also defines `resolve_name(name) -> str | None`, which returns the registered name for a listed one (or `None` when the name is not one of its renamings), so that calls by the listed name reach the tool:

```python
from dataclasses import replace

from promptise.mcp.server import RequestContext, ToolDef

class AudienceNote:
    """Add a note for one audience to every description."""

    def apply(self, tools: list[ToolDef], ctx: RequestContext | None = None) -> list[ToolDef]:
        return [replace(t, description=f"{t.description} Results are in EUR.") for t in tools]

server.add_transform(AudienceNote())
```

The transform classes also work on their own, on any `ToolDef` list (a custom gateway, tests): `TagFilterTransform(required_tags={"public"}).apply(tools)`.

---

## Server Composition

### When you need it

Your company has separate teams building separate MCP servers: `payments`, `users`, `analytics`. You want to expose them all through a single gateway server so agents only connect to one endpoint.

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
if __name__ == "__main__":
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
| **`docs://manifest`** | Skipped (the parent serves its own) |

`mount(parent, child, *, prefix="", tags=None)` returns the number of tools mounted. It copies the child's registrations at the time you call it; later registrations on the child are not seen. Each copy keeps its `auth`, roles, guards and router middleware.

!!! warning "The child's server middleware is not mounted"
    Middleware added with `child.add_middleware(...)` (authentication, rate limits, audit) does not come along: mounted tools run through the **parent's** middleware chain. Add the middleware to the gateway. Name clashes (two tools with the same final name) raise `ValueError`.

### Adding tags during mount

```python
# Tag all payment tools for filtering
mount(gateway, payments, prefix="pay", tags=["payment", "billing"])
mount(gateway, users, prefix="usr", tags=["user-management"])
```

---

## OpenAPI Bridge

### When you need it

You have an existing REST API with an OpenAPI (or Swagger 2) spec and want agents to call it through MCP.

`OpenAPIProvider` registers one tool per operation. Each tool's input schema is the operation's parameters (path-level ones included) plus its JSON request body, with every `$ref` inlined; calls are validated against it and each argument is sent where the spec puts it: path (URL-encoded), query, header, cookie or body. Arguments the caller leaves out are not sent, so the API applies its own defaults.

```python
import os

from promptise.mcp.server import MCPServer, OpenAPIProvider

server = MCPServer(name="petstore-bridge")
provider = OpenAPIProvider(
    "./openapi.json",
    prefix="pets_",
    auth_header=("Authorization", "Bearer " + os.environ["PETSTORE_TOKEN"]),
)
count = provider.register(server)   # one tool per operation
```

Annotations come from the HTTP method: `GET`/`HEAD` are read-only, `DELETE` is destructive, `GET`/`HEAD`/`PUT`/`DELETE` are idempotent, and every tool has `open_world_hint=True`; the title is the operation's summary. A JSON object body contributes its properties as arguments (a parameter with the same name wins); any other JSON body (an array, a string) is the single `body` argument. Only `application/json` bodies are mapped. The tool returns the response's JSON, or `{"status": ..., "body": ...}` when the response is not JSON, whatever the status code.

For a curated, typed server (descriptions written for agents, selected operations, generated tests), use [MCPcast](mcpcast.md) (`promptise mcpcast ./openapi.yaml`) instead.

`OpenAPIProvider(spec, *, base_url=None, prefix="", include=None, exclude=None, auth_header=None, tags=None, allow_private=False)` and `provider.register(server) -> int` (the number of tools):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spec` | (required) | URL (fetched with `httpx`), file path (JSON, or YAML with `.yaml`/`.yml`), JSON string, or dict |
| `base_url` | From spec | Override the API base URL (`servers[0].url`, or Swagger `schemes`/`host`/`basePath`). A relative `servers` URL such as `/v1` is resolved against the spec's URL; pass `base_url` when the spec is a file or dict |
| `prefix` | `""` | Prepended to tool names as given (include your own `_`) |
| `include` | All | Only these operation IDs |
| `exclude` | None | Skip these operation IDs |
| `auth_header` | None | `(header_name, value)` sent on every request |
| `tags` | `[]` | Tags for all generated tools |
| `allow_private` | `False` | Allow a spec URL or base URL on a loopback, private or link-local address. Turn it on only for an internal API you trust |

Tool names are the `operationId` (or `<method>_<path>`) with `-` and `/` replaced by `_`.

By default, a spec URL or base URL that resolves to a private or loopback address is refused, so a spec cannot point the tools at internal services. To bridge an internal API on purpose, pass `allow_private=True`:

```python
provider = OpenAPIProvider(
    "http://10.0.4.12:8080/openapi.json",
    allow_private=True,   # this API lives on the internal network
)
```

`register()` raises a `ValueError` when the tools would have no absolute API address: the spec has no `servers` entry, or a relative one and no spec URL to resolve it against. Pass `base_url` in that case.

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

`register_batch_tool(server, *, name="batch_call", description="Execute multiple tool calls in parallel.", max_parallel=10)`. A batch larger than `max_parallel * 2` calls is refused with a single `{"status": "error", "error": "Batch size ... exceeds max ..."}` entry. Each sub-call runs through the server's full pipeline (middleware, guards, validation) with the caller's request headers, so authentication and permissions apply per sub-call. Failed individual calls return `{"tool": ..., "status": "error", "error": "..."}` without failing the batch; `result` holds the text of each content item.

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
        return {"status": "cancelled", "reason": (answer or {}).get("reason", "User declined")}

    deploy_id = await deploy(service, version)
    return {"status": "deployed", "deploy_id": deploy_id, "version": version}
```

### Graceful degradation

`ask()` sends the request through the SDK's `ServerSession.elicit`, waits up to its `timeout` (default from the `Elicitor`), and returns the response `content` only when the user **accepts**. If the client does not support elicitation (not all MCP clients do), declines, cancels, or the request times out, `ask()` returns `None` and logs the reason at `WARNING`. Design your tools to handle this:

```python
@server.tool()
async def delete_project(project_id: str, elicit: Elicitor = Depends(Elicitor)) -> dict:
    """Delete a project after the user confirms."""
    answer = await elicit.ask(
        f"Delete project {project_id}?",
        schema={"type": "object", "properties": {"confirm": {"type": "boolean"}}, "required": ["confirm"]},
    )
    if answer is None:
        # The client can't elicit, or the user declined: don't act
        return {"error": "This action requires confirmation. Please confirm and retry."}
    if not answer.get("confirm"):
        return {"status": "cancelled"}
    return {"status": "deleted", "project_id": project_id}
```

`elicit.last_failure` says why the last `ask()` returned `None` (for example `"the client did not declare the elicitation capability"` or `"the client answered 'decline'"`), so a tool can tell the user.

`ask(message, schema=None, *, timeout=None)`; without a schema the request is a bare confirmation (an empty object form). Over the initialize handshake, elicitation needs a session that can send requests to the client, so it doesn't work for handshake clients of a server served with `asgi_app(stateless=True)`. 2026-07-28 clients are asked in input rounds instead: see [Elicitation on the 2026-07-28 protocol](#elicitation-on-the-2026-07-28-protocol).

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `timeout` | `60.0` | Seconds to wait for user response (handshake connections only) |

### Elicitation on the 2026-07-28 protocol

How `ask()` reaches the client depends on the protocol version the client connected with (on the mcp 2.x SDK; with the 1.x SDK every client uses the handshake):

| Connection | What `ask()` does | Per-session state (`SessionState`) |
|---|---|---|
| `initialize` handshake (1.x clients, Promptise's `MCPClient`, 2.x clients with `mode="legacy"`) | Sends `elicitation/create` and waits up to `timeout` for the answer | Kept for the connection |
| 2026-07-28 (2.x clients by default) | Ends the call with an `InputRequiredResult` holding the question; the client asks its user and calls the tool again with the answer, and the tool runs again from the start, where `ask()` now returns the answer | None: the tool answers `SESSION_REQUIRED` |
| A stateless server (`asgi_app(stateless=True)`) with a handshake client | No elicitation (no way back to the client): `ask()` returns `None` | None: the tool answers `SESSION_REQUIRED` |

On 2026-07-28, **the tool runs once per question**, so code before `ask()` runs again on every round. Put side effects after the answer:

```python
@server.tool()
async def delete_project(project_id: str, elicit: Elicitor = Depends(Elicitor)) -> str:
    """Delete a project after the user confirms."""
    project = await load_project(project_id)          # read-only: fine to repeat
    answer = await elicit.ask(f"Delete {project.name}?", schema=CONFIRM)
    if not answer or not answer.get("confirm"):
        return f"Not deleted: {elicit.last_failure or 'not confirmed'}"
    await really_delete(project_id)                   # runs once, after the answer
    return "Deleted"
```

A tool that asks several questions takes one round per question. The answers to earlier questions travel in the result's `request_state`, which the server encrypts and binds to the tool, its arguments and the server's name (the SDK's `RequestStateBoundary`), so a client can only send back state this server minted for that call. By default the key is made at startup and held by the process: a round answered by another worker, or after a restart, is refused with `Invalid or expired requestState`. With several workers or replicas, share a key:

```python
from mcp.server.request_state import RequestStateSecurity

server = MCPServer(
    name="deploy-api",
    request_state_security=RequestStateSecurity(keys=[os.environ["MCP_STATE_KEY"]]),  # 32+ bytes
)
```

A client that did not declare the elicitation capability gets no question: `ask()` returns `None` and `last_failure` says why. The [approval gate](../../core/approval.md)'s `ElicitationApprover` works the same way: the client approves in an input round, and a client that can't answer is denied with that reason. `Sampler` is not available on 2026-07-28 (the spec deprecates sampling) and returns `None` with a warning.

---

## Sampling

### When you need it

Your tool needs to call an LLM to process data (summarize text, classify content, extract entities) but the **client** controls which LLM to use. MCP sampling lets the server request completions through the client's model.

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
    max_tokens=1024,             # Maximum tokens to generate
    model="claude-sonnet-5-5",   # Model hint (client may ignore)
    system="You are...",         # System prompt
    temperature=0.7,             # Sampling temperature
    stop_sequences=["\n\n"],     # Stop sequences
)
```

The client controls which model actually runs. The `model` parameter is sent as a model preference hint; the client can override it based on cost, availability, or policy. Message roles other than `"assistant"` are sent as `"user"` (MCP sampling has no system role; use `system=`). `create_message()` returns the generated text, or `None` when the client doesn't support sampling or the request fails (logged at `WARNING`). Like elicitation, sampling needs a session that can send requests to the client.

---

## Session State

### When you need it

A client makes several calls in one conversation and a tool needs to remember something between them: a cart, a selected project, a cursor.

### `SessionState`

```python
from promptise.mcp.server import MCPServer, SessionState, Depends

server = MCPServer(name="cart-api")

@server.tool()
async def add_item(item: str, session: SessionState = Depends(SessionState)) -> dict:
    """Add an item to this session's cart."""
    items = session.get("items", [])
    items.append(item)
    session.set("items", items)
    return {"count": len(items)}

@server.tool()
async def cart(session: SessionState = Depends(SessionState)) -> list[str]:
    """List the items in this session's cart."""
    return session.get("items", [])
```

`SessionState` is a small key-value store: `get(key, default=None)`, `set(key, value)`, `delete(key)`, `clear()`, `keys()`, `to_dict()` and `key in session`. The server keeps one store per MCP connection (in a `SessionManager`), in memory.

!!! warning "Limits in 1.6.0"
    - State lives in the server process: it is lost on restart and not shared between replicas or workers.
    - Stores are not removed when a session ends, so a long-running HTTP server keeps one per connection it has served. Don't put large values in it.
    - The store is keyed by the connection object's Python `id()`. After a connection is closed and garbage-collected, a later connection can receive the same id and see the old store. Don't keep secrets or per-user data that must not leak between sessions in `SessionState`; key such data by `ctx.client.client_id` in your own store instead.
    - Without a session (`asgi_app(stateless=True)`, or `TestClient`) each call gets a fresh store.

---

## Server Manifest

### When you need it

You want agents to introspect your server (discover all tools, their schemas, tags, auth requirements, and rate limits) without calling each tool individually.

### `register_manifest()`

```python
from promptise.mcp.server import MCPServer, register_manifest

server = MCPServer(name="my-api", version="2.0.0", auto_manifest=False)

@server.tool(tags=["math"], roles=["user"])
async def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

@server.tool(tags=["math"], auth=True)
async def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b

register_manifest(server)

# MCPServer(auto_manifest=True), the default, registers docs://manifest when the
# server is built. Call register_manifest() yourself only with auto_manifest=False.
```

Agents read the manifest via MCP's resource protocol:

```json
{
  "server": {"name": "my-api", "version": "2.0.0", "instructions": null},
  "tools": [
    {
      "name": "add",
      "description": "Add two numbers.",
      "input_schema": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
      "tags": ["math"],
      "auth_required": true,
      "roles": ["user"],
      "guards": ["HasRole"]
    },
    {
      "name": "divide",
      "description": "Divide two numbers.",
      "input_schema": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
      "tags": ["math"],
      "auth_required": true
    }
  ],
  "resources": [],
  "resource_templates": [],
  "prompts": []
}
```

Each entry lists only the fields that apply: `tags`, `auth_required`, `roles`, `guards` (class names), `rate_limit`, `timeout`. With `MCPServer(hide_unauthorized_tools=True)` the manifest is filtered per caller like the lists.

### Programmatic access

```python
from promptise.mcp.server import build_manifest

manifest = build_manifest(server)
print(manifest["tools"])  # All registered tools with metadata
```

`build_manifest(server, tools=None, *, resources=None, resource_templates=None, prompts=None)`: pass any of the lists to describe a subset instead of every registration.

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
| `allowed_hosts` | `None` | `Host` allow-list for the child (see [Deployment](deployment.md)) |
| `allowed_origins` | `None` | `Origin` allow-list for the child |

`hot_reload()` restarts **the script you ran** (`sys.argv`) in a child process, so call it from the `if __name__ == "__main__":` block of the file that defines the server. A child that exits with an error waits for the next file change before restarting. Changes are detected by modification time of `*.py` files under `watch_dirs`.

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
| `--dashboard` | off | Enable live dashboard (HTTP/SSE only; ignored with a warning on stdio) |
| `--reload` | off | Enable hot reload |
| `--allowed-host HOST` | none | `Host` header value to accept (repeatable); see [Deployment](deployment.md) |
| `--allowed-origin ORIGIN` | none | `Origin` header value to accept for browser clients (repeatable; needs `--allowed-host` on a non-loopback bind) |

The current directory is put on `sys.path`, so `promptise serve server:server` finds `./server.py`. A target that is not `module:attribute`, can't be imported, or is not an `MCPServer` exits with code 1 and a message on stderr (stdout stays clean for the stdio protocol).

---

## API Summary

| Symbol | Type | Description |
|--------|------|-------------|
| `@server.tool(version=...)` | Decorator option | Several versions of a tool: pinned names `<name>_v<version>`, the bare name runs the latest |
| `server.add_transform(transform)` | Method | Apply a transform to `tools/list` and the manifest |
| `VersionedToolRegistry` | Class | Versions of tools under one base name (`get("name@version")`) |
| `ToolTransform` | Protocol | `apply(tools, ctx=None) -> list[ToolDef]`, optional `resolve_name(name)` |
| `NamespaceTransform(prefix)` | Class | Prefix tool names in a `ToolDef` list |
| `VisibilityTransform(hidden)` | Class | Drop tools whose predicate returns true |
| `TagFilterTransform(required_tags)` | Class | Keep tools with at least one of the tags |
| `ToolDef` | Dataclass | A registered tool: `name`, `description`, `handler`, `input_schema`, `tags`, `auth`, `rate_limit`, `timeout`, `guards`, `roles`, `router_middleware`, `annotations`, `max_concurrent`, `requires_approval`, `cache` |
| `mount(parent, child, prefix, tags)` | Function | Compose servers |
| `OpenAPIProvider(spec, ...)` | Class | One tool per OpenAPI operation, with the spec's input schema |
| `register_batch_tool(server, ...)` | Function | Parallel batch execution |
| `StreamingResult` | Class | Incremental result collection |
| `Elicitor` | Class | Ask user for input mid-execution (via DI) |
| `Sampler` | Class | Request LLM completions from client (via DI) |
| `SessionState` | Class | Per-session key-value store (via DI) |
| `SessionManager` | Class | Holds the `SessionState` stores of one server (created by the server) |
| `build_manifest(server)` | Function | Build server manifest dict |
| `register_manifest(server)` | Function | Register `docs://manifest` resource |
| `hot_reload(server, ...)` | Function | File-watching server restarter |
| `build_serve_parser(...)` | Function | CLI argument parser |
| `resolve_server(target)` | Function | Import server from `module:attr` |
| `run_serve(args, server=None)` | Function | Execute CLI serve command |

## What's Next

- [Deployment](deployment.md): HTTP transport, CORS, containers, reverse proxy
- [Caching & Performance](caching-performance.md): Cache, rate limit, concurrency control
- [Observability & Monitoring](observability.md): Metrics, tracing, Prometheus, logging
- [Resilience Patterns](resilience-patterns.md): Circuit breakers, health checks, webhooks
