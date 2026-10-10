# MCP Tool Adapter

The `MCPToolAdapter` discovers tools from connected MCP servers and converts
them into fully-typed LangChain `BaseTool` instances.  It builds recursive
Pydantic models from MCP JSON Schemas -- preserving nested objects, arrays,
unions, `$ref`/`$defs`, enums, defaults, and descriptions -- so LLMs can
generate correct tool calls on the first attempt.

**Source:** `src/promptise/mcp/client/_tool_adapter.py`

## Quick example

```python
import asyncio
from promptise import MCPClient, MCPMultiClient, MCPToolAdapter, build_agent

async def main():
    multi = MCPMultiClient({
        "math": MCPClient(url="http://localhost:8080/mcp"),
    })

    async with multi:
        adapter = MCPToolAdapter(multi)
        tools = await adapter.as_langchain_tools()

        # Use tools with build_agent
        agent = await build_agent(
            model="openai:gpt-5-mini",
            servers={},
            extra_tools=tools,
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "What is 7 + 3?"}]}
        )
        print(result["messages"][-1].content)

asyncio.run(main())
```

## Concepts

### How MCPToolAdapter fits in

`MCPToolAdapter` is the recommended way to convert MCP tools into LangChain
`BaseTool` instances.  It is backed by `MCPMultiClient`, uses persistent
connections for the entire agent lifetime, supports Bearer token and API key
authentication, and provides recursive schema conversion.

### How it works

1. **Discovery** -- `as_langchain_tools()` calls `MCPMultiClient.list_tools()`
   to fetch tool definitions from all connected servers.
2. **Schema conversion** -- Each tool's `inputSchema` (a JSON Schema dict) is
   recursively converted to a Pydantic `BaseModel` using
   `_jsonschema_to_pydantic`.
3. **Tool wrapping** -- Each tool is wrapped in a `_PromptiseMCPTool` (a
   `BaseTool` subclass) that calls `MCPMultiClient.call_tool()` on invocation.
4. **Result extraction** -- The MCP `CallToolResult` is converted to a plain
   string by concatenating all text content parts.

### Creating an adapter

```python
from promptise import MCPClient, MCPMultiClient, MCPToolAdapter

multi = MCPMultiClient({
    "hr": MCPClient(url="http://hr-server:8080/mcp", bearer_token="..."),
    "docs": MCPClient(url="http://docs-server:9090/mcp", api_key="secret"),
})

async with multi:
    adapter = MCPToolAdapter(multi)
    tools = await adapter.as_langchain_tools()
```

### Getting LangChain tools

`as_langchain_tools()` returns a list of `BaseTool` instances ready for any
LangChain or LangGraph agent:

```python
async with multi:
    adapter = MCPToolAdapter(multi)
    tools = await adapter.as_langchain_tools()

    for tool in tools:
        print(f"{tool.name}: {tool.description}")
        print(f"  Schema: {tool.args_schema.model_json_schema()}")
```

Each tool's `args_schema` is a dynamically-created Pydantic model that
preserves the full structure of the MCP server's JSON Schema, including:

- Nested objects become nested Pydantic models
- Arrays of objects become `list[NestedModel]`
- `$ref` / `$defs` references are resolved and inlined
- `anyOf` / `oneOf` become `Union[...]` or `Optional[...]`
- `allOf` properties are merged
- Enums become `Literal["a", "b", "c"]`
- Descriptions and defaults are preserved as Pydantic `Field` metadata

### Tool introspection

Use `list_tool_info()` to get lightweight metadata about discovered tools
without creating `BaseTool` wrappers:

```python
async with multi:
    adapter = MCPToolAdapter(multi)
    infos = await adapter.list_tool_info()

    for info in infos:
        print(f"{info.server_guess}/{info.name}")
        print(f"  {info.description}")
        print(f"  Schema keys: {list(info.input_schema.get('properties', {}).keys())}")
```

This returns a list of `ToolInfo` dataclass instances -- useful for debugging,
logging, or building tool registries.

### Callback hooks

Attach optional callbacks for tracing every tool invocation:

```python
def on_before(tool_name: str, arguments: dict) -> None:
    print(f"[TRACE] Calling {tool_name}")

def on_after(tool_name: str, result) -> None:
    print(f"[TRACE] {tool_name} completed")

def on_error(tool_name: str, exc: Exception) -> None:
    print(f"[ERROR] {tool_name} failed: {exc}")

adapter = MCPToolAdapter(
    multi,
    on_before=on_before,
    on_after=on_after,
    on_error=on_error,
)
tools = await adapter.as_langchain_tools()
```

Callbacks are wrapped in `contextlib.suppress(Exception)` so a failing callback
never breaks the actual tool call.

### Result extraction

MCP servers return `CallToolResult` objects containing a list of content items
(text, images, embedded resources).  The adapter extracts all text parts and
joins them with newlines:

```python
# What the MCP server returns:
# CallToolResult(content=[
#     TextContent(type="text", text="Found 3 results"),
#     TextContent(type="text", text="Result 1: ..."),
# ])

# What the LangChain tool returns:
# "Found 3 results\nResult 1: ..."
```

If the result has `isError=True` (the server ran the tool and it failed, e.g.
a `ToolError`), the tool raises a LangChain `ToolException` whose message is
that text — see below.

### Error handling

When a tool call fails at the transport or protocol level, the adapter raises
`MCPClientError`. When the server reports a failed call (`isError=True`), it
raises `ToolException` with the server's error text, so LangChain callbacks get
`on_tool_error` (observability records `tool.error`) and the `on_error`
adapter callback runs instead of `on_after`:

```python
from langchain_core.tools import ToolException
from promptise.mcp.client import MCPClientError

try:
    result = await tool.ainvoke({"query": "test"})
except ToolException as exc:
    print(f"Tool reported an error: {exc}")   # e.g. {"error": {"code": "TOOL_ERROR", ...}}
except MCPClientError as exc:
    print(f"Tool call failed: {exc}")
```

Inside a Promptise agent you do not need to catch either: the agent loop marks
the call as failed and passes the error text to the model (`Error:
ToolException: {...}`) so it can correct itself. When you use the tools in
another framework, set `tool.handle_tool_error = True` to have LangChain
return the error text to the model instead of raising.

## API summary

### MCPToolAdapter

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `multi` | `MCPMultiClient` | *required* | Connected multi-server client |
| `on_before` | `OnBefore \| None` | `None` | Callback fired before each tool invocation |
| `on_after` | `OnAfter \| None` | `None` | Callback fired after each tool invocation |
| `on_error` | `OnError \| None` | `None` | Callback fired on tool errors |

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `as_langchain_tools()` | `list[BaseTool]` | Discover tools and return as LangChain `BaseTool` instances |
| `list_tool_info()` | `list[ToolInfo]` | Return tool metadata for introspection |

### Callback types

| Type | Signature |
|------|-----------|
| `OnBefore` | `Callable[[str, dict[str, Any]], None]` |
| `OnAfter` | `Callable[[str, Any], None]` |
| `OnError` | `Callable[[str, Exception], None]` |

### ToolInfo

| Attribute | Type | Description |
|-----------|------|-------------|
| `server_guess` | `str` | Server name that owns the tool |
| `name` | `str` | Tool name |
| `description` | `str` | Human-readable description |
| `input_schema` | `dict[str, Any]` | Raw JSON Schema for tool input |

## Tips and gotchas

!!! tip
    `MCPToolAdapter` uses persistent connections via `MCPMultiClient`.  Create
    the adapter once and reuse the tools for the entire agent session -- do not
    recreate the adapter per invocation.

!!! warning
    You must call `as_langchain_tools()` (or `list_tool_info()`) **inside**
    the `async with multi:` block.  The underlying `MCPMultiClient` must be
    connected for tool discovery to work.

!!! tip
    The recursive schema converter handles complex MCP schemas that would
    otherwise appear as opaque `dict` parameters to the LLM.  If your tool
    call accuracy is low, inspect the generated `args_schema` to verify the
    schema conversion is correct:

    ```python
    for tool in tools:
        print(tool.name, tool.args_schema.model_json_schema())
    ```

!!! warning
    Callback hooks (`on_before`, `on_after`, `on_error`) are silenced on
    failure -- they never propagate exceptions.  If your callback raises,
    the error is swallowed and the tool call proceeds normally.  Use proper
    error handling inside your callbacks.

!!! tip
    For simpler setups, `build_agent` handles tool adaptation internally.
    Use `MCPToolAdapter` directly when you need fine-grained control over
    tool discovery, callbacks, or when integrating with custom LangChain
    pipelines.

## What's next

- [MCP Client](index.md) -- the `MCPClient` and `MCPMultiClient` that power the adapter
- [Tools & Schema Helpers](../../core/tools.md) -- the schema conversion logic shared with the legacy loader
- [Building Agents](../../core/agents/building-agents.md) -- use adapted tools with `build_agent`
