"""Convert MCP tools to LangChain BaseTool instances.

Uses the recursive :func:`~promptise.tools._jsonschema_to_pydantic` to
build fully-typed Pydantic models from MCP JSON Schemas — including
nested objects, arrays-of-objects, ``$ref``/``$defs``, and unions.

Uses the Promptise MCP Client for tool discovery and invocation.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Collection
from typing import Any

from langchain_core.tools import BaseTool
from mcp.types import CallToolResult
from pydantic import BaseModel, PrivateAttr

from ...tools import ToolInfo, _jsonschema_to_pydantic
from ._client import MCPClientError
from ._multi import MCPMultiClient

# Callback types
OnBefore = Callable[[str, dict[str, Any]], None]
OnAfter = Callable[[str, Any], None]
OnError = Callable[[str, Exception], None]


def _extract_text(result: CallToolResult) -> str:
    """Extract text content from a ``CallToolResult``.

    The MCP SDK returns ``CallToolResult`` with a ``.content`` list of
    ``TextContent`` / ``ImageContent`` / ``EmbeddedResource`` objects.
    LangChain's ``BaseTool`` expects a plain string return value.

    Concatenates all text parts with newlines, returning a single string.
    If the result contains an error (``isError=True``), the text is still
    returned so the LLM can see the error message.
    """
    if not hasattr(result, "content") or not result.content:
        return ""
    parts: list[str] = []
    for item in result.content:
        if hasattr(item, "text"):
            parts.append(item.text)
    return "\n".join(parts)


class _PromptiseMCPTool(BaseTool):
    """LangChain ``BaseTool`` that invokes an MCP tool via the Promptise client.

    Uses a persistent ``MCPMultiClient`` that stays connected for the
    agent's lifetime.  When ``forward_caller_token`` is set and the current
    invocation carries a ``CallerContext`` with a ``bearer_token``, the call
    is made with that token (see :meth:`MCPMultiClient.call_tool`).
    """

    name: str
    description: str
    args_schema: type[BaseModel]

    _tool_name: str = PrivateAttr()
    _multi: MCPMultiClient = PrivateAttr()
    _on_before: OnBefore | None = PrivateAttr(default=None)
    _on_after: OnAfter | None = PrivateAttr(default=None)
    _on_error: OnError | None = PrivateAttr(default=None)
    _forward_caller_token: bool = PrivateAttr(default=False)

    def __init__(
        self,
        *,
        name: str,
        description: str,
        args_schema: type[BaseModel],
        tool_name: str,
        multi: MCPMultiClient,
        on_before: OnBefore | None = None,
        on_after: OnAfter | None = None,
        on_error: OnError | None = None,
        forward_caller_token: bool = False,
    ) -> None:
        super().__init__(name=name, description=description, args_schema=args_schema)
        self._tool_name = tool_name
        self._multi = multi
        self._on_before = on_before
        self._on_after = on_after
        self._on_error = on_error
        self._forward_caller_token = forward_caller_token

    def _caller_token(self) -> str | None:
        """The current caller's bearer token, when this tool forwards it."""
        if not self._forward_caller_token:
            return None
        from ...agent import get_current_caller

        caller = get_current_caller()
        return caller.bearer_token if caller is not None and caller.bearer_token else None

    async def _arun(self, **kwargs: Any) -> Any:
        """Execute the MCP tool via the persistent multi-client."""
        if self._on_before:
            with contextlib.suppress(Exception):
                self._on_before(self.name, kwargs)

        try:
            result = await self._multi.call_tool(
                self._tool_name, kwargs, bearer_token=self._caller_token()
            )
        except Exception as exc:
            if self._on_error:
                with contextlib.suppress(Exception):
                    self._on_error(self.name, exc)
            raise MCPClientError(f"Failed to call MCP tool '{self._tool_name}': {exc}") from exc

        if self._on_after:
            with contextlib.suppress(Exception):
                self._on_after(self.name, result)

        # Extract text from CallToolResult for LangChain compatibility.
        # The MCP SDK returns a CallToolResult with a .content list of
        # TextContent / ImageContent / EmbeddedResource objects.
        # LangChain expects a plain string or serializable object.
        return _extract_text(result)

    def _run(self, **kwargs: Any) -> Any:  # pragma: no cover
        import anyio

        return anyio.run(lambda: self._arun(**kwargs))


class MCPToolAdapter:
    """Discover MCP tools and convert them to LangChain ``BaseTool`` instances.

    Backed by the Promptise MCP Client.

    Args:
        multi: Connected ``MCPMultiClient``.
        on_before: Callback fired before each tool invocation.
        on_after: Callback fired after each tool invocation.
        on_error: Callback fired on tool errors.
        optimize: Tool optimization config (schema minification etc.).
        forward_caller_token: Which servers' tools send the invoking
            ``CallerContext.bearer_token`` instead of the client's own
            credentials: ``True`` for every server, ``False`` for none
            (default), or a collection of server names.  stdio servers
            cannot receive it (see :meth:`MCPMultiClient.call_tool`).

    Example::

        multi = MCPMultiClient({"hr": MCPClient(...)})
        async with multi:
            adapter = MCPToolAdapter(multi)
            tools = await adapter.as_langchain_tools()
            # Pass `tools` to build_agent(extra_tools=tools)
    """

    def __init__(
        self,
        multi: MCPMultiClient,
        *,
        on_before: OnBefore | None = None,
        on_after: OnAfter | None = None,
        on_error: OnError | None = None,
        optimize: Any | None = None,
        forward_caller_token: bool | Collection[str] = False,
    ) -> None:
        self._multi = multi
        self._forward_caller_token = forward_caller_token
        self._on_before = on_before
        self._on_after = on_after
        self._on_error = on_error
        self._optimize = optimize

    async def as_langchain_tools(self) -> list[BaseTool]:
        """Discover tools and return them as LangChain ``BaseTool`` instances.

        Each tool's ``args_schema`` is a recursively-built Pydantic model
        that preserves nested object structure, descriptions, defaults,
        and constraints from the MCP server's JSON Schema.

        When ``optimize`` is set, static optimizations (schema
        minification, description truncation) are applied to reduce
        token cost.

        Returns:
            List of ``BaseTool`` instances ready for LangGraph.
        """
        # Resolve optimization config if provided
        resolved = None
        strip_desc = False
        if self._optimize is not None:
            from ...tool_optimization import _resolve_config

            resolved = _resolve_config(self._optimize)
            strip_desc = resolved.minify_schema

        mcp_tools = await self._multi.list_tools()
        tool_to_server = self._multi.tool_to_server
        forward_to = self._forward_caller_token
        forward_servers: set[str] | None
        if isinstance(forward_to, bool):
            forward_servers = None
        elif isinstance(forward_to, str):  # one server name, not its characters
            forward_servers = {forward_to}
        else:
            forward_servers = set(forward_to)

        out: list[BaseTool] = []
        for t in mcp_tools:
            name = t.name
            if forward_servers is None:
                forward = bool(forward_to)
            else:
                forward = tool_to_server.get(name) in forward_servers
            desc = t.description or ""
            schema = t.inputSchema or {}
            model = _jsonschema_to_pydantic(
                schema,
                model_name=f"Args_{name}",
                strip_descriptions=strip_desc,
            )
            out.append(
                _PromptiseMCPTool(
                    name=name,
                    description=desc,
                    args_schema=model,
                    tool_name=name,
                    multi=self._multi,
                    on_before=self._on_before,
                    on_after=self._on_after,
                    on_error=self._on_error,
                    forward_caller_token=forward,
                )
            )

        # Apply static optimizations (truncation, further minification)
        if resolved is not None:
            from ...tool_optimization import apply_static_optimizations

            out = apply_static_optimizations(out, resolved)

        return out

    async def list_tool_info(self) -> list[ToolInfo]:
        """Return human-readable tool metadata for introspection."""
        tools = await self._multi.list_tools()
        return [
            ToolInfo(
                server_guess=self._multi.tool_to_server.get(t.name, ""),
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema or {},
            )
            for t in tools
        ]
