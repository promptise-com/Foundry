"""Convert MCP tools to LangChain BaseTool instances.

Uses the recursive :func:`~promptise.tools._jsonschema_to_pydantic` to
build fully-typed Pydantic models from MCP JSON Schemas — including
nested objects, arrays-of-objects, ``$ref``/``$defs``, and unions.

Uses the Promptise MCP Client for tool discovery and invocation.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Any

from langchain_core.tools import BaseTool, ToolException
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
    agent's lifetime.

    A result the server flags with ``isError=True`` (a ``ToolError``, a
    failed validation…) is a tool error, reported the LangChain way: the
    tool raises ``ToolException`` and handles it (``handle_tool_error``),
    so the error text is still the output -- a plain ``ainvoke(args)``
    returns it as a string, unchanged -- and a tool-call invocation returns
    a ``ToolMessage`` with ``status="error"``.
    """

    name: str
    description: str
    args_schema: type[BaseModel]

    _tool_name: str = PrivateAttr()
    _multi: MCPMultiClient = PrivateAttr()
    _on_before: OnBefore | None = PrivateAttr(default=None)
    _on_after: OnAfter | None = PrivateAttr(default=None)
    _on_error: OnError | None = PrivateAttr(default=None)

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
    ) -> None:
        super().__init__(
            name=name,
            description=description,
            args_schema=args_schema,
            handle_tool_error=True,
        )
        self._tool_name = tool_name
        self._multi = multi
        self._on_before = on_before
        self._on_after = on_after
        self._on_error = on_error

    async def _arun(self, **kwargs: Any) -> Any:
        """Execute the MCP tool via the persistent multi-client."""
        if self._on_before:
            with contextlib.suppress(Exception):
                self._on_before(self.name, kwargs)

        try:
            result = await self._multi.call_tool(self._tool_name, kwargs)
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
        text = _extract_text(result)
        if getattr(result, "isError", False):
            # The text still reaches the model (handle_tool_error=True);
            # the exception marks the call as failed.
            raise ToolException(text or f"MCP tool '{self._tool_name}' reported an error")
        return text

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
    ) -> None:
        self._multi = multi
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

        out: list[BaseTool] = []
        for t in mcp_tools:
            name = t.name
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
