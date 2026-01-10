"""MCP tool discovery and conversion to LangChain tools."""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import anyio
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr, create_model

from .clients import FastMCPMulti

# Callback types for tracing tool calls
OnBefore = Callable[[str, dict[str, Any]], None]
OnAfter = Callable[[str, Any], None]
OnError = Callable[[str, Exception], None]


@dataclass(frozen=True)
class ToolInfo:
    """Human-friendly metadata for a discovered MCP tool."""

    server_guess: str
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPClientError(RuntimeError):
    """Raised when communicating with the MCP client fails."""


# Conservative limits to avoid untrusted schemas causing resource exhaustion.
_MAX_SCHEMA_CHARS = 20_000
_MAX_SCHEMA_PROPERTIES = 50
_MAX_SCHEMA_REQUIRED = 30
_MAX_SCHEMA_DEPTH = 6

# Network resilience defaults.
_DEFAULT_TIMEOUT_S = 10.0
_DEFAULT_RETRIES = 2
_RETRY_BACKOFF_BASE_S = 0.5


def _schema_depth(obj: Any, depth: int = 0) -> int:
    if depth > _MAX_SCHEMA_DEPTH:
        return depth
    if isinstance(obj, dict):
        return max((_schema_depth(v, depth + 1) for v in obj.values()), default=depth)
    if isinstance(obj, list):
        return max((_schema_depth(v, depth + 1) for v in obj), default=depth)
    return depth


def _validate_schema(schema: dict[str, Any]) -> None:
    if not schema:
        return
    # Size guard.
    approx = len(json.dumps(schema, default=str))
    if approx > _MAX_SCHEMA_CHARS:
        raise MCPClientError(
            f"Schema too large ({approx} chars). Limit is {_MAX_SCHEMA_CHARS}."
        )
    props = (schema or {}).get("properties", {}) or {}
    if len(props) > _MAX_SCHEMA_PROPERTIES:
        raise MCPClientError(
            f"Schema has too many properties ({len(props)}). Limit is {_MAX_SCHEMA_PROPERTIES}."
        )
    required = (schema or {}).get("required", []) or []
    if len(required) > _MAX_SCHEMA_REQUIRED:
        raise MCPClientError(
            f"Schema has too many required fields ({len(required)}). Limit is {_MAX_SCHEMA_REQUIRED}."
        )
    depth = _schema_depth(schema)
    if depth > _MAX_SCHEMA_DEPTH:
        raise MCPClientError(
            f"Schema nesting too deep ({depth}). Limit is {_MAX_SCHEMA_DEPTH}."
        )


def _jsonschema_to_pydantic(schema: dict[str, Any], *, model_name: str = "Args") -> type[BaseModel]:
    _validate_schema(schema)
    props = (schema or {}).get("properties", {}) or {}
    required = set((schema or {}).get("required", []) or [])

    # Each value is (annotation, default)
    def f(n: str, p: dict[str, Any]) -> tuple[type[Any], Any]:
        t = p.get("type")
        desc = p.get("description")
        default = p.get("default")
        req = n in required
        enum = p.get("enum")

        def default_val() -> Any:
            return ... if req else default

        if enum and isinstance(enum, list) and enum:
            # Constrain to the enum's base type when possible.
            first = enum[0]
            base_type = type(first) if all(isinstance(e, type(first)) for e in enum) else Any
            return (base_type, Field(default_val(), description=desc, examples=list(enum)))
        if t == "string":
            return (str, Field(default_val(), description=desc))
        if t == "integer":
            return (int, Field(default_val(), description=desc))
        if t == "number":
            return (float, Field(default_val(), description=desc))
        if t == "boolean":
            return (bool, Field(default_val(), description=desc))
        if t == "array":
            return (list, Field(default_val(), description=desc))
        if t == "object":
            return (dict, Field(default_val(), description=desc))
        return (Any, Field(default_val(), description=desc))

    fields: dict[str, tuple[type[Any], Any]] = {
        n: f(n, spec or {}) for n, spec in props.items()
    } or {"payload": (dict, Field(None, description="Raw payload"))}

    safe_name = re.sub(r"[^0-9a-zA-Z_]", "_", model_name) or "Args"

    # Hand the kwargs to pydantic as Any to satisfy the stubbed overloads
    model = create_model(safe_name, **cast(dict[str, Any], fields))
    return cast(type[BaseModel], model)


class _FastMCPTool(BaseTool):
    """LangChain `BaseTool` wrapper that invokes a FastMCP tool by name."""

    name: str
    description: str
    args_schema: type[BaseModel]

    _tool_name: str = PrivateAttr()
    _client: Any = PrivateAttr()
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
        client: Any,
        on_before: OnBefore | None = None,
        on_after: OnAfter | None = None,
        on_error: OnError | None = None,
    ) -> None:
        super().__init__(name=name, description=description, args_schema=args_schema)
        self._tool_name = tool_name
        self._client = client
        self._on_before = on_before
        self._on_after = on_after
        self._on_error = on_error

    async def _arun(self, **kwargs: Any) -> Any:
        """Asynchronously execute the MCP tool via the FastMCP client.

        Handles transport errors gracefully with timeout and retry logic.
        """
        if self._on_before:
            with contextlib.suppress(Exception):
                self._on_before(self.name, kwargs)

        res: Any | None = None
        last_exc: Exception | None = None
        for attempt in range(_DEFAULT_RETRIES + 1):
            try:
                async with anyio.fail_after(_DEFAULT_TIMEOUT_S):
                    async with self._client:
                        res = await self._client.call_tool(self._tool_name, kwargs)
                break
            except Exception as exc:
                last_exc = exc
                if attempt < _DEFAULT_RETRIES:
                    await anyio.sleep(_RETRY_BACKOFF_BASE_S * (2**attempt))
                    continue
                if self._on_error:
                    with contextlib.suppress(Exception):
                        self._on_error(self.name, exc)
                raise MCPClientError(
                    f"Failed to call MCP tool '{self._tool_name}': {exc}"
                ) from exc

        if res is None:
            err = MCPClientError(f"Tool '{self._tool_name}' returned no result")
            if self._on_error:
                with contextlib.suppress(Exception):
                    self._on_error(self.name, err)
            raise err

        if self._on_after:
            with contextlib.suppress(Exception):
                self._on_after(self.name, res)

        return res

    def _run(self, **kwargs: Any) -> Any:  # pragma: no cover
        """Synchronous execution path (rarely used)."""
        import anyio

        return anyio.run(lambda: self._arun(**kwargs))


class MCPToolLoader:
    """Discover MCP tools via FastMCP and convert them to LangChain tools."""

    def __init__(
        self,
        multi: FastMCPMulti,
        *,
        on_before: OnBefore | None = None,
        on_after: OnAfter | None = None,
        on_error: OnError | None = None,
    ) -> None:
        self._multi = multi
        self._on_before = on_before
        self._on_after = on_after
        self._on_error = on_error

    async def _list_tools_raw(self) -> tuple[Any, list[Any]]:
        """Fetch raw tool descriptors from all configured MCP servers with timeout and retry."""
        c = self._multi.client
        tools: list[Any] = []
        for attempt in range(_DEFAULT_RETRIES + 1):
            try:
                async with anyio.fail_after(_DEFAULT_TIMEOUT_S):
                    async with c:
                        tools = await c.list_tools()
                break
            except Exception as exc:
                if attempt < _DEFAULT_RETRIES:
                    await anyio.sleep(_RETRY_BACKOFF_BASE_S * (2**attempt))
                    continue
                raise MCPClientError(
                    f"Failed to list tools from MCP servers: {exc}. "
                    "Check server URLs, network connectivity, and authentication headers."
                ) from exc
        return c, list(tools or [])

    async def get_all_tools(self) -> list[BaseTool]:
        """Return all available tools as LangChain `BaseTool` instances."""
        client, tools = await self._list_tools_raw()

        out: list[BaseTool] = []
        for t in tools:
            name = t.name
            desc = getattr(t, "description", "") or ""
            schema = getattr(t, "inputSchema", None) or {}
            model = _jsonschema_to_pydantic(schema, model_name=f"Args_{name}")
            out.append(
                _FastMCPTool(
                    name=name,
                    description=desc,
                    args_schema=model,
                    tool_name=name,
                    client=client,
                    on_before=self._on_before,
                    on_after=self._on_after,
                    on_error=self._on_error,
                )
            )
        return out

    async def list_tool_info(self) -> list[ToolInfo]:
        """Return human-readable tool metadata for introspection or debugging."""
        _, tools = await self._list_tools_raw()
        return [
            ToolInfo(
                server_guess=(getattr(t, "server", None) or getattr(t, "serverName", None) or ""),
                name=t.name,
                description=getattr(t, "description", "") or "",
                input_schema=getattr(t, "inputSchema", None) or {},
            )
            for t in tools
        ]
