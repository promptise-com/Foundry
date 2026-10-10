"""TestClient for in-process MCP server testing without transport.

Replicates the full server pipeline (validation → DI → guards → middleware
→ handler) so tests exercise real behaviour including auth, guards, middleware,
and error handling — no network required.

Example::

    from promptise.mcp.server import MCPServer
    from promptise.mcp.server.testing import TestClient

    server = MCPServer(name="test")

    @server.tool()
    async def add(a: int, b: int) -> int:
        return a + b

    async def test_add():
        client = TestClient(server)
        result = await client.call_tool("add", {"a": 1, "b": 2})
        assert result[0].text == "3"
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.types import (
    GetPromptResult,
    PromptArgument,
    PromptMessage,
    Resource,
    ResourceTemplate,
    TextContent,
    Tool,
)
from mcp.types import (
    ToolAnnotations as MCPToolAnnotations,
)

from ._context import RequestContext, clear_context, set_context
from ._di import DependencyResolver
from ._errors import AuthenticationError, MCPError
from ._middleware import MiddlewareChain
from ._validation import validate_arguments

logger = logging.getLogger("promptise.server.testing")


async def check_guards(guards: list[Any], ctx: RequestContext) -> None:
    """Run all guards, raising ``AuthenticationError`` if any deny.

    Guards are checked in order.  The first failure short-circuits.
    The error message includes the guard's ``describe_denial()`` output
    when available, so developers can see *why* access was denied (e.g.
    which roles were required vs. which the client has).
    """
    for guard in guards:
        allowed = await guard.check(ctx)
        if not allowed:
            guard_name = type(guard).__name__
            # Use descriptive denial message if the guard provides one
            if hasattr(guard, "describe_denial"):
                detail = guard.describe_denial(ctx)
            else:
                detail = f"Access denied by {guard_name}"
            raise AuthenticationError(
                detail,
                code="ACCESS_DENIED",
                details={"guard": guard_name, "tool": ctx.tool_name},
            )


class TestClient:
    __test__ = False  # Prevent pytest collection

    """In-process test client for :class:`MCPServer`.

    Exercises the **full** call pipeline — validation, dependency injection,
    guard checks, middleware chain, handler invocation, and error serialisation
    — without starting a transport.

    Args:
        server (Any): The ``MCPServer`` instance to test.
        meta (dict[str, Any] | None): Simulated MCP request metadata (e.g.
            ``{"authorization": "Bearer xxx"}``).  Copied into every
            ``RequestContext.meta`` the client creates.

    Example::

        client = TestClient(server, meta={"authorization": "Bearer tok"})
        result = await client.call_tool("search", {"query": "revenue"})
    """

    def __init__(self, server: Any, *, meta: dict[str, Any] | None = None) -> None:
        self._server = server
        self._meta = meta or {}
        # Parity with the live server, which auto-inserts declared per-tool
        # rate-limit enforcement at build time. One persistent instance per
        # TestClient so token buckets accumulate across calls like production.
        from ._rate_limit import DeclaredRateLimitMiddleware

        self._declared_rate_limiter = DeclaredRateLimitMiddleware()

    # ------------------------------------------------------------------
    # Tool operations
    # ------------------------------------------------------------------

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
        *,
        headers: dict[str, str] | None = None,
    ) -> list[Any]:
        """Call a tool through the full middleware pipeline.

        Returns the same content list as the real MCP server (may include
        ``TextContent``, ``ImageContent``, or ``EmbeddedResource``).
        ``MCPError`` sub-classes are serialised into structured error
        JSON — they are **not** raised.

        Args:
            name: Registered tool name.
            arguments: Tool arguments (validated against the input model).
            headers: Simulated HTTP headers (e.g. ``{"x-api-key": "..."}``).
                Merged with client-level meta (headers take precedence).
        """
        # Tenant invariant parity with the live build path
        if getattr(self._server, "_require_tenant", False):
            self._server._apply_require_tenant()

        tdef = self._server._tool_registry.get(name)

        # Approval invariant parity: an ungated requires_approval tool is a
        # configuration error — the live server refuses to build, so raise
        # here too (outside the error-serialising pipeline: this must crash
        # the test, not masquerade as a tool error response).
        if tdef is not None and getattr(tdef, "requires_approval", False):
            from ._approval_gate import ApprovalGateMiddleware

            # Covered by a server-level OR a router-level gate (both compile
            # into the per-tool chain) — parity with the live build invariant.
            covered = any(
                isinstance(m, ApprovalGateMiddleware) for m in self._server._middlewares
            ) or any(
                isinstance(m, ApprovalGateMiddleware)
                for m in getattr(tdef, "router_middleware", [])
            )
            if not covered:
                raise RuntimeError(
                    f"Tool {name!r} declares requires_approval=True but no "
                    "ApprovalGateMiddleware is installed on the server."
                )

        if tdef is None:
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "error": {
                                "code": "TOOL_NOT_FOUND",
                                "message": f"Unknown tool: {name}",
                            }
                        }
                    ),
                )
            ]

        arguments = dict(arguments or {})

        # Merge HTTP request headers (from contextvar, as the transport
        # layer sets them) with explicit client meta.  Explicit meta takes
        # precedence, which is the expected behaviour: test code that
        # constructs ``TestClient(server, meta={...})`` wins over any
        # ambient contextvar.
        from ._context import get_request_headers

        http_headers = dict(get_request_headers())
        merged_meta = {**http_headers, **dict(self._meta), **(headers or {})}
        ctx = RequestContext(
            server_name=self._server.name,
            tool_name=name,
            meta=merged_meta,
        )
        ctx.state["tool_def"] = tdef
        set_context(ctx)

        di_resolver = DependencyResolver()
        try:
            # 1) Validate input
            model = self._server._input_models.get(name)
            if model is not None:
                arguments = validate_arguments(model, arguments)

            # Snapshot validated user args for middleware (approval gate),
            # before DI injects framework objects — parity with the live path
            ctx.state["_tool_arguments"] = dict(arguments)

            # 2) Resolve dependency injection
            arguments = await di_resolver.resolve(tdef.handler, arguments)

            # 2b) Auto-inject RequestContext-typed params
            arguments = _inject_context(tdef.handler, arguments, ctx)

            # 2c) Detect BackgroundTasks in resolved args → store in ctx
            from ._background import BackgroundTasks

            for val in arguments.values():
                if isinstance(val, BackgroundTasks):
                    ctx.state["_background_tasks"] = val
                    break

            # 3) Build middleware chain: server-level + router-level
            all_mw = list(self._server._middlewares)
            # Enforce a declared @server.tool(rate_limit=...) exactly like the
            # live server (auto-inserted, guard against a user-installed copy)
            if tdef.rate_limit:
                from ._rate_limit import DeclaredRateLimitMiddleware

                if not any(isinstance(m, DeclaredRateLimitMiddleware) for m in all_mw):
                    all_mw.append(self._declared_rate_limiter)
            if tdef.router_middleware:
                all_mw.extend(tdef.router_middleware)

            # 4) Wrap handler with guard checks (guards run after
            #    middleware so auth middleware can populate roles first)
            effective_handler = tdef.handler
            if tdef.guards:
                _guards = tdef.guards
                _ctx = ctx
                _real = tdef.handler

                async def _guarded(**kw: Any) -> Any:
                    await check_guards(_guards, _ctx)
                    r = _real(**kw)
                    if asyncio.iscoroutine(r):
                        r = await r
                    return r

                effective_handler = _guarded

            if all_mw:
                chain = MiddlewareChain(all_mw)
                result = await chain.run(ctx, effective_handler, arguments)
            else:
                result = await _invoke_handler(effective_handler, arguments)

            # 5) Serialise result
            serialised = _serialise_result(result)

            # 6) Run background tasks
            bg = ctx.state.get("_background_tasks")
            if bg is not None:
                await bg.execute()

            return serialised

        except MCPError as exc:
            return [TextContent(type="text", text=exc.to_text())]
        except Exception as exc:
            # Try custom exception handlers first
            if hasattr(self._server, "_exception_handlers"):
                mapped = await self._server._exception_handlers.handle(ctx, exc)
                if mapped is not None:
                    return [TextContent(type="text", text=mapped.to_text())]

            logger.exception("Unhandled error in tool '%s'", name)
            err_text = json.dumps(
                {
                    "error": {
                        "code": "INTERNAL_ERROR",
                        "message": str(exc),
                        "retryable": False,
                    }
                }
            )
            return [TextContent(type="text", text=err_text)]
        finally:
            await di_resolver.cleanup()
            clear_context()

    def _request_meta(self, headers: dict[str, str] | None = None) -> dict[str, Any]:
        """Transport headers, then client meta, then per-call headers."""
        from ._context import get_request_headers

        return {**dict(get_request_headers()), **dict(self._meta), **(headers or {})}

    async def list_tools(self, *, headers: dict[str, str] | None = None) -> list[Tool]:
        """List the registered tools (including annotations).

        With ``MCPServer(hide_unauthorized_tools=True)`` only the tools the
        client's credentials may call are listed, as on the live server.

        Args:
            headers: Simulated HTTP headers, merged over the client meta.
        """
        if getattr(self._server, "_require_tenant", False):
            self._server._apply_require_tenant()
        tdefs = self._server._tool_registry.list_all()
        if getattr(self._server, "_hide_unauthorized_tools", False):
            from ._visibility import visible_tools

            tdefs = await visible_tools(
                tdefs,
                self._server._middlewares,
                server_name=self._server.name,
                meta=self._request_meta(headers),
            )
        tools: list[Tool] = []
        for tdef in tdefs:
            mcp_annotations = None
            if tdef.annotations is not None:
                mcp_annotations = MCPToolAnnotations(
                    title=tdef.annotations.title,
                    readOnlyHint=tdef.annotations.read_only_hint,
                    destructiveHint=tdef.annotations.destructive_hint,
                    idempotentHint=tdef.annotations.idempotent_hint,
                    openWorldHint=tdef.annotations.open_world_hint,
                )
            tools.append(
                Tool(
                    name=tdef.name,
                    description=tdef.description,
                    inputSchema=tdef.input_schema,
                    annotations=mcp_annotations,
                )
            )
        return tools

    # ------------------------------------------------------------------
    # Resource operations
    # ------------------------------------------------------------------

    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI.

        Supports both static resources and URI templates.

        Args:
            uri: The resource URI (e.g. ``"config://app"``).

        Raises:
            ValueError: If the resource is not found.
        """
        res_reg = self._server._resource_registry

        # Try static resource first
        rdef = res_reg.get(uri)
        if rdef is not None:
            ctx = RequestContext(
                server_name=self._server.name, tool_name=rdef.name, meta=self._request_meta()
            )
            set_context(ctx)
            try:
                result = rdef.handler()
                if asyncio.iscoroutine(result):
                    result = await result
                return str(result)
            finally:
                clear_context()

        # Try template match
        match = res_reg.match_template(uri)
        if match is not None:
            tmpl_def, params = match
            ctx = RequestContext(
                server_name=self._server.name,
                tool_name=tmpl_def.name,
                meta=self._request_meta(),
            )
            set_context(ctx)
            try:
                result = tmpl_def.handler(**params)
                if asyncio.iscoroutine(result):
                    result = await result
                return str(result)
            finally:
                clear_context()

        raise ValueError(f"Resource not found: {uri}")

    async def list_resources(self) -> list[Resource]:
        """List all registered static resources."""
        return [
            Resource(
                uri=rdef.uri,
                name=rdef.name,
                description=rdef.description,
                mimeType=rdef.mime_type,
            )
            for rdef in self._server._resource_registry.list_all()
        ]

    async def list_resource_templates(self) -> list[ResourceTemplate]:
        """List all registered resource templates."""
        return [
            ResourceTemplate(
                uriTemplate=rdef.uri,
                name=rdef.name,
                description=rdef.description,
                mimeType=rdef.mime_type,
            )
            for rdef in self._server._resource_registry.list_templates()
        ]

    # ------------------------------------------------------------------
    # Prompt operations
    # ------------------------------------------------------------------

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> GetPromptResult:
        """Get a prompt result.

        Args:
            name: Registered prompt name.
            arguments: Prompt arguments (string-valued).

        Raises:
            ValueError: If the prompt is not found.
        """
        pdef = self._server._prompt_registry.get(name)
        if pdef is None:
            raise ValueError(f"Prompt not found: {name}")

        ctx = RequestContext(server_name=self._server.name, tool_name=name)
        set_context(ctx)
        try:
            result = pdef.handler(**(arguments or {}))
            if asyncio.iscoroutine(result):
                result = await result

            if isinstance(result, str):
                return GetPromptResult(
                    description=pdef.description,
                    messages=[
                        PromptMessage(
                            role="user",
                            content=TextContent(type="text", text=result),
                        )
                    ],
                )
            return result
        finally:
            clear_context()

    async def list_prompts(self) -> list[Any]:
        """List all registered prompts."""
        from mcp.types import Prompt as MCPPrompt

        return [
            MCPPrompt(
                name=pdef.name,
                description=pdef.description,
                arguments=[
                    PromptArgument(
                        name=a["name"],
                        description=a.get("description"),
                        required=a.get("required", True),
                    )
                    for a in pdef.arguments
                ],
            )
            for pdef in self._server._prompt_registry.list_all()
        ]


# ------------------------------------------------------------------
# Internal helpers (same logic as _app.py to maintain parity)
# ------------------------------------------------------------------


# Shared with the live transports (_app.py) so injection is identical on
# every path — no test/prod divergence.
from ._context import inject_context as _inject_context


async def _invoke_handler(handler: Any, arguments: dict[str, Any]) -> Any:
    """Call the handler, supporting both sync and async functions."""
    result = handler(**arguments)
    if asyncio.iscoroutine(result):
        result = await result
    return result


def _serialise_result(result: Any) -> list[Any]:
    """Convert a handler return value to MCP content list.

    Delegates to ``_app._serialise_result`` so TestClient and the real
    server always produce identical output.
    """
    from ._app import _serialise_result as _app_serialise

    return _app_serialise(result)
