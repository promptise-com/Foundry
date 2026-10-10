"""Request pipeline for resource reads and prompt requests.

Shared by the live transports (``MCPServer``) and ``TestClient`` so both
run the same steps a tool call does:

    lookup → argument coercion → dependency injection → middleware chain
    → guards → handler → result serialisation

Resources and prompts therefore get the server's logging, rate limits,
audit trail, authentication, ``roles=`` and ``guards=`` exactly like tools.
Middleware tells them apart by ``ctx.request_type``.
"""

from __future__ import annotations

import asyncio
import json
import secrets
from collections.abc import Iterable, Mapping
from typing import Any

from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import (
    EmbeddedResource as MCPEmbeddedResource,
)
from mcp.types import (
    GetPromptResult,
    PromptMessage,
    TextContent,
)
from mcp.types import (
    ImageContent as MCPImageContent,
)
from pydantic import BaseModel

from ._context import RequestContext, clear_context, inject_context, set_context
from ._di import DependencyResolver
from ._errors import MCPError, PromptError, ResourceError
from ._middleware import compile_middleware_chain
from ._validation import _preparse_json_strings, validate_arguments

#: ``MCPError.code`` values for an unknown resource URI / prompt name.
RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
PROMPT_NOT_FOUND = "PROMPT_NOT_FOUND"


async def read_resource(
    server: Any,
    uri: str,
    *,
    meta: Mapping[str, Any] | None = None,
    mcp_session: Any = None,
    extra_middleware: Iterable[Any] = (),
) -> list[ReadResourceContents]:
    """Read *uri* through the full pipeline.

    Raises:
        MCPError: ``RESOURCE_NOT_FOUND`` for an unknown URI, or whatever the
            pipeline raised (authentication, guard denial, validation,
            rate limit, a handler's ``ResourceError`` ...).  Other handler
            exceptions go through the server's exception handlers and are
            re-raised unchanged when none maps them.
    """
    registry = server._resource_registry
    definition = registry.get(uri)
    params: dict[str, Any] = {}
    if definition is None:
        match = registry.match_template(uri)
        if match is None:
            raise ResourceError(
                f"Resource not found: {uri}",
                code=RESOURCE_NOT_FOUND,
                suggestion="List the server's resources and resource templates for valid URIs.",
            )
        definition, params = match

    result = await _run(
        server,
        definition,
        request_type="resource",
        arguments=params,
        meta=meta,
        state={"resource_uri": uri},
        mcp_session=mcp_session,
        extra_middleware=extra_middleware,
    )
    return serialise_resource_result(result, definition.mime_type)


async def get_prompt(
    server: Any,
    name: str,
    arguments: Mapping[str, Any] | None = None,
    *,
    meta: Mapping[str, Any] | None = None,
    mcp_session: Any = None,
    extra_middleware: Iterable[Any] = (),
) -> GetPromptResult:
    """Render prompt *name* through the full pipeline.

    Raises:
        MCPError: ``PROMPT_NOT_FOUND`` for an unknown name, or whatever the
            pipeline raised (see :func:`read_resource`).
        TypeError: The handler returned something that is not a prompt.
    """
    definition = server._prompt_registry.get(name)
    if definition is None:
        raise PromptError(
            f"Prompt not found: {name}",
            code=PROMPT_NOT_FOUND,
            suggestion="List the server's prompts for valid names.",
        )
    result = await _run(
        server,
        definition,
        request_type="prompt",
        arguments=dict(arguments or {}),
        meta=meta,
        state={},
        mcp_session=mcp_session,
        extra_middleware=extra_middleware,
    )
    return normalise_prompt_result(result, definition.description)


async def _run(
    server: Any,
    definition: Any,
    *,
    request_type: str,
    arguments: dict[str, Any],
    meta: Mapping[str, Any] | None,
    state: dict[str, Any],
    mcp_session: Any,
    extra_middleware: Iterable[Any],
) -> Any:
    meta = dict(meta or {})
    ctx = RequestContext(
        server_name=server.name,
        tool_name=definition.name,
        request_id=str(meta.get("x-request-id") or "") or secrets.token_hex(6),
        meta=meta,
        request_type=request_type,
    )
    ctx.state["tool_def"] = definition
    ctx.state.update(state)
    ctx.state["_mcp_session"] = mcp_session
    set_context(ctx)

    di_resolver = DependencyResolver()
    try:
        model = definition.input_model
        if model is not None:
            # MCP carries template parameters and prompt arguments as
            # strings: coerce them to the handler's type hints ("3" → 3,
            # '["a", "b"]' → a list).
            arguments = validate_arguments(model, _preparse_json_strings(arguments, model))
        ctx.state["_tool_arguments"] = dict(arguments)

        arguments = await di_resolver.resolve(definition.handler, arguments)
        arguments = inject_context(definition.handler, arguments, ctx)

        # Guards run inside the chain, after AuthMiddleware has populated
        # the client's roles — the same order as for tools.
        handler = definition.handler
        if definition.guards:
            from ._testing import check_guards

            guards = definition.guards
            real = definition.handler

            async def _guarded(**kw: Any) -> Any:
                await check_guards(guards, ctx)
                r = real(**kw)
                if asyncio.iscoroutine(r):
                    r = await r
                return r

            handler = _guarded

        chain = compile_middleware_chain(
            [*server._middlewares, *extra_middleware, *definition.router_middleware]
        )
        return await chain(ctx, handler, arguments)
    except MCPError:
        raise
    except Exception as exc:
        mapped = await server._exception_handlers.handle(ctx, exc)
        if mapped is not None:
            raise mapped from exc
        raise
    finally:
        await di_resolver.cleanup()
        clear_context()


# ------------------------------------------------------------------
# Result serialisation
# ------------------------------------------------------------------


def serialise_resource_result(result: Any, mime_type: str) -> list[ReadResourceContents]:
    """Convert a resource handler's return value to MCP resource contents.

    - ``str`` → text with the declared MIME type
    - ``bytes`` / ``bytearray`` / ``memoryview`` → binary (sent base64-encoded
      as ``BlobResourceContents``)
    - ``dict`` / ``list`` / ``tuple`` → JSON text
    - a Pydantic model → its JSON
    - ``ReadResourceContents`` or a list of them → passed through (several
      contents, or per-content MIME types)
    - ``None`` → empty text; anything else → ``str(value)``

    A handler with no explicit ``mime_type`` and no telling return
    annotation is registered as ``text/plain``; for such a handler, bytes
    are sent as ``application/octet-stream`` and JSON as
    ``application/json``.
    """
    if isinstance(result, ReadResourceContents):
        return [result]
    if (
        isinstance(result, (list, tuple))
        and result
        and all(isinstance(item, ReadResourceContents) for item in result)
    ):
        return list(result)

    default_mime = mime_type == "text/plain"
    if isinstance(result, (bytes, bytearray, memoryview)):
        mime = "application/octet-stream" if default_mime else mime_type
        return [ReadResourceContents(bytes(result), mime)]
    if isinstance(result, str):
        return [ReadResourceContents(result, mime_type)]
    if result is None:
        return [ReadResourceContents("", mime_type)]
    json_mime = "application/json" if default_mime else mime_type
    if isinstance(result, BaseModel):
        return [ReadResourceContents(result.model_dump_json(), json_mime)]
    if isinstance(result, (dict, list, tuple)):
        return [ReadResourceContents(json.dumps(result, default=str), json_mime)]
    return [ReadResourceContents(str(result), mime_type)]


_CONTENT_TYPES = (TextContent, MCPImageContent, MCPEmbeddedResource)


def normalise_prompt_result(result: Any, description: str | None) -> GetPromptResult:
    """Convert a prompt handler's return value to a ``GetPromptResult``.

    Accepts a ``GetPromptResult``, or one or a list of: ``str`` (a user
    message), ``PromptMessage``, a ``{"role": ..., "content": ...}`` dict
    (string content becomes text), an MCP content block (``TextContent``,
    ``ImageContent``, ``EmbeddedResource``) or the server's
    ``ImageContent`` helper (each a user message).

    Raises:
        TypeError: For any other value, naming what is accepted.
    """
    if isinstance(result, GetPromptResult):
        if result.description is None and description:
            return result.model_copy(update={"description": description})
        return result
    items = list(result) if isinstance(result, (list, tuple)) else [result]
    return GetPromptResult(description=description, messages=[_to_message(i) for i in items])


def _to_message(item: Any) -> PromptMessage:
    from ._types import ImageContent

    if isinstance(item, PromptMessage):
        return item
    if isinstance(item, str):
        return PromptMessage(role="user", content=TextContent(type="text", text=item))
    if isinstance(item, ImageContent):
        return PromptMessage(role="user", content=item.to_mcp())
    if isinstance(item, _CONTENT_TYPES):
        return PromptMessage(role="user", content=item)
    if isinstance(item, Mapping) and "role" in item and "content" in item:
        content = item["content"]
        if isinstance(content, str):
            content = TextContent(type="text", text=content)
        elif isinstance(content, ImageContent):
            content = content.to_mcp()
        return PromptMessage.model_validate({"role": item["role"], "content": content})
    raise TypeError(
        f"A prompt handler returned {type(item).__name__}; return a str, a PromptMessage, "
        "a {'role': ..., 'content': ...} dict, an MCP content block, a list of those, "
        "or a GetPromptResult"
    )


# ------------------------------------------------------------------
# Protocol errors
# ------------------------------------------------------------------

# JSON-RPC error codes (the MCP spec assigns -32002 to "resource not found").
_INVALID_PARAMS = -32602
_INTERNAL_ERROR = -32603
_RESOURCE_NOT_FOUND = -32002
_SERVER_ERROR = -32000


def to_protocol_error(exc: MCPError) -> Exception:
    """Map an ``MCPError`` to the SDK's ``McpError`` (a JSON-RPC error response).

    The message is the error's own; ``data`` carries the structured payload
    (code, retryable, suggestion, details) that tool errors carry in their
    text content.
    """
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData

    if exc.code == RESOURCE_NOT_FOUND:
        rpc_code = _RESOURCE_NOT_FOUND
    elif exc.code in (PROMPT_NOT_FOUND, "VALIDATION_ERROR"):
        rpc_code = _INVALID_PARAMS
    elif exc.code == "INTERNAL_ERROR":
        rpc_code = _INTERNAL_ERROR
    else:
        rpc_code = _SERVER_ERROR
    return McpError(ErrorData(code=rpc_code, message=str(exc), data=exc.detail.to_dict()))


def internal_protocol_error() -> Exception:
    """A generic error for an unhandled exception — never leaks its text."""
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData

    return McpError(
        ErrorData(
            code=_INTERNAL_ERROR,
            message="An internal error occurred.",
            data={
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred.",
                "retryable": False,
            },
        )
    )
