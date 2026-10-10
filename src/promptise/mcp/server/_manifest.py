"""Server manifest: introspectable JSON description of the server.

Auto-generates a ``docs://manifest`` resource containing all registered
tools, resources, prompts, and their metadata.

Example::

    from promptise.mcp.server import MCPServer

    server = MCPServer(name="my-api", version="1.0.0")

    @server.tool(tags=["math"], roles=["user"])
    async def add(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b

    # Manifest auto-registered as docs://manifest
    # Contains tool schemas, tags, auth requirements, etc.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any


def build_manifest(server: Any, tools: Iterable[Any] | None = None) -> dict[str, Any]:
    """Build a JSON-serialisable manifest from a server's registrations.

    Args:
        server: An ``MCPServer`` instance.
        tools: The tool definitions to describe (default: every registered
            tool).

    Returns:
        A dict with ``server``, ``tools``, ``resources``, ``prompts`` sections.
    """
    tool_infos: list[dict[str, Any]] = []
    for tdef in server._tool_registry.list_all() if tools is None else tools:
        tool_info: dict[str, Any] = {
            "name": tdef.name,
            "description": tdef.description,
            "input_schema": tdef.input_schema,
        }
        if tdef.tags:
            tool_info["tags"] = tdef.tags
        if tdef.auth:
            tool_info["auth_required"] = True
        if tdef.roles:
            tool_info["roles"] = tdef.roles
        if tdef.guards:
            tool_info["guards"] = [type(g).__name__ for g in tdef.guards]
        if tdef.rate_limit:
            tool_info["rate_limit"] = tdef.rate_limit
        if tdef.timeout:
            tool_info["timeout"] = tdef.timeout
        tool_infos.append(tool_info)

    resources: list[dict[str, Any]] = []
    for rdef in server._resource_registry.list_all():
        resources.append(
            {
                "uri": rdef.uri,
                "name": rdef.name,
                "description": rdef.description,
                "mime_type": rdef.mime_type,
            }
        )

    templates: list[dict[str, Any]] = []
    for rdef in server._resource_registry.list_templates():
        templates.append(
            {
                "uri_template": rdef.uri,
                "name": rdef.name,
                "description": rdef.description,
                "mime_type": rdef.mime_type,
            }
        )

    prompts: list[dict[str, Any]] = []
    for pdef in server._prompt_registry.list_all():
        prompts.append(
            {
                "name": pdef.name,
                "description": pdef.description,
                "arguments": pdef.arguments,
            }
        )

    return {
        "server": {
            "name": server.name,
            "version": server.version,
            "instructions": server.instructions,
        },
        "tools": tool_infos,
        "resources": resources,
        "resource_templates": templates,
        "prompts": prompts,
    }


def register_manifest(server: Any) -> None:
    """Register a ``docs://manifest`` resource on the server.

    The resource returns a JSON manifest describing all tools,
    resources, and prompts registered on the server.

    Args:
        server: An ``MCPServer`` instance.
    """
    from ._decorators import build_resource_def

    async def manifest_handler() -> str:
        tools = None
        if getattr(server, "_hide_unauthorized_tools", False):
            # Same per-caller view as tools/list.
            from ._context import get_context
            from ._visibility import visible_tools

            ctx = get_context()
            tools = await visible_tools(
                server._tool_registry.list_all(),
                server._middlewares,
                server_name=server.name,
                meta=ctx.meta,
            )
        return json.dumps(build_manifest(server, tools), indent=2, default=str)

    res_def = build_resource_def(
        manifest_handler,
        uri="docs://manifest",
        name="manifest",
        description="Server manifest — all tools, resources, and prompts.",
        mime_type="application/json",
    )
    server._resource_registry.register(res_def)
