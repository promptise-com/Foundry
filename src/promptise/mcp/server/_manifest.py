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
from typing import Any


def build_manifest(server: Any) -> dict[str, Any]:
    """Build a JSON-serialisable manifest from a server's registrations.

    Args:
        server: An ``MCPServer`` instance.

    Returns:
        A dict with ``server``, ``tools``, ``resources``, ``prompts`` sections.
    """
    tools: list[dict[str, Any]] = []
    for tdef in server._tool_registry.list_all():
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
        tools.append(tool_info)

    resources: list[dict[str, Any]] = []
    for rdef in server._resource_registry.list_all():
        resources.append(
            {
                "uri": rdef.uri,
                "name": rdef.name,
                "description": rdef.description,
                "mime_type": rdef.mime_type,
                **_access(rdef),
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
                **_access(rdef),
            }
        )

    prompts: list[dict[str, Any]] = []
    for pdef in server._prompt_registry.list_all():
        prompts.append(
            {
                "name": pdef.name,
                "description": pdef.description,
                "arguments": pdef.arguments,
                **_access(pdef),
            }
        )

    return {
        "server": {
            "name": server.name,
            "version": server.version,
            "instructions": server.instructions,
        },
        "tools": tools,
        "resources": resources,
        "resource_templates": templates,
        "prompts": prompts,
    }


def _access(definition: Any) -> dict[str, Any]:
    """Access-control fields of a resource or prompt, as listed for tools."""
    info: dict[str, Any] = {}
    if getattr(definition, "tags", None):
        info["tags"] = definition.tags
    if getattr(definition, "auth", False):
        info["auth_required"] = True
    if getattr(definition, "roles", None):
        info["roles"] = definition.roles
    if getattr(definition, "guards", None):
        info["guards"] = [type(g).__name__ for g in definition.guards]
    if getattr(definition, "rate_limit", None):
        info["rate_limit"] = definition.rate_limit
    if getattr(definition, "timeout", None):
        info["timeout"] = definition.timeout
    return info


def register_manifest(server: Any) -> None:
    """Register a ``docs://manifest`` resource on the server.

    The resource returns a JSON manifest describing all tools,
    resources, and prompts registered on the server.

    Args:
        server: An ``MCPServer`` instance.
    """
    from ._decorators import build_resource_def

    async def manifest_handler() -> str:
        return json.dumps(build_manifest(server), indent=2, default=str)

    res_def = build_resource_def(
        manifest_handler,
        uri="docs://manifest",
        name="manifest",
        description="Server manifest — all tools, resources, and prompts.",
        mime_type="application/json",
    )
    server._resource_registry.register(res_def)
