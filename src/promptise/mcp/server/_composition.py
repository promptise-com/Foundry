"""Server composition — mount sub-servers into a parent server.

Allows composing multiple MCPServer instances into a single server,
each with its own namespace prefix.

Example::

    from promptise.mcp.server import MCPServer, mount

    main_server = MCPServer(name="gateway")
    math_server = MCPServer(name="math")
    db_server = MCPServer(name="database")

    @math_server.tool()
    async def add(a: int, b: int) -> int:
        return a + b

    @db_server.tool()
    async def query(sql: str) -> list:
        return []

    mount(main_server, math_server, prefix="math")
    mount(main_server, db_server, prefix="db")
    # Tools: math_add, db_query

    main_server.run()
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

logger = logging.getLogger("promptise.server")


def mount(
    parent: Any,
    child: Any,
    *,
    prefix: str = "",
    tags: list[str] | None = None,
) -> int:
    """Mount a child server's tools, resources, and prompts into a parent.

    All tool names are prefixed with ``{prefix}_`` if a prefix is given.
    Tags from the parent are merged with child tags.

    Args:
        parent: The parent ``MCPServer`` to mount into.
        child: The child ``MCPServer`` whose registrations will be copied.
        prefix: Namespace prefix for tool names.
        tags: Additional tags applied to all mounted tools.

    Returns:
        Number of tools mounted.
    """
    extra_tags = tags or []
    count = 0

    # Mount tools
    for tdef in child._tool_registry.list_all():
        name = f"{prefix}_{tdef.name}" if prefix else tdef.name
        merged_tags = list(tdef.tags) + extra_tags

        # replace() copies EVERY field and overrides only the prefixed name +
        # merged tags — no ToolDef field (auth, guards, requires_approval, ...)
        # can be silently dropped when mounting one server into another.
        new_def = replace(
            tdef,
            name=name,
            tags=merged_tags,
            guards=list(tdef.guards),
            roles=list(tdef.roles),
            router_middleware=list(tdef.router_middleware),
        )
        parent._tool_registry.register(new_def)

        # Copy input model
        if tdef.name in child._input_models:
            parent._input_models[name] = child._input_models[tdef.name]

        count += 1

    # Mount resources and prompts.  Copies, so that the parent's server-wide
    # invariants (``require_auth`` / ``require_tenant``, applied in place at
    # build time) never alter the child's own definitions.  Each keeps its
    # auth flag, roles and guards.
    def _copy(definition: Any) -> Any:
        return replace(
            definition,
            guards=list(definition.guards),
            roles=list(definition.roles),
            router_middleware=list(definition.router_middleware),
        )

    for rdef in child._resource_registry.list_all():
        if rdef.uri == "docs://manifest":
            continue  # the parent serves its own manifest
        parent._resource_registry.register(_copy(rdef))

    for rdef in child._resource_registry.list_templates():
        parent._resource_registry.register(_copy(rdef))

    for pdef in child._prompt_registry.list_all():
        parent._prompt_registry.register(_copy(pdef))

    # Copy exception handlers
    for exc_type, handler in child._exception_handlers._handlers.items():
        parent._exception_handlers.register(exc_type, handler)

    logger.info(
        "Mounted %d tools from '%s' into '%s' (prefix=%r)",
        count,
        child.name,
        parent.name,
        prefix,
    )
    return count
