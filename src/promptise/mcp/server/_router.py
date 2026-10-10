"""MCPRouter — modular tool/resource/prompt grouping.

Like FastAPI's ``APIRouter``, allows splitting tools across files
and composing them into a server with shared defaults.

Example::

    from promptise.mcp.server import MCPServer, MCPRouter

    db_router = MCPRouter(prefix="db", tags=["database"])

    @db_router.tool()
    async def query(sql: str) -> list:
        \"\"\"Execute a SQL query.\"\"\"
        return []

    server = MCPServer(name="api")
    server.include_router(db_router)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

from ._decorators import _excluded_params, build_prompt_def, build_resource_def, build_tool_def
from ._registry import PromptRegistry, ResourceRegistry, ToolRegistry
from ._validation import build_input_model


@dataclass
class RouterConfig:
    """Router-level defaults applied to all registrations."""

    prefix: str = ""
    tags: list[str] = field(default_factory=list)
    auth: bool | None = None
    middleware: list[Any] = field(default_factory=list)
    guards: list[Any] = field(default_factory=list)


class MCPRouter:
    """Modular grouping for tools, resources, and prompts.

    Args:
        prefix: Prepended to all tool names (e.g. ``"db"`` → ``"db_search"``).
        tags: Default tags merged with per-tool tags.
        auth: If set, overrides the per-tool / per-resource / per-prompt
            auth flag.
        middleware: Router-level middleware (runs after server middleware)
            for the router's tools, resources and prompts.
        guards: Router-level guards applied to all tools, resources and
            prompts.
    """

    def __init__(
        self,
        *,
        prefix: str = "",
        tags: list[str] | None = None,
        auth: bool | None = None,
        middleware: list[Any] | None = None,
        guards: list[Any] | None = None,
    ) -> None:
        self.config = RouterConfig(
            prefix=prefix,
            tags=tags or [],
            auth=auth,
            middleware=middleware or [],
            guards=guards or [],
        )
        self._tool_registry = ToolRegistry()
        self._resource_registry = ResourceRegistry()
        self._prompt_registry = PromptRegistry()
        self._input_models: dict[str, type] = {}
        self._sub_routers: list[tuple[MCPRouter, RouterConfig]] = []

    def tool(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        tags: list[str] | None = None,
        auth: bool = False,
        rate_limit: str | None = None,
        timeout: float | None = None,
        guards: list[Any] | None = None,
        roles: list[str] | None = None,
        # Tool annotations (MCP spec hints)
        title: str | None = None,
        read_only_hint: bool | None = None,
        destructive_hint: bool | None = None,
        idempotent_hint: bool | None = None,
        open_world_hint: bool | None = None,
        # Per-tool concurrency limit
        max_concurrent: int | None = None,
        # Server-side human-in-the-loop approval
        requires_approval: bool = False,
        # Opt out of CacheMiddleware
        cache: bool = True,
    ) -> Callable[..., Any]:
        """Register a tool (same signature as ``MCPServer.tool()``)."""
        all_guards = list(guards or [])
        if roles:
            from ._guards import HasRole

            all_guards.append(HasRole(*roles))
            # Roles cannot be enforced without authentication — the
            # HasRole guard reads ``ctx.state["roles"]`` which is only
            # populated by ``AuthMiddleware`` when ``tool_def.auth`` is
            # truthy.  Silently upgrading here prevents the footgun
            # where ``roles=[...]`` looks like it enforces RBAC but
            # actually always denies with "client has [(none)]".
            auth = True

        # Build annotations if any hint is provided
        from ._types import ToolAnnotations

        annotations = None
        if any(
            v is not None
            for v in [title, read_only_hint, destructive_hint, idempotent_hint, open_world_hint]
        ):
            annotations = ToolAnnotations(
                title=title,
                read_only_hint=read_only_hint,
                destructive_hint=destructive_hint,
                idempotent_hint=idempotent_hint,
                open_world_hint=open_world_hint,
            )

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            tool_def = build_tool_def(
                func,
                name=name,
                description=description,
                tags=tags,
                auth=auth,
                rate_limit=rate_limit,
                timeout=timeout,
                guards=all_guards,
                roles=roles,
                annotations=annotations,
                max_concurrent=max_concurrent,
                requires_approval=requires_approval,
                cache=cache,
            )
            self._tool_registry.register(tool_def)
            excluded = _excluded_params(func)
            model, _ = build_input_model(func, exclude=excluded)
            self._input_models[tool_def.name] = model
            return func

        return decorator

    def resource(
        self,
        uri: str,
        *,
        name: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
        tags: list[str] | None = None,
        auth: bool = False,
        roles: list[str] | None = None,
        guards: list[Any] | None = None,
        rate_limit: str | None = None,
        timeout: float | None = None,
    ) -> Callable[..., Any]:
        """Register a resource (same signature as ``MCPServer.resource()``)."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            res_def = build_resource_def(
                func,
                uri=uri,
                name=name,
                description=description,
                mime_type=mime_type,
                is_template=False,
                tags=tags,
                auth=auth,
                roles=roles,
                guards=guards,
                rate_limit=rate_limit,
                timeout=timeout,
            )
            self._resource_registry.register(res_def)
            return func

        return decorator

    def resource_template(
        self,
        uri_template: str,
        *,
        name: str | None = None,
        description: str | None = None,
        mime_type: str | None = None,
        tags: list[str] | None = None,
        auth: bool = False,
        roles: list[str] | None = None,
        guards: list[Any] | None = None,
        rate_limit: str | None = None,
        timeout: float | None = None,
    ) -> Callable[..., Any]:
        """Register a resource template (same signature as ``MCPServer.resource_template()``)."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            res_def = build_resource_def(
                func,
                uri=uri_template,
                name=name,
                description=description,
                mime_type=mime_type,
                is_template=True,
                tags=tags,
                auth=auth,
                roles=roles,
                guards=guards,
                rate_limit=rate_limit,
                timeout=timeout,
            )
            self._resource_registry.register(res_def)
            return func

        return decorator

    def prompt(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
        tags: list[str] | None = None,
        auth: bool = False,
        roles: list[str] | None = None,
        guards: list[Any] | None = None,
        rate_limit: str | None = None,
        timeout: float | None = None,
    ) -> Callable[..., Any]:
        """Register a prompt (same signature as ``MCPServer.prompt()``)."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            prompt_def = build_prompt_def(
                func,
                name=name,
                description=description,
                tags=tags,
                auth=auth,
                roles=roles,
                guards=guards,
                rate_limit=rate_limit,
                timeout=timeout,
            )
            self._prompt_registry.register(prompt_def)
            return func

        return decorator

    def include_prompts(self, *sources: Any) -> None:
        """Register Promptise prompts as MCP prompt endpoints on this router.

        Accepts :class:`~promptise.prompts.registry.PromptRegistry`,
        :class:`~promptise.prompts.core.Prompt`, or
        :class:`~promptise.prompts.suite.PromptSuite` instances.

        Args:
            *sources: Prompt registries, individual prompts, or suites.
        """
        from promptise.prompts.core import Prompt as PaCPrompt
        from promptise.prompts.registry import PromptRegistry as PaCRegistry
        from promptise.prompts.suite import PromptSuite

        from ._app import _prompt_to_mcp_def

        for source in sources:
            if isinstance(source, PaCRegistry):
                for name in source.list():
                    p = source.get(name)
                    ver = source.latest_version(name)
                    pdef = _prompt_to_mcp_def(p, version=ver)
                    self._prompt_registry.register(pdef)
            elif isinstance(source, PromptSuite):
                for _name, p in source.prompts.items():
                    pdef = _prompt_to_mcp_def(p)
                    self._prompt_registry.register(pdef)
            elif isinstance(source, PaCPrompt):
                pdef = _prompt_to_mcp_def(source)
                self._prompt_registry.register(pdef)
            else:
                raise TypeError(
                    f"include_prompts() expects Prompt, PromptSuite, or "
                    f"PromptRegistry, got {type(source).__name__}"
                )

    def include_router(
        self,
        router: MCPRouter,
        *,
        prefix: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """Nest a sub-router."""
        override = RouterConfig(
            prefix=prefix or router.config.prefix,
            tags=(tags or []) + router.config.tags,
        )
        self._sub_routers.append((router, override))


# =====================================================================
# Merge logic
# =====================================================================


def _merge_router(
    server: Any,
    router: MCPRouter,
    resolved_prefix: str = "",
    extra_tags: list[str] | None = None,
    parent_middleware: list[Any] | None = None,
    parent_guards: list[Any] | None = None,
) -> None:
    """Recursively merge router registrations into a server.

    Args:
        server: The ``MCPServer`` to merge into.
        router: The ``MCPRouter`` whose registrations are merged.
        resolved_prefix: Final computed prefix (caller combines all parts).
        extra_tags: Additional tags from parent routers.
        parent_middleware: Middleware accumulated from parent routers.
        parent_guards: Guards accumulated from parent routers.
    """
    combined_tags = (extra_tags or []) + router.config.tags
    combined_middleware = (parent_middleware or []) + router.config.middleware
    combined_guards = (parent_guards or []) + router.config.guards

    # Merge tools
    for tdef in router._tool_registry.list_all():
        prefixed_name = f"{resolved_prefix}_{tdef.name}" if resolved_prefix else tdef.name
        merged_tags = combined_tags + list(tdef.tags)
        merged_auth = router.config.auth if router.config.auth is not None else tdef.auth
        # Force auth when server requires it
        if hasattr(server, "_require_auth") and server._require_auth:
            merged_auth = True
        merged_guards = combined_guards + list(tdef.guards)

        # replace() copies EVERY field and overrides only what the merge
        # changes — so a newly added ToolDef field (e.g. requires_approval)
        # can never be silently dropped during composition.
        new_tdef = replace(
            tdef,
            name=prefixed_name,
            tags=merged_tags,
            auth=merged_auth,
            guards=merged_guards,
            router_middleware=combined_middleware,
            roles=list(tdef.roles),
        )
        server._tool_registry.register(new_tdef)

        # Copy input model under new name
        if tdef.name in router._input_models:
            server._input_models[prefixed_name] = router._input_models[tdef.name]

    # Resources and prompts get the router's auth, guards, middleware and
    # tags exactly like tools.  URIs and prompt names are not prefixed.
    def _scoped(definition: Any) -> Any:
        merged_auth = router.config.auth if router.config.auth is not None else definition.auth
        if getattr(server, "_require_auth", False):
            merged_auth = True
        return replace(
            definition,
            tags=combined_tags + list(definition.tags),
            auth=merged_auth,
            guards=combined_guards + list(definition.guards),
            router_middleware=combined_middleware + list(definition.router_middleware),
            roles=list(definition.roles),
        )

    for rdef in router._resource_registry.list_all():
        server._resource_registry.register(_scoped(rdef))
    for rdef in router._resource_registry.list_templates():
        server._resource_registry.register(_scoped(rdef))

    for pdef in router._prompt_registry.list_all():
        server._prompt_registry.register(_scoped(pdef))

    # Recurse into sub-routers
    for sub_router, sub_config in router._sub_routers:
        sub_parts = [p for p in [resolved_prefix, sub_config.prefix] if p]
        sub_prefix = "_".join(sub_parts)
        _merge_router(
            server=server,
            router=sub_router,
            resolved_prefix=sub_prefix,
            extra_tags=combined_tags + sub_config.tags,
            parent_middleware=combined_middleware,
            parent_guards=combined_guards,
        )
