"""MCPServer — production-grade MCP server with decorator-based registration.

Example::

    from promptise.mcp.server import MCPServer

    server = MCPServer(name="my-tools", version="1.0.0")

    @server.tool()
    async def add(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b

    server.run()  # stdio by default
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import secrets
import weakref
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pydantic import AnyUrl

    from ._types import PromptDef

from mcp.server.lowlevel import NotificationOptions
from mcp.server.lowlevel import Server as LowLevelServer
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.types import (
    EmbeddedResource as MCPEmbeddedResource,
)
from mcp.types import (
    GetPromptResult,
    PromptArgument,
    Resource,
    ResourceTemplate,
    TextContent,
    Tool,
)
from mcp.types import (
    ImageContent as MCPImageContent,
)
from mcp.types import (
    ToolAnnotations as MCPToolAnnotations,
)

from ._context import RequestContext, _current_context, clear_context, set_context
from ._decorators import build_prompt_def, build_resource_def, build_tool_def
from ._di import DependencyResolver
from ._errors import MCPError
from ._lifecycle import LifecycleManager
from ._middleware import compile_middleware_chain, with_tool_timeout
from ._registry import PromptRegistry, ResourceRegistry, ToolRegistry
from ._transport import TransportType, run_transport
from ._validation import build_input_model, validate_arguments

logger = logging.getLogger("promptise.server")


class _LowLevelServer(LowLevelServer):  # type: ignore[type-arg]
    """``mcp`` low-level server that advertises ``listChanged`` by default.

    ``MCPServer`` sends ``notifications/{tools,resources,prompts}/list_changed``
    (see :meth:`MCPServer.notify_tools_changed`), so the capability is on
    however the initialisation options are created — including by the SDK's
    own in-memory test transport.
    """

    def create_initialization_options(
        self,
        notification_options: NotificationOptions | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if notification_options is None:
            notification_options = NotificationOptions(
                prompts_changed=True, resources_changed=True, tools_changed=True
            )
        return super().create_initialization_options(notification_options, *args, **kwargs)


class _ToolErrorResult(Exception):
    """A tool call that failed; its message is the error payload (JSON text).

    Raised from the low-level ``call_tool`` handler, which the MCP SDK turns
    into ``CallToolResult(content=[TextContent(text=<message>)], isError=True)``
    -- the spec's tool-execution error, so clients can tell a failure from a
    result.  (Returning a ``CallToolResult`` directly needs a newer SDK than
    the supported ``mcp>=1.9``.)
    """


class MCPServer:
    """Production-grade MCP server with decorator-based tool registration.

    Args:
        name: Server name advertised to MCP clients.
        version: Server version string.
        instructions: Optional instructions sent to clients on initialisation.
        require_auth: Force ``auth=True`` on every registered tool,
            resource and prompt.
        require_tenant: Make tenant identity a server-wide invariant: every
            tool, resource and prompt authenticates and carries a
            ``RequireTenant`` guard, so a client whose token lacks the
            tenant claim is denied on every call.  Implies ``require_auth``.
        hide_unauthorized_tools: Filter ``tools/list``, ``resources/list``,
            ``resources/templates/list``, ``prompts/list`` (and the
            ``docs://manifest`` resource) per request, so each caller sees
            only what its identity may use: the request is authenticated
            with the server's ``AuthMiddleware`` and every definition's
            guards are evaluated against it.  Off by default, in which case
            everything is listed to everyone and guards apply when a tool
            is called, a resource read or a prompt requested.  Guards that
            depend on state set by other middleware (not ``AuthMiddleware``
            or its ``on_authenticate`` hook) cannot be evaluated at list
            time and hide their definition.
        cancel_grace_period: When a client cancels a call to a tool that
            takes a ``CancellationToken``, the token is set and the handler
            gets this many seconds to stop on its own before its task is
            cancelled.  The client is answered immediately either way.
    """

    def __init__(
        self,
        name: str = "promptise-server",
        version: str = "0.1.0",
        *,
        instructions: str | None = None,
        auto_manifest: bool = True,
        shutdown_timeout: float | None = 30.0,
        require_auth: bool = False,
        require_tenant: bool = False,
        hide_unauthorized_tools: bool = False,
        cancel_grace_period: float = 5.0,
    ) -> None:
        self.name = name
        self.version = version
        self.instructions = instructions
        self._shutdown_timeout = shutdown_timeout
        self._require_auth = require_auth or require_tenant
        self._require_tenant = require_tenant
        self._hide_unauthorized_tools = hide_unauthorized_tools
        self._cancel_grace_period = cancel_grace_period

        # BackgroundTasks run after the response is sent, in tasks tracked
        # here so shutdown can let them finish.
        self._background_runs: set[asyncio.Task[None]] = set()
        self._background_drain_registered = False

        self._tool_registry = ToolRegistry(on_change=lambda: self._on_registry_change("tools"))
        self._resource_registry = ResourceRegistry(
            on_change=lambda: self._on_registry_change("resources")
        )
        self._prompt_registry = PromptRegistry(
            on_change=lambda: self._on_registry_change("prompts")
        )
        self._lifecycle = LifecycleManager()

        # MCP sessions seen by this server (for list_changed notifications),
        # and the event loop serving them (set by the first request).
        self._sessions: weakref.WeakSet[Any] = weakref.WeakSet()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending_notifications: set[str] = set()
        self._notification_tasks: set[asyncio.Task[Any]] = set()

        # Middleware chain
        self._middlewares: list[Any] = []

        # Auth provider (auto-tracked from AuthMiddleware for transport gating)
        self._auth_provider: Any = None

        # Exception handler registry
        from ._exception_handlers import ExceptionHandlerRegistry

        self._exception_handlers = ExceptionHandlerRegistry()

        # Pydantic models for input validation, keyed by tool name
        self._input_models: dict[str, type] = {}

        # Auto-register manifest resource
        self._auto_manifest = auto_manifest

        # Token endpoint (None until enable_token_endpoint() is called)
        self._token_endpoint: Any = None

        # Readiness checks behind GET /health/ready (set by
        # HealthCheck.register_resources)
        self._health_check: Any = None

        # Per-session state manager
        from ._session_state import SessionManager

        self._session_manager = SessionManager()

    # ------------------------------------------------------------------
    # Decorator API
    # ------------------------------------------------------------------

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
        """Register a function as an MCP tool.

        Args:
            name: Tool name (defaults to function name).
            description: Tool description (defaults to docstring first line).
            tags: Optional tags for categorisation.
            auth: Require authentication for this tool.
            rate_limit: Rate limit string, e.g. ``"100/min"``.
            timeout: Per-call timeout in seconds.
            guards: Access control guards (checked before handler).
            roles: Required roles shorthand (creates ``HasRole`` guard).
            title: Human-readable title (MCP annotation hint).
            read_only_hint: Tool does not modify state (MCP annotation hint).
            destructive_hint: Tool may perform destructive operations
                (MCP annotation hint).
            idempotent_hint: Repeated calls with same args produce same
                result (MCP annotation hint).
            open_world_hint: Tool may interact with external systems
                (MCP annotation hint).
            max_concurrent: Maximum concurrent calls for this tool.
                When reached, additional calls receive a retryable error.
            requires_approval: Require a human approval decision before every
                call to this tool, enforced **server-side** for any MCP
                client.  The server must install an
                ``ApprovalGateMiddleware`` — building a server with an
                ungated ``requires_approval`` tool raises at build time
                rather than silently not enforcing it.
            cache: ``False`` keeps a server-wide ``CacheMiddleware`` from
                caching this tool's results (use it for tools that change
                data or must always be fresh).  Tools annotated
                ``destructive_hint=True`` are never cached either.

        Example::

            @server.tool()
            async def search(query: str, limit: int = 10) -> list[dict]:
                \"\"\"Search records.\"\"\"
                return await db.search(query, limit)
        """
        # Force auth when server requires it
        if self._require_auth:
            auth = True

        # roles shorthand → HasRole guard
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

            # Pre-build the Pydantic model for validation
            excluded = _excluded_params_for(func)
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
        """Register a function as an MCP resource.

        Reads run through the server's middleware chain, like tool calls.

        Args:
            uri: Static resource URI (e.g. ``"config://app"``).
            name: Resource name (defaults to function name).
            description: Description (defaults to docstring).
            mime_type: MIME type of the resource content.  Defaults to
                ``application/json`` for a handler annotated ``-> dict`` /
                ``-> list`` (or a Pydantic model), ``application/octet-stream``
                for ``-> bytes``, and ``text/plain`` otherwise.
            tags: Optional tags for categorisation.
            auth: Require authentication to read this resource.
            roles: Required roles shorthand (adds a ``HasRole`` guard and
                implies ``auth=True``).
            guards: Access control guards, checked before the handler.
            rate_limit: Rate limit string, e.g. ``"100/min"``.
            timeout: Per-read timeout in seconds (enforced by
                ``TimeoutMiddleware``).
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            res_def = build_resource_def(
                func,
                uri=uri,
                name=name,
                description=description,
                mime_type=mime_type,
                is_template=False,
                tags=tags,
                auth=auth or self._require_auth,
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
        """Register a function as an MCP resource template.

        Each ``{param}`` placeholder matches one path segment and is passed
        to the handler parameter of the same name, coerced to its type hint
        (``{page}`` → ``int``).  ``{param*}`` (or ``{+param}``) matches the
        rest of the URI including ``/``, for hierarchical ids such as
        ``docs://pages/{path*}``.

        Args:
            uri_template: URI template with ``{param}`` placeholders.
            name: Resource name.
            description: Description.
            mime_type: MIME type (see :meth:`resource`).
            tags: Optional tags for categorisation.
            auth: Require authentication to read these resources.
            roles: Required roles shorthand (implies ``auth=True``).
            guards: Access control guards, checked before the handler.
            rate_limit: Rate limit string, e.g. ``"100/min"``.
            timeout: Per-read timeout in seconds.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            res_def = build_resource_def(
                func,
                uri=uri_template,
                name=name,
                description=description,
                mime_type=mime_type,
                is_template=True,
                tags=tags,
                auth=auth or self._require_auth,
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
        """Register a function as an MCP prompt.

        The handler may return a ``str`` (one user message), a
        ``PromptMessage``, a list mixing those (or ``{"role", "content"}``
        dicts), or a full ``GetPromptResult``.  MCP sends arguments as
        strings; they are coerced to the handler's type hints.  Requests run
        through the server's middleware chain, like tool calls.

        Args:
            name: Prompt name (defaults to function name).
            description: Description (defaults to docstring).
            tags: Optional tags for categorisation.
            auth: Require authentication to get this prompt.
            roles: Required roles shorthand (implies ``auth=True``).
            guards: Access control guards, checked before the handler.
            rate_limit: Rate limit string, e.g. ``"100/min"``.
            timeout: Per-request timeout in seconds.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            prompt_def = build_prompt_def(
                func,
                name=name,
                description=description,
                tags=tags,
                auth=auth or self._require_auth,
                roles=roles,
                guards=guards,
                rate_limit=rate_limit,
                timeout=timeout,
            )
            self._prompt_registry.register(prompt_def)
            return func

        return decorator

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    def add_middleware(self, middleware: Any) -> None:
        """Add a middleware to the processing chain.

        The chain runs for tool calls, resource reads and prompt requests;
        ``ctx.request_type`` says which.
        """
        self._middlewares.append(middleware)

        # Auto-track auth provider for transport-level gating
        from ._auth import AuthMiddleware

        if isinstance(middleware, AuthMiddleware):
            self._auth_provider = middleware._provider

    @property
    def middleware(self) -> Callable[..., Any]:
        """Decorator to register a middleware function.

        Example::

            @server.middleware
            async def log_calls(ctx, call_next):
                result = await call_next(ctx)
                return result
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._middlewares.append(func)
            return func

        return decorator

    # ------------------------------------------------------------------
    # Exception handlers
    # ------------------------------------------------------------------

    def exception_handler(
        self,
        exc_type: type[Exception],
    ) -> Callable[..., Any]:
        """Register a custom exception handler.

        The handler receives ``(ctx, exc)`` and should return an
        ``MCPError`` instance.

        Example::

            @server.exception_handler(DatabaseError)
            async def handle_db_error(ctx, exc):
                return ToolError("DB unavailable", code="DB_ERROR", retryable=True)
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            self._exception_handlers.register(exc_type, func)
            return func

        return decorator

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def include_router(
        self,
        router: Any,
        *,
        prefix: str = "",
        tags: list[str] | None = None,
    ) -> None:
        """Merge a router's registrations into this server.

        Args:
            router: The ``MCPRouter`` to include.
            prefix: Additional prefix (combined with router's own prefix).
            tags: Additional tags (combined with router's own tags).
        """
        from ._router import _merge_router

        parts = [p for p in [prefix, router.config.prefix] if p]
        full_prefix = "_".join(parts)
        _merge_router(
            server=self,
            router=router,
            resolved_prefix=full_prefix,
            extra_tags=tags or [],
        )

    # ------------------------------------------------------------------
    # Promptise prompt bridge
    # ------------------------------------------------------------------

    def include_prompts(self, *sources: Any) -> None:
        """Register Promptise prompts as MCP prompt endpoints.

        Accepts any combination of:

        - :class:`~promptise.prompts.registry.PromptRegistry` — exposes
          the latest version of every registered prompt.
        - :class:`~promptise.prompts.core.Prompt` — exposes a single prompt.
        - :class:`~promptise.prompts.suite.PromptSuite` — exposes all
          prompts in the suite.

        Each prompt is converted to an MCP ``PromptDef`` with arguments
        extracted from the prompt's function signature.  Each argument is
        described by the YAML file's ``arguments``, an
        ``Annotated[..., Field(description=...)]`` hint, or else by the
        placeholder it fills, its type and its default.  The MCP handler
        converts the string-valued arguments to the parameters' types and
        calls ``render_async(**arguments)`` to produce fully rendered
        prompt text with context providers, strategy, perspective, and
        constraints applied.  Templates written as docstrings are dedented,
        so the code's indentation is not sent to the model.

        Args:
            *sources: Prompt registries, individual prompts, or suites.

        Example::

            from promptise.prompts.registry import registry
            server.include_prompts(registry)
        """
        from promptise.prompts.core import Prompt as PaCPrompt
        from promptise.prompts.registry import PromptRegistry as PaCRegistry
        from promptise.prompts.suite import PromptSuite

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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_startup(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Register a startup hook."""
        self._lifecycle.add_startup(func)
        return func

    def on_shutdown(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """Register a shutdown hook."""
        self._lifecycle.add_shutdown(func)
        return func

    # ------------------------------------------------------------------
    # Token endpoint (built-in auth for dev/testing)
    # ------------------------------------------------------------------

    def enable_token_endpoint(
        self,
        jwt_auth: Any,
        clients: dict[str, dict[str, Any]],
        *,
        path: str = "/auth/token",
        default_expires_in: int = 86400,
    ) -> None:
        """Enable a built-in token endpoint for development and testing.

        This adds an ``/auth/token`` HTTP endpoint that issues JWT
        tokens using the OAuth2 client_credentials flow.  Clients send
        ``{"client_id": "...", "client_secret": "..."}`` and receive
        a signed JWT back.

        **For production**, use a proper Identity Provider (Auth0,
        Keycloak, Okta, etc.) instead.

        Args:
            jwt_auth: The ``JWTAuth`` instance used to sign tokens
                (same one passed to ``AuthMiddleware``).
            clients: Mapping of ``client_id`` → config dict.  Each
                config must include ``"secret"`` and may include
                ``"roles"`` (list), ``"expires_in"`` (int, seconds),
                and ``"claims"`` (dict of extra JWT claims).
            path: HTTP path for the endpoint (default ``/auth/token``).
            default_expires_in: Default token lifetime in seconds.

        Example::

            jwt_auth = JWTAuth(secret="my-secret")
            server.add_middleware(AuthMiddleware(jwt_auth))

            server.enable_token_endpoint(
                jwt_auth=jwt_auth,
                clients={
                    "agent-admin":  {"secret": "s3cret",  "roles": ["admin"]},
                    "agent-viewer": {"secret": "v1ewer",  "roles": ["viewer"]},
                },
            )
        """
        from ._token_endpoint import TokenEndpointConfig

        self._token_endpoint = TokenEndpointConfig(
            jwt_auth=jwt_auth,
            clients=clients,
            path=path,
            default_expires_in=default_expires_in,
        )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(
        self,
        transport: str = "stdio",
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        dashboard: bool = False,
        cors: Any = None,
        allowed_hosts: list[str] | None = None,
        allowed_origins: list[str] | None = None,
    ) -> None:
        """Start the server (blocking).

        Over HTTP/SSE the server also answers ``GET /health`` (liveness)
        and ``GET /health/ready`` (readiness from a registered
        :class:`HealthCheck`, ``503`` when a required check fails) for
        container and Kubernetes probes.

        Args:
            transport: ``"stdio"``, ``"http"``, or ``"sse"``.
            host: Bind host for HTTP/SSE transports.  Defaults to
                ``"127.0.0.1"`` (reachable from this machine only, with
                Host/Origin validation on).  Pass ``"0.0.0.0"`` to listen on
                every interface — in a container, for example — and name
                the public host in ``allowed_hosts``; without it a
                non-loopback bind does not validate ``Host`` and logs a
                warning at startup.  A non-loopback bind without
                ``AuthMiddleware`` also logs a warning: every tool is then
                callable by anyone who can reach the address.
            port: Bind port for HTTP/SSE transports.
            dashboard: Enable live terminal monitoring dashboard.
            cors: Optional ``CORSConfig`` for HTTP/SSE transports.
            allowed_hosts: ``Host`` header values the HTTP/SSE transports
                accept (``"api.example.com"`` or ``"api.example.com:*"``).
                A loopback bind validates ``Host`` and ``Origin`` against
                the loopback names by default (DNS rebinding protection)
                and this list adds to them — name the public host a
                reverse proxy forwards.  A non-loopback bind validates
                only when this is given, and then exactly these values.
            allowed_origins: ``Origin`` header values to accept, for
                browser clients served from another origin.  Requests
                without an ``Origin`` header always pass this check.
                Requires ``allowed_hosts`` on a non-loopback bind.
        """
        asyncio.run(
            self.run_async(
                transport=transport,
                host=host,
                port=port,
                dashboard=dashboard,
                cors=cors,
                allowed_hosts=allowed_hosts,
                allowed_origins=allowed_origins,
            )
        )

    async def run_async(
        self,
        transport: str = "stdio",
        *,
        host: str = "127.0.0.1",
        port: int = 8080,
        dashboard: bool = False,
        cors: Any = None,
        allowed_hosts: list[str] | None = None,
        allowed_origins: list[str] | None = None,
    ) -> None:
        """Start the server (async).

        Args:
            transport: ``"stdio"``, ``"http"``, or ``"sse"``.
            host: Bind host for HTTP/SSE transports.
            port: Bind port for HTTP/SSE transports.
            dashboard: Enable live terminal monitoring dashboard.
            cors: Optional ``CORSConfig`` for HTTP/SSE transports.
            allowed_hosts: ``Host`` header values the HTTP/SSE transports
                accept; see :meth:`run`.
            allowed_origins: ``Origin`` header values to accept; see
                :meth:`run`.

        Raises:
            ValueError: ``allowed_origins`` without ``allowed_hosts`` on a
                non-loopback bind, or an empty ``allowed_hosts`` list.
        """
        transport_type = TransportType(transport)

        # ---- Host/Origin validation policy (decided before anything binds) ----
        security_settings = None
        if transport_type != TransportType.STDIO:
            from ._transport import build_transport_security

            security_settings = build_transport_security(
                host, allowed_hosts=allowed_hosts, allowed_origins=allowed_origins
            )

        # ---- Dashboard setup (before build to include in compiled chains) ----
        dashboard_state = None
        _dashboard_obj = None

        if dashboard:
            from ._dashboard import Dashboard, DashboardMiddleware, DashboardState

            dashboard_state = DashboardState(
                server_name=self.name,
                version=self.version,
                transport=transport,
                host=host,
                port=port,
            )
            # Insert as outermost middleware (guard against double-insert)
            if not any(isinstance(m, DashboardMiddleware) for m in self._middlewares):
                self._middlewares.insert(0, DashboardMiddleware(dashboard_state))

        # ---- Build lowlevel server (compiles middleware chains) ----
        ll_server = self._build_lowlevel_server()
        init_options = ll_server.create_initialization_options()

        # ---- Populate dashboard with final registration data ----
        if dashboard_state is not None:
            for tdef in self._tool_registry.list_all():
                dashboard_state.tools.append(
                    {
                        "name": tdef.name,
                        "auth": tdef.auth,
                        "roles": list(tdef.roles),
                        "tags": list(tdef.tags),
                    }
                )
            dashboard_state.resource_count = len(list(self._resource_registry.list_all()))
            dashboard_state.prompt_count = len(list(self._prompt_registry.list_all()))
            dashboard_state.middleware_count = len(self._middlewares)
            _dashboard_obj = Dashboard(dashboard_state)

        # ---- Print banner (dashboard off, and never on stdio) ----
        # stdio uses stdout for the JSON-RPC protocol stream — any banner
        # there corrupts it, so the banner is HTTP/SSE-only.
        if not dashboard and transport != "stdio":
            self._print_banner(transport=transport, host=host, port=port)

        # ---- Warn about an unauthenticated server reachable from the network ----
        if transport_type != TransportType.STDIO:
            self._warn_unauthenticated_bind(host, port)

        # ---- Auth gate for transport-level rejection ----
        auth_gate = self._transport_auth_gate()

        # ---- Start ----
        try:
            if _dashboard_obj:
                _dashboard_obj.start()

            await run_transport(
                transport_type,
                ll_server,
                init_options,
                self._lifecycle,
                host=host,
                port=port,
                shutdown_timeout=self._shutdown_timeout,
                dashboard=dashboard_state is not None,
                auth_gate=auth_gate,
                token_endpoint=self._token_endpoint,
                cors=cors,
                security_settings=security_settings,
                health=self._health_check,
            )
        finally:
            if _dashboard_obj:
                _dashboard_obj.stop()

    def asgi_app(
        self,
        transport: str = "http",
        *,
        allowed_hosts: list[str] | None = None,
        allowed_origins: list[str] | None = None,
        cors: Any = None,
        stateless: bool = False,
    ) -> Any:
        """Return the server as an ASGI application, for uvicorn, gunicorn or hypercorn.

        Serves the same routes as :meth:`run` (``/mcp`` — or ``/sse`` and
        ``/messages/`` — plus ``/health`` and ``/health/ready``) with the
        same auth gate, token endpoint and CORS.  The app's lifespan runs
        the server's startup and shutdown hooks and the MCP session
        manager, so the ASGI server must run lifespan events (uvicorn's
        default ``--lifespan auto`` does).

        Example::

            # app.py
            app = server.asgi_app(allowed_hosts=["mcp.example.com"])

            # uvicorn app:app --host 0.0.0.0 --port 8080
            # gunicorn app:app -k uvicorn.workers.UvicornWorker -w 4

        The bind address belongs to the ASGI server, so ``Host``
        validation cannot follow it: the app always accepts the loopback
        names, plus every value in ``allowed_hosts``.  Name the public host
        your clients or proxy send, or every request through it is refused
        with ``421``.

        **Several workers or replicas.** Each process keeps its own MCP
        sessions.  A request whose ``mcp-session-id`` belongs to another
        process is answered ``404``; Promptise clients then open a new
        session, so routing that is not sticky turns into a new session on
        almost every call.  Route each ``mcp-session-id`` to one process
        (sticky sessions), or pass ``stateless=True``.

        Args:
            transport: ``"http"`` (Streamable HTTP) or ``"sse"``.
            allowed_hosts: ``Host`` header values to accept in addition to
                the loopback names, e.g. ``["mcp.example.com"]``.
            allowed_origins: ``Origin`` header values to accept for browser
                clients, in addition to the loopback origins.
            cors: Optional ``CORSConfig``.
            stateless: Streamable HTTP only.  Serve each request without a
                session, so any worker or replica can answer it.  Stateless
                servers cannot send requests back to the client: MCP
                elicitation and sampling (including elicitation approval
                gates) and per-session state are unavailable.

        Raises:
            ValueError: An unknown or stdio *transport*, ``stateless`` with
                SSE, or an empty ``allowed_hosts`` list.
        """
        from ._transport import build_http_app, build_sse_app, build_transport_security

        transport_type = TransportType(transport)
        if transport_type == TransportType.STDIO:
            raise ValueError("asgi_app() serves HTTP; use run(transport='stdio') for stdio")
        if stateless and transport_type != TransportType.HTTP:
            raise ValueError("stateless=True applies to the Streamable HTTP transport only")

        # The ASGI server owns the bind, so validate as a loopback bind does:
        # loopback names always, plus the hosts the operator names.
        security_settings = build_transport_security(
            "127.0.0.1", allowed_hosts=allowed_hosts, allowed_origins=allowed_origins
        )
        ll_server = self._build_lowlevel_server()
        common: dict[str, Any] = {
            "shutdown_timeout": self._shutdown_timeout,
            "auth_gate": self._transport_auth_gate(),
            "token_endpoint": self._token_endpoint,
            "cors": cors,
            "security_settings": security_settings,
            "health": self._health_check,
        }
        if transport_type == TransportType.HTTP:
            return build_http_app(ll_server, self._lifecycle, stateless=stateless, **common)
        return build_sse_app(
            ll_server, ll_server.create_initialization_options(), self._lifecycle, **common
        )

    def _has_auth_middleware(self) -> bool:
        """True when an ``AuthMiddleware`` is on the server or on any router's tools."""
        from ._auth import AuthMiddleware

        if any(isinstance(m, AuthMiddleware) for m in self._middlewares):
            return True
        return any(
            isinstance(m, AuthMiddleware)
            for tdef in self._tool_registry.list_all()
            for m in tdef.router_middleware
        )

    def _warn_unauthenticated_bind(self, host: str, port: int) -> None:
        """Log a warning when a non-loopback HTTP/SSE bind has no ``AuthMiddleware``.

        Such a server answers every client that can reach the address:
        anyone on the network can list and call its tools.
        """
        from ._transport import is_loopback_host

        if is_loopback_host(host) or self._has_auth_middleware():
            return
        logger.warning(
            "MCP server %r is bound to %s:%d, which is reachable from other machines, and "
            "has no AuthMiddleware: anyone who can reach that address can list and call "
            "every tool. Add AuthMiddleware (JWTAuth, APIKeyAuth, ...), front the server "
            "with an authenticating gateway, or bind host='127.0.0.1' (the default).",
            self.name,
            host,
            port,
        )

    def _transport_auth_gate(self) -> Callable[[str], bool] | None:
        """Verifier for the HTTP auth gate when ``require_auth`` is on."""
        if self._require_auth and self._auth_provider:
            if hasattr(self._auth_provider, "verify_token"):
                return cast("Callable[[str], bool]", self._auth_provider.verify_token)
        return None

    # ------------------------------------------------------------------
    # Internal: build the mcp.server.lowlevel.Server
    # ------------------------------------------------------------------

    def _build_lowlevel_server(self) -> LowLevelServer:
        """Wire our registries into an ``mcp.server.lowlevel.Server``."""
        # Register manifest (captures final state of all registrations)
        if self._auto_manifest:
            from ._manifest import register_manifest

            try:
                register_manifest(self)
            except ValueError:
                pass  # Already registered (e.g. run_async called twice)

        # Registered at build time, after the user's hooks, so it runs
        # first on shutdown: background tasks finish before resources close.
        if not self._background_drain_registered:
            self._lifecycle.add_shutdown(self._drain_background_tasks)
            self._background_drain_registered = True

        ll = _LowLevelServer(self.name, self.version, instructions=self.instructions)
        self._register_tool_handlers(ll)
        self._register_resource_handlers(ll)
        self._register_prompt_handlers(ll)
        return ll

    def _print_banner(self, *, transport: str, host: str, port: int) -> None:
        """Print the startup banner to stdout."""
        from ._banner import print_banner

        all_tools = list(self._tool_registry.list_all())
        auth_count = sum(1 for t in all_tools if t.auth)

        print_banner(
            server_name=self.name,
            version=self.version,
            transport=transport,
            host=host,
            port=port,
            tool_count=len(all_tools),
            auth_tool_count=auth_count,
            resource_count=len(list(self._resource_registry.list_all())),
            prompt_count=len(list(self._prompt_registry.list_all())),
            middleware_count=len(self._middlewares),
        )

    def _all_definitions(self) -> list[Any]:
        """Every registered tool, resource, resource template and prompt."""
        return [
            *self._tool_registry.list_all(),
            *self._resource_registry.list_all(),
            *self._resource_registry.list_templates(),
            *self._prompt_registry.list_all(),
        ]

    def _apply_require_tenant(self) -> None:
        """Enforce ``require_auth`` / ``require_tenant`` on every registration.

        ``require_auth`` forces ``auth=True``; ``require_tenant`` also
        appends a ``RequireTenant`` guard to each tool, resource and prompt
        that lacks one.  Covers every registration path (decorator, routers,
        mounts, OpenAPI import, ``include_prompts``, the manifest).
        Idempotent — called at build time, on registrations made while
        serving, and by ``TestClient``, so all execution paths enforce the
        invariant.  Guards fail closed: an unauthenticated call or a token
        without the tenant claim is denied.
        """
        if not self._require_auth:
            return
        from ._guards import RequireTenant

        for definition in self._all_definitions():
            # The definitions are frozen dataclasses; the registry holds the
            # same instance everywhere, so mutate in place via the escape hatch.
            object.__setattr__(definition, "auth", True)
            if self._require_tenant and not any(
                isinstance(g, RequireTenant) for g in definition.guards
            ):
                definition.guards.append(RequireTenant())

    # ------------------------------------------------------------------
    # list_changed notifications
    # ------------------------------------------------------------------

    async def notify_tools_changed(self) -> int:
        """Send ``notifications/tools/list_changed`` to every connected client.

        Registering a tool while the server is serving sends this
        automatically; call it yourself after changing what ``list_tools``
        returns some other way.

        Returns:
            How many sessions the notification reached.
        """
        return await self._broadcast("send_tool_list_changed")

    async def notify_resources_changed(self) -> int:
        """Send ``notifications/resources/list_changed`` to every connected client.

        Sent automatically when a resource or resource template is
        registered while the server is serving.

        Returns:
            How many sessions the notification reached.
        """
        return await self._broadcast("send_resource_list_changed")

    async def notify_prompts_changed(self) -> int:
        """Send ``notifications/prompts/list_changed`` to every connected client.

        Sent automatically when a prompt is registered while the server is
        serving.

        Returns:
            How many sessions the notification reached.
        """
        return await self._broadcast("send_prompt_list_changed")

    async def _broadcast(self, method: str) -> int:
        sent = 0
        for session in list(self._sessions):
            try:
                await getattr(session, method)()
                sent += 1
            except Exception:
                # Closed or stateless session — it will not come back.
                self._sessions.discard(session)
                logger.debug("Dropping MCP session after failed %s", method, exc_info=True)
        return sent

    def _note_session(self, ll: LowLevelServer) -> Any:
        """Remember the MCP session of the current request; return it (or None)."""
        try:
            session = ll.request_context.session
        except LookupError:
            return None
        self._sessions.add(session)
        if self._loop is None:
            self._loop = asyncio.get_running_loop()
        return session

    def _on_registry_change(self, kind: str) -> None:
        """Schedule a ``list_changed`` notification for a registration made while serving."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return  # not serving yet: clients will list after they connect
        # Re-apply server-wide auth/tenant invariants to the new registration.
        self._apply_require_tenant()
        if kind in self._pending_notifications:
            return  # coalesce a burst of registrations into one notification
        self._pending_notifications.add(kind)

        async def _send() -> None:
            self._pending_notifications.discard(kind)
            await {
                "tools": self.notify_tools_changed,
                "resources": self.notify_resources_changed,
                "prompts": self.notify_prompts_changed,
            }[kind]()

        def _schedule() -> None:
            task = loop.create_task(_send())
            self._notification_tasks.add(task)
            task.add_done_callback(self._notification_tasks.discard)

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            _schedule()
        else:
            loop.call_soon_threadsafe(_schedule)

    def _register_tool_handlers(self, ll: LowLevelServer) -> None:
        tool_reg = self._tool_registry
        input_models = self._input_models
        server_name = self.name
        middlewares = self._middlewares
        exception_handlers = self._exception_handlers
        session_manager = self._session_manager

        # Tenant invariant: apply before chains are compiled
        self._apply_require_tenant()

        # Auto-insert per-tool concurrency limiter if any tool has
        # max_concurrent set (guard against double-insert), outside any
        # circuit breaker so capacity refusals never trip it
        has_per_tool_limits = any(getattr(t, "max_concurrent", None) for t in tool_reg.list_all())
        if has_per_tool_limits:
            from ._concurrency import PerToolConcurrencyLimiter, insert_per_tool_limiter

            insert_per_tool_limiter(middlewares, PerToolConcurrencyLimiter())

        # Auto-insert declared rate-limit enforcement if any tool has
        # rate_limit set (guard against double-insert). This makes
        # @server.tool(rate_limit="100/min") enforced with no manual wiring.
        has_declared_rate_limits = any(
            getattr(d, "rate_limit", None) for d in self._all_definitions()
        )
        if has_declared_rate_limits:
            from ._rate_limit import DeclaredRateLimitMiddleware

            if not any(isinstance(m, DeclaredRateLimitMiddleware) for m in middlewares):
                middlewares.append(DeclaredRateLimitMiddleware())

        # Approval invariant: a declared requires_approval MUST be gated.
        # Unlike rate limits, a gate cannot be auto-inserted (someone has to
        # decide WHO approves), so an ungated declaration is a configuration
        # error — refuse to build rather than silently not enforcing it.
        from ._approval_gate import ApprovalGateMiddleware

        server_has_gate = any(isinstance(m, ApprovalGateMiddleware) for m in middlewares)
        # A gated tool is covered when a gate sits in the server chain OR in
        # that tool's own router_middleware (router-level gate) — both are
        # compiled into the per-tool chain below, so either enforces.
        ungated_tools = [
            t.name
            for t in tool_reg.list_all()
            if getattr(t, "requires_approval", False)
            and not server_has_gate
            and not any(
                isinstance(m, ApprovalGateMiddleware) for m in getattr(t, "router_middleware", [])
            )
        ]
        if ungated_tools:
            raise RuntimeError(
                f"Tools {ungated_tools} declare requires_approval=True but no "
                "ApprovalGateMiddleware is installed (neither server- nor "
                "router-level). Add one, e.g.: server.add_middleware("
                "ApprovalGateMiddleware(handler=PendingApprover(server))). "
                "Refusing to start a server whose declared approval gates "
                "would not be enforced."
            )

        # Pre-compile middleware chains per tool at build time so we
        # don't re-build closure chains on every request.
        _compiled_chains: dict[str, Any] = {}
        for _tdef in tool_reg.list_all():
            all_mw = list(middlewares)
            if _tdef.router_middleware:
                all_mw.extend(_tdef.router_middleware)
            _compiled_chains[_tdef.name] = compile_middleware_chain(all_mw)

        # Pre-compiled chain for tools registered after build (fallback)
        _default_chain = compile_middleware_chain(list(middlewares))

        note_session = self._note_session
        hide_unauthorized = self._hide_unauthorized_tools

        @ll.list_tools()
        async def list_tools() -> list[Tool]:
            note_session(ll)
            tdefs = tool_reg.list_all()
            if hide_unauthorized:
                from ._visibility import visible_tools

                tdefs = await visible_tools(
                    tdefs, middlewares, server_name=server_name, meta=_request_headers(ll)
                )
            tools: list[Tool] = []
            for tdef in tdefs:
                # Build MCP ToolAnnotations from our ToolAnnotations
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

        @ll.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any] | None) -> list[Any]:
            note_session(ll)
            tdef = tool_reg.get(name)
            if tdef is None:
                raise _ToolErrorResult(
                    json.dumps(
                        {
                            "error": {
                                "code": "TOOL_NOT_FOUND",
                                "message": f"Unknown tool: {name}",
                            }
                        }
                    )
                )

            arguments = arguments or {}

            # Set up request context with tool_def for middleware access.
            # Populate meta from the HTTP request that carries THIS message
            # (the SDK attaches it to every Streamable HTTP / SSE POST), so
            # that credentials, tenant, roles and X-Request-ID are resolved
            # per request — never from the request that opened the session.
            # The transport contextvars remain the fallback for stdio and
            # for direct handler invocation.
            from ._context import bind_transport_request

            try:
                mcp_request = getattr(ll.request_context, "request", None)
            except LookupError:
                mcp_request = None
            http_headers, _ = bind_transport_request(mcp_request)

            # Request tracing: honour incoming X-Request-ID header,
            # otherwise generate a random one.
            request_id = http_headers.get("x-request-id", "") or secrets.token_hex(6)

            ctx = RequestContext(
                server_name=server_name,
                tool_name=name,
                request_id=request_id,
                meta=dict(http_headers),
            )
            ctx.state["tool_def"] = tdef
            # Expose the MCP session to middleware (e.g. the approval gate's
            # elicitation approver). None when unavailable — consumers must
            # fail closed.
            try:
                ctx.state["_mcp_session"] = ll.request_context.session
            except Exception:
                ctx.state["_mcp_session"] = None
            set_context(ctx)

            di_resolver = DependencyResolver()
            try:
                # Validate input
                model = input_models.get(name)
                if model is not None:
                    arguments = validate_arguments(model, arguments)

                # Snapshot the validated user arguments for middleware that
                # needs them (approval gate) — before DI injects framework
                # objects (Elicitor, SessionState, ...) into the dict.
                ctx.state["_tool_arguments"] = dict(arguments)

                # Resolve dependency injection
                arguments = await di_resolver.resolve(tdef.handler, arguments)

                # Inject RequestContext into a ``ctx: RequestContext`` param, so
                # the documented parameter pattern works on the live transports
                # exactly as it does under TestClient (previously only the test
                # client injected it — a test/prod divergence).
                from ._context import inject_context

                arguments = inject_context(tdef.handler, arguments, ctx)

                # Detect injectable types in resolved args and bind them
                from ._background import BackgroundTasks as _BG
                from ._cancellation import CancellationToken
                from ._elicitation import Elicitor
                from ._logging import ServerLogger
                from ._progress import ProgressReporter
                from ._sampling import Sampler
                from ._session_state import SessionState

                for _val in arguments.values():
                    if isinstance(_val, _BG):
                        ctx.state["_background_tasks"] = _val
                    elif isinstance(_val, (ProgressReporter, ServerLogger)):
                        # Bind to MCP session for progress/log notifications
                        _mcp_session = None
                        _progress_token = None
                        _mcp_request_id = None
                        try:
                            mcp_ctx = ll.request_context
                            _mcp_session = mcp_ctx.session
                            _mcp_request_id = getattr(mcp_ctx, "request_id", None)
                            if hasattr(mcp_ctx, "meta") and mcp_ctx.meta:
                                _progress_token = getattr(mcp_ctx.meta, "progressToken", None)
                        except Exception:
                            logger.debug(
                                "Error extracting MCP request context for progress/logger binding",
                                exc_info=True,
                            )
                        if isinstance(_val, ProgressReporter):
                            _val._bind(_mcp_session, _progress_token, _mcp_request_id)
                            ctx.state["_progress_reporter"] = _val
                        else:
                            _val._bind(_mcp_session, _mcp_request_id)
                            ctx.state["_server_logger"] = _val
                    elif isinstance(_val, CancellationToken):
                        ctx.state["_cancellation_token"] = _val
                    elif isinstance(_val, (Elicitor, Sampler)):
                        # Bind to MCP session for elicitation/sampling
                        _mcp_session = None
                        _mcp_request_id = None
                        try:
                            mcp_ctx = ll.request_context
                            _mcp_session = mcp_ctx.session
                            _mcp_request_id = getattr(mcp_ctx, "request_id", None)
                        except Exception:
                            logger.debug(
                                "Error extracting MCP request context for elicitor/sampler binding",
                                exc_info=True,
                            )
                        _val._bind(_mcp_session, _mcp_request_id)
                    elif isinstance(_val, SessionState):
                        # Populate from SessionManager using MCP session ID
                        _session_id = ctx.request_id  # fallback
                        try:
                            mcp_ctx = ll.request_context
                            if hasattr(mcp_ctx, "session"):
                                _session_id = str(id(mcp_ctx.session))
                        except Exception:
                            logger.debug(
                                "Error extracting MCP session ID for session state", exc_info=True
                            )
                        managed = session_manager.get_or_create(_session_id)
                        _val._data = managed._data
                        ctx.state["_session_state"] = _val

                # Wrap handler with guard checks (guards run after
                # middleware so auth middleware can populate roles first)
                effective_handler = tdef.handler
                if tdef.guards:
                    from ._testing import check_guards

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

                if tdef.timeout:
                    effective_handler = with_tool_timeout(effective_handler, tdef.timeout, name)

                # Use pre-compiled middleware chain (avoids per-request
                # closure construction)
                chain_fn = _compiled_chains.get(name, _default_chain)
                cancel_token = ctx.state.get("_cancellation_token")
                if cancel_token is None:
                    result = await chain_fn(ctx, effective_handler, arguments)
                else:
                    result = await _run_cancellable(
                        chain_fn(ctx, effective_handler, arguments),
                        cancel_token,
                        grace_period=self._cancel_grace_period,
                    )

                # Serialise result
                serialised = _serialise_result(result)

                # Background tasks run after the response is sent — never
                # while the client waits (errors are logged, not returned).
                bg = ctx.state.get("_background_tasks")
                if bg is not None and bg.pending:
                    self._run_background(bg)

                return serialised

            except MCPError as exc:
                # A handler registered for this MCPError subclass may reshape it
                mapped = await exception_handlers.handle(ctx, exc)
                raise _ToolErrorResult((mapped or exc).to_text()) from None
            except Exception as exc:
                # Try custom exception handlers first
                mapped = await exception_handlers.handle(ctx, exc)
                if mapped is not None:
                    raise _ToolErrorResult(mapped.to_text()) from None

                logger.exception("Unhandled error in tool '%s'", name)
                # Return a generic message to clients — full details
                # are in the server log above. Never leak internal
                # exception strings (may contain DB URLs, file paths, etc.).
                err_text = json.dumps(
                    {
                        "error": {
                            "code": "INTERNAL_ERROR",
                            "message": "An internal error occurred.",
                            "retryable": False,
                        }
                    }
                )
                raise _ToolErrorResult(err_text) from None
            finally:
                await di_resolver.cleanup()
                clear_context()

    def _run_background(self, bg: Any) -> None:
        """Run a request's ``BackgroundTasks`` in their own task.

        The task inherits the request's context (``get_context()`` still
        works in it) and starts once the handler has returned, so the
        response is not held back.
        """

        async def _after_response() -> None:
            await asyncio.sleep(0)  # let the response go out first
            await bg.execute()

        task = asyncio.create_task(_after_response(), name=f"{self.name}-background-tasks")
        self._background_runs.add(task)
        task.add_done_callback(self._background_runs.discard)

    async def _drain_background_tasks(self) -> None:
        """Shutdown hook: wait for background tasks still running."""
        if self._background_runs:
            logger.info("Waiting for %d background task run(s)", len(self._background_runs))
            await asyncio.gather(*self._background_runs, return_exceptions=True)

    async def _visible(self, ll: LowLevelServer, definitions: list[Any], kind: str) -> list[Any]:
        """*definitions* filtered for the caller with ``hide_unauthorized_tools``."""
        if not self._hide_unauthorized_tools:
            return definitions
        from ._visibility import visible_tools

        return await visible_tools(
            definitions,
            self._middlewares,
            server_name=self.name,
            meta=_request_headers(ll),
            request_type=kind,
        )

    def _register_resource_handlers(self, ll: LowLevelServer) -> None:
        res_reg = self._resource_registry
        note_session = self._note_session
        visible = self._visible

        @ll.list_resources()
        async def list_resources() -> list[Resource]:
            note_session(ll)
            resources: list[Resource] = []
            for rdef in await visible(ll, res_reg.list_all(), "resource"):
                resources.append(
                    Resource(
                        uri=cast("AnyUrl", rdef.uri),
                        name=rdef.name,
                        description=rdef.description,
                        mimeType=rdef.mime_type,
                    )
                )
            return resources

        @ll.list_resource_templates()
        async def list_resource_templates() -> list[ResourceTemplate]:
            note_session(ll)
            templates: list[ResourceTemplate] = []
            for rdef in await visible(ll, res_reg.list_templates(), "resource"):
                templates.append(
                    ResourceTemplate(
                        uriTemplate=rdef.uri,
                        name=rdef.name,
                        description=rdef.description,
                        mimeType=rdef.mime_type,
                    )
                )
            return templates

        @ll.read_resource()
        async def read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
            from . import _dispatch

            session = note_session(ll)
            headers = _request_headers(ll)
            try:
                return await _dispatch.read_resource(
                    self, str(uri), meta=headers, mcp_session=session
                )
            except MCPError as exc:
                raise _dispatch.to_protocol_error(exc) from exc
            except Exception as exc:
                logger.exception("Unhandled error reading resource '%s'", uri)
                raise _dispatch.internal_protocol_error() from exc

    def _register_prompt_handlers(self, ll: LowLevelServer) -> None:
        prompt_reg = self._prompt_registry
        note_session = self._note_session
        visible = self._visible

        @ll.list_prompts()
        async def list_prompts() -> list[Any]:
            from mcp.types import Prompt as MCPPrompt

            note_session(ll)
            prompts: list[MCPPrompt] = []
            for pdef in await visible(ll, prompt_reg.list_all(), "prompt"):
                args = [
                    PromptArgument(
                        name=a["name"],
                        description=a.get("description"),
                        required=a.get("required", True),
                    )
                    for a in pdef.arguments
                ]
                prompts.append(
                    MCPPrompt(
                        name=pdef.name,
                        description=pdef.description,
                        arguments=args,
                    )
                )
            return prompts

        @ll.get_prompt()
        async def get_prompt(name: str, arguments: dict[str, str] | None) -> GetPromptResult:
            from . import _dispatch

            session = note_session(ll)
            headers = _request_headers(ll)
            try:
                return await _dispatch.get_prompt(
                    self, name, arguments, meta=headers, mcp_session=session
                )
            except MCPError as exc:
                raise _dispatch.to_protocol_error(exc) from exc
            except Exception as exc:
                logger.exception("Unhandled error in prompt '%s'", name)
                raise _dispatch.internal_protocol_error() from exc


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _serialise_result(result: Any) -> list[TextContent | MCPImageContent | MCPEmbeddedResource]:
    """Convert a handler return value to MCP content list.

    Supports:
    - ``str`` → ``TextContent``
    - ``dict`` / ``list`` → JSON-serialised ``TextContent``
    - ``ImageContent`` → ``MCPImageContent`` (base64-encoded)
    - ``MCPImageContent`` / ``MCPEmbeddedResource`` / ``TextContent`` → pass-through
    - ``list`` of the above → mixed content
    - ``None`` → ``TextContent("OK")``
    """
    from ._context import ToolResponse
    from ._streaming import StreamingResult
    from ._types import ImageContent

    # ToolResponse → unwrap content, store metadata on ctx for audit/middleware
    if isinstance(result, ToolResponse):
        ctx = _current_context.get() if _current_context else None
        if ctx is not None:
            ctx.state["response_metadata"] = result.metadata
        return _serialise_result(result.content)

    # StreamingResult → serialize as JSON dict
    if isinstance(result, StreamingResult):
        return [TextContent(type="text", text=json.dumps(result.to_dict(), default=str))]

    # Pass-through for native MCP content types
    if isinstance(result, (TextContent, MCPImageContent, MCPEmbeddedResource)):
        return [result]

    # Our ImageContent helper → MCP ImageContent
    if isinstance(result, ImageContent):
        return [result.to_mcp()]

    # Mixed content list (e.g. [TextContent(...), ImageContent(...)])
    if isinstance(result, list):
        # Check if it's a list of content items (not a plain data list)
        if result and _is_content_list(result):
            items: list[TextContent | MCPImageContent | MCPEmbeddedResource] = []
            for item in result:
                if isinstance(item, ImageContent):
                    items.append(item.to_mcp())
                elif isinstance(item, (TextContent, MCPImageContent, MCPEmbeddedResource)):
                    items.append(item)
                else:
                    items.append(TextContent(type="text", text=str(item)))
            return items
        # Plain data list → JSON
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    if isinstance(result, dict):
        return [TextContent(type="text", text=json.dumps(result, default=str))]
    if result is None:
        return [TextContent(type="text", text="OK")]
    return [TextContent(type="text", text=str(result))]


def _request_headers(ll: LowLevelServer) -> dict[str, str]:
    """Headers of the HTTP request carrying the current MCP message (``{}`` on stdio).

    See :func:`~._context.bind_transport_request` — per request, never the
    request that opened the session.
    """
    from ._context import bind_transport_request

    try:
        mcp_request = getattr(ll.request_context, "request", None)
    except LookupError:
        mcp_request = None
    headers, _ = bind_transport_request(mcp_request)
    return dict(headers)


def _is_content_list(items: list[Any]) -> bool:
    """Check whether a list contains MCP content items (not plain data)."""
    from ._types import ImageContent

    _content_types = (TextContent, MCPImageContent, MCPEmbeddedResource, ImageContent)
    return any(isinstance(item, _content_types) for item in items)


def _excluded_params_for(func: Callable[..., Any]) -> set[str]:
    """Identify params excluded from input schema."""
    from ._decorators import _excluded_params

    return _excluded_params(func)


def _prompt_to_mcp_def(p: Any, *, version: str | None = None) -> PromptDef:
    """Convert a Promptise :class:`Prompt` to an MCP :class:`PromptDef`.

    - **Description** — the prompt's ``description`` (``@prompt(description=...)``
      or a YAML file's ``description``), else the first line of its template,
      followed by its metadata (``[v1.0.0, model: ..., strategy: ...]``).
    - **Arguments** — one per template parameter.  Each description comes
      from the YAML file's ``arguments``, an ``Annotated[..., Field(description=...)]``
      hint, or else names the placeholder, type and default
      (``"Fills {max_words} (int, default 50)."``).
    - **Handler** — coerces the string-valued MCP arguments to the
      parameters' type hints, then renders with ``render_async(**kwargs)``
      (context providers, strategy, perspective and constraints applied).

    Args:
        p: A :class:`~promptise.prompts.core.Prompt` instance.
        version: Optional version string (from registry).

    Returns:
        A :class:`PromptDef` ready for MCP registration.
    """
    from typing import get_type_hints

    from ._types import PromptDef

    description = getattr(p, "description", None) or ""
    if not description:
        first_line = p.template.strip().split("\n")[0].strip() if p.template else ""
        description = first_line or p.name

    meta: list[str] = []
    if version:
        meta.append(f"v{version}")
    meta.append(f"model: {p.model}")
    if p._strategy:
        meta.append(f"strategy: {p._strategy!r}")
    if p._perspective:
        meta.append(f"perspective: {p._perspective!r}")
    if p._constraints:
        meta.append(f"constraints: {len(p._constraints)}")
    description = f"{description} [{', '.join(meta)}]"

    # A model over the prompt's own signature (a YAML file may replace it)
    # with the original function's type hints.
    try:
        hints = get_type_hints(p._fn, include_extras=True)
    except Exception:
        hints = {}
    params = {
        name: param
        for name, param in p._sig.parameters.items()
        if param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }

    def _shape() -> None: ...

    _shape.__name__ = p.name
    _shape.__signature__ = inspect.Signature(list(params.values()))  # type: ignore[attr-defined]
    _shape.__annotations__ = {n: hints[n] for n in params if n in hints}
    input_model, schema = build_input_model(_shape)
    properties = schema.get("properties", {})

    stored_schemas: dict[str, Any] = getattr(p, "_argument_schemas", {}) or {}
    arguments: list[dict[str, Any]] = []
    for param_name, param in params.items():
        stored = stored_schemas.get(param_name)
        desc = (getattr(stored, "description", "") if stored is not None else "") or properties.get(
            param_name, {}
        ).get("description")
        if not desc:
            desc = f"Fills {{{param_name}}} in the prompt"
            hint = hints.get(param_name)
            details = []
            if hint is not None:
                details.append(
                    hint.__name__ if isinstance(hint, type) else str(hint).replace("typing.", "")
                )
            if param.default is not inspect.Parameter.empty and param.default != "":
                details.append(f"default {param.default!r}")
            if details:
                desc += f" ({', '.join(details)})"
            desc += "."
        arguments.append(
            {
                "name": param_name,
                "description": desc,
                "required": param.default is inspect.Parameter.empty,
            }
        )

    prompt_ref = p  # capture for closure

    async def handler(**kwargs: Any) -> str:
        return await prompt_ref.render_async(**kwargs)

    return PromptDef(
        name=p.name,
        description=description,
        handler=handler,
        arguments=arguments,
        input_model=input_model,
    )


async def _run_cancellable(work: Any, token: Any, *, grace_period: float) -> Any:
    """Await a tool call so that MCP cancellation reaches its ``CancellationToken``.

    When the client sends ``notifications/cancelled``, the MCP SDK cancels
    the request's task.  The call runs in a child task instead, so on that
    cancellation the token is set first and the handler gets
    *grace_period* seconds to stop on its own (``cancel.check()``,
    ``cancel.wait()``) before its task is cancelled too.  The cancellation
    is then re-raised: the SDK has already answered the client.
    """
    import anyio

    task = asyncio.ensure_future(work)
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        token.cancel(reason="Request cancelled by the client")
        # Shielded: the request's cancel scope would otherwise interrupt
        # the grace period immediately.
        with anyio.CancelScope(shield=True):
            done, _ = await asyncio.wait({task}, timeout=grace_period)
            if not done:
                task.cancel()
                await asyncio.wait({task})
        if not task.cancelled() and task.exception() is not None:
            logger.debug("Cancelled tool call ended with %r", task.exception())
        raise
