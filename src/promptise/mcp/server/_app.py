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
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pydantic import AnyUrl

    from ._types import PromptDef

from mcp.server.lowlevel import Server as LowLevelServer
from mcp.types import (
    EmbeddedResource as MCPEmbeddedResource,
)
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
from ._middleware import compile_middleware_chain
from ._registry import PromptRegistry, ResourceRegistry, ToolRegistry
from ._transport import TransportType, run_transport
from ._validation import build_input_model, validate_arguments

logger = logging.getLogger("promptise.server")


class MCPServer:
    """Production-grade MCP server with decorator-based tool registration.

    Args:
        name: Server name advertised to MCP clients.
        version: Server version string.
        instructions: Optional instructions sent to clients on initialisation.
        require_auth: Force ``auth=True`` on every registered tool.
        require_tenant: Make tenant identity a server-wide invariant: every
            tool authenticates and carries a ``RequireTenant`` guard, so a
            client whose token lacks the tenant claim is denied on every
            call.  Implies ``require_auth``.
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
    ) -> None:
        self.name = name
        self.version = version
        self.instructions = instructions
        self._shutdown_timeout = shutdown_timeout
        self._require_auth = require_auth or require_tenant
        self._require_tenant = require_tenant

        self._tool_registry = ToolRegistry()
        self._resource_registry = ResourceRegistry()
        self._prompt_registry = PromptRegistry()
        self._lifecycle = LifecycleManager()

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
        mime_type: str = "text/plain",
    ) -> Callable[..., Any]:
        """Register a function as an MCP resource.

        Args:
            uri: Static resource URI (e.g. ``"config://app"``).
            name: Resource name (defaults to function name).
            description: Description (defaults to docstring).
            mime_type: MIME type of the resource content.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            res_def = build_resource_def(
                func,
                uri=uri,
                name=name,
                description=description,
                mime_type=mime_type,
                is_template=False,
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
        mime_type: str = "text/plain",
    ) -> Callable[..., Any]:
        """Register a function as an MCP resource template.

        Args:
            uri_template: URI template with ``{param}`` placeholders.
            name: Resource name.
            description: Description.
            mime_type: MIME type.
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            res_def = build_resource_def(
                func,
                uri=uri_template,
                name=name,
                description=description,
                mime_type=mime_type,
                is_template=True,
            )
            self._resource_registry.register(res_def)
            return func

        return decorator

    def prompt(
        self,
        name: str | None = None,
        *,
        description: str | None = None,
    ) -> Callable[..., Any]:
        """Register a function as an MCP prompt.

        Args:
            name: Prompt name (defaults to function name).
            description: Description (defaults to docstring).
        """

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            prompt_def = build_prompt_def(func, name=name, description=description)
            self._prompt_registry.register(prompt_def)
            return func

        return decorator

    # ------------------------------------------------------------------
    # Middleware
    # ------------------------------------------------------------------

    def add_middleware(self, middleware: Any) -> None:
        """Add a middleware to the processing chain."""
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
        extracted from the prompt's function signature.  The MCP handler
        calls ``render_async(**arguments)`` to produce fully rendered
        prompt text with context providers, strategy, perspective, and
        constraints applied.

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

        ll = LowLevelServer(self.name, self.version, instructions=self.instructions)
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

    def _apply_require_tenant(self) -> None:
        """Enforce ``MCPServer(require_tenant=True)`` on every registered tool.

        Forces ``auth=True`` and appends a ``RequireTenant`` guard to each
        tool that lacks one, covering every registration path (decorator,
        routers, mounts, OpenAPI import).  Idempotent — called at build
        time and by ``TestClient`` so all execution paths enforce the
        tenant invariant.  Guards fail closed: an unauthenticated call or
        a token without the tenant claim is denied.
        """
        if not self._require_tenant:
            return
        from ._guards import RequireTenant

        for tdef in self._tool_registry.list_all():
            # ToolDef is a frozen dataclass; the registry holds the same
            # instance everywhere, so mutate in place via the escape hatch.
            object.__setattr__(tdef, "auth", True)
            if not any(isinstance(g, RequireTenant) for g in tdef.guards):
                tdef.guards.append(RequireTenant())

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
        # max_concurrent set (guard against double-insert)
        has_per_tool_limits = any(getattr(t, "max_concurrent", None) for t in tool_reg.list_all())
        if has_per_tool_limits:
            from ._concurrency import PerToolConcurrencyLimiter

            if not any(isinstance(m, PerToolConcurrencyLimiter) for m in middlewares):
                middlewares.append(PerToolConcurrencyLimiter())

        # Auto-insert declared rate-limit enforcement if any tool has
        # rate_limit set (guard against double-insert). This makes
        # @server.tool(rate_limit="100/min") enforced with no manual wiring.
        has_declared_rate_limits = any(getattr(t, "rate_limit", None) for t in tool_reg.list_all())
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

        @ll.list_tools()
        async def list_tools() -> list[Tool]:
            tools: list[Tool] = []
            for tdef in tool_reg.list_all():
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
            tdef = tool_reg.get(name)
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

                # Use pre-compiled middleware chain (avoids per-request
                # closure construction)
                chain_fn = _compiled_chains.get(name, _default_chain)
                result = await chain_fn(ctx, effective_handler, arguments)

                # Serialise result
                serialised = _serialise_result(result)

                # Run background tasks (fire-and-forget, errors logged)
                bg = ctx.state.get("_background_tasks")
                if bg is not None:
                    await bg.execute()

                return serialised

            except MCPError as exc:
                return [TextContent(type="text", text=exc.to_text())]
            except Exception as exc:
                # Try custom exception handlers first
                mapped = await exception_handlers.handle(ctx, exc)
                if mapped is not None:
                    return [TextContent(type="text", text=mapped.to_text())]

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
                return [TextContent(type="text", text=err_text)]
            finally:
                await di_resolver.cleanup()
                clear_context()

    def _register_resource_handlers(self, ll: LowLevelServer) -> None:
        res_reg = self._resource_registry
        server_name = self.name

        @ll.list_resources()
        async def list_resources() -> list[Resource]:
            resources: list[Resource] = []
            for rdef in res_reg.list_all():
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
            templates: list[ResourceTemplate] = []
            for rdef in res_reg.list_templates():
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
        async def read_resource(uri: str) -> str:
            # Try static resource first
            rdef = res_reg.get(str(uri))
            if rdef is not None:
                ctx = RequestContext(server_name=server_name, tool_name=rdef.name)
                set_context(ctx)
                try:
                    result = rdef.handler()
                    if asyncio.iscoroutine(result):
                        result = await result
                    return str(result)
                finally:
                    clear_context()

            # Try template match
            match = res_reg.match_template(str(uri))
            if match is not None:
                tmpl_def, params = match
                ctx = RequestContext(server_name=server_name, tool_name=tmpl_def.name)
                set_context(ctx)
                try:
                    result = tmpl_def.handler(**params)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return str(result)
                finally:
                    clear_context()

            raise ValueError(f"Resource not found: {uri}")

    def _register_prompt_handlers(self, ll: LowLevelServer) -> None:
        prompt_reg = self._prompt_registry
        server_name = self.name

        @ll.list_prompts()
        async def list_prompts() -> list[Any]:
            from mcp.types import Prompt as MCPPrompt

            prompts: list[MCPPrompt] = []
            for pdef in prompt_reg.list_all():
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
        async def get_prompt(name: str, arguments: dict[str, str] | None) -> Any:
            pdef = prompt_reg.get(name)
            if pdef is None:
                raise ValueError(f"Prompt not found: {name}")

            ctx = RequestContext(server_name=server_name, tool_name=name)
            set_context(ctx)
            try:
                result = pdef.handler(**(arguments or {}))
                if asyncio.iscoroutine(result):
                    result = await result

                # Return as GetPromptResult
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

    Builds a rich description from prompt metadata (template, model,
    version, strategy, perspective, constraints) and creates a handler
    that calls ``render_async(**kwargs)`` to produce fully rendered text.

    Args:
        p: A :class:`~promptise.prompts.core.Prompt` instance.
        version: Optional version string (from registry).

    Returns:
        A :class:`PromptDef` ready for MCP registration.
    """
    from ._types import PromptDef

    # Build description with metadata
    desc_parts: list[str] = []
    if p.template:
        first_line = p.template.split("\n")[0].strip()
        desc_parts.append(first_line)
    else:
        desc_parts.append(p.name)

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
    if meta:
        desc_parts.append(f"[{', '.join(meta)}]")
    description = " ".join(desc_parts)

    # Build arguments from signature
    from ._decorators import _extract_param_doc

    docstring = p.template or ""
    arguments: list[dict[str, Any]] = []
    for param_name, param in p._sig.parameters.items():
        desc = _extract_param_doc(docstring, param_name) or param_name
        arg: dict[str, Any] = {
            "name": param_name,
            "description": desc,
            "required": param.default is inspect.Parameter.empty,
        }
        arguments.append(arg)

    # Build handler — renders the prompt via render_async
    prompt_ref = p  # capture for closure

    async def handler(**kwargs: Any) -> str:
        return await prompt_ref.render_async(**kwargs)

    return PromptDef(
        name=p.name,
        description=description,
        handler=handler,
        arguments=arguments,
    )
