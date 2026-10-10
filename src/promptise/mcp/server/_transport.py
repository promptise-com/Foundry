"""Transport adapters: stdio, Streamable HTTP, SSE.

Each adapter creates the appropriate read/write streams and calls
``lowlevel_server.run()`` with them.
"""

from __future__ import annotations

import contextlib
import hashlib
import ipaddress
import json as _json
import logging
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from mcp.server.lowlevel import Server as LowLevelServer
from mcp.server.transport_security import TransportSecuritySettings

from ._types import TransportType

if TYPE_CHECKING:
    from ._lifecycle import LifecycleManager

logger = logging.getLogger("promptise.server")


# =====================================================================
# Host / Origin validation (DNS rebinding protection)
# =====================================================================

LOOPBACK_ALLOWED_HOSTS: tuple[str, ...] = (
    "127.0.0.1",
    "127.0.0.1:*",
    "localhost",
    "localhost:*",
    "[::1]",
    "[::1]:*",
)
"""``Host`` header values a loopback-bound server accepts (any port)."""

LOOPBACK_ALLOWED_ORIGINS: tuple[str, ...] = (
    "http://127.0.0.1",
    "http://127.0.0.1:*",
    "http://localhost",
    "http://localhost:*",
    "http://[::1]",
    "http://[::1]:*",
    "https://127.0.0.1",
    "https://127.0.0.1:*",
    "https://localhost",
    "https://localhost:*",
    "https://[::1]",
    "https://[::1]:*",
)
"""``Origin`` header values a loopback-bound server accepts (any port)."""


def is_loopback_host(host: str) -> bool:
    """True when *host* is a loopback bind address (``127.0.0.0/8``, ``::1``, ``localhost``).

    Args:
        host: Bind host as passed to ``MCPServer.run`` (an IPv6 literal may
            be bracketed).
    """
    if host in ("localhost", "ip6-localhost"):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _host_pattern(host: str) -> str:
    """``Host`` header base for *host* (IPv6 literals are bracketed)."""
    bare = host.strip("[]")
    return f"[{bare}]" if ":" in bare else bare


def build_transport_security(
    host: str,
    *,
    allowed_hosts: list[str] | None = None,
    allowed_origins: list[str] | None = None,
) -> TransportSecuritySettings | None:
    """Decide the Host/Origin validation policy for an HTTP or SSE bind.

    A loopback bind is protected against DNS rebinding by default: a page in
    the operator's browser whose hostname was rebound to ``127.0.0.1`` sends
    the attacker's ``Host``/``Origin``, and both are refused (``421`` /
    ``403``) before any MCP message is processed.  The loopback names are
    accepted on any port (plus the bind address itself when it is another
    loopback address) — a request that names the bind address is by
    definition not rebound, so an explicit list *adds* the names a reverse
    proxy forwards (``Host: api.example.com``) without taking the loopback
    names away from local clients and health checks.

    A non-loopback bind is not restricted unless the operator names the
    hosts it serves: the framework cannot know the public hostname, and a
    wrong guess would refuse every request.  Pass ``allowed_hosts`` (and
    ``allowed_origins`` for browser clients on another origin) to enable the
    validation there, or terminate at a gateway that validates ``Host`` and
    ``Origin``.  On such a bind the lists are used exactly as given.

    Args:
        host: Bind host.
        allowed_hosts: ``Host`` values to accept, e.g. ``["api.example.com"]``
            or ``["api.example.com:*"]`` (any port).
        allowed_origins: ``Origin`` values to accept, e.g.
            ``["https://app.example.com"]``.  A request without an ``Origin``
            header (non-browser MCP clients) always passes this check.

    Returns:
        The settings to hand to the MCP SDK transports, or ``None`` when
        validation stays off (non-loopback bind without an explicit list).

    Raises:
        ValueError: ``allowed_origins`` without ``allowed_hosts`` on a
            non-loopback bind — the SDK validates ``Host`` whenever the
            protection is on, and an empty host list would refuse every
            request.
    """
    if allowed_hosts is not None and not allowed_hosts:
        raise ValueError("allowed_hosts must name at least one Host value or be None")
    if is_loopback_host(host):
        base = _host_pattern(host)
        hosts = list(LOOPBACK_ALLOWED_HOSTS)
        origins = list(LOOPBACK_ALLOWED_ORIGINS)
        for extra in (base, f"{base}:*", *(allowed_hosts or ())):
            if extra not in hosts:
                hosts.append(extra)
        own_origins = [
            f"{scheme}://{base}{suffix}" for scheme in ("http", "https") for suffix in ("", ":*")
        ]
        for extra in (*own_origins, *(allowed_origins or ())):
            if extra not in origins:
                origins.append(extra)
        return TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=hosts,
            allowed_origins=origins,
        )
    if allowed_hosts is None:
        if allowed_origins is not None:
            raise ValueError(
                "allowed_origins requires allowed_hosts on a non-loopback bind: Host validation "
                "cannot be skipped once the protection is on, and no Host value would be accepted."
            )
        return None
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(allowed_hosts),
        allowed_origins=list(allowed_origins or []),
    )


def _log_transport_security(security: TransportSecuritySettings | None, host: str) -> None:
    if security is None:
        logger.info(
            "Host/Origin validation off for non-loopback bind %s: front the server with a "
            "gateway that validates Host and Origin, or pass allowed_hosts/allowed_origins",
            host,
        )
        return
    logger.info(
        "Host/Origin validation on: hosts=%s origins=%s",
        security.allowed_hosts,
        security.allowed_origins,
    )


# =====================================================================
# CORS configuration
# =====================================================================


@dataclass(frozen=True)
class CORSConfig:
    """CORS configuration for HTTP and SSE transports.

    Args:
        allow_origins: Allowed origin URLs. Use ``["*"]`` to allow all.
        allow_methods: Allowed HTTP methods.
        allow_headers: Allowed request headers.
        allow_credentials: Whether to allow credentials (cookies, auth).
        max_age: Max seconds browsers may cache preflight responses.

    Example::

        server.run(
            transport="http",
            port=8080,
            cors=CORSConfig(
                allow_origins=["https://app.example.com"],
                allow_headers=["Authorization", "x-api-key"],
            ),
        )
    """

    allow_origins: list[str] = field(default_factory=list)
    allow_methods: list[str] = field(default_factory=lambda: ["GET", "POST", "DELETE", "OPTIONS"])
    allow_headers: list[str] = field(
        default_factory=lambda: ["Content-Type", "Authorization", "x-api-key"]
    )
    allow_credentials: bool = False
    max_age: int = 600


# =====================================================================
# Transport-level auth gate (ASGI middleware)
# =====================================================================


def session_principal(scheme: str, credential: str) -> Any:
    """The transport-level principal a verified *credential* represents.

    Returned as the MCP SDK's ``AuthenticatedUser`` and stored on
    ``scope["user"]`` so the SDK's Streamable HTTP and SSE transports bind
    every session to the credential that opened it: a request that presents
    a different credential for an existing session is answered ``404`` as
    if the session did not exist, so a leaked ``mcp-session-id`` cannot be
    ridden by another caller.  The principal is a fingerprint of the
    credential bytes (never the credential itself) prefixed with its
    scheme, so a bearer token and an API key can never collide.

    Args:
        scheme: ``"bearer"`` or ``"api-key"``.
        credential: The verified token or key.
    """
    from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
    from mcp.server.auth.provider import AccessToken

    fingerprint = hashlib.sha256(credential.encode("utf-8")).hexdigest()
    return AuthenticatedUser(
        AccessToken(token=credential, client_id=f"{scheme}:{fingerprint}", scopes=[])
    )


class _AuthGateASGI:
    """ASGI middleware that rejects HTTP requests without valid auth.

    Wraps the Starlette application and checks the ``Authorization``
    or ``x-api-key`` header before forwarding to the MCP session manager.
    Only applied when ``require_auth=True`` on the server.

    Supports two auth methods (checked in order):

    1. **Bearer token** (JWT): ``Authorization: Bearer <token>``
    2. **API key**: ``x-api-key: <key>``

    A request that passes is tagged with the principal of its credential
    (see :func:`session_principal`), which the MCP SDK transports compare
    against the principal that created the session the request addresses.

    Args:
        app: The inner ASGI application.
        verify_fn: Callable ``(token) → bool`` — the primary verifier
            (typically ``JWTAuth.verify_token``).
        skip_paths: Set of paths that bypass authentication (e.g. the
            token endpoint, which *issues* tokens and so cannot require
            one).
        api_key_verify_fn: Optional callable ``(key) → bool`` for API
            key verification.  When not provided but an ``x-api-key``
            header is present, falls back to ``verify_fn``.
    """

    def __init__(
        self,
        app: Any,
        verify_fn: Callable[[str], bool],
        skip_paths: set[str] | None = None,
        api_key_verify_fn: Callable[[str], bool] | None = None,
    ) -> None:
        self.app = app
        self._verify = verify_fn
        self._skip_paths = skip_paths or set()
        self._verify_api_key = api_key_verify_fn

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        # Let lifespan events through unconditionally
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return

        if scope["type"] in ("http", "websocket"):
            # Skip auth for whitelisted paths (e.g. token endpoint)
            path = scope.get("path", "")
            if path in self._skip_paths:
                await self.app(scope, receive, send)
                return

            headers = dict(scope.get("headers", []))

            # Try Bearer token first (Authorization: Bearer <token>)
            auth_value = headers.get(b"authorization", b"").decode("latin-1")
            if auth_value and auth_value.startswith("Bearer "):
                token = auth_value[7:]
                if self._verify(token):
                    scope["user"] = session_principal("bearer", token)
                    await self.app(scope, receive, send)
                    return
                await _send_json(
                    send,
                    401,
                    {
                        "error": "Invalid authentication token",
                        "message": "The provided token could not be verified.",
                    },
                )
                return

            # Try API key (x-api-key header)
            api_key = headers.get(b"x-api-key", b"").decode("latin-1")
            if api_key:
                verify_key = self._verify_api_key or self._verify
                if verify_key(api_key):
                    scope["user"] = session_principal("api-key", api_key)
                    await self.app(scope, receive, send)
                    return
                await _send_json(
                    send,
                    401,
                    {
                        "error": "Invalid API key",
                        "message": "The provided API key could not be verified.",
                    },
                )
                return

            # No credentials at all
            await _send_json(
                send,
                401,
                {
                    "error": "Authentication required",
                    "message": "This server requires authentication. "
                    "Pass a Bearer token via the Authorization header "
                    "or an API key via the x-api-key header.",
                },
            )
            return

        await self.app(scope, receive, send)


async def _send_json(send: Any, status: int, body: dict[str, Any]) -> None:
    """Send a JSON HTTP response via raw ASGI."""
    payload = _json.dumps(body).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                [b"content-type", b"application/json"],
                [b"content-length", str(len(payload)).encode()],
            ],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": payload,
        }
    )


# =====================================================================
# Stdio transport
# =====================================================================


async def run_stdio(
    server: LowLevelServer,
    init_options: Any,
    lifecycle: LifecycleManager,
    *,
    shutdown_timeout: float | None = None,
) -> None:
    """Run the server over stdio (stdin/stdout).

    This is the default transport for local MCP connections.
    """
    from mcp.server.stdio import stdio_server

    await lifecycle.startup()
    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("MCP server running on stdio")
            await server.run(
                read_stream,
                write_stream,
                init_options,
                raise_exceptions=False,
            )
    finally:
        await lifecycle.shutdown(timeout=shutdown_timeout)


# =====================================================================
# Streamable HTTP transport
# =====================================================================


async def run_http(
    server: LowLevelServer,
    init_options: Any,
    lifecycle: LifecycleManager,
    *,
    host: str = "0.0.0.0",  # nosec B104 - public bind is explicit opt-in for server transports
    port: int = 8080,
    shutdown_timeout: float | None = None,
    dashboard: bool = False,
    auth_gate: Callable[[str], bool] | None = None,
    token_endpoint: Any = None,
    cors: CORSConfig | None = None,
    security_settings: TransportSecuritySettings | None = None,
) -> None:
    """Run the server over Streamable HTTP.

    Uses the MCP SDK's ``StreamableHTTPSessionManager`` which handles
    session tracking, transport creation, and ``connect()`` lifecycle
    automatically.  Served via Starlette + uvicorn.

    Args:
        dashboard: When True, suppress uvicorn access logs (the live
            dashboard captures request data via middleware instead).
        auth_gate: Optional callable ``(token_or_key) → bool`` for
            transport-level authentication.  Rejects HTTP requests that
            lack a valid ``Authorization: Bearer <token>`` header or
            ``x-api-key`` header.  Bearer tokens are checked first; if
            absent, the ``x-api-key`` header is tried.
        security_settings: Host/Origin validation policy from
            :func:`build_transport_security`; ``None`` leaves it off.
    """
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Route

    _log_transport_security(security_settings, host)
    session_manager = StreamableHTTPSessionManager(
        app=server,
        event_store=None,
        json_response=False,
        stateless=False,
        security_settings=security_settings,
    )

    # Starlette Route wraps functions/methods in request_response(),
    # but we need raw ASGI (scope, receive, send).  A callable class
    # instance passes Starlette's isfunction/ismethod check and is
    # treated as an ASGI app directly.
    class _AsgiEndpoint:
        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            # Bridge HTTP headers into a contextvar so that call_tool
            # can populate RequestContext.meta (the MCP SDK does not
            # pass transport-level headers to protocol-level handlers).
            from ._context import set_request_client_info, set_request_headers

            if scope["type"] in ("http", "websocket"):
                raw_headers = scope.get("headers", [])
                headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in raw_headers}
                set_request_headers(headers)
                # Bridge ASGI client info (IP, port)
                client = scope.get("client")
                if client:
                    set_request_client_info(tuple(client))

            await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        await lifecycle.startup()
        try:
            async with session_manager.run():
                yield
        finally:
            await lifecycle.shutdown(timeout=shutdown_timeout)

    routes = [
        Route("/mcp", endpoint=_AsgiEndpoint(), methods=["GET", "POST", "DELETE"]),
    ]

    # Token endpoint (built-in auth for dev/testing)
    if token_endpoint is not None:
        from ._token_endpoint import handle_token_request

        _te_config = token_endpoint

        class _TokenEndpointASGI:
            """ASGI wrapper for the token endpoint."""

            async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
                await handle_token_request(scope, receive, send, _te_config)

        routes.append(
            Route(
                token_endpoint.path,
                endpoint=_TokenEndpointASGI(),
                methods=["POST"],
            )
        )
        logger.info("Token endpoint enabled at %s", token_endpoint.path)

    asgi_app: Any = Starlette(
        routes=routes,
        lifespan=lifespan,
    )

    # CORS middleware (applied before auth gate so preflight works)
    if cors is not None:
        from starlette.middleware.cors import CORSMiddleware

        asgi_app.add_middleware(
            CORSMiddleware,
            allow_origins=cors.allow_origins,
            allow_methods=cors.allow_methods,
            allow_headers=cors.allow_headers,
            allow_credentials=cors.allow_credentials,
            max_age=cors.max_age,
        )

    # Transport-level auth gate (does NOT apply to token endpoint —
    # the gate wraps the whole app but the token endpoint is
    # unauthenticated by design since it *issues* tokens)
    if auth_gate is not None:
        # Build an auth gate that skips the token endpoint path
        _skip_paths = set()
        if token_endpoint is not None:
            _skip_paths.add(token_endpoint.path)
        asgi_app = _AuthGateASGI(asgi_app, auth_gate, skip_paths=_skip_paths)

    import uvicorn

    log_level = "critical" if dashboard else "info"
    config = uvicorn.Config(asgi_app, host=host, port=port, log_level=log_level)
    uv_server = uvicorn.Server(config)
    logger.info("MCP server running on http://%s:%d/mcp", host, port)
    await uv_server.serve()


# =====================================================================
# SSE transport (legacy)
# =====================================================================


async def run_sse(
    server: LowLevelServer,
    init_options: Any,
    lifecycle: LifecycleManager,
    *,
    host: str = "0.0.0.0",  # nosec B104 - public bind is explicit opt-in for server transports
    port: int = 8080,
    shutdown_timeout: float | None = None,
    dashboard: bool = False,
    auth_gate: Callable[[str], bool] | None = None,
    token_endpoint: Any = None,
    cors: CORSConfig | None = None,
    security_settings: TransportSecuritySettings | None = None,
) -> None:
    """Run the server over Server-Sent Events (legacy transport).

    Uses the MCP SDK's ``SseServerTransport``.

    Args:
        security_settings: Host/Origin validation policy from
            :func:`build_transport_security`; ``None`` leaves it off.  It
            is enforced on the ``/sse`` stream and on every ``/messages/``
            POST.
    """
    from mcp.server.sse import SseServerTransport
    from mcp.server.transport_security import TransportSecurityMiddleware
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.routing import Mount, Route

    _log_transport_security(security_settings, host)
    sse = SseServerTransport("/messages/", security_settings=security_settings)
    stream_security = TransportSecurityMiddleware(security_settings)

    # Raw ASGI endpoint (a callable instance bypasses Starlette's
    # request_response wrapper): the SSE stream is the whole response, so
    # there is nothing to return once it ends.
    class _SseEndpoint:
        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            # Validate Host/Origin before opening the stream: ``connect_sse``
            # answers a failing request itself and then raises, which would
            # surface as a spurious application error after the response.
            refused = await stream_security.validate_request(Request(scope, receive), is_post=False)
            if refused is not None:
                await refused(scope, receive, send)
                return

            # Bridge HTTP headers and client info for SSE connection
            from ._context import set_request_client_info, set_request_headers

            raw_headers = scope.get("headers", [])
            headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in raw_headers}
            set_request_headers(headers)
            client = scope.get("client")
            if client:
                set_request_client_info(tuple(client))

            async with sse.connect_sse(scope, receive, send) as streams:
                await server.run(streams[0], streams[1], init_options, raise_exceptions=False)

    async def handle_messages(scope: Any, receive: Any, send: Any) -> None:
        # Bridge HTTP headers and client info for message POST requests
        from ._context import set_request_client_info, set_request_headers

        raw_headers = scope.get("headers", [])
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in raw_headers}
        set_request_headers(headers)
        client = scope.get("client")
        if client:
            set_request_client_info(tuple(client))

        await sse.handle_post_message(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        await lifecycle.startup()
        try:
            yield
        finally:
            await lifecycle.shutdown(timeout=shutdown_timeout)

    routes = [
        Route("/sse", endpoint=_SseEndpoint(), methods=["GET"]),
        Mount("/messages/", app=handle_messages),
    ]

    # Token endpoint (built-in auth for dev/testing)
    if token_endpoint is not None:
        from ._token_endpoint import handle_token_request

        _te_config = token_endpoint

        class _TokenEndpointASGI:
            async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
                await handle_token_request(scope, receive, send, _te_config)

        routes.append(
            Route(
                token_endpoint.path,
                endpoint=_TokenEndpointASGI(),
                methods=["POST"],
            )
        )

    asgi_app: Any = Starlette(
        routes=routes,
        lifespan=lifespan,
    )

    # CORS middleware
    if cors is not None:
        from starlette.middleware.cors import CORSMiddleware

        asgi_app.add_middleware(
            CORSMiddleware,
            allow_origins=cors.allow_origins,
            allow_methods=cors.allow_methods,
            allow_headers=cors.allow_headers,
            allow_credentials=cors.allow_credentials,
            max_age=cors.max_age,
        )

    # Transport-level auth gate
    if auth_gate is not None:
        _skip_paths = set()
        if token_endpoint is not None:
            _skip_paths.add(token_endpoint.path)
        asgi_app = _AuthGateASGI(asgi_app, auth_gate, skip_paths=_skip_paths)

    import uvicorn

    log_level = "critical" if dashboard else "info"
    config = uvicorn.Config(asgi_app, host=host, port=port, log_level=log_level)
    uv_server = uvicorn.Server(config)
    logger.info("MCP server running on http://%s:%d/sse (SSE)", host, port)
    await uv_server.serve()


# =====================================================================
# Dispatcher
# =====================================================================


async def run_transport(
    transport_type: TransportType,
    server: LowLevelServer,
    init_options: Any,
    lifecycle: LifecycleManager,
    **kwargs: Any,
) -> None:
    """Dispatch to the appropriate transport runner."""
    runners: dict[TransportType, Callable[..., Any]] = {
        TransportType.STDIO: run_stdio,
        TransportType.HTTP: run_http,
        TransportType.SSE: run_sse,
    }
    runner = runners.get(transport_type)
    if runner is None:
        raise ValueError(f"Unsupported transport: {transport_type}")

    if transport_type == TransportType.STDIO:
        # Stdio doesn't accept network/dashboard/auth kwargs
        stdio_kwargs = {k: v for k, v in kwargs.items() if k == "shutdown_timeout"}
        await runner(server, init_options, lifecycle, **stdio_kwargs)
    else:
        await runner(server, init_options, lifecycle, **kwargs)
