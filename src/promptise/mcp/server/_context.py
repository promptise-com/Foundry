"""Per-request context for MCP server handlers.

Provides :class:`RequestContext` (per-request state available to every
handler and middleware), :class:`ClientContext` (structured authenticated
client information), and :class:`ToolResponse` (response wrapper with
metadata).
"""

from __future__ import annotations

import logging
import secrets
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger("promptise.server")

# Global context variable — set per request, read by handlers / middleware
_current_context: ContextVar[RequestContext | None] = ContextVar(
    "promptise_server_ctx", default=None
)

# HTTP request headers — set by the transport layer (ASGI middleware) and
# re-bound by call_tool to the request carrying the current MCP message
# (see ``bind_transport_request``), then read to populate
# RequestContext.meta.  This bridges the gap between transport-level HTTP
# headers and protocol-level tool handling.
_request_headers: ContextVar[dict[str, str] | None] = ContextVar(
    "promptise_request_headers", default=None
)

# ASGI client info — set by the transport layer, read by call_tool.
_request_client_info: ContextVar[tuple[str, int] | None] = ContextVar(
    "promptise_request_client_info", default=None
)


# =====================================================================
# ClientContext — structured authenticated client info
# =====================================================================


@dataclass
class ClientContext:
    """Structured information about the authenticated client.

    Populated by :class:`AuthMiddleware` after successful authentication.
    Handlers receive it via ``ctx.client`` — no need to dig into
    ``ctx.state["_jwt_payload"]`` or ``ctx.state["roles"]``.

    Attributes:
        client_id: Unique client identifier (from JWT ``sub`` or API key mapping).
        tenant_id: Tenant / organisation the client belongs to.  Populated by
            :class:`AuthMiddleware` from a configurable JWT claim (default
            ``tenant_id``) or from the API-key config dict.  A first-class
            field — rate limiting, audit entries, and tenant guards
            (``RequireTenant`` / ``HasTenant``) all key on it.
        roles: Set of role strings the client holds.
        scopes: Set of OAuth2 scope strings (from JWT ``scope`` claim).
        claims: Full JWT payload dict (empty for API key auth).
        issuer: JWT ``iss`` claim, or ``None``.
        audience: JWT ``aud`` claim (string or list), or ``None``.
        subject: JWT ``sub`` claim, or ``None``.
        issued_at: JWT ``iat`` claim (Unix timestamp), or ``None``.
        expires_at: JWT ``exp`` claim (Unix timestamp), or ``None``.
        ip_address: Client IP address from the transport layer, or ``None``.
        user_agent: ``User-Agent`` header value, or ``None``.
        extra: Custom metadata populated by the server's ``on_authenticate``
            enrichment hook.  Any additional client info (org, plan tier,
            etc.) goes here.

    Example::

        @server.tool(auth=True)
        async def my_tool(ctx: RequestContext) -> str:
            print(ctx.client.client_id)   # "agent-007"
            print(ctx.client.roles)       # {"admin", "analyst"}
            print(ctx.client.scopes)      # {"read", "write"}
            print(ctx.client.issuer)      # "https://auth.example.com"
            print(ctx.client.ip_address)  # "192.168.1.42"
            print(ctx.client.extra)       # {"org_id": "acme", "plan": "enterprise"}
            return "ok"
    """

    client_id: str = "anonymous"
    tenant_id: str | None = None
    roles: set[str] = field(default_factory=set)
    scopes: set[str] = field(default_factory=set)
    claims: dict[str, Any] = field(default_factory=dict)
    issuer: str | None = None
    audience: str | list[str] | None = None
    subject: str | None = None
    issued_at: float | None = None
    expires_at: float | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def has_role(self, role: str) -> bool:
        """Check if the client has a specific role."""
        return role in self.roles

    def has_any_role(self, *roles: str) -> bool:
        """Check if the client has at least one of the given roles."""
        return bool(self.roles & set(roles))

    def has_all_roles(self, *roles: str) -> bool:
        """Check if the client has all of the given roles."""
        return set(roles).issubset(self.roles)

    def has_scope(self, scope: str) -> bool:
        """Check if the client has a specific OAuth2 scope."""
        return scope in self.scopes

    def has_any_scope(self, *scopes: str) -> bool:
        """Check if the client has at least one of the given scopes."""
        return bool(self.scopes & set(scopes))


# =====================================================================
# ToolResponse — response wrapper with metadata
# =====================================================================


@dataclass
class ToolResponse:
    """Response wrapper that lets handlers return content with metadata.

    Handlers can return a ``ToolResponse`` instead of a plain value to
    attach metadata that is logged, recorded in audit trails, and
    available to middleware.  The ``content`` is serialised normally;
    ``metadata`` is stored on ``ctx.state["response_metadata"]`` for
    downstream use (e.g. by audit or webhook middleware).

    Attributes:
        content: The actual tool result (str, dict, list, etc.).
        metadata: Key-value metadata for observability and audit.
            Common keys: ``"cache"`` (``"hit"``/``"miss"``),
            ``"source"`` (where the data came from), ``"version"``.

    Example::

        @server.tool()
        async def search(query: str, ctx: RequestContext) -> ToolResponse:
            results = await db.search(query)
            return ToolResponse(
                content=results,
                metadata={"source": "primary_db", "result_count": len(results)},
            )
    """

    content: Any
    metadata: dict[str, Any] = field(default_factory=dict)


# =====================================================================
# RequestContext — per-request state
# =====================================================================


@dataclass
class RequestContext:
    """Per-request context available to handlers and middleware.

    Automatically created for each tool call, resource read, or prompt
    request.  Handlers can receive it by declaring a parameter typed as
    ``RequestContext``, or by calling :func:`get_context`.

    Attributes:
        server_name: Name of the MCPServer instance.
        tool_name: Name of the tool/resource/prompt being invoked.
        request_id: Unique identifier for this request.  If the client
            sends an ``X-Request-ID`` header, that value is used;
            otherwise a random hex string is generated.
        client_id: Authenticated client identifier (set by auth middleware).
            Shortcut for ``self.client.client_id``.
        client: Structured client context with roles, scopes, claims, IP,
            and custom metadata.  Populated by :class:`AuthMiddleware`.
        meta: Raw HTTP headers (lower-cased names) of the request that
            carries this call — per request, not per MCP session.
        state: Arbitrary per-request state (middleware can read/write).
            ``state["tool_def"]`` holds the definition being invoked (a
            ``ToolDef``, ``ResourceDef`` or ``PromptDef``); a resource read
            also sets ``state["resource_uri"]``.
        request_type: What is being invoked: ``"tool"``, ``"resource"``
            or ``"prompt"``.  Middleware runs for all three; branch on this
            when a middleware only makes sense for one of them.
        logger: Pre-configured logger scoped to this request.
    """

    server_name: str
    tool_name: str = ""
    request_id: str = field(default_factory=lambda: secrets.token_hex(6))
    client_id: str | None = None
    client: ClientContext = field(default_factory=ClientContext)
    meta: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    request_type: str = "tool"

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(f"promptise.server.{self.server_name}")


def get_context() -> RequestContext:
    """Return the current request context.

    Raises:
        RuntimeError: If called outside a request lifecycle.
    """
    ctx = _current_context.get()
    if ctx is None:
        raise RuntimeError(
            "No active RequestContext — this function must be called "
            "from within a tool/resource/prompt handler."
        )
    return ctx


def set_context(ctx: RequestContext) -> None:
    """Set the request context for the current async task."""
    _current_context.set(ctx)


def _wants_request_context(ann: Any) -> bool:
    """True if *ann* is (or contains) ``RequestContext``.

    Matches the bare type, ``Optional[RequestContext]`` / ``RequestContext |
    None`` (Python 3.10's ``get_type_hints`` applies *implicit Optional* to a
    ``ctx: RequestContext = None`` parameter — 3.11 removed that, hence the
    version skew this guards against), and an unresolved string annotation
    (``from __future__ import annotations`` when hint resolution failed).
    """
    if ann is RequestContext:
        return True
    from typing import get_args

    if RequestContext in get_args(ann):  # Optional[...] / Union[...]
        return True
    if isinstance(ann, str):
        # Unresolved annotation string (from __future__ import annotations when
        # hint resolution failed). Match RequestContext as a standalone token,
        # not a substring, so "MyRequestContextThing" is not caught.
        import re

        return re.search(r"\bRequestContext\b", ann) is not None
    return False


def inject_context(
    handler: Any,
    arguments: dict[str, Any],
    ctx: RequestContext,
) -> dict[str, Any]:
    """Bind *ctx* into a handler parameter annotated ``RequestContext``.

    Used by both the live transports and ``TestClient`` so a
    ``ctx: RequestContext`` parameter is populated identically on every
    path — no test/prod divergence.  Parameters already supplied (e.g. by
    dependency injection) are left untouched.  Recognises ``RequestContext``,
    ``Optional[RequestContext]``, and ``RequestContext | None`` on every
    supported Python (3.10–3.12).
    """
    import inspect
    from typing import get_type_hints

    sig = inspect.signature(handler)
    hints: dict[str, Any] = {}
    try:
        hints = get_type_hints(handler)
    except Exception:
        logger.debug("Failed to get type hints for handler %s", handler, exc_info=True)

    for pname, param in sig.parameters.items():
        if pname in arguments:
            continue
        ann = hints.get(pname, param.annotation)
        if _wants_request_context(ann):
            arguments[pname] = ctx
    return arguments


def clear_context() -> None:
    """Clear the request context after the request completes."""
    _current_context.set(None)


# ------------------------------------------------------------------
# HTTP request header bridging
# ------------------------------------------------------------------


def set_request_headers(headers: dict[str, str]) -> None:
    """Store HTTP request headers for the current async context.

    Called by the transport layer (ASGI middleware) so that
    ``call_tool`` can populate ``RequestContext.meta`` with
    headers like ``Authorization``.
    """
    _request_headers.set(headers)


def get_request_headers() -> dict[str, str]:
    """Return HTTP request headers for the current async context.

    Inside a tool handler on the HTTP/SSE transports these are the headers
    of the request that carries the current call (not of the request that
    opened the session).  Returns an empty dict when called outside an
    HTTP request (e.g. stdio transport, tests).
    """
    return _request_headers.get() or {}


def clear_request_headers() -> None:
    """Clear HTTP request headers after the request completes."""
    _request_headers.set(None)


def set_request_client_info(client: tuple[str, int] | None) -> None:
    """Store ASGI client info (host, port) for the current async context.

    Called by the transport layer alongside :func:`set_request_headers`.
    """
    _request_client_info.set(client)


def get_request_client_info() -> tuple[str, int] | None:
    """Return ASGI client info (host, port), or ``None`` for stdio."""
    return _request_client_info.get()


def clear_request_client_info() -> None:
    """Clear ASGI client info after the request completes."""
    _request_client_info.set(None)


def bind_transport_request(request: Any | None) -> tuple[dict[str, str], tuple[str, int] | None]:
    """Resolve the HTTP request that carries the *current* MCP message.

    The Streamable HTTP and SSE transports attach the Starlette ``Request``
    of every POST to the message the MCP SDK dispatches, so its headers and
    peer address describe *this* call.  The contextvars set by the ASGI
    layer cannot be trusted for that: under Streamable HTTP the session's
    server task is spawned inside the ``initialize`` request, so their copy
    would describe the request that opened the session for the session's
    whole lifetime — and caller identity, tenant, roles and ``X-Request-ID``
    would be bound to the session instead of the request.

    When a per-message request is available, the contextvars are re-bound
    to it for the current task so that every downstream reader
    (:func:`get_request_headers`, :func:`get_request_client_info`, and
    therefore ``ClientContext.ip_address``) observes the same request.
    Without one (stdio, direct handler invocation in tests) the contextvars
    remain the source of truth.

    Args:
        request: ``request_context.request`` of the MCP SDK for the message
            being handled, or ``None`` when the transport carries none.

    Returns:
        ``(headers, client)`` — header names lower-cased, and the peer
        ``(host, port)`` or ``None`` when the transport has no peer.
    """
    raw_headers = getattr(request, "headers", None)
    if request is None or raw_headers is None:
        return get_request_headers(), get_request_client_info()

    headers = {str(k).lower(): str(v) for k, v in raw_headers.items()}
    peer = getattr(request, "client", None)
    client_info: tuple[str, int] | None = None
    if peer is not None and getattr(peer, "host", None) is not None:
        client_info = (str(peer.host), int(peer.port or 0))
    set_request_headers(headers)
    set_request_client_info(client_info)
    return headers, client_info
