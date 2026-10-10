"""Guard protocol and built-in guards for role-based access control.

Guards check fine-grained permissions after authentication. JWT payload
carries roles, auth middleware extracts them, guards enforce them.

When a guard denies access, the error message includes *why* — which
roles were required, which the client has, and what's missing.  This
makes debugging auth issues straightforward instead of guessing.

Example::

    from promptise.mcp.server import MCPServer, HasRole, HasScope

    @server.tool(auth=True, roles=["admin"])
    async def delete_user(user_id: str) -> str:
        return f"Deleted {user_id}"

    # Scope-based guard (OAuth2):
    @server.tool(auth=True, guards=[HasScope("write")])
    async def update_record(record_id: str) -> str:
        return f"Updated {record_id}"
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ._context import RequestContext


@runtime_checkable
class Guard(Protocol):
    """Protocol for tool access guards."""

    async def check(self, ctx: RequestContext) -> bool:
        """Check whether the request is allowed.

        Args:
            ctx: The current request context.

        Returns:
            ``True`` if allowed, ``False`` to deny.
        """
        ...

    def describe_denial(self, ctx: RequestContext) -> str:
        """Return a human-readable explanation of why access was denied.

        Guards should override this to provide helpful error messages.
        The default returns a generic message with the guard class name.
        """
        return f"Access denied by {type(self).__name__}"


class RequireAuth(Guard):
    """Guard that requires an authenticated ``client_id`` on the context."""

    async def check(self, ctx: RequestContext) -> bool:
        return ctx.client_id is not None

    def describe_denial(self, ctx: RequestContext) -> str:
        return "Authentication required — no client identity found on the request"


class HasRole(Guard):
    """Guard that requires **any** of the given roles.

    Reads roles from ``ctx.client.roles`` (or ``ctx.state["roles"]``),
    populated by ``AuthMiddleware`` from the JWT payload.

    Args:
        roles: One or more role names. Access granted if the client
            has **at least one** of these roles.
    """

    def __init__(self, *roles: str) -> None:
        self._required = set(roles)

    async def check(self, ctx: RequestContext) -> bool:
        # Prefer auth-verified ctx.client.roles if populated, fall back to
        # ctx.state["roles"] for backward compat with custom middleware
        client_roles = (
            ctx.client.roles if ctx.client and ctx.client.roles else ctx.state.get("roles", set())
        )
        return bool(self._required & client_roles)

    def describe_denial(self, ctx: RequestContext) -> str:
        client_roles = (
            ctx.client.roles if ctx.client and ctx.client.roles else ctx.state.get("roles", set())
        )
        required = ", ".join(sorted(self._required))
        actual = ", ".join(sorted(client_roles)) if client_roles else "(none)"
        return f"Requires any of roles [{required}], but client has [{actual}]"


class HasAllRoles(Guard):
    """Guard that requires **all** of the given roles.

    Args:
        roles: All of these roles must be present on the client.
    """

    def __init__(self, *roles: str) -> None:
        self._required = set(roles)

    async def check(self, ctx: RequestContext) -> bool:
        client_roles = (
            ctx.client.roles if ctx.client and ctx.client.roles else ctx.state.get("roles", set())
        )
        return self._required.issubset(client_roles)

    def describe_denial(self, ctx: RequestContext) -> str:
        client_roles = (
            ctx.client.roles if ctx.client and ctx.client.roles else ctx.state.get("roles", set())
        )
        missing = self._required - client_roles
        actual = ", ".join(sorted(client_roles)) if client_roles else "(none)"
        missing_str = ", ".join(sorted(missing))
        return (
            f"Requires all roles [{', '.join(sorted(self._required))}], "
            f"client has [{actual}], missing [{missing_str}]"
        )


class RequireClientId(Guard):
    """Guard that requires a specific ``client_id``.

    Args:
        allowed_ids: Set of allowed client identifiers.
    """

    def __init__(self, *allowed_ids: str) -> None:
        self._allowed = set(allowed_ids)

    async def check(self, ctx: RequestContext) -> bool:
        return ctx.client_id in self._allowed

    def describe_denial(self, ctx: RequestContext) -> str:
        # Never list the allowed clients: that would tell every refused
        # caller who else may call the tool.
        return f"Client '{ctx.client_id}' is not allowed to call this tool"


class RequireTenant(Guard):
    """Guard that requires the client to belong to *some* tenant.

    Reads ``ctx.client.tenant_id``, populated by ``AuthMiddleware`` from
    the configured JWT tenant claim (default ``tenant_id``) or from the
    API-key config dict.  Fails closed: no authentication → no tenant →
    denied.  Auto-applied to every tool when the server is built with
    ``MCPServer(require_tenant=True)``.

    Example::

        @server.tool(auth=True, guards=[RequireTenant()])
        async def list_records() -> list:
            ...
    """

    async def check(self, ctx: RequestContext) -> bool:
        client = getattr(ctx, "client", None)
        return bool(client is not None and getattr(client, "tenant_id", None))

    def describe_denial(self, ctx: RequestContext) -> str:
        return (
            "This tool requires a tenant identity, but the client presented "
            "no tenant claim (check the token's tenant claim or the API-key "
            "config's tenant_id)"
        )


class HasTenant(Guard):
    """Guard that requires the client to belong to one of the given tenants.

    Args:
        tenant_ids: One or more allowed tenant identifiers.

    Example::

        @server.tool(auth=True, guards=[HasTenant("acme", "globex")])
        async def partner_tool() -> str:
            ...
    """

    def __init__(self, *tenant_ids: str) -> None:
        self._allowed = set(tenant_ids)

    async def check(self, ctx: RequestContext) -> bool:
        client = getattr(ctx, "client", None)
        tenant = getattr(client, "tenant_id", None) if client is not None else None
        return tenant in self._allowed

    def describe_denial(self, ctx: RequestContext) -> str:
        client = getattr(ctx, "client", None)
        tenant = getattr(client, "tenant_id", None) if client is not None else None
        # Never list the allowed tenants: they are other customers.
        if not tenant:
            return "This tool requires a tenant identity, but the client presented none"
        return f"This tool is not available to tenant '{tenant}'"


class HasScope(Guard):
    """Guard that requires **any** of the given OAuth2 scopes.

    Reads scopes from ``ctx.client.scopes``, populated by
    ``AuthMiddleware`` from the JWT ``scope`` claim (space-separated
    string per RFC 8693).

    .. note::

        Scopes are a JWT concept. When using ``APIKeyAuth`` without
        JWT, ``ctx.client.scopes`` will be empty and scope guards
        will always deny.  Use role-based guards (``HasRole``) for
        API key auth.

    Args:
        scopes: One or more scope strings. Access granted if the client
            has **at least one** of these scopes.

    Example::

        @server.tool(auth=True, guards=[HasScope("read", "admin")])
        async def get_data() -> str:
            return "data"
    """

    def __init__(self, *scopes: str) -> None:
        self._required = set(scopes)

    async def check(self, ctx: RequestContext) -> bool:
        if ctx.client is None:
            return False
        return bool(self._required & ctx.client.scopes)

    def describe_denial(self, ctx: RequestContext) -> str:
        client_scopes = ctx.client.scopes if ctx.client else set()
        required = ", ".join(sorted(self._required))
        actual = ", ".join(sorted(client_scopes)) if client_scopes else "(none)"
        return f"Requires any of scopes [{required}], but client has [{actual}]"


class HasAllScopes(Guard):
    """Guard that requires **all** of the given OAuth2 scopes.

    Args:
        scopes: All of these scopes must be present on the client.
    """

    def __init__(self, *scopes: str) -> None:
        self._required = set(scopes)

    async def check(self, ctx: RequestContext) -> bool:
        if ctx.client is None:
            return False
        return self._required.issubset(ctx.client.scopes)

    def describe_denial(self, ctx: RequestContext) -> str:
        client_scopes = ctx.client.scopes if ctx.client else set()
        missing = self._required - client_scopes
        actual = ", ".join(sorted(client_scopes)) if client_scopes else "(none)"
        missing_str = ", ".join(sorted(missing))
        return (
            f"Requires all scopes [{', '.join(sorted(self._required))}], "
            f"client has [{actual}], missing [{missing_str}]"
        )
