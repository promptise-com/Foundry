"""Tests for MCP server context, auth, guards, and response improvements.

Covers:
- ClientContext dataclass and helper methods
- ToolResponse wrapper
- JWT standard claims extraction (iss, aud, sub, iat, exp, scope)
- Request tracing (X-Request-ID propagation)
- Client enrichment hook (on_authenticate)
- Descriptive guard errors
- HasScope / HasAllScopes guards
- Client IP and user-agent extraction
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from promptise.mcp.server._auth import (
    APIKeyAuth,
    AuthMiddleware,
    JWTAuth,
    _build_client_context_from_api_key,
    _build_client_context_from_jwt,
)
from promptise.mcp.server._context import (
    ClientContext,
    RequestContext,
    ToolResponse,
    clear_request_client_info,
    get_request_client_info,
    set_request_client_info,
)
from promptise.mcp.server._errors import AuthenticationError
from promptise.mcp.server._guards import (
    HasAllRoles,
    HasAllScopes,
    HasRole,
    HasScope,
    HasTenant,
    RequireAuth,
    RequireClientId,
)

# =====================================================================
# ClientContext
# =====================================================================


class TestClientContext:
    """Tests for the ClientContext dataclass."""

    def test_defaults(self):
        ctx = ClientContext()
        assert ctx.client_id == "anonymous"
        assert ctx.roles == set()
        assert ctx.scopes == set()
        assert ctx.claims == {}
        assert ctx.issuer is None
        assert ctx.audience is None
        assert ctx.subject is None
        assert ctx.issued_at is None
        assert ctx.expires_at is None
        assert ctx.ip_address is None
        assert ctx.user_agent is None
        assert ctx.extra == {}

    def test_has_role(self):
        ctx = ClientContext(roles={"admin", "analyst"})
        assert ctx.has_role("admin")
        assert ctx.has_role("analyst")
        assert not ctx.has_role("superuser")

    def test_has_any_role(self):
        ctx = ClientContext(roles={"admin"})
        assert ctx.has_any_role("admin", "analyst")
        assert not ctx.has_any_role("analyst", "viewer")

    def test_has_all_roles(self):
        ctx = ClientContext(roles={"admin", "analyst", "viewer"})
        assert ctx.has_all_roles("admin", "analyst")
        assert not ctx.has_all_roles("admin", "superuser")

    def test_has_scope(self):
        ctx = ClientContext(scopes={"read", "write"})
        assert ctx.has_scope("read")
        assert not ctx.has_scope("delete")

    def test_has_any_scope(self):
        ctx = ClientContext(scopes={"read"})
        assert ctx.has_any_scope("read", "write")
        assert not ctx.has_any_scope("delete", "admin")

    def test_extra_metadata(self):
        ctx = ClientContext(extra={"org_id": "acme", "plan": "enterprise"})
        assert ctx.extra["org_id"] == "acme"
        assert ctx.extra["plan"] == "enterprise"


# =====================================================================
# ToolResponse
# =====================================================================


class TestToolResponse:
    """Tests for the ToolResponse wrapper."""

    def test_basic_response(self):
        resp = ToolResponse(content="hello", metadata={"source": "cache"})
        assert resp.content == "hello"
        assert resp.metadata["source"] == "cache"

    def test_default_metadata(self):
        resp = ToolResponse(content={"key": "value"})
        assert resp.metadata == {}

    def test_dict_content(self):
        data = {"results": [1, 2, 3]}
        resp = ToolResponse(content=data, metadata={"count": 3})
        assert resp.content == data
        assert resp.metadata["count"] == 3

    def test_none_content(self):
        resp = ToolResponse(content=None)
        assert resp.content is None


# =====================================================================
# RequestContext with ClientContext
# =====================================================================


class TestRequestContextClient:
    """Tests for ClientContext on RequestContext."""

    def test_default_client(self):
        ctx = RequestContext(server_name="test")
        assert ctx.client.client_id == "anonymous"
        assert ctx.client.roles == set()
        assert ctx.client.scopes == set()

    def test_client_assignment(self):
        ctx = RequestContext(server_name="test")
        ctx.client = ClientContext(
            client_id="agent-007",
            roles={"admin"},
            scopes={"read", "write"},
            issuer="https://auth.example.com",
        )
        assert ctx.client.client_id == "agent-007"
        assert ctx.client.has_role("admin")
        assert ctx.client.has_scope("read")
        assert ctx.client.issuer == "https://auth.example.com"


# =====================================================================
# Client IP bridging
# =====================================================================


class TestClientInfoBridging:
    """Tests for client IP/port bridging via contextvar."""

    def test_set_and_get(self):
        set_request_client_info(("192.168.1.42", 54321))
        info = get_request_client_info()
        assert info == ("192.168.1.42", 54321)
        clear_request_client_info()
        assert get_request_client_info() is None

    def test_default_none(self):
        clear_request_client_info()
        assert get_request_client_info() is None


# =====================================================================
# JWT claims extraction
# =====================================================================


class TestBuildClientContextFromJWT:
    """Tests for _build_client_context_from_jwt."""

    def test_standard_claims(self):
        payload = {
            "sub": "agent-007",
            "iss": "https://auth.example.com",
            "aud": "my-api",
            "iat": 1700000000,
            "exp": 1700003600,
            "roles": ["admin", "analyst"],
            "scope": "read write delete",
        }
        ctx = _build_client_context_from_jwt(payload, "agent-007")
        assert ctx.client_id == "agent-007"
        assert ctx.issuer == "https://auth.example.com"
        assert ctx.audience == "my-api"
        assert ctx.subject == "agent-007"
        assert ctx.issued_at == 1700000000
        assert ctx.expires_at == 1700003600
        assert ctx.roles == {"admin", "analyst"}
        assert ctx.scopes == {"read", "write", "delete"}
        assert ctx.claims == payload

    def test_scope_as_list(self):
        payload = {"sub": "a", "scope": ["read", "write"]}
        ctx = _build_client_context_from_jwt(payload, "a")
        assert ctx.scopes == {"read", "write"}

    def test_empty_scope(self):
        payload = {"sub": "a"}
        ctx = _build_client_context_from_jwt(payload, "a")
        assert ctx.scopes == set()

    def test_role_merging(self):
        payload = {"sub": "a", "roles": ["admin"]}
        ctx = _build_client_context_from_jwt(payload, "a", existing_roles={"analyst"})
        assert ctx.roles == {"admin", "analyst"}

    def test_audience_as_list(self):
        payload = {"sub": "a", "aud": ["api-1", "api-2"]}
        ctx = _build_client_context_from_jwt(payload, "a")
        assert ctx.audience == ["api-1", "api-2"]

    def test_user_agent_from_meta(self):
        payload = {"sub": "a"}
        ctx = _build_client_context_from_jwt(payload, "a", meta={"user-agent": "TestAgent/1.0"})
        assert ctx.user_agent == "TestAgent/1.0"

    def test_ip_from_client_info(self):
        set_request_client_info(("10.0.0.1", 8080))
        try:
            payload = {"sub": "a"}
            ctx = _build_client_context_from_jwt(payload, "a")
            assert ctx.ip_address == "10.0.0.1"
        finally:
            clear_request_client_info()

    def test_missing_optional_claims(self):
        payload = {"sub": "a"}
        ctx = _build_client_context_from_jwt(payload, "a")
        assert ctx.issuer is None
        assert ctx.audience is None
        assert ctx.issued_at is None
        assert ctx.expires_at is None


class TestBuildClientContextFromAPIKey:
    """Tests for _build_client_context_from_api_key."""

    def test_basic(self):
        ctx = _build_client_context_from_api_key("agent-1", {"read"})
        assert ctx.client_id == "agent-1"
        assert ctx.roles == {"read"}
        assert ctx.scopes == set()
        assert ctx.claims == {}

    def test_user_agent(self):
        ctx = _build_client_context_from_api_key("agent-1", set(), meta={"user-agent": "Bot/2.0"})
        assert ctx.user_agent == "Bot/2.0"


# =====================================================================
# AuthMiddleware integration
# =====================================================================


class TestAuthMiddlewareClientContext:
    """Tests that AuthMiddleware populates ctx.client."""

    @pytest.mark.asyncio
    async def test_jwt_populates_client_context(self):
        auth = JWTAuth(secret="test-secret")
        token = auth.create_token(
            {
                "sub": "agent-007",
                "roles": ["admin"],
                "scope": "read write",
                "iss": "test-issuer",
            }
        )

        middleware = AuthMiddleware(auth)

        # Create a mock tool_def that requires auth
        tool_def = MagicMock()
        tool_def.auth = True

        ctx = RequestContext(
            server_name="test",
            tool_name="my_tool",
            meta={"authorization": f"Bearer {token}"},
        )
        ctx.state["tool_def"] = tool_def

        call_next = AsyncMock(return_value="result")
        await middleware(ctx, call_next)

        assert ctx.client_id == "agent-007"
        assert ctx.client.client_id == "agent-007"
        assert ctx.client.roles == {"admin"}
        assert ctx.client.scopes == {"read", "write"}
        assert ctx.client.issuer == "test-issuer"

    @pytest.mark.asyncio
    async def test_api_key_populates_client_context(self):
        auth = APIKeyAuth(
            keys={
                "sk-abc": {"client_id": "agent-1", "roles": ["analyst"]},
            }
        )
        middleware = AuthMiddleware(auth)

        tool_def = MagicMock()
        tool_def.auth = True

        ctx = RequestContext(
            server_name="test",
            tool_name="my_tool",
            meta={"x-api-key": "sk-abc"},
        )
        ctx.state["tool_def"] = tool_def

        call_next = AsyncMock(return_value="result")
        await middleware(ctx, call_next)

        assert ctx.client_id == "agent-1"
        assert ctx.client.client_id == "agent-1"
        assert ctx.client.roles == {"analyst"}
        assert ctx.client.scopes == set()

    @pytest.mark.asyncio
    async def test_enrichment_hook_sync(self):
        auth = JWTAuth(secret="test-secret")
        token = auth.create_token({"sub": "agent-007"})

        def enrich(client_ctx, req_ctx):
            client_ctx.extra["org_id"] = "acme"
            client_ctx.extra["plan"] = "enterprise"

        middleware = AuthMiddleware(auth, on_authenticate=enrich)

        tool_def = MagicMock()
        tool_def.auth = True

        ctx = RequestContext(
            server_name="test",
            meta={"authorization": f"Bearer {token}"},
        )
        ctx.state["tool_def"] = tool_def

        call_next = AsyncMock(return_value="ok")
        await middleware(ctx, call_next)

        assert ctx.client.extra["org_id"] == "acme"
        assert ctx.client.extra["plan"] == "enterprise"

    @pytest.mark.asyncio
    async def test_enrichment_hook_async(self):
        auth = JWTAuth(secret="test-secret")
        token = auth.create_token({"sub": "agent-007"})

        async def enrich(client_ctx, req_ctx):
            client_ctx.extra["tenant"] = "tenant-42"

        middleware = AuthMiddleware(auth, on_authenticate=enrich)

        tool_def = MagicMock()
        tool_def.auth = True

        ctx = RequestContext(
            server_name="test",
            meta={"authorization": f"Bearer {token}"},
        )
        ctx.state["tool_def"] = tool_def

        call_next = AsyncMock(return_value="ok")
        await middleware(ctx, call_next)

        assert ctx.client.extra["tenant"] == "tenant-42"

    @pytest.mark.asyncio
    async def test_no_auth_skips_client_context(self):
        auth = JWTAuth(secret="test-secret")
        middleware = AuthMiddleware(auth)

        tool_def = MagicMock()
        tool_def.auth = False

        ctx = RequestContext(server_name="test")
        ctx.state["tool_def"] = tool_def

        call_next = AsyncMock(return_value="ok")
        await middleware(ctx, call_next)

        # Client should remain the default
        assert ctx.client.client_id == "anonymous"
        assert ctx.client_id is None


# =====================================================================
# Descriptive guard errors
# =====================================================================


class TestDescriptiveGuardErrors:
    """Tests that guard denial messages are informative."""

    def test_has_role_denial(self):
        guard = HasRole("admin", "superuser")
        ctx = RequestContext(server_name="test")
        ctx.state["roles"] = {"viewer"}
        msg = guard.describe_denial(ctx)
        assert "admin" in msg
        assert "superuser" in msg
        assert "viewer" in msg

    def test_has_role_denial_no_roles(self):
        guard = HasRole("admin")
        ctx = RequestContext(server_name="test")
        msg = guard.describe_denial(ctx)
        assert "(none)" in msg

    def test_has_all_roles_denial_shows_missing(self):
        guard = HasAllRoles("admin", "analyst", "ops")
        ctx = RequestContext(server_name="test")
        ctx.state["roles"] = {"admin"}
        msg = guard.describe_denial(ctx)
        assert "missing" in msg.lower()
        assert "analyst" in msg
        assert "ops" in msg

    def test_require_auth_denial(self):
        guard = RequireAuth()
        ctx = RequestContext(server_name="test")
        msg = guard.describe_denial(ctx)
        assert "authentication" in msg.lower()

    def test_require_client_id_denial(self):
        guard = RequireClientId("agent-007", "agent-008")
        ctx = RequestContext(server_name="test")
        ctx.client_id = "agent-999"
        msg = guard.describe_denial(ctx)
        assert "agent-999" in msg
        # The allowed clients are never revealed to a refused caller.
        assert "agent-007" not in msg and "agent-008" not in msg

    def test_has_tenant_denial_never_names_other_tenants(self):
        guard = HasTenant("acme", "initech")
        ctx = RequestContext(server_name="test")
        ctx.client = ClientContext(tenant_id="globex")
        msg = guard.describe_denial(ctx)
        assert "globex" in msg
        assert "acme" not in msg and "initech" not in msg
        ctx.client = ClientContext()
        assert "acme" not in guard.describe_denial(ctx)

    def test_has_scope_denial(self):
        guard = HasScope("read", "write")
        ctx = RequestContext(server_name="test")
        ctx.client = ClientContext(scopes={"delete"})
        msg = guard.describe_denial(ctx)
        assert "read" in msg
        assert "write" in msg
        assert "delete" in msg

    def test_has_all_scopes_denial_shows_missing(self):
        guard = HasAllScopes("read", "write", "admin")
        ctx = RequestContext(server_name="test")
        ctx.client = ClientContext(scopes={"read"})
        msg = guard.describe_denial(ctx)
        assert "missing" in msg.lower()
        assert "write" in msg
        assert "admin" in msg


# =====================================================================
# HasScope / HasAllScopes guards
# =====================================================================


class TestScopeGuards:
    """Tests for the new scope-based guards."""

    @pytest.mark.asyncio
    async def test_has_scope_allows(self):
        guard = HasScope("read", "write")
        ctx = RequestContext(server_name="test")
        ctx.client = ClientContext(scopes={"read"})
        assert await guard.check(ctx) is True

    @pytest.mark.asyncio
    async def test_has_scope_denies(self):
        guard = HasScope("admin")
        ctx = RequestContext(server_name="test")
        ctx.client = ClientContext(scopes={"read"})
        assert await guard.check(ctx) is False

    @pytest.mark.asyncio
    async def test_has_scope_no_client(self):
        guard = HasScope("read")
        ctx = RequestContext(server_name="test")
        ctx.client = ClientContext()  # no scopes
        assert await guard.check(ctx) is False

    @pytest.mark.asyncio
    async def test_has_all_scopes_allows(self):
        guard = HasAllScopes("read", "write")
        ctx = RequestContext(server_name="test")
        ctx.client = ClientContext(scopes={"read", "write", "admin"})
        assert await guard.check(ctx) is True

    @pytest.mark.asyncio
    async def test_has_all_scopes_denies(self):
        guard = HasAllScopes("read", "write")
        ctx = RequestContext(server_name="test")
        ctx.client = ClientContext(scopes={"read"})
        assert await guard.check(ctx) is False


# =====================================================================
# Request tracing
# =====================================================================


class TestRequestTracing:
    """Tests for X-Request-ID propagation."""

    def test_custom_request_id(self):
        ctx = RequestContext(
            server_name="test",
            request_id="custom-trace-123",
        )
        assert ctx.request_id == "custom-trace-123"

    def test_auto_generated_request_id(self):
        ctx = RequestContext(server_name="test")
        assert len(ctx.request_id) == 12  # secrets.token_hex(6) = 12 chars


# =====================================================================
# check_guards with descriptive errors
# =====================================================================


class TestCheckGuardsDescriptive:
    """Tests that check_guards uses describe_denial for error messages."""

    @pytest.mark.asyncio
    async def test_guard_failure_includes_detail(self):
        from promptise.mcp.server._testing import check_guards

        guard = HasRole("admin")
        ctx = RequestContext(server_name="test", tool_name="delete_user")
        ctx.state["roles"] = {"viewer"}

        with pytest.raises(AuthenticationError) as exc_info:
            await check_guards([guard], ctx)

        error = exc_info.value
        assert "admin" in str(error)
        assert "viewer" in str(error)
        assert error.code == "ACCESS_DENIED"
        assert error.details["guard"] == "HasRole"
        assert error.details["tool"] == "delete_user"

    @pytest.mark.asyncio
    async def test_guard_passes_no_error(self):
        from promptise.mcp.server._testing import check_guards

        guard = HasRole("admin")
        ctx = RequestContext(server_name="test")
        ctx.state["roles"] = {"admin"}

        # Should not raise
        await check_guards([guard], ctx)

    @pytest.mark.asyncio
    async def test_multiple_guards_first_failure(self):
        from promptise.mcp.server._testing import check_guards

        guard1 = RequireAuth()
        guard2 = HasRole("admin")

        ctx = RequestContext(server_name="test")
        # No client_id → RequireAuth fails first
        with pytest.raises(AuthenticationError) as exc_info:
            await check_guards([guard1, guard2], ctx)

        assert "RequireAuth" in exc_info.value.details["guard"]


# =====================================================================
# Backward compatibility
# =====================================================================


class TestBackwardCompatibility:
    """Ensure existing ctx.state['roles'] patterns still work."""

    @pytest.mark.asyncio
    async def test_roles_still_in_state(self):
        auth = JWTAuth(secret="test-secret")
        token = auth.create_token({"sub": "a", "roles": ["admin"]})

        middleware = AuthMiddleware(auth)
        tool_def = MagicMock()
        tool_def.auth = True

        ctx = RequestContext(
            server_name="test",
            meta={"authorization": f"Bearer {token}"},
        )
        ctx.state["tool_def"] = tool_def

        call_next = AsyncMock(return_value="ok")
        await middleware(ctx, call_next)

        # Old pattern still works
        assert "admin" in ctx.state["roles"]
        # New pattern also works
        assert ctx.client.has_role("admin")

    @pytest.mark.asyncio
    async def test_has_role_guard_still_works(self):
        guard = HasRole("admin")
        ctx = RequestContext(server_name="test")
        ctx.state["roles"] = {"admin"}
        assert await guard.check(ctx) is True
