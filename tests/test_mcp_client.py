"""Tests for the Promptise MCP Client.

Tests MCPClient, MCPMultiClient, MCPToolAdapter, HTTPServerSpec,
the server-side token endpoint, API key auth, tool result extraction,
and transport-level auth gate.

Security design: The client NEVER generates JWTs.  Tokens are obtained
from an IdP or the server's built-in token endpoint and passed as
``bearer_token``.  For simpler setups, ``api_key`` is supported.
"""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from promptise.config import HTTPServerSpec
from promptise.mcp.client import MCPClient, MCPClientError, MCPMultiClient

# =====================================================================
# MCPClient unit tests
# =====================================================================


class TestMCPClientConstruction:
    def test_creates_http_client(self):
        client = MCPClient(url="http://localhost:8080/mcp")
        assert client._url == "http://localhost:8080/mcp"
        assert client._transport == "http"
        assert client._headers == {}

    def test_creates_with_headers(self):
        client = MCPClient(
            url="http://localhost:8080/mcp",
            headers={"x-custom": "value"},
        )
        assert client._headers["x-custom"] == "value"

    def test_bearer_token_creates_auth_header(self):
        """bearer_token should inject Authorization header."""
        client = MCPClient(
            url="http://localhost:8080/mcp",
            bearer_token="my-token-123",
        )
        assert client._headers["authorization"] == "Bearer my-token-123"

    def test_no_auth_without_bearer_token(self):
        """No token → no Authorization header."""
        client = MCPClient(url="http://localhost:8080/mcp")
        assert "authorization" not in client._headers

    def test_bearer_token_none_no_header(self):
        """Explicit None bearer_token → no auth header."""
        client = MCPClient(
            url="http://localhost:8080/mcp",
            bearer_token=None,
        )
        assert "authorization" not in client._headers

    def test_bearer_token_plus_custom_headers(self):
        """bearer_token + custom headers should coexist."""
        client = MCPClient(
            url="http://localhost:8080/mcp",
            bearer_token="tok",
            headers={"x-custom": "val"},
        )
        assert client._headers["authorization"] == "Bearer tok"
        assert client._headers["x-custom"] == "val"

    def test_manual_auth_header_preserved(self):
        """Manual Authorization header should work without bearer_token."""
        client = MCPClient(
            url="http://localhost:8080/mcp",
            headers={"authorization": "Bearer manual-token"},
        )
        assert client._headers["authorization"] == "Bearer manual-token"

    def test_bearer_token_overrides_manual_header(self):
        """bearer_token should override a manual Authorization header."""
        client = MCPClient(
            url="http://localhost:8080/mcp",
            headers={"authorization": "Bearer old"},
            bearer_token="new-token",
        )
        assert client._headers["authorization"] == "Bearer new-token"

    def test_headers_property_returns_copy(self):
        client = MCPClient(
            url="http://localhost:8080/mcp",
            headers={"key": "val"},
        )
        h = client.headers
        h["new"] = "should not affect original"
        assert "new" not in client._headers

    def test_session_none_before_connect(self):
        client = MCPClient(url="http://localhost:8080/mcp")
        assert client.session is None

    async def test_require_session_raises_when_not_connected(self):
        client = MCPClient(url="http://localhost:8080/mcp")
        with pytest.raises(MCPClientError, match="Not connected"):
            await client.list_tools()

    def test_stdio_client(self):
        client = MCPClient(
            transport="stdio",
            command="python",
            args=["-m", "my_server"],
        )
        assert client._transport == "stdio"
        assert client._command == "python"
        assert client._args == ["-m", "my_server"]

    async def test_unknown_transport_raises(self):
        client = MCPClient(url="http://x", transport="grpc")
        with pytest.raises(MCPClientError, match="Unknown transport"):
            async with client:
                pass


# =====================================================================
# MCPMultiClient unit tests
# =====================================================================


class TestMCPMultiClient:
    def test_construction(self):
        c1 = MCPClient(url="http://a")
        c2 = MCPClient(url="http://b")
        multi = MCPMultiClient({"a": c1, "b": c2})
        assert len(multi.servers) == 2

    async def test_call_tool_without_discovery_raises(self):
        c1 = MCPClient(url="http://a")
        multi = MCPMultiClient({"a": c1})
        with pytest.raises(MCPClientError, match="Unknown tool"):
            await multi.call_tool("foo", {})

    def test_tool_to_server_empty_before_connect(self):
        c1 = MCPClient(url="http://a")
        multi = MCPMultiClient({"a": c1})
        assert multi.tool_to_server == {}


# =====================================================================
# HTTPServerSpec with bearer_token
# =====================================================================


class TestHTTPServerSpecAuth:
    def test_bearer_token_optional(self):
        spec = HTTPServerSpec(url="http://localhost:8080/mcp")
        assert spec.bearer_token is None

    def test_bearer_token_set(self):
        spec = HTTPServerSpec(
            url="http://localhost:8080/mcp",
            bearer_token="my-jwt-token",
        )
        assert spec.bearer_token.get_secret_value() == "my-jwt-token"

    def test_no_jwt_fields(self):
        """Ensure jwt_secret/jwt_claims are NOT accepted."""
        with pytest.raises(Exception):
            HTTPServerSpec(
                url="http://localhost:8080/mcp",
                jwt_secret="secret",  # type: ignore[call-arg]
            )

    def test_backward_compat_headers(self):
        """Old-style manual headers still work."""
        spec = HTTPServerSpec(
            url="http://localhost:8080/mcp",
            headers={"authorization": "Bearer old-token"},
        )
        assert spec.headers["authorization"] == "Bearer old-token"
        assert spec.bearer_token is None

    def test_backward_compat_auth(self):
        """Legacy auth field still accepted."""
        spec = HTTPServerSpec(
            url="http://localhost:8080/mcp",
            auth="some-hint",
        )
        assert spec.auth == "some-hint"


# =====================================================================
# Integration: Bearer token → MCPServer auth (via TestClient)
# =====================================================================


class TestBearerTokenIntegration:
    """Test that pre-issued tokens work with MCPServer's auth pipeline."""

    async def test_bearer_token_auth_roundtrip(self):
        """Token issued by JWTAuth.create_token() works for auth."""
        from promptise.mcp.server import (
            AuthMiddleware,
            JWTAuth,
            MCPServer,
            TestClient,
        )

        secret = "integration-test-secret"
        jwt_auth = JWTAuth(secret=secret)

        server = MCPServer(name="test-auth")
        server.add_middleware(AuthMiddleware(jwt_auth))

        @server.tool(auth=True, roles=["admin"])
        async def admin_only() -> str:
            return "admin-access-granted"

        @server.tool(auth=True, roles=["admin", "viewer"])
        async def read_data() -> str:
            return "data-here"

        # Issue token server-side (as IdP or token endpoint would)
        token = jwt_auth.create_token(
            {"sub": "test-agent", "roles": ["admin"]},
            expires_in=3600,
        )

        # Client receives the pre-issued token
        client = MCPClient(
            url="http://fake",
            bearer_token=token,
        )
        assert client._headers["authorization"] == f"Bearer {token}"

        # Verify via TestClient (simulates the full auth pipeline)
        test_client = TestClient(server, meta={"authorization": f"Bearer {token}"})

        result = await test_client.call_tool("admin_only", {})
        assert result[0].text == "admin-access-granted"

        result2 = await test_client.call_tool("read_data", {})
        assert result2[0].text == "data-here"

    async def test_wrong_role_rejected(self):
        """Token with wrong role should be rejected by guards."""
        from promptise.mcp.server import (
            AuthMiddleware,
            JWTAuth,
            MCPServer,
            TestClient,
        )

        secret = "guard-test-secret"
        jwt_auth = JWTAuth(secret=secret)

        server = MCPServer(name="test-guard")
        server.add_middleware(AuthMiddleware(jwt_auth))

        @server.tool(auth=True, roles=["admin"])
        async def admin_only() -> str:
            return "should-not-see-this"

        # Issue a viewer token (no admin role)
        token = jwt_auth.create_token(
            {"sub": "viewer-agent", "roles": ["viewer"]},
            expires_in=3600,
        )
        test_client = TestClient(server, meta={"authorization": f"Bearer {token}"})

        result = await test_client.call_tool("admin_only", {})
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_no_token_rejected(self):
        """Unauthenticated request should be rejected."""
        from promptise.mcp.server import (
            AuthMiddleware,
            JWTAuth,
            MCPServer,
            TestClient,
        )

        secret = "no-auth-test"
        jwt_auth = JWTAuth(secret=secret)

        server = MCPServer(name="test-noauth")
        server.add_middleware(AuthMiddleware(jwt_auth))

        @server.tool(auth=True, roles=["admin"])
        async def admin_only() -> str:
            return "nope"

        test_client = TestClient(server)
        result = await test_client.call_tool("admin_only", {})
        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "AUTHENTICATION_ERROR"


# =====================================================================
# Token endpoint tests
# =====================================================================


class TestTokenEndpoint:
    """Test the server's built-in token endpoint (ASGI handler)."""

    def _make_server_with_token_endpoint(self):
        """Helper: create a configured server with token endpoint."""
        from promptise.mcp.server import AuthMiddleware, JWTAuth, MCPServer

        secret = "token-endpoint-test"
        jwt_auth = JWTAuth(secret=secret)

        server = MCPServer(name="test-token-ep")
        server.add_middleware(AuthMiddleware(jwt_auth))

        server.enable_token_endpoint(
            jwt_auth=jwt_auth,
            clients={
                "agent-admin": {"secret": "admin-pass", "roles": ["admin", "finance"]},
                "agent-viewer": {"secret": "viewer-pass", "roles": ["viewer"]},
            },
        )

        @server.tool(auth=True, roles=["admin"])
        async def admin_tool() -> str:
            return "admin-ok"

        return server, jwt_auth

    def test_enable_token_endpoint_stores_config(self):
        """MCPServer should store token endpoint config."""
        server, jwt_auth = self._make_server_with_token_endpoint()
        assert server._token_endpoint is not None
        assert server._token_endpoint.path == "/auth/token"
        assert "agent-admin" in server._token_endpoint.clients

    async def test_token_endpoint_issues_valid_token(self):
        """The token endpoint should issue a valid JWT."""
        from promptise.mcp.server import JWTAuth
        from promptise.mcp.server._token_endpoint import (
            TokenEndpointConfig,
            handle_token_request,
        )

        secret = "endpoint-test-secret"
        jwt_auth = JWTAuth(secret=secret)
        config = TokenEndpointConfig(
            jwt_auth=jwt_auth,
            clients={
                "my-agent": {"secret": "agent-pass", "roles": ["admin"]},
            },
        )

        # Simulate ASGI request
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/auth/token",
        }
        body = json.dumps(
            {
                "client_id": "my-agent",
                "client_secret": "agent-pass",
            }
        ).encode()

        received_status = None
        received_body = None

        async def receive():
            return {"body": body, "more_body": False}

        async def send(message):
            nonlocal received_status, received_body
            if message["type"] == "http.response.start":
                received_status = message["status"]
            elif message["type"] == "http.response.body":
                received_body = message["body"]

        await handle_token_request(scope, receive, send, config)

        assert received_status == 200
        data = json.loads(received_body)
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 3600

        # Verify the issued token
        assert jwt_auth.verify_token(data["access_token"]) is True

    async def test_token_endpoint_rejects_bad_secret(self):
        """Wrong client_secret should be rejected."""
        from promptise.mcp.server import JWTAuth
        from promptise.mcp.server._token_endpoint import (
            TokenEndpointConfig,
            handle_token_request,
        )

        jwt_auth = JWTAuth(secret="test")
        config = TokenEndpointConfig(
            jwt_auth=jwt_auth,
            clients={"agent": {"secret": "correct-pass", "roles": []}},
        )

        scope = {"type": "http", "method": "POST", "path": "/auth/token"}
        body = json.dumps(
            {
                "client_id": "agent",
                "client_secret": "wrong-pass",
            }
        ).encode()

        received_status = None

        async def receive():
            return {"body": body, "more_body": False}

        async def send(message):
            nonlocal received_status
            if message["type"] == "http.response.start":
                received_status = message["status"]

        await handle_token_request(scope, receive, send, config)
        assert received_status == 401

    async def test_token_endpoint_rejects_unknown_client(self):
        """Unknown client_id should be rejected."""
        from promptise.mcp.server import JWTAuth
        from promptise.mcp.server._token_endpoint import (
            TokenEndpointConfig,
            handle_token_request,
        )

        jwt_auth = JWTAuth(secret="test")
        config = TokenEndpointConfig(
            jwt_auth=jwt_auth,
            clients={"known": {"secret": "pass", "roles": []}},
        )

        scope = {"type": "http", "method": "POST", "path": "/auth/token"}
        body = json.dumps(
            {
                "client_id": "unknown-agent",
                "client_secret": "pass",
            }
        ).encode()

        received_status = None

        async def receive():
            return {"body": body, "more_body": False}

        async def send(message):
            nonlocal received_status
            if message["type"] == "http.response.start":
                received_status = message["status"]

        await handle_token_request(scope, receive, send, config)
        assert received_status == 401

    async def test_token_endpoint_rejects_get(self):
        """GET requests should be rejected (only POST allowed)."""
        from promptise.mcp.server import JWTAuth
        from promptise.mcp.server._token_endpoint import (
            TokenEndpointConfig,
            handle_token_request,
        )

        jwt_auth = JWTAuth(secret="test")
        config = TokenEndpointConfig(jwt_auth=jwt_auth, clients={})

        scope = {"type": "http", "method": "GET", "path": "/auth/token"}

        received_status = None

        async def receive():
            return {"body": b"", "more_body": False}

        async def send(message):
            nonlocal received_status
            if message["type"] == "http.response.start":
                received_status = message["status"]

        await handle_token_request(scope, receive, send, config)
        assert received_status == 405

    async def test_token_endpoint_rejects_invalid_json(self):
        """Invalid JSON body should return 400."""
        from promptise.mcp.server import JWTAuth
        from promptise.mcp.server._token_endpoint import (
            TokenEndpointConfig,
            handle_token_request,
        )

        jwt_auth = JWTAuth(secret="test")
        config = TokenEndpointConfig(jwt_auth=jwt_auth, clients={})

        scope = {"type": "http", "method": "POST", "path": "/auth/token"}

        received_status = None

        async def receive():
            return {"body": b"not-json", "more_body": False}

        async def send(message):
            nonlocal received_status
            if message["type"] == "http.response.start":
                received_status = message["status"]

        await handle_token_request(scope, receive, send, config)
        assert received_status == 400

    async def test_token_endpoint_custom_expires_in(self):
        """Custom expires_in per client should be respected."""
        from promptise.mcp.server import JWTAuth
        from promptise.mcp.server._token_endpoint import (
            TokenEndpointConfig,
            handle_token_request,
        )

        jwt_auth = JWTAuth(secret="test")
        config = TokenEndpointConfig(
            jwt_auth=jwt_auth,
            clients={
                "fast-agent": {"secret": "pass", "roles": ["admin"], "expires_in": 600},
            },
        )

        scope = {"type": "http", "method": "POST", "path": "/auth/token"}
        body = json.dumps(
            {
                "client_id": "fast-agent",
                "client_secret": "pass",
            }
        ).encode()

        received_body = None

        async def receive():
            return {"body": body, "more_body": False}

        async def send(message):
            nonlocal received_body
            if message["type"] == "http.response.body":
                received_body = message["body"]

        await handle_token_request(scope, receive, send, config)
        data = json.loads(received_body)
        assert data["expires_in"] == 600

    async def test_issued_token_works_with_server_auth(self):
        """Full roundtrip: token endpoint issues token → use with server tool."""
        from promptise.mcp.server import (
            AuthMiddleware,
            JWTAuth,
            MCPServer,
            TestClient,
        )
        from promptise.mcp.server._token_endpoint import (
            TokenEndpointConfig,
            handle_token_request,
        )

        secret = "roundtrip-test"
        jwt_auth = JWTAuth(secret=secret)

        # Step 1: Issue a token via the endpoint
        config = TokenEndpointConfig(
            jwt_auth=jwt_auth,
            clients={
                "agent-admin": {"secret": "s3cret", "roles": ["admin"]},
            },
        )

        scope = {"type": "http", "method": "POST", "path": "/auth/token"}
        body = json.dumps(
            {
                "client_id": "agent-admin",
                "client_secret": "s3cret",
            }
        ).encode()

        received_body = None

        async def receive():
            return {"body": body, "more_body": False}

        async def send(message):
            nonlocal received_body
            if message["type"] == "http.response.body":
                received_body = message["body"]

        await handle_token_request(scope, receive, send, config)
        token = json.loads(received_body)["access_token"]

        # Step 2: Use the token with the server
        server = MCPServer(name="test-roundtrip")
        server.add_middleware(AuthMiddleware(jwt_auth))

        @server.tool(auth=True, roles=["admin"])
        async def protected_tool() -> str:
            return "success"

        test_client = TestClient(server, meta={"authorization": f"Bearer {token}"})
        result = await test_client.call_tool("protected_tool", {})
        assert result[0].text == "success"

    async def test_token_endpoint_missing_fields(self):
        """Missing client_id or client_secret should return 400."""
        from promptise.mcp.server import JWTAuth
        from promptise.mcp.server._token_endpoint import (
            TokenEndpointConfig,
            handle_token_request,
        )

        jwt_auth = JWTAuth(secret="test")
        config = TokenEndpointConfig(jwt_auth=jwt_auth, clients={})

        scope = {"type": "http", "method": "POST", "path": "/auth/token"}

        received_status = None

        async def receive():
            return {"body": json.dumps({"client_id": "x"}).encode(), "more_body": False}

        async def send(message):
            nonlocal received_status
            if message["type"] == "http.response.start":
                received_status = message["status"]

        await handle_token_request(scope, receive, send, config)
        assert received_status == 400


# =====================================================================
# MCPToolAdapter schema conversion test
# =====================================================================


class TestToolAdapterSchemaQuality:
    """Verify that tools discovered by the adapter have proper schemas."""

    async def test_nested_model_produces_structured_schema(self):
        """The adapter should produce Pydantic models, not flat dicts."""
        from pydantic import BaseModel as PydBaseModel

        from promptise.tools import _jsonschema_to_pydantic

        # Simulate a server schema with nested Address inside Employee
        schema = {
            "type": "object",
            "properties": {
                "employee": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Full name"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                                "zip": {"type": "string"},
                            },
                            "required": ["city"],
                        },
                    },
                    "required": ["name", "address"],
                },
            },
            "required": ["employee"],
        }

        model = _jsonschema_to_pydantic(schema, model_name="Args_create")

        # employee field should be a Pydantic model, NOT dict
        emp_ann = model.model_fields["employee"].annotation
        assert issubclass(emp_ann, PydBaseModel)

        # address inside employee should also be a model
        addr_ann = emp_ann.model_fields["address"].annotation
        assert issubclass(addr_ann, PydBaseModel)
        assert "city" in addr_ann.model_fields


# =====================================================================
# API key auth tests — MCPClient
# =====================================================================


class TestMCPClientAPIKeyAuth:
    """Test API key injection via the api_key parameter."""

    def test_api_key_creates_header(self):
        """api_key should inject x-api-key header."""
        client = MCPClient(
            url="http://localhost:8080/mcp",
            api_key="my-secret-key",
        )
        assert client._headers["x-api-key"] == "my-secret-key"

    def test_no_api_key_header_without_param(self):
        """No api_key → no x-api-key header."""
        client = MCPClient(url="http://localhost:8080/mcp")
        assert "x-api-key" not in client._headers

    def test_api_key_none_no_header(self):
        """Explicit None api_key → no header."""
        client = MCPClient(
            url="http://localhost:8080/mcp",
            api_key=None,
        )
        assert "x-api-key" not in client._headers

    def test_api_key_plus_custom_headers(self):
        """api_key + custom headers should coexist."""
        client = MCPClient(
            url="http://localhost:8080/mcp",
            api_key="key123",
            headers={"x-custom": "val"},
        )
        assert client._headers["x-api-key"] == "key123"
        assert client._headers["x-custom"] == "val"

    def test_api_key_plus_bearer_token(self):
        """Both api_key and bearer_token should inject separate headers."""
        client = MCPClient(
            url="http://localhost:8080/mcp",
            bearer_token="my-jwt",
            api_key="my-key",
        )
        assert client._headers["authorization"] == "Bearer my-jwt"
        assert client._headers["x-api-key"] == "my-key"

    def test_api_key_overrides_manual_header(self):
        """api_key param should override a manual x-api-key header."""
        client = MCPClient(
            url="http://localhost:8080/mcp",
            headers={"x-api-key": "old-key"},
            api_key="new-key",
        )
        assert client._headers["x-api-key"] == "new-key"


# =====================================================================
# HTTPServerSpec with api_key
# =====================================================================


class TestHTTPServerSpecAPIKey:
    """Test API key field on HTTPServerSpec."""

    def test_api_key_optional(self):
        spec = HTTPServerSpec(url="http://localhost:8080/mcp")
        assert spec.api_key is None

    def test_api_key_set(self):
        spec = HTTPServerSpec(
            url="http://localhost:8080/mcp",
            api_key="secret-123",
        )
        assert spec.api_key.get_secret_value() == "secret-123"

    def test_api_key_plus_bearer_token(self):
        """Both can be set (though typically only one is used)."""
        spec = HTTPServerSpec(
            url="http://localhost:8080/mcp",
            bearer_token="jwt-tok",
            api_key="key-tok",
        )
        assert spec.bearer_token.get_secret_value() == "jwt-tok"
        assert spec.api_key.get_secret_value() == "key-tok"

    def test_api_key_defaults_none(self):
        """Default api_key should be None, not empty string."""
        spec = HTTPServerSpec(url="http://localhost:8080/mcp")
        assert spec.api_key is None
        assert spec.bearer_token is None


# =====================================================================
# API key auth integration (server-side via TestClient)
# =====================================================================


class TestAPIKeyAuthIntegration:
    """Test that API key auth works end-to-end with the server."""

    async def test_api_key_auth_roundtrip(self):
        """API key should authenticate via x-api-key header."""
        from promptise.mcp.server import (
            APIKeyAuth,
            AuthMiddleware,
            MCPServer,
            TestClient,
        )

        keys = {"secret-key-123": "my-agent"}
        api_auth = APIKeyAuth(keys=keys)

        server = MCPServer(name="test-apikey")
        server.add_middleware(AuthMiddleware(api_auth))

        @server.tool(auth=True)
        async def protected() -> str:
            return "api-key-works"

        test_client = TestClient(server, meta={"x-api-key": "secret-key-123"})
        result = await test_client.call_tool("protected", {})
        assert result[0].text == "api-key-works"

    async def test_api_key_invalid_rejected(self):
        """Invalid API key should be rejected."""
        from promptise.mcp.server import (
            APIKeyAuth,
            AuthMiddleware,
            MCPServer,
            TestClient,
        )

        keys = {"valid-key": "agent"}
        api_auth = APIKeyAuth(keys=keys)

        server = MCPServer(name="test-apikey-reject")
        server.add_middleware(AuthMiddleware(api_auth))

        @server.tool(auth=True)
        async def protected() -> str:
            return "nope"

        test_client = TestClient(server, meta={"x-api-key": "wrong-key"})
        result = await test_client.call_tool("protected", {})
        data = json.loads(result[0].text)
        assert "error" in data

    async def test_api_key_missing_rejected(self):
        """Missing API key should be rejected."""
        from promptise.mcp.server import (
            APIKeyAuth,
            AuthMiddleware,
            MCPServer,
            TestClient,
        )

        keys = {"valid-key": "agent"}
        api_auth = APIKeyAuth(keys=keys)

        server = MCPServer(name="test-apikey-missing")
        server.add_middleware(AuthMiddleware(api_auth))

        @server.tool(auth=True)
        async def protected() -> str:
            return "nope"

        test_client = TestClient(server)
        result = await test_client.call_tool("protected", {})
        data = json.loads(result[0].text)
        assert "error" in data
        assert data["error"]["code"] == "AUTHENTICATION_ERROR"


# =====================================================================
# _extract_text — CallToolResult → string extraction
# =====================================================================


class TestExtractText:
    """Test that _extract_text properly handles CallToolResult objects."""

    def test_extract_single_text_content(self):
        """Single TextContent should be extracted as plain string."""
        from promptise.mcp.client._tool_adapter import _extract_text

        # Simulate a CallToolResult with TextContent
        result = MagicMock()
        content_item = MagicMock()
        content_item.text = "hello world"
        result.content = [content_item]

        assert _extract_text(result) == "hello world"

    def test_extract_multiple_text_content(self):
        """Multiple TextContent items should be joined with newlines."""
        from promptise.mcp.client._tool_adapter import _extract_text

        result = MagicMock()
        item1 = MagicMock()
        item1.text = "line one"
        item2 = MagicMock()
        item2.text = "line two"
        result.content = [item1, item2]

        assert _extract_text(result) == "line one\nline two"

    def test_extract_empty_content(self):
        """Empty content list should return empty string."""
        from promptise.mcp.client._tool_adapter import _extract_text

        result = MagicMock()
        result.content = []

        assert _extract_text(result) == ""

    def test_extract_no_content_attr(self):
        """Result without .content should return empty string."""
        from promptise.mcp.client._tool_adapter import _extract_text

        result = object()  # no .content attribute
        assert _extract_text(result) == ""

    def test_extract_none_content(self):
        """None content should return empty string."""
        from promptise.mcp.client._tool_adapter import _extract_text

        result = MagicMock()
        result.content = None

        assert _extract_text(result) == ""

    def test_extract_skips_non_text_items(self):
        """Non-text content (e.g. ImageContent) should be skipped."""
        from promptise.mcp.client._tool_adapter import _extract_text

        result = MagicMock()
        text_item = MagicMock()
        text_item.text = "text data"
        image_item = MagicMock(spec=[])  # no .text attribute
        result.content = [text_item, image_item]

        assert _extract_text(result) == "text data"


# =====================================================================
# Transport-level auth gate tests
# =====================================================================


class TestAuthGateASGI:
    """Test the transport-level auth gate supports both Bearer and API key."""

    def _make_gate(self, verify_fn, *, skip_paths=None, api_key_verify_fn=None):
        from promptise.mcp.server._transport import _AuthGateASGI

        MagicMock()

        async def mock_app(scope, receive, send):
            scope["_passed"] = True

        return _AuthGateASGI(
            mock_app,
            verify_fn,
            skip_paths=skip_paths,
            api_key_verify_fn=api_key_verify_fn,
        )

    async def test_bearer_token_accepted(self):
        """Valid Bearer token should pass through."""
        from promptise.mcp.server._transport import _AuthGateASGI

        passed = False

        async def inner(scope, receive, send):
            nonlocal passed
            passed = True

        gate = _AuthGateASGI(inner, lambda t: t == "valid-token")

        scope = {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"authorization", b"Bearer valid-token")],
        }
        await gate(scope, None, MagicMock())
        assert passed

    async def test_bearer_token_rejected(self):
        """Invalid Bearer token should return 401."""
        from promptise.mcp.server._transport import _AuthGateASGI

        sent_status = None

        async def inner(scope, receive, send):
            pass  # should not be called

        async def mock_send(msg):
            nonlocal sent_status
            if msg.get("type") == "http.response.start":
                sent_status = msg["status"]

        gate = _AuthGateASGI(inner, lambda t: False)

        scope = {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"authorization", b"Bearer bad-token")],
        }
        await gate(scope, None, mock_send)
        assert sent_status == 401

    async def test_api_key_accepted(self):
        """Valid API key via x-api-key header should pass through."""
        from promptise.mcp.server._transport import _AuthGateASGI

        passed = False

        async def inner(scope, receive, send):
            nonlocal passed
            passed = True

        # api_key_verify_fn checks the key
        gate = _AuthGateASGI(
            inner,
            lambda t: False,  # JWT verify always fails
            api_key_verify_fn=lambda k: k == "my-secret",
        )

        scope = {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"x-api-key", b"my-secret")],
        }
        await gate(scope, None, MagicMock())
        assert passed

    async def test_api_key_rejected(self):
        """Invalid API key should return 401."""
        from promptise.mcp.server._transport import _AuthGateASGI

        sent_status = None

        async def inner(scope, receive, send):
            pass

        async def mock_send(msg):
            nonlocal sent_status
            if msg.get("type") == "http.response.start":
                sent_status = msg["status"]

        gate = _AuthGateASGI(
            inner,
            lambda t: False,
            api_key_verify_fn=lambda k: False,
        )

        scope = {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"x-api-key", b"wrong-key")],
        }
        await gate(scope, None, mock_send)
        assert sent_status == 401

    async def test_no_credentials_rejected(self):
        """Request with no auth headers should return 401."""
        from promptise.mcp.server._transport import _AuthGateASGI

        sent_status = None
        sent_body = None

        async def inner(scope, receive, send):
            pass

        async def mock_send(msg):
            nonlocal sent_status, sent_body
            if msg.get("type") == "http.response.start":
                sent_status = msg["status"]
            elif msg.get("type") == "http.response.body":
                sent_body = msg["body"]

        gate = _AuthGateASGI(inner, lambda t: True)

        scope = {
            "type": "http",
            "path": "/mcp",
            "headers": [],
        }
        await gate(scope, None, mock_send)
        assert sent_status == 401
        body = json.loads(sent_body)
        assert "Authentication required" in body["error"]

    async def test_skip_paths(self):
        """Whitelisted paths should bypass auth."""
        from promptise.mcp.server._transport import _AuthGateASGI

        passed = False

        async def inner(scope, receive, send):
            nonlocal passed
            passed = True

        gate = _AuthGateASGI(
            inner,
            lambda t: False,  # would reject everything
            skip_paths={"/auth/token"},
        )

        scope = {
            "type": "http",
            "path": "/auth/token",
            "headers": [],  # no auth headers
        }
        await gate(scope, None, MagicMock())
        assert passed

    async def test_lifespan_passes_through(self):
        """Lifespan events should pass through without auth checks."""
        from promptise.mcp.server._transport import _AuthGateASGI

        passed = False

        async def inner(scope, receive, send):
            nonlocal passed
            passed = True

        gate = _AuthGateASGI(inner, lambda t: False)

        scope = {"type": "lifespan"}
        await gate(scope, None, None)
        assert passed

    async def test_api_key_fallback_to_verify_fn(self):
        """When api_key_verify_fn is None, should fall back to verify_fn."""
        from promptise.mcp.server._transport import _AuthGateASGI

        passed = False

        async def inner(scope, receive, send):
            nonlocal passed
            passed = True

        # APIKeyAuth.verify_token checks dict lookup — simulate with lambda
        gate = _AuthGateASGI(
            inner,
            lambda t: t == "my-api-key",  # same fn handles both
            api_key_verify_fn=None,  # should fall back to verify_fn
        )

        scope = {
            "type": "http",
            "path": "/mcp",
            "headers": [(b"x-api-key", b"my-api-key")],
        }
        await gate(scope, None, MagicMock())
        assert passed


# =====================================================================
# Tool name collision warning
# =====================================================================


class TestToolNameCollisionWarning:
    """Test that MCPMultiClient logs a warning on tool name collisions."""

    async def test_collision_warning_logged(self, caplog):
        """Duplicate tool names across servers should log a warning."""
        from unittest.mock import AsyncMock

        # Create two mock clients that return a tool with the same name
        tool = MagicMock()
        tool.name = "do_stuff"

        c1 = MagicMock(spec=MCPClient)
        c1.list_tools = AsyncMock(return_value=[tool])
        c1.__aenter__ = AsyncMock(return_value=c1)
        c1.__aexit__ = AsyncMock(return_value=None)

        c2 = MagicMock(spec=MCPClient)
        c2.list_tools = AsyncMock(return_value=[tool])
        c2.__aenter__ = AsyncMock(return_value=c2)
        c2.__aexit__ = AsyncMock(return_value=None)

        multi = MCPMultiClient({"server1": c1, "server2": c2})
        multi._connected = True  # bypass connection

        with caplog.at_level(logging.WARNING, logger="promptise.mcp.client"):
            await multi.list_tools()

        assert "Tool name collision" in caplog.text
        assert "do_stuff" in caplog.text
        assert "server1" in caplog.text
        assert "server2" in caplog.text

        # The last server wins
        assert multi._tool_to_server["do_stuff"] == "server2"


class TestBearerTokenProvider:
    """``bearer_token_provider``: a fresh credential per request, fail closed."""

    def test_rejects_both_static_token_and_provider(self) -> None:
        with pytest.raises(MCPClientError, match="either bearer_token or bearer_token_provider"):
            MCPClient(url="http://x/mcp", bearer_token="t", bearer_token_provider=lambda f: "t")

    def test_rejects_stdio(self) -> None:
        with pytest.raises(MCPClientError, match="HTTP or SSE"):
            MCPClient(transport="stdio", command="x", bearer_token_provider=lambda f: "t")

    def test_static_authorization_header_is_dropped(self) -> None:
        client = MCPClient(
            url="http://x/mcp",
            headers={"Authorization": "Bearer stale", "x-trace": "1"},
            bearer_token_provider=lambda f: "fresh",
        )
        assert client.headers == {"x-trace": "1"}

    async def test_provider_failure_raises_credential_error_without_details(self) -> None:
        from promptise.mcp.client import MCPCredentialError
        from promptise.mcp.client._client import _TokenSource

        def provider(force: bool) -> str:
            raise RuntimeError("secret-token-value leaked by a careless provider")

        with pytest.raises(MCPCredentialError) as info:
            await _TokenSource(provider).current()
        assert "secret-token-value" not in str(info.value)
        assert "RuntimeError" in str(info.value)
        assert isinstance(info.value.__cause__, RuntimeError)

    async def test_non_string_token_is_rejected(self) -> None:
        from promptise.mcp.client import MCPCredentialError
        from promptise.mcp.client._client import _TokenSource

        with pytest.raises(MCPCredentialError, match="expected str"):
            await _TokenSource(lambda force: 42).current()  # type: ignore[arg-type,return-value]

    async def test_concurrent_401s_share_one_forced_refresh(self) -> None:
        import asyncio

        from promptise.mcp.client._client import _TokenSource

        calls: list[bool] = []

        async def provider(force: bool) -> str:
            calls.append(force)
            await asyncio.sleep(0)
            return "fresh" if force else "stale"

        source = _TokenSource(provider)
        assert await source.current() == "stale"
        results = await asyncio.gather(*(source.refresh(rejected="stale") for _ in range(5)))
        assert results == ["fresh"] * 5
        assert calls.count(True) == 1
