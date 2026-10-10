"""End-to-end tests for the Promptise MCP Server framework.

Exercises the full server pipeline — registration, validation, dependency
injection, guards, middleware chain, auth, caching, rate limiting,
concurrency, health checks, exception handling, manifest, and dashboard —
all through the TestClient without starting a transport.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from promptise.mcp.server import (
    APIKeyAuth,
    AuthMiddleware,
    CacheMiddleware,
    ConcurrencyLimiter,
    DashboardMiddleware,
    DashboardState,
    Depends,
    HealthCheck,
    InMemoryCache,
    JWTAuth,
    LoggingMiddleware,
    MCPRouter,
    MCPServer,
    MetricsCollector,
    MetricsMiddleware,
    RateLimitMiddleware,
    RequestContext,
    RequireAuth,
    RequireClientId,
    TestClient,
    TimeoutMiddleware,
    TokenBucketLimiter,
    ToolError,
    build_manifest,
    cached,
    get_context,
)

# =====================================================================
# 1. TestMCPServerCore — core registration and invocation (6 tests)
# =====================================================================


class TestMCPServerCore:
    """Core server functionality: registration and tool invocation via TestClient."""

    async def test_tool_registration_and_invocation(self):
        """@server.tool() registers a tool and TestClient can invoke it."""
        server = MCPServer(name="core-test")

        @server.tool()
        async def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        client = TestClient(server)
        result = await client.call_tool("add", {"a": 3, "b": 7})
        assert result[0].text == "10"

    async def test_resource_registration_and_read(self):
        """@server.resource() registers a resource readable via TestClient."""
        server = MCPServer(name="core-test")

        @server.resource("config://app", name="app_config", description="App config")
        async def get_config() -> str:
            return '{"env": "production"}'

        client = TestClient(server)
        content = await client.read_resource("config://app")
        parsed = json.loads(content)
        assert parsed["env"] == "production"

    async def test_prompt_registration_and_get(self):
        """@server.prompt() registers a prompt retrievable via TestClient."""
        server = MCPServer(name="core-test")

        @server.prompt()
        async def summarize(text: str) -> str:
            """Summarize text."""
            return f"Please summarize the following:\n{text}"

        client = TestClient(server)
        result = await client.get_prompt("summarize", {"text": "Hello world"})
        assert len(result.messages) == 1
        assert "Hello world" in result.messages[0].content.text

    async def test_tool_invocation_via_testclient(self):
        """TestClient call_tool exercises the full pipeline (validation, handler, serialisation)."""
        server = MCPServer(name="core-test")

        @server.tool()
        async def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello, {name}!"

        client = TestClient(server)
        result = await client.call_tool("greet", {"name": "Alice"})
        assert result[0].text == "Hello, Alice!"

    async def test_tool_with_typed_parameters(self):
        """Tools with typed parameters get validated and coerced."""
        server = MCPServer(name="core-test")

        @server.tool()
        async def calculate(x: float, y: float, operation: str = "add") -> dict:
            """Perform a calculation."""
            if operation == "add":
                return {"result": x + y}
            return {"result": x - y}

        client = TestClient(server)
        result = await client.call_tool("calculate", {"x": 1.5, "y": 2.5})
        parsed = json.loads(result[0].text)
        assert parsed["result"] == 4.0

    async def test_tool_returning_structured_data(self):
        """Tools returning dicts/lists are JSON-serialised in TextContent."""
        server = MCPServer(name="core-test")

        @server.tool()
        async def get_users() -> list:
            """Get all users."""
            return [
                {"id": 1, "name": "Alice"},
                {"id": 2, "name": "Bob"},
            ]

        client = TestClient(server)
        result = await client.call_tool("get_users", {})
        parsed = json.loads(result[0].text)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "Alice"
        assert parsed[1]["name"] == "Bob"


# =====================================================================
# 2. TestMCPRouter — router composition (4 tests)
# =====================================================================


class TestMCPRouter:
    """Router creation, composition, namespacing, and nesting."""

    async def test_router_creation_with_prefix(self):
        """MCPRouter with prefix prepends prefix to tool names."""
        server = MCPServer(name="router-test")
        router = MCPRouter(prefix="db")

        @router.tool()
        async def query(sql: str) -> str:
            """Execute SQL."""
            return f"Result: {sql}"

        server.include_router(router)
        client = TestClient(server)
        result = await client.call_tool("db_query", {"sql": "SELECT 1"})
        assert "Result: SELECT 1" in result[0].text

    async def test_include_router_composition(self):
        """Multiple routers compose into the same server."""
        server = MCPServer(name="router-test")

        math_router = MCPRouter(prefix="math")
        text_router = MCPRouter(prefix="text")

        @math_router.tool()
        async def add(a: int, b: int) -> int:
            return a + b

        @text_router.tool()
        async def upper(s: str) -> str:
            return s.upper()

        server.include_router(math_router)
        server.include_router(text_router)

        client = TestClient(server)
        r1 = await client.call_tool("math_add", {"a": 2, "b": 3})
        assert r1[0].text == "5"

        r2 = await client.call_tool("text_upper", {"s": "hello"})
        assert r2[0].text == "HELLO"

    async def test_tool_namespacing(self):
        """Router prefix creates proper tool namespacing visible in list_tools."""
        server = MCPServer(name="router-test")
        router = MCPRouter(prefix="api")

        @router.tool()
        async def search(q: str) -> str:
            return q

        server.include_router(router)
        client = TestClient(server)
        tools = await client.list_tools()
        tool_names = {t.name for t in tools}
        assert "api_search" in tool_names

    async def test_nested_routers(self):
        """Nested routers combine prefixes: parent_child_tool."""
        server = MCPServer(name="router-test")
        parent = MCPRouter(prefix="api")
        child = MCPRouter(prefix="v2")

        @child.tool()
        async def fetch(resource: str) -> str:
            return f"fetched: {resource}"

        parent.include_router(child)
        server.include_router(parent)

        client = TestClient(server)
        result = await client.call_tool("api_v2_fetch", {"resource": "users"})
        assert result[0].text == "fetched: users"


# =====================================================================
# 3. TestAuthentication — JWT, API key, middleware (6 tests)
# =====================================================================


class TestAuthentication:
    """JWT and API key authentication providers and middleware."""

    async def test_jwt_token_creation_and_validation(self):
        """JWTAuth creates tokens that can be verified via authenticate()."""
        auth = JWTAuth(secret="e2e-secret")
        token = auth.create_token({"sub": "agent-1", "roles": ["admin"]})

        ctx = RequestContext(
            server_name="test",
            meta={"authorization": f"Bearer {token}"},
        )
        client_id = await auth.authenticate(ctx)
        assert client_id == "agent-1"
        assert "admin" in ctx.state["_jwt_payload"]["roles"]

    async def test_jwt_invalid_token_rejected(self):
        """JWTAuth rejects tokens signed with a different secret."""
        auth = JWTAuth(secret="correct-secret")
        other = JWTAuth(secret="wrong-secret")
        bad_token = other.create_token({"sub": "agent"})

        ctx = RequestContext(
            server_name="test",
            meta={"authorization": f"Bearer {bad_token}"},
        )
        with pytest.raises(Exception, match="signature"):
            await auth.authenticate(ctx)

    async def test_apikey_valid_key(self):
        """APIKeyAuth accepts a registered key and returns the client_id."""
        auth = APIKeyAuth(keys={"secret-key-123": "client-alpha"})
        ctx = RequestContext(
            server_name="test",
            meta={"x-api-key": "secret-key-123"},
        )
        client_id = await auth.authenticate(ctx)
        assert client_id == "client-alpha"

    async def test_apikey_invalid_key_rejected(self):
        """APIKeyAuth rejects an unregistered key."""
        auth = APIKeyAuth(keys={"valid-key": "client"})
        ctx = RequestContext(
            server_name="test",
            meta={"x-api-key": "invalid-key"},
        )
        with pytest.raises(Exception, match="Invalid"):
            await auth.authenticate(ctx)

    async def test_auth_middleware_chain(self):
        """AuthMiddleware authenticates auth=True tools via the full pipeline."""
        server = MCPServer(name="auth-test")
        jwt = JWTAuth(secret="chain-secret")
        server.add_middleware(AuthMiddleware(jwt))

        @server.tool(auth=True)
        async def secret_data() -> str:
            return "classified"

        token = jwt.create_token({"sub": "agent-42"})
        client = TestClient(server, meta={"authorization": f"Bearer {token}"})
        result = await client.call_tool("secret_data", {})
        assert result[0].text == "classified"

    async def test_token_endpoint_flow(self):
        """enable_token_endpoint() configures token issuance for clients."""
        server = MCPServer(name="token-test")
        jwt = JWTAuth(secret="token-secret")
        server.add_middleware(AuthMiddleware(jwt))

        server.enable_token_endpoint(
            jwt_auth=jwt,
            clients={
                "my-agent": {"secret": "agent-pass", "roles": ["admin"]},
            },
        )

        # The token endpoint config should be set
        assert server._token_endpoint is not None
        assert server._token_endpoint.clients["my-agent"]["secret"] == "agent-pass"

        # Simulate what the token endpoint would do: create a token
        token = jwt.create_token({"sub": "my-agent", "roles": ["admin"]})
        # Verify the token works with the auth middleware
        client = TestClient(server, meta={"authorization": f"Bearer {token}"})

        @server.tool(auth=True, roles=["admin"])
        async def admin_action() -> str:
            return "admin done"

        # We need to set up roles via middleware — AuthMiddleware extracts them
        result = await client.call_tool("admin_action", {})
        assert result[0].text == "admin done"


# =====================================================================
# 4. TestGuards — access control guards (4 tests)
# =====================================================================


class TestGuards:
    """Guard enforcement: RequireAuth, HasRole, RequireClientId."""

    async def test_require_auth_blocks_unauthenticated(self):
        """RequireAuth guard blocks calls with no client_id."""
        server = MCPServer(name="guard-test")

        @server.tool(guards=[RequireAuth()])
        async def protected(x: int) -> int:
            return x

        client = TestClient(server)
        result = await client.call_tool("protected", {"x": 1})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "ACCESS_DENIED"

    async def test_has_role_allows_correct_role(self):
        """HasRole guard allows requests with the required role."""
        server = MCPServer(name="guard-test")

        async def inject_admin_role(ctx, call_next):
            ctx.state["roles"] = {"admin", "user"}
            return await call_next(ctx)

        server.add_middleware(inject_admin_role)

        @server.tool(roles=["admin"])
        async def delete_user(user_id: str) -> str:
            return f"Deleted {user_id}"

        client = TestClient(server)
        result = await client.call_tool("delete_user", {"user_id": "u-42"})
        assert result[0].text == "Deleted u-42"

    async def test_has_role_blocks_wrong_role(self):
        """HasRole guard blocks requests lacking the required role."""
        server = MCPServer(name="guard-test")

        async def inject_viewer_role(ctx, call_next):
            ctx.state["roles"] = {"viewer"}
            return await call_next(ctx)

        server.add_middleware(inject_viewer_role)

        @server.tool(roles=["admin"])
        async def delete_user(user_id: str) -> str:
            return f"Deleted {user_id}"

        client = TestClient(server)
        result = await client.call_tool("delete_user", {"user_id": "u-1"})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "ACCESS_DENIED"

    async def test_require_client_id_enforcement(self):
        """RequireClientId guard blocks calls with wrong client_id."""
        server = MCPServer(name="guard-test")

        @server.tool(guards=[RequireClientId("agent-alpha", "agent-beta")])
        async def internal(x: int) -> int:
            return x

        # No client_id set, should be denied
        client = TestClient(server)
        result = await client.call_tool("internal", {"x": 1})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "ACCESS_DENIED"


# =====================================================================
# 5. TestMiddleware — middleware chain and built-in middleware (6 tests)
# =====================================================================


class TestMiddleware:
    """Built-in middleware: logging, timeout, cache, rate limit, concurrency, ordering."""

    async def test_logging_middleware_captures_logs(self, caplog):
        """LoggingMiddleware logs tool calls with timing."""
        server = MCPServer(name="mw-test")
        server.add_middleware(LoggingMiddleware())

        @server.tool()
        async def ping() -> str:
            return "pong"

        client = TestClient(server)
        with caplog.at_level("INFO", logger="promptise.server.mw-test"):
            result = await client.call_tool("ping", {})
        assert result[0].text == "pong"
        # LoggingMiddleware logs at INFO level on success
        assert any("ping" in r.message and "completed" in r.message for r in caplog.records)

    async def test_timeout_middleware_raises_on_timeout(self):
        """TimeoutMiddleware raises ToolError when handler exceeds timeout."""
        server = MCPServer(name="mw-test")
        server.add_middleware(TimeoutMiddleware(default_timeout=0.05))

        @server.tool()
        async def slow_task() -> str:
            await asyncio.sleep(2.0)
            return "done"

        client = TestClient(server)
        result = await client.call_tool("slow_task", {})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "TIMEOUT"
        assert parsed["error"]["retryable"] is True

    async def test_cache_middleware_caches_results(self):
        """CacheMiddleware returns cached results on repeated calls."""
        call_count = 0

        server = MCPServer(name="mw-test")
        cache_backend = InMemoryCache(cleanup_interval=0)
        server.add_middleware(CacheMiddleware(cache_backend, ttl=60.0))

        @server.tool()
        async def expensive() -> dict:
            nonlocal call_count
            call_count += 1
            return {"count": call_count}

        client = TestClient(server)

        r1 = await client.call_tool("expensive", {})
        r2 = await client.call_tool("expensive", {})

        p1 = json.loads(r1[0].text)
        p2 = json.loads(r2[0].text)

        # Both calls should return count=1 (second was cached)
        assert p1["count"] == 1
        assert p2["count"] == 1
        assert call_count == 1

    async def test_rate_limit_middleware_blocks_excess_requests(self):
        """RateLimitMiddleware rejects requests exceeding the rate limit."""
        server = MCPServer(name="mw-test")
        limiter = TokenBucketLimiter(rate_per_minute=60, burst=2)
        server.add_middleware(RateLimitMiddleware(limiter))

        @server.tool()
        async def fast_action() -> str:
            return "ok"

        client = TestClient(server)

        # Exhaust the burst
        r1 = await client.call_tool("fast_action", {})
        assert r1[0].text == "ok"
        r2 = await client.call_tool("fast_action", {})
        assert r2[0].text == "ok"

        # Third call should be rate limited
        r3 = await client.call_tool("fast_action", {})
        parsed = json.loads(r3[0].text)
        assert parsed["error"]["code"] == "RATE_LIMIT_EXCEEDED"

    async def test_concurrency_limiter_limits_parallel_calls(self):
        """ConcurrencyLimiter rejects when max concurrent is reached."""
        server = MCPServer(name="mw-test")
        limiter = ConcurrencyLimiter(max_concurrent=1)
        server.add_middleware(limiter)

        barrier = asyncio.Event()

        @server.tool()
        async def blocking() -> str:
            await barrier.wait()
            return "done"

        client = TestClient(server)

        # Start first call (will block)
        task1 = asyncio.create_task(client.call_tool("blocking", {}))
        # Give the first task a moment to acquire the semaphore
        await asyncio.sleep(0.05)

        # Second call should be rejected because max_concurrent=1
        r2 = await client.call_tool("blocking", {})
        parsed = json.loads(r2[0].text)
        assert parsed["error"]["code"] == "CONCURRENCY_LIMIT_EXCEEDED"

        # Release the first task
        barrier.set()
        r1 = await task1
        assert r1[0].text == "done"

    async def test_middleware_chain_ordering(self):
        """Server middleware runs in registration order (outermost first)."""
        order: list[str] = []

        async def mw_first(ctx, call_next):
            order.append("first_before")
            result = await call_next(ctx)
            order.append("first_after")
            return result

        async def mw_second(ctx, call_next):
            order.append("second_before")
            result = await call_next(ctx)
            order.append("second_after")
            return result

        server = MCPServer(name="mw-test")
        server.add_middleware(mw_first)
        server.add_middleware(mw_second)

        @server.tool()
        async def ping() -> str:
            order.append("handler")
            return "pong"

        client = TestClient(server)
        await client.call_tool("ping", {})

        assert order == [
            "first_before",
            "second_before",
            "handler",
            "second_after",
            "first_after",
        ]


# =====================================================================
# 6. TestDependencyInjection — Depends() and context injection (3 tests)
# =====================================================================


class TestDependencyInjection:
    """Dependency injection via Depends() and RequestContext."""

    async def test_depends_injects_dependency(self):
        """Depends() resolves and injects a dependency into the handler."""
        server = MCPServer(name="di-test")

        def get_config():
            return {"db_host": "localhost", "db_port": 5432}

        @server.tool()
        async def show_config(config: dict = Depends(get_config)) -> dict:
            return config

        client = TestClient(server)
        result = await client.call_tool("show_config", {})
        parsed = json.loads(result[0].text)
        assert parsed["db_host"] == "localhost"
        assert parsed["db_port"] == 5432

    async def test_request_context_available_in_tool(self):
        """Handlers can declare a RequestContext parameter to receive it."""
        server = MCPServer(name="di-test")

        @server.tool()
        async def check_context(ctx: RequestContext) -> dict:
            return {
                "server_name": ctx.server_name,
                "tool_name": ctx.tool_name,
            }

        client = TestClient(server)
        result = await client.call_tool("check_context", {})
        parsed = json.loads(result[0].text)
        assert parsed["server_name"] == "di-test"
        assert parsed["tool_name"] == "check_context"

    async def test_get_context_returns_current_context(self):
        """get_context() returns the active RequestContext inside a handler."""
        server = MCPServer(name="di-test")
        captured_ctx: dict = {}

        @server.tool()
        async def capture() -> str:
            ctx = get_context()
            captured_ctx["server"] = ctx.server_name
            captured_ctx["tool"] = ctx.tool_name
            captured_ctx["request_id"] = ctx.request_id
            return "ok"

        client = TestClient(server)
        result = await client.call_tool("capture", {})
        assert result[0].text == "ok"
        assert captured_ctx["server"] == "di-test"
        assert captured_ctx["tool"] == "capture"
        assert len(captured_ctx["request_id"]) == 12  # secrets.token_hex(6)


# =====================================================================
# 7. TestHealthCheck — health probes (2 tests)
# =====================================================================


class TestHealthCheck:
    """Health and readiness probes via HealthCheck."""

    async def test_health_endpoint_returns_status(self):
        """HealthCheck registers liveness/readiness resources on the server."""
        server = MCPServer(name="health-test")
        health = HealthCheck()
        health.register_resources(server)

        client = TestClient(server)
        liveness = await client.read_resource("health://liveness")
        parsed = json.loads(liveness)
        assert parsed["status"] == "alive"
        assert "uptime_seconds" in parsed

    async def test_custom_health_check(self):
        """Custom health checks affect readiness status."""
        server = MCPServer(name="health-test")
        health = HealthCheck()

        # Add a check that fails
        health.add_check("database", lambda: False, required_for_ready=True)
        health.register_resources(server)

        client = TestClient(server)
        readiness = await client.read_resource("health://readiness")
        parsed = json.loads(readiness)
        assert parsed["status"] == "not_ready"
        assert parsed["checks"]["database"]["healthy"] is False


# =====================================================================
# 8. TestExceptionHandling — error handling (3 tests)
# =====================================================================


class TestExceptionHandling:
    """Exception handling: ToolError, custom handlers, unhandled errors."""

    async def test_tool_error_handled_gracefully(self):
        """ToolError raised in handler is serialised as structured JSON."""
        server = MCPServer(name="exc-test")

        @server.tool()
        async def fail_tool() -> str:
            raise ToolError(
                "Database connection failed",
                code="DB_ERROR",
                retryable=True,
                suggestion="Check database connectivity",
            )

        client = TestClient(server)
        result = await client.call_tool("fail_tool", {})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "DB_ERROR"
        assert parsed["error"]["retryable"] is True
        assert parsed["error"]["suggestion"] == "Check database connectivity"

    async def test_custom_exception_handler_registered(self):
        """Custom exception handlers map app exceptions to MCPError."""
        server = MCPServer(name="exc-test")

        class AppDatabaseError(Exception):
            pass

        @server.exception_handler(AppDatabaseError)
        async def handle_db(ctx, exc):
            return ToolError(
                f"DB Error: {exc}",
                code="APP_DB_ERROR",
                retryable=True,
            )

        @server.tool()
        async def query_db() -> str:
            raise AppDatabaseError("connection refused")

        client = TestClient(server)
        result = await client.call_tool("query_db", {})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "APP_DB_ERROR"
        assert "connection refused" in parsed["error"]["message"]

    async def test_unhandled_error_returns_error_response(self):
        """Unhandled exceptions become INTERNAL_ERROR responses."""
        server = MCPServer(name="exc-test")

        @server.tool()
        async def crash() -> str:
            raise RuntimeError("unexpected crash")

        client = TestClient(server)
        result = await client.call_tool("crash", {})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "INTERNAL_ERROR"
        # Same as the live server: the exception text stays in the log
        assert "unexpected crash" not in parsed["error"]["message"]
        assert parsed["error"]["message"] == "An internal error occurred."


# =====================================================================
# 9. TestManifest — server manifest discovery (2 tests)
# =====================================================================


class TestManifest:
    """Manifest: auto-discovery of tools, resources, and prompts."""

    def test_build_manifest_discovers_all(self):
        """build_manifest() captures all registered tools, resources, prompts."""
        server = MCPServer(name="manifest-test", version="2.0.0")

        @server.tool(tags=["math"])
        async def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        @server.resource("config://app")
        async def config() -> str:
            return "{}"

        @server.prompt()
        async def review(code: str) -> str:
            """Review code."""
            return code

        manifest = build_manifest(server)
        assert manifest["server"]["name"] == "manifest-test"
        assert manifest["server"]["version"] == "2.0.0"
        assert len(manifest["tools"]) == 1
        assert manifest["tools"][0]["name"] == "add"
        assert "math" in manifest["tools"][0]["tags"]
        assert len(manifest["resources"]) == 1
        assert len(manifest["prompts"]) == 1

    def test_manifest_includes_parameter_schemas(self):
        """Manifest tool entries include input_schema with parameter details."""
        server = MCPServer(name="manifest-test")

        @server.tool()
        async def search(query: str, limit: int = 10) -> list:
            """Search for items."""
            return []

        manifest = build_manifest(server)
        tool_entry = manifest["tools"][0]
        schema = tool_entry["input_schema"]
        assert "query" in schema["properties"]
        assert "limit" in schema["properties"]
        assert "query" in schema.get("required", [])


# =====================================================================
# 10. TestDashboard — dashboard and middleware (2 tests)
# =====================================================================


class TestDashboard:
    """Dashboard creation and DashboardMiddleware request tracking."""

    def test_dashboard_creation(self):
        """Dashboard can be created with DashboardState."""
        state = DashboardState(
            server_name="dash-test",
            version="1.0.0",
            transport="http",
            host="0.0.0.0",
            port=8080,
        )
        assert state.server_name == "dash-test"
        assert state.total_requests == 0
        assert state.total_errors == 0

    async def test_dashboard_middleware_tracks_requests(self):
        """DashboardMiddleware records request metrics in DashboardState."""
        state = DashboardState(server_name="dash-test", version="1.0.0")
        mw = DashboardMiddleware(state)

        server = MCPServer(name="dash-test")
        server.add_middleware(mw)

        @server.tool()
        async def ping() -> str:
            return "pong"

        @server.tool()
        async def fail() -> str:
            raise RuntimeError("boom")

        client = TestClient(server)

        # Successful call
        r1 = await client.call_tool("ping", {})
        assert r1[0].text == "pong"
        assert state.total_requests == 1
        assert state.total_errors == 0

        # Failed call — the DashboardMiddleware records it as an error,
        # but TestClient catches the exception and serialises it
        await client.call_tool("fail", {})
        # The DashboardMiddleware sees the exception propagate through it,
        # so total_errors should increment
        assert state.total_requests == 2
        assert state.total_errors == 1


# =====================================================================
# Bonus: Integration test combining multiple subsystems
# =====================================================================


class TestFullPipelineIntegration:
    """Integration test: auth + guards + middleware + DI through TestClient."""

    async def test_authenticated_guarded_tool_with_middleware(self):
        """Full pipeline: JWT auth -> logging -> role guard -> DI -> handler."""
        server = MCPServer(name="integration")
        jwt = JWTAuth(secret="integration-secret")
        server.add_middleware(AuthMiddleware(jwt))
        server.add_middleware(LoggingMiddleware())

        def get_db():
            return {"connection": "active"}

        @server.tool(auth=True, roles=["admin"])
        async def admin_query(
            sql: str,
            db: dict = Depends(get_db),
        ) -> dict:
            """Execute admin query."""
            return {"sql": sql, "db_status": db["connection"]}

        token = jwt.create_token({"sub": "admin-user", "roles": ["admin"]})
        client = TestClient(server, meta={"authorization": f"Bearer {token}"})

        result = await client.call_tool("admin_query", {"sql": "DELETE FROM logs"})
        parsed = json.loads(result[0].text)
        assert parsed["sql"] == "DELETE FROM logs"
        assert parsed["db_status"] == "active"

    async def test_unauthenticated_user_denied_by_full_pipeline(self):
        """Unauthenticated call to auth=True tool is denied."""
        server = MCPServer(name="integration")
        jwt = JWTAuth(secret="integration-secret")
        server.add_middleware(AuthMiddleware(jwt))

        @server.tool(auth=True)
        async def secret() -> str:
            return "classified"

        client = TestClient(server)
        result = await client.call_tool("secret", {})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "AUTHENTICATION_ERROR"

    async def test_metrics_middleware_records_calls(self):
        """MetricsMiddleware records call counts and latency."""
        server = MCPServer(name="integration")
        collector = MetricsCollector()
        server.add_middleware(MetricsMiddleware(collector))

        @server.tool()
        async def compute(x: int) -> int:
            return x * 2

        client = TestClient(server)
        await client.call_tool("compute", {"x": 5})
        await client.call_tool("compute", {"x": 10})

        snapshot = collector.snapshot()
        assert snapshot["tools"]["compute"]["calls"] == 2
        assert snapshot["tools"]["compute"]["errors"] == 0

    async def test_cached_decorator_on_tool(self):
        """@cached decorator caches handler results across calls."""
        call_count = 0
        cache_backend = InMemoryCache(cleanup_interval=0)

        server = MCPServer(name="integration")

        @server.tool()
        @cached(ttl=60.0, backend=cache_backend)
        async def expensive_compute(x: int) -> dict:
            nonlocal call_count
            call_count += 1
            return {"x": x, "computed": True}

        client = TestClient(server)
        r1 = await client.call_tool("expensive_compute", {"x": 42})
        r2 = await client.call_tool("expensive_compute", {"x": 42})

        p1 = json.loads(r1[0].text)
        p2 = json.loads(r2[0].text)

        assert p1 == p2
        assert call_count == 1  # Second call served from cache
