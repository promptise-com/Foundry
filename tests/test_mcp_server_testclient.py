"""Tests for promptise.mcp.server TestClient."""

from __future__ import annotations

import json

import pytest

from promptise.mcp.server import (
    Depends,
    HasRole,
    MCPRouter,
    MCPServer,
    RequestContext,
    TestClient,
)
from promptise.mcp.server._errors import AuthenticationError, ToolError

# =====================================================================
# Basic tool calling
# =====================================================================


class TestCallTool:
    async def test_simple_tool(self):
        server = MCPServer(name="test")

        @server.tool()
        async def add(a: int, b: int) -> int:
            return a + b

        client = TestClient(server)
        result = await client.call_tool("add", {"a": 1, "b": 2})
        assert len(result) == 1
        assert result[0].text == "3"

    async def test_string_result(self):
        server = MCPServer(name="test")

        @server.tool()
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        client = TestClient(server)
        result = await client.call_tool("greet", {"name": "World"})
        assert result[0].text == "Hello, World!"

    async def test_dict_result(self):
        server = MCPServer(name="test")

        @server.tool()
        async def info(key: str) -> dict:
            return {"key": key, "value": 42}

        client = TestClient(server)
        result = await client.call_tool("info", {"key": "test"})
        parsed = json.loads(result[0].text)
        assert parsed["key"] == "test"
        assert parsed["value"] == 42

    async def test_list_result(self):
        server = MCPServer(name="test")

        @server.tool()
        async def items() -> list:
            return [1, 2, 3]

        client = TestClient(server)
        result = await client.call_tool("items", {})
        parsed = json.loads(result[0].text)
        assert parsed == [1, 2, 3]

    async def test_none_result(self):
        server = MCPServer(name="test")

        @server.tool()
        async def noop() -> None:
            pass

        client = TestClient(server)
        result = await client.call_tool("noop", {})
        assert result[0].text == "OK"

    async def test_unknown_tool(self):
        server = MCPServer(name="test")
        client = TestClient(server)
        result = await client.call_tool("nonexistent", {})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "TOOL_NOT_FOUND"

    async def test_sync_handler(self):
        server = MCPServer(name="test")

        @server.tool()
        def multiply(a: int, b: int) -> int:
            return a * b

        client = TestClient(server)
        result = await client.call_tool("multiply", {"a": 3, "b": 4})
        assert result[0].text == "12"

    async def test_default_arguments(self):
        server = MCPServer(name="test")

        @server.tool()
        async def search(query: str, limit: int = 10) -> dict:
            return {"query": query, "limit": limit}

        client = TestClient(server)
        result = await client.call_tool("search", {"query": "test"})
        parsed = json.loads(result[0].text)
        assert parsed["limit"] == 10


# =====================================================================
# Input validation
# =====================================================================


class TestValidation:
    async def test_validation_error(self):
        server = MCPServer(name="test")

        @server.tool()
        async def add(a: int, b: int) -> int:
            return a + b

        client = TestClient(server)
        result = await client.call_tool("add", {"a": "not_an_int", "b": 2})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "VALIDATION_ERROR"

    async def test_missing_required_field(self):
        server = MCPServer(name="test")

        @server.tool()
        async def add(a: int, b: int) -> int:
            return a + b

        client = TestClient(server)
        result = await client.call_tool("add", {"a": 1})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "VALIDATION_ERROR"


# =====================================================================
# Error handling
# =====================================================================


class TestErrorHandling:
    async def test_mcp_error_serialised(self):
        server = MCPServer(name="test")

        @server.tool()
        async def fail(x: int) -> int:
            raise ToolError("Something broke", code="CUSTOM_ERROR", retryable=True)

        client = TestClient(server)
        result = await client.call_tool("fail", {"x": 1})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "CUSTOM_ERROR"
        assert parsed["error"]["retryable"] is True

    async def test_auth_error_serialised(self):
        server = MCPServer(name="test")

        @server.tool()
        async def protected(x: int) -> int:
            raise AuthenticationError("Not allowed")

        client = TestClient(server)
        result = await client.call_tool("protected", {"x": 1})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "AUTHENTICATION_ERROR"

    async def test_unhandled_exception_serialised(self):
        server = MCPServer(name="test")

        @server.tool()
        async def crash(x: int) -> int:
            raise RuntimeError("Unexpected failure")

        client = TestClient(server)
        result = await client.call_tool("crash", {"x": 1})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "INTERNAL_ERROR"
        # Parity with the live server: never leak the exception text
        assert "Unexpected failure" not in parsed["error"]["message"]
        assert parsed["error"]["message"] == "An internal error occurred."


# =====================================================================
# Dependency injection
# =====================================================================


class TestDependencyInjection:
    async def test_simple_dependency(self):
        server = MCPServer(name="test")

        def get_db():
            return {"type": "mock_db"}

        @server.tool()
        async def query(sql: str, db: dict = Depends(get_db)) -> dict:
            return {"sql": sql, "db": db["type"]}

        client = TestClient(server)
        result = await client.call_tool("query", {"sql": "SELECT 1"})
        parsed = json.loads(result[0].text)
        assert parsed["db"] == "mock_db"

    async def test_generator_dependency_cleanup(self):
        cleanup_called = False

        def get_conn():
            nonlocal cleanup_called
            yield "connection"
            cleanup_called = True

        server = MCPServer(name="test")

        @server.tool()
        async def use_conn(x: int, conn: str = Depends(get_conn)) -> str:
            return conn

        client = TestClient(server)
        result = await client.call_tool("use_conn", {"x": 1})
        assert result[0].text == "connection"
        assert cleanup_called is True

    async def test_async_dependency(self):
        server = MCPServer(name="test")

        async def get_config():
            return {"env": "test"}

        @server.tool()
        async def show_config(config: dict = Depends(get_config)) -> dict:
            return config

        client = TestClient(server)
        result = await client.call_tool("show_config", {})
        parsed = json.loads(result[0].text)
        assert parsed["env"] == "test"


# =====================================================================
# Guards
# =====================================================================


class TestGuards:
    async def test_guard_denies_access(self):
        server = MCPServer(name="test")

        @server.tool(guards=[HasRole("admin")])
        async def delete(x: int) -> int:
            return x

        client = TestClient(server)
        result = await client.call_tool("delete", {"x": 1})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "ACCESS_DENIED"

    async def test_guard_allows_with_role(self):
        server = MCPServer(name="test")

        @server.tool(roles=["admin"])
        async def delete(x: int) -> int:
            return x

        # We need to set roles on the context; simulate via middleware
        async def inject_roles(ctx, call_next):
            ctx.state["roles"] = {"admin", "user"}
            return await call_next(ctx)

        server.add_middleware(inject_roles)
        client = TestClient(server)
        result = await client.call_tool("delete", {"x": 42})
        assert result[0].text == "42"

    async def test_guard_denies_wrong_role(self):
        server = MCPServer(name="test")

        @server.tool(roles=["admin"])
        async def delete(x: int) -> int:
            return x

        async def inject_roles(ctx, call_next):
            ctx.state["roles"] = {"user"}
            return await call_next(ctx)

        server.add_middleware(inject_roles)
        client = TestClient(server)
        result = await client.call_tool("delete", {"x": 1})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "ACCESS_DENIED"


# =====================================================================
# Middleware
# =====================================================================


class TestMiddleware:
    async def test_server_middleware_runs(self):
        server = MCPServer(name="test")
        calls: list[str] = []

        async def track_mw(ctx, call_next):
            calls.append("before")
            result = await call_next(ctx)
            calls.append("after")
            return result

        server.add_middleware(track_mw)

        @server.tool()
        async def ping() -> str:
            return "pong"

        client = TestClient(server)
        result = await client.call_tool("ping", {})
        assert result[0].text == "pong"
        assert calls == ["before", "after"]

    async def test_router_middleware_runs(self):
        calls: list[str] = []

        async def router_mw(ctx, call_next):
            calls.append("router_before")
            result = await call_next(ctx)
            calls.append("router_after")
            return result

        server = MCPServer(name="test")
        router = MCPRouter(middleware=[router_mw])

        @router.tool()
        async def query(sql: str) -> str:
            return sql

        server.include_router(router)
        client = TestClient(server)
        result = await client.call_tool("query", {"sql": "SELECT 1"})
        assert result[0].text == "SELECT 1"
        assert "router_before" in calls
        assert "router_after" in calls

    async def test_middleware_order(self):
        """Server middleware runs before router middleware."""
        order: list[str] = []

        async def server_mw(ctx, call_next):
            order.append("server")
            return await call_next(ctx)

        async def router_mw(ctx, call_next):
            order.append("router")
            return await call_next(ctx)

        server = MCPServer(name="test")
        server.add_middleware(server_mw)
        router = MCPRouter(middleware=[router_mw])

        @router.tool()
        async def ping() -> str:
            return "pong"

        server.include_router(router)
        client = TestClient(server)
        await client.call_tool("ping", {})
        assert order == ["server", "router"]


# =====================================================================
# Meta / request context
# =====================================================================


class TestMeta:
    async def test_meta_passed_to_context(self):
        server = MCPServer(name="test")
        captured_meta: dict = {}

        @server.tool()
        async def check(ctx: RequestContext) -> str:
            captured_meta.update(ctx.meta)
            return "ok"

        client = TestClient(server, meta={"authorization": "Bearer xyz"})
        await client.call_tool("check", {})
        assert captured_meta["authorization"] == "Bearer xyz"


# =====================================================================
# Resource operations
# =====================================================================


class TestResources:
    async def test_read_static_resource(self):
        server = MCPServer(name="test")

        @server.resource("config://app")
        async def app_config() -> str:
            return '{"env": "test"}'

        client = TestClient(server)
        result = await client.read_resource("config://app")
        assert json.loads(result)["env"] == "test"

    async def test_read_template_resource(self):
        server = MCPServer(name="test")

        @server.resource_template("users://{user_id}")
        async def get_user(user_id: str) -> str:
            return f'{{"id": "{user_id}"}}'

        client = TestClient(server)
        result = await client.read_resource("users://42")
        assert json.loads(result)["id"] == "42"

    async def test_resource_not_found(self):
        server = MCPServer(name="test")
        client = TestClient(server)
        with pytest.raises(ValueError, match="Resource not found"):
            await client.read_resource("nonexistent://foo")

    async def test_list_resources(self):
        server = MCPServer(name="test")

        @server.resource("config://a")
        async def a() -> str:
            return "{}"

        @server.resource("config://b")
        async def b() -> str:
            return "{}"

        client = TestClient(server)
        resources = await client.list_resources()
        # The auto-registered manifest is listed, as on the live server
        uris = {str(r.uri) for r in resources}
        assert uris == {"config://a", "config://b", "docs://manifest"}

    async def test_list_resource_templates(self):
        server = MCPServer(name="test")

        @server.resource_template("users://{id}")
        async def get_user(id: str) -> str:
            return id

        client = TestClient(server)
        templates = await client.list_resource_templates()
        assert len(templates) == 1
        assert str(templates[0].uriTemplate) == "users://{id}"


# =====================================================================
# Prompt operations
# =====================================================================


class TestPrompts:
    async def test_get_prompt(self):
        server = MCPServer(name="test")

        @server.prompt()
        async def summarize(text: str) -> str:
            return f"Please summarize: {text}"

        client = TestClient(server)
        result = await client.get_prompt("summarize", {"text": "Hello world"})
        assert len(result.messages) == 1
        assert "Hello world" in result.messages[0].content.text

    async def test_prompt_not_found(self):
        server = MCPServer(name="test")
        client = TestClient(server)
        with pytest.raises(ValueError, match="Prompt not found"):
            await client.get_prompt("nonexistent")

    async def test_list_prompts(self):
        server = MCPServer(name="test")

        @server.prompt()
        async def summarize(text: str) -> str:
            return text

        @server.prompt()
        async def translate(text: str, language: str) -> str:
            return text

        client = TestClient(server)
        prompts = await client.list_prompts()
        assert len(prompts) == 2
        names = {p.name for p in prompts}
        assert "summarize" in names
        assert "translate" in names


# =====================================================================
# List tools
# =====================================================================


class TestListTools:
    async def test_list_tools(self):
        server = MCPServer(name="test")

        @server.tool()
        async def add(a: int, b: int) -> int:
            return a + b

        @server.tool()
        async def search(query: str) -> list:
            return []

        client = TestClient(server)
        tools = await client.list_tools()
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "add" in names
        assert "search" in names

    async def test_list_tools_includes_router(self):
        server = MCPServer(name="test")
        router = MCPRouter(prefix="db")

        @router.tool()
        async def query(sql: str) -> list:
            return []

        server.include_router(router)
        client = TestClient(server)
        tools = await client.list_tools()
        assert any(t.name == "db_query" for t in tools)


# =====================================================================
# Router-integrated tests (prefix, tags, guards through TestClient)
# =====================================================================


class TestRouterIntegration:
    async def test_router_prefixed_tool(self):
        server = MCPServer(name="test")
        router = MCPRouter(prefix="db")

        @router.tool()
        async def query(sql: str) -> str:
            return f"Executed: {sql}"

        server.include_router(router)
        client = TestClient(server)
        result = await client.call_tool("db_query", {"sql": "SELECT 1"})
        assert "Executed: SELECT 1" in result[0].text

    async def test_nested_router_tool(self):
        server = MCPServer(name="test")
        parent = MCPRouter(prefix="api")
        child = MCPRouter(prefix="v1")

        @child.tool()
        async def search(q: str) -> str:
            return q

        parent.include_router(child)
        server.include_router(parent)
        client = TestClient(server)
        result = await client.call_tool("api_v1_search", {"q": "test"})
        assert result[0].text == "test"

    async def test_router_guards_via_testclient(self):
        server = MCPServer(name="test")
        router = MCPRouter(guards=[HasRole("admin")])

        @router.tool()
        async def dangerous(x: int) -> int:
            return x

        server.include_router(router)
        client = TestClient(server)

        # No roles → denied
        result = await client.call_tool("dangerous", {"x": 1})
        parsed = json.loads(result[0].text)
        assert parsed["error"]["code"] == "ACCESS_DENIED"


# =====================================================================
# TestClient importable from testing module
# =====================================================================


class TestImports:
    def test_import_from_testing_module(self):
        from promptise.mcp.server.testing import TestClient as TC

        assert TC is TestClient

    def test_import_from_main_package(self):
        from promptise.mcp.server import TestClient as TC

        assert TC is TestClient
