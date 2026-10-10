"""MCP resources and prompts: middleware, access control, serialisation,
argument coercion, catch-all templates, client methods and agent tools.

Security regression: resource reads and prompt requests used to bypass the
whole middleware chain (logging, rate limits, audit, auth, roles, guards).
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
import uvicorn
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import (
    BlobResourceContents,
    PromptMessage,
    TextContent,
    TextResourceContents,
)
from pydantic import AnyUrl, BaseModel

from promptise.mcp.client import MCPClient, MCPClientError, MCPMultiClient
from promptise.mcp.server import (
    APIKeyAuth,
    AuditMiddleware,
    AuthenticationError,
    AuthMiddleware,
    CacheMiddleware,
    HasRole,
    MCPRouter,
    MCPServer,
    RateLimitError,
    TestClient,
    ValidationError,
)

pytestmark = pytest.mark.asyncio

KEYS = {
    "sk-admin": {"client_id": "alice", "roles": ["admin"]},
    "sk-user": {"client_id": "bob", "roles": ["user"]},
}


def _guarded_server(**server_kwargs: Any) -> MCPServer:
    server = MCPServer(name="guarded", **server_kwargs)
    server.add_middleware(AuthMiddleware(APIKeyAuth(keys=KEYS)))

    @server.resource("secrets://payroll", roles=["admin"])
    async def payroll() -> dict:
        return {"alice": 100, "bob": 90}

    @server.resource_template("secrets://employees/{emp_id}", roles=["admin"])
    async def employee(emp_id: int) -> dict:
        return {"id": emp_id}

    @server.resource("public://hello")
    async def hello() -> str:
        return "hello"

    @server.prompt(roles=["admin"])
    async def fire(name: str) -> str:
        """Write a termination letter."""
        return f"Fire {name}"

    @server.prompt(auth=True)
    async def greet(name: str) -> str:
        """Greet someone."""
        return f"Hello {name}"

    return server


# =====================================================================
# 1. Security: resources and prompts run through middleware and guards
# =====================================================================


class TestAccessControl:
    async def test_role_guarded_resource_denied_without_role(self):
        server = _guarded_server()
        client = TestClient(server, meta={"x-api-key": "sk-user"})
        with pytest.raises(AuthenticationError) as info:
            await client.read_resource("secrets://payroll")
        assert info.value.code == "ACCESS_DENIED"

    async def test_role_guarded_resource_denied_without_credentials(self):
        client = TestClient(_guarded_server())
        with pytest.raises(AuthenticationError):
            await client.read_resource("secrets://payroll")

    async def test_role_guarded_resource_allowed_with_role(self):
        client = TestClient(_guarded_server(), meta={"x-api-key": "sk-admin"})
        assert json.loads(await client.read_resource("secrets://payroll")) == {
            "alice": 100,
            "bob": 90,
        }

    async def test_role_guarded_template_denied_without_role(self):
        client = TestClient(_guarded_server(), meta={"x-api-key": "sk-user"})
        with pytest.raises(AuthenticationError):
            await client.read_resource("secrets://employees/7")

    async def test_unguarded_resource_still_public(self):
        client = TestClient(_guarded_server())
        assert await client.read_resource("public://hello") == "hello"

    async def test_role_guarded_prompt_denied_without_role(self):
        client = TestClient(_guarded_server(), meta={"x-api-key": "sk-user"})
        with pytest.raises(AuthenticationError):
            await client.get_prompt("fire", {"name": "bob"})

    async def test_auth_prompt_denied_without_credentials(self):
        client = TestClient(_guarded_server())
        with pytest.raises(AuthenticationError):
            await client.get_prompt("greet", {"name": "x"})
        allowed = TestClient(_guarded_server(), meta={"x-api-key": "sk-user"})
        result = await allowed.get_prompt("greet", {"name": "x"})
        assert result.messages[0].content.text == "Hello x"

    async def test_explicit_guards(self):
        server = MCPServer(name="s")
        server.add_middleware(AuthMiddleware(APIKeyAuth(keys=KEYS)))

        @server.resource("x://y", auth=True, guards=[HasRole("admin")])
        async def y() -> str:
            return "y"

        with pytest.raises(AuthenticationError):
            await TestClient(server, meta={"x-api-key": "sk-user"}).read_resource("x://y")
        assert (
            await TestClient(server, meta={"x-api-key": "sk-admin"}).read_resource("x://y") == "y"
        )

    async def test_require_auth_covers_resources_and_prompts(self):
        server = MCPServer(name="s", require_auth=True)
        server.add_middleware(AuthMiddleware(APIKeyAuth(keys=KEYS)))

        @server.resource("x://y")
        async def y() -> str:
            return "y"

        @server.prompt()
        async def p() -> str:
            return "p"

        client = TestClient(server)
        with pytest.raises(AuthenticationError):
            await client.read_resource("x://y")
        with pytest.raises(AuthenticationError):
            await client.get_prompt("p")

    async def test_router_guards_apply_to_router_resources_and_prompts(self):
        router = MCPRouter(auth=True, guards=[HasRole("admin")])

        @router.resource("r://doc")
        async def doc() -> str:
            return "doc"

        @router.prompt()
        async def rp() -> str:
            return "rp"

        server = MCPServer(name="s")
        server.add_middleware(AuthMiddleware(APIKeyAuth(keys=KEYS)))
        server.include_router(router)

        user = TestClient(server, meta={"x-api-key": "sk-user"})
        with pytest.raises(AuthenticationError):
            await user.read_resource("r://doc")
        with pytest.raises(AuthenticationError):
            await user.get_prompt("rp")
        admin = TestClient(server, meta={"x-api-key": "sk-admin"})
        assert await admin.read_resource("r://doc") == "doc"

    async def test_middleware_sees_request_type(self):
        seen: list[tuple[str, str]] = []

        async def record(ctx: Any, call_next: Any) -> Any:
            seen.append((ctx.request_type, ctx.tool_name))
            return await call_next(ctx)

        server = MCPServer(name="s")
        server.add_middleware(record)

        @server.resource("x://a")
        async def a() -> str:
            return "a"

        @server.prompt()
        async def b() -> str:
            return "b"

        client = TestClient(server)
        await client.read_resource("x://a")
        await client.get_prompt("b")
        assert seen == [("resource", "a"), ("prompt", "b")]

    async def test_declared_rate_limit_on_resource(self):
        server = MCPServer(name="s")

        @server.resource("x://limited", rate_limit="2/min")
        async def limited() -> str:
            return "ok"

        client = TestClient(server)
        await client.read_resource("x://limited")
        await client.read_resource("x://limited")
        with pytest.raises(RateLimitError):
            await client.read_resource("x://limited")

    async def test_audit_records_resource_reads(self):
        audit = AuditMiddleware()
        server = MCPServer(name="s")
        server.add_middleware(audit)

        @server.resource("x://a")
        async def a() -> str:
            return "a"

        await TestClient(server).read_resource("x://a")
        entries = audit.entries
        assert entries[-1]["request_type"] == "resource"
        assert entries[-1]["uri"] == "x://a"

    async def test_cache_hit_rechecks_guards(self):
        server = MCPServer(name="s")
        server.add_middleware(AuthMiddleware(APIKeyAuth(keys=KEYS)))
        server.add_middleware(CacheMiddleware(ttl=60))

        calls = 0

        @server.resource("x://secret", guards=[HasRole("admin")], auth=True)
        async def secret() -> str:
            nonlocal calls
            calls += 1
            return "s"

        admin = TestClient(server, meta={"x-api-key": "sk-admin"})
        assert await admin.read_resource("x://secret") == "s"
        assert await admin.read_resource("x://secret") == "s"
        assert calls == 1
        with pytest.raises(AuthenticationError):
            await TestClient(server, meta={"x-api-key": "sk-user"}).read_resource("x://secret")

    async def test_cache_before_auth_does_not_skip_authentication(self):
        server = MCPServer(name="s")
        server.add_middleware(CacheMiddleware(ttl=60))
        server.add_middleware(AuthMiddleware(APIKeyAuth(keys=KEYS)))

        @server.resource("x://secret", auth=True)
        async def secret() -> str:
            return "s"

        assert await TestClient(server, meta={"x-api-key": "sk-user"}).read_resource("x://secret")
        with pytest.raises(AuthenticationError):
            await TestClient(server).read_resource("x://secret")

    async def test_live_transport_denies_guarded_resource(self):
        """In memory, no credentials reach the server: the read is refused."""
        server = _guarded_server()
        ll = server._build_lowlevel_server()
        async with create_connected_server_and_client_session(ll) as session:
            with pytest.raises(McpError) as info:
                await session.read_resource(AnyUrl("secrets://payroll"))
            assert info.value.error.data["code"] == "AUTHENTICATION_ERROR"
            assert "payroll" not in str(info.value) and "alice" not in str(info.value)
            ok = await session.read_resource(AnyUrl("public://hello"))
            assert ok.contents[0].text == "hello"
            with pytest.raises(McpError):
                await session.get_prompt("fire", {"name": "x"})


# =====================================================================
# 2. Resource serialisation
# =====================================================================


class _Model(BaseModel):
    a: int


class TestSerialisation:
    async def _contents(self, server: MCPServer, uri: str) -> Any:
        ll = server._build_lowlevel_server()
        async with create_connected_server_and_client_session(ll) as session:
            return (await session.read_resource(AnyUrl(uri))).contents

    async def test_dict_is_json_with_inferred_mime(self):
        server = MCPServer(name="s")

        @server.resource("x://d")
        async def d() -> dict:
            return {"a": 1, "b": [1, 2]}

        (item,) = await self._contents(server, "x://d")
        assert isinstance(item, TextResourceContents)
        assert json.loads(item.text) == {"a": 1, "b": [1, 2]}
        assert item.mimeType == "application/json"

    async def test_declared_mime_type_is_kept(self):
        server = MCPServer(name="s")

        @server.resource("x://d", mime_type="application/vnd.custom+json")
        async def d():
            return {"a": 1}

        (item,) = await self._contents(server, "x://d")
        assert item.mimeType == "application/vnd.custom+json"
        assert json.loads(item.text) == {"a": 1}

    async def test_unannotated_list_is_json(self):
        server = MCPServer(name="s")

        @server.resource("x://l")
        async def lst():
            return [1, "two"]

        (item,) = await self._contents(server, "x://l")
        assert json.loads(item.text) == [1, "two"]
        assert item.mimeType == "application/json"

    async def test_bytes_are_a_base64_blob(self):
        server = MCPServer(name="s")
        payload = b"\x89PNG\x00\xff"

        @server.resource("x://img", mime_type="image/png")
        async def img() -> bytes:
            return payload

        (item,) = await self._contents(server, "x://img")
        assert isinstance(item, BlobResourceContents)
        assert item.mimeType == "image/png"
        assert base64.b64decode(item.blob) == payload

    async def test_bytes_without_mime_are_octet_stream(self):
        server = MCPServer(name="s")

        @server.resource("x://bin")
        async def binary() -> bytes:
            return b"abc"

        (item,) = await self._contents(server, "x://bin")
        assert item.mimeType == "application/octet-stream"
        client = TestClient(server)
        assert await client.read_resource("x://bin") == b"abc"

    async def test_pydantic_model_is_json(self):
        server = MCPServer(name="s")

        @server.resource("x://m")
        async def m() -> _Model:
            return _Model(a=3)

        (item,) = await self._contents(server, "x://m")
        assert json.loads(item.text) == {"a": 3}
        assert item.mimeType == "application/json"

    async def test_text_keeps_text_plain(self):
        server = MCPServer(name="s")

        @server.resource("x://t")
        def t() -> str:
            return "plain"

        (item,) = await self._contents(server, "x://t")
        assert item.text == "plain"
        assert item.mimeType == "text/plain"

    async def test_unknown_uri_is_a_protocol_error(self):
        server = MCPServer(name="s")
        ll = server._build_lowlevel_server()
        async with create_connected_server_and_client_session(ll) as session:
            with pytest.raises(McpError) as info:
                await session.read_resource(AnyUrl("x://missing"))
            assert info.value.error.code == -32002

    async def test_handler_exception_is_not_leaked(self):
        server = MCPServer(name="s")

        @server.resource("x://boom")
        async def boom() -> str:
            raise RuntimeError("postgres://user:secret@db")

        ll = server._build_lowlevel_server()
        async with create_connected_server_and_client_session(ll) as session:
            with pytest.raises(McpError) as info:
                await session.read_resource(AnyUrl("x://boom"))
            assert "secret" not in str(info.value)


# =====================================================================
# 4 + 5. Prompt results and argument coercion
# =====================================================================


class TestPrompts:
    async def test_list_of_prompt_messages(self):
        server = MCPServer(name="s")

        @server.prompt()
        async def convo(topic: str) -> list:
            return [
                PromptMessage(
                    role="user", content=TextContent(type="text", text=f"Tell me {topic}")
                ),
                PromptMessage(role="assistant", content=TextContent(type="text", text="Sure")),
            ]

        ll = server._build_lowlevel_server()
        async with create_connected_server_and_client_session(ll) as session:
            result = await session.get_prompt("convo", {"topic": "x"})
        assert [m.role for m in result.messages] == ["user", "assistant"]
        assert result.messages[0].content.text == "Tell me x"

    async def test_list_of_strings_and_dicts(self):
        server = MCPServer(name="s")

        @server.prompt()
        def mixed() -> list:
            return ["first", {"role": "assistant", "content": "second"}]

        result = await TestClient(server).get_prompt("mixed")
        assert [(m.role, m.content.text) for m in result.messages] == [
            ("user", "first"),
            ("assistant", "second"),
        ]

    async def test_plain_string(self):
        server = MCPServer(name="s")

        @server.prompt()
        def single() -> str:
            """Doc."""
            return "just text"

        result = await TestClient(server).get_prompt("single")
        assert result.messages[0].content.text == "just text"
        assert result.description == "Doc."

    async def test_invalid_return_is_rejected(self):
        server = MCPServer(name="s")

        @server.prompt()
        def bad() -> Any:
            return 42

        with pytest.raises(TypeError):
            await TestClient(server).get_prompt("bad")

    async def test_arguments_coerced_from_strings(self):
        server = MCPServer(name="s")
        seen: dict[str, Any] = {}

        @server.prompt()
        def typed(count: int, ratio: float, verbose: bool, tags: list[str]) -> str:
            seen.update(count=count, ratio=ratio, verbose=verbose, tags=tags)
            return "ok"

        ll = server._build_lowlevel_server()
        async with create_connected_server_and_client_session(ll) as session:
            await session.get_prompt(
                "typed",
                {"count": "3", "ratio": "0.5", "verbose": "true", "tags": '["a", "b"]'},
            )
        assert seen == {"count": 3, "ratio": 0.5, "verbose": True, "tags": ["a", "b"]}

    async def test_bad_argument_is_a_validation_error(self):
        server = MCPServer(name="s")

        @server.prompt()
        def typed(count: int) -> str:
            return str(count)

        with pytest.raises(ValidationError):
            await TestClient(server).get_prompt("typed", {"count": "many"})

    async def test_template_parameters_coerced(self):
        server = MCPServer(name="s")

        @server.resource_template("items://{item_id}/page/{page}")
        def item(item_id: str, page: int) -> dict:
            return {"item": item_id, "page": page, "type": type(page).__name__}

        data = json.loads(await TestClient(server).read_resource("items://abc/page/2"))
        assert data == {"item": "abc", "page": 2, "type": "int"}

    async def test_template_placeholder_without_parameter_is_rejected(self):
        server = MCPServer(name="s")
        with pytest.raises(ValueError, match="missing"):

            @server.resource_template("x://{missing}")
            def handler(other: str) -> str:
                return other


# =====================================================================
# 6. Catch-all template placeholders
# =====================================================================


class TestCatchAll:
    async def test_star_placeholder_matches_slashes(self):
        server = MCPServer(name="s")

        @server.resource_template("files://{path*}")
        def files(path: str) -> str:
            return path

        client = TestClient(server)
        assert await client.read_resource("files://docs/guides/setup.md") == "docs/guides/setup.md"

    async def test_plus_placeholder_matches_slashes(self):
        server = MCPServer(name="s")

        @server.resource_template("repo://{owner}/{+path}")
        def repo(owner: str, path: str) -> str:
            return f"{owner}|{path}"

        assert await TestClient(server).read_resource("repo://acme/src/a/b.py") == "acme|src/a/b.py"

    async def test_simple_placeholder_still_one_segment(self):
        server = MCPServer(name="s")

        @server.resource_template("one://{name}")
        def one(name: str) -> str:
            return name

        with pytest.raises(ValueError):
            await TestClient(server).read_resource("one://a/b")

    async def test_specific_template_wins_over_catch_all(self):
        server = MCPServer(name="s")

        @server.resource_template("docs://{path*}")
        def anything(path: str) -> str:
            return "catch-all"

        @server.resource_template("docs://{slug}/history")
        def history(slug: str) -> str:
            return f"history of {slug}"

        client = TestClient(server)
        assert await client.read_resource("docs://intro/history") == "history of intro"
        assert await client.read_resource("docs://a/b/c") == "catch-all"

    async def test_parameters_are_percent_decoded(self):
        server = MCPServer(name="s")

        @server.resource_template("cities://{name}")
        def city(name: str) -> str:
            return name

        assert await TestClient(server).read_resource("cities://caf%C3%A9") == "café"


# =====================================================================
# 7. list_changed notifications
# =====================================================================


class TestListChanged:
    async def test_capabilities_advertise_list_changed(self):
        server = MCPServer(name="s")

        @server.resource("x://a")
        def a() -> str:
            return "a"

        @server.prompt()
        def p() -> str:
            return "p"

        ll = server._build_lowlevel_server()
        async with create_connected_server_and_client_session(ll) as session:
            caps = session.get_server_capabilities()
        assert caps.resources.listChanged is True
        assert caps.prompts.listChanged is True

    async def test_registration_while_serving_notifies(self):
        server = MCPServer(name="s")

        @server.prompt()
        def p() -> str:
            return "p"

        received: list[str] = []
        got = asyncio.Event()

        async def handler(message: Any) -> None:
            root = getattr(message, "root", None)
            method = getattr(root, "method", None)
            if method:
                received.append(method)
                got.set()

        ll = server._build_lowlevel_server()
        async with create_connected_server_and_client_session(
            ll, message_handler=handler
        ) as session:
            await session.list_prompts()  # the server now knows the session

            @server.prompt()
            def q() -> str:
                return "q"

            await asyncio.wait_for(got.wait(), 5)
            names = {p.name for p in (await session.list_prompts()).prompts}
        assert "notifications/prompts/list_changed" in received
        assert names == {"p", "q"}


# =====================================================================
# 8. Manifest and include_prompts
# =====================================================================


class TestManifestAndIncludePrompts:
    async def test_manifest_listed_and_readable(self):
        server = MCPServer(name="s")

        @server.resource("x://a", roles=["admin"])
        def a() -> str:
            return "a"

        client = TestClient(server)
        uris = {str(r.uri) for r in await client.list_resources()}
        assert "docs://manifest" in uris
        manifest = json.loads(await client.read_resource("docs://manifest"))
        res = next(r for r in manifest["resources"] if r["uri"] == "x://a")
        assert res["roles"] == ["admin"]
        assert res["auth_required"] is True

    async def test_include_prompts_dedents_and_describes_arguments(self):
        from promptise.prompts.core import prompt

        @prompt(model="openai:gpt-5-mini", description="Summarise a text.")
        async def summarize(text: str, max_words: int = 50) -> str:
            """Summarize the text below.

            Text: {text}
            Use at most {max_words} words.
            """

        server = MCPServer(name="s")
        server.include_prompts(summarize)
        pdef = server._prompt_registry.get("summarize")
        assert pdef.description.startswith("Summarise a text.")
        args = {a["name"]: a for a in pdef.arguments}
        assert args["text"]["description"] == "Fills {text} in the prompt (str)."
        assert (
            args["max_words"]["description"] == "Fills {max_words} in the prompt (int, default 50)."
        )
        assert args["max_words"]["required"] is False

        result = await TestClient(server).get_prompt(
            "summarize", {"text": "hello", "max_words": "10"}
        )
        text = result.messages[0].content.text
        assert "Text: hello" in text
        assert "Use at most 10 words." in text
        # The docstring's code indentation is not sent to the model
        assert "\n            " not in text
        assert "\nText: hello" in text


# =====================================================================
# 3 + 9. MCPClient / MCPMultiClient methods and agent tools (live HTTP)
# =====================================================================

# Local test servers stay inside the reserved range 8520-8529.
_TEST_PORTS = range(8520, 8530)


def _free_test_port() -> int:
    import socket

    for port in _TEST_PORTS:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port in 8520-8529 for the test MCP server")


@asynccontextmanager
async def _serve(server: MCPServer, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[str]:
    """Run *server* over Streamable HTTP on a port in 8520–8529."""
    instances: list[uvicorn.Server] = []

    class _Recording(uvicorn.Server):
        def __init__(self, config: uvicorn.Config) -> None:
            super().__init__(config)
            instances.append(self)

    monkeypatch.setattr(uvicorn, "Server", _Recording)
    port = _free_test_port()
    task = asyncio.ensure_future(server.run_async(transport="http", host="127.0.0.1", port=port))
    try:
        for _ in range(400):
            if task.done():
                task.result()
            if instances and instances[0].started:
                break
            await asyncio.sleep(0.025)
        else:
            raise RuntimeError("server did not start")
        instances[0].config.timeout_graceful_shutdown = 5
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        if instances:
            instances[0].should_exit = True
        try:
            await asyncio.wait_for(task, 15)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            task.cancel()


def _client_server() -> MCPServer:
    server = _guarded_server()

    @server.resource_template("files://{path*}", mime_type="text/markdown")
    def files(path: str) -> str:
        return f"# {path}"

    @server.resource("img://logo", mime_type="image/png")
    def logo() -> bytes:
        return b"\x89PNG"

    @server.prompt()
    def review(code: str, strict: bool = False) -> list:
        """Review code."""
        return [f"Review: {code}", {"role": "assistant", "content": f"strict={strict}"}]

    return server


class TestLiveClient:
    async def test_client_methods_and_access_control(self, monkeypatch):
        async with _serve(_client_server(), monkeypatch) as url:
            async with MCPClient(url=url, api_key="sk-user") as user:
                resources = {str(r.uri) for r in await user.list_resources()}
                assert {"secrets://payroll", "public://hello", "docs://manifest"} <= resources
                templates = {t.uriTemplate for t in await user.list_resource_templates()}
                assert "files://{path*}" in templates
                prompts = {p.name for p in await user.list_prompts()}
                assert {"fire", "greet", "review"} <= prompts

                # Role-guarded resource and prompt are denied over HTTP
                with pytest.raises(MCPClientError):
                    await user.read_resource("secrets://payroll")
                with pytest.raises(MCPClientError):
                    await user.get_prompt("fire", {"name": "bob"})

                md = await user.read_resource("files://guides/setup.md")
                assert md.contents[0].text == "# guides/setup.md"
                assert md.contents[0].mimeType == "text/markdown"

                blob = (await user.read_resource("img://logo")).contents[0]
                assert base64.b64decode(blob.blob) == b"\x89PNG"
                assert blob.mimeType == "image/png"

                result = await user.get_prompt("review", {"code": "x = 1", "strict": True})
                assert [m.role for m in result.messages] == ["user", "assistant"]
                assert result.messages[1].content.text == "strict=True"

            async with MCPClient(url=url, api_key="sk-admin") as admin:
                payroll = await admin.read_resource("secrets://payroll")
                assert json.loads(payroll.contents[0].text)["alice"] == 100
                assert payroll.contents[0].mimeType == "application/json"
                emp = await admin.read_resource("secrets://employees/7")
                assert json.loads(emp.contents[0].text) == {"id": 7}

            # Multi-client routing and the agent's context tools
            from promptise.mcp.client._context_tools import (
                make_prompt_tools,
                make_resource_tools,
            )

            other = MCPServer(name="other")

            @other.resource("other://note")
            def note() -> str:
                return "from other"

            async with _serve(other, monkeypatch) as other_url:
                multi = MCPMultiClient(
                    {
                        "main": MCPClient(url=url, api_key="sk-admin"),
                        "other": MCPClient(url=other_url),
                    }
                )
                async with multi:
                    # Lazy discovery routes by URI and by template
                    note_res = await multi.read_resource("other://note")
                    assert note_res.contents[0].text == "from other"
                    page = await multi.read_resource("files://a/b")
                    assert page.contents[0].text == "# a/b"
                    with pytest.raises(MCPClientError):
                        await multi.read_resource("nowhere://x")
                    review = await multi.get_prompt("review", {"code": "y"})
                    assert review.messages[0].content.text == "Review: y"

                    tools = await make_resource_tools(multi, taken=set())
                    tools += await make_prompt_tools(multi, taken={"read_resource"})
                    by_name = {t.name: t for t in tools}
                    assert set(by_name) == {"list_resources", "read_resource", "get_prompt"}
                    assert "files://{path*}" in by_name["read_resource"].description
                    assert "review" in by_name["get_prompt"].description

                    text = await by_name["read_resource"].ainvoke({"uri": "other://note"})
                    assert text == "from other"
                    text = await by_name["read_resource"].ainvoke({"uri": "nowhere://x"})
                    assert text.startswith("Error:")
                    text = await by_name["get_prompt"].ainvoke(
                        {"name": "review", "arguments": {"code": "z"}}
                    )
                    assert "Review: z" in text and "[assistant]" in text

    async def test_context_tools_rename_on_clash(self, monkeypatch):
        from promptise.mcp.client._context_tools import make_resource_tools

        server = MCPServer(name="s")

        @server.resource("x://a")
        def a() -> str:
            return "a"

        async with _serve(server, monkeypatch) as url:
            async with MCPMultiClient({"s": MCPClient(url=url)}) as multi:
                tools = await make_resource_tools(multi, taken={"read_resource"})
        assert {t.name for t in tools} == {"list_resources", "mcp_read_resource"}


# =====================================================================
# Agent tools send the invoking caller's token (forward_caller_token)
# =====================================================================


def _jwt_role_server() -> tuple[MCPServer, Any]:
    from promptise.mcp.server import JWTAuth

    jwt = JWTAuth(secret="resources-prompts-test-secret")
    server = MCPServer(name="hr")
    server.add_middleware(AuthMiddleware(jwt))

    @server.resource("hr://payroll", roles=["hr"])
    async def payroll() -> dict:
        return {"total": 1000}

    @server.prompt(roles=["hr"])
    async def dismissal(name: str) -> str:
        return f"Draft a dismissal letter for {name}"

    @server.tool()
    async def ping() -> str:
        return "pong"

    return server, jwt


class _ReadThenAnswer:
    """Fake chat model: calls one tool once, then answers with its result."""

    @staticmethod
    def make(tool: str, args: dict[str, Any]) -> Any:
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, ToolMessage
        from langchain_core.outputs import ChatGeneration, ChatResult

        class _Model(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "read-then-answer"

            def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                results = [m for m in messages if isinstance(m, ToolMessage)]
                if results:
                    message = AIMessage(content=str(results[-1].content))
                else:
                    message = AIMessage(
                        content="", tool_calls=[{"name": tool, "args": args, "id": "c1"}]
                    )
                return ChatResult(generations=[ChatGeneration(message=message)])

        return _Model()


class TestAgentToolsForwardCallerToken:
    async def _answer(self, agent: Any, caller: Any) -> str:
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "go"}]}, caller=caller
        )
        return str(result["messages"][-1].content)

    async def test_role_guarded_resource_judged_by_the_caller(self, monkeypatch):
        from promptise import CallerContext
        from promptise.agent import build_agent
        from promptise.config import HTTPServerSpec

        server, jwt = _jwt_role_server()
        # The agent's own credential HAS the role: a caller without it must
        # still be denied, so the read cannot be using the agent's identity.
        agent_token = jwt.create_token({"sub": "hr-agent", "roles": ["hr"]})
        clerk = CallerContext(
            user_id="clerk", bearer_token=jwt.create_token({"sub": "clerk", "roles": ["staff"]})
        )
        manager = CallerContext(
            user_id="manager", bearer_token=jwt.create_token({"sub": "manager", "roles": ["hr"]})
        )

        async with _serve(server, monkeypatch) as url:
            agent = await build_agent(
                model=_ReadThenAnswer.make("read_resource", {"uri": "hr://payroll"}),
                servers={"hr": HTTPServerSpec(url=url, bearer_token=agent_token)},
                expose_resources=True,
            )
            try:
                denied = await self._answer(agent, clerk)
                allowed = await self._answer(agent, manager)
            finally:
                await agent.shutdown()

        assert denied.startswith("Error:")
        assert "1000" not in denied
        assert json.loads(allowed) == {"total": 1000}

    async def test_role_guarded_prompt_judged_by_the_caller(self, monkeypatch):
        from promptise import CallerContext
        from promptise.agent import build_agent
        from promptise.config import HTTPServerSpec

        server, jwt = _jwt_role_server()
        agent_token = jwt.create_token({"sub": "hr-agent", "roles": ["hr"]})
        clerk = CallerContext(
            user_id="clerk", bearer_token=jwt.create_token({"sub": "clerk", "roles": ["staff"]})
        )
        manager = CallerContext(
            user_id="manager", bearer_token=jwt.create_token({"sub": "manager", "roles": ["hr"]})
        )
        model = _ReadThenAnswer.make(
            "get_prompt", {"name": "dismissal", "arguments": {"name": "Bob"}}
        )
        async with _serve(server, monkeypatch) as url:
            agent = await build_agent(
                model=model,
                servers={"hr": HTTPServerSpec(url=url, bearer_token=agent_token)},
                expose_prompts=True,
            )
            try:
                denied = await self._answer(agent, clerk)
                allowed = await self._answer(agent, manager)
            finally:
                await agent.shutdown()

        assert denied.startswith("Error:")
        assert allowed == "Draft a dismissal letter for Bob"

    async def test_multi_client_reads_as_bearer_token(self, monkeypatch):
        server, jwt = _jwt_role_server()
        agent_token = jwt.create_token({"sub": "hr-agent", "roles": ["hr"]})
        clerk_token = jwt.create_token({"sub": "clerk", "roles": ["staff"]})
        async with _serve(server, monkeypatch) as url:
            async with MCPMultiClient({"hr": MCPClient(url=url, bearer_token=agent_token)}) as m:
                ok = await m.read_resource("hr://payroll")
                assert json.loads(ok.contents[0].text) == {"total": 1000}
                with pytest.raises(MCPClientError):
                    await m.read_resource("hr://payroll", bearer_token=clerk_token)
                with pytest.raises(MCPClientError):
                    await m.get_prompt("dismissal", {"name": "x"}, bearer_token=clerk_token)


# =====================================================================
# hide_unauthorized_tools filters resources, templates and prompts
# =====================================================================


def _hidden_server() -> MCPServer:
    server = MCPServer(name="s", hide_unauthorized_tools=True)
    server.add_middleware(AuthMiddleware(APIKeyAuth(keys=KEYS)))

    @server.resource("public://info")
    def info() -> str:
        return "i"

    @server.resource("secrets://payroll", roles=["admin"])
    def payroll() -> str:
        return "p"

    @server.resource_template("secrets://employees/{emp_id}", roles=["admin"])
    def employee(emp_id: str) -> str:
        return emp_id

    @server.resource_template("public://pages/{slug}")
    def page(slug: str) -> str:
        return slug

    @server.prompt()
    def hello() -> str:
        return "hi"

    @server.prompt(roles=["admin"])
    def fire() -> str:
        return "fire"

    return server


class TestHideUnauthorized:
    async def test_test_client_filters_per_caller(self):
        server = _hidden_server()
        user = TestClient(server, meta={"x-api-key": "sk-user"})
        admin = TestClient(server, meta={"x-api-key": "sk-admin"})
        anonymous = TestClient(server)

        user_uris = {str(r.uri) for r in await user.list_resources()}
        assert "secrets://payroll" not in user_uris and "public://info" in user_uris
        assert "secrets://payroll" in {str(r.uri) for r in await admin.list_resources()}
        assert "secrets://payroll" not in {str(r.uri) for r in await anonymous.list_resources()}

        assert {t.uriTemplate for t in await user.list_resource_templates()} == {
            "public://pages/{slug}"
        }
        assert {t.uriTemplate for t in await admin.list_resource_templates()} == {
            "public://pages/{slug}",
            "secrets://employees/{emp_id}",
        }
        assert {p.name for p in await user.list_prompts()} == {"hello"}
        assert {p.name for p in await admin.list_prompts()} == {"hello", "fire"}

        manifest = json.loads(await user.read_resource("docs://manifest"))
        assert "secrets://payroll" not in {r["uri"] for r in manifest["resources"]}
        assert {p["name"] for p in manifest["prompts"]} == {"hello"}

    async def test_live_server_filters_per_caller(self, monkeypatch):
        async with _serve(_hidden_server(), monkeypatch) as url:
            async with MCPClient(url=url, api_key="sk-user") as user:
                uris = {str(r.uri) for r in await user.list_resources()}
                templates = {t.uriTemplate for t in await user.list_resource_templates()}
                prompts = {p.name for p in await user.list_prompts()}
            async with MCPClient(url=url, api_key="sk-admin") as admin:
                admin_uris = {str(r.uri) for r in await admin.list_resources()}
                admin_prompts = {p.name for p in await admin.list_prompts()}
        assert "secrets://payroll" not in uris and "public://info" in uris
        assert templates == {"public://pages/{slug}"}
        assert prompts == {"hello"}
        assert "secrets://payroll" in admin_uris
        assert admin_prompts == {"hello", "fire"}

    async def test_without_hiding_everything_is_listed(self):
        client = TestClient(_guarded_server(), meta={"x-api-key": "sk-user"})
        assert "secrets://payroll" in {str(r.uri) for r in await client.list_resources()}
        assert "fire" in {p.name for p in await client.list_prompts()}
