"""Cached tool results are scoped to the caller by default.

Regressions:

* ``CacheMiddleware`` keyed on the tool name alone (it read
  ``ctx.state["arguments"]``, which the server never sets), so every tenant
  — and every argument — got the first caller's result.
* ``@cached`` keyed on the function name plus every keyword argument.  With
  a ``ctx: RequestContext`` parameter the key held a fresh request id, so
  it never hit; without one, a handler that reads the caller through
  ``get_context()`` served one tenant's data to the next.
"""

from __future__ import annotations

import json
import logging

import pytest

from promptise.mcp.server import (
    AuthMiddleware,
    CacheMiddleware,
    HasRole,
    InMemoryCache,
    JWTAuth,
    MCPServer,
    RequestContext,
    TestClient,
    cached,
    get_context,
)

SECRET = "cache-scope-test-secret-0123456789abcdef"
AUTH = JWTAuth(secret=SECRET)

ACCOUNTS = {"acme": ["Initech", "Umbrella"], "globex": ["Hooli", "Pied Piper"]}


def _client(server: MCPServer, sub: str, org: str, **claims) -> TestClient:
    token = AUTH.create_token({"sub": sub, "org": org, **claims})
    return TestClient(server, meta={"authorization": f"Bearer {token}"})


async def _text(client: TestClient, tool: str, args: dict | None = None):
    """The tool's result: decoded JSON, or the raw text of a string result."""
    text = (await client.call_tool(tool, args or {}))[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _server(cache: CacheMiddleware | None = None) -> tuple[MCPServer, dict[str, int]]:
    calls = {"accounts": 0, "account": 0, "report": 0}
    server = MCPServer("crm", require_tenant=True)
    server.add_middleware(AuthMiddleware(AUTH, tenant_claim="org"))
    if cache is not None:
        server.add_middleware(cache)

    @server.tool(read_only_hint=True)
    async def list_accounts(ctx: RequestContext) -> list[str]:
        """Accounts in the caller's organisation."""
        calls["accounts"] += 1
        return ACCOUNTS[ctx.client.tenant_id]

    @server.tool(read_only_hint=True)
    async def get_account(name: str) -> str:
        """One account."""
        calls["account"] += 1
        return f"{get_context().client.tenant_id}/{name}"

    @server.tool(read_only_hint=True, guards=[HasRole("finance")])
    async def revenue_report(ctx: RequestContext) -> str:
        """Finance only."""
        calls["report"] += 1
        return f"revenue for {ctx.client.tenant_id}"

    return server, calls


# =====================================================================
# CacheMiddleware
# =====================================================================


class TestCacheMiddlewareScope:
    async def test_tenants_never_share_an_entry(self):
        server, calls = _server(CacheMiddleware(ttl=60))
        acme = _client(server, "alice", "acme")
        globex = _client(server, "bob", "globex")

        assert await _text(acme, "list_accounts") == ACCOUNTS["acme"]
        assert await _text(globex, "list_accounts") == ACCOUNTS["globex"]
        assert await _text(acme, "list_accounts") == ACCOUNTS["acme"]
        assert calls["accounts"] == 2  # the third call was a hit

    async def test_default_scope_is_the_client_not_the_tenant(self):
        server, calls = _server(CacheMiddleware(ttl=60))
        await _text(_client(server, "alice", "acme"), "list_accounts")
        await _text(_client(server, "carol", "acme"), "list_accounts")
        assert calls["accounts"] == 2

    async def test_same_subject_in_two_tenants_is_two_principals(self):
        server, calls = _server(CacheMiddleware(ttl=60))
        assert await _text(_client(server, "admin", "acme"), "list_accounts") == ACCOUNTS["acme"]
        assert (
            await _text(_client(server, "admin", "globex"), "list_accounts") == ACCOUNTS["globex"]
        )

    async def test_tenant_scope_shares_within_a_tenant_only(self):
        server, calls = _server(CacheMiddleware(ttl=60, scope="tenant"))
        await _text(_client(server, "alice", "acme"), "list_accounts")
        await _text(_client(server, "carol", "acme"), "list_accounts")
        assert await _text(_client(server, "bob", "globex"), "list_accounts") == ACCOUNTS["globex"]
        assert calls["accounts"] == 2

    async def test_shared_scope_is_an_explicit_opt_out(self):
        server, calls = _server(CacheMiddleware(ttl=60, scope="shared"))
        await _text(_client(server, "alice", "acme"), "get_account", {"name": "x"})
        # Same arguments, other tenant: served from the shared entry.
        assert await _text(_client(server, "bob", "globex"), "get_account", {"name": "x"}) == (
            "acme/x"
        )
        assert calls["account"] == 1

    async def test_arguments_are_part_of_the_key(self):
        server, calls = _server(CacheMiddleware(ttl=60))
        alice = _client(server, "alice", "acme")
        assert await _text(alice, "get_account", {"name": "a"}) == "acme/a"
        assert await _text(alice, "get_account", {"name": "b"}) == "acme/b"
        assert await _text(alice, "get_account", {"name": "a"}) == "acme/a"
        assert calls["account"] == 2

    async def test_cache_hit_still_runs_guards(self):
        server, calls = _server(CacheMiddleware(ttl=60, scope="tenant"))
        finance = _client(server, "fin", "acme", roles=["finance"])
        intern = _client(server, "intern", "acme")

        assert await _text(finance, "revenue_report") == "revenue for acme"
        denied = await _text(intern, "revenue_report")
        assert denied["error"]["code"] == "ACCESS_DENIED"
        assert calls["report"] == 1

    async def test_shared_backend_separates_servers(self):
        backend = InMemoryCache(cleanup_interval=0)
        one, _ = _server(CacheMiddleware(backend, ttl=60, scope="shared"))
        two = MCPServer("other")
        two.add_middleware(CacheMiddleware(backend, ttl=60, scope="shared"))

        @two.tool()
        async def get_account(name: str) -> str:
            """Same tool name, different server."""
            return f"other/{name}"

        await _text(_client(one, "alice", "acme"), "get_account", {"name": "x"})
        assert await _text(TestClient(two), "get_account", {"name": "x"}) == "other/x"

    async def test_cache_before_auth_is_bypassed_not_shared(self, caplog):
        calls = {"n": 0}
        server = MCPServer("crm", require_tenant=True)
        server.add_middleware(CacheMiddleware(ttl=60))  # wrong order
        server.add_middleware(AuthMiddleware(AUTH, tenant_claim="org"))

        @server.tool()
        async def list_accounts(ctx: RequestContext) -> list[str]:
            """Accounts."""
            calls["n"] += 1
            return ACCOUNTS[ctx.client.tenant_id]

        with caplog.at_level(logging.WARNING, logger="promptise.server"):
            assert (
                await _text(_client(server, "alice", "acme"), "list_accounts") == ACCOUNTS["acme"]
            )
            assert (
                await _text(_client(server, "bob", "globex"), "list_accounts") == ACCOUNTS["globex"]
            )
        assert calls["n"] == 2
        assert sum("before authentication" in r.getMessage() for r in caplog.records) == 1

    def test_unknown_scope_is_rejected(self):
        with pytest.raises(ValueError, match="scope"):
            CacheMiddleware(scope="global")  # type: ignore[arg-type]


# =====================================================================
# @cached
# =====================================================================


def _decorated_server(**cache_kwargs) -> tuple[MCPServer, dict[str, int]]:
    calls = {"names": 0, "lookup": 0}
    backend = InMemoryCache(cleanup_interval=0)
    server = MCPServer("crm", require_tenant=True)
    server.add_middleware(AuthMiddleware(AUTH, tenant_claim="org"))

    @server.tool()
    @cached(ttl=60, backend=backend, **cache_kwargs)
    async def account_names(ctx: RequestContext) -> list[str]:
        """Names, read through the ctx parameter."""
        calls["names"] += 1
        return ACCOUNTS[ctx.client.tenant_id]

    @server.tool()
    @cached(ttl=60, backend=backend, **cache_kwargs)
    async def lookup(name: str) -> str:
        """Reads the caller through get_context(), not a parameter."""
        calls["lookup"] += 1
        return f"{get_context().client.tenant_id}/{name}"

    return server, calls


class TestCachedDecoratorScope:
    async def test_ctx_parameter_no_longer_defeats_the_cache(self):
        server, calls = _decorated_server()
        alice = _client(server, "alice", "acme")
        assert await _text(alice, "account_names") == ACCOUNTS["acme"]
        assert await _text(alice, "account_names") == ACCOUNTS["acme"]
        assert calls["names"] == 1

    async def test_get_context_handlers_do_not_leak_across_tenants(self):
        server, calls = _decorated_server()
        assert await _text(_client(server, "alice", "acme"), "lookup", {"name": "x"}) == "acme/x"
        assert await _text(_client(server, "bob", "globex"), "lookup", {"name": "x"}) == "globex/x"
        assert calls["lookup"] == 2

    async def test_tenant_scope(self):
        server, calls = _decorated_server(scope="tenant")
        await _text(_client(server, "alice", "acme"), "account_names")
        await _text(_client(server, "carol", "acme"), "account_names")
        assert (
            await _text(_client(server, "bob", "globex"), "account_names") == (ACCOUNTS["globex"])
        )
        assert calls["names"] == 2

    async def test_custom_key_func_is_still_scoped(self):
        server, calls = _decorated_server(key_func=lambda name, kwargs: name)
        assert await _text(_client(server, "alice", "acme"), "lookup", {"name": "x"}) == "acme/x"
        assert await _text(_client(server, "bob", "globex"), "lookup", {"name": "y"}) == "globex/y"

    async def test_same_function_name_in_two_servers_does_not_collide(self):
        def build(label: str) -> MCPServer:
            server = MCPServer(label)

            @server.tool()
            @cached(ttl=60, scope="shared")
            async def describe() -> str:
                """Per-server answer."""
                return label

            return server

        assert await _text(TestClient(build("one")), "describe") == "one"
        assert await _text(TestClient(build("two")), "describe") == "two"

    def test_unknown_scope_is_rejected(self):
        with pytest.raises(ValueError, match="scope"):
            cached(scope="everyone")  # type: ignore[arg-type]
