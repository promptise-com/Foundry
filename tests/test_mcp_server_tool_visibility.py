"""``MCPServer(hide_unauthorized_tools=True)`` lists only callable tools.

By default every tool is listed to every client and guards apply on call.
A tool guarded with ``HasTenant("acme")`` was therefore advertised — name,
description, schema — to every other tenant.  With hiding enabled the list
and the ``docs://manifest`` resource are filtered per request.
"""

from __future__ import annotations

import json

from promptise.mcp.server import (
    AuthMiddleware,
    Guard,
    HasRole,
    HasTenant,
    JWTAuth,
    MCPRouter,
    MCPServer,
    RequestContext,
    TestClient,
)

SECRET = "visibility-test-secret-0123456789abcdef01"
AUTH = JWTAuth(secret=SECRET)


def _bearer(**claims) -> dict[str, str]:
    return {"authorization": f"Bearer {AUTH.create_token(claims)}"}


class _Explodes(Guard):
    async def check(self, ctx: RequestContext) -> bool:
        raise RuntimeError("guard backend down")


def _server(*, hide: bool = True, **kwargs) -> MCPServer:
    server = MCPServer("crm", hide_unauthorized_tools=hide, **kwargs)
    server.add_middleware(AuthMiddleware(AUTH, tenant_claim="org"))

    @server.tool()
    async def public_status() -> str:
        """No auth, no guards."""
        return "ok"

    @server.tool(auth=True)
    async def list_accounts() -> list:
        """Any signed-in caller."""
        return []

    @server.tool(auth=True, guards=[HasTenant("acme")])
    async def forecast_renewals() -> list:
        """Acme pilot feature."""
        return []

    @server.tool(auth=True, guards=[HasRole("admin")])
    async def delete_account(account_id: str) -> str:
        """Admins only."""
        return account_id

    @server.tool(auth=True, guards=[_Explodes()])
    async def flaky() -> str:
        """Guard raises."""
        return "x"

    return server


async def _names(server: MCPServer, meta: dict[str, str] | None = None) -> list[str]:
    return sorted(t.name for t in await TestClient(server, meta=meta).list_tools())


class TestHideUnauthorizedTools:
    async def test_each_tenant_sees_only_its_tools(self):
        server = _server()
        assert await _names(server, _bearer(sub="alice", org="acme")) == [
            "forecast_renewals",
            "list_accounts",
            "public_status",
        ]
        assert await _names(server, _bearer(sub="bob", org="globex")) == [
            "list_accounts",
            "public_status",
        ]

    async def test_roles_are_evaluated(self):
        names = await _names(_server(), _bearer(sub="root", org="globex", roles=["admin"]))
        assert "delete_account" in names

    async def test_unauthenticated_and_forged_callers_see_public_tools_only(self):
        server = _server()
        forged = JWTAuth(secret="wrong-secret-0123456789abcdef0123456789").create_token(
            {"sub": "x", "org": "acme"}
        )
        assert await _names(server) == ["public_status"]
        assert await _names(server, {"authorization": f"Bearer {forged}"}) == ["public_status"]

    async def test_a_guard_that_raises_hides_its_tool(self):
        names = await _names(_server(), _bearer(sub="alice", org="acme"))
        assert "flaky" not in names

    async def test_hidden_tools_are_still_refused_when_called(self):
        client = TestClient(_server(), meta=_bearer(sub="bob", org="globex"))
        error = json.loads((await client.call_tool("forecast_renewals"))[0].text)["error"]
        assert error["code"] == "ACCESS_DENIED"
        # The denial names the caller's tenant, never the tenants allowed.
        assert "acme" not in error["message"]
        assert "globex" in error["message"]

    async def test_require_tenant_hides_everything_from_tenantless_tokens(self):
        # require_tenant guards every tool, public_status included.
        server = _server(require_tenant=True)
        assert await _names(server, _bearer(sub="svc")) == []
        assert "public_status" in await _names(server, _bearer(sub="a", org="acme"))

    async def test_router_level_auth_middleware_is_used(self):
        server = MCPServer("crm", hide_unauthorized_tools=True)
        router = MCPRouter(middleware=[AuthMiddleware(AUTH, tenant_claim="org")])

        @router.tool(auth=True, guards=[HasTenant("acme")])
        async def pilot() -> str:
            """Acme only."""
            return "x"

        server.include_router(router)
        assert await _names(server, _bearer(sub="a", org="acme")) == ["pilot"]
        assert await _names(server, _bearer(sub="b", org="globex")) == []

    async def test_default_lists_every_tool(self):
        names = await _names(_server(hide=False), _bearer(sub="bob", org="globex"))
        assert names == [
            "delete_account",
            "flaky",
            "forecast_renewals",
            "list_accounts",
            "public_status",
        ]

    async def test_manifest_is_filtered_too(self):
        server = _server()
        server._build_lowlevel_server()  # registers docs://manifest

        async def manifest_tools(meta: dict[str, str]) -> list[str]:
            raw = await TestClient(server, meta=meta).read_resource("docs://manifest")
            return sorted(t["name"] for t in json.loads(raw)["tools"])

        assert "forecast_renewals" in await manifest_tools(_bearer(sub="a", org="acme"))
        assert "forecast_renewals" not in await manifest_tools(_bearer(sub="b", org="globex"))


class TestOverHTTP:
    async def test_live_tools_list_is_filtered_per_request(self, monkeypatch):
        from test_caller_token_forwarding import _serve

        from promptise.mcp.client import MCPClient

        async def names(url: str, **claims) -> list[str]:
            token = AUTH.create_token(claims)
            async with MCPClient(url=url, bearer_token=token) as client:
                return sorted(t.name for t in await client.list_tools())

        async with _serve(_server(), monkeypatch) as url:
            acme = await names(url, sub="alice", org="acme")
            globex = await names(url, sub="bob", org="globex")

        assert "forecast_renewals" in acme
        assert "forecast_renewals" not in globex
        assert "list_accounts" in globex
