"""Transport security regressions, exercised over the REAL HTTP/SSE transports.

Two audit findings live here:

* Caller identity is resolved **per request**, never from the request that
  opened the MCP session: credentials, tenant, roles, peer address and
  ``X-Request-ID`` are read from the HTTP request that carries each
  ``tools/call``.  With the transport auth gate on, the session is also
  bound to the credential that created it, so a second credential on a
  known ``mcp-session-id`` is refused outright.
* A loopback bind validates ``Host`` and ``Origin`` (DNS rebinding
  protection) on both transports; an explicit allow-list replaces the
  loopback one.

Every server here binds ``127.0.0.1`` on an ephemeral port in-process and is
driven with raw JSON-RPC over httpx so that each request can carry its own
headers — which is exactly what an MCP client SDK never lets you vary.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import uvicorn

from promptise.mcp.server import (
    APIKeyAuth,
    AuthMiddleware,
    HasRole,
    JWTAuth,
    MCPServer,
    RequestContext,
)
from promptise.mcp.server._serve_cli import build_serve_parser, run_serve
from promptise.mcp.server._transport import (
    LOOPBACK_ALLOWED_HOSTS,
    LOOPBACK_ALLOWED_ORIGINS,
    build_transport_security,
    is_loopback_host,
    session_principal,
)

INIT_PARAMS = {
    "protocolVersion": "2025-03-26",
    "capabilities": {},
    "clientInfo": {"name": "transport-security-tests", "version": "0"},
}
KEY_A = {"x-api-key": "key-a"}
KEY_B = {"x-api-key": "key-b"}


# ---------------------------------------------------------------------------
# Harness: a live server on a loopback port, stopped cleanly afterwards
# ---------------------------------------------------------------------------


@dataclass
class _Live:
    base: str
    port: int


@asynccontextmanager
async def _serve(
    server: MCPServer, transport: str, monkeypatch: pytest.MonkeyPatch, **run_kwargs: Any
) -> AsyncIterator[_Live]:
    """Run *server* on ``127.0.0.1:<ephemeral>`` for the duration of the block."""
    instances: list[uvicorn.Server] = []

    class _Recording(uvicorn.Server):
        def __init__(self, config: uvicorn.Config) -> None:
            super().__init__(config)
            instances.append(self)

    monkeypatch.setattr(uvicorn, "Server", _Recording)
    task = asyncio.ensure_future(
        server.run_async(transport=transport, host="127.0.0.1", port=0, **run_kwargs)
    )
    try:
        for _ in range(400):
            if task.done():
                task.result()  # surfaces a startup failure
            if instances and instances[0].started:
                break
            await asyncio.sleep(0.025)
        else:
            raise RuntimeError("server did not start")
        instances[0].config.timeout_graceful_shutdown = 5
        port = instances[0].servers[0].sockets[0].getsockname()[1]
        yield _Live(base=f"http://127.0.0.1:{port}", port=port)
    finally:
        if instances:
            instances[0].should_exit = True
        try:
            await asyncio.wait_for(task, 15)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            task.cancel()


def _rpc_body(method: str, params: dict[str, Any], rid: int | None) -> dict[str, Any]:
    body: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params}
    if rid is not None:
        body["id"] = rid
    return body


def _sse_data(text: str) -> list[dict[str, Any]]:
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")]


def _tool_text(message: dict[str, Any]) -> str:
    return message["result"]["content"][0]["text"]


class _StreamableHTTP:
    """Raw Streamable HTTP client: one session, arbitrary headers per request."""

    def __init__(self, http: httpx.AsyncClient, base: str) -> None:
        self._http = http
        self.url = f"{base}/mcp"
        self.session_id: str | None = None

    async def post(
        self,
        method: str,
        params: dict[str, Any],
        headers: dict[str, str],
        *,
        rid: int | None = 1,
        session_id: str | None = None,
    ) -> httpx.Response:
        h = {
            "content-type": "application/json",
            "accept": "application/json, text/event-stream",
            **headers,
        }
        sid = session_id if session_id is not None else self.session_id
        if sid:
            h["mcp-session-id"] = sid
        return await self._http.post(self.url, json=_rpc_body(method, params, rid), headers=h)

    async def initialize(self, headers: dict[str, str]) -> httpx.Response:
        r = await self.post("initialize", INIT_PARAMS, headers)
        if r.status_code == 200:
            self.session_id = r.headers["mcp-session-id"]
            n = await self.post("notifications/initialized", {}, headers, rid=None)
            assert n.status_code == 202, n.text
        return r

    async def call(self, tool: str, headers: dict[str, str], **kw: Any) -> httpx.Response:
        return await self.post("tools/call", {"name": tool, "arguments": {}}, headers, rid=2, **kw)


class _SSE:
    """Raw SSE client: the ``/sse`` stream plus arbitrary-header ``/messages/`` POSTs."""

    def __init__(self, http: httpx.AsyncClient, base: str, stream_headers: dict[str, str]) -> None:
        self._http = http
        self._base = base
        self._stream_headers = stream_headers
        self.endpoint: str | None = None
        self._messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._reader: asyncio.Task[None] | None = None
        self._ready = asyncio.Event()
        self.stream_status: int | None = None

    async def __aenter__(self) -> _SSE:
        self._reader = asyncio.ensure_future(self._read())
        await asyncio.wait_for(self._ready.wait(), 10)
        return self

    async def __aexit__(self, *exc: object) -> None:
        assert self._reader is not None
        self._reader.cancel()
        try:
            await self._reader
        except (asyncio.CancelledError, Exception):
            pass

    async def _read(self) -> None:
        try:
            async with self._http.stream(
                "GET", f"{self._base}/sse", headers=self._stream_headers, timeout=None
            ) as resp:
                self.stream_status = resp.status_code
                if resp.status_code != 200:
                    self._ready.set()
                    return
                event = ""
                async for line in resp.aiter_lines():
                    if line.startswith("event: "):
                        event = line[7:].strip()
                    elif line.startswith("data: "):
                        data = line[6:]
                        if event == "endpoint":
                            self.endpoint = self._base + data.strip()
                            self._ready.set()
                        elif event == "message":
                            await self._messages.put(json.loads(data))
                    elif not line:
                        event = ""
        finally:
            self._ready.set()

    async def post(
        self, method: str, params: dict[str, Any], headers: dict[str, str], *, rid: int | None = 1
    ) -> httpx.Response:
        assert self.endpoint is not None
        return await self._http.post(
            self.endpoint,
            json=_rpc_body(method, params, rid),
            headers={"content-type": "application/json", **headers},
        )

    async def message(self) -> dict[str, Any]:
        return await asyncio.wait_for(self._messages.get(), 10)

    async def initialize(self, headers: dict[str, str]) -> dict[str, Any]:
        r = await self.post("initialize", INIT_PARAMS, headers)
        assert r.status_code == 202, r.text
        result = await self.message()
        n = await self.post("notifications/initialized", {}, headers, rid=None)
        assert n.status_code == 202, n.text
        return result

    async def call(self, tool: str, headers: dict[str, str]) -> httpx.Response:
        return await self.post("tools/call", {"name": tool, "arguments": {}}, headers, rid=2)


# ---------------------------------------------------------------------------
# Servers under test
# ---------------------------------------------------------------------------


def _echo_server(*, require_auth: bool = False, auth: bool = False) -> MCPServer:
    """A server whose ``whoami`` tool reports what the framework resolved for the call."""
    server = MCPServer(name="identity-probe", require_auth=require_auth)

    @server.tool(auth=auth)
    async def whoami(ctx: RequestContext) -> dict[str, Any]:
        return {
            "authorization": ctx.meta.get("authorization", "missing"),
            "api_key": ctx.meta.get("x-api-key", "missing"),
            "client_id": ctx.client.client_id,
            "tenant": ctx.client.tenant_id,
            "roles": sorted(ctx.client.roles),
            "request_id": ctx.request_id,
            "ip": ctx.client.ip_address,
        }

    return server


def _api_key_server(*, require_auth: bool) -> MCPServer:
    server = _echo_server(require_auth=require_auth, auth=True)
    server.add_middleware(
        AuthMiddleware(
            APIKeyAuth(
                keys={
                    "key-a": {
                        "client_id": "agent-a",
                        "tenant_id": "tenant-a",
                        "roles": ["approver"],
                    },
                    "key-b": {"client_id": "agent-b", "tenant_id": "tenant-b", "roles": []},
                }
            )
        )
    )

    @server.tool(auth=True, guards=[HasRole("approver")])
    async def approve() -> str:
        return "approved"

    return server


# ---------------------------------------------------------------------------
# H1 — identity is bound to the request, not the session
# ---------------------------------------------------------------------------


class TestStreamableHTTPIdentityIsPerRequest:
    async def test_headers_and_request_id_follow_each_call(self, monkeypatch):
        """Passthrough shape: the session was opened with ``Bearer A``; a later
        call with ``Bearer B`` must see B, and a header-less call must see nothing."""
        async with _serve(_echo_server(), "http", monkeypatch) as live:
            async with httpx.AsyncClient(timeout=10) as http:
                c = _StreamableHTTP(http, live.base)
                r = await c.initialize({"Authorization": "Bearer A", "x-request-id": "req-init"})
                assert r.status_code == 200

                seen = []
                for headers in (
                    {"Authorization": "Bearer A", "x-request-id": "req-a"},
                    {"Authorization": "Bearer B", "x-request-id": "req-b"},
                    {},
                ):
                    r = await c.call("whoami", headers)
                    assert r.status_code == 200, r.text
                    seen.append(json.loads(_tool_text(_sse_data(r.text)[0])))

        assert [s["authorization"] for s in seen] == ["Bearer A", "Bearer B", "missing"]
        assert [s["request_id"] for s in seen[:2]] == ["req-a", "req-b"]
        assert seen[2]["request_id"] not in {"req-init", "req-a", "req-b"}

    async def test_api_key_caller_runs_as_itself_on_a_foreign_session(self, monkeypatch):
        """Without the transport gate the session is shared; tenant B posting its
        own valid key to A's session must run as B — and B has no approver role."""
        async with _serve(_api_key_server(require_auth=False), "http", monkeypatch) as live:
            async with httpx.AsyncClient(timeout=10) as http:
                c = _StreamableHTTP(http, live.base)
                assert (await c.initialize(KEY_A)).status_code == 200

                a = json.loads(_tool_text(_sse_data((await c.call("whoami", KEY_A)).text)[0]))
                b = json.loads(_tool_text(_sse_data((await c.call("whoami", KEY_B)).text)[0]))
                assert (a["client_id"], a["tenant"], a["roles"]) == (
                    "agent-a",
                    "tenant-a",
                    ["approver"],
                )
                assert (b["client_id"], b["tenant"], b["roles"]) == ("agent-b", "tenant-b", [])
                assert a["ip"] == b["ip"] == "127.0.0.1"

                approve_a = _sse_data((await c.call("approve", KEY_A)).text)[0]
                approve_b = _sse_data((await c.call("approve", KEY_B)).text)[0]
                assert _tool_text(approve_a) == "approved"
                assert "approver" in _tool_text(approve_b)  # denied with the reason
                assert _tool_text(approve_b) != "approved"

                unauth = _sse_data((await c.call("whoami", {})).text)[0]
                assert "Missing API key" in _tool_text(unauth)

    async def test_jwt_subject_follows_each_call(self, monkeypatch):
        server = _echo_server(auth=True)
        jwt = JWTAuth(secret="per-request-secret")
        server.add_middleware(AuthMiddleware(jwt))
        token_a = jwt.create_token({"sub": "user-a", "tenant_id": "t-a"})
        token_b = jwt.create_token({"sub": "user-b", "tenant_id": "t-b"})

        async with _serve(server, "http", monkeypatch) as live:
            async with httpx.AsyncClient(timeout=10) as http:
                c = _StreamableHTTP(http, live.base)
                assert (
                    await c.initialize({"Authorization": f"Bearer {token_a}"})
                ).status_code == 200
                b = _sse_data(
                    (await c.call("whoami", {"Authorization": f"Bearer {token_b}"})).text
                )[0]
                assert json.loads(_tool_text(b))["client_id"] == "user-b"
                assert json.loads(_tool_text(b))["tenant"] == "t-b"

    async def test_transport_gate_binds_the_session_to_its_credential(self, monkeypatch):
        """With ``require_auth`` the SDK's session-owner check is armed: another
        valid credential on a known session id is answered as if the session did
        not exist, a header-less request is refused at the gate, and the owner
        keeps working."""
        async with _serve(_api_key_server(require_auth=True), "http", monkeypatch) as live:
            async with httpx.AsyncClient(timeout=10) as http:
                c = _StreamableHTTP(http, live.base)
                assert (await c.initialize(KEY_A)).status_code == 200

                foreign = await c.call("whoami", KEY_B)
                assert foreign.status_code == 404
                assert "Session not found" in foreign.text

                assert (await c.call("whoami", {})).status_code == 401

                approve_b = await c.call("approve", KEY_B)
                assert approve_b.status_code == 404

                own = await c.call("whoami", KEY_A)
                assert own.status_code == 200
                assert json.loads(_tool_text(_sse_data(own.text)[0]))["client_id"] == "agent-a"

                # B gets its own session and runs as B there
                other = _StreamableHTTP(http, live.base)
                assert (await other.initialize(KEY_B)).status_code == 200
                assert other.session_id != c.session_id
                mine = json.loads(
                    _tool_text(_sse_data((await other.call("whoami", KEY_B)).text)[0])
                )
                assert (mine["client_id"], mine["tenant"]) == ("agent-b", "tenant-b")

                # ...and cannot borrow A's session by naming it
                borrowed = await other.call("approve", KEY_B, session_id=c.session_id)
                assert borrowed.status_code == 404

    async def test_gate_binds_bearer_sessions_too(self, monkeypatch):
        server = _echo_server(require_auth=True, auth=True)
        jwt = JWTAuth(secret="gate-secret")
        server.add_middleware(AuthMiddleware(jwt))
        token_a = jwt.create_token({"sub": "user-a"})
        token_b = jwt.create_token({"sub": "user-b"})

        async with _serve(server, "http", monkeypatch) as live:
            async with httpx.AsyncClient(timeout=10) as http:
                c = _StreamableHTTP(http, live.base)
                assert (
                    await c.initialize({"Authorization": f"Bearer {token_a}"})
                ).status_code == 200
                assert (
                    await c.call("whoami", {"Authorization": f"Bearer {token_b}"})
                ).status_code == 404
                own = await c.call("whoami", {"Authorization": f"Bearer {token_a}"})
                assert own.status_code == 200


class TestSSEIdentityIsPerRequest:
    async def test_headers_follow_each_post(self, monkeypatch):
        async with _serve(_echo_server(), "sse", monkeypatch) as live:
            async with httpx.AsyncClient(timeout=10) as http:
                async with _SSE(http, live.base, {"Authorization": "Bearer STREAM"}) as s:
                    await s.initialize({"Authorization": "Bearer A"})
                    seen = []
                    for headers in (
                        {"Authorization": "Bearer B", "x-request-id": "req-b"},
                        {},
                    ):
                        assert (await s.call("whoami", headers)).status_code == 202
                        seen.append(json.loads(_tool_text(await s.message())))
        assert [x["authorization"] for x in seen] == ["Bearer B", "missing"]
        assert seen[0]["request_id"] == "req-b"
        assert seen[1]["request_id"] != "req-b"

    async def test_api_key_caller_runs_as_itself_on_a_foreign_session(self, monkeypatch):
        async with _serve(_api_key_server(require_auth=False), "sse", monkeypatch) as live:
            async with httpx.AsyncClient(timeout=10) as http:
                async with _SSE(http, live.base, KEY_A) as s:
                    await s.initialize(KEY_A)
                    assert (await s.call("whoami", KEY_B)).status_code == 202
                    b = json.loads(_tool_text(await s.message()))
                    assert (b["client_id"], b["tenant"], b["roles"]) == ("agent-b", "tenant-b", [])
                    assert (await s.call("approve", KEY_B)).status_code == 202
                    assert "approver" in _tool_text(await s.message())

    async def test_transport_gate_binds_the_session_to_its_credential(self, monkeypatch):
        async with _serve(_api_key_server(require_auth=True), "sse", monkeypatch) as live:
            async with httpx.AsyncClient(timeout=10) as http:
                async with _SSE(http, live.base, KEY_A) as s:
                    await s.initialize(KEY_A)
                    assert (await s.call("whoami", KEY_B)).status_code == 404
                    assert (await s.call("whoami", {})).status_code == 401
                    assert (await s.call("whoami", KEY_A)).status_code == 202
                    own = json.loads(_tool_text(await s.message()))
                    assert own["client_id"] == "agent-a"


class TestSessionPrincipal:
    def test_fingerprint_never_collides_across_schemes_or_values(self):
        bearer = session_principal("bearer", "secret")
        key = session_principal("api-key", "secret")
        other = session_principal("bearer", "secret2")
        assert bearer.username != key.username != other.username
        assert bearer.username.startswith("bearer:") and key.username.startswith("api-key:")
        assert "secret" not in bearer.username  # a fingerprint, not the credential
        assert bearer.username == session_principal("bearer", "secret").username

    def test_direct_handler_invocation_keeps_the_contextvar_fallback(self):
        """Outside a transport (stdio, direct calls in tests) the contextvar bridge is the
        source of truth and is left untouched."""
        from promptise.mcp.server._context import (
            bind_transport_request,
            clear_request_client_info,
            clear_request_headers,
            get_request_client_info,
            get_request_headers,
            set_request_client_info,
            set_request_headers,
        )

        set_request_headers({"x-api-key": "from-contextvar"})
        set_request_client_info(("10.0.0.9", 4242))
        try:
            headers, client = bind_transport_request(None)
            assert headers == {"x-api-key": "from-contextvar"}
            assert client == ("10.0.0.9", 4242)
        finally:
            clear_request_headers()
            clear_request_client_info()
        assert get_request_headers() == {} and get_request_client_info() is None

    def test_per_message_request_wins_and_rebinds_the_contextvars(self):
        from starlette.requests import Request

        from promptise.mcp.server._context import (
            bind_transport_request,
            clear_request_client_info,
            clear_request_headers,
            get_request_client_info,
            get_request_headers,
            set_request_headers,
        )

        set_request_headers({"authorization": "Bearer SESSION-OPENER"})
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp",
            "query_string": b"",
            "headers": [(b"Authorization", b"Bearer THIS-CALL"), (b"X-Request-ID", b"r-1")],
            "client": ("192.0.2.7", 5151),
        }
        try:
            headers, client = bind_transport_request(Request(scope))
            assert headers == {"authorization": "Bearer THIS-CALL", "x-request-id": "r-1"}
            assert client == ("192.0.2.7", 5151)
            assert get_request_headers() == headers
            assert get_request_client_info() == client
        finally:
            clear_request_headers()
            clear_request_client_info()


# ---------------------------------------------------------------------------
# H2 — Host / Origin validation (DNS rebinding protection)
# ---------------------------------------------------------------------------


class TestTransportSecurityPolicy:
    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"])
    def test_loopback_hosts(self, host):
        assert is_loopback_host(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.5", "api.example.com", ""])
    def test_non_loopback_hosts(self, host):
        assert not is_loopback_host(host)

    def test_loopback_bind_is_protected_by_default(self):
        s = build_transport_security("127.0.0.1")
        assert s is not None and s.enable_dns_rebinding_protection
        assert set(LOOPBACK_ALLOWED_HOSTS) <= set(s.allowed_hosts)
        assert set(LOOPBACK_ALLOWED_ORIGINS) <= set(s.allowed_origins)
        assert "attacker.example:*" not in s.allowed_hosts

    def test_other_loopback_address_is_added_to_the_lists(self):
        s = build_transport_security("127.0.0.2")
        assert s is not None
        assert {"127.0.0.2", "127.0.0.2:*"} <= set(s.allowed_hosts)
        assert {"http://127.0.0.2:*", "https://127.0.0.2:*"} <= set(s.allowed_origins)
        v6 = build_transport_security("::1")
        assert v6 is not None and "[::1]:*" in v6.allowed_hosts
        assert v6.allowed_hosts.count("[::1]:*") == 1

    def test_explicit_lists_extend_the_loopback_names(self):
        """Behind a reverse proxy the backend binds loopback but sees the public
        ``Host``: the list adds it without taking the loopback names away from
        local clients and health checks."""
        s = build_transport_security(
            "127.0.0.1", allowed_hosts=["mcp.internal:*"], allowed_origins=["https://app.internal"]
        )
        assert s is not None
        assert "mcp.internal:*" in s.allowed_hosts
        assert set(LOOPBACK_ALLOWED_HOSTS) <= set(s.allowed_hosts)
        assert "https://app.internal" in s.allowed_origins
        assert set(LOOPBACK_ALLOWED_ORIGINS) <= set(s.allowed_origins)
        assert s.allowed_hosts.count("127.0.0.1:*") == 1  # no duplicates

    def test_public_bind_is_unrestricted_unless_hosts_are_named(self):
        assert build_transport_security("0.0.0.0") is None
        s = build_transport_security("0.0.0.0", allowed_hosts=["api.example.com"])
        assert s is not None and s.enable_dns_rebinding_protection
        assert s.allowed_hosts == ["api.example.com"] and s.allowed_origins == []
        both = build_transport_security(
            "0.0.0.0",
            allowed_hosts=["api.example.com"],
            allowed_origins=["https://app.example.com"],
        )
        assert both is not None and both.allowed_origins == ["https://app.example.com"]

    def test_origins_without_hosts_on_public_bind_is_refused(self):
        with pytest.raises(ValueError, match="allowed_origins requires allowed_hosts"):
            build_transport_security("0.0.0.0", allowed_origins=["https://app.example.com"])

    def test_empty_host_list_is_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            build_transport_security("127.0.0.1", allowed_hosts=[])

    async def test_run_async_refuses_a_bad_policy_before_binding(self, monkeypatch):
        server = MCPServer(name="policy")
        bound = []
        monkeypatch.setattr(
            "promptise.mcp.server._app.run_transport",
            lambda *a, **k: bound.append(k),
        )
        with pytest.raises(ValueError, match="allowed_origins requires allowed_hosts"):
            await server.run_async("http", host="0.0.0.0", port=1, allowed_origins=["https://x"])
        assert bound == []

    async def test_run_async_hands_the_policy_to_the_transport(self, monkeypatch):
        server = MCPServer(name="policy")
        seen: list[dict[str, Any]] = []

        async def fake_transport(*a: Any, **k: Any) -> None:
            seen.append(k)

        monkeypatch.setattr("promptise.mcp.server._app.run_transport", fake_transport)
        await server.run_async("http", host="127.0.0.1", port=0)
        await server.run_async("stdio")
        await server.run_async("http", host="0.0.0.0", port=0)
        await server.run_async("sse", host="0.0.0.0", port=0, allowed_hosts=["api.example.com"])
        assert seen[0]["security_settings"] is not None
        assert "127.0.0.1:*" in seen[0]["security_settings"].allowed_hosts
        assert seen[1]["security_settings"] is None
        assert seen[2]["security_settings"] is None
        assert seen[3]["security_settings"].allowed_hosts == ["api.example.com"]


class TestStreamableHTTPRejectsRebinding:
    async def test_loopback_bind_refuses_foreign_host_and_origin(self, monkeypatch):
        async with _serve(_echo_server(), "http", monkeypatch) as live:
            async with httpx.AsyncClient(timeout=10) as http:
                c = _StreamableHTTP(http, live.base)
                rebound_host = {"Host": f"attacker.example:{live.port}"}
                r = await c.post("initialize", INIT_PARAMS, rebound_host)
                assert r.status_code == 421
                assert "mcp-session-id" not in r.headers

                r = await c.post("initialize", INIT_PARAMS, {"Origin": "http://attacker.example"})
                assert r.status_code == 403

                # Normal loopback traffic is unaffected — any of the loopback names
                for headers in (
                    {},
                    {"Host": f"localhost:{live.port}"},
                    {"Origin": f"http://localhost:{live.port}"},
                ):
                    fresh = _StreamableHTTP(http, live.base)
                    assert (await fresh.initialize(headers)).status_code == 200, headers
                    assert (await fresh.call("whoami", headers)).status_code == 200

                # An established session is protected on every later call too
                assert (await c.initialize({})).status_code == 200
                assert (await c.call("whoami", rebound_host)).status_code == 421
                assert (
                    await c.call("whoami", {"Origin": "http://attacker.example:80"})
                ).status_code == 403
                assert (await c.call("whoami", {})).status_code == 200

    async def test_explicit_allow_list_is_honoured(self, monkeypatch):
        """The operator's list reaches the transport (the same path a public bind
        uses): the proxy-forwarded host and the app origin pass, everything else
        still fails, and the loopback names remain usable for local clients."""
        async with _serve(
            _echo_server(),
            "http",
            monkeypatch,
            allowed_hosts=["mcp.internal:*"],
            allowed_origins=["https://app.internal"],
        ) as live:
            async with httpx.AsyncClient(timeout=10) as http:
                c = _StreamableHTTP(http, live.base)
                proxied = {"Host": f"mcp.internal:{live.port}"}
                assert (await c.initialize(proxied)).status_code == 200
                assert (
                    await c.call("whoami", {**proxied, "Origin": "https://app.internal"})
                ).status_code == 200
                assert (
                    await c.call("whoami", {**proxied, "Origin": "https://evil.internal"})
                ).status_code == 403
                assert (
                    await c.call("whoami", {"Host": f"attacker.example:{live.port}"})
                ).status_code == 421
                assert (await c.call("whoami", {})).status_code == 200  # Host: 127.0.0.1
                assert (
                    await c.call("whoami", {"Host": f"localhost:{live.port}"})
                ).status_code == 200


class TestSSERejectsRebinding:
    async def test_loopback_bind_refuses_foreign_host_and_origin(self, monkeypatch):
        async with _serve(_echo_server(), "sse", monkeypatch) as live:
            async with httpx.AsyncClient(timeout=10) as http:
                async with _SSE(http, live.base, {"Host": f"attacker.example:{live.port}"}) as bad:
                    assert bad.stream_status == 421 and bad.endpoint is None
                async with _SSE(http, live.base, {"Origin": "http://attacker.example"}) as bad:
                    assert bad.stream_status == 403

                async with _SSE(http, live.base, {}) as s:
                    await s.initialize({})
                    r = await s.call("whoami", {"Host": f"attacker.example:{live.port}"})
                    assert r.status_code == 421
                    r = await s.call("whoami", {"Origin": "http://attacker.example"})
                    assert r.status_code == 403
                    assert (
                        await s.call("whoami", {"Origin": f"http://127.0.0.1:{live.port}"})
                    ).status_code == 202
                    assert "authorization" in _tool_text(await s.message())

    async def test_explicit_allow_list_is_honoured(self, monkeypatch):
        async with _serve(
            _echo_server(), "sse", monkeypatch, allowed_hosts=["mcp.internal:*"]
        ) as live:
            async with httpx.AsyncClient(timeout=10) as http:
                async with _SSE(http, live.base, {"Host": f"attacker.example:{live.port}"}) as bad:
                    assert bad.stream_status == 421
                proxied = {"Host": f"mcp.internal:{live.port}"}
                async with _SSE(http, live.base, proxied) as s:
                    await s.initialize(proxied)
                    assert (await s.call("whoami", proxied)).status_code == 202
                    assert "authorization" in _tool_text(await s.message())
                    assert (await s.call("whoami", {})).status_code == 202  # loopback still ok
                    assert "authorization" in _tool_text(await s.message())
                    assert (
                        await s.call("whoami", {"Host": f"attacker.example:{live.port}"})
                    ).status_code == 421


# ---------------------------------------------------------------------------
# CLI plumbing: the allow-lists reach ``MCPServer.run`` / ``hot_reload``
# ---------------------------------------------------------------------------


class TestServeCliAllowLists:
    def test_parser_collects_repeatable_lists(self):
        args = build_serve_parser().parse_args(
            [
                "app:server",
                "-t",
                "http",
                "--allowed-host",
                "api.example.com",
                "--allowed-host",
                "api.example.com:*",
                "--allowed-origin",
                "https://app.example.com",
            ]
        )
        assert args.allowed_hosts == ["api.example.com", "api.example.com:*"]
        assert args.allowed_origins == ["https://app.example.com"]
        assert build_serve_parser().parse_args(["app:server"]).allowed_hosts is None

    def test_run_serve_forwards_lists_only_when_given(self, monkeypatch):
        server = MCPServer(name="serve-lists")
        recorded: list[dict[str, Any]] = []
        monkeypatch.setattr(server, "run", lambda **kw: recorded.append(kw))

        run_serve(build_serve_parser().parse_args(["app:server", "-t", "http"]), server=server)
        assert recorded[-1] == {
            "transport": "http",
            "host": "127.0.0.1",
            "port": 8080,
            "dashboard": False,
        }

        run_serve(
            build_serve_parser().parse_args(
                [
                    "app:server",
                    "-t",
                    "http",
                    "--host",
                    "0.0.0.0",
                    "--allowed-host",
                    "api.example.com",
                ]
            ),
            server=server,
        )
        assert recorded[-1]["allowed_hosts"] == ["api.example.com"]
        assert "allowed_origins" not in recorded[-1]

    def test_hot_reload_child_passes_lists_to_run(self, monkeypatch):
        from promptise.mcp.server._hot_reload import hot_reload

        server = MCPServer(name="reload-lists")
        recorded: dict[str, Any] = {}
        monkeypatch.setattr(server, "run", lambda **kw: recorded.update(kw))
        monkeypatch.setenv("_PROMPTISE_RELOAD_CHILD", "1")
        hot_reload(
            server,
            transport="http",
            host="0.0.0.0",
            port=9000,
            allowed_hosts=["api.example.com"],
            allowed_origins=["https://app.example.com"],
        )
        assert recorded["allowed_hosts"] == ["api.example.com"]
        assert recorded["allowed_origins"] == ["https://app.example.com"]
