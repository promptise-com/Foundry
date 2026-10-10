"""Streamable HTTP transport behaviour, over REAL servers on loopback ports.

* A client survives a server restart: the restarted server answers the old
  ``mcp-session-id`` with ``404``, and the client opens a new session and
  retries the call once (the MCP specification requires a new session on
  that ``404``).  ``MCPMultiClient`` then re-lists the server's tools.
* A URL that is not an MCP endpoint fails as ``MCPConnectionRejectedError``
  (404) instead of the SDK's "Session terminated".
* Browser clients: the default ``CORSConfig`` admits the MCP request headers
  and exposes ``mcp-session-id``; the preflight is answered before the auth
  gate.
* ``run()`` binds loopback by default; an unvalidated public bind warns, and
  so does a public bind without ``AuthMiddleware``.
* ``CORSConfig`` refuses a wildcard origin together with credentials.
* ``GET /health`` and ``GET /health/ready`` serve container probes.
* ``MCPServer.asgi_app()`` runs under any ASGI server.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import socket
import subprocess
import sys
import textwrap
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import pytest
import uvicorn
from sse_starlette.sse import AppStatus

from promptise.mcp.client import (
    MCPClient,
    MCPClientError,
    MCPConnectionRejectedError,
    MCPMultiClient,
    MCPToolAdapter,
)
from promptise.mcp.server import (
    APIKeyAuth,
    AuthMiddleware,
    CORSConfig,
    HealthCheck,
    MCPServer,
)
from promptise.mcp.server._transport import (
    MCP_CORS_ALLOW_HEADERS,
    MCP_CORS_EXPOSE_HEADERS,
    _log_transport_security,
    build_transport_security,
    run_http,
    run_sse,
)

TIMEOUT = 10.0
API_KEY = "key-123"
BROWSER = "https://app.example.com"
INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
        "protocolVersion": "2025-03-26",
        "capabilities": {},
        "clientInfo": {"name": "http-transport-tests", "version": "0"},
    },
}
MCP_HEADERS = {"accept": "application/json, text/event-stream", "content-type": "application/json"}


def _inventory(*, extra_tool: bool = False, require_auth: bool = False) -> MCPServer:
    """A small server; ``extra_tool`` stands for a redeploy that adds a tool."""
    server = MCPServer("inventory", require_auth=require_auth)
    if require_auth:
        server.add_middleware(AuthMiddleware(APIKeyAuth(keys={API_KEY: "agent"})))

    @server.tool()
    async def check_stock(sku: str) -> dict:
        """Units in stock for a product."""
        return {"sku": sku, "in_stock": 42}

    if extra_tool:

        @server.tool()
        async def reserve(sku: str) -> dict:
            """Reserve one unit."""
            return {"sku": sku, "reserved": True}

    return server


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _forget_sse_shutdown() -> None:
    """Clear sse-starlette's process-wide shutdown flag after a server stops.

    A uvicorn server stopped while an SSE stream is open sets
    ``AppStatus.should_exit`` for the whole process, and every later SSE
    response then closes at once.  A real restart is a new process; an
    in-process restart must reset the flag to behave like one.
    """
    AppStatus.should_exit = False


@asynccontextmanager
async def _serve(
    server: MCPServer,
    monkeypatch: pytest.MonkeyPatch,
    *,
    port: int = 0,
    transport: str = "http",
    **run_kwargs: Any,
) -> AsyncIterator[int]:
    """Run *server* through ``run_async`` on ``127.0.0.1``; yield the bound port."""
    instances: list[uvicorn.Server] = []

    class _Recording(uvicorn.Server):
        def __init__(self, config: uvicorn.Config) -> None:
            super().__init__(config)
            instances.append(self)

    monkeypatch.setattr(uvicorn, "Server", _Recording)
    task = asyncio.ensure_future(
        server.run_async(transport=transport, host="127.0.0.1", port=port, **run_kwargs)
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
        instances[0].config.timeout_graceful_shutdown = 2
        yield instances[0].servers[0].sockets[0].getsockname()[1]
    finally:
        if instances:
            instances[0].should_exit = True
        try:
            await asyncio.wait_for(task, 15)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            task.cancel()
        _forget_sse_shutdown()


@asynccontextmanager
async def _serve_asgi(app: Any) -> AsyncIterator[int]:
    """Run an ASGI *app* under uvicorn on ``127.0.0.1``; yield the bound port."""
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    task = asyncio.ensure_future(server.serve())
    try:
        for _ in range(400):
            if task.done():
                task.result()
            if server.started:
                break
            await asyncio.sleep(0.025)
        else:
            raise RuntimeError("server did not start")
        server.config.timeout_graceful_shutdown = 2
        yield server.servers[0].sockets[0].getsockname()[1]
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(task, 15)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            task.cancel()
        _forget_sse_shutdown()


def _text(result: Any) -> str:
    return result.content[0].text


# =====================================================================
# Session loss: re-initialise and retry once
# =====================================================================


class TestReconnectAfterRestart:
    async def test_client_reopens_the_session_after_a_restart(self, monkeypatch):
        client: MCPClient | None = None
        try:
            async with _serve(_inventory(), monkeypatch) as port:
                client = MCPClient(url=f"http://127.0.0.1:{port}/mcp")
                await client.__aenter__()
                first = await asyncio.wait_for(
                    client.call_tool("check_stock", {"sku": "A"}), TIMEOUT
                )
                assert '"in_stock": 42' in _text(first)
                assert client.session_generation == 1
            # Same address, fresh process: the old session id is unknown there.
            async with _serve(_inventory(), monkeypatch, port=port):
                result = await asyncio.wait_for(
                    client.call_tool("check_stock", {"sku": "A"}), TIMEOUT
                )
                assert '"in_stock": 42' in _text(result)
                assert client.session_generation == 2
                # The new session keeps working without another handshake.
                await asyncio.wait_for(client.list_tools(), TIMEOUT)
                assert client.session_generation == 2
        finally:
            if client is not None:
                await client.__aexit__(None, None, None)

    async def test_client_survives_a_restart_of_a_server_process(self, tmp_path):
        """The deploy case end to end: kill the server process, start a new one."""
        script = tmp_path / "inventory.py"
        script.write_text(
            textwrap.dedent(
                """
                import sys
                from promptise.mcp.server import MCPServer

                server = MCPServer("inventory")

                @server.tool()
                async def check_stock(sku: str) -> dict:
                    \"\"\"Units in stock for a product.\"\"\"
                    return {"sku": sku, "in_stock": 42}

                server.run(transport="http", port=int(sys.argv[1]))
                """
            )
        )
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}

        async def start() -> subprocess.Popen[bytes]:
            proc = subprocess.Popen(
                [sys.executable, str(script), str(port)],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            async with httpx.AsyncClient(timeout=1) as http:
                for _ in range(200):
                    try:
                        if (await http.get(f"http://127.0.0.1:{port}/health")).status_code == 200:
                            return proc
                    except httpx.TransportError:
                        pass
                    await asyncio.sleep(0.05)
            proc.kill()
            raise RuntimeError("server process did not start")

        def stop(proc: subprocess.Popen[bytes]) -> None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

        proc = await start()
        client = MCPClient(url=f"http://127.0.0.1:{port}/mcp")
        try:
            await asyncio.wait_for(client.__aenter__(), TIMEOUT)
            first = await asyncio.wait_for(client.call_tool("check_stock", {"sku": "A"}), TIMEOUT)
            assert '"in_stock": 42' in _text(first)
            stop(proc)
            proc = await start()
            again = await asyncio.wait_for(client.call_tool("check_stock", {"sku": "A"}), TIMEOUT)
            assert '"in_stock": 42' in _text(again)
            assert client.session_generation == 2
        finally:
            await client.__aexit__(None, None, None)
            stop(proc)

    async def test_multi_client_relists_tools_from_the_new_deployment(self, monkeypatch):
        async with _serve(_inventory(), monkeypatch) as port:
            multi = MCPMultiClient({"inventory": MCPClient(url=f"http://127.0.0.1:{port}/mcp")})
            await multi.__aenter__()
            await multi.list_tools()
            assert multi.tool_to_server == {"check_stock": "inventory"}
        try:
            async with _serve(_inventory(extra_tool=True), monkeypatch, port=port):
                result = await asyncio.wait_for(
                    multi.call_tool("check_stock", {"sku": "A"}), TIMEOUT
                )
                assert '"in_stock": 42' in _text(result)
                assert multi.tool_to_server == {"check_stock": "inventory", "reserve": "inventory"}
                reserved = await asyncio.wait_for(multi.call_tool("reserve", {"sku": "A"}), TIMEOUT)
                assert '"reserved": true' in _text(reserved)
        finally:
            await multi.__aexit__(None, None, None)

    async def test_agent_tools_keep_working_across_a_restart(self, monkeypatch):
        """The path build_agent uses: LangChain tools over MCPMultiClient."""
        async with _serve(_inventory(), monkeypatch) as port:
            multi = MCPMultiClient({"inventory": MCPClient(url=f"http://127.0.0.1:{port}/mcp")})
            await multi.__aenter__()
            (tool,) = await MCPToolAdapter(multi).as_langchain_tools()
            assert '"in_stock": 42' in str(await tool.ainvoke({"sku": "A"}))
        try:
            async with _serve(_inventory(), monkeypatch, port=port):
                for _ in range(3):
                    out = await asyncio.wait_for(tool.ainvoke({"sku": "A"}), TIMEOUT)
                    assert '"in_stock": 42' in str(out)
            assert multi.servers["inventory"].session_generation == 2
        finally:
            await multi.__aexit__(None, None, None)

    async def test_concurrent_calls_share_one_reinitialisation(self, monkeypatch):
        async with _serve(_inventory(), monkeypatch) as port:
            client = MCPClient(url=f"http://127.0.0.1:{port}/mcp")
            await client.__aenter__()
        try:
            async with _serve(_inventory(), monkeypatch, port=port):
                results = await asyncio.wait_for(
                    asyncio.gather(
                        *(client.call_tool("check_stock", {"sku": str(i)}) for i in range(5))
                    ),
                    TIMEOUT,
                )
                assert all('"in_stock": 42' in _text(r) for r in results)
                assert client.session_generation == 2
        finally:
            await client.__aexit__(None, None, None)

    async def test_server_down_then_back_recovers_on_the_next_call(self, monkeypatch):
        async with _serve(_inventory(), monkeypatch) as port:
            client = MCPClient(url=f"http://127.0.0.1:{port}/mcp", timeout=3)
            await client.__aenter__()
        try:
            # Nothing listens now: the call fails, and is not retried.
            with pytest.raises(MCPClientError):
                await asyncio.wait_for(client.call_tool("check_stock", {"sku": "A"}), TIMEOUT)
            async with _serve(_inventory(), monkeypatch, port=port):
                result = await asyncio.wait_for(
                    client.call_tool("check_stock", {"sku": "A"}), TIMEOUT
                )
                assert '"in_stock": 42' in _text(result)
        finally:
            await client.__aexit__(None, None, None)

    async def test_auto_reconnect_off_reports_the_lost_session(self, monkeypatch):
        async with _serve(_inventory(), monkeypatch) as port:
            client = MCPClient(url=f"http://127.0.0.1:{port}/mcp", auto_reconnect=False)
            await client.__aenter__()
        try:
            async with _serve(_inventory(), monkeypatch, port=port):
                with pytest.raises(MCPClientError, match="no longer knows this MCP session"):
                    await asyncio.wait_for(client.call_tool("check_stock", {"sku": "A"}), TIMEOUT)
                assert client.session_generation == 1
        finally:
            await client.__aexit__(None, None, None)

    async def test_closed_client_does_not_reconnect(self, monkeypatch):
        async with _serve(_inventory(), monkeypatch) as port:
            client = MCPClient(url=f"http://127.0.0.1:{port}/mcp")
            async with client:
                pass
            with pytest.raises(MCPClientError, match="Not connected"):
                await client.call_tool("check_stock", {"sku": "A"})
            assert client.session_generation == 1


# =====================================================================
# Wrong URL: 404 on initialize
# =====================================================================


class TestEndpointNotFound:
    async def test_url_without_mcp_path_is_a_typed_404(self, monkeypatch):
        async with _serve(_inventory(), monkeypatch) as port:
            url = f"http://127.0.0.1:{port}"
            with pytest.raises(MCPConnectionRejectedError) as info:
                async with MCPClient(url=url):
                    pass
        err = info.value
        assert err.status_code == 404
        assert str(err) == (
            f"Server at {url} rejected the connection: 404 Not Found. "
            f"Check the URL ({url}); Promptise servers serve MCP at /mcp."
        )

    async def test_multi_client_names_the_server(self, monkeypatch):
        async with _serve(_inventory(), monkeypatch) as port:
            multi = MCPMultiClient({"inventory": MCPClient(url=f"http://127.0.0.1:{port}/")})
            with pytest.raises(MCPConnectionRejectedError, match="Server 'inventory' rejected"):
                async with multi:
                    pass


# =====================================================================
# CORS for browser clients
# =====================================================================


class TestCORS:
    def test_defaults_cover_the_mcp_headers(self):
        cors = CORSConfig()
        assert cors.allow_origins == []
        for header in ("mcp-session-id", "mcp-protocol-version", "last-event-id"):
            assert header in cors.allow_headers
        assert cors.expose_headers == ["mcp-session-id"]
        assert list(MCP_CORS_ALLOW_HEADERS) == cors.allow_headers
        assert list(MCP_CORS_EXPOSE_HEADERS) == cors.expose_headers
        # Each instance owns its lists.
        assert CORSConfig().allow_headers is not cors.allow_headers

    def test_wildcard_origin_with_credentials_is_refused(self):
        # Starlette would echo every origin back with Allow-Credentials: true.
        with pytest.raises(ValueError, match="allow_credentials"):
            CORSConfig(allow_origins=["*"], allow_credentials=True)
        with pytest.raises(ValueError, match="allow_credentials"):
            CORSConfig(allow_origins=[BROWSER, "*"], allow_credentials=True)
        # Each half on its own stays allowed.
        assert CORSConfig(allow_origins=["*"]).allow_credentials is False
        assert CORSConfig(allow_origins=[BROWSER], allow_credentials=True).allow_credentials

    async def test_default_config_admits_a_browser_client(self, monkeypatch):
        cors = CORSConfig(allow_origins=[BROWSER])
        async with _serve(_inventory(), monkeypatch, cors=cors, allowed_origins=[BROWSER]) as port:
            url = f"http://127.0.0.1:{port}/mcp"
            async with httpx.AsyncClient(timeout=TIMEOUT) as http:
                preflight = await http.options(
                    url,
                    headers={
                        "origin": BROWSER,
                        "access-control-request-method": "POST",
                        "access-control-request-headers": (
                            "content-type,mcp-session-id,mcp-protocol-version"
                        ),
                    },
                )
                assert preflight.status_code == 200
                assert preflight.headers["access-control-allow-origin"] == BROWSER

                init = await http.post(url, json=INIT, headers={**MCP_HEADERS, "origin": BROWSER})
                assert init.status_code == 200
                assert init.headers["mcp-session-id"]
                exposed = init.headers["access-control-expose-headers"].lower()
                assert "mcp-session-id" in exposed

    async def test_preflight_is_answered_before_the_auth_gate(self, monkeypatch):
        server = _inventory(require_auth=True)
        cors = CORSConfig(allow_origins=[BROWSER])
        async with _serve(server, monkeypatch, cors=cors, allowed_origins=[BROWSER]) as port:
            url = f"http://127.0.0.1:{port}/mcp"
            async with httpx.AsyncClient(timeout=TIMEOUT) as http:
                preflight = await http.options(
                    url,
                    headers={
                        "origin": BROWSER,
                        "access-control-request-method": "POST",
                        "access-control-request-headers": "content-type,x-api-key",
                    },
                )
                assert preflight.status_code == 200
                # The page can read the gate's 401 (it carries CORS headers).
                denied = await http.post(url, json=INIT, headers={**MCP_HEADERS, "origin": BROWSER})
                assert denied.status_code == 401
                assert denied.headers["access-control-allow-origin"] == BROWSER
                allowed = await http.post(
                    url, json=INIT, headers={**MCP_HEADERS, "origin": BROWSER, "x-api-key": API_KEY}
                )
                assert allowed.status_code == 200

    async def test_sse_transport_applies_the_same_cors(self, monkeypatch):
        cors = CORSConfig(allow_origins=[BROWSER])
        async with _serve(
            _inventory(), monkeypatch, transport="sse", cors=cors, allowed_origins=[BROWSER]
        ) as port:
            async with httpx.AsyncClient(timeout=TIMEOUT) as http:
                preflight = await http.options(
                    f"http://127.0.0.1:{port}/messages/",
                    headers={
                        "origin": BROWSER,
                        "access-control-request-method": "POST",
                        "access-control-request-headers": "content-type,mcp-protocol-version",
                    },
                )
                assert preflight.status_code == 200


# =====================================================================
# Bind default and the startup policy log
# =====================================================================


class TestBindDefault:
    @pytest.mark.parametrize("method", ["run", "run_async"])
    def test_server_binds_loopback_by_default(self, method):
        default = inspect.signature(getattr(MCPServer, method)).parameters["host"].default
        assert default == "127.0.0.1"

    @pytest.mark.parametrize("runner", [run_http, run_sse])
    def test_transport_runners_bind_loopback_by_default(self, runner):
        assert inspect.signature(runner).parameters["host"].default == "127.0.0.1"

    async def test_default_run_validates_host(self, monkeypatch):
        captured: dict[str, Any] = {}

        async def fake_run_transport(*_args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        monkeypatch.setattr("promptise.mcp.server._app.run_transport", fake_run_transport)
        await _inventory().run_async(transport="http")
        assert captured["host"] == "127.0.0.1"
        assert captured["security_settings"].enable_dns_rebinding_protection is True

    async def test_explicit_public_host_is_kept(self, monkeypatch):
        captured: dict[str, Any] = {}

        async def fake_run_transport(*_args: Any, **kwargs: Any) -> None:
            captured.update(kwargs)

        monkeypatch.setattr("promptise.mcp.server._app.run_transport", fake_run_transport)
        await _inventory().run_async(
            transport="http",
            host="0.0.0.0",  # nosec B104
            allowed_hosts=["mcp.example.com"],
        )
        assert captured["host"] == "0.0.0.0"  # nosec B104
        assert captured["security_settings"].allowed_hosts == ["mcp.example.com"]

    @pytest.mark.parametrize("transport", ["http", "sse"])
    async def test_public_bind_without_auth_warns(self, monkeypatch, caplog, transport):
        async def fake_run_transport(*_args: Any, **_kwargs: Any) -> None:
            return None

        monkeypatch.setattr("promptise.mcp.server._app.run_transport", fake_run_transport)
        with caplog.at_level(logging.WARNING, logger="promptise.server"):
            await _inventory().run_async(transport=transport, host="0.0.0.0")  # nosec B104
        messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
        assert any("has no AuthMiddleware" in m and "0.0.0.0:8080" in m for m in messages)

    @pytest.mark.parametrize(
        ("host", "with_auth", "via_router"),
        [
            ("127.0.0.1", False, False),
            ("localhost", False, False),
            ("0.0.0.0", True, False),  # nosec B104
            ("0.0.0.0", True, True),  # nosec B104
        ],
    )
    async def test_no_auth_warning_on_loopback_or_with_auth(
        self, monkeypatch, caplog, host, with_auth, via_router
    ):
        async def fake_run_transport(*_args: Any, **_kwargs: Any) -> None:
            return None

        monkeypatch.setattr("promptise.mcp.server._app.run_transport", fake_run_transport)
        server = _inventory()
        if with_auth:
            auth = AuthMiddleware(APIKeyAuth(keys={API_KEY: "agent"}))
            if via_router:
                from promptise.mcp.server import MCPRouter

                router = MCPRouter(middleware=[auth])

                @router.tool(auth=True)
                async def secret() -> str:
                    """Protected tool."""
                    return "ok"

                server.include_router(router)
            else:
                server.add_middleware(auth)
        with caplog.at_level(logging.WARNING, logger="promptise.server"):
            await server.run_async(transport="http", host=host)
        assert not any("AuthMiddleware" in r.getMessage() for r in caplog.records)

    async def test_stdio_never_warns_about_auth(self, monkeypatch, caplog):
        async def fake_run_transport(*_args: Any, **_kwargs: Any) -> None:
            return None

        monkeypatch.setattr("promptise.mcp.server._app.run_transport", fake_run_transport)
        with caplog.at_level(logging.WARNING, logger="promptise.server"):
            await _inventory().run_async(transport="stdio", host="0.0.0.0")  # nosec B104
        assert not caplog.records

    def test_unvalidated_public_bind_logs_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="promptise.server"):
            _log_transport_security(build_transport_security("0.0.0.0"), "0.0.0.0")  # nosec B104
        (record,) = caplog.records
        assert record.levelno == logging.WARNING
        assert "Host/Origin validation off" in record.getMessage()

    def test_validated_bind_logs_at_info(self, caplog):
        with caplog.at_level(logging.INFO, logger="promptise.server"):
            _log_transport_security(build_transport_security("127.0.0.1"), "127.0.0.1")
        assert [r.levelno for r in caplog.records] == [logging.INFO]


# =====================================================================
# Health routes
# =====================================================================


class TestHealthRoutes:
    async def test_probes_without_a_health_check(self, monkeypatch):
        async with _serve(_inventory(), monkeypatch) as port:
            async with httpx.AsyncClient(timeout=TIMEOUT) as http:
                live = await http.get(f"http://127.0.0.1:{port}/health")
                ready = await http.get(f"http://127.0.0.1:{port}/health/ready")
                bare_mcp = await http.get(f"http://127.0.0.1:{port}/mcp")
        assert live.status_code == 200
        assert live.json() == {"status": "alive"}
        assert ready.status_code == 200
        assert ready.json() == {"status": "ready", "checks": {}}
        assert bare_mcp.status_code == 406  # why /mcp is no probe target

    async def test_readiness_follows_the_health_check(self, monkeypatch):
        state = {"ok": True}

        def database() -> bool:
            if state["ok"] is None:
                raise RuntimeError("password=hunter2 in connection string")
            return bool(state["ok"])

        server = _inventory(require_auth=True)
        health = HealthCheck()
        health.add_check("database", database, required_for_ready=True)
        health.add_check("cache", lambda: False, required_for_ready=False)
        health.register_resources(server)

        async with _serve(server, monkeypatch) as port:
            base = f"http://127.0.0.1:{port}"
            async with httpx.AsyncClient(timeout=TIMEOUT) as http:
                # Probes carry no credentials and still pass the auth gate.
                live = await http.get(f"{base}/health")
                assert live.status_code == 200
                assert live.json()["status"] == "alive"
                assert "uptime_seconds" in live.json()

                ready = await http.get(f"{base}/health/ready")
                assert ready.status_code == 200
                assert ready.json() == {
                    "status": "ready",
                    "checks": {"database": {"healthy": True}, "cache": {"healthy": False}},
                }

                state["ok"] = False
                not_ready = await http.get(f"{base}/health/ready")
                assert not_ready.status_code == 503
                assert not_ready.json()["status"] == "not_ready"

                state["ok"] = None  # the check raises
                raised = await http.get(f"{base}/health/ready")
                assert raised.status_code == 503
                assert "hunter2" not in raised.text
                assert raised.json()["checks"]["database"] == {"healthy": False}

                # /mcp itself still requires credentials.
                assert (
                    await http.post(f"{base}/mcp", json=INIT, headers=MCP_HEADERS)
                ).status_code == 401

    async def test_sse_transport_serves_the_probes(self, monkeypatch):
        async with _serve(_inventory(), monkeypatch, transport="sse") as port:
            async with httpx.AsyncClient(timeout=TIMEOUT) as http:
                assert (await http.get(f"http://127.0.0.1:{port}/health")).status_code == 200
                assert (await http.get(f"http://127.0.0.1:{port}/health/ready")).status_code == 200

    async def test_mcp_resources_keep_the_error_detail(self):
        health = HealthCheck()

        def broken() -> bool:
            raise RuntimeError("boom")

        health.add_check("broken", broken)
        report = await health.readiness_report()
        assert report["checks"]["broken"] == {"healthy": False, "error": "boom"}


# =====================================================================
# ASGI app
# =====================================================================


class TestASGIApp:
    async def test_runs_under_uvicorn(self):
        app = _inventory().asgi_app()
        async with _serve_asgi(app) as port:
            async with MCPClient(url=f"http://127.0.0.1:{port}/mcp") as client:
                result = await asyncio.wait_for(
                    client.call_tool("check_stock", {"sku": "A"}), TIMEOUT
                )
                assert '"in_stock": 42' in _text(result)
            async with httpx.AsyncClient(timeout=TIMEOUT) as http:
                assert (await http.get(f"http://127.0.0.1:{port}/health/ready")).status_code == 200

    async def test_validates_host_against_loopback_plus_allowed_hosts(self):
        app = _inventory().asgi_app(allowed_hosts=["mcp.example.com"])
        async with _serve_asgi(app) as port:
            url = f"http://127.0.0.1:{port}/mcp"
            async with httpx.AsyncClient(timeout=TIMEOUT) as http:

                async def status(host: str) -> int:
                    response = await http.post(
                        url, json=INIT, headers={**MCP_HEADERS, "host": host}
                    )
                    return response.status_code

                assert await status("mcp.example.com") == 200
                assert await status(f"127.0.0.1:{port}") == 200
                assert await status("attacker.example") == 421

    async def test_stateless_app_needs_no_session(self):
        app = _inventory().asgi_app(stateless=True)
        async with _serve_asgi(app) as port:
            async with httpx.AsyncClient(timeout=TIMEOUT) as http:
                init = await http.post(
                    f"http://127.0.0.1:{port}/mcp", json=INIT, headers=MCP_HEADERS
                )
                assert init.status_code == 200
                assert "mcp-session-id" not in init.headers
                # A tool call with no session id — as another replica would see it.
                call = await http.post(
                    f"http://127.0.0.1:{port}/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {"name": "check_stock", "arguments": {"sku": "A"}},
                    },
                    headers=MCP_HEADERS,
                )
                assert call.status_code == 200
                assert "in_stock" in call.text

    async def test_sse_app(self):
        app = _inventory().asgi_app("sse")
        async with _serve_asgi(app) as port:
            async with MCPClient(url=f"http://127.0.0.1:{port}/sse", transport="sse") as client:
                tools = await asyncio.wait_for(client.list_tools(), TIMEOUT)
                assert [t.name for t in tools] == ["check_stock"]

    def test_rejects_stdio_and_stateless_sse(self):
        with pytest.raises(ValueError, match="stdio"):
            _inventory().asgi_app("stdio")
        with pytest.raises(ValueError, match="stateless"):
            _inventory().asgi_app("sse", stateless=True)
