"""Connection failures surface as clear, typed errors — over the REAL HTTP transport.

Regression: an ``MCPServer(require_auth=True)`` answers ``initialize`` from a
client without credentials with ``401 Unauthorized``.  The MCP SDK raises that
inside its transport task group, which used to be entered in the caller's task
and never exited: the caller's task was cancelled (and stayed cancelled), the
client hung, and teardown died with "Attempted to exit cancel scope in a
different task".  ``build_agent`` inherited all of it.

Now the client owns the transport in its own task, so the failure is unwound
there and reported as :class:`MCPConnectionRejectedError` — fast, and without
touching the caller's task.

Every server binds ``127.0.0.1`` on an ephemeral port in-process.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest
import uvicorn

from promptise import MCPConnectionRejectedError
from promptise.agent import build_agent
from promptise.config import HTTPServerSpec
from promptise.mcp.client import MCPClient, MCPClientError, MCPMultiClient
from promptise.mcp.server import APIKeyAuth, AuthMiddleware, MCPServer

API_KEY = "test-key-123"

# A rejected handshake is one round trip on loopback; anything near this
# bound means the client is hanging again.
FAIL_FAST = 5.0


def _orders_server() -> MCPServer:
    server = MCPServer("orders", require_auth=True)
    server.add_middleware(AuthMiddleware(APIKeyAuth(keys={API_KEY: "support-agent"})))

    @server.tool()
    async def get_order_status(order_id: str) -> dict:
        """Look up an order."""
        return {"order_id": order_id, "status": "shipped"}

    return server


@asynccontextmanager
async def _serve(server: MCPServer, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[str]:
    """Run *server* over Streamable HTTP and yield its ``/mcp`` URL."""
    instances: list[uvicorn.Server] = []

    class _Recording(uvicorn.Server):
        def __init__(self, config: uvicorn.Config) -> None:
            super().__init__(config)
            instances.append(self)

    monkeypatch.setattr(uvicorn, "Server", _Recording)
    task = asyncio.ensure_future(server.run_async(transport="http", host="127.0.0.1", port=0))
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
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        if instances:
            instances[0].should_exit = True
        try:
            await asyncio.wait_for(task, 15)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            task.cancel()


async def _assert_caller_task_healthy() -> None:
    """The failed connect must leave no cancellation pending on this task."""
    task = asyncio.current_task()
    assert task is not None
    if hasattr(task, "cancelling"):  # Python 3.11+
        assert task.cancelling() == 0
    await asyncio.sleep(0.05)  # a leaked cancel scope would cancel this


async def _connect(client: MCPClient | MCPMultiClient) -> None:
    async with client:
        pass


# =====================================================================
# MCPClient
# =====================================================================


class TestMCPClientRejected:
    async def test_missing_credentials_raise_typed_401(self, monkeypatch):
        async with _serve(_orders_server(), monkeypatch) as url:
            with pytest.raises(MCPConnectionRejectedError) as info:
                await asyncio.wait_for(_connect(MCPClient(url=url, api_key=None)), FAIL_FAST)
            await _assert_caller_task_healthy()

        err = info.value
        assert isinstance(err, MCPClientError)
        assert err.status_code == 401
        assert err.reason == "Unauthorized"
        assert err.url == url
        assert str(err) == (
            f"Server at {url} rejected the connection: 401 Unauthorized. "
            "Check the bearer_token/api_key configured for it."
        )

    async def test_wrong_api_key_raises_typed_401(self, monkeypatch):
        async with _serve(_orders_server(), monkeypatch) as url:
            with pytest.raises(MCPConnectionRejectedError, match="401 Unauthorized"):
                client = MCPClient(url=url, api_key="not-the-key")
                await asyncio.wait_for(_connect(client), FAIL_FAST)
            await _assert_caller_task_healthy()

    async def test_valid_key_connects_and_closes_from_another_task(self, monkeypatch):
        async with _serve(_orders_server(), monkeypatch) as url:
            client = MCPClient(url=url, api_key=API_KEY)
            await asyncio.wait_for(client.__aenter__(), FAIL_FAST)
            tools = await asyncio.wait_for(client.list_tools(), FAIL_FAST)
            assert [t.name for t in tools] == ["get_order_status"]
            # The transport is owned by the client, not the entering task,
            # so closing from a different task is safe.
            await asyncio.wait_for(
                asyncio.create_task(client.__aexit__(None, None, None)), FAIL_FAST
            )
            assert client.session is None
            await _assert_caller_task_healthy()

    async def test_unreachable_server_is_a_plain_client_error(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]  # closed once the block exits
        with pytest.raises(MCPClientError, match="Failed to connect to") as info:
            await asyncio.wait_for(
                _connect(MCPClient(url=f"http://127.0.0.1:{port}/mcp")), FAIL_FAST
            )
        assert not isinstance(info.value, MCPConnectionRejectedError)
        await _assert_caller_task_healthy()


# =====================================================================
# MCPMultiClient and build_agent name the server
# =====================================================================


class TestNamedServerRejected:
    async def test_multi_client_names_the_server(self, monkeypatch):
        async with _serve(_orders_server(), monkeypatch) as url:
            multi = MCPMultiClient({"orders": MCPClient(url=url)})
            with pytest.raises(MCPConnectionRejectedError) as info:
                await asyncio.wait_for(_connect(multi), FAIL_FAST)
            await _assert_caller_task_healthy()

        assert info.value.server_name == "orders"
        assert info.value.status_code == 401
        assert str(info.value).startswith(
            "Server 'orders' rejected the connection: 401 Unauthorized."
        )

    async def test_build_agent_fails_fast_with_clear_error(self, monkeypatch):
        async with _serve(_orders_server(), monkeypatch) as url:
            with (
                patch("promptise.agent._normalize_model", return_value=MagicMock()),
                pytest.raises(MCPConnectionRejectedError) as info,
            ):
                await asyncio.wait_for(
                    build_agent(
                        model="openai:gpt-5-mini",
                        servers={"orders": HTTPServerSpec(url=url)},
                    ),
                    FAIL_FAST,
                )
            await _assert_caller_task_healthy()

        assert str(info.value) == (
            "Server 'orders' rejected the connection: 401 Unauthorized. "
            "Check the bearer_token/api_key configured for it."
        )


# =====================================================================
# Message formatting
# =====================================================================


class TestRejectedErrorMessage:
    def test_forbidden_points_at_credentials(self):
        err = MCPConnectionRejectedError(status_code=403, reason="Forbidden", url="http://h/mcp")
        assert "403 Forbidden" in str(err)
        assert "bearer_token/api_key" in str(err)

    def test_not_found_points_at_the_url(self):
        err = MCPConnectionRejectedError(status_code=404, reason="Not Found", url="http://h/x")
        assert str(err) == (
            "Server at http://h/x rejected the connection: 404 Not Found. "
            "Check the URL (http://h/x); Promptise servers serve MCP at /mcp."
        )

    def test_for_server_keeps_the_details(self):
        err = MCPConnectionRejectedError(status_code=401, reason="", url="http://h/mcp")
        named = err.for_server("orders")
        assert (named.status_code, named.url, named.server_name) == (401, "http://h/mcp", "orders")
        assert str(named).startswith("Server 'orders' rejected the connection: 401.")
