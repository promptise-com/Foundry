"""BackgroundTasks run after the response; MCP cancellation sets the CancellationToken.

Regressions (reproduced against v1.2.1 over Streamable HTTP):

- ``_app.py`` awaited ``BackgroundTasks.execute()`` before returning, so a
  client calling a tool with a 3-second background task waited 3 seconds.
- Nothing ever called ``CancellationToken.cancel()``: on
  ``notifications/cancelled`` the MCP SDK cancelled the handler's task and
  the token stayed unset.

Every server binds ``127.0.0.1`` on an ephemeral port in-process.  Tool
handlers rendezvous on events instead of sleeping, so the tests stay fast.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import anyio
import pytest
import uvicorn
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CancelledNotification, CancelledNotificationParams, ClientNotification

from promptise.mcp.client import MCPClient
from promptise.mcp.server import (
    BackgroundTasks,
    CancellationToken,
    Depends,
    MCPServer,
    TestClient,
)
from promptise.mcp.server._app import _run_cancellable

TIMEOUT = 5.0


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


# =====================================================================
# BackgroundTasks
# =====================================================================


class TestBackgroundTasksAfterResponse:
    async def test_response_does_not_wait_for_background_tasks(self, monkeypatch):
        release = asyncio.Event()
        sent: list[str] = []
        server = MCPServer("orders")

        async def send_receipt(order_id: str) -> None:
            await release.wait()  # a slow email API
            sent.append(order_id)

        @server.tool()
        async def place_order(
            order_id: str, bg: BackgroundTasks = Depends(BackgroundTasks)
        ) -> dict:
            """Place an order and email the receipt afterwards."""
            bg.add(send_receipt, order_id)
            return {"order_id": order_id, "status": "placed"}

        async with _serve(server, monkeypatch) as url, MCPClient(url=url) as client:
            # Before the fix this deadlocked: the response waited for the
            # receipt, which waits for the response to arrive.
            result = await asyncio.wait_for(
                client.call_tool("place_order", {"order_id": "A-1001"}), TIMEOUT
            )
            assert '"placed"' in result.content[0].text
            assert sent == []

            release.set()
            for _ in range(200):
                if sent:
                    break
                await asyncio.sleep(0.01)
            assert sent == ["A-1001"]

    async def test_background_task_sees_the_request_context(self):
        from promptise.mcp.server import get_context

        server = MCPServer("ctx")
        seen: list[str] = []
        bg = BackgroundTasks()
        bg.add(lambda: seen.append(get_context().tool_name))

        from promptise.mcp.server._context import RequestContext, clear_context, set_context

        set_context(RequestContext(server_name="ctx", tool_name="place_order"))
        try:
            server._run_background(bg)
        finally:
            clear_context()
        await server._drain_background_tasks()
        assert seen == ["place_order"]

    async def test_shutdown_waits_for_running_background_tasks(self):
        server = MCPServer("drain")
        done: list[str] = []

        async def slow() -> None:
            await asyncio.sleep(0.05)
            done.append("receipt")

        bg = BackgroundTasks()
        bg.add(slow)
        server._run_background(bg)
        assert server._background_runs

        server._build_lowlevel_server()  # registers the drain hook
        await server._lifecycle.shutdown(timeout=TIMEOUT)
        assert done == ["receipt"]
        assert not server._background_runs

    async def test_background_errors_are_logged_not_returned(self, caplog):
        server = MCPServer("errors")

        def boom() -> None:
            raise RuntimeError("smtp down")

        bg = BackgroundTasks()
        bg.add(boom)
        server._run_background(bg)
        await server._drain_background_tasks()
        assert "Background task boom failed" in caplog.text

    async def test_testclient_still_runs_them_before_returning(self):
        """TestClient keeps background tasks inline so tests can assert on them."""
        server = MCPServer("inline")
        sent: list[str] = []

        @server.tool()
        async def place_order(order_id: str, bg: BackgroundTasks = Depends(BackgroundTasks)) -> str:
            bg.add(sent.append, order_id)
            return "placed"

        await TestClient(server).call_tool("place_order", {"order_id": "B-2"})
        assert sent == ["B-2"]


# =====================================================================
# Cancellation → CancellationToken
# =====================================================================


async def _cancel_scope_after(event: asyncio.Event, scope: anyio.CancelScope) -> None:
    await event.wait()
    scope.cancel()


class TestRunCancellable:
    """The helper the server wraps around calls to token-taking tools."""

    async def test_returns_the_result_when_not_cancelled(self):
        token = CancellationToken()

        async def work() -> str:
            return "ok"

        assert await _run_cancellable(work(), token, grace_period=1) == "ok"
        assert not token.is_cancelled

    async def test_cooperative_handler_stops_on_the_token(self):
        token = CancellationToken()
        started = asyncio.Event()
        outcome: list[str] = []

        async def work() -> str:
            started.set()
            await token.wait()
            outcome.append(f"stopped: {token.reason}")
            return "partial"

        async with anyio.create_task_group() as tg:
            with anyio.CancelScope() as scope:
                tg.start_soon(_cancel_scope_after, started, scope)
                await _run_cancellable(work(), token, grace_period=TIMEOUT)
        assert scope.cancelled_caught
        assert outcome == ["stopped: Request cancelled by the client"]

    async def test_handler_ignoring_the_token_is_cancelled_after_the_grace_period(self):
        token = CancellationToken()
        started = asyncio.Event()
        outcome: list[bool] = []

        async def work() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                outcome.append(token.is_cancelled)
                raise

        loop = asyncio.get_running_loop()
        async with anyio.create_task_group() as tg:
            with anyio.CancelScope() as scope:
                tg.start_soon(_cancel_scope_after, started, scope)
                begun = loop.time()
                await _run_cancellable(work(), token, grace_period=0.05)
        assert scope.cancelled_caught
        assert outcome == [True]  # token was set before the task was cancelled
        assert loop.time() - begun < TIMEOUT


class TestCancellationOverHTTP:
    async def test_cancel_notification_sets_the_token(self, monkeypatch):
        started = asyncio.Event()
        stopped = asyncio.Event()
        reasons: list[str | None] = []
        server = MCPServer("feeds")

        @server.tool()
        async def watch_feed(cancel: CancellationToken = Depends(CancellationToken)) -> dict:
            """Watch a feed until the client cancels."""
            started.set()
            await cancel.wait()
            reasons.append(cancel.reason)
            stopped.set()
            return {"stopped_by": "token"}

        async with (
            _serve(server, monkeypatch) as url,
            streamablehttp_client(url) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            call = asyncio.create_task(session.call_tool("watch_feed", {}))
            await asyncio.wait_for(started.wait(), TIMEOUT)
            request_id = session._request_id - 1  # id of the call just sent
            await session.send_notification(
                ClientNotification(
                    CancelledNotification(
                        params=CancelledNotificationParams(
                            requestId=request_id, reason="user stopped it"
                        )
                    )
                )
            )
            await asyncio.wait_for(stopped.wait(), TIMEOUT)
            with pytest.raises(Exception, match="Request cancelled"):
                await asyncio.wait_for(call, TIMEOUT)

        assert reasons == ["Request cancelled by the client"]


def test_grace_period_is_configurable():
    assert MCPServer("a")._cancel_grace_period == 5.0
    assert MCPServer("b", cancel_grace_period=0.5)._cancel_grace_period == 0.5
