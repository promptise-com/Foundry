"""Progress notifications reach Promptise clients and agents.

Regression (v1.2.1): ``MCPClient.call_tool`` never passed a progress
callback, so a server's ``ProgressReporter.report()`` calls reached the
official SDK client but never a Promptise client or agent.

Every server binds ``127.0.0.1`` on an ephemeral port in-process.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import uvicorn

from promptise.agent import build_agent
from promptise.config import HTTPServerSpec
from promptise.events import AgentEvent, CallbackSink, EventNotifier
from promptise.mcp.client import MCPClient, MCPMultiClient, MCPToolAdapter
from promptise.mcp.server import Depends, MCPServer, ProgressReporter

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


def _crawler() -> tuple[MCPServer, list[bool]]:
    """A server whose tool reports progress; records whether a token was sent."""
    server = MCPServer("crawler")
    had_token: list[bool] = []

    @server.tool()
    async def crawl_site(
        pages: int, progress: ProgressReporter = Depends(ProgressReporter)
    ) -> dict:
        """Crawl pages and report progress."""
        had_token.append(progress._progress_token is not None)
        for page in range(1, pages + 1):
            await progress.report(page, total=pages, message=f"Crawled page {page} of {pages}")
        return {"pages": pages}

    return server, had_token


EXPECTED = [
    (1.0, 3.0, "Crawled page 1 of 3"),
    (2.0, 3.0, "Crawled page 2 of 3"),
    (3.0, 3.0, "Crawled page 3 of 3"),
]


class TestMCPClientProgress:
    async def test_progress_callback_receives_notifications(self, monkeypatch):
        server, had_token = _crawler()
        seen: list[tuple[float, float | None, str | None]] = []

        async def on_progress(progress: float, total: float | None, message: str | None) -> None:
            seen.append((progress, total, message))

        async with _serve(server, monkeypatch) as url, MCPClient(url=url) as client:
            result = await asyncio.wait_for(
                client.call_tool("crawl_site", {"pages": 3}, progress_callback=on_progress),
                TIMEOUT,
            )
        assert '"pages": 3' in result.content[0].text
        assert seen == EXPECTED
        assert had_token == [True]

    async def test_no_progress_is_requested_without_a_callback(self, monkeypatch):
        server, had_token = _crawler()
        async with _serve(server, monkeypatch) as url, MCPClient(url=url) as client:
            await asyncio.wait_for(client.call_tool("crawl_site", {"pages": 2}), TIMEOUT)
        assert had_token == [False]

    async def test_multi_client_routes_the_callback(self, monkeypatch):
        server, _ = _crawler()
        seen: list[Any] = []

        async def on_progress(progress: float, total: float | None, message: str | None) -> None:
            seen.append((progress, total, message))

        async with _serve(server, monkeypatch) as url:
            async with MCPMultiClient({"crawler": MCPClient(url=url)}) as multi:
                await multi.list_tools()
                await asyncio.wait_for(
                    multi.call_tool("crawl_site", {"pages": 3}, progress_callback=on_progress),
                    TIMEOUT,
                )
        assert seen == EXPECTED


class TestToolAdapterProgress:
    async def _tool(self, multi: MCPMultiClient, **kwargs: Any) -> Any:
        tools = await MCPToolAdapter(multi, **kwargs).as_langchain_tools()
        return next(t for t in tools if t.name == "crawl_site")

    async def test_on_progress_gets_tool_name_and_values(self, monkeypatch):
        server, _ = _crawler()
        seen: list[Any] = []

        def on_progress(name: str, progress: float, total: float | None, message: str | None):
            seen.append((name, progress, total, message))

        async with _serve(server, monkeypatch) as url:
            async with MCPMultiClient({"crawler": MCPClient(url=url)}) as multi:
                tool = await self._tool(multi, on_progress=on_progress)
                out = await asyncio.wait_for(tool.ainvoke({"pages": 3}), TIMEOUT)
        assert '"pages": 3' in out
        assert seen == [("crawl_site", *e) for e in EXPECTED]

    async def test_async_callback_is_awaited_and_failures_do_not_break_the_call(
        self, monkeypatch, caplog
    ):
        server, _ = _crawler()
        seen: list[float] = []

        async def on_progress(name: str, progress: float, total: Any, message: Any) -> None:
            seen.append(progress)
            if progress == 2:
                raise RuntimeError("UI went away")

        async with _serve(server, monkeypatch) as url:
            async with MCPMultiClient({"crawler": MCPClient(url=url)}) as multi:
                tool = await self._tool(multi, on_progress=on_progress)
                out = await asyncio.wait_for(tool.ainvoke({"pages": 3}), TIMEOUT)
        assert '"pages": 3' in out
        assert seen == [1.0, 2.0, 3.0]
        assert "on_progress callback failed for tool 'crawl_site'" in caplog.text


class TestBuildAgentProgress:
    async def test_on_tool_progress_and_events(self, monkeypatch):
        server, _ = _crawler()
        seen: list[Any] = []
        events: list[AgentEvent] = []
        notifier = EventNotifier(sinks=[CallbackSink(events.append, events=["tool.progress"])])

        async def on_tool_progress(name: str, progress: float, total: Any, message: Any) -> None:
            seen.append((name, progress, total, message))

        async with _serve(server, monkeypatch) as url:
            with patch("promptise.agent._normalize_model", return_value=MagicMock()):
                agent = await build_agent(
                    servers={"crawler": HTTPServerSpec(url=url)},
                    model="openai:gpt-5-mini",
                    on_tool_progress=on_tool_progress,
                    events=notifier,
                )
            try:
                tool = next(t for t in agent.tools if t.name == "crawl_site")
                await asyncio.wait_for(tool.ainvoke({"pages": 3}), TIMEOUT)
                for _ in range(100):
                    if len(events) == 3:
                        break
                    await asyncio.sleep(0.01)
            finally:
                await agent.shutdown()

        assert seen == [("crawl_site", *e) for e in EXPECTED]
        assert [e.data for e in events] == [
            {"tool_name": "crawl_site", "progress": p, "total": t, "message": m}
            for p, t, m in EXPECTED
        ]
        assert {e.severity for e in events} == {"info"}

    async def test_trace_tools_prints_progress(self, monkeypatch, capsys):
        server, _ = _crawler()
        async with _serve(server, monkeypatch) as url:
            with patch("promptise.agent._normalize_model", return_value=MagicMock()):
                agent = await build_agent(
                    servers={"crawler": HTTPServerSpec(url=url)},
                    model="openai:gpt-5-mini",
                    trace_tools=True,
                )
            try:
                tool = next(t for t in agent.tools if t.name == "crawl_site")
                await asyncio.wait_for(tool.ainvoke({"pages": 1}), TIMEOUT)
            finally:
                await agent.shutdown()
        assert "… crawl_site progress 1/1: Crawled page 1 of 1" in capsys.readouterr().out
