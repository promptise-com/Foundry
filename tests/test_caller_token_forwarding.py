"""The invoking caller's bearer token reaches HTTP MCP servers — per call.

Regression: ``build_agent`` opened one long-lived session per server with the
headers from ``HTTPServerSpec``, so ``agent.ainvoke(..., caller=bob)`` called
tools with whatever token the agent was built with.  An agent built with
Alice's token answered Bob with Alice's data.

Now every tool call made during an invocation whose ``CallerContext`` carries
a ``bearer_token`` goes over a session opened for that token, so the server
authenticates the user.  Concurrent invocations never share a session or a
header dict.  The end-to-end tests run a real ``MCPServer`` with ``JWTAuth``
over Streamable HTTP on a loopback port in 8500-8509.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest
import uvicorn
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from promptise import CallerContext
from promptise.agent import build_agent
from promptise.config import HTTPServerSpec
from promptise.mcp.client import MCPClient, MCPClientError, MCPMultiClient
from promptise.mcp.client._caller_sessions import CallerSessionPool
from promptise.mcp.server import AuthMiddleware, JWTAuth, MCPServer, RequestContext

SECRET = "caller-forwarding-test-secret-0123456789abcdef"
ISSUER = JWTAuth(secret=SECRET)


def _token(sub: str, org: str) -> str:
    return ISSUER.create_token({"sub": sub, "org": org})


ALICE = CallerContext(user_id="alice", tenant_id="acme", bearer_token=_token("alice", "acme"))
BOB = CallerContext(user_id="bob", tenant_id="globex", bearer_token=_token("bob", "globex"))
SERVICE_TOKEN = _token("crm-agent", "platform")


def _crm_server(overlap: int = 0) -> MCPServer:
    """A tenant-scoped server whose ``whoami`` reports the caller.

    With ``overlap=n``, each ``whoami`` call waits until *n* calls are in
    flight at once before answering, so a passing test proves the
    invocations really did run concurrently on the server.
    """
    server = MCPServer("crm", require_tenant=True)
    server.add_middleware(AuthMiddleware(JWTAuth(secret=SECRET), tenant_claim="org"))
    in_flight = 0
    all_arrived = asyncio.Event()

    @server.tool(read_only_hint=True)
    async def whoami(ctx: RequestContext) -> dict:
        """Show which user and organisation this server thinks is calling."""
        nonlocal in_flight
        in_flight += 1
        try:
            if overlap:
                if in_flight >= overlap:
                    all_arrived.set()
                await asyncio.wait_for(all_arrived.wait(), 10)
        finally:
            in_flight -= 1
        return {"user": ctx.client.client_id, "tenant": ctx.client.tenant_id}

    return server


# Local test servers stay inside a fixed, reserved port range.
_TEST_PORTS = range(8500, 8510)


def _free_test_port() -> int:
    for port in _TEST_PORTS:
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError("no free port in 8500-8509 for the test MCP server")


@asynccontextmanager
async def _serve(server: MCPServer, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[str]:
    """Run *server* over Streamable HTTP and yield its ``/mcp`` URL."""
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


class _CallWhoami(BaseChatModel):
    """Calls ``whoami`` once, then answers with the tool's result verbatim."""

    @property
    def _llm_type(self) -> str:
        return "call-whoami"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _CallWhoami:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if results:
            message = AIMessage(content=str(results[-1].content))
        else:
            message = AIMessage(
                content="",
                tool_calls=[{"name": "whoami", "args": {}, "id": "call-whoami"}],
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


async def _whoami(agent: Any, caller: CallerContext | None) -> dict:
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Who am I?"}]}, caller=caller
    )
    return json.loads(result["messages"][-1].content)


# =====================================================================
# End to end: build_agent + real HTTP MCPServer with JWTAuth
# =====================================================================


class TestAgentForwardsCallerToken:
    async def test_concurrent_callers_each_see_their_own_identity(self, monkeypatch):
        async with _serve(_crm_server(overlap=2), monkeypatch) as url:
            # The trap from the multi-tenant guide: one agent, built with the
            # first user's token, shared by every user.
            agent = await build_agent(
                model=_CallWhoami(),
                servers={"crm": HTTPServerSpec(url=url, bearer_token=ALICE.bearer_token)},
            )
            try:
                alice, bob = await asyncio.gather(_whoami(agent, ALICE), _whoami(agent, BOB))
            finally:
                await agent.shutdown()

        assert alice == {"user": "alice", "tenant": "acme"}
        assert bob == {"user": "bob", "tenant": "globex"}

    async def test_many_concurrent_callers_never_cross(self, monkeypatch):
        callers = [
            CallerContext(
                user_id=f"user{i}", tenant_id=f"t{i}", bearer_token=_token(f"user{i}", f"t{i}")
            )
            for i in range(8)
        ]
        async with _serve(_crm_server(overlap=len(callers)), monkeypatch) as url:
            agent = await build_agent(
                model=_CallWhoami(),
                servers={"crm": HTTPServerSpec(url=url, bearer_token=SERVICE_TOKEN)},
            )
            try:
                seen = await asyncio.gather(*(_whoami(agent, c) for c in callers * 2))
            finally:
                await agent.shutdown()

        assert seen == [{"user": c.user_id, "tenant": c.tenant_id} for c in callers * 2]

    async def test_without_a_caller_token_the_spec_credential_is_used(self, monkeypatch):
        async with _serve(_crm_server(), monkeypatch) as url:
            agent = await build_agent(
                model=_CallWhoami(),
                servers={"crm": HTTPServerSpec(url=url, bearer_token=SERVICE_TOKEN)},
            )
            try:
                anonymous = await _whoami(agent, None)
                tokenless = await _whoami(agent, CallerContext(user_id="carol", tenant_id="x"))
            finally:
                await agent.shutdown()

        assert anonymous == {"user": "crm-agent", "tenant": "platform"}
        assert tokenless == {"user": "crm-agent", "tenant": "platform"}

    async def test_server_can_opt_out(self, monkeypatch):
        async with _serve(_crm_server(), monkeypatch) as url:
            agent = await build_agent(
                model=_CallWhoami(),
                servers={
                    "crm": HTTPServerSpec(
                        url=url, bearer_token=SERVICE_TOKEN, forward_caller_token=False
                    )
                },
            )
            try:
                seen = await _whoami(agent, BOB)
            finally:
                await agent.shutdown()

        assert seen == {"user": "crm-agent", "tenant": "platform"}

    async def test_rejected_caller_token_fails_only_that_call(self, monkeypatch):
        forged = CallerContext(
            user_id="mallory",
            tenant_id="acme",
            bearer_token=JWTAuth(secret="not-the-server-secret-0123456789abcdef").create_token(
                {"sub": "mallory", "org": "acme"}
            ),
        )
        async with _serve(_crm_server(), monkeypatch) as url:
            agent = await build_agent(
                model=_CallWhoami(),
                servers={"crm": HTTPServerSpec(url=url, bearer_token=SERVICE_TOKEN)},
            )
            try:
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "Who am I?"}]}, caller=forged
                )
                # The failure is the forged token's alone: the tool still
                # works, as its own caller, for everyone else.
                bob = await _whoami(agent, BOB)
            finally:
                await agent.shutdown()

        tool_error = next(m for m in result["messages"] if isinstance(m, ToolMessage)).content
        assert "401 Unauthorized" in tool_error
        assert "crm-agent" not in tool_error and "platform" not in tool_error
        assert forged.bearer_token not in tool_error
        assert bob == {"user": "bob", "tenant": "globex"}


# =====================================================================
# MCPMultiClient / MCPClient units
# =====================================================================


class TestWithBearerToken:
    def test_replaces_authorization_in_any_case_and_keeps_other_headers(self):
        base = MCPClient(
            url="http://example.test/mcp",
            headers={"Authorization": "Bearer agent", "x-trace": "1"},
            api_key="k",
        )
        clone = base.with_bearer_token("user-token")

        assert clone.headers == {
            "x-trace": "1",
            "x-api-key": "k",
            "authorization": "Bearer user-token",
        }
        assert base.headers["Authorization"] == "Bearer agent"

    def test_stdio_cannot_carry_a_token(self):
        client = MCPClient(transport="stdio", command="python")
        assert client.supports_bearer_token is False
        with pytest.raises(MCPClientError, match="stdio"):
            client.with_bearer_token("t")


class _StubClient:
    """Stands in for ``MCPClient``: records connects, closes and calls."""

    def __init__(self, transport: str = "http", token: str | None = None) -> None:
        self.transport = transport
        self.token = token
        self.clones: list[_StubClient] = []
        self.entered = 0
        self.exited = 0
        self.calls: list[str] = []

    @property
    def supports_bearer_token(self) -> bool:
        return self.transport != "stdio"

    def with_bearer_token(self, token: str) -> _StubClient:
        clone = _StubClient(self.transport, token)
        self.clones.append(clone)
        return clone

    async def __aenter__(self) -> _StubClient:
        self.entered += 1
        return self

    async def __aexit__(self, *exc: Any) -> None:
        self.exited += 1

    async def list_tools(self) -> list[Any]:
        from mcp.types import Tool

        return [Tool(name="whoami", inputSchema={"type": "object"})]

    async def call_tool(self, name: str, arguments: Any = None) -> str:
        self.calls.append(name)
        await asyncio.sleep(0.01)
        return f"{name} as {self.token}"


class TestMultiClientPerCallerSessions:
    async def test_one_session_per_token_shared_by_concurrent_calls(self):
        base = _StubClient()
        async with MCPMultiClient({"crm": base}) as multi:  # type: ignore[dict-item]
            await multi.list_tools()
            results = await asyncio.gather(
                *(multi.call_tool("whoami", bearer_token=t) for t in ["a", "b", "a", "b", "a"])
            )

        assert results == [
            "whoami as a",
            "whoami as b",
            "whoami as a",
            "whoami as b",
            "whoami as a",
        ]
        assert sorted(c.token for c in base.clones) == ["a", "b"]
        assert all(c.entered == 1 and c.exited == 1 for c in base.clones)
        assert base.calls == []

    async def test_stdio_ignores_the_token_and_warns_once(self, caplog):
        base = _StubClient(transport="stdio")
        with caplog.at_level(logging.WARNING, logger="promptise.mcp.client"):
            async with MCPMultiClient({"fs": base}) as multi:  # type: ignore[dict-item]
                await multi.list_tools()
                await multi.call_tool("whoami", bearer_token="a")
                await multi.call_tool("whoami", bearer_token="b")

        assert base.calls == ["whoami", "whoami"]
        assert base.clones == []
        warnings = [r for r in caplog.records if "stdio" in r.getMessage()]
        assert len(warnings) == 1
        assert "'fs'" in warnings[0].getMessage()


class TestCallerSessionPool:
    async def test_least_recently_used_idle_session_is_closed_over_the_cap(self):
        base = _StubClient()
        pool = CallerSessionPool(max_sessions=2)
        for token in ["a", "b", "c"]:
            async with pool.lease("crm", base, token):  # type: ignore[arg-type]
                pass
        await asyncio.sleep(0)

        by_token = {c.token: c for c in base.clones}
        assert by_token["a"].exited == 1
        assert by_token["b"].exited == 0 and by_token["c"].exited == 0
        assert len(pool) == 2
        await pool.aclose()
        assert all(c.exited == 1 for c in base.clones)

    async def test_session_in_use_is_never_closed(self):
        base = _StubClient()
        pool = CallerSessionPool(max_sessions=1)
        async with pool.lease("crm", base, "a") as held:  # type: ignore[arg-type]
            async with pool.lease("crm", base, "b"):  # type: ignore[arg-type]
                pass
            assert held.exited == 0
        await pool.aclose()

    async def test_idle_sessions_expire(self):
        base = _StubClient()
        pool = CallerSessionPool(idle_timeout=0.0)
        async with pool.lease("crm", base, "a"):  # type: ignore[arg-type]
            pass
        await asyncio.sleep(0.01)
        async with pool.lease("crm", base, "b"):  # type: ignore[arg-type]
            pass
        await asyncio.sleep(0)

        by_token = {c.token: c for c in base.clones}
        assert by_token["a"].exited == 1
        await pool.aclose()

    async def test_failed_session_is_replaced(self):
        base = _StubClient()
        pool = CallerSessionPool()
        with pytest.raises(MCPClientError):
            async with pool.lease("crm", base, "a"):  # type: ignore[arg-type]
                raise MCPClientError("connection lost")
        async with pool.lease("crm", base, "a"):  # type: ignore[arg-type]
            pass
        await pool.aclose()

        assert len(base.clones) == 2
        assert all(c.exited == 1 for c in base.clones)

    async def test_tokens_are_not_kept_as_keys(self):
        pool = CallerSessionPool()
        async with pool.lease("crm", _StubClient(), "secret-token"):  # type: ignore[arg-type]
            pass
        assert all("secret-token" not in key[1] for key in pool._entries)
        await pool.aclose()
