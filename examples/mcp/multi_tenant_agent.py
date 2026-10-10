"""One agent, many tenants — runnable end to end, offline.

Demonstrates the isolation guarantees a multi-tenant deployment relies on:

  1. Per-caller tokens — one agent, built once, serves two tenants
     concurrently. Each invocation's ``CallerContext.bearer_token`` is sent on
     its tool calls (one MCP session per caller), so the server sees the
     user, not the credential the agent was built with.
  2. Audience-checked tokens — ``JWTAuth(audience=..., issuer=...)`` refuses
     a token minted for another service that shares the secret.
  3. Caller-scoped caching — ``CacheMiddleware`` keys every entry on the
     caller, so one tenant's cached result never answers another.
  4. Hidden tools — ``hide_unauthorized_tools=True`` lists a tenant-only tool
     to that tenant alone.

A real ``MCPServer`` runs over Streamable HTTP on a loopback port. A scripted
chat model stands in for the LLM (it calls ``whoami`` once), so the demo
needs no API key; swap in ``model="openai:gpt-5-mini"`` for a real one.

Run:
    .venv/bin/python examples/mcp/multi_tenant_agent.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from promptise import CallerContext, build_agent
from promptise.config import HTTPServerSpec
from promptise.mcp.server import (
    AuthMiddleware,
    CacheMiddleware,
    HasTenant,
    JWTAuth,
    MCPServer,
    RequestContext,
    TestClient,
)

SECRET = "demo-only-secret-change-me-0123456789abcdef"
AUDIENCE = "api://crm"
ISSUER = "https://auth.example.com"

# Your identity provider issues these; JWTAuth.create_token stands in for it.
issuer = JWTAuth(secret=SECRET, audience=AUDIENCE, issuer=ISSUER)

ACCOUNTS = {"acme": ["Initech", "Umbrella"], "globex": ["Hooli", "Pied Piper"]}


def build_server() -> MCPServer:
    server = MCPServer("crm", require_tenant=True, hide_unauthorized_tools=True)
    server.add_middleware(
        AuthMiddleware(JWTAuth(secret=SECRET, audience=AUDIENCE, issuer=ISSUER), tenant_claim="org")
    )
    server.add_middleware(CacheMiddleware(ttl=60))  # after auth: keyed per caller

    @server.tool(read_only_hint=True)
    async def whoami(ctx: RequestContext) -> dict:
        """Show which user and organisation this server thinks is calling."""
        return {"user": ctx.client.client_id, "tenant": ctx.client.tenant_id}

    @server.tool(read_only_hint=True)
    async def list_accounts(ctx: RequestContext) -> list[str]:
        """Accounts in the caller's organisation."""
        return ACCOUNTS[ctx.client.tenant_id]

    @server.tool(read_only_hint=True, guards=[HasTenant("acme")])
    async def forecast_renewals(ctx: RequestContext) -> list[str]:
        """Pilot feature, switched on for Acme only."""
        return ["Initech renews 2026-11-30"]

    return server


class ScriptedModel(BaseChatModel):
    """Stands in for the LLM: calls ``whoami``, then reports its result."""

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if results:
            message = AIMessage(content=str(results[-1].content))
        else:
            message = AIMessage(
                content="", tool_calls=[{"name": "whoami", "args": {}, "id": "call-1"}]
            )
        return ChatResult(generations=[ChatGeneration(message=message)])


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def per_caller_tokens(server: MCPServer) -> None:
    print("=== 1. One agent, two tenants, concurrently ===")
    port = free_port()
    serving = asyncio.create_task(server.run_async(transport="http", host="127.0.0.1", port=port))
    await asyncio.sleep(1.0)  # let uvicorn bind
    try:
        agent = await build_agent(
            model=ScriptedModel(),
            servers={
                "crm": HTTPServerSpec(
                    url=f"http://127.0.0.1:{port}/mcp",
                    # The agent's own credential: opens the connection and
                    # discovers tools. Tool calls use each caller's token.
                    bearer_token=issuer.create_token({"sub": "crm-agent", "org": "acme"}),
                )
            },
        )
        alice = CallerContext(
            user_id="alice",
            tenant_id="acme",
            bearer_token=issuer.create_token({"sub": "alice", "org": "acme"}),
        )
        bob = CallerContext(
            user_id="bob",
            tenant_id="globex",
            bearer_token=issuer.create_token({"sub": "bob", "org": "globex"}),
        )
        ask = {"messages": [{"role": "user", "content": "Who am I?"}]}
        try:
            results = await asyncio.gather(
                agent.ainvoke(ask, caller=alice), agent.ainvoke(ask, caller=bob)
            )
        finally:
            await agent.shutdown()
        for name, result in zip(("alice", "bob"), results, strict=True):
            print(f"  {name:5} -> server saw {result['messages'][-1].content}")
    finally:
        # The demo stops the server by cancelling its task; silence the
        # traceback uvicorn logs for a cancelled lifespan.
        logging.getLogger("uvicorn.error").setLevel(logging.CRITICAL)
        serving.cancel()
        await asyncio.gather(serving, return_exceptions=True)


async def audience_checks(server: MCPServer) -> None:
    print("\n=== 2. A token for another service is refused ===")
    billing_token = JWTAuth(secret=SECRET).create_token(
        {"sub": "alice", "org": "acme", "aud": "api://billing", "iss": ISSUER}
    )
    client = TestClient(server, meta={"authorization": f"Bearer {billing_token}"})
    error = json.loads((await client.call_tool("whoami"))[0].text)["error"]
    print(f"  aud=api://billing -> {error['code']}: {error['message']}")


async def caller_scoped_cache(server: MCPServer) -> None:
    print("\n=== 3. Cached results never cross tenants ===")
    for sub, org in (("alice", "acme"), ("bob", "globex"), ("alice", "acme")):
        token = issuer.create_token({"sub": sub, "org": org})
        client = TestClient(server, meta={"authorization": f"Bearer {token}"})
        print(f"  {org:6} list_accounts -> {(await client.call_tool('list_accounts'))[0].text}")


async def hidden_tools(server: MCPServer) -> None:
    print("\n=== 4. Each tenant lists only the tools it may call ===")
    for sub, org in (("alice", "acme"), ("bob", "globex")):
        token = issuer.create_token({"sub": sub, "org": org})
        tools = await TestClient(server, meta={"authorization": f"Bearer {token}"}).list_tools()
        print(f"  {org:6} sees {sorted(t.name for t in tools)}")


async def main() -> None:
    await per_caller_tokens(build_server())
    server = build_server()
    await audience_checks(server)
    await caller_scoped_cache(server)
    await hidden_tools(server)


if __name__ == "__main__":
    asyncio.run(main())
