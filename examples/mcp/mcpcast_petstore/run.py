"""MCPcast the public Swagger Petstore and drive it with a real agent — runnable demo.

Turns ``petstore.yaml`` (a trimmed copy of the public Petstore v3 spec) into a
safe, agent-ready MCP server with the ``promptise.mcpcast`` Python API — the same
pipeline as ``promptise mcpcast`` — then proves the result works end to end:

  1. Generate ``generated/`` (mcpcast.plan.yaml, server.py, README.md) under the
     ``full`` profile: every operation becomes a tool, every write/delete is
     approval-gated server-side, and the plan explains anything it leaves out.
  2. A real ``build_agent()`` (openai:gpt-5-mini) connects to the generated
     server over the real MCP stdio transport and answers a question by calling
     the LIVE petstore API through a read tool.
  3. In-process via ``TestClient``: ``delete_pet`` is DENIED when no approver can
     be reached (fail-closed), then runs once an approver says yes — against a
     mock upstream, so no real data is ever touched.

Run:
    OPENAI_API_KEY=... .venv/bin/python examples/mcp/mcpcast_petstore/run.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
from langchain_core.messages import HumanMessage

from promptise import StdioServerSpec, build_agent
from promptise.mcp.server import TestClient
from promptise.mcpcast import (
    AuthMode,
    MCPcastPlan,
    SafetyProfile,
    load_generated_server,
    mcpcast,
    write_project,
)
from promptise.models import load_dotenv_if_present

HERE = Path(__file__).resolve().parent
# The spec is passed by its relative name (main() runs from this directory), so the
# `spec_source` recorded in the plan and in generated docstrings is portable — never a
# developer's absolute path.
SPEC = "petstore.yaml"
OUT = HERE / "generated"
MODEL = "openai:gpt-5-mini"
QUESTION = "Which pets are currently available? List up to five names."


def generate() -> MCPcastPlan:
    """Step 1: spec -> classify -> plan -> generated project, printed as a table."""
    print("=== 1. Generate the MCP server project (profile=full, auth=none) ===")
    plan = mcpcast(SPEC, profile=SafetyProfile.FULL, auth=AuthMode.NONE, name="petstore")
    for path in write_project(plan, OUT):
        print(f"  wrote {path.relative_to(HERE)}")

    print(f"\n  {'tool':<24}{'risk':<13}{'approval':<10}upstream operation")
    for tool in plan.tools:
        route = tool.routes[0]
        gate = "required" if tool.requires_approval else "-"
        print(f"  {tool.name:<24}{tool.risk.value:<13}{gate:<10}{route.method} {route.path}")
    print("\n  not exposed:")
    for dropped in plan.dropped:
        print(f"  - {dropped.operation_id}: {dropped.reason}")
    if not plan.dropped:
        print("  (none — profile 'full' exposes every operation and gates every non-read)")

    # The default profile is read-only: the same spec yields a smaller surface,
    # and every excluded operation is recorded in the plan with its reason.
    read_only = mcpcast(SPEC, auth=AuthMode.NONE, name="petstore")
    first = read_only.dropped[0]
    print(
        f"\n  default read-only profile: {len(read_only.tools)} tools, "
        f"{len(read_only.dropped)} not exposed (e.g. {first.operation_id}: {first.reason})"
    )
    return plan


def final_text(result: Any) -> str:
    """The final assistant text from an agent invocation result."""
    content = result["messages"][-1].content
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


async def ask_live_api() -> None:
    """Step 2: a real agent drives the generated server over MCP stdio."""
    print("\n=== 2. Real agent over MCP stdio -> live petstore API ===")
    agent = await build_agent(
        model=MODEL,
        servers={
            "petstore": StdioServerSpec(command=sys.executable, args=[str(OUT / "server.py")]),
        },
        instructions="Answer questions about the pet store with the tools. Be concise.",
    )
    try:
        print(f"  question: {QUESTION}")
        result = await agent.ainvoke({"messages": [HumanMessage(content=QUESTION)]})
        print(f"  answer:   {final_text(result)}")
    finally:
        await agent.shutdown()


def import_generated() -> ModuleType:
    """Import the generated project through its launcher (what ``--eval`` and tests do)."""
    return load_generated_server(OUT / "server.py")


class MockUpstream:
    """Stands in for petstore3.swagger.io: records every request, answers 200."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"message": "Pet deleted"}, request=request)


async def approval_gate_demo() -> None:
    """Step 3: the destructive tool is fail-closed, then runs once approved."""
    print("\n=== 3. Human approval gate on delete_pet (in-process, mock upstream) ===")
    module = import_generated()
    upstream = MockUpstream()
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as http:
        # Default approver is ElicitationApprover: TestClient has no live MCP
        # session to ask a human through, so the gate denies — fail-closed.
        client = TestClient(module.build_server(http_client=http))
        (reply,) = await client.call_tool("delete_pet", {"petId": 10})
        error = json.loads(reply.text)["error"]
        print(f"  no approver reachable -> {error['code']}: {error['message']}")
        print(f"  DELETE requests that reached upstream: {len(upstream.requests)}")

        # An approver (here: a callback that always says yes) releases the call.
        approved = module.build_server(approval_handler=lambda request: True, http_client=http)
        (reply,) = await TestClient(approved).call_tool("delete_pet", {"petId": 10})
        print(f"  approver says yes     -> {reply.text}")
        sent = upstream.requests[-1]
        print(f"  upstream received     -> {sent.method} {sent.url}")
    print("  (upstream was an httpx.MockTransport — no real pet was deleted)")


async def main() -> None:
    os.chdir(HERE)
    generate()
    load_dotenv_if_present()  # .env next to the project, as build_agent() would
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nOPENAI_API_KEY is not set: step 2 needs a real model. Export it and rerun.")
        sys.exit(1)
    # petstore3.swagger.io is a shared public demo: when it is down the generated tool
    # returns a structured UPSTREAM_ERROR that the agent reports — nothing is caught here.
    await ask_live_api()
    await approval_gate_demo()

    print("\n=== 4. Next steps ===")
    print("  edit generated/mcpcast.plan.yaml, then regenerate and score it with a real agent:")
    print("    promptise mcpcast generated/mcpcast.plan.yaml --out generated --eval")
    print("  connect Claude Desktop / Claude Code / Cursor: see generated/README.md")


if __name__ == "__main__":
    asyncio.run(main())
