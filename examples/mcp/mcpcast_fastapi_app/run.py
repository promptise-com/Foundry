"""Make your own FastAPI app MCP-ready — end to end, with real calls all the way.

``app.py`` is an ordinary FastAPI service (the Bookshelf API). This driver does
exactly what you would do to your own app, in one run:

  1. START     serve ``app.py`` in-process with uvicorn on a free loopback port
  2. GENERATE  ``promptise.mcpcast`` reads ``http://127.0.0.1:<port>/openapi.json``
               and writes an editable MCP server project to ``generated/``
               (profile ``standard``: reads open, writes approval-gated;
               auth ``env-token``: the server presents one bearer token upstream)
  3. DRIVE     ``build_agent("openai:gpt-5-mini")`` launches ``generated/server.py``
               over the real MCP stdio transport and answers a question by
               calling the live app through the generated tools
  4. GOVERN    the same agent tries a write — denied fail-closed over stdio
               (no human can be asked); then, in-process with an approver, the
               same call runs against the live app
  5. SHIP      what to run once this is your product's MCP server

``app.py`` is a FastAPI app, so ``fastapi`` must be installed
(``.venv/bin/python -m pip install fastapi``; ``promptise[dev]`` includes it).
Beyond that only ``OPENAI_API_KEY`` is needed. Steps 1-2 run offline; the script
stops with a clear message before step 3 if the key is missing.

Run:
    OPENAI_API_KEY=... .venv/bin/python examples/mcp/mcpcast_fastapi_app/run.py
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import uvicorn
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from promptise import StdioServerSpec, build_agent
from promptise.approval import ApprovalRequest
from promptise.mcp.server import TestClient
from promptise.mcpcast import (
    AuthMode,
    DroppedOp,
    MCPcastPlan,
    SafetyProfile,
    load_generated_server,
    mcpcast,
    write_project,
)
from promptise.models import load_dotenv_if_present

try:
    import app as bookshelf  # the Bookshelf API next to this file — a FastAPI app
except ModuleNotFoundError as exc:
    if (exc.name or "").partition(".")[0] != "fastapi":
        raise
    raise SystemExit(
        "app.py is a FastAPI app and fastapi is not a Promptise dependency — "
        ".venv/bin/python -m pip install fastapi"
    ) from None

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated"
MODEL = "openai:gpt-5-mini"
# What the generated server sends upstream as the Authorization header. In a
# desktop client this goes in the client's own config (see generated/README.md).
UPSTREAM_TOKEN = f"Bearer {bookshelf.DEMO_TOKEN}"
QUESTION = "Which books by Ursula K. Le Guin do we have, and which of them is the oldest?"
WRITE_REQUEST = "Add the note 'signed first edition' to The Dispossessed."


def section(number: int, title: str) -> None:
    """Print a numbered section header."""
    print(f"\n{'=' * 78}\n{number}. {title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
# 1. Start your app
# ---------------------------------------------------------------------------


def start_app() -> tuple[uvicorn.Server, str]:
    """Serve ``app.py`` in a background thread and return the server and its origin."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config = uvicorn.Config(bookshelf.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive():
            raise SystemExit(f"uvicorn failed to start on port {port} (see the error above)")
        if time.monotonic() > deadline:
            raise SystemExit(f"uvicorn did not start on port {port} within 10 s")
        time.sleep(0.05)
    return server, f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# 2. Generate the MCP server from the running app's spec URL
# ---------------------------------------------------------------------------


def generate(origin: str) -> MCPcastPlan:
    """``/openapi.json`` -> risk-classified plan -> ``generated/`` project."""
    section(2, "Generate the MCP server from the app's /openapi.json")
    spec_url = f"{origin}/openapi.json"
    plan = mcpcast(
        spec_url,
        profile=SafetyProfile.STANDARD,
        auth=AuthMode.ENV_TOKEN,
        name="bookshelf",
    )
    # FastAPI emits no `servers` block, so the API base was resolved against
    # the URL the spec was fetched from — the app's own origin.
    print(f"  spec:     {spec_url}")
    print(f"  base_url: {plan.api.base_url}   (no servers block -> the spec URL's origin)")

    # Your review pass: a liveness probe is not something an agent should call.
    # `--curate` would drop it for you; deterministic generation keeps every
    # read, so move it to `dropped` — the plan is the file you own.
    health = plan.tool("health_check")
    plan = MCPcastPlan(
        api=plan.api,
        profile=plan.profile,
        tools=[t for t in plan.tools if t.name != "health_check"],
        dropped=[
            *plan.dropped,
            DroppedOp(
                operation_id=health.operations[0],
                reason="operational endpoint — for the load balancer, not an agent",
            ),
        ],
    )
    for path in write_project(plan, OUT):
        print(f"  wrote {path.relative_to(HERE)}")

    print(f"\n  {'tool':<16}{'risk':<8}{'approval':<10}{'upstream operation':<26}params")
    for tool in plan.tools:
        route = tool.routes[0]
        gate = "required" if tool.requires_approval else "-"
        print(
            f"  {tool.name:<16}{tool.risk.value:<8}{gate:<10}"
            f"{route.method + ' ' + route.path:<26}{', '.join(tool.visible_params)}"
        )
    print("\n  not exposed (each with its reason, recorded in the plan):")
    for dropped in plan.dropped:
        print(f"    {dropped.operation_id}: {dropped.reason}")
    return plan


# ---------------------------------------------------------------------------
# 3 + 4a. A real agent over MCP stdio, against the live app
# ---------------------------------------------------------------------------


def final_text(result: Any) -> str:
    """The final assistant text of an agent invocation."""
    content = result["messages"][-1].content
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def show_calls(result: Any) -> None:
    """Print every tool call the agent made, with the server's structured errors."""
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                print(f"    {call['name']}({json.dumps(call['args'], ensure_ascii=False)})")
        elif isinstance(message, ToolMessage) and '"error"' in str(message.content):
            error = json.loads(str(message.content))["error"]
            print(f"      -> {error['code']}: {error['message']}")


async def drive_over_stdio() -> None:
    """The generated server, launched by the agent exactly as Claude Desktop would."""
    section(3, "A real agent over MCP stdio -> generated/server.py -> the live app")
    agent = await build_agent(
        model=MODEL,
        servers={
            "bookshelf": StdioServerSpec(
                command=sys.executable,
                args=[str(OUT / "server.py")],
                env={"MCPCAST_UPSTREAM_TOKEN": UPSTREAM_TOKEN},
            )
        },
        instructions=(
            "You are the Bookshelf assistant. Answer from what the tools return. "
            "Be brief. If a tool call is denied, say so and stop."
        ),
        max_agent_iterations=6,
    )
    try:
        print(f"  question: {QUESTION}\n  tools the agent chose:")
        result = await agent.ainvoke({"messages": [HumanMessage(content=QUESTION)]})
        show_calls(result)
        print(f"  answer: {final_text(result)}")

        section(4, "Writes: denied fail-closed over stdio, executed once a human approves")
        print(
            "  a) this MCP client does not support elicitation, so nobody can be asked and the gate denies:"
        )
        print(f"  request: {WRITE_REQUEST}\n  tools the agent chose:")
        result = await agent.ainvoke({"messages": [HumanMessage(content=WRITE_REQUEST)]})
        show_calls(result)
        print(f"  answer: {final_text(result)}")
    finally:
        await agent.shutdown()


# ---------------------------------------------------------------------------
# 4b. The same write, in-process, with an approver
# ---------------------------------------------------------------------------


def import_generated() -> ModuleType:
    """Import the generated project — its ``build_server()`` takes an approval handler."""
    return load_generated_server(OUT / "server.py")


async def approve_and_execute() -> None:
    """A reviewer says yes, the PATCH reaches the live app, and the read confirms it."""
    print("\n  b) in-process, with a human (here: a callback) who approves:")
    os.environ["MCPCAST_UPSTREAM_TOKEN"] = UPSTREAM_TOKEN  # env-token mode reads it per call

    def reviewer(request: ApprovalRequest) -> bool:
        print(f"    approval requested: {request.tool_name} {json.dumps(request.arguments)}")
        return True

    client = TestClient(import_generated().build_server(approval_handler=reviewer))
    (reply,) = await client.call_tool("get_book", {"book_id": 3})
    print(
        f"    get_book(3) before: notes={json.loads(reply.text)['notes']!r}  (the denied call changed nothing)"
    )
    (reply,) = await client.call_tool(
        "update_book", {"book_id": 3, "notes": "signed first edition"}
    )
    print(f"    update_book -> {reply.text}")
    (reply,) = await client.call_tool("get_book", {"book_id": 3})
    print(f"    get_book(3) after:  notes={json.loads(reply.text)['notes']!r}")


# ---------------------------------------------------------------------------


async def main() -> None:
    section(1, "Start the Bookshelf app (app.py) with uvicorn")
    app_server, origin = start_app()
    print(f"  serving {origin}  (bearer token: {bookshelf.DEMO_TOKEN!r})")
    try:
        generate(origin)
        load_dotenv_if_present()  # .env next to the project, as build_agent() would
        if not os.environ.get("OPENAI_API_KEY"):
            print("\nSteps 3-4 drive a real agent: set OPENAI_API_KEY and run this again.")
            print("  export OPENAI_API_KEY=sk-...")
            sys.exit(1)
        await drive_over_stdio()
        await approve_and_execute()
    finally:
        app_server.should_exit = True

    section(5, "Ship it")
    print("  generated/ is a real project: edit mcpcast.plan.yaml (never the package), regenerate.")
    print(
        "    cd generated && pytest                    # its own tests: listing, routing, the gate"
    )
    print("    pip install -e generated && bookshelf-mcp --transport http --port 8080")
    print("    cd generated && promptise serve server:server --transport http --port 8080")
    print(
        "    docker build -t bookshelf-mcp generated && docker run -i --env-file generated/.env \\"
    )
    print("      -e MCPCAST_BASE_URL=https://api.yourcompany.com bookshelf-mcp")
    print("  desktop clients (stdio) get the token in their own config: see generated/README.md")
    print(
        "  env-token = one shared credential and no caller authentication, so every HTTP form"
        " above binds loopback only"
    )
    print(
        "  and the image serves stdio; publish only behind an authenticating gateway"
        " (--host 0.0.0.0 --public)"
    )


if __name__ == "__main__":
    asyncio.run(main())
