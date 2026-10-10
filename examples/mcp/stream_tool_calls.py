"""Stream an agent's tool calls and answer, as a chat UI would receive them.

Starts a small orders MCP server over stdio (this file, with ``--serve``),
builds an agent on it, and prints every event of ``astream_with_tools()``
with the time it arrived:

1. Two orders at once -> both lookups start together and run in parallel;
   each ``tool_end`` has its own ``tool_index`` and ``duration_ms`` (~1 s).
2. An unknown order -> the server raises ``ToolError``; the stream reports
   ``success: false`` with the error's message, and the model answers from it.

Each answer streams token by token after the tools, and ``done`` carries it
as ``full_response``.

Run:
    export OPENAI_API_KEY=sk-...
    python examples/mcp/stream_tool_calls.py
"""

from __future__ import annotations

import asyncio
import sys
import time

from promptise import build_agent
from promptise.config import StdioServerSpec
from promptise.mcp.server import MCPServer, ToolError

ORDERS = {
    "A-1001": {"status": "shipped", "carrier": "DHL", "eta": "2026-10-14"},
    "A-1002": {"status": "processing", "carrier": None, "eta": None},
}


def build_server() -> MCPServer:
    server = MCPServer("orders")

    @server.tool()
    async def get_order_status(order_id: str) -> dict:
        """Get an order's status, carrier and expected delivery date.

        Args:
            order_id: The order ID, for example "A-1001".
        """
        await asyncio.sleep(1)  # a slow warehouse API
        order = ORDERS.get(order_id.strip().upper())
        if order is None:
            raise ToolError(f"No order found with ID {order_id}.")
        return {"order_id": order_id.strip().upper(), **order}

    return server


async def ask(agent, question: str) -> None:
    print(f"\n>>> {question}")
    start = time.monotonic()
    tokens = 0
    async for event in agent.astream_with_tools(
        {"messages": [{"role": "user", "content": question}]}
    ):
        elapsed = f"{time.monotonic() - start:5.1f}s"
        if event.type == "token":
            tokens += 1
            if tokens == 1:
                print(f"{elapsed}  answer streaming: ", end="")
            print(event.text, end="", flush=True)
            continue
        if tokens:
            print()
            tokens = 0
        if event.type == "tool_start":
            print(
                f"{elapsed}  tool_start #{event.tool_index} {event.tool_display_name} {event.arguments}"
            )
        elif event.type == "tool_end":
            status = "ok" if event.success else "FAILED"
            print(
                f"{elapsed}  tool_end   #{event.tool_index} {status} "
                f"in {event.duration_ms:.0f} ms: {event.tool_summary}"
            )
        elif event.type == "done":
            print(f"{elapsed}  done ({len(event.full_response)} characters in full_response)")
        elif event.type == "error":
            print(f"{elapsed}  error: {event.message}")


async def main() -> None:
    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={"orders": StdioServerSpec(command=sys.executable, args=[__file__, "--serve"])},
        instructions=(
            "You are a support assistant. Use your tools to answer questions about orders. "
            "Look up several orders at once when asked about more than one."
        ),
    )
    try:
        await ask(agent, "Where are my orders A-1001 and A-1002?")
        await ask(agent, "Where is my order Z-9999?")
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        build_server().run()
    else:
        asyncio.run(main())
