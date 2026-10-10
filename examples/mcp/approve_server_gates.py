"""Approving a server-side approval gate from a Promptise client.

Starts an MCP server over stdio whose ``refund`` tool is gated with
``ApprovalGateMiddleware(ElicitationApprover())``, the default approver on
MCPcast-generated servers: the server asks the *calling client* to confirm
every gated call through MCP elicitation.  Then:

1. A client with no elicitation handler calls ``refund`` -> ``APPROVAL_DENIED``.
   It declares no elicitation support, and the gate fails closed.
2. A client whose elicitation requests go to an approval handler calls it
   twice. The reviewer sees the server's message, the tool and the arguments,
   approves the small refund and denies the large one.

``build_agent(approval=CallbackApprovalHandler(reviewer), ...)`` wires the same
handler to every server of an agent.

No LLM or API key is needed.

Run:
    python examples/mcp/approve_server_gates.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from mcp.types import CallToolResult

from promptise.approval import (
    ApprovalDecision,
    ApprovalRequest,
    CallbackApprovalHandler,
    approval_elicitation_callback,
)
from promptise.mcp.client import MCPClient
from promptise.mcp.server import ApprovalGateMiddleware, ElicitationApprover, MCPServer


def build_server() -> MCPServer:
    server = MCPServer("billing")
    server.add_middleware(ApprovalGateMiddleware(ElicitationApprover(), timeout=60))

    @server.tool(requires_approval=True)
    async def refund(order_id: str, amount: float) -> dict:
        """Refund an order."""
        return {"order_id": order_id, "refunded": amount}

    return server


async def reviewer(request: ApprovalRequest) -> ApprovalDecision:
    """Stands in for a human: approves refunds under 100."""
    print(f"  reviewer sees: {request.context_summary}")
    print(f"  call:          {request.tool_name}({request.arguments})")
    amount = float(request.arguments.get("amount", 0))
    if amount < 100:
        return ApprovalDecision(approved=True, reviewer_id="dana")
    return ApprovalDecision(approved=False, reviewer_id="dana", reason="over the limit")


def stdio_client(elicitation_callback: Any = None) -> MCPClient:
    return MCPClient(
        transport="stdio",
        command=sys.executable,
        args=[__file__, "--serve"],
        elicitation_callback=elicitation_callback,
    )


def outcome(result: CallToolResult) -> str:
    """The tool's result, or the error code and message the gate returned."""
    payload = json.loads(getattr(result.content[0], "text", "{}"))
    if "error" in payload:
        return f"{payload['error']['code']}: {payload['error']['message']}"
    return f"ran -> {payload}"


async def main() -> None:
    print("--- client without an elicitation handler")
    async with stdio_client() as client:
        result = await client.call_tool("refund", {"order_id": "A-1", "amount": 25})
        print(" ", outcome(result))

    print("\n--- client routing elicitation to an approval handler")
    client = stdio_client(
        elicitation_callback=approval_elicitation_callback(
            CallbackApprovalHandler(reviewer),
            server_name="billing",
            in_flight=lambda: client.in_flight_calls,
        )
    )
    async with client:
        for amount in (25, 500):
            result = await client.call_tool("refund", {"order_id": "A-1", "amount": amount})
            print(f"  result:        {outcome(result)}\n")


if __name__ == "__main__":
    if "--serve" in sys.argv:
        build_server().run(transport="stdio")
    else:
        asyncio.run(main())
