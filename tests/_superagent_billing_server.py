"""Tiny stdio MCP server for the .superagent team tests."""

from __future__ import annotations

from promptise.mcp.server import MCPServer

server = MCPServer("billing")

ACCOUNTS = {"acme": {"plan": "Team", "seats": 12}}


@server.tool()
async def get_account(customer_id: str) -> dict:
    """Get a customer's plan and number of seats.

    Args:
        customer_id: The customer's ID, for example "acme".
    """
    return ACCOUNTS.get(customer_id.strip().lower(), {"error": f"No customer {customer_id}."})


if __name__ == "__main__":
    server.run(transport="stdio")
