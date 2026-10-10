"""What a client sees when an MCP server rejects its credentials.

Starts an API-key protected MCP server on localhost in a subprocess, then
connects to it three times:

1. Without credentials -> ``MCPConnectionRejectedError`` (401), immediately.
2. With the wrong key  -> ``MCPConnectionRejectedError`` (401), immediately.
3. With the right key  -> the tools are listed and one is called.

No LLM or API key is needed.

Run:
    python examples/mcp/client_auth_errors.py
"""

from __future__ import annotations

import asyncio
import sys

from promptise.mcp.client import MCPClient, MCPConnectionRejectedError
from promptise.mcp.server import APIKeyAuth, AuthMiddleware, MCPServer

HOST, PORT = "127.0.0.1", 8765
URL = f"http://{HOST}:{PORT}/mcp"
API_KEY = "test-key-123"


def build_server() -> MCPServer:
    server = MCPServer("orders", require_auth=True)
    server.add_middleware(AuthMiddleware(APIKeyAuth(keys={API_KEY: "support-agent"})))

    @server.tool()
    async def get_order_status(order_id: str) -> dict:
        """Look up an order."""
        return {"order_id": order_id, "status": "shipped"}

    return server


async def wait_until_listening(timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        try:
            _, writer = await asyncio.open_connection(HOST, PORT)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            if loop.time() > deadline:
                raise
            await asyncio.sleep(0.05)


async def connect(label: str, api_key: str | None) -> None:
    print(f"\n--- {label}")
    try:
        async with MCPClient(url=URL, api_key=api_key) as client:
            tools = await client.list_tools()
            print("tools:", [t.name for t in tools])
            result = await client.call_tool("get_order_status", {"order_id": "A-1001"})
            print("result:", result.content[0].text)
    except MCPConnectionRejectedError as exc:
        print(f"rejected with HTTP {exc.status_code}: {exc}")


async def main() -> None:
    server = await asyncio.create_subprocess_exec(
        sys.executable,
        __file__,
        "--serve",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await wait_until_listening()
        await connect("no credentials", api_key=None)
        await connect("wrong API key", api_key="not-the-key")
        await connect("correct API key", api_key=API_KEY)
    finally:
        server.terminate()
        await server.wait()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        build_server().run(transport="http", host=HOST, port=PORT)
    else:
        asyncio.run(main())
