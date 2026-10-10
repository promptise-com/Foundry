"""An MCP client that survives a server redeploy, plus the HTTP health probes.

Starts an inventory MCP server on localhost in a subprocess, then:

1. Probes ``GET /health`` and ``GET /health/ready`` the way Docker or
   Kubernetes would (a plain ``GET /mcp`` is answered 406).
2. Connects with ``MCPMultiClient`` and calls a tool.
3. Redeploys the server: kills the process and starts a new version that
   also has a ``reserve`` tool.  The old session id is unknown to the new
   process (it answers 404), so the client opens a new session, retries the
   call once and re-discovers the server's tools -- the caller sees none of it.
4. Connects to the server's base URL without ``/mcp``, which fails with a
   typed ``MCPConnectionRejectedError`` (404) that says to check the URL.

No LLM or API key is needed.

Run:
    python examples/mcp/http_server_restart.py
"""

from __future__ import annotations

import asyncio
import sys

import httpx

from promptise.mcp.client import MCPClient, MCPConnectionRejectedError, MCPMultiClient
from promptise.mcp.server import HealthCheck, MCPServer

HOST, PORT = "127.0.0.1", 8766
BASE = f"http://{HOST}:{PORT}"
STOCK = {"SKU-1": 42, "SKU-2": 0}


def build_server(version: str) -> MCPServer:
    server = MCPServer("inventory", version=version)

    @server.tool(read_only_hint=True)
    async def check_stock(sku: str) -> dict:
        """Units in stock for a product.

        Args:
            sku: The product code, for example "SKU-1".
        """
        return {"sku": sku, "in_stock": STOCK.get(sku.upper(), 0), "served_by": version}

    if version == "2.0.0":

        @server.tool()
        async def reserve(sku: str) -> dict:
            """Reserve one unit of a product.

            Args:
                sku: The product code.
            """
            return {"sku": sku, "reserved": True}

    # Backs GET /health/ready (and the health://readiness MCP resource).
    health = HealthCheck()
    health.add_check("stock_loaded", lambda: bool(STOCK), required_for_ready=True)
    health.register_resources(server)
    return server


async def start_server(version: str) -> asyncio.subprocess.Process:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        __file__,
        "--serve",
        version,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    async with httpx.AsyncClient(timeout=1) as http:
        for _ in range(200):
            try:
                if (await http.get(f"{BASE}/health")).status_code == 200:
                    return process
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.05)
    process.kill()
    raise RuntimeError("server did not start")


async def stop_server(process: asyncio.subprocess.Process) -> None:
    process.terminate()
    await process.wait()


async def main() -> None:
    server = await start_server("1.0.0")
    try:
        print("--- probes")
        async with httpx.AsyncClient() as http:
            for path in ("/health", "/health/ready", "/mcp"):
                response = await http.get(BASE + path)
                print(f"GET {path:<14} -> {response.status_code} {response.text[:70]}")

        print("\n--- redeploy under a connected client")
        multi = MCPMultiClient({"inventory": MCPClient(url=f"{BASE}/mcp")})
        async with multi:
            await multi.list_tools()
            print("tools:", sorted(multi.tool_to_server))
            result = await multi.call_tool("check_stock", {"sku": "SKU-1"})
            print("before:", result.content[0].text)

            await stop_server(server)
            server = await start_server("2.0.0")

            result = await multi.call_tool("check_stock", {"sku": "SKU-1"})
            print("after: ", result.content[0].text)
            print("sessions opened:", multi.servers["inventory"].session_generation)
            print("tools:", sorted(multi.tool_to_server))

        print("\n--- URL without /mcp")
        try:
            async with MCPClient(url=BASE):
                pass
        except MCPConnectionRejectedError as exc:
            print(f"rejected with HTTP {exc.status_code}: {exc}")
    finally:
        await stop_server(server)


if __name__ == "__main__":
    if "--serve" in sys.argv:
        build_server(sys.argv[-1]).run(transport="http", host=HOST, port=PORT)
    else:
        asyncio.run(main())
