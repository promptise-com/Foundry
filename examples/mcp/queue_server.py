"""Background jobs with MCPQueue: progress, cancellation and per-client ownership.

Starts an API-key protected MCP server on localhost in a subprocess, then
acts as three clients:

1. alice lists the job types ``queue_submit`` advertises, submits a job
   with a missing argument (rejected on submit), then a valid one, and
   polls its progress until it completes.
2. bob tries to read alice's job -> ``JOB_NOT_FOUND`` (jobs are owned by
   the client that submitted them).
3. alice starts a second job and cancels it; ops (admin role) lists
   every job.

No LLM or API key is needed.

Run:
    python examples/mcp/queue_server.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from promptise.mcp.client import MCPClient
from promptise.mcp.server import (
    APIKeyAuth,
    AuthMiddleware,
    CancellationToken,
    MCPQueue,
    MCPServer,
    ProgressReporter,
)

HOST, PORT = "127.0.0.1", 8766
URL = f"http://{HOST}:{PORT}/mcp"
KEYS = {
    "sk-alice": {"client_id": "alice", "roles": ["analyst"]},
    "sk-bob": {"client_id": "bob", "roles": ["analyst"]},
    "sk-ops": {"client_id": "ops", "roles": ["admin"]},
}


def build_server() -> MCPServer:
    server = MCPServer("analytics", require_auth=True)
    server.add_middleware(AuthMiddleware(APIKeyAuth(keys=KEYS)))
    queue = MCPQueue(server, max_workers=2)

    @queue.job(name="generate_report", timeout=60)
    async def generate_report(
        region: str,
        quarter: str,
        progress: ProgressReporter,
        cancel: CancellationToken,
    ) -> dict:
        """Build a quarterly sales report for a region."""
        steps = ["Loading orders", "Aggregating", "Rendering charts", "Writing PDF"]
        for i, step in enumerate(steps):
            cancel.check()
            await progress.report(i, total=len(steps), message=step)
            await asyncio.sleep(0.4)
        return {"region": region, "quarter": quarter, "url": f"/reports/{region}-{quarter}.pdf"}

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


async def call(client: MCPClient, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    result = await client.call_tool(tool, args)
    return json.loads(result.content[0].text)


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
        async with (
            MCPClient(url=URL, api_key="sk-alice") as alice,
            MCPClient(url=URL, api_key="sk-bob") as bob,
            MCPClient(url=URL, api_key="sk-ops") as ops,
        ):
            print("--- alice: what can I submit?")
            submit_tool = next(t for t in await alice.list_tools() if t.name == "queue_submit")
            print(submit_tool.description.split("\n\n", 1)[1])

            print("\n--- alice: a job with a missing argument is rejected on submit")
            bad = {"job_type": "generate_report", "args": {"region": "EMEA"}}
            print((await call(alice, "queue_submit", bad))["error"]["message"])

            print("\n--- alice: submit and poll")
            args = {"region": "EMEA", "quarter": "Q3"}
            job = await call(alice, "queue_submit", {"job_type": "generate_report", "args": args})
            while True:
                status = await call(alice, "queue_status", {"job_id": job["job_id"]})
                print(
                    f"{status['status']:>9}  {status['progress']:.2f}  {status.get('progress_message', '')}"
                )
                if status["status"] not in ("pending", "running"):
                    break
                await asyncio.sleep(0.3)
            print(
                "result:", (await call(alice, "queue_result", {"job_id": job["job_id"]}))["result"]
            )

            print("\n--- bob: alice's job is not his")
            print((await call(bob, "queue_status", {"job_id": job["job_id"]}))["error"]["code"])

            print("\n--- alice: start another job, then cancel it")
            args = {"region": "APAC", "quarter": "Q3"}
            second = await call(
                alice, "queue_submit", {"job_type": "generate_report", "args": args}
            )
            await asyncio.sleep(0.5)
            print(await call(alice, "queue_cancel", {"job_id": second["job_id"]}))

            print("\n--- ops (admin): every job")
            for j in (await call(ops, "queue_list", {}))["jobs"]:
                print(f"{j['job_id']}  {j['status']}")
    finally:
        server.terminate()
        await server.wait()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        build_server().run(transport="http", host=HOST, port=PORT)
    else:
        asyncio.run(main())
