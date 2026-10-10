"""Hot reload support for MCP server development.

Watches Python files for changes and restarts the server process.
Only for development — not for production use.

Example::

    from promptise.mcp.server import MCPServer, hot_reload

    server = MCPServer(name="dev-server")

    @server.tool()
    async def greet(name: str) -> str:
        return f"Hello {name}"

    if __name__ == "__main__":
        hot_reload(server, transport="http", port=8080)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("promptise.server.reload")


def hot_reload(
    server: Any = None,
    *,
    transport: str = "http",
    host: str = "127.0.0.1",
    port: int = 8080,
    watch_dirs: list[str] | None = None,
    poll_interval: float = 1.0,
    dashboard: bool = False,
    allowed_hosts: list[str] | None = None,
    allowed_origins: list[str] | None = None,
) -> None:
    """Run an MCP server with automatic restart on file changes.

    Watches ``*.py`` files in the specified directories (defaults to
    the current working directory).  When a change is detected, the
    server process is restarted.

    This function blocks and runs in the parent process while the
    server runs as a subprocess.

    Args:
        server: Not used directly — the server is started via
            re-running the current script.
        transport: Transport type.
        host: Bind host.
        port: Bind port.
        watch_dirs: Directories to watch (defaults to ``["."]``).
        poll_interval: Seconds between file change checks.
        dashboard: Enable dashboard in the child process.
        allowed_hosts: ``Host`` header allow-list for the child server's
            HTTP/SSE transport (see ``MCPServer.run``).
        allowed_origins: ``Origin`` header allow-list for the child
            server's HTTP/SSE transport (see ``MCPServer.run``).

    Example::

        if __name__ == "__main__":
            hot_reload(transport="http", port=8080)
    """
    # If we're already in the child process, just run the server
    if os.environ.get("_PROMPTISE_RELOAD_CHILD"):
        if server is not None:
            server.run(
                transport=transport,
                host=host,
                port=port,
                dashboard=dashboard,
                allowed_hosts=allowed_hosts,
                allowed_origins=allowed_origins,
            )
        return

    dirs = watch_dirs or ["."]
    script = sys.argv[0]

    logger.info("Hot reload: watching %s for changes", dirs)

    while True:
        # Start the child process
        env = {**os.environ, "_PROMPTISE_RELOAD_CHILD": "1"}
        cmd = [sys.executable, script] + sys.argv[1:]

        print("\n--- Starting server (pid will follow) ---")
        print(f"    Watching: {', '.join(dirs)}")
        print(f"    Poll interval: {poll_interval}s\n")

        proc = subprocess.Popen(cmd, env=env)

        # Take initial snapshot
        snapshot = _file_snapshot(dirs)

        try:
            while proc.poll() is None:
                time.sleep(poll_interval)
                new_snapshot = _file_snapshot(dirs)

                changed = _detect_changes(snapshot, new_snapshot)
                if changed:
                    print(f"\n--- Detected changes in {len(changed)} file(s) ---")
                    for f in sorted(changed)[:5]:
                        print(f"    {f}")
                    if len(changed) > 5:
                        print(f"    ... and {len(changed) - 5} more")
                    print("--- Restarting server ---\n")

                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                    break

                snapshot = new_snapshot

            if proc.poll() is not None and proc.returncode != 0:
                print(f"\n--- Server exited with code {proc.returncode} ---")
                print("--- Waiting for file changes to restart ---\n")
                # Wait for changes before restarting
                while True:
                    time.sleep(poll_interval)
                    new_snapshot = _file_snapshot(dirs)
                    if _detect_changes(snapshot, new_snapshot):
                        break
                    snapshot = new_snapshot

        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            print("\n--- Hot reload stopped ---")
            return


def _file_snapshot(dirs: list[str]) -> dict[str, float]:
    """Take a snapshot of all .py file modification times."""
    snapshot: dict[str, float] = {}
    for d in dirs:
        for path in Path(d).rglob("*.py"):
            try:
                snapshot[str(path)] = path.stat().st_mtime
            except (OSError, PermissionError):
                pass
    return snapshot


def _detect_changes(old: dict[str, float], new: dict[str, float]) -> set[str]:
    """Return set of files that changed between snapshots."""
    changed: set[str] = set()
    for path, mtime in new.items():
        if path not in old or old[path] != mtime:
            changed.add(path)
    # Deleted files
    for path in old:
        if path not in new:
            changed.add(path)
    return changed
