"""Health and readiness probes for MCP servers.

Automatically registers ``health://liveness`` and ``health://readiness``
as MCP resources when attached to a server.  Over the HTTP and SSE
transports the same probes are also served as plain HTTP routes
(``GET /health`` and ``GET /health/ready``) for container and Kubernetes
probes, which cannot speak MCP.

Example::

    from promptise.mcp.server import MCPServer, HealthCheck

    server = MCPServer(name="api")
    health = HealthCheck()
    health.add_check("database", check_db, required_for_ready=True)
    health.register_resources(server)
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from typing import Any


class HealthCheck:
    """Health and readiness probe manager.

    Tracks named health checks and exposes liveness/readiness as
    JSON resources.
    """

    def __init__(self) -> None:
        self._checks: list[tuple[str, Callable[..., Any], bool]] = []
        self._started_at: float = time.time()

    def add_check(
        self,
        name: str,
        check: Callable[..., Any],
        *,
        required_for_ready: bool = True,
    ) -> None:
        """Register a health check.

        Args:
            name: Check name.
            check: Callable returning ``True`` (healthy) or ``False``.
                Can be sync or async.
            required_for_ready: If ``True``, failing this check makes
                readiness return unhealthy.
        """
        self._checks.append((name, check, required_for_ready))

    async def liveness(self) -> str:
        """Return liveness probe result as JSON string."""
        return json.dumps(self.liveness_report())

    def liveness_report(self) -> dict[str, Any]:
        """Liveness probe result as a dict (``status`` is always ``"alive"``)."""
        return {
            "status": "alive",
            "uptime_seconds": round(time.time() - self._started_at, 1),
        }

    async def readiness(self) -> str:
        """Return readiness probe result as JSON string."""
        return json.dumps(await self.readiness_report())

    async def readiness_report(self, *, include_errors: bool = True) -> dict[str, Any]:
        """Run every check and return the readiness result as a dict.

        Args:
            include_errors: Include the message of a check that raised.
                The unauthenticated HTTP probe route passes ``False`` so
                exception text never leaves the process.

        Returns:
            ``{"status": "ready" | "not_ready", "checks": {name: {"healthy": bool, ...}}}``
        """
        results: dict[str, Any] = {}
        all_ready = True

        for name, check, required in self._checks:
            try:
                result = check()
                if asyncio.iscoroutine(result):
                    result = await result
                healthy = bool(result)
            except Exception as exc:
                healthy = False
                results[name] = {"healthy": False}
                if include_errors:
                    results[name]["error"] = str(exc)
                if required:
                    all_ready = False
                continue

            results[name] = {"healthy": healthy}
            if required and not healthy:
                all_ready = False

        return {
            "status": "ready" if all_ready else "not_ready",
            "checks": results,
        }

    def register_resources(self, server: Any) -> None:
        """Register health resources on an ``MCPServer``.

        Also makes this check set back the server's ``GET /health`` and
        ``GET /health/ready`` HTTP routes.

        Args:
            server: The ``MCPServer`` instance.
        """
        health = self
        server._health_check = self

        @server.resource(
            "health://liveness",
            name="liveness",
            description="Liveness probe",
            mime_type="application/json",
        )
        async def _liveness() -> str:
            return await health.liveness()

        @server.resource(
            "health://readiness",
            name="readiness",
            description="Readiness probe",
            mime_type="application/json",
        )
        async def _readiness() -> str:
            return await health.readiness()
