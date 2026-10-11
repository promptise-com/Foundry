"""Prometheus metrics middleware for MCP servers.

Exposes standard Prometheus metrics for tool calls.  Requires the
``prometheus-client`` package (``pip install "promptise[all]"``).

Example::

    from promptise.mcp.server import MCPServer, PrometheusMiddleware

    server = MCPServer(name="api")
    prom = PrometheusMiddleware()
    server.add_middleware(prom)

    # The server adds no /metrics route: serve the text yourself, e.g. as a
    # resource, or call prometheus_client.start_http_server(9090).
    @server.resource("metrics://prometheus", mime_type="text/plain")
    async def prometheus_metrics() -> str:
        return prom.get_metrics_text()
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ._context import RequestContext


class PrometheusMiddleware:
    """Prometheus metrics middleware for MCP tool calls.

    Records:
    - ``mcp_tool_calls_total`` — Counter of tool calls (labels: tool, status)
    - ``mcp_tool_duration_seconds`` — Histogram of call duration (labels: tool)
    - ``mcp_tool_in_flight`` — Gauge of in-flight calls (labels: tool)

    Raises ``ImportError`` if ``prometheus-client`` is not installed
    (``pip install "promptise[all]"``).

    Args:
        namespace: Metric namespace prefix (default ``"mcp"``).
        registry: Custom Prometheus ``CollectorRegistry``. Uses the
            default global registry if not provided.
    """

    def __init__(
        self,
        namespace: str = "mcp",
        *,
        registry: Any = None,
    ) -> None:
        self._enabled = False
        self._calls_counter: Any = None
        self._duration_histogram: Any = None
        self._in_flight_gauge: Any = None
        self._registry = registry

        try:
            from prometheus_client import REGISTRY, Counter, Gauge, Histogram

            reg = registry or REGISTRY

            self._calls_counter = Counter(
                f"{namespace}_tool_calls_total",
                "Total MCP tool calls",
                ["tool", "status"],
                registry=reg,
            )
            self._duration_histogram = Histogram(
                f"{namespace}_tool_duration_seconds",
                "MCP tool call duration in seconds",
                ["tool"],
                registry=reg,
            )
            self._in_flight_gauge = Gauge(
                f"{namespace}_tool_in_flight",
                "MCP tool calls currently in flight",
                ["tool"],
                registry=reg,
            )
            self._enabled = True
        except ImportError as exc:
            raise ImportError(
                "PrometheusMiddleware requires prometheus-client. Install it with: "
                'pip install "promptise[all]" (or just the package: pip install prometheus-client)'
            ) from exc

    async def __call__(self, ctx: RequestContext, call_next: Callable[..., Any]) -> Any:
        if not self._enabled:
            return await call_next(ctx)

        tool = ctx.tool_name
        self._in_flight_gauge.labels(tool=tool).inc()
        start = time.perf_counter()
        try:
            result = await call_next(ctx)
            self._calls_counter.labels(tool=tool, status="ok").inc()
            return result
        except Exception:
            self._calls_counter.labels(tool=tool, status="error").inc()
            raise
        finally:
            elapsed = time.perf_counter() - start
            self._duration_histogram.labels(tool=tool).observe(elapsed)
            self._in_flight_gauge.labels(tool=tool).dec()

    def get_metrics_text(self) -> str:
        """Generate Prometheus text exposition format.

        Returns:
            Metrics in text/plain format for scraping.
        """
        try:
            from prometheus_client import generate_latest

            reg = self._registry
            if reg is None:
                from prometheus_client import REGISTRY

                reg = REGISTRY
            return generate_latest(reg).decode("utf-8")
        except ImportError:
            raise ImportError(
                "prometheus-client is required to scrape metrics. "
                "Install with: pip install prometheus-client"
            )
