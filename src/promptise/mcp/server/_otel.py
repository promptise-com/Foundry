"""OpenTelemetry integration middleware for MCP servers.

Creates spans for each tool call and records latency metrics.
Requires the ``opentelemetry-api`` package (``pip install "promptise[all]"``).

The middleware takes no endpoint: spans and metrics go wherever the tracer
and meter providers export them.  Pass providers explicitly, or set the
global ones once at startup.

Example::

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    from promptise.mcp.server import MCPServer, OTelMiddleware

    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True))
    )

    server = MCPServer(name="api")
    server.add_middleware(OTelMiddleware(service_name="my-mcp-server", tracer_provider=provider))
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from ._context import RequestContext


class OTelMiddleware:
    """OpenTelemetry tracing middleware.

    Creates a span for each tool call with attributes for tool name,
    request ID, client ID, and error status.  Also records a histogram
    metric for tool call duration.

    Raises ``ImportError`` if ``opentelemetry-api`` is not installed
    (``pip install "promptise[all]"``).  It takes no endpoint: configure
    the exporter on the tracer / meter provider.

    Args:
        service_name: Service name for the tracer (default
            ``"promptise-mcp-server"``).
        tracer_provider: Optional custom ``TracerProvider``. If not
            given, uses the global provider.
        meter_provider: Optional custom ``MeterProvider``. If not
            given, uses the global provider.
    """

    def __init__(
        self,
        service_name: str = "promptise-mcp-server",
        *,
        tracer_provider: Any = None,
        meter_provider: Any = None,
    ) -> None:
        self._tracer: Any = None
        self._histogram: Any = None
        self._error_counter: Any = None
        self._enabled = False

        try:
            from opentelemetry import metrics, trace

            if tracer_provider is not None:
                self._tracer = trace.get_tracer(service_name, tracer_provider=tracer_provider)
            else:
                self._tracer = trace.get_tracer(service_name)

            if meter_provider is not None:
                meter = metrics.get_meter(service_name, meter_provider=meter_provider)
            else:
                meter = metrics.get_meter(service_name)

            self._histogram = meter.create_histogram(
                name="mcp.tool.duration",
                description="Tool call duration in milliseconds",
                unit="ms",
            )
            self._error_counter = meter.create_counter(
                name="mcp.tool.errors",
                description="Tool call error count",
            )
            self._enabled = True
        except ImportError as exc:
            raise ImportError(
                "OTelMiddleware requires OpenTelemetry. Install it with: "
                'pip install "promptise[all]" (or just the packages: '
                "pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp)"
            ) from exc

    async def __call__(self, ctx: RequestContext, call_next: Callable[..., Any]) -> Any:
        if not self._enabled:
            return await call_next(ctx)

        from opentelemetry import trace

        # The exception is recorded below with redacted text; the context
        # manager's own recording would export it as written.
        with self._tracer.start_as_current_span(
            f"mcp.tool.{ctx.tool_name}",
            kind=trace.SpanKind.SERVER,
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            span.set_attribute("mcp.tool.name", ctx.tool_name)
            span.set_attribute("mcp.request.id", ctx.request_id)
            if ctx.client_id:
                span.set_attribute("mcp.client.id", ctx.client_id)

            start = time.perf_counter()
            try:
                result = await call_next(ctx)
                span.set_attribute("mcp.status", "ok")
                return result
            except Exception as exc:
                message, stacktrace = _redacted_error(exc)
                span.set_attribute("mcp.status", "error")
                span.set_attribute("mcp.error.message", message)
                # record_exception() would export str(exc) and the traceback
                # as written; pass the redacted copies instead.
                span.record_exception(
                    exc,
                    attributes={
                        "exception.message": message,
                        "exception.stacktrace": stacktrace,
                    },
                )
                span.set_status(trace.Status(trace.StatusCode.ERROR, message))
                if self._error_counter:
                    self._error_counter.add(1, {"tool": ctx.tool_name})
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start) * 1000
                if self._histogram:
                    self._histogram.record(
                        elapsed_ms,
                        {"tool": ctx.tool_name},
                    )


def _redacted_error(exc: BaseException) -> tuple[str, str]:
    """The exception's message and traceback with credentials and PII masked.

    Error text often quotes the arguments a tool was called with, so it goes
    through the same redaction as the agent's observability before it leaves
    the process in a span.
    """
    import traceback

    from ...observability import redact_sensitive

    text = redact_sensitive(
        {
            "message": str(exc),
            "stacktrace": "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
        }
    )
    return str(text["message"]), str(text["stacktrace"])
