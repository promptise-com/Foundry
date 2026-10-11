"""Tests for promptise.server observability (metrics)."""

from __future__ import annotations

import json

import pytest

from promptise.mcp.server import MCPServer
from promptise.mcp.server._context import RequestContext
from promptise.mcp.server._observability import MetricsCollector, MetricsMiddleware

# =====================================================================
# MetricsCollector
# =====================================================================


class TestMetricsCollector:
    def test_record_call(self):
        collector = MetricsCollector()
        collector.record_call("search", 0.1)
        collector.record_call("search", 0.2)
        collector.record_call("query", 0.05, error=True)

        snap = collector.snapshot()
        assert snap["tools"]["search"]["calls"] == 2
        assert snap["tools"]["search"]["errors"] == 0
        assert snap["tools"]["search"]["avg_latency_ms"] == pytest.approx(150.0, abs=1)
        assert snap["tools"]["query"]["calls"] == 1
        assert snap["tools"]["query"]["errors"] == 1

    def test_snapshot_empty(self):
        collector = MetricsCollector()
        snap = collector.snapshot()
        assert snap["tools"] == {}
        assert "uptime_seconds" in snap

    def test_register_resource(self):
        server = MCPServer(name="test")
        collector = MetricsCollector()
        collector.register_resource(server)

        rdef = server._resource_registry.get("metrics://server")
        assert rdef is not None
        assert rdef.mime_type == "application/json"

    async def test_metrics_resource_returns_snapshot(self):
        server = MCPServer(name="test")
        collector = MetricsCollector()
        collector.record_call("add", 0.01)
        collector.register_resource(server)

        rdef = server._resource_registry.get("metrics://server")
        result = json.loads(await rdef.handler())
        assert result["tools"]["add"]["calls"] == 1


# =====================================================================
# MetricsMiddleware
# =====================================================================


class TestMetricsMiddleware:
    async def test_records_successful_call(self):
        collector = MetricsCollector()
        mw = MetricsMiddleware(collector)

        async def call_next(ctx):
            return "ok"

        ctx = RequestContext(server_name="test", tool_name="search")
        result = await mw(ctx, call_next)

        assert result == "ok"
        snap = collector.snapshot()
        assert snap["tools"]["search"]["calls"] == 1
        assert snap["tools"]["search"]["errors"] == 0

    async def test_records_failed_call(self):
        collector = MetricsCollector()
        mw = MetricsMiddleware(collector)

        async def call_next(ctx):
            raise RuntimeError("boom")

        ctx = RequestContext(server_name="test", tool_name="search")
        with pytest.raises(RuntimeError):
            await mw(ctx, call_next)

        snap = collector.snapshot()
        assert snap["tools"]["search"]["calls"] == 1
        assert snap["tools"]["search"]["errors"] == 1

    async def test_creates_default_collector(self):
        mw = MetricsMiddleware()
        assert mw.collector is not None

        async def call_next(ctx):
            return "ok"

        ctx = RequestContext(server_name="test", tool_name="add")
        await mw(ctx, call_next)
        assert mw.collector.snapshot()["tools"]["add"]["calls"] == 1


# =====================================================================
# OTelMiddleware / PrometheusMiddleware: documented usage and errors
# =====================================================================

_OBSERVABILITY_DOC = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "docs"
    / "mcp"
    / "server"
    / "observability.md"
)


def _doc_section(title: str, next_title: str) -> str:
    text = _OBSERVABILITY_DOC.read_text(encoding="utf-8")
    return text[text.index(f"## {title}\n") : text.index(f"## {next_title}\n")]


def _python_blocks(section: str) -> list[str]:
    import re

    return re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL)


class TestOTelMiddlewareDocs:
    def test_documented_examples_run(self, monkeypatch):
        pytest.importorskip("opentelemetry.sdk")
        pytest.importorskip("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")
        from opentelemetry import trace

        from promptise.mcp.server import OTelMiddleware

        # Don't replace this process's global provider.
        monkeypatch.setattr(trace, "set_tracer_provider", lambda provider: None)
        blocks = [
            b
            for b in _python_blocks(_doc_section("OpenTelemetry", "Prometheus Metrics"))
            if "OTelMiddleware(" in b
        ]
        assert len(blocks) == 2
        for block in blocks:
            namespace: dict = {}
            exec(compile(block, str(_OBSERVABILITY_DOC), "exec"), namespace)
            middleware = [
                m for m in namespace["server"]._middlewares if isinstance(m, OTelMiddleware)
            ]
            assert len(middleware) == 1
            namespace["provider"].shutdown()

    def test_endpoint_is_not_a_parameter(self):
        pytest.importorskip("opentelemetry")
        from promptise.mcp.server import OTelMiddleware

        with pytest.raises(TypeError):
            OTelMiddleware(endpoint="http://jaeger:4317")  # type: ignore[call-arg]
        api_row = next(
            line
            for line in _OBSERVABILITY_DOC.read_text(encoding="utf-8").splitlines()
            if line.startswith("| `OTelMiddleware(")
        )
        assert "endpoint" not in api_row

    def test_docs_say_missing_packages_raise(self):
        otel = _doc_section("OpenTelemetry", "Prometheus Metrics")
        prom = _doc_section("Prometheus Metrics", "Structured Logging")
        for section in (otel, prom):
            assert "No-op when not installed" not in section
            assert "raises `ImportError`" in section
            assert 'pip install "promptise[all]"' in section


class TestMissingPackageErrors:
    def test_otel_names_the_extra(self, monkeypatch):
        import sys

        from promptise.mcp.server import OTelMiddleware

        monkeypatch.setitem(sys.modules, "opentelemetry", None)
        with pytest.raises(ImportError, match=r'pip install "promptise\[all\]"'):
            OTelMiddleware()

    def test_prometheus_names_the_extra(self, monkeypatch):
        import sys

        from promptise.mcp.server import PrometheusMiddleware

        monkeypatch.setitem(sys.modules, "prometheus_client", None)
        with pytest.raises(ImportError, match=r'pip install "promptise\[all\]"'):
            PrometheusMiddleware()


class TestOTelErrorRedaction:
    """Error text exported on spans is redacted like the agent's observability."""

    @pytest.mark.asyncio
    async def test_span_error_text_masks_credentials_and_pii(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        from promptise.mcp.server import MCPServer, TestClient
        from promptise.mcp.server._otel import OTelMiddleware

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        server = MCPServer(name="otel-redact")
        server.add_middleware(OTelMiddleware(tracer_provider=provider))

        @server.tool()
        async def charge(email: str) -> str:
            """Charge a customer."""
            raise RuntimeError(
                f"upstream refused {email} with key sk-proj-Ab3_dEf-GhIjKlMnOpQrStUvWxYz"
            )

        client = TestClient(server)
        await client.call_tool("charge", {"email": "alice@example.com"})

        [span] = exporter.get_finished_spans()
        exported = [str(span.attributes.get("mcp.error.message")), str(span.status.description)]
        for event in span.events:
            exported += [str(v) for v in (event.attributes or {}).values()]
        text = "\n".join(exported)
        assert "upstream refused" in text
        assert "alice@example.com" not in text
        assert "sk-proj-Ab3_dEf-GhIjKlMnOpQrStUvWxYz" not in text


@pytest.mark.parametrize(
    "key",
    [
        "sk-abcdefghijklmnopqrstuvwxyz0123",
        "sk-proj-Ab3_dEf-GhIjKlMnOpQrStUvWxYz0123456789",
        "sk-ant-api03-Ab3_dEf-GhIjKlMnOpQrStUvWxYz0123456789",
        "sk-svcacct-Ab3dEfGhIjKlMnOpQrStUvWxYz",
    ],
)
def test_redaction_masks_current_api_key_formats(key):
    from promptise.observability import redact_sensitive

    out = redact_sensitive({"error": f"auth failed for {key} (retry later)"})
    assert key not in out["error"]
    assert "[API_KEY]" in out["error"] and out["error"].endswith("(retry later)")
