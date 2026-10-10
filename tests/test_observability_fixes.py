"""Regression tests for the observability fixes.

* Semantic cache + ``observe=True``: cache events were recorded with an
  unknown ``description=`` keyword, the ``TypeError`` was swallowed as "Cache
  check failed", and the cache never hit.
* HTML report: stat cards and filters used event names that never occur
  (``llm_end``, ``tool_call_start``); ``generate_report(path, title)`` wrote a
  different file and ignored ``title``.
* ``ObserveLevel``: OFF recorded everything, BASIC only dropped ``llm.start``,
  FULL never added prompt/response text.
* Agent input/output were never recorded; failed MCP tool calls were
  recorded as ``tool.result``; tool arguments/results could not be turned off.
* OpenTelemetry: every event was its own zero-length root span with no user,
  session or correlation id.
* The NDJSON event stream grew forever.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from promptise.agent import CallerContext, PromptiseAgent, build_agent
from promptise.callback_handler import PromptiseCallbackHandler
from promptise.observability import (
    AgentRun,
    ObservabilityCollector,
    TimelineEventType,
    _run_ctx_var,
    redact_sensitive,
)
from promptise.observability_config import ObservabilityConfig, ObserveLevel, TransporterType
from promptise.observability_transporters import HTMLReportTransporter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_result(text: str = "answer", prompt_tokens: int = 12, completion_tokens: int = 5):
    msg = AIMessage(
        content=text,
        usage_metadata={
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )
    return LLMResult(
        generations=[[ChatGeneration(message=msg, text=text)]],
        llm_output={"model_name": "gpt-5-mini"},
    )


class _SimulatedGraph:
    """Fake agent graph that fires the LangChain callbacks a real run fires.

    One LLM turn, one tool call (optionally failing), and a final answer.
    """

    def __init__(self, *, tool_fails: bool = False, llm_delay: float = 0.0) -> None:
        self.tool_fails = tool_fails
        self.llm_delay = llm_delay
        self.calls = 0

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self.calls += 1
        handlers = [h for h in (config or {}).get("callbacks", []) if hasattr(h, "on_llm_start")]
        llm_run, tool_run = uuid4(), uuid4()
        for h in handlers:
            h.on_llm_start(
                {"kwargs": {"model_name": "gpt-5-mini"}},
                ["System: be brief\nHuman: hi"],
                run_id=llm_run,
            )
        if self.llm_delay:
            await asyncio.sleep(self.llm_delay)
        for h in handlers:
            h.on_llm_end(_llm_result(), run_id=llm_run)
            h.on_tool_start({"name": "check_stock"}, "{'sku': 'MS-30'}", run_id=tool_run)
        await asyncio.sleep(0.02)
        for h in handlers:
            if self.tool_fails:
                h.on_tool_error(RuntimeError("Unknown SKU MS-30."), run_id=tool_run)
            else:
                h.on_tool_end('{"units": 6}', run_id=tool_run, name="check_stock")
        return {"messages": [*input.get("messages", []), AIMessage(content="6 in stock")]}


def _observed_agent(
    level: ObserveLevel = ObserveLevel.STANDARD,
    record_prompts: bool | None = None,
    record_tool_io: bool = True,
    inner: Any = None,
    **kwargs: Any,
) -> PromptiseAgent:
    collector = ObservabilityCollector("test")
    # record_tool_io is passed only when switched off, so the tests that do
    # not use it also run (and fail for the right reason) against 1.2.1.
    handler_kwargs: dict[str, Any] = {} if record_tool_io else {"record_tool_io": False}
    handler = PromptiseCallbackHandler(
        collector,
        agent_id="shop",
        level=level,
        record_prompts=record_prompts,
        **handler_kwargs,
    )
    return PromptiseAgent(
        inner=inner or _SimulatedGraph(),
        handler=handler,
        collector=collector,
        observe_config=ObservabilityConfig(level=level),
        model_name="openai:gpt-5-mini",
        **kwargs,
    )


def _types(agent: PromptiseAgent) -> list[str]:
    assert agent.collector is not None
    return [e.event_type.value for e in agent.collector.get_timeline()]


USER_MSG = {"messages": [{"role": "user", "content": "Is the MS-30 in stock?"}]}


# ---------------------------------------------------------------------------
# 1. Semantic cache with observability
# ---------------------------------------------------------------------------


class _Embed:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, float(len(t) % 7)] for t in texts]


class TestCacheWithObservability:
    @pytest.mark.asyncio
    async def test_cache_hits_with_observe(self, caplog) -> None:
        from promptise.cache import SemanticCache

        inner = _SimulatedGraph()
        agent = _observed_agent(inner=inner, cache=SemanticCache(embedding=_Embed()))
        caller = CallerContext(user_id="u1")
        with caplog.at_level(logging.WARNING):
            first = await agent.ainvoke(USER_MSG, caller=caller)
            second = await agent.ainvoke(USER_MSG, caller=caller)
        assert "Cache check failed" not in caplog.text
        assert "Cache store failed" not in caplog.text
        assert inner.calls == 1
        assert second["messages"][-1].content == first["messages"][-1].content

        types = _types(agent)
        assert types.count("cache.miss") == 1
        assert types.count("cache.store") == 1
        assert types.count("cache.hit") == 1
        hit = agent.collector.query(event_types=["cache.hit"])[0]
        assert hit.details and "Cache hit" in hit.details
        # The user's text is not copied into event details.
        assert "MS-30" not in (hit.details or "")
        outputs = agent.collector.query(event_types=["agent.output"])
        assert [o.metadata["cache_hit"] for o in outputs] == [False, True]


# ---------------------------------------------------------------------------
# 2. HTML report
# ---------------------------------------------------------------------------


class TestHTMLReport:
    @pytest.mark.asyncio
    async def test_generate_report_writes_exactly_path_with_title(self, tmp_path) -> None:
        agent = _observed_agent()
        await agent.ainvoke(USER_MSG)
        target = tmp_path / "out" / "shop-report.html"
        returned = agent.generate_report(target, title="Shop <assistant> trace")
        assert returned == str(target)
        assert sorted(p.name for p in target.parent.iterdir()) == ["shop-report.html"]
        page = target.read_text()
        assert "<title>Shop &lt;assistant&gt; trace</title>" in page
        assert '"Shop \\u003cassistant> trace"' in page

    @pytest.mark.asyncio
    async def test_report_embeds_collector_stats(self, tmp_path) -> None:
        agent = _observed_agent()
        await agent.ainvoke(USER_MSG)
        path = agent.generate_report(tmp_path / "r.html")
        page = Path(path).read_text()
        data = json.loads(re.search(r"const data = (.*?);\nconst entries", page, re.S).group(1))
        stats = agent.get_stats()
        assert data["stats"]["total_tokens"] == stats["total_tokens"] == 17
        assert data["stats"]["llm_call_count"] == 1
        assert data["stats"]["tool_call_count"] == 1
        # The page reads its numbers from the embedded stats and categorises
        # events by their real (dotted) names.
        assert "stats.total_tokens" in page
        assert "stats.llm_call_count" in page
        assert "'llm.end'" in page and "'tool.call'" in page
        assert "llm_end" not in page and "tool_call_start" not in page

    def test_event_text_cannot_break_out_of_the_page(self, tmp_path) -> None:
        collector = ObservabilityCollector("xss")
        collector.record(
            TimelineEventType.TOOL_RESULT,
            details="</script><script>alert(1)</script>",
            metadata={"result_preview": "<img src=x onerror=alert(1)> & more"},
        )
        t = HTMLReportTransporter(title="t")
        t._collector = collector
        page = t.write(tmp_path / "x.html").read_text()
        script = page[page.index("<script>") : page.rindex("</script>")]
        assert "</script" not in script[len("<script>") :]
        assert "<img" not in page
        assert "\\u003c/script\\u003e" in page
        # Event text is rendered with textContent, never innerHTML.
        assert "innerHTML" not in page

    def test_flush_writes_timestamped_report(self, tmp_path) -> None:
        collector = ObservabilityCollector("sess")
        collector.record(TimelineEventType.TOOL_CALL, details="x")
        t = HTMLReportTransporter(output_dir=str(tmp_path), session_name="sess")
        t._collector = collector
        t.flush()
        [written] = list(tmp_path.iterdir())
        assert re.fullmatch(r"sess-report-\d{8}_\d{6}\.html", written.name)

    def test_generate_report_requires_observability(self) -> None:
        agent = PromptiseAgent(inner=_SimulatedGraph())
        with pytest.raises(RuntimeError, match="observability is not enabled"):
            agent.generate_report("r.html")


# ---------------------------------------------------------------------------
# 3. ObserveLevel semantics
# ---------------------------------------------------------------------------

CONTENT_KEYS = {"prompt_preview", "response_preview", "input_preview", "output_preview"}


def _metadata_keys(agent: PromptiseAgent) -> set[str]:
    return {k for e in agent.collector.get_timeline() for k in e.metadata}


class TestObserveLevels:
    @pytest.mark.asyncio
    async def test_off_records_nothing(self) -> None:
        agent = _observed_agent(ObserveLevel.OFF)
        await agent.ainvoke(USER_MSG)
        assert _types(agent) == []
        assert agent.get_stats()["entry_count"] == 0

    @pytest.mark.asyncio
    async def test_basic_records_io_tools_errors_but_no_llm_turns(self) -> None:
        agent = _observed_agent(ObserveLevel.BASIC, inner=_SimulatedGraph(tool_fails=True))
        await agent.ainvoke(USER_MSG)
        assert _types(agent) == ["agent.input", "tool.call", "tool.error", "agent.output"]
        out = agent.collector.query(event_types=["agent.output"])[0]
        assert (out.metadata["total_tokens"], out.metadata["llm_call_count"]) == (17, 1)
        stats = agent.get_stats()
        # BASIC has no llm.end events; totals come from agent.output.
        assert (stats["total_tokens"], stats["llm_call_count"], stats["error_count"]) == (17, 1, 1)
        assert not CONTENT_KEYS & _metadata_keys(agent)

    @pytest.mark.asyncio
    async def test_standard_adds_llm_turns_without_content(self) -> None:
        agent = _observed_agent(ObserveLevel.STANDARD)
        await agent.ainvoke(USER_MSG)
        assert _types(agent) == [
            "agent.input",
            "llm.start",
            "llm.end",
            "tool.call",
            "tool.result",
            "agent.output",
        ]
        assert agent.get_stats()["total_tokens"] == 17  # not double-counted
        assert not CONTENT_KEYS & _metadata_keys(agent)

    @pytest.mark.asyncio
    async def test_full_adds_content(self) -> None:
        agent = _observed_agent(ObserveLevel.FULL)
        await agent.ainvoke(USER_MSG)
        assert _metadata_keys(agent) >= CONTENT_KEYS
        inp = agent.collector.query(event_types=["agent.input"])[0]
        assert inp.metadata["input_preview"] == "Is the MS-30 in stock?"
        out = agent.collector.query(event_types=["agent.output"])[0]
        assert out.metadata["output_preview"] == "6 in stock"

    @pytest.mark.asyncio
    async def test_full_with_record_prompts_false_has_no_content(self) -> None:
        agent = _observed_agent(ObserveLevel.FULL, record_prompts=False)
        await agent.ainvoke(USER_MSG)
        assert not CONTENT_KEYS & _metadata_keys(agent)

    @pytest.mark.asyncio
    async def test_record_prompts_true_adds_content_below_full(self) -> None:
        agent = _observed_agent(ObserveLevel.STANDARD, record_prompts=True)
        await agent.ainvoke(USER_MSG)
        assert _metadata_keys(agent) >= CONTENT_KEYS

    def test_record_prompts_resolution(self) -> None:
        c = ObservabilityCollector()
        assert PromptiseCallbackHandler(c, level=ObserveLevel.FULL).record_prompts is True
        assert PromptiseCallbackHandler(c, level=ObserveLevel.STANDARD).record_prompts is False
        assert ObservabilityConfig().record_prompts is None

    @pytest.mark.asyncio
    async def test_build_agent_level_off(self, tmp_path) -> None:
        with (
            patch("promptise.agent._normalize_model", return_value=MagicMock()),
            patch("promptise.agent.PromptGraphEngine", return_value=_SimulatedGraph()),
        ):
            agent = await build_agent(
                servers={},
                model="openai:gpt-5-mini",
                observe=ObservabilityConfig(level=ObserveLevel.OFF, output_dir=str(tmp_path)),
            )
        assert agent.collector is not None
        assert agent._transporters == []
        await agent.ainvoke(USER_MSG)
        await agent.shutdown()
        assert _types(agent) == []
        assert list(tmp_path.iterdir()) == []  # no empty report written


# ---------------------------------------------------------------------------
# 4. Agent runs, tool errors, tool I/O switch
# ---------------------------------------------------------------------------


class TestAgentRuns:
    @pytest.mark.asyncio
    async def test_events_are_linked_to_their_run(self) -> None:
        agent = _observed_agent()
        await agent.ainvoke(
            USER_MSG, caller=CallerContext(user_id="u1", metadata={"session_id": "q1"})
        )
        timeline = agent.collector.get_timeline()
        run_input = timeline[0]
        assert run_input.event_type == TimelineEventType.AGENT_INPUT
        assert run_input.parent_id is None
        assert all(e.parent_id == run_input.entry_id for e in timeline[1:])
        assert {(e.user_id, e.session_id) for e in timeline} == {("u1", "q1")}
        out = timeline[-1]
        assert out.metadata["prompt_tokens"] == 12
        assert out.metadata["tool_call_count"] == 1
        assert out.duration is not None and out.duration > 0

    @pytest.mark.asyncio
    async def test_failed_run_records_agent_error(self) -> None:
        inner = MagicMock()
        inner.ainvoke = AsyncMock(side_effect=ValueError("boom"))
        agent = _observed_agent(inner=inner)
        with pytest.raises(ValueError):
            await agent.ainvoke(USER_MSG)
        assert _types(agent) == ["agent.input", "agent.error"]
        err = agent.collector.query(event_types=["agent.error"])[0]
        assert err.metadata["error_type"] == "ValueError"
        assert agent.get_stats()["error_count"] == 1

    @pytest.mark.asyncio
    async def test_run_context_is_reset(self) -> None:
        agent = _observed_agent()
        await agent.ainvoke(USER_MSG)
        assert _run_ctx_var.get() is None

    @pytest.mark.asyncio
    async def test_concurrent_runs_keep_their_own_totals(self) -> None:
        agent = _observed_agent(inner=_SimulatedGraph(llm_delay=0.01))
        await asyncio.gather(*(agent.ainvoke(USER_MSG) for _ in range(3)))
        outputs = agent.collector.query(event_types=["agent.output"])
        assert [o.metadata["total_tokens"] for o in outputs] == [17, 17, 17]
        inputs = {e.entry_id for e in agent.collector.query(event_types=["agent.input"])}
        assert {o.parent_id for o in outputs} == inputs

    @pytest.mark.asyncio
    async def test_streaming_run_is_recorded(self) -> None:
        inner = MagicMock()

        async def astream_events(*args: Any, **kwargs: Any) -> Any:
            chunk = MagicMock()
            chunk.content = "6 in stock"
            yield {"event": "on_chat_model_stream", "data": {"chunk": chunk}}

        inner.astream_events = astream_events
        agent = _observed_agent(ObserveLevel.FULL, inner=inner)
        async for _ in agent.astream_with_tools(USER_MSG):
            pass
        assert _types(agent) == ["agent.input", "agent.output"]
        out = agent.collector.query(event_types=["agent.output"])[0]
        assert out.metadata["output_preview"] == "6 in stock"


class TestToolErrors:
    def test_tool_error_is_recorded_as_error(self) -> None:
        c = ObservabilityCollector()
        h = PromptiseCallbackHandler(c, agent_id="a")
        rid = uuid4()
        h.on_tool_start({"name": "check_stock"}, "{'sku': 'X'}", run_id=rid)
        h.on_tool_error(RuntimeError("Unknown SKU X."), run_id=rid)
        [call, err] = c.get_timeline()
        assert err.event_type == TimelineEventType.TOOL_ERROR
        assert err.metadata["tool_name"] == "check_stock"
        assert err.metadata["error"] == "Unknown SKU X."
        assert c.get_stats()["error_count"] == 1

    def test_error_tool_message_is_recorded_as_error(self) -> None:
        """A tool with handle_tool_error returns ToolMessage(status="error")."""
        c = ObservabilityCollector()
        h = PromptiseCallbackHandler(c)
        rid = uuid4()
        h.on_tool_start({"name": "lookup"}, "{}", run_id=rid)
        h.on_tool_end(
            ToolMessage(content="not found", tool_call_id="t1", status="error"),
            run_id=rid,
            name="lookup",
        )
        assert [e.event_type.value for e in c.get_timeline()] == ["tool.call", "tool.error"]
        assert c.get_stats()["error_count"] == 1

    @pytest.mark.asyncio
    async def test_mcp_error_result_raises_and_records_tool_error(self) -> None:
        from mcp.types import CallToolResult, TextContent

        from promptise.mcp.client._tool_adapter import _PromptiseMCPTool

        multi = MagicMock()
        multi.call_tool = AsyncMock(
            return_value=CallToolResult(
                content=[TextContent(type="text", text='{"error": {"code": "TOOL_ERROR"}}')],
                isError=True,
            )
        )
        on_error, on_after = MagicMock(), MagicMock()
        from pydantic import BaseModel

        class Args(BaseModel):
            sku: str

        tool = _PromptiseMCPTool(
            name="check_stock",
            description="d",
            args_schema=Args,
            tool_name="check_stock",
            multi=multi,
            on_error=on_error,
            on_after=on_after,
        )
        c = ObservabilityCollector()
        h = PromptiseCallbackHandler(c)
        from langchain_core.tools import ToolException

        with pytest.raises(ToolException, match="TOOL_ERROR"):
            await tool.ainvoke({"sku": "XX-999"}, config={"callbacks": [h]})
        assert [e.event_type.value for e in c.get_timeline()] == ["tool.call", "tool.error"]
        assert c.get_stats()["error_count"] == 1
        on_error.assert_called_once()
        on_after.assert_not_called()

    @pytest.mark.asyncio
    async def test_mcp_success_result_is_returned(self) -> None:
        from mcp.types import CallToolResult, TextContent
        from pydantic import BaseModel

        from promptise.mcp.client._tool_adapter import _PromptiseMCPTool

        class Args(BaseModel):
            sku: str

        multi = MagicMock()
        multi.call_tool = AsyncMock(
            return_value=CallToolResult(content=[TextContent(type="text", text="6")], isError=False)
        )
        tool = _PromptiseMCPTool(
            name="check_stock",
            description="d",
            args_schema=Args,
            tool_name="check_stock",
            multi=multi,
        )
        assert await tool.ainvoke({"sku": "MS-30"}) == "6"


class TestServerToolErrors:
    """A Promptise MCP server answers a failed tool call with ``isError=True``."""

    @staticmethod
    async def _call(server: Any, name: str, args: dict[str, Any]) -> Any:
        import mcp.types as t

        ll = server._build_lowlevel_server()
        handler = ll.request_handlers[t.CallToolRequest]
        req = t.CallToolRequest(
            method="tools/call", params=t.CallToolRequestParams(name=name, arguments=args)
        )
        return (await handler(req)).root

    @pytest.mark.asyncio
    async def test_tool_error_sets_is_error(self) -> None:
        from promptise.mcp.server import MCPServer, ToolError

        server = MCPServer("shop")

        @server.tool()
        async def check_stock(sku: str) -> dict:
            """Check stock."""
            if sku != "MS-30":
                raise ToolError(f"Unknown SKU {sku}.")
            return {"units": 6}

        ok = await self._call(server, "check_stock", {"sku": "MS-30"})
        assert ok.isError is False
        failed = await self._call(server, "check_stock", {"sku": "XX-999"})
        assert failed.isError is True
        body = json.loads(failed.content[0].text)
        assert body["error"]["code"] == "TOOL_ERROR"
        assert body["error"]["message"] == "Unknown SKU XX-999."

    @pytest.mark.asyncio
    async def test_unexpected_exception_sets_is_error_without_leaking(self) -> None:
        from promptise.mcp.server import MCPServer

        server = MCPServer("shop")

        @server.tool()
        async def broken() -> str:
            """Always fails."""
            raise RuntimeError("postgres://secret@db")

        res = await self._call(server, "broken", {})
        assert res.isError is True
        assert "secret" not in res.content[0].text
        assert json.loads(res.content[0].text)["error"]["code"] == "INTERNAL_ERROR"

    @pytest.mark.asyncio
    async def test_unknown_tool_sets_is_error(self) -> None:
        from promptise.mcp.server import MCPServer

        res = await self._call(MCPServer("shop"), "nope", {})
        assert res.isError is True
        assert json.loads(res.content[0].text)["error"]["code"] == "TOOL_NOT_FOUND"


class TestRecordToolIO:
    @pytest.mark.asyncio
    async def test_tool_io_recorded_by_default(self) -> None:
        agent = _observed_agent()
        await agent.ainvoke(USER_MSG)
        call = agent.collector.query(event_types=["tool.call"])[0]
        result = agent.collector.query(event_types=["tool.result"])[0]
        assert call.metadata["arguments"] == "{'sku': 'MS-30'}"
        assert result.metadata["result_preview"] == '{"units": 6}'

    @pytest.mark.asyncio
    async def test_tool_io_off_keeps_only_lengths(self) -> None:
        agent = _observed_agent(record_tool_io=False)
        await agent.ainvoke(USER_MSG)
        call = agent.collector.query(event_types=["tool.call"])[0]
        result = agent.collector.query(event_types=["tool.result"])[0]
        assert "arguments" not in call.metadata
        assert call.metadata["arguments_length"] == len("{'sku': 'MS-30'}")
        assert "result_preview" not in result.metadata
        assert result.metadata["result_length"] == len('{"units": 6}')
        assert result.metadata["tool_name"] == "check_stock"

    @pytest.mark.asyncio
    async def test_tool_io_off_drops_error_text(self) -> None:
        agent = _observed_agent(record_tool_io=False, inner=_SimulatedGraph(tool_fails=True))
        await agent.ainvoke(USER_MSG)
        err = agent.collector.query(event_types=["tool.error"])[0]
        assert "error" not in err.metadata and "traceback" not in err.metadata
        assert err.metadata["error_type"] == "RuntimeError"


# ---------------------------------------------------------------------------
# 5. OpenTelemetry traces
# ---------------------------------------------------------------------------


@pytest.fixture
def otel() -> Any:
    sdk = pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = sdk.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _otlp_agent(provider: Any, **kwargs: Any) -> PromptiseAgent:
    from promptise.observability_transporters import OTLPTransporter

    agent = _observed_agent(**kwargs)
    otlp = OTLPTransporter(tracer_provider=provider, correlation_id="req-7f3a")
    agent.collector.add_transporter(otlp)
    agent._transporters.append(otlp)
    return agent


class TestOTLPTraces:
    @pytest.mark.asyncio
    async def test_run_is_one_trace_with_child_spans(self, otel) -> None:
        provider, exporter = otel
        agent = _otlp_agent(provider, inner=_SimulatedGraph(llm_delay=0.05))
        caller = CallerContext(user_id="customer-42", metadata={"session_id": "q1"})
        await agent.ainvoke(USER_MSG, caller=caller)
        spans = {s.name: s for s in exporter.get_finished_spans()}
        assert set(spans) == {"invoke_agent shop", "chat gpt-5-mini", "execute_tool check_stock"}
        run, chat, tool = (
            spans["invoke_agent shop"],
            spans["chat gpt-5-mini"],
            spans["execute_tool check_stock"],
        )
        assert len({s.context.trace_id for s in spans.values()}) == 1
        assert run.parent is None
        assert chat.parent.span_id == run.context.span_id
        assert tool.parent.span_id == run.context.span_id

        def ms(span: Any) -> float:
            return (span.end_time - span.start_time) / 1e6

        assert ms(chat) >= 45  # the real LLM latency, not ~0
        assert ms(tool) >= 15
        assert ms(run) >= ms(chat) + ms(tool) - 1

        for span in spans.values():
            assert span.attributes["enduser.id"] == "customer-42"
            assert span.attributes["session.id"] == "q1"
            assert span.attributes["promptise.correlation_id"] == "req-7f3a"
        assert chat.attributes["gen_ai.usage.input_tokens"] == 12
        assert chat.attributes["gen_ai.usage.output_tokens"] == 5
        assert run.attributes["gen_ai.usage.input_tokens"] == 12
        assert tool.attributes["gen_ai.tool.name"] == "check_stock"
        assert run.attributes["gen_ai.operation.name"] == "invoke_agent"

    @pytest.mark.asyncio
    async def test_failed_tool_span_has_error_status(self, otel) -> None:
        from opentelemetry.trace import StatusCode

        provider, exporter = otel
        agent = _otlp_agent(provider, inner=_SimulatedGraph(tool_fails=True))
        await agent.ainvoke(USER_MSG)
        tool = next(s for s in exporter.get_finished_spans() if s.name.startswith("execute_tool"))
        assert tool.status.status_code == StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_failed_run_span_has_error_status(self, otel) -> None:
        from opentelemetry.trace import StatusCode

        provider, exporter = otel
        inner = MagicMock()
        inner.ainvoke = AsyncMock(side_effect=ValueError("boom"))
        agent = _otlp_agent(provider, inner=inner)
        with pytest.raises(ValueError):
            await agent.ainvoke(USER_MSG)
        [run] = exporter.get_finished_spans()
        assert run.status.status_code == StatusCode.ERROR

    @pytest.mark.asyncio
    async def test_run_joins_the_callers_active_span(self, otel) -> None:
        provider, exporter = otel
        agent = _otlp_agent(provider)
        with provider.get_tracer("app").start_as_current_span("POST /chat") as request_span:
            await agent.ainvoke(USER_MSG)
        run = next(s for s in exporter.get_finished_spans() if s.name.startswith("invoke_agent"))
        assert run.parent.span_id == request_span.get_span_context().span_id

    @pytest.mark.asyncio
    async def test_cache_events_become_span_events(self, otel) -> None:
        from promptise.cache import SemanticCache

        provider, exporter = otel
        agent = _otlp_agent(provider, cache=SemanticCache(embedding=_Embed()))
        caller = CallerContext(user_id="u1")
        await agent.ainvoke(USER_MSG, caller=caller)
        await agent.ainvoke(USER_MSG, caller=caller)
        runs = [s for s in exporter.get_finished_spans() if s.name.startswith("invoke_agent")]
        assert [e.name for e in runs[-1].events] == ["promptise.cache.hit"]

    def test_global_tracer_provider_is_not_replaced(self, otel) -> None:
        from opentelemetry import trace

        from promptise.observability_transporters import OTLPTransporter

        before = trace.get_tracer_provider()
        OTLPTransporter(endpoint="http://127.0.0.1:1")
        assert trace.get_tracer_provider() is before

    def test_close_ends_open_spans(self, otel) -> None:
        from promptise.observability_transporters import OTLPTransporter

        provider, exporter = otel
        collector = ObservabilityCollector()
        otlp = OTLPTransporter(tracer_provider=provider)
        collector.add_transporter(otlp)
        collector.record(TimelineEventType.AGENT_INPUT, agent_id="a")
        assert exporter.get_finished_spans() == ()
        otlp.close()
        assert [s.name for s in exporter.get_finished_spans()] == ["invoke_agent a"]

    def test_standalone_event_becomes_its_own_span(self, otel) -> None:
        from promptise.observability_transporters import OTLPTransporter

        provider, exporter = otel
        collector = ObservabilityCollector()
        collector.add_transporter(OTLPTransporter(tracer_provider=provider))
        collector.record(TimelineEventType.HEALTH_CHECK, details="ok")
        assert [s.name for s in exporter.get_finished_spans()] == ["promptise.health.check"]


# ---------------------------------------------------------------------------
# 6. Secrets and PII are scrubbed before events are stored or exported
# ---------------------------------------------------------------------------

API_KEY = "sk-" + "a1B2c3D4e5F6g7H8i9J0k1L2m3"
BEARER = "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJtYXJhIn0.c2lnbmF0dXJl"
DB_URL = "postgres://billing:hunter2pw@db.internal:5432/ledger"
EMAIL = "mara.lindqvist@example.com"
SECRETS = (API_KEY, BEARER.split()[1], "hunter2pw", EMAIL)


class _LeakyGraph(_SimulatedGraph):
    """A run whose tool call carries credentials and whose result and error carry PII."""

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        handlers = [h for h in (config or {}).get("callbacks", []) if hasattr(h, "on_llm_start")]
        ok, failed = uuid4(), uuid4()
        for h in handlers:
            h.on_tool_start(
                {"name": "charge"},
                str({"api_key": API_KEY, "auth": BEARER, "dsn": DB_URL}),
                run_id=ok,
            )
            h.on_tool_end(f'{{"receipt_to": "{EMAIL}"}}', run_id=ok, name="charge")
            h.on_tool_start({"name": "refund"}, "{}", run_id=failed)
            h.on_tool_error(PermissionError(f"key {API_KEY} rejected"), run_id=failed)
        return {"messages": [*input.get("messages", []), AIMessage(content="done")]}


async def _build_observed(tmp_path: Path, **config: Any) -> PromptiseAgent:
    with (
        patch("promptise.agent._normalize_model", return_value=MagicMock()),
        patch("promptise.agent.PromptGraphEngine", return_value=_LeakyGraph()),
    ):
        return await build_agent(
            servers={},
            model="openai:gpt-5-mini",
            observe=ObservabilityConfig(
                level=ObserveLevel.FULL,
                transporters=[TransporterType.JSON],
                output_dir=str(tmp_path),
                **config,
            ),
        )


class TestSecretRedaction:
    @pytest.mark.asyncio
    async def test_build_agent_scrubs_secrets_from_events_and_exports(self, tmp_path) -> None:
        agent = await _build_observed(tmp_path)
        await agent.ainvoke({"messages": [{"role": "user", "content": f"mail {EMAIL}"}]})
        await agent.shutdown()

        recorded = json.dumps([e.to_dict() for e in agent.collector.get_timeline()])
        exported = "".join(p.read_text() for p in tmp_path.iterdir())
        for text in (recorded, exported):
            for secret in SECRETS:
                assert secret not in text
        for placeholder in ("[API_KEY]", "Bearer [REDACTED]", "://[REDACTED]@", "[EMAIL]"):
            assert placeholder in recorded
        # Structure survives: ids, tool names, numbers.
        call = agent.collector.query(event_types=["tool.call"])[0]
        assert call.metadata["tool_name"] == "charge"
        assert call.parent_id is not None
        err = agent.collector.query(event_types=["tool.error"])[0]
        assert err.metadata["error"] == "key [API_KEY] rejected"

    @pytest.mark.asyncio
    async def test_redaction_can_be_switched_off(self, tmp_path) -> None:
        agent = await _build_observed(tmp_path, redact_sensitive=False)
        await agent.ainvoke(USER_MSG)
        await agent.shutdown()
        recorded = json.dumps([e.to_dict() for e in agent.collector.get_timeline()])
        assert API_KEY in recorded

    def test_details_are_scrubbed_too(self) -> None:
        c = ObservabilityCollector(sanitizer=redact_sensitive)
        e = c.record(TimelineEventType.TOOL_ERROR, details=f"Agent error: bad key {API_KEY}")
        assert e.details == "Agent error: bad key [API_KEY]"

    def test_redact_sensitive_keeps_ids_and_non_strings(self) -> None:
        run_id = "12345678-1234-1234-1234-123456789012"  # matches the card pattern
        meta = {
            "run_id": run_id,
            "latency_ms": 12.5,
            "ok": True,
            "tools_requested": ["search", f"notify {EMAIL}"],
            "nested": {"dsn": DB_URL},
        }
        out = redact_sensitive(meta)
        assert out["run_id"] == run_id
        assert (out["latency_ms"], out["ok"]) == (12.5, True)
        assert out["tools_requested"] == ["search", "notify [EMAIL]"]
        assert out["nested"] == {"dsn": "postgres://[REDACTED]@db.internal:5432/ledger"}
        assert meta["nested"]["dsn"] == DB_URL  # input not mutated

    @pytest.mark.asyncio
    async def test_guardrail_violation_is_recorded_without_the_matched_text(self) -> None:
        from promptise.guardrails import (
            Action,
            GuardrailViolation,
            ScanReport,
            SecurityFinding,
            Severity,
        )

        finding = SecurityFinding(
            detector="ner",
            category="ner_person",
            severity=Severity.MEDIUM,
            confidence=0.9,
            matched_text="Mara Lindqvist",
            start=0,
            end=14,
            action=Action.BLOCK,
            description="person detected: 'Mara Lindqvist'",
        )
        violation = GuardrailViolation(
            ScanReport(
                passed=False,
                findings=[finding],
                duration_ms=1.0,
                scanners_run=["ner"],
                text_length=30,
            )
        )
        inner = MagicMock()
        inner.ainvoke = AsyncMock(side_effect=violation)
        agent = _observed_agent(inner=inner)
        with pytest.raises(GuardrailViolation):
            await agent.ainvoke(USER_MSG)
        err = agent.collector.query(event_types=["agent.error"])[0]
        assert "Mara" not in json.dumps(err.to_dict())
        assert err.metadata["error"] == "Guardrail violation (input): 1 blocked finding(s)"
        assert err.metadata["guardrail_categories"] == ["ner_person"]


# ---------------------------------------------------------------------------
# 7. Config plumbing
# ---------------------------------------------------------------------------


class TestConfigPlumbing:
    def test_superagent_dict_becomes_config(self) -> None:
        from promptise.agent import _observability_config_from_mapping

        cfg = _observability_config_from_mapping(
            {"level": "basic", "transporters": ["json", "otlp"], "record_tool_io": False}
        )
        assert cfg.level == ObserveLevel.BASIC
        assert cfg.transporters == [TransporterType.JSON, TransporterType.OTLP]
        assert cfg.record_tool_io is False

    def test_unknown_key_is_an_error(self) -> None:
        from promptise.agent import _observability_config_from_mapping

        with pytest.raises(TypeError):
            _observability_config_from_mapping({"levle": "basic"})

    def test_superagent_section_defaults(self) -> None:
        from promptise.superagent_schema import ObservabilitySection

        section = ObservabilitySection()
        assert section.record_prompts is None
        assert section.record_tool_io is True

    @pytest.mark.asyncio
    async def test_build_agent_accepts_superagent_dict(self, tmp_path) -> None:
        with (
            patch("promptise.agent._normalize_model", return_value=MagicMock()),
            patch("promptise.agent.PromptGraphEngine", return_value=_SimulatedGraph()),
        ):
            agent = await build_agent(
                servers={},
                model="openai:gpt-5-mini",
                observe={"level": "basic", "transporters": [], "session_name": "from-file"},
            )
        assert agent.collector.session_name == "from-file"
        assert agent._handler.level == ObserveLevel.BASIC

    def test_run_context_dataclass(self) -> None:
        run = AgentRun(entry_id="e", started=time.time(), prompt_tokens=3, completion_tokens=4)
        assert run.total_tokens == 7


class TestPlainStreamRun:
    @pytest.mark.asyncio
    async def test_astream_records_a_run_without_output_text(self) -> None:
        inner = MagicMock()

        async def astream(*args: Any, **kwargs: Any) -> Any:
            yield {"chunk": 1}
            yield {"chunk": 2}

        inner.astream = astream
        agent = _observed_agent(ObserveLevel.FULL, inner=inner)
        chunks = [c async for c in agent.astream(USER_MSG)]
        assert len(chunks) == 2
        assert _types(agent) == ["agent.input", "agent.output"]
        out = agent.collector.query(event_types=["agent.output"])[0]
        assert "output_length" not in out.metadata
        assert "output_preview" not in out.metadata
        assert _run_ctx_var.get() is None
