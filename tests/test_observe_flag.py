"""Comprehensive tests for the observe flag, PromptiseAgent, ObservabilityConfig,
and ObservabilityCollector enhanced features.

Tests are organised into three groups:

1. **PromptiseAgent** — the unified agent returned by ``build_agent()``.
   We construct it directly with a mock ``Runnable`` inner so we never need live MCP servers.
2. **ObservabilityConfig / enums** — default values, enum completeness.
3. **ObservabilityCollector** — ring buffer, query filters, stats, serialisation, and
   transporter dispatch.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptise.agent import PromptiseAgent
from promptise.observability import (
    ObservabilityCollector,
    TimelineEventType,
)
from promptise.observability_config import (
    ObservabilityConfig,
    ObserveLevel,
    TransporterType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_inner() -> MagicMock:
    """Return a mock object that quacks like a LangGraph Runnable."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value={"messages": []})
    mock.invoke = MagicMock(return_value={"messages": []})
    # Give it an extra attribute to test __getattr__ proxying
    mock.some_custom_attr = "hello"
    return mock


def _make_agent(
    inner: MagicMock | None = None,
    collector: ObservabilityCollector | None = None,
) -> PromptiseAgent:
    """Construct a PromptiseAgent with observability for testing."""
    inner = inner or _make_mock_inner()
    collector = collector or ObservabilityCollector(session_name="test")
    handler = MagicMock(name="PromptiseCallbackHandler")
    config = ObservabilityConfig()
    return PromptiseAgent(
        inner=inner,
        handler=handler,
        collector=collector,
        observe_config=config,
    )


# ===========================================================================
# 1. PromptiseAgent wrapper tests
# ===========================================================================


class TestPromptiseAgentAinvoke:
    """PromptiseAgent.ainvoke() injects callback handler into config."""

    @pytest.mark.asyncio
    async def test_ainvoke_injects_handler(self) -> None:
        inner = _make_mock_inner()
        agent = _make_agent(inner=inner)

        await agent.ainvoke({"messages": [{"role": "user", "content": "hi"}]})

        inner.ainvoke.assert_awaited_once()
        _, kwargs = inner.ainvoke.call_args
        config = kwargs["config"]
        assert agent._handler in config["callbacks"]

    @pytest.mark.asyncio
    async def test_ainvoke_returns_inner_result(self) -> None:
        inner = _make_mock_inner()
        inner.ainvoke = AsyncMock(
            return_value={"messages": [{"role": "assistant", "content": "ok"}]}
        )
        agent = _make_agent(inner=inner)

        result = await agent.ainvoke({"messages": []})

        assert result == {"messages": [{"role": "assistant", "content": "ok"}]}


class TestPromptiseAgentInvoke:
    """PromptiseAgent.invoke() delegates through ainvoke for full pipeline."""

    def test_invoke_delegates_to_ainvoke(self) -> None:
        inner = _make_mock_inner()
        inner.ainvoke = AsyncMock(
            return_value={"messages": [{"role": "assistant", "content": "ok"}]}
        )
        agent = _make_agent(inner=inner)

        result = agent.invoke({"messages": []})

        # invoke() now always goes through ainvoke → ainvoke_inner → inner.ainvoke
        inner.ainvoke.assert_called_once()
        assert result["messages"][-1]["content"] == "ok"

    def test_invoke_returns_inner_result(self) -> None:
        inner = _make_mock_inner()
        inner.ainvoke = AsyncMock(return_value={"messages": [{"text": "sync result"}]})
        agent = _make_agent(inner=inner)

        result = agent.invoke({"messages": []})

        assert result == {"messages": [{"text": "sync result"}]}


class TestPromptiseAgentGetStats:
    """PromptiseAgent.get_stats() returns dict from collector."""

    def test_get_stats_returns_dict(self) -> None:
        collector = ObservabilityCollector(session_name="stats-test")
        collector.record(
            TimelineEventType.LLM_END,
            agent_id="a1",
            metadata={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        )
        agent = _make_agent(collector=collector)

        stats = agent.get_stats()

        assert isinstance(stats, dict)
        assert stats["entry_count"] == 1
        assert stats["total_tokens"] == 30

    def test_get_stats_empty_collector(self) -> None:
        collector = ObservabilityCollector(session_name="empty")
        agent = _make_agent(collector=collector)
        stats = agent.get_stats()

        assert stats["entry_count"] == 0
        assert stats["total_tokens"] == 0
        assert stats["error_count"] == 0


class TestPromptiseAgentGenerateReport:
    """PromptiseAgent.generate_report() generates a file."""

    def test_generate_report_creates_file(self) -> None:
        collector = ObservabilityCollector(session_name="report-test")
        collector.record(TimelineEventType.SESSION_START, details="start")
        agent = _make_agent(collector=collector)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "report.html")
            with patch("promptise.observability_transporters.HTMLReportTransporter.flush"):
                result = agent.generate_report(path, title="Test Report")
                assert result == path


class TestPromptiseAgentShutdown:
    """PromptiseAgent.shutdown() calls flush on transporters."""

    @pytest.mark.asyncio
    async def test_shutdown_calls_flush(self) -> None:
        agent = _make_agent()
        t1 = MagicMock()
        t1.flush = MagicMock()
        t2 = MagicMock()
        t2.flush = MagicMock()
        agent._transporters = [t1, t2]

        await agent.shutdown()

        t1.flush.assert_called_once()
        t2.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_handles_async_flush(self) -> None:
        agent = _make_agent()
        t1 = MagicMock()
        t1.flush = AsyncMock()
        agent._transporters = [t1]

        await agent.shutdown()

        t1.flush.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_tolerates_flush_error(self) -> None:
        agent = _make_agent()
        t1 = MagicMock()
        t1.flush = MagicMock(side_effect=RuntimeError("boom"))
        agent._transporters = [t1]

        # Should not raise
        await agent.shutdown()


class TestPromptiseAgentGetattr:
    """PromptiseAgent.__getattr__ proxies to inner."""

    def test_getattr_proxies_to_inner(self) -> None:
        inner = _make_mock_inner()
        inner.custom_method = MagicMock(return_value=42)
        agent = _make_agent(inner=inner)

        assert agent.custom_method() == 42
        inner.custom_method.assert_called_once()

    def test_getattr_proxies_attribute(self) -> None:
        inner = _make_mock_inner()
        agent = _make_agent(inner=inner)

        assert agent.some_custom_attr == "hello"

    def test_getattr_raises_for_missing(self) -> None:
        inner = MagicMock(spec=[])  # Empty spec so getattr raises AttributeError
        agent = _make_agent(inner=inner)

        with pytest.raises(AttributeError):
            _ = agent.nonexistent_attribute


class TestPromptiseAgentPreservesCallbacks:
    """PromptiseAgent preserves existing callbacks in config."""

    @pytest.mark.asyncio
    async def test_ainvoke_preserves_existing_callbacks(self) -> None:
        inner = _make_mock_inner()
        agent = _make_agent(inner=inner)
        existing_cb = MagicMock(name="existing_callback")

        await agent.ainvoke(
            {"messages": []},
            config={"callbacks": [existing_cb]},
        )

        _, kwargs = inner.ainvoke.call_args
        callbacks = kwargs["config"]["callbacks"]
        assert existing_cb in callbacks
        assert agent._handler in callbacks
        assert len(callbacks) == 2

    def test_invoke_preserves_existing_callbacks(self) -> None:
        inner = _make_mock_inner()
        agent = _make_agent(inner=inner)
        existing_cb = MagicMock(name="existing_callback")

        agent.invoke(
            {"messages": []},
            config={"callbacks": [existing_cb]},
        )

        # invoke() now delegates through ainvoke → inner.ainvoke
        inner.ainvoke.assert_called_once()
        _, kwargs = inner.ainvoke.call_args
        callbacks = kwargs["config"]["callbacks"]
        assert existing_cb in callbacks
        assert agent._handler in callbacks


class TestPromptiseAgentConfigNone:
    """PromptiseAgent handles config=None gracefully."""

    @pytest.mark.asyncio
    async def test_ainvoke_config_none(self) -> None:
        inner = _make_mock_inner()
        agent = _make_agent(inner=inner)

        await agent.ainvoke({"messages": []}, config=None)

        inner.ainvoke.assert_awaited_once()
        _, kwargs = inner.ainvoke.call_args
        config = kwargs["config"]
        assert "callbacks" in config
        assert agent._handler in config["callbacks"]

    def test_invoke_config_none(self) -> None:
        inner = _make_mock_inner()
        agent = _make_agent(inner=inner)

        agent.invoke({"messages": []}, config=None)

        # invoke() now delegates through ainvoke
        inner.ainvoke.assert_called()
        _, kwargs = inner.ainvoke.call_args
        config = kwargs["config"]
        assert "callbacks" in config
        assert agent._handler in config["callbacks"]

    @pytest.mark.asyncio
    async def test_ainvoke_no_config_arg(self) -> None:
        """When config is omitted entirely (defaults to None)."""
        inner = _make_mock_inner()
        agent = _make_agent(inner=inner)

        await agent.ainvoke({"messages": []})

        _, kwargs = inner.ainvoke.call_args
        assert agent._handler in kwargs["config"]["callbacks"]

    @pytest.mark.asyncio
    async def test_ainvoke_does_not_mutate_original_config(self) -> None:
        """Verify the original config dict passed by the caller is not mutated."""
        inner = _make_mock_inner()
        agent = _make_agent(inner=inner)
        original_config: dict[str, Any] = {"some_key": "some_value"}

        await agent.ainvoke({"messages": []}, config=original_config)

        # The original dict should NOT have been modified
        assert "callbacks" not in original_config


# ===========================================================================
# 2. ObservabilityConfig and enum tests
# ===========================================================================


class TestObservabilityConfigDefaults:
    """Default config has STANDARD level and HTML transporter."""

    def test_default_level_is_standard(self) -> None:
        config = ObservabilityConfig()
        assert config.level == ObserveLevel.STANDARD

    def test_default_transporters_has_html(self) -> None:
        config = ObservabilityConfig()
        assert config.transporters == [TransporterType.HTML]

    def test_default_record_prompts_follows_level(self) -> None:
        config = ObservabilityConfig()
        # None = follow the level: prompt/response text only at FULL.
        assert config.record_prompts is None

    def test_default_max_entries(self) -> None:
        config = ObservabilityConfig()
        assert config.max_entries == 100_000

    def test_default_session_name(self) -> None:
        config = ObservabilityConfig()
        assert config.session_name == "promptise"


class TestTransporterTypeValues:
    """All TransporterType values are valid."""

    def test_all_values_are_strings(self) -> None:
        for member in TransporterType:
            assert isinstance(member.value, str)

    def test_expected_transporters_exist(self) -> None:
        names = {m.name for m in TransporterType}
        expected = {
            "HTML",
            "JSON",
            "STRUCTURED_LOG",
            "CONSOLE",
            "PROMETHEUS",
            "OTLP",
            "WEBHOOK",
            "CALLBACK",
        }
        assert expected.issubset(names)

    def test_transporter_count(self) -> None:
        assert len(TransporterType) == 8


class TestObserveLevelValues:
    """ObserveLevel values are valid."""

    def test_all_values_are_strings(self) -> None:
        for member in ObserveLevel:
            assert isinstance(member.value, str)

    def test_expected_levels_exist(self) -> None:
        names = {m.name for m in ObserveLevel}
        assert names == {"OFF", "BASIC", "STANDARD", "FULL"}

    def test_level_string_values(self) -> None:
        assert ObserveLevel.OFF.value == "off"
        assert ObserveLevel.BASIC.value == "basic"
        assert ObserveLevel.STANDARD.value == "standard"
        assert ObserveLevel.FULL.value == "full"


# ===========================================================================
# 3. ObservabilityCollector enhanced feature tests
# ===========================================================================


class TestCollectorRingBuffer:
    """max_entries ring buffer evicts oldest."""

    def test_ring_buffer_evicts_oldest(self) -> None:
        collector = ObservabilityCollector(session_name="ring", max_entries=3)
        collector.record(TimelineEventType.TASK_CREATED, details="first")
        collector.record(TimelineEventType.TASK_STARTED, details="second")
        collector.record(TimelineEventType.TASK_COMPLETED, details="third")
        collector.record(TimelineEventType.TASK_FAILED, details="fourth")

        entries = collector.get_timeline()
        assert len(entries) == 3
        details = [e.details for e in entries]
        assert "first" not in details
        assert "second" in details
        assert "third" in details
        assert "fourth" in details

    def test_ring_buffer_at_capacity(self) -> None:
        collector = ObservabilityCollector(session_name="cap", max_entries=2)
        collector.record(TimelineEventType.TASK_CREATED, details="a")
        collector.record(TimelineEventType.TASK_STARTED, details="b")

        entries = collector.get_timeline()
        assert len(entries) == 2

    def test_ring_buffer_multiple_evictions(self) -> None:
        collector = ObservabilityCollector(session_name="multi", max_entries=2)
        for i in range(10):
            collector.record(TimelineEventType.TASK_CREATED, details=f"entry-{i}")

        entries = collector.get_timeline()
        assert len(entries) == 2
        details = [e.details for e in entries]
        assert "entry-8" in details
        assert "entry-9" in details


class TestCollectorQueryEventType:
    """query() filters by event_type."""

    def test_filter_by_single_event_type(self) -> None:
        collector = ObservabilityCollector(session_name="q1")
        collector.record(TimelineEventType.TOOL_CALL, details="call1")
        collector.record(TimelineEventType.TOOL_RESULT, details="result1")
        collector.record(TimelineEventType.TOOL_CALL, details="call2")

        results = collector.query(event_types=[TimelineEventType.TOOL_CALL])
        assert len(results) == 2
        assert all(e.event_type == TimelineEventType.TOOL_CALL for e in results)

    def test_filter_by_event_type_string(self) -> None:
        collector = ObservabilityCollector(session_name="q1str")
        collector.record(TimelineEventType.TOOL_CALL, details="call")
        collector.record(TimelineEventType.TOOL_RESULT, details="result")

        results = collector.query(event_types=["tool.call"])
        assert len(results) == 1
        assert results[0].event_type == TimelineEventType.TOOL_CALL

    def test_filter_by_multiple_event_types(self) -> None:
        collector = ObservabilityCollector(session_name="q1multi")
        collector.record(TimelineEventType.TOOL_CALL, details="call")
        collector.record(TimelineEventType.TOOL_RESULT, details="result")
        collector.record(TimelineEventType.LLM_END, details="llm")

        results = collector.query(
            event_types=[TimelineEventType.TOOL_CALL, TimelineEventType.TOOL_RESULT]
        )
        assert len(results) == 2


class TestCollectorQueryAgentId:
    """query() filters by agent_id."""

    def test_filter_by_agent_id(self) -> None:
        collector = ObservabilityCollector(session_name="q2")
        collector.record(TimelineEventType.TOOL_CALL, agent_id="agent-a", details="a1")
        collector.record(TimelineEventType.TOOL_CALL, agent_id="agent-b", details="b1")
        collector.record(TimelineEventType.TOOL_CALL, agent_id="agent-a", details="a2")

        results = collector.query(agent_ids=["agent-a"])
        assert len(results) == 2
        assert all(e.agent_id == "agent-a" for e in results)

    def test_filter_by_multiple_agent_ids(self) -> None:
        collector = ObservabilityCollector(session_name="q2multi")
        collector.record(TimelineEventType.TOOL_CALL, agent_id="a", details="1")
        collector.record(TimelineEventType.TOOL_CALL, agent_id="b", details="2")
        collector.record(TimelineEventType.TOOL_CALL, agent_id="c", details="3")

        results = collector.query(agent_ids=["a", "c"])
        assert len(results) == 2


class TestCollectorQueryLimitOffset:
    """query() with limit and offset."""

    def test_query_with_limit(self) -> None:
        collector = ObservabilityCollector(session_name="q3")
        for i in range(5):
            collector.record(TimelineEventType.TASK_CREATED, details=f"task-{i}")

        results = collector.query(limit=3)
        assert len(results) == 3

    def test_query_with_offset(self) -> None:
        collector = ObservabilityCollector(session_name="q3off")
        for i in range(5):
            collector.record(TimelineEventType.TASK_CREATED, details=f"task-{i}")

        results = collector.query(offset=2)
        assert len(results) == 3

    def test_query_with_limit_and_offset(self) -> None:
        collector = ObservabilityCollector(session_name="q3both")
        for i in range(10):
            collector.record(TimelineEventType.TASK_CREATED, details=f"task-{i}")

        results = collector.query(limit=3, offset=2)
        assert len(results) == 3
        assert results[0].details == "task-2"
        assert results[2].details == "task-4"

    def test_query_offset_beyond_entries(self) -> None:
        collector = ObservabilityCollector(session_name="q3beyond")
        collector.record(TimelineEventType.TASK_CREATED, details="only")

        results = collector.query(offset=100)
        assert len(results) == 0


class TestCollectorGetStats:
    """get_stats() returns correct structure."""

    def test_stats_keys(self) -> None:
        collector = ObservabilityCollector(session_name="stats")
        stats = collector.get_stats()
        expected_keys = {
            "entry_count",
            "agent_count",
            "total_duration_s",
            "total_prompt_tokens",
            "total_completion_tokens",
            "total_tokens",
            "llm_call_count",
            "tool_call_count",
            "error_count",
            "retry_count",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "tokens_by_agent",
            "events_by_category",
            "events_by_type",
        }
        assert expected_keys == set(stats.keys())

    def test_stats_counts_tokens(self) -> None:
        collector = ObservabilityCollector(session_name="tok")
        collector.record(
            TimelineEventType.LLM_END,
            agent_id="a",
            metadata={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        )
        collector.record(
            TimelineEventType.LLM_END,
            agent_id="a",
            metadata={
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
            },
        )

        stats = collector.get_stats()
        assert stats["total_prompt_tokens"] == 300
        assert stats["total_completion_tokens"] == 150
        assert stats["total_tokens"] == 450
        assert stats["llm_call_count"] == 2

    def test_stats_counts_errors(self) -> None:
        collector = ObservabilityCollector(session_name="err")
        collector.record(TimelineEventType.LLM_ERROR, details="oops")
        collector.record(TimelineEventType.TOOL_ERROR, details="fail")
        collector.record(TimelineEventType.TASK_FAILED, details="boom")

        stats = collector.get_stats()
        assert stats["error_count"] == 3

    def test_stats_counts_tools(self) -> None:
        collector = ObservabilityCollector(session_name="tools")
        collector.record(TimelineEventType.TOOL_CALL, details="call1")
        collector.record(TimelineEventType.TOOL_RESULT, details="result1")
        collector.record(TimelineEventType.TOOL_CALL, details="call2")

        stats = collector.get_stats()
        assert stats["tool_call_count"] == 2

    def test_stats_per_agent_breakdown(self) -> None:
        collector = ObservabilityCollector(session_name="per-agent")
        collector.record(
            TimelineEventType.LLM_END,
            agent_id="alpha",
            metadata={
                "total_tokens": 100,
            },
        )
        collector.record(
            TimelineEventType.LLM_END,
            agent_id="beta",
            metadata={
                "total_tokens": 200,
            },
        )

        stats = collector.get_stats()
        assert stats["tokens_by_agent"]["alpha"] == 100
        assert stats["tokens_by_agent"]["beta"] == 200
        assert stats["agent_count"] == 2


class TestCollectorClear:
    """clear() removes all entries."""

    def test_clear(self) -> None:
        collector = ObservabilityCollector(session_name="clr")
        collector.record(TimelineEventType.TASK_CREATED, details="1")
        collector.record(TimelineEventType.TASK_STARTED, details="2")
        assert len(collector.get_timeline()) == 2

        collector.clear()

        assert len(collector.get_timeline()) == 0

    def test_clear_resets_session_start(self) -> None:
        collector = ObservabilityCollector(session_name="clr2")
        original_start = collector.session_start
        import time

        time.sleep(0.01)

        collector.clear()

        assert collector.session_start > original_start


class TestCollectorToNdjson:
    """to_ndjson() produces valid JSON lines."""

    def test_ndjson_valid_json_per_line(self) -> None:
        collector = ObservabilityCollector(session_name="ndjson-test")
        collector.record(TimelineEventType.TASK_CREATED, details="first")
        collector.record(TimelineEventType.TASK_STARTED, details="second")

        ndjson = collector.to_ndjson()
        lines = ndjson.strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            parsed = json.loads(line)
            assert "entry_id" in parsed
            assert "event_type" in parsed
            assert parsed["session_name"] == "ndjson-test"

    def test_ndjson_empty_collector(self) -> None:
        collector = ObservabilityCollector(session_name="empty")
        ndjson = collector.to_ndjson()
        assert ndjson == ""

    def test_ndjson_preserves_metadata(self) -> None:
        collector = ObservabilityCollector(session_name="meta")
        collector.record(
            TimelineEventType.TOOL_CALL,
            metadata={"tool_name": "search", "arguments": "query=hello"},
        )

        ndjson = collector.to_ndjson()
        parsed = json.loads(ndjson.strip())
        assert parsed["metadata"]["tool_name"] == "search"


class TestCollectorTransporterDispatch:
    """add_transporter + dispatch works."""

    def test_transporter_receives_events(self) -> None:
        collector = ObservabilityCollector(session_name="transport")
        transporter = MagicMock()
        transporter.on_event = MagicMock()
        collector.add_transporter(transporter)

        collector.record(TimelineEventType.TASK_CREATED, details="dispatched")

        transporter.on_event.assert_called_once()
        entry = transporter.on_event.call_args[0][0]
        assert entry.details == "dispatched"

    def test_multiple_transporters_all_receive(self) -> None:
        collector = ObservabilityCollector(session_name="multi-t")
        t1, t2 = MagicMock(), MagicMock()
        t1.on_event = MagicMock()
        t2.on_event = MagicMock()
        collector.add_transporter(t1)
        collector.add_transporter(t2)

        collector.record(TimelineEventType.TASK_STARTED, details="event")

        t1.on_event.assert_called_once()
        t2.on_event.assert_called_once()

    def test_transporter_error_does_not_break_recording(self) -> None:
        collector = ObservabilityCollector(session_name="err-t")
        bad_transporter = MagicMock()
        bad_transporter.on_event = MagicMock(side_effect=RuntimeError("transporter crashed"))
        collector.add_transporter(bad_transporter)

        # Should not raise
        entry = collector.record(TimelineEventType.TASK_CREATED, details="still recorded")

        assert entry.details == "still recorded"
        assert len(collector.get_timeline()) == 1

    def test_transporter_receives_all_events(self) -> None:
        collector = ObservabilityCollector(session_name="all-events")
        transporter = MagicMock()
        transporter.on_event = MagicMock()
        collector.add_transporter(transporter)

        for i in range(5):
            collector.record(TimelineEventType.TOOL_CALL, details=f"call-{i}")

        assert transporter.on_event.call_count == 5


# ===========================================================================
# 4. Additional collector query and metadata tests
# ===========================================================================


class TestCollectorQueryMetadata:
    """query() filters by metadata_key and metadata_value."""

    def test_filter_by_metadata_key(self) -> None:
        collector = ObservabilityCollector(session_name="meta-q")
        collector.record(TimelineEventType.TOOL_CALL, metadata={"tool_name": "search"})
        collector.record(TimelineEventType.TOOL_CALL, metadata={"other": "data"})

        results = collector.query(metadata_key="tool_name")
        assert len(results) == 1
        assert results[0].metadata["tool_name"] == "search"

    def test_filter_by_metadata_key_and_value(self) -> None:
        collector = ObservabilityCollector(session_name="meta-qv")
        collector.record(TimelineEventType.TOOL_CALL, metadata={"tool_name": "search"})
        collector.record(TimelineEventType.TOOL_CALL, metadata={"tool_name": "fetch"})

        results = collector.query(metadata_key="tool_name", metadata_value="fetch")
        assert len(results) == 1
        assert results[0].metadata["tool_name"] == "fetch"


class TestCollectorQueryComposite:
    """query() with combined filters."""

    def test_combined_event_type_and_agent_id(self) -> None:
        collector = ObservabilityCollector(session_name="combo")
        collector.record(TimelineEventType.TOOL_CALL, agent_id="a1", details="tc-a1")
        collector.record(TimelineEventType.TOOL_CALL, agent_id="a2", details="tc-a2")
        collector.record(TimelineEventType.LLM_END, agent_id="a1", details="llm-a1")

        results = collector.query(
            event_types=[TimelineEventType.TOOL_CALL],
            agent_ids=["a1"],
        )
        assert len(results) == 1
        assert results[0].details == "tc-a1"


class TestCollectorSpan:
    """span() context manager sets duration."""

    def test_span_records_duration(self) -> None:
        collector = ObservabilityCollector(session_name="span")

        with collector.span(TimelineEventType.PHASE_START, phase="test") as entry:
            import time

            time.sleep(0.01)

        assert entry.duration is not None
        assert entry.duration >= 0.01


class TestCollectorSerialization:
    """to_dict() and to_json() produce correct output."""

    def test_to_dict_has_required_keys(self) -> None:
        collector = ObservabilityCollector(session_name="ser")
        collector.record(TimelineEventType.SESSION_START, details="begin")

        d = collector.to_dict()
        assert d["session_name"] == "ser"
        assert "entries" in d
        assert "stats" in d
        assert d["entry_count"] == 1

    def test_to_json_is_valid(self) -> None:
        collector = ObservabilityCollector(session_name="json-test")
        collector.record(TimelineEventType.SESSION_START, details="begin")

        j = collector.to_json()
        parsed = json.loads(j)
        assert parsed["session_name"] == "json-test"


# ===========================================================================
# 5. PromptiseAgent — no-observe / no-feature edge cases
# ===========================================================================


class TestPromptiseAgentNoObserve:
    """PromptiseAgent without observability should no-op gracefully."""

    def test_get_stats_returns_empty_dict(self) -> None:
        inner = _make_mock_inner()
        agent = PromptiseAgent(inner=inner)
        assert agent.get_stats() == {}

    def test_generate_report_raises_runtime_error(self) -> None:
        inner = _make_mock_inner()
        agent = PromptiseAgent(inner=inner)
        with pytest.raises(RuntimeError, match="observability is not enabled"):
            agent.generate_report("/tmp/report.html")

    @pytest.mark.asyncio
    async def test_shutdown_no_features(self) -> None:
        """shutdown() should not raise when no features are enabled."""
        inner = _make_mock_inner()
        agent = PromptiseAgent(inner=inner)
        await agent.shutdown()  # Should not raise

    @pytest.mark.asyncio
    async def test_ainvoke_bare_agent(self) -> None:
        """Bare agent (no observe, no memory) should pass through to inner."""
        inner = _make_mock_inner()
        inner.ainvoke = AsyncMock(
            return_value={"messages": [{"role": "assistant", "content": "hi"}]}
        )
        agent = PromptiseAgent(inner=inner)
        result = await agent.ainvoke({"messages": []})
        assert result["messages"][0]["content"] == "hi"
        inner.ainvoke.assert_awaited_once()

    def test_collector_is_none(self) -> None:
        inner = _make_mock_inner()
        agent = PromptiseAgent(inner=inner)
        assert agent.collector is None

    def test_provider_is_none(self) -> None:
        inner = _make_mock_inner()
        agent = PromptiseAgent(inner=inner)
        assert agent.provider is None
