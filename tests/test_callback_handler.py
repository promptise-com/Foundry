"""Tests for PromptiseCallbackHandler — the LangChain callback bridge.

Covers: on_llm_start, on_llm_end, on_llm_new_token, on_llm_error,
on_tool_start, on_tool_end, on_tool_error, on_chain_start, on_chain_end,
on_chain_error, on_retry, cumulative accounting, privacy controls, and
get_summary.
"""

import time
from unittest.mock import MagicMock
from uuid import uuid4

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from promptise.callback_handler import PromptiseCallbackHandler
from promptise.observability import ObservabilityCollector, TimelineEventType
from promptise.observability_config import ObserveLevel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_handler(
    *,
    record_prompts: bool = False,
    level: ObserveLevel = ObserveLevel.STANDARD,
    agent_id: str = "test-agent",
) -> tuple[PromptiseCallbackHandler, ObservabilityCollector]:
    """Create a handler + collector pair wired together."""
    collector = ObservabilityCollector("test-session")
    handler = PromptiseCallbackHandler(
        collector,
        agent_id=agent_id,
        record_prompts=record_prompts,
        level=level,
    )
    return handler, collector


def _make_llm_result(
    text: str = "Hello world",
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int = 0,
    model_name: str = "",
    tool_calls: list | None = None,
    usage_metadata: dict | None = None,
) -> LLMResult:
    """Build a realistic LLMResult with token_usage in llm_output."""
    msg = AIMessage(content=text)
    if tool_calls is not None:
        msg.tool_calls = tool_calls
    if usage_metadata is not None:
        msg.usage_metadata = usage_metadata

    generation = ChatGeneration(message=msg, text=text)
    llm_output = {}
    if prompt_tokens or completion_tokens or total_tokens:
        llm_output["token_usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    if model_name:
        llm_output["model_name"] = model_name

    return LLMResult(generations=[[generation]], llm_output=llm_output or None)


# ---------------------------------------------------------------------------
# 1. on_llm_start
# ---------------------------------------------------------------------------


class TestOnLlmStart:
    def test_records_llm_start_event(self):
        handler, collector = _make_handler()
        run_id = uuid4()
        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o"}, "id": ["langchain", "chat_models", "openai"]},
            ["What is 2+2?"],
            run_id=run_id,
        )
        entries = collector.query(event_types=[TimelineEventType.LLM_START])
        assert len(entries) == 1
        assert entries[0].metadata["model"] == "gpt-4o"
        assert entries[0].metadata["run_id"] == str(run_id)

    def test_extracts_model_from_serialized_id_fallback(self):
        handler, collector = _make_handler()
        run_id = uuid4()
        handler.on_llm_start(
            {"id": ["langchain", "chat_models", "ChatAnthropic"]},
            ["hello"],
            run_id=run_id,
        )
        entries = collector.query(event_types=[TimelineEventType.LLM_START])
        assert entries[0].metadata["model"] == "ChatAnthropic"

    def test_stores_start_time(self):
        handler, _ = _make_handler()
        run_id = uuid4()
        before = time.perf_counter()
        handler.on_llm_start({"id": ["x"]}, ["p"], run_id=run_id)
        after = time.perf_counter()
        assert run_id in handler._llm_starts
        assert before <= handler._llm_starts[run_id] <= after

    def test_basic_level_skips_recording(self):
        handler, collector = _make_handler(level=ObserveLevel.BASIC)
        run_id = uuid4()
        handler.on_llm_start({"id": ["x"]}, ["p"], run_id=run_id)
        entries = collector.query(event_types=[TimelineEventType.LLM_START])
        assert len(entries) == 0
        # But llm_call_count should still be incremented
        assert handler.llm_call_count == 1
        # And the start time should still be tracked for latency
        assert run_id in handler._llm_starts


# ---------------------------------------------------------------------------
# 2. on_llm_end
# ---------------------------------------------------------------------------


class TestOnLlmEnd:
    def test_extracts_tokens_from_llm_output(self):
        handler, collector = _make_handler()
        run_id = uuid4()
        handler.on_llm_start({"id": ["x"]}, ["p"], run_id=run_id)

        result = _make_llm_result(
            prompt_tokens=100, completion_tokens=50, total_tokens=150, model_name="gpt-4o"
        )
        handler.on_llm_end(result, run_id=run_id)

        entries = collector.query(event_types=[TimelineEventType.LLM_END])
        assert len(entries) == 1
        meta = entries[0].metadata
        assert meta["prompt_tokens"] == 100
        assert meta["completion_tokens"] == 50
        assert meta["total_tokens"] == 150

    def test_extracts_tokens_from_usage_metadata(self):
        """When llm_output has no token_usage, fall back to usage_metadata on the message."""
        handler, collector = _make_handler()
        run_id = uuid4()
        handler.on_llm_start({"id": ["x"]}, ["p"], run_id=run_id)

        result = _make_llm_result(
            text="response",
            usage_metadata={"input_tokens": 200, "output_tokens": 80},
        )
        handler.on_llm_end(result, run_id=run_id)

        entries = collector.query(event_types=[TimelineEventType.LLM_END])
        meta = entries[0].metadata
        assert meta["prompt_tokens"] == 200
        assert meta["completion_tokens"] == 80
        assert meta["total_tokens"] == 280

    def test_computes_latency_ms(self):
        handler, collector = _make_handler()
        run_id = uuid4()
        handler.on_llm_start({"id": ["x"]}, ["p"], run_id=run_id)
        time.sleep(0.05)

        result = _make_llm_result()
        handler.on_llm_end(result, run_id=run_id)

        entries = collector.query(event_types=[TimelineEventType.LLM_END])
        latency = entries[0].metadata["latency_ms"]
        assert latency >= 40  # at least ~50ms minus timing variance

    def test_records_response_preview_when_record_prompts_true(self):
        handler, collector = _make_handler(record_prompts=True)
        run_id = uuid4()
        handler.on_llm_start({"id": ["x"]}, ["p"], run_id=run_id)

        result = _make_llm_result(text="This is the response text.")
        handler.on_llm_end(result, run_id=run_id)

        entries = collector.query(event_types=[TimelineEventType.LLM_END])
        assert entries[0].metadata["response_preview"] == "This is the response text."

    def test_records_tool_calls(self):
        handler, collector = _make_handler()
        run_id = uuid4()
        handler.on_llm_start({"id": ["x"]}, ["p"], run_id=run_id)

        result = _make_llm_result(
            text="",
            tool_calls=[
                {"name": "search", "args": {"q": "test"}, "id": "tc1"},
                {"name": "read_file", "args": {"path": "x"}, "id": "tc2"},
            ],
        )
        handler.on_llm_end(result, run_id=run_id)

        entries = collector.query(event_types=[TimelineEventType.LLM_END])
        assert entries[0].metadata["tool_calls"] == ["search", "read_file"]

    def test_streaming_token_summary(self):
        handler, collector = _make_handler(level=ObserveLevel.FULL)
        run_id = uuid4()
        handler.on_llm_start({"id": ["x"]}, ["p"], run_id=run_id)

        # Simulate streaming tokens
        for tok in ["Hello", " ", "world", "!"]:
            handler.on_llm_new_token(tok, run_id=run_id)

        result = _make_llm_result(text="Hello world!")
        handler.on_llm_end(result, run_id=run_id)

        entries = collector.query(event_types=[TimelineEventType.LLM_END])
        assert entries[0].metadata["streamed_token_count"] == 4


# ---------------------------------------------------------------------------
# 3. on_llm_new_token
# ---------------------------------------------------------------------------


class TestOnLlmNewToken:
    def test_accumulates_at_full_level(self):
        handler, _ = _make_handler(level=ObserveLevel.FULL)
        run_id = uuid4()
        handler.on_llm_new_token("Hello", run_id=run_id)
        handler.on_llm_new_token(" world", run_id=run_id)

        assert run_id in handler._streaming_tokens
        assert handler._streaming_tokens[run_id] == ["Hello", " world"]

    def test_ignored_at_standard_level(self):
        handler, _ = _make_handler(level=ObserveLevel.STANDARD)
        run_id = uuid4()
        handler.on_llm_new_token("Hello", run_id=run_id)

        assert run_id not in handler._streaming_tokens


# ---------------------------------------------------------------------------
# 4. on_llm_error
# ---------------------------------------------------------------------------


class TestOnLlmError:
    def test_records_llm_error_with_traceback(self):
        handler, collector = _make_handler()
        run_id = uuid4()
        handler.on_llm_start({"id": ["x"]}, ["p"], run_id=run_id)

        try:
            raise ValueError("Rate limit exceeded")
        except ValueError as exc:
            handler.on_llm_error(exc, run_id=run_id)

        entries = collector.query(event_types=[TimelineEventType.LLM_ERROR])
        assert len(entries) == 1
        meta = entries[0].metadata
        assert meta["error_type"] == "ValueError"
        assert "Rate limit exceeded" in meta["error"]
        assert "traceback" in meta

    def test_increments_error_count(self):
        handler, _ = _make_handler()
        run_id = uuid4()
        handler.on_llm_error(RuntimeError("fail"), run_id=run_id)
        assert handler.error_count == 1

        run_id2 = uuid4()
        handler.on_llm_error(RuntimeError("fail again"), run_id=run_id2)
        assert handler.error_count == 2


# ---------------------------------------------------------------------------
# 5. on_tool_start
# ---------------------------------------------------------------------------


class TestOnToolStart:
    def test_records_tool_call(self):
        handler, collector = _make_handler()
        run_id = uuid4()
        handler.on_tool_start(
            {"name": "search_files"},
            '{"query": "test"}',
            run_id=run_id,
        )

        entries = collector.query(event_types=[TimelineEventType.TOOL_CALL])
        assert len(entries) == 1
        meta = entries[0].metadata
        assert meta["tool_name"] == "search_files"
        assert meta["arguments"] == '{"query": "test"}'
        assert handler.tool_call_count == 1


# ---------------------------------------------------------------------------
# 6. on_tool_end
# ---------------------------------------------------------------------------


class TestOnToolEnd:
    def test_records_tool_result_with_latency(self):
        handler, collector = _make_handler()
        run_id = uuid4()
        handler.on_tool_start({"name": "search"}, "args", run_id=run_id)
        time.sleep(0.05)
        handler.on_tool_end("result data", run_id=run_id)

        entries = collector.query(event_types=[TimelineEventType.TOOL_RESULT])
        assert len(entries) == 1
        meta = entries[0].metadata
        assert "latency_ms" in meta
        assert meta["latency_ms"] >= 40
        assert meta["result_preview"] == "result data"


# ---------------------------------------------------------------------------
# 7. on_tool_error
# ---------------------------------------------------------------------------


class TestOnToolError:
    def test_records_tool_error_with_traceback(self):
        handler, collector = _make_handler()
        run_id = uuid4()
        handler.on_tool_start({"name": "read_file"}, "/bad/path", run_id=run_id)

        try:
            raise FileNotFoundError("/bad/path not found")
        except FileNotFoundError as exc:
            handler.on_tool_error(exc, run_id=run_id)

        entries = collector.query(event_types=[TimelineEventType.TOOL_ERROR])
        assert len(entries) == 1
        meta = entries[0].metadata
        assert meta["error_type"] == "FileNotFoundError"
        assert "/bad/path not found" in meta["error"]
        assert "traceback" in meta
        assert handler.error_count == 1


# ---------------------------------------------------------------------------
# 8. on_chain_start
# ---------------------------------------------------------------------------


class TestOnChainStart:
    def test_records_agent_input_for_top_level(self):
        handler, collector = _make_handler(record_prompts=True)
        run_id = uuid4()
        handler.on_chain_start(
            {},
            {"messages": ["Hello agent"]},
            run_id=run_id,
            parent_run_id=None,
        )

        entries = collector.query(event_types=[TimelineEventType.AGENT_INPUT])
        assert len(entries) == 1
        assert "input_preview" in entries[0].metadata

    def test_skips_sub_chains(self):
        handler, collector = _make_handler()
        parent_id = uuid4()
        child_id = uuid4()
        handler.on_chain_start({}, {"input": "x"}, run_id=parent_id, parent_run_id=None)
        handler.on_chain_start({}, {"input": "y"}, run_id=child_id, parent_run_id=parent_id)

        entries = collector.query(event_types=[TimelineEventType.AGENT_INPUT])
        # Only the top-level chain should produce an AGENT_INPUT event
        assert len(entries) == 1


# ---------------------------------------------------------------------------
# 9. on_chain_end
# ---------------------------------------------------------------------------


class TestOnChainEnd:
    def test_records_agent_output_for_top_level_with_totals(self):
        handler, collector = _make_handler()
        run_id = uuid4()
        handler.on_chain_start({}, {"input": "x"}, run_id=run_id, parent_run_id=None)

        # Simulate some LLM activity to populate totals
        llm_run = uuid4()
        handler.on_llm_start({"id": ["x"]}, ["p"], run_id=llm_run)
        result = _make_llm_result(prompt_tokens=50, completion_tokens=30, total_tokens=80)
        handler.on_llm_end(result, run_id=llm_run)

        handler.on_chain_end(
            {"messages": ["output1", "output2"]},
            run_id=run_id,
            parent_run_id=None,
        )

        entries = collector.query(event_types=[TimelineEventType.AGENT_OUTPUT])
        assert len(entries) == 1
        meta = entries[0].metadata
        assert meta["total_tokens"] == 80
        assert meta["total_prompt_tokens"] == 50
        assert meta["total_completion_tokens"] == 30
        assert meta["llm_call_count"] == 1
        assert meta["message_count"] == 2

    def test_skips_sub_chain_end(self):
        handler, collector = _make_handler()
        parent_id = uuid4()
        child_id = uuid4()
        handler.on_chain_start({}, {}, run_id=parent_id, parent_run_id=None)
        handler.on_chain_start({}, {}, run_id=child_id, parent_run_id=parent_id)
        handler.on_chain_end({}, run_id=child_id, parent_run_id=parent_id)

        entries = collector.query(event_types=[TimelineEventType.AGENT_OUTPUT])
        assert len(entries) == 0


# ---------------------------------------------------------------------------
# 10. on_retry
# ---------------------------------------------------------------------------


class TestOnRetry:
    def test_increments_retry_count(self):
        handler, collector = _make_handler()
        run_id = uuid4()

        retry_state = MagicMock()
        retry_state.attempt_number = 2
        retry_state.outcome = None

        handler.on_retry(retry_state, run_id=run_id)

        assert handler.retry_count == 1
        entries = collector.query(event_types=[TimelineEventType.LLM_RETRY])
        assert len(entries) == 1
        assert entries[0].metadata["attempt"] == 2

    def test_extracts_error_from_retry_state(self):
        handler, collector = _make_handler()
        run_id = uuid4()

        outcome = MagicMock()
        outcome.exception.return_value = TimeoutError("API timeout")

        retry_state = MagicMock()
        retry_state.attempt_number = 3
        retry_state.outcome = outcome

        handler.on_retry(retry_state, run_id=run_id)

        entries = collector.query(event_types=[TimelineEventType.LLM_RETRY])
        meta = entries[0].metadata
        assert meta["error_type"] == "TimeoutError"
        assert "API timeout" in meta["error"]


# ---------------------------------------------------------------------------
# 11. Cumulative accounting
# ---------------------------------------------------------------------------


class TestCumulativeAccounting:
    def test_totals_accumulate_across_multiple_calls(self):
        handler, _ = _make_handler()

        for i in range(3):
            run_id = uuid4()
            handler.on_llm_start({"id": ["x"]}, ["p"], run_id=run_id)
            result = _make_llm_result(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            )
            handler.on_llm_end(result, run_id=run_id)

        assert handler.total_prompt_tokens == 300
        assert handler.total_completion_tokens == 150
        assert handler.total_tokens == 450
        assert handler.llm_call_count == 3


# ---------------------------------------------------------------------------
# 12. Privacy — record_prompts
# ---------------------------------------------------------------------------


class TestPrivacy:
    def test_record_prompts_false_hides_prompt_content(self):
        handler, collector = _make_handler(record_prompts=False)
        run_id = uuid4()
        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o"}, "id": ["x"]},
            ["This is a secret prompt"],
            run_id=run_id,
        )
        entries = collector.query(event_types=[TimelineEventType.LLM_START])
        assert "prompt_preview" not in entries[0].metadata

    def test_record_prompts_true_includes_prompt(self):
        handler, collector = _make_handler(record_prompts=True)
        run_id = uuid4()
        handler.on_llm_start(
            {"kwargs": {"model_name": "gpt-4o"}, "id": ["x"]},
            ["This is a visible prompt"],
            run_id=run_id,
        )
        entries = collector.query(event_types=[TimelineEventType.LLM_START])
        assert entries[0].metadata["prompt_preview"] == "This is a visible prompt"

    def test_record_prompts_false_hides_response_preview(self):
        handler, collector = _make_handler(record_prompts=False)
        run_id = uuid4()
        handler.on_llm_start({"id": ["x"]}, ["p"], run_id=run_id)
        result = _make_llm_result(text="Secret response")
        handler.on_llm_end(result, run_id=run_id)

        entries = collector.query(event_types=[TimelineEventType.LLM_END])
        assert "response_preview" not in entries[0].metadata

    def test_chain_start_hides_input_when_record_prompts_false(self):
        handler, collector = _make_handler(record_prompts=False)
        run_id = uuid4()
        handler.on_chain_start(
            {}, {"messages": ["secret input"]}, run_id=run_id, parent_run_id=None
        )
        entries = collector.query(event_types=[TimelineEventType.AGENT_INPUT])
        meta = entries[0].metadata
        assert "input_preview" not in meta
        assert "input_length" in meta


# ---------------------------------------------------------------------------
# 13. get_summary
# ---------------------------------------------------------------------------


class TestGetSummary:
    def test_returns_correct_totals(self):
        handler, _ = _make_handler()

        # Simulate 2 LLM calls
        for _ in range(2):
            run_id = uuid4()
            handler.on_llm_start({"id": ["x"]}, ["p"], run_id=run_id)
            result = _make_llm_result(prompt_tokens=10, completion_tokens=5, total_tokens=15)
            handler.on_llm_end(result, run_id=run_id)

        # Simulate 1 tool call
        tool_run = uuid4()
        handler.on_tool_start({"name": "t"}, "args", run_id=tool_run)
        handler.on_tool_end("result", run_id=tool_run)

        # Simulate 1 error
        err_run = uuid4()
        handler.on_llm_error(RuntimeError("boom"), run_id=err_run)

        # Simulate 1 retry
        retry_run = uuid4()
        retry_state = MagicMock()
        retry_state.attempt_number = 1
        retry_state.outcome = None
        handler.on_retry(retry_state, run_id=retry_run)

        summary = handler.get_summary()
        assert summary["total_prompt_tokens"] == 20
        assert summary["total_completion_tokens"] == 10
        assert summary["total_tokens"] == 30
        assert summary["llm_call_count"] == 2
        assert summary["tool_call_count"] == 1
        assert summary["error_count"] == 1
        assert summary["retry_count"] == 1


# ---------------------------------------------------------------------------
# 14. on_chain_error
# ---------------------------------------------------------------------------


class TestOnChainError:
    def test_records_error_for_top_level_only(self):
        handler, collector = _make_handler()
        top_id = uuid4()
        child_id = uuid4()

        handler.on_chain_start({}, {}, run_id=top_id, parent_run_id=None)
        handler.on_chain_start({}, {}, run_id=child_id, parent_run_id=top_id)

        try:
            raise RuntimeError("child failure")
        except RuntimeError as exc:
            handler.on_chain_error(exc, run_id=child_id, parent_run_id=top_id)

        # A sub-chain error propagates to its parent chain, so it is neither
        # recorded nor counted here — otherwise one failure counts twice.
        assert handler.error_count == 0
        assert collector.query(event_types=[TimelineEventType.AGENT_ERROR]) == []

        try:
            raise RuntimeError("top-level failure")
        except RuntimeError as exc:
            handler.on_chain_error(exc, run_id=top_id, parent_run_id=None)

        assert handler.error_count == 1
        errors_after_top = collector.query(event_types=[TimelineEventType.AGENT_ERROR])
        assert len(errors_after_top) == 1
        assert collector.query(event_types=[TimelineEventType.TOOL_ERROR]) == []
        meta = errors_after_top[0].metadata
        assert meta["error_type"] == "RuntimeError"
        assert "top-level failure" in meta["error"]
        assert "traceback" in meta
