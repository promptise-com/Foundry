"""Tests for promptise.fallback — Model FallbackChain."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from promptise.fallback import FallbackChain, _CircuitState

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(text: str = "Hello") -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])


class FakeModel:
    """Minimal BaseChatModel mock for testing."""

    def __init__(self, name: str = "fake", fail: bool = False, delay: float = 0):
        self.model_name = name
        self._fail = fail
        self._delay = delay
        self._call_count = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._call_count += 1
        if self._fail:
            raise RuntimeError(f"{self.model_name} is down")
        return _make_result(f"Response from {self.model_name}")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self._call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail:
            raise RuntimeError(f"{self.model_name} is down")
        return _make_result(f"Response from {self.model_name}")


# ---------------------------------------------------------------------------
# CircuitState
# ---------------------------------------------------------------------------


class TestCircuitState:
    def test_initial_state_closed(self):
        cs = _CircuitState(model_id="test")
        assert cs.state == "closed"
        assert cs.should_skip() is False

    def test_opens_after_threshold(self):
        cs = _CircuitState(model_id="test", failure_threshold=2)
        cs.record_failure()
        assert cs.state == "closed"
        cs.record_failure()
        assert cs.state == "open"
        assert cs.should_skip() is True

    def test_success_resets(self):
        cs = _CircuitState(model_id="test", failure_threshold=2)
        cs.record_failure()
        cs.record_failure()
        assert cs.state == "open"
        cs.record_success()
        assert cs.state == "closed"
        assert cs.failures == 0

    def test_recovery_after_timeout(self):
        # Windows' monotonic clock ticks in ~15 ms steps, so a 10 ms timeout can
        # read as elapsed on the very next line; keep both well above a tick.
        cs = _CircuitState(model_id="test", failure_threshold=1, recovery_timeout=0.2)
        cs.record_failure()
        assert cs.state == "open"
        assert cs.should_skip() is True
        # Wait for recovery
        import time

        time.sleep(0.3)
        assert cs.should_skip() is False
        assert cs.state == "half_open"


# ---------------------------------------------------------------------------
# FallbackChain construction
# ---------------------------------------------------------------------------


class TestFallbackChainConstruction:
    def test_requires_at_least_one_model(self):
        with pytest.raises(ValueError, match="at least one"):
            FallbackChain([])

    def test_accepts_model_strings(self):
        chain = FallbackChain(["openai:gpt-5-mini", "anthropic:claude-sonnet-4-20250514"])
        assert len(chain.models) == 2

    def test_accepts_model_instances(self):
        m1 = FakeModel("model-a")
        m2 = FakeModel("model-b")
        chain = FallbackChain([m1, m2])
        assert len(chain.models) == 2

    def test_llm_type(self):
        chain = FallbackChain([FakeModel("test")])
        assert chain._llm_type == "fallback-chain"

    def test_model_name_returns_primary(self):
        chain = FallbackChain([FakeModel("primary"), FakeModel("fallback")])
        assert chain.model_name == "primary"


# ---------------------------------------------------------------------------
# Sync fallback
# ---------------------------------------------------------------------------


class TestSyncFallback:
    def test_primary_succeeds(self):
        m1 = FakeModel("primary")
        m2 = FakeModel("fallback")
        chain = FallbackChain([m1, m2])
        result = chain._generate([HumanMessage(content="Hi")])
        assert "primary" in result.generations[0].text
        assert m1._call_count == 1
        assert m2._call_count == 0

    def test_primary_fails_fallback_succeeds(self):
        m1 = FakeModel("primary", fail=True)
        m2 = FakeModel("fallback")
        chain = FallbackChain([m1, m2])
        result = chain._generate([HumanMessage(content="Hi")])
        assert "fallback" in result.generations[0].text
        assert m1._call_count == 1
        assert m2._call_count == 1

    def test_all_fail_raises_runtime_error(self):
        m1 = FakeModel("a", fail=True)
        m2 = FakeModel("b", fail=True)
        chain = FallbackChain([m1, m2])
        with pytest.raises(RuntimeError, match="All 2 models") as exc_info:
            chain._generate([HumanMessage(content="Hi")])
        # Error message lists each model and its error
        assert "a: RuntimeError" in str(exc_info.value)
        assert "b: RuntimeError" in str(exc_info.value)

    def test_circuit_breaker_skips_broken(self):
        m1 = FakeModel("primary", fail=True)
        m2 = FakeModel("fallback")
        chain = FallbackChain([m1, m2], failure_threshold=1)

        # First call: primary fails, falls to fallback
        chain._generate([HumanMessage(content="Hi")])
        assert m1._call_count == 1

        # Second call: circuit open for primary, goes straight to fallback
        m1._call_count = 0
        chain._generate([HumanMessage(content="Hi")])
        assert m1._call_count == 0  # Skipped!
        assert m2._call_count == 2

    def test_on_fallback_callback(self):
        calls = []
        m1 = FakeModel("primary", fail=True)
        m2 = FakeModel("fallback")
        chain = FallbackChain(
            [m1, m2],
            on_fallback=lambda pri, fb, err: calls.append((pri, fb)),
        )
        chain._generate([HumanMessage(content="Hi")])
        assert calls == [("primary", "fallback")]


# ---------------------------------------------------------------------------
# Async fallback
# ---------------------------------------------------------------------------


class TestAsyncFallback:
    @pytest.mark.asyncio
    async def test_primary_succeeds(self):
        m1 = FakeModel("primary")
        chain = FallbackChain([m1])
        result = await chain._agenerate([HumanMessage(content="Hi")])
        assert "primary" in result.generations[0].text

    @pytest.mark.asyncio
    async def test_primary_fails_fallback_succeeds(self):
        m1 = FakeModel("primary", fail=True)
        m2 = FakeModel("fallback")
        chain = FallbackChain([m1, m2])
        result = await chain._agenerate([HumanMessage(content="Hi")])
        assert "fallback" in result.generations[0].text

    @pytest.mark.asyncio
    async def test_timeout_per_model(self):
        m1 = FakeModel("slow", delay=5.0)  # Will timeout
        m2 = FakeModel("fast")
        chain = FallbackChain([m1, m2], timeout_per_model=0.1)
        result = await chain._agenerate([HumanMessage(content="Hi")])
        assert "fast" in result.generations[0].text

    @pytest.mark.asyncio
    async def test_global_timeout(self):
        m1 = FakeModel("slow1", delay=5.0)
        m2 = FakeModel("slow2", delay=5.0)
        chain = FallbackChain([m1, m2], global_timeout=0.1)
        with pytest.raises(RuntimeError, match="All 2 models"):
            await chain._agenerate([HumanMessage(content="Hi")])

    @pytest.mark.asyncio
    async def test_all_fail_async(self):
        m1 = FakeModel("a", fail=True)
        m2 = FakeModel("b", fail=True)
        chain = FallbackChain([m1, m2])
        with pytest.raises(RuntimeError, match="All 2 models"):
            await chain._agenerate([HumanMessage(content="Hi")])


# ---------------------------------------------------------------------------
# Chain status
# ---------------------------------------------------------------------------


class TestChainStatus:
    def test_initial_status(self):
        chain = FallbackChain([FakeModel("a"), FakeModel("b")])
        status = chain.get_chain_status()
        assert len(status) == 2
        assert status[0]["model_id"] == "a"
        assert status[0]["state"] == "closed"
        assert status[0]["is_primary"] is True
        assert status[1]["is_primary"] is False

    def test_status_after_failure(self):
        m1 = FakeModel("primary", fail=True)
        m2 = FakeModel("fallback")
        chain = FallbackChain([m1, m2], failure_threshold=1)
        chain._generate([HumanMessage(content="Hi")])
        status = chain.get_chain_status()
        assert status[0]["state"] == "open"
        assert status[0]["failures"] == 1
        assert status[1]["state"] == "closed"


# ---------------------------------------------------------------------------
# Active model
# ---------------------------------------------------------------------------


class TestActiveModel:
    def test_returns_primary_when_healthy(self):
        chain = FallbackChain([FakeModel("primary"), FakeModel("fallback")])
        assert chain.active_model == "primary"

    def test_returns_fallback_when_primary_tripped(self):
        m1 = FakeModel("primary", fail=True)
        m2 = FakeModel("fallback")
        chain = FallbackChain([m1, m2], failure_threshold=1)
        chain._generate([HumanMessage(content="Hi")])
        assert chain.active_model == "fallback"


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


class TestServingModelTracking:
    def test_tracks_primary_on_success(self):
        m1 = FakeModel("primary")
        chain = FallbackChain([m1, FakeModel("fallback")])
        chain._generate([HumanMessage(content="Hi")])
        assert chain.model_name == "primary"
        assert chain._last_serving_model == "primary"

    def test_tracks_fallback_on_primary_failure(self):
        m1 = FakeModel("primary", fail=True)
        m2 = FakeModel("fallback")
        chain = FallbackChain([m1, m2])
        chain._generate([HumanMessage(content="Hi")])
        assert chain.model_name == "fallback"
        assert chain._last_serving_model == "fallback"

    @pytest.mark.asyncio
    async def test_tracks_fallback_async(self):
        m1 = FakeModel("primary", fail=True)
        m2 = FakeModel("fallback")
        chain = FallbackChain([m1, m2])
        await chain._agenerate([HumanMessage(content="Hi")])
        assert chain.model_name == "fallback"


class TestExports:
    def test_importable_from_promptise(self):
        from promptise import FallbackChain

        assert FallbackChain is not None


# ---------------------------------------------------------------------------
# Tool binding
# ---------------------------------------------------------------------------


class ToolModel(BaseChatModel):
    """Chat model that supports bind_tools and records what it was called with."""

    model_name: str = "tool-model"
    fail: bool = False
    seen: Any = None  # a shared list; typed Any so pydantic does not copy it

    @property
    def _llm_type(self) -> str:
        return "tool-model"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=[getattr(t, "name", t) for t in tools], **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append((self.model_name, kwargs))
        if self.fail:
            raise RuntimeError(f"{self.model_name} is down")
        return _make_result(f"Response from {self.model_name}")

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._generate(messages, stop, run_manager, **kwargs)


class NoToolsModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "no-tools"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return _make_result("plain")


@tool
def lookup(order_id: str) -> str:
    """Look up an order."""
    return f"order {order_id}"


class TestBindTools:
    @pytest.mark.asyncio
    async def test_tools_reach_every_model_in_the_chain(self):
        seen: list = []
        primary = ToolModel(model_name="primary", fail=True, seen=seen)
        backup = ToolModel(model_name="backup", seen=seen)
        chain = FallbackChain([primary, backup])

        bound = chain.bind_tools([lookup], tool_choice="auto")
        result = await bound.ainvoke([HumanMessage(content="Where is order 7?")])

        assert result.content == "Response from backup"
        assert seen == [
            ("primary", {"tools": ["lookup"], "tool_choice": "auto"}),
            ("backup", {"tools": ["lookup"], "tool_choice": "auto"}),
        ]

    def test_sync_path_uses_the_bound_tools(self):
        seen: list = []
        chain = FallbackChain([ToolModel(model_name="primary", seen=seen)])
        chain.bind_tools([lookup]).invoke([HumanMessage(content="Hi")])
        assert seen == [("primary", {"tools": ["lookup"]})]

    def test_returns_a_fallback_chain_and_leaves_the_original_unbound(self):
        seen: list = []
        chain = FallbackChain([ToolModel(model_name="primary", seen=seen)])
        bound = chain.bind_tools([lookup])
        assert isinstance(bound, FallbackChain) and bound is not chain

        chain.invoke([HumanMessage(content="Hi")])
        assert seen == [("primary", {})]

    @pytest.mark.asyncio
    async def test_bound_chain_shares_circuit_breakers_and_serving_model(self):
        primary = ToolModel(model_name="primary", fail=True, seen=[])
        backup = ToolModel(model_name="backup", seen=[])
        chain = FallbackChain([primary, backup], failure_threshold=2)
        bound = chain.bind_tools([lookup])

        for _ in range(2):
            await bound.ainvoke([HumanMessage(content="Hi")])

        assert chain.get_chain_status()[0]["state"] == "open"
        assert chain.model_name == "backup"
        # A later binding (the engine binds on every step) sees the open circuit
        assert chain.bind_tools([lookup]).active_model == "backup"

    @pytest.mark.asyncio
    async def test_answer_records_the_serving_model(self):
        chain = FallbackChain(
            [
                ToolModel(model_name="primary", fail=True, seen=[]),
                ToolModel(model_name="backup", seen=[]),
            ]
        )
        result = await chain.bind_tools([lookup]).ainvoke([HumanMessage(content="Hi")])
        assert result.response_metadata["fallback_model"] == "backup"

    def test_model_without_tool_support_is_named(self):
        chain = FallbackChain([ToolModel(model_name="primary", seen=[]), NoToolsModel()])
        with pytest.raises(NotImplementedError, match="NoToolsModel"):
            chain.bind_tools([lookup])

    @pytest.mark.asyncio
    async def test_binding_that_is_not_plain_kwargs_is_invoked(self):
        from langchain_core.runnables import RunnableLambda

        class Wrapped(ToolModel):
            def bind_tools(self, tools, **kwargs):
                return RunnableLambda(lambda messages: AIMessage(content=f"wrapped {len(tools)}"))

        chain = FallbackChain([Wrapped(model_name="w", seen=[])])
        result = await chain.bind_tools([lookup]).ainvoke([HumanMessage(content="Hi")])
        assert result.content == "wrapped 1"
        assert result.response_metadata["fallback_model"] == "w"

    @pytest.mark.asyncio
    async def test_build_agent_with_tools(self):
        """The docs' own setup: a FallbackChain driving an agent with tools."""
        from promptise import build_agent

        class CallsLookup(ToolModel):
            async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
                if self.fail:
                    raise RuntimeError("down")
                if "tools" in kwargs and not any(m.type == "tool" for m in messages):
                    call = {"name": "lookup", "args": {"order_id": "7"}, "id": "c1"}
                    return ChatResult(
                        generations=[
                            ChatGeneration(message=AIMessage(content="", tool_calls=[call]))
                        ]
                    )
                return _make_result(f"{self.model_name}: {messages[-1].content}")

        chain = FallbackChain(
            [CallsLookup(model_name="primary", fail=True), CallsLookup(model_name="backup")]
        )
        agent = await build_agent(model=chain, servers={}, extra_tools=[lookup])
        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "Where is order 7?"}]}
            )
        finally:
            await agent.shutdown()
        assert result["messages"][-1].content == "backup: order 7"


# ---------------------------------------------------------------------------
# Which errors count toward the circuit breaker
# ---------------------------------------------------------------------------


class _StatusError(Exception):
    """Shaped like the OpenAI / Anthropic SDK errors: carries status_code."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


class _HTTPStatusError(Exception):
    """Shaped like httpx.HTTPStatusError: the status is on .response."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.response = _Response(status_code)


class RaisingModel(FakeModel):
    def __init__(self, name: str, exc: Exception) -> None:
        super().__init__(name)
        self._exc = exc

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._call_count += 1
        raise self._exc

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        self._call_count += 1
        raise self._exc


class TestErrorClassification:
    @pytest.mark.asyncio
    async def test_rejected_request_falls_back_but_does_not_trip_the_circuit(self):
        """One oversized prompt must not take the primary out for everyone."""
        primary = RaisingModel("primary", _StatusError(400))
        backup = FakeModel("backup")
        chain = FallbackChain([primary, backup], failure_threshold=1)
        chain._ensure_resolved()
        chain._resolved = [primary, backup]

        result = await chain._agenerate([HumanMessage(content="x" * 10_000)])

        assert result.generations[0].message.content == "Response from backup"
        assert chain.get_chain_status()[0]["state"] == "closed"
        assert chain.active_model == "primary"

    @pytest.mark.parametrize("exc", [_StatusError(413), _HTTPStatusError(422)])
    def test_other_request_errors_do_not_trip_the_circuit_either(self, exc):
        chain = FallbackChain([RaisingModel("primary", exc), FakeModel("backup")])
        chain._ensure_resolved()
        chain._resolved = [RaisingModel("primary", exc), FakeModel("backup")]
        chain.failure_threshold = 1
        for c in chain._circuits:
            c.failure_threshold = 1
        chain._generate([HumanMessage(content="hi")])
        assert chain.get_chain_status()[0]["state"] == "closed"

    @pytest.mark.parametrize(
        "exc",
        [_StatusError(429), _StatusError(503), _StatusError(401), TimeoutError(), RuntimeError()],
    )
    @pytest.mark.asyncio
    async def test_provider_failures_trip_the_circuit(self, exc):
        primary = RaisingModel("primary", exc)
        chain = FallbackChain([primary, FakeModel("backup")], failure_threshold=1)
        chain._ensure_resolved()
        chain._resolved = [primary, FakeModel("backup")]
        await chain._agenerate([HumanMessage(content="hi")])
        assert chain.get_chain_status()[0]["state"] == "open"

    @pytest.mark.asyncio
    async def test_all_failed_error_keeps_the_last_cause(self):
        last = _StatusError(503)
        chain = FallbackChain([RaisingModel("a", RuntimeError("a down")), RaisingModel("b", last)])
        chain._ensure_resolved()
        chain._resolved = [RaisingModel("a", RuntimeError("a down")), RaisingModel("b", last)]
        with pytest.raises(RuntimeError, match="All 2 models") as info:
            await chain._agenerate([HumanMessage(content="hi")])
        assert info.value.__cause__ is last


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class StreamingModel(BaseChatModel):
    """Streams its answer word by word; can fail before or after the first word."""

    model_name: str = "streaming"
    fail_before: bool = False
    fail_after: bool = False

    @property
    def _llm_type(self) -> str:
        return "streaming"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.fail_before or self.fail_after:
            raise RuntimeError(f"{self.model_name} is down")
        return _make_result(f"Hello from {self.model_name}")

    async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.messages import AIMessageChunk
        from langchain_core.outputs import ChatGenerationChunk

        if self.fail_before:
            raise RuntimeError(f"{self.model_name} is down")
        for n, word in enumerate(["Hello", " from", f" {self.model_name}"]):
            if self.fail_after and n == 1:
                raise RuntimeError("connection dropped mid-answer")
            yield ChatGenerationChunk(message=AIMessageChunk(content=word))


class TestStreaming:
    @pytest.mark.asyncio
    async def test_streams_token_by_token_from_the_fallback(self):
        chain = FallbackChain(
            [
                StreamingModel(model_name="primary", fail_before=True),
                StreamingModel(model_name="backup"),
            ]
        )
        chunks = [c async for c in chain.astream([HumanMessage(content="Hi")])]
        assert [c.content for c in chunks if c.content] == ["Hello", " from", " backup"]
        assert chain.model_name == "backup"
        merged = chunks[0]
        for c in chunks[1:]:
            merged = merged + c
        assert merged.response_metadata["fallback_model"] == "backup"

    @pytest.mark.asyncio
    async def test_failure_after_the_first_chunk_is_raised_not_spliced(self):
        backup = StreamingModel(model_name="backup")
        chain = FallbackChain([StreamingModel(model_name="primary", fail_after=True), backup])
        seen: list[str] = []
        with pytest.raises(RuntimeError, match="mid-answer"):
            async for chunk in chain.astream([HumanMessage(content="Hi")]):
                seen.append(chunk.content)
        assert seen == ["Hello"]  # no " backup" glued onto the primary's words
        assert chain.get_chain_status()[0]["failures"] == 1

    @pytest.mark.asyncio
    async def test_non_streaming_member_answers_in_one_chunk_with_tool_calls(self):
        class ToolCaller(BaseChatModel):
            @property
            def _llm_type(self) -> str:
                return "tool-caller"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                call = {"name": "lookup", "args": {"order_id": "7"}, "id": "c1"}
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))]
                )

        chain = FallbackChain(
            [StreamingModel(model_name="primary", fail_before=True), ToolCaller()]
        )
        chunks = [c async for c in chain.astream([HumanMessage(content="Where is order 7?")])]
        merged = chunks[0]
        for c in chunks[1:]:
            merged = merged + c
        assert [(t["name"], t["args"]) for t in merged.tool_calls] == [
            ("lookup", {"order_id": "7"})
        ]

    @pytest.mark.asyncio
    async def test_first_chunk_timeout_falls_back(self):
        class Slow(StreamingModel):
            async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
                await asyncio.sleep(5)
                async for chunk in super()._astream(messages, stop, run_manager, **kwargs):
                    yield chunk

        chain = FallbackChain(
            [Slow(model_name="slow"), StreamingModel(model_name="backup")], timeout_per_model=0.05
        )
        chunks = [c async for c in chain.astream([HumanMessage(content="Hi")])]
        assert "".join(str(c.content) for c in chunks) == "Hello from backup"

    @pytest.mark.asyncio
    async def test_agent_streams_the_answer_through_a_fallback_chain(self):
        from promptise import build_agent
        from promptise.streaming import DoneEvent, TokenEvent

        chain = FallbackChain(
            [
                StreamingModel(model_name="primary", fail_before=True),
                StreamingModel(model_name="backup"),
            ]
        )
        agent = await build_agent(model=chain, servers={})
        try:
            events = [
                e
                async for e in agent.astream_with_tools(
                    {"messages": [{"role": "user", "content": "Hi"}]}
                )
            ]
        finally:
            await agent.shutdown()
        tokens = [e.text for e in events if isinstance(e, TokenEvent)]
        done = [e for e in events if isinstance(e, DoneEvent)]
        assert tokens == ["Hello", " from", " backup"]
        assert done and done[0].full_response == "Hello from backup"
