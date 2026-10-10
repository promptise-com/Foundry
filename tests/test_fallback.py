"""Tests for promptise.fallback — Model FallbackChain."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

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
