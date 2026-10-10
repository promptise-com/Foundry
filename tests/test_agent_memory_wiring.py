"""How PromptiseAgent wires memory: caller identity, settings, auto-store, chat().

Covers:

* ``chat(user_id=...)`` is shorthand for ``caller=CallerContext(user_id=...)``
  — a per-user memory provider is searched for that user.  It used to check
  session ownership with the user id but invoke the agent with no caller, so
  per-user memory raised ``MemoryIsolationError`` (logged as "Memory search
  failed") and the agent forgot everything.
* ``build_agent()`` exposes the memory search limit, minimum score and timeout.
* A slow auto-store is not reported as failed: the provider write keeps
  running in its worker thread after the timeout, so it is never cancelled,
  its real outcome is logged, and ``shutdown()`` waits for it.
* ``chat()`` fails closed when the conversation store fails during the
  ownership check, and still answers when only loading or saving history fails.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, SystemMessage

from promptise.agent import CallerContext, PromptiseAgent, build_agent
from promptise.conversations import InMemoryConversationStore, SessionAccessDenied
from promptise.memory import InMemoryProvider, MemoryScope


class _RecordingInner:
    """Fake agent graph: remembers the messages it was invoked with."""

    def __init__(self, reply: str = "ok") -> None:
        self.reply = reply
        self.inputs: list[Any] = []

    async def ainvoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        self.inputs.append(input)
        return {"messages": [*input["messages"], AIMessage(content=self.reply)]}

    def memory_context(self) -> str:
        """The injected memory system message of the last invocation ('' if none)."""
        for msg in self.inputs[-1]["messages"]:
            if isinstance(msg, SystemMessage) and "<memory_context>" in str(msg.content):
                return str(msg.content)
        return ""


async def _per_user_memory() -> InMemoryProvider:
    memory = InMemoryProvider(scope=MemoryScope.PER_USER)
    await memory.add("vegetarian", user_id="mara")
    await memory.add("vegetarian bob likes steak", user_id="bob")
    return memory


# ---------------------------------------------------------------------------
# chat(user_id=...) scopes memory to that user
# ---------------------------------------------------------------------------


class TestChatUserIdScopesMemory:
    @pytest.mark.asyncio
    async def test_user_id_shorthand_searches_that_users_memory(self, caplog) -> None:
        inner = _RecordingInner()
        agent = PromptiseAgent(
            inner=inner,
            memory_provider=await _per_user_memory(),
            conversation_store=InMemoryConversationStore(),
        )
        with caplog.at_level(logging.WARNING, logger="promptise.agent"):
            await agent.chat("vegetarian", session_id="s1", user_id="mara")
        assert "Memory search failed" not in caplog.text
        context = inner.memory_context()
        assert "- vegetarian" in context
        assert "steak" not in context  # bob's memory stays bob's

    @pytest.mark.asyncio
    async def test_user_id_shorthand_equals_caller(self) -> None:
        by_id, by_caller = _RecordingInner(), _RecordingInner()
        memory = await _per_user_memory()
        for inner, kwargs in (
            (by_id, {"user_id": "mara"}),
            (by_caller, {"caller": CallerContext(user_id="mara")}),
        ):
            agent = PromptiseAgent(inner=inner, memory_provider=memory)
            await agent.chat("vegetarian", session_id="s-" + next(iter(kwargs)), **kwargs)
        assert by_id.memory_context() == by_caller.memory_context() != ""

    @pytest.mark.asyncio
    async def test_caller_without_user_id_gets_the_user_id(self) -> None:
        inner = _RecordingInner()
        agent = PromptiseAgent(inner=inner, memory_provider=await _per_user_memory())
        await agent.chat(
            "vegetarian",
            session_id="s1",
            user_id="mara",
            caller=CallerContext(roles={"analyst"}),
        )
        assert "- vegetarian" in inner.memory_context()

    @pytest.mark.asyncio
    async def test_caller_user_id_takes_precedence(self) -> None:
        inner = _RecordingInner()
        agent = PromptiseAgent(inner=inner, memory_provider=await _per_user_memory())
        await agent.chat(
            "steak", session_id="s1", user_id="mara", caller=CallerContext(user_id="bob")
        )
        assert "steak" in inner.memory_context()

    @pytest.mark.asyncio
    async def test_user_id_shorthand_owns_the_session(self) -> None:
        store = InMemoryConversationStore()
        agent = PromptiseAgent(inner=_RecordingInner(), conversation_store=store)
        await agent.chat("hi", session_id="s1", user_id="mara")
        with pytest.raises(SessionAccessDenied):
            await agent.chat("hi", session_id="s1", user_id="bob")
        await agent.chat("again", session_id="s1", caller=CallerContext(user_id="mara"))

    @pytest.mark.asyncio
    async def test_user_id_reaches_ainvoke_caller(self) -> None:
        agent = PromptiseAgent(inner=_RecordingInner())
        seen: list[Any] = []
        original = agent.ainvoke

        async def spy(input: Any, config: Any = None, *, caller: Any = None, **kw: Any) -> Any:
            seen.append(caller)
            return await original(input, config, caller=caller, **kw)

        agent.ainvoke = spy  # type: ignore[method-assign]
        await agent.chat("hi", session_id="s1", user_id="mara")
        assert isinstance(seen[0], CallerContext)
        assert seen[0].user_id == "mara"


# ---------------------------------------------------------------------------
# build_agent() memory settings
# ---------------------------------------------------------------------------


def _mock_engine() -> MagicMock:
    inner = MagicMock()
    inner.ainvoke = AsyncMock(return_value={"messages": []})
    return inner


async def _build(**kwargs: Any) -> PromptiseAgent:
    with (
        patch("promptise.agent._normalize_model", return_value=MagicMock()),
        patch("promptise.agent.PromptGraphEngine", return_value=_mock_engine()),
    ):
        return await build_agent(servers={}, model="openai:gpt-5-mini", **kwargs)


class TestBuildAgentMemorySettings:
    @pytest.mark.asyncio
    async def test_defaults(self) -> None:
        agent = await _build(memory=InMemoryProvider())
        assert (agent._memory_max, agent._memory_min_score, agent._memory_timeout) == (5, 0.0, 5.0)

    @pytest.mark.asyncio
    async def test_settings_are_passed_through(self) -> None:
        agent = await _build(
            memory=InMemoryProvider(),
            memory_max_results=3,
            memory_min_score=0.4,
            memory_timeout=30.0,
        )
        assert (agent._memory_max, agent._memory_min_score, agent._memory_timeout) == (3, 0.4, 30.0)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"memory_max_results": 0},
            {"memory_min_score": 1.5},
            {"memory_min_score": -0.1},
            {"memory_timeout": 0},
        ],
    )
    @pytest.mark.asyncio
    async def test_invalid_settings_raise(self, kwargs: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            await _build(memory=InMemoryProvider(), **kwargs)

    @pytest.mark.asyncio
    async def test_max_results_and_min_score_apply(self) -> None:
        provider = MagicMock()
        provider.scope = MemoryScope.SHARED
        from promptise.memory import MemoryResult

        provider.search = AsyncMock(
            return_value=[
                MemoryResult(content="relevant", score=0.8, memory_id="a"),
                MemoryResult(content="noise", score=0.1, memory_id="b"),
            ]
        )
        agent = PromptiseAgent(
            inner=_RecordingInner(), memory_provider=provider, memory_max=2, memory_min_score=0.5
        )
        results = await agent._search_memory("query")
        assert provider.search.await_args.kwargs["limit"] == 2
        assert [r.content for r in results] == ["relevant"]


# ---------------------------------------------------------------------------
# Auto-store: a slow write is not reported as failed
# ---------------------------------------------------------------------------


class _SlowProvider(InMemoryProvider):
    """Writes land after ``delay`` seconds, like a cold Chroma embedding model."""

    def __init__(self, delay: float, fail: bool = False) -> None:
        super().__init__()
        self.delay = delay
        self.fail = fail

    async def add(self, content: str, **kwargs: Any) -> str:
        await asyncio.sleep(self.delay)
        if self.fail:
            raise ConnectionError("store unreachable")
        return await super().add(content, **kwargs)


class TestAutoStoreTimeout:
    @pytest.mark.asyncio
    async def test_slow_write_is_not_logged_as_failed(self, caplog) -> None:
        provider = _SlowProvider(delay=0.3)
        agent = PromptiseAgent(
            inner=_RecordingInner("noted"),
            memory_provider=provider,
            memory_auto_store=True,
            memory_timeout=0.05,
        )
        caplog.set_level(logging.INFO, logger="promptise.memory")
        await agent.ainvoke({"messages": [{"role": "user", "content": "I am vegetarian"}]})

        assert "Memory auto-store failed" not in caplog.text
        assert "has not finished after 0.1s" in caplog.text
        assert len(agent._pending_memory_writes) == 1

        await agent.shutdown()  # waits for the write before closing the provider
        assert "Memory auto-store completed after" in caplog.text
        assert "Memory auto-store failed" not in caplog.text
        assert not agent._pending_memory_writes

    @pytest.mark.asyncio
    async def test_slow_write_lands(self) -> None:
        provider = _SlowProvider(delay=0.2)
        agent = PromptiseAgent(
            inner=_RecordingInner("noted"),
            memory_provider=provider,
            memory_auto_store=True,
            memory_timeout=0.02,
        )
        await agent.ainvoke({"messages": [{"role": "user", "content": "I am vegetarian"}]})
        assert provider._store == {}
        await asyncio.wait(agent._pending_memory_writes)
        stored = [row[0] for row in provider._store.values()]
        assert stored == ["User: I am vegetarian\nAssistant: noted"]

    @pytest.mark.asyncio
    async def test_failed_write_is_logged_as_failed(self, caplog) -> None:
        provider = _SlowProvider(delay=0.0, fail=True)
        agent = PromptiseAgent(
            inner=_RecordingInner(),
            memory_provider=provider,
            memory_auto_store=True,
            memory_timeout=1.0,
        )
        await agent.ainvoke({"messages": [{"role": "user", "content": "hello"}]})
        await asyncio.sleep(0)  # let the done-callback run
        assert "Memory auto-store failed" in caplog.text

    @pytest.mark.asyncio
    async def test_late_failure_is_logged_once_it_happens(self, caplog) -> None:
        provider = _SlowProvider(delay=0.1, fail=True)
        agent = PromptiseAgent(
            inner=_RecordingInner(),
            memory_provider=provider,
            memory_auto_store=True,
            memory_timeout=0.01,
        )
        await agent.ainvoke({"messages": [{"role": "user", "content": "hello"}]})
        assert "Memory auto-store failed" not in caplog.text
        await agent.shutdown()
        assert "Memory auto-store failed" in caplog.text

    @pytest.mark.asyncio
    async def test_search_timeout_message(self, caplog) -> None:
        provider = MagicMock()

        async def slow_search(*args: Any, **kwargs: Any) -> list[Any]:
            await asyncio.sleep(1)
            return []

        provider.search = slow_search
        agent = PromptiseAgent(
            inner=_RecordingInner(), memory_provider=provider, memory_timeout=0.01
        )
        assert await agent._search_memory("query") == []
        assert "continuing without memory context" in caplog.text


class TestStreamingAutoStore:
    @staticmethod
    def _streaming_inner(text: str) -> MagicMock:
        inner = MagicMock()

        async def astream_events(*args: Any, **kwargs: Any) -> Any:
            chunk = MagicMock()
            chunk.content = text
            yield {"event": "on_chat_model_stream", "data": {"chunk": chunk}}

        inner.astream_events = astream_events
        return inner

    @pytest.mark.asyncio
    async def test_stream_respects_memory_auto_store_off(self) -> None:
        provider = InMemoryProvider()
        agent = PromptiseAgent(inner=self._streaming_inner("hi"), memory_provider=provider)
        events = [
            e
            async for e in agent.astream_with_tools(
                {"messages": [{"role": "user", "content": "hello"}]}
            )
        ]
        assert events[-1].type == "done"
        assert provider._store == {}

    @pytest.mark.asyncio
    async def test_stream_stores_for_the_caller(self) -> None:
        provider = InMemoryProvider(scope=MemoryScope.PER_USER)
        agent = PromptiseAgent(
            inner=self._streaming_inner("hi there"),
            memory_provider=provider,
            memory_auto_store=True,
        )
        async for _ in agent.astream_with_tools(
            {"messages": [{"role": "user", "content": "hello"}]},
            caller=CallerContext(user_id="mara"),
        ):
            pass
        rows = list(provider._store.values())
        assert [(r[0], r[3]) for r in rows] == [("User: hello\nAssistant: hi there", "mara")]

    @pytest.mark.asyncio
    async def test_stream_stores_the_redacted_reply(self) -> None:
        """What the output guardrails removed from the reply never reaches memory."""
        guardrails = MagicMock()
        guardrails.check_input = AsyncMock(return_value=None)
        guardrails.check_output = AsyncMock(
            side_effect=lambda text: text.replace("4111 1111 1111 1111", "[CARD]")
        )
        provider = InMemoryProvider()  # shared scope: 1.2.1 stored here too
        agent = PromptiseAgent(
            inner=self._streaming_inner("Your card 4111 1111 1111 1111 is on file"),
            memory_provider=provider,
            memory_auto_store=True,
            guardrails=guardrails,
        )
        events = [
            e
            async for e in agent.astream_with_tools(
                {"messages": [{"role": "user", "content": "which card?"}]},
                caller=CallerContext(user_id="mara"),
            )
        ]
        assert events[-1].full_response == "Your card [CARD] is on file"
        [row] = provider._store.values()
        assert row[0] == "User: which card?\nAssistant: Your card [CARD] is on file"
        assert "4111" not in row[0]


# ---------------------------------------------------------------------------
# chat() and conversation-store failures (documented behavior)
# ---------------------------------------------------------------------------


class TestChatStoreFailures:
    @pytest.mark.asyncio
    async def test_store_error_during_ownership_check_fails_closed(self) -> None:
        store = InMemoryConversationStore()
        store.get_session = AsyncMock(side_effect=ConnectionError("db down"))  # type: ignore[method-assign]
        inner = _RecordingInner()
        agent = PromptiseAgent(inner=inner, conversation_store=store)
        with pytest.raises(RuntimeError, match="Cannot verify session ownership"):
            await agent.chat("hi", session_id="s1", user_id="mara")
        assert inner.inputs == []  # the model was never called

    @pytest.mark.asyncio
    async def test_load_and_save_errors_still_answer(self, caplog) -> None:
        store = InMemoryConversationStore()
        store.load_messages = AsyncMock(side_effect=ConnectionError("db down"))  # type: ignore[method-assign]
        store.save_messages = AsyncMock(side_effect=ConnectionError("db down"))  # type: ignore[method-assign]
        agent = PromptiseAgent(inner=_RecordingInner("answer"), conversation_store=store)
        assert await agent.chat("hi", session_id="s1", user_id="mara") == "answer"
        assert "Failed to load conversation history" in caplog.text
        assert "Failed to save conversation history" in caplog.text
