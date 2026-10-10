"""SemanticCache wired into a real agent: conversations, tool turns, writes.

Every test builds an agent with ``build_agent()`` and a scripted chat model,
so the cache is exercised through ``PromptiseAgent.ainvoke()`` exactly as an
application uses it.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool

from promptise import CallerContext, FallbackChain, SemanticCache, build_agent
from promptise.observability import TimelineEventType

ALICE = CallerContext(user_id="alice")


class ExactEmbeddings:
    """Identical text → similarity 1.0; different text → near 0."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            vec = np.random.RandomState(seed).randn(64)
            out.append((vec / np.linalg.norm(vec)).tolist())
        return out


def _result(message: AIMessage) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=message)])


class SupportModel(BaseChatModel):
    """Scripted support agent.

    - "How many ..." → calls ``count_open_tickets``
    - "Open a ticket ..." → calls ``open_ticket``
    - after a tool result → answers with it
    - "What river ..." → answers from the city named in the first user turn
    - anything else → echoes the question
    """

    model_name: str = "support-model"
    fail: bool = False
    fail_after_tool: bool = False
    calls: Any = None  # shared list of call counts; Any so pydantic keeps the object

    @property
    def _llm_type(self) -> str:
        return "support-model"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self.bind(tools=[t.name for t in tools], **kwargs)

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self.calls is not None:
            self.calls.append(self.model_name)
        if self.fail:
            raise RuntimeError(f"{self.model_name} is down")
        last = messages[-1]
        if isinstance(last, ToolMessage):
            if self.fail_after_tool:
                raise RuntimeError("provider failed after the tool ran")
            return _result(AIMessage(content=f"[{self.model_name}] {last.content}"))
        text = str(last.content)
        tools = kwargs.get("tools", [])
        if text.startswith("How many") and "count_open_tickets" in tools:
            call = {"name": "count_open_tickets", "args": {}, "id": "c1"}
            return _result(AIMessage(content="", tool_calls=[call]))
        if text.startswith("Open a ticket") and "open_ticket" in tools:
            call = {"name": "open_ticket", "args": {"subject": text}, "id": "c2"}
            return _result(AIMessage(content="", tool_calls=[call]))
        if text.startswith("What river"):
            first = next(str(m.content) for m in messages if isinstance(m, HumanMessage))
            river = "The Seine" if "Paris" in first else "The Thames"
            return _result(AIMessage(content=f"{river} runs through it."))
        return _result(AIMessage(content=f"[{self.model_name}] {text}"))


def _ticket_tools(tickets: list[str]) -> list[StructuredTool]:
    def count_open_tickets() -> int:
        """Count the customer's open tickets."""
        return len(tickets)

    def open_ticket(subject: str) -> str:
        """Open a support ticket."""
        tickets.append(subject)
        return f"T-{100 + len(tickets)} opened"

    return [
        # What MCPToolAdapter records for a server tool with read_only_hint=True
        StructuredTool.from_function(
            count_open_tickets, metadata={"mcp_annotations": {"readOnlyHint": True}}
        ),
        # No annotations: the MCP default, readOnlyHint=false
        StructuredTool.from_function(open_ticket),
    ]


async def _agent(cache: SemanticCache, tickets: list[str] | None = None, **kwargs: Any):
    kwargs.setdefault("model", SupportModel())
    return await build_agent(
        servers={},
        cache=cache,
        extra_tools=_ticket_tools(tickets if tickets is not None else []),
        **kwargs,
    )


async def _ask(agent: Any, text: str | list[dict[str, str]]) -> str:
    messages = [{"role": "user", "content": text}] if isinstance(text, str) else text
    result = await agent.ainvoke({"messages": messages}, caller=ALICE)
    return str(result["messages"][-1].content)


PARIS = [
    {"role": "user", "content": "Tell me about the city of Paris."},
    {"role": "assistant", "content": "Paris is the capital of France."},
    {"role": "user", "content": "What river runs through it?"},
]
LONDON = [
    {"role": "user", "content": "Tell me about the city of London."},
    {"role": "assistant", "content": "London is the capital of the United Kingdom."},
    {"role": "user", "content": "What river runs through it?"},
]


class TestConversations:
    @pytest.mark.asyncio
    async def test_follow_ups_in_different_conversations_do_not_collide(self):
        cache = SemanticCache(embedding=ExactEmbeddings())
        agent = await _agent(cache)
        try:
            assert await _ask(agent, PARIS) == "The Seine runs through it."
            assert await _ask(agent, LONDON) == "The Thames runs through it."
        finally:
            await agent.shutdown()
        # cache_multi_turn=False: conversations with history are not cached at all
        stats = await cache.stats()
        assert (stats.hits, stats.stores) == (0, 0)

    @pytest.mark.asyncio
    async def test_cache_multi_turn_hits_only_for_the_same_history(self):
        cache = SemanticCache(embedding=ExactEmbeddings(), cache_multi_turn=True)
        calls: list[str] = []
        agent = await _agent(cache, model=SupportModel(calls=calls))
        try:
            answers = [await _ask(agent, conv) for conv in (PARIS, LONDON, PARIS, LONDON)]
        finally:
            await agent.shutdown()
        assert answers == ["The Seine runs through it.", "The Thames runs through it."] * 2
        assert len(calls) == 2  # the repeats were served from the cache
        assert (await cache.stats()).hits == 2

    @pytest.mark.asyncio
    async def test_single_turn_questions_are_still_cached(self):
        cache = SemanticCache(embedding=ExactEmbeddings())
        calls: list[str] = []
        agent = await _agent(cache, model=SupportModel(calls=calls))
        try:
            for _ in range(2):
                await _ask(agent, "What are your opening hours?")
        finally:
            await agent.shutdown()
        assert len(calls) == 1
        assert (await cache.stats()).hits == 1


class TestToolTurns:
    @pytest.mark.asyncio
    async def test_tool_turns_are_not_cached_by_default(self):
        tickets = ["a", "b"]
        cache = SemanticCache(embedding=ExactEmbeddings())
        agent = await _agent(cache, tickets)
        try:
            assert await _ask(agent, "How many open tickets do I have?") == "[support-model] 2"
            tickets.append("c")  # changed outside the agent
            assert await _ask(agent, "How many open tickets do I have?") == "[support-model] 3"
        finally:
            await agent.shutdown()
        assert (await cache.stats()).stores == 0

    @pytest.mark.asyncio
    async def test_write_turn_is_never_replayed(self):
        tickets: list[str] = []
        # Even with tool turns cached, a turn that wrote must run every time
        cache = SemanticCache(embedding=ExactEmbeddings(), cache_tool_turns=True)
        agent = await _agent(cache, tickets)
        try:
            first = await _ask(agent, "Open a ticket: the invoice PDF is blank.")
            second = await _ask(agent, "Open a ticket: the invoice PDF is blank.")
        finally:
            await agent.shutdown()
        assert (first, second) == ("[support-model] T-101 opened", "[support-model] T-102 opened")
        assert len(tickets) == 2

    @pytest.mark.asyncio
    async def test_write_invalidates_cached_read_only_answers(self):
        tickets = ["a", "b"]
        cache = SemanticCache(embedding=ExactEmbeddings(), cache_tool_turns=True)
        agent = await _agent(cache, tickets)
        try:
            assert await _ask(agent, "How many open tickets do I have?") == "[support-model] 2"
            assert await _ask(agent, "How many open tickets do I have?") == "[support-model] 2"
            assert (await cache.stats()).hits == 1  # read-only tool turn was cached

            await _ask(agent, "Open a ticket: printer on fire")
            assert await _ask(agent, "How many open tickets do I have?") == "[support-model] 3"
        finally:
            await agent.shutdown()

    @pytest.mark.asyncio
    async def test_write_evicts_even_when_the_run_then_fails(self):
        cache = SemanticCache(embedding=ExactEmbeddings())
        agent = await _agent(cache)
        await _ask(agent, "What are your opening hours?")
        assert (await cache.stats()).stores == 1

        failing = await _agent(cache, model=SupportModel(fail_after_tool=True))
        try:
            with pytest.raises(Exception, match="provider failed after the tool ran"):
                await _ask(failing, "Open a ticket: broken login")
            calls: list[str] = []
            fresh = await _agent(cache, model=SupportModel(calls=calls))
            await _ask(fresh, "What are your opening hours?")
            assert calls, "the cached answer from before the write was served"
            await fresh.shutdown()
        finally:
            await failing.shutdown()
            await agent.shutdown()

    @pytest.mark.asyncio
    async def test_invalidate_on_write_false_keeps_answers(self):
        cache = SemanticCache(embedding=ExactEmbeddings(), invalidate_on_write=False)
        calls: list[str] = []
        agent = await _agent(cache, model=SupportModel(calls=calls))
        try:
            await _ask(agent, "What are your opening hours?")
            await _ask(agent, "Open a ticket: broken login")
            calls.clear()
            await _ask(agent, "What are your opening hours?")
        finally:
            await agent.shutdown()
        assert calls == []

    @pytest.mark.asyncio
    async def test_read_only_tools_list_covers_unannotated_tools(self):
        tickets: list[str] = []
        cache = SemanticCache(
            embedding=ExactEmbeddings(), cache_tool_turns=True, read_only_tools=["open_*"]
        )
        agent = await _agent(cache, tickets)
        try:
            for _ in range(2):
                await _ask(agent, "Open a ticket: once")
        finally:
            await agent.shutdown()
        # Declared read-only, so the second turn was (wrongly but as configured) cached
        assert len(tickets) == 1


class TestObservedCache:
    @pytest.mark.asyncio
    async def test_cache_hits_with_observe_and_records_similarity_and_age(self, caplog):
        cache = SemanticCache(embedding=ExactEmbeddings())
        calls: list[str] = []
        agent = await _agent(cache, model=SupportModel(calls=calls), observe=True)
        try:
            with caplog.at_level(logging.WARNING):
                for _ in range(2):
                    await _ask(agent, "How do I reset my password?")
            timeline = agent.collector.get_timeline()
        finally:
            await agent.shutdown()

        assert len(calls) == 1
        assert not [r for r in caplog.records if "Cache" in r.getMessage()]
        events = [e.event_type for e in timeline if e.event_type.value.startswith("cache.")]
        assert events == [
            TimelineEventType.CACHE_MISS,
            TimelineEventType.CACHE_STORE,
            TimelineEventType.CACHE_HIT,
        ]
        hit = next(e for e in timeline if e.event_type == TimelineEventType.CACHE_HIT)
        assert hit.metadata["similarity"] == pytest.approx(1.0, abs=1e-3)
        assert 0 <= hit.metadata["age_seconds"] < 5
        assert "similarity" in (hit.details or "")


class TestFallbackModelKey:
    @pytest.mark.asyncio
    async def test_fallback_answer_is_cached_under_the_model_that_wrote_it(self):
        primary = SupportModel(model_name="primary", fail=True)
        calls: list[str] = []
        backup = SupportModel(model_name="backup", calls=calls)
        chain = FallbackChain([primary, backup], failure_threshold=2)
        cache = SemanticCache(embedding=ExactEmbeddings())
        agent = await _agent(cache, model=chain)
        try:
            assert agent.model_name == "primary"
            assert await _ask(agent, "Hello") == "[backup] Hello"
            entries = [e for es in cache._backend._entries.values() for e in es]
            assert [e.model_id for e in entries] == ["backup"]

            # Primary's circuit still closed: a primary-keyed lookup misses
            await _ask(agent, "Hello")
            assert len(calls) == 2
            # Circuit now open: the backup is serving, its cached answer applies
            assert chain.active_model == "backup"
            await _ask(agent, "Hello")
            assert len(calls) == 2
        finally:
            await agent.shutdown()


class TestBuild:
    @pytest.mark.asyncio
    async def test_missing_sentence_transformers_fails_the_build(self, monkeypatch):
        import importlib.util

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **k: None if name == "sentence_transformers" else real(name, *a, **k),
        )
        with pytest.raises(ImportError, match="sentence-transformers"):
            await build_agent(servers={}, model=SupportModel(), cache=SemanticCache())


class TestMCPAnnotations:
    @pytest.mark.asyncio
    async def test_adapter_keeps_tool_annotations(self):
        from mcp.types import Tool, ToolAnnotations

        from promptise.cache import tool_annotations
        from promptise.mcp.client import MCPToolAdapter

        class FakeMulti:
            tool_to_server: dict[str, str] = {}

            async def list_tools(self) -> list[Tool]:
                schema = {"type": "object", "properties": {}}
                return [
                    Tool(
                        name="count",
                        inputSchema=schema,
                        annotations=ToolAnnotations(readOnlyHint=True),
                    ),
                    Tool(name="open", inputSchema=schema),
                ]

        tools = await MCPToolAdapter(FakeMulti()).as_langchain_tools()  # type: ignore[arg-type]
        by_name = {t.name: tool_annotations(t) for t in tools}
        assert by_name == {"count": {"readOnlyHint": True}, "open": None}


class _CountTool(BaseTool):
    """``count_open_tickets`` as a plain BaseTool, annotated read-only."""

    name: str = "count_open_tickets"
    description: str = "Count the customer's open tickets."
    tickets: Any = None

    def _run(self, **kwargs: Any) -> int:
        return len(self.tickets)

    async def _arun(self, **kwargs: Any) -> int:
        return len(self.tickets)


async def _gated_agent(cache: SemanticCache, tickets: list[str], reviewer: Any) -> Any:
    from promptise.approval import ApprovalPolicy

    count = _CountTool(tickets=tickets, metadata={"mcp_annotations": {"readOnlyHint": True}})
    return await build_agent(
        servers={},
        model=SupportModel(),
        cache=cache,
        extra_tools=[count],
        approval=ApprovalPolicy(tools=["count_*"], handler=reviewer),
    )


class TestApprovalGatedTools:
    @pytest.mark.asyncio
    async def test_cached_answer_never_skips_an_approval(self):
        """A read-only tool behind an approval gate: every ask needs a decision."""
        asked: list[str] = []

        async def reviewer(request: Any) -> bool:
            asked.append(request.tool_name)
            return True

        cache = SemanticCache(embedding=ExactEmbeddings(), cache_tool_turns=True)
        agent = await _gated_agent(cache, ["a", "b"], reviewer)
        try:
            for _ in range(2):
                assert await _ask(agent, "How many open tickets do I have?") == "[support-model] 2"
        finally:
            await agent.shutdown()
        assert asked == ["count_open_tickets", "count_open_tickets"]
        assert (await cache.stats()).stores == 0

    @pytest.mark.asyncio
    async def test_a_denial_is_not_replayed_either(self):
        decisions = iter([False, True])

        async def reviewer(request: Any) -> bool:
            return next(decisions)

        cache = SemanticCache(embedding=ExactEmbeddings(), cache_tool_turns=True)
        agent = await _gated_agent(cache, ["a"], reviewer)
        try:
            denied = await _ask(agent, "How many open tickets do I have?")
            approved = await _ask(agent, "How many open tickets do I have?")
        finally:
            await agent.shutdown()
        assert approved == "[support-model] 1"
        assert denied != approved


class _SameVector:
    """Every text embeds to the same vector, like two close paraphrases."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0] + [0.0] * 15 for _ in texts]


class TestReplayedOutput:
    @pytest.mark.asyncio
    async def test_hit_returns_this_requests_messages_not_the_original_askers(self):
        cache = SemanticCache(
            embedding=_SameVector(), scope="shared", shared_data_acknowledged=True
        )
        calls: list[str] = []
        agent = await _agent(cache, model=SupportModel(calls=calls))
        try:
            first = await agent.ainvoke(
                {
                    "messages": [
                        {"role": "user", "content": "I'm Alice (acct 4417). Opening hours?"}
                    ]
                },
                caller=ALICE,
            )
            second = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "Opening hours?"}]},
                caller=CallerContext(user_id="bob"),
            )
        finally:
            await agent.shutdown()

        assert len(calls) == 1  # served from the shared cache
        answer = first["messages"][-1].content
        texts = [str(m.content) for m in second["messages"]]
        assert texts == ["Opening hours?", answer]
        assert not any("4417" in t for t in texts[:-1])

    @pytest.mark.asyncio
    async def test_cache_stores_only_the_answer(self):
        cache = SemanticCache(embedding=ExactEmbeddings())
        agent = await _agent(cache)
        try:
            await _ask(agent, "What are your opening hours?")
        finally:
            await agent.shutdown()
        (entry,) = [e for es in cache._backend._entries.values() for e in es]
        assert [m.type for m in entry.output["messages"]] == ["ai"]
