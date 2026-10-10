"""ConversationFlow first message, token budget and inspector; FlowSessions;
Prompt inspector traces; PromptAssembler ordering and fill_slot semantics."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptise.prompts import prompt
from promptise.prompts.blocks import (
    ContextSlot,
    Examples,
    Identity,
    PromptAssembler,
    Section,
)
from promptise.prompts.flows import ConversationFlow, FlowSessions, TurnContext, phase
from promptise.prompts.guards import GuardError, content_filter
from promptise.prompts.inspector import PromptInspector


class EchoFlow(ConversationFlow):
    """Records what each handler run saw; moves to "done" on the word "done"."""

    base_blocks = [Identity("Assistant")]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.seen: list[tuple[int, list[str]]] = []

    @phase("open", initial=True)
    async def open_(self, ctx: TurnContext) -> None:
        self.seen.append((ctx.turn, [m["content"] for m in ctx.history]))
        if any("done" in m["content"] for m in ctx.history):
            ctx.transition("done")

    @phase("done")
    async def done(self, ctx: TurnContext) -> None:
        pass


class BudgetFlow(ConversationFlow):
    base_blocks = [
        Identity("Assistant"),
        Section("filler", "word " * 300, priority=1),
        Section("rules", "Be brief.", priority=9),
    ]

    @phase("only", initial=True)
    async def only(self, ctx: TurnContext) -> None:
        pass


# ---------------------------------------------------------------------------
# ConversationFlow
# ---------------------------------------------------------------------------


class TestStartWithFirstMessage:
    async def test_handler_sees_the_first_message_on_turn_zero(self):
        flow = EchoFlow()
        await flow.start("hello")
        assert flow.seen == [(0, ["hello"])]
        assert flow._history == [{"role": "user", "content": "hello"}]

    async def test_first_message_can_trigger_a_transition(self):
        flow = EchoFlow()
        await flow.start("all done")
        assert flow.current_phase == "done"

    async def test_start_without_message_is_unchanged(self):
        flow = EchoFlow()
        await flow.start()
        assert flow.seen == [(0, [])]
        assert flow._history == []
        assert flow.current_phase == "open"

    def test_current_phase_is_none_before_start(self):
        assert EchoFlow().current_phase is None


class TestFlowTokenBudget:
    async def test_no_budget_keeps_every_block(self):
        prompt_ = await BudgetFlow().start()
        assert prompt_.included == ["identity", "filler", "rules"]

    async def test_constructor_budget_drops_lowest_priority_first(self):
        prompt_ = await BudgetFlow(token_budget=50).start()
        assert prompt_.excluded == ["filler"]
        assert prompt_.included == ["identity", "rules"]

    async def test_class_attribute_budget(self):
        class Capped(BudgetFlow):
            token_budget = 50

        flow = Capped()
        await flow.start()
        assert "filler" in (await flow.next_turn("hi")).excluded
        assert "filler" in flow.get_prompt().excluded


class TestFlowInspector:
    async def test_each_turn_is_recorded_with_phase_and_turn(self):
        inspector = PromptInspector()
        flow = EchoFlow(inspector=inspector)
        await flow.start("hi")
        await flow.next_turn("still going")
        await flow.transition("done")
        flow.get_prompt()  # reading the prompt records nothing

        rows = [(t.prompt_name, t.flow_phase, t.flow_turn) for t in inspector.traces]
        assert rows == [("EchoFlow", "open", 0), ("EchoFlow", "open", 1), ("EchoFlow", "done", 1)]
        assert inspector.last().input_text.startswith("You are Assistant.")
        assert "Flow: phase=done turn=1" in inspector.summary()


# ---------------------------------------------------------------------------
# FlowSessions
# ---------------------------------------------------------------------------


class TestFlowSessions:
    async def test_template_copies_are_independent(self):
        template = EchoFlow()
        sessions = FlowSessions(template)
        a, _ = await sessions.advance("a", ["done"])
        b, _ = await sessions.advance("b", ["hello"])
        assert a is not b and a is not template
        assert (a.current_phase, b.current_phase, template.current_phase) == ("done", "open", None)

    async def test_copies_share_the_template_inspector(self):
        inspector = PromptInspector()
        sessions = FlowSessions(EchoFlow(inspector=inspector))
        await sessions.advance("a", ["x"])
        await sessions.advance("b", ["y"])
        assert len(inspector.traces) == 2

    async def test_running_flow_takes_only_the_newest_message(self):
        sessions = FlowSessions(EchoFlow)
        await sessions.advance("a", ["one"])
        flow, _ = await sessions.advance("a", ["one", "two"])
        assert [m["content"] for m in flow._history] == ["one", "two"]
        assert flow._turn == 1

    async def test_new_flow_replays_every_message(self):
        sessions = FlowSessions(EchoFlow)
        flow, _ = await sessions.advance("a", ["one", "two", "three"])
        assert flow.seen == [(0, ["one"]), (1, ["one", "two"]), (2, ["one", "two", "three"])]

    async def test_no_key_is_not_kept(self):
        sessions = FlowSessions(EchoFlow)
        first, _ = await sessions.advance(None, ["x"])
        second, _ = await sessions.advance(None, ["x"])
        assert first is not second
        assert len(sessions) == 0

    async def test_least_recently_used_flow_is_evicted(self):
        sessions = FlowSessions(EchoFlow, max_sessions=2)
        await sessions.advance("a", ["x"])
        await sessions.advance("b", ["x"])
        await sessions.advance("a", ["x"])  # touch a
        await sessions.advance("c", ["x"])
        assert sessions.get("b") is None
        assert sessions.get("a") is not None and sessions.get("c") is not None

    async def test_failed_start_is_not_kept(self):
        class Broken(EchoFlow):
            @phase("open", initial=True)
            async def open_(self, ctx: TurnContext) -> None:
                raise RuntimeError("boom")

        sessions = FlowSessions(Broken)
        with pytest.raises(RuntimeError):
            await sessions.advance("a", ["x"])
        assert sessions.get("a") is None

    async def test_turns_for_one_key_do_not_interleave(self):
        order: list[str] = []

        class Slow(EchoFlow):
            @phase("open", initial=True)
            async def open_(self, ctx: TurnContext) -> None:
                order.append(f"in {ctx.turn}")
                await asyncio.sleep(0)
                order.append(f"out {ctx.turn}")

        sessions = FlowSessions(Slow)
        await asyncio.gather(*(sessions.advance("a", [f"m{i}"]) for i in range(3)))
        assert order == ["in 0", "out 0", "in 1", "out 1", "in 2", "out 2"]

    async def test_discard_by_predicate(self):
        sessions = FlowSessions(EchoFlow)
        await sessions.advance(("session", "", "s1"), ["x"])
        await sessions.advance(("caller", "u1"), ["x"])
        assert sessions.discard(lambda key: key[0] == "session") == 1
        assert len(sessions) == 1

    def test_rejects_non_flow_sources(self):
        with pytest.raises(TypeError, match="ConversationFlow"):
            FlowSessions("SupportFlow")  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            FlowSessions(EchoFlow, max_sessions=0)

    async def test_factory_must_return_a_flow(self):
        sessions = FlowSessions(lambda: object())  # type: ignore[arg-type,return-value]
        with pytest.raises(TypeError, match="not a ConversationFlow"):
            await sessions.advance("a", ["x"])


# ---------------------------------------------------------------------------
# Prompt inspector traces
# ---------------------------------------------------------------------------


def _llm(reply: str) -> AsyncMock:
    message = MagicMock()
    message.content = reply
    model = AsyncMock()
    model.ainvoke = AsyncMock(return_value=message)
    return model


class TestPromptInspectorTraces:
    async def test_call_without_blocks_records_a_full_trace(self):
        inspector = PromptInspector()

        @prompt(model="openai:gpt-5-mini", inspect=inspector)
        async def greet(name: str) -> str:
            """Hello {name}."""

        with patch("promptise.prompts.core.init_chat_model", return_value=_llm("Hi!")):
            await greet("Ada")

        trace = inspector.last()
        assert trace is not None
        assert trace.input_text == "Hello Ada."
        assert trace.output_text == "Hi!"
        assert trace.latency_ms > 0

    async def test_input_text_is_the_final_prompt_not_just_the_blocks(self):
        inspector = PromptInspector()

        @prompt(model="openai:gpt-5-mini", inspect=inspector)
        async def analyze(text: str) -> str:
            """Analyze: {text}"""

        analyze = analyze.with_blocks(Identity("Analyst")).with_constraints("Be brief")
        with patch("promptise.prompts.core.init_chat_model", return_value=_llm("ok")):
            await analyze("data")

        text = inspector.last().input_text
        assert text.startswith("You are Analyst.")
        assert "Analyze: data" in text and "Be brief" in text

    async def test_guard_results_are_recorded(self):
        inspector = PromptInspector()

        @prompt(model="openai:gpt-5-mini", inspect=inspector)
        async def say(text: str) -> str:
            """Say: {text}"""

        guarded = say.with_guards(content_filter(blocked=["secret"]))
        with patch("promptise.prompts.core.init_chat_model", return_value=_llm("fine")):
            await guarded("hello")
            assert inspector.last().guards_passed == ["ContentFilterGuard"]
            with pytest.raises(GuardError):
                await guarded("the secret")
        assert inspector.last().guards_failed == ["ContentFilterGuard"]

    async def test_render_async_records_a_trace(self):
        inspector = PromptInspector()

        @prompt(model="openai:gpt-5-mini", inspect=inspector)
        async def sysp() -> str:
            """You are helpful."""

        text = await sysp.render_async()
        assert inspector.last().input_text == text


# ---------------------------------------------------------------------------
# PromptAssembler: list order, and fill_slot mutates (as documented)
# ---------------------------------------------------------------------------


class TestAssemblerDocumentedBehaviour:
    def test_blocks_keep_list_order_regardless_of_priority(self):
        assembled = PromptAssembler(
            Examples([{"input": "x", "output": "y"}]), Identity("A")
        ).assemble()
        assert assembled.included == ["examples", "identity"]

    def test_budget_drops_lowest_priority_but_keeps_list_order(self):
        assembled = PromptAssembler(
            Section("low", "word " * 100, priority=1),
            Section("tail", "Keep me.", priority=9),
            Identity("A"),
        ).assemble(token_budget=20)
        assert assembled.excluded == ["low"]
        assert assembled.included == ["tail", "identity"]

    def test_fill_slot_updates_the_assembler_in_place(self):
        assembler = PromptAssembler(Identity("A"), ContextSlot("data"))
        assert assembler.fill_slot("data", "Revenue: 2.3M") is assembler
        assert "Revenue: 2.3M" in assembler.assemble().text
