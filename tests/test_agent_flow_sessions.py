"""build_agent(flow=...) keeps one ConversationFlow per conversation.

Covers per-session/per-caller isolation (including concurrent sessions),
the first user message reaching the flow, the single merged system
message, and a Prompt's guards and inspector when it is the agent's
instructions.  A recording chat model stands in for the LLM.
"""

from __future__ import annotations

import asyncio
import re
import threading
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from promptise import build_agent
from promptise.agent import CallerContext
from promptise.conversations import InMemoryConversationStore
from promptise.prompt import DEFAULT_SYSTEM_PROMPT
from promptise.prompts import prompt
from promptise.prompts.blocks import ContextSlot, Identity, Rules, Section
from promptise.prompts.flows import ConversationFlow, TurnContext, phase
from promptise.prompts.guards import GuardError, content_filter, input_validator, output_validator
from promptise.prompts.inspector import PromptInspector

ORDER_ID = re.compile(r"\bA-\d{4}\b")


class SupportFlow(ConversationFlow):
    """Triage until the customer gives an order ID, then resolve that order."""

    base_blocks = [Identity("the support assistant"), Rules(["Never share one customer's data"])]

    def __init__(self, *, business: bool = False) -> None:
        super().__init__()
        self.business = business

    @phase(
        "triage",
        initial=True,
        blocks=[Section("triage", "Current phase: triage. Ask for the order ID.")],
    )
    async def triage(self, ctx: TurnContext) -> None:
        ctx.state.setdefault("business", self.business)
        for message in ctx.history:
            match = ORDER_ID.search(message["content"])
            if match:
                ctx.state["order_id"] = match.group()
                ctx.transition("resolution")

    @phase(
        "resolution",
        blocks=[Section("resolution", "Current phase: resolution."), ContextSlot("case")],
    )
    async def resolution(self, ctx: TurnContext) -> None:
        ctx.fill_slot("case", f"Case: order {ctx.state['order_id']}.")


class Recorder(BaseChatModel):
    """Chat model that records the messages of every call.

    ``replies`` are returned in order (then ``"ok"``).  When ``rendezvous``
    is set, each call waits on it, so concurrent calls overlap.
    """

    calls: list[list[Any]] = []
    replies: list[AIMessage] = []
    rendezvous: Any = None

    @property
    def _llm_type(self) -> str:
        return "recorder"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Recorder:
        return self

    def _next(self, messages: list[Any]) -> ChatResult:
        self.calls.append(list(messages))
        reply = self.replies.pop(0) if self.replies else AIMessage(content="ok")
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return self._next(messages)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.rendezvous is not None:
            await self.rendezvous.wait()
        return self._next(messages)


@tool
def get_order(order_id: str) -> str:
    """Look up an order."""
    return f'{{"order_id": "{order_id}", "customer": "Dana"}}'


def _model() -> Recorder:
    return Recorder(calls=[], replies=[])


def _systems(messages: list[Any]) -> list[str]:
    return [str(m.content) for m in messages if isinstance(m, SystemMessage)]


def _system_text(messages: list[Any]) -> str:
    return "\n".join(_systems(messages))


async def _agent(model: Recorder, **kwargs: Any) -> Any:
    kwargs.setdefault("flow", SupportFlow())
    return await build_agent(model=model, servers={}, extra_tools=[get_order], **kwargs)


def _user(text: str) -> dict[str, str]:
    return {"role": "user", "content": text}


# ---------------------------------------------------------------------------
# 1. One flow per conversation
# ---------------------------------------------------------------------------


class TestPerSessionFlows:
    async def test_sessions_do_not_share_flow_state(self):
        model = _model()
        template = SupportFlow()
        agent = await _agent(model, flow=template)
        try:
            await agent.chat("Hi there.", session_id="dana", user_id="dana")
            await agent.chat("My mug from order A-1001 broke.", session_id="dana", user_id="dana")
            dana_prompt = _system_text(model.calls[-1])
            assert "Case: order A-1001." in dana_prompt

            await agent.chat("I have a question about my desk.", session_id="lee", user_id="lee")
            lee_prompt = _system_text(model.calls[-1])
            assert "A-1001" not in lee_prompt
            assert "Current phase: triage." in lee_prompt

            assert agent.get_flow("dana", user_id="dana").current_phase == "resolution"
            assert agent.get_flow("lee", user_id="lee").current_phase == "triage"
            # The template passed to build_agent is never advanced.
            assert template.current_phase is None
        finally:
            await agent.shutdown()

    async def test_concurrent_sessions_keep_their_own_state(self):
        model = _model()
        agent = await _agent(model)
        gate = asyncio.Event()
        model.rendezvous = gate
        try:
            dana = asyncio.create_task(
                agent.chat("Order A-1001 arrived broken.", session_id="s-dana")
            )
            lee = asyncio.create_task(agent.chat("Question about my desk.", session_id="s-lee"))
            # Both conversations advance their flows before either reaches the model.
            for _ in range(200):
                if agent._flows is not None and len(agent._flows) == 2:
                    break
                await asyncio.sleep(0.005)
            gate.set()
            await asyncio.gather(dana, lee)

            by_user = {
                next(m.content for m in call if isinstance(m, HumanMessage)): _system_text(call)
                for call in model.calls
            }
            assert "Case: order A-1001." in by_user["Order A-1001 arrived broken."]
            assert "A-1001" not in by_user["Question about my desk."]
            assert "Current phase: triage." in by_user["Question about my desk."]
        finally:
            await agent.shutdown()

    async def test_concurrent_turns_in_one_session_are_serialized(self):
        model = _model()
        agent = await _agent(model)
        try:
            await asyncio.gather(
                *(
                    agent.ainvoke({"messages": [_user(f"message {i}")]}, session_id="s1")
                    for i in range(5)
                )
            )
            flow = agent.get_flow("s1")
            assert len(flow._history) == 5
            assert flow._turn == 4  # start() is turn 0, then four next_turn() calls
        finally:
            await agent.shutdown()

    async def test_flows_are_kept_per_caller(self):
        model = _model()
        agent = await _agent(model)
        dana = CallerContext(user_id="dana")
        lee = CallerContext(user_id="lee")
        try:
            await agent.ainvoke({"messages": [_user("Order A-1001 broke.")]}, caller=dana)
            await agent.ainvoke({"messages": [_user("Hello?")]}, caller=lee)
            assert "A-1001" not in _system_text(model.calls[-1])

            # The same caller continues its own flow, one message per call.
            await agent.ainvoke({"messages": [_user("Thanks.")]}, caller=dana)
            assert "Case: order A-1001." in _system_text(model.calls[-1])
            assert agent.get_flow(caller=dana)._turn == 1
        finally:
            await agent.shutdown()

    async def test_same_user_in_two_tenants_gets_two_flows(self):
        model = _model()
        agent = await _agent(model)
        acme = CallerContext(user_id="u1", tenant_id="acme")
        globex = CallerContext(user_id="u1", tenant_id="globex")
        try:
            await agent.ainvoke({"messages": [_user("Order A-1001 broke.")]}, caller=acme)
            await agent.ainvoke({"messages": [_user("Hello?")]}, caller=globex)
            assert "A-1001" not in _system_text(model.calls[-1])
            assert agent.get_flow(caller=acme) is not agent.get_flow(caller=globex)
        finally:
            await agent.shutdown()

    async def test_same_session_id_for_two_users_is_two_flows(self):
        model = _model()
        agent = await _agent(model)
        try:
            await agent.chat("Order A-1001 broke.", session_id="shared", user_id="dana")
            await agent.chat("Hello?", session_id="shared", user_id="lee")
            assert "A-1001" not in _system_text(model.calls[-1])
        finally:
            await agent.shutdown()

    async def test_anonymous_calls_get_a_throwaway_flow(self):
        model = _model()
        agent = await _agent(model)
        try:
            await agent.ainvoke({"messages": [_user("Order A-1001 broke.")]})
            await agent.ainvoke({"messages": [_user("Question about my desk.")]})
            assert "A-1001" not in _system_text(model.calls[-1])
            assert len(agent._flows) == 0
        finally:
            await agent.shutdown()

    async def test_anonymous_call_replays_the_history_it_is_given(self):
        model = _model()
        agent = await _agent(model)
        try:
            await agent.ainvoke(
                {
                    "messages": [
                        _user("Hi."),
                        {"role": "assistant", "content": "What is your order ID?"},
                        _user("It is A-1001."),
                    ]
                }
            )
            assert "Case: order A-1001." in _system_text(model.calls[-1])
        finally:
            await agent.shutdown()

    async def test_evicted_session_is_rebuilt_from_stored_history(self):
        model = _model()
        agent = await _agent(model, conversation_store=InMemoryConversationStore())
        try:
            await agent.chat("Order A-1001 broke.", session_id="s1", user_id="dana")
            assert agent._flows.discard(lambda key: True) == 1  # simulate eviction/restart

            await agent.chat("Can you help?", session_id="s1", user_id="dana")
            assert "Case: order A-1001." in _system_text(model.calls[-1])
            assert agent.get_flow("s1", user_id="dana").current_phase == "resolution"
        finally:
            await agent.shutdown()

    async def test_delete_session_drops_its_flow(self):
        model = _model()
        agent = await _agent(model, conversation_store=InMemoryConversationStore())
        try:
            await agent.chat("Hi.", session_id="s1")
            assert agent.get_flow("s1") is not None
            await agent.delete_session("s1")
            assert agent.get_flow("s1") is None
        finally:
            await agent.shutdown()

    async def test_flow_class_and_factory_are_accepted(self):
        for source in (SupportFlow, lambda: SupportFlow(business=True)):
            model = _model()
            agent = await _agent(model, flow=source)
            try:
                await agent.chat("Order A-1001 broke.", session_id="s1")
                assert agent.get_flow("s1").current_phase == "resolution"
            finally:
                await agent.shutdown()

    async def test_started_template_is_reset_for_each_conversation(self):
        template = SupportFlow()
        await template.start("Order A-9999 broke.")
        assert template.current_phase == "resolution"

        model = _model()
        agent = await _agent(model, flow=template)
        try:
            await agent.chat("Hello?", session_id="s1")
            assert "A-9999" not in _system_text(model.calls[-1])
            assert agent.get_flow("s1").current_phase == "triage"
        finally:
            await agent.shutdown()

    async def test_template_that_cannot_be_copied_fails_at_build(self):
        class LockedFlow(SupportFlow):
            def __init__(self) -> None:
                super().__init__()
                self.lock = threading.Lock()

        with pytest.raises(TypeError, match="Pass a factory"):
            await build_agent(model=_model(), servers={}, flow=LockedFlow())


# ---------------------------------------------------------------------------
# 2. The first user message reaches the flow
# ---------------------------------------------------------------------------


class TestFirstMessage:
    async def test_first_message_drives_the_initial_phase(self):
        model = _model()
        agent = await _agent(model)
        try:
            await agent.ainvoke({"messages": [_user("My mug from order A-1001 broke.")]})
            prompt_text = _system_text(model.calls[-1])
            assert "Current phase: resolution." in prompt_text
            assert "Case: order A-1001." in prompt_text
        finally:
            await agent.shutdown()

    async def test_first_message_is_in_the_flow_history(self):
        model = _model()
        agent = await _agent(model)
        try:
            await agent.chat("First words.", session_id="s1")
            await agent.chat("Second words.", session_id="s1")
            history = [m["content"] for m in agent.get_flow("s1")._history]
            assert history == ["First words.", "Second words."]
        finally:
            await agent.shutdown()


# ---------------------------------------------------------------------------
# 3. One system message, no default when a flow provides the prompt
# ---------------------------------------------------------------------------


class TestSingleSystemMessage:
    async def test_flow_and_instructions_merge_into_one_system_message(self):
        model = _model()
        agent = await _agent(model, instructions="Use the tools to look up orders.")
        try:
            await agent.ainvoke({"messages": [_user("Hi.")]})
            systems = _systems(model.calls[-1])
            assert len(systems) == 1
            text = systems[0]
            assert text.startswith("You are the support assistant.")
            assert text.index("Current phase: triage.") < text.index("Use the tools")
            assert text.index("Use the tools") < text.index("Available tools:")
            assert DEFAULT_SYSTEM_PROMPT not in text
        finally:
            await agent.shutdown()

    @pytest.mark.parametrize("instructions", [None, ""])
    async def test_no_default_prompt_when_flow_provides_it(self, instructions):
        model = _model()
        agent = await _agent(model, instructions=instructions)
        try:
            await agent.ainvoke({"messages": [_user("Hi.")]})
            systems = _systems(model.calls[-1])
            assert len(systems) == 1
            assert "capable deep agent" not in systems[0]
            assert "Available tools:" in systems[0]
        finally:
            await agent.shutdown()

    async def test_default_prompt_still_used_without_a_flow(self):
        model = _model()
        agent = await build_agent(model=model, servers={}, extra_tools=[get_order])
        try:
            await agent.ainvoke({"messages": [_user("Hi.")]})
            assert DEFAULT_SYSTEM_PROMPT in _system_text(model.calls[-1])
        finally:
            await agent.shutdown()

    async def test_one_system_message_across_a_tool_loop(self):
        model = _model()
        model.replies = [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_order", "args": {"order_id": "A-1001"}, "id": "c1"}],
            ),
            AIMessage(content="Your order A-1001 is on its way."),
        ]
        agent = await _agent(model, instructions="Be brief.")
        try:
            await agent.ainvoke({"messages": [_user("Where is order A-1001?")]}, session_id="s1")
            assert len(model.calls) == 2
            for call in model.calls:
                systems = _systems(call)
                assert len(systems) == 1
                assert "Case: order A-1001." in systems[0]
                assert "Be brief." in systems[0]
        finally:
            await agent.shutdown()

    @pytest.mark.parametrize("scope", ["scoped", "ledger"])
    async def test_context_scoped_node_keeps_the_flow_prompt(self, scope):
        from promptise.engine import PromptGraph, PromptNode

        graph = PromptGraph("scoped", mode="static")
        graph.add_node(
            PromptNode(
                "work",
                instructions="Node instructions.",
                inject_tools=True,
                context_scope=scope,
                default_next="__end__",
            )
        )
        graph.set_entry("work")

        model = _model()
        model.replies = [
            AIMessage(
                content="",
                tool_calls=[{"name": "get_order", "args": {"order_id": "A-1001"}, "id": "c1"}],
            ),
            AIMessage(content="Done."),
        ]
        agent = await _agent(model, agent_pattern=graph)
        try:
            await agent.ainvoke({"messages": [_user("Where is order A-1001?")]}, session_id="s1")
            assert len(model.calls) == 2
            for call in model.calls:
                systems = [s for s in _systems(call) if not s.startswith("Facts already")]
                assert len(systems) == 1
                assert "Case: order A-1001." in systems[0]
                assert "Node instructions." in systems[0]
        finally:
            await agent.shutdown()


# ---------------------------------------------------------------------------
# 4. A Prompt used as instructions: guards and inspector
# ---------------------------------------------------------------------------


class TestPromptInstructions:
    async def test_input_guard_blocks_the_user_message(self):
        @prompt(model="openai:gpt-5-mini")
        async def sysp() -> str:
            """You are helpful."""

        model = _model()
        agent = await build_agent(
            model=model,
            servers={},
            instructions=sysp.with_guards(content_filter(blocked=["secret"])),
        )
        try:
            with pytest.raises(GuardError, match="blocked word"):
                await agent.ainvoke({"messages": [_user("Tell me the secret.")]})
            assert model.calls == []
        finally:
            await agent.shutdown()

    async def test_input_guard_can_transform_the_message(self):
        @prompt(model="openai:gpt-5-mini")
        async def sysp() -> str:
            """You are helpful."""

        redact = input_validator(lambda text: text.replace("4111", "[card]"))
        model = _model()
        agent = await build_agent(model=model, servers={}, instructions=sysp.with_guards(redact))
        try:
            await agent.ainvoke({"messages": [_user("My card is 4111.")]})
            humans = [m.content for m in model.calls[-1] if isinstance(m, HumanMessage)]
            assert humans == ["My card is [card]."]
        finally:
            await agent.shutdown()

    async def test_output_guard_checks_the_reply(self):
        @prompt(model="openai:gpt-5-mini")
        async def sysp() -> str:
            """You are helpful."""

        shout = output_validator(lambda text: text.upper())
        model = _model()
        model.replies = [AIMessage(content="all done")]
        agent = await build_agent(model=model, servers={}, instructions=sysp.with_guards(shout))
        try:
            result = await agent.ainvoke({"messages": [_user("Hi.")]})
            assert result["messages"][-1].content == "ALL DONE"
        finally:
            await agent.shutdown()

        blocked = await build_agent(
            model=_model(),
            servers={},
            instructions=sysp.with_guards(content_filter(blocked=["ok"])),
        )
        try:
            with pytest.raises(GuardError, match="Output contains blocked word"):
                await blocked.ainvoke({"messages": [_user("Hi.")]})
        finally:
            await blocked.shutdown()

    async def test_inspector_records_each_agent_turn(self):
        inspector = PromptInspector()

        @prompt(model="openai:gpt-5-mini", inspect=inspector)
        async def sysp() -> str:
            """You are helpful."""

        agent = await build_agent(model=_model(), servers={}, instructions=sysp)
        try:
            await agent.ainvoke({"messages": [_user("Hi.")]})
            await agent.ainvoke({"messages": [_user("Again.")]})
            assert len(inspector.traces) == 2
            assert inspector.last().prompt_name == "sysp"
            assert "You are helpful." in inspector.last().input_text
        finally:
            await agent.shutdown()

    async def test_flow_inspector_records_agent_turns(self):
        inspector = PromptInspector()

        class InspectedFlow(SupportFlow):
            pass

        InspectedFlow.inspector = inspector
        agent = await _agent(_model(), flow=InspectedFlow())
        try:
            await agent.chat("Hi.", session_id="s1")
            await agent.chat("Order A-1001.", session_id="s1")
            phases = [(t.prompt_name, t.flow_phase, t.flow_turn) for t in inspector.traces]
            assert phases == [("InspectedFlow", "triage", 0), ("InspectedFlow", "resolution", 1)]
        finally:
            await agent.shutdown()

    async def test_streaming_runs_prompt_guards(self):
        @prompt(model="openai:gpt-5-mini")
        async def sysp() -> str:
            """You are helpful."""

        model = _model()
        agent = await build_agent(
            model=model,
            servers={},
            instructions=sysp.with_guards(content_filter(blocked=["secret"])),
        )
        try:
            events = [
                e
                async for e in agent.astream_with_tools(
                    {"messages": [_user("Tell me the secret.")]}
                )
            ]
            assert [e.type for e in events] == ["error"]
            assert "prompt guard" in events[0].message
            assert model.calls == []
        finally:
            await agent.shutdown()


# ---------------------------------------------------------------------------
# Semantic cache: hits still advance the flow; phases don't share entries
# ---------------------------------------------------------------------------


class _ExactCache:
    """Duck-typed cache keyed on (query, instruction hash)."""

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], Any] = {}
        self.hashes: list[str] = []

    async def check(self, query, *, instruction_hash, **kwargs):
        self.hashes.append(instruction_hash)
        output = self.entries.get((query, instruction_hash))
        if output is None:
            return None

        class _Hit:
            scope_key = "test"
            ttl = 60

        hit = _Hit()
        hit.output = output  # type: ignore[attr-defined]
        return hit

    async def store(self, query, response, output, *, instruction_hash, **kwargs):
        self.entries[(query, instruction_hash)] = output


class TestFlowWithCache:
    async def test_cache_hit_still_advances_the_flow(self):
        model = _model()
        agent = await _agent(model, cache=_ExactCache())
        try:
            await agent.chat("Hi.", session_id="s1")
            await agent.chat("Hi.", session_id="s1")  # served from the cache
            assert len(model.calls) == 1
            assert [m["content"] for m in agent.get_flow("s1")._history] == ["Hi.", "Hi."]
        finally:
            await agent.shutdown()

    async def test_each_phase_has_its_own_cache_key(self):
        cache = _ExactCache()
        agent = await _agent(_model(), cache=cache)
        try:
            await agent.chat("Hi.", session_id="triage")
            await agent.chat("Order A-1001.", session_id="resolved")
            await agent.chat("Hi.", session_id="resolved")
            triage_hash, _, resolution_hash = cache.hashes
            assert triage_hash != resolution_hash
        finally:
            await agent.shutdown()
