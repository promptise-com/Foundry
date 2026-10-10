"""Context compaction keeps what the model needs and actually shrinks the rest.

A scripted chat model returns pre-written tool calls and records every
request, so each test asserts on exactly what the model was sent across a
tool loop long enough to compact (more than 6 tool results).
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from promptise import ContextEngine, build_agent
from promptise.conversations import InMemoryConversationStore
from promptise.engine import ContextCompaction, PromptGraph, PromptNode
from promptise.engine.compaction import LEDGER_HEADER, build_compacted_view, split_input
from promptise.engine.state import GraphState

SERVICES = [
    "auth",
    "billing",
    "gateway",
    "inventory",
    "notifications",
    "orders",
    "payments",
    "search",
    "shipping",
    "users",
]
QUESTION = "Read every service log, then give the number of ERROR lines per service."
INSTRUCTIONS = "You are an SRE assistant."
PADDING = "x" * 6000


@tool
async def log_summary(service: str) -> str:
    """Summarize one service's log."""
    return f"{service}: 3 ERROR lines. {PADDING}"


class ScriptedModel(BaseChatModel):
    """Replays a script of responses and records every request."""

    script: list[AIMessage]
    seen: list[list[BaseMessage]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:  # type: ignore[override]
        return self

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any) -> Any:
        raise NotImplementedError

    async def _agenerate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        self.seen.append(list(messages))
        msg = self.script.pop(0) if self.script else AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _call(service: str, i: int) -> dict[str, Any]:
    return {"name": "log_summary", "args": {"service": service}, "id": f"call-{i}"}


def sequential(n: int) -> list[AIMessage]:
    """One tool call per step for *n* services, then a final answer."""
    steps = [AIMessage(content="", tool_calls=[_call(s, i)]) for i, s in enumerate(SERVICES[:n])]
    return steps + [AIMessage(content="final answer")]


def text(messages: list[BaseMessage]) -> str:
    return "\n".join(str(m.content) for m in messages)


async def run_agent(messages: list[Any], script: list[AIMessage], **kwargs: Any) -> ScriptedModel:
    model = ScriptedModel(script=script, seen=[])
    agent = await build_agent(
        model=model,
        servers={},
        extra_tools=[log_summary],
        instructions=INSTRUCTIONS,
        **kwargs,
    )
    try:
        await agent.ainvoke({"messages": messages})
    finally:
        await agent.shutdown()
    return model


# ---------------------------------------------------------------------------
# 1. The question survives compaction, in every input form
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        {"role": "user", "content": QUESTION},
        HumanMessage(QUESTION),
        ("user", QUESTION),
    ],
    ids=["dict", "HumanMessage", "tuple"],
)
async def test_question_survives_a_long_tool_loop(question: Any) -> None:
    model = await run_agent([question], sequential(8))

    assert len(model.seen) == 9
    for i, request in enumerate(model.seen, 1):
        assert QUESTION in text(request), f"question missing from call {i}"
    # The loop really compacted from call 7 on, so this covers the ledger.
    assert LEDGER_HEADER not in text(model.seen[5])
    assert LEDGER_HEADER in text(model.seen[6])
    assert LEDGER_HEADER in text(model.seen[-1])


async def test_question_is_a_human_message_after_compaction() -> None:
    model = await run_agent([{"role": "user", "content": QUESTION}], sequential(8))
    humans = [m for m in model.seen[-1] if isinstance(m, HumanMessage)]
    assert [m.content for m in humans] == [QUESTION]


# ---------------------------------------------------------------------------
# 2. Multi-turn: the current question, not the session's first
# ---------------------------------------------------------------------------


async def test_history_input_keeps_the_current_question() -> None:
    first = "Which services have logs?"
    history = [
        {"role": "user", "content": first},
        {"role": "assistant", "content": "auth, billing, gateway and seven more."},
        {"role": "user", "content": QUESTION},
    ]
    model = await run_agent(history, sequential(8))

    for request in model.seen:
        humans = [m for m in request if isinstance(m, HumanMessage)]
        assert humans[-1].content == QUESTION
    compacted = model.seen[-1]
    # Earlier turns become a short note, not a second user question.
    assert [m.content for m in compacted if isinstance(m, HumanMessage)] == [QUESTION]
    note = next(
        m
        for m in compacted
        if isinstance(m, SystemMessage) and "Earlier in this conversation" in str(m.content)
    )
    assert first in str(note.content)


async def test_chat_sessions_answer_the_current_question() -> None:
    store = InMemoryConversationStore()
    model = ScriptedModel(script=[AIMessage(content="auth, billing, ...")] + sequential(8), seen=[])
    agent = await build_agent(
        model=model,
        servers={},
        extra_tools=[log_summary],
        instructions=INSTRUCTIONS,
        conversation_store=store,
    )
    try:
        await agent.chat("Which services have logs?", session_id="s1")
        await agent.chat(QUESTION, session_id="s1")
    finally:
        await agent.shutdown()

    second_turn = model.seen[1:]
    assert len(second_turn) == 9
    for request in second_turn:
        humans = [m for m in request if isinstance(m, HumanMessage)]
        assert humans[-1].content == QUESTION
    assert LEDGER_HEADER in text(second_turn[-1])


# ---------------------------------------------------------------------------
# 3. System and runtime-injected messages are pinned
# ---------------------------------------------------------------------------


async def test_system_messages_are_pinned() -> None:
    messages = [
        SystemMessage("Incident commander: Priya Raman."),
        {"role": "system", "content": "[Context State] {'ticket': 'INC-42'}"},
        {"role": "user", "content": QUESTION},
    ]
    model = await run_agent(messages, sequential(8))

    for i, request in enumerate(model.seen, 1):
        sent = text(request)
        assert "Priya Raman" in sent, f"system message missing from call {i}"
        assert "[Context State]" in sent, f"runtime context missing from call {i}"
    # The node's own instructions are still sent, once.
    assert text(model.seen[-1]).count(INSTRUCTIONS) == 1


# ---------------------------------------------------------------------------
# 4. Compaction shrinks: no duplicates, old results cut to an excerpt
# ---------------------------------------------------------------------------


async def test_parallel_batch_is_not_sent_twice() -> None:
    batch = AIMessage(content="", tool_calls=[_call(s, i) for i, s in enumerate(SERVICES)])
    model = await run_agent([HumanMessage(QUESTION)], [batch, AIMessage(content="final")])

    final = model.seen[-1]
    assert sum(1 for m in final if isinstance(m, ToolMessage)) == 10
    for service in SERVICES:
        assert text(final).count(f"{service}: 3 ERROR lines.") == 1
    raw = 10 * len(f"auth: 3 ERROR lines. {PADDING}")
    assert len(text(final)) < raw * 1.1


async def test_old_results_are_cut_to_an_excerpt_with_a_reference() -> None:
    model = await run_agent([HumanMessage(QUESTION)], sequential(10))

    final = model.seen[-1]
    ledger = next(m for m in final if LEDGER_HEADER in str(m.content))
    assert 'log_summary({"service": "auth"})' in str(ledger.content)
    assert "call log_summary with the same arguments" in str(ledger.content)
    # Older results show 2000 of their 6000+ characters.
    assert PADDING not in str(ledger.content)
    # Only the latest exchange is verbatim.
    assert [m.tool_call_id for m in final if isinstance(m, ToolMessage)] == ["call-9"]
    # Far smaller than the ten full results the transcript would carry.
    full = 10 * len(f"auth: 3 ERROR lines. {PADDING}")
    assert len(text(final)) < 0.6 * full


async def test_repeated_calls_are_listed_once() -> None:
    script = [AIMessage(content="", tool_calls=[_call("auth", i)]) for i in range(8)]
    model = await run_agent([HumanMessage(QUESTION)], script + [AIMessage(content="final")])
    ledger = next(m for m in model.seen[-1] if LEDGER_HEADER in str(m.content))
    assert str(ledger.content).count("log_summary(") == 1


# ---------------------------------------------------------------------------
# 5. Settings: off, threshold, token budget
# ---------------------------------------------------------------------------


async def test_compaction_can_be_turned_off() -> None:
    model = await run_agent(
        [{"role": "user", "content": QUESTION}], sequential(8), context_compaction=False
    )
    final = model.seen[-1]
    assert LEDGER_HEADER not in text(final)
    assert sum(1 for m in final if isinstance(m, ToolMessage)) == 8
    assert QUESTION in text(final)


async def test_compaction_threshold_is_configurable() -> None:
    model = await run_agent([HumanMessage(QUESTION)], sequential(4), context_compaction=2)
    assert LEDGER_HEADER not in text(model.seen[1])
    assert LEDGER_HEADER in text(model.seen[2])


async def test_token_budget_shrinks_entries_to_references() -> None:
    settings = ContextCompaction(after_tool_results=100, max_tokens=2_500)
    model = await run_agent([HumanMessage(QUESTION)], sequential(8), context_compaction=settings)
    final = text(model.seen[-1])
    assert LEDGER_HEADER in final  # compacted by the budget, not the count
    assert "[result not shown:" in final
    assert QUESTION in final


def test_coerce() -> None:
    assert ContextCompaction.coerce(None) == ContextCompaction()
    assert ContextCompaction.coerce(True) == ContextCompaction()
    assert not ContextCompaction.coerce(False).enabled
    assert ContextCompaction.coerce(3).after_tool_results == 3
    with pytest.raises(ValueError):
        ContextCompaction.coerce(0)
    with pytest.raises(TypeError):
        ContextCompaction.coerce("ledger")


def test_node_settings_win_over_the_engines() -> None:
    node = PromptNode("reason", context_scope="auto", compaction=False)
    state = GraphState(messages=[])
    for i in range(20):
        state.add_observation(tool_name="t", result="r", args={"i": i}, success=True)
    config = {"_engine_compaction": ContextCompaction(after_tool_results=1)}
    assert node._effective_context_scope(state, config) == "full"
    assert PromptNode("b", context_scope="auto")._effective_context_scope(state, config) == "ledger"


# ---------------------------------------------------------------------------
# 6. ContextEngine layers reach the model on every call
# ---------------------------------------------------------------------------


async def test_engine_layers_reach_every_call_and_persist() -> None:
    engine = ContextEngine(model="openai:gpt-5-mini")
    engine.add_layer("runbook", priority=7, required=True, content="Escalation: Priya Raman.")
    model = await run_agent([HumanMessage(QUESTION)], sequential(8), context_engine=engine)

    for i, request in enumerate(model.seen, 1):
        sent = text(request)
        assert "Priya Raman" in sent, f"layer missing from call {i}"
        assert QUESTION in sent, f"question missing from call {i}"
        assert sent.count(INSTRUCTIONS) == 1, f"instructions repeated in call {i}"
    assert engine.get_content("runbook") == "Escalation: Priya Raman."
    report = engine.last_report
    assert report is not None
    layers = {layer["name"]: layer for layer in report.layers}
    assert layers["tools"]["tokens"] > 0 and not layers["tools"]["sent"]
    assert not layers["identity"]["sent"]
    assert layers["runbook"]["sent"]


async def test_engine_keeps_input_system_messages() -> None:
    engine = ContextEngine(model="openai:gpt-5-mini")
    messages = [
        {"role": "system", "content": "[Context State] {'ticket': 'INC-42'}"},
        {"role": "user", "content": QUESTION},
    ]
    model = await run_agent(messages, sequential(8), context_engine=engine)
    for request in model.seen:
        assert "[Context State]" in text(request)


async def test_engine_without_builtin_layers_keeps_question_and_history() -> None:
    engine = ContextEngine(model="openai:gpt-5-mini", auto_register_builtins=False)
    engine.add_layer("runbook", priority=7, content="Escalation: Priya Raman.")
    history = [
        {"role": "user", "content": "Which services have logs?"},
        {"role": "assistant", "content": "auth, billing and eight more."},
        {"role": "user", "content": QUESTION},
    ]
    model = await run_agent(history, sequential(2), context_engine=engine)
    first = text(model.seen[0])
    assert "Priya Raman" in first
    assert "Which services have logs?" in first
    for request in model.seen:
        humans = [m for m in request if isinstance(m, HumanMessage)]
        assert humans[-1].content == QUESTION


class _QuarterCounter:
    def count(self, text: str) -> int:
        return len(text) // 4


async def test_engine_budget_bounds_the_tool_loop() -> None:
    engine = ContextEngine(
        model_context_window=4_000, response_reserve=1_000, tokenizer=_QuarterCounter()
    )
    model = await run_agent(
        [HumanMessage(QUESTION)],
        sequential(5),
        context_engine=engine,
        context_compaction=ContextCompaction(after_tool_results=100),
    )
    # Two 6k-character results already pass the 3k-token budget, so the
    # budget compacts the loop long before the 100-result threshold.
    assert LEDGER_HEADER in text(model.seen[-1])
    assert QUESTION in text(model.seen[-1])


def test_assemble_overrides_without_mutating() -> None:
    engine = ContextEngine(model_context_window=10_000)
    engine.add_layer("policy", priority=7, content="Be kind.")
    messages = engine.assemble(
        {"user_message": "hi", "identity": "I am an agent."}, budget_only=("identity",)
    )
    assert {"role": "user", "content": "hi"} in messages
    assert not any(m["content"] == "I am an agent." for m in messages)
    assert engine.get_content("user_message") == ""
    assert engine.get_content("policy") == "Be kind."
    with pytest.raises(KeyError):
        engine.assemble({"nope": "x"})


# ---------------------------------------------------------------------------
# 7. Custom graphs
# ---------------------------------------------------------------------------


async def test_single_node_graph_runs_once() -> None:
    graph = PromptGraph("one")  # default mode, no edges
    graph.add_node(PromptNode("answer", instructions="Answer briefly.", default_next="__end__"))
    graph.set_entry("answer")
    # A planner would happily keep choosing the only node.
    routing = AIMessage(content='{"next_node": "answer", "reason": "answer it"}')
    model = ScriptedModel(script=[routing] * 6, seen=[])
    agent = await build_agent(model=model, servers={}, agent_pattern=graph)
    try:
        await agent.ainvoke({"messages": [HumanMessage("hi")]})
    finally:
        await agent.shutdown()
    assert len(model.seen) == 1
    assert "Answer briefly." in text(model.seen[0])


async def test_inject_tools_node_gets_build_agent_tools() -> None:
    graph = PromptGraph("inject", mode="static")
    graph.add_node(PromptNode("reason", inject_tools=True, default_next="__end__"))
    graph.set_entry("reason")
    model = ScriptedModel(
        script=[AIMessage(content="", tool_calls=[_call("auth", 0)]), AIMessage(content="ok")],
        seen=[],
    )
    agent = await build_agent(
        model=model, servers={}, extra_tools=[log_summary], agent_pattern=graph
    )
    try:
        out = await agent.ainvoke({"messages": [HumanMessage("go")]})
    finally:
        await agent.shutdown()
    results = [m for m in out["messages"] if isinstance(m, ToolMessage)]
    assert results and str(results[0].content).startswith("auth: 3 ERROR lines.")


# ---------------------------------------------------------------------------
# 8. State and view building blocks
# ---------------------------------------------------------------------------


def test_graph_state_normalizes_dict_messages() -> None:
    state = GraphState(
        messages=[
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
    )
    assert [type(m) for m in state.messages] == [
        SystemMessage,
        HumanMessage,
        AIMessage,
        HumanMessage,
    ]
    assert state.turn_question is state.messages[3]
    assert state.turn_input == state.messages


def test_trim_keeps_the_question_and_no_orphan_tool_results() -> None:
    msgs: list[Any] = [SystemMessage("sys"), HumanMessage(QUESTION)]
    for i in range(30):
        msgs.append(AIMessage(content="", tool_calls=[_call("auth", i)]))
        msgs.append(ToolMessage(content=f"r{i}", tool_call_id=f"call-{i}"))
    state = GraphState(messages=msgs, max_messages=10)
    state.trim_messages()
    assert state.messages[0].content == "sys"
    assert state.messages[1].content == QUESTION
    assert not isinstance(state.messages[2], ToolMessage)
    assert len(state.messages) <= 10


def test_view_never_sends_a_tool_result_without_its_call() -> None:
    question = HumanMessage(QUESTION)
    msgs: list[Any] = [question, ToolMessage(content="orphan", tool_call_id="x")]
    q, head = split_input(msgs)
    view = build_compacted_view(msgs, question=q, head=head, settings=ContextCompaction())
    # Providers reject a tool message without its call; the result stays as ledger text.
    assert view[0] is question
    assert not any(isinstance(m, ToolMessage) for m in view)
    assert "orphan" in str(view[-1].content)


def test_scoped_view_keeps_system_messages_and_the_current_question() -> None:
    msgs: list[Any] = [
        SystemMessage("pinned"),
        HumanMessage("old question"),
        AIMessage("old answer"),
        HumanMessage(QUESTION),
        AIMessage(content="", tool_calls=[_call("auth", 0)]),
        ToolMessage(content="r0", tool_call_id="call-0"),
    ]
    q, head = split_input(msgs)
    view = build_compacted_view(
        msgs, question=q, head=head, settings=ContextCompaction(), scoped=True
    )
    assert view[0].content == "pinned"
    assert [m.content for m in view if isinstance(m, HumanMessage)] == [QUESTION]
    assert isinstance(view[-1], ToolMessage)
