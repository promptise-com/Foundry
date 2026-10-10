"""Regression tests for Reasoning Graph bugs fixed in 1.2.2.

Every test runs offline against a scripted chat model:

- custom graphs passed to ``build_agent`` inject the discovered tools
- ``agent_pattern`` names are validated (``"pipeline"`` and typos raise)
- ``peoatr`` terminates with a written answer
- a re-entered ``PromptNode`` rebuilds its prompt from fresh inputs
- a node's ``max_iterations`` is enforced
- ``ExecutionReport.total_tokens`` counts usage
- a Pydantic ``output_schema`` routes like a TypedDict
- YAML save/load round-trips, and refuses what it can't represent
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import Any, TypedDict

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from pydantic import BaseModel, PrivateAttr

from promptise.engine import (
    GraphState,
    NodeFlag,
    NodeResult,
    PromptGraph,
    PromptGraphEngine,
    PromptNode,
    node,
)
from promptise.engine.prebuilts import PeoatrPlan, PeoatrReflection, PeoatrThought
from promptise.engine.reasoning_nodes import PlanNode, ReflectNode, ThinkNode, ValidateNode
from promptise.engine.serialization import (
    GraphSerializationError,
    graph_from_config,
    graph_to_config,
    load_graph,
    node_from_config,
    node_to_config,
    save_graph,
)

SERVER_SCRIPT = os.path.join(os.path.dirname(__file__), "_e2e_reasoning_server.py")

Reply = Callable[[list[BaseMessage], list[str], Any], Any]


class ScriptedChat(BaseChatModel):
    """A chat model whose answers come from ``reply(messages, tool_names, schema)``.

    Records every call. Plain calls return an ``AIMessage`` (with usage);
    ``with_structured_output(schema, include_raw=True)`` wraps the reply in
    LangChain's ``{"raw", "parsed", "parsing_error"}`` envelope.
    """

    _reply: Any = PrivateAttr()
    _bound: list = PrivateAttr(default_factory=list)
    _calls: list = PrivateAttr(default_factory=list)

    def __init__(self, reply: Reply, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._reply = reply

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._calls

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChat:  # type: ignore[override]
        clone = ScriptedChat(self._reply)
        clone._calls = self._calls
        clone._bound = [t.name for t in tools]
        return clone

    def with_structured_output(self, schema: Any, *, include_raw: bool = False, **kw: Any) -> Any:  # type: ignore[override]
        outer = self

        class _Structured:
            async def ainvoke(self, messages: list[BaseMessage], config: Any = None) -> Any:
                outer._calls.append({"messages": messages, "tools": [], "schema": schema})
                parsed = outer._reply(messages, [], schema)
                raw = AIMessage(
                    content="",
                    usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                )
                if include_raw:
                    return {"raw": raw, "parsed": parsed, "parsing_error": None}
                return parsed

        return _Structured()

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[override]
        self._calls.append({"messages": messages, "tools": list(self._bound), "schema": None})
        msg = self._reply(messages, list(self._bound), None)
        if not isinstance(msg, AIMessage):
            msg = AIMessage(content=str(msg))
        if msg.usage_metadata is None:
            msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _system_text(messages: list[BaseMessage]) -> str:
    return "\n".join(str(m.content) for m in messages if isinstance(m, SystemMessage))


@tool
def lookup(query: str) -> str:
    """Look something up."""
    return f"result for {query}"


# Module-level so YAML can reference them.
class Verdict(BaseModel):
    passes: bool
    issues: list[str]


class VerdictDict(TypedDict):
    passes: bool
    issues: list[str]


async def double_it(state: GraphState, config: dict) -> None:
    """A preprocessor YAML can reference."""


@node("count_step", default_next="__end__")
async def count_step(state: GraphState) -> NodeResult:
    state.context["counted"] = True
    return NodeResult(node_name="count_step")


def needs_retry(result: NodeResult) -> bool:
    return bool(result.error)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. inject_tools in a custom graph passed to build_agent
# ═══════════════════════════════════════════════════════════════════════════════


class TestInjectTools:
    @pytest.mark.asyncio
    async def test_custom_graph_inject_tools_node_sees_discovered_mcp_tools(self, monkeypatch):
        from promptise import build_agent
        from promptise.config import StdioServerSpec

        monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")
        model = ScriptedChat(lambda m, tools, s: AIMessage(content=f"tools: {sorted(tools)}"))
        graph = PromptGraph("probe", mode="static")
        graph.add_node(PromptNode("act", inject_tools=True, default_next="__end__"))
        graph.set_entry("act")

        agent = await build_agent(
            model=model,
            servers={"tools": StdioServerSpec(command=sys.executable, args=[SERVER_SCRIPT])},
            agent_pattern=graph,
        )
        try:
            await agent.ainvoke({"messages": [HumanMessage(content="hi")]})
            discovered = sorted(agent.tool_names)
        finally:
            await agent.shutdown()

        assert discovered  # the server's tools were discovered
        assert sorted(model.calls[0]["tools"]) == discovered  # ...and bound for the node
        assert "Available tools:" in _system_text(model.calls[0]["messages"])

    @pytest.mark.asyncio
    async def test_engine_tools_param_feeds_inject_tools_nodes_only(self):
        model = ScriptedChat(lambda m, tools, s: AIMessage(content="done"))
        graph = PromptGraph("g", mode="static")
        graph.add_node(PromptNode("plain", default_next="injected"))
        graph.add_node(PromptNode("injected", inject_tools=True, default_next="__end__"))
        graph.set_entry("plain")

        engine = PromptGraphEngine(graph=graph, model=model, tools=[lookup])
        await engine.ainvoke({"messages": [HumanMessage(content="go")]})

        assert [c["tools"] for c in model.calls] == [[], ["lookup"]]

    @pytest.mark.asyncio
    async def test_node_own_tools_are_not_duplicated(self):
        model = ScriptedChat(lambda m, tools, s: AIMessage(content="done"))
        graph = PromptGraph("g", mode="static")
        graph.add_node(PromptNode("act", tools=[lookup], inject_tools=True, default_next="__end__"))
        graph.set_entry("act")

        await PromptGraphEngine(graph=graph, model=model, tools=[lookup]).ainvoke(
            {"messages": [HumanMessage(content="go")]}
        )
        assert model.calls[0]["tools"] == ["lookup"]


# ═══════════════════════════════════════════════════════════════════════════════
# 2 + 3. agent_pattern names
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentPatternNames:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["plan-act-verify", "ReAct", "copy", "validate"])
    async def test_unknown_pattern_raises_with_valid_names(self, name, monkeypatch):
        from promptise import build_agent
        from promptise.agent import AGENT_PATTERNS

        monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")
        model = ScriptedChat(lambda m, t, s: AIMessage(content="x"))
        with pytest.raises(ValueError, match="Unknown agent_pattern") as info:
            await build_agent(model=model, servers={}, agent_pattern=name)
        for valid in AGENT_PATTERNS:
            assert repr(valid) in str(info.value)

    @pytest.mark.asyncio
    async def test_pipeline_string_explains_how_to_build_one(self, monkeypatch):
        from promptise import build_agent

        monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")
        model = ScriptedChat(lambda m, t, s: AIMessage(content="x"))
        with pytest.raises(ValueError, match=r"PromptGraph\.pipeline\("):
            await build_agent(model=model, servers={}, agent_pattern="pipeline")

    @pytest.mark.asyncio
    async def test_pipeline_graph_works(self, monkeypatch):
        from promptise import build_agent

        monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")
        model = ScriptedChat(lambda m, t, s: AIMessage(content="ok"))
        graph = PromptGraph.pipeline(PromptNode("a"), PromptNode("b"))
        agent = await build_agent(model=model, servers={}, agent_pattern=graph)
        try:
            await agent.ainvoke({"messages": [HumanMessage(content="go")]})
            assert agent.last_report.nodes_visited == ["a", "b"]
        finally:
            await agent.shutdown()

    @pytest.mark.asyncio
    async def test_every_pattern_name_builds(self, monkeypatch):
        from promptise import build_agent
        from promptise.agent import AGENT_PATTERNS

        monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")
        model = ScriptedChat(lambda m, t, s: AIMessage(content="x"))
        for name in AGENT_PATTERNS:
            if name == "code-action":
                continue  # needs Docker
            agent = await build_agent(model=model, servers={}, agent_pattern=name)
            await agent.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
# 4. peoatr terminates
# ═══════════════════════════════════════════════════════════════════════════════


def _peoatr_reply(route_think: str, route_reflect: str) -> Reply:
    def reply(messages: list[BaseMessage], tools: list[str], schema: Any) -> Any:
        if schema is PeoatrPlan:
            return PeoatrPlan(subgoals=["look it up"], active_subgoal="look it up", quality_score=4)
        if schema is PeoatrThought:
            return PeoatrThought(analysis="found it", subgoal_complete=True, route=route_think)
        if schema is PeoatrReflection:
            return PeoatrReflection(
                progress="done", mistake="", correction="", confidence=5, route=route_reflect
            )
        if tools and not any(
            getattr(m, "type", None) == "tool" for m in messages[-1:]
        ):  # act: call the tool once, then summarize
            return AIMessage(
                content="", tool_calls=[{"name": "lookup", "args": {"query": "x"}, "id": "c1"}]
            )
        if tools:
            return AIMessage(content="The tool says: result for x")
        return AIMessage(content="FINAL ANSWER: result for x")

    return reply


class TestPeoatr:
    @pytest.mark.asyncio
    async def test_peoatr_answers_and_ends(self):
        model = ScriptedChat(_peoatr_reply("reflect", "answer"))
        engine = PromptGraphEngine(graph=PromptGraph.peoatr(tools=[lookup]), model=model)
        result = await engine.ainvoke({"messages": [HumanMessage(content="What is x?")]})

        assert engine.last_report.nodes_visited == [
            "plan",
            "act",
            "act",
            "think",
            "reflect",
            "answer",
        ]
        assert result["messages"][-1].content == "FINAL ANSWER: result for x"
        assert engine.last_report.tool_calls == 1

    @pytest.mark.asyncio
    async def test_peoatr_ends_with_an_answer_when_the_model_never_says_answer(self):
        model = ScriptedChat(_peoatr_reply("continue", "continue"))
        engine = PromptGraphEngine(graph=PromptGraph.peoatr(tools=[lookup]), model=model)
        result = await engine.ainvoke({"messages": [HumanMessage(content="What is x?")]})

        visited = engine.last_report.nodes_visited
        assert visited[-1] == "answer"
        assert result["messages"][-1].content == "FINAL ANSWER: result for x"
        assert len(visited) < 50  # bounded by per-node budgets, not the engine cap

    @pytest.mark.asyncio
    async def test_peoatr_replan_routes_to_plan(self):
        routes = iter(["replan", "answer"])

        base = _peoatr_reply("reflect", "answer")

        def reply(messages, tools, schema):
            if schema is PeoatrReflection:
                return PeoatrReflection(
                    progress="p", mistake="m", correction="c", confidence=2, route=next(routes)
                )
            return base(messages, tools, schema)

        engine = PromptGraphEngine(
            graph=PromptGraph.peoatr(tools=[lookup]), model=ScriptedChat(reply)
        )
        await engine.ainvoke({"messages": [HumanMessage(content="What is x?")]})
        visited = engine.last_report.nodes_visited
        assert visited.count("plan") == 2
        assert visited[-1] == "answer"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. re-entered PromptNode rebuilds its prompt
# ═══════════════════════════════════════════════════════════════════════════════


def _echo_note(messages: list[BaseMessage], tools: list[str], schema: Any) -> AIMessage:
    for line in _system_text(messages).splitlines():
        if line.startswith("note: "):
            return AIMessage(content=line.removeprefix("note: "))
    return AIMessage(content="none")


class TestPromptRebuiltOnReentry:
    @pytest.mark.asyncio
    async def test_input_keys_are_fresh_on_reentry(self):
        @node("bump")
        async def bump(state: GraphState) -> NodeResult:
            state.context["round"] = state.context.get("round", 0) + 1
            state.context["note"] = ["apple", "banana"][state.context["round"] - 1]
            return NodeResult(node_name="bump", next_node="speak")

        @node("check")
        async def check(state: GraphState) -> NodeResult:
            done = state.context["round"] >= 2
            return NodeResult(node_name="check", next_node="__end__" if done else "bump")

        graph = PromptGraph("reentry", mode="static")
        graph.add_node(bump)
        graph.add_node(PromptNode("speak", input_keys=["note"], default_next="check"))
        graph.add_node(check)
        graph.set_entry("bump")

        result = await PromptGraphEngine(graph=graph, model=ScriptedChat(_echo_note)).ainvoke(
            {"messages": [HumanMessage(content="Say the note.")]}
        )
        replies = [m.content for m in result["messages"] if isinstance(m, AIMessage)]
        assert replies == ["apple", "banana"]

    @pytest.mark.asyncio
    async def test_plan_and_reflections_are_fresh_on_reentry(self):
        seen: list[str] = []

        def reply(messages, tools, schema):
            seen.append(_system_text(messages))
            return AIMessage(content="ok")

        @node("learn")
        async def learn(state: GraphState) -> NodeResult:
            n = len(state.reflections)
            state.plan = [f"step {n}"]
            state.add_reflection(iteration=n, mistake=f"mistake {n}", correction="fix")
            return NodeResult(node_name="learn", next_node="speak" if n < 2 else "__end__")

        graph = PromptGraph("reentry", mode="static")
        graph.add_node(learn)
        graph.add_node(PromptNode("speak", default_next="learn"))
        graph.set_entry("learn")
        await PromptGraphEngine(graph=graph, model=ScriptedChat(reply)).ainvoke(
            {"messages": [HumanMessage(content="go")]}
        )

        assert "step 0" in seen[0] and "step 1" not in seen[0]
        assert "step 1" in seen[1] and "mistake 1" in seen[1]

    @pytest.mark.asyncio
    async def test_tool_loop_reuses_binding(self):
        binds: list[int] = []

        class CountingChat(ScriptedChat):
            def bind_tools(self, tools, **kwargs):  # type: ignore[override]
                binds.append(1)
                return super().bind_tools(tools, **kwargs)

        def reply(messages, tools, schema):
            if not any(getattr(m, "type", None) == "tool" for m in messages):
                return AIMessage(
                    content="", tool_calls=[{"name": "lookup", "args": {"query": "q"}, "id": "1"}]
                )
            return AIMessage(content="done")

        graph = PromptGraph.react(tools=[lookup])
        await PromptGraphEngine(graph=graph, model=CountingChat(reply)).ainvoke(
            {"messages": [HumanMessage(content="go")]}
        )
        assert len(binds) == 1  # two executions of "reason", one binding


# ═══════════════════════════════════════════════════════════════════════════════
# 6. per-node max_iterations
# ═══════════════════════════════════════════════════════════════════════════════


class TestNodeMaxIterations:
    @pytest.mark.asyncio
    async def test_loop_stops_running_a_node_at_its_max_iterations(self):
        graph = PromptGraph("loop", mode="static")
        graph.add_node(PromptNode("act", default_next="verify"))
        graph.add_node(PromptNode("verify", max_iterations=3, default_next="act"))
        graph.set_entry("act")

        engine = PromptGraphEngine(
            graph=graph, model=ScriptedChat(lambda m, t, s: "x"), max_iterations=50
        )
        await engine.ainvoke({"messages": [HumanMessage(content="go")]})
        assert engine.last_report.nodes_visited.count("verify") == 3

    @pytest.mark.asyncio
    async def test_exhausted_node_follows_its_error_transition(self):
        graph = PromptGraph("loop", mode="static")
        graph.add_node(PromptNode("act", default_next="verify"))
        graph.add_node(
            PromptNode(
                "verify", max_iterations=2, default_next="act", transitions={"error": "give_up"}
            )
        )
        graph.add_node(PromptNode("give_up", default_next="__end__"))
        graph.set_entry("act")

        engine = PromptGraphEngine(graph=graph, model=ScriptedChat(lambda m, t, s: "x"))
        await engine.ainvoke({"messages": [HumanMessage(content="go")]})
        assert engine.last_report.nodes_visited == [
            "act",
            "verify",
            "act",
            "verify",
            "act",
            "give_up",
        ]

    @pytest.mark.asyncio
    async def test_engine_max_node_iterations_still_caps_nodes_with_a_higher_budget(self):
        graph = PromptGraph("loop", mode="static")
        graph.add_node(PromptNode("spin", max_iterations=100, default_next="spin"))
        graph.set_entry("spin")
        engine = PromptGraphEngine(
            graph=graph, model=ScriptedChat(lambda m, t, s: "x"), max_node_iterations=4
        )
        await engine.ainvoke({"messages": [HumanMessage(content="go")]})
        assert engine.last_report.nodes_visited == ["spin"] * 4

    @pytest.mark.asyncio
    async def test_streaming_enforces_the_budget_too(self):
        graph = PromptGraph("loop", mode="static")
        graph.add_node(PromptNode("act", default_next="verify"))
        graph.add_node(PromptNode("verify", max_iterations=2, default_next="act"))
        graph.set_entry("act")
        engine = PromptGraphEngine(graph=graph, model=ScriptedChat(lambda m, t, s: "x"))
        starts = [
            e["name"]
            async for e in engine.astream_events({"messages": [HumanMessage(content="go")]})
            if e["event"] == "on_node_start" and "node_type" in e["data"]  # engine-level
        ]
        assert starts.count("verify") == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 7. ExecutionReport.total_tokens
# ═══════════════════════════════════════════════════════════════════════════════


class TestTokenAccounting:
    @pytest.mark.asyncio
    async def test_report_total_tokens_counts_message_usage(self):
        graph = PromptGraph.pipeline(PromptNode("a"), PromptNode("b"))
        engine = PromptGraphEngine(graph=graph, model=ScriptedChat(lambda m, t, s: "x"))
        await engine.ainvoke({"messages": [HumanMessage(content="go")]})
        assert engine.last_report.total_tokens == 30  # 2 calls × 15

    @pytest.mark.asyncio
    async def test_structured_output_usage_is_counted(self):
        validate = ValidateNode("verify", output_schema=Verdict, on_pass="__end__")
        graph = PromptGraph("g", mode="static")
        graph.add_node(validate)
        graph.set_entry("verify")
        engine = PromptGraphEngine(
            graph=graph, model=ScriptedChat(lambda m, t, s: Verdict(passes=True, issues=[]))
        )
        await engine.ainvoke({"messages": [HumanMessage(content="go")]})
        assert engine.last_report.total_tokens == 10

    def test_token_usage_accepts_dicts_and_objects(self):
        from types import SimpleNamespace

        from promptise.engine.nodes import token_usage

        msg = AIMessage(
            content="x", usage_metadata={"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}
        )
        assert token_usage(msg) == (3, 4)
        obj = SimpleNamespace(usage_metadata=SimpleNamespace(input_tokens=1, output_tokens=2))
        assert token_usage(obj) == (1, 2)
        assert token_usage(AIMessage(content="x")) == (0, 0)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Pydantic output_schema routes like a TypedDict
# ═══════════════════════════════════════════════════════════════════════════════


class TestPydanticOutputSchema:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("schema", [Verdict, VerdictDict])
    @pytest.mark.parametrize("passes,expected", [(True, "__end__"), (False, "act")])
    async def test_validate_node_routes(self, schema, passes, expected):
        def reply(m, t, s):
            if s is Verdict:
                return Verdict(passes=passes, issues=[])
            return {"passes": passes, "issues": []}

        node_ = ValidateNode("verify", output_schema=schema, on_pass="__end__", on_fail="act")
        state = GraphState(messages=[HumanMessage(content="2+2?"), AIMessage(content="4")])
        result = await node_.execute(state, {"_engine_model": ScriptedChat(reply)})

        assert result.output == {"passes": passes, "issues": []}
        assert result.next_node == expected
        assert state.context["validation"] == {"passes": passes, "issues": []}

    @pytest.mark.asyncio
    async def test_plan_node_with_pydantic_schema_updates_the_plan_and_replans(self):
        class Plan(BaseModel):
            subgoals: list[str]
            quality_score: int

        node_ = PlanNode("plan", output_schema=Plan, transitions={"replan": "plan"})
        state = GraphState(messages=[HumanMessage(content="go")])
        model = ScriptedChat(lambda m, t, s: Plan(subgoals=["a", "b"], quality_score=1))
        result = await node_.execute(state, {"_engine_model": model})

        assert state.plan == ["a", "b"]
        assert result.next_node == "plan"  # quality 1 < threshold 3

    @pytest.mark.asyncio
    async def test_route_field_follows_transition_keys(self):
        class Route(BaseModel):
            route: str

        graph = PromptGraph("g", mode="static")
        graph.add_node(
            ReflectNode(
                "reflect",
                output_schema=Route,
                transitions={"answer": "write", "continue": "reflect"},
            )
        )
        graph.add_node(PromptNode("write", default_next="__end__"))
        graph.set_entry("reflect")

        def reply(m, t, s):
            return Route(route="answer") if s is Route else AIMessage(content="done")

        engine = PromptGraphEngine(graph=graph, model=ScriptedChat(reply))
        await engine.ainvoke({"messages": [HumanMessage(content="go")]})
        assert engine.last_report.nodes_visited == ["reflect", "write"]

    @pytest.mark.asyncio
    async def test_parsing_error_is_a_node_failure(self):
        class Strict(ScriptedChat):
            def with_structured_output(self, schema, *, include_raw=False, **kw):  # type: ignore[override]
                class _S:
                    async def ainvoke(self, messages, config=None):
                        return {
                            "raw": AIMessage(content="not json"),
                            "parsed": None,
                            "parsing_error": ValueError("bad json"),
                        }

                return _S()

        node_ = ValidateNode("verify", output_schema=Verdict)
        state = GraphState(messages=[HumanMessage(content="go")])
        result = await node_.execute(
            state, {"_engine_model": Strict(lambda m, t, s: AIMessage(content=""))}
        )
        assert result.error and "bad json" in result.error


# ═══════════════════════════════════════════════════════════════════════════════
# 9. YAML round trip
# ═══════════════════════════════════════════════════════════════════════════════


def _rich_graph() -> PromptGraph:
    graph = PromptGraph("rich", mode="static")
    graph.add_node(
        PlanNode(
            "plan",
            instructions="Plan it.",
            output_schema=VerdictDict,
            max_subgoals=3,
            quality_threshold=4,
            is_entry=True,
        )
    )
    graph.add_node(
        PromptNode(
            "act",
            instructions="Act.",
            tools=[lookup],
            inject_tools=True,
            input_keys=["plan_output"],
            output_key="draft",
            inherit_context_from="plan",
            context_scope="scoped",
            preprocessor=double_it,
            include_plan=False,
            temperature=0.3,
            model_override="openai:gpt-5-mini",
            flags={NodeFlag.CACHEABLE, NodeFlag.RETRYABLE},
            max_iterations=4,
        )
    )
    graph.add_node(
        ValidateNode(
            "verify",
            criteria=["It is right"],
            output_schema=Verdict,
            on_pass="__end__",
            on_fail="reflect",
            max_iterations=3,
        )
    )
    graph.add_node(ReflectNode("reflect", input_keys=["validation"], review_depth=5))
    graph.add_node(ThinkNode("think", focus_areas=["cost"]))
    graph.add_node(count_step)
    graph.sequential("plan", "act", "verify")
    graph.always("reflect", "act")
    graph.on_output("think", "act", "ready", True)
    graph.on_confidence("think", "count_step", 0.8)
    graph.when("act", "reflect", condition=needs_retry, label="retry")
    graph.set_entry("plan")
    return graph


class TestYamlRoundTrip:
    def test_rich_graph_round_trips(self, tmp_path):
        graph = _rich_graph()
        path = tmp_path / "graph.yaml"
        save_graph(graph, str(path))

        loaded = load_graph(
            str(path),
            tools=[lookup],
            refs=[Verdict, VerdictDict, double_it, needs_retry, count_step],
        )
        assert graph_to_config(loaded) == graph_to_config(graph)

        act = loaded.get_node("act")
        assert isinstance(act, PromptNode)
        assert act.tools == [lookup]
        assert act.inject_tools
        assert act.input_keys == ["plan_output"]
        assert act.output_key == "draft"
        assert act.preprocessor is double_it
        assert act.flags == graph.get_node("act").flags
        verify = loaded.get_node("verify")
        assert verify.output_schema is Verdict
        assert (verify._on_pass, verify._on_fail) == ("__end__", "reflect")
        assert verify.instructions == graph.get_node("verify").instructions
        assert loaded.get_node("count_step") is count_step
        assert loaded.mode == "static"
        conditions = {(e.from_node, e.to_node): e.condition for e in loaded.edges}
        assert conditions[("act", "reflect")] is needs_retry
        assert conditions[("think", "act")](NodeResult(node_name="t", output={"ready": True}))

    def test_tools_are_stored_by_name_and_required_on_load(self, tmp_path):
        graph = PromptGraph.react(tools=[lookup])
        config = graph_to_config(graph)
        assert config["nodes"]["reason"]["tools"] == ["lookup"]

        with pytest.raises(GraphSerializationError, match=r"tools \['lookup'\]"):
            graph_from_config(config)
        assert graph_from_config(config, tools={"lookup": lookup}).get_node("reason").tools == [
            lookup
        ]

    def test_references_need_refs_or_allow_imports(self):
        config = graph_to_config(_rich_graph())
        with pytest.raises(GraphSerializationError, match="refs="):
            graph_from_config(config, tools=[lookup])
        loaded = graph_from_config(config, tools=[lookup], allow_imports=True)
        assert loaded.get_node("verify").output_schema is Verdict

    def test_promptise_references_resolve_without_refs(self):
        graph = PromptGraph.peoatr(tools=[lookup])
        loaded = graph_from_config(graph_to_config(graph), tools=[lookup])
        assert loaded.get_node("plan").output_schema is PeoatrPlan

    @pytest.mark.parametrize(
        "build",
        [
            lambda: PromptGraph.react(tools=[lookup]),
            lambda: PromptGraph.managed(tools=[lookup]),
            lambda: PromptGraph.verify(tools=[lookup]),
            lambda: PromptGraph.peoatr(tools=[lookup]),
            lambda: PromptGraph.research(search_tools=[lookup]),
            lambda: PromptGraph.deliberate(tools=[lookup]),
            lambda: PromptGraph.debate(),
            lambda: PromptGraph.autonomous(tools=[lookup]),
            lambda: PromptGraph.pipeline(PromptNode("a"), PromptNode("b")),
        ],
    )
    def test_prebuilt_graphs_round_trip(self, build):
        config = graph_to_config(build())
        assert graph_to_config(graph_from_config(config, tools=[lookup])) == config

    def test_lambda_condition_refuses_to_save(self, tmp_path):
        graph = PromptGraph.react(tools=[])
        graph.when("reason", "__end__", condition=lambda r: True)
        path = tmp_path / "g.yaml"
        with pytest.raises(GraphSerializationError, match="lambda"):
            save_graph(graph, str(path))
        assert not path.exists()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"output_schema": TypedDict("Local", {"a": int})},
            {"postprocessor": lambda out, state, config: out},
            {"guards": [object()]},
            {"model_override": ScriptedChat(lambda m, t, s: "x")},
        ],
    )
    def test_unrepresentable_settings_refuse_to_save(self, kwargs):
        with pytest.raises(GraphSerializationError, match="Node 'n'"):
            node_to_config(PromptNode("n", **kwargs))

    def test_unregistered_node_class_refuses_to_save(self):
        class Custom(PromptNode):
            pass

        with pytest.raises(GraphSerializationError, match="register_node_type"):
            node_to_config(Custom("n"))

    def test_registered_subclass_with_own_params_needs_hooks(self, monkeypatch):
        from promptise.engine import serialization as ser
        from promptise.engine.serialization import register_node_type

        ser._ensure_registry()
        monkeypatch.setattr(ser, "_NODE_TYPES", dict(ser._NODE_TYPES))

        class Scored(PromptNode):
            def __init__(self, name: str, *, threshold: float = 0.5, **kwargs: Any) -> None:
                super().__init__(name, **kwargs)
                self.threshold = threshold

        register_node_type("scored_test", Scored)
        with pytest.raises(GraphSerializationError, match="threshold"):
            node_to_config(Scored("n", threshold=0.9))

        class Hooked(Scored):
            def to_config(self) -> dict[str, Any]:
                return {"threshold": self.threshold, "instructions": self.instructions}

            @classmethod
            def from_config(cls, config: dict[str, Any]) -> Hooked:
                return cls(
                    config["name"],
                    threshold=config["threshold"],
                    instructions=config.get("instructions", ""),
                )

        register_node_type("hooked_test", Hooked)
        loaded = node_from_config(node_to_config(Hooked("n", threshold=0.9, instructions="x")))
        assert isinstance(loaded, Hooked) and loaded.threshold == 0.9

    def test_unknown_fields_and_types_raise_on_load(self):
        with pytest.raises(GraphSerializationError, match="unknown field 'tool_list'"):
            node_from_config({"name": "n", "type": "prompt", "tool_list": ["a"]})
        with pytest.raises(GraphSerializationError, match="strategy: can't be loaded"):
            node_from_config({"name": "n", "type": "prompt", "strategy": "ChainOfThought"})
        with pytest.raises(GraphSerializationError, match="unknown node type"):
            node_from_config({"name": "n", "type": "nope"})

    def test_hand_written_yaml_flags_still_load(self):
        loaded = node_from_config(
            {"name": "search", "type": "prompt", "inject_tools": True, "default_next": "observe"}
        )
        assert loaded.inject_tools and loaded.default_next == "observe"

    def test_every_constructor_parameter_is_covered(self):
        """A new node parameter must get a serializer field (or an explicit
        decision), or saving would silently drop it."""
        from promptise.engine import serialization as ser

        ser._ensure_registry()
        for cls, spec in ser._SPECS.items():
            type_name = cls.__name__
            known = (
                {f.key for f in spec.fields}
                | {f.key for f in ser._BASE_FIELDS}
                | ser._ABSORBED_PARAMS
            )
            uncovered = [p for p in ser._constructor_params(cls) if p not in known]
            assert not uncovered, f"{type_name}: {uncovered}"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Smaller pieces of the same fixes
# ═══════════════════════════════════════════════════════════════════════════════


class TestRoutingHint:
    @pytest.mark.asyncio
    async def test_hint_lists_transition_keys_and_never_the_error_route(self):
        class Route(BaseModel):
            route: str

        seen: list[str] = []

        def reply(messages, tools, schema):
            seen.append(_system_text(messages))
            return Route(route="answer")

        node_ = PromptNode(
            "decide",
            output_schema=Route,
            transitions={"answer": "write", "replan": "plan", "error": "give_up"},
        )
        await node_.execute(
            GraphState(messages=[HumanMessage(content="go")]),
            {"_engine_model": ScriptedChat(reply)},
        )
        hint = next(line for line in seen[0].splitlines() if line.startswith("Available next"))
        assert "answer (→ write)" in hint and "replan (→ plan)" in hint
        assert "error" not in hint and "give_up" not in hint
        assert "one of: answer, replan." in seen[0]


class TestGraphCopy:
    def test_copy_keeps_the_mode(self):
        # The engine runs a copy of the graph; YAML saves graph.mode.
        assert PromptGraph("g", mode="static").copy().mode == "static"


class TestCodeActionTokens:
    @pytest.mark.asyncio
    async def test_code_action_counts_dict_usage_metadata(self):
        from promptise.engine import CodeActionNode
        from promptise.sandbox.session import CommandResult

        class Session:
            async def execute(self, command, timeout=None, workdir=None):
                if command.startswith("python3"):
                    return CommandResult(0, "RESULT: 42\n", "")
                return CommandResult(0, "", "")

            async def list_files(self, directory):
                return []

            async def cleanup(self):
                pass

        async def factory():
            return Session()

        class Model:
            async def ainvoke(self, messages, config=None):
                return AIMessage(
                    content="```python\nprint('RESULT:', 42)\n```",
                    usage_metadata={"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
                )

        node_ = CodeActionNode("reason", sandbox_factory=factory)
        result = await node_.execute(
            GraphState(messages=[HumanMessage(content="6 times 7?")]), {"_engine_model": Model()}
        )
        assert result.output == "42"
        assert (result.prompt_tokens, result.completion_tokens) == (11, 4)
