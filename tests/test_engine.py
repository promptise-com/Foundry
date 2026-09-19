"""Tests for the PromptGraph engine.

Tests graph construction, node execution, transitions, hooks,
serialization, and prebuilt patterns. Uses real assertions,
not mocks.
"""

from __future__ import annotations

import pytest

from promptise.engine import (
    CycleDetectionHook,
    ExecutionReport,
    GraphMutation,
    GraphState,
    GuardNode,
    LoggingHook,
    NodeFlag,
    NodeResult,
    PromptGraph,
    PromptGraphEngine,
    PromptNode,
    TransformNode,
    node,
)
from promptise.engine.reasoning_nodes import (
    CritiqueNode,
    FanOutNode,
    JustifyNode,
    ObserveNode,
    PlanNode,
    ReflectNode,
    RetryNode,
    SynthesizeNode,
    ThinkNode,
    ValidateNode,
)
from promptise.engine.serialization import (
    graph_from_config,
    graph_to_config,
    node_from_config,
    node_to_config,
)
from promptise.engine.skills import (
    planner,
    summarizer,
    web_researcher,
)
from promptise.prompts.blocks import (
    ObservationBlock,
    PhaseBlock,
    PlanBlock,
    ReflectionBlock,
    SimpleBlock,
    block,
)

# ═══════════════════════════════════════════════════════════════════════════════
# State
# ═══════════════════════════════════════════════════════════════════════════════


class TestGraphState:
    def test_initial_state(self):
        state = GraphState()
        assert state.messages == []
        assert state.context == {}
        assert state.iteration == 0
        assert state.tool_calls_made == 0

    def test_add_observation(self):
        state = GraphState()
        state.add_observation("search", "found 3 results", {"query": "test"})
        assert len(state.observations) == 1
        assert state.observations[0]["tool"] == "search"
        assert state.tool_calls_made == 1

    def test_add_reflection(self):
        state = GraphState()
        state.add_reflection(1, "wrong tool", "use search instead")
        assert len(state.reflections) == 1
        assert state.reflections[0]["mistake"] == "wrong tool"

    def test_reflections_capped_at_5(self):
        state = GraphState()
        for i in range(10):
            state.add_reflection(i, f"mistake {i}", f"fix {i}")
        assert len(state.reflections) == 5
        assert state.reflections[0]["iteration"] == 5  # Oldest kept

    def test_complete_subgoal(self):
        state = GraphState()
        state.plan = ["step 1", "step 2", "step 3"]
        state.complete_subgoal("step 1")
        assert "step 1" in state.completed
        assert state.active_subgoal == "step 2"

    def test_all_subgoals_complete(self):
        state = GraphState()
        state.plan = ["a", "b"]
        assert not state.all_subgoals_complete
        state.complete_subgoal("a")
        state.complete_subgoal("b")
        assert state.all_subgoals_complete

    def test_node_timing(self):
        state = GraphState()
        state.record_node_timing("a", 100.0)
        state.record_node_timing("a", 50.0)
        assert state.node_timings["a"] == 150.0

    def test_increment_node_iteration(self):
        state = GraphState()
        assert state.increment_node_iteration("a") == 1
        assert state.increment_node_iteration("a") == 2
        assert state.increment_node_iteration("b") == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Graph
# ═══════════════════════════════════════════════════════════════════════════════


class TestPromptGraph:
    def test_add_node(self):
        g = PromptGraph("test")
        g.add_node(PromptNode("a"))
        assert g.has_node("a")

    def test_remove_node(self):
        g = PromptGraph("test")
        g.add_node(PromptNode("a"))
        g.add_node(PromptNode("b"))
        g.add_edge("a", "b")
        g.remove_node("a")
        assert not g.has_node("a")
        assert len(g.edges) == 0  # Edge removed too

    def test_set_entry(self):
        g = PromptGraph("test")
        g.add_node(PromptNode("a"))
        g.set_entry("a")
        assert g.entry == "a"

    def test_set_entry_nonexistent_raises(self):
        g = PromptGraph("test")
        with pytest.raises(KeyError):
            g.set_entry("nonexistent")

    def test_get_node_raises_on_missing(self):
        g = PromptGraph("test")
        with pytest.raises(KeyError, match="not found"):
            g.get_node("nonexistent")

    def test_add_edge(self):
        g = PromptGraph("test")
        g.add_node(PromptNode("a"))
        g.add_node(PromptNode("b"))
        g.add_edge("a", "b")
        assert len(g.edges) == 1
        assert g.edges[0].from_node == "a"
        assert g.edges[0].to_node == "b"

    def test_edge_helpers(self):
        g = PromptGraph("test")
        g.add_node(PromptNode("a"))
        g.add_node(PromptNode("b"))
        g.add_node(PromptNode("c"))
        g.always("a", "b")
        g.when("b", "c", lambda r: True, label="test_condition")
        assert len(g.edges) == 2

    def test_on_output_edge(self):
        g = PromptGraph("test")
        g.add_node(PromptNode("a"))
        g.add_node(PromptNode("b"))
        g.on_output("a", "b", "done", True)
        edge = g.edges[0]
        assert edge.label == "done=True"
        # Test the condition
        result = NodeResult(output={"done": True})
        assert edge.condition(result) is True
        result2 = NodeResult(output={"done": False})
        assert edge.condition(result2) is False

    def test_copy(self):
        g = PromptGraph("test")
        g.add_node(PromptNode("a"))
        g.set_entry("a")
        copy = g.copy()
        assert copy.name == g.name
        assert copy.entry == g.entry
        assert copy.has_node("a")
        # Modifying copy doesn't affect original
        copy.add_node(PromptNode("b"))
        assert not g.has_node("b")

    def test_validate_empty_graph(self):
        g = PromptGraph("test")
        errors = g.validate()
        assert any("No entry node" in e for e in errors)

    def test_validate_clean_graph(self):
        g = PromptGraph("test")
        g.add_node(PromptNode("a", default_next="__end__"))
        g.set_entry("a")
        errors = g.validate()
        assert errors == []

    def test_validate_unreachable_node(self):
        g = PromptGraph("test")
        g.add_node(PromptNode("a"))
        g.add_node(PromptNode("orphan"))
        g.set_entry("a")
        errors = g.validate()
        assert any("unreachable" in e for e in errors)

    def test_describe(self):
        g = PromptGraph("test")
        g.add_node(PromptNode("a", default_next="__end__"))
        g.set_entry("a")
        desc = g.describe()
        assert "test" in desc
        assert "[entry] a" in desc

    def test_to_mermaid(self):
        g = PromptGraph("test")
        g.add_node(PromptNode("a"))
        g.add_node(PromptNode("b"))
        g.always("a", "b")
        g.set_entry("a")
        mermaid = g.to_mermaid()
        assert "a --> b" in mermaid

    def test_apply_mutation_add_node(self):
        g = PromptGraph("test")
        g.apply_mutation(
            GraphMutation(
                action="add_node",
                node_config={"name": "new_node", "instructions": "test"},
            )
        )
        assert g.has_node("new_node")

    def test_apply_mutation_remove_node(self):
        g = PromptGraph("test")
        g.add_node(PromptNode("a"))
        g.apply_mutation(GraphMutation(action="remove_node", node_name="a"))
        assert not g.has_node("a")


# ═══════════════════════════════════════════════════════════════════════════════
# Blocks
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgenticBlocks:
    def test_simple_block(self):
        b = SimpleBlock("test", "Hello world", priority=5)
        assert b.name == "test"
        assert b.priority == 5
        assert b.render() == "Hello world"

    def test_simple_block_callable(self):
        b = SimpleBlock("dynamic", lambda ctx: "computed", priority=3)
        assert b.render() == "computed"

    def test_block_decorator(self):
        @block("my_rules", priority=9)
        def my_rules(ctx=None):
            return "Be helpful"

        assert my_rules.name == "my_rules"
        assert my_rules.priority == 9
        assert my_rules.render() == "Be helpful"

    def test_observation_block_empty(self):
        b = ObservationBlock()
        assert b.render() == ""

    def test_observation_block_with_data(self):
        b = ObservationBlock(
            [
                {"tool": "search", "result": "found 3 results", "success": True},
                {"tool": "calc", "result": "42", "success": True},
            ]
        )
        rendered = b.render()
        assert "search" in rendered
        assert "calc" in rendered
        assert "✓" in rendered

    def test_plan_block(self):
        b = PlanBlock(
            subgoals=["Step 1", "Step 2", "Step 3"],
            completed=["Step 1"],
            active="Step 2",
        )
        rendered = b.render()
        assert "[✓] Step 1" in rendered
        assert "[→] Step 2" in rendered
        assert "[ ] Step 3" in rendered
        assert "1/3 complete" in rendered

    def test_reflection_block(self):
        b = ReflectionBlock(
            [
                {"iteration": 1, "mistake": "wrong tool", "correction": "use search"},
            ]
        )
        rendered = b.render()
        assert "wrong tool" in rendered
        assert "use search" in rendered

    def test_phase_block(self):
        b = PhaseBlock(
            instructions={"plan": "Create a plan", "act": "Execute it"},
            current_phase="plan",
        )
        rendered = b.render()
        assert "plan" in rendered
        assert "Create a plan" in rendered

    def test_phase_block_switch(self):
        b = PhaseBlock(instructions={"a": "Phase A", "b": "Phase B"}, current_phase="a")
        b2 = b.set_phase("b")
        assert "Phase B" in b2.render()


# ═══════════════════════════════════════════════════════════════════════════════
# Nodes
# ═══════════════════════════════════════════════════════════════════════════════


class TestPromptNode:
    def test_construction(self):
        n = PromptNode(
            "test",
            instructions="Do something",
            input_keys=["query"],
            output_key="result",
            include_observations=False,
        )
        assert n.name == "test"
        assert n.input_keys == ["query"]
        assert n.output_key == "result"
        assert n.include_observations is False

    def test_from_config(self):
        n = PromptNode.from_config(
            {
                "name": "test",
                "instructions": "Hello",
                "input_keys": ["a", "b"],
                "output_key": "c",
                "temperature": 0.7,
                "include_plan": False,
            }
        )
        assert n.name == "test"
        assert n.input_keys == ["a", "b"]
        assert n.output_key == "c"
        assert n.temperature == 0.7
        assert n.include_plan is False


class TestTransformNode:
    @pytest.mark.asyncio
    async def test_sync_transform(self):
        n = TransformNode(
            "fmt",
            transform=lambda state: {"answer": state.context.get("raw", "")},
            output_key="formatted",
        )
        state = GraphState(context={"raw": "hello"})
        result = await n.execute(state, {})
        assert result.output == {"answer": "hello"}
        assert state.context["formatted"] == {"answer": "hello"}


class TestGuardNode:
    @pytest.mark.asyncio
    async def test_pass(self):
        n = GuardNode(
            "check",
            guards=[lambda x: True],
            on_pass="next",
            on_fail="retry",
        )
        state = GraphState(messages=[])
        result = await n.execute(state, {})
        assert result.next_node == "next"

    @pytest.mark.asyncio
    async def test_fail(self):
        n = GuardNode(
            "check",
            guards=[lambda x: False],
            on_pass="next",
            on_fail="retry",
        )
        state = GraphState(messages=[])
        result = await n.execute(state, {})
        assert result.next_node == "retry"


class TestFunctionalNode:
    @pytest.mark.asyncio
    async def test_node_decorator(self):
        @node("greet", default_next="done")
        async def greet(state: GraphState) -> NodeResult:
            return NodeResult(
                node_name="greet", output=f"Hello {state.context.get('name', 'world')}"
            )

        state = GraphState(context={"name": "Alice"})
        result = await greet.execute(state, {})
        assert result.output == "Hello Alice"
        assert greet.default_next == "done"


# ═══════════════════════════════════════════════════════════════════════════════
# Serialization
# ═══════════════════════════════════════════════════════════════════════════════


class TestSerialization:
    def test_node_to_config(self):
        n = PromptNode("test", instructions="Hello", temperature=0.5)
        config = node_to_config(n)
        assert config["name"] == "test"
        assert config["instructions"] == "Hello"
        assert config["temperature"] == 0.5

    def test_node_from_config(self):
        config = {"name": "test", "type": "prompt", "instructions": "Hello"}
        n = node_from_config(config)
        assert n.name == "test"
        assert isinstance(n, PromptNode)

    def test_graph_roundtrip(self):
        g = PromptGraph("test")
        g.add_node(PromptNode("a", instructions="Plan"))
        g.add_node(PromptNode("b", instructions="Act"))
        g.add_edge("a", "b")
        g.set_entry("a")

        config = graph_to_config(g)
        g2 = graph_from_config(config)

        assert g2.name == "test"
        assert g2.has_node("a")
        assert g2.has_node("b")
        assert g2.entry == "a"


# ═══════════════════════════════════════════════════════════════════════════════
# Skills
# ═══════════════════════════════════════════════════════════════════════════════


class TestSkills:
    def test_web_researcher(self):
        n = web_researcher("search")
        assert n.name == "search"
        assert "research" in n.instructions.lower()

    def test_summarizer(self):
        n = summarizer("conclude")
        assert n.name == "conclude"
        assert n.default_next == "__end__"

    def test_planner(self):
        n = planner("plan")
        assert n.name == "plan"
        assert "subgoal" in n.instructions.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Prebuilt Graphs
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrebuiltGraphs:
    def test_react(self):
        g = PromptGraph.react(tools=[], system_prompt="test")
        assert g.entry == "reason"
        assert g.has_node("reason")
        assert g.validate() == []

    def test_verify(self):
        # Single-pass self-verifying reasoning prebuilt.
        g = PromptGraph.verify(tools=[], system_prompt="test")
        assert g.entry == "reason"
        assert g.has_node("reason")
        assert g.get_node("reason").default_next == "__end__"
        assert g.validate() == []

    def test_managed(self):
        # Context-managed tool loop prebuilt (deduplicated facts ledger).
        g = PromptGraph.managed(tools=[], system_prompt="test")
        assert g.entry == "reason"
        assert g.has_node("reason")
        node = g.get_node("reason")
        assert node.context_scope == "ledger"
        assert node.default_next == "__end__"
        assert g.validate() == []

    def test_peoatr(self):
        g = PromptGraph.peoatr(tools=[])
        assert g.entry == "plan"
        assert g.has_node("plan")
        assert g.has_node("act")
        assert g.has_node("think")
        assert g.has_node("reflect")

    def test_research(self):
        g = PromptGraph.research(search_tools=[])
        assert g.entry == "search"
        assert g.has_node("search")
        assert g.has_node("synthesize")

    def test_autonomous(self):
        g = PromptGraph.autonomous(tools=[])
        assert g.entry == "agent"


# ═══════════════════════════════════════════════════════════════════════════════
# Hooks
# ═══════════════════════════════════════════════════════════════════════════════


class TestHooks:
    @pytest.mark.asyncio
    async def test_logging_hook(self):
        hook = LoggingHook()
        state = GraphState(iteration=1)
        n = PromptNode("test")
        state = await hook.pre_node(n, state)
        result = NodeResult(node_name="test")
        result = await hook.post_node(n, result, state)
        assert isinstance(result, NodeResult)

    @pytest.mark.asyncio
    async def test_cycle_detection(self):
        hook = CycleDetectionHook(sequence_length=2, max_repeats=2)
        state = GraphState(visited=["a", "b", "a", "b", "a", "b"])
        n = PromptNode("a")
        state = await hook.pre_node(n, state)
        assert state.current_node == "__end__"


# ═══════════════════════════════════════════════════════════════════════════════
# Execution Report
# ═══════════════════════════════════════════════════════════════════════════════


class TestExecutionReport:
    def test_summary(self):
        report = ExecutionReport(
            total_iterations=5,
            total_tokens=1000,
            total_duration_ms=2500.0,
            nodes_visited=["a", "b", "c"],
            tool_calls=2,
        )
        s = report.summary()
        assert "5" in s
        assert "1,000" in s
        assert "a → b → c" in s


# ═══════════════════════════════════════════════════════════════════════════════
# Flag Processing — engine-level tests
# ═══════════════════════════════════════════════════════════════════════════════


def _make_mock_model():
    """Create a minimal mock model that returns an AIMessage."""
    from unittest.mock import AsyncMock, MagicMock

    from langchain_core.messages import AIMessage

    model = MagicMock(spec=["ainvoke", "bind_tools", "with_structured_output"])
    model.ainvoke = AsyncMock(return_value=AIMessage(content="mock response"))
    model.bind_tools = MagicMock(return_value=model)
    return model


class TestCriticalFlag:
    @pytest.mark.asyncio
    async def test_critical_aborts_on_error(self):
        """CRITICAL node error → graph aborts immediately, report.error set."""
        call_count = 0

        @node("step1", default_next="step2", flags={NodeFlag.CRITICAL})
        async def step1(state: GraphState) -> NodeResult:
            nonlocal call_count
            call_count += 1
            raise RuntimeError("boom")

        @node("step2", default_next="__end__")
        async def step2(state: GraphState) -> NodeResult:
            nonlocal call_count
            call_count += 1
            return NodeResult(node_name="step2", output="done")

        graph = PromptGraph("test", mode="static")
        graph.add_node(step1)
        graph.add_node(step2)
        graph.set_entry("step1")
        graph.add_edge("step1", "step2")

        engine = PromptGraphEngine(graph=graph, model=_make_mock_model())
        # The abort reaches the caller — it is never a silent return
        from promptise.engine.execution import GraphExecutionError

        with pytest.raises(GraphExecutionError, match="step1.*boom") as info:
            await engine.ainvoke({"messages": []})
        assert isinstance(info.value.__cause__, RuntimeError)

        assert call_count == 1  # step2 never ran
        assert engine.last_report is not None
        assert engine.last_report.error is not None
        assert "boom" in engine.last_report.error
        assert info.value.report is engine.last_report

    @pytest.mark.asyncio
    async def test_critical_passes_on_success(self):
        """CRITICAL node that succeeds → normal transition."""

        @node("step1", default_next="step2", flags={NodeFlag.CRITICAL})
        async def step1(state: GraphState) -> NodeResult:
            return NodeResult(node_name="step1", output="ok")

        @node("step2", default_next="__end__")
        async def step2(state: GraphState) -> NodeResult:
            return NodeResult(node_name="step2", output="done")

        graph = PromptGraph("test", mode="static")
        graph.add_node(step1)
        graph.add_node(step2)
        graph.set_entry("step1")
        graph.add_edge("step1", "step2")

        engine = PromptGraphEngine(graph=graph, model=_make_mock_model())
        await engine.ainvoke({"messages": []})

        assert engine.last_report is not None
        assert engine.last_report.error is None
        assert "step2" in engine.last_report.nodes_visited


class TestSkipOnErrorFlag:
    @pytest.mark.asyncio
    async def test_skips_after_error(self):
        """SKIP_ON_ERROR node is skipped when previous node errored."""
        skipped_ran = False

        @node("fail_node", default_next="skippable")
        async def fail_node(state: GraphState) -> NodeResult:
            return NodeResult(node_name="fail_node", error="failed")

        @node("skippable", default_next="final", flags={NodeFlag.SKIP_ON_ERROR})
        async def skippable(state: GraphState) -> NodeResult:
            nonlocal skipped_ran
            skipped_ran = True
            return NodeResult(node_name="skippable", output="ran")

        @node("final", default_next="__end__")
        async def final_node(state: GraphState) -> NodeResult:
            return NodeResult(node_name="final", output="done")

        graph = PromptGraph("test", mode="static")
        graph.add_node(fail_node)
        graph.add_node(skippable)
        graph.add_node(final_node)
        graph.set_entry("fail_node")
        graph.add_edge("fail_node", "skippable")
        graph.add_edge("skippable", "final")

        engine = PromptGraphEngine(graph=graph, model=_make_mock_model())
        await engine.ainvoke({"messages": []})

        assert not skipped_ran

    @pytest.mark.asyncio
    async def test_runs_after_success(self):
        """SKIP_ON_ERROR node runs normally when previous node succeeded."""
        skipped_ran = False

        @node("ok_node", default_next="skippable")
        async def ok_node(state: GraphState) -> NodeResult:
            return NodeResult(node_name="ok_node", output="ok")

        @node("skippable", default_next="__end__", flags={NodeFlag.SKIP_ON_ERROR})
        async def skippable(state: GraphState) -> NodeResult:
            nonlocal skipped_ran
            skipped_ran = True
            return NodeResult(node_name="skippable", output="ran")

        graph = PromptGraph("test", mode="static")
        graph.add_node(ok_node)
        graph.add_node(skippable)
        graph.set_entry("ok_node")
        graph.add_edge("ok_node", "skippable")

        engine = PromptGraphEngine(graph=graph, model=_make_mock_model())
        await engine.ainvoke({"messages": []})

        assert skipped_ran


class TestRetryableFlag:
    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self):
        """RETRYABLE node fails twice, succeeds on third attempt."""
        attempt = 0

        @node("retry_me", default_next="__end__", flags={NodeFlag.RETRYABLE}, max_iterations=3)
        async def retry_me(state: GraphState) -> NodeResult:
            nonlocal attempt
            attempt += 1
            if attempt < 3:
                return NodeResult(node_name="retry_me", error=f"fail {attempt}")
            return NodeResult(node_name="retry_me", output="success")

        graph = PromptGraph("test", mode="static")
        graph.add_node(retry_me)
        graph.set_entry("retry_me")

        engine = PromptGraphEngine(graph=graph, model=_make_mock_model())
        await engine.ainvoke({"messages": []})

        assert attempt == 3
        # The successful result should be in history
        history = engine.last_report.nodes_visited
        assert "retry_me" in history

    @pytest.mark.asyncio
    async def test_exhausts_retries(self):
        """RETRYABLE node fails all attempts → error result."""

        @node("always_fail", default_next="__end__", flags={NodeFlag.RETRYABLE}, max_iterations=2)
        async def always_fail(state: GraphState) -> NodeResult:
            return NodeResult(node_name="always_fail", error="nope")

        graph = PromptGraph("test", mode="static")
        graph.add_node(always_fail)
        graph.set_entry("always_fail")

        engine = PromptGraphEngine(graph=graph, model=_make_mock_model())
        from promptise.engine.execution import GraphExecutionError

        # Out of attempts with nothing to recover → the run failed, loudly
        with pytest.raises(GraphExecutionError, match="always_fail.*nope"):
            await engine.ainvoke({"messages": []})

        # ... and recorded the error
        assert engine.last_report is not None
        assert engine.last_report.error == "nope"


class TestNoHistoryFlag:
    @pytest.mark.asyncio
    async def test_clears_messages(self):
        """NO_HISTORY node receives no conversation history."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages_seen = None

        @node("history_check", default_next="__end__", flags={NodeFlag.NO_HISTORY})
        async def history_check(state: GraphState) -> NodeResult:
            nonlocal messages_seen
            messages_seen = list(state.messages)
            return NodeResult(node_name="history_check", output="ok")

        graph = PromptGraph("test", mode="static")
        graph.add_node(history_check)
        graph.set_entry("history_check")

        engine = PromptGraphEngine(graph=graph, model=_make_mock_model())
        await engine.ainvoke(
            {
                "messages": [
                    SystemMessage(content="system"),
                    HumanMessage(content="hello"),
                    HumanMessage(content="world"),
                ]
            }
        )

        # Node should only see the system message
        assert messages_seen is not None
        assert len(messages_seen) == 1
        assert messages_seen[0].content == "system"

    @pytest.mark.asyncio
    async def test_restores_messages_after(self):
        """Messages are restored after NO_HISTORY node execution."""
        from langchain_core.messages import HumanMessage, SystemMessage

        @node("no_hist", default_next="check", flags={NodeFlag.NO_HISTORY})
        async def no_hist(state: GraphState) -> NodeResult:
            return NodeResult(node_name="no_hist", output="ok")

        check_messages_len = None

        @node("check", default_next="__end__")
        async def check(state: GraphState) -> NodeResult:
            nonlocal check_messages_len
            check_messages_len = len(state.messages)
            return NodeResult(node_name="check", output="ok")

        graph = PromptGraph("test", mode="static")
        graph.add_node(no_hist)
        graph.add_node(check)
        graph.set_entry("no_hist")
        graph.add_edge("no_hist", "check")

        engine = PromptGraphEngine(graph=graph, model=_make_mock_model())
        await engine.ainvoke(
            {
                "messages": [
                    SystemMessage(content="sys"),
                    HumanMessage(content="hi"),
                ]
            }
        )

        # Next node should see the original messages restored
        assert check_messages_len is not None
        assert check_messages_len >= 2


class TestIsolatedContextFlag:
    @pytest.mark.asyncio
    async def test_provides_clean_context(self):
        """ISOLATED_CONTEXT node sees only its input_keys data."""
        context_seen = None

        @node("isolated", default_next="__end__", flags={NodeFlag.ISOLATED_CONTEXT})
        async def isolated(state: GraphState) -> NodeResult:
            nonlocal context_seen
            context_seen = dict(state.context)
            state.context["isolated_output"] = "from_isolated"
            return NodeResult(node_name="isolated", output="ok")

        # Manually set input_keys on the functional node
        isolated.input_keys = ["needed_key"]

        graph = PromptGraph("test", mode="static")
        graph.add_node(isolated)
        graph.set_entry("isolated")

        engine = PromptGraphEngine(graph=graph, model=_make_mock_model())
        state_input = {"messages": []}

        # Pre-set context through a first node
        @node("setup", default_next="isolated")
        async def setup(state: GraphState) -> NodeResult:
            state.context["needed_key"] = "value1"
            state.context["extra_key"] = "should_not_see"
            return NodeResult(node_name="setup", output="ok")

        graph = PromptGraph("test", mode="static")
        graph.add_node(setup)
        graph.add_node(isolated)
        graph.set_entry("setup")
        graph.add_edge("setup", "isolated")

        engine = PromptGraphEngine(graph=graph, model=_make_mock_model())
        await engine.ainvoke(state_input)

        # Isolated node should only see needed_key
        assert context_seen is not None
        assert "needed_key" in context_seen
        assert "extra_key" not in context_seen

    @pytest.mark.asyncio
    async def test_merges_output_key_back(self):
        """After ISOLATED_CONTEXT, output_key value appears in restored context."""
        final_context = None

        @node("isolated", default_next="check", flags={NodeFlag.ISOLATED_CONTEXT})
        async def isolated(state: GraphState) -> NodeResult:
            state.context["my_output"] = "produced"
            return NodeResult(node_name="isolated", output="ok")

        isolated.output_key = "my_output"

        @node("check", default_next="__end__")
        async def check(state: GraphState) -> NodeResult:
            nonlocal final_context
            final_context = dict(state.context)
            return NodeResult(node_name="check", output="done")

        @node("setup", default_next="isolated")
        async def setup(state: GraphState) -> NodeResult:
            state.context["existing"] = "preserved"
            return NodeResult(node_name="setup", output="ok")

        graph = PromptGraph("test", mode="static")
        graph.add_node(setup)
        graph.add_node(isolated)
        graph.add_node(check)
        graph.set_entry("setup")
        graph.add_edge("setup", "isolated")
        graph.add_edge("isolated", "check")

        engine = PromptGraphEngine(graph=graph, model=_make_mock_model())
        await engine.ainvoke({"messages": []})

        assert final_context is not None
        assert final_context["existing"] == "preserved"
        assert final_context["my_output"] == "produced"


class TestCacheableFlag:
    @pytest.mark.asyncio
    async def test_returns_cached_on_second_run(self):
        """CACHEABLE node returns cached result on second visit."""
        call_count = 0

        @node("cacheable", default_next="loop_or_end", flags={NodeFlag.CACHEABLE})
        async def cacheable(state: GraphState) -> NodeResult:
            nonlocal call_count
            call_count += 1
            return NodeResult(node_name="cacheable", output=f"run_{call_count}")

        @node("loop_or_end")
        async def loop_or_end(state: GraphState) -> NodeResult:
            # Loop back once, then end
            if state.iteration < 2:
                return NodeResult(node_name="loop_or_end", next_node="cacheable")
            return NodeResult(node_name="loop_or_end", next_node="__end__")

        graph = PromptGraph("test", mode="static")
        graph.add_node(cacheable)
        graph.add_node(loop_or_end)
        graph.set_entry("cacheable")
        graph.add_edge("cacheable", "loop_or_end")

        engine = PromptGraphEngine(graph=graph, model=_make_mock_model())
        await engine.ainvoke({"messages": []})

        # Should only execute once — second visit hits cache
        assert call_count == 1


class TestReasoningNodeDefaultFlags:
    def test_think_node_flags(self):
        n = ThinkNode("t")
        assert NodeFlag.READONLY in n.flags
        assert NodeFlag.LIGHTWEIGHT in n.flags

    def test_reflect_node_flags(self):
        n = ReflectNode("r")
        assert NodeFlag.STATEFUL in n.flags
        assert NodeFlag.OBSERVABLE in n.flags

    def test_observe_node_flags(self):
        n = ObserveNode("o")
        assert NodeFlag.STATEFUL in n.flags

    def test_justify_node_flags(self):
        n = JustifyNode("j")
        assert NodeFlag.READONLY in n.flags
        assert NodeFlag.OBSERVABLE in n.flags
        assert NodeFlag.VERBOSE in n.flags

    def test_critique_node_flags(self):
        n = CritiqueNode("c")
        assert NodeFlag.READONLY in n.flags
        assert NodeFlag.OBSERVABLE in n.flags

    def test_plan_node_flags(self):
        n = PlanNode("p")
        assert NodeFlag.STATEFUL in n.flags
        assert NodeFlag.OBSERVABLE in n.flags

    def test_synthesize_node_flags(self):
        n = SynthesizeNode("s")
        assert NodeFlag.OBSERVABLE in n.flags

    def test_validate_node_flags(self):
        n = ValidateNode("v")
        assert NodeFlag.READONLY in n.flags
        assert NodeFlag.VALIDATE_OUTPUT in n.flags

    def test_retry_node_flags(self):
        inner = TransformNode("inner", transform=lambda s, _: NodeResult(node_name="inner"))
        n = RetryNode("r", wrapped_node=inner)
        assert NodeFlag.RETRYABLE in n.flags

    def test_fan_out_node_flags(self):
        n = FanOutNode("f")
        assert NodeFlag.PARALLEL_SAFE in n.flags
