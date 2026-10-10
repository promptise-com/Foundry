"""Tests for PromptNode.execute() pipeline and PromptGraphEngine edge cases."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage

from promptise.engine import (
    GraphState,
    NodeResult,
    PromptGraph,
    PromptGraphEngine,
    PromptNode,
    TransformNode,
    node,
)


def _make_model(content: str = "mock response", tool_calls: list | None = None):
    """Create a mock model returning a single AIMessage."""
    msg = AIMessage(content=content)
    if tool_calls:
        msg.tool_calls = tool_calls
    model = MagicMock(spec=["ainvoke", "bind_tools", "with_structured_output"])
    model.ainvoke = AsyncMock(return_value=msg)
    model.bind_tools = MagicMock(return_value=model)
    return model


def _make_sequence_model(responses: list[str]):
    """Model that returns different AIMessages in sequence."""
    msgs = [AIMessage(content=r) for r in responses]
    model = MagicMock(spec=["ainvoke", "bind_tools", "with_structured_output"])
    model.ainvoke = AsyncMock(side_effect=msgs)
    model.bind_tools = MagicMock(return_value=model)
    return model


class _RejectedByProvider(BaseChatModel):
    """A chat model whose provider rejects every call (a wrong API key)."""

    @property
    def _llm_type(self) -> str:
        return "rejected-by-provider"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise PermissionError("Error code: 401 - Incorrect API key provided: sk-inval**-key")


# ═══════════════════════════════════════════════════════════════════════════════
# PromptNode Execution Pipeline
# ═══════════════════════════════════════════════════════════════════════════════


class TestPromptNodeExecution:
    @pytest.mark.asyncio
    async def test_basic_prompt_returns_content(self):
        n = PromptNode("test", instructions="Analyze this.")
        model = _make_model("The answer is 42")
        state = GraphState(messages=[HumanMessage(content="What is 42?")])
        config = {"_engine_model": model}

        result = await n.execute(state, config)

        assert result.raw_output == "The answer is 42"
        assert len(result.messages_added) > 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_input_keys_injected(self):
        n = PromptNode("test", instructions="Analyze.", input_keys=["query", "source"])
        model = _make_model("analyzed")
        state = GraphState(
            messages=[HumanMessage(content="go")],
            context={"query": "test query", "source": "web", "irrelevant": "skip"},
        )
        config = {"_engine_model": model}

        await n.execute(state, config)

        # Check that model.ainvoke was called with messages containing input data
        call_args = model.ainvoke.call_args[0][0]  # First positional arg = messages
        system_content = call_args[0].content
        assert "query: test query" in system_content
        assert "source: web" in system_content

    @pytest.mark.asyncio
    async def test_output_key_written_to_state(self):
        n = PromptNode("test", instructions="Analyze.", output_key="result")
        model = _make_model("the output")
        state = GraphState(messages=[HumanMessage(content="go")])
        config = {"_engine_model": model}

        await n.execute(state, config)

        assert "test_output" in state.context  # Auto-written {name}_output

    @pytest.mark.asyncio
    async def test_inherit_context_from(self):
        n = PromptNode("step2", instructions="Continue.", inherit_context_from="step1")
        model = _make_model("continued")
        state = GraphState(
            messages=[HumanMessage(content="go")],
            context={"step1_output": "previous result"},
        )
        config = {"_engine_model": model}

        await n.execute(state, config)

        call_args = model.ainvoke.call_args[0][0]
        system_content = call_args[0].content
        assert "previous result" in system_content

    @pytest.mark.asyncio
    async def test_model_override_string_fails_gracefully(self):
        n = PromptNode("test", instructions="Go.", model_override="invalid:nonexistent-model")
        state = GraphState(messages=[HumanMessage(content="go")])
        config = {"_engine_model": _make_model()}

        result = await n.execute(state, config)

        assert result.error is not None
        assert "Failed to initialize" in result.error

    @pytest.mark.asyncio
    async def test_model_override_failure_never_leaks_credentials(self, monkeypatch):
        """The node error names the model by spec — a Model carries the API
        key and the error string lands in logs, graph history and events."""
        from promptise.models import Model

        monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")
        # region= is not an OpenAI setting → Model.resolve() raises
        cfg = Model("gpt-5", provider="openai", api_key="sk-SECRET-abc123", region="us")
        n = PromptNode("test", instructions="Go.", model_override=cfg)
        state = GraphState(messages=[HumanMessage(content="go")])

        result = await n.execute(state, {"_engine_model": _make_model()})

        assert result.error is not None
        assert "Failed to initialize model openai:gpt-5" in result.error
        assert "region" in result.error
        assert "sk-SECRET" not in result.error

    @pytest.mark.asyncio
    async def test_model_override_instance_used(self):
        override_model = _make_model("from override")
        engine_model = _make_model("from engine")

        n = PromptNode("test", instructions="Go.", model_override=override_model)
        state = GraphState(messages=[HumanMessage(content="go")])
        config = {"_engine_model": engine_model}

        result = await n.execute(state, config)

        assert override_model.ainvoke.called
        assert not engine_model.ainvoke.called
        assert result.raw_output == "from override"

    @pytest.mark.asyncio
    async def test_preprocessor_runs(self):
        enriched = False

        def my_pre(state, config):
            nonlocal enriched
            enriched = True
            state.context["enriched"] = True

        n = PromptNode("test", instructions="Go.", preprocessor=my_pre)
        state = GraphState(messages=[HumanMessage(content="go")])
        config = {"_engine_model": _make_model()}

        await n.execute(state, config)

        assert enriched
        assert state.context.get("enriched") is True

    @pytest.mark.asyncio
    async def test_postprocessor_transforms_output(self):
        def my_post(output, state, config):
            return {"transformed": True, "original": output}

        n = PromptNode("test", instructions="Go.", postprocessor=my_post)
        state = GraphState(messages=[HumanMessage(content="go")])
        config = {"_engine_model": _make_model("raw output")}

        result = await n.execute(state, config)

        assert result.output["transformed"] is True

    @pytest.mark.asyncio
    async def test_sync_guard_runs(self):
        class SyncGuard:
            def check_output(self, output):
                pass  # No raise = pass

        n = PromptNode("test", instructions="Go.", guards=[SyncGuard()])
        state = GraphState(messages=[HumanMessage(content="go")])
        config = {"_engine_model": _make_model()}

        result = await n.execute(state, config)

        assert "SyncGuard" in result.guards_passed

    @pytest.mark.asyncio
    async def test_async_guard_runs(self):
        class AsyncGuard:
            async def check_output(self, output):
                pass  # No raise = pass

        n = PromptNode("test", instructions="Go.", guards=[AsyncGuard()])
        state = GraphState(messages=[HumanMessage(content="go")])
        config = {"_engine_model": _make_model()}

        result = await n.execute(state, config)

        assert "AsyncGuard" in result.guards_passed

    @pytest.mark.asyncio
    async def test_guard_failure_recorded(self):
        class FailGuard:
            def check_output(self, output):
                raise ValueError("output is invalid")

        n = PromptNode("test", instructions="Go.", guards=[FailGuard()])
        state = GraphState(messages=[HumanMessage(content="go")])
        config = {"_engine_model": _make_model()}

        result = await n.execute(state, config)

        assert len(result.guards_failed) == 1
        assert "output is invalid" in result.guards_failed[0]

    @pytest.mark.asyncio
    async def test_observations_auto_injected(self):
        n = PromptNode("test", instructions="Analyze.", include_observations=True)
        model = _make_model("ok")
        state = GraphState(messages=[HumanMessage(content="go")])
        state.add_observation("web_search", "found 3 results", {"query": "test"})
        config = {"_engine_model": model}

        await n.execute(state, config)

        call_args = model.ainvoke.call_args[0][0]
        system_content = call_args[0].content
        assert "web_search" in system_content

    @pytest.mark.asyncio
    async def test_plan_auto_injected(self):
        n = PromptNode("test", instructions="Continue.", include_plan=True)
        model = _make_model("ok")
        state = GraphState(messages=[HumanMessage(content="go")])
        state.plan = ["Step 1: Gather data", "Step 2: Analyze"]
        config = {"_engine_model": model}

        await n.execute(state, config)

        call_args = model.ainvoke.call_args[0][0]
        system_content = call_args[0].content
        assert "Gather data" in system_content

    @pytest.mark.asyncio
    async def test_no_model_returns_error(self):
        n = PromptNode("test", instructions="Go.")
        state = GraphState(messages=[HumanMessage(content="go")])
        config = {}  # No model

        result = await n.execute(state, config)

        assert result.error is not None
        assert "No model" in result.error


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Calling
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolCalling:
    @pytest.mark.asyncio
    async def test_tool_injection_merges(self):
        """inject_tools=True should merge engine tools with local tools."""
        local_tool = MagicMock()
        local_tool.name = "local_tool"

        engine_tool = MagicMock()
        engine_tool.name = "engine_tool"

        n = PromptNode("test", instructions="Go.", tools=[local_tool], inject_tools=True)
        model = _make_model("done")
        state = GraphState(messages=[HumanMessage(content="go")])
        config = {"_engine_model": model, "_engine_tools": [engine_tool]}

        await n.execute(state, config)

        # bind_tools should have been called with both tools
        bind_call = model.bind_tools.call_args[0][0]
        tool_names = {t.name for t in bind_call}
        assert "local_tool" in tool_names
        assert "engine_tool" in tool_names


# ═══════════════════════════════════════════════════════════════════════════════
# Engine Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEngineEdgeCases:
    @pytest.mark.asyncio
    async def test_missing_node_ends_gracefully(self):
        """If current_node points to a nonexistent node, graph ends without crash."""

        @node("start", default_next="nonexistent")
        async def start(state: GraphState) -> NodeResult:
            return NodeResult(node_name="start", output="ok")

        graph = PromptGraph("test", mode="static")
        graph.add_node(start)
        graph.set_entry("start")

        engine = PromptGraphEngine(graph=graph, model=_make_model())
        await engine.ainvoke({"messages": []})

        # Should not crash — ends gracefully
        assert engine.last_report is not None

    @pytest.mark.asyncio
    async def test_max_iterations_stops_graph(self):
        count = 0

        @node("loop")
        async def loop(state: GraphState) -> NodeResult:
            nonlocal count
            count += 1
            return NodeResult(node_name="loop", next_node="loop")

        graph = PromptGraph("test", mode="static")
        graph.add_node(loop)
        graph.set_entry("loop")

        engine = PromptGraphEngine(graph=graph, model=_make_model(), max_iterations=5)
        await engine.ainvoke({"messages": []})

        assert count <= 6  # max_iterations + 1 safety margin
        assert engine.last_report.total_iterations <= 6

    @pytest.mark.asyncio
    async def test_concurrent_ainvoke(self):
        """Two concurrent ainvoke calls should not interfere with each other."""

        @node("step", default_next="__end__")
        async def step(state: GraphState) -> NodeResult:
            await asyncio.sleep(0.01)
            return NodeResult(node_name="step", output="done")

        graph = PromptGraph("test", mode="static")
        graph.add_node(step)
        graph.set_entry("step")

        engine = PromptGraphEngine(graph=graph, model=_make_model())

        results = await asyncio.gather(
            engine.ainvoke({"messages": [HumanMessage(content="call 1")]}),
            engine.ainvoke({"messages": [HumanMessage(content="call 2")]}),
        )

        assert len(results) == 2
        # Both should have messages
        assert "messages" in results[0]
        assert "messages" in results[1]

    @pytest.mark.asyncio
    async def test_empty_messages_input(self):
        @node("step", default_next="__end__")
        async def step(state: GraphState) -> NodeResult:
            return NodeResult(node_name="step", output="ok")

        graph = PromptGraph("test", mode="static")
        graph.add_node(step)
        graph.set_entry("step")

        engine = PromptGraphEngine(graph=graph, model=_make_model())
        result = await engine.ainvoke({"messages": []})

        assert "messages" in result

    @pytest.mark.asyncio
    async def test_no_entry_raises(self):
        graph = PromptGraph("test", mode="static")
        graph.add_node(TransformNode("orphan", transform=lambda s, c: None))
        # Don't set entry

        engine = PromptGraphEngine(graph=graph, model=_make_model())

        with pytest.raises(ValueError, match="no entry"):
            await engine.ainvoke({"messages": []})

    @pytest.mark.asyncio
    async def test_execution_report_populated(self):
        @node("a", default_next="b")
        async def a(state: GraphState) -> NodeResult:
            return NodeResult(node_name="a", output="ok", total_tokens=100)

        @node("b", default_next="__end__")
        async def b(state: GraphState) -> NodeResult:
            return NodeResult(node_name="b", output="done", total_tokens=200)

        graph = PromptGraph("test", mode="static")
        graph.add_node(a)
        graph.add_node(b)
        graph.set_entry("a")
        graph.add_edge("a", "b")

        engine = PromptGraphEngine(graph=graph, model=_make_model())
        await engine.ainvoke({"messages": []})

        report = engine.last_report
        assert report is not None
        assert report.total_iterations == 2
        assert report.total_tokens == 300
        assert report.nodes_visited == ["a", "b"]
        # Windows ``time.perf_counter()`` can report 0.0 for sub-ms durations;
        # accept any non-negative value.
        assert report.total_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_hook_forced_end_respected(self):
        """If a post_node hook sets state.current_node = '__end__', engine stops."""

        class ForceEndHook:
            async def post_node(self, node, result, state):
                state.current_node = "__end__"
                return result

        @node("step", default_next="step")  # Would loop forever
        async def step(state: GraphState) -> NodeResult:
            return NodeResult(node_name="step", output="ok")

        graph = PromptGraph("test", mode="static")
        graph.add_node(step)
        graph.set_entry("step")

        engine = PromptGraphEngine(graph=graph, model=_make_model(), hooks=[ForceEndHook()])
        await engine.ainvoke({"messages": []})

        # Hook forced end after first node — only 1 node visited
        assert engine.last_report.nodes_visited == ["step"]


# ═══════════════════════════════════════════════════════════════════════════════
# Failures reach the caller — never a silent "answer"
# ═══════════════════════════════════════════════════════════════════════════════


class TestFailuresReachTheCaller:
    """A provider error used to be stored on the NodeResult only, and
    ``ainvoke()`` returned the input messages as the "answer"."""

    @pytest.mark.asyncio
    async def test_engine_raises_when_model_call_fails(self):
        from promptise.engine.execution import GraphExecutionError

        model = _make_model()
        model.ainvoke = AsyncMock(side_effect=ConnectionError("provider unreachable"))
        graph = PromptGraph("pipeline")
        graph.add_node(PromptNode("reason", instructions="Go."))
        graph.set_entry("reason")
        engine = PromptGraphEngine(graph=graph, model=model)

        with pytest.raises(GraphExecutionError) as info:
            await engine.ainvoke({"messages": [HumanMessage(content="Say hi")]})

        exc = info.value
        assert exc.graph_name == "pipeline" and exc.node_name == "reason"
        assert "Graph 'pipeline' failed at node 'reason'" in str(exc)
        assert "ConnectionError: provider unreachable" in str(exc)
        assert isinstance(exc.__cause__, ConnectionError)
        # Node-level recording for hooks/observability is intact
        assert engine.last_report is not None
        assert engine.last_report.error == "ConnectionError: provider unreachable"
        assert engine.last_report.nodes_visited == ["reason"]

    @pytest.mark.asyncio
    async def test_failure_routed_to_a_handler_node_is_recovery(self):
        model = _make_model("Sorry, the upstream model is unavailable right now.")
        model.ainvoke = AsyncMock(
            side_effect=[ConnectionError("down"), AIMessage(content="fallback answer")]
        )
        graph = PromptGraph("handled")
        graph.add_node(PromptNode("reason", instructions="Go.", default_next="apologize"))
        graph.add_node(PromptNode("apologize", instructions="Apologize."))
        graph.set_entry("reason")
        engine = PromptGraphEngine(graph=graph, model=model)

        out = await engine.ainvoke({"messages": [HumanMessage(content="hi")]})

        assert out["messages"][-1].content == "fallback answer"
        assert engine.last_report is not None
        assert engine.last_report.error is None
        assert engine.last_report.nodes_visited == ["reason", "apologize"]

    @pytest.mark.asyncio
    async def test_hook_recovery_suppresses_the_raise(self):
        class Recover:
            async def post_node(self, node, result, state):
                if result.error:
                    result.error_recovered = True
                return result

        model = _make_model()
        model.ainvoke = AsyncMock(side_effect=ConnectionError("down"))
        graph = PromptGraph("recovered")
        graph.add_node(PromptNode("reason", instructions="Go."))
        graph.set_entry("reason")
        engine = PromptGraphEngine(graph=graph, model=model, hooks=[Recover()])

        out = await engine.ainvoke({"messages": [HumanMessage(content="hi")]})
        assert [type(m).__name__ for m in out["messages"]] == ["HumanMessage"]

    @pytest.mark.asyncio
    async def test_astream_events_raises_after_its_events(self):
        from promptise.engine.execution import GraphExecutionError

        model = MagicMock(spec=["ainvoke", "astream", "bind_tools", "with_structured_output"])

        async def _boom(*args, **kwargs):
            raise ConnectionError("stream down")
            yield  # pragma: no cover - makes this an async generator

        model.astream = _boom
        model.ainvoke = AsyncMock(side_effect=ConnectionError("stream down"))
        model.bind_tools = MagicMock(return_value=model)
        graph = PromptGraph("streamed")
        graph.add_node(PromptNode("reason", instructions="Go."))
        graph.set_entry("reason")
        engine = PromptGraphEngine(graph=graph, model=model)

        events = []
        with pytest.raises(GraphExecutionError, match="reason.*stream down") as info:
            async for event in engine.astream_events({"messages": [HumanMessage(content="hi")]}):
                events.append(event["event"])
        assert "on_node_error" in events
        assert isinstance(info.value.__cause__, ConnectionError)

    @pytest.mark.asyncio
    async def test_build_agent_ainvoke_raises_on_provider_failure(self, monkeypatch):
        from promptise import build_agent
        from promptise.engine.execution import GraphExecutionError

        monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")
        agent = await build_agent(model=_RejectedByProvider(), servers={})

        with pytest.raises(GraphExecutionError) as info:
            await agent.ainvoke({"messages": [HumanMessage(content="Say hi")]})

        message = str(info.value)
        assert "failed at node 'reason'" in message
        assert "Incorrect API key" in message
        assert isinstance(info.value.__cause__, PermissionError)
