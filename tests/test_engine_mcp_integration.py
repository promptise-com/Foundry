"""Tests for MCP Server → Engine integration and prebuilt pattern construction."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import AIMessage

from promptise.engine import (
    GraphState,
    NodeResult,
    PromptGraphEngine,
    PromptNode,
    node,
)
from promptise.engine.nodes import AutonomousNode
from promptise.engine.prebuilts import (
    build_autonomous_graph,
    build_debate_graph,
    build_deliberate_graph,
    build_peoatr_graph,
    build_pipeline_graph,
    build_react_graph,
    build_research_graph,
)
from promptise.mcp.server import MCPServer, TestClient

# ═══════════════════════════════════════════════════════════════════════════════
# MCP Server setup
# ═══════════════════════════════════════════════════════════════════════════════


def _build_test_server():
    """Create a simple MCP server with math tools."""
    server = MCPServer("math")

    @server.tool()
    async def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    @server.tool()
    async def multiply(a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b

    @server.tool()
    async def divide(a: float, b: float) -> float:
        """Divide two numbers."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b

    return server


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Tool Discovery
# ═══════════════════════════════════════════════════════════════════════════════


class TestMCPToolDiscovery:
    @pytest.mark.asyncio
    async def test_list_tools(self):
        server = _build_test_server()
        client = TestClient(server)
        tools = await client.list_tools()

        names = [t.name for t in tools]
        assert "add" in names
        assert "multiply" in names
        assert "divide" in names

    @pytest.mark.asyncio
    async def test_tool_invocation(self):
        server = _build_test_server()
        client = TestClient(server)

        result = await client.call_tool("add", {"a": 3, "b": 7})
        assert result[0].text == "10"

        result = await client.call_tool("multiply", {"a": 4, "b": 5})
        assert result[0].text == "20"

    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        server = _build_test_server()
        client = TestClient(server)

        result = await client.call_tool("divide", {"a": 10, "b": 0})
        text = result[0].text
        parsed = json.loads(text)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        server = _build_test_server()
        client = TestClient(server)

        result = await client.call_tool("nonexistent", {})
        text = result[0].text
        parsed = json.loads(text)
        assert "error" in parsed


# ═══════════════════════════════════════════════════════════════════════════════
# MCP Server Auth & Middleware
# ═══════════════════════════════════════════════════════════════════════════════


class TestMCPServerFeatures:
    @pytest.mark.asyncio
    async def test_router_namespacing(self):
        from promptise.mcp.server import MCPRouter

        server = MCPServer("multi")
        router = MCPRouter(prefix="math")

        @router.tool()
        async def square(x: int) -> int:
            """Square a number."""
            return x * x

        server.include_router(router)
        client = TestClient(server)

        result = await client.call_tool("math_square", {"x": 5})
        assert result[0].text == "25"

    @pytest.mark.asyncio
    async def test_dependency_injection(self):
        from promptise.mcp.server import Depends

        def get_multiplier():
            return 10

        server = MCPServer("di-test")

        @server.tool()
        async def scaled(value: int, mult: int = Depends(get_multiplier)) -> int:
            """Scale a value."""
            return value * mult

        client = TestClient(server)
        result = await client.call_tool("scaled", {"value": 5})
        assert result[0].text == "50"


# ═══════════════════════════════════════════════════════════════════════════════
# Prebuilt Pattern Construction
# ═══════════════════════════════════════════════════════════════════════════════


class TestPrebuiltPatterns:
    def test_react_graph(self):
        g = build_react_graph()
        assert g.entry is not None
        assert len(g.nodes) == 1
        assert "reason" in g.nodes

    def test_peoatr_graph(self):
        g = build_peoatr_graph()
        assert g.entry is not None
        assert len(g.nodes) == 5
        for name in ["plan", "act", "think", "reflect", "answer"]:
            assert name in g.nodes, f"Missing node: {name}"

    def test_research_graph(self):
        g = build_research_graph()
        assert g.entry is not None
        assert len(g.nodes) >= 2  # search + synthesize (verify optional)
        assert "search" in g.nodes
        assert "synthesize" in g.nodes

    def test_autonomous_graph(self):
        g = build_autonomous_graph()
        assert g.entry is not None
        assert len(g.nodes) >= 1
        # Should contain an AutonomousNode
        has_autonomous = any(isinstance(n, AutonomousNode) for n in g.nodes.values())
        assert has_autonomous

    def test_deliberate_graph(self):
        g = build_deliberate_graph()
        assert g.entry is not None
        assert len(g.nodes) == 5
        for name in ["think", "plan", "act", "observe", "reflect"]:
            assert name in g.nodes, f"Missing node: {name}"

    def test_debate_graph(self):
        g = build_debate_graph()
        assert g.entry is not None
        assert len(g.nodes) == 3
        for name in ["proposer", "critic", "judge"]:
            assert name in g.nodes, f"Missing node: {name}"

    def test_pipeline_graph(self):
        a = PromptNode("step1", instructions="First")
        b = PromptNode("step2", instructions="Second")
        c = PromptNode("step3", instructions="Third")

        g = build_pipeline_graph(a, b, c)

        assert g.entry == "step1"
        assert len(g.nodes) == 3
        # Verify edges chain correctly
        edges_from_1 = g.get_edges_from("step1")
        assert any(e.to_node == "step2" for e in edges_from_1)
        edges_from_2 = g.get_edges_from("step2")
        assert any(e.to_node == "step3" for e in edges_from_2)

    def test_all_patterns_have_entry(self):
        """Every pattern must produce a graph with a valid entry node."""
        patterns = [
            build_react_graph(),
            build_peoatr_graph(),
            build_research_graph(),
            build_autonomous_graph(),
            build_deliberate_graph(),
            build_debate_graph(),
            build_pipeline_graph(PromptNode("a"), PromptNode("b")),
        ]
        for g in patterns:
            assert g.entry is not None, f"Graph {g.name!r} has no entry"
            assert g.entry in g.nodes, f"Graph {g.name!r} entry {g.entry!r} not in nodes"


# ═══════════════════════════════════════════════════════════════════════════════
# Pattern Execution Through Engine
# ═══════════════════════════════════════════════════════════════════════════════


class TestPatternExecution:
    @pytest.mark.asyncio
    async def test_pipeline_executes_all_nodes(self):
        """Pipeline graph should visit all nodes in order."""
        visited = []

        @node("a", default_next="b")
        async def a(state: GraphState) -> NodeResult:
            visited.append("a")
            return NodeResult(node_name="a", output="ok")

        @node("b", default_next="c")
        async def b(state: GraphState) -> NodeResult:
            visited.append("b")
            return NodeResult(node_name="b", output="ok")

        @node("c", default_next="__end__")
        async def c(state: GraphState) -> NodeResult:
            visited.append("c")
            return NodeResult(node_name="c", output="ok")

        g = build_pipeline_graph(a, b, c)
        model = MagicMock(spec=["ainvoke", "bind_tools"])
        model.ainvoke = AsyncMock(return_value=AIMessage(content="mock"))
        model.bind_tools = MagicMock(return_value=model)

        engine = PromptGraphEngine(graph=g, model=model)
        await engine.ainvoke({"messages": []})

        assert visited == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_graph_serialization_roundtrip(self):
        """Verify graphs can be serialized and deserialized."""
        from promptise.engine.serialization import graph_from_config, graph_to_config

        g = build_peoatr_graph()
        config = graph_to_config(g)
        g2 = graph_from_config(config)

        assert g2.name == g.name
        assert set(g2.nodes.keys()) == set(g.nodes.keys())
        assert g2.entry == g.entry
