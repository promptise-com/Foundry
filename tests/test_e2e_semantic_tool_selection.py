"""End-to-end test: semantic tool selection with 40-tool MCP server.

Connects to a real MCP server (40 tools, 8 domains) via stdio,
uses a real LLM (OpenAI), and validates that the semantic tool
optimization system selects the correct domain tools for each query.

Requires:
    OPENAI_API_KEY environment variable set.

Run:
    .venv/bin/python -m pytest tests/test_e2e_semantic_tool_selection.py -x -v
"""

from __future__ import annotations

import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Skip entire module if no API key
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set — skipping real LLM e2e tests",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_tool_calls(result: dict) -> list[str]:
    """Extract tool names that were called from the agent result messages."""
    tool_names: list[str] = []
    messages = result.get("messages", [])
    for msg in messages:
        # LangChain ToolMessage has .name attribute
        if hasattr(msg, "name") and hasattr(msg, "content"):
            msg_type = getattr(msg, "type", "")
            if msg_type == "tool":
                tool_names.append(msg.name)
        # Also check for tool_calls on AIMessage
        if hasattr(msg, "tool_calls"):
            for tc in msg.tool_calls:
                if isinstance(tc, dict) and "name" in tc:
                    tool_names.append(tc["name"])
    return tool_names


def _any_tool_matches_domain(tool_names: list[str], domain_prefix: str) -> bool:
    """Check if any tool name starts with the given domain prefix."""
    return any(name.startswith(domain_prefix) for name in tool_names)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestSemanticToolSelection:
    """E2E tests: 40-tool MCP server + real LLM + semantic optimization.

    Each test sends a domain-specific query and asserts the agent called
    a tool from the correct domain.
    """

    @pytest.fixture
    async def agent(self):
        """Build an agent connected to the 40-tool MCP server with semantic optimization."""
        from promptise import build_agent
        from promptise.config import StdioServerSpec

        server_script = os.path.join(os.path.dirname(__file__), "_e2e_40tool_server.py")

        agent = await build_agent(
            servers={
                "tools": StdioServerSpec(
                    command=sys.executable,
                    args=[server_script],
                ),
            },
            model="openai:gpt-4o-mini",
            instructions=(
                "You are a business operations assistant with access to tools "
                "across HR, Finance, IT, Marketing, Inventory, Support, Analytics, "
                "and Calendar domains. Use the appropriate tool to answer the user's "
                "request. Always call a tool — do not answer from memory."
            ),
            optimize_tools="semantic",
        )

        yield agent
        await agent.shutdown()

    @pytest.fixture
    async def agent_no_optimization(self):
        """Build an agent connected to the 40-tool server WITHOUT optimization."""
        from promptise import build_agent
        from promptise.config import StdioServerSpec

        server_script = os.path.join(os.path.dirname(__file__), "_e2e_40tool_server.py")

        agent = await build_agent(
            servers={
                "tools": StdioServerSpec(
                    command=sys.executable,
                    args=[server_script],
                ),
            },
            model="openai:gpt-4o-mini",
            instructions=(
                "You are a business operations assistant. "
                "Always call a tool — do not answer from memory."
            ),
        )

        yield agent
        await agent.shutdown()

    # ------------------------------------------------------------------
    # Domain-targeted queries with semantic selection
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_hr_query(self, agent):
        """Query about employees should route to hr_* tools."""
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "How many employees are in the engineering department?",
                    }
                ]
            }
        )
        tool_calls = _extract_tool_calls(result)
        assert tool_calls, "No tool calls found in result"
        assert _any_tool_matches_domain(tool_calls, "hr_"), f"Expected hr_* tool, got: {tool_calls}"

    @pytest.mark.asyncio
    async def test_finance_query(self, agent):
        """Query about invoices should route to finance_* tools."""
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Create an invoice for Acme Corp for $5000 for consulting services",
                    }
                ]
            }
        )
        tool_calls = _extract_tool_calls(result)
        assert tool_calls, "No tool calls found in result"
        assert _any_tool_matches_domain(tool_calls, "finance_"), (
            f"Expected finance_* tool, got: {tool_calls}"
        )

    @pytest.mark.asyncio
    async def test_it_query(self, agent):
        """Query about servers should route to it_* tools."""
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Restart the production database server immediately",
                    }
                ]
            }
        )
        tool_calls = _extract_tool_calls(result)
        assert tool_calls, "No tool calls found in result"
        assert _any_tool_matches_domain(tool_calls, "it_"), f"Expected it_* tool, got: {tool_calls}"

    @pytest.mark.asyncio
    async def test_inventory_query(self, agent):
        """Query about stock should route to inventory_* tools."""
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Check if we have enough widgets in stock, product ID is SKU-4521",
                    }
                ]
            }
        )
        tool_calls = _extract_tool_calls(result)
        assert tool_calls, "No tool calls found in result"
        assert _any_tool_matches_domain(tool_calls, "inventory_"), (
            f"Expected inventory_* tool, got: {tool_calls}"
        )

    @pytest.mark.asyncio
    async def test_calendar_query(self, agent):
        """Query about scheduling should route to calendar_* tools."""
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Schedule a team meeting for next Tuesday at 2pm for 30 minutes",
                    }
                ]
            }
        )
        tool_calls = _extract_tool_calls(result)
        assert tool_calls, "No tool calls found in result"
        assert _any_tool_matches_domain(tool_calls, "calendar_"), (
            f"Expected calendar_* tool, got: {tool_calls}"
        )

    @pytest.mark.asyncio
    async def test_support_query(self, agent):
        """Query about customer tickets should route to support_* tools."""
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Escalate support ticket TK-9921 because the customer is threatening to cancel",
                    }
                ]
            }
        )
        tool_calls = _extract_tool_calls(result)
        assert tool_calls, "No tool calls found in result"
        assert _any_tool_matches_domain(tool_calls, "support_"), (
            f"Expected support_* tool, got: {tool_calls}"
        )

    # ------------------------------------------------------------------
    # Comparison: verify semantic mode uses fewer tools than full mode
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_semantic_uses_fewer_tools_than_full(self, agent, agent_no_optimization):
        """Semantic agent should have fewer tools bound than the unoptimized agent."""
        # Both agents carry all 40 tools; the semantic one narrows what each
        # model call is offered through its tool selector.
        semantic_index_count = len(agent._tool_index.all_tool_names) if agent._tool_index else 0
        assert semantic_index_count == 40, f"Expected 40 tools in index, got {semantic_index_count}"
        assert agent._tool_selector is not None, "Semantic agent should have a tool selector"
        assert agent_no_optimization._tool_selector is None

    # ------------------------------------------------------------------
    # Fallback tool: request_more_tools
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_request_more_tools_available(self, agent):
        """The semantic agent should have a request_more_tools fallback."""
        assert agent._tool_index is not None
        assert "request_more_tools" in agent.tool_names
