"""Plan → Act → Verify — a custom reasoning graph, saved to YAML and run.

Demonstrates a custom ``PromptGraph`` passed to ``build_agent``:

- ``inject_tools=True`` — the ``act`` node receives the tools ``build_agent``
  discovers from the MCP server, without naming them in the graph.
- Pydantic ``output_schema`` — the plan and the verdict are structured, so
  ``ValidateNode`` routes on ``passes`` (pass → end, fail → reflect → act).
- ``max_iterations=3`` on ``verify`` — the draft is checked at most three
  times; after that the engine stops looping and the run ends.
- ``save_graph`` / ``load_graph`` — the graph round-trips through YAML: the
  schemas are stored as import references and ``act`` keeps its
  ``inject_tools`` flag.
- ``agent.last_report`` — path, tool calls and token usage of the run.

The example starts its own MCP server (this file, run with ``--serve``).

Run:
    export OPENAI_API_KEY=...
    python examples/reasoning/plan_act_verify.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from typing import Any

from pydantic import BaseModel, Field

from promptise import build_agent
from promptise.config import StdioServerSpec
from promptise.engine import PlanNode, PromptGraph, PromptNode, ReflectNode, ValidateNode
from promptise.engine.serialization import load_graph, save_graph
from promptise.mcp.server import MCPServer

# ── The MCP server: a stand-in sales warehouse (amounts in USD) ──────────────

server = MCPServer("sales")

TARGETS = {"Q3": {"North": 300_000, "South": 250_000, "East": 180_000, "West": 220_000}}
REVENUE = {
    ("North", "Q3"): [("Jul", 104_200, 3_100), ("Aug", 98_750, 2_400), ("Sep", 101_300, 4_050)],
    ("South", "Q3"): [("Jul", 86_400, 1_200), ("Aug", 82_100, 950), ("Sep", 88_900, 1_650)],
    ("East", "Q3"): [("Jul", 55_300, 2_900), ("Aug", 61_800, 3_300), ("Sep", 58_050, 2_150)],
    ("West", "Q3"): [("Jul", 74_900, 1_100), ("Aug", 71_250, 2_600), ("Sep", 76_400, 1_900)],
}


@server.tool()
async def get_targets(quarter: str) -> dict:
    """Get each region's net revenue target for a quarter.

    Args:
        quarter: The quarter, for example "Q3".
    """
    return {"quarter": quarter, "net_revenue_targets": TARGETS.get(quarter.upper(), {})}


@server.tool()
async def get_monthly_revenue(region: str, quarter: str) -> dict:
    """Get a region's gross revenue and refunds for each month of a quarter.

    Args:
        region: North, South, East or West.
        quarter: The quarter, for example "Q3".
    """
    rows = REVENUE.get((region.strip().title(), quarter.upper()))
    if rows is None:
        return {"error": f"No data for {region} in {quarter}."}
    months = [{"month": m, "gross": g, "refunds": r} for m, g, r in rows]
    return {"region": region.strip().title(), "quarter": quarter.upper(), "months": months}


# ── Structured outputs (module level, so YAML can reference them) ───────────


class Plan(BaseModel):
    subgoals: list[str] = Field(description="4 or fewer steps, each one tool call or computation")
    quality_score: int = Field(description="Plan quality from 1 to 5")


class Verdict(BaseModel):
    passes: bool = Field(description="Whether the answer meets every criterion")
    issues: list[str] = Field(description="Each criterion the answer misses, and why")


# ── The graph ───────────────────────────────────────────────────────────────


def plan_act_verify(instructions: str) -> PromptGraph:
    graph = PromptGraph("plan-act-verify", mode="static")
    graph.add_node(
        PlanNode(
            "plan",
            output_schema=Plan,
            instructions=(
                f"{instructions}\n\nPlan how to answer the question with the sales tools. "
                "Never plan to ask the user for data a tool can fetch."
            ),
        )
    )
    graph.add_node(
        PromptNode(
            "act",
            instructions=f"{instructions}\n\nWork through the plan with your tools, then write the answer.",
            inject_tools=True,  # gets the MCP tools build_agent discovers
        )
    )
    graph.add_node(
        ValidateNode(
            "verify",
            criteria=[
                "Names every region that missed, with the shortfall in dollars and percent",
                "Every number matches the tool results",
                "Uses net revenue (gross minus refunds), the definition of the targets",
            ],
            output_schema=Verdict,
            on_pass="__end__",
            on_fail="reflect",
            max_iterations=3,  # check the draft at most three times
        )
    )
    graph.add_node(ReflectNode("reflect", input_keys=["validation"]))
    graph.sequential("plan", "act", "verify")
    graph.always("reflect", "act")
    graph.set_entry("plan")
    return graph


async def main(model: Any = "openai:gpt-5-mini") -> None:
    instructions = "You are a sales analyst. Use the sales tools for every number."
    graph = plan_act_verify(instructions)

    # Round-trip the graph through YAML. The schemas are stored as references
    # ("__main__:Plan" here) and resolved from refs= on load.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "plan_act_verify.yaml")
        save_graph(graph, path)
        graph = load_graph(path, refs=[Plan, Verdict])
    print(graph.describe(), "\n")

    agent = await build_agent(
        model=model,
        servers={"sales": StdioServerSpec(command=sys.executable, args=[__file__, "--serve"])},
        agent_pattern=graph,
    )
    try:
        question = (
            "Which regions missed their Q3 revenue target, and by how much in dollars and percent?"
        )
        result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
        print(result["messages"][-1].content, "\n")
        report = agent.last_report
        print("path:", " → ".join(report.nodes_visited))
        print(f"tool calls: {report.tool_calls}, tokens: {report.total_tokens:,}")
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    if "--serve" in sys.argv:
        server.run(transport="stdio")
    else:
        asyncio.run(main())
