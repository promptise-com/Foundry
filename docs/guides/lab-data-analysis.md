# Lab: Data Analysis Agent

Build an agent that converts natural language questions into SQL queries, cross-references data across tables, and produces accurate analytical reports. Uses a specialized reasoning pattern that outperforms generic ReAct by 60% on accuracy benchmarks.

## What You'll Build

An agent that:

- Accepts natural language analytical questions
- Plans which tables and queries are needed
- Executes SQL queries against a database
- Cross-references results across tables
- Validates calculations before presenting
- Produces structured reports with exact numbers

## Prerequisites

```bash
pip install promptise
export OPENAI_API_KEY=sk-...
```

## Step 1 — Build the Database Server

Create an MCP server with SQL-like query tools.

```python
# analytics_server.py
from promptise.mcp.server import MCPServer

server = MCPServer("analytics")

# Sample business data
EMPLOYEES = [
    {"id": 1, "name": "Alice", "dept": "Engineering", "salary": 185000, "region": "NA"},
    {"id": 2, "name": "Bob", "dept": "Engineering", "salary": 165000, "region": "EU"},
    {"id": 3, "name": "Carol", "dept": "Sales", "salary": 120000, "region": "NA"},
    {"id": 4, "name": "David", "dept": "Sales", "salary": 135000, "region": "APAC"},
    {"id": 5, "name": "Eva", "dept": "Marketing", "salary": 105000, "region": "NA"},
]

DEALS = [
    {"customer": "Acme", "amount": 450000, "stage": "closed_won", "quarter": "Q1-2025", "rep_id": 3},
    {"customer": "Globex", "amount": 280000, "stage": "closed_won", "quarter": "Q1-2025", "rep_id": 4},
    {"customer": "Initech", "amount": 120000, "stage": "closed_lost", "quarter": "Q2-2025", "rep_id": 3},
    {"customer": "Umbrella", "amount": 890000, "stage": "closed_won", "quarter": "Q2-2025", "rep_id": 4},
    {"customer": "Stark", "amount": 1200000, "stage": "negotiation", "quarter": "Q3-2025", "rep_id": 3},
]

@server.tool()
async def query_employees(department: str = "", region: str = "") -> str:
    """Query employees. Optional filter by department or region."""
    rows = EMPLOYEES
    if department:
        rows = [r for r in rows if r["dept"].lower() == department.lower()]
    if region:
        rows = [r for r in rows if r["region"].lower() == region.lower()]
    lines = [f"  {r['name']}: {r['dept']}, ${r['salary']:,}, {r['region']}" for r in rows]
    return f"Found {len(rows)} employees:\n" + "\n".join(lines)

@server.tool()
async def query_deals(stage: str = "", quarter: str = "") -> str:
    """Query deals. Optional filter by stage (closed_won/closed_lost/negotiation) or quarter."""
    rows = DEALS
    if stage:
        rows = [r for r in rows if r["stage"] == stage.lower()]
    if quarter:
        rows = [r for r in rows if r["quarter"] == quarter]
    lines = [f"  {r['customer']}: ${r['amount']:,} ({r['stage']}, {r['quarter']})" for r in rows]
    return f"Found {len(rows)} deals:\n" + "\n".join(lines)

@server.tool()
async def aggregate(table: str, operation: str, column: str, filter_field: str = "", filter_value: str = "") -> str:
    """Run SUM/AVG/COUNT/MIN/MAX on a table column. Optional filter."""
    data = {"employees": EMPLOYEES, "deals": DEALS}.get(table, [])
    if filter_field and filter_value:
        data = [r for r in data if str(r.get(filter_field, "")).lower() == filter_value.lower()]
    vals = [r[column] for r in data if isinstance(r.get(column), (int, float))]
    if operation == "SUM": return f"SUM({column}) = {sum(vals):,}"
    if operation == "AVG": return f"AVG({column}) = {sum(vals)/len(vals):,.2f}" if vals else "0"
    if operation == "COUNT": return f"COUNT = {len(data)}"
    if operation == "MAX": return f"MAX({column}) = {max(vals):,}" if vals else "0"
    if operation == "MIN": return f"MIN({column}) = {min(vals):,}" if vals else "0"
    return f"Unknown operation: {operation}"

@server.tool()
async def calculate(expression: str) -> str:
    """Evaluate a math expression for derived calculations."""
    try:
        allowed = set("0123456789+-*/.() ")
        if all(c in allowed for c in expression):
            return str(round(eval(expression), 2))  # noqa: S307
        return f"Invalid: {expression}"
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    server.run(transport="stdio")
```

## Step 2 — Build the Specialized Reasoning Pattern

The key insight: a domain-specific prompt with schema documentation produces better results than a generic "use tools" instruction.

```python
# analyst_agent.py
import asyncio
import sys
from promptise import build_agent
from promptise.config import StdioServerSpec
from promptise.engine import PromptGraph, PromptNode, NodeFlag
from promptise.engine.reasoning_nodes import (
    PlanNode, ObserveNode, ReflectNode, SynthesizeNode,
)

async def main():
    graph = PromptGraph("data-analyst", nodes=[
        # Plan: decompose the question into specific queries
        PlanNode("plan_queries", is_entry=True),

        # Execute: run the planned queries
        PromptNode("execute",
            instructions=(
                "Execute the planned queries using available tools.\n\n"
                "TABLES:\n"
                "- employees: id, name, dept, salary, region\n"
                "- deals: customer, amount, stage, quarter, rep_id\n\n"
                "STRATEGY:\n"
                "1. Use 'aggregate' for SUM/AVG/COUNT — fastest path\n"
                "2. Use 'query_employees' or 'query_deals' for row-level data\n"
                "3. Use 'calculate' for derived math\n"
                "4. Execute ALL planned queries — don't stop early"
            ),
            inject_tools=True,
            flags={NodeFlag.RETRYABLE},
        ),

        # Observe: interpret raw results
        ObserveNode("interpret"),

        # Reflect: verify the data makes sense
        ReflectNode("verify"),

        # Synthesize: produce the final report
        SynthesizeNode("report", is_terminal=True),
    ])

    agent = await build_agent(
        model="openai:gpt-4o-mini",
        servers={
            "db": StdioServerSpec(
                command=sys.executable,
                args=["analytics_server.py"],
            ),
        },
        agent_pattern=graph,
        instructions=(
            "You are a senior data analyst. Answer questions by querying "
            "the database. Be precise with numbers. Show your work."
        ),
    )

    # Test questions
    questions = [
        "What is the total closed-won deal revenue?",
        "Who is the top sales rep by closed-won revenue? Look up their name.",
        "What is the average Engineering salary and how does it compare to the company average?",
    ]

    for q in questions:
        print(f"\n{'='*60}")
        print(f"Q: {q}")
        print('='*60)

        result = await agent.ainvoke({"messages": [{"role": "user", "content": q}]})

        for msg in reversed(result["messages"]):
            if getattr(msg, "type", "") == "ai" and msg.content:
                print(msg.content)
                break

    await agent.shutdown()

asyncio.run(main())
```

## Step 3 — Add Caching for Repeated Queries

Use the CACHEABLE flag to avoid re-executing identical queries:

```python
PromptNode("execute",
    ...,
    flags={NodeFlag.RETRYABLE, NodeFlag.CACHEABLE},
    input_keys=["query_plan"],  # Cache key based on the plan
)
```

Same question asked twice? The second call returns instantly from cache.

## Step 4 — Add Semantic Cache for Similar Questions

```python
from promptise.cache import SemanticCache

agent = await build_agent(
    ...,
    cache=SemanticCache(backend="memory", similarity_threshold=0.9),
)
```

"What was Q1 revenue?" and "How much revenue did we make in Q1?" will return the same cached answer.

## Why This Pattern Wins

In benchmarks against generic ReAct (LangGraph's default), this pattern:

- **Scored 8/13 accuracy** vs ReAct's 5/13 on complex analytical questions
- **Used fewer tool calls** (the plan step prevents redundant queries)
- **Produced more precise answers** (the verify step catches calculation errors)

The key difference: the schema-aware prompt tells the LLM exact column names and types. Generic ReAct guesses.

## Next Steps

- [Custom Reasoning Guide](custom-reasoning.md) — Build more complex patterns
- [Node Flags Reference](../core/engine-flags.md) — All 18 production flags
- [Reasoning Patterns](../core/agents/reasoning-patterns.md) — 10 built-in patterns
