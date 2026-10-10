"""Lab: MCPcast the Storefront API — generate, drive, govern, measure, improve.

A fictional commerce SaaS already ships a REST API (``storefront.yaml``). It does
not want to build a chatbot; it wants its customers to use Claude, Cursor or any
other AI client *with the product*. This lab does that end to end, offline except
for the real model calls:

  1. GENERATE  ``promptise.mcpcast`` turns the spec into an editable project —
     a risk-classified tool plan, ``server.py`` and a README — under the ``full``
     profile with ``--auth api-key``. Every operation it leaves out is recorded
     with a reason.
  2. DRIVE     A real ``build_agent("openai:gpt-5-mini")`` answers a business
     question through the generated server, in-process, against a fake upstream
     that records every HTTP request it receives.
  3. GOVERN    The agent tries to refund an order. The server-side approval gate
     holds the call: nothing reaches the API until a *different* human of the
     same tenant approves it — and reviewers of other tenants see nothing.
  4. MEASURE   The Agent Readiness Score: real tasks, real agent, spec-derived
     mocks, a grade and specific tool-design fixes.
  5. IMPROVE   Apply the fixes to the plan in code, regenerate, re-run the same
     tasks, and compare the grades honestly.

Only the upstream HTTP API is faked (``fake_api.py``), the way you would fake
Stripe in your own test suite. The agents, tools and approvals are real.

Run:
    OPENAI_API_KEY=... .venv/bin/python examples/mcp/mcpcast_storefront_lab/run.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from fake_api import FakeStorefront
from langchain_core.messages import HumanMessage

from promptise import build_agent
from promptise.mcp.server import TestClient
from promptise.mcpcast import (
    AuthMode,
    CallRecorder,
    DroppedOp,
    EvalReport,
    EvalTask,
    MCPcastPlan,
    ParamPlan,
    RiskClass,
    SafetyProfile,
    ToolPlan,
    classify,
    evaluate,
    extract_operations,
    load_generated_server,
    load_spec,
    mcpcast,
    tools_from_server,
    write_eval,
    write_project,
)
from promptise.models import load_dotenv_if_present

HERE = Path(__file__).resolve().parent
# Passed by its relative name (main() runs from this directory), so the `spec_source`
# recorded in the plan and generated docstrings is portable, never an absolute path.
SPEC = "storefront.yaml"
OUT = HERE / "generated" / "v1"
OUT_V2 = HERE / "generated" / "v2"
MODEL = "openai:gpt-5-mini"

# MCP clients authenticate with an API key; each key names the tenant whose
# upstream credential the server presents. Reviewers carry the "approver" role.
AGENT_KEY = "sk-northwind-agent"
REVIEWER_KEY = "sk-northwind-dana"
SELF_APPROVE_KEY = "sk-northwind-agent-approver"
OTHER_TENANT_KEY = "sk-globex-gus"
CLIENT_KEYS = {
    AGENT_KEY: {"client_id": "support-agent", "tenant_id": "northwind"},
    REVIEWER_KEY: {"client_id": "dana", "tenant_id": "northwind", "roles": ["approver"]},
    # The same principal as the caller, holding the approver role — four-eyes
    # still refuses: nobody approves their own request.
    SELF_APPROVE_KEY: {
        "client_id": "support-agent",
        "tenant_id": "northwind",
        "roles": ["approver"],
    },
    OTHER_TENANT_KEY: {"client_id": "gus", "tenant_id": "globex", "roles": ["approver"]},
}
UPSTREAM_TOKENS = {
    "northwind": "Bearer northwind-upstream-token",
    "globex": "Bearer globex-upstream-token",
}


def section(number: int, title: str) -> None:
    """Print a numbered section header."""
    print(f"\n{'=' * 78}\n{number}. {title}\n{'=' * 78}")


def configure_environment() -> None:
    """Configure the generated server before it is imported.

    ``MCPCAST_CLIENT_KEYS`` and ``MCPCAST_UPSTREAM_TOKENS`` are what an operator
    sets in production; the approval timeout is shortened so the lab does not
    sit for five minutes if nobody reviews.
    """
    os.environ["MCPCAST_CLIENT_KEYS"] = json.dumps(CLIENT_KEYS)
    os.environ["MCPCAST_UPSTREAM_TOKENS"] = json.dumps(UPSTREAM_TOKENS)
    os.environ.setdefault("MCPCAST_APPROVAL_TIMEOUT", "90")


def import_generated(out_dir: Path) -> ModuleType:
    """Import a generated project through its launcher (a fresh package each time)."""
    return load_generated_server(out_dir / "server.py")


def final_text(result: Any) -> str:
    """The final assistant text of an agent invocation."""
    content = result["messages"][-1].content
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def js(value: Any) -> str:
    """Compact JSON for printing (keeps text the model wrote readable)."""
    return json.dumps(value, ensure_ascii=False)


def show_calls(recorder: CallRecorder, task_id: str) -> None:
    """Print the tools the agent chose, in order."""
    for call in recorder.calls_for(task_id):
        args = js({k: v for k, v in call.arguments.items() if v is not None})
        print(f"    {call.tool}({args}) -> {'ok' if call.ok else call.error_code}")


# ---------------------------------------------------------------------------
# 1. Generate
# ---------------------------------------------------------------------------


def generate() -> tuple[MCPcastPlan, list[Any]]:
    """Spec -> risk classification -> plan -> an editable project on disk."""
    section(1, "Generate the MCP server from the OpenAPI spec")
    operations = extract_operations(load_spec(SPEC))
    plan = mcpcast(SPEC, profile=SafetyProfile.FULL, auth=AuthMode.API_KEY, name="storefront")
    for path in write_project(plan, OUT):
        print(f"  wrote {path.relative_to(HERE)}")

    print(f"\n  {'tool':<22}{'risk':<13}{'approval':<10}upstream operation")
    for tool in plan.tools:
        route = tool.routes[0]
        gate = "human" if tool.requires_approval else "-"
        print(f"  {tool.name:<22}{tool.risk.value:<13}{gate:<10}{route.method} {route.path}")

    print("\n  why the classifier decided that (deterministic, no model):")
    reasons = {op.operation_id: classify(op) for op in operations}
    for tool in plan.tools:
        cls = reasons[tool.routes[0].operation_id]
        # Plain "GET is a read" / "POST is a write" needs no explanation.
        if cls.risk is not cls.base or not cls.reasons[0].endswith(("is a read", "is a write")):
            print(f"    {tool.name:<22}{'; '.join(cls.reasons)}")

    print("\n  not exposed (nothing vanishes silently):")
    for dropped in plan.dropped:
        print(f"    {dropped.operation_id}: {dropped.reason}")

    print("\n  the same spec under each safety profile:")
    for profile in SafetyProfile:
        other = mcpcast(SPEC, profile=profile, auth=AuthMode.API_KEY, name="storefront")
        print(
            f"    {profile.value:<12}{len(other.tools):>3} tools "
            f"({len(other.gated_tools)} approval-gated), {len(other.dropped)} not exposed"
        )
    print(f"\n  auth: {plan.api.auth.value}   approval: {plan.api.approval_mode.value} (four-eyes)")
    return plan, operations


# ---------------------------------------------------------------------------
# 2. Drive
# ---------------------------------------------------------------------------


async def business_task(module: ModuleType) -> None:
    """A real agent answers a real business question through the generated server."""
    section(2, "A real agent drives the generated server")
    api = FakeStorefront()
    question = (
        "A customer wrote in from ada@northwind.example. Who are they, "
        "and what are their two most recent orders?"
    )
    async with api.client() as http:
        server = module.build_server(http_client=http)
        recorder = CallRecorder()
        recorder.begin("lookup")
        tools = await tools_from_server(server, recorder=recorder, headers={"x-api-key": AGENT_KEY})
        agent = await build_agent(
            model=MODEL,
            servers={},
            extra_tools=tools,
            instructions=(
                "You are a support assistant for the Storefront commerce platform. "
                "Answer with data you fetched from the tools. Only read data in this "
                "conversation — never call a tool that changes anything."
            ),
            max_agent_iterations=6,
        )
        try:
            print(f"  question: {question}")
            result = await agent.ainvoke({"messages": [HumanMessage(content=question)]})
            print("\n  tools the agent chose:")
            show_calls(recorder, "lookup")
            print(f"\n  answer: {final_text(result)}")
        finally:
            await agent.shutdown()
    print(f"\n  requests the API received: {api.log}")
    if api.calls:
        credentials = sorted({c.authorization or "(none)" for c in api.calls})
        print(f"  the credential each one carried: {', '.join(repr(c) for c in credentials)}")


# ---------------------------------------------------------------------------
# 3. Govern
# ---------------------------------------------------------------------------


async def wait_for_pending(client: TestClient, tool: str, timeout: float = 60.0) -> Any:
    """Poll ``approvals_list`` as the reviewer until *tool* is waiting, or give up."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        (reply,) = await client.call_tool("approvals_list", {}, headers={"x-api-key": REVIEWER_KEY})
        entries = json.loads(reply.text)
        if isinstance(entries, list):
            match = next((e for e in entries if e["tool"] == tool), None)
            if match is not None:
                return match
        await asyncio.sleep(0.25)
    return None


async def governance(module: ModuleType) -> None:
    """The refund is held by the server until a second human releases it."""
    section(3, "Governance: the refund waits for a human")
    api = FakeStorefront()
    request = "Order ORD-1003 arrived damaged. Refund the customer 24 euros for it."
    async with api.client() as http:
        server = module.build_server(http_client=http)
        reviewer = TestClient(server)
        recorder = CallRecorder()
        recorder.begin("refund")
        tools = await tools_from_server(server, recorder=recorder, headers={"x-api-key": AGENT_KEY})
        agent = await build_agent(
            model=MODEL,
            servers={},
            extra_tools=tools,
            instructions=(
                "You are a support assistant for the Storefront commerce platform. "
                "Carry out the request with the tools you have. Be brief."
            ),
            max_agent_iterations=6,
        )
        try:
            print(f"  request: {request}")
            call = asyncio.create_task(agent.ainvoke({"messages": [HumanMessage(content=request)]}))
            pending = await wait_for_pending(reviewer, "refund_order")
            if pending is None:
                call.cancel()
                print("  the agent never reached refund_order on this run — nothing to approve.")
                return

            print("\n  the gate is holding the call:")
            print(f"    {pending['tool']} {js(pending['arguments'])}")
            print(
                f"    requested by client_id={pending['client_id']!r} tenant={pending['tenant_id']!r}"
            )
            print(
                f"    refund requests the API has received so far: "
                f"{len(api.received('POST', '/refund'))}"
            )
            print(f"    API log so far: {api.log}")

            print("\n  a reviewer from ANOTHER tenant:")
            (reply,) = await reviewer.call_tool(
                "approvals_list", {}, headers={"x-api-key": OTHER_TENANT_KEY}
            )
            print(f"    globex/gus approvals_list -> {reply.text}   (sees nothing)")
            (reply,) = await reviewer.call_tool(
                "approvals_decide",
                {"request_id": pending["request_id"], "approve": True},
                headers={"x-api-key": OTHER_TENANT_KEY},
            )
            print(f"    globex/gus tries to approve -> {json.loads(reply.text)['error']['code']}")

            print("\n  the caller approving their own request (same principal, approver role):")
            (reply,) = await reviewer.call_tool(
                "approvals_decide",
                {"request_id": pending["request_id"], "approve": True},
                headers={"x-api-key": SELF_APPROVE_KEY},
            )
            error = json.loads(reply.text)["error"]
            print(f"    -> {error['code']}: {error['message']}")

            print("\n  dana (northwind, approver role) approves:")
            (reply,) = await reviewer.call_tool(
                "approvals_decide",
                {"request_id": pending["request_id"], "approve": True, "reason": "damaged goods"},
                headers={"x-api-key": REVIEWER_KEY},
            )
            print(f"    approvals_decide -> {reply.text}")
            result = await asyncio.wait_for(call, timeout=120)
            print("\n  tools the agent chose:")
            show_calls(recorder, "refund")
            print(f"\n  answer: {final_text(result)}")
        finally:
            await agent.shutdown()

    refunds = api.received("POST", "/refund")
    print(f"\n  refund requests the API received: {len(refunds)}")
    for refund in refunds:
        print(f"    {refund} body={js(refund.body)}")
        print(f"    authorization={refund.authorization!r}  (northwind's upstream credential)")
    print(f"  refunds now on record: {js(api.refunds)}")


# ---------------------------------------------------------------------------
# 4. Measure
# ---------------------------------------------------------------------------


def tasks_for(plan: MCPcastPlan) -> list[EvalTask]:
    """The same eight user requests, targeted at whichever plan we are scoring.

    ``--eval`` writes tasks like these with a model; spelling them out keeps the
    two runs comparable, which is the whole point of a before/after.
    """
    merged = "find_customer" in plan.tool_names
    by_email = "find_customer" if merged else "search_customers"
    by_id = "find_customer" if merged else "get_customer"
    requests: list[tuple[str, str, str]] = [
        ("t1", "A customer wrote in from ada@northwind.example — who are they?", by_email),
        ("t2", "Pull up order ORD-1002.", "get_order"),
        ("t3", "Which orders does customer CUS-1001 have, newest first?", "list_orders"),
        ("t4", "Order ORD-1003 arrived damaged — refund 24 euros for it.", "refund_order"),
        ("t5", "How much revenue did we make in 2026-02?", "get_revenue_report"),
        ("t6", "Show me the record for customer CUS-1002.", by_id),
        ("t7", "Which of our customers are on the enterprise plan?", "list_customers"),
        (
            "t8",
            "Start a draft order for customer CUS-1003: one MAT-DESK-90 at 89 euros.",
            "create_order",
        ),
    ]
    return [EvalTask(id=i, prompt=prompt, expected_tool=tool) for i, prompt, tool in requests]


async def measure(
    plan: MCPcastPlan, module: ModuleType, operations: list[Any], out_dir: Path
) -> EvalReport:
    """Run the Agent Readiness evaluation and print the grade with its evidence."""
    tasks = tasks_for(plan)
    report = await evaluate(
        plan,
        module.build_server,
        model=MODEL,
        tasks=tasks,
        operations=operations,
        live_reads=False,  # this API is fictional; mock every route from the spec
        headers={"x-api-key": AGENT_KEY},
    )
    print(f"\n  Agent Readiness: {report.grade}   score {report.score:.2f}")
    print(
        f"    tasks succeeded {report.tasks_succeeded}/{report.tasks_total}   "
        f"correct tool first {report.selection_rate:.0%}   "
        f"parameter errors {report.param_error_rate:.0%}"
    )
    for result in report.results:
        called = " -> ".join(c.tool for c in result.calls) or "(no tool call)"
        print(
            f"    {result.task.id}  expected {result.task.expected_tool:<19}"
            f"called {called:<46} {'ok' if result.success else 'MISS'}"
        )
    print("\n  what the report says to fix:")
    for fix in report.fixes:
        print(f"    {fix}")
    tasks_path, report_path = write_eval(report, tasks, out_dir)
    print(f"\n  wrote {tasks_path.relative_to(HERE)} and {report_path.relative_to(HERE)}")
    return report


# ---------------------------------------------------------------------------
# 5. Improve
# ---------------------------------------------------------------------------


def improve(plan: MCPcastPlan) -> MCPcastPlan:
    """Apply the report's advice to the plan — the plan is the source of truth.

    Four edits, each one something the readiness report asked for:

    - merge the two customer lookups into one intent tool with two routes,
      so the agent picks a *customer*, not an HTTP endpoint;
    - give ``list_customers`` and ``list_orders`` the examples the report
      says they are missing;
    - drop the health check no agent task ever needs;
    - put ``notify_customer`` on a param diet: hidden, and always ``false``,
      so an agent creating a draft order can never email a customer.
    """
    get_customer = plan.tool("get_customer")
    search_customers = plan.tool("search_customers")
    find_customer = ToolPlan(
        name="find_customer",
        description=(
            "Find ONE customer, by id or by email address. Pass customer_id when you "
            "know it (e.g. 'CUS-1001'), otherwise pass email. Use list_customers only "
            "to browse or filter the whole directory."
        ),
        risk=RiskClass.READ,
        routes=[get_customer.routes[0], search_customers.routes[0]],
        params={
            "customer_id": ParamPlan(
                description="The customer identifier, e.g. 'CUS-1001'.",
                json_schema={"type": "string"},
            ),
            "email": ParamPlan(
                description="The customer's email address, matched exactly.",
                json_schema={"type": "string", "format": "email"},
            ),
        },
        example={"email": "ada@northwind.example"},
        tags=["customers"],
    )

    create_order = plan.tool("create_order")
    params = dict(create_order.params)
    params["notify_customer"] = params["notify_customer"].model_copy(
        update={"hidden": True, "default": False}
    )
    create_order = create_order.model_copy(update={"params": params})

    replacements = {
        "get_customer": find_customer,
        "create_order": create_order,
        "list_customers": plan.tool("list_customers").model_copy(
            update={"example": {"plan": "enterprise", "limit": 5}}
        ),
        "list_orders": plan.tool("list_orders").model_copy(
            update={"example": {"customer_id": "CUS-1001", "status": "open"}}
        ),
    }
    tools = [
        replacements.get(tool.name, tool)
        for tool in plan.tools
        if tool.name not in ("get_health", "search_customers")
    ]
    return MCPcastPlan(
        api=plan.api,
        profile=plan.profile,
        tools=tools,
        dropped=[
            *plan.dropped,
            DroppedOp(
                operation_id="getHealth",
                reason="operational endpoint — no agent task needs it (Agent Readiness: not covered)",
            ),
        ],
    )


# ---------------------------------------------------------------------------


async def main() -> None:
    os.chdir(HERE)
    configure_environment()
    plan, operations = generate()

    load_dotenv_if_present()  # .env next to the project, as build_agent() would
    if not os.environ.get("OPENAI_API_KEY"):
        print(
            "\nSteps 2-5 drive a real agent: set OPENAI_API_KEY and run this again.\n"
            "  export OPENAI_API_KEY=sk-..."
        )
        sys.exit(1)

    module = import_generated(OUT)
    await business_task(module)
    await governance(module)

    section(4, "Measure: the Agent Readiness Score")
    print("  8 tasks, a real agent, the full server pipeline, spec-derived mock responses")
    print("  (nothing an evaluation does can reach real data)")
    before = await measure(plan, module, operations, OUT)

    section(5, "Improve the plan, regenerate, re-measure")
    better = improve(plan)
    for path in write_project(better, OUT_V2):
        print(f"  wrote {path.relative_to(HERE)}")
    print(
        f"  merged get_customer + search_customers -> find_customer "
        f"({len(better.tool('find_customer').routes)} routes, dispatched by which id you pass)"
    )
    print("  added the missing examples; dropped get_health")
    print(f"  create_order description now says: {'Always sends: notify_customer=false'!r}")
    print(
        f"  surface: {len(plan.tools)} -> {len(better.tools)} tools, "
        f"{sum(1 for t in plan.tools if t.example)} -> "
        f"{sum(1 for t in better.tools if t.example)} of them with a worked example"
    )
    after = await measure(better, import_generated(OUT_V2), operations, OUT_V2)

    print(
        f"\n  before: {before.grade} ({before.score:.2f})   after: {after.grade} "
        f"({after.score:.2f})"
    )
    if after.score > before.score:
        print("  the edits paid off — the same eight requests land more often now.")
    elif after.score == before.score:
        print(
            "  identical score on this run — the surface was already unambiguous for these "
            "eight requests, and the edits kept it that way with two fewer tools. That is a "
            "real win (less to choose from, fewer tokens), and it is what honest measurement "
            "looks like: the number does not move just because you changed something."
        )
    else:
        print(
            "  the score went DOWN. That is the point of measuring: revert the edit, or "
            "sharpen the merged description, and run it again."
        )

    print("\nNext: point Claude Desktop / Claude Code / Cursor at generated/v2/server.py")
    print("  (see README.md), or run the same pipeline from the CLI:")
    print("    promptise mcpcast examples/mcp/mcpcast_storefront_lab/storefront.yaml \\")
    print("      --profile full --auth api-key --out storefront-mcp --eval")


if __name__ == "__main__":
    asyncio.run(main())
