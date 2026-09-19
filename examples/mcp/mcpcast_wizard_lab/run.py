"""Lab: the guided setup, end to end — you drive the wizard, the lab proves the result.

``app.py`` is an ordinary FastAPI service (the Helpdesk API). This driver does
what you would do with your own app, with you at the keyboard for the part
that matters:

  1. START    serve ``app.py`` with uvicorn on a local port the wizard probes
  2. WIZARD   ``promptise mcpcast`` opens in this terminal: detect the running
              API, pick the model (or offline), the safety profile, the auth
              mode, review the plan, write ``generated/helpdesk-mcp/``
  3. INSPECT  the lab prints what was written and the computed review
              warnings — descriptions that name tools the plan does not
              expose, parameters hidden from the agent
  4. DRIVE    ``build_agent("openai:gpt-5-mini")`` launches the generated
              ``server.py`` over the real MCP stdio transport and answers a
              question by calling the live app through the generated tools
  5. GOVERN   the same agent tries to close a ticket and to refund a customer:
              denied fail-closed over stdio (no human to ask) or, under
              ``read-only``, impossible — the live app confirms nothing changed
  6. MEASURE  the Agent Readiness Score: 8 generated tasks, a real agent, the
              full server pipeline, a grade and the fixes to make
  7. NEXT     the exact non-interactive command, how to regenerate and
              re-measure after editing the plan (with the token the live reads
              need), and the line that adds it to Claude Code

``app.py`` is a FastAPI app, so ``fastapi`` must be installed
(``.venv/bin/python -m pip install fastapi``; ``promptise[dev]`` includes it).
Beyond that only ``OPENAI_API_KEY`` is needed (in ``.env`` or exported); the
wizard itself shows whether it is picked up. Pass ``--no-eval`` to skip step 6.

Run:
    .venv/bin/python examples/mcp/mcpcast_wizard_lab/run.py
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from promptise import StdioServerSpec, build_agent
from promptise.mcpcast import (
    AuthMode,
    MCPcastPlan,
    RiskClass,
    SafetyProfile,
    extract_operations,
    load_generated_server,
    load_spec,
)
from promptise.mcpcast.readiness import evaluate, write_eval
from promptise.mcpcast.wizard import (
    PROBE_PORTS,
    WizardResult,
    quote_argument,
    review_warnings,
    run_wizard,
)

try:
    import app as helpdesk  # the Helpdesk API next to this file — a FastAPI app
except ModuleNotFoundError as exc:
    if (exc.name or "").partition(".")[0] != "fastapi":
        raise
    raise SystemExit(
        "app.py is a FastAPI app and fastapi is not a Promptise dependency — "
        ".venv/bin/python -m pip install fastapi"
    ) from None

HERE = Path(__file__).resolve().parent
OUT = HERE / "generated" / "helpdesk-mcp"
MODEL = "openai:gpt-5-mini"
# What the generated server sends upstream as the Authorization header. In a
# desktop client this goes in the client's own config (see the generated README).
UPSTREAM_TOKEN = f"Bearer {helpdesk.DEMO_TOKEN}"
QUESTION = "Which open tickets does customer cus_ada (Ada Lovelace) have, and what is each about?"
WRITE_REQUEST = "Close ticket 2 with the resolution 'export fixed in release 4.3'."
MONEY_REQUEST = "Refund 29.90 EUR on ticket 1; the customer was charged twice."


def section(number: int, title: str) -> None:
    """Print a numbered section header."""
    print(f"\n{'=' * 78}\n{number}. {title}\n{'=' * 78}")


def shell_line(parts: list[str]) -> str:
    """One command line, each argument quoted for this platform's shell."""
    return " ".join(quote_argument(part) for part in parts)


def with_env(name: str, value: str, parts: list[str]) -> list[str]:
    """The lines that run *parts* with ``name=value`` in its environment.

    POSIX shells take the assignment as a prefix on the same line. ``cmd.exe``
    has no such form, so it gets a ``set "NAME=value"`` line first — the quotes
    keep spaces and ``&|<>^`` in the value and are not part of it.
    """
    if os.name == "nt":
        return [f'set "{name}={value}"', shell_line(parts)]
    return [f"{name}={quote_argument(value)} {shell_line(parts)}"]


# ---------------------------------------------------------------------------
# 1. Start your app on a port the wizard looks at
# ---------------------------------------------------------------------------


def free_probe_port() -> int:
    """The first free port among the ones the wizard's detection probes."""
    for port in (8001, 4000, 8888, 9000, 5001, *PROBE_PORTS):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise SystemExit("no free port among the ones the wizard probes; stop a local server")


def start_app() -> tuple[uvicorn.Server, str]:
    """Serve ``app.py`` in a background thread and return the server and its origin."""
    port = free_probe_port()
    config = uvicorn.Config(helpdesk.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if not thread.is_alive():
            raise SystemExit(f"uvicorn failed to start on port {port} (see the error above)")
        if time.monotonic() > deadline:
            raise SystemExit(f"uvicorn did not start on port {port} within 10 s")
        time.sleep(0.05)
    return server, f"http://127.0.0.1:{port}"


# ---------------------------------------------------------------------------
# 2. The wizard — you drive it
# ---------------------------------------------------------------------------


def wizard_card(origin: str) -> None:
    """What to choose at each step, and what to look for."""
    print(
        f"""
  The wizard opens in this terminal. Suggested path for this lab:

    1 API spec   Enter on the detected "Helpdesk API" ({origin}/openapi.json)
                 — 12 operations: 5 read · 4 write · 2 destructive · 1 financial
    2 Model      "Design the tools with a model" if the key line is green,
                 otherwise Offline (everything after still works)
    3 Safety     standard  — reads open, writes approval-gated, refund/delete
                 not generated. Compare the counts on the three rows.
    4 Auth       Personal — a desktop client  (--auth env-token)
    5 Project    keep the defaults (name helpdesk, folder generated/helpdesk-mcp)
    6 Review     read every row. Look for a description that names refund_ticket
                 or delete_ticket — tools this profile did not generate — and for
                 hidden parameters. Fix them later in mcpcast.plan.yaml.
    7 Write      Finish (Enter). The lab continues from here.

  Keys: Enter continue · Esc back · Tab fields · ↑↓ menus · F1 help · Ctrl+Q quit
"""
    )


# ---------------------------------------------------------------------------
# 3. Inspect what was written
# ---------------------------------------------------------------------------


def inspect(result: WizardResult) -> MCPcastPlan:
    """Print the plan the way a reviewer reads it, plus the computed warnings."""
    section(3, "What the wizard wrote — read it like a reviewer")
    plan = result.plan
    for path in result.written:
        print(f"  wrote {path.relative_to(HERE)}")
    print(
        f"\n  profile {plan.profile.value} · auth {plan.api.auth.value} · base_url {plan.api.base_url}"
    )
    print(f"\n  {'tool':<24}{'risk':<13}{'approval':<10}{'upstream operation(s)'}")
    for tool in plan.tools:
        gate = "required" if tool.requires_approval else "-"
        ops = ", ".join(f"{r.method} {r.path}" for r in tool.routes)
        print(f"  {tool.name:<24}{tool.risk.value:<13}{gate:<10}{ops}")
    print("\n  not exposed (each with its reason, recorded in the plan):")
    for dropped in plan.dropped:
        print(f"    {dropped.operation_id}: {dropped.reason}")
    warnings = review_warnings(plan)
    print("\n  computed review warnings:")
    if warnings:
        for line in warnings:
            print(f"    ! {line}")
        print(f"    -> fix in {OUT.relative_to(HERE) / 'mcpcast.plan.yaml'}, then regenerate")
    else:
        print("    none — still read every description; the checks are not a reviewer")
    return plan


# ---------------------------------------------------------------------------
# 4-5. A real agent over MCP stdio, reads and gated writes
# ---------------------------------------------------------------------------


def final_text(result: Any) -> str:
    """The final assistant text of an agent invocation."""
    content = result["messages"][-1].content
    if isinstance(content, list):
        return "".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return str(content)


def show_calls(result: Any) -> None:
    """Print every tool call the agent made, with the server's structured errors."""
    for message in result["messages"]:
        if isinstance(message, AIMessage):
            for call in message.tool_calls:
                print(f"    {call['name']}({json.dumps(call['args'], ensure_ascii=False)})")
        elif isinstance(message, ToolMessage) and '"error"' in str(message.content):
            try:
                error = json.loads(str(message.content))["error"]
            except (ValueError, KeyError):
                continue
            print(f"      -> {error['code']}: {error['message']}")


def tools_of(plan: MCPcastPlan, risk: RiskClass) -> list[str]:
    return [t.name for t in plan.tools if t.risk is risk]


async def drive_and_govern(plan: MCPcastPlan) -> None:
    """The generated server, launched by the agent exactly as Claude Desktop would."""
    section(4, "A real agent over MCP stdio -> generated server.py -> the live app")
    agent = await build_agent(
        model=MODEL,
        servers={
            "helpdesk": StdioServerSpec(
                command=sys.executable,
                args=[str(OUT / "server.py")],
                env={"MCPCAST_UPSTREAM_TOKEN": UPSTREAM_TOKEN},
            )
        },
        instructions=(
            "You are the helpdesk assistant. Answer from what the tools return. Be brief. "
            "If a tool call is denied or no tool can do it, say so plainly and stop."
        ),
        max_agent_iterations=6,
    )
    try:
        print(f"  question: {QUESTION}\n  tools the agent chose:")
        result = await agent.ainvoke({"messages": [HumanMessage(content=QUESTION)]})
        show_calls(result)
        print(f"  answer: {final_text(result)}")

        section(5, "Writes and money: what the safety profile and the gate do")
        writes = tools_of(plan, RiskClass.WRITE)
        money = tools_of(plan, RiskClass.FINANCIAL)
        before = helpdesk.TICKETS[2].status
        print(
            f"  profile {plan.profile.value}: write tools {writes or 'none'} · "
            f"financial tools {money or 'none'}"
        )
        print(f"\n  a) {WRITE_REQUEST}")
        if writes:
            print(
                "     (a write tool exists; over stdio no human can be asked, so the gate denies)"
            )
        else:
            print("     (read-only profile: no write tool was generated at all)")
        print("     tools the agent chose:")
        result = await agent.ainvoke({"messages": [HumanMessage(content=WRITE_REQUEST)]})
        show_calls(result)
        print(f"     answer: {final_text(result)}")
        after = helpdesk.TICKETS[2].status
        if after != before:
            raise SystemExit(
                f"     live app: ticket 2 changed from {before!r} to {after!r} — the write "
                "reached the API without approval; the gate is not doing its job"
            )
        print(f"     live app: ticket 2 is still {after!r} — nothing reached the API")

        print(f"\n  b) {MONEY_REQUEST}")
        if money:
            print(
                "     (profile full: refund_ticket exists and is approval-gated — denied over stdio)"
            )
        else:
            print(
                f"     (profile {plan.profile.value}: no refund tool exists; the agent cannot even try)"
            )
        print("     tools the agent chose:")
        result = await agent.ainvoke({"messages": [HumanMessage(content=MONEY_REQUEST)]})
        show_calls(result)
        print(f"     answer: {final_text(result)}")
        refunded = helpdesk.TICKETS[1].refunded
        if refunded:
            raise SystemExit(
                f"     live app: {refunded:.2f} EUR was refunded — money moved without approval"
            )
        print("     live app: ticket 1 refunded total is 0.00 EUR — nothing reached the API")
    finally:
        await agent.shutdown()


# ---------------------------------------------------------------------------
# 6. Measure
# ---------------------------------------------------------------------------


async def measure(result: WizardResult, spec_url: str) -> None:
    """The Agent Readiness Score for what was written (or the one the wizard already ran)."""
    section(6, "Measure: the Agent Readiness Score")
    report = result.report
    if report is None:
        print(f"  8 generated tasks, a real agent ({MODEL}), the full server pipeline …")
        operations = extract_operations(load_spec(spec_url), spec_url=spec_url)
        module = load_generated_server(OUT / "server.py")
        report = await evaluate(
            result.plan, module.build_server, model=MODEL, tasks=8, operations=operations
        )
        tasks_path, report_path = write_eval(report, [r.task for r in report.results], OUT)
        print(f"  wrote {report_path.relative_to(HERE)} and {tasks_path.relative_to(HERE)}")
    else:
        print("  (the wizard ran the evaluation; this is its report)")
    print()
    print("\n".join(f"  {line}" for line in report.render_summary().splitlines()))


# ---------------------------------------------------------------------------


def main() -> None:
    skip_eval = "--no-eval" in sys.argv[1:]
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise SystemExit(
            "This lab drives the guided setup, which needs an interactive terminal. "
            "Run it from a terminal, not from a pipe or an editor task."
        )

    section(1, "Start the Helpdesk app (app.py) with uvicorn")
    app_server, origin = start_app()
    spec_url = f"{origin}/openapi.json"
    print(f"  serving {origin}  (bearer token: {helpdesk.DEMO_TOKEN!r})")
    print(f"  the wizard will find it at {spec_url}")
    # The Agent Readiness evaluation runs read tools against the live app. In
    # env-token mode that credential comes from MCPCAST_EVAL_AUTHORIZATION —
    # for the lab's own step 6 and for the wizard's evaluation switch alike.
    os.environ.setdefault("MCPCAST_EVAL_AUTHORIZATION", UPSTREAM_TOKEN)

    try:
        section(2, "The guided setup — your turn")
        wizard_card(origin)
        input("  Press Enter to open the wizard … ")
        result = run_wizard(cwd=HERE, out_dir=str(OUT.relative_to(HERE)))
        if result is None:
            raise SystemExit("\nNothing written — the wizard was closed before step 7.")
        if result.out_dir.resolve() != OUT.resolve():
            raise SystemExit(
                f"\nThe project was written to {result.out_dir}; this lab expects "
                f"{OUT.relative_to(HERE)} — keep the default folder in step 5 and run again."
            )
        print(f"\n  Next time, without the wizard:\n    {result.command}")

        plan = inspect(result)
        if plan.api.auth is not AuthMode.ENV_TOKEN:
            raise SystemExit(
                "\nThis lab drives the generated server over stdio with one upstream token "
                f"(auth mode env-token); the wizard wrote auth {plan.api.auth.value!r}. Run the "
                "lab again and choose 'Personal — a desktop client' in step 4 — the other modes "
                "need an HTTP client that sends headers (see the README)."
            )
        if not os.environ.get("OPENAI_API_KEY"):
            print("\nSteps 4-6 drive a real agent: put OPENAI_API_KEY in .env and run this again.")
            raise SystemExit(1)
        asyncio.run(drive_and_govern(plan))
        if skip_eval:
            print("\n  (--no-eval: skipping the Agent Readiness Score)")
        else:
            asyncio.run(measure(result, spec_url))
    finally:
        app_server.should_exit = True

    section(7, "Next")
    plan_file = str(OUT.relative_to(HERE) / "mcpcast.plan.yaml")
    port = origin.rsplit(":", 1)[1]
    print(
        f"  The app is stopped now; the plan records it at {origin}. Start it again on that port:"
    )
    print(f"    {shell_line(['cd', str(HERE)])} && uvicorn app:app --port {port}")
    print(f"  1. Read and fix {plan_file} (the warnings above are the first stops).")
    print(f"  2. Regenerate: {shell_line(['promptise', 'mcpcast', plan_file])}")
    # `--eval` writes a fresh task set from the fixed plan. Its live reads need
    # the API's token, exactly as step 6 did — without it the grade measures
    # the missing credential, not the tools.
    print("  3. Re-measure (app running; the live reads carry the API's token):")
    for line in with_env(
        "MCPCAST_EVAL_AUTHORIZATION",
        UPSTREAM_TOKEN,
        ["promptise", "mcpcast", plan_file, "--eval"],
    ):
        print(f"     {line}")
    claude_line = shell_line(
        [
            "claude",
            "mcp",
            "add",
            "helpdesk",
            "-e",
            f"MCPCAST_UPSTREAM_TOKEN={UPSTREAM_TOKEN}",
            "--",
            sys.executable,
            str(OUT / "server.py"),
        ]
    )
    print(f"  4. Try it in Claude Code (app running):\n     {claude_line}")
    if plan.profile is not SafetyProfile.FULL:
        print("  5. Run the wizard again with profile full to see refund_ticket approval-gated.")


if __name__ == "__main__":
    main()
