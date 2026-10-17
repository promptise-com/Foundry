"""Build the `promptise run` screenshots in docs/assets/run/ (SVG).

Runs the real `examples/run/support_bot` project, in a temporary copy whose
model is a deterministic demo model (no network, no API key, no cost), and
captures:

- the terminal UI (`promptise run --tui`): chat, trace, tools, metrics,
  approvals, agents and run tabs, from a real child process;
- the dev tools trace (`promptise run --devtools`) as it prints in a terminal.

Usage (from the repository root)::

    python scripts/build_run_screenshots.py
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "examples" / "run" / "support_bot"
OUT = ROOT / "docs" / "assets" / "run"
SIZE = (132, 38)

DEMO_MODEL = '''
"""A deterministic stand-in for gpt-5-mini, for screenshots: no network."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


class DemoModel(BaseChatModel):
    model_name: str = "gpt-5-mini"
    tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "demo"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "DemoModel":
        names = [getattr(t, "name", None) or t.get("name") for t in tools]
        return self.model_copy(update={"tool_names": names})

    def _generate(self, messages: list[BaseMessage], stop: Any = None, **kw: Any) -> ChatResult:
        raise NotImplementedError

    def _plan(self, question: str, results: list[ToolMessage]) -> AIMessage:
        done = {m.name for m in results}
        order = re.search(r"\\b([A-Z]\\d{4})\\b", question)
        email = re.search(r"[\\w.]+@[\\w-]+(?:\\.[\\w-]+)+", question)
        wants_refund = any(w in question.lower() for w in ("refund", "dented", "damaged"))
        if order and "lookup_order" not in done:
            return self._call("lookup_order", {"order_id": order.group(1)})
        if wants_refund and "open_ticket" not in done:
            return self._call(
                "open_ticket",
                {
                    "customer_email": email.group(0) if email else "unknown",
                    "summary": f"Damaged item, refund requested ({order.group(1) if order else '?'})",
                    "priority": "high",
                },
            )
        if not order and "search_help_center" not in done:
            return self._call("search_help_center", {"query": question})
        return AIMessage(content=self._answer(question, results))

    def _call(self, name: str, args: dict[str, Any]) -> AIMessage:
        return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}])

    def _answer(self, question: str, results: list[ToolMessage]) -> str:
        by_name = {m.name: str(m.content) for m in results}
        if "open_ticket" in by_name:
            ticket = re.search(r"T-\\d{4}", by_name["open_ticket"])
            return (
                "I'm sorry your bottle arrived dented. I've passed this to our team as "
                f"{ticket.group(0) if ticket else 'a ticket'}, marked high priority; a person "
                "will reply within one business day and refund you without a return.\\n\\n"
                "Trailhead Support"
            )
        if "lookup_order" in by_name:
            order = json.loads(by_name["lookup_order"])
            if order.get("status") == "in transit":
                return (
                    f"Your {order['item'].lower()} ({order['order_id']}) are on the way with "
                    f"{order['carrier']} and should arrive by {order['expected_on']}. The tracking "
                    "link is in your shipping confirmation email.\\n\\nTrailhead Support"
                )
            return (
                f"Your {order['item'].lower()} ({order['order_id']}) was delivered on "
                f"{order.get('delivered_on', 'its delivery date')}.\\n\\nTrailhead Support"
            )
        return (
            "You can return unworn items within 30 days of delivery. Start the return at "
            "trailhead.example.com/returns; the label is free.\\n\\nTrailhead Support"
        )

    async def _agenerate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        question = next(
            (str(m.content) for m in reversed(messages) if isinstance(m, HumanMessage)), ""
        )
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        results = [m for m in messages[last_human:] if isinstance(m, ToolMessage)]
        await asyncio.sleep(0.35 + 0.1 * len(results))
        message = self._plan(question, results)
        prompt = sum(_tokens(str(m.content)) for m in messages) + 640
        completion = _tokens(str(message.content)) + 12 * len(message.tool_calls)
        message.usage_metadata = {
            "input_tokens": prompt,
            "output_tokens": completion,
            "total_tokens": prompt + completion,
        }
        return ChatResult(
            generations=[ChatGeneration(message=message)],
            llm_output={
                "model_name": self.model_name,
                "token_usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                },
            },
        )
'''


def make_project(tmp: Path) -> Path:
    """A copy of the example whose agent uses the demo model and asks before tickets."""
    project = tmp / "support-bot"
    shutil.copytree(EXAMPLE, project, ignore=shutil.ignore_patterns("__pycache__", ".promptise"))
    (project / "demo_model.py").write_text(DEMO_MODEL, encoding="utf-8")
    agent = (project / "agent.py").read_text(encoding="utf-8")
    agent = agent.replace(
        "        model=model,\n", "        model=__import__('demo_model').DemoModel(),\n"
    )
    agent = agent.replace(
        "        extra_tools=[lookup_order, search_help_center, open_ticket],\n",
        "        extra_tools=[lookup_order, search_help_center, open_ticket],\n"
        "        approval=__import__('promptise.approval', fromlist=['x']).ApprovalPolicy(\n"
        "            tools=['open_ticket'],\n"
        "            handler=__import__('promptise.approval', fromlist=['x']).QueueApprovalHandler(),\n"
        "        ),\n",
    )
    assert "DemoModel" in agent and "ApprovalPolicy" in agent, "the example's agent.py changed"
    (project / "agent.py").write_text(agent, encoding="utf-8")
    manifest = (project / "promptise.toml").read_text(encoding="utf-8")
    assert "[pricing]" in manifest, "the example should price gpt-5-mini for the cost column"
    return project


def child_env(project: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(project), env.get("PYTHONPATH", "")]
    )
    env["PROMPTISE_NO_DOTENV"] = "1"
    env["OPENAI_API_KEY"] = "sk-demo"
    env.pop("PROMPTISE_RUN_CHILD", None)
    return env


async def tui_shots(project: Path) -> None:
    from promptise.run.supervisor import ChildRunSource
    from promptise.run.tui.app import RunTUI

    os.environ.update(child_env(project))
    source = ChildRunSource(["--mode", "chat"], cwd=project)
    await source.start()
    app = RunTUI(source, title="support-bot")

    async def until(check, timeout: float = 60) -> None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            app.flush_events()
            await pilot.pause()
            if check():
                return
            await asyncio.sleep(0.1)
        raise SystemExit("timed out")

    def done_turns() -> int:
        return sum(1 for t in app.state.chat if getattr(t, "done", False))

    async def ask(text: str) -> None:
        before = done_turns()
        app.query_one("#prompt").value = text
        await pilot.press("enter")
        await until(lambda: done_turns() > before)

    async def shot(name: str) -> None:
        await asyncio.sleep(0.6)  # one stats tick
        app.flush_events()
        await pilot.pause()
        app.save_screenshot(str(OUT / f"{name}.svg"))
        print("wrote", name)

    async with app.run_test(size=SIZE) as pilot:
        await until(lambda: bool(app.state.agents) and app.state.state == "idle")
        await ask("Hi, where is my order A1002? It was supposed to arrive by now.")
        await ask("How do I return a jacket that doesn't fit?")
        await shot("tui-chat")

        # The third ticket needs a ticket for a person: open_ticket waits for approval.
        app.query_one(
            "#prompt"
        ).value = "My order A1003 arrived dented. I'd like a refund please. I'm lee@example.com."
        await pilot.press("enter")
        await until(lambda: any(getattr(a, "pending", False) for a in app.state.approvals))
        await pilot.press("escape", "6")
        await shot("tui-approvals")
        await pilot.press("a")
        before = 2
        await until(lambda: done_turns() > before)

        await pilot.press("escape", "2")
        await shot("tui-trace")
        await pilot.press("3")
        await shot("tui-tools")
        await pilot.press("4")
        await shot("tui-metrics")
        await pilot.press("7")
        await shot("tui-agents")
        await pilot.press("0")
        await shot("tui-run")
        await pilot.press("1")
        await shot("tui-chat-approved")
    await source.close()


def devtools_shot(project: Path) -> None:
    from rich.console import Console
    from rich.text import Text

    # The devtools run has no approval handler waiting: use the plain example agent.
    plain = project.parent / "support-bot-dev"
    if plain.exists():
        shutil.rmtree(plain)
    shutil.copytree(project, plain)
    agent = (plain / "agent.py").read_text(encoding="utf-8")
    agent = re.sub(r"        approval=.*?\n        \),\n", "", agent, flags=re.S)
    (plain / "agent.py").write_text(agent, encoding="utf-8")
    env = child_env(plain)
    env.update(
        {"FORCE_COLOR": "1", "COLUMNS": "118", "TERM": "xterm-256color", "PYTHONUNBUFFERED": "1"}
    )
    env.pop("NO_COLOR", None)
    # One pipe for both streams, unbuffered: the order a terminal shows.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "promptise",
            "run",
            "--devtools",
            "main.py",
            "Hi, where is my order A1002? It was supposed to arrive by now.",
        ],
        cwd=plain,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    if proc.returncode != 0:
        raise SystemExit(proc.stdout)
    with open(os.devnull, "w") as sink:
        console = Console(record=True, width=118, file=sink, force_terminal=True)
        console.print(
            Text(
                '$ promptise run --devtools main.py "Hi, where is my order A1002? …"', style="bold"
            )
        )
        console.print(Text.from_ansi(proc.stdout.rstrip()))
    svg = console.export_svg(title="promptise run --devtools")
    (OUT / "devtools.svg").write_text(svg, encoding="utf-8")
    print("wrote devtools")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "src"))
    # A short, readable path for the Run tab and the report line.
    base = (
        Path("/tmp/promptise-demo")
        if os.name == "posix"
        else Path(tempfile.gettempdir()) / "promptise-demo"
    )
    shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True)
    try:
        project = make_project(base)
        devtools_shot(project)
        asyncio.run(tui_shots(project))
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
