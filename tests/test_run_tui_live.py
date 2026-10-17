"""The terminal UI against a real program: ``RunTUI`` over a ``ChildRunSource``.

The child is a real ``python -m promptise run`` of a small project whose agent
uses the scripted test model (no network).  This checks the whole path the
user gets with ``promptise run --tui``: the child's run events over the local
channel, its output through the pipes, and commands back.
"""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from promptise.run.supervisor import ChildRunSource
from promptise.run.tui.app import RunTUI

TESTS = Path(__file__).parent
SRC = TESTS.parent / "src"

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="child process + pty timing")


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "bot"
    _write(root / "promptise.toml", 'name = "live-bot"\nagent = "agent.py:create_agent"\n')
    _write(
        root / "agent.py",
        """
        from _scripted_model import Scripted
        from langchain_core.tools import tool
        from promptise import build_agent

        @tool
        def get_account(account_id: str) -> str:
            \"\"\"Look up an account by id.\"\"\"
            return f"account {account_id}: active"

        async def create_agent(ctx=None):
            return await build_agent(
                model=Scripted(mode="tool:get_account", args={"account_id": "42"}),
                extra_tools=[get_account],
            )
        """,
    )
    _write(
        root / "main.py",
        """
        import sys
        print("hello from main", sys.argv[1:])
        name = input("Name: ")
        print(f"bye {name}")
        """,
    )
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join([str(SRC), str(TESTS), os.environ.get("PYTHONPATH", "")])
    )
    monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")
    monkeypatch.delenv("PROMPTISE_RUN_CHILD", raising=False)
    return root


async def _until(app: RunTUI, pilot: Any, check: Callable[[], bool], timeout: float = 40) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.flush_events()
        await pilot.pause()
        if check():
            return
        await asyncio.sleep(0.1)
    raise AssertionError("timed out waiting for the UI")


async def test_chat_with_a_real_agent(project: Path) -> None:
    source = ChildRunSource(["agent.py"], cwd=project)
    await source.start()
    app = RunTUI(source)
    async with app.run_test(size=(140, 45)) as pilot:
        await _until(app, pilot, lambda: bool(app.state.agents) and app.state.state == "idle")
        assert "agent" in app.state.agents
        app.query_one("#prompt").value = "look up 42"  # type: ignore[attr-defined]
        await pilot.press("enter")
        await _until(app, pilot, lambda: any(getattr(t, "done", False) for t in app.state.chat))
        turn = next(t for t in app.state.chat if getattr(t, "done", False))
        assert turn.output == "ANSWER: account 42: active"
        await _until(app, pilot, lambda: len(app.state.tool_calls) >= 1)
        await _until(app, pilot, lambda: len(app.state.timeline) >= 6)
        await pilot.press("escape", "3")
        await pilot.pause()
    await source.close()


async def test_console_and_restart_with_a_real_script(project: Path) -> None:
    source = ChildRunSource(["main.py", "x"], cwd=project)
    await source.start()
    app = RunTUI(source)
    async with app.run_test(size=(140, 45)) as pilot:

        def texts() -> list[str]:
            return [str(getattr(line, "text", "")) for line in app.state.console]

        await _until(app, pilot, lambda: "Name: " in texts())
        assert "hello from main ['x']" in texts()
        app.query_one("#prompt").value = "Ada"  # type: ignore[attr-defined]
        await pilot.press("enter")
        await _until(app, pilot, lambda: "bye Ada" in texts())
        await _until(app, pilot, lambda: app.state.state == "exited")
        assert app.state.last_exit is not None and app.state.last_exit.get("code") == 0

        # r restarts the program: a second run starts and prints again.
        await pilot.press("escape", "r")
        await _until(app, pilot, lambda: texts().count("Name: ") >= 2)
        assert len(app.state.runs) >= 2
    await source.close()
