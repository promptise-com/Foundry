"""The ``promptise run`` example projects under ``examples/run/`` are valid and work.

Each project's manifest must load, and its Python files must compile.  The
support bot's factory and program also run here with a scripted chat model in
place of the provider (no network, no API key), so a renamed tool or a broken
``main.py`` fails here.
"""

from __future__ import annotations

import importlib.util
import json
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from _scripted_model import Scripted

import promptise.agent as agent_module
from promptise.run import RunContext, load_manifest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "run"
SUPPORT_BOT = EXAMPLES / "support_bot"
PROJECTS = sorted(p.parent for p in EXAMPLES.glob("*/promptise.toml"))


def _import(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_there_are_example_projects() -> None:
    assert SUPPORT_BOT in PROJECTS


@pytest.mark.parametrize("project", PROJECTS, ids=lambda p: p.name)
def test_manifest_loads_and_its_files_exist(project: Path) -> None:
    manifest = load_manifest(project / "promptise.toml")
    main = manifest.resolve_main()
    assert main is not None and main.is_file()
    if manifest.agent:
        file_part, _, factory = manifest.agent.partition(":")
        assert (project / file_part).is_file()
        assert f"def {factory}(" in (project / file_part).read_text(encoding="utf-8")
    assert (project / "README.md").is_file()


@pytest.mark.parametrize("project", PROJECTS, ids=lambda p: p.name)
def test_python_files_compile(project: Path) -> None:
    files = sorted(project.rglob("*.py"))
    assert files
    for path in files:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_support_bot_manifest() -> None:
    manifest = load_manifest(SUPPORT_BOT / "promptise.toml")
    assert manifest.name == "support-bot"
    assert manifest.main == "main.py"
    assert manifest.agent == "agent.py:create_agent"
    assert {"dev", "ui", "start", "serve", "check"} <= set(manifest.scripts)
    assert manifest.env.required == ["OPENAI_API_KEY"]
    assert manifest.env.vars == {"SUPPORT_BOT_MODEL": "openai:gpt-5-mini"}
    assert manifest.pricing["gpt-5-mini*"].output == 2.0


# ---------------------------------------------------------------------------
# The support bot runs (scripted model)
# ---------------------------------------------------------------------------


@pytest.fixture
def support_bot(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """The support bot's ``agent.py``, with every model resolved to the scripted one.

    ``fake:lookup`` calls ``lookup_order("A1002")`` and answers with its
    result; every other model string answers ``"ok"``.
    """

    def normalize(model: Any) -> Any:
        if model == "fake:lookup":
            return Scripted(mode="tool:lookup_order", args={"order_id": "a1002"})
        return Scripted(mode="answer")

    monkeypatch.setattr(agent_module, "_normalize_model", normalize)
    monkeypatch.delenv("SUPPORT_BOT_MODEL", raising=False)
    return _import(SUPPORT_BOT / "agent.py", "support_bot_agent")


def test_support_bot_tools_are_local(support_bot: ModuleType) -> None:
    order = json.loads(support_bot.lookup_order.invoke({"order_id": " a1002 "}))
    assert order["order_id"] == "A1002" and order["status"] == "in transit"
    assert "No order" in support_bot.lookup_order.invoke({"order_id": "Z9"})
    assert "30 days" in support_bot.search_help_center.invoke({"query": "how do returns work"})
    assert "No article" in support_bot.search_help_center.invoke({"query": "xyzzy"})
    before = len(support_bot.TICKETS)
    reply = support_bot.open_ticket.invoke(
        {"customer_email": "lee@example.com", "summary": "dented bottle", "priority": "high"}
    )
    assert reply.startswith("Opened T-") and len(support_bot.TICKETS) == before + 1


async def test_support_bot_factory_uses_the_override_model(support_bot: ModuleType) -> None:
    ctx = RunContext(project="support-bot", root=SUPPORT_BOT, mode="serve", model="fake:lookup")
    agent = await support_bot.create_agent(ctx)
    try:
        assert sorted(t.name for t in agent._tools) == [
            "lookup_order",
            "open_ticket",
            "search_help_center",
        ]
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "A1002?"}]})
        assert "in transit" in result["messages"][-1].content
    finally:
        await agent.shutdown()


async def test_support_bot_model_from_env(
    support_bot: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPPORT_BOT_MODEL", "fake:lookup")
    agent = await support_bot.create_agent()
    try:
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "A1002?"}]})
        assert "in transit" in result["messages"][-1].content
    finally:
        await agent.shutdown()


def test_support_bot_main_runs_as_a_plain_script(
    support_bot: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``python main.py``: answers the three sample tickets and exits."""
    monkeypatch.syspath_prepend(str(SUPPORT_BOT))
    monkeypatch.setattr(sys, "argv", [str(SUPPORT_BOT / "main.py")])
    monkeypatch.delitem(sys.modules, "agent", raising=False)
    try:
        runpy.run_path(str(SUPPORT_BOT / "main.py"), run_name="__main__")
    finally:
        sys.modules.pop("agent", None)
    out = capsys.readouterr().out
    assert out.count("--- Ticket ") == 3
    assert out.count("\nok\n") == 3
