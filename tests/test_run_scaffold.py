"""``promptise init --project``: the files :func:`scaffold_project` writes, and that they work.

The generated ``agent.py`` and ``main.py`` run here for real, with a scripted
chat model standing in for the provider (no network, no API key).
"""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from _scripted_model import Scripted

import promptise.agent as agent_module
from promptise.run import RunContext, load_manifest
from promptise.run.scaffold import provider_env_vars, scaffold_project

PROJECT_FILES = ["promptise.toml", "agent.py", "main.py", ".env.example", "README.md"]


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve every model string to the scripted model.

    ``fake:answer`` answers ``"ok"``; anything else (the generated project's
    real model string included) calls the ``add`` tool with 2 and 3 and then
    answers ``"ANSWER: <result>"``.
    """

    def normalize(model: Any) -> Any:
        if model == "fake:answer":
            return Scripted(mode="answer")
        return Scripted(mode="tool:add", args={"a": 2, "b": 3})

    monkeypatch.setattr(agent_module, "_normalize_model", normalize)


def _import(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ctx(root: Path, model: str | None) -> RunContext:
    return RunContext(project="demo", root=root, mode="once", model=model)


# ---------------------------------------------------------------------------
# Files and contents
# ---------------------------------------------------------------------------


def test_creates_every_file_and_returns_them(tmp_path: Path) -> None:
    project = tmp_path / "demo-bot"
    written = scaffold_project(project)
    assert [p.name for p in written] == [*PROJECT_FILES, ".gitignore"]
    assert all(p.is_file() and p.parent == project for p in written)


def test_manifest_loads_and_names_main_agent_and_scripts(tmp_path: Path) -> None:
    scaffold_project(tmp_path / "demo-bot")
    manifest = load_manifest(tmp_path / "demo-bot" / "promptise.toml")
    assert manifest.name == "demo-bot"
    assert manifest.version == "0.1.0"
    assert manifest.description
    assert manifest.main == "main.py"
    assert manifest.agent == "agent.py:create_agent"
    assert manifest.resolve_main() == (tmp_path / "demo-bot" / "main.py").resolve()
    assert set(manifest.scripts) == {"dev", "ui", "start", "serve", "check"}
    assert manifest.scripts["dev"] == "promptise run --devtools --watch"
    assert manifest.scripts["ui"] == "promptise run --tui"
    assert manifest.scripts["start"] == "promptise run --prod"
    assert "--serve" in manifest.scripts["serve"]
    assert "--check" in manifest.scripts["check"]
    assert manifest.env.files == [".env"]
    assert manifest.env.required == ["OPENAI_API_KEY"]
    assert manifest.pricing == {}  # the [pricing] example is commented out


def test_name_and_model_options(tmp_path: Path) -> None:
    scaffold_project(tmp_path, name='odd "name"', model="anthropic:claude-sonnet-4-5")
    manifest = load_manifest(tmp_path / "promptise.toml")
    assert manifest.name == 'odd "name"'
    assert manifest.env.required == ["ANTHROPIC_API_KEY"]
    assert 'MODEL = "anthropic:claude-sonnet-4-5"' in (tmp_path / "agent.py").read_text()
    env_example = (tmp_path / ".env.example").read_text()
    assert "ANTHROPIC_API_KEY=" in env_example and "OPENAI_API_KEY" not in env_example
    assert (tmp_path / "README.md").read_text().startswith('# odd "name"\n')


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("openai:gpt-5-mini", ["OPENAI_API_KEY"]),
        ("anthropic:claude-sonnet-4-5", ["ANTHROPIC_API_KEY"]),
        ("gemini:gemini-2.5-pro", ["GOOGLE_API_KEY"]),
        ("ollama:llama3.1", []),
        ("gpt-5-mini", ["OPENAI_API_KEY"]),
        ("not-a-provider:model", ["OPENAI_API_KEY"]),
    ],
)
def test_provider_env_vars(model: str, expected: list[str]) -> None:
    assert provider_env_vars(model) == expected


def test_keyless_model_requires_nothing(tmp_path: Path) -> None:
    scaffold_project(tmp_path, model="ollama:llama3.1")
    assert load_manifest(tmp_path / "promptise.toml").env.required == []
    assert "needs no API key" in (tmp_path / ".env.example").read_text()


def test_generated_python_compiles(tmp_path: Path) -> None:
    scaffold_project(tmp_path)
    for name in ("agent.py", "main.py"):
        path = tmp_path / name
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_main_is_a_script_and_readme_shows_the_commands(tmp_path: Path) -> None:
    scaffold_project(tmp_path)
    main = (tmp_path / "main.py").read_text()
    assert 'if __name__ == "__main__":' in main
    assert "asyncio.run(main())" in main
    assert "await agent.shutdown()" in main
    readme = (tmp_path / "README.md").read_text()
    for command in ("promptise run\n", "promptise run dev", "promptise run --tui", "--serve"):
        assert command in readme or command.strip() in readme


# ---------------------------------------------------------------------------
# Existing files
# ---------------------------------------------------------------------------


def test_refuses_to_overwrite_and_lists_every_conflict(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('mine')\n")
    (tmp_path / "README.md").write_text("# mine\n")
    with pytest.raises(FileExistsError) as exc:
        scaffold_project(tmp_path)
    message = str(exc.value)
    assert "main.py" in message and "README.md" in message
    assert "promptise.toml" not in message
    # Nothing was written.
    assert (tmp_path / "main.py").read_text() == "print('mine')\n"
    assert not (tmp_path / "promptise.toml").exists()
    assert not (tmp_path / ".gitignore").exists()


def test_force_overwrites(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('mine')\n")
    written = scaffold_project(tmp_path, force=True)
    assert tmp_path / "main.py" in written
    assert "create_agent" in (tmp_path / "main.py").read_text()


def test_scaffolding_twice_needs_force(tmp_path: Path) -> None:
    scaffold_project(tmp_path)
    with pytest.raises(FileExistsError):
        scaffold_project(tmp_path)
    written = scaffold_project(tmp_path, force=True)
    # .gitignore already has every line: it is not changed, so not returned.
    assert [p.name for p in written] == PROJECT_FILES


def test_refuses_a_file_as_directory(tmp_path: Path) -> None:
    target = tmp_path / "file"
    target.write_text("x")
    with pytest.raises(NotADirectoryError):
        scaffold_project(target)


def test_gitignore_is_merged_without_duplicates(tmp_path: Path) -> None:
    (tmp_path / ".gitignore").write_text("*.log\n.env\n/__pycache__")  # no trailing newline
    written = scaffold_project(tmp_path)
    assert tmp_path / ".gitignore" in written
    lines = (tmp_path / ".gitignore").read_text().splitlines()
    assert lines[:3] == ["*.log", ".env", "/__pycache__"]
    assert lines.count(".env") == 1
    assert ".promptise/" in lines
    assert "__pycache__/" not in lines  # "/__pycache__" already covers it


def test_gitignore_with_everything_is_left_alone(tmp_path: Path) -> None:
    original = ".env\n.promptise/\n__pycache__/\n"
    (tmp_path / ".gitignore").write_text(original)
    written = scaffold_project(tmp_path)
    assert tmp_path / ".gitignore" not in written
    assert (tmp_path / ".gitignore").read_text() == original


# ---------------------------------------------------------------------------
# The generated code works
# ---------------------------------------------------------------------------


async def test_generated_factory_builds_a_working_agent(tmp_path: Path, fake_models: None) -> None:
    scaffold_project(tmp_path)
    module = _import(tmp_path / "agent.py", "scaffolded_agent_factory")

    agent = await module.create_agent(_ctx(tmp_path, "fake:answer"))
    try:
        assert [t.name for t in agent._tools] == ["add"]
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "hi"}]})
        assert result["messages"][-1].content == "ok"
    finally:
        await agent.shutdown()


async def test_generated_factory_without_context_uses_its_model_and_tool(
    tmp_path: Path, fake_models: None
) -> None:
    scaffold_project(tmp_path)
    module = _import(tmp_path / "agent.py", "scaffolded_agent_no_ctx")

    for ctx in (None, _ctx(tmp_path, None)):
        agent = await module.create_agent(ctx)
        try:
            result = await agent.ainvoke({"messages": [{"role": "user", "content": "2 + 3?"}]})
            assert result["messages"][-1].content == "ANSWER: 5.0"
        finally:
            await agent.shutdown()


def test_generated_main_runs_as_a_plain_script(
    tmp_path: Path,
    fake_models: None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``python main.py "question"``: imports agent.py, answers, prints, exits."""
    scaffold_project(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(sys, "argv", [str(tmp_path / "main.py"), "What", "is", "2 + 3?"])
    monkeypatch.delitem(sys.modules, "agent", raising=False)
    try:
        runpy.run_path(str(tmp_path / "main.py"), run_name="__main__")
    finally:
        sys.modules.pop("agent", None)
    assert capsys.readouterr().out.strip() == "ANSWER: 5.0"
