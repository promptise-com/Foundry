"""`promptise.toml` / `[tool.promptise]`: parsing, validation, discovery and defaults."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from promptise.run.manifest import (
    ManifestError,
    ProjectManifest,
    default_manifest,
    find_manifest,
    load_manifest,
)


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


def test_full_manifest(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "promptise.toml",
        """
        name = "support-bot"
        version = "0.1.0"
        main = "app.py"
        agent = "agent.py:create_agent"
        python_path = [".", "src"]

        [scripts]
        dev = "promptise run --devtools --watch"
        eval = "python evals.py"

        [env]
        files = [".env", ".env.local"]
        required = ["OPENAI_API_KEY"]
        vars = { LOG_LEVEL = "info" }

        [dev]
        observe = "standard"
        watch = ["src"]
        ignore = ["*.log"]
        report = "out/report.json"

        [prod]
        log_format = "text"
        log_level = "warning"
        observe = "basic"
        shutdown_timeout = 5

        [serve]
        host = "0.0.0.0"
        port = 9000
        auth_token_env = "MY_TOKEN"
        max_concurrency = 2

        [pricing]
        "gpt-5-mini*" = { input = 0.25, output = 2.0 }
        """,
    )
    m = load_manifest(path)
    assert m.root == tmp_path and m.path == path
    assert m.project_name == "support-bot"
    assert m.main == "app.py" and m.agent == "agent.py:create_agent"
    assert m.scripts["dev"].startswith("promptise run")
    assert m.env.files == [".env", ".env.local"] and m.env.vars == {"LOG_LEVEL": "info"}
    assert m.dev.observe == "standard" and m.dev.report == "out/report.json"
    assert m.prod.log_level == "WARNING" and m.prod.observe == "basic"
    assert m.serve.port == 9000 and m.serve.max_concurrency == 2
    assert m.pricing["gpt-5-mini*"].output == 2.0
    public = m.to_public_dict()
    assert public["root"] == str(tmp_path) and public["main"] == "app.py"


def test_defaults(tmp_path: Path) -> None:
    m = load_manifest(_write(tmp_path / "promptise.toml", 'name = "x"\n'))
    assert m.env.files == [".env"] and m.env.required == []
    assert m.dev.observe == "full" and m.dev.watch == ["."]
    assert m.prod.log_format == "json" and m.prod.observe == "off"
    assert m.serve.host == "127.0.0.1" and m.serve.auth_token_env == "PROMPTISE_SERVE_TOKEN"
    assert m.python_path == ["."]


def test_unknown_keys_are_errors_with_their_path(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "promptise.toml",
        """
        mian = "main.py"
        [dev]
        obsrve = "full"
        """,
    )
    with pytest.raises(ManifestError) as exc:
        load_manifest(path)
    assert "mian: unknown key" in str(exc.value)
    assert "dev.obsrve: unknown key" in str(exc.value)
    assert len(exc.value.problems) == 2


def test_invalid_values(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "promptise.toml",
        """
        [prod]
        log_level = "loud"
        [scripts]
        "bad name" = "x"
        """,
    )
    with pytest.raises(ManifestError) as exc:
        load_manifest(path)
    text = str(exc.value)
    assert "prod.log_level" in text and "scripts" in text


def test_invalid_toml(tmp_path: Path) -> None:
    path = _write(tmp_path / "promptise.toml", "name = \n")
    with pytest.raises(ManifestError, match="invalid TOML"):
        load_manifest(path)


def test_pyproject_table(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "pyproject.toml",
        """
        [project]
        name = "pkg"

        [tool.promptise]
        main = "run.py"
        [tool.promptise.scripts]
        dev = "promptise run -d"
        """,
    )
    m = load_manifest(path)
    assert m.main == "run.py" and m.scripts == {"dev": "promptise run -d"}
    bad = _write(tmp_path / "b" / "pyproject.toml", "[tool.promptise]\nmian = 1\n")
    with pytest.raises(ManifestError, match="tool.promptise.mian"):
        load_manifest(bad)


def test_find_manifest_walks_up_and_stops_at_git(tmp_path: Path) -> None:
    project = tmp_path / "proj"
    _write(project / "promptise.toml", 'name = "p"\n')
    deep = project / "a" / "b"
    deep.mkdir(parents=True)
    assert find_manifest(deep) == project / "promptise.toml"

    # promptise.toml wins over a pyproject.toml with [tool.promptise] in the same dir.
    _write(project / "pyproject.toml", "[tool.promptise]\nmain = 'x.py'\n")
    assert find_manifest(project) == project / "promptise.toml"

    # A repository root stops the search: the manifest above it is not used.
    repo = project / "repo"
    (repo / ".git").mkdir(parents=True)
    assert find_manifest(repo) is None

    # A pyproject.toml without our table is skipped.
    other = tmp_path / "other"
    _write(other / "pyproject.toml", "[project]\nname = 'o'\n")
    (other / ".git").mkdir()
    assert find_manifest(other) is None


def test_resolve_main(tmp_path: Path) -> None:
    m = default_manifest(tmp_path)
    assert m.resolve_main() is None
    (tmp_path / "app.py").write_text("print(1)\n")
    assert m.resolve_main() == (tmp_path / "app.py").resolve()
    (tmp_path / "agent.py").write_text("x = 1\n")
    assert m.resolve_main() == (tmp_path / "agent.py").resolve()
    (tmp_path / "main.py").write_text("x = 1\n")
    assert m.resolve_main() == (tmp_path / "main.py").resolve()

    named = ProjectManifest(main="agent.py:create_agent")
    named.root = tmp_path
    assert named.resolve_main() == (tmp_path / "agent.py").resolve()
