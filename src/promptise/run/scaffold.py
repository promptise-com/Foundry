"""Create a new ``promptise run`` project: what ``promptise init --project`` writes.

:func:`scaffold_project` writes a small, working project into a directory:

- ``promptise.toml``: the manifest (``main``, ``agent``, ``[scripts]``, ``[env]``).
- ``agent.py``: ``create_agent``, the agent factory, with one local tool.
- ``main.py``: a normal Python program that builds the agent, asks it one
  question and prints the answer.  ``python main.py`` runs it as well.
- ``.env.example``: the variables the model's provider needs.
- ``README.md``: how to run the project.
- ``.gitignore``: ``.env``, ``.promptise/`` and ``__pycache__/`` added to it.

It never touches the network and never overwrites a file unless asked to.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["DEFAULT_MODEL", "provider_env_vars", "scaffold_project"]

DEFAULT_MODEL = "openai:gpt-5-mini"
"""The model a new project uses when none is given."""

_GITIGNORE_LINES = (".env", ".promptise/", "__pycache__/")


def provider_env_vars(model: str) -> list[str]:
    """The environment variables the provider of *model* requires.

    ``"anthropic:claude-sonnet-4-5"`` → ``["ANTHROPIC_API_KEY"]``.  A provider
    that needs no key (a local Ollama) gives ``[]``.  A model string without a
    known provider prefix gives ``["OPENAI_API_KEY"]``.
    """
    from ..models import parse_model

    provider, _, _ = parse_model(model)
    if provider is None:
        return ["OPENAI_API_KEY"]
    return [var.name for var in provider.env if var.required]


def _toml_str(value: str) -> str:
    """*value* as a double-quoted string literal, valid in TOML and in Python.

    JSON string escapes are valid in both.
    """
    return json.dumps(value, ensure_ascii=False)


def _toml_list(values: list[str]) -> str:
    return "[" + ", ".join(_toml_str(v) for v in values) + "]"


def _bare_model(model: str) -> str:
    """``"openai:gpt-5-mini"`` → ``"gpt-5-mini"``."""
    return model.split(":", 1)[1] if ":" in model else model


def _manifest(name: str, model: str, required: list[str]) -> str:
    pattern = _toml_str(_bare_model(model) + "*")
    return f"""\
# How this project runs, like package.json for npm.
# Reference: https://docs.promptise.com/core/projects/

name = {_toml_str(name)}
version = "0.1.0"
description = "An agent built with Promptise"
main = "main.py"                  # the program `promptise run` runs
agent = "agent.py:create_agent"   # the agent factory for --serve, -i and chat mode

[scripts]                         # `promptise run <name> [args]`, like `npm run`
dev = "promptise run --devtools --watch"
ui = "promptise run --tui"
start = "promptise run --prod"
serve = "promptise run --serve"
check = "promptise run --prod --check"

[env]
files = [".env"]                  # loaded in order; never overrides a variable already set
required = {_toml_list(required)}

# Cost estimates in --devtools and --tui: USD per 1M tokens, keyed by model name
# (globs allowed). The numbers below are an example; use your provider's prices.
# [pricing]
# {pattern} = {{ input = 0.25, output = 2.0 }}
"""


def _agent_py(model: str) -> str:
    return f'''\
"""The project's agent.

``create_agent`` is the agent factory.  ``main.py`` calls it, and
``promptise run`` calls it for ``--serve``, ``-i`` and chat mode, passing a
``RunContext``.  Every agent it builds is traced by ``promptise run`` with no
changes here.
"""

from __future__ import annotations

from langchain_core.tools import tool
from promptise import PromptiseAgent, build_agent
from promptise.run import RunContext

MODEL = {_toml_str(model)}

INSTRUCTIONS = (
    "You are a helpful assistant. Answer briefly. "
    "Use the add tool for arithmetic instead of computing in your head."
)


@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


async def create_agent(ctx: RunContext | None = None) -> PromptiseAgent:
    """Build the agent.

    Args:
        ctx: Passed by ``promptise run`` in its agent modes; ``None`` when
            ``main.py`` (or your own code) calls this.  ``ctx.model`` is the
            ``--model`` override, if any.
    """
    model = (ctx.model if ctx is not None else None) or MODEL
    return await build_agent(
        model=model,
        instructions=INSTRUCTIONS,
        extra_tools=[add],
    )
'''


_MAIN_PY = '''\
"""The project's program: the manifest's ``main``.

A normal Python script.  ``promptise run`` runs it like ``python main.py``
and adds the project's environment, dev tools or production behavior.
``python main.py "your question"`` works too.
"""

from __future__ import annotations

import asyncio
import sys

from agent import create_agent

DEFAULT_QUESTION = "What is 1234 + 5678?"


async def main() -> None:
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    agent = await create_agent()
    messages = [{"role": "user", "content": question}]
    try:
        result = await agent.ainvoke({"messages": messages})
        print(result["messages"][-1].content)
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
'''


def _env_example(model: str, required: list[str]) -> str:
    from ..models import parse_model

    provider, _, _ = parse_model(model)
    where = {var.name: var.where for var in provider.env} if provider is not None else {}
    lines = [
        "# Copy this file to .env and fill in the values.",
        "# promptise run loads .env and never overrides a variable that is already set.",
        "",
    ]
    for name in required:
        if where.get(name):
            lines.append(f"# {where[name]}")
        lines.append(f"{name}=")
    if not required:
        lines.append(f"# {model} needs no API key.")
    lines += [
        "",
        "# Bearer token clients send to `promptise run --serve`",
        "# (required when it listens on a non-loopback host).",
        "# PROMPTISE_SERVE_TOKEN=",
    ]
    return "\n".join(lines) + "\n"


def _readme(name: str, model: str, required: list[str]) -> str:
    if required:
        keys = ", ".join(f"`{v}`" for v in required)
        setup = (
            f"Copy `.env.example` to `.env` and set {keys}:\n\n```bash\ncp .env.example .env\n```"
        )
    else:
        setup = f"`{model}` needs no API key."
    return f"""\
# {name}

An agent project built with [Promptise Foundry](https://docs.promptise.com).

| File | What it is |
|---|---|
| `promptise.toml` | How the project runs: `main`, the agent factory, scripts, env |
| `main.py` | The program. Builds the agent, asks one question, prints the answer |
| `agent.py` | `create_agent`, the agent factory, with one tool (`add`) |

## Setup

```bash
pip install promptise
```

{setup}

## Run

```bash
promptise run                          # run main.py, like `python main.py`
promptise run main.py "What is 2 + 3?" # pass arguments to main.py
promptise run --devtools               # live traces, tool calls, tokens and cost
promptise run dev                      # dev tools, restart on file changes
promptise run --tui                    # the full terminal UI (`promptise run ui`)
promptise run --serve                  # an HTTP API for the agent on 127.0.0.1:8080
promptise run --prod                   # JSON logs and preflight checks (`promptise run start`)
promptise run --list                   # list the scripts in promptise.toml
```

`python main.py` works too: `promptise run` adds the project's environment
and instruments every agent the program builds, with no code changes.

Ask the HTTP API a question:

```bash
curl -s localhost:8080/v1/invoke -H 'content-type: application/json' \\
  -d '{{"input": "What is 2 + 3?"}}'
```

See [Projects & promptise run](https://docs.promptise.com/core/projects/) for
the manifest reference and every flag.
"""


def _merge_gitignore(path: Path) -> bool:
    """Add the lines this project needs to *path*; return whether it changed."""
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    present = {line.strip() for line in existing.splitlines()}
    # "/.env" or ".env/" already cover the same thing for git's purposes.
    present |= {p.strip("/") for p in present}
    missing = [line for line in _GITIGNORE_LINES if line.strip("/") not in present]
    if not missing:
        return False
    text = existing
    if text and not text.endswith("\n"):
        text += "\n"
    if text:
        text += "\n# promptise run\n"
    text += "\n".join(missing) + "\n"
    path.write_text(text, encoding="utf-8")
    return True


def scaffold_project(
    directory: Path,
    *,
    name: str | None = None,
    model: str = DEFAULT_MODEL,
    force: bool = False,
) -> list[Path]:
    """Create a new ``promptise run`` project in *directory*.

    Writes ``promptise.toml``, ``agent.py``, ``main.py``, ``.env.example`` and
    ``README.md``, and adds ``.env``, ``.promptise/`` and ``__pycache__/`` to
    ``.gitignore`` (created if missing, never duplicating a line).  The
    directory is created if needed.  Nothing is fetched from the network.

    Args:
        directory: Where to create the project.
        name: The project name (default: the directory's name).
        model: The model string the agent uses; it also decides which API
            key ``[env] required`` lists.
        force: Overwrite files that already exist.

    Returns:
        The files created or changed, in the order written.

    Raises:
        FileExistsError: A file the project needs exists and *force* is
            false.  The message lists every such file; nothing is written.
        NotADirectoryError: *directory* exists and is not a directory.
    """
    directory = Path(directory)
    if directory.exists() and not directory.is_dir():
        raise NotADirectoryError(f"{directory} exists and is not a directory")
    project_name = name or directory.resolve().name or "agent-project"
    required = provider_env_vars(model)

    files: dict[str, str] = {
        "promptise.toml": _manifest(project_name, model, required),
        "agent.py": _agent_py(model),
        "main.py": _MAIN_PY,
        ".env.example": _env_example(model, required),
        "README.md": _readme(project_name, model, required),
    }

    if not force:
        existing = [directory / rel for rel in files if (directory / rel).exists()]
        if existing:
            listed = ", ".join(str(p) for p in existing)
            raise FileExistsError(f"refusing to overwrite existing files: {listed}")

    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for rel, content in files.items():
        path = directory / rel
        path.write_text(content, encoding="utf-8")
        written.append(path)
    gitignore = directory / ".gitignore"
    if _merge_gitignore(gitignore):
        written.append(gitignore)
    return written
