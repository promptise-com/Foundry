"""The project manifest: ``promptise.toml`` or ``[tool.promptise]`` in ``pyproject.toml``.

Like ``package.json`` for a Node project, the manifest says which file is the
program (``main``), which named commands the project has (``[scripts]``), which
``.env`` files and variables it needs (``[env]``), and how the dev, prod and
serve modes behave.  It describes how to *run* a project, not the agent: the
agent stays in your Python code.

Example::

    name = "support-bot"
    main = "main.py"

    [scripts]
    dev = "promptise run --devtools --watch"

    [env]
    required = ["OPENAI_API_KEY"]
"""

from __future__ import annotations

import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

__all__ = [
    "MANIFEST_FILE",
    "DevSettings",
    "EnvSettings",
    "ManifestError",
    "Price",
    "ProdSettings",
    "ProjectManifest",
    "ServeSettings",
    "find_manifest",
    "load_manifest",
]

MANIFEST_FILE = "promptise.toml"
"""The manifest's file name."""

DEFAULT_MAINS: tuple[str, ...] = ("main.py", "agent.py", "app.py")
"""Files tried, in order, when the manifest names no ``main``."""

ObserveLevelName = Literal["off", "basic", "standard", "full"]


class ManifestError(ValueError):
    """The manifest cannot be found, read or validated.

    Every problem is listed in :attr:`problems` (one line each); ``str()``
    joins them under the manifest's path.
    """

    def __init__(self, path: Path | None, problems: list[str]) -> None:
        self.path = path
        self.problems = problems
        where = f"{path}: " if path is not None else ""
        if len(problems) == 1:
            super().__init__(f"{where}{problems[0]}")
        else:
            super().__init__(where + "\n  - " + "\n  - ".join(problems))


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EnvSettings(_Section):
    """``[env]``: the variables the project needs."""

    files: list[str] = Field(default_factory=lambda: [".env"])
    """``.env``-style files loaded in order (relative to the project root).
    A variable already set is never overwritten; a missing file is skipped."""

    required: list[str] = Field(default_factory=list)
    """Variables that must be set before the program starts.  Every missing
    one is reported at once."""

    vars: dict[str, str] = Field(default_factory=dict)
    """Default values, set only when the variable is not set already."""


class DevSettings(_Section):
    """``[dev]``: applies with ``--devtools``, ``--tui`` and ``--watch``."""

    observe: Literal["basic", "standard", "full"] = "full"
    """How much the dev tools record for agents that set no ``observe`` themselves."""

    watch: list[str] = Field(default_factory=lambda: ["."])
    """Paths ``--watch`` watches, relative to the project root."""

    ignore: list[str] = Field(default_factory=list)
    """Extra glob patterns ``--watch`` ignores.  ``.git``, ``__pycache__``,
    ``.venv``, ``venv``, ``node_modules`` and ``.promptise`` are always ignored."""

    report: str = ""
    """Write a JSON run report to this path on exit (empty: no report)."""


class ProdSettings(_Section):
    """``[prod]``: applies with ``--prod``."""

    log_format: Literal["json", "text"] = "json"
    """``json``: one JSON object per log line on stderr, for log pipelines."""

    log_level: str = "INFO"
    """Root log level."""

    observe: ObserveLevelName = "off"
    """Record agent activity at this level for agents that set no ``observe``
    themselves; each timeline event becomes a JSON log line."""

    shutdown_timeout: float = 30.0
    """Seconds in-flight work may take to finish after SIGTERM."""

    @field_validator("log_level")
    @classmethod
    def _level(cls, value: str) -> str:
        upper = value.upper()
        if upper not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("must be DEBUG, INFO, WARNING, ERROR or CRITICAL")
        return upper


class ServeSettings(_Section):
    """``[serve]``: the HTTP API of ``promptise run --serve``."""

    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=0, le=65535)
    auth_token_env: str | None = "PROMPTISE_SERVE_TOKEN"
    """Environment variable holding the bearer token clients must send.  A
    token is required when ``host`` is not a loopback address."""

    max_concurrency: int = Field(default=8, ge=1)
    """Requests answered at once; more get ``429``."""


class Price(_Section):
    """Price of a model in USD per one million tokens."""

    input: float = Field(ge=0)
    output: float = Field(ge=0)


class ProjectManifest(_Section):
    """A parsed ``promptise.toml`` (or ``[tool.promptise]``)."""

    name: str | None = None
    version: str | None = None
    description: str | None = None
    main: str | None = None
    """The program ``promptise run`` runs, relative to the root.  Defaults to the
    first existing of ``main.py``, ``agent.py``, ``app.py``."""

    agent: str | None = None
    """An agent factory (``agent.py:create_agent`` or ``pkg.module:create_agent``)
    for the chat, once and serve modes."""

    python_path: list[str] = Field(default_factory=lambda: ["."])
    """Directories added to ``sys.path``, relative to the root."""

    scripts: dict[str, str] = Field(default_factory=dict)
    """Named commands: ``promptise run <name> [args]`` runs one, like ``npm run``."""

    env: EnvSettings = Field(default_factory=EnvSettings)
    dev: DevSettings = Field(default_factory=DevSettings)
    prod: ProdSettings = Field(default_factory=ProdSettings)
    serve: ServeSettings = Field(default_factory=ServeSettings)
    pricing: dict[str, Price] = Field(default_factory=dict)

    # Set by the loader, not read from the file.
    path: Path | None = Field(default=None, exclude=True)
    """The file the manifest was read from (``None`` for the built-in default)."""

    root: Path = Field(default_factory=Path.cwd, exclude=True)
    """The project root: the manifest's directory."""

    @field_validator("scripts")
    @classmethod
    def _script_names(cls, value: dict[str, str]) -> dict[str, str]:
        for name, cmd in value.items():
            if not name or name.startswith("-") or any(c.isspace() for c in name):
                raise ValueError(f"invalid script name {name!r}")
            if not cmd.strip():
                raise ValueError(f"script {name!r} is empty")
        return value

    @property
    def project_name(self) -> str:
        """``name``, or the root directory's name."""
        return self.name or self.root.name

    def resolve_main(self) -> Path | None:
        """The absolute path of the main program, or ``None`` when there is none.

        A ``main`` of the form ``file.py:factory`` resolves to ``file.py``.
        """
        if self.main:
            file_part = self.main.split(":", 1)[0] if _is_file_ref(self.main) else self.main
            return (self.root / file_part).resolve()
        for candidate in DEFAULT_MAINS:
            path = self.root / candidate
            if path.is_file():
                return path.resolve()
        return None

    def to_public_dict(self) -> dict[str, Any]:
        """The manifest as a dict for display, with nothing secret in it.

        The manifest itself holds no secrets (tokens are read from the
        environment), so this is the validated content plus the path and root.
        """
        data = self.model_dump(mode="json")
        data["path"] = str(self.path) if self.path else None
        data["root"] = str(self.root)
        return data


def _is_file_ref(ref: str) -> bool:
    """Whether *ref* is ``something.py:attr`` (a file with a factory)."""
    head, sep, attr = ref.rpartition(":")
    # "C:\\x\\main.py" has a colon but no attribute after a .py file.
    return bool(sep) and head.endswith(".py") and attr.isidentifier()


def _load_toml(path: Path) -> dict[str, Any]:
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - exercised on 3.10 only
        import tomli as tomllib
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise ManifestError(path, [f"invalid TOML: {exc}"]) from exc
    except OSError as exc:
        raise ManifestError(path, [f"cannot read: {exc}"]) from exc


def _pyproject_table(path: Path) -> dict[str, Any] | None:
    """The ``[tool.promptise]`` table of *path*, or ``None`` when it has none."""
    data = _load_toml(path)
    tool = data.get("tool")
    if not isinstance(tool, dict):
        return None
    table = tool.get("promptise")
    return table if isinstance(table, dict) else None


def find_manifest(start: Path | None = None) -> Path | None:
    """The nearest manifest from *start* (default: the cwd) upwards, or ``None``.

    In each directory ``promptise.toml`` wins over a ``pyproject.toml`` with a
    ``[tool.promptise]`` table.  The search stops at the first directory holding
    ``.git`` (the repository root), so a manifest above the repository is never
    used.
    """
    here = (start or Path.cwd()).absolute()
    for directory in (here, *here.parents):
        candidate = directory / MANIFEST_FILE
        if candidate.is_file():
            _check_owner(candidate)
            return candidate
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file():
            try:
                ours = _pyproject_table(pyproject) is not None
            except ManifestError:
                ours = False  # a broken pyproject.toml that isn't ours is not our problem
            if ours:
                _check_owner(pyproject)
                return pyproject
        if (directory / ".git").exists():
            break
    return None


def _check_owner(path: Path) -> None:
    """Refuse a manifest someone else could have written.

    The manifest decides what runs, with which environment.  As for ``.env``
    files, one found above the working directory (``/tmp/promptise.toml`` on a
    shared machine) must be a regular file owned by you that others cannot
    write.

    Raises:
        ManifestError: The file is someone else's or anyone can write it.
    """
    from ..models import _unsafe_dotenv

    try:
        problem = _unsafe_dotenv(path, path.stat())
    except OSError as exc:
        raise ManifestError(path, [f"cannot read: {exc}"]) from exc
    if problem is not None:
        raise ManifestError(path, [f"refusing to use this manifest: {problem}"])


def _format_errors(exc: ValidationError) -> list[str]:
    problems = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        msg = err["msg"]
        if err["type"] == "extra_forbidden":
            msg = "unknown key (check the spelling)"
        problems.append(f"{loc}: {msg}" if loc else msg)
    return problems


def load_manifest(path: Path) -> ProjectManifest:
    """Read and validate the manifest at *path* (``promptise.toml`` or ``pyproject.toml``).

    Raises:
        ManifestError: The file cannot be read or parsed, or a value is invalid.
    """
    path = path.absolute()
    if path.name == "pyproject.toml":
        table = _pyproject_table(path)
        if table is None:
            raise ManifestError(path, ["has no [tool.promptise] table"])
        prefix = "tool.promptise."
    else:
        table = _load_toml(path)
        prefix = ""
    try:
        manifest = ProjectManifest.model_validate(table)
    except ValidationError as exc:
        raise ManifestError(path, [prefix + p for p in _format_errors(exc)]) from exc
    manifest.path = path
    manifest.root = path.parent
    return manifest


def default_manifest(root: Path) -> ProjectManifest:
    """The manifest used when a project has none: every default, rooted at *root*."""
    manifest = ProjectManifest()
    manifest.root = root.absolute()
    return manifest


def ignored_dir_names() -> Iterable[str]:
    """Directory names ``--watch`` always ignores."""
    return (
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        ".promptise",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    )
