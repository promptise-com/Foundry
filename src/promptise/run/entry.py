"""What ``promptise run`` runs: a script, or an agent factory.

A *script* is a normal Python program, run like ``python main.py``.  A
*factory* is a function that returns an agent (``agent.py:create_agent``);
``promptise run`` builds the agent and gives it a chat, a one-shot question
or an HTTP API.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from .context import RunContext

__all__ = ["DEFAULT_FACTORY", "Entry", "EntryError", "detect_kind", "load_factory", "parse_ref"]

DEFAULT_FACTORY = "create_agent"
"""The factory name looked for when a file names none."""


class EntryError(ValueError):
    """The entry cannot be found or loaded (exit code 2)."""


@dataclass(frozen=True)
class Entry:
    """A resolved entry.

    Attributes:
        raw: What the user wrote (``main.py``, ``agent.py:create_agent``, ``pkg.mod:make``).
        path: The Python file, when the entry is a file.
        module: The dotted module, when the entry is ``pkg.module:attr``.
        attr: The factory's name, or ``None`` for a script.
    """

    raw: str
    path: Path | None
    module: str | None
    attr: str | None

    @property
    def kind(self) -> Literal["script", "factory"]:
        return "factory" if self.attr else "script"

    def describe(self, root: Path | None = None) -> str:
        """``file.py`` / ``file.py:factory`` / ``pkg.mod:factory`` (relative to *root* or the cwd)."""
        if self.path is not None:
            where = str(self.path)
            for base in (root, Path.cwd()):
                if base is None:
                    continue
                try:
                    where = str(self.path.relative_to(base.resolve()))
                    break
                except ValueError:
                    continue
        else:
            where = self.module or self.raw
        return f"{where}:{self.attr}" if self.attr else where


def parse_ref(ref: str, root: Path) -> Entry:
    """Parse *ref* (``file.py``, ``file.py:factory``, ``pkg.module:factory``) relative to *root*.

    Raises:
        EntryError: The file does not exist, or *ref* is not one of the forms.
    """
    head, sep, attr = ref.rpartition(":")
    if sep and attr.isidentifier() and (head.endswith(".py") or _is_dotted(head)):
        if head.endswith(".py"):
            path = _file(head, root)
            return Entry(raw=ref, path=path, module=None, attr=attr)
        return Entry(raw=ref, path=None, module=head, attr=attr)
    if ref.endswith(".py") or Path(ref).suffix == ".py":
        return Entry(raw=ref, path=_file(ref, root), module=None, attr=None)
    raise EntryError(
        f"{ref!r} is not a script name, a .py file, file.py:factory or package.module:factory"
    )


def _is_dotted(text: str) -> bool:
    return bool(text) and all(part.isidentifier() for part in text.split("."))


def _file(text: str, root: Path) -> Path:
    path = Path(text)
    if not path.is_absolute():
        # A path the user typed is relative to where they are; one from the
        # manifest is relative to the root.  Try the cwd first, then the root.
        cwd_path = Path.cwd() / path
        path = cwd_path if cwd_path.exists() else root / path
    if not path.is_file():
        raise EntryError(f"{text}: no such file")
    return path.resolve()


def _is_main_guard(node: ast.stmt) -> bool:
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
        sides = [test.left, *test.comparators]
        names = [s for s in sides if isinstance(s, ast.Name) and s.id == "__name__"]
        consts = [s for s in sides if isinstance(s, ast.Constant) and s.value == "__main__"]
        return bool(names and consts)
    return False


def detect_kind(path: Path) -> tuple[Literal["script", "factory"], str | None]:
    """Whether the file at *path* is a script or a module with an agent factory.

    A file with an ``if __name__ == "__main__":`` block is a script.  Otherwise
    a file that defines :data:`DEFAULT_FACTORY` (``create_agent``) at the top
    level is a factory module.  Anything else runs as a script.

    Returns:
        ``("script", None)`` or ``("factory", "create_agent")``.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return "script", None  # running it reports the real error
    if any(_is_main_guard(node) for node in tree.body):
        return "script", None
    has_factory = any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == DEFAULT_FACTORY
        for node in tree.body
    )
    return ("factory", DEFAULT_FACTORY) if has_factory else ("script", None)


def _module_name_for(path: Path, root: Path) -> str:
    """A module name for *path* that matches how the project imports it."""
    try:
        rel = path.relative_to(root)
        parts = list(rel.with_suffix("").parts)
        if parts and all(p.isidentifier() for p in parts):
            return ".".join(parts)
    except ValueError:
        pass
    return path.stem if path.stem.isidentifier() else "promptise_run_entry"


def _import_entry(entry: Entry, root: Path) -> Any:
    if entry.module is not None:
        try:
            return importlib.import_module(entry.module)
        except ModuleNotFoundError as exc:
            if exc.name and entry.module.startswith(exc.name):
                raise EntryError(f"cannot import {entry.module!r}: {exc}") from exc
            raise
    assert entry.path is not None
    name = _module_name_for(entry.path, root)
    existing = sys.modules.get(name)
    if existing is not None and getattr(existing, "__file__", None) == str(entry.path):
        return existing
    spec = importlib.util.spec_from_file_location(name, entry.path)
    if spec is None or spec.loader is None:
        raise EntryError(f"cannot import {entry.path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _looks_like_agent(value: Any) -> bool:
    return hasattr(value, "ainvoke") and hasattr(value, "shutdown")


async def load_factory(entry: Entry, ctx: RunContext) -> Any:
    """Import *entry*'s module, call its factory and return the agent.

    The factory takes no argument or a :class:`RunContext`, and may be async.

    Raises:
        EntryError: The factory is missing, takes the wrong arguments, or
            does not return an agent.
    """
    module = _import_entry(entry, ctx.root)
    attr = entry.attr or DEFAULT_FACTORY
    factory = getattr(module, attr, None)
    if factory is None:
        raise EntryError(f"{entry.describe()}: no {attr!r} in the module")
    if _looks_like_agent(factory):
        raise EntryError(
            f"{entry.describe()} is an agent built at import time; make it a function "
            f"that builds and returns the agent (async def {attr}(ctx): ...), so "
            "promptise run can build it inside its event loop"
        )
    if not callable(factory):
        raise EntryError(f"{entry.describe()} is not callable")
    try:
        params = [
            p
            for p in inspect.signature(factory).parameters.values()
            if p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        all_params = list(inspect.signature(factory).parameters.values())
    except (TypeError, ValueError):
        params, all_params = [], []
    if len(params) > 1:
        raise EntryError(f"{entry.describe()} must take no argument or one (ctx: RunContext)")
    result = factory(ctx) if params or all_params else factory()
    if inspect.isawaitable(result):
        result = await result
    if not _looks_like_agent(result):
        raise EntryError(
            f"{entry.describe()} returned {type(result).__name__}, not an agent "
            "(return the result of `await build_agent(...)`)"
        )
    return result
