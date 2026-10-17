"""Run a Python file as a complete program, the way ``python file.py`` does."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

__all__ = ["exit_code", "run_script"]


def exit_code(exc: SystemExit) -> int:
    """The process exit code for *exc*, as the interpreter computes it."""
    code = exc.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


def run_script(path: Path, args: Sequence[str]) -> int:
    """Run *path* as ``__main__`` with ``sys.argv = [path, *args]``.

    ``sys.path[0]`` is the script's directory, as with ``python path``.
    ``SystemExit`` becomes the return value; any other exception propagates.
    """
    argv, path0 = list(sys.argv), list(sys.path)
    sys.argv = [str(path), *args]
    sys.path.insert(0, str(path.parent))
    try:
        runpy.run_path(str(path), run_name="__main__")
    except SystemExit as exc:
        return exit_code(exc)
    finally:
        sys.argv = argv
        sys.path[:] = path0
    return 0
