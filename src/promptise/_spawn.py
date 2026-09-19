"""Spawn helpers shared by the shell hook and the template shell executor.

CPython spawns subprocesses through ``posix_spawn`` only under specific
conditions; otherwise it ``fork()``s. A forked child of a multithreaded process
that has used macOS system frameworks (any ``httpx`` client does, through the
proxy lookup) can die with SIGSEGV before ``exec`` — the process reports
``exited -11``. These helpers make the common case (no working directory
override) eligible for ``posix_spawn`` on every platform.
"""

from __future__ import annotations

import errno
import shutil
from typing import Any

__all__ = ["resolve_executable", "spawn_options"]


def spawn_options(cwd: str | None) -> dict[str, Any]:
    """Keyword arguments that let CPython spawn without ``fork()``.

    CPython uses ``posix_spawn`` only when no file descriptors have to be
    closed after the fork and no working directory is set; otherwise it forks.
    A forked child of a multithreaded process that has used macOS frameworks
    (any ``httpx`` client does, through the system proxy lookup) can die with
    SIGSEGV before ``exec`` — ``exited -11``. ``close_fds=False`` is safe on
    Python 3.4+: descriptors are non-inheritable unless a caller made one
    inheritable on purpose, so nothing of the parent's leaks into the hook.
    """
    return {} if cwd is not None else {"close_fds": False}


def resolve_executable(args: list[str]) -> list[str]:
    """*args* with ``args[0]`` resolved to an absolute path via ``PATH``.

    ``posix_spawn`` is only chosen for an executable given with a directory
    component; a bare name would silently fall back to ``fork()``. An
    unknown command raises the same ``FileNotFoundError`` the spawn would.
    """
    head = args[0]
    if "/" in head or "\\" in head:
        return args
    found = shutil.which(head)
    if found is None:
        raise FileNotFoundError(errno.ENOENT, f"command not found: {head}")
    return [found, *args[1:]]
