"""The subprocess helpers keep hook and template spawns ``posix_spawn``-eligible."""

from __future__ import annotations

import subprocess
import sys
from unittest.mock import patch

import pytest

from promptise._spawn import resolve_executable, spawn_options
from promptise.prompts.template import ShellExecutionError, SubprocessShellExecutor


def test_no_cwd_means_no_fd_closing_so_cpython_can_posix_spawn() -> None:
    assert spawn_options(None) == {"close_fds": False}
    assert spawn_options("/tmp") == {}


def test_bare_command_is_resolved_to_a_path() -> None:
    resolved = resolve_executable(["echo", "hi"])
    assert resolved[0].endswith("echo") and "/" in resolved[0] or "\\" in resolved[0]
    assert resolved[1:] == ["hi"]
    assert resolve_executable([sys.executable, "-c", "pass"]) == [sys.executable, "-c", "pass"]
    with pytest.raises(FileNotFoundError, match="command not found"):
        resolve_executable(["definitely-not-a-command-4f9k"])


@pytest.mark.skipif(sys.platform == "win32", reason="posix_spawn is a POSIX detail")
def test_exec_form_takes_the_posix_spawn_path() -> None:
    """The exact conditions CPython checks before choosing ``posix_spawn``."""
    seen: dict[str, object] = {}
    real = subprocess.Popen._execute_child

    def spy(self, args, executable, preexec_fn, close_fds, pass_fds, cwd, *rest):
        seen.update(args=list(args), close_fds=close_fds, cwd=cwd, preexec_fn=preexec_fn)
        return real(self, args, executable, preexec_fn, close_fds, pass_fds, cwd, *rest)

    with patch.object(subprocess.Popen, "_execute_child", spy):
        assert SubprocessShellExecutor(shell=False)("echo posix") == "posix"
    assert seen["cwd"] is None and seen["preexec_fn"] is None and seen["close_fds"] is False
    assert "/" in str(seen["args"][0])  # a path: posix_spawn needs a directory component


def test_exec_form_reports_unknown_commands_as_shell_errors() -> None:
    with pytest.raises(ShellExecutionError, match="failed to start"):
        SubprocessShellExecutor(shell=False)("definitely-not-a-command-4f9k --flag")
    with pytest.raises(ShellExecutionError, match="failed to start"):
        SubprocessShellExecutor(shell=False)("echo 'unbalanced")
