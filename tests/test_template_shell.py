"""Tests for opt-in shell command interpolation in TemplateEngine."""

from __future__ import annotations

import pytest

from promptise.prompts.template import (
    ShellExecutionError,
    SubprocessShellExecutor,
    TemplateEngine,
    render_template,
)


class TestShellInterpolationDisabledByDefault:
    def test_syntax_left_alone_when_no_executor(self):
        """Without a shell_executor, !`cmd` is just literal text."""
        out = render_template("Today is !`date`", {})
        assert out == "Today is !`date`"  # untouched

    def test_render_template_helper_default(self):
        out = render_template("hello !`whoami`", {})
        assert "!`whoami`" in out


# These tests check what a command returns, not how fast: a loaded Windows
# runner can take seconds just to start cmd.exe.
_GENEROUS = 30.0


class TestShellInterpolationOptIn:
    def test_explicit_executor_runs_command(self):
        engine = TemplateEngine(
            shell_executor=lambda cmd: f"<{cmd}>",
        )
        out = engine.render("Today is !`date`. Branch: !`git branch`.", {})
        assert out == "Today is <date>. Branch: <git branch>."

    def test_subprocess_executor_basic_echo(self):
        engine = TemplateEngine(shell_executor=SubprocessShellExecutor(timeout=_GENEROUS))
        out = engine.render("hello !`echo world`", {})
        assert out == "hello world"

    def test_subprocess_executor_multiple_commands(self):
        engine = TemplateEngine(shell_executor=SubprocessShellExecutor(timeout=_GENEROUS))
        out = engine.render("a=!`echo 1`, b=!`echo 2`", {})
        assert out == "a=1, b=2"

    def test_subprocess_executor_failure_raises(self):
        engine = TemplateEngine(shell_executor=SubprocessShellExecutor(timeout=_GENEROUS))
        with pytest.raises(ShellExecutionError, match="exited"):
            engine.render("oops: !`false`", {})

    def test_subprocess_executor_timeout(self):
        engine = TemplateEngine(shell_executor=SubprocessShellExecutor(timeout=0.1))
        with pytest.raises(ShellExecutionError, match="timed out"):
            engine.render("!`sleep 5`", {})

    def test_subprocess_executor_allowlist_blocks_others(self):
        executor = SubprocessShellExecutor(timeout=2.0, allowlist={"echo"})
        engine = TemplateEngine(shell_executor=executor)

        # Allowed command works
        assert engine.render("!`echo ok`", {}) == "ok"

        # Disallowed command is blocked
        with pytest.raises(ShellExecutionError, match="not in shell executor allowlist"):
            engine.render("!`whoami`", {})


class TestShellInterpolationOrderOfOperations:
    def test_runs_after_includes(self):
        # Include resolves first, then shell command from inside the include
        engine = TemplateEngine(
            includes={"banner": "[!`echo BANNER`]"},
            shell_executor=lambda cmd: f"X{cmd}X",
        )
        out = engine.render('{% include "banner" %}', {})
        assert out == "[Xecho BANNERX]"

    def test_does_not_break_variable_interpolation(self):
        engine = TemplateEngine(shell_executor=lambda cmd: "RESULT")
        out = engine.render("user={name}, time=!`date`", {"name": "alice"})
        assert out == "user=alice, time=RESULT"

    def test_only_bang_backtick_is_intercepted(self):
        # A bare backtick (e.g. inline markdown code) should NOT be touched.
        engine = TemplateEngine(shell_executor=lambda cmd: "EXEC")
        out = engine.render("Use the `git status` command, not !`git status`.", {})
        # Backtick code stays, bang-backtick is replaced
        assert out == "Use the `git status` command, not EXEC."
