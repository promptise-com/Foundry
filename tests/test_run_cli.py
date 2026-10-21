"""`promptise run` end to end: projects, scripts, modes, dev tools, prod, watch and the UI channel.

Each test builds a small project in a temporary directory whose agents use the
scripted test model (no network), and runs the real CLI in a subprocess.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from promptise import _harness, cli

TESTS = Path(__file__).parent
SRC = TESTS.parent / "src"


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "bot"
    _write(
        root / "promptise.toml",
        """
        name = "demo-bot"
        version = "0.1.0"
        main = "main.py"
        agent = "agent.py:create_agent"

        [scripts]
        hello = "python -c \\"import sys; print('hello', sys.argv[1:])\\""
        checkit = "promptise run --check"

        [env]
        required = ["DEMO_TOKEN"]
        vars = { DEMO_TOKEN = "from-manifest" }
        """,
    )
    _write(
        root / "agent.py",
        """
        from _scripted_model import Scripted
        from langchain_core.tools import tool
        from promptise import build_agent

        @tool
        def get_account(account_id: str) -> str:
            \"\"\"Look up an account by id.\"\"\"
            return f"account {account_id}: active"

        async def create_agent(ctx=None):
            return await build_agent(
                model=Scripted(mode="tool:get_account", args={"account_id": "42"}),
                extra_tools=[get_account],
            )
        """,
    )
    _write(
        root / "main.py",
        """
        import asyncio
        import os
        import sys

        from agent import create_agent

        async def main():
            agent = await create_agent()
            try:
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": " ".join(sys.argv[1:]) or "hi"}]}
                )
                print("RESULT", result["messages"][-1].content, os.environ["DEMO_TOKEN"])
            finally:
                await agent.shutdown()

        if __name__ == "__main__":
            asyncio.run(main())
            sys.exit(int(os.environ.get("EXIT_WITH", "0")))
        """,
    )
    return root


def _env(**extra: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SRC), str(TESTS), env.get("PYTHONPATH", "")])
    env["PROMPTISE_NO_DOTENV"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["COLUMNS"] = "200"
    env.pop("PROMPTISE_RUN_CHILD", None)
    env.pop("PROMPTISE_RUN_IPC", None)
    env.update(extra)
    return env


def _run(
    cwd: Path, *args: str, input: str | None = None, timeout: float = 60, **env: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "promptise", "run", *args],
        cwd=cwd,
        env=_env(**env),
        input=input,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# scripts and plain runs
# ---------------------------------------------------------------------------


def test_main_runs_as_a_normal_program(project: Path) -> None:
    proc = _run(project, "main.py", "look", "up", EXIT_WITH="3")
    assert proc.returncode == 3, proc.stderr
    assert "RESULT ANSWER: account 42: active from-manifest" in proc.stdout
    assert proc.stderr.strip() == ""  # a plain run adds nothing


def test_no_target_runs_main_from_a_subdirectory(project: Path) -> None:
    sub = project / "sub"
    sub.mkdir()
    proc = _run(sub)
    assert proc.returncode == 0, proc.stderr
    assert "RESULT ANSWER" in proc.stdout


def test_named_script_gets_extra_args(project: Path) -> None:
    proc = _run(project, "hello", "a", "b c")
    assert proc.returncode == 0, proc.stderr
    assert "hello ['a', 'b c']" in proc.stdout
    assert "> demo-bot hello:" in proc.stderr


def test_promptise_script_gets_our_flags(project: Path) -> None:
    proc = _run(project, "checkit")
    assert proc.returncode == 0, proc.stderr
    assert "is ready" in proc.stderr


def test_unknown_target_suggests_a_script(project: Path) -> None:
    proc = _run(project, "helo")
    assert proc.returncode == 2
    assert "did you mean 'hello'" in proc.stderr


def test_list_and_check(project: Path) -> None:
    listed = _run(project, "--list")
    assert listed.returncode == 0 and "hello" in listed.stdout and "checkit" in listed.stdout
    checked = _run(project, "--check")
    assert checked.returncode == 0, checked.stderr
    assert "main.py (script) is ready" in checked.stderr
    factory = _run(project, "--check", "--serve")
    assert factory.returncode == 0 and "agent.py:create_agent (serve)" in factory.stderr


def test_preflight_lists_every_problem(project: Path) -> None:
    (project / "promptise.toml").write_text(
        'name = "x"\nmain = "nope.py"\n[env]\nrequired = ["RUN_A_MISSING", "RUN_B_MISSING"]\n'
    )
    proc = _run(project)
    assert proc.returncode == 2
    assert "nope.py" in proc.stderr and "RUN_A_MISSING, RUN_B_MISSING" in proc.stderr
    prod = _run(project, "--prod")
    assert prod.returncode == 2
    line = json.loads(prod.stderr.strip().splitlines()[-1])
    assert line["msg"] == "preflight failed" and len(line["problems"]) == 2


def test_manifest_errors_exit_2(project: Path) -> None:
    (project / "promptise.toml").write_text('mian = "main.py"\n')
    proc = _run(project)
    assert proc.returncode == 2 and "mian: unknown key" in proc.stderr


def test_without_a_manifest(tmp_path: Path) -> None:
    _write(tmp_path / ".git" / "HEAD", "ref\n")  # stop the manifest search here
    _write(tmp_path / "solo.py", "import sys\nprint('solo', sys.argv[1:])\n")
    proc = _run(tmp_path, "solo.py", "x")
    assert proc.returncode == 0 and "solo ['x']" in proc.stdout
    nothing = _run(tmp_path)
    assert nothing.returncode == 2 and "nothing to run" in nothing.stderr


# ---------------------------------------------------------------------------
# dev tools, once, chat, prod
# ---------------------------------------------------------------------------


def test_devtools_traces_and_summarises(project: Path) -> None:
    proc = _run(project, "--devtools", "--report", "out/report.json", "main.py")
    assert proc.returncode == 0, proc.stderr
    err = proc.stderr
    assert "● agent" in err and "⚙ get_account" in err and "✓ get_account" in err
    assert "◆ llm" in err and "promptise run · exit 0" in err
    assert "RESULT ANSWER" in proc.stdout
    report = json.loads((project / "out" / "report.json").read_text())
    assert report["stats"]["totals"]["tool_calls"] == 1
    kinds = {e["kind"] for e in report["events"]}
    assert {"run.start", "agent.built", "timeline", "stats", "run.exit"} <= kinds


def test_once_json(project: Path) -> None:
    proc = _run(project, "-i", "hello", "--json")
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    assert data["output"] == "ANSWER: account 42: active" and data["error"] is None
    assert data["usage"]["tool_calls"] == 1 and data["usage"]["llm_calls"] == 2


def test_once_reads_stdin(project: Path) -> None:
    proc = _run(project, "-i", "-", input="from stdin\n")
    assert proc.returncode == 0 and proc.stdout.strip() == "ANSWER: account 42: active"


def test_chat_repl(project: Path) -> None:
    proc = _run(project, "agent.py", input="hello\n/tools\n/stats\n/bogus\n/exit\n")
    assert proc.returncode == 0, proc.stderr
    assert "agent › ANSWER: account 42: active" in proc.stdout
    assert "get_account" in proc.stderr and "1 tool call(s)" in proc.stderr
    assert "unknown command /bogus" in proc.stderr


def test_model_override_reaches_every_agent(project: Path) -> None:
    _write(
        project / "override.py",
        """
        import asyncio
        from promptise import build_agent

        async def main():
            agent = await build_agent(model="openai:does-not-matter")
            print("MODEL", agent.model_name)
            await agent.shutdown()

        asyncio.run(main())
        """,
    )
    # A provider id the scripted test helpers don't know is still just a string here;
    # check the override is applied by reading the agent's model name.
    proc = _run(project, "--model", "openai:gpt-override", "override.py", OPENAI_API_KEY="sk-test")
    assert "MODEL openai:gpt-override" in proc.stdout, proc.stderr


def test_prod_writes_json_lines(project: Path) -> None:
    proc = _run(project, "--prod", "--observe", "basic", "main.py")
    assert proc.returncode == 0, proc.stderr
    lines = [json.loads(line) for line in proc.stderr.strip().splitlines()]
    msgs = [line["msg"] for line in lines]
    assert msgs[0] == "run.start" and "agent.built" in msgs and "timeline" in msgs
    assert msgs[-1] == "stats" and "run.exit" in msgs
    assert "RESULT ANSWER" in proc.stdout


@pytest.mark.skipif(os.name != "posix", reason="SIGTERM semantics")
def test_prod_sigterm_shuts_down_gracefully(project: Path) -> None:
    _write(
        project / "slow.py",
        """
        import asyncio
        import sys

        async def main():
            print("READY", flush=True)
            try:
                await asyncio.sleep(60)
            finally:
                print("CLEANUP", flush=True)

        asyncio.run(main())
        """,
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "promptise", "run", "--prod", "slow.py"],
        cwd=project,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    assert proc.stdout.readline().strip() == "READY"
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=30)
    assert "CLEANUP" in out
    assert proc.returncode == 143, err
    exits = [json.loads(line) for line in err.splitlines() if '"run.exit"' in line]
    assert exits and exits[0]["code"] == 143


def test_flag_conflicts(project: Path) -> None:
    assert _run(project, "--tui", "--prod").returncode == 2
    assert _run(project, "--watch", "--prod").returncode == 2
    assert _run(project, "--mode", "bogus").returncode == 2


# ---------------------------------------------------------------------------
# --watch and the terminal UI's channel
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="process groups")
def test_watch_restarts_on_change(project: Path) -> None:
    script = _write(project / "tick.py", "print('VERSION 1', flush=True)\n")
    proc = subprocess.Popen(
        [sys.executable, "-m", "promptise", "run", "--watch", "tick.py"],
        cwd=project,
        env=_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stdout is not None
    try:
        seen: list[str] = []
        deadline = time.time() + 30
        while time.time() < deadline:
            line = proc.stdout.readline()
            seen.append(line)
            if "waiting for file changes" in line:
                break
        assert any("VERSION 1" in s for s in seen), seen
        time.sleep(0.5)
        script.write_text("print('VERSION 2', flush=True)\n")
        while time.time() < deadline:
            line = proc.stdout.readline()
            seen.append(line)
            if "VERSION 2" in line:
                break
        assert any("VERSION 2" in s for s in seen), seen
        assert any("restarting" in s for s in seen)
    finally:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


async def _collect(
    project: Path,
    args: list[str],
    commands: list[dict[str, Any]],
    until: str,
    reply: Any = None,
) -> list[dict[str, Any]]:
    from promptise.run.supervisor import ChildRunSource

    old = dict(os.environ)
    os.environ.update(_env())
    try:
        source = ChildRunSource(args, cwd=project)
        await source.start()
    finally:
        os.environ.clear()
        os.environ.update(old)
    seen: list[dict[str, Any]] = []

    async def read() -> None:
        async for event in source.events():
            seen.append(event)
            if event["kind"] == "state" and event.get("state") == "idle" and commands:
                await source.send(commands.pop(0))
            if reply is not None:
                answer = reply(event)
                if answer is not None:
                    await source.send(answer)
            if event["kind"] == until:
                return

    try:
        await asyncio.wait_for(read(), 60)
    finally:
        await source.close()
    return seen


async def test_child_source_script_mode(project: Path) -> None:
    events = await _collect(project, ["main.py", "hi"], [], "run.exit")
    kinds = [e["kind"] for e in events]
    assert kinds.count("run.start") == 1 and "agent.built" in kinds and "timeline" in kinds
    outputs = [e["text"] for e in events if e["kind"] == "output"]
    assert any("RESULT ANSWER" in line for line in outputs)
    run_start = next(e for e in events if e["kind"] == "run.start")
    assert run_start["flags"]["tui"] is True and run_start["mode"] == "script"
    assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)


async def test_child_source_chat_mode(project: Path) -> None:
    events = await _collect(
        project, ["agent.py"], [{"cmd": "chat", "text": "hello", "turn_id": "t1"}], "turn.end"
    )
    end = next(e for e in events if e["kind"] == "turn.end")
    assert end["turn_id"] == "t1" and end["output"] == "ANSWER: account 42: active"
    streamed = [e["event"]["type"] for e in events if e["kind"] == "stream"]
    assert "tool_start" in streamed and streamed[-1] == "done"


# ---------------------------------------------------------------------------
# the CLI command itself (in process)
# ---------------------------------------------------------------------------


def test_inline_chat_flags_need_model_id() -> None:
    result = CliRunner().invoke(cli.app, ["run", "--stdio", "name=x command=y"])
    assert result.exit_code == 2 and "--model-id" in result.output


def test_in_process_list(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)
    result = CliRunner().invoke(cli.app, ["run", "--list"])
    assert result.exit_code == 0
    assert _harness.current() is None


async def test_child_source_answers_approvals(project: Path) -> None:
    _write(
        project / "gated.py",
        """
        from _scripted_model import Scripted
        from langchain_core.tools import tool
        from promptise import build_agent
        from promptise.approval import ApprovalPolicy, QueueApprovalHandler

        @tool
        def get_account(account_id: str) -> str:
            \"\"\"Look up an account by id.\"\"\"
            return f"account {account_id}: active"

        async def create_agent(ctx=None):
            return await build_agent(
                model=Scripted(mode="tool:get_account", args={"account_id": "7"}),
                extra_tools=[get_account],
                approval=ApprovalPolicy(tools=["get_account"], handler=QueueApprovalHandler()),
            )
        """,
    )

    def approve(event: dict[str, Any]) -> dict[str, Any] | None:
        if event["kind"] == "approval.pending":
            return {
                "cmd": "approval",
                "request_id": event["request"]["request_id"],
                "approved": True,
                "reason": "looks fine",
            }
        return None

    events = await _collect(
        project,
        ["gated.py"],
        [{"cmd": "chat", "text": "check 7", "turn_id": "t1"}],
        "turn.end",
        reply=approve,
    )
    pending = next(e for e in events if e["kind"] == "approval.pending")
    assert pending["request"]["tool_name"] == "get_account" and pending["deadline"] > 0
    resolved = next(e for e in events if e["kind"] == "approval.resolved")
    assert resolved["outcome"] == "approved" and resolved["reason"] == "looks fine"
    end = next(e for e in events if e["kind"] == "turn.end")
    assert end["output"] == "ANSWER: account 7: active"


@pytest.mark.skipif(os.name != "posix", reason="SIGTERM semantics")
def test_serve_over_http(project: Path) -> None:
    import re
    import urllib.request

    proc = subprocess.Popen(
        [sys.executable, "-m", "promptise", "run", "--serve", "--port", "0"],
        cwd=project,
        env=_env(PROMPTISE_SERVE_TOKEN="s3cret"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.stderr is not None
    try:
        url = None
        deadline = time.time() + 30
        while time.time() < deadline and url is None:
            line = proc.stderr.readline()
            match = re.search(r"serving demo-bot on (http://\S+)", line)
            if match:
                url = match.group(1)
        assert url, "server did not start"
        with urllib.request.urlopen(url + "/healthz", timeout=10) as resp:
            assert json.loads(resp.read())["status"] == "ok"
        request = urllib.request.Request(
            url + "/v1/invoke",
            data=json.dumps({"input": "check 42"}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer s3cret"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            body = json.loads(resp.read())
        assert body["output"] == "ANSWER: account 42: active"
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    assert proc.returncode == 0


def _late_pipes_source(
    args: list[str], project: Path, delay: float, monkeypatch: pytest.MonkeyPatch
) -> Any:
    """A ``ChildRunSource`` whose pipes hand over what the child prints *delay* late.

    As on a busy machine, the child's run events (over its channel) then get
    to the parent before the lines it printed last (over the pipes).  The
    environment stays set for the test, so a restart's child gets it too.
    """
    from promptise.run.supervisor import ChildRunSource

    class LatePipes(ChildRunSource):
        async def _pump(self, stream: asyncio.StreamReader | None, name: str) -> None:
            if stream is not None and delay:
                loop = asyncio.get_running_loop()
                feed_data, feed_eof = stream.feed_data, stream.feed_eof
                # Hand chunks over strictly in arrival order. Timers that are due
                # at the same moment can fire in any order, and on Windows the
                # clock ticks every ~15 ms, so "hello" and the "\r\n" written
                # right after it would otherwise swap now and then.
                late: list[Any] = []

                def hand_over() -> None:
                    feed, *data = late.pop(0)
                    feed(*data)

                def later(feed: Any, *data: bytes) -> None:
                    late.append((feed, *data))
                    loop.call_later(delay, hand_over)

                monkeypatch.setattr(stream, "feed_data", lambda data: later(feed_data, data))
                monkeypatch.setattr(stream, "feed_eof", lambda: later(feed_eof))
            await super()._pump(stream, name)

    for key, value in _env().items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("PROMPTISE_RUN_CHILD", raising=False)
    monkeypatch.delenv("PROMPTISE_RUN_IPC", raising=False)
    return LatePipes(args, cwd=project)


@pytest.mark.parametrize("pipe_delay", [0.0, 0.5], ids=["pipes-on-time", "pipes-late"])
async def test_child_source_feeds_stdin_and_shows_prompts(
    project: Path, pipe_delay: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(project / "ask.py", "name = input('Name: ')\nprint(f'hello {name}')\n")
    source = _late_pipes_source(["ask.py"], project, pipe_delay, monkeypatch)
    await source.start()
    events: list[dict[str, Any]] = []

    async def read() -> None:
        async for event in source.events():
            events.append(event)
            if event["kind"] == "output" and event["text"] == "Name: ":
                await source.send({"cmd": "stdin", "text": "Ada"})
            if event["kind"] == "state" and event.get("state") == "exited":
                return

    try:
        await asyncio.wait_for(read(), 60)
    finally:
        await source.close()
    outputs = [(e["stream"], e["text"]) for e in events if e["kind"] == "output"]
    assert outputs == [("stdout", "Name: "), ("stdout", "hello Ada")]
    # The run ends after its last line, however late the pipes deliver it:
    # one run.exit (the child's own, with its duration), then one "exited".
    ends = [e["kind"] for e in events if e["kind"] in ("output", "run.exit", "state")]
    assert ends[-3:] == ["output", "run.exit", "state"]
    exits = [e for e in events if e["kind"] == "run.exit"]
    assert len(exits) == 1 and exits[0]["code"] == 0 and exits[0]["duration_s"] is not None
    assert [e for e in events if e.get("state") == "exited"] == [events[-1]]


@pytest.mark.skipif(os.name != "posix", reason="SIGTERM semantics")
async def test_child_source_restart_ends_the_old_run_after_its_output(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(
        project / "ask.py",
        """
        try:
            input("Name: ")
        except KeyboardInterrupt:
            print("stopped")
            raise
        """,
    )
    source = _late_pipes_source(["ask.py"], project, 0.5, monkeypatch)
    await source.start()
    events: list[dict[str, Any]] = []

    async def read() -> None:
        prompts = 0
        async for event in source.events():
            events.append(event)
            if event["kind"] == "output" and event["text"] == "Name: ":
                prompts += 1
                if prompts == 1:
                    await source.restart()
                else:
                    return

    try:
        await asyncio.wait_for(read(), 60)
    finally:
        await source.close()
    first = events[: next(i for i, e in enumerate(events) if e.get("state") == "starting" and i)]
    kinds = [e["kind"] for e in first if e["kind"] in ("output", "run.exit")]
    assert kinds == ["output", "output", "run.exit"]
    assert [e["text"] for e in first if e["kind"] == "output"] == ["Name: ", "stopped"]
    exit_event = next(e for e in first if e["kind"] == "run.exit")
    assert exit_event["code"] == 143 and exit_event["duration_s"] is not None
    assert [e["kind"] for e in events].count("run.exit") == 1


def test_script_echo_masks_env_values(project: Path) -> None:
    proc = _run(project, "checkit", "--env", "SECRET_THING=sk-live-123")
    assert proc.returncode == 0, proc.stderr
    assert "sk-live-123" not in proc.stderr and "SECRET_THING=***" in proc.stderr
