"""Unit tests for sandbox hardening — no Docker required.

Covers the secure network default, fail-closed ``network="restricted"``,
killing a timed-out command, exec-based ``write_file``, unknown-key
rejection, surfaced backend errors, package installs on the read-only
rootfs, path validation, and the ``build_agent`` wiring (no silent fallback,
``code_action=`` options, tracing of non-MCP tools, conversation context for
code-action).
"""

from __future__ import annotations

import base64
import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from pydantic import ValidationError

from promptise import build_agent
from promptise.engine.code_action import (
    CodeActionConfig,
    CodeActionNode,
    _extract_context,
)
from promptise.engine.state import GraphState
from promptise.sandbox import (
    CommandResult,
    DockerBackend,
    NetworkMode,
    SandboxConfig,
    SandboxManager,
    SandboxSession,
)
from promptise.sandbox.backends import EXEC_MARKER_ENV

# ---------------------------------------------------------------------------
# Config: secure defaults and unknown keys
# ---------------------------------------------------------------------------


class TestNetworkDefault:
    def test_default_is_none(self):
        assert SandboxConfig().network == NetworkMode.NONE

    def test_custom_config_without_network_stays_none(self):
        """A custom config that does not mention the network must not open it."""
        cfg = SandboxManager({"memory_limit": "512M"}).config
        assert cfg.network == NetworkMode.NONE
        backend = DockerBackend(cfg)
        assert backend._build_container_config()["network_mode"] == "none"

    def test_explicit_network_is_honoured(self):
        cfg = SandboxConfig(network="full")
        assert DockerBackend(cfg)._build_container_config()["network_mode"] == "bridge"


class TestUnknownKeys:
    def test_network_mode_is_rejected_with_hint(self):
        with pytest.raises(ValidationError, match="'network_mode': use 'network'"):
            SandboxConfig(**{"network_mode": "none"})

    def test_removed_tools_key_points_at_image(self):
        with pytest.raises(ValidationError, match="'tools'.*image"):
            SandboxConfig.from_dict({"tools": ["python", "node"]})

    def test_typo_gets_suggestion(self):
        with pytest.raises(ValidationError, match="did you mean 'memory_limit'"):
            SandboxConfig.from_dict({"memory_limt": "1G"})

    def test_manager_rejects_unknown_key(self):
        with pytest.raises(ValueError, match="network_mode"):
            SandboxManager({"network_mode": "none"})

    def test_superagent_section_rejects_unknown_key(self):
        from promptise.superagent_schema import SandboxConfigSection

        with pytest.raises(ValidationError):
            SandboxConfigSection(tools=["python"])
        section = SandboxConfigSection()
        assert section.network == "none"
        # Every key the section dumps is a valid SandboxConfig key.
        assert SandboxConfig(**section.model_dump()).network == NetworkMode.NONE

    @pytest.mark.asyncio
    async def test_build_agent_rejects_unknown_key_before_docker(self):
        with patch("promptise.sandbox.backends.DockerBackend.create_container") as create:
            with pytest.raises(ValidationError, match="network_mode"):
                await build_agent(
                    servers={},
                    model=FakeListChatModel(responses=["x"]),
                    agent_pattern="code-action",
                    sandbox={"network_mode": "none"},
                )
            create.assert_not_called()


# ---------------------------------------------------------------------------
# Container configuration
# ---------------------------------------------------------------------------


class TestContainerConfig:
    def test_limits_applied(self):
        cfg = SandboxConfig(disk_limit="256M", pids_limit=64, memory_limit="512M")
        conf = DockerBackend(cfg)._build_container_config()["host_config"]
        assert conf["pids_limit"] == 64
        assert conf["mem_limit"] == 512 * 1024**2
        assert conf["tmpfs"]["/workspace"] == f"rw,size={256 * 1024**2}"
        # /tmp and /var/tmp never exceed the disk limit
        assert conf["tmpfs"]["/tmp"].endswith(f"size={256 * 1024**2}")
        assert conf["read_only"] is True

    def test_workspace_tmpfs_even_with_writable_rootfs(self):
        cfg = SandboxConfig(read_only_rootfs=False, disk_limit="2G")
        conf = DockerBackend(cfg)._build_container_config()["host_config"]
        assert conf["read_only"] is False
        assert conf["tmpfs"]["/workspace"] == f"rw,size={2 * 1024**3}"

    def test_security_opts_and_caps(self):
        conf = DockerBackend(SandboxConfig())._build_container_config()
        assert conf["security_opt"] == ["no-new-privileges"]
        assert "CAP_SYS_ADMIN" in conf["host_config"]["cap_drop"]
        assert "CAP_SETUID" in conf["host_config"]["cap_drop"]

    def test_allow_sudo_restores_only_setuid_setgid(self):
        conf = DockerBackend(SandboxConfig(allow_sudo=True))._build_container_config()
        caps = conf["host_config"]["cap_drop"]
        assert "CAP_SETUID" not in caps and "CAP_SETGID" not in caps
        assert "CAP_SYS_ADMIN" in caps

    def test_home_points_at_workspace(self):
        backend = DockerBackend(SandboxConfig(env={"FOO": "1"}))
        env = backend._build_container_config()["environment"]
        assert env == {"HOME": "/workspace", "FOO": "1"}
        # An explicit HOME wins
        assert DockerBackend(SandboxConfig(env={"HOME": "/tmp"}))._environment()["HOME"] == "/tmp"

    def test_gvisor_sets_runtime(self):
        conf = DockerBackend(SandboxConfig(backend="gvisor"))._build_container_config()
        assert conf["host_config"]["runtime"] == "runsc"


# ---------------------------------------------------------------------------
# Fakes for the Docker SDK
# ---------------------------------------------------------------------------


def _exec_result(exit_code: int, stdout: bytes = b"", stderr: bytes = b"") -> Any:
    return SimpleNamespace(exit_code=exit_code, output=(stdout, stderr))


def _backend_with_container(container: Any, config: SandboxConfig | None = None) -> DockerBackend:
    backend = DockerBackend(config or SandboxConfig())
    client = MagicMock()
    client.containers.get.return_value = container
    client.containers.create.return_value = container
    backend._docker_client = client
    return backend


class TestRestrictedFailsClosed:
    @pytest.mark.asyncio
    async def test_missing_iptables_removes_container_and_raises(self):
        container = MagicMock(id="c1")
        container.exec_run.return_value = _exec_result(3, b"", b"iptables is not installed")
        backend = _backend_with_container(container, SandboxConfig(network="restricted"))

        with pytest.raises(RuntimeError, match="refused to start.*iptables is not installed"):
            await backend.create_container()
        container.remove.assert_called_once_with(force=True)
        # The filter runs as a privileged exec, not as sandboxed code.
        kwargs = container.exec_run.call_args.kwargs
        assert kwargs["privileged"] is True and kwargs["user"] == "root"

    @pytest.mark.asyncio
    async def test_filter_applied(self):
        container = MagicMock(id="c1")
        container.exec_run.return_value = _exec_result(0)
        backend = _backend_with_container(container, SandboxConfig(network="restricted"))
        assert await backend.create_container() == "c1"
        container.remove.assert_not_called()
        script = container.exec_run.call_args.kwargs["cmd"][2]
        assert "ip6tables" in script and "-P OUTPUT DROP" in script

    @pytest.mark.asyncio
    async def test_none_network_runs_no_filter(self):
        container = MagicMock(id="c1")
        backend = _backend_with_container(container)
        await backend.create_container()
        container.exec_run.assert_not_called()


class TestTimeoutKill:
    @pytest.mark.asyncio
    async def test_timed_out_exec_is_killed(self):
        released = threading.Event()
        calls: list[dict[str, Any]] = []

        def exec_run(**kwargs: Any) -> Any:
            calls.append(kwargs)
            if kwargs["cmd"][:2] == ["bash", "-c"]:
                # The user command: runs until the kill script fires.
                released.wait(5)
                return _exec_result(137, b"partial output\n", b"")
            released.set()
            return _exec_result(0)

        container = MagicMock()
        container.exec_run.side_effect = exec_run
        backend = _backend_with_container(container)

        result = await backend.execute_command("c1", "while true; do :; done", 1, "/workspace")

        assert result.timeout is True
        assert result.success is False
        assert "its processes were killed" in result.stderr
        assert result.stdout == "partial output\n"
        marker = calls[0]["environment"][EXEC_MARKER_ENV]
        kill = calls[1]
        assert kill["cmd"][-1] == marker
        assert kill["privileged"] is True

    @staticmethod
    def _unkillable_container() -> Any:
        """A container whose kill script fails (e.g. no pids left to fork)."""

        def exec_run(**kwargs: Any) -> Any:
            if kwargs["cmd"][:2] == ["bash", "-c"]:
                threading.Event().wait(1.5)
                return _exec_result(137)
            if "iptables" in kwargs["cmd"][2]:
                return _exec_result(0)
            return _exec_result(1, b"", b"still running: 42")

        container = MagicMock(id="c1")
        container.exec_run.side_effect = exec_run
        return container

    @pytest.mark.asyncio
    async def test_unkillable_exec_restarts_container(self):
        container = self._unkillable_container()
        backend = _backend_with_container(container)
        result = await backend.execute_command("c1", "sleep 100", 1, "/workspace")
        assert result.timeout is True
        assert "container was restarted" in result.stderr
        container.restart.assert_called_once()
        container.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_reapplies_restricted_filter(self):
        container = self._unkillable_container()
        backend = _backend_with_container(container, SandboxConfig(network="restricted"))
        result = await backend.execute_command("c1", "sleep 100", 1, "/workspace")
        assert "container was restarted" in result.stderr
        scripts = [c.kwargs["cmd"][2] for c in container.exec_run.call_args_list]
        assert any("ip6tables" in s for s in scripts)
        container.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_restart_stops_container_when_filter_fails(self):
        container = self._unkillable_container()
        original = container.exec_run.side_effect

        def exec_run(**kwargs: Any) -> Any:
            if kwargs["cmd"][:2] == ["sh", "-c"] and "iptables" in kwargs["cmd"][2]:
                return _exec_result(3, b"", b"iptables is not installed")
            return original(**kwargs)

        container.exec_run.side_effect = exec_run
        backend = _backend_with_container(container, SandboxConfig(network="restricted"))
        result = await backend.execute_command("c1", "sleep 100", 1, "/workspace")
        assert "could not be restarted" in result.stderr
        container.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_restart_is_reported(self):
        container = self._unkillable_container()
        container.restart.side_effect = RuntimeError("daemon gone")
        backend = _backend_with_container(container)
        result = await backend.execute_command("c1", "sleep 100", 1, "/workspace")
        assert result.timeout is True
        assert "could not be killed and the sandbox container could not be restarted" in (
            result.stderr
        )

    @pytest.mark.asyncio
    async def test_fast_command_is_not_killed(self):
        container = MagicMock()
        container.exec_run.return_value = _exec_result(0, b"ok\n")
        backend = _backend_with_container(container)
        result = await backend.execute_command("c1", "echo ok", 5, "/workspace")
        assert result.success and result.stdout == "ok\n"
        assert container.exec_run.call_count == 1


class TestWriteFile:
    @staticmethod
    def _recording_backend(fail_on: int | None = None) -> tuple[DockerBackend, list[str]]:
        backend = DockerBackend(SandboxConfig())
        commands: list[str] = []

        async def execute_command(container_id, command, timeout, workdir):
            commands.append(command)
            if fail_on is not None and len(commands) == fail_on:
                return CommandResult(1, "", "No space left on device")
            return CommandResult(0, "", "")

        backend.execute_command = execute_command  # type: ignore[method-assign]
        return backend, commands

    @staticmethod
    def _decode(commands: list[str]) -> bytes:
        import shlex

        data = b""
        for cmd in commands:
            if "base64 -d" not in cmd:
                continue
            printf = cmd[cmd.index("printf %s ") :]
            chunk = shlex.split(printf.split("|")[0])[2]
            data += base64.b64decode(chunk)
        return data

    @pytest.mark.asyncio
    async def test_small_file_is_one_exec_without_put_archive(self):
        backend, commands = self._recording_backend()
        content = "print('it''s $HOME `id`')\n"
        await backend.write_file("c1", "/workspace/sub/hostile.py", content)
        assert len(commands) == 1
        assert commands[0].startswith("mkdir -p -- /workspace/sub && ")
        assert "mv -f -- " in commands[0]
        assert self._decode(commands).decode() == content

    @pytest.mark.asyncio
    async def test_large_file_is_chunked_under_argv_limit(self):
        backend, commands = self._recording_backend()
        content = "x" * 300_000
        await backend.write_file("c1", "/workspace/big.txt", content)
        assert len(commands) > 1
        assert all(len(c) < 128 * 1024 for c in commands)
        assert self._decode(commands).decode() == content

    @pytest.mark.asyncio
    async def test_failure_cleans_up_and_raises(self):
        backend, commands = self._recording_backend(fail_on=1)
        with pytest.raises(RuntimeError, match="No space left on device"):
            await backend.write_file("c1", "/workspace/a.txt", "data")
        assert commands[-1].startswith("rm -f -- ")


class TestBackendErrors:
    @pytest.mark.asyncio
    async def test_gvisor_error_is_surfaced(self):
        manager = SandboxManager({"backend": "gvisor"})
        client = MagicMock()
        client.info.return_value = {"Runtimes": {"runc": {}, "io.containerd.runc.v2": {}}}
        manager.backend._docker_client = client  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="gVisor runtime 'runsc' is not registered") as exc:
            await manager.create_session()
        assert "Please ensure Docker is installed" not in str(exc.value)
        assert "available runtimes: io.containerd.runc.v2, runc" in str(exc.value)

    @pytest.mark.asyncio
    async def test_daemon_error_is_surfaced(self):
        manager = SandboxManager(True)
        client = MagicMock()
        client.ping.side_effect = ConnectionError("socket not found")
        manager.backend._docker_client = client  # type: ignore[attr-defined]
        with pytest.raises(RuntimeError, match="Cannot connect to the Docker daemon.*socket"):
            await manager.create_session()


# ---------------------------------------------------------------------------
# Session: paths and package installs
# ---------------------------------------------------------------------------


class TestPathValidation:
    @pytest.mark.parametrize(
        "path", ["../etc/passwd", "/workspace/../etc/passwd", "/workspace2/x", "/tmpfoo"]
    )
    def test_traversal_rejected(self, path):
        with pytest.raises(ValueError, match="outside allowed directories"):
            SandboxSession._validate_sandbox_path(path)

    @pytest.mark.parametrize(
        ("path", "expected"),
        [("data/a.csv", "/workspace/data/a.csv"), ("/workspace", "/workspace")],
    )
    def test_valid_paths(self, path, expected):
        assert SandboxSession._validate_sandbox_path(path) == expected


class TestInstallPackage:
    @staticmethod
    def _session(result: CommandResult, network: str = "none") -> tuple[SandboxSession, list[str]]:
        commands: list[str] = []
        backend = MagicMock()

        async def execute_command(container_id, command, timeout, workdir):
            commands.append(command)
            return result

        backend.execute_command = execute_command
        return SandboxSession("c1", backend, SandboxConfig(network=network)), commands

    @pytest.mark.asyncio
    async def test_commands_install_into_workspace(self):
        session, commands = self._session(CommandResult(0, "", ""))
        await session.install_package("pandas")
        await session.install_package("lodash", tool="node")
        await session.install_package("ripgrep", tool="rust")
        await session.install_package("example.com/x@latest", tool="go")
        assert commands[0] == "python3 -m pip install --no-input pandas"
        assert commands[1].startswith("npm install ") and "-g" not in commands[1]
        assert "--root /workspace/.cargo" in commands[2]
        assert commands[3].startswith("GOPATH=/workspace/go ")

    @pytest.mark.asyncio
    async def test_no_network_hint(self):
        session, _ = self._session(CommandResult(1, "", "Could not find a version"))
        result = await session.install_package("pandas")
        assert "no network access (network='none')" in result.stderr

    @pytest.mark.asyncio
    async def test_no_hint_with_network(self):
        session, _ = self._session(CommandResult(1, "", "boom"), network="full")
        result = await session.install_package("pandas")
        assert result.stderr == "boom"


# ---------------------------------------------------------------------------
# build_agent wiring
# ---------------------------------------------------------------------------


class TestBuildAgentSandbox:
    @pytest.mark.asyncio
    async def test_sandbox_failure_raises_instead_of_dropping_tools(self):
        async def boom(self):
            raise RuntimeError("Sandbox backend 'docker' is not available: no daemon")

        with patch("promptise.sandbox.SandboxManager.create_session", boom):
            with pytest.raises(RuntimeError, match="requires a working Docker sandbox.*no daemon"):
                await build_agent(
                    servers={}, model=FakeListChatModel(responses=["x"]), sandbox=True
                )

    @pytest.mark.asyncio
    async def test_code_action_options_reach_the_node(self):
        probe = MagicMock()

        async def cleanup():
            return None

        probe.cleanup = cleanup

        async def create_session(self):
            return probe

        with patch("promptise.sandbox.SandboxManager.create_session", create_session):
            agent = await build_agent(
                servers={},
                model=FakeListChatModel(responses=["x"]),
                agent_pattern="code-action",
                code_action={"exec_timeout": 7, "max_repairs": 3, "max_tool_calls": 9},
            )
        node = agent._inner.graph._nodes["reason"]
        assert isinstance(node, CodeActionNode)
        assert (node.exec_timeout, node.max_repairs, node.max_tool_calls) == (7, 3, 9)
        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_code_action_options_validated(self):
        with pytest.raises(ValidationError):
            CodeActionConfig.model_validate({"exec_timeout": 7, "timeout": 3})
        with pytest.raises(ValueError, match="only valid with agent_pattern='code-action'"):
            await build_agent(
                servers={},
                model=FakeListChatModel(responses=["x"]),
                code_action={"exec_timeout": 7},
            )

    @pytest.mark.asyncio
    async def test_trace_tools_covers_extra_tools(self, capsys):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        agent = await build_agent(
            servers={},
            model=FakeListChatModel(responses=["x"]),
            extra_tools=[add],
            trace_tools=True,
        )
        wrapped = next(t for t in agent._tools if t.name == "add")
        assert wrapped.args == add.args
        assert await wrapped.ainvoke({"a": 2, "b": 3}) == 5
        out = capsys.readouterr().out
        assert "→ Invoking tool: add with {'a': 2, 'b': 3}" in out
        assert "✔ Tool result from add: 5" in out
        await agent.shutdown()

    @pytest.mark.asyncio
    async def test_extra_tools_untouched_without_tracing(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        agent = await build_agent(
            servers={}, model=FakeListChatModel(responses=["x"]), extra_tools=[add]
        )
        assert next(t for t in agent._tools if t.name == "add") is add
        await agent.shutdown()


class TestCodeActionContext:
    def test_history_and_system_context_are_kept(self):
        state = GraphState(
            messages=[
                SystemMessage(content="Relevant memory: the user works in EUR."),
                HumanMessage(content="Total of January orders?"),
                AIMessage(content="1,240"),
                {"role": "user", "content": "And February?"},
            ]
        )
        system_texts, history = _extract_context(state)
        assert system_texts == ["Relevant memory: the user works in EUR."]
        assert [type(m) for m in history] == [HumanMessage, AIMessage]

        node = CodeActionNode("reason", system_prompt="You are a data analyst.")
        messages = node._build_prompt("And February?", [], (system_texts, history))
        assert isinstance(messages[0], SystemMessage)
        assert "You are a data analyst." in messages[0].content
        assert "the user works in EUR" in messages[0].content
        assert messages[1].content == "Total of January orders?"
        assert messages[2].content == "1,240"
        assert messages[-1].content == "Question: And February?"

    def test_single_question_has_no_history(self):
        state = GraphState(messages=[HumanMessage(content="2+2?")])
        assert _extract_context(state) == ([], [])
