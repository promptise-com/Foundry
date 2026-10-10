"""End-to-end tests for the sandbox subsystem.

Tests cover: SandboxConfig, SandboxManager, CommandResult, DockerBackend,
sandbox tools, and config edge cases -- all without requiring Docker.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from promptise.sandbox import (
    CommandResult,
    DockerBackend,
    NetworkMode,
    SandboxConfig,
    SandboxManager,
    SandboxSession,
)
from promptise.sandbox.tools import create_sandbox_tools

# ---------------------------------------------------------------------------
# 1. SandboxConfig tests (5)
# ---------------------------------------------------------------------------


class TestSandboxConfigDefaults:
    """Verify that default SandboxConfig uses secure, restrictive defaults."""

    def test_secure_defaults(self):
        """Default config should be locked-down: no network, no sudo,
        read-only rootfs, moderate resource limits."""
        cfg = SandboxConfig()

        assert cfg.backend == "docker"
        assert cfg.image == "python:3.11-slim"
        assert cfg.cpu_limit == 2
        assert cfg.memory_limit == "4G"
        assert cfg.disk_limit == "1G"
        assert cfg.pids_limit == 256
        assert cfg.network == NetworkMode.NONE
        assert cfg.persistent is False
        assert cfg.timeout == 300
        assert cfg.workdir == "/workspace"
        assert cfg.env == {}
        assert cfg.allow_sudo is False
        assert cfg.read_only_rootfs is True
        assert cfg.runtime is None
        assert not hasattr(cfg, "tools")
        assert not hasattr(cfg, "security_opt")
        assert not hasattr(cfg, "cap_drop")


class TestSandboxConfigFromDict:
    """Verify SandboxConfig.from_dict with custom values."""

    def test_custom_values(self):
        """from_dict should correctly populate all overridden fields."""
        data = {
            "backend": "gvisor",
            "cpu_limit": 8,
            "memory_limit": "16G",
            "disk_limit": "50G",
            "network": "full",
            "persistent": True,
            "timeout": 600,
            "pids_limit": 64,
            "workdir": "/app",
            "env": {"MY_VAR": "hello"},
            "allow_sudo": True,
        }
        cfg = SandboxConfig.from_dict(data)

        assert cfg.backend == "gvisor"
        assert cfg.cpu_limit == 8
        assert cfg.memory_limit == "16G"
        assert cfg.disk_limit == "50G"
        assert cfg.network == NetworkMode.FULL
        assert cfg.persistent is True
        assert cfg.timeout == 600
        assert cfg.pids_limit == 64
        assert cfg.workdir == "/app"
        assert cfg.env == {"MY_VAR": "hello"}
        assert cfg.allow_sudo is True


class TestNetworkModeEnum:
    """Verify NetworkMode enum has exactly the expected members and values."""

    def test_members(self):
        assert NetworkMode.NONE.value == "none"
        assert NetworkMode.RESTRICTED.value == "restricted"
        assert NetworkMode.FULL.value == "full"

    def test_is_string_enum(self):
        """NetworkMode should behave as a str so Pydantic can coerce from raw strings."""
        assert isinstance(NetworkMode.NONE, str)
        assert NetworkMode("restricted") == NetworkMode.RESTRICTED

    def test_all_members_count(self):
        assert len(NetworkMode) == 3


class TestResourceLimitsValidation:
    """Verify that invalid resource values are rejected by Pydantic."""

    def test_cpu_limit_too_low(self):
        with pytest.raises(ValidationError):
            SandboxConfig(cpu_limit=0)

    def test_cpu_limit_too_high(self):
        with pytest.raises(ValidationError):
            SandboxConfig(cpu_limit=33)

    def test_memory_limit_missing_unit(self):
        with pytest.raises(ValidationError):
            SandboxConfig(memory_limit="1024")

    def test_memory_limit_invalid_format(self):
        with pytest.raises(ValidationError):
            SandboxConfig(memory_limit="lots")

    def test_disk_limit_negative(self):
        with pytest.raises(ValidationError):
            SandboxConfig(disk_limit="-1G")

    def test_timeout_too_low(self):
        with pytest.raises(ValidationError):
            SandboxConfig(timeout=0)

    def test_timeout_too_high(self):
        with pytest.raises(ValidationError):
            SandboxConfig(timeout=3601)

    def test_valid_sizes_accepted(self):
        """All valid unit suffixes should be accepted."""
        for size in ("1K", "512M", "4G", "1T"):
            cfg = SandboxConfig(memory_limit=size)
            assert cfg.memory_limit == size


class TestBackendValidation:
    """Verify that only supported backend strings are accepted."""

    def test_valid_backends(self):
        for backend in ("docker", "gvisor"):
            cfg = SandboxConfig(backend=backend)
            assert cfg.backend == backend

    def test_invalid_backend_rejected(self):
        with pytest.raises(ValidationError):
            SandboxConfig(backend="lxc")


# ---------------------------------------------------------------------------
# 2. SandboxManager tests (2)
# ---------------------------------------------------------------------------


class TestSandboxManagerCreationWithoutDocker:
    """SandboxManager should instantiate and wire up the backend without
    needing a live Docker daemon."""

    def test_manager_from_bool(self):
        """Creating a manager with True should give default config + DockerBackend."""
        manager = SandboxManager(True)
        assert isinstance(manager.config, SandboxConfig)
        assert manager.config.backend == "docker"
        assert isinstance(manager.backend, DockerBackend)

    def test_manager_from_dict(self):
        manager = SandboxManager({"backend": "gvisor", "cpu_limit": 4})
        assert manager.config.backend == "gvisor"
        assert manager.config.runtime == "runsc"
        assert isinstance(manager.backend, DockerBackend)

    def test_manager_false_rejected(self):
        with pytest.raises(ValueError, match="sandbox=False"):
            SandboxManager(False)

    def test_manager_invalid_type_rejected(self):
        with pytest.raises(ValueError, match="Invalid config type"):
            SandboxManager(42)  # type: ignore[arg-type]


class TestSandboxManagerFields:
    """Verify SandboxManager exposes the expected public fields."""

    def test_fields_present(self):
        manager = SandboxManager(True)
        assert hasattr(manager, "config")
        assert hasattr(manager, "backend")
        assert hasattr(manager, "_sessions")
        assert isinstance(manager._sessions, list)
        assert len(manager._sessions) == 0


# ---------------------------------------------------------------------------
# 3. CommandResult test (1)
# ---------------------------------------------------------------------------


class TestCommandResult:
    """Verify CommandResult dataclass fields and the `success` property."""

    def test_fields_accessible(self):
        result = CommandResult(
            exit_code=0,
            stdout="hello world",
            stderr="",
            timeout=False,
            duration=1.23,
        )
        assert result.exit_code == 0
        assert result.stdout == "hello world"
        assert result.stderr == ""
        assert result.timeout is False
        assert result.duration == 1.23

    def test_success_true_on_zero_exit(self):
        result = CommandResult(exit_code=0, stdout="", stderr="")
        assert result.success is True

    def test_success_false_on_nonzero_exit(self):
        result = CommandResult(exit_code=1, stdout="", stderr="err")
        assert result.success is False

    def test_success_false_on_timeout(self):
        result = CommandResult(exit_code=0, stdout="", stderr="", timeout=True)
        assert result.success is False


# ---------------------------------------------------------------------------
# 4. DockerBackend init test (1)
# ---------------------------------------------------------------------------


class TestDockerBackendInit:
    """DockerBackend should initialise without connecting to Docker."""

    def test_init_stores_config(self):
        cfg = SandboxConfig()
        backend = DockerBackend(cfg)
        assert backend.config is cfg
        # No Docker client should be created until first use
        assert backend._docker_client is None

    def test_gvisor_config_sets_runtime(self):
        cfg = SandboxConfig(backend="gvisor")
        cfg.runtime = "runsc"
        backend = DockerBackend(cfg)
        assert backend.config.runtime == "runsc"


# ---------------------------------------------------------------------------
# 5. Sandbox tools tests (2)
# ---------------------------------------------------------------------------


class TestCreateSandboxTools:
    """Verify create_sandbox_tools returns the correct tool set."""

    def _mock_session(self) -> MagicMock:
        return MagicMock(spec=SandboxSession)

    def test_returns_five_tools(self):
        tools = create_sandbox_tools(self._mock_session())
        assert len(tools) == 5

    def test_tool_names_correct(self):
        tools = create_sandbox_tools(self._mock_session())
        names = {t.name for t in tools}
        expected = {
            "sandbox_exec",
            "sandbox_read_file",
            "sandbox_write_file",
            "sandbox_list_files",
            "sandbox_install_package",
        }
        assert names == expected


# ---------------------------------------------------------------------------
# 6. Config edge case: serialization roundtrip (1)
# ---------------------------------------------------------------------------


class TestSandboxConfigSerializationRoundtrip:
    """SandboxConfig should survive a dict -> model -> dict -> model cycle."""

    def test_roundtrip(self):
        original = SandboxConfig(
            backend="gvisor",
            cpu_limit=4,
            memory_limit="8G",
            disk_limit="20G",
            network=NetworkMode.FULL,
            persistent=True,
            timeout=120,
            pids_limit=128,
            workdir="/data",
            env={"FOO": "bar"},
            allow_sudo=True,
        )

        dumped = original.model_dump()
        restored = SandboxConfig(**dumped)

        assert restored.backend == original.backend
        assert restored.cpu_limit == original.cpu_limit
        assert restored.memory_limit == original.memory_limit
        assert restored.disk_limit == original.disk_limit
        assert restored.network == original.network
        assert restored.persistent == original.persistent
        assert restored.timeout == original.timeout
        assert restored.pids_limit == original.pids_limit
        assert restored.workdir == original.workdir
        assert restored.env == original.env
        assert restored.allow_sudo == original.allow_sudo
        assert restored.read_only_rootfs == original.read_only_rootfs

    def test_roundtrip_defaults(self):
        """Even a default config should survive the roundtrip."""
        original = SandboxConfig()
        restored = SandboxConfig(**original.model_dump())
        assert restored.model_dump() == original.model_dump()
