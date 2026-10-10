"""Sandbox configuration schemas and defaults."""

from __future__ import annotations

import difflib
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class NetworkMode(str, Enum):
    """Network isolation modes for sandbox."""

    NONE = "none"  # No network interface besides loopback
    RESTRICTED = "restricted"  # Outbound DNS + TCP 80/443 only, enforced with iptables
    FULL = "full"  # Full network access


# Keys that are not sandbox options but are easy to reach for. Each maps to
# the hint shown in the validation error.
_KEY_HINTS: dict[str, str] = {
    "network_mode": 'use \'network\' ("none", "restricted" or "full")',
    "tools": (
        "the sandbox does not install tool ecosystems; pick an 'image' that "
        "already contains them (for example 'node:22-slim')"
    ),
    "security_opt": "security options are fixed by the sandbox and cannot be overridden",
    "cap_drop": "the sandbox always drops every capability it can; this is not configurable",
}

_SIZE_UNITS = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_size(size: str) -> int:
    """Convert a size string such as ``"512M"`` or ``"4G"`` to bytes.

    Args:
        size: A positive integer followed by ``K``, ``M``, ``G`` or ``T``.

    Returns:
        The size in bytes.
    """
    return int(size[:-1]) * _SIZE_UNITS[size[-1]]


class SandboxConfig(BaseModel):
    """Configuration for sandbox environment.

    Unknown keys are rejected, so a misspelled option (``network_mode``
    instead of ``network``) fails loudly instead of being ignored.

    Attributes:
        backend: Container backend to use (docker, gvisor)
        image: Base container image (default: python:3.11-slim)
        cpu_limit: Maximum CPU cores (default: 2)
        memory_limit: Maximum memory (default: "4G")
        disk_limit: Size of the writable workspace (a tmpfs mounted at
            ``workdir``; it counts toward ``memory_limit``) (default: "1G")
        pids_limit: Maximum number of processes and threads (default: 256)
        network: Network isolation mode (default: "none")
        persistent: Keep the container running after the session ends (default: False)
        timeout: Max execution time in seconds (default: 300)
        workdir: Working directory inside container (default: "/workspace")
        env: Additional environment variables
        allow_sudo: Allow sudo access in container (default: False)
        runtime: Docker runtime to run the container with (``"runsc"`` for gVisor)
        read_only_rootfs: Mount the root filesystem read-only (default: True)

    Examples:
        >>> # Minimal config
        >>> config = SandboxConfig()
        >>>
        >>> # Custom config
        >>> config = SandboxConfig(
        ...     backend="gvisor",
        ...     cpu_limit=4,
        ...     memory_limit="8G",
        ...     network=NetworkMode.FULL,
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["docker", "gvisor"] = Field("docker", description="Container backend")
    image: str = Field("python:3.11-slim", description="Base container image")
    cpu_limit: int = Field(2, gt=0, le=32, description="Maximum CPU cores")
    memory_limit: str = Field("4G", description="Maximum memory (e.g., '4G', '512M')")
    disk_limit: str = Field("1G", description="Size of the writable workspace (e.g., '1G')")
    pids_limit: int = Field(256, gt=0, le=65536, description="Maximum processes and threads")
    network: NetworkMode = Field(NetworkMode.NONE, description="Network isolation mode")
    persistent: bool = Field(False, description="Keep the container after the session ends")
    timeout: int = Field(300, gt=0, le=3600, description="Max execution time in seconds")
    workdir: str = Field("/workspace", description="Working directory inside container")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    allow_sudo: bool = Field(False, description="Allow sudo access in container")
    runtime: str | None = Field(None, description="Docker runtime (e.g., 'runsc' for gVisor)")
    read_only_rootfs: bool = Field(True, description="Read-only root filesystem")

    @model_validator(mode="before")
    @classmethod
    def _reject_unknown_keys(cls, data: Any) -> Any:
        """Reject unknown keys with a hint instead of silently ignoring them."""
        if not isinstance(data, dict):
            return data
        unknown = [key for key in data if key not in cls.model_fields]
        if not unknown:
            return data
        problems = []
        for key in unknown:
            hint = _KEY_HINTS.get(key)
            if hint is None:
                close = difflib.get_close_matches(str(key), list(cls.model_fields), n=1)
                hint = f"did you mean '{close[0]}'?" if close else "not a sandbox option"
            problems.append(f"'{key}': {hint}")
        raise ValueError(
            "Unknown sandbox option(s): "
            + "; ".join(problems)
            + f". Valid options: {', '.join(cls.model_fields)}"
        )

    @field_validator("memory_limit", "disk_limit")
    @classmethod
    def validate_size(cls, v: str) -> str:
        """Validate memory/disk size format."""
        if not v:
            raise ValueError("Size cannot be empty")
        if v[-1] not in _SIZE_UNITS:
            raise ValueError(f"Size must end with K, M, G, or T: {v}")
        try:
            size = int(v[:-1])
        except ValueError:
            raise ValueError(f"Invalid size format: {v}") from None
        if size <= 0:
            raise ValueError(f"Size must be positive: {v}")
        return v

    @classmethod
    def from_simple(cls, enabled: bool = True) -> SandboxConfig:
        """Create config from simple boolean flag.

        Args:
            enabled: If True, use default config

        Returns:
            SandboxConfig with defaults
        """
        if not enabled:
            raise ValueError("Cannot create config when sandbox is disabled")
        return cls()

    @classmethod
    def from_dict(cls, data: dict) -> SandboxConfig:
        """Create config from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            SandboxConfig instance

        Raises:
            pydantic.ValidationError: If a key is unknown or a value is invalid.
        """
        return cls(**data)


# Capabilities to drop (all except what's needed)
DEFAULT_CAP_DROP = [
    "CAP_AUDIT_CONTROL",
    "CAP_AUDIT_READ",
    "CAP_AUDIT_WRITE",
    "CAP_BLOCK_SUSPEND",
    "CAP_BPF",
    "CAP_CHECKPOINT_RESTORE",
    "CAP_DAC_OVERRIDE",
    "CAP_DAC_READ_SEARCH",
    "CAP_FOWNER",
    "CAP_FSETID",
    "CAP_IPC_LOCK",
    "CAP_IPC_OWNER",
    "CAP_KILL",
    "CAP_LEASE",
    "CAP_LINUX_IMMUTABLE",
    "CAP_MAC_ADMIN",
    "CAP_MAC_OVERRIDE",
    "CAP_MKNOD",
    "CAP_NET_ADMIN",
    "CAP_NET_BIND_SERVICE",
    "CAP_NET_BROADCAST",
    "CAP_NET_RAW",
    "CAP_PERFMON",
    "CAP_SETFCAP",
    "CAP_SETGID",
    "CAP_SETPCAP",
    "CAP_SETUID",
    "CAP_SYS_ADMIN",
    "CAP_SYS_BOOT",
    "CAP_SYS_CHROOT",
    "CAP_SYS_MODULE",
    "CAP_SYS_NICE",
    "CAP_SYS_PACCT",
    "CAP_SYS_PTRACE",
    "CAP_SYS_RAWIO",
    "CAP_SYS_RESOURCE",
    "CAP_SYS_TIME",
    "CAP_SYS_TTY_CONFIG",
    "CAP_SYSLOG",
    "CAP_WAKE_ALARM",
]
