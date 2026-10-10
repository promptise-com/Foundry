"""Secure sandbox for agent code execution.

This module provides a zero-configuration sandbox environment for agents to
safely execute arbitrary commands and code. The sandbox is implemented using
container technology with multiple security layers.

Key Features:
- Zero-config: Enable with sandbox=True (install with ``promptise[sandbox]``)
- Multiple backends: Docker (with optional gVisor runtime)
- Layered security: no network by default, Docker's default seccomp profile,
  dropped capabilities, no-new-privileges, read-only root filesystem,
  optional gVisor
- Full CLI access: Execute any command within the isolated environment
- Resource limits: CPU, memory, processes, workspace size, execution time

Example:
    >>> from promptise import build_agent
    >>>
    >>> # Simple usage
    >>> agent = await build_agent(
    ...     model="anthropic:claude-opus-4.5",
    ...     instructions="You are a coding assistant.",
    ...     sandbox=True
    ... )
    >>>
    >>> # Advanced configuration
    >>> agent = await build_agent(
    ...     model="anthropic:claude-opus-4.5",
    ...     instructions="...",
    ...     sandbox={
    ...         "backend": "gvisor",
    ...         "cpu_limit": 2,
    ...         "memory_limit": "4G",
    ...         "network": "none",
    ...     }
    ... )
"""

from .backends import DockerBackend, SandboxBackend
from .config import NetworkMode, SandboxConfig
from .manager import SandboxManager
from .session import CommandResult, SandboxSession
from .utils import SandboxContainerManager, cleanup_on_exit

__all__ = [
    "SandboxBackend",
    "DockerBackend",
    "SandboxConfig",
    "NetworkMode",
    "SandboxManager",
    "SandboxSession",
    "CommandResult",
    "SandboxContainerManager",
    "cleanup_on_exit",
]
