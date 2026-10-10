"""Sandbox session management for command execution."""

from __future__ import annotations

import posixpath
import shlex
from dataclasses import dataclass
from typing import Any


@dataclass
class CommandResult:
    """Result of a command execution in the sandbox.

    Attributes:
        exit_code: Process exit code (0 = success)
        stdout: Standard output from command
        stderr: Standard error from command
        timeout: Whether command timed out
        duration: Execution time in seconds
    """

    exit_code: int
    stdout: str
    stderr: str
    timeout: bool = False
    duration: float = 0.0

    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.exit_code == 0 and not self.timeout


class SandboxSession:
    """Manages a persistent sandbox session for command execution.

    This class provides a high-level interface for executing commands in a
    sandbox container, handling timeouts, and managing the session lifecycle.

    Attributes:
        container_id: ID of the running container
        backend: Backend instance managing the container
        config: Sandbox configuration
        _running: Whether session is active
    """

    def __init__(self, container_id: str, backend: Any, config: Any):
        """Initialize sandbox session.

        Args:
            container_id: ID of running container
            backend: Backend instance
            config: Sandbox configuration
        """
        self.container_id = container_id
        self.backend = backend
        self.config = config
        self._running = True

    async def execute(
        self, command: str, timeout: int | None = None, workdir: str | None = None
    ) -> CommandResult:
        """Execute a command in the sandbox.

        Args:
            command: Shell command to execute
            timeout: Optional timeout in seconds (uses config.timeout if None)
            workdir: Optional working directory (uses config.workdir if None)

        Returns:
            CommandResult with execution details

        Raises:
            RuntimeError: If session is not running
        """
        if not self._running:
            raise RuntimeError("Sandbox session is not running")

        timeout = timeout or self.config.timeout
        workdir = workdir or self.config.workdir

        return await self.backend.execute_command(
            self.container_id, command, timeout=timeout, workdir=workdir
        )

    @staticmethod
    def _validate_sandbox_path(path: str) -> str:
        """Validate a file path is safe for sandbox access.

        Rejects path traversal attempts and ensures the path is
        within allowed sandbox directories.

        Args:
            path: Path to validate.

        Returns:
            The normalised path.

        Raises:
            ValueError: If the path is unsafe.
        """
        # Reject null bytes — can truncate strings in C-level code
        if "\x00" in path:
            raise ValueError("Null bytes are not allowed in file paths")
        # Relative paths are relative to the workspace. Normalise AFTER
        # joining so "../etc/passwd" cannot climb out of it.
        if not path.startswith("/"):
            path = posixpath.join("/workspace", path)
        normalised = posixpath.normpath(path)
        # Block traversal outside allowed roots ("/workspace2" is not inside
        # "/workspace").
        allowed_roots = ("/workspace", "/tmp", "/home")  # nosec B108 - container path whitelist, not host
        if not any(
            normalised == root or normalised.startswith(root + "/") for root in allowed_roots
        ):
            raise ValueError(
                f"Path {path!r} resolves to {normalised!r} which is outside "
                f"allowed directories: {allowed_roots}"
            )
        return normalised

    async def read_file(self, file_path: str) -> str:
        """Read a file from the sandbox.

        Args:
            file_path: Path to file inside sandbox (must be within
                ``/workspace``, ``/tmp``, or ``/home``).

        Returns:
            File contents as string.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If path is outside allowed directories.
            RuntimeError: If session is not running.
        """
        if not self._running:
            raise RuntimeError("Sandbox session is not running")

        safe_path = self._validate_sandbox_path(file_path)
        return await self.backend.read_file(self.container_id, safe_path)

    async def write_file(self, file_path: str, content: str) -> None:
        """Write a file to the sandbox.

        Args:
            file_path: Path to file inside sandbox (must be within
                ``/workspace``, ``/tmp``, or ``/home``).
            content: File contents to write.

        Raises:
            ValueError: If path is outside allowed directories.
            RuntimeError: If session is not running, or the write fails.
        """
        if not self._running:
            raise RuntimeError("Sandbox session is not running")

        safe_path = self._validate_sandbox_path(file_path)
        await self.backend.write_file(self.container_id, safe_path, content)

    async def list_files(self, directory: str = "/workspace") -> list[str]:
        """List files in a directory.

        Args:
            directory: Directory path inside sandbox.

        Returns:
            List of file/directory names.

        Raises:
            ValueError: If directory is outside allowed directories.
            RuntimeError: If session is not running.
        """
        if not self._running:
            raise RuntimeError("Sandbox session is not running")

        safe_dir = self._validate_sandbox_path(directory)
        result = await self.execute(f"ls -1 -- {shlex.quote(safe_dir)}")
        if result.success:
            return [line.strip() for line in result.stdout.split("\n") if line.strip()]
        return []

    async def install_package(self, package: str, tool: str = "python") -> CommandResult:
        """Install a package in the sandbox.

        Packages are installed into the writable workspace, so this works on
        the default read-only root filesystem: ``pip`` falls back to a user
        install under ``$HOME`` (the workspace), ``npm`` installs into
        ``<workdir>/node_modules``, ``cargo`` into ``<workdir>/.cargo`` and
        ``go`` into ``<workdir>/go``. Downloading a package needs network
        access, which the default ``network="none"`` does not provide; bake
        packages into a custom ``image`` instead, or enable the network.

        Args:
            package: Package name to install (shell-escaped internally).
            tool: Tool ecosystem (``"python"``, ``"node"``, ``"rust"``,
                ``"go"``).

        Returns:
            CommandResult from installation.

        Raises:
            RuntimeError: If session is not running.
            ValueError: If tool is not supported.
        """
        if not self._running:
            raise RuntimeError("Sandbox session is not running")

        safe_pkg = shlex.quote(package)
        workdir = shlex.quote(self.config.workdir)
        install_commands = {
            "python": f"python3 -m pip install --no-input {safe_pkg}",
            "node": f"npm install --no-fund --no-audit {safe_pkg}",
            "rust": f"cargo install --root {workdir}/.cargo {safe_pkg}",
            "go": f"GOPATH={workdir}/go go install {safe_pkg}",
        }

        if tool not in install_commands:
            raise ValueError(
                f"Unsupported tool: {tool}. Use one of: {list(install_commands.keys())}"
            )

        result = await self.execute(install_commands[tool])
        network = getattr(self.config.network, "value", self.config.network)
        if not result.success and network == "none":
            result.stderr = (
                f"{result.stderr}\nNote: this sandbox has no network access "
                "(network='none'), so packages cannot be downloaded. Use an image "
                "that already contains the package, or configure network access."
            ).lstrip("\n")
        return result

    async def cleanup(self) -> None:
        """Clean up the sandbox session.

        Stops and removes the container unless persistent mode is enabled.
        """
        if self._running:
            self._running = False
            if not self.config.persistent:
                await self.backend.stop_container(self.container_id)
                await self.backend.remove_container(self.container_id)

    async def __aenter__(self) -> SandboxSession:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup()
