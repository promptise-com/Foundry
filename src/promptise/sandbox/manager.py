"""Sandbox manager for lifecycle management and tool integration."""

from __future__ import annotations

import logging
from typing import Any

from .backends import DockerBackend, SandboxBackend
from .config import SandboxConfig
from .session import SandboxSession

logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages sandbox lifecycle and provides integration with agents.

    The SandboxManager handles:
    - Backend selection and initialization
    - Container lifecycle management
    - Session creation and cleanup
    - Health monitoring

    Example:
        >>> config = SandboxConfig()
        >>> manager = SandboxManager(config)
        >>>
        >>> async with await manager.create_session() as session:
        ...     result = await session.execute("python --version")
        ...     print(result.stdout)
    """

    def __init__(self, config: SandboxConfig | dict[str, Any] | bool):
        """Initialize sandbox manager.

        Args:
            config: Sandbox configuration (SandboxConfig, dict, or bool)

        Raises:
            ValueError: If config is invalid (``pydantic.ValidationError``,
                a ``ValueError`` subclass, for unknown keys or bad values)
        """
        # Normalize config
        if isinstance(config, bool):
            if not config:
                raise ValueError("Cannot create SandboxManager with sandbox=False")
            self.config = SandboxConfig.from_simple(enabled=True)
        elif isinstance(config, dict):
            self.config = SandboxConfig.from_dict(config)
        elif isinstance(config, SandboxConfig):
            self.config = config
        else:
            raise ValueError(f"Invalid config type: {type(config)}")

        # Select and initialize backend
        self.backend = self._create_backend()
        self._sessions: list[SandboxSession] = []

    def _create_backend(self) -> SandboxBackend:
        """Create backend based on configuration.

        Returns:
            SandboxBackend instance

        Raises:
            ValueError: If backend is not supported
        """
        backend_type = self.config.backend

        if backend_type in ("docker", "gvisor"):
            # gVisor is just Docker with runsc runtime
            if backend_type == "gvisor":
                self.config.runtime = "runsc"
            return DockerBackend(self.config)

        else:
            raise ValueError(f"Unsupported backend: {backend_type}. Use one of: docker, gvisor")

    async def create_session(self) -> SandboxSession:
        """Create a new sandbox session.

        Returns:
            SandboxSession for command execution

        Raises:
            RuntimeError: If backend is not healthy or container creation fails
        """
        try:
            await self.backend.ensure_available()
        except Exception as e:
            raise RuntimeError(
                f"Sandbox backend '{self.config.backend}' is not available: {e}"
            ) from e

        # Create container
        try:
            container_id = await self.backend.create_container()
        except Exception as e:
            raise RuntimeError(f"Failed to create sandbox container: {e}") from e

        # Create session
        session = SandboxSession(container_id, self.backend, self.config)
        self._sessions.append(session)

        return session

    async def cleanup_all(self) -> None:
        """Clean up all active sessions."""
        for session in self._sessions:
            try:
                await session.cleanup()
            except Exception as e:
                logger.warning("[sandbox] Failed to clean up session: %s", e)

        self._sessions.clear()

    async def __aenter__(self) -> SandboxManager:
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.cleanup_all()
