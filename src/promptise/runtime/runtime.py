"""AgentRuntime — multi-process manager for agent processes.

The :class:`AgentRuntime` is the operating system kernel for your AI
agents: it manages a collection of :class:`AgentProcess` instances,
handles shared resources (EventBus, MessageBroker), and provides the
surface API consumed by the CLI and programmatic callers.

Example::

    from promptise.runtime import AgentRuntime, ProcessConfig, TriggerConfig

    async with AgentRuntime() as runtime:
        await runtime.add_process(
            "data-watcher",
            ProcessConfig(
                model="openai:gpt-5-mini",
                instructions="You monitor data pipelines.",
                triggers=[
                    TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
                ],
            ),
        )
        await runtime.start_all()

        # Runtime manages processes until stopped …
        await runtime.stop_all()
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from .config import ProcessConfig, RuntimeConfig
from .exceptions import ManifestError, RuntimeBaseError
from .lifecycle import ProcessState
from .manifest import (
    load_manifest,
    manifest_to_process_config,
)
from .process import AgentProcess

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Multi-process agent runtime manager.

    Manages a dictionary of :class:`AgentProcess` instances with shared
    resources and coordinated lifecycle.

    Args:
        config: Optional :class:`RuntimeConfig` for global settings.
        event_bus: Optional shared EventBus for inter-process events.
        broker: Optional shared MessageBroker for message triggers.
    """

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        event_bus: Any | None = None,
        broker: Any | None = None,
        event_notifier: Any | None = None,
    ) -> None:
        self._config = config or RuntimeConfig()
        self._event_bus = event_bus
        self._broker = broker
        self._event_notifier = event_notifier

        # Process registry
        self._processes: dict[str, AgentProcess] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    async def add_process(
        self,
        name: str,
        config: ProcessConfig,
        *,
        process_id: str | None = None,
    ) -> AgentProcess:
        """Register a new agent process.

        Args:
            name: Unique process name.
            config: Process configuration.
            process_id: Optional explicit process ID.

        Returns:
            The created :class:`AgentProcess`.

        Raises:
            RuntimeBaseError: If a process with this name already exists.
        """
        async with self._lock:
            if name in self._processes:
                raise RuntimeBaseError(
                    f"Process {name!r} already exists. Use remove_process() "
                    "first or choose a different name."
                )

            process = AgentProcess(
                name=name,
                config=config,
                process_id=process_id,
                event_bus=self._event_bus,
                broker=self._broker,
                runtime=self,
                event_notifier=self._event_notifier,
            )
            self._processes[name] = process
            logger.info("AgentRuntime: registered process %r", name)
            return process

    async def remove_process(self, name: str) -> None:
        """Remove a process (stopping it first if running).

        Args:
            name: Process name to remove.

        Raises:
            KeyError: If the process does not exist.
        """
        async with self._lock:
            process = self._processes.get(name)
            if process is None:
                raise KeyError(f"No process named {name!r}")

            if process.state not in (
                ProcessState.STOPPED,
                ProcessState.CREATED,
                ProcessState.FAILED,
            ):
                await process.stop()

            del self._processes[name]
            logger.info("AgentRuntime: removed process %r", name)

    def get_process(self, name: str) -> AgentProcess:
        """Get a process by name.

        Args:
            name: Process name.

        Returns:
            The :class:`AgentProcess`.

        Raises:
            KeyError: If the process does not exist.
        """
        process = self._processes.get(name)
        if process is None:
            raise KeyError(f"No process named {name!r}")
        return process

    @property
    def processes(self) -> dict[str, AgentProcess]:
        """Read-only view of registered processes."""
        return dict(self._processes)

    # ------------------------------------------------------------------
    # Manifest loading
    # ------------------------------------------------------------------

    async def load_manifest(
        self,
        path: str | Path,
        *,
        name_override: str | None = None,
    ) -> AgentProcess:
        """Load and register a process from a ``.agent`` manifest.

        Args:
            path: Path to the ``.agent`` file.
            name_override: Override the process name from the manifest.

        Returns:
            The created :class:`AgentProcess`.

        Raises:
            ManifestError: If the manifest cannot be loaded.
        """
        manifest = load_manifest(path)
        config = manifest_to_process_config(manifest)
        name = name_override or manifest.name
        return await self.add_process(name, config)

    async def load_directory(self, directory: str | Path) -> list[str]:
        """Load all ``.agent`` files from a directory.

        Args:
            directory: Directory containing ``.agent`` files.

        Returns:
            List of process names that were loaded.

        Raises:
            ManifestError: If any manifest fails to load.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise ManifestError(f"Not a directory: {directory}")

        loaded: list[str] = []
        for agent_file in sorted(directory.glob("*.agent")):
            try:
                process = await self.load_manifest(agent_file)
                loaded.append(process.name)
                logger.info(
                    "AgentRuntime: loaded %s from %s",
                    process.name,
                    agent_file,
                )
            except Exception as exc:
                logger.error(
                    "AgentRuntime: failed to load %s: %s",
                    agent_file,
                    exc,
                )
                raise

        return loaded

    # ------------------------------------------------------------------
    # Lifecycle control
    # ------------------------------------------------------------------

    async def start_process(self, name: str) -> None:
        """Start a single process.

        Args:
            name: Process name.

        Raises:
            KeyError: If the process does not exist.
        """
        process = self.get_process(name)
        await process.start()

    async def stop_process(self, name: str) -> None:
        """Stop a single process.

        Args:
            name: Process name.

        Raises:
            KeyError: If the process does not exist.
        """
        process = self.get_process(name)
        await process.stop()

    async def restart_process(self, name: str) -> None:
        """Restart a single process (stop then start).

        Args:
            name: Process name.

        Raises:
            KeyError: If the process does not exist.
        """
        process = self.get_process(name)
        if process.state not in (
            ProcessState.STOPPED,
            ProcessState.CREATED,
            ProcessState.FAILED,
        ):
            await process.stop()
        await process.start()

    async def start_all(self) -> None:
        """Start all registered processes."""
        for name, process in self._processes.items():
            if process.state in (
                ProcessState.CREATED,
                ProcessState.STOPPED,
                ProcessState.FAILED,
            ):
                try:
                    await process.start()
                except Exception as exc:
                    logger.error("AgentRuntime: failed to start %s: %s", name, exc)

    async def stop_all(self) -> None:
        """Stop all running processes.

        Afterwards the shared event notifier (if any) is drained, so every
        ``process.stopped`` event is delivered before this returns.
        """
        for name, process in list(self._processes.items()):
            if process.state not in (
                ProcessState.STOPPED,
                ProcessState.CREATED,
            ):
                try:
                    await process.stop()
                except Exception as exc:
                    logger.error("AgentRuntime: failed to stop %s: %s", name, exc)
        if self._event_notifier is not None and hasattr(self._event_notifier, "stop"):
            try:
                await self._event_notifier.stop()
            except Exception as exc:
                logger.error("AgentRuntime: failed to drain event notifier: %s", exc)

    # ------------------------------------------------------------------
    # Status and monitoring
    # ------------------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Global runtime status.

        Returns:
            Dict with process count and per-process status.
        """
        process_statuses = {}
        for name, process in self._processes.items():
            process_statuses[name] = process.status()

        return {
            "process_count": len(self._processes),
            "processes": process_statuses,
        }

    def process_status(self, name: str) -> dict[str, Any]:
        """Status for a single process.

        Args:
            name: Process name.

        Returns:
            Process status dict.

        Raises:
            KeyError: If the process does not exist.
        """
        return self.get_process(name).status()

    def list_processes(self) -> list[dict[str, Any]]:
        """List all processes with summary info.

        Returns:
            List of dicts with name, state, and uptime for each process.
        """
        result: list[dict[str, Any]] = []
        for name, process in self._processes.items():
            result.append(
                {
                    "name": name,
                    "state": process.state.value,
                    "process_id": process.process_id,
                }
            )
        return result

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> AgentRuntime:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.stop_all()

    # ------------------------------------------------------------------
    # Config-based initialization
    # ------------------------------------------------------------------

    @classmethod
    async def from_config(
        cls,
        config: RuntimeConfig,
        *,
        event_bus: Any | None = None,
        broker: Any | None = None,
        event_notifier: Any | None = None,
    ) -> AgentRuntime:
        """Create a runtime from a :class:`RuntimeConfig` with all
        processes pre-registered.

        Args:
            config: Runtime configuration.
            event_bus: Optional shared EventBus.
            broker: Optional shared MessageBroker.
            event_notifier: Optional :class:`EventNotifier` for all processes.

        Returns:
            A new :class:`AgentRuntime` with processes registered.
        """
        runtime = cls(
            config=config,
            event_bus=event_bus,
            broker=broker,
            event_notifier=event_notifier,
        )
        for name, process_config in config.processes.items():
            await runtime.add_process(name, process_config)
        return runtime

    def __repr__(self) -> str:
        states = {}
        for name, process in self._processes.items():
            state = process.state.value
            states[state] = states.get(state, 0) + 1
        state_str = ", ".join(f"{v} {k}" for k, v in states.items())
        return (
            f"AgentRuntime(processes={len(self._processes)}{', ' + state_str if state_str else ''})"
        )
