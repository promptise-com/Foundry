"""Sandbox utilities for container management and monitoring."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

try:
    import docker
except ImportError:
    docker = None  # type: ignore[assignment]


class SandboxContainerManager:
    """Utility for managing and monitoring sandbox containers.

    Provides tools to:
    - List all sandbox containers (running and stopped)
    - Clean up orphaned containers
    - Monitor resource usage
    - Force cleanup of stuck containers
    """

    def __init__(self):
        """Initialize container manager."""
        if docker is None:
            raise RuntimeError(
                "The 'docker' package is required for the sandbox. "
                "Install it with: pip install 'promptise[sandbox]'"
            )
        self.client = docker.from_env()
        self.label_prefix = "promptise.sandbox"

    def list_containers(self, all: bool = False) -> list[dict[str, Any]]:
        """List all sandbox containers.

        Args:
            all: If True, include stopped containers

        Returns:
            List of container info dictionaries
        """
        filters = {"label": self.label_prefix}
        containers = self.client.containers.list(all=all, filters=filters)

        result = []
        for container in containers:
            info = {
                "id": container.id[:12],
                "full_id": container.id,
                "name": container.name,
                "status": container.status,
                "image": container.image.tags[0]
                if container.image.tags
                else container.image.id[:12],
                "created": container.attrs["Created"],
                "labels": container.labels,
            }

            # Add runtime stats for running containers
            if container.status == "running":
                try:
                    stats = container.stats(stream=False)
                    info["cpu_usage"] = self._calculate_cpu_percent(stats)
                    info["memory_usage"] = self._format_memory(stats)
                except Exception:
                    info["cpu_usage"] = "N/A"
                    info["memory_usage"] = "N/A"

            result.append(info)

        return result

    def cleanup_all(self, force: bool = False) -> int:
        """Clean up all sandbox containers.

        Args:
            force: If True, force remove running containers

        Returns:
            Number of containers cleaned up
        """
        filters = {"label": self.label_prefix}
        containers = self.client.containers.list(all=True, filters=filters)

        count = 0
        for container in containers:
            try:
                if container.status == "running":
                    if force:
                        container.stop(timeout=5)
                        container.remove(force=True)
                        count += 1
                        print(
                            f"[sandbox] Forcefully removed running container: {container.id[:12]}"
                        )
                    else:
                        print(
                            f"[sandbox] Skipping running container: {container.id[:12]} (use force=True)"
                        )
                else:
                    container.remove()
                    count += 1
                    print(f"[sandbox] Removed stopped container: {container.id[:12]}")
            except Exception as e:
                print(f"[sandbox] Failed to remove container {container.id[:12]}: {e}")

        return count

    def cleanup_old(self, max_age_hours: int = 24) -> int:
        """Clean up containers older than specified age.

        Args:
            max_age_hours: Maximum age in hours

        Returns:
            Number of containers cleaned up
        """
        from datetime import timedelta

        filters = {"label": self.label_prefix}
        containers = self.client.containers.list(all=True, filters=filters)

        cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        count = 0

        for container in containers:
            created_str = container.attrs["Created"]
            # Parse ISO format timestamp
            created = datetime.fromisoformat(created_str.replace("Z", "+00:00"))

            if created < cutoff:
                try:
                    if container.status == "running":
                        container.stop(timeout=5)
                    container.remove(force=True)
                    count += 1
                    print(
                        f"[sandbox] Removed old container: {container.id[:12]} (created {created})"
                    )
                except Exception as e:
                    print(f"[sandbox] Failed to remove old container {container.id[:12]}: {e}")

        return count

    def get_stats(self, container_id: str) -> dict[str, Any]:
        """Get detailed stats for a container.

        Args:
            container_id: Container ID (short or full)

        Returns:
            Dictionary with container statistics

        Raises:
            docker.errors.NotFound: If container doesn't exist
        """
        container = self.client.containers.get(container_id)
        stats = container.stats(stream=False)

        return {
            "id": container.id[:12],
            "name": container.name,
            "status": container.status,
            "cpu_percent": self._calculate_cpu_percent(stats),
            "memory_usage": self._format_memory(stats),
            "memory_percent": self._calculate_memory_percent(stats),
            "network_rx": self._format_bytes(self._get_network_rx(stats)),
            "network_tx": self._format_bytes(self._get_network_tx(stats)),
            "block_read": self._format_bytes(self._get_block_read(stats)),
            "block_write": self._format_bytes(self._get_block_write(stats)),
        }

    def _calculate_cpu_percent(self, stats: dict) -> str:
        """Calculate CPU usage percentage."""
        try:
            cpu_delta = (
                stats["cpu_stats"]["cpu_usage"]["total_usage"]
                - stats["precpu_stats"]["cpu_usage"]["total_usage"]
            )
            system_delta = (
                stats["cpu_stats"]["system_cpu_usage"] - stats["precpu_stats"]["system_cpu_usage"]
            )
            cpu_count = stats["cpu_stats"]["online_cpus"]

            if system_delta > 0 and cpu_delta > 0:
                cpu_percent = (cpu_delta / system_delta) * cpu_count * 100.0
                return f"{cpu_percent:.2f}%"
        except (KeyError, ZeroDivisionError):
            pass
        return "0.00%"

    def _format_memory(self, stats: dict) -> str:
        """Format memory usage."""
        try:
            used = stats["memory_stats"]["usage"]
            limit = stats["memory_stats"]["limit"]
            return f"{self._format_bytes(used)} / {self._format_bytes(limit)}"
        except KeyError:
            return "N/A"

    def _calculate_memory_percent(self, stats: dict) -> str:
        """Calculate memory usage percentage."""
        try:
            used = stats["memory_stats"]["usage"]
            limit = stats["memory_stats"]["limit"]
            percent = (used / limit) * 100.0
            return f"{percent:.2f}%"
        except (KeyError, ZeroDivisionError):
            return "0.00%"

    def _get_network_rx(self, stats: dict) -> int:
        """Get network bytes received."""
        try:
            return sum(iface["rx_bytes"] for iface in stats["networks"].values())
        except (KeyError, AttributeError):
            return 0

    def _get_network_tx(self, stats: dict) -> int:
        """Get network bytes transmitted."""
        try:
            return sum(iface["tx_bytes"] for iface in stats["networks"].values())
        except (KeyError, AttributeError):
            return 0

    def _get_block_read(self, stats: dict) -> int:
        """Get disk bytes read."""
        try:
            return sum(
                stat["value"]
                for stat in stats["blkio_stats"]["io_service_bytes_recursive"]
                if stat["op"] == "read"
            )
        except (KeyError, AttributeError, TypeError):
            return 0

    def _get_block_write(self, stats: dict) -> int:
        """Get disk bytes written."""
        try:
            return sum(
                stat["value"]
                for stat in stats["blkio_stats"]["io_service_bytes_recursive"]
                if stat["op"] == "write"
            )
        except (KeyError, AttributeError, TypeError):
            return 0

    def _format_bytes(self, bytes_val: int) -> str:
        """Format bytes to human-readable string."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if bytes_val < 1024.0:
                return f"{bytes_val:.2f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.2f} PB"


async def cleanup_on_exit(session: Any) -> None:
    """Ensure session is cleaned up on process exit.

    Args:
        session: SandboxSession to clean up
    """
    import atexit
    import signal
    import sys

    def sync_cleanup():
        """Synchronous cleanup for atexit."""
        try:
            # Run async cleanup in new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(session.cleanup())
            loop.close()
        except Exception as e:
            print(f"[sandbox] Warning: Cleanup failed: {e}")

    def signal_handler(signum, frame):
        """Handle termination signals."""
        print(f"\n[sandbox] Received signal {signum}, cleaning up...")
        sync_cleanup()
        sys.exit(0)

    # Register cleanup handlers
    atexit.register(sync_cleanup)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
