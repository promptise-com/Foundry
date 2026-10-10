"""Tests for sandbox/utils.py — SandboxContainerManager and helpers.

All Docker interactions are mocked so these tests run without Docker installed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from promptise.sandbox.utils import SandboxContainerManager

# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------


def make_mock_container(
    container_id: str = "abc123def456",
    name: str = "promptise-sandbox-1",
    status: str = "exited",
    created: str | None = None,
    tags: list[str] | None = None,
) -> MagicMock:
    c = MagicMock()
    c.id = container_id
    c.name = name
    c.status = status
    c.image.tags = tags or ["python:3.12-slim"]
    c.image.id = "sha256:abc"
    c.attrs = {"Created": created or "2026-01-01T00:00:00Z"}
    c.labels = {"promptise.sandbox": "true", "promptise.session": "sess-1"}
    return c


def _make_manager() -> SandboxContainerManager:
    """Return a SandboxContainerManager with a mocked Docker client."""
    from promptise.sandbox.utils import SandboxContainerManager

    with patch("promptise.sandbox.utils.docker") as mock_docker:
        mock_docker.from_env.return_value = MagicMock()
        manager = SandboxContainerManager.__new__(SandboxContainerManager)
        manager.client = MagicMock()
        manager.label_prefix = "promptise.sandbox"
    return manager


# ===========================================================================
# Initialisation
# ===========================================================================


class TestSandboxContainerManagerInit:
    def test_raises_when_docker_not_installed(self) -> None:
        with patch("promptise.sandbox.utils.docker", None):
            from promptise.sandbox.utils import SandboxContainerManager

            with pytest.raises(RuntimeError, match=r"promptise\[sandbox\]"):
                SandboxContainerManager()

    def test_init_with_docker_available(self) -> None:
        with patch("promptise.sandbox.utils.docker") as mock_docker:
            mock_docker.from_env.return_value = MagicMock()
            from promptise.sandbox.utils import SandboxContainerManager

            manager = SandboxContainerManager()
            assert manager.label_prefix == "promptise.sandbox"
            mock_docker.from_env.assert_called_once()


# ===========================================================================
# list_containers
# ===========================================================================


class TestListContainers:
    def test_returns_empty_list_when_no_containers(self) -> None:
        manager = _make_manager()
        manager.client.containers.list.return_value = []
        result = manager.list_containers()
        assert result == []

    def test_returns_basic_info_for_stopped_container(self) -> None:
        manager = _make_manager()
        c = make_mock_container(
            container_id="abc123def456789",
            name="sandbox-1",
            status="exited",
            created="2026-01-15T10:00:00Z",
        )
        manager.client.containers.list.return_value = [c]

        result = manager.list_containers()

        assert len(result) == 1
        info = result[0]
        assert info["id"] == "abc123def456"  # First 12 chars
        assert info["full_id"] == "abc123def456789"
        assert info["name"] == "sandbox-1"
        assert info["status"] == "exited"
        assert "cpu_usage" not in info  # Only added for running containers

    def test_adds_cpu_and_memory_for_running_container(self) -> None:
        manager = _make_manager()
        c = make_mock_container(status="running")
        c.stats.return_value = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 200_000_000},
                "system_cpu_usage": 2_000_000_000,
                "online_cpus": 4,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 100_000_000},
                "system_cpu_usage": 1_000_000_000,
            },
            "memory_stats": {"usage": 50 * 1024 * 1024, "limit": 512 * 1024 * 1024},
        }
        manager.client.containers.list.return_value = [c]

        result = manager.list_containers()

        info = result[0]
        assert "cpu_usage" in info
        assert "memory_usage" in info

    def test_running_container_stat_error_falls_back_to_na(self) -> None:
        manager = _make_manager()
        c = make_mock_container(status="running")
        c.stats.side_effect = Exception("stats unavailable")
        manager.client.containers.list.return_value = [c]

        result = manager.list_containers()

        info = result[0]
        assert info["cpu_usage"] == "N/A"
        assert info["memory_usage"] == "N/A"

    def test_passes_all_flag_to_docker_client(self) -> None:
        manager = _make_manager()
        manager.client.containers.list.return_value = []
        manager.list_containers(all=True)
        call_kwargs = manager.client.containers.list.call_args[1]
        assert call_kwargs.get("all") is True

    def test_uses_no_image_tag_fallback(self) -> None:
        manager = _make_manager()
        c = make_mock_container()
        c.image.tags = []  # No tags
        c.image.id = "sha256:abcdef1234567890"
        manager.client.containers.list.return_value = [c]
        result = manager.list_containers()
        # Falls back to first 12 chars of image id
        assert result[0]["image"] == "sha256:abcde"


# ===========================================================================
# cleanup_all
# ===========================================================================


class TestCleanupAll:
    def test_removes_stopped_containers(self) -> None:
        manager = _make_manager()
        c = make_mock_container(status="exited")
        manager.client.containers.list.return_value = [c]

        count = manager.cleanup_all()

        assert count == 1
        c.remove.assert_called_once()

    def test_skips_running_containers_without_force(self) -> None:
        manager = _make_manager()
        c = make_mock_container(status="running")
        manager.client.containers.list.return_value = [c]

        count = manager.cleanup_all(force=False)

        assert count == 0
        c.stop.assert_not_called()
        c.remove.assert_not_called()

    def test_force_removes_running_containers(self) -> None:
        manager = _make_manager()
        c = make_mock_container(status="running")
        manager.client.containers.list.return_value = [c]

        count = manager.cleanup_all(force=True)

        assert count == 1
        c.stop.assert_called_once_with(timeout=5)
        c.remove.assert_called_once_with(force=True)

    def test_remove_error_is_caught_not_raised(self) -> None:
        manager = _make_manager()
        c = make_mock_container(status="exited")
        c.remove.side_effect = Exception("permission denied")
        manager.client.containers.list.return_value = [c]

        # Should not raise
        count = manager.cleanup_all()
        assert count == 0

    def test_multiple_containers_all_removed(self) -> None:
        manager = _make_manager()
        containers = [make_mock_container(container_id=f"id{i}", status="exited") for i in range(5)]
        manager.client.containers.list.return_value = containers

        count = manager.cleanup_all()

        assert count == 5


# ===========================================================================
# cleanup_old
# ===========================================================================


class TestCleanupOld:
    def test_removes_containers_older_than_cutoff(self) -> None:
        manager = _make_manager()
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
        c = make_mock_container(status="exited", created=old_time)
        manager.client.containers.list.return_value = [c]

        count = manager.cleanup_old(max_age_hours=24)

        assert count == 1
        c.remove.assert_called_once_with(force=True)

    def test_keeps_recent_containers(self) -> None:
        manager = _make_manager()
        recent_time = (datetime.now(timezone.utc) - timedelta(hours=1)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        c = make_mock_container(status="exited", created=recent_time)
        manager.client.containers.list.return_value = [c]

        count = manager.cleanup_old(max_age_hours=24)

        assert count == 0
        c.remove.assert_not_called()

    def test_stops_running_old_containers_before_removal(self) -> None:
        manager = _make_manager()
        old_time = (datetime.now(timezone.utc) - timedelta(hours=100)).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        c = make_mock_container(status="running", created=old_time)
        manager.client.containers.list.return_value = [c]

        count = manager.cleanup_old(max_age_hours=24)

        assert count == 1
        c.stop.assert_called_once_with(timeout=5)
        c.remove.assert_called_once_with(force=True)

    def test_removal_error_is_caught(self) -> None:
        manager = _make_manager()
        old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%SZ")
        c = make_mock_container(status="exited", created=old_time)
        c.remove.side_effect = Exception("error")
        manager.client.containers.list.return_value = [c]

        count = manager.cleanup_old(max_age_hours=24)

        assert count == 0


# ===========================================================================
# Private helpers
# ===========================================================================


class TestPrivateHelpers:
    """Tests for internal helper methods."""

    def setup_method(self) -> None:
        self.manager = _make_manager()

    def test_calculate_cpu_percent_normal(self) -> None:
        stats = {
            "cpu_stats": {
                "cpu_usage": {"total_usage": 200_000_000},
                "system_cpu_usage": 2_000_000_000,
                "online_cpus": 4,
            },
            "precpu_stats": {
                "cpu_usage": {"total_usage": 100_000_000},
                "system_cpu_usage": 1_000_000_000,
            },
        }
        result = self.manager._calculate_cpu_percent(stats)
        assert result.endswith("%")
        assert float(result[:-1]) > 0

    def test_calculate_cpu_percent_missing_keys(self) -> None:
        result = self.manager._calculate_cpu_percent({})
        assert result == "0.00%"

    def test_format_memory_normal(self) -> None:
        stats = {
            "memory_stats": {
                "usage": 100 * 1024 * 1024,  # 100 MB
                "limit": 512 * 1024 * 1024,  # 512 MB
            }
        }
        result = self.manager._format_memory(stats)
        assert "MB" in result

    def test_format_memory_missing_keys(self) -> None:
        result = self.manager._format_memory({})
        assert result == "N/A"

    def test_calculate_memory_percent(self) -> None:
        stats = {
            "memory_stats": {
                "usage": 256 * 1024 * 1024,
                "limit": 512 * 1024 * 1024,
            }
        }
        result = self.manager._calculate_memory_percent(stats)
        assert result == "50.00%"

    def test_calculate_memory_percent_missing_keys(self) -> None:
        result = self.manager._calculate_memory_percent({})
        assert result == "0.00%"

    def test_get_network_rx(self) -> None:
        stats = {
            "networks": {
                "eth0": {"rx_bytes": 1000, "tx_bytes": 500},
                "eth1": {"rx_bytes": 2000, "tx_bytes": 1000},
            }
        }
        assert self.manager._get_network_rx(stats) == 3000

    def test_get_network_rx_missing(self) -> None:
        assert self.manager._get_network_rx({}) == 0

    def test_get_network_tx(self) -> None:
        stats = {
            "networks": {
                "eth0": {"rx_bytes": 1000, "tx_bytes": 500},
            }
        }
        assert self.manager._get_network_tx(stats) == 500

    def test_get_block_read(self) -> None:
        stats = {
            "blkio_stats": {
                "io_service_bytes_recursive": [
                    {"op": "read", "value": 4096},
                    {"op": "write", "value": 2048},
                    {"op": "read", "value": 8192},
                ]
            }
        }
        assert self.manager._get_block_read(stats) == 12288

    def test_get_block_write(self) -> None:
        stats = {
            "blkio_stats": {
                "io_service_bytes_recursive": [
                    {"op": "read", "value": 4096},
                    {"op": "write", "value": 2048},
                ]
            }
        }
        assert self.manager._get_block_write(stats) == 2048

    def test_get_block_read_missing(self) -> None:
        assert self.manager._get_block_read({}) == 0

    def test_format_bytes_bytes(self) -> None:
        assert self.manager._format_bytes(512) == "512.00 B"

    def test_format_bytes_kilobytes(self) -> None:
        result = self.manager._format_bytes(2048)
        assert "KB" in result

    def test_format_bytes_megabytes(self) -> None:
        result = self.manager._format_bytes(5 * 1024 * 1024)
        assert "MB" in result

    def test_format_bytes_gigabytes(self) -> None:
        result = self.manager._format_bytes(3 * 1024 * 1024 * 1024)
        assert "GB" in result
