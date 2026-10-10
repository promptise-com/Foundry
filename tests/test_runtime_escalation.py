"""Tests for promptise.runtime.escalation — escalate() and EscalationTarget."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from promptise.runtime.config import EscalationTarget
from promptise.runtime.escalation import _fire_webhook, escalate


class TestEscalationTarget:
    def test_creation_with_webhook(self) -> None:
        target = EscalationTarget(webhook_url="https://hooks.example.com/")
        assert target.webhook_url == "https://hooks.example.com/"
        assert target.event_type is None

    def test_creation_with_event(self) -> None:
        target = EscalationTarget(event_type="budget.violation")
        assert target.webhook_url is None
        assert target.event_type == "budget.violation"

    def test_creation_with_both(self) -> None:
        target = EscalationTarget(
            webhook_url="https://example.com",
            event_type="alert.fired",
        )
        assert target.webhook_url is not None
        assert target.event_type is not None

    def test_creation_with_neither(self) -> None:
        target = EscalationTarget()
        assert target.webhook_url is None
        assert target.event_type is None


class TestEscalate:
    @pytest.mark.asyncio
    async def test_fires_webhook_when_url_set(self) -> None:
        target = EscalationTarget(webhook_url="https://hooks.example.com/")
        with patch(
            "promptise.runtime.escalation._fire_webhook", new_callable=AsyncMock
        ) as mock_fire:
            await escalate(target, {"reason": "test"})
            mock_fire.assert_awaited_once_with("https://hooks.example.com/", {"reason": "test"})

    @pytest.mark.asyncio
    async def test_skips_webhook_when_url_none(self) -> None:
        target = EscalationTarget(event_type="alert")
        with patch(
            "promptise.runtime.escalation._fire_webhook", new_callable=AsyncMock
        ) as mock_fire:
            await escalate(target, {"reason": "test"})
            mock_fire.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_emits_event_when_event_bus_provided(self) -> None:
        target = EscalationTarget(event_type="budget.exceeded")
        event_bus = AsyncMock()
        event_bus.emit = AsyncMock()
        await escalate(target, {"limit": "tool_calls"}, event_bus=event_bus)
        event_bus.emit.assert_awaited_once_with("budget.exceeded", {"limit": "tool_calls"})

    @pytest.mark.asyncio
    async def test_skips_event_when_no_event_bus(self) -> None:
        target = EscalationTarget(event_type="budget.exceeded")
        # Should not raise even without event bus
        await escalate(target, {"reason": "test"})

    @pytest.mark.asyncio
    async def test_event_bus_error_logged_not_raised(self) -> None:
        target = EscalationTarget(event_type="alert")
        event_bus = AsyncMock()
        event_bus.emit = AsyncMock(side_effect=RuntimeError("bus error"))
        # Should not raise
        await escalate(target, {"reason": "test"}, event_bus=event_bus)

    @pytest.mark.asyncio
    async def test_fires_both_webhook_and_event(self) -> None:
        target = EscalationTarget(
            webhook_url="https://example.com",
            event_type="alert",
        )
        event_bus = AsyncMock()
        event_bus.emit = AsyncMock()
        with patch(
            "promptise.runtime.escalation._fire_webhook", new_callable=AsyncMock
        ) as mock_fire:
            await escalate(target, {"data": 1}, event_bus=event_bus)
            mock_fire.assert_awaited_once()
            event_bus.emit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_target_is_noop(self) -> None:
        target = EscalationTarget()
        # Should complete without error
        await escalate(target, {"nothing": True})


class TestFireWebhook:
    @pytest.mark.asyncio
    async def test_posts_json_payload(self, monkeypatch) -> None:
        import socket

        # The host is resolved and checked before the POST (DNS rebinding
        # protection); answer with a public address instead of real DNS.
        monkeypatch.setattr(
            socket,
            "getaddrinfo",
            lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
        )
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_cls.return_value = mock_client

            await _fire_webhook("https://hooks.example.com/", {"key": "value"})

            mock_client.post.assert_awaited_once()
            call_kwargs = mock_client.post.call_args
            assert call_kwargs[1]["json"] == {"key": "value"}
            assert call_kwargs[0][0] == "https://93.184.216.34/"
            assert call_kwargs[1]["headers"]["Host"] == "hooks.example.com"

    @pytest.mark.asyncio
    async def test_httpx_not_installed_degrades_gracefully(self) -> None:
        """When httpx is not installed, webhook logs warning and returns (no crash)."""
        with patch.dict("sys.modules", {"httpx": None}):
            # Should NOT raise — graceful degradation
            await _fire_webhook("https://example.com", {})

    @pytest.mark.asyncio
    async def test_delivery_failure_logged_not_raised(self) -> None:
        with patch("httpx.AsyncClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.post = AsyncMock(side_effect=Exception("connection refused"))
            mock_cls.return_value = mock_client

            # Should not raise
            await _fire_webhook("https://unreachable.example.com/", {"test": True})
