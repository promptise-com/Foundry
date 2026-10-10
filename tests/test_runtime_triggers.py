"""Tests for runtime triggers: base, cron, event, message, registry."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptise.runtime.config import TriggerConfig
from promptise.runtime.exceptions import TriggerError
from promptise.runtime.triggers import (
    create_trigger,
    register_trigger_type,
    registered_trigger_types,
    unregister_trigger_type,
)
from promptise.runtime.triggers.base import BaseTrigger, TriggerEvent
from promptise.runtime.triggers.cron import CronTrigger
from promptise.runtime.triggers.event import EventTrigger, MessageTrigger

# =========================================================================
# TriggerEvent
# =========================================================================


class TestTriggerEvent:
    """Verify TriggerEvent creation and serialization."""

    def test_create(self) -> None:
        event = TriggerEvent(
            trigger_id="t-1",
            trigger_type="cron",
            payload={"scheduled_time": "2026-03-01T10:00:00"},
        )
        assert event.trigger_id == "t-1"
        assert event.trigger_type == "cron"
        assert event.event_id  # auto-generated UUID
        assert event.timestamp.tzinfo is not None

    def test_to_dict_from_dict_roundtrip(self) -> None:
        event = TriggerEvent(
            trigger_id="t-1",
            trigger_type="webhook",
            payload={"body": {"action": "push"}},
            metadata={"source": "github"},
        )
        data = event.to_dict()
        restored = TriggerEvent.from_dict(data)
        assert restored.trigger_id == event.trigger_id
        assert restored.trigger_type == event.trigger_type
        assert restored.payload == event.payload
        assert restored.metadata == event.metadata

    def test_defaults(self) -> None:
        event = TriggerEvent(trigger_id="t-1")
        assert event.trigger_type == ""
        assert event.payload == {}
        assert event.metadata == {}


# =========================================================================
# CronTrigger
# =========================================================================


class TestCronTrigger:
    """Verify cron trigger behavior."""

    def test_create(self) -> None:
        trigger = CronTrigger("*/5 * * * *")
        assert trigger.trigger_id.startswith("cron-")
        assert repr(trigger).startswith("CronTrigger(")

    def test_custom_id(self) -> None:
        trigger = CronTrigger("*/5 * * * *", trigger_id="my-cron")
        assert trigger.trigger_id == "my-cron"

    async def test_start_stop(self) -> None:
        trigger = CronTrigger("*/5 * * * *")
        await trigger.start()
        assert trigger._running is True
        await trigger.stop()
        assert trigger._running is False

    async def test_wait_for_next_produces_event(self) -> None:
        trigger = CronTrigger("* * * * *")  # Every minute
        await trigger.start()

        # Patch sleep to return immediately
        with patch("promptise.runtime.triggers.cron.asyncio.sleep", new_callable=AsyncMock):
            event = await trigger.wait_for_next()
            assert event.trigger_type == "cron"
            assert "scheduled_time" in event.payload
            assert "cron_expression" in event.payload

        await trigger.stop()

    async def test_stop_unblocks_waiter(self) -> None:
        trigger = CronTrigger("0 0 1 1 *")  # Very far in future
        await trigger.start()

        async def stop_soon() -> None:
            await asyncio.sleep(0.05)
            await trigger.stop()

        asyncio.create_task(stop_soon())

        with pytest.raises(asyncio.CancelledError):
            await trigger.wait_for_next()

    def test_simple_next_fire_interval(self) -> None:
        trigger = CronTrigger("*/10 * * * *")
        now = datetime.now(timezone.utc)
        next_fire = trigger._simple_next_fire(now)
        # Should be ~10 minutes from now
        diff = (next_fire - now).total_seconds()
        assert 590 <= diff <= 610

    def test_simple_next_fire_every_minute(self) -> None:
        trigger = CronTrigger("* * * * *")
        now = datetime.now(timezone.utc)
        next_fire = trigger._simple_next_fire(now)
        diff = (next_fire - now).total_seconds()
        assert 50 <= diff <= 70

    def test_invalid_cron_raises(self) -> None:
        # Rejected when the trigger is created, not on the first wait.
        with pytest.raises(TriggerError, match="Invalid cron"):
            CronTrigger("bad")

    def test_simple_fallback_rejects_short_expression(self) -> None:
        with pytest.raises(TriggerError, match="Invalid cron"):
            CronTrigger._simple_next_fire_for("bad", datetime.now(timezone.utc))


# =========================================================================
# EventTrigger
# =========================================================================


class TestEventTrigger:
    """Verify EventBus trigger adapter."""

    def _make_bus(self) -> MagicMock:
        """Create a mock EventBus."""
        bus = AsyncMock()
        bus.subscribe = AsyncMock()
        bus.unsubscribe = AsyncMock()
        return bus

    async def test_start_subscribes(self) -> None:
        bus = self._make_bus()
        trigger = EventTrigger(bus, "task.completed")
        await trigger.start()
        bus.subscribe.assert_awaited_once_with("task.completed", trigger._handler)

    async def test_stop_unsubscribes(self) -> None:
        bus = self._make_bus()
        trigger = EventTrigger(bus, "task.completed")
        await trigger.start()
        await trigger.stop()
        bus.unsubscribe.assert_awaited_once()

    async def test_handler_produces_trigger_event(self) -> None:
        bus = self._make_bus()
        trigger = EventTrigger(bus, "task.completed")
        await trigger.start()

        # Simulate an event from the bus
        mock_event = MagicMock()
        mock_event.event_id = "ev-123"
        mock_event.source = "agent-1"
        mock_event.data = {"result": "ok"}

        await trigger._handler(mock_event)

        # Should be in the queue
        assert not trigger._queue.empty()
        event = await trigger.wait_for_next()
        assert event.trigger_type == "event"
        assert event.payload["event_type"] == "task.completed"
        assert event.payload["source"] == "agent-1"

    async def test_source_filter(self) -> None:
        bus = self._make_bus()
        trigger = EventTrigger(bus, "task.completed", source_filter="agent-1")
        await trigger.start()

        # Event from wrong source
        wrong = MagicMock()
        wrong.source = "agent-2"
        await trigger._handler(wrong)
        assert trigger._queue.empty()

        # Event from correct source
        right = MagicMock()
        right.source = "agent-1"
        right.event_id = "ev-1"
        right.data = {}
        await trigger._handler(right)
        assert not trigger._queue.empty()

    async def test_stop_unblocks_waiter(self) -> None:
        bus = self._make_bus()
        trigger = EventTrigger(bus, "task.completed")
        await trigger.start()

        async def stop_soon() -> None:
            await asyncio.sleep(0.05)
            await trigger.stop()

        asyncio.create_task(stop_soon())

        with pytest.raises(asyncio.CancelledError):
            await trigger.wait_for_next()

    def test_repr(self) -> None:
        bus = AsyncMock()
        trigger = EventTrigger(bus, "task.completed", trigger_id="my-evt")
        r = repr(trigger)
        assert "my-evt" in r
        assert "task.completed" in r


# =========================================================================
# MessageTrigger
# =========================================================================


class TestMessageTrigger:
    """Verify MessageBroker trigger adapter."""

    def _make_broker(self) -> MagicMock:
        broker = AsyncMock()
        broker.subscribe = AsyncMock(return_value="sub-123")
        broker.unsubscribe = AsyncMock()
        return broker

    async def test_start_subscribes(self) -> None:
        broker = self._make_broker()
        trigger = MessageTrigger(broker, "reports.*")
        await trigger.start()
        broker.subscribe.assert_awaited_once()

    async def test_handler_produces_event(self) -> None:
        broker = self._make_broker()
        trigger = MessageTrigger(broker, "reports.*")
        await trigger.start()

        msg = MagicMock()
        msg.message_id = "m-1"
        msg.sender = "agent-1"
        msg.content = "report ready"

        await trigger._handler(msg)
        event = await trigger.wait_for_next()
        assert event.trigger_type == "message"
        assert event.payload["topic"] == "reports.*"

    async def test_stop_unsubscribes(self) -> None:
        broker = self._make_broker()
        trigger = MessageTrigger(broker, "reports.*")
        await trigger.start()
        await trigger.stop()
        broker.unsubscribe.assert_awaited_once_with("sub-123")

    def test_repr(self) -> None:
        broker = AsyncMock()
        trigger = MessageTrigger(broker, "reports.*", trigger_id="my-msg")
        assert "reports.*" in repr(trigger)


# =========================================================================
# create_trigger factory
# =========================================================================


class TestCreateTrigger:
    """Verify the trigger factory function."""

    def test_cron(self) -> None:
        config = TriggerConfig(type="cron", cron_expression="*/5 * * * *")
        trigger = create_trigger(config)
        assert isinstance(trigger, CronTrigger)

    def test_event(self) -> None:
        bus = AsyncMock()
        config = TriggerConfig(type="event", event_type="task.completed")
        trigger = create_trigger(config, event_bus=bus)
        assert isinstance(trigger, EventTrigger)

    def test_event_without_bus_raises(self) -> None:
        config = TriggerConfig(type="event", event_type="task.completed")
        with pytest.raises(TriggerError, match="EventBus"):
            create_trigger(config)

    def test_message(self) -> None:
        broker = AsyncMock()
        config = TriggerConfig(type="message", topic="reports.*")
        trigger = create_trigger(config, broker=broker)
        assert isinstance(trigger, MessageTrigger)

    def test_message_without_broker_raises(self) -> None:
        config = TriggerConfig(type="message", topic="reports.*")
        with pytest.raises(TriggerError, match="MessageBroker"):
            create_trigger(config)


# =========================================================================
# BaseTrigger protocol
# =========================================================================


class TestBaseTriggerProtocol:
    """Verify that concrete triggers satisfy the protocol."""

    def test_cron_satisfies_protocol(self) -> None:
        trigger = CronTrigger("* * * * *")
        assert isinstance(trigger, BaseTrigger)

    def test_event_satisfies_protocol(self) -> None:
        bus = AsyncMock()
        trigger = EventTrigger(bus, "test")
        assert isinstance(trigger, BaseTrigger)

    def test_message_satisfies_protocol(self) -> None:
        broker = AsyncMock()
        trigger = MessageTrigger(broker, "test")
        assert isinstance(trigger, BaseTrigger)


# =========================================================================
# Trigger registry
# =========================================================================


class TestTriggerRegistry:
    """Verify the custom trigger registration system."""

    def test_builtin_types_registered(self) -> None:
        types = registered_trigger_types()
        for name in ("cron", "event", "message", "webhook", "file_watch"):
            assert name in types

    def test_register_custom_type(self) -> None:
        def my_factory(config, *, event_bus=None, broker=None):
            return CronTrigger("* * * * *")

        register_trigger_type("_test_custom", my_factory)
        try:
            assert "_test_custom" in registered_trigger_types()
        finally:
            unregister_trigger_type("_test_custom")

        assert "_test_custom" not in registered_trigger_types()

    def test_register_duplicate_raises(self) -> None:
        def factory(c, **kw):
            return CronTrigger("* * * * *")

        register_trigger_type("_test_dup", factory)
        try:
            with pytest.raises(ValueError, match="already registered"):
                register_trigger_type("_test_dup", factory)
        finally:
            unregister_trigger_type("_test_dup")

    def test_register_with_overwrite(self) -> None:
        def factory1(c, **kw):
            return CronTrigger("* * * * *")

        def factory2(c, **kw):
            return CronTrigger("*/5 * * * *")

        register_trigger_type("_test_ow", factory1)
        try:
            # Should not raise with overwrite=True
            register_trigger_type("_test_ow", factory2, overwrite=True)
        finally:
            unregister_trigger_type("_test_ow")

    def test_create_trigger_with_custom_type(self) -> None:
        def my_factory(config, *, event_bus=None, broker=None):
            expr = config.custom_config.get("expression", "* * * * *")
            return CronTrigger(expr)

        register_trigger_type("_test_custom_create", my_factory)
        try:
            config = TriggerConfig(
                type="_test_custom_create",
                custom_config={"expression": "*/2 * * * *"},
            )
            trigger = create_trigger(config)
            assert isinstance(trigger, CronTrigger)
        finally:
            unregister_trigger_type("_test_custom_create")

    def test_unknown_type_raises(self) -> None:
        config = TriggerConfig(type="_nonexistent_test_type")
        with pytest.raises(TriggerError, match="Unknown trigger type"):
            create_trigger(config)

    def test_unregister_unknown_is_noop(self) -> None:
        # Should not raise
        unregister_trigger_type("_definitely_not_registered")

    def test_trigger_config_custom_config_field(self) -> None:
        config = TriggerConfig(
            type="cron",
            cron_expression="* * * * *",
            custom_config={"key": "value"},
        )
        assert config.custom_config == {"key": "value"}

    def test_trigger_config_accepts_str_type(self) -> None:
        # TriggerConfig.type is now str, not Literal
        config = TriggerConfig(type="any_custom_name")
        assert config.type == "any_custom_name"
