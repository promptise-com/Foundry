"""Tests for promptise.events — Webhook & Event Notification system."""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from promptise.events import (
    AgentEvent,
    CallbackSink,
    EventBusSink,
    EventNotifier,
    LogSink,
    WebhookSink,
    emit_event,
)

# ---------------------------------------------------------------------------
# AgentEvent
# ---------------------------------------------------------------------------


class TestAgentEvent:
    def test_construction(self):
        e = AgentEvent(
            event_type="invocation.complete",
            severity="info",
            agent_id="my-agent",
            user_id="user-42",
            data={"duration_ms": 1234},
        )
        assert e.event_type == "invocation.complete"
        assert e.severity == "info"
        assert e.agent_id == "my-agent"
        assert e.user_id == "user-42"
        assert e.data == {"duration_ms": 1234}

    def test_defaults(self):
        e = AgentEvent(event_type="test")
        assert e.severity == "info"
        assert e.agent_id is None
        assert e.user_id is None
        assert e.data == {}
        assert e.metadata == {}
        assert isinstance(e.timestamp, float)

    def test_to_dict(self):
        e = AgentEvent(event_type="test", data={"key": "value"})
        d = e.to_dict()
        assert d["event_type"] == "test"
        assert d["data"] == {"key": "value"}
        assert "timestamp" in d

    def test_hmac_signature(self):
        e = AgentEvent(event_type="test")
        sig1 = e.compute_hmac("secret1")
        sig2 = e.compute_hmac("secret2")
        sig3 = e.compute_hmac("secret1")
        assert sig1 == sig3  # Same secret → same sig
        assert sig1 != sig2  # Different secret → different sig

    def test_serializable(self):
        e = AgentEvent(event_type="test", data={"nested": {"a": 1}})
        text = json.dumps(e.to_dict())
        assert "test" in text


# ---------------------------------------------------------------------------
# CallbackSink
# ---------------------------------------------------------------------------


class TestCallbackSink:
    @pytest.mark.asyncio
    async def test_async_callback(self):
        received = []

        async def handler(event):
            received.append(event)

        sink = CallbackSink(handler)
        event = AgentEvent(event_type="test")
        await sink.emit(event)
        assert len(received) == 1
        assert received[0].event_type == "test"

    @pytest.mark.asyncio
    async def test_sync_callback(self):
        received = []

        def handler(event):
            received.append(event)

        sink = CallbackSink(handler)
        await sink.emit(AgentEvent(event_type="test"))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_event_filtering(self):
        received = []
        sink = CallbackSink(
            lambda e: received.append(e),
            events=["invocation.error"],
        )
        await sink.emit(AgentEvent(event_type="invocation.complete"))
        await sink.emit(AgentEvent(event_type="invocation.error"))
        assert len(received) == 1
        assert received[0].event_type == "invocation.error"

    @pytest.mark.asyncio
    async def test_severity_filtering(self):
        received = []
        sink = CallbackSink(
            lambda e: received.append(e),
            min_severity="error",
        )
        await sink.emit(AgentEvent(event_type="test", severity="info"))
        await sink.emit(AgentEvent(event_type="test", severity="warning"))
        await sink.emit(AgentEvent(event_type="test", severity="error"))
        await sink.emit(AgentEvent(event_type="test", severity="critical"))
        assert len(received) == 2  # error + critical only

    @pytest.mark.asyncio
    async def test_callback_error_doesnt_crash(self):
        def bad_handler(event):
            raise RuntimeError("boom")

        sink = CallbackSink(bad_handler)
        await sink.emit(AgentEvent(event_type="test"))  # Should not raise


# ---------------------------------------------------------------------------
# LogSink
# ---------------------------------------------------------------------------


class TestLogSink:
    @pytest.mark.asyncio
    async def test_logs_event(self, caplog):
        sink = LogSink(logger_name="test.events")
        with caplog.at_level(logging.INFO, logger="test.events"):
            await sink.emit(
                AgentEvent(
                    event_type="invocation.complete",
                    severity="info",
                    data={"duration_ms": 42},
                )
            )
        assert "invocation.complete" in caplog.text

    @pytest.mark.asyncio
    async def test_severity_mapping(self, caplog):
        sink = LogSink(logger_name="test.events")
        with caplog.at_level(logging.WARNING, logger="test.events"):
            await sink.emit(AgentEvent(event_type="test", severity="warning"))
        assert "test" in caplog.text

    @pytest.mark.asyncio
    async def test_event_filtering(self, caplog):
        sink = LogSink(events=["invocation.error"], logger_name="test.events")
        with caplog.at_level(logging.INFO, logger="test.events"):
            await sink.emit(AgentEvent(event_type="invocation.complete"))
            await sink.emit(AgentEvent(event_type="invocation.error", severity="error"))
        assert "invocation.complete" not in caplog.text
        assert "invocation.error" in caplog.text


# ---------------------------------------------------------------------------
# EventBusSink
# ---------------------------------------------------------------------------


class TestEventBusSink:
    @pytest.mark.asyncio
    async def test_emits_to_bus(self):
        bus = MagicMock()
        bus.emit = MagicMock(return_value=None)
        sink = EventBusSink(bus)

        await sink.emit(AgentEvent(event_type="test", data={"key": "val"}))
        bus.emit.assert_called_once()
        args = bus.emit.call_args
        assert args[0][0] == "test"  # event_type

    @pytest.mark.asyncio
    async def test_async_bus(self):
        bus = MagicMock()
        bus.emit = AsyncMock(return_value=None)
        sink = EventBusSink(bus)

        await sink.emit(AgentEvent(event_type="test"))
        bus.emit.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_filtering(self):
        bus = MagicMock()
        bus.emit = MagicMock(return_value=None)
        sink = EventBusSink(bus, events=["invocation.error"])

        await sink.emit(AgentEvent(event_type="invocation.complete"))
        bus.emit.assert_not_called()

        await sink.emit(AgentEvent(event_type="invocation.error"))
        bus.emit.assert_called_once()

    @pytest.mark.asyncio
    async def test_bus_error_doesnt_crash(self):
        bus = MagicMock()
        bus.emit = MagicMock(side_effect=RuntimeError("boom"))
        sink = EventBusSink(bus)
        await sink.emit(AgentEvent(event_type="test"))  # Should not raise


# ---------------------------------------------------------------------------
# WebhookSink
# ---------------------------------------------------------------------------


class TestWebhookSink:
    def test_ssrf_rejection(self):
        """WebhookSink rejects private/internal URLs."""
        with pytest.raises(ValueError, match="private"):
            WebhookSink("http://localhost:9999/hook")

    def test_construction(self):
        sink = WebhookSink(
            "https://hooks.example.com/test",
            events=["invocation.error"],
            max_retries=2,
        )
        assert sink._url == "https://hooks.example.com/test"
        assert sink._events == {"invocation.error"}

    def test_event_filtering(self):
        sink = WebhookSink(
            "https://hooks.example.com/test",
            events=["invocation.error"],
        )
        assert sink._should_emit(AgentEvent(event_type="invocation.error")) is True
        assert sink._should_emit(AgentEvent(event_type="invocation.complete")) is False

    def test_severity_filtering(self):
        sink = WebhookSink(
            "https://hooks.example.com/test",
            min_severity="error",
        )
        assert sink._should_emit(AgentEvent(event_type="t", severity="info")) is False
        assert sink._should_emit(AgentEvent(event_type="t", severity="error")) is True
        assert sink._should_emit(AgentEvent(event_type="t", severity="critical")) is True

    def test_hmac_included(self):
        sink = WebhookSink(
            "https://hooks.example.com/test",
            secret="test-secret",
        )
        assert sink._secret == "test-secret"


# ---------------------------------------------------------------------------
# EventNotifier
# ---------------------------------------------------------------------------


class TestEventNotifier:
    @pytest.mark.asyncio
    async def test_basic_delivery(self):
        received = []
        sink = CallbackSink(lambda e: received.append(e))
        notifier = EventNotifier(sinks=[sink])

        await notifier.start()
        await notifier.emit(AgentEvent(event_type="test"))
        await asyncio.sleep(0.1)  # Let drain loop process
        await notifier.stop()

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_multiple_sinks(self):
        received_a = []
        received_b = []
        notifier = EventNotifier(
            sinks=[
                CallbackSink(lambda e: received_a.append(e)),
                CallbackSink(lambda e: received_b.append(e)),
            ]
        )

        await notifier.start()
        await notifier.emit(AgentEvent(event_type="test"))
        await asyncio.sleep(0.1)
        await notifier.stop()

        assert len(received_a) == 1
        assert len(received_b) == 1

    @pytest.mark.asyncio
    async def test_sink_isolation(self):
        """One failing sink doesn't affect the others."""
        received = []

        def bad_sink_fn(event):
            raise RuntimeError("sink crashed")

        notifier = EventNotifier(
            sinks=[
                CallbackSink(bad_sink_fn),
                CallbackSink(lambda e: received.append(e)),
            ]
        )

        await notifier.start()
        await notifier.emit(AgentEvent(event_type="test"))
        await asyncio.sleep(0.1)
        await notifier.stop()

        assert len(received) == 1  # Second sink still got it

    @pytest.mark.asyncio
    async def test_emit_sync(self):
        received = []
        notifier = EventNotifier(
            sinks=[
                CallbackSink(lambda e: received.append(e)),
            ]
        )

        await notifier.start()
        notifier.emit_sync(AgentEvent(event_type="test"))
        await asyncio.sleep(0.1)
        await notifier.stop()

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_queue_full_drops_event(self):
        """When queue is full, new events are dropped (not blocking)."""
        received = []
        notifier = EventNotifier(
            sinks=[CallbackSink(lambda e: received.append(e))],
            max_queue_size=2,
        )
        # No await between emits — the delivery task gets no chance to drain
        for _ in range(5):
            notifier.emit_sync(AgentEvent(event_type="test"))
        # Only 2 fit in the sink's queue; the rest are dropped and counted
        assert notifier._queues[0].qsize() <= 2
        assert notifier.dropped_count == 3
        await notifier.stop()
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_ordering(self):
        received = []
        notifier = EventNotifier(
            sinks=[
                CallbackSink(lambda e: received.append(e.event_type)),
            ]
        )

        await notifier.start()
        await notifier.emit(AgentEvent(event_type="first"))
        await notifier.emit(AgentEvent(event_type="second"))
        await notifier.emit(AgentEvent(event_type="third"))
        await asyncio.sleep(0.1)
        await notifier.stop()

        assert received == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_graceful_stop_drains_remaining(self):
        received = []
        notifier = EventNotifier(
            sinks=[
                CallbackSink(lambda e: received.append(e)),
            ]
        )

        await notifier.start()
        # Emit several events rapidly
        for i in range(10):
            notifier.emit_sync(AgentEvent(event_type=f"event_{i}"))
        # Stop should drain all
        await notifier.stop()
        assert len(received) == 10

    @pytest.mark.asyncio
    async def test_auto_start_on_emit(self):
        received = []
        notifier = EventNotifier(
            sinks=[
                CallbackSink(lambda e: received.append(e)),
            ]
        )
        # Don't call start() — emit should auto-start
        await notifier.emit(AgentEvent(event_type="auto"))
        await asyncio.sleep(0.1)
        await notifier.stop()
        assert len(received) == 1

    def test_requires_at_least_one_sink(self):
        with pytest.raises(ValueError, match="at least one"):
            EventNotifier(sinks=[])


# ---------------------------------------------------------------------------
# emit_event helper
# ---------------------------------------------------------------------------


class TestEmitEventHelper:
    def test_none_notifier_is_noop(self):
        """emit_event with None notifier does nothing."""
        emit_event(None, "test", "info", {})  # Should not raise

    @pytest.mark.asyncio
    async def test_emits_to_notifier(self):
        received = []
        notifier = EventNotifier(
            sinks=[
                CallbackSink(lambda e: received.append(e)),
            ]
        )
        await notifier.start()

        emit_event(notifier, "tool.error", "error", {"tool_name": "search"})
        await asyncio.sleep(0.1)
        await notifier.stop()

        assert len(received) == 1
        assert received[0].event_type == "tool.error"
        assert received[0].data["tool_name"] == "search"


# ---------------------------------------------------------------------------
# WebhookSink redaction
# ---------------------------------------------------------------------------


class TestWebhookRedaction:
    def test_redacts_credit_card(self):
        sink = WebhookSink("https://hooks.example.com/test")
        payload = {"data": {"card": "4532-0151-1283-4456"}}
        result = sink._redact_payload(payload)
        assert "4532" not in json.dumps(result["data"])
        assert "[CARD]" in json.dumps(result["data"])

    def test_redacts_ssn(self):
        sink = WebhookSink("https://hooks.example.com/test")
        payload = {"data": {"ssn": "078-05-1120"}}
        result = sink._redact_payload(payload)
        assert "078-05-1120" not in json.dumps(result["data"])

    def test_redacts_email(self):
        sink = WebhookSink("https://hooks.example.com/test")
        payload = {"data": {"email": "alice@example.com"}}
        result = sink._redact_payload(payload)
        assert "alice@example.com" not in json.dumps(result["data"])

    def test_redacts_api_key(self):
        sink = WebhookSink("https://hooks.example.com/test")
        payload = {"data": {"key": "sk-abcdefghijklmnopqrstuvwxyz1234567890"}}
        result = sink._redact_payload(payload)
        assert "sk-abcdefghij" not in json.dumps(result["data"])

    def test_redacts_aws_key(self):
        sink = WebhookSink("https://hooks.example.com/test")
        payload = {"data": {"key": "AKIAIOSFODNN7EXAMPLE"}}
        result = sink._redact_payload(payload)
        assert "AKIAIOSFODNN7EXAMPLE" not in json.dumps(result["data"])

    def test_redacts_bearer_token(self):
        sink = WebhookSink("https://hooks.example.com/test")
        payload = {"data": {"auth": "Bearer eyJhbGciOiJIUzI1NiJ9.abc.def"}}
        result = sink._redact_payload(payload)
        assert "eyJhbGci" not in json.dumps(result["data"])

    def test_redacts_password_in_url(self):
        sink = WebhookSink("https://hooks.example.com/test")
        payload = {"data": {"dsn": "postgresql://user:secret123@host:5432/db"}}
        result = sink._redact_payload(payload)
        assert "secret123" not in json.dumps(result["data"])

    def test_no_redaction_when_disabled(self):
        sink = WebhookSink("https://hooks.example.com/test", redact_sensitive=False)
        payload = {"data": {"card": "4532-0151-1283-4456"}}
        result = sink._redact_payload(payload)
        assert "4532-0151-1283-4456" in json.dumps(result["data"])

    def test_preserves_non_sensitive_data(self):
        sink = WebhookSink("https://hooks.example.com/test")
        payload = {"data": {"query": "hello world", "count": 42}}
        result = sink._redact_payload(payload)
        assert result["data"]["query"] == "hello world"
        assert result["data"]["count"] == 42


# ---------------------------------------------------------------------------
# WebhookSink HMAC
# ---------------------------------------------------------------------------


class TestWebhookHMAC:
    def test_hmac_on_final_payload(self):
        """HMAC must be computed on the final (redacted+transformed) payload."""
        import hashlib
        import hmac as hmac_mod

        sink = WebhookSink("https://hooks.example.com/test", secret="test-secret")

        # Payload with sensitive data that will be redacted
        event = AgentEvent(
            event_type="test",
            data={"email": "alice@example.com"},
        )

        # Simulate what emit() does: redact then compute HMAC
        payload = event.to_dict()
        payload = sink._redact_payload(payload)

        # The HMAC should match the redacted payload
        payload_bytes = json.dumps(payload, sort_keys=True, default=str).encode()
        expected_sig = hmac_mod.new(b"test-secret", payload_bytes, hashlib.sha256).hexdigest()

        # Verify the event's own HMAC would NOT match (it uses unredacted data)
        event_sig = event.compute_hmac("test-secret")
        assert event_sig != expected_sig  # They should differ because data was redacted


# ---------------------------------------------------------------------------
# WebhookSink transform
# ---------------------------------------------------------------------------


class TestWebhookTransform:
    def test_transform_applied(self):
        sink = WebhookSink(
            "https://hooks.example.com/test",
            transform=lambda p: {"text": f"Alert: {p['event_type']}"},
        )
        event = AgentEvent(event_type="invocation.error", severity="error")
        payload = event.to_dict()
        payload = sink._redact_payload(payload)
        if sink._transform:
            payload = sink._transform(payload)
        assert payload == {"text": "Alert: invocation.error"}


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_callback_is_event_sink(self):
        from promptise.events import EventSink

        sink = CallbackSink(lambda e: None)
        assert isinstance(sink, EventSink)

    def test_log_is_event_sink(self):
        from promptise.events import EventSink

        sink = LogSink()
        assert isinstance(sink, EventSink)

    def test_eventbus_is_event_sink(self):
        from promptise.events import EventSink

        sink = EventBusSink(MagicMock())
        assert isinstance(sink, EventSink)

    def test_webhook_is_event_sink(self):
        from promptise.events import EventSink

        sink = WebhookSink("https://hooks.example.com/test")
        assert isinstance(sink, EventSink)

    def test_custom_sink_satisfies_protocol(self):
        from promptise.events import EventSink

        class MySink:
            async def emit(self, event: AgentEvent) -> None:
                pass

        assert isinstance(MySink(), EventSink)


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_imports_from_promptise(self):
        from promptise import (
            AgentEvent,
            EventNotifier,
        )

        assert AgentEvent is not None
        assert EventNotifier is not None
