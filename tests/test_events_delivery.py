"""Tests for event delivery: per-sink isolation, shutdown drain, signing, SSRF."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import socket
import threading
import time

import pytest
from aiohttp import web
from multidict import CIMultiDict

from promptise import verify_event_signature
from promptise.events import AgentEvent, CallbackSink, EventNotifier, WebhookSink

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Receiver:
    """A local aiohttp webhook receiver (127.0.0.1, port 0) that records requests.

    Answers ``fail_first`` requests with 503, the rest with 204.
    """

    def __init__(self, fail_first: int = 0) -> None:
        self.requests: list[tuple[bytes, CIMultiDict[str]]] = []
        self.fail_first = fail_first
        self.port = 0
        self.url = ""
        self._runner: web.AppRunner | None = None

    async def _handle(self, request: web.Request) -> web.Response:
        raw = await request.read()
        self.requests.append((raw, request.headers.copy()))
        return web.Response(status=503 if len(self.requests) <= self.fail_first else 204)

    async def __aenter__(self) -> _Receiver:
        app = web.Application()
        app.router.add_route("POST", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        server = site._server
        assert server is not None
        self.port = server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self.url = f"http://127.0.0.1:{self.port}/hook"
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._runner is not None:
            await self._runner.cleanup()


class _SlowSink:
    """A sink that takes ``delay`` seconds per event."""

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.received: list[AgentEvent] = []
        self.closed = False

    async def emit(self, event: AgentEvent) -> None:
        await asyncio.sleep(self.delay)
        self.received.append(event)

    async def close(self) -> None:
        self.closed = True


def _event(event_type: str = "test", **kwargs: object) -> AgentEvent:
    return AgentEvent(event_type=event_type, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Sink isolation
# ---------------------------------------------------------------------------


class TestSinkIsolation:
    @pytest.mark.asyncio
    async def test_slow_sink_does_not_delay_other_sinks(self):
        slow = _SlowSink(delay=2.0)
        fast: list[tuple[float, str]] = []
        start = time.monotonic()
        notifier = EventNotifier(
            sinks=[
                slow,
                CallbackSink(lambda e: fast.append((time.monotonic() - start, e.event_type))),
            ],
            shutdown_timeout=0.1,
        )
        await notifier.start()
        await notifier.emit(_event("a"))
        await notifier.emit(_event("b"))
        await asyncio.sleep(0.2)
        assert [t for _, t in fast] == ["a", "b"]
        assert all(elapsed < 0.5 for elapsed, _ in fast)
        await notifier.stop()

    @pytest.mark.asyncio
    async def test_retrying_webhook_does_not_delay_callback(self):
        """A webhook whose receiver is down must not hold up other sinks."""
        delivered: list[str] = []
        async with _Receiver(fail_first=100) as receiver:
            notifier = EventNotifier(
                sinks=[
                    WebhookSink(receiver.url, retry_delay=0.5, allow_private_networks=True),
                    CallbackSink(lambda e: delivered.append(e.event_type)),
                ],
                shutdown_timeout=0.2,
            )
            await notifier.start()
            await notifier.emit(_event("tool.error"))
            await notifier.emit(_event("invocation.complete"))
            await asyncio.sleep(0.3)
            assert delivered == ["tool.error", "invocation.complete"]
            await notifier.stop()

    @pytest.mark.asyncio
    async def test_order_is_kept_per_sink(self):
        received: list[str] = []
        notifier = EventNotifier(sinks=[CallbackSink(lambda e: received.append(e.event_type))])
        for i in range(20):
            await notifier.emit(_event(f"e{i}"))
        await notifier.stop()
        assert received == [f"e{i}" for i in range(20)]

    @pytest.mark.asyncio
    async def test_full_queue_drops_only_for_that_sink(self):
        slow = _SlowSink(delay=0.2)
        fast: list[AgentEvent] = []
        notifier = EventNotifier(sinks=[slow, CallbackSink(fast.append)], max_queue_size=1)
        await notifier.start()
        for i in range(3):
            notifier.emit_sync(_event(f"e{i}"))
            await asyncio.sleep(0.02)
        # slow: e0 in flight, e1 queued, e2 dropped; fast kept up with all three
        assert [e.event_type for e in fast] == ["e0", "e1", "e2"]
        assert notifier.dropped_count == 1
        await notifier.stop()
        assert [e.event_type for e in slow.received] == ["e0", "e1"]


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------


class TestShutdown:
    @pytest.mark.asyncio
    async def test_stop_waits_for_slow_sink_within_timeout(self):
        slow = _SlowSink(delay=0.3)
        notifier = EventNotifier(sinks=[slow], shutdown_timeout=5.0)
        await notifier.emit(_event("a"))
        await notifier.emit(_event("b"))
        await notifier.stop()
        assert [e.event_type for e in slow.received] == ["a", "b"]
        assert notifier.dropped_count == 0
        assert slow.closed  # sinks with close() are closed on stop

    @pytest.mark.asyncio
    async def test_stop_logs_and_counts_dropped_events(self, caplog):
        slow = _SlowSink(delay=10.0)
        notifier = EventNotifier(sinks=[slow], shutdown_timeout=0.2)
        await notifier.emit(_event("first"))
        await notifier.emit(_event("second"))
        started = time.monotonic()
        with caplog.at_level(logging.WARNING, logger="promptise.events"):
            await notifier.stop()
        assert time.monotonic() - started < 2.0
        assert notifier.dropped_count == 2
        message = " ".join(r.getMessage() for r in caplog.records)
        assert "_SlowSink" in message
        assert "dropped 2 event(s): first, second" in message
        assert slow.received == []

    @pytest.mark.asyncio
    async def test_stop_timeout_override(self):
        slow = _SlowSink(delay=0.3)
        notifier = EventNotifier(sinks=[slow], shutdown_timeout=0.01)
        await notifier.emit(_event("a"))
        await notifier.stop(timeout=5.0)
        assert len(slow.received) == 1
        assert notifier.dropped_count == 0

    def test_negative_timeout_rejected(self):
        with pytest.raises(ValueError):
            EventNotifier(sinks=[CallbackSink(print)], shutdown_timeout=-1)

    @pytest.mark.asyncio
    async def test_flush(self):
        slow = _SlowSink(delay=0.2)
        notifier = EventNotifier(sinks=[slow])
        await notifier.emit(_event("a"))
        assert await notifier.flush(timeout=0.01) is False
        assert await notifier.flush(timeout=5.0) is True
        assert len(slow.received) == 1
        assert notifier.is_running
        await notifier.stop()
        assert not notifier.is_running

    @pytest.mark.asyncio
    async def test_restart_after_stop(self):
        received: list[str] = []
        notifier = EventNotifier(sinks=[CallbackSink(lambda e: received.append(e.event_type))])
        await notifier.emit(_event("one"))
        await notifier.stop()
        await notifier.emit(_event("two"))
        await notifier.stop()
        assert received == ["one", "two"]


# ---------------------------------------------------------------------------
# emit_sync
# ---------------------------------------------------------------------------


class TestEmitSync:
    @pytest.mark.asyncio
    async def test_emit_sync_starts_notifier_in_running_loop(self):
        received: list[str] = []
        notifier = EventNotifier(sinks=[CallbackSink(lambda e: received.append(e.event_type))])
        notifier.emit_sync(_event("early"))
        assert notifier.is_running
        await notifier.stop()
        assert received == ["early"]

    @pytest.mark.asyncio
    async def test_emit_sync_from_another_thread(self):
        received: list[str] = []
        notifier = EventNotifier(sinks=[CallbackSink(lambda e: received.append(e.event_type))])
        await notifier.start()
        thread = threading.Thread(target=notifier.emit_sync, args=(_event("from-thread"),))
        thread.start()
        thread.join()
        await asyncio.sleep(0.05)
        await notifier.stop()
        assert received == ["from-thread"]

    def test_queued_without_loop_then_delivered(self):
        received: list[str] = []
        notifier = EventNotifier(sinks=[CallbackSink(lambda e: received.append(e.event_type))])
        notifier.emit_sync(_event("queued"))  # no running loop: just queued

        async def run() -> None:
            await notifier.start()
            await notifier.stop()

        asyncio.run(run())
        assert received == ["queued"]

    def test_notifier_survives_a_closed_loop(self):
        """A notifier left running by one asyncio.run() works in the next."""
        received: list[str] = []
        notifier = EventNotifier(sinks=[CallbackSink(lambda e: received.append(e.event_type))])

        async def first() -> None:
            await notifier.emit(_event("one"))
            await notifier.flush(timeout=1)

        async def second() -> None:
            await notifier.emit(_event("two"))
            await notifier.stop()

        asyncio.run(first())
        asyncio.run(second())
        assert received == ["one", "two"]


# ---------------------------------------------------------------------------
# Webhook signing
# ---------------------------------------------------------------------------


SECRET = "whsec-test"


class TestWebhookSigning:
    @pytest.mark.asyncio
    async def test_signature_verifies_against_raw_body(self):
        async with _Receiver() as receiver:
            sink = WebhookSink(receiver.url, secret=SECRET, allow_private_networks=True)
            await sink.emit(_event("tool.error", severity="error", data={"tool_name": "x"}))
            await sink.close()
        raw, headers = receiver.requests[0]
        signature = headers["X-Promptise-Signature"]
        assert signature.startswith("t=") and ",v1=" in signature
        timestamp = int(signature.split(",")[0][2:])
        assert headers["X-Promptise-Timestamp"] == str(timestamp)
        assert headers["X-Promptise-Event"] == "tool.error"
        assert abs(time.time() - timestamp) < 60
        # The documented recipe, by hand: HMAC over b"<t>." + raw body
        expected = hmac.new(
            SECRET.encode(), f"{timestamp}.".encode() + raw, hashlib.sha256
        ).hexdigest()
        assert signature.endswith(f"v1={expected}")
        assert verify_event_signature(raw, signature, SECRET)
        assert json.loads(raw)["event_type"] == "tool.error"

    @pytest.mark.asyncio
    async def test_signature_covers_transformed_body(self):
        async with _Receiver() as receiver:
            sink = WebhookSink(
                receiver.url,
                secret=SECRET,
                transform=lambda p: {"text": p["event_type"]},
                allow_private_networks=True,
            )
            await sink.emit(_event("budget.exceeded"))
            await sink.close()
        raw, headers = receiver.requests[0]
        assert json.loads(raw) == {"text": "budget.exceeded"}
        assert verify_event_signature(raw, headers["X-Promptise-Signature"], SECRET)

    @pytest.mark.asyncio
    async def test_retries_keep_delivery_id_and_stay_verifiable(self):
        async with _Receiver(fail_first=2) as receiver:
            sink = WebhookSink(
                receiver.url, secret=SECRET, retry_delay=0.01, allow_private_networks=True
            )
            await sink.emit(_event("tool.error"))
            await sink.close()
        assert len(receiver.requests) == 3
        ids = {h["X-Promptise-Delivery"] for _, h in receiver.requests}
        assert len(ids) == 1
        for raw, headers in receiver.requests:
            assert verify_event_signature(raw, headers["X-Promptise-Signature"], SECRET)

    def test_generated_secret_is_readable(self):
        sink = WebhookSink("https://hooks.example.com/x")
        assert len(sink.secret) == 64


class TestVerifyEventSignature:
    BODY = b'{"event_type":"tool.error"}'

    def _sign(self, body: bytes = BODY, secret: str = SECRET, timestamp: int | None = None) -> str:
        ts = int(time.time()) if timestamp is None else timestamp
        digest = hmac.new(secret.encode(), f"{ts}.".encode() + body, hashlib.sha256).hexdigest()
        return f"t={ts},v1={digest}"

    def test_valid(self):
        assert verify_event_signature(self.BODY, self._sign(), SECRET)

    def test_str_body(self):
        assert verify_event_signature(self.BODY.decode(), self._sign(), SECRET)

    def test_tampered_body(self):
        assert not verify_event_signature(self.BODY + b" ", self._sign(), SECRET)

    def test_wrong_secret(self):
        assert not verify_event_signature(self.BODY, self._sign(), "other")

    def test_stale_timestamp_rejected(self):
        old = self._sign(timestamp=int(time.time()) - 3600)
        assert not verify_event_signature(self.BODY, old, SECRET)
        assert verify_event_signature(self.BODY, old, SECRET, tolerance=None)
        assert verify_event_signature(self.BODY, old, SECRET, tolerance=7200)

    def test_future_timestamp_rejected(self):
        future = self._sign(timestamp=int(time.time()) + 3600)
        assert not verify_event_signature(self.BODY, future, SECRET)

    def test_now_override(self):
        sig = self._sign(timestamp=1_000)
        assert verify_event_signature(self.BODY, sig, SECRET, now=1_100)
        assert not verify_event_signature(self.BODY, sig, SECRET, now=2_000)

    def test_secret_rotation(self):
        sig = self._sign(secret="old")
        assert verify_event_signature(self.BODY, sig, ["new", "old"])
        assert not verify_event_signature(self.BODY, sig, ["new"])

    @pytest.mark.parametrize(
        "header",
        ["", None, "v1=abc", "t=123", "t=notanint,v1=abc", "garbage", "t=,v1="],
    )
    def test_malformed_headers(self, header):
        assert not verify_event_signature(self.BODY, header, SECRET)

    def test_legacy_bare_hex_rejected(self):
        legacy = hmac.new(SECRET.encode(), self.BODY, hashlib.sha256).hexdigest()
        assert not verify_event_signature(self.BODY, legacy, SECRET)


# ---------------------------------------------------------------------------
# SSRF protection
# ---------------------------------------------------------------------------


class TestWebhookPrivateNetworks:
    @pytest.mark.parametrize(
        "url",
        ["http://127.0.0.1:8390/hook", "http://localhost:8390/hook", "http://10.0.0.5/alerts"],
    )
    def test_private_refused_with_actionable_message(self, url):
        with pytest.raises(ValueError) as info:
            WebhookSink(url)
        message = str(info.value)
        assert "allow_private_networks=True" in message
        assert "base_url" not in message

    @pytest.mark.parametrize(
        "url",
        ["http://127.0.0.1:8390/hook", "http://localhost:8390/hook", "http://10.0.0.5/alerts"],
    )
    def test_private_allowed_with_flag(self, url):
        WebhookSink(url, allow_private_networks=True)

    @pytest.mark.parametrize("url", ["ftp://example.com/x", "not a url", "http:///nohost"])
    def test_invalid_urls_rejected_even_with_flag(self, url):
        with pytest.raises(ValueError):
            WebhookSink(url, allow_private_networks=True)

    def test_openapi_hint_unchanged(self):
        from promptise.mcp.server._openapi import _validate_url_not_private

        with pytest.raises(ValueError, match="base_url override"):
            _validate_url_not_private("http://localhost/x")

    @pytest.mark.parametrize(
        "url",
        [
            "http://100.100.100.200/latest/meta-data/",  # Alibaba Cloud metadata (CGNAT)
            "http://100.64.0.1/hook",  # shared address space
            "http://0.0.0.0:8390/hook",  # unspecified: reaches localhost
            "http://224.0.0.1/hook",  # multicast
            "http://[::ffff:169.254.169.254]/latest",  # IPv4-mapped metadata address
        ],
    )
    def test_non_public_addresses_refused(self, url):
        with pytest.raises(ValueError, match="allow_private_networks=True"):
            WebhookSink(url)


def _fake_dns(monkeypatch: pytest.MonkeyPatch, host: str, answers: list[str]) -> list[str]:
    """Resolve *host* to ``answers[i]`` on the i-th lookup (the last one repeats).

    Returns the list of answers handed out, one per lookup.
    """
    real = socket.getaddrinfo
    handed_out: list[str] = []

    def fake(name, port, *args, **kwargs):  # type: ignore[no-untyped-def]
        # anyio (under httpx) passes the IDNA-encoded name as bytes
        if (name.decode() if isinstance(name, bytes) else name) != host:
            return real(name, port, *args, **kwargs)
        ip = answers[min(len(handed_out), len(answers) - 1)]
        handed_out.append(ip)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    return handed_out


class TestWebhookDnsRebinding:
    @pytest.mark.asyncio
    async def test_host_rebound_to_loopback_after_construction_is_not_contacted(
        self, monkeypatch, caplog
    ):
        """The URL is checked at construction; DNS then points it at localhost."""
        async with _Receiver() as receiver:
            lookups = _fake_dns(monkeypatch, "hooks.rebind.test", ["93.184.216.34", "127.0.0.1"])
            sink = WebhookSink(f"http://hooks.rebind.test:{receiver.port}/hook", max_retries=0)
            assert lookups == ["93.184.216.34"]  # passed the construction-time check
            with caplog.at_level(logging.WARNING, logger="promptise.events"):
                await sink.emit(_event("tool.error"))
            await sink.close()
        assert receiver.requests == []
        assert "private/internal IP 127.0.0.1" in caplog.text
        assert "allow_private_networks=True" in caplog.text

    @pytest.mark.asyncio
    async def test_connects_to_the_checked_address_without_a_second_lookup(self, monkeypatch):
        """httpx must not resolve the name again after the check (rebinding window)."""
        import promptise.mcp.server._openapi as openapi

        # Treat 127.0.0.1 as public for this test, so the receiver is reachable.
        monkeypatch.setattr(openapi, "_is_private_ip", lambda ip: str(ip) != "127.0.0.1")
        async with _Receiver() as receiver:
            lookups = _fake_dns(monkeypatch, "hooks.pin.test", ["127.0.0.1"])
            sink = WebhookSink(
                f"http://hooks.pin.test:{receiver.port}/hook", secret=SECRET, max_retries=0
            )
            await sink.emit(_event("tool.error"))
            await sink.close()
        assert len(receiver.requests) == 1
        raw, headers = receiver.requests[0]
        assert headers["Host"] == f"hooks.pin.test:{receiver.port}"
        assert verify_event_signature(raw, headers["X-Promptise-Signature"], SECRET)
        assert len(lookups) == 2  # construction + the delivery-time check, nothing else

    @pytest.mark.asyncio
    async def test_allow_private_networks_skips_the_delivery_check(self):
        async with _Receiver() as receiver:
            sink = WebhookSink(
                f"http://localhost:{receiver.port}/hook", allow_private_networks=True
            )
            await sink.emit(_event("tool.error"))
            await sink.close()
        assert len(receiver.requests) == 1
        assert receiver.requests[0][1]["Host"] == f"localhost:{receiver.port}"


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class TestWebhookRedactionScope:
    def test_redacts_user_id_session_and_metadata(self):
        sink = WebhookSink("https://hooks.example.com/x")
        payload = _event(
            "invocation.complete",
            user_id="maya@shop.example",
            session_id="sess-for-dana@example.com",
            metadata={"note": "key sk-abcdefghijklmnopqrstuvwxyz"},
            data={"duration_ms": 12.5},
        ).to_dict()
        redacted = sink._redact_payload(payload)
        assert redacted["user_id"] == "[EMAIL]"
        assert "[EMAIL]" in redacted["session_id"]
        assert redacted["metadata"]["note"] == "key [API_KEY]"
        assert redacted["data"] == {"duration_ms": 12.5}
        assert redacted["event_type"] == "invocation.complete"
        assert redacted["timestamp"] == payload["timestamp"]

    @pytest.mark.parametrize(
        ("url", "expected"),
        [
            (
                "postgres://admin:s3cret@db.internal:5432/app",
                "postgres://[REDACTED]@db.internal:5432/app",
            ),
            ("redis://default:p:a:ss@cache:6379/0", "redis://[REDACTED]@cache:6379/0"),
            ("https://bot:tok3n@hooks.example.com/x", "https://[REDACTED]@hooks.example.com/x"),
        ],
    )
    def test_url_credentials_redact_user_and_password(self, url, expected):
        from promptise.events import default_pii_sanitizer

        redacted = default_pii_sanitizer({"error": f"could not connect to {url}"})
        assert redacted == {"error": f"could not connect to {expected}"}
        assert "admin" not in str(redacted) and "s3cret" not in str(redacted)

    def test_url_credentials_pattern_stays_within_one_value(self):
        from promptise.events import default_pii_sanitizer

        data = {"link": "see http://docs.example/page", "note": "x:y@z", "n": 1}
        assert default_pii_sanitizer(data) == data

    def test_no_redaction_when_disabled(self):
        sink = WebhookSink("https://hooks.example.com/x", redact_sensitive=False)
        payload = _event("x", user_id="maya@shop.example").to_dict()
        assert sink._redact_payload(payload)["user_id"] == "maya@shop.example"


# ---------------------------------------------------------------------------
# .superagent YAML
# ---------------------------------------------------------------------------


class TestSuperAgentEvents:
    def _kwargs(self, tmp_path, events_block: str) -> dict:
        from promptise.superagent import SuperAgentLoader

        path = tmp_path / "agent.superagent"
        path.write_text(
            'version: "1.0"\n'
            "agent:\n"
            '  model: "openai:gpt-5-mini"\n'
            "servers:\n"
            "  tools:\n"
            "    type: http\n"
            '    url: "https://mcp.internal"\n' + events_block
        )
        return SuperAgentLoader.from_file(path).to_agent_config().to_build_kwargs()

    def test_new_options(self, tmp_path):
        kwargs = self._kwargs(
            tmp_path,
            "events:\n"
            "  shutdown_timeout: 2.5\n"
            "  slow_tool_threshold: 1.0\n"
            "  sinks:\n"
            "    - type: webhook\n"
            "      url: http://127.0.0.1:8390/promptise\n"
            "      allow_private_networks: true\n",
        )
        notifier = kwargs["events"]
        assert notifier.shutdown_timeout == 2.5
        assert notifier.slow_tool_threshold == 1.0
        assert isinstance(notifier.sinks[0], WebhookSink)

    def test_defaults_refuse_localhost(self, tmp_path):
        with pytest.raises(ValueError, match="allow_private_networks"):
            self._kwargs(
                tmp_path,
                "events:\n  sinks:\n    - type: webhook\n      url: http://localhost:8390/x\n",
            )

    def test_defaults(self, tmp_path):
        kwargs = self._kwargs(tmp_path, "events:\n  sinks:\n    - type: log\n")
        assert kwargs["events"].shutdown_timeout == 10.0
        assert kwargs["events"].slow_tool_threshold == 5.0
