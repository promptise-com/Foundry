"""DNS rebinding protection for outbound webhooks other than the event sink.

``WebhookApprovalHandler`` (agent-side gate and MCP server approval gate) and
the runtime's escalation webhook used to check their URL once, then let httpx
resolve the name again when connecting.  A DNS answer that changes in between
(DNS rebinding, a short TTL) sent the request to localhost or the cloud
metadata service.  Every request must now resolve the name, refuse any
non-public address and connect to the address that was checked, keeping the
``Host`` header, and never follow redirects.
"""

from __future__ import annotations

import socket

import httpx
import pytest
from aiohttp import web

from promptise.approval import ApprovalRequest, WebhookApprovalHandler
from promptise.runtime.escalation import _fire_webhook

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Receiver:
    """A local aiohttp server (127.0.0.1, port 0) that records every request.

    ``POST /approvals`` answers ``post_status`` (with ``Location: /stolen``
    for a 3xx), ``GET /approvals/{id}`` answers an approval, anything else 204.
    """

    def __init__(self, post_status: int = 202) -> None:
        self.post_status = post_status
        self.requests: list[tuple[str, str, dict[str, str]]] = []
        self.port = 0
        self._runner: web.AppRunner | None = None

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        await request.read()
        self.requests.append((request.method, request.path, dict(request.headers)))
        if request.method == "POST" and request.path == "/approvals":
            if 300 <= self.post_status < 400:
                return web.Response(status=self.post_status, headers={"Location": "/stolen"})
            return web.Response(status=self.post_status)
        if request.method == "GET" and request.path.startswith("/approvals/"):
            return web.json_response({"approved": True, "reviewer_id": "ops"})
        return web.Response(status=204)

    @property
    def paths(self) -> list[str]:
        return [path for _, path, _ in self.requests]

    async def __aenter__(self) -> _Receiver:
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self._handle)
        self._runner = web.AppRunner(app, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        server = site._server
        assert server is not None
        self.port = server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._runner is not None:
            await self._runner.cleanup()


def _fake_dns(monkeypatch: pytest.MonkeyPatch, host: str, answers: list[str]) -> list[str]:
    """Resolve *host* to ``answers[i]`` on the i-th lookup (the last one repeats).

    Patches :func:`socket.getaddrinfo`, which both the asyncio resolver and
    httpx (through anyio) end up calling.  Returns the answers handed out.
    """
    real = socket.getaddrinfo
    handed_out: list[str] = []

    def fake(name, port, *args, **kwargs):  # type: ignore[no-untyped-def]
        if (name.decode() if isinstance(name, bytes) else name) != host:
            return real(name, port, *args, **kwargs)
        ip = answers[min(len(handed_out), len(answers) - 1)]
        handed_out.append(ip)
        family = socket.AF_INET6 if ":" in ip else socket.AF_INET
        sockaddr = (ip, port or 0, 0, 0) if family == socket.AF_INET6 else (ip, port or 0)
        return [(family, socket.SOCK_STREAM, 6, "", sockaddr)]

    monkeypatch.setattr(socket, "getaddrinfo", fake)
    return handed_out


def _only_private(monkeypatch: pytest.MonkeyPatch, *private: str) -> None:
    """Treat exactly the addresses in *private* as non-public.

    Lets a test reach the 127.0.0.1 receiver as if it were a public host.
    """
    import promptise.mcp.server._openapi as openapi

    monkeypatch.setattr(openapi, "_is_private_ip", lambda ip: str(ip) in private)


def _request(timeout: float = 3.0) -> ApprovalRequest:
    return ApprovalRequest(
        request_id="req-1",
        tool_name="issue_refund",
        arguments={"order_id": "A-1001"},
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# WebhookApprovalHandler
# ---------------------------------------------------------------------------


class TestApprovalWebhookRebinding:
    async def test_host_rebound_to_loopback_after_construction_is_not_contacted(self, monkeypatch):
        """The URL passes the construction check; DNS then points it at 127.0.0.1."""
        async with _Receiver() as receiver:
            lookups = _fake_dns(
                monkeypatch, "approvals.rebind.test", ["93.184.216.34", "127.0.0.1"]
            )
            handler = WebhookApprovalHandler(
                f"http://approvals.rebind.test:{receiver.port}/approvals", poll_interval=0.5
            )
            assert lookups == ["93.184.216.34"]  # passed the construction-time check
            with pytest.raises(ValueError, match="private/internal IP 127.0.0.1"):
                await handler.request_approval(_request())
        assert receiver.requests == []

    async def test_poll_target_rebound_to_a_private_address_fails_closed(self, monkeypatch):
        """Each poll is checked too, not only the POST."""
        _only_private(monkeypatch, "10.0.0.7")
        async with _Receiver() as receiver:
            _fake_dns(monkeypatch, "approvals.poll.test", ["127.0.0.1", "127.0.0.1", "10.0.0.7"])
            handler = WebhookApprovalHandler(
                f"http://approvals.poll.test:{receiver.port}/approvals", poll_interval=0.5
            )
            with pytest.raises(ValueError, match="private/internal IP 10.0.0.7"):
                await handler.request_approval(_request())
        assert receiver.paths == ["/approvals"]

    async def test_connects_to_the_checked_address_without_a_second_lookup(self, monkeypatch):
        _only_private(monkeypatch)
        async with _Receiver() as receiver:
            lookups = _fake_dns(monkeypatch, "approvals.pin.test", ["127.0.0.1"])
            handler = WebhookApprovalHandler(
                f"http://approvals.pin.test:{receiver.port}/approvals", poll_interval=0.5
            )
            decision = await handler.request_approval(_request())
        assert decision.approved and decision.reviewer_id == "ops"
        assert receiver.paths == ["/approvals", "/approvals/req-1"]
        for _, _, headers in receiver.requests:
            assert headers["Host"] == f"approvals.pin.test:{receiver.port}"
        # construction + one check per request; httpx never resolves the name
        assert len(lookups) == 1 + len(receiver.requests)

    async def test_redirects_are_not_followed_even_with_a_following_client(self, monkeypatch):
        _only_private(monkeypatch)
        async with _Receiver(post_status=307) as receiver:
            _fake_dns(monkeypatch, "approvals.redirect.test", ["127.0.0.1"])
            client = httpx.AsyncClient(follow_redirects=True)
            handler = WebhookApprovalHandler(
                f"http://approvals.redirect.test:{receiver.port}/approvals",
                http_client=client,
                poll_interval=0.5,
            )
            with pytest.raises(httpx.HTTPStatusError):
                await handler.request_approval(_request())
            await client.aclose()
        assert receiver.paths == ["/approvals"]

    async def test_allow_private_networks_still_delivers(self):
        async with _Receiver() as receiver:
            handler = WebhookApprovalHandler(
                f"http://localhost:{receiver.port}/approvals",
                poll_interval=0.5,
                allow_private_networks=True,
            )
            decision = await handler.request_approval(_request())
        assert decision.approved
        assert receiver.paths == ["/approvals", "/approvals/req-1"]
        assert receiver.requests[0][2]["Host"] == f"localhost:{receiver.port}"

    async def test_https_keeps_the_host_name_for_tls(self, monkeypatch):
        """The request goes to the IP; SNI and Host still name the host."""
        seen: list[httpx.Request] = []

        def service(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            if request.method == "POST":
                return httpx.Response(202)
            return httpx.Response(200, json={"approved": False, "reason": "no"})

        _fake_dns(monkeypatch, "approvals.tls.test", ["93.184.216.34"])
        client = httpx.AsyncClient(transport=httpx.MockTransport(service))
        handler = WebhookApprovalHandler(
            "https://approvals.tls.test/approvals", http_client=client, poll_interval=0.5
        )
        decision = await handler.request_approval(_request())
        await client.aclose()
        assert not decision.approved
        assert [str(r.url) for r in seen] == [
            "https://93.184.216.34/approvals",
            "https://93.184.216.34/approvals/req-1",
        ]
        for request in seen:
            assert request.headers["Host"] == "approvals.tls.test"
            assert request.extensions["sni_hostname"] == "approvals.tls.test"


# ---------------------------------------------------------------------------
# Runtime escalation webhook
# ---------------------------------------------------------------------------


class TestEscalationWebhookRebinding:
    async def test_answer_changing_between_check_and_connect_is_not_contacted(self, monkeypatch):
        """The check saw a public address; the connection must not get another one.

        Here ``::1`` stands in for the public answer (nothing listens there on
        the receiver's port) and 127.0.0.1 is the private one.
        """
        _only_private(monkeypatch, "127.0.0.1")
        async with _Receiver() as receiver:
            _fake_dns(monkeypatch, "hooks.rebind.test", ["::1", "127.0.0.1"])
            await _fire_webhook(f"http://hooks.rebind.test:{receiver.port}/alert", {"x": 1})
        assert receiver.requests == []

    async def test_private_answer_is_refused(self, monkeypatch, caplog):
        async with _Receiver() as receiver:
            _fake_dns(monkeypatch, "hooks.private.test", ["127.0.0.1"])
            await _fire_webhook(f"http://hooks.private.test:{receiver.port}/alert", {"x": 1})
        assert receiver.requests == []
        assert "private/internal IP 127.0.0.1" in caplog.text

    async def test_delivers_to_the_checked_address_with_the_host_header(self, monkeypatch):
        _only_private(monkeypatch)
        async with _Receiver() as receiver:
            lookups = _fake_dns(monkeypatch, "hooks.pin.test", ["127.0.0.1"])
            await _fire_webhook(f"http://hooks.pin.test:{receiver.port}/alert", {"x": 1})
        assert receiver.paths == ["/alert"]
        assert receiver.requests[0][2]["Host"] == f"hooks.pin.test:{receiver.port}"
        assert len(lookups) == 1

    async def test_redirects_are_not_followed(self, monkeypatch):
        _only_private(monkeypatch)
        async with _Receiver(post_status=307) as receiver:
            _fake_dns(monkeypatch, "hooks.redirect.test", ["127.0.0.1"])
            await _fire_webhook(f"http://hooks.redirect.test:{receiver.port}/approvals", {})
        assert receiver.paths == ["/approvals"]
