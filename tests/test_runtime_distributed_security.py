"""Security and correctness tests for RuntimeTransport and RuntimeCoordinator.

Every server binds 127.0.0.1 on an OS-assigned port (``port=0``).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import aiohttp
import pytest

from promptise.runtime.config import ProcessConfig
from promptise.runtime.distributed.coordinator import RuntimeCoordinator
from promptise.runtime.distributed.transport import RuntimeTransport
from promptise.runtime.runtime import AgentRuntime

TOKEN = "s3cret-token"


def _mock_agent() -> AsyncMock:
    agent = AsyncMock()
    agent.ainvoke = AsyncMock(return_value={"messages": []})
    agent.shutdown = AsyncMock()
    return agent


@pytest.fixture
async def runtime() -> AsyncIterator[AgentRuntime]:
    rt = AgentRuntime()
    await rt.add_process("worker", ProcessConfig())
    yield rt
    await rt.stop_all()


async def _serve(runtime: AgentRuntime, **kwargs: Any) -> RuntimeTransport:
    transport = RuntimeTransport(runtime, host="127.0.0.1", port=0, **kwargs)
    await transport.start()
    return transport


def _url(transport: RuntimeTransport, path: str) -> str:
    return f"http://127.0.0.1:{transport.port}{path}"


# =========================================================================
# Bind and token policy
# =========================================================================


class TestBindPolicy:
    @pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.5", "node1.internal"])
    def test_public_bind_without_token_is_refused(self, host: str) -> None:
        with pytest.raises(ValueError, match="auth_token"):
            RuntimeTransport(AgentRuntime(), host=host)

    def test_public_bind_with_token_is_allowed(self) -> None:
        RuntimeTransport(AgentRuntime(), host="0.0.0.0", auth_token=TOKEN)

    def test_explicit_opt_out_allows_public_bind(self) -> None:
        RuntimeTransport(AgentRuntime(), host="0.0.0.0", allow_unauthenticated=True)

    @pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.2"])
    def test_loopback_bind_without_token_is_allowed(self, host: str) -> None:
        RuntimeTransport(AgentRuntime(), host=host)

    @pytest.mark.parametrize("token", ["", "   "])
    def test_empty_token_is_refused(self, token: str) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            RuntimeTransport(AgentRuntime(), auth_token=token)

    async def test_port_zero_reports_the_bound_port(self, runtime: AgentRuntime) -> None:
        transport = await _serve(runtime)
        try:
            assert transport.port > 0
            async with aiohttp.ClientSession() as s, s.get(_url(transport, "/health")) as r:
                assert r.status == 200
        finally:
            await transport.stop()


# =========================================================================
# Authentication
# =========================================================================


class TestAuthentication:
    async def test_control_endpoints_require_the_token(self, runtime: AgentRuntime) -> None:
        transport = await _serve(runtime, auth_token=TOKEN)
        try:
            async with aiohttp.ClientSession() as s:
                for method, path in [
                    ("GET", "/status"),
                    ("GET", "/processes"),
                    ("GET", "/processes/worker/status"),
                    ("POST", "/processes/worker/start"),
                    ("POST", "/processes/worker/stop"),
                    ("POST", "/processes/worker/restart"),
                    ("POST", "/processes/worker/event"),
                ]:
                    async with s.request(method, _url(transport, path)) as r:
                        assert r.status == 401, (method, path)
                    async with s.request(
                        method,
                        _url(transport, path),
                        headers={"Authorization": "Bearer wrong"},
                    ) as r:
                        assert r.status == 401, (method, path)
                async with s.get(
                    _url(transport, "/processes"),
                    headers={"Authorization": f"Bearer {TOKEN}"},
                ) as r:
                    assert r.status == 200
                # Liveness probe stays open.
                async with s.get(_url(transport, "/health")) as r:
                    assert r.status == 200
        finally:
            await transport.stop()


# =========================================================================
# Browser protection on a loopback bind (no token)
# =========================================================================


class TestLoopbackBrowserProtection:
    async def test_foreign_host_header_is_refused(self, runtime: AgentRuntime) -> None:
        transport = await _serve(runtime)
        inject = AsyncMock()
        try:
            with patch.object(runtime.get_process("worker"), "inject", inject):
                async with aiohttp.ClientSession() as s:
                    async with s.post(
                        _url(transport, "/processes/worker/event"),
                        json={"payload": {"x": 1}},
                        headers={"Host": f"evil.example:{transport.port}"},
                    ) as r:
                        assert r.status == 421
                    async with s.get(
                        _url(transport, "/processes"), headers={"Host": "evil.example"}
                    ) as r:
                        assert r.status == 421
            inject.assert_not_awaited()
        finally:
            await transport.stop()

    async def test_cross_origin_request_is_refused(self, runtime: AgentRuntime) -> None:
        transport = await _serve(runtime)
        inject = AsyncMock()
        try:
            with patch.object(runtime.get_process("worker"), "inject", inject):
                async with aiohttp.ClientSession() as s:
                    # A page on another site posting a "simple" text/plain body.
                    async with s.post(
                        _url(transport, "/processes/worker/event"),
                        data='{"payload": {"x": 1}}',
                        headers={
                            "Origin": "https://evil.example",
                            "Content-Type": "text/plain",
                        },
                    ) as r:
                        assert r.status == 403
                    async with s.post(
                        _url(transport, "/processes/worker/stop"),
                        headers={"Origin": "null"},
                    ) as r:
                        assert r.status == 403
            inject.assert_not_awaited()
        finally:
            await transport.stop()

    async def test_local_clients_still_work(self, runtime: AgentRuntime) -> None:
        transport = await _serve(runtime)
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(_url(transport, "/processes")) as r:
                    assert r.status == 200
                async with s.get(
                    _url(transport, "/processes"),
                    headers={
                        "Host": f"localhost:{transport.port}",
                        "Origin": "http://localhost:3000",
                    },
                ) as r:
                    assert r.status == 200
        finally:
            await transport.stop()


# =========================================================================
# Event injection input validation
# =========================================================================


class TestEventValidation:
    async def test_payload_must_be_an_object(self, runtime: AgentRuntime) -> None:
        transport = await _serve(runtime)
        inject = AsyncMock()
        try:
            with patch.object(runtime.get_process("worker"), "inject", inject):
                async with aiohttp.ClientSession() as s:
                    url = _url(transport, "/processes/worker/event")
                    for body in ([1, 2], {"payload": "text"}, {"metadata": [1]}):
                        async with s.post(url, json=body) as r:
                            assert r.status == 400, body
                    inject.assert_not_awaited()
                    async with s.post(url, json={"trigger_type": "remote"}) as r:
                        assert r.status == 202
            event = inject.await_args.args[0]
            assert event.payload == {}
            assert event.metadata == {}
        finally:
            await transport.stop()


# =========================================================================
# Coordinator
# =========================================================================


class TestCoordinatorAuth:
    async def test_coordinator_sends_the_node_token(self, runtime: AgentRuntime) -> None:
        transport = await _serve(runtime, auth_token=TOKEN)
        try:
            coordinator = RuntimeCoordinator(auth_token=TOKEN)
            coordinator.register_node("n1", f"http://127.0.0.1:{transport.port}")
            status = await coordinator.get_node_status("n1")
            assert status["node_id"] == "node-1"
            assert "worker" in str(status)
        finally:
            await transport.stop()

    async def test_per_node_token_overrides_the_default(self, runtime: AgentRuntime) -> None:
        transport = await _serve(runtime, auth_token=TOKEN)
        try:
            coordinator = RuntimeCoordinator(auth_token="other-cluster-token")
            coordinator.register_node("n1", f"http://127.0.0.1:{transport.port}", auth_token=TOKEN)
            with patch("promptise.agent.build_agent", AsyncMock(return_value=_mock_agent())):
                result = await coordinator.start_process_on_node("n1", "worker")
            assert result == {"status": "started", "name": "worker"}
            stopped = await coordinator.stop_process_on_node("n1", "worker")
            assert stopped["status"] == "stopped"
        finally:
            await transport.stop()

    async def test_rejected_requests_raise(self, runtime: AgentRuntime) -> None:
        transport = await _serve(runtime, auth_token=TOKEN)
        try:
            url = f"http://127.0.0.1:{transport.port}"
            unauth = RuntimeCoordinator()
            unauth.register_node("n1", url)
            with pytest.raises(RuntimeError, match="401"):
                await unauth.start_process_on_node("n1", "worker")
            with pytest.raises(RuntimeError, match="401"):
                await unauth.inject_event_on_node("n1", "worker", {"x": 1})

            authed = RuntimeCoordinator(auth_token=TOKEN)
            authed.register_node("n1", url)
            with pytest.raises(RuntimeError, match="404"):
                await authed.start_process_on_node("n1", "missing")
        finally:
            await transport.stop()

    async def test_token_never_appears_in_node_info(self) -> None:
        coordinator = RuntimeCoordinator(auth_token=TOKEN)
        node = coordinator.register_node("n1", "http://127.0.0.1:1", auth_token="node-secret")
        assert TOKEN not in str(node.to_dict())
        assert "node-secret" not in str(node.to_dict())


class TestCoordinatorHealth:
    async def test_single_missed_check_within_node_timeout_keeps_node_healthy(self) -> None:
        coordinator = RuntimeCoordinator(node_timeout=60)
        node = coordinator.register_node("n1", "http://127.0.0.1:1")
        with patch.object(
            coordinator, "_check_node_health", AsyncMock(return_value={"process_count": 2})
        ):
            await coordinator.check_health()
        assert node.is_healthy

        with patch.object(
            coordinator, "_check_node_health", AsyncMock(side_effect=OSError("refused"))
        ):
            result = await coordinator.check_health()
        assert node.is_healthy
        assert result["n1"]["missed_check"] is True

    async def test_node_is_unhealthy_after_node_timeout(self) -> None:
        coordinator = RuntimeCoordinator(node_timeout=0)
        node = coordinator.register_node("n1", "http://127.0.0.1:1")
        with patch.object(
            coordinator, "_check_node_health", AsyncMock(return_value={"process_count": 0})
        ):
            await coordinator.check_health()
        with patch.object(
            coordinator, "_check_node_health", AsyncMock(side_effect=OSError("refused"))
        ):
            result = await coordinator.check_health()
        assert not node.is_healthy
        assert result["n1"]["status"] == "unhealthy"

    async def test_cancelled_node_check_marks_node_unhealthy(self) -> None:
        coordinator = RuntimeCoordinator()
        node = coordinator.register_node("n1", "http://127.0.0.1:1")
        with patch.object(
            coordinator,
            "_check_node_health",
            AsyncMock(side_effect=asyncio.CancelledError()),
        ):
            result = await coordinator.check_health()
        assert result["n1"]["status"] == "unhealthy"
        assert not node.is_healthy

    async def test_monitor_survives_a_failing_round(self) -> None:
        coordinator = RuntimeCoordinator(health_check_interval=0.01)
        calls = 0

        async def flaky() -> dict[str, Any]:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("transient")
            return {}

        with patch.object(coordinator, "check_health", side_effect=flaky):
            await coordinator.start_health_monitor()
            try:
                for _ in range(200):
                    if calls >= 3:
                        break
                    await asyncio.sleep(0.01)
            finally:
                await coordinator.stop_health_monitor()
        assert calls >= 3
