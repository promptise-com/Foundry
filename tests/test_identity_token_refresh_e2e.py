"""Short-lived identity credentials end to end — real HTTP, a real JWKS issuer.

Regression: ``build_agent`` fetched the identity's credential once and sent
it on every request.  Once it expired (plus the server's leeway) the server
answered ``401``; the MCP SDK tore down the transport, but the tool call
waiting for its response never returned.

Now the client asks the identity for a credential on every request, reopens
the session when the credential changes, refreshes once on ``401``, and a
call whose session dies fails at once.  Every test bounds every call with
``FAIL_FAST`` so a regression shows up as a failure, never as a hung suite.

A local OIDC issuer (discovery document + JWKS, threaded HTTP server) signs
tokens with a TTL of a few seconds; the MCP server verifies them with
``JwksAuth.from_discovery(..., leeway=0)`` and ``require_auth=True``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import jwt as pyjwt
import pytest
import uvicorn
from cryptography.hazmat.primitives.asymmetric import rsa
from langchain_core.messages import AIMessage, ToolMessage
from sse_starlette.sse import AppStatus

from promptise.agent import build_agent
from promptise.config import HTTPServerSpec
from promptise.identity import AgentIdentity, CallableTokenProvider, CredentialAcquisitionError
from promptise.mcp.client import (
    MCPClient,
    MCPConnectionRejectedError,
    MCPCredentialError,
    MCPMultiClient,
)
from promptise.mcp.server import AuthMiddleware, JwksAuth, MCPServer, RequestContext

AUDIENCE = "api://support-tools"
TTL = 3  # seconds
# Every call is one or two round trips on loopback (plus a reconnect).
FAIL_FAST = 10.0


# =====================================================================
# A local OIDC issuer
# =====================================================================


class LocalIdP:
    """Signs short-lived JWTs and publishes its JWKS over HTTP."""

    def __init__(self, ttl: int = TTL) -> None:
        self.ttl = ttl
        self.jwks_fetches = 0
        self.minted = 0
        # Every token minted, so tests can check none of them leaks into a
        # log line or an error message.
        self.tokens: list[str] = []
        # While set, minting fails like an unreachable IdP.
        self.down = False
        self._new_key()
        idp = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/.well-known/openid-configuration":
                    body: dict[str, Any] = {
                        "issuer": idp.issuer,
                        "jwks_uri": f"{idp.issuer}/jwks.json",
                    }
                elif self.path == "/jwks.json":
                    idp.jwks_fetches += 1
                    body = {"keys": [idp.public_jwk]}
                else:
                    self.send_response(404)
                    self.end_headers()
                    return
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self._http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.issuer = f"http://127.0.0.1:{self._http.server_address[1]}"
        threading.Thread(target=self._http.serve_forever, daemon=True).start()

    def _new_key(self) -> None:
        self._key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.kid = uuid.uuid4().hex[:8]
        self.public_jwk = {
            **json.loads(pyjwt.algorithms.RSAAlgorithm.to_jwk(self._key.public_key())),
            "kid": self.kid,
            "use": "sig",
            "alg": "RS256",
        }

    def rotate(self) -> None:
        """Replace the signing key; the old one is no longer published."""
        self._new_key()

    def mint(self, audience: str | None = None) -> str:
        if self.down:
            raise CredentialAcquisitionError("[local-idp] the IdP is unreachable")
        self.minted += 1
        now = int(time.time())
        claims = {
            "iss": self.issuer,
            "sub": "support-bot",
            "aud": audience or AUDIENCE,
            "iat": now,
            "exp": now + self.ttl,
            "jti": uuid.uuid4().hex,
        }
        token = pyjwt.encode(claims, self._key, algorithm="RS256", headers={"kid": self.kid})
        self.tokens.append(token)
        return token

    def close(self) -> None:
        self._http.shutdown()
        self._http.server_close()


class RevocableJwksAuth:
    """``JwksAuth`` plus a revocation list, like an IdP-backed deny list.

    Lets a test make the server reject a credential the agent's identity
    still believes is valid — the case a 401-triggered refresh exists for.
    """

    def __init__(self, inner: JwksAuth) -> None:
        self._inner = inner
        self.revoked: set[str] = set()
        # Refuse every credential (the server captures ``verify_token`` at
        # start-up, so the switch lives here rather than in a replacement).
        self.refuse_all = False

    def _revoked(self, token: str) -> bool:
        if self.refuse_all:
            return True
        return pyjwt.decode(token, options={"verify_signature": False}).get("jti") in self.revoked

    def verify_token(self, token: str) -> bool:
        return not self._revoked(token) and self._inner.verify_token(token)

    async def authenticate(self, ctx: RequestContext) -> str:
        return await self._inner.authenticate(ctx)


def _support_server(idp: LocalIdP, *, revocable: bool = False) -> tuple[MCPServer, Any]:
    auth: Any = JwksAuth.from_discovery(issuer=idp.issuer, audience=AUDIENCE, leeway=0)
    if revocable:
        auth = RevocableJwksAuth(auth)
    server = MCPServer("support-tools", require_auth=True)
    server.add_middleware(AuthMiddleware(auth))

    @server.tool()
    async def whoami(ctx: RequestContext) -> dict:
        """Show which credential this server verified for the call."""
        payload = ctx.state.get("_jwt_payload", {})
        return {"client_id": ctx.client_id, "jti": payload.get("jti")}

    @server.tool()
    async def hold(ctx: RequestContext) -> dict:
        """Wait until the test releases the call (a long-running tool)."""
        payload = ctx.state.get("_jwt_payload", {})
        HOLD.invocations += 1
        HOLD.entered.set()
        await asyncio.wait_for(HOLD.release.wait(), FAIL_FAST)
        return {"jti": payload.get("jti")}

    return server, auth


class _Hold:
    """Rendezvous for the ``hold`` tool (server and test share one loop)."""

    def reset(self) -> None:
        self.invocations = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()


HOLD = _Hold()


def _assert_no_token_leaked(idp: LocalIdP, caplog: pytest.LogCaptureFixture, *texts: str) -> None:
    """No minted credential (nor a recognisable part of one) in logs or errors."""
    assert idp.tokens
    haystack = [r.getMessage() for r in caplog.records]
    haystack += [r.exc_text or "" for r in caplog.records]
    haystack += list(texts)
    for token in idp.tokens:
        signature = token.rsplit(".", 1)[1]
        for text in haystack:
            assert token not in text
            assert signature not in text


class _RecordingTransport:
    """Records the ``Authorization`` header of every request the client sends."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.authorizations: list[str | None] = []
        original = httpx.AsyncHTTPTransport.handle_async_request
        recorder = self

        async def handle(transport: Any, request: httpx.Request) -> httpx.Response:
            recorder.authorizations.append(request.headers.get("authorization"))
            return await original(transport, request)

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle)


@asynccontextmanager
async def _serve(server: MCPServer, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[str]:
    """Run *server* over Streamable HTTP and yield its ``/mcp`` URL."""
    instances: list[uvicorn.Server] = []

    class _Recording(uvicorn.Server):
        def __init__(self, config: uvicorn.Config) -> None:
            super().__init__(config)
            instances.append(self)

    monkeypatch.setattr(uvicorn, "Server", _Recording)
    task = asyncio.ensure_future(server.run_async(transport="http", host="127.0.0.1", port=0))
    try:
        for _ in range(400):
            if task.done():
                task.result()
            if instances and instances[0].started:
                break
            await asyncio.sleep(0.025)
        else:
            raise RuntimeError("server did not start")
        instances[0].config.timeout_graceful_shutdown = 5
        port = instances[0].servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        if instances:
            instances[0].should_exit = True
        try:
            await asyncio.wait_for(task, 15)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            task.cancel()
        # A uvicorn server stopped while an SSE stream is open sets
        # sse-starlette's process-wide ``AppStatus.should_exit``; every later
        # SSE response (the next server's ``initialize``) would then close at
        # once and leave the client waiting. A real restart is a new process.
        AppStatus.should_exit = False


@pytest.fixture
def idp() -> Any:
    issuer = LocalIdP()
    yield issuer
    issuer.close()


def _identity(idp: LocalIdP) -> AgentIdentity:
    return AgentIdentity(
        "support-bot",
        credential=CallableTokenProvider(
            token_fn=idp.mint, provider_label="local-idp", default_audience=AUDIENCE
        ),
    )


def _whoami(result: Any) -> dict:
    return json.loads(result.content[0].text)


async def _wait_until_expired(token: str) -> None:
    exp = pyjwt.decode(token, options={"verify_signature": False})["exp"]
    await asyncio.sleep(max(exp - time.time(), 0) + 1.2)


def _scripted_model(*replies: AIMessage) -> MagicMock:
    model = MagicMock(spec=["ainvoke", "bind_tools", "with_structured_output"])
    model.ainvoke = AsyncMock(side_effect=list(replies))
    model.bind_tools = MagicMock(return_value=model)
    return model


def _call_whoami(call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": "whoami", "args": {}, "id": call_id}])


# =====================================================================
# build_agent with an identity: the call after expiry succeeds
# =====================================================================


class TestAgentIdentityRefresh:
    async def test_tool_call_after_expiry_succeeds_with_a_fresh_credential(
        self, idp, monkeypatch, caplog
    ):
        caplog.set_level(logging.DEBUG)
        server, _ = _support_server(idp)
        async with _serve(server, monkeypatch) as url:
            model = _scripted_model(
                _call_whoami("c1"),
                AIMessage(content="first done"),
                _call_whoami("c2"),
                AIMessage(content="second done"),
            )
            with (
                patch("promptise.agent._normalize_model", return_value=model),
                patch.dict("sys.modules", {"deepagents": None}),
            ):
                agent = await asyncio.wait_for(
                    build_agent(
                        model="openai:gpt-5-mini",
                        servers={"support": HTTPServerSpec(url=url, audience=AUDIENCE)},
                        identity=_identity(idp),
                    ),
                    FAIL_FAST,
                )
            try:
                first = await asyncio.wait_for(
                    agent.ainvoke({"messages": [{"role": "user", "content": "who am I?"}]}),
                    FAIL_FAST,
                )
                first_result = json.loads(
                    next(m for m in first["messages"] if isinstance(m, ToolMessage)).content
                )
                assert first_result["client_id"] == "support-bot"

                # Let the credential expire (the server allows no leeway).
                await asyncio.sleep(TTL + 1.2)

                started = time.monotonic()
                second = await asyncio.wait_for(
                    agent.ainvoke({"messages": [{"role": "user", "content": "and now?"}]}),
                    FAIL_FAST,
                )
                assert time.monotonic() - started < FAIL_FAST
                second_result = json.loads(
                    next(m for m in second["messages"] if isinstance(m, ToolMessage)).content
                )
                assert second_result["client_id"] == "support-bot"
                # A different, freshly minted credential was presented.
                assert second_result["jti"] != first_result["jti"]
            finally:
                await agent.shutdown()
        _assert_no_token_leaked(idp, caplog)

    async def test_revoked_credential_is_refreshed_once_on_401(self, idp, monkeypatch):
        """The identity still caches a credential the server now refuses:
        the 401 forces one refresh, and the call goes through on a session
        opened with the new credential."""
        idp.ttl = 3600  # the identity has no reason to refresh on its own
        server, auth = _support_server(idp, revocable=True)
        identity = _identity(idp)
        async with _serve(server, monkeypatch) as url:
            client = MCPClient(
                url=url,
                bearer_token_provider=lambda force: identity.get_credential(
                    AUDIENCE, force_refresh=force
                ),
            )
            async with client:
                first = _whoami(await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST))
                auth.revoked.add(first["jti"])

                second = _whoami(await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST))
                assert second["client_id"] == "support-bot"
                assert second["jti"] != first["jti"]
                assert idp.minted == 2

                # The refreshed credential is cached and reused from here on.
                third = _whoami(await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST))
                assert third["jti"] == second["jti"]
                assert idp.minted == 2

    async def test_refresh_that_is_refused_too_fails_fast(self, idp, monkeypatch):
        """If even a fresh credential is refused, the call fails with a
        typed error that says so — it never hangs."""
        idp.ttl = 3600
        server, auth = _support_server(idp, revocable=True)
        identity = _identity(idp)
        async with _serve(server, monkeypatch) as url:
            client = MCPClient(
                url=url,
                bearer_token_provider=lambda force: identity.get_credential(
                    AUDIENCE, force_refresh=force
                ),
            )
            async with client:
                await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST)
                auth.refuse_all = True
                started = time.monotonic()
                with pytest.raises(MCPConnectionRejectedError) as info:
                    await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST)
                assert time.monotonic() - started < FAIL_FAST
                assert info.value.status_code == 401
                assert info.value.mid_session is True
                assert "freshly acquired token was rejected too" in str(info.value)

                # The client recovers once the server accepts credentials again.
                auth.refuse_all = False
                result = await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST)
                assert _whoami(result)["client_id"] == "support-bot"


# =====================================================================
# Failing closed when no credential can be acquired
# =====================================================================


class TestFailClosed:
    async def test_idp_outage_fails_the_call_and_sends_nothing_unauthenticated(
        self, idp, monkeypatch, caplog
    ):
        """The credential expired and the IdP is down: the call fails with
        MCPCredentialError before anything is sent; it never goes out
        without a credential. Once the IdP is back, calls succeed again."""
        caplog.set_level(logging.DEBUG)
        server, _ = _support_server(idp)
        identity = _identity(idp)
        async with _serve(server, monkeypatch) as url:
            sent = _RecordingTransport(monkeypatch)
            client = MCPClient(
                url=url,
                bearer_token_provider=lambda force: identity.get_credential(
                    AUDIENCE, force_refresh=force
                ),
            )
            async with client:
                first = await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST)
                idp.down = True
                await asyncio.sleep(TTL + 1.2)

                sent_before = len(sent.authorizations)
                with pytest.raises(MCPCredentialError) as info:
                    await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST)
                assert isinstance(info.value.__cause__, CredentialAcquisitionError)
                # Nothing at all was sent for the failed call.
                assert len(sent.authorizations) == sent_before

                idp.down = False
                second = await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST)
                assert _whoami(second)["jti"] != _whoami(first)["jti"]
            assert sent.authorizations
            assert all(a and a.startswith("Bearer ") for a in sent.authorizations)
        _assert_no_token_leaked(idp, caplog, str(info.value))

    async def test_refresh_after_401_with_idp_down_fails_closed(self, idp, monkeypatch, caplog):
        """The server rejects the cached credential and the forced refresh
        cannot reach the IdP: the call fails, and the 401 is not answered
        by re-sending the request without a credential."""
        caplog.set_level(logging.DEBUG)
        idp.ttl = 3600
        server, auth = _support_server(idp, revocable=True)
        identity = _identity(idp)
        async with _serve(server, monkeypatch) as url:
            sent = _RecordingTransport(monkeypatch)
            client = MCPClient(
                url=url,
                bearer_token_provider=lambda force: identity.get_credential(
                    AUDIENCE, force_refresh=force
                ),
            )
            async with client:
                first = _whoami(await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST))
                auth.revoked.add(first["jti"])
                idp.down = True
                with pytest.raises(MCPCredentialError) as info:
                    await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST)

                idp.down = False
                second = _whoami(await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST))
                assert second["jti"] != first["jti"]
            assert all(a and a.startswith("Bearer ") for a in sent.authorizations)
        _assert_no_token_leaked(idp, caplog, str(info.value))


# =====================================================================
# Renewal while a call is still running
# =====================================================================


class TestRenewalWithCallInFlight:
    async def test_in_flight_call_finishes_on_its_session_and_runs_once(self, idp, monkeypatch):
        """A renewed credential opens a new session for new calls; a call
        already running on the old session is neither cancelled nor
        re-sent — it completes there, exactly once."""
        HOLD.reset()
        server, _ = _support_server(idp)
        identity = _identity(idp)
        async with _serve(server, monkeypatch) as url:
            client = MCPClient(
                url=url,
                bearer_token_provider=lambda force: identity.get_credential(
                    AUDIENCE, force_refresh=force
                ),
            )
            async with client:
                first = _whoami(await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST))
                held = asyncio.ensure_future(client.call_tool("hold", {}))
                await asyncio.wait_for(HOLD.entered.wait(), FAIL_FAST)

                # Past half the credential's lifetime the identity renews it.
                await asyncio.sleep(TTL / 2 + 0.3)
                second = _whoami(await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST))
                assert second["jti"] != first["jti"]
                assert not held.done()

                HOLD.release.set()
                result = json.loads((await asyncio.wait_for(held, FAIL_FAST)).content[0].text)
                assert result["jti"] == first["jti"]  # ran on the original session
                assert HOLD.invocations == 1


# =====================================================================
# A static bearer token: the call after expiry fails fast, never hangs
# =====================================================================


class TestStaticTokenExpiry:
    async def test_call_after_expiry_fails_fast_with_a_clear_error(self, idp, monkeypatch):
        server, _ = _support_server(idp)
        token = idp.mint()
        async with _serve(server, monkeypatch) as url:
            async with MCPClient(url=url, bearer_token=token) as client:
                await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST)
                await _wait_until_expired(token)

                started = time.monotonic()
                with pytest.raises(MCPConnectionRejectedError) as info:
                    await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST)
                assert time.monotonic() - started < FAIL_FAST
                err = info.value
                assert err.status_code == 401
                assert err.mid_session is True
                assert "401 Unauthorized" in str(err)
                assert "bearer_token_provider" in str(err)
                assert token not in str(err)

                # The session is gone; later calls fail the same way, at once.
                with pytest.raises(MCPConnectionRejectedError):
                    await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST)
                # A static token is never swapped for another credential.
                assert idp.minted == 1

    async def test_multi_client_names_the_server_and_keeps_the_route(self, idp, monkeypatch):
        server, _ = _support_server(idp)
        token = idp.mint()
        async with _serve(server, monkeypatch) as url:
            multi = MCPMultiClient({"support": MCPClient(url=url, bearer_token=token)})
            async with multi:
                await multi.list_tools()
                await _wait_until_expired(token)
                with pytest.raises(MCPConnectionRejectedError, match="Server 'support'"):
                    await asyncio.wait_for(multi.call_tool("whoami", {}), FAIL_FAST)
                # Still routed: the next error is the rejection again, not
                # "Unknown tool".
                assert multi.tool_to_server["whoami"] == "support"


# =====================================================================
# Key rotation at the issuer
# =====================================================================


class TestKeyRotation:
    async def test_rotated_key_is_accepted_on_first_use(self, idp, monkeypatch):
        """A token signed with a key minted after the server cached the JWKS
        is accepted at once — no wait for a cache expiry or cooldown."""
        idp.ttl = 3600
        server, _ = _support_server(idp)
        async with _serve(server, monkeypatch) as url:
            async with MCPClient(url=url, bearer_token=idp.mint()) as client:
                await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST)
            assert idp.jwks_fetches == 1

            old_kid = idp.kid
            idp.rotate()
            assert idp.kid != old_kid
            async with MCPClient(url=url, bearer_token=idp.mint()) as client:
                result = await asyncio.wait_for(client.call_tool("whoami", {}), FAIL_FAST)
            assert _whoami(result)["client_id"] == "support-bot"
            assert idp.jwks_fetches == 2

    async def test_unknown_kid_refetch_is_rate_limited_per_kid(self, idp, monkeypatch):
        """Made-up key ids cannot make every request fetch the JWKS."""
        idp.ttl = 3600
        auth = JwksAuth.from_discovery(issuer=idp.issuer, audience=AUDIENCE)
        assert auth.verify_token(idp.mint()) is True
        assert idp.jwks_fetches == 1

        forged = pyjwt.encode(
            {"iss": idp.issuer, "aud": AUDIENCE, "exp": int(time.time()) + 60},
            rsa.generate_private_key(public_exponent=65537, key_size=2048),
            algorithm="RS256",
            headers={"kid": "made-up"},
        )
        key_set = auth._client()
        key_set._min_refetch_interval = 0  # isolate the per-kid limit
        assert auth.verify_token(forged) is False
        assert idp.jwks_fetches == 2  # one immediate re-fetch for the new kid
        for _ in range(5):
            assert auth.verify_token(forged) is False
        assert idp.jwks_fetches == 2  # not again for the same kid

        # A genuinely new key still gets its own immediate re-fetch.
        idp.rotate()
        assert auth.verify_token(idp.mint()) is True
        assert idp.jwks_fetches == 3
