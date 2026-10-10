"""Tests for promptise.server authentication providers and middleware."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from promptise.mcp.server._auth import (
    APIKeyAuth,
    AuthMiddleware,
    JwksAuth,
    JWTAuth,
)
from promptise.mcp.server._context import RequestContext
from promptise.mcp.server._errors import AuthenticationError
from promptise.mcp.server._types import ToolDef

# =====================================================================
# JWTAuth
# =====================================================================


class TestJWTAuth:
    def setup_method(self):
        self.auth = JWTAuth(secret="test-secret-key")

    async def test_authenticate_valid_token(self):
        token = self.auth.create_token({"sub": "user-42"})
        ctx = RequestContext(
            server_name="test",
            meta={"authorization": f"Bearer {token}"},
        )
        client_id = await self.auth.authenticate(ctx)
        assert client_id == "user-42"

    async def test_authenticate_without_bearer_prefix(self):
        token = self.auth.create_token({"sub": "user-42"})
        ctx = RequestContext(
            server_name="test",
            meta={"authorization": token},
        )
        client_id = await self.auth.authenticate(ctx)
        assert client_id == "user-42"

    async def test_authenticate_missing_token(self):
        ctx = RequestContext(server_name="test", meta={})
        with pytest.raises(AuthenticationError, match="Missing"):
            await self.auth.authenticate(ctx)

    async def test_authenticate_malformed_token(self):
        ctx = RequestContext(
            server_name="test",
            meta={"authorization": "not.a.jwt.at.all"},
        )
        with pytest.raises(AuthenticationError):
            await self.auth.authenticate(ctx)

    async def test_authenticate_invalid_signature(self):
        other_auth = JWTAuth(secret="different-secret")
        token = other_auth.create_token({"sub": "user"})
        ctx = RequestContext(
            server_name="test",
            meta={"authorization": f"Bearer {token}"},
        )
        with pytest.raises(AuthenticationError, match="signature"):
            await self.auth.authenticate(ctx)

    async def test_authenticate_expired_token(self):
        token = self.auth.create_token({"sub": "user"}, expires_in=-10)
        ctx = RequestContext(
            server_name="test",
            meta={"authorization": f"Bearer {token}"},
        )
        with pytest.raises(AuthenticationError, match="expired"):
            await self.auth.authenticate(ctx)

    async def test_create_token_includes_expiry(self):
        token = self.auth.create_token({"sub": "user"}, expires_in=3600)
        # Token should be verifiable
        ctx = RequestContext(
            server_name="test",
            meta={"authorization": token},
        )
        client_id = await self.auth.authenticate(ctx)
        assert client_id == "user"

    async def test_custom_meta_key(self):
        auth = JWTAuth(secret="secret", meta_key="x-token")
        token = auth.create_token({"sub": "user"})
        ctx = RequestContext(
            server_name="test",
            meta={"x-token": token},
        )
        client_id = await auth.authenticate(ctx)
        assert client_id == "user"

    async def test_client_id_fallback(self):
        token = self.auth.create_token({"client_id": "svc-1"})
        ctx = RequestContext(
            server_name="test",
            meta={"authorization": token},
        )
        client_id = await self.auth.authenticate(ctx)
        assert client_id == "svc-1"


# =====================================================================
# APIKeyAuth
# =====================================================================


class TestAPIKeyAuth:
    def setup_method(self):
        self.auth = APIKeyAuth(
            keys={
                "key-abc-123": "client-a",
                "key-def-456": "client-b",
            }
        )

    async def test_authenticate_valid_key(self):
        ctx = RequestContext(
            server_name="test",
            meta={"x-api-key": "key-abc-123"},
        )
        client_id = await self.auth.authenticate(ctx)
        assert client_id == "client-a"

    async def test_authenticate_missing_key(self):
        ctx = RequestContext(server_name="test", meta={})
        with pytest.raises(AuthenticationError, match="Missing"):
            await self.auth.authenticate(ctx)

    async def test_authenticate_invalid_key(self):
        ctx = RequestContext(
            server_name="test",
            meta={"x-api-key": "wrong-key"},
        )
        with pytest.raises(AuthenticationError, match="Invalid"):
            await self.auth.authenticate(ctx)

    async def test_custom_meta_key(self):
        auth = APIKeyAuth(
            keys={"my-key": "client"},
            header="api-token",
        )
        ctx = RequestContext(
            server_name="test",
            meta={"api-token": "my-key"},
        )
        client_id = await auth.authenticate(ctx)
        assert client_id == "client"


# =====================================================================
# AuthMiddleware
# =====================================================================


class TestAuthMiddleware:
    async def test_skips_non_auth_tools(self):
        auth = APIKeyAuth(keys={"k": "c"})
        mw = AuthMiddleware(auth)

        tdef = ToolDef(
            name="public",
            description="",
            handler=lambda: None,
            input_schema={},
            auth=False,
        )
        ctx = RequestContext(server_name="test", tool_name="public")
        ctx.state["tool_def"] = tdef

        called = False

        async def call_next(ctx):
            nonlocal called
            called = True
            return "ok"

        result = await mw(ctx, call_next)
        assert result == "ok"
        assert called
        assert ctx.client_id is None

    async def test_authenticates_auth_tools(self):
        auth = APIKeyAuth(keys={"key-1": "client-1"})
        mw = AuthMiddleware(auth)

        tdef = ToolDef(
            name="private",
            description="",
            handler=lambda: None,
            input_schema={},
            auth=True,
        )
        ctx = RequestContext(server_name="test", tool_name="private")
        ctx.state["tool_def"] = tdef
        ctx.meta["x-api-key"] = "key-1"

        async def call_next(ctx):
            return "secret"

        result = await mw(ctx, call_next)
        assert result == "secret"
        assert ctx.client_id == "client-1"

    async def test_rejects_auth_tools_without_creds(self):
        auth = APIKeyAuth(keys={"k": "c"})
        mw = AuthMiddleware(auth)

        tdef = ToolDef(
            name="private",
            description="",
            handler=lambda: None,
            input_schema={},
            auth=True,
        )
        ctx = RequestContext(server_name="test", tool_name="private")
        ctx.state["tool_def"] = tdef

        async def call_next(ctx):
            return "secret"

        with pytest.raises(AuthenticationError):
            await mw(ctx, call_next)


# =====================================================================
# JwksAuth — verify IdP-issued agent tokens against a JWKS
# =====================================================================

_JWKS_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_JWKS_PUBLIC_KEY = _JWKS_PRIVATE_KEY.public_key()
_OTHER_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _mint(claims: dict, *, key=None) -> str:
    return pyjwt.encode(claims, key or _JWKS_PRIVATE_KEY, algorithm="RS256", headers={"kid": "k1"})


class _StubJwkClient:
    """Stand-in for the JWKS key set — returns a fixed public key."""

    def __init__(self, public_key, *, fail: bool = False) -> None:
        self._public_key = public_key
        self._fail = fail

    def get_signing_key_from_jwt(self, token: str):
        if self._fail:
            raise RuntimeError("no matching kid in JWKS")
        return SimpleNamespace(key=self._public_key)


def _jwks_auth(*, audience="api://mcp", issuer=None, fail_keys=False) -> JwksAuth:
    auth = JwksAuth(jwks_url="https://idp.example.com/jwks", audience=audience, issuer=issuer)
    auth._jwk_client = _StubJwkClient(_JWKS_PUBLIC_KEY, fail=fail_keys)
    return auth


class TestJwksAuth:
    async def test_valid_token_surfaces_subject(self) -> None:
        auth = _jwks_auth(audience="api://mcp", issuer="https://idp")
        token = _mint(
            {"sub": "spiffe://acme/billing-bot", "aud": "api://mcp", "iss": "https://idp"}
        )
        ctx = RequestContext(server_name="t", meta={"authorization": f"Bearer {token}"})
        assert await auth.authenticate(ctx) == "spiffe://acme/billing-bot"
        assert ctx.state["_jwt_payload"]["sub"] == "spiffe://acme/billing-bot"

    async def test_missing_token_raises(self) -> None:
        ctx = RequestContext(server_name="t", meta={})
        with pytest.raises(AuthenticationError, match="Missing"):
            await _jwks_auth().authenticate(ctx)

    async def test_wrong_audience_rejected(self) -> None:
        auth = _jwks_auth(audience="api://mcp")
        token = _mint({"sub": "bot", "aud": "api://other"})
        ctx = RequestContext(server_name="t", meta={"authorization": f"Bearer {token}"})
        with pytest.raises(AuthenticationError, match="Invalid JWT"):
            await auth.authenticate(ctx)

    async def test_wrong_issuer_rejected(self) -> None:
        auth = _jwks_auth(issuer="https://idp")
        token = _mint({"sub": "bot", "aud": "api://mcp", "iss": "https://evil"})
        ctx = RequestContext(server_name="t", meta={"authorization": f"Bearer {token}"})
        with pytest.raises(AuthenticationError, match="Invalid JWT"):
            await auth.authenticate(ctx)

    async def test_audience_is_required(self) -> None:
        with pytest.raises(ValueError, match="non-empty audience"):
            JwksAuth(jwks_url="https://idp/jwks", audience="")

    async def test_expired_token_rejected(self) -> None:
        # Expired well beyond the clock-skew leeway (default 60s).
        auth = _jwks_auth()
        token = _mint({"sub": "bot", "aud": "api://mcp", "exp": int(time.time()) - 300})
        ctx = RequestContext(server_name="t", meta={"authorization": f"Bearer {token}"})
        with pytest.raises(AuthenticationError, match="expired"):
            await auth.authenticate(ctx)

    async def test_token_within_leeway_still_verifies(self) -> None:
        # A token a few seconds past exp must still verify — small NTP skew
        # between the IdP and this server should not reject a valid caller.
        auth = _jwks_auth()  # default leeway = 60s
        token = _mint({"sub": "bot", "aud": "api://mcp", "exp": int(time.time()) - 15})
        ctx = RequestContext(server_name="t", meta={"authorization": f"Bearer {token}"})
        assert await auth.authenticate(ctx) == "bot"

    async def test_leeway_is_configurable_to_zero(self) -> None:
        # Opt out of skew tolerance: leeway=0 rejects any past-exp token.
        auth = JwksAuth(jwks_url="https://idp/jwks", audience="api://mcp", leeway=0)
        auth._jwk_client = _StubJwkClient(_JWKS_PUBLIC_KEY)
        token = _mint({"sub": "bot", "aud": "api://mcp", "exp": int(time.time()) - 5})
        ctx = RequestContext(server_name="t", meta={"authorization": f"Bearer {token}"})
        with pytest.raises(AuthenticationError, match="expired"):
            await auth.authenticate(ctx)

    async def test_bad_signature_rejected(self) -> None:
        # Token signed by a different key than the JWKS returns.
        auth = _jwks_auth()
        token = _mint({"sub": "bot"}, key=_OTHER_KEY)
        ctx = RequestContext(server_name="t", meta={"authorization": f"Bearer {token}"})
        with pytest.raises(AuthenticationError, match="Invalid JWT"):
            await auth.authenticate(ctx)

    async def test_unresolvable_key_rejected(self) -> None:
        auth = _jwks_auth(fail_keys=True)
        token = _mint({"sub": "bot"})
        ctx = RequestContext(server_name="t", meta={"authorization": f"Bearer {token}"})
        with pytest.raises(AuthenticationError, match="Could not resolve a signing key"):
            await auth.authenticate(ctx)

    def test_verify_token_bool(self) -> None:
        auth = _jwks_auth()
        assert auth.verify_token(_mint({"sub": "bot", "aud": "api://mcp"})) is True
        assert auth.verify_token(_mint({"sub": "bot", "aud": "api://mcp"}, key=_OTHER_KEY)) is False

    async def test_middleware_surfaces_agent_identity(self) -> None:
        """The validated agent identity (sub + claims) lands on ctx.client so
        the server knows which agent called."""
        auth = _jwks_auth(audience="api://mcp")
        mw = AuthMiddleware(auth)
        tdef = ToolDef(
            name="private",
            description="",
            handler=lambda: None,
            input_schema={},
            auth=True,
        )
        token = _mint({"sub": "agent-billing", "aud": "api://mcp", "iss": "https://idp"})
        ctx = RequestContext(server_name="t", tool_name="private")
        ctx.state["tool_def"] = tdef
        ctx.meta["authorization"] = f"Bearer {token}"

        async def call_next(ctx: RequestContext) -> str:
            return "ok"

        assert await mw(ctx, call_next) == "ok"
        assert ctx.client_id == "agent-billing"
        assert ctx.client.subject == "agent-billing"
        assert ctx.client.issuer == "https://idp"


# =====================================================================
# JwksAuth.from_discovery — resolve the JWKS via OIDC discovery
# =====================================================================

_ISSUER = "https://idp.example.com"


def _discovery_get(*, issuer=_ISSUER, jwks_uri="https://idp.example.com/keys", status=200):
    def _get(url, **kwargs):
        assert url == f"{_ISSUER}/.well-known/openid-configuration"
        if status != 200:
            return httpx.Response(status, text="nope")
        body = {"issuer": issuer}
        if jwks_uri is not None:
            body["jwks_uri"] = jwks_uri
        return httpx.Response(200, json=body)

    return _get


def _stub_jwk_factory(url, **kwargs):
    return _StubJwkClient(_JWKS_PUBLIC_KEY)


def _disc_ctx(claims=None):
    claims = claims or {"sub": "agent-x", "aud": "api://mcp", "iss": _ISSUER}
    token = _mint(claims)
    return RequestContext(server_name="t", meta={"authorization": f"Bearer {token}"})


class TestJwksDiscovery:
    async def test_resolves_jwks_and_verifies(self) -> None:
        auth = JwksAuth.from_discovery(issuer=_ISSUER, audience="api://mcp")
        with (
            patch.object(httpx, "get", _discovery_get()),
            patch("promptise.mcp.server._auth._JwksKeySet", _stub_jwk_factory),
        ):
            assert await auth.authenticate(_disc_ctx()) == "agent-x"
        assert auth._jwks_url == "https://idp.example.com/keys"  # cached

    async def test_issuer_mismatch_rejected(self) -> None:
        auth = JwksAuth.from_discovery(issuer=_ISSUER, audience="api://mcp")
        with (
            patch.object(httpx, "get", _discovery_get(issuer="https://evil")),
            patch("promptise.mcp.server._auth._JwksKeySet", _stub_jwk_factory),
        ):
            with pytest.raises(AuthenticationError, match="issuer mismatch"):
                await auth.authenticate(_disc_ctx())

    async def test_missing_jwks_uri_rejected(self) -> None:
        auth = JwksAuth.from_discovery(issuer=_ISSUER, audience="api://mcp")
        with patch.object(httpx, "get", _discovery_get(jwks_uri=None)):
            with pytest.raises(AuthenticationError, match="jwks_uri"):
                await auth.authenticate(_disc_ctx())

    async def test_discovery_unreachable(self) -> None:
        auth = JwksAuth.from_discovery(issuer=_ISSUER, audience="api://mcp")

        def _boom(url, **kwargs):
            raise httpx.ConnectError("no route")

        with patch.object(httpx, "get", _boom):
            with pytest.raises(AuthenticationError, match="Could not fetch OIDC discovery"):
                await auth.authenticate(_disc_ctx())

    async def test_discovery_non_200(self) -> None:
        auth = JwksAuth.from_discovery(issuer=_ISSUER, audience="api://mcp")
        with patch.object(httpx, "get", _discovery_get(status=503)):
            with pytest.raises(AuthenticationError, match="HTTP 503"):
                await auth.authenticate(_disc_ctx())

    async def test_discovery_fetched_once(self) -> None:
        auth = JwksAuth.from_discovery(issuer=_ISSUER, audience="api://mcp")
        calls = {"n": 0}

        def _counting_get(url, **kwargs):
            calls["n"] += 1
            return httpx.Response(
                200, json={"issuer": _ISSUER, "jwks_uri": "https://idp.example.com/keys"}
            )

        with (
            patch.object(httpx, "get", _counting_get),
            patch("promptise.mcp.server._auth._JwksKeySet", _stub_jwk_factory),
        ):
            await auth.authenticate(_disc_ctx())
            await auth.authenticate(_disc_ctx())
        assert calls["n"] == 1  # discovery + client are cached after first use

    def test_requires_issuer(self) -> None:
        with pytest.raises(ValueError, match="non-empty issuer"):
            JwksAuth.from_discovery(issuer="", audience="api://mcp")

    def test_requires_audience(self) -> None:
        with pytest.raises(ValueError, match="non-empty audience"):
            JwksAuth.from_discovery(issuer=_ISSUER, audience="")


# =====================================================================
# JWKS key set: rotation, rate limits, IdP outages
# =====================================================================


def _rsa_jwk(kid: str, *, use: str = "sig") -> tuple[object, dict]:
    import json as _json

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = _json.loads(pyjwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk.update({"kid": kid, "use": use, "alg": "RS256"})
    return key, jwk


class _FakeJwksEndpoint:
    def __init__(self, *jwks: dict) -> None:
        self.keys = list(jwks)
        self.fetches = 0
        self.down = False

    def get(self, url: str, **kwargs: object) -> httpx.Response:
        self.fetches += 1
        if self.down:
            raise httpx.ConnectError("IdP unreachable")
        return httpx.Response(200, json={"keys": self.keys})


class TestJwksKeySet:
    def _key_set(self, endpoint: _FakeJwksEndpoint, monkeypatch, **kwargs):
        from promptise.mcp.server._auth import _JwksKeySet

        monkeypatch.setattr(httpx, "get", endpoint.get)
        return _JwksKeySet("https://idp.example.com/keys", timeout=1.0, **kwargs)

    def test_rotated_key_is_fetched_immediately(self, monkeypatch) -> None:
        _, old = _rsa_jwk("old")
        endpoint = _FakeJwksEndpoint(old)
        keys = self._key_set(endpoint, monkeypatch)
        assert keys.get_signing_key("old").key_id == "old"
        _, new = _rsa_jwk("new")
        endpoint.keys.append(new)
        assert keys.get_signing_key("new").key_id == "new"
        assert endpoint.fetches == 2

    def test_unknown_kids_share_a_global_refetch_limit(self, monkeypatch) -> None:
        _, jwk = _rsa_jwk("k1")
        endpoint = _FakeJwksEndpoint(jwk)
        keys = self._key_set(endpoint, monkeypatch, min_refetch_interval=60.0)
        keys.get_signing_key("k1")
        for kid in ("a", "b", "c", "d"):
            with pytest.raises(LookupError):
                keys.get_signing_key(kid)
        # One re-fetch for the first unknown kid; the others are throttled.
        assert endpoint.fetches == 2

    def test_outage_keeps_verifying_with_cached_keys(self, monkeypatch, caplog) -> None:
        _, jwk = _rsa_jwk("k1")
        endpoint = _FakeJwksEndpoint(jwk)
        keys = self._key_set(endpoint, monkeypatch, lifespan=0.0)
        keys.get_signing_key("k1")
        endpoint.down = True
        # The set is due for a refresh, the IdP is down: the cached key serves.
        assert keys.get_signing_key("k1").key_id == "k1"
        assert any("using the cached keys" in r.getMessage() for r in caplog.records)

    def test_first_fetch_failure_is_an_error(self, monkeypatch) -> None:
        endpoint = _FakeJwksEndpoint()
        endpoint.down = True
        keys = self._key_set(endpoint, monkeypatch)
        with pytest.raises(httpx.ConnectError):
            keys.get_signing_key("k1")

    def test_encryption_keys_are_not_signing_keys(self, monkeypatch) -> None:
        _, enc = _rsa_jwk("k1", use="enc")
        keys = self._key_set(_FakeJwksEndpoint(enc), monkeypatch)
        with pytest.raises(LookupError):
            keys.get_signing_key("k1")

    async def test_jwks_auth_accepts_a_token_signed_with_a_rotated_key(self, monkeypatch) -> None:
        old_key, old = _rsa_jwk("old")
        endpoint = _FakeJwksEndpoint(old)
        monkeypatch.setattr(httpx, "get", endpoint.get)
        auth = JwksAuth(jwks_url="https://idp.example.com/keys", audience="api://mcp")

        def mint(key: object, kid: str) -> str:
            claims = {"sub": "agent-x", "aud": "api://mcp", "exp": int(time.time()) + 60}
            return pyjwt.encode(claims, key, algorithm="RS256", headers={"kid": kid})

        assert auth.verify_token(mint(old_key, "old")) is True
        new_key, new = _rsa_jwk("new")
        endpoint.keys = [new]
        assert auth.verify_token(mint(new_key, "new")) is True
        # Garbage kids do not turn every request into a JWKS fetch.
        fetches = endpoint.fetches
        forged = mint(rsa.generate_private_key(public_exponent=65537, key_size=2048), "forged")
        for _ in range(10):
            assert auth.verify_token(forged) is False
        assert endpoint.fetches <= fetches + 1
