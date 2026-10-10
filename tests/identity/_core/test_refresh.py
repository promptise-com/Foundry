"""Credential refresh semantics of the identity providers.

Regressions covered:

* A credential that lives less than twice the refresh buffer was treated as
  stale from the moment it was acquired, so it was re-acquired on every
  request (a round trip to the IdP per MCP request).
* ``get_credential(None)`` and ``get_credential(<default audience>)`` kept
  two cache entries and fetched the same credential twice.
* Passive providers (projected token file, OIDC) cached one entry per
  requested audience although their token has a fixed audience.
* There was no way to bypass the cache after a resource rejected a
  credential before its ``exp`` (``force_refresh``).
"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import httpx
import pytest

from promptise.identity import (
    AgentIdentity,
    CachedCredential,
    CallableTokenProvider,
    CredentialAcquisitionError,
    EntraManagedIdentityProvider,
)
from promptise.identity._core.cache import CREDENTIAL_REFRESH_BUFFER_SECONDS
from promptise.identity._core.file_provider import FileTokenProvider
from promptise.identity.providers.oidc import OidcCallableProvider


def _jwt(exp: float, n: int = 0) -> str:
    def b64(obj: object) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    return f"{b64({'alg': 'none'})}.{b64({'sub': 'bot', 'exp': exp, 'n': n})}.sig"


class _Minter:
    def __init__(self, ttl: float = 3600) -> None:
        self.ttl = ttl
        self.audiences: list[str | None] = []

    def __call__(self, audience: str | None = None) -> str:
        self.audiences.append(audience)
        return _jwt(time.time() + self.ttl, n=len(self.audiences))


# -- Short-lived credentials --------------------------------------------


def test_short_lived_credential_is_reused_for_half_its_lifetime() -> None:
    now = time.time()
    lifetime = CREDENTIAL_REFRESH_BUFFER_SECONDS  # shorter than twice the buffer
    cred = CachedCredential(token="t", expires_at_epoch=now + lifetime, acquired_at_epoch=now)
    assert cred.is_stale() is False
    # Past half its lifetime it is renewed.
    late = CachedCredential(
        token="t",
        expires_at_epoch=now + lifetime * 0.4,
        acquired_at_epoch=now - lifetime * 0.6,
    )
    assert late.is_stale() is True


def test_short_lived_credential_is_not_reacquired_on_every_call() -> None:
    mint = _Minter(ttl=30)  # well under twice the 60s refresh buffer
    provider = CallableTokenProvider(token_fn=mint)
    first = provider.get_credential()
    assert provider.get_credential() == first
    assert len(mint.audiences) == 1


# -- Default audience ----------------------------------------------------


def test_default_audience_and_none_share_one_credential() -> None:
    mint = _Minter()
    provider = CallableTokenProvider(token_fn=mint, default_audience="api://tools")
    first = provider.get_credential()
    assert provider.get_credential("api://tools") == first
    assert mint.audiences == ["api://tools"]  # minted once, for the explicit audience
    assert provider.default_audience == "api://tools"


def test_other_audiences_still_get_their_own_credential() -> None:
    mint = _Minter()
    provider = CallableTokenProvider(token_fn=mint, default_audience="api://tools")
    provider.get_credential()
    provider.get_credential("api://other")
    assert mint.audiences == ["api://tools", "api://other"]


# -- force_refresh -------------------------------------------------------


def test_force_refresh_bypasses_a_valid_cached_credential() -> None:
    mint = _Minter()
    provider = CallableTokenProvider(token_fn=mint)
    first = provider.get_credential("api://a")
    refreshed = provider.get_credential("api://a", force_refresh=True)
    assert refreshed != first
    # The refreshed credential replaces the cached one.
    assert provider.get_credential("api://a") == refreshed
    assert len(mint.audiences) == 2


def test_agent_identity_forwards_force_refresh() -> None:
    mint = _Minter()
    identity = AgentIdentity("bot", credential=CallableTokenProvider(token_fn=mint))
    first = identity.get_credential("api://a")
    assert identity.get_credential("api://a", force_refresh=True) != first


# -- Passive providers ---------------------------------------------------


def test_projected_file_is_read_once_for_any_audience(tmp_path: Path) -> None:
    token_file = tmp_path / "token"
    token_file.write_text(_jwt(time.time() + 3600), encoding="utf-8")
    provider = FileTokenProvider(token_file=token_file)
    first = provider.get_credential("api://a")
    token_file.write_text(_jwt(time.time() + 3600, n=2), encoding="utf-8")
    # Same fixed-audience token for every audience: served from one entry.
    assert provider.get_credential("api://b") == first
    assert provider.get_credential() == first


def test_oidc_callable_is_called_once_for_any_audience() -> None:
    calls: list[int] = []

    def token_fn() -> str:
        calls.append(1)
        return _jwt(time.time() + 3600, n=len(calls))

    provider = OidcCallableProvider(issuer="https://ci", token_fn=token_fn)
    provider.get_credential("api://a")
    provider.get_credential("api://b")
    assert len(calls) == 1


# -- No credential in error messages -------------------------------------


def test_entra_non_json_body_is_not_echoed(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = _jwt(time.time() + 3600)

    def mocked_get(url: str, **kwargs: object) -> httpx.Response:
        return httpx.Response(200, text=secret, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", mocked_get)
    provider = EntraManagedIdentityProvider()
    with pytest.raises(CredentialAcquisitionError, match="non-JSON") as info:
        provider.get_credential()
    message = str(info.value)
    assert secret not in message
    assert secret[:20] not in message
