"""``JWTAuth`` and ``AsymmetricJWTAuth`` validate ``aud`` and ``iss``.

Regression: the multi-tenancy docs configured ``JWTAuth(secret=..., audience=...)``,
which raised ``TypeError`` — ``JWTAuth`` had no audience parameter and checked
neither ``aud`` nor ``iss``, so a token minted for one service sharing the
secret was accepted by every other.
"""

from __future__ import annotations

import json

import pytest

from promptise.mcp.server import (
    AsymmetricJWTAuth,
    AuthMiddleware,
    JWTAuth,
    MCPServer,
    RequestContext,
    TestClient,
)

SECRET = "audience-test-secret-0123456789abcdef0123"
ISSUER = "https://auth.example.com"


def _unchecked() -> JWTAuth:
    """Signs tokens with the same secret, stamping no defaults."""
    return JWTAuth(secret=SECRET)


class TestJWTAuthAudience:
    def test_matching_audience_is_accepted(self):
        auth = JWTAuth(secret=SECRET, audience="crm")
        assert auth.verify_token(_unchecked().create_token({"sub": "a", "aud": "crm"}))

    def test_token_for_another_service_is_rejected(self):
        auth = JWTAuth(secret=SECRET, audience="crm")
        assert not auth.verify_token(_unchecked().create_token({"sub": "a", "aud": "billing"}))

    def test_token_without_audience_is_rejected(self):
        auth = JWTAuth(secret=SECRET, audience="crm")
        assert not auth.verify_token(_unchecked().create_token({"sub": "a"}))

    def test_list_audiences_on_either_side(self):
        auth = JWTAuth(secret=SECRET, audience=["crm", "crm-staging"])
        assert auth.verify_token(_unchecked().create_token({"aud": ["billing", "crm-staging"]}))
        assert not auth.verify_token(_unchecked().create_token({"aud": ["billing", "hr"]}))

    def test_non_string_audience_claim_is_rejected(self):
        auth = JWTAuth(secret=SECRET, audience="crm")
        assert not auth.verify_token(_unchecked().create_token({"aud": 42}))

    def test_no_audience_configured_keeps_accepting_any(self):
        auth = JWTAuth(secret=SECRET)
        assert auth.verify_token(_unchecked().create_token({"aud": "anything"}))
        assert auth.verify_token(_unchecked().create_token({}))

    @pytest.mark.parametrize("bad", ["", [], [""], [1]])
    def test_invalid_audience_argument(self, bad):
        with pytest.raises(ValueError, match="audience"):
            JWTAuth(secret=SECRET, audience=bad)


class TestJWTAuthIssuer:
    def test_matching_issuer_is_accepted(self):
        auth = JWTAuth(secret=SECRET, issuer=ISSUER)
        assert auth.verify_token(_unchecked().create_token({"iss": ISSUER}))

    def test_other_or_missing_issuer_is_rejected(self):
        auth = JWTAuth(secret=SECRET, issuer=ISSUER)
        assert not auth.verify_token(_unchecked().create_token({"iss": "https://evil.example"}))
        assert not auth.verify_token(_unchecked().create_token({}))

    def test_empty_issuer_argument(self):
        with pytest.raises(ValueError, match="issuer"):
            JWTAuth(secret=SECRET, issuer="")


class TestCreateTokenStampsExpectedClaims:
    def test_tokens_from_a_configured_provider_verify_against_it(self):
        auth = JWTAuth(secret=SECRET, audience=["crm", "crm-staging"], issuer=ISSUER)
        token = auth.create_token({"sub": "a"})
        assert auth.verify_token(token)
        payload = auth._verify_token(token)
        assert payload["aud"] == "crm"
        assert payload["iss"] == ISSUER

    def test_explicit_claims_win(self):
        auth = JWTAuth(secret=SECRET, audience="crm")
        assert not auth.verify_token(auth.create_token({"aud": "billing"}))


class TestThroughTheServer:
    async def test_rejection_is_an_authentication_error_and_claims_reach_the_client(self):
        auth = JWTAuth(secret=SECRET, audience="crm", issuer=ISSUER)
        server = MCPServer("crm", require_auth=True)
        server.add_middleware(AuthMiddleware(auth))

        @server.tool()
        async def whoami(ctx: RequestContext) -> dict:
            """Caller."""
            return {"aud": ctx.client.audience, "iss": ctx.client.issuer}

        good = TestClient(server, meta={"authorization": f"Bearer {auth.create_token({})}"})
        assert json.loads((await good.call_tool("whoami"))[0].text) == {
            "aud": "crm",
            "iss": ISSUER,
        }

        foreign = _unchecked().create_token({"aud": "billing", "iss": ISSUER})
        bad = TestClient(server, meta={"authorization": f"Bearer {foreign}"})
        error = json.loads((await bad.call_tool("whoami"))[0].text)["error"]
        assert error["code"] == "AUTHENTICATION_ERROR"
        assert "audience" in error["message"]


@pytest.fixture(scope="module")
def keys():
    pytest.importorskip("cryptography")
    jwt = pytest.importorskip("jwt")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_pem = (
        private.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )

    def sign(claims: dict) -> str:
        return jwt.encode({"exp": 2_000_000_000, **claims}, private, algorithm="RS256")

    return public_pem, sign


class TestAsymmetricJWTAuth:
    def test_audience_and_issuer_are_enforced(self, keys):
        public_pem, sign = keys
        auth = AsymmetricJWTAuth(public_pem, audience="crm", issuer=ISSUER)
        assert auth.verify_token(sign({"aud": "crm", "iss": ISSUER}))
        assert not auth.verify_token(sign({"aud": "billing", "iss": ISSUER}))
        assert not auth.verify_token(sign({"aud": "crm", "iss": "https://evil.example"}))
        assert not auth.verify_token(sign({"iss": ISSUER}))

    def test_audience_lets_idp_tokens_with_aud_through(self, keys):
        public_pem, sign = keys
        # Without an expected audience PyJWT refuses any token carrying aud.
        assert not AsymmetricJWTAuth(public_pem).verify_token(sign({"aud": "crm"}))
        assert AsymmetricJWTAuth(public_pem, audience="crm").verify_token(sign({"aud": "crm"}))
