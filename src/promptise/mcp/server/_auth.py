"""Authentication providers for MCP server tools.

Provides pluggable auth that integrates as middleware.  After
authentication, the :class:`~._context.ClientContext` on the request
is fully populated with client identity, roles, scopes, standard JWT
claims, IP address, and user-agent.

Example::

    from promptise.mcp.server import MCPServer, AuthMiddleware, JWTAuth

    auth = JWTAuth(secret="my-secret")
    server = MCPServer(name="secure-api")
    server.add_middleware(AuthMiddleware(auth))

    @server.tool(auth=True)
    async def secret_data(ctx: RequestContext) -> str:
        print(ctx.client.client_id)   # "agent-007"
        print(ctx.client.roles)       # {"admin"}
        print(ctx.client.scopes)      # {"read", "write"}
        print(ctx.client.issuer)      # "https://auth.example.com"
        print(ctx.client.ip_address)  # "10.0.0.1"
        return "classified"
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from ._context import ClientContext, RequestContext, get_request_client_info
from ._errors import AuthenticationError

logger = logging.getLogger("promptise.server")

# Type alias for the on_authenticate enrichment hook.
OnAuthenticateHook = Callable[[ClientContext, RequestContext], Awaitable[None] | None]


@runtime_checkable
class AuthProvider(Protocol):
    """Protocol for authentication providers."""

    async def authenticate(self, ctx: RequestContext) -> str:
        """Authenticate the request.

        Returns:
            Client identifier string on success.

        Raises:
            AuthenticationError: On authentication failure.
        """
        ...


# =====================================================================
# Token verification cache
# =====================================================================


class _TokenCache:
    """LRU cache for verified JWT payloads with TTL expiry.

    Avoids re-computing HMAC-SHA256 + base64 decoding on every request
    when the same token is reused (which is the common case for agents).

    Thread-safe for use in multi-worker uvicorn deployments.

    Args:
        max_size: Maximum cached tokens.  Oldest entries evicted first.
    """

    def __init__(self, max_size: int = 256) -> None:
        self._store: OrderedDict[str, tuple[dict[str, Any], float]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, token: str) -> dict[str, Any] | None:
        """Return cached payload if valid, else ``None``.

        The read path is lock-free for performance under high concurrency.
        Python's GIL guarantees ``dict.get()`` atomicity.  Best-effort
        LRU reordering and expiry eviction are wrapped in try/except to
        tolerate benign races with concurrent ``put()`` calls.
        """
        entry = self._store.get(token)
        if entry is None:
            return None
        payload, cached_at = entry
        # Honour JWT expiry — evict if token has since expired
        exp = payload.get("exp")
        if exp is not None and exp < time.time():
            try:
                del self._store[token]
            except KeyError:
                pass  # Concurrent eviction — benign
            return None
        # Best-effort LRU reordering (race with put is harmless)
        try:
            self._store.move_to_end(token)
        except KeyError:
            pass  # Concurrent eviction — benign
        return payload

    def put(self, token: str, payload: dict[str, Any]) -> None:
        """Cache a verified payload."""
        with self._lock:
            if token in self._store:
                self._store.move_to_end(token)
                self._store[token] = (payload, time.time())
                return
            if len(self._store) >= self._max_size:
                self._store.popitem(last=False)  # evict oldest
            self._store[token] = (payload, time.time())

    def invalidate(self, token: str) -> None:
        """Remove a specific token from the cache."""
        with self._lock:
            self._store.pop(token, None)

    def clear(self) -> None:
        """Remove all cached tokens."""
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


class JWTAuth:
    """JWT-based authentication provider.

    Validates JWT tokens from the request metadata using HMAC-SHA256.
    Verified tokens are cached in an LRU to avoid repeated crypto
    operations on the hot path.

    Args:
        secret: Shared secret for HS256 signature verification.
        meta_key: Key in ``ctx.meta`` where the token is expected.
        cache_size: Max number of verified tokens to cache (0 to disable).
    """

    def __init__(
        self,
        secret: str,
        *,
        meta_key: str = "authorization",
        cache_size: int = 256,
    ) -> None:
        self._secret = secret.encode()
        self._meta_key = meta_key
        self._cache = _TokenCache(max_size=cache_size) if cache_size > 0 else None

    async def authenticate(self, ctx: RequestContext) -> str:
        token = ctx.meta.get(self._meta_key, "")
        if token.startswith("Bearer "):
            token = token[7:]
        if not token:
            raise AuthenticationError("Missing authentication token")

        # Fast path: return cached payload without crypto
        if self._cache is not None:
            cached_payload = self._cache.get(token)
            if cached_payload is not None:
                ctx.state["_jwt_payload"] = cached_payload
                return cached_payload.get("sub", cached_payload.get("client_id", "unknown"))

        payload = self._verify_token(token)

        # Cache the verified payload
        if self._cache is not None:
            self._cache.put(token, payload)

        # Store full payload for downstream use (e.g. role extraction)
        ctx.state["_jwt_payload"] = payload
        return payload.get("sub", payload.get("client_id", "unknown"))

    def _verify_token(self, token: str) -> dict[str, Any]:
        """Verify and decode a HS256 JWT token."""
        parts = token.split(".")
        if len(parts) != 3:
            raise AuthenticationError("Malformed JWT token")

        header_b64, payload_b64, signature_b64 = parts

        # Validate header algorithm — reject tokens that claim anything other
        # than HS256 to prevent algorithm confusion attacks.
        try:
            header_json = base64.urlsafe_b64decode(header_b64 + "==")
            header = json.loads(header_json)
        except Exception:
            raise AuthenticationError("Invalid JWT header")

        if header.get("alg") != "HS256":
            raise AuthenticationError(
                f"Unsupported JWT algorithm: {header.get('alg')!r}. Only HS256 is accepted."
            )

        # Verify signature
        signing_input = f"{header_b64}.{payload_b64}".encode()
        expected_sig = hmac.new(self._secret, signing_input, hashlib.sha256).digest()

        try:
            actual_sig = base64.urlsafe_b64decode(signature_b64 + "==")
        except Exception:
            raise AuthenticationError("Invalid JWT signature encoding")

        if not hmac.compare_digest(expected_sig, actual_sig):
            raise AuthenticationError("Invalid JWT signature")

        # Decode payload
        try:
            payload_json = base64.urlsafe_b64decode(payload_b64 + "==")
            payload = json.loads(payload_json)
        except Exception:
            raise AuthenticationError("Invalid JWT payload")

        # Check expiry — require exp claim to prevent indefinite tokens
        now = time.time()
        if "exp" not in payload:
            raise AuthenticationError(
                "Token missing 'exp' (expiry) claim. Tokens without expiry are not accepted.",
                suggestion="Include an 'exp' claim when generating JWT tokens.",
            )
        if payload["exp"] < now:
            raise AuthenticationError(
                "Token expired",
                suggestion="Request a new authentication token",
            )

        # Check not-before
        if "nbf" in payload and payload["nbf"] > now:
            raise AuthenticationError("Token not yet valid")

        return payload

    def verify_token(self, token: str) -> bool:
        """Check if a token is valid without requiring a request context.

        Useful for transport-level auth gating.
        """
        try:
            self._verify_token(token)
            return True
        except AuthenticationError:
            return False

    def create_token(self, payload: dict[str, Any], *, expires_in: int = 3600) -> str:
        """Create a signed JWT token (utility for testing).

        Args:
            payload: Claims to include in the token.
            expires_in: Token lifetime in seconds.
        """
        header = (
            base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
            .rstrip(b"=")
            .decode()
        )

        full_payload = {**payload, "exp": int(time.time()) + expires_in}
        payload_b64 = (
            base64.urlsafe_b64encode(json.dumps(full_payload).encode()).rstrip(b"=").decode()
        )

        signing_input = f"{header}.{payload_b64}".encode()
        signature = hmac.new(self._secret, signing_input, hashlib.sha256).digest()
        sig_b64 = base64.urlsafe_b64encode(signature).rstrip(b"=").decode()

        return f"{header}.{payload_b64}.{sig_b64}"


class AsymmetricJWTAuth:
    """JWT authentication using asymmetric algorithms (RS256, ES256).

    Validates JWT tokens signed with RSA or ECDSA keys.  Requires the
    ``PyJWT`` and ``cryptography`` packages (optional dependencies).

    Args:
        public_key: PEM-encoded public key string, or path to a PEM
            file.  Used for signature verification.
        algorithm: JWT algorithm (``"RS256"`` or ``"ES256"``).
        meta_key: Key in ``ctx.meta`` where the token is expected.
        cache_size: Max cached tokens (0 to disable).

    Example::

        auth = AsymmetricJWTAuth(
            public_key=open("public.pem").read(),
            algorithm="RS256",
        )
        server.add_middleware(AuthMiddleware(auth))
    """

    def __init__(
        self,
        public_key: str,
        *,
        algorithm: str = "RS256",
        meta_key: str = "authorization",
        cache_size: int = 256,
    ) -> None:
        if algorithm not in ("RS256", "ES256"):
            raise ValueError(f"Unsupported algorithm: {algorithm}. Use RS256 or ES256.")

        self._algorithm = algorithm
        self._meta_key = meta_key
        self._cache = _TokenCache(max_size=cache_size) if cache_size > 0 else None

        # Load public key
        import pathlib

        key_str = public_key.strip()
        if not key_str.startswith("-----"):
            path = pathlib.Path(key_str)
            if path.exists():
                key_str = path.read_text().strip()

        self._public_key_pem = key_str

    async def authenticate(self, ctx: RequestContext) -> str:
        """Authenticate using an asymmetric JWT token."""
        token = ctx.meta.get(self._meta_key, "")
        if token.startswith("Bearer "):
            token = token[7:]
        if not token:
            raise AuthenticationError("Missing authentication token")

        if self._cache is not None:
            cached_payload = self._cache.get(token)
            if cached_payload is not None:
                ctx.state["_jwt_payload"] = cached_payload
                return cached_payload.get("sub", cached_payload.get("client_id", "unknown"))

        payload = self._verify_token(token)

        if self._cache is not None:
            self._cache.put(token, payload)

        ctx.state["_jwt_payload"] = payload
        return payload.get("sub", payload.get("client_id", "unknown"))

    def _verify_token(self, token: str) -> dict[str, Any]:
        """Verify and decode a JWT using PyJWT."""
        try:
            import jwt as pyjwt
        except ImportError:
            raise ImportError(
                "PyJWT and cryptography are required for asymmetric JWT. "
                "Install with: pip install PyJWT cryptography"
            )

        try:
            payload = pyjwt.decode(
                token,
                self._public_key_pem,
                algorithms=[self._algorithm],
            )
            return payload
        except pyjwt.ExpiredSignatureError:
            raise AuthenticationError(
                "Token expired",
                suggestion="Request a new authentication token",
            )
        except pyjwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid JWT: {e}")

    def verify_token(self, token: str) -> bool:
        """Check if a token is valid without requiring a request context."""
        try:
            self._verify_token(token)
            return True
        except (AuthenticationError, ImportError):
            return False


class _JwksKeySet:
    """An IdP's signing keys, fetched from its JWKS endpoint and cached.

    The whole set is re-fetched once it is ``lifespan`` seconds old.  A
    token whose ``kid`` is not in the cached set is most likely signed with
    a key the IdP has just rotated in, so the set is re-fetched right away
    rather than refusing the token until the cache expires.  To keep a
    stream of made-up ``kid`` values from turning every request into a
    JWKS fetch, that re-fetch happens at most once per ``kid`` every
    ``unknown_kid_cooldown`` seconds, and at most once every
    ``min_refetch_interval`` seconds across all unknown ``kid`` values.
    """

    #: Unknown ``kid`` values remembered for the per-kid rate limit.
    _MAX_TRACKED_KIDS = 1024

    def __init__(
        self,
        url: str,
        *,
        timeout: float,
        lifespan: float = 300.0,
        unknown_kid_cooldown: float = 60.0,
        min_refetch_interval: float = 1.0,
    ) -> None:
        self.url = url
        self._timeout = timeout
        self._lifespan = lifespan
        self._unknown_kid_cooldown = unknown_kid_cooldown
        self._min_refetch_interval = min_refetch_interval
        self._keys: dict[str, Any] = {}
        self._fetched_at: float | None = None
        self._last_kid_refetch: float | None = None
        self._kid_refetched_at: OrderedDict[str, float] = OrderedDict()
        self._lock = threading.Lock()

    def get_signing_key_from_jwt(self, token: str) -> Any:
        """Return the cached key (a ``jwt.PyJWK``) matching the token's ``kid``."""
        import jwt as pyjwt

        kid = pyjwt.get_unverified_header(token).get("kid")
        if not isinstance(kid, str) or not kid:
            raise LookupError("the token has no 'kid' header")
        return self.get_signing_key(kid)

    def get_signing_key(self, kid: str) -> Any:
        """Return the key for *kid*, re-fetching the set if it is unknown."""
        with self._lock:
            now = time.monotonic()
            if self._fetched_at is None or now - self._fetched_at >= self._lifespan:
                self._refresh(now)
            key = self._keys.get(kid)
            if key is not None:
                return key
            last = self._kid_refetched_at.get(kid)
            if (last is None or now - last >= self._unknown_kid_cooldown) and (
                self._last_kid_refetch is None
                or now - self._last_kid_refetch >= self._min_refetch_interval
            ):
                self._last_kid_refetch = now
                self._kid_refetched_at[kid] = now
                self._kid_refetched_at.move_to_end(kid)
                while len(self._kid_refetched_at) > self._MAX_TRACKED_KIDS:
                    self._kid_refetched_at.popitem(last=False)
                logger.info("JWKS has no key %r; re-fetching %s", kid, self.url)
                self._fetch(now)
                key = self._keys.get(kid)
                if key is not None:
                    return key
            raise LookupError(f"no signing key with kid {kid!r} is published")

    def _refresh(self, now: float) -> None:
        """Re-fetch an expired set, keeping the old keys if the IdP is down."""
        try:
            self._fetch(now)
        except Exception as exc:
            if not self._keys:
                raise
            # Keep verifying with the keys we have and retry in a while,
            # rather than failing every caller while the IdP is unreachable.
            assert self._fetched_at is not None
            self._fetched_at = now - self._lifespan + min(30.0, self._lifespan)
            logger.warning(
                "Could not refresh the JWKS at %s (%s: %s); using the cached keys",
                self.url,
                type(exc).__name__,
                exc,
            )

    def _fetch(self, now: float) -> None:
        import httpx
        import jwt as pyjwt

        response = httpx.get(self.url, timeout=self._timeout)
        if response.status_code != 200:
            raise LookupError(f"the JWKS endpoint answered HTTP {response.status_code}")
        keys: dict[str, Any] = {}
        for key in pyjwt.PyJWKSet.from_dict(response.json()).keys:
            if key.public_key_use in ("sig", None) and key.key_id:
                keys[key.key_id] = key
        self._keys = keys
        self._fetched_at = now


class JwksAuth:
    """JWT authentication against an identity provider's JWKS endpoint.

    Verifies tokens issued by an external identity provider — Microsoft
    Entra, an OIDC provider, an internal IdP — by fetching the provider's
    public keys from its JWKS URL and selecting the one matching the
    token's ``kid``. Keys are fetched on demand and cached, so issuer key
    rotation is handled without reconfiguration.

    This is the server-side counterpart to ``promptise.identity``: an
    agent presents an IdP-issued identity credential, this provider
    verifies it, and the validated ``sub`` / claims are surfaced on
    ``ctx.client`` so guards (``RequireClientId``, ``HasRole``) and
    audit logs can see *which agent* called.

    Requires the ``PyJWT`` and ``cryptography`` packages.

    Args:
        jwks_url: The IdP's JWKS endpoint (for Entra, e.g.
            ``https://login.microsoftonline.com/<tenant>/discovery/v2.0/keys``).
        audience: Expected ``aud`` claim — **required**. Set it to the
            resource the agents target (e.g. the App ID URI of this
            server). It is verified on every token, which is what prevents
            an agent from replaying a token it was issued for a *different*
            resource of the same IdP (token substitution).
        issuer: Expected ``iss`` claim. Strongly recommended: when set,
            tokens from any other issuer are rejected (defence in depth on
            top of the JWKS key set and audience check).
        algorithms: Accepted signing algorithms. Asymmetric only, so a
            token claiming ``HS256`` (algorithm-confusion) is rejected.
        meta_key: Key in ``ctx.meta`` holding the bearer token.
        leeway: Seconds of clock skew tolerated when checking ``exp``,
            ``nbf`` and ``iat``.

    Keys are cached for five minutes.  A token signed with a key that is
    not in the cache (the IdP rotated its keys) makes the server re-fetch
    the JWKS immediately, at most once per key id per minute, so a
    rotated key is accepted on its first use.

    Example::

        auth = JwksAuth(
            jwks_url="https://idp.example.com/.well-known/jwks.json",
            audience="api://my-mcp-server",
            issuer="https://idp.example.com",
        )
        server.add_middleware(AuthMiddleware(auth))

    Or let the JWKS endpoint be discovered from the issuer's OIDC
    ``.well-known/openid-configuration`` (see :meth:`from_discovery`)::

        auth = JwksAuth.from_discovery(
            issuer="https://login.microsoftonline.com/<tenant>/v2.0",
            audience="api://my-mcp-server",
        )
    """

    def __init__(
        self,
        *,
        jwks_url: str,
        audience: str,
        issuer: str | None = None,
        algorithms: tuple[str, ...] = ("RS256", "ES256"),
        meta_key: str = "authorization",
        leeway: float = 60.0,
    ) -> None:
        if not audience:
            raise ValueError(
                "JwksAuth requires a non-empty audience — the resource this "
                "server represents. Verifying only the signature would accept "
                "any token from this IdP, including ones minted for other "
                "resources (token substitution)."
            )
        self._jwks_url = jwks_url
        self._issuer = issuer
        self._audience = audience
        self._algorithms = list(algorithms)
        self._meta_key = meta_key
        # Tolerate small NTP clock differences between the IdP and this server
        # when checking exp/nbf/iat, so a token that is valid "now" is not
        # rejected because the two clocks differ by a few seconds.
        self._leeway = leeway
        self._jwk_client: Any = None
        # Discovery mode (set by from_discovery): jwks_url is resolved lazily
        # from the issuer's OIDC discovery document.
        self._discovery_issuer: str | None = None
        self._request_timeout: float = 5.0

    @classmethod
    def from_discovery(
        cls,
        *,
        issuer: str,
        audience: str,
        algorithms: tuple[str, ...] = ("RS256", "ES256"),
        meta_key: str = "authorization",
        request_timeout: float = 5.0,
        leeway: float = 60.0,
    ) -> JwksAuth:
        """Build a :class:`JwksAuth` that discovers its JWKS from the issuer.

        Resolves the JWKS endpoint from the IdP's OIDC discovery document at
        ``{issuer}/.well-known/openid-configuration`` instead of taking a
        ``jwks_url`` directly. The document is fetched once (lazily, on the
        first verification) and its ``issuer`` is checked against the
        configured ``issuer`` to prevent a spoofed discovery host. Tokens are
        then verified with their ``iss`` pinned to ``issuer`` (and ``aud`` to
        ``audience``, which is required).

        Args:
            issuer: The OIDC issuer URL (its ``.well-known`` is read).
            audience: Expected ``aud`` claim — required.
            algorithms: Accepted signing algorithms.
            meta_key: Key in ``ctx.meta`` holding the bearer token.
            request_timeout: Seconds to wait for the discovery and JWKS
                fetches.
            leeway: Seconds of clock skew tolerated when checking ``exp``,
                ``nbf`` and ``iat`` (as for the constructor).
        """
        if not issuer:
            raise ValueError("JwksAuth.from_discovery requires a non-empty issuer.")
        auth = cls(
            jwks_url="",  # resolved lazily from discovery
            audience=audience,
            issuer=issuer,
            algorithms=algorithms,
            meta_key=meta_key,
            leeway=leeway,
        )
        auth._discovery_issuer = issuer
        auth._request_timeout = request_timeout
        return auth

    def _discover_jwks_url(self) -> str:
        """Fetch and validate the JWKS URL from the OIDC discovery document."""
        import httpx

        issuer = self._discovery_issuer or ""
        url = issuer.rstrip("/") + "/.well-known/openid-configuration"
        try:
            response = httpx.get(url, timeout=self._request_timeout)
        except httpx.HTTPError as exc:
            raise AuthenticationError(
                f"Could not fetch OIDC discovery document from {url} ({type(exc).__name__}: {exc})."
            )
        if response.status_code != 200:
            raise AuthenticationError(
                f"OIDC discovery at {url} returned HTTP {response.status_code}."
            )
        try:
            doc = response.json()
        except ValueError:
            raise AuthenticationError(f"OIDC discovery at {url} returned a non-JSON body.")
        if doc.get("issuer") != issuer:
            raise AuthenticationError(
                f"OIDC discovery issuer mismatch: document declares "
                f"{doc.get('issuer')!r} but {issuer!r} was configured. "
                f"Refusing to trust a discovery host that claims a different "
                f"issuer."
            )
        jwks_uri = doc.get("jwks_uri")
        if not isinstance(jwks_uri, str) or not jwks_uri:
            raise AuthenticationError(f"OIDC discovery at {url} did not advertise a 'jwks_uri'.")
        return jwks_uri

    def _client(self) -> Any:
        if self._jwk_client is None:
            try:
                import jwt  # noqa: F401
            except ImportError:
                raise ImportError(
                    "PyJWT and cryptography are required for JWKS auth. "
                    "Install with: pip install PyJWT cryptography"
                )
            jwks_url = self._jwks_url
            if self._discovery_issuer is not None:
                jwks_url = self._discover_jwks_url()
                self._jwks_url = jwks_url  # cache for diagnostics
            self._jwk_client = _JwksKeySet(jwks_url, timeout=self._request_timeout)
        return self._jwk_client

    async def authenticate(self, ctx: RequestContext) -> str:
        """Authenticate by verifying a JWT against the IdP's JWKS."""
        token = ctx.meta.get(self._meta_key, "")
        if token.startswith("Bearer "):
            token = token[7:]
        if not token:
            raise AuthenticationError("Missing authentication token")

        payload = self._verify_token(token)
        ctx.state["_jwt_payload"] = payload
        return payload.get("sub", payload.get("client_id", "unknown"))

    def _verify_token(self, token: str) -> dict[str, Any]:
        """Verify a JWT against the JWKS and return its claims."""
        try:
            import jwt as pyjwt
        except ImportError:
            raise ImportError(
                "PyJWT and cryptography are required for JWKS auth. "
                "Install with: pip install PyJWT cryptography"
            )

        # Resolve the JWKS client outside the key-lookup try so discovery /
        # import errors surface with their own clear message.
        client = self._client()
        try:
            signing_key = client.get_signing_key_from_jwt(token)
        except Exception as exc:  # noqa: BLE001 — JWKS fetch / kid mismatch
            raise AuthenticationError(
                f"Could not resolve a signing key from the JWKS at "
                f"{self._jwks_url} ({type(exc).__name__}: {exc}). Most common "
                f"cause: the token's 'kid' is not published there, or the JWKS "
                f"URL is wrong."
            )

        # Audience is required (enforced at construction); issuer is verified
        # when configured. Signature + exp are always verified by PyJWT.
        kwargs: dict[str, Any] = {
            "algorithms": self._algorithms,
            "audience": self._audience,
            "leeway": self._leeway,
        }
        if self._issuer is not None:
            kwargs["issuer"] = self._issuer

        try:
            payload: dict[str, Any] = pyjwt.decode(token, signing_key.key, **kwargs)
            return payload
        except pyjwt.ExpiredSignatureError:
            raise AuthenticationError(
                "Token expired",
                suggestion="Have the agent acquire a fresh identity credential",
            )
        except pyjwt.InvalidTokenError as exc:
            raise AuthenticationError(f"Invalid JWT: {exc}")

    def verify_token(self, token: str) -> bool:
        """Check if a token is valid without requiring a request context."""
        try:
            self._verify_token(token)
            return True
        except (AuthenticationError, ImportError):
            return False


class APIKeyAuth:
    """API key-based authentication provider.

    Supports two key formats:

    **Simple** — ``{api_key: client_id}``::

        APIKeyAuth(keys={"sk-abc": "agent-1", "sk-xyz": "agent-2"})

    **Rich** — ``{api_key: {"client_id": ..., "roles": [...]}}``::

        APIKeyAuth(keys={
            "sk-abc": {"client_id": "agent-1", "roles": ["admin", "write"]},
            "sk-xyz": {"client_id": "agent-2", "roles": ["read"]},
        })

    Rich keys populate ``ctx.state["roles"]`` so that role-based
    guards (``HasRole``, ``HasAllRoles``) work out of the box.

    Args:
        keys: Mapping of ``{api_key: client_id_or_config}``.
        header: HTTP header name used to transmit the key.
    """

    def __init__(
        self,
        keys: dict[str, str | dict[str, Any]],
        *,
        header: str = "x-api-key",
    ) -> None:
        # Normalise to rich format internally
        self._keys: dict[str, dict[str, Any]] = {}
        for k, v in keys.items():
            if isinstance(v, str):
                self._keys[k] = {"client_id": v, "roles": [], "tenant_id": None}
            else:
                tenant = v.get("tenant_id")
                self._keys[k] = {
                    "client_id": v.get("client_id", "unknown"),
                    "roles": list(v.get("roles", [])),
                    "tenant_id": tenant if isinstance(tenant, str) and tenant.strip() else None,
                }
        self._meta_key = header

    async def authenticate(self, ctx: RequestContext) -> str:
        """Authenticate using an API key from request metadata.

        On success, sets ``ctx.state["roles"]`` if the key config
        includes roles.

        Returns:
            The ``client_id`` associated with the key.

        Raises:
            AuthenticationError: If the key is missing or invalid.
        """
        key = ctx.meta.get(self._meta_key, "")
        if not key:
            raise AuthenticationError("Missing API key")
        # Timing-safe key comparison: iterate all registered keys
        # using hmac.compare_digest to prevent timing attacks.
        entry = None
        for registered_key, registered_entry in self._keys.items():
            if hmac.compare_digest(key.encode(), registered_key.encode()):
                entry = registered_entry
                break
        if entry is None:
            raise AuthenticationError("Invalid API key")
        # Populate roles for guard compatibility
        if entry["roles"]:
            ctx.state["roles"] = set(entry["roles"])
        # Stash the key's tenant so AuthMiddleware can attach it to
        # ClientContext.tenant_id (there is no JWT claim to read from)
        if entry.get("tenant_id"):
            ctx.state["_api_key_tenant"] = entry["tenant_id"]
        return entry["client_id"]

    def verify_token(self, key: str) -> bool:
        """Check if an API key is valid without requiring a context."""
        # Use timing-safe comparison to prevent key enumeration.
        for registered_key in self._keys:
            if hmac.compare_digest(key.encode(), registered_key.encode()):
                return True
        return False


# =====================================================================
# Helper: build ClientContext from JWT payload
# =====================================================================


def _build_client_context_from_jwt(
    payload: dict[str, Any],
    client_id: str,
    *,
    existing_roles: set[str] | None = None,
    meta: dict[str, Any] | None = None,
    tenant_claim: str = "tenant_id",
) -> ClientContext:
    """Build a :class:`ClientContext` from a verified JWT payload.

    Extracts standard claims (iss, aud, sub, iat, exp) and the
    ``scope`` claim (space-separated string per RFC 8693) into typed
    fields.

    Args:
        payload: Decoded JWT claims dict.
        client_id: Client identifier string (from ``sub`` or provider).
        existing_roles: Roles already extracted by the provider (e.g.
            from API key config).  Merged with JWT ``roles`` claim.
        meta: HTTP headers dict to extract user-agent from.
        tenant_claim: JWT claim carrying the tenant / organisation id.
            Only string claim values are accepted; anything else leaves
            ``tenant_id`` unset (tenant guards then fail closed).
    """
    # Roles: merge JWT "roles" claim with any provider-supplied roles
    jwt_roles = set(payload.get("roles", []))
    all_roles = (existing_roles or set()) | jwt_roles

    # Scopes: parse space-separated "scope" claim (RFC 8693 / OAuth 2.0)
    scope_claim = payload.get("scope", "")
    scopes: set[str] = set()
    if isinstance(scope_claim, str) and scope_claim.strip():
        scopes = set(scope_claim.split())
    elif isinstance(scope_claim, list):
        scopes = set(scope_claim)

    # IP and User-Agent from transport
    ip_address: str | None = None
    user_agent: str | None = None
    if meta:
        user_agent = meta.get("user-agent")
    client_info = get_request_client_info()
    if client_info:
        ip_address = client_info[0]

    # Tenant: configurable claim; only string values are trusted
    tenant_value = payload.get(tenant_claim)
    tenant_id = tenant_value if isinstance(tenant_value, str) and tenant_value.strip() else None

    return ClientContext(
        client_id=client_id,
        tenant_id=tenant_id,
        roles=all_roles,
        scopes=scopes,
        claims=payload,
        issuer=payload.get("iss"),
        audience=payload.get("aud"),
        subject=payload.get("sub"),
        issued_at=payload.get("iat"),
        expires_at=payload.get("exp"),
        ip_address=ip_address,
        user_agent=user_agent,
    )


def _build_client_context_from_api_key(
    client_id: str,
    roles: set[str],
    *,
    meta: dict[str, Any] | None = None,
    tenant_id: str | None = None,
) -> ClientContext:
    """Build a :class:`ClientContext` for API key auth (no JWT claims)."""
    ip_address: str | None = None
    user_agent: str | None = None
    if meta:
        user_agent = meta.get("user-agent")
    client_info = get_request_client_info()
    if client_info:
        ip_address = client_info[0]

    return ClientContext(
        client_id=client_id,
        tenant_id=tenant_id,
        roles=roles,
        ip_address=ip_address,
        user_agent=user_agent,
    )


# =====================================================================
# AuthMiddleware
# =====================================================================


class AuthMiddleware:
    """Middleware that enforces authentication on tools with ``auth=True``.

    After successful authentication, populates ``ctx.client`` with a
    fully structured :class:`ClientContext` containing roles, scopes,
    standard JWT claims, client IP, and user-agent.

    Optionally accepts an ``on_authenticate`` hook for custom
    enrichment (e.g. loading org, tenant, or plan info from a database).

    Args:
        provider: An ``AuthProvider`` implementation.
        on_authenticate: Optional async or sync callable that receives
            ``(client_ctx, request_ctx)`` and can mutate ``client_ctx``
            (e.g. populate ``client_ctx.extra``).
        tenant_claim: JWT claim name carrying the client's tenant /
            organisation id (default ``"tenant_id"``).  The value lands
            on ``ClientContext.tenant_id``, where rate limiting, audit
            entries, and tenant guards read it.  For ``APIKeyAuth``, the
            tenant comes from the key's config dict instead.

    Example::

        async def enrich_client(client: ClientContext, ctx: RequestContext):
            org = await db.get_org_for_client(client.client_id)
            client.extra["org_id"] = org.id
            client.extra["plan"] = org.plan

        server.add_middleware(AuthMiddleware(auth, on_authenticate=enrich_client))
    """

    def __init__(
        self,
        provider: Any,
        *,
        on_authenticate: OnAuthenticateHook | None = None,
        tenant_claim: str = "tenant_id",
    ) -> None:
        self._provider = provider
        self._on_authenticate = on_authenticate
        self._tenant_claim = tenant_claim

    async def __call__(self, ctx: RequestContext, call_next: Callable[..., Any]) -> Any:
        tool_def = ctx.state.get("tool_def")
        if tool_def and tool_def.auth:
            client_id = await self._provider.authenticate(ctx)
            ctx.client_id = client_id

            # Build structured ClientContext
            jwt_payload = ctx.state.get("_jwt_payload", {})
            existing_roles = ctx.state.get("roles", set())

            if jwt_payload:
                # JWT-based auth — extract standard claims + scopes
                client_ctx = _build_client_context_from_jwt(
                    jwt_payload,
                    client_id,
                    existing_roles=existing_roles,
                    meta=ctx.meta,
                    tenant_claim=self._tenant_claim,
                )
            else:
                # API key auth — no JWT claims; tenant comes from key config
                client_ctx = _build_client_context_from_api_key(
                    client_id,
                    existing_roles,
                    meta=ctx.meta,
                    tenant_id=ctx.state.get("_api_key_tenant"),
                )

            # Merge roles back to ctx.state for backward compatibility
            # with guards that read from ctx.state["roles"]
            jwt_roles = set(jwt_payload.get("roles", []))
            ctx.state["roles"] = existing_roles | jwt_roles

            # Run enrichment hook if configured
            if self._on_authenticate is not None:
                result = self._on_authenticate(client_ctx, ctx)
                if asyncio.iscoroutine(result):
                    await result

            # Attach to request context
            ctx.client = client_ctx

            logger.debug(
                "Authenticated client=%s roles=%s scopes=%s ip=%s",
                client_ctx.client_id,
                client_ctx.roles,
                client_ctx.scopes,
                client_ctx.ip_address,
            )

        return await call_next(ctx)
