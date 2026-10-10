""":class:`IdentityProvider` — a verifiable-credential source for an agent.

A credential provider mints a short-lived JWT that proves *"this really
is agent X."* It is the optional, *verifiable* backing of an
:class:`~promptise.identity.AgentIdentity`: a plain identity is just an
``agent_id`` used for local attribution, while a verifiable identity can
present this credential to the resources it calls (an MCP server, an
HTTP API) so they can authenticate and attribute the caller
cryptographically.

Every concrete provider subclasses :class:`IdentityProvider` and
implements two members:

* the :attr:`provider_name` property — a short label used in logs and
  error output;
* the :meth:`_acquire_upstream_jwt` method — fetches a fresh JWT from
  the platform (metadata service, STS, Workload API, or a file).

The base class owns the small cache: it holds one :class:`CachedCredential`,
serialises concurrent callers with a :class:`threading.Lock`, and
re-acquires the JWT when it nears its ``exp``.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod

from .cache import CachedCredential, decode_jwt_expiry


class IdentityProvider(ABC):
    """Abstract base for a verifiable agent-identity credential source.

    A single instance is safe to share across threads. Concurrent calls
    to :meth:`get_credential` collapse into one acquisition, serialised
    by the lock. The cache is process-local: each process acquires its
    own credential rather than sharing one.
    """

    #: Whether the requested audience selects the credential. ``False`` for
    #: passive providers (projected token files, fixed-audience OIDC), whose
    #: token is the same whatever audience is asked for, so every audience
    #: shares one cached credential.
    _audience_scoped: bool = True

    def __init__(self) -> None:
        # One cached credential per audience. Lets one identity present to
        # several resources that each require their own ``aud``. Keys are
        # normalised (see ``_cache_key``) so ``None`` and the provider's
        # default audience share one entry.
        self._cached: dict[str | None, CachedCredential] = {}
        self._lock: threading.Lock = threading.Lock()
        self._default_audience: str | None = None

    @property
    def default_audience(self) -> str | None:
        """The audience requested when none is given, if the provider knows it."""
        return self._default_audience

    def _cache_key(self, audience: str | None) -> str | None:
        """Normalise a requested audience to the audience actually minted for."""
        if not self._audience_scoped:
            return None
        return audience or self._default_audience

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """A short identifier used in log messages and error output."""

    @abstractmethod
    def _acquire_upstream_jwt(self, audience: str | None = None) -> str:
        """Fetch a fresh identity JWT from the platform.

        Subclasses call the platform's metadata service, SDK, or read a
        projected token file and return the JWT as a bare string.

        Args:
            audience: The resource the credential targets. **Active**
                providers (Entra IMDS, AWS STS, GCP metadata, SPIFFE SDK)
                mint a credential for this audience when given, falling
                back to their configured default when ``None``. **Passive**
                providers (projected token files, file/env/callable OIDC)
                serve a token whose audience the platform fixed and ignore
                this argument.

        Raises:
            CredentialAcquisitionError: When the platform cannot supply a
                JWT. Subclasses wrap their underlying exception with
                ``raise … from exc`` so the original cause stays attached.
        """

    def get_credential(self, audience: str | None = None, *, force_refresh: bool = False) -> str:
        """Return a currently-valid identity credential (a JWT).

        Returns the cached credential for ``audience`` when it is still
        well inside its lifetime; otherwise acquires a fresh one. The
        credential's expiry is read from its ``exp`` claim; a credential
        with no decodable expiry (an opaque token, or a rotated file token)
        is re-acquired on every call.

        Args:
            audience: The resource the credential is for. ``None`` uses the
                provider's default audience; asking for the default audience
                by name returns the same cached credential. An active provider
                mints a separate credential per audience (cached per
                audience); a passive provider's token has a fixed audience and
                this is ignored.
            force_refresh: Skip the cache and acquire a fresh credential.
                Use it when a resource rejected the cached one (an HTTP
                ``401``) although it has not reached its ``exp``: the key
                was rotated or revoked, or the clocks disagree.

        Returns:
            The credential JWT to present to a resource.

        Raises:
            CredentialAcquisitionError: When the platform cannot supply
                a fresh JWT and none is safely cached.
        """
        key = self._cache_key(audience)
        with self._lock:
            cached = self._cached.get(key)
            if cached is not None and not force_refresh and not cached.is_stale():
                return cached.token
            jwt = self._acquire_upstream_jwt(key)
            self._cached[key] = CachedCredential(
                token=jwt,
                expires_at_epoch=decode_jwt_expiry(jwt),
                acquired_at_epoch=time.time(),
            )
            return jwt

    def auth_header(self, audience: str | None = None) -> dict[str, str]:
        """Return a ready-to-use ``Authorization`` bearer header.

        Convenience for presenting the identity credential to a resource
        — an MCP server or a third-party API the agent calls.

        Args:
            audience: The resource the credential is for (see
                :meth:`get_credential`).

        Returns:
            ``{"Authorization": "Bearer <credential>"}``.
        """
        return {"Authorization": f"Bearer {self.get_credential(audience)}"}
