"""Cached-credential primitive for verifiable agent identities.

A verifiable :class:`~promptise.identity.AgentIdentity` is backed by a
credential provider that mints a short-lived JWT proving *"this really
is agent X."* That JWT is the credential the agent presents to the
resources it calls (an MCP server, an HTTP API). This module owns the
small cache that holds one such JWT and decides when to re-acquire it.

Unlike an LLM access token, a workload JWT carries its own absolute
expiry in the standard ``exp`` claim, so the cache compares against
wall-clock time rather than a relative ``expires_in``. When a token has
no decodable ``exp`` (an opaque or non-JWT credential, or a rotated file
token), the cache treats it as always-stale and re-acquires on every
call — which is exactly what file-projected tokens want, since the
platform rotates them in place.
"""

from __future__ import annotations

import base64
import binascii
import json
import time
from dataclasses import dataclass
from typing import Any

#: Seconds before a credential's ``exp`` at which it is re-acquired, so a
#: token is never presented to a resource right as it expires.
CREDENTIAL_REFRESH_BUFFER_SECONDS: int = 60


@dataclass(frozen=True, slots=True)
class CachedCredential:
    """One cached identity credential (a verifiable JWT) and its expiry.

    Immutable. A provider holds at most one instance and replaces it
    rather than mutating it.

    Args:
        token: The credential JWT the agent presents to resources.
        expires_at_epoch: The Unix timestamp from the JWT's ``exp``
            claim, or ``None`` when no expiry could be decoded (in which
            case the credential is always considered stale).
        acquired_at_epoch: When the credential was acquired. When known,
            the refresh buffer is capped at half the credential's lifetime,
            so a token that lives less than twice the buffer is still
            reused for a while instead of being re-acquired on every call.
    """

    token: str
    expires_at_epoch: float | None
    acquired_at_epoch: float | None = None

    def is_stale(self, buffer_seconds: int = CREDENTIAL_REFRESH_BUFFER_SECONDS) -> bool:
        """Return ``True`` when the credential should be re-acquired.

        A credential with no known expiry is always stale (re-acquired
        on every use). Otherwise it is stale once the current time is
        within ``buffer_seconds`` of the ``exp`` claim — or within half its
        lifetime of it, if that is shorter.
        """
        if self.expires_at_epoch is None:
            return True
        buffer: float = buffer_seconds
        if self.acquired_at_epoch is not None:
            lifetime = max(self.expires_at_epoch - self.acquired_at_epoch, 0.0)
            buffer = min(buffer, lifetime / 2)
        return time.time() + buffer >= self.expires_at_epoch


def decode_jwt_claims(jwt: str) -> dict[str, Any]:
    """Best-effort decode of a JWT's claims payload, without verifying.

    The holder of a credential does not verify its own signature — that
    is the receiving resource's job. This decodes the payload segment so
    the framework can read identity claims (``sub``, ``oid``, …) and the
    expiry. Returns an empty dict if anything is malformed.

    Args:
        jwt: A compact-serialization JWT (``header.payload.signature``).

    Returns:
        The decoded claims, or ``{}`` if the JWT cannot be parsed.
    """
    parts = jwt.split(".")
    if len(parts) < 2:
        return {}
    padded = parts[1] + "=" * (-len(parts[1]) % 4)
    try:
        claims = json.loads(base64.urlsafe_b64decode(padded))
    except (binascii.Error, ValueError, TypeError):
        return {}
    return claims if isinstance(claims, dict) else {}


def decode_jwt_expiry(jwt: str) -> float | None:
    """Best-effort read of the ``exp`` claim from a JWT, without verifying.

    Args:
        jwt: A compact-serialization JWT (``header.payload.signature``).

    Returns:
        The ``exp`` claim as a Unix timestamp, or ``None`` if absent or
        malformed.
    """
    exp = decode_jwt_claims(jwt).get("exp")
    if isinstance(exp, (int, float)) and not isinstance(exp, bool):
        return float(exp)
    return None
