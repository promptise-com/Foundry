"""Generic OIDC credential provider.

For any standards-compliant OIDC issuer that Promptise does not ship a
dedicated provider for — GitLab CI, CircleCI, Azure DevOps, Keycloak,
Authentik, Dex, and others — as well as local development. The identity
JWT reaches the framework via exactly one of three mutually-exclusive
sources:

* a **file** on disk (``token_file``) — for systems that project the
  OIDC token to a path;
* a **callable** (``token_fn``) — for programmatic acquisition;
* an **environment variable** (``token_env_var``) — the simplest path,
  used by CI systems that expose the token directly in the environment.

The cloud providers (Entra, AWS, GCP, SPIFFE) all reduce to one of the
two base mechanisms; this module is the foundation they build on.

Example::

    from promptise.identity import AgentIdentity

    # GitLab CI exposes its OIDC token in an environment variable.
    identity = AgentIdentity.from_oidc(
        "release-bot",
        issuer="https://gitlab.com",
        token_env_var="CI_JOB_JWT_V2",
    )
    credential = identity.get_credential()   # present this to a resource
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import cast

from .._core.callable_provider import CallableTokenProvider
from .._core.errors import CredentialAcquisitionError, ProviderConfigError
from .._core.file_provider import FileTokenProvider


class OidcFileProvider(FileTokenProvider):
    """Generic OIDC credential source that reads the JWT from a file path.

    A thin specialisation of :class:`FileTokenProvider` that records the
    issuer and fixes the provider label to ``"oidc:<issuer>"``.
    Construct via :meth:`AgentIdentity.from_oidc` rather than directly.

    Args:
        issuer: The OIDC issuer URL — the ``iss`` claim your JWTs carry.
            Recorded for diagnostics and embedded in the provider label.
        token_file: See :class:`FileTokenProvider`.
    """

    def __init__(
        self,
        *,
        issuer: str,
        token_file: str | Path,
    ) -> None:
        super().__init__(
            token_file=token_file,
            provider_label=f"oidc:{issuer}",
        )
        self.issuer: str = issuer


class OidcCallableProvider(CallableTokenProvider):
    """Generic OIDC credential source that invokes a callable for the JWT.

    A thin specialisation of :class:`CallableTokenProvider` that records
    the issuer and fixes the provider label to ``"oidc:<issuer>"``.
    Construct via :meth:`AgentIdentity.from_oidc` rather than directly.
    Environment-variable mode is implemented on top of this class with a
    callable that reads the variable fresh on every refresh.

    Args:
        issuer: The OIDC issuer URL — the ``iss`` claim your JWTs carry.
        token_fn: A zero-argument callable returning the JWT. The token's
            audience is fixed by the issuer, so a per-resource audience
            request is ignored (this is a passive provider).
    """

    _audience_scoped = False

    def __init__(
        self,
        *,
        issuer: str,
        token_fn: Callable[[], str],
    ) -> None:
        # Adapt the user's zero-arg callable to the audience-aware contract.
        # OIDC tokens have a fixed audience, so the requested audience is
        # ignored.
        def _ignore_audience(audience: str | None = None) -> str:
            return token_fn()

        super().__init__(
            token_fn=_ignore_audience,
            provider_label=f"oidc:{issuer}",
        )
        self.issuer: str = issuer


def _make_env_var_reader(var_name: str) -> Callable[[], str]:
    """Return a callable that reads the JWT from ``var_name`` each call.

    The variable is read fresh on every refresh — some CI systems
    rotate the projected OIDC token in place, just as Kubernetes
    rotates projected files.
    """

    def _read() -> str:
        value = os.environ.get(var_name)
        if value is None or not value.strip():
            raise CredentialAcquisitionError(
                f"OIDC token environment variable {var_name!r} is not set "
                f"(or is empty) at refresh time. Most common cause: the CI "
                f"system that injects the OIDC token did not run for this "
                f"job, or the variable name is misspelled."
            )
        return value

    return _read


def from_oidc(
    issuer: str,
    *,
    token_file: str | Path | None = None,
    token_fn: Callable[[], str] | None = None,
    token_env_var: str | None = None,
) -> OidcFileProvider | OidcCallableProvider:
    """Build a generic OIDC credential source.

    Exactly one of ``token_file``, ``token_fn``, or ``token_env_var``
    must be supplied — they are the three mutually-exclusive ways the
    issuer's JWT reaches the framework.

    Args:
        issuer: The OIDC issuer URL (the ``iss`` claim your JWTs carry).
            Used in the provider label and recorded for diagnostics.
        token_file: Path to a file the issuer projects the JWT into.
        token_fn: A zero-argument callable returning the JWT string.
        token_env_var: Name of an environment variable holding the JWT.
            Read fresh on every refresh.

    Returns:
        An :class:`OidcFileProvider` for ``token_file`` mode, or an
        :class:`OidcCallableProvider` for ``token_fn`` and
        ``token_env_var`` modes.

    Raises:
        ProviderConfigError: If zero or more than one token source is
            supplied.
    """
    supplied = [
        name
        for name, value in (
            ("token_file", token_file),
            ("token_fn", token_fn),
            ("token_env_var", token_env_var),
        )
        if value is not None
    ]
    if len(supplied) == 0:
        raise ProviderConfigError(
            "from_oidc requires exactly one token source but none was "
            "supplied. Pass one of token_file=..., token_fn=..., or "
            "token_env_var=.... Most common cause: the JWT source for your "
            "CI system was not wired up — e.g. GitLab CI exposes the token "
            "in CI_JOB_JWT_V2, so pass token_env_var='CI_JOB_JWT_V2'."
        )
    if len(supplied) > 1:
        raise ProviderConfigError(
            f"from_oidc requires exactly one token source but "
            f"{len(supplied)} were supplied ({', '.join(supplied)}). These "
            f"are mutually exclusive — choose the single way the issuer's "
            f"JWT reaches this workload and remove the others."
        )

    if token_file is not None:
        return OidcFileProvider(issuer=issuer, token_file=token_file)

    resolved_fn = (
        token_fn if token_fn is not None else _make_env_var_reader(cast(str, token_env_var))
    )
    return OidcCallableProvider(issuer=issuer, token_fn=resolved_fn)
