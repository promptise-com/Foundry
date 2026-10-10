"""Callable-backed credential provider — invokes a function for the JWT.

Used by:

* AWS STS ``get_web_identity_token`` (wrapped in a callable inside
  :class:`AwsStsProvider`).
* GCP metadata server ``identity`` endpoint.
* SPIFFE Workload API via :mod:`pyspiffe`.
* Generic OIDC callable mode.

Any exception raised by the user-supplied callable is wrapped in
:class:`CredentialAcquisitionError` with the provider label in the
message and the underlying exception chained on ``__cause__``.
"""

from __future__ import annotations

from collections.abc import Callable

from .errors import CredentialAcquisitionError, IdentityError
from .provider import IdentityProvider


class CallableTokenProvider(IdentityProvider):
    """Invokes a user-supplied callable to acquire the identity JWT.

    Args:
        token_fn: A callable that returns the JWT as a string, accepting
            the requested ``audience`` (``str | None``). Called fresh on
            every refresh — caching the result would defeat the platform's
            own rotation. Active providers honour the audience; passive
            ones ignore it.
        provider_label: Short string used in log messages and error
            output. Concrete provider subclasses (Entra IMDS, AWS STS,
            GCP metadata, SPIFFE SDK, generic OIDC callable) pass
            their own label.
        default_audience: The audience ``token_fn`` mints for when it is
            called with ``None``. When given, ``token_fn`` always receives
            an explicit audience, and asking for ``None`` or for this
            audience shares one cached credential instead of fetching twice.
    """

    def __init__(
        self,
        *,
        token_fn: Callable[[str | None], str],
        provider_label: str = "callable",
        default_audience: str | None = None,
    ) -> None:
        super().__init__()
        self._token_fn: Callable[[str | None], str] = token_fn
        self._provider_label: str = provider_label
        self._default_audience = default_audience or None

    @property
    def provider_name(self) -> str:
        return self._provider_label

    def _acquire_upstream_jwt(self, audience: str | None = None) -> str:
        try:
            token = self._token_fn(audience)
        except IdentityError:
            # The callable already produced a precise, typed identity
            # error — a CredentialAcquisitionError from a metadata-service
            # provider, or a ProviderConfigError from a missing optional
            # dependency. Let it propagate unchanged rather than wrapping
            # it in a second, more generic error.
            raise
        except Exception as exc:
            raise CredentialAcquisitionError(
                f"[{self._provider_label}] identity JWT callable raised "
                f"{type(exc).__name__}: {exc}. Most common cause: the cloud "
                f"SDK or metadata service is unreachable from this workload."
            ) from exc
        if not isinstance(token, str):
            raise CredentialAcquisitionError(
                f"[{self._provider_label}] identity JWT callable returned "
                f"{type(token).__name__}, expected str. Fix the callable to "
                f"return the JWT as a string."
            )
        token = token.strip()
        if not token:
            raise CredentialAcquisitionError(
                f"[{self._provider_label}] identity JWT callable returned an "
                f"empty string. Most common cause: the metadata service "
                f"responded successfully but the JWT body was empty "
                f"(transient issue, or a misconfigured audience parameter)."
            )
        return token
