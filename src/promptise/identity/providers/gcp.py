"""Google Cloud credential provider.

Mints a verifiable identity JWT for an agent by reading an OIDC identity
token from the Google Compute metadata server. Works on Compute Engine,
GKE, Cloud Run, Cloud Functions, and any other GCP runtime exposing the
metadata server at ``metadata.google.internal``.

The metadata identity endpoint returns the JWT as a **plain string** in
the response body — it is *not* a JSON wrapper. The audience is supplied
as a query parameter. The response body is used directly, never parsed
as JSON.

GCP needs no cloud SDK — the metadata server is a plain HTTP GET — so
there is no optional dependency for this provider.
"""

from __future__ import annotations

import httpx

from .._core.callable_provider import CallableTokenProvider
from .._core.errors import CredentialAcquisitionError
from .._core.retry import http_get_with_retry

#: Template for the GCP metadata identity endpoint. The service-account
#: email segment selects which attached service account issues the token.
_METADATA_IDENTITY_URL_TEMPLATE: str = (
    "http://metadata.google.internal/computeMetadata/v1/instance/"
    "service-accounts/{service_account_email}/identity"
)

#: Default audience requested in the identity token — the resource the
#: agent authenticates to. Override with the audience your resource
#: expects (for example, the URL of an MCP server).
_DEFAULT_AUDIENCE: str = "api://promptise-agent"

#: Default service account — the instance's primary attached account.
_DEFAULT_SERVICE_ACCOUNT_EMAIL: str = "default"


class GcpMetadataProvider(CallableTokenProvider):
    """GCP credential source backed by the Compute metadata server.

    Args:
        audience: Audience to request in the identity token — the
            resource the agent authenticates to.
        service_account_email: The attached service account whose
            identity to request. ``"default"`` (the default) selects the
            instance's primary service account; pass a full email to
            select a specific one.
        metadata_endpoint: Override for the metadata URL — primarily for
            tests. When ``None`` it is derived from
            ``service_account_email``.
        request_timeout: Seconds to wait for the metadata response.
            Defaults to five seconds; the metadata server is link-local.
    """

    def __init__(
        self,
        *,
        audience: str = _DEFAULT_AUDIENCE,
        service_account_email: str = _DEFAULT_SERVICE_ACCOUNT_EMAIL,
        metadata_endpoint: str | None = None,
        request_timeout: float = 5.0,
    ) -> None:
        self._audience: str = audience
        self._service_account_email: str = service_account_email
        self._metadata_endpoint: str = (
            metadata_endpoint
            if metadata_endpoint is not None
            else _METADATA_IDENTITY_URL_TEMPLATE.format(service_account_email=service_account_email)
        )
        self._request_timeout: float = request_timeout
        super().__init__(
            token_fn=self._fetch_metadata_identity_token,
            provider_label="gcp-metadata",
            default_audience=audience,
        )

    def _fetch_metadata_identity_token(self, audience: str | None = None) -> str:
        """Fetch the OIDC identity token from the GCP metadata server.

        Requests ``audience`` when given, otherwise the configured default.
        The response body is the JWT itself — returned verbatim, never
        parsed as JSON. Raises a precise
        :class:`CredentialAcquisitionError` on every failure mode.
        """
        requested_audience = audience or self._audience
        try:
            response = http_get_with_retry(
                self._metadata_endpoint,
                params={"audience": requested_audience},
                headers={"Metadata-Flavor": "Google"},
                timeout=self._request_timeout,
            )
        except httpx.TimeoutException as exc:
            raise CredentialAcquisitionError(
                f"[gcp-metadata] the Google Compute metadata server at "
                f"{self._metadata_endpoint} timed out after "
                f"{self._request_timeout}s. Most common cause: this workload "
                f"is not running on Google Cloud, or egress to "
                f"metadata.google.internal is blocked."
            ) from exc
        except httpx.HTTPError as exc:
            raise CredentialAcquisitionError(
                f"[gcp-metadata] could not reach the Google Compute metadata "
                f"server at {self._metadata_endpoint} ({type(exc).__name__}). "
                f"Most common cause: this workload is not running on Google "
                f"Cloud compute. Underlying error: {exc}"
            ) from exc

        if response.status_code != 200:
            raise CredentialAcquisitionError(
                f"[gcp-metadata] the metadata server returned HTTP "
                f"{response.status_code}. Most common cause: the service "
                f"account {self._service_account_email!r} is not attached to "
                f"this instance, or the audience {requested_audience!r} is not "
                f"permitted. Response body: {response.text}"
            )

        # The identity endpoint returns the JWT directly as the body —
        # plain text, not JSON. Use it verbatim.
        token = response.text.strip()
        if not token:
            raise CredentialAcquisitionError(
                "[gcp-metadata] the metadata server returned an empty body. "
                "Most common cause: a transient metadata-server issue or a "
                "malformed audience parameter."
            )
        return token


def from_gcp(
    *,
    audience: str = _DEFAULT_AUDIENCE,
    service_account_email: str = _DEFAULT_SERVICE_ACCOUNT_EMAIL,
    request_timeout: float = 5.0,
) -> GcpMetadataProvider:
    """Build a Google Cloud credential source.

    Args:
        audience: Audience requested in the identity token — the resource
            the agent authenticates to.
        service_account_email: The attached service account whose
            identity to request. Defaults to ``"default"``.
        request_timeout: Seconds to wait for the metadata response.

    Returns:
        A :class:`GcpMetadataProvider`.
    """
    return GcpMetadataProvider(
        audience=audience,
        service_account_email=service_account_email,
        request_timeout=request_timeout,
    )
