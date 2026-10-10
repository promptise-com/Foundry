"""Microsoft Entra ID credential provider.

Mints a verifiable identity JWT for an agent from Microsoft Entra, in
two acquisition modes:

* **Managed identity (IMDS)** — for Azure VMs, VM Scale Sets, and other
  compute with a system- or user-assigned managed identity. Calls the
  Azure Instance Metadata Service and extracts the OIDC ``id_token``.
* **Projected token (AKS Workload Identity)** — for AKS pods using the
  Workload Identity admission webhook, which projects a federated token
  to the file named by ``$AZURE_FEDERATED_TOKEN_FILE`` (default
  ``/var/run/secrets/azure/tokens/azure-identity-token``).

The :func:`from_entra` factory's ``auto`` mode selects projected when
``AZURE_FEDERATED_TOKEN_FILE`` is present (the signal that the workload
runs under AKS Workload Identity), and IMDS otherwise.

Entra needs no cloud SDK — IMDS is a plain HTTP GET — so there is no
optional dependency for this provider.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import httpx

from .._core.callable_provider import CallableTokenProvider
from .._core.errors import CredentialAcquisitionError, ProviderConfigError
from .._core.file_provider import FileTokenProvider
from .._core.retry import http_get_with_retry

#: Azure Instance Metadata Service token endpoint (link-local address).
_IMDS_TOKEN_ENDPOINT: str = "http://169.254.169.254/metadata/identity/oauth2/token"

#: Environment variable AKS Workload Identity sets to the projected
#: token file path.
ENV_AZURE_FEDERATED_TOKEN_FILE: str = "AZURE_FEDERATED_TOKEN_FILE"

#: Default projected-token path used when the env var is unset.
DEFAULT_AZURE_FEDERATED_TOKEN_FILE: str = "/var/run/secrets/azure/tokens/azure-identity-token"

#: Default resource (audience) requested from IMDS. Set this to the
#: resource the agent authenticates to (for example, the App ID URI of
#: an MCP server protected by Entra).
_DEFAULT_RESOURCE: str = "api://promptise-agent"

#: Default IMDS API version.
_DEFAULT_IMDS_API_VERSION: str = "2018-02-01"


class EntraManagedIdentityProvider(CallableTokenProvider):
    """Entra managed-identity credential source backed by Azure IMDS.

    Args:
        client_id: Optional client ID of a user-assigned managed
            identity. Omit for the resource's system-assigned identity.
        resource: The resource (audience) to request — set this to the
            resource the agent presents its identity to.
        api_version: IMDS API version. Defaults to ``2018-02-01``.
        imds_endpoint: Override for the IMDS URL — primarily for tests.
        request_timeout: Seconds to wait for the IMDS response.
            Defaults to five seconds; IMDS is link-local and fast.
    """

    def __init__(
        self,
        *,
        client_id: str | None = None,
        resource: str = _DEFAULT_RESOURCE,
        api_version: str = _DEFAULT_IMDS_API_VERSION,
        imds_endpoint: str = _IMDS_TOKEN_ENDPOINT,
        request_timeout: float = 5.0,
    ) -> None:
        self._client_id: str | None = client_id
        self._resource: str = resource
        self._api_version: str = api_version
        self._imds_endpoint: str = imds_endpoint
        self._request_timeout: float = request_timeout
        super().__init__(
            token_fn=self._fetch_imds_id_token,
            provider_label="entra-imds",
            default_audience=resource,
        )

    def _fetch_imds_id_token(self, audience: str | None = None) -> str:
        """Fetch the OIDC ``id_token`` from the Azure IMDS endpoint.

        Requests ``audience`` as the resource when given, otherwise the
        configured default. Raises a precise
        :class:`CredentialAcquisitionError` on every failure mode; the
        base :class:`CallableTokenProvider` passes typed errors through.
        """
        resource = audience or self._resource
        params: dict[str, str] = {
            "api-version": self._api_version,
            "resource": resource,
        }
        if self._client_id is not None:
            params["client_id"] = self._client_id

        try:
            response = http_get_with_retry(
                self._imds_endpoint,
                params=params,
                headers={"Metadata": "true"},
                timeout=self._request_timeout,
            )
        except httpx.TimeoutException as exc:
            raise CredentialAcquisitionError(
                f"[entra-imds] the Azure Instance Metadata Service at "
                f"{self._imds_endpoint} timed out after "
                f"{self._request_timeout}s. Most common cause: this workload "
                f"is not running on Azure compute, or egress to "
                f"169.254.169.254 is blocked."
            ) from exc
        except httpx.HTTPError as exc:
            raise CredentialAcquisitionError(
                f"[entra-imds] could not reach the Azure Instance Metadata "
                f"Service at {self._imds_endpoint} ({type(exc).__name__}). "
                f"Most common cause: this workload is not running on Azure "
                f"compute with a managed identity assigned. Underlying error: "
                f"{exc}"
            ) from exc

        if response.status_code != 200:
            raise CredentialAcquisitionError(
                f"[entra-imds] IMDS returned HTTP {response.status_code}. "
                f"Most common cause: no managed identity is assigned to this "
                f"resource, or the requested resource {resource!r} is "
                f"not permitted. Response body: {response.text}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            # Never echo the body: a misbehaving endpoint (or a proxy in front
            # of it) may answer with a bare token, which must not end up in an
            # exception message or a log line.
            raise CredentialAcquisitionError(
                f"[entra-imds] IMDS returned a non-JSON body "
                f"({len(response.content)} bytes, content-type "
                f"{response.headers.get('content-type', 'unknown')!r})."
            ) from exc

        id_token = body.get("id_token")
        if not isinstance(id_token, str) or not id_token:
            raise CredentialAcquisitionError(
                f"[entra-imds] IMDS response did not contain an 'id_token' "
                f"field. Most common cause: the managed identity returned "
                f"only an access_token; the federation flow needs the OIDC "
                f"id_token. Body keys: {list(body.keys())}"
            )
        return id_token


class EntraProjectedTokenProvider(FileTokenProvider):
    """Entra projected-token credential source for AKS Workload Identity.

    Args:
        token_file: Path to the projected token. Defaults to the value
            of ``$AZURE_FEDERATED_TOKEN_FILE`` or, if that is unset,
            :data:`DEFAULT_AZURE_FEDERATED_TOKEN_FILE`.
    """

    def __init__(
        self,
        *,
        token_file: str | Path | None = None,
    ) -> None:
        resolved_path: str | Path
        if token_file is not None:
            resolved_path = token_file
        else:
            resolved_path = os.environ.get(
                ENV_AZURE_FEDERATED_TOKEN_FILE,
                DEFAULT_AZURE_FEDERATED_TOKEN_FILE,
            )
        super().__init__(
            token_file=resolved_path,
            provider_label="entra-projected",
        )


def from_entra(
    *,
    mode: Literal["auto", "imds", "projected"] = "auto",
    client_id: str | None = None,
    token_file: str | Path | None = None,
    resource: str = _DEFAULT_RESOURCE,
) -> EntraManagedIdentityProvider | EntraProjectedTokenProvider:
    """Build a Microsoft Entra ID credential source.

    Args:
        mode: ``"auto"`` (default) picks projected when
            ``AZURE_FEDERATED_TOKEN_FILE`` is set, otherwise IMDS.
            Force a mode with ``"imds"`` or ``"projected"``.
        client_id: Client ID of a user-assigned managed identity
            (IMDS mode only).
        token_file: Projected-token path (projected mode only).
            Defaults to ``$AZURE_FEDERATED_TOKEN_FILE``.
        resource: Resource/audience to request from IMDS (IMDS mode
            only) — the resource the agent authenticates to.

    Returns:
        An :class:`EntraManagedIdentityProvider` for IMDS mode or an
        :class:`EntraProjectedTokenProvider` for projected mode.

    Raises:
        ProviderConfigError: If ``mode`` is not one of ``auto``,
            ``imds``, or ``projected``.
    """
    resolved_mode = mode
    if resolved_mode == "auto":
        resolved_mode = "projected" if os.environ.get(ENV_AZURE_FEDERATED_TOKEN_FILE) else "imds"

    if resolved_mode == "projected":
        return EntraProjectedTokenProvider(token_file=token_file)
    if resolved_mode == "imds":
        return EntraManagedIdentityProvider(client_id=client_id, resource=resource)
    raise ProviderConfigError(
        f"Unknown Entra mode {mode!r}. Valid modes are 'auto', 'imds', and 'projected'."
    )
