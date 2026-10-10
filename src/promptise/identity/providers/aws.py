"""AWS IAM credential provider.

Mints a verifiable identity JWT for an agent from AWS IAM, in two
acquisition modes:

* **STS GetWebIdentityToken** — for Lambda, EC2, ECS, and EKS workloads
  with an IAM role. Lazily imports boto3 and calls STS to mint an OIDC
  web-identity token scoped to the configured audience.
* **EKS projected token** — reads a projected federated token from
  ``$PROMPTISE_IDENTITY_TOKEN_FILE`` (default
  ``/var/run/secrets/promptise/token``).

boto3 is an *optional* dependency (``pip install promptise[identity-aws]``).
The STS provider imports it inside the acquisition method, never at
module top, so a workload that only uses the EKS-projected mode — or
only uses a different cloud — does not need boto3 installed. When the
import fails the provider raises :class:`ProviderConfigError` naming the
exact install command.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from .._core.callable_provider import CallableTokenProvider
from .._core.errors import CredentialAcquisitionError, ProviderConfigError
from .._core.file_provider import FileTokenProvider
from .._core.retry import aws_is_transient, retry_call

#: Environment variable naming the projected federated-token file.
ENV_PROMPTISE_IDENTITY_TOKEN_FILE: str = "PROMPTISE_IDENTITY_TOKEN_FILE"

#: Default projected-token path when the env var is unset.
DEFAULT_PROMPTISE_IDENTITY_TOKEN_FILE: str = "/var/run/secrets/promptise/token"

#: Environment variables AWS uses to expose the active region.
ENV_AWS_REGION: str = "AWS_REGION"
ENV_AWS_DEFAULT_REGION: str = "AWS_DEFAULT_REGION"

#: Default audience requested from STS — the resource the agent
#: authenticates to. Override with the audience your resource expects.
_DEFAULT_AUDIENCE: str = "api://promptise-agent"

#: Default JWT signing algorithm requested from STS.
_DEFAULT_SIGNING_ALGORITHM: str = "RS256"

#: The exact install command surfaced when boto3 is missing.
_INSTALL_HINT: str = "pip install promptise[identity-aws]"


class AwsStsProvider(CallableTokenProvider):
    """AWS STS web-identity-token credential source (lazily imports boto3).

    STS is regional, so a region is required. It is resolved from the
    ``region`` argument, then ``AWS_REGION``, then ``AWS_DEFAULT_REGION``.

    Args:
        region: AWS region for the STS client. Falls back to
            ``AWS_REGION`` / ``AWS_DEFAULT_REGION``.
        audience: Audience to request in the web-identity token — the
            resource the agent authenticates to.
        signing_algorithm: JWT signing algorithm STS should use.
            Defaults to ``RS256``.

    Raises:
        ProviderConfigError: At construction, if no region can be
            resolved.
    """

    def __init__(
        self,
        *,
        region: str | None = None,
        audience: str = _DEFAULT_AUDIENCE,
        signing_algorithm: str = _DEFAULT_SIGNING_ALGORITHM,
    ) -> None:
        resolved_region = (
            region or os.environ.get(ENV_AWS_REGION) or os.environ.get(ENV_AWS_DEFAULT_REGION)
        )
        if not resolved_region:
            raise ProviderConfigError(
                "AWS STS is regional but no region was supplied. Pass "
                "region='us-east-1' (or your region) or set AWS_REGION / "
                "AWS_DEFAULT_REGION in the environment. Most common cause: "
                "constructing the provider outside a configured AWS runtime."
            )
        self._region: str = resolved_region
        self._audience: str = audience
        self._signing_algorithm: str = signing_algorithm
        super().__init__(
            token_fn=self._fetch_sts_web_identity_token,
            provider_label="aws-sts",
            default_audience=audience,
        )

    def _fetch_sts_web_identity_token(self, audience: str | None = None) -> str:
        """Mint an OIDC web-identity token via STS.

        Requests ``audience`` when given, otherwise the configured default.
        Lazily imports boto3. Raises :class:`ProviderConfigError` when
        boto3 is missing and :class:`CredentialAcquisitionError` when the
        STS call itself fails; both propagate through the base unchanged.
        """
        try:
            import boto3  # noqa: E402  -- lazy import is intentional
        except ImportError as exc:
            raise ProviderConfigError(
                f"[aws-sts] AWS STS mode requires boto3, which is not "
                f"installed. Install it with: {_INSTALL_HINT}. A workload "
                f"that only uses the EKS-projected mode does not need boto3."
            ) from exc

        try:
            client = boto3.client("sts", region_name=self._region)
            response: dict[str, Any] = retry_call(
                lambda: client.get_web_identity_token(
                    Audience=[audience or self._audience],
                    SigningAlgorithm=self._signing_algorithm,
                ),
                is_transient=aws_is_transient,
            )
        except Exception as exc:
            raise CredentialAcquisitionError(
                f"[aws-sts] STS GetWebIdentityToken failed "
                f"({type(exc).__name__}: {exc}). Most common cause: this "
                f"workload has no IAM role attached, the role lacks "
                f"sts:GetWebIdentityToken permission, or the region "
                f"{self._region!r} is wrong."
            ) from exc

        token = response.get("WebIdentityToken")
        if not isinstance(token, str) or not token:
            raise CredentialAcquisitionError(
                f"[aws-sts] STS response did not contain a WebIdentityToken "
                f"string. Most common cause: AWS changed the "
                f"GetWebIdentityToken response schema. Response keys: "
                f"{list(response.keys())}"
            )
        return token


class AwsEksProjectedProvider(FileTokenProvider):
    """AWS EKS projected-token credential source.

    Reads a projected federated token from
    ``$PROMPTISE_IDENTITY_TOKEN_FILE`` (default
    ``/var/run/secrets/promptise/token``). Requires no boto3.

    Args:
        token_file: Projected-token path. Defaults to
            ``$PROMPTISE_IDENTITY_TOKEN_FILE`` or, if unset,
            :data:`DEFAULT_PROMPTISE_IDENTITY_TOKEN_FILE`.
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
                ENV_PROMPTISE_IDENTITY_TOKEN_FILE,
                DEFAULT_PROMPTISE_IDENTITY_TOKEN_FILE,
            )
        super().__init__(
            token_file=resolved_path,
            provider_label="aws-eks-projected",
        )


def from_aws(
    *,
    mode: Literal["auto", "sts", "projected"] = "auto",
    region: str | None = None,
    token_file: str | Path | None = None,
    audience: str = _DEFAULT_AUDIENCE,
    signing_algorithm: str = _DEFAULT_SIGNING_ALGORITHM,
) -> AwsStsProvider | AwsEksProjectedProvider:
    """Build an AWS IAM credential source.

    Args:
        mode: ``"auto"`` (default) picks projected when
            ``PROMPTISE_IDENTITY_TOKEN_FILE`` is set, otherwise STS.
            Force a mode with ``"sts"`` or ``"projected"``.
        region: AWS region for STS mode. Falls back to ``AWS_REGION`` /
            ``AWS_DEFAULT_REGION``.
        token_file: Projected-token path for projected mode. Falls back
            to ``$PROMPTISE_IDENTITY_TOKEN_FILE``.
        audience: Audience for the STS web-identity token (STS mode).
        signing_algorithm: JWT signing algorithm for STS (STS mode).

    Returns:
        An :class:`AwsStsProvider` for STS mode or an
        :class:`AwsEksProjectedProvider` for projected mode.

    Raises:
        ProviderConfigError: If ``mode`` is not ``auto``/``sts``/
            ``projected``, or if STS mode cannot resolve a region.
    """
    resolved_mode = mode
    if resolved_mode == "auto":
        resolved_mode = "projected" if os.environ.get(ENV_PROMPTISE_IDENTITY_TOKEN_FILE) else "sts"

    if resolved_mode == "projected":
        return AwsEksProjectedProvider(token_file=token_file)
    if resolved_mode == "sts":
        return AwsStsProvider(
            region=region,
            audience=audience,
            signing_algorithm=signing_algorithm,
        )
    raise ProviderConfigError(
        f"Unknown AWS mode {mode!r}. Valid modes are 'auto', 'sts', and 'projected'."
    )
