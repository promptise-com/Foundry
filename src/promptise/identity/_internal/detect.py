"""Platform auto-detection for :meth:`AgentIdentity.auto`.

A workload usually knows which cloud it runs on without being told —
the runtime sets characteristic environment variables. This module
reads those markers and returns a short platform identifier that
:meth:`AgentIdentity.auto` dispatches on.

The detection order (build plan section 5.13) follows the rough order
of cloud market share among likely Promptise users, and is
deterministic: when a workload sets markers for more than one platform
(an AKS pod that also has AWS credentials mounted, say), the
**first** match in the table wins. Entra precedes AWS precedes GCP
precedes SPIFFE. No metadata-server probe is performed — detection is
purely environment-variable based, so it is fast, side-effect free,
and safe to call when offline.
"""

from __future__ import annotations

import os
from typing import Literal

from .._core.errors import PlatformDetectionError

#: Platform identifier returned by :func:`detect_platform`.
Platform = Literal["entra", "aws", "gcp", "spiffe"]

#: Entra markers: AKS Workload Identity projects a token file; VM/MSI
#: workloads expose the managed-identity client id.
_ENTRA_ENV_VARS: tuple[str, ...] = (
    "AZURE_FEDERATED_TOKEN_FILE",
    "AZURE_CLIENT_ID",
)

#: AWS markers: Lambda, the generic execution-env stamp, the EKS pod
#: name, IRSA's web-identity token file, and the Promptise projected token
#: file. They line up with the two modes :func:`from_aws` picks between:
#: ``PROMPTISE_IDENTITY_TOKEN_FILE`` selects projected mode (that file is
#: what it presents); every other marker, IRSA included, leads to STS mode.
#: IRSA's own token (``AWS_WEB_IDENTITY_TOKEN_FILE``) is never presented:
#: its audience is ``sts.amazonaws.com``. It only signals that boto3 has a
#: role to call STS with.
_AWS_ENV_VARS: tuple[str, ...] = (
    "AWS_LAMBDA_FUNCTION_NAME",
    "AWS_EXECUTION_ENV",
    "EKS_POD_NAME",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "PROMPTISE_IDENTITY_TOKEN_FILE",
)

#: GCP markers: the project id, the Cloud Run service name, and the
#: metadata-server IP hint.
_GCP_ENV_VARS: tuple[str, ...] = (
    "GOOGLE_CLOUD_PROJECT",
    "K_SERVICE",
    "GCE_METADATA_IP",
)

#: SPIFFE marker: the Workload API socket address.
_SPIFFE_ENV_VARS: tuple[str, ...] = ("SPIFFE_ENDPOINT_SOCKET",)

#: Ordered detection table. The first platform whose marker set has any
#: variable present wins.
_DETECTION_ORDER: tuple[tuple[Platform, tuple[str, ...]], ...] = (
    ("entra", _ENTRA_ENV_VARS),
    ("aws", _AWS_ENV_VARS),
    ("gcp", _GCP_ENV_VARS),
    ("spiffe", _SPIFFE_ENV_VARS),
)


def _any_env_set(names: tuple[str, ...]) -> bool:
    """Return ``True`` if any of ``names`` is set to a non-empty value.

    An environment variable set to the empty string counts as unset —
    a common shape in CI systems that declare a variable without giving
    it a value.
    """
    return any(os.environ.get(name) for name in names)


def detect_platform() -> Platform:
    """Detect the federation platform from environment markers.

    Returns:
        One of ``"entra"``, ``"aws"``, ``"gcp"``, or ``"spiffe"`` — the
        first platform in the detection order whose markers are present.

    Raises:
        PlatformDetectionError: When no platform marker is found. The
            message names every platform that was probed and points at
            the explicit ``from_*`` factories as the fallback.
    """
    for identifier, env_vars in _DETECTION_ORDER:
        if _any_env_set(env_vars):
            return identifier
    raise PlatformDetectionError(
        "could not auto-detect a federation platform: no Entra, AWS, "
        "GCP, or SPIFFE environment markers were found. Most common "
        "cause: AgentIdentity.auto() was called off-platform (a laptop "
        "or a generic CI runner). Set the platform's environment "
        "variables, or construct the provider explicitly with "
        "AgentIdentity.from_entra(), .from_aws(), .from_gcp(), "
        ".from_spiffe(), or .from_oidc()."
    )
