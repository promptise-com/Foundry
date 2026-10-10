"""Unit tests for platform auto-detection and ``AgentIdentity.auto``."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptise.identity import (
    AgentIdentity,
    AwsEksProjectedProvider,
    AwsStsProvider,
    EntraProjectedTokenProvider,
    GcpMetadataProvider,
    PlatformDetectionError,
    SpiffeSdkProvider,
)
from promptise.identity._internal.detect import detect_platform

_ALL_MARKERS: tuple[str, ...] = (
    "AZURE_FEDERATED_TOKEN_FILE",
    "AZURE_CLIENT_ID",
    "AWS_LAMBDA_FUNCTION_NAME",
    "AWS_EXECUTION_ENV",
    "EKS_POD_NAME",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "PROMPTISE_IDENTITY_TOKEN_FILE",
    "GOOGLE_CLOUD_PROJECT",
    "K_SERVICE",
    "GCE_METADATA_IP",
    "SPIFFE_ENDPOINT_SOCKET",
)


@pytest.fixture(autouse=True)
def _clean_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ALL_MARKERS:
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("marker", ["AZURE_FEDERATED_TOKEN_FILE", "AZURE_CLIENT_ID"])
def test_detects_entra(monkeypatch: pytest.MonkeyPatch, marker: str) -> None:
    monkeypatch.setenv(marker, "x")
    assert detect_platform() == "entra"


@pytest.mark.parametrize(
    "marker",
    [
        "AWS_LAMBDA_FUNCTION_NAME",
        "AWS_EXECUTION_ENV",
        "EKS_POD_NAME",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
        "PROMPTISE_IDENTITY_TOKEN_FILE",
    ],
)
def test_detects_aws(monkeypatch: pytest.MonkeyPatch, marker: str) -> None:
    monkeypatch.setenv(marker, "x")
    assert detect_platform() == "aws"


@pytest.mark.parametrize("marker", ["GOOGLE_CLOUD_PROJECT", "K_SERVICE", "GCE_METADATA_IP"])
def test_detects_gcp(monkeypatch: pytest.MonkeyPatch, marker: str) -> None:
    monkeypatch.setenv(marker, "x")
    assert detect_platform() == "gcp"


def test_detects_spiffe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPIFFE_ENDPOINT_SOCKET", "unix:///a.sock")
    assert detect_platform() == "spiffe"


def test_precedence_entra_over_aws(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_CLIENT_ID", "x")
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "y")
    assert detect_platform() == "entra"


def test_aws_over_gcp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_EXECUTION_ENV", "x")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "y")
    assert detect_platform() == "aws"


def test_gcp_over_spiffe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("K_SERVICE", "x")
    monkeypatch.setenv("SPIFFE_ENDPOINT_SOCKET", "unix:///a.sock")
    assert detect_platform() == "gcp"


def test_empty_marker_counts_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_CLIENT_ID", "")
    with pytest.raises(PlatformDetectionError):
        detect_platform()


def test_no_markers_raises() -> None:
    with pytest.raises(PlatformDetectionError, match="could not auto-detect"):
        detect_platform()


# -- AgentIdentity.auto dispatch ------------------------------------------


def test_auto_entra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f = tmp_path / "t"
    f.write_text("h.p.s", encoding="utf-8")
    monkeypatch.setenv("AZURE_FEDERATED_TOKEN_FILE", str(f))
    ident = AgentIdentity.auto("bot")
    assert ident.agent_id == "bot"
    assert isinstance(ident.credential, EntraProjectedTokenProvider)


def test_auto_aws(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    f = tmp_path / "t"
    f.write_text("h.p.s", encoding="utf-8")
    monkeypatch.setenv("EKS_POD_NAME", "pod")
    monkeypatch.setenv("PROMPTISE_IDENTITY_TOKEN_FILE", str(f))
    assert isinstance(AgentIdentity.auto("bot").credential, AwsEksProjectedProvider)


def test_auto_aws_projected_file_alone_selects_projected_mode(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The file projected mode reads is enough for auto() to find AWS."""
    f = tmp_path / "t"
    f.write_text("h.p.s", encoding="utf-8")
    monkeypatch.setenv("PROMPTISE_IDENTITY_TOKEN_FILE", str(f))
    credential = AgentIdentity.auto("bot").credential
    assert isinstance(credential, AwsEksProjectedProvider)
    assert credential.token_file == f


def test_auto_aws_irsa_selects_sts_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """IRSA's token (audience sts.amazonaws.com) is never presented: STS mode."""
    monkeypatch.setenv("AWS_WEB_IDENTITY_TOKEN_FILE", "/var/run/secrets/eks/token")
    monkeypatch.setenv("AWS_REGION", "eu-central-1")
    assert isinstance(AgentIdentity.auto("bot").credential, AwsStsProvider)


def test_auto_gcp(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj")
    assert isinstance(AgentIdentity.auto("bot").credential, GcpMetadataProvider)


def test_auto_spiffe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPIFFE_ENDPOINT_SOCKET", "unix:///a.sock")
    assert isinstance(AgentIdentity.auto("bot").credential, SpiffeSdkProvider)


def test_auto_no_platform_raises() -> None:
    with pytest.raises(PlatformDetectionError):
        AgentIdentity.auto("bot")
