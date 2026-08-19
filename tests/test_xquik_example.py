"""Tests for the remote Xquik MCP example."""

from __future__ import annotations

import pytest

from examples.mcp.xquik_agent import (
    DEFAULT_REQUEST,
    XQUIK_MCP_URL,
    request_from_args,
    required_environment,
    xquik_server,
)


def test_xquik_server_uses_streamable_http_bearer_auth() -> None:
    """Send the key as a bearer token over Streamable HTTP."""
    server = xquik_server("test-key")

    assert server.url == XQUIK_MCP_URL
    assert server.transport == "streamable-http"
    assert server.api_key is None
    assert server.bearer_token is not None
    assert server.bearer_token.get_secret_value() == "test-key"


def test_required_environment_strips_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normalize a configured environment value."""
    monkeypatch.setenv("XQUIK_TEST_VALUE", "  configured  ")

    assert required_environment("XQUIK_TEST_VALUE") == "configured"


def test_required_environment_rejects_blank_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject an absent secret without including a value in the error."""
    monkeypatch.setenv("XQUIK_TEST_VALUE", "  ")

    with pytest.raises(
        RuntimeError,
        match=r"^XQUIK_TEST_VALUE is required\. Set it before running this example\.$",
    ):
        required_environment("XQUIK_TEST_VALUE")


def test_request_from_args_uses_arguments_or_default() -> None:
    """Join explicit arguments and preserve the documented fallback."""
    assert request_from_args(["recent", "agent", "posts"]) == "recent agent posts"
    assert request_from_args([" "]) == DEFAULT_REQUEST
