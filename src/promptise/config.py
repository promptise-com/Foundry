"""Typed server specifications and conversion helpers.

Supports the Promptise MCP Client with token-based authentication
(Bearer token or API key).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr


class _BaseServer(BaseModel):
    """Base model for server specs."""

    model_config = ConfigDict(extra="forbid")


class StdioServerSpec(_BaseServer):
    """Specification for a local MCP server launched via stdio.

    Attributes:
        command: Executable to launch (e.g., ``"python"``).
        args: Positional arguments for the process.
        env: Environment variables to set for the process.
        cwd: Optional working directory.
        keep_alive: Whether the client should try to keep a persistent session.
    """

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    keep_alive: bool = True


class HTTPServerSpec(_BaseServer):
    """Specification for a remote MCP server reachable via HTTP/SSE.

    Supports two authentication methods:

    1. **Bearer token** (JWT): pass a pre-issued token via ``bearer_token``
       and it is automatically injected as ``Authorization: Bearer <token>``.
    2. **API key** (simple secret): pass a pre-shared key via ``api_key``
       and it is automatically injected as ``x-api-key: <key>``.

    Tokens should be obtained from your Identity Provider (Auth0, Keycloak,
    Okta, etc.) or from the MCP server's built-in token endpoint (for
    development/testing).  The agent **never** generates tokens itself.

    Attributes:
        url: Full endpoint URL (e.g., ``http://127.0.0.1:8080/mcp``).
        transport: ``"http"``, ``"streamable-http"``, or ``"sse"``.
        headers: Extra HTTP headers sent on every request.
        auth: Legacy auth hint.
        bearer_token: Pre-issued Bearer token.  When set, an
            ``Authorization: Bearer <token>`` header is created
            automatically.
        api_key: Pre-shared API key.  When set, an ``x-api-key``
            header is created automatically.  Use this for simple
            secret-based auth when JWT is overkill.
        forward_caller_token: When ``True`` (default) and an invocation
            carries a :class:`~promptise.CallerContext` with a
            ``bearer_token``, this server's tools are called with
            ``Authorization: Bearer <caller token>`` over a session
            opened for that caller, so the server sees the user, not
            whoever built the agent.  ``bearer_token`` (or the agent
            identity) is still used for tool discovery and for calls
            without a caller token.  Set ``False`` for a third-party
            server that must not receive your users' tokens, or one that
            should always see the agent's own credential.

    Example — Bearer token::

        HTTPServerSpec(
            url="http://localhost:8080/mcp",
            bearer_token="eyJhbGciOiJIUzI1NiIs...",
        )

    Example — API key::

        HTTPServerSpec(
            url="http://localhost:8080/mcp",
            api_key="my-secret-key",
        )

    Example — manual header::

        HTTPServerSpec(
            url="http://localhost:8080/mcp",
            headers={"authorization": "Bearer <token>"},
        )
    """

    url: str
    transport: Literal["http", "streamable-http", "sse"] = "http"
    headers: dict[str, str] = Field(default_factory=dict)
    auth: str | None = None

    # Resource audience this server expects. When an agent identity is
    # attached and no explicit bearer_token is set, build_agent presents a
    # credential minted for this audience (per-resource identity).
    audience: str | None = Field(
        default=None,
        description="Resource audience this server expects. When an agent "
        "identity is attached, a credential scoped to this audience is "
        "presented automatically.",
    )

    # Token auth — pass a pre-issued Bearer token (from an IdP or server token endpoint)
    bearer_token: SecretStr | None = Field(
        default=None,
        description="Pre-issued Bearer token for authentication. "
        "Obtain from your Identity Provider or the server's token endpoint.",
    )

    # API key auth — simple pre-shared secret (injected as x-api-key header)
    api_key: SecretStr | None = Field(
        default=None,
        description="Pre-shared API key for simple secret-based authentication. "
        "Injected as an x-api-key header.",
    )

    # Per-invocation identity: send the invoking user's token, not the agent's.
    forward_caller_token: bool = Field(
        default=True,
        description="When an invocation carries a CallerContext with a "
        "bearer_token, call this server's tools with that token (one session "
        "per caller) instead of bearer_token / the agent identity. Set False "
        "for servers outside your trust boundary, or whose credential is the "
        "agent's own rather than the user's.",
    )


ServerSpec = StdioServerSpec | HTTPServerSpec
"""Union of supported server specifications."""


def servers_to_mcp_config(servers: Mapping[str, ServerSpec]) -> dict[str, dict[str, object]]:
    """Convert programmatic server specs to a configuration dict.

    Args:
        servers: Mapping of server name to specification.

    Returns:
        Dict with server configurations keyed by name.
    """
    cfg: dict[str, dict[str, object]] = {}
    for name, s in servers.items():
        if isinstance(s, StdioServerSpec):
            cfg[name] = {
                "transport": "stdio",
                "command": s.command,
                "args": s.args,
                "env": s.env,  # Always pass dict (empty dict is valid)
                "cwd": s.cwd,  # None is acceptable for optional fields
                "keep_alive": s.keep_alive,
            }
        else:
            entry: dict[str, object] = {
                "transport": s.transport,
                "url": s.url,
            }
            if s.headers:
                entry["headers"] = s.headers
            if s.auth is not None:
                entry["auth"] = s.auth
            if s.bearer_token is not None:
                entry["bearer_token"] = s.bearer_token.get_secret_value()
            if s.api_key is not None:
                entry["api_key"] = s.api_key.get_secret_value()
            cfg[name] = entry
    return cfg
