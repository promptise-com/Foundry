"""Tests for ``build_agent(identity=...)`` — agent identity attribution.

A supplied AgentIdentity is attached as ``agent.identity`` and, by
default, attributes every recorded event to ``identity.agent_id`` so the
observability timeline answers "which agent did what". The identity does
not touch the LLM credential.
"""

from __future__ import annotations

import base64
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptise.agent import PromptiseAgent, _normalize_model, build_agent
from promptise.identity import AgentIdentity

FAKE_JWT = "header.payload.sig"


def _identity() -> AgentIdentity:
    return AgentIdentity("billing-bot", name="Billing Bot")


def _jwt(claims: dict[str, Any]) -> str:
    h = base64.urlsafe_b64encode(b'{"alg":"none"}').rstrip(b"=").decode()
    p = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=").decode()
    return f"{h}.{p}."


def _make_mock_inner() -> MagicMock:
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value={"messages": []})
    mock.invoke = MagicMock(return_value={"messages": []})
    return mock


# -- agent.identity attachment --------------------------------------------


@pytest.mark.asyncio
async def test_build_agent_attaches_identity() -> None:
    identity = _identity()
    with (
        patch("promptise.agent._normalize_model", return_value=MagicMock()),
        patch("promptise.agent.PromptGraphEngine", return_value=_make_mock_inner()),
        patch.dict("sys.modules", {"deepagents": None}),
    ):
        agent = await build_agent(servers={}, model="openai:gpt-5-mini", identity=identity)
    assert isinstance(agent, PromptiseAgent)
    assert agent.identity is identity


@pytest.mark.asyncio
async def test_build_agent_identity_defaults_to_none() -> None:
    with (
        patch("promptise.agent._normalize_model", return_value=MagicMock()),
        patch("promptise.agent.PromptGraphEngine", return_value=_make_mock_inner()),
        patch.dict("sys.modules", {"deepagents": None}),
    ):
        agent = await build_agent(servers={}, model="openai:gpt-5-mini")
    assert agent.identity is None


# -- The identity does not touch the model credential ---------------------


@pytest.mark.asyncio
async def test_identity_is_not_injected_into_the_model() -> None:
    """_normalize_model must be called with the model only — the identity
    is for attribution, never for authenticating the LLM call."""
    with (
        patch("promptise.agent._normalize_model") as mock_norm,
        patch("promptise.agent.PromptGraphEngine", return_value=_make_mock_inner()),
        patch.dict("sys.modules", {"deepagents": None}),
    ):
        mock_norm.return_value = MagicMock()
        await build_agent(servers={}, model="openai:gpt-5-mini", identity=_identity())
    assert mock_norm.call_args.args == ("openai:gpt-5-mini",)
    assert mock_norm.call_args.kwargs == {}


def test_normalize_model_takes_only_the_model() -> None:
    sentinel = MagicMock()
    assert _normalize_model(sentinel) is sentinel


def test_actor_attribution_prefers_identity() -> None:
    """Event notifications attribute to the agent identity when present, and
    fall back to the model name otherwise (no change for non-identity agents)."""
    agent = PromptiseAgent(inner=MagicMock(), model_name="anthropic:claude-x")
    assert agent._actor() == "anthropic:claude-x"  # no identity → model name

    agent.identity = AgentIdentity("billing-bot")
    agent._actor_id = "billing-bot"
    assert agent._actor() == "billing-bot"  # identity → resolved actor id

    agent._actor_id = None  # identity present but subject unresolved at build
    assert agent._actor() == "anthropic:claude-x"  # falls back to model name


@pytest.mark.asyncio
async def test_build_agent_sets_actor_id() -> None:
    with (
        patch("promptise.agent._normalize_model", return_value=MagicMock()),
        patch("promptise.agent.PromptGraphEngine", return_value=_make_mock_inner()),
        patch.dict("sys.modules", {"deepagents": None}),
    ):
        agent = await build_agent(servers={}, model="openai:gpt-5-mini", identity=_identity())
    assert agent._actor_id == "billing-bot"


# -- MCP credential presentation ------------------------------------------


def _patch_mcp(captured: dict[str, Any]) -> list[Any]:
    """Return patch context managers that capture MCPClient kwargs and stub
    the multi-client/adapter so a non-empty `servers` build runs offline."""

    def _fake_client(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return MagicMock()

    multi = MagicMock()
    multi.__aenter__ = AsyncMock(return_value=multi)
    multi.__aexit__ = AsyncMock(return_value=None)
    adapter = MagicMock()
    adapter.as_langchain_tools = AsyncMock(return_value=[])
    return [
        patch("promptise.agent._normalize_model", return_value=MagicMock()),
        patch("promptise.agent.PromptGraphEngine", return_value=_make_mock_inner()),
        patch("promptise.mcp.client.MCPClient", side_effect=_fake_client),
        patch("promptise.mcp.client.MCPMultiClient", return_value=multi),
        patch("promptise.mcp.client.MCPToolAdapter", return_value=adapter),
        patch.dict("sys.modules", {"deepagents": None}),
    ]


@pytest.mark.asyncio
async def test_verifiable_identity_is_presented_to_mcp_server() -> None:
    from contextlib import ExitStack

    from promptise.config import HTTPServerSpec

    token = _jwt({"sub": "agent-x"})
    identity = AgentIdentity.from_oidc("bot", issuer="https://idp", token_fn=lambda: token)
    captured: dict[str, Any] = {}
    with ExitStack() as stack:
        for cm in _patch_mcp(captured):
            stack.enter_context(cm)
        await build_agent(
            servers={"tools": HTTPServerSpec(url="https://mcp.internal")},
            model="openai:gpt-5-mini",
            identity=identity,
        )
    # The client asks for the credential per request (so it is renewed before
    # it expires) instead of receiving one token fetched at build time.
    assert captured["bearer_token"] is None
    assert captured["bearer_token_provider"](False) == token


@pytest.mark.asyncio
async def test_explicit_server_bearer_is_not_overridden() -> None:
    from contextlib import ExitStack

    from promptise.config import HTTPServerSpec

    identity = AgentIdentity.from_oidc(
        "bot", issuer="https://idp", token_fn=lambda: _jwt({"sub": "agent-x"})
    )
    captured: dict[str, Any] = {}
    with ExitStack() as stack:
        for cm in _patch_mcp(captured):
            stack.enter_context(cm)
        await build_agent(
            servers={
                "tools": HTTPServerSpec(url="https://mcp.internal", bearer_token="server-set")
            },
            model="openai:gpt-5-mini",
            identity=identity,
        )
    assert captured["bearer_token"] == "server-set"
    assert captured["bearer_token_provider"] is None


@pytest.mark.asyncio
async def test_per_server_audience_scopes_the_credential() -> None:
    """Each server's ``audience`` is forwarded to the credential provider so
    one identity presents a resource-scoped token to each MCP server."""
    from contextlib import ExitStack

    from promptise.config import HTTPServerSpec
    from promptise.identity import CallableTokenProvider

    seen: list[str | None] = []

    def mint(audience: str | None = None) -> str:
        seen.append(audience)
        return f"token-for-{audience}"

    identity = AgentIdentity("bot", credential=CallableTokenProvider(token_fn=mint))

    bearers: list[Any] = []

    def _fake_client(**kwargs: Any) -> MagicMock:
        bearers.append(kwargs["bearer_token_provider"](False))
        return MagicMock()

    multi = MagicMock()
    multi.__aenter__ = AsyncMock(return_value=multi)
    multi.__aexit__ = AsyncMock(return_value=None)
    adapter = MagicMock()
    adapter.as_langchain_tools = AsyncMock(return_value=[])

    with ExitStack() as stack:
        for cm in (
            patch("promptise.agent._normalize_model", return_value=MagicMock()),
            patch(
                "promptise.agent.PromptGraphEngine",
                return_value=_make_mock_inner(),
            ),
            patch("promptise.mcp.client.MCPClient", side_effect=_fake_client),
            patch("promptise.mcp.client.MCPMultiClient", return_value=multi),
            patch("promptise.mcp.client.MCPToolAdapter", return_value=adapter),
            patch.dict("sys.modules", {"deepagents": None}),
        ):
            stack.enter_context(cm)
        await build_agent(
            servers={
                "billing": HTTPServerSpec(url="https://billing.internal", audience="api://billing"),
                "crm": HTTPServerSpec(url="https://crm.internal", audience="api://crm"),
            },
            model="openai:gpt-5-mini",
            identity=identity,
        )
    assert set(seen) == {"api://billing", "api://crm"}
    assert "token-for-api://billing" in bearers
    assert "token-for-api://crm" in bearers


@pytest.mark.asyncio
async def test_local_identity_presents_no_mcp_credential() -> None:
    from contextlib import ExitStack

    from promptise.config import HTTPServerSpec

    captured: dict[str, Any] = {}
    with ExitStack() as stack:
        for cm in _patch_mcp(captured):
            stack.enter_context(cm)
        await build_agent(
            servers={"tools": HTTPServerSpec(url="https://mcp.internal")},
            model="openai:gpt-5-mini",
            identity=AgentIdentity("local-bot"),
        )
    assert captured["bearer_token"] is None
    assert captured["bearer_token_provider"] is None


@pytest.mark.asyncio
async def test_unreachable_idp_fails_closed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If the IdP cannot mint a credential, the provider handed to the MCP
    client raises instead of returning ``None``, so the client fails the
    request rather than sending it without the agent's credential."""
    import logging
    from contextlib import ExitStack

    from promptise.config import HTTPServerSpec
    from promptise.identity import CallableTokenProvider, CredentialAcquisitionError

    healthy = {"value": False}
    token = _jwt({"sub": "bot", "exp": 4102444800})

    def mint(audience: str | None = None) -> str:
        if not healthy["value"]:
            raise CredentialAcquisitionError("metadata server unreachable")
        return token

    identity = AgentIdentity("bot", credential=CallableTokenProvider(token_fn=mint))

    captured: dict[str, Any] = {}
    with ExitStack() as stack:
        for cm in _patch_mcp(captured):
            stack.enter_context(cm)
        with caplog.at_level(logging.INFO):
            agent = await build_agent(
                servers={"tools": HTTPServerSpec(url="https://mcp.internal")},
                model="openai:gpt-5-mini",
                identity=identity,
            )
            assert isinstance(agent, PromptiseAgent)
            provide = captured["bearer_token_provider"]
            # Never None (which would mean "send it unauthenticated").
            with pytest.raises(CredentialAcquisitionError):
                provide(False)
            with pytest.raises(CredentialAcquisitionError):
                provide(True)
            healthy["value"] = True
            assert provide(False) == token
    # The operator is told, loudly — but once per outage, not per request —
    # and told again when it recovers.
    warnings = [r for r in caplog.records if "could not acquire a credential" in r.getMessage()]
    assert len(warnings) == 1
    assert "requests to it fail" in warnings[0].getMessage()
    assert any("acquired a credential" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_authorization_header_counts_as_the_servers_own_bearer() -> None:
    """A server configured with its own Authorization header keeps it; the
    identity is not presented on top of (or instead of) it."""
    from contextlib import ExitStack

    from promptise.config import HTTPServerSpec

    identity = AgentIdentity.from_oidc(
        "bot", issuer="https://idp", token_fn=lambda: _jwt({"sub": "agent-x"})
    )
    captured: dict[str, Any] = {}
    with ExitStack() as stack:
        for cm in _patch_mcp(captured):
            stack.enter_context(cm)
        await build_agent(
            servers={
                "tools": HTTPServerSpec(
                    url="https://mcp.internal", headers={"Authorization": "Bearer own"}
                )
            },
            model="openai:gpt-5-mini",
            identity=identity,
        )
    assert captured["bearer_token_provider"] is None
    assert captured["headers"] == {"Authorization": "Bearer own"}


@pytest.mark.asyncio
async def test_identity_credential_is_refreshed_per_request() -> None:
    """A renewed credential is what the next request presents, and a 401
    (force_refresh=True) bypasses the identity's cache."""
    from contextlib import ExitStack

    from promptise.config import HTTPServerSpec
    from promptise.identity import CallableTokenProvider

    minted: list[str] = []

    def mint(audience: str | None = None) -> str:
        minted.append(_jwt({"sub": "bot", "n": len(minted), "exp": 4102444800}))
        return minted[-1]

    identity = AgentIdentity("bot", credential=CallableTokenProvider(token_fn=mint))
    captured: dict[str, Any] = {}
    with ExitStack() as stack:
        for cm in _patch_mcp(captured):
            stack.enter_context(cm)
        await build_agent(
            servers={"tools": HTTPServerSpec(url="https://mcp.internal", audience="api://t")},
            model="openai:gpt-5-mini",
            identity=identity,
        )
    provide = captured["bearer_token_provider"]
    first = provide(False)
    assert provide(False) == first  # still valid: served from the cache
    refreshed = provide(True)  # the server answered 401
    assert refreshed != first
    assert provide(False) == refreshed
    assert len(minted) == 2


# -- Attribution: recorded events are stamped with the agent identity -----


@pytest.mark.asyncio
async def test_observability_is_attributed_to_the_agent() -> None:
    captured: dict[str, Any] = {}

    def _fake_handler(collector: Any, *, agent_id: str, **kwargs: Any) -> MagicMock:
        captured["agent_id"] = agent_id
        return MagicMock()

    with (
        patch("promptise.agent._normalize_model", return_value=MagicMock()),
        patch("promptise.agent.PromptGraphEngine", return_value=_make_mock_inner()),
        patch(
            "promptise.callback_handler.PromptiseCallbackHandler",
            side_effect=_fake_handler,
        ),
        patch.dict("sys.modules", {"deepagents": None}),
    ):
        await build_agent(
            servers={},
            model="openai:gpt-5-mini",
            identity=_identity(),
            observe=True,
        )
    assert captured["agent_id"] == "billing-bot"


@pytest.mark.asyncio
async def test_attribution_uses_idp_subject_when_no_agent_id() -> None:
    """A verifiable identity with no local agent_id is attributed to the IdP
    subject read from its credential."""
    captured: dict[str, Any] = {}

    def _fake_handler(collector: Any, *, agent_id: str, **kwargs: Any) -> MagicMock:
        captured["agent_id"] = agent_id
        return MagicMock()

    token = _jwt({"sub": "spiffe://acme/billing-bot"})
    identity = AgentIdentity.from_oidc(issuer="https://idp", token_fn=lambda: token)
    assert identity.agent_id is None

    with (
        patch("promptise.agent._normalize_model", return_value=MagicMock()),
        patch("promptise.agent.PromptGraphEngine", return_value=_make_mock_inner()),
        patch(
            "promptise.callback_handler.PromptiseCallbackHandler",
            side_effect=_fake_handler,
        ),
        patch.dict("sys.modules", {"deepagents": None}),
    ):
        agent = await build_agent(
            servers={}, model="openai:gpt-5-mini", identity=identity, observe=True
        )
    assert captured["agent_id"] == "spiffe://acme/billing-bot"
    assert agent.identity is identity


@pytest.mark.asyncio
async def test_explicit_observer_agent_id_wins() -> None:
    captured: dict[str, Any] = {}

    def _fake_handler(collector: Any, *, agent_id: str, **kwargs: Any) -> MagicMock:
        captured["agent_id"] = agent_id
        return MagicMock()

    with (
        patch("promptise.agent._normalize_model", return_value=MagicMock()),
        patch("promptise.agent.PromptGraphEngine", return_value=_make_mock_inner()),
        patch(
            "promptise.callback_handler.PromptiseCallbackHandler",
            side_effect=_fake_handler,
        ),
        patch.dict("sys.modules", {"deepagents": None}),
    ):
        await build_agent(
            servers={},
            model="openai:gpt-5-mini",
            identity=_identity(),
            observer_agent_id="explicit-id",
            observe=True,
        )
    assert captured["agent_id"] == "explicit-id"


@pytest.mark.asyncio
async def test_agent_exposes_its_tools():
    """`agent.tools` / `agent.tool_names` list every tool the model was bound to."""
    from langchain_core.tools import tool

    @tool
    def ping(x: str) -> str:
        """Ping."""
        return x

    # The model is patched like everywhere else in this file: constructing a real
    # provider client needs a key CI does not have, and on macOS it initialises
    # CoreFoundation, after which the subprocess-based hook tests fork-crash.
    with (
        patch("promptise.agent._normalize_model", return_value=MagicMock()),
        patch("promptise.agent.PromptGraphEngine", return_value=_make_mock_inner()),
        patch.dict("sys.modules", {"deepagents": None}),
    ):
        agent = await build_agent(servers={}, model="openai:gpt-5-mini", extra_tools=[ping])
    assert agent.tool_names == ["ping"]
    assert [t.name for t in agent.tools] == ["ping"]
    await agent.shutdown()
