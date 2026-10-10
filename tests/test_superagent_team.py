"""`.superagent` files: auth, approval, paths, .env, whole-team building and the CLI."""

from __future__ import annotations

import os
import sys
import textwrap
import warnings
from pathlib import Path
from typing import Any

import pytest
from _scripted_model import Scripted
from typer.testing import CliRunner

import promptise.agent as agent_module
from promptise.approval import QueueApprovalHandler, WebhookApprovalHandler
from promptise.exceptions import SuperAgentError, SuperAgentValidationError
from promptise.superagent import SuperAgentLoader, build_superagent, load_superagent_file

BILLING_SERVER = str(Path(__file__).with_name("_superagent_billing_server.py"))


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip())
    return path


def _errors(exc: SuperAgentValidationError) -> str:
    return " | ".join(str(e["msg"]) for e in exc.errors)


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")


@pytest.fixture
def fake_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve model strings ``fake:<mode>`` to the scripted model."""
    original = agent_module._normalize_model

    def normalize(model: Any) -> Any:
        if isinstance(model, str) and model.startswith("fake:"):
            return Scripted(mode=model[5:])
        return original(model)

    monkeypatch.setattr(agent_module, "_normalize_model", normalize)


# ---------------------------------------------------------------------------
# HTTP server auth
# ---------------------------------------------------------------------------


def test_auth_field_is_rejected_with_a_clear_message(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.superagent",
        """
        agent: {model: "openai:gpt-5-mini"}
        servers:
          incidents: {type: http, url: "http://127.0.0.1:8270/mcp", auth: "${KEY}"}
        """,
    )
    with pytest.raises(SuperAgentValidationError) as exc:
        SuperAgentLoader.from_file(path)
    msg = _errors(exc.value)
    assert "'auth' is not supported: it was never sent to the server" in msg
    assert "bearer_token" in msg and "api_key" in msg


def test_bearer_token_and_api_key_resolve_from_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INCIDENTS_KEY", "k-1")
    monkeypatch.setenv("TOOLS_TOKEN", "t-1")
    path = _write(
        tmp_path / "a.superagent",
        """
        agent: {model: "openai:gpt-5-mini"}
        servers:
          incidents: {type: http, url: "http://x/mcp", api_key: "${INCIDENTS_KEY}"}
          tools: {type: http, url: "http://y/mcp", bearer_token: "${TOOLS_TOKEN}"}
        """,
    )
    loader, _ = load_superagent_file(path)
    specs = loader.to_server_specs()
    assert specs["incidents"].api_key.get_secret_value() == "k-1"
    assert specs["incidents"].bearer_token is None
    assert specs["tools"].bearer_token.get_secret_value() == "t-1"


async def test_api_key_reaches_the_mcp_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HTTPServerSpec credentials from a file are what build_agent hands to MCPClient."""
    seen: dict[str, Any] = {}

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            seen.update(kwargs)

    class _Multi:
        def __init__(self, clients: Any) -> None:
            pass

        async def __aenter__(self) -> _Multi:
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

    class _Adapter:
        def __init__(self, *a: Any, **k: Any) -> None:
            pass

        async def as_langchain_tools(self) -> list[Any]:
            return []

    import promptise.mcp.client as client_module

    monkeypatch.setattr(client_module, "MCPClient", _Client)
    monkeypatch.setattr(client_module, "MCPMultiClient", _Multi)
    monkeypatch.setattr(client_module, "MCPToolAdapter", _Adapter)
    monkeypatch.setenv("INCIDENTS_KEY", "k-1")
    path = _write(
        tmp_path / "a.superagent",
        """
        agent: {model: "openai:gpt-5-mini"}
        servers:
          incidents: {type: http, url: "http://x/mcp", api_key: "${INCIDENTS_KEY}"}
        """,
    )
    loader, _ = load_superagent_file(path)
    kwargs = loader.to_agent_config().to_build_kwargs()
    agent = await agent_module.build_agent(**{**kwargs, "model": Scripted()})
    await agent.shutdown()
    assert seen["api_key"] == "k-1"


def test_python_spec_auth_warns_that_it_is_ignored() -> None:
    from promptise.config import HTTPServerSpec

    with pytest.warns(FutureWarning, match="HTTPServerSpec.auth is ignored"):
        HTTPServerSpec(url="http://x/mcp", auth="secret")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HTTPServerSpec(url="http://x/mcp", api_key="secret")


# ---------------------------------------------------------------------------
# Approval handler
# ---------------------------------------------------------------------------

_APPROVAL_BASE = """
agent: {model: "openai:gpt-5-mini"}
servers:
  t: {type: http, url: "http://x/mcp"}
approval:
  tools: ["get_runbook"]
"""


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ("  handler: callback\n", "handler 'callback' needs a Python function"),
        ("  handler: webhook\n", "webhook_url is required when handler is 'webhook'"),
        ("  handler: queue\n  webhook_secret: s\n", "webhook_secret applies only to"),
    ],
)
def test_unbuildable_approval_is_rejected_at_load(tmp_path: Path, extra: str, message: str) -> None:
    path = tmp_path / "a.superagent"
    path.write_text(textwrap.dedent(_APPROVAL_BASE).lstrip() + extra)
    with pytest.raises(SuperAgentValidationError) as exc:
        SuperAgentLoader.from_file(path)
    assert message in _errors(exc.value)


def test_webhook_secret_is_passed_to_the_handler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("APPROVAL_SECRET", "shh")
    path = tmp_path / "a.superagent"
    path.write_text(
        textwrap.dedent(_APPROVAL_BASE).lstrip()
        + "  handler: webhook\n"
        + "  webhook_url: https://approvals.example.com/requests\n"
        + '  webhook_secret: "${APPROVAL_SECRET}"\n'
    )
    loader, _ = load_superagent_file(path)
    policy = loader.to_agent_config().to_build_kwargs()["approval"]
    assert isinstance(policy.handler, WebhookApprovalHandler)
    assert policy.handler._secret == "shh"


def test_queue_handler_builds(tmp_path: Path) -> None:
    path = tmp_path / "a.superagent"
    path.write_text(textwrap.dedent(_APPROVAL_BASE).lstrip() + "  handler: queue\n")
    loader, _ = load_superagent_file(path)
    policy = loader.to_agent_config().to_build_kwargs()["approval"]
    assert isinstance(policy.handler, QueueApprovalHandler)


# ---------------------------------------------------------------------------
# version, delegation settings
# ---------------------------------------------------------------------------


def test_version_is_optional(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.superagent",
        """
        agent: {model: "openai:gpt-5-mini"}
        servers: {t: {type: http, url: "http://x/mcp"}}
        """,
    )
    assert SuperAgentLoader.from_file(path).schema.version == "1.0"


def test_delegation_settings_reach_build_kwargs(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.superagent",
        """
        agent: {model: "openai:gpt-5-mini"}
        servers: {t: {type: http, url: "http://x/mcp"}}
        max_delegation_depth: 2
        delegation_timeout: 45
        include_broadcast: true
        """,
    )
    kwargs = SuperAgentLoader.from_file(path).to_agent_config().to_build_kwargs()
    assert kwargs["max_delegation_depth"] == 2
    assert kwargs["delegation_timeout"] == 45
    assert kwargs["include_broadcast"] is True


def test_delegation_depth_must_be_positive(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.superagent",
        """
        agent: {model: "openai:gpt-5-mini"}
        servers: {t: {type: http, url: "http://x/mcp"}}
        max_delegation_depth: 0
        """,
    )
    with pytest.raises(SuperAgentValidationError):
        SuperAgentLoader.from_file(path)


# ---------------------------------------------------------------------------
# stdio paths are relative to the file
# ---------------------------------------------------------------------------


def test_stdio_paths_resolve_relative_to_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "oncall-agent"
    path = _write(
        folder / "a.superagent",
        """
        agent: {model: "openai:gpt-5-mini"}
        servers:
          default: {type: stdio, command: python, args: ["incidents_server.py"]}
          relative: {type: stdio, command: ./bin/server, cwd: tools}
          absolute: {type: stdio, command: /usr/bin/env, cwd: /tmp}
        """,
    )
    monkeypatch.chdir(tmp_path)  # run from the parent folder, as in the bug report
    specs = SuperAgentLoader.from_file("oncall-agent/a.superagent").to_server_specs()
    assert specs["default"].command == "python"  # bare names still come from PATH
    assert specs["default"].cwd == str(path.parent.resolve())
    assert specs["relative"].command == str(path.parent.resolve() / "bin" / "server")
    assert specs["relative"].cwd == str(path.parent.resolve() / "tools")
    assert specs["absolute"].command == "/usr/bin/env"
    assert specs["absolute"].cwd == "/tmp"


async def test_stdio_server_starts_from_another_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / "team"
    (folder).mkdir()
    (folder / "server.py").write_text(Path(BILLING_SERVER).read_text())
    _write(
        folder / "a.superagent",
        f"""
        agent: {{model: "openai:gpt-5-mini"}}
        servers:
          billing: {{type: stdio, command: "{sys.executable}", args: ["server.py"]}}
        """,
    )
    monkeypatch.chdir(tmp_path)
    loader, _ = load_superagent_file("team/a.superagent")
    kwargs = loader.to_agent_config().to_build_kwargs()
    agent = await agent_module.build_agent(**{**kwargs, "model": Scripted()})
    try:
        assert [t.name for t in agent._tools] == ["get_account"]
    finally:
        await agent.shutdown()


# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------


def test_load_superagent_file_reads_dotenv_like_the_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PROMPTISE_NO_DOTENV", raising=False)
    monkeypatch.delenv("PAGER_API_TOKEN", raising=False)
    (tmp_path / "pyproject.toml").write_text("")
    dotenv = tmp_path / ".env"
    dotenv.write_text("PAGER_API_TOKEN=from-dotenv\n")
    os.chmod(dotenv, 0o600)
    _write(
        tmp_path / "a.superagent",
        """
        agent: {model: "openai:gpt-5-mini"}
        servers:
          s: {type: stdio, command: python, env: {PAGER_API_TOKEN: "${PAGER_API_TOKEN}"}}
        """,
    )
    monkeypatch.chdir(tmp_path)
    try:
        loader, _ = load_superagent_file("a.superagent")
        assert loader.to_server_specs()["s"].env == {"PAGER_API_TOKEN": "from-dotenv"}
    finally:
        os.environ.pop("PAGER_API_TOKEN", None)


# ---------------------------------------------------------------------------
# Whole-team building
# ---------------------------------------------------------------------------


def _team(tmp_path: Path) -> Path:
    """coordinator → support → billing (stdio MCP server)."""
    _write(
        tmp_path / "agents" / "billing.superagent",
        f"""
        agent:
          model: "fake:tool:get_account"
          trace: false
        servers:
          billing: {{type: stdio, command: "{sys.executable}", args: ["{BILLING_SERVER}"]}}
        max_invocation_time: 30
        """,
    )
    _write(
        tmp_path / "agents" / "support.superagent",
        """
        agent:
          model: "fake:tool:ask_agent_billing"
          trace: false
        cross_agents:
          billing: {file: ./billing.superagent, description: "Accounts.", timeout: 20}
        approval:
          tools: ["nothing_matches_*"]
          handler: queue
        """,
    )
    return _write(
        tmp_path / "coordinator.superagent",
        """
        agent:
          model: "fake:tool:ask_agent_support"
          trace: false
        cross_agents:
          support: {file: ./agents/support.superagent, description: "Support desk."}
        """,
    )


def test_cross_agents_resolve_at_every_depth(tmp_path: Path) -> None:
    loader, children = load_superagent_file(_team(tmp_path))
    assert list(children) == ["support"]
    assert loader.cross_loaders is children
    assert list(children["support"].cross_loaders or {}) == ["billing"]


def test_to_build_kwargs_warns_when_cross_agents_are_dropped(tmp_path: Path) -> None:
    loader, _ = load_superagent_file(_team(tmp_path))
    with pytest.warns(UserWarning, match="build_superagent"):
        kwargs = loader.to_agent_config().to_build_kwargs()
    assert "cross_agents" not in kwargs
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        kwargs = loader.to_agent_config().to_build_kwargs(cross_agents={})
    assert kwargs["cross_agents"] == {}


async def test_build_superagent_builds_the_whole_team_in_one_loop(
    tmp_path: Path, fake_models: None
) -> None:
    team = await build_superagent(_team(tmp_path))
    try:
        support = team._owned_agents[0]
        billing = support._owned_agents[0]
        # Each agent keeps its own file's settings.
        assert [t.name for t in team._tools] == ["ask_agent_support"]
        assert isinstance(support._approval.handler, QueueApprovalHandler)
        assert billing._max_invocation_time == 30
        # The stdio session opened while building is usable when the team runs.
        result = await team.ainvoke({"messages": [{"role": "user", "content": "acme?"}]})
        answer = result["messages"][-1].content
        assert '"plan": "Team"' in answer and "Not connected" not in answer
    finally:
        await team.shutdown()
    assert team._owned_agents == []
    assert billing._mcp_multi is None  # specialists are shut down with the coordinator


async def test_build_superagent_overrides_apply_to_the_top_agent(
    tmp_path: Path, fake_models: None
) -> None:
    team = await build_superagent(_team(tmp_path), model="fake:answer", trace=True)
    try:
        result = await team.ainvoke({"messages": [{"role": "user", "content": "hi"}]})
        assert result["messages"][-1].content == "ok"
    finally:
        await team.shutdown()


async def test_failed_build_shuts_down_the_agents_already_built(
    tmp_path: Path, fake_models: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _team(tmp_path)
    # The coordinator gets a server that exits at once; its peers build first.
    text = path.read_text() + (
        f'servers:\n  broken: {{type: stdio, command: "{sys.executable}", '
        'args: ["-c", "import sys; sys.exit(3)"]}\n'
    )
    path.write_text(text)
    closed: list[str] = []
    original = agent_module.PromptiseAgent.shutdown

    async def spy(self: Any) -> None:
        closed.append(self.model_name or "?")
        await original(self)

    monkeypatch.setattr(agent_module.PromptiseAgent, "shutdown", spy)
    with pytest.raises(Exception, match="broken"):
        await build_superagent(path)
    assert len(closed) == 2  # support and billing


async def test_failed_specialist_names_its_file(tmp_path: Path, fake_models: None) -> None:
    _write(
        tmp_path / "bad.superagent",
        f"""
        agent: {{model: "fake:answer"}}
        servers:
          broken: {{type: stdio, command: "{sys.executable}", args: ["-c", "raise SystemExit(3)"]}}
        """,
    )
    path = _write(
        tmp_path / "top.superagent",
        """
        agent: {model: "fake:answer"}
        cross_agents:
          bad: {file: ./bad.superagent}
        """,
    )
    with pytest.raises(SuperAgentError, match=r"Failed to build cross-agent 'bad' from .*bad"):
        await build_superagent(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(args: list[str], input: str | None = None) -> Any:
    from promptise.cli import app

    return CliRunner().invoke(app, args, input=input, env={"COLUMNS": "200"})


def test_validate_fails_on_missing_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PAGER_API_TOKEN", raising=False)
    path = _write(
        tmp_path / "a.superagent",
        """
        agent: {model: "openai:gpt-5-mini"}
        servers:
          s: {type: stdio, command: python, env: {PAGER_API_TOKEN: "${PAGER_API_TOKEN}"}}
        """,
    )
    failed = _cli(["validate", str(path)])
    assert failed.exit_code == 1
    assert "Missing environment variables" in failed.output
    assert "PAGER_API_TOKEN" in failed.output

    allowed = _cli(["validate", str(path), "--allow-missing-env"])
    assert allowed.exit_code == 0, allowed.output
    assert "PAGER_API_TOKEN" in allowed.output

    skipped = _cli(["validate", str(path), "--no-check-env"])
    assert skipped.exit_code == 0, skipped.output


def test_validate_checks_env_of_every_cross_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEP_TOKEN", raising=False)
    _write(
        tmp_path / "c.superagent",
        """
        agent: {model: "openai:gpt-5-mini"}
        servers: {t: {type: http, url: "http://x/mcp", api_key: "${DEEP_TOKEN}"}}
        """,
    )
    _write(
        tmp_path / "b.superagent",
        """
        agent: {model: "openai:gpt-5-mini"}
        cross_agents: {c: {file: ./c.superagent}}
        """,
    )
    top = _write(
        tmp_path / "a.superagent",
        """
        agent: {model: "openai:gpt-5-mini"}
        cross_agents: {b: {file: ./b.superagent}}
        """,
    )
    result = _cli(["validate", str(top)])
    assert result.exit_code == 1
    assert "All 2 cross-agent reference(s) valid" in result.output
    assert "DEEP_TOKEN" in result.output and "c.superagent" in result.output


def test_validate_rejects_callback_approval(tmp_path: Path) -> None:
    path = tmp_path / "a.superagent"
    path.write_text(textwrap.dedent(_APPROVAL_BASE).lstrip() + "  handler: callback\n")
    result = _cli(["validate", str(path)])
    assert result.exit_code == 1
    assert "handler 'callback' needs a Python function" in result.output


def test_agent_reports_a_failing_stdio_server_in_one_line(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "a.superagent",
        f"""
        agent: {{model: "fake:answer"}}
        servers:
          incidents: {{type: stdio, command: "{sys.executable}", args: ["missing_server.py"]}}
        """,
    )
    result = _cli(["agent", str(path)], input="exit\n")
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    lines = [line for line in result.output.splitlines() if "Failed to start the agent" in line]
    assert len(lines) == 1
    assert "Failed to connect to server 'incidents'" in lines[0]


def test_agent_runs_a_team_with_a_stdio_specialist(tmp_path: Path, fake_models: None) -> None:
    result = _cli(["agent", str(_team(tmp_path))], input="What plan is acme on?\nexit\n")
    assert result.exit_code == 0, result.output
    assert "Loaded 2 cross-agent(s)" in result.output
    assert '"plan": "Team"' in result.output
    assert "Not connected" not in result.output
    # A queue approval cannot be answered without a terminal; the CLI says so.
    assert "approval requests cannot be answered here" in result.output


def _gate_billing(path: Path, timeout: float) -> None:
    """Require approval for the billing specialist's get_account tool."""
    billing = path.parent / "agents" / "billing.superagent"
    billing.write_text(
        billing.read_text()
        + f"approval:\n  tools: [get_account]\n  handler: queue\n  timeout: {timeout}\n"
    )


def test_agent_enforces_a_cross_agents_own_approval(tmp_path: Path, fake_models: None) -> None:
    """Security: a specialist's approval applies when it runs inside a team.

    Before the fix, `promptise agent` built each cross-agent from its servers,
    model and instructions only, so the specialist ran get_account with no
    approval at all.
    """
    path = _team(tmp_path)
    _gate_billing(path, timeout=1)
    result = _cli(["agent", str(path)], input="What plan is acme on?\nexit\n")
    assert result.exit_code == 0, result.output
    assert "DENIED" in result.output
    assert '"plan": "Team"' not in result.output


async def test_terminal_answers_a_specialists_queue_approval(
    tmp_path: Path, fake_models: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """At a terminal, `promptise agent` asks each queue approval as a y/N question."""
    import builtins

    from promptise import cli

    path = _team(tmp_path)
    _gate_billing(path, timeout=20)
    team = await build_superagent(path)
    answers = iter(["y", "n"])
    prompts: list[str] = []
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(builtins, "input", lambda *a: next(answers))
    monkeypatch.setattr(cli.console, "print", lambda *a, **k: prompts.append(" ".join(map(str, a))))
    tasks = cli._start_approval_prompts(team, cli._LineReader())
    try:
        assert len(tasks) == 2  # support's and billing's queue handlers
        question = {"messages": [{"role": "user", "content": "acme?"}]}
        approved = (await team.ainvoke(question))["messages"][-1].content
        denied = (await team.ainvoke(question))["messages"][-1].content
    finally:
        for task in tasks:
            task.cancel()
        await team.shutdown()
    assert '"plan": "Team"' in approved
    assert "DENIED" in denied and '"plan": "Team"' not in denied
    assert sum("Allow get_account? [y/N]" in p for p in prompts) == 2


def test_http_auth_option_is_rejected() -> None:
    import typer

    from promptise.cli import _merge_servers

    with pytest.raises(typer.BadParameter, match="bearer_token=<token> or api_key=<key>"):
        _merge_servers([], ["name=r url=http://x/mcp auth=secret"])
    spec = _merge_servers([], ["name=r url=http://x/mcp api_key=secret"])["r"]
    assert spec.api_key.get_secret_value() == "secret"


@pytest.mark.parametrize("template", ["basic", "http", "stdio", "advanced"])
def test_init_templates_pass_the_schema(tmp_path: Path, template: str) -> None:
    out = tmp_path / f"{template}.superagent"
    result = _cli(["init", "--output", str(out), "--template", template])
    assert result.exit_code == 0, result.output
    SuperAgentLoader.from_file(out)  # no validation error
