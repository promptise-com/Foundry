"""Tests for .superagent file loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from promptise.config import HTTPServerSpec, StdioServerSpec
from promptise.exceptions import SuperAgentError, SuperAgentValidationError
from promptise.superagent import SuperAgentLoader, load_superagent_file


def test_load_valid_file(tmp_path: Path) -> None:
    """Test loading a valid .superagent file."""
    config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
  trace: true
servers:
  test:
    type: http
    url: "http://test"
"""
    file_path = tmp_path / "test.superagent"
    file_path.write_text(config)

    loader = SuperAgentLoader.from_file(file_path)
    assert loader.schema.version == "1.0"
    assert loader.schema.agent.model == "openai:gpt-4.1"
    assert "test" in loader.schema.servers


def test_load_file_with_yaml_extension(tmp_path: Path) -> None:
    """Test loading with .superagent.yaml extension."""
    config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
servers:
  test:
    type: http
    url: "http://test"
"""
    file_path = tmp_path / "test.superagent.yaml"
    file_path.write_text(config)

    loader = SuperAgentLoader.from_file(file_path)
    assert loader.schema.version == "1.0"


def test_load_file_with_yml_extension(tmp_path: Path) -> None:
    """Test loading with .superagent.yml extension."""
    config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
servers:
  test:
    type: http
    url: "http://test"
"""
    file_path = tmp_path / "test.superagent.yml"
    file_path.write_text(config)

    loader = SuperAgentLoader.from_file(file_path)
    assert loader.schema.version == "1.0"


def test_invalid_extension(tmp_path: Path) -> None:
    """Test rejection of invalid file extension."""
    file_path = tmp_path / "test.yaml"
    file_path.write_text("version: '1.0'")

    with pytest.raises(SuperAgentError, match="Invalid file extension"):
        SuperAgentLoader.from_file(file_path)


def test_file_not_found(tmp_path: Path) -> None:
    """Test error when file doesn't exist."""
    file_path = tmp_path / "nonexistent.superagent"

    with pytest.raises(SuperAgentError, match="File not found"):
        SuperAgentLoader.from_file(file_path)


def test_invalid_yaml_syntax(tmp_path: Path) -> None:
    """Test error on invalid YAML syntax."""
    config = """
    invalid: yaml: syntax:
      - missing proper
    indentation
    """
    file_path = tmp_path / "test.superagent"
    file_path.write_text(config)

    with pytest.raises(SuperAgentError, match="YAML parse error"):
        SuperAgentLoader.from_file(file_path)


def test_yaml_not_dict(tmp_path: Path) -> None:
    """Test error when YAML is not a dict."""
    config = "- list\n- instead\n- of\n- dict"
    file_path = tmp_path / "test.superagent"
    file_path.write_text(config)

    with pytest.raises(SuperAgentError, match="expected dict"):
        SuperAgentLoader.from_file(file_path)


def test_schema_validation_error(tmp_path: Path) -> None:
    """Test schema validation error handling."""
    config = """version: "1.0"
# Missing required 'agent' field
servers:
  test:
    type: http
    url: "http://test"
"""
    file_path = tmp_path / "test.superagent"
    file_path.write_text(config)

    with pytest.raises(SuperAgentValidationError) as exc_info:
        SuperAgentLoader.from_file(file_path)

    assert "Schema validation failed" in str(exc_info.value)
    assert exc_info.value.file_path == str(file_path)
    assert len(exc_info.value.errors) > 0


def test_validate_env_vars_all_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test env var validation when all vars are available."""
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("TOKEN", "token123")

    config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
servers:
  test:
    type: http
    url: "http://test"
    headers:
      Authorization: "Bearer ${TOKEN}"
    api_key: "${API_KEY}"
"""
    file_path = tmp_path / "test.superagent"
    file_path.write_text(config)

    loader = SuperAgentLoader.from_file(file_path)
    missing = loader.validate_env_vars()
    assert missing == []


def test_validate_env_vars_some_missing(tmp_path: Path) -> None:
    """Test env var validation identifies missing vars."""
    config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
servers:
  test:
    type: http
    url: "${API_URL}"
    headers:
      Authorization: "Bearer ${API_TOKEN}"
"""
    file_path = tmp_path / "test.superagent"
    file_path.write_text(config)

    loader = SuperAgentLoader.from_file(file_path)
    missing = loader.validate_env_vars()
    assert set(missing) == {"API_URL", "API_TOKEN"}


def test_resolve_env_vars_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful environment variable resolution."""
    monkeypatch.setenv("TEST_URL", "http://resolved.com")
    monkeypatch.setenv("TOKEN", "secret123")

    config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
servers:
  test:
    type: http
    url: "${TEST_URL}/mcp"
    headers:
      Authorization: "Bearer ${TOKEN}"
"""
    file_path = tmp_path / "test.superagent"
    file_path.write_text(config)

    loader = SuperAgentLoader.from_file(file_path)
    loader.resolve_env_vars()

    assert loader.resolved_schema is not None
    servers = loader.to_server_specs()
    assert servers["test"].url == "http://resolved.com/mcp"
    assert isinstance(servers["test"], HTTPServerSpec)
    assert servers["test"].headers["Authorization"] == "Bearer secret123"


def test_resolve_env_vars_missing_raises(tmp_path: Path) -> None:
    """Test that missing env vars raise error during resolution."""
    config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
servers:
  test:
    type: http
    url: "${MISSING_VAR}"
"""
    file_path = tmp_path / "test.superagent"
    file_path.write_text(config)

    loader = SuperAgentLoader.from_file(file_path)
    with pytest.raises(SuperAgentError, match="Missing required environment variables"):
        loader.resolve_env_vars()


def test_to_server_specs_http(tmp_path: Path) -> None:
    """Test conversion of HTTP server to ServerSpec."""
    config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
servers:
  http_server:
    type: http
    url: "http://127.0.0.1:8000/mcp"
    transport: sse
    headers:
      X-Custom: "value"
    api_key: "secret"
    bearer_token: "tok"
    audience: "api://tools"
"""
    file_path = tmp_path / "test.superagent"
    file_path.write_text(config)

    loader = SuperAgentLoader.from_file(file_path)
    specs = loader.to_server_specs()

    assert "http_server" in specs
    spec = specs["http_server"]
    assert isinstance(spec, HTTPServerSpec)
    assert spec.url == "http://127.0.0.1:8000/mcp"
    assert spec.transport == "sse"
    assert spec.headers == {"X-Custom": "value"}
    assert spec.api_key is not None and spec.api_key.get_secret_value() == "secret"
    assert spec.bearer_token is not None and spec.bearer_token.get_secret_value() == "tok"
    assert spec.audience == "api://tools"


def test_to_server_specs_stdio(tmp_path: Path) -> None:
    """Test conversion of stdio server to ServerSpec."""
    config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
servers:
  local:
    type: stdio
    command: python
    args: ["-m", "myserver"]
    env:
      DEBUG: "true"
    cwd: /tmp
    keep_alive: false
"""
    file_path = tmp_path / "test.superagent"
    file_path.write_text(config)

    loader = SuperAgentLoader.from_file(file_path)
    specs = loader.to_server_specs()

    assert "local" in specs
    spec = specs["local"]
    assert isinstance(spec, StdioServerSpec)
    assert spec.command == "python"
    assert spec.args == ["-m", "myserver"]
    assert spec.env == {"DEBUG": "true"}
    assert Path(spec.cwd) == Path("/tmp")  # kept as written, not joined to the file folder
    assert spec.keep_alive is False


def test_to_model_string_simple(tmp_path: Path) -> None:
    """Test model string conversion for simple config."""
    config = """version: "1.0"
agent:
  model: "anthropic:claude-opus-4.5"
servers:
  test:
    type: http
    url: "http://test"
"""
    file_path = tmp_path / "test.superagent"
    file_path.write_text(config)

    loader = SuperAgentLoader.from_file(file_path)
    model_str = loader.to_model_string()
    assert model_str == "anthropic:claude-opus-4.5"


def test_to_model_string_detailed(tmp_path: Path) -> None:
    """Test model string conversion for detailed config."""
    config = """version: "1.0"
agent:
  model:
    provider: openai
    name: gpt-4.1
    temperature: 0.7
servers:
  test:
    type: http
    url: "http://test"
"""
    file_path = tmp_path / "test.superagent"
    file_path.write_text(config)

    loader = SuperAgentLoader.from_file(file_path)
    model_str = loader.to_model_string()
    assert model_str == "openai:gpt-4.1"


def test_to_agent_config(tmp_path: Path) -> None:
    """Test conversion to SuperAgentConfig."""
    config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
  instructions: "Custom prompt"
  trace: false
servers:
  test:
    type: http
    url: "http://test"
"""
    file_path = tmp_path / "test.superagent"
    file_path.write_text(config)

    loader = SuperAgentLoader.from_file(file_path)
    agent_config = loader.to_agent_config()

    assert agent_config.model == "openai:gpt-4.1"
    assert agent_config.instructions == "Custom prompt"
    assert agent_config.trace is False
    assert "test" in agent_config.servers


def test_agent_config_to_build_kwargs() -> None:
    """Test SuperAgentConfig conversion to build kwargs."""
    from promptise.superagent import SuperAgentConfig

    config = SuperAgentConfig(
        model="openai:gpt-4.1",
        servers={},
        instructions="Test prompt",
        trace=True,
    )

    kwargs = config.to_build_kwargs()
    assert kwargs["model"] == "openai:gpt-4.1"
    assert kwargs["servers"] == {}
    assert kwargs["instructions"] == "Test prompt"
    assert kwargs["trace_tools"] is True


def test_cross_agent_resolution_simple(tmp_path: Path) -> None:
    """Test loading a cross-agent reference."""
    # Create peer agent file
    peer_config = """version: "1.0"
agent:
  model: "openai:gpt-5-mini"
servers:
  math:
    type: http
    url: "http://math.example.com"
"""
    peer_path = tmp_path / "peer.superagent"
    peer_path.write_text(peer_config)

    # Create main agent file
    main_config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
servers:
  general:
    type: http
    url: "http://general.example.com"
cross_agents:
  mathpeer:
    file: peer.superagent
    description: "Math specialist"
"""
    main_path = tmp_path / "main.superagent"
    main_path.write_text(main_config)

    loader = SuperAgentLoader.from_file(main_path)
    cross_loaders = loader.resolve_cross_agents(recursive=False)

    assert "mathpeer" in cross_loaders
    assert cross_loaders["mathpeer"].schema.agent.model == "openai:gpt-5-mini"


def test_cross_agent_circular_reference(tmp_path: Path) -> None:
    """Test circular reference detection."""
    # Create circular references
    main_config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
servers:
  test:
    type: http
    url: "http://test"
cross_agents:
  peer:
    file: peer.superagent
    description: "Peer"
"""
    peer_config = """version: "1.0"
agent:
  model: "openai:gpt-5-mini"
servers:
  test:
    type: http
    url: "http://test"
cross_agents:
  main:
    file: main.superagent
    description: "Main"
"""
    main_path = tmp_path / "main.superagent"
    peer_path = tmp_path / "peer.superagent"
    main_path.write_text(main_config)
    peer_path.write_text(peer_config)

    loader = SuperAgentLoader.from_file(main_path)
    with pytest.raises(SuperAgentError, match="Circular cross-agent reference"):
        loader.resolve_cross_agents()


def test_cross_agent_file_not_found(tmp_path: Path) -> None:
    """Test error when cross-agent file doesn't exist."""
    main_config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
servers:
  test:
    type: http
    url: "http://test"
cross_agents:
  missing:
    file: nonexistent.superagent
    description: "Missing"
"""
    main_path = tmp_path / "main.superagent"
    main_path.write_text(main_config)

    loader = SuperAgentLoader.from_file(main_path)
    with pytest.raises(SuperAgentError, match="Failed to load cross-agent"):
        loader.resolve_cross_agents()


def test_cross_agent_relative_path_resolution(tmp_path: Path) -> None:
    """Test that cross-agent paths are resolved relative to parent file."""
    # Create subdirectory
    agents_dir = tmp_path / "agents"
    agents_dir.mkdir()

    # Create peer in subdirectory
    peer_config = """version: "1.0"
agent:
  model: "openai:gpt-5-mini"
servers:
  test:
    type: http
    url: "http://test"
"""
    peer_path = agents_dir / "peer.superagent"
    peer_path.write_text(peer_config)

    # Create main in parent directory
    main_config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
servers:
  test:
    type: http
    url: "http://test"
cross_agents:
  peer:
    file: agents/peer.superagent
    description: "Peer"
"""
    main_path = tmp_path / "main.superagent"
    main_path.write_text(main_config)

    loader = SuperAgentLoader.from_file(main_path)
    cross_loaders = loader.resolve_cross_agents(recursive=False)

    assert "peer" in cross_loaders


def test_load_superagent_file_convenience(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test convenience function load_superagent_file."""
    monkeypatch.setenv("API_KEY", "secret")

    config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
servers:
  test:
    type: http
    url: "http://test"
    api_key: "${API_KEY}"
"""
    file_path = tmp_path / "test.superagent"
    file_path.write_text(config)

    main, cross_agents = load_superagent_file(file_path, resolve_refs=False)

    assert main.resolved_schema is not None
    assert len(cross_agents) == 0
    specs = main.to_server_specs()
    assert specs["test"].api_key.get_secret_value() == "secret"


def test_load_superagent_file_with_cross_agents(tmp_path: Path) -> None:
    """Test convenience function with cross-agent resolution."""
    peer_config = """version: "1.0"
agent:
  model: "openai:gpt-5-mini"
servers:
  test:
    type: http
    url: "http://test"
"""
    peer_path = tmp_path / "peer.superagent"
    peer_path.write_text(peer_config)

    main_config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
servers:
  test:
    type: http
    url: "http://test"
cross_agents:
  peer:
    file: peer.superagent
    description: "Peer"
"""
    main_path = tmp_path / "main.superagent"
    main_path.write_text(main_config)

    main, cross_agents = load_superagent_file(main_path, resolve_refs=True)

    assert main.resolved_schema is not None
    assert len(cross_agents) == 1
    assert "peer" in cross_agents


def test_resolve_env_vars_chainable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that resolve_env_vars returns self for chaining."""
    monkeypatch.setenv("API_KEY", "secret")

    config = """version: "1.0"
agent:
  model: "openai:gpt-4.1"
servers:
  test:
    type: http
    url: "http://test"
    api_key: "${API_KEY}"
"""
    file_path = tmp_path / "test.superagent"
    file_path.write_text(config)

    loader = SuperAgentLoader.from_file(file_path)
    result = loader.resolve_env_vars()

    assert result is loader
    assert loader.resolved_schema is not None


# ---------------------------------------------------------------------------
# Agent identity section
# ---------------------------------------------------------------------------


def _agent_with_identity(tmp_path: Path, identity_block: str) -> SuperAgentLoader:
    body = (
        'version: "1.0"\n'
        "agent:\n"
        '  model: "openai:gpt-5-mini"\n'
        f"{identity_block}"
        "servers:\n"
        "  tools:\n"
        "    type: http\n"
        '    url: "https://mcp.internal"\n'
    )
    path = tmp_path / "agent.superagent"
    path.write_text(body)
    return SuperAgentLoader.from_file(path)


def test_identity_local_is_built_and_passed_to_build_kwargs(tmp_path: Path) -> None:
    loader = _agent_with_identity(
        tmp_path,
        "identity:\n"
        "  provider: local\n"
        "  agent_id: billing-bot\n"
        "  name: Billing Bot\n"
        "  owner: payments\n"
        "  labels: {env: prod}\n",
    )
    ident = loader.to_identity()
    assert ident is not None
    assert ident.agent_id == "billing-bot"
    assert ident.name == "Billing Bot"
    assert ident.owner == "payments"
    assert ident.labels == {"env": "prod"}
    assert ident.is_verifiable is False
    # The identity flows all the way into build_agent(**kwargs).
    kwargs = loader.to_agent_config().to_build_kwargs()
    assert kwargs["identity"].agent_id == "billing-bot"


def test_no_identity_section_yields_none(tmp_path: Path) -> None:
    loader = _agent_with_identity(tmp_path, "")
    assert loader.to_identity() is None
    assert "identity" not in loader.to_agent_config().to_build_kwargs()


def test_identity_oidc_resolves_env_and_is_verifiable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_ISSUER", "https://gitlab.com")
    monkeypatch.setenv("MY_JWT_VAR", "CI_JOB_JWT_V2")
    loader = _agent_with_identity(
        tmp_path,
        "identity:\n"
        "  provider: oidc\n"
        "  agent_id: release-bot\n"
        "  issuer: ${MY_ISSUER}\n"
        "  token_env_var: ${MY_JWT_VAR}\n",
    ).resolve_env_vars()
    ident = loader.to_identity()
    assert ident is not None
    assert ident.is_verifiable is True
    assert ident.credential_provider == "oidc:https://gitlab.com"


def test_identity_entra_imds_selects_managed_identity_provider(tmp_path: Path) -> None:
    loader = _agent_with_identity(
        tmp_path,
        "identity:\n"
        "  provider: entra\n"
        "  agent_id: data-bot\n"
        "  mode: imds\n"
        "  client_id: abc-123\n"
        "  resource: api://internal\n",
    )
    ident = loader.to_identity()
    assert ident is not None
    assert ident.credential_provider == "entra-imds"


def test_identity_gcp_selects_metadata_provider(tmp_path: Path) -> None:
    loader = _agent_with_identity(
        tmp_path,
        "identity:\n  provider: gcp\n  agent_id: g-bot\n  audience: api://m\n",
    )
    ident = loader.to_identity()
    assert ident is not None
    assert ident.credential_provider == "gcp-metadata"


def test_identity_aws_sts_selects_sts_provider(tmp_path: Path) -> None:
    loader = _agent_with_identity(
        tmp_path,
        "identity:\n  provider: aws\n  agent_id: a-bot\n  mode: sts\n"
        "  region: us-east-1\n  audience: api://m\n",
    )
    ident = loader.to_identity()
    assert ident is not None
    assert ident.credential_provider == "aws-sts"


def test_identity_spiffe_sdk_selects_sdk_provider(tmp_path: Path) -> None:
    loader = _agent_with_identity(
        tmp_path,
        "identity:\n  provider: spiffe\n  agent_id: s-bot\n  mode: sdk\n  audience: api://m\n",
    )
    ident = loader.to_identity()
    assert ident is not None
    assert ident.credential_provider == "spiffe-sdk"


def test_identity_auto_uses_detected_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "my-project")
    loader = _agent_with_identity(tmp_path, "identity:\n  provider: auto\n  agent_id: auto-bot\n")
    ident = loader.to_identity()
    assert ident is not None
    assert ident.credential_provider == "gcp-metadata"


def test_identity_local_without_agent_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SuperAgentValidationError, match="requires 'agent_id'"):
        _agent_with_identity(tmp_path, "identity:\n  provider: local\n")


def test_identity_oidc_without_issuer_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SuperAgentValidationError, match="requires 'issuer'"):
        _agent_with_identity(
            tmp_path,
            "identity:\n  provider: oidc\n  agent_id: b\n  token_env_var: X\n",
        )


def test_identity_oidc_without_token_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SuperAgentValidationError, match="exactly one of"):
        _agent_with_identity(
            tmp_path,
            "identity:\n  provider: oidc\n  agent_id: b\n  issuer: https://idp\n",
        )


def test_identity_unknown_field_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(SuperAgentValidationError):
        _agent_with_identity(
            tmp_path,
            "identity:\n  provider: local\n  agent_id: b\n  bogus: x\n",
        )
