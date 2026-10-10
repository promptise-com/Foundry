"""Tests for .superagent schema validation."""

from __future__ import annotations

import warnings

import pytest
from pydantic import ValidationError

from promptise.superagent_schema import (
    AgentSection,
    CrossAgentConfig,
    DetailedModelConfig,
    HTTPServerConfig,
    StdioServerConfig,
    SuperAgentSchema,
)


def test_simple_model_config() -> None:
    """Test simple string model configuration."""
    schema = SuperAgentSchema(
        version="1.0",
        agent=AgentSection(model="openai:gpt-4.1"),
        servers={"test": HTTPServerConfig(type="http", url="http://test")},
    )
    assert isinstance(schema.agent.model, str)
    assert schema.agent.model == "openai:gpt-4.1"


def test_detailed_model_config() -> None:
    """Test detailed model configuration with parameters."""
    schema = SuperAgentSchema(
        version="1.0",
        agent=AgentSection(
            model=DetailedModelConfig(
                provider="openai",
                name="gpt-4.1",
                temperature=0.7,
                api_key="${API_KEY}",
            )
        ),
        servers={"test": HTTPServerConfig(type="http", url="http://test")},
    )
    assert isinstance(schema.agent.model, DetailedModelConfig)
    assert schema.agent.model.temperature == 0.7
    assert schema.agent.model.provider == "openai"
    assert schema.agent.model.name == "gpt-4.1"


def test_detailed_model_config_all_fields() -> None:
    """Test detailed model config with all optional fields."""
    config = DetailedModelConfig(
        provider="anthropic",
        name="claude-opus-4.5",
        api_key="${ANTHROPIC_API_KEY}",
        temperature=0.5,
        max_tokens=4096,
        timeout=120,
        base_url="https://api.custom.com",
        extra={"custom_param": "value"},
    )
    assert config.provider == "anthropic"
    assert config.temperature == 0.5
    assert config.max_tokens == 4096
    assert config.timeout == 120
    assert config.base_url == "https://api.custom.com"
    assert config.extra == {"custom_param": "value"}


def test_detailed_model_config_temperature_validation() -> None:
    """Test temperature range validation."""
    # Valid range
    DetailedModelConfig(provider="openai", name="gpt-4", temperature=0.0)
    DetailedModelConfig(provider="openai", name="gpt-4", temperature=2.0)
    DetailedModelConfig(provider="openai", name="gpt-4", temperature=1.0)

    # Out of range
    with pytest.raises(ValidationError):
        DetailedModelConfig(provider="openai", name="gpt-4", temperature=-0.1)

    with pytest.raises(ValidationError):
        DetailedModelConfig(provider="openai", name="gpt-4", temperature=2.1)


def test_detailed_model_config_max_tokens_validation() -> None:
    """Test max_tokens must be positive."""
    DetailedModelConfig(provider="openai", name="gpt-4", max_tokens=100)

    with pytest.raises(ValidationError):
        DetailedModelConfig(provider="openai", name="gpt-4", max_tokens=0)

    with pytest.raises(ValidationError):
        DetailedModelConfig(provider="openai", name="gpt-4", max_tokens=-1)


def test_detailed_model_config_timeout_validation() -> None:
    """Test timeout must be positive."""
    DetailedModelConfig(provider="openai", name="gpt-4", timeout=60)

    with pytest.raises(ValidationError):
        DetailedModelConfig(provider="openai", name="gpt-4", timeout=0)

    with pytest.raises(ValidationError):
        DetailedModelConfig(provider="openai", name="gpt-4", timeout=-1)


def test_api_key_warning_for_direct_value() -> None:
    """Test warning for direct API key values."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        DetailedModelConfig(
            provider="openai",
            name="gpt-4",
            api_key="sk-proj-very-long-key-string-here",
        )
        assert len(w) == 1
        assert "Direct API key detected" in str(w[0].message)


def test_no_warning_for_env_var_api_key() -> None:
    """Test no warning for environment variable reference."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        DetailedModelConfig(
            provider="openai",
            name="gpt-4",
            api_key="${OPENAI_API_KEY}",
        )
        assert len(w) == 0


def test_no_warning_for_none_api_key() -> None:
    """Test no warning when API key is None."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        DetailedModelConfig(provider="openai", name="gpt-4", api_key=None)
        assert len(w) == 0


def test_http_server_config() -> None:
    """Test HTTP server configuration."""
    server = HTTPServerConfig(
        type="http",
        url="http://127.0.0.1:8000/mcp",
        transport="http",
    )
    assert server.type == "http"
    assert server.url == "http://127.0.0.1:8000/mcp"
    assert server.transport == "http"
    assert server.headers == {}
    assert server.bearer_token is None
    assert server.api_key is None


def test_http_server_config_with_headers() -> None:
    """Test HTTP server with custom headers."""
    server = HTTPServerConfig(
        type="http",
        url="http://api.example.com/mcp",
        headers={
            "Authorization": "Bearer ${TOKEN}",
            "X-Custom": "value",
        },
    )
    assert server.headers["Authorization"] == "Bearer ${TOKEN}"
    assert server.headers["X-Custom"] == "value"


def test_http_server_config_all_transports() -> None:
    """Test all valid transport types."""
    HTTPServerConfig(type="http", url="http://test", transport="http")
    HTTPServerConfig(type="http", url="http://test", transport="streamable-http")
    HTTPServerConfig(type="http", url="http://test", transport="sse")


def test_http_server_config_invalid_transport() -> None:
    """Test invalid transport type is rejected."""
    with pytest.raises(ValidationError):
        HTTPServerConfig(type="http", url="http://test", transport="invalid")  # type: ignore


def test_stdio_server_config() -> None:
    """Test stdio server configuration."""
    server = StdioServerConfig(
        type="stdio",
        command="python",
        args=["-m", "mypackage.server"],
    )
    assert server.type == "stdio"
    assert server.command == "python"
    assert server.args == ["-m", "mypackage.server"]
    assert server.env == {}
    assert server.cwd is None
    assert server.keep_alive is True


def test_stdio_server_config_all_fields() -> None:
    """Test stdio server with all optional fields."""
    server = StdioServerConfig(
        type="stdio",
        command="python",
        args=["-m", "server", "--port", "3000"],
        env={"API_KEY": "${MY_KEY}", "DEBUG": "true"},
        cwd="/tmp",
        keep_alive=False,
    )
    assert server.env == {"API_KEY": "${MY_KEY}", "DEBUG": "true"}
    assert server.cwd == "/tmp"
    assert server.keep_alive is False


def test_server_discriminated_union_http() -> None:
    """Test server type discrimination for HTTP."""
    schema = SuperAgentSchema(
        version="1.0",
        agent=AgentSection(model="test:model"),
        servers={"http": HTTPServerConfig(type="http", url="http://test")},
    )
    assert isinstance(schema.servers["http"], HTTPServerConfig)


def test_server_discriminated_union_stdio() -> None:
    """Test server type discrimination for stdio."""
    schema = SuperAgentSchema(
        version="1.0",
        agent=AgentSection(model="test:model"),
        servers={"stdio": StdioServerConfig(type="stdio", command="python")},
    )
    assert isinstance(schema.servers["stdio"], StdioServerConfig)


def test_cross_agent_config() -> None:
    """Test cross-agent configuration."""
    config = CrossAgentConfig(
        file="./agents/peer.superagent",
        description="Peer agent description",
    )
    assert config.file == "./agents/peer.superagent"
    assert config.description == "Peer agent description"


def test_cross_agent_config_empty_description() -> None:
    """Test cross-agent with empty description."""
    config = CrossAgentConfig(file="./peer.superagent")
    assert config.file == "./peer.superagent"
    assert config.description == ""


def test_agent_section_simple() -> None:
    """Test agent section with simple model."""
    agent = AgentSection(model="openai:gpt-4.1")
    assert agent.model == "openai:gpt-4.1"
    assert agent.instructions is None
    assert agent.trace is True


def test_agent_section_all_fields() -> None:
    """Test agent section with all fields."""
    agent = AgentSection(
        model="openai:gpt-4.1",
        instructions="Custom prompt",
        trace=False,
    )
    assert agent.instructions == "Custom prompt"
    assert agent.trace is False


def test_superagent_schema_minimal() -> None:
    """Test minimal valid schema."""
    schema = SuperAgentSchema(
        version="1.0",
        agent=AgentSection(model="openai:gpt-4.1"),
        servers={"test": HTTPServerConfig(type="http", url="http://test")},
    )
    assert schema.version == "1.0"
    assert isinstance(schema.agent, AgentSection)
    assert "test" in schema.servers
    assert schema.cross_agents is None


def test_superagent_schema_with_cross_agents() -> None:
    """Test schema with cross-agent references."""
    schema = SuperAgentSchema(
        version="1.0",
        agent=AgentSection(model="openai:gpt-4.1"),
        servers={"test": HTTPServerConfig(type="http", url="http://test")},
        cross_agents={
            "peer1": CrossAgentConfig(file="./peer1.superagent", description="Peer 1"),
            "peer2": CrossAgentConfig(file="./peer2.superagent", description="Peer 2"),
        },
    )
    assert len(schema.cross_agents) == 2  # type: ignore
    assert "peer1" in schema.cross_agents  # type: ignore
    assert "peer2" in schema.cross_agents  # type: ignore


def test_superagent_schema_only_cross_agents() -> None:
    """Test schema with only cross-agents (no servers)."""
    schema = SuperAgentSchema(
        version="1.0",
        agent=AgentSection(model="openai:gpt-4.1"),
        cross_agents={
            "peer": CrossAgentConfig(file="./peer.superagent"),
        },
    )
    assert len(schema.servers) == 0
    assert len(schema.cross_agents) == 1  # type: ignore


def test_superagent_schema_must_have_servers_or_cross_agents() -> None:
    """Test that at least servers or cross_agents is required."""
    with pytest.raises(ValidationError) as exc_info:
        SuperAgentSchema(
            version="1.0",
            agent=AgentSection(model="test:model"),
            servers={},
            cross_agents=None,
        )
    assert "At least one of 'servers', 'cross_agents', or 'sandbox'" in str(exc_info.value)


def test_superagent_schema_version_must_be_1_0() -> None:
    """Test that version must be exactly '1.0'."""
    with pytest.raises(ValidationError):
        SuperAgentSchema(
            version="2.0",  # type: ignore
            agent=AgentSection(model="test:model"),
            servers={"test": HTTPServerConfig(type="http", url="http://test")},
        )


def test_extra_fields_forbidden_detailed_model() -> None:
    """Test that extra fields are rejected in DetailedModelConfig."""
    with pytest.raises(ValidationError):
        DetailedModelConfig(
            provider="openai",
            name="gpt-4",
            unknown_field="should_fail",  # type: ignore
        )


def test_extra_fields_forbidden_http_server() -> None:
    """Test that extra fields are rejected in HTTPServerConfig."""
    with pytest.raises(ValidationError):
        HTTPServerConfig(
            type="http",
            url="http://test",
            unknown_field="should_fail",  # type: ignore
        )


def test_extra_fields_forbidden_stdio_server() -> None:
    """Test that extra fields are rejected in StdioServerConfig."""
    with pytest.raises(ValidationError):
        StdioServerConfig(
            type="stdio",
            command="python",
            unknown_field="should_fail",  # type: ignore
        )


def test_extra_fields_forbidden_cross_agent() -> None:
    """Test that extra fields are rejected in CrossAgentConfig."""
    with pytest.raises(ValidationError):
        CrossAgentConfig(
            file="./peer.superagent",
            unknown_field="should_fail",  # type: ignore
        )


def test_extra_fields_forbidden_agent_section() -> None:
    """Test that extra fields are rejected in AgentSection."""
    with pytest.raises(ValidationError):
        AgentSection(
            model="test:model",
            unknown_field="should_fail",  # type: ignore
        )


def test_extra_fields_forbidden_superagent_schema() -> None:
    """Test that extra fields are rejected in SuperAgentSchema."""
    with pytest.raises(ValidationError):
        SuperAgentSchema(
            version="1.0",
            agent=AgentSection(model="test:model"),
            servers={"test": HTTPServerConfig(type="http", url="http://test")},
            unknown_field="should_fail",  # type: ignore
        )


def test_required_fields_model_config() -> None:
    """Test required fields in DetailedModelConfig."""
    # Missing provider
    with pytest.raises(ValidationError):
        DetailedModelConfig(name="gpt-4")  # type: ignore

    # Missing name
    with pytest.raises(ValidationError):
        DetailedModelConfig(provider="openai")  # type: ignore


def test_required_fields_http_server() -> None:
    """Test required fields in HTTPServerConfig."""
    # Missing url
    with pytest.raises(ValidationError):
        HTTPServerConfig(type="http")  # type: ignore


def test_required_fields_stdio_server() -> None:
    """Test required fields in StdioServerConfig."""
    # Missing command
    with pytest.raises(ValidationError):
        StdioServerConfig(type="stdio")  # type: ignore


def test_required_fields_cross_agent() -> None:
    """Test required fields in CrossAgentConfig."""
    # Missing file
    with pytest.raises(ValidationError):
        CrossAgentConfig(description="test")  # type: ignore


def test_required_fields_agent_section() -> None:
    """Test required fields in AgentSection."""
    # Missing model
    with pytest.raises(ValidationError):
        AgentSection()  # type: ignore


def test_required_fields_superagent_schema() -> None:
    """Test required fields in SuperAgentSchema."""
    # Missing agent
    with pytest.raises(ValidationError):
        SuperAgentSchema(  # type: ignore
            version="1.0",
            servers={"test": HTTPServerConfig(type="http", url="http://test")},
        )


def test_complex_schema_integration() -> None:
    """Test a complex schema with all features."""
    schema = SuperAgentSchema(
        version="1.0",
        agent=AgentSection(
            model=DetailedModelConfig(
                provider="openai",
                name="gpt-4.1",
                temperature=0.7,
                max_tokens=4096,
                api_key="${OPENAI_API_KEY}",
            ),
            instructions="You are a helpful agent with access to tools.",
            trace=True,
        ),
        servers={
            "http_server": HTTPServerConfig(
                type="http",
                url="https://api.example.com/mcp",
                transport="http",
                headers={"Authorization": "Bearer ${API_TOKEN}"},
                bearer_token="${AUTH_SECRET}",
            ),
            "local_server": StdioServerConfig(
                type="stdio",
                command="python",
                args=["-m", "myserver", "--port", "3000"],
                env={"DEBUG": "true", "API_KEY": "${MY_KEY}"},
                cwd="/tmp",
                keep_alive=True,
            ),
        },
        cross_agents={
            "specialist": CrossAgentConfig(
                file="./agents/specialist.superagent",
                description="Domain specialist agent",
            ),
        },
    )

    # Verify structure
    assert isinstance(schema.agent.model, DetailedModelConfig)
    assert len(schema.servers) == 2
    assert len(schema.cross_agents) == 1  # type: ignore
    assert isinstance(schema.servers["http_server"], HTTPServerConfig)
    assert isinstance(schema.servers["local_server"], StdioServerConfig)
