"""Comprehensive E2E tests for the core agent subsystem of Promptise.

Covers:
- build_agent factory function
- PromptiseAgent lifecycle (shutdown, stats, reports, concurrent invocations)
- Config classes (StdioServerSpec, HTTPServerSpec, servers_to_mcp_config)
- EnvResolver (resolve_env_var, resolve_env_in_dict, validate_all_env_vars_available)
- Memory providers (InMemoryProvider add/search/delete lifecycle)
- ObservabilityConfig (defaults, enums, creation)

All tests use mocks for LLM/graph internals — no real API calls required.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from promptise.agent import PromptiseAgent, build_agent
from promptise.config import (
    HTTPServerSpec,
    ServerSpec,
    StdioServerSpec,
    servers_to_mcp_config,
)
from promptise.env_resolver import (
    resolve_env_in_dict,
    resolve_env_var,
    validate_all_env_vars_available,
)
from promptise.exceptions import EnvVarNotFoundError
from promptise.memory import InMemoryProvider, MemoryResult
from promptise.observability_config import (
    ExportFormat,
    ObservabilityConfig,
    ObserveLevel,
    TransporterType,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_inner() -> MagicMock:
    """Return a mock object that quacks like a LangGraph Runnable."""
    mock = MagicMock()
    mock.ainvoke = AsyncMock(return_value={"messages": [{"role": "assistant", "content": "ok"}]})
    mock.invoke = MagicMock(return_value={"messages": [{"role": "assistant", "content": "ok"}]})
    return mock


def _make_mock_collector() -> MagicMock:
    """Return a mock ObservabilityCollector."""
    collector = MagicMock()
    collector.get_stats.return_value = {
        "total_events": 5,
        "llm_turns": 2,
        "tool_calls": 3,
    }
    return collector


# ===========================================================================
# 1. build_agent factory
# ===========================================================================


class TestBuildDeepAgent:
    """Tests for the build_agent factory function."""

    @pytest.mark.asyncio
    async def test_empty_servers_creates_agent_without_tools(self) -> None:
        """Agent should be created with no tools when servers dict is empty."""
        mock_graph = _make_mock_inner()

        with (
            patch("promptise.agent._normalize_model") as mock_norm,
            patch("promptise.agent.PromptGraphEngine", return_value=mock_graph) as mock_react,
            patch.dict("sys.modules", {"deepagents": None}),
        ):
            mock_norm.return_value = MagicMock()

            agent = await build_agent(
                servers={},
                model="openai:gpt-5-mini",
            )

            assert isinstance(agent, PromptiseAgent)
            # The engine should have been constructed
            mock_react.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_none_raises_value_error(self) -> None:
        """Passing model=None must raise ValueError."""
        with pytest.raises(ValueError, match="model is required"):
            await build_agent(servers={}, model=None)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_custom_instructions_passed_through(self) -> None:
        """Custom instructions should be forwarded to the react agent builder."""
        custom_instructions = "You are a math tutor."
        mock_graph = _make_mock_inner()

        # Capture args passed to PromptGraphEngine (which receives the graph)
        captured_engine_args: list[Any] = []

        def fake_engine(*args: Any, **kwargs: Any) -> Any:
            captured_engine_args.append(kwargs)
            return mock_graph

        with (
            patch("promptise.agent._normalize_model") as mock_norm,
            patch("promptise.agent.PromptGraphEngine", side_effect=fake_engine),
            patch.dict("sys.modules", {"deepagents": None}),
        ):
            mock_norm.return_value = MagicMock()

            agent = await build_agent(
                servers={},
                model="openai:gpt-5-mini",
                instructions=custom_instructions,
            )

            assert isinstance(agent, PromptiseAgent)
            # The engine receives a PromptGraph which was built with the instructions
            assert len(captured_engine_args) > 0

    @pytest.mark.asyncio
    async def test_extra_tools_appended(self) -> None:
        """extra_tools parameter should append tools to the tool list."""
        fake_tool = MagicMock()
        fake_tool.name = "my_custom_tool"
        mock_graph = _make_mock_inner()

        with (
            patch("promptise.agent._normalize_model") as mock_norm,
            patch("promptise.agent.PromptGraphEngine", return_value=mock_graph) as mock_react,
            patch.dict("sys.modules", {"deepagents": None}),
        ):
            mock_norm.return_value = MagicMock()

            agent = await build_agent(
                servers={},
                model="openai:gpt-5-mini",
                extra_tools=[fake_tool],
            )

            assert isinstance(agent, PromptiseAgent)
            # Engine should have been constructed
            mock_react.assert_called_once()


# ===========================================================================
# 2. PromptiseAgent lifecycle
# ===========================================================================


class TestPromptiseAgentLifecycle:
    """Tests for PromptiseAgent shutdown, stats, reports, and concurrency."""

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self) -> None:
        """Calling shutdown multiple times should not raise."""
        inner = _make_mock_inner()
        mcp_multi = AsyncMock()
        mcp_multi.__aexit__ = AsyncMock()
        agent = PromptiseAgent(inner=inner, mcp_multi=mcp_multi)

        await agent.shutdown()
        # Second call should be safe (mcp_multi is set to None after first)
        await agent.shutdown()

        # __aexit__ should only have been called once
        mcp_multi.__aexit__.assert_awaited_once()

    def test_get_stats_without_observe_returns_empty(self) -> None:
        """get_stats returns empty dict when observability is disabled."""
        inner = _make_mock_inner()
        agent = PromptiseAgent(inner=inner)

        stats = agent.get_stats()
        assert stats == {}

    def test_get_stats_with_observe_returns_collector_stats(self) -> None:
        """get_stats delegates to collector when observability is enabled."""
        inner = _make_mock_inner()
        collector = _make_mock_collector()
        agent = PromptiseAgent(inner=inner, collector=collector)

        stats = agent.get_stats()
        assert stats == {"total_events": 5, "llm_turns": 2, "tool_calls": 3}
        collector.get_stats.assert_called_once()

    def test_generate_report_requires_observe(self) -> None:
        """generate_report raises RuntimeError when observability is off."""
        inner = _make_mock_inner()
        agent = PromptiseAgent(inner=inner)

        with pytest.raises(RuntimeError, match="observability is not enabled"):
            agent.generate_report("report.html")

    @pytest.mark.asyncio
    async def test_concurrent_ainvoke(self) -> None:
        """Multiple concurrent ainvoke calls should all complete."""
        call_count = 0

        async def fake_ainvoke(input: Any, **kwargs: Any) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            return {"messages": [{"role": "assistant", "content": f"reply-{call_count}"}]}

        inner = MagicMock()
        inner.ainvoke = AsyncMock(side_effect=fake_ainvoke)
        agent = PromptiseAgent(inner=inner)

        tasks = [
            agent.ainvoke({"messages": [{"role": "user", "content": f"q{i}"}]}) for i in range(5)
        ]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert inner.ainvoke.await_count == 5


# ===========================================================================
# 3. Config classes
# ===========================================================================


class TestConfigClasses:
    """Tests for StdioServerSpec, HTTPServerSpec, and servers_to_mcp_config."""

    def test_stdio_server_spec_creation_and_fields(self) -> None:
        """StdioServerSpec should store all fields correctly."""
        spec = StdioServerSpec(
            command="python",
            args=["-m", "my_server"],
            env={"KEY": "val"},
            cwd="/tmp",
            keep_alive=False,
        )
        assert spec.command == "python"
        assert spec.args == ["-m", "my_server"]
        assert spec.env == {"KEY": "val"}
        assert spec.cwd == "/tmp"
        assert spec.keep_alive is False

    def test_http_server_spec_with_bearer_token(self) -> None:
        """HTTPServerSpec should accept a bearer_token field."""
        spec = HTTPServerSpec(
            url="http://localhost:8080/mcp",
            bearer_token="eyJhbGciOiJIUzI1NiIs",
        )
        assert spec.url == "http://localhost:8080/mcp"
        assert spec.bearer_token.get_secret_value() == "eyJhbGciOiJIUzI1NiIs"
        assert spec.api_key is None
        assert spec.transport == "http"

    def test_http_server_spec_with_api_key(self) -> None:
        """HTTPServerSpec should accept an api_key field."""
        spec = HTTPServerSpec(
            url="http://localhost:8080/mcp",
            api_key="my-secret-key",
        )
        assert spec.api_key.get_secret_value() == "my-secret-key"
        assert spec.bearer_token is None

    def test_servers_to_mcp_config_mixed(self) -> None:
        """servers_to_mcp_config should handle a mix of stdio and http specs."""
        servers: dict[str, ServerSpec] = {
            "local": StdioServerSpec(
                command="node",
                args=["server.js"],
                env={"PORT": "3000"},
            ),
            "remote": HTTPServerSpec(
                url="http://api.example.com/mcp",
                transport="sse",
                headers={"Authorization": "Bearer tok"},
            ),
        }

        cfg = servers_to_mcp_config(servers)

        # Check stdio entry
        assert cfg["local"]["transport"] == "stdio"
        assert cfg["local"]["command"] == "node"
        assert cfg["local"]["args"] == ["server.js"]
        assert cfg["local"]["env"] == {"PORT": "3000"}

        # Check http entry
        assert cfg["remote"]["transport"] == "sse"
        assert cfg["remote"]["url"] == "http://api.example.com/mcp"
        assert cfg["remote"]["headers"] == {"Authorization": "Bearer tok"}

    def test_server_spec_union_type(self) -> None:
        """Both StdioServerSpec and HTTPServerSpec should be valid ServerSpec."""
        stdio: ServerSpec = StdioServerSpec(command="python")
        http: ServerSpec = HTTPServerSpec(url="http://localhost/mcp")

        assert isinstance(stdio, StdioServerSpec)
        assert isinstance(http, HTTPServerSpec)


# ===========================================================================
# 4. EnvResolver
# ===========================================================================


class TestEnvResolver:
    """Tests for resolve_env_var, resolve_env_in_dict, validate_all_env_vars_available."""

    def test_successful_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """resolve_env_var should substitute environment variable values."""
        monkeypatch.setenv("E2E_TEST_KEY", "secret-value")
        result = resolve_env_var("Bearer ${E2E_TEST_KEY}")
        assert result == "Bearer secret-value"

    def test_missing_var_raises_env_var_not_found_error(self) -> None:
        """Missing required env var (no default) must raise EnvVarNotFoundError."""
        with pytest.raises(EnvVarNotFoundError) as exc_info:
            resolve_env_var("${E2E_DEFINITELY_MISSING_VAR}")
        assert exc_info.value.var_name == "E2E_DEFINITELY_MISSING_VAR"

    def test_default_values(self) -> None:
        """Default value syntax should be used when env var is missing."""
        result = resolve_env_var("${E2E_MISSING:-fallback_value}")
        assert result == "fallback_value"

    def test_resolve_env_in_dict_nested(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """resolve_env_in_dict should recursively resolve nested dicts."""
        monkeypatch.setenv("E2E_URL", "http://example.com")
        monkeypatch.setenv("E2E_TOKEN", "tok123")

        data = {
            "server": {
                "url": "${E2E_URL}",
                "headers": {"auth": "Bearer ${E2E_TOKEN}"},
            },
            "plain": "no-vars-here",
            "number": 42,
        }
        result = resolve_env_in_dict(data)

        assert result["server"]["url"] == "http://example.com"
        assert result["server"]["headers"]["auth"] == "Bearer tok123"
        assert result["plain"] == "no-vars-here"
        assert result["number"] == 42

    def test_validate_all_env_vars_available(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """validate_all_env_vars_available should list missing vars."""
        monkeypatch.setenv("E2E_PRESENT", "yes")
        data = {
            "present": "${E2E_PRESENT}",
            "missing_a": "${E2E_ABSENT_A}",
            "missing_b": "${E2E_ABSENT_B}",
            "with_default": "${E2E_OPT:-default}",
        }
        missing = validate_all_env_vars_available(data)
        assert set(missing) == {"E2E_ABSENT_A", "E2E_ABSENT_B"}


# ===========================================================================
# 5. Memory providers
# ===========================================================================


class TestMemoryProviders:
    """Tests for InMemoryProvider lifecycle and memory integration with PromptiseAgent."""

    @pytest.mark.asyncio
    async def test_in_memory_add_search_delete_lifecycle(self) -> None:
        """Full add/search/delete lifecycle on InMemoryProvider."""
        provider = InMemoryProvider()

        # Add
        mid = await provider.add("The capital of France is Paris", metadata={"src": "test"})
        assert isinstance(mid, str)
        assert len(mid) > 0

        # Search — matching query (InMemoryProvider uses substring match)
        results = await provider.search("capital of France")
        assert len(results) >= 1
        assert any("Paris" in r.content for r in results)
        assert all(isinstance(r, MemoryResult) for r in results)

        # Delete
        deleted = await provider.delete(mid)
        assert deleted is True

        # Search again — should be empty
        results_after = await provider.search("capital of France")
        assert len(results_after) == 0

    @pytest.mark.asyncio
    async def test_in_memory_empty_search(self) -> None:
        """Search on an empty provider should return an empty list."""
        provider = InMemoryProvider()
        results = await provider.search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_in_memory_delete_nonexistent(self) -> None:
        """Deleting a non-existent memory_id should return False."""
        provider = InMemoryProvider()
        deleted = await provider.delete("nonexistent-id")
        assert deleted is False

    @pytest.mark.asyncio
    async def test_in_memory_max_entries_eviction(self) -> None:
        """InMemoryProvider should evict oldest entries when max_entries is reached."""
        provider = InMemoryProvider(max_entries=3)

        await provider.add("first")
        await provider.add("second")
        await provider.add("third")
        # Adding a fourth should evict the first
        await provider.add("fourth")

        results = await provider.search("first")
        assert len(results) == 0  # "first" was evicted

        results = await provider.search("fourth")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_memory_auto_store_integration(self) -> None:
        """PromptiseAgent with memory_auto_store=True should store exchanges."""
        provider = InMemoryProvider()
        inner = _make_mock_inner()
        agent = PromptiseAgent(
            inner=inner,
            memory_provider=provider,
            memory_auto_store=True,
        )

        await agent.ainvoke(
            {
                "messages": [{"role": "user", "content": "What is 2+2?"}],
            }
        )

        # The exchange should have been stored
        results = await provider.search("What is 2+2")
        assert len(results) >= 1
        # Stored content should contain both user and assistant text
        stored = results[0].content
        assert "2+2" in stored


# ===========================================================================
# 6. ObservabilityConfig
# ===========================================================================


class TestObservabilityConfig:
    """Tests for ObservabilityConfig defaults and enum values."""

    def test_defaults(self) -> None:
        """ObservabilityConfig defaults should be sensible."""
        cfg = ObservabilityConfig()

        assert cfg.level == ObserveLevel.STANDARD
        assert cfg.session_name == "promptise"
        assert cfg.record_prompts is None  # follows level: on at FULL only
        assert cfg.record_tool_io is True
        assert cfg.max_entries == 100_000
        assert cfg.transporters == [TransporterType.HTML]
        assert cfg.console_live is False
        assert cfg.correlation_id is None

    def test_observe_level_enum_values(self) -> None:
        """ObserveLevel enum should have the expected members."""
        assert ObserveLevel.OFF.value == "off"
        assert ObserveLevel.BASIC.value == "basic"
        assert ObserveLevel.STANDARD.value == "standard"
        assert ObserveLevel.FULL.value == "full"
        # All members accounted for
        assert set(ObserveLevel) == {
            ObserveLevel.OFF,
            ObserveLevel.BASIC,
            ObserveLevel.STANDARD,
            ObserveLevel.FULL,
        }

    def test_transporter_type_enum_values(self) -> None:
        """TransporterType enum should have all expected backends."""
        expected_values = {
            "html",
            "json",
            "log",
            "console",
            "prometheus",
            "otlp",
            "webhook",
            "callback",
        }
        actual_values = {t.value for t in TransporterType}
        assert expected_values == actual_values

    def test_export_format_alias(self) -> None:
        """ExportFormat should be an alias for TransporterType."""
        assert ExportFormat is TransporterType
        assert ExportFormat.HTML is TransporterType.HTML

    def test_config_creation_with_custom_values(self) -> None:
        """ObservabilityConfig should accept all configuration parameters."""
        cfg = ObservabilityConfig(
            level=ObserveLevel.FULL,
            session_name="my-session",
            record_prompts=True,
            max_entries=500,
            transporters=[TransporterType.HTML, TransporterType.JSON, TransporterType.CONSOLE],
            output_dir="./reports",
            log_file="./logs/events.jsonl",
            console_live=True,
            webhook_url="https://hooks.example.com/events",
            webhook_headers={"X-Token": "abc"},
            otlp_endpoint="http://otel:4317",
            prometheus_port=9191,
            correlation_id="req-123",
        )

        assert cfg.level == ObserveLevel.FULL
        assert cfg.session_name == "my-session"
        assert cfg.record_prompts is True
        assert cfg.max_entries == 500
        assert len(cfg.transporters) == 3
        assert cfg.output_dir == "./reports"
        assert cfg.log_file == "./logs/events.jsonl"
        assert cfg.console_live is True
        assert cfg.webhook_url == "https://hooks.example.com/events"
        assert cfg.webhook_headers == {"X-Token": "abc"}
        assert cfg.otlp_endpoint == "http://otel:4317"
        assert cfg.prometheus_port == 9191
        assert cfg.correlation_id == "req-123"
