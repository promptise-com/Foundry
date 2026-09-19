from __future__ import annotations

from promptise.cli import _merge_servers, _parse_kv


def test_parse_kv_simple() -> None:
    assert _parse_kv(["a=1", "b = two"]) == {"a": "1", "b": "two"}


def test_merge_servers_http_and_stdio() -> None:
    stdios = [
        "name=echo command=python args='-m mypkg.server --port 3333' env.API_KEY=xyz keep_alive=false",
    ]
    https = [
        "name=remote url=https://example.com/mcp transport=http header.Authorization='Bearer abc'",
    ]
    servers = _merge_servers(stdios, https)
    assert set(servers.keys()) == {"echo", "remote"}
    # HTTP server spec
    http = servers["remote"]
    assert http.url == "https://example.com/mcp"
    assert http.headers["Authorization"] == "Bearer abc"
    # stdio server spec
    stdio = servers["echo"]
    assert stdio.command == "python"
    assert stdio.args[0] == "-m"
    assert stdio.keep_alive is False


from unittest.mock import AsyncMock, MagicMock  # noqa: E402


class TestReplShutdown:
    """The REPL commands must close MCP sessions in the task that opened them —
    otherwise interpreter exit tears them down from another task and anyio
    raises "Attempted to exit cancel scope in a different task"."""

    @staticmethod
    def _fake_agent():
        from unittest.mock import AsyncMock, MagicMock

        agent = MagicMock()
        agent.ainvoke = AsyncMock(return_value={"messages": []})
        agent.shutdown = AsyncMock()
        return agent

    def test_run_shuts_the_agent_down_on_exit(self, monkeypatch):
        from typer.testing import CliRunner

        from promptise import cli

        agent = self._fake_agent()
        monkeypatch.setattr(cli, "build_agent", AsyncMock(return_value=agent))
        result = CliRunner().invoke(
            cli.app, ["run", "--model-id", "openai:gpt-5-mini"], input="exit\n"
        )
        assert result.exit_code == 0, result.output
        agent.shutdown.assert_awaited_once()

    def test_run_shuts_down_even_when_a_turn_fails(self, monkeypatch):
        from typer.testing import CliRunner

        from promptise import cli

        agent = self._fake_agent()
        agent.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
        monkeypatch.setattr(cli, "build_agent", AsyncMock(return_value=agent))
        result = CliRunner().invoke(
            cli.app, ["run", "--model-id", "openai:gpt-5-mini"], input="hi\nexit\n"
        )
        assert result.exit_code == 0 and "boom" in result.output
        agent.shutdown.assert_awaited_once()

    def test_list_tools_uses_agent_tools_and_shuts_down(self, monkeypatch):
        from typer.testing import CliRunner

        from promptise import cli

        agent = self._fake_agent()
        tool = MagicMock()
        tool.name, tool.description, tool.args_schema = (
            "find_books",
            "Find books.",
            {"type": "object"},
        )
        agent.tools = [tool]
        monkeypatch.setattr(cli, "build_agent", AsyncMock(return_value=agent))
        result = CliRunner().invoke(cli.app, ["list-tools", "--model-id", "openai:gpt-5-mini"])
        assert result.exit_code == 0, result.output
        assert "find_books" in result.output
        agent.shutdown.assert_awaited_once()
