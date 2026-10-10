"""Tests for the ``promptise serve`` CLI command.

The serve command imports a ``module:attribute`` target, validates it is an
``MCPServer``, and dispatches to ``server.run()`` (or ``hot_reload`` with
``--reload``). These tests cover registration, target resolution errors,
type validation, and dispatch — without starting a real transport.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from promptise.cli import app
from promptise.mcp.server import MCPServer
from promptise.mcp.server._serve_cli import resolve_server

runner = CliRunner(env={"COLUMNS": "200"})  # Rich wraps at COLUMNS; keep phrases on one line


def _all_output(result) -> str:
    """stdout + stderr regardless of click version's capture mode."""
    out = result.output
    try:
        out += result.stderr
    except (ValueError, AttributeError):
        pass  # older click mixes stderr into .output
    return out


# ---------------------------------------------------------------------------
# resolve_server
# ---------------------------------------------------------------------------


class TestResolveServer:
    def test_missing_colon_raises(self):
        import pytest

        with pytest.raises(ValueError, match="module.path:attribute"):
            resolve_server("just_a_module")

    def test_unimportable_module_raises(self):
        import pytest

        with pytest.raises(ImportError, match="Cannot import"):
            resolve_server("no_such_module_xyz_123:server")

    def test_missing_attribute_raises(self):
        import pytest

        with pytest.raises(AttributeError, match="no attribute"):
            resolve_server("promptise.cli:no_such_attr_xyz")

    def test_resolves_existing_attribute(self):
        obj = resolve_server("promptise.cli:console")
        assert obj is not None


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


class TestServeCommand:
    def test_registered_and_help_renders(self):
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "module.path:attribute" in _all_output(result)

    def test_invalid_target_format_exits_1(self):
        result = runner.invoke(app, ["serve", "notarget"])
        assert result.exit_code == 1
        assert "module.path:attribute" in _all_output(result)

    def test_unimportable_module_exits_1(self):
        result = runner.invoke(app, ["serve", "no_such_module_xyz_123:server"])
        assert result.exit_code == 1
        assert "Cannot import" in _all_output(result)

    def test_non_server_attribute_exits_1(self):
        # promptise.cli:console imports fine but is a rich Console, not an MCPServer
        result = runner.invoke(app, ["serve", "promptise.cli:console"])
        assert result.exit_code == 1
        assert "not an MCPServer" in _all_output(result)

    def test_invalid_transport_rejected(self):
        result = runner.invoke(app, ["serve", "x:y", "--transport", "grpc"])
        assert result.exit_code != 0

    def test_dispatches_to_server_run(self, monkeypatch):
        server = MCPServer(name="serve-test")
        recorded: dict = {}
        monkeypatch.setattr(server, "run", lambda **kw: recorded.update(kw))
        monkeypatch.setattr("promptise.mcp.server._serve_cli.resolve_server", lambda target: server)

        result = runner.invoke(
            app,
            ["serve", "myapp:server", "-t", "http", "-p", "9090", "--host", "0.0.0.0"],
        )
        assert result.exit_code == 0, _all_output(result)
        assert recorded == {
            "transport": "http",
            "host": "0.0.0.0",
            "port": 9090,
            "dashboard": False,
        }

    def test_defaults_are_stdio_loopback_8080(self, monkeypatch):
        server = MCPServer(name="serve-test")
        recorded: dict = {}
        monkeypatch.setattr(server, "run", lambda **kw: recorded.update(kw))
        monkeypatch.setattr("promptise.mcp.server._serve_cli.resolve_server", lambda target: server)

        result = runner.invoke(app, ["serve", "myapp:server"])
        assert result.exit_code == 0, _all_output(result)
        assert recorded == {
            "transport": "stdio",
            "host": "127.0.0.1",
            "port": 8080,
            "dashboard": False,
        }

    def test_reload_dispatches_to_hot_reload(self, monkeypatch):
        server = MCPServer(name="serve-test")
        recorded: dict = {}

        def fake_hot_reload(srv, **kw):
            recorded["server"] = srv
            recorded.update(kw)

        monkeypatch.setattr("promptise.mcp.server._serve_cli.resolve_server", lambda target: server)
        monkeypatch.setattr("promptise.mcp.server._hot_reload.hot_reload", fake_hot_reload)

        result = runner.invoke(app, ["serve", "myapp:server", "--reload", "-t", "http"])
        assert result.exit_code == 0, _all_output(result)
        assert recorded["server"] is server
        assert recorded["transport"] == "http"

    def test_allow_lists_reach_run_serve(self, monkeypatch):
        """--allowed-host / --allowed-origin are repeatable and land on the
        namespace under the names build_serve_parser() uses."""
        server = MCPServer(name="serve-test")
        recorded: dict = {}
        monkeypatch.setattr("promptise.mcp.server._serve_cli.resolve_server", lambda target: server)
        monkeypatch.setattr(
            "promptise.mcp.server._serve_cli.run_serve",
            lambda args, server=None: recorded.update(args=args, server=server),
        )

        result = runner.invoke(
            app,
            [
                "serve",
                "myapp:server",
                "-t",
                "http",
                "--host",
                "0.0.0.0",
                "--allowed-host",
                "api.example.com",
                "--allowed-host",
                "api.example.com:*",
                "--allowed-origin",
                "https://app.example.com",
            ],
        )
        assert result.exit_code == 0, _all_output(result)
        assert recorded["server"] is server
        args = recorded["args"]
        assert args.allowed_hosts == ["api.example.com", "api.example.com:*"]
        assert args.allowed_origins == ["https://app.example.com"]
        assert (args.transport, args.host, args.port) == ("http", "0.0.0.0", 8080)

    def test_allow_lists_are_forwarded_to_server_run_only_when_given(self, monkeypatch):
        server = MCPServer(name="serve-test")
        recorded: dict = {}
        monkeypatch.setattr(server, "run", lambda **kw: recorded.update(kw))
        monkeypatch.setattr("promptise.mcp.server._serve_cli.resolve_server", lambda target: server)

        result = runner.invoke(
            app, ["serve", "myapp:server", "-t", "http", "--allowed-host", "api.example.com"]
        )
        assert result.exit_code == 0, _all_output(result)
        assert recorded["allowed_hosts"] == ["api.example.com"]
        assert "allowed_origins" not in recorded  # not given: the bind-derived default policy

        recorded.clear()
        assert runner.invoke(app, ["serve", "myapp:server", "-t", "http"]).exit_code == 0
        assert "allowed_hosts" not in recorded and "allowed_origins" not in recorded

    def test_stdio_dashboard_warns_on_stderr(self, monkeypatch):
        server = MCPServer(name="serve-test")
        monkeypatch.setattr(server, "run", lambda **kw: None)
        monkeypatch.setattr("promptise.mcp.server._serve_cli.resolve_server", lambda target: server)

        result = runner.invoke(app, ["serve", "myapp:server", "--dashboard"])
        assert result.exit_code == 0
        assert "--dashboard has no effect" in _all_output(result)


class TestStdioBannerSuppressed:
    """The startup banner must never print for stdio (stdout is the JSON-RPC
    protocol stream); it prints for http/sse."""

    @pytest.mark.asyncio
    async def test_no_banner_on_stdio(self, monkeypatch):
        server = MCPServer(name="banner-probe")
        printed = {"n": 0}
        monkeypatch.setattr(
            server, "_print_banner", lambda **kw: printed.__setitem__("n", printed["n"] + 1)
        )

        async def fake_transport(*a, **k):
            return None

        monkeypatch.setattr("promptise.mcp.server._app.run_transport", fake_transport)

        await server.run_async(transport="stdio")
        assert printed["n"] == 0, "banner must not print on stdio (corrupts JSON-RPC)"

        await server.run_async(transport="http", host="127.0.0.1", port=0)
        assert printed["n"] == 1, "banner should print for http"
