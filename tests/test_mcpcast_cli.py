"""Tests for the ``promptise mcpcast`` CLI command."""

from __future__ import annotations

import base64
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from promptise.cli import app
from promptise.mcpcast import MCPcastPlan
from promptise.mcpcast.readiness import EvalReport, TaskResult, ToolCall
from promptise.mcpcast.readiness import EvalTask as _EvalTask

runner = CliRunner()

SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Widgets API", "description": "Widgets."},
    "servers": [{"url": "https://w.example.com"}],
    "paths": {
        "/widgets": {
            "get": {"operationId": "listWidgets", "summary": "List widgets"},
            "post": {"operationId": "createWidget", "summary": "Create widget"},
        },
        "/widgets/{id}": {
            "delete": {
                "operationId": "deleteWidget",
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
            },
        },
    },
}


def _tools_source(out: Path) -> str:
    """Every generated tools module of the project under *out*, concatenated."""
    return "\n".join(p.read_text() for p in out.glob("*_mcp/tools/*.py"))


def _out(result) -> str:
    out = result.output
    try:
        out += result.stderr
    except (ValueError, AttributeError):
        pass
    return out


def _flat(result) -> str:
    """Output with Rich box-drawing and line wrapping collapsed to single spaces."""
    return re.sub(r"[│╭╮╰╯─\s]+", " ", _out(result))


@pytest.fixture()
def spec_file(tmp_path: Path) -> Path:
    p = tmp_path / "openapi.json"
    p.write_text(json.dumps(SPEC))
    return p


class TestArguments:
    def test_help(self):
        result = runner.invoke(app, ["mcpcast", "--help"])
        assert result.exit_code == 0
        assert "--no-curate" in _out(result) and "--profile" in _out(result)

    @pytest.mark.parametrize(
        "args",
        [
            ["--profile", "nope"],
            ["--auth", "magic"],
            ["--approval", "nobody"],
            ["--transport", "carrier-pigeon"],
            ["--max-tools", "0"],
            ["--eval-tasks", "0"],
        ],
    )
    def test_bad_option_exits_2(self, spec_file, args):
        result = runner.invoke(app, ["mcpcast", str(spec_file), "--no-curate", *args])
        assert result.exit_code == 2

    def test_missing_spec_exits_1(self, tmp_path):
        result = runner.invoke(app, ["mcpcast", str(tmp_path / "nope.yaml"), "--no-curate"])
        assert result.exit_code == 1
        assert "spec not found" in _out(result)


class TestGenerate:
    def test_no_curate_writes_project(self, spec_file, tmp_path):
        out = tmp_path / "widgets-mcp"
        result = runner.invoke(app, ["mcpcast", str(spec_file), "--no-curate", "--out", str(out)])
        assert result.exit_code == 0, _out(result)
        assert {p.name for p in out.iterdir()} >= {
            "mcpcast.plan.yaml",
            "server.py",
            "README.md",
            "pyproject.toml",
            "widgets_mcp",
            "tests",
        }
        plan = MCPcastPlan.load(out / "mcpcast.plan.yaml")
        assert plan.tool_names == ["list_widgets"]  # read-only default
        assert plan.api.name == "widgets" and plan.api.spec_source == str(spec_file)
        assert "tools: 1" in _out(result)

    def test_default_out_dir_uses_api_name(self, spec_file, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(
            app,
            [
                "mcpcast",
                str(spec_file),
                "--no-curate",
                "--profile",
                "full",
                "--auth",
                "none",
                "--name",
                "wid",
            ],
        )
        assert result.exit_code == 0, _out(result)
        plan = MCPcastPlan.load(tmp_path / "wid-mcp" / "mcpcast.plan.yaml")
        assert {t.name for t in plan.gated_tools} == {"create_widget", "delete_widget"}
        assert plan.api.auth.value == "none"

    def test_regenerate_from_plan(self, spec_file, tmp_path):
        out = tmp_path / "p"
        assert (
            runner.invoke(
                app,
                [
                    "mcpcast",
                    str(spec_file),
                    "--no-curate",
                    "--out",
                    str(out),
                    "--profile",
                    "standard",
                ],
            ).exit_code
            == 0
        )
        plan_path = out / "mcpcast.plan.yaml"
        plan_path.write_text(
            plan_path.read_text().replace("name: list_widgets", "name: browse_widgets")
        )
        (out / "server.py").write_text("stale")
        result = runner.invoke(app, ["mcpcast", str(plan_path), "--out", str(out)])
        assert result.exit_code == 0, _out(result)
        assert "Regenerating from plan" in _out(result)
        assert "stale" not in (out / "server.py").read_text()
        assert "browse_widgets" in _tools_source(out)
        assert MCPcastPlan.load(plan_path).profile.value == "standard"

    def test_eval_task_default_is_shared_with_the_wizard(self):
        from promptise import cli as cli_module
        from promptise.mcpcast.readiness import DEFAULT_EVAL_TASKS
        from promptise.mcpcast.wizard import WizardSettings

        assert cli_module._MCPCAST_EVAL_TASKS == DEFAULT_EVAL_TASKS
        assert WizardSettings(spec="x").eval_tasks == DEFAULT_EVAL_TASKS

    def test_regenerate_from_plan_url(self, spec_file, tmp_path, monkeypatch):
        """A plan served over HTTP regenerates too (nothing re-reads it as a local path)."""
        import httpx

        out = tmp_path / "p"
        assert (
            runner.invoke(
                app, ["mcpcast", str(spec_file), "--no-curate", "--out", str(out)]
            ).exit_code
            == 0
        )
        plan_text = (
            (out / "mcpcast.plan.yaml")
            .read_text()
            .replace("name: list_widgets", "name: browse_widgets")
        )
        real_client = httpx.Client

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/mcpcast.plan.yaml"
            return httpx.Response(200, text=plan_text)

        monkeypatch.setattr(
            httpx,
            "Client",
            lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
        )
        target = tmp_path / "from-url"
        result = runner.invoke(
            app, ["mcpcast", "https://plans.example.com/mcpcast.plan.yaml", "--out", str(target)]
        )
        assert result.exit_code == 0, _out(result)
        assert "Regenerating from plan" in _out(result)
        assert "browse_widgets" in _tools_source(target)
        # The fetched plan is written next to the package: the project is self-contained.
        assert "browse_widgets" in (target / "mcpcast.plan.yaml").read_text()

    def test_plan_file_rejects_spec_only_flags(self, spec_file, tmp_path):
        out = tmp_path / "p"
        runner.invoke(app, ["mcpcast", str(spec_file), "--no-curate", "--out", str(out)])
        result = runner.invoke(
            app, ["mcpcast", str(out / "mcpcast.plan.yaml"), "--profile", "full"]
        )
        assert result.exit_code == 2
        assert "cannot be combined with a plan file" in _flat(result)

    def test_review_confirm_and_abort(self, spec_file, tmp_path):
        out = tmp_path / "r"
        result = runner.invoke(
            app,
            [
                "mcpcast",
                str(spec_file),
                "--no-curate",
                "--out",
                str(out),
                "--review",
                "--profile",
                "standard",
            ],
            input="n\n",
        )
        assert result.exit_code == 1
        assert "Aborted" in _out(result) and not out.exists()
        assert "create_widget" in _out(result) and "deleteWidget" in _out(
            result
        )  # kept + dropped tables
        result = runner.invoke(
            app, ["mcpcast", str(spec_file), "--no-curate", "--out", str(out), "--review", "--yes"]
        )
        assert result.exit_code == 0, _out(result)
        assert (out / "server.py").exists()

    def test_curate_uses_model_and_budget(self, spec_file, tmp_path, monkeypatch):
        captured = {}

        async def fake_curate(operations, **kw):
            captured.update(kw)
            from promptise.mcpcast.plan import build_plan

            return build_plan(
                operations,
                **{
                    k: v
                    for k, v in kw.items()
                    if k
                    in {
                        "profile",
                        "base_url",
                        "auth",
                        "approval",
                        "name",
                        "description",
                        "spec_source",
                    }
                },
            )

        monkeypatch.setattr("promptise.mcpcast.curate", fake_curate)
        out = tmp_path / "c"
        result = runner.invoke(
            app,
            [
                "mcpcast",
                str(spec_file),
                "--out",
                str(out),
                "--model",
                "openai:gpt-5",
                "--max-tools",
                "7",
            ],
        )
        assert result.exit_code == 0, _out(result)
        assert captured["model"] == "openai:gpt-5" and captured["max_tools"] == 7
        assert "Curating with openai:gpt-5" in _out(result)

    def test_curation_failure_exits_1(self, spec_file, tmp_path, monkeypatch):
        from promptise.mcpcast.schema import MCPcastError

        async def failing(operations, **kw):
            raise MCPcastError("curation failed after 3 attempt(s)")

        monkeypatch.setattr("promptise.mcpcast.curate", failing)
        result = runner.invoke(app, ["mcpcast", str(spec_file), "--out", str(tmp_path / "x")])
        assert result.exit_code == 1 and "curation failed" in _out(result)


class TestActions:
    def test_eval_writes_report(self, spec_file, tmp_path, monkeypatch):
        seen = {}

        async def fake_evaluate(plan, build_server, **kw):
            seen["build_server"] = build_server
            seen.update(kw)
            task = _EvalTask(id="t1", prompt="List widgets", expected_tool="list_widgets")
            return EvalReport(
                grade="A",
                score=1.0,
                tasks_total=1,
                tasks_succeeded=1,
                selection_rate=1.0,
                param_error_rate=0.0,
                results=[
                    TaskResult(
                        task=task,
                        calls=[ToolCall(tool="list_widgets")],
                        success=True,
                        selected_correctly=True,
                    )
                ],
            )

        monkeypatch.setattr("promptise.mcpcast.readiness.evaluate", fake_evaluate)
        out = tmp_path / "e"
        result = runner.invoke(
            app,
            [
                "mcpcast",
                str(spec_file),
                "--no-curate",
                "--out",
                str(out),
                "--eval",
                "--eval-tasks",
                "3",
            ],
        )
        assert result.exit_code == 0, _out(result)
        assert callable(seen["build_server"]) and seen["tasks"] == 3
        assert seen["operations"] is not None
        assert (out / "eval" / "report.md").read_text().startswith("# Agent Readiness: A")
        assert "list_widgets" in (out / "eval" / "tasks.yaml").read_text()
        assert "Agent Readiness: A" in _out(result)

    def test_eval_with_no_tools_exits_1(self, tmp_path):
        spec = {**SPEC, "paths": {"/w": {"post": {"operationId": "createW"}}}}
        p = tmp_path / "s.json"
        p.write_text(json.dumps(spec))
        result = runner.invoke(
            app, ["mcpcast", str(p), "--no-curate", "--out", str(tmp_path / "o"), "--eval"]
        )
        assert result.exit_code == 1 and "nothing to evaluate" in _out(result)

    def test_serve_runs_generated_server(self, spec_file, tmp_path, monkeypatch):
        from promptise.mcp.server import MCPServer

        called = {}
        monkeypatch.setattr(MCPServer, "run", lambda self, **kw: called.update(kw))
        out = tmp_path / "s"
        result = runner.invoke(
            app,
            [
                "mcpcast",
                str(spec_file),
                "--no-curate",
                "--out",
                str(out),
                "--serve",
                "-t",
                "http",
                "-p",
                "9001",
            ],
        )
        assert result.exit_code == 0, _out(result)
        assert called == {"transport": "http", "host": "127.0.0.1", "port": 9001}
        assert "Serving" in _out(result)


class TestEvalFromPlan:
    def test_eval_from_plan_reloads_spec_for_mocks(self, spec_file, tmp_path, monkeypatch):
        seen = {}

        async def fake_evaluate(plan, build_server, **kw):
            seen.update(kw)
            task = _EvalTask(id="t1", prompt="List widgets", expected_tool="list_widgets")
            return EvalReport(
                grade="B",
                score=0.8,
                tasks_total=1,
                tasks_succeeded=1,
                selection_rate=1.0,
                param_error_rate=0.0,
                results=[
                    TaskResult(
                        task=task,
                        calls=[ToolCall(tool="list_widgets")],
                        success=True,
                        selected_correctly=True,
                    )
                ],
            )

        monkeypatch.setattr("promptise.mcpcast.readiness.evaluate", fake_evaluate)
        out = tmp_path / "p"
        assert (
            runner.invoke(
                app, ["mcpcast", str(spec_file), "--no-curate", "--out", str(out)]
            ).exit_code
            == 0
        )
        result = runner.invoke(
            app, ["mcpcast", str(out / "mcpcast.plan.yaml"), "--out", str(out), "--eval"]
        )
        assert result.exit_code == 0, _out(result)
        assert seen["operations"] is not None and {o.operation_id for o in seen["operations"]} == {
            "listWidgets",
            "createWidget",
            "deleteWidget",
        }

    def test_eval_from_plan_with_missing_spec_source_is_explicit(
        self, spec_file, tmp_path, monkeypatch
    ):
        out = tmp_path / "p"
        assert (
            runner.invoke(
                app, ["mcpcast", str(spec_file), "--no-curate", "--out", str(out)]
            ).exit_code
            == 0
        )
        spec_file.unlink()
        result = runner.invoke(
            app, ["mcpcast", str(out / "mcpcast.plan.yaml"), "--out", str(out), "--eval"]
        )
        assert result.exit_code == 1
        assert "could not be loaded" in _flat(result)

    def test_env_token_auth_accepted(self, spec_file, tmp_path):
        out = tmp_path / "e"
        result = runner.invoke(
            app,
            ["mcpcast", str(spec_file), "--no-curate", "--auth", "env-token", "--out", str(out)],
        )
        assert result.exit_code == 0, _out(result)
        assert MCPcastPlan.load(out / "mcpcast.plan.yaml").api.auth.value == "env-token"


class TestProjectSafety:
    def test_output_alias_and_force(self, spec_file, tmp_path):
        out = tmp_path / "w"
        assert (
            runner.invoke(
                app, ["mcpcast", str(spec_file), "--no-curate", "--output", str(out)]
            ).exit_code
            == 0
        )
        result = runner.invoke(
            app, ["mcpcast", str(spec_file), "--no-curate", "--output", str(out)]
        )
        assert result.exit_code == 1 and "already contains an mcpcast project" in _flat(result)
        assert (
            runner.invoke(
                app, ["mcpcast", str(spec_file), "--no-curate", "-o", str(out), "--force"]
            ).exit_code
            == 0
        )

    def test_plan_mode_defaults_to_its_directory_and_keeps_comments(self, spec_file, tmp_path):
        out = tmp_path / "w"
        assert (
            runner.invoke(
                app, ["mcpcast", str(spec_file), "--no-curate", "--out", str(out)]
            ).exit_code
            == 0
        )
        plan_path = out / "mcpcast.plan.yaml"
        plan_path.write_text(
            "# keep me\n"
            + plan_path.read_text().replace("name: list_widgets", "name: browse_widgets")
        )
        (out / "server.py").write_text("stale")
        result = runner.invoke(app, ["mcpcast", str(plan_path)])  # no --out
        assert result.exit_code == 0, _out(result)
        assert "stale" not in (out / "server.py").read_text()
        assert "browse_widgets" in _tools_source(out)
        assert plan_path.read_text().startswith("# keep me")
        assert not (out / "w-mcp").exists() and not (tmp_path / "widgets-mcp").exists()

    def test_review_escapes_markup(self, spec_file, tmp_path):
        out = tmp_path / "w"
        assert (
            runner.invoke(
                app, ["mcpcast", str(spec_file), "--no-curate", "--out", str(out)]
            ).exit_code
            == 0
        )
        plan_path = out / "mcpcast.plan.yaml"
        plan_path.write_text(
            plan_path.read_text().replace(
                "reason: write operation excluded by profile 'read-only'",
                "reason: internal [admin] endpoint [/bold]",
            )
        )
        result = runner.invoke(app, ["mcpcast", str(plan_path), "--review", "--yes"])
        assert result.exit_code == 0, _out(result)
        assert "[admin]" in _out(result) and "Description" in _out(result)

    def test_review_scrubs_terminal_control_sequences(self, tmp_path):
        """Rich escapes markup but lets ESC and the C1 CSI byte through, and a
        spec can put them in every cell of the human review table — an
        `ESC[2K` blanks the row of a destructive tool before the confirm."""
        blank = "\x1b[2K"
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Hostile API"},
            "servers": [{"url": "https://h.example.com"}],
            "paths": {
                f"/b{blank}": {
                    "delete": {
                        "operationId": "wipeAll",
                        "summary": f"Wipe all data{blank}",
                        "parameters": [
                            {
                                "name": f"scope{blank}",
                                "in": "query",
                                "schema": {"type": "string"},
                            }
                        ],
                    },
                    "post": {
                        "operationId": "uploadBlob",
                        "summary": "Upload",
                        "requestBody": {
                            "content": {f"application/octet-stream{blank}": {"schema": {}}}
                        },
                    },
                },
            },
        }
        path = tmp_path / "hostile.json"
        path.write_text(json.dumps(spec))
        out = tmp_path / "h"
        result = runner.invoke(
            app,
            [
                "mcpcast",
                str(path),
                "--no-curate",
                "--profile",
                "full",
                "--auth",
                "none",
                "--out",
                str(out),
                "--review",
                "--yes",
            ],
        )
        assert result.exit_code == 0, _out(result)
        text = _out(result)
        assert "\x1b" not in text and "\x9b" not in text
        assert "wipe_all" in text and "destructive" in text  # the row is still there
        assert "Not exposed" in text and "octet-stream" in text  # and so is the drop reason

    def test_review_table_scrubs_every_plan_cell(self):
        """Defence in depth below the parser: a plan object that still carries
        control characters (an older plan file, model output) renders inert."""
        import io
        from types import SimpleNamespace

        from rich.console import Console

        from promptise.cli import _print_plan_review

        esc, csi = "\x1b[2K", "\x9b2K"
        route = SimpleNamespace(method="DELETE", path=f"/b{esc}", operation_id=f"wipe{csi}All")
        tool = SimpleNamespace(
            name=f"wipe_all{esc}",
            risk=SimpleNamespace(value="destructive"),
            requires_approval=True,
            routes=[route],
            visible_params=[f"scope{esc}", "id"],
            description=f"{esc}" * 5 + "Wipe" + esc + " all data\nsecond line",
        )
        plan = SimpleNamespace(
            profile=SimpleNamespace(value="full"),
            tools=[tool],
            dropped=[SimpleNamespace(operation_id=f"op{csi}", reason=f"unsupported {esc}")],
        )
        buffer = io.StringIO()
        _print_plan_review(plan, Console(file=buffer, width=200))
        text = buffer.getvalue()
        assert "\x1b" not in text and "\x9b" not in text
        assert "wipe_all" in text and "DELETE /b" in text and "scope" in text
        assert "Wipe" in text and "all data" in text and "second line" not in text
        assert "unsupported" in text and "renamed from" in text

    def test_console_safe_scrubs_c0_c1_and_del_but_keeps_text(self):
        from promptise.cli import _console_safe

        assert _console_safe("a\x1b[2Kb\x9b2Kc\x7fd\x00e") == "a [2Kb 2Kc d e"
        assert _console_safe("tab\tnew\nline [bold]") == "tab\tnew\nline \\[bold]"

    def test_model_failure_is_a_clean_error(self, spec_file, tmp_path, monkeypatch):
        async def boom(operations, **kw):
            raise RuntimeError("Missing credentials")

        monkeypatch.setattr("promptise.mcpcast.curate", boom)
        result = runner.invoke(app, ["mcpcast", str(spec_file), "--out", str(tmp_path / "x")])
        assert result.exit_code == 1
        assert "could not be used for curation" in _flat(result) and "--no-curate" in _flat(result)
        assert "Traceback" not in _out(result)

    def test_import_failure_is_a_clean_error(self, spec_file, tmp_path, monkeypatch):
        monkeypatch.setenv("MCPCAST_CLIENT_KEYS", "oops")
        result = runner.invoke(
            app,
            [
                "mcpcast",
                str(spec_file),
                "--no-curate",
                "--auth",
                "api-key",
                "--out",
                str(tmp_path / "k"),
                "--serve",
            ],
        )
        # Whether the generated server validates its configuration at import
        # or at start, the CLI reports it as one clean error.
        assert result.exit_code == 1
        assert "MCPCAST_CLIENT_KEYS must be a JSON object" in _flat(result)
        assert "could not import" in _flat(result) or "could not start" in _flat(result)
        assert "Traceback" not in _out(result)

    def test_serve_goes_through_the_bind_guard(self, spec_file, tmp_path, monkeypatch):
        from promptise.mcp.server import MCPServer

        called = {}
        monkeypatch.setattr(MCPServer, "run", lambda self, **kw: called.update(kw))
        result = runner.invoke(
            app,
            [
                "mcpcast",
                str(spec_file),
                "--no-curate",
                "--auth",
                "none",
                "--out",
                str(tmp_path / "n"),
                "--serve",
                "-t",
                "http",
                "--host",
                "0.0.0.0",
            ],
        )
        assert result.exit_code == 2 and called == {}
        assert "non-loopback" in _flat(result)

    def test_public_needs_serve(self, spec_file, tmp_path):
        result = runner.invoke(
            app,
            ["mcpcast", str(spec_file), "--no-curate", "--out", str(tmp_path / "p"), "--public"],
        )
        assert result.exit_code == 2 and "--public only applies with --serve" in _flat(result)
        assert not (tmp_path / "p").exists()

    @pytest.mark.parametrize("auth", ["api-key", "passthrough"])
    def test_public_is_refused_for_modes_that_authenticate_callers(
        self, spec_file, tmp_path, monkeypatch, auth
    ):
        """An api-key / passthrough server already binds any host and has no
        --public switch: the CLI must say so before writing anything instead
        of regenerating the project and dying in the generated argparse."""
        from promptise.mcp.server import MCPServer

        monkeypatch.setenv("MCPCAST_CLIENT_KEYS", json.dumps({"k": {"client_id": "c"}}))
        called = {}
        monkeypatch.setattr(MCPServer, "run", lambda self, **kw: called.update(kw))
        out = tmp_path / auth
        result = runner.invoke(
            app,
            [
                "mcpcast",
                str(spec_file),
                "--no-curate",
                "--auth",
                auth,
                "--out",
                str(out),
                "--serve",
                "-t",
                "http",
                "--host",
                "0.0.0.0",
                "--public",
            ],
        )
        assert result.exit_code == 2, _out(result)
        assert "--public only applies to --auth env-token / none" in _flat(result)
        assert f"this project uses {auth}" in _flat(result)
        assert "unrecognized arguments" not in _out(result) and called == {}
        assert not out.exists()  # nothing written, nothing served

    def test_public_is_refused_when_regenerating_an_api_key_plan(self, spec_file, tmp_path):
        out = tmp_path / "k"
        assert (
            runner.invoke(
                app,
                ["mcpcast", str(spec_file), "--no-curate", "--auth", "api-key", "--out", str(out)],
            ).exit_code
            == 0
        )
        stamp = (out / "server.py").read_text()
        result = runner.invoke(
            app,
            ["mcpcast", str(out / "mcpcast.plan.yaml"), "--serve", "-t", "http", "--public"],
        )
        assert result.exit_code == 2 and "this project uses api-key" in _flat(result)
        assert (out / "server.py").read_text() == stamp  # not regenerated

    def test_public_is_forwarded_for_env_token(self, spec_file, tmp_path, monkeypatch):
        from promptise.mcp.server import MCPServer

        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", "tok")
        called = {}
        monkeypatch.setattr(MCPServer, "run", lambda self, **kw: called.update(kw))
        result = runner.invoke(
            app,
            [
                "mcpcast",
                str(spec_file),
                "--no-curate",
                "--auth",
                "env-token",
                "--out",
                str(tmp_path / "e"),
                "--serve",
                "-t",
                "http",
                "--host",
                "0.0.0.0",
                "--public",
            ],
        )
        assert result.exit_code == 0, _out(result)
        assert called == {"transport": "http", "host": "0.0.0.0", "port": 8080}

    def test_stdout_stays_clean_for_stdio_serve_with_curation(
        self, spec_file, tmp_path, monkeypatch
    ):
        from promptise.mcp.server import MCPServer

        async def fake_curate(operations, **kw):
            print("[promptise] No tools discovered from MCP servers; agent will run without tools.")
            from promptise.mcpcast.plan import build_plan

            return build_plan(
                operations,
                **{
                    k: v
                    for k, v in kw.items()
                    if k
                    in {
                        "profile",
                        "base_url",
                        "auth",
                        "approval",
                        "name",
                        "description",
                        "spec_source",
                    }
                },
            )

        monkeypatch.setattr("promptise.mcpcast.curate", fake_curate)
        monkeypatch.setattr(MCPServer, "run", lambda self, **kw: None)
        result = (
            CliRunner(mix_stderr=False).invoke(
                app, ["mcpcast", str(spec_file), "--out", str(tmp_path / "s"), "--serve"]
            )
            if "mix_stderr" in CliRunner.__init__.__code__.co_varnames
            else runner.invoke(
                app, ["mcpcast", str(spec_file), "--out", str(tmp_path / "s"), "--serve"]
            )
        )
        assert result.exit_code == 0, _out(result)
        assert result.stdout == ""

    def test_inline_spec_is_not_recorded_verbatim(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["mcpcast", json.dumps(SPEC), "--no-curate"])
        assert result.exit_code == 0, _out(result)
        plan = MCPcastPlan.load(tmp_path / "widgets-mcp" / "mcpcast.plan.yaml")
        assert plan.api.spec_source == "<inline>"

    def test_regenerate_from_a_long_inline_plan(self, spec_file, tmp_path, monkeypatch):
        """Real plans are longer than any filesystem's name limit (1024 on
        macOS, 4096 on Linux): the inline document must never be tried as a
        path, and never be echoed back."""
        out = tmp_path / "p"
        assert (
            runner.invoke(
                app,
                ["mcpcast", str(spec_file), "--no-curate", "--out", str(out), "--profile", "full"],
            ).exit_code
            == 0
        )
        plan = MCPcastPlan.load(out / "mcpcast.plan.yaml")
        # Pad the plan past 4096 bytes without changing what it generates.
        plan.tools[0].description = "List widgets. " + "Filler text for length. " * 200
        document = json.dumps(plan.model_dump(mode="json"))
        assert len(document) > 4096
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["mcpcast", document])
        assert result.exit_code == 0, _out(result)
        assert "Regenerating from plan <inline>" in _out(result)
        assert "Filler text" not in _out(result) and "Traceback" not in _out(result)
        target = tmp_path / "widgets-mcp"
        assert (target / "mcpcast.plan.yaml").exists() and (target / "server.py").exists()
        assert "Filler text" in (target / "mcpcast.plan.yaml").read_text()

    def test_is_file_never_raises_for_an_overlong_name(self):
        from promptise.cli import _is_file

        assert _is_file("{" + "x" * 5000) is False
        assert _is_file("\x00") is False
        assert _is_file(__file__) is True


class TestPlainHttpWarning:
    """A credentialed project whose upstream is plain http:// is told at
    generation time that every call fails with UPSTREAM_INSECURE until
    MCPCAST_ALLOW_INSECURE_HTTP=1 accepts the risk."""

    @pytest.mark.parametrize(
        "server, auth, warned",
        [
            ("http://api.intranet.corp:8080", "env-token", True),
            ("http://api.intranet.corp:8080", "api-key", True),
            ("http://api.intranet.corp:8080", "none", False),  # no credential travels
            ("https://w.example.com", "env-token", False),
            ("http://127.0.0.1:8080", "env-token", False),  # loopback is fine
        ],
    )
    def test_warning_names_the_hosts_and_the_variable(self, tmp_path, server, auth, warned):
        spec = {**SPEC, "servers": [{"url": server}]}
        path = tmp_path / "s.json"
        path.write_text(json.dumps(spec))
        result = runner.invoke(
            app,
            ["mcpcast", str(path), "--no-curate", "--auth", auth, "--out", str(tmp_path / "o")],
        )
        assert result.exit_code == 0, _out(result)
        text = _flat(result)
        assert ("MCPCAST_ALLOW_INSECURE_HTTP" in text) is warned
        assert ("api.intranet.corp:8080" in text and "UPSTREAM_INSECURE" in text) is warned


class TestEvalCredentialHint:
    """The hint before a paid --eval run fires only when live reads would
    really go out with the placeholder credential."""

    @pytest.mark.parametrize(
        "auth, env, hinted",
        [
            ("env-token", {}, True),
            ("env-token", {"MCPCAST_UPSTREAM_TOKEN": "real"}, False),
            ("env-token", {"MCPCAST_EVAL_AUTHORIZATION": "Bearer x"}, False),
            ("api-key", {}, True),
            ("api-key", {"MCPCAST_UPSTREAM_TOKENS": json.dumps({"acme": "t"})}, True),
            ("api-key", {"MCPCAST_UPSTREAM_TOKENS": json.dumps({"mcpcast-eval": "t"})}, False),
            ("api-key", {"MCPCAST_UPSTREAM_TOKENS": "not json"}, True),
            ("passthrough", {"MCPCAST_UPSTREAM_TOKEN": "real"}, True),  # not used by passthrough
            ("passthrough", {"MCPCAST_EVAL_HEADERS": '{"authorization": "Bearer x"}'}, False),
            ("none", {}, False),
        ],
    )
    def test_hint(self, spec_file, tmp_path, monkeypatch, auth, env, hinted):
        for name in (
            "MCPCAST_EVAL_HEADERS",
            "MCPCAST_EVAL_AUTHORIZATION",
            "MCPCAST_UPSTREAM_TOKEN",
            "MCPCAST_UPSTREAM_TOKENS",
        ):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)

        async def fake_evaluate(plan, build_server, **kw):
            raise RuntimeError("stop before any model call")

        monkeypatch.setattr("promptise.mcpcast.readiness.evaluate", fake_evaluate)
        result = runner.invoke(
            app,
            [
                "mcpcast",
                str(spec_file),
                "--no-curate",
                "--auth",
                auth,
                "--out",
                str(tmp_path / "o"),
                "--eval",
            ],
        )
        assert result.exit_code == 1 and "stop before any model call" in _flat(result)
        assert ("MCPCAST_EVAL_AUTHORIZATION='Bearer <token>'" in _flat(result)) is hinted


class TestGuidedSetupFlags:
    """The wizard branch: which flags pre-fill it, which are refused, what is printed."""

    @pytest.mark.parametrize(
        "args",
        [
            ["--yes"],
            ["--transport", "http"],
            ["--host", "0.0.0.0"],
            ["--port", "9"],
            ["--profile", "full"],
            ["--eval"],
            ["--no-curate"],
        ],
    )
    def test_flags_the_wizard_collects_itself_are_refused(self, args):
        result = runner.invoke(app, ["mcpcast", *args])
        assert result.exit_code == 2, _out(result)
        assert f"{args[0]} cannot be combined with the guided setup" in _flat(result)

    @pytest.mark.parametrize(
        "args",
        [["--model", "anthropic:claude-sonnet-4.5"], ["--eval-tasks", "5"], ["--force"]],
    )
    def test_prefill_flags_pass_the_conflict_check(self, args):
        # Without a terminal the next gate is the TTY check — not a conflict.
        result = runner.invoke(app, ["mcpcast", *args])
        assert result.exit_code == 1
        assert "needs an interactive terminal" in _out(result)

    def test_prefills_reach_the_wizard_and_the_summary_is_printed(
        self, spec_file, tmp_path, monkeypatch
    ):
        import promptise.cli as cli_module
        import promptise.mcpcast.wizard as wizard
        from promptise.mcpcast import mcpcast as build
        from promptise.mcpcast.wizard import WizardResult

        received = {}

        def fake_wizard(spec, **kwargs):
            received["spec"] = spec
            received.update(kwargs)
            plan = build(SPEC, name="widgets")
            out = tmp_path / "widgets-mcp"
            return WizardResult(
                plan=plan,
                out_dir=out,
                written=[out / "server.py"],
                command="promptise mcpcast openapi.json --eval",
                eval_requested=True,  # switched on, but the wizard was quit before it finished
            )

        monkeypatch.setattr(cli_module, "_interactive_terminal", lambda: True)
        monkeypatch.setattr(wizard, "run_wizard", fake_wizard)
        result = runner.invoke(
            app,
            [
                "mcpcast",
                str(spec_file),
                "-i",
                "--model",
                "anthropic:claude-sonnet-4.5",
                "--eval-tasks",
                "5",
                "--force",
                "--out",
                "x",
                "--base-url",
                "https://a",
            ],
        )
        assert result.exit_code == 0, _out(result)
        assert received == {
            "spec": str(spec_file),
            "base_url": "https://a",
            "out_dir": "x",
            "model": "anthropic:claude-sonnet-4.5",
            "eval_tasks": 5,
            "force": True,
        }
        assert "tools: 1" in _out(result)
        assert "Agent Readiness: not completed" in _flat(result)
        assert "mcpcast.plan.yaml --eval" in _flat(result)
        assert "promptise mcpcast openapi.json --eval" in _out(result)

    def test_defaults_reach_the_wizard_as_given(self, tmp_path, monkeypatch):
        """``--eval-tasks 20`` stays 20 (the CLI and wizard share the default); the
        model is only pre-filled when it differs from the default."""
        import promptise.cli as cli_module
        import promptise.mcpcast.wizard as wizard

        received = {}

        def fake_wizard(spec, **kwargs):
            received.update(kwargs)
            return None

        monkeypatch.setattr(cli_module, "_interactive_terminal", lambda: True)
        monkeypatch.setattr(wizard, "run_wizard", fake_wizard)
        result = runner.invoke(
            app, ["mcpcast", "--model", "openai:gpt-5-mini", "--eval-tasks", "20"]
        )
        assert result.exit_code == 1 and "Nothing written." in _out(result)
        assert received["model"] is None and received["eval_tasks"] == 20
        assert received["force"] is False


class TestOccupiedOutputDirectory:
    def test_a_folder_with_files_needs_force(self, spec_file, tmp_path):
        mine = tmp_path / "my-existing-app"
        mine.mkdir()
        (mine / "README.md").write_text("# mine\n")
        result = runner.invoke(app, ["mcpcast", str(spec_file), "--no-curate", "--out", str(mine)])
        assert result.exit_code == 1
        assert "not an mcpcast project" in _flat(result) and "--force" in _flat(result)
        assert (mine / "README.md").read_text() == "# mine\n"
        result = runner.invoke(
            app, ["mcpcast", str(spec_file), "--no-curate", "--out", str(mine), "--force"]
        )
        assert result.exit_code == 0, _out(result)
        assert (mine / "README.md").read_text() != "# mine\n"
        assert (mine / "mcpcast.plan.yaml").exists()

    def test_an_empty_folder_is_fine(self, spec_file, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        result = runner.invoke(app, ["mcpcast", str(spec_file), "--no-curate", "--out", str(empty)])
        assert result.exit_code == 0, _out(result)

    def test_write_failures_are_clean_errors(self, spec_file, tmp_path, monkeypatch):
        import promptise.mcpcast as mcpcast_pkg

        def refuse(plan, out_dir, **kwargs):
            raise UnicodeEncodeError("utf-8", "\ud83d", 0, 1, "surrogates not allowed")

        monkeypatch.setattr(mcpcast_pkg, "write_project", refuse)
        result = runner.invoke(
            app, ["mcpcast", str(spec_file), "--no-curate", "--out", str(tmp_path / "w")]
        )
        assert result.exit_code == 1
        assert "could not write" in _flat(result) and "UnicodeEncodeError" in _flat(result)
        assert "Traceback" not in _out(result)


# ---------------------------------------------------------------------------
# Audit findings: credentials in a spec URL, malformed documents, slow servers
# ---------------------------------------------------------------------------

SERVERLESS_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Ledger", "description": "Ledger entries."},
    "paths": {"/entries": {"get": {"operationId": "listEntries", "summary": "List entries"}}},
}


class _SpecHandler(BaseHTTPRequestHandler):
    """Serves whatever the test put in ``served`` and records every request."""

    served: dict[str, bytes] = {}
    seen: list[dict[str, str | None]] = []

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        type(self).seen.append(
            {"path": self.path, "authorization": self.headers.get("Authorization")}
        )
        body = type(self).served.get(self.path.split("?")[0])
        if body is None:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:  # keep the test output clean
        pass


@pytest.fixture()
def spec_server():
    """A loopback HTTP server: yields ``(origin, handler class)``."""
    _SpecHandler.served = {"/docs/openapi.json": json.dumps(SERVERLESS_SPEC).encode()}
    _SpecHandler.seen = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SpecHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", _SpecHandler
    finally:
        server.shutdown()
        server.server_close()


def _files(out: Path) -> dict[Path, str]:
    return {p: p.read_text() for p in out.rglob("*") if p.is_file()}


class TestCredentialsInSpecUrl:
    def test_credential_fetches_the_spec_and_goes_nowhere_else(self, spec_server, tmp_path):
        """The audit repro: userinfo + query on the spec URL, a spec with no servers block."""
        origin, handler = spec_server
        secret_url = (
            origin.replace("http://", "http://svc-user:S3CRET@")
            + "/docs/openapi.json?api_key=QUERY-SECRET"
        )
        out = tmp_path / "ledger-mcp"
        result = runner.invoke(
            app,
            ["mcpcast", secret_url, "--no-curate", "--auth", "env-token", "--out", str(out)],
        )
        assert result.exit_code == 0, _out(result)

        # the credential authenticated the download …
        (request,) = handler.seen
        assert request["path"] == "/docs/openapi.json?api_key=QUERY-SECRET"
        assert request["authorization"] == "Basic " + base64.b64encode(b"svc-user:S3CRET").decode()

        # … and appears nowhere else: not on stderr, not in any generated file
        output = _out(result)
        assert f"Parsed 1 operations from {origin}/docs/openapi.json" in output
        for secret in ("S3CRET", "svc-user", "QUERY-SECRET", "api_key="):
            assert secret not in output, secret
        files = _files(out)
        assert {p.name for p in files} >= {"mcpcast.plan.yaml", "README.md", ".env.example"}
        for path, text in files.items():
            for secret in ("S3CRET", "svc-user", "QUERY-SECRET", "api_key="):
                assert secret not in text, (path, secret)
        plan = MCPcastPlan.load(out / "mcpcast.plan.yaml")
        assert plan.api.base_url == origin and "@" not in plan.api.base_url
        assert plan.api.spec_source == f"{origin}/docs/openapi.json"
        config = (out / "ledger_mcp" / "config.py").read_text()
        assert f'"{origin}"' in config and "S3CRET" not in config

    def test_fetch_failure_does_not_echo_the_credential(self, spec_server):
        origin, _ = spec_server
        secret_url = (
            origin.replace("http://", "http://svc-user:S3CRET@") + "/missing.json?api_key=Q"
        )
        result = runner.invoke(app, ["mcpcast", secret_url, "--no-curate"])
        assert result.exit_code == 1
        output = _out(result)
        assert (
            f"Error: could not fetch spec from {origin}/missing.json" in output and "404" in output
        )
        assert "S3CRET" not in output and "api_key" not in output and "Traceback" not in output

    def test_plan_fetched_with_a_credential_is_echoed_scrubbed(
        self, spec_server, spec_file, tmp_path
    ):
        origin, handler = spec_server
        first = tmp_path / "first"
        assert (
            runner.invoke(
                app, ["mcpcast", str(spec_file), "--no-curate", "--out", str(first)]
            ).exit_code
            == 0
        )
        handler.served["/plans/mcpcast.plan.yaml"] = (first / "mcpcast.plan.yaml").read_bytes()
        out = tmp_path / "second"
        result = runner.invoke(
            app,
            [
                "mcpcast",
                origin.replace("http://", "http://u:S3CRET@") + "/plans/mcpcast.plan.yaml?k=Q",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0, _out(result)
        assert f"Regenerating from plan {origin}/plans/mcpcast.plan.yaml" in _out(result)
        assert "S3CRET" not in _out(result) and "k=Q" not in _out(result)

    def test_base_url_option_with_a_credential_is_refused(self, spec_file, tmp_path):
        result = runner.invoke(
            app,
            [
                "mcpcast",
                str(spec_file),
                "--no-curate",
                "--base-url",
                "https://u:S3CRET@api.test",
                "--out",
                str(tmp_path / "o"),
            ],
        )
        assert result.exit_code == 1
        assert "must not carry credentials" in _out(result) and "MCPCAST_UPSTREAM_TOKEN" in _out(
            result
        )
        assert "Traceback" not in _out(result) and not (tmp_path / "o").exists()


_DEEP = '{"a":' * 100_000 + "1" + "}" * 100_000
_BASE = {"openapi": "3.0.0", "info": {"title": "T"}, "servers": [{"url": "https://t.io"}]}


class TestMalformedDocuments:
    """The seven audit repros: never a traceback. A whole-document defect is an ``Error:``;
    a defect inside one operation drops that operation with its reason in the plan."""

    @pytest.mark.parametrize(
        ("document", "outcome", "detail"),
        [
            (
                json.dumps({**_BASE, "info": "oops", "paths": {"/x": {"get": {}}}}),
                "error",
                "'info' must be a mapping, got string",
            ),
            (
                json.dumps(
                    {
                        **_BASE,
                        "paths": {"/x": {"get": {"operationId": "getX", "parameters": "nope"}}},
                    }
                ),
                "dropped",
                "malformed parameters: expected a list of mappings, got string",
            ),
            (
                json.dumps(
                    {
                        **_BASE,
                        "paths": {"/x": {"get": {"operationId": "getX", "parameters": ["nope"]}}},
                    }
                ),
                "dropped",
                "malformed parameters entry: expected a mapping, got string",
            ),
            (json.dumps({**_BASE, "paths": {"/x": {"get": {"operationId": 5}}}}), "tool", "op_5"),
            (
                json.dumps({**_BASE, "paths": {"/x": {"get": {"operationId": "getX", "tags": 5}}}}),
                "tool",
                "get_x",
            ),
            (
                json.dumps(
                    {
                        **_BASE,
                        "paths": {
                            "/x": {
                                "post": {"operationId": "makeX", "requestBody": {"content": "nope"}}
                            }
                        },
                    }
                ),
                "dropped",
                "malformed requestBody.content: expected a mapping, got string",
            ),
            (_DEEP, "error", "document is nested too deeply to parse"),
        ],
        ids=[
            "info-string",
            "parameters-string",
            "parameters-list-of-strings",
            "operationId-int",
            "tags-int",
            "content-string",
            "100000-deep",
        ],
    )
    def test_no_traceback(self, tmp_path, document, outcome, detail):
        spec = tmp_path / "spec.json"
        spec.write_text(document)
        out = tmp_path / "out"
        result = runner.invoke(
            app, ["mcpcast", str(spec), "--no-curate", "--profile", "full", "--out", str(out)]
        )
        output = _out(result)
        assert "Traceback" not in output
        if outcome == "error":
            assert result.exit_code == 1, output
            assert f"Error: {spec}: {detail}" in output
            assert not out.exists()
            return
        assert result.exit_code == 0 and "Error:" not in output, output
        plan = MCPcastPlan.load(out / "mcpcast.plan.yaml")
        if outcome == "dropped":
            assert plan.tools == []
            (dropped,) = plan.dropped
            assert dropped.reason == f"unsupported by mcpcast: {detail}"
        else:
            assert plan.tool_names == [detail] and plan.dropped == []

    def test_alias_bomb_is_refused_quickly(self, tmp_path):
        lines = ["a: &a [" + ", ".join(["x"] * 10) + "]"]
        for i in range(1, 9):
            cur, prev = chr(ord("a") + i), chr(ord("a") + i - 1)
            lines.append(f"{cur}: &{cur} [" + ", ".join([f"*{prev}"] * 10) + "]")
        spec = tmp_path / "bomb.yaml"
        spec.write_text(
            "openapi: 3.0.0\ninfo: {title: bomb}\npaths:\n  /x: {get: {}}\n"
            + "\n".join(lines)
            + "\n"
        )
        assert spec.stat().st_size < 1024
        started = time.monotonic()
        result = runner.invoke(app, ["mcpcast", str(spec), "--no-curate"])
        assert time.monotonic() - started < 3
        assert result.exit_code == 1 and "Traceback" not in _out(result)
        assert f"Error: {spec}: document expands to more than 2000000 nodes" in _out(result)
        assert "MCPCAST_MAX_SPEC_NODES" in _out(result)


class TestSlowSpecServer:
    def test_trickling_server_hits_the_deadline(self, monkeypatch):
        class Trickle(httpx.SyncByteStream):
            def __iter__(self):
                for chunk in [b'{"openapi": "3.0.0", "paths": {}', *([b" "] * 500), b"}"]:
                    time.sleep(0.01)
                    yield chunk

        real_client = httpx.Client
        monkeypatch.setattr(
            httpx,
            "Client",
            lambda **kw: real_client(
                transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=Trickle())), **kw
            ),
        )
        monkeypatch.setenv("MCPCAST_FETCH_SECONDS", "0.3")
        started = time.monotonic()
        result = runner.invoke(app, ["mcpcast", "https://slow.example/openapi.json", "--no-curate"])
        assert time.monotonic() - started < 3
        assert result.exit_code == 1 and "Traceback" not in _out(result)
        assert (
            "Error: downloading the spec from https://slow.example/openapi.json took longer than 0.3s"
            in _out(result)
        )
        assert "MCPCAST_FETCH_SECONDS" in _out(result)
