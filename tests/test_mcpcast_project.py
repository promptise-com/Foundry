"""The generated project layout: an installable package, a launcher, tests, scaffold.

These tests treat the output the way a developer would — import it, lint the
layout, run its own test suite — rather than grepping one big file.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from promptise.mcpcast import (
    SCAFFOLD_ONCE,
    MCPcastPlan,
    load_generated_server,
    mcpcast,
    render_project,
    write_project,
)
from promptise.mcpcast.emit import package_name, tool_group
from promptise.mcpcast.schema import ApprovalMode, AuthMode, SafetyProfile

SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Helpdesk API", "description": "Tickets and customers."},
    "servers": [{"url": "https://help.example/api"}],
    "paths": {
        "/tickets": {
            "get": {"operationId": "listTickets", "summary": "List tickets", "tags": ["tickets"]},
            "post": {
                "operationId": "createTicket",
                "summary": "Open a ticket",
                "tags": ["tickets"],
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["subject"],
                                "properties": {"subject": {"type": "string"}},
                            }
                        }
                    }
                },
            },
        },
        "/tickets/{id}": {
            "delete": {
                "operationId": "deleteTicket",
                "summary": "Delete a ticket",
                "tags": ["tickets"],
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
            }
        },
        "/customers/{id}": {
            "get": {
                "operationId": "getCustomer",
                "summary": "Get a customer",
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
            }
        },
        "/v1/admin/purge": {"post": {"operationId": "purgeAll", "summary": "Purge everything"}},
    },
}

PY_FILES = {
    "server.py",
    "helpdesk_mcp/__init__.py",
    "helpdesk_mcp/__main__.py",
    "helpdesk_mcp/config.py",
    "helpdesk_mcp/upstream.py",
    "helpdesk_mcp/approval.py",
    "helpdesk_mcp/server.py",
    "helpdesk_mcp/tools/__init__.py",
    "tests/conftest.py",
    "tests/test_tools.py",
}


def _plan(**kw) -> MCPcastPlan:
    return mcpcast(SPEC, name="helpdesk", **kw)


class TestLayout:
    def test_every_file_of_the_project(self) -> None:
        files = render_project(_plan(profile=SafetyProfile.FULL))
        assert set(files) == PY_FILES | SCAFFOLD_ONCE | {
            "README.md",
            "helpdesk_mcp/tools/tickets.py",
            "helpdesk_mcp/tools/customers.py",
            "helpdesk_mcp/tools/admin.py",
        }
        for path, source in files.items():
            if path.endswith(".py"):
                compile(source, path, "exec")

    def test_package_and_group_names(self) -> None:
        assert package_name("helpdesk") == "helpdesk_mcp"
        assert package_name("widgets-api") == "widgets_api_mcp"
        assert package_name("1shop") == "api_1shop_mcp"
        plan = _plan(profile=SafetyProfile.FULL)
        groups = {t.name: tool_group(t) for t in plan.tools}
        assert groups["list_tickets"] == "tickets"  # from the tag
        assert groups["get_customer"] == "customers"  # from the path
        assert groups["purge_all"] == "admin"  # the /v1 prefix is skipped

    def test_pyproject_is_a_real_project(self) -> None:
        files = render_project(_plan())
        toml = files["pyproject.toml"]
        assert 'name = "helpdesk-mcp"' in toml
        assert 'helpdesk-mcp = "helpdesk_mcp.__main__:main"' in toml
        assert '"promptise>=' in toml and '"httpx>=' in toml
        assert "[tool.ruff]" in toml and "asyncio_mode" in toml
        assert 'CMD ["helpdesk-mcp", "--transport", "http"' in files["Dockerfile"]
        assert "# MCPCAST_BASE_URL=https://help.example/api" in files[".env.example"]

    def test_env_example_follows_the_auth_mode(self) -> None:
        env_token = render_project(_plan(auth=AuthMode.ENV_TOKEN))[".env.example"]
        assert "MCPCAST_UPSTREAM_TOKEN=" in env_token and "# MCPCAST_PUBLIC=1" in env_token
        api_key = render_project(_plan(auth=AuthMode.API_KEY))[".env.example"]
        assert "MCPCAST_CLIENT_KEYS=" in api_key and "MCPCAST_UPSTREAM_TOKENS=" in api_key
        assert "MCPCAST_MAX_PENDING=100" in api_key
        assert "MCPCAST_MAX_PENDING_PER_TENANT=40" in api_key
        assert "MCPCAST_MAX_PENDING_PER_CLIENT=20" in api_key
        none = render_project(_plan(auth=AuthMode.NONE))
        assert "MCPCAST_UPSTREAM_TOKEN=" not in none[".env.example"]
        assert "MCPCAST_MAX_PENDING" not in none[".env.example"]  # pending needs api-key
        for text in (env_token, api_key, none[".env.example"]):
            assert "MCPCAST_TIMEOUT=30" in text and "Total time one upstream call" in text
            assert "MCPCAST_MAX_RESPONSE_BYTES=1048576" in text
            assert "MCPCAST_ERROR_EXCERPT_CHARS=500" in text
            assert "# MCPCAST_ALLOW_INSECURE_HTTP=1" in text
            # the base URL is compiled in; the override is opt-in, never pre-set
            assert "\nMCPCAST_BASE_URL=" not in text

    def test_dockerfile_is_hardened_and_follows_the_auth_mode(self) -> None:
        for auth in AuthMode:
            docker = render_project(_plan(auth=auth))["Dockerfile"]
            assert docker.startswith("FROM python:3.12-slim-bookworm\n")
            assert "useradd --system --create-home" in docker and "\nUSER app\n" in docker
            assert "ENV " not in docker  # nothing baked in: no base URL, no credential
            assert "MCPCAST_UPSTREAM_TOKEN=" not in docker and "MCPCAST_CLIENT_KEYS=" not in docker
        # no MCP-level authentication: stdio only, --public / MCPCAST_PUBLIC=1 documented
        for auth in (AuthMode.NONE, AuthMode.ENV_TOKEN):
            docker = render_project(_plan(auth=auth))["Dockerfile"]
            assert docker.rstrip().endswith('CMD ["helpdesk-mcp"]') and '"0.0.0.0"' not in docker
            assert "--public" in docker and "MCPCAST_PUBLIC=1" in docker
            assert "authenticating gateway" in docker
        passthrough = render_project(_plan(auth=AuthMode.PASSTHROUGH))["Dockerfile"]
        assert '"--host", "0.0.0.0"' in passthrough
        assert "unauthenticated relay" in passthrough and "gateway" in passthrough
        api_key = render_project(_plan(auth=AuthMode.API_KEY))["Dockerfile"]
        assert '"--host", "0.0.0.0"' in api_key and "MCPCAST_CLIENT_KEYS" in api_key

    def test_approval_module_matches_the_plan(self) -> None:
        read_only = render_project(_plan())["helpdesk_mcp/approval.py"]
        assert "ApprovalGateMiddleware" not in read_only and "nothing to gate" in read_only
        elicit = render_project(_plan(profile=SafetyProfile.STANDARD))["helpdesk_mcp/approval.py"]
        assert "ElicitationApprover()" in elicit and "PendingApprover" not in elicit
        pending = render_project(
            _plan(profile=SafetyProfile.FULL, auth=AuthMode.API_KEY, approval=ApprovalMode.PENDING)
        )["helpdesk_mcp/approval.py"]
        assert "register_tenant_scoped_approvals" in pending and "approvals_decide" in pending


class TestWriteProject:
    def test_scaffold_is_written_once_and_derived_files_are_rewritten(self, tmp_path: Path) -> None:
        out = tmp_path / "helpdesk-mcp"
        first = {p.relative_to(out).as_posix() for p in write_project(_plan(), out)}
        assert first >= SCAFFOLD_ONCE | PY_FILES | {"README.md", "mcpcast.plan.yaml"}
        (out / "pyproject.toml").write_text("# my packaging\n", encoding="utf-8")
        (out / "Dockerfile").write_text("# my image\n", encoding="utf-8")
        (out / "helpdesk_mcp" / "config.py").write_text("garbage", encoding="utf-8")
        second = {p.relative_to(out).as_posix() for p in write_project(_plan(), out)}
        assert not (second & SCAFFOLD_ONCE)  # kept
        assert (out / "pyproject.toml").read_text(encoding="utf-8") == "# my packaging\n"
        assert (out / "Dockerfile").read_text(encoding="utf-8") == "# my image\n"
        assert "garbage" not in (out / "helpdesk_mcp" / "config.py").read_text(encoding="utf-8")

    def test_stale_tools_modules_are_removed(self, tmp_path: Path) -> None:
        out = tmp_path / "helpdesk-mcp"
        write_project(_plan(profile=SafetyProfile.FULL), out)
        assert (out / "helpdesk_mcp" / "tools" / "admin.py").exists()
        write_project(_plan(profile=SafetyProfile.READ_ONLY), out)  # purge_all is gone
        assert not (out / "helpdesk_mcp" / "tools" / "admin.py").exists()
        assert "admin" not in (out / "helpdesk_mcp" / "tools" / "__init__.py").read_text(
            encoding="utf-8"
        )
        module = load_generated_server(out / "server.py")
        assert {t.name for t in module.server._tool_registry.list_all()} == {
            "list_tickets",
            "get_customer",
        }

    def test_same_named_projects_load_independently(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a" / "helpdesk-mcp", tmp_path / "b" / "helpdesk-mcp"
        write_project(_plan(profile=SafetyProfile.READ_ONLY), a)
        write_project(_plan(profile=SafetyProfile.FULL), b)
        names_a = {t.name for t in load_generated_server(a).server._tool_registry.list_all()}
        names_b = {t.name for t in load_generated_server(b).server._tool_registry.list_all()}
        assert names_a == {"list_tickets", "get_customer"}
        assert names_b > names_a and "purge_all" in names_b
        names_a_again = {
            t.name for t in load_generated_server(a / "server.py").server._tool_registry.list_all()
        }
        assert names_a_again == names_a  # not the cached copy of b

    def test_launcher_main_serves_the_module_server(self, tmp_path: Path, monkeypatch) -> None:
        out = tmp_path / "helpdesk-mcp"
        write_project(_plan(), out)
        module = load_generated_server(out)
        called: dict = {}
        monkeypatch.setattr(module.server, "run", lambda **kw: called.update(kw))
        module.main(["--transport", "http", "--port", "9001"])
        assert called == {"transport": "http", "host": "127.0.0.1", "port": 9001}


KEBAB_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Graph"},
    "servers": [{"url": "https://graph.example/v1"}],
    "paths": {
        "/users/{user-id}": {
            "get": {
                "operationId": "getUser",
                "parameters": [
                    {
                        "name": "user-id",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
            }
        },
        "/events": {
            "get": {
                "operationId": "listEvents",
                "parameters": [
                    {
                        "name": "from",
                        "in": "query",
                        "required": True,
                        "schema": {"type": "string", "format": "date"},
                    },
                    {"name": "to", "in": "query", "required": True, "schema": {"type": "string"}},
                    {"name": "type", "in": "query", "schema": {"type": "string"}},
                    {"name": "fields", "in": "query", "schema": {"type": "string"}},
                ],
            }
        },
        "/teams/{enterprise-team}/members": {
            "post": {
                "operationId": "addMember",
                "parameters": [
                    {
                        "name": "enterprise-team",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    }
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["class"],
                                "properties": {"class": {"type": "string"}},
                            }
                        }
                    },
                },
            }
        },
    },
}


def _run_generated_suite(out: Path, tmp_path: Path) -> subprocess.CompletedProcess:
    """Run a generated project's own tests exactly as a developer would."""
    # No cwd= and close_fds=False: CPython then spawns with posix_spawn instead of
    # fork+exec, which a parent full of native threads (torch, transformers) cannot
    # survive on macOS. The project's pyproject.toml supplies rootdir and pythonpath.
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-c",
            str(out / "pyproject.toml"),
            "--rootdir",
            str(out),
            str(out / "tests"),
        ],
        close_fds=False,
        capture_output=True,
        text=True,
        timeout=180,
        env=_hermetic_env(tmp_path),
    )


def _hermetic_env(tmp_path: Path) -> dict[str, str]:
    """The parent's environment without any MCPcast/Promptise configuration.

    The system variables stay: on Windows a child without ``SYSTEMROOT`` cannot
    initialise Winsock (``import asyncio`` fails with WinError 10106), and
    ``PATH``/``TEMP`` are needed everywhere. Only the variables a generated
    server or ``promptise`` would read are dropped, so the generated suite runs
    as it would for a developer with a clean shell.
    """
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("MCPCAST_", "PROMPTISE_", "OPENAI_", "ANTHROPIC_"))
    }
    env["HOME"] = str(tmp_path)
    env["USERPROFILE"] = str(tmp_path)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


@pytest.fixture(scope="session")
def mypy_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One mypy cache for the session: the dependencies are analysed once, each project quickly."""
    return tmp_path_factory.mktemp("mypy-cache")


def _assert_lints_clean(out: Path, mypy_cache: Path) -> None:
    """The project is clean under its own ``[tool.ruff]`` and under mypy's defaults.

    ``ruff check`` runs with ``E501`` added to the project's rule set: the
    generated ``pyproject.toml`` declares ``line-length = 100`` and the
    generator promises to honour it, so an over-long line is a bug even
    though ruff's default rules would not report it.  ``ruff format --check``
    and ``mypy`` run exactly as a developer would run them in the project.
    """
    pytest.importorskip("ruff")
    pytest.importorskip("mypy")
    config = str(out / "pyproject.toml")
    commands = [
        ["ruff", "check", "--no-cache", "--extend-select", "E501", "--config", config, str(out)],
        ["ruff", "format", "--check", "--no-cache", "--config", config, str(out)],
        ["mypy", "--config-file", config, "--cache-dir", str(mypy_cache), str(out)],
    ]
    for command in commands:
        result = subprocess.run(
            [sys.executable, "-m", *command],
            close_fds=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, " ".join(command) + "\n" + result.stdout + result.stderr


def test_generated_tests_use_the_identifiers_the_tools_expose(
    tmp_path: Path, mypy_cache: Path
) -> None:
    """``{user-id}`` → ``user_id``, ``from`` → ``from_``, ``class`` → ``class_``: the generated
    tests must call the tools the way a client has to, or they fail out of the box."""
    plan = mcpcast(KEBAB_SPEC, name="graph", profile=SafetyProfile.STANDARD, auth=AuthMode.NONE)
    out = tmp_path / "graph-mcp"
    write_project(plan, out)
    tests = (out / "tests" / "test_tools.py").read_text(encoding="utf-8")
    assert 'call_tool("get_user", {"user_id": "123"})' in tests
    assert 'call_tool("list_events", {"from_": "2026-01-15", "to": "string"})' in tests
    assert 'call_tool("add_member", {"enterprise_team": "string", "class_": "string"})' in tests
    assert '"user-id"' not in tests and '"from"' not in tests
    result = _run_generated_suite(out, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "5 passed" in result.stdout  # listing + 3 tools + 1 approval test
    _assert_lints_clean(out, mypy_cache)


@pytest.mark.parametrize(
    ("profile", "auth", "approval"),
    [
        (SafetyProfile.READ_ONLY, AuthMode.NONE, None),
        (SafetyProfile.STANDARD, AuthMode.ENV_TOKEN, None),
        (SafetyProfile.FULL, AuthMode.PASSTHROUGH, None),
        (SafetyProfile.FULL, AuthMode.API_KEY, ApprovalMode.PENDING),
    ],
    ids=["read-only/none", "standard/env-token", "full/passthrough", "full/api-key/pending"],
)
def test_generated_test_suite_passes(
    tmp_path: Path, mypy_cache: Path, profile, auth, approval
) -> None:
    """The project's own tests/ run green with pytest, exactly as a developer would run them,
    and the code lints, is formatted and type-checks under the project's own configuration."""
    out = tmp_path / "helpdesk-mcp"
    write_project(_plan(profile=profile, auth=auth, approval=approval), out)
    result = _run_generated_suite(out, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    plan = mcpcast(SPEC, name="helpdesk", profile=profile)
    expected = 1 + len(plan.tools) + len(plan.gated_tools)
    assert f"{expected} passed" in result.stdout
    _assert_lints_clean(out, mypy_cache)


_LONG_TAGS = [f"resource-number-{i:02d}-with-a-rather-long-tag-name" for i in range(14)]
_LONG_OPERATION = "archiveEveryCustomerRecordOlderThanTheRetentionPolicyAllows"  # 59 chars
_LONG_PARAM = "the_customer_account_identifier_that_selects_the_record_sets"  # 60 chars
# Past the identifier cap: an optional string, an object and an array whose annotations
# would push a signature line over the margin at 73, 62 and 67 characters.
_LONGER_STRING = "the_optional_free_text_note_the_operator_attaches_to_the_archive_requests"  # 73
_LONGER_OBJECT = "the_retention_policy_override_object_with_days_and_reason_keys"  # 62
_LONGER_ARRAY = "the_list_of_customer_record_identifiers_to_skip_during_archival_run"  # 67
_LONG_PATH = (
    "/organisations/{organisationId}/departments/{departmentId}/customer-records"
    "/archive-older-than-the-retention-policy-allows"
)

HOSTILE_NAMES_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Hostile Names API", "description": "Every name at its limit."},
    "servers": [{"url": "https://hostile.example/api/v2"}],
    "paths": {
        **{
            f"/{tag}": {
                "get": {
                    "operationId": "list" + tag.title().replace("-", ""),
                    "summary": f"List everything under {tag}",
                    "tags": [tag],
                }
            }
            for tag in _LONG_TAGS
        },
        _LONG_PATH: {
            "post": {
                "operationId": _LONG_OPERATION,
                "summary": "Archive every customer record older than the retention policy allows",
                "tags": [_LONG_TAGS[0]],
                "parameters": [
                    {
                        "name": "organisationId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                    {
                        "name": "departmentId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                    {"name": "dry-run", "in": "query", "schema": {"type": "boolean"}},
                    # names the generated handler or its annotations use themselves
                    {"name": "route", "in": "query", "schema": {"type": "string"}},
                    {"name": "str", "in": "query", "schema": {"type": "boolean"}},
                    {"name": "Any", "in": "query", "schema": {"type": "string"}},
                    {"name": _LONGER_STRING, "in": "query", "schema": {"type": "string"}},
                ],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": [_LONG_PARAM],
                                "properties": {
                                    _LONG_PARAM: {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "Which record sets to archive.",
                                    },
                                    "retention_override_days": {"type": "integer"},
                                    _LONGER_OBJECT: {
                                        "type": "object",
                                        "properties": {
                                            "days": {"type": "integer"},
                                            "reason": {"type": "string"},
                                        },
                                    },
                                    _LONGER_ARRAY: {"type": "array", "items": {"type": "string"}},
                                },
                            }
                        }
                    },
                },
            }
        },
    },
}


@pytest.mark.parametrize(
    ("auth", "approval"),
    [
        (AuthMode.NONE, None),
        (AuthMode.ENV_TOKEN, None),
        (AuthMode.PASSTHROUGH, None),
        (AuthMode.API_KEY, ApprovalMode.PENDING),
    ],
    ids=["none", "env-token", "passthrough", "api-key/pending"],
)
def test_hostile_names_stay_within_the_margin(
    tmp_path: Path, mypy_cache: Path, auth, approval
) -> None:
    """Fourteen tools modules, a 59-character gated operation id, parameters of 60 to 74
    characters, parameters named ``route``, ``str`` and ``Any`` and a path longer than a line:
    every generated file stays within 100 columns, is formatted the way ruff would format it,
    type-checks, and the project's own tests pass."""
    assert len(_LONG_OPERATION) >= 56 and len(_LONG_PARAM) == 60 and len(_LONG_TAGS) >= 14
    assert (len(_LONGER_STRING), len(_LONGER_OBJECT), len(_LONGER_ARRAY)) == (73, 62, 67)
    plan = mcpcast(
        HOSTILE_NAMES_SPEC, name="hostile", profile=SafetyProfile.FULL, auth=auth, approval=approval
    )
    gated = plan.gated_tools
    assert [t.name for t in gated] == [
        "archive_every_customer_record_older_than_the_retention_policy_al"
    ]
    assert _LONG_PARAM in gated[0].params and len(plan.tools) == 15
    assert {"route", "str", "Any", _LONGER_STRING, _LONGER_OBJECT, _LONGER_ARRAY} <= set(
        gated[0].params
    )
    out = tmp_path / "hostile-mcp"
    write_project(plan, out)
    module = out / "hostile_mcp" / "tools" / f"{tool_group(gated[0])}.py"
    src = module.read_text(encoding="utf-8")
    assert "        route: str | None = None," in src  # keeps its wire name: the local is _route
    assert "        str_: bool | None = None," in src and "        Any_: str | None = None," in src
    assert f"        {_LONGER_STRING[:56]}: str | None = None," in src
    assert f"        {_LONGER_OBJECT[:56]}: dict[str, Any] | None = None," in src
    assert f"        {_LONGER_ARRAY[:56]}: list[Any] | None = None," in src
    assert f'"{_LONGER_STRING}": (' in src  # the wire name travels in full
    for path in out.rglob("*.py"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            assert len(line) <= 100, f"{path.relative_to(out)}:{number} is {len(line)} columns"
    _assert_lints_clean(out, mypy_cache)
    result = _run_generated_suite(out, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "17 passed" in result.stdout  # listing + 15 tools + 1 approval test


ENCODED_PATHS_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Directory"},
    "servers": [{"url": "https://dir.example/v1"}],
    "paths": {
        "/users/{email}": {
            "get": {
                "operationId": "getUser",
                "parameters": [
                    {
                        "name": "email",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string", "format": "email"},
                    }
                ],
            }
        },
        "/reports/{at}": {
            "get": {
                "operationId": "getReport",
                "parameters": [
                    {
                        "name": "at",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string", "format": "date-time"},
                    }
                ],
            }
        },
        "/objects/{arn}": {
            "get": {
                "operationId": "getObject",
                "parameters": [
                    {
                        "name": "arn",
                        "in": "path",
                        "required": True,
                        "schema": {
                            "type": "string",
                            "example": "arn:aws:s3:::bucket/reports 2026/日本.csv",
                        },
                    }
                ],
            }
        },
    },
}


def test_generated_tests_pass_for_percent_encoded_path_examples(
    tmp_path: Path, mypy_cache: Path
) -> None:
    """``format: email`` / ``date-time`` examples and one with ``/``, ``:``, a space and non-ASCII:
    the generated tests compare the path as the upstream receives it (percent-encoded), so
    the suite is green out of the box instead of failing on httpx's decoded ``URL.path``."""
    plan = mcpcast(ENCODED_PATHS_SPEC, name="directory", auth=AuthMode.NONE)
    out = tmp_path / "directory-mcp"
    write_project(plan, out)
    tests = (out / "tests" / "test_tools.py").read_text(encoding="utf-8")
    assert 'assert path_of(upstream.last) == "/v1/users/ada%40example.com"' in tests
    assert 'assert path_of(upstream.last) == "/v1/reports/2026-01-15T09%3A30%3A00Z"' in tests
    assert (
        "/v1/objects/arn%3Aaws%3As3%3A%3A%3Abucket%2Freports%202026%2F%E6%97%A5%E6%9C%AC.csv"
        in (
            tests.replace('"\n        "', "")  # the long literal is split across lines
        )
    )
    assert "url.path" not in tests
    result = _run_generated_suite(out, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "4 passed" in result.stdout  # listing + 3 tools
    _assert_lints_clean(out, mypy_cache)


PLAIN_HTTP_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Intranet"},
    "servers": [{"url": "http://api.intranet.corp:8080"}],
    "paths": {
        "/ping": {"get": {"operationId": "ping"}},
        "/legacy": {
            "get": {
                "operationId": "legacyPing",
                "servers": [{"url": "http://legacy.example.test"}],
            }
        },
    },
}


@pytest.mark.parametrize(
    ("auth", "approval"),
    [
        (AuthMode.ENV_TOKEN, None),
        (AuthMode.API_KEY, ApprovalMode.PENDING),
    ],
    ids=["env-token", "api-key/pending"],
)
def test_plain_http_project_passes_its_own_tests(
    tmp_path: Path, mypy_cache: Path, auth, approval
) -> None:
    """A plain-http upstream (and an operation served from its own plain-http host) is refused a
    credential at runtime until ``MCPCAST_ALLOW_INSECURE_HTTP=1``; the generated conftest lifts
    that guard for the tests, so the suite is green and lint-clean rather than red."""
    plan = mcpcast(PLAIN_HTTP_SPEC, name="intranet", auth=auth, approval=approval)
    out = tmp_path / "intranet-mcp"
    write_project(plan, out)
    conftest = (out / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert 'monkeypatch.setattr("intranet_mcp.upstream.ALLOW_INSECURE_HTTP", True)' in conftest
    result = _run_generated_suite(out, tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "3 passed" in result.stdout  # listing + 2 tools
    _assert_lints_clean(out, mypy_cache)
