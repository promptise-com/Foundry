"""Tests for code generation (``promptise.mcpcast.emit``).

The strongest test here writes the generated ``server.py`` to a temp dir,
imports it, and drives it with ``TestClient`` against a mock upstream —
proving the generator emits *working software*, not just text.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import httpx
import pytest

from promptise.mcp.server import TestClient
from promptise.mcpcast import MCPcastPlan, load_generated_server, mcpcast, write_project
from promptise.mcpcast.emit import (
    SCAFFOLD_ONCE,
    package_name,
    python_identifier,
    python_type,
    render_project,
    render_readme,
    tool_group,
)
from promptise.mcpcast.schema import (
    ApprovalMode,
    AuthMode,
    MCPcastError,
    RiskClass,
    RouteParam,
    SafetyProfile,
)

SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Petstore API", "description": "Manage pets."},
    "servers": [{"url": "https://petstore.example/api/v3"}],
    "components": {
        "schemas": {
            "Pet": {
                "type": "object",
                "required": ["name"],
                "properties": {
                    "id": {"type": "integer"},
                    "name": {"type": "string", "description": "Pet name"},
                    "status": {"type": "string", "enum": ["available", "sold"]},
                },
            }
        }
    },
    "paths": {
        "/pet/{petId}": {
            "parameters": [
                {
                    "name": "petId",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "integer"},
                    "description": "ID of pet",
                }
            ],
            "get": {"operationId": "getPetById", "summary": "Find pet by ID", "tags": ["pet"]},
            "delete": {"operationId": "deletePet", "summary": "Deletes a pet", "tags": ["pet"]},
        },
        "/pet": {
            "post": {
                "operationId": "addPet",
                "summary": "Add a new pet",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/Pet"}}
                    },
                },
            }
        },
        "/pet/findByStatus": {
            "get": {
                "operationId": "findPetsByStatus",
                "summary": "Finds Pets by status",
                "parameters": [
                    {
                        "name": "status",
                        "in": "query",
                        "schema": {"type": "string", "enum": ["available", "pending", "sold"]},
                    },
                    {"name": "X-Trace", "in": "header", "schema": {"type": "string"}},
                ],
            }
        },
        "/user/{user-name}/from": {
            "get": {
                "operationId": "weird-op/name",
                "summary": "Awkward names",
                "parameters": [
                    {
                        "name": "user-name",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                    {"name": "from", "in": "query", "schema": {"type": "string"}},
                    {"name": "ctx", "in": "query", "schema": {"type": "boolean"}},
                ],
            }
        },
        "/upload": {
            "post": {
                "operationId": "uploadForm",
                "requestBody": {
                    "content": {
                        "application/x-www-form-urlencoded": {
                            "schema": {"type": "object", "properties": {"note": {"type": "string"}}}
                        }
                    }
                },
            }
        },
        "/legacy": {"get": {"operationId": "legacy", "deprecated": True}},
    },
}

DERIVED = {
    "server.py",
    "README.md",
    "tests/conftest.py",
    "tests/test_tools.py",
    "petstore_mcp/__init__.py",
    "petstore_mcp/__main__.py",
    "petstore_mcp/config.py",
    "petstore_mcp/upstream.py",
    "petstore_mcp/approval.py",
    "petstore_mcp/server.py",
    "petstore_mcp/tools/__init__.py",
}


def _import(path: Path):
    """Load a generated project through the framework helper (fresh package each time)."""
    return load_generated_server(path)


def render_server(plan) -> str:
    """Every generated Python file, concatenated — for text-level assertions."""
    files = render_project(plan)
    return "\n".join(files[k] for k in sorted(files) if k.endswith(".py"))


def _compile_all(plan) -> None:
    """Every generated Python file must compile on its own."""
    for path, src in render_project(plan).items():
        if path.endswith(".py"):
            compile(src, path, "exec")


def _tools_module(out: Path, plan, tool_name: str) -> Path:
    return out / package_name(plan.api.name) / "tools" / f"{tool_group(plan.tool(tool_name))}.py"


def _help_text(mod) -> str:
    """The generated command line's --help output."""
    import contextlib
    import io

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), pytest.raises(SystemExit):
        mod.main(["--help"])
    return buffer.getvalue()


def _generate(tmp_path: Path, **kw):
    plan = mcpcast(SPEC, name="petstore", **kw)
    out = tmp_path / "proj"
    files = write_project(plan, out)
    written = {str(f.relative_to(out)) for f in files}
    assert written >= DERIVED | SCAFFOLD_ONCE | {"mcpcast.plan.yaml"}
    assert all(
        w.startswith("petstore_mcp/tools/")
        for w in written - DERIVED - SCAFFOLD_ONCE - {"mcpcast.plan.yaml"}
    )
    return plan, out, _import(out / "server.py")


class _Upstream:
    """Records requests and answers with a fixed JSON body."""

    def __init__(self, status=200, body=None):
        self.requests: list[httpx.Request] = []
        self.status = status
        self.body = body if body is not None else {"id": 1, "name": "doggie"}

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body, request=request)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    @pytest.mark.parametrize(
        ("wire", "expected"),
        [
            ("petId", "petId"),
            ("user-id", "user_id"),
            ("from", "from_"),
            ("ctx", "ctx_"),
            ("1st", "p_1st"),
            ("", "param"),
            ("__x", "x"),
            ("validate", "validate_"),
            ("model_id", "model_id_"),
            ("ROUTES", "ROUTES_"),
            ("select_route", "select_route_"),
        ],
    )
    def test_python_identifier(self, wire, expected):
        assert python_identifier(wire) == expected

    def test_python_identifier_unique(self):
        taken: set[str] = set()
        assert python_identifier("a-b", taken) == "a_b"
        assert python_identifier("a_b", taken) == "a_b_2"

    @pytest.mark.parametrize(
        ("schema", "expected"),
        [
            ({"type": "string"}, "str"),
            ({"type": "integer"}, "int"),
            ({"type": "number"}, "float"),
            ({"type": "boolean"}, "bool"),
            ({"type": "array"}, "list[Any]"),
            ({"type": "object"}, "dict[str, Any]"),
            ({"properties": {}}, "dict[str, Any]"),
            ({"type": ["string", "null"]}, "str"),
            ({"type": ["string", "integer"]}, "Any"),
            ({}, "Any"),
        ],
    )
    def test_python_type(self, schema, expected):
        assert python_type(schema) == expected


# ---------------------------------------------------------------------------
# Generated server — imported and driven
# ---------------------------------------------------------------------------


class TestGeneratedServerRuns:
    @pytest.mark.asyncio
    async def test_read_only_exposes_only_reads(self, tmp_path):
        plan, out, mod = _generate(tmp_path, profile=SafetyProfile.READ_ONLY)
        names = {t.name for t in mod.server._tool_registry.list_all()}
        assert names == {"get_pet_by_id", "find_pets_by_status", "weird_op_name"}
        assert not plan.gated_tools
        assert "ApprovalGateMiddleware" not in (out / "petstore_mcp" / "approval.py").read_text()

    @pytest.mark.asyncio
    async def test_full_profile_drives_every_tool_through_testclient(self, tmp_path):
        plan, out, mod = _generate(tmp_path, profile=SafetyProfile.FULL)
        upstream = _Upstream()
        async with upstream.client() as http:
            server = mod.build_server(approval_handler=lambda request: True, http_client=http)
            client = TestClient(server, meta={"authorization": "Bearer abc"})

            tools = {t.name: t for t in await client.list_tools()}
            assert set(tools) == {
                "get_pet_by_id",
                "delete_pet",
                "add_pet",
                "find_pets_by_status",
                "weird_op_name",
                "upload_form",
            }
            assert tools["get_pet_by_id"].inputSchema["required"] == ["petId"]
            assert "Parameters:" in tools["add_pet"].description
            assert "Example:" in tools["add_pet"].description

            (r,) = await client.call_tool("get_pet_by_id", {"petId": 7})
            assert json.loads(r.text) == {"id": 1, "name": "doggie"}
            (r,) = await client.call_tool("delete_pet", {"petId": 7})
            assert json.loads(r.text) == {"id": 1, "name": "doggie"}
            (r,) = await client.call_tool("add_pet", {"name": "rex", "status": "sold"})
            assert "error" not in r.text
            (r,) = await client.call_tool(
                "weird_op_name", {"user_name": "a/b", "from_": "x", "ctx_": True}
            )
            (r,) = await client.call_tool("find_pets_by_status", {})
            (r,) = await client.call_tool("upload_form", {"note": "hi"})

        reqs = [
            (q.method, str(q.url), q.headers.get("authorization"), q.content)
            for q in upstream.requests
        ]
        assert reqs[0] == ("GET", "https://petstore.example/api/v3/pet/7", "Bearer abc", b"")
        assert reqs[1][:2] == ("DELETE", "https://petstore.example/api/v3/pet/7")
        assert reqs[2][:2] == ("POST", "https://petstore.example/api/v3/pet")
        assert json.loads(reqs[2][3]) == {
            "name": "rex",
            "status": "sold",
        }  # None-valued optionals omitted
        assert reqs[3][1] == "https://petstore.example/api/v3/user/a%2Fb/from?from=x&ctx=true"
        assert reqs[4][1] == "https://petstore.example/api/v3/pet/findByStatus"
        assert (
            upstream.requests[5]
            .headers["content-type"]
            .startswith("application/x-www-form-urlencoded")
        )
        assert upstream.requests[5].content == b"note=hi"

    @pytest.mark.asyncio
    async def test_gated_tool_denied_without_live_approver(self, tmp_path):
        _plan, _out, mod = _generate(tmp_path, profile=SafetyProfile.FULL)
        upstream = _Upstream()
        async with upstream.client() as http:
            server = mod.build_server(http_client=http)  # default ElicitationApprover
            client = TestClient(server, meta={"authorization": "Bearer abc"})
            (r,) = await client.call_tool("delete_pet", {"petId": 7})
            assert json.loads(r.text)["error"]["code"] == "APPROVAL_DENIED"
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 7})  # reads are not gated
            assert "error" not in r.text
        assert [q.method for q in upstream.requests] == ["GET"]  # the DELETE never reached upstream

    @pytest.mark.asyncio
    async def test_passthrough_requires_authorization_header(self, tmp_path):
        _plan, _out, mod = _generate(tmp_path)
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1})
            assert json.loads(r.text)["error"]["code"] == "UPSTREAM_AUTH_MISSING"
        assert upstream.requests == []

    @pytest.mark.asyncio
    async def test_none_auth_sends_no_credentials(self, tmp_path):
        _plan, _out, mod = _generate(tmp_path, auth=AuthMode.NONE)
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1})
            assert "error" not in r.text
        assert "authorization" not in upstream.requests[0].headers

    @pytest.mark.parametrize("auth", [AuthMode.NONE, AuthMode.ENV_TOKEN])
    def test_unauthenticated_modes_refuse_public_bind(self, tmp_path, monkeypatch, auth):
        """`none` sends nothing upstream, `env-token` sends the operator's secret: neither
        has MCP-level authentication, so both bind loopback only unless told otherwise."""
        monkeypatch.delenv("MCPCAST_PUBLIC", raising=False)
        _plan, _out, mod = _generate(tmp_path, auth=auth)
        called = {}
        # Patched first: if the guard ever regressed, main() must not reach a real bind.
        monkeypatch.setattr(mod.server, "run", lambda **kw: called.update(kw))
        with pytest.raises(SystemExit) as exc:
            mod.main(["--transport", "http", "--host", "0.0.0.0"])
        assert exc.value.code == 2 and called == {}
        mod.main(["--transport", "http", "--host", "127.0.0.1", "--port", "9999"])
        assert called == {"transport": "http", "host": "127.0.0.1", "port": 9999}
        assert mod.server.public is False
        # the explicit opt-in: --public on the command line…
        mod.main(["--transport", "http", "--host", "0.0.0.0", "--public"])
        assert called["host"] == "0.0.0.0" and mod.server.public is True
        help_text = _help_text(mod)
        assert "--public" in help_text and "authenticating gateway" in help_text
        assert "MCPCAST_PUBLIC=1" in help_text

    @pytest.mark.parametrize("auth", [AuthMode.NONE, AuthMode.ENV_TOKEN])
    def test_public_env_var_is_honoured_by_the_command_line(self, tmp_path, monkeypatch, auth):
        monkeypatch.setenv("MCPCAST_PUBLIC", "1")
        _plan, _out, mod = _generate(tmp_path, auth=auth)
        called = {}
        monkeypatch.setattr(mod.server, "run", lambda **kw: called.update(kw))
        mod.main(["--transport", "http", "--host", "0.0.0.0"])
        assert called["host"] == "0.0.0.0"

    def test_identified_modes_have_no_public_flag(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MCPCAST_CLIENT_KEYS", json.dumps({"k": {"client_id": "a"}}))
        _plan, _out, mod = _generate(tmp_path, auth=AuthMode.API_KEY)
        called = {}
        monkeypatch.setattr(mod.server, "run", lambda **kw: called.update(kw))
        mod.main(["--transport", "http", "--host", "0.0.0.0"])
        assert called["host"] == "0.0.0.0"
        assert "--public" not in _help_text(mod)

    @pytest.mark.parametrize("auth", [AuthMode.NONE, AuthMode.ENV_TOKEN])
    @pytest.mark.asyncio
    async def test_bind_guard_lives_on_the_server_too(self, tmp_path, monkeypatch, auth):
        """`promptise serve` / `--serve` call the server directly — the guard must hold there."""
        import asyncio

        monkeypatch.delenv("MCPCAST_PUBLIC", raising=False)
        _plan, _out, mod = _generate(tmp_path, auth=auth)
        with pytest.raises(RuntimeError, match="non-loopback"):
            await mod.server.run_async("http", host="0.0.0.0", port=1)
        with pytest.raises(RuntimeError, match="non-loopback"):
            await mod.server.run_async("sse", host="[::]", port=1)

        async def stops_before_binding(**kwargs):
            # loopback aliases and the opt-ins are fine (cancelled before a real bind)
            task = asyncio.ensure_future(mod.server.run_async("http", port=0, **kwargs))
            await asyncio.sleep(0)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        await stops_before_binding(host="127.0.0.2")
        mod.server.public = True
        await stops_before_binding(host="0.0.0.0")
        mod.server.public = False
        monkeypatch.setenv("MCPCAST_PUBLIC", "1")
        mod = _import(tmp_path / "proj" / "server.py")  # config reads the environment at import
        await stops_before_binding(host="0.0.0.0")

    @pytest.mark.asyncio
    async def test_api_key_mode_maps_tenant_to_upstream_token(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "MCPCAST_CLIENT_KEYS",
            json.dumps(
                {
                    "sk-acme-agent": {"client_id": "agent", "tenant_id": "acme"},
                    "sk-globex": {"client_id": "agent", "tenant_id": "globex"},
                }
            ),
        )
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKENS", json.dumps({"acme": "Bearer acme-secret"}))
        plan, _out, mod = _generate(tmp_path, auth=AuthMode.API_KEY, profile=SafetyProfile.FULL)
        assert plan.api.approval_mode is ApprovalMode.PENDING
        names = {t.name for t in mod.server._tool_registry.list_all()}
        assert {"approvals_list", "approvals_decide"} <= names  # four-eyes admin tools
        upstream = _Upstream()
        async with upstream.client() as http:
            server = mod.build_server(http_client=http)
            client = TestClient(server)
            (r,) = await client.call_tool(
                "get_pet_by_id", {"petId": 1}, headers={"x-api-key": "sk-acme-agent"}
            )
            assert "error" not in r.text
            (r,) = await client.call_tool(
                "get_pet_by_id", {"petId": 1}, headers={"x-api-key": "sk-globex"}
            )
            assert (
                json.loads(r.text)["error"]["code"] == "UPSTREAM_AUTH_MISSING"
            )  # no token for globex
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1})
            assert "error" in r.text  # unauthenticated → rejected before upstream
        assert [q.headers.get("authorization") for q in upstream.requests] == ["Bearer acme-secret"]

    @pytest.mark.asyncio
    async def test_upstream_errors_are_structured(self, tmp_path):
        _plan, _out, mod = _generate(tmp_path, auth=AuthMode.NONE)
        upstream = _Upstream(status=503, body={"msg": "down"})
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1})
            err = json.loads(r.text)["error"]
            assert err["code"] == "UPSTREAM_ERROR" and err["retryable"] is True
            assert err["details"] == {"status": 503}

        async def unreachable(request):
            raise httpx.ConnectError("refused by host-with-details", request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(unreachable)) as http:
            client = TestClient(mod.build_server(http_client=http))
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1})
            err = json.loads(r.text)["error"]
            assert err["code"] == "UPSTREAM_UNREACHABLE" and err["retryable"] is True
            # the exception class, never its text (which quotes the rejected header value)
            assert err["message"] == "GET /pet/{petId} failed: ConnectError"

    @pytest.mark.asyncio
    async def test_non_json_and_empty_responses(self, tmp_path):
        _plan, _out, mod = _generate(tmp_path, auth=AuthMode.NONE)

        async def handler(request):
            if "findByStatus" in str(request.url):
                return httpx.Response(204, request=request)
            return httpx.Response(
                200, text="plain", headers={"content-type": "text/plain"}, request=request
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = TestClient(mod.build_server(http_client=http))
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1})
            assert json.loads(r.text) == {"status": 200, "body": "plain"}
            (r,) = await client.call_tool("find_pets_by_status", {})
            assert json.loads(r.text) == {"status": 204}


# ---------------------------------------------------------------------------
# Plan is the source of truth
# ---------------------------------------------------------------------------


class TestRegenerateFromEditedPlan:
    @pytest.mark.asyncio
    async def test_edits_flow_into_server(self, tmp_path):
        plan, out, _mod = _generate(tmp_path, profile=SafetyProfile.STANDARD)
        text = (out / "mcpcast.plan.yaml").read_text()
        text = text.replace("name: find_pets_by_status", "name: list_pets").replace(
            "description: Finds Pets by status",
            "description: List pets, optionally filtered by status.",
        )
        edited = MCPcastPlan.from_yaml(text)
        # Hide `status` behind a default and collapse get+list into one tool with two routes.
        edited.tool("list_pets").params["status"].hidden = True
        edited.tool("list_pets").params["status"].default = "available"
        write_project(edited, out)
        mod = _import(out / "server.py")
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(
                mod.build_server(http_client=http), meta={"authorization": "Bearer t"}
            )
            tools = {t.name: t for t in await client.list_tools()}
            assert "list_pets" in tools and "find_pets_by_status" not in tools
            assert tools["list_pets"].inputSchema.get("properties", {}) == {}  # hidden
            assert tools["list_pets"].description.startswith("List pets, optionally")
            await client.call_tool("list_pets", {})
        assert str(upstream.requests[0].url).endswith("/pet/findByStatus?status=available")

    @pytest.mark.asyncio
    async def test_multi_route_dispatch_by_required_params(self, tmp_path):
        plan, out, _ = _generate(tmp_path, profile=SafetyProfile.READ_ONLY)
        get = plan.tool("get_pet_by_id")
        find = plan.tool("find_pets_by_status")
        merged = get.model_copy(
            update={
                "name": "find_pet",
                "routes": [get.routes[0], find.routes[0]],
                "params": {**get.params, **find.params},
                "example": None,
            }
        )
        merged.params["petId"].required = False
        plan.tools = [merged, plan.tool("weird_op_name")]
        write_project(plan, out)
        mod = _import(out / "server.py")
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(
                mod.build_server(http_client=http), meta={"authorization": "Bearer t"}
            )
            await client.call_tool("find_pet", {"petId": 3})
            await client.call_tool("find_pet", {"status": "sold"})
        urls = [str(q.url) for q in upstream.requests]
        assert urls == [
            "https://petstore.example/api/v3/pet/3",
            "https://petstore.example/api/v3/pet/findByStatus?status=sold",
        ]


# ---------------------------------------------------------------------------
# Text artefacts
# ---------------------------------------------------------------------------


class TestRenderedText:
    def test_server_source_is_readable(self):
        plan = mcpcast(SPEC, name="petstore", profile=SafetyProfile.FULL)
        files = render_project(plan)
        for path, src in files.items():
            if path.endswith(".py"):
                compile(src, path, "exec")
        init = files["petstore_mcp/__init__.py"]
        assert init.startswith(
            '"""petstore — an MCP server for the API, as an installable package.'
        )
        assert "Petstore API: Manage pets." in init
        assert all(
            "promptise mcpcast mcpcast.plan.yaml" in files[p]
            for p in files
            if p.endswith(".py") and not p.startswith("tests/")
        )
        assert files["server.py"].splitlines()[0].startswith('"""Run the petstore MCP server')
        src = render_server(plan)
        assert "requires_approval=True" in src
        assert "read_only_hint=True" in src and "destructive_hint=True" in src
        assert "def build_server(" in src and 'if __name__ == "__main__":' in src

    def test_readme_snippets_and_tables(self):
        plan = mcpcast(SPEC, name="petstore", profile=SafetyProfile.FULL, auth=AuthMode.API_KEY)
        md = render_readme(plan)
        assert "claude mcp add petstore" in md
        assert (
            '"mcpServers"' in md and "claude_desktop_config.json" in md and ".cursor/mcp.json" in md
        )
        assert "| `delete_pet` | destructive | **required** |" in md
        assert "| `legacy` | deprecated in spec |" in md
        assert "MCPCAST_CLIENT_KEYS" in md and "MCPCAST_UPSTREAM_TOKENS" in md
        assert "`pending`" in md and "approvals_list" in md
        md_pt = render_readme(mcpcast(SPEC, profile=SafetyProfile.STANDARD))
        assert "`elicitation`" in md_pt and "`passthrough`" in md_pt
        md_ro = render_readme(mcpcast(SPEC))
        assert "no approval gate is installed" in md_ro

    def test_write_project_overwrites(self, tmp_path):
        plan = mcpcast(SPEC, name="petstore")
        out = tmp_path / "p"
        write_project(plan, out)
        (out / "server.py").write_text("garbage")
        module = _tools_module(out, plan, "get_pet_by_id")
        generated = render_project(plan)[str(module.relative_to(out))]
        stale = module.with_name("stale.py")
        stale.write_text("# no header: not ours\n")
        generated_stale = module.with_name("old_resource.py")
        generated_stale.write_text(
            generated.replace(f"Tools for {module.stem} —", "Tools for old_resource —", 1)
        )
        copied = module.with_name("copied_pets.py")
        copied.write_text(generated)  # our header, but it names pets.py: a copy the developer kept
        mine = module.with_name("my_custom_tools.py")
        mine.write_text("# hand-written helpers next to the generated modules\n")
        (out / "pyproject.toml").write_text("# mine\n")
        write_project(plan, out)
        assert "garbage" not in (out / "server.py").read_text()
        assert not generated_stale.exists()  # generated for a group the plan lacks: removed
        assert stale.exists() and mine.exists() and copied.exists()  # everything else stays
        assert (out / "pyproject.toml").read_text() == "# mine\n"  # scaffold: written once
        assert MCPcastPlan.load(out / "mcpcast.plan.yaml") == plan
        # A rendered path holding a file without our header is never overwritten silently.
        module.write_text("garbage")
        (out / "server.py").write_text("garbage")
        with pytest.raises(MCPcastError, match="exists and was not generated by mcpcast"):
            write_project(plan, out)
        assert module.read_text() == "garbage" and (out / "server.py").read_text() == "garbage"
        write_project(plan, out, force=True)
        assert (
            "garbage" not in module.read_text() and "garbage" not in (out / "server.py").read_text()
        )


class TestWireAliases:
    @pytest.mark.asyncio
    async def test_colliding_names_reach_the_right_wire_location(self, tmp_path):
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Users"},
            "servers": [{"url": "https://u.example"}],
            "paths": {
                "/user/{username}": {
                    "put": {
                        "operationId": "updateUser",
                        "parameters": [
                            {
                                "name": "username",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            },
                            {"name": "username", "in": "query", "schema": {"type": "string"}},
                        ],
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "username": {"type": "string"},
                                            "email": {"type": "string"},
                                        },
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }
        plan = mcpcast(spec, name="users", profile=SafetyProfile.STANDARD, auth=AuthMode.NONE)
        out = tmp_path / "u"
        write_project(plan, out)
        assert "aliases={" in _tools_module(out, plan, "update_user").read_text()
        mod = _import(out / "server.py")
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(approval_handler=lambda r: True, http_client=http))
            tools = {t.name: t for t in await client.list_tools()}
            assert set(tools["update_user"].inputSchema["properties"]) == {
                "username",
                "query_username",
                "body_username",
                "email",
            }
            (r,) = await client.call_tool(
                "update_user",
                {
                    "username": "ada",
                    "query_username": "q-ada",
                    "body_username": "b-ada",
                    "email": "a@x",
                },
            )
            assert "error" not in r.text
        (req,) = upstream.requests
        assert str(req.url) == "https://u.example/user/ada?username=q-ada"
        assert json.loads(req.content) == {"username": "b-ada", "email": "a@x"}


class TestEnvTokenAuth:
    @pytest.mark.asyncio
    async def test_env_token_is_sent_and_missing_is_structured(self, tmp_path, monkeypatch):
        plan, out, mod = _generate(tmp_path, auth=AuthMode.ENV_TOKEN)
        assert plan.api.approval_mode is ApprovalMode.ELICITATION
        assert "MCPCAST_UPSTREAM_TOKEN" in (out / "README.md").read_text()
        upstream = _Upstream()
        monkeypatch.delenv("MCPCAST_UPSTREAM_TOKEN", raising=False)
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))  # no headers at all (stdio)
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1})
            err = json.loads(r.text)["error"]
            assert (
                err["code"] == "UPSTREAM_AUTH_MISSING"
                and "MCPCAST_UPSTREAM_TOKEN" in err["message"]
            )
            monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", "Bearer personal-token")
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1})
            assert "error" not in r.text
        assert [q.headers.get("authorization") for q in upstream.requests] == [
            "Bearer personal-token"
        ]

    @pytest.mark.asyncio
    async def test_passthrough_error_points_to_env_token(self, tmp_path):
        _plan, _out, mod = _generate(tmp_path)
        async with _Upstream().client() as http:
            (r,) = await TestClient(mod.build_server(http_client=http)).call_tool(
                "get_pet_by_id", {"petId": 1}
            )
        assert "--auth env-token" in json.loads(r.text)["error"]["message"]


class TestGeneratedCodeIsInjectionSafe:
    HOSTILE_TITLE = (
        'Evil""" ; import pathlib; pathlib.Path("/tmp/mcpcast-pwned").write_text("x"); x = """'
    )

    def test_spec_text_cannot_become_code(self, tmp_path):
        spec = {
            **SPEC,
            "info": {
                "title": self.HOSTILE_TITLE,
                "description": self.HOSTILE_TITLE + "\nsecond \\U0001 line",
            },
        }
        plan = mcpcast(spec, name="petstore")
        out = tmp_path / "evil"
        write_project(plan, out)
        marker = Path("/tmp/mcpcast-pwned")
        marker.unlink(missing_ok=True)
        _import(out / "server.py")
        assert not marker.exists()
        import ast

        for path in out.rglob("*.py"):
            tree = ast.parse(path.read_text(), str(path))
            # The hostile text may only ever sit inside a string literal: no call
            # to write_text and no pathlib import outside the launcher, which
            # legitimately uses pathlib to find the package.
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    assert node.func.attr != "write_text", path
                if isinstance(node, ast.Import | ast.ImportFrom) and path.name != "server.py":
                    assert "pathlib" not in ast.dump(node), path

    def test_windows_path_and_quotes_compile(self, tmp_path):
        spec = {
            **SPEC,
            "paths": {
                "/q": {"get": {"operationId": "q", "summary": 'Returns the user\'s "profile"'}}
            },
        }
        plan = mcpcast(spec, name="petstore", base_url="https://x")
        plan = plan.model_copy(
            update={
                "api": plan.api.model_copy(update={"spec_source": "C:\\Users\\nick\\openapi.yaml"})
            }
        )
        _compile_all(plan)
        write_project(plan, tmp_path / "w")
        mod = _import(tmp_path / "w" / "server.py")
        assert mod.server._tool_registry.get("q").description.startswith("Returns the user's")


class TestUpstreamEncoding:
    @pytest.mark.asyncio
    async def test_path_segments_must_be_real_segments(self, tmp_path):
        _plan, _out, mod = _generate(tmp_path, auth=AuthMode.NONE)
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))
            for bad in ("", ".", "..", "a\nb"):
                (r,) = await client.call_tool("weird_op_name", {"user_name": bad})
                assert json.loads(r.text)["error"]["code"] == "VALIDATION_ERROR", bad
            (r,) = await client.call_tool("weird_op_name", {"user_name": "a/b"})
            assert "error" not in r.text
        assert [str(q.url) for q in upstream.requests] == [
            "https://petstore.example/api/v3/user/a%2Fb/from"
        ]

    @pytest.mark.asyncio
    async def test_form_and_deep_object_encoding(self, tmp_path):
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Pay"},
            "servers": [{"url": "https://pay.example"}],
            "paths": {
                "/checkout": {
                    "post": {
                        "operationId": "checkout",
                        "requestBody": {
                            "content": {
                                "application/x-www-form-urlencoded": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "line_items": {
                                                "type": "array",
                                                "items": {"type": "object"},
                                            },
                                            "metadata": {"type": "object"},
                                            "live": {"type": "boolean"},
                                        },
                                    }
                                }
                            }
                        },
                    }
                },
                "/search": {
                    "get": {
                        "operationId": "search",
                        "parameters": [
                            {"name": "filter", "in": "query", "schema": {"type": "object"}},
                            {
                                "name": "ids",
                                "in": "query",
                                "schema": {"type": "array", "items": {"type": "integer"}},
                            },
                            {"name": "flag", "in": "query", "schema": {"type": "boolean"}},
                            {
                                "name": "tags",
                                "in": "query",
                                "schema": {"type": "array", "items": {"type": "object"}},
                            },
                            {
                                "name": "ranges",
                                "in": "query",
                                "schema": {
                                    "type": "array",
                                    "items": {"type": "array", "items": {"type": "integer"}},
                                },
                            },
                        ],
                    }
                },
            },
        }
        plan = mcpcast(spec, name="pay", profile=SafetyProfile.FULL, auth=AuthMode.NONE)
        assert plan.tool("checkout").risk is RiskClass.FINANCIAL
        out = tmp_path / "pay"
        write_project(plan, out)
        mod = _import(out / "server.py")
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(approval_handler=lambda r: True, http_client=http))
            await client.call_tool(
                "checkout",
                {
                    "line_items": [{"price": "p1", "quantity": 2}],
                    "metadata": {"order": "42"},
                    "live": False,
                },
            )
            await client.call_tool(
                "search",
                {
                    "filter": {"status": "open"},
                    "ids": [1, 2],
                    "flag": True,
                    # every element of an array of objects (or arrays) keeps its own index
                    "tags": [{"a": 1}, {"a": 2}],
                    "ranges": [[1, 2], [3]],
                },
            )
        form, query = upstream.requests
        assert (
            form.content.decode()
            == "line_items%5B0%5D%5Bprice%5D=p1&line_items%5B0%5D%5Bquantity%5D=2&metadata%5Border%5D=42&live=false"
        )
        assert str(query.url) == (
            "https://pay.example/search?filter%5Bstatus%5D=open&ids=1&ids=2&flag=true"
            "&tags%5B0%5D%5Ba%5D=1&tags%5B1%5D%5Ba%5D=2"
            "&ranges%5B0%5D%5B0%5D=1&ranges%5B0%5D%5B1%5D=2&ranges%5B1%5D%5B0%5D=3"
        )

    @pytest.mark.asyncio
    async def test_get_routes_send_the_body_they_declare(self, tmp_path):
        """A search-style GET with a request body is not silently sent empty (only HEAD is)."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Search"},
            "servers": [{"url": "https://g.example.com"}],
            "paths": {
                "/search": {
                    "get": {
                        "operationId": "searchThings",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["query"],
                                        "properties": {
                                            "query": {"type": "string"},
                                            "limit": {"type": "integer"},
                                        },
                                    }
                                }
                            },
                        },
                    }
                },
                "/lookup": {
                    "get": {
                        "operationId": "lookup",
                        "requestBody": {
                            "content": {
                                "application/x-www-form-urlencoded": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"q": {"type": "string"}},
                                    }
                                }
                            }
                        },
                    }
                },
                "/plain": {"get": {"operationId": "plain"}},
            },
        }
        plan = mcpcast(spec, name="search", auth=AuthMode.NONE)
        assert plan.tool("search_things").routes[0].params["query"].location == "body"
        out = tmp_path / "s"
        write_project(plan, out)
        mod = _import(out / "server.py")
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))
            for tool, args in (
                ("search_things", {"query": "dune", "limit": 5}),
                ("lookup", {"q": "dune"}),
                ("plain", {}),
            ):
                (r,) = await client.call_tool(tool, args)
                assert "error" not in r.text, tool
        search, lookup, plain = upstream.requests
        assert [q.method for q in upstream.requests] == ["GET", "GET", "GET"]
        assert json.loads(search.content) == {"query": "dune", "limit": 5}
        assert search.headers["content-type"] == "application/json"
        assert lookup.content == b"q=dune"
        assert lookup.headers["content-type"] == "application/x-www-form-urlencoded"
        assert plain.content == b"" and "content-type" not in plain.headers

    @pytest.mark.asyncio
    async def test_route_level_base_url(self, tmp_path):
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Files"},
            "servers": [{"url": "https://api.example"}],
            "paths": {
                "/files": {
                    "post": {
                        "operationId": "upload",
                        "servers": [{"url": "https://files.example/"}],
                    }
                },
                "/items": {"get": {"operationId": "items"}},
            },
        }
        plan = mcpcast(spec, name="files", profile=SafetyProfile.STANDARD, auth=AuthMode.NONE)
        assert plan.tool("upload").routes[0].base_url == "https://files.example"
        assert plan.tool("items").routes[0].base_url is None
        out = tmp_path / "f"
        write_project(plan, out)
        mod = _import(out / "server.py")
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(approval_handler=lambda r: True, http_client=http))
            await client.call_tool("upload", {})
            await client.call_tool("items", {})
        assert [str(q.url) for q in upstream.requests] == [
            "https://files.example/files",
            "https://api.example/items",
        ]


class TestDescriptionsMatchSchema:
    def test_parameter_notes_and_example_use_python_identifiers(self):
        plan = mcpcast(SPEC, name="petstore")
        tool = plan.tool("weird_op_name")
        tool.params["user-name"].description = "The user"
        tool.example = {"user-name": "ada", "from": "x"}
        src = render_server(plan)
        assert "user_name (string, required): The user" in src
        assert "from_ (string)" in src and "ctx_ (boolean)" in src
        assert 'Example: {\\"user_name\\": \\"ada\\", \\"from_\\": \\"x\\"}' in src.replace(
            '"', '\\"'
        ) or 'Example: {"user_name": "ada", "from_": "x"}' in src.replace("\\", "")

    def test_hidden_defaults_and_alternatives_are_spelled_out(self):
        plan = mcpcast(SPEC, name="petstore", profile=SafetyProfile.FULL)
        delete = plan.tool("delete_pet")
        delete.params["cascade"] = __import__(
            "promptise.mcpcast.schema", fromlist=["ParamPlan"]
        ).ParamPlan(hidden=True, default=True, json_schema={"type": "boolean"})
        delete.routes[0].params["cascade"] = __import__(
            "promptise.mcpcast.schema", fromlist=["RouteParam"]
        ).RouteParam(location="query")
        get, find = plan.tool("get_pet_by_id"), plan.tool("find_pets_by_status")
        merged = get.model_copy(
            update={
                "name": "find_pet",
                "routes": [get.routes[0], find.routes[0]],
                "params": {**get.params, **find.params},
                "example": None,
            }
        )
        merged.params["petId"].required = False
        plan.tools = [merged, delete]
        src = render_server(plan)
        assert "Always sends: cascade=true" in src
        assert "Provide one of: petId | (no parameters)" in src
        md = render_readme(plan)
        assert "`delete_pet` always sends `cascade=true`" in md

    def test_yaml_dates_become_json_native(self):
        spec = {
            **SPEC,
            "paths": {
                "/d": {
                    "get": {
                        "operationId": "d",
                        "parameters": [
                            {
                                "name": "day",
                                "in": "query",
                                "required": True,
                                "schema": {
                                    "type": "string",
                                    "format": "date",
                                    "example": __import__("datetime").date(2026, 1, 15),
                                },
                            }
                        ],
                    }
                }
            },
        }
        plan = mcpcast(spec, name="petstore")
        assert plan.tool("d").example == {"day": "2026-01-15"}
        _compile_all(plan)
        MCPcastPlan.from_yaml(plan.to_yaml())

    def test_write_plan_false_leaves_plan_untouched(self, tmp_path):
        plan = mcpcast(SPEC, name="petstore")
        out = tmp_path / "p"
        write_project(plan, out)
        (out / "mcpcast.plan.yaml").write_text(
            "# my comment\n" + (out / "mcpcast.plan.yaml").read_text()
        )
        files = write_project(plan, out, write_plan=False)
        assert "mcpcast.plan.yaml" not in {f.name for f in files}
        assert (out / "mcpcast.plan.yaml").read_text().startswith("# my comment")


class TestTenantScopedApprovals:
    @pytest.mark.asyncio
    async def test_reviewers_only_see_and_decide_their_tenant(self, tmp_path, monkeypatch):
        import asyncio

        monkeypatch.setenv(
            "MCPCAST_CLIENT_KEYS",
            json.dumps(
                {
                    "sk-acme-agent": {"client_id": "acme-agent", "tenant_id": "acme"},
                    "sk-acme-rev": {
                        "client_id": "dana",
                        "tenant_id": "acme",
                        "roles": ["approver"],
                    },
                    "sk-globex-rev": {
                        "client_id": "gus",
                        "tenant_id": "globex",
                        "roles": ["approver"],
                    },
                }
            ),
        )
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKENS", json.dumps({"acme": "Bearer acme"}))
        monkeypatch.setenv("MCPCAST_APPROVAL_TIMEOUT", "5")
        _plan, _out, mod = _generate(tmp_path, auth=AuthMode.API_KEY, profile=SafetyProfile.FULL)
        upstream = _Upstream()
        async with upstream.client() as http:
            server = mod.build_server(http_client=http)
            client = TestClient(server)
            call = asyncio.ensure_future(
                client.call_tool("delete_pet", {"petId": 7}, headers={"x-api-key": "sk-acme-agent"})
            )
            await asyncio.sleep(0.05)
            (r,) = await client.call_tool(
                "approvals_list", {}, headers={"x-api-key": "sk-globex-rev"}
            )
            assert json.loads(r.text) == []  # other tenant sees nothing
            (r,) = await client.call_tool(
                "approvals_list", {}, headers={"x-api-key": "sk-acme-rev"}
            )
            (pending,) = json.loads(r.text)
            assert pending["tool"] == "delete_pet" and pending["tenant_id"] == "acme"
            (r,) = await client.call_tool(
                "approvals_decide",
                {"request_id": pending["request_id"], "approve": True},
                headers={"x-api-key": "sk-globex-rev"},
            )
            assert (
                json.loads(r.text)["error"]["code"] == "NOT_FOUND"
            )  # cannot decide across tenants
            (r,) = await client.call_tool(
                "approvals_decide",
                {"request_id": pending["request_id"], "approve": True},
                headers={"x-api-key": "sk-acme-rev"},
            )
            assert json.loads(r.text)["resolved"] is True
            (result,) = await call
            assert "error" not in result.text
        assert [q.method for q in upstream.requests] == ["DELETE"]


class TestPendingCapacityPerClient:
    @pytest.mark.asyncio
    async def test_one_client_cannot_fill_the_queue_for_everyone(self, tmp_path, monkeypatch):
        """One tenant's agent looping on a gated tool must not get other tenants denied."""
        import asyncio

        monkeypatch.setenv(
            "MCPCAST_CLIENT_KEYS",
            json.dumps(
                {
                    "sk-a": {"client_id": "agent-a", "tenant_id": "a"},
                    "sk-b": {"client_id": "agent-b", "tenant_id": "b"},
                }
            ),
        )
        monkeypatch.setenv(
            "MCPCAST_UPSTREAM_TOKENS", json.dumps({"a": "Bearer a", "b": "Bearer b"})
        )
        monkeypatch.setenv("MCPCAST_MAX_PENDING_PER_CLIENT", "2")
        monkeypatch.setenv("MCPCAST_APPROVAL_TIMEOUT", "5")
        _plan, out, mod = _generate(tmp_path, auth=AuthMode.API_KEY, profile=SafetyProfile.FULL)
        assert (
            "max_pending_per_client=MAX_PENDING_PER_CLIENT"
            in (out / "petstore_mcp" / "approval.py").read_text()
        )
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))
            waiting = [
                asyncio.ensure_future(
                    client.call_tool("delete_pet", {"petId": i}, headers={"x-api-key": "sk-a"})
                )
                for i in range(2)
            ]
            await asyncio.sleep(0.05)
            (r,) = await client.call_tool("delete_pet", {"petId": 9}, headers={"x-api-key": "sk-a"})
            err = json.loads(r.text)["error"]
            assert err["code"] == "APPROVAL_DENIED" and "per-client" in err["message"]
            other = asyncio.ensure_future(
                client.call_tool("delete_pet", {"petId": 1}, headers={"x-api-key": "sk-b"})
            )
            await asyncio.sleep(0.05)
            assert not other.done()  # tenant b's call is waiting for its reviewer, not denied
            for future in [*waiting, other]:
                future.cancel()
            await asyncio.gather(*waiting, other, return_exceptions=True)
        assert upstream.requests == []


class TestPendingCapacityPerTenant:
    @pytest.mark.asyncio
    async def test_a_tenant_with_many_keys_cannot_fill_the_queue(self, tmp_path, monkeypatch):
        """The per-client cap counts API keys; a tenant holding several keys is bounded too."""
        import asyncio

        monkeypatch.setenv(
            "MCPCAST_CLIENT_KEYS",
            json.dumps(
                {
                    "sk-a1": {"client_id": "agent-a1", "tenant_id": "a"},
                    "sk-a2": {"client_id": "agent-a2", "tenant_id": "a"},
                    "sk-a3": {"client_id": "agent-a3", "tenant_id": "a"},
                    "sk-b": {"client_id": "agent-b", "tenant_id": "b"},
                }
            ),
        )
        monkeypatch.setenv(
            "MCPCAST_UPSTREAM_TOKENS", json.dumps({"a": "Bearer a", "b": "Bearer b"})
        )
        monkeypatch.setenv("MCPCAST_MAX_PENDING", "10")
        monkeypatch.setenv("MCPCAST_MAX_PENDING_PER_TENANT", "2")
        monkeypatch.setenv("MCPCAST_MAX_PENDING_PER_CLIENT", "1")
        monkeypatch.setenv("MCPCAST_APPROVAL_TIMEOUT", "5")
        _plan, out, mod = _generate(tmp_path, auth=AuthMode.API_KEY, profile=SafetyProfile.FULL)
        approval = (out / "petstore_mcp" / "approval.py").read_text()
        assert "max_pending=MAX_PENDING," in approval
        assert "max_pending_per_tenant=MAX_PENDING_PER_TENANT," in approval
        assert "max_pending_per_client=MAX_PENDING_PER_CLIENT," in approval
        for text in ((out / ".env.example").read_text(), (out / "README.md").read_text()):
            assert "MCPCAST_MAX_PENDING=" in text or "`MCPCAST_MAX_PENDING`" in text
            assert "MCPCAST_MAX_PENDING_PER_TENANT" in text
            assert "MCPCAST_MAX_PENDING_PER_CLIENT" in text
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))

            def call(key: str, pet: int):
                return client.call_tool("delete_pet", {"petId": pet}, headers={"x-api-key": key})

            waiting = [
                asyncio.ensure_future(call("sk-a1", 1)),
                asyncio.ensure_future(call("sk-a2", 2)),
            ]
            await asyncio.sleep(0.05)
            assert not any(f.done() for f in waiting)  # tenant a: two calls parked, two keys
            (r,) = await call(
                "sk-a3", 3
            )  # a fresh key of the same tenant: the tenant is at its cap
            err = json.loads(r.text)["error"]
            assert err["code"] == "APPROVAL_DENIED" and "per-tenant" in err["message"]
            assert "'a'" in err["message"] and "2 per tenant" in err["message"]
            other = asyncio.ensure_future(call("sk-b", 4))
            await asyncio.sleep(0.05)
            assert not other.done()  # tenant b's call waits for its reviewer, not denied
            for future in [*waiting, other]:
                future.cancel()
            await asyncio.gather(*waiting, other, return_exceptions=True)
        assert upstream.requests == []

    def test_caps_must_nest_or_the_server_refuses_to_start(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "MCPCAST_CLIENT_KEYS", json.dumps({"sk-a": {"client_id": "a", "tenant_id": "a"}})
        )
        monkeypatch.setenv("MCPCAST_MAX_PENDING_PER_TENANT", "10")
        monkeypatch.setenv("MCPCAST_MAX_PENDING_PER_CLIENT", "11")
        _plan, _out, mod = _generate(tmp_path, auth=AuthMode.API_KEY, profile=SafetyProfile.FULL)
        with pytest.raises(ValueError, match="max_pending_per_client must be between 1 and"):
            mod.build_server()


class TestClientSnippetsMatchAuthMode:
    """A desktop client launched from a GUI never inherits the shell."""

    def test_env_token_puts_the_credential_in_the_client_config(self, tmp_path):
        plan, out, _mod = _generate(tmp_path, auth=AuthMode.ENV_TOKEN)
        md = (out / "README.md").read_text()
        assert '"env": {"MCPCAST_UPSTREAM_TOKEN": "Bearer <your API token>"}' in md
        assert (
            'claude mcp add petstore -e MCPCAST_UPSTREAM_TOKEN="Bearer <your API token>" --' in md
        )
        assert "does not inherit your shell" in md
        assert plan.api.auth is AuthMode.ENV_TOKEN

    def test_header_modes_warn_that_stdio_cannot_work(self, tmp_path):
        md = (_generate(tmp_path)[1] / "README.md").read_text()
        assert "needs an `Authorization` header on every request" in md
        assert "--auth env-token" in md and '"env"' not in md

    def test_none_auth_snippets_stay_plain(self, tmp_path):
        md = (_generate(tmp_path / "n", auth=AuthMode.NONE)[1] / "README.md").read_text()
        assert '"env"' not in md and "claude mcp add petstore -- python" in md


class TestNullableParams:
    """FastAPI emits ``Optional[str]`` as ``anyOf: [{type: string}, {type: "null"}]``."""

    def test_nullable_anyof_is_typed_not_object(self):
        nullable = {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "Author",
            "default": None,
        }
        assert python_type(nullable) == "str"
        from promptise.mcpcast.emit import _displayed_type

        assert _displayed_type(nullable) == "string"
        assert (
            python_type({"anyOf": [{"type": "integer"}, {"type": "string"}]}) == "Any"
        )  # a real union stays Any
        assert (
            _displayed_type(
                {"anyOf": [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]}
            )
            == "array"
        )

    @pytest.mark.asyncio
    async def test_generated_schema_carries_the_type(self, tmp_path):
        spec = {
            "openapi": "3.1.0",
            "info": {"title": "Shelf"},
            "servers": [{"url": "https://s.example"}],
            "paths": {
                "/books": {
                    "get": {
                        "operationId": "list_books",
                        "parameters": [
                            {
                                "name": "author",
                                "in": "query",
                                "schema": {
                                    "anyOf": [{"type": "string"}, {"type": "null"}],
                                    "default": None,
                                    "description": "Exact author",
                                },
                            },
                            {
                                "name": "limit",
                                "in": "query",
                                "schema": {
                                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                                    "default": 10,
                                },
                            },
                        ],
                    }
                }
            },
        }
        plan = mcpcast(spec, name="shelf", auth=AuthMode.NONE)
        out = tmp_path / "s"
        write_project(plan, out)
        mod = _import(out / "server.py")
        client = TestClient(mod.build_server())
        (tool,) = await client.list_tools()
        assert tool.inputSchema["properties"]["author"]["anyOf"][0] == {"type": "string"}
        assert "author (string): Exact author" in tool.description
        assert "limit (integer)" in tool.description


# ---------------------------------------------------------------------------
# Audit findings: credentials, transport security, base URL override
# ---------------------------------------------------------------------------


class TestCredentialHygiene:
    """A secret never reaches an MCP client, not even inside an error message."""

    @pytest.mark.asyncio
    async def test_env_token_with_control_or_non_ascii_is_rejected_silently(
        self, tmp_path, monkeypatch
    ):
        _plan, _out, mod = _generate(tmp_path, auth=AuthMode.ENV_TOKEN)
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))
            for bad in ("Bearer sk_live_51H8\nx9aBcDeFg", "Bearer sk\rlive", "Bearer sécret"):
                monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", bad)
                (r,) = await client.call_tool("get_pet_by_id", {"petId": 1})
                err = json.loads(r.text)["error"]
                assert err["code"] == "UPSTREAM_AUTH_INVALID", bad
                assert "MCPCAST_UPSTREAM_TOKEN" in err["message"]
                assert "sk" not in err["message"] and "cret" not in err["message"]
            # surrounding whitespace (the trailing newline `echo` adds) is stripped
            monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", "  Bearer clean-token\n")
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1})
            assert "error" not in r.text
        assert [q.headers["authorization"] for q in upstream.requests] == ["Bearer clean-token"]

    @pytest.mark.asyncio
    async def test_tenant_token_with_embedded_newline_is_rejected_silently(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv(
            "MCPCAST_CLIENT_KEYS",
            json.dumps({"sk-acme-agent": {"client_id": "a", "tenant_id": "acme"}}),
        )
        monkeypatch.setenv(
            "MCPCAST_UPSTREAM_TOKENS", json.dumps({"acme": "Bearer sk_live_51H8\nx9aBcDeFgHiJk"})
        )
        _plan, _out, mod = _generate(tmp_path, auth=AuthMode.API_KEY)
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))
            (r,) = await client.call_tool(
                "get_pet_by_id", {"petId": 1}, headers={"x-api-key": "sk-acme-agent"}
            )
        err = json.loads(r.text)["error"]
        assert err["code"] == "UPSTREAM_AUTH_INVALID"
        assert "MCPCAST_UPSTREAM_TOKENS['acme']" in err["message"]
        assert "sk_live" not in json.dumps(err)
        assert upstream.requests == []

    @pytest.mark.asyncio
    async def test_passthrough_forwards_bearer_tokens_only(self, tmp_path):
        _plan, _out, mod = _generate(tmp_path)
        upstream = _Upstream()
        async with upstream.client() as http:
            server = mod.build_server(http_client=http)
            for bad in ("garbage-not-a-bearer", "Basic dXNlcjpwYXNz", "bearer lowercase"):
                (r,) = await TestClient(server, meta={"authorization": bad}).call_tool(
                    "get_pet_by_id", {"petId": 1}
                )
                err = json.loads(r.text)["error"]
                assert err["code"] == "UPSTREAM_AUTH_MISSING" and "Bearer <token>" in err["message"]
            (r,) = await TestClient(server, meta={"authorization": "Bearer o\nk"}).call_tool(
                "get_pet_by_id", {"petId": 1}
            )
            assert json.loads(r.text)["error"]["code"] == "UPSTREAM_AUTH_INVALID"
            (r,) = await TestClient(server, meta={"authorization": "Bearer ok"}).call_tool(
                "get_pet_by_id", {"petId": 1}
            )
            assert "error" not in r.text
        assert [q.headers["authorization"] for q in upstream.requests] == ["Bearer ok"]

    @pytest.mark.asyncio
    async def test_transport_errors_never_quote_the_header(self, tmp_path, monkeypatch):
        """h11 rejects an illegal header with its full value in the exception text."""
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", "Bearer top-secret")
        _plan, _out, mod = _generate(tmp_path, auth=AuthMode.ENV_TOKEN)

        async def protocol_error(request):
            raise httpx.LocalProtocolError("Illegal header value b'Bearer top-secret\\n'")

        async with httpx.AsyncClient(transport=httpx.MockTransport(protocol_error)) as http:
            client = TestClient(mod.build_server(http_client=http))
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1})
        err = json.loads(r.text)["error"]
        assert err["code"] == "UPSTREAM_UNREACHABLE"
        assert err["message"] == "GET /pet/{petId} failed: LocalProtocolError"
        assert "top-secret" not in r.text

    @pytest.mark.parametrize("auth", [AuthMode.ENV_TOKEN, AuthMode.API_KEY, AuthMode.PASSTHROUGH])
    @pytest.mark.asyncio
    async def test_upstream_bodies_never_echo_the_credential(self, tmp_path, monkeypatch, auth):
        """An upstream that quotes the header it received (a debug page, ``Invalid API key:
        Bearer …``) must not hand the token to the MCP client — in an error excerpt, a plain
        body or a JSON body, and not in part when the token straddles the excerpt cut."""
        secret = "Bearer SECRET-TENANT-UPSTREAM-TOKEN-xyz"
        token = secret.split(" ", 1)[1]
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", secret)
        monkeypatch.setenv(
            "MCPCAST_CLIENT_KEYS", json.dumps({"sk-t": {"client_id": "a", "tenant_id": "acme"}})
        )
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKENS", json.dumps({"acme": secret}))
        _plan, _out, mod = _generate(tmp_path, auth=auth)
        headers = {"x-api-key": "sk-t"} if auth is AuthMode.API_KEY else {}
        meta = {"authorization": secret} if auth is AuthMode.PASSTHROUGH else None
        answers: dict[str, httpx.Response] = {
            "error-json": httpx.Response(401, json={"error": f"Invalid API key: {secret}"}),
            "error-text": httpx.Response(500, text=f"<pre>Authorization: {secret}</pre>"),
            # 490 + len("[redacted]") == 500: cut exactly at the excerpt limit once scrubbed
            "straddle": httpx.Response(502, text="x" * 490 + secret + "y" * 100),
            "bare-token": httpx.Response(403, text=f"token {token} rejected"),
            "plain-body": httpx.Response(
                200, content=f"echo: {secret}".encode(), headers={"content-type": "text/plain"}
            ),
            "json-body": httpx.Response(
                200, json={"debug": {"authorization": secret, "token": token}, "id": 1}
            ),
        }
        seen: list[httpx.Request] = []

        async def echoing(request):
            seen.append(request)
            return answers[request.url.params["status"]]

        async with httpx.AsyncClient(transport=httpx.MockTransport(echoing)) as http:
            client = TestClient(mod.build_server(http_client=http), meta=meta)
            results = {}
            for case in answers:
                (r,) = await client.call_tool(
                    "find_pets_by_status", {"status": case}, headers=headers
                )
                results[case] = json.loads(r.text)
                assert token not in r.text and "SECRET" not in r.text, case
        # the credential did travel — only the echo of it is scrubbed
        assert {q.headers["authorization"] for q in seen} == {secret}
        errors = {case: results[case]["error"] for case in list(answers)[:4]}
        assert all(e["code"] == "UPSTREAM_ERROR" for e in errors.values())
        assert errors["error-json"]["message"].endswith(
            'HTTP 401: {"error":"Invalid API key: [redacted]"}'
        )
        assert errors["error-text"]["message"].endswith(
            "HTTP 500: <pre>Authorization: [redacted]</pre>"
        )
        assert errors["straddle"]["message"].endswith("HTTP 502: " + "x" * 490 + "[redacted]")
        assert errors["bare-token"]["message"].endswith("HTTP 403: token [redacted] rejected")
        assert results["plain-body"] == {"status": 200, "body": "echo: [redacted]"}
        assert results["json-body"] == {
            "debug": {"authorization": "[redacted]", "token": "[redacted]"},
            "id": 1,
        }

    @pytest.mark.asyncio
    async def test_url_encoded_echo_of_the_credential_is_scrubbed_too(self, tmp_path, monkeypatch):
        """A credential echoed from a request line comes back percent-encoded."""
        from urllib.parse import quote

        token = "ab+cd/ef==ghijkl"
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", f"Bearer {token}")
        _plan, _out, mod = _generate(tmp_path, auth=AuthMode.ENV_TOKEN)

        async def echo(request):
            return httpx.Response(
                400, text=f"bad request line: ?access_token={quote(token, safe='')}"
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(echo)) as http:
            client = TestClient(mod.build_server(http_client=http))
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1})
        message = json.loads(r.text)["error"]["message"]
        assert message.endswith("HTTP 400: bad request line: ?access_token=[redacted]")
        assert "ghijkl" not in r.text

    @pytest.mark.asyncio
    async def test_placeholder_credentials_are_refused_before_any_request(
        self, tmp_path, monkeypatch
    ):
        """``.env.example`` copied unchanged leaves ``<…>`` in place: refused, never sent."""
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", "Bearer <your API token>")
        _plan, _out, mod = _generate(tmp_path, auth=AuthMode.ENV_TOKEN)
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1})
        err = json.loads(r.text)["error"]
        assert err["code"] == "UPSTREAM_AUTH_INVALID"
        assert "MCPCAST_UPSTREAM_TOKEN still holds a placeholder" in err["message"]
        assert upstream.requests == []
        monkeypatch.setenv(
            "MCPCAST_CLIENT_KEYS", json.dumps({"sk-t": {"client_id": "a", "tenant_id": "acme"}})
        )
        monkeypatch.setenv(
            "MCPCAST_UPSTREAM_TOKENS", json.dumps({"acme": "Bearer <acme upstream token>"})
        )
        _plan, _out, mod = _generate(tmp_path / "api-key", auth=AuthMode.API_KEY)
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))
            (r,) = await client.call_tool(
                "get_pet_by_id", {"petId": 1}, headers={"x-api-key": "sk-t"}
            )
        err = json.loads(r.text)["error"]
        assert err["code"] == "UPSTREAM_AUTH_INVALID"
        assert "MCPCAST_UPSTREAM_TOKENS['acme'] still holds a placeholder" in err["message"]
        assert upstream.requests == []


class TestTransportSecurity:
    """A credential travels over https, loopback, or by explicit opt-in — never plain http."""

    @pytest.mark.parametrize("auth", [AuthMode.ENV_TOKEN, AuthMode.PASSTHROUGH, AuthMode.API_KEY])
    @pytest.mark.asyncio
    async def test_credentialed_modes_refuse_plain_http(self, tmp_path, monkeypatch, auth):
        monkeypatch.delenv("MCPCAST_ALLOW_INSECURE_HTTP", raising=False)
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", "Bearer t")
        monkeypatch.setenv(
            "MCPCAST_CLIENT_KEYS", json.dumps({"sk": {"client_id": "a", "tenant_id": "acme"}})
        )
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKENS", json.dumps({"acme": "Bearer t"}))
        plan = mcpcast(SPEC, name="petstore", base_url="http://api.acme.test/v1", auth=auth)
        out = tmp_path / "insecure"
        write_project(plan, out)
        mod = _import(out / "server.py")
        upstream = _Upstream()
        headers = {"x-api-key": "sk"} if auth is AuthMode.API_KEY else {}
        meta = {"authorization": "Bearer t"} if auth is AuthMode.PASSTHROUGH else None
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http), meta=meta)
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1}, headers=headers)
            err = json.loads(r.text)["error"]
            assert err["code"] == "UPSTREAM_INSECURE"
            assert "'api.acme.test'" in err["message"]
            assert "MCPCAST_ALLOW_INSECURE_HTTP=1" in err["message"]
            assert upstream.requests == []
            # the documented override
            monkeypatch.setenv("MCPCAST_ALLOW_INSECURE_HTTP", "1")
            mod = _import(out / "server.py")
            client = TestClient(mod.build_server(http_client=http), meta=meta)
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1}, headers=headers)
            assert "error" not in r.text
        assert [str(q.url) for q in upstream.requests] == ["http://api.acme.test/v1/pet/1"]

    @pytest.mark.asyncio
    async def test_loopback_http_and_credential_free_calls_are_fine(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MCPCAST_ALLOW_INSECURE_HTTP", raising=False)
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", "Bearer t")
        for base in ("http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000"):
            plan = mcpcast(SPEC, name="petstore", base_url=base, auth=AuthMode.ENV_TOKEN)
            out = tmp_path / "loop"
            write_project(plan, out)
            mod = _import(out / "server.py")
            upstream = _Upstream()
            async with upstream.client() as http:
                client = TestClient(mod.build_server(http_client=http))
                (r,) = await client.call_tool("get_pet_by_id", {"petId": 1})
                assert "error" not in r.text, base
        # auth 'none' sends no credential, so plain http anywhere is not a leak
        plan = mcpcast(SPEC, name="petstore", base_url="http://api.acme.test", auth=AuthMode.NONE)
        write_project(plan, tmp_path / "none")
        mod = _import(tmp_path / "none" / "server.py")
        upstream = _Upstream()
        async with upstream.client() as http:
            (r,) = await TestClient(mod.build_server(http_client=http)).call_tool(
                "get_pet_by_id", {"petId": 1}
            )
            assert "error" not in r.text


class TestBaseUrlOverride:
    """``MCPCAST_BASE_URL`` moves every tool, including routes with their own server."""

    FILES_SPEC = {
        "openapi": "3.0.0",
        "info": {"title": "Files"},
        "servers": [{"url": "https://api.example"}],
        "paths": {
            "/files": {
                "post": {
                    "operationId": "upload",
                    "servers": [{"url": "https://files.example/"}],
                }
            },
            "/items": {"get": {"operationId": "items"}},
        },
    }

    def test_override_is_refused_unless_it_is_one_clean_url(self, tmp_path, monkeypatch):
        """The three refusals of the plan's ``base_url``, mirrored at start-up — without
        echoing the value, which is what a ``?api_key=`` would make a secret."""
        plan = mcpcast(SPEC, name="petstore", auth=AuthMode.NONE)
        out = tmp_path / "b"
        write_project(plan, out)
        for bad, rule in (
            ("https://x.test/v1?api_key=SECRET123", "query string or fragment"),
            ("https://x.test/v1#SECRET123", "query string or fragment"),
            ("https://user:SECRET123@x.test/v1", "must not carry credentials"),
            ("https://x.test/v1 SECRET123", "whitespace or a control character"),
            ("https://x.test/v1\nSECRET123=1", "whitespace or a control character"),
            ("x.test/SECRET123", "absolute http:// or https:// URL"),
        ):
            monkeypatch.setenv("MCPCAST_BASE_URL", bad)
            with pytest.raises(RuntimeError, match="^MCPCAST_BASE_URL .*" + rule) as exc:
                _import(out / "server.py")
            assert "SECRET123" not in str(exc.value), bad
        monkeypatch.setenv("MCPCAST_BASE_URL", " https://staging.example/v1/ ")
        assert _import(out / "server.py").build_server() is not None
        assert sys.modules["petstore_mcp.config"].BASE_URL == "https://staging.example/v1"

    @pytest.mark.asyncio
    async def test_override_applies_to_route_level_hosts(self, tmp_path, monkeypatch):
        plan = mcpcast(
            self.FILES_SPEC, name="files", profile=SafetyProfile.STANDARD, auth=AuthMode.NONE
        )
        assert plan.tool("upload").routes[0].base_url == "https://files.example"
        out = tmp_path / "f"
        write_project(plan, out)
        monkeypatch.setenv("MCPCAST_BASE_URL", "https://staging.example/v1/")
        mod = _import(out / "server.py")
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(approval_handler=lambda r: True, http_client=http))
            await client.call_tool("upload", {})
            await client.call_tool("items", {})
        assert [str(q.url) for q in upstream.requests] == [
            "https://staging.example/v1/files",
            "https://staging.example/v1/items",
        ]
        # an empty value is "unset", not "the empty host"
        monkeypatch.setenv("MCPCAST_BASE_URL", "")
        mod = _import(out / "server.py")
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(approval_handler=lambda r: True, http_client=http))
            await client.call_tool("upload", {})
        assert str(upstream.requests[0].url) == "https://files.example/files"

    def test_readme_and_instructions_name_the_route_hosts(self):
        plan = mcpcast(
            self.FILES_SPEC, name="files", profile=SafetyProfile.STANDARD, auth=AuthMode.NONE
        )
        md = render_readme(plan)
        assert (
            "| `upload` | write | **required** | `POST /files` on `https://files.example` |" in md
        )
        assert "| `items` | read | — | `GET /items` |" in md
        assert (
            "- **Upstream API:** `https://api.example` (operations with their own server: "
            "`https://files.example`)" in md
        )
        from promptise.mcpcast.emit import _instructions

        text = _instructions(plan)
        assert "Every tool calls" not in text
        assert "some operations are served from their own host (https://files.example)" in text
        plain = _instructions(mcpcast(SPEC, name="petstore"))
        assert "Every tool calls the upstream API at https://petstore.example/api/v3." in plain

    @pytest.mark.asyncio
    async def test_servers_less_spec_from_a_url_carries_no_route_hosts(self, tmp_path, monkeypatch):
        """A FastAPI app's spec has no servers block: fetched from a URL, its origin is the
        plan's base — and must not be frozen into every route."""
        from promptise.mcpcast import build_plan, extract_operations

        spec = {
            "openapi": "3.1.0",
            "info": {"title": "Shop"},
            "paths": {
                "/items": {"get": {"operationId": "list_items"}},
                "/items/{id}": {
                    "get": {
                        "operationId": "get_item",
                        "parameters": [
                            {
                                "name": "id",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            }
                        ],
                    }
                },
            },
        }
        spec_url = "http://127.0.0.1:8000/openapi.json"
        ops = extract_operations(spec, spec_url=spec_url)
        # the wizard re-plans with a base URL the user typed
        plan = build_plan(ops, base_url="https://shop.example/api", name="shop", auth=AuthMode.NONE)
        assert plan.api.base_url == "https://shop.example/api"
        assert all(r.base_url is None for t in plan.tools for r in t.routes)
        out = tmp_path / "shop"
        write_project(plan, out)
        assert "base_url=" not in (out / "shop_mcp" / "tools" / "items.py").read_text()
        monkeypatch.setenv("MCPCAST_BASE_URL", "https://prod.example")
        mod = _import(out / "server.py")
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))
            await client.call_tool("list_items", {})
            await client.call_tool("get_item", {"id": "7"})
        assert [str(q.url) for q in upstream.requests] == [
            "https://prod.example/items",
            "https://prod.example/items/7",
        ]


# ---------------------------------------------------------------------------
# Audit findings: responses, routing, api-key start-up
# ---------------------------------------------------------------------------


class TestResponses:
    async def _call(self, mod, handler, tool="get_pet_by_id", args=None):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = TestClient(mod.build_server(http_client=http))
            (r,) = await client.call_tool(tool, args if args is not None else {"petId": 1})
        return json.loads(r.text)

    @pytest.mark.asyncio
    async def test_non_json_body_with_json_content_type_is_an_upstream_error(self, tmp_path):
        _plan, _out, mod = _generate(tmp_path, auth=AuthMode.NONE)

        async def waf(request):
            return httpx.Response(
                200,
                content=b"<html>WAF challenge</html>",
                headers={"content-type": "application/json"},
            )

        err = (await self._call(mod, waf))["error"]
        assert err["code"] == "UPSTREAM_ERROR" and err["details"] == {"status": 200}
        assert "not JSON" in err["message"] and "application/json" in err["message"]

    @pytest.mark.asyncio
    async def test_rate_limits_and_timeouts_are_retryable_with_retry_after(self, tmp_path):
        _plan, _out, mod = _generate(tmp_path, auth=AuthMode.NONE)
        for status in (408, 425, 429, 500, 503):

            async def limited(request, status=status):
                return httpx.Response(
                    status, json={"error": "rate limited"}, headers={"retry-after": "3"}
                )

            err = (await self._call(mod, limited))["error"]
            assert err["code"] == "UPSTREAM_ERROR" and err["retryable"] is True, status
            assert err["details"] == {"status": status, "retry_after": "3"}
        for status in (400, 401, 404, 422):

            async def rejected(request, status=status):
                return httpx.Response(status, json={"error": "no"})

            err = (await self._call(mod, rejected))["error"]
            assert err["retryable"] is False and err["details"] == {"status": status}

    @pytest.mark.asyncio
    async def test_error_body_excerpt_is_bounded_and_configurable(self, tmp_path, monkeypatch):
        _plan, out, mod = _generate(tmp_path, auth=AuthMode.NONE)

        async def verbose(request):
            return httpx.Response(500, text="x" * 2000)

        err = (await self._call(mod, verbose))["error"]
        assert err["message"] == "GET /pet/{petId} returned HTTP 500: " + "x" * 500
        monkeypatch.setenv("MCPCAST_ERROR_EXCERPT_CHARS", "0")
        mod = _import(out / "server.py")
        err = (await self._call(mod, verbose))["error"]
        assert err["message"] == "GET /pet/{petId} returned HTTP 500"
        assert "MCPCAST_ERROR_EXCERPT_CHARS" in (out / "README.md").read_text()

    @pytest.mark.asyncio
    async def test_response_bodies_are_capped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MCPCAST_MAX_RESPONSE_BYTES", "1000")
        _plan, out, mod = _generate(tmp_path, auth=AuthMode.NONE)

        async def big(request):
            return httpx.Response(200, content=b"x" * 1001, headers={"content-type": "text/plain"})

        err = (await self._call(mod, big))["error"]
        assert err["code"] == "UPSTREAM_RESPONSE_TOO_LARGE"
        assert err["details"] == {"status": 200, "limit": 1000}
        assert "MCPCAST_MAX_RESPONSE_BYTES" in err["message"]

        async def announced(request):
            # refused on the Content-Length alone, before a byte of body is read
            return httpx.Response(200, headers={"content-length": "999999"}, content=b"{}")

        assert (await self._call(mod, announced))["error"]["code"] == "UPSTREAM_RESPONSE_TOO_LARGE"

        async def streamed(request):
            async def chunks():
                for _ in range(20):
                    yield b"y" * 100

            return httpx.Response(
                200, stream=_AsyncChunks(chunks()), headers={"content-type": "text/plain"}
            )

        assert (await self._call(mod, streamed))["error"]["code"] == "UPSTREAM_RESPONSE_TOO_LARGE"

        async def fits(request):
            return httpx.Response(200, content=b"z" * 1000, headers={"content-type": "text/plain"})

        assert (await self._call(mod, fits)) == {"status": 200, "body": "z" * 1000}

    @pytest.mark.asyncio
    async def test_one_client_is_reused_and_closed_on_shutdown(self, tmp_path):
        _plan, _out, mod = _generate(tmp_path, auth=AuthMode.NONE)
        server = mod.build_server()
        (upstream,) = [
            m.__self__ for m in server._lifecycle._shutdown_hooks if hasattr(m, "__self__")
        ]
        assert upstream._owned is None
        first = upstream.client()
        assert upstream.client() is first  # pooled, not one per call
        await server._lifecycle.shutdown()
        assert upstream._owned is None and first.is_closed


class _AsyncChunks(httpx.AsyncByteStream):
    def __init__(self, gen):
        self._gen = gen

    async def __aiter__(self):
        async for chunk in self._gen:
            yield chunk


class TestUpstreamDeadline:
    """MCPCAST_TIMEOUT bounds the whole call, not each read: a peer that trickles bytes
    cannot hold a tool call and its buffer open for as long as it likes."""

    async def _call(self, mod, handler):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = TestClient(mod.build_server(http_client=http))
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1})
        return json.loads(r.text)

    @pytest.mark.asyncio
    async def test_slow_drip_body_is_cut_at_the_deadline(self, tmp_path, monkeypatch):
        import asyncio
        import time

        monkeypatch.setenv("MCPCAST_TIMEOUT", "0.3")
        _plan, out, mod = _generate(tmp_path, auth=AuthMode.NONE)

        async def trickle(request):
            async def chunks():
                for _ in range(50):  # 5 s in total, no single pause near the deadline
                    await asyncio.sleep(0.1)
                    yield b"x"

            return httpx.Response(
                200, stream=_AsyncChunks(chunks()), headers={"content-type": "text/plain"}
            )

        started = time.monotonic()
        err = (await self._call(mod, trickle))["error"]
        assert time.monotonic() - started < 2
        assert err["code"] == "UPSTREAM_TIMEOUT" and err["retryable"] is True
        assert err["message"] == "GET /pet/{petId} did not complete within 0.3s (MCPCAST_TIMEOUT)"
        assert err["details"] == {"timeout": 0.3}

        async def stalled(request):
            await asyncio.sleep(5)  # never answers the status line in time
            return httpx.Response(200, json={})

        assert (await self._call(mod, stalled))["error"]["code"] == "UPSTREAM_TIMEOUT"

        async def httpx_timeout(request):
            raise httpx.ReadTimeout("timed out")  # httpx's own per-operation timeout

        err = (await self._call(mod, httpx_timeout))["error"]
        assert err["code"] == "UPSTREAM_TIMEOUT" and err["retryable"] is True

        async def prompt(request):
            return httpx.Response(200, json={"id": 1})

        assert (await self._call(mod, prompt)) == {"id": 1}

        readme = (out / "README.md").read_text()
        assert "| `MCPCAST_TIMEOUT` | Total time one upstream call may take" in readme
        assert "`UPSTREAM_TIMEOUT`" in readme
        env = (out / ".env.example").read_text()
        assert "# Total time one upstream call may take" in env and "UPSTREAM_TIMEOUT" in env
        upstream = (out / "petstore_mcp" / "upstream.py").read_text()
        assert "asyncio.wait_for(" in upstream and "UPSTREAM_TIMEOUT" in upstream
        assert "httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)" in upstream  # per-operation too


class TestRouteSelection:
    @pytest.mark.asyncio
    async def test_no_matching_route_is_a_validation_error_not_the_first_route(self, tmp_path):
        plan, out, _ = _generate(tmp_path, profile=SafetyProfile.READ_ONLY, auth=AuthMode.NONE)
        get = plan.tool("get_pet_by_id")
        find = plan.tool("find_pets_by_status")
        merged = get.model_copy(
            update={
                "name": "find_pet",
                "routes": [get.routes[0], find.routes[0]],
                "params": {**get.params, **find.params},
                "example": None,
            }
        )
        merged.params["petId"].required = False
        merged.routes[1].params["status"] = RouteParam(location="query", required=True)
        plan.tools = [merged]
        write_project(plan, out)
        mod = _import(out / "server.py")
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))
            (r,) = await client.call_tool("find_pet", {})
            err = json.loads(r.text)["error"]
            assert err["code"] == "VALIDATION_ERROR"
            assert err["message"].endswith("Provide one of: petId | status")
            (r,) = await client.call_tool("find_pet", {"status": "sold"})
            assert "error" not in r.text
        assert [str(q.url) for q in upstream.requests] == [
            "https://petstore.example/api/v3/pet/findByStatus?status=sold"
        ]

    def test_runtime_hint_uses_python_identifiers(self):
        plan = mcpcast(SPEC, name="petstore")
        weird = plan.tool("weird_op_name")
        find = plan.tool("find_pets_by_status")
        merged = weird.model_copy(
            update={
                "name": "either",
                "routes": [weird.routes[0], find.routes[0]],
                "params": {**weird.params, **find.params},
                "example": None,
            }
        )
        merged.params["user-name"].required = False
        merged.routes[1].params["status"] = RouteParam(location="query", required=True)
        plan.tools = [merged]
        src = render_server(plan)
        assert 'hint="user_name | status"' in src


class TestApiKeyStartup:
    def test_build_server_refuses_an_empty_key_set(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MCPCAST_CLIENT_KEYS", raising=False)
        plan = mcpcast(SPEC, name="petstore", auth=AuthMode.API_KEY)
        out = tmp_path / "k"
        write_project(plan, out)
        mod = _import(out / "server.py")  # importing is fine: nothing is built yet
        with pytest.raises(RuntimeError, match="MCPCAST_CLIENT_KEYS is empty"):
            mod.build_server()
        with pytest.raises(RuntimeError, match="MCPCAST_CLIENT_KEYS is empty"):
            mod.server  # noqa: B018 — the lazy attribute builds on first access
        monkeypatch.setenv("MCPCAST_CLIENT_KEYS", "{}")
        with pytest.raises(RuntimeError, match="at least one client key"):
            mod.build_server()
        monkeypatch.setenv("MCPCAST_CLIENT_KEYS", json.dumps({"k": {"client_id": "a"}}))
        assert mod.server is mod.server  # built once, then reused
        assert mod.build_server() is not mod.server

    def test_build_server_refuses_documentation_and_placeholder_keys(self, tmp_path, monkeypatch):
        """A key from the README is public and a ``<placeholder>`` was never filled in."""
        import secrets

        plan = mcpcast(SPEC, name="petstore", auth=AuthMode.API_KEY)
        out = tmp_path / "k"
        write_project(plan, out)
        mod = _import(out / "server.py")
        real = "sk-" + secrets.token_urlsafe(32)
        for key in ("sk-acme", "sk-reviewer", "sk-<generated>", "<paste the key here>"):
            monkeypatch.setenv(
                "MCPCAST_CLIENT_KEYS",
                json.dumps(
                    {
                        real: {"client_id": "agent", "tenant_id": "acme"},
                        key: {"client_id": "other", "tenant_id": "acme"},
                    }
                ),
            )
            with pytest.raises(RuntimeError, match="placeholder or documentation key") as exc:
                mod.build_server()
            assert "secrets.token_urlsafe(32)" in str(exc.value)
            assert "generated>" not in str(exc.value) and "paste" not in str(exc.value)
        monkeypatch.setenv(
            "MCPCAST_CLIENT_KEYS", json.dumps({real: {"client_id": "agent", "tenant_id": "acme"}})
        )
        assert mod.build_server() is not None

    def test_env_example_ships_no_working_client_key(self, monkeypatch):
        """The shape is a comment, the variable is blank and a verbatim copy fails closed."""
        from dotenv import dotenv_values

        plan = mcpcast(SPEC, name="petstore", auth=AuthMode.API_KEY)
        env = render_project(plan)[".env.example"]
        active = [line for line in env.splitlines() if line.startswith("MCPCAST_CLIENT_KEYS=")]
        assert active == ["MCPCAST_CLIENT_KEYS="]
        assert not any("sk-acme" in line for line in env.splitlines() if not line.startswith("#"))
        assert '#   MCPCAST_CLIENT_KEYS={"sk-<generated>": {"client_id": "acme-agent", ' in env
        assert "python -c 'import secrets; print(\"sk-\" + secrets.token_urlsafe(32))'" in env
        assert "refuses to start with no key" in env
        for name, value in dotenv_values(stream=__import__("io").StringIO(env)).items():
            monkeypatch.setenv(name, value or "")
        assert os.environ["MCPCAST_CLIENT_KEYS"] == ""
        assert "<your API token>" in os.environ["MCPCAST_UPSTREAM_TOKENS"]
        files = render_project(plan)
        assert "def build_server" in files["petstore_mcp/server.py"]

    def test_config_docstring_and_readme_state_when_credentials_are_read(self, tmp_path):
        plan, out, _mod = _generate(tmp_path, auth=AuthMode.ENV_TOKEN)
        config = (out / "petstore_mcp" / "config.py").read_text()
        assert "read once at import" not in config
        assert "read by upstream.py on every call" in config
        assert "MCPCAST_CLIENT_KEYS (api-key mode)" in config and "needs a restart" in config
        md = render_readme(mcpcast(SPEC, name="petstore", auth=AuthMode.API_KEY))
        assert "rotating them needs no restart" not in md
        assert "read once at start-up" in md and "rotate without a restart" in md


# ---------------------------------------------------------------------------
# Audit findings: names and hostile text
# ---------------------------------------------------------------------------


class TestNames:
    @pytest.mark.parametrize(
        ("tag", "expected"),
        [
            ("annotations", "annotations_tools"),
            ("config", "config_tools"),
            ("upstream", "upstream_tools"),
            ("approval", "approval_tools"),
            ("server", "server_tools"),
            ("tools", "tools_tools"),
            ("register", "register_tools"),
            ("import", "import_tools"),
            ("match", "match_tools"),
            ("type", "type_tools"),
            ("pet", "pet"),
            ("x" * 300, "x" * 40),
            ("a-" * 30, "a_" * 19 + "a"),
        ],
    )
    def test_tool_group_never_breaks_the_import(self, tag, expected):
        plan = mcpcast(SPEC, name="petstore")
        tool = plan.tool("get_pet_by_id")
        tool.tags = [tag]
        assert tool_group(tool) == expected
        assert len(tool_group(tool)) <= 40

    def test_annotations_tag_generates_an_importable_server(self, tmp_path):
        """``from . import annotations`` binds the __future__ feature, not the module."""
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Grafana"},
            "servers": [{"url": "https://g.example"}],
            "paths": {
                "/api/annotations": {
                    "get": {"operationId": "getAnnotations", "tags": ["annotations"]}
                },
                "/api/config": {"get": {"operationId": "getConfig", "tags": ["config"]}},
            },
        }
        plan = mcpcast(spec, name="grafana", auth=AuthMode.ENV_TOKEN)
        out = tmp_path / "g"
        write_project(plan, out)
        assert (out / "grafana_mcp" / "tools" / "annotations_tools.py").exists()
        assert (out / "grafana_mcp" / "tools" / "config_tools.py").exists()
        mod = _import(out / "server.py")
        names = {t.name for t in mod.server._tool_registry.list_all()}
        assert names == {"get_annotations", "get_config"}

    @pytest.mark.asyncio
    async def test_select_route_and_routes_as_spec_names_cannot_shadow(self, tmp_path):
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Routing"},
            "servers": [{"url": "https://r.example"}],
            "paths": {
                "/routes/select": {"get": {"operationId": "selectRoute"}},
                "/things": {
                    "get": {
                        "operationId": "getThing",
                        "parameters": [
                            {"name": "ROUTES", "in": "query", "schema": {"type": "string"}},
                            {"name": "select_route", "in": "query", "schema": {"type": "string"}},
                        ],
                    }
                },
                "/register": {"get": {"operationId": "register"}},
            },
        }
        plan = mcpcast(spec, name="routing", auth=AuthMode.NONE)
        assert plan.tool_names == ["select_route_op", "get_thing", "register_op"]
        out = tmp_path / "r"
        write_project(plan, out)
        mod = _import(out / "server.py")
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))
            tools = {t.name: t for t in await client.list_tools()}
            assert set(tools["get_thing"].inputSchema["properties"]) == {"ROUTES_", "select_route_"}
            (r,) = await client.call_tool("select_route_op", {})
            assert "error" not in r.text
            (r,) = await client.call_tool("get_thing", {"ROUTES_": "a", "select_route_": "b"})
            assert "error" not in r.text
            (r,) = await client.call_tool("get_thing", {})
            assert "error" not in r.text
        assert [str(q.url) for q in upstream.requests] == [
            "https://r.example/routes/select",
            "https://r.example/things?ROUTES=a&select_route=b",
            "https://r.example/things",
        ]


class TestHostileText:
    def _plan_with_info(self, **info):
        return mcpcast({**SPEC, "info": {"title": "Acme", **info}}, name="acme")

    @pytest.mark.parametrize(
        "description",
        ["Files under C:\\Users\\acme", 'Acme \\ foo "quoted" \\N{BULLET} \\x00 \\"', "a\x00b\tc"],
    )
    def test_pyproject_stays_valid_toml(self, description):
        tomllib = pytest.importorskip("tomllib")
        toml = render_project(self._plan_with_info(description=description))["pyproject.toml"]
        data = tomllib.loads(toml)
        assert data["project"]["description"].startswith("MCP server for Acme: ")
        assert data["project"]["description"].endswith("— generated by promptise mcpcast")
        assert "\x00" not in data["project"]["description"]

    def test_lone_surrogates_do_not_crash_generation(self, tmp_path):
        bad = "Acme \ud83d API"
        plan = self._plan_with_info(description=bad)
        plan = plan.model_copy(update={"api": plan.api.model_copy(update={"spec_source": bad})})
        plan.tools[0].description = bad
        plan.tools[0].params["petId"].description = bad
        plan.tools[0].tags = [bad]
        plan.dropped[0].reason = bad
        files = render_project(plan)
        for path, text in files.items():
            text.encode("utf-8")  # would raise on a lone surrogate
            assert "\ud83d" not in text, path
        write_project(plan, tmp_path / "s")
        mod = _import(tmp_path / "s" / "server.py")
        assert mod.server._tool_registry.get("get_pet_by_id").description.startswith("Acme ")

    @pytest.mark.asyncio
    async def test_non_finite_numbers_never_become_bare_names(self, tmp_path):
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Gauge"},
            "servers": [{"url": "https://g.example"}],
            "paths": {
                "/gauge": {
                    "post": {
                        "operationId": "setGauge",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["value"],
                                        "properties": {
                                            "value": {"type": "number", "example": float("inf")},
                                            "ratio": {"type": "number", "default": float("nan")},
                                        },
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }
        plan = mcpcast(spec, name="gauge", profile=SafetyProfile.STANDARD, auth=AuthMode.NONE)
        # a non-finite spec example cannot fit a number: the plan derives a fitting one instead
        assert plan.tool("set_gauge").example == {"value": 1.0}
        files = render_project(plan)
        for path, src in files.items():
            if path.endswith(".py"):
                assert not re.search(r"\b(inf|nan)\b", src), path
        out = tmp_path / "g"
        write_project(plan, out)
        mod = _import(out / "server.py")
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(approval_handler=lambda r: True, http_client=http))
            (r,) = await client.call_tool("set_gauge", {"value": 1.5})
            assert "error" not in r.text

    def test_readme_escapes_markdown_from_the_spec(self):
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": "Acme [phish](https://evil.test/login) <img src=x onerror=alert(1)>",
                "description": "Acme | **bold** `code` <b>x</b>",
            },
            "servers": [{"url": "https://api.acme.test/v1"}],
            "paths": {
                "/x` | **bold** [phish](https://evil.test) <b>html</b> `": {
                    "get": {"operationId": "weirdPath"}
                }
            },
        }
        plan = mcpcast(spec, name="acme")
        plan = plan.model_copy(
            update={
                "api": plan.api.model_copy(
                    update={
                        "spec_source": "spec`s [x](y) <z> | q.yaml",
                        "base_url": "https://a.test/v1",
                    }
                )
            }
        )
        md = render_readme(plan)
        prose = re.sub(r"``.*?``|`[^`\n]*`", "", md)  # code spans render literally anyway
        assert re.search(r"(?<!\\)\[phish\]", prose) is None  # no unescaped link anywhere
        assert re.search(r"(?<!\\)<(img|b)\b", prose) is None  # no unescaped tag anywhere
        assert md.startswith(
            "# Acme \\[phish](https://evil.test/login) \\<img src=x onerror=alert(1)>: "
            "Acme \\| **bold** \\`code\\` \\<b>x\\</b> — MCP server"
        )
        assert "\\*\\*bold\\*\\*" not in md  # only what is needed is escaped
        row = next(line for line in md.splitlines() if line.startswith("| `weird_path`"))
        assert row.count("|") == 5 + row.count("\\|")  # the row still has four cells
        assert "`` GET /x` \\| **bold** [phish](https://evil.test) <b>html</b> ` ``" in row
        assert "from\n``spec`s [x](y) <z> \\| q.yaml``." in md

    def test_layout_is_linear_in_the_example_depth(self):
        import time

        from promptise.mcpcast.emit import _literal

        value: dict = {"leaf": "x" * 90}
        for depth in range(60):
            value = {"n": [value, {"k": depth}]}
        started = time.perf_counter()
        rendered = _literal({"body": value}, 8)
        # Linear: well under a second; the old O(2**depth) layout took minutes at depth 22,
        # so a generous bound still separates the two on a slow or loaded interpreter.
        assert time.perf_counter() - started < 10.0
        assert eval(rendered) == {"body": value}  # noqa: S307 — our own literal


class TestWriteProjectGuards:
    def test_non_empty_directory_without_a_plan_is_refused(self, tmp_path):
        out = tmp_path / "mine"
        out.mkdir()
        (out / "README.md").write_text("# mine\n")
        (out / "server.py").write_text("print(1)\n")
        (out / "tests").mkdir()
        (out / "tests" / "conftest.py").write_text("# mine\n")
        plan = mcpcast(SPEC, name="petstore")
        with pytest.raises(MCPcastError, match="not empty and holds no mcpcast.plan.yaml") as exc:
            write_project(plan, out)
        assert "README.md" in str(exc.value) and "server.py" in str(exc.value)
        assert (out / "README.md").read_text() == "# mine\n"
        assert (out / "server.py").read_text() == "print(1)\n"
        # even a directory holding only unrelated files needs --force
        other = tmp_path / "other"
        other.mkdir()
        (other / "notes.txt").write_text("x")
        with pytest.raises(MCPcastError, match="no generated file exists there yet"):
            write_project(plan, other)
        write_project(plan, other, force=True)
        assert (other / "notes.txt").exists() and (other / "server.py").exists()
        write_project(plan, out, force=True)
        assert (out / "README.md").read_text().startswith("# Petstore API")

    def test_empty_or_missing_or_own_project_directories_are_fine(self, tmp_path):
        plan = mcpcast(SPEC, name="petstore")
        empty = tmp_path / "empty"
        empty.mkdir()
        write_project(plan, empty)
        write_project(plan, tmp_path / "missing" / "nested")
        # regenerating into a directory that holds a plan file never needs force
        (empty / "notes.txt").write_text("x")
        write_project(plan, empty)
        write_project(plan, empty, write_plan=False)
        assert (empty / "notes.txt").exists()

    def test_launcher_import_has_no_side_effects(self, tmp_path):
        plan, out, mod = _generate(tmp_path, auth=AuthMode.NONE)
        assert "server" not in vars(mod)  # not built at import…
        server = mod.server  # …but on first use, once
        assert vars(mod)["server"] is server and mod.server is server
        with pytest.raises(AttributeError):
            mod.nope  # noqa: B018
        assert '"""' + "Run the petstore MCP server" in (out / "server.py").read_text()


class TestDotenv:
    """``.env.example`` says "copy to .env": the launcher and the command line read it —
    at serve time only, never at import or in ``build_server()``."""

    @staticmethod
    def _scrub(monkeypatch):
        monkeypatch.delenv("PROMPTISE_NO_DOTENV", raising=False)
        for name in ("MCPCAST_CLIENT_KEYS", "MCPCAST_UPSTREAM_TOKENS"):
            monkeypatch.setenv(name, "sentinel")  # a restore point for the file's later write
            monkeypatch.delenv(name)

    @pytest.mark.asyncio
    async def test_launcher_main_reads_the_env_beside_it(self, tmp_path, monkeypatch):
        self._scrub(monkeypatch)
        plan = mcpcast(SPEC, name="petstore", auth=AuthMode.API_KEY)
        out = tmp_path / "proj"
        write_project(plan, out)
        (out / ".env").write_text(
            'MCPCAST_CLIENT_KEYS={"sk-from-dotenv": {"client_id": "a", "tenant_id": "acme"}}\n'
            'MCPCAST_UPSTREAM_TOKENS={"acme": "Bearer from-dotenv"}\n'
        )
        monkeypatch.chdir(tmp_path)  # a GUI client starts the launcher from anywhere
        mod = _import(out / "server.py")
        assert "MCPCAST_CLIENT_KEYS" not in os.environ  # importing reads nothing…
        with pytest.raises(RuntimeError, match="MCPCAST_CLIENT_KEYS is empty"):
            mod.build_server()  # …and neither does building: tests and evaluations own the env
        served: dict = {}
        monkeypatch.setattr(
            mod, "run", lambda server, argv: served.update(server=server, argv=argv)
        )
        mod.main(["--transport", "http"])
        assert served == {"server": mod.server, "argv": ["--transport", "http"]}
        assert "sk-from-dotenv" in os.environ["MCPCAST_CLIENT_KEYS"]
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http))
            (r,) = await client.call_tool(
                "get_pet_by_id", {"petId": 1}, headers={"x-api-key": "sk-from-dotenv"}
            )
            assert "error" not in r.text
            (r,) = await client.call_tool(
                "get_pet_by_id", {"petId": 1}, headers={"x-api-key": "no"}
            )
            assert "error" in r.text
        assert [q.headers["authorization"] for q in upstream.requests] == ["Bearer from-dotenv"]

    def test_command_line_reads_the_env_of_the_working_directory(self, tmp_path, monkeypatch):
        self._scrub(monkeypatch)
        plan = mcpcast(SPEC, name="petstore", auth=AuthMode.API_KEY)
        out = tmp_path / "proj"
        write_project(plan, out)
        (out / ".env").write_text(
            'MCPCAST_CLIENT_KEYS={"sk-from-dotenv": {"client_id": "a", "tenant_id": "acme"}}\n'
        )
        _import(out / "server.py")
        cli = sys.modules["petstore_mcp.__main__"]  # `python -m petstore_mcp` / `petstore-mcp`
        served: dict = {}
        monkeypatch.setattr(
            cli, "run", lambda server, argv: served.update(server=server, argv=argv)
        )
        monkeypatch.chdir(out)
        cli.main([])
        assert served["argv"] == [] and served["server"] is not None
        assert "sk-from-dotenv" in os.environ["MCPCAST_CLIENT_KEYS"]

    def test_env_example_says_who_reads_it(self):
        for auth in AuthMode:
            env = render_project(mcpcast(SPEC, name="petstore", auth=auth))[".env.example"]
            assert "Who reads .env: `python server.py`" in env
            assert "`python -m petstore_mcp` and `petstore-mcp`" in env
            assert "`docker run --env-file .env`" in env and "PROMPTISE_NO_DOTENV=1" in env
            assert "The generated\n# tests never read it." in env


class TestPlainHttpHosts:
    """A plain-http upstream is refused at run time; the project must say so up front."""

    SPEC = {
        "openapi": "3.0.0",
        "info": {"title": "Intranet"},
        "servers": [{"url": "http://api.intranet.corp:8080"}],
        "paths": {
            "/a": {"get": {"operationId": "a", "servers": [{"url": "http://legacy.example.test"}]}},
            "/b": {"get": {"operationId": "b", "servers": [{"url": "http://localhost:9000"}]}},
            "/c": {"get": {"operationId": "c", "servers": [{"url": "https://ok.example"}]}},
            "/d": {
                "get": {"operationId": "d", "servers": [{"url": "http://legacy.example.test/"}]}
            },
        },
    }

    def test_plain_http_hosts_names_every_credentialed_plain_http_host_once(self):
        from promptise.mcpcast import plain_http_hosts

        for auth in (AuthMode.ENV_TOKEN, AuthMode.API_KEY, AuthMode.PASSTHROUGH):
            plan = mcpcast(self.SPEC, name="intranet", auth=auth)
            assert plain_http_hosts(plan) == ["api.intranet.corp:8080", "legacy.example.test"]
        assert plain_http_hosts(mcpcast(self.SPEC, name="intranet", auth=AuthMode.NONE)) == []
        for base in ("https://api.example", "http://localhost:8000", "http://[::1]:8000"):
            plan = mcpcast(self.SPEC, name="intranet", base_url=base, auth=AuthMode.ENV_TOKEN)
            assert plain_http_hosts(plan) == [], base
        assert plain_http_hosts(mcpcast(SPEC, name="petstore", auth=AuthMode.ENV_TOKEN)) == []

    def test_env_example_opts_in_and_names_the_hosts(self):
        env = render_project(mcpcast(self.SPEC, name="intranet", auth=AuthMode.ENV_TOKEN))[
            ".env.example"
        ]
        assert "\nMCPCAST_ALLOW_INSECURE_HTTP=1\n" in env
        assert "# plan calls api.intranet.corp:8080, legacy.example.test over plain http" in env
        assert "UPSTREAM_INSECURE" in env
        for plan in (
            mcpcast(self.SPEC, name="intranet", auth=AuthMode.NONE),  # nothing to leak
            mcpcast(SPEC, name="petstore", auth=AuthMode.ENV_TOKEN),  # https
        ):
            env = render_project(plan)[".env.example"]
            assert "# MCPCAST_ALLOW_INSECURE_HTTP=1" in env
            assert "\nMCPCAST_ALLOW_INSECURE_HTTP=1" not in env


def _with_credential(plan: MCPcastPlan, location: str, name: str) -> MCPcastPlan:
    """*plan* with the upstream credential moved to *location*/*name*, re-validated."""
    data = plan.model_dump(mode="json")
    data["api"].update(credential_location=location, credential_name=name)
    return MCPcastPlan.model_validate(data)


class TestCredentialLocation:
    """The credential travels where the spec's security scheme puts it: the Authorization
    header by default, a custom header or a query parameter for an ``apiKey`` scheme."""

    async def _one_call(self, tmp_path, plan, *, headers=None, meta=None):
        out = tmp_path / plan.api.credential_name.lower()
        write_project(plan, out)
        mod = _import(out / "server.py")
        upstream = _Upstream()
        async with upstream.client() as http:
            client = TestClient(mod.build_server(http_client=http), meta=meta)
            (r,) = await client.call_tool("get_pet_by_id", {"petId": 1}, headers=headers or {})
        return json.loads(r.text), upstream.requests, out

    @pytest.mark.asyncio
    async def test_env_token_presents_a_custom_header(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", "my-x-api-key-value")
        plan = _with_credential(
            mcpcast(SPEC, name="petstore", auth=AuthMode.ENV_TOKEN), "header", "X-API-Key"
        )
        result, requests, out = await self._one_call(tmp_path, plan)
        assert "error" not in result
        (request,) = requests
        assert request.headers["x-api-key"] == "my-x-api-key-value"
        assert "authorization" not in request.headers
        assert str(request.url) == "https://petstore.example/api/v3/pet/1"
        env = (out / ".env.example").read_text()
        assert "# env-token: the X-API-Key header value sent on every upstream call." in env
        assert "MCPCAST_UPSTREAM_TOKEN=<your API key>" in env
        assert 'CREDENTIAL_NAME = "X-API-Key"' in (out / "petstore_mcp" / "config.py").read_text()

    @pytest.mark.asyncio
    async def test_env_token_presents_a_query_parameter(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", "QUERY-KEY-123456")
        plan = _with_credential(
            mcpcast(SPEC, name="petstore", auth=AuthMode.ENV_TOKEN), "query", "api_key"
        )
        result, requests, out = await self._one_call(tmp_path, plan)
        assert "error" not in result
        (request,) = requests
        assert str(request.url) == "https://petstore.example/api/v3/pet/1?api_key=QUERY-KEY-123456"
        assert "authorization" not in request.headers and "api_key" not in request.headers
        env = (out / ".env.example").read_text()
        assert "# env-token: the api_key query parameter value sent on every upstream call." in env
        config = (out / "petstore_mcp" / "config.py").read_text()
        assert 'CREDENTIAL_LOCATION = "query"' in config and "api_key query parameter" in config

    @pytest.mark.asyncio
    async def test_query_credential_keeps_the_tool_arguments_and_is_never_echoed(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", "QUERY-KEY-123456")
        plan = _with_credential(
            mcpcast(SPEC, name="petstore", auth=AuthMode.ENV_TOKEN), "query", "api_key"
        )
        out = tmp_path / "q"
        write_project(plan, out)
        mod = _import(out / "server.py")

        async def echo(request):
            return httpx.Response(401, text=f"bad key in {request.url}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(echo)) as http:
            client = TestClient(mod.build_server(http_client=http))
            (r,) = await client.call_tool("find_pets_by_status", {"status": "sold"})
        err = json.loads(r.text)["error"]
        assert err["code"] == "UPSTREAM_ERROR"
        assert err["message"].endswith(
            "HTTP 401: bad key in https://petstore.example/api/v3/pet/findByStatus"
            "?status=sold&api_key=[redacted]"
        )
        assert "QUERY-KEY" not in r.text

    @pytest.mark.asyncio
    async def test_query_credential_refuses_plain_http_too(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MCPCAST_ALLOW_INSECURE_HTTP", raising=False)
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", "QUERY-KEY-123456")
        plan = _with_credential(
            mcpcast(
                SPEC, name="petstore", base_url="http://api.acme.test", auth=AuthMode.ENV_TOKEN
            ),
            "query",
            "api_key",
        )
        result, requests, out = await self._one_call(tmp_path, plan)
        assert result["error"]["code"] == "UPSTREAM_INSECURE" and requests == []
        assert "MCPCAST_ALLOW_INSECURE_HTTP=1" in (out / ".env.example").read_text()

    @pytest.mark.asyncio
    async def test_api_key_tenants_present_the_custom_slot(self, tmp_path, monkeypatch):
        monkeypatch.setenv(
            "MCPCAST_CLIENT_KEYS", json.dumps({"sk-t": {"client_id": "a", "tenant_id": "acme"}})
        )
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKENS", json.dumps({"acme": "acme-subscription-key"}))
        plan = _with_credential(
            mcpcast(SPEC, name="petstore", auth=AuthMode.API_KEY),
            "header",
            "Ocp-Apim-Subscription-Key",
        )
        result, requests, out = await self._one_call(tmp_path, plan, headers={"x-api-key": "sk-t"})
        assert "error" not in result
        assert requests[0].headers["ocp-apim-subscription-key"] == "acme-subscription-key"
        assert "authorization" not in requests[0].headers
        env = (out / ".env.example").read_text()
        assert "tenant -> Ocp-Apim-Subscription-Key header value" in env
        assert 'MCPCAST_UPSTREAM_TOKENS={"acme": "<your API key>"}' in env
        # a missing tenant credential names the real slot
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKENS", "{}")
        result, _requests, _out = await self._one_call(
            tmp_path / "missing", plan, headers={"x-api-key": "sk-t"}
        )
        assert result["error"]["code"] == "UPSTREAM_AUTH_MISSING"
        assert "Ocp-Apim-Subscription-Key header value" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_authorization_header_stays_the_default(self, tmp_path, monkeypatch):
        """No scheme in the spec: the Authorization header, with the bearer hint, as before."""
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", "")
        plan = mcpcast(SPEC, name="petstore", auth=AuthMode.ENV_TOKEN)
        assert (plan.api.credential_location, plan.api.credential_name) == (
            "header",
            "Authorization",
        )
        result, requests, out = await self._one_call(tmp_path, plan)
        assert result["error"]["code"] == "UPSTREAM_AUTH_MISSING" and requests == []
        assert "as the Authorization header on every upstream call" in result["error"]["message"]
        assert "MCPCAST_UPSTREAM_TOKEN='Bearer <your API token>'" in result["error"]["message"]
        env = (out / ".env.example").read_text()
        assert "# env-token: the Authorization header value sent on every upstream call." in env
        assert "MCPCAST_UPSTREAM_TOKEN=Bearer <your API token>" in env
        config = (out / "petstore_mcp" / "config.py").read_text()
        assert 'CREDENTIAL_LOCATION = "header"' in config
        assert 'CREDENTIAL_NAME = "Authorization"' in config

    @pytest.mark.asyncio
    async def test_passthrough_relays_the_authorization_header_only(self, tmp_path):
        """Whatever the scheme says, passthrough forwards the caller's own header as such."""
        plan = mcpcast(SPEC, name="petstore", auth=AuthMode.PASSTHROUGH)
        result, requests, _out = await self._one_call(
            tmp_path, plan, meta={"authorization": "Bearer caller-token"}
        )
        assert "error" not in result
        assert requests[0].headers["authorization"] == "Bearer caller-token"
        assert str(requests[0].url) == "https://petstore.example/api/v3/pet/1"
