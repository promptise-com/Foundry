"""Tests for the Agent Readiness Score (``promptise.mcpcast.readiness``)."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from langchain_core.messages import AIMessage

from promptise.mcp.server import MCPServer
from promptise.mcpcast import mcpcast, write_project
from promptise.mcpcast.parse import extract_operations
from promptise.mcpcast.readiness import (
    NO_MOCK_STATUS,
    CallRecorder,
    EvalTask,
    TaskResult,
    ToolCall,
    base_url_override,
    credential_slot,
    evaluate,
    generate_tasks,
    grade_for,
    mock_transport,
    score,
    tools_from_server,
    write_eval,
)
from promptise.mcpcast.schema import MCPcastError, SafetyProfile

SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Petstore"},
    "servers": [{"url": "https://petstore.example/v3"}],
    "paths": {
        "/pet/{petId}": {
            "get": {
                "operationId": "getPet",
                "summary": "Find pet by ID",
                "parameters": [
                    {"name": "petId", "in": "path", "required": True, "schema": {"type": "integer"}}
                ],
                "responses": {
                    "200": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "integer"},
                                        "name": {"type": "string"},
                                    },
                                }
                            }
                        }
                    }
                },
            },
            "delete": {
                "operationId": "deletePet",
                "summary": "Delete a pet",
                "parameters": [
                    {"name": "petId", "in": "path", "required": True, "schema": {"type": "integer"}}
                ],
                "responses": {
                    "200": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"deleted": {"type": "boolean"}},
                                }
                            }
                        }
                    }
                },
            },
        },
        "/pet": {
            "post": {
                "operationId": "addPet",
                "summary": "Add a pet",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["name"],
                                "properties": {"name": {"type": "string"}},
                            }
                        }
                    }
                },
            }
        },
    },
}


def _plan(profile=SafetyProfile.FULL):
    return mcpcast(SPEC, name="petstore", profile=profile, auth="none")  # type: ignore[arg-type]


@contextmanager
def _upstream(routes: dict[str, tuple[int, dict]]) -> Iterator[tuple[int, list[str]]]:
    """A real HTTP server on ``127.0.0.1``: *routes* maps a path to ``(status, JSON body)``.

    Anything else is a 404. Yields the port and the list of ``"METHOD /path"``
    it received.
    """
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — http.server's name
            hits.append(f"GET {self.path}")
            status, body = routes.get(self.path, (404, {"error": "no such pet"}))
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: object) -> None:  # silence the test output
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1], hits
    finally:
        server.shutdown()
        server.server_close()


def _task(i, expected):
    return EvalTask(id=f"t{i}", prompt=f"task {i}", expected_tool=expected)


def _result(expected, calls, error=None):
    task = _task(expected, expected)
    return TaskResult(
        task=task,
        calls=calls,
        success=error is None and any(c.tool == expected and c.ok for c in calls),
        selected_correctly=bool(calls) and calls[0].tool == expected,
        error=error,
    )


# ---------------------------------------------------------------------------
# Scoring maths
# ---------------------------------------------------------------------------


class TestScore:
    @pytest.mark.parametrize(
        ("s", "g"),
        [(1.0, "A"), (0.9, "A"), (0.89, "B"), (0.75, "B"), (0.6, "C"), (0.5, "D"), (0.39, "F")],
    )
    def test_grade_thresholds(self, s, g):
        assert grade_for(s) == g

    def test_perfect_run(self):
        plan = _plan()
        results = [
            _result("get_pet", [ToolCall(tool="get_pet", arguments={"petId": 1})]),
            _result("add_pet", [ToolCall(tool="add_pet", arguments={"name": "x"})]),
            _result("delete_pet", [ToolCall(tool="delete_pet", arguments={"petId": 1})]),
        ]
        report = score(plan, results)
        assert report.grade == "A" and report.score == 1.0
        assert report.tasks_succeeded == 3 and report.selection_rate == 1.0
        assert (
            report.param_error_rate == 0.0
            and report.never_used == []
            and report.confused_pairs == []
        )
        assert report.fixes == []

    def test_confusion_param_errors_and_unused(self):
        plan = _plan()
        results = [
            # wrong tool first, then recovers → success but not selected correctly
            _result(
                "get_pet",
                [
                    ToolCall(tool="add_pet", arguments={}, ok=False, error_code="VALIDATION_ERROR"),
                    ToolCall(tool="get_pet", arguments={"petId": 1}),
                ],
            ),
            _result("get_pet", [ToolCall(tool="add_pet", arguments={"name": "x"})]),
            _result("add_pet", [ToolCall(tool="add_pet", arguments={"name": "x"})]),
            _result("get_pet", [], error="RuntimeError: boom"),
        ]
        report = score(plan, results)
        assert report.tasks_succeeded == 2 and report.selection_rate == 0.25
        assert report.param_error_rate == 0.25
        assert report.never_used == [] and report.not_covered == ["delete_pet"]
        (pair,) = report.confused_pairs
        assert (pair.expected, pair.chosen, pair.count) == ("get_pet", "add_pet", 2)
        assert report.grade == "D"
        text = "\n".join(report.fixes)
        assert "`get_pet` vs `add_pet` are ambiguous" in text and "2/3 runs" in text
        assert "`add_pet`: 1 parameter error" in text
        assert (
            "`name` has no description and no example" not in text
        )  # add_pet has an example for name
        assert "1 tool not covered by any task: `delete_pet`" in text
        assert "crashed: RuntimeError: boom" in text

    def test_crashed_tasks_lead_the_fixes_and_never_mark_tools_unused(self):
        """A task that crashed before the agent answered says nothing about the
        tool it targeted — the report must show the crash, not 'never used'."""
        plan = _plan()
        results = [
            _result("get_pet", [], error="GraphExecutionError: failed at node 'reason': 401"),
            _result("delete_pet", [], error="GraphExecutionError: failed at node 'reason': 401"),
            _result("add_pet", [ToolCall(tool="add_pet", arguments={"name": "x"})]),
        ]
        report = score(plan, results)
        assert report.never_used == [] and report.not_covered == []
        assert report.fixes[0].startswith("✗ 2 of 3 tasks crashed: GraphExecutionError")
        assert "first of several" in report.fixes[0]
        assert not any("never used" in fix for fix in report.fixes)

    def test_auth_rejections_are_named_as_fixes(self):
        plan = _plan()
        results = [
            _result(
                "get_pet",
                [
                    ToolCall(
                        tool="get_pet", arguments={}, ok=False, error_code="AUTHENTICATION_ERROR"
                    ),
                    ToolCall(
                        tool="get_pet", arguments={}, ok=False, error_code="AUTHENTICATION_ERROR"
                    ),
                ],
            ),
            _result(
                "add_pet",
                [
                    ToolCall(
                        tool="add_pet", arguments={}, ok=False, error_code="UPSTREAM_AUTH_MISSING"
                    )
                ],
            ),
            _result(
                "delete_pet",
                [
                    ToolCall(
                        tool="delete_pet",
                        arguments={},
                        ok=False,
                        error_code="UPSTREAM_ERROR",
                        details_status=401,
                    )
                ],
            ),
        ]
        text = "\n".join(score(plan, results).fixes)
        assert "2 calls were rejected by the server's own auth (AUTHENTICATION_ERROR)" in text
        assert "MCPCAST_EVAL_HEADERS" in text and "MCPCAST_CLIENT_KEYS" in text
        assert "2 calls reached the API without a valid upstream credential" in text
        assert "MCPCAST_EVAL_AUTHORIZATION" in text

    def test_markdown_and_summary(self):
        plan = _plan()
        report = score(
            plan, [_result("get_pet", [ToolCall(tool="get_pet", arguments={"petId": 1})])]
        )
        md = report.render_markdown()
        assert md.startswith("# Agent Readiness: ")
        assert "| t_get_pet | task get_pet | `get_pet` | get_pet | ✓ |" in md.replace(
            "tget_pet", "t_get_pet"
        )
        assert report.render_summary().startswith("Agent Readiness: ")

    def test_empty_results(self):
        report = score(_plan(), [])
        assert report.grade == "F" and report.tasks_total == 0

    def test_insecure_and_unrouted_calls_are_named_as_fixes(self):
        """A plain-http upstream refuses every credentialed call before it is
        sent, and a request the evaluation transport had no route for is
        neither live nor mocked — both are said, never folded into the grade."""
        plan = _plan()
        results = [
            _result(
                "get_pet",
                [ToolCall(tool="get_pet", arguments={}, ok=False, error_code="UPSTREAM_INSECURE")],
            ),
            _result(
                "add_pet",
                [
                    ToolCall(
                        tool="add_pet",
                        arguments={},
                        ok=False,
                        error_code="UPSTREAM_ERROR",
                        details_status=NO_MOCK_STATUS,
                    )
                ],
            ),
        ]
        report = score(plan, results, unmatched=["POST /api/v9/pet", "POST /api/v9/pet"])
        text = "\n".join(report.fixes)
        assert "1 call was refused before leaving the server (UPSTREAM_INSECURE)" in text
        assert "MCPCAST_ALLOW_INSECURE_HTTP=1" in text and "use https" in text
        assert "2 calls reached no mocked route (POST /api/v9/pet)" in text
        assert "check MCPCAST_BASE_URL" in text
        assert "identifiers it does not recognise" not in text  # a 502 is not a stale id
        assert report.tasks_succeeded == 0  # neither call counts as a success
        # nothing unrouted, nothing to say
        assert not any("no mocked route" in f for f in score(plan, results).fixes)

    def test_upstream_auth_hint_names_the_credential_slot(self):
        """The hint says where the API expects the credential — the header or
        query parameter the plan records — not always ``Authorization``."""
        rejected = [
            _result(
                "get_pet",
                [
                    ToolCall(
                        tool="get_pet", arguments={}, ok=False, error_code="UPSTREAM_AUTH_MISSING"
                    )
                ],
            )
        ]

        def with_slot(auth, location, name):
            plan = mcpcast(SPEC, name="petstore", profile=SafetyProfile.FULL, auth=auth)
            api = plan.api.model_validate(
                {**plan.api.model_dump(), "credential_location": location, "credential_name": name}
            )
            return plan.model_copy(update={"api": api})

        header_key = with_slot("env-token", "header", "X-API-Key")
        assert credential_slot(header_key) == "X-API-Key header"
        assert "set MCPCAST_EVAL_AUTHORIZATION to a real X-API-Key header value" in "\n".join(
            score(header_key, rejected).fixes
        )
        query_key = with_slot("api-key", "query", "api_key")
        assert credential_slot(query_key) == "api_key query parameter"
        assert "a real api_key query parameter value" in "\n".join(score(query_key, rejected).fixes)
        # passthrough relays the caller's Authorization header, whatever the spec says
        relayed = with_slot("passthrough", "header", "Authorization")
        assert credential_slot(relayed) == "Authorization header"
        assert credential_slot(_plan()) == "Authorization header"  # the default slot


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------


class TestGenerateTasks:
    @pytest.mark.asyncio
    async def test_filters_unknown_tools_and_dedupes_ids(self):
        async def complete(system, user):
            assert "get_pet" in user and "petstore" in user
            return json.dumps(
                {
                    "tasks": [
                        {"id": "t1", "prompt": "Show pet 1", "expected_tool": "get_pet"},
                        {"id": "t1", "prompt": "Add rex", "expected_tool": "add_pet"},
                        {"id": "t3", "prompt": "Fly to the moon", "expected_tool": "launch_rocket"},
                        {"prompt": "", "expected_tool": "get_pet"},
                        {"id": "t5", "prompt": "Delete pet 2", "expected_tool": "delete_pet"},
                    ]
                }
            )

        tasks = await generate_tasks(_plan(), count=2, complete=complete)
        assert [(t.id, t.expected_tool) for t in tasks] == [("t1", "get_pet"), ("t2", "add_pet")]

    @pytest.mark.asyncio
    async def test_no_usable_tasks_raises(self):
        async def complete(system, user):
            return '{"tasks": [{"id": "x", "prompt": "p", "expected_tool": "nope"}]}'

        with pytest.raises(MCPcastError, match="no task"):
            await generate_tasks(_plan(), complete=complete)

        async def garbage(system, user):
            return "no json"

        with pytest.raises(MCPcastError, match="task generation failed"):
            await generate_tasks(_plan(), complete=garbage)

    @pytest.mark.asyncio
    async def test_plan_without_tools(self):
        with pytest.raises(MCPcastError, match="no tools"):
            await generate_tasks(
                _plan(SafetyProfile.READ_ONLY).model_copy(update={"tools": []}), complete=None
            )


# ---------------------------------------------------------------------------
# Server bridge + mocks
# ---------------------------------------------------------------------------


class TestBridge:
    @pytest.mark.asyncio
    async def test_tools_from_server_record_calls_and_errors(self):
        server = MCPServer(name="t")

        @server.tool()
        async def add(a: int, b: int) -> int:
            """Add."""
            return a + b

        recorder = CallRecorder()
        (tool,) = await tools_from_server(server, recorder=recorder)
        assert tool.name == "add" and tool.description == "Add."
        recorder.begin("t1")
        assert await tool.ainvoke({"a": 1, "b": 2}) == "3"
        out = await tool.ainvoke({"a": "x", "b": 2})
        assert "VALIDATION_ERROR" in out
        calls = recorder.calls_for("t1")
        assert [(c.tool, c.ok, c.error_code) for c in calls] == [
            ("add", True, None),
            ("add", False, "VALIDATION_ERROR"),
        ]
        assert len(recorder.all_calls) == 2

    @pytest.mark.asyncio
    async def test_mock_transport_routes(self):
        plan = _plan()
        ops = extract_operations(SPEC)
        transport = mock_transport(plan, operations=ops, live_reads=False)
        async with httpx.AsyncClient(transport=transport) as client:
            r = await client.get("https://petstore.example/v3/pet/7")
            assert r.json() == {
                "id": 1,
                "name": "example",
            }  # spec-derived, GET mocked (live_reads=False)
            r = await client.delete("https://petstore.example/v3/pet/7")
            assert r.json() == {"deleted": True}
            r = await client.post("https://petstore.example/v3/pet", json={"name": "x"})
            assert r.json() == {"ok": True, "mock": "addPet"}  # no response schema → echo
            # No route in the plan: never an invented success — a structured
            # non-2xx naming the path, and the transport remembers it.
            r = await client.post("https://petstore.example/v3/unknown")
            assert r.status_code == NO_MOCK_STATUS
            assert r.json()["error"]["code"] == "EVAL_NO_MOCK"
            assert "POST /v3/unknown" in r.json()["error"]["message"]
            assert "'/v3'" in r.json()["error"]["message"]  # the base path it expected
            assert transport.unmatched == ["POST /v3/unknown"]

    @pytest.mark.asyncio
    async def test_live_reads_use_real_transport(self, monkeypatch):
        seen = []

        async def fake_real(self, request):
            seen.append(request.url.path)
            return httpx.Response(200, json={"live": True}, request=request)

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fake_real)
        transport = mock_transport(_plan(), live_reads=True)
        async with httpx.AsyncClient(transport=transport) as client:
            assert (await client.get("https://petstore.example/v3/pet/1")).json() == {"live": True}
            assert (await client.delete("https://petstore.example/v3/pet/1")).json()["ok"] is True
        assert seen == ["/v3/pet/1"]

    @pytest.mark.asyncio
    async def test_transport_expects_routes_where_mcpcast_base_url_sends_them(self, monkeypatch):
        """``MCPCAST_BASE_URL`` moves every route — the plan's and an operation's
        own server alike — exactly as the generated ``config.py`` does; a
        read under the new prefix is live, and the old prefix is no route."""
        seen = []

        async def fake_real(self, request):
            seen.append(f"{request.method} {request.url.path}")
            return httpx.Response(200, json={"live": True}, request=request)

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fake_real)
        spec = {
            **SPEC,
            "paths": {
                **SPEC["paths"],
                "/legacy/ping": {
                    "get": {
                        "operationId": "legacyPing",
                        "servers": [{"url": "https://legacy.example/old"}],
                    }
                },
            },
        }
        plan = mcpcast(spec, name="petstore", profile=SafetyProfile.FULL, auth="none")  # type: ignore[arg-type]
        assert plan.tool("legacy_ping").routes[0].base_url == "https://legacy.example/old"
        override = "https://staging.example.invalid/api/v9"
        transport = mock_transport(plan, live_reads=True, base_url=override)
        async with httpx.AsyncClient(transport=transport) as client:
            live = await client.get(f"{override}/pet/1")
            assert live.json() == {"live": True}
            assert (await client.get(f"{override}/legacy/ping")).json() == {"live": True}
            assert (await client.delete(f"{override}/pet/1")).json() == {
                "ok": True,
                "mock": "deletePet",
            }
            # the plan's own prefix is not where the server sends anything now
            stale = await client.get("https://petstore.example/v3/pet/1")
            assert stale.status_code == NO_MOCK_STATUS
            assert "'/api/v9'" in stale.json()["error"]["message"]
            old_server = await client.get("https://legacy.example/old/legacy/ping")
            assert old_server.status_code == NO_MOCK_STATUS
        assert seen == ["GET /api/v9/pet/1", "GET /api/v9/legacy/ping"]
        assert transport.unmatched == ["GET /v3/pet/1", "GET /old/legacy/ping"]
        # without an override an operation's own server is honoured, and the
        # plan's base is not applied to it
        plain = mock_transport(plan, live_reads=False)
        async with httpx.AsyncClient(transport=plain) as client:
            assert (await client.get("https://legacy.example/old/legacy/ping")).status_code == 200
            assert (await client.get("https://petstore.example/v3/legacy/ping")).status_code == (
                NO_MOCK_STATUS
            )

    def test_base_url_override_reads_the_environment_like_config_py(self, monkeypatch):
        monkeypatch.delenv("MCPCAST_BASE_URL", raising=False)
        assert base_url_override() is None
        monkeypatch.setenv("MCPCAST_BASE_URL", "  ")
        assert base_url_override() is None
        monkeypatch.setenv("MCPCAST_BASE_URL", " https://staging.example/api/ ")
        assert base_url_override() == "https://staging.example/api"


# ---------------------------------------------------------------------------
# End to end with a scripted agent model
# ---------------------------------------------------------------------------


def _scripted_model(turns: list[list[dict] | str]):
    """A chat model whose successive ``ainvoke`` calls return scripted turns."""
    messages = []
    for i, turn in enumerate(turns):
        if isinstance(turn, str):
            messages.append(AIMessage(content=turn))
        else:
            msg = AIMessage(content="")
            msg.tool_calls = [
                {"name": tc["name"], "args": tc["args"], "id": f"call_{i}_{j}", "type": "tool_call"}
                for j, tc in enumerate(turn)
            ]
            messages.append(msg)
    model = MagicMock(spec=["ainvoke", "bind_tools", "with_structured_output"])
    model.ainvoke = AsyncMock(side_effect=messages)
    model.bind_tools = MagicMock(return_value=model)
    return model


def _import(path):
    from promptise.mcpcast import load_generated_server

    return load_generated_server(path)


class TestEvaluateEndToEnd:
    @pytest.mark.asyncio
    async def test_evaluate_runs_agent_against_generated_server(self, tmp_path):
        plan = _plan()
        out = tmp_path / "proj"
        write_project(plan, out)
        module = _import(out / "server.py")
        ops = extract_operations(SPEC)
        tasks = [_task(1, "get_pet"), _task(2, "delete_pet"), _task(3, "add_pet")]
        model = _scripted_model(
            [
                [{"name": "get_pet", "args": {"petId": 1}}],
                "Pet 1 is example.",
                [{"name": "delete_pet", "args": {"petId": 1}}],
                "Deleted.",
                [{"name": "get_pet", "args": {"petId": "not-a-number"}}],
                "Could not add.",
            ]
        )
        report = await evaluate(
            plan, module.build_server, model=model, tasks=tasks, operations=ops, live_reads=False
        )
        assert report.tasks_total == 3 and report.tasks_succeeded == 2
        assert report.selection_rate == pytest.approx(2 / 3, abs=1e-3)
        assert report.never_used == ["add_pet"]
        assert [(p.expected, p.chosen) for p in report.confused_pairs] == [("add_pet", "get_pet")]
        assert report.param_error_rate == pytest.approx(1 / 3, abs=1e-3)
        assert report.results[1].calls[0].ok  # destructive call auto-approved against the mock
        assert report.results[0].answer == "Pet 1 is example."
        tasks_path, report_path = write_eval(report, tasks, out)
        assert tasks_path.read_text(encoding="utf-8").count("expected_tool") == 3
        assert report_path.read_text(encoding="utf-8").startswith("# Agent Readiness: ")

    @pytest.mark.asyncio
    async def test_provider_failure_is_recorded_per_task(self, tmp_path):
        """A rejected model credential surfaces as the task's error (the engine
        raises), never as a tool-design verdict."""
        plan = _plan()
        out = tmp_path / "proj"
        write_project(plan, out)
        module = _import(out / "server.py")
        model = _scripted_model([[{"name": "get_pet", "args": {"petId": 1}}], "Pet 1."])
        model.ainvoke = AsyncMock(
            side_effect=[
                PermissionError("Error code: 401 - Incorrect API key provided"),
                *model.ainvoke.side_effect,
            ]
        )
        report = await evaluate(
            plan,
            module.build_server,
            model=model,
            tasks=[_task(1, "delete_pet"), _task(2, "get_pet")],
            live_reads=False,
        )
        crashed, ran = report.results
        assert crashed.error is not None
        assert "failed at node 'reason'" in crashed.error
        assert "Incorrect API key" in crashed.error
        assert crashed.success is False and crashed.calls == []
        assert ran.success is True
        assert report.never_used == []  # delete_pet's task never ran — no verdict
        assert report.fixes[0].startswith("✗ 1 of 2 tasks crashed: GraphExecutionError")

    @pytest.mark.asyncio
    async def test_evaluate_raises_when_no_task_could_run(self, tmp_path):
        plan = _plan()
        out = tmp_path / "proj"
        write_project(plan, out)
        module = _import(out / "server.py")
        model = _scripted_model(["unused"])
        model.ainvoke = AsyncMock(side_effect=PermissionError("Incorrect API key provided"))
        with pytest.raises(MCPcastError, match="could not run: all 2 tasks crashed") as info:
            await evaluate(
                plan,
                module.build_server,
                model=model,
                tasks=[_task(1, "get_pet"), _task(2, "add_pet")],
                live_reads=False,
            )
        assert "Incorrect API key" in str(info.value)

    @pytest.mark.asyncio
    async def test_evaluate_rejects_unknown_task_tools(self, tmp_path):
        plan = _plan()
        with pytest.raises(MCPcastError, match="not in the plan"):
            await evaluate(plan, lambda **kw: None, tasks=[_task(1, "ghost")])
        with pytest.raises(MCPcastError, match="no evaluation tasks"):
            await evaluate(plan, lambda **kw: None, tasks=[])

    @pytest.mark.asyncio
    async def test_live_reads_follow_mcpcast_base_url_to_the_real_upstream(
        self, tmp_path, monkeypatch
    ):
        """Under ``MCPCAST_BASE_URL`` with another path prefix the reads used to
        be answered by a 200 echo mock — grade A on nothing. They must reach
        the upstream, whose 404 for a made-up id then costs the grade."""
        with _upstream({"/api/v9/pet/1": (200, {"id": 1, "name": "rex"})}) as (port, hits):
            monkeypatch.setenv("MCPCAST_BASE_URL", f"http://127.0.0.1:{port}/api/v9")
            plan = _plan(SafetyProfile.READ_ONLY)
            out = tmp_path / "proj"
            write_project(plan, out)
            module = _import(out / "server.py")  # config.py binds MCPCAST_BASE_URL at import
            model = _scripted_model(
                [
                    [{"name": "get_pet", "args": {"petId": 1}}],
                    "Pet 1 is rex.",
                    [{"name": "get_pet", "args": {"petId": 999}}],
                    "No such pet.",
                ]
            )
            report = await evaluate(
                plan,
                module.build_server,
                model=model,
                tasks=[_task(1, "get_pet"), _task(2, "get_pet")],
                operations=extract_operations(SPEC),
            )
        assert hits == ["GET /api/v9/pet/1", "GET /api/v9/pet/999"]  # both went live
        first, second = report.results
        assert first.success and first.answer == "Pet 1 is rex."
        assert not second.success and second.calls[0].details_status == 404
        assert report.tasks_succeeded == 1 and report.grade != "A"
        assert any("identifiers it does not recognise" in fix for fix in report.fixes)
        assert not any("no mocked route" in fix for fix in report.fixes)

    @pytest.mark.asyncio
    async def test_a_request_the_transport_has_no_route_for_is_never_a_success(
        self, tmp_path, monkeypatch
    ):
        """The server (built with no override) sends to the plan's base; the
        evaluation (started under another ``MCPCAST_BASE_URL``) expects the
        override's prefix. Nothing matches — and nothing is invented: the
        call fails, and the report says which requests reached no route."""
        monkeypatch.delenv("MCPCAST_BASE_URL", raising=False)
        plan = _plan(SafetyProfile.READ_ONLY)
        out = tmp_path / "proj"
        write_project(plan, out)
        module = _import(out / "server.py")
        monkeypatch.setenv("MCPCAST_BASE_URL", "https://staging.example.invalid/api/v9")
        model = _scripted_model([[{"name": "get_pet", "args": {"petId": 1}}], "Done."])
        report = await evaluate(
            plan, module.build_server, model=model, tasks=[_task(1, "get_pet")], live_reads=False
        )
        (result,) = report.results
        (call,) = result.calls
        assert call.ok is False and call.error_code == "UPSTREAM_ERROR"
        assert call.details_status == NO_MOCK_STATUS
        assert result.success is False and report.tasks_succeeded == 0
        assert any(
            "1 call reached no mocked route (GET /v3/pet/1)" in fix and "MCPCAST_BASE_URL" in fix
            for fix in report.fixes
        )


class TestMocksByRisk:
    @pytest.mark.asyncio
    async def test_live_split_follows_tool_risk_not_method(self, monkeypatch):
        seen = []

        async def fake_real(self, request):
            seen.append(f"{request.method} {request.url.path}")
            return httpx.Response(200, json={"live": True}, request=request)

        monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fake_real)
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Mixed"},
            "servers": [{"url": "https://m.example/v1"}],
            "paths": {
                "/pets/search": {"post": {"operationId": "searchPets"}},  # read-classified POST
                "/admin/pets": {"get": {"operationId": "adminPets"}},  # escalated GET
                "/pets/{id}": {
                    "get": {
                        "operationId": "getPet",
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
                "/pets/{id}/photos": {
                    "post": {
                        "operationId": "addPhoto",
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
        plan = mcpcast(spec, name="mixed", profile=SafetyProfile.FULL, auth="none")  # type: ignore[arg-type]
        assert (
            plan.tool("search_pets").risk.value == "read"
            and plan.tool("admin_pets").risk.value == "write"
        )
        transport = mock_transport(plan, live_reads=True)
        async with httpx.AsyncClient(transport=transport) as client:
            assert (await client.post("https://m.example/v1/pets/search")).json() == {"live": True}
            assert (await client.get("https://m.example/v1/admin/pets")).json()[
                "mock"
            ] == "adminPets"
            assert (await client.get("https://m.example/v1/pets/7")).json() == {"live": True}
            # anchored: a longer path that merely ends with a template is not that
            # route — and not a success either
            other = await client.get("https://m.example/v1/other/pets/7")
            assert other.status_code == NO_MOCK_STATUS
            assert other.json()["error"]["code"] == "EVAL_NO_MOCK"
            assert transport.unmatched == ["GET /v1/other/pets/7"]
            # most specific template wins
            assert (await client.post("https://m.example/v1/pets/7/photos")).json()[
                "mock"
            ] == "addPhoto"
        assert seen == ["POST /v1/pets/search", "GET /v1/pets/7"]


class TestEvalCredentials:
    @pytest.mark.asyncio
    async def test_api_key_and_env_token_servers_are_callable_during_eval(
        self, tmp_path, monkeypatch
    ):
        for key in (
            "MCPCAST_CLIENT_KEYS",
            "MCPCAST_UPSTREAM_TOKENS",
            "MCPCAST_UPSTREAM_TOKEN",
            "MCPCAST_EVAL_HEADERS",
        ):
            monkeypatch.delenv(key, raising=False)
        from promptise.mcpcast import AuthMode

        for auth in (AuthMode.API_KEY, AuthMode.ENV_TOKEN):
            plan = mcpcast(SPEC, name="petstore", profile=SafetyProfile.FULL, auth=auth)
            out = tmp_path / auth.value
            write_project(plan, out)
            module = _import(out / "server.py")
            model = _scripted_model([[{"name": "get_pet", "args": {"petId": 1}}], "ok"])
            report = await evaluate(
                plan,
                module.build_server,
                model=model,
                tasks=[_task(1, "get_pet")],
                live_reads=False,
            )
            assert report.tasks_succeeded == 1, auth
            assert report.results[0].calls[0].ok, auth
        assert "MCPCAST_CLIENT_KEYS" not in __import__("os").environ  # restored

    @pytest.mark.asyncio
    async def test_operator_keys_are_merged_not_bypassed(self, tmp_path, monkeypatch):
        """With the operator's real MCPCAST_CLIENT_KEYS set (a .env inside the
        project), every evaluation call used to be AUTHENTICATION_ERROR."""
        import os

        from promptise.mcpcast import AuthMode

        real_keys = json.dumps({"real-key": {"client_id": "ops", "tenant_id": "acme", "roles": []}})
        real_tokens = json.dumps({"acme": "Bearer acme-upstream"})
        monkeypatch.setenv("MCPCAST_CLIENT_KEYS", real_keys)
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKENS", real_tokens)
        monkeypatch.delenv("MCPCAST_EVAL_HEADERS", raising=False)
        monkeypatch.delenv("MCPCAST_EVAL_AUTHORIZATION", raising=False)

        plan = mcpcast(SPEC, name="petstore", profile=SafetyProfile.FULL, auth=AuthMode.API_KEY)
        out = tmp_path / "apikey"
        write_project(plan, out)
        module = _import(out / "server.py")
        model = _scripted_model([[{"name": "get_pet", "args": {"petId": 1}}], "ok"])

        from promptise.mcpcast.readiness import _eval_credentials

        with _eval_credentials(plan):
            merged_keys = json.loads(os.environ["MCPCAST_CLIENT_KEYS"])
            merged_tokens = json.loads(os.environ["MCPCAST_UPSTREAM_TOKENS"])
            assert merged_keys["real-key"]["tenant_id"] == "acme"  # operator's key kept
            assert merged_keys["mcpcast-eval"]["tenant_id"] == "mcpcast-eval"
            assert merged_tokens == {
                "acme": "Bearer acme-upstream",
                "mcpcast-eval": "Bearer mcpcast-eval",
            }
        # restored to the exact previous values, never popped
        assert os.environ["MCPCAST_CLIENT_KEYS"] == real_keys
        assert os.environ["MCPCAST_UPSTREAM_TOKENS"] == real_tokens

        report = await evaluate(
            plan, module.build_server, model=model, tasks=[_task(1, "get_pet")], live_reads=False
        )
        (result,) = report.results
        assert result.calls[0].ok and result.calls[0].error_code is None
        assert report.tasks_succeeded == 1
        assert os.environ["MCPCAST_CLIENT_KEYS"] == real_keys

    def test_operator_configured_eval_identity_and_env_token_are_left_alone(self, monkeypatch):
        import os

        from promptise.mcpcast import AuthMode
        from promptise.mcpcast.readiness import _eval_credentials

        own = json.dumps({"mcpcast-eval": {"client_id": "me", "tenant_id": "mine", "roles": []}})
        monkeypatch.setenv("MCPCAST_CLIENT_KEYS", own)
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKENS", json.dumps({"mine": "Bearer real"}))
        plan = mcpcast(SPEC, name="petstore", profile=SafetyProfile.FULL, auth=AuthMode.API_KEY)
        with _eval_credentials(plan):
            assert os.environ["MCPCAST_CLIENT_KEYS"] == own  # not overwritten
            assert json.loads(os.environ["MCPCAST_UPSTREAM_TOKENS"])["mcpcast-eval"].startswith(
                "Bearer"
            )
        assert os.environ["MCPCAST_UPSTREAM_TOKENS"] == json.dumps({"mine": "Bearer real"})

        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKEN", "Bearer operator")
        plan = mcpcast(SPEC, name="petstore", profile=SafetyProfile.FULL, auth=AuthMode.ENV_TOKEN)
        with _eval_credentials(plan):
            assert os.environ["MCPCAST_UPSTREAM_TOKEN"] == "Bearer operator"
        assert os.environ["MCPCAST_UPSTREAM_TOKEN"] == "Bearer operator"

    def test_malformed_operator_keys_are_an_error(self, monkeypatch):
        from promptise.mcpcast import AuthMode
        from promptise.mcpcast.readiness import _eval_credentials

        monkeypatch.setenv("MCPCAST_CLIENT_KEYS", "{not json")
        plan = mcpcast(SPEC, name="petstore", profile=SafetyProfile.FULL, auth=AuthMode.API_KEY)
        with pytest.raises(MCPcastError, match="MCPCAST_CLIENT_KEYS must be a JSON object"):
            with _eval_credentials(plan):
                pass

    def test_a_failing_second_merge_restores_the_first(self, monkeypatch):
        """Valid client keys plus malformed upstream tokens: the error must not
        leave the evaluation identity merged into MCPCAST_CLIENT_KEYS."""
        import os

        from promptise.mcpcast import AuthMode
        from promptise.mcpcast.readiness import _eval_credentials

        real_keys = json.dumps({"real-key": {"client_id": "ops", "tenant_id": "acme", "roles": []}})
        monkeypatch.setenv("MCPCAST_CLIENT_KEYS", real_keys)
        monkeypatch.setenv("MCPCAST_UPSTREAM_TOKENS", "{bad")
        plan = mcpcast(SPEC, name="petstore", profile=SafetyProfile.FULL, auth=AuthMode.API_KEY)
        with pytest.raises(MCPcastError, match="MCPCAST_UPSTREAM_TOKENS must be a JSON object"):
            with _eval_credentials(plan):
                pass
        assert os.environ["MCPCAST_CLIENT_KEYS"] == real_keys
        assert os.environ["MCPCAST_UPSTREAM_TOKENS"] == "{bad"

    def test_placeholder_credential_fits_the_slot_the_api_expects(self, monkeypatch):
        """A bearer token for the Authorization header; a bare key for an API
        key header or query parameter — never a ``<placeholder>`` the
        generated server would refuse before sending."""
        import os

        from promptise.mcpcast import AuthMode
        from promptise.mcpcast.readiness import _eval_credentials

        for key in ("MCPCAST_UPSTREAM_TOKEN", "MCPCAST_UPSTREAM_TOKENS", "MCPCAST_CLIENT_KEYS"):
            monkeypatch.delenv(key, raising=False)
        monkeypatch.delenv("MCPCAST_EVAL_AUTHORIZATION", raising=False)
        plan = mcpcast(SPEC, name="petstore", profile=SafetyProfile.FULL, auth=AuthMode.ENV_TOKEN)
        with _eval_credentials(plan):
            assert os.environ["MCPCAST_UPSTREAM_TOKEN"] == "Bearer mcpcast-eval"
        keyed = plan.model_copy(
            update={
                "api": plan.api.model_validate(
                    {
                        **plan.api.model_dump(),
                        "credential_location": "header",
                        "credential_name": "X-API-Key",
                    }
                )
            }
        )
        with _eval_credentials(keyed):
            assert os.environ["MCPCAST_UPSTREAM_TOKEN"] == "mcpcast-eval"
        assert "MCPCAST_UPSTREAM_TOKEN" not in os.environ
        monkeypatch.setenv("MCPCAST_EVAL_AUTHORIZATION", "real-key-value")
        with _eval_credentials(keyed):
            assert os.environ["MCPCAST_UPSTREAM_TOKEN"] == "real-key-value"


class TestExpectedToolErrorsIsNotSuccess:
    @pytest.mark.asyncio
    async def test_expected_tool_with_validation_error_fails_task(self, tmp_path):
        plan = _plan()
        out = tmp_path / "p"
        write_project(plan, out)
        module = _import(out / "server.py")
        model = _scripted_model(
            [[{"name": "get_pet", "args": {"petId": "not-a-number"}}], "gave up"]
        )
        report = await evaluate(
            plan, module.build_server, model=model, tasks=[_task(1, "get_pet")], live_reads=False
        )
        (result,) = report.results
        assert [(c.tool, c.ok, c.error_code) for c in result.calls] == [
            ("get_pet", False, "VALIDATION_ERROR")
        ]
        assert result.success is False and result.selected_correctly is True
        assert report.tasks_succeeded == 0 and report.grade == "D"


def test_task_ids_are_unique_even_when_model_repeats_them():
    import asyncio

    async def complete(system, user):
        return json.dumps(
            {
                "tasks": [
                    {"id": "t2", "prompt": "a", "expected_tool": "get_pet"},
                    {"id": "t2", "prompt": "b", "expected_tool": "get_pet"},
                    {"id": "t1", "prompt": "c", "expected_tool": "get_pet"},
                ]
            }
        )

    tasks = asyncio.run(generate_tasks(_plan(), count=5, complete=complete))
    assert [t.id for t in tasks] == ["t2", "t1", "t3"]


class TestStaleExampleIdentifiers:
    """A 404 on a read is usually the plan's own example teaching a bad id."""

    def test_upstream_404_on_a_read_names_the_examples(self):
        plan = _plan()
        results = [
            _result(
                "get_pet",
                [
                    ToolCall(
                        tool="get_pet", ok=False, error_code="UPSTREAM_ERROR", details_status=404
                    )
                ],
            ),
            _result(
                "add_pet",
                [
                    ToolCall(
                        tool="add_pet", ok=False, error_code="UPSTREAM_ERROR", details_status=500
                    )
                ],
            ),
        ]
        text = "\n".join(score(plan, results).fixes)
        assert "`get_pet` called the API with identifiers it does not recognise" in text
        assert "add_pet` called the API with identifiers" not in text  # a 500 is not a bad id

    @pytest.mark.asyncio
    async def test_status_is_recorded_from_the_tool_error(self):
        server = MCPServer(name="t")

        @server.tool()
        async def fetch(pet_id: str) -> dict:
            """Fetch."""
            from promptise.mcp.server import ToolError

            raise ToolError("gone", code="UPSTREAM_ERROR", details={"status": 404})

        recorder = CallRecorder()
        (tool,) = await tools_from_server(server, recorder=recorder)
        recorder.begin("t1")
        await tool.ainvoke({"pet_id": "ORD-1007"})
        (call,) = recorder.calls_for("t1")
        assert call.error_code == "UPSTREAM_ERROR" and call.details_status == 404
