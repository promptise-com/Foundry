"""Tests for the ``promptise mcpcast`` guided setup (``promptise.mcpcast.wizard``).

The plain helpers are tested directly; the Textual app is driven headlessly
with ``App.run_test`` — every step, every validation message, the model
path with a scripted completer, and the CLI's non-interactive guard.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from typer.testing import CliRunner

from promptise.cli import app as cli
from promptise.mcpcast import MCPcastPlan, mcpcast
from promptise.mcpcast.schema import ApprovalMode, AuthMode, MCPcastError, SafetyProfile
from promptise.mcpcast.wizard import (
    DEFAULT_MODEL,
    STEP_NAMES,
    Candidate,
    Detection,
    MCPcastWizard,
    ParsedSpec,
    Usage,
    WizardSettings,
    detect_local_apis,
    equivalent_command,
    preview_profile,
    probe_local_apis,
    public_source,
    quote_argument,
    recommended_auth,
    review_warnings,
)

SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Widgets API", "version": "2.1", "description": "Widgets for everyone."},
    "servers": [{"url": "https://w.example.com"}],
    "paths": {
        "/widgets": {
            "get": {"operationId": "listWidgets", "summary": "List widgets"},
            "post": {"operationId": "createWidget", "summary": "Create widget"},
        },
        "/widgets/{id}": {
            "get": {
                "operationId": "getWidget",
                "summary": "Get a widget",
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
            },
            "delete": {
                "operationId": "deleteWidget",
                "parameters": [
                    {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
                ],
            },
        },
    },
}

PROPOSAL = {
    "tools": [
        {
            "name": "find_widgets",
            "description": "List or look up widgets. Use before changing one.",
            "risk": "read",
            "operations": ["getWidget", "listWidgets"],
            "example": {"id": "w_1"},
        },
        {
            "name": "create_widget",
            "description": "Create a widget. Ask the user to confirm the details first.",
            "risk": "write",
            "operations": ["createWidget"],
        },
    ],
    "dropped": [{"operation_id": "deleteWidget", "reason": "destructive"}],
}


@pytest.fixture
def spec_file(tmp_path: Path) -> Path:
    path = tmp_path / "openapi.json"
    path.write_text(json.dumps(SPEC), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _scripted(*responses: object):
    queue = [r if isinstance(r, str) else json.dumps(r) for r in responses]

    async def complete(system: str, user: str) -> str:
        assert queue, "model called more times than scripted"
        return queue.pop(0)

    return complete


async def _settle(pilot, seconds: float = 0.3) -> None:
    """Let thread workers (spec loading, detection) finish and messages drain."""
    await pilot.pause(seconds)
    await pilot.pause()


async def _until(pilot, ready, timeout: float = 10.0) -> None:
    """Pause until ``ready()`` holds — for work whose duration the host decides
    (a refused connection takes a few ms on Linux and seconds on Windows)."""
    deadline = time.monotonic() + timeout
    while not ready():
        assert time.monotonic() < deadline, "condition not met in time"
        await pilot.pause(0.1)
    await pilot.pause()


def _plain(text: str) -> str:
    """*text* without escape sequences and with Rich wrapping collapsed."""
    return re.sub(r"[│╭╮╰╯─\s]+", " ", re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text))


def _text(app: MCPcastWizard, selector: str) -> str:
    return str(app.query_one(selector).content)


Responder = Callable[[BaseHTTPRequestHandler], None]
"""Answers one ``GET`` on the loopback test server (``handler.path`` says which)."""


@contextmanager
def _loopback(respond: Responder) -> Iterator[int]:
    """A real HTTP server on ``127.0.0.1:<port>`` answering every GET through *respond*."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 — http.server's name
            respond(self)

        def log_message(self, *args: object) -> None:  # silence the test output
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    server.block_on_close = False  # a handler still trickling must not hold the test
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


def _send(handler: BaseHTTPRequestHandler, body: bytes, status: int = 200, **headers: str) -> None:
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    for name, value in headers.items():
        handler.send_header(name.replace("_", "-"), value)
    handler.end_headers()
    handler.wfile.write(body)


def _trickle(period: float = 0.1) -> tuple[Responder, threading.Event]:
    """A responder that sends ``SPEC`` one byte every *period* seconds (20+ s in all).

    The event is set when the client hangs up before the last byte — the sign
    that a download was abandoned rather than read to the end.
    """
    body = json.dumps(SPEC).encode()
    dropped = threading.Event()

    def respond(h: BaseHTTPRequestHandler) -> None:
        h.send_response(200)
        h.send_header("Content-Type", "application/json")
        h.end_headers()
        for byte in body:
            try:
                h.wfile.write(bytes([byte]))
                h.wfile.flush()
            except OSError:
                dropped.set()
                return
            time.sleep(period)

    return respond, dropped


# ---------------------------------------------------------------------------
# Plain helpers
# ---------------------------------------------------------------------------


class TestPublicSource:
    def test_strips_userinfo_query_and_fragment(self) -> None:
        assert (
            public_source("https://u:p@127.0.0.1:1/openapi.json?api_key=x")
            == "https://127.0.0.1:1/openapi.json"
        )
        assert public_source("http://host/spec.yaml#frag") == "http://host/spec.yaml"
        assert public_source("http://u:p@host?api_key=x") == "http://host"
        assert public_source("http://host/v1/openapi.json") == "http://host/v1/openapi.json"

    def test_paths_and_inline_text_are_unchanged(self) -> None:
        assert public_source("./api/openapi.yaml") == "./api/openapi.yaml"
        assert public_source('{"openapi": "3.0.0"}') == '{"openapi": "3.0.0"}'

    @pytest.mark.parametrize(
        "odd",
        ["http://", "http://u:p?x@host/", "http://host#frag@x/", "https://[::1/x", "http://@@@"],
    )
    def test_never_raises_and_never_keeps_a_userinfo(self, odd: str) -> None:
        out = public_source(odd)
        assert "@" not in out and "?" not in out and "#" not in out


class TestRedact:
    def test_scrubs_the_url_and_its_parts(self) -> None:
        from promptise.mcpcast.wizard import _redact

        source = "https://svc:S3CRET@127.0.0.1:1/openapi.json?api_key=QUERY-SECRET"
        assert _redact(f"could not fetch spec from {source}: refused", source) == (
            "could not fetch spec from https://127.0.0.1:1/openapi.json: refused"
        )
        # a library printing the URL its own way (httpx keeps the username)
        out = _redact(
            "401 for url 'https://svc:[secure]@127.0.0.1:1/x?api_key=QUERY-SECRET'", source
        )
        assert out == "401 for url 'https://127.0.0.1:1/x'"
        assert _redact("no such file: openapi.json", "openapi.json") == "no such file: openapi.json"
        assert _redact("spec not found: 'x?y'", "x?y") == "spec not found: 'x?y'"  # not a URL


class TestParsedSpec:
    def test_credentials_in_the_url_are_used_for_the_fetch_only(self) -> None:
        secret_url = "https://u:p@127.0.0.1:1/openapi.json?api_key=x"
        parsed = ParsedSpec.load(secret_url, document=SPEC)  # loaded already: no fetch
        assert parsed.source == parsed.label == "https://127.0.0.1:1/openapi.json"
        assert parsed.base_url == "https://w.example.com"
        s = WizardSettings(
            spec=parsed.source, name="widgets", derived_name="widgets", auth=AuthMode.PASSTHROUGH
        )
        assert equivalent_command(s) == "promptise mcpcast https://127.0.0.1:1/openapi.json"

    def test_relative_servers_resolve_against_the_public_url(self) -> None:
        spec = {**SPEC, "servers": [{"url": "/api"}]}
        parsed = ParsedSpec.load("http://u:secret@127.0.0.1:1/openapi.json", document=spec)
        # the fetch origin, without the userinfo, is what the plan gets
        assert parsed.base_url == "http://127.0.0.1:1/api"
        assert parsed.declared_base_url == "http://127.0.0.1:1/api"

    def test_declared_base_url_survives_an_override(self) -> None:
        parsed = ParsedSpec.load(json.dumps(SPEC), base_url="https://staging.example.com")
        assert parsed.base_url == "https://staging.example.com"
        assert parsed.declared_base_url == "https://w.example.com"
        assert all(op.base_url == "https://staging.example.com" for op in parsed.operations)
        back = parsed.with_base_url(None)
        assert back.base_url == back.declared_base_url == "https://w.example.com"
        assert back.document is parsed.document

    def test_declared_base_url_is_empty_when_the_spec_cannot_resolve_one(self) -> None:
        spec = {**SPEC, "servers": [{"url": "https://{host}/api", "variables": {"host": {}}}]}
        with pytest.raises(MCPcastError, match="no default"):
            ParsedSpec.load(json.dumps(spec))
        parsed = ParsedSpec.load(json.dumps(spec), base_url="https://api.example.com/api")
        assert parsed.declared_base_url == ""
        assert parsed.base_url == "https://api.example.com/api"

    def test_a_cancelled_download_is_abandoned(self) -> None:
        polls = 0

        def cancelled() -> bool:
            nonlocal polls
            polls += 1
            return True

        with _loopback(lambda h: _send(h, json.dumps(SPEC).encode())) as port:
            with pytest.raises(MCPcastError, match="cancelled"):
                ParsedSpec.load(f"http://127.0.0.1:{port}/openapi.json", cancelled=cancelled)
        assert polls >= 1
        # a document given up front is never fetched, so nothing polls the predicate
        polls = 0
        parsed = ParsedSpec.load(
            "http://127.0.0.1:1/openapi.json", document=SPEC, cancelled=cancelled
        )
        assert parsed.title == "Widgets API" and polls == 0

    def test_load_file(self, spec_file: Path) -> None:
        parsed = ParsedSpec.load(str(spec_file))
        assert parsed.title == "Widgets API"
        assert parsed.version == "2.1"
        assert parsed.api_name == "widgets"
        assert parsed.base_url == "https://w.example.com"
        assert parsed.label == str(spec_file)
        assert len(parsed.operations) == 4
        counts = {r.value: n for r, n in parsed.risk_counts.items()}
        assert counts == {"read": 2, "write": 1, "destructive": 1, "financial": 0}
        assert parsed.summary() == (
            "Widgets API v2.1 — 4 operations: 2 read · 1 write · 1 destructive"
        )
        assert parsed.description == "Widgets API: Widgets for everyone."

    def test_load_inline_and_base_url_override(self) -> None:
        parsed = ParsedSpec.load(json.dumps(SPEC), base_url="http://localhost:9999/")
        assert parsed.label == "<inline>"
        assert parsed.base_url == "http://localhost:9999"

    def test_rejects_plan_file(self, tmp_path: Path) -> None:
        plan = mcpcast(SPEC, name="widgets")
        plan_path = tmp_path / "mcpcast.plan.yaml"
        plan_path.write_text(plan.to_yaml(), encoding="utf-8")
        with pytest.raises(MCPcastError, match="is an mcpcast plan"):
            ParsedSpec.load(str(plan_path))

    def test_rejects_empty_and_operation_less(self) -> None:
        with pytest.raises(MCPcastError, match="Enter the path"):
            ParsedSpec.load("   ")
        with pytest.raises(MCPcastError, match="no 'paths'"):
            ParsedSpec.load(json.dumps({"openapi": "3.0.0", "info": {"title": "x"}, "paths": {}}))
        with pytest.raises(MCPcastError, match="no operations"):
            ParsedSpec.load(
                json.dumps({"openapi": "3.0.0", "info": {"title": "x"}, "paths": {"/x": {}}})
            )


class TestPreviews:
    def test_profile_preview_counts(self, spec_file: Path) -> None:
        parsed = ParsedSpec.load(str(spec_file))
        ro = preview_profile(parsed, SafetyProfile.READ_ONLY)
        st = preview_profile(parsed, SafetyProfile.STANDARD)
        full = preview_profile(parsed, SafetyProfile.FULL)
        assert (ro.tools, ro.gated, ro.excluded) == (2, 0, 2)
        assert (st.tools, st.gated, st.excluded) == (3, 1, 1)
        assert (full.tools, full.gated, full.excluded) == (4, 2, 0)
        assert ro.line() == "2 tools · reads only"
        assert st.line() == "3 tools · 1 require human approval"

    def test_recommended_auth(self) -> None:
        assert recommended_auth(Usage.PERSONAL) is AuthMode.ENV_TOKEN
        assert recommended_auth(Usage.SHARED) is AuthMode.PASSTHROUGH
        assert recommended_auth(Usage.TENANTS) is AuthMode.API_KEY
        assert recommended_auth(Usage.OPEN) is AuthMode.NONE


class TestReviewWarnings:
    def test_ghost_tool_mention(self) -> None:
        # standard profile: deleteWidget is excluded, so a description naming
        # delete_widget sends the agent after a tool that does not exist.
        plan = mcpcast(SPEC, profile=SafetyProfile.STANDARD, name="widgets")
        plan.tools[0].description += " To remove one use delete_widget."
        warnings = review_warnings(plan)
        assert len(warnings) == 1
        assert warnings[0].startswith(f"{plan.tools[0].name}: the description names delete_widget")

    def test_hidden_parameter_is_reported(self) -> None:
        plan = mcpcast(SPEC, profile=SafetyProfile.FULL, name="widgets")
        get = plan.tool("get_widget")
        get.params["id"].hidden = True
        get.params["id"].default = "w_1"
        assert any(w.startswith("get_widget: hides id") for w in review_warnings(plan))

    def test_clean_plan_has_no_warnings(self) -> None:
        assert review_warnings(mcpcast(SPEC, profile=SafetyProfile.FULL, name="widgets")) == []


class TestEquivalentCommand:
    def test_defaults_are_omitted(self) -> None:
        s = WizardSettings(
            spec="openapi.json",
            profile=SafetyProfile.READ_ONLY,
            auth=AuthMode.PASSTHROUGH,
            name="widgets",
            derived_name="widgets",
            out_dir="widgets-mcp",
        )
        assert equivalent_command(s) == "promptise mcpcast openapi.json"

    def test_command_is_quoted_for_the_platform_shell(self, monkeypatch):
        # POSIX: shlex quoting.
        assert quote_argument("C:\\my specs\\api.yaml", windows=False) == "'C:\\my specs\\api.yaml'"
        assert quote_argument("openapi.json", windows=False) == "openapi.json"
        # Windows: cmd.exe does not understand single quotes; double quotes work in
        # cmd.exe and PowerShell alike, and only arguments that need them get them.
        assert quote_argument("C:\\my specs\\api.yaml", windows=True) == '"C:\\my specs\\api.yaml"'
        assert quote_argument("C:\\specs\\api.yaml", windows=True) == "C:\\specs\\api.yaml"
        assert quote_argument('say "hi"', windows=True) == '"say \\"hi\\""'
        monkeypatch.setattr("promptise.mcpcast.wizard._on_windows", lambda: True)
        s = WizardSettings(
            spec="C:\\my specs\\api.yaml", auth=AuthMode.PASSTHROUGH, out_dir="D:\\out dir"
        )
        assert equivalent_command(s) == (
            'promptise mcpcast "C:\\my specs\\api.yaml" --out "D:\\out dir"'
        )

    def test_offline_and_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("promptise.mcpcast.wizard._on_windows", lambda: False)
        s = WizardSettings(
            spec="my spec.yaml",
            base_url="https://api.example.com",
            base_url_override=True,
            curate=False,
            profile=SafetyProfile.STANDARD,
            auth=AuthMode.ENV_TOKEN,
            approval=ApprovalMode.ELICITATION,
            name="shop",
            derived_name="my-shop",
            out_dir="out",
            max_tools=8,
            force=True,
        )
        assert equivalent_command(s) == (
            "promptise mcpcast 'my spec.yaml' --base-url https://api.example.com "
            "--profile standard --auth env-token --approval elicitation --no-curate "
            "--max-tools 8 --name shop --out out --force"
        )

    def test_model_budget_and_eval(self) -> None:
        s = WizardSettings(
            spec="s.json",
            model="anthropic:claude-sonnet-4.5",
            name="s",
            derived_name="s",
            out_dir="s-mcp",
            auth=AuthMode.PASSTHROUGH,
            max_tools=25,
            evaluate=True,
            eval_tasks=12,
        )
        # 25 is the curation default: not repeated on the command line.
        assert equivalent_command(s) == (
            "promptise mcpcast s.json --model anthropic:claude-sonnet-4.5 --eval --eval-tasks 12"
        )
        s.max_tools = 10
        s.model = DEFAULT_MODEL
        s.eval_tasks = 20
        assert equivalent_command(s) == "promptise mcpcast s.json --max-tools 10 --eval"

    def test_effective_budget(self) -> None:
        assert WizardSettings(curate=True).effective_budget == 25
        assert WizardSettings(curate=False).effective_budget is None
        assert WizardSettings(curate=False, max_tools=3).effective_budget == 3


class TestDetectLocalApis:
    def test_first_hit_per_port_and_junk_skipped(self) -> None:
        body = json.dumps(SPEC)
        seen: list[str] = []

        def fetch(url: str) -> str | None:
            seen.append(url)
            if url == "http://127.0.0.1:8000/openapi.json":
                return "<html>not a spec</html>"
            if url == "http://127.0.0.1:8000/swagger.json":
                return body
            if url == "http://127.0.0.1:3000/openapi.json":
                return body
            return None

        found = detect_local_apis(
            ports=(8000, 8080, 3000), paths=("/openapi.json", "/swagger.json"), fetch=fetch
        )
        assert found == [
            Candidate(url="http://127.0.0.1:8000/swagger.json", title="Widgets API", operations=4),
            Candidate(url="http://127.0.0.1:3000/openapi.json", title="Widgets API", operations=4),
        ]
        # once a port hits, its remaining paths are not probed
        assert "http://127.0.0.1:3000/swagger.json" not in seen

    def test_plan_documents_are_not_candidates(self) -> None:
        plan = mcpcast(SPEC, name="widgets").to_yaml()
        found = detect_local_apis(ports=(1,), paths=("/openapi.yaml",), fetch=lambda url: plan)
        assert found == []

    def test_a_body_is_parsed_never_fetched_or_read(self, spec_file: Path) -> None:
        # The audit steered detection off loopback with a body that load_spec()
        # would treat as a URL to GET, and read a file from disk with a body
        # that is a path. Both are parse failures now.
        seen: list[str] = []
        bodies = {
            "http://127.0.0.1:8000/openapi.json": "http://127.0.0.1:9/steered.json",
            "http://127.0.0.1:8080/openapi.json": str(spec_file),
        }

        def fetch(url: str) -> str | None:
            seen.append(url)
            return bodies.get(url)

        detection = probe_local_apis(ports=(8000, 8080), paths=("/openapi.json",), fetch=fetch)
        assert detection.candidates == [] and detection.skipped_ports == ()
        assert seen == list(bodies)  # only the two loopback probes, nothing else

    def test_malformed_and_hostile_bodies_are_skipped_not_raised(self) -> None:
        deep = '{"a":' * 100_000 + "1" + "}" * 100_000
        bomb = "a: &a [x, x, x, x, x, x, x, x, x]\n" + "".join(
            f"{chr(98 + i)}: &{chr(98 + i)} [*{chr(97 + i)}, *{chr(97 + i)}, *{chr(97 + i)}, "
            f"*{chr(97 + i)}, *{chr(97 + i)}, *{chr(97 + i)}, *{chr(97 + i)}, *{chr(97 + i)}, "
            f"*{chr(97 + i)}]\n"
            for i in range(8)
        )
        # Defects of the whole document: never a candidate, never an exception.
        hostile = [
            json.dumps({"openapi": "3.0.0", "info": "oops", "paths": {"/x": {"get": {}}}}),
            json.dumps({**SPEC, "paths": "nope"}),
            json.dumps({**SPEC, "paths": {}}),
            deep,
            "openapi: 3.0.0\ninfo: {title: bomb}\npaths:\n  /x: {get: {}}\n" + bomb,
            "\x00\x01 binary",
            "",
        ]
        # Defects inside one operation: the document is listed, as the CLI accepts
        # it — the operation is extracted with the defect as its reason and the
        # planner drops it, so the wizard's review shows why (see parse.py).
        defective = [
            json.dumps({**SPEC, "paths": {"/x": {"get": {"parameters": "nope"}}}}),
            json.dumps({**SPEC, "paths": {"/x": {"get": {"parameters": ["nope"]}}}}),
            json.dumps({**SPEC, "paths": {"/x": {"post": {"requestBody": {"content": "no"}}}}}),
            json.dumps({**SPEC, "paths": {"/x": {"get": {"operationId": 5}}}}),
        ]
        bodies = [*hostile, *defective]
        ports = tuple(range(1, len(bodies) + 1)) + (99,)
        by_port = {
            f"http://127.0.0.1:{p}/openapi.json": b for p, b in zip(ports, bodies, strict=False)
        }
        by_port["http://127.0.0.1:99/openapi.json"] = json.dumps(SPEC)
        started = time.monotonic()
        found = detect_local_apis(ports=ports, paths=("/openapi.json",), fetch=by_port.get)
        listed = [p for p in ports if p > len(hostile)]
        assert [c.url for c in found] == [f"http://127.0.0.1:{p}/openapi.json" for p in listed]
        assert all(c.operations == 1 for c in found[:-1]) and found[-1].operations == 4
        assert time.monotonic() - started < 8  # the alias bomb is refused, not expanded

    def test_probe_urls_are_loopback_only(self) -> None:
        with pytest.raises(ValueError, match="plain absolute path"):
            probe_local_apis(ports=(8000,), paths=("/x?y=1",), fetch=lambda url: None)
        with pytest.raises(ValueError, match="plain absolute path"):
            probe_local_apis(ports=(8000,), paths=("@evil.example/x",), fetch=lambda url: None)
        with pytest.raises(ValueError, match="port out of range"):
            probe_local_apis(ports=(70000,), paths=("/x",), fetch=lambda url: None)

    def test_budget_reports_the_ports_it_did_not_reach(self) -> None:
        body = json.dumps(SPEC)

        def slow(url: str) -> str | None:
            time.sleep(0.03)
            return body if url.startswith("http://127.0.0.1:1/") else None

        detection = probe_local_apis(
            ports=(1, 2, 3, 4), paths=("/a", "/b"), budget=0.05, fetch=slow
        )
        assert [c.url for c in detection.candidates] == ["http://127.0.0.1:1/a"]
        assert 1 not in detection.skipped_ports and 4 in detection.skipped_ports
        assert "were not checked" in detection.skipped_line()
        assert detection.elapsed < 1
        assert Detection(candidates=[]).skipped_line() == ""

    def test_cancellation_stops_the_probe(self) -> None:
        calls: list[str] = []
        detection = probe_local_apis(
            ports=(1, 2), paths=("/a",), fetch=lambda url: calls.append(url), cancelled=lambda: True
        )
        assert calls == [] and detection.skipped_ports == (1, 2)

    def test_cancellation_lets_go_of_a_trickling_body(self) -> None:
        trickle, dropped = _trickle(period=0.1)
        with _loopback(trickle) as port:
            started = time.monotonic()
            cancel_at = started + 0.3
            detection = probe_local_apis(
                ports=(port,),
                paths=("/openapi.json", "/swagger.json"),
                cancelled=lambda: time.monotonic() > cancel_at,
            )
            assert detection.candidates == [] and detection.skipped_ports == (port,)
            assert time.monotonic() - started < 3  # not the 10 s budget, not the whole trickle
            assert dropped.wait(2)  # the connection was dropped mid-body

    def test_real_fetcher_follows_no_redirect_and_caps_the_body(self) -> None:
        spec = json.dumps(SPEC).encode()
        upstream_hits: list[str] = []

        def target(h: BaseHTTPRequestHandler) -> None:
            upstream_hits.append(h.path)
            _send(h, spec)

        with _loopback(target) as target_port:

            def probed(h: BaseHTTPRequestHandler) -> None:
                if h.path == "/openapi.json":  # steer to another origin
                    _send(h, b"", 302, Location=f"http://127.0.0.1:{target_port}/evil.json")
                elif h.path == "/swagger.json":  # a document over the cap
                    big = {**SPEC, "info": {**SPEC["info"], "description": "x" * 300_000}}
                    _send(h, json.dumps(big).encode())
                elif h.path == "/api-docs":  # trickles: 1 byte every 100 ms
                    h.send_response(200)
                    h.send_header("Content-Type", "application/json")
                    h.end_headers()
                    for byte in spec:
                        try:
                            h.wfile.write(bytes([byte]))
                            h.wfile.flush()
                        except OSError:
                            return
                        time.sleep(0.1)
                else:
                    _send(h, spec)

            with _loopback(probed) as port:
                redirect = probe_local_apis(ports=(port,), paths=("/openapi.json",))
                assert redirect.candidates == [] and upstream_hits == []
                capped = probe_local_apis(
                    ports=(port,), paths=("/swagger.json",), max_bytes=100_000
                )
                assert capped.candidates == []
                under = probe_local_apis(ports=(port,), paths=("/swagger.json",))
                assert [c.title for c in under.candidates] == ["Widgets API"]
                started = time.monotonic()
                slow = probe_local_apis(
                    ports=(port,), paths=("/api-docs", "/v3/api-docs"), budget=0.4
                )
                assert slow.candidates == [] and slow.skipped_ports == (port,)
                assert time.monotonic() - started < 4  # not the 20+ s the trickle would take
                plain = probe_local_apis(ports=(port,), paths=("/v3/api-docs",))
                assert [c.url for c in plain.candidates] == [f"http://127.0.0.1:{port}/v3/api-docs"]


# ---------------------------------------------------------------------------
# The app, headless
# ---------------------------------------------------------------------------


def _app(tmp_path: Path, spec: str | None = None, **kwargs) -> MCPcastWizard:
    return MCPcastWizard(spec=spec, cwd=tmp_path, auto_detect=False, **kwargs)


class TestWizardFlow:
    async def test_offline_end_to_end(self, tmp_path: Path, spec_file: Path) -> None:
        app = _app(tmp_path, str(spec_file))
        async with app.run_test(size=(100, 34)) as pilot:
            assert app.pane.id == "welcome"
            await pilot.press("enter")  # Start → spec loads from the pre-filled source
            await _settle(pilot)
            assert app.pane.id == "spec"
            assert app.parsed is not None and app.parsed.api_name == "widgets"
            assert "✓ Widgets API v2.1" in _text(app, "#spec-status")
            assert app.query_one("#spec-base-url").value == "https://w.example.com"

            await pilot.press("enter")  # Continue
            await pilot.pause()
            assert app.pane.id == "model"
            await pilot.press("down", "enter")  # Offline
            await pilot.pause()
            assert app.settings.curate is False
            assert app.pane.id == "safety"
            assert "2 tools · reads only" in str(
                app.query_one("#safety-choice").get_option_at_index(0).prompt
            )

            await pilot.press("down", "enter")  # standard
            await pilot.pause()
            assert app.settings.profile is SafetyProfile.STANDARD
            assert app.pane.id == "auth"
            assert "MCPCAST_UPSTREAM_TOKEN" in _text(app, "#auth-hint")
            await pilot.press("enter")  # personal → env-token
            await pilot.pause()
            assert app.settings.auth is AuthMode.ENV_TOKEN
            assert app.pane.id == "project"
            assert app.query_one("#project-name").value == "widgets"
            assert app.query_one("#project-out").value == "widgets-mcp"
            assert app.query_one("#project-eval").disabled  # offline: no model to evaluate with

            await pilot.press("enter")  # defaults → review builds the plan offline
            await _settle(pilot)
            assert app.pane.id == "review"
            assert app.plan is not None
            assert [t.name for t in app.plan.tools] == [
                "list_widgets",
                "create_widget",
                "get_widget",
            ]
            assert "3 tools (1 require approval)" in _text(app, "#review-log")
            assert app.query_one("#review-table").row_count == 3

            await pilot.press("enter")  # Write project
            await _settle(pilot)
            assert app.pane.id == "write"
            assert app.result is not None
            assert {p.name for p in app.result.written} >= {
                "README.md",
                "mcpcast.plan.yaml",
                "server.py",
                "pyproject.toml",
                "test_tools.py",
            }
            assert app.result.command == (
                f"promptise mcpcast {spec_file} --profile standard --auth env-token --no-curate"
            )
            await pilot.press("enter")  # Finish
        result = app.return_value
        assert result is not None
        out = tmp_path / "widgets-mcp"
        assert (out / "server.py").exists()
        plan = MCPcastPlan.from_yaml((out / "mcpcast.plan.yaml").read_text(encoding="utf-8"))
        expected = mcpcast(
            SPEC, profile=SafetyProfile.STANDARD, auth=AuthMode.ENV_TOKEN, name="widgets"
        )
        assert [t.name for t in plan.tools] == [t.name for t in expected.tools]
        assert plan.api.auth is AuthMode.ENV_TOKEN

    async def test_model_path_with_scripted_curation(
        self, tmp_path: Path, spec_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        app = _app(tmp_path, str(spec_file), completer=_scripted(PROPOSAL))
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")  # → model
            await pilot.pause()
            assert "✓ openai:gpt-5-mini" in _text(app, "#model-status")
            assert "OPENAI_API_KEY is set" in _text(app, "#model-status")
            await pilot.press("enter")  # design with a model
            await pilot.press("down", "down", "enter")  # full
            await pilot.press("down", "enter")  # shared → passthrough
            await pilot.pause()
            assert app.settings.auth is AuthMode.PASSTHROUGH
            await pilot.press("enter")  # project defaults
            await _settle(pilot)
            assert app.pane.id == "review"
            assert app.plan is not None
            assert [t.name for t in app.plan.tools] == ["find_widgets", "create_widget"]
            assert app.query_one("#review-table").row_count == 2
            detail = app.query_one("#review-tool")
            assert "renamed from `get_widget`" in detail.source
            assert app.query_one("#review-dropped").title == "Not exposed (1)"
            await pilot.press("enter")  # write
            await _settle(pilot)
            assert app.result is not None
            assert app.result.command == f"promptise mcpcast {spec_file} --profile full"

    async def test_model_step_blocks_without_a_key(self, tmp_path: Path, spec_file: Path) -> None:
        app = _app(tmp_path, str(spec_file))
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")
            await pilot.pause()
            assert app.pane.id == "model"
            status = _text(app, "#model-status")
            assert "OPENAI_API_KEY is not set" in status and ".env" in status
            await pilot.press("enter")  # try to continue with the model
            await pilot.pause()
            assert app.pane.id == "model"
            assert "cannot be used yet" in _text(app, "#model-error")
            await pilot.press("down", "enter")  # Offline is always possible
            await pilot.pause()
            assert app.pane.id == "safety"

    async def test_spec_errors_stay_on_step_one(self, tmp_path: Path) -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await pilot.pause()
            assert app.pane.id == "spec"
            await pilot.press("enter")  # empty
            await pilot.pause()
            assert "Enter the path or URL" in _text(app, "#spec-error")
            await pilot.press(*"nope.json", "enter")
            await _settle(pilot)
            assert app.parsed is None
            assert app.pane.id == "spec"
            assert _text(app, "#spec-error")  # load_spec's own message
            app.query_one("#spec-input").value = json.dumps(SPEC)
            await pilot.press("enter")
            await _settle(pilot)
            assert app.parsed is not None and app.parsed.label == "<inline>"

    async def test_base_url_override_is_validated_and_recorded(self, tmp_path: Path) -> None:
        spec = {**SPEC, "servers": [{"url": "/api"}]}  # relative: the planner needs --base-url
        path = tmp_path / "rel.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        app = _app(tmp_path, str(path))
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")  # Continue without a usable base URL
            await pilot.pause()
            assert app.pane.id == "spec"
            assert "API base URL" in _text(app, "#spec-error")
            app.query_one("#spec-base-url").value = "https://api.example.com/api"
            await _settle(pilot)  # a Button ignores a second press within its 0.2 s active effect
            await pilot.press("enter")
            await pilot.pause()
            assert app.pane.id == "model"
            assert app.settings.base_url == "https://api.example.com/api"
            assert app.settings.base_url_override is True

    async def test_detection_lists_candidates(self, tmp_path: Path) -> None:
        body = json.dumps(SPEC)

        def fetch(url: str) -> str | None:
            return body if url == "http://127.0.0.1:8080/openapi.json" else None

        app = _app(tmp_path, fetch=fetch)
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await pilot.pause()
            await pilot.click("#spec-detect")
            await _settle(pilot)
            options = app.query_one("#spec-candidates")
            assert options.display and options.option_count == 1
            assert "Found 1" in _text(app, "#spec-status")
            await pilot.press("enter")  # pick it
            await _settle(pilot)
            assert app.parsed is not None
            assert app.settings.spec == "http://127.0.0.1:8080/openapi.json"

    async def test_project_validation(self, tmp_path: Path, spec_file: Path) -> None:
        (tmp_path / "taken-mcp").mkdir()
        (tmp_path / "taken-mcp" / "mcpcast.plan.yaml").write_text("version: 1\n", encoding="utf-8")
        app = _app(tmp_path, str(spec_file))
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")
            await pilot.press("down", "enter")  # offline
            await pilot.press("enter")  # read-only
            await pilot.press("enter")  # personal
            await pilot.pause()
            assert app.pane.id == "project"
            app.query_one("#project-name").value = "Bad Name!"
            await pilot.press("enter")
            await pilot.pause()
            assert "Server name" in _text(app, "#project-error")
            app.query_one("#project-name").value = "taken"
            await pilot.pause()
            assert app.query_one("#project-out").value == "taken-mcp"
            assert app.query_one("#project-force-row").display
            app.query_one("#project-budget").value = "0"
            await pilot.press("enter")
            await pilot.pause()
            assert "Tool budget" in _text(app, "#project-error")
            app.query_one("#project-budget").value = "2"
            await pilot.press("enter")
            await pilot.pause()
            assert "already contains" in _text(app, "#project-error")
            app.query_one("#project-force").value = True
            await pilot.press("enter")
            await _settle(pilot)
            assert app.pane.id == "review"
            assert app.settings.force is True and app.settings.max_tools == 2
            assert len(app.plan.tools) == 2

    async def test_pending_approval_needs_api_key_auth(
        self, tmp_path: Path, spec_file: Path
    ) -> None:
        app = _app(tmp_path, str(spec_file))
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")
            await pilot.press("down", "enter")
            await pilot.press("enter")
            await pilot.pause()
            assert app.pane.id == "auth"
            app.query_one("#auth-approval").value = "pending"
            await pilot.pause()
            await pilot.press("enter")  # personal (env-token) + pending → refused
            await pilot.pause()
            assert app.pane.id == "auth"
            assert "pending" in _text(app, "#auth-error")
            await pilot.press("down", "down", "enter")  # multi-tenant → api-key
            await pilot.pause()
            assert app.pane.id == "project"
            assert app.settings.approval is ApprovalMode.PENDING

    async def test_passthrough_is_refused_for_an_api_key_scheme(self, tmp_path: Path) -> None:
        """A spec keyed by ``X-API-Key`` previews fine but refuses the shared/passthrough choice."""
        spec = {
            **SPEC,
            "components": {
                "securitySchemes": {
                    "ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
                }
            },
            "security": [{"ApiKeyAuth": []}],
        }
        path = tmp_path / "keyed.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        parsed = ParsedSpec.load(str(path))
        assert preview_profile(parsed, SafetyProfile.FULL).tools == 4  # counts need no auth mode
        app = _app(tmp_path, str(path))
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")
            await pilot.press("down", "enter")
            await pilot.press("enter")
            await pilot.pause()
            assert app.pane.id == "auth"
            await pilot.press("down", "enter")  # shared → passthrough: cannot relay X-API-Key
            await pilot.pause()
            assert app.pane.id == "auth"
            assert "X-API-Key" in _text(app, "#auth-error")
            assert "passthrough" in _text(app, "#auth-error")
            await pilot.press("up", "enter")  # personal → env-token presents it
            await pilot.pause()
            assert app.pane.id == "project"
            assert app.settings.auth is AuthMode.ENV_TOKEN

    async def test_back_and_plan_invalidation(self, tmp_path: Path, spec_file: Path) -> None:
        app = _app(tmp_path, str(spec_file))
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")
            await pilot.press("down", "enter")
            await pilot.press("enter")  # read-only
            await pilot.press("enter")
            await pilot.press("enter")
            await _settle(pilot)
            assert app.pane.id == "review" and len(app.plan.tools) == 2
            first_key = app.plan_built_for
            await pilot.press("escape")  # back to project
            await pilot.press("escape")  # auth
            await pilot.press("escape")  # safety
            await pilot.pause()
            assert app.pane.id == "safety"
            await pilot.press("down", "down", "enter")  # full
            await pilot.press("enter")
            await pilot.press("enter")
            await _settle(pilot)
            assert app.pane.id == "review"
            assert app.plan_built_for != first_key
            assert len(app.plan.tools) == 4
            assert "4 tools (2 require approval)" in _text(app, "#review-log")

    async def test_out_dir_prefill_and_eval_hint(
        self, tmp_path: Path, spec_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("MCPCAST_EVAL_AUTHORIZATION", raising=False)
        monkeypatch.delenv("MCPCAST_EVAL_HEADERS", raising=False)
        app = MCPcastWizard(
            spec=str(spec_file), cwd=tmp_path, out_dir="generated/widgets-mcp", auto_detect=False
        )
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")  # → model
            await pilot.press("enter")  # design with a model (key is "set")
            await pilot.press("enter")  # read-only
            await pilot.press("enter")  # personal → env-token
            await pilot.pause()
            assert app.pane.id == "project"
            assert app.query_one("#project-out").value == "generated/widgets-mcp"
            assert _text(app, "#project-eval-hint") == ""
            app.query_one("#project-eval").value = True
            await pilot.pause()
            assert "MCPCAST_EVAL_AUTHORIZATION is not set" in _text(app, "#project-eval-hint")
            monkeypatch.setenv("MCPCAST_EVAL_AUTHORIZATION", "Bearer t")
            app.query_one("#project-eval").value = False
            await pilot.pause()
            app.query_one("#project-eval").value = True
            await pilot.pause()
            assert "✓ MCPCAST_EVAL_AUTHORIZATION is set" in _text(app, "#project-eval-hint")
            app.query_one("#project-eval").value = False  # do not run a real evaluation here
            await pilot.pause()
            assert app.settings.out_dir == "generated/widgets-mcp"

    async def test_help_screen(self, tmp_path: Path) -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("f1")
            await pilot.pause()
            assert app.screen.__class__.__name__ == "HelpScreen"
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen.__class__.__name__ != "HelpScreen"

    async def test_quit_returns_nothing(self, tmp_path: Path) -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("ctrl+q")
        assert app.return_value is None

    def test_step_names_match_panes(self) -> None:
        assert len(STEP_NAMES) == 7


async def _to_review(pilot, app: MCPcastWizard, *, safety: int = 0) -> None:
    """Start → load the pre-filled spec → offline → profile row *safety* → personal → project."""
    await pilot.press("enter")
    await _settle(pilot)
    assert app.parsed is not None
    await pilot.press("enter")  # → model
    await pilot.press("down", "enter")  # offline
    await pilot.press(*(["down"] * safety), "enter")  # safety
    await pilot.press("enter")  # personal → env-token
    await pilot.pause()
    assert app.pane.id == "project"


def _routes_base_urls(plan: MCPcastPlan) -> list[str | None]:
    return [r.base_url for t in plan.tools for r in t.routes]


class TestAuditedBehaviour:
    """One test per finding of the wizard audit — each was a reproduced failure."""

    async def test_typed_base_url_is_what_the_plan_and_every_route_use(
        self, tmp_path: Path
    ) -> None:
        spec = {**SPEC, "servers": [{"url": "https://prod.example.com"}]}
        path = tmp_path / "prod.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        app = _app(tmp_path, str(path))
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            assert app.query_one("#spec-base-url").value == "https://prod.example.com"
            app.query_one("#spec-base-url").value = "https://staging.example.com"
            await pilot.pause()
            await pilot.press("enter")  # Continue: re-extracts with the override
            await pilot.press("down", "enter")  # offline
            await pilot.press("enter")  # read-only
            await pilot.press("enter")  # personal
            await pilot.press("enter")  # project defaults
            await _settle(pilot)
            assert app.pane.id == "review" and app.plan is not None
            assert app.settings.base_url == "https://staging.example.com"
            assert app.settings.base_url_override is True
            assert app.plan.api.base_url == "https://staging.example.com"
            assert _routes_base_urls(app.plan) == [None, None]
            await pilot.press("enter")  # write
            await _settle(pilot)
            assert app.result is not None
            assert "--base-url https://staging.example.com" in app.result.command
        out = tmp_path / "widgets-mcp"
        written = MCPcastPlan.from_yaml((out / "mcpcast.plan.yaml").read_text(encoding="utf-8"))
        assert written.api.base_url == "https://staging.example.com"
        assert _routes_base_urls(written) == [None, None]
        generated = "\n".join(p.read_text(encoding="utf-8") for p in out.glob("*_mcp/**/*.py"))
        assert "prod.example.com" not in generated

    async def test_base_url_from_the_cli_stays_in_the_command(
        self, tmp_path: Path, spec_file: Path
    ) -> None:
        app = MCPcastWizard(
            spec=str(spec_file),
            base_url="https://staging.example.com",
            cwd=tmp_path,
            auto_detect=False,
        )
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            assert app.parsed is not None
            assert app.parsed.declared_base_url == "https://w.example.com"
            await pilot.press("enter")  # Continue without touching the field
            await pilot.pause()
            assert app.pane.id == "model"
            assert app.settings.base_url_override is True
            assert "--base-url https://staging.example.com" in equivalent_command(app.settings)
            await pilot.press("escape")
            await pilot.pause()
            app.query_one("#spec-base-url").value = "https://w.example.com"  # back to the spec's
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.settings.base_url_override is False
            assert "--base-url" not in equivalent_command(app.settings)
            assert app.parsed.base_url == "https://w.example.com"

    async def test_an_occupied_folder_needs_the_overwrite_switch(
        self, tmp_path: Path, spec_file: Path
    ) -> None:
        mine = tmp_path / "my-existing-app"
        mine.mkdir()
        (mine / "README.md").write_text("# mine\n", encoding="utf-8")
        (mine / "server.py").write_text("print('mine')\n", encoding="utf-8")
        app = _app(tmp_path, str(spec_file))
        async with app.run_test(size=(100, 34)) as pilot:
            await _to_review(pilot, app)
            assert not app.query_one("#project-force-row").display
            app.query_one("#project-out").value = "my-existing-app"
            await pilot.pause()
            assert app.query_one("#project-force-row").display
            label = _text(app, "#project-force-label")
            assert "not an mcpcast project" in label and "README.md, server.py" in label
            await pilot.press("enter")
            await pilot.pause()
            assert app.pane.id == "project"
            error = _text(app, "#project-error")
            assert "is not an mcpcast project" in error and "turn on overwrite" in error
            assert (mine / "README.md").read_text(encoding="utf-8") == "# mine\n"
            app.query_one("#project-force").value = True
            await pilot.pause()
            await pilot.press("enter")
            await _settle(pilot)
            assert app.pane.id == "review"
            await pilot.press("enter")
            await _settle(pilot)
            assert app.result is not None and app.result.out_dir == mine
            assert app.result.command.endswith("--out my-existing-app --force")
        assert (mine / "README.md").read_text(encoding="utf-8") != "# mine\n"
        assert (mine / "mcpcast.plan.yaml").exists()

    async def test_tilde_in_the_output_folder_is_expanded(
        self, tmp_path: Path, spec_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv("USERPROFILE", str(home))  # what ~ means on Windows
        app = _app(tmp_path, str(spec_file))
        async with app.run_test(size=(100, 34)) as pilot:
            await _to_review(pilot, app)
            app.query_one("#project-out").value = "~/mcpcast-out"
            await pilot.pause()
            await pilot.press("enter")
            await _settle(pilot)
            assert app.pane.id == "review"
            assert app.settings.out_dir == str(home / "mcpcast-out")
            await pilot.press("enter")
            await _settle(pilot)
            assert app.result is not None and app.result.out_dir == home / "mcpcast-out"
            assert f"--out {home / 'mcpcast-out'}" in app.result.command
        assert (home / "mcpcast-out" / "server.py").exists()
        assert not (tmp_path / "~").exists()

    async def test_reloading_an_edited_spec_rebuilds_the_plan(
        self, tmp_path: Path, spec_file: Path
    ) -> None:
        app = _app(tmp_path, str(spec_file))
        async with app.run_test(size=(100, 34)) as pilot:
            await _to_review(pilot, app)
            await pilot.press("enter")
            await _settle(pilot)
            assert app.pane.id == "review" and app.plan is not None
            assert [t.name for t in app.plan.tools] == ["list_widgets", "get_widget"]
            first_generation = app.spec_generation
            for _ in range(5):
                await pilot.press("escape")
            await pilot.pause()
            assert app.pane.id == "spec"
            edited = json.loads(spec_file.read_text(encoding="utf-8"))
            edited["paths"]["/stats"] = {"get": {"operationId": "getStats", "summary": "Stats"}}
            spec_file.write_text(json.dumps(edited), encoding="utf-8")
            await pilot.click("#spec-load")
            await _settle(pilot)
            assert app.parsed is not None and len(app.parsed.operations) == 5
            assert app.spec_generation > first_generation
            for _ in range(5):
                await pilot.press("enter")
                await pilot.pause()
            await _settle(pilot)
            assert app.pane.id == "review"
            assert [t.name for t in app.plan.tools] == ["list_widgets", "get_widget", "get_stats"]
            assert "3 tools" in _text(app, "#review-log")
            await pilot.press("enter")
            await _settle(pilot)
        plan = MCPcastPlan.from_yaml(
            (tmp_path / "widgets-mcp" / "mcpcast.plan.yaml").read_text(encoding="utf-8")
        )
        assert "get_stats" in plan.tool_names

    async def test_writing_again_after_changing_settings_writes_again(
        self, tmp_path: Path, spec_file: Path
    ) -> None:
        app = _app(tmp_path, str(spec_file))
        out = tmp_path / "widgets-mcp"
        async with app.run_test(size=(100, 34)) as pilot:
            await _to_review(pilot, app)
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")  # write (read-only: 2 tools)
            await _settle(pilot)
            assert app.result is not None and len(app.result.plan.tools) == 2
            first = app.result
            for _ in range(4):
                await pilot.press("escape")
            await pilot.pause()
            assert app.pane.id == "safety"
            await pilot.press("down", "down", "enter")  # full
            await pilot.press("enter")  # auth
            await pilot.pause()
            assert app.pane.id == "project"
            assert app.query_one("#project-force-row").display
            assert "already holds an mcpcast project" in _text(app, "#project-force-label")
            app.query_one("#project-force").value = True
            await pilot.pause()
            await pilot.press("enter")
            await _settle(pilot)
            assert app.pane.id == "review" and len(app.plan.tools) == 4
            await pilot.press("enter")  # write again
            await _settle(pilot)
            assert app.pane.id == "write"
            assert app.result is not first and len(app.result.plan.tools) == 4
            assert "--profile full" in app.result.command and "--force" in app.result.command
            assert "4 tools · 2 require approval" in _text(app, "#write-files")
            await pilot.press("escape")  # back to review, nothing changed…
            await pilot.pause()
            await pilot.press("enter")  # …so returning does not write a third time
            await pilot.pause()
            assert app.result is not first and len(app.result.plan.tools) == 4
            await pilot.press("enter")  # Finish
        assert app.return_value is app.result
        plan = MCPcastPlan.from_yaml((out / "mcpcast.plan.yaml").read_text(encoding="utf-8"))
        assert len(plan.tools) == 4 and plan.profile is SafetyProfile.FULL

    async def test_ctrl_q_after_writing_returns_what_is_on_disk(
        self, tmp_path: Path, spec_file: Path
    ) -> None:
        app = _app(tmp_path, str(spec_file))
        async with app.run_test(size=(100, 34)) as pilot:
            await _to_review(pilot, app)
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")  # write
            await _settle(pilot)
            assert app.result is not None
            await pilot.press("ctrl+q")
        assert app.return_value is app.result
        assert (tmp_path / "widgets-mcp" / "server.py").exists()

    async def test_a_superseded_load_never_lands(self, tmp_path: Path) -> None:
        fast = json.dumps({**SPEC, "info": {**SPEC["info"], "title": "FAST API"}}).encode()
        slow = json.dumps({**SPEC, "info": {**SPEC["info"], "title": "SLOW API"}}).encode()

        requested: list[str] = []

        def respond(h: BaseHTTPRequestHandler) -> None:
            requested.append(h.path)
            if h.path == "/slow.json":
                time.sleep(0.8)
                _send(h, slow)
            else:
                _send(h, fast)

        with _loopback(respond) as port:
            app = _app(tmp_path)
            async with app.run_test(size=(100, 34)) as pilot:
                await pilot.press("enter")
                spec_input = app.query_one("#spec-input")
                # Focus moves asynchronously in Textual; on a slow runner an
                # Enter pressed before it lands goes to the app, not the field,
                # and the second load never starts. Wait for each step instead
                # of guessing how long it takes.
                await _until(pilot, lambda: app.focused is spec_input)
                spec_input.value = f"http://127.0.0.1:{port}/slow.json"
                await pilot.press("enter")
                await _until(pilot, lambda: "/slow.json" in requested)
                await _until(pilot, lambda: app.focused is spec_input)
                spec_input.value = f"http://127.0.0.1:{port}/fast.json"
                await pilot.press("enter")
                await _until(pilot, lambda: app.parsed is not None)
                # the first spec to land must be the newer one, never the superseded slow one
                assert app.parsed is not None and app.parsed.title == "FAST API"
                generation = app.spec_generation
                await _settle(pilot, 1.2)  # the slow answer arrives now — and is dropped
                assert app.parsed.title == "FAST API"
                assert app.spec_generation == generation
                assert app.settings.spec == f"http://127.0.0.1:{port}/fast.json"
                assert "FAST API" in _text(app, "#spec-status")
                await pilot.press("enter")
                await pilot.pause()
                assert app.pane.id == "model"

    async def test_a_newer_load_cuts_the_old_download_short(self, tmp_path: Path) -> None:
        fast = json.dumps({**SPEC, "info": {**SPEC["info"], "title": "FAST API"}}).encode()
        trickle, dropped = _trickle(period=0.2)

        def respond(h: BaseHTTPRequestHandler) -> None:
            if h.path == "/slow.json":
                trickle(h)
            else:
                _send(h, fast)

        with _loopback(respond) as port:
            app = _app(tmp_path)
            async with app.run_test(size=(100, 34)) as pilot:
                await pilot.press("enter")
                await pilot.pause()
                app.query_one("#spec-input").value = f"http://127.0.0.1:{port}/slow.json"
                await pilot.press("enter")
                await pilot.pause(0.5)  # the slow download is in flight
                assert app.query_one("#spec-loading").display
                app.query_one("#spec-input").value = f"http://127.0.0.1:{port}/fast.json"
                superseded = time.monotonic()
                await pilot.press("enter")
                await _settle(pilot)
                assert app.parsed is not None and app.parsed.title == "FAST API"
                assert dropped.wait(2)  # abandoned within a chunk, not read to the end
                assert time.monotonic() - superseded < 3
                await _settle(pilot, 0.5)  # nothing from the old load lands later
                assert app.parsed.title == "FAST API"
                assert "FAST API" in _text(app, "#spec-status")
                assert app.settings.spec == f"http://127.0.0.1:{port}/fast.json"

    def test_ctrl_q_does_not_wait_for_a_trickling_download(self, tmp_path: Path) -> None:
        # The real exit path, as run_wizard() takes it: App.run() -> asyncio.run(),
        # whose shutdown joins the executor thread the loader runs on. A download
        # that ignored cancellation would hold the terminal until its last byte.
        trickle, dropped = _trickle(period=0.2)
        pressed: list[float] = []

        with _loopback(trickle) as port:
            app = _app(tmp_path)

            async def script(pilot) -> None:
                await pilot.press("enter")
                await pilot.pause()
                app.query_one("#spec-input").value = f"http://127.0.0.1:{port}/openapi.json"
                await pilot.press("enter")
                await pilot.pause(0.6)  # the download is in flight
                assert app.query_one("#spec-loading").display
                pressed.append(time.monotonic())
                await pilot.press("ctrl+q")

            result = app.run(headless=True, size=(100, 34), auto_pilot=script)
            returned = time.monotonic()
            assert dropped.wait(2)  # the server saw the client go, not a full read
        assert result is None and pressed
        assert returned - pressed[0] < 3  # not the 20+ s the trickle would take

    async def test_credentials_in_the_url_reach_the_fetch_and_nothing_else(
        self, tmp_path: Path
    ) -> None:
        auth_headers: list[str | None] = []

        def respond(h: BaseHTTPRequestHandler) -> None:
            auth_headers.append(h.headers.get("Authorization"))
            _send(h, json.dumps(SPEC).encode())

        with _loopback(respond) as port:
            app = _app(tmp_path)
            async with app.run_test(size=(100, 34)) as pilot:
                await pilot.press("enter")
                await pilot.pause()
                secret = (
                    f"http://svc:S3CRET-TOKEN@127.0.0.1:{port}/openapi.json?api_key=QUERY-SECRET"
                )
                public = f"http://127.0.0.1:{port}/openapi.json"
                app.query_one("#spec-input").value = secret
                await pilot.press("enter")
                await _settle(pilot)
                assert auth_headers and auth_headers[0].startswith("Basic ")  # used for the fetch
                assert app.parsed is not None and app.parsed.source == public
                assert app.settings.spec == public
                assert app.query_one("#spec-input").value == public
                assert "Credentials removed" in _text(app, "#spec-status")
                await pilot.press("enter")  # Enter again continues: no second fetch
                await pilot.press("down", "enter")
                await pilot.press("enter")
                await pilot.press("enter")
                await pilot.press("enter")
                await _settle(pilot)
                await pilot.press("enter")  # write
                await _settle(pilot)
                assert app.result is not None
                assert (
                    app.result.command == f"promptise mcpcast {public} --auth env-token --no-curate"
                )
                assert len(auth_headers) == 1
        out = tmp_path / "widgets-mcp"
        for path in out.rglob("*"):
            if path.is_file():
                assert "S3CRET-TOKEN" not in path.read_text(
                    encoding="utf-8"
                ) and "QUERY-SECRET" not in path.read_text(encoding="utf-8")
        assert (
            MCPcastPlan.from_yaml(
                (out / "mcpcast.plan.yaml").read_text(encoding="utf-8")
            ).api.spec_source
            == public
        )

    async def test_a_failed_fetch_never_echoes_the_credential(self, tmp_path: Path) -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await pilot.pause()
            app.query_one("#spec-input").value = "https://u:p@127.0.0.1:1/openapi.json?api_key=x"
            await pilot.press("enter")
            await _until(pilot, lambda: bool(_text(app, "#spec-error")))  # port 1 refuses
            assert app.parsed is None and app.pane.id == "spec"
            error = _text(app, "#spec-error")
            assert error and "u:p" not in error and "api_key" not in error
            assert "https://127.0.0.1:1/openapi.json" in error

    async def test_load_failures_are_shown_not_raised(self, tmp_path: Path) -> None:
        app = _app(tmp_path)
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await pilot.pause()
            app.query_one("#spec-input").value = '{"a":' * 100_000 + "1" + "}" * 100_000
            await pilot.press("enter")
            await _settle(pilot, 1.0)
            assert app.parsed is None and app.pane.id == "spec"
            assert _text(app, "#spec-error")
            app.query_one("#spec-input").value = json.dumps(SPEC)
            await pilot.press("enter")
            await _settle(pilot)
            assert app.parsed is not None  # the wizard is still usable

    async def test_detection_failures_and_skips_are_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import promptise.mcpcast.wizard as wizard

        def hostile(url: str) -> str | None:
            return json.dumps({"openapi": "3.0.0", "info": "oops", "paths": {"/x": {"get": {}}}})

        app = _app(tmp_path, fetch=hostile)
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await pilot.pause()
            await pilot.click("#spec-detect")
            await _settle(pilot)
            assert "No local API found" in _text(app, "#spec-status")  # not a crash

            def out_of_time(**kwargs: object) -> Detection:
                return Detection(candidates=[], skipped_ports=(8888, 9000), elapsed=10.2)

            monkeypatch.setattr(wizard, "probe_local_apis", out_of_time)
            await pilot.click("#spec-detect")
            await _settle(pilot)
            status = _text(app, "#spec-status")
            assert "Stopped after 10 s: ports 8888, 9000 were not checked" in status
            assert "9000" not in status.split("Stopped")[0]  # not listed as probed

            def broken(**kwargs: object) -> Detection:
                raise RuntimeError("boom")

            monkeypatch.setattr(wizard, "probe_local_apis", broken)
            await pilot.click("#spec-detect")
            await _settle(pilot)
            assert "Detection failed: RuntimeError: boom" in _text(app, "#spec-error")
            assert app.pane.id == "spec"

    async def test_square_brackets_in_user_text_do_not_crash_the_wizard(
        self, tmp_path: Path, spec_file: Path
    ) -> None:
        app = _app(tmp_path, str(spec_file))
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            app.query_one("#spec-base-url").value = "api.example.com/[/v1]"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.pane.id == "spec"
            assert "API base URL" in _text(app, "#spec-error")
            app.query_one("#spec-base-url").value = "https://w.example.com"
            await _settle(pilot)
            await pilot.press("enter")
            await pilot.press("down", "enter")  # offline
            await pilot.press("enter")  # read-only
            await pilot.press("enter")  # personal
            await pilot.pause()
            assert app.pane.id == "project"
            (tmp_path / "[red]x[bold]").mkdir()
            (tmp_path / "[red]x[bold]" / "keep").write_text("", encoding="utf-8")
            app.query_one("#project-out").value = "[red]x[bold]"
            await pilot.pause()
            await pilot.press("enter")
            await pilot.pause()
            assert app.pane.id == "project"
            assert "[red]x[bold]/ has files in it" in _text(app, "#project-error")

    async def test_a_failing_write_is_shown_and_leaves_no_result(
        self, tmp_path: Path, spec_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import promptise.mcpcast.wizard as wizard

        def refuse(plan: object, out_dir: object, **kwargs: object) -> list[Path]:
            raise UnicodeEncodeError("utf-8", "\ud83d", 0, 1, "surrogates not allowed")

        monkeypatch.setattr(wizard, "write_project", refuse)
        app = _app(tmp_path, str(spec_file))
        async with app.run_test(size=(100, 34)) as pilot:
            await _to_review(pilot, app)
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")  # write
            await _settle(pilot)
            assert app.pane.id == "write" and app.result is None
            error = _text(app, "#write-error")
            assert "Could not write" in error and "UnicodeEncodeError" in error
            await pilot.press("enter")  # Finish
        assert app.return_value is None

    async def test_copy_says_the_terminal_may_not_support_it(
        self, tmp_path: Path, spec_file: Path
    ) -> None:
        app = _app(tmp_path, str(spec_file))
        async with app.run_test(size=(100, 34)) as pilot:
            await _to_review(pilot, app)
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")  # write
            await _settle(pilot)
            await pilot.press("c")
            await pilot.pause()
            notifications = [n.message for n in app._notifications]
            assert any("OSC 52" in n and "not every terminal" in n for n in notifications)

    async def test_model_eval_tasks_and_force_prefill(
        self, tmp_path: Path, spec_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        app = MCPcastWizard(
            spec=str(spec_file),
            cwd=tmp_path,
            auto_detect=False,
            model="anthropic:claude-sonnet-4.5",
            eval_tasks=5,
            force=True,
        )
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")  # → model
            await pilot.pause()
            assert app.query_one("#model-input").value == "anthropic:claude-sonnet-4.5"
            assert "✓ anthropic:claude-sonnet-4.5" in _text(app, "#model-status")
            await pilot.press("enter")  # design with a model
            await pilot.press("enter")  # read-only
            await pilot.press("enter")  # personal
            await pilot.pause()
            assert app.pane.id == "project"
            assert app.query_one("#project-force").value is True
            assert app.settings.eval_tasks == 5
            app.query_one("#project-eval").value = True
            await pilot.pause()
            app.query_one("#project-eval").value = False
            await pilot.pause()
            assert equivalent_command(app.settings).endswith(
                "--auth env-token --model anthropic:claude-sonnet-4.5 --force"
            )
            app.settings.evaluate = True
            assert equivalent_command(app.settings).endswith("--force --eval --eval-tasks 5")


# ---------------------------------------------------------------------------
# Round-2 audit: terminal safety, stale plans, links, large text, quoting
# ---------------------------------------------------------------------------

OSC52 = "\x1b]52;c;aGVsbG8=\x07"  # writes "hello" to the clipboard on terminals that honour it
ERASE = "\x1b[2K"  # erases the current line


def _hostile_spec() -> dict:
    """SPEC with terminal escape sequences in the title, a summary, a path and a parameter."""
    return {
        **SPEC,
        "info": {**SPEC["info"], "title": f"Widgets{OSC52} API", "version": f"2.1{ERASE}"},
        "paths": {
            **SPEC["paths"],
            f"/wipe{ERASE}": {
                "delete": {
                    "operationId": "wipeAll",
                    "summary": f"Wipe all data{ERASE}",
                    "parameters": [
                        {
                            "name": "confirm\x9b2K",
                            "in": "query",
                            "schema": {"type": "string"},
                        }
                    ],
                }
            },
        },
    }


def _clean(text: str) -> bool:
    """No terminal control character left: ESC, BEL, the C1 CSI byte, DEL."""
    return not any(ch in text for ch in ("\x1b", "\x07", "\x9b", "\x7f", "\x85"))


def _deep_spec(levels: int = 300) -> dict:
    """SPEC plus an operation whose request body nests *levels* ``properties`` deep."""
    schema: dict = {"type": "string"}
    for _ in range(levels):
        schema = {"type": "object", "properties": {"inner": schema}}
    return {
        **SPEC,
        "paths": {
            **SPEC["paths"],
            "/deep": {
                "post": {
                    "operationId": "deepPost",
                    "requestBody": {"content": {"application/json": {"schema": schema}}},
                }
            },
        },
    }


class TestTerminalSafety:
    """H3/L2: spec text reaches the terminal only after every control character is deleted."""

    def test_console_safe_deletes_controls_and_keeps_newline_and_tab(self) -> None:
        from promptise.mcpcast.wizard import _console_safe

        # the control bytes go; what they carried stays as harmless text
        assert _console_safe(f"a{ERASE}b\x00c\x7fd\x9be\x85f") == "a[2Kbcdef"
        assert _console_safe(f"t{OSC52}") == "t]52;c;aGVsbG8="
        assert _console_safe("line\n\ttab") == "line\n\ttab"

    def test_detection_candidates_and_documents_are_scrubbed(self) -> None:
        body = json.dumps(_hostile_spec())
        found = detect_local_apis(ports=(1,), paths=("/openapi.json",), fetch=lambda url: body)
        (candidate,) = found
        assert _clean(candidate.title) and candidate.title.startswith("Widgets")
        assert _clean(json.dumps(candidate.document))
        parsed = ParsedSpec.load(candidate.url, document=candidate.document)
        assert _clean(parsed.summary()) and "— 5 operations" in parsed.summary()
        # a document handed to load() directly is scrubbed too
        direct = ParsedSpec.load("http://127.0.0.1:1/openapi.json", document=_hostile_spec())
        assert _clean(direct.summary()) and _clean(direct.title)
        assert all(_clean(op.path) and _clean(op.summary) for op in direct.operations)
        assert all(_clean(p.name) for op in direct.operations for p in op.params)

    def test_a_document_over_the_depth_cap_is_not_a_candidate(self) -> None:
        body = json.dumps(_deep_spec())
        detection = probe_local_apis(ports=(1,), paths=("/openapi.json",), fetch=lambda url: body)
        assert detection.candidates == [] and detection.skipped_ports == ()
        with pytest.raises(MCPcastError, match="nested more than 256 levels"):
            ParsedSpec.load("http://127.0.0.1:1/openapi.json", document=_deep_spec())
        # the same document as text is refused by load_spec the same way
        with pytest.raises(MCPcastError, match="nested more than 256 levels"):
            ParsedSpec.load(body)

    def test_detection_honours_the_node_budget_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("MCPCAST_MAX_SPEC_NODES", "20")
        body = json.dumps(SPEC)
        detection = probe_local_apis(ports=(1,), paths=("/openapi.json",), fetch=lambda url: body)
        assert detection.candidates == []
        monkeypatch.delenv("MCPCAST_MAX_SPEC_NODES")
        assert (
            len(detect_local_apis(ports=(1,), paths=("/openapi.json",), fetch=lambda url: body))
            == 1
        )

    async def test_nothing_rendered_on_the_review_step_carries_an_escape(
        self, tmp_path: Path
    ) -> None:
        body = json.dumps(_hostile_spec())

        def fetch(url: str) -> str | None:
            return body if url == "http://127.0.0.1:8000/openapi.json" else None

        app = _app(tmp_path, fetch=fetch)
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await pilot.pause()
            await pilot.click("#spec-detect")
            await _settle(pilot)
            prompt = str(app.query_one("#spec-candidates").get_option_at_index(0).prompt)
            assert _clean(prompt) and "Widgets" in prompt
            await pilot.press("enter")  # pick it
            await _settle(pilot)
            assert app.parsed is not None
            assert "\x1b" not in _text(app, "#spec-status")
            await pilot.press("enter")  # → model
            await pilot.press("down", "enter")  # offline
            await pilot.press("down", "down", "enter")  # full: the wipe operation is a tool
            await pilot.press("enter")  # personal
            await pilot.press("enter")  # project defaults
            await _settle(pilot)
            assert app.pane.id == "review" and app.plan is not None
            table = app.query_one("#review-table")
            cells = [str(c) for row in range(table.row_count) for c in table.get_row_at(row)]
            assert any("wipe" in c for c in cells)
            rendered = "\n".join(
                [
                    *cells,
                    _text(app, "#review-log"),
                    app.query_one("#review-tool").source,
                    app.query_one("#review-dropped-list").source,
                ]
            )
            assert _clean(rendered)
            assert _clean(app.plan.to_yaml())


class TestStalePlanIsNeverWritten:
    """H4: Enter while the plan is rebuilt writes nothing — not the previous plan."""

    async def test_enter_during_a_rebuild_is_refused(
        self, tmp_path: Path, spec_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import asyncio

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        calls = 0
        full_proposal = {
            **PROPOSAL,
            "tools": [
                *PROPOSAL["tools"],
                {
                    "name": "delete_widget",
                    "description": "Delete a widget for good.",
                    "risk": "destructive",
                    "operations": ["deleteWidget"],
                },
            ],
            "dropped": [],
        }

        async def complete(system: str, user: str) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                return json.dumps(PROPOSAL)
            await asyncio.sleep(1.5)  # the re-curation for the new profile takes a while
            return json.dumps(full_proposal)

        app = _app(tmp_path, str(spec_file), completer=complete)
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")  # → model
            await pilot.press("enter")  # design with a model
            await pilot.press("enter")  # read-only
            await pilot.press("enter")  # personal
            await pilot.press("enter")  # project defaults
            await _settle(pilot)
            assert app.pane.id == "review" and app.plan is not None
            first_plan = app.plan
            assert [t.name for t in first_plan.tools] == ["find_widgets"]  # read-only
            for _ in range(3):
                await pilot.press("escape")
            await pilot.pause()
            assert app.pane.id == "safety"
            await pilot.press("down", "down", "enter")  # full
            await pilot.press("enter")  # auth
            await pilot.press("enter")  # project → review rebuilds (call 2, sleeping)
            await pilot.pause()
            assert app.pane.id == "review" and app.plan is first_plan  # old plan still in memory
            assert app.plan_built_for != app.plan_key()
            assert app.query_one("#review-loading").display
            assert app.focused is app.query_one("#review-back")  # not a hidden Input
            app.action_next()  # what Enter on a submitted field would do
            await pilot.pause()
            assert app.pane.id == "review" and app.result is None
            assert "being rebuilt" in _text(app, "#review-error")
            assert not (tmp_path / "widgets-mcp").exists()
            await pilot.press("enter")  # Enter itself lands on Back: harmless
            await pilot.pause()
            assert app.pane.id == "project" and app.result is None
            await pilot.press("enter")  # return to the review; the rebuild is still running
            await _settle(pilot, 2.0)  # …and now it has landed
            assert app.pane.id == "review"
            assert app.plan is not first_plan and app.plan_built_for == app.plan_key()
            assert [t.name for t in app.plan.tools] == [
                "find_widgets",
                "create_widget",
                "delete_widget",
            ]
            await pilot.press("enter")  # write — the plan for the chosen settings
            await _settle(pilot)
            assert app.pane.id == "write" and app.result is not None
            assert app.result.plan.profile is SafetyProfile.FULL
            assert "--profile full" in app.result.command
        written = MCPcastPlan.from_yaml(
            (tmp_path / "widgets-mcp" / "mcpcast.plan.yaml").read_text(encoding="utf-8")
        )
        assert written.profile is SafetyProfile.FULL and "delete_widget" in written.tool_names

    async def test_a_failed_rebuild_keeps_the_old_plan_unwritable(
        self, tmp_path: Path, spec_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        calls = 0

        async def complete(system: str, user: str) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                return json.dumps(PROPOSAL)
            raise RuntimeError("provider down")

        app = _app(tmp_path, str(spec_file), completer=complete)
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")
            await pilot.press("enter")
            await pilot.press("enter")  # read-only
            await pilot.press("enter")
            await pilot.press("enter")
            await _settle(pilot)
            assert app.pane.id == "review" and app.plan is not None
            for _ in range(3):
                await pilot.press("escape")
            await pilot.pause()
            await pilot.press("down", "down", "enter")  # full
            await pilot.press("enter")
            await pilot.press("enter")
            await _settle(pilot)
            assert app.pane.id == "review"
            assert "provider down" in _text(app, "#review-error")
            assert app.plan is not None and app.plan_built_for != app.plan_key()
            assert app.query_one("#review-next").disabled
            app.action_next()
            await pilot.pause()
            assert app.pane.id == "review" and app.result is None
            assert "being rebuilt" in _text(app, "#review-error")
            # belt and braces: the write step itself refuses a stale plan
            app._show(7)
            await pilot.pause()
            assert app.pane.id == "review" and app.result is None
            assert any("being rebuilt" in n.message for n in app._notifications)
        assert not (tmp_path / "widgets-mcp").exists()

    async def test_enter_during_the_evaluation_does_not_finish_early(
        self, tmp_path: Path, spec_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """While the evaluation runs, Finish is disabled and Enter must not land
        on a hidden field whose submit would exit with the report missing."""
        import asyncio

        import promptise.mcpcast.readiness as readiness

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        seen: dict[str, object] = {}
        release = asyncio.Event()  # the evaluation ends when the test says so, not on a clock

        async def fake_evaluate(plan: MCPcastPlan, build_server: object, **kwargs: object):
            seen["model"] = kwargs.get("model")
            await release.wait()
            tool = plan.tools[0].name
            task = readiness.EvalTask(id="t1", prompt="list them", expected_tool=tool)
            result = readiness.TaskResult(
                task=task,
                calls=[readiness.ToolCall(tool=tool, arguments={"id": "w\x1b[2K1"})],
                success=True,
                selected_correctly=True,
            )
            return readiness.score(plan, [result])

        monkeypatch.setattr(readiness, "evaluate", fake_evaluate)
        app = _app(tmp_path, str(spec_file), completer=_scripted(PROPOSAL))
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")  # → model
            await pilot.press("enter")  # design with a model
            await pilot.press("enter")  # read-only
            await pilot.press("enter")  # personal
            await pilot.pause()
            assert app.pane.id == "project"
            app.query_one("#project-eval").value = True
            await pilot.pause()
            await pilot.press("enter")  # → review (scripted curation)
            await _settle(pilot)
            assert app.pane.id == "review"
            await pilot.press("enter")  # write; the evaluation starts
            await pilot.pause()
            assert app.pane.id == "write" and app.result is not None
            assert app.result.eval_requested and app.result.report is None
            assert app.query_one("#write-loading").display
            assert app.query_one("#write-next-btn").disabled
            assert app.focused is app.query_one("#write-copy")
            await pilot.press("enter")  # Copy: harmless, the wizard stays
            await pilot.pause()
            assert app.pane.id == "write" and app.is_running
            release.set()  # now let the evaluation finish
            await _settle(pilot, 1.5)
            assert app.result.report is not None and app.result.report.grade == "A"
            assert seen["model"] == DEFAULT_MODEL
            shown = _text(app, "#write-eval")
            assert "Agent Readiness  A" in shown and "1/1 tasks" in shown
            assert app.focused is app.query_one("#write-next-btn")
            await pilot.press("enter")  # Finish, now that the report is in
        assert app.return_value is app.result
        assert (tmp_path / "widgets-mcp" / "eval" / "report.md").exists()


class TestMarkdownLinks:
    """M4: a link in spec or model text opens only when it is http(s)."""

    async def test_only_http_links_are_opened(
        self, tmp_path: Path, spec_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import webbrowser

        from textual.widgets import Markdown

        opened: list[str] = []
        monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: opened.append(url) or True)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        proposal = {
            **PROPOSAL,
            "tools": [
                {
                    **PROPOSAL["tools"][0],
                    "description": "[docs](ssh://evil.example/x) first, then [site](https://ok.example/).",
                },
                PROPOSAL["tools"][1],
            ],
        }
        app = _app(tmp_path, str(spec_file), completer=_scripted(proposal))
        # tall enough for the description paragraph to be on screen for the click
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")
            await pilot.press("enter")
            await pilot.press("enter")
            await pilot.press("enter")
            await pilot.press("enter")
            await _settle(pilot)
            assert app.pane.id == "review"
            detail = app.query_one("#review-tool", Markdown)
            assert "ssh://evil.example/x" in detail.source
            detail.post_message(Markdown.LinkClicked(detail, "ssh://evil.example/x"))
            # the notice is posted asynchronously; wait for it rather than one frame
            await _until(
                pilot,
                lambda: any(
                    "Link not opened: ssh://evil.example/x" in n.message for n in app._notifications
                ),
            )
            assert opened == []
            for href in (
                "x-apple.systempreferences:com.apple.preference.security",
                "//evil/share/p.exe",
            ):
                detail.post_message(Markdown.LinkClicked(detail, href))
                await pilot.pause()
            assert opened == []
            detail.post_message(Markdown.LinkClicked(detail, "https://ok.example/"))
            await pilot.pause()
            assert opened == ["https://ok.example/"]
            # a real click on the rendered link goes the same way
            from textual.widgets._markdown import MarkdownParagraph

            paragraph = next(
                b for b in detail.query(MarkdownParagraph) if "docs" in str(b._content)
            )
            before = sum("Link not opened" in n.message for n in app._notifications)
            await pilot.click(paragraph, offset=(1, 0))  # on the word "docs"
            await pilot.pause()
            assert opened == ["https://ok.example/"]  # ssh link clicked: still not opened
            after = [n.message for n in app._notifications if "Link not opened" in n.message]
            assert len(after) == before + 1 and after[-1].endswith("ssh://evil.example/x")


class TestLargeText:
    """L9: a huge description or drop list must not stall the review pane."""

    def test_tool_markdown_is_clamped(self) -> None:
        from promptise.mcpcast.wizard import _DETAIL_MAX_CHARS, _tool_markdown

        spec = json.loads(json.dumps(SPEC))
        spec["paths"]["/widgets"]["get"]["description"] = "\n".join(
            f"- item {i}" for i in range(5000)
        )
        plan = mcpcast(spec, name="widgets")
        md = _tool_markdown(plan.tool("list_widgets"))
        assert md.count("- item ") <= 50
        note = re.search(r"\*… (\d+) more lines — full text in mcpcast.plan.yaml\*", md)
        assert note and int(note.group(1)) >= 4950
        assert len(md) < _DETAIL_MAX_CHARS + 1000
        spec["paths"]["/widgets"]["get"]["description"] = "x" * 20_000
        long_line = _tool_markdown(mcpcast(spec, name="widgets").tool("list_widgets"))
        assert "*… truncated — full text in mcpcast.plan.yaml*" in long_line
        assert len(long_line) < _DETAIL_MAX_CHARS + 1000

    def test_parameters_hidden_and_example_are_capped(self) -> None:
        from promptise.mcpcast.wizard import _tool_markdown

        spec = json.loads(json.dumps(SPEC))
        spec["paths"]["/widgets"]["post"]["requestBody"] = {
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {
                            f"field_{i}": {"type": "string", "description": "d " * 400}
                            for i in range(150)
                        },
                    }
                }
            }
        }
        plan = mcpcast(spec, name="widgets", profile=SafetyProfile.STANDARD)
        tool = plan.tool("create_widget")
        tool.example = {"payload": "p" * 20_000}
        md = _tool_markdown(tool)
        assert md.count("\n- `field_") == 100
        assert "*… and 50 more parameters — full text in mcpcast.plan.yaml*" in md
        assert all(len(line) < 400 for line in md.splitlines() if line.startswith("- `field_"))
        assert len(md) < 60_000

    def test_dropped_list_is_capped(self) -> None:
        from promptise.mcpcast.wizard import _dropped_markdown

        spec = json.loads(json.dumps(SPEC))
        spec["paths"] = {
            **SPEC["paths"],
            **{f"/things/{i}": {"delete": {"operationId": f"deleteThing{i}"}} for i in range(250)},
        }
        plan = mcpcast(spec, name="widgets")  # read-only: every delete is dropped
        md = _dropped_markdown(plan)
        assert md.count("- `") == 200
        assert "- *… and 52 more — full text in mcpcast.plan.yaml*" in md
        assert _dropped_markdown(mcpcast(SPEC, name="w", profile=SafetyProfile.FULL)) == (
            "*Every operation is exposed.*"
        )

    async def test_review_is_ready_quickly_with_a_5000_item_description(
        self, tmp_path: Path
    ) -> None:
        spec = json.loads(json.dumps(SPEC))
        spec["paths"]["/widgets"]["get"]["description"] = "\n".join(
            f"- item {i}" for i in range(5000)
        )
        path = tmp_path / "big.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        app = _app(tmp_path, str(path))
        async with app.run_test(size=(100, 34)) as pilot:
            await _to_review(pilot, app)
            started = time.monotonic()
            await pilot.press("enter")  # project defaults → review (offline)
            await _settle(pilot)
            assert app.pane.id == "review" and app.plan is not None
            detail = app.query_one("#review-tool").source
            assert "more lines — full text in mcpcast.plan.yaml" in detail
            await pilot.press("down", "up")  # every cursor move re-renders the detail
            await _settle(pilot)
            # Unclamped, 5,000 items took over a minute; a generous bound keeps the
            # test meaningful on a loaded CI runner.
            assert time.monotonic() - started < 20


class TestNextSteps:
    """L12/L15: every path is quoted, and the Measure line carries the eval credential."""

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX paths and quoting")
    def test_posix_lines_are_quoted_and_carry_the_credential(self) -> None:
        from promptise.mcpcast.wizard import _next_steps_markdown

        plan = mcpcast(SPEC, name="widgets", auth=AuthMode.ENV_TOKEN)
        md = _next_steps_markdown(plan, Path("My Projects/widgets-mcp"), windows=False)
        assert "`cd 'My Projects/widgets-mcp' && pytest`" in md
        assert "`promptise mcpcast 'My Projects/widgets-mcp/mcpcast.plan.yaml'`" in md
        assert (
            "`claude mcp add widgets -e 'MCPCAST_UPSTREAM_TOKEN=Bearer <your API token>' -- python "
            "'My Projects/widgets-mcp/server.py'`"
        ) in md
        assert (
            "`MCPCAST_EVAL_AUTHORIZATION='Bearer <your API token>' promptise mcpcast "
            "'My Projects/widgets-mcp/mcpcast.plan.yaml' --eval`"
        ) in md
        # no credential is involved: no env prefix, no -e flag
        open_md = _next_steps_markdown(
            mcpcast(SPEC, name="widgets", auth=AuthMode.NONE), Path("out"), windows=False
        )
        assert "`claude mcp add widgets -- python out/server.py`" in open_md
        assert "`promptise mcpcast out/mcpcast.plan.yaml --eval`" in open_md
        assert "MCPCAST_EVAL_AUTHORIZATION" not in open_md
        # header-bearing modes cannot be launched over stdio: say so instead
        shared = _next_steps_markdown(
            mcpcast(SPEC, name="widgets", auth=AuthMode.PASSTHROUGH), Path("out"), windows=False
        )
        assert "claude mcp add" not in shared
        assert (
            "`python out/server.py --transport http`" in shared
            and "`Authorization` header" in shared
        )
        assert "MCPCAST_EVAL_AUTHORIZATION='Bearer <your API token>' promptise mcpcast" in shared

    def test_windows_lines_use_double_quotes_and_a_set_line(self) -> None:
        from promptise.mcpcast.wizard import _next_steps_markdown

        plan = mcpcast(SPEC, name="widgets", auth=AuthMode.ENV_TOKEN)
        out = Path("D:\\My Projects\\widgets-mcp")  # joined by pathlib: \\ on Windows, / here
        md = _next_steps_markdown(plan, out, windows=True)
        assert f'`cd "{out}" && pytest`' in md
        assert (
            '`claude mcp add widgets -e "MCPCAST_UPSTREAM_TOKEN=Bearer <your API token>" -- python '
            f'"{out / "server.py"}"`'
        ) in md
        assert '`set "MCPCAST_EVAL_AUTHORIZATION=Bearer <your API token>"`' in md
        assert '`$env:MCPCAST_EVAL_AUTHORIZATION = "Bearer <your API token>"`' in md
        assert f'then `promptise mcpcast "{out / "mcpcast.plan.yaml"}" --eval`' in md

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX paths and quoting")
    async def test_the_write_step_shows_the_quoted_lines(
        self, tmp_path: Path, spec_file: Path
    ) -> None:
        app = _app(tmp_path, str(spec_file))
        async with app.run_test(size=(100, 34)) as pilot:
            await _to_review(pilot, app)
            app.query_one("#project-out").value = "my widgets"
            await pilot.pause()
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")  # write
            await _settle(pilot)
            assert app.result is not None
            md = app.query_one("#write-next").source
            assert "`cd 'my widgets' && pytest`" in md
            assert "MCPCAST_EVAL_AUTHORIZATION='Bearer <your API token>' promptise mcpcast" in md
            assert "'my widgets/mcpcast.plan.yaml' --eval" in md


class TestPlainHttpWarning:
    """M10: a plain-http upstream is said on the review and write steps."""

    async def test_review_and_write_name_the_insecure_host(self, tmp_path: Path) -> None:
        spec = {**SPEC, "servers": [{"url": "http://api.intranet.example:8080"}]}
        path = tmp_path / "intranet.json"
        path.write_text(json.dumps(spec), encoding="utf-8")
        app = _app(tmp_path, str(path))
        async with app.run_test(size=(100, 34)) as pilot:
            await _to_review(pilot, app)  # personal → env-token: a credential travels
            await pilot.press("enter")
            await _settle(pilot)
            assert app.pane.id == "review"
            log = _text(app, "#review-log")
            assert "plain http upstream: api.intranet.example:8080" in log
            assert "MCPCAST_ALLOW_INSECURE_HTTP=1" in log
            await pilot.press("enter")  # write
            await _settle(pilot)
            assert app.result is not None
            warning = _text(app, "#write-warning")
            assert "api.intranet.example:8080 is plain http" in warning
            assert "MCPCAST_ALLOW_INSECURE_HTTP=1" in warning and "UPSTREAM_INSECURE" in warning

    async def test_no_warning_for_https_or_no_credential(
        self, tmp_path: Path, spec_file: Path
    ) -> None:
        app = _app(tmp_path, str(spec_file))  # https://w.example.com
        async with app.run_test(size=(100, 34)) as pilot:
            await _to_review(pilot, app)
            await pilot.press("enter")
            await _settle(pilot)
            assert "plain http" not in _text(app, "#review-log")
            await pilot.press("enter")
            await _settle(pilot)
            assert _text(app, "#write-warning") == ""


class TestClippedDescriptionWarning:
    """M23: a description the generated tool cannot carry in full is flagged on review."""

    async def test_review_summary_names_the_clipped_tool(
        self, tmp_path: Path, spec_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from promptise.mcpcast.emit import clipped_descriptions

        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        proposal = {
            **PROPOSAL,
            "tools": [
                {**PROPOSAL["tools"][0], "description": "Find widgets. " + "More detail. " * 400},
                PROPOSAL["tools"][1],
            ],
        }
        app = _app(tmp_path, str(spec_file), completer=_scripted(proposal))
        async with app.run_test(size=(100, 34)) as pilot:
            await pilot.press("enter")
            await _settle(pilot)
            for _ in range(5):
                await pilot.press("enter")
            await _settle(pilot)
            assert app.pane.id == "review" and app.plan is not None
            assert clipped_descriptions(app.plan) == ["find_widgets"]
            log = _text(app, "#review-log")
            assert "⚠ description cut for the agent: find_widgets" in log


class TestWriteRefusalsAndRenamedPackages:
    """write_project's own refusals are shown as they are; a renamed api.name is warned about."""

    async def test_a_refusal_from_write_project_is_shown_verbatim(
        self, tmp_path: Path, spec_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import promptise.mcpcast.wizard as wizard

        def refuse(plan: object, out_dir: object, **kwargs: object) -> list[Path]:
            raise MCPcastError("widgets_mcp/tools/extra.py exists and was not generated by mcpcast")

        monkeypatch.setattr(wizard, "write_project", refuse)
        app = _app(tmp_path, str(spec_file))
        async with app.run_test(size=(100, 34)) as pilot:
            await _to_review(pilot, app)
            await pilot.press("enter")
            await _settle(pilot)
            await pilot.press("enter")  # write
            await _settle(pilot)
            assert app.pane.id == "write" and app.result is None
            error = _text(app, "#write-error")
            assert error.startswith("widgets_mcp/tools/extra.py exists")
            assert "Could not write" not in error and "MCPcastError" not in error

    async def test_a_package_from_a_previous_name_is_named_before_writing(
        self, tmp_path: Path, spec_file: Path
    ) -> None:
        from promptise.mcpcast import write_project

        out = tmp_path / "shop-mcp"
        write_project(mcpcast(SPEC, name="shop"), out)  # shop_mcp/ under the old name
        app = _app(tmp_path, str(spec_file))
        async with app.run_test(size=(100, 34)) as pilot:
            await _to_review(pilot, app)
            app.query_one("#project-name").value = "widgets"
            app.query_one("#project-out").value = "shop-mcp"
            await pilot.pause()
            label = _text(app, "#project-force-label")
            assert app.query_one("#project-force-row").display
            assert "It holds shop_mcp/, generated under another name" in label
            assert "writes widgets_mcp/ beside it" in label and "Dockerfile" in label
            await pilot.press("enter")  # overwrite is off: refused, with the same note
            await pilot.pause()
            assert app.pane.id == "project"
            error = _text(app, "#project-error")
            assert "already contains an mcpcast project" in error
            assert "It holds shop_mcp/, generated under another name" in error
            app.query_one("#project-name").value = "shop"  # the previous name: nothing to say
            await pilot.pause()
            assert "generated under another name" not in _text(app, "#project-force-label")
        assert (out / "shop_mcp").is_dir() and not (out / "widgets_mcp").exists()


class TestDotenvErrors:
    """A .env that cannot be read is reported on the welcome screen, not raised."""

    @pytest.mark.skipif(sys.platform == "win32", reason="chmod 000 does not block reads")
    async def test_unreadable_dotenv_is_shown_on_welcome(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import os

        if hasattr(os, "geteuid") and os.geteuid() == 0:
            pytest.skip("root can read a chmod-000 file")
        monkeypatch.delenv("PROMPTISE_NO_DOTENV")
        dotenv = tmp_path / ".env"
        dotenv.write_text("OPENAI_API_KEY=sk-test\n", encoding="utf-8")
        dotenv.chmod(0)
        try:
            app = _app(tmp_path)
            async with app.run_test(size=(100, 34)) as pilot:
                await pilot.pause()
                assert app.pane.id == "welcome"
                assert app.dotenv is None and app.dotenv_error is not None
                assert str(dotenv) in app.dotenv_error
                assert "could not be read" in _text(app, "#welcome-env")
                error = _text(app, "#welcome-error")
                assert f"cannot read {dotenv}" in error
                await pilot.press("enter")  # the wizard goes on regardless
                await pilot.pause()
                assert app.pane.id == "spec"
        finally:
            dotenv.chmod(0o600)


class TestQuitDuringAParse:
    """L1: Ctrl+Q does not wait for a CPU-bound parse."""

    def test_ctrl_q_returns_while_a_slow_parse_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import promptise.mcpcast.wizard as wizard

        started_parse = threading.Event()

        def slow_load(cls, source: str, **kwargs: object) -> ParsedSpec:
            started_parse.set()
            time.sleep(8)  # a pure-Python YAML parse of a 13 MiB document, roughly
            raise AssertionError("never reached before the wizard exited")

        monkeypatch.setattr(wizard.ParsedSpec, "load", classmethod(slow_load))
        pressed: list[float] = []
        app = _app(tmp_path)

        async def script(pilot) -> None:
            await pilot.press("enter")
            await pilot.pause()
            app.query_one("#spec-input").value = "openapi.json"
            await pilot.press("enter")
            await pilot.pause(0.3)
            assert started_parse.is_set() and app.query_one("#spec-loading").display
            pressed.append(time.monotonic())
            await pilot.press("ctrl+q")

        result = app.run(headless=True, size=(100, 34), auto_pilot=script)
        returned = time.monotonic()
        assert result is None and pressed
        assert returned - pressed[0] < 3  # not the 8 s the parse takes

    async def test_a_superseded_parse_result_is_dropped(self, tmp_path: Path) -> None:
        import promptise.mcpcast.wizard as wizard

        real_load = ParsedSpec.load.__func__  # type: ignore[attr-defined]
        release = threading.Event()

        def gated_load(cls, source: str, **kwargs: object) -> ParsedSpec:
            if source.endswith("slow.json"):
                release.wait(5)
            return real_load(cls, source, **kwargs)

        slow = tmp_path / "slow.json"
        slow.write_text(
            json.dumps({**SPEC, "info": {**SPEC["info"], "title": "SLOW API"}}), encoding="utf-8"
        )
        fast = tmp_path / "fast.json"
        fast.write_text(
            json.dumps({**SPEC, "info": {**SPEC["info"], "title": "FAST API"}}), encoding="utf-8"
        )
        app = _app(tmp_path)
        async with app.run_test(size=(100, 34)) as pilot:
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(wizard.ParsedSpec, "load", classmethod(gated_load))
                await pilot.press("enter")
                await pilot.pause()
                app.query_one("#spec-input").value = str(slow)
                await pilot.press("enter")
                await pilot.pause(0.2)
                app.query_one("#spec-input").value = str(fast)
                await pilot.press("enter")
                await _settle(pilot)
                assert app.parsed is not None and app.parsed.title == "FAST API"
                generation = app.spec_generation
                release.set()  # the slow parse finishes now — and lands nowhere
                await _settle(pilot, 0.5)
                assert app.parsed.title == "FAST API" and app.spec_generation == generation
                assert "FAST API" in _text(app, "#spec-status")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli:
    def test_no_spec_without_a_terminal(self) -> None:
        result = CliRunner(env={"COLUMNS": "200"}).invoke(cli, ["mcpcast"])
        assert result.exit_code == 1
        assert "needs an interactive terminal" in _plain(result.output)
        assert "promptise mcpcast openapi.json --no-curate" in _plain(result.output)

    def test_wizard_rejects_other_flags(self, spec_file: Path) -> None:
        result = CliRunner(env={"COLUMNS": "200"}).invoke(cli, ["mcpcast", "--profile", "full"])
        assert result.exit_code == 2
        assert "cannot be combined with the guided setup" in _plain(result.output)
        result = CliRunner(env={"COLUMNS": "200"}).invoke(
            cli, ["mcpcast", str(spec_file), "-i", "--eval"]
        )
        assert result.exit_code == 2
        assert "--eval cannot be combined" in _plain(result.output)
        # --out and --base-url pre-fill the wizard; without a terminal that is the TTY error
        result = CliRunner(env={"COLUMNS": "200"}).invoke(
            cli, ["mcpcast", "-i", "--out", "x", "--base-url", "https://a"]
        )
        assert result.exit_code == 1
        assert "needs an interactive terminal" in _plain(result.output)

    def test_help_mentions_the_guided_setup(self) -> None:
        result = CliRunner(env={"COLUMNS": "200"}).invoke(cli, ["mcpcast", "--help"])
        assert result.exit_code == 0
        assert "guided setup" in _plain(result.output)
