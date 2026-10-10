"""The human behind a Promptise client can approve server-side approval gates.

A server-side gate (``ApprovalGateMiddleware`` + ``ElicitationApprover``, the
default on MCPcast-generated servers) asks the *calling client* to confirm a
gated tool call through MCP elicitation.  These tests drive a real
MCPcast-generated server over stdio against a fake upstream API and check the
upstream effect: an approving handler lets exactly the approved request reach
the API, a denying one (or none, a timeout, a crash, modified arguments) lets
nothing through.  ``build_agent(approval=...)`` routes the gate to the agent's
handler; the mapping rules of ``approval_elicitation_callback`` are covered
directly at the bottom.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from mcp import types
from pydantic import PrivateAttr

from promptise import build_agent
from promptise.agent import CallerContext, _caller_ctx_var, get_current_caller
from promptise.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    CallbackApprovalHandler,
    approval_elicitation_callback,
)
from promptise.approval_classifier import ApprovalRule, AutoApprovalClassifier
from promptise.config import StdioServerSpec
from promptise.mcp.client import InFlightToolCall, MCPClient, MCPMultiClient
from promptise.mcpcast import mcpcast, write_project
from promptise.mcpcast.schema import AuthMode, SafetyProfile

PETSTORE = {
    "openapi": "3.0.0",
    "info": {"title": "Petstore"},
    "servers": [{"url": "https://petstore.example/api/v3"}],
    "paths": {
        "/pet": {
            "post": {
                "operationId": "addPet",
                "summary": "Add a new pet to the store",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["name"],
                                "properties": {"name": {"type": "string"}},
                            }
                        }
                    },
                },
                "responses": {"200": {"description": "The created pet"}},
            }
        },
        "/pet/{petId}": {
            "get": {
                "operationId": "getPetById",
                "summary": "Find a pet by ID",
                "parameters": [
                    {
                        "name": "petId",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "integer"},
                    }
                ],
                "responses": {"200": {"description": "The pet"}},
            }
        },
    },
}


# ---------------------------------------------------------------------------
# Fixtures: a fake upstream API and a generated MCPcast server
# ---------------------------------------------------------------------------


class FakeUpstream:
    """A local HTTP API that records every request it receives."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def _answer(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length) if length else b""
                upstream.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "authorization": self.headers.get("authorization"),
                        "json": json.loads(body) if body else None,
                    }
                )
                payload = json.dumps({"id": 7, "status": "available"}).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = do_POST = _answer

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def upstream():
    api = FakeUpstream()
    yield api
    api.close()


@pytest.fixture(scope="module")
def petstore_server(tmp_path_factory) -> Path:
    """``promptise mcpcast <petstore> --no-curate --profile standard --auth env-token``."""
    plan = mcpcast(
        PETSTORE, name="petstore", profile=SafetyProfile.STANDARD, auth=AuthMode.ENV_TOKEN
    )
    assert plan.api.approval_mode.value == "elicitation"  # the default for env-token
    out = tmp_path_factory.mktemp("mcpcast") / "petstore-mcp"
    write_project(plan, out)
    return out / "server.py"


def _server_env(upstream: FakeUpstream) -> dict[str, str]:
    return {
        "MCPCAST_UPSTREAM_TOKEN": "Bearer demo",
        "MCPCAST_BASE_URL": upstream.base_url,
        "MCPCAST_ALLOW_INSECURE_HTTP": "1",  # the fake upstream is plain http on loopback
    }


def _client(server: Path, upstream: FakeUpstream, callback: Any = None) -> MCPClient:
    return MCPClient(
        transport="stdio",
        command=sys.executable,
        args=[str(server)],
        env=_server_env(upstream),
        elicitation_callback=callback,
    )


def _text(result: types.CallToolResult) -> str:
    return "".join(getattr(block, "text", "") for block in result.content)


class Recorder:
    """An approval callback that records each request and answers with *decision*."""

    def __init__(self, decision: Any = True, *, delay: float = 0.0, error: bool = False):
        self.requests: list[ApprovalRequest] = []
        self.callers: list[CallerContext | None] = []
        self._decision = decision
        self._delay = delay
        self._error = error

    async def __call__(self, request: ApprovalRequest) -> Any:
        self.requests.append(request)
        self.callers.append(get_current_caller())
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error:
            raise RuntimeError("approval UI crashed")
        return self._decision


# ---------------------------------------------------------------------------
# MCPClient against a real MCPcast-generated server
# ---------------------------------------------------------------------------


class TestServerGateOverStdio:
    async def test_without_a_handler_the_gate_denies_fail_closed(self, petstore_server, upstream):
        async with _client(petstore_server, upstream) as client:
            result = await client.call_tool("add_pet", {"name": "Rex"})
        assert "APPROVAL_DENIED" in _text(result)
        assert upstream.requests == []

    async def test_approving_handler_releases_exactly_the_approved_call(
        self, petstore_server, upstream
    ):
        recorder = Recorder(True)
        client = _client(
            petstore_server,
            upstream,
            approval_elicitation_callback(
                CallbackApprovalHandler(recorder),
                server_name="petstore",
                in_flight=lambda: client.in_flight_calls,
            ),
        )
        async with client:
            result = await client.call_tool("add_pet", {"name": "Rex"})
            assert client.in_flight_calls == []

        assert json.loads(_text(result))["id"] == 7
        assert upstream.requests == [
            {
                "method": "POST",
                "path": "/pet",
                "authorization": "Bearer demo",
                "json": {"name": "Rex"},
            }
        ]
        [request] = recorder.requests
        assert request.tool_name == "add_pet"
        assert request.arguments == {"name": "Rex"}
        # The reviewer sees the server's own message, which names the call too.
        assert request.context_summary.startswith("Server 'petstore' asks: ")
        assert "add_pet" in request.context_summary and "Rex" in request.context_summary
        assert request.metadata["source"] == "mcp_elicitation"
        assert request.metadata["server"] == "petstore"
        assert request.metadata["in_flight_tools"] == ["add_pet"]
        assert request.metadata["requested_schema"]["required"] == ["approve"]

    @pytest.mark.parametrize(
        "decision",
        [
            False,
            ApprovalDecision(approved=False, reason="not today"),
            ApprovalDecision(approved=True, modified_arguments={"name": "Max"}),
        ],
        ids=["deny", "deny-with-reason", "modified-arguments"],
    )
    async def test_anything_but_a_plain_approval_blocks_the_call(
        self, petstore_server, upstream, decision
    ):
        recorder = Recorder(decision)
        client = _client(
            petstore_server,
            upstream,
            approval_elicitation_callback(
                recorder, server_name="petstore", in_flight=lambda: client.in_flight_calls
            ),
        )
        async with client:
            result = await client.call_tool("add_pet", {"name": "Rex"})
        assert len(recorder.requests) == 1
        assert "APPROVAL_DENIED" in _text(result)
        assert upstream.requests == []

    async def test_timeout_declines_even_when_the_policy_allows_on_timeout(
        self, petstore_server, upstream
    ):
        recorder = Recorder(True, delay=5)
        policy = ApprovalPolicy(
            tools=["*"], handler=recorder, timeout=0.3, on_timeout="allow", redact_sensitive=False
        )
        async with _client(
            petstore_server, upstream, approval_elicitation_callback(policy)
        ) as client:
            result = await client.call_tool("add_pet", {"name": "Rex"})
        assert len(recorder.requests) == 1
        assert "APPROVAL_DENIED" in _text(result)
        assert upstream.requests == []

    async def test_handler_error_declines(self, petstore_server, upstream):
        recorder = Recorder(error=True)
        async with _client(
            petstore_server, upstream, approval_elicitation_callback(recorder)
        ) as client:
            result = await client.call_tool("add_pet", {"name": "Rex"})
        assert len(recorder.requests) == 1
        assert "APPROVAL_DENIED" in _text(result)
        assert upstream.requests == []

    async def test_ungated_reads_never_ask(self, petstore_server, upstream):
        recorder = Recorder(False)
        async with _client(
            petstore_server, upstream, approval_elicitation_callback(recorder)
        ) as client:
            result = await client.call_tool("get_pet_by_id", {"petId": 7})
        assert recorder.requests == []
        assert json.loads(_text(result))["id"] == 7
        assert [r["path"] for r in upstream.requests] == ["/pet/7"]


# ---------------------------------------------------------------------------
# build_agent routes server-side gates to the agent's approval handler
# ---------------------------------------------------------------------------


class _ScriptedModel(BaseChatModel):
    """Calls ``add_pet`` once, then answers — enough to drive the agent loop."""

    _turns: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._turns += 1
        if self._turns == 1:
            message = AIMessage(
                content="",
                tool_calls=[{"name": "add_pet", "args": {"name": "Rex"}, "id": "call_1"}],
            )
        else:
            message = AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=message)])


async def _run_agent(server: Path, upstream: FakeUpstream, approval: Any) -> str:
    agent = await build_agent(
        model=_ScriptedModel(),
        servers={
            "petstore": StdioServerSpec(
                command=sys.executable, args=[str(server)], env=_server_env(upstream)
            )
        },
        approval=approval,
    )
    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content="Add a pet called Rex")]},
            caller=CallerContext(user_id="alice"),
        )
    finally:
        await agent.shutdown()
    [tool_message] = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    return str(tool_message.content)


class TestBuildAgent:
    async def test_approving_handler_lets_the_agent_add_the_pet(self, petstore_server, upstream):
        recorder = Recorder(True)
        output = await _run_agent(petstore_server, upstream, CallbackApprovalHandler(recorder))
        assert '"id": 7' in output
        assert [(r["method"], r["path"], r["json"]) for r in upstream.requests] == [
            ("POST", "/pet", {"name": "Rex"})
        ]
        [request] = recorder.requests
        assert (request.tool_name, request.arguments) == ("add_pet", {"name": "Rex"})
        assert request.metadata["server"] == "petstore"
        # The handler runs in the invoking caller's context.
        assert request.caller_user_id == "alice"
        assert recorder.callers[0] is not None and recorder.callers[0].user_id == "alice"

    async def test_denying_handler_blocks_the_call(self, petstore_server, upstream):
        recorder = Recorder(False)
        output = await _run_agent(petstore_server, upstream, CallbackApprovalHandler(recorder))
        assert "APPROVAL_DENIED" in output
        assert len(recorder.requests) == 1
        assert upstream.requests == []

    async def test_policy_patterns_do_not_decide_server_gates(self, petstore_server, upstream):
        # The policy gates none of the agent's own tools; the server's gate
        # still reaches its handler — once, not twice.
        recorder = Recorder(True)
        policy = ApprovalPolicy(tools=["unrelated_*"], handler=recorder, redact_sensitive=False)
        output = await _run_agent(petstore_server, upstream, policy)
        assert '"id": 7' in output
        assert [r.tool_name for r in recorder.requests] == ["add_pet"]
        assert len(upstream.requests) == 1

    async def test_without_approval_the_gate_still_denies(self, petstore_server, upstream):
        output = await _run_agent(petstore_server, upstream, None)
        assert "APPROVAL_DENIED" in output
        assert upstream.requests == []

    async def test_rejects_an_approval_that_is_not_a_handler(self):
        with pytest.raises(TypeError, match="approval must be"):
            await build_agent(model=_ScriptedModel(), servers={}, approval="yes")


# ---------------------------------------------------------------------------
# Capability declaration and a crashing callback (minimal Promptise server)
# ---------------------------------------------------------------------------

_PROBE_SERVER = """
from promptise.mcp.server import Depends, Elicitor, MCPServer

server = MCPServer(name="probe")


@server.tool()
async def client_can_elicit(elicit: Elicitor = Depends(Elicitor)) -> bool:
    caps = elicit._session.client_params.capabilities
    return caps.elicitation is not None


@server.tool()
async def confirm(elicit: Elicitor = Depends(Elicitor)) -> str:
    answer = await elicit.ask("Proceed?")
    return "accepted" if answer is not None else "declined"


if __name__ == "__main__":
    server.run(transport="stdio")
"""


@pytest.fixture(scope="module")
def probe_server(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("probe") / "probe_server.py"
    path.write_text(_PROBE_SERVER, encoding="utf-8")
    return path


def _probe(server: Path, callback: Any = None) -> MCPClient:
    return MCPClient(
        transport="stdio",
        command=sys.executable,
        args=[str(server)],
        elicitation_callback=callback,
    )


class TestElicitationCapability:
    async def test_declared_only_when_a_callback_is_configured(self, probe_server):
        async def accept(context: Any, params: Any) -> types.ElicitResult:
            return types.ElicitResult(action="accept", content={})

        async with _probe(probe_server) as plain:
            assert _text(await plain.call_tool("client_can_elicit", {})) == "False"
        async with _probe(probe_server, accept) as eliciting:
            assert _text(await eliciting.call_tool("client_can_elicit", {})) == "True"
            assert _text(await eliciting.call_tool("confirm", {})) == "accepted"

    async def test_a_crashing_callback_is_a_refusal_and_keeps_the_session(self, probe_server):
        async def crash(context: Any, params: Any) -> types.ElicitResult:
            raise RuntimeError("boom")

        async with _probe(probe_server, crash) as client:
            assert _text(await client.call_tool("confirm", {})) == "declined"
            assert _text(await client.call_tool("client_can_elicit", {})) == "True"

    def test_multi_client_installs_a_default_without_overriding(self):
        async def own(context: Any, params: Any) -> Any: ...

        async def default(context: Any, params: Any) -> Any: ...

        mine = MCPClient(url="http://a.example/mcp", elicitation_callback=own)
        bare = MCPClient(url="http://b.example/mcp")
        MCPMultiClient({"a": mine, "b": bare}, elicitation_callback=default)
        assert mine._elicitation_callback is own
        assert bare._elicitation_callback is default


# ---------------------------------------------------------------------------
# approval_elicitation_callback mapping rules
# ---------------------------------------------------------------------------

APPROVE_SCHEMA = {
    "type": "object",
    "properties": {"approve": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["approve"],
}


def _form(schema: Any = APPROVE_SCHEMA, message: str = "Approve add_pet?") -> Any:
    return types.ElicitRequestFormParams(message=message, requestedSchema=schema)


def _call(name: str, arguments: dict[str, Any]) -> InFlightToolCall:
    import contextvars

    return InFlightToolCall(name=name, arguments=arguments, context=contextvars.copy_context())


class TestMapping:
    async def test_approval_fills_the_decision_field_and_reason(self):
        callback = approval_elicitation_callback(
            lambda r: ApprovalDecision(approved=True, reason="looks right")
        )
        result = await callback(None, _form())
        assert result == types.ElicitResult(
            action="accept", content={"approve": True, "reason": "looks right"}
        )

    async def test_bare_confirmation_is_accepted_with_empty_content(self):
        callback = approval_elicitation_callback(lambda r: True)
        result = await callback(None, _form({"type": "object", "properties": {}}))
        assert result == types.ElicitResult(action="accept", content={})

    @pytest.mark.parametrize(
        "schema",
        [
            {"type": "object", "properties": {"env": {"type": "string"}}, "required": ["env"]},
            {
                "type": "object",
                "properties": {"approve": {"type": "boolean"}, "env": {"type": "string"}},
                "required": ["approve", "env"],
            },
            {"type": "object", "properties": {"delete_everything": {"type": "boolean"}}},
            {
                "type": "object",
                "properties": {"approve": {"type": "boolean"}, "confirm": {"type": "boolean"}},
            },
        ],
        ids=["free-text", "extra-required-field", "unknown-boolean", "two-decision-fields"],
    )
    async def test_input_that_is_not_a_confirmation_is_declined_unasked(self, schema):
        recorder = Recorder(True)
        callback = approval_elicitation_callback(recorder)
        result = await callback(None, _form(schema))
        assert result == types.ElicitResult(action="decline")
        assert recorder.requests == []

    async def test_url_mode_is_declined_unasked(self):
        recorder = Recorder(True)
        callback = approval_elicitation_callback(recorder)
        params = types.ElicitRequestURLParams(
            message="Sign in", url="https://idp.example/authorize", elicitationId="e1"
        )
        assert await callback(None, params) == types.ElicitResult(action="decline")
        assert recorder.requests == []

    async def test_several_calls_in_flight_are_not_guessed(self):
        recorder = Recorder(False)
        calls = [_call("add_pet", {"name": "Rex"}), _call("add_pet", {"name": "Max"})]
        callback = approval_elicitation_callback(recorder, in_flight=lambda: calls)
        await callback(None, _form())
        [request] = recorder.requests
        assert request.tool_name == ""
        assert request.arguments == {}
        assert request.metadata["in_flight_tools"] == ["add_pet", "add_pet"]
        assert request.context_summary == "The MCP server asks: Approve add_pet?"

    async def test_policy_can_hide_arguments(self):
        recorder = Recorder(True)
        policy = ApprovalPolicy(tools=["*"], handler=recorder, include_arguments=False)
        calls = [_call("add_pet", {"name": "Rex"})]
        callback = approval_elicitation_callback(policy, in_flight=lambda: calls)
        await callback(None, _form())
        assert recorder.requests[0].tool_name == "add_pet"
        assert recorder.requests[0].arguments == {}

    async def test_handler_runs_in_the_callers_context(self):
        recorder = Recorder(True)
        token = _caller_ctx_var.set(CallerContext(user_id="bob"))
        try:
            calls = [_call("add_pet", {"name": "Rex"})]
        finally:
            _caller_ctx_var.reset(token)
        callback = approval_elicitation_callback(recorder, in_flight=lambda: calls)
        assert get_current_caller() is None
        await callback(None, _form())
        assert recorder.requests[0].caller_user_id == "bob"
        assert recorder.callers[0] is not None and recorder.callers[0].user_id == "bob"

    async def test_auto_classifier_rules_never_clear_a_server_gate(self):
        human = Recorder(False)
        classifier = AutoApprovalClassifier(
            allow_rules=[ApprovalRule(tool="*")],
            fallback=CallbackApprovalHandler(human),
        )
        calls = [_call("add_pet", {"name": "Rex"})]
        callback = approval_elicitation_callback(classifier, in_flight=lambda: calls)
        assert await callback(None, _form()) == types.ElicitResult(action="decline")
        assert len(human.requests) == 1  # the human was asked, the allow-all rule was not

    async def test_policy_on_decision_records_every_outcome(self):
        seen: list[tuple[str, bool, str]] = []

        def audit(request: ApprovalRequest, decision: ApprovalDecision) -> None:
            seen.append((request.metadata["source"], decision.approved, decision.decided_by))

        calls = [_call("add_pet", {"name": "Rex"})]
        outcomes = [
            True,
            False,
            ApprovalDecision(approved=True, modified_arguments={"name": "Max"}),
        ]
        for outcome in outcomes:
            policy = ApprovalPolicy(tools=["*"], handler=Recorder(outcome), on_decision=audit)
            await approval_elicitation_callback(policy, in_flight=lambda: calls)(None, _form())
        crashing = ApprovalPolicy(tools=["*"], handler=Recorder(error=True), on_decision=audit)
        await approval_elicitation_callback(crashing, in_flight=lambda: calls)(None, _form())

        assert seen == [
            ("mcp_elicitation", True, "reviewer"),
            ("mcp_elicitation", False, "reviewer"),
            # A modified approval is declined by the bridge, and recorded so
            ("mcp_elicitation", False, "gate"),
            ("mcp_elicitation", False, "gate"),
        ]

    async def test_events_carry_the_source(self):
        from promptise.events import CallbackSink, EventNotifier

        seen: list[Any] = []
        notifier = EventNotifier(sinks=[CallbackSink(seen.append)])
        await notifier.start()
        callback = approval_elicitation_callback(
            lambda r: True, server_name="petstore", event_notifier=notifier
        )
        await callback(None, _form())
        await notifier.stop()  # drains the queue
        types_seen = [e.event_type for e in seen]
        assert types_seen == ["approval.requested", "approval.granted"]
        assert all(e.data["source"] == "mcp_elicitation" for e in seen)
        assert all(e.data["server"] == "petstore" for e in seen)

    def test_rejects_bad_configuration(self):
        with pytest.raises(TypeError):
            approval_elicitation_callback(42)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            approval_elicitation_callback(lambda r: True, timeout=0)
