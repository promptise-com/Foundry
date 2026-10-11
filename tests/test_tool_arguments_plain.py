"""Tool-call arguments reach MCP servers, reviewers, traces and events as the model gave them.

Regressions (v1.3.0):

1. LangChain's tool input parsing filled in every optional parameter the model
   left out (``None`` unless the schema had a default), and the MCP tool sent
   them all: a server that validates its input (every MCPcast server) rejected
   the ``null``.  An explicit ``null`` for an optional, non-nullable parameter
   failed validation before the call was sent, with nothing in ``trace_tools``.
2. Nested arguments were generated model instances (``Args_create_order_items_Item_2(...)``)
   in ``trace_tools``, ``ApprovalRequest.arguments`` / ``raw_arguments`` (so
   ``ApprovalRule.argument_contains`` matched a repr), events and observability.
3. With ``optimize_tools``, minified tool schemas lost the constraints shown to
   the model and the "takes any keys" setting of free-form tools, whose calls
   were then sent as ``{}``.

Every server binds ``127.0.0.1`` on an OS-assigned port in-process; the agents
run on a scripted chat model (no LLM API).
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime
import importlib.util
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
import pytest
import uvicorn
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

from promptise import build_agent
from promptise.approval import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from promptise.approval_classifier import ApprovalRule, AutoApprovalClassifier
from promptise.config import HTTPServerSpec
from promptise.events import AgentEvent, CallbackSink, EventNotifier
from promptise.mcp.client import MCPMultiClient
from promptise.mcp.server import MCPServer
from promptise.mcpcast import load_generated_server, mcpcast, write_project
from promptise.mcpcast.schema import SafetyProfile
from promptise.observability import ObservabilityCollector
from promptise.tools import _jsonschema_to_pydantic

TIMEOUT = 15.0


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


@asynccontextmanager
async def _serve(server: MCPServer, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[str]:
    """Run *server* over Streamable HTTP and yield its ``/mcp`` URL."""
    instances: list[uvicorn.Server] = []

    class _Recording(uvicorn.Server):
        def __init__(self, config: uvicorn.Config) -> None:
            super().__init__(config)
            instances.append(self)

    monkeypatch.setattr(uvicorn, "Server", _Recording)
    task = asyncio.ensure_future(server.run_async(transport="http", host="127.0.0.1", port=0))
    try:
        for _ in range(400):
            if task.done():
                task.result()
            if instances and instances[0].started:
                break
            await asyncio.sleep(0.025)
        else:
            raise RuntimeError("server did not start")
        instances[0].config.timeout_graceful_shutdown = 5
        port = instances[0].servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        if instances:
            instances[0].should_exit = True
        try:
            await asyncio.wait_for(task, 15)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            task.cancel()


class _ScriptedModel(BaseChatModel):
    """Emits the scripted tool calls, one turn each, then answers ``done``.

    Records the messages it was given, so tests can read the tool results.
    """

    _script: list[list[dict[str, Any]]] = PrivateAttr()
    _seen: list[list[BaseMessage]] = PrivateAttr()

    def __init__(self, script: list[list[dict[str, Any]]]) -> None:
        super().__init__()
        self._script = list(script)
        self._seen = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._seen.append(list(messages))
        if self._script:
            message = AIMessage(content="", tool_calls=self._script.pop(0))
        else:
            message = AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=message)])

    def tool_results(self) -> list[ToolMessage]:
        last = self._seen[-1] if self._seen else []
        return [m for m in last if isinstance(m, ToolMessage)]


def _calls(*calls: tuple[str, dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """One turn per call."""
    return [[{"name": name, "args": args, "id": f"c{i}"}] for i, (name, args) in enumerate(calls)]


@pytest.fixture
def sent(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Every ``(tool, arguments)`` the agent's MCP client sends."""
    record: list[tuple[str, dict[str, Any]]] = []
    original = MCPMultiClient.call_tool

    async def call_tool(self: MCPMultiClient, name: str, arguments: Any, **kwargs: Any) -> Any:
        record.append((name, arguments))
        return await original(self, name, arguments, **kwargs)

    monkeypatch.setattr(MCPMultiClient, "call_tool", call_tool)
    return record


async def _run(url: str, model: _ScriptedModel, **kwargs: Any) -> Any:
    agent = await build_agent(servers={"shop": HTTPServerSpec(url=url)}, model=model, **kwargs)
    try:
        return await asyncio.wait_for(
            agent.ainvoke({"messages": [HumanMessage(content="go")]}), TIMEOUT
        )
    finally:
        await agent.shutdown()


def _is_plain(value: Any) -> bool:
    """JSON data only: no model instances or other objects anywhere."""
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_plain(v) for k, v in value.items())
    if isinstance(value, list):
        return all(_is_plain(v) for v in value)
    return value is None or isinstance(value, (str, int, float, bool))


# ---------------------------------------------------------------------------
# A Promptise MCP server with optional int / str / enum / nested parameters
# ---------------------------------------------------------------------------


class Item(BaseModel):
    sku: str
    qty: int = 1
    note: str | None = None


class Address(BaseModel):
    city: str
    country: str = "CH"


def _shop() -> tuple[MCPServer, list[tuple[str, dict[str, Any]]]]:
    server = MCPServer("shop")
    received: list[tuple[str, dict[str, Any]]] = []

    @server.tool()
    async def search_orders(
        query: str,
        limit: int = 10,
        status: Literal["open", "closed"] = "open",
        label: str = "all",
        region: str | None = None,
    ) -> dict:
        """Search orders."""
        args = {"query": query, "limit": limit, "status": status, "label": label, "region": region}
        received.append(("search_orders", args))
        return args

    @server.tool()
    async def create_order(
        customer: str, items: list[Item], ship_to: Address | None = None
    ) -> dict:
        """Create an order."""
        args = {
            "customer": customer,
            "items": [i.model_dump() for i in items],
            "ship_to": ship_to.model_dump() if ship_to else None,
        }
        received.append(("create_order", args))
        return {"order_id": "O-1"}

    return server, received


class TestOptionalArgumentsAreNotSent:
    async def test_omitted_optional_parameters_are_left_out(self, monkeypatch, sent):
        server, received = _shop()
        model = _ScriptedModel(_calls(("search_orders", {"query": "lamps"})))
        async with _serve(server, monkeypatch) as url:
            await _run(url, model)
        assert sent == [("search_orders", {"query": "lamps"})]
        # The server applied its own defaults.
        assert received == [
            (
                "search_orders",
                {"query": "lamps", "limit": 10, "status": "open", "label": "all", "region": None},
            )
        ]
        [result] = model.tool_results()
        assert result.status == "success"

    async def test_null_for_a_non_nullable_optional_is_not_given(self, monkeypatch, sent, caplog):
        server, received = _shop()
        model = _ScriptedModel(
            _calls(("search_orders", {"query": "lamps", "limit": None, "status": None}))
        )
        with caplog.at_level("DEBUG", logger="promptise.tools"):
            async with _serve(server, monkeypatch) as url:
                await _run(url, model)
        assert sent == [("search_orders", {"query": "lamps"})]
        assert received[0][1]["limit"] == 10
        [result] = model.tool_results()
        assert result.status == "success" and "Invalid" not in str(result.content)
        assert "'limit' is null but its schema does not allow null" in caplog.text

    async def test_null_for_a_nullable_parameter_is_sent(self, monkeypatch, sent):
        server, _ = _shop()
        model = _ScriptedModel(_calls(("search_orders", {"query": "lamps", "region": None})))
        async with _serve(server, monkeypatch) as url:
            await _run(url, model)
        assert sent == [("search_orders", {"query": "lamps", "region": None})]

    async def test_nested_objects_are_sent_as_given(self, monkeypatch, sent):
        server, received = _shop()
        args = {
            "customer": "dana",
            "items": [{"sku": "A-1"}, {"sku": "B-2", "qty": None, "note": None}],
            "ship_to": {"city": "Bern"},
        }
        model = _ScriptedModel(_calls(("create_order", args)))
        async with _serve(server, monkeypatch) as url:
            await _run(url, model)
        [(name, arguments)] = sent
        assert name == "create_order"
        assert arguments == {
            "customer": "dana",
            "items": [{"sku": "A-1"}, {"sku": "B-2", "note": None}],
            "ship_to": {"city": "Bern"},
        }
        assert _is_plain(arguments)
        assert received[0][1]["items"][1] == {"sku": "B-2", "qty": 1, "note": None}


class TestInvalidArgumentsReachTheModel:
    async def test_the_model_sees_the_error_and_corrects_the_call(self, monkeypatch, sent, capsys):
        server, _ = _shop()
        model = _ScriptedModel(
            _calls(
                ("search_orders", {"query": "lamps", "limit": "lots"}),
                ("search_orders", {"query": "lamps", "limit": 5}),
            )
        )
        async with _serve(server, monkeypatch) as url:
            await _run(url, model, trace_tools=True)
        # The invalid call was never sent; the corrected one was.
        assert sent == [("search_orders", {"query": "lamps", "limit": 5})]
        first, second = model.tool_results()
        assert first.status == "error"
        assert "Invalid arguments for tool 'search_orders'" in str(first.content)
        assert "- limit:" in str(first.content) and '"lots"' in str(first.content)
        assert "call the tool again" in str(first.content)
        assert '"limit": 5' in str(second.content)
        out = capsys.readouterr().out
        assert "→ Invoking tool: search_orders with {'query': 'lamps', 'limit': 'lots'}" in out
        assert "✖ search_orders error: Invalid arguments for tool 'search_orders'" in out

    async def test_a_null_for_a_required_parameter_is_reported(self, monkeypatch, sent):
        server, _ = _shop()
        model = _ScriptedModel(_calls(("search_orders", {"query": None})))
        async with _serve(server, monkeypatch) as url:
            await _run(url, model)
        assert sent == []
        [result] = model.tool_results()
        assert "- query:" in str(result.content)

    async def test_tool_error_event(self, monkeypatch, sent):
        server, _ = _shop()
        events: list[AgentEvent] = []
        notifier = EventNotifier(sinks=[CallbackSink(events.append, events=["tool.error"])])
        model = _ScriptedModel(_calls(("search_orders", {"query": "x", "limit": "lots"})))
        async with _serve(server, monkeypatch) as url:
            await _run(url, model, events=notifier)
            for _ in range(100):
                if events:
                    break
                await asyncio.sleep(0.01)
        [event] = events
        assert event.data["tool_name"] == "search_orders"
        assert event.data["code"] == "INVALID_ARGUMENTS"
        assert event.data["error_type"] == "ToolArgumentError"


class TestDownstreamSeesPlainData:
    ARGS: dict[str, Any] = {
        "customer": "dana",
        "items": [{"sku": "BANNED-1", "qty": 2}],
        "ship_to": {"city": "Bern"},
    }

    async def test_trace_and_observability_show_data_not_reprs(self, monkeypatch, sent, capsys):
        server, _ = _shop()
        collector = ObservabilityCollector("t")
        model = _ScriptedModel(_calls(("create_order", self.ARGS)))
        async with _serve(server, monkeypatch) as url:
            await _run(url, model, trace_tools=True, observer=collector)
        out = capsys.readouterr().out
        assert "Args_" not in out
        assert "'items': [{'sku': 'BANNED-1', 'qty': 2}]" in out
        [call] = collector.query(event_types=["tool.call"])
        assert "Args_" not in call.metadata["arguments"]
        assert "{'sku': 'BANNED-1', 'qty': 2}" in call.metadata["arguments"]

    async def test_approval_request_and_events_carry_plain_arguments(self, monkeypatch, sent):
        server, _ = _shop()
        requests: list[ApprovalRequest] = []

        async def handler(request: ApprovalRequest) -> ApprovalDecision:
            requests.append(request)
            return ApprovalDecision(approved=True)

        events: list[AgentEvent] = []
        notifier = EventNotifier(sinks=[CallbackSink(events.append, events=["approval.requested"])])
        model = _ScriptedModel(_calls(("create_order", self.ARGS)))
        async with _serve(server, monkeypatch) as url:
            await _run(
                url,
                model,
                approval=ApprovalPolicy(tools=["create_order"], handler=handler),
                events=notifier,
            )
            for _ in range(100):
                if events:
                    break
                await asyncio.sleep(0.01)
        [request] = requests
        assert request.arguments == self.ARGS
        assert request.raw_arguments == self.ARGS
        assert _is_plain(request.arguments) and _is_plain(request.raw_arguments)
        json.dumps(request.to_dict())  # no default=str needed
        [event] = events
        assert event.data["arguments"] == self.ARGS
        # What the server got after approval: the same plain arguments.
        assert sent == [("create_order", self.ARGS)]

    async def test_argument_contains_matches_nested_values(self, monkeypatch, sent):
        server, received = _shop()
        fallback_requests: list[ApprovalRequest] = []

        async def fallback(request: ApprovalRequest) -> ApprovalDecision:
            fallback_requests.append(request)
            return ApprovalDecision(approved=True)

        classifier = AutoApprovalClassifier(
            deny_rules=[
                ApprovalRule(
                    tool="create_order",
                    argument_contains='"sku": "BANNED-1"',
                    reason="banned sku",
                )
            ],
            fallback=fallback,
        )
        model = _ScriptedModel(_calls(("create_order", self.ARGS)))
        async with _serve(server, monkeypatch) as url:
            await _run(url, model, approval=ApprovalPolicy(tools=["*"], handler=classifier))
        assert fallback_requests == []
        assert sent == [] and received == []
        [result] = model.tool_results()
        assert "DENIED: banned sku" in str(result.content)

    async def test_omitted_and_null_arguments_through_the_approval_gate(self, monkeypatch, sent):
        server, _ = _shop()
        requests: list[ApprovalRequest] = []

        async def handler(request: ApprovalRequest) -> bool:
            requests.append(request)
            return True

        model = _ScriptedModel(
            _calls(("search_orders", {"query": "lamps", "limit": None, "region": None}))
        )
        async with _serve(server, monkeypatch) as url:
            await _run(url, model, approval=ApprovalPolicy(tools=["*"], handler=handler))
        assert requests[0].arguments == {"query": "lamps", "region": None}
        assert sent == [("search_orders", {"query": "lamps", "region": None})]

    async def test_invalid_arguments_behind_the_gate_are_not_sent_for_approval(
        self, monkeypatch, sent, capsys
    ):
        server, _ = _shop()
        requests: list[ApprovalRequest] = []

        async def handler(request: ApprovalRequest) -> bool:
            requests.append(request)
            return True

        model = _ScriptedModel(_calls(("search_orders", {"query": "x", "limit": "lots"})))
        async with _serve(server, monkeypatch) as url:
            await _run(
                url,
                model,
                trace_tools=True,
                approval=ApprovalPolicy(tools=["*"], handler=handler),
            )
        assert requests == [] and sent == []
        [result] = model.tool_results()
        assert "Invalid arguments for tool 'search_orders'" in str(result.content)
        assert "✖ search_orders error: Invalid arguments" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Non-MCP tools behind the gate keep their typed arguments
# ---------------------------------------------------------------------------


class Line(BaseModel):
    sku: str
    qty: int = 1


got: list[Any] = []


@tool
async def reserve(lines: list[Line], memo: str = "none") -> str:
    """Reserve stock."""
    got.append((lines, memo))
    return "reserved"


class TestPythonToolsBehindWrappers:
    async def test_approved_python_tool_gets_model_instances_and_its_defaults(self):
        got.clear()
        requests: list[ApprovalRequest] = []

        async def handler(request: ApprovalRequest) -> bool:
            requests.append(request)
            return True

        model = _ScriptedModel(_calls(("reserve", {"lines": [{"sku": "A"}], "memo": None})))
        agent = await build_agent(
            servers={},
            model=model,
            extra_tools=[reserve],
            trace_tools=True,
            approval=ApprovalPolicy(tools=["*"], handler=handler),
        )
        try:
            await agent.ainvoke({"messages": [HumanMessage(content="go")]})
        finally:
            await agent.shutdown()
        [(lines, memo)] = got
        assert isinstance(lines[0], Line) and lines[0].sku == "A"
        assert memo == "none"
        assert requests[0].arguments == {"lines": [{"sku": "A"}]}

    async def test_traced_python_tool_reports_invalid_arguments(self, capsys):
        model = _ScriptedModel(_calls(("reserve", {"lines": "nope"})))
        agent = await build_agent(servers={}, model=model, extra_tools=[reserve], trace_tools=True)
        try:
            await agent.ainvoke({"messages": [HumanMessage(content="go")]})
        finally:
            await agent.shutdown()
        out = capsys.readouterr().out
        assert "→ Invoking tool: reserve with {'lines': 'nope'}" in out
        assert "✖ reserve error: Invalid arguments for tool 'reserve'" in out


# ---------------------------------------------------------------------------
# An MCPcast-generated server
# ---------------------------------------------------------------------------

PETSTORE: dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "Pets", "version": "1"},
    "servers": [{"url": "https://pets.example"}],
    "paths": {
        "/pets": {
            "get": {
                "operationId": "listPets",
                "summary": "List pets",
                "parameters": [
                    {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                    {
                        "name": "status",
                        "in": "query",
                        "schema": {"type": "string", "enum": ["available", "sold"]},
                    },
                ],
                "responses": {"200": {"description": "ok"}},
            },
            "post": {
                "operationId": "addPet",
                "summary": "Add a pet",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "required": ["name"],
                                "properties": {
                                    "name": {"type": "string"},
                                    "tag": {"type": "string"},
                                    "age": {"type": "integer"},
                                    "category": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "integer"},
                                            "name": {"type": "string"},
                                        },
                                    },
                                },
                            }
                        }
                    },
                },
                "responses": {"200": {"description": "ok"}},
            },
        }
    },
}


class _Upstream:
    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"ok": True}, request=request)


def _petstore(tmp_path: Path, http: httpx.AsyncClient) -> MCPServer:
    plan = mcpcast(PETSTORE, name="pets", profile=SafetyProfile.STANDARD, auth="none")  # type: ignore[arg-type]
    write_project(plan, tmp_path / "proj")
    module = load_generated_server(tmp_path / "proj" / "server.py")
    return module.build_server(approval_handler=lambda request: True, http_client=http)


class TestMCPcastServer:
    async def test_optional_parameters_omitted_or_null_are_not_sent(
        self, tmp_path, monkeypatch, sent
    ):
        upstream = _Upstream()
        requests: list[ApprovalRequest] = []

        async def handler(request: ApprovalRequest) -> bool:
            requests.append(request)
            return True

        model = _ScriptedModel(
            _calls(
                ("list_pets", {}),
                ("list_pets", {"limit": None, "status": None}),
                ("add_pet", {"name": "Rex", "age": None, "category": {"name": "dogs", "id": None}}),
            )
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler)) as http:
            server = _petstore(tmp_path, http)
            async with _serve(server, monkeypatch) as url:
                await _run(url, model, approval=ApprovalPolicy(tools=["add_*"], handler=handler))
        results = model.tool_results()
        assert [r.status for r in results] == ["success"] * 3, [r.content for r in results]
        get1, get2, post = upstream.requests
        assert dict(get1.url.params) == {} and dict(get2.url.params) == {}
        assert json.loads(post.content) == {"name": "Rex", "category": {"name": "dogs"}}
        # add_pet reached the agent's approval gate with only the given fields.
        assert requests[0].arguments == {"name": "Rex", "category": {"name": "dogs"}}
        assert sent[-1] == ("add_pet", {"name": "Rex", "category": {"name": "dogs"}})

    async def test_readiness_tools_send_what_the_agent_gave(self, tmp_path):
        """MCPcast's readiness run calls the server in-process the same way."""
        from promptise.mcpcast.readiness import tools_from_server

        upstream = _Upstream()
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream.handler)) as http:
            tools = {t.name: t for t in await tools_from_server(_petstore(tmp_path, http))}
            out = await tools["list_pets"].ainvoke({"limit": None})
            await tools["add_pet"].ainvoke({"name": "Rex", "category": {"name": "dogs"}})
        assert "VALIDATION_ERROR" not in out
        get, post = upstream.requests
        assert dict(get.url.params) == {}
        assert json.loads(post.content) == {"name": "Rex", "category": {"name": "dogs"}}


class TestFreeFormTool:
    async def test_free_form_arguments_are_sent(self):
        from mcp.types import CallToolResult, TextContent

        from promptise.mcp.client._tool_adapter import _PromptiseMCPTool

        calls: list[dict[str, Any]] = []

        class _Multi:
            async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
                calls.append(arguments)
                return CallToolResult(content=[TextContent(type="text", text="ok")])

        free_form = _jsonschema_to_pydantic({"type": "object", "additionalProperties": True})
        tool_ = _PromptiseMCPTool(
            name="anything",
            description="Takes any keys.",
            args_schema=free_form,
            tool_name="anything",
            multi=_Multi(),  # type: ignore[arg-type]
        )
        assert await tool_.ainvoke({"a": 1, "b": {"c": None}}) == "ok"
        assert calls == [{"a": 1, "b": {"c": None}}]


# ---------------------------------------------------------------------------
# The helpers
# ---------------------------------------------------------------------------


class TestParseToolArguments:
    SCHEMA: dict[str, Any] = {
        "type": "object",
        "required": ["q"],
        "properties": {
            "q": {"type": "string"},
            "n": {"type": "integer", "default": 3},
            "since": {"type": ["string", "null"]},
            "until": {"type": "string", "nullable": True},
            "mode": {"type": "string", "enum": ["a", "b"]},
            "filter": {
                "type": "object",
                "properties": {"tag": {"type": "string"}, "min": {"type": "integer"}},
            },
            "rows": {
                "type": "array",
                "items": {"type": "object", "properties": {"k": {"type": "integer"}}},
            },
        },
    }

    def _parse(self, args: dict[str, Any]) -> dict[str, Any]:
        from promptise.tools import parse_tool_arguments

        return parse_tool_arguments("t", _jsonschema_to_pydantic(self.SCHEMA), args)

    def test_only_given_fields(self):
        assert self._parse({"q": "x"}) == {"q": "x"}

    def test_nullable_forms_keep_null(self):
        assert self._parse({"q": "x", "since": None, "until": None}) == {
            "q": "x",
            "since": None,
            "until": None,
        }

    def test_non_nullable_nulls_are_dropped_at_every_level(self):
        assert self._parse(
            {
                "q": "x",
                "n": None,
                "mode": None,
                "filter": {"tag": None, "min": 2},
                "rows": [{"k": None}, {"k": 1}],
            }
        ) == {"q": "x", "filter": {"min": 2}, "rows": [{}, {"k": 1}]}

    def test_errors_name_the_path(self):
        from promptise.tools import ToolArgumentError

        with pytest.raises(ToolArgumentError) as info:
            self._parse({"q": "x", "rows": [{"k": "many"}]})
        assert "- rows.0.k: Input should be a valid integer" in info.value.message
        assert info.value.code == "INVALID_ARGUMENTS"

    def test_free_form_objects_pass_their_keys(self):
        from promptise.tools import parse_tool_arguments

        model = _jsonschema_to_pydantic({"type": "object", "additionalProperties": True})
        assert parse_tool_arguments("t", model, {"a": 1, "b": {"c": None}}) == {
            "a": 1,
            "b": {"c": None},
        }

    def test_plain_arguments_dumps_models(self):
        from promptise.tools import plain_arguments

        model = _jsonschema_to_pydantic(self.SCHEMA)
        instance = model.model_validate({"q": "x", "filter": {"tag": "t"}})
        assert plain_arguments({"filter": instance.filter, "n": (1, 2)}) == {  # type: ignore[attr-defined]
            "filter": {"tag": "t"},
            "n": [1, 2],
        }


# ---------------------------------------------------------------------------
# optimize_tools: minified schemas keep everything but descriptions
# ---------------------------------------------------------------------------
#
# Regression (v1.3.0): with ``optimize_tools`` on, each tool's args model was
# rebuilt without the constraints shown to the model (enum, pattern, bounds,
# formats...) and without the "takes any keys" setting of a free-form
# schema, whose calls were then sent as ``{}``.


class Leg(BaseModel):
    code: Annotated[str, Field(pattern=r"^[A-Z]{3}$", description="IATA airport code.")]
    seats: Annotated[int, Field(ge=1, le=9, description="Seats to book.")] = 1
    note: str | None = None


def _travel() -> tuple[MCPServer, list[tuple[str, dict[str, Any]]]]:
    server = MCPServer("travel")
    received: list[tuple[str, dict[str, Any]]] = []

    @server.tool()
    async def book_trip(
        traveler: Annotated[str, Field(min_length=2, max_length=40, description="Full name.")],
        legs: Annotated[list[Leg], Field(min_length=1, max_length=4)],
        cabin: Literal["economy", "business"] = "economy",
        budget: Annotated[float, Field(gt=0, le=10000)] = 500.0,
        depart: datetime.date | None = None,
        home: Address | None = None,
    ) -> dict:
        """Book a trip."""
        args = {
            "traveler": traveler,
            "legs": [leg.model_dump() for leg in legs],
            "cabin": cabin,
            "budget": budget,
            "depart": depart.isoformat() if depart else None,
            "home": home.model_dump() if home else None,
        }
        received.append(("book_trip", args))
        return {"booking": "B-1"}

    @server.tool()
    async def configure(**settings: Any) -> dict:
        """Set any configuration keys."""
        received.append(("configure", settings))
        return {"ok": True}

    # A free-form tool: an object schema without properties that takes any keys.
    tool_def = server._tool_registry.get("configure")
    assert tool_def is not None
    server._tool_registry.replace(
        dataclasses.replace(tool_def, input_schema={"type": "object", "additionalProperties": True})
    )
    server._input_models.pop("configure", None)
    return server, received


class _OfferedModel(_ScriptedModel):
    """A scripted model that records the parameter schema of every tool it is offered."""

    _offered: dict[str, dict[str, Any]] = PrivateAttr()

    def __init__(self, script: list[list[dict[str, Any]]]) -> None:
        super().__init__(script)
        self._offered = {}

    def bind_tools(self, tools: Any, **kwargs: Any) -> _OfferedModel:
        for t in tools:
            function = convert_to_openai_tool(t)["function"]
            self._offered[function["name"]] = function.get("parameters", {})
        return self


def _without_descriptions(schema: Any) -> Any:
    """*schema* without descriptions and titles (generated model names differ)."""
    if isinstance(schema, dict):
        return {
            k: _without_descriptions(v)
            for k, v in schema.items()
            if not (k in ("description", "title") and isinstance(v, str))
        }
    if isinstance(schema, list):
        return [_without_descriptions(v) for v in schema]
    return schema


TRIP_ARGS: dict[str, Any] = {
    "traveler": "Dana Scully",
    "legs": [{"code": "ZRH"}, {"code": "LIS", "seats": None, "note": None}],
    "cabin": None,
    "depart": None,
    "home": {"city": "Bern"},
}
# Omitted parameters and nulls for non-nullable ones are left out; nulls for
# nullable ones are kept; nested objects are plain JSON.
TRIP_SENT: dict[str, Any] = {
    "traveler": "Dana Scully",
    "legs": [{"code": "ZRH"}, {"code": "LIS", "note": None}],
    "depart": None,
    "home": {"city": "Bern"},
}
SETTINGS: dict[str, Any] = {"region": "eu", "limits": {"rps": 5, "burst": None}}

_HAS_SEMANTIC = importlib.util.find_spec("sentence_transformers") is not None


class TestOptimizedToolSchemas:
    async def _offered_and_sent(
        self, url: str, sent: list[tuple[str, dict[str, Any]]], **kwargs: Any
    ) -> tuple[_OfferedModel, list[tuple[str, dict[str, Any]]], list[ApprovalRequest]]:
        requests: list[ApprovalRequest] = []

        async def handler(request: ApprovalRequest) -> bool:
            requests.append(request)
            return True

        model = _OfferedModel(_calls(("book_trip", TRIP_ARGS), ("configure", SETTINGS)))
        sent.clear()
        # configure goes through the approval gate (a WrappingTool), book_trip does not.
        await _run(
            url, model, approval=ApprovalPolicy(tools=["configure"], handler=handler), **kwargs
        )
        return model, list(sent), requests

    @pytest.mark.parametrize(
        "level",
        [
            "minimal",
            "standard",
            pytest.param(
                "semantic",
                marks=pytest.mark.skipif(
                    not _HAS_SEMANTIC, reason="sentence-transformers not installed"
                ),
            ),
        ],
    )
    async def test_minified_tools_keep_their_schema_and_send_what_the_model_gave(
        self, level, monkeypatch, sent
    ):
        server, received = _travel()
        async with _serve(server, monkeypatch) as url:
            plain_model, plain_sent, _ = await self._offered_and_sent(url, sent)
            received.clear()
            model, optimized_sent, requests = await self._offered_and_sent(
                url, sent, optimize_tools=level
            )

        # What goes over MCP: the same as without optimization.
        assert optimized_sent == plain_sent == [("book_trip", TRIP_SENT), ("configure", SETTINGS)]
        assert all(_is_plain(arguments) for _, arguments in optimized_sent)
        assert received == [
            (
                "book_trip",
                {
                    "traveler": "Dana Scully",
                    "legs": [
                        {"code": "ZRH", "seats": 1, "note": None},
                        {"code": "LIS", "seats": 1, "note": None},
                    ],
                    "cabin": "economy",
                    "budget": 500.0,
                    "depart": None,
                    "home": {"city": "Bern", "country": "CH"},
                },
            ),
            ("configure", SETTINGS),
        ]
        assert [request.arguments for request in requests] == [SETTINGS]

        # What the model is offered: the full schema, without descriptions.
        trip = model._offered["book_trip"]
        assert "description" not in trip["properties"]["traveler"]
        assert "description" in plain_model._offered["book_trip"]["properties"]["traveler"]
        assert _without_descriptions(trip) == _without_descriptions(
            plain_model._offered["book_trip"]
        )
        props = trip["properties"]
        assert sorted(trip["required"]) == ["legs", "traveler"]
        assert (props["traveler"]["minLength"], props["traveler"]["maxLength"]) == (2, 40)
        assert (props["legs"]["minItems"], props["legs"]["maxItems"]) == (1, 4)
        leg = props["legs"]["items"]
        assert leg["properties"]["code"]["pattern"] == "^[A-Z]{3}$"
        assert (leg["properties"]["seats"]["minimum"], leg["properties"]["seats"]["maximum"]) == (
            1,
            9,
        )
        assert leg["required"] == ["code"]
        assert props["cabin"]["enum"] == ["economy", "business"]
        assert (props["budget"]["exclusiveMinimum"], props["budget"]["maximum"]) == (0, 10000)
        assert {"type": "string", "format": "date"} in props["depart"]["anyOf"]
        assert {"type": "null"} in props["depart"]["anyOf"]
        assert {"type": "null"} in props["home"]["anyOf"]
        assert model._offered["configure"] == plain_model._offered["configure"]
        assert model._offered["configure"]["additionalProperties"] is True

        results = model.tool_results()
        assert [r.status for r in results] == ["success", "success"], [r.content for r in results]


class TestMinifiedModel:
    """``_minify_pydantic_model`` drops descriptions and nothing the argument helpers rely on."""

    SCHEMA: dict[str, Any] = {
        "type": "object",
        "required": ["q"],
        "properties": {
            "q": {"type": "string", "minLength": 2, "pattern": "^[a-z]+$", "description": "Q."},
            "n": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            "ratio": {"type": "number", "enum": [0.5, 1.5]},
            "when": {"type": ["string", "null"], "format": "date"},
            "rows": {
                "type": ["array", "null"],
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "k": {"type": "integer", "minimum": 0, "description": "K."},
                        "sub": {
                            "type": "object",
                            "properties": {"tag": {"type": "string", "maxLength": 5}},
                        },
                    },
                },
            },
        },
    }

    @staticmethod
    def _minify(schema: dict[str, Any], **kwargs: Any) -> type[BaseModel]:
        from promptise.tool_optimization import _minify_pydantic_model

        return _minify_pydantic_model(_jsonschema_to_pydantic(schema), **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [{}, {"strip_nested": True}, {"strip_nested": True, "max_depth": 3}],
    )
    def test_schema_is_the_original_without_descriptions(self, kwargs):
        original = _jsonschema_to_pydantic(self.SCHEMA)
        minified = self._minify(self.SCHEMA, **kwargs)
        shown = convert_to_openai_tool(minified)["function"]["parameters"]
        assert "description" not in shown["properties"]["q"]
        assert _without_descriptions(shown) == _without_descriptions(
            convert_to_openai_tool(original)["function"]["parameters"]
        )

    def test_argument_helpers_work_on_the_minified_model(self):
        from promptise.tools import parse_tool_arguments

        minified = self._minify(self.SCHEMA, strip_nested=True)
        given = {
            "q": "ab",
            "n": None,
            "ratio": None,
            "when": None,
            "rows": [{"k": None, "sub": {"tag": None}}, {"k": 2}],
        }
        assert parse_tool_arguments("t", minified, given) == {
            "q": "ab",
            "when": None,
            "rows": [{"sub": {}}, {"k": 2}],
        }
        assert parse_tool_arguments("t", minified, {"q": "ab"}) == {"q": "ab"}

    def test_depth_flattening_keeps_nullability_and_constraints(self):
        from promptise.tools import parse_tool_arguments

        minified = self._minify(self.SCHEMA, strip_nested=True, max_depth=1)
        rows = convert_to_openai_tool(minified)["function"]["parameters"]["properties"]["rows"]
        assert {"type": "null"} in rows["anyOf"]
        [array] = [v for v in rows["anyOf"] if v.get("type") == "array"]
        assert array["items"] == {"type": "object", "additionalProperties": True}
        assert array["maxItems"] == 3
        given = {"q": "ab", "rows": [{"k": 1, "sub": {"tag": "x"}}], "when": None}
        assert parse_tool_arguments("t", minified, given) == given

    def test_free_form_model_keeps_taking_any_keys(self):
        from promptise.tools import accepts_free_form_arguments, parse_tool_arguments

        minified = self._minify({"type": "object", "additionalProperties": True})
        assert accepts_free_form_arguments(minified)
        assert parse_tool_arguments("t", minified, SETTINGS) == SETTINGS

    def test_pydantic_constraints_and_field_settings_are_kept(self):
        from promptise.tool_optimization import _minify_pydantic_model

        class Args(BaseModel):
            model_config = ConfigDict(populate_by_name=True)

            count: Annotated[int, Field(gt=0, description="How many.")] = 1
            label: str = Field(
                default="x", alias="Label", examples=["a"], json_schema_extra={"format": "slug"}
            )

        minified = _minify_pydantic_model(Args)
        shown = minified.model_json_schema()["properties"]
        assert shown["count"] == {
            "default": 1,
            "exclusiveMinimum": 0,
            "title": "Count",
            "type": "integer",
        }
        assert shown["Label"]["examples"] == ["a"] and shown["Label"]["format"] == "slug"
        assert minified.model_validate({"label": "y"}).model_dump() == {"count": 1, "label": "y"}
        with pytest.raises(ValidationError):
            minified.model_validate({"count": 0})
