"""Offline unit tests for the code-action reasoning pattern.

No Docker and no API key required: an in-memory fake sandbox session and a
canned fake model exercise the full node orchestration (prompt -> code ->
write -> execute -> parse), the repair loop, the host-side tool bridge, the
pure helpers, and the prebuilt/agent wiring. The real Docker + real API paths
are covered separately in ``tests/test_code_action_integration.py``.
"""

from __future__ import annotations

import asyncio
import base64
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from promptise.engine import CodeActionNode, PromptGraph
from promptise.engine.code_action import (
    _extract_task,
    _render_tools_module,
    extract_code,
    parse_result,
    render_api_spec,
)
from promptise.engine.state import GraphState
from promptise.sandbox.session import CommandResult


# ──────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────
class FakeModel:
    """Returns canned AIMessages (one per ainvoke), tracking call count."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls = 0

    async def ainvoke(self, messages, config=None):
        self.calls += 1
        text = self._responses.pop(0) if self._responses else ""
        return AIMessage(content=text)


class FakeSession:
    """In-memory sandbox: supports mkdir, base64-write, cat, python3, ls."""

    def __init__(self, program_results: list[CommandResult]):
        self.fs: dict[str, str] = {}
        self.program_results = list(program_results)
        self.cleaned = False
        self.execs: list[str] = []

    async def execute(self, command: str, timeout=None, workdir=None) -> CommandResult:
        self.execs.append(command)
        if command.startswith("mkdir"):
            return CommandResult(0, "", "")
        if "| base64 -d >" in command:
            left, path = command.split("| base64 -d >")
            path = path.strip().strip("'")
            b64 = left.replace("printf %s", "").strip().strip("'")
            self.fs[path] = base64.b64decode(b64).decode()
            return CommandResult(0, "", "")
        if command.startswith("cat "):
            path = command[4:].strip().strip("'")
            if path in self.fs:
                return CommandResult(0, self.fs[path], "")
            return CommandResult(1, "", "No such file")
        if command.startswith("python3"):
            return self.program_results.pop(0)
        return CommandResult(0, "", "")

    async def write_file(self, path: str, content: str) -> None:
        self.execs.append(f"write {path}")
        self.fs[path] = content

    async def list_files(self, directory: str) -> list[str]:
        prefix = directory.rstrip("/") + "/"
        return [
            p[len(prefix) :]
            for p in self.fs
            if p.startswith(prefix) and "/" not in p[len(prefix) :]
        ]

    async def cleanup(self) -> None:
        self.cleaned = True


def _factory(session: FakeSession):
    async def make():
        return session

    return make


# ──────────────────────────────────────────────────────────────────────────
# Pure helpers
# ──────────────────────────────────────────────────────────────────────────
def test_extract_code_from_fence():
    assert extract_code("blah\n```python\nprint(1)\n```\nmore") == "print(1)"
    assert extract_code("```\nx = 2\n```") == "x = 2"
    assert extract_code("no fence here") == "no fence here"


def test_parse_result_marker_and_fallback():
    assert parse_result("noise\nRESULT: 635\n", "RESULT:") == "635"
    assert parse_result("a\nRESULT: 1\nRESULT: 2\n", "RESULT:") == "2"  # last wins
    assert parse_result("only line", "RESULT:") == "only line"  # fallback
    assert parse_result("", "RESULT:") == ""


def test_render_api_spec_and_tools_module():
    @tool("get_employee")
    def get_employee(name: str) -> str:
        """Get an employee record by name."""
        return name

    spec = render_api_spec([get_employee])
    assert "def get_employee(name)" in spec
    assert "Get an employee record" in spec

    module = _render_tools_module([get_employee], call_timeout=10)
    assert "def _call(" in module
    assert "def get_employee(name):" in module
    assert "_call('get_employee', {'name': name})" in module
    # No data functions → graceful spec
    assert "no data functions" in render_api_spec([]).lower()


def test_extract_task_objects_and_dicts():
    s1 = GraphState(messages=[HumanMessage(content="What is 2+2?")])
    assert _extract_task(s1) == "What is 2+2?"
    s2 = GraphState(messages=[{"role": "user", "content": "hello"}])
    assert _extract_task(s2) == "hello"


# ──────────────────────────────────────────────────────────────────────────
# Node orchestration
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_happy_path_writes_runs_parses():
    session = FakeSession([CommandResult(0, "RESULT: 42\n", "")])
    model = FakeModel(["```python\nprint('RESULT:', 6 * 7)\n```"])
    node = CodeActionNode("reason", sandbox_factory=_factory(session))
    state = GraphState(messages=[HumanMessage(content="6 times 7?")])

    result = await node.execute(state, {"_engine_model": model})

    assert result.error is None
    assert result.output == "42"
    assert model.calls == 1
    # program + tools module were written into the sandbox
    assert "/workspace/program.py" in session.fs
    assert "from promptise_tools import *" in session.fs["/workspace/program.py"]
    # final answer appended as an AIMessage
    assert any(isinstance(m, AIMessage) and m.content == "42" for m in state.messages)
    assert session.cleaned is True


@pytest.mark.asyncio
async def test_repair_loop_recovers_from_crash():
    session = FakeSession(
        [
            CommandResult(1, "", "NameError: name 'x' is not defined"),  # first attempt crashes
            CommandResult(0, "RESULT: 7\n", ""),  # repaired attempt
        ]
    )
    model = FakeModel(
        [
            "```python\nprint(x)\n```",  # buggy
            "```python\nprint('RESULT:', 7)\n```",  # fixed
        ]
    )
    node = CodeActionNode("reason", sandbox_factory=_factory(session), max_repairs=1)
    state = GraphState(messages=[HumanMessage(content="give 7")])

    result = await node.execute(state, {"_engine_model": model})

    assert result.error is None
    assert result.output == "7"
    assert model.calls == 2  # one repair


@pytest.mark.asyncio
async def test_exhausted_repairs_reports_error():
    session = FakeSession(
        [
            CommandResult(1, "", "boom"),
            CommandResult(1, "", "boom again"),
        ]
    )
    model = FakeModel(["```python\nbad\n```", "```python\nstill bad\n```"])
    node = CodeActionNode("reason", sandbox_factory=_factory(session), max_repairs=1)
    state = GraphState(messages=[HumanMessage(content="x")])

    result = await node.execute(state, {"_engine_model": model})
    assert result.error is not None
    assert "failed" in result.error


@pytest.mark.asyncio
async def test_missing_model_and_missing_sandbox_error_cleanly():
    node_no_sandbox = CodeActionNode("reason", sandbox_factory=None)
    r1 = await node_no_sandbox.execute(
        GraphState(messages=[HumanMessage(content="x")]), {"_engine_model": FakeModel(["x"])}
    )
    assert r1.error and "sandbox" in r1.error.lower()

    node = CodeActionNode("reason", sandbox_factory=_factory(FakeSession([])))
    r2 = await node.execute(GraphState(messages=[HumanMessage(content="x")]), {})
    assert r2.error and "model" in r2.error.lower()


# ──────────────────────────────────────────────────────────────────────────
# Host-side tool bridge
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_bridge_services_a_tool_request():
    @tool("add")
    def add(a: int, b: int) -> str:
        """Add two numbers."""
        return str(a + b)

    session = FakeSession([])
    # Pre-seed a pending request as if the in-sandbox program wrote it.
    session.fs["/workspace/_rpc/req_abc.json"] = json.dumps(
        {"tool": "add", "args": {"a": 1, "b": 2}}
    )

    node = CodeActionNode("reason", tools=[add], sandbox_factory=_factory(session))
    state = GraphState(messages=[])
    from promptise.engine.state import NodeResult

    result = NodeResult(node_name="reason")
    stop = asyncio.Event()
    task = asyncio.create_task(node._bridge_loop(session, {"add": add}, state, {}, result, stop))
    # give the loop a few cycles to service the request, then stop
    for _ in range(50):
        if "/workspace/_rpc/resp_abc.done" in session.fs:
            break
        await asyncio.sleep(0.02)
    stop.set()
    await task

    assert json.loads(session.fs["/workspace/_rpc/resp_abc.json"]) == {"result": "3"}
    assert state.observations and state.observations[0]["result"] == "3"
    assert result.tool_calls and result.tool_calls[0]["name"] == "add"


@pytest.mark.asyncio
async def test_bridge_unknown_tool_returns_error():
    session = FakeSession([])
    node = CodeActionNode("reason", sandbox_factory=_factory(session))
    from promptise.engine.state import NodeResult

    resp = await node._invoke_tool({}, GraphState(messages=[]), {}, "nope", {}, NodeResult())
    assert "error" in resp and "unknown tool" in resp["error"]


# ──────────────────────────────────────────────────────────────────────────
# Prebuilt + pattern wiring
# ──────────────────────────────────────────────────────────────────────────
def test_prebuilt_builds_code_action_graph():
    g = PromptGraph.code_action(tools=[], system_prompt="hi")
    assert g.name == "code-action"
    node = g.get_node("reason")
    assert isinstance(node, CodeActionNode)
    assert node.default_next == "__end__"


# ──────────────────────────────────────────────────────────────────────────
# Security: hard tool-call cap + bridge request-id validation
# ──────────────────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_max_tool_calls_cap_is_enforced():
    @tool("ping")
    def ping() -> str:
        """Return pong."""
        return "pong"

    from promptise.engine.state import NodeResult

    node = CodeActionNode("reason", sandbox_factory=_factory(FakeSession([])), max_tool_calls=2)
    result = NodeResult()
    state = GraphState(messages=[])
    r1 = await node._invoke_tool({"ping": ping}, state, {}, "ping", {}, result)
    r2 = await node._invoke_tool({"ping": ping}, state, {}, "ping", {}, result)
    r3 = await node._invoke_tool({"ping": ping}, state, {}, "ping", {}, result)  # over the cap

    assert r1 == {"result": "pong"} and r2 == {"result": "pong"}
    assert "error" in r3 and "budget" in r3["error"]
    assert len(result.tool_calls) == 2  # the 3rd call never executed the tool


def test_safe_rid_regex_blocks_traversal():
    from promptise.engine.code_action import _SAFE_RID

    assert _SAFE_RID.match("a1b2c3deadbeef")
    assert _SAFE_RID.match("abc-123_DEF")
    assert not _SAFE_RID.match("..")
    assert not _SAFE_RID.match("a/b")
    assert not _SAFE_RID.match("a b")
    assert not _SAFE_RID.match("")
    assert not _SAFE_RID.match("x" * 65)  # too long


@pytest.mark.asyncio
async def test_bridge_ignores_unsafe_request_id():
    session = FakeSession([])
    # A request file whose id contains a space — must NOT be serviced.
    session.fs["/workspace/_rpc/req_bad id.json"] = json.dumps({"tool": "x", "args": {}})
    node = CodeActionNode("reason", sandbox_factory=_factory(session))
    from promptise.engine.state import NodeResult

    result = NodeResult()
    stop = asyncio.Event()
    task = asyncio.create_task(
        node._bridge_loop(session, {}, GraphState(messages=[]), {}, result, stop)
    )
    await asyncio.sleep(0.15)
    stop.set()
    await task

    # No response was written for the unsafe id, and no tool was invoked.
    assert not any(k.startswith("/workspace/_rpc/resp_") for k in session.fs)
    assert result.tool_calls == []
