"""Streaming runs the same steps as ``ainvoke()``: tokens after a tool, one
model call per step, parallel tools, per-call timings and indexes, and tool
errors reported as failures.

The model is a scripted fake chat model that emits tool calls and streams
its text word by word; it fails the test on any model call beyond its script.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterator
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import StructuredTool
from mcp.types import CallToolResult, TextContent
from pydantic import BaseModel, Field

from promptise.engine import PromptGraph, PromptGraphEngine, PromptNode
from promptise.engine.base import BaseNode
from promptise.engine.execution import GraphExecutionError
from promptise.engine.state import NodeEvent, NodeResult
from promptise.mcp.client import MCPToolError
from promptise.mcp.client._tool_adapter import _PromptiseMCPTool
from promptise.mcp.server import MCPServer, ToolError

# ---------------------------------------------------------------------------
# Fixtures: a scripted model and tools
# ---------------------------------------------------------------------------


def _tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


class ScriptedModel(BaseChatModel):
    """Answers call *n* with ``script[n]``; streams its text word by word.

    ``delays[n]`` seconds pass before call *n* answers.  ``calls`` records the
    messages of every call; a call past the end of the script fails the test.
    """

    script: list[AIMessage]
    delays: list[float] = Field(default_factory=list)
    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        return self

    def _next(self, messages: list[BaseMessage]) -> tuple[AIMessage, float]:
        n = len(self.calls)
        self.calls.append(list(messages))
        if n >= len(self.script):
            raise AssertionError(f"unexpected model call #{n + 1} (script has {len(self.script)})")
        delay = self.delays[n] if n < len(self.delays) else 0.0
        return self.script[n], delay

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message, _delay = self._next(messages)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        message, delay = self._next(messages)
        await asyncio.sleep(delay)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _astream(
        self, messages, stop=None, run_manager=None, **kwargs
    ) -> AsyncIterator[ChatGenerationChunk]:
        message, delay = self._next(messages)
        await asyncio.sleep(delay)
        for word in re.findall(r"\S+\s*", str(message.content)):
            yield ChatGenerationChunk(message=AIMessageChunk(content=word))
        for index, tc in enumerate(message.tool_calls):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[
                        {
                            "name": tc["name"],
                            "args": json.dumps(tc["args"]),
                            "id": tc["id"],
                            "index": index,
                        }
                    ],
                )
            )


ORDERS = {
    "A-1001": {"status": "shipped", "carrier": "DHL"},
    "A-1002": {"status": "processing", "carrier": None},
}


def _order_tool(delays: dict[str, float], log: list[tuple[str, str, float]]) -> StructuredTool:
    """``get_order_status`` that sleeps ``delays[order_id]`` and logs start/end times."""

    async def get_order_status(order_id: str) -> str:
        log.append(("start", order_id, time.monotonic()))
        await asyncio.sleep(delays.get(order_id, 0.0))
        log.append(("end", order_id, time.monotonic()))
        return json.dumps({"order_id": order_id, **ORDERS[order_id]})

    return StructuredTool.from_function(
        coroutine=get_order_status,
        name="get_order_status",
        description="Get an order's status.",
    )


class _ErrorResultMulti:
    """Stands in for ``MCPMultiClient``: every call returns an MCP tool error."""

    def __init__(self, text: str) -> None:
        self.text = text

    async def call_tool(
        self, name: str, arguments: dict[str, Any], **kwargs: Any
    ) -> CallToolResult:
        return CallToolResult(content=[TextContent(type="text", text=self.text)], isError=True)


class _OrderArgs(BaseModel):
    order_id: str


def _mcp_tool_returning_error(message: str) -> _PromptiseMCPTool:
    """An MCP-backed tool whose server answers with ``ToolError(message)``."""
    return _PromptiseMCPTool(
        name="get_order_status",
        description="Get an order's status.",
        args_schema=_OrderArgs,
        tool_name="get_order_status",
        multi=_ErrorResultMulti(ToolError(message).to_text()),  # type: ignore[arg-type]
    )


async def _agent(model: ScriptedModel, tools: list[Any], monkeypatch: pytest.MonkeyPatch) -> Any:
    from promptise import build_agent

    monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")
    return await build_agent(model=model, servers={}, extra_tools=tools)


async def _collect(agent: Any, question: str) -> list[Any]:
    return [
        event
        async for event in agent.astream_with_tools(
            {"messages": [{"role": "user", "content": question}]}
        )
    ]


# ---------------------------------------------------------------------------
# astream_with_tools()
# ---------------------------------------------------------------------------


class TestAnswerAfterTool:
    @pytest.mark.asyncio
    async def test_answer_streams_token_by_token_after_the_tool(self, monkeypatch):
        answer = "Your order A-1001 has shipped with DHL."
        model = ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("get_order_status", {"order_id": "A-1001"}, "c1")],
                ),
                AIMessage(content=answer),
            ]
        )
        agent = await _agent(model, [_order_tool({}, [])], monkeypatch)

        events = await _collect(agent, "Where is A-1001?")

        types = [e.type for e in events]
        tokens = [e for e in events if e.type == "token"]
        assert types[:2] == ["tool_start", "tool_end"]
        assert types[2:-1] == ["token"] * len(tokens)
        assert types[-1] == "done"
        assert [t.text for t in tokens] == re.findall(r"\S+\s*", answer)
        assert tokens[-1].cumulative_text == answer

        done = events[-1]
        assert done.full_response == answer
        assert done.tool_calls == [
            {"name": "get_order_status", "summary": events[1].tool_summary, "success": True}
        ]

    @pytest.mark.asyncio
    async def test_one_model_call_per_step(self, monkeypatch):
        model = ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("get_order_status", {"order_id": "A-1001"}, "c1")],
                ),
                AIMessage(content="Shipped."),
            ]
        )
        agent = await _agent(model, [_order_tool({}, [])], monkeypatch)

        events = await _collect(agent, "Where is A-1001?")

        assert events[-1].type == "done"
        assert len(model.calls) == 2
        # The second call answers from the tool result.
        tool_messages = [m for m in model.calls[1] if isinstance(m, ToolMessage)]
        assert len(tool_messages) == 1
        assert json.loads(tool_messages[0].content)["status"] == "shipped"

    @pytest.mark.asyncio
    async def test_each_step_makes_exactly_one_model_call(self):
        model = ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("get_order_status", {"order_id": "A-1001"}, "c1")],
                ),
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("get_order_status", {"order_id": "A-1002"}, "c2")],
                ),
                AIMessage(content="Shipped."),
            ]
        )
        graph = PromptGraph.react(tools=[_order_tool({}, [])], system_prompt="Help.")
        engine = PromptGraphEngine(graph=graph, model=model)

        events = [
            e
            async for e in engine.astream_events(
                {"messages": [{"role": "user", "content": "Where is A-1001?"}]}
            )
        ]

        steps = [e for e in events if e["event"] == "on_node_end"]
        # Steps 1 and 2 each ask for a tool, step 3 answers: three calls.
        assert len(steps) == 3
        assert len(model.calls) == 3
        assert len([e for e in events if e["event"] == "on_node_start"]) == 3
        assert [
            e["data"]["input"]["order_id"] for e in events if e["event"] == "on_tool_start"
        ] == [
            "A-1001",
            "A-1002",
        ]

    @pytest.mark.asyncio
    async def test_answer_without_a_tool_is_one_model_call(self, monkeypatch):
        answer = "I can check the status of your orders."
        model = ScriptedModel(script=[AIMessage(content=answer)])
        agent = await _agent(model, [_order_tool({}, [])], monkeypatch)

        events = await _collect(agent, "What can you do?")

        assert [e.type for e in events][-1] == "done"
        assert "".join(e.text for e in events if e.type == "token") == answer
        assert events[-1].full_response == answer
        assert len(model.calls) == 1

    @pytest.mark.asyncio
    async def test_full_response_is_the_final_answer_not_the_preamble(self, monkeypatch):
        model = ScriptedModel(
            script=[
                AIMessage(
                    content="Let me look that up. ",
                    tool_calls=[_tool_call("get_order_status", {"order_id": "A-1001"}, "c1")],
                ),
                AIMessage(content="It has shipped."),
            ]
        )
        agent = await _agent(model, [_order_tool({}, [])], monkeypatch)

        events = await _collect(agent, "Where is A-1001?")

        tokens = [e for e in events if e.type == "token"]
        assert tokens[-1].cumulative_text == "Let me look that up. It has shipped."
        assert events[-1].full_response == "It has shipped."


class TestToolTimingsAndIndexes:
    @pytest.mark.asyncio
    async def test_parallel_calls_have_their_own_index_and_duration(self, monkeypatch):
        log: list[tuple[str, str, float]] = []
        model = ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("get_order_status", {"order_id": "A-1001"}, "slow"),
                        _tool_call("get_order_status", {"order_id": "A-1002"}, "fast"),
                    ],
                ),
                AIMessage(content="Both found."),
            ],
            # The first model call takes a while: a duration measured from
            # the start of the run would include it.
            delays=[0.3, 0.0],
        )
        tool = _order_tool({"A-1001": 0.4, "A-1002": 0.1}, log)
        agent = await _agent(model, [tool], monkeypatch)

        events = await _collect(agent, "Where are A-1001 and A-1002?")

        starts = [e for e in events if e.type == "tool_start"]
        ends = [e for e in events if e.type == "tool_end"]
        assert [(s.arguments["order_id"], s.tool_index) for s in starts] == [
            ("A-1001", 0),
            ("A-1002", 1),
        ]
        # Both start before either ends; the fast call's end comes first.
        types = [e.type for e in events]
        assert types.index("tool_end") > types.index("tool_start", types.index("tool_start") + 1)
        assert [e.tool_index for e in ends] == [1, 0]

        fast, slow = ends
        assert 80 <= fast.duration_ms < 300
        assert 380 <= slow.duration_ms < 600

    @pytest.mark.asyncio
    async def test_tools_run_in_parallel_while_streaming(self, monkeypatch):
        log: list[tuple[str, str, float]] = []
        model = ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("get_order_status", {"order_id": "A-1001"}, "c1"),
                        _tool_call("get_order_status", {"order_id": "A-1002"}, "c2"),
                    ],
                ),
                AIMessage(content="Done."),
            ]
        )
        agent = await _agent(model, [_order_tool({"A-1001": 0.2, "A-1002": 0.2}, log)], monkeypatch)

        await _collect(agent, "Both orders?")

        times = {(kind, order): t for kind, order, t in log}
        # The second call starts before the first one ends.
        assert times[("start", "A-1002")] < times[("end", "A-1001")]

    @pytest.mark.asyncio
    async def test_tools_run_in_parallel_in_ainvoke_too(self, monkeypatch):
        log: list[tuple[str, str, float]] = []
        model = ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("get_order_status", {"order_id": "A-1001"}, "c1"),
                        _tool_call("get_order_status", {"order_id": "A-1002"}, "c2"),
                    ],
                ),
                AIMessage(content="Done."),
            ]
        )
        agent = await _agent(model, [_order_tool({"A-1001": 0.2, "A-1002": 0.2}, log)], monkeypatch)

        await agent.ainvoke({"messages": [{"role": "user", "content": "Both orders?"}]})

        times = {(kind, order): t for kind, order, t in log}
        assert times[("start", "A-1002")] < times[("end", "A-1001")]
        assert len(model.calls) == 2


class TestToolErrors:
    @pytest.mark.asyncio
    async def test_tool_error_is_success_false_with_its_message(self, monkeypatch):
        model = ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("get_order_status", {"order_id": "Z-9"}, "c1")],
                ),
                AIMessage(content="I could not find that order."),
            ]
        )
        agent = await _agent(
            model, [_mcp_tool_returning_error("No order found with ID Z-9.")], monkeypatch
        )

        events = await _collect(agent, "Where is Z-9?")

        tool_end = next(e for e in events if e.type == "tool_end")
        assert tool_end.success is False
        assert tool_end.tool_summary == "No order found with ID Z-9."
        assert events[-1].tool_calls == [
            {"name": "get_order_status", "summary": "No order found with ID Z-9.", "success": False}
        ]
        # The model still sees the structured error, flagged as an error.
        tool_message = next(m for m in model.calls[1] if isinstance(m, ToolMessage))
        assert tool_message.status == "error"
        assert json.loads(tool_message.content)["error"]["message"] == "No order found with ID Z-9."

    @pytest.mark.asyncio
    async def test_tool_error_counts_as_failed_in_ainvoke(self, monkeypatch):
        model = ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("get_order_status", {"order_id": "Z-9"}, "c1")],
                ),
                AIMessage(content="Not found."),
            ]
        )
        agent = await _agent(model, [_mcp_tool_returning_error("No order Z-9.")], monkeypatch)

        out = await agent.ainvoke({"messages": [{"role": "user", "content": "Where is Z-9?"}]})

        tool_message = next(m for m in out["messages"] if isinstance(m, ToolMessage))
        assert tool_message.status == "error"

    @pytest.mark.asyncio
    async def test_raising_tool_is_success_false_without_its_exception_text(self, monkeypatch):
        async def get_order_status(order_id: str) -> str:
            raise RuntimeError("connect to postgres://admin:hunter2@db failed")

        tool = StructuredTool.from_function(
            coroutine=get_order_status, name="get_order_status", description="Status."
        )
        model = ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("get_order_status", {"order_id": "A-1"}, "c1")],
                ),
                AIMessage(content="The lookup failed."),
            ]
        )
        agent = await _agent(model, [tool], monkeypatch)

        events = await _collect(agent, "Where is A-1?")

        tool_end = next(e for e in events if e.type == "tool_end")
        assert tool_end.success is False
        assert tool_end.tool_summary == "Tool call failed"
        assert tool_end.tool_index == 0
        assert "hunter2" not in "".join(e.to_json() for e in events)
        assert events[-1].full_response == "The lookup failed."

    @pytest.mark.asyncio
    async def test_approval_wrapper_keeps_the_error_status(self):
        from unittest.mock import AsyncMock

        from promptise.approval import (
            ApprovalDecision,
            ApprovalPolicy,
            CallbackApprovalHandler,
            wrap_tools_with_approval,
        )

        policy = ApprovalPolicy(
            tools=["get_*"],
            handler=CallbackApprovalHandler(
                AsyncMock(return_value=ApprovalDecision(approved=True))
            ),
        )
        [wrapped] = wrap_tools_with_approval([_mcp_tool_returning_error("No order Z-9.")], policy)

        # The wrapper does not turn the server's error into a result.
        with pytest.raises(MCPToolError) as info:
            await wrapped.ainvoke(_tool_call("get_order_status", {"order_id": "Z-9"}, "c1"))
        assert info.value.message == "No order Z-9."


class TestServerToolErrors:
    @staticmethod
    async def _call(server: MCPServer, name: str, args: dict[str, Any]) -> CallToolResult:
        import mcp.types as t

        handler = server._build_lowlevel_server().request_handlers[t.CallToolRequest]
        res = await handler(
            t.CallToolRequest(
                method="tools/call", params=t.CallToolRequestParams(name=name, arguments=args)
            )
        )
        return res.root

    @pytest.mark.asyncio
    async def test_tool_error_is_an_is_error_result(self):
        server = MCPServer("orders")

        @server.tool()
        async def get_order_status(order_id: str) -> dict:
            """Get an order."""
            if order_id not in ORDERS:
                raise ToolError(f"No order found with ID {order_id}.")
            return ORDERS[order_id]

        failed = await self._call(server, "get_order_status", {"order_id": "Z-9"})
        assert failed.isError is True
        error = json.loads(failed.content[0].text)["error"]
        assert error == {
            "code": "TOOL_ERROR",
            "message": "No order found with ID Z-9.",
            "retryable": False,
        }

        ok = await self._call(server, "get_order_status", {"order_id": "A-1001"})
        assert ok.isError is False
        assert json.loads(ok.content[0].text)["status"] == "shipped"

    @pytest.mark.asyncio
    async def test_unexpected_exception_is_an_is_error_result_without_details(self):
        server = MCPServer("orders")

        @server.tool()
        async def get_order_status(order_id: str) -> dict:
            """Get an order."""
            raise RuntimeError("password=hunter2")

        failed = await self._call(server, "get_order_status", {"order_id": "A-1"})
        assert failed.isError is True
        assert "hunter2" not in failed.content[0].text
        assert json.loads(failed.content[0].text)["error"]["code"] == "INTERNAL_ERROR"

    @pytest.mark.asyncio
    async def test_unknown_tool_is_an_is_error_result(self):
        server = MCPServer("orders")

        @server.tool()
        async def ping() -> str:
            """Ping."""
            return "pong"

        failed = await self._call(server, "nope", {})
        assert failed.isError is True
        assert json.loads(failed.content[0].text)["error"]["code"] == "TOOL_NOT_FOUND"


# ---------------------------------------------------------------------------
# agent.astream()
# ---------------------------------------------------------------------------


class TestAgentAstream:
    @pytest.mark.asyncio
    async def test_yields_the_conversation_after_each_step(self, monkeypatch):
        def script() -> list[AIMessage]:
            return [
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("get_order_status", {"order_id": "A-1001"}, "c1")],
                ),
                AIMessage(content="It has shipped."),
            ]

        model = ScriptedModel(script=script())
        agent = await _agent(model, [_order_tool({}, [])], monkeypatch)
        question = {"messages": [{"role": "user", "content": "Where is A-1001?"}]}

        chunks = [chunk async for chunk in agent.astream(question)]

        assert len(chunks) == 2
        assert [type(m).__name__ for m in chunks[0]["messages"][1:]] == ["AIMessage", "ToolMessage"]
        assert chunks[-1]["messages"][-1].content == "It has shipped."
        assert len(model.calls) == 2

        # The last chunk is what ainvoke() returns.
        model2 = ScriptedModel(script=script())
        agent2 = await _agent(model2, [_order_tool({}, [])], monkeypatch)
        out = await agent2.ainvoke(question)
        assert [type(m).__name__ for m in out["messages"]] == [
            type(m).__name__ for m in chunks[-1]["messages"]
        ]
        assert out["messages"][-1].content == chunks[-1]["messages"][-1].content

    @pytest.mark.asyncio
    async def test_input_guardrail_violation_raises(self, monkeypatch):
        class Blocked(Exception):
            pass

        class BlockingGuard:
            async def check_input(self, text: str) -> str:
                raise Blocked("blocked")

            async def check_output(self, text: str) -> str:
                return text

        model = ScriptedModel(script=[AIMessage(content="never")])
        agent = await _agent(model, [], monkeypatch)
        agent._guardrails = BlockingGuard()

        with pytest.raises(Blocked):
            async for _chunk in agent.astream({"messages": [{"role": "user", "content": "hi"}]}):
                pass
        assert model.calls == []

    @pytest.mark.asyncio
    async def test_output_guardrail_redacts_the_answer_chunk(self, monkeypatch):
        class RedactingGuard:
            async def check_input(self, text: str) -> str:
                return text

            async def check_output(self, text: str) -> str:
                return text.replace("4111-1111", "[REDACTED]")

        model = ScriptedModel(script=[AIMessage(content="Card 4111-1111 is on file.")])
        agent = await _agent(model, [], monkeypatch)
        agent._guardrails = RedactingGuard()

        chunks = [c async for c in agent.astream({"messages": [{"role": "user", "content": "hi"}]})]

        assert chunks[-1]["messages"][-1].content == "Card [REDACTED] is on file."


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestEngineStreaming:
    @pytest.mark.asyncio
    async def test_tool_events_keep_their_run_id(self):
        model = ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[
                        _tool_call("get_order_status", {"order_id": "A-1001"}, "c1"),
                        _tool_call("get_order_status", {"order_id": "A-1002"}, "c2"),
                    ],
                ),
                AIMessage(content="Done."),
            ]
        )
        tool = _order_tool({"A-1001": 0.05}, [])
        graph = PromptGraph.react(tools=[tool], system_prompt="Help.")
        engine = PromptGraphEngine(graph=graph, model=model)

        events = [
            e
            async for e in engine.astream_events(
                {"messages": [{"role": "user", "content": "Both?"}]}
            )
        ]

        starts = {e["run_id"] for e in events if e["event"] == "on_tool_start"}
        ends = {e["run_id"] for e in events if e["event"] == "on_tool_end"}
        assert len(starts) == 2
        assert starts == ends
        token_runs = {e["data"]["run_id"] for e in events if e["event"] == "on_chat_model_stream"}
        assert len(token_runs) == 1  # one model call streamed text

    @pytest.mark.asyncio
    async def test_stream_without_a_result_fails_instead_of_running_the_node_again(self):
        executions: list[str] = []

        class NoResultNode(BaseNode):
            async def execute(self, state, config) -> NodeResult:
                executions.append("execute")
                return NodeResult(node_name=self.name)

            async def stream(self, state, config) -> AsyncIterator[NodeEvent]:
                executions.append("stream")
                yield NodeEvent(event="on_node_end", node_name=self.name)

        graph = PromptGraph("g")
        graph.add_node(NoResultNode("custom"))
        graph.set_entry("custom")
        engine = PromptGraphEngine(graph=graph, model=ScriptedModel(script=[]))

        with pytest.raises(GraphExecutionError, match="ended without an on_node_end"):
            async for _event in engine.astream_events({"messages": []}):
                pass
        assert executions == ["stream"]

    @pytest.mark.asyncio
    async def test_prompt_node_subclass_with_its_own_execute_runs_once(self):
        class Shouting(PromptNode):
            async def execute(self, state, config) -> NodeResult:
                result = await super().execute(state, config)
                result.output = str(result.raw_output).upper()
                return result

        model = ScriptedModel(script=[AIMessage(content="hello")])
        graph = PromptGraph("g")
        graph.add_node(Shouting("shout"))
        graph.set_entry("shout")
        engine = PromptGraphEngine(graph=graph, model=model)

        chunks = [
            c async for c in engine.astream({"messages": [{"role": "user", "content": "hi"}]})
        ]

        assert len(model.calls) == 1
        assert chunks[-1]["messages"][-1].content == "hello"


class TestContentBlockChunks:
    @pytest.mark.asyncio
    async def test_text_blocks_stream_as_tokens(self, monkeypatch):
        class BlockModel(ScriptedModel):
            """Streams its text as Anthropic-style content blocks."""

            async def _astream(
                self, messages, stop=None, run_manager=None, **kwargs
            ) -> AsyncIterator[ChatGenerationChunk]:
                message, _delay = self._next(messages)
                for index, word in enumerate(re.findall(r"\S+\s*", str(message.content))):
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(
                            content=[{"type": "text", "text": word, "index": index}]
                        )
                    )

        model = BlockModel(script=[AIMessage(content="Hello from blocks.")])
        agent = await _agent(model, [], monkeypatch)

        events = await _collect(agent, "hi")

        assert [e.type for e in events] == ["token", "token", "token", "done"]
        assert events[-1].full_response == "Hello from blocks."


# ---------------------------------------------------------------------------
# Engine: a streamed run is the same traversal as ainvoke()
# ---------------------------------------------------------------------------


class _RecordingHook:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.results: list[NodeResult] = []

    async def pre_node(self, node, state):
        self.calls.append(("pre", node.name))
        return state

    async def post_node(self, node, result, state):
        self.calls.append(("post", node.name))
        self.results.append(result)
        return result


class TestEngineStreamedRunMatchesAinvoke:
    @staticmethod
    def _engine(model: ScriptedModel, hook: _RecordingHook) -> PromptGraphEngine:
        graph = PromptGraph.react(tools=[_order_tool({}, [])], system_prompt="Help.")
        return PromptGraphEngine(graph=graph, model=model, hooks=[hook])

    @staticmethod
    def _script() -> list[AIMessage]:
        return [
            AIMessage(
                content="",
                tool_calls=[_tool_call("get_order_status", {"order_id": "A-1001"}, "c1")],
            ),
            AIMessage(content="Shipped."),
        ]

    @pytest.mark.asyncio
    async def test_hooks_run_when_streaming(self):
        hook = _RecordingHook()
        engine = self._engine(ScriptedModel(script=self._script()), hook)

        async for _event in engine.astream_events(
            {"messages": [{"role": "user", "content": "Where is A-1001?"}]}
        ):
            pass

        hook_ainvoke = _RecordingHook()
        await self._engine(ScriptedModel(script=self._script()), hook_ainvoke).ainvoke(
            {"messages": [{"role": "user", "content": "Where is A-1001?"}]}
        )
        assert hook.calls == hook_ainvoke.calls
        assert len(hook.calls) == 4  # pre and post for each of the two steps

    @pytest.mark.asyncio
    async def test_last_report_after_a_stream(self):
        engine = self._engine(ScriptedModel(script=self._script()), _RecordingHook())

        chunks = [
            c
            async for c in engine.astream(
                {"messages": [{"role": "user", "content": "Where is A-1001?"}]}
            )
        ]

        assert chunks[-1]["messages"][-1].content == "Shipped."
        report = engine.last_report
        assert report is not None
        assert report.total_iterations == 2
        assert report.tool_calls == 1

    @pytest.mark.asyncio
    async def test_failed_node_emits_on_node_error_then_raises(self):
        class Broken(ScriptedModel):
            async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
                raise RuntimeError("401 invalid api key")
                yield  # pragma: no cover

        graph = PromptGraph.react(tools=[_order_tool({}, [])], system_prompt="Help.")
        engine = PromptGraphEngine(graph=graph, model=Broken(script=[]))

        events: list[dict[str, Any]] = []
        with pytest.raises(GraphExecutionError):
            async for event in engine.astream_events(
                {"messages": [{"role": "user", "content": "hi"}]}
            ):
                events.append(event)

        errors = [e for e in events if e["event"] == "on_node_error"]
        assert len(errors) == 1
        assert "401 invalid api key" in errors[0]["data"]["error"]
        assert engine.last_report is not None and engine.last_report.error


# ---------------------------------------------------------------------------
# Tool errors in ainvoke(): counted as failed, never cached as facts
# ---------------------------------------------------------------------------


class _CountingErrorMulti(_ErrorResultMulti):
    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.count = 0

    async def call_tool(self, name: str, arguments: dict[str, Any], **kwargs: Any) -> Any:
        self.count += 1
        return await super().call_tool(name, arguments, **kwargs)


class TestToolErrorBookkeeping:
    @pytest.mark.asyncio
    async def test_tool_error_counts_in_tool_calls_failed(self):
        hook = _RecordingHook()
        model = ScriptedModel(
            script=[
                AIMessage(
                    content="",
                    tool_calls=[_tool_call("get_order_status", {"order_id": "Z-9"}, "c1")],
                ),
                AIMessage(content="Not found."),
            ]
        )
        graph = PromptGraph.react(
            tools=[_mcp_tool_returning_error("No order Z-9.")], system_prompt="Help."
        )
        engine = PromptGraphEngine(graph=graph, model=model, hooks=[hook])

        await engine.ainvoke({"messages": [{"role": "user", "content": "Where is Z-9?"}]})

        assert hook.results[0].tool_calls_failed == 1
        assert hook.results[0].tool_calls[0]["success"] is False

    @pytest.mark.asyncio
    async def test_ledger_does_not_serve_a_failed_call_from_cache(self):
        multi = _CountingErrorMulti(ToolError("Warehouse busy, retry.").to_text())
        tool = _PromptiseMCPTool(
            name="get_order_status",
            description="Get an order's status.",
            args_schema=_OrderArgs,
            tool_name="get_order_status",
            multi=multi,  # type: ignore[arg-type]
        )
        retry = [_tool_call("get_order_status", {"order_id": "A-1"}, "c1")]
        model = ScriptedModel(
            script=[
                AIMessage(content="", tool_calls=retry),
                AIMessage(content="", tool_calls=[{**retry[0], "id": "c2"}]),
                AIMessage(content="Still busy."),
            ]
        )
        graph = PromptGraph("g")
        graph.add_node(
            PromptNode("ask", tools=[tool], context_scope="ledger", instructions="Help.")
        )
        graph.set_entry("ask")
        engine = PromptGraphEngine(graph=graph, model=model)

        await engine.ainvoke({"messages": [{"role": "user", "content": "A-1?"}]})

        # The retry reached the server: an error is not a fact to reuse.
        assert multi.count == 2


# ---------------------------------------------------------------------------
# Callbacks receive a tool call's ToolMessage
# ---------------------------------------------------------------------------


class TestCallbacksWithToolMessages:
    def test_observability_records_the_content_and_error_status(self):
        from uuid import uuid4

        from promptise.callback_handler import PromptiseCallbackHandler
        from promptise.observability import ObservabilityCollector, TimelineEventType

        collector = ObservabilityCollector("s")
        handler = PromptiseCallbackHandler(collector, agent_id="a")
        run_id = uuid4()
        handler.on_tool_start({"name": "get_order_status"}, "{}", run_id=run_id)
        handler.on_tool_end(
            ToolMessage(content="No order Z-9.", tool_call_id="c1", status="error"),
            run_id=run_id,
        )

        [entry] = collector.query(event_types=[TimelineEventType.TOOL_RESULT])
        assert entry.metadata["result_preview"] == "No order Z-9."
        assert entry.metadata["status"] == "error"

    @pytest.mark.asyncio
    async def test_runtime_health_records_the_content(self):
        from unittest.mock import AsyncMock

        from promptise.runtime.callbacks import RuntimeCallbackHandler

        health = AsyncMock()
        handler = RuntimeCallbackHandler(health=health)

        await handler.on_tool_end(ToolMessage(content="shipped", tool_call_id="c1"))

        health.record_response.assert_awaited_once_with("shipped")


# ---------------------------------------------------------------------------
# Caller identity and memory while streaming
# ---------------------------------------------------------------------------


def _caller_tool(seen: list[Any]) -> StructuredTool:
    async def whoami() -> str:
        from promptise.agent import get_current_caller

        caller = get_current_caller()
        seen.append(caller.user_id if caller is not None else None)
        return "ok"

    return StructuredTool.from_function(coroutine=whoami, name="whoami", description="Who?")


class TestStreamingCallerAndMemory:
    @staticmethod
    def _script() -> list[AIMessage]:
        return [
            AIMessage(content="", tool_calls=[_tool_call("whoami", {}, "c1")]),
            AIMessage(content="You are known."),
        ]

    @pytest.mark.asyncio
    async def test_astream_inherits_the_ambient_caller(self, monkeypatch):
        from promptise.agent import CallerContext, _caller_ctx_var

        seen: list[Any] = []
        agent = await _agent(
            ScriptedModel(script=self._script()), [_caller_tool(seen)], monkeypatch
        )
        token = _caller_ctx_var.set(CallerContext(user_id="alice"))
        try:
            async for _chunk in agent.astream({"messages": [{"role": "user", "content": "me?"}]}):
                pass
        finally:
            _caller_ctx_var.reset(token)

        assert seen == ["alice"]

    @pytest.mark.asyncio
    async def test_astream_with_tools_inherits_the_ambient_caller(self, monkeypatch):
        from promptise.agent import CallerContext, _caller_ctx_var

        seen: list[Any] = []
        agent = await _agent(
            ScriptedModel(script=self._script()), [_caller_tool(seen)], monkeypatch
        )
        token = _caller_ctx_var.set(CallerContext(user_id="alice"))
        try:
            await _collect(agent, "me?")
        finally:
            _caller_ctx_var.reset(token)

        assert seen == ["alice"]

    @pytest.mark.asyncio
    async def test_astream_stores_the_exchange_in_memory(self, monkeypatch):
        from promptise import build_agent
        from promptise.memory import InMemoryProvider

        monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")
        provider = InMemoryProvider()
        agent = await build_agent(
            model=ScriptedModel(script=[AIMessage(content="Paris.")]),
            servers={},
            memory=provider,
            memory_auto_store=True,
        )

        async for _chunk in agent.astream(
            {"messages": [{"role": "user", "content": "Capital of France?"}]}
        ):
            pass

        stored = [content for content, *_rest in provider._store.values()]
        assert any("Capital of France?" in c and "Paris." in c for c in stored)


# ---------------------------------------------------------------------------
# Adaptive strategy while streaming: failures recorded, lessons injected
# ---------------------------------------------------------------------------


class _Synthesis:
    """Lesson synthesis model: answers with one lesson, records prompts."""

    LESSON = "Order IDs look like A-1001: the letter A, a dash and four digits."

    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def ainvoke(self, prompt: Any) -> Any:
        from types import SimpleNamespace

        self.prompts.append(str(prompt))
        lessons = [{"tool": "get_order_status", "lesson": self.LESSON}]
        return SimpleNamespace(content=json.dumps({"lessons": lessons}))


class TestStreamingAdaptiveStrategy:
    @staticmethod
    def _failing_turn(order_id: str, call_id: str) -> list[AIMessage]:
        return [
            AIMessage(
                content="",
                tool_calls=[_tool_call("get_order_status", {"order_id": order_id}, call_id)],
            ),
            AIMessage(content="I could not find it."),
        ]

    @staticmethod
    async def _adaptive_agent(model: ScriptedModel, synth: _Synthesis, monkeypatch) -> Any:
        from promptise import build_agent
        from promptise.memory import InMemoryProvider
        from promptise.strategy import AdaptiveStrategyConfig

        monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")
        return await build_agent(
            model=model,
            servers={},
            extra_tools=[_mcp_tool_returning_error("No order found with ID Z-9.")],
            memory=InMemoryProvider(),
            adaptive=AdaptiveStrategyConfig(
                enabled=True, synthesis_threshold=1, synthesis_model=synth
            ),
        )

    @pytest.mark.asyncio
    async def test_astream_with_tools_records_failures_and_injects_lessons(self, monkeypatch):
        from promptise.agent import CallerContext

        alice = CallerContext(user_id="alice")
        synth = _Synthesis()
        model = ScriptedModel(
            script=self._failing_turn("Z-9", "c1") + [AIMessage(content="Hello again.")]
        )
        agent = await self._adaptive_agent(model, synth, monkeypatch)
        try:
            events = [
                e
                async for e in agent.astream_with_tools(
                    {"messages": [{"role": "user", "content": "Where is order Z-9?"}]}, caller=alice
                )
            ]
            assert next(e for e in events if e.type == "tool_end").success is False
            # The MCP error reached adaptive strategy, under its MCP code.
            assert len(synth.prompts) == 1
            assert "Tool 'get_order_status' failed with TOOL_ERROR" in synth.prompts[0]

            # The next streamed run gets the lesson.
            async for _event in agent.astream_with_tools(
                {"messages": [{"role": "user", "content": "Where is order Z-9?"}]}, caller=alice
            ):
                pass
            prompt = "\n".join(str(m.content) for m in model.calls[-1])
            assert _Synthesis.LESSON in prompt
        finally:
            await agent.shutdown()

    @pytest.mark.asyncio
    async def test_astream_records_failures_and_injects_lessons(self, monkeypatch):
        from promptise.agent import CallerContext

        alice = CallerContext(user_id="alice")
        synth = _Synthesis()
        model = ScriptedModel(
            script=self._failing_turn("Z-9", "c1") + [AIMessage(content="Hello again.")]
        )
        agent = await self._adaptive_agent(model, synth, monkeypatch)
        try:
            async for _chunk in agent.astream(
                {"messages": [{"role": "user", "content": "Where is order Z-9?"}]}, caller=alice
            ):
                pass
            assert len(synth.prompts) == 1
            assert "Tool 'get_order_status' failed with TOOL_ERROR" in synth.prompts[0]

            async for _chunk in agent.astream(
                {"messages": [{"role": "user", "content": "Where is order Z-9?"}]}, caller=alice
            ):
                pass
            prompt = "\n".join(str(m.content) for m in model.calls[-1])
            assert _Synthesis.LESSON in prompt
        finally:
            await agent.shutdown()
