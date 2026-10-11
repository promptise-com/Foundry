"""Semantic tool selection end to end: what each model call is offered.

Uses a scripted chat model that records the tools bound to every model
call, so these run without an API key (the embedding model runs locally).
"""

from __future__ import annotations

import importlib.util
import json
from typing import Any

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from mcp.types import Tool
from pydantic import BaseModel, Field

from promptise import OptimizationLevel, ToolOptimizationConfig, build_agent
from promptise.engine import PromptGraph, PromptGraphEngine
from promptise.engine.nodes import TOOL_SELECTOR_KEY
from promptise.mcp.client import MCPToolAdapter
from promptise.tool_optimization import (
    ToolIndex,
    _RequestMoreToolsTool,
    _resolve_config,
    _ToolSelector,
    build_selection_query,
    require_semantic_dependencies,
)

# ======================================================================
# Helpers
# ======================================================================

TOOL_DOCS = {
    "suspend_user": "Suspend a user so they can't sign in; use when someone leaves the company.",
    "invite_user": "Send an email invitation to join the organization.",
    "list_users": "List the users in the organization.",
    "rollback_deployment": "Roll production back to the previous successful deployment.",
    "promote_deployment": "Promote a preview deployment to production.",
    "cancel_deployment": "Cancel a deployment that is still building.",
    "list_audit_events": "List audit log events: who did what and when.",
    "create_database_backup": "Take an on-demand backup of a database now.",
    "restore_database_backup": "Restore a database from one of its backups.",
    "rotate_api_key": "Replace an API key's secret with a new one.",
    "create_dns_record": "Create a DNS record in a domain's zone.",
    "acknowledge_incident": "Acknowledge an incident so on-call stops paging.",
    "get_organization": "Show the organization's name, owner and region.",
}


class _IdArgs(BaseModel):
    target: str = Field(default="", description="What to act on.")


def _make_tools(calls: list[str]) -> list[BaseTool]:
    def make(name: str, doc: str) -> BaseTool:
        async def run(target: str = "") -> str:
            calls.append(name)
            return f"ok: {name}"

        return StructuredTool.from_function(
            coroutine=run, name=name, description=doc, args_schema=_IdArgs
        )

    return [make(name, doc) for name, doc in TOOL_DOCS.items()]


class _Scripted(BaseChatModel):
    """Chat model that replays scripted replies; tools are bound as invocation params."""

    script: list[Any] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self.bind(tools=[convert_to_openai_tool(t) for t in tools], **kwargs)

    def _generate(self, messages: Any, stop: Any = None, run_manager: Any = None, **kw: Any):
        reply = self.script.pop(0) if self.script else AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=reply)])


class _ToolsSeen(AsyncCallbackHandler):
    """Records the tool names of every model call."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def on_chat_model_start(self, serialized: Any, messages: Any, **kwargs: Any) -> None:
        tools = (kwargs.get("invocation_params") or {}).get("tools") or []
        self.calls.append([t["function"]["name"] for t in tools])


def _tool_call(name: str, args: dict[str, Any], call_id: str = "c1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


async def _semantic_agent(config: ToolOptimizationConfig, script: list[Any] | None = None):
    calls: list[str] = []
    agent = await build_agent(
        model=_Scripted(script=list(script or [])),
        servers={},
        extra_tools=_make_tools(calls),
        optimize_tools=config,
    )
    return agent, calls


async def _offered(agent: Any, messages: list[Any]) -> list[list[str]]:
    seen = _ToolsSeen()
    await agent.ainvoke({"messages": messages}, config={"callbacks": [seen]})
    return seen.calls


def _user(text: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text}]


SEMANTIC = OptimizationLevel.SEMANTIC


# ======================================================================
# build_selection_query
# ======================================================================


class TestBuildSelectionQuery:
    def test_single_message_is_the_query(self):
        assert build_selection_query(_user("Suspend u_42")) == "Suspend u_42"

    def test_follow_up_keeps_the_request_it_answers(self):
        messages = [
            {"role": "user", "content": "The last deploy broke checkout. What can we do?"},
            {"role": "assistant", "content": "I can roll production back. Shall I?"},
            {"role": "user", "content": "Yes, go ahead."},
        ]
        query = build_selection_query(messages)
        lines = query.splitlines()
        assert lines[0] == "Yes, go ahead."
        assert lines[1] == "I can roll production back. Shall I?"
        assert "The last deploy broke checkout" in query

    def test_user_turns_limits_history(self):
        messages = [
            HumanMessage(content="first"),
            AIMessage(content="a"),
            HumanMessage(content="second"),
            AIMessage(content="b"),
            HumanMessage(content="third"),
        ]
        query = build_selection_query(messages, user_turns=2)
        assert "third" in query and "second" in query
        assert "first" not in query

    def test_includes_tool_calls_of_previous_and_current_turn(self):
        messages = [
            HumanMessage(content="old request"),
            _tool_call("list_users", {}, "a"),
            ToolMessage(content="...", tool_call_id="a"),
            HumanMessage(content="which deploy failed?"),
            _tool_call("cancel_deployment", {"target": "dep_1"}, "b"),
            ToolMessage(content="...", tool_call_id="b"),
            HumanMessage(content="now roll it back"),
            _tool_call("rollback_deployment", {"target": "prj"}, "c"),
        ]
        query = build_selection_query(messages)
        assert "cancel_deployment(target=dep_1)" in query
        assert "rollback_deployment(target=prj)" in query
        assert "list_users" not in query  # two turns back

    def test_openai_style_dict_tool_calls(self):
        messages = [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "rotate_api_key", "arguments": '{"target": "k"}'}}
                ],
            },
        ]
        assert "rotate_api_key" in build_selection_query(messages)

    def test_no_user_message(self):
        assert build_selection_query([AIMessage(content="hi")]) == ""


# ======================================================================
# ToolIndex
# ======================================================================


@pytest.fixture(scope="module")
def index() -> ToolIndex:
    return ToolIndex(_make_tools([]))


class TestToolIndexSelection:
    def test_returns_exactly_top_k(self, index):
        for k in (1, 3, 5):
            assert len(index.select("suspend a user", top_k=k)) == k

    def test_preserved_tools_are_added_on_top(self, index):
        selected = [
            t.name for t in index.select("suspend a user", top_k=3, preserve={"get_organization"})
        ]
        assert len(selected) == 4
        assert selected[-1] == "get_organization"
        assert selected[0] == "suspend_user"

    def test_unknown_preserved_names_ignored(self, index):
        assert len(index.select("x", top_k=2, preserve={"nope"})) == 2

    def test_query_scores_are_cached(self, index, monkeypatch):
        index.select("cache me please", top_k=1)
        monkeypatch.setattr(index, "_embed_fn", lambda *a, **k: pytest.fail("re-embedded"))
        index.select("cache me please", top_k=2)

    def test_contains_and_summaries(self, index):
        assert "suspend_user" in index and "nope" not in index
        assert index.summaries(["invite_user", "nope"]) == (
            "- invite_user: Send an email invitation to join the organization."
        )


class TestMissingSentenceTransformers:
    def test_require_names_the_extra(self, monkeypatch):
        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a: None if name == "sentence_transformers" else real(name, *a),
        )
        with pytest.raises(ImportError, match=r"promptise\[tool-optimization\]"):
            require_semantic_dependencies()

    def test_tool_index_names_the_extra(self, monkeypatch):
        monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)
        with pytest.raises(ImportError, match=r"promptise\[tool-optimization\]"):
            ToolIndex(_make_tools([]))

    async def test_build_agent_fails_before_connecting(self, monkeypatch):
        from promptise.config import StdioServerSpec

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a: None if name == "sentence_transformers" else real(name, *a),
        )
        # The server command doesn't exist: reaching the connect step would
        # raise a different error.
        with pytest.raises(ImportError, match="sentence-transformers"):
            await build_agent(
                model=_Scripted(),
                servers={"x": StdioServerSpec(command="/nonexistent/server")},
                optimize_tools="semantic",
            )


class TestConfigValidation:
    @pytest.mark.parametrize("field", ["semantic_top_k", "semantic_context_turns"])
    def test_rejects_zero(self, field):
        with pytest.raises(ValueError, match=field):
            _resolve_config(ToolOptimizationConfig(level=SEMANTIC, **{field: 0}))

    def test_context_turns_default(self):
        assert _resolve_config(ToolOptimizationConfig(level=SEMANTIC)).semantic_context_turns == 3


# ======================================================================
# _ToolSelector and request_more_tools
# ======================================================================


class _State:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages


class TestToolSelector:
    def _selector(self, index: ToolIndex, **overrides: Any) -> _ToolSelector:
        cfg = _resolve_config(ToolOptimizationConfig(level=SEMANTIC, semantic_top_k=2, **overrides))
        return _ToolSelector(index, cfg)

    def test_unmanaged_tools_pass_through(self, index):
        extra = StructuredTool.from_function(func=lambda: "x", name="custom", description="c")
        offered = self._selector(index)(index.all_tools + [extra], _State(_user("suspend u_1")))
        names = [t.name for t in offered]
        assert "custom" in names
        assert len(names) == 3

    def test_preserve_and_recent_calls_are_kept(self, index):
        messages = [
            HumanMessage(content="suspend u_1"),
            _tool_call("create_dns_record", {}),
            ToolMessage(content="ok", tool_call_id="c1"),
        ]
        names = self._selector(index, preserve_tools={"get_organization"}).selected_names(messages)
        assert {"suspend_user", "get_organization", "create_dns_record"} <= names

    @pytest.mark.parametrize(
        ("args", "expected"),
        [
            ({"tool_names": ["list_audit_events", "nope"]}, {"list_audit_events"}),
            ({"query": "audit log of who did what"}, {"list_audit_events"}),
            ('{"tool_names": ["rotate_api_key"]}', {"rotate_api_key"}),
        ],
    )
    def test_request_more_tools_unlocks_what_it_returned(self, index, args, expected):
        messages = [
            HumanMessage(content="suspend u_1"),
            AIMessage(
                content="",
                tool_calls=[{"name": "request_more_tools", "args": {}, "id": "r"}],
            ),
        ]
        messages[1].tool_calls[0]["args"] = args  # type: ignore[index]
        assert expected <= self._selector(index).selected_names(messages)


class TestRequestMoreToolsIsBounded:
    """An argument-less request_more_tools call never enables the whole catalogue."""

    def _selector(self, index: ToolIndex, top_k: int = 4, **overrides: Any) -> _ToolSelector:
        cfg = _resolve_config(
            ToolOptimizationConfig(level=SEMANTIC, semantic_top_k=top_k, **overrides)
        )
        return _ToolSelector(index, cfg)

    @staticmethod
    def _browse(call_id: str) -> AIMessage:
        return _tool_call("request_more_tools", {}, call_id)

    @staticmethod
    async def _page(selector: _ToolSelector, tool: BaseTool, messages: list[Any]) -> set[str]:
        """Run the model call for *messages*, then an argument-less request: the tools listed."""
        selector(tool._tool_index.all_tools, _State(messages))  # type: ignore[attr-defined]
        result = await tool._arun()
        return {line[2:].split(":", 1)[0] for line in result.splitlines() if line.startswith("- ")}

    async def test_selector_unlocks_a_bounded_page(self, big_index):
        base = [HumanMessage(content="Rotate the API key of the billing service")]
        selector = self._selector(big_index)
        tool = _RequestMoreToolsTool(tool_index=big_index, top_k=4)
        page = await self._page(selector, tool, base)
        offered = selector.selected_names(base)
        after = selector.selected_names(
            [*base, self._browse("r1"), ToolMessage(content="...", tool_call_id="r1")]
        )
        assert len(offered) == 4
        assert len(page) == 4 and not page & offered  # new tools, not the ones offered
        assert page <= after
        assert len(after) <= 8  # top_k + one page, never the 200-tool catalogue

    async def test_each_further_call_pages_on(self, big_index):
        selector = self._selector(big_index)
        tool = _RequestMoreToolsTool(tool_index=big_index, top_k=4)
        turn = [HumanMessage(content="Rotate the API key of the billing service")]
        first = await self._page(selector, tool, turn)
        turn += [self._browse("r1"), ToolMessage(content="...", tool_call_id="r1")]
        second = await self._page(selector, tool, turn)
        turn += [self._browse("r2"), ToolMessage(content="...", tool_call_id="r2")]
        final = selector.selected_names(turn)
        assert len(first) == len(second) == 4
        assert not first & second
        assert first | second <= final
        assert len(final) <= 12

    async def test_tool_reply_says_how_to_get_more(self, big_index):
        result = await _RequestMoreToolsTool(tool_index=big_index, top_k=4)._arun()
        assert result.startswith("4 of 200 tools are now available")
        assert "again without arguments for more" in result

    async def test_tool_names_are_capped(self, big_index):
        from promptise.tool_optimization import MAX_TOOLS_PER_REQUEST

        names = big_index.all_tool_names[:150]
        result = await _RequestMoreToolsTool(tool_index=big_index)._arun(tool_names=names)
        assert result.startswith(f"{MAX_TOOLS_PER_REQUEST} of 200 tools are now available")
        assert f"{150 - MAX_TOOLS_PER_REQUEST} more of the named tools were not enabled" in result
        messages = [
            HumanMessage(content="enable them all"),
            _tool_call("request_more_tools", {"tool_names": names}, "r1"),
        ]
        assert len(self._selector(big_index).selected_names(messages)) <= MAX_TOOLS_PER_REQUEST + 4

    def test_offered_tools_never_pass_the_safe_maximum(self, big_index):
        from promptise.tool_optimization import MAX_OFFERED_TOOLS

        selector = self._selector(
            big_index, top_k=90, preserve_tools=set(big_index.all_tool_names[:60])
        )
        messages = [
            HumanMessage(content="do everything"),
            _tool_call("request_more_tools", {"tool_names": big_index.all_tool_names[60:]}, "r1"),
            ToolMessage(content="...", tool_call_id="r1"),
            self._browse("r2"),
        ]
        names = selector.selected_names(messages)
        assert len(names) == MAX_OFFERED_TOOLS < 128
        # Preserved tools are never the ones dropped.
        assert set(big_index.all_tool_names[:60]) <= names

    async def test_agent_never_offers_the_whole_catalogue(self):
        script = [
            _tool_call("request_more_tools", {}, "c1"),
            AIMessage(content="done"),
        ]
        agent = await build_agent(
            model=_Scripted(script=script),
            servers={},
            extra_tools=_catalogue(200),
            optimize_tools=ToolOptimizationConfig(level=SEMANTIC, semantic_top_k=5),
        )
        seen = _ToolsSeen()
        try:
            result = await agent.ainvoke(
                {"messages": _user("Rotate the API key of the billing service")},
                config={"callbacks": [seen]},
            )
        finally:
            await agent.shutdown()
        offered = seen.calls
        # + request_more_tools; at most one page (5) more, not all 200 tools.
        assert len(offered) == 2
        assert len(offered[0]) == 6
        assert 6 <= len(offered[1]) <= 11
        # The reply lists the page the next call is offered: tools not offered before.
        [reply] = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
        listed = {line[2:].split(":", 1)[0] for line in reply.splitlines() if line.startswith("- ")}
        assert len(listed) == 5
        assert listed <= set(offered[1]) and not listed & set(offered[0])


def _catalogue(n: int) -> list[BaseTool]:
    services = ["billing", "search", "auth", "storage", "email", "dns", "queue", "cache"]
    actions = ["Create", "Delete", "List", "Restart", "Rotate the API key of", "Back up"]

    async def run(target: str = "") -> str:
        return "ok"

    return [
        StructuredTool.from_function(
            coroutine=run,
            name=f"tool_{i:03d}",
            description=f"{actions[i % len(actions)]} the {services[i % len(services)]} "
            f"service in region {i // len(services)}.",
            args_schema=_IdArgs,
        )
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def big_index() -> ToolIndex:
    return ToolIndex(_catalogue(200))


class TestRequestMoreToolsTool:
    def test_args_schema(self, index):
        schema = convert_to_openai_tool(_RequestMoreToolsTool(tool_index=index))
        assert set(schema["function"]["parameters"]["properties"]) == {"query", "tool_names"}

    async def test_query_lists_matches(self, index):
        tool = _RequestMoreToolsTool(tool_index=index, top_k=2)
        result = await tool._arun(query="audit log")
        assert result.startswith(f"2 of {len(TOOL_DOCS)} tools are now available")
        assert "list_audit_events" in result

    async def test_unknown_names_are_reported(self, index):
        result = await _RequestMoreToolsTool(tool_index=index)._arun(
            tool_names=["rotate_api_key", "nope"]
        )
        assert "- rotate_api_key:" in result
        assert "No tool is named: nope" in result


# ======================================================================
# Agent: what each model call is offered
# ======================================================================


class TestAgentToolSelection:
    async def test_semantic_top_k_is_honored(self):
        for k in (2, 5):
            agent, _ = await _semantic_agent(
                ToolOptimizationConfig(level=SEMANTIC, semantic_top_k=k)
            )
            [first] = await _offered(agent, _user("Suspend user u_42, she left."))
            assert len(first) == k + 1  # + request_more_tools
            assert "request_more_tools" in first
            await agent.shutdown()

    async def test_preserved_tool_always_offered(self):
        agent, _ = await _semantic_agent(
            ToolOptimizationConfig(
                level=SEMANTIC, semantic_top_k=2, preserve_tools={"get_organization"}
            )
        )
        [first] = await _offered(agent, _user("Rotate API key key_8812."))
        assert "get_organization" in first and "rotate_api_key" in first
        await agent.shutdown()

    async def test_fallback_can_be_disabled(self):
        agent, _ = await _semantic_agent(
            ToolOptimizationConfig(level=SEMANTIC, always_include_fallback=False)
        )
        [first] = await _offered(agent, _user("Rotate API key key_8812."))
        assert "request_more_tools" not in first
        assert "request_more_tools" not in agent.tool_names
        await agent.shutdown()

    async def test_follow_up_keeps_the_tools_of_the_request(self):
        agent, _ = await _semantic_agent(ToolOptimizationConfig(level=SEMANTIC, semantic_top_k=3))
        [first] = await _offered(
            agent,
            [
                {"role": "user", "content": "The last deploy broke checkout. What can we do?"},
                {"role": "assistant", "content": "I can roll production back. Shall I?"},
                {"role": "user", "content": "Yes, go ahead."},
            ],
        )
        assert "rollback_deployment" in first
        await agent.shutdown()

    async def test_request_more_tools_round_trip(self):
        script = [
            _tool_call("request_more_tools", {"query": "audit log: who deleted what"}, "c1"),
            _tool_call("list_audit_events", {"target": "db_staging"}, "c2"),
            AIMessage(content="It was u_7."),
        ]
        agent, calls = await _semantic_agent(
            ToolOptimizationConfig(level=SEMANTIC, semantic_top_k=2), script
        )
        offered = await _offered(agent, _user("Rotate API key key_8812, then report."))
        assert "list_audit_events" not in offered[0]
        assert "list_audit_events" in offered[1]  # unlocked by the fallback
        assert "list_audit_events" in offered[2]  # still there after being called
        assert calls == ["list_audit_events"]
        await agent.shutdown()

    async def test_tool_left_out_this_step_still_executes(self):
        script = [_tool_call("create_dns_record", {"target": "www"}), AIMessage(content="ok")]
        agent, calls = await _semantic_agent(
            ToolOptimizationConfig(level=SEMANTIC, semantic_top_k=1), script
        )
        offered = await _offered(agent, _user("Suspend user u_42."))
        assert "create_dns_record" not in offered[0]
        assert calls == ["create_dns_record"]
        await agent.shutdown()

    async def test_streaming_is_narrowed_too(self):
        agent, _ = await _semantic_agent(ToolOptimizationConfig(level=SEMANTIC, semantic_top_k=2))
        seen = _ToolsSeen()
        async for _event in agent.astream_with_tools(
            {"messages": _user("Suspend user u_42.")}, config={"callbacks": [seen]}
        ):
            pass
        assert seen.calls and len(seen.calls[0]) == 3
        await agent.shutdown()

    async def test_no_optimization_offers_everything(self):
        calls: list[str] = []
        agent = await build_agent(model=_Scripted(), servers={}, extra_tools=_make_tools(calls))
        [first] = await _offered(agent, _user("Suspend user u_42."))
        assert set(first) == set(TOOL_DOCS)
        await agent.shutdown()


# ======================================================================
# Engine: per-step selector and rebinding
# ======================================================================


class TestEngineToolSelector:
    async def test_node_rebinds_when_selection_changes(self):
        tools = _make_tools([])
        steps: list[int] = []

        def selector(candidates: list[BaseTool], state: Any) -> list[BaseTool]:
            # Step 1 offers one tool, every later step another one.
            steps.append(len(steps))
            wanted = "suspend_user" if len(steps) == 1 else "invite_user"
            return [t for t in candidates if t.name == wanted]

        model = _Scripted(script=[_tool_call("suspend_user", {}), AIMessage(content="done")])
        graph = PromptGraph.react(tools=tools, system_prompt="sys")
        seen = _ToolsSeen()
        await PromptGraphEngine(graph=graph, model=model).ainvoke(
            {"messages": [HumanMessage(content="go")]},
            config={TOOL_SELECTOR_KEY: selector, "callbacks": [seen]},
        )
        assert seen.calls == [["suspend_user"], ["invite_user"]]

    async def test_failing_selector_offers_all(self):
        def selector(candidates: Any, state: Any) -> Any:
            raise RuntimeError("boom")

        graph = PromptGraph.react(tools=_make_tools([]), system_prompt="sys")
        seen = _ToolsSeen()
        await PromptGraphEngine(graph=graph, model=_Scripted()).ainvoke(
            {"messages": [HumanMessage(content="go")]},
            config={TOOL_SELECTOR_KEY: selector, "callbacks": [seen]},
        )
        assert len(seen.calls[0]) == len(TOOL_DOCS)


# ======================================================================
# MCP adapter: preserve_tools schemas and no-argument tools
# ======================================================================


class _FakeMulti:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = tools
        self.tool_to_server: dict[str, str] = {}
        self.called: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[Tool]:
        return self._tools

    async def call_tool(self, name: str, args: dict[str, Any], **_: Any) -> Any:
        self.called.append((name, args))
        return type("R", (), {"content": []})()


_SUSPEND_SCHEMA = {
    "type": "object",
    "properties": {
        "user_id": {"type": "string", "description": "The user to suspend."},
        "reason": {"type": "string", "description": "Why, recorded in the audit log."},
    },
    "required": ["user_id", "reason"],
}


def _params(tool: BaseTool) -> dict[str, Any]:
    return convert_to_openai_tool(tool)["function"]["parameters"]


class TestAdapter:
    async def test_preserved_tool_keeps_parameter_descriptions(self):
        multi = _FakeMulti(
            [
                Tool(name="suspend_user", description="Suspend.", inputSchema=_SUSPEND_SCHEMA),
                Tool(name="invite_user", description="Invite.", inputSchema=_SUSPEND_SCHEMA),
            ]
        )
        adapter = MCPToolAdapter(
            multi,  # type: ignore[arg-type]
            optimize=ToolOptimizationConfig(
                level=OptimizationLevel.MINIMAL, preserve_tools={"suspend_user"}
            ),
        )
        tools = {t.name: t for t in await adapter.as_langchain_tools()}
        kept = _params(tools["suspend_user"])["properties"]
        assert kept["user_id"]["description"] == "The user to suspend."
        stripped = _params(tools["invite_user"])["properties"]
        assert "description" not in stripped["user_id"]

    async def test_no_argument_tool_has_empty_schema(self):
        multi = _FakeMulti(
            [
                Tool(
                    name="get_organization",
                    description="Show the organization.",
                    inputSchema={"type": "object", "properties": {}},
                )
            ]
        )
        [tool] = await MCPToolAdapter(multi).as_langchain_tools()  # type: ignore[arg-type]
        assert json.dumps(_params(tool)) == '{"properties": {}, "type": "object"}'
        await tool.ainvoke({})
        assert multi.called == [("get_organization", {})]
