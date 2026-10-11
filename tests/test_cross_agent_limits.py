"""Cross-agent delegation limits, timeouts, tracing and the opt-in broadcast tool."""

from __future__ import annotations

import asyncio
from fnmatch import fnmatch
from typing import Any

import pytest
from _scripted_model import Scripted, calls
from langchain_core.runnables import RunnableLambda

from promptise import build_agent
from promptise.cross_agent import (
    CrossAgent,
    DelegationError,
    get_delegation_chain,
    make_cross_agent_tools,
)


def _reply(text: str) -> dict[str, Any]:
    return {"messages": [{"role": "assistant", "content": text}]}


def _tool(tools: list, name: str):
    return next(t for t in tools if t.name == name)


# ---------------------------------------------------------------------------
# Depth limit and loop detection
# ---------------------------------------------------------------------------


def _chain_of(n: int, max_depth: int | None) -> tuple[Any, list[tuple[str, ...]]]:
    """Agents a0 → a1 → … → a(n-1); each asks the next. Returns (ask tool of a0, chains seen)."""
    seen: list[tuple[str, ...]] = []
    tools: dict[int, Any] = {}

    def make(i: int) -> Any:
        async def run(payload: dict[str, Any]) -> dict[str, Any]:
            seen.append(get_delegation_chain())
            if i + 1 < n:
                try:
                    text = await tools[i + 1].ainvoke({"message": "deeper"})
                except DelegationError as exc:
                    text = f"refused: {exc}"
                return _reply(text)
            return _reply(f"leaf {i}")

        return RunnableLambda(run)

    peers = [make(i) for i in range(n)]
    limit = {} if max_depth is None else {"max_delegation_depth": max_depth}
    for i in range(n):
        tools[i] = make_cross_agent_tools(
            {f"a{i}": CrossAgent(agent=peers[i])}, include_broadcast=False, **limit
        )[0]
    return tools[0], seen


async def test_delegation_within_the_limit_reaches_the_leaf() -> None:
    ask, seen = _chain_of(3, max_depth=3)
    assert await ask.ainvoke({"message": "go"}) == "leaf 2"
    assert seen == [("a0",), ("a0", "a1"), ("a0", "a1", "a2")]
    assert get_delegation_chain() == ()  # reset after the call


async def test_delegation_past_the_limit_is_refused_with_a_clear_error() -> None:
    ask, seen = _chain_of(5, max_depth=3)
    result = await ask.ainvoke({"message": "go"})
    # a0, a1, a2 ran; a3 would be level 4.
    assert len(seen) == 3
    assert result.startswith("refused: Delegation to 'a3' refused")
    assert "a0 → a1 → a2 → a3" in result
    assert "max_delegation_depth=3" in result


async def test_default_limit_is_three() -> None:
    ask, seen = _chain_of(6, max_depth=None)
    result = await ask.ainvoke({"message": "go"})
    assert len(seen) == 3
    assert "max_delegation_depth=3" in result


async def test_delegating_back_into_a_busy_peer_is_a_loop() -> None:
    holder: dict[str, Any] = {}

    async def helper(payload: dict[str, Any]) -> dict[str, Any]:
        # The helper asks itself again (through the same tool).
        try:
            return _reply(await holder["ask"].ainvoke({"message": "again"}))
        except DelegationError as exc:
            return _reply(f"refused: {exc}")

    holder["ask"] = make_cross_agent_tools(
        {"helper": CrossAgent(agent=RunnableLambda(helper))}, include_broadcast=False
    )[0]
    result = await holder["ask"].ainvoke({"message": "go"})
    assert "'helper' is already working on this request (helper → helper)" in result


def test_invalid_limits_are_rejected() -> None:
    peer = {"p": CrossAgent(agent=RunnableLambda(lambda p: p))}
    with pytest.raises(ValueError, match="max_delegation_depth"):
        make_cross_agent_tools(peer, max_delegation_depth=0)
    with pytest.raises(ValueError, match="timeout"):
        make_cross_agent_tools(peer, timeout=0)
    with pytest.raises(ValueError, match="CrossAgent 'q'"):
        make_cross_agent_tools({"q": CrossAgent(agent=RunnableLambda(lambda p: p), timeout=-1)})


async def test_build_agent_rejects_a_bad_limit_before_connecting() -> None:
    with pytest.raises(ValueError, match="max_delegation_depth"):
        await build_agent(model=Scripted(), servers={}, max_delegation_depth=0)
    with pytest.raises(ValueError, match="delegation_timeout"):
        await build_agent(model=Scripted(), servers={}, delegation_timeout=-5)


async def test_self_delegating_agent_stops_quickly() -> None:
    """The guide16 probe: an agent that always delegates, reaching itself through a peer."""
    agents: dict[str, Any] = {}

    async def back_into_a(payload: dict[str, Any]) -> Any:
        return await agents["a"].ainvoke(payload)

    agents["a"] = await build_agent(
        model=Scripted(mode="delegate"),
        servers={},
        cross_agents={"helper": CrossAgent(agent=RunnableLambda(back_into_a))},
        max_agent_iterations=3,
    )
    before = calls["n"]
    try:
        await agents["a"].ainvoke({"messages": [{"role": "user", "content": "What is 2 + 2?"}]})
    except Exception:
        pass  # the scripted model never answers; only the number of calls matters
    finally:
        await agents["a"].shutdown()
    # Without the limit this fanned out to well over a thousand model calls.
    assert calls["n"] - before <= 20


# ---------------------------------------------------------------------------
# Timeouts are set in Python, not by the model
# ---------------------------------------------------------------------------


async def _slow(payload: dict[str, Any]) -> dict[str, Any]:
    await asyncio.sleep(30)
    return _reply("too late")


def test_the_model_cannot_choose_a_timeout() -> None:
    tools = make_cross_agent_tools({"p": CrossAgent(agent=RunnableLambda(_slow))})
    for tool in tools:
        assert "timeout_s" not in tool.args


async def test_per_peer_timeout() -> None:
    ask = make_cross_agent_tools(
        {"slow": CrossAgent(agent=RunnableLambda(_slow), timeout=0.05)}, include_broadcast=False
    )[0]
    assert await ask.ainvoke({"message": "hi"}) == "Timed out waiting for peer agent reply."


async def test_default_timeout_and_peer_override() -> None:
    async def quick(payload: dict[str, Any]) -> dict[str, Any]:
        await asyncio.sleep(0.1)
        return _reply("quick")

    tools = make_cross_agent_tools(
        {
            "slow": CrossAgent(agent=RunnableLambda(_slow)),
            "quick": CrossAgent(agent=RunnableLambda(quick), timeout=5),
        },
        timeout=0.05,
    )
    assert await _tool(tools, "ask_agent_slow").ainvoke({"message": "hi"}) == (
        "Timed out waiting for peer agent reply."
    )
    assert await _tool(tools, "ask_agent_quick").ainvoke({"message": "hi"}) == "quick"
    result = await _tool(tools, "broadcast_to_agents").ainvoke({"message": "hi"})
    assert result == {"slow": "Timed out", "quick": "quick"}


async def test_build_agent_passes_delegation_timeout() -> None:
    agent = await build_agent(
        model=Scripted(mode="tool:ask_agent_slow"),
        servers={},
        cross_agents={"slow": CrossAgent(agent=RunnableLambda(_slow))},
        delegation_timeout=0.05,
    )
    try:
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "hi"}]})
    finally:
        await agent.shutdown()
    assert result["messages"][-1].content == "ANSWER: Timed out waiting for peer agent reply."


# ---------------------------------------------------------------------------
# Broadcast: opt-in, and it forwards context
# ---------------------------------------------------------------------------


async def test_build_agent_adds_no_broadcast_tool_by_default() -> None:
    peer = {"p": CrossAgent(agent=RunnableLambda(lambda p: _reply("x")))}
    plain = await build_agent(model=Scripted(), servers={}, cross_agents=peer)
    with_broadcast = await build_agent(
        model=Scripted(), servers={}, cross_agents=peer, include_broadcast=True
    )
    try:
        assert [t.name for t in plain._tools] == ["ask_agent_p"]
        assert {t.name for t in with_broadcast._tools} == {"ask_agent_p", "broadcast_to_agents"}
    finally:
        await plain.shutdown()
        await with_broadcast.shutdown()


async def test_broadcast_forwards_context() -> None:
    received: list[list[dict[str, Any]]] = []

    async def peer(payload: dict[str, Any]) -> dict[str, Any]:
        received.append(payload["messages"])
        return _reply("ok")

    tools = make_cross_agent_tools(
        {"a": CrossAgent(agent=RunnableLambda(peer)), "b": CrossAgent(agent=RunnableLambda(peer))}
    )
    bcast = _tool(tools, "broadcast_to_agents")
    assert "context" in bcast.args
    await bcast.ainvoke({"message": "Review this", "context": "EU customers only"})
    assert len(received) == 2
    for messages in received:
        assert messages[0] == {"role": "system", "content": "Caller context: EU customers only"}
        assert messages[-1] == {"role": "user", "content": "Review this"}


async def test_broadcast_reports_a_refused_peer_per_peer() -> None:
    holder: dict[str, Any] = {}

    async def a(payload: dict[str, Any]) -> dict[str, Any]:
        # a broadcasts to itself and b: a is refused (loop), b answers.
        return _reply(str(await holder["bcast"].ainvoke({"message": "again"})))

    tools = make_cross_agent_tools(
        {
            "a": CrossAgent(agent=RunnableLambda(a)),
            "b": CrossAgent(agent=RunnableLambda(lambda p: _reply("b says hi"))),
        }
    )
    holder["bcast"] = _tool(tools, "broadcast_to_agents")
    result = await _tool(tools, "ask_agent_a").ainvoke({"message": "go"})
    assert "b says hi" in result
    assert "Error: Delegation to 'a' refused" in result


# ---------------------------------------------------------------------------
# trace_tools shows delegation
# ---------------------------------------------------------------------------


async def test_trace_tools_prints_ask_agent_calls(capsys: pytest.CaptureFixture[str]) -> None:
    agent = await build_agent(
        model=Scripted(mode="tool:ask_agent_billing"),
        servers={},
        trace_tools=True,
        cross_agents={
            "billing": CrossAgent(agent=RunnableLambda(lambda p: _reply("Team plan, 12 seats")))
        },
    )
    try:
        await agent.ainvoke({"messages": [{"role": "user", "content": "acme?"}]})
    finally:
        await agent.shutdown()
    out = capsys.readouterr().out
    assert "→ Invoking tool: ask_agent_billing with {'message': 'What plan is acme on?'}" in out
    assert "✔ Tool result from ask_agent_billing: Team plan, 12 seats" in out


async def test_trace_tools_prints_each_delegation_once(capsys: pytest.CaptureFixture[str]) -> None:
    runs = {"n": 0}

    def peer(payload: dict[str, Any]) -> dict[str, Any]:
        runs["n"] += 1
        return _reply("Team plan, 12 seats")

    agent = await build_agent(
        model=Scripted(mode="tool:ask_agent_billing"),
        servers={},
        trace_tools=True,
        cross_agents={"billing": CrossAgent(agent=RunnableLambda(peer))},
    )
    try:
        await agent.ainvoke({"messages": [{"role": "user", "content": "acme?"}]})
    finally:
        await agent.shutdown()
    out = capsys.readouterr().out
    assert runs["n"] == 1
    assert out.count("→ Invoking tool: ask_agent_billing") == 1
    assert out.count("✔ Tool result from ask_agent_billing") == 1
    # The delegation's own arguments are kept (no LangChain-filled defaults).
    assert "→ Invoking tool: ask_agent_billing with {'message': 'What plan is acme on?'}" in out


async def test_trace_tools_prints_a_broadcast_once(capsys: pytest.CaptureFixture[str]) -> None:
    agent = await build_agent(
        model=Scripted(mode="tool:broadcast_to_agents", args={"message": "status?"}),
        servers={},
        trace_tools=True,
        include_broadcast=True,
        cross_agents={
            "a": CrossAgent(agent=RunnableLambda(lambda p: _reply("a ok"))),
            "b": CrossAgent(agent=RunnableLambda(lambda p: _reply("b ok"))),
        },
    )
    try:
        await agent.ainvoke({"messages": [{"role": "user", "content": "status"}]})
    finally:
        await agent.shutdown()
    out = capsys.readouterr().out
    assert out.count("→ Invoking tool: broadcast_to_agents") == 1
    assert out.count("✔ Tool result from broadcast_to_agents") == 1
    assert "'a': 'a ok'" in out and "'b': 'b ok'" in out


async def test_observer_records_each_delegation_once() -> None:
    from promptise.observability import ObservabilityCollector, TimelineEventType

    collector = ObservabilityCollector("t")
    agent = await build_agent(
        model=Scripted(mode="tool:ask_agent_billing"),
        servers={},
        observer=collector,
        cross_agents={"billing": CrossAgent(agent=RunnableLambda(lambda p: _reply("Team")))},
    )
    try:
        await agent.ainvoke({"messages": [{"role": "user", "content": "acme?"}]})
    finally:
        await agent.shutdown()

    def _count(kind: TimelineEventType) -> int:
        return sum(
            1
            for e in collector.get_timeline()
            if e.event_type == kind and (e.metadata or {}).get("tool_name") == "ask_agent_billing"
        )

    assert _count(TimelineEventType.TOOL_CALL) == 1
    assert _count(TimelineEventType.TOOL_RESULT) == 1


async def test_hooks_report_errors() -> None:
    events: list[tuple[str, str]] = []

    async def broken(payload: dict[str, Any]) -> dict[str, Any]:
        raise ConnectionError("billing database unreachable")

    ask = make_cross_agent_tools(
        {"billing": CrossAgent(agent=RunnableLambda(broken))},
        include_broadcast=False,
        on_before=lambda name, args: events.append(("before", name)),
        on_after=lambda name, res: events.append(("after", name)),
        on_error=lambda name, exc: events.append(("error", f"{name}: {exc}")),
    )[0]
    with pytest.raises(ConnectionError):
        await ask.ainvoke({"message": "hi"})
    assert events == [
        ("before", "ask_agent_billing"),
        ("error", "ask_agent_billing: billing database unreachable"),
    ]


# ---------------------------------------------------------------------------
# Approval: a broadcast must not reach a peer whose ask tool needs approval
# ---------------------------------------------------------------------------


async def test_broadcast_does_not_bypass_approval_of_an_ask_tool() -> None:
    from promptise.approval import ApprovalPolicy

    reached: list[str] = []
    asked: list[str] = []

    def peer(name: str) -> CrossAgent:
        async def run(payload: dict[str, Any]) -> dict[str, Any]:
            reached.append(name)
            return _reply(f"{name} done")

        return CrossAgent(agent=RunnableLambda(run))

    async def reviewer(request: Any) -> bool:
        asked.append(request.tool_name)
        return False

    agent = await build_agent(
        model=Scripted(mode="tool:broadcast_to_agents", args={"message": "Refund order 42"}),
        servers={},
        cross_agents={"payments": peer("payments"), "notes": peer("notes")},
        include_broadcast=True,
        approval=ApprovalPolicy(tools=["ask_agent_payments"], handler=reviewer),
    )
    try:
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "refund"}]})
    finally:
        await agent.shutdown()
    answer = result["messages"][-1].content
    # Before the fix the broadcast called the payments agent with no approval.
    assert reached == ["notes"]
    assert asked == []
    assert "'payments' needs approval" in answer and "Use ask_agent_payments" in answer
    assert "notes done" in answer


async def test_broadcast_that_needs_approval_itself_reaches_every_peer() -> None:
    reached: list[str] = []

    def peer(name: str) -> CrossAgent:
        return CrossAgent(agent=RunnableLambda(lambda p: reached.append(name) or _reply(name)))

    patterns = ["ask_agent_payments", "broadcast_*"]
    tools = make_cross_agent_tools(
        {"payments": peer("payments"), "notes": peer("notes")},
        requires_approval=lambda name: any(fnmatch(name, p) for p in patterns),
    )
    # The approval gate sits on the broadcast tool itself, so it covers every peer.
    result = await _tool(tools, "broadcast_to_agents").ainvoke({"message": "hi"})
    assert result == {"payments": "payments", "notes": "notes"}


# ---------------------------------------------------------------------------
# Identity: a nested hop never reports an outer agent as the delegator
# ---------------------------------------------------------------------------


async def test_nested_hop_without_identity_does_not_inherit_the_outer_delegator() -> None:
    from promptise.identity import AgentIdentity
    from promptise.observability import get_current_delegation

    seen: dict[str, Any] = {}

    async def leaf(payload: dict[str, Any]) -> dict[str, Any]:
        seen["leaf"] = get_current_delegation()
        return _reply("leaf")

    # middle has no identity of its own; it asks the leaf.
    middle_ask = make_cross_agent_tools(
        {"leaf": CrossAgent(agent=RunnableLambda(leaf))}, include_broadcast=False
    )[0]

    async def middle(payload: dict[str, Any]) -> dict[str, Any]:
        seen["middle"] = get_current_delegation()
        return _reply(await middle_ask.ainvoke({"message": "deeper"}))

    top_ask = make_cross_agent_tools(
        {"middle": CrossAgent(agent=RunnableLambda(middle))},
        include_broadcast=False,
        caller_identity=AgentIdentity("coordinator-bot"),
    )[0]
    assert await top_ask.ainvoke({"message": "go"}) == "leaf"
    assert seen["middle"]["agent_id"] == "coordinator-bot"
    # Before the fix the leaf was told coordinator-bot delegated to it.
    assert seen["leaf"] is None
    assert get_current_delegation() is None
