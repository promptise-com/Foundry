"""Regression tests for runtime hardening fixes.

Covers per-call budget enforcement, health monitoring signals, the
agent's final reply, journal wiring, journal replay, the restart policy,
stopping a process from its own worker, and event delivery on stop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from promptise.events import CallbackSink, EventNotifier
from promptise.runtime import BudgetExceededError
from promptise.runtime.budget import BudgetState
from promptise.runtime.config import (
    BudgetConfig,
    HealthConfig,
    JournalConfig,
    ProcessConfig,
    ToolCostAnnotation,
)
from promptise.runtime.health import AnomalyType, HealthMonitor
from promptise.runtime.journal import InMemoryJournal, JournalEntry, ReplayEngine
from promptise.runtime.lifecycle import ProcessLifecycle, ProcessState
from promptise.runtime.process import AgentProcess
from promptise.runtime.runtime import AgentRuntime
from promptise.runtime.triggers.base import TriggerEvent

BUILD_TARGET = "promptise.agent.build_agent"


async def _wait_for(cond: Callable[[], bool], timeout: float = 5.0) -> None:
    """Poll *cond* until it is true (fails the test after *timeout*)."""
    deadline = asyncio.get_running_loop().time() + timeout
    while not cond():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("condition not met in time")
        await asyncio.sleep(0.01)


def _event(n: int = 0) -> TriggerEvent:
    return TriggerEvent(trigger_id="t", trigger_type="manual", payload={"n": n})


def _tool_agent(
    the_tool: Any,
    *,
    calls: int = 1,
    reply: str = "done",
    args: Callable[[int], dict[str, Any]] | None = None,
) -> AsyncMock:
    """Mock agent that calls *the_tool* ``calls`` times, then replies.

    Tool errors become the tool result, as in the engine's tool loop.
    """

    async def ainvoke(inp: dict[str, Any], config: Any = None) -> dict[str, Any]:
        for i in range(calls):
            try:
                await the_tool.ainvoke(args(i) if args else {"text": str(i)}, config=config or {})
            except Exception:
                pass
        return {"messages": [*inp["messages"], AIMessage(content=reply)]}

    agent = AsyncMock()
    agent.ainvoke = AsyncMock(side_effect=ainvoke)
    agent.shutdown = AsyncMock()
    return agent


def _patch_build(agent: Any) -> Any:
    return patch(BUILD_TARGET, new_callable=lambda: AsyncMock(return_value=agent))


# =========================================================================
# Budget: limits are enforced before the tool runs
# =========================================================================


class TestBudgetEnforcement:
    async def test_enforced_call_over_limit_is_blocked_and_not_counted(self) -> None:
        state = BudgetState(BudgetConfig(enabled=True, max_tool_calls_per_run=2))
        assert await state.record_tool_call("a", enforce=True) is None
        assert await state.record_tool_call("a", enforce=True) is None
        violation = await state.record_tool_call("a", enforce=True)
        assert violation is not None
        assert violation.blocked is True
        assert violation.limit_name == "max_tool_calls_per_run"
        assert violation.current_value == 3  # the value the call would produce
        assert state.run_tool_calls == 2
        assert state.daily_tool_calls == 2

    async def test_cost_limit_blocks_the_call_that_would_exceed_it(self) -> None:
        state = BudgetState(
            BudgetConfig(
                enabled=True,
                max_cost_per_run=5.0,
                tool_costs={"pricey": ToolCostAnnotation(cost_weight=3.0)},
            )
        )
        assert await state.record_tool_call("pricey", enforce=True) is None
        violation = await state.record_tool_call("pricey", enforce=True)
        assert violation is not None and violation.blocked
        assert violation.limit_name == "max_cost_per_run"
        assert state.run_cost == 3.0

    def test_budget_exceeded_error_is_exported(self) -> None:
        from promptise import runtime

        assert "BudgetExceededError" in runtime.__all__

    async def test_process_never_runs_tool_calls_over_the_limit(self) -> None:
        ran: list[str] = []

        @tool
        def search(text: str) -> str:
            """Search."""
            ran.append(text)
            return "result"

        agent = _tool_agent(search, calls=4)
        config = ProcessConfig(
            budget=BudgetConfig(enabled=True, max_tool_calls_per_run=2, on_exceeded="pause")
        )
        with _patch_build(agent):
            process = AgentProcess("budgeted", config)
            await process.start()
            await process.inject(_event())
            await _wait_for(lambda: process.state == ProcessState.SUSPENDED)
            await process.stop()

        assert ran == ["0", "1"]
        assert process._budget is not None and process._budget.run_tool_calls == 2

    async def test_engine_returns_the_budget_error_as_the_tool_result(self) -> None:
        """Through the real tool loop: the run goes on, the blocked call never runs."""
        from unittest.mock import MagicMock

        from promptise.engine import PromptGraph, PromptGraphEngine, PromptNode
        from promptise.runtime.callbacks import RuntimeCallbackHandler

        ran: list[str] = []

        @tool
        def send_email(text: str) -> str:
            """Send an email (irreversible)."""
            ran.append(text)
            return "sent"

        calls = [{"name": "send_email", "args": {"text": f"m{i}"}, "id": f"c{i}"} for i in range(3)]
        model = MagicMock(spec=["ainvoke", "bind_tools", "with_structured_output"])
        model.ainvoke = AsyncMock(
            side_effect=[AIMessage(content="", tool_calls=calls), AIMessage(content="stopped")]
        )
        model.bind_tools = MagicMock(return_value=model)

        graph = PromptGraph("budget", mode="static")
        graph.add_node(PromptNode("act", instructions="Act.", tools=[send_email]))
        graph.set_entry("act")
        engine = PromptGraphEngine(graph=graph, model=model)

        state = BudgetState(
            BudgetConfig(
                enabled=True,
                max_irreversible_per_run=2,
                tool_costs={"send_email": ToolCostAnnotation(irreversible=True)},
            )
        )
        handler = RuntimeCallbackHandler(budget=state)
        result = await engine.ainvoke(
            {"messages": [HumanMessage(content="email everyone")]},
            config={"callbacks": handler.callbacks()},
        )

        assert sorted(ran) == ["m0", "m1"]
        tool_results = [m.content for m in result["messages"] if isinstance(m, ToolMessage)]
        assert sum("BudgetExceededError" in str(c) for c in tool_results) == 1
        assert handler.pending_violations[0].limit_name == "max_irreversible_per_run"

    async def test_budget_stop_from_the_worker_stops_the_process(self) -> None:
        @tool
        def act(text: str) -> str:
            """Act."""
            return "ok"

        agent = _tool_agent(act, calls=3)
        config = ProcessConfig(
            budget=BudgetConfig(enabled=True, max_tool_calls_per_run=1, on_exceeded="stop")
        )
        with _patch_build(agent):
            process = AgentProcess("stopper", config)
            await process.start()
            await process.inject(_event())
            await _wait_for(lambda: process.state == ProcessState.STOPPED)
            # Every worker task ended (the stopping worker included).
            await asyncio.sleep(0.05)
            assert process._worker_tasks == []

    async def test_blocked_tool_error_message_tells_agent_to_stop(self) -> None:
        state = BudgetState(BudgetConfig(enabled=True, max_tool_calls_per_run=1))
        await state.record_tool_call("x", enforce=True)
        violation = await state.record_tool_call("x", enforce=True)
        err = BudgetExceededError(violation)
        assert "max_tool_calls_per_run" in str(err)
        assert "was not executed" in str(err)
        assert err.violation is violation


# =========================================================================
# Health: signals that are not false positives
# =========================================================================


class TestHealthSignals:
    async def test_recovery_is_reported_once_per_incident(self) -> None:
        monitor = HealthMonitor(HealthConfig(enabled=True, stuck_threshold=2), "p")
        await monitor.record_tool_call("t", {"a": 1})
        assert await monitor.record_tool_call("t", {"a": 1}) is not None  # stuck
        assert await monitor.record_success() is True
        assert await monitor.record_success() is False
        assert await monitor.record_success() is False

    async def test_success_without_anomaly_is_not_a_recovery(self) -> None:
        monitor = HealthMonitor(HealthConfig(enabled=True), "p")
        assert await monitor.record_success() is False

    async def test_begin_invocation_forgets_previous_tool_calls(self) -> None:
        monitor = HealthMonitor(HealthConfig(enabled=True, stuck_threshold=2), "p")
        await monitor.record_tool_call("check_queue", {})
        monitor.begin_invocation()
        assert await monitor.record_tool_call("check_queue", {}) is None

    async def test_same_tool_once_per_run_is_not_stuck(self) -> None:
        @tool
        def check_queue(text: str) -> str:
            """Check the queue."""
            return "3 items"

        agent = _tool_agent(
            check_queue,
            calls=1,
            args=lambda i: {"text": "same"},
            reply="Queue checked: 3 items pending.",
        )
        config = ProcessConfig(health=HealthConfig(enabled=True, stuck_threshold=3))
        with _patch_build(agent):
            process = AgentProcess("poller", config)
            await process.start()
            for n in range(4):
                await process.inject(_event(n))
            await _wait_for(lambda: process._invocation_count == 4)
            await process.stop()

        assert process._health is not None
        assert process._health.anomalies == []

    async def test_empty_tool_output_is_not_an_empty_response(self) -> None:
        @tool
        def poll(text: str) -> str:
            """Poll."""
            return "[]"

        agent = _tool_agent(poll, calls=1, reply="Nothing new in the queue.")
        config = ProcessConfig(health=HealthConfig(enabled=True, empty_threshold=2))
        with _patch_build(agent):
            process = AgentProcess("quiet", config)
            await process.start()
            for n in range(3):
                await process.inject(_event(n))
            await _wait_for(lambda: process._invocation_count == 3)
            await process.stop()

        assert process._health is not None
        assert process._health.anomalies == []

    async def test_empty_final_replies_are_detected(self) -> None:
        agent = AsyncMock()
        agent.ainvoke = AsyncMock(
            side_effect=lambda inp, config=None: {
                "messages": [*inp["messages"], AIMessage(content="")]
            }
        )
        agent.shutdown = AsyncMock()
        config = ProcessConfig(health=HealthConfig(enabled=True, empty_threshold=2))
        with _patch_build(agent):
            process = AgentProcess("mute", config)
            await process.start()
            for n in range(2):
                await process.inject(_event(n))
            await _wait_for(lambda: process._invocation_count == 2)
            await process.stop()

        assert process._health is not None
        assert [a.anomaly_type for a in process._health.anomalies] == [AnomalyType.EMPTY_RESPONSE]


# =========================================================================
# Final reply: the conversation buffer stores this run's answer
# =========================================================================


class TestFinalReply:
    async def test_buffer_gets_last_reply_not_echoed_history(self) -> None:
        async def ainvoke(inp: dict[str, Any], config: Any = None) -> dict[str, Any]:
            return {
                "messages": [
                    SystemMessage(content="context"),
                    HumanMessage(content="earlier question"),
                    AIMessage(content="earlier answer"),
                    HumanMessage(content="trigger"),
                    AIMessage(
                        content="",
                        tool_calls=[{"name": "search", "args": {}, "id": "c1"}],
                    ),
                    ToolMessage(content="found", tool_call_id="c1"),
                    AIMessage(content="the final answer"),
                ]
            }

        agent = AsyncMock()
        agent.ainvoke = AsyncMock(side_effect=ainvoke)
        agent.shutdown = AsyncMock()
        with _patch_build(agent):
            process = AgentProcess("chat", ProcessConfig())
            await process.start()
            await process.inject(_event())
            await _wait_for(lambda: process._invocation_count == 1)
            snapshot = await process._conversation_buffer.async_snapshot()
            await process.stop()

        assert snapshot[-1] == {"role": "assistant", "content": "the final answer"}
        assert [m["role"] for m in snapshot] == ["user", "assistant"]


# =========================================================================
# Journal: ProcessConfig.journal is honoured
# =========================================================================


class TestProcessJournal:
    def test_journal_is_off_by_default(self) -> None:
        process = AgentProcess("plain", ProcessConfig())
        assert process._journal is None
        assert process.status()["journal_enabled"] is False

    async def test_checkpoint_level_records_lifecycle_results_and_checkpoints(self) -> None:
        agent = _tool_agent(None, calls=0, reply="handled")
        config = ProcessConfig(journal=JournalConfig(level="checkpoint", backend="memory"))
        with _patch_build(agent):
            process = AgentProcess("journaled", config)
            journal = process._journal
            assert isinstance(journal, InMemoryJournal)
            await process.start()
            await process.inject(_event())
            await _wait_for(lambda: process._invocation_count == 1)
            await process.stop()

        entries = await journal.read("journaled")
        types = [e.entry_type for e in entries]
        transitions = [
            (e.data["from_state"], e.data["to_state"])
            for e in entries
            if e.entry_type == "state_transition"
        ]
        assert transitions == [
            ("created", "starting"),
            ("starting", "running"),
            ("running", "stopping"),
            ("stopping", "stopped"),
        ]
        assert "invocation_result" in types
        assert "checkpoint" in types
        # Tool-level entries only at "full".
        assert "tool_call" not in types
        result = next(e for e in entries if e.entry_type == "invocation_result")
        assert result.data["response"] == "handled"

        recovered = await ReplayEngine(journal).recover("journaled")
        assert recovered["lifecycle_state"] == "stopped"

    async def test_full_level_records_triggers_and_tool_calls(self) -> None:
        @tool
        def lookup(text: str) -> str:
            """Look up."""
            return "value"

        agent = _tool_agent(lookup, calls=1)
        config = ProcessConfig(journal=JournalConfig(level="full", backend="memory"))
        with _patch_build(agent):
            process = AgentProcess("audited", config)
            journal = process._journal
            await process.start()
            await process.inject(_event(7))
            await _wait_for(lambda: process._invocation_count == 1)
            await process.stop()

        entries = await journal.read("audited")
        types = [e.entry_type for e in entries]
        for expected in ("trigger_event", "invocation_start", "tool_call", "invocation_result"):
            assert expected in types
        trigger = next(e for e in entries if e.entry_type == "trigger_event")
        assert trigger.data["payload"] == {"n": 7}
        call = next(e for e in entries if e.entry_type == "tool_call")
        assert call.data == {"tool": "lookup", "args": {"text": "0"}}

    async def test_failed_invocation_is_journaled(self) -> None:
        agent = AsyncMock()
        agent.ainvoke = AsyncMock(side_effect=RuntimeError("model down"))
        agent.shutdown = AsyncMock()
        config = ProcessConfig(
            journal=JournalConfig(level="checkpoint", backend="memory"),
            max_consecutive_failures=5,
        )
        with _patch_build(agent):
            process = AgentProcess("flaky", config)
            journal = process._journal
            await process.start()
            await process.inject(_event())
            await _wait_for(lambda: agent.ainvoke.await_count == 1)
            await asyncio.sleep(0.05)
            await process.stop()

        errors = await journal.read("flaky", entry_type="error")
        assert len(errors) == 1
        assert errors[0].data["error_type"] == "RuntimeError"


# =========================================================================
# Replay: only entries after the LAST checkpoint are replayed
# =========================================================================


class TestReplay:
    async def test_entries_before_the_latest_checkpoint_are_not_replayed(self) -> None:
        journal = InMemoryJournal()
        await journal.checkpoint("p", {"context_state": {"count": 1}, "lifecycle_state": "running"})
        await journal.append(
            JournalEntry(
                process_id="p", entry_type="context_update", data={"key": "count", "value": 2}
            )
        )
        # The process moved on (count=5) and checkpointed that state.
        await journal.checkpoint("p", {"context_state": {"count": 5}, "lifecycle_state": "running"})
        await journal.append(
            JournalEntry(
                process_id="p", entry_type="context_update", data={"key": "other", "value": "x"}
            )
        )

        recovered = await ReplayEngine(journal).recover("p")

        assert recovered["context_state"] == {"count": 5, "other": "x"}
        assert recovered["entries_replayed"] == 1

    async def test_recover_does_not_mutate_the_stored_checkpoint(self) -> None:
        journal = InMemoryJournal()
        await journal.checkpoint("p", {"context_state": {"a": 1}, "lifecycle_state": "running"})
        await journal.append(
            JournalEntry(process_id="p", entry_type="context_update", data={"key": "b", "value": 2})
        )

        await ReplayEngine(journal).recover("p")

        stored = await journal.last_checkpoint("p")
        assert stored is not None and stored["context_state"] == {"a": 1}


# =========================================================================
# Lifecycle listeners
# =========================================================================


class TestLifecycleListeners:
    async def test_listeners_run_after_each_transition_and_errors_are_isolated(self) -> None:
        lifecycle = ProcessLifecycle()
        seen: list[tuple[str, str]] = []

        def broken(_t: Any) -> None:
            raise RuntimeError("listener bug")

        async def record(t: Any) -> None:
            seen.append((t.from_state.value, t.to_state.value))

        lifecycle.add_listener(broken)
        lifecycle.add_listener(record)
        await lifecycle.transition(ProcessState.STARTING)
        await lifecycle.transition(ProcessState.RUNNING)

        assert lifecycle.state == ProcessState.RUNNING
        assert seen == [("created", "starting"), ("starting", "running")]


# =========================================================================
# Restart policy
# =========================================================================


def _failing_agent() -> AsyncMock:
    agent = AsyncMock()
    agent.ainvoke = AsyncMock(side_effect=RuntimeError("boom"))
    agent.shutdown = AsyncMock()
    return agent


class TestRestartPolicy:
    async def test_on_failure_restarts_a_failed_process(self) -> None:
        agent = _failing_agent()
        config = ProcessConfig(
            restart_policy="on_failure",
            max_restarts=3,
            restart_backoff=0,
            max_consecutive_failures=1,
        )
        with _patch_build(agent) as build:
            process = AgentProcess("phoenix", config)
            await process.start()
            await process.inject(_event())
            await _wait_for(
                lambda: build.await_count == 2 and process.state == ProcessState.RUNNING
            )
            assert process.status()["restart_count"] == 1
            assert process._consecutive_failures == 0
            await process.stop()
        assert process.state == ProcessState.STOPPED

    async def test_never_policy_leaves_the_process_failed(self) -> None:
        agent = _failing_agent()
        config = ProcessConfig(max_consecutive_failures=1, restart_backoff=0)
        with _patch_build(agent) as build:
            process = AgentProcess("once", config)
            await process.start()
            await process.inject(_event())
            await _wait_for(lambda: process.state == ProcessState.FAILED)
            await asyncio.sleep(0.05)
            assert build.await_count == 1
            await process.stop()

    async def test_restarts_stop_at_max_restarts(self) -> None:
        agent = _failing_agent()
        build = AsyncMock(side_effect=[agent, RuntimeError("no model"), RuntimeError("no model")])
        config = ProcessConfig(
            restart_policy="on_failure",
            max_restarts=2,
            restart_backoff=0,
            max_consecutive_failures=1,
        )
        with patch(BUILD_TARGET, new=build):
            process = AgentProcess("doomed", config)
            await process.start()
            await process.inject(_event())
            await _wait_for(lambda: build.await_count == 3 and process.state == ProcessState.FAILED)
            await asyncio.sleep(0.05)
            assert build.await_count == 3
            assert process.status()["restart_count"] == 2
            await process.stop()

    async def test_stop_cancels_a_pending_restart(self) -> None:
        agent = _failing_agent()
        config = ProcessConfig(
            restart_policy="on_failure",
            restart_backoff=30,
            max_consecutive_failures=1,
        )
        with _patch_build(agent) as build:
            process = AgentProcess("halted", config)
            await process.start()
            await process.inject(_event())
            await _wait_for(lambda: process._restart_task is not None)
            await process.stop()
            assert process._restart_task is None
            await asyncio.sleep(0.05)
            assert build.await_count == 1

    async def test_failed_start_is_not_retried(self) -> None:
        build = AsyncMock(side_effect=RuntimeError("bad config"))
        config = ProcessConfig(restart_policy="always", restart_backoff=0)
        with patch(BUILD_TARGET, new=build):
            process = AgentProcess("broken", config)
            with pytest.raises(RuntimeError, match="bad config"):
                await process.start()
            await asyncio.sleep(0.05)
            assert build.await_count == 1
            assert process._restart_task is None

    async def test_always_recycles_at_max_lifetime(self) -> None:
        agent = _tool_agent(None, calls=0)
        config = ProcessConfig(
            restart_policy="always",
            heartbeat_interval=0.02,
            max_lifetime=0.05,
        )
        with _patch_build(agent) as build:
            process = AgentProcess("recycled", config)
            await process.start()
            await _wait_for(
                lambda: build.await_count >= 2 and process.state == ProcessState.RUNNING
            )
            await process.stop()
        assert process.state == ProcessState.STOPPED
        reasons = [t.reason for t in process.lifecycle.history]
        assert "stopped for restart" in reasons

    async def test_process_failed_event_on_runtime_failure(self) -> None:
        received: list[str] = []
        notifier = EventNotifier([CallbackSink(lambda e: received.append(e.event_type))])
        await notifier.start()
        agent = _failing_agent()
        config = ProcessConfig(max_consecutive_failures=1)
        with _patch_build(agent):
            process = AgentProcess("crashy", config, event_notifier=notifier)
            await process.start()
            await process.inject(_event())
            await _wait_for(lambda: process.state == ProcessState.FAILED)
            await process.stop()
        await notifier.stop()
        assert "process.failed" in received


# =========================================================================
# Events: process.stopped is delivered
# =========================================================================


def _agent_owning(notifier: EventNotifier) -> AsyncMock:
    """Mock agent whose shutdown stops its notifier, like PromptiseAgent."""
    agent = _tool_agent(None, calls=0)
    agent._event_notifier = notifier

    async def shutdown() -> None:
        if agent._event_notifier is not None:
            await agent._event_notifier.stop()

    agent.shutdown = AsyncMock(side_effect=shutdown)
    return agent


class TestStopEvents:
    async def test_standalone_process_delivers_process_stopped(self) -> None:
        received: list[str] = []
        notifier = EventNotifier([CallbackSink(lambda e: received.append(e.event_type))])
        await notifier.start()
        with _patch_build(_agent_owning(notifier)):
            process = AgentProcess("solo", ProcessConfig(), event_notifier=notifier)
            await process.start()
            await process.stop()
        assert received == ["process.started", "process.stopped"]

    async def test_runtime_delivers_every_process_stopped(self) -> None:
        received: list[tuple[str, str]] = []
        notifier = EventNotifier(
            [
                CallbackSink(
                    lambda e: received.append((e.event_type, e.data.get("process_name", ""))),
                    events=["process.stopped"],
                )
            ]
        )
        await notifier.start()
        with _patch_build(_agent_owning(notifier)):
            runtime = AgentRuntime(event_notifier=notifier)
            await runtime.add_process("a", ProcessConfig())
            await runtime.add_process("b", ProcessConfig())
            await runtime.start_all()
            await runtime.stop_all()
        assert sorted(received) == [("process.stopped", "a"), ("process.stopped", "b")]
