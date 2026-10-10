"""Tests for promptise.runtime.callbacks — RuntimeCallbackHandler."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from promptise.runtime.budget import BudgetViolation
from promptise.runtime.callbacks import BudgetGuard, RuntimeCallbackHandler
from promptise.runtime.exceptions import BudgetExceededError


class TestInit:
    def test_creation_with_both(self) -> None:
        budget = MagicMock()
        health = MagicMock()
        handler = RuntimeCallbackHandler(budget=budget, health=health)
        assert handler._budget is budget
        assert handler._health is health
        assert handler.pending_violations == []

    def test_creation_with_neither(self) -> None:
        handler = RuntimeCallbackHandler()
        assert handler._budget is None
        assert handler._health is None

    def test_creation_budget_only(self) -> None:
        budget = MagicMock()
        handler = RuntimeCallbackHandler(budget=budget)
        assert handler._budget is budget
        assert handler._health is None


class TestReset:
    def test_clears_pending_violations(self) -> None:
        handler = RuntimeCallbackHandler()
        handler.pending_violations.append(MagicMock())
        handler.pending_violations.append(MagicMock())
        assert len(handler.pending_violations) == 2
        handler.reset()
        assert handler.pending_violations == []


class TestBudgetGuard:
    """Per-call budget enforcement (runs before the tool executes)."""

    def _guard(self, side_effect: object) -> tuple[RuntimeCallbackHandler, MagicMock]:
        budget = MagicMock()
        if isinstance(side_effect, list):
            budget.record_tool_call_sync = MagicMock(side_effect=side_effect)
        else:
            budget.record_tool_call_sync = MagicMock(return_value=side_effect)
        return RuntimeCallbackHandler(budget=budget), budget

    def test_guard_is_sync_inline_and_raises(self) -> None:
        # Sync + inline + raise_error is what makes LangChain propagate the
        # error before the tool body runs, for sync and async tools alike.
        handler, _ = self._guard(None)
        guard = handler.budget_guard
        assert isinstance(guard, BudgetGuard)
        assert guard.raise_error is True and guard.run_inline is True
        assert not asyncio.iscoroutinefunction(guard.on_tool_start)
        assert handler.callbacks() == [guard, handler]

    def test_no_guard_without_budget(self) -> None:
        handler = RuntimeCallbackHandler()
        assert handler.budget_guard is None
        assert handler.callbacks() == [handler]

    def test_records_tool_call_with_enforcement(self) -> None:
        handler, budget = self._guard(None)
        handler.budget_guard.on_tool_start({"name": "search"}, "{}")
        budget.record_tool_call_sync.assert_called_once_with("search", enforce=True)

    def test_tool_name_from_id_fallback_and_non_dict(self) -> None:
        handler, budget = self._guard(None)
        handler.budget_guard.on_tool_start({"id": "fallback_name"}, "{}")
        handler.budget_guard.on_tool_start("not_a_dict", "{}")
        assert [c.args[0] for c in budget.record_tool_call_sync.call_args_list] == [
            "fallback_name",
            "",
        ]

    def test_collects_unblocked_violation(self) -> None:
        violation = BudgetViolation("max_cost_per_day", 10, 11, "tool")
        handler, _ = self._guard(violation)
        handler.budget_guard.on_tool_start({"name": "tool"}, "{}")
        assert handler.pending_violations == [violation]

    def test_blocked_violation_raises(self) -> None:
        violation = BudgetViolation("max_irreversible_per_run", 1, 2, "post", blocked=True)
        handler, _ = self._guard(violation)
        with pytest.raises(BudgetExceededError, match="max_irreversible_per_run"):
            handler.budget_guard.on_tool_start({"name": "post"}, "{}")
        assert handler.pending_violations == [violation]

    def test_same_limit_collected_once(self) -> None:
        handler, _ = self._guard(
            [
                BudgetViolation("max_tool_calls_per_run", 2, 3, "a", blocked=True),
                BudgetViolation("max_tool_calls_per_run", 2, 3, "b", blocked=True),
            ]
        )
        for name in ("a", "b"):
            with pytest.raises(BudgetExceededError):
                handler.budget_guard.on_tool_start({"name": name}, "{}")
        assert len(handler.pending_violations) == 1

    def test_budget_error_logged_not_raised(self) -> None:
        handler, budget = self._guard(None)
        budget.record_tool_call_sync.side_effect = RuntimeError("boom")
        handler.budget_guard.on_tool_start({"name": "tool"}, "{}")  # no raise

    def test_reset_clears_guard_violations(self) -> None:
        handler, _ = self._guard(BudgetViolation("max_cost_per_day", 1, 2, "t"))
        handler.budget_guard.on_tool_start({"name": "t"}, "{}")
        handler.reset()
        assert handler.pending_violations == []

    @pytest.mark.asyncio
    async def test_blocks_real_sync_and_async_tools(self) -> None:
        """End to end through LangChain: the tool body never runs."""
        from langchain_core.tools import tool

        from promptise.runtime.budget import BudgetState
        from promptise.runtime.config import BudgetConfig, ToolCostAnnotation

        ran: list[str] = []

        @tool
        def post_sync(text: str) -> str:
            """Irreversible (sync)."""
            ran.append(f"sync:{text}")
            return "ok"

        @tool
        async def post_async(text: str) -> str:
            """Irreversible (async)."""
            ran.append(f"async:{text}")
            return "ok"

        state = BudgetState(
            BudgetConfig(
                enabled=True,
                max_irreversible_per_run=1,
                tool_costs={
                    "post_sync": ToolCostAnnotation(irreversible=True),
                    "post_async": ToolCostAnnotation(irreversible=True),
                },
            )
        )
        handler = RuntimeCallbackHandler(budget=state)
        cfg = {"callbacks": handler.callbacks()}
        assert await post_sync.ainvoke({"text": "1"}, config=cfg) == "ok"
        with pytest.raises(BudgetExceededError):
            await post_sync.ainvoke({"text": "2"}, config=cfg)
        with pytest.raises(BudgetExceededError):
            await post_async.ainvoke({"text": "3"}, config=cfg)
        assert ran == ["sync:1"]
        assert state.run_irreversible == 1  # blocked calls are not counted


class TestOnToolStart:
    @pytest.mark.asyncio
    async def test_records_health_tool_call(self) -> None:
        health = AsyncMock()
        health.record_tool_call = AsyncMock(return_value=None)
        handler = RuntimeCallbackHandler(health=health)

        await handler.on_tool_start({"name": "search"}, '{"q": "test"}')

        health.record_tool_call.assert_awaited_once()
        call_args = health.record_tool_call.call_args
        assert call_args[0][0] == "search"
        assert call_args[0][1] == {"q": "test"}

    @pytest.mark.asyncio
    async def test_journal_records_tool_call_and_result(self) -> None:
        records: list[tuple[str, dict]] = []

        async def journal(entry_type: str, data: dict) -> None:
            records.append((entry_type, data))

        handler = RuntimeCallbackHandler(journal=journal)
        await handler.on_tool_start({"name": "search"}, '{"q": "x"}')
        await handler.on_tool_end("found it", name="search")

        assert records == [
            ("tool_call", {"tool": "search", "args": {"q": "x"}}),
            ("tool_result", {"tool": "search", "result": "found it"}),
        ]

    @pytest.mark.asyncio
    async def test_health_error_logged_not_raised(self) -> None:
        health = AsyncMock()
        health.record_tool_call = AsyncMock(side_effect=RuntimeError("boom"))
        handler = RuntimeCallbackHandler(health=health)

        await handler.on_tool_start({"name": "tool"}, "{}")

    @pytest.mark.asyncio
    async def test_skipped_when_no_budget_no_health(self) -> None:
        handler = RuntimeCallbackHandler()
        # Should complete without error
        await handler.on_tool_start({"name": "tool"}, "{}")

    @pytest.mark.asyncio
    async def test_parses_dict_input(self) -> None:
        health = AsyncMock()
        health.record_tool_call = AsyncMock(return_value=None)
        handler = RuntimeCallbackHandler(health=health)

        await handler.on_tool_start({"name": "t"}, {"already": "parsed"})

        call_args = health.record_tool_call.call_args
        assert call_args[0][1] == {"already": "parsed"}

    @pytest.mark.asyncio
    async def test_parses_invalid_json_input(self) -> None:
        health = AsyncMock()
        health.record_tool_call = AsyncMock(return_value=None)
        handler = RuntimeCallbackHandler(health=health)

        await handler.on_tool_start({"name": "t"}, "not json at all")

        call_args = health.record_tool_call.call_args
        assert call_args[0][1] == {"raw": "not json at all"}


class TestOnToolEnd:
    @pytest.mark.asyncio
    async def test_tool_output_is_not_an_agent_response(self) -> None:
        # A quiet queue returning "[]" must not count toward empty_response.
        health = AsyncMock()
        handler = RuntimeCallbackHandler(health=health)

        await handler.on_tool_end("[]")
        await handler.on_tool_end(None)

        health.record_response.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_noop_when_no_health(self) -> None:
        handler = RuntimeCallbackHandler()
        await handler.on_tool_end("output")  # Should not raise


class TestOnLlmStart:
    @pytest.mark.asyncio
    async def test_records_budget_llm_turn(self) -> None:
        budget = AsyncMock()
        budget.record_llm_turn = AsyncMock(return_value=None)
        handler = RuntimeCallbackHandler(budget=budget)

        await handler.on_llm_start({}, ["prompt"])

        budget.record_llm_turn.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_collects_llm_violation(self) -> None:
        violation = MagicMock()
        budget = AsyncMock()
        budget.record_llm_turn = AsyncMock(return_value=violation)
        handler = RuntimeCallbackHandler(budget=budget)

        await handler.on_llm_start({}, ["prompt"])

        assert len(handler.pending_violations) == 1

    @pytest.mark.asyncio
    async def test_budget_error_logged_not_raised(self) -> None:
        budget = AsyncMock()
        budget.record_llm_turn = AsyncMock(side_effect=RuntimeError("fail"))
        handler = RuntimeCallbackHandler(budget=budget)

        await handler.on_llm_start({}, ["prompt"])

    @pytest.mark.asyncio
    async def test_noop_when_no_budget(self) -> None:
        handler = RuntimeCallbackHandler()
        await handler.on_llm_start({}, ["prompt"])


class TestParseToolArgs:
    def test_dict_passthrough(self) -> None:
        result = RuntimeCallbackHandler._parse_tool_args({"key": "val"})
        assert result == {"key": "val"}

    def test_valid_json_string(self) -> None:
        result = RuntimeCallbackHandler._parse_tool_args('{"key": "val"}')
        assert result == {"key": "val"}

    def test_invalid_json_string(self) -> None:
        result = RuntimeCallbackHandler._parse_tool_args("not json")
        assert result == {"raw": "not json"}

    def test_non_dict_json(self) -> None:
        result = RuntimeCallbackHandler._parse_tool_args("[1, 2, 3]")
        assert result == {"raw": "[1, 2, 3]"}

    def test_non_string_non_dict(self) -> None:
        result = RuntimeCallbackHandler._parse_tool_args(12345)
        assert result == {}
