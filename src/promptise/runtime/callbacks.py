"""Runtime callback handlers for budget, health and journal integration.

Bridges LangChain's callback protocol to the runtime's budget tracking,
behavioral health monitoring and journal.  Passed to ``agent.ainvoke()``
via ``config={"callbacks": handler.callbacks()}``.

Call :meth:`RuntimeCallbackHandler.reset` before each invocation to clear
pending violations.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from .exceptions import BudgetExceededError

logger = logging.getLogger("promptise.runtime.callbacks")

if TYPE_CHECKING:
    from .budget import BudgetState, BudgetViolation
    from .health import HealthMonitor

try:
    from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
except ImportError:  # pragma: no cover

    class BaseCallbackHandler:  # type: ignore[no-redef]
        """Minimal stub when langchain_core is not installed."""

        raise_error = False
        run_inline = False

    class AsyncCallbackHandler:  # type: ignore[no-redef]
        """Minimal stub when langchain_core is not installed."""

        async def on_tool_start(self, serialized: Any, input_str: str, **kwargs: Any) -> None: ...
        async def on_tool_end(self, output: Any, **kwargs: Any) -> None: ...
        async def on_llm_start(
            self, serialized: Any, prompts: list[str], **kwargs: Any
        ) -> None: ...
        async def on_llm_end(self, response: Any, **kwargs: Any) -> None: ...


def _tool_name(serialized: Any) -> str:
    """Tool name from LangChain's ``serialized`` callback argument."""
    if isinstance(serialized, dict):
        return str(serialized.get("name", serialized.get("id", "")) or "")
    return ""


class BudgetGuard(BaseCallbackHandler):
    """Refuses tool calls that would exceed an autonomy budget limit.

    A *synchronous*, inline handler with ``raise_error`` set: LangChain
    runs it before a tool executes, on the async path and on the sync
    path (synchronous tools run in an executor), and propagates the
    :class:`~promptise.runtime.exceptions.BudgetExceededError` it raises,
    so the tool body never runs.  (Errors raised by an *async* handler are
    swallowed on the sync path.)  The agent receives the error as the
    tool's result.

    Args:
        budget: The process's budget state.
        violations: List that collects violations (shared with
            :class:`RuntimeCallbackHandler`).
    """

    raise_error = True
    run_inline = True

    def __init__(self, budget: BudgetState, violations: list[BudgetViolation]) -> None:
        self._budget = budget
        self._violations = violations

    def on_tool_start(self, serialized: Any, input_str: str, **kwargs: Any) -> None:
        """Count the call, or raise if running it would exceed a limit."""
        tool_name = _tool_name(serialized)
        try:
            violation = self._budget.record_tool_call_sync(tool_name, enforce=True)
        except Exception as exc:
            logger.debug("Budget tool call recording failed: %s", exc)
            return
        if violation is None:
            return
        if not any(v.limit_name == violation.limit_name for v in self._violations):
            self._violations.append(violation)
        if violation.blocked:
            raise BudgetExceededError(violation)


class RuntimeCallbackHandler(AsyncCallbackHandler):
    """Feeds tool/LLM events to budget, health and journal subsystems.

    Budget limits on tool calls, cost and irreversible actions are
    enforced **per call** by :attr:`budget_guard`, a :class:`BudgetGuard`
    that runs first in the list returned by :meth:`callbacks`: a call
    that would exceed a limit raises
    :class:`~promptise.runtime.exceptions.BudgetExceededError` before the
    tool runs, and the agent receives the error as the tool result.
    Violations are collected in :attr:`pending_violations` so the process
    can apply ``on_exceeded`` after ``ainvoke()`` returns.

    Tool *outputs* are not fed to health monitoring as agent responses:
    an empty list from a quiet queue is not an empty reply.  The process
    records the agent's final reply itself.

    Usage::

        handler = RuntimeCallbackHandler(budget=budget_state, health=monitor)
        handler.reset()  # clear before each invocation
        result = await agent.ainvoke(input, config={"callbacks": handler.callbacks()})
        if handler.pending_violations:
            # handle first violation ...

    Args:
        budget: Optional budget state tracker.
        health: Optional behavioral health monitor.
        journal: Optional async callback ``(entry_type, data)`` that
            records ``tool_call`` / ``tool_result`` journal entries.
    """

    def __init__(
        self,
        budget: BudgetState | None = None,
        health: HealthMonitor | None = None,
        journal: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    ) -> None:
        self._budget = budget
        self._health = health
        self._journal = journal
        self.pending_violations: list[BudgetViolation] = []
        self.budget_guard: BudgetGuard | None = (
            BudgetGuard(budget, self.pending_violations) if budget is not None else None
        )

    def callbacks(self) -> list[Any]:
        """Handlers to pass as ``config["callbacks"]`` (budget guard first)."""
        return [self.budget_guard, self] if self.budget_guard is not None else [self]

    def reset(self) -> None:
        """Clear pending violations.  Call before each invocation."""
        self.pending_violations.clear()

    async def on_tool_start(self, serialized: Any, input_str: str, **kwargs: Any) -> None:
        """Called when a tool is about to run (after the budget guard allowed it)."""
        tool_name = _tool_name(serialized)
        # LangChain passes the parsed input as ``inputs``; ``input_str`` is
        # only a string rendering of it (not always JSON).
        inputs = kwargs.get("inputs")
        args = dict(inputs) if isinstance(inputs, dict) else self._parse_tool_args(input_str)

        if self._journal is not None:
            try:
                await self._journal("tool_call", {"tool": tool_name, "args": args})
            except Exception as exc:
                logger.debug("Journal tool call recording failed: %s", exc)

        if self._health is not None:
            try:
                await self._health.record_tool_call(tool_name, args)
            except Exception as exc:
                logger.debug("Health tool call recording failed: %s", exc)

    async def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Called when a tool finishes.  Journals the result (full level)."""
        if self._journal is not None:
            try:
                text = str(getattr(output, "content", output)) if output is not None else ""
                await self._journal(
                    "tool_result",
                    {"tool": kwargs.get("name", ""), "result": text[:2000]},
                )
            except Exception as exc:
                logger.debug("Journal tool result recording failed: %s", exc)

    async def on_llm_start(self, serialized: Any, prompts: list[str], **kwargs: Any) -> None:
        """Called when an LLM call is about to be made."""
        if self._budget is not None:
            try:
                violation = await self._budget.record_llm_turn()
                if violation is not None:
                    self.pending_violations.append(violation)
            except Exception as exc:
                logger.debug("Budget LLM turn recording failed: %s", exc)

    @staticmethod
    def _parse_tool_args(input_str: Any) -> dict[str, Any]:
        """Best-effort parse tool arguments from input."""
        if isinstance(input_str, dict):
            return input_str
        if isinstance(input_str, str):
            try:
                parsed = json.loads(input_str)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
            return {"raw": input_str}
        return {}
