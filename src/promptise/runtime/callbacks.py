"""Runtime callback handler for budget and health integration.

Bridges LangChain's callback protocol to the runtime's budget tracking
and behavioral health monitoring.  Passed to ``agent.ainvoke()`` via
``config={"callbacks": [handler]}``.

Call :meth:`reset` before each invocation to clear pending violations.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("promptise.runtime.callbacks")

if TYPE_CHECKING:
    from .budget import BudgetState, BudgetViolation
    from .health import HealthMonitor

try:
    from langchain_core.callbacks import AsyncCallbackHandler
except ImportError:  # pragma: no cover

    class AsyncCallbackHandler:  # type: ignore[no-redef]
        """Minimal stub when langchain_core is not installed."""

        async def on_tool_start(self, serialized: Any, input_str: str, **kwargs: Any) -> None: ...
        async def on_tool_end(self, output: Any, **kwargs: Any) -> None: ...
        async def on_llm_start(
            self, serialized: Any, prompts: list[str], **kwargs: Any
        ) -> None: ...
        async def on_llm_end(self, response: Any, **kwargs: Any) -> None: ...


class RuntimeCallbackHandler(AsyncCallbackHandler):
    """Feeds tool/LLM events to budget and health subsystems.

    Collects :class:`BudgetViolation` instances during an invocation
    so the process can handle them after ``ainvoke()`` returns.

    Usage::

        handler = RuntimeCallbackHandler(budget=budget_state, health=monitor)
        handler.reset()  # clear before each invocation
        result = await agent.ainvoke(input, config={"callbacks": [handler]})
        if handler.pending_violations:
            # handle first violation ...

    Args:
        budget: Optional budget state tracker.
        health: Optional behavioral health monitor.
    """

    def __init__(
        self,
        budget: BudgetState | None = None,
        health: HealthMonitor | None = None,
    ) -> None:
        self._budget = budget
        self._health = health
        self.pending_violations: list[BudgetViolation] = []

    def reset(self) -> None:
        """Clear pending violations.  Call before each invocation."""
        self.pending_violations.clear()

    async def on_tool_start(self, serialized: Any, input_str: str, **kwargs: Any) -> None:
        """Called when a tool is about to be invoked."""
        tool_name = ""
        if isinstance(serialized, dict):
            tool_name = serialized.get("name", serialized.get("id", ""))

        if self._budget is not None:
            try:
                violation = await self._budget.record_tool_call(tool_name)
                if violation is not None:
                    self.pending_violations.append(violation)
            except Exception as exc:
                logger.debug("Budget tool call recording failed: %s", exc)

        if self._health is not None:
            try:
                args = self._parse_tool_args(input_str)
                await self._health.record_tool_call(tool_name, args)
            except Exception as exc:
                logger.debug("Health tool call recording failed: %s", exc)

    async def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Called when a tool finishes.  Records response for health."""
        if self._health is not None:
            try:
                # A tool invoked as a tool call returns a ToolMessage.
                content = getattr(output, "content", output)
                text = str(content) if content is not None else ""
                await self._health.record_response(text)
            except Exception as exc:
                logger.debug("Health tool end recording failed: %s", exc)

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
