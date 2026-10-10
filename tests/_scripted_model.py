"""A deterministic chat model for agent tests: no network, scripted tool calls.

``Scripted(mode=...)``:

- ``"answer"`` — reply ``"ok"`` without calling a tool.
- ``"tool:<name>"`` — call tool ``<name>`` once, then reply
  ``"ANSWER: <tool result>"``.
- ``"delegate"`` — call the first ``ask_agent_*`` tool on every turn and never
  answer (an agent that keeps delegating).

``calls`` counts every model turn across all instances.
"""

from __future__ import annotations

import itertools
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

calls = {"n": 0}
_ids = itertools.count()


def _result(message: AIMessage) -> ChatResult:
    return ChatResult(generations=[ChatGeneration(message=message)])


def _call(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"c{next(_ids)}"}])


class Scripted(BaseChatModel):
    mode: str = "answer"
    args: dict[str, Any] = {}
    tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Scripted:  # type: ignore[override]
        names = [getattr(t, "name", None) or t.get("name") for t in tools]
        return self.model_copy(update={"tool_names": names})

    def _generate(self, messages: list[BaseMessage], stop: Any = None, **kw: Any) -> ChatResult:
        raise NotImplementedError("async only")

    async def _agenerate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        calls["n"] += 1
        last_tool = next((m for m in reversed(messages) if isinstance(m, ToolMessage)), None)
        if self.mode == "delegate":
            ask = [n for n in self.tool_names if n.startswith("ask_agent_")]
            if ask:
                return _result(_call(ask[0], {"message": "What is 2 + 2?"}))
            return _result(AIMessage(content="4"))
        if self.mode.startswith("tool:"):
            name = self.mode.split(":", 1)[1]
            if last_tool is None:
                args = self.args or (
                    {"message": "What plan is acme on?"}
                    if name.startswith("ask_agent_")
                    else {"customer_id": "acme"}
                )
                return _result(_call(name, args))
            return _result(AIMessage(content=f"ANSWER: {last_tool.content}"))
        return _result(AIMessage(content="ok"))
