"""Minimal LLM plumbing shared by curation and readiness evaluation.

Every model call goes through :func:`promptise.agent.build_agent` — the
tool dogfoods the framework it ships with.  Tests inject a scripted
``Completer`` instead of a model; production code never does.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from .schema import MCPcastError

__all__ = ["Completer", "complete", "extract_json_object", "final_text"]

Completer = Callable[[str, str], Awaitable[str]]
"""``async (system_prompt, user_prompt) -> completion text``."""

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


async def complete(model: Any, system: str, user: str) -> str:
    """One completion from *model* (an id string or a LangChain chat model).

    Curation and task generation are deliberately tool-less, so the agent's
    "no tools discovered" notice is expected here and is swallowed rather
    than printed into the CLI's output.
    """
    import contextlib
    import io

    from langchain_core.messages import HumanMessage

    from promptise.agent import build_agent

    if isinstance(model, str):
        # Fail before any prompt is built: a wrong prefix or a missing key is a
        # setup problem, and the message says exactly what to fix.
        from promptise.models import resolve_model

        model = resolve_model(model)
    with contextlib.redirect_stdout(io.StringIO()):
        agent = await build_agent(servers={}, model=model, instructions=system)
    result = await agent.ainvoke({"messages": [HumanMessage(content=user)]})
    return final_text(result)


def completer_for(model: Any) -> Completer:
    """Bind *model* into a :data:`Completer`."""

    async def _run(system: str, user: str) -> str:
        return await complete(model, system, user)

    return _run


def final_text(result: Any) -> str:
    """The final *assistant* text from an agent invocation result.

    Walks the message list backwards to the last AI message, so a run that
    stopped mid tool-loop (last message is a tool result) yields ``""``
    rather than raw tool output.
    """
    if isinstance(result, dict) and result.get("messages"):
        for message in reversed(result["messages"]):
            kind = getattr(message, "type", None) or (
                message.get("role") if isinstance(message, dict) else None
            )
            if kind in ("ai", "assistant"):
                return _content_text(getattr(message, "content", message))
        return ""
    return str(result)


def _content_text(content: Any) -> str:
    if isinstance(content, dict):
        content = content.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            c.get("text", "") if isinstance(c, dict) else str(c)
            for c in content
            if not (isinstance(c, dict) and c.get("type") not in (None, "text"))
        )
    return str(content)


def extract_json_object(text: str) -> dict[str, Any]:
    """The JSON object in *text* — fenced, bare, or surrounded by prose.

    Raises:
        MCPcastError: If no JSON object can be decoded.
    """
    candidates: list[str] = [m.group(1) for m in _FENCE.finditer(text)]
    stripped = text.strip()
    candidates.append(stripped)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise MCPcastError("model response did not contain a JSON object")
