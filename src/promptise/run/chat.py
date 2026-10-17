"""The chat and once modes: talk to the agent a factory builds.

:class:`Conversation` runs one turn at a time with
:meth:`~promptise.agent.PromptiseAgent.astream_with_tools`, keeps the
conversation's text history, and publishes ``turn.start``, ``stream`` and
``turn.end`` run events.  :func:`repl` is the interactive terminal chat,
:func:`ask_once` answers one question.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.markup import escape

from .devtools import print_summary
from .events import EventBus
from .harness import RunHarness

__all__ = ["Conversation", "ask_once", "repl"]

_HELP = """[bold]Commands[/bold]
  /help    this help
  /stats   tokens, cost and tool calls so far
  /tools   the agent's tools
  /reset   start a new conversation
  /exit    quit (also Ctrl+D)"""


class Conversation:
    """One conversation with *agent* (text history, run events per turn)."""

    def __init__(self, agent: Any, bus: EventBus | None, harness: RunHarness | None = None) -> None:
        self.agent = agent
        self.bus = bus
        self.harness = harness
        self.id = uuid.uuid4().hex
        self.messages: list[dict[str, str]] = []

    def reset(self) -> str:
        """Forget the history; return the new conversation id."""
        self.id = uuid.uuid4().hex
        self.messages = []
        return self.id

    def _publish(self, kind: str, **payload: Any) -> None:
        if self.bus is not None:
            self.bus.publish(kind, **payload)

    async def turn(
        self,
        text: str,
        *,
        turn_id: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[str, str | None]:
        """Ask *text*; return ``(answer, error)``.  *on_event* sees each stream event."""
        turn_id = turn_id or uuid.uuid4().hex
        self._publish("turn.start", turn_id=turn_id, input=text, conversation_id=self.id)
        started = time.perf_counter()
        output, error = "", None
        messages = [*self.messages, {"role": "user", "content": text}]
        try:
            async for event in self.agent.astream_with_tools({"messages": messages}):
                data = event.to_dict() if hasattr(event, "to_dict") else dict(event)
                self._publish("stream", turn_id=turn_id, event=data)
                if on_event is not None:
                    on_event(data)
                if data.get("type") == "done":
                    output = str(data.get("full_response") or "")
                elif data.get("type") == "error":
                    error = str(data.get("message") or "error")
        except asyncio.CancelledError:
            error = "cancelled"
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            self._publish(
                "turn.end",
                turn_id=turn_id,
                output=output,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
                error=error,
            )
            if self.harness is not None:
                self.harness.flush_stats()
        if error is None:
            self.messages = [*messages, {"role": "assistant", "content": output}]
        return output, error


async def ask_once(
    agent: Any,
    question: str,
    *,
    bus: EventBus | None,
    harness: RunHarness | None,
    as_json: bool = False,
) -> int:
    """Answer *question*, print the answer (or JSON) to stdout, return the exit code."""
    convo = Conversation(agent, bus, harness)
    started = time.perf_counter()
    output, error = await convo.turn(question)
    if as_json:
        totals = harness.stats.snapshot()["totals"] if harness is not None else {}
        print(
            json.dumps(
                {
                    "output": output,
                    "error": error,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    "usage": totals,
                },
                default=str,
            )
        )
    elif error is None:
        print(output)
    else:
        print(f"error: {error}", file=sys.stderr)
    return 0 if error is None else 1


async def repl(
    agent: Any,
    *,
    bus: EventBus | None,
    harness: RunHarness | None,
    title: str,
    devtools: bool,
    read_line: Callable[[str], str] = input,
) -> int:
    """The interactive chat: stream answers, show tool calls, handle slash commands."""
    out = Console(highlight=False)
    err = Console(stderr=True, highlight=False)
    convo = Conversation(agent, bus, harness)
    model = getattr(agent, "model_name", None) or "agent"
    err.print(
        f"[bold cyan]{escape(title)}[/bold cyan] · chatting with [bold]{escape(str(model))}[/bold] "
        "· [dim]/help for commands, Ctrl+D to quit[/dim]"
    )
    started = time.time()
    while True:
        try:
            line = await asyncio.to_thread(read_line, "\nyou › ")
        except (EOFError, KeyboardInterrupt):
            err.print()
            break
        text = line.strip()
        if not text:
            continue
        if text.startswith("/"):
            command = text.split()[0].lower()
            if command in ("/exit", "/quit"):
                break
            if command == "/help":
                err.print(_HELP)
            elif command == "/reset":
                convo.reset()
                err.print("[dim]new conversation[/dim]")
            elif command == "/tools":
                for tool in getattr(agent, "tools", None) or []:
                    desc = " ".join(str(getattr(tool, "description", "") or "").split())[:90]
                    err.print(f"  [yellow]{escape(tool.name)}[/yellow] [dim]{escape(desc)}[/dim]")
            elif command == "/stats":
                if harness is not None:
                    print_summary(
                        harness.stats.snapshot(), duration_s=time.time() - started, console=err
                    )
                else:
                    err.print("[dim]stats need --devtools[/dim]")
            else:
                err.print(f"[red]unknown command {escape(command)}[/red] (/help)")
            continue

        streamed: list[str] = []

        def show(event: dict[str, Any], streamed: list[str] = streamed) -> None:
            kind = event.get("type")
            if kind == "token" and not devtools:
                # With dev tools on, the trace prints while the agent works;
                # the answer is printed whole at the end so the two don't mix.
                chunk = str(event.get("text") or "")
                if chunk:
                    if not streamed:
                        out.print("[bold green]agent ›[/bold green] ", end="")
                    streamed.append(chunk)
                    out.print(chunk, end="", markup=False, highlight=False)
            elif kind == "tool_start" and not devtools:
                name = event.get("tool_display_name") or event.get("tool_name")
                if streamed:
                    out.print()
                    streamed.clear()
                err.print(f"[dim]  ⚙ {escape(str(name))}…[/dim]")
            elif kind == "tool_end" and not devtools and not event.get("success", True):
                err.print(
                    f"[red]  ✗ {escape(str(event.get('tool_summary') or 'tool failed'))}[/red]"
                )

        try:
            output, error = await convo.turn(text, on_event=show)
        except KeyboardInterrupt:
            err.print("\n[dim]interrupted[/dim]")
            continue
        if streamed:
            out.print()
        elif output:
            out.print("[bold green]agent ›[/bold green] ", end="")
            out.print(output, markup=False, highlight=False)
        if error is not None:
            err.print(f"[red]error:[/red] {escape(error)}")
    return 0
