"""Test helpers for ``promptise run``: a scripted :class:`~promptise.run.source.RunSource`."""

from __future__ import annotations

import asyncio
import itertools
import time
from collections.abc import AsyncIterator
from typing import Any

from .events import EVENT_VERSION, RunEvent

__all__ = ["FakeRunSource"]


class FakeRunSource:
    """A :class:`~promptise.run.source.RunSource` driven by the test.

    Push events with :meth:`emit`; read what the UI did from :attr:`sent`,
    :attr:`restarts`, :attr:`stops` and :attr:`closed`.  With *echo*, a
    ``chat`` command is answered at once with ``turn.start``, one ``token``
    stream event per word and ``turn.end``.
    """

    def __init__(self, *, echo: bool = False) -> None:
        self._seq = itertools.count(1)
        self._history: list[RunEvent] = []
        self._waiters: list[asyncio.Queue[RunEvent | None]] = []
        self.sent: list[dict[str, Any]] = []
        self.restarts = 0
        self.stops = 0
        self.closed = False
        self.echo = echo

    def emit(self, kind: str, **payload: Any) -> RunEvent:
        """Publish an event of *kind* to every :meth:`events` iterator."""
        event: RunEvent = {"v": EVENT_VERSION, "seq": next(self._seq), "ts": time.time()}
        event["kind"] = kind
        event.update(payload)
        self._history.append(event)
        for queue in self._waiters:
            queue.put_nowait(event)
        return event

    async def events(self) -> AsyncIterator[RunEvent]:
        queue: asyncio.Queue[RunEvent | None] = asyncio.Queue()
        for event in self._history:
            queue.put_nowait(event)
        self._waiters.append(queue)
        try:
            while True:
                item = await queue.get()
                if item is None:
                    return
                yield item
        finally:
            self._waiters.remove(queue)

    async def send(self, command: dict[str, Any]) -> None:
        self.sent.append(command)
        if self.echo and command.get("cmd") == "chat":
            turn = command.get("turn_id") or f"t{len(self.sent)}"
            text = str(command.get("text", ""))
            self.emit("turn.start", turn_id=turn, input=text, conversation_id="c1")
            so_far = ""
            for word in text.split():
                so_far += word + " "
                self.emit(
                    "stream",
                    turn_id=turn,
                    event={"type": "token", "text": word + " ", "cumulative_text": so_far},
                )
            self.emit(
                "stream",
                turn_id=turn,
                event={"type": "done", "full_response": text, "tool_calls": [], "duration_ms": 1.0},
            )
            self.emit("turn.end", turn_id=turn, output=text, duration_ms=1.0, error=None)

    async def restart(self) -> None:
        self.restarts += 1

    async def stop(self) -> None:
        self.stops += 1

    async def close(self) -> None:
        self.closed = True
        for queue in self._waiters:
            queue.put_nowait(None)
