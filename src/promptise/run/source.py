"""What the terminal UI reads from and controls: a :class:`RunSource`.

The UI never touches an agent directly.  It reads run events (see
:mod:`promptise.run.events`) from a source and sends it commands, so the same
UI works for a program running in a child process (the normal case, see
:mod:`promptise.run.supervisor`) and for a scripted fake in tests
(:class:`promptise.run.testing.FakeRunSource`).

Commands (dicts):

``{"cmd": "chat", "text": str, "turn_id": str}``
    Ask the agent (chat mode).  The answer arrives as ``turn.start``,
    ``stream`` and ``turn.end`` events with the same ``turn_id``.
``{"cmd": "stdin", "text": str}``
    Write one line to the program's standard input (script mode).
``{"cmd": "reset"}``
    Start a new conversation (chat mode).
``{"cmd": "cancel", "turn_id": str}``
    Cancel a running turn.
``{"cmd": "approval", "request_id": str, "approved": bool, "reason": str | None}``
    Answer an ``approval.pending`` request.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from .events import RunEvent

__all__ = ["RunSource"]


@runtime_checkable
class RunSource(Protocol):
    """A running (or finished) program the terminal UI shows."""

    def events(self) -> AsyncIterator[RunEvent]:
        """Every event so far, then new ones as they happen.

        The iterator spans restarts (a restart's new ``run.start`` follows
        the old run's ``run.exit``) and ends only when the source is closed.
        """
        ...

    async def send(self, command: dict[str, Any]) -> None:
        """Send *command* to the program (see the module docstring)."""
        ...

    async def restart(self) -> None:
        """Stop the program if it runs, and start it again."""
        ...

    async def stop(self) -> None:
        """Stop the program; the source stays open, so the UI can still show the run."""
        ...

    async def close(self) -> None:
        """Stop the program and end :meth:`events`."""
        ...
