"""The ``promptise run --tui`` terminal UI.

:func:`run_tui` shows a :class:`~promptise.run.source.RunSource` (normally a
program running in a child process) in a full-screen Textual app: chat or
console, trace, tool calls, metrics, agent events, agents, MCP servers, logs
and the run itself.  The Textual app is imported only when :func:`run_tui`
runs, so importing this package stays cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..source import RunSource
    from .app import RunTUI
    from .state import RunState

__all__ = ["RunState", "RunTUI", "run_tui"]


async def run_tui(source: RunSource, *, title: str = "promptise run") -> int:
    """Show *source* in the terminal UI until the user quits; return the program's exit code.

    The source is closed when the UI ends (``q``, ``ctrl+c``, or an error).
    The result is the exit code of the program's last run, or 0 when no run
    ended (it was still running when the user quit).
    """
    from .app import RunTUI

    app = RunTUI(source, title=title)
    try:
        await app.run_async()
    finally:
        await app.close_source()
    return app.exit_code


def __getattr__(name: str) -> Any:
    if name == "RunTUI":
        from .app import RunTUI

        return RunTUI
    if name == "RunState":
        from .state import RunState

        return RunState
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
