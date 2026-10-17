"""The hook ``promptise run`` uses to instrument the agents a program builds.

:func:`promptise.agent.build_agent` asks the installed :class:`Harness` (if
any) for overrides of a few arguments and tells it about every agent it
builds and every agent that shuts down.  Nothing is installed outside
``promptise run``, so for every other program this module does nothing.

The hook is process-global rather than a context variable on purpose: the
program ``promptise run`` runs may build agents from threads it starts.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol

__all__ = ["Harness", "current", "install", "uninstall"]


class Harness(Protocol):
    """What ``build_agent`` and ``PromptiseAgent.shutdown`` call."""

    def build_overrides(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return *args* (``model``, ``observe``, ``observer_agent_id``,
        ``events``, ``identity``) with the values the run wants changed."""
        ...

    def agent_built(self, agent: Any) -> None:
        """Called with each agent right before ``build_agent`` returns it."""
        ...

    def agent_closed(self, agent: Any) -> None:
        """Called at the end of ``PromptiseAgent.shutdown()``."""
        ...


_lock = threading.Lock()
_current: Harness | None = None


def current() -> Harness | None:
    """The installed harness, or ``None`` (the normal case)."""
    return _current


def install(harness: Harness) -> None:
    """Install *harness* for the whole process.

    Raises:
        RuntimeError: Another harness is installed already.
    """
    global _current
    with _lock:
        if _current is not None and _current is not harness:
            raise RuntimeError("a promptise run harness is already installed")
        _current = harness


def uninstall(harness: Harness | None = None) -> None:
    """Remove the installed harness (only *harness*, when given)."""
    global _current
    with _lock:
        if harness is None or _current is harness:
            _current = None
