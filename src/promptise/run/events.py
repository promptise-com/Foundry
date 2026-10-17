"""Run events: the one stream of data every ``promptise run`` view consumes.

Everything the dev tools print, the terminal UI shows, the prod logs write and
the run report saves is a *run event*: a JSON-serialisable dict
``{"v": 1, "seq": int, "ts": float, "kind": str, ...payload}``.

Kinds and their payload fields:

``run.start``
    ``project, project_version, main, mode, entry_kind, pid, python,
    promptise_version, cwd, root, argv, flags {devtools, prod, tui, watch},
    model_override, config, env [{name, set, required}], scripts``
``run.exit``
    ``code, error, traceback, duration_s``
``state``
    ``state`` (``starting``, ``running``, ``idle``, ``busy``, ``exited``,
    ``restarting``, ``failed``), ``detail``
``agent.built``
    ``agent, model, observe_level, tools [{name, description, server,
    args_schema}], servers [{name, transport, target}], features, instructions``
``agent.closed``
    ``agent``
``timeline``
    ``agent, entry`` (:meth:`promptise.observability.TimelineEntry.to_dict`)
``event``
    ``agent, event`` (:meth:`promptise.events.AgentEvent.to_dict`)
``stats``
    ``totals, agents, models, tools`` (see :mod:`promptise.run.stats`)
``turn.start`` / ``stream`` / ``turn.end``
    one chat / once / serve turn: ``turn_id, input, conversation_id`` /
    ``turn_id, event`` (:meth:`promptise.streaming.StreamEvent.to_dict`) /
    ``turn_id, output, duration_ms, error``
``http``
    ``method, path, status, duration_ms, request_id``
``approval.pending`` / ``approval.resolved``
    a tool call waiting for a human (terminal UI, agents with a
    ``QueueApprovalHandler``): ``agent, request, deadline`` /
    ``request_id, approved, reason, outcome`` (``approved``, ``denied``, ``expired``)
``log``
    ``logger, level, message, exc_text``
``output``
    ``stream`` (``stdout``/``stderr``), ``text``: one line the program printed
    (made by the terminal UI from the child process's pipes)
"""

from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from typing import Any

__all__ = ["EVENT_VERSION", "EventBus", "RunEvent", "Subscriber", "make_event"]

EVENT_VERSION = 1

RunEvent = dict[str, Any]
"""One run event (see the module docstring for the kinds)."""

Subscriber = Callable[[RunEvent], None]
"""A function called with every published event, in the publishing thread.

It must return quickly and never raise: hand slow work to a queue."""


def make_event(kind: str, seq: int, **payload: Any) -> RunEvent:
    """A run event of *kind* with sequence number *seq* and *payload*."""
    event: RunEvent = {"v": EVENT_VERSION, "seq": seq, "ts": time.time(), "kind": kind}
    event.update(payload)
    return event


class EventBus:
    """A thread-safe publish/subscribe channel for run events, with a bounded history.

    :meth:`publish` may be called from any thread (agents record from the
    event loop, worker threads and callback handlers).  Subscribers run in the
    publishing thread, in subscription order; one that raises is unsubscribed
    so it cannot break the program.

    Args:
        history: How many recent events :meth:`history` keeps.
    """

    def __init__(self, *, history: int = 20_000) -> None:
        self._lock = threading.Lock()
        self._seq = itertools.count(1)
        self._subscribers: list[Subscriber] = []
        self._history: deque[RunEvent] = deque(maxlen=history)
        self.dropped_subscribers = 0

    def publish(self, kind: str, **payload: Any) -> RunEvent:
        """Publish an event of *kind* and return it."""
        with self._lock:
            event = make_event(kind, next(self._seq), **payload)
            self._history.append(event)
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber(event)
            except Exception:
                self.unsubscribe(subscriber)
                self.dropped_subscribers += 1
        return event

    def subscribe(self, subscriber: Subscriber, *, replay: bool = False) -> Subscriber:
        """Call *subscriber* with every event from now on (and the history first, with *replay*)."""
        with self._lock:
            past = list(self._history) if replay else []
            self._subscribers.append(subscriber)
        for event in past:
            try:
                subscriber(event)
            except Exception:
                self.unsubscribe(subscriber)
                self.dropped_subscribers += 1
                break
        return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        """Stop calling *subscriber* (no-op when it is not subscribed)."""
        with self._lock:
            try:
                self._subscribers.remove(subscriber)
            except ValueError:
                pass

    def history(self, kinds: Iterable[str] | None = None) -> list[RunEvent]:
        """The kept events, oldest first, optionally only those of *kinds*."""
        with self._lock:
            events = list(self._history)
        if kinds is None:
            return events
        wanted = set(kinds)
        return [e for e in events if e["kind"] in wanted]
