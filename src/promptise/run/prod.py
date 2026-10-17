"""``--prod``: structured logs, preflight checks and a graceful SIGTERM.

In production ``promptise run``:

- writes logs as one JSON object per line on stderr (``[prod] log_format``),
  so stdout stays the program's own output;
- with ``[prod] observe`` set, writes every agent timeline event as a JSON
  log line too (``logger: "promptise.run.events"``);
- checks the project before starting (:func:`preflight`) and exits with code 2
  listing *every* problem, instead of failing on the first one at runtime;
- turns SIGTERM into a normal shutdown: the program sees ``KeyboardInterrupt``
  (``asyncio.run`` cancels its main task, ``finally`` blocks run, agents shut
  down), and the process exits with 143.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import IO, Any

from .events import RunEvent

__all__ = [
    "JsonEventLog",
    "JsonFormatter",
    "SigtermHandler",
    "configure_logging",
    "missing_env",
]

_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """Formats a log record as one JSON object: ``ts, level, logger, msg`` (+ extras)."""

    def format(self, record: logging.LogRecord) -> str:
        data: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                data[key] = value
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        return json.dumps(data, default=str, ensure_ascii=False)


def configure_logging(level: str, fmt: str = "json", stream: IO[str] | None = None) -> None:
    """Send every log record to *stream* (default stderr), as JSON or plain text."""
    handler = logging.StreamHandler(stream or sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())
    logging.captureWarnings(True)


class JsonEventLog:
    """Writes run events as JSON log lines (a bus subscriber).

    Timeline and agent events are written when *timeline* is on;
    ``run.start``, ``agent.built``, ``agent.closed``, ``run.exit`` and the
    final ``stats`` always are.
    """

    _ALWAYS = {"run.start", "run.exit", "agent.built", "agent.closed", "http"}

    def __init__(self, *, timeline: bool, stream: IO[str] | None = None) -> None:
        self.timeline = timeline
        self._stream = stream
        self._lock = threading.Lock()

    def __call__(self, event: RunEvent) -> None:
        kind = event.get("kind")
        if kind not in self._ALWAYS and not (self.timeline and kind in ("timeline", "event")):
            return
        payload = {k: v for k, v in event.items() if k not in ("v", "seq", "ts", "kind")}
        line = {
            "ts": datetime.fromtimestamp(event.get("ts", time.time()), tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": "info",
            "logger": "promptise.run.events",
            "msg": kind,
            **payload,
        }
        if kind == "agent.built":
            # Tool schemas are long; the names are what a log reader needs.
            line["tools"] = [t.get("name") for t in payload.get("tools") or []]
            line.pop("instructions", None)
        text = json.dumps(line, default=str, ensure_ascii=False)
        with self._lock:
            stream = self._stream or sys.stderr
            stream.write(text + "\n")
            stream.flush()

    def final_stats(self, stats: dict[str, Any]) -> None:
        """Write the run's totals as one ``stats`` line."""
        line = {
            "ts": datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds"),
            "level": "info",
            "logger": "promptise.run.events",
            "msg": "stats",
            **stats,
        }
        with self._lock:
            stream = self._stream or sys.stderr
            stream.write(json.dumps(line, default=str) + "\n")
            stream.flush()


def missing_env(names: list[str]) -> list[str]:
    """The names in *names* that are not set (or set to an empty string)."""
    return [n for n in names if not os.environ.get(n)]


class SigtermHandler:
    """Turns SIGTERM into ``KeyboardInterrupt`` in the main thread.

    The program then shuts down the way it does on Ctrl+C.  A second SIGTERM
    while shutting down exits at once with 143.  :attr:`received` tells the
    caller to exit with 143 instead of 130.
    """

    def __init__(self, on_term: Callable[[], None] | None = None) -> None:
        self.received = False
        self._on_term = on_term
        self._previous: Any = None

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        if not hasattr(signal, "SIGTERM"):
            return
        self._previous = signal.signal(signal.SIGTERM, self._handle)

    def uninstall(self) -> None:
        if self._previous is not None:
            try:
                signal.signal(signal.SIGTERM, self._previous)
            except (ValueError, OSError):
                pass
            self._previous = None

    def _handle(self, signum: int, frame: Any) -> None:
        if self.received:
            os._exit(143)
        self.received = True
        if self._on_term is not None:
            try:
                self._on_term()
            except Exception:
                pass
        raise KeyboardInterrupt
