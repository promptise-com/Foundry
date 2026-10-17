"""``--watch``: notice when project files change (watchdog, debounced)."""

from __future__ import annotations

import asyncio
import fnmatch
import os
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from .manifest import ignored_dir_names

__all__ = ["WATCHED_SUFFIXES", "FileWatcher"]

WATCHED_SUFFIXES = {
    ".py",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".superagent",
    ".agent",
    ".md",
    ".txt",
    ".env",
    ".j2",
    ".jinja",
}
"""File types whose changes restart the program (``.env`` files included)."""


class FileWatcher:
    """Calls *on_change* with the changed paths, at most once per *debounce* seconds.

    Args:
        paths: Files or directories to watch (recursively).
        on_change: Called from a timer thread with the sorted changed paths.
        ignore: Extra glob patterns (matched against the path relative to
            *root* and against the file name).
        root: Base for relative patterns.
        debounce: Quiet period before *on_change* fires.
    """

    def __init__(
        self,
        paths: Iterable[Path],
        on_change: Callable[[list[str]], object],
        *,
        ignore: Iterable[str] = (),
        root: Path | None = None,
        debounce: float = 0.3,
    ) -> None:
        self.paths = [Path(p).absolute() for p in paths]
        self.on_change = on_change
        self.ignore = list(ignore)
        self.root = (root or Path.cwd()).absolute()
        self.debounce = debounce
        self._ignored_dirs = set(ignored_dir_names())
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._observer: Any = None

    def relevant(self, path: str) -> bool:
        """Whether a change to *path* should restart the program."""
        p = Path(path)
        if any(part in self._ignored_dirs for part in p.parts):
            return False
        name = p.name
        if name.startswith(".#") or name.endswith(("~", ".swp", ".swx", ".tmp")):
            return False
        if p.suffix not in WATCHED_SUFFIXES and not name.startswith(".env"):
            return False
        try:
            rel = str(p.absolute().relative_to(self.root))
        except ValueError:
            rel = str(p)
        return not any(
            fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(name, pat) for pat in self.ignore
        )

    def _changed(self, path: str) -> None:
        if not self.relevant(path):
            return
        with self._lock:
            self._pending.add(path)
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        with self._lock:
            changed, self._pending = sorted(self._pending), set()
            self._timer = None
        if changed:
            self.on_change(changed)

    def start(self) -> None:
        from watchdog.events import FileSystemEvent, FileSystemEventHandler
        from watchdog.observers import Observer

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_any_event(self, event: FileSystemEvent) -> None:
                if event.event_type in ("opened", "closed_no_write"):
                    return
                if event.is_directory:
                    return
                for attr in ("src_path", "dest_path"):
                    value = getattr(event, attr, None)
                    if value:
                        watcher._changed(os.fsdecode(value))

        observer = Observer()
        handler = _Handler()
        for path in self.paths:
            if path.is_dir():
                observer.schedule(handler, str(path), recursive=True)
            elif path.exists():
                observer.schedule(handler, str(path.parent), recursive=False)
        observer.daemon = True
        observer.start()
        self._observer = observer

    def stop(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2)
            self._observer = None


def async_callback(
    loop: asyncio.AbstractEventLoop, callback: Callable[[list[str]], Any]
) -> Callable[[list[str]], None]:
    """Wrap an event-loop *callback* so a watcher thread can call it."""

    def _call(paths: list[str]) -> None:
        loop.call_soon_threadsafe(callback, paths)

    return _call
