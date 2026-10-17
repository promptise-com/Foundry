"""Run the program in a child process: for ``--watch`` restarts and the terminal UI.

The child is ``python -m promptise run`` with the same options minus
``--watch`` and ``--tui``, so it behaves exactly like a direct run.

- :func:`supervise_watch` (``--watch`` without the UI): the child shares the
  terminal; it is restarted when a watched file changes, and after it exits
  the supervisor waits for the next change, like ``nodemon``.
- :class:`ChildRunSource` (``--tui``): a :class:`~promptise.run.source.RunSource`
  whose child reports its run events over :mod:`promptise.run.ipc` and whose
  stdout/stderr are read line by line, so nothing the program prints can draw
  over the UI.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import os
import signal
import sys
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape

from .events import EVENT_VERSION, RunEvent
from .ipc import IPC_ENV, IpcServer
from .watch import FileWatcher

__all__ = ["CHILD_ENV", "ChildRunSource", "child_command", "supervise_watch"]

CHILD_ENV = "PROMPTISE_RUN_CHILD"
"""Set to ``1`` in the child, so it never starts a supervisor of its own."""

_TERM_TIMEOUT = 8.0
_PARTIAL_LINE_DELAY = 0.15


def child_command(args: Sequence[str]) -> list[str]:
    """The command line of a child ``promptise run`` with *args*."""
    return [sys.executable, "-m", "promptise", "run", *args]


def _child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env[CHILD_ENV] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.update(extra or {})
    return env


def _spawn_kwargs() -> dict[str, Any]:
    if os.name == "posix":
        return {"start_new_session": True}
    import subprocess

    return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}


async def _terminate(
    proc: asyncio.subprocess.Process, timeout: float = _TERM_TIMEOUT
) -> int | None:
    """Ask *proc* to stop (SIGTERM), then kill it and its group after *timeout*."""
    if proc.returncode is not None:
        return proc.returncode
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        return await asyncio.wait_for(proc.wait(), timeout)
    except asyncio.TimeoutError:
        pass
    with contextlib.suppress(ProcessLookupError, PermissionError, OSError):
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    with contextlib.suppress(Exception):
        return await asyncio.wait_for(proc.wait(), 5)
    return None


async def supervise_watch(
    args: Sequence[str],
    *,
    cwd: Path,
    watch_paths: Sequence[Path],
    ignore: Sequence[str],
    root: Path,
) -> int:
    """Run the child, restart it on changes; return 130 when interrupted."""
    console = Console(stderr=True, highlight=False)
    loop = asyncio.get_running_loop()
    changed: list[str] = []
    restart = asyncio.Event()

    def on_change(paths: list[str]) -> None:
        changed[:] = paths
        restart.set()

    watcher = FileWatcher(
        watch_paths,
        lambda paths: loop.call_soon_threadsafe(on_change, paths),
        ignore=ignore,
        root=root,
    )
    watcher.start()
    console.print(
        f"[dim]watching {', '.join(escape(str(p)) for p in watch_paths)} · Ctrl+C to quit[/dim]"
    )

    def rel(path: str) -> str:
        try:
            return str(Path(path).relative_to(root))
        except ValueError:
            return path

    proc: asyncio.subprocess.Process | None = None
    try:
        while True:
            proc = await asyncio.create_subprocess_exec(
                *child_command(args), cwd=str(cwd), env=_child_env(), **_spawn_kwargs()
            )
            wait = asyncio.ensure_future(proc.wait())
            change = asyncio.ensure_future(restart.wait())
            done, _ = await asyncio.wait({wait, change}, return_when=asyncio.FIRST_COMPLETED)
            if change in done:
                wait.cancel()
                restart.clear()
                console.print(
                    f"[cyan]↻ restarting[/cyan] [dim]({escape(', '.join(rel(p) for p in changed[:5]))})[/dim]"
                )
                await _terminate(proc)
                continue
            change.cancel()
            code = wait.result()
            color = "green" if code == 0 else "red"
            console.print(
                f"[{color}]program exited with code {code}[/{color}] "
                "[dim]· waiting for file changes (Ctrl+C to quit)[/dim]"
            )
            await restart.wait()
            restart.clear()
            console.print(
                f"[cyan]↻ restarting[/cyan] [dim]({escape(', '.join(rel(p) for p in changed[:5]))})[/dim]"
            )
    except asyncio.CancelledError:
        return 130
    finally:
        watcher.stop()
        if proc is not None:
            await _terminate(proc)


class ChildRunSource:
    """A :class:`~promptise.run.source.RunSource` backed by a child process.

    Args:
        args: The child's ``promptise run`` arguments.
        cwd: The child's working directory.
        watch: When given, ``(paths, ignore, root)``: restart on changes.
    """

    def __init__(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        watch: tuple[Sequence[Path], Sequence[str], Path] | None = None,
    ) -> None:
        self.args = list(args)
        self.cwd = cwd
        self._watch_cfg = watch
        self._watcher: FileWatcher | None = None
        self._seq = itertools.count(1)
        self._history: list[RunEvent] = []
        self._waiters: list[asyncio.Queue[RunEvent | None]] = []
        self._ipc = IpcServer(self._from_child)
        self._proc: asyncio.subprocess.Process | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._run_exited = False
        self._lock = asyncio.Lock()
        self.last_code: int | None = None
        self.closed = False

    # -- events -----------------------------------------------------------

    def _push(self, event: RunEvent) -> None:
        event["seq"] = next(self._seq)
        if len(self._history) >= 50_000:
            del self._history[:10_000]
        self._history.append(event)
        for queue in self._waiters:
            queue.put_nowait(event)

    def _emit(self, kind: str, **payload: Any) -> None:
        event: RunEvent = {"v": EVENT_VERSION, "seq": 0, "ts": time.time(), "kind": kind}
        event.update(payload)
        self._push(event)

    def _from_child(self, event: RunEvent) -> None:
        if event.get("kind") == "run.exit":
            self._run_exited = True
            code = event.get("code")
            if isinstance(code, int):
                self.last_code = code
        self._push(event)

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
            with contextlib.suppress(ValueError):
                self._waiters.remove(queue)

    # -- process ----------------------------------------------------------

    def _task(self, coro: Any) -> asyncio.Task[Any]:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def start(self) -> None:
        """Open the channel, start watching (if asked) and start the child."""
        await self._ipc.start()
        if self._watch_cfg is not None:
            paths, ignore, root = self._watch_cfg
            loop = asyncio.get_running_loop()

            def on_change(changed: list[str]) -> None:
                self._emit(
                    "state", state="restarting", detail="files changed: " + ", ".join(changed[:5])
                )
                self._task(self.restart())

            self._watcher = FileWatcher(
                paths,
                lambda changed: loop.call_soon_threadsafe(on_change, changed),
                ignore=ignore,
                root=root,
            )
            self._watcher.start()
        await self._spawn()

    async def _spawn(self) -> None:
        self._run_exited = False
        self._emit("state", state="starting", detail="")
        try:
            proc = await asyncio.create_subprocess_exec(
                *child_command(self.args),
                cwd=str(self.cwd),
                env=_child_env({IPC_ENV: self._ipc.address}),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=4 * 1024 * 1024,
                **_spawn_kwargs(),
            )
        except OSError as exc:
            self._emit("state", state="failed", detail=f"could not start: {exc}")
            return
        self._proc = proc
        readers = [
            self._task(self._pump(proc.stdout, "stdout")),
            self._task(self._pump(proc.stderr, "stderr")),
        ]
        self._task(self._wait(proc, readers))

    async def _pump(self, stream: asyncio.StreamReader | None, name: str) -> None:
        """Turn the child's *stream* into ``output`` events, one per line.

        A partial line (an ``input("Name: ")`` prompt) is sent once the child
        has been quiet for a moment, so prompts show up before the user answers.
        """
        if stream is None:
            return
        buffer = b""
        while True:
            try:
                chunk = await asyncio.wait_for(stream.read(65536), _PARTIAL_LINE_DELAY)
            except asyncio.TimeoutError:
                if buffer:
                    self._emit("output", stream=name, text=buffer.decode("utf-8", "replace"))
                    buffer = b""
                continue
            if not chunk:
                if buffer:
                    self._emit("output", stream=name, text=buffer.decode("utf-8", "replace"))
                return
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                self._emit("output", stream=name, text=line.decode("utf-8", "replace").rstrip("\r"))

    async def _wait(
        self, proc: asyncio.subprocess.Process, readers: list[asyncio.Task[Any]]
    ) -> None:
        code = await proc.wait()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(asyncio.gather(*readers), 2)
        self.last_code = code
        if self._proc is proc:
            if not self._run_exited:
                self._emit("run.exit", code=code, error=None, traceback=None, duration_s=None)
            self._emit("state", state="exited", detail=f"exit code {code}")

    async def send(self, command: dict[str, Any]) -> None:
        if command.get("cmd") == "stdin":
            proc = self._proc
            if proc is None or proc.stdin is None or proc.returncode is not None:
                return
            text = str(command.get("text", ""))
            try:
                proc.stdin.write((text + "\n").encode("utf-8"))
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            return
        await self._ipc.send(command)

    async def restart(self) -> None:
        async with self._lock:
            if self.closed:
                return
            self._emit("state", state="restarting", detail="")
            proc, self._proc = self._proc, None
            if proc is not None:
                await _terminate(proc)
                if not self._run_exited:
                    self._emit(
                        "run.exit",
                        code=proc.returncode,
                        error=None,
                        traceback=None,
                        duration_s=None,
                    )
            await self._spawn()

    async def stop(self) -> None:
        async with self._lock:
            proc = self._proc
            if proc is not None and proc.returncode is None:
                self._emit("state", state="stopping", detail="")
                await _terminate(proc)

    async def close(self) -> None:
        if self.closed:
            return
        await self.stop()
        self.closed = True
        if self._watcher is not None:
            self._watcher.stop()
        await self._ipc.close()
        for task in list(self._tasks):
            task.cancel()
        for queue in self._waiters:
            queue.put_nowait(None)
