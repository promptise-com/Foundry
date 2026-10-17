"""The local channel between the terminal UI (parent) and the program (child).

The parent listens on ``127.0.0.1`` on a free port and starts the child with
``PROMPTISE_RUN_IPC=127.0.0.1:<port>:<token>``.  The child connects, sends the
token, then streams its run events as JSON lines; the parent sends commands
(see :mod:`promptise.run.source`) as JSON lines the other way.  The token
keeps other local processes from attaching to the channel.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import json
import logging
import os
import queue
import secrets
import socket
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from .events import EventBus, RunEvent

__all__ = ["IPC_ENV", "IpcClient", "IpcServer", "LogBridge"]

IPC_ENV = "PROMPTISE_RUN_IPC"

_MAX_LINE = 16 * 1024 * 1024


def _dumps(obj: Any) -> bytes:
    return (json.dumps(obj, default=str, ensure_ascii=False) + "\n").encode("utf-8")


class IpcClient:
    """The child's end: sends run events, receives commands.

    Sending happens on a writer thread, so publishing never blocks the
    program.  Commands are handed to :attr:`on_command` on a reader thread.
    """

    def __init__(self, address: str) -> None:
        host, port, token = address.rsplit(":", 2)
        self._sock = socket.create_connection((host, int(port)), timeout=10)
        self._sock.settimeout(None)
        self._sock.sendall(_dumps({"token": token, "pid": os.getpid()}))
        self._out: queue.Queue[bytes | None] = queue.Queue()
        self._closed = False
        self.on_command: Callable[[dict[str, Any]], None] | None = None
        self.on_close: Callable[[], None] | None = None
        """Called (on the reader thread) when the parent closes the channel."""
        self._writer = threading.Thread(
            target=self._write_loop, name="promptise-ipc-w", daemon=True
        )
        self._reader = threading.Thread(target=self._read_loop, name="promptise-ipc-r", daemon=True)
        self._writer.start()
        self._reader.start()

    @classmethod
    def from_env(cls) -> IpcClient | None:
        address = os.environ.get(IPC_ENV)
        if not address:
            return None
        return cls(address)

    def __call__(self, event: RunEvent) -> None:
        """Bus subscriber: queue *event* for the parent."""
        if not self._closed:
            self._out.put(_dumps(event))

    def _write_loop(self) -> None:
        while True:
            data = self._out.get()
            if data is None:
                return
            try:
                self._sock.sendall(data)
            except OSError:
                self._closed = True
                return

    def _read_loop(self) -> None:
        try:
            self._read_commands()
        finally:
            handler = self.on_close
            if handler is not None:
                with contextlib.suppress(Exception):
                    handler()

    def _read_commands(self) -> None:
        buf = b""
        try:
            while True:
                chunk = self._sock.recv(65536)
                if not chunk:
                    return
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        command = json.loads(line)
                    except ValueError:
                        continue
                    handler = self.on_command
                    if handler is not None and isinstance(command, dict):
                        try:
                            handler(command)
                        except Exception:
                            logging.getLogger("promptise.run").debug(
                                "command handler failed", exc_info=True
                            )
        except OSError:
            return

    def close(self, timeout: float = 5.0) -> None:
        """Send what is queued, then close the connection."""
        if self._closed:
            return
        self._out.put(None)
        self._writer.join(timeout)
        self._closed = True
        with contextlib.suppress(OSError):
            self._sock.shutdown(socket.SHUT_RDWR)
        with contextlib.suppress(OSError):
            self._sock.close()


class LogBridge(logging.Handler):
    """A logging handler that publishes records as ``log`` run events."""

    def __init__(self, bus: EventBus, level: int = logging.INFO) -> None:
        super().__init__(level)
        self.bus = bus

    def emit(self, record: logging.LogRecord) -> None:
        try:
            exc_text = None
            if record.exc_info:
                exc_text = logging.Formatter().formatException(record.exc_info)
            self.bus.publish(
                "log",
                logger=record.name,
                level=record.levelname,
                message=record.getMessage(),
                exc_text=exc_text,
            )
        except Exception:
            self.handleError(record)


class IpcServer:
    """The parent's end: accepts one child connection at a time.

    Args:
        on_event: Called with every event the child sends.
        on_disconnect: Called when the child's connection ends.
    """

    def __init__(
        self,
        on_event: Callable[[RunEvent], Awaitable[None] | None],
        on_disconnect: Callable[[], None] | None = None,
    ) -> None:
        self._on_event = on_event
        self._on_disconnect = on_disconnect
        self.token = secrets.token_hex(16)
        self._server: asyncio.AbstractServer | None = None
        self._writer: asyncio.StreamWriter | None = None
        self.port = 0

    @property
    def address(self) -> str:
        return f"127.0.0.1:{self.port}:{self.token}"

    @property
    def connected(self) -> bool:
        return self._writer is not None and not self._writer.is_closing()

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._handle, host="127.0.0.1", port=0, limit=_MAX_LINE
        )
        self.port = self._server.sockets[0].getsockname()[1]

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            hello = await asyncio.wait_for(reader.readline(), timeout=10)
            data = json.loads(hello or b"{}")
            if not hmac.compare_digest(str(data.get("token", "")), self.token):
                writer.close()
                return
        except (asyncio.TimeoutError, ValueError, OSError):
            writer.close()
            return
        if self._writer is not None:
            self._writer.close()
        self._writer = writer
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if isinstance(event, dict):
                    result = self._on_event(event)
                    if asyncio.iscoroutine(result):
                        await result
        except (OSError, asyncio.IncompleteReadError, ValueError):
            pass
        finally:
            if self._writer is writer:
                self._writer = None
                if self._on_disconnect is not None:
                    self._on_disconnect()
            writer.close()

    async def send(self, command: dict[str, Any]) -> bool:
        """Send *command* to the connected child; ``False`` when none is connected."""
        writer = self._writer
        if writer is None or writer.is_closing():
            return False
        try:
            writer.write(_dumps(command))
            await writer.drain()
            return True
        except OSError:
            return False

    async def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
