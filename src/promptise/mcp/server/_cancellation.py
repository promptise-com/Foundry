"""Cancellation token for interruptible MCP tool calls.

Provides a ``CancellationToken`` that tool handlers can check during
long-running operations.  When the MCP client sends a cancellation
notification, the token is marked as cancelled and the handler gets the
server's ``cancel_grace_period`` to stop on its own before its task is
cancelled.  ``MCPQueue`` sets a job's token when the job is cancelled.

Example::

    from promptise.mcp.server import MCPServer, Depends, CancellationToken

    server = MCPServer(name="search")

    @server.tool()
    async def deep_search(
        query: str,
        cancel: CancellationToken = Depends(CancellationToken),
    ) -> list[dict]:
        results = []
        for batch in paginate(query):
            cancel.check()  # Raises CancelledError if cancelled
            results.extend(await process(batch))
        return results
"""

from __future__ import annotations

import asyncio


class CancelledError(Exception):
    """Raised when a tool call is cancelled by the client."""

    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason
        msg = "Tool call cancelled"
        if reason:
            msg += f": {reason}"
        super().__init__(msg)


class CancellationToken:
    """Token for checking and responding to client cancellation requests.

    The framework sets the token to cancelled state when the MCP client
    sends a ``notifications/cancelled`` notification.  Tool handlers can
    poll this token during long-running operations.

    Use via dependency injection::

        @server.tool()
        async def process(cancel: CancellationToken = Depends(CancellationToken)):
            for item in items:
                cancel.check()  # raises CancelledError if cancelled
                await do_work(item)
    """

    def __init__(self) -> None:
        self._cancelled = False
        self._reason: str | None = None
        self._event: asyncio.Event | None = None

    @property
    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        return self._cancelled

    @property
    def reason(self) -> str | None:
        """The cancellation reason, if any."""
        return self._reason

    def cancel(self, reason: str | None = None) -> None:
        """Mark this token as cancelled.

        Called by the framework when a client cancellation is received.
        """
        self._cancelled = True
        self._reason = reason
        if self._event is not None:
            self._event.set()

    def check(self) -> None:
        """Check if cancelled and raise ``CancelledError`` if so.

        This is the primary API for handlers — call it at safe points
        in long-running loops.
        """
        if self._cancelled:
            raise CancelledError(self._reason)

    async def wait(self, timeout: float | None = None) -> bool:
        """Wait until cancelled or timeout expires.

        Returns ``True`` if cancelled, ``False`` if timed out.
        """
        if self._cancelled:
            return True
        if self._event is None:
            self._event = asyncio.Event()
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False
