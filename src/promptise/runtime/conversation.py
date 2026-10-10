"""Short-term conversation memory for agent processes.

:class:`ConversationBuffer` maintains a rolling window of message
exchanges across trigger invocations within a single process lifecycle.
Messages are lost on process restart — persistent memory across restarts
is handled by long-term :class:`~promptise.memory.MemoryProvider`.

Thread-safe for concurrent access via an internal :class:`asyncio.Lock`
(acquired only in async helpers; synchronous methods remain lock-free
for use outside async contexts such as tests and serialization).

Example::

    buffer = ConversationBuffer(max_messages=50)
    buffer.append({"role": "user", "content": "Hello!"})
    buffer.append({"role": "assistant", "content": "Hi there!"})

    # Next invocation sees the previous exchange
    history = buffer.get_messages()
    # [{"role": "user", ...}, {"role": "assistant", ...}]
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationBuffer:
    """Rolling buffer of conversation messages (short-term memory).

    Maintains the last *max_messages* message exchanges across
    invocations within a single process lifecycle.  Oldest messages
    are evicted when the buffer is full.

    The buffer is safe for concurrent async access: the async helpers
    :meth:`async_snapshot` and :meth:`async_replace` use an internal
    :class:`asyncio.Lock`.  The synchronous methods (:meth:`append`,
    :meth:`get_messages`, etc.) are lock-free and intended for single-
    threaded or test use.

    Args:
        max_messages: Maximum messages to retain.  Oldest are evicted.
            ``0`` means unlimited.
        enabled: ``False`` turns the buffer off: nothing is retained, so
            each invocation starts without history.
    """

    max_messages: int = 100
    enabled: bool = True
    _messages: deque[dict[str, Any]] = field(
        default_factory=lambda: deque(),
        repr=False,
    )
    _lock: asyncio.Lock = field(
        default_factory=asyncio.Lock,
        repr=False,
    )

    def append(self, message: dict[str, Any]) -> None:
        """Add a single message to the buffer.

        Args:
            message: Message dict with ``role`` and ``content`` keys.
        """
        self._messages.append(message)
        self._evict_if_needed()

    def extend(self, messages: list[dict[str, Any]]) -> None:
        """Add multiple messages to the buffer.

        Args:
            messages: List of message dicts.
        """
        self._messages.extend(messages)
        self._evict_if_needed()

    def get_messages(self) -> list[dict[str, Any]]:
        """Return all buffered messages as a list.

        Returns:
            Ordered list of message dicts (oldest first).
        """
        return list(self._messages)

    def set_messages(self, messages: list[dict[str, Any]]) -> None:
        """Replace the buffer contents.

        Used during hot-reload to preserve conversation across
        agent rebuilds.

        Args:
            messages: New message list to set.
        """
        self._messages = deque(messages)
        self._evict_if_needed()

    def clear(self) -> None:
        """Clear all messages from the buffer."""
        self._messages.clear()

    # ------------------------------------------------------------------
    # Async-safe helpers (for concurrent process access)
    # ------------------------------------------------------------------

    async def async_snapshot(self) -> list[dict[str, Any]]:
        """Thread-safe snapshot of all messages (async).

        Returns:
            Copy of the current message list.
        """
        async with self._lock:
            return list(self._messages)

    async def async_replace(self, messages: list[dict[str, Any]]) -> None:
        """Thread-safe replacement of the entire buffer (async).

        Args:
            messages: New message list.
        """
        async with self._lock:
            self._messages = deque(messages)
            self._evict_if_needed()

    async def async_append(self, message: dict[str, Any]) -> None:
        """Thread-safe append (async).

        Args:
            message: Message dict to append.
        """
        async with self._lock:
            self._messages.append(message)
            self._evict_if_needed()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        """Remove oldest messages if over capacity (all of them when disabled)."""
        if not self.enabled:
            self._messages.clear()
            return
        while self.max_messages > 0 and len(self._messages) > self.max_messages:
            self._messages.popleft()

    def to_dict(self) -> dict[str, Any]:
        """Serialize for journal checkpointing.

        Returns:
            Dict with ``max_messages`` and ``messages`` keys.
        """
        return {
            "max_messages": self.max_messages,
            "enabled": self.enabled,
            "messages": list(self._messages),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationBuffer:
        """Reconstruct from a serialized dict.

        Args:
            data: Dict produced by :meth:`to_dict`.

        Returns:
            Restored :class:`ConversationBuffer` instance.
        """
        buf = cls(
            max_messages=data.get("max_messages", 100),
            enabled=data.get("enabled", True),
        )
        buf._messages = deque(data.get("messages", []))
        return buf

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"ConversationBuffer(messages={len(self._messages)}, max={self.max_messages})"
