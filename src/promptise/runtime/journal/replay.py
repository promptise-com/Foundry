"""Replay engine: reconstruct process state from journal.

Used for crash recovery — reads the journal, finds the last checkpoint,
and replays subsequent entries to rebuild the process state.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import JournalEntry, JournalProvider

logger = logging.getLogger(__name__)


class ReplayEngine:
    """Replays journal entries to reconstruct process state.

    Args:
        journal: :class:`JournalProvider` to read from.

    Example::

        engine = ReplayEngine(journal)
        recovered = await engine.recover("process-1")
        # recovered = {
        #     "context_state": {...},
        #     "lifecycle_state": "running",
        #     "last_entry_type": "invocation_result",
        # }
    """

    def __init__(self, journal: JournalProvider) -> None:
        self._journal = journal

    async def recover(self, process_id: str) -> dict[str, Any]:
        """Recover process state from journal.

        1. Load the last checkpoint (if any).
        2. Read all entries after the checkpoint.
        3. Replay state transitions and context mutations.

        Args:
            process_id: Process to recover.

        Returns:
            Dict with ``context_state``, ``lifecycle_state``,
            ``last_entry_type``, and ``entries_replayed``.
        """
        # 1. Get last checkpoint
        checkpoint = await self._journal.last_checkpoint(process_id)
        context_state: dict[str, Any] = {}
        lifecycle_state: str = "created"

        if checkpoint:
            # Copy: replay mutates it, and the backend may hand out its own dict.
            context_state = dict(checkpoint.get("context_state") or {})
            lifecycle_state = checkpoint.get("lifecycle_state", "running")
            logger.info(
                "Replay: loaded checkpoint for %s (state=%s)",
                process_id,
                lifecycle_state,
            )

        # 2. Read entries after checkpoint
        all_entries = await self._journal.read(process_id)

        # Replay only what follows the LAST checkpoint entry: earlier entries
        # (including older checkpoints) are already folded into the snapshot,
        # and replaying them would overwrite newer state with stale values.
        entries_to_replay: list[JournalEntry]
        if checkpoint is None:
            entries_to_replay = list(all_entries)
        else:
            last_cp = max(
                (i for i, e in enumerate(all_entries) if e.entry_type == "checkpoint"),
                default=None,
            )
            entries_to_replay = [] if last_cp is None else all_entries[last_cp + 1 :]

        # 3. Replay
        last_entry_type = ""
        for entry in entries_to_replay:
            last_entry_type = entry.entry_type

            if entry.entry_type == "state_transition":
                new_state = entry.data.get("to_state", "")
                if new_state:
                    lifecycle_state = new_state

            elif entry.entry_type == "context_update":
                key = entry.data.get("key")
                value = entry.data.get("value")
                if key is not None:
                    context_state[key] = value

        logger.info(
            "Replay: replayed %d entries for %s (final state=%s)",
            len(entries_to_replay),
            process_id,
            lifecycle_state,
        )

        return {
            "context_state": context_state,
            "lifecycle_state": lifecycle_state,
            "last_entry_type": last_entry_type,
            "entries_replayed": len(entries_to_replay),
        }
