"""Answer an agent's approval requests from the terminal UI.

An agent whose :class:`~promptise.approval.ApprovalPolicy` uses a
:class:`~promptise.approval.QueueApprovalHandler` waits for someone to read its
queue and decide.  Under ``promptise run --tui`` :class:`ApprovalBridge` reads
that queue, publishes each request as an ``approval.pending`` run event, and
submits the decision the UI sends back (``{"cmd": "approval", ...}``),
publishing ``approval.resolved``.  Other handlers (webhooks, callbacks) are
left alone: they already have someone answering them.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
from typing import Any

from .events import EventBus

__all__ = ["ApprovalBridge", "queue_handler_of"]

logger = logging.getLogger("promptise.run")


def queue_handler_of(agent: Any) -> Any | None:
    """The :class:`QueueApprovalHandler` behind *agent*'s approval policy, or ``None``.

    Wrappers (the adaptive strategy's denial learner) are looked through.
    """
    from ..approval import QueueApprovalHandler

    policy = getattr(agent, "_approval", None)
    handler = getattr(policy, "handler", None)
    for _ in range(5):
        if handler is None or isinstance(handler, QueueApprovalHandler):
            return handler
        handler = getattr(handler, "_inner", None)
    return None


class ApprovalBridge:
    """Publishes queued approval requests and submits the UI's decisions."""

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        self._lock = threading.Lock()
        self._requests: dict[str, tuple[Any, asyncio.AbstractEventLoop]] = {}
        self._watched: set[int] = set()
        self._tasks: set[asyncio.Task[Any]] = set()

    def watch(self, agent: Any, label: str) -> bool:
        """Start reading *agent*'s approval queue (when it has one); ``True`` if it does."""
        handler = queue_handler_of(agent)
        if handler is None or id(handler) in self._watched:
            return handler is not None
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return False
        self._watched.add(id(handler))
        task = loop.create_task(self._consume(handler, label, loop), name="promptise-run-approvals")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def _consume(self, handler: Any, label: str, loop: asyncio.AbstractEventLoop) -> None:
        while True:
            request = await handler.request_queue.get()
            with self._lock:
                self._requests[request.request_id] = (handler, loop)
            data = request.to_dict()
            data.pop("raw_arguments", None)
            self.bus.publish(
                "approval.pending",
                agent=label,
                request=data,
                deadline=float(request.timestamp) + float(request.timeout),
            )

    def decide(
        self,
        request_id: str,
        *,
        approved: bool,
        reason: str | None = None,
        modified_arguments: dict[str, Any] | None = None,
    ) -> None:
        """Submit a decision (from any thread)."""
        with self._lock:
            found = self._requests.pop(request_id, None)
        if found is None:
            self.bus.publish(
                "approval.resolved",
                request_id=request_id,
                approved=None,
                reason="no such pending request",
                outcome="expired",
            )
            return
        handler, loop = found

        def submit() -> None:
            from ..approval import ApprovalDecision

            try:
                handler.submit_decision(
                    request_id,
                    ApprovalDecision(
                        approved=approved,
                        reason=reason,
                        reviewer_id="promptise-run",
                        modified_arguments=modified_arguments,
                    ),
                )
            except KeyError:
                outcome = "expired"
            else:
                outcome = "approved" if approved else "denied"
            self.bus.publish(
                "approval.resolved",
                request_id=request_id,
                approved=approved if outcome != "expired" else None,
                reason=reason,
                outcome=outcome,
            )

        if loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(submit)

    def handle_command(self, command: dict[str, Any]) -> bool:
        """Handle an ``approval`` command; ``True`` when it was one."""
        if command.get("cmd") != "approval":
            return False
        request_id = command.get("request_id")
        if not isinstance(request_id, str):
            return True
        args = command.get("modified_arguments")
        self.decide(
            request_id,
            approved=bool(command.get("approved")),
            reason=command.get("reason") if isinstance(command.get("reason"), str) else None,
            modified_arguments=args if isinstance(args, dict) else None,
        )
        return True
