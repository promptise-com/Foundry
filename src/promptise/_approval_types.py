"""Core approval data types, shared by :mod:`promptise.approval` and
:mod:`promptise.approval_classifier`.

They live here, not in either public module, so neither has to import the
other to name them: ``ApprovalDecision.trace`` is a
:class:`ClassifierDecisionTrace` and the classifier's rules take an
:class:`ApprovalRequest`, which used to make the two modules import each
other.  Import them from the public modules; this one is private.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac as _hmac_mod
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Literal, Protocol, runtime_checkable

# The rule predicate's errors are logged under the classifier's logger, as before.
logger = logging.getLogger("promptise.approval.classifier")

#: The :class:`ApprovalRequest` fields the webhook signature covers.
SIGNED_FIELDS: tuple[str, ...] = (
    "request_id",
    "tool_name",
    "arguments",
    "agent_id",
    "caller_user_id",
    "timestamp",
)


def _signature_payload(fields: Mapping[str, Any]) -> bytes:
    """The bytes ``X-Promptise-Signature`` is the HMAC-SHA256 of."""
    return json.dumps(
        {name: fields.get(name) for name in SIGNED_FIELDS}, sort_keys=True, default=str
    ).encode()


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ApprovalRequest:
    """A request for human approval of a tool call.

    Attributes:
        request_id: Unique cryptographic ID for this request.
        tool_name: Name of the tool requiring approval.
        arguments: Tool arguments (redacted if configured).
        agent_id: Agent or process identifier: the agent identity's id or
            ``build_agent(observer_agent_id=...)``; in the Agent Runtime,
            the process name.  ``None`` when the agent has neither.
        caller_user_id: User who triggered the agent (from CallerContext).
        context_summary: The last few user/assistant messages of the
            conversation, for reviewer context (redacted like the
            arguments).  See ``ApprovalPolicy(context_messages=...)``.
        timestamp: When the request was created (``time.time()``).
        timeout: Seconds until auto-deny/allow.
        metadata: ``source`` (``"agent"`` for the agent's own gate),
            ``session_id`` and ``tenant_id`` when known, plus whatever
            ``ApprovalPolicy(metadata=...)`` adds.
        tool_annotations: The tool's MCP annotations (``readOnlyHint``,
            ``destructiveHint``, ``idempotentHint``, ``openWorldHint``,
            ``title``) when it has them; empty otherwise.  Hints from the
            server, not guarantees.
        raw_arguments: The unredacted arguments, set by the agent's gate
            for in-process rule evaluation
            (:class:`~promptise.approval_classifier.AutoApprovalClassifier`
            matches its rules against them).  Never serialized, signed or
            shown in ``repr()``; handlers that display or log requests use
            ``arguments``.  ``None`` outside the agent's gate.
    """

    request_id: str
    tool_name: str
    arguments: dict[str, Any]
    agent_id: str | None = None
    caller_user_id: str | None = None
    context_summary: str = ""
    timestamp: float = field(default_factory=time.time)
    timeout: float = 300.0
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_annotations: dict[str, Any] = field(default_factory=dict)
    raw_arguments: dict[str, Any] | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (for webhook payloads)."""
        return {
            "request_id": self.request_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "agent_id": self.agent_id,
            "caller_user_id": self.caller_user_id,
            "context_summary": self.context_summary,
            "timestamp": self.timestamp,
            "timeout": self.timeout,
            "metadata": self.metadata,
            "tool_annotations": self.tool_annotations,
        }

    def compute_hmac(self, secret: str) -> str:
        """Compute the HMAC-SHA256 signature sent as ``X-Promptise-Signature``.

        Covers the fields in :data:`SIGNED_FIELDS` (``request_id``,
        ``tool_name``, ``arguments``, ``agent_id``, ``caller_user_id``,
        ``timestamp``), serialized as sorted-key JSON.  Verify it on the
        receiving side with :func:`verify_webhook_signature`.
        """
        payload = _signature_payload(
            {name: getattr(self, name) for name in SIGNED_FIELDS},
        )
        return _hmac_mod.new(secret.encode(), payload, hashlib.sha256).hexdigest()


@dataclass
class ApprovalDecision:
    """The decision on an approval request.

    Attributes:
        approved: Whether the tool call is approved.
        modified_arguments: If the reviewer edited arguments.
        reviewer_id: Who made the decision.
        reason: Optional explanation.
        timestamp: When the decision was made.
        decided_by: What made the decision: ``"reviewer"`` (the handler —
            a person, or your own handler code; the default),
            ``"classifier"`` (an
            :class:`~promptise.approval_classifier.AutoApprovalClassifier`
            rule, its read-only check or its LLM classifier) or ``"gate"``
            (the agent's approval gate itself: ``max_pending``, the
            ``max_retries_after_deny`` limit, a timeout or a handler
            error).  Only ``"reviewer"`` denials (and timeouts) count
            towards ``max_retries_after_deny``.
        trace: The classifier layer that produced the decision, when an
            :class:`~promptise.approval_classifier.AutoApprovalClassifier`
            was involved; ``None`` otherwise.
    """

    approved: bool
    modified_arguments: dict[str, Any] | None = None
    reviewer_id: str | None = None
    reason: str | None = None
    timestamp: float = field(default_factory=time.time)
    decided_by: Literal["reviewer", "classifier", "gate"] = "reviewer"
    trace: ClassifierDecisionTrace | None = None


# ---------------------------------------------------------------------------
# Handler protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class ApprovalHandler(Protocol):
    """Protocol for approval handlers.

    Implementations receive an :class:`ApprovalRequest` and must return
    an :class:`ApprovalDecision`.  The handler is async — it can await
    webhooks, poll APIs, or wait on queues.
    """

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Ask for a decision on *request* and return it."""


# It lives here, next to the protocol, so that approval_classifier can wrap a
# function fallback without importing promptise.approval (an import cycle);
# promptise.approval re-exports it.
class CallbackApprovalHandler:
    """Approval handler that delegates to an async Python callable.

    The simplest handler — pass any ``async def handler(request) -> decision``
    function and it will be called for each approval request.

    Args:
        callback: Async callable that receives an :class:`ApprovalRequest`
            and returns an :class:`ApprovalDecision`.
    """

    def __init__(self, callback: Callable[..., Any]) -> None:
        self._callback = callback

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Delegate to the user-provided callback."""
        result = self._callback(request)
        if asyncio.iscoroutine(result) or asyncio.isfuture(result):
            result = await result
        if isinstance(result, ApprovalDecision):
            return result
        # Allow returning a plain bool for convenience
        if isinstance(result, bool):
            return ApprovalDecision(approved=result)
        raise TypeError(
            f"Approval callback must return ApprovalDecision or bool, got {type(result).__name__}"
        )


def _arguments_json(arguments: Any) -> str:
    """The text ``ApprovalRule.argument_contains`` is matched against."""
    return json.dumps(arguments, sort_keys=True, default=str, ensure_ascii=False)


@dataclass
class ApprovalRule:
    """A single deny, ask or allow rule for the classifier.

    A rule matches a request if **all** of its non-empty filters
    match. Empty filters are wildcards.

    Attributes:
        tool: Glob pattern matched against ``request.tool_name``.
            Empty string disables this filter.
        argument_contains: Substring that must appear in the arguments
            serialized as JSON with sorted keys —
            ``json.dumps(arguments, sort_keys=True, ensure_ascii=False)``,
            so ``'"force": true'`` (not ``"'force': True"``).  Matching is
            case-sensitive.  Useful for blocking ``rm -rf /``-style
            commands. Empty string disables this filter.
        user: Optional user identifier (from ``CallerContext``)
            this rule applies to. Empty matches any user.
        predicate: Optional async callable
            ``(request) -> bool`` for custom logic. Empty disables.
            Inside an :class:`~promptise.approval_classifier.AutoApprovalClassifier`
            a predicate that raises never lets a call through: a raising
            deny rule denies the call, and a raising ask or allow rule
            sends it to the human fallback.  :meth:`matches` on its own
            treats the error as no match (logged).
        reason: Human-readable reason recorded with the decision.

    The filters see the call's real arguments: the classifier hands
    rules a request whose ``arguments`` are ``raw_arguments`` when the
    agent's gate set them, so redaction for the reviewer
    (``ApprovalPolicy(redact_sensitive=True)``) doesn't hide an email
    address from ``argument_contains="@competitor.example"``.
    """

    tool: str = ""
    argument_contains: str = ""
    user: str = ""
    predicate: Callable[[ApprovalRequest], Awaitable[bool]] | None = None
    reason: str = ""

    async def matches(self, request: ApprovalRequest) -> bool:
        """Return True if this rule applies to ``request``.

        A rule that cannot be evaluated (its predicate raised, or the
        arguments could not be serialized for ``argument_contains``) counts
        as no match here, with the error logged.  The classifier does not
        use this lenient form: see :class:`RuleEvaluationError`.
        """
        try:
            return await self.evaluate(request)
        except RuleEvaluationError:
            logger.exception("approval rule could not be evaluated; treating as no-match")
            return False

    async def evaluate(self, request: ApprovalRequest) -> bool:
        """Like :meth:`matches`, but raise when the rule cannot be evaluated.

        Raises:
            RuleEvaluationError: The predicate raised, or the arguments
                could not be serialized for ``argument_contains``.  The
                original exception is the ``__cause__``.
        """
        if self.tool and not fnmatch(request.tool_name, self.tool):
            return False
        if self.user and request.caller_user_id != self.user:
            return False
        if self.argument_contains:
            try:
                blob = _arguments_json(request.arguments)
            except Exception as exc:  # noqa: BLE001
                raise RuleEvaluationError(self, exc) from exc
            if self.argument_contains not in blob:
                return False
        if self.predicate is not None:
            try:
                if not await self.predicate(request):
                    return False
            except Exception as exc:  # noqa: BLE001
                raise RuleEvaluationError(self, exc) from exc
        return True


class RuleEvaluationError(Exception):
    """An :class:`ApprovalRule` could not be evaluated.

    Raised by :meth:`ApprovalRule.evaluate` when the rule's predicate
    raises, or when the arguments cannot be serialized for
    ``argument_contains``.  The classifier fails closed on it: a deny rule
    that cannot be evaluated denies the call.

    Attributes:
        rule: The rule that failed.
        error: The exception it raised.
    """

    def __init__(self, rule: ApprovalRule, error: BaseException) -> None:
        self.rule = rule
        self.error = error
        label = f" ({rule.reason})" if rule.reason else ""
        super().__init__(f"approval rule{label} raised {type(error).__name__}: {error}")


@dataclass
class ClassifierDecisionTrace:
    """Diagnostic record of which layer decided a request.

    Attached to every decision the classifier returns as
    ``ApprovalDecision.trace``, so audit code can read it from the
    decision itself — concurrent requests can't overwrite it.

    Attributes:
        layer: The layer that decided.  ``"ask_rule"``,
            ``"llm_escalate_then_fallback"``, ``"fallback"`` and
            ``"error"`` mean the fallback handler made the decision.
        rule_reason: The matched rule's ``reason``, or the LLM
            classifier's reason.
        matched_rule: The rule that matched, for rule layers.
    """

    layer: Literal[
        "deny_rule",
        "ask_rule",
        "allow_rule",
        "read_only",
        "llm_allow",
        "llm_deny",
        "llm_escalate_then_fallback",
        "fallback",
        "error",
    ]
    rule_reason: str = ""
    matched_rule: ApprovalRule | None = None
