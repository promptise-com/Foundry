"""AutoApprovalClassifier — explicit decision hierarchy for approval requests.

Wraps an upstream :class:`~promptise.approval.ApprovalHandler` with a
deterministic, ordered policy for deciding when to auto-allow, when to
auto-deny, and when to escalate to the human handler.

The five-step hierarchy (in priority order):

1. **Explicit allow rules** — patterns / predicates that always allow.
   First match wins.
2. **Explicit deny rules** — patterns / predicates that always deny.
   First match wins.
3. **Read-only auto-allow** — if the tool name matches a list of
   read-only prefixes (``get_*``, ``list_*``, ``read_*``, ``search_*``,
   ``find_*``) and the rule is enabled, allow without prompting.
4. **LLM classifier** — optional async function that returns
   ``(decision, reason)``. Use this for fuzzy decisions: "is this
   destructive?", "is this user-data-leaking?". Cheap, fast, and only
   runs when the rule-based steps don't match.
5. **Fallback handler** — the wrapped ApprovalHandler. The classifier
   sends the request to a human (webhook, queue, callback) as the last
   resort.

Every decision is recorded in :attr:`stats` so you can audit which
layer fired. The classifier is fully async, thread-safe, and exposes
the same protocol as any other ``ApprovalHandler``, so dropping it
into an existing ``ApprovalPolicy`` is a one-line change::

    from promptise import ApprovalPolicy, WebhookApprovalHandler
    from promptise.approval_classifier import (
        AutoApprovalClassifier,
        ApprovalRule,
    )

    classifier = AutoApprovalClassifier(
        allow_rules=[
            ApprovalRule(tool="get_*"),
            ApprovalRule(tool="list_*"),
        ],
        deny_rules=[
            ApprovalRule(tool="delete_*"),
            ApprovalRule(tool="exec_shell"),
        ],
        read_only_auto_allow=True,
        fallback=WebhookApprovalHandler(url="https://approvals.local/api"),
    )

    policy = ApprovalPolicy(
        tools=["*"],
        handler=classifier,
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Literal

from .approval import ApprovalDecision, ApprovalHandler, ApprovalRequest

logger = logging.getLogger("promptise.approval.classifier")


#: Default tool-name prefixes considered "read-only" — safe to auto-allow.
DEFAULT_READ_ONLY_PREFIXES: tuple[str, ...] = (
    "get_",
    "list_",
    "read_",
    "search_",
    "find_",
    "fetch_",
    "describe_",
    "show_",
    "view_",
    "lookup_",
    "query_",
    "head_",
    "stat_",
    "exists_",
    "count_",
)


@dataclass
class ApprovalRule:
    """A single allow- or deny- rule for the classifier.

    A rule matches a request if **all** of its non-empty filters
    match. Empty filters are wildcards.

    Attributes:
        tool: Glob pattern matched against ``request.tool_name``.
            Empty string disables this filter.
        argument_contains: Substring that must appear somewhere in
            the JSON-serialized arguments. Useful for blocking
            ``rm -rf /``-style commands. Empty string disables this
            filter.
        user: Optional user identifier (from ``CallerContext``)
            this rule applies to. Empty matches any user.
        predicate: Optional async callable
            ``(request) -> bool`` for custom logic. Empty disables.
        reason: Human-readable reason recorded with the decision.
    """

    tool: str = ""
    argument_contains: str = ""
    user: str = ""
    predicate: Callable[[ApprovalRequest], Awaitable[bool]] | None = None
    reason: str = ""

    async def matches(self, request: ApprovalRequest) -> bool:
        """Return True if this rule applies to ``request``."""
        if self.tool and not fnmatch(request.tool_name, self.tool):
            return False
        if self.user and request.caller_user_id != self.user:
            return False
        if self.argument_contains:
            try:
                blob = str(request.arguments)
            except Exception:  # noqa: BLE001
                blob = ""
            if self.argument_contains not in blob:
                return False
        if self.predicate is not None:
            try:
                if not await self.predicate(request):
                    return False
            except Exception:  # noqa: BLE001
                logger.exception("approval rule predicate raised; treating as no-match")
                return False
        return True


#: Async callable that returns ``("allow" | "deny" | "escalate", reason)``
#: for a given approval request. Used by the classifier as step 4 of the
#: decision hierarchy. Designed to be implemented by an LLM call, but
#: any logic that returns those three outcomes works.
LLMClassifierFn = Callable[
    [ApprovalRequest], Awaitable[tuple[Literal["allow", "deny", "escalate"], str]]
]


@dataclass
class ClassifierStats:
    """Cumulative counts of decisions by hierarchy layer.

    Read this any time to see which layer is doing the most work —
    a hint that you may want to tune your rules.
    """

    allow_rule_hits: int = 0
    deny_rule_hits: int = 0
    read_only_allows: int = 0
    llm_allows: int = 0
    llm_denies: int = 0
    llm_escalations: int = 0
    fallback_allows: int = 0
    fallback_denies: int = 0
    errors: int = 0
    last_updated: float = field(default_factory=time.time)


@dataclass
class ClassifierDecisionTrace:
    """Diagnostic record returned alongside an :class:`ApprovalDecision`.

    Useful for audit logs that need to know *why* a request was
    approved or denied.
    """

    layer: Literal[
        "allow_rule",
        "deny_rule",
        "read_only",
        "llm_allow",
        "llm_deny",
        "llm_escalate_then_fallback",
        "fallback",
        "error",
    ]
    rule_reason: str = ""
    matched_rule: ApprovalRule | None = None


class AutoApprovalClassifier:
    """Decision-hierarchy wrapper for approval handlers.

    Drop-in replacement for an :class:`ApprovalHandler`. Inspect each
    request through five ordered layers (allow rules → deny rules →
    read-only → LLM → fallback) and return the first definitive
    decision. The wrapped fallback handler is only called when no
    earlier layer fires.

    Args:
        allow_rules: Rules that always allow when matched. First
            match wins. Defaults to empty.
        deny_rules: Rules that always deny when matched. First match
            wins. Defaults to empty.
        read_only_auto_allow: If True (default), tool names that
            start with one of :data:`DEFAULT_READ_ONLY_PREFIXES` (or
            the override below) are auto-allowed. Set False to
            disable this layer.
        read_only_prefixes: Override the default read-only prefix
            list. Pass a tuple/list of strings.
        llm_classifier: Optional async callable taking an
            ``ApprovalRequest`` and returning ``(verdict, reason)``,
            where verdict is one of ``"allow"``, ``"deny"``,
            ``"escalate"``. ``escalate`` defers to the fallback
            handler.
        fallback: The :class:`ApprovalHandler` to delegate to as a
            last resort. Required.
        reviewer_id: Identifier recorded on auto-decisions. Defaults
            to ``"auto-classifier"``.

    Example::

        classifier = AutoApprovalClassifier(
            allow_rules=[ApprovalRule(tool="get_*", reason="read-only")],
            deny_rules=[ApprovalRule(tool="exec_shell", reason="too risky")],
            read_only_auto_allow=True,
            fallback=QueueApprovalHandler(...),
        )
    """

    def __init__(
        self,
        *,
        allow_rules: list[ApprovalRule] | None = None,
        deny_rules: list[ApprovalRule] | None = None,
        read_only_auto_allow: bool = True,
        read_only_prefixes: tuple[str, ...] | list[str] | None = None,
        llm_classifier: LLMClassifierFn | None = None,
        fallback: ApprovalHandler,
        reviewer_id: str = "auto-classifier",
    ) -> None:
        if fallback is None:
            raise ValueError("AutoApprovalClassifier requires a fallback handler")
        self._allow_rules = list(allow_rules or [])
        self._deny_rules = list(deny_rules or [])
        self._read_only_auto_allow = read_only_auto_allow
        self._read_only_prefixes: tuple[str, ...] = tuple(
            read_only_prefixes or DEFAULT_READ_ONLY_PREFIXES
        )
        self._llm_classifier = llm_classifier
        self._fallback = fallback
        self._reviewer_id = reviewer_id

        self.stats = ClassifierStats()
        self._last_trace: ClassifierDecisionTrace | None = None
        self._lock = asyncio.Lock()

    @property
    def fallback(self) -> ApprovalHandler:
        """The human handler that receives requests no rule decides.

        Server-side approval gates reached through MCP elicitation go
        straight here (see :func:`~promptise.approval.approval_elicitation_callback`).
        """
        return self._fallback

    # -- Protocol method --

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Run the decision hierarchy and return a definitive decision.

        Implements the :class:`ApprovalHandler` protocol so this
        object can be plugged directly into an
        :class:`~promptise.approval.ApprovalPolicy`.
        """
        try:
            return await self._classify(request)
        except Exception:  # noqa: BLE001
            async with self._lock:
                self.stats.errors += 1
                self.stats.last_updated = time.time()
            self._last_trace = ClassifierDecisionTrace(layer="error")
            logger.exception("AutoApprovalClassifier raised; falling back to handler")
            return await self._fallback.request_approval(request)

    async def _classify(self, request: ApprovalRequest) -> ApprovalDecision:
        # 1. Allow rules
        for rule in self._allow_rules:
            if await rule.matches(request):
                async with self._lock:
                    self.stats.allow_rule_hits += 1
                    self.stats.last_updated = time.time()
                self._last_trace = ClassifierDecisionTrace(
                    layer="allow_rule",
                    rule_reason=rule.reason,
                    matched_rule=rule,
                )
                logger.debug(
                    "approval allow rule matched: tool=%s reason=%s",
                    request.tool_name,
                    rule.reason,
                )
                return ApprovalDecision(
                    approved=True,
                    reviewer_id=self._reviewer_id,
                    reason=rule.reason or "matched allow rule",
                )

        # 2. Deny rules
        for rule in self._deny_rules:
            if await rule.matches(request):
                async with self._lock:
                    self.stats.deny_rule_hits += 1
                    self.stats.last_updated = time.time()
                self._last_trace = ClassifierDecisionTrace(
                    layer="deny_rule",
                    rule_reason=rule.reason,
                    matched_rule=rule,
                )
                logger.info(
                    "approval deny rule matched: tool=%s reason=%s",
                    request.tool_name,
                    rule.reason,
                )
                return ApprovalDecision(
                    approved=False,
                    reviewer_id=self._reviewer_id,
                    reason=rule.reason or "matched deny rule",
                )

        # 3. Read-only auto-allow
        if self._read_only_auto_allow and self._is_read_only(request.tool_name):
            async with self._lock:
                self.stats.read_only_allows += 1
                self.stats.last_updated = time.time()
            self._last_trace = ClassifierDecisionTrace(layer="read_only")
            logger.debug("approval read-only auto-allow: %s", request.tool_name)
            return ApprovalDecision(
                approved=True,
                reviewer_id=self._reviewer_id,
                reason="read-only tool auto-allowed",
            )

        # 4. LLM classifier
        if self._llm_classifier is not None:
            verdict, reason = await self._llm_classifier(request)
            if verdict == "allow":
                async with self._lock:
                    self.stats.llm_allows += 1
                    self.stats.last_updated = time.time()
                self._last_trace = ClassifierDecisionTrace(layer="llm_allow", rule_reason=reason)
                return ApprovalDecision(
                    approved=True,
                    reviewer_id=self._reviewer_id,
                    reason=reason or "llm classifier allow",
                )
            if verdict == "deny":
                async with self._lock:
                    self.stats.llm_denies += 1
                    self.stats.last_updated = time.time()
                self._last_trace = ClassifierDecisionTrace(layer="llm_deny", rule_reason=reason)
                return ApprovalDecision(
                    approved=False,
                    reviewer_id=self._reviewer_id,
                    reason=reason or "llm classifier deny",
                )
            # verdict == "escalate" → fall through
            async with self._lock:
                self.stats.llm_escalations += 1
                self.stats.last_updated = time.time()
            self._last_trace = ClassifierDecisionTrace(
                layer="llm_escalate_then_fallback", rule_reason=reason
            )

        # 5. Fallback handler
        decision = await self._fallback.request_approval(request)
        async with self._lock:
            if decision.approved:
                self.stats.fallback_allows += 1
            else:
                self.stats.fallback_denies += 1
            self.stats.last_updated = time.time()
        if self._last_trace is None or self._last_trace.layer != "llm_escalate_then_fallback":
            self._last_trace = ClassifierDecisionTrace(layer="fallback")
        return decision

    # -- Helpers --

    def _is_read_only(self, tool_name: str) -> bool:
        return any(tool_name.startswith(p) for p in self._read_only_prefixes)

    @property
    def last_trace(self) -> ClassifierDecisionTrace | None:
        """Diagnostic trace from the most recent decision."""
        return self._last_trace

    def reset_stats(self) -> None:
        """Zero out :attr:`stats`. Useful between runs / tests."""
        self.stats = ClassifierStats()


__all__ = [
    "ApprovalRule",
    "AutoApprovalClassifier",
    "ClassifierDecisionTrace",
    "ClassifierStats",
    "DEFAULT_READ_ONLY_PREFIXES",
    "LLMClassifierFn",
]
