"""AutoApprovalClassifier — explicit decision hierarchy for approval requests.

Wraps an upstream :class:`~promptise.approval.ApprovalHandler` with a
deterministic, ordered policy for deciding when to auto-allow, when to
auto-deny, and when to escalate to the human handler.

The six-step hierarchy (in priority order):

1. **Deny rules** — patterns / predicates that always deny.  First
   match wins.  Deny beats everything below it, so a broad allow rule
   can't approve a call a deny rule forbids.
2. **Ask rules** — patterns / predicates that always go to the human
   fallback, skipping every automatic layer below.
3. **Allow rules** — patterns / predicates that always allow.  First
   match wins.
4. **Read-only auto-allow** — the tool is read-only: its MCP
   annotations say ``readOnlyHint=True``, or its name starts with a
   read-only prefix (``get_``, ``list_``, ``read_``, ...).  Never for a
   tool annotated ``destructiveHint=True`` / ``readOnlyHint=False``, or
   one whose name contains a destructive verb (``fetch_and_purge_cache``,
   ``show_and_delete``).
5. **LLM classifier** — optional async function that returns
   ``(decision, reason)``. Use this for fuzzy decisions: "is this
   destructive?", "is this user-data-leaking?". Only runs when the
   rule-based steps don't match.
6. **Fallback handler** — the wrapped ApprovalHandler. The classifier
   sends the request to a human (webhook, queue, callback) as the last
   resort.

Rules, the read-only check and the LLM classifier see the call's real
arguments (``ApprovalRequest.raw_arguments``, set by the agent's gate);
the fallback receives the reviewer's copy, redacted when
``ApprovalPolicy(redact_sensitive=True)``.

Every decision carries its :class:`ClassifierDecisionTrace` in
``ApprovalDecision.trace`` and is counted in :attr:`stats`.  The
classifier exposes the same protocol as any other ``ApprovalHandler``, so
dropping it into an existing ``ApprovalPolicy`` is a one-line change::

    from promptise import ApprovalPolicy, WebhookApprovalHandler
    from promptise.approval_classifier import (
        AutoApprovalClassifier,
        ApprovalRule,
    )

    classifier = AutoApprovalClassifier(
        deny_rules=[
            ApprovalRule(tool="delete_*"),
            ApprovalRule(tool="exec_shell"),
        ],
        ask_rules=[ApprovalRule(tool="send_email")],
        allow_rules=[ApprovalRule(tool="add_ticket_note")],
        read_only_auto_allow=True,
        fallback=WebhookApprovalHandler(url="https://approvals.example.com/api"),
    )

    policy = ApprovalPolicy(
        tools=["*"],
        handler=classifier,
        on_decision=write_audit_line,  # every decision, with decision.trace
    )
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, Literal

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

#: Words that keep a tool out of the read-only layer when they appear
#: anywhere in its name (as a whole word: ``fetch_and_purge_cache``,
#: ``showAndDelete``), whatever its prefix or annotations say.
DEFAULT_DESTRUCTIVE_VERBS: tuple[str, ...] = (
    "add",
    "append",
    "approve",
    "assign",
    "ban",
    "cancel",
    "charge",
    "clear",
    "create",
    "deactivate",
    "del",
    "delete",
    "deploy",
    "destroy",
    "disable",
    "drop",
    "edit",
    "erase",
    "exec",
    "execute",
    "flush",
    "grant",
    "insert",
    "install",
    "kill",
    "merge",
    "migrate",
    "modify",
    "move",
    "overwrite",
    "patch",
    "pay",
    "publish",
    "purge",
    "push",
    "put",
    "reboot",
    "refund",
    "remove",
    "rename",
    "replace",
    "reset",
    "restart",
    "restore",
    "revoke",
    "rm",
    "rollback",
    "send",
    "set",
    "shutdown",
    "submit",
    "suspend",
    "terminate",
    "transfer",
    "truncate",
    "uninstall",
    "unlink",
    "update",
    "upload",
    "upsert",
    "wipe",
    "withdraw",
    "write",
)

_NAME_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def _name_words(tool_name: str) -> list[str]:
    """``"fetchAndPurge_cache"`` → ``["fetch", "and", "purge", "cache"]``."""
    return [word.lower() for word in _NAME_WORD.findall(tool_name)]


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
            A predicate that raises counts as no match (logged).
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
        """Return True if this rule applies to ``request``."""
        if self.tool and not fnmatch(request.tool_name, self.tool):
            return False
        if self.user and request.caller_user_id != self.user:
            return False
        if self.argument_contains:
            try:
                blob = _arguments_json(request.arguments)
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
#: for a given approval request. Used by the classifier as step 5 of the
#: decision hierarchy. Designed to be implemented by an LLM call, but
#: any logic that returns those three outcomes works.
LLMClassifierFn = Callable[
    [ApprovalRequest], Awaitable[tuple[Literal["allow", "deny", "escalate"], str]]
]


@dataclass
class ClassifierStats:
    """Cumulative counts of decisions by hierarchy layer.

    Read this any time to see which layer is doing the most work —
    a hint that you may want to tune your rules.  ``ask_rule_hits``
    counts requests an ask rule sent to the fallback; their outcome is
    also counted in ``fallback_allows`` / ``fallback_denies``.
    """

    allow_rule_hits: int = 0
    deny_rule_hits: int = 0
    ask_rule_hits: int = 0
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


class AutoApprovalClassifier:
    """Decision-hierarchy wrapper for approval handlers.

    Drop-in replacement for an :class:`ApprovalHandler`. Inspect each
    request through six ordered layers (deny rules → ask rules → allow
    rules → read-only → LLM → fallback) and return the first definitive
    decision. The wrapped fallback handler is only called when no
    earlier layer decides, or when an ask rule matches.

    Args:
        deny_rules: Rules that always deny when matched. Checked first,
            so a deny rule wins over every allow rule. First match wins.
        ask_rules: Rules that always send the request to the fallback
            handler (a human), even when an allow rule, the read-only
            check or the LLM classifier would have approved it.
        allow_rules: Rules that always allow when matched (unless a
            deny or ask rule matched first). First match wins.
        read_only_auto_allow: If True (default), read-only tools are
            auto-allowed: tools annotated ``readOnlyHint=True``, and
            unannotated tools whose name starts with one of
            :data:`DEFAULT_READ_ONLY_PREFIXES` (or the override below).
            A tool annotated ``destructiveHint=True`` or
            ``readOnlyHint=False``, or whose name contains one of
            ``destructive_verbs`` as a word, is never auto-allowed here.
            Set False to disable this layer.
        read_only_prefixes: Override the default read-only prefix
            list. Pass a tuple/list of strings.
        destructive_verbs: Override :data:`DEFAULT_DESTRUCTIVE_VERBS`,
            the words that keep a tool name out of the read-only layer.
        use_tool_annotations: Read the tool's MCP annotations
            (``ApprovalRequest.tool_annotations``) in the read-only
            layer.  Default True.  With False, only the name decides.
        llm_classifier: Optional async callable taking an
            ``ApprovalRequest`` and returning ``(verdict, reason)``,
            where verdict is one of ``"allow"``, ``"deny"``,
            ``"escalate"``. ``escalate`` defers to the fallback
            handler.  It receives the real (unredacted) arguments.
        fallback: The :class:`ApprovalHandler` to delegate to as a
            last resort. Required.  It receives the reviewer's copy of
            the request (``arguments`` redacted per the policy, no
            ``raw_arguments``).
        reviewer_id: Identifier recorded on auto-decisions. Defaults
            to ``"auto-classifier"``.

    Example::

        classifier = AutoApprovalClassifier(
            deny_rules=[ApprovalRule(tool="exec_shell", reason="too risky")],
            ask_rules=[ApprovalRule(tool="delete_*", reason="a person decides")],
            allow_rules=[ApprovalRule(tool="add_note", reason="internal note")],
            read_only_auto_allow=True,
            fallback=QueueApprovalHandler(...),
        )
    """

    def __init__(
        self,
        *,
        allow_rules: list[ApprovalRule] | None = None,
        deny_rules: list[ApprovalRule] | None = None,
        ask_rules: list[ApprovalRule] | None = None,
        read_only_auto_allow: bool = True,
        read_only_prefixes: tuple[str, ...] | list[str] | None = None,
        destructive_verbs: tuple[str, ...] | list[str] | None = None,
        use_tool_annotations: bool = True,
        llm_classifier: LLMClassifierFn | None = None,
        fallback: ApprovalHandler,
        reviewer_id: str = "auto-classifier",
    ) -> None:
        if fallback is None:
            raise ValueError("AutoApprovalClassifier requires a fallback handler")
        self._allow_rules = list(allow_rules or [])
        self._deny_rules = list(deny_rules or [])
        self._ask_rules = list(ask_rules or [])
        self._read_only_auto_allow = read_only_auto_allow
        self._read_only_prefixes: tuple[str, ...] = tuple(
            read_only_prefixes or DEFAULT_READ_ONLY_PREFIXES
        )
        self._destructive_verbs: frozenset[str] = frozenset(
            verb.lower()
            for verb in (
                destructive_verbs if destructive_verbs is not None else DEFAULT_DESTRUCTIVE_VERBS
            )
        )
        self._use_tool_annotations = use_tool_annotations
        self._llm_classifier = llm_classifier
        self._fallback = fallback
        self._reviewer_id = reviewer_id

        self.stats = ClassifierStats()
        self._last_trace: ClassifierDecisionTrace | None = None
        self._stats_lock = threading.Lock()

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
        :class:`~promptise.approval.ApprovalPolicy`.  The returned
        decision's ``trace`` says which layer decided.
        """
        # Rules and the LLM classifier judge the real call; the fallback
        # (a person, a webhook, a log) gets the reviewer's redacted copy.
        if request.raw_arguments is not None:
            rule_view = dataclasses.replace(request, arguments=dict(request.raw_arguments))
        else:
            rule_view = request
        reviewer_view = (
            request
            if isinstance(self._fallback, AutoApprovalClassifier)
            else dataclasses.replace(request, raw_arguments=None)
        )
        try:
            decision, trace = await self._classify(rule_view, reviewer_view)
        except Exception:  # noqa: BLE001
            self._count("errors")
            logger.exception("AutoApprovalClassifier raised; falling back to handler")
            trace = ClassifierDecisionTrace(layer="error")
            decision = await self._ask_fallback(reviewer_view, trace)
        self._last_trace = trace
        return decision

    async def _classify(
        self, request: ApprovalRequest, reviewer_view: ApprovalRequest
    ) -> tuple[ApprovalDecision, ClassifierDecisionTrace]:
        # 1. Deny rules
        for rule in self._deny_rules:
            if await rule.matches(request):
                self._count("deny_rule_hits")
                logger.info(
                    "approval deny rule matched: tool=%s reason=%s",
                    request.tool_name,
                    rule.reason,
                )
                return self._auto(
                    False,
                    rule.reason or "matched deny rule",
                    ClassifierDecisionTrace(
                        layer="deny_rule", rule_reason=rule.reason, matched_rule=rule
                    ),
                )

        # 2. Ask rules — straight to the human
        for rule in self._ask_rules:
            if await rule.matches(request):
                self._count("ask_rule_hits")
                logger.debug(
                    "approval ask rule matched: tool=%s reason=%s",
                    request.tool_name,
                    rule.reason,
                )
                trace = ClassifierDecisionTrace(
                    layer="ask_rule", rule_reason=rule.reason, matched_rule=rule
                )
                return await self._ask_fallback(reviewer_view, trace), trace

        # 3. Allow rules
        for rule in self._allow_rules:
            if await rule.matches(request):
                self._count("allow_rule_hits")
                logger.debug(
                    "approval allow rule matched: tool=%s reason=%s",
                    request.tool_name,
                    rule.reason,
                )
                return self._auto(
                    True,
                    rule.reason or "matched allow rule",
                    ClassifierDecisionTrace(
                        layer="allow_rule", rule_reason=rule.reason, matched_rule=rule
                    ),
                )

        # 4. Read-only auto-allow
        if self._read_only_auto_allow and self.is_read_only(request):
            self._count("read_only_allows")
            logger.debug("approval read-only auto-allow: %s", request.tool_name)
            return self._auto(
                True,
                "read-only tool auto-allowed",
                ClassifierDecisionTrace(layer="read_only"),
            )

        # 5. LLM classifier
        trace = ClassifierDecisionTrace(layer="fallback")
        if self._llm_classifier is not None:
            verdict, reason = await self._llm_classifier(request)
            if verdict == "allow":
                self._count("llm_allows")
                return self._auto(
                    True,
                    reason or "llm classifier allow",
                    ClassifierDecisionTrace(layer="llm_allow", rule_reason=reason),
                )
            if verdict == "deny":
                self._count("llm_denies")
                return self._auto(
                    False,
                    reason or "llm classifier deny",
                    ClassifierDecisionTrace(layer="llm_deny", rule_reason=reason),
                )
            # verdict == "escalate" → fall through
            self._count("llm_escalations")
            trace = ClassifierDecisionTrace(layer="llm_escalate_then_fallback", rule_reason=reason)

        # 6. Fallback handler
        return await self._ask_fallback(reviewer_view, trace), trace

    # -- Helpers --

    def _auto(
        self, approved: bool, reason: str, trace: ClassifierDecisionTrace
    ) -> tuple[ApprovalDecision, ClassifierDecisionTrace]:
        """A decision made by the classifier itself."""
        decision = ApprovalDecision(
            approved=approved,
            reviewer_id=self._reviewer_id,
            reason=reason,
            decided_by="classifier",
            trace=trace,
        )
        return decision, trace

    async def _ask_fallback(
        self, request: ApprovalRequest, trace: ClassifierDecisionTrace
    ) -> ApprovalDecision:
        """The fallback's decision, with *trace* attached."""
        decision = await self._fallback.request_approval(request)
        self._count("fallback_allows" if decision.approved else "fallback_denies")
        if not isinstance(decision, ApprovalDecision):
            return decision
        # A copy: the handler may hand out a shared decision object.
        return dataclasses.replace(decision, trace=trace)

    def _count(self, stat: str) -> None:
        with self._stats_lock:
            setattr(self.stats, stat, getattr(self.stats, stat) + 1)
            self.stats.last_updated = time.time()

    def is_read_only(self, request: ApprovalRequest) -> bool:
        """Whether the read-only layer would auto-allow *request*.

        Read-only means: no word of the tool name is a destructive verb,
        and either the tool's annotations say ``readOnlyHint=True`` or
        (with no ``readOnlyHint`` either way) the name starts with a
        read-only prefix.  ``destructiveHint=True`` or
        ``readOnlyHint=False`` always means "not read-only".
        """
        name = request.tool_name
        if any(word in self._destructive_verbs for word in _name_words(name)):
            return False
        annotations: Mapping[str, Any] = (
            request.tool_annotations if self._use_tool_annotations else {}
        )
        read_only_hint = annotations.get("readOnlyHint")
        if read_only_hint is False or annotations.get("destructiveHint") is True:
            return False
        if read_only_hint is True:
            return True
        return name.startswith(self._read_only_prefixes)

    @property
    def last_trace(self) -> ClassifierDecisionTrace | None:
        """Trace of the most recently *finished* decision.

        Shared by every request: with concurrent requests (several gated
        calls in one turn, or a request waiting on a human) it may belong
        to a different request than the one you just awaited.  Read
        ``decision.trace`` instead.
        """
        return self._last_trace

    def reset_stats(self) -> None:
        """Zero out :attr:`stats`. Useful between runs / tests."""
        self.stats = ClassifierStats()


__all__ = [
    "ApprovalRule",
    "AutoApprovalClassifier",
    "ClassifierDecisionTrace",
    "ClassifierStats",
    "DEFAULT_DESTRUCTIVE_VERBS",
    "DEFAULT_READ_ONLY_PREFIXES",
    "LLMClassifierFn",
]
