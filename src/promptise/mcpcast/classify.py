"""Deterministic risk classification for ``promptise mcpcast``.

Every operation gets a :class:`~promptise.mcpcast.schema.RiskClass` from a
fixed, ordered rule set — no network, no model, no surprises:

1. ``GET`` / ``HEAD`` / ``OPTIONS`` / ``TRACE`` → ``read``
2. ``DELETE`` → ``destructive``
3. id / path / summary mention a destructive verb
   (``delete``, ``remove``, ``purge``, ``revoke``, ``terminate``, ``cancel``,
   ``deactivate``, …) → ``destructive``
4. …mention money (``charge``, ``payment``, ``refund``, ``invoice``,
   ``transfer``, ``payout``, ``subscription``, ``billing``, …) → ``financial``
5. a ``POST`` whose only verb is a query (``search``, ``query``, ``find``,
   ``lookup``, ``list``, ``validate``, …) and that mentions no mutating verb
   (``create``, ``update``, ``upload``, …) → ``read`` — ``POST /search`` is a
   read, ``POST /users/createWithList`` and ``POST /charges`` are not
6. ``POST`` / ``PUT`` / ``PATCH`` → ``write``

Then each *escalation signal* moves the result one step up the ladder:
an OAuth scope containing ``admin`` (or ``root`` / ``superuser``), a path
segment such as ``admin`` / ``internal``, or ``deprecated: true``.  Escalation is an
*exposure* decision, not a semantic one — a deprecated or admin-scoped
read should not be handed to an agent under ``read-only``.

The LLM curator may later *escalate* a class, never relax it — see
:mod:`promptise.mcpcast.curate`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parse import Operation
from .schema import RiskClass

__all__ = [
    "Classification",
    "DESTRUCTIVE_WORDS",
    "FINANCIAL_WORDS",
    "MUTATING_WORDS",
    "QUERY_WORDS",
    "SENSITIVE_PATH_WORDS",
    "classify",
    "classify_operation",
    "risk_floor",
]

DESTRUCTIVE_WORDS: tuple[str, ...] = (
    "delete",
    "remove",
    "purge",
    "revoke",
    "terminate",
    "cancel",
    "deactivate",
    "destroy",
    "erase",
    "wipe",
    "reset",
)

FINANCIAL_WORDS: tuple[str, ...] = (
    "charge",
    "payment",
    "pay",
    "refund",
    "invoice",
    "transfer",
    "payout",
    "subscription",
    "billing",
    "checkout",
    "purchase",
    "withdraw",
    "deposit",
)

QUERY_WORDS: tuple[str, ...] = (
    "search",
    "query",
    "find",
    "lookup",
    "list",
    "fetch",
    "preview",
    "validate",
    "calculate",
    "estimate",
)
# NOTE: "graphql" is deliberately absent — a GraphQL endpoint accepts mutations
# as readily as queries, so POST /graphql must not be classified as a read.

MUTATING_WORDS: tuple[str, ...] = (
    "create",
    "add",
    "new",
    "update",
    "edit",
    "modify",
    "set",
    "save",
    "upload",
    "import",
    "register",
    "submit",
    "send",
    "finalize",
    "confirm",
    "capture",
    "activate",
    "archive",
    "accept",
    "complete",
    "verify",
    "apply",
    "invite",
    "redeliver",
    "ping",
    "generate",
    "assign",
    "attach",
    "publish",
    "execute",
    "run",
    "trigger",
    "start",
    "stop",
    "restart",
    "move",
    "rename",
    "replace",
    "merge",
    "sync",
    "approve",
    "reject",
    "enable",
    "disable",
    "login",
    "logout",
)
"""Verbs that make a ``POST`` a write even when a query word is also present
(``POST /users/createWithList`` contains ``list`` but creates users)."""

SENSITIVE_PATH_WORDS: tuple[str, ...] = ("admin", "internal", "sudo", "impersonate")

_SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
_HTTP_VERB_TOKENS = frozenset({"get", "post", "put", "patch", "delete", "head", "options"})
# Only privilege markers escalate. A ``write:`` scope on a read is a spec
# artefact (Petstore declares ``write:pets`` on ``GET /pet/{id}``) — the
# operation is still a read, and gating every scoped write as destructive
# would hide ordinary writes behind the ``full`` profile.
_ESCALATING_SCOPE_MARKERS = ("admin", "root", "superuser")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SPLIT = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Classification:
    """The outcome of classifying one operation, with its reasoning."""

    risk: RiskClass
    base: RiskClass
    reasons: list[str] = field(default_factory=list)


def tokens(text: str) -> list[str]:
    """Lower-case word tokens from ids, paths and prose (camelCase-aware)."""
    spaced = _CAMEL.sub(" ", text)
    return [t for t in _SPLIT.split(spaced.lower()) if t]


def _matches(token: str, word: str, *, exact: bool) -> bool:
    if token == word or (token.endswith("s") and token[:-1] == word):
        return True
    # Stem-ish prefix match for verbs (cancel → cancellation, refund → refunded);
    # short words never prefix-match so ``pay`` cannot hit ``payload``.
    return not exact and len(word) >= 5 and token.startswith(word)


def _find(words: tuple[str, ...], toks: list[str], *, exact: bool = False) -> str | None:
    for tok in toks:
        for word in words:
            if _matches(tok, word, exact=exact):
                return word
    return None


def _query_signal(op: Operation, toks: list[str]) -> str | None:
    """A query verb that *leads* the operation: first token of the id or
    summary, or the last path segment — never buried in prose."""
    leading: list[str] = []
    for text in (op.operation_id, op.summary):
        words = tokens(text)
        if words and words[0] in _HTTP_VERB_TOKENS:  # generated ids: post_search → search
            words = words[1:]
        leading.extend(words[:1])
    last_segment = op.path.rstrip("/").rsplit("/", 1)[-1]
    if not last_segment.startswith("{"):
        leading.extend(tokens(last_segment))
    if _find(MUTATING_WORDS, toks) is not None:
        return None
    return _find(QUERY_WORDS, leading, exact=True)


def _base_class(op: Operation, toks: list[str]) -> tuple[RiskClass, str]:
    """Rules 1–6: the class before escalation, with the reason."""
    method = op.method.upper()
    if method in _SAFE_METHODS:
        return RiskClass.READ, f"{method} is a read"
    if method == "DELETE":
        return RiskClass.DESTRUCTIVE, "DELETE is destructive"
    if (word := _find(DESTRUCTIVE_WORDS, toks)) is not None:
        return RiskClass.DESTRUCTIVE, f"mentions destructive verb {word!r}"
    if (word := _find(FINANCIAL_WORDS, toks)) is not None:
        return RiskClass.FINANCIAL, f"mentions money ({word!r})"
    if method == "POST" and (word := _query_signal(op, toks)) is not None:
        return RiskClass.READ, f"POST that only queries ({word!r})"
    return RiskClass.WRITE, f"{method} is a write"


def _escalated(op: Operation, base: RiskClass, toks: list[str]) -> tuple[RiskClass, list[str]]:
    """*base* moved one step up the ladder per escalation signal, with the reasons."""
    method = op.method.upper()
    risk = base
    reasons: list[str] = []
    if method in _SAFE_METHODS:
        # A GET that says it deletes, revokes or cancels is a legacy
        # side-effecting read: expose it only where writes are allowed.
        # (Money nouns on a GET — listing invoices — are ordinary reads.)
        hit = _find(DESTRUCTIVE_WORDS, toks, exact=True)
        if hit is not None:
            risk = risk.escalate()
            reasons.append(f"escalated: {method} that appears to mutate ({hit!r})")
    scope_hit = next(
        (s for s in op.scopes if any(m in s.lower() for m in _ESCALATING_SCOPE_MARKERS)), None
    )
    if scope_hit is not None:
        risk = risk.escalate()
        reasons.append(f"escalated: requires scope {scope_hit!r}")
    path_hit = _find(SENSITIVE_PATH_WORDS, tokens(op.path), exact=True)
    if path_hit is not None:
        risk = risk.escalate()
        reasons.append(f"escalated: path is {path_hit}-only")
    if op.deprecated:
        risk = risk.escalate()
        reasons.append("escalated: deprecated")
    return risk, reasons


def classify(op: Operation) -> Classification:
    """Classify *op* and explain the decision."""
    toks = tokens(op.signal_text)
    base, why = _base_class(op, toks)
    risk, escalations = _escalated(op, base, toks)
    return Classification(risk=risk, base=base, reasons=[why, *escalations])


def classify_operation(op: Operation) -> RiskClass:
    """The :class:`RiskClass` for *op* (see :func:`classify` for the reasoning)."""
    return classify(op).risk


def risk_floor(*, operation_id: str, method: str, path: str) -> RiskClass:
    """The lowest class :func:`classify` could assign to an operation with this wire mapping.

    A plan file records a tool's method, path and operation id but not the
    summary, scopes or ``deprecated`` flag the classifier also read, so a
    plan cannot be re-classified exactly — but it can be bounded. Every rule
    that raises the class needs only tokens the mapping carries (the method,
    a destructive or money word in the id or path, an admin-style path
    segment), and text the plan lacks can only *add* tokens, so the class is
    never lower than what those tokens give. The one rule that lowers it —
    a ``POST`` led by a query verb (``search``, ``validate``…) is a read —
    may have been satisfied by a summary the plan does not carry, so a
    ``POST`` whose id and path mention no mutating verb is floored at
    ``read`` rather than ``write``. :class:`MCPcastPlan` refuses a tool
    declared below this floor: a DELETE can never be edited into a read.

    Args:
        operation_id: The route's operation id (already sanitised).
        method: The HTTP method.
        path: The path template.

    Returns:
        The floor; the deterministic classification of the same operation
        with its summary, scopes and deprecation is always at least this.
    """
    bare = Operation(operation_id=operation_id, method=method, path=path)  # type: ignore[arg-type]
    toks = tokens(bare.signal_text)
    base, _ = _base_class(bare, toks)
    if base is RiskClass.WRITE and bare.method == "POST" and _find(MUTATING_WORDS, toks) is None:
        base = RiskClass.READ  # only a summary could have led with a query verb
    return _escalated(bare, base, toks)[0]
