"""LLM-assisted tool design for ``promptise mcpcast`` — the curation step.

The deterministic planner exposes one tool per operation.  Curation asks a
model to design the surface an *agent* needs instead: pick the most useful
operations within a budget, drop what an end user's agent has no business
calling, collapse redundant routes into intent tools, rename them in the
product's domain language, rewrite descriptions for an LLM audience, put
rarely-needed parameters on a diet, and give every tool a worked example.

The model proposes; the code decides.  Every proposal is checked against
post-conditions that are enforced here, not trusted to the model:

- at most ``max_tools`` tools, with valid, unique names
- every referenced operation exists in the spec, is used by at most one
  tool, and is never both kept and dropped
- **risk is never downgraded** below the deterministic classifier — the
  model may escalate a class, never relax it, so a hallucinating model
  cannot turn a ``DELETE`` into a "read"
- parameters exist on the operations they belong to; hidden required
  parameters carry a default; examples only use visible parameters
- deprecated operations are dropped
- descriptions only name tools the generated server exposes — not a tool
  the safety profile excluded, an operation that was dropped or merged
  into another tool, or anything else that is not in the final tool set

A proposal that violates a post-condition is sent back with the violations
for another attempt; after ``max_attempts`` the run fails loudly.  Nothing
falls back to a silently different plan.  The one exception is a dangling
tool reference that survives every attempt: the sentence naming it is
removed (see :func:`apply_curation`), because a hint to the agent is not
worth failing a run over — the same reasoning as repairing a bad example.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ._llm import Completer, completer_for, extract_json_object
from .classify import Classification, classify
from .parse import Operation
from .plan import (
    _describe,
    credential_slot,
    derive_tool_name,
    example_mismatch,
    make_example,
    refuse_unrelayable_credential,
    resolve_base_url,
    route_from_operation,
    unmappable_reason,
)
from .schema import (
    ApiPlan,
    ApprovalMode,
    AuthMode,
    DroppedOp,
    MCPcastError,
    MCPcastPlan,
    ParamPlan,
    RiskClass,
    SafetyProfile,
    ToolPlan,
    render_validation_errors,
    valid_tool_name,
)

__all__ = [
    "CURATION_SYSTEM",
    "MAX_PROMPT_CHARS",
    "CuratedParam",
    "CuratedTool",
    "CurationResult",
    "CurationViolation",
    "apply_curation",
    "check_postconditions",
    "example_mismatch",
    "curate",
    "render_curation_prompt",
]

_EXPOSED = frozenset({"path", "query", "body", "raw_body"})

MAX_PROMPT_CHARS = 600_000
"""Largest curation prompt sent to a model (~150k tokens). Bigger specs must be
narrowed first — the prompt is re-sent in full on every retry."""


class CurationViolation(MCPcastError):
    """A curation proposal broke one or more post-conditions."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        super().__init__("curation proposal rejected:\n- " + "\n- ".join(violations))


# ---------------------------------------------------------------------------
# What the model returns
# ---------------------------------------------------------------------------


class CuratedParam(BaseModel):
    """The model's adjustments to one parameter."""

    model_config = ConfigDict(extra="ignore")

    description: str = ""
    hidden: bool = False
    default: Any = None


class CuratedTool(BaseModel):
    """One tool as proposed by the model."""

    model_config = ConfigDict(extra="ignore")

    name: str
    description: str
    risk: RiskClass
    operations: list[str] = Field(..., min_length=1)
    params: dict[str, CuratedParam] = Field(default_factory=dict)
    example: dict[str, Any] | None = None
    tags: list[str] = Field(default_factory=list)


class CurationResult(BaseModel):
    """The model's complete proposal."""

    model_config = ConfigDict(extra="ignore")

    tools: list[CuratedTool] = Field(default_factory=list)
    dropped: list[DroppedOp] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

CURATION_SYSTEM = """\
You design tool surfaces for AI agents that must use an existing HTTP API through MCP.
Success is a SMALL set of tools an agent uses correctly — never one tool per route.

Rules, in priority order:
1. BUDGET — keep at most the requested number of tools, ranked by usefulness to an end
   user's agent.
2. DROP — health checks, webhook receivers, internal/admin endpoints, deprecated
   operations, bulk exports, and anything an end user's agent has no business calling.
   Give every dropped operation a short, honest reason.
3. COLLAPSE — fold redundant routes into one intent tool when the SAME agent goal is
   served (e.g. get-by-id + search → find_customer). Never collapse a read with a write.
   DISPATCH RULE — a tool picks the first listed operation whose required parameters were
   all supplied, so operations you merge must be TELLABLE APART by their required
   parameters: list the most specific first, never merge two operations that require the
   same set (get-by-id + delete-by-id both need only {id} — keep them separate), and if
   one operation requires nothing at all it must be listed last. When in doubt, do not
   merge: two clear tools beat one that can never reach its second route.
4. RENAME — name tools as intents in the product's domain language, lowercase
   snake_case (cancel_subscription, not post_v2_customers_id_subscriptions_cancel).
5. DESCRIBE FOR AN LLM — what the tool does, WHEN to use it, when NOT to, and what comes
   back. Two to four sentences. Mention related tools by name — but ONLY tools in your
   proposal whose risk the safety profile exposes. A tool the profile excludes, or an
   operation you dropped or merged, does not exist in the server: never name it.
6. PARAM DIET — keep required parameters visible; hide rarely-needed optionals by setting
   hidden=true with a sensible default; improve parameter descriptions.
7. EXAMPLE — one realistic worked example per tool using only visible parameters.
8. RISK — you may raise an operation's risk class (read < write < destructive/financial)
   when the name or description shows the classifier underestimated it. You may NEVER
   lower it.

Respond with ONE JSON object and nothing else:
{
  "tools": [
    {
      "name": "find_customer",
      "description": "…",
      "risk": "read",
      "operations": ["getCustomerById", "searchCustomers"],
      "params": {"email": {"description": "…", "hidden": false, "default": null}},
      "example": {"email": "ada@example.com"},
      "tags": ["customers"]
    }
  ],
  "dropped": [{"operation_id": "healthCheck", "reason": "not useful to an agent"}]
}
Every operation in the catalogue must appear exactly once — in a tool or in dropped.
"""


def _catalogue_entry(op: Operation, cls: Classification) -> dict[str, Any]:
    return {
        "operation_id": op.operation_id,
        "method": op.method,
        "path": op.path,
        "summary": op.summary,
        "description": op.description[:400],
        "tags": op.tags,
        "deprecated": op.deprecated,
        "risk": cls.risk.value,
        "params": [
            {
                "name": p.name,
                "in": p.location,
                "required": p.required,
                "type": p.json_schema.get("type", "string"),
                "description": p.description[:200],
            }
            for p in op.params
            if p.location in _EXPOSED
        ],
    }


def render_curation_prompt(
    operations: Iterable[Operation],
    classifications: dict[str, Classification],
    *,
    max_tools: int,
    profile: SafetyProfile,
    api_name: str = "api",
    api_description: str = "",
) -> str:
    """The user prompt: the operation catalogue plus the budget."""
    ops = list(operations)
    catalogue = [_catalogue_entry(op, classifications[op.operation_id]) for op in ops]
    intro = f"API: {api_name}"
    exposed = ", ".join(r.value for r in RiskClass if profile.allows(r))
    if api_description:
        intro += f" — {api_description.strip()[:500]}"
    return (
        f"{intro}\n"
        f"Operations in the spec: {len(ops)}. Tool budget: at most {max_tools} tools.\n"
        f"Safety profile that will be applied afterwards: {profile.value} "
        "(design the best surface regardless; exposure is gated later). "
        f"It exposes only these risk classes: {exposed}; tools of any other class are "
        "removed, so no description may name them.\n\n"
        "Operation catalogue (JSON lines):\n"
        + "\n".join(json.dumps(entry, ensure_ascii=False) for entry in catalogue)
    )


# ---------------------------------------------------------------------------
# Post-conditions
# ---------------------------------------------------------------------------


def _exposed_params(op: Operation) -> dict[str, Any]:
    return {p.name: p for p in op.params if p.location in _EXPOSED}


def check_postconditions(
    result: CurationResult,
    operations: Iterable[Operation],
    classifications: dict[str, Classification],
    *,
    max_tools: int,
) -> list[str]:
    """Every post-condition violation in *result* (empty means it passed)."""
    ops = {op.operation_id: op for op in operations}
    violations: list[str] = []

    if len(result.tools) > max_tools:
        violations.append(f"{len(result.tools)} tools exceed the budget of {max_tools}")

    names: set[str] = set()
    owner: dict[str, str] = {}
    for tool in result.tools:
        problem = valid_tool_name(tool.name)
        if problem:
            violations.append(problem)
        if tool.name in names:
            violations.append(f"duplicate tool name {tool.name!r}")
        names.add(tool.name)
        if not tool.description.strip():
            violations.append(f"tool {tool.name!r} has an empty description")

        known_ops = [oid for oid in tool.operations if oid in ops]
        for oid in tool.operations:
            if oid not in ops:
                violations.append(f"tool {tool.name!r} references unknown operation {oid!r}")
            elif oid in owner:
                violations.append(
                    f"operation {oid!r} is used by both {owner[oid]!r} and {tool.name!r}"
                )
            else:
                owner[oid] = tool.name
                if ops[oid].deprecated:
                    violations.append(
                        f"tool {tool.name!r} keeps deprecated operation {oid!r}; drop it"
                    )
        if len(set(tool.operations)) != len(tool.operations):
            violations.append(f"tool {tool.name!r} lists an operation twice")

        if known_ops:
            floor = max(
                (classifications[oid].risk for oid in known_ops if oid in classifications),
                key=lambda r: r.severity,
                default=RiskClass.READ,
            )
            if not tool.risk.at_least(floor):
                violations.append(
                    f"tool {tool.name!r} downgrades risk to {tool.risk.value!r}; the classifier "
                    f"requires at least {floor.value!r} (risk may be raised, never lowered)"
                )
            union: dict[str, Any] = {}
            required_anywhere: set[str] = set()
            for oid in known_ops:
                exposed = _exposed_params(ops[oid])
                union.update(exposed)
                required_anywhere.update(n for n, p in exposed.items() if p.required)
            for pname, cp in tool.params.items():
                if pname not in union:
                    violations.append(f"tool {tool.name!r} adjusts unknown parameter {pname!r}")
                elif cp.hidden and cp.default is None and pname in required_anywhere:
                    violations.append(
                        f"tool {tool.name!r} hides required parameter {pname!r} without a default"
                    )
            hidden = {p for p, cp in tool.params.items() if cp.hidden}
            effective = [
                {n for n, p in _exposed_params(ops[oid]).items() if p.required and n not in hidden}
                for oid in known_ops
            ]
            for i, later in enumerate(effective):
                for j in range(i):
                    if effective[j] <= later:
                        violations.append(
                            f"tool {tool.name!r}: operation {known_ops[i]!r} can never be "
                            f"selected because {known_ops[j]!r} (listed earlier) needs a subset "
                            "of its parameters; list the more specific operation first"
                        )
                        break
            if tool.example:
                bad = [k for k in tool.example if k not in union or k in hidden]
                if bad:
                    violations.append(
                        f"tool {tool.name!r} example uses unknown or hidden parameters {bad}"
                    )

    dropped_ids: set[str] = set()
    for d in result.dropped:
        if d.operation_id not in ops:
            violations.append(f"dropped unknown operation {d.operation_id!r}")
        if d.operation_id in owner:
            violations.append(
                f"operation {d.operation_id!r} is both kept ({owner[d.operation_id]!r}) and dropped"
            )
        if d.operation_id in dropped_ids:
            violations.append(f"operation {d.operation_id!r} is dropped twice")
        dropped_ids.add(d.operation_id)
    return violations


# ---------------------------------------------------------------------------
# Proposal → plan
# ---------------------------------------------------------------------------


def _repair_example(
    example: dict[str, Any] | None, params: dict[str, ParamPlan]
) -> dict[str, Any] | None:
    """Drop the entries of a curated *example* that contradict the API's schema.

    Models reliably invent plausible-looking values (``{"sku": …, "quantity": …}``
    for a schema of ``{"sku", "qty"}``).  Such a value would teach every agent
    reading the tool the wrong shape, so it is discarded; what is left — or a
    spec-derived example — is used instead.
    """
    if not example:
        return None
    kept = {
        name: value
        for name, value in example.items()
        if name in params
        and not params[name].hidden
        and example_mismatch(value, params[name].json_schema) is None
    }
    return kept or None


_TOOL_LIKE = re.compile(r"(?<![A-Za-z0-9_])[a-z][a-z0-9]*(?:_[a-z0-9]+)+(?![A-Za-z0-9_])")
"""A snake_case identifier with at least one underscore — the shape of a tool
name.  Single words (``search``) are not matched: in prose they are English."""

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[^a-z])|\n+")


def _phantom_tools(
    result: CurationResult,
    spec_operations: Iterable[Operation],
    exposed: set[str],
    profile: SafetyProfile,
) -> dict[str, str]:
    """Names an agent would read as a tool the generated server does not have.

    Every name the model proposed and the would-be tool name of every
    operation in the spec (``deletePet`` → ``delete_pet``), minus the tools
    that are exposed — mapped to why each one is missing, for the feedback.
    Parameter names are never phantoms, so ``pet_id`` in prose is not
    mistaken for a tool.
    """
    owner = {oid: ct for ct in result.tools for oid in ct.operations}
    phantoms: dict[str, str] = {}
    for ct in result.tools:
        if ct.name not in exposed:
            phantoms.setdefault(ct.name, profile.exclusion_reason(ct.risk))
    param_names: set[str] = set()
    for op in spec_operations:
        param_names.update(p.name for p in op.params)
        name = derive_tool_name(op.operation_id)
        if name in exposed or name in phantoms:
            continue
        tool = owner.get(op.operation_id)
        if tool is None:
            why = f"operation {op.operation_id!r} is not in the generated server"
        elif tool.name in exposed:
            why = f"operation {op.operation_id!r} is served by {tool.name!r}; name that instead"
        else:
            why = f"operation {op.operation_id!r} is {profile.exclusion_reason(tool.risk)}"
        phantoms[name] = why
    return {name: why for name, why in phantoms.items() if name not in param_names}


def _named_phantoms(text: str, phantoms: dict[str, str]) -> list[str]:
    """The phantom tools *text* names, in first-mention order."""
    return list(dict.fromkeys(m for m in _TOOL_LIKE.findall(text) if m in phantoms))


def _strip_phantoms(text: str, phantoms: dict[str, str]) -> str:
    """*text* without the sentences that name a phantom tool."""
    kept = [s.strip() for s in _SENTENCE_BREAK.split(text) if not _named_phantoms(s, phantoms)]
    return " ".join(s for s in kept if s)


def _dangling(where: str, named: list[str], phantoms: dict[str, str], exposed: list[str]) -> str:
    return (
        f"{where} names {', '.join(f'{n} ({phantoms[n]})' for n in named)} — no such tool "
        "will exist in the generated server; remove the reference or name one of: "
        + ", ".join(exposed)
    )


def _merge_params(ops: list[Operation], curated: dict[str, CuratedParam]) -> dict[str, ParamPlan]:
    """Agent-facing parameters: the union across routes, with the model's edits.

    A parameter is *required* only if every route needs it — dispatch picks a
    route by which parameters were supplied, so anything optional on one
    route must stay optional on the tool.
    """
    params: dict[str, ParamPlan] = {}
    for op in ops:
        for spec in op.params:
            if spec.location in _EXPOSED and spec.name not in params:
                params[spec.name] = ParamPlan(
                    description=spec.description,
                    json_schema=dict(spec.json_schema) or {"type": "string"},
                    required=spec.required,
                )
    for op in ops:
        required_here = {p.name for p in op.params if p.required and p.location in _EXPOSED}
        for name, plan in params.items():
            if name not in required_here:
                plan.required = False
    for name, cp in curated.items():
        if name not in params:
            continue
        plan = params[name]
        if cp.description.strip():
            plan.description = cp.description.strip()
        if cp.default is not None:
            plan.default = cp.default
        if cp.hidden:
            plan.hidden = True
    return {
        name: ParamPlan.model_validate(p.model_dump()) for name, p in params.items()
    }  # re-run validators after mutation


def apply_curation(
    result: CurationResult,
    operations: Iterable[Operation],
    classifications: dict[str, Classification],
    *,
    profile: SafetyProfile,
    max_tools: int,
    base_url: str | None = None,
    auth: AuthMode = AuthMode.PASSTHROUGH,
    approval: ApprovalMode | None = None,
    name: str = "api",
    description: str = "",
    spec_source: str | None = None,
    spec_operations: Iterable[Operation] | None = None,
    repair_references: bool = False,
) -> MCPcastPlan:
    """Turn an accepted proposal into a validated :class:`MCPcastPlan`.

    A tool or parameter description that names a tool the plan does not
    expose — one the profile excluded, an operation that was dropped or
    merged, any snake_case form of an operation id in the spec that is not a
    tool — is a violation.  With *repair_references* (``curate()`` sets it
    on its last attempt) the sentences naming such a tool are removed
    instead; a description left empty falls back to the spec's.

    Args:
        spec_operations: Every operation in the spec, including any not
            passed as *operations* (unmappable ones); their would-be tool
            names count as references to tools that do not exist.  Defaults
            to *operations*.
        repair_references: Strip dangling tool references instead of
            rejecting the proposal.

    Raises:
        CurationViolation: If any post-condition fails.
        MCPcastError: If *auth* is ``passthrough`` and the spec's credential
            is not the ``Authorization`` header (see :func:`~promptise.mcpcast.plan.build_plan`).
    """
    ops = list(operations)
    by_id = {op.operation_id: op for op in ops}
    resolved_base = resolve_base_url(ops, base_url)
    credential = credential_slot(ops)
    refuse_unrelayable_credential(auth, credential)
    violations = check_postconditions(result, ops, classifications, max_tools=max_tools)
    if violations:
        raise CurationViolation(violations)

    exposed = [ct.name for ct in result.tools if profile.allows(ct.risk)]
    phantoms = _phantom_tools(
        result, ops if spec_operations is None else spec_operations, set(exposed), profile
    )
    dangling: list[str] = []
    tools: list[ToolPlan] = []
    dropped: list[DroppedOp] = list(result.dropped)
    mentioned = {d.operation_id for d in dropped}
    for ct in result.tools:
        tool_ops = [by_id[oid] for oid in ct.operations]
        mentioned.update(ct.operations)
        if not profile.allows(ct.risk):
            dropped.extend(
                DroppedOp(operation_id=oid, reason=profile.exclusion_reason(ct.risk))
                for oid in ct.operations
            )
            continue
        params = _merge_params(tool_ops, ct.params)
        tool_description = ct.description
        if named := _named_phantoms(tool_description, phantoms):
            if repair_references:
                tool_description = _strip_phantoms(tool_description, phantoms) or (
                    _strip_phantoms(_describe(tool_ops[0]), phantoms)
                    or f"{tool_ops[0].method} {tool_ops[0].path}"
                )
            else:
                dangling.append(
                    _dangling(f"tool {ct.name!r} description", named, phantoms, exposed)
                )
        for pname, pplan in params.items():
            if named := _named_phantoms(pplan.description, phantoms):
                if repair_references:
                    spec_text = next(
                        (p.description for op in tool_ops for p in op.params if p.name == pname),
                        "",
                    )
                    pplan.description = _strip_phantoms(
                        pplan.description, phantoms
                    ) or _strip_phantoms(spec_text, phantoms)
                else:
                    dangling.append(
                        _dangling(
                            f"tool {ct.name!r} parameter {pname!r} description",
                            named,
                            phantoms,
                            exposed,
                        )
                    )
        first_required = [
            p.name for p in tool_ops[0].params if p.required and p.location in _EXPOSED
        ]
        example = _repair_example(ct.example, params) or make_example(params, prefer=first_required)
        tools.append(
            ToolPlan(
                name=ct.name,
                description=tool_description,
                risk=ct.risk,
                routes=[route_from_operation(op, base_url=resolved_base) for op in tool_ops],
                params=params,
                example=example,
                requires_approval=profile.requires_approval(ct.risk),
                tags=ct.tags or sorted({t for op in tool_ops for t in op.tags}),
            )
        )
    if dangling:
        raise CurationViolation(dangling)
    for op in ops:
        if op.operation_id not in mentioned:
            dropped.append(
                DroppedOp(operation_id=op.operation_id, reason="not selected by curation")
            )

    try:
        return MCPcastPlan(
            api=ApiPlan(
                name=name,
                base_url=resolved_base,
                auth=auth,
                approval=approval,
                description=description,
                spec_source=spec_source,
                credential_location=credential.location,
                credential_name=credential.name,
            ),
            profile=profile,
            tools=tools,
            dropped=dropped,
        )
    except ValidationError as exc:
        # Fed back to the model on retry and printed on failure: the
        # listing must never echo a value (a credential in a base_url).
        raise CurationViolation([render_validation_errors(exc)]) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def curate(
    operations: Iterable[Operation],
    *,
    model: Any = "openai:gpt-5-mini",
    max_tools: int = 25,
    profile: SafetyProfile = SafetyProfile.READ_ONLY,
    base_url: str | None = None,
    auth: AuthMode = AuthMode.PASSTHROUGH,
    approval: ApprovalMode | None = None,
    name: str = "api",
    description: str = "",
    spec_source: str | None = None,
    max_attempts: int = 3,
    complete: Completer | None = None,
) -> MCPcastPlan:
    """Design the tool surface with a model, enforcing every post-condition in code.

    Args:
        operations: Parsed operations.
        model: Model id or LangChain chat model used for curation.
        max_tools: Tool budget.
        profile: Safety profile applied *after* curation.
        base_url: API base URL recorded on the plan (defaults to the spec's).
        auth: How the generated server authenticates against the API.
        approval: Who approves gated calls (defaults per *auth*).
        name: Server slug recorded on the plan.
        description: One-line API description (becomes server instructions).
        spec_source: Where the spec came from (recorded on the plan).
        max_attempts: Proposals rejected for violations are retried with the
            violations as feedback, up to this many times.  On the last
            attempt a description that still names a tool the server will
            not have loses the sentence that names it (see
            :func:`apply_curation`) instead of failing the run.
        complete: Override the completion function (tests inject a script).

    Raises:
        MCPcastError: If no valid proposal is obtained within ``max_attempts``,
            or the plan cannot be built at all (no base URL, a ``base_url``
            with a credential, a spec credential ``passthrough`` cannot
            relay) — settled before any model call.
    """
    all_ops = list(operations)
    if not all_ops:
        raise MCPcastError("nothing to curate: the spec has no operations")
    if max_tools < 1:
        raise ValueError(f"max_tools must be >= 1, got {max_tools}")
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    # Settle everything the model cannot influence before spending a call on it.
    resolved_base = resolve_base_url(all_ops, base_url)
    credential = credential_slot(all_ops)
    refuse_unrelayable_credential(auth, credential)
    try:
        ApiPlan(
            name=name,
            base_url=resolved_base,
            auth=auth,
            approval=approval,
            description=description,
            spec_source=spec_source,
            credential_location=credential.location,
            credential_name=credential.name,
        )
    except ValidationError as exc:
        # The listing never echoes the offending value: it may be a credential.
        raise MCPcastError(f"could not build plan:\n{render_validation_errors(exc)}") from exc
    # Operations the emitter cannot map are settled before the model sees
    # anything — it cannot fix a spec defect, and they must still be listed.
    unmappable = [
        DroppedOp(operation_id=op.operation_id, reason=why)
        for op in all_ops
        if (why := unmappable_reason(op, base_url=resolved_base, credential=credential)) is not None
    ]
    skipped = {d.operation_id for d in unmappable}
    ops = [op for op in all_ops if op.operation_id not in skipped]
    if not ops:
        raise MCPcastError(
            "nothing to curate: no operation in the spec can be mapped to a tool "
            f"({unmappable[0].reason})"
        )
    classifications = {op.operation_id: classify(op) for op in ops}
    completer = complete or completer_for(model)
    prompt = render_curation_prompt(
        ops,
        classifications,
        max_tools=max_tools,
        profile=profile,
        api_name=name,
        api_description=description,
    )

    if len(prompt) > MAX_PROMPT_CHARS:
        raise MCPcastError(
            f"the curation prompt for {len(ops)} operations is {len(prompt):,} characters "
            f"(limit {MAX_PROMPT_CHARS:,}); narrow the spec first (curate one tag or path "
            "prefix at a time, or use --no-curate and edit the plan)"
        )

    feedback = ""
    last_error = "no attempts made"
    for attempt in range(max_attempts):
        text = await completer(CURATION_SYSTEM, prompt + feedback)
        try:
            result = CurationResult.model_validate(extract_json_object(text))
        except (MCPcastError, ValidationError) as exc:
            last_error = f"invalid response: {exc}"
            feedback = (
                "\n\nYour previous answer could not be parsed as the required JSON object "
                f"({exc}). Respond with only the JSON object."
            )
            continue
        try:
            plan = apply_curation(
                result,
                ops,
                classifications,
                profile=profile,
                max_tools=max_tools,
                base_url=base_url,
                auth=auth,
                approval=approval,
                name=name,
                description=description,
                spec_source=spec_source,
                spec_operations=all_ops,
                repair_references=attempt == max_attempts - 1,
            )
            if unmappable:
                plan = plan.model_copy(update={"dropped": [*plan.dropped, *unmappable]})
            return plan
        except CurationViolation as exc:
            last_error = str(exc)
            # Each attempt is a fresh completion: hand back the proposal so the
            # model edits it instead of re-rolling the whole surface.
            feedback = (
                "\n\nYour previous proposal was rejected. Fix these problems and respond "
                "with only the corrected JSON object:\n- "
                + "\n- ".join(exc.violations)
                + "\n\nYour previous proposal:\n"
                + json.dumps(result.model_dump(mode="json"), ensure_ascii=False)
            )
    raise MCPcastError(
        f"curation failed after {max_attempts} attempt(s): {last_error}\n"
        "Re-run with --no-curate for a deterministic one-tool-per-operation plan."
    )
