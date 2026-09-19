"""Deterministic planning for ``promptise mcpcast`` — no LLM, no network.

:func:`build_plan` maps operations 1:1 onto tools, filtered by the safety
profile.  Everything that is not generated lands in ``plan.dropped`` **with
a reason** — nothing silently vanishes.  This is the ``--no-curate`` path
and the substrate the LLM curator refines.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable
from typing import Any, NamedTuple

from pydantic import ValidationError

from .classify import Classification, classify
from .parse import Operation, ParamSpec
from .schema import (
    ApiPlan,
    ApprovalMode,
    AuthMode,
    CredentialLocation,
    DroppedOp,
    MCPcastError,
    MCPcastPlan,
    ParamPlan,
    RiskClass,
    RouteParam,
    RoutePlan,
    SafetyProfile,
    ToolPlan,
    render_validation_errors,
    valid_tool_name,
)

__all__ = [
    "AUTHORIZATION_HEADER",
    "CredentialSlot",
    "build_plan",
    "credential_slot",
    "derive_tool_name",
    "example_mismatch",
    "example_value",
    "make_example",
    "refuse_unrelayable_credential",
    "resolve_base_url",
    "route_base_url",
    "route_from_operation",
    "tool_from_operation",
    "unmappable_reason",
]

_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_EXPOSED_LOCATIONS = frozenset({"path", "query", "body", "raw_body"})
_SENT_BY_RUNTIME = frozenset(
    {"authorization", "content-type", "accept", "host", "content-length", "user-agent"}
)
"""Header names the generated server (or httpx underneath it) sets on every
call. Many specs declare these as required header parameters; nothing else a
spec requires in a header or cookie is ever sent, so an operation that needs
one cannot be served."""


class CredentialSlot(NamedTuple):
    """Where the generated server presents the upstream credential."""

    location: CredentialLocation
    """``header`` or ``query``."""
    name: str
    """The header or query parameter name."""

    @property
    def is_authorization_header(self) -> bool:
        """``True`` for the ``Authorization`` header — the only slot ``passthrough`` can relay."""
        return self.location == "header" and self.name.lower() == "authorization"

    def matches(self, other: CredentialSlot) -> bool:
        """Same slot (header names compare case-insensitively, query names exactly)."""
        if self.location != other.location:
            return False
        return (
            self.name.lower() == other.name.lower()
            if self.location == "header"
            else self.name == other.name
        )

    def describe(self) -> str:
        """``header 'X-API-Key'`` / ``query parameter 'api_key'`` for messages."""
        kind = "header" if self.location == "header" else "query parameter"
        return f"{kind} {self.name!r}"


AUTHORIZATION_HEADER = CredentialSlot("header", "Authorization")
"""The default slot: HTTP bearer/basic, OAuth 2 and OpenID Connect all use it."""


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------


def resolve_base_url(operations: Iterable[Operation], override: str | None = None) -> str:
    """The plan's API base URL: *override*, else the document-level server,
    else the first operation's own base."""
    if override:
        return override.rstrip("/")
    ops = list(operations)
    return next((o.doc_base_url for o in ops if o.doc_base_url), "") or next(
        (o.base_url for o in ops if o.base_url), ""
    )


def derive_tool_name(operation_id: str, taken: set[str] | None = None) -> str:
    """``getPetById`` → ``get_pet_by_id`` (unique within *taken*, if given).

    The result is always a valid tool name: an operation whose id is a Python
    keyword (``import``), a soft keyword (``match``, ``type``) or a name the
    generated server reserves (``list``, ``server``, ``approvals_list``…) is
    suffixed — ``list_op``, ``type_op``, ``server_op`` — never dropped.
    """
    snake = _CAMEL.sub("_", operation_id).lower()
    snake = re.sub(r"[^a-z0-9_]+", "_", snake)
    snake = re.sub(r"_+", "_", snake).strip("_")
    if not snake or not snake[0].isalpha():
        snake = f"op_{snake}" if snake else "operation"
    snake = snake[:64]
    if valid_tool_name(snake) is not None:
        snake = f"{snake[:61]}_op"
    if taken is None:
        return snake
    candidate, n = snake, 2
    while candidate in taken:
        suffix = f"_{n}"
        candidate = f"{snake[: 64 - len(suffix)]}{suffix}"
        n += 1
    taken.add(candidate)
    return candidate


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def _declared_type(schema: dict[str, Any]) -> str | None:
    """The schema's ``type`` (the non-null one of a list), or ``None``."""
    declared = schema.get("type")
    if isinstance(declared, list):
        declared = next((t for t in declared if t != "null"), None)
    return str(declared) if declared else None


def example_mismatch(value: Any, schema: dict[str, Any]) -> str | None:
    """Why *value* does not fit *schema*, or ``None`` if it plausibly does.

    A shallow structural check — enough to catch a model inventing a shape the
    API never declared (``items: [{sku, quantity}]`` against a schema of
    ``{sku, qty}``), or a spec example of the wrong JSON type (``example:
    2019`` on a string), without re-implementing JSON Schema.  Used to
    *repair* an example rather than to reject anything: an example is a hint
    to the agent, and a wrong one is worth replacing, not worth failing a run.
    """
    declared = _declared_type(schema)
    expected = _JSON_TYPES.get(declared) if declared else None
    if expected is not None:
        if isinstance(value, bool) and expected is not bool:
            # bool is an int to Python; it is not one to the API.
            return f"is a boolean but the API declares {declared}"
        if not isinstance(value, expected):
            return f"is a {type(value).__name__} but the API declares {declared}"
    enum = schema.get("enum")
    if isinstance(enum, list) and enum and value not in enum:
        return f"is not one of the declared values {enum[:8]}"
    if isinstance(value, dict) and isinstance(schema.get("properties"), dict):
        unknown = sorted(set(value) - set(schema["properties"]))
        if unknown and schema.get("additionalProperties") is not True:
            return f"uses fields {unknown} the API does not declare"
    if isinstance(value, list) and isinstance(schema.get("items"), dict) and value:
        return (
            None
            if (problem := example_mismatch(value[0], schema["items"])) is None
            else f"has a first item that {problem}"
        )
    return None


_UNFIT = object()
_INTEGER_LITERAL = re.compile(r"[+-]?\d+")


def _fit_example(value: Any, schema: dict[str, Any]) -> Any:
    """*value* as the example for *schema*, coerced across an obvious type slip, or ``_UNFIT``.

    Spec authors write ``example: 2019`` on a string-typed year and
    ``example: "42"`` on an integer; copied verbatim, the tool's own
    ``Example:`` line contradicts its signature and the generated test fails
    on the tool's first call. An int, float or bool on a string node becomes
    its text; a numeric string on a numeric node becomes the number; a whole
    float on an integer node becomes the integer. Anything else that does
    not fit — a non-finite number (``.inf`` parses from YAML but is not
    JSON), a value outside the enum, an invented object shape — is unfit,
    and the caller synthesises an example instead.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return _UNFIT
    if example_mismatch(value, schema) is None:
        return value
    declared = _declared_type(schema)
    coerced: Any = _UNFIT
    if declared == "string" and isinstance(value, (bool, int, float)):
        coerced = ("true" if value else "false") if isinstance(value, bool) else str(value)
    elif declared in ("integer", "number") and isinstance(value, str):
        coerced = _number(value.strip(), integer=declared == "integer")
    elif declared == "integer" and isinstance(value, float) and value.is_integer():
        coerced = int(value)
    if coerced is _UNFIT or example_mismatch(coerced, schema) is not None:
        return _UNFIT
    return coerced


def _number(text: str, *, integer: bool) -> Any:
    """*text* as an int (when *integer*) or a finite float, else ``_UNFIT``."""
    if integer:
        return int(text) if _INTEGER_LITERAL.fullmatch(text) else _UNFIT
    try:
        number = float(text)
    except ValueError:
        return _UNFIT
    return number if math.isfinite(number) else _UNFIT


def example_value(schema: dict[str, Any], name: str = "value") -> Any:
    """A plausible example for one JSON Schema (used for tool examples and mocks).

    The spec's own ``example``, ``default`` or first ``examples`` entry is
    used when it fits the declared type — coerced across an obvious slip
    (``example: 2019`` on a string becomes ``"2019"``, ``"42"`` on an integer
    becomes ``42``), dropped when it cannot fit (a non-finite number, a value
    outside the enum). Otherwise the example is synthesised from the type,
    format and parameter name.
    """
    for key in ("example", "default"):
        if key in schema and schema[key] is not None:
            fitted = _fit_example(schema[key], schema)
            if fitted is not _UNFIT:
                return fitted
    examples = schema.get("examples")
    if isinstance(examples, list) and examples and examples[0] is not None:
        fitted = _fit_example(examples[0], schema)
        if fitted is not _UNFIT:
            return fitted
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    for combinator in ("oneOf", "anyOf", "allOf"):
        options = schema.get(combinator)
        if isinstance(options, list) and options and isinstance(options[0], dict):
            return example_value(options[0], name)

    typ = _declared_type(schema)
    fmt = schema.get("format", "")
    if typ == "integer":
        return int(schema.get("minimum", 1))
    if typ == "number":
        return float(schema.get("minimum", 1.0))
    if typ == "boolean":
        return True
    if typ == "array":
        items = schema.get("items")
        return [example_value(items, name)] if isinstance(items, dict) else []
    if typ == "object" or "properties" in schema:
        # A property whose schema is not a mapping (a spec that names a
        # property ``$ref`` and a deref that once misread it) has no example.
        props = schema.get("properties")
        if not isinstance(props, dict):
            return {}
        return {
            k: example_value(v or {}, k) for k, v in props.items() if isinstance(v, dict) or not v
        }
    if fmt == "date-time":
        return "2026-01-15T09:30:00Z"
    if fmt == "date":
        return "2026-01-15"
    if fmt == "email":
        return "ada@example.com"
    if fmt in ("uri", "url"):
        return "https://example.com"
    if fmt == "uuid":
        return "123e4567-e89b-12d3-a456-426614174000"
    lowered = name.lower()
    if lowered.endswith("id"):
        return "123"
    if "email" in lowered:
        return "ada@example.com"
    if "name" in lowered:
        return "example"
    return "string"


def make_example(
    params: dict[str, ParamPlan], *, prefer: Iterable[str] | None = None
) -> dict[str, Any] | None:
    """One worked example covering every required, visible parameter.

    For a collapsed tool nothing may be required on the tool itself; pass
    the first route's required parameters as *prefer* so the example still
    shows a call that works.
    """
    names = [n for n, p in params.items() if p.required and not p.hidden]
    if not names and prefer:
        names = [n for n in prefer if n in params and not params[n].hidden]
    example = {name: _jsonable(example_value(params[name].json_schema, name)) for name in names}
    return example or None


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str), parse_constant=str)


# ---------------------------------------------------------------------------
# Operation → tool
# ---------------------------------------------------------------------------


def route_from_operation(op: Operation, *, base_url: str | None = None) -> RoutePlan:
    """The wire mapping of *op* (header/cookie parameters are not exposed).

    *base_url* is the plan's API base; when *op* is served elsewhere (its own
    ``servers`` entry — see :func:`route_base_url`) the route carries that
    host, validated like every other ``base_url``: a ``user:password@`` or a
    query string in an operation-level server URL is refused here, never
    copied into the plan.

    Raises:
        pydantic.ValidationError: If the spec cannot be mapped (e.g. a path
            placeholder with no matching path parameter, or an own server URL
            that carries a credential).  Use :func:`unmappable_reason` to
            test first.
    """
    return RoutePlan(
        operation_id=op.operation_id,
        method=op.method,
        path=op.path,
        params={
            p.name: RouteParam(
                location=p.location,  # type: ignore[arg-type]
                required=p.required,
                wire_name=p.wire_name,
            )
            for p in op.params
            if p.location in _EXPOSED_LOCATIONS
        },
        body_encoding=op.body_encoding,
        base_url=route_base_url(op, base_url),
    )


def unmappable_reason(
    op: Operation,
    *,
    base_url: str | None = None,
    credential: CredentialSlot | None = None,
) -> str | None:
    """Why *op* cannot become a tool, or ``None`` if it can.

    Only the wire mapping is judged here — an unsupported request body, a
    ``$ref`` the parser could not resolve, a path placeholder without its
    parameter, an own server URL that carries a credential, a required
    header or cookie parameter the generated server would never send, or a
    security scheme it cannot satisfy; the operation id never disqualifies
    an operation, since :func:`derive_tool_name` always produces a valid
    tool name.

    Args:
        op: The operation.
        base_url: The plan's API base URL (an own server URL equal to it is
            not a route-level override).
        credential: The slot the plan's server presents its credential in
            (see :func:`credential_slot`); when given, an operation whose
            scheme needs another slot, or one no server can present (an API
            key in a cookie), is unmappable, and a required header parameter
            with the slot's name counts as sent.
    """
    if op.unsupported_body:
        return f"unsupported by mcpcast: {op.unsupported_body}"
    try:
        route_from_operation(op, base_url=base_url)
    except ValidationError as exc:
        first = exc.errors()[0]["msg"] if exc.errors() else str(exc)
        return "unsupported by mcpcast: " + first.removeprefix("Value error, ").strip()
    sent = set(_SENT_BY_RUNTIME)
    if credential is not None and credential.location == "header":
        sent.add(credential.name.lower())
    for p in op.params:
        if p.location in ("header", "cookie") and p.required and p.wire.lower() not in sent:
            return (
                f"unsupported by mcpcast: required {p.location} parameter {p.wire!r} cannot be sent"
            )
    scheme = op.security_scheme
    if credential is not None and scheme is not None:
        needed = scheme.credential
        if needed is None:
            return (
                f"unsupported by mcpcast: requires {scheme.describe()}, which the generated "
                "server cannot present"
            )
        if not CredentialSlot(*needed).matches(credential):
            return (
                f"unsupported by mcpcast: requires {scheme.describe()}, but this server "
                f"presents its credential in {credential.describe()}"
            )
    return None


def credential_slot(operations: Iterable[Operation]) -> CredentialSlot:
    """The one slot the plan's server presents its credential in, from the spec's schemes.

    Every operation records the security scheme it resolves to
    (:attr:`~promptise.mcpcast.parse.Operation.security_scheme`); the slot
    the most operations need wins (ties go to spec order — the document-level
    scheme, which most operations inherit, normally is the majority), so a
    mixed API loses the minority rather than the whole surface. The
    ``Authorization`` header when no operation declares a presentable scheme.
    Operations that need a different slot are dropped by
    :func:`unmappable_reason` with that reason.
    """
    counts: dict[tuple[str, str], int] = {}
    spelling: dict[tuple[str, str], CredentialSlot] = {}
    for op in operations:
        scheme = op.security_scheme
        if scheme is None or scheme.credential is None:
            continue
        slot = CredentialSlot(*scheme.credential)
        key = (slot.location, slot.name.lower() if slot.location == "header" else slot.name)
        counts[key] = counts.get(key, 0) + 1
        spelling.setdefault(key, slot)
    if not counts:
        return AUTHORIZATION_HEADER
    return spelling[max(counts, key=counts.__getitem__)]


def refuse_unrelayable_credential(auth: AuthMode, credential: CredentialSlot) -> None:
    """``passthrough`` relays the caller's ``Authorization`` header and nothing else.

    Raises:
        MCPcastError: When *credential* is any other slot under that mode.
    """
    if auth is AuthMode.PASSTHROUGH and not credential.is_authorization_header:
        raise MCPcastError(
            f"the spec authenticates with a credential in {credential.describe()}, which "
            "--auth passthrough cannot relay (it forwards only the caller's Authorization "
            "header) — generate with --auth env-token (one upstream credential from "
            "MCPCAST_UPSTREAM_TOKEN, presented there) or --auth api-key (one per tenant)"
        )


def _param_plan(spec: ParamSpec) -> ParamPlan:
    return ParamPlan(
        description=spec.description,
        json_schema=dict(spec.json_schema) or {"type": "string"},
        required=spec.required,
    )


def _describe(op: Operation) -> str:
    text = op.summary or op.description or f"{op.method} {op.path}"
    if op.summary and op.description and op.description != op.summary:
        text = f"{op.summary.rstrip('.')}. {op.description}"
    return text


def route_base_url(op: Operation, base_url: str | None) -> str | None:
    """The host a route must carry, or ``None`` when the plan's *base_url* serves it.

    Only an operation that declares its **own** ``servers`` (a base that
    differs from the document's) is served elsewhere.  When the document had
    no ``servers`` block and its base was resolved from the spec URL's origin,
    every operation shares that base — it must not be frozen into the routes,
    or the plan's base URL (edited in the wizard, or ``MCPCAST_BASE_URL`` at
    run time) would silently stop applying to them.
    """
    own = op.base_url.rstrip("/") if op.base_url else ""
    if not own or own == (op.doc_base_url or "").rstrip("/"):
        return None
    if base_url and own == base_url.rstrip("/"):
        return None
    return own


def tool_from_operation(
    op: Operation,
    *,
    name: str,
    risk: RiskClass,
    profile: SafetyProfile,
    base_url: str | None = None,
) -> ToolPlan:
    """Build the 1:1 tool for *op* under *profile*.

    *base_url* is the plan's API base; a route whose operation is served
    elsewhere (operation-level ``servers``) records its own base URL — see
    :func:`route_base_url`.
    """
    params = {p.name: _param_plan(p) for p in op.params if p.location in _EXPOSED_LOCATIONS}
    return ToolPlan(
        name=name,
        description=_describe(op),
        risk=risk,
        routes=[route_from_operation(op, base_url=base_url)],
        params=params,
        example=make_example(params),
        requires_approval=profile.requires_approval(risk),
        tags=list(op.tags),
    )


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def build_plan(
    operations: Iterable[Operation],
    *,
    profile: SafetyProfile = SafetyProfile.READ_ONLY,
    base_url: str | None = None,
    auth: AuthMode = AuthMode.PASSTHROUGH,
    approval: ApprovalMode | None = None,
    name: str = "api",
    description: str = "",
    spec_source: str | None = None,
    max_tools: int | None = None,
    classifications: dict[str, Classification] | None = None,
) -> MCPcastPlan:
    """Map operations 1:1 onto tools, filtered by *profile*.

    Args:
        operations: Parsed operations (see :func:`~promptise.mcpcast.parse.extract_operations`).
        profile: Safety profile deciding which risk classes are generated.
        base_url: API base URL; defaults to the one declared by the spec.
        auth: How the generated server authenticates against the API.
        approval: Who approves gated calls (defaults per *auth*).
        name: Slug used for the server name.
        description: One-line API description (becomes server instructions).
        spec_source: Where the spec came from (recorded in the plan).
        max_tools: Optional budget; reads are kept first, then spec order.
            Everything over budget is dropped with an explicit reason.
        classifications: Precomputed classifications keyed by operation id
            (computed with :func:`~promptise.mcpcast.classify.classify` if omitted).

    Returns:
        A validated :class:`MCPcastPlan`.

    Raises:
        MCPcastError: If the plan cannot be built — no base URL, a
            ``base_url`` that carries a credential, or *auth* is
            ``passthrough`` while the spec authenticates with something other
            than the ``Authorization`` header (an API key in a custom header
            or a query parameter), which passthrough cannot relay.
    """
    ops = list(operations)
    if max_tools is not None and max_tools < 1:
        raise ValueError(f"max_tools must be >= 1, got {max_tools}")
    resolved_base = resolve_base_url(ops, base_url)
    credential = credential_slot(ops)
    refuse_unrelayable_credential(auth, credential)
    classes = classifications or {o.operation_id: classify(o) for o in ops}

    dropped: list[DroppedOp] = []
    candidates: list[tuple[Operation, RiskClass]] = []
    for op in ops:
        cls = classes.get(op.operation_id) or classify(op)
        if op.deprecated:
            dropped.append(DroppedOp(operation_id=op.operation_id, reason="deprecated in spec"))
            continue
        if (
            why := unmappable_reason(op, base_url=resolved_base, credential=credential)
        ) is not None:
            dropped.append(DroppedOp(operation_id=op.operation_id, reason=why))
            continue
        if not profile.allows(cls.risk):
            dropped.append(
                DroppedOp(operation_id=op.operation_id, reason=profile.exclusion_reason(cls.risk))
            )
            continue
        candidates.append((op, cls.risk))

    if max_tools is not None and len(candidates) > max_tools:
        ranked = sorted(enumerate(candidates), key=lambda e: (e[1][1].severity, e[0]))
        keep_idx = {i for i, _ in ranked[:max_tools]}
        for i, (op, _risk) in enumerate(candidates):
            if i not in keep_idx:
                dropped.append(
                    DroppedOp(
                        operation_id=op.operation_id,
                        reason=(
                            f"over tool budget (max_tools={max_tools}); raise --max-tools "
                            "or let curation pick the most useful tools"
                        ),
                    )
                )
        candidates = [c for i, c in enumerate(candidates) if i in keep_idx]

    taken: set[str] = set()
    tools: list[ToolPlan] = []
    for op, risk in candidates:
        try:
            tools.append(
                tool_from_operation(
                    op,
                    name=derive_tool_name(op.operation_id, taken),
                    risk=risk,
                    profile=profile,
                    base_url=resolved_base,
                )
            )
        except (ValidationError, ValueError, TypeError) as exc:
            # One odd schema must not abort the run: record it with its reason.
            first = exc.errors()[0]["msg"] if isinstance(exc, ValidationError) else str(exc)
            dropped.append(
                DroppedOp(
                    operation_id=op.operation_id,
                    reason="unsupported by mcpcast: " + first.removeprefix("Value error, ").strip(),
                )
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
        raise MCPcastError(f"could not build plan:\n{render_validation_errors(exc)}") from exc
