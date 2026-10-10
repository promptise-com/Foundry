"""Plan schema for ``promptise mcpcast`` — the editable source of truth.

An :class:`MCPcastPlan` describes the agent-facing tool surface generated
from an existing API: which operations become which tools, how risky each
tool is, which ones demand human approval, and which operations were
deliberately left out (and why).  It round-trips to ``mcpcast.plan.yaml``;
``server.py`` is always regenerated *from the plan*, never edited in place
by the tool.

The schema mirrors the philosophy of ``.superagent`` / ``.agent`` manifests:
validated, versioned, diffable, and re-runnable.

Example::

    from promptise.mcpcast import MCPcastPlan

    plan = MCPcastPlan.load("mcpcast.plan.yaml")
    for tool in plan.tools:
        print(tool.name, tool.risk.value, "approval" if tool.requires_approval else "")
"""

from __future__ import annotations

import json
import keyword
import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

__all__ = [
    "RESERVED_CREDENTIAL_HEADERS",
    "RESERVED_PARAM_NAMES",
    "RESERVED_TOOL_NAMES",
    "SOFT_KEYWORDS",
    "TOOL_NAME_PATTERN",
    "ApiPlan",
    "ApprovalMode",
    "AuthMode",
    "CredentialLocation",
    "DroppedOp",
    "HttpMethod",
    "MCPcastError",
    "MCPcastPlan",
    "ParamLocation",
    "ParamPlan",
    "RiskClass",
    "RouteParam",
    "RoutePlan",
    "SafetyProfile",
    "ToolPlan",
    "is_plan_document",
    "is_python_keyword",
    "render_validation_errors",
    "scrub_text",
    "valid_tool_name",
]


class MCPcastError(Exception):
    """Raised for an invalid spec, plan, or curation result."""


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RiskClass(str, Enum):
    """What calling a tool can do to the world.

    The classes form a severity ladder used by safety profiles and by the
    "never downgrade" post-condition on LLM curation:

    ``READ (0) < WRITE (1) < DESTRUCTIVE (2) == FINANCIAL (2)``

    ``DESTRUCTIVE`` and ``FINANCIAL`` share the top severity: both are
    excluded by the ``standard`` profile and both require approval under
    ``full``.  Reclassifying one as the other is therefore neither an
    escalation nor a downgrade.
    """

    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    FINANCIAL = "financial"

    @property
    def severity(self) -> int:
        """Position on the severity ladder (higher is more dangerous)."""
        return _SEVERITY[self]

    def escalate(self) -> RiskClass:
        """Return the class one level up the ladder (top classes stay put)."""
        if self is RiskClass.READ:
            return RiskClass.WRITE
        if self is RiskClass.WRITE:
            return RiskClass.DESTRUCTIVE
        return self

    def at_least(self, other: RiskClass) -> bool:
        """``True`` if this class is at least as severe as *other*."""
        return self.severity >= other.severity


_SEVERITY: dict[RiskClass, int] = {
    RiskClass.READ: 0,
    RiskClass.WRITE: 1,
    RiskClass.DESTRUCTIVE: 2,
    RiskClass.FINANCIAL: 2,
}


class SafetyProfile(str, Enum):
    """How much of the API an agent is allowed to reach.

    ==============  ========  =======================  ==============================
    Profile         ``read``  ``write``                ``destructive`` / ``financial``
    ==============  ========  =======================  ==============================
    ``read-only``   exposed   not generated            not generated
    ``standard``    exposed   ``requires_approval``    not generated
    ``full``        exposed   ``requires_approval``    exposed, ``requires_approval``
    ==============  ========  =======================  ==============================
    """

    READ_ONLY = "read-only"
    STANDARD = "standard"
    FULL = "full"

    def allows(self, risk: RiskClass) -> bool:
        """Whether a tool of class *risk* may be generated under this profile."""
        if self is SafetyProfile.READ_ONLY:
            return risk is RiskClass.READ
        if self is SafetyProfile.STANDARD:
            return risk in (RiskClass.READ, RiskClass.WRITE)
        return True

    def requires_approval(self, risk: RiskClass) -> bool:
        """Whether a generated tool of class *risk* must be approval-gated."""
        return risk is not RiskClass.READ

    def exclusion_reason(self, risk: RiskClass) -> str:
        """Human-readable reason recorded when *risk* is excluded by this profile."""
        return f"{risk.value} operation excluded by profile {self.value!r}"


class AuthMode(str, Enum):
    """How the generated server authenticates against the upstream API."""

    PASSTHROUGH = "passthrough"
    """Forward the MCP caller's ``Authorization`` header to the API (HTTP/SSE
    deployments where every caller brings their own token)."""

    ENV_TOKEN = "env-token"
    """Present one upstream credential from ``MCPCAST_UPSTREAM_TOKEN`` on every
    call — the standard pattern for a personal server launched over stdio by
    Claude Desktop, Claude Code or Cursor."""

    API_KEY = "api-key"
    """MCP clients present an API key; each key maps to a tenant whose
    upstream credential is configured on the server."""

    NONE = "none"
    """No credentials.  Local/dev only — the server refuses to bind to a
    non-loopback address."""


class ApprovalMode(str, Enum):
    """Who approves approval-gated tool calls."""

    ELICITATION = "elicitation"
    """Ask the human behind the *calling* MCP client (confirm-your-own-action)."""

    PENDING = "pending"
    """Park the call in a pending store until a *different* human holding the
    ``approver`` role decides (independent four-eyes review)."""


HttpMethod = Literal["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "TRACE"]
ParamLocation = Literal["path", "query", "body", "raw_body"]
CredentialLocation = Literal["header", "query"]
"""Where the generated server presents the upstream credential: a request
header (``Authorization``, ``X-API-Key``…) or a query parameter (``api_key``)."""

TOOL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_API_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,62}$")
_PATH_PLACEHOLDER = re.compile(r"\{([^{}/]+)\}")
# RFC 7230 ``token``: the characters an HTTP header name may contain.
_HEADER_NAME = re.compile(r"^[A-Za-z0-9!#$%&'*+.^_`|~-]+$")

RESERVED_CREDENTIAL_HEADERS: frozenset[str] = frozenset(
    {"cookie", "host", "content-length", "transfer-encoding"}
)
"""Headers the HTTP client owns; a plan may not route the upstream credential through them."""

RESERVED_TOOL_NAMES: frozenset[str] = frozenset(
    {
        # registered by the approval machinery
        "approvals_list",
        "approvals_decide",
        # names the generated handler bodies rely on — a tool function of the
        # same name would shadow them inside the registration scope
        "server",
        "upstream",
        "ctx",
        "args",
        "main",
        "register",
        "select_route",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "tuple",
        "set",
        "object",
    }
)
"""Tool names that would break or hijack the generated server."""

RESERVED_PARAM_NAMES: frozenset[str] = frozenset(
    {
        # names the generated handler bodies use themselves
        "ctx",
        "self",
        "args",
        "server",
        "upstream",
        "ROUTES",
        "select_route",
        # names the generated signatures use as annotations: a parameter
        # called ``str`` or ``Any`` would shadow the type inside the module
        "str",
        "Any",
        # pydantic BaseModel attributes: the server builds the tool's input
        # model from the signature, and a field of these names shadows them
        "validate",
        "schema",
        "json",
        "dict",
        "copy",
        "construct",
        "fields",
        "parse_obj",
        "parse_raw",
        "parse_file",
        "from_orm",
        "schema_json",
        "update_forward_refs",
    }
)
"""Parameter identifiers that would collide inside the generated server."""


_CONTROL_CHARACTERS = str.maketrans(
    dict.fromkeys((*range(0x00, 0x09), *range(0x0B, 0x20), 0x7F, *range(0x80, 0xA0)), None)
)
"""C0 and C1 control characters except newline and tab — the bytes terminal
escape sequences are built from (ESC, the C1 CSI 0x9B, OSC…)."""


def scrub_text(value: str) -> str:
    """*value* with control characters deleted and lone surrogates replaced.

    Spec and model text ends up in the ``--review`` tables, the wizard, tool
    descriptions and the README; an ``\\x1b[2K`` in a summary would blank the
    row a reviewer is looking at and an OSC 52 sequence would write the
    clipboard. Every C0/C1 control character except ``\\n`` and ``\\t`` is
    deleted. A lone UTF-16 surrogate (``"\\ud800"``, which JSON and YAML both
    accept) is replaced so the string can be written as UTF-8.

    Args:
        value: Any string from a spec, a model response or a plan file.

    Returns:
        The cleaned string (*value* itself when nothing had to change).
    """
    return value.translate(_CONTROL_CHARACTERS).encode("utf-8", "replace").decode("utf-8")


def _scrub_tree(node: Any) -> Any:
    """:func:`scrub_text` applied to every string in *node* — keys and values, any depth."""
    if isinstance(node, str):
        return scrub_text(node)
    if isinstance(node, dict):
        return {_scrub_tree(k): _scrub_tree(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_scrub_tree(v) for v in node]
    return node


def _scrub_string(value: Any) -> Any:
    """``mode="before"`` scrub for one string field; non-strings pass through to the type check."""
    return scrub_text(value) if isinstance(value, str) else value


def _scrub_keys(value: Any) -> Any:
    """``mode="before"`` scrub for the keys of a ``params`` mapping (the parameter names)."""
    if isinstance(value, dict):
        return {(scrub_text(k) if isinstance(k, str) else k): v for k, v in value.items()}
    return value


def _jsonable(value: Any) -> Any:
    """*value* with non-JSON scalars (YAML dates, Decimals, ``inf``/``nan``) rendered as
    strings and every string scrubbed (see :func:`scrub_text`)."""
    return _scrub_tree(json.loads(json.dumps(value, default=str), parse_constant=str))


_URL_TOKEN_BREAKERS = re.compile(r"[\s\x00-\x1f\x7f]")
_URL_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*:")


def _url_authority(value: str) -> str:
    """The authority (``user:pw@host:port``) of *value*, read leniently.

    ``urlsplit`` finds an authority only after ``//``; ``user:pw@host/v1``
    parses as scheme ``user`` and ``mailto:user:pw@host`` as a path, so a
    credential written without a scheme (or with a single slash) would slip
    past a netloc check. Here an optional scheme and any run of slashes are
    stripped and everything up to the first ``/``, ``?`` or ``#`` is the
    authority, whatever the shape of the value.
    """
    rest = _URL_SCHEME.sub("", value, count=1).lstrip("/")
    return re.split(r"[/?#]", rest, maxsplit=1)[0]


def _url_token(value: str, label: str) -> str:
    """*value* stripped and without a trailing slash, refused unless it is one clean URL token.

    Userinfo (``https://user:token@host``) is refused first, whether the plan
    was generated or hand-edited: the value is copied into ``config.py``,
    ``.env.example``, the README and the server's ``instructions`` (sent to
    every MCP client), none of which may carry a credential. The generated
    server authenticates with ``MCPCAST_UPSTREAM_TOKEN`` (``--auth
    env-token``) instead. No refusal echoes the value: the credential message
    names the host only, the whitespace message names the character and its
    offset, and the query/fragment message names nothing.

    Whitespace and control characters are refused because the value is
    written into files a shell, Docker or ``--env-file`` reads line by line,
    so a newline in it would become a second line there. A query string or
    fragment is refused because route paths are appended to the base URL
    (``httpx`` replaces a base query with the request's own parameters, so
    ``?api_key=…`` would never be sent) and because the value is copied into
    the files above.
    """
    value = value.strip().rstrip("/")
    authority = _url_authority(value)
    if "@" in authority:
        host = _URL_TOKEN_BREAKERS.sub("?", authority.rpartition("@")[2])
        raise ValueError(
            f"{label} must not carry credentials (user:password@ before {host!r}): the "
            "generated server never stores an upstream credential in its files — use "
            "--auth env-token and set MCPCAST_UPSTREAM_TOKEN in its environment instead"
        )
    breaker = _URL_TOKEN_BREAKERS.search(value)
    if breaker is not None:
        raise ValueError(
            f"{label} contains whitespace or control characters ({breaker.group()!r} at "
            f"position {breaker.start()})"
        )
    parts = urlsplit(value)
    if parts.query or parts.fragment:
        raise ValueError(
            f"{label} must not carry a query string or fragment: route paths are appended to "
            "it and the value is copied into config.py, README.md, .env.example and the "
            "server instructions — pass query parameters through the tool's own arguments, "
            "or a credential via --auth env-token"
        )
    return value


def render_validation_errors(exc: ValidationError) -> str:
    """pydantic's error listing (title, location, message) without ``input_value``.

    A plan refused for carrying a credential in a ``base_url`` must not have
    that credential echoed back by the refusal — the message goes to a
    terminal, a CI log, a bug report. The location and the message already
    say which value is wrong; the plan author has the file.
    """
    errors = exc.errors(include_url=False, include_input=False)
    lines = [f"{len(errors)} validation error{'s' if len(errors) != 1 else ''} for {exc.title}"]
    for error in errors:
        lines.append(".".join(str(part) for part in error["loc"]) or "(root)")
        lines.append(f"  {error['msg']}")
    return "\n".join(lines)


SOFT_KEYWORDS: frozenset[str] = frozenset({"_", "case", "match", "type"})
"""Soft keywords the generator treats as reserved on every Python version.

:func:`keyword.issoftkeyword` answers for the running interpreter only (``type``
is soft from 3.12), so a plan generated on 3.10 would name a tool ``type`` while
3.12 named it ``type_op``. A fixed set keeps generated names identical across
the supported versions.
"""


def is_python_keyword(name: str) -> bool:
    """``True`` if *name* is a Python keyword, hard or soft, on any supported version."""
    return keyword.iskeyword(name) or name in SOFT_KEYWORDS


def valid_tool_name(name: str) -> str | None:
    """Why *name* cannot be a tool name, or ``None`` if it can."""
    if not TOOL_NAME_PATTERN.fullmatch(name):
        return (
            f"tool name {name!r} must match {TOOL_NAME_PATTERN.pattern} "
            "(lowercase snake_case, max 64 chars)"
        )
    if is_python_keyword(name):
        return f"tool name {name!r} is a Python keyword"
    if name in RESERVED_TOOL_NAMES:
        return f"tool name {name!r} is reserved by the generated server"
    return None


# ---------------------------------------------------------------------------
# Plan models
# ---------------------------------------------------------------------------


class RouteParam(BaseModel):
    """Where one parameter of a route travels on the wire."""

    model_config = ConfigDict(extra="forbid")

    location: ParamLocation
    required: bool = False
    wire_name: str | None = None
    """The name sent to the API when it differs from the tool parameter name
    (e.g. a body property that collides with a path parameter is exposed as
    ``body_<name>`` and sent as ``<name>``)."""

    @field_validator("wire_name", mode="before")
    @classmethod
    def _clean_wire_name(cls, v: Any) -> Any:
        return _scrub_string(v)


class RoutePlan(BaseModel):
    """One upstream HTTP operation a tool can execute.

    A tool with several routes dispatches to the first route whose required
    parameters were all supplied (in plan order), so a collapsed tool such
    as ``find_customer(id | email)`` can serve both ``GET /customers/{id}``
    and ``GET /customers?email=``.
    """

    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(..., min_length=1)
    method: HttpMethod
    path: str = Field(..., min_length=1)
    params: dict[str, RouteParam] = Field(default_factory=dict)
    body_encoding: Literal["json", "form"] = "json"
    base_url: str | None = None
    """Operation-level server override (only when it differs from ``api.base_url``)."""

    @field_validator("operation_id", "path", mode="before")
    @classmethod
    def _clean_text(cls, v: Any) -> Any:
        return _scrub_string(v)

    @field_validator("params", mode="before")
    @classmethod
    def _clean_param_names(cls, v: Any) -> Any:
        return _scrub_keys(v)

    @field_validator("base_url")
    @classmethod
    def _valid_route_base(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = _url_token(v, "route base_url")
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"route base_url must be absolute (got {v!r})")
        return v

    @field_validator("path")
    @classmethod
    def _path_starts_with_slash(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError(f"route path must start with '/': {v!r}")
        return v

    @property
    def path_params(self) -> tuple[str, ...]:
        """Placeholders that appear in the path template, in order."""
        return tuple(_PATH_PLACEHOLDER.findall(self.path))

    @property
    def required_params(self) -> tuple[str, ...]:
        """Parameter names this route requires, in declaration order."""
        return tuple(name for name, p in self.params.items() if p.required)

    def wire(self, name: str) -> str:
        """The wire name for tool parameter *name*."""
        param = self.params.get(name)
        return param.wire_name if param is not None and param.wire_name else name

    @model_validator(mode="after")
    def _placeholders_declared(self) -> RoutePlan:
        declared_path = {self.wire(n) for n, p in self.params.items() if p.location == "path"}
        placeholders = set(self.path_params)
        missing = placeholders - declared_path
        if missing:
            raise ValueError(
                f"route {self.operation_id!r}: path placeholders {sorted(missing)} "
                "have no matching path parameter"
            )
        extra = declared_path - placeholders
        if extra:
            raise ValueError(
                f"route {self.operation_id!r}: path parameters {sorted(extra)} "
                f"do not appear in path {self.path!r}"
            )
        raw = [n for n, p in self.params.items() if p.location == "raw_body"]
        if len(raw) > 1:
            raise ValueError(f"route {self.operation_id!r}: only one raw_body parameter is allowed")
        return self


class ParamPlan(BaseModel):
    """The agent-facing shape of one tool parameter."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    json_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "string"})
    required: bool = False
    default: Any = None
    hidden: bool = False
    """Hidden parameters are not exposed to the agent; ``default`` is sent
    on every call instead (the "param diet")."""

    @field_validator("description", mode="before")
    @classmethod
    def _clean_description(cls, v: Any) -> Any:
        return _scrub_string(v)

    @field_validator("json_schema", "default", mode="before")
    @classmethod
    def _json_native(cls, v: Any) -> Any:
        # YAML turns unquoted dates into datetime objects; everything that
        # reaches generated code or the wire must be JSON-native.
        return _jsonable(v) if v is not None else None

    @model_validator(mode="after")
    def _hidden_needs_default_when_required(self) -> ParamPlan:
        if self.hidden and self.required and self.default is None:
            raise ValueError("a hidden parameter that is required must have a default")
        return self


class ToolPlan(BaseModel):
    """One generated MCP tool."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    risk: RiskClass
    routes: list[RoutePlan] = Field(..., min_length=1)
    params: dict[str, ParamPlan] = Field(default_factory=dict)
    example: dict[str, Any] | None = None
    requires_approval: bool = False
    tags: list[str] = Field(default_factory=list)

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        problem = valid_tool_name(v)
        if problem:
            raise ValueError(problem)
        return v

    @field_validator("example", mode="before")
    @classmethod
    def _example_json_native(cls, v: Any) -> Any:
        return _jsonable(v) if v is not None else None

    @field_validator("description", mode="before")
    @classmethod
    def _clean_description(cls, v: Any) -> Any:
        return _scrub_string(v)

    @field_validator("tags", mode="before")
    @classmethod
    def _clean_tags(cls, v: Any) -> Any:
        return [_scrub_string(t) for t in v] if isinstance(v, list) else v

    @field_validator("params", mode="before")
    @classmethod
    def _clean_param_names(cls, v: Any) -> Any:
        return _scrub_keys(v)

    @field_validator("description")
    @classmethod
    def _non_blank_description(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("tool description must not be blank")
        return v.strip()

    @property
    def operations(self) -> list[str]:
        """Operation ids this tool covers, in dispatch order."""
        return [r.operation_id for r in self.routes]

    @property
    def visible_params(self) -> dict[str, ParamPlan]:
        """Parameters the agent sees (everything not hidden)."""
        return {n: p for n, p in self.params.items() if not p.hidden}

    @model_validator(mode="after")
    def _consistent(self) -> ToolPlan:
        seen: set[str] = set()
        for route in self.routes:
            if route.operation_id in seen:
                raise ValueError(f"tool {self.name!r} lists operation {route.operation_id!r} twice")
            seen.add(route.operation_id)
            unknown = set(route.params) - set(self.params)
            if unknown:
                raise ValueError(
                    f"tool {self.name!r}: route {route.operation_id!r} uses parameters "
                    f"{sorted(unknown)} that are not declared in params"
                )
            for pname, rparam in route.params.items():
                plan = self.params[pname]
                if rparam.required and plan.hidden and plan.default is None:
                    raise ValueError(
                        f"tool {self.name!r}: parameter {pname!r} is required by route "
                        f"{route.operation_id!r} but hidden without a default"
                    )
        # Dispatch picks the first route whose required (visible) parameters
        # were supplied, so a later route whose requirement set contains an
        # earlier route's can never be selected — its operation would vanish.
        effective: list[set[str]] = [
            {n for n in r.required_params if not self.params[n].hidden} for r in self.routes
        ]
        for i, later in enumerate(effective):
            for j in range(i):
                if effective[j] <= later:
                    raise ValueError(
                        f"tool {self.name!r}: route {self.routes[i].operation_id!r} can never "
                        f"be selected — route {self.routes[j].operation_id!r} (listed earlier) "
                        "is satisfied by any call that satisfies it; reorder the routes so "
                        "the more specific one comes first"
                    )
        if self.example is not None:
            bad = set(self.example) - set(self.visible_params)
            if bad:
                raise ValueError(
                    f"tool {self.name!r}: example uses unknown or hidden parameters {sorted(bad)}"
                )
        return self


class DroppedOp(BaseModel):
    """An operation deliberately left out of the tool surface."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str = Field(..., min_length=1)
    reason: str = Field(..., min_length=1)

    @field_validator("operation_id", "reason", mode="before")
    @classmethod
    def _clean_text(cls, v: Any) -> Any:
        return _scrub_string(v)


class ApiPlan(BaseModel):
    """The upstream API and how the generated server talks to it."""

    model_config = ConfigDict(extra="forbid")

    name: str
    base_url: str
    auth: AuthMode = AuthMode.PASSTHROUGH
    approval: ApprovalMode | None = None
    description: str = ""
    spec_source: str | None = None
    credential_location: CredentialLocation = "header"
    """Where the generated server presents the upstream credential under
    ``env-token`` and ``api-key``: a request header or a query parameter.
    Set from the spec's effective security scheme (an ``apiKey`` scheme names
    its location); HTTP ``bearer``/``basic``, OAuth 2 and OpenID Connect all
    travel in the ``Authorization`` header, the default."""
    credential_name: str = "Authorization"
    """The header or query parameter that carries the credential
    (``Authorization``, ``X-API-Key``, ``api_key``…). ``passthrough`` can only
    relay the caller's ``Authorization`` header, so a plan that names anything
    else under that mode is refused at build time."""

    @field_validator("description", "spec_source", mode="before")
    @classmethod
    def _clean_text(cls, v: Any) -> Any:
        return _scrub_string(v)

    @field_validator("name")
    @classmethod
    def _valid_api_name(cls, v: str) -> str:
        if not _API_NAME_PATTERN.fullmatch(v):
            raise ValueError(
                f"api name {v!r} must match {_API_NAME_PATTERN.pattern} "
                "(lowercase, digits, '-' or '_')"
            )
        return v

    @field_validator("base_url")
    @classmethod
    def _valid_base_url(cls, v: str) -> str:
        v = _url_token(v, "base_url")
        if not (v.startswith("http://") or v.startswith("https://")):
            hint = (
                f"the spec declares a relative server URL {v!r}; pass "
                f"--base-url https://<api-host>{v}"
                if v.startswith("/")
                else "pass --base-url if the spec does not declare a server"
            )
            raise ValueError(f"base_url must start with http:// or https:// (got {v!r}); {hint}")
        return v

    @field_validator("spec_source")
    @classmethod
    def _public_spec_source(cls, v: str | None) -> str | None:
        """A URL source keeps only scheme, host and path.

        Userinfo (``user:pass@``), the query string (``?api_key=…``) and the
        fragment are dropped: the source is echoed into generated docstrings,
        the README and the plan file, none of which may carry a credential.
        """
        if v is None or not v.strip().lower().startswith(("http://", "https://")):
            return v
        parts = urlsplit(v.strip())
        host = parts.netloc.rpartition("@")[2]
        return urlunsplit((parts.scheme, host, parts.path, "", ""))

    @model_validator(mode="after")
    def _pending_needs_identified_callers(self) -> ApiPlan:
        if self.approval is ApprovalMode.PENDING and self.auth is not AuthMode.API_KEY:
            raise ValueError(
                "approval mode 'pending' (independent four-eyes review) needs identified "
                "callers so a caller can never approve their own request — use "
                "--auth api-key, or approval 'elicitation'"
            )
        return self

    @model_validator(mode="after")
    def _valid_credential_slot(self) -> ApiPlan:
        name = self.credential_name
        if self.credential_location == "header":
            if not _HEADER_NAME.fullmatch(name):
                raise ValueError(
                    f"credential_name {name!r} is not a valid HTTP header name (letters, digits "
                    "and the RFC 7230 token characters only)"
                )
            if name.lower() in RESERVED_CREDENTIAL_HEADERS:
                raise ValueError(
                    f"credential_name {name!r}: the {name.lower()} header belongs to the HTTP "
                    "client and cannot carry the upstream credential"
                )
        elif not name or _URL_TOKEN_BREAKERS.search(name):
            raise ValueError(
                "credential_name for a query credential must be a non-empty parameter name "
                "without whitespace or control characters"
            )
        return self

    @property
    def credential_is_authorization_header(self) -> bool:
        """``True`` when the credential is the ``Authorization`` header (what ``passthrough`` relays)."""
        return (
            self.credential_location == "header" and self.credential_name.lower() == "authorization"
        )

    @property
    def approval_mode(self) -> ApprovalMode:
        """Effective approver: explicit setting, else ``pending`` for
        ``api-key`` auth (clients are identified) and ``elicitation`` otherwise."""
        if self.approval is not None:
            return self.approval
        return ApprovalMode.PENDING if self.auth is AuthMode.API_KEY else ApprovalMode.ELICITATION


class _PlanDumper(yaml.SafeDumper):
    """``SafeDumper`` whose text survives a reload byte for byte.

    With ``allow_unicode=True`` PyYAML writes U+0085 (NEL) literally in a
    plain or single-quoted scalar, and YAML 1.1 reads a literal NEL as a line
    break — ``'a\\x85b'`` came back as ``'a b'``. Any string holding one is
    forced into double-quoted style, where the emitter escapes it as ``\\N``.
    Validated plans never contain NEL (see :func:`scrub_text`); this keeps
    :meth:`MCPcastPlan.to_yaml` lossless for any string it is handed.
    """


def _represent_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str", value, style='"' if "\x85" in value else None
    )


_PlanDumper.add_representer(str, _represent_str)


class MCPcastPlan(BaseModel):
    """The complete, validated tool plan for one API."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = 1
    api: ApiPlan
    profile: SafetyProfile = SafetyProfile.READ_ONLY
    tools: list[ToolPlan] = Field(default_factory=list)
    dropped: list[DroppedOp] = Field(default_factory=list)

    # -- invariants ------------------------------------------------------

    @model_validator(mode="after")
    def _invariants(self) -> MCPcastPlan:
        # classify imports parse, which imports this module: resolve it late.
        from .classify import risk_floor

        names: set[str] = set()
        kept: dict[str, str] = {}
        for tool in self.tools:
            if tool.name in names:
                raise ValueError(f"duplicate tool name {tool.name!r}")
            names.add(tool.name)
            for op_id in tool.operations:
                if op_id in kept:
                    raise ValueError(
                        f"operation {op_id!r} is used by both {kept[op_id]!r} and {tool.name!r}"
                    )
                kept[op_id] = tool.name
            # A plan is an editing surface, not a way around the classifier: a
            # DELETE declared ``read`` would be emitted un-gated with
            # read_only_hint=True. The floor is what the classifier can tell
            # from the wire mapping alone (method, path, operation id).
            for route in tool.routes:
                floor = risk_floor(
                    operation_id=route.operation_id, method=route.method, path=route.path
                )
                if not tool.risk.at_least(floor):
                    raise ValueError(
                        f"tool {tool.name!r} is declared {tool.risk.value!r} but its route "
                        f"{route.operation_id!r} ({route.method} {route.path}) is at least "
                        f"{floor.value!r}: a plan may raise a tool's risk, never lower it"
                    )
            if not self.profile.allows(tool.risk):
                raise ValueError(
                    f"tool {tool.name!r} is {tool.risk.value} but profile "
                    f"{self.profile.value!r} does not allow it"
                )
            if self.profile.requires_approval(tool.risk) and not tool.requires_approval:
                raise ValueError(
                    f"tool {tool.name!r} is {tool.risk.value}: profile "
                    f"{self.profile.value!r} requires requires_approval=true"
                )
        dropped_ids: set[str] = set()
        for d in self.dropped:
            if d.operation_id in kept:
                raise ValueError(
                    f"operation {d.operation_id!r} is both kept (tool {kept[d.operation_id]!r}) "
                    "and dropped"
                )
            if d.operation_id in dropped_ids:
                raise ValueError(f"operation {d.operation_id!r} is dropped twice")
            dropped_ids.add(d.operation_id)
        return self

    # -- convenience -----------------------------------------------------

    @property
    def tool_names(self) -> list[str]:
        """Tool names in plan order."""
        return [t.name for t in self.tools]

    @property
    def gated_tools(self) -> list[ToolPlan]:
        """Tools that demand human approval."""
        return [t for t in self.tools if t.requires_approval]

    @property
    def kept_operations(self) -> set[str]:
        """Every operation id some tool exposes."""
        return {op for t in self.tools for op in t.operations}

    def tool(self, name: str) -> ToolPlan:
        """Look up a tool by name (raises ``KeyError`` if absent)."""
        for t in self.tools:
            if t.name == name:
                return t
        raise KeyError(name)

    # -- YAML round-trip -------------------------------------------------

    def to_yaml(self) -> str:
        """Serialise to the ``mcpcast.plan.yaml`` text form."""
        # Compact form: defaults are omitted everywhere except the keys a
        # reviewer must always see (version, api.auth, profile).
        data = self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
        api = {"name": self.api.name, "base_url": self.api.base_url, "auth": self.api.auth.value}
        api.update({k: v for k, v in data.get("api", {}).items() if k not in api})
        data = {
            "version": 1,
            "api": api,
            "profile": self.profile.value,
            "tools": data.get("tools", []),
            "dropped": data.get("dropped", []),
        }
        header = (
            "# mcpcast.plan.yaml — the editable source of truth for this MCP server.\n"
            "# Edit tool names, descriptions, examples, hidden params or the dropped\n"
            "# list, then regenerate (this file is left untouched):  promptise mcpcast mcpcast.plan.yaml\n"
        )
        return header + yaml.dump(
            data, Dumper=_PlanDumper, sort_keys=False, allow_unicode=True, width=100
        )

    @classmethod
    def from_yaml(cls, text: str) -> MCPcastPlan:
        """Parse and validate a plan from YAML text.

        Raises:
            MCPcastError: If the document is not a plan or fails validation.
        """
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise MCPcastError(f"plan is not valid YAML: {exc}") from exc
        return cls.from_document(data)

    @classmethod
    def from_document(cls, data: Any) -> MCPcastPlan:
        """Validate an already parsed plan document (a mapping).

        This is what :meth:`from_yaml` ends in, and what the CLI uses for a plan
        that :func:`~promptise.mcpcast.parse.load_spec` fetched from a URL or read
        from a file.

        Raises:
            MCPcastError: If *data* is not a mapping or fails validation.
        """
        if not isinstance(data, dict):
            raise MCPcastError("plan must be a YAML mapping")
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise MCPcastError(f"invalid plan:\n{render_validation_errors(exc)}") from exc

    @classmethod
    def load(cls, path: str | Path) -> MCPcastPlan:
        """Load a plan from ``path``."""
        return cls.from_yaml(Path(path).read_text(encoding="utf-8"))

    def save(self, path: str | Path) -> Path:
        """Write the plan to ``path`` and return it."""
        target = Path(path)
        target.write_text(self.to_yaml(), encoding="utf-8")
        return target


def is_plan_document(data: Any) -> bool:
    """``True`` if a parsed YAML/JSON document looks like an :class:`MCPcastPlan`
    rather than an OpenAPI spec."""
    return (
        isinstance(data, dict)
        and "tools" in data
        and "api" in data
        and "paths" not in data
        and "openapi" not in data
        and "swagger" not in data
    )
