"""``promptise mcpcast`` — turn an existing API into a curated, safe, agent-ready MCP server.

Pipeline::

    spec ──▶ parse ──▶ classify ──▶ curate ──▶ review ──▶ emit ──▶ eval
             (OpenAPI)  (risk)      (LLM)      (human)   (code)   (score)

- :func:`mcpcast` runs the deterministic path (parse → classify → plan) with
  no model and no network.
- :func:`curate` designs the surface with a model, with every
  post-condition enforced in code.
- :func:`write_project` emits the project: an installable package, a launcher
  ``server.py``, tests, ``README.md`` and the scaffold (``pyproject.toml``, ``Dockerfile``…).
- :func:`evaluate` scores how well a real agent can drive the result.

Example::

    from promptise.mcpcast import SafetyProfile, mcpcast, write_project

    plan = mcpcast("openapi.yaml", profile=SafetyProfile.STANDARD)
    write_project(plan, "myapi-mcp")
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._llm import Completer
from .classify import Classification, classify, classify_operation, risk_floor
from .curate import (
    CuratedParam,
    CuratedTool,
    CurationResult,
    CurationViolation,
    apply_curation,
    check_postconditions,
    curate,
    render_curation_prompt,
)
from .emit import (
    SCAFFOLD_ONCE,
    describe_written,
    load_generated_server,
    package_name,
    plain_http_hosts,
    render_project,
    render_readme,
    tool_group,
    write_project,
)
from .parse import (
    MAX_DOCUMENT_NODES,
    Operation,
    ParamSpec,
    SecurityScheme,
    api_name_from_spec,
    check_document,
    expanded_nodes,
    extract_operations,
    is_url,
    load_spec,
    public_url,
    scrub_strings,
    spec_base_url,
    spec_description,
    spec_summary_line,
    spec_title,
)
from .plan import (
    CredentialSlot,
    build_plan,
    credential_slot,
    derive_tool_name,
    example_mismatch,
    example_value,
    make_example,
    resolve_base_url,
    unmappable_reason,
)
from .readiness import (
    CallRecorder,
    ConfusedPair,
    EvalReport,
    EvalTask,
    TaskResult,
    ToolCall,
    evaluate,
    generate_tasks,
    grade_for,
    mock_transport,
    score,
    tools_from_server,
    write_eval,
)
from .schema import (
    RESERVED_TOOL_NAMES,
    ApiPlan,
    ApprovalMode,
    AuthMode,
    DroppedOp,
    MCPcastError,
    MCPcastPlan,
    ParamPlan,
    RiskClass,
    RouteParam,
    RoutePlan,
    SafetyProfile,
    ToolPlan,
    is_plan_document,
    valid_tool_name,
)

__all__ = [
    # schema
    "ApiPlan",
    "ApprovalMode",
    "AuthMode",
    "DroppedOp",
    "MCPcastError",
    "MCPcastPlan",
    "ParamPlan",
    "RiskClass",
    "RouteParam",
    "RoutePlan",
    "SafetyProfile",
    "ToolPlan",
    "RESERVED_TOOL_NAMES",
    "is_plan_document",
    "valid_tool_name",
    # parse
    "MAX_DOCUMENT_NODES",
    "Operation",
    "ParamSpec",
    "SecurityScheme",
    "api_name_from_spec",
    "check_document",
    "expanded_nodes",
    "extract_operations",
    "is_url",
    "load_spec",
    "public_url",
    "scrub_strings",
    "spec_base_url",
    "spec_description",
    "spec_summary_line",
    "spec_title",
    # classify
    "Classification",
    "classify",
    "classify_operation",
    "risk_floor",
    # plan
    "CredentialSlot",
    "build_plan",
    "credential_slot",
    "derive_tool_name",
    "example_mismatch",
    "example_value",
    "make_example",
    "mcpcast",
    "resolve_base_url",
    "unmappable_reason",
    # curate
    "Completer",
    "CuratedParam",
    "CuratedTool",
    "CurationResult",
    "CurationViolation",
    "apply_curation",
    "check_postconditions",
    "curate",
    "render_curation_prompt",
    # emit
    "SCAFFOLD_ONCE",
    "describe_written",
    "load_generated_server",
    "package_name",
    "plain_http_hosts",
    "render_project",
    "render_readme",
    "tool_group",
    "write_project",
    # readiness
    "CallRecorder",
    "ConfusedPair",
    "EvalReport",
    "EvalTask",
    "TaskResult",
    "ToolCall",
    "evaluate",
    "generate_tasks",
    "grade_for",
    "mock_transport",
    "score",
    "tools_from_server",
    "write_eval",
]


def mcpcast(
    source: str | Path | Mapping[str, Any],
    *,
    profile: SafetyProfile = SafetyProfile.READ_ONLY,
    base_url: str | None = None,
    auth: AuthMode = AuthMode.PASSTHROUGH,
    approval: ApprovalMode | None = None,
    name: str | None = None,
    max_tools: int | None = None,
) -> MCPcastPlan:
    """The deterministic pipeline: load → extract → classify → plan.

    No model, no network beyond fetching *source* when it is a URL.

    Args:
        source: OpenAPI spec as a file path, URL, JSON/YAML text, or dict.
        profile: Safety profile (default ``read-only``).
        base_url: Override the API base URL declared by the spec.
        auth: How the generated server authenticates against the API.
        approval: Who approves gated calls (defaults per *auth*).
        name: Server slug (defaults to the spec title).
        max_tools: Optional tool budget (reads kept first).

    Returns:
        A validated :class:`MCPcastPlan`.
    """
    spec = load_spec(source)
    if isinstance(source, Mapping):
        source_label = None
    elif str(source).lstrip().startswith(("{", "openapi:", "swagger:")):
        source_label = "<inline>"  # never record a whole document as its own source
    else:
        # A URL may carry a credential for the fetch (done above); nothing
        # derived from it — base URL, name, spec_source — may keep it.
        source_label = public_url(str(source))
    operations = extract_operations(
        spec, base_url=base_url, spec_url=source_label if is_url(source_label) else None
    )
    return build_plan(
        operations,
        profile=profile,
        base_url=base_url,
        auth=auth,
        approval=approval,
        name=name or api_name_from_spec(spec, source_label),
        description=spec_summary_line(spec),
        spec_source=source_label,
        max_tools=max_tools,
    )
