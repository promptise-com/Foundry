"""Agent Readiness Score for ``promptise mcpcast`` (``--eval``).

Tool design is usually guesswork.  This module turns it into measurement:

1. Generate realistic user tasks from the tool set (one expected tool each).
2. Run a real :func:`~promptise.agent.build_agent` against the generated
   server **in-process** through :class:`~promptise.mcp.server.TestClient`
   — no ports, no network between agent and server.  Reads may hit the live
   API; writes, destructive and financial calls hit spec-derived mocks and
   an auto-approver, so an evaluation never changes real data.
3. Measure task success, correct-tool selection, parameter errors, tools
   that were never used, and tool pairs the agent confuses.
4. Grade A–F and emit **specific tool-design fixes**.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import yaml
from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from ._llm import Completer, completer_for, extract_json_object, final_text
from .parse import Operation
from .plan import example_value
from .schema import AuthMode, MCPcastError, MCPcastPlan, RiskClass

DEFAULT_EVAL_TASKS = 20
"""How many tasks an evaluation generates when no number is given (the CLI and wizard default)."""

__all__ = [
    "DEFAULT_EVAL_TASKS",
    "NO_MOCK_STATUS",
    "CallRecorder",
    "ConfusedPair",
    "EvalReport",
    "EvalTask",
    "EvalTransport",
    "TaskResult",
    "ToolCall",
    "base_url_override",
    "credential_slot",
    "evaluate",
    "generate_tasks",
    "grade_for",
    "mock_transport",
    "score",
    "tools_from_server",
    "write_eval",
]

EVAL_AGENT_INSTRUCTIONS = (
    "You are an assistant that completes the user's request by calling the available "
    "tools. Call a tool whenever the request needs data or an action; do not ask "
    "clarifying questions. When the task is done, answer briefly."
)

TASK_SYSTEM = """\
You write realistic evaluation tasks for an AI agent that has a set of MCP tools.
Each task is one natural user request that a competent agent should complete by calling
ONE specific tool (it may need to call that tool once or twice). Tasks must be concrete —
include the identifiers, names, or values the agent needs, taken from the tool examples.
Spread tasks across the tools; every tool should be the target of at least one task when
the count allows. Phrase tasks the way real users talk, not like API documentation.

Respond with ONE JSON object and nothing else:
{"tasks": [{"id": "t1", "prompt": "…", "expected_tool": "tool_name", "rationale": "…"}]}
"""


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class EvalTask(BaseModel):
    """One user request and the tool it is designed to exercise."""

    model_config = ConfigDict(extra="ignore")

    id: str
    prompt: str = Field(..., min_length=1)
    expected_tool: str
    rationale: str = ""


class ToolCall(BaseModel):
    """One tool invocation observed during a task."""

    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    error_code: str | None = None
    details_status: int | None = None
    """Upstream HTTP status when the tool reported one."""


class TaskResult(BaseModel):
    """What happened when the agent attempted one task."""

    task: EvalTask
    calls: list[ToolCall] = Field(default_factory=list)
    success: bool = False
    selected_correctly: bool = False
    answer: str = ""
    error: str | None = None


class ConfusedPair(BaseModel):
    """The agent chose *chosen* first when *expected* was the target."""

    expected: str
    chosen: str
    count: int


class EvalReport(BaseModel):
    """The Agent Readiness Score with its evidence."""

    grade: str
    score: float
    tasks_total: int
    tasks_succeeded: int
    selection_rate: float
    param_error_rate: float
    never_used: list[str] = Field(default_factory=list)
    """Tools some task targeted that the agent never called."""
    not_covered: list[str] = Field(default_factory=list)
    """Tools no task targeted — nothing is known about them; raise the task count."""
    confused_pairs: list[ConfusedPair] = Field(default_factory=list)
    fixes: list[str] = Field(default_factory=list)
    results: list[TaskResult] = Field(default_factory=list)

    def render_markdown(self) -> str:
        """The human-readable report."""
        lines = [
            f"# Agent Readiness: {self.grade}  ({self.tasks_succeeded}/{self.tasks_total} tasks succeeded)",
            "",
            f"- Score: {self.score:.2f}",
            f"- Correct tool selected first: {self.selection_rate:.0%}",
            f"- Parameter error rate: {self.param_error_rate:.0%}",
            f"- Tools never used: {len(self.never_used)}",
            f"- Tools not covered by any task: {len(self.not_covered)}",
            "",
        ]
        if self.fixes:
            lines.append("## Fixes")
            lines.append("")
            lines.extend(f"- {fix}" for fix in self.fixes)
            lines.append("")
        lines.append("## Tasks")
        lines.append("")
        lines.append("| # | Task | Expected | Called | Result |")
        lines.append("|---|---|---|---|---|")
        for r in self.results:
            called = " → ".join(c.tool + ("" if c.ok else " ✗") for c in r.calls) or "—"
            status = "✓" if r.success else ("error" if r.error else "✗")
            prompt = r.task.prompt.replace("|", "\\|")
            lines.append(
                f"| {r.task.id} | {prompt} | `{r.task.expected_tool}` | {called} | {status} |"
            )
        lines.append("")
        return "\n".join(lines)

    def render_summary(self) -> str:
        """A compact terminal summary."""
        lines = [
            f"Agent Readiness: {self.grade}  ({self.tasks_succeeded}/{self.tasks_total} tasks succeeded)"
        ]
        lines.extend(f"  {fix}" for fix in self.fixes)
        return "\n".join(lines)


def grade_for(score: float) -> str:
    """Letter grade for a score in ``[0, 1]``."""
    if score >= 0.9:
        return "A"
    if score >= 0.75:
        return "B"
    if score >= 0.6:
        return "C"
    if score >= 0.4:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------


def _tool_catalogue(plan: MCPcastPlan) -> str:
    entries = [
        {
            "name": t.name,
            "risk": t.risk.value,
            "description": t.description,
            "params": {
                n: {"required": p.required, "type": p.json_schema.get("type", "string")}
                for n, p in t.visible_params.items()
            },
            "example": t.example,
        }
        for t in plan.tools
    ]
    return "\n".join(json.dumps(e, ensure_ascii=False) for e in entries)


async def generate_tasks(
    plan: MCPcastPlan,
    *,
    model: Any = "openai:gpt-5-mini",
    count: int = 20,
    complete: Completer | None = None,
) -> list[EvalTask]:
    """Generate up to *count* tasks from the plan's tool set.

    Tasks naming a tool that does not exist are discarded; ids are made
    unique.  Raises :class:`MCPcastError` if the model produced no usable task.
    """
    if not plan.tools:
        raise MCPcastError("cannot evaluate a plan with no tools")
    completer = complete or completer_for(model)
    prompt = (
        f"Write {count} tasks for this tool set (API: {plan.api.name}).\n\n"
        "Tools (JSON lines):\n" + _tool_catalogue(plan)
    )
    text = await completer(TASK_SYSTEM, prompt)
    try:
        raw = extract_json_object(text).get("tasks", [])
    except MCPcastError as exc:
        raise MCPcastError(f"task generation failed: {exc}") from exc
    names = set(plan.tool_names)
    tasks: list[EvalTask] = []
    seen: set[str] = set()
    counter = 0
    for item in raw if isinstance(raw, list) else []:
        try:
            task = EvalTask.model_validate(item)
        except ValidationError:
            continue
        if task.expected_tool not in names:
            continue
        if not task.id or task.id in seen:
            counter += 1
            while f"t{counter}" in seen:
                counter += 1
            task.id = f"t{counter}"
        seen.add(task.id)
        tasks.append(task)
        if len(tasks) >= count:
            break
    if not tasks:
        raise MCPcastError("task generation produced no task that targets an existing tool")
    return tasks


# ---------------------------------------------------------------------------
# Server bridge
# ---------------------------------------------------------------------------


class CallRecorder:
    """Records every tool call the agent makes, grouped by task id."""

    def __init__(self) -> None:
        self._current: str | None = None
        self._calls: dict[str, list[ToolCall]] = {}

    def begin(self, task_id: str) -> None:
        """Attribute subsequent calls to *task_id*."""
        self._current = task_id
        self._calls.setdefault(task_id, [])

    def record(self, call: ToolCall) -> None:
        """Record one call under the current task."""
        self._calls.setdefault(self._current or "_", []).append(call)

    def calls_for(self, task_id: str) -> list[ToolCall]:
        """The calls recorded for *task_id*, in order."""
        return list(self._calls.get(task_id, []))

    @property
    def all_calls(self) -> list[ToolCall]:
        """Every recorded call across tasks."""
        return [c for calls in self._calls.values() for c in calls]


def _error_detail(text: str) -> tuple[str | None, int | None]:
    """``(error code, upstream HTTP status)`` from a serialised tool error."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, None
    if isinstance(data, dict) and isinstance(data.get("error"), dict):
        error = data["error"]
        status = (error.get("details") or {}).get("status")
        return str(error.get("code") or "ERROR"), status if isinstance(status, int) else None
    return None, None


async def tools_from_server(
    server: Any,
    *,
    recorder: CallRecorder | None = None,
    headers: dict[str, str] | None = None,
) -> list[BaseTool]:
    """LangChain tools that call *server* in-process through ``TestClient``.

    Each call runs the full server pipeline (validation, guards, middleware,
    approval gate, handler).  Structured errors are returned as text so the
    agent can react, and are recorded with their error code.
    """
    from promptise.mcp.server import TestClient
    from promptise.tools import _jsonschema_to_pydantic

    client = TestClient(server, meta=dict(headers or {}))
    tools: list[BaseTool] = []
    for mcp_tool in await client.list_tools():
        name = mcp_tool.name

        def _make(tool_name: str) -> tuple[Any, Any]:
            async def _call(**kwargs: Any) -> str:
                content = await client.call_tool(tool_name, kwargs)
                text = "\n".join(str(getattr(c, "text", c)) for c in content)
                code, status = _error_detail(text)
                if recorder is not None:
                    recorder.record(
                        ToolCall(
                            tool=tool_name,
                            arguments=kwargs,
                            ok=code is None,
                            error_code=code,
                            details_status=status,
                        )
                    )
                return text

            def _on_validation_error(exc: Any) -> str:
                # LangChain validates against the typed schema before the call
                # reaches the server. A real MCP client would have sent it and
                # got the server's VALIDATION_ERROR back — record it the same way
                # so parameter errors count, and hand the agent the same shape.
                if recorder is not None:
                    recorder.record(
                        ToolCall(tool=tool_name, ok=False, error_code="VALIDATION_ERROR")
                    )
                return json.dumps(
                    {"error": {"code": "VALIDATION_ERROR", "message": str(exc)}}, indent=2
                )

            return _call, _on_validation_error

        call, on_validation_error = _make(name)
        tools.append(
            StructuredTool.from_function(
                coroutine=call,
                name=name,
                description=mcp_tool.description or name,
                args_schema=_jsonschema_to_pydantic(
                    mcp_tool.inputSchema or {}, model_name=f"{name}_args"
                ),
                handle_validation_error=on_validation_error,
            )
        )
    return tools


# ---------------------------------------------------------------------------
# Upstream mocks
# ---------------------------------------------------------------------------


def _template_regex(base_path: str, path: str) -> re.Pattern[str]:
    """Full-match regex for *base_path* + a templated *path* (``{x}`` → one segment)."""
    template = re.sub(
        r"\{[^{}/]+\}", r"[^/]+", re.escape(path).replace(r"\{", "{").replace(r"\}", "}")
    )
    return re.compile(re.escape(base_path.rstrip("/")) + template + r"/?")


def _specificity(path: str) -> tuple[int, int, int]:
    """Sort key putting the most specific route first.

    Literal segments beat placeholders: ``/users/wipe`` (two literals) sorts
    ahead of ``/users/{id}`` (one literal, one placeholder) however long the
    placeholder's regex is. Ties go to the longer literal text.
    """
    segments = [s for s in path.strip("/").split("/") if s]
    templated = [s for s in segments if "{" in s]
    literal = [s for s in segments if "{" not in s]
    return (-len(literal), len(templated), -sum(len(s) for s in literal))


def base_url_override() -> str | None:
    """``MCPCAST_BASE_URL`` exactly as the generated ``config.py`` reads it (``None`` when unset).

    The generated server sends every request to ``MCPCAST_BASE_URL`` when it
    is set — replacing the plan's base URL *and* any operation-level server —
    so the evaluation transport must expect the same paths, or a live read
    would never be recognised as one.
    """
    return os.environ.get("MCPCAST_BASE_URL", "").strip().rstrip("/") or None


NO_MOCK_STATUS = 502
"""HTTP status the evaluation transport answers a request it has no route for.

Not a 2xx — a fake success would grade the agent on nothing — and not a 404,
which the score reads as "the example identifier does not exist upstream".
"""


class EvalTransport(httpx.AsyncBaseTransport):
    """The evaluation's split transport: live reads, spec-derived mocks, nothing invented.

    Built by :func:`mock_transport`; the attribute :attr:`unmatched` is what
    :func:`score` needs to report requests that reached no route.
    """

    def __init__(
        self,
        routes: Sequence[tuple[str, re.Pattern[str], str, RiskClass]],
        responses: dict[str, dict[str, Any] | None],
        *,
        live_reads: bool,
        expected_bases: Sequence[str],
    ) -> None:
        self._routes = list(routes)
        self._responses = responses
        self._live_reads = live_reads
        self._expected_bases = list(expected_bases)
        self._real = httpx.AsyncHTTPTransport()
        self.unmatched: list[str] = []
        """``"METHOD /path"`` of every request no route in the plan matched, in order."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Route *request*: live for a ``read`` tool's route, a mock for every other route.

        A request that matches no route is answered with :data:`NO_MOCK_STATUS`
        and a body saying so — never with an invented success.
        """
        matches = [
            (op_id, risk)
            for method, regex, op_id, risk in self._routes
            if method == request.method and regex.fullmatch(request.url.path)
        ]
        # The most specific route answers the mock. Going live is decided over
        # *every* route the request matches: one non-read among them and it is
        # mocked, so a templated read (``/users/{id}``) can never carry a gated
        # route (``/users/wipe``) to the real API.
        match = matches[0] if matches else None
        if (
            match is not None
            and self._live_reads
            and all(risk is RiskClass.READ for _, risk in matches)
        ):
            return await self._real.handle_async_request(request)
        if match is not None:
            schema = self._responses.get(match[0])
            body = example_value(schema, match[0]) if schema else {"ok": True, "mock": match[0]}
            return httpx.Response(200, json=body, request=request)
        where = f"{request.method} {request.url.path}"
        self.unmatched.append(where)
        expected = ", ".join(repr(b) for b in self._expected_bases) or "''"
        return httpx.Response(
            NO_MOCK_STATUS,
            json={
                "error": {
                    "code": "EVAL_NO_MOCK",
                    "message": (
                        f"the evaluation transport has no route for {where}: the plan's "
                        f"routes are expected under base path {expected} — check "
                        "MCPCAST_BASE_URL and the spec's base path"
                    ),
                }
            },
            request=request,
        )

    async def aclose(self) -> None:
        """Close the real transport behind the live reads."""
        await self._real.aclose()


def mock_transport(
    plan: MCPcastPlan,
    *,
    operations: Sequence[Operation] | None = None,
    live_reads: bool = True,
    base_url: str | None = None,
) -> EvalTransport:
    """An httpx transport: live ``GET``/``HEAD`` (optional), spec-derived mocks otherwise.

    Whether a request goes live is decided by the **tool's risk class**, not
    the HTTP method: only routes of ``read`` tools reach the real API (when
    *live_reads* is on); every other route — including a ``GET`` the
    classifier escalated — is answered by a mock.  Mock responses are built
    from each operation's success response schema when *operations* are
    given, and echo the request otherwise.  Nothing that changes data ever
    reaches the real API during an evaluation.

    Each route is expected under exactly the base the generated server uses
    for it — *base_url* when given, else the operation's own server, else
    the plan's ``api.base_url`` (the resolution order of the generated
    ``upstream.py``).  A request that matches no route is **not** answered
    with a success: it gets :data:`NO_MOCK_STATUS` and is recorded on the
    transport's ``unmatched`` list, which :func:`score` reports.

    Args:
        plan: The plan the server was generated from.
        operations: Parsed operations, for the response schemas the mocks
            are built from.
        live_reads: Let the routes of ``read`` tools reach the real API.
        base_url: The ``MCPCAST_BASE_URL`` override in force, as the
            generated ``config.py`` reads it (:func:`base_url_override`);
            :func:`evaluate` passes the environment's value.
    """
    responses: dict[str, dict[str, Any] | None] = {
        op.operation_id: op.response_schema for op in (operations or [])
    }
    bases: list[str] = []
    entries: list[tuple[str, re.Pattern[str], str, RiskClass]] = []
    keys: list[tuple[int, int, int]] = []
    for tool in plan.tools:
        for route in tool.routes:
            base = base_url or route.base_url or plan.api.base_url
            base_path = urlparse(base).path
            if base_path not in bases:
                bases.append(base_path)
            entries.append(
                (
                    route.method,
                    _template_regex(base_path, route.path),
                    route.operation_id,
                    tool.risk,
                )
            )
            keys.append(_specificity(base_path.rstrip("/") + route.path))
    routes = [e for _, e in sorted(zip(keys, entries, strict=True), key=lambda ke: ke[0])]
    return EvalTransport(routes, responses, live_reads=live_reads, expected_bases=bases)


def _auto_approve(request: Any) -> bool:
    return True


def _looks_missing(call: ToolCall) -> bool:
    """``True`` when an upstream error reads like a 404 for a made-up identifier."""
    return call.details_status in (404, 410) if call.details_status else False


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def credential_slot(plan: MCPcastPlan) -> str:
    """Where the generated server presents the upstream credential, for a hint.

    ``"Authorization header"``, ``"X-API-Key header"`` or ``"api_key query
    parameter"`` — from the plan's ``credential_location``/``credential_name``;
    under ``passthrough`` always the ``Authorization`` header, which is the
    only thing that mode relays.
    """
    api = plan.api
    if api.auth is AuthMode.PASSTHROUGH or api.credential_location == "header":
        name = "Authorization" if api.auth is AuthMode.PASSTHROUGH else api.credential_name
        return f"{name} header"
    return f"{api.credential_name} query parameter"


def score(
    plan: MCPcastPlan, results: Sequence[TaskResult], *, unmatched: Sequence[str] = ()
) -> EvalReport:
    """Compute the report from task results (pure; no model, no network).

    Args:
        plan: The plan the server was generated from.
        results: One :class:`TaskResult` per task the agent attempted.
        unmatched: ``"METHOD /path"`` of the upstream requests the evaluation
            transport had no route for (:attr:`EvalTransport.unmatched`);
            each is a call that was neither live nor mocked and is reported
            as such rather than folded into the grade silently.
    """
    total = len(results)
    succeeded = sum(1 for r in results if r.success)
    selected = sum(1 for r in results if r.selected_correctly)
    calls = [c for r in results for c in r.calls]
    param_errors = [c for c in calls if c.error_code == "VALIDATION_ERROR"]
    crashed = [r for r in results if r.error]
    used = {c.tool for c in calls}
    targeted = {r.task.expected_tool for r in results}
    # A tool is "never used" only when a task that actually RAN targeted it;
    # a crashed task says nothing about the tool it was written for.
    targeted_by_a_run = {r.task.expected_tool for r in results if not r.error}
    never_used = [t.name for t in plan.tools if t.name in targeted_by_a_run and t.name not in used]
    not_covered = [t.name for t in plan.tools if t.name not in targeted]

    expected_counts: Counter[str] = Counter(r.task.expected_tool for r in results)
    confusion: Counter[tuple[str, str]] = Counter(
        (r.task.expected_tool, r.calls[0].tool)
        for r in results
        if r.calls and r.calls[0].tool != r.task.expected_tool
    )
    pairs = [ConfusedPair(expected=e, chosen=c, count=n) for (e, c), n in confusion.most_common()]

    success_rate = succeeded / total if total else 0.0
    selection_rate = selected / total if total else 0.0
    param_error_rate = len(param_errors) / len(calls) if calls else 0.0
    overall = 0.6 * success_rate + 0.4 * selection_rate

    fixes: list[str] = []
    if crashed:
        first = crashed[0]
        fixes.append(
            f"✗ {len(crashed)} of {total} task{'s' if total != 1 else ''} crashed: "
            f"{first.error[:200] if first.error else 'unknown error'}"
            + (" (first of several; see the task table)" if len(crashed) > 1 else "")
        )
    server_auth = [c for c in calls if c.error_code == "AUTHENTICATION_ERROR"]
    if server_auth:
        fixes.append(
            f"✗ {len(server_auth)} call{'s were' if len(server_auth) != 1 else ' was'} rejected "
            "by the server's own auth (AUTHENTICATION_ERROR) — the run measured the "
            "evaluation credentials, not the tool design: set MCPCAST_EVAL_HEADERS to a JSON "
            'object carrying a key from MCPCAST_CLIENT_KEYS (e.g. {"x-api-key": "sk-…"})'
        )
    upstream_auth = [
        c
        for c in calls
        if c.error_code in ("UPSTREAM_AUTH_MISSING", "UPSTREAM_AUTH_INVALID")
        or (c.error_code == "UPSTREAM_ERROR" and c.details_status in (401, 403))
    ]
    if upstream_auth:
        fixes.append(
            f"✗ {len(upstream_auth)} call{'s' if len(upstream_auth) != 1 else ''} reached the "
            "API without a valid upstream credential (UPSTREAM_AUTH_MISSING / HTTP 401/403) — "
            f"set MCPCAST_EVAL_AUTHORIZATION to a real {credential_slot(plan)} value, or "
            "configure MCPCAST_UPSTREAM_TOKEN / MCPCAST_UPSTREAM_TOKENS, or re-run with live "
            "reads off"
        )
    insecure = [c for c in calls if c.error_code == "UPSTREAM_INSECURE"]
    if insecure:
        fixes.append(
            f"✗ {len(insecure)} call{'s were' if len(insecure) != 1 else ' was'} refused before "
            "leaving the server (UPSTREAM_INSECURE): the upstream is plain http, so the "
            "credential is not sent — set MCPCAST_ALLOW_INSECURE_HTTP=1 only for a trusted "
            "network, or use https"
        )
    if unmatched:
        shown = ", ".join(dict.fromkeys(unmatched[:4]))
        fixes.append(
            f"✗ {len(unmatched)} call{'s' if len(unmatched) != 1 else ''} reached no mocked "
            f"route ({shown}{', …' if len(unmatched) > 4 else ''}) — neither live nor mocked, "
            "so the run measured the transport, not the tool design: check MCPCAST_BASE_URL "
            "and the spec's base path against the routes in the plan"
        )
    for pair in pairs:
        n = expected_counts[pair.expected]
        fixes.append(
            f"✗ `{pair.expected}` vs `{pair.chosen}` are ambiguous — the agent picked "
            f"`{pair.chosen}` in {pair.count}/{n} runs that needed `{pair.expected}` → merge "
            "them, or say in each description when NOT to use it"
        )
    stale = sorted(
        {
            c.tool
            for r in results
            for c in r.calls
            if c.error_code == "UPSTREAM_ERROR" and _looks_missing(c)
        }
    )
    if stale:
        listed = ", ".join(f"`{n}`" for n in stale[:8])
        fixes.append(
            f"✗ {listed} called the API with identifiers it does not recognise — the "
            "example in the plan teaches both the task writer and the agent, so replace "
            "those example values with ones that exist"
        )
    errors_by_tool: Counter[str] = Counter(c.tool for c in param_errors)
    for tool_name, n in errors_by_tool.most_common():
        hint = ""
        try:
            tool = plan.tool(tool_name)
        except KeyError:
            tool = None
        if tool is not None:
            weak = [
                p
                for p, spec in tool.visible_params.items()
                if not spec.description and (not tool.example or p not in tool.example)
            ]
            if weak:
                hint = (
                    f" — `{'`, `'.join(weak)}` "
                    + ("has" if len(weak) == 1 else "have")
                    + " no description and no example"
                )
        fixes.append(f"✗ `{tool_name}`: {n} parameter error{'s' if n != 1 else ''}{hint}")
    if never_used:
        listed = ", ".join(f"`{n}`" for n in never_used[:8])
        more = f" (+{len(never_used) - 8} more)" if len(never_used) > 8 else ""
        fixes.append(
            f"• {len(never_used)} tool{'s were' if len(never_used) != 1 else ' was'} never used "
            f"even when a task needed {'them' if len(never_used) != 1 else 'it'}: "
            f"{listed}{more} — consider dropping them or sharpening their descriptions"
        )
    if not_covered:
        listed = ", ".join(f"`{n}`" for n in not_covered[:8])
        more = f" (+{len(not_covered) - 8} more)" if len(not_covered) > 8 else ""
        fixes.append(
            f"• {len(not_covered)} tool{'s' if len(not_covered) != 1 else ''} not covered by any "
            f"task: {listed}{more} — raise --eval-tasks to score them"
        )
    for tool in plan.tools:
        if not tool.example and tool.visible_params:
            fixes.append(f"• `{tool.name}` has no example — agents lean on examples heavily")

    return EvalReport(
        grade=grade_for(overall),
        score=round(overall, 3),
        tasks_total=total,
        tasks_succeeded=succeeded,
        selection_rate=round(selection_rate, 3),
        param_error_rate=round(param_error_rate, 3),
        never_used=never_used,
        not_covered=not_covered,
        confused_pairs=pairs,
        fixes=fixes,
        results=list(results),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


EVAL_API_KEY = "mcpcast-eval"
EVAL_TENANT = "mcpcast-eval"
_PLACEHOLDER_BEARER = "Bearer mcpcast-eval"
_PLACEHOLDER_KEY = "mcpcast-eval"


def _placeholder_credential(plan: MCPcastPlan) -> str:
    """The stand-in upstream credential when ``MCPCAST_EVAL_AUTHORIZATION`` is not set.

    A bearer token where the API expects the ``Authorization`` header, a bare
    key where it expects an API key in another header or a query parameter —
    the shape the real API would reject with a 401, never one the generated
    server refuses before sending (it rejects ``<placeholder>`` values).
    """
    if plan.api.auth is AuthMode.PASSTHROUGH or plan.api.credential_is_authorization_header:
        return _PLACEHOLDER_BEARER
    return _PLACEHOLDER_KEY


def _default_headers(plan: MCPcastPlan) -> dict[str, str]:
    raw = os.environ.get("MCPCAST_EVAL_HEADERS")
    if raw:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise MCPcastError(f"MCPCAST_EVAL_HEADERS must be a JSON object: {exc}") from exc
        if not isinstance(data, dict):
            raise MCPcastError("MCPCAST_EVAL_HEADERS must be a JSON object")
        return {str(k): str(v) for k, v in data.items()}
    if plan.api.auth is AuthMode.PASSTHROUGH:
        # passthrough relays the caller's Authorization header and nothing else
        return {"authorization": os.environ.get("MCPCAST_EVAL_AUTHORIZATION", _PLACEHOLDER_BEARER)}
    if plan.api.auth is AuthMode.API_KEY:
        return {"x-api-key": EVAL_API_KEY}
    return {}


@contextlib.contextmanager
def _eval_credentials(plan: MCPcastPlan) -> Iterator[None]:
    """Make the generated server callable during an evaluation.

    ``api-key`` servers reject every call without a configured client key,
    and ``env-token`` servers without an upstream token.  For the duration
    of the run an evaluation-only key (``mcpcast-eval``, tenant
    ``mcpcast-eval``) is **merged into** ``MCPCAST_CLIENT_KEYS`` and
    ``MCPCAST_UPSTREAM_TOKENS`` — the operator's real keys stay valid, and
    an evaluation started inside a configured project (where ``.env`` sets
    them) still authenticates — and ``MCPCAST_UPSTREAM_TOKEN`` is set only
    when it is absent.  Live reads then carry a placeholder credential in
    the slot the API expects (:func:`credential_slot`) — set
    ``MCPCAST_EVAL_AUTHORIZATION`` to the real value for that slot (or the
    real ``MCPCAST_UPSTREAM_TOKEN`` / ``MCPCAST_UPSTREAM_TOKENS``) to
    authenticate them, or ``MCPCAST_EVAL_HEADERS`` to evaluate as one of
    the configured keys.

    Every variable touched is restored to its exact previous value (or
    removed again) when the run ends — also when the setup itself fails
    half-way (a malformed ``MCPCAST_UPSTREAM_TOKENS`` after
    ``MCPCAST_CLIENT_KEYS`` was already merged).

    Raises:
        MCPcastError: When ``MCPCAST_CLIENT_KEYS`` or ``MCPCAST_UPSTREAM_TOKENS``
            is set but is not a JSON object.
    """
    previous: dict[str, str | None] = {}

    def _set(name: str, value: str) -> None:
        previous.setdefault(name, os.environ.get(name))
        os.environ[name] = value

    def _merge(name: str, key: str, value: Any) -> None:
        raw = os.environ.get(name)
        try:
            current = json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise MCPcastError(f"{name} must be a JSON object: {exc}") from exc
        if not isinstance(current, dict):
            raise MCPcastError(f"{name} must be a JSON object")
        if key in current:
            return  # the operator configured the evaluation identity themselves
        _set(name, json.dumps({**current, key: value}))

    try:
        # Inside the try: a failing second merge must restore the first.
        upstream = os.environ.get("MCPCAST_EVAL_AUTHORIZATION", _placeholder_credential(plan))
        if plan.api.auth is AuthMode.API_KEY:
            _merge(
                "MCPCAST_CLIENT_KEYS",
                EVAL_API_KEY,
                {"client_id": "mcpcast-eval", "tenant_id": EVAL_TENANT, "roles": []},
            )
            _merge("MCPCAST_UPSTREAM_TOKENS", EVAL_TENANT, upstream)
        elif plan.api.auth is AuthMode.ENV_TOKEN and not os.environ.get("MCPCAST_UPSTREAM_TOKEN"):
            _set("MCPCAST_UPSTREAM_TOKEN", upstream)
        yield
    finally:
        for name, old_value in previous.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value


async def evaluate(
    plan: MCPcastPlan,
    build_server: Any,
    *,
    model: Any = "openai:gpt-5-mini",
    tasks: int | Sequence[EvalTask] = DEFAULT_EVAL_TASKS,
    operations: Sequence[Operation] | None = None,
    live_reads: bool = True,
    headers: dict[str, str] | None = None,
    complete: Completer | None = None,
    max_agent_iterations: int = 8,
) -> EvalReport:
    """Run the Agent Readiness evaluation.

    Args:
        plan: The plan the server was generated from.
        build_server: The generated module's ``build_server`` factory.
        model: Model for the driving agent (and task generation).
        tasks: Number of tasks to generate, or an explicit task list.
        operations: Parsed operations, used to derive mock responses from
            the spec's response schemas.
        live_reads: Let the routes of ``read`` tools reach the real API.
            Whether a request goes live is decided by the tool's **risk
            class**, not its HTTP method: a ``GET`` the classifier escalated
            to ``destructive`` is mocked like any write.  The transport
            expects every route where the generated server sends it —
            ``MCPCAST_BASE_URL`` when set (:func:`base_url_override`), else
            the operation's own server, else the plan's base URL — and a
            request it has no route for is answered with
            :data:`NO_MOCK_STATUS`, never a made-up success; the report
            names such calls.
        headers: MCP request headers for the in-process client (defaults
            from ``MCPCAST_EVAL_HEADERS``; else a placeholder bearer token for
            ``passthrough`` auth, or the evaluation key for ``api-key``).
            For ``api-key`` and ``env-token`` servers an evaluation-only
            identity is merged into the credential variables for the run
            (see :func:`_eval_credentials`) — real keys keep working.
        complete: Override the task-generation completion (tests).
        max_agent_iterations: Agent loop cap per task.

    Raises:
        MCPcastError: When there is nothing to evaluate (no tasks, tasks that
            name unknown tools), or when **no task could run at all** — every
            attempt crashed before the agent answered (a rejected model
            credential, a server that fails to build) — since a score for a
            run that never happened would be a measurement of nothing.
    """

    task_list = (
        await generate_tasks(plan, model=model, count=tasks, complete=complete)
        if isinstance(tasks, int)
        else list(tasks)
    )
    if not task_list:
        raise MCPcastError("no evaluation tasks")
    unknown = sorted({t.expected_tool for t in task_list} - set(plan.tool_names))
    if unknown:
        raise MCPcastError(f"tasks reference tools not in the plan: {unknown}")

    transport = mock_transport(
        plan, operations=operations, live_reads=live_reads, base_url=base_url_override()
    )
    with _eval_credentials(plan):
        async with httpx.AsyncClient(transport=transport, timeout=30) as http_client:
            report = await _run_tasks(
                plan,
                build_server,
                http_client,
                model,
                task_list,
                headers,
                max_agent_iterations,
                unmatched=transport.unmatched,
            )
    return report


async def _run_tasks(
    plan: MCPcastPlan,
    build_server: Any,
    http_client: httpx.AsyncClient,
    model: Any,
    task_list: list[EvalTask],
    headers: dict[str, str] | None,
    max_agent_iterations: int,
    *,
    unmatched: Sequence[str] = (),
) -> EvalReport:
    from promptise.agent import build_agent

    server = build_server(approval_handler=_auto_approve, http_client=http_client)
    recorder = CallRecorder()
    tools = await tools_from_server(
        server, recorder=recorder, headers=headers or _default_headers(plan)
    )
    agent = await build_agent(
        servers={},
        model=model,
        extra_tools=tools,
        instructions=EVAL_AGENT_INSTRUCTIONS,
        max_agent_iterations=max_agent_iterations,
    )
    results: list[TaskResult] = []
    for task in task_list:
        recorder.begin(task.id)
        answer, error = "", None
        try:
            out = await agent.ainvoke({"messages": [HumanMessage(content=task.prompt)]})
            answer = final_text(out)
        except Exception as exc:
            # The agent crashed (a provider rejected the model credential,
            # the graph raised…) — recorded per task, never hidden.
            error = f"{type(exc).__name__}: {exc}"
        calls = recorder.calls_for(task.id)
        results.append(
            TaskResult(
                task=task,
                calls=calls,
                success=error is None and any(c.tool == task.expected_tool and c.ok for c in calls),
                selected_correctly=bool(calls) and calls[0].tool == task.expected_tool,
                answer=answer,
                error=error,
            )
        )
    if all(r.error for r in results):
        raise MCPcastError(
            f"the evaluation could not run: all {len(results)} task"
            f"{'s' if len(results) != 1 else ''} crashed before the agent answered — "
            f"{results[0].error}"
        )
    return score(plan, results, unmatched=unmatched)


def write_eval(
    report: EvalReport, tasks: Sequence[EvalTask], out_dir: str | Path
) -> tuple[Path, Path]:
    """Write ``eval/tasks.yaml`` and ``eval/report.md`` under *out_dir*."""
    target = Path(out_dir) / "eval"
    target.mkdir(parents=True, exist_ok=True)
    tasks_path = target / "tasks.yaml"
    tasks_path.write_text(
        yaml.safe_dump(
            {"tasks": [t.model_dump(exclude_defaults=True) for t in tasks]},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    report_path = target / "report.md"
    report_path.write_text(report.render_markdown(), encoding="utf-8")
    return tasks_path, report_path
