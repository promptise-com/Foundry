"""``promptise mcpcast`` guided setup — the full-screen terminal UI.

Running ``promptise mcpcast`` with no spec opens this wizard: seven steps from
"where is your API's OpenAPI document" to a written, reviewed project, with
the explanation a first-time user needs next to each choice.  Every value the
wizard collects maps 1:1 onto a ``promptise mcpcast`` flag, and the
equivalent command is shown at the end so the second run can be scripted.

The pipeline is exactly the CLI's — :func:`~promptise.mcpcast.parse.load_spec`,
:func:`~promptise.mcpcast.plan.build_plan`, :func:`~promptise.mcpcast.curate.curate`,
:func:`~promptise.mcpcast.emit.write_project` and
:func:`~promptise.mcpcast.readiness.evaluate`; the wizard only collects their
arguments.

The plain functions at the top of this module (spec loading, the per-profile
preview, the auth recommendation, local API detection, the equivalent
command) carry no UI state and are tested directly; the Textual widgets
below them are driven headlessly with :meth:`textual.app.App.run_test`.

Example::

    from promptise.mcpcast.wizard import run_wizard

    result = run_wizard()            # opens the terminal UI
    if result:
        print(result.command)        # the non-interactive equivalent
"""

from __future__ import annotations

import json
import os
import re
import shlex
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, cast
from urllib.parse import urlsplit

import yaml
from rich.text import Text
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import (
    Button,
    Collapsible,
    ContentSwitcher,
    DataTable,
    Footer,
    Input,
    LoadingIndicator,
    Markdown,
    OptionList,
    Select,
    Static,
    Switch,
)
from textual.widgets.option_list import Option
from textual.worker import get_current_worker

from ._llm import Completer
from .classify import Classification, classify
from .emit import (
    _stale_packages,
    clipped_descriptions,
    describe_written,
    load_generated_server,
    package_name,
    plain_http_hosts,
    write_project,
)
from .parse import (
    Operation,
    api_name_from_spec,
    check_document,
    extract_operations,
    is_url,
    load_spec,
    scrub_strings,
    spec_base_url,
    spec_summary_line,
    spec_title,
)
from .plan import (
    build_plan,
    credential_slot,
    derive_tool_name,
    refuse_unrelayable_credential,
    resolve_base_url,
)
from .readiness import DEFAULT_EVAL_TASKS, EvalReport, write_eval
from .schema import (
    ApprovalMode,
    AuthMode,
    MCPcastError,
    MCPcastPlan,
    RiskClass,
    SafetyProfile,
    ToolPlan,
    is_plan_document,
)

__all__ = [
    "DEFAULT_MODEL",
    "DETECT_BUDGET",
    "DETECT_MAX_BYTES",
    "PROBE_PATHS",
    "PROBE_PORTS",
    "STEP_NAMES",
    "Candidate",
    "Detection",
    "MCPcastWizard",
    "ParsedSpec",
    "ProfilePreview",
    "Usage",
    "WizardResult",
    "WizardSettings",
    "detect_local_apis",
    "equivalent_command",
    "preview_profile",
    "probe_local_apis",
    "public_source",
    "quote_argument",
    "recommended_auth",
    "review_warnings",
    "run_wizard",
]

DEFAULT_MODEL = "openai:gpt-5-mini"
"""The curation model when none is chosen (the CLI's ``--model`` default)."""

STEP_NAMES: tuple[str, ...] = ("API spec", "Model", "Safety", "Auth", "Project", "Review", "Write")
"""The seven steps, in order (the welcome screen is step 0)."""

PROBE_PORTS: tuple[int, ...] = (8000, 8080, 8765, 3000, 5000, 5001, 4000, 8001, 8888, 9000)
"""Loopback ports :func:`detect_local_apis` looks at, most common first."""

PROBE_PATHS: tuple[str, ...] = (
    "/openapi.json",  # FastAPI, Django Ninja, Litestar
    "/swagger.json",  # Flask-RESTX, Swashbuckle (older)
    "/api-docs",  # springdoc (older), NestJS
    "/v3/api-docs",  # springdoc
    "/swagger/v1/swagger.json",  # ASP.NET Core
    "/api/openapi.json",
    "/docs/openapi.json",
    "/openapi.yaml",
)
"""Paths an API commonly serves its OpenAPI document at."""

DETECT_BUDGET = 10.0
"""Seconds :func:`probe_local_apis` spends in total before it stops probing."""

DETECT_MAX_BYTES = 5 * 1024 * 1024
"""Largest response body (after content decoding) a probe reads."""

_CURATION_BUDGET = 25  # the CLI's default budget under curation


# ---------------------------------------------------------------------------
# Plain data + logic (no UI)
# ---------------------------------------------------------------------------


class Usage(str, Enum):
    """How the generated server will be used — the question that decides the auth mode."""

    PERSONAL = "personal"
    """A desktop client (Claude Desktop, Claude Code, Cursor) launches it over stdio."""
    SHARED = "shared"
    """A shared HTTP deployment where every caller brings their own token."""
    TENANTS = "tenants"
    """A multi-tenant product: MCP clients present API keys, one per tenant."""
    OPEN = "open"
    """The API needs no credentials (open, or local development only)."""


_USAGE_AUTH: dict[Usage, AuthMode] = {
    Usage.PERSONAL: AuthMode.ENV_TOKEN,
    Usage.SHARED: AuthMode.PASSTHROUGH,
    Usage.TENANTS: AuthMode.API_KEY,
    Usage.OPEN: AuthMode.NONE,
}


def recommended_auth(usage: Usage) -> AuthMode:
    """The auth mode that fits *usage* (see :class:`Usage`)."""
    return _USAGE_AUTH[usage]


_AUTH_ENV_HINT: dict[AuthMode, str] = {
    AuthMode.ENV_TOKEN: (
        "You will set MCPCAST_UPSTREAM_TOKEN='Bearer <your API token>' in the client's env "
        "block (or the server's environment). Approval: the client asks its user."
    ),
    AuthMode.PASSTHROUGH: (
        "Callers connect over HTTP with their own Authorization header, which is forwarded. "
        "Nothing to set on the server. Approval: the client asks its user."
    ),
    AuthMode.API_KEY: (
        "You will set MCPCAST_CLIENT_KEYS (key -> tenant) and MCPCAST_UPSTREAM_TOKENS "
        "(tenant -> Authorization value). Approval: pending — a second person of the "
        "same tenant decides."
    ),
    AuthMode.NONE: (
        "No credentials are sent. The server refuses to bind to anything but loopback."
    ),
}


def public_source(source: str) -> str:
    """*source* with nothing secret in it: a URL loses its userinfo, query and fragment.

    Credentials in a spec URL (``https://user:token@host/…`` or
    ``?api_key=…``) are used for the fetch only.  What the wizard shows, what
    the plan records as ``spec_source`` and what the equivalent command
    repeats is this form.  Paths and inline documents are returned unchanged.
    Never raises, whatever the string looks like.
    """
    if not is_url(source):
        return source
    scheme, _, rest = source.strip().partition("://")
    end = len(rest)
    for separator in "/?#":  # the authority ends at the first of these (RFC 3986)
        index = rest.find(separator)
        if index != -1:
            end = min(end, index)
    authority = rest[:end].rpartition("@")[2]
    path = rest[end:] if rest[end : end + 1] == "/" else ""
    path = path.split("#", 1)[0].split("?", 1)[0]
    return f"{scheme}://{authority}{path}"


_URL_USERINFO = re.compile(r"(https?://)[^/\s'\"@]*@")
_URL_QUERY = re.compile(r"(https?://[^\s'\"?#]*)[?#][^\s'\"]*")


def _redact(message: str, source: str) -> str:
    """*message* with credentials removed: *source* itself, and any URL's userinfo or query.

    A library may print the URL in its own form (httpx keeps the username
    and masks the password), so every URL-shaped token in the message loses
    its ``user:secret@`` and its ``?query`` too.
    """
    message = message.replace(source, public_source(source))
    message = _URL_USERINFO.sub(r"\1", message)
    return _URL_QUERY.sub(r"\1", message)


_TERMINAL_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f\x80-\x9f]")
"""C0 controls except newline and tab, DEL, and the C1 range (0x9B is a CSI on its own)."""


def _console_safe(text: str) -> str:
    """*text* with every terminal control character deleted; newline and tab stay.

    Rich and Textual let ESC and the C1 controls through to the terminal, so
    a spec or model string carrying ``\\x1b[2K`` (erase the line) or an OSC 52
    sequence (write the clipboard) would act on the reviewer's terminal.
    The documents are scrubbed when they are loaded; this is the belt and
    braces applied wherever spec-, plan- or model-derived text is rendered.
    """
    return _TERMINAL_CONTROL.sub("", text)


def _declared_base_url(document: dict[str, Any], spec_url: str | None) -> str:
    """The base URL *document* declares on its own, ``""`` when it declares no usable one.

    A ``servers[0].url`` with a variable that has no default is such a case:
    the planner needs an explicit base URL, so whatever the user types is an
    override.
    """
    try:
        return spec_base_url(document, spec_url=spec_url)
    except MCPcastError:
        return ""


@dataclass
class ParsedSpec:
    """An OpenAPI document loaded and classified, ready to be planned."""

    source: str
    """Where the document came from, as the user typed it — minus credentials
    (see :func:`public_source`): a path, a URL or inline text."""
    label: str | None
    """What the plan records as ``spec_source`` (``"<inline>"`` for pasted text)."""
    document: dict[str, Any]
    operations: list[Operation]
    classifications: dict[str, Classification]
    title: str
    version: str
    description: str
    api_name: str
    base_url: str
    """The effective API base URL (override, else the spec's, else the fetch origin)."""
    declared_base_url: str = ""
    """The base URL the document itself declares (``servers[0]``, the Swagger 2
    host, or the fetch origin); empty when it declares no usable one.  A typed
    base URL that differs from it is an override (``--base-url``)."""

    @classmethod
    def load(
        cls,
        source: str,
        *,
        base_url: str | None = None,
        document: dict[str, Any] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> ParsedSpec:
        """Load, extract and classify *source*.

        Args:
            source: File path, URL or inline text.  A URL may carry
                credentials (userinfo, query string); they are used for the
                fetch and stripped from everything that is recorded.
            base_url: Override the API base URL declared by the document.
                Applied when the operations are extracted, exactly as the
                CLI's ``--base-url`` is.
            document: The already-loaded document (a detected local API), so
                *source* is only recorded, not fetched again.  It goes
                through the same checks and the same scrub as a fetched one
                (:func:`~promptise.mcpcast.parse.check_document`,
                :func:`~promptise.mcpcast.parse.scrub_strings`): the node
                and depth caps hold, and no control character survives.
            cancelled: Polled after every chunk while a URL downloads
                (:func:`~promptise.mcpcast.parse.load_spec`); ``True`` abandons
                the download.  The wizard passes its load's cancellation flag,
                so Ctrl+Q — or a newer load — stops a slow server's transfer
                within one chunk instead of waiting for it to finish.

        Raises:
            MCPcastError: when the document cannot be loaded or the download
                was cancelled, when it breaks a document cap, or when it is
                an mcpcast plan rather than an OpenAPI document.
        """
        source = source.strip()
        if not source:
            raise MCPcastError("Enter the path or URL of your API's OpenAPI document.")
        shown = public_source(source)
        if document is None:
            # The full URL: credentials are for the fetch only.
            document = load_spec(source, cancelled=cancelled)
        else:
            document = scrub_strings(check_document(document, hint=shown))
        return cls._from_document(shown, document, base_url)

    @classmethod
    def _from_document(
        cls, shown: str, document: dict[str, Any], base_url: str | None
    ) -> ParsedSpec:
        """Extract and classify a *document* that is already checked and scrubbed.

        *shown* is the public form of the source (see :func:`public_source`).
        """
        if is_plan_document(document):
            raise MCPcastError(
                f"{shown} is an mcpcast plan, not an OpenAPI document. To regenerate a "
                f"project from an edited plan run: promptise mcpcast {shown}"
            )
        inline = shown.lstrip().startswith(("{", "openapi:", "swagger:"))
        label = "<inline>" if inline else shown
        spec_url = shown if is_url(shown) else None
        operations = extract_operations(document, base_url=base_url or None, spec_url=spec_url)
        if not operations:
            raise MCPcastError(f"{shown} declares no operations under 'paths'.")
        info = document.get("info") or {}
        return cls(
            source=shown,
            label=label,
            document=document,
            operations=operations,
            classifications={op.operation_id: classify(op) for op in operations},
            title=spec_title(document),
            version=str(info.get("version") or "").strip(),
            description=spec_summary_line(document),
            api_name=api_name_from_spec(document, label),
            base_url=resolve_base_url(operations, base_url),
            declared_base_url=(
                _declared_base_url(document, spec_url) if base_url else resolve_base_url(operations)
            ),
        )

    def with_base_url(self, base_url: str | None) -> ParsedSpec:
        """This document re-extracted with *base_url* as the override (``None`` = as declared).

        The CLI applies ``--base-url`` when it extracts the operations; doing
        the same when the wizard's base URL field changes keeps the written
        project identical to what the equivalent command produces.  No
        fetch: the loaded document is reused as it is — it was checked and
        scrubbed when it was loaded.
        """
        return ParsedSpec._from_document(self.source, self.document, base_url)

    @property
    def risk_counts(self) -> dict[RiskClass, int]:
        """How many operations fall in each risk class (all four keys present)."""
        counts = dict.fromkeys(RiskClass, 0)
        for c in self.classifications.values():
            counts[c.risk] += 1
        return counts

    def summary(self) -> str:
        """``"Bookshelf API v1.0 — 9 operations: 5 read · 2 write · 2 destructive"``.

        Terminal-safe: the title and version are spec text (see :func:`_console_safe`).
        """
        head = " ".join(
            p
            for p in (self.title or self.api_name, f"v{self.version}" if self.version else "")
            if p
        )
        parts = [f"{n} {risk.value}" for risk, n in self.risk_counts.items() if n]
        return _console_safe(f"{head} — {len(self.operations)} operations: {' · '.join(parts)}")


@dataclass(frozen=True)
class ProfilePreview:
    """What a safety profile would generate from a parsed spec."""

    profile: SafetyProfile
    tools: int
    gated: int
    """Tools that require human approval."""
    excluded: int
    """Operations not exposed (by the profile, deprecated, or unmappable)."""

    def line(self) -> str:
        """A one-line description for a menu row."""
        if self.tools == 0:
            return "no tools would be generated"
        if self.gated == 0:
            return f"{self.tools} tool{'s' if self.tools != 1 else ''} · reads only"
        return f"{self.tools} tools · {self.gated} require human approval"


def preview_profile(parsed: ParsedSpec, profile: SafetyProfile) -> ProfilePreview:
    """Count what :func:`~promptise.mcpcast.plan.build_plan` would keep under *profile*.

    Runs the deterministic planner, so the numbers are exact for the offline
    path and an upper bound for curation (which may merge or drop tools).
    """
    # Counts only: env-token can present any credential slot the spec declares,
    # so the preview never fails on the auth mode chosen later (step 4 checks that).
    plan = build_plan(
        parsed.operations,
        profile=profile,
        base_url=parsed.base_url,
        auth=AuthMode.ENV_TOKEN,
        name=parsed.api_name,
        description=parsed.description,
        spec_source=parsed.label,
        classifications=parsed.classifications,
    )
    return ProfilePreview(
        profile=profile,
        tools=len(plan.tools),
        gated=len(plan.gated_tools),
        excluded=len(plan.dropped),
    )


@dataclass(frozen=True)
class Candidate:
    """A running local API that serves an OpenAPI document."""

    url: str
    title: str
    operations: int
    document: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)
    """The document as fetched, so picking a candidate needs no second request."""


Fetcher = Callable[[str], str | None]
"""``fetch(url) -> body text``, or ``None`` when the URL does not answer."""


def _http_fetch(
    url: str,
    *,
    connect_timeout: float,
    read_timeout: float,
    max_bytes: int,
    deadline: float,
    cancelled: Callable[[], bool] | None = None,
) -> str | None:
    """``GET`` *url* on loopback and return its body, or ``None`` for anything else.

    Redirects are not followed (a 3xx is "no document here"), the body is
    read in chunks and abandoned past *max_bytes* or *deadline*
    (:func:`time.monotonic`) or once *cancelled* returns ``True``, so a
    service that redirects elsewhere, trickles bytes or streams gigabytes
    cannot steer or stall detection, and a cancelled probe lets go of a
    trickling body within one chunk.
    """
    import httpx

    timeout = httpx.Timeout(
        connect=connect_timeout, read=read_timeout, write=read_timeout, pool=connect_timeout
    )
    chunks: list[bytes] = []
    size = 0
    try:
        with (
            # trust_env=False: no HTTP_PROXY/ALL_PROXY may carry a loopback probe elsewhere
            httpx.Client(timeout=timeout, follow_redirects=False, trust_env=False) as client,
            client.stream("GET", url) as resp,
        ):
            if resp.status_code != 200:
                return None
            declared = resp.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > max_bytes:
                return None
            for chunk in resp.iter_bytes():  # decoded: a gzip bomb is measured inflated
                size += len(chunk)
                if size > max_bytes or time.monotonic() > deadline:
                    return None
                if cancelled is not None and cancelled():
                    return None
                chunks.append(chunk)
    except httpx.HTTPError:
        return None
    try:
        return b"".join(chunks).decode("utf-8-sig")
    except UnicodeDecodeError:
        return None


def _parse_document(text: str, *, hint: str = "<probe>") -> dict[str, Any]:
    """Parse a fetched body strictly — JSON first, then YAML — into a mapping.

    Unlike :func:`~promptise.mcpcast.parse.load_spec`, the body is never
    interpreted: a body reading ``http://…`` or ``/etc/passwd`` is a parse
    failure, not a URL to fetch or a file to read.  What comes out is held
    to the same caps and cleaned the same way as a document ``load_spec``
    returns — :func:`~promptise.mcpcast.parse.check_document` (the node
    budget ``MCPCAST_MAX_SPEC_NODES``, the nesting depth) and
    :func:`~promptise.mcpcast.parse.scrub_strings` (terminal control
    characters, lone surrogates) — so a local API that ``load_spec`` would
    refuse is simply not a candidate, and a title carrying an escape
    sequence cannot reach the terminal through the candidate list.

    Args:
        text: The response body.
        hint: What to call the document in an error (the probed URL).

    Raises:
        MCPcastError: when the text is neither JSON nor YAML, is not a
            mapping, or breaks a document cap.
    """
    stripped = text.strip()
    if not stripped:
        raise MCPcastError("empty body")
    data: Any
    try:
        data = json.loads(stripped)
    except (ValueError, RecursionError):
        try:
            # The pure-Python safe loader on purpose: libyaml's composer
            # recurses in C and a body nested 100,000 levels deep crashes the
            # interpreter with it, where this one raises RecursionError.
            data = yaml.safe_load(stripped)
        except (yaml.YAMLError, RecursionError) as exc:
            raise MCPcastError(f"not JSON or YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise MCPcastError("document is not a mapping")
    return scrub_strings(check_document(data, hint=hint))


_PROBE_PATH = re.compile(r"/[A-Za-z0-9._~/-]*")


def _probe_url(port: int, path: str) -> str:
    """``http://127.0.0.1:<port><path>`` — the only shape a probe may take.

    Raises:
        ValueError: for a port outside 1–65535 or a path that is not a plain
            absolute path (no query, fragment, userinfo or whitespace).
    """
    port = int(port)
    if not 1 <= port <= 65535:
        raise ValueError(f"probe port out of range: {port}")
    if not _PROBE_PATH.fullmatch(path):
        raise ValueError(f"probe path must be a plain absolute path: {path!r}")
    return f"http://127.0.0.1:{port}{path}"


@dataclass(frozen=True)
class Detection:
    """What :func:`probe_local_apis` found, and what it did not get to."""

    candidates: list[Candidate]
    """Running local APIs serving an OpenAPI document, in port order."""
    skipped_ports: tuple[int, ...] = ()
    """Ports not (fully) probed because the time budget ran out or the probe was cancelled."""
    elapsed: float = 0.0
    """Seconds the probe took."""

    def skipped_line(self) -> str:
        """``"Stopped after 10 s: ports 8888, 9000 were not checked."`` or ``""``."""
        if not self.skipped_ports:
            return ""
        return (
            f"Stopped after {self.elapsed:.0f} s: ports "
            f"{', '.join(map(str, self.skipped_ports))} were not checked."
        )


def probe_local_apis(
    *,
    ports: Sequence[int] = PROBE_PORTS,
    paths: Sequence[str] = PROBE_PATHS,
    timeout: float = 0.4,
    read_timeout: float = 2.0,
    budget: float = DETECT_BUDGET,
    max_bytes: int = DETECT_MAX_BYTES,
    fetch: Fetcher | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> Detection:
    """Look for OpenAPI documents served by APIs running on this machine.

    Probes ``http://127.0.0.1:<port><path>`` for every port and path (first
    hit per port wins) and keeps the ones that parse as an OpenAPI document
    with at least one operation.  Loopback only, ``GET`` only, redirects not
    followed, bodies parsed strictly (never fetched or read as a path),
    capped at *max_bytes* and held to the node budget and nesting depth
    :func:`~promptise.mcpcast.parse.load_spec` enforces, every string
    scrubbed of terminal control characters (see :func:`_parse_document`);
    whatever a probed service answers, a malformed or hostile body is
    skipped, never raised.  Probing stops once *budget* seconds have passed
    — the ports not reached are reported, not silently dropped.

    Args:
        ports: Ports to probe (see :data:`PROBE_PORTS`).
        paths: Paths to try on each port (see :data:`PROBE_PATHS`).
        timeout: Connect timeout per request, in seconds.
        read_timeout: Timeout per socket read, in seconds.
        budget: Total seconds for the whole probe (see :data:`DETECT_BUDGET`).
        max_bytes: Largest body read (see :data:`DETECT_MAX_BYTES`).
        fetch: Override the HTTP fetch (tests inject a stub).
        cancelled: Polled between probes and, with the built-in fetch, after
            every chunk of a body; ``True`` stops the probe early.

    Returns:
        The :class:`Detection`.
    """
    started = time.monotonic()
    deadline = started + budget
    get = fetch or (
        lambda url: _http_fetch(
            url,
            connect_timeout=timeout,
            read_timeout=read_timeout,
            max_bytes=max_bytes,
            deadline=deadline,
            cancelled=cancelled,
        )
    )
    found: list[Candidate] = []
    skipped: list[int] = []
    for port in ports:
        for path in paths:
            if time.monotonic() > deadline or (cancelled is not None and cancelled()):
                skipped.append(port)
                break
            url = _probe_url(port, path)
            try:
                body = get(url)
                if not body or len(body) > max_bytes:
                    continue
                document = _parse_document(body, hint=url)
                if is_plan_document(document):
                    continue
                ops = extract_operations(document, spec_url=url)
                candidate = Candidate(
                    url=url,
                    title=spec_title(document) or url,
                    operations=len(ops),
                    document=document,
                )
            except Exception:  # whatever a probed service answers must never take the wizard down
                continue
            if ops:
                found.append(candidate)
                break
    return Detection(
        candidates=found,
        skipped_ports=tuple(dict.fromkeys(skipped)),
        elapsed=time.monotonic() - started,
    )


def detect_local_apis(
    *,
    ports: Sequence[int] = PROBE_PORTS,
    paths: Sequence[str] = PROBE_PATHS,
    timeout: float = 0.4,
    fetch: Fetcher | None = None,
) -> list[Candidate]:
    """The candidates :func:`probe_local_apis` finds, with its default budget and caps.

    Args:
        ports: Ports to probe (see :data:`PROBE_PORTS`).
        paths: Paths to try on each port (see :data:`PROBE_PATHS`).
        timeout: Connect timeout per request, in seconds.
        fetch: Override the HTTP fetch (tests inject a stub).

    Returns:
        Candidates in port order.
    """
    return probe_local_apis(ports=ports, paths=paths, timeout=timeout, fetch=fetch).candidates


@dataclass
class WizardSettings:
    """Everything the wizard collects — one field per ``promptise mcpcast`` flag."""

    spec: str = ""
    base_url: str = ""
    """The effective API base URL; ``base_url_override`` says whether the user changed it."""
    base_url_override: bool = False
    curate: bool = True
    model: str = DEFAULT_MODEL
    profile: SafetyProfile = SafetyProfile.READ_ONLY
    usage: Usage = Usage.PERSONAL
    auth: AuthMode = AuthMode.ENV_TOKEN
    approval: ApprovalMode | None = None
    name: str = ""
    derived_name: str = ""
    out_dir: str = ""
    max_tools: int | None = None
    """``None`` means the CLI default: 25 under curation, unlimited offline."""
    evaluate: bool = False
    eval_tasks: int = DEFAULT_EVAL_TASKS
    force: bool = False

    @property
    def effective_budget(self) -> int | None:
        """The budget passed to the planner."""
        if self.max_tools is not None:
            return self.max_tools
        return _CURATION_BUDGET if self.curate else None

    @property
    def default_out_dir(self) -> str:
        """The output folder when none was typed: ``<name>-mcp``, the CLI's ``--out`` default."""
        return f"{self.name or self.derived_name}-mcp"


def _on_windows() -> bool:
    return os.name == "nt"


def quote_argument(part: str, *, windows: bool | None = None) -> str:
    """Quote one command-line argument for the shell the user will paste it into.

    POSIX shells get :func:`shlex.quote`. On Windows (``os.name == "nt"`` unless
    *windows* says otherwise) single quotes mean nothing to ``cmd.exe``, so an
    argument that needs quoting is wrapped in double quotes — which both
    ``cmd.exe`` and PowerShell accept — with embedded quotes backslash-escaped.
    """
    if windows is None:
        windows = _on_windows()
    if not windows:
        return shlex.quote(part)
    if part and not any(c.isspace() or c in '"&|<>^' for c in part):
        return part
    return '"' + part.replace('"', '\\"') + '"'


def equivalent_command(s: WizardSettings) -> str:
    """The non-interactive ``promptise mcpcast`` command that reproduces *s*.

    Only flags that differ from the CLI defaults are included, so the line
    stays short; it is shown on the last step and printed after the wizard
    exits. Arguments are quoted for the current platform's shell (see
    :func:`quote_argument`).
    """
    parts = ["promptise", "mcpcast", s.spec]
    if s.base_url_override and s.base_url:
        parts += ["--base-url", s.base_url]
    if s.profile is not SafetyProfile.READ_ONLY:
        parts += ["--profile", s.profile.value]
    if s.auth is not AuthMode.PASSTHROUGH:
        parts += ["--auth", s.auth.value]
    if s.approval is not None:
        parts += ["--approval", s.approval.value]
    if not s.curate:
        parts.append("--no-curate")
    elif s.model != DEFAULT_MODEL:
        parts += ["--model", s.model]
    if s.max_tools is not None and (not s.curate or s.max_tools != _CURATION_BUDGET):
        parts += ["--max-tools", str(s.max_tools)]
    if s.name and s.name != s.derived_name:
        parts += ["--name", s.name]
    if s.out_dir and s.out_dir != s.default_out_dir:
        parts += ["--out", s.out_dir]
    if s.force:
        parts.append("--force")
    if s.evaluate:
        parts.append("--eval")
        if s.eval_tasks != DEFAULT_EVAL_TASKS:
            parts += ["--eval-tasks", str(s.eval_tasks)]
    return " ".join(quote_argument(p) for p in parts)


@dataclass
class WizardResult:
    """What the wizard wrote — the last write, which is what is on disk."""

    plan: MCPcastPlan
    out_dir: Path
    written: list[Path]
    command: str
    """The equivalent non-interactive command (see :func:`equivalent_command`)."""
    report: EvalReport | None = None
    """The Agent Readiness report, when the evaluation was run."""
    eval_requested: bool = False
    """The evaluation was switched on; with ``report`` still ``None`` it did not
    complete (the wizard was quit while it ran, or it failed)."""


def review_warnings(plan: MCPcastPlan) -> list[str]:
    """Things a reviewer should look at first — computed from the plan, not guessed.

    - A description that names a tool the plan does not expose. The classic
      misread: the model writes *"to remove it use delete_book"* for an
      operation the safety profile excluded, and the agent goes looking for a
      tool that does not exist.
    - Parameters hidden from the agent (sent as fixed defaults), which your
      users may need to set.

    Returns:
        One line per finding, ``"<tool>: <what to check>"``; empty when there
        is nothing to flag.
    """
    exposed = {t.name for t in plan.tools}
    ghosts = {derive_tool_name(d.operation_id) for d in plan.dropped} - exposed
    out: list[str] = []
    for tool in plan.tools:
        named = sorted(g for g in ghosts if re.search(rf"\b{re.escape(g)}\b", tool.description))
        if named:
            out.append(
                f"{tool.name}: the description names {', '.join(named)} — not exposed by this "
                "plan, so the agent will look for a tool that does not exist"
            )
        hidden = [n for n, p in tool.params.items() if p.hidden]
        if hidden:
            out.append(
                f"{tool.name}: hides {', '.join(hidden)} (sent as defaults) — fine unless your "
                "users need to set them"
            )
    return out


def _ready_providers() -> list[str]:
    """Display names of providers whose required variables are all set."""
    from promptise.models import PROVIDERS

    ready = []
    for p in PROVIDERS:
        if p.env and not p.missing(set()):
            ready.append(p.display)
    return ready


def _model_status(spec: str) -> tuple[bool, str]:
    """``(ok, message)`` for the model string, without calling anything."""
    from promptise.models import check_model

    try:
        check = check_model(spec)
    except ValueError as exc:  # unparsable string
        return False, f"✗ {exc}"
    if check.provider is None:
        return True, f"✓ {check.canonical} — passed to LangChain as-is"
    if check.ok:
        names = ", ".join(v.name for v in check.provider.env if v.required and v.word == "api_key")
        found = f" ({names} is set)" if names else ""
        return True, f"✓ {check.canonical} — {check.provider.title}{found}"
    return False, "✗ " + "\n  ".join(check.problems)


def _risk_style(risk: RiskClass) -> str:
    return {"read": "green", "write": "yellow"}.get(risk.value, "red")


_DETAIL_MAX_LINES = 50
"""Lines of a tool description the review detail renders before it is cut."""
_DETAIL_MAX_CHARS = 4000
"""Characters of a tool description the review detail renders before it is cut."""
_DETAIL_MAX_PARAMS = 100
"""Parameter rows the review detail lists before ``… and N more``."""
_DETAIL_MAX_PARAM_CHARS = 300
"""Characters of one parameter description (or example) shown in the review detail."""
_DROPPED_MAX_ROWS = 200
"""Rows of the "Not exposed" list before ``… and N more``."""
_FULL_TEXT_NOTE = "full text in mcpcast.plan.yaml"


def _clamp_lines(text: str, *, max_lines: int, max_chars: int) -> tuple[str, int, bool]:
    """The head of *text* that fits *max_lines* whole lines and *max_chars* characters.

    Returns:
        ``(kept, dropped_lines, cut)`` — the lines kept, how many whole lines
        were left out, and whether a line was cut in the middle (a single
        line longer than *max_chars* keeps its first *max_chars* characters
        so that something is shown).
    """
    lines = text.splitlines()
    kept: list[str] = []
    size = 0
    cut = False
    for line in lines:
        if len(kept) >= max_lines:
            break
        if size + len(line) > max_chars:
            if not kept:
                kept.append(line[:max_chars])
                cut = True
            break
        kept.append(line)
        size += len(line) + 1
    return "\n".join(kept), len(lines) - len(kept), cut


def _one_line(text: str, max_chars: int = _DETAIL_MAX_PARAM_CHARS) -> str:
    """*text* on one line, at most *max_chars* characters, terminal-safe."""
    flat = " ".join(_console_safe(text).split())
    return flat if len(flat) <= max_chars else flat[: max_chars - 1].rstrip() + "…"


def _tool_markdown(tool: ToolPlan) -> str:
    """The detail panel for one tool on the review step.

    Clamped by lines and characters (:data:`_DETAIL_MAX_LINES`,
    :data:`_DETAIL_MAX_CHARS`, :data:`_DETAIL_MAX_PARAMS`): the panel is
    re-rendered on every cursor move, and a description that is a
    5,000-item Markdown list would mount tens of thousands of widgets.  The
    plan file always has the full text, and the panel says so.  Every string
    is spec or model text, so it is terminal-safe first (:func:`_console_safe`).
    """
    head = f"**{tool.name}** · {tool.risk.value}"
    if tool.requires_approval:
        head += " · approval required"
    routes = ", ".join(f"`{r.method} {_one_line(r.path)}`" for r in tool.routes)
    original = derive_tool_name(tool.routes[0].operation_id)
    if tool.name != original:
        routes += f" — renamed from `{original}`"
    description, dropped, cut = _clamp_lines(
        _console_safe(tool.description.strip()),
        max_lines=_DETAIL_MAX_LINES,
        max_chars=_DETAIL_MAX_CHARS,
    )
    lines = [head, "", routes, "", description or "*(no description)*"]
    if dropped:
        lines.append(f"\n*… {dropped} more line{'s' if dropped != 1 else ''} — {_FULL_TEXT_NOTE}*")
    elif cut:
        lines.append(f"\n*… truncated — {_FULL_TEXT_NOTE}*")
    visible = tool.visible_params
    if visible:
        lines += ["", "**Parameters**"]
        for name, p in list(visible.items())[:_DETAIL_MAX_PARAMS]:
            kind = str(p.json_schema.get("type") or "object")
            req = ", required" if p.required else ""
            desc = f" — {_one_line(p.description)}" if p.description else ""
            lines.append(f"- `{_one_line(name)}` ({kind}{req}){desc}")
        if len(visible) > _DETAIL_MAX_PARAMS:
            more = len(visible) - _DETAIL_MAX_PARAMS
            lines.append(
                f"- *… and {more} more parameter{'s' if more != 1 else ''} — {_FULL_TEXT_NOTE}*"
            )
    hidden = [n for n, p in tool.params.items() if p.hidden]
    if hidden:
        shown = ", ".join(f"`{_one_line(h)}`" for h in hidden[:_DETAIL_MAX_PARAMS])
        if len(hidden) > _DETAIL_MAX_PARAMS:
            shown += f" … and {len(hidden) - _DETAIL_MAX_PARAMS} more"
        lines += ["", f"**Hidden from the agent**: {shown}"]
    if tool.example:
        example = _one_line(json.dumps(tool.example), _DETAIL_MAX_CHARS)
        lines += ["", f"**Example**: `{example}`"]
    return "\n".join(lines)


def _dropped_markdown(plan: MCPcastPlan) -> str:
    """The "Not exposed" list: one row per dropped operation, capped at :data:`_DROPPED_MAX_ROWS`."""
    if not plan.dropped:
        return "*Every operation is exposed.*"
    rows = [
        f"- `{_one_line(d.operation_id)}` — {_one_line(d.reason)}"
        for d in plan.dropped[:_DROPPED_MAX_ROWS]
    ]
    if len(plan.dropped) > _DROPPED_MAX_ROWS:
        more = len(plan.dropped) - _DROPPED_MAX_ROWS
        rows.append(f"- *… and {more} more — {_FULL_TEXT_NOTE}*")
    return "\n".join(rows)


_TOKEN_PLACEHOLDER = "Bearer <your API token>"


def _env_assignment(name: str, value: str, *, windows: bool) -> str:
    """``NAME=value`` as a shell prefix (POSIX) or a ``set`` line (Windows), quoted."""
    if windows:
        # cmd.exe's own idiom: the quotes wrap the whole assignment and are not
        # part of the value (set X="a b" would keep them; < and > redirect).
        return f'set "{name}={value}"'
    return f"{name}={quote_argument(value, windows=False)}"


def _next_steps_markdown(plan: MCPcastPlan, out_dir: Path, *, windows: bool | None = None) -> str:
    """The "Next steps" block of the write step, every command quoted for the shell.

    Paths and arguments go through :func:`quote_argument` (built from parts,
    never interpolated), so a folder with a space or a ``&`` in its name
    pastes correctly on POSIX shells and on Windows.  The evaluation line
    carries ``MCPCAST_EVAL_AUTHORIZATION`` for every auth mode that presents
    a credential — without it the re-run measures the missing token, not the
    tool design.

    Args:
        plan: The plan that was written.
        out_dir: The project folder, as the user typed it.
        windows: Quote for ``cmd.exe``/PowerShell instead of a POSIX shell
            (default: the current platform).
    """
    if windows is None:
        windows = _on_windows()

    def q(part: str) -> str:
        return quote_argument(part, windows=windows)

    def cmd(parts: Sequence[str]) -> str:
        return " ".join(q(p) for p in parts)

    plan_file = str(out_dir / "mcpcast.plan.yaml")
    server = str(out_dir / "server.py")
    auth = plan.api.auth
    if auth in (AuthMode.PASSTHROUGH, AuthMode.API_KEY):
        header = "Authorization" if auth is AuthMode.PASSTHROUGH else "x-api-key"
        try_it = (
            f"4. Try it: `{cmd(['python', server, '--transport', 'http'])}`, then point your "
            f"MCP client at `http://127.0.0.1:8080/mcp` with the `{header}` header — a stdio "
            "client cannot send it (the README has the details)."
        )
    else:
        add = ["claude", "mcp", "add", plan.api.name]
        if auth is AuthMode.ENV_TOKEN:
            add += ["-e", f"MCPCAST_UPSTREAM_TOKEN={_TOKEN_PLACEHOLDER}"]
        add += ["--", "python", server]
        try_it = f"4. Try it in Claude Code: `{cmd(add)}`"
    measure = cmd(["promptise", "mcpcast", plan_file, "--eval"])
    if auth is AuthMode.NONE:
        measure_line = f"5. Measure: `{measure}`"
    elif windows:
        assignment = _env_assignment("MCPCAST_EVAL_AUTHORIZATION", _TOKEN_PLACEHOLDER, windows=True)
        measure_line = (
            f"5. Measure: `{assignment}` (PowerShell: "
            f"`$env:MCPCAST_EVAL_AUTHORIZATION = {q(_TOKEN_PLACEHOLDER)}`), then `{measure}` — "
            "the evaluation's live reads carry that credential."
        )
    else:
        assignment = _env_assignment(
            "MCPCAST_EVAL_AUTHORIZATION", _TOKEN_PLACEHOLDER, windows=False
        )
        measure_line = (
            f"5. Measure: `{assignment} {measure}` — the evaluation's live reads carry that "
            "credential."
        )
    return "\n".join(
        [
            "**Next steps**",
            "",
            f"1. Read `{plan_file}` — fix names, descriptions, hidden parameters.",
            f"2. Regenerate after edits: `{cmd(['promptise', 'mcpcast', plan_file])}`",
            f"3. Run its tests: `{cmd(['cd', str(out_dir)])} && pytest`",
            try_it,
            measure_line,
            "",
            f"`{out_dir / 'README.md'}` has the Claude Desktop and Cursor snippets, "
            "`pip install -e .`, and the Docker deployment.",
        ]
    )


# ---------------------------------------------------------------------------
# Explanations — the text next to each choice
# ---------------------------------------------------------------------------

_EXPLAIN_WELCOME = """\
**What will happen**

1. **API spec** — point at your OpenAPI document (a file, a URL, or a running local API).
2. **Model** — pick the model that designs the tool surface, or go offline.
3. **Safety** — what an agent may do: reads only, or also writes with human approval.
4. **Auth** — how the generated server authenticates against *your* API.
5. **Project** — name, output folder, tool budget.
6. **Review** — read what the model decided. It will misread something; here is where you catch it.
7. **Write** — a real project: an installable package, `server.py`, tests, `pyproject.toml`, \
`Dockerfile`, `README.md` — and the exact command to do it again.

Nothing is written before step 7. The plan file is the artifact you own; the server is generated from it.
"""

_EXPLAIN_SPEC = """\
MCPcast reads your API's **OpenAPI document** (3.x or Swagger 2). Every path, operation, \
parameter and schema in it is the raw material for tools.

- **FastAPI, Django Ninja, Litestar, NestJS, Spring, ASP.NET** serve it while running — \
usually at `/openapi.json`, `/swagger.json` or `/v3/api-docs`. *Detect* looks there on the \
usual local ports.
- A **file** works the same: `openapi.yaml`, `swagger.json`, or a URL to either.
- Quality in, quality out: `summary`, `description`, `operationId` and parameter descriptions \
are what become tool names and descriptions.

The **API base URL** is where the generated server sends requests. It comes from the spec's \
`servers` block, or from the URL the spec was fetched from. Override it when the spec is \
relative or points at the wrong environment.
"""

_EXPLAIN_MODEL = """\
**Curation** is where a model reads every operation and designs the *tool surface*: which \
operations deserve a tool, what to call them, how to describe them so an assistant picks \
the right one, which parameters to hide. Post-conditions are enforced in code — risk is \
never downgraded, no operation is invented — but the wording is the model's, which is why \
step 6 exists.

**Offline** derives one tool per operation from the spec, deterministically: names from \
`operationId`, descriptions from `summary`. No model, no network. Good for a first look, \
CI, or an air-gapped machine.

The model's provider key is read from `.env` in the directory you run from, or from the \
environment — never typed here. Any provider works: `openai:gpt-5-mini`, \
`anthropic:claude-sonnet-4.5`, `azure:<deployment>`, `ollama:llama3` … \
`promptise models list` shows them all, `promptise models env <provider>` prints the \
variables to set.
"""

_EXPLAIN_SAFETY = """\
Every operation is classified by what it can do to the world: **read**, **write**, \
**destructive** (delete, reset, revoke …) or **financial** (pay, refund, transfer …). \
The profile decides which classes become tools:

| profile | read | write | destructive / financial |
|---|---|---|---|
| read-only | tools | not generated | not generated |
| standard | tools | tools, **approval required** | not generated |
| full | tools | approval required | tools, **approval required** |

*Approval required* means the generated server parks the call until a human says yes — \
enforced server-side, for any MCP client, deny-by-default on timeout. Start read-only; \
widen after you have read the plan.
"""

_EXPLAIN_AUTH = """\
Two credentials are involved, and they are unrelated: the model key from step 2, and \
**your API's own token**, which the generated server presents on every upstream call. \
How that token arrives is the auth mode:

- **env-token** — one token from `MCPCAST_UPSTREAM_TOKEN` in the server's environment. \
For a personal server launched over stdio by Claude Desktop, Claude Code or Cursor (a \
subprocess cannot receive headers).
- **passthrough** — each caller's own `Authorization` header is forwarded. For a shared \
HTTP deployment where every user acts as themselves.
- **api-key** — MCP clients present a key (`MCPCAST_CLIENT_KEYS`); each key maps to a \
tenant whose upstream token lives in `MCPCAST_UPSTREAM_TOKENS`. Approval becomes \
*pending*: a second person of the same tenant reviews.
- **none** — no credentials; the server refuses to bind to anything but loopback.

The credential is never written into the generated files — they only read it.
"""

_EXPLAIN_PROJECT = """\
- **Name** is the MCP server name clients display, and the default folder `<name>-mcp`.
- **Output folder** receives a real project: `mcpcast.plan.yaml` (the editable source of \
truth), a `<name>_mcp/` package (config, HTTP client, approval gate, one tools module per \
resource), `server.py` (runs it without installing), `tests/`, `pyproject.toml`, `Dockerfile`, \
`.env.example` and `README.md`. A folder that already has files in it is never written to \
unless you switch overwrite on — regenerate an existing project from its plan instead, to keep \
your edits.
- **Tool budget** caps the surface. Assistants choose better among 10 well-described tools \
than 60; curation ranks by usefulness, offline keeps reads first.
- **Agent Readiness** runs a real agent against the generated server after writing and \
grades how reliably it picks the right tool with the right arguments (A–F). Uses the \
model; takes a minute or two.
"""

_EXPLAIN_REVIEW = """\
**Never trust it blindly.** The model designed this surface and it *will* misread \
something. Read every row:

- **Name** — a verb your users would say (`search_books`, not `post_books_search`)?
- **Description** — says *when* to use it and what it returns? Mentions no tool that is \
not in this list?
- **Risk** — a `read` that changes state must be `write`; escalation is a one-line plan edit.
- **Params** — is anything your users need hidden? Are examples real identifiers?
- **Not exposed** — does every reason make sense to you?

Anything wrong: write the project anyway, edit `mcpcast.plan.yaml`, and regenerate with \
`promptise mcpcast <folder>/mcpcast.plan.yaml`. The plan is yours; the server is derived.
"""

_CHECKLIST = Text.assemble(
    ("• Name", "bold"),
    (" — a verb your users would say?\n", ""),
    ("• Description", "bold"),
    (" — says when to use it? Names only tools in this list?\n", ""),
    ("• Risk", "bold"),
    (" — a read that changes state must be write.\n", ""),
    ("• Params", "bold"),
    (" — nothing your users need hidden? Examples real?\n", ""),
    ("• Not exposed", "bold"),
    (" — every reason makes sense?\n", ""),
    ("Wrong? Write anyway, edit mcpcast.plan.yaml, regenerate.", "#9ca3af"),
)

_EXPLAIN_WRITE = """\
- `mcpcast.plan.yaml` — every tool, parameter, risk class and exclusion reason. Edit it; \
regenerate with `promptise mcpcast <folder>/mcpcast.plan.yaml`.
- `<name>_mcp/` — the server as a package: `config.py`, `upstream.py` (the HTTP client), \
`approval.py` (the gate), `server.py` (`build_server()`), `tools/` (one module per resource). \
Rewritten on regeneration.
- `server.py` — runs it without installing: `python server.py` speaks stdio for desktop \
clients, `--transport http` listens for shared use. `pip install -e .` gives you the \
`<name>-mcp` command instead.
- `tests/` — a pytest suite through the full pipeline: every tool listed, routed to the right \
upstream operation, and gated when it changes data. `pytest` runs it.
- `pyproject.toml`, `Dockerfile`, `.env.example`, `.gitignore`, `README.md` — packaging, \
image, configuration template, install snippets. The first four are yours after the first \
write; the README is regenerated.
"""

_HELP = """\
## Keys

| key | action |
|---|---|
| `Enter` | continue — confirm the field or menu row you are on |
| `Esc` | back one step |
| `Tab` / `Shift+Tab` | move between fields |
| `↑` `↓` | move in a menu or table |
| `F1` | this help |
| `Ctrl+Q` | quit — nothing is written before step 7; after it, the written project is reported |

The mouse works too.

## Steps

1. **API spec** — file, URL or a detected local API. The base URL is where requests go.
2. **Model** — curation model (needs its provider key in `.env` or the environment), or offline.
3. **Safety** — read-only / standard / full. Writes always require human approval.
4. **Auth** — how the generated server authenticates against your API.
5. **Project** — name, folder, tool budget, optional Agent Readiness evaluation.
6. **Review** — read the plan. Fix misreadings in `mcpcast.plan.yaml` after writing.
7. **Write** — files are written; the equivalent command is shown.
"""


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------

_BRAND = Theme(
    name="promptise",
    primary="#818cf8",
    secondary="#6366f1",
    accent="#a5b4fc",
    success="#34d399",
    warning="#fbbf24",
    error="#f87171",
    background="#0f1117",
    surface="#161a26",
    panel="#1e2333",
    dark=True,
)


class BrandHeader(Static):
    """The Promptise mark and the wizard's one-line purpose."""

    DEFAULT_CSS = """
    BrandHeader {
        height: 4;
        padding: 0 1;
        background: $panel;
        border-bottom: tall $primary;
    }
    """

    def render(self) -> Text:
        """The three-line banner: the mark, the product name and the safety promise."""
        t = Text()
        t.append("█▀▀▄ ", style="bold #818cf8")
        t.append("PROMPTISE", style="bold #c7d2fe")
        t.append("  ·  ", style="#4b5563")
        t.append("MCPcast", style="bold #818cf8")
        t.append("  guided setup\n", style="#9ca3af")
        t.append("█▄▄▀ ", style="bold #6366f1")
        t.append(
            "Turn your existing API into a curated, safe, agent-ready MCP server\n", style="#e5e7eb"
        )
        t.append("█    ", style="bold #4f46e5")
        t.append(
            "Read-only by default · writes need human approval · you review everything",
            style="#6b7280",
        )
        return t


class StepList(Static):
    """The sidebar: ● done, ▶ current, ○ pending."""

    DEFAULT_CSS = """
    StepList {
        width: 20;
        height: 100%;
        padding: 1 1;
        background: $surface;
        border-right: solid $primary-darken-2;
    }
    """

    def show(self, current: int) -> None:
        """Redraw the list with step *current* (1-based; 0 is the welcome screen) as active."""
        t = Text()
        for i, name in enumerate(STEP_NAMES, start=1):
            if i < current:
                t.append(f"● {i} {name}\n", style="#34d399")
            elif i == current:
                t.append(f"▶ {i} {name}\n", style="bold #c7d2fe")
            else:
                t.append(f"○ {i} {name}\n", style="#6b7280")
        self.update(t)


class Explain(Vertical):
    """A bordered "Why this matters" box with Markdown inside."""

    DEFAULT_CSS = """
    Explain {
        height: auto;
        border: round $primary-darken-1;
        border-title-color: $primary;
        padding: 0 1;
        margin: 1 1 0 1;
    }
    Explain Markdown { margin: 0; padding: 0; }
    """

    def __init__(self, markdown: str, title: str = "Why this matters", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._markdown = markdown
        self.border_title = title

    def compose(self) -> ComposeResult:
        """The Markdown body inside the border."""
        yield Markdown(self._markdown, open_links=False)  # links go through the app's gate


class StepPane(Vertical):
    """One step: a heading, a form, the explanation, and Back / Continue."""

    STEP: ClassVar[int] = 0
    TITLE: ClassVar[str] = ""
    EXPLAIN: ClassVar[str] = ""
    NEXT_LABEL: ClassVar[str] = "Continue"

    DEFAULT_CSS = """
    StepPane { height: 1fr; }
    StepPane .heading { height: 2; padding: 0 1; text-style: bold; color: $primary; }
    StepPane .body { height: 1fr; }
    StepPane .label { color: $text-muted; padding: 1 1 0 1; height: auto; }
    StepPane .status { padding: 0 1; height: auto; }
    StepPane .error { color: $error; padding: 0 1; height: auto; }
    StepPane .buttons { height: 3; dock: bottom; align-horizontal: right; padding: 0 1; }
    StepPane .buttons Button { margin-left: 1; min-width: 14; }
    StepPane Input { margin: 0 1; }
    StepPane OptionList { height: auto; margin: 0 1; }
    StepPane Switch { margin: 0 1; }
    StepPane LoadingIndicator { height: 1; margin: 0 1; }
    StepPane .row { height: auto; margin: 0 1; }
    StepPane .row Button { margin-right: 1; }
    StepPane OptionList > .option-list--option-highlighted { background: $primary 25%; }
    StepPane DataTable > .datatable--cursor { background: $primary 35%; }
    """

    @property
    def wizard(self) -> MCPcastWizard:
        """The app this pane belongs to, typed."""
        return cast(MCPcastWizard, self.app)

    @property
    def settings(self) -> WizardSettings:
        """The wizard's :class:`WizardSettings` — what every step reads and fills in."""
        return self.wizard.settings

    def compose(self) -> ComposeResult:
        """The shared frame: heading, scrollable form, error line, explanation, buttons."""
        total = len(STEP_NAMES)
        head = f"Step {self.STEP} of {total} · {self.TITLE}" if self.STEP else self.TITLE
        yield Static(head, classes="heading")
        with VerticalScroll(classes="body"):
            yield from self.compose_form()
            yield Static("", classes="error", id=f"{self.id}-error")
            if self.EXPLAIN:
                yield Explain(self.EXPLAIN)
        with Horizontal(classes="buttons"):
            if self.STEP:
                yield Button("Back", id=f"{self.id}-back")
            yield Button(self.NEXT_LABEL, variant="primary", id=f"{self.id}-next")

    def compose_form(self) -> ComposeResult:
        """The step's own widgets, placed inside the frame; panes override this."""
        yield from ()

    def enter(self) -> None:
        """Called every time the pane becomes visible."""

    def validate(self) -> str | None:
        """Return an error message, or ``None`` when the step may be left."""
        return None

    def show_error(self, message: str | None) -> None:
        """Show *message* on the pane's error line, or hide the line for ``None``."""
        error = self.query_one(f"#{self.id}-error", Static)
        # As Text, never as a str: a str is parsed as markup, and messages quote
        # user input (a path, a URL, a pydantic error echoing the value) — and
        # spec text, so terminal controls are deleted too.
        error.update(Text(_console_safe(message or "")))
        error.display = bool(message)

    def on_mount(self) -> None:
        """Start with the error line hidden."""
        self.query_one(f"#{self.id}-error", Static).display = False

    @on(Button.Pressed)
    def _buttons(self, event: Button.Pressed) -> None:
        if event.button.id == f"{self.id}-next":
            event.stop()
            self.wizard.action_next()
        elif event.button.id == f"{self.id}-back":
            event.stop()
            self.wizard.action_previous()


class WelcomePane(StepPane):
    """Step 0: what the wizard does, plus the working directory, ``.env`` and model keys found."""

    TITLE = "Welcome"
    EXPLAIN = _EXPLAIN_WELCOME
    NEXT_LABEL = "Start"

    def compose_form(self) -> ComposeResult:
        """One status block, filled in by :meth:`enter`."""
        yield Static("", classes="status", id="welcome-env")

    def enter(self) -> None:
        """Report the working directory, the ``.env`` loaded and the providers with a key."""
        w = self.wizard
        t = Text()
        t.append("Working directory  ", style="#9ca3af")
        t.append(f"{w.cwd}\n")
        t.append("Secrets            ", style="#9ca3af")
        if w.dotenv_error:
            t.append("a .env was found but could not be read — see below\n", style="#f87171")
        elif w.dotenv:
            t.append(f".env loaded from {w.dotenv}\n", style="#34d399")
        else:
            t.append(
                f"no .env found — keys go in {w.cwd / '.env'} or the environment\n", style="#fbbf24"
            )
        t.append("Model keys found   ", style="#9ca3af")
        ready = _ready_providers()
        if ready:
            t.append(", ".join(ready), style="#34d399")
        else:
            t.append("none — step 2 explains where to put one, or choose offline", style="#fbbf24")
        self.query_one("#welcome-env", Static).update(t)
        self.show_error(w.dotenv_error)
        self.query_one("#welcome-next", Button).focus()


class SpecPane(StepPane):
    """Step 1: load the OpenAPI document — typed, or picked from a detected local API.

    Loading runs on a daemon thread of its own (see :meth:`_load`) and
    detection on a thread worker; both poll a cancellation flag, so a newer
    load, a new probe or Ctrl+Q cuts a slow transfer short rather than
    waiting for it (see :meth:`ParsedSpec.load`).
    """

    STEP = 1
    TITLE = "Where is your API's OpenAPI document?"
    EXPLAIN = _EXPLAIN_SPEC

    class Loaded(Message):
        """Posted by the loading thread when :meth:`ParsedSpec.load` returned or failed."""

        def __init__(self, generation: int, parsed: ParsedSpec | None, error: str | None) -> None:
            super().__init__()
            self.generation = generation
            self.parsed = parsed
            self.error = error

    def compose_form(self) -> ComposeResult:
        """The source field, Load / Detect buttons, candidate list, status and base URL field."""
        yield Static("File path, URL, or Enter to load", classes="label")
        yield Input(
            placeholder="openapi.json · ./api/openapi.yaml · http://127.0.0.1:8000/openapi.json",
            id="spec-input",
        )
        with Horizontal(classes="row"):
            yield Button("Load", id="spec-load")
            yield Button("Detect a running local API", id="spec-detect")
        yield OptionList(id="spec-candidates")
        yield LoadingIndicator(id="spec-loading")
        yield Static("", classes="status", id="spec-status")
        yield Static("API base URL — where the generated server sends requests", classes="label")
        yield Input(placeholder="https://api.example.com", id="spec-base-url")

    def on_mount(self) -> None:
        """Hide the candidate list and the spinner until there is something to show."""
        super().on_mount()
        self.query_one("#spec-candidates").display = False
        self.query_one("#spec-loading").display = False

    def enter(self) -> None:
        """Pre-fill from the CLI, load a pre-filled spec, or start detection on an empty field."""
        spec_input = self.query_one("#spec-input", Input)
        base_input = self.query_one("#spec-base-url", Input)
        if self.settings.base_url_override and not base_input.value:
            base_input.value = self.settings.base_url  # --base-url from the CLI
        if self.settings.spec and not spec_input.value:
            spec_input.value = self.settings.spec
            self._load(self.settings.spec)
        elif not spec_input.value and self.wizard.auto_detect and not self._detected:
            self._detect()
        spec_input.focus()

    _detected = False
    _loading = False
    _load_generation = 0
    """Bumped per load; a thread whose number is no longer current is superseded."""
    _load_cancel: threading.Event | None = None
    """The current load's cancellation flag — set by a newer load and by Ctrl+Q."""

    @on(Input.Submitted, "#spec-input")
    def _submit_spec(self, event: Input.Submitted) -> None:
        event.stop()
        parsed = self.wizard.parsed
        if (
            parsed is not None
            and parsed.source == public_source(event.value.strip())
            and not self._loading
        ):
            self.wizard.action_next()
        else:
            self._load(event.value)

    @on(Input.Submitted, "#spec-base-url")
    def _submit_base(self, event: Input.Submitted) -> None:
        event.stop()
        self.wizard.action_next()

    @on(Button.Pressed, "#spec-load")
    def _load_button(self, event: Button.Pressed) -> None:
        event.stop()
        self._load(self.query_one("#spec-input", Input).value)

    @on(Button.Pressed, "#spec-detect")
    def _detect_button(self, event: Button.Pressed) -> None:
        event.stop()
        self._detect()

    _candidates: dict[str, Candidate] = {}

    @on(OptionList.OptionSelected, "#spec-candidates")
    def _pick_candidate(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        url = str(event.option.id)
        self.query_one("#spec-input", Input).value = url
        candidate = self._candidates.get(url)
        self._load(url, document=candidate.document if candidate else None)

    def _load(self, source: str, *, document: dict[str, Any] | None = None) -> None:
        source = source.strip()
        self.show_error(None)
        self.wizard.set_parsed(None)
        if not source:
            self.show_error("Enter the path or URL of your API's OpenAPI document.")
            return
        override = self.query_one("#spec-base-url", Input).value.strip()
        self._loading = True
        self._load_generation += 1
        self.abandon_load()  # a newer load supersedes the one in flight
        cancel = threading.Event()
        self._load_cancel = cancel
        self.query_one("#spec-loading").display = True
        self.query_one("#spec-status", Static).update(
            Text(f"Loading {public_source(source)} …", style="#9ca3af")
        )
        # A daemon thread of its own rather than a thread worker: a worker runs
        # on asyncio's default executor, which App.run joins on the way out, so
        # Ctrl+Q would wait for a CPU-bound parse (seconds for a large YAML
        # document) to finish. A daemon thread is left behind instead; its
        # outcome is dropped when its generation is no longer current.
        threading.Thread(
            target=self._load_in_thread,
            args=(
                self._load_generation,
                source,
                override or None if self.settings.base_url_override else None,
                document,
                cancel,
            ),
            name=f"mcpcast-spec-load-{self._load_generation}",
            daemon=True,
        ).start()

    def abandon_load(self) -> None:
        """Cancel the load in flight, if any: its download stops within one chunk."""
        if self._load_cancel is not None:
            self._load_cancel.set()
            self._load_cancel = None

    def _load_in_thread(
        self,
        generation: int,
        source: str,
        override: str | None,
        document: dict[str, Any] | None,
        cancel: threading.Event,
    ) -> None:
        # Every failure — network, parsing (a RecursionError from a deeply
        # nested document included), validation — is reported, never raised
        # out of the thread; the messages never repeat a credential.
        # The download polls the flag after every chunk, so a newer load or
        # Ctrl+Q ends a slow transfer within one chunk instead of after the
        # whole document. A parse already under way cannot be interrupted, but
        # nothing waits for it: the thread is a daemon and its outcome is
        # dropped once superseded or once the app is gone.
        try:
            parsed = ParsedSpec.load(
                source, base_url=override, document=document, cancelled=cancel.is_set
            )
        except MCPcastError as exc:
            outcome: tuple[ParsedSpec | None, str | None] = (None, _redact(str(exc), source))
        except Exception as exc:
            outcome = (None, _redact(f"{type(exc).__name__}: {exc}", source))
        else:
            outcome = (parsed, None)
        if cancel.is_set() or generation != self._load_generation:
            return
        try:
            # post_message is thread-safe and never blocks; after the app
            # exited the pane is closed (False) or its loop is (RuntimeError).
            self.post_message(self.Loaded(generation, *outcome))
        except RuntimeError:
            return

    @on(Loaded)
    def _loaded_message(self, event: Loaded) -> None:
        event.stop()
        self._loaded(event.generation, event.parsed, event.error)

    def _loaded(self, generation: int, parsed: ParsedSpec | None, error: str | None) -> None:
        if generation != self._load_generation:
            return  # superseded: a newer load owns the screen and the settings
        self._loading = False
        self._load_cancel = None
        self.query_one("#spec-loading").display = False
        status = self.query_one("#spec-status", Static)
        spec_input = self.query_one("#spec-input", Input)
        if parsed is None:
            status.update("")
            self.show_error(error)
            spec_input.focus()
            return
        self.wizard.set_parsed(parsed)
        self.settings.spec = parsed.source
        self.settings.derived_name = parsed.api_name
        typed = spec_input.value.strip()
        redacted = typed != parsed.source and public_source(typed) == parsed.source
        if redacted:
            spec_input.value = parsed.source
        base = self.query_one("#spec-base-url", Input)
        if not self.settings.base_url_override:
            base.value = parsed.base_url
        self.settings.base_url = base.value.strip() or parsed.base_url
        t = Text()
        t.append("✓ ", style="bold #34d399")
        t.append(parsed.summary())
        if redacted:
            t.append(
                "\n⚠ Credentials removed from the URL — they were used for this fetch only and "
                "are recorded nowhere.",
                style="#fbbf24",
            )
        status.update(t)
        self.query_one("#spec-next", Button).focus()

    def _detect(self) -> None:
        self._detected = True
        self.show_error(None)
        self.query_one("#spec-loading").display = True
        self.query_one("#spec-status", Static).update(
            Text("Looking for a running local API on the usual ports …", style="#9ca3af")
        )
        self._detect_worker()

    @work(thread=True, exclusive=True, group="detect")
    def _detect_worker(self) -> None:
        worker = get_current_worker()
        try:
            detection = probe_local_apis(
                fetch=self.wizard.fetch, cancelled=lambda: worker.is_cancelled
            )
        except Exception as exc:  # a probe is best effort: report, never crash the wizard
            if not worker.is_cancelled:
                self.app.call_from_thread(self._detect_failed, f"{type(exc).__name__}: {exc}")
            return
        if not worker.is_cancelled:
            self.app.call_from_thread(self._detected_done, detection)

    def _detect_failed(self, message: str) -> None:
        self.query_one("#spec-loading").display = False
        self.query_one("#spec-status", Static).update("")
        self.show_error(f"Detection failed: {message}. Enter a file or URL instead.")

    def _detected_done(self, detection: Detection) -> None:
        self.query_one("#spec-loading").display = False
        options = self.query_one("#spec-candidates", OptionList)
        options.clear_options()
        found = detection.candidates
        self._candidates = {c.url: c for c in found}
        status = self.query_one("#spec-status", Static)
        skipped = detection.skipped_line()
        if not found:
            options.display = False
            probed = [p for p in PROBE_PORTS if p not in detection.skipped_ports]
            t = Text(
                f"No local API found on ports {', '.join(map(str, probed))}."
                if probed
                else "No local API found.",
                style="#fbbf24",
            )
            if skipped:
                t.append(f" {skipped}", style="#fbbf24")
            t.append(" Start yours, or enter a file or URL.", style="#fbbf24")
            status.update(t)
            return
        for c in found:
            prompt = Text()
            prompt.append(_console_safe(c.title), style="bold")  # info.title is spec text
            prompt.append(f"  {c.operations} operations\n", style="#9ca3af")
            prompt.append(c.url, style="#818cf8")
            options.add_option(Option(prompt, id=c.url))
        options.display = True
        options.highlighted = 0  # Enter picks the first one straight away
        t = Text(f"Found {len(found)} — Enter picks the highlighted one:", style="#34d399")
        if skipped:
            t.append(f"\n{skipped}", style="#fbbf24")
        status.update(t)
        options.focus()

    @on(Input.Changed, "#spec-base-url")
    def _base_changed(self, event: Input.Changed) -> None:
        parsed = self.wizard.parsed
        value = event.value.strip()
        self.settings.base_url = value
        if parsed is not None:
            self.settings.base_url_override = bool(value and value != parsed.declared_base_url)

    def validate(self) -> str | None:
        """A document must be loaded; a changed base URL re-extracts the operations with it."""
        parsed = self.wizard.parsed
        if parsed is None:
            return "Load an OpenAPI document first (Enter in the field above)."
        base = self.query_one("#spec-base-url", Input).value.strip()
        override = base if base and base != parsed.declared_base_url else None
        try:
            if (override or parsed.declared_base_url) != parsed.base_url:
                # The field changed since the operations were extracted: extract
                # again with the override, as the CLI does for --base-url, so the
                # plan (and every route in it) uses what was typed.
                parsed = parsed.with_base_url(override)
                self.wizard.set_parsed(parsed)
            build_plan(
                parsed.operations,
                base_url=override,
                auth=AuthMode.ENV_TOKEN,  # validates the URL; the auth mode is step 4's
                name=parsed.api_name,
                classifications=parsed.classifications,
            )
        except (ValueError, MCPcastError) as exc:
            return f"API base URL: {exc}"
        self.settings.base_url = parsed.base_url
        self.settings.base_url_override = override is not None
        return None


class ModelPane(StepPane):
    """Step 2: curate with a model (``--model``) or derive the tools offline (``--no-curate``)."""

    STEP = 2
    TITLE = "Who designs the tools?"
    EXPLAIN = _EXPLAIN_MODEL

    def compose_form(self) -> ComposeResult:
        """The curate / offline choice, the model field and its readiness line."""
        yield OptionList(
            Option(
                Text.assemble(
                    ("Design the tools with a model", "bold"),
                    ("  recommended\n", "#34d399"),
                    (
                        "Names, descriptions and parameter choices written for an AI assistant",
                        "#9ca3af",
                    ),
                ),
                id="curate",
            ),
            Option(
                Text.assemble(
                    ("Offline — derive tools from the spec only\n", "bold"),
                    ("One tool per operation, deterministic, no model and no network", "#9ca3af"),
                ),
                id="offline",
            ),
            id="model-choice",
        )
        yield Static("Model  (provider:model — Enter to check and continue)", classes="label")
        yield Input(value=self.settings.model, id="model-input")  # --model pre-fills it
        yield Static("", classes="status", id="model-status")

    def enter(self) -> None:
        """Highlight the current choice and check the model's credentials straight away."""
        choice = self.query_one("#model-choice", OptionList)
        choice.highlighted = 0 if self.settings.curate else 1
        self._check()
        choice.focus()

    @on(OptionList.OptionHighlighted, "#model-choice")
    def _highlight(self, event: OptionList.OptionHighlighted) -> None:
        curate = event.option.id == "curate"
        self.settings.curate = curate
        self.query_one("#model-input", Input).disabled = not curate
        self._check()

    @on(OptionList.OptionSelected, "#model-choice")
    def _select(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.settings.curate = event.option.id == "curate"
        self.wizard.action_next()

    @on(Input.Changed, "#model-input")
    def _model_changed(self, event: Input.Changed) -> None:
        self.settings.model = event.value.strip() or DEFAULT_MODEL
        self._check()

    @on(Input.Submitted, "#model-input")
    def _model_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.wizard.action_next()

    def _check(self) -> tuple[bool, str]:
        status = self.query_one("#model-status", Static)
        if not self.settings.curate:
            status.update(Text("Offline: no model, no key needed.", style="#9ca3af"))
            return True, ""
        ok, message = _model_status(self.settings.model)
        status.update(Text(message, style="#34d399" if ok else "#f87171"))
        return ok, message

    def validate(self) -> str | None:
        """Under curation the model must be usable now (its variables set, or offline chosen)."""
        ok, message = self._check()
        if ok:
            self.show_error(None)
            return None
        return (
            "This model cannot be used yet — fix the variables above, pick another "
            "provider, or choose Offline."
        )


class SafetyPane(StepPane):
    """Step 3: the safety profile (``--profile``), each row previewing what it would generate."""

    STEP = 3
    TITLE = "What may an agent do?"
    EXPLAIN = _EXPLAIN_SAFETY

    def compose_form(self) -> ComposeResult:
        """The profile list; its rows are computed from the loaded spec in :meth:`enter`."""
        yield OptionList(id="safety-choice")

    def enter(self) -> None:
        """Rebuild the rows with :func:`preview_profile` counts for the loaded spec."""
        parsed = self.wizard.parsed
        choice = self.query_one("#safety-choice", OptionList)
        choice.clear_options()
        notes = {
            SafetyProfile.READ_ONLY: ("read-only", "recommended to start"),
            SafetyProfile.STANDARD: ("standard", "writes are generated, each one approval-gated"),
            SafetyProfile.FULL: (
                "full",
                "everything, destructive and financial calls approval-gated",
            ),
        }
        for profile in SafetyProfile:
            label, note = notes[profile]
            prompt = Text()
            prompt.append(f"{label:<10}", style="bold")
            if parsed is not None:
                prompt.append(preview_profile(parsed, profile).line(), style="#c7d2fe")
            prompt.append(f"\n{'':<10}{note}", style="#9ca3af")
            choice.add_option(Option(prompt, id=profile.value))
        choice.highlighted = list(SafetyProfile).index(self.settings.profile)
        choice.focus()

    @on(OptionList.OptionSelected, "#safety-choice")
    def _select(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.settings.profile = SafetyProfile(str(event.option.id))
        self.wizard.action_next()

    def validate(self) -> str | None:
        """Record the highlighted profile; there is nothing to reject."""
        choice = self.query_one("#safety-choice", OptionList)
        if choice.highlighted is not None:
            self.settings.profile = SafetyProfile(
                str(choice.get_option_at_index(choice.highlighted).id)
            )
        return None


_USAGE_ROWS: tuple[tuple[Usage, str, str], ...] = (
    (
        Usage.PERSONAL,
        "Personal — a desktop client",
        "Claude Desktop, Claude Code or Cursor launches it over stdio",
    ),
    (
        Usage.SHARED,
        "Shared service — users bring their own token",
        "One HTTP deployment; every caller acts as themselves",
    ),
    (
        Usage.TENANTS,
        "Multi-tenant product",
        "MCP clients present API keys; approvals reviewed by a second person",
    ),
    (Usage.OPEN, "No credentials", "The API is open, or this is local development only"),
)


class AuthPane(StepPane):
    """Step 4: how the server is used, which decides ``--auth`` and the approval mode."""

    STEP = 4
    TITLE = "How will this server be used?"
    EXPLAIN = _EXPLAIN_AUTH

    def compose_form(self) -> ComposeResult:
        """The usage list and the line explaining the auth mode it implies."""
        options = []
        for usage, title, note in _USAGE_ROWS:
            prompt = Text()
            prompt.append(title, style="bold")
            prompt.append(f"  → --auth {recommended_auth(usage).value}\n", style="#818cf8")
            prompt.append(note, style="#9ca3af")
            options.append(Option(prompt, id=usage.value))
        yield OptionList(*options, id="auth-choice")
        yield Static("", classes="status", id="auth-hint")
        yield Static("Approval for gated calls", classes="label")
        yield Select(
            [
                ("Default for this auth mode", "default"),
                ("elicitation — the calling client asks its own user", "elicitation"),
                ("pending — a second person decides (api-key only)", "pending"),
            ],
            value="default",
            allow_blank=False,
            id="auth-approval",
        )

    def enter(self) -> None:
        """Highlight the current usage and explain the auth mode it recommends."""
        choice = self.query_one("#auth-choice", OptionList)
        choice.highlighted = [u for u, _, _ in _USAGE_ROWS].index(self.settings.usage)
        self._hint()
        choice.focus()

    def _hint(self) -> None:
        self.query_one("#auth-hint", Static).update(
            Text(_AUTH_ENV_HINT[recommended_auth(self.settings.usage)], style="#c7d2fe")
        )

    @on(OptionList.OptionHighlighted, "#auth-choice")
    def _highlight(self, event: OptionList.OptionHighlighted) -> None:
        self.settings.usage = Usage(str(event.option.id))
        self.settings.auth = recommended_auth(self.settings.usage)
        self._hint()

    @on(OptionList.OptionSelected, "#auth-choice")
    def _select(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.settings.usage = Usage(str(event.option.id))
        self.settings.auth = recommended_auth(self.settings.usage)
        self.wizard.action_next()

    @on(Select.Changed, "#auth-approval")
    def _approval(self, event: Select.Changed) -> None:
        value = event.value
        self.settings.approval = (
            None if value in ("default", Select.BLANK) else ApprovalMode(str(value))
        )

    def validate(self) -> str | None:
        """Refuse combinations the planner would refuse later, with the same words.

        ``pending`` approval needs identified callers (API-key auth), and
        ``passthrough`` can only relay the caller's ``Authorization`` header —
        a spec whose credential lives in another header or a query parameter
        (its ``securitySchemes``) needs ``env-token`` or ``api-key``.
        """
        if (
            self.settings.approval is ApprovalMode.PENDING
            and self.settings.auth is not AuthMode.API_KEY
        ):
            return (
                "Approval 'pending' needs identified callers so nobody approves their own "
                "request — choose the multi-tenant option (api-key), or elicitation."
            )
        parsed = self.wizard.parsed
        if parsed is not None:
            try:
                refuse_unrelayable_credential(
                    self.settings.auth, credential_slot(parsed.operations)
                )
            except MCPcastError as exc:
                return str(exc)
        return None


_DERIVED_FILES = ("README.md", "server.py", "tests/conftest.py", "tests/test_tools.py")
"""Project files a write regenerates unconditionally (besides the package and the plan)."""


def _folder_state(target: Path, pkg: str) -> tuple[str, list[str]]:
    """What is at *target*, and which of the files a write regenerates are already there.

    Returns:
        ``(state, replaced)`` — *state* is ``"free"`` (absent or empty),
        ``"project"`` (holds ``mcpcast.plan.yaml``), ``"occupied"`` (has
        files, but no plan: somebody else's folder) or ``"file"``; *replaced*
        lists the derived files present (``<pkg>/`` for the package).

    Raises:
        OSError: when the folder cannot be read.
    """
    if not target.exists():
        return "free", []
    if not target.is_dir():
        return "file", []
    if not any(target.iterdir()):
        return "free", []
    replaced = [f for f in _DERIVED_FILES if (target / f).exists()]
    if (target / pkg).is_dir():
        replaced.append(f"{pkg}/")
    if (target / "mcpcast.plan.yaml").exists():
        return "project", replaced
    return "occupied", replaced


def _renamed_package_note(target: Path, name: str) -> str:
    """A sentence naming the generated package(s) in *target* that another ``api.name`` left.

    Writing with overwrite on puts the new ``<name>_mcp/`` beside the old
    package and leaves ``pyproject.toml``/``Dockerfile`` (which still ship
    the old one) to the user — :func:`~promptise.mcpcast.emit.write_project`
    refuses the same situation without ``--force``.  Empty when there is
    nothing to say, or the folder cannot be read.
    """
    if not re.match(r"^[a-z0-9][a-z0-9_-]{0,62}$", name):
        return ""
    try:
        stale = _stale_packages(target, package_name(name))
    except OSError:
        return ""
    if not stale:
        return ""
    old = ", ".join(f"{s}/" for s in stale)
    return (
        f" It holds {old}, generated under another name: overwrite writes {package_name(name)}/ "
        f"beside it and leaves pyproject.toml and Dockerfile — which still ship {old} — to you; "
        f"delete {old} first, or keep the previous name."
    )


class ProjectPane(StepPane):
    """Step 5: server name, output folder, tool budget, evaluation and overwrite switches."""

    STEP = 5
    TITLE = "Name, folder and budget"
    EXPLAIN = _EXPLAIN_PROJECT

    def compose_form(self) -> ComposeResult:
        """Name, folder and budget fields plus the evaluation and overwrite switches."""
        yield Static("Server name  (lowercase, digits, - or _)", classes="label")
        yield Input(id="project-name")
        yield Static("Output folder", classes="label")
        yield Input(id="project-out")
        yield Static(
            "Tool budget  (blank = default: 25 with a model, unlimited offline)", classes="label"
        )
        yield Input(placeholder="25", id="project-budget")
        with Horizontal(classes="switch-row"):
            yield Switch(id="project-eval")
            yield Static(
                "Run the Agent Readiness evaluation after writing (uses the model)",
                classes="switch-label",
            )
        yield Static("", classes="status", id="project-eval-hint")
        with Horizontal(classes="switch-row", id="project-force-row"):
            yield Switch(id="project-force")
            yield Static("", classes="switch-label", id="project-force-label")

    DEFAULT_CSS = """
    ProjectPane .switch-row { height: auto; margin: 1 1 0 1; }
    ProjectPane .switch-label { padding: 1 0 0 1; width: 1fr; }
    """

    def on_mount(self) -> None:
        """Hide the overwrite switch until the folder turns out to be occupied."""
        super().on_mount()
        self.query_one("#project-force-row").display = False

    def enter(self) -> None:
        """Fill the fields from the settings (name from the spec, folder from ``--out``)."""
        s = self.settings
        name = self.query_one("#project-name", Input)
        if not name.value:
            name.value = s.name or s.derived_name
        out = self.query_one("#project-out", Input)
        if not out.value:
            out.value = s.out_dir or f"{name.value}-mcp"
        eval_switch = self.query_one("#project-eval", Switch)
        eval_switch.disabled = not s.curate
        if not s.curate:
            eval_switch.value = False
        self.query_one("#project-budget", Input).placeholder = (
            str(_CURATION_BUDGET) if s.curate else "unlimited"
        )
        if s.force:  # --force pre-fills the overwrite switch
            self.query_one("#project-force", Switch).value = True
        self._check_existing()
        self._eval_hint()
        name.focus()

    def _target(self, out: str) -> Path:
        """*out* as the folder that would be written, ``~`` expanded, relative to the cwd."""
        return self.wizard.cwd / Path(out).expanduser()

    @on(Switch.Changed, "#project-eval")
    def _eval_toggled(self, event: Switch.Changed) -> None:
        self._eval_hint()

    def _eval_hint(self) -> None:
        """Live reads during the evaluation reach your API: say how they authenticate."""
        hint = self.query_one("#project-eval-hint", Static)
        if not self.query_one("#project-eval", Switch).value or self.settings.auth is AuthMode.NONE:
            hint.update("")
            return
        if os.environ.get("MCPCAST_EVAL_AUTHORIZATION") or os.environ.get("MCPCAST_EVAL_HEADERS"):
            hint.update(
                Text(
                    "✓ MCPCAST_EVAL_AUTHORIZATION is set — the evaluation's live reads reach "
                    "your API authenticated.",
                    style="#34d399",
                )
            )
        else:
            hint.update(
                Text(
                    "⚠ MCPCAST_EVAL_AUTHORIZATION is not set: the evaluation's live reads will "
                    "carry a placeholder credential. If your API needs auth, quit (Ctrl+Q), "
                    "export MCPCAST_EVAL_AUTHORIZATION='Bearer <your API token>' and start "
                    "again — or evaluate later with --eval.",
                    style="#fbbf24",
                )
            )

    @on(Input.Changed, "#project-name")
    def _name_changed(self, event: Input.Changed) -> None:
        out = self.query_one("#project-out", Input)
        previous_default = f"{self.settings.name or self.settings.derived_name}-mcp"
        self.settings.name = event.value.strip()
        if out.value in ("", previous_default):
            out.value = f"{self.settings.name}-mcp"
        else:
            self._check_existing()  # the package name (<name>_mcp/) is part of what a write replaces

    @on(Input.Changed, "#project-out")
    def _out_changed(self, event: Input.Changed) -> None:
        self.settings.out_dir = event.value.strip()
        self._check_existing()

    def _check_existing(self) -> None:
        """Show the overwrite switch when the folder is taken — by a project or by anything else."""
        out = self.query_one("#project-out", Input).value.strip()
        name = self.query_one("#project-name", Input).value.strip()
        row = self.query_one("#project-force-row")
        state, replaced = "free", list[str]()
        if out:
            try:
                state, replaced = _folder_state(self._target(out), package_name(name))
            except (OSError, RuntimeError):  # unreadable, or ~ without a home: validate() says so
                state = "free"
        row.display = state in ("project", "occupied")
        if state == "project":
            renamed = _renamed_package_note(self._target(out), name)
            self.query_one("#project-force-label", Static).update(
                Text(
                    f"{out}/ already holds an mcpcast project — overwrite it? "
                    "(to keep plan edits, regenerate from its plan instead)" + renamed,
                    style="#fbbf24",
                )
            )
        elif state == "occupied":
            would = f" {', '.join(replaced)} would be replaced." if replaced else ""
            self.query_one("#project-force-label", Static).update(
                Text(
                    f"{out}/ has files in it and is not an mcpcast project — write into it "
                    f"anyway?{would}",
                    style="#fbbf24",
                )
            )

    @on(Input.Submitted)
    def _submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.wizard.action_next()

    def validate(self) -> str | None:
        """Record the fields; reject a bad name, budget, or a folder that must not be overwritten."""
        s = self.settings
        s.name = self.query_one("#project-name", Input).value.strip()
        typed_out = self.query_one("#project-out", Input).value.strip()
        budget = self.query_one("#project-budget", Input).value.strip()
        s.evaluate = self.query_one("#project-eval", Switch).value
        s.force = self.query_one("#project-force", Switch).value

        if not re.match(r"^[a-z0-9][a-z0-9_-]{0,62}$", s.name):
            return "Server name: lowercase letters, digits, '-' or '_' (max 63 characters)."
        if not typed_out:
            return "Output folder: enter a directory (it is created if missing)."
        try:
            # ~ is expanded here, once: the equivalent command then carries the
            # real path (a quoted '~/x' would not expand in a shell either).
            s.out_dir = str(Path(typed_out).expanduser())
            state, replaced = _folder_state(self._target(typed_out), package_name(s.name))
        except (OSError, RuntimeError) as exc:
            return f"Output folder: {exc}"
        if budget:
            if not budget.isdigit() or int(budget) < 1:
                return "Tool budget: a whole number of at least 1, or blank."
            s.max_tools = int(budget)
        else:
            s.max_tools = None
        if state == "file":
            return f"Output folder: {s.out_dir} is a file — enter a directory."
        if state == "project" and not s.force:
            return (
                f"{s.out_dir}/ already contains an mcpcast project. Switch on overwrite, "
                "choose another folder, or regenerate from its plan with "
                f"promptise mcpcast {s.out_dir}/mcpcast.plan.yaml"
                + _renamed_package_note(self._target(typed_out), s.name)
            )
        if state == "occupied" and not s.force:
            would = f" ({', '.join(replaced)} would be replaced)" if replaced else ""
            return (
                f"{s.out_dir}/ has files in it and is not an mcpcast project; choose another "
                f"folder or turn on overwrite{would}."
            )
        return None


class ReviewPane(StepPane):
    """Step 6: build the plan (curation or offline) and show every tool for review.

    The plan is rebuilt whenever :meth:`MCPcastWizard.plan_key` changed since it
    was built, so a reloaded spec or a changed setting is never reviewed stale.
    """

    STEP = 6
    TITLE = "Review what was decided"
    EXPLAIN = _EXPLAIN_REVIEW
    NEXT_LABEL = "Write project"

    DEFAULT_CSS = """
    ReviewPane #review-split { height: 1fr; margin: 0 1; }
    ReviewPane #review-left { width: 1fr; height: 1fr; }
    ReviewPane #review-right { width: 1fr; height: 1fr; margin-left: 1; }
    ReviewPane DataTable { height: 1fr; }
    ReviewPane #review-detail {
        height: 1fr; border: round $primary-darken-2; border-title-color: $primary; padding: 0 1;
    }
    ReviewPane #review-checklist {
        height: auto; margin-top: 1; padding: 0 1;
        border: round $warning; border-title-color: $warning;
    }
    ReviewPane #review-log { padding: 0 1; height: auto; color: $text-muted; }
    ReviewPane Collapsible { margin-top: 1; }
    """

    def compose(self) -> ComposeResult:
        """The review workspace: tool table and dropped list left, selected tool and checklist right."""
        # A workspace, not a page: everything a reviewer needs on screen at once.
        total = len(STEP_NAMES)
        yield Static(f"Step {self.STEP} of {total} · {self.TITLE}", classes="heading")
        yield LoadingIndicator(id="review-loading")
        yield Static("", id="review-log")
        yield Static("", classes="error", id="review-error")
        # open_links=False on every Markdown that shows spec or model text: a
        # link click reaches MCPcastWizard._link_clicked, which opens http(s)
        # only, instead of handing any URL scheme to the desktop.
        with Horizontal(id="review-split"):
            with Vertical(id="review-left"):
                yield DataTable(id="review-table", cursor_type="row", zebra_stripes=True)
                with Collapsible(title="Not exposed", id="review-dropped", collapsed=True):
                    yield Markdown("", id="review-dropped-list", open_links=False)
            with Vertical(id="review-right"):
                with VerticalScroll(id="review-detail") as detail:
                    detail.border_title = "Selected tool"
                    yield Markdown("", id="review-tool", open_links=False)
                checklist = Static(_CHECKLIST, id="review-checklist")
                checklist.border_title = "Never trust it blindly"
                yield checklist
        with Horizontal(classes="buttons"):
            yield Button("Back", id="review-back")
            yield Button(self.NEXT_LABEL, variant="primary", id="review-next")

    def on_mount(self) -> None:
        """Set up the tool table's columns."""
        super().on_mount()
        table = self.query_one("#review-table", DataTable)
        table.add_columns("Tool", "Risk", "Approval", "Operations")
        self._show_progress(False)

    def _show_progress(self, on: bool) -> None:
        self.query_one("#review-loading").display = on
        self.query_one("#review-split").display = not on
        self.query_one("#review-next", Button).disabled = on
        if on:
            # Nothing on the pane may take Enter while the plan is rebuilt —
            # least of all a field of the pane that was just hidden.
            self.query_one("#review-back", Button).focus()

    def enter(self) -> None:
        """Show the plan built for the current settings, or build it on a worker."""
        self.show_error(None)
        key = self.wizard.plan_key()
        if self.wizard.plan is not None and self.wizard.plan_built_for == key:
            self._fill(self.wizard.plan)
            return
        self._show_progress(True)
        s = self.settings
        if s.curate:
            self._log(
                f"Curating with {s.model} — designing up to {s.effective_budget} tools from "
                f"{len(self.wizard.parsed.operations) if self.wizard.parsed else 0} operations. "
                "This is a real model call and can take a minute …"
            )
        else:
            self._log("Deriving tools from the spec (offline) …")
        self._build_worker(key)

    def _log(self, text: str) -> None:
        self.query_one("#review-log", Static).update(Text(text, style="#9ca3af"))

    @work(exclusive=True, group="plan")
    async def _build_worker(self, key: tuple[Any, ...]) -> None:
        w = self.wizard
        parsed = w.parsed
        assert parsed is not None
        s = self.settings
        try:
            if s.curate:
                from .curate import curate

                plan = await curate(
                    parsed.operations,
                    model=s.model,
                    max_tools=s.effective_budget or _CURATION_BUDGET,
                    profile=s.profile,
                    base_url=s.base_url,
                    auth=s.auth,
                    approval=s.approval,
                    name=s.name,
                    description=parsed.description,
                    spec_source=parsed.label,
                    complete=w.completer,
                )
            else:
                plan = build_plan(
                    parsed.operations,
                    profile=s.profile,
                    base_url=s.base_url,
                    auth=s.auth,
                    approval=s.approval,
                    name=s.name,
                    description=parsed.description,
                    spec_source=parsed.label,
                    max_tools=s.effective_budget,
                    classifications=parsed.classifications,
                )
        except MCPcastError as exc:
            self._failed(str(exc))
            return
        except Exception as exc:  # provider/credential failures: the user must see them
            self._failed(f"{type(exc).__name__}: {exc}")
            return
        w.plan = plan
        w.plan_built_for = key
        self._fill(plan)

    def _failed(self, message: str) -> None:
        # The previous plan (if any) stays in memory but plan_built_for still
        # names the settings it was built for, so validate() refuses to write it.
        self._show_progress(False)
        self.query_one("#review-split").display = False
        self.query_one("#review-next", Button).disabled = True
        self._log("")
        self.show_error(
            message + "\n\nGo back (Esc) to change the model or choose Offline, then return here."
        )
        self.query_one("#review-back", Button).focus()

    def _fill(self, plan: MCPcastPlan) -> None:
        self._show_progress(False)
        gated = len(plan.gated_tools)
        summary = Text(
            f"{len(plan.tools)} tools ({gated} require approval) · "
            f"{len(plan.dropped)} operations not exposed · profile {plan.profile.value} · "
            f"auth {plan.api.auth.value} · approval {plan.api.approval_mode.value}",
            style="#9ca3af",
        )
        warnings = review_warnings(plan)
        for line in warnings[:3]:
            # the finding only; the explanation after the dash is in the plan review below
            summary.append(f"\n⚠ {_console_safe(line.split(' — ')[0])}", style="#fbbf24")
        if len(warnings) > 3:
            summary.append(f"\n⚠ +{len(warnings) - 3} more — read every row", style="#fbbf24")
        insecure = plain_http_hosts(plan)
        if insecure:
            summary.append(
                f"\n⚠ plain http upstream: {', '.join(insecure)} — the generated server refuses "
                "to send the credential there (UPSTREAM_INSECURE) unless "
                "MCPCAST_ALLOW_INSECURE_HTTP=1; prefer https",
                style="#fbbf24",
            )
        clipped = clipped_descriptions(plan)
        if clipped:
            shown = ", ".join(clipped[:5]) + (
                f" (+{len(clipped) - 5} more)" if len(clipped) > 5 else ""
            )
            summary.append(
                f"\n⚠ description cut for the agent: {shown} — longer than the generated "
                "tool description allows; shorten the description or example in the plan",
                style="#fbbf24",
            )
        self.query_one("#review-log", Static).update(summary)
        table = self.query_one("#review-table", DataTable)
        table.clear()
        for tool in plan.tools:
            first = tool.routes[0]
            ops = f"{first.method} {_one_line(first.path)}"  # the path is spec text
            if len(tool.routes) > 1:
                ops += f"  (+{len(tool.routes) - 1})"
            table.add_row(
                Text(tool.name, style="bold"),
                Text(tool.risk.value, style=_risk_style(tool.risk)),
                Text("required", style="#f87171")
                if tool.requires_approval
                else Text("—", style="#6b7280"),
                Text(ops),
                key=tool.name,
            )
        dropped = self.query_one("#review-dropped", Collapsible)
        dropped.title = f"Not exposed ({len(plan.dropped)})"
        self.query_one("#review-dropped-list", Markdown).update(_dropped_markdown(plan))
        if plan.tools:
            table.focus()
            table.move_cursor(row=0)
            self._detail(plan.tools[0])
        else:
            self.query_one("#review-tool", Markdown).update(
                "*No tools — widen the profile or the budget.*"
            )

    def _detail(self, tool: ToolPlan) -> None:
        self.query_one("#review-tool", Markdown).update(_tool_markdown(tool))

    @on(DataTable.RowHighlighted, "#review-table")
    def _row(self, event: DataTable.RowHighlighted) -> None:
        plan = self.wizard.plan
        if plan is None or event.row_key is None:
            return
        name = event.row_key.value
        for tool in plan.tools:
            if tool.name == name:
                self._detail(tool)
                break

    @on(DataTable.RowSelected, "#review-table")
    def _row_selected(self, event: DataTable.RowSelected) -> None:
        event.stop()
        self.wizard.action_next()

    def validate(self) -> str | None:
        """The plan must be built *for the current settings* and expose at least one tool.

        While a rebuild is in flight (or after one failed) the previous plan
        is still in memory; writing it would put a project on disk whose
        profile or auth mode is not what was chosen, under an equivalent
        command that does not reproduce it.
        """
        w = self.wizard
        if w.plan is None or w.plan_built_for != w.plan_key():
            return _PLAN_REBUILDING
        if not w.plan.tools:
            return "Nothing to write: the plan exposes no tools. Go back and widen the profile or budget."
        return None


_PLAN_REBUILDING = "The plan is being rebuilt for the current settings — wait, or go back."


class WritePane(StepPane):
    """Step 7: write the project, run the optional evaluation, show the equivalent command."""

    STEP = 7
    TITLE = "Written"
    EXPLAIN = _EXPLAIN_WRITE
    NEXT_LABEL = "Finish"

    DEFAULT_CSS = """
    WritePane #write-command { margin: 0 1; padding: 0 1; border: round $success; height: auto; }
    """

    def compose_form(self) -> ComposeResult:
        """The files written, the evaluation status, the command and what to do next."""
        yield Static("", classes="status", id="write-files")
        yield Static("", classes="status", id="write-warning")
        yield LoadingIndicator(id="write-loading")
        yield Static("", classes="status", id="write-eval")
        yield Static("Next time, without the wizard  (c copies it)", classes="label")
        yield Static("", id="write-command")
        yield Markdown("", id="write-next", open_links=False)

    def on_mount(self) -> None:
        """Start with the evaluation spinner hidden."""
        super().on_mount()
        self.query_one("#write-loading").display = False

    def enter(self) -> None:
        """Write the project unless this exact one is on disk already, then evaluate if asked."""
        w = self.wizard
        if w.plan is None or w.plan_built_for != w.plan_key():
            # ReviewPane.validate() refused already; never write a plan that
            # was built for other settings than the ones on screen.
            w.notify(_PLAN_REBUILDING, severity="error", timeout=6, markup=False)
            w.action_previous()
            return
        key = w.write_key()
        if w.result is not None and w.result_key == key:  # this exact project is on disk
            self.query_one("#write-next-btn", Button).focus()
            return
        plan = w.plan
        s = self.settings
        out_dir = w.cwd / s.out_dir
        self.show_error(None)
        for panel in ("#write-files", "#write-warning", "#write-eval", "#write-command"):
            self.query_one(panel, Static).update("")
        self.query_one("#write-next", Markdown).update("")
        try:
            written = write_project(plan, out_dir, force=s.force)
        except MCPcastError as exc:
            # A refusal with its own full message: a tools module that is not
            # ours, a package generated under a previous name, …
            self.show_error(str(exc))
            self.query_one("#write-next-btn", Button).focus()
            return
        except (
            Exception
        ) as exc:  # OSError, a surrogate the encoder refuses, … — the user must see it
            self.show_error(f"Could not write {out_dir}: {type(exc).__name__}: {exc}")
            self.query_one("#write-next-btn", Button).focus()
            return
        command = equivalent_command(s)
        w.result = WizardResult(
            plan=plan,
            out_dir=out_dir,
            written=written,
            command=command,
            eval_requested=s.evaluate,
        )
        w.result_key = key
        t = Text()
        t.append("✓ ", style="bold #34d399")
        t.append(f"{plan.api.name}", style="bold")
        t.append(f"  →  {s.out_dir}/\n")
        t.append(f"    {_console_safe(describe_written(written, out_dir))}\n", style="#c7d2fe")
        t.append(
            f"    {len(plan.tools)} tools · {len(plan.gated_tools)} require approval · "
            f"{len(plan.dropped)} not exposed",
            style="#9ca3af",
        )
        self.query_one("#write-files", Static).update(t)
        insecure = plain_http_hosts(plan)
        if insecure:
            self.query_one("#write-warning", Static).update(
                Text(
                    f"⚠ {', '.join(insecure)} is plain http: the generated server refuses to "
                    "send the upstream credential there (UPSTREAM_INSECURE) — every call, the "
                    "evaluation's included, fails until MCPCAST_ALLOW_INSECURE_HTTP=1 is set in "
                    "its environment. Do that only on a trusted network, or use an https base URL.",
                    style="#fbbf24",
                )
            )
        self.query_one("#write-command", Static).update(Text(command, style="bold #34d399"))
        self._next_steps(plan, Path(s.out_dir))
        if s.evaluate:
            self.query_one("#write-loading").display = True
            self.query_one("#write-eval", Static).update(
                Text(
                    f"Agent Readiness: running a real agent over the server with {s.model} "
                    f"({s.eval_tasks} tasks) …",
                    style="#9ca3af",
                )
            )
            self.query_one("#write-next-btn", Button).disabled = True
            # Enter must have a harmless target while the evaluation runs —
            # never a field of a hidden pane, whose submit would finish early.
            self.query_one("#write-copy", Button).focus()
            self._eval_worker(out_dir)
        else:
            self.query_one("#write-next-btn", Button).focus()

    def _next_steps(self, plan: MCPcastPlan, out_dir: Path) -> None:
        self.query_one("#write-next", Markdown).update(_next_steps_markdown(plan, out_dir))

    @work(exclusive=True, group="eval")
    async def _eval_worker(self, out_dir: Path) -> None:
        w = self.wizard
        s = self.settings
        assert w.plan is not None and w.parsed is not None and w.result is not None
        try:
            from .readiness import evaluate

            module = load_generated_server(out_dir / "server.py")
            report = await evaluate(
                w.plan,
                module.build_server,
                model=s.model,
                tasks=s.eval_tasks,
                operations=w.parsed.operations,
                complete=w.completer,
            )
            tasks_path, report_path = write_eval(report, [r.task for r in report.results], out_dir)
        except Exception as exc:
            self.query_one("#write-loading").display = False
            self.query_one("#write-eval", Static).update(
                Text(f"Agent Readiness failed: {type(exc).__name__}: {exc}", style="#f87171")
            )
            self.query_one("#write-next-btn", Button).disabled = False
            self.query_one("#write-next-btn", Button).focus()
            return
        w.result.report = report
        self.query_one("#write-loading").display = False
        grade_style = (
            "#34d399"
            if report.grade in ("A", "B")
            else "#fbbf24"
            if report.grade == "C"
            else "#f87171"
        )
        t = Text()
        t.append("Agent Readiness  ", style="#9ca3af")
        t.append(f"{report.grade}", style=f"bold {grade_style}")
        t.append(
            f"  {report.tasks_succeeded}/{report.tasks_total} tasks · "
            f"selection {report.selection_rate:.0%}\n"
        )
        for fix in report.fixes[:3]:
            t.append(f"  • {_console_safe(fix)}\n", style="#c7d2fe")  # names spec parameters
        t.append(f"  report: {report_path}", style="#6b7280")
        self.query_one("#write-eval", Static).update(t)
        self.query_one("#write-next-btn", Button).disabled = False
        self.query_one("#write-next-btn", Button).focus()

    def compose(self) -> ComposeResult:
        """The step frame with Copy command and Finish buttons."""
        # Same frame as StepPane, with the Finish button addressable for focus.
        total = len(STEP_NAMES)
        yield Static(f"Step {self.STEP} of {total} · {self.TITLE}", classes="heading")
        with VerticalScroll(classes="body"):
            yield from self.compose_form()
            yield Static("", classes="error", id=f"{self.id}-error")
            yield Explain(self.EXPLAIN, title="What was written")
        with Horizontal(classes="buttons"):
            yield Button("Copy command", id="write-copy")
            yield Button(self.NEXT_LABEL, variant="primary", id="write-next-btn")

    @on(Button.Pressed, "#write-copy")
    def _copy(self, event: Button.Pressed) -> None:
        event.stop()
        self.wizard.action_copy()

    @on(Button.Pressed, "#write-next-btn")
    def _finish(self, event: Button.Pressed) -> None:
        event.stop()
        self.wizard.action_next()


class HelpScreen(ModalScreen[None]):
    """F1: keys and what each step means."""

    BINDINGS = [Binding("escape,f1,q", "dismiss", "Close")]

    DEFAULT_CSS = """
    HelpScreen { align: center middle; }
    HelpScreen > VerticalScroll {
        width: 76; height: 80%; border: round $primary; background: $panel; padding: 0 2;
        border-title-color: $primary;
    }
    """

    def compose(self) -> ComposeResult:
        """The help text in a scrollable, bordered box."""
        with VerticalScroll() as box:
            box.border_title = "Help — Esc to close"
            yield Markdown(_HELP, open_links=False)


# ---------------------------------------------------------------------------
# The app
# ---------------------------------------------------------------------------

_PANES: tuple[type[StepPane], ...] = (
    WelcomePane,
    SpecPane,
    ModelPane,
    SafetyPane,
    AuthPane,
    ProjectPane,
    ReviewPane,
    WritePane,
)
_PANE_IDS = ("welcome", "spec", "model", "safety", "auth", "project", "review", "write")


class MCPcastWizard(App[WizardResult | None]):
    """The guided setup.

    Args:
        spec: Pre-fill the OpenAPI source (``promptise mcpcast SPEC --interactive``).
        base_url: Pre-fill the API base URL override.
        out_dir: Pre-fill the output folder (``--out``); default ``<name>-mcp``.
        cwd: Directory relative paths resolve against (default: the process cwd).
        completer: Override the model completion used by curation and the
            evaluation (tests inject a script; production leaves it ``None``).
        fetch: Override the HTTP fetch used by local API detection (tests).
        auto_detect: Look for a running local API when the spec step opens
            empty (off in tests).
        model: Pre-fill the curation model (``--model``); ``None`` is
            :data:`DEFAULT_MODEL`.
        eval_tasks: Pre-fill the Agent Readiness task count (``--eval-tasks``);
            ``None`` is the wizard's default.
        force: Pre-fill the overwrite switch (``--force``).
    """

    TITLE = "Promptise MCPcast"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("escape", "previous", "Back"),
        Binding("f1", "help", "Help"),
        Binding("c", "copy", "Copy command", show=False),
        Binding("ctrl+q", "quit", "Quit", priority=True),
    ]
    CSS = """
    Screen { layout: vertical; }
    #main { height: 1fr; }
    #panes { width: 1fr; }
    """

    def __init__(
        self,
        spec: str | None = None,
        *,
        base_url: str | None = None,
        out_dir: str | os.PathLike[str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
        completer: Completer | None = None,
        fetch: Fetcher | None = None,
        auto_detect: bool = True,
        model: str | None = None,
        eval_tasks: int | None = None,
        force: bool = False,
    ) -> None:
        super().__init__()
        self.cwd = Path(cwd or os.getcwd()).resolve()
        self.settings = WizardSettings(
            spec=spec or "", out_dir=str(out_dir) if out_dir else "", force=force
        )
        if base_url:
            self.settings.base_url = base_url
            self.settings.base_url_override = True
        if model:
            self.settings.model = model
        if eval_tasks is not None:
            self.settings.eval_tasks = eval_tasks
        self.completer = completer
        self.fetch = fetch
        self.auto_detect = auto_detect
        self.parsed: ParsedSpec | None = None
        self.spec_generation = 0
        """Bumped whenever :attr:`parsed` is replaced — part of :meth:`plan_key`, so a
        reloaded (edited) spec, or one re-extracted with another base URL, invalidates
        the built plan even when every setting reads the same."""
        self.plan: MCPcastPlan | None = None
        self.plan_built_for: tuple[Any, ...] | None = None
        self.result: WizardResult | None = None
        self.result_key: tuple[Any, ...] | None = None
        """The :meth:`write_key` the project on disk was written for."""
        self.dotenv: str | None = None
        self.dotenv_error: str | None = None
        """Why the nearest ``.env`` could not be loaded (unreadable, not UTF-8), shown on
        the welcome screen; ``None`` when there was no such file or it loaded."""
        self._index = 0

    def set_parsed(self, parsed: ParsedSpec | None) -> None:
        """Replace the loaded spec (see :attr:`spec_generation`)."""
        self.parsed = parsed
        self.spec_generation += 1

    def compose(self) -> ComposeResult:
        """Header, step list beside the pane switcher, footer."""
        yield BrandHeader()
        with Horizontal(id="main"):
            yield StepList(id="steps")
            with ContentSwitcher(initial="welcome", id="panes"):
                for pane_cls, pane_id in zip(_PANES, _PANE_IDS, strict=True):
                    yield pane_cls(id=pane_id)
        yield Footer()

    def on_mount(self) -> None:
        """Apply the theme, load the nearest ``.env`` and open the welcome screen.

        A ``.env`` that cannot be read (permissions, not UTF-8) is reported on
        the welcome screen — the message says what to do — rather than
        taking the wizard down; the CLI catches the same error before the
        wizard starts, so this matters when :func:`run_wizard` is called
        directly.
        """
        from promptise.models import ModelSetupError, load_dotenv_if_present

        self.register_theme(_BRAND)
        self.theme = "promptise"
        try:
            self.dotenv = load_dotenv_if_present(cwd=self.cwd)
        except ModelSetupError as exc:
            self.dotenv_error = str(exc)
        self._show(0)

    # -- navigation ---------------------------------------------------------

    @property
    def pane(self) -> StepPane:
        """The step currently shown."""
        return self.query_one(f"#{_PANE_IDS[self._index]}", StepPane)

    def _show(self, index: int) -> None:
        self._index = index
        self.query_one("#panes", ContentSwitcher).current = _PANE_IDS[index]
        self.query_one("#steps", StepList).show(index)
        # Hiding a pane moves the focus to the next focusable widget, which can
        # be a field of the pane just hidden (still focusable, only invisible):
        # Enter would then submit that field and move on. The pane about to be
        # shown decides what gets the focus.
        self.screen.set_focus(None)
        self.pane.enter()

    def action_next(self) -> None:
        """Validate the current step and move on (Enter / Continue)."""
        pane = self.pane
        error = pane.validate()
        if error:
            pane.show_error(error)
            # User text (a path, a URL) ends up in these messages: never markup,
            # and never a control character (spec text is quoted too).
            self.notify(
                _console_safe(error.splitlines()[0]), severity="error", timeout=6, markup=False
            )
            return
        pane.show_error(None)
        if self._index == len(_PANE_IDS) - 1:
            self.exit(self.result)
            return
        self._show(self._index + 1)

    def action_previous(self) -> None:
        """Go back one step (Esc).

        After a write the files stay on disk; :attr:`result` keeps describing
        them until the write step writes again (it does when anything the
        project depends on changed — see :meth:`write_key`).
        """
        if self._index == 0:
            return
        self._show(self._index - 1)

    async def action_quit(self) -> None:
        """Ctrl+Q: leave now.

        Work in flight — a spec download, a probe, curation, an evaluation —
        is cancelled, not awaited: the download and the probe poll their
        cancellation flag after every chunk they receive and unwind on their
        own, so leaving does not wait for a slow server's transfer to finish.
        A CPU-bound parse of a large document already under way cannot be
        interrupted; it runs on a daemon thread that nothing waits for, so
        the wizard still exits — after the moment it takes the parser to
        give the interpreter back (a second or two for a 13 MiB YAML
        document, not the whole parse).  Before step 7 nothing was written
        and ``None`` is returned; after it the project on disk is reported
        like a Finish, so the caller prints what was written (an evaluation
        still running is abandoned).
        """
        for spec_pane in self.query(SpecPane):
            spec_pane.abandon_load()
        self.workers.cancel_all()
        self.exit(self.result)

    def action_help(self) -> None:
        """F1: open the help screen."""
        self.push_screen(HelpScreen())

    @on(Markdown.LinkClicked)
    def _link_clicked(self, event: Markdown.LinkClicked) -> None:
        """Open a clicked Markdown link only when it is ``http`` or ``https``.

        Descriptions on the review step are spec or model text; every
        Markdown in the wizard has ``open_links=False`` so a link reaches this
        gate instead of ``webbrowser.open``, which would hand any registered
        URL scheme (``ssh://``, ``vnc://``, a system-preferences pane) to the
        desktop.
        """
        event.stop()
        href = event.href.strip()
        if urlsplit(href).scheme.lower() in ("http", "https"):
            self.open_url(href)
            return
        self.notify(
            f"Link not opened: {_console_safe(href)[:200]}", severity="warning", markup=False
        )

    def action_copy(self) -> None:
        """``c``: send the equivalent command to the clipboard once a project was written."""
        if self.result is None:
            return
        self.copy_to_clipboard(self.result.command)
        # Textual sends an OSC 52 sequence; the terminal decides whether to honour it.
        self.notify(
            "Command sent to the clipboard (OSC 52) — not every terminal supports it; "
            "the command stays on screen and is printed when the wizard exits.",
            timeout=5,
        )

    def plan_key(self) -> tuple[Any, ...]:
        """Everything the plan depends on — a change invalidates the built plan.

        Includes :attr:`spec_generation`, so a reloaded or re-extracted spec
        counts as a change even when the source string is the same.
        """
        s = self.settings
        return (
            self.spec_generation,
            s.spec,
            s.base_url,
            s.curate,
            s.model if s.curate else None,
            s.profile,
            s.auth,
            s.approval,
            s.name,
            s.effective_budget,
        )

    def write_key(self) -> tuple[Any, ...]:
        """Everything the written project depends on: the plan, the folder, the evaluation."""
        s = self.settings
        return (*self.plan_key(), s.out_dir, s.evaluate, s.eval_tasks if s.evaluate else None)


def run_wizard(
    spec: str | None = None,
    *,
    base_url: str | None = None,
    out_dir: str | os.PathLike[str] | None = None,
    cwd: str | os.PathLike[str] | None = None,
    model: str | None = None,
    eval_tasks: int | None = None,
    force: bool = False,
) -> WizardResult | None:
    """Open the guided setup in the current terminal and return what it wrote.

    Args:
        spec: Pre-fill the OpenAPI source.
        base_url: Pre-fill the API base URL override.
        out_dir: Pre-fill the output folder (default ``<name>-mcp``).
        cwd: Directory relative paths resolve against.
        model: Pre-fill the curation model (``None``: :data:`DEFAULT_MODEL`).
        eval_tasks: Pre-fill the Agent Readiness task count (``None``:
            :data:`~promptise.mcpcast.readiness.DEFAULT_EVAL_TASKS`).
        force: Pre-fill the overwrite switch.

    Returns:
        The :class:`WizardResult` — the project on disk, also when the wizard
        was quit with Ctrl+Q after writing — or ``None`` when nothing was written.
    """
    return MCPcastWizard(
        spec=spec,
        base_url=base_url,
        out_dir=out_dir,
        cwd=cwd,
        model=model,
        eval_tasks=eval_tasks,
        force=force,
    ).run()
