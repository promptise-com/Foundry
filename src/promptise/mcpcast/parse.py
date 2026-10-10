"""OpenAPI parsing for ``promptise mcpcast``.

Turns an OpenAPI 3.x (or Swagger 2.x) document into a flat list of
:class:`Operation` records carrying everything the classifier, planner,
curator and emitter need: method, path, per-parameter wire location,
dereferenced JSON Schemas, OAuth scopes, deprecation, tags and the success
response schema.

This is deliberately separate from
:class:`~promptise.mcp.server.OpenAPIProvider`, the *runtime* bridge that
registers one tool per route.  ``mcpcast`` is an authoring tool a developer
runs by hand against their own API — it needs richer metadata (scopes,
``deprecated``, parameter locations, response schemas) and must accept
``localhost`` base URLs that the provider's SSRF guard rightly rejects for
a long-running server.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field

from .schema import CredentialLocation, HttpMethod, MCPcastError, _scrub_tree

__all__ = [
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
]

_METHODS: tuple[str, ...] = ("get", "head", "post", "put", "patch", "delete", "options", "trace")
# Reference hops to follow before a recursive schema bottoms out. Two levels
# keep nested objects intact (Pet → Category) while staying linear on specs
# like Stripe's, whose mutually recursive schemas grow exponentially per hop
# (3 hops already expand to >200 MB of JSON).
_MAX_REF_DEPTH = 2
_DEFAULT_MAX_SPEC_BYTES = 20 * 1024 * 1024
_DEFAULT_FETCH_SECONDS = 60.0
MAX_DOCUMENT_NODES = 2_000_000
"""Default ceiling on the number of nodes a parsed document may expand to.

A 20 MiB document parses to a few million nodes at most; a 1 KB YAML alias
bomb (``a: &a [x]``, ``b: &b [*a, *a]``, …) expands to billions. The count is
taken *after* parsing, on the object graph, so it also catches aliases that
share one Python object many times over — every visit counts, so a shared
subtree costs what copying it would. Override with ``MCPCAST_MAX_SPEC_NODES``.
"""


def _max_spec_bytes() -> int:
    """The largest document the parser accepts (``MCPCAST_MAX_SPEC_BYTES``, default 20 MiB).

    A spec is fetched from a URL the developer chose, but a redirect can land
    anywhere; the cap keeps a hostile or runaway document from filling memory
    before it is even parsed.
    """
    return _positive_int_env("MCPCAST_MAX_SPEC_BYTES", _DEFAULT_MAX_SPEC_BYTES)


def _max_document_nodes() -> int:
    """The node budget for a parsed document (``MCPCAST_MAX_SPEC_NODES``, default 2,000,000)."""
    return _positive_int_env("MCPCAST_MAX_SPEC_NODES", MAX_DOCUMENT_NODES)


def _fetch_seconds() -> float:
    """Wall-clock budget for downloading a spec (``MCPCAST_FETCH_SECONDS``, default 60).

    ``httpx``'s timeout is per socket operation, so a server that trickles one
    byte every few seconds would otherwise hold the CLI open forever.
    """
    raw = os.environ.get("MCPCAST_FETCH_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_FETCH_SECONDS
    try:
        value = float(raw)
    except ValueError as exc:
        raise MCPcastError(f"MCPCAST_FETCH_SECONDS must be a number, got {raw!r}") from exc
    if value <= 0:
        raise MCPcastError(f"MCPCAST_FETCH_SECONDS must be positive, got {value}")
    return value


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise MCPcastError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise MCPcastError(f"{name} must be positive, got {value}")
    return value


class ParamSpec(BaseModel):
    """One input of an operation, with its wire location."""

    model_config = ConfigDict(extra="forbid")

    name: str
    location: Literal["path", "query", "header", "cookie", "body", "raw_body"]
    required: bool = False
    json_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "string"})
    description: str = ""
    wire_name: str | None = None
    """Name on the wire when it differs from ``name`` (set when a spec uses
    the same name in two locations, e.g. ``{username}`` in the path and
    ``username`` in the body — the body one is exposed as ``body_username``)."""

    @property
    def wire(self) -> str:
        """The name sent to the API (``wire_name`` when set, else ``name``)."""
        return self.wire_name or self.name


_SCHEME_TYPES = {
    "apikey": "apiKey",
    "http": "http",
    "oauth2": "oauth2",
    "openidconnect": "openIdConnect",
    "mutualtls": "mutualTLS",
    "basic": "basic",  # Swagger 2
}


class SecurityScheme(BaseModel):
    """One ``components.securitySchemes`` entry (Swagger 2: ``securityDefinitions``), resolved.

    What the planner needs to know about how an operation authenticates:
    where the credential travels. An ``apiKey`` scheme names a header, query
    parameter or cookie; HTTP ``bearer``/``basic``, OAuth 2 and OpenID
    Connect all arrive in the ``Authorization`` header.
    """

    model_config = ConfigDict(extra="forbid")

    key: str
    """Its name in the components map — what ``security`` requirements refer to."""
    type: str
    """``apiKey``, ``http``, ``oauth2``, ``openIdConnect``, ``mutualTLS`` or
    Swagger 2's ``basic`` (case-normalised)."""
    location: Literal["header", "query", "cookie"] | None = None
    """For ``apiKey``: where the key travels."""
    name: str | None = None
    """For ``apiKey``: the header or query parameter name."""
    scheme: str | None = None
    """For ``http``: the scheme, lower-cased (``bearer``, ``basic``, ``digest``…)."""

    @property
    def credential(self) -> tuple[CredentialLocation, str] | None:
        """``(location, name)`` the generated server must present the credential as.

        ``None`` when it cannot: an API key in a cookie, mutual TLS, or an
        unknown scheme type. Every ``Authorization``-borne scheme maps to
        ``("header", "Authorization")``.
        """
        if self.type == "apiKey":
            if self.location in ("header", "query") and self.name:
                return self.location, self.name  # type: ignore[return-value]
            return None
        if self.type in ("http", "basic", "oauth2", "openIdConnect"):
            return "header", "Authorization"
        return None

    def describe(self) -> str:
        """``an API key in header 'X-API-Key'``, ``HTTP bearer``… for messages."""
        if self.type == "apiKey":
            where = self.location or "an unknown location"
            return f"an API key in {where} {self.name or '<unnamed>'!r}"
        if self.type == "http":
            return f"HTTP {self.scheme or 'unknown'} authentication"
        if self.type == "basic":
            return "HTTP basic authentication"
        if self.type == "oauth2":
            return "OAuth 2"
        if self.type == "openIdConnect":
            return "OpenID Connect"
        return f"{self.type} authentication"


class Operation(BaseModel):
    """One HTTP operation extracted from the spec."""

    model_config = ConfigDict(extra="forbid")

    operation_id: str
    method: HttpMethod
    path: str
    summary: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    deprecated: bool = False
    scopes: list[str] = Field(default_factory=list)
    security_scheme: SecurityScheme | None = None
    """How the operation authenticates: the first alternative of its
    effective ``security`` requirement (operation-level overrides
    document-level) that the generated server can present, else the first
    one that resolves at all; ``None`` when the operation declares no
    security or names schemes the document does not define."""
    params: list[ParamSpec] = Field(default_factory=list)
    body_encoding: Literal["json", "form"] = "json"
    response_schema: dict[str, Any] | None = None
    base_url: str = ""
    """Effective base URL for this operation (operation/path-level ``servers``
    win over the document's)."""
    doc_base_url: str = ""
    """The document-level base URL, which becomes the plan's ``api.base_url``."""
    unsupported_body: str | None = None
    """Set when the request body uses a media type the generated server cannot
    send (``multipart/form-data``, ``text/plain``, …), when a ``$ref`` in the
    operation's parameters or body cannot be resolved (external or dangling),
    or when the operation is malformed (``parameters`` that is not a list of
    mappings, a ``requestBody.content`` or schema that is a string, …); such
    operations are dropped with this reason rather than emitted as tools that
    can never work."""

    @property
    def text(self) -> str:
        """Everything worth pattern-matching: id, path, summary, description."""
        return f"{self.operation_id} {self.path} {self.summary} {self.description}"

    @property
    def signal_text(self) -> str:
        """The text risk words are matched against: id, path and summary.

        The free-form description is deliberately excluded — prose that merely
        mentions billing or deletion must not reclassify an ordinary write.
        """
        return f"{self.operation_id} {self.path} {self.summary}"

    def param(self, name: str) -> ParamSpec | None:
        """The parameter called *name*, or ``None``."""
        for p in self.params:
            if p.name == name:
                return p
        return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_spec(
    source: str | Path | Mapping[str, Any],
    *,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Load an OpenAPI document from a URL, file path, raw JSON/YAML text, or dict.

    Whatever the source, the result has been through :func:`check_document`
    (node budget, nesting depth, ``info``/``paths`` shape) and
    :func:`scrub_strings` (control characters and lone surrogates removed
    from every string), so nothing downstream has to defend against them.

    Args:
        source: A URL, a file path (``~`` is expanded), an inline JSON/YAML
            document, or an already-parsed mapping.
        cancelled: Polled while a URL is downloading; returning ``True``
            aborts the download (the wizard wires its interrupt here).

    Raises:
        MCPcastError: If the source cannot be read or parsed, if the download
            exceeds ``MCPCAST_FETCH_SECONDS`` or is cancelled, or if the
            document is nested too deeply or expands past
            ``MCPCAST_MAX_SPEC_NODES``.
    """
    if isinstance(source, Mapping):
        return scrub_strings(check_document(dict(source), hint="<mapping>"))

    if isinstance(source, Path):
        return _load_file(source.expanduser())

    raw = source.strip()
    if is_url(raw):
        return _parse_text(_fetch(raw, cancelled=cancelled), hint=public_url(raw))

    if raw.startswith("{") or raw.startswith("openapi:") or raw.startswith("swagger:"):
        return _parse_text(raw, hint="<inline>")

    try:
        path = Path(raw).expanduser()
        exists = path.exists()
    except (OSError, RuntimeError):  # a long inline document that is not a valid file name;
        exists = False  # RuntimeError: ``~user`` with no home directory to resolve
    if exists:
        return _load_file(path)

    raise MCPcastError(
        f"spec not found: {source!r} (expected a file path, URL, or inline document)"
    )


def is_url(source: Any) -> bool:
    """``True`` if *source* is an ``http(s)://`` URL string (the scheme is case-insensitive)."""
    return isinstance(source, str) and source.strip().lower().startswith(("http://", "https://"))


def public_url(url: str) -> str:
    """*url* with its userinfo, query string and fragment removed.

    A spec URL may carry a credential (``https://user:token@host/openapi.json``
    or ``?api_key=…``) so that the *download* is authenticated — httpx sends
    userinfo as HTTP Basic auth. Nothing derived from that URL may keep the
    secret: not the plan's ``base_url``, not the generated ``config.py`` or
    ``README.md``, not a log line. Everything that records, prints or joins a
    spec URL goes through this function first. Strings that are not
    ``http(s)://`` URLs (file paths, inline documents) are returned unchanged.

    Args:
        url: The spec source as the user typed it.

    Returns:
        ``scheme://host[:port]/path`` for a URL; *url* itself otherwise.
    """
    if not is_url(url):
        return url
    parts = urlsplit(url.strip())
    host = parts.netloc.rpartition("@")[2]  # keeps the port and IPv6 brackets
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _fetch(url: str, *, cancelled: Callable[[], bool] | None = None) -> str:
    """Download *url* under three limits: size, wall-clock deadline and cancellation.

    Args:
        url: The URL to fetch. Userinfo in it is sent as Basic auth by httpx
            and never echoed — every error message uses :func:`public_url`.
        cancelled: Polled after every chunk; ``True`` aborts the download.

    Raises:
        MCPcastError: If the body exceeds ``MCPCAST_MAX_SPEC_BYTES``, the
            download takes longer than ``MCPCAST_FETCH_SECONDS``, *cancelled*
            returns ``True``, or the request fails.
    """
    import httpx

    shown = public_url(url)
    limit = _max_spec_bytes()
    budget = _fetch_seconds()
    too_large = MCPcastError(
        f"spec at {shown} is larger than {limit} bytes; raise MCPCAST_MAX_SPEC_BYTES if that "
        "is expected"
    )
    deadline = time.monotonic() + budget
    try:
        with (
            httpx.Client(timeout=30, follow_redirects=True) as client,
            client.stream("GET", url) as resp,
        ):
            resp.raise_for_status()
            declared = resp.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > limit:
                raise too_large
            chunks: list[bytes] = []
            received = 0
            for chunk in resp.iter_bytes():
                if cancelled is not None and cancelled():
                    raise MCPcastError("spec download cancelled")
                if time.monotonic() > deadline:
                    raise MCPcastError(
                        f"downloading the spec from {shown} took longer than {budget:g}s; "
                        "raise MCPCAST_FETCH_SECONDS if the server is that slow"
                    )
                received += len(chunk)
                if received > limit:
                    raise too_large
                chunks.append(chunk)
            encoding = resp.encoding or "utf-8"
    except httpx.HTTPError as exc:
        raise MCPcastError(
            f"could not fetch spec from {shown}: {type(exc).__name__}: {_scrub_urls(str(exc))}"
        ) from exc
    return b"".join(chunks).decode(encoding, errors="replace")


_URL_TOKEN = re.compile(r"https?://[^\s'\"<>]+")


def _scrub_urls(message: str) -> str:
    """*message* with every ``http(s)://`` token reduced to its :func:`public_url`.

    httpx puts the request URL — userinfo and query included — into its
    error messages (``Client error '401 Unauthorized' for url 'http://u:p@…'``),
    and after a redirect that may not be the URL the user typed, so the
    scrub works on whatever URLs the message actually contains.
    """
    return _URL_TOKEN.sub(lambda m: public_url(m.group(0)), message)


def _load_file(path: Path) -> dict[str, Any]:
    try:
        size = path.stat().st_size
        if size > _max_spec_bytes():
            raise MCPcastError(
                f"spec {path} is {size} bytes, larger than MCPCAST_MAX_SPEC_BYTES "
                f"({_max_spec_bytes()})"
            )
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MCPcastError(f"could not read spec {path}: {exc}") from exc
    return _parse_text(text, hint=str(path))


def _parse_text(text: str, *, hint: str) -> dict[str, Any]:
    if len(text) > _max_spec_bytes():
        raise MCPcastError(
            f"{hint}: document is {len(text)} characters, larger than MCPCAST_MAX_SPEC_BYTES "
            f"({_max_spec_bytes()})"
        )
    stripped = text.lstrip()
    try:
        # _Loader composes with SafeConstructor: python/* tags are refused (see
        # _safe_loader and TestYamlLoader::test_loader_is_safe); only the scanner
        # and parser come from libyaml.
        data = (
            json.loads(stripped) if stripped.startswith("{") else yaml.load(stripped, _Loader)  # nosec B506 - SafeConstructor-based loader
        )
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise MCPcastError(f"{hint}: not valid JSON or YAML: {exc}") from exc
    except RecursionError as exc:
        raise MCPcastError(f"{hint}: document is nested too deeply to parse") from exc
    if not isinstance(data, dict):
        raise MCPcastError(f"{hint}: document is not a mapping")
    # Budget first: the scrub copies the tree, which is exactly what an alias
    # bomb is waiting for.
    return scrub_strings(check_document(data, hint=hint))


def _safe_loader() -> type[yaml.SafeLoader]:
    """The safe YAML loader: libyaml's scanner and parser when PyYAML has them.

    ``yaml.safe_load`` is pure Python and holds the GIL for 15–20 s on a
    10 MiB spec, which is what a wizard user waits through after Ctrl+Q.
    ``yaml.CSafeLoader`` is three times faster but composes the node tree in
    C, so a document nested a few thousand levels deep overflows the C stack
    and kills the process — there is no ``RecursionError`` to catch, and a
    spec is untrusted input fetched from a URL. This loader takes libyaml's
    scanner and parser (where the time goes) and keeps PyYAML's Python
    composer and constructor, so nesting hits the interpreter's recursion
    limit exactly as ``SafeLoader`` does and :func:`_parse_text` reports it.
    Without libyaml, ``SafeLoader`` itself.
    """
    try:
        from yaml.cyaml import CParser  # type: ignore[attr-defined]
    except ImportError:  # PyYAML built without libyaml
        return yaml.SafeLoader

    class _CSafeLoader(
        yaml.composer.Composer, CParser, yaml.constructor.SafeConstructor, yaml.resolver.Resolver
    ):  # type: ignore[misc]
        def __init__(self, stream: str) -> None:
            CParser.__init__(self, stream)
            yaml.composer.Composer.__init__(self)
            yaml.constructor.SafeConstructor.__init__(self)
            yaml.resolver.Resolver.__init__(self)

    return _CSafeLoader  # type: ignore[return-value]


_Loader = _safe_loader()


_MAX_DOCUMENT_DEPTH = 256
"""Nesting levels a document may have. Real specs stay well under 100 even
with deeply nested inline schemas (each object level costs two: ``properties``
and the property); the recursive walks that follow (surrogate scrub, ``$ref``
inlining) must never be the thing that fails."""


def check_document(document: dict[str, Any], *, hint: str) -> dict[str, Any]:
    """Refuse whole-document defects before anything walks or copies *document*.

    Checks, in order: the node budget (``MCPCAST_MAX_SPEC_NODES``), the
    nesting depth, and that ``info`` and ``paths`` are mappings when present.
    Anything past this point may assume those hold. :func:`load_spec` runs it
    on every document; code that parses a document itself (the wizard's local
    API detection) calls it, then :func:`scrub_strings`, before using the
    result.

    Args:
        document: The parsed document.
        hint: What to call it in an error (file, URL or ``<inline>``).

    Returns:
        *document* itself.

    Raises:
        MCPcastError: Naming *hint* and the defect.
    """
    budget = _max_document_nodes()
    count, depth = _walk(document, max_nodes=budget, max_depth=_MAX_DOCUMENT_DEPTH)
    if count > budget:
        raise MCPcastError(
            f"{hint}: document expands to more than {budget} nodes (YAML aliases count every "
            "time they are used); raise MCPCAST_MAX_SPEC_NODES if that is expected"
        )
    if depth > _MAX_DOCUMENT_DEPTH:
        raise MCPcastError(
            f"{hint}: document is nested more than {_MAX_DOCUMENT_DEPTH} levels deep"
        )
    for key in ("info", "paths"):
        if key in document and not isinstance(document[key], Mapping):
            raise MCPcastError(f"{hint}: {key!r} must be a mapping, got {_kind(document[key])}")
    return document


def expanded_nodes(document: Any, *, limit: int | None = None) -> int:
    """Count the nodes of *document* as extraction would visit them, stopping past *limit*.

    Every mapping value and list item counts, and a subtree reached through
    several YAML aliases counts once per visit — that is what copying it
    (surrogate scrubbing, ``$ref`` inlining) would cost. The walk is
    iterative, so depth cannot overflow the stack, and it stops as soon as
    the count passes *limit*, so a 1 KB alias bomb costs at most *limit*
    steps and no memory.

    Args:
        document: A parsed JSON/YAML value.
        limit: Stop counting past this many nodes (default
            ``MCPCAST_MAX_SPEC_NODES``, see :data:`MAX_DOCUMENT_NODES`).

    Returns:
        The node count, or a value above *limit* when the document expands
        past it.
    """
    count, _ = _walk(document, max_nodes=_max_document_nodes() if limit is None else limit)
    return count


def _walk(document: Any, *, max_nodes: int, max_depth: int | None = None) -> tuple[int, int]:
    """``(nodes, depth)`` of *document*, stopping once either passes its bound.

    Two parallel stacks (nodes and their depths) keep the walk allocation-free
    per node: the full default budget costs ~0.15 s, so a bomb is refused
    before the user notices.
    """
    count = 1
    deepest = 0
    stack: list[Any] = [document]
    depths: list[int] = [0]
    while stack:
        node = stack.pop()
        depth = depths.pop()
        if depth > deepest:
            deepest = depth
            if max_depth is not None and depth > max_depth:
                return count, deepest
        if isinstance(node, dict):
            children: Any = node.values()
        elif isinstance(node, list):
            children = node
        else:
            continue
        size = len(children)
        count += size
        if count > max_nodes:
            return count, deepest
        stack.extend(children)
        depths.extend([depth + 1] * size)
    return count, deepest


def scrub_strings(document: Any) -> Any:
    """*document* with every string — keys and values, at any depth — passed through
    :func:`~promptise.mcpcast.schema.scrub_text`.

    Spec text reaches the terminal (the ``--review`` tables, the wizard),
    tool descriptions and the README, so control characters other than
    newline and tab are deleted at this boundary: an ``\\x1b[2K`` in a
    summary, path, parameter name or media-type key would otherwise erase
    the review row a human is about to approve. Lone UTF-16 surrogates
    (``"\\ud800"``, which ``json.loads`` and YAML escapes happily produce)
    are replaced so every string can be written as UTF-8. :func:`load_spec`
    applies it to every document it returns; callers that parse a document
    themselves apply it after :func:`check_document`.

    Args:
        document: A parsed JSON/YAML value.

    Returns:
        A cleaned copy (scalars other than strings are returned as they are).
    """
    return _scrub_tree(document)


# ---------------------------------------------------------------------------
# Spec-level metadata
# ---------------------------------------------------------------------------


def spec_title(spec: Mapping[str, Any]) -> str:
    """The spec's ``info.title``, stripped (empty if absent or if ``info`` is not a mapping)."""
    return _info_field(spec, "title")


def spec_description(spec: Mapping[str, Any]) -> str:
    """The spec's ``info.description``, stripped (empty if absent or if ``info`` is not a mapping)."""
    return _info_field(spec, "description")


def _info_field(spec: Mapping[str, Any], key: str) -> str:
    info = spec.get("info")
    if not isinstance(info, Mapping):
        # A malformed ``info`` is refused by load_spec/extract_operations; the
        # getters themselves stay total so a label can always be computed.
        return ""
    return str(info.get(key) or "").strip()


def spec_summary_line(spec: Mapping[str, Any]) -> str:
    """One line for the plan's ``api.description`` (and the server's
    ``instructions``): the title, plus the first line of the description when
    it adds something (``"Bookshelf API: Books, notes and search."``)."""
    title = spec_title(spec)
    desc = spec_description(spec).splitlines()
    first = desc[0].strip() if desc else ""
    if title and first and first.lower() != title.lower():
        return f"{title}: {first}"[:300]
    return (title or first)[:300]


def spec_base_url(
    spec: Mapping[str, Any], override: str | None = None, *, spec_url: str | None = None
) -> str:
    """The API base URL: *override*, else ``servers[0].url``, else Swagger 2 host.

    A relative server URL (``/api/v3``, allowed by OpenAPI and common in
    specs served by the API itself) is resolved against *spec_url* when the
    document was fetched from one; otherwise it is returned as-is and the
    planner asks for ``--base-url``.
    """
    if override:
        return override.rstrip("/")
    first = _first_server(spec.get("servers"))
    if first is not None:
        return _server_url(first, spec_url)
    if "host" in spec:
        schemes = spec.get("schemes")
        scheme = schemes[0] if isinstance(schemes, list) and schemes else "https"
        return f"{scheme}://{spec['host']}{spec.get('basePath') or ''}".rstrip("/")
    if spec_url:
        # No servers block at all: the API lives where the spec was served
        # from — the host only, never the credential the spec was fetched with.
        return urljoin(public_url(spec_url), "/").rstrip("/")
    return ""


def _first_server(servers: Any) -> Mapping[str, Any] | None:
    """The first ``servers[]`` entry that is a mapping, or ``None`` (a malformed block is ignored)."""
    if isinstance(servers, list) and servers and isinstance(servers[0], Mapping):
        return servers[0]
    return None


def _server_url(server: Mapping[str, Any], spec_url: str | None) -> str:
    """One ``servers[]`` entry as an absolute URL, variables substituted.

    Raises:
        MCPcastError: If a server variable has no default (OpenAPI requires
            one), since guessing a host is not an option.
    """
    url = str(server.get("url") or "").strip()
    variables = server.get("variables")
    if isinstance(variables, Mapping):
        for var, cfg in variables.items():
            default = cfg.get("default") if isinstance(cfg, Mapping) else None
            if default is not None:
                url = url.replace("{" + str(var) + "}", str(default))
    if "{" in url:
        raise MCPcastError(
            f"server URL {url!r} has a variable with no default; pass --base-url with the "
            "resolved host"
        )
    if url and not url.startswith(("http://", "https://")) and spec_url:
        url = urljoin(public_url(spec_url), url)
    return url.rstrip("/")


def api_name_from_spec(spec: Mapping[str, Any], source: str | None = None) -> str:
    """A slug for the API, from ``info.title``.

    An untitled spec is named after its source file or URL; an untitled
    inline document (the text itself, or the ``<inline>`` label recorded for
    it) has no source to name it after and becomes ``api`` — the same name
    whether it reaches the wizard, the CLI or :func:`~promptise.mcpcast.mcpcast`.
    """
    title = spec_title(spec)
    # An inline document is not a file to name the API after — neither the
    # text itself nor the ``<inline>`` label the wizard and CLI record for it.
    if not title and source and not source.lstrip().startswith(("{", "openapi:", "swagger:", "<")):
        # public_url: a query string on a spec URL must not end up in the name.
        title = Path(public_url(source)).stem
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    # "Swagger Petstore - OpenAPI 3.0" → "petstore": drop filler words and versions.
    words = [w for w in slug.split("-") if w and w not in _NAME_FILLER and not _VERSION.match(w)]
    slug = "-".join(words) or slug
    return (slug or "api")[:63]


_NAME_FILLER = frozenset(
    {"api", "apis", "openapi", "swagger", "spec", "specification", "rest", "json"}
)
_VERSION = re.compile(r"^v?\d+(\.\d+)*$")


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def extract_operations(
    spec: Mapping[str, Any], *, base_url: str | None = None, spec_url: str | None = None
) -> list[Operation]:
    """Extract every operation from *spec*, in document order.

    Args:
        spec: A parsed OpenAPI 3.x or Swagger 2.x document.
        base_url: Override the base URL declared by the spec.
        spec_url: Where the spec was fetched from, used to resolve a relative
            ``servers[].url``.

    A defect inside one operation (``parameters`` that is not a list of
    mappings, a ``requestBody.content`` that is not a mapping, a schema that
    is a string, …) never fails the whole spec: that operation is returned
    with :attr:`Operation.unsupported_body` naming the defect and the
    planner drops it with that reason, exactly like an unresolvable
    ``$ref``. Scalars that merely have the wrong type are coerced
    (``operationId: 5`` → ``"5"``; ``tags`` that are not a list are ignored).

    Raises:
        MCPcastError: If the document has no ``paths``, or ``info`` or
            ``paths`` is not a mapping — defects of the whole document.
    """
    if spec_url:
        spec_url = public_url(spec_url)
    info = spec.get("info")
    if info is not None and not isinstance(info, Mapping):
        raise MCPcastError(f"spec 'info' must be a mapping, got {_kind(info)}")
    paths = spec.get("paths")
    if paths is not None and not isinstance(paths, Mapping):
        raise MCPcastError(f"spec 'paths' must be a mapping, got {_kind(paths)}")
    if not paths:
        if spec.get("webhooks"):
            raise MCPcastError(
                "spec declares only webhooks (calls the API sends to you); there are no "
                "operations an agent can call"
            )
        raise MCPcastError("spec has no 'paths' — is this an OpenAPI document?")

    base = spec_base_url(spec, base_url, spec_url=spec_url)
    global_security = spec.get("security")
    schemes = _security_schemes(spec)
    ops: list[Operation] = []
    taken: set[str] = set()

    for path, item in paths.items():
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("$ref"), str):
            try:
                item = _deref(item, spec)
            except _UnresolvedRef as exc:
                # Nothing to enumerate: the operations live in a document we do
                # not have, so there is no operation id to record a drop under.
                raise MCPcastError(f"path item {path!r}: {exc.reason}") from exc
        shared_error: str | None = None
        shared_params: list[Mapping[str, Any]] = []
        try:
            shared_params = _parameter_list(item.get("parameters"), spec, where="path parameters")
        except (_UnresolvedRef, _Malformed) as exc:
            shared_error = exc.reason
        path_servers = item.get("servers")
        for method in _METHODS:
            operation = item.get(method)
            if not isinstance(operation, dict):
                continue
            raw_id = operation.get("operationId")
            op_id = _unique(_sanitize_id(str(raw_id) if raw_id else f"{method}_{path}"), taken)
            if shared_error is None:
                params, encoding, unsupported = _collect_params(operation, shared_params, spec)
            else:
                params, encoding, unsupported = [], "json", shared_error
            security = operation.get("security", global_security)
            # Operation- and path-level servers override the document's (unless
            # the caller forced a base URL).
            op_base = base
            if base_url is None:
                for servers in (operation.get("servers"), path_servers):
                    first = _first_server(servers)
                    if first is not None:
                        op_base = _server_url(first, spec_url)
                        break
            ops.append(
                Operation(
                    operation_id=op_id,
                    method=method.upper(),  # type: ignore[arg-type]
                    path=str(path),
                    summary=str(operation.get("summary") or "").strip(),
                    description=str(operation.get("description") or "").strip(),
                    tags=_tags(operation.get("tags")),
                    deprecated=bool(operation.get("deprecated", False)),
                    scopes=_flatten_scopes(security),
                    security_scheme=_effective_scheme(security, schemes),
                    params=params,
                    body_encoding=encoding,
                    response_schema=_success_response_schema(operation, spec),
                    base_url=op_base,
                    doc_base_url=base,
                    unsupported_body=unsupported,
                )
            )
    return ops


def _sanitize_id(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", raw).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned or "operation"


def _tags(raw: Any) -> list[str]:
    """``tags`` as a list of strings; anything that is not a list of scalars is ignored."""
    if not isinstance(raw, list):
        return []
    return [str(t) for t in raw if isinstance(t, (str, int, float)) and not isinstance(t, bool)]


_KINDS = {dict: "mapping", list: "list", str: "string", bool: "boolean", type(None): "null"}


def _kind(value: Any) -> str:
    """The JSON-ish type name of *value* for error messages (``mapping``, ``string``, …)."""
    return _KINDS.get(type(value), type(value).__name__)


class _Malformed(Exception):
    """A structural defect inside one operation: its drop reason, never fatal to the spec."""

    def __init__(self, where: str, expected: str, got: Any) -> None:
        self.reason = f"malformed {where}: expected {expected}, got {_kind(got)}"
        super().__init__(self.reason)


def _mapping(node: Any, where: str) -> Mapping[str, Any]:
    """*node* if it is a mapping, else :class:`_Malformed` naming *where*."""
    if not isinstance(node, Mapping):
        raise _Malformed(where, "a mapping", node)
    return node


def _parameter_list(raw: Any, spec: Mapping[str, Any], *, where: str) -> list[Mapping[str, Any]]:
    """A ``parameters`` block as dereferenced mappings.

    Raises:
        _Malformed: If the block is not a list, or an entry is not a mapping
            once its ``$ref`` (if any) is followed.
        _UnresolvedRef: If an entry's ``$ref`` cannot be followed.
    """
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise _Malformed(where, "a list of mappings", raw)
    return [_mapping(_deref(p, spec), f"{where} entry") for p in raw]


def _unique(candidate: str, taken: set[str]) -> str:
    name, n = candidate, 2
    while name in taken:
        name = f"{candidate}_{n}"
        n += 1
    taken.add(name)
    return name


def _security_schemes(spec: Mapping[str, Any]) -> dict[str, SecurityScheme]:
    """Every resolvable entry of ``components.securitySchemes`` / ``securityDefinitions``.

    An entry that is not a mapping (once a local ``$ref`` is followed), has
    no recognised ``type``, or refers outside the document is left out — it
    cannot say where a credential travels, so the operations that name it
    fall back to the ``Authorization`` header like an unsecured operation.
    """
    components = spec.get("components")
    raw: Any = None
    if isinstance(components, Mapping):
        raw = components.get("securitySchemes")
    if not isinstance(raw, Mapping):
        raw = spec.get("securityDefinitions")  # Swagger 2
    if not isinstance(raw, Mapping):
        return {}
    schemes: dict[str, SecurityScheme] = {}
    for key, entry in raw.items():
        try:
            entry = _deref(entry, spec)
        except _UnresolvedRef:
            continue
        if not isinstance(entry, Mapping):
            continue
        kind = _SCHEME_TYPES.get(str(entry.get("type") or "").strip().lower())
        if kind is None:
            continue
        location: Any = None
        name: str | None = None
        http_scheme: str | None = None
        if kind == "apiKey":
            where = str(entry.get("in") or "").strip().lower()
            location = where if where in ("header", "query", "cookie") else None
            name = str(entry.get("name") or "").strip() or None
        elif kind == "http":
            http_scheme = str(entry.get("scheme") or "").strip().lower() or None
        schemes[str(key)] = SecurityScheme(
            key=str(key), type=kind, location=location, name=name, scheme=http_scheme
        )
    return schemes


def _effective_scheme(
    security: Any, schemes: Mapping[str, SecurityScheme]
) -> SecurityScheme | None:
    """The scheme an operation's ``security`` requirement resolves to.

    A requirement is a list of alternatives (any one satisfies the API), each
    a mapping of scheme name to scopes (all of them together). The generated
    server presents one credential, so the first alternative made of exactly
    one scheme that it can present wins; when none qualifies, the first
    scheme that resolves at all is recorded so the planner can say why the
    operation cannot be served. ``None`` for no security, ``security: []``,
    or names the document does not define.
    """
    if not isinstance(security, list):
        return None
    fallback: SecurityScheme | None = None
    for alternative in security:
        if not isinstance(alternative, Mapping):
            continue
        resolved = [schemes[str(k)] for k in alternative if str(k) in schemes]
        if len(alternative) == 1 and resolved and resolved[0].credential is not None:
            return resolved[0]
        if fallback is None and resolved:
            fallback = resolved[0]
    return fallback


def _flatten_scopes(security: Any) -> list[str]:
    scopes: list[str] = []
    if not isinstance(security, list):
        return scopes
    for requirement in security:
        if not isinstance(requirement, dict):
            continue
        for scheme, values in requirement.items():
            if isinstance(values, list) and values:
                scopes.extend(str(v) for v in values)
            else:
                scopes.append(str(scheme))
    return sorted(set(scopes))


def _collect_params(
    operation: Mapping[str, Any],
    shared: list[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> tuple[list[ParamSpec], Literal["json", "form"], str | None]:
    """``(parameters, body encoding, why the operation cannot be served)``.

    A ``$ref`` that cannot be resolved anywhere in the parameters or the
    request body makes the whole operation unsupported — a body that quietly
    became ``{}`` would emit a tool that posts nothing. A structural defect
    (``parameters: "nope"``, ``requestBody.content: 5``, a schema that is a
    string) is reported the same way, naming the field.
    """
    try:
        return _collect_params_resolved(operation, shared, spec)
    except (_UnresolvedRef, _Malformed) as exc:
        return [], "json", exc.reason


def _collect_params_resolved(
    operation: Mapping[str, Any],
    shared: list[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> tuple[list[ParamSpec], Literal["json", "form"], str | None]:
    params: list[ParamSpec] = []
    seen: set[tuple[str, str]] = set()
    encoding: Literal["json", "form"] = "json"
    unsupported: str | None = None

    own = _parameter_list(operation.get("parameters"), spec, where="parameters")
    # Operation-level parameters override path-level ones with the same (name, in)
    for p in [*own, *shared]:
        name = str(p.get("name") or "")
        loc = str(p.get("in") or "query")
        if not name or (name, loc) in seen:
            continue
        seen.add((name, loc))
        where = f"parameter {name!r} schema"
        if loc == "body":  # Swagger 2 whole-body parameter
            schema = _mapping(_deref(p.get("schema") or {}, spec), where)
            params.extend(_body_params(schema, bool(p.get("required", False)), spec))
            continue
        if loc == "formData":
            if str(p.get("type") or "") == "file":
                # Swagger 2 file upload: the same multipart body OpenAPI 3 declares
                # via requestBody, which the generated server cannot send either.
                unsupported = "unsupported request body media type multipart/form-data"
                continue
            encoding = "form"
            location: str = "body"
        elif loc in ("path", "query", "header", "cookie"):
            location = loc
        else:
            continue
        raw = p.get("schema") or _pick_media(p.get("content"))[1] or _swagger2_param_schema(p)
        schema = dict(_mapping(_deref(raw, spec), where))
        if "example" in p and not ({"example", "examples"} & set(schema)):
            # OpenAPI 3 lets the example sit on the parameter instead of its schema.
            schema["example"] = p["example"]
        params.append(
            ParamSpec(
                name=name,
                location=location,  # type: ignore[arg-type]
                required=bool(p.get("required", False)) or loc == "path",
                json_schema=schema,
                # A description may sit on the parameter or inside its schema
                # (pydantic Field(description=...) lands in the schema).
                description=str(p.get("description") or schema.get("description") or "").strip(),
            )
        )

    body = operation.get("requestBody")
    if body is not None:
        body = _mapping(_deref(body, spec), "requestBody")
        content = _mapping(body.get("content") or {}, "requestBody.content")
        media, schema = _pick_media(content)
        if media is not None:
            params.extend(
                _body_params(
                    _mapping(_deref(schema, spec), f"requestBody.content[{media!r}].schema"),
                    bool(body.get("required")),
                    spec,
                )
            )
            if media == "application/x-www-form-urlencoded":
                encoding = "form"
        elif content:
            unsupported = "unsupported request body media type " + ", ".join(
                sorted(str(m) for m in content)
            )
    return _disambiguate(params), encoding, unsupported


_LOCATION_PRIORITY = {"path": 0, "query": 1, "body": 2, "raw_body": 2}


def _disambiguate(params: list[ParamSpec]) -> list[ParamSpec]:
    """Give exposed parameters unique tool-facing names.

    A spec may legally use one name in several locations (``{id}`` in the
    path and ``id`` in the body).  Path parameters keep their name, then
    query, then body; a later collision is exposed as ``<location>_<name>``
    and still sent under its original wire name.  Header/cookie parameters
    are never exposed, so they do not take part.
    """
    exposed = [p for p in params if p.location in _LOCATION_PRIORITY]
    order = sorted(range(len(exposed)), key=lambda i: (_LOCATION_PRIORITY[exposed[i].location], i))
    taken: set[str] = set()
    for i in order:
        p = exposed[i]
        if p.name in taken:
            base = f"{p.location}_{p.name}"
            candidate, n = base, 2
            while candidate in taken:
                candidate = f"{base}_{n}"
                n += 1
            exposed[i] = p.model_copy(update={"name": candidate, "wire_name": p.name})
        taken.add(exposed[i].name)
    replaced = iter(exposed)
    return [next(replaced) if p.location in _LOCATION_PRIORITY else p for p in params]


def _pick_media(content: Any) -> tuple[str | None, Any]:
    """``(media type, its raw schema)`` for the first JSON/form entry of a ``content`` block.

    The schema is returned as found — the caller checks it is a mapping once
    it has followed any ``$ref``. A *content* that is not a mapping yields
    ``(None, {})``.
    """
    if not isinstance(content, Mapping):
        return None, {}
    for preferred in ("application/json", "application/x-www-form-urlencoded"):
        if preferred in content and isinstance(content[preferred], Mapping):
            return preferred, content[preferred].get("schema") or {}
    for media, entry in content.items():
        media = str(media)
        if isinstance(entry, Mapping) and (media.endswith("+json") or "json" in media):
            return media, entry.get("schema") or {}
    return None, {}


def _body_params(
    schema: Mapping[str, Any], body_required: bool, spec: Mapping[str, Any]
) -> list[ParamSpec]:
    properties = schema.get("properties")
    if (
        isinstance(properties, Mapping)
        and not properties
        and not any(k in schema for k in ("oneOf", "anyOf", "allOf", "items"))
        and schema.get("additionalProperties") in (None, False)
    ):
        return []  # an object that cannot carry data (Stripe declares these on GETs)
    if isinstance(properties, Mapping) and properties:
        declared = schema.get("required")
        required = {str(r) for r in declared} if isinstance(declared, list) else set()
        resolved = {
            str(name): _mapping(_deref(prop or {}, spec), f"body property {name!r} schema")
            for name, prop in properties.items()
        }
        return [
            ParamSpec(
                name=name,
                location="body",
                required=name in required,
                json_schema=_request_shape(prop),
                description=str(prop.get("description") or "").strip(),
            )
            for name, prop in resolved.items()
            # The server sets a readOnly property (an id, a timestamp); a request
            # never carries it, and its ``required`` only binds responses.
            if prop.get("readOnly") is not True
        ]
    if schema:
        return [
            ParamSpec(
                name="body",
                location="raw_body",
                required=body_required,
                json_schema=_request_shape(schema),
                description=str(schema.get("description") or "Request body").strip(),
            )
        ]
    return []


def _request_shape(schema: Mapping[str, Any]) -> dict[str, Any]:
    """*schema* as a request sends it: nested ``readOnly`` properties left out.

    Data (``example``, ``default``, ``enum``…) is copied as written.
    """
    out: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _DATA_KEYS or not isinstance(value, (dict, list)):
            out[key] = value
        elif key in _NAMED_SCHEMA_MAPS and isinstance(value, dict):
            out[key] = {
                n: _request_shape(s) if isinstance(s, Mapping) else s
                for n, s in value.items()
                if not (
                    key == "properties" and isinstance(s, Mapping) and s.get("readOnly") is True
                )
            }
        elif isinstance(value, list):
            out[key] = [_request_shape(v) if isinstance(v, Mapping) else v for v in value]
        else:
            out[key] = _request_shape(value)
    properties = schema.get("properties")
    if isinstance(properties, Mapping) and isinstance(out.get("required"), list):
        dropped = {
            n for n, s in properties.items() if isinstance(s, Mapping) and s.get("readOnly") is True
        }
        out["required"] = [r for r in out["required"] if r not in dropped]
    return out


_SWAGGER2_SCHEMA_KEYS = (
    "type",
    "format",
    "enum",
    "default",
    "items",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
    "x-nullable",
)
"""The JSON Schema keywords a Swagger 2 parameter carries on itself (no ``schema``)."""


def _swagger2_param_schema(param: Mapping[str, Any]) -> dict[str, Any]:
    schema = {key: param[key] for key in _SWAGGER2_SCHEMA_KEYS if key in param}
    return schema or {"type": "string"}


def _success_response_schema(
    operation: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any] | None:
    responses = operation.get("responses")
    if not isinstance(responses, Mapping):
        return None
    try:
        for code in sorted(str(c) for c in responses):
            if not (code.startswith("2") or code == "default"):
                continue
            raw = responses[code] if code in responses else responses.get(int(code))
            entry = _deref(raw, spec)
            if not isinstance(entry, Mapping):
                continue
            if "schema" in entry:  # Swagger 2
                schema = _deref(entry["schema"], spec)
                if isinstance(schema, Mapping):
                    return dict(schema)
                continue
            _, schema = _pick_media(entry.get("content"))
            if schema:
                schema = _deref(schema, spec)
                if isinstance(schema, Mapping):
                    return dict(schema)
    except (_UnresolvedRef, ValueError):
        # The response shape only feeds evaluation mocks; a tool works without
        # it (ValueError: a code like "2XX" stored under a non-string key).
        return None
    return None


# ---------------------------------------------------------------------------
# $ref resolution
# ---------------------------------------------------------------------------


class _UnresolvedRef(Exception):
    """A ``$ref`` the parser cannot follow: external, or pointing nowhere."""

    def __init__(self, ref: Any, why: str) -> None:
        self.reason = f"{why} $ref {ref!r} is not supported; bundle the spec into one document"
        super().__init__(self.reason)


def _deref(node: Any, spec: Mapping[str, Any], _hops: int = 0) -> Any:
    """Inline local ``$ref`` pointers recursively.

    Only *following a reference* counts towards the depth limit — plain
    nesting never does, and scalars and lists are returned unchanged — so a
    deeply nested schema stays intact and a recursive one (``Node.next →
    Node``) bottoms out at a bare ``{"type": "object"}`` after
    ``_MAX_REF_DEPTH`` hops.

    Only a ``$ref`` whose value is a string is a reference. A *property*
    called ``$ref`` or ``$schema`` (schema registries, JSON-Schema-about-
    JSON-Schema APIs) is a property: the maps that hold named schemas —
    ``properties``, ``patternProperties``, ``definitions``, ``$defs`` — are
    walked value by value without ever reading the map itself as a schema.
    ``example``, ``examples``, ``default``, ``enum`` and ``const`` hold data,
    not schemas, and are copied through untouched, so an example object with
    a ``$ref`` key stays exactly as the spec wrote it.

    Raises:
        _UnresolvedRef: For an external reference (``other.yaml#/X``) or a
            local one that points nowhere — never silently ``{}``.
    """
    if isinstance(node, list):
        return [_deref(item, spec, _hops) for item in node]
    if not isinstance(node, dict):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str):
        if _hops >= _MAX_REF_DEPTH:
            return {"type": "object"}
        target = _resolve_pointer(ref, spec)
        siblings = {k: v for k, v in node.items() if k != "$ref"}
        merged = {**target, **siblings} if isinstance(target, dict) else dict(siblings)
        return _deref(merged, spec, _hops + 1)
    resolved: dict[str, Any] = {}
    for key, value in node.items():
        if key in _DATA_KEYS:
            resolved[key] = value
        elif key in _NAMED_SCHEMA_MAPS and isinstance(value, dict):
            resolved[key] = {name: _deref(schema, spec, _hops) for name, schema in value.items()}
        else:
            resolved[key] = _deref(value, spec, _hops)
    return resolved


_NAMED_SCHEMA_MAPS = frozenset({"properties", "patternProperties", "definitions", "$defs"})
"""Keys whose value maps *names* to schemas: the names are data (``$ref`` included)."""
_DATA_KEYS = frozenset({"example", "examples", "default", "enum", "const"})
"""Keys whose value is data, never a schema: copied through without dereferencing."""


def _resolve_pointer(ref: str, spec: Mapping[str, Any]) -> Any:
    if not ref.startswith("#/"):
        raise _UnresolvedRef(ref, "external")
    node: Any = spec
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            raise _UnresolvedRef(ref, "dangling")
    return node
