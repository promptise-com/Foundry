"""Decorator internals for @server.tool(), @server.resource(), @server.prompt().

Introspects function signatures at registration time:
- Extracts parameter names, types, defaults
- Detects ``Depends()`` markers for dependency injection
- Detects ``RequestContext`` typed parameters for auto-injection
- Builds Pydantic models for input validation
- Uses docstrings as descriptions when none provided: the summary paragraph
  becomes the tool/resource/prompt description, and the ``Args:`` section
  (Google style, or Sphinx ``:param name:`` fields) describes parameters
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any, get_origin, get_type_hints

from ._context import _wants_request_context
from ._types import PromptDef, ResourceDef, ToolDef
from ._validation import build_input_model


class _DependsMarker:
    """Sentinel that marks a parameter as dependency-injected.

    This is the *internal* type.  The public ``Depends`` class is in
    ``_di.py`` and creates instances of this marker.
    """

    def __init__(self, dependency: Callable[..., Any], *, use_cache: bool = True) -> None:
        self.dependency = dependency
        self.use_cache = use_cache


# Google-style section headers.  A line consisting of one of these followed
# by ``:`` ends the summary paragraph; the parameter sections hold the
# per-parameter descriptions.
_PARAM_SECTIONS = frozenset(
    {
        "args",
        "arguments",
        "parameters",
        "params",
        "keyword args",
        "keyword arguments",
        "other parameters",
    }
)
_SECTIONS = _PARAM_SECTIONS | {
    "returns",
    "return",
    "yields",
    "yield",
    "raises",
    "exceptions",
    "example",
    "examples",
    "note",
    "notes",
    "warning",
    "warnings",
    "see also",
    "attributes",
    "todo",
}

# ``name: text`` or ``name (type): text`` inside an Args section.
_GOOGLE_PARAM = re.compile(r"^\*{0,2}(\w+)\s*(?:\([^)]*\))?\s*:(.*)$")
# ``:param name: text`` or ``:param type name: text``.
_SPHINX_PARAM = re.compile(r"^:param\s+(?:[^:]*\s)?\*{0,2}(\w+)\s*:(.*)$")


def _section_header(line: str) -> str | None:
    """Return the lower-cased section name if *line* is a Google-style header."""
    stripped = line.strip()
    if not stripped.endswith(":"):
        return None
    name = stripped[:-1].strip().lower()
    return name if name in _SECTIONS else None


def _docstring_summary(doc: str) -> str:
    """Return the summary paragraph of an ``inspect.getdoc()``-cleaned docstring.

    The summary is every line up to the first blank line, section header
    (``Args:``, ``Returns:``, ...), or Sphinx field (``:param x:``), joined
    with single spaces so a summary wrapped over several lines reads as one
    sentence.
    """
    parts: list[str] = []
    for line in doc.splitlines():
        stripped = line.strip()
        if not stripped or _section_header(line) or stripped.startswith(":"):
            break
        parts.append(stripped)
    return " ".join(parts)


def _get_description(func: Callable[..., Any], explicit: str | None) -> str:
    """Return *explicit*, else the docstring's summary paragraph, else the function name.

    Only the summary paragraph is used: later paragraphs of a Python
    docstring are usually written for maintainers, and the ``Args:`` section
    goes into the parameter schema instead.  Pass ``description=`` to send
    more text.
    """
    if explicit:
        return explicit
    doc = inspect.getdoc(func)
    if doc:
        summary = _docstring_summary(doc)
        if summary:
            return summary
    return func.__name__


def _parse_param_docs(docstring: str) -> dict[str, str]:
    """Extract parameter descriptions from a docstring.

    Understands the Google style (an ``Args:`` / ``Arguments:`` /
    ``Parameters:`` section with ``name: text`` or ``name (type): text``
    entries) and Sphinx ``:param name:`` fields.  Continuation lines
    indented under an entry are joined with single spaces.  ``*args`` and
    ``**kwargs`` entries are keyed without their stars.
    """
    docs: dict[str, str] = {}
    current: str | None = None  # parameter whose description is being read
    current_indent = 0  # indentation of that parameter's entry line
    section_indent: int | None = None  # indentation of the open Args header
    entry_indent: int | None = None  # indentation of entries in that section

    for line in docstring.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip())

        header = _section_header(line)
        if header is not None:
            section_indent = indent if header in _PARAM_SECTIONS else None
            entry_indent = None
            current = None
            continue

        sphinx = _SPHINX_PARAM.match(stripped)
        if sphinx:
            current, current_indent = sphinx.group(1), indent
            docs[current] = sphinx.group(2).strip()
            section_indent = None
            continue
        if stripped.startswith(":"):
            # Another Sphinx field (``:returns:``, ``:type x:``, ...)
            current = None
            continue

        if section_indent is not None and indent <= section_indent:
            # Dedented text closes the Args section.
            section_indent = None
            current = None

        if section_indent is not None:
            if entry_indent is None:
                entry_indent = indent
            if indent <= entry_indent:
                google = _GOOGLE_PARAM.match(stripped)
                if google:
                    current, current_indent = google.group(1), indent
                    docs[current] = google.group(2).strip()
                else:
                    current = None
                continue

        if current is not None and indent > current_indent:
            docs[current] = f"{docs[current]} {stripped}".strip()
        else:
            current = None

    return {name: text for name, text in docs.items() if text}


def _excluded_params(func: Callable[..., Any]) -> set[str]:
    """Identify parameters that should NOT appear in the input schema.

    Excluded:
    - ``self``
    - Parameters annotated as ``RequestContext`` (including
      ``Optional[RequestContext]`` / ``RequestContext | None``, which is also
      what Python 3.10 produces for ``ctx: RequestContext = None``)
    - Parameters whose default is a ``_DependsMarker``
    """
    excluded: set[str] = set()
    sig = inspect.signature(func)
    hints = {}
    try:
        hints = get_type_hints(func)
    except Exception:
        pass

    for name, param in sig.parameters.items():
        if name == "self":
            excluded.add(name)
            continue
        # Check type annotation — must use the SAME recognition as
        # inject_context, else an Optional[RequestContext] param leaks into the
        # input schema, gets validated to None, and injection then skips it.
        ann = hints.get(name, param.annotation)
        if _wants_request_context(ann):
            excluded.add(name)
            continue
        # Check if default is a Depends() marker
        if isinstance(param.default, _DependsMarker):
            excluded.add(name)
            continue

    return excluded


def build_tool_def(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    auth: bool = False,
    rate_limit: str | None = None,
    timeout: float | None = None,
    guards: list[Any] | None = None,
    roles: list[str] | None = None,
    annotations: Any | None = None,
    max_concurrent: int | None = None,
    requires_approval: bool = False,
    cache: bool = True,
) -> ToolDef:
    """Build a ``ToolDef`` from a decorated function."""
    tool_name = name or func.__name__
    tool_desc = _get_description(func, description)
    excluded = _excluded_params(func)

    # Fail fast on a malformed rate-limit spec: a typo like "100/mn" must
    # error at registration, not silently never limit at request time.
    if rate_limit is not None:
        from ._rate_limit import parse_rate_limit

        parse_rate_limit(rate_limit)

    _, schema = build_input_model(
        func,
        exclude=excluded,
        param_docs=_parse_param_docs(inspect.getdoc(func) or ""),
    )

    return ToolDef(
        name=tool_name,
        description=tool_desc,
        handler=func,
        input_schema=schema,
        tags=tags or [],
        auth=auth,
        rate_limit=rate_limit,
        timeout=timeout,
        guards=guards or [],
        roles=roles or [],
        annotations=annotations,
        max_concurrent=max_concurrent,
        requires_approval=requires_approval,
        cache=cache,
    )


def _infer_mime_type(func: Callable[..., Any]) -> str:
    """Pick a resource MIME type from the handler's return annotation.

    ``bytes`` → ``application/octet-stream``; ``dict`` / ``list`` (bare or
    parameterised) or a Pydantic model → ``application/json``; anything
    else → ``text/plain``.
    """
    try:
        ret = get_type_hints(func).get("return")
    except Exception:
        ret = None
    if ret is None:
        return "text/plain"
    target = get_origin(ret) or ret
    if isinstance(target, type):
        if issubclass(target, (bytes, bytearray)):
            return "application/octet-stream"
        if issubclass(target, (dict, list)):
            return "application/json"
        from pydantic import BaseModel

        if issubclass(target, BaseModel):
            return "application/json"
    return "text/plain"


def _guards_for(guards: list[Any] | None, roles: list[str] | None) -> list[Any]:
    """Return *guards* plus a ``HasRole`` guard for the ``roles=`` shorthand."""
    all_guards = list(guards or [])
    if roles:
        from ._guards import HasRole

        all_guards.append(HasRole(*roles))
    return all_guards


def _template_params(uri_template: str) -> list[str]:
    from ._registry import _PLACEHOLDER

    return [m.group(2) for m in _PLACEHOLDER.finditer(uri_template)]


def build_resource_def(
    func: Callable[..., Any],
    *,
    uri: str,
    name: str | None = None,
    description: str | None = None,
    mime_type: str | None = None,
    is_template: bool = False,
    tags: list[str] | None = None,
    auth: bool = False,
    rate_limit: str | None = None,
    timeout: float | None = None,
    guards: list[Any] | None = None,
    roles: list[str] | None = None,
) -> ResourceDef:
    """Build a ``ResourceDef`` from a decorated function.

    ``roles=`` adds a ``HasRole`` guard and forces ``auth=True`` (roles are
    only known after authentication).  ``mime_type=None`` infers the type
    from the return annotation (see :func:`_infer_mime_type`).

    For a template, every ``{placeholder}`` must name a handler parameter,
    and the parameters are coerced to their type hints on each read.
    """
    res_name = name or func.__name__
    res_desc = _get_description(func, description)
    if rate_limit is not None:
        from ._rate_limit import parse_rate_limit

        parse_rate_limit(rate_limit)

    input_model = None
    if is_template:
        excluded = _excluded_params(func)
        params = inspect.signature(func).parameters
        accepts_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        missing = [p for p in _template_params(uri) if p not in params or p in excluded]
        if missing and not accepts_kwargs:
            raise ValueError(
                f"Resource template {uri!r}: handler {func.__name__}() has no "
                f"parameter for {', '.join('{' + m + '}' for m in missing)}"
            )
        if not accepts_kwargs:
            input_model, _ = build_input_model(func, exclude=excluded)

    return ResourceDef(
        uri=uri,
        name=res_name,
        description=res_desc,
        handler=func,
        mime_type=mime_type or _infer_mime_type(func),
        is_template=is_template,
        tags=list(tags or []),
        auth=auth or bool(roles),
        rate_limit=rate_limit,
        timeout=timeout,
        guards=_guards_for(guards, roles),
        roles=list(roles or []),
        input_model=input_model,
    )


def build_prompt_def(
    func: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
    tags: list[str] | None = None,
    auth: bool = False,
    rate_limit: str | None = None,
    timeout: float | None = None,
    guards: list[Any] | None = None,
    roles: list[str] | None = None,
) -> PromptDef:
    """Build a ``PromptDef`` from a decorated function.

    MCP sends every prompt argument as a string; the returned definition
    carries a Pydantic model that coerces them to the handler's type hints
    (``"3"`` → ``3`` for an ``int`` parameter, JSON text for a ``list``).
    """
    prompt_name = name or func.__name__
    prompt_desc = _get_description(func, description)
    if rate_limit is not None:
        from ._rate_limit import parse_rate_limit

        parse_rate_limit(rate_limit)

    # Build argument list from signature (for MCP PromptArgument)
    sig = inspect.signature(func)
    excluded = _excluded_params(func)
    # Per-param descriptions from the docstring (Args: section)
    param_docs = _parse_param_docs(inspect.getdoc(func) or "")
    arguments: list[dict[str, Any]] = []
    for param_name, param in sig.parameters.items():
        if param_name in excluded or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        arg: dict[str, Any] = {"name": param_name}
        arg["description"] = param_docs.get(param_name) or param_name
        arg["required"] = param.default is inspect.Parameter.empty
        arguments.append(arg)

    has_var_args = any(
        p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        for p in sig.parameters.values()
    )
    input_model = None if has_var_args else build_input_model(func, exclude=excluded)[0]

    return PromptDef(
        name=prompt_name,
        description=prompt_desc,
        handler=func,
        arguments=arguments,
        tags=list(tags or []),
        auth=auth or bool(roles),
        rate_limit=rate_limit,
        timeout=timeout,
        guards=_guards_for(guards, roles),
        roles=list(roles or []),
        input_model=input_model,
    )


def _extract_param_doc(docstring: str, param_name: str) -> str | None:
    """Extract one parameter's description from a docstring.

    See :func:`_parse_param_docs` for the supported formats.
    """
    return _parse_param_docs(docstring).get(param_name)
