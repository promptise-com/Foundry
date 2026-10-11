"""MCP tool discovery and conversion to LangChain tools.

The key function :func:`_jsonschema_to_pydantic` recursively converts a
JSON Schema (as emitted by MCP servers) into a Pydantic model so that
LLMs see fully-typed, described parameters — including nested objects,
arrays of objects, ``anyOf``/``oneOf`` unions, enums, and ``$ref``/``$defs``.

This is critical for tool-calling accuracy: without proper nested models
the LLM only sees ``dict`` and has to guess the structure, leading to
many retries.
"""

from __future__ import annotations

import itertools
import json
import logging
import re
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional, Union, cast, get_args, get_origin

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model
from pydantic_core import to_jsonable_python

logger = logging.getLogger("promptise.tools")

# Callback types for tracing tool calls
OnBefore = Callable[[str, dict[str, Any]], None]
OnAfter = Callable[[str, Any], None]
OnError = Callable[[str, Exception], None]


@dataclass(frozen=True)
class ToolInfo:
    """Human-friendly metadata for a discovered MCP tool."""

    server_guess: str
    name: str
    description: str
    input_schema: dict[str, Any]


class MCPClientError(RuntimeError):
    """Raised when communicating with the MCP client fails."""


# ------------------------------------------------------------------
# JSON Schema -> Pydantic model (recursive)
# ------------------------------------------------------------------

_MODEL_COUNTER = itertools.count(1)


def _safe_model_name(raw: str) -> str:
    """Sanitise a string into a valid Python class name."""
    name = re.sub(r"[^0-9a-zA-Z_]", "_", raw).strip("_")
    return name or "Model"


def _unique_name(base: str) -> str:
    """Generate a unique model name to avoid Pydantic collisions."""
    return f"{_safe_model_name(base)}_{next(_MODEL_COUNTER)}"


def _resolve_refs(schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Inline a single ``$ref`` using the top-level ``$defs``."""
    if "$ref" in schema:
        ref = schema["$ref"]
        parts = ref.lstrip("#/").split("/")
        node: Any = {"$defs": defs}
        for p in parts:
            if isinstance(node, dict) and p in node:
                node = node[p]
            else:
                return schema  # unresolvable ref — return as-is
        if not isinstance(node, dict):
            return schema
        # Merge sibling keys (e.g. description alongside $ref)
        merged = dict(node)
        for k, v in schema.items():
            if k != "$ref":
                merged[k] = v
        return merged
    return schema


def _schema_to_annotation(
    prop: dict[str, Any],
    defs: dict[str, Any],
    name_hint: str,
) -> type[Any]:
    """Convert a single JSON Schema property to a Python type annotation.

    Recursively builds Pydantic models for nested ``object`` types and
    ``array`` types whose ``items`` are objects.
    """
    # Resolve $ref first
    prop = _resolve_refs(prop, defs)

    # Handle anyOf / oneOf (e.g. Optional types from Pydantic)
    for union_key in ("anyOf", "oneOf"):
        if union_key in prop:
            variants = prop[union_key]
            non_null = [v for v in variants if v.get("type") != "null"]
            has_null = len(non_null) < len(variants)
            if len(non_null) == 1:
                inner = _schema_to_annotation(non_null[0], defs, name_hint)
                return Optional[inner] if has_null else inner  # type: ignore[return-value]
            # Multiple non-null variants
            types = tuple(
                _schema_to_annotation(v, defs, f"{name_hint}_{i}") for i, v in enumerate(non_null)
            )
            if has_null:
                # Union[str, int, None]: Optional[...] takes one type, not a tuple.
                return Union[types + (type(None),)]  # type: ignore[return-value]
            return Union[types]  # type: ignore[return-value]

    # Handle allOf (merge all schemas)
    if "allOf" in prop:
        merged: dict[str, Any] = {}
        for sub in prop["allOf"]:
            resolved = _resolve_refs(sub, defs)
            for k, v in resolved.items():
                if k == "properties" and "properties" in merged:
                    merged["properties"] = {**merged["properties"], **v}
                elif k == "required" and "required" in merged:
                    merged["required"] = list(set(merged["required"]) | set(v))
                else:
                    merged[k] = v
        for k, v in prop.items():
            if k != "allOf" and k not in merged:
                merged[k] = v
        return _schema_to_annotation(merged, defs, name_hint)

    t = prop.get("type")

    # A list of types (JSON Schema, OpenAPI 3.1): ``["string", "null"]`` is
    # Optional[str], ``["string", "integer"]`` is Union[str, int].
    if isinstance(t, list):
        members = [m for m in t if m != "null"]
        if not members:
            return Any  # type: ignore[return-value]
        types = tuple(
            _schema_to_annotation({**prop, "type": m}, defs, f"{name_hint}_{i}")
            for i, m in enumerate(members)
        )
        member: Any = types[0] if len(types) == 1 else Union[types]
        return Optional[member] if "null" in t else member  # type: ignore[return-value]

    # OpenAPI 3.0's ``nullable: true``
    if prop.get("nullable") is True:
        rest = {k: v for k, v in prop.items() if k != "nullable"}
        return Optional[_schema_to_annotation(rest, defs, name_hint)]  # type: ignore[return-value]

    # Nested object with properties -> build a Pydantic model
    if t == "object" and prop.get("properties"):
        return _jsonschema_to_pydantic(prop, model_name=name_hint, _defs=defs)

    extras = _displayed_keywords(prop)
    annotation: Any

    if t == "array":
        # Items keep their own shape: models, enums, unions, constraints.
        items = prop.get("items")
        if isinstance(items, dict) and items:
            annotation = list[_schema_to_annotation(items, defs, f"{name_hint}_Item")]  # type: ignore[misc]
        else:
            annotation = list
    elif _literal_values(prop.get("enum")):
        values = prop["enum"]
        literal: Any = Literal[tuple(v for v in values if v is not None)]  # type: ignore[valid-type]
        annotation = Optional[literal] if None in values else literal
    elif "const" in prop and _literal_values([prop["const"]]):
        annotation = Literal[prop["const"]]
    else:
        annotation = _primitive_type(t)
        enum = prop.get("enum")
        if isinstance(enum, list) and enum:
            extras["enum"] = enum  # values a Literal cannot hold (floats, objects)

    if extras:
        # Shown to the model in the tool's schema; the server enforces them.
        return Annotated[annotation, Field(json_schema_extra=extras)]  # type: ignore[return-value]
    return annotation  # type: ignore[no-any-return]


_DISPLAYED_KEYWORDS = (
    "format",
    "pattern",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
)
"""Constraints a tool's parameter schema carries over to the model the agent sees.

They are shown, not enforced: the MCP server validates its own arguments, and
a pattern written for JavaScript's regex engine need not compile in Python's."""


def _displayed_keywords(prop: dict[str, Any]) -> dict[str, Any]:
    return {k: prop[k] for k in _DISPLAYED_KEYWORDS if k in prop}


def _literal_values(values: Any) -> bool:
    """Whether *values* (``null`` aside) can be a ``Literal``: strings and integers only."""
    if not isinstance(values, list):
        return False
    real = [v for v in values if v is not None]
    return bool(real) and all(
        isinstance(v, str) or (isinstance(v, int) and not isinstance(v, bool)) for v in real
    )


def _primitive_type(t: Any) -> type[Any]:
    """Map a JSON Schema type string to a Python primitive."""
    mapping: dict[str | None, type[Any]] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
        None: Any,
    }
    return mapping.get(t, Any) if t is None or isinstance(t, str) else Any


def _jsonschema_to_pydantic(
    schema: dict[str, Any],
    *,
    model_name: str = "Args",
    _defs: dict[str, Any] | None = None,
    strip_descriptions: bool = False,
) -> type[BaseModel]:
    """Recursively convert a JSON Schema to a Pydantic model.

    Handles nested objects, arrays-of-objects, ``$ref``/``$defs``,
    ``anyOf``/``oneOf``/``allOf``, enums, and all primitive types.

    This ensures that LLMs see fully-typed, fully-described tool
    parameters — including nested structures — so they can generate
    correct tool calls on the first attempt.

    Args:
        schema: JSON Schema dict (as returned by MCP ``list_tools``).
        model_name: Name for the generated Pydantic model class.
        _defs: Top-level ``$defs`` for recursive ``$ref`` resolution.
        strip_descriptions: When ``True``, omit ``description`` from
            ``Field()`` metadata.  Used by the tool optimization system
            to reduce token cost.

    Returns:
        A dynamically-created Pydantic ``BaseModel`` subclass.
    """
    schema = schema or {}

    # Extract $defs from schema (top-level call) or use passed-in defs
    defs = _defs if _defs is not None else schema.get("$defs", {})

    # Resolve top-level $ref if present
    schema = _resolve_refs(schema, defs)

    props = schema.get("properties", {}) or {}
    required = set(schema.get("required", []) or [])

    if not props:
        # A tool without parameters: an empty object schema. Free-form
        # objects (``additionalProperties`` set) pass their keys through.
        extra = schema.get("additionalProperties")
        free_form = extra is True or isinstance(extra, dict)
        model = create_model(
            _unique_name(model_name),
            __config__=ConfigDict(extra="allow") if free_form else None,
        )
        return cast(type[BaseModel], model)

    fields: dict[str, Any] = {}

    for prop_name, prop_schema in props.items():
        prop_schema = prop_schema or {}
        prop_schema = _resolve_refs(prop_schema, defs)

        desc = prop_schema.get("description") if not strip_descriptions else None
        default = prop_schema.get("default")
        is_required = prop_name in required

        annotation = _schema_to_annotation(
            prop_schema,
            defs,
            _safe_model_name(f"{model_name}_{prop_name}"),
        )

        if is_required:
            field = Field(..., description=desc)
        elif default is not None:
            field = Field(default=default, description=desc)
        else:
            field = Field(default=None, description=desc)

        fields[prop_name] = (annotation, field)

    safe_name = _unique_name(model_name)
    model = create_model(safe_name, **cast(dict[str, Any], fields))
    return cast(type[BaseModel], model)


# ------------------------------------------------------------------
# Tool arguments: what the model gave, as plain JSON
# ------------------------------------------------------------------
#
# LangChain's ``BaseTool._parse_input`` validates a call against the
# ``args_schema`` and hands the tool every field with a default (``None`` for
# an optional parameter the model left out) and nested objects as instances
# of the generated models.  Sent to an MCP server as-is, the defaults become
# ``null`` arguments a validating server rejects, and reviewers, traces and
# events see model reprs instead of data.  The helpers below keep only the
# arguments the model gave, as plain dicts and lists.


class ToolArgumentError(ToolException):
    """A tool call's arguments do not match the tool's parameter schema.

    Raised before the call is sent, so nothing ran.  The message lists each
    problem by argument path and tells the model to correct the call; the
    agent loop shows it to the model as the tool's result, and callbacks get
    ``on_tool_error``.

    Attributes:
        tool_name: The tool that was called.
        code: Always ``"INVALID_ARGUMENTS"``.
        message: The model-facing message.
        text: Same as ``message`` (what the agent loop shows the model).
        errors: Pydantic's error list (``ValidationError.errors()``).
    """

    code = "INVALID_ARGUMENTS"

    def __init__(
        self, tool_name: str, message: str, *, errors: list[dict[str, Any]] | None = None
    ) -> None:
        self.tool_name = tool_name
        self.message = message
        self.text = message
        self.errors = errors or []
        super().__init__(message)

    @classmethod
    def from_validation_error(cls, tool_name: str, exc: ValidationError) -> ToolArgumentError:
        errors = cast(list[dict[str, Any]], exc.errors(include_url=False))
        lines = []
        for err in errors:
            path = ".".join(str(part) for part in err.get("loc", ())) or "(arguments)"
            line = f"- {path}: {err.get('msg', 'invalid value')}"
            if err.get("type") != "missing" and "input" in err:
                line += f" (got {_short_repr(err['input'])})"
            lines.append(line)
        message = (
            f"Invalid arguments for tool '{tool_name}':\n"
            + "\n".join(lines)
            + "\nCorrect the arguments and call the tool again."
        )
        return cls(tool_name, message, errors=errors)


def _short_repr(value: Any, limit: int = 80) -> str:
    """*value* as short JSON for an error message."""
    try:
        text = json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _is_model_class(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def _accepts_none(annotation: Any) -> bool:
    """Whether a field typed *annotation* takes ``None`` (``Optional``, ``Any``...)."""
    if annotation is None or annotation is type(None) or annotation is Any:
        return True
    origin = get_origin(annotation)
    if origin is Annotated:
        return _accepts_none(get_args(annotation)[0])
    if origin is Union or origin is types.UnionType:
        return any(_accepts_none(arg) for arg in get_args(annotation))
    if origin is Literal:
        return None in get_args(annotation)
    return False


def _strip_annotation(annotation: Any) -> Any:
    """*annotation* without ``Annotated[...]`` and ``Optional[...]`` around it."""
    while True:
        origin = get_origin(annotation)
        if origin is Annotated:
            annotation = get_args(annotation)[0]
            continue
        if origin is Union or origin is types.UnionType:
            members = [a for a in get_args(annotation) if a is not type(None)]
            if len(members) == 1:
                annotation = members[0]
                continue
        return annotation


def _drop_nulls_in(annotation: Any, value: Any, tool_name: str, path: str) -> Any:
    """Apply :func:`drop_disallowed_nulls` inside a nested object or list of objects."""
    inner = _strip_annotation(annotation)
    if _is_model_class(inner) and isinstance(value, Mapping):
        return drop_disallowed_nulls(inner, value, tool_name=tool_name, _path=path)
    if get_origin(inner) is list and isinstance(value, list):
        args = get_args(inner)
        if args:
            return [
                _drop_nulls_in(args[0], item, tool_name, f"{path}[{i}]")
                for i, item in enumerate(value)
            ]
    return value


def drop_disallowed_nulls(
    args_schema: type[BaseModel],
    arguments: Mapping[str, Any],
    *,
    tool_name: str = "",
    _path: str = "",
) -> dict[str, Any]:
    """*arguments* without explicit ``null`` for optional fields that cannot be null.

    A model often writes ``null`` for an optional parameter it means to leave
    out.  When the parameter's schema does not admit ``null``, the ``null``
    is dropped, so the call goes out as if the model had not given it (logged
    at debug level).  A ``null`` for a nullable field is kept, and so is one
    for a required field, which then fails validation and is reported to the
    model.  Nested objects and lists of objects are cleaned the same way.
    """
    fields = args_schema.model_fields
    out: dict[str, Any] = {}
    for key, value in arguments.items():
        field = fields.get(key)
        if field is None:
            out[key] = value
            continue
        path = f"{_path}.{key}" if _path else key
        if value is None:
            if not field.is_required() and not _accepts_none(field.annotation):
                logger.debug(
                    "Tool %r: argument %r is null but its schema does not allow null; "
                    "treating it as not given",
                    tool_name,
                    path,
                )
                continue
            out[key] = value
            continue
        out[key] = _drop_nulls_in(field.annotation, value, tool_name, path)
    return out


def plain_arguments(value: Any) -> Any:
    """*value* as plain JSON-like data: dicts, lists, strings, numbers, bools, ``None``.

    Pydantic models become dicts of the fields that were set
    (``model_dump(mode="json", exclude_unset=True)``); other values go
    through pydantic's JSON conversion, and anything it cannot convert is
    kept as is.
    """
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_unset=True)
    if isinstance(value, Mapping):
        return {str(k): plain_arguments(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain_arguments(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        return to_jsonable_python(value)
    except Exception:
        return value


def accepts_free_form_arguments(args_schema: Any) -> bool:
    """Whether *args_schema* is a model without fields that takes any keys."""
    return (
        _is_model_class(args_schema)
        and not args_schema.model_fields
        and args_schema.model_config.get("extra") == "allow"
    )


def parse_tool_arguments(
    tool_name: str, args_schema: Any, tool_input: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a tool call and return the arguments the model gave, as plain JSON.

    Explicit ``null`` for optional, non-nullable fields is treated as not
    given (see :func:`drop_disallowed_nulls`).  The result holds only the
    fields the model set, at every level: no defaults are filled in, so a
    server applies its own.

    Raises:
        ToolArgumentError: The arguments do not match *args_schema*.
    """
    if not _is_model_class(args_schema):
        return cast(dict[str, Any], plain_arguments(dict(tool_input)))
    cleaned = drop_disallowed_nulls(args_schema, tool_input, tool_name=tool_name)
    try:
        model = args_schema.model_validate(cleaned)
    except ValidationError as exc:
        raise ToolArgumentError.from_validation_error(tool_name, exc) from exc
    return model.model_dump(mode="json", exclude_unset=True)


class WrappingTool(BaseTool):
    """Base for a tool that wraps another and shares its ``args_schema``.

    Used by the approval gate, guardrail scanning and tool tracing.  The
    wrapper's ``_arun`` receives the arguments the model gave, validated
    against the shared schema, without the defaults LangChain fills in
    (``None`` for every optional parameter left out) and without ``null``
    for optional parameters that cannot be null.  Values keep their
    validated types (nested objects are model instances, as the wrapped
    tool's own ``_arun`` expects them); use :func:`plain_arguments` for a
    JSON view.

    When validation fails, the wrapped tool is asked to parse the same
    input, so it reports the failure its own way (an MCP tool fires its
    trace hooks and raises :class:`ToolArgumentError`).
    """

    def _to_args_and_kwargs(
        self, tool_input: str | dict[str, Any], tool_call_id: str | None
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        # BaseTool drops every key of a schema without fields, even one
        # that takes free-form keys.
        if isinstance(tool_input, dict) and accepts_free_form_arguments(self.args_schema):
            return (), dict(tool_input)
        return super()._to_args_and_kwargs(tool_input, tool_call_id)

    def _parse_input(
        self, tool_input: str | dict[str, Any], tool_call_id: str | None
    ) -> str | dict[str, Any]:
        schema = self.args_schema
        if not isinstance(tool_input, dict) or not _is_model_class(schema):
            return super()._parse_input(tool_input, tool_call_id)
        cleaned = drop_disallowed_nulls(
            cast(type[BaseModel], schema), tool_input, tool_name=self.name
        )
        try:
            parsed = super()._parse_input(cleaned, tool_call_id)
        except ValidationError as exc:
            self._on_invalid_arguments(cleaned, exc)
            inner = getattr(self, "_inner", None)
            if isinstance(inner, BaseTool):
                inner._to_args_and_kwargs(dict(cleaned), tool_call_id)
            raise
        if not isinstance(parsed, dict):
            return parsed
        # LangChain adds every field with a default; keep what the model
        # gave and injected values (InjectedToolCallId is written into
        # ``cleaned``).
        injected: frozenset[str] = getattr(self, "_injected_args_keys", frozenset())
        return {k: v for k, v in parsed.items() if k in cleaned or k in injected}

    def _on_invalid_arguments(self, arguments: dict[str, Any], exc: ValidationError) -> None:
        """Called when a call's arguments fail validation, before the error is raised."""


# Legacy aliases removed — use promptise.mcp.client.MCPToolAdapter instead.
