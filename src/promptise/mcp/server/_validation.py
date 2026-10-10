"""Input validation: build Pydantic models from function type hints.

At registration time the decorator introspects the handler's signature and
builds a Pydantic model.  At call time incoming ``arguments`` are validated
against that model, producing clear field-level errors on failure.

**Schema dereferencing:** Pydantic v2 generates JSON Schemas containing
``$defs`` / ``$ref`` for nested ``BaseModel`` types.  Many MCP clients
(Gemini CLI, LM Studio, VS Code Copilot, LangChain.js, Claude Desktop)
cannot resolve these references, and most LLM APIs (Claude, GPT-4) produce
lower-quality tool calls for deeply nested schemas.  To maximise
compatibility we **inline** all ``$ref`` entries at registration time so
the ``inputSchema`` sent to clients is self-contained without references.

**JSON string pre-parsing:** Some MCP clients (Claude Code, Cursor,
Qwen Code) incorrectly serialise nested-object arguments as JSON *strings*
instead of native dicts.  ``validate_arguments`` detects this and parses
the string back before Pydantic validation.

**Parameter metadata:** ``Annotated[T, Field(...)]`` metadata (``description``,
``ge``, ``pattern``, ``examples``, ...) is kept, and a ``Field(...)`` used as
the parameter default works the same way.  When neither carries a
description, the one from the docstring's ``Args:`` section (passed in as
*param_docs*) is used.

**Performance note:** ``build_input_model`` detects at build time whether
any parameter is a ``BaseModel`` subclass.  When all fields are primitives
the faster ``model_dump()`` path is used; otherwise the ``getattr()`` loop
preserves model instances.
"""

from __future__ import annotations

import inspect
import json as _json
from collections.abc import Mapping
from typing import Annotated, Any, get_args, get_origin, get_type_hints

from pydantic import BaseModel, Field, create_model
from pydantic import ValidationError as PydanticValidationError
from pydantic.fields import FieldInfo

from ._errors import ValidationError


def _python_type_to_json_type(annotation: Any) -> str:
    """Map a Python type annotation to a JSON Schema type string."""
    origin = get_origin(annotation)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    mapping: dict[Any, str] = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }
    return mapping.get(annotation, "string")


def _has_model_fields(model: type[BaseModel]) -> bool:
    """Return ``True`` if any field in *model* is a ``BaseModel`` subclass.

    Checks the outer annotation and the first generic argument of
    ``Optional[X]`` / ``list[X]`` / ``X | None`` so that fields like
    ``address: Address`` and ``items: list[Item]`` are detected.
    """
    for info in model.model_fields.values():
        ann = info.annotation
        if ann is None:
            continue
        # Direct BaseModel subclass?
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            return True
        # Unwrap Optional / Union / list / dict
        origin = get_origin(ann)
        if origin is not None:
            for arg in get_args(ann):
                if isinstance(arg, type) and issubclass(arg, BaseModel):
                    return True
    return False


# ------------------------------------------------------------------
# Schema dereferencing
# ------------------------------------------------------------------


def _resolve_ref(ref: str, defs: dict[str, Any]) -> dict[str, Any]:
    """Resolve a single ``$ref`` string like ``#/$defs/Address``."""
    parts = ref.lstrip("#/").split("/")
    node: Any = {"$defs": defs}
    for part in parts:
        node = node[part]
    return node  # type: ignore[return-value]


def _dereference_node(node: Any, defs: dict[str, Any], _seen: set[str] | None = None) -> Any:
    """Recursively inline all ``$ref`` entries in a JSON Schema node.

    Handles:
    - ``{"$ref": "#/$defs/Foo"}`` → inlined copy of ``Foo``
    - ``allOf``, ``anyOf``, ``oneOf`` with ``$ref`` items
    - Nested ``properties``, ``items``, ``additionalProperties``
    - Circular references (detected via *_seen* to avoid infinite recursion)
    """
    if _seen is None:
        _seen = set()

    if isinstance(node, dict):
        # Direct $ref replacement
        if "$ref" in node:
            ref = node["$ref"]
            if ref in _seen:
                # Circular reference — return empty object to break the loop
                return {"type": "object"}
            _seen = _seen | {ref}
            resolved = _resolve_ref(ref, defs)
            # Merge any sibling keys (e.g. description alongside $ref)
            merged = {**_dereference_node(resolved, defs, _seen)}
            for k, v in node.items():
                if k != "$ref":
                    merged[k] = _dereference_node(v, defs, _seen)
            return merged

        # Recurse into all keys (skip $defs itself, it will be removed)
        return {k: _dereference_node(v, defs, _seen) for k, v in node.items() if k != "$defs"}

    if isinstance(node, list):
        return [_dereference_node(item, defs, _seen) for item in node]

    return node


def dereference_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline all ``$ref`` / ``$defs`` in a JSON Schema.

    Returns a new schema dict with no ``$ref`` or ``$defs`` keys.
    Every reference is replaced by the full inlined definition.

    This is critical for MCP client compatibility — many clients
    (Gemini CLI, LM Studio, Claude Desktop, VS Code Copilot,
    LangChain.js) cannot resolve ``$ref`` references.

    Example::

        >>> schema = {
        ...     "$defs": {"Addr": {"type": "object", "properties": {"city": {"type": "string"}}}},
        ...     "properties": {"address": {"$ref": "#/$defs/Addr"}},
        ...     "type": "object",
        ... }
        >>> dereference_schema(schema)
        {'properties': {'address': {'type': 'object', 'properties': {'city': {'type': 'string'}}}}, 'type': 'object'}
    """
    defs = schema.get("$defs", {})
    if not defs:
        # Nothing to dereference — strip $defs if empty and return
        result = {k: v for k, v in schema.items() if k != "$defs"}
        return result

    return _dereference_node(schema, defs)


# ------------------------------------------------------------------
# JSON string pre-parsing
# ------------------------------------------------------------------


def _preparse_json_strings(
    arguments: dict[str, Any],
    model: type[BaseModel],
) -> dict[str, Any]:
    """Pre-parse JSON string arguments that should be objects/arrays.

    Some MCP clients (Claude Code, Cursor, Qwen Code) incorrectly
    serialise nested-object arguments as JSON *strings* instead of native
    dicts.  For example, the client sends::

        {"employee": "{\"name\": \"Alice\", \"address\": {...}}"}

    instead of::

        {"employee": {"name": "Alice", "address": {...}}}

    This function detects string values where the model expects a dict /
    list / BaseModel and parses them back.
    """
    if not arguments:
        return arguments

    result = dict(arguments)
    for field_name, field_info in model.model_fields.items():
        if field_name not in result:
            continue
        value = result[field_name]
        if not isinstance(value, str):
            continue

        ann = field_info.annotation
        if ann is None:
            continue

        # Check if the annotation expects a non-string type
        expects_object = False
        if (
            isinstance(ann, type)
            and issubclass(ann, BaseModel)
            or isinstance(ann, type)
            and issubclass(ann, (dict, list))
        ):
            expects_object = True
        else:
            origin = get_origin(ann)
            if origin in (dict, list):
                expects_object = True
            # Check Optional[BaseModel] / Optional[list] etc.
            if origin is not None:
                for arg in get_args(ann):
                    if isinstance(arg, type) and issubclass(arg, (BaseModel, dict, list)):
                        expects_object = True
                        break

        if expects_object:
            try:
                parsed = _json.loads(value)
                if isinstance(parsed, (dict, list)):
                    result[field_name] = parsed
            except (ValueError, TypeError):
                pass  # Let Pydantic validation produce a clear error

    return result


def _has_description(annotation: Any, field_info: FieldInfo) -> bool:
    """Return ``True`` if the field already carries a description.

    Looks at the ``Field(...)`` default and at ``Field(...)`` entries in
    ``Annotated[...]`` metadata (nested ``Annotated`` is flattened by
    ``typing``, so one level is enough).
    """
    if field_info.description:
        return True
    if get_origin(annotation) is Annotated:
        for meta in annotation.__metadata__:
            if isinstance(meta, FieldInfo) and meta.description:
                return True
    return False


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def build_input_model(
    func: Any,
    *,
    exclude: set[str] | None = None,
    param_docs: Mapping[str, str] | None = None,
) -> tuple[type[BaseModel], dict[str, Any]]:
    """Build a Pydantic model and JSON Schema from a function signature.

    Parameters whose names are in *exclude* (e.g. dependency-injected
    params or ``RequestContext``) are skipped.

    Field metadata is taken from ``Annotated[T, Field(...)]`` and from a
    ``Field(...)`` default.  *param_docs* maps parameter names to
    descriptions (usually parsed from the docstring's ``Args:`` section);
    an entry is used only when the parameter has no ``Field`` description.

    The returned schema has all ``$ref`` / ``$defs`` inlined for maximum
    MCP client compatibility.

    Returns:
        ``(PydanticModel, json_schema_dict)``
    """
    exclude = exclude or set()
    param_docs = param_docs or {}
    sig = inspect.signature(func)
    fields: dict[str, Any] = {}

    # Resolve stringified annotations from ``from __future__ import annotations``
    # so Pydantic receives real types (e.g. ``Optional[str]`` instead of the
    # string ``'Optional[str]'``).  ``include_extras`` keeps ``Annotated``
    # metadata -- without it every ``Field(description=...)`` is dropped.
    try:
        resolved_hints = get_type_hints(func, include_extras=True)
    except Exception:
        resolved_hints = {}

    for name, param in sig.parameters.items():
        if name in exclude or name == "self":
            continue

        annotation = resolved_hints.get(name, param.annotation)
        if annotation is inspect.Parameter.empty:
            annotation = str  # default to str if untyped

        if isinstance(param.default, FieldInfo):
            # ``order_id: str = Field(description=...)``
            field_info = param.default
        elif param.default is inspect.Parameter.empty:
            # Required field
            field_info = Field(...)
        else:
            field_info = Field(default=param.default)

        doc = param_docs.get(name)
        if doc and not _has_description(annotation, field_info):
            # Annotated metadata merges with the assigned FieldInfo, so this
            # adds the description without touching other constraints.
            annotation = Annotated[annotation, Field(description=doc)]

        fields[name] = (annotation, field_info)

    model = create_model(f"{func.__name__}_Input", **fields)
    schema = model.model_json_schema()
    # Remove pydantic internal keys that MCP doesn't need
    schema.pop("title", None)

    # Inline all $ref/$defs for MCP client compatibility.
    # Many clients (Gemini CLI, LM Studio, Claude Desktop, VS Code
    # Copilot, LangChain.js) cannot resolve $ref references, and most
    # LLMs produce lower-quality tool calls for schemas with $ref.
    schema = dereference_schema(schema)

    # Stash a flag so validate_arguments can pick the fast path.
    # BaseModel subclasses need getattr() to preserve instances;
    # primitive-only models can use the faster model_dump().
    model.__promptise_has_models__ = _has_model_fields(model)  # type: ignore[attr-defined]

    return model, schema


def validate_arguments(
    model: type[BaseModel],
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Validate *arguments* against *model*, raising ``ValidationError`` on failure.

    Uses a fast ``model_dump()`` path for primitive-only tools, and a
    slower ``getattr()`` loop for tools that have ``BaseModel`` subclass
    parameters (to preserve model instances).

    Pre-parses JSON string arguments for broken clients that serialise
    nested objects as strings.
    """
    # Pre-parse JSON strings from broken MCP clients (Claude Code,
    # Cursor, Qwen Code) that serialise objects as strings.
    has_models: bool = getattr(model, "__promptise_has_models__", True)
    if has_models:
        arguments = _preparse_json_strings(arguments, model)

    try:
        instance = model.model_validate(arguments)

        # Fast path: no BaseModel subclass fields -> model_dump is faster
        # and allocates fewer intermediate objects.
        if not has_models:
            return instance.model_dump()

        # Slow path: preserve Pydantic model instances -- don't flatten
        # to dicts. getattr returns the validated Python value, keeping
        # BaseModel subclass instances intact while returning primitives
        # as-is.
        return {field_name: getattr(instance, field_name) for field_name in model.model_fields}
    except PydanticValidationError as exc:
        field_errors: dict[str, str] = {}
        messages: list[str] = []
        error_types: set[str] = set()
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            msg = err["msg"]
            field_errors[loc] = msg
            messages.append(f"{loc}: {msg}")
            error_types.add(err.get("type", ""))

        # Generate specific suggestion based on error types
        if "missing" in error_types:
            suggestion = (
                "Required fields are missing. Check that all required parameters are provided."
            )
        elif error_types & {"string_type", "int_type", "float_type", "bool_type", "type_error"}:
            suggestion = "Parameter type mismatch. Check that values match the expected types in the tool schema."
        elif "value_error" in error_types or "assertion_error" in error_types:
            suggestion = "Parameter value out of range or failed a constraint check."
        else:
            suggestion = "Check the parameter types and constraints in the tool description."

        raise ValidationError(
            f"Invalid input: {'; '.join(messages)}",
            field_errors=field_errors,
            suggestion=suggestion,
        ) from exc
