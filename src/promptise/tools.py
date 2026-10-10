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
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Union, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

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
                return Optional[types]  # type: ignore[return-value]
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

    # Nested object with properties -> build a Pydantic model
    if t == "object" and prop.get("properties"):
        return _jsonschema_to_pydantic(prop, model_name=name_hint, _defs=defs)

    # Array with structured items -> list[Model]
    if t == "array":
        items = prop.get("items", {})
        if items:
            items = _resolve_refs(items, defs)
            if items.get("type") == "object" and items.get("properties"):
                inner_model = _jsonschema_to_pydantic(
                    items,
                    model_name=f"{name_hint}_Item",
                    _defs=defs,
                )
                return list[inner_model]  # type: ignore[valid-type]
            inner_type = _primitive_type(items.get("type"))
            if inner_type is not Any:
                return list[inner_type]  # type: ignore[valid-type]
        return list

    # Enum
    if "enum" in prop:
        vals = prop["enum"]
        if all(isinstance(v, str) for v in vals):
            from typing import Literal

            return Literal[tuple(vals)]  # type: ignore[valid-type,return-value]

    return _primitive_type(t)


def _primitive_type(t: str | None) -> type[Any]:
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
    return mapping.get(t, Any)


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


# Legacy aliases removed — use promptise.mcp.client.MCPToolAdapter instead.
