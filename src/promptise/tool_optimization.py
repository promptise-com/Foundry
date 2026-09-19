"""MCP tool token optimization.

Provides opt-in strategies to reduce the token cost of tool definitions
sent to the LLM with every invocation:

* **Static optimization** — schema minification, description truncation,
  depth flattening.  Applied once at build time.
* **Semantic tool selection** — embeds tool descriptions and selects only
  the most relevant tools per invocation based on the user's query.

Enable via ``build_agent(optimize_tools=True)`` for sensible defaults,
or pass an :class:`OptimizationLevel` string or a full
:class:`ToolOptimizationConfig` for fine-grained control.
"""

from __future__ import annotations

import itertools
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, create_model

logger = logging.getLogger(__name__)

# ======================================================================
# Configuration
# ======================================================================


class OptimizationLevel(str, Enum):
    """Preset optimization levels.

    * ``MINIMAL`` — schema minification + description truncation.
    * ``STANDARD`` — deeper minification + nested description stripping.
    * ``SEMANTIC`` — all static optimizations + per-invocation semantic
      tool selection.
    """

    MINIMAL = "minimal"
    STANDARD = "standard"
    SEMANTIC = "semantic"


@dataclass
class ToolOptimizationConfig:
    """Configuration for MCP tool token optimization.

    Pass a preset :attr:`level` for sensible defaults, or override
    individual settings for fine-grained control.  Any field set
    explicitly takes precedence over the preset.

    Args:
        level: Preset optimization level.
        minify_schema: Strip ``description`` from Pydantic Field metadata.
        max_description_length: Truncate tool descriptions at *N* chars.
        strip_nested_descriptions: Remove descriptions from nested
            model fields (keeps top-level field descriptions).
        max_schema_depth: Flatten nested objects beyond this depth to
            ``dict``.  ``None`` means no limit.
        semantic_selection: Enable per-invocation semantic tool selection.
        semantic_top_k: Number of most-relevant tools to select per
            invocation.
        always_include_fallback: Include a ``request_more_tools``
            fallback tool when semantic selection is active.
        embedding_model: Model name or **local path** for
            ``sentence-transformers``.  Defaults to
            ``"all-MiniLM-L6-v2"`` (downloaded once, then cached in
            ``~/.cache/huggingface/``).  Point to a local directory
            for fully-offline / air-gapped deployments::

                embedding_model="/models/all-MiniLM-L6-v2"

        preserve_tools: Tool names that are never optimized and always
            included in semantic selection.
    """

    level: OptimizationLevel | None = None

    # Static optimization overrides
    minify_schema: bool | None = None
    max_description_length: int | None = None
    strip_nested_descriptions: bool | None = None
    max_schema_depth: int | None = None

    # Semantic selection overrides
    semantic_selection: bool | None = None
    semantic_top_k: int | None = None
    always_include_fallback: bool | None = None
    embedding_model: str | None = None

    # Shared
    preserve_tools: set[str] | None = None


# ======================================================================
# Resolved config (internal)
# ======================================================================

_PRESETS: dict[OptimizationLevel, dict[str, Any]] = {
    OptimizationLevel.MINIMAL: {
        "minify_schema": True,
        "max_description_length": 200,
        "strip_nested_descriptions": False,
        "max_schema_depth": None,
        "semantic_selection": False,
        "semantic_top_k": 8,
        "always_include_fallback": True,
    },
    OptimizationLevel.STANDARD: {
        "minify_schema": True,
        "max_description_length": 150,
        "strip_nested_descriptions": True,
        "max_schema_depth": 3,
        "semantic_selection": False,
        "semantic_top_k": 8,
        "always_include_fallback": True,
    },
    OptimizationLevel.SEMANTIC: {
        "minify_schema": True,
        "max_description_length": 100,
        "strip_nested_descriptions": True,
        "max_schema_depth": 2,
        "semantic_selection": True,
        "semantic_top_k": 8,
        "always_include_fallback": True,
    },
}

_DEFAULTS = _PRESETS[OptimizationLevel.MINIMAL]


@dataclass(frozen=True)
class _ResolvedConfig:
    """Fully-resolved optimization settings (no ``None`` values)."""

    minify_schema: bool
    max_description_length: int
    strip_nested_descriptions: bool
    max_schema_depth: int | None
    semantic_selection: bool
    semantic_top_k: int
    always_include_fallback: bool
    embedding_model: str
    preserve_tools: frozenset[str]


#: Default embedding model — small, fast, runs locally.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _resolve_config(config: ToolOptimizationConfig) -> _ResolvedConfig:
    """Merge a :class:`ToolOptimizationConfig` into a fully-resolved config."""
    preset = _PRESETS.get(config.level, _DEFAULTS) if config.level else _DEFAULTS

    def _pick(field_name: str) -> Any:
        explicit = getattr(config, field_name, None)
        return explicit if explicit is not None else preset.get(field_name)

    return _ResolvedConfig(
        minify_schema=_pick("minify_schema"),
        max_description_length=_pick("max_description_length"),
        strip_nested_descriptions=_pick("strip_nested_descriptions"),
        max_schema_depth=_pick("max_schema_depth"),
        semantic_selection=_pick("semantic_selection"),
        semantic_top_k=_pick("semantic_top_k"),
        always_include_fallback=_pick("always_include_fallback"),
        embedding_model=config.embedding_model or DEFAULT_EMBEDDING_MODEL,
        preserve_tools=frozenset(config.preserve_tools or ()),
    )


# ======================================================================
# Static optimization: description truncation
# ======================================================================


def _truncate_description(desc: str, max_len: int) -> str:
    """Truncate *desc* at a word boundary, appending ``...`` if trimmed."""
    if not desc or len(desc) <= max_len:
        return desc
    # Find last space before max_len
    trunc = desc[: max_len - 3]
    last_space = trunc.rfind(" ")
    if last_space > max_len // 2:
        trunc = trunc[:last_space]
    return trunc.rstrip() + "..."


# ======================================================================
# Static optimization: schema minification
# ======================================================================

_MODEL_COUNTER = itertools.count(1)


def _mini_model_name(base: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_]", "_", base).strip("_") or "Mini"
    return f"{safe}_opt_{next(_MODEL_COUNTER)}"


def _minify_pydantic_model(
    model: type[BaseModel],
    *,
    strip_nested: bool = False,
    max_depth: int | None = None,
    _depth: int = 0,
) -> type[BaseModel]:
    """Rebuild a Pydantic model with stripped Field descriptions.

    Args:
        model: The original Pydantic model to minify.
        strip_nested: If True, also strip descriptions from nested
            model fields (not just the top level).
        max_depth: If set, replace nested model fields beyond this
            depth with ``dict``.
        _depth: Internal recursion depth counter.
    """
    if max_depth is not None and _depth >= max_depth:
        # Beyond max depth — this should have been replaced with dict
        # by the parent call.  Return as-is.
        return model

    fields: dict[str, Any] = {}

    for name, field_info in model.model_fields.items():
        annotation = field_info.annotation
        default = field_info.default

        # Strip descriptions: always at top level (for token savings),
        # and at nested levels when strip_nested=True.
        should_strip = strip_nested or _depth == 0

        # Handle nested Pydantic models
        inner_model = _unwrap_model(annotation)
        if inner_model is not None and issubclass(inner_model, BaseModel):
            if max_depth is not None and _depth + 1 >= max_depth:
                # Flatten to dict at this depth
                annotation = _replace_model_with_dict(annotation, inner_model)
            else:
                # Recurse
                minified = _minify_pydantic_model(
                    inner_model,
                    strip_nested=strip_nested,
                    max_depth=max_depth,
                    _depth=_depth + 1,
                )
                annotation = _replace_model_in_annotation(
                    annotation,
                    inner_model,
                    minified,
                )

        # Build field without description (or with it, if not stripping)
        default_factory = field_info.default_factory
        desc = None if should_strip else field_info.description

        if field_info.is_required():
            field_def = Field(..., description=desc)
        elif default_factory is not None:
            field_def = Field(default_factory=default_factory, description=desc)
        elif default is not None:
            field_def = Field(default=default, description=desc)
        else:
            field_def = Field(default=None, description=desc)

        fields[name] = (annotation, field_def)

    new_name = _mini_model_name(model.__name__)
    return cast(
        type[BaseModel],
        create_model(new_name, **cast(dict[str, Any], fields)),
    )


def _unwrap_model(annotation: Any) -> type | None:
    """Extract a BaseModel subclass from a possibly-wrapped annotation.

    Handles ``list[Model]``, ``Optional[Model]``, ``Model`` directly.
    Returns ``None`` if the annotation doesn't contain a model.
    """
    origin = getattr(annotation, "__origin__", None)

    if origin is list:
        args = getattr(annotation, "__args__", ())
        if args and isinstance(args[0], type) and issubclass(args[0], BaseModel):
            return args[0]
        return None

    # Optional[X] is Union[X, None]
    import typing

    if origin is typing.Union:
        args = getattr(annotation, "__args__", ())
        non_none = [a for a in args if a is not type(None)]
        if (
            len(non_none) == 1
            and isinstance(non_none[0], type)
            and issubclass(non_none[0], BaseModel)
        ):
            return non_none[0]
        return None

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation

    return None


def _replace_model_with_dict(annotation: Any, model: type) -> Any:
    """Replace a model type with ``dict`` in the annotation."""
    return _replace_model_in_annotation(annotation, model, dict)


def _replace_model_in_annotation(
    annotation: Any,
    old: type,
    new: type,
) -> Any:
    """Replace *old* type with *new* in a possibly-wrapped annotation."""
    origin = getattr(annotation, "__origin__", None)

    if origin is list:
        args = getattr(annotation, "__args__", ())
        if args and args[0] is old:
            return list[new]  # type: ignore[valid-type]
        return annotation

    import typing

    if origin is typing.Union:
        args = getattr(annotation, "__args__", ())
        new_args = tuple(new if a is old else a for a in args)
        if len(new_args) == 2 and type(None) in new_args:
            inner = [a for a in new_args if a is not type(None)][0]
            from typing import Optional

            return Optional[inner]  # type: ignore[valid-type]
        return typing.Union[new_args]  # type: ignore[valid-type]

    if annotation is old:
        return new

    return annotation


# ======================================================================
# Static optimization: apply to tool list
# ======================================================================


def apply_static_optimizations(
    tools: list[BaseTool],
    config: _ResolvedConfig,
) -> list[BaseTool]:
    """Apply static optimizations (description truncation + schema
    minification) to a list of tools in-place and return the same list.

    Tools whose names are in ``config.preserve_tools`` are skipped.
    """
    for tool in tools:
        if tool.name in config.preserve_tools:
            continue

        # Truncate description
        if config.max_description_length and tool.description:
            tool.description = _truncate_description(
                tool.description,
                config.max_description_length,
            )

        # Minify schema
        if (
            config.minify_schema
            and tool.args_schema is not None
            and isinstance(tool.args_schema, type)
            and issubclass(tool.args_schema, BaseModel)
        ):
            tool.args_schema = _minify_pydantic_model(
                tool.args_schema,
                strip_nested=config.strip_nested_descriptions,
                max_depth=config.max_schema_depth,
            )

    return tools


# ======================================================================
# Semantic tool selection
# ======================================================================


class ToolIndex:
    """In-memory semantic index over tool descriptions.

    Embeds all tool descriptions using ``sentence_transformers``
    and selects tools via cosine similarity.

    Args:
        tools: The full set of ``BaseTool`` instances to index.
        model_name_or_path: A HuggingFace model name (e.g.
            ``"all-MiniLM-L6-v2"``) or a **local directory path**
            containing the model files for fully-offline deployments.
            Defaults to :data:`DEFAULT_EMBEDDING_MODEL`.

    Example — local / air-gapped deployment::

        index = ToolIndex(tools, model_name_or_path="/models/all-MiniLM-L6-v2")
    """

    def __init__(
        self,
        tools: list[BaseTool],
        model_name_or_path: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._tools = {t.name: t for t in tools}
        self._names = [t.name for t in tools]
        self._texts = [f"{t.name}: {t.description}" for t in tools]
        self._init_embeddings(model_name_or_path)

    def _init_embeddings(self, model_name_or_path: str) -> None:
        """Embed all tool descriptions with sentence_transformers."""
        import warnings

        from sentence_transformers import SentenceTransformer

        # Suppress the harmless "embeddings.position_ids UNEXPECTED"
        # warning from older model checkpoints on newer transformers.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*position_ids.*")
            model = SentenceTransformer(model_name_or_path)
        self._embed_fn = model.encode
        self._embeddings = model.encode(self._texts, normalize_embeddings=True)
        logger.debug(
            "ToolIndex: embedded %d tools with model %r",
            len(self._texts),
            model_name_or_path,
        )

    def select(
        self,
        query: str,
        top_k: int = 8,
        preserve: frozenset[str] | None = None,
    ) -> list[BaseTool]:
        """Select the *top_k* most relevant tools for *query*.

        Tools in *preserve* are always included regardless of relevance.

        Returns:
            Ordered list of selected ``BaseTool`` instances.
        """
        preserve = preserve or frozenset()
        scores = self._embedding_scores(query)

        # Sort by score descending
        ranked = sorted(
            zip(self._names, scores, strict=False),
            key=lambda x: x[1],
            reverse=True,
        )

        selected: dict[str, BaseTool] = {}

        # Always include preserved tools
        for name in preserve:
            if name in self._tools:
                selected[name] = self._tools[name]

        # Add top-k by score (skip already-selected preserved tools)
        for name, _score in ranked:
            if len(selected) >= top_k:
                break
            if name not in selected:
                selected[name] = self._tools[name]

        return list(selected.values())

    @property
    def all_tool_names(self) -> list[str]:
        """All indexed tool names."""
        return list(self._names)

    @property
    def all_tools(self) -> list[BaseTool]:
        """All indexed tools."""
        return [self._tools[n] for n in self._names]

    @property
    def tool_summaries(self) -> str:
        """One-line summaries of all tools for the fallback tool."""
        lines = []
        for name, text in zip(self._names, self._texts, strict=False):
            # Truncate to 80 chars for the summary
            desc = self._tools[name].description or ""
            short = desc[:77] + "..." if len(desc) > 80 else desc
            lines.append(f"- {name}: {short}")
        return "\n".join(lines)

    def _embedding_scores(self, query: str) -> list[float]:
        """Cosine similarity scores using pre-computed embeddings."""
        import numpy as np

        q_emb = self._embed_fn([query], normalize_embeddings=True)[0]
        # self._embeddings shape: (N, D), q_emb shape: (D,)
        scores = np.dot(self._embeddings, q_emb).tolist()
        return scores


# ======================================================================
# Fallback tool: request_more_tools
# ======================================================================


class _RequestMoreToolsTool(BaseTool):
    """Fallback tool that lists all available tools.

    When semantic tool selection is active, the agent may not have the
    tool it needs.  Calling this tool returns the full list of available
    tools so the agent can refine its approach.
    """

    name: str = "request_more_tools"
    description: str = (
        "If you need a tool that is not currently available, call this "
        "to see all available tools and their descriptions."
    )

    _tool_index: ToolIndex

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, tool_index: ToolIndex, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        object.__setattr__(self, "_tool_index", tool_index)

    async def _arun(self, **kwargs: Any) -> str:
        """Return all tool summaries."""
        count = len(self._tool_index.all_tool_names)
        return (
            f"There are {count} tools available. Here is the full list:\n\n"
            f"{self._tool_index.tool_summaries}\n\n"
            "You can now use these tools by name in your next action."
        )

    def _run(self, **kwargs: Any) -> str:
        import anyio

        return anyio.run(lambda: self._arun(**kwargs))
