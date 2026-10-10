"""MCP tool token optimization.

Provides opt-in strategies to reduce the token cost of tool definitions
sent to the LLM with every invocation:

* **Static optimization** — schema minification, description truncation,
  depth flattening.  Applied once at build time.
* **Semantic tool selection** — embeds tool descriptions and, before every
  model call, offers only the tools most relevant to the recent
  conversation (plus preserved tools and a ``request_more_tools``
  fallback that unlocks more on demand).

Enable via ``build_agent(optimize_tools=True)`` for sensible defaults,
or pass an :class:`OptimizationLevel` string or a full
:class:`ToolOptimizationConfig` for fine-grained control.
"""

from __future__ import annotations

import importlib.util
import itertools
import json
import logging
import re
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model

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
        semantic_selection: Enable semantic tool selection: before every
            model call, offer only the tools most relevant to the recent
            conversation.  Requires ``sentence-transformers``
            (``pip install "promptise[tool-optimization]"``).
        semantic_top_k: Number of most-relevant tools to offer per model
            call.  Preserved tools, tools called in the previous or
            current turn, tools unlocked via ``request_more_tools`` and the
            fallback itself are added on top.
        semantic_context_turns: How many of the most recent user turns
            make up the selection query (together with the latest
            assistant reply and the tool calls of the previous and current
            turn), so a
            follow-up like "Yes, go ahead." keeps the tools of the turn it
            answers.
        always_include_fallback: Include a ``request_more_tools``
            fallback tool when semantic selection is active.
        embedding_model: Model name or **local path** for
            ``sentence-transformers``.  Defaults to
            ``"all-MiniLM-L6-v2"`` (downloaded once, then cached in
            ``~/.cache/huggingface/``).  Point to a local directory
            for fully-offline / air-gapped deployments::

                embedding_model="/models/all-MiniLM-L6-v2"

        preserve_tools: Tool names that are never optimized (full
            description and parameter descriptions are kept) and always
            offered by semantic selection.
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
    semantic_context_turns: int | None = None
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
        "semantic_context_turns": 3,
        "always_include_fallback": True,
    },
    OptimizationLevel.STANDARD: {
        "minify_schema": True,
        "max_description_length": 150,
        "strip_nested_descriptions": True,
        "max_schema_depth": 3,
        "semantic_selection": False,
        "semantic_top_k": 8,
        "semantic_context_turns": 3,
        "always_include_fallback": True,
    },
    OptimizationLevel.SEMANTIC: {
        "minify_schema": True,
        "max_description_length": 100,
        "strip_nested_descriptions": True,
        "max_schema_depth": 2,
        "semantic_selection": True,
        "semantic_top_k": 8,
        "semantic_context_turns": 3,
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
    semantic_context_turns: int
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

    for field_name in ("semantic_top_k", "semantic_context_turns"):
        if _pick(field_name) < 1:
            raise ValueError(f"ToolOptimizationConfig.{field_name} must be at least 1")

    return _ResolvedConfig(
        minify_schema=_pick("minify_schema"),
        max_description_length=_pick("max_description_length"),
        strip_nested_descriptions=_pick("strip_nested_descriptions"),
        max_schema_depth=_pick("max_schema_depth"),
        semantic_selection=_pick("semantic_selection"),
        semantic_top_k=_pick("semantic_top_k"),
        semantic_context_turns=_pick("semantic_context_turns"),
        always_include_fallback=_pick("always_include_fallback"),
        embedding_model=config.embedding_model or DEFAULT_EMBEDDING_MODEL,
        preserve_tools=frozenset(config.preserve_tools or ()),
    )


# ======================================================================
# Static optimization: description truncation
# ======================================================================


_SENTENCE_END = re.compile(r"[.!?](?=\s|$)")


def _is_abbreviation(text: str, period_at: int) -> bool:
    """Whether the ``.`` at *period_at* ends an abbreviation like ``e.g.``."""
    word = text[: period_at + 1].rsplit(None, 1)[-1]
    return "." in word[:-1] or len(word) <= 2


def _truncate_description(desc: str, max_len: int) -> str:
    """Shorten *desc* to at most *max_len* characters.

    Ends on a whole sentence (no ellipsis) when that keeps at least two
    thirds of the budget.  Otherwise cuts at the last word boundary,
    drops trailing punctuation and appends a single ``...``.  Only falls
    back to a hard cut when no word boundary lies in the second half of
    the budget.
    """
    if not desc or len(desc) <= max_len:
        return desc
    if max_len <= 3:
        return desc[:max_len]

    head = desc[:max_len]
    min_keep = max_len // 2

    # 1. Whole sentences.
    cut = -1
    for match in _SENTENCE_END.finditer(head):
        if match.end() * 3 >= max_len * 2 and not (
            match.group() == "." and _is_abbreviation(head, match.start())
        ):
            cut = match.end()
    if cut != -1:
        return head[:cut]

    # 2. Word boundary + "...".
    body = desc[: max_len - 3]
    if not desc[max_len - 3].isspace():  # the cut splits a word
        last_space = body.rfind(" ")
        if last_space >= min_keep:
            body = body[:last_space]
    body = body.rstrip().rstrip(".,;:!?-–—(").rstrip()
    return body + "..."


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

#: The extra that installs the semantic selection dependencies.
SEMANTIC_EXTRA = "tool-optimization"

_MISSING_SENTENCE_TRANSFORMERS = (
    "Semantic tool selection (optimize_tools='semantic' or "
    "ToolOptimizationConfig(semantic_selection=True)) needs the "
    "'sentence-transformers' package, which is not installed. Install it with "
    f'pip install "promptise[{SEMANTIC_EXTRA}]" (also included in '
    '"promptise[all]"), or use optimize_tools="standard" for static '
    "optimization only."
)


def require_semantic_dependencies() -> None:
    """Raise a clear :class:`ImportError` if semantic selection can't run.

    ``build_agent()`` calls this before connecting to any MCP server, so a
    missing optional dependency fails fast with the extra to install
    instead of a bare ``ModuleNotFoundError`` halfway through the build.
    """
    if importlib.util.find_spec("sentence_transformers") is None:
        raise ImportError(_MISSING_SENTENCE_TRANSFORMERS)


class ToolIndex:
    """In-memory semantic index over tool descriptions.

    Embeds every tool's ``"name: description"`` once with
    ``sentence-transformers`` and ranks tools against a query by cosine
    similarity.  ``build_agent(optimize_tools="semantic")`` builds one
    for you; build your own to check offline which tools a query would
    get before paying for any model call.

    Requires ``pip install "promptise[tool-optimization]"``.

    Args:
        tools: The ``BaseTool`` instances to index.  Duplicate names keep
            the first tool.
        model_name_or_path: A HuggingFace model name (e.g.
            ``"all-MiniLM-L6-v2"``) or a **local directory path**
            containing the model files for fully-offline deployments.
            Defaults to :data:`DEFAULT_EMBEDDING_MODEL`.

    Example::

        index = ToolIndex(agent.tools, model_name_or_path="/models/all-MiniLM-L6-v2")
        names = [t.name for t in index.select("Suspend user u_42", top_k=8)]
    """

    _QUERY_CACHE_SIZE = 256

    def __init__(
        self,
        tools: Sequence[BaseTool],
        model_name_or_path: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        for tool in tools:
            self._tools.setdefault(tool.name, tool)
        self._names = list(self._tools)
        self._texts = [f"{name}: {self._tools[name].description}" for name in self._names]
        self._score_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._init_embeddings(model_name_or_path)

    def _init_embeddings(self, model_name_or_path: str) -> None:
        """Embed all tool descriptions with sentence_transformers."""
        import warnings

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(_MISSING_SENTENCE_TRANSFORMERS) from exc

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
        preserve: frozenset[str] | set[str] | None = None,
    ) -> list[BaseTool]:
        """Return the *top_k* tools most relevant to *query*, plus *preserve*.

        Preserved tools are always included and don't take one of the
        *top_k* relevance slots, so ``top_k=3`` with two preserved tools
        returns up to five tools.  Names in *preserve* that aren't
        indexed are ignored.

        Returns:
            The ranked tools (most relevant first), then the preserved ones.
        """
        preserve = frozenset(preserve or ())
        scores = self._embedding_scores(query)
        ranked = sorted(
            (name for name in self._names if name not in preserve),
            key=dict(zip(self._names, scores, strict=True)).__getitem__,
            reverse=True,
        )
        selected = ranked[: max(top_k, 0)]
        selected += [name for name in self._names if name in preserve]
        return [self._tools[name] for name in selected]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

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
        """One-line summaries of all tools (as listed by ``request_more_tools``)."""
        return self.summaries(self._names, max_length=80)

    def summaries(self, names: Sequence[str], *, max_length: int | None = None) -> str:
        """``- name: description`` lines for the indexed tools in *names*."""
        lines = []
        for name in names:
            tool = self._tools.get(name)
            if tool is None:
                continue
            desc = tool.description or ""
            if max_length is not None:
                desc = _truncate_description(desc, max_length)
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def _embedding_scores(self, query: str) -> list[float]:
        """Cosine similarity of *query* to every tool (cached per query)."""
        cached = self._score_cache.get(query)
        if cached is not None:
            self._score_cache.move_to_end(query)
            return cached

        import numpy as np

        q_emb = self._embed_fn([query], normalize_embeddings=True)[0]
        # self._embeddings shape: (N, D), q_emb shape: (D,)
        scores: list[float] = np.dot(self._embeddings, q_emb).tolist()
        self._score_cache[query] = scores
        if len(self._score_cache) > self._QUERY_CACHE_SIZE:
            self._score_cache.popitem(last=False)
        return scores


# ----------------------------------------------------------------------
# Selection query from the conversation
# ----------------------------------------------------------------------

_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "ai": "assistant",
    "assistant": "assistant",
    "tool": "tool",
    "function": "tool",
    "system": "system",
}

_MAX_ASSISTANT_CHARS = 500
_MAX_QUERY_CHARS = 2000


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return " ".join(parts)
    return ""


def _message_parts(msg: Any) -> tuple[str, str, list[tuple[str, Any]]]:
    """``(role, text, [(tool_name, args), ...])`` for a dict or LangChain message."""
    if isinstance(msg, dict):
        role = str(msg.get("role") or msg.get("type") or "")
        content = msg.get("content")
        raw_calls = msg.get("tool_calls") or []
    else:
        role = str(getattr(msg, "type", "") or getattr(msg, "role", ""))
        content = getattr(msg, "content", None)
        raw_calls = getattr(msg, "tool_calls", None) or []

    calls: list[tuple[str, Any]] = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else None
        name = (fn or call).get("name")
        args = (fn or {}).get("arguments") if fn else call.get("args")
        if isinstance(name, str) and name:
            calls.append((name, args))
    return _ROLE_ALIASES.get(role.lower(), role.lower()), _message_text(content), calls


def _format_call(name: str, args: Any) -> str:
    if isinstance(args, dict) and args:
        rendered = ", ".join(f"{k}={v}" for k, v in args.items())
        return f"{name}({rendered[:200]})"
    if isinstance(args, str) and args.strip() not in ("", "{}"):
        return f"{name}({args[:200]})"
    return name


def _recent_tool_calls(messages: Sequence[Any]) -> list[tuple[str, Any]]:
    """Tool calls made since the second-to-last user message.

    That is the previous turn (what the assistant did before the user's
    latest message) plus the current turn (calls made while answering it).
    """
    user_seen = 0
    calls: list[tuple[str, Any]] = []
    for msg in reversed(messages):
        role, _text, msg_calls = _message_parts(msg)
        if role == "user":
            user_seen += 1
            if user_seen == 2:
                break
            continue
        calls[:0] = msg_calls
    return calls


def build_selection_query(messages: Sequence[Any], *, user_turns: int = 3) -> str:
    """Build the semantic-selection query from the recent conversation.

    Embedding models truncate long input, so the most telling parts come
    first: the latest user message, the assistant reply it answers, the
    tool calls of the previous and current turn, then up to
    ``user_turns - 1`` earlier user messages (newest first).  A follow-up
    such as "Yes, go ahead." therefore still selects the tools of the
    request it confirms.

    Args:
        messages: The conversation, as LangChain messages or
            ``{"role": ..., "content": ...}`` dicts.
        user_turns: How many of the most recent user messages to include.

    Returns:
        The query text (empty if the conversation has no user message).
    """
    users: list[str] = []
    assistant_reply = ""
    for msg in reversed(messages):
        role, text, _calls = _message_parts(msg)
        text = text.strip()
        if not text:
            continue
        if role == "user":
            users.append(text)
            if len(users) >= max(user_turns, 1):
                break
        elif role == "assistant" and len(users) == 1 and not assistant_reply:
            assistant_reply = text[:_MAX_ASSISTANT_CHARS]
    if not users:
        return ""

    parts = [users[0]]
    if assistant_reply:
        parts.append(assistant_reply)
    calls = _recent_tool_calls(messages)
    if calls:
        parts.append("Tools used: " + "; ".join(_format_call(n, a) for n, a in calls))
    parts.extend(users[1:])
    return "\n".join(parts)[:_MAX_QUERY_CHARS]


# ----------------------------------------------------------------------
# Tool selector
# ----------------------------------------------------------------------

#: Name of the fallback tool semantic selection adds.
REQUEST_MORE_TOOLS = "request_more_tools"


def _fallback_matches(
    index: ToolIndex, query: Any, tool_names: Any, top_k: int
) -> tuple[list[str], list[str]]:
    """``(found, unknown)`` tool names for a ``request_more_tools`` call."""
    if isinstance(tool_names, list) and tool_names:
        names = [n for n in dict.fromkeys(tool_names) if isinstance(n, str)]
        return [n for n in names if n in index], [n for n in names if n not in index]
    if isinstance(query, str) and query.strip():
        return [t.name for t in index.select(query, top_k=top_k)], []
    return index.all_tool_names, []


def _call_args(args: Any) -> dict[str, Any]:
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return {}
    return args if isinstance(args, dict) else {}


class _ToolSelector:
    """Decides which tools each model call is offered.

    Installed as the engine's tool selector (see
    :data:`promptise.engine.nodes.TOOL_SELECTOR_KEY`): before every model
    call the node's candidate tools are narrowed to

    * the ``semantic_top_k`` tools most relevant to the recent
      conversation (:func:`build_selection_query`),
    * the ``preserve_tools``,
    * tools called in the previous or current turn,
    * tools a ``request_more_tools`` call in the previous or current turn
      returned (recomputed from its arguments, so nothing is stored
      between calls and concurrent runs can't affect each other),

    while tools the index doesn't manage (the fallback itself, tools a
    custom graph brings) always pass through.
    """

    def __init__(self, index: ToolIndex, config: _ResolvedConfig) -> None:
        self._index = index
        self._config = config

    def selected_names(self, messages: Sequence[Any]) -> set[str]:
        """Names of the indexed tools to offer for *messages*."""
        query = build_selection_query(messages, user_turns=self._config.semantic_context_turns)
        selected = self._index.select(
            query,
            top_k=self._config.semantic_top_k,
            preserve=self._config.preserve_tools,
        )
        names = {t.name for t in selected}
        for name, args in _recent_tool_calls(messages):
            if name == REQUEST_MORE_TOOLS:
                parsed = _call_args(args)
                found, _unknown = _fallback_matches(
                    self._index,
                    parsed.get("query"),
                    parsed.get("tool_names"),
                    self._config.semantic_top_k,
                )
                names.update(found)
            else:
                names.add(name)
        return names

    def __call__(self, candidates: Sequence[BaseTool], state: Any) -> list[BaseTool]:
        keep = self.selected_names(getattr(state, "messages", None) or [])
        return [t for t in candidates if t.name not in self._index or t.name in keep]


# ======================================================================
# Fallback tool: request_more_tools
# ======================================================================


class _RequestMoreToolsArgs(BaseModel):
    query: str | None = Field(
        default=None,
        description="The capability you need, in a few words (e.g. 'read the audit log').",
    )
    tool_names: list[str] | None = Field(
        default=None,
        description="Exact names of tools to enable, if you already know them.",
    )


class _RequestMoreToolsTool(BaseTool):
    """Fallback tool that finds and unlocks tools semantic selection left out.

    Every tool it returns becomes callable on the agent's next model call
    in the same run: matches for ``query``, the named ``tool_names``, or —
    called without arguments — the whole catalogue.
    """

    name: str = REQUEST_MORE_TOOLS
    description: str = (
        "Call this when none of your current tools can do what is needed. "
        "Describe the capability in `query` (or give exact `tool_names`); "
        "the matching tools are returned and you can call them on your next "
        "step. Without arguments it lists and enables every available tool."
    )
    args_schema: type[BaseModel] = _RequestMoreToolsArgs

    _tool_index: ToolIndex = PrivateAttr()
    _top_k: int = PrivateAttr(default=8)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, tool_index: ToolIndex, top_k: int = 8, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tool_index = tool_index
        self._top_k = top_k

    async def _arun(
        self,
        query: str | None = None,
        tool_names: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        # The agent's tool selector recomputes the same matches from this
        # call's arguments and offers them from the next model call on.
        index = self._tool_index
        found, unknown = _fallback_matches(index, query, tool_names, self._top_k)
        listing = (
            index.summaries(found)
            if (tool_names or (query and query.strip()))
            else index.tool_summaries
        )

        parts = []
        if found:
            parts.append(
                f"{len(found)} of {len(index.all_tool_names)} tools are now available; "
                f"call them by name on your next step:\n\n{listing}"
            )
        if unknown:
            parts.append(
                "No tool is named: " + ", ".join(unknown) + ". Call request_more_tools "
                "with a `query` describing what you need to search for it."
            )
        return "\n\n".join(parts) or "No matching tools."

    def _run(self, **kwargs: Any) -> str:
        import anyio

        return anyio.run(lambda: self._arun(**kwargs))
