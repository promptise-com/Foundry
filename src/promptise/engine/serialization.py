"""Serialization for PromptGraph nodes and graphs.

Converts graphs to and from configuration dicts and YAML files without
losing information: every node setting that is data round-trips as data,
tools are stored by name and resolved again on load, and Python objects
(an ``output_schema`` model, a ``preprocessor``, an edge condition, an
``@node`` function) are stored as import references (``"module:QualName"``).
Anything that can't be represented — a lambda, a function defined inside
another function, a strategy or guard instance, a model object — makes
:func:`graph_to_config` / :func:`save_graph` raise
:class:`GraphSerializationError` naming the node and the setting, instead
of writing a file that silently loads as a different graph.

Example — save, then load with the same tools::

    save_graph(graph, "research-agent.yaml")

    graph = load_graph("research-agent.yaml", tools=agent.tools, refs=[Verdict])
    engine = PromptGraphEngine(graph=graph, model=my_model)

Loading resolves import references only from ``refs`` (and from the
``promptise`` package itself) unless ``allow_imports=True``: importing a
module runs its code, so only allow imports for files you trust.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import yaml

from .base import BaseNode, _FunctionalNode
from .graph import EdgeCondition, PromptGraph
from .state import NodeFlag

logger = logging.getLogger("promptise.engine")

#: Version of the config format written by :func:`graph_to_config`.
FORMAT_VERSION = 2


class GraphSerializationError(ValueError):
    """A graph can't be saved without losing information, or a config can't
    be loaded as the graph it describes."""


# ---------------------------------------------------------------------------
# Field specs
# ---------------------------------------------------------------------------

_DATA = "data"  # plain YAML data (str, int, float, bool, None, lists, dicts)
_TOOLS = "tools"  # list of tools, stored by name
_REF = "ref"  # Python object, stored as "module:QualName"
_MODEL = "model"  # model override: a model id string only
_NODE = "node"  # a nested node
_NODES = "nodes"  # a list (or name → node dict) of nested nodes
_GRAPH = "graph"  # a nested graph
_BRANCHES = "branches"  # FanOutNode's (node, overrides) pairs
_UNSUPPORTED = "unsupported"  # Python objects with state — must be empty to save


@dataclass(frozen=True)
class _Field:
    key: str  # config key == constructor keyword
    kind: str
    attr: str | None = None  # attribute name, when it differs from the key

    @property
    def attribute(self) -> str:
        return self.attr or self.key


_BASE_FIELDS = (
    _Field("instructions", _DATA),
    _Field("description", _DATA),
    _Field("transitions", _DATA),
    _Field("default_next", _DATA),
    _Field("max_iterations", _DATA),
    _Field("metadata", _DATA),
)

_PROMPT_FIELDS = (
    _Field("blocks", _UNSUPPORTED),
    _Field("strategy", _UNSUPPORTED),
    _Field("perspective", _UNSUPPORTED),
    _Field("tools", _TOOLS),
    _Field("tool_choice", _DATA),
    _Field("output_schema", _REF),
    _Field("guards", _UNSUPPORTED),
    _Field("model_override", _MODEL),
    _Field("context_layers", _DATA),
    _Field("max_tokens", _DATA),
    _Field("temperature", _DATA),
    _Field("input_keys", _DATA),
    _Field("output_key", _DATA),
    _Field("inherit_context_from", _DATA),
    _Field("context_scope", _DATA),
    _Field("auto_ledger_after", _DATA),
    _Field("preprocessor", _REF),
    _Field("postprocessor", _REF),
    _Field("include_observations", _DATA),
    _Field("include_plan", _DATA),
    _Field("include_reflections", _DATA),
)

# Constructor parameters deliberately not stored as fields: their effect is
# captured by ``flags`` (inject_tools, is_entry, is_terminal) or baked into
# the stored ``instructions`` (criteria, focus_areas).
_ABSORBED_PARAMS = frozenset(
    {"name", "flags", "inject_tools", "is_entry", "is_terminal", "criteria", "focus_areas"}
)
_FLAG_PARAMS = {
    "inject_tools": NodeFlag.INJECT_TOOLS,
    "is_entry": NodeFlag.ENTRY,
    "is_terminal": NodeFlag.TERMINAL,
}


@dataclass(frozen=True)
class _Spec:
    fields: tuple[_Field, ...]
    # Base fields the class computes itself and doesn't accept as keywords.
    skip_base: frozenset[str] = frozenset()


# Registry of node types: type name → class, and class → field spec.
_NODE_TYPES: dict[str, type] = {}
_SPECS: dict[type, _Spec] = {}


def register_node_type(name: str, cls: type) -> None:
    """Register a node type for YAML/dict (de)serialization.

    A registered subclass of a built-in node keeps the built-in's settings.
    A class whose constructor takes parameters of its own must define
    ``to_config(self) -> dict`` (plain data) and a ``from_config(cls,
    config)`` classmethod; otherwise saving it raises
    :class:`GraphSerializationError`, since those parameters would be lost.
    """
    _ensure_registry()
    _NODE_TYPES[name] = cls


def _ensure_registry() -> None:
    """Lazy-populate the registry on first use."""
    if _SPECS:
        return
    from .code_action import CodeActionNode
    from .nodes import (
        AutonomousNode,
        GuardNode,
        HumanNode,
        LoopNode,
        ParallelNode,
        PromptNode,
        RouterNode,
        SubgraphNode,
        ToolNode,
        TransformNode,
    )
    from .reasoning_nodes import (
        CritiqueNode,
        FanOutNode,
        JustifyNode,
        ObserveNode,
        PlanNode,
        ReflectNode,
        RetryNode,
        SynthesizeNode,
        ThinkNode,
        ValidateNode,
    )

    prompt = _Spec(_PROMPT_FIELDS)
    _SPECS.update(
        {
            PromptNode: prompt,
            ThinkNode: prompt,
            ObserveNode: prompt,
            JustifyNode: prompt,
            SynthesizeNode: prompt,
            ReflectNode: _Spec(_PROMPT_FIELDS + (_Field("review_depth", _DATA, "_review_depth"),)),
            CritiqueNode: _Spec(
                _PROMPT_FIELDS + (_Field("severity_threshold", _DATA, "_severity_threshold"),)
            ),
            PlanNode: _Spec(
                _PROMPT_FIELDS
                + (
                    _Field("max_subgoals", _DATA, "_max_subgoals"),
                    _Field("quality_threshold", _DATA, "_quality_threshold"),
                )
            ),
            ValidateNode: _Spec(
                _PROMPT_FIELDS
                + (_Field("on_pass", _DATA, "_on_pass"), _Field("on_fail", _DATA, "_on_fail"))
            ),
            ToolNode: _Spec(
                (
                    _Field("tools", _TOOLS),
                    _Field("validate_inputs", _DATA),
                    _Field("deduplicate", _DATA),
                    _Field("max_result_chars", _DATA),
                    _Field("tool_selector", _REF),
                )
            ),
            RouterNode: _Spec(
                (
                    _Field("routes", _DATA),
                    _Field("context_blocks", _UNSUPPORTED),
                    _Field("model_override", _MODEL),
                )
            ),
            GuardNode: _Spec(
                (
                    _Field("guards", _UNSUPPORTED),
                    _Field("target_key", _DATA),
                    _Field("on_pass", _DATA),
                    _Field("on_fail", _DATA),
                )
            ),
            ParallelNode: _Spec(
                (
                    _Field("nodes", _NODES, "child_nodes"),
                    _Field("merge_strategy", _DATA),
                    _Field("merge_fn", _REF),
                )
            ),
            LoopNode: _Spec(
                (
                    _Field("body_node", _NODE),
                    _Field("condition", _REF),
                    _Field("max_loop_iterations", _DATA),
                )
            ),
            HumanNode: _Spec(
                (
                    _Field("prompt_template", _DATA),
                    _Field("timeout", _DATA),
                    _Field("on_approve", _DATA),
                    _Field("on_deny", _DATA),
                    _Field("on_timeout", _DATA),
                )
            ),
            TransformNode: _Spec((_Field("transform", _REF), _Field("output_key", _DATA))),
            SubgraphNode: _Spec((_Field("subgraph", _GRAPH), _Field("inherit_state", _DATA))),
            AutonomousNode: _Spec(
                (
                    _Field("node_pool", _NODES),
                    _Field("planner_instructions", _DATA),
                    _Field("allow_repeat", _DATA),
                    _Field("max_steps", _DATA),
                    _Field("entry_node", _DATA),
                    _Field("terminal_nodes", _DATA),
                )
            ),
            RetryNode: _Spec(
                (
                    _Field("wrapped_node", _NODE),
                    _Field("max_retries", _DATA),
                    _Field("backoff_factor", _DATA),
                ),
                skip_base=frozenset({"description"}),
            ),
            FanOutNode: _Spec(
                (_Field("branches", _BRANCHES), _Field("merge_strategy", _DATA)),
                skip_base=frozenset({"description"}),
            ),
            CodeActionNode: _Spec(
                (
                    _Field("tools", _TOOLS),
                    _Field("system_prompt", _DATA),
                    _Field("blocks", _UNSUPPORTED),
                    _Field("sandbox_factory", _REF),
                    _Field("model_override", _MODEL),
                    _Field("max_repairs", _DATA),
                    _Field("result_marker", _DATA),
                    _Field("exec_timeout", _DATA),
                    _Field("max_tool_calls", _DATA),
                )
            ),
        }
    )
    _NODE_TYPES.update(
        {
            "prompt": PromptNode,
            "tool": ToolNode,
            "router": RouterNode,
            "guard": GuardNode,
            "parallel": ParallelNode,
            "loop": LoopNode,
            "human": HumanNode,
            "transform": TransformNode,
            "subgraph": SubgraphNode,
            "autonomous": AutonomousNode,
            "code_action": CodeActionNode,
            # Reasoning nodes
            "think": ThinkNode,
            "reflect": ReflectNode,
            "observe": ObserveNode,
            "justify": JustifyNode,
            "critique": CritiqueNode,
            "plan": PlanNode,
            "synthesize": SynthesizeNode,
            "validate": ValidateNode,
            "retry": RetryNode,
            "fan_out": FanOutNode,
        }
    )


def _spec_for(cls: type) -> _Spec:
    """The field spec of *cls*: its own, or its nearest built-in ancestor's."""
    for klass in cls.__mro__:
        if klass in _SPECS:
            return _SPECS[klass]
    return _Spec(())


def _type_name(cls: type) -> str | None:
    for name, registered in _NODE_TYPES.items():
        if registered is cls:
            return name
    return None


def _constructor_params(cls: type) -> list[str]:
    """Keyword parameters ``cls(...)`` accepts, following ``**kwargs`` up the MRO."""
    params: list[str] = []
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        has_var_kw = False
        for p in inspect.signature(init).parameters.values():
            if p.name == "self":
                continue
            if p.kind is inspect.Parameter.VAR_KEYWORD:
                has_var_kw = True
            elif p.kind is not inspect.Parameter.VAR_POSITIONAL and p.name not in params:
                params.append(p.name)
        if not has_var_kw:
            break
    return params


_DEFAULT_INSTANCES: dict[type, Any] = {}


def _default_instance(cls: type) -> Any:
    """A node of *cls* built from a name alone, or ``None`` when the class
    has required arguments."""
    if cls not in _DEFAULT_INSTANCES:
        try:
            _DEFAULT_INSTANCES[cls] = cls("__default__")
        except Exception:
            _DEFAULT_INSTANCES[cls] = None
    return _DEFAULT_INSTANCES[cls]


def _default_of(cls: type, field: _Field) -> Any:
    """What a node of *cls* built with no arguments has for *field* — the
    value loading reproduces when the field is left out."""
    instance = _default_instance(cls)
    if instance is not None:
        return getattr(instance, field.attribute, None)
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        p = inspect.signature(init).parameters.get(field.key) if init else None
        if p is not None and p.default is not inspect.Parameter.empty:
            return p.default
    return None


def _is_default(cls: type, field: _Field, value: Any) -> bool:
    default = _default_of(cls, field)
    if _is_empty(value):
        return field.kind != _DATA or _is_empty(default)
    return field.kind == _DATA and type(value) is type(default) and value == default


def _has_own_config_hooks(cls: type) -> bool:
    """Whether a custom class provides ``to_config``/``from_config``."""
    return callable(getattr(cls, "to_config", None)) and callable(getattr(cls, "from_config", None))


# ---------------------------------------------------------------------------
# Encoding (save)
# ---------------------------------------------------------------------------


def _is_plain(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_plain(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_plain(v) for k, v in value.items())
    return False


def _plain(value: Any) -> Any:
    """Tuples → lists, recursively (YAML-safe)."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, (list, tuple, dict, set)) and not value)


def _import_ref(ref: str) -> Any:
    module_name, _, qualname = ref.partition(":")
    if not module_name or not qualname:
        raise GraphSerializationError(f"Invalid reference {ref!r}: expected 'module:QualName'")
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _ref_of(obj: Any, where: str) -> str:
    """``"module:QualName"`` for an importable object, else raise."""
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None)
    if not module or not qualname or "<" in qualname:
        if qualname and "<" in qualname:
            reason = "it is a lambda or defined inside a function"
        else:
            reason = f"a {type(obj).__name__} instance has no import path"
        raise GraphSerializationError(
            f"{where}: can't be saved — {reason}. Define it at module level in an "
            "importable module, or set it in code after loading the graph."
        )
    ref = f"{module}:{qualname}"
    try:
        resolved = _import_ref(ref)
    except Exception as exc:
        raise GraphSerializationError(f"{where}: {ref!r} is not importable ({exc})") from exc
    if resolved is not obj:
        raise GraphSerializationError(
            f"{where}: {ref!r} does not import back to the same object, so it can't be saved."
        )
    return ref


def _encode(field: _Field, value: Any, where: str) -> Any:
    kind = field.kind
    if kind == _DATA:
        if not _is_plain(value):
            raise GraphSerializationError(
                f"{where}: {type(value).__name__} value is not plain YAML data"
            )
        return _plain(value)
    if kind == _TOOLS:
        return [t.name for t in value]
    if kind == _REF:
        return _ref_of(value, where)
    if kind == _MODEL:
        if isinstance(value, str):
            return value
        raise GraphSerializationError(
            f"{where}: a model object can't be saved (it may carry credentials). "
            "Use a model id string such as 'openai:gpt-5-mini'."
        )
    if kind == _NODE:
        return node_to_config(value)
    if kind == _NODES:
        nodes = value.values() if isinstance(value, dict) else value
        return [node_to_config(n) for n in nodes]
    if kind == _GRAPH:
        return _graph_body(value)
    if kind == _BRANCHES:
        branches = []
        for node, overrides in value:
            if not _is_plain(overrides):
                raise GraphSerializationError(f"{where}: branch overrides must be plain data")
            branches.append({"node": node_to_config(node), "overrides": _plain(overrides)})
        return branches
    # _UNSUPPORTED with a non-empty value
    raise GraphSerializationError(
        f"{where}: {type(value).__name__} objects can't be saved to YAML. "
        "Remove it before saving and set it in code after loading the graph."
    )


def node_to_config(node: BaseNode) -> dict[str, Any]:
    """Convert a node to a configuration dict (plain data).

    Args:
        node: The node to serialize.

    Returns:
        A dict suitable for YAML serialization.

    Raises:
        GraphSerializationError: When a setting of the node can't be
            represented, so the saved config would load as a different node.
    """
    _ensure_registry()
    where = f"Node {node.name!r}"

    # An @node function is saved as a reference to the module-level node.
    if isinstance(node, _FunctionalNode):
        return {"name": node.name, "type": "ref", "ref": _ref_of(node, where)}

    cls = type(node)
    type_name = _type_name(cls)
    if type_name is None:
        raise GraphSerializationError(
            f"{where}: node class {cls.__qualname__} is not registered. Call "
            f"register_node_type('<type>', {cls.__name__}) before saving."
        )

    spec = _spec_for(cls)
    config: dict[str, Any] = {"name": node.name, "type": type_name}

    for field in (f for f in _BASE_FIELDS if f.key not in spec.skip_base):
        value = getattr(node, field.attribute)
        if field.key == "description" and value == node.instructions[:80]:
            default_node = _default_instance(cls)
            if default_node is not None and default_node.description == "":
                continue  # loading derives it from instructions again
        if not _is_default(cls, field, value):
            config[field.key] = _encode(field, value, f"{where} {field.key}")

    for field in spec.fields:
        value = getattr(node, field.attribute)
        if not _is_default(cls, field, value):
            config[field.key] = _encode(field, value, f"{where} {field.key}")

    if _has_own_config_hooks(cls) and cls not in _SPECS:
        extra = node.to_config()  # type: ignore[attr-defined]
        if not isinstance(extra, dict) or not _is_plain(extra):
            raise GraphSerializationError(f"{where}: to_config() must return plain data")
        config.update(_plain(extra))
    else:
        known = {f.key for f in spec.fields} | {f.key for f in _BASE_FIELDS} | _ABSORBED_PARAMS
        missing = [p for p in _constructor_params(cls) if p not in known]
        if missing:
            raise GraphSerializationError(
                f"{where}: {cls.__qualname__} takes {missing}, which the serializer "
                "doesn't know how to save. Define to_config()/from_config() on the class."
            )

    flags = []
    for flag in node.flags:
        if not isinstance(flag, NodeFlag):
            raise GraphSerializationError(f"{where}: flag {flag!r} is not a NodeFlag")
        flags.append(flag.value)
    if flags:
        config["flags"] = sorted(flags)
    return config


def _edge_to_config(edge: Any) -> dict[str, Any]:
    config: dict[str, Any] = {"from": edge.from_node, "to": edge.to_node}
    if edge.label:
        config["label"] = edge.label
    if edge.priority:
        config["priority"] = edge.priority
    condition = edge.condition
    if isinstance(condition, EdgeCondition):
        cond: dict[str, Any] = {"kind": condition.kind}
        if condition.kind == "output":
            if not _is_plain(condition.value):
                raise GraphSerializationError(
                    f"Edge {edge.from_node} → {edge.to_node}: condition value is not plain data"
                )
            cond.update(key=condition.key, value=_plain(condition.value))
        elif condition.kind == "confidence":
            cond["min_confidence"] = condition.min_confidence
        config["condition"] = cond
    elif condition is not None:
        config["condition"] = {
            "ref": _ref_of(condition, f"Edge {edge.from_node} → {edge.to_node} condition")
        }
    return config


def _graph_body(graph: PromptGraph) -> dict[str, Any]:
    return {
        "name": graph.name,
        "mode": graph.mode,
        "entry": graph.entry,
        "nodes": {
            name: {k: v for k, v in node_to_config(node).items() if k != "name"}
            for name, node in graph.nodes.items()
        },
        "edges": [_edge_to_config(edge) for edge in graph.edges],
    }


def graph_to_config(graph: PromptGraph) -> dict[str, Any]:
    """Convert a PromptGraph to a configuration dict (plain data).

    Args:
        graph: The graph to serialize.

    Returns:
        A dict suitable for YAML serialization, with a ``version`` key.

    Raises:
        GraphSerializationError: When any node or edge setting can't be
            represented (see :func:`node_to_config`).
    """
    return {"version": FORMAT_VERSION, **_graph_body(graph)}


# ---------------------------------------------------------------------------
# Decoding (load)
# ---------------------------------------------------------------------------


@dataclass
class _Resolver:
    tools: dict[str, Any]
    refs: dict[str, Any]
    allow_imports: bool

    @classmethod
    def build(
        cls,
        tools: Iterable[Any] | Mapping[str, Any] | None,
        refs: Iterable[Any] | Mapping[str, Any] | None,
        allow_imports: bool,
    ) -> _Resolver:
        tool_map = dict(tools) if isinstance(tools, Mapping) else {t.name: t for t in (tools or [])}
        if isinstance(refs, Mapping):
            ref_map = dict(refs)
        else:
            ref_map = {}
            for obj in refs or []:
                ref_map[f"{obj.__module__}:{obj.__qualname__}"] = obj
        return cls(tool_map, ref_map, allow_imports)

    def tool_list(self, names: Any, where: str) -> list[Any]:
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise GraphSerializationError(f"{where}: tools must be a list of tool names")
        missing = [n for n in names if n not in self.tools]
        if missing:
            raise GraphSerializationError(
                f"{where}: tools {missing} were not provided. Pass them to "
                "load_graph(..., tools=agent.tools), or give the node "
                "flags: [inject_tools] to receive the agent's tools at run time."
            )
        return [self.tools[n] for n in names]

    def ref(self, ref: Any, where: str) -> Any:
        if not isinstance(ref, str):
            raise GraphSerializationError(f"{where}: expected a 'module:QualName' reference")
        if ref in self.refs:
            return self.refs[ref]
        module = ref.partition(":")[0]
        if self.allow_imports or module == "promptise" or module.startswith("promptise."):
            try:
                return _import_ref(ref)
            except GraphSerializationError:
                raise
            except Exception as exc:
                raise GraphSerializationError(f"{where}: can't import {ref!r} ({exc})") from exc
        raise GraphSerializationError(
            f"{where}: {ref!r} is a Python reference. Pass the object in "
            "load_graph(..., refs=[...]), or allow_imports=True to import it "
            "(only for files you trust — importing runs the module's code)."
        )


def _decode(field: _Field, value: Any, where: str, resolver: _Resolver) -> Any:
    kind = field.kind
    if kind == _DATA:
        return value
    if kind == _TOOLS:
        return resolver.tool_list(value, where)
    if kind == _REF:
        return None if value is None else resolver.ref(value, where)
    if kind == _MODEL:
        if value is None or isinstance(value, str):
            return value
        raise GraphSerializationError(f"{where}: model_override must be a model id string")
    if kind == _NODE:
        return _node_from_config(value, resolver)
    if kind == _NODES:
        if not isinstance(value, list):
            raise GraphSerializationError(f"{where}: expected a list of nodes")
        return [_node_from_config(v, resolver) for v in value]
    if kind == _GRAPH:
        return _graph_from_config(value, resolver)
    if kind == _BRANCHES:
        if not isinstance(value, list):
            raise GraphSerializationError(f"{where}: expected a list of branches")
        return [
            (_node_from_config(b["node"], resolver), dict(b.get("overrides") or {})) for b in value
        ]
    if _is_empty(value):
        return None
    raise GraphSerializationError(
        f"{where}: can't be loaded from YAML — set it in code after loading the graph."
    )


def _node_from_config(config: Any, resolver: _Resolver) -> BaseNode:
    _ensure_registry()
    if not isinstance(config, dict):
        raise GraphSerializationError(f"Node config must be a mapping, got {type(config).__name__}")
    cfg = dict(config)
    name = cfg.pop("name", None)
    if not isinstance(name, str) or not name:
        raise GraphSerializationError(f"Node config {config!r} has no name")
    where = f"Node {name!r}"
    node_type = cfg.pop("type", "prompt")

    if node_type == "ref":
        node = resolver.ref(cfg.pop("ref", None), where)
        if cfg:
            raise GraphSerializationError(f"{where}: unexpected fields {sorted(cfg)}")
        if not isinstance(node, BaseNode) or node.name != name:
            raise GraphSerializationError(f"{where}: reference is not the node {name!r}")
        return node

    cls = _NODE_TYPES.get(node_type)
    if cls is None:
        raise GraphSerializationError(
            f"{where}: unknown node type {node_type!r}. Available: {sorted(_NODE_TYPES)}"
        )

    flag_values = cfg.pop("flags", None)
    implied = {flag for key, flag in _FLAG_PARAMS.items() if cfg.pop(key, False)}

    if _has_own_config_hooks(cls) and cls not in _SPECS:
        node = cls.from_config({"name": name, **cfg})  # type: ignore[attr-defined]
    else:
        spec = _spec_for(cls)
        fields = {f.key: f for f in _BASE_FIELDS if f.key not in spec.skip_base}
        fields.update({f.key: f for f in spec.fields})
        kwargs: dict[str, Any] = {}
        for key, value in cfg.items():
            field = fields.get(key)
            if field is None:
                raise GraphSerializationError(
                    f"{where}: unknown field {key!r} for node type {node_type!r}"
                )
            decoded = _decode(field, value, f"{where} {key}", resolver)
            if decoded is not None or field.kind == _DATA:
                kwargs[key] = decoded
        try:
            node = cls(name, **kwargs)
        except TypeError as exc:
            raise GraphSerializationError(f"{where}: {exc}") from exc

    if flag_values is not None:
        try:
            node.flags = {NodeFlag(v) for v in flag_values} | implied
        except ValueError as exc:
            raise GraphSerializationError(f"{where}: {exc}") from exc
    else:
        node.flags |= implied
    return node


def _graph_from_config(config: Any, resolver: _Resolver) -> PromptGraph:
    if not isinstance(config, dict):
        raise GraphSerializationError(
            f"Graph config must be a mapping, got {type(config).__name__}"
        )
    version = config.get("version", FORMAT_VERSION)
    if not isinstance(version, int) or version > FORMAT_VERSION:
        raise GraphSerializationError(
            f"Graph config version {version!r} is newer than this Promptise "
            f"supports ({FORMAT_VERSION}). Upgrade promptise."
        )
    graph = PromptGraph(name=config.get("name", "graph"), mode=config.get("mode", "autonomous"))

    for name, node_config in (config.get("nodes") or {}).items():
        node_config = dict(node_config or {})
        node_config.setdefault("name", name)
        graph.add_node(_node_from_config(node_config, resolver))

    for edge_config in config.get("edges") or []:
        condition: Any = None
        cond = edge_config.get("condition")
        where = f"Edge {edge_config.get('from')} → {edge_config.get('to')}"
        if isinstance(cond, dict) and "ref" in cond:
            condition = resolver.ref(cond["ref"], f"{where} condition")
        elif isinstance(cond, dict):
            try:
                condition = EdgeCondition(**cond)
            except (TypeError, ValueError) as exc:
                raise GraphSerializationError(f"{where}: {exc}") from exc
        elif cond is not None:
            raise GraphSerializationError(f"{where}: condition must be a mapping")
        graph.add_edge(
            edge_config["from"],
            edge_config["to"],
            condition=condition,
            label=edge_config.get("label", ""),
            priority=edge_config.get("priority", 0),
        )

    entry = config.get("entry")
    if entry is not None:
        if not graph.has_node(entry):
            raise GraphSerializationError(f"Entry node {entry!r} is not in the graph")
        graph.set_entry(entry)
    return graph


def node_from_config(
    config: dict[str, Any],
    *,
    tools: Iterable[Any] | Mapping[str, Any] | None = None,
    refs: Iterable[Any] | Mapping[str, Any] | None = None,
    allow_imports: bool = False,
) -> BaseNode:
    """Create a node from a configuration dict.

    The ``type`` field picks the node class (default ``"prompt"``).

    Args:
        config: Node configuration with at least ``name``.
        tools: Tools to resolve tool names against (a list, or a name →
            tool mapping) — typically ``agent.tools``.
        refs: Objects the config references by import path (schemas,
            processors, conditions), as a list or a ``"module:QualName"`` →
            object mapping. References into ``promptise`` resolve without it.
        allow_imports: Import any other reference. Importing runs the
            module's code — only for configs you trust.

    Returns:
        A ``BaseNode`` instance.

    Raises:
        GraphSerializationError: For an unknown type or field, a tool that
            was not provided, or a reference that can't be resolved.
    """
    return _node_from_config(config, _Resolver.build(tools, refs, allow_imports))


def graph_from_config(
    config: dict[str, Any],
    *,
    tools: Iterable[Any] | Mapping[str, Any] | None = None,
    refs: Iterable[Any] | Mapping[str, Any] | None = None,
    allow_imports: bool = False,
) -> PromptGraph:
    """Create a PromptGraph from a configuration dict.

    Args:
        config: Graph configuration with ``name``, ``entry``, ``nodes``,
            and optional ``mode`` and ``edges``.
        tools: See :func:`node_from_config`.
        refs: See :func:`node_from_config`.
        allow_imports: See :func:`node_from_config`.

    Returns:
        A ``PromptGraph`` instance.
    """
    return _graph_from_config(config, _Resolver.build(tools, refs, allow_imports))


def load_graph(
    path: str,
    *,
    tools: Iterable[Any] | Mapping[str, Any] | None = None,
    refs: Iterable[Any] | Mapping[str, Any] | None = None,
    allow_imports: bool = False,
) -> PromptGraph:
    """Load a PromptGraph from a YAML file.

    Args:
        path: Path to the YAML file.
        tools: See :func:`node_from_config`.
        refs: See :func:`node_from_config`.
        allow_imports: See :func:`node_from_config`.

    Returns:
        A ``PromptGraph`` instance.
    """
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise GraphSerializationError(f"Expected a mapping in {path}, got {type(config).__name__}")

    # Support both top-level graph config and nested under "graph" key
    if "graph" in config:
        config = config["graph"]

    return graph_from_config(config, tools=tools, refs=refs, allow_imports=allow_imports)


def save_graph(graph: PromptGraph, path: str) -> None:
    """Save a PromptGraph to a YAML file.

    The file is written only when the whole graph can be represented.

    Args:
        graph: The graph to save.
        path: Output file path.

    Raises:
        GraphSerializationError: When a node or edge setting can't be
            represented (see :func:`node_to_config`).
    """
    config = graph_to_config(graph)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
