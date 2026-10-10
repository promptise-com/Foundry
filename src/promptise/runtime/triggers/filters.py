"""Safe filter expressions for triggers.

``TriggerConfig.filter_expression`` decides, before any LLM call, whether a
trigger event should run the agent.  It is either a Python callable that
receives the :class:`~promptise.runtime.triggers.base.TriggerEvent`, or a
string in a small expression language.

The string form is parsed with :mod:`ast` and evaluated by a whitelist
interpreter — it is **never** passed to ``eval``.  Only these constructs are
allowed:

* Literals: strings, numbers, ``True`` / ``False`` / ``None``, lists,
  tuples and sets of literals.
* Names: ``payload``, ``metadata``, ``trigger_type``, ``trigger_id``,
  ``event_id``.  Any other bare name is looked up as a key of the payload,
  so ``action == 'opened'`` is shorthand for ``payload['action'] == 'opened'``.
* Lookups: ``payload['key']``, ``payload['items'][0]`` and the dotted form
  ``payload.pull_request.state``.  A missing key yields ``None`` instead of
  raising.
* Comparisons: ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``in``,
  ``not in``, ``is``, ``is not`` (chains allowed).
* Boolean logic: ``and``, ``or``, ``not``.
* Functions: ``len(x)``, ``lower(x)``, ``upper(x)``, ``str(x)``,
  ``int(x)``, ``float(x)``, ``bool(x)``, ``startswith(x, prefix)``,
  ``endswith(x, suffix)``, ``contains(x, item)`` and the string methods
  ``x.lower()``, ``x.upper()``, ``x.strip()``, ``x.startswith(...)``,
  ``x.endswith(...)``.

Everything else — attribute access on non-dict objects, arithmetic,
comprehensions, lambdas, imports, dunder names — is rejected when the
expression is compiled, so a bad expression fails at configuration time
rather than on the first event.

Example::

    from promptise.runtime.triggers.filters import compile_filter

    accept = compile_filter("payload['action'] == 'opened' and 'bug' in labels")
    accept(event)  # True / False
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Callable
from typing import Any

from .base import TriggerEvent

logger = logging.getLogger(__name__)

__all__ = ["EventFilter", "FilterExpressionError", "compile_filter"]

#: A compiled filter: returns ``True`` when the event should run the agent.
EventFilter = Callable[[TriggerEvent], bool]

#: Upper bounds that keep a filter cheap to evaluate.
MAX_EXPRESSION_LENGTH = 2000
MAX_NODES = 200

_CONSTANT_NAMES = {"True": True, "False": False, "None": None}


def _contains(container: Any, item: Any) -> bool:
    try:
        return item in container
    except TypeError:
        return False


def _number(kind: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Wrap int()/float() so a huge numeric string can't make parsing slow."""

    def convert(value: Any) -> Any:
        if isinstance(value, str) and len(value) > 64:
            raise ValueError("numeric string too long")
        return kind(value)

    return convert


_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "len": len,
    "str": str,
    "int": _number(int),
    "float": _number(float),
    "bool": bool,
    "lower": lambda s: str(s).lower(),
    "upper": lambda s: str(s).upper(),
    "startswith": lambda s, prefix: str(s).startswith(prefix),
    "endswith": lambda s, suffix: str(s).endswith(suffix),
    "contains": _contains,
}
_STR_METHODS = frozenset({"lower", "upper", "strip", "startswith", "endswith"})

_COMPARE_OPS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: lambda a, b: a == b,
    ast.NotEq: lambda a, b: a != b,
    ast.Lt: lambda a, b: a < b,
    ast.LtE: lambda a, b: a <= b,
    ast.Gt: lambda a, b: a > b,
    ast.GtE: lambda a, b: a >= b,
    ast.In: lambda a, b: _contains(b, a),
    ast.NotIn: lambda a, b: not _contains(b, a),
    ast.Is: lambda a, b: a is b,
    ast.IsNot: lambda a, b: a is not b,
}


class FilterExpressionError(ValueError):
    """A filter expression is malformed or uses a construct that is not allowed."""


# ---------------------------------------------------------------------------
# Validation (compile time)
# ---------------------------------------------------------------------------


def _check(node: ast.AST, expr: str) -> None:
    """Reject any node outside the whitelist."""

    def fail(why: str) -> FilterExpressionError:
        return FilterExpressionError(f"Invalid filter_expression {expr!r}: {why}")

    if isinstance(node, ast.Expression):
        _check(node.body, expr)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (str, int, float, bool, type(None))):
            raise fail(f"unsupported literal {node.value!r}")
    elif isinstance(node, ast.Name):
        if node.id.startswith("_"):
            raise fail(f"name {node.id!r} is not allowed")
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for elt in node.elts:
            _check(elt, expr)
    elif isinstance(node, ast.BoolOp):
        for value in node.values:
            _check(value, expr)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.Not, ast.USub)):
            raise fail("only 'not' and unary '-' are allowed")
        _check(node.operand, expr)
    elif isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _COMPARE_OPS:
                raise fail(f"operator {type(op).__name__} is not allowed")
        _check(node.left, expr)
        for comparator in node.comparators:
            _check(comparator, expr)
    elif isinstance(node, ast.Subscript):
        if isinstance(node.slice, ast.Slice):
            raise fail("slices are not allowed")
        _check(node.value, expr)
        _check(node.slice, expr)
    elif isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            raise fail(f"attribute {node.attr!r} is not allowed")
        _check(node.value, expr)
    elif isinstance(node, ast.Call):
        if node.keywords:
            raise fail("keyword arguments are not allowed")
        func = node.func
        if isinstance(func, ast.Name):
            if func.id not in _FUNCTIONS:
                allowed = ", ".join(sorted(_FUNCTIONS))
                raise fail(f"function {func.id!r} is not allowed (allowed: {allowed})")
        elif isinstance(func, ast.Attribute):
            if func.attr not in _STR_METHODS:
                allowed = ", ".join(sorted(_STR_METHODS))
                raise fail(f"method {func.attr!r} is not allowed (allowed: {allowed})")
            _check(func.value, expr)
        else:
            raise fail("only named functions can be called")
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise fail("star arguments are not allowed")
            _check(arg, expr)
    else:
        raise fail(f"{type(node).__name__} is not allowed")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _lookup(container: Any, key: Any) -> Any:
    """Dict/list lookup that yields ``None`` instead of raising."""
    if isinstance(container, dict):
        return container.get(key)
    if isinstance(container, (list, tuple, str)) and isinstance(key, int):
        try:
            return container[key]
        except IndexError:
            return None
    return None


def _eval(node: ast.AST, names: dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, names)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        if node.id in _CONSTANT_NAMES:
            return _CONSTANT_NAMES[node.id]
        return _lookup(names.get("payload"), node.id)
    if isinstance(node, ast.List):
        return [_eval(e, names) for e in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_eval(e, names) for e in node.elts)
    if isinstance(node, ast.Set):
        return {_eval(e, names) for e in node.elts}
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result: Any = True
            for value in node.values:
                result = _eval(value, names)
                if not result:
                    return result
            return result
        result = False
        for value in node.values:
            result = _eval(value, names)
            if result:
                return result
        return result
    if isinstance(node, ast.UnaryOp):
        operand = _eval(node.operand, names)
        return (not operand) if isinstance(node.op, ast.Not) else -operand
    if isinstance(node, ast.Compare):
        left = _eval(node.left, names)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = _eval(comparator, names)
            if not _COMPARE_OPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Subscript):
        return _lookup(_eval(node.value, names), _eval(node.slice, names))
    if isinstance(node, ast.Attribute):
        return _lookup(_eval(node.value, names), node.attr)
    if isinstance(node, ast.Call):
        args = [_eval(a, names) for a in node.args]
        if isinstance(node.func, ast.Name):
            return _FUNCTIONS[node.func.id](*args)
        assert isinstance(node.func, ast.Attribute)
        target = _eval(node.func.value, names)
        if not isinstance(target, str):
            raise TypeError(f".{node.func.attr}() needs a string, got {type(target).__name__}")
        return getattr(target, node.func.attr)(*args)
    raise FilterExpressionError(f"unsupported node {type(node).__name__}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_filter_expression(expr: str) -> ast.Expression:
    """Parse and whitelist-check *expr*.

    Returns:
        The parsed expression tree.

    Raises:
        FilterExpressionError: If the expression is malformed or uses a
            construct outside the expression language.
    """
    if not expr.strip():
        raise FilterExpressionError("filter_expression is empty")
    if len(expr) > MAX_EXPRESSION_LENGTH:
        raise FilterExpressionError(
            f"filter_expression is longer than {MAX_EXPRESSION_LENGTH} characters"
        )
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as exc:
        raise FilterExpressionError(f"Invalid filter_expression {expr!r}: {exc.msg}") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise FilterExpressionError(f"filter_expression has more than {MAX_NODES} elements")
    _check(tree, expr)
    return tree


def compile_filter(expression: str | Callable[[TriggerEvent], Any]) -> EventFilter:
    """Turn a ``filter_expression`` into a predicate over trigger events.

    Args:
        expression: An expression string (see the module docs) or a callable
            that takes a :class:`TriggerEvent` and returns a truthy value.

    Returns:
        A function returning ``True`` when the event should run the agent.
        Evaluation errors (for example comparing ``None < 3``) count as
        *no match*: the event is skipped and a warning is logged.

    Raises:
        FilterExpressionError: If a string expression is invalid.
    """
    if callable(expression):
        func = expression

        def _call(event: TriggerEvent) -> bool:
            try:
                return bool(func(event))
            except Exception as exc:
                logger.warning("filter callable raised %r; skipping event %s", exc, event.event_id)
                return False

        return _call

    tree = validate_filter_expression(expression)

    def _match(event: TriggerEvent) -> bool:
        names = {
            "payload": event.payload,
            "metadata": event.metadata,
            "trigger_type": event.trigger_type,
            "trigger_id": event.trigger_id,
            "event_id": event.event_id,
        }
        try:
            return bool(_eval(tree, names))
        except Exception as exc:
            logger.warning(
                "filter_expression %r could not be evaluated (%s); skipping event %s",
                expression,
                exc,
                event.event_id,
            )
            return False

    return _match
