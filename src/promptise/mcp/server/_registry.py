"""Internal registries for tools, resources, and prompts."""

from __future__ import annotations

import re
from collections.abc import Callable
from urllib.parse import unquote

from ._types import PromptDef, ResourceDef, ToolDef

#: Called after a registration changes the registry's listing.
OnChange = Callable[[], None]


class ToolRegistry:
    """Stores registered tool definitions."""

    def __init__(self, *, on_change: OnChange | None = None) -> None:
        self._tools: dict[str, ToolDef] = {}
        self.on_change = on_change

    def register(self, tool_def: ToolDef) -> None:
        if tool_def.name in self._tools:
            raise ValueError(f"Tool '{tool_def.name}' is already registered")
        self._tools[tool_def.name] = tool_def
        if self.on_change is not None:
            self.on_change()

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def replace(self, tool_def: ToolDef) -> None:
        """Swap in an updated definition for an already-registered tool."""
        if tool_def.name not in self._tools:
            raise KeyError(f"Tool '{tool_def.name}' is not registered")
        self._tools[tool_def.name] = tool_def

    def list_all(self) -> list[ToolDef]:
        return list(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


class ResourceRegistry:
    """Stores registered resource definitions.

    Resource templates have their URI patterns pre-compiled to regex at
    registration time for O(1)-per-pattern matching at request time.
    Templates without a catch-all placeholder are tried first, so
    ``docs://{slug}/history`` wins over ``docs://{path*}`` whatever the
    registration order.
    """

    def __init__(self, *, on_change: OnChange | None = None) -> None:
        self._resources: dict[str, ResourceDef] = {}
        self._templates: dict[str, ResourceDef] = {}
        # Pre-compiled regex patterns: [(compiled_regex, param_names, ResourceDef)]
        self._compiled_patterns: list[tuple[re.Pattern[str], list[str], ResourceDef]] = []
        self._catch_all_patterns: list[tuple[re.Pattern[str], list[str], ResourceDef]] = []
        self.on_change = on_change

    def register(self, res_def: ResourceDef) -> None:
        if res_def.is_template:
            if res_def.uri in self._templates:
                raise ValueError(f"Resource template '{res_def.uri}' is already registered")
            # Pre-compile the regex pattern at registration time (a malformed
            # template raises here, before anything is registered)
            pattern, param_names = _compile_uri_template(res_def.uri)
            self._templates[res_def.uri] = res_def
            if _has_catch_all(res_def.uri):
                self._catch_all_patterns.append((pattern, param_names, res_def))
            else:
                self._compiled_patterns.append((pattern, param_names, res_def))
        else:
            if res_def.uri in self._resources:
                raise ValueError(f"Resource '{res_def.uri}' is already registered")
            self._resources[res_def.uri] = res_def
        if self.on_change is not None:
            self.on_change()

    def get(self, uri: str) -> ResourceDef | None:
        return self._resources.get(uri)

    def get_template(self, uri_template: str) -> ResourceDef | None:
        return self._templates.get(uri_template)

    def match_template(self, uri: str) -> tuple[ResourceDef, dict[str, str]] | None:
        """Match a concrete URI against registered templates.

        Returns the matching template def and extracted parameters, or ``None``.
        Uses pre-compiled regex patterns for fast matching.  Parameters are
        percent-decoded (``caf%C3%A9`` → ``café``), since clients encode
        them when they expand the template.
        """
        for pattern, param_names, tmpl_def in (
            *self._compiled_patterns,
            *self._catch_all_patterns,
        ):
            match = pattern.match(uri)
            if match is not None:
                return tmpl_def, {p: unquote(match.group(p)) for p in param_names}
        return None

    def list_all(self) -> list[ResourceDef]:
        return list(self._resources.values())

    def list_templates(self) -> list[ResourceDef]:
        return list(self._templates.values())

    def __len__(self) -> int:
        return len(self._resources) + len(self._templates)


class PromptRegistry:
    """Stores registered prompt definitions."""

    def __init__(self, *, on_change: OnChange | None = None) -> None:
        self._prompts: dict[str, PromptDef] = {}
        self.on_change = on_change

    def register(self, prompt_def: PromptDef) -> None:
        if prompt_def.name in self._prompts:
            raise ValueError(f"Prompt '{prompt_def.name}' is already registered")
        self._prompts[prompt_def.name] = prompt_def
        if self.on_change is not None:
            self.on_change()

    def get(self, name: str) -> PromptDef | None:
        return self._prompts.get(name)

    def list_all(self) -> list[PromptDef]:
        return list(self._prompts.values())

    def __len__(self) -> int:
        return len(self._prompts)


# ``{name}`` matches one path segment; ``{name*}`` and ``{+name}`` (RFC 6570
# explode / reserved expansion) match the rest of the URI, slashes included.
_PLACEHOLDER = re.compile(r"\{(\+)?(\w+)(\*)?\}")


def _has_catch_all(template: str) -> bool:
    return any(m.group(1) or m.group(3) for m in _PLACEHOLDER.finditer(template))


def _compile_uri_template(template: str) -> tuple[re.Pattern[str], list[str]]:
    """Compile a ``{param}`` style URI template to a regex pattern.

    Returns ``(compiled_pattern, param_names)``.  Called once at
    registration time so matching at request time is a single
    ``pattern.match()`` call.

    ``{param}`` matches a single path segment (no ``/``).  ``{param*}`` and
    ``{+param}`` match one or more characters *including* ``/``, for
    hierarchical ids such as ``docs://pages/{path*}`` →
    ``docs://pages/guides/setup``.

    Example::

        >>> pattern, names = _compile_uri_template("config://{key}")
        >>> pattern.match("config://database").group("key")
        'database'

    Raises:
        ValueError: A parameter name is used twice in the template.
    """
    pattern_str = "^"
    last_end = 0
    params: list[str] = []
    for m in _PLACEHOLDER.finditer(template):
        pattern_str += re.escape(template[last_end : m.start()])
        param_name = m.group(2)
        if param_name in params:
            raise ValueError(f"URI template {template!r} uses {{{param_name}}} more than once")
        params.append(param_name)
        catch_all = bool(m.group(1) or m.group(3))
        pattern_str += f"(?P<{param_name}>.+)" if catch_all else f"(?P<{param_name}>[^/]+)"
        last_end = m.end()
    pattern_str += re.escape(template[last_end:])
    pattern_str += "$"
    return re.compile(pattern_str), params


def _match_uri_template(template: str, uri: str) -> dict[str, str] | None:
    """Match *uri* against a ``{param}`` style template.

    Backward-compatible helper that compiles and matches in one shot.
    Prefer :func:`_compile_uri_template` for hot paths.
    """
    pattern, params = _compile_uri_template(template)
    match = pattern.match(uri)
    if match is None:
        return None
    return {p: unquote(match.group(p)) for p in params}
