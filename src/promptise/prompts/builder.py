"""Fluent API for runtime prompt assembly.

:class:`PromptBuilder` lets you construct prompts programmatically
at runtime rather than at decoration time.  Useful when prompt
configuration depends on runtime conditions.

Example::

    from promptise.prompts.builder import PromptBuilder
    from promptise.prompts.context import user_context, tool_context, BaseContext
    from promptise.prompts.strategies import chain_of_thought, analyst
    from promptise.prompts.guards import content_filter, length

    prompt = (
        PromptBuilder("analyze")
        .system("Expert data analyst")
        .user(UserContext(expertise_level="expert"))
        .world(project=BaseContext(name="Alpha", deadline="March 2026"))
        .context(tool_context(), memory_context())
        .strategy(chain_of_thought)
        .perspective(analyst)
        .constraint("Must include confidence scores")
        .guard(content_filter(blocked=["secret"]), length(max_length=5000))
        .template("Analyze: {data}")
        .output_type(Analysis)
        .model("openai:gpt-5-mini")
        .build()
    )
    result = await prompt(data="quarterly figures...")
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .context import (
    BaseContext,
    ContextProvider,
    EnvironmentContext,
    UserContext,
)
from .core import Prompt
from .guards import Guard
from .strategies import Perspective, Strategy

__all__ = ["PromptBuilder"]


class PromptBuilder:
    """Fluent builder for constructing :class:`Prompt` instances at runtime.

    All methods return ``self`` for chaining.  Call :meth:`build` to
    produce the final :class:`Prompt`.

    Args:
        name: Prompt name (used for logging and observability).
    """

    def __init__(self, name: str = "builder_prompt") -> None:
        self._name = name
        self._system_text: str = ""
        self._template_text: str = ""
        self._model_name: str = "openai:gpt-5-mini"
        self._observe: bool = False
        self._output_type_cls: type | None = None
        self._context_providers: list[ContextProvider] = []
        self._strategy_obj: Strategy | None = None
        self._perspective_obj: Perspective | None = None
        self._constraints: list[str] = []
        self._input_guards: list[Guard] = []
        self._output_guards: list[Guard] = []
        self._world_contexts: dict[str, BaseContext] = {}
        self._before_hook: Callable[..., Any] | None = None
        self._after_hook: Callable[..., Any] | None = None
        self._error_hook: Callable[..., Any] | None = None

    # ------------------------------------------------------------------
    # Builder methods (all return self)
    # ------------------------------------------------------------------

    def system(self, text: str) -> PromptBuilder:
        """Set the system/instruction text (prepended to template)."""
        self._system_text = text
        return self

    def template(self, text: str) -> PromptBuilder:
        """Set the prompt template with ``{variable}`` placeholders."""
        self._template_text = text
        return self

    def model(self, name: str) -> PromptBuilder:
        """Set the LLM model identifier."""
        self._model_name = name
        return self

    def observe(self, enabled: bool = True) -> PromptBuilder:
        """Enable or disable observability recording."""
        self._observe = enabled
        return self

    def output_type(self, t: type) -> PromptBuilder:
        """Set the output type for structured parsing."""
        self._output_type_cls = t
        return self

    def user(self, user_ctx: UserContext) -> PromptBuilder:
        """Set the user context."""
        self._world_contexts["user"] = user_ctx
        return self

    def env(self, env_ctx: EnvironmentContext) -> PromptBuilder:
        """Set the environment context."""
        self._world_contexts["environment"] = env_ctx
        return self

    def world(self, **contexts: BaseContext) -> PromptBuilder:
        """Add world contexts by name."""
        self._world_contexts.update(contexts)
        return self

    def context(self, *providers: ContextProvider) -> PromptBuilder:
        """Add context providers."""
        self._context_providers.extend(providers)
        return self

    def strategy(self, s: Strategy) -> PromptBuilder:
        """Set the reasoning strategy."""
        self._strategy_obj = s
        return self

    def perspective(self, p: Perspective) -> PromptBuilder:
        """Set the cognitive perspective."""
        self._perspective_obj = p
        return self

    def constraint(self, *texts: str) -> PromptBuilder:
        """Add constraints."""
        self._constraints.extend(texts)
        return self

    def guard(self, *guards: Guard) -> PromptBuilder:
        """Add guards (applied to both input and output)."""
        for g in guards:
            self._input_guards.append(g)
            self._output_guards.append(g)
        return self

    def on_before(self, fn: Callable[..., Any]) -> PromptBuilder:
        """Set the before-execution hook."""
        self._before_hook = fn
        return self

    def on_after(self, fn: Callable[..., Any]) -> PromptBuilder:
        """Set the after-execution hook."""
        self._after_hook = fn
        return self

    def on_error(self, fn: Callable[..., Any]) -> PromptBuilder:
        """Set the error-handling hook."""
        self._error_hook = fn
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> Prompt:
        """Construct the :class:`Prompt` from accumulated configuration.

        Creates a synthetic function from the template text, then wraps
        it in a Prompt with all configured components.

        Returns:
            Configured :class:`Prompt` instance.
        """
        # Combine system text and template
        full_template = ""
        if self._system_text:
            full_template = self._system_text
        if self._template_text:
            if full_template:
                full_template += "\n\n"
            full_template += self._template_text

        if not full_template:
            full_template = "Respond to the user's request."

        # Extract variable names from template
        import re

        var_pattern = re.compile(r"\{(\w+)\}")
        var_names = list(dict.fromkeys(var_pattern.findall(full_template)))

        # Validate prompt name is a safe Python identifier — prevents
        # code injection via crafted names in exec() below.
        import re as _re

        if not _re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", self._name):
            raise ValueError(f"Prompt name must be a valid Python identifier, got: {self._name!r}")

        # Validate ALL variable names — prevents code injection via exec()
        for v in var_names:
            if not _re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", v):
                raise ValueError(
                    f"Template variable name must be a valid Python identifier, got: {v!r}"
                )

        # Build synthetic function
        if var_names:
            params = ", ".join(f"{v}: str = ''" for v in var_names)
            fn_code = f"async def {self._name}({params}):\n"
        else:
            fn_code = f"async def {self._name}(**kwargs):\n"
        # Use repr to safely embed the template as the docstring
        fn_code += f"    {full_template!r}\n"

        namespace: dict[str, Any] = {}
        exec(fn_code, namespace)  # noqa: S102  # nosec B102 - builds wrapper for library-controlled template
        fn = namespace[self._name]
        fn.__doc__ = full_template

        # Set return type if specified
        if self._output_type_cls is not None:
            fn.__annotations__["return"] = self._output_type_cls

        # Create Prompt
        p = Prompt(fn, model=self._model_name, observe=self._observe)
        p._name = self._name
        # The template is exact text, not a docstring: keep its indentation.
        p._template = full_template.strip()

        # Attach components
        p._context_providers = list(self._context_providers)
        p._strategy = self._strategy_obj
        p._perspective = self._perspective_obj
        p._constraints = list(self._constraints)
        p._input_guards = list(self._input_guards)
        p._output_guards = list(self._output_guards)
        p._world = dict(self._world_contexts)
        p._on_before = self._before_hook
        p._on_after = self._after_hook
        p._on_error = self._error_hook

        return p

    def __repr__(self) -> str:
        return f"<PromptBuilder {self._name!r}>"
