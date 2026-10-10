"""Core prompt class and ``@prompt`` decorator.

The :class:`Prompt` class is dual-mode:

1. **Standalone** — ``await my_prompt("input")`` calls the LLM directly
   with the full pipeline (template → context → perspective → strategy →
   constraints → guards → LLM → parse → output guards).

2. **Agent-integrated** — ``build_agent(instructions=my_prompt)``
   uses :meth:`render_async` to build dynamic system prompts on every
   agent invocation.  Context providers fire each time, strategies shape
   reasoning, guards protect every call.

Example::

    from promptise.prompts import prompt, guard, context
    from promptise.prompts.guards import content_filter, length
    from promptise.prompts.context import user_context, tool_context
    from promptise.prompts.strategies import chain_of_thought, analyst

    @prompt(model="openai:gpt-5-mini")
    @context(user_context(), tool_context())
    @guard(content_filter(blocked=["secret"]), length(max_length=2000))
    async def analyze(text: str) -> str:
        \"""Analyze the following: {text}\"""

    result = await analyze("quarterly earnings report...")

    # Or with composition
    configured = (
        analyze
        .with_strategy(chain_of_thought)
        .with_perspective(analyst)
        .with_constraints("Under 500 words", "Include confidence scores")
    )
"""

from __future__ import annotations

import asyncio
import inspect
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, cast, get_type_hints

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from .context import BaseContext, ContextProvider, PromptContext
from .guards import Guard
from .strategies import Perspective, Strategy
from .template import render_template

__all__ = ["Prompt", "prompt", "constraint"]

# Type-only forward references for blocks/inspector (avoid circular imports)
_Block = Any  # blocks.Block
_Inspector = Any  # inspector.PromptInspector


# ---------------------------------------------------------------------------
# Output parsing helpers
# ---------------------------------------------------------------------------


def _guard_name(g: Any) -> str:
    """Name a guard for traces: its ``name`` attribute or class name."""
    return str(getattr(g, "name", None) or type(g).__name__)


def _strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers."""
    stripped = text.strip()
    # Match ```json\n...\n``` or ```\n...\n```
    m = re.match(r"^```(?:json|JSON)?\s*\n(.*?)```\s*$", stripped, re.DOTALL)
    if m:
        return m.group(1).strip()
    return stripped


def _extract_json(text: str) -> str:
    """Extract the first JSON object or array from text.

    Tracks whether characters are inside JSON string literals so that
    braces/brackets within strings do not affect depth counting.
    """
    text = _strip_markdown_fences(text)
    # Find first { or [
    for i, ch in enumerate(text):
        if ch in "{[":
            opener = ch
            closer = "}" if ch == "{" else "]"
            depth = 0
            in_string = False
            escape = False
            for j in range(i, len(text)):
                c = text[j]
                if escape:
                    escape = False
                    continue
                if c == "\\" and in_string:
                    escape = True
                    continue
                if c == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if c == opener:
                    depth += 1
                elif c == closer:
                    depth -= 1
                    if depth == 0:
                        return text[i : j + 1]
            break
    return text


def _parse_output(raw: str, return_type: type | None) -> Any:
    """Parse raw LLM output to match the declared return type."""
    if return_type is None or return_type is str:
        return raw

    if return_type is int:
        # Extract first integer
        m = re.search(r"-?\d+", raw)
        return int(m.group()) if m else int(raw.strip())

    if return_type is float:
        m = re.search(r"-?\d+\.?\d*", raw)
        return float(m.group()) if m else float(raw.strip())

    if return_type is bool:
        lower = raw.strip().lower()
        if lower in ("true", "yes", "1"):
            return True
        if lower in ("false", "no", "0"):
            return False
        return bool(raw.strip())

    if return_type is list or return_type is dict:
        return json.loads(_extract_json(raw))

    # Check for Pydantic BaseModel
    try:
        from pydantic import BaseModel

        if isinstance(return_type, type) and issubclass(return_type, BaseModel):
            parsed = json.loads(_extract_json(raw))
            return cast(type[BaseModel], return_type).model_validate(parsed)
    except ImportError:
        pass

    # Check for dataclass
    if is_dataclass(return_type) and isinstance(return_type, type):
        parsed = json.loads(_extract_json(raw))
        return return_type(**parsed)

    # Fallback: try json.loads
    try:
        return json.loads(_extract_json(raw))
    except (json.JSONDecodeError, ValueError):
        return raw


def _build_schema_instructions(return_type: type | None) -> str:
    """Generate JSON schema instructions for structured output."""
    if return_type is None or return_type in (str, int, float, bool):
        return ""

    schema_lines = ["Respond with valid JSON matching this structure:"]

    # Pydantic BaseModel
    try:
        from pydantic import BaseModel

        if isinstance(return_type, type) and issubclass(return_type, BaseModel):
            schema = cast(type[BaseModel], return_type).model_json_schema()
            schema_lines.append(f"```json\n{json.dumps(schema, indent=2)}\n```")
            return "\n".join(schema_lines)
    except ImportError:
        pass

    # Dataclass
    if is_dataclass(return_type) and isinstance(return_type, type):
        field_descs = {}
        for f in fields(return_type):
            field_descs[f.name] = (
                f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))
            )
        schema_lines.append(f"```json\n{json.dumps(field_descs, indent=2)}\n```")
        return "\n".join(schema_lines)

    if return_type in (list, dict):
        schema_lines.append(f"Respond with a JSON {'array' if return_type is list else 'object'}.")
        return "\n".join(schema_lines)

    return ""


# ---------------------------------------------------------------------------
# Prompt class
# ---------------------------------------------------------------------------


@dataclass
class PromptStats:
    """Execution statistics for a single prompt call."""

    prompt_name: str = ""
    model: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    context_providers_used: list[str] | None = None
    strategy: str = ""
    perspective: str = ""


class Prompt:
    """Dual-mode prompt: standalone LLM caller + agent instruction source.

    Wraps a function whose docstring is the prompt template.  Decorated
    with ``@prompt(model=...)`` to create a Prompt instance.

    **Standalone mode** — ``await prompt("text")``::

        result = await analyze("quarterly report")

    **Agent-integrated mode** — render without LLM call::

        text = await prompt.render_async(ctx)  # full prompt text
        messages = await prompt.to_messages(ctx)  # LangChain messages

    All ``with_*()`` methods return copies (immutable composition).
    """

    # Optional metadata attached by the YAML loader (see prompts/loader.py).
    # Declared here so static analysis recognises them as class attributes.
    _description: str | None = None
    _author: str | None = None
    _tags: list[str] = []
    _version: str | None = None
    _argument_schemas: dict[str, Any] = {}

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        model: str = "openai:gpt-5-mini",
        observe: bool = False,
        description: str | None = None,
    ) -> None:
        self._fn = fn
        self._model = model
        self._observe = observe
        self._name = fn.__name__
        # A docstring is indented like the code around it: clean it the way
        # ``inspect.getdoc`` does, so a multi-line template does not send
        # that indentation to the model.
        self._template = inspect.cleandoc(fn.__doc__ or "")
        self._sig = inspect.signature(fn)
        if description is not None:
            self._description = description

        # Resolve return type
        try:
            hints = get_type_hints(fn)
            self._return_type: type | None = hints.get("return")
        except Exception:
            # Fallback: read raw annotations when get_type_hints fails
            # (e.g. locally-defined types with `from __future__ import annotations`)
            raw = fn.__annotations__.get("return")
            if isinstance(raw, type):
                self._return_type = raw
            else:
                self._return_type = None

        # Pipeline components
        self._context_providers: list[ContextProvider] = []
        self._strategy: Strategy | None = None
        self._perspective: Perspective | None = None
        self._constraints: list[str] = []
        self._input_guards: list[Guard] = []
        self._output_guards: list[Guard] = []
        self._world: dict[str, BaseContext] = {}

        # Blocks (Layer 1 — composable prompt components)
        self._blocks: list[_Block] = []

        # Inspector (prompt assembly tracing)
        self._inspector: _Inspector | None = None

        # Lifecycle hooks
        self._on_before: Callable[..., Any] | None = None
        self._on_after: Callable[..., Any] | None = None
        self._on_error: Callable[..., Any] | None = None

        # Observer
        self._observer: Any | None = None

        # Stats from last call
        self.last_stats: PromptStats | None = None

        # Pick up pending guards/context/constraints/blocks/version from decorators applied before @prompt
        if hasattr(fn, "_pending_guards"):
            for g in fn._pending_guards:
                self._input_guards.append(g)
                self._output_guards.append(g)
        if hasattr(fn, "_pending_context"):
            self._context_providers.extend(fn._pending_context)
        if hasattr(fn, "_pending_constraints"):
            self._constraints.extend(fn._pending_constraints)
        if hasattr(fn, "_pending_blocks"):
            self._blocks.extend(fn._pending_blocks)
        if hasattr(fn, "_pending_version"):
            from .registry import registry

            registry.register(self._name, fn._pending_version, self)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Prompt name (derived from the function name)."""
        return self._name

    @property
    def model(self) -> str:
        """LLM model identifier."""
        return self._model

    @property
    def template(self) -> str:
        """Raw prompt template text."""
        return self._template

    @property
    def description(self) -> str | None:
        """What the prompt is for (``@prompt(description=...)`` or a YAML file's
        ``description``), or ``None``."""
        return self._description

    @property
    def return_type(self) -> type | None:
        """Declared return type."""
        return self._return_type

    # ------------------------------------------------------------------
    # Immutable composition — with_*() methods return copies
    # ------------------------------------------------------------------

    def _copy(self) -> Prompt:
        """Create a shallow copy preserving all configuration."""
        new = Prompt.__new__(Prompt)
        new._fn = self._fn
        new._model = self._model
        new._observe = self._observe
        new._name = self._name
        new._template = self._template
        new._sig = self._sig
        new._return_type = self._return_type
        new._context_providers = list(self._context_providers)
        new._strategy = self._strategy
        new._perspective = self._perspective
        new._constraints = list(self._constraints)
        new._input_guards = list(self._input_guards)
        new._output_guards = list(self._output_guards)
        new._blocks = list(self._blocks)
        new._inspector = self._inspector
        new._world = dict(self._world)
        new._on_before = self._on_before
        new._on_after = self._on_after
        new._on_error = self._on_error
        new._observer = self._observer
        new.last_stats = None
        # Metadata from @prompt(description=...) / the YAML loader, when set
        for attr in ("_description", "_author", "_tags", "_version", "_argument_schemas"):
            if attr in self.__dict__:
                setattr(new, attr, self.__dict__[attr])
        return new

    def with_model(self, model: str) -> Prompt:
        """Return a copy with a different model."""
        p = self._copy()
        p._model = model
        return p

    def with_context(self, *providers: ContextProvider) -> Prompt:
        """Return a copy with additional context providers."""
        p = self._copy()
        p._context_providers.extend(providers)
        return p

    def with_strategy(self, strategy: Strategy) -> Prompt:
        """Return a copy with a reasoning strategy."""
        p = self._copy()
        p._strategy = strategy
        return p

    def with_perspective(self, perspective: Perspective) -> Prompt:
        """Return a copy with a cognitive perspective."""
        p = self._copy()
        p._perspective = perspective
        return p

    def with_constraints(self, *texts: str) -> Prompt:
        """Return a copy with additional constraints."""
        p = self._copy()
        p._constraints.extend(texts)
        return p

    def with_guards(self, *guards: Guard) -> Prompt:
        """Return a copy with additional guards."""
        p = self._copy()
        for g in guards:
            p._input_guards.append(g)
            p._output_guards.append(g)
        return p

    def with_blocks(self, *blocks: _Block) -> Prompt:
        """Return a copy with additional prompt blocks."""
        p = self._copy()
        p._blocks.extend(blocks)
        return p

    def with_inspector(self, inspector: _Inspector) -> Prompt:
        """Return a copy with a PromptInspector attached."""
        p = self._copy()
        p._inspector = inspector
        return p

    def with_world(self, **contexts: BaseContext) -> Prompt:
        """Return a copy with pre-populated world contexts."""
        p = self._copy()
        p._world.update(contexts)
        return p

    def on_before(self, fn: Callable[..., Any]) -> Prompt:
        """Return a copy with a before-execution hook."""
        p = self._copy()
        p._on_before = fn
        return p

    def on_after(self, fn: Callable[..., Any]) -> Prompt:
        """Return a copy with an after-execution hook."""
        p = self._copy()
        p._on_after = fn
        return p

    def on_error(self, fn: Callable[..., Any]) -> Prompt:
        """Return a copy with an error-handling hook."""
        p = self._copy()
        p._on_error = fn
        return p

    # ------------------------------------------------------------------
    # Render (Mode 2: agent integration — no LLM call)
    # ------------------------------------------------------------------

    def render(self, ctx: PromptContext | None = None, **kwargs: Any) -> str:
        """Render the full prompt text WITHOUT calling the LLM.

        Runs: blocks → template → perspective → strategy → constraints.
        Async providers are skipped; use :meth:`render_async` for
        the full pipeline.

        Args:
            ctx: Optional :class:`PromptContext` for provider access.
            **kwargs: Template variables for ``{var}`` substitution.

        Returns:
            Rendered prompt text.
        """
        text = render_template(self._template, kwargs) if kwargs else self._template
        if ctx is None:
            ctx = self._build_context(kwargs)

        # Blocks assembly (Layer 1)
        if self._blocks:
            from .blocks import BlockContext, PromptAssembler

            block_ctx = BlockContext(
                state=kwargs,
                active_tools=[],
                metadata={},
            )
            assembled = PromptAssembler(*self._blocks).assemble(block_ctx)
            if assembled.text:
                text = f"{assembled.text}\n\n{text}"

        # Perspective
        if self._perspective is not None:
            text = self._perspective.apply(text, ctx)

        # Strategy
        if self._strategy is not None:
            text = self._strategy.wrap(text, ctx)

        # Constraints (after strategy — top-level requirements, not buried inside reasoning)
        if self._constraints:
            text = self._inject_constraints(text)

        # Structured output schema
        schema_instr = _build_schema_instructions(self._return_type)
        if schema_instr:
            text = f"{text}\n\n{schema_instr}"

        return text

    async def render_async(self, ctx: PromptContext | None = None, **kwargs: Any) -> str:
        """Async render with context providers (no LLM call).

        Full pipeline: blocks → template → context providers → perspective →
        strategy → constraints.  When the prompt has an inspector, the
        rendered text and block assembly are recorded as a trace.  Guards
        don't run here: they check an LLM call's input and output, which
        a render doesn't make (as agent ``instructions``, the agent runs
        them on each turn).

        Args:
            ctx: Optional :class:`PromptContext`.
            **kwargs: Template variables.

        Returns:
            Rendered prompt text with all dynamic context injected.
        """
        from .blocks import AssembledPrompt

        text = render_template(self._template, kwargs) if kwargs else self._template
        if ctx is None:
            ctx = self._build_context(kwargs)

        # Blocks assembly (Layer 1)
        assembled = AssembledPrompt(text="")
        if self._blocks:
            from .blocks import BlockContext, PromptAssembler

            block_ctx = BlockContext(
                state=kwargs,
                active_tools=[],
                metadata={},
            )
            assembled = PromptAssembler(*self._blocks).assemble(block_ctx)
            if assembled.text:
                text = f"{assembled.text}\n\n{text}"

        # Context providers
        sections = await self._run_context_providers(ctx)
        if sections:
            text = f"{text}\n\n{sections}"

        # Perspective
        if self._perspective is not None:
            text = self._perspective.apply(text, ctx)

        # Strategy
        if self._strategy is not None:
            text = self._strategy.wrap(text, ctx)

        # Constraints (after strategy — top-level requirements, not buried inside reasoning)
        if self._constraints:
            text = self._inject_constraints(text)

        # Structured output schema
        schema_instr = _build_schema_instructions(self._return_type)
        if schema_instr:
            text = f"{text}\n\n{schema_instr}"

        if self._inspector is not None:
            trace = self._inspector.record_assembly(
                assembled, prompt_name=self._name, model=self._model
            )
            trace.input_text = text

        return text

    async def to_messages(self, ctx: PromptContext | None = None, **kwargs: Any) -> list[Any]:
        """Produce LangChain message objects for agent integration.

        Returns ``[SystemMessage(rendered_prompt), HumanMessage(input)]``
        when kwargs are provided, or just ``[SystemMessage]`` for
        system prompt use.

        Args:
            ctx: Optional :class:`PromptContext`.
            **kwargs: Template variables.

        Returns:
            List of LangChain message objects.
        """
        rendered = await self.render_async(ctx, **kwargs)
        messages: list[Any] = [SystemMessage(content=rendered)]

        # If there's an input argument, add as HumanMessage
        input_text = ctx.rendered_text if ctx and ctx.rendered_text else ""
        if input_text:
            messages.append(HumanMessage(content=input_text))

        return messages

    # ------------------------------------------------------------------
    # Execute (Mode 1: standalone LLM call)
    # ------------------------------------------------------------------

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the full prompt pipeline with an LLM call.

        Pipeline:
        1. Bind arguments to function signature
        2. Render template with variables
        3. Build PromptContext
        4. Run on_before hook
        5. Run context providers
        6. Apply perspective
        7. Apply strategy
        8. Inject constraints + schema instructions
        9. Run input guards
        10. Call LLM
        11. Parse strategy output
        12. Parse return type
        13. Run output guards
        14. Run on_after hook
        15. Return typed result
        """
        start = time.perf_counter()

        # 1. Bind arguments
        bound = self._sig.bind(*args, **kwargs)
        bound.apply_defaults()
        variables = dict(bound.arguments)

        # 2. Render template
        text = render_template(self._template, variables)

        # 3. Build PromptContext
        ctx = self._build_context(variables)
        ctx.rendered_text = text

        try:
            # 4. on_before hook
            await self._run_hook(self._on_before, ctx)

            # 4.5. Blocks assembly (Layer 1)
            from .blocks import AssembledPrompt

            assembled = AssembledPrompt(text="")
            if self._blocks:
                from .blocks import BlockContext, PromptAssembler

                block_ctx = BlockContext(
                    state=variables,
                    active_tools=[],
                    metadata={},
                )
                assembled = PromptAssembler(*self._blocks).assemble(block_ctx)
                if assembled.text:
                    text = f"{assembled.text}\n\n{text}"

            # Record in inspector (completed after the LLM call)
            trace = None
            if self._inspector is not None:
                trace = self._inspector.record_assembly(
                    assembled,
                    prompt_name=self._name,
                    model=self._model,
                )

            # 5. Context providers
            sections = await self._run_context_providers(ctx)
            if sections:
                text = f"{text}\n\n{sections}"

            # 6. Perspective
            if self._perspective is not None:
                text = self._perspective.apply(text, ctx)

            # 7. Strategy wrapping
            if self._strategy is not None:
                text = self._strategy.wrap(text, ctx)

            # 8. Constraints + schema
            if self._constraints:
                text = self._inject_constraints(text)
            schema_instr = _build_schema_instructions(self._return_type)
            if schema_instr:
                text = f"{text}\n\n{schema_instr}"

            # 9. Input guards
            if trace is not None:
                trace.input_text = text
            for g in self._input_guards:
                try:
                    text = await g.check_input(text)
                except Exception:
                    if trace is not None and self._inspector is not None:
                        self._inspector.record_guard(trace, _guard_name(g), passed=False)
                    raise
            if trace is not None and self._inspector is not None:
                trace.input_text = text

            # 10. Call LLM
            llm = init_chat_model(self._model)

            # Find timeout guard if present
            timeout_seconds: float | None = None
            for g in self._input_guards:
                if hasattr(g, "seconds"):
                    timeout_seconds = g.seconds
                    break

            messages = [SystemMessage(content=text)]
            if timeout_seconds is not None:
                raw_msg = await asyncio.wait_for(llm.ainvoke(messages), timeout=timeout_seconds)
            else:
                raw_msg = await llm.ainvoke(messages)

            _raw_content = raw_msg.content if hasattr(raw_msg, "content") else str(raw_msg)
            raw_output: str = (
                _raw_content if isinstance(_raw_content, str) else str(_raw_content or "")
            )

            # 11. Strategy parsing
            if self._strategy is not None:
                raw_output = self._strategy.parse(raw_output, ctx)

            # 12. Parse return type
            result = _parse_output(raw_output, self._return_type)

            # 13. Output guards
            for g in self._output_guards:
                try:
                    result = await g.check_output(result)
                except Exception:
                    if trace is not None and self._inspector is not None:
                        self._inspector.record_guard(trace, _guard_name(g), passed=False)
                    raise
                if trace is not None and self._inspector is not None:
                    self._inspector.record_guard(trace, _guard_name(g), passed=True)

            # 14. Stats
            elapsed = (time.perf_counter() - start) * 1000
            if trace is not None and self._inspector is not None:
                self._inspector.record_execution(trace, raw_output, elapsed)
            self.last_stats = PromptStats(
                prompt_name=self._name,
                model=self._model,
                latency_ms=elapsed,
                context_providers_used=[type(p).__name__ for p in self._context_providers],
                strategy=repr(self._strategy) if self._strategy else "",
                perspective=repr(self._perspective) if self._perspective else "",
            )

            # 15. on_after hook
            await self._run_hook(self._on_after, ctx, result)

            return result

        except Exception as exc:
            await self._run_hook(self._on_error, ctx, exc)
            raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, variables: dict[str, Any]) -> PromptContext:
        """Build a :class:`PromptContext` from input variables and world."""
        ctx = PromptContext(
            prompt_name=self._name,
            model=self._model,
            input_args=variables,
            world=dict(self._world),
        )
        return ctx

    async def _run_context_providers(self, ctx: PromptContext) -> str:
        """Run all context providers and join their outputs."""
        parts: list[str] = []
        for provider in self._context_providers:
            section = await provider.provide(ctx)
            if section:
                parts.append(section)
        return "\n\n".join(parts)

    def _inject_constraints(self, text: str) -> str:
        """Append constraints as numbered requirements."""
        lines = ["", "You MUST follow these requirements:"]
        for i, c in enumerate(self._constraints, 1):
            lines.append(f"{i}. {c}")
        return text + "\n".join(lines)

    @staticmethod
    async def _run_hook(hook: Callable[..., Any] | None, *args: Any) -> None:
        """Run a lifecycle hook (sync or async)."""
        if hook is None:
            return
        if inspect.iscoroutinefunction(hook):
            await hook(*args)
        else:
            hook(*args)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = [f"name={self._name!r}", f"model={self._model!r}"]
        if self._context_providers:
            parts.append(f"providers={len(self._context_providers)}")
        if self._strategy:
            parts.append(f"strategy={self._strategy!r}")
        if self._perspective:
            parts.append(f"perspective={self._perspective!r}")
        if self._constraints:
            parts.append(f"constraints={len(self._constraints)}")
        if self._input_guards:
            parts.append(f"guards={len(self._input_guards)}")
        return f"<Prompt {', '.join(parts)}>"


# ---------------------------------------------------------------------------
# @prompt decorator
# ---------------------------------------------------------------------------


def prompt(
    model: str = "openai:gpt-5-mini",
    *,
    observe: bool = False,
    inspect: _Inspector | None = None,
    description: str | None = None,
) -> Callable[[Callable[..., Any]], Prompt]:
    """Decorator that turns a function into a :class:`Prompt`.

    The function's docstring becomes the prompt template.  Parameters
    become template variables (``{param_name}``).  The return type
    annotation determines output parsing.

    Usage::

        @prompt(model="openai:gpt-5-mini")
        async def summarize(text: str, max_words: int = 100) -> str:
            \"""Summarize in {max_words} words: {text}\"""

        result = await summarize("long article...")

    Args:
        model: LLM model identifier (e.g. ``"openai:gpt-4o"``,
            ``"anthropic:claude-sonnet-4-20250514"``).
        observe: Enable observability recording.
        inspect: Optional :class:`PromptInspector` for assembly tracing.
        description: What the prompt is for.  Shown to MCP clients when the
            prompt is served with ``MCPServer.include_prompts()``.

    Returns:
        Decorator that produces a :class:`Prompt` instance.
    """

    def decorator(fn: Callable[..., Any]) -> Prompt:
        p = Prompt(fn, model=model, observe=observe, description=description)
        if inspect is not None:
            p._inspector = inspect
        return p

    return decorator


# ---------------------------------------------------------------------------
# @constraint decorator
# ---------------------------------------------------------------------------


def constraint(text: str) -> Callable[..., Any]:
    """Decorator that attaches a constraint to a :class:`Prompt`.

    Constraints are hard requirements appended to the prompt text as
    numbered instructions before the LLM call.

    Usage::

        @prompt(model="openai:gpt-5-mini")
        @constraint("Must cite at least 2 sources")
        @constraint("Under 300 words")
        async def write_argument(topic: str) -> str:
            \"""Write a persuasive argument about: {topic}\"""
    """

    def decorator(prompt_or_func: Any) -> Any:
        if isinstance(prompt_or_func, Prompt):
            prompt_or_func._constraints.append(text)
            return prompt_or_func
        # Applied before @prompt — store as pending
        if not hasattr(prompt_or_func, "_pending_constraints"):
            prompt_or_func._pending_constraints = []
        prompt_or_func._pending_constraints.append(text)
        return prompt_or_func

    return decorator
