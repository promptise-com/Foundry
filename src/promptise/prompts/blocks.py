"""Composable prompt components with priority-based assembly.

PromptBlocks are typed, reusable components that compose into full system
prompts.  The :class:`PromptAssembler` assembles blocks in priority order
for optimal prompt construction.

Example::

    from promptise.prompts.blocks import (
        Identity, Rules, ContextSlot, OutputFormat, PromptAssembler,
    )

    assembler = PromptAssembler(
        Identity("Expert data analyst"),
        Rules(["Always cite sources", "Include confidence intervals"]),
        ContextSlot("user_data"),
        OutputFormat(format="json", schema=AnalysisResult),
    )
    assembled = assembler.fill_slot("user_data", csv_text).assemble()
    print(assembled.text)          # Full prompt
    print(assembled.included)      # Blocks included in assembly

Or with the ``@prompt`` decorator::

    from promptise.prompts import prompt
    from promptise.prompts.blocks import blocks, Identity, Rules

    @prompt(model="openai:gpt-5-mini")
    @blocks(Identity("Expert analyst"), Rules(["Cite sources"]))
    async def analyze(text: str) -> str:
        \"""Analyze: {text}\"""
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Block",
    "BlockContext",
    "BlockTrace",
    "AssembledPrompt",
    "Identity",
    "Rules",
    "OutputFormat",
    "ContextSlot",
    "Section",
    "Examples",
    "Conditional",
    "Composite",
    "PromptAssembler",
    "blocks",
    # Custom block helpers
    "SimpleBlock",
    "block",
    # Agentic blocks
    "ToolsBlock",
    "ObservationBlock",
    "PlanBlock",
    "ReflectionBlock",
    "PhaseBlock",
]


# ---------------------------------------------------------------------------
# Block protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Block(Protocol):
    """A composable unit of prompt content.

    Implement this protocol to create custom block types.
    The ``priority`` determines survival when token budgets are tight
    (10 = always included, 1 = nice-to-have).

    Three ways to create custom blocks:

    1. **Class**: Implement the Block protocol::

        class MyBlock:
            name = "my_block"
            priority = 5
            def render(self, ctx=None) -> str:
                return "Custom content"

    2. **@block decorator**: Turn a function into a block::

        @block("my_block", priority=5)
        def my_block(ctx=None) -> str:
            return "Custom content"

    3. **SimpleBlock**: Inline with just a string::

        my_block = SimpleBlock("my_block", "Custom content", priority=5)
    """

    @property
    def name(self) -> str:
        """Unique identifier within an assembly."""
        ...

    @property
    def priority(self) -> int:
        """Importance 1-10.  Higher = more likely to survive budget cuts."""
        ...

    def render(self, ctx: BlockContext | None = None) -> str:
        """Render this block to text."""
        ...


class SimpleBlock:
    """A block created from a string or callable.

    The simplest way to create a custom block::

        disclaimer = SimpleBlock("disclaimer", "This is not financial advice.", priority=9)
        dynamic = SimpleBlock("stats", lambda ctx: f"Users: {ctx.metadata['users']}", priority=5)
    """

    def __init__(self, name: str, content: str | Callable, priority: int = 5) -> None:
        self._name = name
        self._content = content
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def render(self, ctx: BlockContext | None = None) -> str:
        if callable(self._content):
            return self._content(ctx)
        return self._content


def block(name: str, *, priority: int = 5) -> Callable:
    """Decorator that turns a function into a Block.

    Usage::

        @block("safety_rules", priority=9)
        def safety_rules(ctx=None) -> str:
            return "1. Never share personal data\\n2. Always cite sources"

        # Use in PromptAssembler or PromptNode:
        assembler = PromptAssembler(Identity("Analyst"), safety_rules)
    """

    def decorator(func: Callable) -> SimpleBlock:
        return SimpleBlock(name, func, priority=priority)

    return decorator


# ---------------------------------------------------------------------------
# Block context
# ---------------------------------------------------------------------------


@dataclass
class BlockContext:
    """Runtime context available to blocks during rendering."""

    state: dict[str, Any] = field(default_factory=dict)
    turn: int = 0
    phase: str = ""
    active_tools: list[str] = field(default_factory=list)
    conversation_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tracing dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BlockTrace:
    """Per-block assembly trace."""

    name: str
    priority: int
    rendered_length: int
    estimated_tokens: int
    included: bool
    render_time_ms: float


@dataclass
class AssembledPrompt:
    """Result of assembling prompt blocks."""

    text: str
    included: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    estimated_tokens: int = 0
    block_details: list[BlockTrace] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Fast token estimate — ``word_count * 1.3``."""
    if not text:
        return 0
    return int(len(text.split()) * 1.3)


# ---------------------------------------------------------------------------
# Built-in block types
# ---------------------------------------------------------------------------


class Identity:
    """Who the agent is — always included (priority 10).

    Args:
        description: Core identity statement.
        name: Block name.  Defaults to ``"identity"``.
        traits: Optional list of personality / capability traits.
    """

    def __init__(
        self,
        description: str,
        *,
        name: str = "identity",
        traits: list[str] | None = None,
    ) -> None:
        self._name = name
        self._description = description
        self._traits = traits or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return 10

    def render(self, ctx: BlockContext | None = None) -> str:
        lines = [f"You are {self._description}."]
        if self._traits:
            lines.append("")
            lines.append("Key traits:")
            for trait in self._traits:
                lines.append(f"- {trait}")
        return "\n".join(lines)


class Rules:
    """Behavioral constraints (priority 9).

    Args:
        rules: List of rule strings.
        name: Block name.  Defaults to ``"rules"``.
    """

    def __init__(
        self,
        rules: list[str],
        *,
        name: str = "rules",
    ) -> None:
        self._name = name
        self._rules = rules

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return 9

    def render(self, ctx: BlockContext | None = None) -> str:
        if not self._rules:
            return ""
        lines = ["Rules:"]
        for i, rule in enumerate(self._rules, 1):
            lines.append(f"{i}. {rule}")
        return "\n".join(lines)


class OutputFormat:
    """Response structure specification (priority 8).

    Args:
        format: Output format — ``"text"``, ``"json"``, ``"markdown"``.
        schema: Optional Pydantic model or dataclass for JSON output.
        instructions: Additional formatting instructions.
        name: Block name.  Defaults to ``"output_format"``.
    """

    def __init__(
        self,
        *,
        format: str = "text",
        schema: type | None = None,
        instructions: str = "",
        name: str = "output_format",
    ) -> None:
        self._name = name
        self._format = format
        self._schema = schema
        self._instructions = instructions

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return 8

    def render(self, ctx: BlockContext | None = None) -> str:
        parts: list[str] = []
        if self._format == "json":
            parts.append("Respond with valid JSON.")
        elif self._format == "markdown":
            parts.append("Respond in Markdown format.")

        if self._schema is not None:
            # Extract schema from Pydantic or dataclass
            schema_desc = _describe_schema(self._schema)
            if schema_desc:
                parts.append(f"Schema:\n{schema_desc}")

        if self._instructions:
            parts.append(self._instructions)

        return "\n".join(parts)


class ContextSlot:
    """Dynamic injection point, filled at runtime (priority configurable).

    Use :meth:`fill` to provide content.  Unfilled slots render the
    ``default`` value (empty string if not set).

    Args:
        name: Slot identifier (also the block name).
        priority: Importance 1-10.  Default 6.
        default: Fallback text when unfilled.
    """

    def __init__(
        self,
        name: str,
        *,
        priority: int = 6,
        default: str = "",
    ) -> None:
        self._name = name
        self._priority = priority
        self._default = default
        self._content: str | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def fill(self, content: str) -> ContextSlot:
        """Return a copy with content filled."""
        slot = ContextSlot(self._name, priority=self._priority, default=self._default)
        slot._content = content
        return slot

    def render(self, ctx: BlockContext | None = None) -> str:
        if self._content is not None:
            return self._content
        return self._default


class Section:
    """Custom named section (priority configurable).

    Args:
        name: Section identifier (also the block name).
        content: Static text or a callable ``(ctx) -> str``.
        priority: Importance 1-10.  Default 5.
    """

    def __init__(
        self,
        name: str,
        content: str | Callable[..., str],
        *,
        priority: int = 5,
    ) -> None:
        self._name = name
        self._content = content
        self._priority = priority

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._priority

    def render(self, ctx: BlockContext | None = None) -> str:
        if callable(self._content):
            import inspect

            try:
                sig = inspect.signature(self._content)
                params = [
                    p for p in sig.parameters.values() if p.default is inspect.Parameter.empty
                ]
                if params:
                    return self._content(ctx)
                return self._content()
            except (ValueError, TypeError):
                return self._content(ctx)
        return self._content


class Examples:
    """Few-shot examples with auto-truncation (priority 4).

    When token budget is tight, fewer examples are included.

    Args:
        examples: List of ``{"input": "...", "output": "..."}`` dicts.
        name: Block name.  Defaults to ``"examples"``.
        max_count: Maximum examples to include.  Default: all.
    """

    def __init__(
        self,
        examples: list[dict[str, str]],
        *,
        name: str = "examples",
        max_count: int | None = None,
    ) -> None:
        self._name = name
        self._examples = examples
        self._max_count = max_count

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return 4

    def render(self, ctx: BlockContext | None = None) -> str:
        if not self._examples:
            return ""

        # Determine how many to include
        count = len(self._examples)
        if self._max_count is not None:
            count = min(count, self._max_count)

        selected = self._examples[:count]
        lines = ["Examples:"]
        for i, ex in enumerate(selected, 1):
            lines.append(self._render_example(ex, i))
        return "\n".join(lines)

    @staticmethod
    def _render_example(ex: dict[str, str], num: int = 0) -> str:
        prefix = f"\n--- Example {num} ---\n" if num else ""
        parts: list[str] = [prefix] if prefix else []
        for key, value in ex.items():
            parts.append(f"{key.title()}: {value}")
        return "\n".join(parts)


class Conditional:
    """Block that renders only when a condition is true.

    Inherits priority from the inner block.

    Args:
        name: Block name.
        block: The inner block to conditionally render.
        condition: ``(ctx: BlockContext) -> bool`` predicate.
    """

    def __init__(
        self,
        name: str,
        block: Block,
        *,
        condition: Callable[[BlockContext | None], bool],
    ) -> None:
        self._name = name
        self._block = block
        self._condition = condition

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        return self._block.priority

    def render(self, ctx: BlockContext | None = None) -> str:
        if not self._condition(ctx):
            return ""
        return self._block.render(ctx)


class Composite:
    """Groups multiple blocks as a single unit.

    Priority is the maximum of all inner blocks.

    Args:
        name: Block name.
        blocks: List of inner blocks.
        separator: Join string between blocks.
    """

    def __init__(
        self,
        name: str,
        blocks: list[Block],
        *,
        separator: str = "\n\n",
    ) -> None:
        self._name = name
        self._blocks = blocks
        self._separator = separator

    @property
    def name(self) -> str:
        return self._name

    @property
    def priority(self) -> int:
        if not self._blocks:
            return 1
        return max(b.priority for b in self._blocks)

    def render(self, ctx: BlockContext | None = None) -> str:
        parts = [b.render(ctx) for b in self._blocks]
        return self._separator.join(p for p in parts if p)


# ---------------------------------------------------------------------------
# PromptAssembler
# ---------------------------------------------------------------------------


class PromptAssembler:
    """Assembles blocks into a final prompt with optional token budgeting.

    When a ``token_budget`` is given to :meth:`assemble`, blocks are
    dropped lowest-priority-first until the total fits within budget.
    Higher-priority blocks survive; lower-priority blocks are listed
    in ``excluded``.

    Args:
        *initial_blocks: Blocks to include initially.
        separator: String between rendered blocks.
        token_budget: Default token budget (``None`` = unlimited).
    """

    def __init__(
        self,
        *initial_blocks: Block,
        separator: str = "\n\n",
        token_budget: int | None = None,
    ) -> None:
        self._blocks: list[Block] = list(initial_blocks)
        self._separator = separator
        self._token_budget = token_budget
        self._slot_fills: dict[str, str] = {}

    def add(self, block: Block) -> PromptAssembler:
        """Add a block.  Returns self for chaining."""
        self._blocks.append(block)
        return self

    def remove(self, name: str) -> PromptAssembler:
        """Remove a block by name.  Returns self for chaining."""
        self._blocks = [b for b in self._blocks if b.name != name]
        return self

    def fill_slot(self, slot_name: str, content: str) -> PromptAssembler:
        """Fill a :class:`ContextSlot` by name.  Returns self for chaining."""
        self._slot_fills[slot_name] = content
        return self

    def assemble(
        self,
        ctx: BlockContext | None = None,
        *,
        token_budget: int | None = None,
    ) -> AssembledPrompt:
        """Assemble all blocks into a final prompt.

        When a token budget is set (either here or in the constructor),
        blocks are dropped lowest-priority-first until the assembled
        prompt fits within the budget.  Blocks with the same priority
        are dropped in reverse insertion order (later blocks first).

        Args:
            ctx: Optional rendering context for blocks.
            token_budget: Override the constructor budget for this call.
                ``None`` uses the constructor default. ``0`` or negative
                means unlimited.

        Returns:
            An :class:`AssembledPrompt` with the final text, included/
            excluded block names, token estimate, and per-block traces.
        """
        budget = token_budget if token_budget is not None else self._token_budget

        # 1. Prepare blocks — apply slot fills
        prepared: list[Block] = []
        for block in self._blocks:
            if isinstance(block, ContextSlot) and block.name in self._slot_fills:
                prepared.append(block.fill(self._slot_fills[block.name]))
            else:
                prepared.append(block)

        # 2. Render all blocks and collect traces
        rendered: list[tuple[int, Block, str, BlockTrace]] = []
        for idx, block in enumerate(prepared):
            t0 = time.perf_counter()
            text = block.render(ctx)
            elapsed = (time.perf_counter() - t0) * 1000
            tokens = _estimate_tokens(text)
            trace = BlockTrace(
                name=block.name,
                priority=block.priority,
                rendered_length=len(text),
                estimated_tokens=tokens,
                included=bool(text.strip()),
                render_time_ms=elapsed,
            )
            rendered.append((idx, block, text, trace))

        # 3. Separate non-empty from empty
        non_empty = [(i, b, t, tr) for i, b, t, tr in rendered if t.strip()]
        empty_names = [tr.name for _, _, t, tr in rendered if not t.strip()]

        # 4. Apply token budget — drop lowest-priority blocks first
        excluded_names: list[str] = list(empty_names)
        if budget is not None and budget > 0:
            # Sort candidates by priority ASC, then by insertion order DESC
            # (lowest priority + latest insertion = first to drop)
            candidates = sorted(
                non_empty,
                key=lambda x: (x[1].priority, -x[0]),
            )
            total = sum(tr.estimated_tokens for _, _, _, tr in candidates)

            while total > budget and candidates:
                # Drop the lowest-priority candidate
                dropped = candidates.pop(0)
                dropped[3].included = False
                excluded_names.append(dropped[3].name)
                total -= dropped[3].estimated_tokens

            non_empty = [c for c in candidates if c[3].included]

        # 5. Re-sort by original insertion order for final text
        non_empty.sort(key=lambda x: x[0])

        # 6. Join texts
        texts = [t for _, _, t, _ in non_empty]
        final_text = self._separator.join(texts)
        final_tokens = _estimate_tokens(final_text)

        # 7. Collect all traces (in original block order)
        all_traces: list[BlockTrace] = []
        for block in self._blocks:
            found = False
            for _, _, _, tr in rendered:
                if tr.name == block.name:
                    all_traces.append(tr)
                    found = True
                    break
            if not found:
                all_traces.append(
                    BlockTrace(
                        name=block.name,
                        priority=block.priority,
                        rendered_length=0,
                        estimated_tokens=0,
                        included=False,
                        render_time_ms=0.0,
                    )
                )

        included_names = [tr.name for tr in all_traces if tr.included]

        return AssembledPrompt(
            text=final_text,
            included=included_names,
            excluded=excluded_names,
            estimated_tokens=final_tokens,
            block_details=all_traces,
        )


# ---------------------------------------------------------------------------
# @blocks decorator
# ---------------------------------------------------------------------------


def blocks(*block_list: Block) -> Callable[..., Any]:
    """Decorator that attaches blocks to a :class:`~promptise.prompts.core.Prompt`.

    Blocks are assembled and prepended to the system prompt at
    execution time.

    Usage::

        @prompt(model="openai:gpt-5-mini")
        @blocks(Identity("Expert analyst"), Rules(["Cite sources"]))
        async def analyze(text: str) -> str:
            \"""Analyze: {text}\"""
    """

    def decorator(prompt_or_func: Any) -> Any:
        if hasattr(prompt_or_func, "_blocks"):
            # Applied to a Prompt object
            prompt_or_func._blocks = list(block_list)
            return prompt_or_func
        # Applied to a plain function (before @prompt)
        if not hasattr(prompt_or_func, "_pending_blocks"):
            prompt_or_func._pending_blocks = []
        prompt_or_func._pending_blocks.extend(block_list)
        return prompt_or_func

    return decorator


# ---------------------------------------------------------------------------
# Schema description helper
# ---------------------------------------------------------------------------


def _describe_schema(schema_type: type) -> str:
    """Generate a text description of a Pydantic model or dataclass."""
    # Pydantic v2
    if hasattr(schema_type, "model_json_schema"):
        import json

        schema = schema_type.model_json_schema()
        return f"```json\n{json.dumps(schema, indent=2)}\n```"

    # Pydantic v1
    if hasattr(schema_type, "schema"):
        import json

        schema = schema_type.schema()
        return f"```json\n{json.dumps(schema, indent=2)}\n```"

    # dataclass
    from dataclasses import fields as dc_fields
    from dataclasses import is_dataclass

    if is_dataclass(schema_type):
        lines = ["{"]
        for f in dc_fields(schema_type):
            lines.append(f'  "{f.name}": {f.type}')
        lines.append("}")
        return "```\n" + "\n".join(lines) + "\n```"

    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# Agentic blocks — for use with the PromptGraph engine
# ═══════════════════════════════════════════════════════════════════════════════


class ToolsBlock:
    """Render available tool schemas for the LLM.

    Auto-formats tool names, descriptions, and parameter schemas
    from ``BaseTool`` instances.  Priority 9 (always included —
    tools are essential for agentic reasoning).

    Args:
        tools: List of LangChain ``BaseTool`` instances.
        show_schemas: Include full JSON parameter schemas.
        max_tools: Maximum tools to show (rest truncated with count).
    """

    def __init__(
        self,
        tools: list | None = None,
        *,
        show_schemas: bool = True,
        max_tools: int = 50,
    ) -> None:
        self._tools = list(tools) if tools else []
        self._show_schemas = show_schemas
        self._max_tools = max_tools

    @property
    def name(self) -> str:
        return "tools"

    @property
    def priority(self) -> int:
        return 9

    def render(self, ctx: BlockContext | None = None) -> str:
        if not self._tools:
            return ""
        lines = [f"Available tools ({len(self._tools)}):"]
        for i, tool in enumerate(self._tools[: self._max_tools]):
            desc = getattr(tool, "description", "") or ""
            lines.append(f"\n{i + 1}. **{tool.name}** — {desc}")
            if self._show_schemas:
                schema = getattr(tool, "args_schema", None)
                if schema and hasattr(schema, "model_json_schema"):
                    import json as _json

                    s = _json.dumps(schema.model_json_schema(), indent=2)
                    lines.append(f"   Parameters: ```json\n{s}\n```")
        if len(self._tools) > self._max_tools:
            lines.append(f"\n... and {len(self._tools) - self._max_tools} more tools")
        return "\n".join(lines)


class ObservationBlock:
    """Inject recent tool results into the prompt.

    Shows the most recent tool observations from state.
    Priority 6 (trimmed if token budget is tight).

    Args:
        observations: List of observation dicts.
        max_observations: Maximum observations to show.
        max_result_length: Truncate each result to this length.
    """

    def __init__(
        self,
        observations: list[dict] | None = None,
        *,
        max_observations: int = 5,
        max_result_length: int = 500,
    ) -> None:
        self._observations = list(observations) if observations else []
        self._max = max_observations
        self._max_len = max_result_length

    @property
    def name(self) -> str:
        return "observations"

    @property
    def priority(self) -> int:
        return 6

    def update(self, observations: list[dict]) -> ObservationBlock:
        """Return a new block with updated observations."""
        return ObservationBlock(
            observations=observations,
            max_observations=self._max,
            max_result_length=self._max_len,
        )

    def render(self, ctx: BlockContext | None = None) -> str:
        obs = self._observations[-self._max :] if self._observations else []
        if not obs:
            return ""
        lines = ["Recent tool results:"]
        for o in obs:
            tool = o.get("tool", "unknown")
            result = str(o.get("result", ""))
            success = "✓" if o.get("success", True) else "✗"
            lines.append(f"  [{success}] {tool} → {result[: self._max_len]}")
        return "\n".join(lines)


class PlanBlock:
    """Render the current plan with subgoal progress.

    Shows each subgoal with completion status.
    Priority 7 (important for multi-step reasoning).

    Args:
        subgoals: List of subgoal descriptions.
        completed: List of completed subgoal descriptions.
        active: The currently active subgoal.
    """

    def __init__(
        self,
        subgoals: list[str] | None = None,
        completed: list[str] | None = None,
        active: str = "",
    ) -> None:
        self._subgoals = list(subgoals) if subgoals else []
        self._completed = set(completed) if completed else set()
        self._active = active

    @property
    def name(self) -> str:
        return "plan"

    @property
    def priority(self) -> int:
        return 7

    def update(self, subgoals: list[str], completed: list[str], active: str) -> PlanBlock:
        """Return a new block with updated plan state."""
        return PlanBlock(subgoals=subgoals, completed=completed, active=active)

    def render(self, ctx: BlockContext | None = None) -> str:
        if not self._subgoals:
            return ""
        lines = ["Current plan:"]
        for sg in self._subgoals:
            if sg in self._completed:
                status = "✓"
            elif sg == self._active:
                status = "→"
            else:
                status = " "
            lines.append(f"  [{status}] {sg}")
        done = len(self._completed)
        total = len(self._subgoals)
        lines.append(f"  Progress: {done}/{total} complete")
        return "\n".join(lines)


class ReflectionBlock:
    """Inject past learnings from reflection/evaluation.

    Shows recent mistakes and corrections to prevent repeating them.
    Priority 4 (dropped early under token pressure).

    Args:
        reflections: List of reflection dicts with iteration,
            mistake, correction, confidence fields.
        max_reflections: Maximum reflections to show.
    """

    def __init__(
        self,
        reflections: list[dict] | None = None,
        *,
        max_reflections: int = 3,
    ) -> None:
        self._reflections = list(reflections) if reflections else []
        self._max = max_reflections

    @property
    def name(self) -> str:
        return "reflections"

    @property
    def priority(self) -> int:
        return 4

    def update(self, reflections: list[dict]) -> ReflectionBlock:
        """Return a new block with updated reflections."""
        return ReflectionBlock(reflections=reflections, max_reflections=self._max)

    def render(self, ctx: BlockContext | None = None) -> str:
        refs = self._reflections[-self._max :] if self._reflections else []
        if not refs:
            return ""
        lines = ["Past learnings (avoid repeating these mistakes):"]
        for r in refs:
            iteration = r.get("iteration", "?")
            mistake = r.get("mistake", "")
            correction = r.get("correction", "")
            lines.append(f"  Iter {iteration}: {mistake} → Fix: {correction}")
        return "\n".join(lines)


class PhaseBlock:
    """Stage-specific instructions that change per reasoning phase.

    Each phase has its own instructions. The block renders only
    the instructions for the current phase.
    Priority 8 (guides behavior at each step).

    Args:
        instructions: Mapping of phase names to instruction text.
        current_phase: The currently active phase.
    """

    def __init__(
        self,
        instructions: dict[str, str] | None = None,
        current_phase: str = "",
    ) -> None:
        self._instructions = dict(instructions) if instructions else {}
        self._current_phase = current_phase

    @property
    def name(self) -> str:
        return "phase"

    @property
    def priority(self) -> int:
        return 8

    def set_phase(self, phase: str) -> PhaseBlock:
        """Return a new block with the phase set."""
        return PhaseBlock(instructions=self._instructions, current_phase=phase)

    def render(self, ctx: BlockContext | None = None) -> str:
        phase = self._current_phase
        if ctx and hasattr(ctx, "metadata") and ctx.metadata:
            phase = ctx.metadata.get("phase", phase)
        text = self._instructions.get(phase, "")
        if not text:
            return ""
        return f"Current phase: {phase}\n{text}"
