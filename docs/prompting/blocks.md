# Layer 1: PromptBlocks

Build prompts from composable, priority-ranked blocks with intelligent assembly.

## Quick Example

```python
from promptise.prompts import prompt
from promptise.prompts.blocks import blocks, Identity, Rules, OutputFormat

@prompt(model="openai:gpt-5-mini")
@blocks(
    Identity("Expert data analyst", traits=["statistical thinking", "clear communication"]),
    Rules(["Cite specific data points", "Include confidence intervals"]),
    OutputFormat(format="markdown", instructions="Use ## headers for each section"),
)
async def analyze(dataset: str, question: str) -> str:
    """Given this dataset:
{dataset}

Answer this question: {question}"""

result = await analyze(
    dataset="Monthly users: Jan=10k, Feb=12k, Mar=15k",
    question="What is the growth trajectory?",
)
```

## Concepts

PromptBlocks are typed, reusable components that compose into system prompts. The `PromptAssembler` renders the blocks and joins them **in the order you list them**. Each block also has a **priority** (1--10). Priority never reorders the prompt: it only decides which blocks are dropped first when you set a [token budget](#token-budgeting).

Priority scale (higher survives a budget longer):

- **10** = Identity (the last to be dropped)
- **9** = Critical rules (Rules)
- **8** = Output specification (OutputFormat)
- **6** = Runtime context (ContextSlot)
- **5** = Custom sections (Section)
- **4** = Nice-to-have (Examples)
- **1--3** = Background, extras

## Block Types

### Identity

Defines who the agent is. Priority 10, so it is the last block dropped under a token budget.

```python
from promptise.prompts.blocks import Identity

identity = Identity(
    "Senior financial analyst",
    traits=["quantitative", "risk-aware", "concise"],
)
```

Renders as:

```text
You are Senior financial analyst.

Key traits:
- quantitative
- risk-aware
- concise
```

### Rules

Behavioral constraints for the agent. Priority 9.

```python
from promptise.prompts.blocks import Rules

rules = Rules([
    "Always cite specific numbers from the data",
    "Include a confidence level for each insight",
    "Keep analysis under 100 words",
])
```

### OutputFormat

Specifies the response structure. Priority 8. Accepts `format` (`"text"`, `"json"`, `"markdown"`), optional `schema` (Pydantic model or dataclass), and `instructions`.

```python
from promptise.prompts.blocks import OutputFormat
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    summary: str
    confidence: float
    recommendations: list[str]

fmt = OutputFormat(
    format="json",
    schema=AnalysisResult,
    instructions="Include exactly 3 recommendations",
)
```

### ContextSlot

Dynamic injection point filled at runtime. Priority configurable (default 6). Call `.fill()` to provide content -- it returns a new copy with the content set.

```python
from promptise.prompts.blocks import ContextSlot, PromptAssembler, Identity

slot = ContextSlot("user_data", priority=6, default="No data provided")

assembler = PromptAssembler(
    Identity("Analyst"),
    slot,
)

# Fill at runtime -- updates this assembler and returns it, for chaining
assembler.fill_slot("user_data", "Revenue: $2.3M, Growth: 15%")
assembled = assembler.assemble()
```

`PromptAssembler.fill_slot()` changes the assembler in place (like `add()` and `remove()`) and returns the same object. To keep an unfilled version, build a second assembler. `ContextSlot.fill()`, by contrast, returns a new block.

### Section

Custom named section with static text or a dynamic callable. Priority configurable (default 5).

```python
from promptise.prompts.blocks import Section

# Static content
static = Section("guidelines", "Follow company style guide v3.2", priority=5)

# Dynamic content -- callable receives BlockContext
dynamic = Section(
    "status",
    lambda ctx: f"Current phase: {ctx.phase}, turn: {ctx.turn}",
    priority=5,
)
```

### Examples

Few-shot examples with automatic truncation. Priority 4.

```python
from promptise.prompts.blocks import Examples

examples = Examples(
    [
        {"input": "Revenue up 10%", "output": "Positive growth trend"},
        {"input": "Revenue down 5%", "output": "Concerning decline"},
        {"input": "Revenue flat", "output": "Stable but stagnant"},
    ],
    max_count=2,  # Optionally cap the number of examples
)
```

The `Examples` block automatically manages the number of examples included.

### Conditional

Renders only when a condition is true. Inherits priority from the inner block.

```python
from promptise.prompts.blocks import Conditional, Rules, BlockContext

technical_rules = Conditional(
    "technical_rules",
    Rules(["Include error codes", "Reference API docs"]),
    condition=lambda ctx: ctx is not None and ctx.state.get("audience") == "technical",
)

general_rules = Conditional(
    "general_rules",
    Rules(["Use simple language", "Include analogies"]),
    condition=lambda ctx: ctx is not None and ctx.state.get("audience") == "general",
)
```

Assemble with a `BlockContext` to activate the right blocks:

```python
from promptise.prompts.blocks import PromptAssembler, Identity, BlockContext

assembler = PromptAssembler(Identity("Support agent"), technical_rules, general_rules)
ctx = BlockContext(state={"audience": "technical"})
assembled = assembler.assemble(ctx)
```

### Composite

Groups multiple blocks as a single unit. Priority is the maximum of all inner blocks.

```python
from promptise.prompts.blocks import Composite, Rules, Section

safety_package = Composite(
    "safety",
    [
        Rules(["Never reveal internal data"]),
        Section("disclaimer", "Responses are for informational purposes only."),
    ],
    separator="\n\n",
)
```

## PromptAssembler

The `PromptAssembler` renders and composes blocks into a final prompt.

```python
from promptise.prompts.blocks import (
    PromptAssembler, Identity, Rules, OutputFormat, Examples, Section
)

assembler = PromptAssembler(
    Identity("Senior financial analyst"),              # priority 10
    Rules(["Show reasoning", "Include risk assessment"]),  # priority 9
    OutputFormat(format="markdown"),                    # priority 8
    Section("background", long_background_text, priority=3),  # priority 3
    Examples(few_shot_examples),                        # priority 4
)

assembled = assembler.assemble()
print(assembled.included)
# ['identity', 'rules', 'output_format', 'background', 'examples']  -- list order
```

The prompt text follows the order of the blocks you pass, whatever their priority. Blocks that render to an empty string are left out.

### Token Budgeting

Pass `token_budget` (to the constructor or to `assemble()`) to cap the prompt's estimated size. When the blocks don't fit, the assembler drops the lowest-priority block first (and, among equal priorities, the one listed last) until the rest fit. The blocks that remain keep their list order.

```python
assembled = assembler.assemble(token_budget=300)
print(assembled.excluded)   # e.g. ['background'] -- priority 3 went first
```

Token counts are estimates (about 1.3 tokens per word), not a tokenizer count. A [`ConversationFlow`](flows.md#token-budget) takes the same `token_budget`.

### AssembledPrompt

The `assemble()` method returns an `AssembledPrompt` with full introspection:

| Attribute | Type | Description |
|-----------|------|-------------|
| `text` | `str` | The final assembled prompt string |
| `included` | `list[str]` | Names of blocks that made it in |
| `excluded` | `list[str]` | Names of blocks that were not included |
| `estimated_tokens` | `int` | Token estimate for the final text |
| `block_details` | `list[BlockTrace]` | Per-block trace with priority, tokens, render time |

### Chaining Methods

`PromptAssembler` supports method chaining:

```python
assembled = (
    PromptAssembler(Identity("Analyst"))
    .add(Rules(["Be concise"]))
    .fill_slot("data", csv_content)
    .assemble()
)
```

| Method | Description |
|--------|-------------|
| `add(block)` | Add a block, returns self |
| `remove(name)` | Remove a block by name, returns self |
| `fill_slot(name, content)` | Fill a `ContextSlot` by name, returns self |
| `assemble(ctx=None, *, token_budget=None)` | Build the final prompt, returns `AssembledPrompt` |

All three builder methods change the assembler in place and return it, so a chain and separate calls behave the same.

## The `@blocks` Decorator

The `@blocks` decorator attaches blocks to a `@prompt`-decorated function. It stacks **below** `@prompt`:

```python
from promptise.prompts import prompt
from promptise.prompts.blocks import blocks, Identity, Rules

@prompt(model="openai:gpt-5-mini")
@blocks(Identity("Expert analyst"), Rules(["Cite sources"]))
async def analyze(text: str) -> str:
    """Analyze: {text}"""
```

### Runtime Composition with `with_blocks()`

Attach blocks at runtime without the decorator:

```python
@prompt(model="openai:gpt-5-mini")
async def analyze(text: str) -> str:
    """Analyze: {text}"""

configured = analyze.with_blocks(
    Identity("Senior data analyst", traits=["precise"]),
    Rules(["Always cite numbers", "Keep under 100 words"]),
    OutputFormat(format="markdown"),
)
result = await configured("Q1 revenue: $2.3M, Q2: $2.8M")
```

## Block Protocol

Create custom block types by implementing the `Block` protocol:

```python
from promptise.prompts.blocks import Block, BlockContext

class Disclaimer:
    """Custom block that always adds a legal disclaimer."""

    @property
    def name(self) -> str:
        return "disclaimer"

    @property
    def priority(self) -> int:
        return 7

    def render(self, ctx: BlockContext | None = None) -> str:
        return "DISCLAIMER: This analysis is for informational purposes only."
```

!!! warning "Empty renders are excluded"
    If a block's `render()` returns an empty string, it is automatically excluded from the assembled prompt regardless of priority.

## Tips

- Use `assembled.block_details` to inspect per-block token usage and render times.
- `ContextSlot.fill()` returns a copy -- the original slot is unchanged, so you can reuse assemblers across requests.
- Combine `Conditional` blocks with `BlockContext.state` to build audience-aware or phase-aware prompts without manually managing if/else logic.

## What's Next?

- [Layer 2: ConversationFlow](flows.md) -- Evolve prompts across conversation turns
- [Strategies](strategies.md) -- Prompt strategies and composition patterns
