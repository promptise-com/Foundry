"""Context Engine — unified context assembly with token budgeting.

Opt-in system that replaces the ad-hoc context injection in
``_ainvoke_inner()`` with a single-pass assembly pipeline.  Each
context source is a :class:`ContextLayer` with a name, priority,
and content.  When the total context exceeds the model's window,
lowest-priority layers are trimmed first.

Example::

    from promptise import build_agent, ContextEngine

    engine = ContextEngine(model_context_window=128_000)
    agent = await build_agent(
        ...,
        context_engine=engine,
    )
"""

from __future__ import annotations

import logging
import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("promptise.context_engine")

__all__ = [
    "ContextEngine",
    "ContextLayer",
    "ContextReport",
    "Tokenizer",
]


# ---------------------------------------------------------------------------
# Tokenizer protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for token counters.

    Implementations must provide a ``count(text) -> int`` method that
    returns the exact (or best-estimate) token count for a string.
    """

    def count(self, text: str) -> int: ...


class _TiktokenCounter:
    """Exact token counter using OpenAI's tiktoken library."""

    def __init__(self, model: str = "gpt-4") -> None:
        try:
            import tiktoken

            # Try model-specific encoding, fall back to cl100k_base
            try:
                self._enc = tiktoken.encoding_for_model(model)
            except KeyError:
                self._enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            raise ImportError(
                "tiktoken is required for exact token counting. Install with: pip install tiktoken"
            )

    def count(self, text: str) -> int:
        """Count tokens using tiktoken (exact for OpenAI models)."""
        return len(self._enc.encode(text))


class _EstimateCounter:
    """Fallback token counter using character-based estimation.

    Uses chars/3.5 which is ~90% accurate for English text across
    most tokenizers (GPT, Claude, Llama).
    """

    def count(self, text: str) -> int:
        """Estimate token count from character length."""
        if not text:
            return 0
        return max(1, math.ceil(len(text) / 3.5))


# ---------------------------------------------------------------------------
# Context layer
# ---------------------------------------------------------------------------


@dataclass
class ContextLayer:
    """A single context source in the assembly pipeline.

    Attributes:
        name: Unique layer identifier (e.g. ``"memory"``, ``"strategies"``).
        priority: Assembly priority (0-10).  Higher = kept longer when
            trimming.  Required layers (priority >= 9) are never dropped.
        content: The text content for this layer.
        required: If ``True``, this layer is never trimmed (overrides priority).
        trim_strategy: How to trim this layer — ``"truncate"`` (cut from end)
            or ``"oldest_first"`` (for conversation history, removes oldest
            messages first).
        metadata: Developer-provided metadata for reporting.
    """

    name: str
    priority: int = 5
    content: str = ""
    required: bool = False
    trim_strategy: str = "truncate"  # "truncate" or "oldest_first"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        """Rough token estimate (chars / 3.5) for quick checks."""
        return max(0, math.ceil(len(self.content) / 3.5)) if self.content else 0


@dataclass
class ContextReport:
    """Report of context assembly — what was included, trimmed, and total usage.

    Attributes:
        total_tokens: Total tokens in the assembled context.
        budget: The model's context window minus response reserve.
        layers: Per-layer token counts and trim status.
        trimmed_layers: Layers that were trimmed or dropped.
        utilization: Fraction of budget used (0.0-1.0).
    """

    total_tokens: int = 0
    budget: int = 0
    layers: list[dict[str, Any]] = field(default_factory=list)
    trimmed_layers: list[str] = field(default_factory=list)

    @property
    def utilization(self) -> float:
        """Fraction of budget used."""
        return self.total_tokens / self.budget if self.budget > 0 else 0.0


# ---------------------------------------------------------------------------
# Context Engine
# ---------------------------------------------------------------------------


# Well-known model context windows (input + output), as the providers list them.
# build_agent prefers the chat model's own profile metadata when it has one.
_MODEL_WINDOWS: dict[str, int] = {
    "gpt-4": 8_192,
    "gpt-4-turbo": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4.1": 1_047_576,
    "gpt-4.1-mini": 1_047_576,
    "gpt-4.1-nano": 1_047_576,
    "gpt-5": 400_000,
    "gpt-5-mini": 400_000,
    "gpt-5-nano": 400_000,
    "o4-mini": 200_000,
    "claude-3-haiku": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-opus": 200_000,
    "claude-sonnet-4": 200_000,
    "claude-opus-4": 200_000,
    "llama3": 8_192,
    "llama3-70b": 8_192,
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-pro": 32_768,
    "mistral-large": 128_000,
}


# Models whose input is capped below window - output: the API rejects a
# request whose input passes this, whatever the response reserve.
_MODEL_MAX_INPUT: dict[str, int] = {
    "gpt-5": 272_000,
    "gpt-5-mini": 272_000,
    "gpt-5-nano": 272_000,
}


def _lookup(table: dict[str, int], model: str | None) -> int | None:
    """Exact, then longest-substring match of *model* (provider prefix stripped)."""
    if not model:
        return None
    bare = model.split(":")[-1] if ":" in model else model
    if bare in table:
        return table[bare]
    bare_lower = bare.lower()
    for name in sorted(table.keys(), key=len, reverse=True):
        if name in bare_lower:
            return table[name]
    return None


def _detect_context_window(model: str | None) -> int:
    """Detect context window from model name string.

    Falls back to 128K if model is unknown.
    """
    return _lookup(_MODEL_WINDOWS, model) or 128_000


class ContextEngine:
    """Unified context assembly with token budgeting.

    Replaces ad-hoc context injection with a single-pass pipeline
    that respects the model's context window and trims by priority.

    Args:
        model_context_window: Total context window in tokens.
            Auto-detected from ``model`` if not provided.
        model: Model name string for auto-detection and tokenizer selection.
        response_reserve: Tokens reserved for the model's response.
        tokenizer: Custom tokenizer.  If ``None``, uses tiktoken for
            OpenAI models (exact) or character estimation (fallback).
        auto_register_builtins: Register built-in layers (identity,
            memory, strategies, etc.) automatically.

    Example::

        engine = ContextEngine(model="openai:gpt-5-mini")
        engine.add_layer("company_policy", priority=7, content="We are Acme Corp...")
        agent = await build_agent(..., context_engine=engine)
    """

    def __init__(
        self,
        *,
        model_context_window: int = 0,
        model: str | None = None,
        response_reserve: int = 4096,
        tokenizer: Tokenizer | None = None,
        auto_register_builtins: bool = True,
    ) -> None:
        self._model = model
        self._window_explicit = bool(model_context_window)
        self._window = model_context_window or _detect_context_window(model)
        self._max_input = None if self._window_explicit else _lookup(_MODEL_MAX_INPUT, model)
        self._response_reserve = response_reserve
        self._budget = self._compute_budget()

        # Tokenizer: try tiktoken (exact), fall back to estimation
        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = self._build_tokenizer(model)

        # Layer registry: name → ContextLayer
        self._layers: dict[str, ContextLayer] = {}

        # Register built-in layers with default priorities
        if auto_register_builtins:
            self._register_builtins()

        # Last assembly report
        self._last_report: ContextReport | None = None

        logger.info(
            "ContextEngine: window=%d, reserve=%d, budget=%d, tokenizer=%s",
            self._window,
            self._response_reserve,
            self._budget,
            type(self._tokenizer).__name__,
        )

    def _compute_budget(self) -> int:
        budget = self._window - self._response_reserve
        if self._max_input is not None:
            budget = min(budget, self._max_input)
        return budget

    def _apply_model_profile(self, chat_model: Any) -> bool:
        """Take the window from a chat model's profile metadata.

        LangChain chat models carry a ``profile`` with ``max_input_tokens``
        and ``max_output_tokens``.  Used unless ``model_context_window``
        was passed explicitly.  Returns whether the limits changed.
        """
        if self._window_explicit:
            return False
        profile = getattr(chat_model, "profile", None)
        if not isinstance(profile, Mapping):
            return False
        max_in = profile.get("max_input_tokens")
        if not isinstance(max_in, int) or max_in <= 0:
            return False
        max_out = profile.get("max_output_tokens")
        self._max_input = max_in
        self._window = max_in + (max_out if isinstance(max_out, int) and max_out > 0 else 0)
        self._budget = self._compute_budget()
        logger.info(
            "ContextEngine: window=%d, max input=%d from the model profile, budget=%d",
            self._window,
            max_in,
            self._budget,
        )
        return True

    @staticmethod
    def _build_tokenizer(model: str | None) -> Tokenizer:
        """Build the best available tokenizer."""
        # Try tiktoken for OpenAI-compatible models
        if model and ("openai" in model or "gpt" in model.lower()):
            try:
                bare = model.split(":")[-1] if ":" in model else model
                return _TiktokenCounter(bare)
            except ImportError:
                pass

        # Try tiktoken with default encoding
        try:
            return _TiktokenCounter()
        except ImportError:
            pass

        # Fallback to estimation
        return _EstimateCounter()

    def _register_builtins(self) -> None:
        """Register the standard context layers with default priorities."""
        builtins = [
            ("identity", 10, True),  # Agent identity/instructions
            ("tools", 9, True),  # Tool definitions (never drop)
            ("user_message", 10, True),  # Current user query (never drop)
            ("flow", 9, False),  # Conversation flow phase
            ("prompt_blocks", 8, False),  # PromptBlocks assembly
            ("output_format", 8, False),  # Expected output structure
            ("context_state", 6, False),  # Runtime: AgentContext state
            ("mission", 6, False),  # Runtime: mission objective
            ("budget", 5, False),  # Runtime: budget remaining
            ("inbox", 4, False),  # Runtime: human operator messages
            ("memory", 3, False),  # Long-term memory recall
            ("strategies", 2, False),  # Learned strategies
            ("conversation", 1, False),  # Conversation history
        ]
        for name, priority, required in builtins:
            self._layers[name] = ContextLayer(
                name=name,
                priority=priority,
                required=required,
                trim_strategy="oldest_first" if name == "conversation" else "truncate",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def budget(self) -> int:
        """Available token budget (window - response reserve)."""
        return self._budget

    @property
    def window(self) -> int:
        """Model context window size."""
        return self._window

    @property
    def max_input_tokens(self) -> int | None:
        """The model's input limit, when known (it caps :attr:`budget`)."""
        return self._max_input

    @property
    def last_report(self) -> ContextReport | None:
        """Report from the most recent assembly."""
        return self._last_report

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        if not text:
            return 0
        return self._tokenizer.count(text)

    def add_layer(
        self,
        name: str,
        *,
        priority: int = 5,
        content: str = "",
        required: bool = False,
        trim_strategy: str = "truncate",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Register a custom context layer.

        Args:
            name: Unique layer name.
            priority: 0-10 (higher = kept longer when trimming).
            content: Initial content (can be updated later via ``set_content``).
            required: Never trim this layer.
            trim_strategy: ``"truncate"`` or ``"oldest_first"``.
            metadata: Custom metadata for reporting.
        """
        self._layers[name] = ContextLayer(
            name=name,
            priority=min(10, max(0, priority)),
            content=content,
            required=required,
            trim_strategy=trim_strategy,
            metadata=metadata or {},
        )

    def set_content(self, name: str, content: str) -> None:
        """Set content for a registered layer.

        Args:
            name: Layer name (must be registered via ``add_layer`` or a builtin).
            content: The text content for this layer.

        Raises:
            KeyError: If the layer name is not registered.
        """
        if name not in self._layers:
            raise KeyError(
                f"Context layer '{name}' not registered. Available: {list(self._layers.keys())}"
            )
        self._layers[name].content = content

    def get_content(self, name: str) -> str:
        """Get current content for a layer."""
        layer = self._layers.get(name)
        return layer.content if layer else ""

    def clear_content(self, name: str) -> None:
        """Clear content for a layer (sets to empty string)."""
        if name in self._layers:
            self._layers[name].content = ""

    def clear_all(self) -> None:
        """Clear content from all layers (keeps registrations)."""
        for layer in self._layers.values():
            layer.content = ""

    def remove_layer(self, name: str) -> None:
        """Remove a custom layer entirely."""
        self._layers.pop(name, None)

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    def assemble(
        self,
        content: Mapping[str, str] | None = None,
        *,
        budget_only: Collection[str] = (),
    ) -> list[dict[str, Any]]:
        """Assemble all layers into a message array, respecting token budget.

        Returns a list of ``{"role": "system", "content": "..."}`` dicts
        for system context, plus the user message and conversation history.

        Layers with empty content are skipped.  When total tokens exceed
        the budget, lowest-priority non-required layers are trimmed first.
        Assembly never changes the registered layers.

        Args:
            content: Content for this assembly only, by layer name.  It
                overrides the stored content without changing it, so
                concurrent assemblies do not interfere.
            budget_only: Layers that count toward the budget and appear in
                the report but are not returned as messages, because
                something else sends them (``build_agent`` sends the
                instructions and tool definitions itself).

        Returns:
            Ordered list of message dicts ready for LangGraph.

        Raises:
            KeyError: If *content* names a layer that is not registered.
        """
        overrides = dict(content or {})
        unknown = [name for name in overrides if name not in self._layers]
        if unknown:
            raise KeyError(
                f"Context layer(s) {unknown} not registered. Available: {list(self._layers.keys())}"
            )
        # Work on copies: trimming changes content, the registry never does.
        working = [
            replace(layer, content=overrides.get(name, layer.content))
            for name, layer in self._layers.items()
        ]

        # Collect non-empty layers
        active: list[tuple[ContextLayer, int]] = []  # (layer, tokens)
        for layer in working:
            if not layer.content:
                continue
            tokens = self._tokenizer.count(layer.content)
            active.append((layer, tokens))

        # Sort by priority descending (highest priority first)
        active.sort(key=lambda x: x[0].priority, reverse=True)

        # Calculate total
        total = sum(t for _, t in active)

        # Trim if over budget
        trimmed_names: list[str] = []
        if total > self._budget:
            total, trimmed_names = self._trim(active, total)

        # Build message array in correct order for LangGraph:
        # 1. System messages (sorted by priority, highest first)
        # 2. Conversation history (chronological)
        # 3. Current user message (ALWAYS last)
        system_layers: list[tuple[ContextLayer, int]] = []
        conversation_layer: tuple[ContextLayer, int] | None = None
        user_layer: tuple[ContextLayer, int] | None = None
        not_sent: list[ContextLayer] = []

        for layer, tokens in active:
            if not layer.content:
                continue
            if layer.name in budget_only:
                not_sent.append(layer)
            elif layer.name == "user_message":
                user_layer = (layer, tokens)
            elif layer.name == "conversation":
                conversation_layer = (layer, tokens)
            else:
                system_layers.append((layer, tokens))

        messages: list[dict[str, Any]] = []
        layer_reports: list[dict[str, Any]] = []

        def _report(layer: ContextLayer, sent: bool = True) -> None:
            layer_reports.append(
                {
                    "name": layer.name,
                    "priority": layer.priority,
                    "tokens": self._tokenizer.count(layer.content),
                    "required": layer.required,
                    "trimmed": layer.name in trimmed_names,
                    "sent": sent,
                }
            )

        for layer in not_sent:
            _report(layer, sent=False)

        # 1. System messages (already sorted by priority from active list)
        for layer, _tokens in system_layers:
            messages.append({"role": "system", "content": layer.content})
            _report(layer)

        # 2. Conversation history (chronological order)
        if conversation_layer is not None:
            layer, _tokens = conversation_layer
            messages.extend(self._parse_conversation(layer.content))
            _report(layer)

        # 3. User message ALWAYS last
        if user_layer is not None:
            layer, _tokens = user_layer
            messages.append({"role": "user", "content": layer.content})
            _report(layer)

        # Store report
        self._last_report = ContextReport(
            total_tokens=sum(r["tokens"] for r in layer_reports),
            budget=self._budget,
            layers=layer_reports,
            trimmed_layers=trimmed_names,
        )

        logger.debug(
            "ContextEngine: assembled %d layers, %d/%d tokens (%.0f%% utilization), %d trimmed",
            len(layer_reports),
            self._last_report.total_tokens,
            self._budget,
            self._last_report.utilization * 100,
            len(trimmed_names),
        )

        return messages

    def _trim(
        self,
        active: list[tuple[ContextLayer, int]],
        total: int,
    ) -> tuple[int, list[str]]:
        """Trim lowest-priority layers until total fits within budget.

        Returns (new_total, list_of_trimmed_layer_names).
        """
        trimmed: list[str] = []
        overflow = total - self._budget

        # Sort candidates by priority ascending (lowest priority first = trim first)
        candidates = [
            (layer, tokens) for layer, tokens in active if not layer.required and layer.content
        ]
        candidates.sort(key=lambda x: x[0].priority)

        for layer, tokens in candidates:
            if overflow <= 0:
                break

            if layer.trim_strategy == "oldest_first" and layer.name == "conversation":
                # Trim oldest messages from conversation
                trimmed_content, saved = self._trim_conversation(layer.content, tokens, overflow)
                layer.content = trimmed_content
                overflow -= saved
                total -= saved
                if saved > 0:
                    trimmed.append(layer.name)
            else:
                # Truncate from the end
                if overflow >= tokens:
                    # Drop entirely
                    layer.content = ""
                    overflow -= tokens
                    total -= tokens
                    trimmed.append(layer.name)
                else:
                    # Partial truncation — use binary search with actual tokenizer
                    target_tokens = tokens - overflow
                    low, high = 0, len(layer.content)
                    while low < high:
                        mid = (low + high + 1) // 2
                        if self._tokenizer.count(layer.content[:mid]) <= target_tokens:
                            low = mid
                        else:
                            high = mid - 1
                    layer.content = layer.content[:low].rstrip() + "..."
                    new_tokens = self._tokenizer.count(layer.content)
                    total -= tokens - new_tokens
                    overflow -= tokens - new_tokens
                    trimmed.append(layer.name)

        if overflow > 0:
            logger.warning(
                "ContextEngine: still %d tokens over budget after trimming "
                "all non-required layers. Required layers consume %d tokens "
                "(budget: %d).",
                overflow,
                total,
                self._budget,
            )

        return total, trimmed

    def _trim_conversation(
        self, content: str, current_tokens: int, overflow: int
    ) -> tuple[str, int]:
        """Trim oldest messages from conversation history.

        Parses into user/assistant pairs first, then removes entire
        pairs from the oldest end.  Never splits a pair.
        """
        # Parse into structured pairs
        parsed = self._parse_conversation(content)
        if not parsed:
            return "", current_tokens

        # Group into pairs (user + assistant)
        pairs: list[list[dict[str, Any]]] = []
        current_pair: list[dict[str, Any]] = []
        for msg in parsed:
            current_pair.append(msg)
            if msg["role"] in ("assistant", "ai"):
                pairs.append(current_pair)
                current_pair = []
        if current_pair:  # Trailing user message without response
            pairs.append(current_pair)

        # Remove oldest pairs until we've saved enough tokens
        saved = 0
        while pairs and saved < overflow:
            removed_pair = pairs.pop(0)
            for msg in removed_pair:
                saved += self._tokenizer.count(msg["content"])

        # Reconstruct content from remaining pairs
        remaining_lines: list[str] = []
        for pair in pairs:
            for msg in pair:
                prefix = "User" if msg["role"] == "user" else "Assistant"
                remaining_lines.append(f"{prefix}: {msg['content']}")

        return "\n".join(remaining_lines), saved

    @staticmethod
    def _parse_conversation(content: str) -> list[dict[str, Any]]:
        """Parse conversation content into message dicts."""
        messages: list[dict[str, Any]] = []
        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith(("User:", "Human:", "user:")):
                content = line.split(":", 1)[1].strip()
                if content:
                    messages.append({"role": "user", "content": content})
            elif line.startswith(("Assistant:", "AI:", "assistant:")):
                content = line.split(":", 1)[1].strip()
                if content:
                    messages.append({"role": "assistant", "content": content})
            elif line.startswith(("System:", "system:")):
                content = line.split(":", 1)[1].strip()
                if content:
                    messages.append({"role": "system", "content": content})
            else:
                # Continuation of previous message
                if messages:
                    messages[-1]["content"] += "\n" + line
        return messages

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def get_report(self) -> ContextReport | None:
        """Get the most recent assembly report.

        Returns ``None`` if ``assemble()`` has not been called yet.
        """
        return self._last_report

    def get_layer_info(self) -> list[dict[str, Any]]:
        """Get info about all registered layers."""
        return [
            {
                "name": layer.name,
                "priority": layer.priority,
                "required": layer.required,
                "has_content": bool(layer.content),
                "estimated_tokens": layer.token_estimate,
                "trim_strategy": layer.trim_strategy,
            }
            for layer in sorted(self._layers.values(), key=lambda l: l.priority, reverse=True)
        ]
