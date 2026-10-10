"""Model fallback chain for LLM provider reliability.

Wraps multiple LLM providers in a priority chain. If the primary model
fails (error, timeout, rate limit), the next model in the chain is tried
automatically. Each model has an independent circuit breaker to skip
known-broken providers and recover gracefully.

Example::

    from promptise import build_agent, FallbackChain

    agent = await build_agent(
        model=FallbackChain([
            "openai:gpt-5-mini",           # Primary
            "anthropic:claude-sonnet-4-20250514",    # Fallback 1
            "ollama:llama3",                # Fallback 2 (local, always up)
        ]),
        servers=servers,
    )
    # If OpenAI is down, Claude handles it. If both are down, local Llama.

Example with per-model timeouts::

    agent = await build_agent(
        model=FallbackChain(
            models=["openai:gpt-5-mini", "anthropic:claude-sonnet-4-20250514"],
            timeout_per_model=15.0,   # 15s per attempt
            global_timeout=30.0,      # 30s total across all attempts
        ),
        servers=servers,
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import ConfigDict

from .models import Model

logger = logging.getLogger("promptise.fallback")

__all__ = ["FallbackChain"]


# ---------------------------------------------------------------------------
# Circuit breaker (per-model)
# ---------------------------------------------------------------------------


@dataclass
class _CircuitState:
    """Tracks health of a single model in the chain."""

    model_id: str
    failures: int = 0
    last_failure: float = 0.0
    state: str = "closed"  # closed (healthy), open (skip), half_open (testing)
    failure_threshold: int = 3
    recovery_timeout: float = 60.0

    def record_success(self) -> None:
        """Reset on success — model is healthy."""
        self.failures = 0
        self.state = "closed"

    def record_failure(self) -> None:
        """Track failure. Open circuit after threshold."""
        self.failures += 1
        self.last_failure = time.monotonic()
        if self.failures >= self.failure_threshold:
            self.state = "open"
            logger.warning(
                "FallbackChain: circuit OPEN for %s after %d consecutive failures",
                self.model_id,
                self.failures,
            )

    def should_skip(self) -> bool:
        """Check if this model should be skipped."""
        if self.state == "closed":
            return False
        if self.state == "open":
            # Check if recovery timeout has elapsed
            elapsed = time.monotonic() - self.last_failure
            if elapsed >= self.recovery_timeout:
                self.state = "half_open"
                logger.info(
                    "FallbackChain: circuit HALF_OPEN for %s (testing recovery)",
                    self.model_id,
                )
                return False  # Try once
            return True  # Still broken, skip
        # half_open — allow one attempt
        return False


# ---------------------------------------------------------------------------
# FallbackChain
# ---------------------------------------------------------------------------


class FallbackChain(BaseChatModel):
    """Chain of LLM models with automatic failover.

    Tries models in order. If one fails (exception, timeout), the next
    is tried. Each model has an independent circuit breaker — after
    ``failure_threshold`` consecutive failures, the model is skipped
    for ``recovery_timeout`` seconds before being tested again.

    Passes through to ``build_agent(model=...)`` seamlessly — it's a
    ``BaseChatModel`` subclass, so LangChain treats it like any other model.

    Args:
        models: Ordered list of model identifiers (strings like
            ``"openai:gpt-5-mini"``) or ``BaseChatModel`` instances.
            First model is primary, rest are fallbacks.
        timeout_per_model: Maximum seconds per model attempt.
            ``0`` = no per-model timeout (use provider default).
        global_timeout: Maximum seconds across ALL attempts combined.
            ``0`` = unlimited (each model gets its full timeout).
        failure_threshold: Consecutive failures before a model's
            circuit breaker opens. Default: 3.
        recovery_timeout: Seconds before a tripped circuit breaker
            allows a test request. Default: 60.
        on_fallback: Optional callback ``(primary_model, fallback_model, error)``
            called each time a fallback is activated.

    Raises:
        ValueError: If ``models`` is empty.
        RuntimeError: If all models in the chain fail.
    """

    # Pydantic fields (BaseChatModel requires these)
    models: list[Any] = []
    timeout_per_model: float = 0
    global_timeout: float = 0
    failure_threshold: int = 3
    recovery_timeout: float = 60.0
    on_fallback: Any = None

    # Internal state (not serialized)
    _resolved: list[BaseChatModel] = []
    _circuits: list[_CircuitState] = []
    _model_ids: list[str] = []
    _initialized: bool = False
    _last_serving_model: str = ""  # Tracks which model actually served the last request

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        models: Sequence[str | Model | BaseChatModel] | None = None,
        *,
        timeout_per_model: float = 0,
        global_timeout: float = 0,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        on_fallback: Any = None,
        **kwargs: Any,
    ) -> None:
        # BaseChatModel uses Pydantic; these are declared as class-level fields
        # above which Pydantic routes through to __init__, but mypy cannot see
        # them on BaseChatModel's typed signature.
        super().__init__(  # type: ignore[call-arg]
            models=list(models or []),
            timeout_per_model=timeout_per_model,
            global_timeout=global_timeout,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            on_fallback=on_fallback,
            **kwargs,
        )
        if not self.models:
            raise ValueError("FallbackChain requires at least one model")

        self._model_ids = []
        self._resolved = []
        self._circuits = []
        self._initialized = False

    def _ensure_resolved(self) -> None:
        """Lazily resolve model strings to BaseChatModel instances.

        Builds into temporary lists and assigns atomically to prevent
        duplicate entries if resolution fails partway through.
        """
        if self._initialized:
            return

        from .models import Model, resolve_model

        # Build into temps — if any model fails to resolve, no partial state
        ids: list[str] = []
        resolved: list[Any] = []
        circuits: list[_CircuitState] = []

        for m in self.models:
            if isinstance(m, str):
                # Same resolver as build_agent(): aliases, .env and actionable errors.
                model_id = m
                model_obj = resolve_model(m)
            elif isinstance(m, Model):
                model_id = m.spec
                model_obj = m.resolve()
            elif (
                hasattr(m, "_generate") or hasattr(m, "_agenerate") or isinstance(m, BaseChatModel)
            ):
                model_id = (
                    getattr(m, "model_name", None)
                    or getattr(m, "model", None)
                    or str(type(m).__name__)
                )
                model_obj = m
            else:
                raise TypeError(
                    f"Expected str or model with _generate/_agenerate, got {type(m).__name__}"
                )

            ids.append(str(model_id))
            resolved.append(model_obj)
            circuits.append(
                _CircuitState(
                    model_id=str(model_id),
                    failure_threshold=self.failure_threshold,
                    recovery_timeout=self.recovery_timeout,
                )
            )

        # Atomic assignment — either all or nothing
        self._model_ids = ids
        self._resolved = resolved
        self._circuits = circuits
        self._initialized = True
        logger.info(
            "FallbackChain initialized: %s",
            " → ".join(self._model_ids),
        )

    @property
    def _llm_type(self) -> str:
        return "fallback-chain"

    @property
    def model_name(self) -> str:
        """Return the model that last served a request.

        Before any request is made, returns the primary model's name.
        After a request, returns the model that actually served it —
        this is what observability and cache use.
        """
        self._ensure_resolved()
        if self._last_serving_model:
            return self._last_serving_model
        return self._model_ids[0] if self._model_ids else "fallback-chain"

    @property
    def active_model(self) -> str:
        """Return the first non-skipped model's name."""
        self._ensure_resolved()
        for i, circuit in enumerate(self._circuits):
            if not circuit.should_skip():
                return self._model_ids[i]
        return self._model_ids[0]  # All tripped — try primary anyway

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Synchronous generation with fallback (used by LangChain internally)."""
        self._ensure_resolved()

        global_deadline = (
            time.monotonic() + self.global_timeout if self.global_timeout > 0 else float("inf")
        )
        errors: list[tuple[str, Exception]] = []

        for i, (model, circuit) in enumerate(zip(self._resolved, self._circuits, strict=False)):
            if circuit.should_skip():
                continue

            if time.monotonic() >= global_deadline:
                break

            try:
                result = model._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
                circuit.record_success()
                self._last_serving_model = self._model_ids[i]
                return result
            except Exception as exc:
                errors.append((self._model_ids[i], exc))
                circuit.record_failure()
                logger.warning(
                    "FallbackChain: %s failed (%s), trying next",
                    self._model_ids[i],
                    type(exc).__name__,
                )
                if self.on_fallback and i + 1 < len(self._resolved):
                    try:
                        self.on_fallback(self._model_ids[i], self._model_ids[i + 1], exc)
                    except Exception:
                        pass

        detail = "\n".join(f"  {mid}: {type(err).__name__}: {err}" for mid, err in errors)
        skipped = [self._model_ids[i] for i, c in enumerate(self._circuits) if c.state == "open"]
        if skipped:
            detail += f"\n  Skipped (circuit open): {', '.join(skipped)}"
        raise RuntimeError(f"All {len(self._resolved)} models in FallbackChain failed.\n{detail}")

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation with fallback and per-model timeouts."""
        self._ensure_resolved()

        global_deadline = (
            time.monotonic() + self.global_timeout if self.global_timeout > 0 else float("inf")
        )
        errors: list[tuple[str, Exception]] = []

        for i, (model, circuit) in enumerate(zip(self._resolved, self._circuits, strict=False)):
            if circuit.should_skip():
                continue

            remaining_global = global_deadline - time.monotonic()
            if remaining_global <= 0:
                break

            # Determine timeout for this attempt
            if self.timeout_per_model > 0:
                timeout = min(self.timeout_per_model, remaining_global)
            elif remaining_global < float("inf"):
                timeout = remaining_global
            else:
                timeout = None  # No timeout

            try:
                coro = model._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
                if timeout:
                    result = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    result = await coro
                circuit.record_success()
                self._last_serving_model = self._model_ids[i]
                return result
            except Exception as exc:
                errors.append((self._model_ids[i], exc))
                circuit.record_failure()
                logger.warning(
                    "FallbackChain: %s failed (%s: %s), trying next",
                    self._model_ids[i],
                    type(exc).__name__,
                    str(exc)[:100],
                )
                if self.on_fallback and i + 1 < len(self._resolved):
                    try:
                        self.on_fallback(self._model_ids[i], self._model_ids[i + 1], exc)
                    except Exception:
                        pass

        detail = "\n".join(f"  {mid}: {type(err).__name__}: {err}" for mid, err in errors)
        skipped = [self._model_ids[i] for i, c in enumerate(self._circuits) if c.state == "open"]
        if skipped:
            detail += f"\n  Skipped (circuit open): {', '.join(skipped)}"
        raise RuntimeError(f"All {len(self._resolved)} models in FallbackChain failed.\n{detail}")

    def get_chain_status(self) -> list[dict[str, Any]]:
        """Get the health status of each model in the chain.

        Returns:
            List of dicts with ``model_id``, ``state`` (closed/open/half_open),
            ``failures``, and ``is_primary``.
        """
        self._ensure_resolved()
        return [
            {
                "model_id": circuit.model_id,
                "state": circuit.state,
                "failures": circuit.failures,
                "is_primary": i == 0,
            }
            for i, circuit in enumerate(self._circuits)
        ]
