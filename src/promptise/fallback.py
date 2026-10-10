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

Tools work as with any chat model: ``build_agent()`` calls
:meth:`FallbackChain.bind_tools`, which binds the tools to every model in
the chain, so a fallback model can call them too.

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
import json
import logging
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.messages.tool import tool_call_chunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableBinding
from langchain_core.tools import BaseTool
from pydantic import ConfigDict

from .models import Model

logger = logging.getLogger("promptise.fallback")

__all__ = ["FallbackChain", "is_request_error"]


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
    :meth:`bind_tools` binds tools to every model in the chain.

    Each answer records the model that wrote it in
    ``AIMessage.response_metadata["fallback_model"]``.

    Every exception moves on to the next model, but a request the provider
    rejects as invalid (HTTP 400/413/422, see :func:`is_request_error`)
    does not count toward the circuit breaker.  Streaming falls back only
    until the first chunk: after that the answer is committed to the model
    that sent it, and a failure is raised.

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
    # Per model: extra call kwargs from bind_tools() (a dict), or a bound
    # Runnable when the model's binding is not a plain kwargs binding.
    _bindings: list[Any] = []
    # The chain a bind_tools() copy came from; it reports the serving model.
    _root: Any = None

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
        self._bindings = []
        self._root = None

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
        root = self._root or self
        if root._last_serving_model:
            return str(root._last_serving_model)
        return self._model_ids[0] if self._model_ids else "fallback-chain"

    @property
    def active_model(self) -> str:
        """Return the first non-skipped model's name."""
        self._ensure_resolved()
        for i, circuit in enumerate(self._circuits):
            if not circuit.should_skip():
                return self._model_ids[i]
        return self._model_ids[0]  # All tripped — try primary anyway

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> FallbackChain:
        """Bind tools to every model in the chain.

        Returns a copy of the chain whose models all have the tools bound.
        The copy shares this chain's circuit breakers, so failures seen
        while calling tools count toward the same thresholds, and
        :attr:`model_name` / :meth:`get_chain_status` on this chain reflect
        requests served through the copy.

        Raises:
            NotImplementedError: If a model in the chain cannot call tools.
        """
        self._ensure_resolved()
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        bindings: list[Any] = []
        for model_id, model in zip(self._model_ids, self._resolved, strict=True):
            try:
                bound = model.bind_tools(tools, **kwargs)
            except NotImplementedError as exc:
                raise NotImplementedError(
                    f"FallbackChain: model {model_id!r} does not support tool calling, "
                    "so it cannot serve an agent with tools. Remove it from the chain "
                    "or use a model that supports tools."
                ) from exc
            if isinstance(bound, RunnableBinding) and bound.bound is model:
                bindings.append(dict(bound.kwargs))
            else:
                bindings.append(bound)
        chain = self.model_copy()
        chain._bindings = bindings
        chain._root = self._root or self
        return chain

    def _mark_serving(self, index: int) -> None:
        """Record that model ``index`` is answering the current request."""
        model_id = self._model_ids[index]
        (self._root or self)._last_serving_model = model_id
        self._last_serving_model = model_id

    def _served(self, index: int, result: ChatResult) -> ChatResult:
        """Record that model ``index`` served ``result``; tag its messages."""
        model_id = self._model_ids[index]
        self._mark_serving(index)
        for generation in result.generations:
            message = getattr(generation, "message", None)
            if message is not None:
                message.response_metadata["fallback_model"] = model_id
        return result

    def _binding(self, index: int) -> Any:
        return self._bindings[index] if self._bindings else {}

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
                binding = self._binding(i)
                if isinstance(binding, Runnable):
                    if stop is not None:
                        kwargs = {**kwargs, "stop": stop}
                    message = binding.invoke(messages, **kwargs)
                    result = ChatResult(generations=[ChatGeneration(message=message)])
                else:
                    result = model._generate(
                        messages, stop=stop, run_manager=run_manager, **{**binding, **kwargs}
                    )
                circuit.record_success()
                return self._served(i, result)
            except Exception as exc:
                self._attempt_failed(i, exc, errors)

        raise self._all_failed(errors)

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
                binding = self._binding(i)
                if isinstance(binding, Runnable):
                    coro = _ainvoke_as_result(binding, messages, stop, kwargs)
                else:
                    coro = model._agenerate(
                        messages, stop=stop, run_manager=run_manager, **{**binding, **kwargs}
                    )
                if timeout:
                    result = await asyncio.wait_for(coro, timeout=timeout)
                else:
                    result = await coro
                circuit.record_success()
                return self._served(i, result)
            except Exception as exc:
                self._attempt_failed(i, exc, errors)

        raise self._all_failed(errors)

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream from the first model that starts answering.

        A model that fails (or times out) before its first chunk is skipped
        for the next, as in :meth:`_agenerate`.  Once a model has sent a
        chunk the answer is committed to it: a failure after that is raised,
        since the chunks already sent cannot be taken back.  With
        ``timeout_per_model`` / ``global_timeout``, the timeout bounds the
        wait for the first chunk.  A model that cannot stream answers in
        one chunk.
        """
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
            if self.timeout_per_model > 0:
                timeout: float | None = min(self.timeout_per_model, remaining_global)
            elif remaining_global < float("inf"):
                timeout = remaining_global
            else:
                timeout = None

            stream = self._member_astream(i, model, messages, stop, kwargs)
            try:
                first = stream.__anext__()
                chunk = await (asyncio.wait_for(first, timeout) if timeout else first)
            except StopAsyncIteration:
                circuit.record_success()
                self._mark_serving(i)
                return
            except Exception as exc:
                await _aclose_quietly(stream)
                self._attempt_failed(i, exc, errors)
                continue

            # Committed: from here on the caller has this model's output.
            self._mark_serving(i)
            chunk.message.response_metadata["fallback_model"] = self._model_ids[i]
            yield chunk
            try:
                async for chunk in stream:
                    yield chunk
            except Exception as exc:
                if not is_request_error(exc):
                    circuit.record_failure()
                raise
            circuit.record_success()
            return

        raise self._all_failed(errors)

    async def _member_astream(
        self,
        index: int,
        model: BaseChatModel,
        messages: list[BaseMessage],
        stop: list[str] | None,
        kwargs: dict[str, Any],
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream one chain member's answer as generation chunks."""
        binding = self._binding(index)
        if isinstance(binding, Runnable):
            call_kwargs = {**kwargs, "stop": stop} if stop is not None else kwargs
            async for message in binding.astream(messages, **call_kwargs):
                yield ChatGenerationChunk(message=_as_chunk(message))
            return
        call_kwargs = {**binding, **kwargs}
        if _can_stream(model):
            # No run_manager: the chain's own run reports each token once.
            async for chunk in model._astream(messages, stop=stop, **call_kwargs):
                yield chunk
            return
        result = await model._agenerate(messages, stop=stop, **call_kwargs)
        for generation in result.generations:
            yield ChatGenerationChunk(
                message=_as_chunk(generation.message),
                generation_info=generation.generation_info,
            )

    def _attempt_failed(
        self, index: int, exc: Exception, errors: list[tuple[str, Exception]]
    ) -> None:
        """Record a failed attempt and tell ``on_fallback``.

        Every exception moves on to the next model.  Only failures that say
        something about the provider's health count toward its circuit
        breaker; a request the provider rejected as invalid
        (:func:`is_request_error`) does not, so one oversized or malformed
        request cannot take a healthy model out of rotation for everyone.
        """
        model_id = self._model_ids[index]
        errors.append((model_id, exc))
        if is_request_error(exc):
            logger.warning(
                "FallbackChain: %s rejected the request (%s: %s), trying next "
                "(not counted toward its circuit breaker)",
                model_id,
                type(exc).__name__,
                str(exc)[:100],
            )
        else:
            self._circuits[index].record_failure()
            logger.warning(
                "FallbackChain: %s failed (%s: %s), trying next",
                model_id,
                type(exc).__name__,
                str(exc)[:100],
            )
        if self.on_fallback and index + 1 < len(self._resolved):
            try:
                self.on_fallback(model_id, self._model_ids[index + 1], exc)
            except Exception:
                pass

    def _all_failed(self, errors: list[tuple[str, Exception]]) -> RuntimeError:
        """The error raised when no model could answer, chained to the last failure."""
        detail = "\n".join(f"  {mid}: {type(err).__name__}: {err}" for mid, err in errors)
        skipped = [self._model_ids[i] for i, c in enumerate(self._circuits) if c.state == "open"]
        if skipped:
            detail += f"\n  Skipped (circuit open): {', '.join(skipped)}"
        error = RuntimeError(f"All {len(self._resolved)} models in FallbackChain failed.\n{detail}")
        if errors:
            error.__cause__ = errors[-1][1]
        return error

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


# HTTP statuses with which a provider rejects the request itself (malformed,
# too large, unprocessable) rather than failing to serve it.
_REQUEST_ERROR_STATUSES = frozenset({400, 413, 422})


def is_request_error(exc: BaseException) -> bool:
    """Whether ``exc`` is a provider rejecting the request as invalid.

    True for errors carrying HTTP status 400, 413 or 422 (on the exception's
    ``status_code``, as the OpenAI and Anthropic SDKs set it, or on its
    ``response``).  Such errors still fall back to the next model -- it may
    accept the request, for example with a larger context window -- but
    they do not count toward the model's circuit breaker.  Timeouts,
    connection errors, rate limits (429), server errors (5xx) and
    authentication errors (401/403) do.
    """
    status = getattr(exc, "status_code", None)
    if not isinstance(status, int):
        status = getattr(getattr(exc, "response", None), "status_code", None)
    return isinstance(status, int) and status in _REQUEST_ERROR_STATUSES


def _can_stream(model: BaseChatModel) -> bool:
    """Whether the model implements streaming itself."""
    return (
        type(model)._astream is not BaseChatModel._astream
        or type(model)._stream is not BaseChatModel._stream
    )


def _as_chunk(message: Any) -> BaseMessageChunk:
    """An answer message as a streaming chunk (tool calls included)."""
    if isinstance(message, BaseMessageChunk):
        return message
    tool_call_chunks = [
        tool_call_chunk(
            name=call.get("name"),
            args=json.dumps(call.get("args") or {}),
            id=call.get("id"),
            index=n,
        )
        for n, call in enumerate(getattr(message, "tool_calls", None) or [])
    ]
    return AIMessageChunk(
        content=message.content,
        additional_kwargs=dict(message.additional_kwargs),
        response_metadata=dict(message.response_metadata),
        id=message.id,
        usage_metadata=getattr(message, "usage_metadata", None),
        tool_call_chunks=tool_call_chunks,
    )


async def _aclose_quietly(stream: AsyncIterator[Any]) -> None:
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        try:
            await aclose()
        except Exception:
            pass


async def _ainvoke_as_result(
    runnable: Runnable[Any, Any],
    messages: list[BaseMessage],
    stop: list[str] | None,
    kwargs: dict[str, Any],
) -> ChatResult:
    """Call a bound model that is not a plain kwargs binding."""
    if stop is not None:
        kwargs = {**kwargs, "stop": stop}
    message = await runnable.ainvoke(messages, **kwargs)
    return ChatResult(generations=[ChatGeneration(message=message)])
