"""LangChain callback handler that bridges LLM events into the ObservabilityCollector.

This is the core of Promptise's plug-and-play observability.  When attached to a
LangChain agent's ``config["callbacks"]``, it captures **every** LLM turn, tool
call, token count, latency, retry, and error — automatically.

The handler is designed to be instantiated once per agent and reused across
multiple ``ainvoke()`` calls.  It accumulates session-wide totals (tokens)
and records per-call detail to the collector timeline.

Usage::

    from promptise.observability import ObservabilityCollector
    from promptise.callback_handler import PromptiseCallbackHandler

    collector = ObservabilityCollector("my-session")
    handler = PromptiseCallbackHandler(collector, agent_id="my-agent")

    result = await agent.ainvoke(input, config={"callbacks": [handler]})

    # Token totals are now available
    print(handler.total_tokens)
"""

from __future__ import annotations

import time
import traceback
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import ToolMessage
from langchain_core.outputs import LLMResult

from .observability_config import ObserveLevel


class PromptiseCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler → ObservabilityCollector bridge.

    Captures every LLM turn, tool call, token count, latency,
    retry, and error.  Designed to be the *only* integration point needed
    between LangChain's event system and Promptise's observability.

    What is recorded depends on ``level`` (see
    :class:`~promptise.observability_config.ObserveLevel`): ``OFF`` records
    nothing, ``BASIC`` skips the per-turn ``llm.start`` / ``llm.end``
    events (token usage is still added to the current agent run),
    ``STANDARD`` records every turn, and ``FULL`` also records prompt and
    response text unless ``record_prompts=False``.

    Args:
        collector: The :class:`~promptise.observability.ObservabilityCollector`.
        agent_id: Agent identifier stamped on every event.
        record_prompts: Record prompt and response text.  ``None`` follows
            ``level`` (on at ``FULL`` only); ``True`` / ``False`` force it.
        level: Detail level.
        record_tool_io: Record tool arguments and result previews.  When
            ``False``, only their lengths are recorded.
    """

    # Tell LangChain we handle *all* event types.
    raise_error = False
    # We are synchronous (BaseCallbackHandler, not AsyncCallbackHandler).
    # LangChain will call us from inside the async chain on the event loop
    # thread, which is fine because ObservabilityCollector is thread-safe.

    def __init__(
        self,
        collector: Any,  # ObservabilityCollector — typed as Any to avoid circular import
        agent_id: str | None = None,
        *,
        record_prompts: bool | None = None,
        level: ObserveLevel = ObserveLevel.STANDARD,
        record_tool_io: bool = True,
    ) -> None:
        super().__init__()
        self.collector = collector
        self.agent_id = agent_id
        self.level = level
        #: Whether prompt / response text is recorded (resolved from
        #: ``record_prompts`` and ``level``).
        self.record_prompts: bool = (
            level == ObserveLevel.FULL if record_prompts is None else record_prompts
        )
        self.record_tool_io = record_tool_io

        # --- Timing bookkeeping (run_id → start epoch) ---
        self._llm_starts: dict[UUID, float] = {}
        self._tool_starts: dict[UUID, float] = {}
        self._tool_names: dict[UUID, str] = {}
        self._chain_starts: dict[UUID, float] = {}

        # --- Cumulative session accounting ---
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0

        self.llm_call_count: int = 0
        self.tool_call_count: int = 0
        self.error_count: int = 0
        self.retry_count: int = 0

        # --- Streaming accumulation ---
        self._streaming_tokens: dict[UUID, list[str]] = {}

        # --- Run hierarchy ---
        self._run_parents: dict[UUID, UUID | None] = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate(self, value: str, max_len: int = 2000) -> str:
        """Truncate long strings for safe metadata inclusion."""
        if len(value) <= max_len:
            return value
        return value[:max_len] + f"... [truncated, {len(value)} total chars]"

    def _record(self, event_type_value: str, **kwargs: Any) -> Any:
        """Record an event, lazily importing the enum to avoid circular deps."""
        if self.level == ObserveLevel.OFF:
            return None
        from .observability import TimelineEventType

        # Map string to enum member
        try:
            evt = TimelineEventType(event_type_value)
        except ValueError:
            # Fallback: use LLM_TURN for unknown types
            evt = TimelineEventType.LLM_TURN

        return self.collector.record(evt, agent_id=self.agent_id, **kwargs)

    @property
    def _records_llm_turns(self) -> bool:
        """``llm.start`` / ``llm.end`` are recorded at STANDARD and FULL."""
        return self.level in (ObserveLevel.STANDARD, ObserveLevel.FULL)

    @staticmethod
    def _current_run() -> Any:
        """The agent run being observed in this context, if any."""
        from .observability import get_current_run

        return get_current_run()

    # ------------------------------------------------------------------
    # LLM events
    # ------------------------------------------------------------------

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._llm_starts[run_id] = time.time()
        self._run_parents[run_id] = parent_run_id
        self.llm_call_count += 1
        run = self._current_run()
        if run is not None:
            run.llm_calls += 1

        if not self._records_llm_turns:
            return  # BASIC records no per-turn LLM events

        metadata: dict[str, Any] = {"run_id": str(run_id)}

        # Extract model name from serialized info
        model_name = (
            serialized.get("kwargs", {}).get("model_name") or serialized.get("id", [""])[-1]
        )
        if model_name:
            metadata["model"] = model_name

        if self.record_prompts and prompts:
            metadata["prompt_preview"] = self._truncate(prompts[0])

        self._record("llm.start", details="LLM call started", metadata=metadata)

    def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        """Capture streaming tokens (FULL level only).

        ``token`` is typed to match LangChain-core's ``LLMManagerMixin`` base,
        which widened it from ``str`` to ``str | list[str | dict]`` in the 1.x
        line. The wider annotation satisfies the override contract on both the
        0.3.x and 1.x major versions.
        """
        if self.level != ObserveLevel.FULL:
            return

        tokens = self._streaming_tokens.setdefault(run_id, [])
        tokens.append(token if isinstance(token, str) else str(token))

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        start = self._llm_starts.pop(run_id, None)
        duration = time.time() - start if start else None

        metadata: dict[str, Any] = {"run_id": str(run_id)}
        if duration is not None:
            metadata["latency_ms"] = round(duration * 1000, 1)

        prompt_tok = 0
        completion_tok = 0
        total_tok = 0
        model_name = ""

        # --- Extract token usage from LLMResult.llm_output ---
        if response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            if usage:
                prompt_tok = usage.get("prompt_tokens", 0)
                completion_tok = usage.get("completion_tokens", 0)
                total_tok = usage.get("total_tokens", 0)
            model_name = response.llm_output.get("model_name", "")

        # --- Also check per-generation usage_metadata (LangChain >=0.3) ---
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    usage_meta = getattr(msg, "usage_metadata", None)
                    if usage_meta and isinstance(usage_meta, dict):
                        prompt_tok = prompt_tok or usage_meta.get("input_tokens", 0)
                        completion_tok = completion_tok or usage_meta.get("output_tokens", 0)
                        total_tok = total_tok or (prompt_tok + completion_tok)

        if not total_tok and (prompt_tok or completion_tok):
            total_tok = prompt_tok + completion_tok

        # Accumulate session totals
        self.total_prompt_tokens += prompt_tok
        self.total_completion_tokens += completion_tok
        self.total_tokens += total_tok
        run = self._current_run()
        if run is not None:
            run.prompt_tokens += prompt_tok
            run.completion_tokens += completion_tok

        if not self._records_llm_turns:
            self._streaming_tokens.pop(run_id, None)
            return

        metadata["prompt_tokens"] = prompt_tok
        metadata["completion_tokens"] = completion_tok
        metadata["total_tokens"] = total_tok
        if model_name:
            metadata["model"] = model_name

        # --- Streamed tokens (FULL only accumulates them) ---
        streamed = self._streaming_tokens.pop(run_id, None)

        # --- Response preview (when recording prompts) ---
        if self.record_prompts:
            for gen_list in response.generations:
                for gen in gen_list:
                    text = getattr(gen, "text", "")
                    if text:
                        metadata["response_preview"] = self._truncate(text)
                        break
            if "response_preview" not in metadata and streamed:
                metadata["response_preview"] = self._truncate("".join(streamed))

        # --- Tool calls in the response ---
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    tool_calls = getattr(msg, "tool_calls", None) or []
                    if tool_calls:
                        metadata["tool_calls"] = [
                            tc.get("name", "unknown")
                            if isinstance(tc, dict)
                            else getattr(tc, "name", "unknown")
                            for tc in tool_calls
                        ]

        # --- Streaming token summary ---
        if streamed:
            metadata["streamed_token_count"] = len(streamed)

        self._record(
            "llm.end",
            details=f"LLM call completed ({total_tok} tokens, {metadata.get('latency_ms', '?')}ms)",
            duration=duration,
            metadata=metadata,
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        start = self._llm_starts.pop(run_id, None)
        duration = time.time() - start if start else None
        self.error_count += 1
        self._streaming_tokens.pop(run_id, None)
        run = self._current_run()
        if run is not None:
            run.errors += 1

        self._record(
            "llm.error",
            details=f"LLM error: {type(error).__name__}: {str(error)[:200]}",
            duration=duration,
            metadata={
                "run_id": str(run_id),
                "error": str(error)[:500],
                "error_type": type(error).__name__,
                "traceback": self._truncate(
                    "".join(traceback.format_exception(type(error), error, error.__traceback__))
                ),
            },
        )

    # ------------------------------------------------------------------
    # Tool events
    # ------------------------------------------------------------------

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._tool_starts[run_id] = time.time()
        self._run_parents[run_id] = parent_run_id
        self.tool_call_count += 1
        run = self._current_run()
        if run is not None:
            run.tool_calls += 1

        tool_name = serialized.get("name") or kwargs.get("name") or "unknown"
        self._tool_names[run_id] = tool_name

        metadata: dict[str, Any] = {"tool_name": tool_name, "run_id": str(run_id)}
        if self.record_tool_io:
            metadata["arguments"] = self._truncate(input_str)
        else:
            metadata["arguments_length"] = len(input_str)
        self._record("tool.call", details=f"Calling tool: {tool_name}", metadata=metadata)

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        start = self._tool_starts.pop(run_id, None)
        duration = time.time() - start if start else None
        tool_name = self._tool_names.pop(run_id, None) or kwargs.get("name") or "unknown"

        # A tool invoked as a tool call returns a ToolMessage; record its
        # content, and its status when the tool reported an error.  A tool
        # that handles its own errors (``handle_tool_error``) returns
        # status="error" instead of raising: that is a failed call, so it
        # counts as an error here and in ``get_stats()``.
        status = None
        if isinstance(output, ToolMessage):
            status = output.status
            output = output.content
        failed = status == "error"
        if failed:
            self.error_count += 1
            run = self._current_run()
            if run is not None:
                run.errors += 1

        result_text = str(output)
        metadata: dict[str, Any] = {"run_id": str(run_id)}
        if self.record_tool_io:
            metadata["result_preview"] = self._truncate(result_text)
        else:
            metadata["result_length"] = len(result_text)
        if failed:
            metadata["status"] = "error"
        if duration is not None:
            metadata["latency_ms"] = round(duration * 1000, 1)
        if tool_name != "unknown":
            metadata["tool_name"] = tool_name

        self._record(
            "tool.result",
            details=f"Tool reported an error: {tool_name}"
            if failed
            else f"Tool completed: {tool_name}",
            duration=duration,
            metadata=metadata,
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        start = self._tool_starts.pop(run_id, None)
        duration = time.time() - start if start else None
        self.error_count += 1
        run = self._current_run()
        if run is not None:
            run.errors += 1

        tool_name = self._tool_names.pop(run_id, None) or kwargs.get("name") or "unknown"
        metadata: dict[str, Any] = {
            "run_id": str(run_id),
            "tool_name": tool_name,
            "error_type": type(error).__name__,
        }
        if duration is not None:
            metadata["latency_ms"] = round(duration * 1000, 1)
        # The error message of a failed tool call usually echoes its input,
        # so it follows the same switch as tool arguments and results.
        if self.record_tool_io:
            metadata["error"] = str(error)[:500]
            if error.__traceback__ is not None:
                metadata["traceback"] = self._truncate(
                    "".join(traceback.format_exception(type(error), error, error.__traceback__))
                )
        self._record(
            "tool.error",
            details=f"Tool error in {tool_name}: {type(error).__name__}",
            duration=duration,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Chain (agent-level) events
    # ------------------------------------------------------------------

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self._chain_starts[run_id] = time.time()
        self._run_parents[run_id] = parent_run_id

        # Only record AGENT_INPUT for the top-level chain (not sub-chains),
        # and only when the handler is used on its own: a PromptiseAgent
        # records agent.input / agent.output itself for every invocation.
        if parent_run_id is not None or self._current_run() is not None:
            return

        input_text = ""
        if isinstance(inputs, dict):
            # LangGraph agents use "messages" key
            messages = inputs.get("messages", inputs.get("input", ""))
            input_text = str(messages)
        else:
            input_text = str(inputs)

        metadata: dict[str, Any] = {"run_id": str(run_id)}
        if self.record_prompts:
            metadata["input_preview"] = self._truncate(input_text)
        else:
            metadata["input_length"] = len(input_text)

        self._record(
            "agent.input",
            details="Agent invocation started",
            metadata=metadata,
        )

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        start = self._chain_starts.pop(run_id, None)
        duration = time.time() - start if start else None

        # Only record AGENT_OUTPUT for the top-level chain
        if parent_run_id is not None or self._current_run() is not None:
            return

        metadata: dict[str, Any] = {
            "run_id": str(run_id),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_call_count": self.llm_call_count,
            "tool_call_count": self.tool_call_count,
            "error_count": self.error_count,
        }
        if duration is not None:
            metadata["total_duration_ms"] = round(duration * 1000, 1)

        # Count messages in output
        if isinstance(outputs, dict):
            messages = outputs.get("messages", [])
            if isinstance(messages, list):
                metadata["message_count"] = len(messages)

        self._record(
            "agent.output",
            details=f"Agent completed ({self.total_tokens} tokens)",
            duration=duration,
            metadata=metadata,
        )

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        start = self._chain_starts.pop(run_id, None)
        duration = time.time() - start if start else None

        if parent_run_id is not None or self._current_run() is not None:
            return  # Only record top-level chain errors of standalone use
        self.error_count += 1

        self._record(
            "agent.error",
            details=f"Agent error: {type(error).__name__}: {str(error)[:200]}",
            duration=duration,
            metadata={
                "run_id": str(run_id),
                "error": str(error)[:500],
                "error_type": type(error).__name__,
                "traceback": self._truncate(
                    "".join(traceback.format_exception(type(error), error, error.__traceback__))
                ),
            },
        )

    # ------------------------------------------------------------------
    # Retry events
    # ------------------------------------------------------------------

    def on_retry(
        self,
        retry_state: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.retry_count += 1

        metadata: dict[str, Any] = {"run_id": str(run_id)}

        # Extract retry info if retry_state has expected attributes
        attempt = getattr(retry_state, "attempt_number", self.retry_count)
        metadata["attempt"] = attempt

        last_error = getattr(retry_state, "outcome", None)
        if last_error is not None:
            exc = getattr(last_error, "exception", lambda: None)()
            if exc is not None:
                metadata["error"] = str(exc)[:500]
                metadata["error_type"] = type(exc).__name__

        self._record(
            "llm.retry",
            details=f"Retry attempt {attempt}",
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Return a summary of all tracked metrics."""
        return {
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_call_count": self.llm_call_count,
            "tool_call_count": self.tool_call_count,
            "error_count": self.error_count,
            "retry_count": self.retry_count,
        }
