"""Agent builders — Promptise MCP Client for tool discovery and invocation."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import time
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool

from promptise.engine import PromptGraph, PromptGraphEngine

from .config import HTTPServerSpec, ServerSpec
from .cross_agent import CrossAgent, make_cross_agent_tools
from .identity import AgentIdentity, IdentityError
from .models import Model
from .prompt import DEFAULT_SYSTEM_PROMPT
from .prompts.flows import FlowSessions


@dataclass
class CallerContext:
    """Identity and metadata for the caller of an agent invocation.

    Pass this to ``ainvoke()`` or ``chat()`` to carry per-request
    identity through the entire invocation — guardrails, conversation
    ownership, observability, and (future) MCP token forwarding.

    Attributes:
        user_id: Unique user identifier.  Used for conversation session
            ownership and observability correlation.
        tenant_id: Organisation / tenant the caller belongs to.  When set,
            it becomes part of :attr:`isolation_key`, so every per-user
            isolation surface (semantic cache, memory search, conversation
            ownership) is scoped per tenant — two tenants with the same
            ``user_id`` can never see each other's data.
        bearer_token: JWT or OAuth token for the caller.  Currently
            available for guardrails and logging; MCP token forwarding
            is a planned enhancement.
        roles: Caller's roles (e.g. ``{"admin", "analyst"}``).
            Available for custom guardrail rules and logging.
        scopes: OAuth scopes (e.g. ``{"read", "write"}``).
        metadata: Arbitrary key-value metadata (IP, user-agent, etc.).

    Example::

        caller = CallerContext(
            user_id="user-42",
            tenant_id="acme",
            bearer_token="eyJhbGciOiJIUzI1NiIs...",
            roles={"analyst"},
        )
        result = await agent.ainvoke(input, caller=caller)
        reply = await agent.chat("Hello", session_id=sid, caller=caller)
    """

    user_id: str | None = None
    tenant_id: str | None = None
    bearer_token: str | None = field(default=None, repr=False)
    roles: set[str] = field(default_factory=set)
    scopes: set[str] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def isolation_key(self) -> str | None:
        """The identity key used for every per-user isolation surface.

        ``"{tenant_id}::{user_id}"`` when a tenant is set, otherwise the
        plain ``user_id``.  This single derivation is what the semantic
        cache, memory scoping, and conversation ownership all key on, so
        tenant isolation is a structural invariant rather than a
        convention.  ``None`` when ``user_id`` is unset.
        """
        if self.user_id is None:
            return None
        if self.tenant_id:
            return f"{self.tenant_id}::{self.user_id}"
        return self.user_id

    def __post_init__(self) -> None:
        # ``isolation_key`` joins tenant and user as ``f"{tenant}::{user}"``.
        # Rejecting only the ``"::"`` *substring* in each component is NOT
        # enough — the join can still *synthesize* a ``::`` at the boundary
        # when the tenant ends with ``:`` and the user starts with ``:``
        # (``"a" + "::" + ":b"`` == ``"a:" + "::" + "b"`` == ``"a:::b"``), so
        # two distinct pairs would collide. To make the map injective we
        # require the tenant to be **colon-free** (the first ``:`` then
        # unambiguously starts the separator) and forbid ``"::"`` in the user
        # (so the tenanted keyspace — always containing ``::`` — stays
        # disjoint from the untenanted one, a raw user_id). This is what makes
        # cross-tenant isolation structural rather than advisory. Single
        # colons in ``user_id`` (SSO ids like ``google:12345``) remain fine.
        if self.tenant_id is not None:
            if not self.tenant_id.strip():
                raise ValueError("tenant_id must be non-empty when provided")
            if ":" in self.tenant_id:
                raise ValueError(
                    f"tenant_id must not contain ':' (got {self.tenant_id!r}) — "
                    "it delimits the tenant isolation key"
                )
        if self.user_id is not None and "::" in self.user_id:
            raise ValueError(
                f"user_id must not contain '::' (got {self.user_id!r}) — "
                "it is the tenant isolation-key separator"
            )


# Thread/async-safe per-request caller context.
# Each concurrent ainvoke() gets its own copy via contextvars.
_caller_ctx_var: contextvars.ContextVar[CallerContext | None] = contextvars.ContextVar(
    "promptise_caller", default=None
)

# Owner of the session that chat() is serving, when it was named with
# ``user_id=`` rather than a CallerContext. Scopes per-session flow state.
_session_owner_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "promptise_session_owner", default=None
)


def get_current_caller() -> CallerContext | None:
    """Return the :class:`CallerContext` for the current invocation.

    Safe to call from guardrails, context providers, observability
    handlers, or any code running inside an ``ainvoke()`` / ``chat()``
    call.  Returns ``None`` outside of an invocation.
    """
    return _caller_ctx_var.get()


from .tools import MCPClientError

# Model can be a provider string (handled by LangChain), a chat model instance, or a Runnable.
ModelLike: TypeAlias = str | Model | BaseChatModel | Runnable[Any, Any]
"""Type alias for model parameter: string, BaseChatModel, Runnable, or FallbackChain."""

logger = logging.getLogger("promptise.agent")


# ---------------------------------------------------------------------------
# PromptiseAgent — the unified agent with opt-in capabilities
# ---------------------------------------------------------------------------


class PromptiseAgent:
    """The unified Promptise agent.

    Always returned by :func:`build_agent`.  Observability and memory
    are opt-in capabilities activated by constructor parameters — disabled
    features no-op or return sensible defaults, so callers never need to
    check what type they got back.

    .. code-block:: python

        # Simple — no observe, no memory
        agent = await build_agent(servers=..., model="openai:gpt-5-mini")
        result = await agent.ainvoke({"messages": [...]})
        await agent.shutdown()

        # With observability
        agent = await build_agent(..., observe=True)
        result = await agent.ainvoke({"messages": [...]})
        stats = agent.get_stats()
        agent.generate_report("report.html")
        await agent.shutdown()

        # With memory
        agent = await build_agent(..., memory=InMemoryProvider())
        result = await agent.ainvoke({"messages": [...]})  # auto-injects context
        await agent.shutdown()

    Attributes:
        collector: The :class:`ObservabilityCollector` holding recorded events,
            or ``None`` when observability is not enabled.
        provider: The :class:`~promptise.memory.MemoryProvider` instance,
            or ``None`` when memory is not enabled.
    """

    def __init__(
        self,
        inner: Runnable[Any, Any],
        *,
        # Observability (all optional — None = disabled)
        handler: Any | None = None,
        collector: Any | None = None,
        observe_config: Any | None = None,
        transporters: list[Any] | None = None,
        # Memory (all optional — None = disabled)
        memory_provider: Any | None = None,
        memory_max: int = 5,
        memory_min_score: float = 0.0,
        memory_timeout: float = 5.0,
        memory_auto_store: bool = False,
        # MCP lifecycle
        mcp_multi: Any | None = None,
        # Model identity (for prompt context)
        model_name: str | None = None,
        # Prompt framework integration
        prompt_config: Any | None = None,
        # Conversation persistence (optional)
        conversation_store: Any | None = None,
        conversation_max_messages: int = 0,
        # Tool optimization (semantic selection)
        tool_index: Any | None = None,
        all_tools: list[Any] | None = None,
        graph_builder_fn: Any | None = None,
        tools: list[BaseTool] | None = None,
        # Security guardrails
        guardrails: Any | None = None,
        # Semantic cache
        cache: Any | None = None,
        # Human-in-the-loop approval
        approval: Any | None = None,
        # Event notifications
        event_notifier: Any | None = None,
    ) -> None:
        self._inner = inner

        # Observability
        self._handler = handler
        self.collector = collector
        self._observe_config = observe_config
        self._transporters: list[Any] = transporters or []

        # Memory
        self.provider = memory_provider
        self._memory_max = memory_max
        self._memory_min_score = memory_min_score
        self._memory_timeout = memory_timeout
        self._memory_auto_store = memory_auto_store

        # MCP lifecycle
        self._mcp_multi = mcp_multi

        # Model identity — stored explicitly so prompt context can
        # resolve the model name without digging into the LangGraph graph.
        self.model_name: str | None = model_name

        # Prompt framework — Prompt or PromptSuite for dynamic context
        self._prompt_config = prompt_config

        # Conversation persistence
        self._conversation_store = conversation_store
        self._conversation_max_messages = conversation_max_messages

        # Conversation flow (Layer 2): one flow per session or caller
        self._flows: FlowSessions | None = None

        # Tool optimization — semantic selection
        self._tool_index = tool_index
        self._all_tools = all_tools or []
        self._graph_builder_fn = graph_builder_fn
        # Every tool the agent can call: MCP-discovered, extra_tools, sandbox,
        # cross-agent and meta tools alike.
        self._tools: list[BaseTool] = list(tools or [])

        # Security guardrails (PromptiseSecurityScanner or Guard protocol)
        self._guardrails = guardrails

        # Semantic cache
        self._cache = cache

        # Human-in-the-loop approval
        self._approval = approval

        # Event notifications
        self._event_notifier = event_notifier

        # Invocation timeout (0 = no limit)
        self._max_invocation_time: float = 0

        # Adaptive strategy (learning from failure)
        self._strategy_manager: Any | None = None

        # Context Engine (opt-in unified assembly)
        self._context_engine: Any | None = None

        # Optional state attached post-construction by build_agent().
        self._raw_instructions: str = ""
        self._sandbox_session: Any | None = None
        self._sandbox_manager: Any | None = None

        # Federated workload identity (optional; set by build_agent()).
        # Exposed publicly so MCP tools and downstream calls can mint
        # authenticated headers via ``agent.identity.get_auth_header()``.
        self.identity: AgentIdentity | None = None

        # Resolved attribution id (the agent's identifier, computed once by
        # build_agent). Used to attribute event notifications to the agent.
        self._actor_id: str | None = None

    def _actor(self) -> str | None:
        """Return the id to attribute the agent's events to.

        When an identity is attached, this is the agent's resolved
        identifier (``agent_id`` or the IdP subject); otherwise it is the
        model name, so agents without an identity are unaffected.
        """
        if self.identity is not None:
            return self._actor_id or self.model_name
        return self.model_name

    # -----------------------------------------------------------------
    # Core invocation methods
    # -----------------------------------------------------------------

    async def ainvoke(
        self,
        input: Any,
        config: dict[str, Any] | None = None,
        *,
        caller: CallerContext | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the agent asynchronously.

        Args:
            input: LangGraph-style input dict with ``messages``.
            config: LangGraph config dict (callbacks, etc.).
            session_id: Optional conversation ID.  With a conversation
                ``flow``, each session (per caller) keeps its own flow
                state.  :meth:`chat` passes its ``session_id`` here.
            caller: Optional :class:`CallerContext` with per-request
                identity.  When provided, ``user_id`` is used for
                conversation ownership and the full context is
                available to guardrails and observability.  When omitted,
                the ambient ``CallerContext`` (if one is already active in
                this task) is inherited, so cross-agent delegation keeps
                the original human principal: a peer invoked inside
                another agent's request scopes its cache, memory,
                guardrails, and conversations to the original user.

        When memory is enabled, relevant context is searched and injected
        as a ``SystemMessage`` before the inner graph runs.  When
        observability is enabled, a callback handler is attached to
        capture every LLM turn, tool call, and token count.
        """
        # Store caller in async-safe contextvar (not on instance — avoids
        # race conditions when multiple concurrent ainvoke() share one agent).
        # Inherit the ambient CallerContext when no explicit caller is given:
        # without this, a peer agent invoked mid-delegation would overwrite
        # the active context with None and the original user's identity would
        # be dropped at every cross-agent hop.
        if caller is None:
            caller = _caller_ctx_var.get()
        _ctx_token = _caller_ctx_var.set(caller)
        try:
            # Enforce max_invocation_time if configured
            timeout = getattr(self, "_max_invocation_time", 0)
            if timeout and timeout > 0:
                try:
                    return await asyncio.wait_for(
                        self._ainvoke_inner(input, config, session_id=session_id, **kwargs),
                        timeout=timeout,
                    )
                except asyncio.TimeoutError:
                    if self._event_notifier is not None:
                        from .events import emit_event

                        emit_event(
                            self._event_notifier,
                            "invocation.timeout",
                            "error",
                            {"timeout_seconds": timeout},
                            agent_id=self._actor(),
                        )
                    raise TimeoutError(f"Agent invocation exceeded {timeout}s timeout")
            else:
                return await self._ainvoke_inner(input, config, session_id=session_id, **kwargs)
        except Exception as exc:
            # Emit invocation.error event on any unhandled exception
            if self._event_notifier is not None:
                from .events import emit_event

                emit_event(
                    self._event_notifier,
                    "invocation.error",
                    "error",
                    {"error": str(exc)[:200], "error_type": type(exc).__name__},
                    agent_id=self._actor(),
                )
            raise
        finally:
            _caller_ctx_var.reset(_ctx_token)

    async def _ainvoke_inner(
        self,
        input: Any,
        config: dict[str, Any] | None = None,
        *,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Inner implementation — runs with CallerContext in contextvar."""
        # Defensive copy — prevents mutation of caller's input dict
        # when multiple concurrent ainvoke() calls share the same input
        if isinstance(input, dict):
            input = {**input}
            if "messages" in input:
                input["messages"] = list(input["messages"])

        _cache_query: str = ""
        _ctx_fp: str = ""
        _inst_hash: str = ""
        _start_time = time.monotonic()

        # Emit invocation.start event
        if self._event_notifier is not None:
            from .events import emit_event

            emit_event(
                self._event_notifier,
                "invocation.start",
                "info",
                {"model": self.model_name},
                agent_id=self._actor(),
            )

        # Step 0: Guardrails — scan input BEFORE anything else
        if self._guardrails is not None:
            from .memory import _extract_user_text as _ext

            raw_text = _ext(input)
            if raw_text:
                try:
                    await self._guardrails.check_input(raw_text)
                except Exception as guard_exc:
                    if self._event_notifier is not None:
                        from .events import emit_event

                        emit_event(
                            self._event_notifier,
                            "guardrail.blocked",
                            "warning",
                            {"direction": "input", "error": type(guard_exc).__name__},
                        )
                    raise  # Re-raise — don't swallow the violation

        # ── Context Engine path (opt-in) ──
        # When a ContextEngine is configured, it replaces the ad-hoc
        # injection steps (memory, strategy, prompt blocks, flow).
        # The engine populates layers, assembles with token budgeting,
        # and the result replaces the input messages.
        _engine_active = self._context_engine is not None

        # Extract user text (always — needed for cache, strategies, tool optimization)
        from .memory import _extract_user_text

        user_text = _extract_user_text(input) if input else ""

        # Step 0.5: Guards of a Prompt used as instructions check the user message
        if self._prompt_config is not None and user_text:
            checked_text = await self._check_prompt_input(user_text)
            if checked_text != user_text:
                input = _replace_last_user_text(input, checked_text)
                user_text = checked_text

        # Step 1: Memory — search (always, for cache fingerprint) and inject (legacy only)
        _memory_results: list[Any] = []
        if self.provider is not None:
            from .memory import _format_memory_context, _inject_memory_into_messages

            _memory_results = await self._search_memory(user_text)
            if _memory_results and not _engine_active:
                context = _format_memory_context(_memory_results)
                input = _inject_memory_into_messages(input, context)

        # Step 1.1: Adaptive strategy — inject learned strategies (legacy path only)
        if self._strategy_manager is not None and user_text and not _engine_active:
            try:
                strategies = await self._strategy_manager.get_relevant_strategies(user_text)
                if strategies:
                    block = self._strategy_manager.format_strategy_block(strategies)
                    if block and isinstance(input, dict) and "messages" in input:
                        # Fix: create a copy to avoid mutating shared input
                        from langchain_core.messages import SystemMessage as _SM

                        messages = list(input["messages"])
                        # Insert after all leading system messages (same algorithm as memory)
                        insert_idx = 0
                        for i, msg in enumerate(messages):
                            is_sys = isinstance(msg, _SM) or (
                                isinstance(msg, dict) and msg.get("role") == "system"
                            )
                            if is_sys:
                                insert_idx = i + 1
                            else:
                                break
                        messages.insert(insert_idx, _SM(content=block))
                        input = {**input, "messages": messages}
            except Exception:
                logger.debug("Strategy injection failed, continuing", exc_info=True)

        # ── Context Engine assembly (when active, replaces ad-hoc injection) ──
        if _engine_active:
            assert self._context_engine is not None  # narrowed by _engine_active
            engine = self._context_engine
            engine.clear_all()

            # Populate layers from collected data
            engine.set_content("identity", getattr(self, "_raw_instructions", "") or "")
            if user_text:
                engine.set_content("user_message", user_text)

            # Preserve conversation history from original input
            if isinstance(input, dict):
                history_msgs = input.get("messages", [])
                conv_lines = []
                for msg in history_msgs:
                    role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", "")
                    content = (
                        msg.get("content", "")
                        if isinstance(msg, dict)
                        else getattr(msg, "content", "")
                    )
                    if role in ("user", "human") and content and content != user_text:
                        conv_lines.append(f"User: {content}")
                    elif role in ("assistant", "ai") and content:
                        conv_lines.append(f"Assistant: {content}")
                if conv_lines:
                    engine.set_content("conversation", "\n".join(conv_lines))

            if _memory_results:
                from .memory import _format_memory_context

                engine.set_content("memory", _format_memory_context(_memory_results))
            if self._strategy_manager is not None and user_text:
                try:
                    strategies = await self._strategy_manager.get_relevant_strategies(user_text)
                    if strategies:
                        engine.set_content(
                            "strategies", self._strategy_manager.format_strategy_block(strategies)
                        )
                except Exception:
                    pass

            # Assemble with token budgeting
            assembled = engine.assemble()
            if assembled:
                input = {"messages": assembled}

        # Step 1.4: Conversation flow — evolve system prompt (legacy path only).
        # Before the cache check, so a cache hit still advances the flow and a
        # reply cached under another phase's prompt is not served.
        _flow_text = ""
        if self._flows is not None and not _engine_active:
            input, _flow_text = await self._inject_flow_context(input, session_id)

        # Step 1.5: Cache check — AFTER memory so fingerprint includes memory content
        if self._cache is not None:
            try:
                from .cache import compute_context_fingerprint, compute_instruction_hash

                if not _cache_query:
                    from .memory import _extract_user_text as _ext_cache

                    _cache_query = _ext_cache(input)

                if _cache_query:
                    _instructions = getattr(self, "_raw_instructions", None)
                    if _flow_text:
                        _instructions = f"{_flow_text}\n\n{_instructions or ''}"
                    _inst_hash = compute_instruction_hash(_instructions)
                    _ctx_fp = compute_context_fingerprint(
                        memory_results=_memory_results,
                        conversation_length=len(input.get("messages", []))
                        if isinstance(input, dict)
                        else 0,
                        instruction_hash=_inst_hash,
                    )
                    cached = await self._cache.check(
                        _cache_query,
                        context_fingerprint=_ctx_fp,
                        caller=get_current_caller(),
                        model_id=self.model_name,
                        instruction_hash=_inst_hash,
                    )
                    if cached is not None:
                        # Record cache hit in observability
                        if self.collector is not None:
                            from .observability import TimelineEventType

                            self.collector.record(
                                TimelineEventType.CACHE_HIT,
                                description=f"Cache hit (similarity match for: {_cache_query[:80]})",
                                metadata={"scope": cached.scope_key, "ttl": cached.ttl},
                            )
                        output = cached.output
                        # Output guardrails ALWAYS run on cached responses
                        if self._guardrails is not None:
                            response_text = _extract_response_text(output)
                            if response_text:
                                checked = await self._guardrails.check_output(response_text)
                                if isinstance(checked, str) and checked != response_text:
                                    output = self._replace_response_text(output, checked)
                        if self._prompt_config is not None:
                            output = await self._check_prompt_output(output)
                        return output
                    else:
                        # Record cache miss
                        if self.collector is not None:
                            from .observability import TimelineEventType

                            self.collector.record(
                                TimelineEventType.CACHE_MISS,
                                description=f"Cache miss for: {_cache_query[:80]}",
                            )
            except Exception:
                # Cache errors never crash the agent — graceful degradation
                logger.warning("Cache check failed, continuing without cache", exc_info=True)

        # Step 1.5: Prompt framework — run context providers (legacy path only)
        if self._prompt_config is not None and not _engine_active:
            input = await self._inject_prompt_context(input, user_text)

        # Step 1.8: Semantic tool selection — rebuild graph with relevant tools
        _invocation_graph = None
        if self._tool_index is not None and self._graph_builder_fn is not None:
            query = user_text
            if not query:
                from .memory import _extract_user_text

                query = _extract_user_text(input)
            if query:
                selected = self._tool_index.select(query)
                # Build a per-invocation graph (don't mutate self._inner —
                # concurrent ainvoke() calls would race on it)
                _invocation_graph = self._graph_builder_fn(selected)
            else:
                _invocation_graph = None

        # Use per-invocation graph if tool selection rebuilt it
        _active_graph = (
            _invocation_graph
            if (_invocation_graph is not None and self._tool_index is not None)
            else self._inner
        )

        # Step 2: Observability — inject callback handler
        if self._handler is not None:
            config = dict(config) if config else {}
            callbacks = list(config.get("callbacks", []))
            callbacks.append(self._handler)
            config["callbacks"] = callbacks

        # Step 3: Delegate to inner graph
        output = await _active_graph.ainvoke(
            input, config=cast("RunnableConfig | None", config), **kwargs
        )

        # Step 3.5: Guardrails — scan output BEFORE returning
        if self._guardrails is not None:
            response_text = _extract_response_text(output)
            if response_text:
                checked = await self._guardrails.check_output(response_text)
                # If check_output returns redacted text, update the output
                if isinstance(checked, str) and checked != response_text:
                    output = self._replace_response_text(output, checked)
                    # Emit guardrail.redacted event
                    if self._event_notifier is not None:
                        from .events import emit_event

                        emit_event(
                            self._event_notifier,
                            "guardrail.redacted",
                            "info",
                            {"direction": "output"},
                        )

        # Step 3.6: Guards of a Prompt used as instructions check the reply
        if self._prompt_config is not None:
            output = await self._check_prompt_output(output)

        # Step 3.75: Store in cache AFTER guardrails (store post-redacted output)
        if self._cache is not None and _cache_query:
            try:
                _resp = _extract_response_text(output)
                if _resp:
                    # Extract tool names used in this invocation for invalidation
                    _tools_used: list[str] = []
                    if isinstance(output, dict):
                        for msg in output.get("messages", []):
                            if hasattr(msg, "tool_calls"):
                                for tc in msg.tool_calls or []:
                                    name = (
                                        tc.get("name")
                                        if isinstance(tc, dict)
                                        else getattr(tc, "name", None)
                                    )
                                    if name:
                                        _tools_used.append(name)

                    await self._cache.store(
                        _cache_query,
                        _resp,
                        output,
                        context_fingerprint=_ctx_fp,
                        caller=get_current_caller(),
                        model_id=self.model_name,
                        instruction_hash=_inst_hash,
                        tools_used=_tools_used,
                    )
                    if self.collector is not None:
                        from .observability import TimelineEventType

                        self.collector.record(
                            TimelineEventType.CACHE_STORE,
                            description=f"Cached response for: {_cache_query[:80]}",
                        )
            except Exception:
                logger.warning("Cache store failed", exc_info=True)

        # Step 4: Memory — auto-store exchange
        if self.provider is not None and user_text:
            await self._maybe_store(user_text, output)

        # Step 4.5: Adaptive strategy — record failures from this invocation
        if self._strategy_manager is not None and self._handler is not None:
            failures = getattr(self._handler, "_current_failures", [])
            if failures:
                from .strategy import FailureLog, classify_failure

                for f in failures:
                    category = classify_failure(f.get("error_type", ""), f.get("error_message", ""))
                    try:
                        await self._strategy_manager.record_failure(
                            FailureLog(
                                tool_name=f.get("tool_name", "unknown"),
                                error_type=f.get("error_type", ""),
                                error_message=f.get("error_message", ""),
                                category=category,
                                args_preview=f.get("args_preview", ""),
                                timestamp=f.get("timestamp", time.time()),
                            )
                        )
                    except Exception:
                        pass
                failures.clear()

        # Emit invocation.complete event
        if self._event_notifier is not None:
            from .events import emit_event

            emit_event(
                self._event_notifier,
                "invocation.complete",
                "info",
                {"duration_ms": round((time.monotonic() - _start_time) * 1000, 1)},
                agent_id=self._actor(),
            )

        return output

    def invoke(
        self,
        input: Any,
        config: dict[str, Any] | None = None,
        *,
        caller: CallerContext | None = None,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the agent synchronously.

        Memory injection requires async I/O.  When a running event loop
        is detected (e.g. inside Jupyter), memory injection is skipped
        for the sync path — use :meth:`ainvoke` instead.
        """
        # Always run through ainvoke to ensure guardrails, cache, memory,
        # and all other features are applied — even from sync callers.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Already inside an async event loop (Jupyter, FastAPI, etc.).
            # Run ainvoke in a separate thread to avoid blocking the loop.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.ainvoke(
                        input, config=config, caller=caller, session_id=session_id, **kwargs
                    ),
                )
                return future.result()
        else:
            return asyncio.run(
                self.ainvoke(input, config=config, caller=caller, session_id=session_id, **kwargs)
            )

    async def astream(
        self,
        input: Any,
        config: dict[str, Any] | None = None,
        *,
        caller: CallerContext | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream the agent asynchronously with memory and observability.

        Args:
            caller: Optional :class:`CallerContext` for per-request identity.
        """
        _ctx_token = _caller_ctx_var.set(caller)
        try:
            async for chunk in self._astream_inner(input, config, **kwargs):
                yield chunk
        finally:
            _caller_ctx_var.reset(_ctx_token)

    async def _astream_inner(
        self,
        input: Any,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Inner stream — runs with CallerContext in contextvar."""
        # Step 1: Memory — search and inject context
        if self.provider is not None:
            from .memory import (
                _extract_user_text,
                _format_memory_context,
                _inject_memory_into_messages,
            )

            user_text = _extract_user_text(input)
            results = await self._search_memory(user_text)
            if results:
                context = _format_memory_context(results)
                input = _inject_memory_into_messages(input, context)

        # Step 2: Observability — inject callback handler
        if self._handler is not None:
            config = dict(config) if config else {}
            callbacks = list(config.get("callbacks", []))
            callbacks.append(self._handler)
            config["callbacks"] = callbacks

        # Step 3: Delegate to inner graph
        async for chunk in self._inner.astream(
            input, config=cast("RunnableConfig | None", config), **kwargs
        ):
            yield chunk

    async def astream_with_tools(
        self,
        input: Any,
        config: dict[str, Any] | None = None,
        *,
        caller: CallerContext | None = None,
        include_arguments: bool = True,
        tool_display_names: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Stream agent execution with tool visibility.

        Yields structured :class:`StreamEvent` objects that show the
        complete agent reasoning process: tool calls, their results,
        LLM tokens, and the final response.

        Args:
            input: Agent input (same format as ``ainvoke``).
            config: LangChain config dict.
            caller: Per-request identity for multi-user.
            include_arguments: Include tool arguments in events.
            tool_display_names: Custom display names for tools.

        Yields:
            :class:`ToolStartEvent`, :class:`ToolEndEvent`,
            :class:`TokenEvent`, :class:`DoneEvent`, or
            :class:`ErrorEvent`.

        Example::

            async for event in agent.astream_with_tools(input, caller=caller):
                if event.type == "tool_start":
                    print(f"🔧 {event.tool_display_name}...")
                elif event.type == "token":
                    print(event.text, end="", flush=True)
        """
        from .streaming import (
            DoneEvent,
            ErrorEvent,
            TokenEvent,
            ToolEndEvent,
            ToolStartEvent,
        )
        from .streaming import (
            redact_tool_args as _redact_args,
        )
        from .streaming import (
            tool_display_name as _display_name,
        )
        from .streaming import (
            tool_summary as _summary,
        )

        _ctx_token = _caller_ctx_var.set(caller)
        _start = time.monotonic()
        _cumulative = ""
        _tool_counter = 0
        _tool_starts: dict[str, tuple[float, int]] = {}
        _all_tool_calls: list[dict[str, Any]] = []

        try:
            # Emit invocation.start event notification
            if self._event_notifier is not None:
                from .events import emit_event

                emit_event(
                    self._event_notifier,
                    "invocation.start",
                    "info",
                    {"model": self.model_name, "streaming": True},
                    agent_id=self._actor(),
                )

            # Step 0: Input guardrails
            if self._guardrails is not None:
                from .memory import _extract_user_text as _ext

                raw_text = _ext(input)
                if raw_text:
                    try:
                        await self._guardrails.check_input(raw_text)
                    except Exception:
                        if self._event_notifier is not None:
                            from .events import emit_event

                            emit_event(
                                self._event_notifier,
                                "guardrail.blocked",
                                "warning",
                                {"direction": "input"},
                            )
                        yield ErrorEvent(
                            message="Input blocked by safety policy.",
                            recoverable=False,
                        )
                        return

            # Step 0.5: Guards of a Prompt used as instructions
            if self._prompt_config is not None:
                from .memory import _extract_user_text as _ext_prompt

                prompt_text = _ext_prompt(input)
                if prompt_text:
                    try:
                        checked_text = await self._check_prompt_input(prompt_text)
                    except Exception:
                        yield ErrorEvent(
                            message="Input blocked by prompt guard.",
                            recoverable=False,
                        )
                        return
                    if checked_text != prompt_text:
                        input = _replace_last_user_text(input, checked_text)

            # Step 1: Memory injection
            if self.provider is not None:
                from .memory import (
                    _extract_user_text,
                    _format_memory_context,
                    _inject_memory_into_messages,
                )

                user_text = _extract_user_text(input)
                results = await self._search_memory(user_text)
                if results:
                    context = _format_memory_context(results)
                    input = _inject_memory_into_messages(input, context)

            # Step 2: Inject callback handler
            if self._handler is not None:
                config = dict(config) if config else {}
                callbacks = list(config.get("callbacks", []))
                callbacks.append(self._handler)
                config["callbacks"] = callbacks

            # Step 3: Stream via LangGraph astream_events
            try:
                async for event in self._inner.astream_events(
                    input, config=cast("RunnableConfig | None", config), version="v2", **kwargs
                ):
                    etype = event.get("event", "")

                    if etype == "on_tool_start":
                        tool_name = event.get("name", "unknown")
                        run_id = event.get("run_id", "")
                        args: dict[str, Any] = event.get("data", {}).get("input", {})
                        if isinstance(args, str):
                            args = {"input": args}
                        if include_arguments and self._guardrails is not None:
                            args = await _redact_args(args, self._guardrails)
                        elif not include_arguments:
                            args = {}
                        # Use run_id as key (handles parallel calls to same tool)
                        _tool_starts[run_id] = (time.monotonic(), _tool_counter)
                        yield ToolStartEvent(
                            tool_name=tool_name,
                            tool_display_name=_display_name(tool_name, tool_display_names),
                            arguments=args if include_arguments else {},
                            tool_index=_tool_counter,
                        )
                        _tool_counter += 1

                    elif etype == "on_tool_end":
                        tool_name = event.get("name", "unknown")
                        run_id = event.get("run_id", "")
                        output = event.get("data", {}).get("output", "")
                        result_str = str(output) if output else ""
                        start_t, idx = _tool_starts.pop(run_id, (_start, 0))
                        summary = _summary(result_str)
                        yield ToolEndEvent(
                            tool_name=tool_name,
                            tool_summary=summary,
                            duration_ms=round((time.monotonic() - start_t) * 1000, 1),
                            success=True,
                            tool_index=idx,
                        )
                        _all_tool_calls.append(
                            {"name": tool_name, "summary": summary, "success": True}
                        )

                    elif etype == "on_tool_error":
                        tool_name = event.get("name", "unknown")
                        run_id = event.get("run_id", "")
                        start_t, idx = _tool_starts.pop(run_id, (_start, 0))
                        yield ToolEndEvent(
                            tool_name=tool_name,
                            tool_summary="Error occurred",
                            duration_ms=round((time.monotonic() - start_t) * 1000, 1),
                            success=False,
                            tool_index=idx,
                        )
                        _all_tool_calls.append(
                            {"name": tool_name, "summary": "Error", "success": False}
                        )

                    elif etype == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if chunk is not None:
                            content = getattr(chunk, "content", None)
                            if content:
                                _cumulative += content
                                yield TokenEvent(
                                    text=content,
                                    cumulative_text=_cumulative,
                                )

            except Exception as exc:
                yield ErrorEvent(
                    message="An error occurred during processing.",
                    recoverable=False,
                )
                if self._event_notifier is not None:
                    from .events import emit_event

                    emit_event(
                        self._event_notifier,
                        "invocation.error",
                        "error",
                        {"error": type(exc).__name__, "streaming": True},
                        agent_id=self._actor(),
                    )
                return

            # Step 4: Output guardrails on accumulated text
            final_response = _cumulative
            if self._guardrails is not None and final_response:
                try:
                    checked = await self._guardrails.check_output(final_response)
                    if isinstance(checked, str) and checked != final_response:
                        final_response = checked
                        if self._event_notifier is not None:
                            from .events import emit_event

                            emit_event(
                                self._event_notifier,
                                "guardrail.redacted",
                                "info",
                                {"direction": "output", "streaming": True},
                            )
                except Exception as guard_exc:
                    # GuardrailViolation = output blocked. Yield error, don't
                    # serve the unsafe response. Same security as ainvoke().
                    guard_type = type(guard_exc).__name__
                    if "Violation" in guard_type or "Guardrail" in guard_type:
                        if self._event_notifier is not None:
                            from .events import emit_event

                            emit_event(
                                self._event_notifier,
                                "guardrail.blocked",
                                "warning",
                                {"direction": "output", "streaming": True},
                            )
                        yield ErrorEvent(
                            message="Output blocked by safety policy.",
                            recoverable=False,
                        )
                        return
                    # Other exceptions — log and continue with unredacted response
                    logger.warning("Output guardrail error in stream: %s", guard_exc)

            if self._prompt_config is not None and final_response:
                checked_output: Any = final_response
                try:
                    for g in self._prompt_guards("output"):
                        checked_output = await g.check_output(checked_output)
                except Exception:
                    yield ErrorEvent(
                        message="Output blocked by prompt guard.",
                        recoverable=False,
                    )
                    return
                if isinstance(checked_output, str):
                    final_response = checked_output

            # Step 5: Memory auto-store
            if self.provider is not None and _cumulative:
                try:
                    from .memory import _extract_user_text

                    user_text = _extract_user_text(input)
                    if user_text:
                        await self.provider.add(
                            f"User: {user_text}\nAssistant: {_cumulative[:500]}"
                        )
                except Exception:
                    pass

            # Step 6: Yield done event
            duration = round((time.monotonic() - _start) * 1000, 1)
            yield DoneEvent(
                full_response=final_response,
                tool_calls=_all_tool_calls,
                duration_ms=duration,
                cache_hit=False,
            )

            # Emit invocation.complete
            if self._event_notifier is not None:
                from .events import emit_event

                emit_event(
                    self._event_notifier,
                    "invocation.complete",
                    "info",
                    {"duration_ms": duration, "streaming": True},
                    agent_id=self._actor(),
                )

        finally:
            _caller_ctx_var.reset(_ctx_token)

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    async def shutdown(self) -> None:
        """Flush transporters, close MCP connections, and release resources.

        Always safe to call — no-ops for features that are not enabled.
        Call this when the agent is no longer needed.
        """
        # Close persistent MCP connections.
        # Catch BaseException (not just Exception) because the MCP SDK's
        # Streamable HTTP transport can raise asyncio.CancelledError
        # (a BaseException) via anyio cancel scopes during cleanup.
        if self._mcp_multi is not None:
            try:
                await self._mcp_multi.__aexit__(None, None, None)
            except BaseException:
                logger.debug("MCP cleanup error during shutdown", exc_info=True)
            self._mcp_multi = None

        # Flush observability transporters
        for t in self._transporters:
            try:
                if hasattr(t, "flush"):
                    result = t.flush()
                    if hasattr(result, "__await__"):
                        await result
            except Exception:
                logger.debug("Transporter flush error during shutdown", exc_info=True)

        # Stop event notifier (drain remaining events)
        if self._event_notifier is not None and hasattr(self._event_notifier, "stop"):
            try:
                await self._event_notifier.stop()
            except Exception:
                logger.debug("Event notifier stop error during shutdown", exc_info=True)

        # Close semantic cache (Redis connections, etc.)
        if self._cache is not None and hasattr(self._cache, "close"):
            try:
                await self._cache.close()
            except Exception:
                logger.debug("Cache close error during shutdown", exc_info=True)

        # Close conversation store
        if self._conversation_store is not None and hasattr(self._conversation_store, "close"):
            try:
                result = self._conversation_store.close()
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.debug("Conversation store close error during shutdown", exc_info=True)

        # Close memory provider if it has a close method
        if self.provider is not None and hasattr(self.provider, "close"):
            try:
                result = self.provider.close()
                if hasattr(result, "__await__"):
                    await result
            except Exception:
                logger.debug("Memory provider close error during shutdown", exc_info=True)

        # Clean up sandbox session and manager
        sandbox_session = getattr(self, "_sandbox_session", None)
        if sandbox_session is not None:
            try:
                await sandbox_session.cleanup()
            except Exception:
                logger.debug("Sandbox session cleanup error during shutdown", exc_info=True)
            self._sandbox_session = None

        sandbox_manager = getattr(self, "_sandbox_manager", None)
        if sandbox_manager is not None:
            try:
                await sandbox_manager.cleanup_all()
            except Exception:
                logger.debug("Sandbox manager cleanup error during shutdown", exc_info=True)
            self._sandbox_manager = None

    # -----------------------------------------------------------------
    # Observability accessors
    # -----------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate observability statistics.

        Returns an empty dict when observability is not enabled.
        """
        if self.collector is None:
            return {}
        return self.collector.get_stats()

    def generate_report(
        self,
        path: str,
        title: str = "Agent Observability Report",
    ) -> str:
        """Generate an interactive HTML report and return its file path.

        Raises:
            RuntimeError: When observability is not enabled.
        """
        if self.collector is None:
            raise RuntimeError(
                "Cannot generate report: observability is not enabled. "
                "Pass observe=True to build_agent()."
            )
        from .observability_transporters import HTMLReportTransporter

        transporter = HTMLReportTransporter(
            output_dir=str(Path(path).parent),
            session_name=Path(path).stem,
        )
        transporter._collector = self.collector
        transporter.flush()
        return path

    # -----------------------------------------------------------------
    # Memory helpers (internal)
    # -----------------------------------------------------------------

    async def _search_memory(self, query: str) -> list[Any]:
        """Search memory with timeout and graceful degradation.

        Callers must guard with ``self.provider is not None`` before invoking.
        """
        if not query.strip():
            return []
        assert self.provider is not None  # guarded by caller
        caller = get_current_caller()
        # Memory scoping keys on the isolation key (tenant::user) so tenants
        # with identical user ids never share memories.
        user_id = caller.isolation_key if caller is not None else None
        try:
            results = await asyncio.wait_for(
                self.provider.search(query, limit=self._memory_max, user_id=user_id),
                timeout=self._memory_timeout,
            )
            if self._memory_min_score > 0.0:
                results = [r for r in results if r.score >= self._memory_min_score]
            return results
        except asyncio.TimeoutError:
            logger.warning("Memory search timed out after %.1fs", self._memory_timeout)
            return []
        except Exception:
            logger.warning("Memory search failed", exc_info=True)
            return []

    async def _maybe_store(self, user_text: str, output: Any) -> None:
        """Optionally store the exchange in memory after invocation.

        Callers must guard with ``self.provider is not None`` before invoking.
        """
        if not self._memory_auto_store:
            return
        assert self.provider is not None  # guarded by caller
        from .memory import _extract_user_text

        caller = get_current_caller()
        # Memory scoping keys on the isolation key (tenant::user) so tenants
        # with identical user ids never share memories.
        user_id = caller.isolation_key if caller is not None else None
        output_text = _extract_user_text(output)
        content = f"User: {user_text}\nAssistant: {output_text}"
        try:
            await asyncio.wait_for(
                self.provider.add(
                    content,
                    metadata={"source": "auto_store"},
                    user_id=user_id,
                ),
                timeout=self._memory_timeout,
            )
        except Exception:
            logger.warning("Memory auto-store failed", exc_info=True)

    # -----------------------------------------------------------------
    # High-level chat API with session persistence
    # -----------------------------------------------------------------

    async def chat(
        self,
        message: str,
        *,
        session_id: str,
        user_id: str | None = None,
        caller: CallerContext | None = None,
        metadata: dict[str, Any] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        """Send a message and get a response, with automatic session persistence.

        This is the high-level API for building chat applications.  The
        conversation store (if configured) handles loading history,
        persisting new messages, and session lifecycle automatically.

        If no conversation store is configured, this still works — it
        just has no history beyond the current call.

        **Ownership enforcement**: When ``user_id`` (or ``caller.user_id``)
        is provided and the session already exists, the store checks that
        the session belongs to that user.  If it belongs to a different
        user, :class:`~promptise.conversations.SessionAccessDenied` is raised.

        Args:
            message: The user's message text.
            session_id: Unique identifier for this conversation session.
                Use :func:`~promptise.conversations.generate_session_id`
                to create cryptographically secure, non-enumerable IDs.
            user_id: Optional user identifier.  Shorthand for
                ``caller=CallerContext(user_id=...)``.  If both ``user_id``
                and ``caller`` are provided, ``caller.user_id`` takes
                precedence.
            caller: Optional :class:`CallerContext` with per-request
                identity.  Carries user_id, bearer_token, roles, scopes,
                and metadata through the entire invocation.
            metadata: Optional metadata to attach to the user message
                (e.g. source, IP, device).
            system_prompt: Optional per-call system prompt override.

        Returns:
            The assistant's response text.

        Raises:
            SessionAccessDenied: If ``user_id`` is provided and the
                session belongs to a different user.

        Example::

            from promptise.conversations import generate_session_id

            agent = await build_agent(..., conversation_store=store)
            sid = generate_session_id()
            reply = await agent.chat("Hello!", session_id=sid, user_id="user-42")
            reply = await agent.chat("What did I say?", session_id=sid, user_id="user-42")
        """
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        from .conversations import Message

        # Resolve user_id: caller takes precedence over explicit user_id
        if caller is not None and caller.user_id is not None:
            # Ownership keys on the isolation key (tenant::user when a
            # tenant is set) so sessions can never cross tenants even for
            # identical user ids.
            user_id = caller.isolation_key

        # Step 1: Ownership check — before loading messages
        is_new_session = True
        if self._conversation_store is not None and hasattr(
            self._conversation_store, "get_session"
        ):
            try:
                existing = await self._conversation_store.get_session(session_id)
                if existing is not None:
                    is_new_session = False
                    self._enforce_ownership(existing, user_id)
            except PermissionError:
                raise  # Re-raise SessionAccessDenied
            except (ConnectionError, OSError) as exc:
                # Connection errors are security-relevant — if we can't verify
                # ownership, we must not proceed. Fail closed.
                raise RuntimeError(
                    f"Cannot verify session ownership (store unreachable): {exc}"
                ) from exc
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot verify session ownership (unexpected error): {exc}"
                ) from exc

        # Step 2: Load history from store
        history_messages: list[Message] = []
        if self._conversation_store is not None:
            try:
                history_messages = await self._conversation_store.load_messages(session_id)
            except Exception:
                logger.warning(
                    "Failed to load conversation history for session %s", session_id, exc_info=True
                )

        # Step 3: Build LangChain message list
        lc_messages: list[Any] = []
        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))

        for msg in history_messages:
            if msg.role == "user":
                lc_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_messages.append(AIMessage(content=msg.content))
            elif msg.role == "system":
                lc_messages.append(SystemMessage(content=msg.content))

        lc_messages.append(HumanMessage(content=message))

        # Step 4: Invoke the agent
        _owner_token = _session_owner_var.set(user_id)
        try:
            output = await self.ainvoke(
                {"messages": lc_messages}, caller=caller, session_id=session_id
            )
        finally:
            _session_owner_var.reset(_owner_token)

        # Step 5: Extract assistant response text
        response_text = _extract_response_text(output)

        # Step 6: Persist updated history
        if self._conversation_store is not None:
            user_msg = Message(
                role="user",
                content=message,
                metadata=metadata or {},
            )
            assistant_msg = Message(
                role="assistant",
                content=response_text,
            )
            updated = history_messages + [user_msg, assistant_msg]

            # Enforce rolling window
            if self._conversation_max_messages > 0:
                updated = updated[-self._conversation_max_messages :]

            try:
                await self._conversation_store.save_messages(session_id, updated)
            except Exception:
                logger.warning(
                    "Failed to save conversation history for session %s", session_id, exc_info=True
                )

            # Assign ownership on new sessions
            if is_new_session and user_id is not None:
                try:
                    if hasattr(self._conversation_store, "update_session"):
                        await self._conversation_store.update_session(
                            session_id, user_id=user_id, title=message[:100]
                        )
                except Exception:
                    logger.debug("Failed to update session metadata", exc_info=True)

        return response_text

    async def list_sessions(
        self,
        *,
        user_id: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Any]:
        """List conversation sessions from the configured store.

        Args:
            user_id: Filter by user.  ``None`` returns all sessions.
            limit: Maximum sessions to return.
            offset: Pagination offset.

        Returns:
            List of :class:`~promptise.conversations.SessionInfo` objects.

        Raises:
            RuntimeError: If no conversation store is configured.
        """
        if self._conversation_store is None:
            raise RuntimeError(
                "No conversation store configured. Pass conversation_store= to build_agent()."
            )
        return await self._conversation_store.list_sessions(
            user_id=user_id,
            limit=limit,
            offset=offset,
        )

    async def delete_session(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
    ) -> bool:
        """Delete a conversation session and all its messages.

        Args:
            session_id: The session to delete.
            user_id: When provided, verifies the session belongs to this
                user before deleting.

        Returns:
            ``True`` if the session existed and was deleted.

        Raises:
            RuntimeError: If no conversation store is configured.
            SessionAccessDenied: If ``user_id`` is provided and the
                session belongs to a different user.
        """
        if self._conversation_store is None:
            raise RuntimeError(
                "No conversation store configured. Pass conversation_store= to build_agent()."
            )
        # Ownership check
        if user_id is not None and hasattr(self._conversation_store, "get_session"):
            existing = await self._conversation_store.get_session(session_id)
            if existing is not None:
                self._enforce_ownership(existing, user_id)
        if self._flows is not None:
            self._flows.discard(
                lambda key: isinstance(key, tuple) and key[0] == "session" and key[2] == session_id
            )
        return await self._conversation_store.delete_session(session_id)

    async def update_session(
        self,
        session_id: str,
        *,
        calling_user_id: str | None = None,
        user_id: str | None = ...,  # type: ignore[assignment]
        title: str | None = ...,  # type: ignore[assignment]
        metadata: dict[str, Any] | None = ...,  # type: ignore[assignment]
    ) -> bool:
        """Update session metadata (title, user_id, custom metadata).

        Only provided fields are updated — omitted fields are left unchanged.

        Args:
            session_id: The session to update.
            calling_user_id: The user making this request.  When provided,
                verifies ownership before applying changes.
            user_id: New user_id to assign (for ownership transfer).
            title: New title.
            metadata: New metadata dict.

        Raises:
            RuntimeError: If no conversation store is configured or the
                store does not support ``update_session``.
            SessionAccessDenied: If ``calling_user_id`` is provided and
                the session belongs to a different user.
        """
        if self._conversation_store is None:
            raise RuntimeError(
                "No conversation store configured. Pass conversation_store= to build_agent()."
            )
        if not hasattr(self._conversation_store, "update_session"):
            raise RuntimeError(
                f"{type(self._conversation_store).__name__} does not support update_session."
            )
        # Ownership check
        if calling_user_id is not None and hasattr(self._conversation_store, "get_session"):
            existing = await self._conversation_store.get_session(session_id)
            if existing is not None:
                self._enforce_ownership(existing, calling_user_id)

        kwargs: dict[str, Any] = {}
        if user_id is not ...:
            kwargs["user_id"] = user_id
        if title is not ...:
            kwargs["title"] = title
        if metadata is not ...:
            kwargs["metadata"] = metadata
        return await self._conversation_store.update_session(session_id, **kwargs)

    async def get_session(self, session_id: str, *, user_id: str | None = None) -> Any:
        """Get session metadata.

        Args:
            session_id: The session to retrieve.
            user_id: When provided, verifies the session belongs to this user.

        Returns:
            :class:`~promptise.conversations.SessionInfo` or ``None``.

        Raises:
            RuntimeError: If no conversation store is configured.
            SessionAccessDenied: If ``user_id`` is provided and the
                session belongs to a different user.
        """
        if self._conversation_store is None:
            raise RuntimeError(
                "No conversation store configured. Pass conversation_store= to build_agent()."
            )
        if not hasattr(self._conversation_store, "get_session"):
            raise RuntimeError(
                f"{type(self._conversation_store).__name__} does not support get_session."
            )
        info = await self._conversation_store.get_session(session_id)
        if info is not None and user_id is not None:
            self._enforce_ownership(info, user_id)
        return info

    @staticmethod
    def _enforce_ownership(session_info: Any, user_id: str | None) -> None:
        """Check that user_id matches the session owner.

        Does nothing if:
        - ``user_id`` is ``None`` (no enforcement requested)
        - The session has no ``user_id`` set (unowned session)

        Raises:
            SessionAccessDenied: If the session belongs to a different user.
        """
        if user_id is None:
            return
        session_owner = session_info.user_id
        if session_owner is None:
            # Unowned session — anyone can access
            return
        if session_owner != user_id:
            from .conversations import SessionAccessDenied

            raise SessionAccessDenied(
                session_id=session_info.session_id,
                attempted_user_id=user_id,
                owner_user_id=session_owner,
            )

    @staticmethod
    def _replace_response_text(output: Any, new_text: str) -> Any:
        """Replace the last AI message content with redacted text."""
        if isinstance(output, dict):
            messages = output.get("messages")
            if messages and isinstance(messages, list):
                for msg in reversed(messages):
                    if hasattr(msg, "type") and msg.type == "ai" and hasattr(msg, "content"):
                        msg.content = new_text
                        break
        return output

    def _prompt_guards(self, direction: str) -> list[Any]:
        """Guards of the Prompt or PromptSuite used as instructions."""
        from .prompts.core import Prompt
        from .prompts.suite import PromptSuite

        cfg = self._prompt_config
        if isinstance(cfg, Prompt):
            prompts = [cfg]
        elif isinstance(cfg, PromptSuite):
            prompts = list(cfg.prompts.values())
        else:
            return []
        guards: list[Any] = []
        seen: set[int] = set()
        for p in prompts:
            for g in p._input_guards if direction == "input" else p._output_guards:
                if id(g) not in seen:
                    seen.add(id(g))
                    guards.append(g)
        return guards

    async def _check_prompt_input(self, user_text: str) -> str:
        """Run the instruction Prompt's input guards on the user message.

        A guard may transform the text or raise
        :class:`~promptise.prompts.guards.GuardError` to reject it.
        """
        for g in self._prompt_guards("input"):
            user_text = await g.check_input(user_text)
        return user_text

    async def _check_prompt_output(self, output: Any) -> Any:
        """Run the instruction Prompt's output guards on the agent's reply."""
        guards = self._prompt_guards("output")
        if not guards:
            return output
        response_text = _extract_response_text(output)
        if not response_text:
            return output
        checked: Any = response_text
        for g in guards:
            checked = await g.check_output(checked)
        if isinstance(checked, str) and checked != response_text:
            output = self._replace_response_text(output, checked)
        return output

    async def _inject_prompt_context(self, input: Any, user_text: str) -> Any:
        """Run prompt context providers and inject dynamic context."""
        try:
            from .prompts.context import PromptContext
            from .prompts.core import Prompt
            from .prompts.suite import PromptSuite

            prompt_obj = self._prompt_config
            ctx = PromptContext(
                prompt_name=getattr(prompt_obj, "name", "agent"),
                model=self.model_name or "",
                rendered_text=user_text,
                agent=self,
            )

            # Resolve active prompt from suite if needed
            if isinstance(prompt_obj, PromptSuite):
                # Render all suite prompts
                context_text = await prompt_obj.render_async(ctx)
            elif isinstance(prompt_obj, Prompt):
                context_text = await prompt_obj.render_async(ctx)
            else:
                return input

            if not context_text:
                return input

            # Inject as SystemMessage into the messages
            from langchain_core.messages import SystemMessage

            if isinstance(input, dict) and "messages" in input:
                messages = list(input["messages"])
                # Insert prompt context after any existing system messages
                # (handles both LangChain SystemMessage objects and plain dicts)
                insert_idx = 0
                for i, msg in enumerate(messages):
                    is_sys = isinstance(msg, SystemMessage) or (
                        isinstance(msg, dict) and msg.get("role") == "system"
                    )
                    if is_sys:
                        insert_idx = i + 1
                    else:
                        break
                messages.insert(insert_idx, SystemMessage(content=context_text))
                return {**input, "messages": messages}

            return input
        except ImportError:
            logger.debug("Prompt framework not available, skipping context injection")
            return input
        except Exception:
            logger.warning("Prompt context injection failed", exc_info=True)
            return input

    def _flow_key(self, session_id: str | None) -> tuple[str, ...] | None:
        """Identify the conversation whose flow state this call advances.

        Flow state is kept per session (scoped to its owner) when a
        ``session_id`` is given, otherwise per caller.  Calls with neither
        get a throwaway flow, so state never leaks between conversations.
        """
        caller = _caller_ctx_var.get()
        owner = caller.isolation_key if caller is not None else None
        if owner is None:
            owner = _session_owner_var.get()
        if session_id is not None:
            return ("session", owner or "", session_id)
        if owner is not None:
            return ("caller", owner)
        return None

    def get_flow(
        self,
        session_id: str | None = None,
        *,
        caller: CallerContext | None = None,
        user_id: str | None = None,
    ) -> Any | None:
        """Return the conversation flow kept for a session or caller.

        Use it to inspect a conversation's phase and prompt, e.g.
        ``agent.get_flow("sess-1").get_prompt().included``.

        Args:
            session_id: The session passed to :meth:`chat` or :meth:`ainvoke`.
            caller: The caller the session or calls were made with.
            user_id: The ``user_id`` passed to :meth:`chat` (when no
                ``caller`` was used).

        Returns:
            The :class:`~promptise.prompts.flows.ConversationFlow`, or
            ``None`` if the agent has no flow or has not seen that
            conversation.
        """
        if self._flows is None:
            return None
        owner = caller.isolation_key if caller is not None else None
        if owner is None:
            owner = user_id
        if session_id is not None:
            return self._flows.get(("session", owner or "", session_id))
        if owner is not None:
            return self._flows.get(("caller", owner))
        return None

    async def _inject_flow_context(self, input: Any, session_id: str | None) -> tuple[Any, str]:
        """Advance this conversation's flow and put its prompt first.

        Returns the new input and the flow's prompt text (``""`` if none).

        The prompt goes in as a leading system message tagged
        :data:`~promptise.engine.nodes.AGENT_PROMPT_MESSAGE_ID`; the
        model node folds its own instructions and tool list into that
        message, so the model gets one system prompt.

        Callers must guard with ``self._flows is not None`` before invoking.
        """
        try:
            from langchain_core.messages import SystemMessage

            from .engine.nodes import AGENT_PROMPT_MESSAGE_ID

            assert self._flows is not None  # guarded by caller
            _, assembled = await self._flows.advance(
                self._flow_key(session_id), _user_messages(input)
            )

            if not assembled.text:
                return input, ""

            if isinstance(input, dict) and "messages" in input:
                messages = list(input["messages"])
                # Flow goes at index 0 — it defines the agent's current
                # behavioral phase (e.g., "opening", "analysis", "conclusion")
                # and has the highest effective priority in the context stack.
                messages.insert(
                    0, SystemMessage(content=assembled.text, id=AGENT_PROMPT_MESSAGE_ID)
                )
                return {**input, "messages": messages}, assembled.text

            return input, ""
        except Exception:
            logger.warning("Flow context injection failed", exc_info=True)
            return input, ""

    # -----------------------------------------------------------------
    # Passthrough for inner graph attributes
    # -----------------------------------------------------------------

    @property
    def tools(self) -> list[BaseTool]:
        """Every tool this agent can call, as LangChain tools.

        MCP-discovered tools, ``extra_tools``, sandbox tools, cross-agent and
        meta tools all appear here — the list the model was bound to.
        """
        return list(self._tools)

    @property
    def tool_names(self) -> list[str]:
        """The names of :attr:`tools`, in binding order."""
        return [t.name for t in self._tools]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _replace_last_user_text(input: Any, text: str) -> Any:
    """Return *input* with its last user message's text set to *text*."""
    if isinstance(input, str):
        return text
    if not isinstance(input, dict) or not input.get("messages"):
        return input
    messages = list(input["messages"])
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, dict) and msg.get("role") in ("user", "human"):
            messages[i] = {**msg, "content": text}
            break
        if getattr(msg, "type", None) == "human" and hasattr(msg, "model_copy"):
            messages[i] = msg.model_copy(update={"content": text})
            break
    return {**input, "messages": messages}


def _user_messages(input: Any) -> list[str]:
    """Return the text of every user message in an invocation input, in order."""
    if isinstance(input, str):
        return [input] if input else []
    if not isinstance(input, dict):
        return []
    texts: list[str] = []
    for msg in input.get("messages", []) or []:
        if isinstance(msg, dict):
            if msg.get("role") not in ("user", "human"):
                continue
            content = msg.get("content", "")
        elif getattr(msg, "type", None) == "human":
            content = getattr(msg, "content", "")
        else:
            continue
        if isinstance(content, list):
            # Multimodal content: keep the text parts.
            content = " ".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
                if isinstance(part, str) or (isinstance(part, dict) and part.get("type") == "text")
            )
        if content:
            texts.append(str(content))
    return texts


def _extract_response_text(output: Any) -> str:
    """Extract the assistant's response text from agent output.

    Handles the multiple output formats LangGraph agents can return:
    dict with ``messages``, dict with ``output``, or raw string.
    """
    if isinstance(output, dict):
        # LangGraph ReAct agent returns {"messages": [...]}
        messages = output.get("messages")
        if messages and isinstance(messages, list):
            # Walk backwards to find the last AI message
            for msg in reversed(messages):
                if hasattr(msg, "content") and getattr(msg, "type", None) == "ai":
                    content = msg.content
                    if isinstance(content, str) and content.strip():
                        return content
                    # Handle list content (multimodal)
                    if isinstance(content, list):
                        texts = [
                            c.get("text", "") if isinstance(c, dict) else str(c) for c in content
                        ]
                        joined = " ".join(t for t in texts if t)
                        if joined:
                            return joined
            # Fallback: last message with content
            for msg in reversed(messages):
                if hasattr(msg, "content"):
                    content = msg.content
                    if isinstance(content, str) and content.strip():
                        return content

        # Older agent format
        if "output" in output:
            return str(output["output"])

    if isinstance(output, str):
        return output

    return str(output)


def _normalize_model(model: ModelLike) -> Runnable[Any, Any]:
    """Normalize the supplied model into a Runnable.

    A string goes through :func:`promptise.models.resolve_model`, which
    accepts every provider prefix and alias (``azure:``, ``gemini:``,
    ``bedrock:`` …) and turns a missing package or environment variable into
    an actionable :class:`~promptise.models.ModelSetupError`.
    """
    if isinstance(model, str):
        from .models import resolve_model

        return cast(Runnable[Any, Any], resolve_model(model))
    if isinstance(model, Model):
        return cast(Runnable[Any, Any], model.resolve())
    # Already BaseChatModel or Runnable
    return cast(Runnable[Any, Any], model)


async def build_agent(
    *,
    servers: Mapping[str, ServerSpec],
    model: ModelLike,
    instructions: str | Any | None = None,
    identity: AgentIdentity | None = None,
    trace_tools: bool = False,
    cross_agents: Mapping[str, CrossAgent] | None = None,
    sandbox: bool | dict[str, Any] | None = None,
    observer: Any | None = None,
    observer_agent_id: str | None = None,
    observe: bool | Any | None = None,
    memory: Any | None = None,
    memory_auto_store: bool = False,
    extra_tools: list[BaseTool] | None = None,
    flow: Any | None = None,
    conversation_store: Any | None = None,
    conversation_max_messages: int = 0,
    optimize_tools: Any | None = None,
    guardrails: Any | None = None,
    cache: Any | None = None,
    approval: Any | None = None,
    events: Any | None = None,
    max_invocation_time: float = 0,
    adaptive: Any | None = None,
    context_engine: Any | None = None,
    agent_pattern: str | Any | None = None,
    pattern: str | Any | None = None,  # Deprecated alias for agent_pattern
    graph_blocks: list[Any] | None = None,
    node_pool: list[Any] | None = None,
    max_agent_iterations: int = 25,
) -> PromptiseAgent:
    """Build an MCP-first agent and return a :class:`PromptiseAgent`.

    Discovers tools from the configured MCP servers, converts them into
    LangChain tools, and builds an agent graph.  The result is always a
    :class:`PromptiseAgent` with observability and memory as opt-in
    capabilities.

    Args:
        servers: Mapping of server name to spec (HTTP/SSE recommended).
        model: REQUIRED. A LangChain chat model instance, a provider id
            string such as ``"openai:gpt-5-mini"`` or ``"azure:my-deployment"``
            (see :mod:`promptise.models` for every provider and alias), or a
            Runnable.
        instructions: Optional system prompt.  Defaults to the built-in
            ``DEFAULT_SYSTEM_PROMPT``.
        identity: Optional :class:`~promptise.identity.AgentIdentity`
            giving the agent a traceable identity. When provided, every
            tool call and LLM turn the agent records is attributed to the
            agent's identifier — its ``agent_id`` handle, or, for an
            IdP-backed identity with no handle, the IdP ``subject`` —
            unless ``observer_agent_id`` is set explicitly. A **verifiable**
            identity is also presented to MCP servers that have no bearer
            of their own (its credential becomes their ``bearer_token``),
            so the server can authenticate and attribute the calling
            agent. The identity is exposed as
            :attr:`PromptiseAgent.identity`.
        trace_tools: Print each tool invocation and result to stdout.
        cross_agents: Optional mapping of peer name → CrossAgent.  Each
            peer is exposed as an ``ask_agent_<name>`` tool.
        memory: Optional :class:`~promptise.memory.MemoryProvider`.
            When provided, the agent automatically searches memory before
            each invocation and injects relevant context as a
            ``SystemMessage``.
        memory_auto_store: When ``True`` and *memory* is provided,
            automatically store each exchange in long-term memory after
            invocation.  Defaults to ``False``.
        sandbox: Optional sandbox configuration (``True``, dict, or
            ``None``).
        observer: Optional :class:`ObservabilityCollector` to reuse.
        observer_agent_id: Agent identifier for tool-event recording.
        observe: Plug-and-play observability.  Can be:
            - ``True``: Enable with defaults (STANDARD level, HTML report).
            - :class:`ObservabilityConfig`: Full configuration.
            - ``None``/``False``: Disabled (default).
        extra_tools: Optional additional :class:`BaseTool` instances to
            include alongside MCP-discovered tools.  Used by the runtime
            for meta-tools (open mode) and custom agent-created tools.
        flow: Optional :class:`~promptise.prompts.flows.ConversationFlow`
            instance, subclass, or zero-argument factory.  The system
            prompt then evolves across turns with the flow's phase and
            active blocks.  Each conversation gets its own flow: per
            ``session_id`` (see :meth:`PromptiseAgent.chat`), else per
            :class:`CallerContext`; an instance is deep-copied as a
            template.  Calls with neither get a throwaway flow built from
            the messages passed in.  The flow's prompt comes first, and
            ``instructions`` and the tool list are appended to it in the
            same system message; with no ``instructions``, the default
            system prompt is not added.
        conversation_store: Optional
            :class:`~promptise.conversations.ConversationStore`.  When
            provided, the agent's :meth:`~PromptiseAgent.chat` method
            automatically loads and persists conversation history.
        conversation_max_messages: Maximum messages to keep per session
            when using the conversation store.  ``0`` = unlimited.
            Oldest messages are dropped when the limit is reached.

    Returns:
        A :class:`PromptiseAgent` instance.
        Use ``agent.get_stats()``, ``agent.generate_report(path)``, and
        ``await agent.shutdown()`` for observability and cleanup.
    """
    if model is None:  # Defensive check; CLI/code must always pass a model now.
        raise ValueError("A model is required. Provide a model instance or a provider id string.")

    # Validate the flow before any server connects: a template that can't be
    # copied per conversation fails here, not after resources are open.
    _flow_sessions = FlowSessions(flow) if flow is not None else None

    # Attribute recorded events to the agent's identity by default, so the
    # observability timeline answers "which agent did what" without extra
    # wiring. An explicit observer_agent_id still wins. When the identity is
    # IdP-backed with no local handle, derive the id from the IdP subject —
    # best-effort, so an unreachable IdP at build time does not fail the build.
    if identity is not None and observer_agent_id is None:
        if identity.agent_id is not None:
            observer_agent_id = identity.agent_id
        else:
            try:
                observer_agent_id = identity.subject()
            except IdentityError as exc:
                # Attribution falls back to the default; lower stakes than the
                # credential presentation above, so this is a debug note.
                logger.debug(
                    "Could not resolve IdP subject for attribution (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
                observer_agent_id = None

    # Simple printing callbacks for tracing (kept dependency-free).
    # When an observer is provided the callbacks also record tool events
    # to the timeline for full agent-level transparency.
    _obs = observer
    _obs_aid = observer_agent_id

    def _before(name: str, kwargs: dict[str, Any]) -> None:
        if trace_tools:
            print(f"→ Invoking tool: {name} with {kwargs}")
        if _obs is not None:
            from .observability import TimelineEventType, _truncate_for_metadata

            _obs.record(
                TimelineEventType.TOOL_CALL,
                agent_id=_obs_aid,
                details=f"Calling tool: {name}",
                metadata={
                    "tool_name": name,
                    "arguments": _truncate_for_metadata(str(kwargs)),
                },
            )

    def _after(name: str, res: Any) -> None:
        # Extract a human-readable preview from the raw CallToolResult.
        # The MCP SDK returns CallToolResult with .content = [TextContent, ...]
        # so we extract .text from each item for a clean preview string.
        pretty = res
        if hasattr(res, "content") and isinstance(res.content, list):
            texts = [item.text for item in res.content if hasattr(item, "text") and item.text]
            if texts:
                pretty = "\n".join(texts)
        else:
            for attr in ("data", "text", "result"):
                try:
                    val = getattr(res, attr, None)
                    if val not in (None, ""):
                        pretty = val
                        break
                except Exception:
                    continue
        if trace_tools:
            print(f"✔ Tool result from {name}: {pretty}")
        if _obs is not None:
            from .observability import TimelineEventType, _truncate_for_metadata

            _obs.record(
                TimelineEventType.TOOL_RESULT,
                agent_id=_obs_aid,
                details=f"Tool result from: {name}",
                metadata={
                    "tool_name": name,
                    "result_preview": _truncate_for_metadata(str(pretty)),
                },
            )

    def _error(name: str, exc: Exception) -> None:
        if trace_tools:
            print(f"✖ {name} error: {exc}")
        if _obs is not None:
            from .observability import TimelineEventType

            _obs.record(
                TimelineEventType.TOOL_ERROR,
                agent_id=_obs_aid,
                details=f"Tool error in: {name}",
                metadata={"tool_name": name, "error": str(exc)[:500]},
            )

    # ------------------------------------------------------------------
    # Normalize optimize_tools parameter
    # ------------------------------------------------------------------
    _opt_config = None
    if optimize_tools is not None and optimize_tools is not False:
        from .tool_optimization import OptimizationLevel, ToolOptimizationConfig

        if optimize_tools is True:
            _opt_config = ToolOptimizationConfig(level=OptimizationLevel.MINIMAL)
        elif isinstance(optimize_tools, str):
            _opt_config = ToolOptimizationConfig(
                level=OptimizationLevel(optimize_tools),
            )
        elif isinstance(optimize_tools, ToolOptimizationConfig):
            _opt_config = optimize_tools
        else:
            _opt_config = ToolOptimizationConfig(level=OptimizationLevel.MINIMAL)

    # Only create MCP client if there are servers to connect to
    tools: list[BaseTool] = []
    _promptise_multi = None  # track for cleanup

    if servers:
        _enable_callbacks = trace_tools or _obs is not None
        _cb_before = _before if _enable_callbacks else None
        _cb_after = _after if _enable_callbacks else None
        _cb_error = _error if _enable_callbacks else None

        from .mcp.client import MCPClient, MCPMultiClient, MCPToolAdapter

        # When the agent carries a verifiable identity, present its identity
        # credential to MCP servers that have no explicit bearer of their own —
        # scoped to each server's ``audience`` so one identity can serve several
        # resources. Best-effort: an unreachable IdP must not fail the build.
        def _identity_bearer_for(spec: HTTPServerSpec) -> str | None:
            if identity is None or not identity.is_verifiable:
                return None
            try:
                return identity.get_credential(spec.audience)
            except IdentityError as exc:
                # Do not silently drop the credential: an operator who
                # configured a verifiable identity expects authenticated MCP
                # calls. Surface the failure loudly; servers requiring auth
                # will then reject the unauthenticated calls.
                logger.warning(
                    "Agent identity %r could not acquire a credential for MCP "
                    "server audience %r (%s: %s); connecting without it.",
                    identity.agent_id,
                    spec.audience,
                    type(exc).__name__,
                    exc,
                )
                return None

        clients: dict[str, MCPClient] = {}
        for sname, spec in servers.items():
            if isinstance(spec, HTTPServerSpec):
                clients[sname] = MCPClient(
                    url=spec.url,
                    transport=spec.transport,
                    headers=spec.headers,
                    bearer_token=spec.bearer_token.get_secret_value()
                    if spec.bearer_token
                    else _identity_bearer_for(spec),
                    api_key=spec.api_key.get_secret_value() if spec.api_key else None,
                )
            else:
                # StdioServerSpec
                clients[sname] = MCPClient(
                    transport="stdio",
                    command=spec.command,
                    args=spec.args,
                    env=spec.env,
                    cwd=spec.cwd,
                )

        _promptise_multi = MCPMultiClient(clients)
        await _promptise_multi.__aenter__()

        adapter = MCPToolAdapter(
            _promptise_multi,
            on_before=_cb_before,
            on_after=_cb_after,
            on_error=_cb_error,
            optimize=_opt_config,
        )
        try:
            discovered = await adapter.as_langchain_tools()
            tools = list(discovered) if discovered else []
        except MCPClientError as exc:
            # Clean up on failure
            try:
                await _promptise_multi.__aexit__(None, None, None)
            except Exception:
                logger.debug("MCP multi-client cleanup error", exc_info=True)
            _promptise_multi = None
            raise RuntimeError(
                f"Failed to initialize agent because tool discovery failed. Details: {exc}"
            ) from exc

    # Attach cross-agent tools if provided
    if cross_agents:
        tools.extend(make_cross_agent_tools(cross_agents, caller_identity=identity))

    # The code-action pattern REQUIRES a sandbox (the model writes a program we
    # run in a container). Auto-enable one — with no network, since the program
    # reaches the outside world only through the host-bridged tools.
    _ca_name = (
        agent_pattern
        if isinstance(agent_pattern, str)
        else (pattern if isinstance(pattern, str) else None)
    )
    _is_code_action = _ca_name == "code-action"
    if _is_code_action and not sandbox:
        sandbox = {"network": "none"}

    # Attach sandbox tools if enabled
    sandbox_manager = None
    sandbox_session = None
    if sandbox:
        try:
            from .sandbox import SandboxManager
            from .sandbox.tools import create_sandbox_tools

            print("[promptise] Initializing sandbox environment...")
            sandbox_manager = SandboxManager(sandbox)
            if _is_code_action:
                # code-action creates a fresh session per run and does NOT expose
                # the sandbox meta-tools to the agent. Probe once so we fail fast
                # with a clear error if Docker is unavailable.
                _probe = await sandbox_manager.create_session()
                await _probe.cleanup()
                print(
                    "[promptise] Sandbox ready for code-action "
                    f"(backend: {sandbox_manager.config.backend})"
                )
            else:
                sandbox_session = await sandbox_manager.create_session()
                # Nested try to ensure cleanup if tool creation fails
                try:
                    sandbox_tools = create_sandbox_tools(sandbox_session)
                    tools.extend(sandbox_tools)
                    print(
                        f"[promptise] Sandbox ready: {len(sandbox_tools)} sandbox tools added "
                        f"(backend: {sandbox_manager.config.backend})"
                    )
                except Exception as tool_error:
                    # Clean up session before re-raising
                    if sandbox_session:
                        await sandbox_session.cleanup()
                    raise tool_error

        except Exception as e:
            if _is_code_action:
                # No silent fallback — code-action cannot run without a sandbox.
                raise RuntimeError(
                    "agent_pattern='code-action' requires a working Docker sandbox, "
                    f"but it could not be initialized: {e}"
                ) from e
            print(f"[promptise] Warning: Failed to initialize sandbox: {e}")
            print("[promptise] Agent will continue without sandbox capabilities.")
            # Ensure session is cleared on failure
            sandbox_manager = None
            sandbox_session = None

    # code-action: a factory that yields a fresh sandbox session per run.
    _code_action_factory: Any | None = None
    if _is_code_action and sandbox_manager is not None:
        _ca_mgr = sandbox_manager

        async def _code_action_factory() -> Any:
            return await _ca_mgr.create_session()

    # ------------------------------------------------------------------
    # Memory: normalize provider (actual wrapping happens after graph)
    # ------------------------------------------------------------------
    _memory_provider = None
    _memory_auto_store = memory_auto_store
    if memory is not None:
        from .memory import MemoryProvider

        if isinstance(memory, MemoryProvider):
            _memory_provider = memory
        elif isinstance(memory, dict):
            # Config dict from superagent — extract provider settings
            _memory_provider = _build_provider_from_config(memory)
        if _memory_provider is None:
            raise TypeError(
                f"Unrecognized memory type: {type(memory).__name__}. "
                f"Expected a MemoryProvider instance or config dict."
            )

    # Append extra tools (meta-tools, custom tools from runtime open mode)
    if extra_tools:
        tools.extend(extra_tools)

    if not tools:
        print("[promptise] No tools discovered from MCP servers; agent will run without tools.")

    chat: Runnable[Any, Any] = _normalize_model(model)

    # Handle Prompt/PromptSuite as instructions
    _prompt_config = None
    if instructions is not None and not isinstance(instructions, str):
        try:
            from .prompts.core import Prompt
            from .prompts.suite import PromptSuite

            if isinstance(instructions, (Prompt, PromptSuite)):
                _prompt_config = instructions
                # Extract static text for graph construction
                sys_prompt = (
                    instructions.render() if hasattr(instructions, "render") else str(instructions)
                )
            else:
                sys_prompt = str(instructions)
        except ImportError:
            sys_prompt = str(instructions)
    elif flow is not None:
        # The flow supplies the prompt; don't append the generic default.
        sys_prompt = instructions or ""
    else:
        sys_prompt = instructions or DEFAULT_SYSTEM_PROMPT

    # ----------------------------------------------------------------------
    # Build the PromptGraph engine (replaces LangGraph).
    # ----------------------------------------------------------------------

    def _build_graph(graph_tools: list[BaseTool]) -> Runnable[Any, Any]:
        """Build a PromptGraph engine with the given tools.

        Extracted as a function so the semantic tool selection system
        can cheaply rebuild the engine with different tool subsets.
        """
        # Resolve agent_pattern (with backward compat for pattern)
        _pattern = agent_pattern or pattern

        # Priority 1: node_pool — build autonomous graph from pool
        if node_pool:
            graph = PromptGraph.from_pool(node_pool, system_prompt=sys_prompt, name="autonomous")
        # Priority 2: PromptGraph instance passed directly
        elif isinstance(_pattern, PromptGraph):
            graph = _pattern
            # If autonomous mode and no edges, wrap in AutonomousNode
            if graph.mode == "autonomous" and not graph.edges:
                pool = list(graph.nodes.values())
                if pool:
                    graph = PromptGraph.from_pool(pool, system_prompt=sys_prompt)
        # Priority 3: String pattern name
        elif isinstance(_pattern, str):
            builders = {
                "react": lambda: PromptGraph.react(
                    tools=graph_tools, system_prompt=sys_prompt, blocks=graph_blocks
                ),
                "managed": lambda: PromptGraph.managed(
                    tools=graph_tools, system_prompt=sys_prompt, blocks=graph_blocks
                ),
                "code-action": lambda: PromptGraph.code_action(
                    tools=graph_tools,
                    system_prompt=sys_prompt,
                    blocks=graph_blocks,
                    sandbox_factory=_code_action_factory,
                ),
                "verify": lambda: PromptGraph.verify(
                    tools=graph_tools, system_prompt=sys_prompt, blocks=graph_blocks
                ),
                "peoatr": lambda: PromptGraph.peoatr(
                    tools=graph_tools, system_prompt=sys_prompt, blocks=graph_blocks
                ),
                "research": lambda: PromptGraph.research(
                    search_tools=graph_tools, system_prompt=sys_prompt, blocks=graph_blocks
                ),
                "autonomous": lambda: PromptGraph.autonomous(
                    tools=graph_tools, system_prompt=sys_prompt
                ),
                "deliberate": lambda: PromptGraph.deliberate(
                    tools=graph_tools, system_prompt=sys_prompt
                ),
                "debate": lambda: PromptGraph.debate(system_prompt=sys_prompt),
            }
            builder = builders.get(_pattern)
            if builder:
                graph = builder()
            else:
                # Try dynamic lookup on PromptGraph
                fn = getattr(PromptGraph, _pattern, None)
                if fn:
                    graph = fn(tools=graph_tools, system_prompt=sys_prompt)
                else:
                    graph = PromptGraph.react(tools=graph_tools, system_prompt=sys_prompt)
        # Priority 4: Duck-type PromptGraph-like object
        elif _pattern is not None and hasattr(_pattern, "_nodes"):
            graph = _pattern
        # Default: ReAct
        else:
            graph = PromptGraph.react(tools=graph_tools, system_prompt=sys_prompt)

        from langchain_core.language_models import BaseChatModel

        return cast(
            Runnable[Any, Any],
            PromptGraphEngine(
                graph=graph,
                model=cast(BaseChatModel, chat),
                max_iterations=max_agent_iterations,
            ),
        )

    # ------------------------------------------------------------------
    # Wrap tools with approval gates if configured
    # ------------------------------------------------------------------
    if approval is not None:
        from .approval import wrap_tools_with_approval

        tools = wrap_tools_with_approval(tools, approval, event_notifier=events)

    graph = _build_graph(tools)

    # ------------------------------------------------------------------
    # Prepare observability components (no wrapping — passed to agent)
    # ------------------------------------------------------------------
    _callback_handler = None
    _collector = None
    _observe_cfg = None
    _created_transporters: list[Any] = []

    if observe is not None and observe is not False:
        from .callback_handler import PromptiseCallbackHandler
        from .observability import ObservabilityCollector
        from .observability_config import ObservabilityConfig

        # Normalize observe=True to default config
        if observe is True:
            _observe_cfg = ObservabilityConfig()
        elif isinstance(observe, ObservabilityConfig):
            _observe_cfg = observe
        else:
            _observe_cfg = ObservabilityConfig()

        # Create or reuse the collector
        if _obs is not None:
            _collector = _obs
        else:
            _collector = ObservabilityCollector(
                session_name=_observe_cfg.session_name,
                max_entries=_observe_cfg.max_entries,
            )

        # Create the callback handler
        _agent_id = _obs_aid or "agent"
        _callback_handler = PromptiseCallbackHandler(
            _collector,
            agent_id=_agent_id,
            record_prompts=_observe_cfg.record_prompts,
            level=_observe_cfg.level,
        )

        # Create and register transporters from config
        from .observability_transporters import create_transporters

        _created_transporters = create_transporters(_observe_cfg, _collector)

    # ------------------------------------------------------------------
    # Construct unified PromptiseAgent — no wrapper chain needed
    # ------------------------------------------------------------------
    # Set up semantic tool selection if enabled
    # ------------------------------------------------------------------
    _tool_index = None
    _all_tools = None
    _graph_builder_fn = None

    if _opt_config is not None:
        from .tool_optimization import ToolIndex, _RequestMoreToolsTool, _resolve_config

        resolved = _resolve_config(_opt_config)
        if resolved.semantic_selection and tools:
            _tool_index = ToolIndex(tools, model_name_or_path=resolved.embedding_model)
            _all_tools = list(tools)

            # Add fallback tool if enabled
            if resolved.always_include_fallback:
                fallback = _RequestMoreToolsTool(tool_index=_tool_index)
                tools.append(fallback)
                # Rebuild graph with fallback included
                graph = _build_graph(tools)

            _graph_builder_fn = _build_graph

    # ------------------------------------------------------------------
    # Resolve model name string for prompt context
    _model_name: str | None = None
    if isinstance(model, str):
        _model_name = model
    elif isinstance(model, Model):
        _model_name = model.spec
    else:
        # Try to extract from BaseChatModel
        _model_name = getattr(model, "model_name", None) or getattr(model, "model", None)
        if _model_name is not None:
            _model_name = str(_model_name)

    agent = PromptiseAgent(
        inner=graph,
        handler=_callback_handler,
        collector=_collector,
        observe_config=_observe_cfg,
        transporters=_created_transporters,
        memory_provider=_memory_provider,
        memory_auto_store=_memory_auto_store,
        mcp_multi=_promptise_multi,
        model_name=_model_name,
        prompt_config=_prompt_config,
        conversation_store=conversation_store,
        conversation_max_messages=conversation_max_messages,
        tool_index=_tool_index,
        all_tools=_all_tools,
        graph_builder_fn=_graph_builder_fn,
        tools=list(tools),
        guardrails=guardrails,
        cache=cache,
        approval=approval,
        event_notifier=events,
    )

    # Attach the federated identity (if any) for downstream/tool auth.
    agent.identity = identity
    agent._actor_id = observer_agent_id

    # Set invocation timeout
    if max_invocation_time > 0:
        agent._max_invocation_time = max_invocation_time

    # Set up adaptive strategy manager
    if adaptive is not None and _memory_provider is not None:
        from .strategy import AdaptiveStrategyConfig, AdaptiveStrategyManager

        if isinstance(adaptive, bool) and adaptive:
            adaptive_config = AdaptiveStrategyConfig(enabled=True)
        elif isinstance(adaptive, AdaptiveStrategyConfig):
            adaptive_config = adaptive
        else:
            adaptive_config = None

        if adaptive_config is not None and adaptive_config.enabled:
            agent._strategy_manager = AdaptiveStrategyManager(
                config=adaptive_config,
                memory=_memory_provider,
                agent_model=_model_name,
                guardrails=guardrails,
            )

    # Wire context engine
    if context_engine is not None:
        agent._context_engine = context_engine

    # Wire event notifier to callback handler and cache
    if events is not None:
        if _callback_handler is not None:
            _callback_handler._event_notifier = events
        if cache is not None:
            cache._event_notifier = events
        # Auto-start the notifier
        await events.start()

    # Store raw instructions for context engine and cache fingerprinting
    agent._raw_instructions = sys_prompt if isinstance(sys_prompt, str) else str(instructions or "")

    # Attach conversation flow if provided
    if _flow_sessions is not None:
        agent._flows = _flow_sessions

    # Attach sandbox for cleanup on shutdown
    if sandbox_session is not None:
        agent._sandbox_session = sandbox_session
    if sandbox_manager is not None:
        agent._sandbox_manager = sandbox_manager

    return agent


def _build_provider_from_config(config: dict[str, Any]) -> Any:
    """Build a MemoryProvider from a superagent config dict."""
    from .memory import InMemoryProvider

    provider_type = config.get("provider", config.get("backend", "in_memory"))

    if provider_type == "mem0":
        from .memory import Mem0Provider

        return Mem0Provider(
            user_id=config.get("user_id", "default"),
            agent_id=config.get("agent_id"),
            config=config.get("config"),
        )
    elif provider_type == "chroma":
        from .memory import ChromaProvider

        return ChromaProvider(
            collection_name=config.get("collection", "agent_memory"),
            persist_directory=config.get("persist_directory"),
        )
    else:
        # in_memory or legacy backends (file, sqlite) → InMemoryProvider
        return InMemoryProvider()
