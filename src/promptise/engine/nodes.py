"""Built-in node types for the PromptGraph engine.

Nine composable node types that cover 95% of agent reasoning patterns.
Developers configure nodes — they don't subclass them.

Node types:

- **PromptNode** — LLM reasoning with blocks, tools, guards, strategy
- **ToolNode** — Explicit tool execution with validation and dedup
- **RouterNode** — LLM-based routing to decide the next path
- **GuardNode** — Validate state and gate transitions
- **ParallelNode** — Run multiple nodes concurrently
- **LoopNode** — Repeat a subgraph until a condition is met
- **HumanNode** — Pause for human input or approval
- **TransformNode** — Pure data transformation (no LLM call)
- **SubgraphNode** — Embed a complete sub-graph
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, cast
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .base import BaseNode
from .state import GraphState, NodeEvent, NodeResult

logger = logging.getLogger("promptise.engine")

# A node records a failure as ``NodeResult.error`` (a string — what hooks,
# history and reports carry) and keeps the exception behind it on the result
# under this attribute, so the engine can raise with the provider error as
# ``__cause__`` when the run ends on that failure.  It is deliberately not a
# dataclass field: it never serializes, never compares, never prints.
_FAILURE_CAUSE_ATTR = "_failure_cause"


def record_failure(result: NodeResult, exc: BaseException, error: str | None = None) -> None:
    """Record *exc* as the failure of *result*.

    Sets ``result.error`` (to *error*, or ``"ExcType: message"``) and keeps
    the exception object for :func:`failure_cause`.

    Args:
        result: The node result to mark as failed.
        exc: The exception that caused the failure.
        error: Error text to record; defaults to ``"ExcType: message"``.
    """
    result.error = error if error is not None else f"{type(exc).__name__}: {exc}"
    setattr(result, _FAILURE_CAUSE_ATTR, exc)


def failure_cause(result: NodeResult) -> BaseException | None:
    """The exception behind ``result.error``, or ``None`` when the failure was
    recorded without one (a node that set ``error`` itself)."""
    cause = getattr(result, _FAILURE_CAUSE_ATTR, None)
    return cause if isinstance(cause, BaseException) else None


# ---------------------------------------------------------------------------
# PromptNode — LLM reasoning step (the core node type)
# ---------------------------------------------------------------------------


class PromptNode(BaseNode):
    """A complete reasoning unit in the graph.

    Each PromptNode is a self-contained processing pipeline:

    **Input** → **Preprocess** → **Context Assembly** → **LLM Call** →
    **Tool Execution** → **Postprocess** → **Guards** → **Output**

    Every aspect is configurable:
    - What the LLM sees (blocks, strategy, perspective, context layers)
    - What the LLM can do (tools, tool_choice)
    - What the LLM must produce (output_schema, guards)
    - How data flows in (input_keys read from state.context)
    - How data flows out (output_key writes to state.context)
    - Pre/post processing (preprocessor, postprocessor callables)
    - Context from previous node (inherit_context_from)

    Args:
        name: Unique node identifier.

        blocks: PromptBlocks to assemble for this node's system prompt.
        strategy: Reasoning strategy (ChainOfThought, SelfCritique, etc.).
        perspective: Cognitive perspective (Analyst, Critic, etc.).

        tools: Tools available at THIS node.  ``None`` disables tools.
        tool_choice: ``"auto"`` (LLM decides), ``"required"`` (must call
            a tool), or ``"none"`` (tools shown but not callable).
        inject_tools: When ``True``, the engine auto-injects all
            discovered MCP tools into this node at runtime.  Use this
            when defining custom graphs where nodes should receive
            tools from ``build_agent(servers=...)`` without hardcoding
            them.  The injected tools merge with any tools already
            set on the node (no duplicates).

        output_schema: Pydantic model for structured output.
        guards: Guards that validate the output before proceeding.

        context_layers: Extra named context layers with priorities.
            Merged into the system prompt during assembly.
        max_tokens: Max response tokens for this node.
        temperature: LLM temperature for this node.

        input_keys: Keys to read from ``state.context`` and inject
            into the system prompt.  E.g. ``["search_results", "user_query"]``
            makes those values available in the prompt.
        output_key: Key to write this node's output to in ``state.context``.
            The next node can read it via ``input_keys``.
        inherit_context_from: Name of a previous node whose output
            should be injected as context.  Reads from
            ``state.context["{node_name}_output"]``.

        preprocessor: Async callable ``(state, config) → state`` that
            runs before the LLM call.  Use for data transformation,
            enrichment, or filtering.
        postprocessor: Async callable ``(output, state, config) → output``
            that runs after the LLM call.  Use for output transformation,
            validation, or enrichment.

        include_observations: Auto-inject recent tool results from state.
        include_plan: Auto-inject current plan/subgoals from state.
        include_reflections: Auto-inject past learnings from state.

        context_scope: How much history the model sees: ``"full"``,
            ``"auto"`` (full, compacting once the tool loop gets long),
            ``"ledger"`` (always compacted) or ``"scoped"``.  See
            :mod:`promptise.engine.compaction`.
        auto_ledger_after: Tool results before an ``"auto"`` node compacts
            (default 6).  Shorthand for
            ``compaction=ContextCompaction(after_tool_results=...)``.
        compaction: :class:`~promptise.engine.compaction.ContextCompaction`
            settings for this node (or ``True`` / ``False`` / an ``int``).
            Without it the node uses the settings ``build_agent`` passed,
            else the defaults.
    """

    def __init__(
        self,
        name: str,
        *,
        # Prompt composition
        blocks: list[Any] | None = None,
        strategy: Any | None = None,
        perspective: Any | None = None,
        # Tools
        tools: list[BaseTool] | None = None,
        tool_choice: str = "auto",
        inject_tools: bool = False,
        # Output
        output_schema: type | None = None,
        guards: list[Any] | None = None,
        # Model
        model_override: Any | None = None,
        # Context
        context_layers: dict[str, int] | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
        # Data flow
        input_keys: list[str] | None = None,
        output_key: str | None = None,
        inherit_context_from: str | None = None,
        context_scope: str = "full",
        auto_ledger_after: int | None = None,
        compaction: Any | None = None,
        # Processing pipeline
        preprocessor: Callable | None = None,
        postprocessor: Callable | None = None,
        # Auto-inject state data
        include_observations: bool = True,
        include_plan: bool = True,
        include_reflections: bool = True,
        # BaseNode params
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.blocks = list(blocks) if blocks else []
        self.strategy = strategy
        self.perspective = perspective
        from .state import NodeFlag

        self.tools = list(tools) if tools else []
        self.tool_choice = tool_choice
        # inject_tools maps to NodeFlag.INJECT_TOOLS
        if inject_tools:
            self.flags.add(NodeFlag.INJECT_TOOLS)
        self.model_override = model_override
        self.output_schema = output_schema
        self.guards = list(guards) if guards else []
        self.context_layers = dict(context_layers) if context_layers else {}
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Data flow
        self.input_keys = list(input_keys) if input_keys else []
        self.output_key = output_key
        self.inherit_context_from = inherit_context_from
        # Context lifecycle management — how much history this node sees:
        #   "auto": behaves like "full" for short tool loops, then switches to
        #     "ledger" once the loop grows past ``auto_ledger_after`` tool
        #     results. The default for the built-in ReAct pattern: simple tasks
        #     are unchanged, deep tool loops stay bounded — context handling and
        #     token efficiency without choosing a pattern.
        #   "full": the whole accumulated transcript.
        #   "scoped": only the system prompt (with any inherited/​injected
        #     distilled state), the original task, and the node's own
        #     in-progress tool loop — NOT other stages' verbose messages.
        #     Bounds token growth across a multi-stage reasoning graph.
        #   "ledger": for long single-node tool loops — replaces the growing
        #     tool-call/result transcript with a compact, DEDUPLICATED "facts
        #     gathered" ledger (from state.observations), so the model stops
        #     re-querying facts it already has.
        self.context_scope = context_scope
        # "auto" flips to ledger once this many tool results have accumulated.
        self._auto_ledger_explicit = auto_ledger_after is not None
        self.auto_ledger_after = auto_ledger_after if auto_ledger_after is not None else 6
        # Node-level compaction settings; None defers to the engine's.
        from .compaction import ContextCompaction

        self.compaction: ContextCompaction | None = (
            ContextCompaction.coerce(compaction) if compaction is not None else None
        )
        # Pipeline
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        # Auto-inject
        self.include_observations = include_observations
        self.include_plan = include_plan
        self.include_reflections = include_reflections

    @property
    def inject_tools(self) -> bool:
        """Whether this node receives MCP tools at runtime."""
        from .state import NodeFlag

        return NodeFlag.INJECT_TOOLS in self.flags

    def _compaction_settings(self, config: dict[str, Any] | None = None) -> Any:
        """The :class:`ContextCompaction` settings this node runs with.

        The node's own ``compaction`` wins, then the engine's (from
        ``build_agent(context_compaction=...)``), then the defaults.  An
        explicit ``auto_ledger_after`` overrides the threshold of the
        engine's settings.
        """
        from dataclasses import replace

        from .compaction import ContextCompaction

        if self.compaction is not None:
            return self.compaction
        settings = (config or {}).get("_engine_compaction") or ContextCompaction()
        if self._auto_ledger_explicit:
            settings = replace(settings, after_tool_results=self.auto_ledger_after)
        return settings

    def _effective_context_scope(
        self, state: GraphState, config: dict[str, Any] | None = None
    ) -> str:
        """Resolve ``context_scope``, expanding ``"auto"`` for the current state.

        ``"auto"`` stays ``"full"`` while the tool loop is short (no change to
        simple tasks) and switches to ``"ledger"`` once enough tool results have
        accumulated, or once the transcript passes the token budget, so deep
        tool loops stay bounded automatically.  With compaction disabled it
        stays ``"full"``.
        """
        if self.context_scope != "auto":
            return self.context_scope
        settings = self._compaction_settings(config)
        if not settings.enabled:
            return "full"
        if len(state.observations) >= settings.after_tool_results:
            return "ledger"
        if (
            settings.max_tokens is not None
            and settings.tokens(state.messages) > settings.max_tokens
        ):
            return "ledger"
        return "full"

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Execute the full node pipeline:

        1. Preprocessor (custom data transformation)
        2. Context assembly (blocks + input_keys + inherited context + state)
        3. LLM call (with tools if configured)
        4. Tool execution (auto-loop if tools called)
        5. Postprocessor (custom output transformation)
        6. Guards (validate output)
        7. Write output to state.context[output_key]
        """
        start = time.monotonic()
        result = NodeResult(node_name=self.name, node_type="prompt", iteration=state.iteration)

        # Resolve model — per-node override takes priority
        if self.model_override is not None:
            model = self.model_override
            # A string ("azure:chat-prod") or a Model resolves like build_agent(model=...)
            if isinstance(model, str) or type(model).__name__ == "Model":
                try:
                    from promptise.models import Model, resolve_model

                    model = model.resolve() if isinstance(model, Model) else resolve_model(model)
                except Exception as exc:
                    # Name the model by its spec, never by the object: a Model
                    # carries credentials and this string lands in logs, the
                    # graph history and the on_node_error event payload.
                    label = model.spec if isinstance(model, Model) else repr(model)
                    record_failure(result, exc, f"Failed to initialize model {label}: {exc}")
                    return result
        else:
            model = config.get("_engine_model")
        if model is None:
            result.error = "No model available in config"
            return result

        # ── 0. Run preprocessor ──
        if self.preprocessor:
            try:
                preprocess_result = self.preprocessor(state, config)
                if asyncio.iscoroutine(preprocess_result):
                    await preprocess_result
            except Exception as exc:
                logger.warning("Preprocessor failed in node %r: %s", self.name, exc)

        # ── 1. Assemble system prompt from blocks ──
        system_parts: list[str] = []
        if self.instructions:
            system_parts.append(self.instructions)

        # Auto-inject tool schemas into the prompt so the LLM knows exact
        # parameter names, types, and descriptions — without this, the LLM
        # has to guess from tool names alone, which causes wrong parameters.
        # This is a major tool-accuracy improvement.
        _has_tools = bool(self.tools or self.inject_tools)
        if _has_tools:
            _all_tools = list(self.tools)
            if self.inject_tools:
                _all_tools.extend(config.get("_engine_tools", []))
            if _all_tools:
                schema_lines = ["Available tools:"]
                seen_tool_names: set[str] = set()
                for t in _all_tools:
                    if t.name in seen_tool_names:
                        continue
                    seen_tool_names.add(t.name)
                    desc = getattr(t, "description", "") or ""
                    # Extract parameter info from schema
                    params = ""
                    args_schema = getattr(t, "args_schema", None)
                    if args_schema:
                        try:
                            schema = (
                                args_schema.model_json_schema()
                                if hasattr(args_schema, "model_json_schema")
                                else {}
                            )
                            props = schema.get("properties", {})
                            required = set(schema.get("required", []))
                            parts = []
                            for pname, pinfo in props.items():
                                ptype = pinfo.get("type", "any")
                                req = "*" if pname in required else ""
                                default = pinfo.get("default")
                                d = f"={default}" if default is not None else ""
                                parts.append(f"{pname}: {ptype}{d}{req}")
                            if parts:
                                params = f"({', '.join(parts)})"
                        except Exception as _schema_exc:
                            logger.debug(
                                "Schema extraction failed for tool %s: %s", t.name, _schema_exc
                            )
                            params = "(...)"  # Show generic hint
                    schema_lines.append(f"  - {t.name}{params}: {desc[:120]}")
                system_parts.append("\n".join(schema_lines))

        # Inject input_keys from state.context
        if self.input_keys:
            input_parts = []
            for key in self.input_keys:
                val = state.context.get(key)
                if val is not None:
                    input_parts.append(f"{key}: {val}")
            if input_parts:
                system_parts.append("Input data:\n" + "\n".join(input_parts))

        # Inherit output from a previous node
        if self.inherit_context_from:
            prev_output = state.context.get(f"{self.inherit_context_from}_output")
            if prev_output is not None:
                system_parts.append(f"Output from '{self.inherit_context_from}':\n{prev_output}")

        # Inject custom context layers
        for layer_name, _priority in self.context_layers.items():
            layer_val = state.context.get(layer_name)
            if layer_val is not None:
                system_parts.append(f"{layer_name}:\n{layer_val}")

        for block in self.blocks:
            try:
                rendered = block.render(None)
                if rendered:
                    system_parts.append(rendered)
                    result.blocks_used.append(getattr(block, "name", type(block).__name__))
            except Exception as exc:
                logger.debug("Block render failed in %s: %s", self.name, exc)
                result.blocks_dropped.append(getattr(block, "name", type(block).__name__))

        # Inject observations from state (controlled by include_observations flag)
        # Skip injection if this node has tools — the tool results are already
        # in the conversation as ToolMessages and injecting them again into the
        # system prompt causes duplication, token waste, and confuses the LLM.
        if self.include_observations and state.observations and not _has_tools:
            obs_lines = []
            for obs in state.observations[-5:]:
                tool_name = obs.get("tool", "unknown")
                tool_result = str(obs.get("result", ""))
                obs_lines.append(f"[{tool_name}] → {tool_result}")
            if obs_lines:
                system_parts.append("Recent tool results:\n" + "\n".join(obs_lines))

        # Inject plan from state (controlled by include_plan flag)
        if self.include_plan and state.plan:
            plan_lines = []
            for sg in state.plan:
                status = (
                    "✓" if sg in state.completed else ("→" if sg == state.active_subgoal else " ")
                )
                plan_lines.append(f"[{status}] {sg}")
            system_parts.append("Plan:\n" + "\n".join(plan_lines))

        # Inject reflections from state (controlled by include_reflections flag)
        if self.include_reflections and state.reflections:
            ref_lines = [
                f"Iter {r['iteration']}: {r['mistake']} → {r['correction']}"
                for r in state.reflections[-3:]
            ]
            system_parts.append("Past learnings:\n" + "\n".join(ref_lines))

        # Inject available transitions so the LLM can route dynamically.
        # Only do this when there is a genuine CHOICE — i.e. the node has
        # explicit transitions to two or more distinct targets. A node that
        # only has a single linear default_next has nothing to route, so
        # telling the model to "choose the next step" is noise that a weaker
        # model can latch onto and emit as its answer.
        if self.transitions:
            route_options = list(self.transitions.values())
            if self.default_next:
                route_options.append(self.default_next)
            # Deduplicate while preserving order; "__end__" is not a step the
            # model should be told to pick.
            seen: set[str] = set()
            unique_routes = []
            for r in route_options:
                if r not in seen and r != "__end__":
                    seen.add(r)
                    unique_routes.append(r)
            if len(unique_routes) > 1 and not self.tools:
                # Only show routing instructions for non-tool nodes with a
                # real branch (tool nodes route automatically on tool_calls).
                system_parts.append(
                    "Available next steps: " + ", ".join(unique_routes) + "\n"
                    "Set the 'route' field in your response to choose which step to take next."
                )

        # Apply strategy wrapping
        if self.strategy and hasattr(self.strategy, "wrap"):
            full_prompt = "\n\n".join(system_parts)
            wrapped = self.strategy.wrap(full_prompt, None)
            if wrapped is not None:
                system_parts = [wrapped]
                result.strategy_applied = type(self.strategy).__name__
            else:
                logger.warning(
                    "Strategy %r returned None — using unwrapped prompt",
                    type(self.strategy).__name__,
                )

        # Apply perspective
        if self.perspective and hasattr(self.perspective, "framing"):
            system_parts.insert(0, self.perspective.framing)
            result.perspective_applied = type(self.perspective).__name__

        # Build messages with system prompt.
        # On tool-loop re-entries (same node called again after tool execution),
        # reuse the cached system message and model binding from config to avoid
        # redundant string assembly, marker scanning, and tool resolution.
        _cache_key = f"_node_cache_{self.name}"
        _cached = config.get(_cache_key)

        if _cached is not None:
            # Fast path: reuse cached system message + model binding
            node_sys_msg = _cached["sys_msg"]
            model_to_use = _cached["model"]
            active_tools = _cached["tools"]
            # Build messages: insert our SystemMessage as on the first call.
            # (It is never in state.messages, so nothing there to replace;
            # replacing overwrote the input's second system message.)
            messages = list(state.messages)
            if messages and isinstance(messages[0], SystemMessage):
                messages.insert(1, node_sys_msg)
            else:
                messages.insert(0, node_sys_msg)
        else:
            # First call for this node — build system prompt and cache it
            system_text = "\n\n".join(system_parts)
            node_sys_msg = SystemMessage(content=system_text)

            messages = list(state.messages)
            if messages and isinstance(messages[0], SystemMessage):
                messages.insert(1, node_sys_msg)
            else:
                messages.insert(0, node_sys_msg)

            # ── 2. Resolve tools (runtime injection if flagged) ──
            active_tools = list(self.tools)
            if self.inject_tools:
                engine_tools = config.get("_engine_tools", [])
                existing_names = {t.name for t in active_tools}
                for et in engine_tools:
                    if et.name not in existing_names:
                        active_tools.append(et)

            if active_tools:
                model_to_use = model.bind_tools(active_tools)
            else:
                model_to_use = model

            # Apply structured output if schema set
            if self.output_schema and hasattr(model_to_use, "with_structured_output"):
                try:
                    model_to_use = model_to_use.with_structured_output(self.output_schema)
                except Exception:
                    pass

            # Cache for tool-loop re-entries
            config[_cache_key] = {
                "sys_msg": node_sys_msg,
                "model": model_to_use,
                "tools": active_tools,
            }

        # Structured output is already applied in the cached model_to_use

        # ── 2b. Context scoping ──
        # When the node is context-scoped, replace the full transcript with a
        # bounded working set: this node's system prompt (already carrying any
        # inherited/​injected distilled state), the original task, and this
        # node's own in-progress tool loop. The verbose intermediate messages
        # produced by *other* stages are dropped, so token usage does not grow
        # super-linearly across a multi-stage reasoning graph. ``"auto"`` resolves
        # to "full" (short loops) or "ledger" (deep loops) based on the state.
        _scope = self._effective_context_scope(state, config)
        if _scope in ("scoped", "ledger"):
            # Compaction (see promptise.engine.compaction): pinned input system
            # messages, a note on earlier conversation, the CURRENT question,
            # then for "ledger" the latest tool exchange verbatim plus a
            # deduplicated ledger of older results (long ones cut to an
            # excerpt); for "scoped" this node's own trailing tool loop.
            from .compaction import build_compacted_view

            messages = build_compacted_view(
                state.messages,
                question=state.turn_question,
                head=state.turn_input,
                settings=self._compaction_settings(config),
                scoped=_scope == "scoped",
            )
            if messages and isinstance(messages[0], SystemMessage):
                messages.insert(1, node_sys_msg)
            else:
                messages.insert(0, node_sys_msg)

        # ── 3. Call LLM ──
        llm_start = time.monotonic()
        try:
            response = await model_to_use.ainvoke(messages, config=config)
            result.llm_duration_ms = (time.monotonic() - llm_start) * 1000
        except Exception as exc:
            # Recorded on the result for hooks/history; the engine raises
            # GraphExecutionError (chained to this exception) if the run
            # ends here, so a provider failure never becomes a silent "answer".
            record_failure(result, exc)
            result.duration_ms = (time.monotonic() - start) * 1000
            return result

        # ── 4. Process response ──
        if isinstance(response, AIMessage):
            # AIMessage.content is str | list[...] in LangChain's typing; coerce to str.
            _content = response.content
            result.raw_output = _content if isinstance(_content, str) else str(_content or "")
            result.messages_added.append(response)
            state.messages.append(response)

            # Extract token usage if available
            usage = getattr(response, "usage_metadata", None)
            if usage:
                result.prompt_tokens = getattr(usage, "input_tokens", 0)
                result.completion_tokens = getattr(usage, "output_tokens", 0)
                result.total_tokens = result.prompt_tokens + result.completion_tokens

            # Handle tool calls
            tool_calls = getattr(response, "tool_calls", None) or []
            if tool_calls:
                # Reuse cached tool_map if available
                _tool_map_key = f"_tool_map_{self.name}"
                tool_map = config.get(_tool_map_key)
                if tool_map is None:
                    tool_map = {t.name: t for t in active_tools}
                    config[_tool_map_key] = tool_map
                tool_start = time.monotonic()
                hooks = config.get("_engine_hooks", [])

                async def _exec_one_tool(tc: dict) -> tuple[dict, str]:
                    """Execute a single tool call. Returns (tc_record, content)."""
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("args", {})

                    # Ledger mode: serve a previously-gathered fact from cache
                    # instead of re-executing the same (tool, args). Deep tasks
                    # otherwise re-fetch identical facts dozens of times.
                    if self._effective_context_scope(state, config) == "ledger":
                        # Latest result wins, as in the ledger.
                        for obs in reversed(state.observations):
                            if (
                                obs.get("tool") == tool_name
                                and obs.get("args") == tool_args
                                and obs.get("success")
                            ):
                                cached = str(obs.get("result", ""))
                                return (
                                    {
                                        "name": tool_name,
                                        "args": tool_args,
                                        "result": cached,
                                        "success": True,
                                        "cached": True,
                                    },
                                    cached,
                                )

                    # Pre-tool hooks
                    for hook in hooks:
                        if hasattr(hook, "pre_tool"):
                            tool_args = await hook.pre_tool(tool_name, tool_args, state)

                    tc_record: dict[str, Any] = {"name": tool_name, "args": tool_args}

                    if tool_name not in tool_map:
                        content = f"Error: Unknown tool '{tool_name}'"
                        tc_record["result"] = content
                        tc_record["success"] = False
                    else:
                        try:
                            tool_result = await tool_map[tool_name].ainvoke(
                                tool_args, config=config
                            )
                            content = str(tool_result) if tool_result is not None else ""
                            tc_record["result"] = content
                            tc_record["success"] = True
                        except Exception as exc:
                            content = f"Error: {type(exc).__name__}: {exc}"
                            tc_record["result"] = content
                            tc_record["success"] = False

                    # Post-tool hooks
                    for hook in hooks:
                        if hasattr(hook, "post_tool"):
                            content = await hook.post_tool(tool_name, content, tool_args, state)

                    return tc_record, content

                # Execute tools — parallel if 2+ independent calls,
                # sequential if only 1 (avoid asyncio.gather overhead)
                if len(tool_calls) >= 2:
                    gathered = await asyncio.gather(
                        *[_exec_one_tool(tc) for tc in tool_calls],
                        return_exceptions=True,
                    )
                    tool_results_ordered: list[tuple[dict[str, Any], Any]] = []
                    for i, res_or_exc in enumerate(gathered):
                        if isinstance(res_or_exc, BaseException):
                            tc = tool_calls[i]
                            tc_rec = {
                                "name": tc.get("name", ""),
                                "args": tc.get("args", {}),
                                "result": f"Error: {res_or_exc}",
                                "success": False,
                            }
                            tool_results_ordered.append((tc_rec, tc_rec["result"]))
                        else:
                            tool_results_ordered.append(res_or_exc)
                else:
                    tool_results_ordered = [await _exec_one_tool(tool_calls[0])]

                # Append results in original order (preserves tool_call_id alignment)
                for i, (tc_record, content) in enumerate(tool_results_ordered):
                    tc = tool_calls[i]
                    tool_id = tc.get("id", str(uuid4()))

                    if not tc_record.get("success", True):
                        result.tool_calls_failed += 1

                    tool_msg = ToolMessage(content=content, tool_call_id=tool_id)
                    state.messages.append(tool_msg)
                    result.messages_added.append(tool_msg)
                    result.tool_calls.append(tc_record)

                    state.add_observation(
                        tool_name=tc_record["name"],
                        result=content,
                        args=tc_record.get("args", {}),
                        success=tc_record.get("success", False),
                    )

                result.tool_duration_ms = (time.monotonic() - tool_start) * 1000

                # Signal that tools were called — engine will re-enter this node
                result.transition_reason = "tool_calls_present"

        elif isinstance(response, dict):
            # Structured output (from with_structured_output)
            result.output = response
            result.raw_output = json.dumps(response, default=str)
        else:
            # Pydantic model or other
            result.output = response
            result.raw_output = str(response)

        # ── 5. Parse structured output for transition ──
        if result.output and isinstance(result.output, dict):
            # Check for transition keys in the output
            for key, target in self.transitions.items():
                if key in result.output:
                    val = result.output[key]
                    if isinstance(val, bool) and val:
                        result.next_node = target
                        result.transition_reason = f"output.{key}={val}"
                        break
                    elif isinstance(val, str) and val == key:
                        result.next_node = target
                        result.transition_reason = f"output.{key}={val!r}"
                        break

        # ── 6. Run guards ──
        for guard in self.guards:
            try:
                if hasattr(guard, "check_output"):
                    check = guard.check_output(result.raw_output)
                    if asyncio.iscoroutine(check):
                        await check
                result.guards_passed.append(type(guard).__name__)
            except Exception as exc:
                result.guards_failed.append(f"{type(guard).__name__}: {exc}")

        # ── 7. Strategy parsing ──
        if self.strategy and hasattr(self.strategy, "parse") and result.raw_output:
            try:
                parsed = self.strategy.parse(result.raw_output, None)
                if parsed != result.raw_output:
                    result.output = parsed
            except Exception:
                pass

        # ── 8. Run postprocessor ──
        if self.postprocessor:
            try:
                post_result = self.postprocessor(result.output, state, config)
                if asyncio.iscoroutine(post_result):
                    post_result = await post_result
                if post_result is not None:
                    result.output = post_result
            except Exception as exc:
                logger.warning("Postprocessor failed in node %r: %s", self.name, exc)

        # ── 9. Write output to state.context ──
        if self.output_key and result.output is not None:
            state.context[self.output_key] = result.output
        # Always write to {name}_output for context inheritance
        if result.raw_output:
            state.context[f"{self.name}_output"] = result.raw_output
        elif result.output is not None:
            state.context[f"{self.name}_output"] = result.output

        result.duration_ms = (time.monotonic() - start) * 1000
        return result

    async def stream(self, state: GraphState, config: dict[str, Any]) -> AsyncIterator[NodeEvent]:
        """Stream LLM tokens and tool events."""
        yield NodeEvent(event="on_node_start", node_name=self.name)

        _model_opt = config.get("_engine_model")
        if _model_opt is None:
            yield NodeEvent(event="on_node_end", node_name=self.name, data={"error": "No model"})
            return
        model: BaseChatModel = _model_opt

        # Build messages (same as execute but simplified for streaming)
        system_parts = [self.instructions] if self.instructions else []
        for block in self.blocks:
            try:
                rendered = block.render(None)
                if rendered:
                    system_parts.append(rendered)
            except Exception:
                pass

        system_text = "\n\n".join(system_parts)
        messages = list(state.messages)
        if messages and isinstance(messages[0], SystemMessage):
            messages[0] = SystemMessage(content=f"{messages[0].content}\n\n{system_text}")
        else:
            messages.insert(0, SystemMessage(content=system_text))

        # Resolve tools (runtime injection if flagged) — same as execute()
        active_tools = list(self.tools)
        if self.inject_tools:
            engine_tools = config.get("_engine_tools", [])
            existing_names = {t.name for t in active_tools}
            for et in engine_tools:
                if et.name not in existing_names:
                    active_tools.append(et)

        model_to_use = model.bind_tools(active_tools) if active_tools else model

        # Stream LLM response
        full_content = ""
        tool_call_chunks: list[dict] = []
        run_id = str(uuid4())

        async for chunk in model_to_use.astream(
            messages, config=cast("RunnableConfig | None", config)
        ):
            if hasattr(chunk, "content") and chunk.content:
                _chunk_content = chunk.content
                full_content += (
                    _chunk_content if isinstance(_chunk_content, str) else str(_chunk_content)
                )
                yield NodeEvent(
                    event="on_chat_model_stream",
                    node_name=self.name,
                    data={"chunk": chunk, "run_id": run_id},
                )

            # Accumulate tool call chunks
            if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                for tc in chunk.tool_call_chunks:
                    # Merge partial tool calls
                    if tc.get("index") is not None:
                        idx = tc["index"]
                        while len(tool_call_chunks) <= idx:
                            tool_call_chunks.append({"name": "", "args": "", "id": ""})
                        entry = tool_call_chunks[idx]
                        entry["name"] += tc.get("name", "") or ""
                        entry["args"] += tc.get("args", "") or ""
                        entry["id"] = tc.get("id") or entry.get("id", "")

        # Build final message
        parsed_tool_calls = []
        for tc in tool_call_chunks:
            try:
                args = json.loads(tc["args"]) if tc["args"] else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            parsed_tool_calls.append(
                {
                    "name": tc["name"],
                    "args": args,
                    "id": tc.get("id", str(uuid4())),
                }
            )

        response = AIMessage(content=full_content, tool_calls=parsed_tool_calls)
        state.messages.append(response)

        # Execute tool calls with events
        if parsed_tool_calls:
            tool_map = {t.name: t for t in active_tools}
            for tc in parsed_tool_calls:
                tc_run_id = str(uuid4())
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc.get("id", "")

                yield NodeEvent(
                    event="on_tool_start",
                    node_name=tool_name,
                    data={"input": tool_args, "run_id": tc_run_id},
                )

                try:
                    if tool_name in tool_map:
                        result = await tool_map[tool_name].ainvoke(
                            tool_args, config=cast("RunnableConfig | None", config)
                        )
                        content = str(result) if result is not None else ""
                    else:
                        content = f"Error: Unknown tool '{tool_name}'"

                    yield NodeEvent(
                        event="on_tool_end",
                        node_name=tool_name,
                        data={"output": content, "run_id": tc_run_id},
                    )
                except Exception as exc:
                    content = f"Error: {type(exc).__name__}: {exc}"
                    yield NodeEvent(
                        event="on_tool_error",
                        node_name=tool_name,
                        data={"error": str(exc), "run_id": tc_run_id},
                    )

                state.messages.append(ToolMessage(content=content, tool_call_id=tool_id))

        yield NodeEvent(event="on_node_end", node_name=self.name)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> PromptNode:
        """Create a PromptNode from a configuration dict.

        Handles all PromptNode parameters including data flow,
        preprocessing, and context control.
        """
        return cls(
            name=config.get("name", "unnamed"),
            instructions=config.get("instructions", ""),
            description=config.get("description", ""),
            blocks=config.get("blocks"),
            strategy=config.get("strategy"),
            perspective=config.get("perspective"),
            tools=config.get("tools"),
            tool_choice=config.get("tool_choice", "auto"),
            inject_tools=config.get("inject_tools", False),
            model_override=config.get("model_override"),
            output_schema=config.get("output_schema"),
            guards=config.get("guards"),
            context_layers=config.get("context_layers"),
            max_tokens=config.get("max_tokens", 4096),
            temperature=config.get("temperature", 0.0),
            input_keys=config.get("input_keys"),
            output_key=config.get("output_key"),
            inherit_context_from=config.get("inherit_context_from"),
            preprocessor=config.get("preprocessor"),
            postprocessor=config.get("postprocessor"),
            include_observations=config.get("include_observations", True),
            include_plan=config.get("include_plan", True),
            include_reflections=config.get("include_reflections", True),
            transitions=config.get("transitions"),
            default_next=config.get("default_next"),
            max_iterations=config.get("max_iterations", 10),
            metadata=config.get("metadata"),
        )


# ---------------------------------------------------------------------------
# ToolNode — explicit tool execution
# ---------------------------------------------------------------------------


class ToolNode(BaseNode):
    """Execute a specific tool or let the engine pick from available tools.

    Unlike PromptNode (where tools are called by the LLM), ToolNode
    executes tools directly based on state context.  Useful when tool
    selection is deterministic rather than LLM-driven.

    Args:
        tools: Available tools for this node.
        validate_inputs: Validate tool arguments against schema.
        deduplicate: Block identical (tool, args) calls.
        max_result_chars: Cap tool result size.
        tool_selector: Optional callable that selects which tool
            to call based on state.  If ``None``, uses the first
            tool call from the most recent AI message.
    """

    def __init__(
        self,
        name: str,
        *,
        tools: list[BaseTool] | None = None,
        validate_inputs: bool = True,
        deduplicate: bool = True,
        max_result_chars: int = 4000,
        tool_selector: Callable[[GraphState], tuple[str, dict]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.tools = list(tools) if tools else []
        self.validate_inputs = validate_inputs
        self.deduplicate = deduplicate
        self.max_result_chars = max_result_chars
        self.tool_selector = tool_selector
        self._called_fingerprints: set[str] = set()

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Execute tools from state or selector."""
        start = time.monotonic()
        result = NodeResult(node_name=self.name, node_type="tool", iteration=state.iteration)
        tool_map = {t.name: t for t in self.tools}

        # Determine what to call
        calls: list[tuple[str, dict[str, Any], str]]
        if self.tool_selector:
            tool_name, tool_args = self.tool_selector(state)
            calls = [(tool_name, tool_args, str(uuid4()))]
        else:
            # Extract from the last AI message's tool_calls
            calls = []
            for msg in reversed(state.messages):
                if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                    for tc in msg.tool_calls:
                        calls.append(
                            (tc["name"], tc.get("args", {}) or {}, tc.get("id") or str(uuid4()))
                        )
                    break

        for tool_name, tool_args, tool_id in calls:
            tc_record: dict[str, Any] = {"name": tool_name, "args": tool_args}

            # Dedup check
            if self.deduplicate:
                fp = f"{tool_name}::{json.dumps(tool_args, sort_keys=True)}"
                if fp in self._called_fingerprints:
                    tc_record["result"] = "DUPLICATE: identical call already made"
                    tc_record["success"] = False
                    result.tool_calls_deduplicated += 1
                    result.tool_calls.append(tc_record)
                    continue
                self._called_fingerprints.add(fp)

            if tool_name not in tool_map:
                content = f"Error: Unknown tool '{tool_name}'"
                tc_record["result"] = content
                tc_record["success"] = False
            else:
                try:
                    tool_result = await tool_map[tool_name].ainvoke(
                        tool_args, config=cast("RunnableConfig | None", config)
                    )
                    content = str(tool_result)[: self.max_result_chars] if tool_result else ""
                    tc_record["result"] = content
                    tc_record["success"] = True
                except Exception as exc:
                    content = f"Error: {type(exc).__name__}: {exc}"
                    tc_record["result"] = content
                    tc_record["success"] = False
                    result.tool_calls_failed += 1

            state.messages.append(ToolMessage(content=content, tool_call_id=tool_id))
            result.messages_added.append(ToolMessage(content=content, tool_call_id=tool_id))
            result.tool_calls.append(tc_record)

            state.add_observation(
                tool_name=tool_name,
                result=content[:500],
                args=tool_args,
                success=tc_record.get("success", False),
            )

        result.duration_ms = (time.monotonic() - start) * 1000
        return result


# ---------------------------------------------------------------------------
# RouterNode — LLM-based routing
# ---------------------------------------------------------------------------


class RouterNode(BaseNode):
    """Lightweight LLM call that decides which path to take.

    No tool calling.  The LLM sees the current state (via context
    blocks) and picks from a list of named routes.

    Args:
        routes: Mapping of route names to next-node names.
        context_blocks: Blocks to render for the LLM's context.
        model_override: Optional different model for routing (e.g.,
            a smaller/faster model for routing decisions).
    """

    def __init__(
        self,
        name: str,
        *,
        routes: dict[str, str] | None = None,
        context_blocks: list[Any] | None = None,
        model_override: BaseChatModel | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.routes = dict(routes) if routes else {}
        self.context_blocks = list(context_blocks) if context_blocks else []
        self.model_override = model_override

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Ask the LLM to choose a route."""
        start = time.monotonic()
        result = NodeResult(node_name=self.name, node_type="router", iteration=state.iteration)

        model = self.model_override or config.get("_engine_model")
        if model is None:
            result.error = "No model available"
            return result

        # Build routing prompt
        route_names = list(self.routes.keys())
        context_parts = [self.instructions] if self.instructions else []
        for block in self.context_blocks:
            try:
                rendered = block.render(None)
                if rendered:
                    context_parts.append(rendered)
            except Exception:
                pass

        context_parts.append(
            f"Choose ONE of these routes: {route_names}\n"
            f"Respond with ONLY the route name, nothing else."
        )

        messages = list(state.messages)
        messages.append(SystemMessage(content="\n\n".join(context_parts)))

        try:
            response = await model.ainvoke(messages, config=cast("RunnableConfig | None", config))
            result.llm_duration_ms = (time.monotonic() - start) * 1000

            _resp_content = getattr(response, "content", None)
            _resp_str = _resp_content if isinstance(_resp_content, str) else str(response)
            raw = (
                _resp_str.strip().lower()
                if hasattr(response, "content")
                else str(response).strip().lower()
            )
            result.raw_output = raw

            # Find matching route
            for route_name in route_names:
                if route_name.lower() in raw:
                    result.next_node = self.routes[route_name]
                    result.transition_reason = f"LLM chose route '{route_name}'"
                    break

            if result.next_node is None:
                if self.default_next:
                    result.next_node = self.default_next
                    result.transition_reason = "No route matched, using default"
                else:
                    result.error = (
                        f"No route matched from {list(self.routes.keys())} "
                        f"(LLM output: {raw[:100]!r}) and no default_next configured"
                    )

        except Exception as exc:
            record_failure(result, exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        return result


# ---------------------------------------------------------------------------
# GuardNode — validate and gate
# ---------------------------------------------------------------------------


class GuardNode(BaseNode):
    """Validate state against guards and route based on pass/fail.

    No LLM call.  Runs guards against the current state and routes
    to ``on_pass`` or ``on_fail`` nodes.

    Args:
        guards: List of guard instances to run.
        target_key: Key in ``state.context`` to validate.  If ``None``,
            validates the last message content.
        on_pass: Node name if all guards pass.
        on_fail: Node name if any guard fails.
    """

    def __init__(
        self,
        name: str,
        *,
        guards: list[Any] | None = None,
        target_key: str | None = None,
        on_pass: str = "__end__",
        on_fail: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.guards = list(guards) if guards else []
        self.target_key = target_key
        self.on_pass = on_pass
        self.on_fail = on_fail

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Run guards and route accordingly."""
        start = time.monotonic()
        result = NodeResult(node_name=self.name, node_type="guard", iteration=state.iteration)

        # Get the value to validate
        if self.target_key:
            target = state.context.get(self.target_key, "")
        else:
            # Last message content
            target = ""
            for msg in reversed(state.messages):
                if hasattr(msg, "content") and msg.content:
                    target = msg.content
                    break

        all_passed = True
        for guard in self.guards:
            try:
                if hasattr(guard, "check_output"):
                    await guard.check_output(target)
                elif callable(guard):
                    guard_result = guard(target)
                    if asyncio.iscoroutine(guard_result):
                        guard_result = await guard_result
                    if not guard_result:
                        raise ValueError("Guard returned False")
                result.guards_passed.append(
                    type(guard).__name__ if not callable(guard) else str(guard)
                )
            except Exception as exc:
                all_passed = False
                result.guards_failed.append(f"{type(guard).__name__}: {exc}")

        if all_passed:
            result.next_node = self.on_pass
            result.transition_reason = f"All {len(self.guards)} guards passed"
        else:
            result.next_node = self.on_fail or self.default_next
            result.transition_reason = f"{len(result.guards_failed)} guards failed"

        result.duration_ms = (time.monotonic() - start) * 1000
        return result


# ---------------------------------------------------------------------------
# ParallelNode — concurrent execution
# ---------------------------------------------------------------------------


class ParallelNode(BaseNode):
    """Run multiple child nodes concurrently and merge results.

    Args:
        nodes: Child nodes to execute in parallel.
        merge_strategy: How to merge results.
            ``"concatenate"`` — append all observations.
            ``"dict"`` — return ``{node_name: output}`` dict.
            ``"custom"`` — use ``merge_fn``.
        merge_fn: Custom merge function (when merge_strategy is
            ``"custom"``).
    """

    def __init__(
        self,
        name: str,
        *,
        nodes: list[BaseNode] | None = None,
        merge_strategy: str = "concatenate",
        merge_fn: Callable | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.child_nodes = list(nodes) if nodes else []
        self.merge_strategy = merge_strategy
        self.merge_fn = merge_fn

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Execute all child nodes concurrently."""
        start = time.monotonic()
        result = NodeResult(node_name=self.name, node_type="parallel", iteration=state.iteration)

        async def run_child(child_node: BaseNode) -> NodeResult:
            # Each child gets a shallow copy of state context
            child_state = GraphState(
                messages=list(state.messages),
                context=dict(state.context),
                current_node=child_node.name,
                observations=list(state.observations),
            )
            return await child_node.execute(child_state, config)

        child_results = await asyncio.gather(
            *[run_child(n) for n in self.child_nodes],
            return_exceptions=True,
        )

        # Merge results
        outputs: dict[str, Any] = {}
        errors: list[str] = []
        for child, child_result in zip(self.child_nodes, child_results, strict=False):
            if isinstance(child_result, BaseException):
                errors.append(f"{child.name}: {child_result}")
                continue

            outputs[child.name] = child_result.output
            result.tool_calls.extend(child_result.tool_calls)
            result.total_tokens += child_result.total_tokens

            # Merge observations into parent state
            for tc in child_result.tool_calls:
                if tc.get("success"):
                    state.add_observation(
                        tool_name=tc.get("name", ""),
                        result=str(tc.get("result", "")),
                        args=tc.get("args", {}),
                        success=True,
                    )

        if errors:
            result.error = "; ".join(errors)

        if self.merge_strategy == "dict":
            result.output = outputs
        elif self.merge_strategy == "custom" and self.merge_fn:
            result.output = self.merge_fn(outputs)
        else:
            result.output = outputs

        result.duration_ms = (time.monotonic() - start) * 1000
        return result


# ---------------------------------------------------------------------------
# LoopNode — repeat a subgraph
# ---------------------------------------------------------------------------


class LoopNode(BaseNode):
    """Repeat a subgraph until a condition is met.

    Args:
        body_node: The node to repeat (can be a SubgraphNode).
        condition: Callable that receives state and returns ``True``
            when the loop should stop.
        max_loop_iterations: Maximum iterations before forcing exit.
    """

    def __init__(
        self,
        name: str,
        *,
        body_node: BaseNode | None = None,
        condition: Callable[[GraphState], bool] | None = None,
        max_loop_iterations: int = 5,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.body_node = body_node
        self.condition = condition
        self.max_loop_iterations = max_loop_iterations

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Execute body node in a loop until condition met."""
        start = time.monotonic()
        result = NodeResult(node_name=self.name, node_type="loop", iteration=state.iteration)

        if self.body_node is None:
            result.error = "No body_node configured"
            result.duration_ms = (time.monotonic() - start) * 1000
            return result

        for i in range(self.max_loop_iterations):
            # Check exit condition
            if self.condition and self.condition(state):
                result.transition_reason = f"Condition met after {i} iterations"
                break

            child_result = await self.body_node.execute(state, config)
            result.tool_calls.extend(child_result.tool_calls)
            result.total_tokens += child_result.total_tokens

            if child_result.error:
                result.error = f"Loop iteration {i}: {child_result.error}"
                child_cause = failure_cause(child_result)
                if child_cause is not None:
                    setattr(result, _FAILURE_CAUSE_ATTR, child_cause)
                break
        else:
            result.transition_reason = f"Max loop iterations ({self.max_loop_iterations}) reached"

        result.output = state.context.get("loop_result")
        result.duration_ms = (time.monotonic() - start) * 1000
        return result


# ---------------------------------------------------------------------------
# HumanNode — pause for human input
# ---------------------------------------------------------------------------


class HumanNode(BaseNode):
    """Pause execution and wait for human input.

    Integrates with the existing ``ApprovalPolicy`` system.

    Args:
        prompt_template: Template shown to the human.
        timeout: Seconds to wait before applying ``on_timeout``.
        on_approve: Next node if human approves.
        on_deny: Next node if human denies.
        on_timeout: Next node if timeout expires.
    """

    def __init__(
        self,
        name: str,
        *,
        prompt_template: str = "Approve this action?",
        timeout: float = 300.0,
        on_approve: str = "__end__",
        on_deny: str | None = None,
        on_timeout: str = "__end__",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.prompt_template = prompt_template
        self.timeout = timeout
        self.on_approve = on_approve
        self.on_deny = on_deny
        self.on_timeout = on_timeout

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Pause for human input."""
        start = time.monotonic()
        result = NodeResult(node_name=self.name, node_type="human", iteration=state.iteration)

        # Check if there's a human handler in config
        handler = config.get("_human_handler")
        if handler is None:
            # No handler — auto-approve
            result.next_node = self.on_approve
            result.transition_reason = "No human handler configured, auto-approved"
            result.duration_ms = (time.monotonic() - start) * 1000
            return result

        try:
            prompt = self.prompt_template.format(**state.context)
            decision = await asyncio.wait_for(handler(prompt, state), timeout=self.timeout)

            if decision:
                result.next_node = self.on_approve
                result.transition_reason = "Human approved"
            else:
                result.next_node = self.on_deny or self.default_next
                result.transition_reason = "Human denied"

        except asyncio.TimeoutError:
            result.next_node = self.on_timeout
            result.transition_reason = f"Timeout after {self.timeout}s"

        except Exception as exc:
            record_failure(result, exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        return result


# ---------------------------------------------------------------------------
# TransformNode — pure data transformation
# ---------------------------------------------------------------------------


class TransformNode(BaseNode):
    """Transform state data without calling the LLM.

    Useful for formatting, aggregation, data extraction, or
    preparing state for the next node.

    Args:
        transform: Callable that receives state and returns a value
            to store in ``state.context[output_key]``.
        output_key: Key in ``state.context`` to store the result.
    """

    def __init__(
        self,
        name: str,
        *,
        transform: Callable[[GraphState], Any] | None = None,
        output_key: str = "transform_result",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.transform = transform
        self.output_key = output_key

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Execute the transform function."""
        start = time.monotonic()
        result = NodeResult(node_name=self.name, node_type="transform", iteration=state.iteration)

        if self.transform is None:
            result.error = "No transform function configured"
            result.duration_ms = (time.monotonic() - start) * 1000
            return result

        try:
            output = self.transform(state)
            if asyncio.iscoroutine(output):
                output = await output
            state.context[self.output_key] = output
            result.output = output
        except Exception as exc:
            record_failure(result, exc)

        result.duration_ms = (time.monotonic() - start) * 1000
        return result


# ---------------------------------------------------------------------------
# SubgraphNode — embed another graph
# ---------------------------------------------------------------------------


class SubgraphNode(BaseNode):
    """Embed a complete sub-graph as a single node.

    The subgraph runs to completion, then the parent graph continues.
    State can be shared (``inherit_state=True``) or isolated.

    Args:
        subgraph: A ``PromptGraph`` instance to run.
        inherit_state: If ``True``, the subgraph shares the parent's
            state (messages, context, observations).  If ``False``,
            the subgraph gets a fresh state and only its final
            messages are appended to the parent.
    """

    def __init__(
        self,
        name: str,
        *,
        subgraph: Any = None,  # PromptGraph — forward ref
        inherit_state: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.subgraph = subgraph
        self.inherit_state = inherit_state

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Run the subgraph to completion."""
        start = time.monotonic()
        result = NodeResult(node_name=self.name, node_type="subgraph", iteration=state.iteration)

        if self.subgraph is None:
            result.error = "No subgraph configured"
            result.duration_ms = (time.monotonic() - start) * 1000
            return result

        # Import here to avoid circular imports
        from .execution import PromptGraphEngine

        model = config.get("_engine_model")
        if model is None:
            result.error = "No _engine_model in config"
            result.duration_ms = (time.monotonic() - start) * 1000
            return result
        engine = PromptGraphEngine(
            graph=self.subgraph,
            model=model,
            max_iterations=config.get("_max_iterations", 25),
        )

        if self.inherit_state:
            sub_input = {"messages": state.messages}
        else:
            sub_input = {"messages": []}

        try:
            sub_output = await engine.ainvoke(sub_input, config=config)

            if not self.inherit_state:
                # Append subgraph's messages to parent state
                for msg in sub_output.get("messages", []):
                    state.messages.append(msg)
                    result.messages_added.append(msg)

            result.output = sub_output
            _last_report = getattr(engine, "_last_report", None)
            result.total_tokens = _last_report.total_tokens if _last_report is not None else 0

        except Exception as exc:
            record_failure(result, exc, f"Subgraph error: {type(exc).__name__}: {exc}")

        result.duration_ms = (time.monotonic() - start) * 1000
        return result


# ---------------------------------------------------------------------------
# AutonomousNode — the agent builds its own reasoning path at runtime
# ---------------------------------------------------------------------------


class AutonomousNode(BaseNode):
    """Meta-node: the LLM receives a pool of configured nodes and
    dynamically composes its own reasoning path at runtime.

    Instead of following a pre-defined graph, the agent:

    1. Sees all available nodes as a menu of capabilities
    2. Decides which node to execute next based on the task
    3. Executes that node (with its tools, blocks, guards)
    4. Evaluates the result
    5. Decides the next node or finishes

    The developer provides the LEGO blocks.
    The agent builds the model.

    Usage::

        autonomous = AutonomousNode(
            "agent",
            node_pool=[
                web_researcher("search", tools=search_tools),
                data_analyst("analyze", tools=db_tools),
                fact_checker("verify"),
                summarizer("conclude"),
                planner("plan"),
            ],
            planner_instructions="You are a research assistant. "
                "Use the available steps to answer the user's question thoroughly.",
        )

        graph = PromptGraph("autonomous")
        graph.add_node(autonomous)
        graph.set_entry("agent")

    Args:
        node_pool: Available nodes the agent can choose from.
            Each node is fully configured with its own tools,
            blocks, guards, and instructions.
        planner_instructions: How the agent should approach
            selecting and sequencing nodes.
        allow_repeat: Whether the agent can re-use the same
            node multiple times.
        max_steps: Maximum nodes the agent can execute per run.
    """

    def __init__(
        self,
        name: str = "autonomous",
        *,
        node_pool: list[BaseNode] | None = None,
        planner_instructions: str = "",
        allow_repeat: bool = True,
        max_steps: int = 15,
        entry_node: str | None = None,
        terminal_nodes: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.node_pool: dict[str, BaseNode] = {n.name: n for n in (node_pool or [])}
        self.planner_instructions = planner_instructions
        self.allow_repeat = allow_repeat
        self.max_steps = max_steps
        self.entry_node = entry_node
        self.terminal_nodes = list(terminal_nodes) if terminal_nodes else []

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Autonomously select and execute nodes from the pool."""
        start = time.monotonic()
        result = NodeResult(node_name=self.name, node_type="autonomous", iteration=state.iteration)

        model: BaseChatModel | None = config.get("_engine_model")
        if model is None:
            result.error = "No model available"
            return result

        if not self.node_pool:
            result.error = "No nodes in the pool"
            return result

        # Build the capability catalog for the LLM
        catalog_lines = ["Available reasoning steps:"]
        for nname, nobj in self.node_pool.items():
            ntype = type(nobj).__name__
            desc = nobj.instructions or nobj.description or "(no description)"
            has_tools = bool(getattr(nobj, "tools", None) or getattr(nobj, "inject_tools", False))
            tools_info = ""
            if has_tools:
                tool_names = [t.name for t in getattr(nobj, "tools", []) or []]
                if getattr(nobj, "inject_tools", False):
                    tools_info = " [has tools: runtime-injected]"
                elif tool_names:
                    tools_info = f" [tools: {', '.join(tool_names)}]"
            terminal_mark = " [TERMINAL — can finish here]" if nname in self.terminal_nodes else ""
            catalog_lines.append(f"  - {nname} ({ntype}): {desc}{tools_info}{terminal_mark}")

        catalog = "\n".join(catalog_lines)
        used_nodes: list[str] = []

        # If entry_node is set, execute it first
        if self.entry_node and self.entry_node in self.node_pool:
            entry = self.node_pool[self.entry_node]
            logger.info("Autonomous: starting with entry node %r", self.entry_node)
            used_nodes.append(self.entry_node)
            try:
                child_result = await entry.execute(state, config)
                result.tool_calls.extend(child_result.tool_calls)
                result.total_tokens += child_result.total_tokens
                # If entry is terminal and completed, we're done
                if self.entry_node in self.terminal_nodes and not child_result.error:
                    result.output = {"steps": used_nodes, "total": len(used_nodes)}
                    result.duration_ms = (time.monotonic() - start) * 1000
                    return result
            except Exception as exc:
                state.messages.append(SystemMessage(content=f"Entry node failed: {exc}"))

        for step_idx in range(self.max_steps):
            # Ask the LLM which node to execute next
            routing_prompt = (
                f"{self.planner_instructions}\n\n"
                f"{catalog}\n\n"
                f"Steps completed: {used_nodes if used_nodes else '(none)'}\n"
                f"Step {step_idx + 1}/{self.max_steps}\n\n"
                "Respond with ONLY a JSON object:\n"
                '{"next_node": "<node_name>", "reason": "<brief reason>"}\n'
                "To finish and give final answer: "
                '{"next_node": "__done__", "reason": "<summary>"}'
            )

            state.messages.append(SystemMessage(content=routing_prompt))

            try:
                response = await model.ainvoke(
                    state.messages, config=cast("RunnableConfig | None", config)
                )
                state.messages.append(response)
                _raw_content = response.content if hasattr(response, "content") else str(response)
                raw = _raw_content if isinstance(_raw_content, str) else str(_raw_content or "")

                # Extract JSON from response (handles nested braces)
                choice = self._extract_json(raw)
                if choice is None:
                    choice = {"next_node": "__done__", "reason": "unparseable"}

            except Exception as exc:
                record_failure(result, exc, f"Routing failed at step {step_idx}: {exc}")
                break

            chosen_name = choice.get("next_node", "__done__")
            reason = choice.get("reason", "")

            # Finished?
            if chosen_name == "__done__":
                result.transition_reason = f"Agent finished: {reason}"
                break

            # Validate
            if chosen_name not in self.node_pool:
                state.messages.append(
                    SystemMessage(
                        content=(
                            f"'{chosen_name}' is not available. "
                            f"Choose from: {list(self.node_pool.keys())}"
                        )
                    )
                )
                continue

            if not self.allow_repeat and chosen_name in used_nodes:
                state.messages.append(
                    SystemMessage(content=f"'{chosen_name}' already used. Choose another.")
                )
                continue

            # Execute the chosen node
            logger.info(
                "Autonomous step %d: executing %r (%s)",
                step_idx + 1,
                chosen_name,
                reason,
            )
            chosen_node = self.node_pool[chosen_name]
            used_nodes.append(chosen_name)

            try:
                child_result = await chosen_node.execute(state, config)
                result.tool_calls.extend(child_result.tool_calls)
                result.total_tokens += child_result.total_tokens

                # If this was a terminal node and it succeeded, we're done
                if chosen_name in self.terminal_nodes and not child_result.error:
                    result.transition_reason = f"Terminal node '{chosen_name}' completed"
                    break
            except Exception as exc:
                state.messages.append(SystemMessage(content=f"Step '{chosen_name}' failed: {exc}"))

        result.output = {"steps": used_nodes, "total": len(used_nodes)}
        result.duration_ms = (time.monotonic() - start) * 1000
        return result

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Extract the first valid JSON object from text, handling nested braces."""
        depth = 0
        start = None
        for i, char in enumerate(text):
            if char == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        return json.loads(text[start : i + 1])
                    except (json.JSONDecodeError, ValueError):
                        start = None  # Try next JSON object
        return None
