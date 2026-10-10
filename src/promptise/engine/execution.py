"""PromptGraphEngine — adaptive graph traversal with runtime self-modification.

The engine traverses a ``PromptGraph``, executing nodes and following
edges.  It provides composable reasoning patterns, parallel tool
execution, and production safety features.

The engine exposes the standard agent interface:
- ``ainvoke(input, config=config)`` → ``{"messages": [...]}``
- ``astream_events(input, config=config, version="v2")`` → event stream

A run that ends on a failed node — a model call the provider rejected, a
CRITICAL node error, a RETRYABLE node out of attempts — raises
:class:`GraphExecutionError` (chained to the provider exception) instead of
returning the conversation as if it were an answer.  Node-level errors stay
recorded in ``GraphState.node_history`` / ``ExecutionReport`` for hooks and
observability; a graph that routes a failure to a handler node that
succeeds is not a failed run.

Key features:
- **Adaptive traversal**: nodes can modify the graph at runtime
- **Per-invocation isolation**: each ``ainvoke()`` works on a graph copy
- **Hook system**: pre_node, post_node for developer interception
- **Rich observability**: per-node NodeResult with full execution trace
- **Tool auto-execution**: PromptNodes with tools loop until LLM stops
- **Safety**: iteration limits, stuck-node recovery, cycle detection
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .base import BaseNode
from .graph import PromptGraph
from .nodes import failure_cause, record_failure
from .state import ExecutionReport, GraphMutation, GraphState, NodeEvent, NodeFlag, NodeResult

logger = logging.getLogger("promptise.engine")


class GraphExecutionError(RuntimeError):
    """A graph run ended on a node failure the graph did not recover from.

    Raised by :meth:`PromptGraphEngine.ainvoke` and, after its events, by
    :meth:`PromptGraphEngine.astream_events`.  The provider exception (a
    rejected API key, a network failure…) is chained as ``__cause__`` when
    the node recorded one.  ``PromptiseAgent.ainvoke()`` lets it propagate,
    so a broken credential is an exception at the call site, never an
    "answer" that echoes the question.

    Attributes:
        graph_name: Name of the graph that failed.
        node_name: Node whose failure ended the run.
        error: The node's recorded error text.
        report: The :class:`ExecutionReport` of the failed run.
    """

    def __init__(
        self,
        graph_name: str,
        node_name: str,
        error: str,
        *,
        report: ExecutionReport | None = None,
    ) -> None:
        super().__init__(f"Graph {graph_name!r} failed at node {node_name!r}: {error}")
        self.graph_name = graph_name
        self.node_name = node_name
        self.error = error
        self.report = report


class PromptGraphEngine:
    """Adaptive graph traversal engine.

    Traverses a ``PromptGraph`` by executing nodes and following edges.
    The graph is copied per invocation so concurrent calls are safe
    and runtime mutations don't affect the original graph.

    Args:
        graph: The graph to traverse.
        model: LangChain ``BaseChatModel`` for LLM calls.
        max_iterations: Maximum total node executions per run.
        max_node_iterations: Maximum times a single node can execute
            (prevents infinite tool-calling loops).
        hooks: List of hook instances for interception.
        allow_self_modification: Allow the LLM to modify the graph
            via structured output ``_graph_action`` fields.
        max_mutations_per_run: Cap on graph mutations per run.
        tools: Tools for nodes with ``inject_tools=True`` (``build_agent``
            passes the MCP tools it discovered).  Tools set on the graph's
            nodes are added to them.
        compaction: :class:`~promptise.engine.compaction.ContextCompaction`
            settings for nodes that compact (``context_scope="auto"`` /
            ``"ledger"``) and set none of their own.
    """

    def __init__(
        self,
        graph: PromptGraph,
        model: BaseChatModel,
        *,
        max_iterations: int = 50,
        max_node_iterations: int = 25,
        hooks: list[Any] | None = None,
        allow_self_modification: bool = True,
        max_mutations_per_run: int = 10,
        lightweight_model: BaseChatModel | None = None,
        tools: list[Any] | None = None,
        compaction: Any | None = None,
    ) -> None:
        self.graph = graph
        self.tools = list(tools) if tools else []
        self.compaction = compaction
        self.model = model
        self.max_iterations = max_iterations
        self.max_node_iterations = max_node_iterations
        self.hooks = list(hooks) if hooks else []
        self.allow_self_modification = allow_self_modification
        self.max_mutations_per_run = max_mutations_per_run
        self.lightweight_model = lightweight_model
        self._last_report: ExecutionReport | None = None

    # ──────────────────────────────────────────────────────────────────
    # ainvoke — synchronous (full) execution
    # ──────────────────────────────────────────────────────────────────

    async def ainvoke(
        self,
        input: dict[str, Any],
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run the graph to completion.

        Returns ``{"messages": [...]}`` matching the LangGraph contract.
        """
        config = dict(config) if config else {}
        config["_engine_model"] = self.model
        config["_max_iterations"] = self.max_iterations
        config["_engine_hooks"] = self.hooks
        if self.compaction is not None and "_engine_compaction" not in config:
            config["_engine_compaction"] = self.compaction

        # Collect all tools from all nodes for runtime injection
        if "_engine_tools" not in config:
            all_tools: list = list(self.tools)
            seen_names: set[str] = {t.name for t in all_tools}
            for node_obj in self.graph.nodes.values():
                for tool in getattr(node_obj, "tools", []) or []:
                    if tool.name not in seen_names:
                        all_tools.append(tool)
                        seen_names.add(tool.name)
            config["_engine_tools"] = all_tools

        run_start = time.monotonic()
        mutations_count = 0

        # Work on a COPY — original graph is never mutated
        live_graph = self.graph.copy()

        if not live_graph.entry:
            raise ValueError(
                f"Graph {live_graph.name!r} has no entry node. "
                f"Call graph.set_entry(name) before running."
            )

        state = GraphState(
            messages=list(input.get("messages", [])),
            current_node=live_graph.entry,
            graph=live_graph,
        )

        # The failure the run ended on, if any. Only the *last* executed node
        # decides: a failure the graph routed to a handler node that then
        # succeeded is recovery, not failure.
        last_failure: NodeResult | None = None

        while state.current_node != "__end__":
            try:
                node = live_graph.get_node(state.current_node)
            except (KeyError, ValueError):
                node = None
            if node is None:
                logger.error(
                    "Node %r not found in graph %r — ending execution",
                    state.current_node,
                    live_graph.name,
                )
                break
            state.visited.append(state.current_node)

            # ── Pre-node hooks ──
            for hook in self.hooks:
                if hasattr(hook, "pre_node"):
                    try:
                        state = await hook.pre_node(node, state)
                    except Exception as exc:
                        logger.warning("pre_node hook %r failed: %s", type(hook).__name__, exc)

            # ── Pre-execute flag processing ──
            node_start = time.monotonic()
            skip_result = await self._pre_execute_flags(node, state, config)

            if skip_result is not None:
                # Node was skipped (SKIP_ON_ERROR or CACHEABLE cache hit)
                result = skip_result
            else:
                # ── Execute node (with retry if RETRYABLE) ──
                if node.has_flag(NodeFlag.RETRYABLE):
                    result = await self._execute_with_retry(node, state, config)
                else:
                    try:
                        result = await node.execute(state, config)
                    except Exception as exc:
                        result = NodeResult(
                            node_name=node.name,
                            node_type=type(node).__name__.lower(),
                            iteration=state.iteration,
                        )
                        record_failure(result, exc)
                        logger.error("Node %r execution failed: %s", node.name, exc)

                # ── Post-execute flag processing ──
                await self._post_execute_flags(node, result, state, config)

            result.duration_ms = (time.monotonic() - node_start) * 1000

            # Did the node's own execution fail? Decided BEFORE post-node
            # hooks: a hook may annotate a successful result (TimingHook /
            # BudgetHook write ``error`` as a signal and BudgetHook stops the
            # run) — that is governance, not a failed node — or recover a
            # failed one by clearing ``error`` / setting ``error_recovered``.
            node_failed = bool(result.error) and not result.error_recovered

            # ── CRITICAL flag — abort on error ──
            if node.has_flag(NodeFlag.CRITICAL) and result.error:
                logger.error(
                    "CRITICAL node %r failed — aborting graph: %s", node.name, result.error
                )
                state.node_history.append(result)
                state.record_node_timing(node.name, result.duration_ms)
                last_failure = result
                break

            # ── Runtime graph mutations ──
            if result.graph_mutations:
                for mutation in result.graph_mutations:
                    if mutations_count < self.max_mutations_per_run:
                        live_graph.apply_mutation(mutation)
                        mutations_count += 1
                    else:
                        logger.warning(
                            "Mutation budget exhausted (%d/%d)",
                            mutations_count,
                            self.max_mutations_per_run,
                        )

            # LLM-requested graph actions
            if (
                self.allow_self_modification
                and isinstance(result.output, dict)
                and result.output.get("_graph_action")
                and mutations_count < self.max_mutations_per_run
            ):
                self._apply_llm_graph_action(live_graph, result.output, state)
                mutations_count += 1

            # ── Post-node hooks ──
            for hook in self.hooks:
                if hasattr(hook, "post_node"):
                    try:
                        result = await hook.post_node(node, result, state)
                    except Exception as exc:
                        logger.warning("post_node hook %r failed: %s", type(hook).__name__, exc)

            # ── Update state ──
            state.record_node_timing(node.name, result.duration_ms)
            state.total_tokens += result.total_tokens
            state.node_history.append(result)
            state.trim_messages()
            last_failure = (
                result if node_failed and result.error and not result.error_recovered else None
            )

            # ── Resolve next node ──
            # If a hook forced __end__ (e.g. BudgetHook), respect it
            if state.current_node == "__end__":
                break
            next_node = self._resolve_transition(node, result, state, live_graph)
            state.current_node = next_node

            # ── Safety checks ──
            state.iteration += 1
            node_count = state.increment_node_iteration(node.name)

            if state.iteration > self.max_iterations:
                logger.warning(
                    "Max iterations (%d) reached in graph %r", self.max_iterations, live_graph.name
                )
                break

            if node_count > self.max_node_iterations:
                logger.warning(
                    "Node %r exceeded max iterations (%d)", node.name, self.max_node_iterations
                )
                state.current_node = self._handle_stuck_node(node, state, live_graph)

        # ── Build report ──
        self._last_report = ExecutionReport(
            total_iterations=state.iteration,
            total_tokens=state.total_tokens,
            total_duration_ms=(time.monotonic() - run_start) * 1000,
            nodes_visited=state.visited,
            tool_calls=state.tool_calls_made,
            graph_mutations=mutations_count,
            guards_passed=sum(len(nr.guards_passed) for nr in state.node_history),
            guards_failed=sum(len(nr.guards_failed) for nr in state.node_history),
            error=last_failure.error if last_failure is not None else None,
        )

        if last_failure is not None:
            # The run ended on a failure nothing recovered from: surface it.
            # Returning the messages here would hand the caller its own
            # question back as the "answer".
            raise GraphExecutionError(
                live_graph.name,
                last_failure.node_name,
                last_failure.error or "unknown error",
                report=self._last_report,
            ) from failure_cause(last_failure)

        return {"messages": state.messages}

    # ──────────────────────────────────────────────────────────────────
    # astream_events — streaming execution (LangGraph v2 compat)
    # ──────────────────────────────────────────────────────────────────

    async def astream_events(
        self,
        input: dict[str, Any],
        *,
        config: dict[str, Any] | None = None,
        version: str = "v2",
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream execution events matching LangGraph v2 format.

        Yields event dicts consumed by ``PromptiseAgent.astream_with_tools()``:
        - ``on_tool_start``, ``on_tool_end``, ``on_tool_error``
        - ``on_chat_model_stream``
        - ``on_node_start``, ``on_node_end`` (engine-specific)
        """
        config = dict(config) if config else {}
        config["_engine_model"] = self.model
        config["_max_iterations"] = self.max_iterations
        config["_engine_hooks"] = self.hooks
        if self.compaction is not None and "_engine_compaction" not in config:
            config["_engine_compaction"] = self.compaction

        # Collect all tools for runtime injection
        if "_engine_tools" not in config:
            all_tools_s: list = list(self.tools)
            seen_s: set[str] = {t.name for t in all_tools_s}
            for node_obj in self.graph.nodes.values():
                for tool in getattr(node_obj, "tools", []) or []:
                    if tool.name not in seen_s:
                        all_tools_s.append(tool)
                        seen_s.add(tool.name)
            config["_engine_tools"] = all_tools_s

        live_graph = self.graph.copy()
        if not live_graph.entry:
            return

        state = GraphState(
            messages=list(input.get("messages", [])),
            current_node=live_graph.entry,
            graph=live_graph,
        )
        last_failure: NodeResult | None = None

        while state.current_node != "__end__":
            try:
                node = live_graph.get_node(state.current_node)
            except (KeyError, ValueError):
                node = None
            if node is None:
                logger.error(
                    "Node %r not found in graph %r — ending stream",
                    state.current_node,
                    live_graph.name,
                )
                break
            state.visited.append(state.current_node)

            # Yield node_start
            yield {
                "event": "on_node_start",
                "name": node.name,
                "run_id": str(uuid4()),
                "data": {"node_type": type(node).__name__},
            }

            # ── Pre-execute flag processing ──
            node_start = time.monotonic()
            skip_result = await self._pre_execute_flags(node, state, config)

            if skip_result is not None:
                last_result = skip_result
            else:
                # Stream node events — initialize default to prevent AttributeError
                last_result = NodeResult(
                    node_name=node.name,
                    node_type=type(node).__name__.lower(),
                    iteration=state.iteration,
                )

                if node.has_flag(NodeFlag.RETRYABLE):
                    # For retryable nodes, use execute (not stream) with retry
                    last_result = await self._execute_with_retry(node, state, config)
                else:
                    streaming_failed = False
                    try:
                        async for event in node.stream(state, config):
                            yield {
                                "event": event.event,
                                "name": event.node_name,
                                "run_id": str(uuid4()),
                                "data": event.data,
                            }
                            if event.event == "on_node_end" and "result" in event.data:
                                last_result = event.data["result"]

                    except Exception as exc:
                        logger.error("Streaming error in node %r: %s", node.name, exc)
                        last_result = NodeResult(
                            node_name=node.name,
                            node_type=type(node).__name__.lower(),
                            iteration=state.iteration,
                        )
                        record_failure(last_result, exc)
                        streaming_failed = True
                        yield {
                            "event": "on_node_error",
                            "name": node.name,
                            "run_id": str(uuid4()),
                            "data": {"error": str(exc)},
                        }

                    # If streaming didn't produce a real result, execute normally
                    if (
                        not last_result.raw_output
                        and not last_result.error
                        and not streaming_failed
                    ):
                        try:
                            last_result = await node.execute(state, config)
                        except Exception as exc:
                            last_result = NodeResult(
                                node_name=node.name,
                                node_type=type(node).__name__.lower(),
                                iteration=state.iteration,
                            )
                            record_failure(last_result, exc)

                # Post-execute flag processing (always runs — restores state)
                await self._post_execute_flags(node, last_result, state, config)

            duration_ms = (time.monotonic() - node_start) * 1000
            last_result.duration_ms = duration_ms
            last_failure = (
                last_result if last_result.error and not last_result.error_recovered else None
            )

            # ── CRITICAL flag — abort on error ──
            if node.has_flag(NodeFlag.CRITICAL) and last_result.error:
                logger.error(
                    "CRITICAL node %r failed — aborting graph: %s",
                    node.name,
                    last_result.error,
                )
                state.node_history.append(last_result)
                state.record_node_timing(node.name, duration_ms)
                state.trim_messages()
                yield {
                    "event": "on_node_error",
                    "name": node.name,
                    "run_id": str(uuid4()),
                    "data": {"error": last_result.error, "critical": True},
                }
                break

            # Yield node_end
            yield {
                "event": "on_node_end",
                "name": node.name,
                "run_id": str(uuid4()),
                "data": {
                    "duration_ms": duration_ms,
                    "tokens": last_result.total_tokens,
                },
            }

            # Update state
            state.record_node_timing(node.name, duration_ms)
            state.total_tokens += last_result.total_tokens
            state.node_history.append(last_result)
            state.trim_messages()

            # Resolve next
            state.current_node = self._resolve_transition(node, last_result, state, live_graph)

            # Safety
            state.iteration += 1
            if state.iteration > self.max_iterations:
                break
            if state.increment_node_iteration(node.name) > self.max_node_iterations:
                state.current_node = self._handle_stuck_node(node, state, live_graph)

        if last_failure is not None:
            # Consumers have seen every event (including on_node_error);
            # the run still failed, and that must not end as a silent stop.
            raise GraphExecutionError(
                live_graph.name, last_failure.node_name, last_failure.error or "unknown error"
            ) from failure_cause(last_failure)

    # ──────────────────────────────────────────────────────────────────
    # Transition resolution
    # ──────────────────────────────────────────────────────────────────

    def _resolve_transition(
        self,
        node: BaseNode,
        result: NodeResult,
        state: GraphState,
        graph: PromptGraph,
    ) -> str:
        """Determine the next node after execution.

        Resolution order:
        1. If node had tool_calls → re-enter same node (tool loop)
        2. If NodeResult.next_node is set → use it
        3. Check graph edges with conditions
        4. Check node's transition dict against output
        5. Fall back to node.default_next
        6. Fall back to ``__end__``
        """
        # 1. Tool loop: if tools were called, re-enter the same node
        #    so the LLM sees the tool results and can decide next action
        if result.tool_calls and result.transition_reason == "tool_calls_present":
            return node.name

        # 2. Node explicitly set next_node (from NodeResult)
        if result.next_node:
            return result.next_node

        # 3. LLM-directed routing: if the output contains a _next or route
        #    field that names a valid node, go there directly.
        #    This makes every PromptNode a dynamic router.
        if isinstance(result.output, dict):
            for route_key in ("_next", "route", "next_step", "goto"):
                target = result.output.get(route_key)
                if isinstance(target, str):
                    if target == "__end__" or graph.has_node(target):
                        result.transition_reason = f"LLM routed via output.{route_key}={target!r}"
                        return target

        # 4. Check graph edges (conditional edges, priority-sorted)
        for edge in graph.get_edges_from(node.name):
            if edge.condition is None:
                # Always edge — take it
                return edge.to_node
            try:
                if edge.condition(result):
                    return edge.to_node
            except Exception as exc:
                logger.debug(
                    "Edge condition failed (%s → %s): %s", edge.from_node, edge.to_node, exc
                )

        # 5. Check node transitions against output keys
        if isinstance(result.output, dict):
            for key, target in node.transitions.items():
                val = result.output.get(key)
                if val is True or val == key or (isinstance(val, str) and val):
                    return target

        # 5. Default next
        if node.default_next:
            return node.default_next

        # 6. No transition found — end
        logger.debug("No transition from node %r — ending graph", node.name)
        return "__end__"

    # ──────────────────────────────────────────────────────────────────
    # Error recovery
    # ──────────────────────────────────────────────────────────────────

    def _handle_stuck_node(
        self,
        node: BaseNode,
        state: GraphState,
        graph: PromptGraph,
    ) -> str:
        """Recover when a node exceeds its iteration limit."""
        # Try error transition
        if "error" in node.transitions:
            logger.info("Stuck node %r → using error transition", node.name)
            return node.transitions["error"]

        # Try global error handler
        if graph.has_node("__error__"):
            logger.info("Stuck node %r → using __error__ handler", node.name)
            return "__error__"

        # Force end
        logger.warning("Stuck node %r — forcing graph end", node.name)
        return "__end__"

    # ──────────────────────────────────────────────────────────────────
    # LLM graph actions
    # ──────────────────────────────────────────────────────────────────

    def _apply_llm_graph_action(
        self,
        graph: PromptGraph,
        output: dict[str, Any],
        state: GraphState,
    ) -> None:
        """Apply a graph mutation requested by the LLM."""
        action = output.get("_graph_action", "")

        if action == "add_node" and output.get("_node_config"):
            mutation = GraphMutation(
                action="add_node",
                node_config=output["_node_config"],
                node_name=output["_node_config"].get("name", ""),
            )
            graph.apply_mutation(mutation)
            logger.info("LLM added node %r to graph", mutation.node_name)

        elif action == "skip_to" and output.get("_target"):
            target = output["_target"]
            if graph.has_node(target) or target == "__end__":
                state.current_node = target
                logger.info("LLM skipped to node %r", target)
            else:
                logger.warning("LLM tried to skip to unknown node %r", target)

        elif action == "add_edge":
            from_n = output.get("_from", "")
            to_n = output.get("_to", "")
            if from_n and to_n:
                graph.add_edge(from_n, to_n)
                logger.info("LLM added edge %s → %s", from_n, to_n)

        elif action == "retry_with" and output.get("_context_update"):
            state.context.update(output["_context_update"])
            logger.info("LLM requested retry with updated context")

    # ──────────────────────────────────────────────────────────────────
    # Flag processing
    # ──────────────────────────────────────────────────────────────────

    async def _pre_execute_flags(
        self,
        node: BaseNode,
        state: GraphState,
        config: dict[str, Any],
    ) -> NodeResult | None:
        """Process flags before node execution.

        Returns a ``NodeResult`` if the node should be skipped entirely
        (e.g. SKIP_ON_ERROR, CACHEABLE hit). Returns ``None`` to proceed
        with normal execution.
        """
        # SKIP_ON_ERROR — skip this node if the previous one errored
        if node.has_flag(NodeFlag.SKIP_ON_ERROR) and state.node_history:
            last = state.node_history[-1]
            if last.error:
                logger.info(
                    "Skipping node %r (SKIP_ON_ERROR) — previous node %r errored",
                    node.name,
                    last.node_name,
                )
                return NodeResult(
                    node_name=node.name,
                    node_type=type(node).__name__.lower(),
                    iteration=state.iteration,
                    transition_reason="skipped_on_error",
                )

        # CACHEABLE — return cached result if available
        if node.has_flag(NodeFlag.CACHEABLE):
            cache_key = self._build_cache_key(node, state)
            cache = state.context.get("_node_cache", {})
            if cache_key in cache:
                logger.debug("Cache hit for node %r (key=%s)", node.name, cache_key[:16])
                cached = cache[cache_key]
                # Return a copy so mutations don't corrupt the cache
                import copy as _copy

                return NodeResult(
                    node_name=cached.node_name,
                    node_type=cached.node_type,
                    iteration=state.iteration,
                    output=_copy.deepcopy(cached.output),
                    raw_output=cached.raw_output,
                    next_node=cached.next_node,
                    transition_reason="cache_hit",
                )
            config["_cache_key"] = cache_key

        # NO_HISTORY — strip conversation messages, keep only system message
        if node.has_flag(NodeFlag.NO_HISTORY):
            config["_saved_messages"] = list(state.messages)
            # Keep only the first message if it's a system message
            if (
                state.messages
                and hasattr(state.messages[0], "type")
                and state.messages[0].type == "system"
            ):
                state.messages = [state.messages[0]]
            else:
                state.messages = []

        # ISOLATED_CONTEXT — isolate state.context for this node
        if node.has_flag(NodeFlag.ISOLATED_CONTEXT):
            config["_saved_context"] = dict(state.context)
            # Only pass input_keys data if the node declares them
            input_keys = getattr(node, "input_keys", None) or []
            if input_keys:
                isolated = {k: state.context[k] for k in input_keys if k in state.context}
            else:
                isolated = {}
            state.context = isolated

        # LIGHTWEIGHT — swap to lightweight model if available
        if (
            node.has_flag(NodeFlag.LIGHTWEIGHT)
            and self.lightweight_model is not None
            and not getattr(node, "model_override", None)
        ):
            config["_original_model"] = config.get("_engine_model")
            config["_engine_model"] = self.lightweight_model

        # REQUIRES_HUMAN — flag in state and emit event to hooks
        if node.has_flag(NodeFlag.REQUIRES_HUMAN):
            state.context["_awaiting_human"] = node.name
            for hook in self.hooks:
                if hasattr(hook, "on_human_required"):
                    await hook.on_human_required(node.name, state)

        return None

    async def _post_execute_flags(
        self,
        node: BaseNode,
        result: NodeResult,
        state: GraphState,
        config: dict[str, Any],
    ) -> None:
        """Process flags after node execution. Mutates ``result`` and ``state`` in place."""
        # Restore NO_HISTORY — put back original messages + any new ones from this node
        if node.has_flag(NodeFlag.NO_HISTORY) and "_saved_messages" in config:
            original = config.pop("_saved_messages")
            # Append messages the node added during execution
            if result.messages_added:
                original.extend(result.messages_added)
            state.messages = original

        # Restore ISOLATED_CONTEXT — merge back only the output
        if node.has_flag(NodeFlag.ISOLATED_CONTEXT) and "_saved_context" in config:
            saved = config.pop("_saved_context")
            output_key = getattr(node, "output_key", None)
            if output_key and output_key in state.context:
                saved[output_key] = state.context[output_key]
            # Also merge the auto-written {name}_output
            auto_key = f"{node.name}_output"
            if auto_key in state.context:
                saved[auto_key] = state.context[auto_key]
            state.context = saved

        # Restore LIGHTWEIGHT model
        if node.has_flag(NodeFlag.LIGHTWEIGHT) and "_original_model" in config:
            config["_engine_model"] = config.pop("_original_model")

        # CACHEABLE — store result for future cache hits
        if node.has_flag(NodeFlag.CACHEABLE) and "_cache_key" in config:
            cache_key = config.pop("_cache_key")
            if not result.error:
                cache = state.context.setdefault("_node_cache", {})
                _MAX_NODE_CACHE = 128
                if len(cache) >= _MAX_NODE_CACHE:
                    # Evict oldest entry (FIFO) to bound memory
                    cache.pop(next(iter(cache)))
                cache[cache_key] = result

        # VERBOSE — log full output at debug level
        if node.has_flag(NodeFlag.VERBOSE):
            logger.debug(
                "VERBOSE node %r output (%d chars): %s",
                node.name,
                len(result.raw_output),
                result.raw_output[:2000],
            )

        # OBSERVABLE — emit metrics event to hooks
        if node.has_flag(NodeFlag.OBSERVABLE):
            metrics_data = {
                "node_name": node.name,
                "duration_ms": result.duration_ms,
                "total_tokens": result.total_tokens,
                "tool_calls": len(result.tool_calls),
                "error": result.error,
            }
            for hook in self.hooks:
                if hasattr(hook, "on_observable_event"):
                    await hook.on_observable_event(
                        NodeEvent(
                            event="on_node_metrics",
                            node_name=node.name,
                            data=metrics_data,
                        ),
                        state,
                    )

        # SUMMARIZE_OUTPUT — use LLM to actually summarize long outputs
        if node.has_flag(NodeFlag.SUMMARIZE_OUTPUT):
            if isinstance(result.raw_output, str) and len(result.raw_output) > 1000:
                model = config.get("_engine_model")
                if model is not None:
                    try:
                        summary_response = await model.ainvoke(
                            [
                                SystemMessage(
                                    content=(
                                        "Summarize the following output concisely. "
                                        "Preserve all key facts, numbers, and conclusions. "
                                        "Remove redundancy and verbose explanations."
                                    )
                                ),
                                HumanMessage(content=result.raw_output),
                            ]
                        )
                        summarized = (
                            summary_response.content
                            if hasattr(summary_response, "content")
                            else str(summary_response)
                        )
                        result.raw_output = summarized
                        logger.debug("Summarized output for node %r", node.name)
                    except Exception as exc:
                        logger.warning("Failed to summarize output for node %r: %s", node.name, exc)

        # VALIDATE_OUTPUT — validate against output_schema
        if node.has_flag(NodeFlag.VALIDATE_OUTPUT):
            output_schema = getattr(node, "output_schema", None)
            if output_schema is not None and result.output is not None:
                validation_errors = self._validate_against_schema(result.output, output_schema)
                if validation_errors:
                    result.guards_failed.extend(validation_errors)
                    logger.info(
                        "VALIDATE_OUTPUT failed for node %r: %s",
                        node.name,
                        validation_errors,
                    )
                else:
                    result.guards_passed.append("output_schema_validation")

    async def _execute_with_retry(
        self,
        node: BaseNode,
        state: GraphState,
        config: dict[str, Any],
    ) -> NodeResult:
        """Execute a RETRYABLE node with exponential backoff on failure.

        Uses capped exponential backoff: ``0.5 * 2^attempt`` seconds,
        maxing out at 8 seconds per wait.
        """
        max_retries = max(getattr(node, "max_iterations", 3), 1)
        last_error: str | None = None
        result = NodeResult(
            node_name=node.name,
            node_type=type(node).__name__.lower(),
            iteration=state.iteration,
            error="retry never executed",
        )

        for attempt in range(max_retries):
            try:
                result = await node.execute(state, config)
                if not result.error:
                    # Success — clean up retry context
                    state.context.pop("_retry_attempt", None)
                    state.context.pop("_retry_error", None)
                    return result
                last_error = result.error
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                result = NodeResult(
                    node_name=node.name,
                    node_type=type(node).__name__.lower(),
                    iteration=state.iteration,
                )
                record_failure(result, exc, last_error)

            # Enrich state for next attempt so the node can adapt
            state.context["_retry_attempt"] = attempt + 1
            state.context["_retry_error"] = last_error

            if attempt < max_retries - 1:
                backoff = min(0.5 * (2**attempt), 8.0)  # Cap at 8s
                logger.info(
                    "RETRYABLE node %r attempt %d/%d failed (%s), retrying in %.1fs",
                    node.name,
                    attempt + 1,
                    max_retries,
                    last_error,
                    backoff,
                )
                await asyncio.sleep(backoff)

        logger.warning(
            "RETRYABLE node %r exhausted %d attempts. Last error: %s",
            node.name,
            max_retries,
            last_error,
        )
        state.context.pop("_retry_attempt", None)
        state.context.pop("_retry_error", None)
        return result

    @staticmethod
    def _build_cache_key(node: BaseNode, state: GraphState) -> str:
        """Build a deterministic cache key for a CACHEABLE node."""
        input_keys = getattr(node, "input_keys", None) or []
        if input_keys:
            relevant = {k: state.context.get(k) for k in input_keys}
        else:
            relevant = {}
        raw = f"{node.name}:{json.dumps(relevant, sort_keys=True, default=str)}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    @staticmethod
    def _validate_against_schema(output: Any, schema: type) -> list[str]:
        """Validate output against a Pydantic model or callable schema.

        Returns a list of error messages (empty if valid).
        """
        errors: list[str] = []

        # Pydantic model validation
        if hasattr(schema, "model_validate"):
            try:
                if isinstance(output, dict):
                    schema.model_validate(output)
                elif not isinstance(output, schema):
                    errors.append(f"Expected {schema.__name__}, got {type(output).__name__}")
            except Exception as exc:
                errors.append(f"Schema validation failed: {exc}")
        # Pydantic v1 style
        elif hasattr(schema, "validate"):
            try:
                schema.validate(output)
            except Exception as exc:
                errors.append(f"Schema validation failed: {exc}")
        # Dict schema — check required keys
        elif isinstance(schema, dict):
            if isinstance(output, dict):
                for key in schema:
                    if key not in output:
                        errors.append(f"Missing required key: {key}")
            else:
                errors.append(f"Expected dict, got {type(output).__name__}")

        return errors

    # ──────────────────────────────────────────────────────────────────
    # Report
    # ──────────────────────────────────────────────────────────────────

    @property
    def last_report(self) -> ExecutionReport | None:
        """The execution report from the last ``ainvoke()`` call."""
        return self._last_report
