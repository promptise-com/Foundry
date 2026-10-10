"""Specialized reasoning nodes — composable building bricks for agent reasoning.

Each node comes fully configured with its own instructions, output schema,
context management, and state updates.  Developers just pick which nodes
to include — the node handles everything else internally.

Usage::

    from promptise.engine.reasoning_nodes import (
        PlanNode, ThinkNode, ReflectNode, ObserveNode,
        JustifyNode, CritiqueNode, SynthesizeNode, ValidateNode,
    )

    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers=my_servers,
        agent_pattern=PromptGraph("my-agent", nodes=[
            PlanNode("plan", is_entry=True),
            PromptNode("act", inject_tools=True),
            ThinkNode("think"),
            ReflectNode("reflect"),
            SynthesizeNode("answer", is_terminal=True),
        ]),
    )
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .base import BaseNode
from .nodes import PromptNode
from .state import GraphState, NodeFlag, NodeResult

logger = logging.getLogger("promptise.engine")


# ═══════════════════════════════════════════════════════════════════════════════
# ThinkNode — Pure reasoning, gap analysis, no tools
# ═══════════════════════════════════════════════════════════════════════════════


class ThinkNode(PromptNode):
    """Analyzes current state, identifies gaps, recommends next step.

    Pre-configured with:
    - Instructions for structured gap analysis
    - Auto-injects observations and plan from state
    - No tools (pure reasoning)
    - Outputs: gap_analysis, confidence, next_step, reasoning

    The developer just creates ``ThinkNode("think")`` — everything
    else is handled internally.
    """

    def __init__(
        self, name: str = "think", *, focus_areas: list[str] | None = None, **kwargs: Any
    ) -> None:
        focus = ""
        if focus_areas:
            focus = "\nFocus on: " + ", ".join(focus_areas)

        kwargs.setdefault(
            "instructions",
            (
                "You are an analytical reasoning engine. Examine the current state carefully.\n\n"
                "1. What information do you already have?\n"
                "2. What gaps remain — what do you still need?\n"
                "3. How confident are you in the current findings? (1-5)\n"
                "4. What is the single best next step?\n\n"
                "Be precise and honest about uncertainty."
                f"{focus}"
            ),
        )
        kwargs.setdefault("output_key", "think_output")
        kwargs.setdefault("include_observations", True)
        kwargs.setdefault("include_plan", True)
        kwargs.setdefault("include_reflections", False)
        kwargs.setdefault("tools", None)
        kwargs.setdefault("description", "Gap analysis and next-step reasoning")
        super().__init__(name, **kwargs)
        self.flags.update({NodeFlag.READONLY, NodeFlag.LIGHTWEIGHT})


# ═══════════════════════════════════════════════════════════════════════════════
# ReflectNode — Self-evaluation, mistake identification
# ═══════════════════════════════════════════════════════════════════════════════


class ReflectNode(PromptNode):
    """Reviews what happened, identifies mistakes, generates corrections.

    Pre-configured with:
    - Instructions for honest self-assessment
    - Auto-injects observations and past reflections
    - Auto-stores new reflections in state
    - Outputs: progress_assessment, mistake, correction, confidence, route
    """

    def __init__(self, name: str = "reflect", *, review_depth: int = 3, **kwargs: Any) -> None:
        kwargs.setdefault(
            "instructions",
            (
                "You are a reflection engine. Review what happened in the last steps.\n\n"
                "1. Assess overall progress honestly (one sentence)\n"
                "2. Identify any mistakes or suboptimal decisions\n"
                "3. Suggest a specific correction for each mistake\n"
                "4. Rate your confidence in the current direction (1-5)\n"
                "5. Decide the route: 'continue' (keep going), 'replan' (start over), or 'answer' (ready to finish)\n\n"
                "Be self-critical. Surface problems early."
            ),
        )
        kwargs.setdefault("output_key", "reflection")
        kwargs.setdefault("include_observations", True)
        kwargs.setdefault("include_reflections", True)
        kwargs.setdefault("include_plan", True)
        kwargs.setdefault("tools", None)
        kwargs.setdefault("description", "Self-evaluation and mistake identification")
        self._review_depth = review_depth
        super().__init__(name, **kwargs)
        self.flags.update({NodeFlag.STATEFUL, NodeFlag.OBSERVABLE})

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Execute reflection and auto-store in state."""
        result = await super().execute(state, config)
        if result.error:
            logger.debug(
                "%s %r skipped state mutation due to error: %s",
                type(self).__name__,
                self.name,
                result.error,
            )
            return result

        # Auto-store reflection in state
        if result.output and isinstance(result.output, dict):
            mistake = result.output.get("mistake", "")
            correction = result.output.get("correction", "")
            raw_confidence = result.output.get("confidence", 0.5)
            try:
                confidence = float(raw_confidence) if raw_confidence is not None else 0.5
            except (TypeError, ValueError):
                confidence = 0.5
            if mistake:
                state.add_reflection(
                    iteration=state.iteration,
                    mistake=mistake,
                    correction=correction,
                    confidence=confidence,
                    stage="reflect",
                )

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# ObserveNode — Interpret tool results, extract structured data
# ═══════════════════════════════════════════════════════════════════════════════


class ObserveNode(PromptNode):
    """Processes raw tool results into structured, actionable data.

    Pre-configured with:
    - Instructions to extract entities, facts, and key findings
    - Auto-injects recent observations
    - Enriches state.context with extracted data
    - Outputs: summary, entities, facts, key_findings
    """

    def __init__(self, name: str = "observe", **kwargs: Any) -> None:
        kwargs.setdefault(
            "instructions",
            (
                "You are an observation processor. Analyze the raw tool results.\n\n"
                "1. Summarize what was found (2-3 sentences)\n"
                "2. Extract named entities (people, companies, dates, locations)\n"
                "3. Extract factual claims that can be verified\n"
                "4. Identify the key findings that matter most for the task\n\n"
                "Be precise. Distinguish facts from opinions."
            ),
        )
        kwargs.setdefault("output_key", "observation")
        kwargs.setdefault("include_observations", True)
        kwargs.setdefault("include_plan", False)
        kwargs.setdefault("include_reflections", False)
        kwargs.setdefault("tools", None)
        kwargs.setdefault("description", "Tool result interpretation and data extraction")
        super().__init__(name, **kwargs)
        self.flags.add(NodeFlag.STATEFUL)

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Execute observation and enrich state with extracted data."""
        result = await super().execute(state, config)
        if result.error:
            logger.debug(
                "%s %r skipped state mutation due to error: %s",
                type(self).__name__,
                self.name,
                result.error,
            )
            return result

        # Merge extracted entities/facts into state.context
        if result.output and isinstance(result.output, dict):
            entities = result.output.get("entities", [])
            facts = result.output.get("facts", [])
            # Ensure lists before merging (LLM may return strings)
            if isinstance(entities, list) and entities:
                existing = state.context.get("extracted_entities", [])
                state.context["extracted_entities"] = existing + entities
            if isinstance(facts, list) and facts:
                existing = state.context.get("extracted_facts", [])
                state.context["extracted_facts"] = existing + facts

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# JustifyNode — Chain-of-thought justification for audit trail
# ═══════════════════════════════════════════════════════════════════════════════


class JustifyNode(PromptNode):
    """Forces explicit justification of the last decision/action.

    Pre-configured with:
    - Instructions for structured reasoning chain
    - Auto-injects the last action from state history
    - Stores justification in state for audit trail
    - Outputs: reasoning_chain, evidence, conclusion, confidence
    """

    def __init__(self, name: str = "justify", **kwargs: Any) -> None:
        kwargs.setdefault(
            "instructions",
            (
                "You are a justification engine. Explain WHY the last action was taken.\n\n"
                "1. Lay out the reasoning chain step by step\n"
                "2. Cite specific evidence for each step\n"
                "3. State the conclusion clearly\n"
                "4. Rate confidence in this justification (1-5)\n\n"
                "This is for an audit trail. Be thorough and precise."
            ),
        )
        kwargs.setdefault("output_key", "justification")
        kwargs.setdefault("include_observations", True)
        kwargs.setdefault("include_plan", False)
        kwargs.setdefault("include_reflections", False)
        kwargs.setdefault("tools", None)
        kwargs.setdefault("description", "Structured justification for audit trail")
        super().__init__(name, **kwargs)
        self.flags.update({NodeFlag.READONLY, NodeFlag.OBSERVABLE, NodeFlag.VERBOSE})

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Execute justification and store in state."""
        result = await super().execute(state, config)
        if result.error:
            logger.debug(
                "%s %r skipped state mutation due to error: %s",
                type(self).__name__,
                self.name,
                result.error,
            )
            return result

        if result.output and isinstance(result.output, dict):
            raw_confidence = result.output.get("confidence", 0)
            try:
                confidence = float(raw_confidence) if raw_confidence is not None else 0
            except (TypeError, ValueError):
                confidence = 0
            justifications = state.context.get("justifications", [])
            justifications.append(
                {
                    "iteration": state.iteration,
                    "reasoning": result.output.get("reasoning_chain", []),
                    "conclusion": result.output.get("conclusion", ""),
                    "confidence": confidence,
                }
            )
            state.context["justifications"] = justifications[-5:]  # Keep last 5

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# CritiqueNode — Adversarial self-review
# ═══════════════════════════════════════════════════════════════════════════════


class CritiqueNode(PromptNode):
    """Challenges the current answer/plan with counter-arguments.

    Pre-configured with:
    - Instructions for adversarial review
    - Severity scoring (routes to revision if above threshold)
    - Outputs: weaknesses, counter_arguments, improvements, severity
    """

    def __init__(
        self, name: str = "critique", *, severity_threshold: float = 0.5, **kwargs: Any
    ) -> None:
        kwargs.setdefault(
            "instructions",
            (
                "You are an adversarial critic. Challenge the current answer or plan.\n\n"
                "1. Identify weaknesses and gaps\n"
                "2. Present counter-arguments\n"
                "3. Suggest specific improvements\n"
                "4. Rate overall severity of issues (0.0 = perfect, 1.0 = fundamentally flawed)\n\n"
                "Be tough but constructive. The goal is improvement, not destruction."
            ),
        )
        kwargs.setdefault("output_key", "critique")
        kwargs.setdefault("include_observations", True)
        kwargs.setdefault("include_plan", True)
        kwargs.setdefault("include_reflections", True)
        kwargs.setdefault("tools", None)
        kwargs.setdefault("description", "Adversarial self-review with counter-arguments")
        self._severity_threshold = severity_threshold
        super().__init__(name, **kwargs)
        self.flags.update({NodeFlag.READONLY, NodeFlag.OBSERVABLE})

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Execute critique and route based on severity."""
        result = await super().execute(state, config)
        if result.error:
            logger.debug(
                "%s %r skipped state mutation due to error: %s",
                type(self).__name__,
                self.name,
                result.error,
            )
            return result

        if result.output and isinstance(result.output, dict):
            raw_severity = result.output.get("severity", 0.0)
            try:
                severity = float(raw_severity) if raw_severity is not None else 0.0
            except (TypeError, ValueError):
                severity = 0.0
            if severity > self._severity_threshold:
                result.transition_reason = (
                    f"severity {severity} > threshold {self._severity_threshold}"
                )
                # Route to revision if transitions configured
                for key in ("revise", "improve", "retry", "replan"):
                    if key in self.transitions:
                        result.next_node = self.transitions[key]
                        break

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# PlanNode — Structured planning with subgoal management
# ═══════════════════════════════════════════════════════════════════════════════


class PlanNode(PromptNode):
    """Creates structured plans with self-evaluation and subgoal tracking.

    Pre-configured with:
    - Instructions for creating prioritized subgoals
    - Quality self-evaluation (re-plans if below threshold)
    - Auto-manages state.plan and state.completed
    - Outputs: subgoals, priorities, active_subgoal, quality_score
    """

    def __init__(
        self,
        name: str = "plan",
        *,
        max_subgoals: int = 4,
        quality_threshold: int = 3,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault(
            "instructions",
            (
                f"You are a strategic planner. Create a clear, actionable plan.\n\n"
                f"1. Break the task into {max_subgoals} or fewer subgoals\n"
                "2. Prioritize them (most impactful first)\n"
                "3. Identify the first subgoal to tackle\n"
                f"4. Self-evaluate: rate plan quality 1-5 (below {quality_threshold} = re-plan)\n\n"
                "Each subgoal must be specific, measurable, and achievable in one step."
            ),
        )
        kwargs.setdefault("output_key", "plan_output")
        kwargs.setdefault("include_observations", False)
        kwargs.setdefault("include_plan", False)
        kwargs.setdefault("include_reflections", True)
        kwargs.setdefault("tools", None)
        kwargs.setdefault("description", "Structured planning with quality self-evaluation")
        self._max_subgoals = max_subgoals
        self._quality_threshold = quality_threshold
        super().__init__(name, **kwargs)
        self.flags.update({NodeFlag.STATEFUL, NodeFlag.OBSERVABLE})

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Execute planning and update state.plan."""
        result = await super().execute(state, config)
        if result.error:
            logger.debug(
                "%s %r skipped state mutation due to error: %s",
                type(self).__name__,
                self.name,
                result.error,
            )
            return result

        if result.output and isinstance(result.output, dict):
            subgoals = result.output.get("subgoals", [])
            if not isinstance(subgoals, list):
                subgoals = []
            raw_quality = result.output.get("quality_score", result.output.get("quality", 3))
            try:
                quality = float(raw_quality) if raw_quality is not None else 3
            except (TypeError, ValueError):
                quality = 3

            # Update state plan
            if subgoals:
                state.plan = subgoals[: self._max_subgoals]

            # Quality gate — re-plan if below threshold
            if quality < self._quality_threshold:
                result.transition_reason = (
                    f"plan quality {quality} < threshold {self._quality_threshold}"
                )
                if "replan" in self.transitions:
                    result.next_node = self.transitions["replan"]
                elif self.name in list(state.graph.nodes if state.graph else {}):
                    result.next_node = self.name  # Loop back to self

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# SynthesizeNode — Combine all findings into final answer
# ═══════════════════════════════════════════════════════════════════════════════


class SynthesizeNode(PromptNode):
    """Combines all gathered data into a comprehensive final answer.

    Pre-configured with:
    - Instructions for synthesis with source citation
    - Auto-injects ALL observations, reflections, plan progress
    - Default terminal node (ends the graph)
    - Outputs: answer, confidence, sources
    """

    def __init__(self, name: str = "synthesize", **kwargs: Any) -> None:
        kwargs.setdefault(
            "instructions",
            (
                "You are a synthesis engine. Combine all gathered information "
                "into a clear, comprehensive final answer.\n\n"
                "1. Address the original question directly\n"
                "2. Support claims with evidence from your research\n"
                "3. Cite sources where possible\n"
                "4. Rate your confidence in the answer (1-5)\n"
                "5. Note any remaining uncertainties\n\n"
                "Be thorough but concise. Quality over quantity."
            ),
        )
        kwargs.setdefault("output_key", "synthesis")
        kwargs.setdefault("include_observations", True)
        kwargs.setdefault("include_plan", True)
        kwargs.setdefault("include_reflections", True)
        kwargs.setdefault("tools", None)
        kwargs.setdefault("default_next", "__end__")
        kwargs.setdefault("description", "Final answer synthesis from all gathered data")
        super().__init__(name, **kwargs)
        self.flags.add(NodeFlag.OBSERVABLE)


# ═══════════════════════════════════════════════════════════════════════════════
# ValidateNode — LLM-powered validation against criteria
# ═══════════════════════════════════════════════════════════════════════════════


class ValidateNode(PromptNode):
    """LLM evaluates whether output meets specified criteria.

    Pre-configured with:
    - Natural language validation criteria
    - Pass/fail routing
    - Outputs: passes, issues, suggestions, confidence
    """

    def __init__(
        self,
        name: str = "validate",
        *,
        criteria: list[str] | None = None,
        on_pass: str = "__end__",
        on_fail: str | None = None,
        **kwargs: Any,
    ) -> None:
        criteria_text = ""
        if criteria:
            criteria_text = "\n".join(f"- {c}" for c in criteria)
        else:
            criteria_text = "- The answer is complete and addresses the question\n- Claims are supported by evidence\n- No factual errors"

        kwargs.setdefault(
            "instructions",
            (
                "You are a quality validator. Evaluate whether the current output meets these criteria:\n\n"
                f"{criteria_text}\n\n"
                "1. Does it pass ALL criteria? (true/false)\n"
                "2. List any issues found\n"
                "3. Suggest improvements for each issue\n"
                "4. Rate confidence in your assessment (1-5)"
            ),
        )
        kwargs.setdefault("output_key", "validation")
        kwargs.setdefault("include_observations", True)
        kwargs.setdefault("include_plan", False)
        kwargs.setdefault("include_reflections", False)
        kwargs.setdefault("tools", None)
        kwargs.setdefault("description", "LLM-powered quality validation")
        self._on_pass = on_pass
        self._on_fail = on_fail
        super().__init__(name, **kwargs)
        self.flags.update({NodeFlag.READONLY, NodeFlag.VALIDATE_OUTPUT})

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Execute validation and route based on pass/fail."""
        result = await super().execute(state, config)
        if result.error:
            logger.debug(
                "%s %r skipped state mutation due to error: %s",
                type(self).__name__,
                self.name,
                result.error,
            )
            return result

        if result.output and isinstance(result.output, dict):
            passes = result.output.get("passes", result.output.get("pass", False))
            if passes:
                result.next_node = self._on_pass
                result.transition_reason = "validation passed"
            elif self._on_fail:
                result.next_node = self._on_fail
                result.transition_reason = f"validation failed: {result.output.get('issues', [])}"

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# RetryNode — Wraps another node with retry logic
# ═══════════════════════════════════════════════════════════════════════════════


class RetryNode(BaseNode):
    """Wraps another node with retry logic and error enrichment.

    On failure, enriches the state with error context so the
    wrapped node can adapt on the next attempt.
    """

    def __init__(
        self,
        name: str,
        *,
        wrapped_node: BaseNode,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, description=f"Retry wrapper for {wrapped_node.name}", **kwargs)
        self.wrapped_node = wrapped_node
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.flags.add(NodeFlag.RETRYABLE)

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Execute wrapped node with retries."""
        start = time.perf_counter()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                result = await self.wrapped_node.execute(state, config)
                if not result.error:
                    result.node_name = self.name
                    result.duration_ms = (time.perf_counter() - start) * 1000
                    return result
                last_error = result.error
            except Exception as exc:
                last_error = str(exc)

            # Enrich state with error context for next attempt
            state.context["_retry_attempt"] = attempt + 1
            state.context["_retry_error"] = last_error
            state.context["_retry_node"] = self.wrapped_node.name

            if attempt < self.max_retries:
                delay = min(self.backoff_factor * (2**attempt), 8.0)  # Cap at 8s
                await asyncio.sleep(delay)

        return NodeResult(
            node_name=self.name,
            error=f"All {self.max_retries + 1} attempts failed. Last error: {last_error}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FanOutNode — Send different prompts to parallel children
# ═══════════════════════════════════════════════════════════════════════════════


class FanOutNode(BaseNode):
    """Sends different sub-questions to parallel child nodes.

    Unlike ParallelNode (which runs the same state through each child),
    FanOutNode gives each child a focused sub-question via state overrides.
    """

    def __init__(
        self,
        name: str,
        *,
        branches: list[tuple[BaseNode, dict[str, Any]]] | None = None,
        merge_strategy: str = "dict",
        **kwargs: Any,
    ) -> None:
        super().__init__(name, description="Fan-out to parallel sub-questions", **kwargs)
        self.branches = list(branches) if branches else []
        self.merge_strategy = merge_strategy
        self.flags.add(NodeFlag.PARALLEL_SAFE)

    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        """Execute each branch with its state override concurrently."""
        start = time.perf_counter()
        result = NodeResult(node_name=self.name, node_type="fan_out", iteration=state.iteration)

        async def run_branch(node: BaseNode, overrides: dict[str, Any]) -> NodeResult:
            child_state = GraphState(
                messages=list(state.messages),
                context={**state.context, **overrides},
                current_node=node.name,
                observations=list(state.observations),
            )
            return await node.execute(child_state, config)

        child_results = await asyncio.gather(
            *[run_branch(node, ovr) for node, ovr in self.branches],
            return_exceptions=True,
        )

        outputs: dict[str, Any] = {}
        for (node, _), child_result in zip(self.branches, child_results, strict=False):
            # asyncio.gather(return_exceptions=True) may return BaseException
            # (CancelledError included); narrow on BaseException, not Exception.
            if isinstance(child_result, BaseException):
                outputs[node.name] = {"error": str(child_result)}
            else:
                outputs[node.name] = child_result.output
                result.tool_calls.extend(child_result.tool_calls)
                result.total_tokens += child_result.total_tokens

        result.output = outputs
        result.duration_ms = (time.perf_counter() - start) * 1000
        return result
