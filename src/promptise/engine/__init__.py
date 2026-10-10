"""Reasoning Graph Engine — composable agent reasoning with rich nodes.

The engine provides a graph of composable reasoning nodes where each
node is a complete processing pipeline: its own blocks, tools, guards,
strategy, context, and transitions.

Quick start::

    from promptise.engine import PromptGraph, PromptNode, PromptGraphEngine

    graph = PromptGraph.react(tools=my_tools, system_prompt="You are helpful.")
    engine = PromptGraphEngine(graph=graph, model=my_model)
    result = await engine.ainvoke({"messages": [...]})
"""

# Import prebuilts to register factory methods on PromptGraph
from . import prebuilts as _prebuilts  # noqa: F401
from .base import BaseNode, NodeProtocol, node
from .code_action import CodeActionNode
from .compaction import ContextCompaction
from .execution import GraphExecutionError, PromptGraphEngine
from .graph import Edge, PromptGraph
from .hooks import BudgetHook, CycleDetectionHook, Hook, LoggingHook, MetricsHook, TimingHook
from .nodes import (
    AutonomousNode,
    GuardNode,
    HumanNode,
    LoopNode,
    ParallelNode,
    PromptNode,
    RouterNode,
    SubgraphNode,
    ToolNode,
    TransformNode,
)
from .reasoning_nodes import (
    CritiqueNode,
    FanOutNode,
    JustifyNode,
    ObserveNode,
    PlanNode,
    ReflectNode,
    RetryNode,
    SynthesizeNode,
    ThinkNode,
    ValidateNode,
)
from .skill_registry import Skill, SkillRegistry, parse_frontmatter
from .state import (
    ExecutionReport,
    GraphMutation,
    GraphState,
    NodeEvent,
    NodeFlag,
    NodeResult,
)

__all__ = [
    # Core
    "PromptGraph",
    "PromptGraphEngine",
    "ContextCompaction",
    "Edge",
    # Standard Nodes
    "BaseNode",
    "NodeProtocol",
    "PromptNode",
    "AutonomousNode",
    "ToolNode",
    "RouterNode",
    "GuardNode",
    "ParallelNode",
    "LoopNode",
    "HumanNode",
    "TransformNode",
    "SubgraphNode",
    "CodeActionNode",
    "node",
    # Reasoning Nodes
    "ThinkNode",
    "ReflectNode",
    "ObserveNode",
    "JustifyNode",
    "CritiqueNode",
    "PlanNode",
    "SynthesizeNode",
    "ValidateNode",
    "RetryNode",
    "FanOutNode",
    # State
    "GraphState",
    "NodeResult",
    "NodeEvent",
    "GraphMutation",
    "ExecutionReport",
    "GraphExecutionError",
    "NodeFlag",
    # Hooks
    "Hook",
    "LoggingHook",
    "TimingHook",
    "CycleDetectionHook",
    "MetricsHook",
    "BudgetHook",
    # Skill registry
    "Skill",
    "SkillRegistry",
    "parse_frontmatter",
]
