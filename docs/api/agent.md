# Agent API Reference

Core agent creation, invocation, identity, conversations, and the full reasoning engine — every public class across `promptise.agent`, `promptise.engine`, and `promptise.conversations`.

## Building Agents

### build_agent

::: promptise.agent.build_agent
    options:
      show_source: false
      heading_level: 3

### PromptiseAgent

::: promptise.agent.PromptiseAgent
    options:
      show_source: false
      heading_level: 3
      members:
        - ainvoke
        - astream
        - astream_with_tools
        - invoke
        - chat
        - get_session
        - list_sessions
        - delete_session
        - update_session
        - shutdown
        - get_stats
        - generate_report

### CallerContext

::: promptise.agent.CallerContext
    options:
      show_source: false
      heading_level: 3

### get_current_caller

::: promptise.agent.get_current_caller
    options:
      show_source: false
      heading_level: 3

---

## Reasoning Graph Engine

### PromptGraphEngine

::: promptise.engine.execution.PromptGraphEngine
    options:
      show_source: false
      heading_level: 4
      members:
        - ainvoke
        - astream_events
        - last_report

### GraphExecutionError

::: promptise.engine.execution.GraphExecutionError
    options:
      show_source: false
      heading_level: 4

### PromptGraph

::: promptise.engine.graph.PromptGraph
    options:
      show_source: false
      heading_level: 4

### Edge

::: promptise.engine.graph.Edge
    options:
      show_source: false
      heading_level: 4

---

## Graph State

### GraphState

::: promptise.engine.state.GraphState
    options:
      show_source: false
      heading_level: 4

### NodeResult

::: promptise.engine.state.NodeResult
    options:
      show_source: false
      heading_level: 4

### NodeEvent

::: promptise.engine.state.NodeEvent
    options:
      show_source: false
      heading_level: 4

### NodeFlag

::: promptise.engine.state.NodeFlag
    options:
      show_source: false
      heading_level: 4

### GraphMutation

::: promptise.engine.state.GraphMutation
    options:
      show_source: false
      heading_level: 4

### ExecutionReport

::: promptise.engine.state.ExecutionReport
    options:
      show_source: false
      heading_level: 4

---

## Base Node Types

### BaseNode

::: promptise.engine.base.BaseNode
    options:
      show_source: false
      heading_level: 4

### NodeProtocol

::: promptise.engine.base.NodeProtocol
    options:
      show_source: false
      heading_level: 4

### @node decorator

::: promptise.engine.base.node
    options:
      show_source: false
      heading_level: 4

---

## Standard Nodes

### PromptNode

::: promptise.engine.nodes.PromptNode
    options:
      show_source: false
      heading_level: 4

### ToolNode

::: promptise.engine.nodes.ToolNode
    options:
      show_source: false
      heading_level: 4

### RouterNode

::: promptise.engine.nodes.RouterNode
    options:
      show_source: false
      heading_level: 4

### GuardNode

::: promptise.engine.nodes.GuardNode
    options:
      show_source: false
      heading_level: 4

### ParallelNode

::: promptise.engine.nodes.ParallelNode
    options:
      show_source: false
      heading_level: 4

### LoopNode

::: promptise.engine.nodes.LoopNode
    options:
      show_source: false
      heading_level: 4

### HumanNode

::: promptise.engine.nodes.HumanNode
    options:
      show_source: false
      heading_level: 4

### TransformNode

::: promptise.engine.nodes.TransformNode
    options:
      show_source: false
      heading_level: 4

### SubgraphNode

::: promptise.engine.nodes.SubgraphNode
    options:
      show_source: false
      heading_level: 4

### AutonomousNode

::: promptise.engine.nodes.AutonomousNode
    options:
      show_source: false
      heading_level: 4

### CodeActionNode

::: promptise.engine.code_action.CodeActionNode
    options:
      show_source: false
      heading_level: 4

---

## Reasoning Nodes

### ThinkNode

::: promptise.engine.reasoning_nodes.ThinkNode
    options:
      show_source: false
      heading_level: 4

### PlanNode

::: promptise.engine.reasoning_nodes.PlanNode
    options:
      show_source: false
      heading_level: 4

### ReflectNode

::: promptise.engine.reasoning_nodes.ReflectNode
    options:
      show_source: false
      heading_level: 4

### CritiqueNode

::: promptise.engine.reasoning_nodes.CritiqueNode
    options:
      show_source: false
      heading_level: 4

### ObserveNode

::: promptise.engine.reasoning_nodes.ObserveNode
    options:
      show_source: false
      heading_level: 4

### SynthesizeNode

::: promptise.engine.reasoning_nodes.SynthesizeNode
    options:
      show_source: false
      heading_level: 4

### ValidateNode

::: promptise.engine.reasoning_nodes.ValidateNode
    options:
      show_source: false
      heading_level: 4

### JustifyNode

::: promptise.engine.reasoning_nodes.JustifyNode
    options:
      show_source: false
      heading_level: 4

### RetryNode

::: promptise.engine.reasoning_nodes.RetryNode
    options:
      show_source: false
      heading_level: 4

### FanOutNode

::: promptise.engine.reasoning_nodes.FanOutNode
    options:
      show_source: false
      heading_level: 4

---

## Prebuilt Graph Factories

Factory functions that build common reasoning patterns. All exposed as static methods on `PromptGraph` (e.g. `PromptGraph.react(...)`).

### build_react_graph

::: promptise.engine.prebuilts.build_react_graph
    options:
      show_source: false
      heading_level: 4

### build_verify_graph

::: promptise.engine.prebuilts.build_verify_graph
    options:
      show_source: false
      heading_level: 4

### build_managed_graph

::: promptise.engine.prebuilts.build_managed_graph
    options:
      show_source: false
      heading_level: 4

### build_code_action_graph

::: promptise.engine.prebuilts.build_code_action_graph
    options:
      show_source: false
      heading_level: 4

### build_peoatr_graph

::: promptise.engine.prebuilts.build_peoatr_graph
    options:
      show_source: false
      heading_level: 4

### build_research_graph

::: promptise.engine.prebuilts.build_research_graph
    options:
      show_source: false
      heading_level: 4

### build_autonomous_graph

::: promptise.engine.prebuilts.build_autonomous_graph
    options:
      show_source: false
      heading_level: 4

### build_deliberate_graph

::: promptise.engine.prebuilts.build_deliberate_graph
    options:
      show_source: false
      heading_level: 4

### build_debate_graph

::: promptise.engine.prebuilts.build_debate_graph
    options:
      show_source: false
      heading_level: 4

### build_pipeline_graph

::: promptise.engine.prebuilts.build_pipeline_graph
    options:
      show_source: false
      heading_level: 4

---

## Engine Hooks

Observers that the engine calls at each node boundary. Used for logging, metrics, cycle detection, and budget enforcement.

### Hook

::: promptise.engine.hooks.Hook
    options:
      show_source: false
      heading_level: 4

### LoggingHook

::: promptise.engine.hooks.LoggingHook
    options:
      show_source: false
      heading_level: 4

### TimingHook

::: promptise.engine.hooks.TimingHook
    options:
      show_source: false
      heading_level: 4

### CycleDetectionHook

::: promptise.engine.hooks.CycleDetectionHook
    options:
      show_source: false
      heading_level: 4

### MetricsHook

::: promptise.engine.hooks.MetricsHook
    options:
      show_source: false
      heading_level: 4

### BudgetHook

::: promptise.engine.hooks.BudgetHook
    options:
      show_source: false
      heading_level: 4

---

## Preprocessors and Postprocessors

Pluggable functions that run before/after a node's LLM call. Compose with `chain_preprocessors` and `chain_postprocessors`.

### context_enricher

::: promptise.engine.processors.context_enricher
    options:
      show_source: false
      heading_level: 4

### state_summarizer

::: promptise.engine.processors.state_summarizer
    options:
      show_source: false
      heading_level: 4

### input_validator

::: promptise.engine.processors.input_validator
    options:
      show_source: false
      heading_level: 4

### json_extractor

::: promptise.engine.processors.json_extractor
    options:
      show_source: false
      heading_level: 4

### confidence_scorer

::: promptise.engine.processors.confidence_scorer
    options:
      show_source: false
      heading_level: 4

### state_writer

::: promptise.engine.processors.state_writer
    options:
      show_source: false
      heading_level: 4

### output_truncator

::: promptise.engine.processors.output_truncator
    options:
      show_source: false
      heading_level: 4

### chain_preprocessors

::: promptise.engine.processors.chain_preprocessors
    options:
      show_source: false
      heading_level: 4

### chain_postprocessors

::: promptise.engine.processors.chain_postprocessors
    options:
      show_source: false
      heading_level: 4

---

## Conversations

### ConversationStore

::: promptise.conversations.ConversationStore
    options:
      show_source: false
      heading_level: 3

### Message

::: promptise.conversations.Message
    options:
      show_source: false
      heading_level: 3

### SessionInfo

::: promptise.conversations.SessionInfo
    options:
      show_source: false
      heading_level: 3

### Built-in Stores

::: promptise.conversations.InMemoryConversationStore
    options:
      show_source: false
      heading_level: 4

::: promptise.conversations.SQLiteConversationStore
    options:
      show_source: false
      heading_level: 4

::: promptise.conversations.PostgresConversationStore
    options:
      show_source: false
      heading_level: 4

::: promptise.conversations.RedisConversationStore
    options:
      show_source: false
      heading_level: 4
