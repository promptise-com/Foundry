# Utilities API Reference

Root-level modules: approval, cache, events, fallback, guardrails, streaming, tool optimization, adaptive strategy, context engine, env resolution, types, and exceptions.

## Environment Variables

Parse and resolve `${ENV_VAR}` syntax with default values and recursive resolution.

### resolve_env_var

::: promptise.env_resolver.resolve_env_var
    options:
      show_source: false
      heading_level: 4

### resolve_env_in_dict

::: promptise.env_resolver.resolve_env_in_dict
    options:
      show_source: false
      heading_level: 4

### validate_all_env_vars_available

::: promptise.env_resolver.validate_all_env_vars_available
    options:
      show_source: false
      heading_level: 4

---

## MCP Tool Discovery

### ToolInfo

::: promptise.tools.ToolInfo
    options:
      show_source: false
      heading_level: 4

### MCPClientError

::: promptise.tools.MCPClientError
    options:
      show_source: false
      heading_level: 4

---

## Approval (Human-in-the-loop)

Pause the agent before destructive actions and wait for explicit human approval.

### ApprovalRequest

::: promptise.approval.ApprovalRequest
    options:
      show_source: false
      heading_level: 4

### ApprovalDecision

::: promptise.approval.ApprovalDecision
    options:
      show_source: false
      heading_level: 4

### ApprovalHandler

::: promptise.approval.ApprovalHandler
    options:
      show_source: false
      heading_level: 4

### CallbackApprovalHandler

::: promptise.approval.CallbackApprovalHandler
    options:
      show_source: false
      heading_level: 4

### WebhookApprovalHandler

::: promptise.approval.WebhookApprovalHandler
    options:
      show_source: false
      heading_level: 4

### QueueApprovalHandler

::: promptise.approval.QueueApprovalHandler
    options:
      show_source: false
      heading_level: 4

### ApprovalPolicy

::: promptise.approval.ApprovalPolicy
    options:
      show_source: false
      heading_level: 4

### approval_elicitation_callback

::: promptise.approval.approval_elicitation_callback
    options:
      show_source: false
      heading_level: 4

---

## Semantic Cache

Serves cached responses for semantically similar queries. In-memory or Redis backend with per-user/per-session/shared scope isolation.

### SemanticCache

::: promptise.cache.SemanticCache
    options:
      show_source: false
      heading_level: 4

### CacheEntry

::: promptise.cache.CacheEntry
    options:
      show_source: false
      heading_level: 4

### CacheStats

::: promptise.cache.CacheStats
    options:
      show_source: false
      heading_level: 4

### EmbeddingProvider

::: promptise.cache.EmbeddingProvider
    options:
      show_source: false
      heading_level: 4

### LocalEmbeddingProvider

::: promptise.cache.LocalEmbeddingProvider
    options:
      show_source: false
      heading_level: 4

### OpenAIEmbeddingProvider

::: promptise.cache.OpenAIEmbeddingProvider
    options:
      show_source: false
      heading_level: 4

### InMemoryCacheBackend

::: promptise.cache.InMemoryCacheBackend
    options:
      show_source: false
      heading_level: 4

### RedisCacheBackend

::: promptise.cache.RedisCacheBackend
    options:
      show_source: false
      heading_level: 4

---

## Event Notifications

Webhook + callback sinks for structured agent events (invocation errors, tool failures, budget violations, etc).

### AgentEvent

::: promptise.events.AgentEvent
    options:
      show_source: false
      heading_level: 4

### EventSink

::: promptise.events.EventSink
    options:
      show_source: false
      heading_level: 4

### EventNotifier

::: promptise.events.EventNotifier
    options:
      show_source: false
      heading_level: 4

### WebhookSink

::: promptise.events.WebhookSink
    options:
      show_source: false
      heading_level: 4

### CallbackSink

::: promptise.events.CallbackSink
    options:
      show_source: false
      heading_level: 4

### LogSink

::: promptise.events.LogSink
    options:
      show_source: false
      heading_level: 4

### EventBusSink

::: promptise.events.EventBusSink
    options:
      show_source: false
      heading_level: 4

### default_pii_sanitizer

::: promptise.events.default_pii_sanitizer
    options:
      show_source: false
      heading_level: 4

---

## Model Fallback

Automatic provider failover. Uses a circuit breaker to route around unhealthy providers without adding latency.

### FallbackChain

::: promptise.fallback.FallbackChain
    options:
      show_source: false
      heading_level: 4

---

## Security Guardrails

Multi-head security scanner: prompt injection (DeBERTa ML), PII detection (69 regex patterns), credential detection (96 patterns), NER (GLiNER), content safety, custom rules.

### PromptiseSecurityScanner

::: promptise.guardrails.PromptiseSecurityScanner
    options:
      show_source: false
      heading_level: 4

### SecurityFinding

::: promptise.guardrails.SecurityFinding
    options:
      show_source: false
      heading_level: 4

### ScanReport

::: promptise.guardrails.ScanReport
    options:
      show_source: false
      heading_level: 4

### PIICategory

::: promptise.guardrails.PIICategory
    options:
      show_source: false
      heading_level: 4

### CredentialCategory

::: promptise.guardrails.CredentialCategory
    options:
      show_source: false
      heading_level: 4

### Severity

::: promptise.guardrails.Severity
    options:
      show_source: false
      heading_level: 4

### Action

::: promptise.guardrails.Action
    options:
      show_source: false
      heading_level: 4

---

## Streaming

### StreamEvent

::: promptise.streaming.StreamEvent
    options:
      show_source: false
      heading_level: 4

### ToolStartEvent

::: promptise.streaming.ToolStartEvent
    options:
      show_source: false
      heading_level: 4

### ToolEndEvent

::: promptise.streaming.ToolEndEvent
    options:
      show_source: false
      heading_level: 4

### TokenEvent

::: promptise.streaming.TokenEvent
    options:
      show_source: false
      heading_level: 4

### DoneEvent

::: promptise.streaming.DoneEvent
    options:
      show_source: false
      heading_level: 4

### ErrorEvent

::: promptise.streaming.ErrorEvent
    options:
      show_source: false
      heading_level: 4

---

## Tool Optimization

Static schema minification + semantic tool selection (local embeddings) to cut prompt tokens by 40-70% on agents with many tools.

### OptimizationLevel

::: promptise.tool_optimization.OptimizationLevel
    options:
      show_source: false
      heading_level: 4

### ToolOptimizationConfig

::: promptise.tool_optimization.ToolOptimizationConfig
    options:
      show_source: false
      heading_level: 4

---

## Adaptive Strategy

Failure classification and strategy learning. Agents track past failures, categorize them, and adjust behavior.

### FailureCategory

::: promptise.strategy.FailureCategory
    options:
      show_source: false
      heading_level: 4

### FailureLog

::: promptise.strategy.FailureLog
    options:
      show_source: false
      heading_level: 4

### AdaptiveStrategyConfig

::: promptise.strategy.AdaptiveStrategyConfig
    options:
      show_source: false
      heading_level: 4

### AdaptiveStrategyManager

::: promptise.strategy.AdaptiveStrategyManager
    options:
      show_source: false
      heading_level: 4

### classify_failure

::: promptise.strategy.classify_failure
    options:
      show_source: false
      heading_level: 4

---

## Context Engine

Token budgeting across multiple context layers with model-aware window detection.

### ContextLayer

::: promptise.context_engine.ContextLayer
    options:
      show_source: false
      heading_level: 4

### ContextReport

::: promptise.context_engine.ContextReport
    options:
      show_source: false
      heading_level: 4

### Tokenizer

::: promptise.context_engine.Tokenizer
    options:
      show_source: false
      heading_level: 4

---

## Callback Handler

Bridges LangChain callbacks into the observability collector.

### PromptiseCallbackHandler

::: promptise.callback_handler.PromptiseCallbackHandler
    options:
      show_source: false
      heading_level: 4

---

## Exceptions

### SuperAgentError

::: promptise.exceptions.SuperAgentError
    options:
      show_source: false
      heading_level: 4

### SuperAgentValidationError

::: promptise.exceptions.SuperAgentValidationError
    options:
      show_source: false
      heading_level: 4

### EnvVarNotFoundError

::: promptise.exceptions.EnvVarNotFoundError
    options:
      show_source: false
      heading_level: 4

---

## Type Definitions

### ModelLike

::: promptise.types.ModelLike
    options:
      show_source: false
      heading_level: 4

---

## System Prompt

### DEFAULT_SYSTEM_PROMPT

::: promptise.prompt.DEFAULT_SYSTEM_PROMPT
    options:
      show_source: false
      heading_level: 4
