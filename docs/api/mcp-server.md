# MCP Server API Reference

Production framework for building MCP-compliant tool servers. Every middleware, auth provider, guard, transform, and utility is documented here.

## Core

### MCPServer

::: promptise.mcp.server.MCPServer
    options:
      show_source: false
      heading_level: 4

### MCPRouter

::: promptise.mcp.server.MCPRouter
    options:
      show_source: false
      heading_level: 4

### ServerSettings

::: promptise.mcp.server.ServerSettings
    options:
      show_source: false
      heading_level: 4

### Depends

::: promptise.mcp.server.Depends
    options:
      show_source: false
      heading_level: 4

### mount (server composition)

::: promptise.mcp.server.mount
    options:
      show_source: false
      heading_level: 4

### hot_reload

::: promptise.mcp.server.hot_reload
    options:
      show_source: false
      heading_level: 4

---

## Context

### ClientContext

::: promptise.mcp.server.ClientContext
    options:
      show_source: false
      heading_level: 4

### RequestContext

::: promptise.mcp.server.RequestContext
    options:
      show_source: false
      heading_level: 4

### ToolResponse

::: promptise.mcp.server.ToolResponse
    options:
      show_source: false
      heading_level: 4

### get_context

::: promptise.mcp.server.get_context
    options:
      show_source: false
      heading_level: 4

### get_request_headers

::: promptise.mcp.server.get_request_headers
    options:
      show_source: false
      heading_level: 4

### set_request_headers

::: promptise.mcp.server.set_request_headers
    options:
      show_source: false
      heading_level: 4

### clear_request_headers

::: promptise.mcp.server.clear_request_headers
    options:
      show_source: false
      heading_level: 4

---

## Authentication

### JWTAuth

::: promptise.mcp.server.JWTAuth
    options:
      show_source: false
      heading_level: 4

### AsymmetricJWTAuth

::: promptise.mcp.server.AsymmetricJWTAuth
    options:
      show_source: false
      heading_level: 4

### JwksAuth

::: promptise.mcp.server.JwksAuth
    options:
      show_source: false
      heading_level: 4

### APIKeyAuth

::: promptise.mcp.server.APIKeyAuth
    options:
      show_source: false
      heading_level: 4

### AuthProvider

::: promptise.mcp.server.AuthProvider
    options:
      show_source: false
      heading_level: 4

### AuthMiddleware

::: promptise.mcp.server.AuthMiddleware
    options:
      show_source: false
      heading_level: 4

### OnAuthenticateHook

::: promptise.mcp.server.OnAuthenticateHook
    options:
      show_source: false
      heading_level: 4

### TokenEndpointConfig

::: promptise.mcp.server.TokenEndpointConfig
    options:
      show_source: false
      heading_level: 4

---

## Guards

### Guard

::: promptise.mcp.server.Guard
    options:
      show_source: false
      heading_level: 4

### RequireAuth

::: promptise.mcp.server.RequireAuth
    options:
      show_source: false
      heading_level: 4

### HasRole

::: promptise.mcp.server.HasRole
    options:
      show_source: false
      heading_level: 4

### HasAllRoles

::: promptise.mcp.server.HasAllRoles
    options:
      show_source: false
      heading_level: 4

### HasScope

::: promptise.mcp.server.HasScope
    options:
      show_source: false
      heading_level: 4

### HasAllScopes

::: promptise.mcp.server.HasAllScopes
    options:
      show_source: false
      heading_level: 4

### RequireClientId

::: promptise.mcp.server.RequireClientId
    options:
      show_source: false
      heading_level: 4

---

## Middleware

Composable middleware pipeline. Order matters: outermost first.

### Middleware

::: promptise.mcp.server.Middleware
    options:
      show_source: false
      heading_level: 4

### MiddlewareChain

::: promptise.mcp.server.MiddlewareChain
    options:
      show_source: false
      heading_level: 4

### LoggingMiddleware

::: promptise.mcp.server.LoggingMiddleware
    options:
      show_source: false
      heading_level: 4

### StructuredLoggingMiddleware

::: promptise.mcp.server.StructuredLoggingMiddleware
    options:
      show_source: false
      heading_level: 4

### TimeoutMiddleware

::: promptise.mcp.server.TimeoutMiddleware
    options:
      show_source: false
      heading_level: 4

### RateLimitMiddleware

::: promptise.mcp.server.RateLimitMiddleware
    options:
      show_source: false
      heading_level: 4

### TokenBucketLimiter

::: promptise.mcp.server.TokenBucketLimiter
    options:
      show_source: false
      heading_level: 4

### CircuitBreakerMiddleware

::: promptise.mcp.server.CircuitBreakerMiddleware
    options:
      show_source: false
      heading_level: 4

### CircuitOpenError

::: promptise.mcp.server.CircuitOpenError
    options:
      show_source: false
      heading_level: 4

### is_upstream_failure

::: promptise.mcp.server.is_upstream_failure
    options:
      show_source: false
      heading_level: 4

### CircuitState

::: promptise.mcp.server.CircuitState
    options:
      show_source: false
      heading_level: 4

### AuditMiddleware

::: promptise.mcp.server.AuditMiddleware
    options:
      show_source: false
      heading_level: 4

### WebhookMiddleware

::: promptise.mcp.server.WebhookMiddleware
    options:
      show_source: false
      heading_level: 4

### ConcurrencyLimiter

::: promptise.mcp.server.ConcurrencyLimiter
    options:
      show_source: false
      heading_level: 4

### PerToolConcurrencyLimiter

::: promptise.mcp.server.PerToolConcurrencyLimiter
    options:
      show_source: false
      heading_level: 4

### BackgroundTasks

::: promptise.mcp.server.BackgroundTasks
    options:
      show_source: false
      heading_level: 4

---

## Observability

### HealthCheck

::: promptise.mcp.server.HealthCheck
    options:
      show_source: false
      heading_level: 4

### MetricsCollector

::: promptise.mcp.server.MetricsCollector
    options:
      show_source: false
      heading_level: 4

### MetricsMiddleware

::: promptise.mcp.server.MetricsMiddleware
    options:
      show_source: false
      heading_level: 4

### PrometheusMiddleware

::: promptise.mcp.server.PrometheusMiddleware
    options:
      show_source: false
      heading_level: 4

### OTelMiddleware

::: promptise.mcp.server.OTelMiddleware
    options:
      show_source: false
      heading_level: 4

### Dashboard

::: promptise.mcp.server.Dashboard
    options:
      show_source: false
      heading_level: 4

### DashboardMiddleware

::: promptise.mcp.server.DashboardMiddleware
    options:
      show_source: false
      heading_level: 4

### DashboardState

::: promptise.mcp.server.DashboardState
    options:
      show_source: false
      heading_level: 4

---

## Caching

### CacheBackend

::: promptise.mcp.server.CacheBackend
    options:
      show_source: false
      heading_level: 4

### InMemoryCache

::: promptise.mcp.server.InMemoryCache
    options:
      show_source: false
      heading_level: 4

### RedisCache

::: promptise.mcp.server.RedisCache
    options:
      show_source: false
      heading_level: 4

### CacheMiddleware

::: promptise.mcp.server.CacheMiddleware
    options:
      show_source: false
      heading_level: 4

### cached (decorator)

::: promptise.mcp.server.cached
    options:
      show_source: false
      heading_level: 4

---

## Job Queue

Background task queue with priority, retry, progress, and cancellation.

### MCPQueue

::: promptise.mcp.server.MCPQueue
    options:
      show_source: false
      heading_level: 4

### QueueBackend

::: promptise.mcp.server.QueueBackend
    options:
      show_source: false
      heading_level: 4

### InMemoryQueueBackend

::: promptise.mcp.server.InMemoryQueueBackend
    options:
      show_source: false
      heading_level: 4

### QueueCaller

::: promptise.mcp.server.QueueCaller
    options:
      show_source: false
      heading_level: 4

---

## Streaming & Progress

### StreamingResult

::: promptise.mcp.server.StreamingResult
    options:
      show_source: false
      heading_level: 4

### ProgressReporter

::: promptise.mcp.server.ProgressReporter
    options:
      show_source: false
      heading_level: 4

### CancellationToken

::: promptise.mcp.server.CancellationToken
    options:
      show_source: false
      heading_level: 4

---

## Session State

Per-client key-value storage that persists across tool calls.

### SessionState

::: promptise.mcp.server.SessionState
    options:
      show_source: false
      heading_level: 4

### SessionManager

::: promptise.mcp.server.SessionManager
    options:
      show_source: false
      heading_level: 4

---

## Tool Versioning and Transforms

### VersionedToolRegistry

::: promptise.mcp.server.VersionedToolRegistry
    options:
      show_source: false
      heading_level: 4

### ToolTransform

::: promptise.mcp.server.ToolTransform
    options:
      show_source: false
      heading_level: 4

### NamespaceTransform

::: promptise.mcp.server.NamespaceTransform
    options:
      show_source: false
      heading_level: 4

### VisibilityTransform

::: promptise.mcp.server.VisibilityTransform
    options:
      show_source: false
      heading_level: 4

### TagFilterTransform

::: promptise.mcp.server.TagFilterTransform
    options:
      show_source: false
      heading_level: 4

---

## Elicitation and Sampling

Request structured input or LLM completions from the client mid-execution.

### Elicitor

::: promptise.mcp.server.Elicitor
    options:
      show_source: false
      heading_level: 4

### Sampler

::: promptise.mcp.server.Sampler
    options:
      show_source: false
      heading_level: 4

---

## OpenAPI

Generate MCP tools from OpenAPI specs automatically.

### OpenAPIProvider

::: promptise.mcp.server.OpenAPIProvider
    options:
      show_source: false
      heading_level: 4

---

## Exception Handlers

### ExceptionHandlerRegistry

::: promptise.mcp.server.ExceptionHandlerRegistry
    options:
      show_source: false
      heading_level: 4

---

## Errors

### MCPError

::: promptise.mcp.server.MCPError
    options:
      show_source: false
      heading_level: 4

### ToolError

::: promptise.mcp.server.ToolError
    options:
      show_source: false
      heading_level: 4

### ResourceError

::: promptise.mcp.server.ResourceError
    options:
      show_source: false
      heading_level: 4

### PromptError

::: promptise.mcp.server.PromptError
    options:
      show_source: false
      heading_level: 4

### AuthenticationError

::: promptise.mcp.server.AuthenticationError
    options:
      show_source: false
      heading_level: 4

### RateLimitError

::: promptise.mcp.server.RateLimitError
    options:
      show_source: false
      heading_level: 4

### ConcurrencyLimitError

::: promptise.mcp.server.ConcurrencyLimitError
    options:
      show_source: false
      heading_level: 4

### ValidationError

::: promptise.mcp.server.ValidationError
    options:
      show_source: false
      heading_level: 4

### CancelledError

::: promptise.mcp.server.CancelledError
    options:
      show_source: false
      heading_level: 4

---

## Tool Types

### ToolDef

::: promptise.mcp.server.ToolDef
    options:
      show_source: false
      heading_level: 4

### ResourceDef

::: promptise.mcp.server.ResourceDef
    options:
      show_source: false
      heading_level: 4

### PromptDef

::: promptise.mcp.server.PromptDef
    options:
      show_source: false
      heading_level: 4

### ToolAnnotations

::: promptise.mcp.server.ToolAnnotations
    options:
      show_source: false
      heading_level: 4

### Content

::: promptise.mcp.server.Content
    options:
      show_source: false
      heading_level: 4

### ImageContent

::: promptise.mcp.server.ImageContent
    options:
      show_source: false
      heading_level: 4

### TransportType

::: promptise.mcp.server.TransportType
    options:
      show_source: false
      heading_level: 4

### CORSConfig

::: promptise.mcp.server.CORSConfig
    options:
      show_source: false
      heading_level: 4

---

## Testing

### TestClient

::: promptise.mcp.server.TestClient
    options:
      show_source: false
      heading_level: 4

---

## Logging

### ServerLogger

::: promptise.mcp.server.ServerLogger
    options:
      show_source: false
      heading_level: 4

---

## Multi-Tenancy

### RequireTenant

::: promptise.mcp.server.RequireTenant
    options:
      show_source: false
      heading_level: 4

### HasTenant

::: promptise.mcp.server.HasTenant
    options:
      show_source: false
      heading_level: 4

---

## Approval Gates

### ApprovalGateMiddleware

::: promptise.mcp.server.ApprovalGateMiddleware
    options:
      show_source: false
      heading_level: 4

### PendingApprover

::: promptise.mcp.server.PendingApprover
    options:
      show_source: false
      heading_level: 4

### ElicitationApprover

::: promptise.mcp.server.ElicitationApprover
    options:
      show_source: false
      heading_level: 4

### ApprovalDeniedError

::: promptise.mcp.server.ApprovalDeniedError
    options:
      show_source: false
      heading_level: 4

---

## Rate Limiting

### parse_rate_limit

::: promptise.mcp.server.parse_rate_limit
    options:
      show_source: false
      heading_level: 4

### DeclaredRateLimitMiddleware

::: promptise.mcp.server.DeclaredRateLimitMiddleware
    options:
      show_source: false
      heading_level: 4
