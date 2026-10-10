"""Promptise MCP Server Framework.

Build production-ready MCP servers like APIs. APIs that AIs understand.

Example::

    from promptise.mcp.server import MCPServer

    server = MCPServer(name="my-tools", version="1.0.0")

    @server.tool()
    async def add(a: int, b: int) -> int:
        \"\"\"Add two numbers.\"\"\"
        return a + b

    server.run()
"""

from ._app import MCPServer
from ._approval_gate import (
    ApprovalGateMiddleware,
    ElicitationApprover,
    PendingApprover,
)
from ._audit import AuditMiddleware
from ._auth import (
    APIKeyAuth,
    AsymmetricJWTAuth,
    AuthMiddleware,
    AuthProvider,
    JwksAuth,
    JWTAuth,
    OnAuthenticateHook,
)
from ._background import BackgroundTasks
from ._batch import register_batch_tool
from ._cache import CacheBackend, CacheMiddleware, CacheScope, InMemoryCache, cached
from ._cancellation import CancellationToken, CancelledError
from ._circuit_breaker import (
    CircuitBreakerMiddleware,
    CircuitOpenError,
    CircuitState,
    is_upstream_failure,
)
from ._composition import mount
from ._concurrency import ConcurrencyLimiter, PerToolConcurrencyLimiter
from ._context import (
    ClientContext,
    RequestContext,
    ToolResponse,
    clear_request_headers,
    get_context,
    get_request_headers,
    set_request_headers,
)
from ._dashboard import Dashboard, DashboardMiddleware, DashboardState
from ._di import Depends
from ._elicitation import Elicitor
from ._errors import (
    ApprovalDeniedError,
    AuthenticationError,
    ConcurrencyLimitError,
    MCPError,
    PromptError,
    RateLimitError,
    ResourceError,
    ToolError,
    ValidationError,
)
from ._exception_handlers import ExceptionHandlerRegistry
from ._guards import (
    Guard,
    HasAllRoles,
    HasAllScopes,
    HasRole,
    HasScope,
    HasTenant,
    RequireAuth,
    RequireClientId,
    RequireTenant,
)
from ._health import HealthCheck
from ._hot_reload import hot_reload
from ._logging import ServerLogger
from ._manifest import build_manifest, register_manifest
from ._middleware import (
    LoggingMiddleware,
    Middleware,
    MiddlewareChain,
    TimeoutMiddleware,
)
from ._observability import MetricsCollector, MetricsMiddleware
from ._openapi import OpenAPIProvider
from ._otel import OTelMiddleware
from ._progress import ProgressReporter
from ._prometheus import PrometheusMiddleware
from ._queue import InMemoryQueueBackend, MCPQueue, QueueBackend
from ._rate_limit import (
    DeclaredRateLimitMiddleware,
    RateLimitMiddleware,
    TokenBucketLimiter,
    parse_rate_limit,
)
from ._redis_cache import RedisCache
from ._router import MCPRouter
from ._sampling import Sampler
from ._serve_cli import build_serve_parser, resolve_server, run_serve
from ._session_state import SessionManager, SessionState
from ._settings import ServerSettings
from ._streaming import StreamingResult
from ._structured_logging import StructuredLoggingMiddleware
from ._testing import TestClient
from ._token_endpoint import TokenEndpointConfig
from ._transforms import NamespaceTransform, TagFilterTransform, ToolTransform, VisibilityTransform
from ._transport import CORSConfig
from ._types import (
    Content,
    ImageContent,
    PromptDef,
    ResourceDef,
    ToolAnnotations,
    ToolDef,
    TransportType,
)
from ._validation import dereference_schema
from ._versioning import VersionedToolRegistry
from ._webhooks import WebhookMiddleware

__all__ = [
    # Core
    "MCPServer",
    "MCPRouter",
    "TestClient",
    "RequestContext",
    "ClientContext",
    "ToolResponse",
    "get_context",
    "get_request_headers",
    "set_request_headers",
    "clear_request_headers",
    # Dependency injection
    "Depends",
    # Middleware
    "Middleware",
    "MiddlewareChain",
    "LoggingMiddleware",
    "TimeoutMiddleware",
    "StructuredLoggingMiddleware",
    # Auth
    "AuthProvider",
    "JWTAuth",
    "AsymmetricJWTAuth",
    "JwksAuth",
    "APIKeyAuth",
    "AuthMiddleware",
    "OnAuthenticateHook",
    # Guards
    "Guard",
    "RequireAuth",
    "HasRole",
    "HasAllRoles",
    "HasScope",
    "HasTenant",
    "RequireTenant",
    "HasAllScopes",
    "RequireClientId",
    # Concurrency
    "ConcurrencyLimiter",
    "PerToolConcurrencyLimiter",
    # Rate limiting
    "TokenBucketLimiter",
    "RateLimitMiddleware",
    "DeclaredRateLimitMiddleware",
    "parse_rate_limit",
    "ApprovalGateMiddleware",
    "PendingApprover",
    "ElicitationApprover",
    "ApprovalDeniedError",
    # Health
    "HealthCheck",
    # Observability
    "MetricsCollector",
    "MetricsMiddleware",
    "OTelMiddleware",
    "PrometheusMiddleware",
    # Exception handlers
    "ExceptionHandlerRegistry",
    # Errors
    "MCPError",
    "ToolError",
    "ResourceError",
    "PromptError",
    "AuthenticationError",
    "RateLimitError",
    "ConcurrencyLimitError",
    "ValidationError",
    "CancelledError",
    # Background tasks
    "BackgroundTasks",
    # Progress & logging
    "ProgressReporter",
    "ServerLogger",
    # Cancellation
    "CancellationToken",
    # Session state
    "SessionState",
    "SessionManager",
    # Settings
    "ServerSettings",
    # Caching
    "CacheBackend",
    "InMemoryCache",
    "RedisCache",
    "CacheMiddleware",
    "CacheScope",
    "cached",
    # Manifest
    "build_manifest",
    "register_manifest",
    # Dashboard
    "Dashboard",
    "DashboardMiddleware",
    "DashboardState",
    # Token endpoint
    "TokenEndpointConfig",
    # Schema utilities
    "dereference_schema",
    # Types
    "ToolDef",
    "ResourceDef",
    "PromptDef",
    "TransportType",
    "ToolAnnotations",
    "ImageContent",
    "Content",
    # CORS
    "CORSConfig",
    # Server composition
    "mount",
    # OpenAPI
    "OpenAPIProvider",
    # Hot reload
    "hot_reload",
    # CLI serve
    "build_serve_parser",
    "resolve_server",
    "run_serve",
    # Tool versioning
    "VersionedToolRegistry",
    # Tool transforms
    "ToolTransform",
    "NamespaceTransform",
    "VisibilityTransform",
    "TagFilterTransform",
    # Elicitation & Sampling
    "Elicitor",
    "Sampler",
    # Circuit breaker
    "CircuitBreakerMiddleware",
    "CircuitOpenError",
    "is_upstream_failure",
    "CircuitState",
    # Audit logging
    "AuditMiddleware",
    # Webhooks
    "WebhookMiddleware",
    # Batch tool calls
    "register_batch_tool",
    # Streaming results
    "StreamingResult",
    # Queue
    "MCPQueue",
    "QueueBackend",
    "InMemoryQueueBackend",
]
