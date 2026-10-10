"""Public API for promptise."""

from .agent import CallerContext, PromptiseAgent, build_agent, get_current_caller
from .approval import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalPolicy,
    ApprovalRequest,
    CallbackApprovalHandler,
    QueueApprovalHandler,
    WebhookApprovalHandler,
)
from .approval_classifier import (
    DEFAULT_READ_ONLY_PREFIXES,
    ApprovalRule,
    AutoApprovalClassifier,
    ClassifierDecisionTrace,
    ClassifierStats,
)

# Semantic Cache
from .cache import (
    CacheStats,
    EmbeddingProvider,
    LocalEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SemanticCache,
)
from .callback_handler import PromptiseCallbackHandler
from .config import (
    HTTPServerSpec,
    ServerSpec,
    StdioServerSpec,
    servers_to_mcp_config,
)
from .context_engine import ContextEngine
from .conversations import (
    ConversationStore,
    InMemoryConversationStore,
    PostgresConversationStore,
    RedisConversationStore,
    SessionAccessDenied,
    SessionInfo,
    SQLiteConversationStore,
    generate_session_id,
)
from .conversations import (
    Message as ConversationMessage,
)

# PromptGraph Engine
from .engine import (
    BaseNode,
    Edge,
    GraphState,
    NodeResult,
    PromptGraph,
    PromptGraphEngine,
    PromptNode,
)
from .events import (
    AgentEvent,
    CallbackSink,
    EventBusSink,
    EventNotifier,
    EventSink,
    LogSink,
    WebhookSink,
)
from .exceptions import (
    EnvVarNotFoundError,
    SuperAgentError,
    SuperAgentValidationError,
)
from .fallback import FallbackChain

# Guardrails
from .guardrails import (
    ContentSafetyDetector,
    CredentialCategory,
    CredentialDetector,
    CustomRule,
    GuardrailViolation,
    InjectionDetector,
    NERDetector,
    PIICategory,
    PIIDetector,
    PromptiseSecurityScanner,
    ScanReport,
    SecurityFinding,
)
from .identity import AgentIdentity, IdentityError
from .mcp.client import (
    MCPClient,
    MCPClientError,
    MCPConnectionRejectedError,
    MCPMultiClient,
    MCPToolAdapter,
)
from .memory import (
    ChromaProvider,
    InMemoryProvider,
    Mem0Provider,
    MemoryAgent,
    MemoryProvider,
    MemoryResult,
    sanitize_memory_content,
)
from .models import PROVIDERS, Model, ModelSetupError, check_model, resolve_model
from .observability_config import (
    ExportFormat,
    ObservabilityConfig,
    ObserveLevel,
    TransporterType,
)
from .prompts import (
    BaseContext,
    Prompt,
    PromptBuilder,
    PromptContext,
    PromptSuite,
    prompt,
)
from .rag import (
    Chunk,
    Chunker,
    Document,
    DocumentLoader,
    Embedder,
    IndexReport,
    InMemoryVectorStore,
    RAGPipeline,
    RecursiveTextChunker,
    RetrievalResult,
    VectorStore,
    content_hash,
    rag_to_tool,
)

# Runtime
from .runtime import (
    AgentContext,
    AgentProcess,
    AgentRuntime,
    BudgetConfig,
    HealthConfig,
    MissionConfig,
    MissionState,
    ProcessConfig,
    ProcessState,
    RuntimeConfig,
    SecretScopeConfig,
)
from .strategy import (
    AdaptiveStrategyConfig,
    AdaptiveStrategyManager,
    FailureCategory,
    FailureLog,
    classify_failure,
)
from .streaming import (
    DoneEvent,
    ErrorEvent,
    StreamEvent,
    TokenEvent,
    ToolEndEvent,
    ToolStartEvent,
)
from .superagent import SuperAgentConfig, SuperAgentLoader, load_superagent_file
from .superagent_schema import (
    AgentSection,
    CrossAgentConfig,
    DetailedModelConfig,
    HTTPServerConfig,
    MemorySection,
    ModelConfig,
    SandboxConfigSection,
    ServerConfig,
    StdioServerConfig,
    SuperAgentSchema,
)
from .tool_optimization import (
    OptimizationLevel,
    ToolOptimizationConfig,
)
from .tools import ToolInfo

__all__ = [
    "HTTPServerSpec",
    "ServerSpec",
    "StdioServerSpec",
    "servers_to_mcp_config",
    "ToolInfo",
    "build_agent",
    # Models (bring your own provider)
    "Model",
    "PROVIDERS",
    "ModelSetupError",
    "check_model",
    "resolve_model",
    # Tool Optimization
    "OptimizationLevel",
    "ToolOptimizationConfig",
    # MCP Client
    "MCPClient",
    "MCPClientError",
    "MCPConnectionRejectedError",
    "MCPMultiClient",
    "MCPToolAdapter",
    # Agent
    "PromptiseAgent",
    "CallerContext",
    "get_current_caller",
    # Agent Identity
    "AgentIdentity",
    "IdentityError",
    # Streaming
    "StreamEvent",
    "ToolStartEvent",
    "ToolEndEvent",
    "TokenEvent",
    "DoneEvent",
    "ErrorEvent",
    # Fallback
    "FallbackChain",
    # Context Engine
    "ContextEngine",
    # Adaptive Strategy
    "AdaptiveStrategyConfig",
    "AdaptiveStrategyManager",
    "FailureCategory",
    "FailureLog",
    "classify_failure",
    # Events
    "AgentEvent",
    "EventNotifier",
    "EventSink",
    "WebhookSink",
    "CallbackSink",
    "LogSink",
    "EventBusSink",
    # Approval (HITL)
    "ApprovalPolicy",
    "ApprovalRequest",
    "ApprovalDecision",
    "ApprovalHandler",
    "CallbackApprovalHandler",
    "WebhookApprovalHandler",
    "QueueApprovalHandler",
    "AutoApprovalClassifier",
    "ApprovalRule",
    "ClassifierStats",
    "ClassifierDecisionTrace",
    "DEFAULT_READ_ONLY_PREFIXES",
    # Semantic Cache
    "SemanticCache",
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "CacheStats",
    # Guardrails
    "PromptiseSecurityScanner",
    "SecurityFinding",
    "ScanReport",
    "GuardrailViolation",
    "PIICategory",
    "CredentialCategory",
    "InjectionDetector",
    "PIIDetector",
    "CredentialDetector",
    "CustomRule",
    "NERDetector",
    "ContentSafetyDetector",
    # Observability
    "ObservabilityConfig",
    "ObserveLevel",
    "TransporterType",
    "ExportFormat",
    "PromptiseCallbackHandler",
    # SuperAgent types
    "SuperAgentLoader",
    "SuperAgentConfig",
    "load_superagent_file",
    "SuperAgentSchema",
    "AgentSection",
    "DetailedModelConfig",
    "ModelConfig",
    "HTTPServerConfig",
    "StdioServerConfig",
    "ServerConfig",
    "CrossAgentConfig",
    "SandboxConfigSection",
    "MemorySection",
    # Memory
    "MemoryProvider",
    "MemoryResult",
    "MemoryAgent",
    "InMemoryProvider",
    "Mem0Provider",
    "ChromaProvider",
    "sanitize_memory_content",
    # Conversations
    "ConversationStore",
    "ConversationMessage",
    "SessionInfo",
    "SessionAccessDenied",
    "generate_session_id",
    "InMemoryConversationStore",
    "PostgresConversationStore",
    "SQLiteConversationStore",
    "RedisConversationStore",
    # Prompt framework
    "prompt",
    "Prompt",
    "PromptBuilder",
    "PromptContext",
    "PromptSuite",
    "BaseContext",
    # Exceptions
    "SuperAgentError",
    "SuperAgentValidationError",
    "EnvVarNotFoundError",
    # Runtime
    "AgentProcess",
    "AgentRuntime",
    "AgentContext",
    "ProcessConfig",
    "ProcessState",
    "RuntimeConfig",
    # Runtime governance
    "BudgetConfig",
    "HealthConfig",
    "MissionConfig",
    "MissionState",
    "SecretScopeConfig",
    # PromptGraph Engine
    "PromptGraph",
    "PromptGraphEngine",
    "PromptNode",
    "BaseNode",
    "Edge",
    "GraphState",
    "NodeResult",
    # RAG base foundation
    "Document",
    "Chunk",
    "RetrievalResult",
    "IndexReport",
    "DocumentLoader",
    "Chunker",
    "Embedder",
    "VectorStore",
    "RecursiveTextChunker",
    "InMemoryVectorStore",
    "RAGPipeline",
    "rag_to_tool",
    "content_hash",
]
