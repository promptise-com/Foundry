"""Pydantic models for .superagent YAML schema validation.

This module defines the complete schema for .superagent configuration files,
including support for both simple and detailed model configurations, server
specifications, cross-agent references, and environment variable resolution.

The schema uses Pydantic v2 for strict validation and supports discriminated
unions for type-safe server configuration.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

if TYPE_CHECKING:
    from .identity import AgentIdentity

# =============================================================================
# Model Configuration Types
# =============================================================================


class DetailedModelConfig(BaseModel):
    """Detailed model configuration with provider-specific parameters.

    This configuration format allows fine-grained control over model
    initialization, including API keys, temperature, token limits, and
    provider-specific parameters.

    Attributes:
        provider: Model provider (e.g., "openai", "anthropic", "ollama").
        name: Model name/ID (e.g., "gpt-4.1", "claude-opus-4.5").
        api_key: Optional API key (supports ${ENV_VAR} syntax).
        temperature: Optional temperature parameter (0.0-2.0).
        max_tokens: Optional maximum tokens for generation.
        timeout: Optional request timeout in seconds.
        base_url: Optional custom API base URL.
        extra: Additional provider-specific parameters.

    Examples:
        >>> config = DetailedModelConfig(
        ...     provider="openai",
        ...     name="gpt-4.1",
        ...     api_key="${OPENAI_API_KEY}",
        ...     temperature=0.7
        ... )

    Azure OpenAI — the model, the deployment you named it, and where it lives
    are separate fields::

        model:
          provider: azure
          model: gpt-4o
          deployment: chat-prod
          endpoint: https://my-resource.openai.azure.com/
          api_key: ${AZURE_OPENAI_API_KEY}
          api_version: "2024-10-21"

    The common words (``deployment``, ``api_key``, ``endpoint``,
    ``api_version``, ``region``, ``project``) are translated to each
    provider's own keyword arguments exactly like
    :class:`promptise.models.Model`; a value set here counts as provided, so
    the matching environment variable is not required.  ``name:`` and
    ``model:`` are synonyms.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    provider: str = Field(..., description="Model provider name (any prefix or alias)")
    name: str = Field(..., alias="model", description="Model name (also accepted as `model:`)")
    deployment: str | None = Field(
        None,
        description="Azure OpenAI deployment name (what you named the model in Azure AI Foundry)",
    )
    api_key: str | None = Field(None, description="API key (supports ${ENV_VAR})")
    temperature: float | None = Field(None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(None, gt=0)
    timeout: int | None = Field(None, gt=0, description="Request timeout in seconds")
    base_url: str | None = Field(None, description="Custom API base URL (same as endpoint)")
    endpoint: str | None = Field(
        None,
        description=(
            "Where to send requests: an Azure OpenAI resource endpoint, an Azure AI "
            "Foundry inference endpoint, or the /v1 URL of an OpenAI-compatible server"
        ),
    )
    api_version: str | None = Field(None, description="Azure OpenAI REST API version")
    region: str | None = Field(None, description="Bedrock region or Vertex AI location")
    project: str | None = Field(None, description="Google Cloud project id (Vertex AI)")
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Additional provider-specific parameters"
    )

    @field_validator("api_key")
    @classmethod
    def warn_direct_key(cls, v: str | None) -> str | None:
        """Warn if API key appears to be a direct value (not env var)."""
        if (
            v
            and not v.startswith("${")
            and (v.startswith("sk-") or v.startswith("pk-") or len(v) > 20)
        ):
            warnings.warn(
                "Direct API key detected in config. Consider using ${ENV_VAR} syntax for security.",
                UserWarning,
                stacklevel=2,
            )
        return v


# Simple form: just a string like "openai:gpt-4.1"
ModelConfig = str | DetailedModelConfig
"""Union type supporting both simple string and detailed model configuration.

Simple form:
    model: "openai:gpt-4.1"

Detailed form:
    model:
      provider: openai
      name: gpt-4.1
      temperature: 0.7
"""


# =============================================================================
# Server Configuration Types
# =============================================================================


class HTTPServerConfig(BaseModel):
    """HTTP/SSE MCP server configuration.

    Configuration for remote MCP servers accessible via HTTP, streamable HTTP,
    or Server-Sent Events (SSE) transports.

    Attributes:
        type: Always "http" for this variant (discriminator field).
        url: Full endpoint URL (supports ${ENV_VAR}).
        transport: Transport protocol ("http", "streamable-http", "sse").
        headers: Optional HTTP headers (values support ${ENV_VAR}).
        auth: Optional auth token (supports ${ENV_VAR}).

    Examples:
        >>> server = HTTPServerConfig(
        ...     type="http",
        ...     url="http://127.0.0.1:8000/mcp",
        ...     headers={"Authorization": "Bearer ${API_TOKEN}"}
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["http"] = Field("http", description="Server type discriminator")
    url: str = Field(..., description="Full MCP endpoint URL")
    transport: Literal["http", "streamable-http", "sse"] = Field(
        "http", description="Transport protocol"
    )
    headers: dict[str, str] = Field(
        default_factory=dict, description="HTTP headers (values support ${ENV_VAR})"
    )
    auth: str | None = Field(None, description="Auth token (supports ${ENV_VAR})")


class StdioServerConfig(BaseModel):
    """Stdio (local process) MCP server configuration.

    Configuration for local MCP servers that communicate via standard
    input/output, typically a subprocess launched by the agent.

    Attributes:
        type: Always "stdio" for this variant (discriminator field).
        command: Executable command to launch.
        args: Command-line arguments.
        env: Environment variables (values support ${ENV_VAR}).
        cwd: Optional working directory.
        keep_alive: Whether to maintain persistent connection.

    Examples:
        >>> server = StdioServerConfig(
        ...     type="stdio",
        ...     command="python",
        ...     args=["-m", "mypkg.server"],
        ...     env={"API_KEY": "${MY_API_KEY}"}
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["stdio"] = Field("stdio", description="Server type discriminator")
    command: str = Field(..., description="Executable command")
    args: list[str] = Field(default_factory=list, description="Command arguments")
    env: dict[str, str] = Field(
        default_factory=dict, description="Environment variables (values support ${ENV_VAR})"
    )
    cwd: str | None = Field(None, description="Working directory")
    keep_alive: bool = Field(True, description="Maintain persistent connection")


ServerConfig = Annotated[HTTPServerConfig | StdioServerConfig, Field(discriminator="type")]
"""Union type for server configurations with discriminated union on 'type' field.

The 'type' field ("http" or "stdio") determines which server configuration
variant is used. Pydantic automatically validates the correct fields based
on this discriminator.
"""


# =============================================================================
# Cross-Agent Configuration
# =============================================================================


class CrossAgentConfig(BaseModel):
    """Cross-agent reference configuration.

    Defines a reference to another agent's .superagent configuration file,
    allowing multi-agent coordination and delegation.

    Attributes:
        file: Path to referenced .superagent file (relative to current file).
        description: Human-readable description for tool discovery.

    Examples:
        >>> config = CrossAgentConfig(
        ...     file="./agents/math_specialist.superagent",
        ...     description="Specialized math and calculation agent"
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    file: str = Field(..., description="Path to .superagent file")
    description: str = Field("", description="Agent description for tool discovery")


# =============================================================================
# Top-Level Agent Configuration
# =============================================================================


class AgentSection(BaseModel):
    """Agent-level configuration section.

    Defines the core agent configuration including model selection,
    system prompt, and tool tracing settings.

    Attributes:
        model: Model configuration (simple string or detailed object).
        instructions: Optional system prompt override.
        trace: Enable tool invocation tracing.

    Examples:
        >>> agent = AgentSection(
        ...     model="openai:gpt-4.1",
        ...     instructions="You are a helpful assistant.",
        ...     trace=True
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(..., description="Model configuration")
    instructions: str | None = Field(None, description="System prompt override")
    trace: bool = Field(True, description="Enable tool tracing")


class SandboxConfigSection(BaseModel):
    """Sandbox configuration section for .superagent files.

    Attributes:
        backend: Container backend (docker, gvisor).
        image: Base container image.
        cpu_limit: Maximum CPU cores.
        memory_limit: Maximum memory (e.g., "4G").
        disk_limit: Size of the writable workspace (e.g., "1G").
        pids_limit: Maximum number of processes and threads.
        network: Network isolation mode (none, restricted, full).
        persistent: Keep the container after the session ends.
        timeout: Max execution time in seconds.
        workdir: Working directory inside container.
        env: Additional environment variables.
        allow_sudo: Allow sudo access in container.

    Examples:
        >>> config = SandboxConfigSection(
        ...     backend="gvisor",
        ...     cpu_limit=2,
        ...     memory_limit="4G"
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["docker", "gvisor"] = Field("docker", description="Container backend")
    image: str = Field("python:3.11-slim", description="Base container image")
    cpu_limit: int = Field(2, gt=0, le=32, description="Maximum CPU cores")
    memory_limit: str = Field("4G", description="Maximum memory")
    disk_limit: str = Field("1G", description="Size of the writable workspace")
    pids_limit: int = Field(256, gt=0, le=65536, description="Maximum processes and threads")
    network: Literal["none", "restricted", "full"] = Field(
        "none", description="Network isolation mode"
    )
    persistent: bool = Field(False, description="Keep the container after the session ends")
    timeout: int = Field(300, gt=0, le=3600, description="Max execution time in seconds")
    workdir: str = Field("/workspace", description="Working directory")
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    allow_sudo: bool = Field(False, description="Allow sudo access")


class MemorySection(BaseModel):
    """Memory configuration section for .superagent files.

    Attributes:
        provider: Memory provider type (``"in_memory"``, ``"chroma"``, ``"mem0"``).
        collection: ChromaDB collection name (chroma provider).
        persist_directory: ChromaDB persistence path (chroma provider).
        user_id: Mem0 user scope (mem0 provider).
        agent_id: Mem0 agent scope (mem0 provider).

    Examples:
        >>> config = MemorySection(provider="chroma", persist_directory=".promptise/chroma")
        >>> config = MemorySection(provider="mem0", user_id="user-123")
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["in_memory", "chroma", "mem0"] = Field(
        "in_memory", description="Memory provider type"
    )
    # ChromaDB options
    collection: str = Field("agent_memory", description="ChromaDB collection name")
    persist_directory: str | None = Field(None, description="ChromaDB persistence path")
    # Mem0 options
    user_id: str = Field("default", description="Mem0 user scope")
    agent_id: str | None = Field(None, description="Mem0 agent scope")
    # Legacy alias
    backend: str | None = Field(None, description="Deprecated: use 'provider' instead")
    path: str | None = Field(None, description="Deprecated: use provider-specific options")


class ObservabilitySection(BaseModel):
    """Observability configuration section for .superagent files.

    Attributes:
        level: Detail level (off, basic, standard, full).
        session_name: Human-readable session identifier.
        record_prompts: Store full prompt/response text.
        transporters: List of transporter type strings.
        output_dir: Directory for HTML and JSON output.
        log_file: File path for structured log transporter.
        console_live: Real-time console printing.
        webhook_url: Target URL for webhook transporter.
        prometheus_port: Port for Prometheus metrics.
        otlp_endpoint: gRPC endpoint for OpenTelemetry.
        correlation_id: External correlation ID.
    """

    model_config = ConfigDict(extra="forbid")

    level: Literal["off", "basic", "standard", "full"] = Field(
        "standard", description="Detail level"
    )
    session_name: str = Field("promptise", description="Session identifier")
    record_prompts: bool = Field(False, description="Store full prompt/response text")
    transporters: list[str] = Field(
        default_factory=lambda: ["html"], description="Transporter types"
    )
    output_dir: str | None = Field(None, description="Output directory")
    log_file: str | None = Field(None, description="Structured log file path")
    console_live: bool = Field(False, description="Real-time console output")
    webhook_url: str | None = Field(None, description="Webhook transporter URL")
    prometheus_port: int = Field(9090, description="Prometheus metrics port")
    otlp_endpoint: str = Field("http://localhost:4317", description="OpenTelemetry endpoint")
    correlation_id: str | None = Field(None, description="External correlation ID")


class ToolOptimizationSection(BaseModel):
    """Tool optimization configuration for .superagent files.

    Attributes:
        level: Optimization level (none, minimal, standard, aggressive, semantic).
        embedding_model: Sentence-transformers model name or local path.
        top_k: Number of tools to select per query (semantic mode).
        score_threshold: Minimum similarity score (semantic mode).
    """

    model_config = ConfigDict(extra="forbid")

    level: Literal["minimal", "standard", "semantic"] = Field(
        "semantic", description="Optimization level"
    )
    embedding_model: str = Field(
        "all-MiniLM-L6-v2", description="Embedding model name or local path"
    )
    top_k: int = Field(10, gt=0, description="Tools to select per query")
    score_threshold: float = Field(0.1, ge=0.0, le=1.0, description="Min similarity")


class CacheSection(BaseModel):
    """Semantic cache configuration for .superagent files.

    Attributes:
        backend: Cache backend — ``"memory"`` (default) or ``"redis"``.
        redis_url: Redis connection URL (required when backend is ``"redis"``).
        similarity_threshold: Minimum cosine similarity for a cache hit (0.0–1.0).
        default_ttl: Default time-to-live in seconds for cached entries.
        scope: Cache isolation — ``"per_user"``, ``"per_session"``, or ``"shared"``.
        max_entries_per_user: Maximum cached entries per user scope.
        embedding_model: Sentence-transformers model name or local path.
        encrypt_values: Encrypt cached values at rest (Redis backend).
    """

    model_config = ConfigDict(extra="forbid")

    backend: Literal["memory", "redis"] = Field("memory", description="Cache backend")
    redis_url: str | None = Field(None, description="Redis connection URL")
    similarity_threshold: float = Field(
        0.92, ge=0.0, le=1.0, description="Min similarity for cache hit"
    )
    default_ttl: int = Field(3600, gt=0, description="Default TTL in seconds")
    scope: Literal["per_user", "per_session", "shared"] = Field(
        "per_user", description="Cache isolation scope"
    )
    max_entries_per_user: int = Field(1000, gt=0, description="Max entries per user")
    embedding_model: str = Field(
        "all-MiniLM-L6-v2", description="Embedding model name or local path"
    )
    encrypt_values: bool = Field(False, description="Encrypt values at rest")


class ApprovalSection(BaseModel):
    """Human-in-the-loop approval configuration for .superagent files.

    Attributes:
        tools: Glob patterns for tool names requiring approval.
        handler: Handler type — ``"webhook"``, ``"callback"``, or ``"queue"``.
        webhook_url: Webhook URL (required when handler is ``"webhook"``).
        timeout: Seconds to wait for approval decision.
        on_timeout: Action when timeout expires — ``"deny"`` or ``"allow"``.
        max_pending: Maximum concurrent pending approvals.
        redact_sensitive: Redact PII/credentials in approval requests.
        max_retries_after_deny: Denials of one tool (per ``deny_scope``,
            within ``deny_window``) after which the reviewer is not asked.
        deny_window: Seconds a denial counts towards the limit.
        deny_scope: ``"session"``, ``"user"`` or ``"agent"``.
        sequential: Ask for one approval at a time per invocation.
        context_messages: Conversation messages in ``context_summary``.
        webhook_allow_private_networks: Allow ``webhook_url`` on a
            private network (``WebhookApprovalHandler(allow_private_networks=True)``).
    """

    model_config = ConfigDict(extra="forbid")

    tools: list[str] = Field(..., description="Tool name patterns requiring approval")
    handler: Literal["webhook", "callback", "queue"] = Field(
        "webhook", description="Approval handler type"
    )
    webhook_url: str | None = Field(None, description="Webhook URL for approval requests")
    timeout: float = Field(300, gt=0, le=86400, description="Approval timeout in seconds")
    on_timeout: Literal["deny", "allow"] = Field("deny", description="Action on timeout")
    max_pending: int = Field(10, gt=0, description="Max concurrent pending approvals")
    redact_sensitive: bool = Field(True, description="Redact PII/credentials in requests")
    max_retries_after_deny: int = Field(3, gt=0, description="Max retries after denial")
    deny_window: float = Field(600, gt=0, description="Seconds a denial counts towards the limit")
    deny_scope: Literal["session", "user", "agent"] = Field(
        "session", description="Whose denials count together"
    )
    sequential: bool = Field(False, description="Ask for one approval at a time per invocation")
    context_messages: int = Field(
        3, ge=0, description="Conversation messages included in context_summary"
    )
    webhook_allow_private_networks: bool = Field(
        False, description="Allow webhook_url to point at a private network"
    )


class EventSinkConfig(BaseModel):
    """Configuration for a single event notification sink."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["webhook", "log"] = Field(
        ..., description="Sink type (callback/eventbus require Python, use build_agent())"
    )
    url: str | None = Field(None, description="Webhook URL")
    events: list[str] | None = Field(None, description="Event types to subscribe to (None = all)")
    headers: dict[str, str] = Field(default_factory=dict, description="Custom HTTP headers")
    secret: str | None = Field(None, description="HMAC signing secret")
    min_severity: str | None = Field(
        None, description="Minimum severity (info/warning/error/critical)"
    )
    max_retries: int = Field(3, ge=0, description="Max retry attempts")
    redact_sensitive: bool = Field(True, description="Redact PII in payloads")


class EventsSection(BaseModel):
    """Event notification configuration for .superagent files."""

    model_config = ConfigDict(extra="forbid")

    sinks: list[EventSinkConfig] = Field(default_factory=list, description="Notification sinks")


class AdaptiveSection(BaseModel):
    """Adaptive strategy (learning from failure) configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(True, description="Enable adaptive strategy learning")
    synthesis_threshold: int = Field(5, gt=0, description="Synthesize after N strategy failures")
    synthesis_model: str | None = Field(
        None, description="Model for synthesis (None = agent's model)"
    )
    max_strategies: int = Field(20, gt=0, description="Max stored strategies")
    auto_cleanup: bool = Field(True, description="Delete raw failure logs after synthesis")
    strategy_ttl: int = Field(0, ge=0, description="Strategy expiry in seconds (0 = never)")
    failure_retention: int = Field(50, gt=0, description="Max raw failure logs to keep")
    verify_human_feedback: bool = Field(True, description="LLM-as-judge on corrections")
    feedback_rate_limit: int = Field(10, ge=0, description="Max corrections per hour per sender")
    scope: Literal["per_user", "per_tenant", "per_session", "shared"] = Field(
        "per_user",
        description="Who shares failures and lessons (derived from the CallerContext)",
    )
    confidence_half_life: float = Field(
        0.0, ge=0, description="Seconds for a synthesized lesson's confidence to halve (0 = off)"
    )
    min_confidence: float = Field(
        0.3, ge=0.0, le=1.0, description="Lessons below this confidence are dropped"
    )
    allowed_tools: list[str] | None = Field(
        None, description="Only learn from failures of these tools (None = all)"
    )
    review_lessons: bool = Field(
        False, description="Hold synthesized lessons as pending until approved"
    )
    learn_from_approval_denials: bool = Field(
        True, description="Store approval denial reasons as human corrections"
    )


class GuardrailsSection(BaseModel):
    """Security guardrails configuration."""

    model_config = ConfigDict(extra="forbid")

    detect_injection: bool = Field(True, description="Enable prompt injection detection")
    detect_pii: bool = Field(True, description="Enable PII detection and redaction")
    detect_credentials: bool = Field(True, description="Enable credential detection")
    detect_toxicity: bool = Field(False, description="Enable toxicity detection")
    injection_threshold: float = Field(
        0.85, ge=0.0, le=1.0, description="Injection confidence threshold"
    )
    warmup: bool = Field(True, description="Pre-load ML models at startup")


class IdentityConfig(BaseModel):
    """Agent identity configuration for ``.superagent`` files.

    Gives the agent a stable, traceable identity (see
    :mod:`promptise.identity`). A **local** identity (``provider: local``)
    needs only an ``agent_id`` and tags the agent's actions for
    attribution. A **verifiable** identity (any cloud provider, or
    ``auto``) additionally mints a signed credential the agent presents to
    the MCP servers it calls, so they can authenticate and attribute it.

    All string fields support ``${ENV_VAR}`` resolution. Fields that do not
    apply to the chosen ``provider`` are ignored.

    Examples:
        Local (attribution only)::

            identity:
              provider: local
              agent_id: billing-bot
              owner: payments
              labels: {env: prod}

        Verifiable via Microsoft Entra::

            identity:
              provider: entra
              agent_id: billing-bot
              client_id: ${AZURE_CLIENT_ID}
              resource: api://my-mcp-server
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["local", "entra", "aws", "gcp", "spiffe", "oidc", "auto"] = Field(
        "local",
        description=(
            "Identity backing. 'local' = attribution-only (no credential); "
            "'entra'/'aws'/'gcp'/'spiffe'/'oidc' = verifiable via that IdP; "
            "'auto' = detect the platform from the environment."
        ),
    )
    agent_id: str | None = Field(
        None,
        description=(
            "Stable identifier. Required for 'local'; optional for a "
            "verifiable identity, which can derive it from the IdP's "
            "sub/oid claim."
        ),
    )
    name: str | None = Field(None, description="Human-readable display name.")
    owner: str | None = Field(None, description="Owning team or person.")
    labels: dict[str, str] = Field(
        default_factory=dict, description="Free-form key/value metadata."
    )

    # Provider-specific options (all support ${ENV_VAR}).
    mode: str | None = Field(
        None,
        description="Provider mode (entra: auto|imds|projected; aws: "
        "auto|sts|projected; spiffe: auto|file|sdk).",
    )
    client_id: str | None = Field(
        None, description="Entra managed-identity client id (entra IMDS)."
    )
    resource: str | None = Field(
        None, description="Resource/audience the credential targets (entra)."
    )
    region: str | None = Field(None, description="AWS region for STS (aws).")
    audience: str | None = Field(
        None, description="Audience the credential targets (aws/gcp/spiffe)."
    )
    service_account_email: str | None = Field(
        None, description="Attached service account whose identity to request (gcp)."
    )
    socket_path: str | None = Field(
        None, description="SPIFFE Workload API socket (spiffe sdk mode)."
    )
    issuer: str | None = Field(None, description="OIDC issuer URL — required for provider 'oidc'.")
    token_file: str | None = Field(
        None, description="Path to a file holding the JWT (entra/aws/spiffe/oidc)."
    )
    token_env_var: str | None = Field(
        None, description="Env var holding the JWT, re-read each refresh (oidc)."
    )

    @model_validator(mode="after")
    def _validate_provider_requirements(self) -> IdentityConfig:
        """Enforce the per-provider required fields up front."""
        if self.provider == "local" and not (self.agent_id and self.agent_id.strip()):
            raise ValueError(
                "identity.provider 'local' requires 'agent_id' — a local "
                "identity has no IdP to derive its identifier from."
            )
        if self.provider == "oidc":
            if not (self.issuer and self.issuer.strip()):
                raise ValueError(
                    "identity.provider 'oidc' requires 'issuer' (the OIDC "
                    "issuer URL whose JWTs this agent presents)."
                )
            sources = [s for s in (self.token_file, self.token_env_var) if s]
            if len(sources) != 1:
                raise ValueError(
                    "identity.provider 'oidc' requires exactly one of "
                    "'token_file' or 'token_env_var' (the way the issuer's "
                    "JWT reaches this workload)."
                )
        return self

    def to_identity(self) -> AgentIdentity:
        """Build the :class:`~promptise.identity.AgentIdentity` this describes.

        Dispatches to the matching identity factory by ``provider``. Shared
        by every declarative surface (``.superagent`` files and ``.agent``
        runtime manifests) so identity is constructed identically everywhere.

        Environment variables are expected to be already resolved on this
        config by the caller's loader.

        Returns:
            The constructed ``AgentIdentity``.

        Raises:
            ProviderConfigError: If the configuration is invalid for the
                chosen provider (e.g. a local identity with no ``agent_id``).
        """
        from .identity import AgentIdentity

        def _opt(**kwargs: Any) -> dict[str, Any]:
            # Pass only the options that were set, letting the factory apply
            # its own defaults for the rest.
            return {k: v for k, v in kwargs.items() if v is not None}

        labels = self.labels or None
        if self.provider == "local":
            return AgentIdentity(self.agent_id, name=self.name, owner=self.owner, labels=labels)
        if self.provider == "auto":
            return AgentIdentity.auto(
                self.agent_id, name=self.name, owner=self.owner, labels=labels
            )
        if self.provider == "entra":
            return AgentIdentity.from_entra(
                self.agent_id,
                name=self.name,
                owner=self.owner,
                labels=labels,
                **_opt(
                    mode=self.mode,
                    client_id=self.client_id,
                    token_file=self.token_file,
                    resource=self.resource,
                ),
            )
        if self.provider == "aws":
            return AgentIdentity.from_aws(
                self.agent_id,
                name=self.name,
                owner=self.owner,
                labels=labels,
                **_opt(
                    mode=self.mode,
                    region=self.region,
                    token_file=self.token_file,
                    audience=self.audience,
                ),
            )
        if self.provider == "gcp":
            return AgentIdentity.from_gcp(
                self.agent_id,
                name=self.name,
                owner=self.owner,
                labels=labels,
                **_opt(audience=self.audience, service_account_email=self.service_account_email),
            )
        if self.provider == "spiffe":
            return AgentIdentity.from_spiffe(
                self.agent_id,
                name=self.name,
                owner=self.owner,
                labels=labels,
                **_opt(
                    mode=self.mode,
                    token_file=self.token_file,
                    socket_path=self.socket_path,
                    audience=self.audience,
                ),
            )
        # provider == "oidc": issuer + exactly one token source (schema-validated)
        return AgentIdentity.from_oidc(
            self.agent_id,
            issuer=self.issuer,  # type: ignore[arg-type]  # required by validator
            name=self.name,
            owner=self.owner,
            labels=labels,
            **_opt(token_file=self.token_file, token_env_var=self.token_env_var),
        )


class SuperAgentSchema(BaseModel):
    """Root schema for .superagent YAML files.

    This is the top-level schema that validates the entire .superagent file
    structure. It supports versioning for future compatibility and ensures
    at least one of servers, cross_agents, or sandbox is configured.

    Attributes:
        version: Schema version (currently "1.0").
        agent: Agent-level configuration (model, instructions, trace).
        servers: Named MCP server configurations.
        cross_agents: Optional cross-agent references.
        sandbox: Optional sandbox configuration (bool or detailed config).
        memory: Optional memory configuration.
        observability: Optional observability configuration (True or detailed).
        optimize_tools: Optional tool optimization (True, string level, or detailed).

    Examples:
        >>> schema = SuperAgentSchema(
        ...     version="1.0",
        ...     agent=AgentSection(model="openai:gpt-5-mini"),
        ...     servers={"math": HTTPServerConfig(url="http://...")},
        ...     sandbox=True,
        ...     observability=True,
        ...     optimize_tools="semantic",
        ... )
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal["1.0"] = Field("1.0", description="Schema version")
    agent: AgentSection = Field(..., description="Agent configuration")
    identity: IdentityConfig | None = Field(
        None,
        description=(
            "Agent identity — who is acting. A local identity tags the "
            "agent's actions for attribution; a verifiable identity (Entra, "
            "AWS, GCP, SPIFFE, OIDC) also presents a signed credential to "
            "the MCP servers it calls."
        ),
    )
    servers: dict[str, ServerConfig] = Field(
        default_factory=dict, description="Named MCP server configurations"
    )
    cross_agents: dict[str, CrossAgentConfig] | None = Field(
        None, description="Cross-agent references"
    )
    sandbox: bool | SandboxConfigSection | None = Field(
        None, description="Sandbox configuration (True for defaults, or detailed config)"
    )
    memory: MemorySection | None = Field(
        None, description="Agent memory configuration for persistent knowledge"
    )
    observability: bool | ObservabilitySection | None = Field(
        None, description="Observability config (True for defaults, or detailed)"
    )
    optimize_tools: bool | str | ToolOptimizationSection | None = Field(
        None,
        description=("Tool optimization (True or 'semantic' for defaults, or detailed config)"),
    )
    cache: bool | CacheSection | None = Field(
        None,
        description=(
            "Semantic caching (True for defaults, or detailed config). "
            "Caches LLM responses for similar queries to reduce API costs."
        ),
    )
    approval: ApprovalSection | None = Field(
        None,
        description=(
            "Human-in-the-loop approval for sensitive tool calls. "
            "Pauses agent execution and awaits human decision."
        ),
    )
    events: EventsSection | None = Field(
        None,
        description=(
            "Webhook and event notification sinks. "
            "Emits structured notifications on invocation, tool, guardrail, "
            "budget, health, mission, and process events."
        ),
    )
    adaptive: bool | AdaptiveSection | None = Field(
        None,
        description=(
            "Adaptive strategy learning. Agents learn from failures "
            "and adjust approach across invocations."
        ),
    )
    guardrails: bool | GuardrailsSection | None = Field(
        None,
        description=(
            "Security guardrails. Blocks prompt injection, redacts PII, "
            "detects credentials. True for defaults."
        ),
    )
    max_invocation_time: float = Field(
        0,
        ge=0,
        description="Max seconds per invocation (0 = unlimited).",
    )

    @model_validator(mode="after")
    def validate_has_config(self) -> SuperAgentSchema:
        """Ensure at least servers, cross_agents, or sandbox is configured."""
        if not self.servers and not self.cross_agents and not self.sandbox:
            raise ValueError(
                "At least one of 'servers', 'cross_agents', or 'sandbox' must be configured"
            )
        return self
