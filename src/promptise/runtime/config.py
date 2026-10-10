"""Configuration schemas for the agent runtime.

All configuration is expressed as Pydantic ``BaseModel`` subclasses with
sensible defaults.  Atomic configs (:class:`TriggerConfig`,
:class:`JournalConfig`, etc.) are composed into :class:`ProcessConfig`
(single process) and :class:`RuntimeConfig` (multi-process manager).

Two preset configurations are provided for common environments:

* :data:`DEFAULT_DEVELOPMENT_CONFIG` — local-only defaults.
* :data:`DEFAULT_PRODUCTION_CONFIG` — distributed enabled.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ---------------------------------------------------------------------------
# Execution modes
# ---------------------------------------------------------------------------


class ExecutionMode(str, Enum):
    """Agent execution mode.

    Attributes:
        STRICT: Agent cannot modify itself.  Current behavior, default.
        OPEN: Agent can adapt identity, memory, tools, and triggers at
            runtime — a self-evolving, autonomous agent.
    """

    STRICT = "strict"
    OPEN = "open"


class OpenModeConfig(BaseModel):
    """Guardrails for open execution mode.

    Controls what the agent is allowed to self-modify and how far.
    Only takes effect when ``ProcessConfig.execution_mode`` is
    :attr:`ExecutionMode.OPEN`.

    Attributes:
        allow_identity_change: Agent can modify its own instructions.
        allow_tool_creation: Agent can define new Python tools.
        allow_mcp_connect: Agent can connect to new MCP servers.
        allow_trigger_management: Agent can add/remove triggers.
        allow_memory_management: Agent can explicitly store/search/forget.
        max_custom_tools: Max number of agent-created tools.
        max_dynamic_triggers: Max number of dynamically added triggers.
        max_instruction_length: Max character length for modified instructions.
        max_rebuilds: Max agent rebuilds per lifetime (``None`` = unlimited).
        allowed_mcp_urls: Whitelist of MCP server URLs (empty = any).
        sandbox_custom_tools: Execute agent-written tools in sandbox.
        allow_process_spawn: Agent can spawn new processes within the
            runtime.  Defaults to ``False``.
        max_spawned_processes: Max number of processes this agent can spawn.
    """

    allow_identity_change: bool = Field(True, description="Agent can modify its own instructions")
    allow_tool_creation: bool = Field(True, description="Agent can define new Python tools")
    allow_mcp_connect: bool = Field(True, description="Agent can connect to new MCP servers")
    allow_trigger_management: bool = Field(True, description="Agent can add/remove triggers")
    allow_memory_management: bool = Field(
        True, description="Agent can explicitly store/search/forget memories"
    )
    max_custom_tools: int = Field(20, ge=0, description="Max number of agent-created tools")
    max_dynamic_triggers: int = Field(
        10, ge=0, description="Max number of dynamically added triggers"
    )
    max_instruction_length: int = Field(
        10_000, ge=100, description="Max character length for modified instructions"
    )
    max_rebuilds: int | None = Field(
        None, description="Max agent rebuilds per lifetime (None = unlimited)"
    )
    allowed_mcp_urls: list[str] = Field(
        default_factory=list,
        description="Whitelist of MCP server URLs (empty = any)",
    )
    sandbox_custom_tools: bool = Field(True, description="Execute agent-written tools in sandbox")
    allow_process_spawn: bool = Field(
        False,
        description=(
            "Agent can spawn new processes within the runtime. "
            "Defaults to False — enable explicitly when needed."
        ),
    )
    max_spawned_processes: int = Field(
        3, ge=0, description="Max number of processes this agent can spawn"
    )


# ---------------------------------------------------------------------------
# Atomic configs
# ---------------------------------------------------------------------------


class TriggerConfig(BaseModel):
    """Configuration for a single trigger.

    Each trigger ``type`` requires a different subset of fields.  The
    :meth:`_validate_trigger_fields` model validator enforces this for
    built-in types.  Custom trigger types use the ``custom_config`` dict.

    Attributes:
        type: Trigger type (built-in or custom-registered via
            :func:`~promptise.runtime.triggers.register_trigger_type`).
        cron_expression: Cron expression (``cron`` type only).
        webhook_path: URL path for the webhook endpoint (``webhook`` only).
        webhook_port: Listening port (``webhook`` only).
        watch_path: Directory to watch (``file_watch`` only).
        watch_patterns: Glob patterns (``file_watch`` only).
        watch_events: Filesystem events to react to (``file_watch`` only).
        event_type: EventBus event type string (``event`` only).
        event_source: Optional source filter (``event`` only).
        topic: MessageBroker topic (``message`` only).
        custom_config: Additional key-value configuration for custom
            trigger types.
        filter_expression: Cheap pre-filter (all types, optional).
    """

    type: str = Field(..., description="Trigger type (built-in or custom-registered)")

    # -- Cron --
    cron_expression: str | None = Field(None, description="Cron expression (e.g. '*/5 * * * *')")

    # -- Webhook --
    webhook_path: str = Field("/webhook", description="Webhook URL path")
    webhook_port: int = Field(9090, gt=1024, le=65535, description="Webhook listen port")

    # -- File watch --
    watch_path: str | None = Field(None, description="Directory to watch for changes")
    watch_patterns: list[str] = Field(
        default_factory=lambda: ["*"],
        description="Glob patterns for file watch",
    )
    watch_events: list[str] = Field(
        default_factory=lambda: ["created", "modified"],
        description="Filesystem events to react to",
    )

    # -- Event --
    event_type: str | None = Field(None, description="EventBus event type to subscribe to")
    event_source: str | None = Field(None, description="Optional event source filter")

    # -- Message --
    topic: str | None = Field(None, description="MessageBroker topic to subscribe to")

    # -- Custom --
    custom_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional configuration for custom trigger types",
    )

    # -- Common --
    filter_expression: str | None = Field(
        None,
        description="Cheap filter expression evaluated before LLM invocation",
    )

    @model_validator(mode="after")
    def _validate_trigger_fields(self) -> TriggerConfig:
        """Enforce that type-specific required fields are provided."""
        if self.type == "cron" and not self.cron_expression:
            raise ValueError("Cron trigger requires 'cron_expression'")
        if self.type == "file_watch" and not self.watch_path:
            raise ValueError("File watch trigger requires 'watch_path'")
        if self.type == "event" and not self.event_type:
            raise ValueError("Event trigger requires 'event_type'")
        if self.type == "message" and not self.topic:
            raise ValueError("Message trigger requires 'topic'")
        return self


class JournalConfig(BaseModel):
    """Journal (durable audit log) configuration.

    Attributes:
        level: Detail level — ``none`` (disabled), ``checkpoint``
            (state snapshots per cycle), or ``full`` (every side effect).
        backend: Storage backend.
        path: Directory for the file backend.
    """

    level: Literal["none", "checkpoint", "full"] = Field(
        "checkpoint", description="Journal detail level"
    )
    backend: Literal["file", "memory"] = Field("file", description="Journal storage backend")
    path: str = Field(".promptise/journal", description="Base directory for journal files")


class ContextConfig(BaseModel):
    """AgentContext configuration.

    Attributes:
        writable_keys: State keys the agent is allowed to write to.
            Empty list means **all** keys are writable.
        memory_provider: Memory provider type (or ``None`` to disable).
        memory_max: Max memories to inject per invocation.
        memory_min_score: Min relevance score for memory injection.
        memory_timeout: Seconds to wait for a memory search or auto-store
            write (a slower write finishes in the background).
        memory_auto_store: Automatically store exchanges in long-term memory.
        memory_collection: Collection name for ChromaDB backend.
        memory_persist_directory: Persist directory for ChromaDB.
        memory_user_id: User ID for Mem0 scoping.
        conversation_max_messages: Max messages in conversation buffer
            (short-term memory).
        file_mounts: Mapping of logical name → filesystem path.
        env_prefix: Only expose environment variables with this prefix.
        initial_state: Pre-populated key-value state.
    """

    writable_keys: list[str] = Field(
        default_factory=list,
        description="State keys the agent can write (empty = all writable)",
    )

    # -- Long-term memory --
    memory_provider: Literal["in_memory", "chroma", "mem0"] | None = Field(
        None, description="Memory provider type (None = disabled)"
    )
    memory_max: int = Field(5, ge=1, description="Max memories per invocation")
    memory_min_score: float = Field(0.0, ge=0.0, le=1.0, description="Min relevance score")
    memory_timeout: float = Field(
        5.0, gt=0, description="Seconds to wait for memory search / auto-store"
    )
    memory_auto_store: bool = Field(
        False,
        description="Automatically store exchanges in long-term memory",
    )
    memory_collection: str = Field("agent_memory", description="Collection name for ChromaDB")
    memory_persist_directory: str | None = Field(None, description="Persist directory for ChromaDB")
    memory_user_id: str = Field("default", description="User ID for Mem0 scoping")

    # -- Short-term memory --
    conversation_max_messages: int = Field(
        100,
        ge=0,
        description="Max messages in conversation buffer (0 = disabled)",
    )

    # -- Environment --
    file_mounts: dict[str, str] = Field(
        default_factory=dict, description="Logical name → filesystem path"
    )
    env_prefix: str = Field("AGENT_", description="Prefix for exposed environment variables")
    initial_state: dict[str, Any] = Field(default_factory=dict, description="Pre-populated state")


# ---------------------------------------------------------------------------
# Escalation & governance configs
# ---------------------------------------------------------------------------


class EscalationTarget(BaseModel):
    """Where to send escalation notifications.

    Attributes:
        webhook_url: HTTP endpoint to POST a JSON payload to.
        event_type: EventBus event type to emit.
    """

    webhook_url: str | None = Field(None, description="Webhook endpoint URL")
    event_type: str | None = Field(None, description="EventBus event type")


class ToolCostAnnotation(BaseModel):
    """Cost and reversibility annotation for a tool.

    Attributes:
        cost_weight: Abstract cost units per invocation (default 1.0).
        irreversible: Whether the tool performs an irreversible action.
    """

    cost_weight: float = Field(1.0, gt=0, description="Cost weight per call")
    irreversible: bool = Field(False, description="Irreversible action flag")


class SecretScopeConfig(BaseModel):
    """Per-process secret scoping configuration.

    Attributes:
        enabled: Enable secret scoping for this process.
        secrets: Secret name to value or ``${ENV_VAR}`` reference.
        default_ttl: Default TTL in seconds (``None`` = no expiry).
        ttls: Per-secret TTL overrides in seconds.
        revoke_on_stop: Zero-fill and remove all secrets on stop.
    """

    enabled: bool = Field(False, description="Enable secret scoping")
    secrets: dict[str, str] = Field(
        default_factory=dict, description="Secret name → value or ${ENV_VAR}"
    )
    default_ttl: float | None = Field(None, description="Default TTL in seconds (None = no expiry)")
    ttls: dict[str, float] = Field(default_factory=dict, description="Per-secret TTL overrides")
    revoke_on_stop: bool = Field(True, description="Revoke all secrets on stop")


class BudgetConfig(BaseModel):
    """Autonomy budget configuration.

    Attributes:
        enabled: Enable budget tracking for this process.
        max_tool_calls_per_run: Max tool calls per invocation.
        max_llm_turns_per_run: Max LLM turns per invocation.
        max_cost_per_run: Max cost units per invocation.
        max_irreversible_per_run: Max irreversible actions per invocation.
        max_tool_calls_per_day: Max tool calls in a day.
        max_runs_per_day: Max invocations in a day.
        max_cost_per_day: Max cost units in a day.
        tool_costs: Per-tool cost annotations.
        on_exceeded: Action on budget violation.
        escalation: Escalation target for violations.
        daily_reset_hour_utc: UTC hour to reset daily counters.
        inject_remaining: Inject remaining budget into agent context.
    """

    enabled: bool = Field(False, description="Enable budget tracking")
    max_tool_calls_per_run: int | None = Field(None, ge=1)
    max_llm_turns_per_run: int | None = Field(None, ge=1)
    max_cost_per_run: float | None = Field(None, gt=0)
    max_irreversible_per_run: int | None = Field(None, ge=0)
    max_tool_calls_per_day: int | None = Field(None, ge=1)
    max_runs_per_day: int | None = Field(None, ge=1)
    max_cost_per_day: float | None = Field(None, gt=0)
    tool_costs: dict[str, ToolCostAnnotation] = Field(default_factory=dict)
    on_exceeded: Literal["pause", "stop", "escalate"] = Field("pause")
    escalation: EscalationTarget | None = Field(None)
    daily_reset_hour_utc: int = Field(0, ge=0, le=23)
    inject_remaining: bool = Field(True)


class HealthConfig(BaseModel):
    """Behavioral health monitoring configuration.

    Attributes:
        enabled: Enable health monitoring for this process.
        stuck_threshold: Same tool+args N times = stuck.
        loop_window: Tool calls to examine for loop detection.
        loop_min_repeats: Min pattern repeats to trigger.
        empty_threshold: N consecutive short responses = anomaly.
        empty_max_chars: Below this = trivial response.
        error_window: Sliding window for error rate.
        error_rate_threshold: Error rate above this triggers anomaly.
        on_anomaly: Action when anomaly is detected.
        cooldown: Seconds between same anomaly type.
        escalation: Escalation target for anomalies.
    """

    enabled: bool = Field(False, description="Enable health monitoring")
    stuck_threshold: int = Field(3, ge=2)
    loop_window: int = Field(20, ge=4)
    loop_min_repeats: int = Field(2, ge=2)
    empty_threshold: int = Field(3, ge=2)
    empty_max_chars: int = Field(10, ge=0)
    error_window: int = Field(10, ge=2)
    error_rate_threshold: float = Field(0.5, gt=0, le=1.0)
    on_anomaly: Literal["log", "pause", "escalate"] = Field("log")
    cooldown: float = Field(300.0, ge=0)
    escalation: EscalationTarget | None = Field(None)


class MissionConfig(BaseModel):
    """Mission-oriented process configuration.

    Attributes:
        enabled: Enable mission tracking for this process.
        objective: What the agent is trying to achieve.
        success_criteria: How to judge completion.
        eval_model: LLM for evaluation (None = use process model).
        eval_every: Evaluate every N invocations.
        confidence_threshold: Below this = pause + escalate.
        timeout_hours: Max hours (0 = no timeout).
        max_invocations: Max invocations (0 = unlimited).
        auto_complete: Stop process when mission achieved.
        on_complete: Behavior when mission completes.
        escalation: Escalation target for low confidence.
    """

    enabled: bool = Field(False, description="Enable mission tracking")
    objective: str = Field("", description="Mission objective")
    success_criteria: str = Field("", description="Success criteria")
    eval_model: str | None = Field(None)
    eval_every: int = Field(1, ge=1)
    confidence_threshold: float = Field(0.7, ge=0, le=1.0)
    timeout_hours: float = Field(0.0, ge=0)
    max_invocations: int = Field(0, ge=0)
    auto_complete: bool = Field(True)
    on_complete: Literal["stop", "continue", "suspend"] = Field("stop")
    escalation: EscalationTarget | None = Field(None)


# ---------------------------------------------------------------------------
# Composite configs
# ---------------------------------------------------------------------------


class InboxConfig(BaseModel):
    """Configuration for the human-to-agent message inbox.

    Attributes:
        enabled: Enable the message inbox.
        max_messages: Maximum messages in the inbox at once.
        max_message_length: Maximum characters per message.
        default_ttl: Default time-to-live in seconds (0 = no expiry).
        max_ttl: Maximum allowed TTL in seconds.
        rate_limit_per_sender: Maximum messages per sender per hour.
        scan_with_guardrails: Scan incoming messages for prompt injection.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(False, description="Enable the message inbox")
    max_messages: int = Field(50, gt=0, description="Max inbox size")
    max_message_length: int = Field(2000, gt=0, description="Max chars per message")
    default_ttl: float = Field(3600, ge=0, description="Default TTL in seconds")
    max_ttl: float = Field(86400, ge=0, description="Max TTL in seconds")
    rate_limit_per_sender: int = Field(20, gt=0, description="Max messages per sender per hour")
    scan_with_guardrails: bool = Field(True, description="Scan messages for injection")


class ProcessConfig(BaseModel):
    """Configuration for a single agent process.

    Composes all atomic configs and adds process-level settings.

    Attributes:
        model: LLM model identifier (e.g. ``openai:gpt-5-mini``).
        instructions: System prompt for the agent.
        servers: MCP server specifications (same format as ``.superagent``).
        triggers: Trigger configurations.
        journal: Journal configuration.
        context: AgentContext configuration.
        concurrency: Max concurrent trigger invocations.
        heartbeat_interval: Heartbeat period in seconds.
        idle_timeout: Seconds of inactivity before suspending (0 = never).
        max_lifetime: Max process lifetime in seconds (0 = unlimited).
        max_consecutive_failures: Consecutive failures before FAILED state.
        restart_policy: When to restart a failed process.
        max_restarts: Max restart attempts (for ``on_failure`` / ``always``).
    """

    model: str = Field("openai:gpt-5-mini", description="LLM model ID")
    instructions: str | None = Field(None, description="System prompt")
    execution_mode: ExecutionMode = Field(
        ExecutionMode.STRICT,
        description="Agent execution mode: strict (immutable) or open (self-modifying)",
    )
    open_mode: OpenModeConfig = Field(
        default_factory=OpenModeConfig,
        description="Guardrails for open execution mode (ignored in strict mode)",
    )
    servers: dict[str, Any] = Field(default_factory=dict, description="MCP server specifications")
    triggers: list[TriggerConfig] = Field(
        default_factory=list, description="Trigger configurations"
    )
    journal: JournalConfig = Field(
        default_factory=JournalConfig, description="Journal configuration"
    )
    context: ContextConfig = Field(
        default_factory=ContextConfig, description="AgentContext configuration"
    )
    concurrency: int = Field(1, ge=1, le=100, description="Max concurrent invocations")
    heartbeat_interval: float = Field(10.0, gt=0, description="Heartbeat interval (seconds)")
    idle_timeout: float = Field(0.0, ge=0, description="Idle timeout before suspend (0 = never)")
    max_lifetime: float = Field(0.0, ge=0, description="Max lifetime in seconds (0 = unlimited)")
    max_consecutive_failures: int = Field(
        3, gt=0, description="Consecutive failures before FAILED state"
    )
    restart_policy: Literal["always", "on_failure", "never"] = Field(
        "never", description="Restart policy for failed processes"
    )
    max_restarts: int = Field(3, ge=0, description="Max restart attempts")

    # -- Governance (all opt-in, zero overhead when disabled) --
    secrets: SecretScopeConfig = Field(
        default_factory=SecretScopeConfig,
        description="Per-process secret scoping",
    )
    budget: BudgetConfig = Field(
        default_factory=BudgetConfig,
        description="Autonomy budget limits",
    )
    health: HealthConfig = Field(
        default_factory=HealthConfig,
        description="Behavioral health monitoring",
    )
    mission: MissionConfig = Field(
        default_factory=MissionConfig,
        description="Mission-oriented process configuration",
    )

    # -- Message inbox (opt-in) --
    inbox: InboxConfig = Field(
        default_factory=InboxConfig,
        description="Human-to-agent message inbox configuration",
    )

    # -- Human-in-the-loop approval (opt-in) --
    approval: Any | None = Field(
        None,
        description=(
            "ApprovalPolicy instance or dict for human-in-the-loop "
            "tool call approval. When set, matching tools require human "
            "approval before execution."
        ),
    )

    # -- Agent identity (opt-in) --
    identity: Any | None = Field(
        None,
        description=(
            "AgentIdentity instance — who this process is. Attributes the "
            "process's actions to the agent and, when verifiable, presents "
            "its credential to the MCP servers it calls. Built from the "
            "manifest's 'identity' block."
        ),
    )


class DistributedConfig(BaseModel):
    """Distributed runtime coordination configuration.

    Attributes:
        enabled: Enable distributed mode.
        coordinator_url: URL of the coordinator node.
        transport_port: Port for the management HTTP transport.
        discovery_method: How runtime nodes discover each other.
        heartbeat_interval: Health check interval for node monitoring.
    """

    enabled: bool = Field(False, description="Enable distributed mode")
    coordinator_url: str | None = Field(None, description="Coordinator node URL")
    transport_port: int = Field(9100, gt=1024, le=65535, description="Management transport port")
    discovery_method: Literal["registry", "multicast"] = Field(
        "registry", description="Node discovery method"
    )
    heartbeat_interval: float = Field(
        15.0, gt=0, description="Node health check interval (seconds)"
    )


class RuntimeConfig(BaseModel):
    """Top-level runtime configuration (composite).

    Aggregates per-process configs and global settings.

    Attributes:
        processes: Named process configurations.
        distributed: Distributed coordination settings.
    """

    processes: dict[str, ProcessConfig] = Field(
        default_factory=dict, description="Named process configurations"
    )
    distributed: DistributedConfig = Field(
        default_factory=DistributedConfig,
        description="Distributed coordination",
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeConfig:
        """Deserialize from a dict."""
        return cls.model_validate(data)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

DEFAULT_DEVELOPMENT_CONFIG = RuntimeConfig(
    distributed=DistributedConfig(enabled=False),
)

DEFAULT_PRODUCTION_CONFIG = RuntimeConfig(
    distributed=DistributedConfig(enabled=True),
)
