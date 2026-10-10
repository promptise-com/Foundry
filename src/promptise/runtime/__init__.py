"""Agent Runtime for Promptise Foundry.

Long-running agent processes with triggers, journals,
and distributed coordination.  The runtime is the operating system for
your AI agents — it wraps a :class:`~promptise.agent.PromptiseAgent`
and gives it lifecycle management, scheduling, persistent state, crash
recovery, and inter-agent communication.

Quick start::

    from promptise.runtime import AgentProcess, ProcessConfig, TriggerConfig

    process = AgentProcess(
        name="data-watcher",
        config=ProcessConfig(
            model="openai:gpt-5-mini",
            instructions="You monitor data pipelines.",
            triggers=[
                TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
            ],
        ),
    )
    await process.start()
    # Process runs until stopped …
    await process.stop()
"""

from __future__ import annotations

# -- Dashboard --
from ._dashboard import (
    RuntimeDashboard,
    RuntimeDashboardState,
    RuntimeDataCollector,
)

# -- Orchestration API --
from .api import OrchestrationAPI
from .api_client import OrchestrationClient

# -- Governance --
from .budget import BudgetEnforcer, BudgetState, BudgetViolation

# -- Configuration --
from .config import (
    DEFAULT_DEVELOPMENT_CONFIG,
    DEFAULT_PRODUCTION_CONFIG,
    BudgetConfig,
    ContextConfig,
    DistributedConfig,
    EscalationTarget,
    ExecutionMode,
    HealthConfig,
    InboxConfig,
    JournalConfig,
    MissionConfig,
    OpenModeConfig,
    ProcessConfig,
    RuntimeConfig,
    SecretScopeConfig,
    ToolCostAnnotation,
    TriggerConfig,
)

# -- Context --
from .context import AgentContext, StateEntry

# -- Short-term Memory --
from .conversation import ConversationBuffer

# -- Distributed --
from .distributed import (
    DiscoveredNode,
    NodeInfo,
    RegistryDiscovery,
    RuntimeCoordinator,
    StaticDiscovery,
)

# -- Exceptions --
from .exceptions import (
    BudgetExceededError,
    JournalError,
    ManifestError,
    ManifestValidationError,
    ProcessStateError,
    RuntimeBaseError,
    TriggerError,
)
from .health import Anomaly, AnomalyType, HealthMonitor

# -- Hooks --
from .hooks import (
    DispatchResult,
    HookBlocked,
    HookCallable,
    HookContext,
    HookEvent,
    HookManager,
)

# -- Message Inbox --
from .inbox import InboxMessage, InboxResponse, MessageInbox, MessageType

# -- Journal --
from .journal import (
    FileJournal,
    InMemoryJournal,
    JournalEntry,
    JournalLevel,
    JournalProvider,
    ReplayEngine,
    RewindEngine,
    RewindMode,
    RewindPlan,
    RewindResult,
)

# -- Lifecycle --
from .lifecycle import (
    VALID_TRANSITIONS,
    ProcessLifecycle,
    ProcessState,
    ProcessTransition,
    StateError,
)

# -- Manifest --
from .manifest import (
    AgentManifestSchema,
    load_manifest,
    manifest_to_process_config,
    save_manifest,
    validate_manifest,
)
from .mission import MissionEvaluation, MissionEvidence, MissionState, MissionTracker

# -- Process --
from .process import AgentProcess

# -- Runtime --
from .runtime import AgentRuntime
from .secrets import SecretScope
from .shell_hook import ShellHook, ShellHookResult

# -- Triggers --
from .triggers import (
    create_trigger,
    register_trigger_type,
    registered_trigger_types,
)
from .triggers.base import BaseTrigger, TriggerEvent
from .triggers.cron import CronTrigger
from .triggers.event import EventTrigger, MessageTrigger

__all__ = [
    # Lifecycle
    "ProcessState",
    "ProcessTransition",
    "ProcessLifecycle",
    "StateError",
    "VALID_TRANSITIONS",
    # Config
    "ExecutionMode",
    "OpenModeConfig",
    "TriggerConfig",
    "JournalConfig",
    "ContextConfig",
    "ProcessConfig",
    "DistributedConfig",
    "RuntimeConfig",
    "EscalationTarget",
    "ToolCostAnnotation",
    "SecretScopeConfig",
    "BudgetConfig",
    "InboxConfig",
    # Orchestration API
    "OrchestrationAPI",
    "OrchestrationClient",
    # Message Inbox
    "MessageInbox",
    "InboxMessage",
    "InboxResponse",
    "MessageType",
    "HealthConfig",
    "MissionConfig",
    "DEFAULT_DEVELOPMENT_CONFIG",
    "DEFAULT_PRODUCTION_CONFIG",
    # Governance
    "SecretScope",
    "BudgetState",
    "BudgetEnforcer",
    "BudgetViolation",
    "HealthMonitor",
    "Anomaly",
    "AnomalyType",
    "MissionTracker",
    "MissionEvidence",
    "MissionEvaluation",
    "MissionState",
    # Context
    "AgentContext",
    "StateEntry",
    # Conversation
    "ConversationBuffer",
    # Exceptions
    "RuntimeBaseError",
    "BudgetExceededError",
    "ProcessStateError",
    "ManifestError",
    "ManifestValidationError",
    "TriggerError",
    "JournalError",
    # Process
    "AgentProcess",
    # Triggers
    "BaseTrigger",
    "TriggerEvent",
    "CronTrigger",
    "EventTrigger",
    "MessageTrigger",
    "create_trigger",
    "register_trigger_type",
    "registered_trigger_types",
    # Journal
    "JournalProvider",
    "JournalEntry",
    "JournalLevel",
    "FileJournal",
    "InMemoryJournal",
    "ReplayEngine",
    "RewindEngine",
    "RewindMode",
    "RewindPlan",
    "RewindResult",
    # Manifest
    "AgentManifestSchema",
    "load_manifest",
    "save_manifest",
    "validate_manifest",
    "manifest_to_process_config",
    # Runtime
    "AgentRuntime",
    # Dashboard
    "RuntimeDashboard",
    "RuntimeDashboardState",
    "RuntimeDataCollector",
    # Distributed
    "RuntimeCoordinator",
    "NodeInfo",
    "DiscoveredNode",
    "StaticDiscovery",
    "RegistryDiscovery",
    # Hooks
    "HookEvent",
    "HookContext",
    "HookCallable",
    "HookManager",
    "HookBlocked",
    "DispatchResult",
    "ShellHook",
    "ShellHookResult",
]
