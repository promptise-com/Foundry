"""Schema and loader for ``.agent`` manifest files.

The ``.agent`` format is a YAML file that declaratively defines an agent
process: model, instructions, servers, triggers, world state,
memory, and config.

Example ``.agent`` file::

    version: "1.0"
    name: data-watcher
    model: openai:gpt-5-mini
    instructions: |
      You monitor data pipelines and alert on anomalies.

    servers:
      data_tools:
        type: http
        url: http://localhost:8000/mcp

    triggers:
      - type: cron
        cron_expression: "*/5 * * * *"

    world:
      pipeline_status: healthy

    config:
      concurrency: 2
      heartbeat_interval: 30

Usage::

    from promptise.runtime.manifest import load_manifest

    manifest = load_manifest("agents/watcher.agent")
    config = manifest_to_process_config(manifest)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from promptise.config import HTTPServerSpec, StdioServerSpec

from .config import (
    BudgetConfig,
    ContextConfig,
    ExecutionMode,
    HealthConfig,
    JournalConfig,
    MissionConfig,
    OpenModeConfig,
    ProcessConfig,
    SecretScopeConfig,
    TriggerConfig,
)
from .exceptions import ManifestError, ManifestValidationError

logger = logging.getLogger(__name__)


class AgentManifestSchema(BaseModel):
    """Root schema for ``.agent`` manifest files.

    Attributes:
        version: Schema version (currently ``"1.0"``).
        name: Process name (unique identifier).
        model: LLM model ID.
        instructions: System prompt.
        servers: MCP server specifications.
        triggers: Trigger configurations (list of dicts).
        world: Initial world state (key-value).
        memory: Memory provider configuration.
        journal: Journal configuration.
        config: Additional process config overrides.
        entrypoint: Optional Python module for custom hooks.
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal["1.0"] = Field("1.0", description="Schema version")
    name: str = Field(..., description="Process name")
    model: str = Field("openai:gpt-5-mini", description="LLM model ID")
    instructions: str | None = Field(None, description="System prompt")
    identity: dict[str, Any] | None = Field(
        None,
        description=(
            "Agent identity — who is acting. Same shape as the .superagent "
            "'identity' block: provider (local/entra/aws/gcp/spiffe/oidc/"
            "auto), agent_id, and provider options."
        ),
    )
    servers: dict[str, Any] = Field(default_factory=dict, description="MCP server specifications")
    triggers: list[dict[str, Any]] = Field(
        default_factory=list, description="Trigger configurations"
    )
    world: dict[str, Any] = Field(default_factory=dict, description="Initial world state")
    memory: dict[str, Any] | None = Field(None, description="Memory provider config")
    journal: dict[str, Any] | None = Field(None, description="Journal configuration")
    config: dict[str, Any] = Field(default_factory=dict, description="Process config overrides")
    execution_mode: str | None = Field(None, description="Execution mode: 'strict' or 'open'")
    open_mode: dict[str, Any] | None = Field(None, description="Open mode guardrails configuration")
    entrypoint: str | None = Field(None, description="Python module for custom hooks")
    secrets: dict[str, Any] | None = Field(
        None, description="Per-process secret scoping configuration"
    )
    budget: dict[str, Any] | None = Field(None, description="Autonomy budget configuration")
    health: dict[str, Any] | None = Field(
        None, description="Behavioral health monitoring configuration"
    )
    mission: dict[str, Any] | None = Field(
        None, description="Mission-oriented process configuration"
    )
    context: dict[str, Any] | None = Field(
        None,
        description=(
            "Context config overrides: writable_keys, file_mounts, "
            "env_prefix, conversation_max_messages"
        ),
    )

    @model_validator(mode="after")
    def _validate_manifest(self) -> AgentManifestSchema:
        """Validate cross-field constraints."""
        # Validate trigger dicts have a 'type' field
        for i, trigger in enumerate(self.triggers):
            if "type" not in trigger:
                raise ValueError(f"Trigger at index {i} is missing required 'type' field")
        return self


def load_manifest(path: str | Path) -> AgentManifestSchema:
    """Load and validate a ``.agent`` manifest file.

    Args:
        path: Path to the ``.agent`` YAML file.

    Returns:
        Validated :class:`AgentManifestSchema`.

    Raises:
        ManifestError: If the file cannot be read or parsed.
        ManifestValidationError: If schema validation fails.
    """
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"Manifest not found: {path}")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ManifestError(f"Cannot read manifest: {exc}") from exc

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ManifestError(f"Invalid YAML in {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ManifestError(f"Manifest must be a YAML mapping, got {type(data).__name__}")

    # Resolve environment variables if env_resolver is available
    try:
        from promptise.env_resolver import resolve_env_in_dict

        data = resolve_env_in_dict(data)
    except ImportError:
        pass

    try:
        return AgentManifestSchema.model_validate(data)
    except ValidationError as exc:
        raise ManifestValidationError(
            f"Manifest validation failed: {exc}",
            errors=exc.errors(),
            file_path=str(path),
        ) from exc


def save_manifest(manifest: AgentManifestSchema, path: str | Path) -> None:
    """Write a manifest to a YAML file.

    Args:
        manifest: The manifest to save.
        path: Destination file path.
    """
    path = Path(path)
    data = manifest.model_dump(mode="json", exclude_none=True)
    path.write_text(
        yaml.dump(data, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )


def validate_manifest(path: str | Path) -> list[str]:
    """Validate a manifest file, returning a list of warnings.

    Args:
        path: Path to the manifest file.

    Returns:
        List of warning messages (empty if fully valid).

    Raises:
        ManifestError: If the file cannot be loaded.
        ManifestValidationError: If the schema is invalid.
    """
    warnings: list[str] = []
    manifest = load_manifest(path)

    if not manifest.instructions:
        warnings.append("No instructions provided — agent will use default behavior")
    if not manifest.triggers:
        warnings.append("No triggers defined — agent will only respond to manual events")
    if not manifest.servers:
        warnings.append("No MCP servers configured — agent has no tools")

    return warnings


def manifest_to_process_config(
    manifest: AgentManifestSchema,
) -> ProcessConfig:
    """Convert a manifest into a :class:`ProcessConfig`.

    Args:
        manifest: Validated manifest schema.

    Returns:
        A :class:`ProcessConfig` ready to pass to :class:`AgentProcess`.
    """
    # Parse triggers
    triggers: list[TriggerConfig] = []
    for trigger_data in manifest.triggers:
        triggers.append(TriggerConfig.model_validate(trigger_data))

    # Parse journal
    journal = JournalConfig()
    if manifest.journal:
        journal = JournalConfig.model_validate(manifest.journal)

    # Parse context
    context_kwargs: dict[str, Any] = {"initial_state": manifest.world}
    if manifest.memory:
        provider_type = manifest.memory.get("provider")
        if provider_type:
            context_kwargs["memory_provider"] = provider_type
        if "collection" in manifest.memory:
            context_kwargs["memory_collection"] = manifest.memory["collection"]
        if "persist_directory" in manifest.memory:
            context_kwargs["memory_persist_directory"] = manifest.memory["persist_directory"]
        if "user_id" in manifest.memory:
            context_kwargs["memory_user_id"] = manifest.memory["user_id"]
        if "auto_store" in manifest.memory:
            context_kwargs["memory_auto_store"] = manifest.memory["auto_store"]
        if "max" in manifest.memory:
            context_kwargs["memory_max"] = manifest.memory["max"]
        if "min_score" in manifest.memory:
            context_kwargs["memory_min_score"] = manifest.memory["min_score"]
        if "timeout" in manifest.memory:
            context_kwargs["memory_timeout"] = manifest.memory["timeout"]
    # Apply explicit context overrides from the 'context' block
    if manifest.context:
        for key in ("writable_keys", "file_mounts", "env_prefix", "conversation_max_messages"):
            if key in manifest.context:
                context_kwargs[key] = manifest.context[key]

    context = ContextConfig(**context_kwargs)

    # Parse execution mode
    execution_mode = ExecutionMode.STRICT
    if manifest.execution_mode:
        execution_mode = ExecutionMode(manifest.execution_mode)

    # Parse open mode config
    open_mode = OpenModeConfig()
    if manifest.open_mode:
        open_mode = OpenModeConfig.model_validate(manifest.open_mode)

    # Parse servers — convert manifest dicts to ServerSpec instances
    servers: dict[str, HTTPServerSpec | StdioServerSpec | Any] = {}
    for server_name, server_data in manifest.servers.items():
        if isinstance(server_data, dict):
            transport = server_data.get("type", server_data.get("transport", ""))
            if transport in ("http", "streamable-http", "sse") or "url" in server_data:
                spec_kwargs: dict[str, Any] = {"url": server_data["url"]}
                if "transport" in server_data:
                    spec_kwargs["transport"] = server_data["transport"]
                elif "type" in server_data:
                    spec_kwargs["transport"] = server_data["type"]
                for opt in ("headers", "auth", "bearer_token", "api_key", "audience"):
                    if opt in server_data:
                        spec_kwargs[opt] = server_data[opt]
                servers[server_name] = HTTPServerSpec(**spec_kwargs)
            elif "command" in server_data:
                servers[server_name] = StdioServerSpec(
                    command=server_data["command"],
                    args=server_data.get("args", []),
                    env=server_data.get("env", {}),
                    cwd=server_data.get("cwd"),
                    keep_alive=server_data.get("keep_alive", True),
                )
            else:
                servers[server_name] = server_data
        else:
            servers[server_name] = server_data

    # Parse governance configs
    secrets_cfg = SecretScopeConfig()
    if manifest.secrets:
        secrets_cfg = SecretScopeConfig.model_validate(manifest.secrets)

    budget_cfg = BudgetConfig()
    if manifest.budget:
        budget_cfg = BudgetConfig.model_validate(manifest.budget)

    health_cfg = HealthConfig()
    if manifest.health:
        health_cfg = HealthConfig.model_validate(manifest.health)

    mission_cfg = MissionConfig()
    if manifest.mission:
        mission_cfg = MissionConfig.model_validate(manifest.mission)

    # Build the agent identity (env vars were already resolved in load_manifest).
    identity = None
    if manifest.identity:
        from promptise.superagent_schema import IdentityConfig

        identity = IdentityConfig.model_validate(manifest.identity).to_identity()

    # Build ProcessConfig with overrides from manifest.config
    config_data: dict[str, Any] = {
        "model": manifest.model,
        "instructions": manifest.instructions,
        "identity": identity,
        "execution_mode": execution_mode.value,
        "open_mode": open_mode.model_dump(),
        "servers": servers,
        "triggers": [t.model_dump() for t in triggers],
        "journal": journal.model_dump(),
        "context": context.model_dump(),
        "secrets": secrets_cfg.model_dump(),
        "budget": budget_cfg.model_dump(),
        "health": health_cfg.model_dump(),
        "mission": mission_cfg.model_dump(),
    }

    # Apply any extra config overrides
    for key, value in manifest.config.items():
        config_data[key] = value

    return ProcessConfig.model_validate(config_data)
