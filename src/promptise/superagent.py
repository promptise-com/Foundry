"""SuperAgent file loader and validator.

This module handles loading, parsing, validating, and resolving .superagent
configuration files, including cross-agent reference resolution with cycle detection.

The loader supports:
- Auto-detection of file format by extension (.superagent, .superagent.yaml, .superagent.yml)
- Environment variable resolution using ${VAR} and ${VAR:-default} syntax
- Cross-agent file reference resolution with circular reference detection
- Path resolution relative to config file location
- Conversion to Foundry native types (ServerSpec, model strings, etc.)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import ValidationError

from .config import HTTPServerSpec, ServerSpec, StdioServerSpec
from .env_resolver import resolve_env_in_dict, validate_all_env_vars_available
from .exceptions import SuperAgentError, SuperAgentValidationError
from .superagent_schema import (
    HTTPServerConfig,
    IdentityConfig,
    StdioServerConfig,
    SuperAgentSchema,
)

if TYPE_CHECKING:
    from .identity import AgentIdentity


class SuperAgentLoader:
    """Load and parse .superagent configuration files.

    This class handles the complete lifecycle of loading a .superagent file:
    parsing YAML, validating against schema, resolving environment variables,
    loading cross-agent references, and converting to native types.

    Supports:
    - Auto-detection of file format by extension
    - Environment variable resolution
    - Cross-agent file reference resolution
    - Circular reference detection
    - Path resolution relative to config file location

    Attributes:
        file_path: Resolved absolute path to the .superagent file.
        schema: Parsed and validated SuperAgentSchema.
        resolved_schema: Schema with environment variables resolved (after resolve_env_vars()).

    Examples:
        >>> loader = SuperAgentLoader.from_file("agent.superagent")
        >>> loader.resolve_env_vars()
        >>> config = loader.to_agent_config()
    """

    def __init__(
        self,
        file_path: Path,
        schema: SuperAgentSchema,
        *,
        _loading_chain: set[Path] | None = None,
    ) -> None:
        """Initialize loader with parsed schema.

        Args:
            file_path: Path to the source file.
            schema: Validated SuperAgentSchema.
            _loading_chain: Internal use for cycle detection in cross-agent loading.
        """
        self.file_path = file_path.resolve()
        self.schema = schema
        self._loading_chain = _loading_chain or {self.file_path}
        self.resolved_schema: SuperAgentSchema | None = None

    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
        *,
        _loading_chain: set[Path] | None = None,
    ) -> SuperAgentLoader:
        """Load a .superagent file from disk.

        Parses YAML content and validates against SuperAgentSchema.
        Supports .superagent, .superagent.yaml, and .superagent.yml extensions.

        Args:
            file_path: Path to .superagent file.
            _loading_chain: Internal use for cycle detection.

        Returns:
            SuperAgentLoader instance with parsed schema.

        Raises:
            SuperAgentError: If file not found, invalid format, or parse error.
            SuperAgentValidationError: If schema validation fails.

        Examples:
            >>> loader = SuperAgentLoader.from_file("my_agent.superagent")
            >>> # Or with explicit extension:
            >>> loader = SuperAgentLoader.from_file("my_agent.superagent.yaml")
        """
        path = Path(file_path).resolve()

        # Check file exists
        if not path.exists():
            raise SuperAgentError(f"File not found: {path}")

        # Validate extension
        if not cls._is_superagent_file(path):
            raise SuperAgentError(
                f"Invalid file extension. Expected .superagent, "
                f".superagent.yaml, or .superagent.yml. Got: {path.name}"
            )

        # Load YAML
        try:
            with path.open("r", encoding="utf-8") as f:
                raw_data = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise SuperAgentError(f"YAML parse error in {path}: {exc}") from exc
        except Exception as exc:
            raise SuperAgentError(f"Failed to read {path}: {exc}") from exc

        if not isinstance(raw_data, dict):
            raise SuperAgentError(
                f"Invalid YAML format in {path}: expected dict, got {type(raw_data)}"
            )

        # Validate schema
        try:
            schema = SuperAgentSchema.model_validate(raw_data)
        except ValidationError as exc:
            raise SuperAgentValidationError(
                f"Schema validation failed for {path}",
                errors=[dict(e) for e in exc.errors()],
                file_path=str(path),
            ) from exc

        return cls(path, schema, _loading_chain=_loading_chain)

    @staticmethod
    def _is_superagent_file(path: Path) -> bool:
        """Check if file has valid .superagent extension.

        Accepts: .superagent, .superagent.yaml, .superagent.yml

        Args:
            path: Path to check.

        Returns:
            True if file has valid extension.
        """
        name = path.name.lower()
        return (
            name.endswith(".superagent")
            or name.endswith(".superagent.yaml")
            or name.endswith(".superagent.yml")
        )

    def validate_env_vars(self) -> list[str]:
        """Check for missing environment variables.

        Scans the schema for ${VAR} references and checks if each variable
        is set in the environment. Variables with defaults (${VAR:-default})
        are not reported as missing.

        Returns:
            List of missing environment variable names (empty if all available).

        Examples:
            >>> loader = SuperAgentLoader.from_file("agent.superagent")
            >>> missing = loader.validate_env_vars()
            >>> if missing:
            ...     print(f"Missing: {', '.join(missing)}")
        """
        return validate_all_env_vars_available(self.schema.model_dump())

    def resolve_env_vars(self) -> SuperAgentLoader:
        """Resolve all environment variables in the schema.

        Replaces all ${VAR} and ${VAR:-default} references with actual values.
        Re-validates the schema after resolution to ensure it's still valid.

        Returns:
            Self (for method chaining).

        Raises:
            SuperAgentError: If required environment variables are missing.
            SuperAgentValidationError: If schema is invalid after resolution.

        Examples:
            >>> loader = SuperAgentLoader.from_file("agent.superagent")
            >>> loader.resolve_env_vars()  # Chainable
            >>> # Now loader.resolved_schema contains resolved values
        """
        # Check for missing vars first
        missing = self.validate_env_vars()
        if missing:
            raise SuperAgentError(
                f"Missing required environment variables in {self.file_path}: {', '.join(missing)}"
            )

        # Resolve all env vars in schema
        raw_dict = self.schema.model_dump()
        resolved_dict = resolve_env_in_dict(raw_dict, context_prefix=str(self.file_path))

        # Re-validate resolved schema
        try:
            self.resolved_schema = SuperAgentSchema.model_validate(resolved_dict)
        except ValidationError as exc:
            raise SuperAgentValidationError(
                f"Schema validation failed after env var resolution for {self.file_path}",
                errors=[dict(e) for e in exc.errors()],
                file_path=str(self.file_path),
            ) from exc

        return self

    def resolve_cross_agents(
        self,
        *,
        recursive: bool = True,
    ) -> dict[str, SuperAgentLoader]:
        """Load and resolve cross-agent references.

        Loads .superagent files referenced in the cross_agents section,
        with circular reference detection.

        Args:
            recursive: If True, recursively load cross-agents of cross-agents.

        Returns:
            Mapping of cross-agent name to loaded SuperAgentLoader.

        Raises:
            SuperAgentError: If circular reference detected or file not found.

        Examples:
            >>> loader = SuperAgentLoader.from_file("main.superagent")
            >>> cross_agents = loader.resolve_cross_agents()
            >>> for name, agent_loader in cross_agents.items():
            ...     print(f"Loaded cross-agent: {name}")
        """
        if not self.schema.cross_agents:
            return {}

        cross_loaders: dict[str, SuperAgentLoader] = {}

        for name, config in self.schema.cross_agents.items():
            # Resolve path relative to current file
            ref_path = (self.file_path.parent / config.file).resolve()

            # Check for circular reference
            if ref_path in self._loading_chain:
                chain_str = " -> ".join(str(p) for p in self._loading_chain)
                raise SuperAgentError(
                    f"Circular cross-agent reference detected: {chain_str} -> {ref_path}"
                )

            # Load referenced agent
            new_chain = self._loading_chain | {ref_path}
            try:
                cross_loader = SuperAgentLoader.from_file(
                    ref_path,
                    _loading_chain=new_chain,
                )

                if recursive:
                    cross_loader.resolve_env_vars()
                    # Recursively resolve its cross-agents too
                    cross_loader.resolve_cross_agents(recursive=True)

                cross_loaders[name] = cross_loader

            except Exception as exc:
                raise SuperAgentError(
                    f"Failed to load cross-agent '{name}' from {ref_path}: {exc}"
                ) from exc

        return cross_loaders

    def to_server_specs(self) -> dict[str, ServerSpec]:
        """Convert server configurations to ServerSpec objects.

        Converts the parsed YAML server configs to Foundry's native
        ServerSpec types (HTTPServerSpec and StdioServerSpec).

        Returns:
            Mapping of server name to ServerSpec.

        Examples:
            >>> loader = SuperAgentLoader.from_file("agent.superagent")
            >>> loader.resolve_env_vars()
            >>> servers = loader.to_server_specs()
            >>> # Use with build_agent(servers=servers, ...)
        """
        schema = self.resolved_schema or self.schema
        specs: dict[str, ServerSpec] = {}

        for name, config in schema.servers.items():
            if isinstance(config, HTTPServerConfig):
                specs[name] = HTTPServerSpec(
                    url=config.url,
                    transport=config.transport,
                    headers=config.headers,
                    auth=config.auth,
                )
            elif isinstance(config, StdioServerConfig):
                specs[name] = StdioServerSpec(
                    command=config.command,
                    args=config.args,
                    env=config.env,
                    cwd=config.cwd,
                    keep_alive=config.keep_alive,
                )

        return specs

    def to_model_string(self) -> str:
        """Convert model configuration to LangChain init string.

        Converts both simple and detailed model configs to the format
        expected by LangChain's init_chat_model() function.

        Returns:
            Model string suitable for init_chat_model (e.g., "openai:gpt-4.1").

        Examples:
            >>> loader = SuperAgentLoader.from_file("agent.superagent")
            >>> model_str = loader.to_model_string()
            >>> # "openai:gpt-4.1"
        """
        schema = self.resolved_schema or self.schema
        model_config = schema.agent.model

        if isinstance(model_config, str):
            return model_config

        # DetailedModelConfig: construct provider:name string
        return f"{model_config.provider}:{model_config.name}"

    def to_model_kwargs(self) -> dict[str, Any]:
        """Extract model kwargs from DetailedModelConfig (temperature, etc.).

        Returns:
            Dict of model kwargs to pass to init_chat_model.
            Empty dict if model is a simple string.
        """
        schema = self.resolved_schema or self.schema
        model_config = schema.agent.model
        if isinstance(model_config, str):
            return {}

        # One translation for YAML and code: the same words, the same provider
        # keyword arguments (endpoint -> azure_endpoint for Azure, …).
        from .models import Model

        return Model(
            model_config.name,
            provider=model_config.provider,
            deployment=model_config.deployment,
            api_key=model_config.api_key,
            endpoint=model_config.endpoint or model_config.base_url,
            api_version=model_config.api_version,
            region=model_config.region,
            project=model_config.project,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            timeout=model_config.timeout,
            extra=dict(model_config.extra),
        ).kwargs()

    def to_identity(self) -> AgentIdentity | None:
        """Build the agent's :class:`~promptise.identity.AgentIdentity`.

        Maps the resolved ``identity:`` section to the matching identity
        factory. Returns ``None`` when no ``identity:`` block is present.
        Environment variables in the section are already resolved (via
        :meth:`resolve_env_vars`) by the time this runs.

        Returns:
            The constructed ``AgentIdentity``, or ``None``.

        Raises:
            ProviderConfigError: If the identity configuration is invalid
                (e.g. a local identity with no ``agent_id``).
        """
        schema = self.resolved_schema or self.schema
        cfg: IdentityConfig | None = schema.identity
        if cfg is None:
            return None
        return cfg.to_identity()

    def to_agent_config(self) -> SuperAgentConfig:
        """Convert loaded schema to agent build configuration.

        Creates a SuperAgentConfig object ready to be used with
        build_agent().

        Returns:
            SuperAgentConfig ready for agent building.

        Examples:
            >>> loader = SuperAgentLoader.from_file("agent.superagent")
            >>> loader.resolve_env_vars()
            >>> config = loader.to_agent_config()
            >>> # Use with: await build_agent(**config.to_build_kwargs())
        """
        schema = self.resolved_schema or self.schema

        # Convert sandbox config if present
        sandbox_config: bool | dict[str, Any] | None = None
        if schema.sandbox is not None:
            if isinstance(schema.sandbox, bool):
                sandbox_config = schema.sandbox
            else:
                # SandboxConfigSection -> dict for build_agent
                sandbox_config = schema.sandbox.model_dump()

        # Convert memory config if present
        memory_config: dict[str, Any] | None = None
        if schema.memory is not None:
            memory_config = schema.memory.model_dump()

        # Convert observability config if present
        observe_config: bool | dict[str, Any] | None = None
        if schema.observability is not None:
            if isinstance(schema.observability, bool):
                observe_config = schema.observability
            else:
                observe_config = schema.observability.model_dump()

        # Convert tool optimization config if present
        optimize_config: bool | str | dict[str, Any] | None = None
        if schema.optimize_tools is not None:
            if isinstance(schema.optimize_tools, (bool, str)):
                optimize_config = schema.optimize_tools
            else:
                optimize_config = schema.optimize_tools.model_dump()

        # Convert cache config if present
        cache_config: bool | dict[str, Any] | None = None
        if schema.cache is not None:
            if isinstance(schema.cache, bool):
                cache_config = schema.cache
            else:
                cache_config = schema.cache.model_dump()

        # Convert approval config if present
        approval_config: dict[str, Any] | None = None
        if schema.approval is not None:
            approval_config = schema.approval.model_dump()

        return SuperAgentConfig(
            model=self.to_model_string(),
            model_kwargs=self.to_model_kwargs(),
            servers=self.to_server_specs(),
            instructions=schema.agent.instructions,
            trace=schema.agent.trace,
            identity=self.to_identity(),
            cross_agents=schema.cross_agents or {},
            sandbox=sandbox_config,
            memory=memory_config,
            observe=observe_config,
            optimize_tools=optimize_config,
            cache=cache_config,
            approval=approval_config,
            events=schema.events.model_dump() if schema.events else None,
            adaptive=(
                schema.adaptive.model_dump()
                if hasattr(schema.adaptive, "model_dump")
                else schema.adaptive
            )
            if schema.adaptive is not None
            else None,
            guardrails=(
                schema.guardrails.model_dump()
                if hasattr(schema.guardrails, "model_dump")
                else schema.guardrails
            )
            if schema.guardrails is not None
            else None,
            max_invocation_time=schema.max_invocation_time,
        )


class SuperAgentConfig:
    """Processed configuration ready for agent building.

    This class holds the final, resolved configuration that can be passed
    directly to build_agent().

    Attributes:
        model: Model string for init_chat_model.
        servers: Mapping of server name to ServerSpec.
        instructions: Optional system prompt override.
        trace: Enable tool tracing.
        cross_agents: Raw cross-agent configs (resolved separately).
        sandbox: Optional sandbox configuration (bool or dict).

    Examples:
        >>> config = SuperAgentConfig(
        ...     model="openai:gpt-4.1",
        ...     servers={"test": HTTPServerSpec(url="http://test")},
        ...     trace=True,
        ...     sandbox=True
        ... )
    """

    def __init__(
        self,
        *,
        model: str,
        model_kwargs: dict[str, Any] | None = None,
        servers: dict[str, ServerSpec],
        instructions: str | None = None,
        trace: bool = True,
        identity: AgentIdentity | None = None,
        cross_agents: dict[str, Any] | None = None,
        sandbox: bool | dict[str, Any] | None = None,
        memory: dict[str, Any] | None = None,
        observe: bool | dict[str, Any] | None = None,
        optimize_tools: bool | str | dict[str, Any] | None = None,
        cache: bool | dict[str, Any] | None = None,
        approval: dict[str, Any] | None = None,
        events: dict[str, Any] | None = None,
        adaptive: bool | dict[str, Any] | None = None,
        guardrails: bool | dict[str, Any] | None = None,
        max_invocation_time: float = 0,
    ) -> None:
        """Initialize SuperAgentConfig.

        Args:
            model: Model string (e.g., "openai:gpt-5-mini").
            model_kwargs: Extra kwargs for init_chat_model (temperature, etc.).
            servers: Mapping of server name to ServerSpec.
            instructions: Optional system prompt.
            trace: Enable tool tracing.
            cross_agents: Cross-agent configuration dict.
            sandbox: Optional sandbox configuration (bool or dict).
            memory: Optional memory configuration dict.
            observe: Optional observability config (bool or dict).
            optimize_tools: Optional tool optimization (bool, str, or dict).
            cache: Optional semantic cache config (bool or dict).
        """
        self.model = model
        self.model_kwargs = model_kwargs or {}
        self.servers = servers
        self.instructions = instructions
        self.trace = trace
        self.identity = identity
        self.cross_agents = cross_agents or {}
        self.sandbox = sandbox
        self.memory = memory
        self.observe = observe
        self.optimize_tools = optimize_tools
        self.cache = cache
        self.approval = approval
        self.events = events
        self.adaptive = adaptive
        self.guardrails = guardrails
        self.max_invocation_time = max_invocation_time

    def to_build_kwargs(self) -> dict[str, Any]:
        """Convert to kwargs dict for build_agent().

        Returns:
            Dict ready to unpack: build_agent(**config.to_build_kwargs())

        Note:
            cross_agents needs special handling to build actual CrossAgent
            objects - see CLI integration for example.

        Examples:
            >>> config = loader.to_agent_config()
            >>> kwargs = config.to_build_kwargs()
            >>> agent = await build_agent(**kwargs)
        """
        # When model_kwargs are present (temperature, max_tokens, etc.),
        # create an actual model instance so these params take effect.
        model: Any = self.model
        if self.model_kwargs:
            from .models import resolve_model

            model = resolve_model(self.model, **self.model_kwargs)

        kwargs = {
            "model": model,
            "servers": self.servers,
            "instructions": self.instructions,
            "trace_tools": self.trace,
            # Note: cross_agents needs special handling to build actual
            # CrossAgent objects - see CLI integration
        }

        if self.identity is not None:
            kwargs["identity"] = self.identity

        if self.sandbox is not None:
            kwargs["sandbox"] = self.sandbox

        if self.memory is not None:
            # Pass config dict to build_agent which calls _build_provider_from_config
            kwargs["memory"] = self.memory

        if self.observe is not None:
            kwargs["observe"] = self.observe

        if self.optimize_tools is not None:
            kwargs["optimize_tools"] = self.optimize_tools

        if self.cache is not None:
            if isinstance(self.cache, bool) and self.cache:
                from .cache import SemanticCache

                kwargs["cache"] = SemanticCache()
            elif isinstance(self.cache, dict):
                from .cache import SemanticCache

                kwargs["cache"] = SemanticCache(**self.cache)

        if self.approval is not None:
            from .approval import ApprovalPolicy, QueueApprovalHandler, WebhookApprovalHandler

            approval_data = dict(self.approval)
            handler_type = approval_data.pop("handler", "webhook")
            tools = approval_data.pop("tools")
            webhook_url = approval_data.pop("webhook_url", None)

            handler: WebhookApprovalHandler | QueueApprovalHandler
            if handler_type == "webhook":
                if not webhook_url:
                    raise ValueError("approval.webhook_url required when handler is 'webhook'")
                handler = WebhookApprovalHandler(url=webhook_url)
            elif handler_type == "queue":
                handler = QueueApprovalHandler()
            else:
                raise ValueError(f"Unsupported approval handler type: {handler_type!r}")

            kwargs["approval"] = ApprovalPolicy(
                tools=tools,
                handler=handler,
                **{k: v for k, v in approval_data.items() if v is not None},
            )

        if self.events is not None:
            from .events import EventNotifier, LogSink, WebhookSink

            sinks_config = self.events.get("sinks", [])
            sinks: list[Any] = []
            for sc in sinks_config:
                sink_type = sc.get("type", "log")
                if sink_type == "webhook" and sc.get("url"):
                    sinks.append(
                        WebhookSink(
                            url=sc["url"],
                            events=sc.get("events"),
                            headers=sc.get("headers", {}),
                            secret=sc.get("secret"),
                            min_severity=sc.get("min_severity"),
                            max_retries=sc.get("max_retries", 3),
                            redact_sensitive=sc.get("redact_sensitive", True),
                        )
                    )
                elif sink_type == "log":
                    sinks.append(
                        LogSink(
                            events=sc.get("events"),
                            min_severity=sc.get("min_severity"),
                        )
                    )
            if sinks:
                kwargs["events"] = EventNotifier(sinks=sinks)

        # Adaptive strategy
        if self.adaptive is not None:
            if isinstance(self.adaptive, bool) and self.adaptive:
                kwargs["adaptive"] = True
            elif isinstance(self.adaptive, dict):
                from .strategy import AdaptiveStrategyConfig

                kwargs["adaptive"] = AdaptiveStrategyConfig(**self.adaptive)

        # Guardrails
        if self.guardrails is not None:
            if isinstance(self.guardrails, bool) and self.guardrails:
                from .guardrails import PromptiseSecurityScanner

                scanner = PromptiseSecurityScanner.default()
                scanner.warmup()
                kwargs["guardrails"] = scanner
            elif isinstance(self.guardrails, dict):
                from .guardrails import PromptiseSecurityScanner

                scanner = PromptiseSecurityScanner(
                    **{k: v for k, v in self.guardrails.items() if k != "warmup"}
                )
                if self.guardrails.get("warmup", True):
                    scanner.warmup()
                kwargs["guardrails"] = scanner

        # Max invocation time
        if self.max_invocation_time and self.max_invocation_time > 0:
            kwargs["max_invocation_time"] = self.max_invocation_time

        return kwargs


def load_superagent_file(
    file_path: str | Path,
    *,
    resolve_refs: bool = True,
) -> tuple[SuperAgentLoader, dict[str, SuperAgentLoader]]:
    """Convenience function to load a .superagent file with all resolutions.

    Loads the file, resolves environment variables, and optionally resolves
    cross-agent references in a single call.

    Args:
        file_path: Path to .superagent file.
        resolve_refs: If True, resolve cross-agent references.

    Returns:
        Tuple of (main_loader, cross_agent_loaders).

    Raises:
        SuperAgentError: If loading or validation fails.

    Examples:
        >>> main, cross_agents = load_superagent_file("agent.superagent")
        >>> config = main.to_agent_config()
        >>> # Build the agent
        >>> agent = await build_agent(**config.to_build_kwargs())
    """
    loader = SuperAgentLoader.from_file(file_path)
    loader.resolve_env_vars()

    cross_loaders: dict[str, SuperAgentLoader] = {}
    if resolve_refs:
        cross_loaders = loader.resolve_cross_agents(recursive=True)

    return loader, cross_loaders
