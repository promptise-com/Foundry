# SuperAgent API Reference

SuperAgent file loader, validator, and schema definitions. Handles loading, parsing, validating, and resolving `.superagent` configuration files, including cross-agent reference resolution with cycle detection and environment variable substitution.

## SuperAgentLoader

::: promptise.superagent.SuperAgentLoader
    options:
      show_source: false
      heading_level: 3

## SuperAgentConfig

::: promptise.superagent.SuperAgentConfig
    options:
      show_source: false
      heading_level: 3

## load_superagent_file

::: promptise.superagent.load_superagent_file
    options:
      show_source: false
      heading_level: 3

## Schema Definitions

### SuperAgentSchema

::: promptise.superagent_schema.SuperAgentSchema
    options:
      show_source: false
      heading_level: 4

### AgentSection

::: promptise.superagent_schema.AgentSection
    options:
      show_source: false
      heading_level: 4

### IdentityConfig

The `identity:` section — gives the agent a local or verifiable
[Agent Identity](../identity/overview.md). See the
[`.superagent` reference](../core/agents/superagent-files.md#identity).

::: promptise.superagent_schema.IdentityConfig
    options:
      show_source: false
      heading_level: 4

### Server Configuration

#### ServerConfig

Discriminated union type: `HTTPServerConfig | StdioServerConfig`, selected by the `type` field (`"http"` or `"stdio"`).

#### HTTPServerConfig

::: promptise.superagent_schema.HTTPServerConfig
    options:
      show_source: false
      heading_level: 5

#### StdioServerConfig

::: promptise.superagent_schema.StdioServerConfig
    options:
      show_source: false
      heading_level: 5

### Model Configuration

#### ModelConfig

Union type supporting both a simple string (e.g., `"openai:gpt-4.1"`) and a `DetailedModelConfig` object. Either form is resolved through [`promptise.models.resolve_model`](models.md#resolve_model), so provider aliases (`azure:`, `foundry:`, `gemini:`, `bedrock:`, ...) work in both, and credentials given in the file (`api_key`, `endpoint`, `api_version`, `region`, `project`) satisfy the provider's environment-variable check. See [SuperAgent Files](../core/agents/superagent-files.md#agent) for an Azure example.

#### DetailedModelConfig

::: promptise.superagent_schema.DetailedModelConfig
    options:
      show_source: false
      heading_level: 5

### MemorySection

::: promptise.superagent_schema.MemorySection
    options:
      show_source: false
      heading_level: 4

### SandboxConfigSection

::: promptise.superagent_schema.SandboxConfigSection
    options:
      show_source: false
      heading_level: 4

### CrossAgentConfig

::: promptise.superagent_schema.CrossAgentConfig
    options:
      show_source: false
      heading_level: 4
