# SuperAgent Files

Define agents declaratively using `.superagent` YAML files -- configure model, servers, memory, sandbox, and cross-agent references in a single file.

## Quick Example

```yaml
# analyst.superagent
version: "1.0"

agent:
  model: "openai:gpt-5-mini"
  instructions: "You are a data analyst. Use available tools to answer questions."
  trace: true

servers:
  database:
    type: http
    url: "http://localhost:9000/mcp"
    headers:
      Authorization: "Bearer ${DB_TOKEN}"
```

Load and run it in Python:

```python
import asyncio
from promptise import build_agent
from promptise.superagent import load_superagent_file

async def main():
    loader, cross_agents = load_superagent_file("analyst.superagent")
    config = loader.to_agent_config()
    agent = await build_agent(**config.to_build_kwargs())

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Show me top 10 customers by revenue"}]
    })
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

## Concepts

A `.superagent` file is a YAML document validated against `SuperAgentSchema`. It replaces the programmatic `build_agent()` call with a declarative configuration file that can be version-controlled, shared across teams, and loaded at runtime.

The loader pipeline works in three steps:

1. **Parse and validate** -- `SuperAgentLoader.from_file()` reads YAML and validates it against the Pydantic schema. Invalid fields are rejected immediately.
2. **Resolve environment variables** -- `resolve_env_vars()` replaces `${VAR}` and `${VAR:-default}` placeholders with actual values from the environment.
3. **Convert to native types** -- `to_agent_config()` produces a `SuperAgentConfig` object whose `to_build_kwargs()` method returns a dict ready for `build_agent(**kwargs)`.

## Full YAML Schema

Here is a `.superagent` file using every available section:

```yaml
version: "1.0"

agent:
  model: "openai:gpt-5-mini"          # or detailed config (see below)
  instructions: "You are a research assistant."
  trace: true

identity:
  provider: entra                      # local|entra|aws|gcp|spiffe|oidc|auto
  agent_id: research-bot               # who is acting (for attribution)
  owner: research-team
  labels: {env: prod}
  client_id: "${AZURE_CLIENT_ID}"      # provider-specific (Entra here)
  resource: api://search.example.com   # audience the credential targets

servers:
  search:
    type: http
    url: "https://search.example.com/mcp"
    transport: streamable-http
    headers:
      Authorization: "Bearer ${SEARCH_TOKEN}"
  local_tools:
    type: stdio
    command: python
    args: ["-m", "my_tools.server"]
    env:
      API_KEY: "${MY_API_KEY}"
    cwd: "/opt/tools"
    keep_alive: true

cross_agents:
  math_expert:
    file: "./agents/math.superagent"
    description: "Specialized math and calculation agent"

memory:
  provider: chroma                     # "in_memory", "chroma", or "mem0"
  collection: research_memory
  persist_directory: ".promptise/chroma"

sandbox:
  backend: docker
  image: "python:3.11-slim"
  cpu_limit: 2
  memory_limit: "4G"
  disk_limit: "1G"
  pids_limit: 256
  network: none
  timeout: 300
  workdir: "/workspace"
  allow_sudo: false
```

### Section Reference

#### `version`

Always `"1.0"`. Required for forward compatibility.

#### `agent`

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | `str \| DetailedModelConfig` | **required** | Model identifier or detailed configuration. |
| `instructions` | `str \| None` | `None` | System prompt override. |
| `trace` | `bool` | `true` | Print tool invocations to stdout. |

The `model` field accepts either a simple string or a detailed configuration object:

=== "Simple string"

    ```yaml
    agent:
      model: "openai:gpt-5-mini"
    ```

=== "Detailed config"

    ```yaml
    agent:
      model:
        provider: openai
        name: gpt-5-mini
        api_key: "${OPENAI_API_KEY}"
        temperature: 0.7
        max_tokens: 4096
        timeout: 30
        base_url: "https://custom-endpoint.example.com/v1"
    ```

=== "Azure OpenAI"

    ```yaml
    agent:
      model:
        provider: azure                       # alias of azure_openai
        name: chat-prod                       # your DEPLOYMENT name in Azure AI Foundry
        api_key: "${AZURE_OPENAI_API_KEY}"
        endpoint: "https://my-resource.openai.azure.com/"
        api_version: "2024-10-21"
    ```

=== "Azure AI Foundry catalog"

    ```yaml
    agent:
      model:
        provider: foundry                     # alias of azure_ai -- Llama, Mistral, DeepSeek, Phi, ...
        name: Llama-3.3-70B-Instruct
        api_key: "${AZURE_INFERENCE_CREDENTIAL}"
        endpoint: "https://my-resource.services.ai.azure.com/models"
    ```

=== "Bedrock"

    ```yaml
    agent:
      model:
        provider: bedrock
        name: anthropic.claude-sonnet-4-20250514-v1:0
        region: us-east-1                     # credentials from the boto3 chain (profile, SSO, role)
    ```

Both forms go through the same resolver as `build_agent(model=...)` (see [Model Setup](../../getting-started/model-setup.md)), so every provider prefix **and its aliases** work in `provider:` too -- `azure`, `foundry`, `gemini`, `vertex`, `bedrock`, `mistral`, `grok`, `hf`. The `provider` and `name` fields are joined into `provider:name`. The detailed form has these fields (`extra="forbid"`, so a typo is a validation error):

| Field | Type | Description |
|---|---|---|
| `provider` | `str` | Provider prefix or alias (`openai`, `azure`, `foundry`, `bedrock`, ...). |
| `name` (or `model`) | `str` | Model name -- or, for `azure`, your **deployment** name when `deployment` is omitted. `model:` is accepted as a synonym. |
| `deployment` | `str \| None` | Azure OpenAI deployment name -- what you called the model in Azure AI Foundry. Lets `name` stay the real model (`gpt-4o`) while requests go to the deployment. |
| `api_key` | `str \| None` | The provider's API key (Azure AI Foundry: the deployment key). Not applicable to `bedrock`, `vertex` and `ollama`. |
| `endpoint` | `str \| None` | Where to send requests: an Azure OpenAI resource endpoint, an Azure AI Foundry inference endpoint, or the `/v1` URL of an OpenAI-compatible server. |
| `base_url` | `str \| None` | Same as `endpoint` (kept for existing files). |
| `api_version` | `str \| None` | Azure OpenAI REST API version (`"2024-10-21"`). |
| `region` | `str \| None` | Bedrock region or Vertex AI location. |
| `project` | `str \| None` | Google Cloud project id (Vertex AI). |
| `temperature`, `max_tokens`, `timeout` | | Sampling and request settings. |
| `extra` | `dict` | Provider-specific keyword arguments, passed through verbatim. |

The same six words (`deployment`, `api_key`, `endpoint`, `api_version`, `region`, `project`) work for every provider and are translated to the provider's own keyword arguments exactly like [`promptise.models.Model`](../../api/models.md#model) -- `endpoint` becomes `azure_endpoint=` for Azure OpenAI and `base_url=` for OpenAI, `region` becomes `region_name=` for Bedrock and `location=` for Vertex AI. A word the provider has no setting for (`api_key` on Bedrock, `api_version` on OpenAI, `deployment` outside Azure OpenAI) is rejected with a `ModelSetupError` that says what to use instead.

Credentials in the file **satisfy the environment check**: a provider's required variables are only demanded when neither the variable is set nor the matching field is given. In the Azure example above, `api_key` stands in for `AZURE_OPENAI_API_KEY`, `endpoint` for `AZURE_OPENAI_ENDPOINT` and `api_version` for `OPENAI_API_VERSION`, so the file works with nothing exported except `AZURE_OPENAI_API_KEY` for the `${...}` reference. A model that is still missing something fails when the agent is built (`to_build_kwargs()` / `build_agent()`) with a `ModelSetupError` naming the variable and where to find its value -- the same text `promptise models check azure:chat-prod` prints.

#### `identity`

Optional. Gives the agent a stable, traceable [Agent Identity](../../identity/overview.md)
— *who is acting*. A **local** identity (`provider: local`) tags the
agent's actions for attribution; a **verifiable** identity (any cloud
provider, or `auto`) additionally presents a signed credential to the MCP
servers it calls, so they can authenticate and attribute it. All string
fields support `${ENV_VAR}`.

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `"local" \| "entra" \| "aws" \| "gcp" \| "spiffe" \| "oidc" \| "auto"` | `"local"` | Identity backing. |
| `agent_id` | `str \| None` | `None` | Stable id. **Required** for `local`; optional for verifiable (derived from the IdP `sub`/`oid`). |
| `name` | `str \| None` | `None` | Human-readable display name. |
| `owner` | `str \| None` | `None` | Owning team or person. |
| `labels` | `dict[str, str]` | `{}` | Free-form metadata. |
| `mode` | `str \| None` | `None` | Provider mode (`entra`: `auto`/`imds`/`projected`; `aws`: `auto`/`sts`/`projected`; `spiffe`: `auto`/`file`/`sdk`). |
| `client_id` | `str \| None` | `None` | Entra managed-identity client id. |
| `resource` | `str \| None` | `None` | Resource/audience the credential targets (Entra). |
| `region` | `str \| None` | `None` | AWS region for STS. |
| `audience` | `str \| None` | `None` | Audience the credential targets (AWS/GCP/SPIFFE). |
| `service_account_email` | `str \| None` | `None` | Attached service account (GCP). |
| `socket_path` | `str \| None` | `None` | SPIFFE Workload API socket (SDK mode). |
| `issuer` | `str \| None` | `None` | OIDC issuer URL. **Required** for `provider: oidc`. |
| `token_file` | `str \| None` | `None` | Path to a JWT file (Entra/AWS/SPIFFE/OIDC). |
| `token_env_var` | `str \| None` | `None` | Env var holding the JWT, re-read each refresh (OIDC). |

=== "Local (attribution only)"

    ```yaml
    identity:
      provider: local
      agent_id: billing-bot
      owner: payments
      labels: {env: prod}
    ```

=== "Verifiable (Microsoft Entra)"

    ```yaml
    identity:
      provider: entra
      agent_id: billing-bot
      client_id: "${AZURE_CLIENT_ID}"
      resource: api://my-mcp-server
    ```

=== "Verifiable (generic OIDC / CI)"

    ```yaml
    identity:
      provider: oidc
      agent_id: release-bot
      issuer: https://gitlab.com
      token_env_var: CI_JOB_JWT_V2
    ```

=== "Auto-detect the platform"

    ```yaml
    identity:
      provider: auto
      agent_id: data-bot
    ```

For `provider: oidc`, exactly one of `token_file` or `token_env_var` is
required. See the [Agent Identity guide](../../identity/guide.md) for the
end-to-end flow (outbound auth, server-side verification, audit).

#### `servers`

A dict of named server configurations. Each entry requires a `type` discriminator field.

=== "HTTP server"

    | Field | Type | Default | Description |
    |---|---|---|---|
    | `type` | `"http"` | **required** | Discriminator. |
    | `url` | `str` | **required** | Full MCP endpoint URL. |
    | `transport` | `"http" \| "streamable-http" \| "sse"` | `"http"` | Transport protocol. |
    | `headers` | `dict[str, str]` | `{}` | HTTP headers (values support `${ENV_VAR}`). |
    | `auth` | `str \| None` | `None` | Legacy auth token. |

=== "Stdio server"

    | Field | Type | Default | Description |
    |---|---|---|---|
    | `type` | `"stdio"` | **required** | Discriminator. |
    | `command` | `str` | **required** | Executable command. |
    | `args` | `list[str]` | `[]` | Command arguments. |
    | `env` | `dict[str, str]` | `{}` | Environment variables (values support `${ENV_VAR}`). |
    | `cwd` | `str \| None` | `None` | Working directory. |
    | `keep_alive` | `bool` | `true` | Maintain persistent connection. |

#### `cross_agents`

Optional. Maps a peer name to a file reference.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | `str` | **required** | Path to the peer's `.superagent` file (relative to this file). |
| `description` | `str` | `""` | Description shown in the auto-generated `ask_agent_<name>` tool. |

#### `memory`

Optional. Configures persistent agent memory.

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | `"in_memory" \| "chroma" \| "mem0"` | `"in_memory"` | Memory backend. |
| `collection` | `str` | `"agent_memory"` | ChromaDB collection name. |
| `persist_directory` | `str \| None` | `None` | ChromaDB persistence path. |
| `user_id` | `str` | `"default"` | Mem0 user scope. |
| `agent_id` | `str \| None` | `None` | Mem0 agent scope. |

#### `sandbox`

Optional. Can be `true` for defaults or a detailed configuration object.
Unknown keys are rejected. If the sandbox cannot be started (Docker not
running, the `promptise[sandbox]` extra missing, gVisor not installed), loading
the agent fails instead of running it without a sandbox.

| Field | Type | Default | Description |
|---|---|---|---|
| `backend` | `"docker" \| "gvisor"` | `"docker"` | Container backend. |
| `image` | `str` | `"python:3.11-slim"` | Base container image. |
| `cpu_limit` | `int` | `2` | Maximum CPU cores (1--32). |
| `memory_limit` | `str` | `"4G"` | Maximum memory. |
| `disk_limit` | `str` | `"1G"` | Size of the writable workspace. |
| `pids_limit` | `int` | `256` | Maximum processes and threads. |
| `network` | `"none" \| "restricted" \| "full"` | `"none"` | Network isolation mode. `"restricted"` needs `iptables` in the image and refuses to start without it. |
| `persistent` | `bool` | `false` | Keep the container after the session ends. |
| `timeout` | `int` | `300` | Max execution time in seconds (1--3600). |
| `workdir` | `str` | `"/workspace"` | Working directory inside container. |
| `env` | `dict[str, str]` | `{}` | Additional environment variables. |
| `allow_sudo` | `bool` | `false` | Allow sudo access in container. |

## Environment Variable Resolution

All string values in the YAML support environment variable substitution:

| Syntax | Behavior |
|---|---|
| `${VAR}` | Replaced with the value of `VAR`. Raises an error if not set. |
| `${VAR:-default}` | Replaced with the value of `VAR`, or `"default"` if not set. |

```yaml
servers:
  api:
    type: http
    url: "${API_URL:-http://localhost:8000/mcp}"
    headers:
      Authorization: "Bearer ${API_TOKEN}"
```

Call `loader.validate_env_vars()` to check which variables are missing before resolving:

```python
loader = SuperAgentLoader.from_file("agent.superagent")
missing = loader.validate_env_vars()
if missing:
    print(f"Set these env vars: {', '.join(missing)}")
else:
    loader.resolve_env_vars()
```

## Cross-Agent References and Cycle Detection

The loader recursively resolves cross-agent references. If agent A references agent B and agent B references agent A, the loader raises a `SuperAgentError` with the full reference chain.

```yaml
# main.superagent
cross_agents:
  researcher:
    file: "./researcher.superagent"
    description: "Web research specialist"
  analyst:
    file: "./analyst.superagent"
    description: "Data analysis specialist"
```

```python
loader, cross_loaders = load_superagent_file("main.superagent")

for name, cross_loader in cross_loaders.items():
    print(f"Loaded peer: {name} from {cross_loader.file_path}")
```

## The `SuperAgentLoader` Class

```python
from promptise.superagent import SuperAgentLoader, load_superagent_file

# Step-by-step usage
loader = SuperAgentLoader.from_file("agent.superagent")
loader.resolve_env_vars()
cross_loaders = loader.resolve_cross_agents(recursive=True)

servers = loader.to_server_specs()      # dict[str, ServerSpec]
model   = loader.to_model_string()      # e.g. "openai:gpt-5-mini"
config  = loader.to_agent_config()      # SuperAgentConfig

# Or use the convenience function
loader, cross_loaders = load_superagent_file("agent.superagent")
config = loader.to_agent_config()
kwargs = config.to_build_kwargs()
agent = await build_agent(**kwargs)
```

## API Summary

| Symbol | Import | Description |
|---|---|---|
| `SuperAgentLoader` | `from promptise.superagent import SuperAgentLoader` | Loads, validates, and resolves `.superagent` files. Key methods: `from_file()`, `resolve_env_vars()`, `resolve_cross_agents()`, `to_agent_config()`. |
| `load_superagent_file()` | `from promptise.superagent import load_superagent_file` | Convenience function that loads, resolves env vars, and resolves cross-agent refs in one call. Returns `(loader, cross_loaders)`. |
| `SuperAgentSchema` | `from promptise.superagent_schema import SuperAgentSchema` | Pydantic model for the full `.superagent` YAML schema. |
| `SuperAgentConfig` | `from promptise.superagent import SuperAgentConfig` | Processed config with `to_build_kwargs()` for `build_agent()`. |

!!! tip "File extensions"
    The loader accepts `.superagent`, `.superagent.yaml`, and `.superagent.yml` extensions.

!!! tip "Schema validation"
    All sections use `extra="forbid"`, so misspelled fields (e.g. `instuctions` instead of `instructions`) produce a clear validation error rather than being silently ignored.

!!! warning "Direct API keys in YAML"
    The schema validator warns if an `api_key` field looks like a direct secret (starts with `sk-` or `pk-`). Always use `${ENV_VAR}` syntax for credentials.

!!! warning "At least one capability required"
    The schema requires at least one of `servers`, `cross_agents`, or `sandbox` to be configured. A file with only `agent` and `version` fails validation.

## What's Next?

- [Building Agents](building-agents.md) -- the `build_agent()` function that SuperAgent files feed into.
- [Server Configuration](server-specs.md) -- details on `StdioServerSpec` and `HTTPServerSpec`.
- [Cross-Agent Delegation](cross-agent.md) -- runtime behavior of the delegation tools generated from cross-agent references.
