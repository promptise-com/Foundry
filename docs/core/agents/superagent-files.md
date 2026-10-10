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
    bearer_token: "${DB_TOKEN}"
```

Run it from the terminal with `promptise agent analyst.superagent`, or load and run it in Python:

```python
import asyncio
from promptise import build_superagent

async def main():
    agent = await build_superagent("analyst.superagent")
    try:
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "Show me top 10 customers by revenue"}]
        })
        print(result["messages"][-1].content)
    finally:
        await agent.shutdown()

asyncio.run(main())
```

`build_superagent()` loads the file, reads `.env`, resolves `${VAR}` references and builds the agent -- together with every agent listed under `cross_agents:`, at any depth (see [Teams of agents](#teams-of-agents-cross_agents)).

## Concepts

A `.superagent` file is a YAML document validated against `SuperAgentSchema`. It replaces the programmatic `build_agent()` call with a declarative configuration file that can be version-controlled, shared across teams, and loaded at runtime.

The loader pipeline works in three steps:

1. **Parse and validate** -- `SuperAgentLoader.from_file()` reads YAML and validates it against the Pydantic schema. Invalid fields -- and settings that could never be built, such as `approval.handler: callback` -- are rejected immediately.
2. **Resolve environment variables** -- `resolve_env_vars()` loads the nearest `.env` file, then replaces `${VAR}` and `${VAR:-default}` placeholders with actual values from the environment.
3. **Convert to native types** -- `to_agent_config()` produces a `SuperAgentConfig` object whose `to_build_kwargs()` method returns a dict ready for `build_agent(**kwargs)`.

`build_superagent()` runs all three steps for the file and each of its cross-agents, and builds them.

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
    bearer_token: "${SEARCH_TOKEN}"   # or api_key: "${SEARCH_KEY}"
  local_tools:
    type: stdio
    command: python
    args: ["-m", "my_tools.server"]
    env:
      API_KEY: "${MY_API_KEY}"
    cwd: "/opt/tools"                 # default: this file's folder
    keep_alive: true

cross_agents:
  math_expert:
    file: "./agents/math.superagent"
    description: "Specialized math and calculation agent"
    timeout: 120                      # seconds to wait for its answer

max_delegation_depth: 3               # nested delegations per request
delegation_timeout: 300               # default wait for any cross-agent
include_broadcast: false              # add broadcast_to_agents

approval:
  tools: ["send_email", "delete_*"]
  handler: webhook                    # or queue
  webhook_url: "https://approvals.example.com/requests"
  webhook_secret: "${APPROVAL_WEBHOOK_SECRET}"

max_invocation_time: 600

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

Optional. `"1.0"` is the only schema version and the default, so a file without `version:` is read as `"1.0"`. Any other value is a validation error. Writing `version: "1.0"` keeps the file explicit if a later version is added.

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
    | `bearer_token` | `str \| None` | `None` | Sent as `Authorization: Bearer <token>`. Use `${ENV_VAR}`. |
    | `api_key` | `str \| None` | `None` | Sent as `x-api-key: <key>`. Use `${ENV_VAR}`. |
    | `audience` | `str \| None` | `None` | Audience of the credential minted from the agent's `identity:` when no `bearer_token` is set. |

    `auth:` is rejected with a validation error. Earlier versions accepted it but never sent it to the server, so the server answered `401`. Use `bearer_token:` or `api_key:` instead.

=== "Stdio server"

    | Field | Type | Default | Description |
    |---|---|---|---|
    | `type` | `"stdio"` | **required** | Discriminator. |
    | `command` | `str` | **required** | Executable command. |
    | `args` | `list[str]` | `[]` | Command arguments. |
    | `env` | `dict[str, str]` | `{}` | Environment variables (values support `${ENV_VAR}`). |
    | `cwd` | `str \| None` | this file's folder | Working directory of the server process. A relative `cwd` is relative to this file's folder. |
    | `keep_alive` | `bool` | `true` | Maintain persistent connection. |

    Paths are relative to the `.superagent` file, not to the directory you run from: the server starts in the file's folder, so `args: ["incidents_server.py"]` finds the script next to the file. A relative `command` with a path separator (`./bin/server`) is resolved against the file's folder too; a bare name such as `python` or `npx` is looked up on `PATH`. A path with a root (`/opt/tools`, `C:\tools`, `\\server\share`) is used as written.

    On Windows, write paths in single quotes or with forward slashes. Inside double quotes YAML reads a backslash as an escape sequence: `"C:\Python312\python.exe"` is a parse error and `"C:\tools\new"` silently contains a tab and a newline.

    ```yaml
    command: 'C:\Python312\python.exe'   # single quotes: backslashes are literal
    cwd: C:/tools                         # forward slashes work everywhere
    ```

#### `cross_agents`

Optional. Maps a peer name to a file reference.

| Field | Type | Default | Description |
|---|---|---|---|
| `file` | `str` | **required** | Path to the peer's `.superagent` file (relative to this file). |
| `description` | `str` | `""` | Description shown in the auto-generated `ask_agent_<name>` tool. |
| `timeout` | `float \| None` | `None` | Seconds to wait for this agent's answer; overrides `delegation_timeout`. After it, the tool returns `"Timed out waiting for peer agent reply."`. |

These top-level fields control delegation:

| Field | Type | Default | Description |
|---|---|---|---|
| `max_delegation_depth` | `int` | `3` | Most nested delegations one request may make (this agent → peer → peer …). A deeper call, or a call back into an agent already working on the request, is refused with an error the model sees. |
| `delegation_timeout` | `float \| None` | `None` | Default seconds to wait for any cross-agent. `None` = no limit beyond the peer's own `max_invocation_time`. |
| `include_broadcast` | `bool` | `false` | Also add the `broadcast_to_agents` tool. |

#### `approval`

Optional. Human approval for matching tool calls -- see [Human-in-the-loop approval](../approval.md#yaml-configuration-superagent).

| Field | Type | Default | Description |
|---|---|---|---|
| `tools` | `list[str]` | **required** | Glob patterns of tool names that need approval. |
| `handler` | `"webhook" \| "queue"` | `"webhook"` | `callback` is rejected: it needs a Python function. |
| `webhook_url` | `str \| None` | `None` | Required for `webhook`. |
| `webhook_secret` | `str \| None` | `None` | HMAC secret for the `X-Promptise-Signature` header (`webhook` only). Use `${ENV_VAR}`. |
| `timeout` | `float` | `300` | Seconds to wait for a decision. |
| `on_timeout` | `"deny" \| "allow"` | `"deny"` | What happens when nobody answers. |

A `queue` handler is answered by code in the same process. `promptise agent` asks each request as a y/N question when you run it at a terminal; with piped input nobody can answer, so requests time out.

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
    bearer_token: "${API_TOKEN}"
```

Before resolving, the loader loads the nearest `.env` file -- the same file, by the same rule, as the `promptise` CLI and model setup: the working directory or a parent up to the project root (the folder with `pyproject.toml` or `.git`), never overriding a variable that is already set. `PROMPTISE_NO_DOTENV=1` turns it off. So `python run_agent.py` and `promptise agent` see the same variables.

Call `loader.validate_env_vars()` to check which variables are missing before resolving:

```python
loader = SuperAgentLoader.from_file("agent.superagent")
missing = loader.validate_env_vars()
if missing:
    print(f"Set these env vars: {', '.join(missing)}")
else:
    loader.resolve_env_vars()
```

## Teams of agents (`cross_agents`)

A file can list other `.superagent` files under `cross_agents:`. Each becomes an `ask_agent_<name>` tool of this agent. The referenced files can have cross-agents of their own.

The loader resolves references at every depth. If agent A references agent B and agent B references agent A, the loader raises a `SuperAgentError` with the full reference chain.

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
    # its own references are in cross_loader.cross_loaders
```

Build a team with `build_superagent()` (or run it with `promptise agent main.superagent`):

```python
from promptise import build_superagent

agent = await build_superagent("main.superagent")
try:
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "..."}]})
finally:
    await agent.shutdown()   # shuts down every agent in the team
```

Every cross-agent is built from its whole file -- servers, approval, memory, guardrails, its own cross-agents -- and the whole team is built in the running event loop. Build, use and shut down the team in the same task (one `asyncio.run`): MCP sessions, stdio ones above all, belong to the task that opened them.

`config.to_build_kwargs()` cannot build cross-agents, because each is an agent of its own. For a file with `cross_agents:` it warns and leaves them out; use `build_superagent()`, or build the peers yourself and pass `to_build_kwargs(cross_agents={...})`.

At run time, delegation is bounded by `max_delegation_depth` (default 3) and by loop detection, and each call can be limited with `timeout:` / `delegation_timeout:`. See [Cross-Agent Delegation](cross-agent.md#delegation-limits).

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
kwargs = config.to_build_kwargs()      # warns if the file has cross_agents
agent = await build_agent(**kwargs)

# Or build the agent and its whole team
agent = await build_superagent("agent.superagent")
```

## API Summary

| Symbol | Import | Description |
|---|---|---|
| `SuperAgentLoader` | `from promptise.superagent import SuperAgentLoader` | Loads, validates, and resolves `.superagent` files. Key methods: `from_file()`, `resolve_env_vars()`, `resolve_cross_agents()`, `to_agent_config()`. |
| `load_superagent_file()` | `from promptise.superagent import load_superagent_file` | Convenience function that loads (reading `.env`), resolves env vars, and resolves cross-agent refs at every depth in one call. Returns `(loader, cross_loaders)`. |
| `build_superagent()` | `from promptise import build_superagent` | `await build_superagent(path_or_loader, *, model=None, instructions=None, trace=None, extra_servers=None)` -- builds the agent and every cross-agent in its file, at any depth, in the running event loop. `shutdown()` on the result shuts down the team. Overrides apply to the top agent. |
| `SuperAgentSchema` | `from promptise.superagent_schema import SuperAgentSchema` | Pydantic model for the full `.superagent` YAML schema. |
| `SuperAgentConfig` | `from promptise.superagent import SuperAgentConfig` | Processed config with `to_build_kwargs(*, cross_agents=None)` for `build_agent()`. |

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
