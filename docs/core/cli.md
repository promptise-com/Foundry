# CLI Reference

The `promptise` CLI provides commands for listing tools, running interactive agent sessions, serving an MCP server from a module (`serve`), turning an existing API into an MCP server (`mcpcast`), checking model providers (`models`), and managing runtime processes.

```bash
promptise --version
```

---

## Global Options

| Flag | Description |
|---|---|
| `--version` | Print version and exit |
| `--help` | Show help for any command |

---

## `promptise list-tools` -- List Available Tools

Discover tools exposed by MCP servers without writing code.

```bash
# List tools from an HTTP MCP server
promptise list-tools --model-id openai:gpt-5-mini --http "name=my_tools url=http://localhost:8000/mcp"

# List tools from a stdio MCP server
promptise list-tools --model-id openai:gpt-5-mini --stdio "name=echo command=python args='-m mytools.server'"
```

### Options

| Flag | Description |
|---|---|
| `--model-id` | LLM model identifier (required, e.g. `openai:gpt-5-mini`) |
| `--stdio` | Add a stdio MCP server (repeatable) |
| `--http` | Add an HTTP MCP server (repeatable) |
| `--instructions` | Optional system prompt override |

### Server Spec Syntax

Both `--stdio` and `--http` accept a quoted string of `key=value` pairs:

**HTTP servers:**

```bash
--http "name=my_tools url=http://localhost:8000/mcp"
```

| Key | Required | Description |
|---|---|---|
| `name` | yes | Server name (used as identifier) |
| `url` | yes | HTTP endpoint URL |

**Stdio servers:**

```bash
--stdio "name=echo command=python args='-m mytools.server --port 3333' env.API_KEY=xyz"
```

| Key | Required | Description |
|---|---|---|
| `name` | yes | Server name |
| `command` | yes | Executable to run |
| `args` | no | Command arguments (quote for spaces) |
| `env.*` | no | Environment variables (e.g. `env.API_KEY=xyz`) |
| `cwd` | no | Working directory |

Repeat `--stdio` or `--http` for multiple servers.

---

## `promptise run` -- Interactive REPL Session

Start an interactive REPL session with inline MCP server specs (no `.superagent` file needed).

```bash
promptise run --model-id openai:gpt-5-mini \
    --http "name=tools url=http://localhost:8000/mcp"
```

### Options

| Flag | Description |
|---|---|
| `--model-id` | LLM model identifier (required, e.g. `openai:gpt-5-mini`) |
| `--stdio` | Add a stdio MCP server (repeatable) |
| `--http` | Add an HTTP MCP server (repeatable) |
| `--instructions` | Optional system prompt override |
| `--trace/--no-trace` | Print tool invocations and results (default: `--trace`) |
| `--raw/--no-raw` | Also print raw result object (default: `--no-raw`) |

---

## `promptise agent` -- Agent from .superagent File

Run an interactive agent session from a `.superagent` configuration file. CLI flags override file settings.

```bash
# From a .superagent file
promptise agent config.superagent

# Override model
promptise agent config.superagent --model-id openai:gpt-5-mini

# With additional servers from CLI
promptise agent config.superagent --http "name=extra url=http://localhost:9000/mcp"
```

### Options

| Flag | Description |
|---|---|
| `--model-id` | Override model from config file |
| `--instructions` | Override instructions from config file |
| `--trace/--no-trace` | Override trace setting from config file |
| `--stdio` | Additional stdio server (merged with config file servers) |
| `--http` | Additional HTTP server (merged with config file servers) |
| `--raw/--no-raw` | Also print raw result object |

---

## `promptise validate` -- Validate .superagent File

Validate a `.superagent` configuration file without building the agent.

```bash
# Full validation
promptise validate my_agent.superagent

# Skip environment variable checks
promptise validate my_agent.superagent --no-check-env

# Skip cross-agent reference checks
promptise validate my_agent.superagent --no-check-refs
```

### Options

| Flag | Description |
|---|---|
| `--check-env/--no-check-env` | Check environment variable availability (default: `--check-env`) |
| `--check-refs/--no-check-refs` | Validate cross-agent references (default: `--check-refs`) |

Performs YAML syntax checks, schema validation, environment variable availability checks, and cross-agent reference validation.

---

## `promptise init` -- Generate Template .superagent File

Create a starter `.superagent` configuration file with common patterns.

```bash
# Basic template
promptise init

# HTTP server template with auth headers
promptise init --output api_agent.superagent --template http

# Full-featured template
promptise init -o advanced.superagent -t advanced --force
```

### Options

| Flag | Description |
|---|---|
| `--output`, `-o` | Output file path (default: `agent.superagent`) |
| `--template`, `-t` | Template type: `basic`, `http`, `stdio`, `cross-agent`, `advanced` |
| `--force/--no-force` | Overwrite existing file |

---

## `promptise serve` -- Run an MCP Server

Run an `MCPServer` from a Python module over stdio, HTTP or SSE -- the MCP equivalent of `uvicorn myapp:app`. The target is `module.path:attribute`; the CLI imports it, checks that it is an `MCPServer` instance (exit code 1 otherwise) and serves it.

```bash
# stdio, for Claude Desktop, Claude Code and Cursor
promptise serve myapp.server:server

# HTTP on the loopback address, with the live dashboard
promptise serve myapp.server:server -t http --port 8080 --dashboard

# A public bind behind a gateway: name the hosts (and browser origins) you serve
promptise serve myapp.server:server -t http --host 0.0.0.0 \
    --allowed-host api.example.com --allowed-origin https://app.example.com

# Development: restart on source changes
promptise serve myapp.server:server -t http --reload
```

### Options

| Flag | Description |
|---|---|
| `TARGET` (argument) | `module.path:attribute` naming the `MCPServer` instance (e.g. `myapp.server:server`) |
| `--transport`, `-t` | `stdio` (default), `http`, or `sse` |
| `--host` | Bind host for HTTP/SSE (default: `127.0.0.1`) |
| `--port`, `-p` | Bind port for HTTP/SSE (default: `8080`) |
| `--dashboard` | Live terminal dashboard (HTTP/SSE only; a warning is printed with `stdio`, whose terminal is the protocol stream) |
| `--reload` | Hot-reload on source changes (development only) |
| `--allowed-host HOST` | `Host` header value to accept on HTTP/SSE, e.g. `api.example.com` or `api.example.com:*` (repeatable). A loopback bind validates `Host` and `Origin` against the loopback names by default and the list **adds** to them (a reverse proxy forwarding the public `Host` to `127.0.0.1`); a non-loopback bind has no restriction until this names the hosts it serves |
| `--allowed-origin ORIGIN` | `Origin` header value to accept for browser clients, e.g. `https://app.example.com` (repeatable). Requests without an `Origin` header (every non-browser MCP client) always pass. Requires `--allowed-host` on a non-loopback bind |

See [Deployment](../mcp/server/deployment.md#host-and-origin-validation) for what the validation protects against (DNS rebinding), the reverse-proxy setup and the CORS configuration that complements it.

---

## `promptise mcpcast` -- Generate an MCP Server from an API

Turn an existing API (OpenAPI 3.x or Swagger 2) into a curated, safe, agent-ready MCP server, emitted as editable code. The command parses the spec, classifies every operation's risk (`read`, `write`, `destructive`, `financial`), designs the tool surface (with a model, or deterministically with `--no-curate`), and writes `<name>-mcp/`: `mcpcast.plan.yaml` (the editable source of truth), the server as an installable `<name>_mcp/` package (config, HTTP client, approval gate, one tools module per resource — regenerated from the plan), a `server.py` launcher, a `tests/` suite, `README.md` (Claude Desktop / Claude Code / Cursor install snippets), and — written once — `pyproject.toml`, `Dockerfile`, `.env.example` and `.gitignore`. Read-only by default: write, destructive and financial tools are only generated under `--profile standard` / `full`, and every non-read tool is approval-gated server-side by `ApprovalGateMiddleware`.

```bash
# No arguments: the guided setup — a full-screen terminal wizard
promptise mcpcast

# Deterministic, fully offline: one read tool per GET operation
promptise mcpcast openapi.yaml --no-curate

# LLM-curated surface, reads + approval-gated writes, review the plan before writing
promptise mcpcast https://api.example.com/openapi.json --profile standard --review

# Everything exposed, API-key auth for MCP clients, then score it with a real agent
promptise mcpcast openapi.yaml --profile full --auth api-key --eval

# Regenerate server.py next to an edited plan, then serve a fresh one over HTTP
promptise mcpcast mcpcast.plan.yaml
promptise mcpcast openapi.yaml --no-curate --serve -t http
```

### Options

| Flag | Description |
|---|---|
| `SPEC` (argument) | OpenAPI spec as a file path, URL, or inline JSON -- or an existing `mcpcast.plan.yaml` to regenerate the server from an edited plan. **Omit it to open the [guided setup](../mcpcast/guided-setup.md)**, a full-screen terminal wizard that collects every option below |
| `--interactive`, `-i` | Open the guided setup pre-filled with `SPEC`, `--base-url`, `--out`, `--model`, `--eval-tasks` and `--force`. Every other flag is refused together with it -- choose it in the wizard. Without a terminal (CI, a pipe) the wizard exits with code 1 and prints the non-interactive form to run |
| `--output`, `--out`, `-o` | Output directory (default: `./<name>-mcp`, or the plan's own directory when `SPEC` is a `mcpcast.plan.yaml`). A directory that already has files in it -- an mcpcast project or anything else -- is refused without `--force` |
| `--force` | Write into an output directory that already has files in it: an existing mcpcast project (so an edited plan is never replaced by a re-derived one without saying so) or any other non-empty directory, whose `README.md`, `server.py`, `tests/` and package would be replaced |
| `--base-url` | API base URL; required when the spec declares no absolute server URL and was not fetched from a URL it can be resolved against |
| `--profile` | Safety profile: `read-only` (default), `standard`, `full` |
| `--auth` | Upstream auth: `passthrough` (default; each caller forwards its own `Authorization` header over HTTP/SSE), `env-token` (one credential from `MCPCAST_UPSTREAM_TOKEN` — for stdio/desktop clients; binds loopback only unless `--public`), `api-key` (per-tenant upstream tokens), `none` (no credentials, loopback only unless `--public`) |
| `--approval` | Who approves gated calls: `elicitation`, `pending` (default: `pending` with `--auth api-key`, `elicitation` otherwise) |
| `--name` | Server name (default: derived from the spec title) |
| `--curate/--no-curate` | LLM-assisted tool design (default: `--curate`); `--no-curate` is fully offline and maps one tool per operation |
| `--model` | Model for curation and `--eval` (default: `openai:gpt-5-mini`). Any provider string works -- `azure:chat-prod`, `foundry:Llama-3.3-70B-Instruct`, `bedrock:...` -- see [`promptise models`](#promptise-models-model-providers) |
| `--max-tools` | Tool budget (default: 25 with curation; unlimited with `--no-curate`) |
| `--review` | Print the kept / not-exposed tables and confirm before writing. Every cell that comes from the spec, the plan or the model (tool names, descriptions, paths, parameter names, operation ids, drop reasons) is rendered inert: Rich markup is escaped and terminal control characters (ESC, the C1 range, DEL) become spaces, so a hostile spec cannot blank or rewrite the row of a destructive tool before you confirm |
| `--yes`, `-y` | With `--review`: write without asking |
| `--eval` | Score the generated server with a real agent (Agent Readiness Score); writes `eval/tasks.yaml` and `eval/report.md`. When live reads would go out with the placeholder credential a hint asks for `MCPCAST_EVAL_AUTHORIZATION`; an existing `MCPCAST_UPSTREAM_TOKEN` (env-token) or an `MCPCAST_UPSTREAM_TOKENS` entry for the evaluation tenant `mcpcast-eval` (api-key) counts as configured, as does `MCPCAST_EVAL_HEADERS` |
| `--eval-tasks` | Number of tasks to generate for `--eval` (default: `20`) |
| `--serve` | Run the generated server immediately after writing it |
| `--transport`, `-t` | With `--serve`: `stdio` (default), `http`, or `sse` |
| `--host` | With `--serve`: bind host for HTTP/SSE (default: `127.0.0.1`) |
| `--port`, `-p` | With `--serve`: bind port for HTTP/SSE (default: `8080`) |
| `--public` | With `--serve`: let an `env-token` or `none` server bind a non-loopback address. Such a server has no MCP-level authentication and acts with the operator's credential -- only behind an authenticating gateway. Refused with exit code 2 without `--serve`, and -- once the plan is known, before anything is written -- for an `api-key` or `passthrough` project (`--public only applies to --auth env-token / none; this project uses api-key, which already binds any host`): those modes authenticate every caller and have no such switch |

**Plain-http upstreams.** When the plan's `base_url` (or an operation-level server) is a non-loopback `http://` host and the auth mode carries a credential (`env-token`, `api-key`, `passthrough`), the summary is followed by a yellow warning naming the hosts: the generated server refuses every call to them with `UPSTREAM_INSECURE` until `MCPCAST_ALLOW_INSECURE_HTTP=1` is set where it runs (an intranet or staging API), or the plan points at an `https://` base URL. With `--auth none` no credential travels and nothing is said.

**Fetching the spec from a URL.** A URL `SPEC` may carry a credential for the download only -- `https://user:token@host/openapi.json` is sent as HTTP Basic auth, and `?api_key=...` reaches the server as typed. Nothing derived from that URL keeps it: the plan's `base_url` (when the spec declares no `servers`, the API is taken to live at the URL's origin), `spec_source`, the server name and the `Parsed ... from` line all use the URL without its userinfo, query string and fragment. A `base_url` that carries `user:password@` -- on `--base-url` or in a hand-edited plan -- is refused with a pointer at `--auth env-token` and `MCPCAST_UPSTREAM_TOKEN`, the supported way to give the generated server an upstream credential. The download is bounded three ways: `MCPCAST_MAX_SPEC_BYTES` (default 20 MiB, checked before the body is buffered), `MCPCAST_FETCH_SECONDS` (default 60; a wall-clock deadline for the whole download, since a server that trickles bytes would otherwise never trip httpx's per-read timeout) and `MCPCAST_MAX_SPEC_NODES` (default 2,000,000 nodes after YAML aliases are expanded, which is what turns a 1 KB alias bomb into gigabytes). Exceeding any of them prints `Error:` naming the variable to raise.

**Regenerating from a plan.** Pass `mcpcast.plan.yaml` as `SPEC` (a path, a URL or inline text — recognised by shape; an inline plan of any length is written into `./<name>-mcp` and shown as `Regenerating from plan <inline>`, never echoed) to rebuild the package, `server.py`, `tests/` and `README.md` next to it from the edited plan without re-curating (a plan fetched from a URL is written into `./<name>-mcp`); the plan file itself is not rewritten (comments survive), and neither are the scaffold files (`pyproject.toml`, `Dockerfile`, `.env.example`, `.gitignore`) once they exist. Tools modules the plan no longer produces are removed. Spec-only flags (`--profile`, `--auth`, `--approval`, `--base-url`, `--name`, `--max-tools`) are rejected with exit code 2 in this mode -- change those values in the plan file instead. With `--eval` the spec recorded in the plan is re-read for realistic mocks.

**Exit codes and output.** An invalid option exits with code 2; a spec, plan, curation or model failure (a missing provider key, an unimportable generated server) prints `Error: ...` and exits with code 1 (if curation fails after its retries, re-run with `--no-curate`); aborting at the `--review` prompt exits with code 1 without writing. A document that is OpenAPI-shaped but broken never produces a traceback: a defect of the whole document (`info` or `paths` that is not a mapping, nesting deeper than 256 levels, more than 2,000,000 nodes once YAML aliases are expanded) is an `Error:`, while a defect inside one operation (`parameters` that is not a list of mappings, a `requestBody.content` or schema that is a string) drops only that operation, with the reason recorded under `dropped:` in the plan; `operationId: 5` is read as `"5"` and `tags` that are not a list are ignored. All progress -- including the model calls of curation and `--eval` -- is printed to stderr, so with `--serve` over `stdio` stdout stays clean as the MCP protocol stream. `--serve` goes through the generated entry point, so an `--auth none` server still refuses a non-loopback `--host`.

See the [MCPcast guide](../mcp/server/mcpcast.md) for safety profiles, auth modes and the generated server's runtime contract, and the [MCPcast API reference](../api/mcpcast.md) for the Python API.

---

## `promptise models` -- Model Providers

Every place Promptise takes a model -- `build_agent(model=...)`, `.superagent` and `.agent` files, `promptise mcpcast --model`, `--model-id` on `run` / `agent` / `list-tools` -- accepts one `provider:model` string, and every string goes through the same registry (`promptise.models`). The `models` subcommand group shows that registry, diagnoses one string before you put it in a config file, and prints the environment variables a provider needs. Friendly aliases are accepted everywhere: `azure:` for Azure OpenAI, `foundry:` for the Azure AI Foundry model catalog, `gemini:`, `vertex:`, `bedrock:`, `mistral:`, `grok:`, `hf:`.

```bash
# Every provider: prefixes and aliases, install status, which env vars are set
promptise models list

# What one string resolves to and what is missing to use it (exit 1 if not usable).
# Configuration only -- except that a local or keyless server (Ollama, a vLLM on
# localhost) is checked for a listener, since it has no credential to check
promptise models check azure:chat-prod

# The same, plus a real one-token call to the model
promptise models check openai:gpt-5-mini --ping

# Export lines for a provider, with placeholders and where to find each value
promptise models env azure
```

### Subcommands

| Command | Description |
|---|---|
| `models list` | Table of every provider: the `provider=` name with its aliases, its route (`native` — the core integration — or `OpenAI-compatible`), whether its required env vars are `set` / `missing` / `none required`, and an example model string. Nothing needs installing for any row. |
| `models check <provider:model> [--ping]` | Explains what the string resolves to (alias → canonical name), what the model part means for that provider (a deployment name on Azure, a model id elsewhere), the route it will use (the exact OpenAI-compatible URL with the endpoint, region and project filled in from the environment -- `http://localhost:11434/v1` for a default Ollama -- or the native integration), each env var with its state -- `set`, `set (from /path/.env)` when the `.env` file supplied it, `MISSING`, `MISSING (VAR is exported but empty — unset it or give it a value)`, `unset (optional)` -- and where to find its value, and the provider's notes. When the string is not usable, every problem is listed with its fix (`  - ...` lines, the same text `build_agent()` puts in its `ModelSetupError`). Otherwise it prints `Configuration OK` -- the settings are in place, which is not the same as a working model: nothing is called. For a keyless provider (Ollama) or an endpoint on this machine (`OPENAI_BASE_URL=http://localhost:8000/v1`) it also tries a TCP connection (1 s timeout, nothing is sent) and reports `Not reachable.` with what to try (`nothing is listening at http://localhost:11434 — is Ollama running? (ollama serve) ...`) when nothing answers. `--ping` additionally makes a real one-token call; when it fails, the provider's error is followed by a `→` line saying what it most likely means: nothing listening, a timeout, a rejected key (naming the variable), a model or deployment that does not exist (`ollama pull <model>` for Ollama), missing model access, an account out of credits, a rate limit or a provider outage. |
| `models env <provider>` | Prints `export` lines for the provider's env vars with example values and a hint per line — paste them into `.env`. Any name or alias works (`azure`, `aoai`, `bedrock`). |

**Exit codes.** `models check` exits with `1` when the string is not usable yet (a required env var unset, or `native=True` without its package), when a local or keyless server is not listening, or when the `--ping` call fails -- so it works as a preflight step in CI and container entrypoints; `0` when the configuration is complete (and, for a local server, something is listening). `models env` exits with `2` for an unknown provider. `models list` always exits `0`.

With nothing configured, `promptise models check azure:chat-prod` prints:

```text
$ promptise models check azure:chat-prod
azure:chat-prod → azure_openai:chat-prod  (Azure OpenAI (OpenAI models deployed in Azure AI Foundry))
  model part: chat-prod — in the string form, your DEPLOYMENT name (Foundry → Deployments → Name); with Model(...), the model name (gpt-4o) — the deployment goes in deployment=
  route: native integration (core)
  AZURE_OPENAI_ENDPOINT: MISSING — Azure AI Foundry portal → your resource → Overview → Endpoint (https://<resource>.openai.azure.com/, no path)
  AZURE_OPENAI_API_KEY: MISSING — Azure AI Foundry portal → your resource → Keys and Endpoint → KEY 1 (or Entra ID: pass extra={'azure_ad_token_provider': ...})
  OPENAI_API_VERSION: MISSING — the REST API version your deployment supports, e.g. 2024-10-21 (Azure docs → 'API version lifecycle')
  Azure routes requests by deployment name, not model name: Model('gpt-4o', provider='azure', deployment='chat-prod', ...).
Not usable yet.
  - AZURE_OPENAI_ENDPOINT is not set — Azure AI Foundry portal → your resource → Overview → Endpoint (https://<resource>.openai.azure.com/, no path) (e.g. https://my-resource.openai.azure.com/)
  - AZURE_OPENAI_API_KEY is not set — Azure AI Foundry portal → your resource → Keys and Endpoint → KEY 1 (or Entra ID: pass extra={'azure_ad_token_provider': ...})
  - OPENAI_API_VERSION is not set — the REST API version your deployment supports, e.g. 2024-10-21 (Azure docs → 'API version lifecycle') (e.g. 2024-10-21)
  - put AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION in a .env file next to your script (loaded automatically; a variable already exported with a non-empty value wins, an empty one is filled from the file), export it, or pass it in code: Model(..., endpoint=, api_key=, api_version=) — see promptise models env azure
$ echo $?
1
```

On a machine where Ollama is not running, `promptise models check ollama:llama3.1` says so instead of passing:

```text
$ promptise models check ollama:llama3.1
ollama:llama3.1 → ollama:llama3.1  (Ollama (local models))
  model part: llama3.1 — a model you have pulled (`ollama pull llama3.1`)
  route: OpenAI-compatible endpoint http://localhost:11434/v1 (core, nothing to install)
  OLLAMA_HOST: unset (optional) — only if Ollama is not on the default http://localhost:11434
  No API key. The model must support tool calling to drive MCP tools.
Not reachable. The configuration is complete, but nothing is listening at http://localhost:11434 — is Ollama running? (ollama serve) If it runs on another host or port, set OLLAMA_HOST.
$ echo $?
1
```

and `--ping` against a server that is down names the cause after the error:

```text
Pinging… failed
OpenAIConnectionError: Connection error.
  → nothing is listening at http://localhost:11434 — is Ollama running? (ollama serve) If it runs on another host or port, set OLLAMA_HOST.
```

A variable that is exported but empty (`export OPENAI_API_KEY=` left in a shell profile) is reported as such -- `OPENAI_API_KEY: MISSING (OPENAI_API_KEY is exported but empty — unset it or give it a value)` -- and one the `.env` file filled names the file: `OPENAI_API_KEY: set (from /home/me/project/.env)`.

`promptise models env azure` then gives you the three lines to fill in:

```text
$ promptise models env azure
# Azure OpenAI (OpenAI models deployed in Azure AI Foundry) — model string example: azure:chat-prod
export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/
#   ↳ Azure AI Foundry portal → your resource → Overview → Endpoint (https://<resource>.openai.azure.com/, no path)
export AZURE_OPENAI_API_KEY=...
#   ↳ Azure AI Foundry portal → your resource → Keys and Endpoint → KEY 1 (or Entra ID: pass extra={'azure_ad_token_provider': ...})
export OPENAI_API_VERSION=2024-10-21
#   ↳ the REST API version your deployment supports, e.g. 2024-10-21 (Azure docs → 'API version lifecycle')
```

The same checks run inside `build_agent()`: a string that `models check` reports as not usable raises a `ModelSetupError` with the same text instead of a provider stack trace. See [Model Setup](../getting-started/model-setup.md) for every provider and the [Models API reference](../api/models.md) for `resolve_model()` / `check_model()`.

---

## `promptise runtime` -- Process Management

The `runtime` subcommand group manages long-running agent processes.

### `promptise runtime start` -- Start Processes

Start agent process(es) from a `.agent` manifest file or directory.

```bash
# Start a single manifest
promptise runtime start agents/watcher.agent

# Start all manifests in a directory
promptise runtime start agents/

# Override process name
promptise runtime start agents/watcher.agent --name custom-watcher

# Run in the background
promptise runtime start agents/watcher.agent --detach

# Start with live monitoring dashboard
promptise runtime start agents/ --dashboard
```

| Flag | Description |
|---|---|
| `--name`, `-n` | Override process name from manifest |
| `--detach`, `-d` | Run in the background |
| `--dashboard` | Enable live terminal monitoring dashboard |

Press `Ctrl+C` to stop all processes (in foreground mode).

### `promptise runtime stop` -- Stop a Process

```bash
promptise runtime stop data-watcher
promptise runtime stop data-watcher --force
```

| Flag | Description |
|---|---|
| `--force` | Force stop the process |

### `promptise runtime status` -- Show Status

```bash
# Show status of all processes
promptise runtime status

# Show status of a specific process
promptise runtime status data-watcher

# Output as JSON
promptise runtime status --json
```

| Flag | Description |
|---|---|
| `--json` | Output status as JSON |

### `promptise runtime logs` -- View Journal

Show journal entries for a running process.

```bash
# Show last 20 entries
promptise runtime logs data-watcher

# Show last 50 entries
promptise runtime logs data-watcher --lines 50

# Follow new entries
promptise runtime logs data-watcher --follow

# Custom journal path
promptise runtime logs data-watcher --journal-path .promptise/journal
```

| Flag | Description |
|---|---|
| `--lines`, `-n` | Number of entries to show (default: 20) |
| `--follow`, `-f` | Follow new entries in real time |
| `--journal-path` | Journal directory (default: `.promptise/journal`) |

### `promptise runtime restart` -- Restart a Process

```bash
promptise runtime restart data-watcher
```

### `promptise runtime validate` -- Validate a Manifest

Check a `.agent` manifest file for schema errors and warnings.

```bash
promptise runtime validate agents/watcher.agent
```

Outputs a summary table with name, model, version, instructions, server count, and trigger count. Warnings are printed for common issues like missing instructions, triggers, or servers.

### `promptise runtime init` -- Generate a Template

Create a template `.agent` manifest file to get started quickly.

```bash
# Basic template
promptise runtime init -o my-agent.agent

# Cron-based template
promptise runtime init --template cron -o watcher.agent

# Webhook template
promptise runtime init --template webhook -o handler.agent

# Full-featured template
promptise runtime init --template full -o production.agent

# Overwrite existing file
promptise runtime init --template cron -o watcher.agent --force
```

| Flag | Description |
|---|---|
| `--output`, `-o` | Output file path (default: `agent.agent`) |
| `--template`, `-t` | Template type: `basic`, `cron`, `webhook`, `full` |
| `--force` | Overwrite existing file |

---

## Common Workflows

### Develop and Test an Agent Process

```bash
# 1. Generate a template manifest
promptise runtime init --template cron -o watcher.agent

# 2. Edit the manifest with your instructions and servers
# (edit watcher.agent)

# 3. Validate the manifest
promptise runtime validate watcher.agent

# 4. Start the process
promptise runtime start watcher.agent

# 5. View logs in another terminal
promptise runtime logs pipeline-watcher --follow
```

### MCPcast an Existing API

```bash
# 1. Generate a curated, read-only MCP server from your OpenAPI spec and review the plan
promptise mcpcast openapi.yaml --review

# 2. Edit mcpcast.plan.yaml (names, descriptions, hidden params, dropped list), then regenerate
promptise mcpcast my-api-mcp/mcpcast.plan.yaml

# 3. Score how well a real agent drives the result
promptise mcpcast openapi.yaml --profile standard --eval

# 4. Serve it for any MCP client
cd my-api-mcp && promptise serve server:server --transport http
```

### Switch to Another Model Provider

```bash
# 1. See what the string needs (exit 1 until it is usable)
promptise models check azure:chat-prod

# 2. Get the export lines, fill them in
promptise models env azure >> .env

# 3. Confirm with a real one-token call, then use the same string anywhere
promptise models check azure:chat-prod --ping
promptise run --model-id azure:chat-prod --http "name=tools url=http://localhost:8000/mcp"
```

### Discover Available Tools

```bash
# Check what tools an MCP server exposes
promptise list-tools --model-id openai:gpt-5-mini \
    --http "name=my_tools url=http://localhost:8000/mcp"
```

### Run a Quick Interactive Session

```bash
# Inline servers (no .superagent file)
promptise run --model-id openai:gpt-5-mini \
    --http "name=tools url=http://localhost:8000/mcp"

# From a .superagent file
promptise agent config.superagent
```

---

!!! tip "Environment variables"
    A `.env` file in the working directory (or a parent, up to the project root -- the directory holding `pyproject.toml` or `.git`) is loaded automatically -- by the CLI and by plain scripts alike, never overriding a variable that is already set; `PROMPTISE_NO_DOTENV=1` turns it off. Only a regular file is loaded -- on POSIX one you own that nobody else can write -- and anything else is skipped with a warning naming it. The CLI loads it once, before any command runs, and prints `.env loaded from <path>` on stderr; a file it cannot read (permissions, not UTF-8) is one `Error:` line with exit code 2 naming the file, while `promptise --version` and `promptise --help` never touch it. Put `OPENAI_API_KEY` there for seamless usage; `promptise models env <provider>` prints the lines any other provider needs. See [Configuration & Secrets](../getting-started/configuration.md).

---

## What's Next?

- [Agent Runtime Overview](../runtime/index.md) -- architecture and lifecycle concepts
- [Agent Manifests](../runtime/manifests.md) -- `.agent` YAML format reference
- [Observability](observability.md) -- `--observe` flag and observability
