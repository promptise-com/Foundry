# Examples Gallery

Complete, runnable examples demonstrating every capability of Promptise Foundry. All examples use real LLM calls -- no mocks or stubs.

Every example is designed to run end-to-end with just an API key set. The default model across all examples is `openai:gpt-5-mini` (fast, affordable, reliable).

---

## Running Examples

```bash
# 1. Set your API key
export OPENAI_API_KEY=sk-...

# 2. Run any example directly
python examples/prompts/01_blocks_composition.py
python examples/mcp/agent.py
```

Some examples require a running MCP server. Where needed, start the server in a separate terminal first:

```bash
# Terminal 1 -- start the server
python examples/mcp/server.py

# Terminal 2 -- run the client or agent
python examples/mcp/agent.py
```

---

## MCP Server & Client

Build production MCP servers with authentication, middleware, and Pydantic validation, then connect via raw clients or LLM agents.

| File | Description | Difficulty |
|------|-------------|------------|
| `examples/mcp/server.py` | Production MCP server with JWT auth, middleware stack, Pydantic models, caching, routers, background tasks, and live dashboard | Intermediate |
| `examples/mcp/agent.py` | LLM agent connecting to an MCP server with JWT authentication and natural-language tool invocation | Beginner |
| `examples/mcp/client.py` | Raw MCP client with multi-server routing, token acquisition, LangChain tool adapters, and tracing callbacks | Intermediate |
| `examples/mcp/client_auth_errors.py` | What a client sees when an MCP server rejects its credentials: a typed `MCPConnectionRejectedError` (401) without and with the wrong API key, then a successful call with the right one. No LLM needed | Beginner |
| `examples/mcp/multi_tenant_agent.py` | One agent serving two tenants concurrently over a real HTTP server: each caller's token reaches the server, a token for another service is refused, cached results stay per caller, and each tenant lists only its tools. Runs offline (scripted model) | Intermediate |
| `examples/mcp/http_server_restart.py` | A client that survives a server redeploy: the restarted server forgets the session, the client re-initialises, retries the call and re-discovers the new tool set. Also the `/health` and `/health/ready` probes and the typed 404 for a URL without `/mcp`. No LLM needed | Intermediate |
| `examples/mcp/mcpcast_petstore/run.py` | Generate an MCP server from an OpenAPI spec with `promptise mcpcast`, drive it with a real agent over stdio against the live Petstore API, and show a destructive tool blocked by the server-side approval gate | Intermediate |
| `examples/mcp/mcpcast_storefront_lab/run.py` | A SaaS storefront API MCPcast end to end: generated tools with risk classes, a real agent completing a business task, a refund held by the server-side approval gate until a second human approves, and an Agent Readiness Score before and after fixing the plan | Advanced |
| `examples/mcp/mcpcast_wizard_lab/run.py` | The guided setup as a lab: start a real Helpdesk API, drive `promptise mcpcast` in the terminal (detection, model, profile counts, auth, review), then let the lab prove the result — computed review warnings on what the model wrote, a real agent over MCP stdio against the live app, a write denied by the approval gate, a refund tool that the `standard` profile never generated, and an Agent Readiness Score | Intermediate |
| `examples/mcp/mcpcast_fastapi_app/run.py` | Make your own FastAPI app MCP-ready: start the app, generate an MCP server from its live `/openapi.json` with `promptise mcpcast` (reads open, writes gated, admin/deprecated/health dropped with reasons), drive it with a real agent over MCP stdio, and show a write denied fail-closed then executed once a human approves | Intermediate |

**Quick start:**

```bash
# Terminal 1
python examples/mcp/server.py

# Terminal 2 -- agent (requires OPENAI_API_KEY)
python examples/mcp/agent.py

# Terminal 2 -- or raw client (no LLM required)
python examples/mcp/client.py
```

**What you will learn:**

- Creating an `MCPServer` with `@server.tool()` decorators
- JWT authentication with `AuthMiddleware` and role-based guards
- Pydantic model validation for tool parameters (nested models, constraints)
- `MCPRouter` for grouping tools under prefixes
- `MCPClient` and `MCPMultiClient` for programmatic server access
- `MCPToolAdapter` for converting MCP tools to LangChain `BaseTool` instances

---

## Agent Identity

Give an agent a stable, traceable identity — and let the resources it calls
cryptographically verify and attribute the caller. The first two run on a laptop
with no cloud and no API key.

| Example | What it demonstrates | Level |
| --- | --- | --- |
| `examples/identity/local/app.py` | A **local identity** (`agent_id`, attribution, `claims()`) — the 30-second on-ramp | Beginner |
| `examples/identity/verifiable_mcp/app.py` | The **headline value** end-to-end: agent presents a signed JWT, a server's `JwksAuth` verifies + attributes it (and rejects a wrong-audience token) | Intermediate |
| `examples/identity/github_actions/` | Production OIDC from GitHub Actions (`from_oidc`) — exercised in CI against a real token | Intermediate |
| `examples/identity/{aws_lambda,gke_pod,aks_workload,spire}/` | Production credential federation on each platform (`from_aws` / `from_gcp` / `from_entra` / `from_spiffe`) | Advanced |

```bash
python examples/identity/local/app.py
python examples/identity/verifiable_mcp/app.py
```

See [`examples/identity/README.md`](https://github.com/promptise-com/foundry/blob/main/examples/identity/README.md) for the full map and the [Agent Identity docs](../identity/overview.md).

---

## Prompt Engineering

Compose prompts from blocks, evolve them across conversation turns, and enhance them with strategies, guards, and chaining.

| File | Description | Difficulty |
|------|-------------|------------|
| `examples/prompts/01_blocks_composition.py` | Composable prompt blocks, priority-based assembly, conditional blocks, `@blocks` decorator | Beginner |
| `examples/prompts/02_conversation_flow.py` | Multi-phase customer support agent where the system prompt evolves per turn | Intermediate |
| `examples/prompts/04_inspector_debugging.py` | Full prompt tracing with `PromptInspector` -- see blocks included, tokens used, and execution path | Intermediate |
| `examples/prompts/05_full_integration.py` | Both layers combined (blocks + flow + inspector) in a research analysis agent | Advanced |

### Labs

| Directory | Description | Difficulty |
|-----------|-------------|------------|
| `examples/prompts/content_studio_lab/` | AI Content Creation Studio demonstrating prompt blocks, flows, guards, strategies, context providers, chain operators, registry, inspector, and templates -- 9 runnable demos with real LLM calls | Advanced |

**Quick start:**

```bash
python examples/prompts/01_blocks_composition.py

# Flagship lab -- all prompt features in one studio
python examples/prompts/content_studio_lab/main.py
```

**Two-layer architecture:**

```
Layer 2: ConversationFlow     Turn-aware prompt evolution
         |
Layer 1: PromptBlocks         Composable prompt components
```

Each layer is independent. Use one or both depending on your needs.

---

## Agent Runtime

Turn stateless LLM agents into persistent, autonomous processes with triggers, crash recovery, and distributed coordination.

| File | Description | Difficulty |
|------|-------------|------------|
| `examples/runtime/pipeline_watcher.agent` | Declarative `.agent` manifest defining an autonomous pipeline watchdog process | Beginner |
| `examples/runtime/autonomous_agent.agent` | Another agent manifest example for autonomous operation | Beginner |
| `examples/runtime/server.py` | MCP server with pipeline monitoring tools (health checks, metrics, alerts, repairs) | Intermediate |
| `examples/runtime/main.py` | Full runtime API walkthrough covering `AgentProcess`, triggers, context, journal, and distributed coordination | Advanced |

### Labs

| Directory | Description | Difficulty |
|-----------|-------------|------------|
| `examples/runtime/data_pipeline_lab/` | Data Pipeline Monitoring System with 8 examples covering process lifecycle, triggers (cron, event, custom SQS), journal system (InMemory, File, ReplayEngine), AgentContext, ConversationBuffer, multi-process AgentRuntime, `.agent` manifest loading, and open mode with meta-tools -- all with real LLM calls | Advanced |

**Quick start:**

```bash
# Terminal 1 -- start the MCP server (tools for the agent)
python examples/runtime/server.py

# Terminal 2 -- run the full walkthrough
python examples/runtime/main.py

# Flagship lab -- all runtime features in one pipeline monitoring system
# Terminal 1 -- start the lab MCP server
python examples/runtime/data_pipeline_lab/tools_server.py
# Terminal 2 -- run the lab
python examples/runtime/data_pipeline_lab/main.py

# Or use the CLI directly
promptise runtime validate examples/runtime/pipeline_watcher.agent
promptise runtime start examples/runtime/pipeline_watcher.agent
```

**What you will learn:**

- Defining agent processes with `.agent` YAML manifests
- `AgentProcess` lifecycle: CREATED, STARTING, RUNNING, SUSPENDED, STOPPED
- Triggers: `CronTrigger`, `WebhookTrigger`, `FileWatchTrigger`, `EventTrigger`, `MessageTrigger`
- `AgentContext` for unified state, environment variables, and file mounts
- `Journal` and `ReplayEngine` for crash recovery
- `AgentRuntime` for managing multiple agent processes
- Open mode with meta-tools for self-modifying agents
- Distributed coordination with `RuntimeTransport` and `RuntimeCoordinator`

