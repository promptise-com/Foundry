---
title: Promptise Foundry FAQ — Production Python framework for AI agents and MCP
description: Honest, complete answers about Promptise Foundry — what it is, how it compares to LangChain, LangGraph, and CrewAI, what models it supports, security, MCP, autonomous runtime, governance, and production deployment.
keywords: Promptise Foundry FAQ, Python AI agent framework, MCP server framework, LangChain alternative, LangGraph alternative, CrewAI alternative, agentic AI, autonomous agents, Model Context Protocol
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "FAQPage",
  "mainEntity": [
    {"@type":"Question","name":"What is Promptise Foundry?","acceptedAnswer":{"@type":"Answer","text":"Promptise Foundry is a production-grade Python framework for building AI agents, MCP (Model Context Protocol) servers, prompt engineering systems, and autonomous runtimes. It is secure by default, model-agnostic, MCP-native, and designed for teams shipping agentic AI to production."}},
    {"@type":"Question","name":"How is Promptise Foundry different from LangChain?","acceptedAnswer":{"@type":"Answer","text":"LangChain is a general-purpose LLM toolkit with hundreds of integrations. Promptise Foundry is a focused production framework that ships one coherent stack covering agents, MCP servers, prompt engineering, and autonomous runtime, with built-in access control, audit trails, sandboxing, and governance. Promptise has fewer abstractions, no silent fallbacks, is async-first, and uses MCP-native tool discovery instead of manual wiring."}},
    {"@type":"Question","name":"How does Promptise Foundry compare to LangGraph?","acceptedAnswer":{"@type":"Answer","text":"LangGraph is an orchestration layer focused on stateful graphs. Promptise Foundry includes a full Agent Runtime with crash recovery via journals, governance (budget, health, mission), and five trigger types (cron, event, message, webhook, file watch) — built for long-running autonomous agents that survive restarts."}},
    {"@type":"Question","name":"How does Promptise Foundry compare to CrewAI?","acceptedAnswer":{"@type":"Answer","text":"CrewAI focuses on multi-agent role-playing workflows. Promptise Foundry covers single agents, multi-agent coordination via cross-agent delegation (ask_peer and broadcast over HTTP+JWT), plus the production infrastructure — MCP servers, governance, observability, sandboxing."}},
    {"@type":"Question","name":"What is MCP and why does Promptise Foundry use it?","acceptedAnswer":{"@type":"Answer","text":"MCP (Model Context Protocol) is the open standard for connecting LLMs to tools, resources, and prompts. Promptise Foundry is MCP-native — agents auto-discover tools from any MCP server URL, schemas convert to typed tools automatically, and the framework includes a production SDK for building MCP servers."}},
    {"@type":"Question","name":"Does Promptise Foundry have its own MCP client?","acceptedAnswer":{"@type":"Answer","text":"Yes. Promptise Foundry ships a native MCP client built from scratch — MCPClient for single servers, MCPMultiClient for N servers with unified tool list and auto-routing, and MCPToolAdapter for LangChain BaseTool conversion. HTTP, SSE, and stdio transports with bearer token and API key authentication. No third-party MCP client dependencies."}},
    {"@type":"Question","name":"Which LLM models does Promptise Foundry support?","acceptedAnswer":{"@type":"Answer","text":"Any model. Promptise Foundry is model-agnostic. Use a string like openai:gpt-5-mini, anthropic:claude-sonnet-4-5, or ollama:llama3, or pass any LangChain BaseChatModel directly. Switching providers requires changing one string."}},
    {"@type":"Question","name":"Is Promptise Foundry production-ready?","acceptedAnswer":{"@type":"Answer","text":"Yes. No stub implementations, no half-finished features, no NotImplementedError in shipped code. Access control, capability-based policies, audit logging, encrypted transport, sandboxed code execution, observability with 8 transporters, and crash-recovery via journals are built in. Every backend listed as a parameter option works."}},
    {"@type":"Question","name":"How do I install Promptise Foundry?","acceptedAnswer":{"@type":"Answer","text":"Run pip install promptise. Python 3.10 or newer is required. Optional extras for memory backends (Chroma, Mem0), sandboxing (Docker, gVisor), and observability (Prometheus, OpenTelemetry) are available."}},
    {"@type":"Question","name":"Does Promptise Foundry support self-hosted or local LLMs?","acceptedAnswer":{"@type":"Answer","text":"Yes. Use any Ollama model via the ollama:model-name string. Local embeddings via SentenceTransformers, local guardrail models (DeBERTa, GLiNER, Llama Guard), and local memory backends (ChromaDB) make air-gapped deployments fully supported."}},
    {"@type":"Question","name":"Is Promptise Foundry open source?","acceptedAnswer":{"@type":"Answer","text":"Yes — Apache 2.0 licensed. The source code is at github.com/promptise-com/foundry."}},
    {"@type":"Question","name":"What is the Agent Runtime?","acceptedAnswer":{"@type":"Answer","text":"The Agent Runtime is the operating system for autonomous AI agents. It turns stateless LLM calls into persistent, governed processes with lifecycle states (CREATED, STARTING, RUNNING, SUSPENDED, STOPPING, STOPPED, FAILED), trigger queues, heartbeat monitoring, concurrency control, conversation buffer, and journaled state for crash recovery."}},
    {"@type":"Question","name":"What trigger types can launch an autonomous agent?","acceptedAnswer":{"@type":"Answer","text":"Five built-in trigger types — CronTrigger (cron expressions), EventTrigger (EventBus subscription), MessageTrigger (topic-based pub/sub with wildcards), WebhookTrigger (HTTP POST with HMAC verification), and FileWatchTrigger (directory monitoring with glob patterns). Multiple triggers can compose on one process."}},
    {"@type":"Question","name":"How does Promptise Foundry handle crash recovery?","acceptedAnswer":{"@type":"Answer","text":"Through journals. InMemoryJournal and FileJournal record every state transition, trigger event, and invocation result. The ReplayEngine reconstructs state from a checkpoint plus replay log. When a process crashes, it restarts from the last known good state."}},
    {"@type":"Question","name":"Does Promptise Foundry handle multi-user access control?","acceptedAnswer":{"@type":"Answer","text":"Yes. JWTAuth (HS256), AsymmetricJWTAuth (RS256/ES256), and APIKeyAuth. Capability-based access policies, per-tool permission guards, per-user audit trails (HMAC-chained for tamper detection), and CallerContext propagation across the entire stack via async contextvars."}},
    {"@type":"Question","name":"How does Promptise Foundry handle prompt injection and security?","acceptedAnswer":{"@type":"Answer","text":"PromptiseSecurityScanner with six detection heads — DeBERTa-based prompt injection detection, PII detection (69 regex patterns), credential detection (96 patterns), GLiNER NER, content safety via Llama Guard or Azure AI, and custom rules. All models run locally."}},
    {"@type":"Question","name":"What memory backends does Promptise Foundry support?","acceptedAnswer":{"@type":"Answer","text":"Three providers — InMemoryProvider (testing), ChromaProvider (local vector search, persistent), and Mem0Provider (enterprise-grade graph search). The agent auto-searches memory before every invocation and injects relevant results with prompt-injection mitigation."}},
    {"@type":"Question","name":"Does Promptise Foundry support conversation persistence?","acceptedAnswer":{"@type":"Answer","text":"Yes. ConversationStore protocol with four backends — InMemoryConversationStore, SQLiteConversationStore, PostgresConversationStore, RedisConversationStore. The chat() method handles load, invoke, and persist automatically with session ownership enforcement."}},
    {"@type":"Question","name":"Does Promptise Foundry support semantic caching?","acceptedAnswer":{"@type":"Answer","text":"Yes. SemanticCache with in-memory or Redis backends. Per-user, per-session, or shared scope isolation. Serves cached responses for semantically similar queries — typically 30-50% cost reduction. GDPR purge_user() supported. Encrypted-at-rest option for Redis."}},
    {"@type":"Question","name":"Can I sandbox code execution in Promptise Foundry?","acceptedAnswer":{"@type":"Answer","text":"Yes. Docker-based sandbox with seccomp syscall filtering, capability dropping, read-only rootfs, resource limits, and network isolation. Optional gVisor kernel. Five agent tools auto-injected when sandbox is enabled. Path traversal and shell injection prevention built in."}},
    {"@type":"Question","name":"How does observability work?","acceptedAnswer":{"@type":"Answer","text":"Four levels — OFF, BASIC, STANDARD, FULL. Every LLM turn, tool call, token count, latency, retry, cache hit/miss is recorded. Eight transporters — HTML report, JSON file, structured log, console, Prometheus, OpenTelemetry, webhook, callback."}},
    {"@type":"Question","name":"What governance does the Agent Runtime provide?","acceptedAnswer":{"@type":"Answer","text":"Four governance subsystems — Budget (tool calls, LLM turns, cost units, irreversible actions), Health (anomaly detection), Mission (LLM-as-judge evaluation against success criteria), and Secrets (per-process credential context with TTL expiry, rotation, zero-fill revocation)."}},
    {"@type":"Question","name":"Does Promptise Foundry support multi-agent systems?","acceptedAnswer":{"@type":"Answer","text":"Yes. Cross-agent delegation via ask_peer (send a question to another agent over HTTP+JWT and await the answer) and broadcast (send to multiple peers in parallel with timeout). Graceful degradation if a peer fails."}},
    {"@type":"Question","name":"What is a SuperAgent file?","acceptedAnswer":{"@type":"Answer","text":"A .superagent YAML file that defines an entire agent declaratively — model, instructions, MCP servers, memory, sandbox, observability, cache, guardrails, cross-agents. Environment variable resolution via ${VAR}. Loadable via the CLI."}},
    {"@type":"Question","name":"What is the Open Mode?","acceptedAnswer":{"@type":"Answer","text":"Open Mode lets agents self-modify with 14 meta-tools — modify_instructions, create_tool, connect_mcp_server, add_trigger, remove_trigger, spawn_process, list_processes, store_memory, search_memory, forget_memory, list_capabilities, get_secret, check_budget, check_mission. Guardrails enforce limits and require sandbox for agent-written code."}},
    {"@type":"Question","name":"Does Promptise Foundry have prompt engineering features?","acceptedAnswer":{"@type":"Answer","text":"Yes. The @prompt decorator, 8 PromptBlock types with priority-based token budgeting, ConversationFlow for phase-based system prompts, 5 composable strategies (ChainOfThought, StructuredReasoning, SelfCritique, PlanAndExecute, Decompose), 4 perspectives (Analyst, Critic, Advisor, Creative), guards, 14 context providers, and a PromptInspector."}},
    {"@type":"Question","name":"Can I build my own MCP server with Promptise Foundry?","acceptedAnswer":{"@type":"Answer","text":"Yes. Decorators for tools, resources, and prompts. Schema auto-generated from type hints. Middleware (logging, timeout, rate limit, circuit breaker, audit). Authentication, guards, caching, health checks, metrics, exception handlers, webhooks, FastAPI-style Depends, session state, versioning, namespace transforms, OpenAPI ingestion, streaming, elicitation, sampling, and a TestClient for in-process testing."}},
    {"@type":"Question","name":"How do I deploy a Promptise Foundry MCP server?","acceptedAnswer":{"@type":"Answer","text":"Run promptise serve myapp:server --transport http --port 8080. Supports stdio, streamable HTTP, and SSE transports with configurable CORS. Auth gate at the transport level. CLI flags --dashboard (live terminal UI) and --reload (hot-reload). Kubernetes health probes (liveness, readiness, startup) built in."}},
    {"@type":"Question","name":"Does Promptise Foundry support distributed deployments?","acceptedAnswer":{"@type":"Answer","text":"Yes. RuntimeCoordinator for multi-node coordination, with StaticDiscovery and RegistryDiscovery for node discovery. Health checks over HTTP. No etcd or Consul dependency."}},
    {"@type":"Question","name":"Can I track costs with Promptise Foundry?","acceptedAnswer":{"@type":"Answer","text":"Promptise Foundry does not estimate or track LLM provider prices. The Budget governance subsystem tracks tool calls, LLM turns, and abstract cost units that you can map to your own pricing model. ToolCostAnnotations on tools let you assign per-call cost weights."}},
    {"@type":"Question","name":"How is testing done?","acceptedAnswer":{"@type":"Answer","text":"For MCP servers, TestClient runs the full pipeline in-process — validation, DI, guards, middleware, handler — with no network. For prompts, mock_llm, mock_context, assert_schema, assert_contains, assert_latency, and assert_guard_passed work with pytest."}},
    {"@type":"Question","name":"Where can I find examples?","acceptedAnswer":{"@type":"Answer","text":"Runnable examples at github.com/promptise-com/foundry/tree/main/examples — covering agents, MCP servers, prompt engineering, and runtime use cases. Every example uses real LLM calls."}}
  ]
}
</script>

# Frequently Asked Questions

Honest, complete answers about Promptise Foundry — the production Python framework for building AI agents, MCP servers, and autonomous runtimes.

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## Foundations

### What is Promptise Foundry?

Promptise Foundry is a production-grade Python framework for building AI agents, MCP (Model Context Protocol) servers, prompt engineering systems, and autonomous runtimes. It is **secure by default, model-agnostic, MCP-native**, and designed for teams shipping agentic AI to production. One coherent stack instead of gluing libraries together.

### Is Promptise Foundry production-ready?

**Yes.** No stub implementations, no half-finished features, no `NotImplementedError` in shipped code. Built-in: access control, capability-based policies, audit logging, encrypted transport, sandboxed code execution, observability with 8 transporters, crash-recovery via journals. Every backend listed as a parameter option works — if it's documented, it's implemented.

### Is Promptise Foundry open source?

**Apache 2.0** — source at [github.com/promptise-com/foundry](https://github.com/promptise-com/foundry).

### How do I install Promptise Foundry?

```bash
pip install promptise
```

Python 3.10 or newer required. Optional extras for memory backends, sandboxing, and observability — see [Installation Extras](getting-started/installation-extras.md).

## Comparisons

### How is Promptise Foundry different from LangChain?

LangChain is a general-purpose LLM toolkit with hundreds of integrations. Promptise Foundry is a focused production framework — one coherent stack covering agents, MCP servers, prompt engineering, and autonomous runtime, with built-in access control, audit trails, sandboxing, and governance. **Promptise has fewer abstractions, no silent fallbacks (errors raise instead of degrading), is async-first, and uses MCP-native tool discovery instead of manual wiring.**

### How does Promptise Foundry compare to LangGraph?

LangGraph is an orchestration layer focused on stateful graphs. Promptise Foundry includes a full **Agent Runtime** with crash recovery via journals, governance (budget, health, mission), and five trigger types (cron, event, message, webhook, file watch) — built for long-running autonomous agents that survive restarts, not just stateful conversations.

### How does Promptise Foundry compare to CrewAI?

CrewAI focuses on multi-agent role-playing workflows. Promptise Foundry covers single agents, multi-agent coordination via the **cross-agent delegation system** (`ask_peer` / `broadcast` over HTTP+JWT), plus the production infrastructure — MCP servers, governance, observability, sandboxing — that real deployments need.

## Models & Local-First

### Which LLM models does Promptise Foundry support?

**Any model.** Use a string like `"openai:gpt-5-mini"`, `"anthropic:claude-sonnet-4-5"`, or `"ollama:llama3"`, or pass any LangChain `BaseChatModel` directly. Switching providers requires changing one string in `build_agent()`.

### Does Promptise Foundry support self-hosted or local LLMs?

**Yes.** Use any Ollama model via `"ollama:model-name"`. Local embeddings via SentenceTransformers, local guardrail models (DeBERTa, GLiNER, Llama Guard), and local memory backends (ChromaDB) make **air-gapped deployments** fully supported.

## MCP — Model Context Protocol

### What is MCP and why does Promptise Foundry use it?

MCP ([Model Context Protocol](https://modelcontextprotocol.io)) is the open standard for connecting LLMs to tools, resources, and prompts. Promptise Foundry is **MCP-native** — agents auto-discover tools from any MCP server URL, schemas convert to typed tools automatically, and you can build your own MCP servers with the included SDK.

### Does Promptise Foundry have its own MCP client?

Yes — built from scratch, no third-party MCP client dependencies:

- `MCPClient` for single servers
- `MCPMultiClient` for connecting to N servers with unified tool list and auto-routing
- `MCPToolAdapter` for converting MCP tools to LangChain `BaseTool` objects with recursive schema handling

Supports HTTP, SSE, and stdio transports. Bearer token and API key authentication.

### Can I build my own MCP server with Promptise Foundry?

Yes — same relationship to MCP that FastAPI has to REST. Decorators for tools, resources, and prompts. Schema auto-generated from type hints. Middleware chain. Authentication, guards, caching, health checks, metrics, exception handlers, webhooks, dependency injection (FastAPI-style `Depends`), session state, versioning, namespace transforms, OpenAPI tool generation, streaming, elicitation, sampling, and a `TestClient` for in-process testing. See [Building Production MCP Servers](guides/production-mcp-servers.md).

### How do I deploy a Promptise Foundry MCP server?

```bash
promptise serve myapp:server --transport http --port 8080
```

Supports `stdio`, `streamable HTTP`, and `SSE` transports with configurable CORS. Auth gate at the transport level. CLI flags: `--dashboard` (live terminal UI), `--reload` (hot-reload during development). Kubernetes health probes (liveness, readiness, startup) built in.

## Agent Runtime

### What is the Agent Runtime?

The operating system for autonomous AI agents. Turns stateless LLM calls into **persistent, governed processes**. Each `AgentProcess` has lifecycle states (CREATED → STARTING → RUNNING → SUSPENDED → STOPPING → STOPPED/FAILED), trigger queues, heartbeat monitoring, concurrency control, conversation buffer, and journaled state for crash recovery.

### What's the difference between PromptiseAgent and the Agent Runtime?

| | PromptiseAgent | Agent Runtime |
|---|---|---|
| **What** | Single agent created via `build_agent()` | Wraps an agent in an `AgentProcess` |
| **Invocation** | `.run()` or `.chat()` | Triggered by events, cron, webhooks, files |
| **Lifecycle** | Stateless between calls | Long-running, persistent, recoverable |
| **Use case** | Request/response | Autonomous, ambient, scheduled |

### What trigger types can launch an autonomous agent?

Five built-in:

- **CronTrigger** — cron expressions
- **EventTrigger** — `EventBus` subscription
- **MessageTrigger** — topic-based pub/sub with wildcards
- **WebhookTrigger** — HTTP POST with HMAC verification
- **FileWatchTrigger** — directory monitoring with glob patterns

Multiple triggers compose on one process. Custom trigger types can be registered.

### How does Promptise Foundry handle crash recovery?

Through journals. `InMemoryJournal` and `FileJournal` record every state transition, trigger event, and invocation result. The `ReplayEngine` reconstructs state from a checkpoint plus replay log. **When a process crashes, it restarts from the last known good state** — no lost conversations, no lost trigger queue, no data loss.

### Does Promptise Foundry support distributed deployments?

Yes. `RuntimeCoordinator` for multi-node coordination, with `StaticDiscovery` and `RegistryDiscovery` for node discovery. Health checks over HTTP. **No etcd or Consul dependency** required.

### What governance does the Agent Runtime provide?

Four subsystems:

| Subsystem | What it does |
|---|---|
| **Budget** | Per-run and daily limits on tool calls, LLM turns, cost units, irreversible actions |
| **Health** | Anomaly detection — stuck loops, repeating patterns, empty responses, high error rates |
| **Mission** | LLM-as-judge evaluation against success criteria with confidence thresholds |
| **Secrets** | Per-process credentials with TTL expiry, rotation without restart, zero-fill revocation, never serialized to journal |

Escalation via webhook POST and `EventBus` emission. Enforcement: log, pause, stop, or escalate.

## Security & Multi-User

### Does Promptise Foundry handle multi-user access control?

Yes. The MCP server SDK ships:

- `JWTAuth` (HS256), `AsymmetricJWTAuth` (RS256/ES256), `APIKeyAuth`
- Capability-based access policies
- Per-tool permission guards: `HasRole`, `HasAllRoles`, `RequireAuth`, `RequireClientId`
- Per-user audit trails (HMAC-chained for tamper detection)
- `CallerContext` propagation across the entire stack via async contextvars

Built for multi-tenant production deployments. See [Building Multi-User Systems](guides/multi-user-systems.md).

### How does Promptise Foundry handle prompt injection and security?

`PromptiseSecurityScanner` with **six detection heads**:

| Head | Detects |
|---|---|
| DeBERTa | Prompt injection (ML model) |
| Regex (69 patterns) | PII |
| Regex (96 patterns) | Credentials |
| GLiNER | Named entities |
| Llama Guard / Azure AI | Content safety |
| Custom rules | Domain-specific |

All models run **locally**. Input blocking and output redaction built in. Memory retrieval and semantic cache responses are also rescanned.

### Can I sandbox code execution?

Yes. Docker-based sandbox with **seccomp syscall filtering**, capability dropping (~40 caps), read-only rootfs, resource limits (CPU, memory, time), and network isolation (none, restricted, full). Optional **gVisor** kernel for stronger isolation. Five agent tools auto-injected when sandbox is enabled — execute, read file, write file, list files, install package. Path traversal and shell injection prevention built in.

## Memory, Cache, and Persistence

### What memory backends does Promptise Foundry support?

Three providers ship in the framework:

- **`InMemoryProvider`** — testing
- **`ChromaProvider`** — local vector search, persistent
- **`Mem0Provider`** — enterprise-grade graph search

Configured on `build_agent()`. Before every invocation, the agent **auto-searches memory** and injects relevant results into the system prompt with prompt-injection mitigation built in.

### Does Promptise Foundry support conversation persistence?

Yes — `ConversationStore` protocol with four backends:

- `InMemoryConversationStore`
- `SQLiteConversationStore`
- `PostgresConversationStore`
- `RedisConversationStore`

The `chat()` method handles load → invoke → persist automatically. Session ownership is enforced.

### Does Promptise Foundry support semantic caching?

Yes. `SemanticCache` with in-memory or Redis backends. Per-user, per-session, or shared scope isolation. Serves cached responses for **semantically similar queries** — typically 30–50% cost reduction. Output guardrails re-scan cached responses. GDPR `purge_user()` supported. Encrypted-at-rest option available for Redis.

## Prompt Engineering

### Does Promptise Foundry have prompt engineering features?

Yes — prompts as software components:

- `@prompt` decorator
- **8 PromptBlock types** (`Identity`, `Rules`, `OutputFormat`, `ContextSlot`, `Section`, `Examples`, `Conditional`, `Composite`) with priority-based token budgeting
- `ConversationFlow` — system prompts that transform across phases
- **5 composable strategies**: `ChainOfThought`, `StructuredReasoning`, `SelfCritique`, `PlanAndExecute`, `Decompose`
- **4 built-in perspectives**: `Analyst`, `Critic`, `Advisor`, `Creative`
- Guards: `ContentFilterGuard`, `LengthGuard`, `SchemaStrictGuard`, custom validators
- **14 context providers** — Tool, Memory, Task, Blackboard, User, Environment, Conversation, Team, Error, Output, Static, Callable, Conditional, World
- `PromptInspector` for tracing assembly step-by-step

## Multi-Agent Systems

### Does Promptise Foundry support multi-agent systems?

Yes. **Cross-agent delegation**:

- `ask_peer()` — send a question to another agent over HTTP+JWT and await the answer
- `broadcast()` — send to multiple peers in parallel with timeout

Graceful degradation if a peer fails. SuperAgent YAML files declare cross-agent references with cycle detection.

## Configuration Files

### What is a SuperAgent file?

A `.superagent` YAML file that defines an entire agent declaratively — model, instructions, MCP servers, memory, sandbox, observability, cache, guardrails, cross-agents. Environment variable resolution via `${VAR}` and `${VAR:-default}`. Loadable via the CLI:

```bash
promptise agent file.superagent
```

### What is an .agent manifest?

A YAML manifest for the **Agent Runtime** — declares model, instructions, MCP servers, **triggers**, world state, memory, journal, open mode, budget, health, mission, and secrets. Validated, savable, deployable from the CLI. Distinct from `.superagent` (one-shot agents); `.agent` is for runtime processes with triggers and lifecycle.

### What is Open Mode?

Self-modifying agents with **14 meta-tools** — `modify_instructions`, `create_tool`, `connect_mcp_server`, `add_trigger`, `remove_trigger`, `spawn_process`, `list_processes`, `store_memory`, `search_memory`, `forget_memory`, `list_capabilities`, `get_secret`, `check_budget`, `check_mission`. Guardrails: max instruction length, max custom tools, MCP URL whitelist, mandatory sandbox for agent-written code. Hot-reload without losing conversation state. Rollback to original config.

## Operations

### How does observability work?

Four levels — `OFF`, `BASIC`, `STANDARD`, `FULL`. Every LLM turn, tool call, token count, latency, retry, cache hit/miss is recorded. **8 transporters**: HTML report, JSON file, structured log, console, Prometheus, OpenTelemetry, webhook, callback. Ring buffer with configurable max entries. Thread-safe.

### Is there a dashboard for monitoring agents?

Yes. Two dashboards — one for the MCP server (six tabs: server overview, tool stats, agents, request log, performance, raw logs) and one for the Agent Runtime (process state, invocation counts, trigger status, context inspection, memory usage, journal history). Both are live terminal UIs.

### Can I track costs with Promptise Foundry?

Promptise Foundry **does not estimate or track LLM provider prices** — those change weekly and would require constant maintenance. The **Budget governance system** tracks tool calls, LLM turns, and abstract cost units that you can map to your own pricing model. `ToolCostAnnotations` on tools let you assign per-call cost weights.

### What about job queues?

The MCP server SDK includes `MCPQueue` with priority scheduling, retry with exponential backoff, progress reporting, and cancellation. Auto-registered tools — `queue_submit`, `queue_status`, `queue_result`, `queue_cancel`, `queue_list`. Background tasks supported for fire-and-forget work after a handler returns.

### How is testing done?

For **MCP servers**: `TestClient` runs the full pipeline in-process — validation, dependency injection, guards, middleware, handler — with no network. For **prompts**: `mock_llm()`, `mock_context()`, `assert_schema()`, `assert_contains()`, `assert_latency()`, `assert_guard_passed()` helpers work with pytest.

### Does Promptise Foundry have a CLI?

Yes:

| Command | What it does |
|---|---|
| `promptise agent <file>` | Run a `.superagent` |
| `promptise validate` | Validate a config file |
| `promptise list-tools` | Discover tools from MCP servers |
| `promptise run` | Run a one-shot prompt |
| `promptise serve` | Serve an MCP server |

The runtime has its own CLI for managing processes, triggers, and manifests.

## Examples and Community

### Where can I find examples?

Runnable examples at [github.com/promptise-com/foundry/tree/main/examples](https://github.com/promptise-com/foundry/tree/main/examples) — covering agents, MCP servers (`examples/mcp/`), prompt engineering (`examples/prompts/`), and runtime use cases (`examples/runtime/`). **Every example uses real LLM calls** — no mocks, no stubs.

### How can I contribute?

Issues and pull requests welcome at [github.com/promptise-com/foundry](https://github.com/promptise-com/foundry). Conventional commits required (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`). Type hints on all public APIs (Python 3.10+ syntax), Google-style docstrings, and tests for all new functionality. See [Contributing](resources/contributing.md).

---

## Still have questions?

- [Open an issue on GitHub](https://github.com/promptise-com/foundry/issues)
- [Read the full documentation](index.md)
- [See examples](resources/examples.md)
- [Get started in 5 minutes](getting-started/quickstart.md)
