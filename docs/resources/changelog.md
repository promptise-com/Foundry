# Changelog

All notable changes to Promptise Foundry are documented here.

---

## v1.1.1 — 2026-07-17

### Fixed

- **Dependencies: cap `mcp<2.0`** -- mcp 2.0 changed the low-level `Server()` constructor signature and renamed `ResourceTemplate.uriTemplate` to `uri_template`, which breaks the MCP server SDK. With the previous uncapped `mcp>=1.9.0`, a fresh `pip install promptise` resolved to mcp 2.0 and the server SDK failed at runtime. Pinned to `mcp>=1.9.0,<2.0` (validated against mcp 1.29.0 with the full suite green alongside the latest cryptography, starlette, langchain-openai, and pydantic). Support for the mcp 2.0 API is tracked as a follow-up.

---

## v1.1.0 — 2026-07-08

### Added

- **Identity: Agent Identity subsystem (`promptise.identity`)** -- every agent gets a stable, traceable identity (*who is acting*), so its tool calls, audit entries, and outbound requests are attributable. An identity can be **local** (just an `agent_id`) or **verifiable** -- backed by a credential provider that mints a signed JWT the agent presents to the resources it calls. One class, `AgentIdentity`, with `from_*` factories and `AgentIdentity.auto()` platform auto-detection. Providers: **Microsoft Entra ID**, **AWS IAM** (STS + EKS), **Google Cloud**, **SPIFFE/SPIRE**, and a **generic OIDC** issuer -- with per-resource credentials, token caching, and declarative manifest config. Wired through `build_agent(identity=...)`, cross-agent calls, MCP server auth + HMAC-chained audit, runtime, and observability. Cloud SDKs are optional per provider. **Hardened**: providers retry transient credential-acquisition failures (timeout/connection/429/5xx) but never a 4xx auth failure; server-side JWT verification tolerates a configurable clock-skew `leeway`; thread-safe caching is concurrency-tested. **Easier**: `AgentIdentity`/`IdentityError` re-exported from top-level `promptise`; laptop-runnable `local` and `verifiable_mcp` examples (no cloud/key); opt-in platform-gated live integration smoke tests. **Docs**: enterprise "why this matters", a provider decision guide, and an honest "verification status".
- **Engine: automatic context handling by default (`context_scope="auto"`)** -- the default ReAct agent (and therefore `build_agent()`) now bounds context automatically. It behaves exactly like `"full"` while a tool loop is short (zero change to simple tasks) and switches to the deduplicated facts-ledger once the loop passes `auto_ledger_after` (default 6) tool results, so deep tool loops stay token-efficient with no pattern to choose. An efficiency/context primitive, not an accuracy claim (use `code-action` for accurate aggregation over data).
- **Engine: `code-action` reasoning pattern** -- `build_agent(agent_pattern="code-action")`. For aggregation / data-traversal tasks the model writes **one Python program** over your tools in a single LLM turn, instead of chaining dozens of conversational tool calls. The program runs in the hardened Docker sandbox (read-only rootfs, dropped capabilities, **no network**); its tool calls bridge back to the real host tools over a filesystem-RPC channel, where each tool keeps its protections -- approval gates if configured, plus budget/health/audit hooks when the Agent Runtime has attached them -- and the node enforces a hard per-run `max_tool_calls` cap regardless. The sandbox is auto-enabled (requires Docker), with bounded self-repair on a crash. Best with tools that return **structured data** (lists/dicts/numbers).
- **Engine: `verify` reasoning pattern** -- `build_agent(agent_pattern="verify")` runs a single-pass self-verifying reasoning node (plan -> solve -> self-check -> final answer) at **one-turn** latency. It matches a plain prompt on models that already reason internally, and recovers errors a single pass would miss on weaker/mainstream models -- a good default when you want a self-checking answer without a multi-stage pipeline.
- **Engine: `PromptNode(context_scope="scoped")`** -- context-lifecycle management for multi-stage reasoning graphs. A scoped node sees only its own working set -- its system prompt (carrying any inherited/distilled state), the task, and its in-progress tool loop -- not the verbose messages produced by other stages, so token usage does not grow super-linearly across stages. Opt-in; `context_scope="full"` (the default) preserves existing behavior.
- **Engine: `managed` tool pattern + `PromptNode(context_scope="ledger")`** -- `build_agent(agent_pattern="managed")` runs a context-managed tool loop for **deep multi-tool tasks**. Instead of an ever-growing transcript (where the model loses track and re-queries the same facts), the node keeps a compact, **deduplicated "facts gathered" ledger** and serves identical `(tool, args)` calls from cache. It **cuts redundant tool calls** and bounds token growth at equal accuracy -- an efficiency/cost win for long chains (an efficiency primitive, not an accuracy claim).

- **MCP server: first-class multi-tenancy** -- `tenant_id` as a structural isolation invariant. Server side: `ClientContext.tenant_id` from a configurable JWT claim or API-key config, tenant-qualified rate-limit buckets, tenant in audit entries, `RequireTenant`/`HasTenant` guards, `MCPServer(require_tenant=True)`. Agent side: `CallerContext.tenant_id` + `isolation_key` (`tenant::user`) feeding cache scope keys, memory scoping, and conversation ownership -- two tenants with the same `user_id` can never see each other's data. `SemanticCache.purge_user(user_id, tenant_id=...)` purges exactly one tenant's scope.
- **MCP server: server-side approval gates (HITL)** -- `@server.tool(requires_approval=True)` + `ApprovalGateMiddleware` enforce human approval for any MCP client. Fail-closed: denied on timeout by default, denied on `modified_arguments`, and an ungated declaration refuses to build. Guards run before a reviewer is asked, self-approval is rejected (four-eyes), and the flag survives router/mount composition. Approvers: `PendingApprover` (pending store + role-guarded `approvals_list`/`approvals_decide` admin tools), `ElicitationApprover` (asks the human behind the calling client; fail-closed without a live session), or any existing `promptise.approval` handler (shared `ApprovalRequest`/`ApprovalDecision` protocol).

### Changed

- **Dependencies: eight Dependabot updates consolidated** into this release and tested together, so the whole upgrade lands and verifies as one unit -- pydantic >=2.13.3, cryptography >=46, orjson >=3.11.8, numpy >=2.2.6, and the latest langchain / langchain-openai lines, plus GitHub Actions bumps (checkout v7, setup-python v6, codecov v7, codeql v4).
- **CI: verified against the latest majors a clean install resolves** -- langchain **1.x** (langchain-core 1.4.8), numpy 2.5, mypy 2.x, pytest 9.x -- the framework is runtime-compatible (full suite green) with no dependency caps. Compatibility fixes: dependency-resolution unblock (drop `pyspiffe` -> `grpcio` from the `dev` extra), an import-cycle break (`cross_agent` <-> `observability`), a numpy-2.x stub `mypy` override, and a widened `on_llm_new_token` override for langchain-core 1.x.
- **CI: cross-platform green** -- two POSIX-only identity tests made platform-agnostic for Windows; the real-model ML-guardrail tests are skipped on macOS hosted CI only (gated on `darwin && $CI`), where the DeBERTa injection scores flap below the 0.85 threshold, and still run on Linux CI, Windows CI, and local macOS dev. The full 30-check matrix passes.

### Fixed

- **Engine: routing-hint noise** -- a linear node (a single `default_next` with no real branch) no longer receives a "choose the next step" instruction. The hint could be echoed as the answer by weaker models; it now appears only at genuine branch points (two or more distinct targets).
- **CLI: `promptise serve` now ships** -- the documented deployment command was never registered in the CLI. It now resolves the `module:attribute` target, validates it is an `MCPServer`, and serves over stdio / HTTP / SSE with `--dashboard` and `--reload` support.
- **MCP server: declared per-tool rate limits are enforced** -- `@server.tool(rate_limit="100/min")` was stored but never read. The spec is parsed at registration (malformed specs raise immediately) and enforced automatically -- per client when authenticated, a shared bucket otherwise -- raising `RateLimitError` with a `retry_after_seconds` hint. `TestClient` enforces the same contract. New public API: `parse_rate_limit`, `DeclaredRateLimitMiddleware`.
- **Core: `CallerContext` survives cross-agent delegation** -- a peer invoked via `ask_peer`/`broadcast` overwrote the ambient caller with `None`, dropping the original user at every hop. `PromptiseAgent.ainvoke` now inherits the ambient `CallerContext` when no explicit `caller` is passed (an explicit `caller=` still wins), keeping peer cache/memory/guardrails/conversations attributed to the original human principal.
- **Docs: removed the phantom `AgentAccessPolicy`** -- design docs referenced a class that never existed; replaced with the real layered access-control model (transport auth providers, per-tool guards, `CallerContext`, runtime open-mode guardrails).
- **Config: dict-form server specs carry every field** -- a server passed as a raw dict (in runtime config and `.agent` manifests) no longer silently drops supported fields: HTTP keeps `auth`, stdio keeps `cwd` and `keep_alive`, so a dict config behaves identically to the equivalent `HTTPServerSpec`/`StdioServerSpec`. The runtime normalization is factored into a single `_resolve_server_specs` helper.
- **MCP client: stdio `cwd` is honored end-to-end** -- `StdioServerSpec(cwd=...)` previously never reached the launched MCP subprocess (the native `MCPClient` had no `cwd` parameter), so the server always started in the parent's working directory. `cwd` is now plumbed through `MCPClient` into the SDK's subprocess launch, on both the agent and runtime paths.
- **Runtime: agent state history is an immutable snapshot** -- `AgentContext.put()` (and initial state) recorded the live mutable value in its timestamped mutation history by reference, so a later in-place mutation retroactively rewrote past audit records. Each history entry is now an independent snapshot while the live blackboard value stays mutable.
- **Observability: cross-agent delegation stamping is snapshotted** -- the `delegated_by` identity written onto a peer's timeline is now copied per entry, so a later mutation of one entry's metadata can't retroactively alter its siblings.

---

## v0.6.1

### Fixed

- **Runtime: cross-task MCP client issue** -- `AgentProcess._build_agent()` now pre-discovers MCP tools before agent construction, resolving `CancelledError` when the MCP SDK's cancel scopes crossed asyncio task boundaries. Tools are pre-discovered and passed as `extra_tools` instead of relying on the session-bound MCP client.
- **Runtime: manifest server conversion** -- `manifest_to_process_config()` now converts plain server dicts from `.agent` manifests into `HTTPServerSpec`/`StdioServerSpec` objects, fixing `AttributeError: 'dict' object has no attribute 'transport'` when starting processes from manifest files.

### Added

- **Data Pipeline Monitoring Lab** (`examples/runtime/data_pipeline_lab/`) -- 8 production examples demonstrating process lifecycle, triggers, journals, AgentContext, ConversationBuffer, multi-process AgentRuntime, manifest loading, and open mode with meta-tools. All examples use real LLM calls.
- **AI Content Creation Studio** (`examples/prompts/content_studio_lab/`) -- 9 runnable demos covering prompt blocks, flows, guards, strategies, context providers, chain operators, registry, inspector, and templates. All demos use real LLM calls.

---

## v0.6.0

**Renamed package from `deepmcpagent` to `promptise`.**

### Added

- **Prompt Engineering framework** -- 2-layer system for composing production prompts
    - Layer 1: `PromptBlocks` -- composable blocks with priority-based assembly
    - Layer 2: `ConversationFlow` -- turn-aware prompt evolution across conversation phases
    - `PromptInspector` for full prompt tracing and debugging
- **Agent Runtime** -- lifecycle container for autonomous agent processes
    - `AgentProcess` with state machine (CREATED, RUNNING, SUSPENDED, STOPPED, FAILED)
    - Triggers: `CronTrigger`, `WebhookTrigger`, `FileWatchTrigger`, `EventTrigger`, `MessageTrigger`
    - `AgentContext` for unified state, environment, and file mounts
    - `Journal` and `ReplayEngine` for crash recovery
    - `.agent` YAML manifest format for declarative process definition
    - `AgentRuntime` multi-process manager with distributed coordination
    - CLI: `promptise runtime validate|start|logs|init`
- **MCP Server framework** -- build production MCP servers
    - `MCPServer` with `@server.tool()` decorator and Pydantic model validation
    - `MCPRouter` for grouping tools under shared prefixes and policies
    - `AuthMiddleware` with `JWTAuth` and role-based guards
    - `LoggingMiddleware`, `TimeoutMiddleware`, `ConcurrencyLimiter`
    - `BackgroundTasks` via dependency injection
    - `@cached` decorator with `InMemoryCache` backend
    - `ServerSettings` for typed, environment-backed configuration
    - Exception handlers for structured MCP error responses
    - Lifecycle hooks (`on_startup`, `on_shutdown`)
    - Live monitoring dashboard
    - `require_auth` option for fully authenticated servers
- **MCP Client library**
    - `MCPClient` for single-server connections with `fetch_token()` helper
    - `MCPMultiClient` for multi-server routing with automatic tool discovery
    - `MCPToolAdapter` for converting MCP tools to LangChain `BaseTool` instances
    - Tracing callbacks (`on_before`, `on_after`, `on_error`)
### Changed

- Package name: `deepmcpagent` renamed to `promptise`
- CLI command: `deepmcpagent` renamed to `promptise`
- All import paths: `from deepmcpagent` changed to `from promptise`

---

## v0.5.0

Initial release as `deepmcpagent`.

### Added

- Core agent building with `build_agent()`
- MCP integration with `HTTPServerSpec` and `StdioServerSpec`
- Cross-agent communication and delegation via `CrossAgent`
- SuperAgent `.superagent` YAML configuration format
- CLI: `deepmcpagent agent|validate|init|list-tools`
- Sandbox execution with `SandboxConfig`
