# Changelog

## Unreleased

### Changed

- **Events: webhook signatures cover the raw body and a timestamp (breaking for receivers)** -- `WebhookSink` signed `json.dumps(payload, sort_keys=True)` while sending a differently serialised body, so verifying the raw request body (what most web frameworks hand you, and what signature guides tell you to do) failed, and with no timestamp a captured request could be replayed forever. Every request now carries `X-Promptise-Signature: t=<unix seconds>,v1=<hex>`, the HMAC-SHA256 of `<t>.` followed by the exact body bytes (Stripe-style), plus `X-Promptise-Timestamp` and an `X-Promptise-Delivery` id that stays the same across retries. Retries are re-signed, so a late retry is not rejected as stale. The new `verify_event_signature(body, signature, secret, tolerance=300)` checks it in constant time, rejects signatures older than five minutes and accepts a list of secrets while rotating. A sink's generated secret is readable as `sink.secret`. **Receivers that compare the header to a bare hex HMAC of the sorted-key JSON must switch to `verify_event_signature()`.** `AgentEvent.compute_hmac()` is unchanged; it does not produce the webhook signature.

### Fixed

- **Events: an agent's events disagreed on `agent_id`** -- `invocation.*` events carried the model name even when `observer_agent_id` was set, and every other agent event (`tool.*`, `guardrail.*`, `approval.*`, `cache.*`) carried none. All events an agent emits now share one `agent_id`: `observer_agent_id`, else the identity's `agent_id` (or IdP subject); agents run by an `AgentProcess` use the process name; otherwise the model name, as before. The model is also in `metadata["model"]`.
- **Events: `tool.error` never fired for failing MCP tools** -- an MCP tool that fails returns a normal result with an error in it (a Promptise server's `ToolError` arrives as `{"error": {"code", "message", "retryable"}}`), so no exception was seen and no event was emitted; `tool.error` only fired for raising `extra_tools`, only with `observe=True`, and with `tool_name: "unknown"`. Error results (`isError`, the `ToolError` envelope, a `ToolMessage` with `status="error"`) and raised exceptions now emit `tool.error` with the real tool name, `error`, `error_type`, `duration_ms`, and `code`/`retryable` when the tool reports them. A domain answer such as `{"error": "No invoice X"}` (no `code`) is not treated as a failure.
- **Events: `tool.error` and `tool.slow` required `observe=True`** -- both were emitted by the observability callback handler, which only exists with observability on (undocumented). They now come from a handler attached to every invocation of an agent built with `events=`, for `ainvoke()`, `chat()` and the streaming methods, without duplicates when observability is on. The `tool.slow` threshold is configurable: `EventNotifier(slow_tool_threshold=5.0)` (seconds; `None` turns `tool.slow` off); the event now includes `threshold_ms`.
- **Events: `process.stopped` was never delivered, `process.failed` at startup only by accident** -- `AgentProcess.stop()` shut its agent down, which stopped the notifier, and only then queued `process.stopped`; a startup failure queued `process.failed` before anything had started the notifier. The process now starts the notifier before startup can fail and, on `stop()`, emits `process.stopped` and stops the notifier last. A notifier shared by an `AgentRuntime` is no longer stopped when one of its processes stops (which silenced the others); `runtime.stop_all()` stops it after the last process.
- **Events: one slow sink delayed every other sink, and shutdown dropped events silently** -- the notifier delivered each event to its sinks one after another, so a webhook retrying against a receiver that was down (about 7 s of backoff with the defaults) held up every later sink and event, and `stop()` cancelled whatever was left after 5 s without a word (with the receiver down, the fourth attempt and the `CallbackSink` never ran). Each sink now has its own queue and delivery task, so other sinks get every event at once and order is kept per sink. `stop()` waits up to `EventNotifier(shutdown_timeout=10.0)` (or `stop(timeout=...)`), then logs every dropped event with its sink and type and counts it in `notifier.dropped_count`; a full queue now drops for that sink only. New: `notifier.flush(timeout=)`, `notifier.is_running`, and sinks with a `close()` are closed on stop. `emit_sync()` is safe from other threads and starts the notifier inside a running loop; a notifier left running by an earlier `asyncio.run()` recovers in the next.
- **Events: `WebhookSink` could be pointed at internal hosts after the SSRF check (security)** -- the URL was checked once, when the sink was created, and httpx resolved the name again for every delivery, so a DNS record that changed afterwards (DNS rebinding, or a host that did not resolve at startup) delivered events, with their signature, to `127.0.0.1`, a private network or a cloud metadata service. The host is now resolved and checked before every delivery and the request goes to the checked address (original `Host` header and TLS server name kept), so httpx never resolves the name again; a delivery to an internal address is dropped with a warning and not retried. Redirects are never followed. The shared private-address check (also used by `WebhookApprovalHandler`, escalation webhooks and `OpenAPIProvider`) now rejects every non-public address, which adds shared/carrier-grade NAT space `100.64.0.0/10` (Alibaba Cloud's metadata service is `100.100.100.200`), `0.0.0.0` and multicast.
- **Events: `WebhookSink` refused every local or private URL with advice that did not exist** -- the error suggested a "base_url override" that `WebhookSink` does not have. Pass `allow_private_networks=True` to deliver to `localhost` or a private network (also `allow_private_networks: true` per sink in `.superagent` files); the error now says so. URLs that are not `http(s)` with a host are rejected either way.
- **Events: thin payloads and a redaction gap** -- `session_id` and `metadata` were always empty, `guardrail.blocked` did not say why, and `approval.requested` did not include the arguments under review. `session_id` is now the `chat()` session (else `caller.metadata["session_id"]`); `metadata` carries `model` and an `invocation_id` shared by every event of one run, plus `process_name`, `process_id` and `trigger_type` in the runtime; `guardrail.blocked` has `reason` and `findings` (detector, category, severity, description — never the matched text); `approval.requested` has `arguments` as the policy redacts them. `WebhookSink` redaction covered `data` only, so a `user_id` that was an email address was sent in clear; it now covers the whole payload (`redact_sensitive=False` turns it off). `emit_event()` fills `agent_id`, `session_id` and `metadata` from the running invocation and takes a `user_id`.
- **Docs: `docs/core/events.md` listed 20 of the 23 events** -- `budget.warning`, `budget.daily_reset` and `health.recovered` were emitted but undocumented. The page now lists all 23 with each event's `data` fields, documents signature verification, `allow_private_networks`, per-sink delivery and the shutdown timeout, and the `.superagent` `events:` section gains `shutdown_timeout` and `slow_tool_threshold`.

## 1.2.1 — 2026-10-10

### Fixed

- **MCP client: a rejected connection hung, then failed with a cancel-scope traceback** -- connecting to an HTTP server that answers the handshake with a 4xx (most often `401 Unauthorized` from a server built with `require_auth=True` when no or the wrong `api_key`/`bearer_token` is configured) left `MCPClient`, `MCPMultiClient` and `build_agent()` waiting, then crashed with `RuntimeError: Attempted to exit cancel scope in a different task`. Connecting now fails at once with a typed `MCPConnectionRejectedError` (a subclass of `MCPClientError`) carrying `status_code`, `reason`, `url` and `server_name`, and a message that names the server and says what to check: `Server 'orders' rejected the connection: 401 Unauthorized. Check the bearer_token/api_key configured for it.` The client now owns its transport in a task of its own, so other connection failures raise a readable `MCPClientError` too, never cancel the caller's task, and a client can be closed from any task. New example: `examples/mcp/client_auth_errors.py`.

- **MCP server SDK: tool parameters lost their descriptions** -- `@server.tool()` resolved type hints without `include_extras`, so `Annotated[str, Field(description=..., ge=..., pattern=...)]` reached Pydantic as a bare `str`: the description and every constraint disappeared from `inputSchema` and were not enforced at call time. `Annotated` metadata is now kept, and a `Field(...)` used as the parameter default (`order_id: str = Field(description=...)`) works the same way (before, such a parameter silently became optional, with the `FieldInfo` object as its value). A parameter without a `Field` description takes it from the docstring: Google-style `Args:` / `Arguments:` / `Parameters:` entries and Sphinx `:param name:` fields are read, wrapped continuation lines are joined, and a `Field` description wins when both are given. Prompt arguments use the same parser, so their multi-line `Args:` entries are no longer cut after the first line.

### Changed

- **MCP server SDK: the description is the docstring's summary paragraph** -- tools, resources and prompts without `description=` used only the first docstring line, so a summary wrapped over two lines was cut mid-sentence. The description is now every line up to the first blank line, section header (`Args:`, `Returns:`, ...) or Sphinx field, joined with single spaces. Later paragraphs are still not sent: they are usually notes for maintainers, and the `Args:` section goes into the parameter schema. Pass `description=` for anything longer. Documented under [Describing parameters](https://docs.promptise.com/mcp/server/building-servers/#describing-parameters).

## 1.2.0 — 2026-10-10

### Added

- **MCP: `promptise mcpcast` — turn an existing API into a curated, safe, agent-ready MCP server (`promptise.mcpcast`)** -- point it at an OpenAPI 3.x / Swagger 2 spec (file path, URL, or inline JSON/YAML) and get a **real, editable project** out -- `<name>-mcp/` with `mcpcast.plan.yaml` (the source of truth), the server as an installable `<name>_mcp/` package (`config.py`, `upstream.py` for the HTTP client and credentials, `approval.py` for the gate, `server.py` with `build_server()`, one `tools/<resource>.py` module per resource; depends only on `promptise` + `httpx`), a `server.py` launcher (runs without installing; exposes a module-level `server` so `promptise serve server:server` works), a generated `tests/` suite (every tool listed, routed to the right upstream operation on a fake upstream, and gated when it changes data -- `pytest` runs it), a `README.md` with ready-to-paste **Claude Desktop, Claude Code, and Cursor** snippets, and, written once and then yours, `pyproject.toml` (`pip install -e .` gives a `<name>-mcp` command and `python -m <name>_mcp`), `Dockerfile`, `.env.example` and `.gitignore` -- so any MCP-capable AI can use the API and the result can be reviewed, tested and shipped like any other service. The generated code is formatted to 100 columns with double quotes, block-style collections and wrapped descriptions, and lints clean under the project's own `[tool.ruff]`. A **deterministic risk classifier** labels every operation `read` / `write` / `destructive` / `financial` from its method and the verbs and money words in its id, path and summary (never the free-form description), then escalates one level for `admin`/`root`/`superuser` OAuth scopes, `admin`/`internal`/`sudo`/`impersonate` path segments, `deprecated: true`, and a `GET` that names a destructive verb (deprecated operations are always dropped). **Safety profiles** decide what is generated: `read-only` (default) emits reads only, `standard` adds writes, `full` adds destructive and financial operations -- and every non-read tool is emitted with `requires_approval=True`, enforced **server-side** by `ApprovalGateMiddleware` for any MCP client, deny-by-default on timeout. **LLM curation** (default on; `--no-curate` is the fully offline path) lets the model propose the tool budget, drops, collapses, renames, agent-audience descriptions, a parameter diet (hidden parameters sent as defaults), and examples, while the code enforces the post-conditions: at most `--max-tools` (25 by default), valid unique names, no unknown operations, no operation in more than one tool or both kept and dropped, hidden-required parameters need a default, deprecated operations must be dropped, and **risk is never downgraded** below the classifier (escalation only). Violations are fed back to the model for up to three attempts, then it fails loudly with `MCPcastError` -- no silent fallback (operations the model never mentions are dropped as "not selected by curation"; the profile is applied after curation). The **plan file is the source of truth**: edit `mcpcast.plan.yaml` and re-run `promptise mcpcast mcpcast.plan.yaml` to regenerate the package, launcher, tests and README next to it (the plan and the scaffold files are left untouched, comments included; tools modules the plan no longer produces are removed; a spec run into an existing project is refused without `--force`). **Auth modes**: `passthrough` (forwards the caller's `Authorization` header over HTTP/SSE; a missing one fails with `UPSTREAM_AUTH_MISSING`), `env-token` (one credential from `MCPCAST_UPSTREAM_TOKEN` on every call — the setup for a personal server launched over stdio by Claude Desktop, Claude Code or Cursor), `api-key` (`MCPCAST_CLIENT_KEYS` for callers, `require_tenant=True`, per-tenant upstream tokens from `MCPCAST_UPSTREAM_TOKENS`), and `none` (no credentials, loopback only). **Approval modes**: `elicitation` (asks the human behind the calling client; fail-closed without a live session) or `pending` (four-eyes review through generated, **tenant-scoped** `approvals_list` / `approvals_decide` tools for an `approver` role; only with `api-key` auth, since it needs identified callers). Upstream failures surface as structured `ToolError` codes (`UPSTREAM_AUTH_MISSING`, `UPSTREAM_UNREACHABLE`, `UPSTREAM_ERROR` with the status in `details`); path values must be a single real segment (no `..`), nested form and `deepObject` query values are bracket-encoded, and operation-level `servers` are honoured per route. Generated code is injection-safe by construction: spec text only ever lands in string literals or comments. CLI flags: `--review` (show the kept/dropped plan and confirm), `--serve` (run the result immediately; `-t stdio|http|sse`), and `--eval`, which produces an **Agent Readiness Score**: a real `build_agent()` agent drives the generated server **in-process** via `TestClient` on generated tasks (one per tool); routes of `read` tools may hit the live API (decided by risk class, not HTTP method) while everything else hits spec-derived mocks behind an auto-approver, so an evaluation never changes real data; `api-key` and `env-token` servers get evaluation-only credentials for the run. The report grades **A-F** (`0.6 * task success + 0.4 * correct-tool-selected-first`) and lists confused tool pairs, parameter errors, tools a task needed but the agent never called, tools no task covered, and concrete fixes ("`x` vs `y` are ambiguous -- merge them", "no description and no example") in `eval/report.md` + `eval/tasks.yaml`. Curation and eval both call the model through `build_agent()`. Curation also *repairs* rather than rejects one class of model output: an example whose values contradict the API's declared schema (`{sku, quantity}` against a schema of `{sku, qty}`) is replaced with a spec-derived one, because a hint to the agent is not worth failing a run over. Public API in `promptise.mcpcast`: `mcpcast()`, `curate()`, `build_plan()`, `write_project()`, `render_project()`, `load_generated_server()`, `evaluate()`, `MCPcastPlan` (`load`/`save`), `SafetyProfile`, `AuthMode`, `ApprovalMode`, `RiskClass`. **Docs**: `docs/mcp/server/mcpcast.md` (guide) and `docs/api/mcpcast.md` (API reference); **example**: `examples/mcp/mcpcast_petstore/`.
- **MCPcast: a guided setup in the terminal (`promptise mcpcast` with no arguments)** -- a full-screen, keyboard-driven wizard built with Textual (a new core dependency, pure Python) that walks through the seven decisions -- OpenAPI source (file, URL, or a *detected* API running on a local port), curation model or offline, safety profile with the exact tool counts each profile would generate from *your* spec, how the server will be used (mapped to the auth mode, with the environment variable you will set), name / folder / budget / evaluation, a review table with a per-tool detail panel and the *never trust it blindly* checklist, and the write step -- with the explanation each choice needs next to it. It runs the same `promptise.mcpcast` pipeline as the CLI, shows whether the model's key is picked up before anything is called, invalidates the plan when an earlier choice changes, and ends by printing the exact non-interactive command that reproduces the run. `promptise mcpcast SPEC --interactive` opens it pre-filled; without a terminal (CI, pipes) the command refuses with the non-interactive form to run instead. The review step also shows *computed* warnings — descriptions that name a tool the plan does not expose (the model writing "use delete_book" for an operation the profile excluded), and parameters hidden from the agent — available as `review_warnings(plan)`. Public API: `promptise.mcpcast.wizard` (`run_wizard`, `MCPcastWizard`, `detect_local_apis`, `review_warnings`, `equivalent_command`). Docs: [Guided Setup](https://docs.promptise.com/mcpcast/guided-setup/); lab: [`examples/mcp/mcpcast_wizard_lab/`](https://docs.promptise.com/guides/lab-mcpcast-wizard/) — a real Helpdesk API, the wizard, then a real agent, the gate and the Agent Readiness Score against the live app.
- **Core: `.env` is loaded everywhere** -- a `.env` file in the working directory (or a parent) is loaded before a provider's variables are checked, by plain scripts as well as the CLI, so `python my_agent.py` and `promptise run` see the same keys. An already-set environment variable always wins over the file; `PROMPTISE_NO_DOTENV=1` disables loading. Every missing-credential error now names the three ways to supply it (`.env`, export, `Model(..., api_key=)`), and the new [Configuration & Secrets](https://docs.promptise.com/getting-started/configuration/) page documents where keys live and the precedence between them.
- **Core: bring your own model — one install, any provider, credentials in code (`promptise.models`)** -- every place Promptise takes a model now accepts a `provider:model` string, a `Model(...)` object with the same words for every provider (`model`, `provider`, `deployment`, `api_key`, `endpoint`, `api_version`, `region`, `project`, settings and `extra`), or the same fields in a `.superagent` file. **No per-provider packages**: OpenAI, Azure OpenAI and Anthropic use their native integrations (core), and Groq, Gemini, Vertex AI, Bedrock (with a Bedrock API key), Mistral, DeepSeek, xAI, Together, Fireworks, Cohere, OpenRouter, Perplexity, NVIDIA, Hugging Face, Ollama and the Azure AI Foundry catalog are reached through their OpenAI-compatible endpoints with `langchain-openai` (core) — `Model(..., native=True)` opts into a provider's own package when installed. Azure AI Foundry is first-class: `provider="azure"` with `deployment=` for OpenAI deployments, `provider="foundry"` for catalog models. A missing credential raises `ModelSetupError` naming the variable, where its value lives in the provider's console, and the three ways to supply it; a word that does not apply to a provider (`api_key` on Ollama, `deployment` outside Azure) is refused with what to use instead. New CLI: `promptise models list|check [--ping]|env`. Docs: [Model Setup](https://docs.promptise.com/getting-started/model-setup/), [Configuration & Secrets](https://docs.promptise.com/getting-started/configuration/), [API reference](https://docs.promptise.com/api/models/).

### Changed

- **MCPcast generated projects pin their lint rules** -- the generated `pyproject.toml` now declares `[tool.ruff.lint] select` (and `target-version`) instead of relying on ruff's defaults, which changed in 0.16 and made previously clean projects report `TRY004`/`RUF100`. A generated project lints the same whichever ruff its owner has, and the rule set is theirs to widen.
- **Maintenance for this release** -- Python 3.13 joins the test matrix (and the package classifiers name 3.10-3.13); `ruff` is allowed up to 0.17; `actions/setup-python` and `actions/github-script` are current. The security job's dependency check used to be `pip freeze | safety check --stdin || true`, which could never fail: it now runs `pip-audit` over everything `promptise[all]` resolves to and fails the build, with `.github/pip-audit-ignore.txt` holding the advisories we knowingly tolerate (today: four ChromaDB server advisories with no fixed release — `ChromaProvider` embeds Chroma rather than exposing its HTTP server, which `docs/core/memory.md` now states).

- **Security gate: the CI environment's own build tools are kept current** -- the `pip-audit` job audits the whole environment, so the runner's preinstalled `setuptools` 79 (PYSEC-2026-3447) failed the release gate although no Promptise dependency was affected; the job now upgrades `setuptools` (>= 83) alongside `pip` before auditing.
- **Packaging: SPDX license metadata** -- `license = "Apache-2.0"` with `license-files`, replacing the table form and the license classifier that setuptools deprecates (builds would stop working in February 2027). The wheel now carries Metadata 2.4 with `License-Expression: Apache-2.0`; the build backend requires `setuptools>=77`.

- **Dependencies: `httpx` is now an explicit core dependency** -- generated `mcpcast` servers (and the OpenAPI provider) import it directly, so it is declared as `httpx>=0.27` instead of being relied on implicitly. It was already pulled in transitively by `mcp` and `langchain-core`, so a fresh install brings in nothing new.

### Fixed

- **MCPcast generated servers: hardened after an adversarial audit** -- an `env-token` server (one shared credential, no caller authentication) now binds loopback only, like `none`; `--public` on the generated command line and on `promptise mcpcast --serve` (or `MCPCAST_PUBLIC=1`) is the explicit opt-in for a gateway-fronted deployment, and the env-token Docker image serves stdio. `passthrough` requires a `Bearer ` value and is documented as a relay that belongs behind an authenticating gateway; `api-key` refuses to start with an empty `MCPCAST_CLIENT_KEYS`. Credentials are validated when read (`UPSTREAM_AUTH_INVALID` names the variable, never the value), no upstream error message echoes a header, and a credential is never sent to a plain-`http://` non-loopback host without `MCPCAST_ALLOW_INSECURE_HTTP=1` (`UPSTREAM_INSECURE`). Responses are streamed and capped (`MCPCAST_MAX_RESPONSE_BYTES`, `UPSTREAM_RESPONSE_TOO_LARGE`), a JSON content type with a non-JSON body is a structured `UPSTREAM_ERROR`, 408/425/429 are retryable with `retry_after`, and the error-body excerpt is configurable (`MCPCAST_ERROR_EXCERPT_CHARS`). A multi-route tool whose arguments match no route fails with `VALIDATION_ERROR` naming the alternatives instead of guessing. `MCPCAST_BASE_URL` now overrides route-level hosts too, and routes only carry their own host when the operation declares one -- a FastAPI spec fetched from a URL no longer freezes that origin into every route. Newlines and control characters are rejected in base URLs (they could reach the Dockerfile and `.env.example`), `spec_source` is recorded without userinfo, query or fragment, the pyproject description escapes backslashes and quotes, lone surrogates and non-finite numbers cannot crash generation, README cells are escaped, generated tests call tools by the Python identifiers they expose, reserved and keyword `operationId`s are suffixed `_op` instead of dropped, tool groups avoid `annotations` and the package's own module names, regeneration only removes tools modules it generated, a non-empty output directory without a plan is refused without `force`, external or dangling `$ref`s drop the operation with a reason, and fetched specs are capped (`MCPCAST_MAX_SPEC_BYTES`). The Dockerfile uses `python:3.12-slim-bookworm`, a non-root user and no baked-in configuration. Every generated project passes `ruff check`, `ruff format --check`, `mypy` and its own tests -- asserted in the test suite for all four auth modes.
- **MCPcast guided setup: hardened after an adversarial audit** -- local API detection never follows redirects, never treats a response body as a path or URL (bodies are parsed strictly as JSON/YAML, capped at 5 MiB, alias bombs refused), stops after a ten-second budget and names the ports it skipped, and cannot be crashed by a malformed local service; spec loads are generation-numbered so a superseded load never lands and reloading an edited spec invalidates the plan; writing again after changing an answer really writes again; a folder that already has files in it (an mcpcast project or anything else) is refused without the overwrite switch (`--force` on the CLI), and `~` is expanded; credentials in a spec URL are used for the fetch only and never recorded, shown, copied or printed; the base URL typed after loading is the one every route uses; `Ctrl+Q` after a write reports the project instead of "Nothing written."; `--model`, `--eval-tasks` and `--force` pre-fill the wizard while `--yes`, `--transport`, `--host` and `--port` are refused with it like the other flags; user text is never interpreted as markup; every write failure is shown.
- **MCP server SDK: elicitation and sampling never reached the client** -- `Elicitor.ask()` called `session.send_elicitation_request(...)`, and `Sampler.create_message()` passed `model=`/`system=`, neither of which exists on the pinned `mcp` SDK's `ServerSession` (`elicit(message, requestedSchema)` and `create_message(messages, max_tokens, system_prompt, model_preferences, …)` are the real signatures); both errors were swallowed, so `ElicitationApprover` denied every gated call for every client and sampling always returned `None`. Both now call the real methods (a declined/cancelled/timed-out elicitation still denies, fail-closed, and is logged at `WARNING`), `Elicitor.ask()` honours its `timeout`, the tests mock the real method names, and SDK contract tests assert the signatures still bind -- a renamed SDK method fails loudly next time. An in-process `mcp.ClientSession` round-trip test drives the approval gate through real elicitation.
- **Engine: a failed model call no longer produces an "answer"** -- when a run ended on a node whose model call failed (a rejected API key, a provider outage, an exhausted `RETRYABLE` node, a `CRITICAL` abort) `ainvoke()` returned `{"messages": ...}` with nothing but the user's question, so examples printed the question as the answer and the Agent Readiness evaluation graded a wrong key F with bogus advice. `PromptGraphEngine.ainvoke()` and `astream_events()` now raise `GraphExecutionError` (graph and node name, the `ExecutionReport`, the provider exception as `__cause__`); routing a failure to a handler node that succeeds, or a hook marking it recovered, still counts as recovery. The readiness report names crashed tasks first, excludes them from "never used", and `evaluate()` raises when no task could run.
- **`Model` no longer prints credentials** -- `repr()`/`str()` of `Model(..., api_key=...)` showed the key (and `extra`), and a failed `model_override` wrote it into `NodeResult.error`, logs and graph history. Both fields are excluded from the representation and node errors name the model spec, not the object.
- **Readiness: an approval-gated route could reach the real API during an evaluation** -- the evaluation transport ordered routes by the length of their regex and took the first match, so a templated read (`GET /users/{id}`, whose `{id}` becomes a 5-character pattern) was tried before an escalated route with a short literal segment (`GET /users/wipe`), and the gated call went live through the auto-approver with the real upstream credential. Routes are now ordered by specificity (literal segments before placeholders), and a request goes live only when *every* route it matches is a read; any other match is mocked.
- **Readiness: evaluating a project whose operator already configured `MCPCAST_CLIENT_KEYS` rejected every call** -- the evaluation key and tenant are now merged into the existing `MCPCAST_CLIENT_KEYS` / `MCPCAST_UPSTREAM_TOKENS` for the run and every variable is restored afterwards; auth rejections get their own line in the fixes.
- **Approval gate: one caller could fill the shared pending queue** -- `PendingApprover(max_pending_per_client=...)` bounds what a single client or tenant can park (identified by `caller_user_id`, then `client_id`/`tenant_id` metadata, then `agent_id`); generated MCPcast servers set it from `MCPCAST_MAX_PENDING_PER_CLIENT` (default 20).
- **`.env` loading: an exported-but-empty variable blocked the file's value** -- non-empty values from `.env` now fill variables that are set to an empty string, and `promptise models check` says so; Vertex AI ADC failures name the google-auth reason instead of swallowing it.
- **Docs: the model-setup page advertised providers that did not work as written** -- the provider table listed `google:gemini-2.5-pro`, which failed with LangChain's "Unable to infer model provider" (the LangChain prefix is `google_genai`; `google:` and `gemini:` are now registered aliases), and described Ollama as "no key needed" while `langchain-ollama` was neither installed nor declared (Ollama now goes through its OpenAI-compatible `/v1` endpoint and needs nothing). `anthropic:` only worked when `langchain-anthropic` happened to be installed -- it was never a declared dependency; it is now a core dependency. Every model string on the page now resolves through the registry that `promptise models check` reports on.
- **CLI: `promptise list-tools` works again, and the REPLs exit cleanly** -- `list-tools` crashed with `'PromptGraphEngine' object has no attribute 'tools'`; `PromptiseAgent` now exposes `tools` and `tool_names` (every tool the model was bound to: MCP-discovered, `extra_tools`, sandbox, cross-agent and meta tools), and the command uses them. `promptise run`, `promptise agent` and `list-tools` now call `agent.shutdown()` in the task that opened the MCP sessions, which removes the `RuntimeError: Attempted to exit cancel scope in a different task` traceback that stdio servers produced on `exit`.
- **MCPcast docs: the architecture, drawn** -- the end-to-end page opens with the runtime picture (assistant ⇄ generated server ⇄ your API, with the auth, gate, routing and upstream layers inside the server), shows the MCP contract and the life of one gated `tools/call` as sequence diagrams, the build pipeline as a flowchart and where the upstream credential comes from per auth mode; the reference draws the four deployment topologies and the guided-setup page the seven steps against their CLI flags.
- **MCP server SDK: caller identity is per request, and loopback servers validate `Host`/`Origin`** -- under Streamable HTTP the handler read the headers of the request that *initialized* the session, so `passthrough` forwarded that first bearer token on every later call (even header-less ones) and, with `api-key`, anyone holding a valid key plus a leaked `mcp-session-id` ran as the tenant that opened the session, approver role included. Headers, `X-Request-ID` and the client address now come from the current message (`bind_transport_request()`), and with `require_auth=True` a session is bound to the credential that opened it (another credential gets `404 Session not found`). The HTTP/SSE transports had no DNS-rebinding protection: a web page could drive a loopback-bound `env-token`/`none` server with the operator's credential. A loopback bind now validates `Host`/`Origin` against the loopback names (421/403 otherwise); `MCPServer.run()`/`run_async()` take `allowed_hosts=`/`allowed_origins=` (added to the loopback list; required to restrict a public bind) and `promptise serve` exposes them as `--allowed-host`/`--allowed-origin`. 36 tests over a real socket.
- **MCPcast generated servers: credential handling, hardened again** -- a `user:password@` in an operation-level `servers` URL bypassed the validator through `model_copy()` and landed in the plan, README, `config.py` (inside the `instructions` every client receives) and the tools modules; routes are validated now and such operations are dropped with a non-echoing reason. An upstream that quotes the `Authorization` header in an error body handed the server's own token to the MCP client through the error excerpt; the credential (raw, bare and URL-encoded) is replaced by `[redacted]` before the excerpt is cut, and non-JSON bodies are scrubbed too. `.env.example` shipped a working client key (`sk-acme`): it now ships an empty `MCPCAST_CLIENT_KEYS` with the shape as a comment and a mint command, and `build_server()` refuses the documentation keys and `<placeholders>`. `MCPCAST_BASE_URL` and `base_url` refuse a query string or fragment (route paths are appended and the value is copied into files). `securitySchemes` are honoured: an API keyed by `apiKey` in a custom header or query parameter gets `credential_location`/`credential_name` in the plan, the credential presented there at run time, and `--auth passthrough` refused with a pointer (it can only relay `Authorization`); operations that need a required header/cookie parameter the runtime cannot send are dropped with that reason. Generated launchers read `.env` next to `server.py` (as the file's header promised), GET bodies declared by the spec are sent, and array-of-object query parameters are indexed correctly (`tags[0][a]=1&tags[1][a]=2`).
- **MCPcast: plans and regeneration** -- a plan file could downgrade a `DELETE`/`POST` tool to `read` (emitted un-gated with `readOnlyHint`): `MCPcastPlan` now refuses a risk below the deterministic floor of the tool's routes. Regenerating no longer overwrites a hand-written `tools/<group>.py` or deletes a user's copy of a generated module (refused without `--force`; ownership is by the module's own generated header, found even in modules with hundreds of tools), and a plan whose `api.name` changed is refused while the old package and scaffold would still be packaged and shipped. Spec examples of the wrong JSON type are coerced or replaced (the tool no longer rejects its own example), properties or examples with a key literally named `$ref` are no longer dereferenced, tool descriptions are capped at render time (8 000 characters, with the plan keeping the full text and the wizard warning), generated tests compare the encoded request path (`format: email`/`date-time`/`uri` examples pass), parameters named `route`/`str`/`Any` and identifiers over 56 characters type-check and format, and README cells escape plan text. Client snippets in the generated README match the auth mode: env-token carries the token in `env`, api-key/passthrough get HTTP configurations (`HTTPServerSpec`, `claude mcp add --transport http … --header`) instead of stdio ones that cannot work.
- **MCPcast guided setup and CLI: hardened again** -- terminal escape sequences from spec or model text no longer reach the terminal (a hostile spec could blank or forge rows of the `--review` table, or write the clipboard through local API detection): control characters are removed at the parse boundary, refused in plan names, and scrubbed by every console renderer. Enter during a plan rebuild can no longer write the previous plan under new settings; review-pane links open only `http(s)` URLs; local API detection enforces the same depth/node caps as `load_spec`; Ctrl+Q no longer waits for a CPU-bound YAML parse (libyaml's parser with Python's composer, on a daemon thread); very large descriptions and dropped lists are clamped in the review pane; `--serve --public` is refused for auth modes that authenticate callers instead of crashing the generated entry point; an inline plan longer than the filesystem name limit regenerates; the printed next steps quote every path and carry the evaluation credential; `~` is expanded in the spec field; untitled inline specs get the same name in the wizard, the CLI and `mcpcast()`; a plain-`http://` upstream is called out with `MCPCAST_ALLOW_INSECURE_HTTP` (and the generated tests opt in for those hosts); Agent Readiness live reads follow `MCPCAST_BASE_URL` and a request the evaluation has no mock for is a scored failure, never a fake 200.
- **`.env` discovery, hardened** -- the walk skipped nothing: a world-writable or foreign-owned `/tmp/.env` could redirect an exported key, and an unreadable or non-UTF-8 `.env` up the tree crashed every `promptise` command at import. Files that are not regular, not owned by the current user or world-writable are skipped with a warning, the search stops at the project root (`pyproject.toml`/`.git`), an unreadable file is a `ModelSetupError` naming it, the CLI and `promptise models check` say which file filled a variable, `Model(api_key="")` counts as not given, and the Vertex AI token is refreshed before it expires instead of being pinned once.
- **Model resolution: one `httpx` client pair per model** -- langchain-openai caches one process-wide client, whose pooled connections belong to the first event loop; the second `asyncio.run()` in a process (curation then evaluation, the guided setup then an agent run) failed with `RuntimeError: Event loop is closed` on the latest SDKs. Every resolved OpenAI-route model now gets its own clients (a caller-supplied `http_client`/`http_async_client` wins).
- **Shell hooks and the template shell executor spawn without `fork()`** -- on macOS a forked child of a multithreaded process that has used system frameworks (any `httpx` client does) could die with SIGSEGV before `exec` (`exited -11`). Unless `cwd` is set, `ShellHook` and `SubprocessShellExecutor` now spawn through CPython's `posix_spawn` path (executable resolved on `PATH`, `close_fds=False` — descriptors are non-inheritable anyway), and `shell=False` command lines are tokenized like a terminal would.
- **MCPcast: a credential in the spec URL never reaches the project** -- `promptise mcpcast https://user:token@host/openapi.json?api_key=…` used to derive `base_url` from the raw URL, so the token landed in `mcpcast.plan.yaml`, `config.py` (and the `instructions` every MCP client receives), `.env.example` and the README, and was echoed on stderr. The credential is now used for the download only (`public_url()` strips userinfo, query and fragment before anything is derived, printed or recorded; fetch errors are scrubbed), and `ApiPlan.base_url` / route hosts refuse `user:password@` outright with a pointer to `--auth env-token` -- validation errors are rendered without pydantic's `input_value` (`render_validation_errors()`), so a refused value is never echoed either.
- **MCPcast: malformed specs fail with `Error:`, not a traceback; alias bombs are refused** -- `info` or `paths` that are not mappings, documents nested too deeply (`RecursionError`, depth cap 256) and YAML alias bombs (`MAX_DOCUMENT_NODES` = 2 000 000, env `MCPCAST_MAX_SPEC_NODES`; `expanded_nodes()` counts without expanding) raise `MCPcastError` naming the source; per-operation defects (`parameters` that are not a list of mappings, a `requestBody.content` that is not a mapping) drop that operation with the reason recorded, and a numeric `operationId` or `tags` value is coerced. The wizard's detection shares the same budget. Downloads have a wall-clock deadline (`MCPCAST_FETCH_SECONDS`, default 60) and a cancellation hook (`load_spec(cancelled=…)`): `Ctrl+Q` in the guided setup abandons an in-flight download or probe within one chunk instead of waiting for the transfer to finish (57 s → 2 s against a trickling server).
- **MCPcast generated servers: upstream deadline and pending capacity** -- `MCPCAST_TIMEOUT` is now a wall-clock deadline for the whole upstream call (connect, headers and body; new retryable `UPSTREAM_TIMEOUT` error, `UPSTREAM_UNREACHABLE` is connection/DNS/protocol only). `PendingApprover(max_pending_per_tenant=…)` caps what one tenant can park across all its API keys; generated `api-key` servers set `MCPCAST_MAX_PENDING` (100), `MCPCAST_MAX_PENDING_PER_TENANT` (40) and `MCPCAST_MAX_PENDING_PER_CLIENT` (20), and refuse to start when the caps do not nest. Generated code stays within 100 columns for names at their limits (14 resources, 59-character operation ids, 60-character parameters) -- asserted with `ruff check --extend-select E501`, `ruff format --check`, `mypy` and the generated suite in all four auth modes.
- **MCPcast: generated names are identical on every supported Python** -- keyword handling used `keyword.issoftkeyword()`, which answers for the running interpreter (`type` is soft from 3.12 only), so a plan built on 3.10 named a tool `type` where 3.12 named it `type_op` and a tag `type` became a different module. A fixed soft-keyword set (`SOFT_KEYWORDS`, `is_python_keyword()`) is used for tool, parameter and module names.
- **MCPcast CLI: a plan given as a URL or inline text crashed on regeneration** -- `promptise mcpcast https://…/mcpcast.plan.yaml` fetched the plan and then re-read it as a local path (`FileNotFoundError` traceback). The loaded document is validated directly (`MCPcastPlan.from_document()`), a plan without a directory of its own is written into `./<name>-mcp`, and the plan file is included in that project.
- **Guided setup: the printed "next time" command and the evaluation default** -- the command is now quoted for the shell it will be pasted into (double quotes on Windows, where `cmd.exe` takes single quotes literally; `quote_argument()`), and an explicit `--eval-tasks 20` with `-i` is honoured instead of being replaced by the wizard's own default: the CLI, the wizard and `evaluate()` share one `DEFAULT_EVAL_TASKS` (20).
- **Pydantic deprecation warnings on import** -- `FallbackChain` and the semantic-selection `request_more_tools` tool still used a class-based `Config`, so every import of `promptise` (including a generated MCPcast project's own test run) printed `PydanticDeprecatedSince20`; both use `ConfigDict` now, and a generated project's `pytest` run is warning-free.
- **MCPcast: nullable parameters are typed** -- a FastAPI `Optional[str]` (`anyOf: [{type: string}, {type: "null"}]`) reached agents as an untyped `author (object)`; the emitter now unwraps nullable `anyOf`/`oneOf` to the real type, and parameter descriptions are also read from the schema (where pydantic `Field(description=...)` puts them).

---

## 1.1.1 — 2026-07-17

### Fixed

- **Dependencies: cap `mcp<2.0`** -- mcp 2.0 changed the low-level `Server()` constructor signature and renamed `ResourceTemplate.uriTemplate` to `uri_template`, which breaks the MCP server SDK. With the previous uncapped `mcp>=1.9.0`, a fresh `pip install promptise` resolved to mcp 2.0 and the server SDK failed at runtime. Pinned to `mcp>=1.9.0,<2.0` (validated against mcp 1.29.0 with the full suite green alongside the latest cryptography, starlette, langchain-openai, and pydantic). Support for the mcp 2.0 API is tracked as a follow-up.

---

## 1.1.0 — 2026-07-08

### Identity

- **Agent Identity subsystem (`promptise.identity`)** — every agent gets a stable, traceable identity (*who is acting*), so its tool calls, audit entries, and outbound requests are all attributable. An identity can be **local** (just an `agent_id`) or **verifiable** — backed by a credential provider that mints a signed JWT the agent presents to the resources it calls (e.g. MCP servers). One user-facing class, `AgentIdentity`, with `from_*` factories and `AgentIdentity.auto()` platform auto-detection.
- **Workload-identity-federation providers** — Microsoft Entra ID (managed identity + projected token), AWS IAM (STS + EKS projected), Google Cloud (metadata), SPIFFE/SPIRE (file + SDK), and a generic OIDC issuer (file + callable). Per-resource credentials, token caching with refresh buffer, and declarative configuration via `.superagent` / `.agent` manifests.
- **Wired through the stack** — `build_agent(identity=...)`, cross-agent calls, MCP server auth + HMAC-chained audit, runtime processes, and observability all carry the agent identity. No third-party identity SDKs required for the core; cloud SDKs are optional per provider.
- **Production hardening** — providers now retry transient credential-acquisition failures (timeout / connection / 429 / 5xx, STS throttling) with jittered backoff and never retry a 4xx auth failure (fixes a real bug where one metadata blip silently degraded a verifiable agent to *unauthenticated*); server-side JWT verification tolerates a configurable clock-skew `leeway` (default 60s); thread-safe per-audience credential caching is now covered by a concurrency test.
- **Easier to adopt** — `AgentIdentity` and `IdentityError` are re-exported from top-level `promptise`; two **laptop-runnable** examples (no cloud, no API key) — a local-identity on-ramp and an end-to-end *verifiable identity → MCP server verifies + attributes the caller* demo; and opt-in, platform-gated live integration smoke tests (`tests/identity/integration/`) that mint a real token per provider so the live cloud round-trip is confirmable in your environment.
- **Comprehensive docs** — an enterprise "why this matters" (problem → risk → who/when), a "which provider" decision guide, an honest "verification status" (unit-mocked vs live-smoke-tested), and a corrected architecture diagram.

### Engine

- **Automatic context handling by default (`context_scope="auto"`)** — the default ReAct agent (and thus `build_agent()`) now manages context automatically: it behaves exactly like `"full"` while a tool loop is short (zero change to simple tasks), and switches to the bounded, deduplicated facts-ledger once the loop grows past `auto_ledger_after` (default 6) tool results. Deep tool loops stay token-efficient and context-bounded with no pattern to choose. It's an efficiency/context primitive — not an accuracy claim (for accurate aggregation over data, use `code-action`).
- **`code-action` reasoning pattern** — `build_agent(agent_pattern="code-action")`: for aggregation / data-traversal tasks the model writes **one Python program** over your tools (in a single LLM turn) instead of chaining dozens of conversational tool calls. The program runs in the hardened Docker sandbox (read-only rootfs, dropped capabilities, **no network**); its tool calls bridge back to the real host tools over a filesystem-RPC channel, where each tool keeps its protections — approval gates if configured, plus budget/health/audit hooks when the Agent Runtime has attached them — and the node enforces a hard per-run `max_tool_calls` cap regardless. Sandbox is auto-enabled (requires Docker). Bounded self-repair on a crash. Best with tools that return structured data (lists/dicts/numbers).
- **`verify` reasoning pattern** — `build_agent(agent_pattern="verify")` runs single-pass self-verifying reasoning (plan → solve → self-check → final answer) at one-turn latency. Matches a plain prompt on models that already reason internally; recovers errors a single pass would miss on weaker/mainstream models.
- **`context_scope="scoped"` on `PromptNode`** — context-lifecycle management: a scoped stage sees only its working set (system prompt + task + its own tool loop), not the whole transcript, bounding token growth across multi-stage reasoning graphs. Opt-in; `default="full"` preserves existing behavior.
- **`managed` tool pattern + `context_scope="ledger"`** — `build_agent(agent_pattern="managed")`: a context-managed tool loop for deep multi-tool tasks. Instead of an ever-growing transcript (where the model re-queries the same facts), the node keeps a compact deduplicated "facts gathered" ledger and serves identical `(tool, args)` calls from cache. Cuts redundant tool calls and bounds token growth at equal accuracy — an efficiency primitive, not an accuracy claim.
- **Fix: routing-hint noise** — linear nodes (single `default_next`, no real branch) no longer receive a spurious "choose the next step" instruction that weaker models could emit as their answer.

### MCP Server

- **First-class multi-tenancy** — `tenant_id` becomes a structural isolation invariant across the stack. Server side: `ClientContext.tenant_id` populated by `AuthMiddleware` from a configurable JWT claim (default `tenant_id`) or the API-key config; tenant-qualified rate-limit buckets (one tenant's traffic can never exhaust another's quota); `tenant_id` in every audit entry; `RequireTenant`/`HasTenant` guards; and `MCPServer(require_tenant=True)` to force authentication + tenant identity on every tool. Agent side: `CallerContext.tenant_id` + one derivation (`isolation_key`, `tenant::user`) feeds semantic-cache scope keys, memory scoping, and conversation ownership — two tenants with the same `user_id` can never see each other's data (`SessionAccessDenied` on cross-tenant session access, structurally impossible cache hits, provider-level memory isolation). `SemanticCache.purge_user(user_id, tenant_id=...)` purges exactly the tenant's scope.
- **Server-side approval gates (HITL where the tool lives)** — `@server.tool(requires_approval=True)` + `ApprovalGateMiddleware` enforce human approval for **any** MCP client, not just Promptise agents. Fail-closed semantics: denied by default on timeout, denied on handler crash, denied if a decision carries `modified_arguments` (the gate cannot rewrite bound arguments). An ungated `requires_approval` declaration **refuses to build** rather than silently not enforcing. Three approvers: `PendingApprover` (blocking pending store + auto-registered role-guarded `approvals_list`/`approvals_decide` admin tools — independent four-eyes review with **enforced separation of duties**: a caller cannot approve their own request), `ElicitationApprover` (MCP elicitation confirms with the human behind the calling client; denies fail-closed without a live session), and any existing `promptise.approval` handler (callback, HMAC-signed webhook) via the shared `ApprovalRequest`/`ApprovalDecision`/`ApprovalHandler` protocol. The gate evaluates the tool's guards **before** requesting approval (unauthorized callers never reach a reviewer), and `requires_approval` survives `include_router`/`mount` composition. Approval requests carry client id, tenant, and JWT subject; outcomes surface as structured `APPROVAL_DENIED` errors visible to the audit chain.

### Dependencies & CI

- **Eight Dependabot updates consolidated** into this single release and tested together, so the whole upgrade lands and verifies as one unit rather than eight separate merges — pydantic ≥2.13.3, cryptography ≥46, orjson ≥3.11.8, numpy ≥2.2.6, and the latest langchain / langchain-openai lines, plus GitHub Actions bumps (checkout v7, setup-python v6, codecov v7, codeql v4).
- **Verified against the latest majors a clean install resolves** — langchain **1.x** (langchain-core 1.4.8), numpy 2.5, mypy 2.x, pytest 9.x — the framework is runtime-compatible (full suite green) with **no dependency caps**. Compatibility fixes: dependency-resolution unblock (drop `pyspiffe` → `grpcio` from the `dev` extra), an import-cycle break (`cross_agent` ↔ `observability`), a numpy-2.x stub `mypy` override, and a widened `on_llm_new_token` override for langchain-core 1.x.
- **Cross-platform CI green** — two POSIX-only identity tests made platform-agnostic for Windows; the real-model ML-guardrail tests are skipped on **macOS hosted CI only** (gated on `darwin && $CI`), where the DeBERTa injection scores flap below the 0.85 threshold — they still run on Linux CI, Windows CI, and local macOS dev. The full 30-check matrix passes.

### Fixed

- **CLI: `promptise serve` now ships** — the documented deployment command (`promptise serve myapp:server --transport http --port 8080 --dashboard --reload`) existed in the docs but was never registered in the CLI. It now resolves the `module:attribute` target, validates it is an `MCPServer`, and serves over stdio / HTTP / SSE, with `--dashboard` and `--reload` support and clean stderr errors (stdout stays protocol-clean for stdio).
- **MCP server: declared per-tool rate limits are enforced** — `@server.tool(rate_limit="100/min")` was accepted and stored but never read. The spec is now parsed at registration (a malformed spec raises `ValueError` immediately) and enforced automatically via an auto-inserted middleware — per client when authentication populates `client_id`, a shared bucket otherwise — raising `RateLimitError` with a `retry_after_seconds` hint. `TestClient` enforces the same contract. New public API: `parse_rate_limit`, `DeclaredRateLimitMiddleware`.
- **Core: `CallerContext` survives cross-agent delegation** — a peer agent invoked via `ask_peer`/`broadcast` overwrote the ambient caller with `None`, dropping the original user's identity at every delegation hop. `PromptiseAgent.ainvoke` now inherits the ambient `CallerContext` when no explicit `caller` is passed, so a peer's cache scoping, memory search, guardrail tagging, and conversation ownership stay attributed to the original human principal (an explicit `caller=` still takes precedence).
- **Docs: removed the phantom `AgentAccessPolicy`** — design docs referenced a class that never existed in the codebase. Replaced with the real layered access-control model: transport-level auth providers, per-tool `Guard`s, per-request `CallerContext`, and runtime `OpenModeConfig` guardrails.

## 1.0.0 — 2026-03-26

### Promptise Foundry — Production Release

**The complete framework for building production agentic AI systems.** This release marks the transition from DeepMCPAgent to Promptise Foundry with a ground-up rebuild of every module.

---

### Core Agent

The `build_agent()` factory now accepts 18 opt-in parameters, each enabling a production capability with zero overhead when disabled:

- **Semantic Tool Optimization** — `optimize_tools="semantic"` sends only relevant tools per query using local embeddings (40-70% token savings). Configurable embedding model, supports local paths for air-gapped deployments.
- **Conversation Persistence** — `conversation_store=` with 4 backends (InMemory, SQLite, PostgreSQL, Redis). Session ownership enforcement prevents cross-user access. `chat()` method handles load/save/ownership automatically.
- **Semantic Cache** — `cache=SemanticCache()` serves similar queries from cache (30-50% cost savings). Per-user isolation by default, GDPR `purge_user()`, optional Redis backend with AES encryption at rest.
- **Security Guardrails** — `guardrails=PromptiseSecurityScanner.default()` with 6 detection heads: prompt injection (DeBERTa ML model), PII detection (69 regex patterns + Luhn validation), credential detection (96 patterns), Named Entity Recognition (GLiNER), content safety (Llama Guard / Azure AI), custom rules. All models run locally.
- **Human-in-the-Loop Approval** — `approval=ApprovalPolicy(tools=["send_*"])` pauses execution on sensitive tool calls, sends approval request to webhook/callback/queue, waits for human decision. HMAC-signed requests, replay protection, max pending limits.
- **Event Notifications** — `events=EventNotifier(sinks=[WebhookSink(...)])` emits structured events on 20 event types across 9 categories (invocation, tool, guardrail, budget, approval, mission, health, process, cache). 4 sink types: webhook (HMAC-signed, retries, SSRF-protected), callback, log, EventBus.
- **Streaming with Tool Visibility** — `astream_with_tools()` yields 5 event types (ToolStartEvent, ToolEndEvent, TokenEvent, DoneEvent, ErrorEvent) for real-time chat UIs. Auto-generated tool display names, argument redaction via guardrails.
- **Model Fallback Chain** — `model=FallbackChain(["openai:gpt-5-mini", "anthropic:claude-sonnet-4-20250514"])` with per-model circuit breakers, global timeout, configurable failure threshold and recovery window.
- **Adaptive Strategy** — `adaptive=True` captures tool failures, classifies them (infrastructure vs strategy vs unknown), synthesizes actionable strategies via LLM after threshold failures, injects learnings as context. Human feedback with LLM-as-judge verification.
- **Context Engine** — `context_engine=ContextEngine(budget=128000)` provides token-budgeted context assembly. Register layers by priority (identity, rules, memory, strategies, conversation, user message). Exact token counting via tiktoken. Trims lowest-priority content first. Snapshot/restore prevents permanent mutation.
- **Invocation Timeout** — `max_invocation_time=30` enforces maximum seconds per invocation with `asyncio.wait_for`, emits `invocation.timeout` event.
- **CallerContext** — Per-request identity (user_id, roles, scopes, metadata) propagated via contextvars to cache, guardrails, conversations, events, memory scoping.

### MCP Server SDK

Production framework for building MCP tool servers:

- **Authentication** — JWTAuth (HS256), AsymmetricJWTAuth (RS256/ES256), APIKeyAuth. Token caching with LRU eviction. TokenEndpointConfig for OAuth2 client_credentials.
- **Guards** — Per-tool authorization: HasRole, HasAllRoles, HasScope, HasAllScopes, RequireAuth, RequireClientId. Custom guards via protocol.
- **8 Middleware Types** — Logging, Timeout, RateLimit, CircuitBreaker, ConcurrencyLimiter, PerToolConcurrencyLimiter, StructuredLogging, Audit (HMAC-chained tamper-evident entries).
- **Caching** — InMemoryCache (LRU+TTL), RedisCache, @cached decorator, CacheMiddleware.
- **Job Queue** — MCPQueue with priority scheduling, retry, progress reporting, cancellation. 5 auto-registered tools.
- **Health Checks** — Liveness, readiness, startup probes. Kubernetes-native.
- **Metrics** — Prometheus /metrics endpoint, OpenTelemetry spans.
- **Dashboard** — Live terminal UI with 6 tabs.
- **OpenAPI Import** — OpenAPIProvider auto-generates MCP tools from OpenAPI specs.
- **Streaming** — StreamingResult for chunked responses. ProgressReporter for real-time updates.
- **Elicitation & Sampling** — Request user input or LLM completions mid-execution.
- **TestClient** — Full pipeline testing in-process (no network).
- **3 Transports** — stdio, streamable HTTP, SSE. CORS configurable.
- **MCP Client** — MCPClient (single), MCPMultiClient (N servers, auto-routing), MCPToolAdapter (MCP → LangChain BaseTool).

### Agent Runtime

Operating system for autonomous agents:

- **AgentProcess** — 6-state lifecycle (CREATED → STARTING → RUNNING → SUSPENDED → STOPPING → STOPPED/FAILED). Deterministic state machine with logged transitions.
- **5 Trigger Types** — Cron, Webhook (HMAC verified), File Watch (glob patterns), Event (EventBus), Message (topic pub/sub with wildcards). Custom trigger types via `register_trigger_type()`.
- **AgentContext** — Key-value state with write permissions, mutation history, memory access, environment variables, file mounts.
- **Journals** — InMemoryJournal, FileJournal. ReplayEngine for crash recovery from checkpoint + replay.
- **Governance: Budget** — Per-run and daily limits (tool calls, LLM turns, cost, irreversible actions). ToolCostAnnotation per tool. Warning at 80% threshold. Enforcement: pause, stop, or escalate.
- **Governance: Health** — Behavioral anomaly detection: stuck (identical calls N times), loop (repeating patterns), empty response, high error rate. Cooldown between alerts. Recovery detection.
- **Governance: Mission** — Objective + success criteria. LLM-as-judge evaluation every N invocations. Confidence thresholds. Timeout and invocation limits. Auto-complete on success.
- **Governance: Secrets** — Per-process credential context. ${ENV_VAR} resolution. TTL-based expiry. Zero-fill revocation. Access logging. Values never serialized.
- **Open Mode** — 14 meta-tools for self-modifying agents: modify_instructions, create_tool, connect_mcp_server, add/remove_trigger, spawn/list_processes, store/search/forget_memory, list_capabilities, get_secret, check_budget, check_mission. Guardrails: max instruction length, max custom tools, MCP URL whitelist, mandatory sandbox.
- **Live Agent Conversation** — MessageInbox with TTL, priority, rate limiting. `send_message()` and `ask()` methods. Messages injected into agent context. Answer extraction from agent responses.
- **Orchestration API** — 37 REST endpoints for managing deployed agents without code changes. Deploy, start, stop, restart, suspend, resume. Update instructions, budget, health, mission at runtime. Trigger management. Secret rotation. Journal reading. Context inspection. All endpoints authenticated. OrchestrationClient typed Python SDK with 37 matching methods.
- **Distributed** — RuntimeTransport (HTTP management API with auth), RuntimeCoordinator (cluster membership), StaticDiscovery / RegistryDiscovery. No etcd/Consul dependency.
- **.agent Manifests** — Declarative YAML for model, instructions, servers, triggers, context, journal, budget, health, mission, secrets, open mode.
- **Dashboard** — Live terminal UI with process states, invocation counts, trigger status.

### Prompt & Context Engineering

Prompts as software:

- **8 Block Types** — Identity (priority 10), Rules (9), OutputFormat (8), ContextSlot (configurable), Section (configurable), Examples (4), Conditional, Composite. Priority-based token budgeting drops lowest-priority blocks first.
- **ConversationFlow** — Phase-based system prompt evolution. Phases with active blocks and lifecycle hooks.
- **5 Strategies** — ChainOfThought, StructuredReasoning, SelfCritique, PlanAndExecute, Decompose. Composable: `chain_of_thought + self_critique`.
- **4 Perspectives** — Analyst, Critic, Advisor, Creative. CustomPerspective for domain-specific framing.
- **5 Guards** — ContentFilter, Length, SchemaStrict (JSON validation with retry), InputValidator, OutputValidator.
- **11 Context Providers** — Tool, Memory, Task, User, Environment, Conversation, Team, Error, Output, Static, Callable, Conditional, World.
- **PromptBuilder** — Fluent API for runtime construction.
- **Registry** — Semantic versioning. Rollback. Duplicate detection.
- **PromptInspector** — Traces assembly: blocks included/excluded, tokens per block, guard results.
- **Chaining** — chain(), parallel(), branch(), retry(), fallback().
- **YAML Loader** — .prompt files with templates, blocks, strategies, guards.
- **Testing** — mock_llm(), mock_context(), assert_schema(), assert_contains().

### Security (68 findings audited, 27+ fixed)

- SSRF protection on all URL inputs (_validate_url_not_private blocks private IPs, loopback, metadata endpoints)
- JWT algorithm validation (rejects alg confusion attacks)
- Timing-safe comparisons on all secret checks (hmac.compare_digest)
- Shell injection fix in sandbox read_file (shlex.quote)
- CAP_SYS_ADMIN removed from allow_sudo (container escape prevention)
- Null byte rejection in all path validation
- Safe template formatter (blocks attribute access, prevents SSTI)
- exec() code injection prevention in PromptBuilder (regex-validated names)
- Generic error messages to clients (no internal details leaked)
- Batch call auth bypass fixed (parent context propagated)
- Escalation webhook SSRF protection
- Distributed transport auth token + default localhost binding
- Audit chain race condition fixed (asyncio.Lock)
- Rate limiter thread safety + stale bucket eviction
- Memory content sanitization (12 injection patterns, case-insensitive)

### RAG Foundation

Pluggable base classes for retrieval-augmented generation:

- **4 Base Classes** — DocumentLoader, Chunker, Embedder, VectorStore. Subclass for your provider, plug into the pipeline.
- **RAGPipeline** — Orchestrator: `index()` ingests documents, `retrieve()` queries, `delete_document()` cleans up. Returns structured `IndexReport`.
- **Built-in Implementations** — RecursiveTextChunker (separator-aware splitting with overlap), InMemoryVectorStore (cosine similarity, metadata filtering). Zero external dependencies.
- **rag_to_tool()** — Wraps a pipeline as a LangChain tool the agent can call. Markdown, JSON, or text output. Configurable limit.
- **content_hash()** — Stable 12-character hash for dedup and incremental indexing.

### Runtime Lifecycle Hooks

Event-driven hook system for agent processes:

- **HookManager** — 14 lifecycle events: SESSION_START/END, USER_PROMPT_SUBMIT, PERMISSION_REQUEST/DENIED, SUBAGENT_START/STOP, PRE/POST_COMPACT, FILE_CHANGED, CONFIG_CHANGE, TASK_CREATED/COMPLETED.
- **`once: true`** — Hook auto-deregisters after first invocation. One-shot setup, first-run greetings, "next time X happens" patterns.
- **Priority ordering** — Higher priority hooks run first. Exception isolation: one broken hook never blocks the rest.
- **HookBlocked** — Raise to short-circuit and cancel the action that triggered the hook.
- **ShellHook** — Runtime hook backed by an external command. JSON event on stdin, JSON response on stdout. Supports blocking via `{"block": true}` and data mutation via `{"data": {...}}`. Configurable timeout, cwd, env.

### Shell Context Injection in Templates

Opt-in `!`cmd`` syntax in prompt templates:

- **SubprocessShellExecutor** — Runs shell commands with configurable timeout and allowlist. Only active when explicitly passed to the TemplateEngine.
- **Disabled by default** — Without a shell_executor, the syntax is left as literal text. Zero security exposure.
- **Allowlist support** — Restrict which commands are permitted for hardened environments.

### Multi-Granularity Rewind

Non-destructive rollback over the journal:

- **RewindEngine** — 5 modes: BOTH (full rollback), CONVERSATION_ONLY (keep tool results), CODE_ONLY (keep chat), SUMMARIZE (inject summary instead of dropping), CANCEL (dry-run preview).
- **Non-destructive** — Original journal entries stay on disk. The rewind itself is recorded as a `rewind` entry.
- **plan()** — Preview what would happen before committing.

### Path-Scoped Skill Activation

Skills that only activate when the codebase matches:

- **SkillRegistry** — Register skills with `paths: ["**/*.tsx", "src/**/*.ts"]` globs. `activate_for(cwd)` returns only skills whose globs match files under the working directory.
- **File-based loading** — Write a `.py` file with YAML frontmatter (`name`, `description`, `paths`) and a `create()` function. Load an entire directory with `load_directory()`.
- **Frontmatter parser** — Minimal YAML-ish parser, no external dependencies.

### AutoApprovalClassifier

Explicit decision hierarchy for approval requests:

- **5-layer hierarchy** — (1) Allow rules → (2) deny rules → (3) read-only auto-allow → (4) optional LLM classifier → (5) fallback to human handler.
- **ApprovalRule** — Match by tool glob, user ID, argument substring, or async predicate.
- **Drop-in replacement** — Implements the ApprovalHandler protocol. One-line swap in any existing ApprovalPolicy.
- **ClassifierStats** — Per-layer hit counts for audit and tuning.

### SuperAgent YAML

All features configurable via .superagent files:

- memory, observability, optimize_tools, cache, approval, events, adaptive, guardrails, max_invocation_time

### Simplified Install — Two Choices

Old multi-extra matrix (`[ml]`, `[infra]`, `[observability]`, `[mcp]`, `[docs]`, `[deep]`) has been collapsed into two clear extras:

- `pip install promptise` — base install is now complete (agent, MCP server + client, runtime, prompts, CLI, OpenAI, `aiohttp`, `watchdog`, cryptography)
- `pip install "promptise[all]"` — everything production-ready: ChromaDB, Mem0, sentence-transformers, transformers, numpy, Redis, Docker, OpenTelemetry, Prometheus
- `pip install "promptise[dev]"` — contributors only: everything in `[all]` plus pytest, mypy, ruff, mkdocs tooling

The `[mcp]` extra is removed — MCP is core. The `[deep]` extra is removed — install `deepagents` separately if you need it.

### Breaking Changes from DeepMCPAgent

- **Package**: `deepmcpagent` → `promptise`
- **Imports**: `from deepmcpagent` → `from promptise`
- **CLI**: `deepmcpagent` → `promptise`
- **Factory**: `build_deep_agent()` → `build_agent()`
- **Servers**: Dict format → `HTTPServerSpec` / `StdioServerSpec` objects

### Statistics

- 161 source files
- 120+ test files, 3400+ tests
- 130+ documentation pages, 0 build warnings
- Apache 2.0 license

---

## 0.5.0 — 2025-10-18

### Added

- Cross-Agent Communication (in-process) with `cross_agent.py`.
- `CrossAgent`, `make_cross_agent_tools`, `ask_agent_<name>`, and `broadcast_to_agents`.

---

## 0.4.1 — 2025-10-17

### Fixed

- Fixed `TypeError` when falling back to `create_react_agent()` with `langgraph>=0.6`.

---

## 0.4.0

### Added

- CLI with pretty console output, `--trace/--no-trace`, and `--raw` modes.
- HTTP server specs with block string syntax.
- Tool tracing hooks integrated into agent layer.

---

## 0.3.0

### Added

- Improved JSON Schema → Pydantic mapping.
- PyPI Trusted Publishing workflow.

---

## 0.1.0

- Initial MCP client edition.
