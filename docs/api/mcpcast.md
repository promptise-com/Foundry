# MCPcast API Reference

`promptise mcpcast` turns an existing API (OpenAPI 3.x or Swagger 2) into a curated, safe, agent-ready MCP server, emitted as an editable project: `mcpcast.plan.yaml` (the source of truth), an installable `<name>_mcp/` package with its own tests (regenerated from the plan), a `server.py` launcher, a README and the packaging scaffold. The pipeline is parse → classify (risk) → curate (LLM, optional) → review (human) → emit (code) → eval (Agent Readiness Score). Every public symbol of `promptise.mcpcast` is documented here; for the walkthrough, safety profiles, auth modes and the generated server's runtime contract see the [MCPcast guide](../mcp/server/mcpcast.md).

```python
from promptise.mcpcast import SafetyProfile, mcpcast, write_project

plan = mcpcast("openapi.yaml", profile=SafetyProfile.STANDARD)
write_project(plan, "myapi-mcp")
```

## Entry points

`mcpcast()` is the deterministic path (no model, no network beyond fetching a spec URL). `curate()` designs the surface with a model, with every post-condition enforced in code. `write_project()` emits the project. `evaluate()` scores how well a real agent can drive the result.

### mcpcast

::: promptise.mcpcast.mcpcast
    options:
      show_source: false
      heading_level: 4

### curate

::: promptise.mcpcast.curate.curate
    options:
      show_source: false
      heading_level: 4

### write_project

::: promptise.mcpcast.write_project
    options:
      show_source: false
      heading_level: 4

### evaluate

::: promptise.mcpcast.evaluate
    options:
      show_source: false
      heading_level: 4

---

## Plan schema

The validated, versioned plan that round-trips to `mcpcast.plan.yaml`. `MCPcastPlan` enforces the invariants (unique tool names, each operation in at most one tool, never both kept and dropped, profile allows every tool's risk, gated risks carry `requires_approval=True`).

### MCPcastPlan

::: promptise.mcpcast.MCPcastPlan
    options:
      show_source: false
      heading_level: 4

### ToolPlan

::: promptise.mcpcast.ToolPlan
    options:
      show_source: false
      heading_level: 4

### RoutePlan

::: promptise.mcpcast.RoutePlan
    options:
      show_source: false
      heading_level: 4

### RouteParam

::: promptise.mcpcast.RouteParam
    options:
      show_source: false
      heading_level: 4

### ParamPlan

::: promptise.mcpcast.ParamPlan
    options:
      show_source: false
      heading_level: 4

### DroppedOp

::: promptise.mcpcast.DroppedOp
    options:
      show_source: false
      heading_level: 4

### ApiPlan

::: promptise.mcpcast.ApiPlan
    options:
      show_source: false
      heading_level: 4

### RiskClass

::: promptise.mcpcast.RiskClass
    options:
      show_source: false
      heading_level: 4

### SafetyProfile

::: promptise.mcpcast.SafetyProfile
    options:
      show_source: false
      heading_level: 4

### AuthMode

::: promptise.mcpcast.AuthMode
    options:
      show_source: false
      heading_level: 4

### ApprovalMode

::: promptise.mcpcast.ApprovalMode
    options:
      show_source: false
      heading_level: 4

### MCPcastError

::: promptise.mcpcast.MCPcastError
    options:
      show_source: false
      heading_level: 4

### valid_tool_name

::: promptise.mcpcast.valid_tool_name
    options:
      show_source: false
      heading_level: 4

### RESERVED_TOOL_NAMES

Tool names that would break or hijack the generated server (`approvals_list`, `approvals_decide`, `server`, `upstream`, `str`, …); `valid_tool_name()` rejects them alongside Python keywords.

### RESERVED_CREDENTIAL_HEADERS

Headers the HTTP client owns (`cookie`, `host`, `content-length`, `transfer-encoding`); `ApiPlan.credential_name` may not name one. `ApiPlan.credential_location` / `credential_name` record where the generated server presents the upstream credential, set by the planner from the spec's security scheme (default: the `Authorization` header).

### scrub_text

::: promptise.mcpcast.schema.scrub_text
    options:
      show_source: false
      heading_level: 4

### is_plan_document

::: promptise.mcpcast.is_plan_document
    options:
      show_source: false
      heading_level: 4

### render_validation_errors

::: promptise.mcpcast.schema.render_validation_errors
    options:
      show_source: false
      heading_level: 4

### is_python_keyword

::: promptise.mcpcast.schema.is_python_keyword
    options:
      show_source: false
      heading_level: 4

### SOFT_KEYWORDS

::: promptise.mcpcast.schema.SOFT_KEYWORDS
    options:
      show_source: false
      heading_level: 4

---

## Parsing

Load an OpenAPI document (file path, URL, inline JSON/YAML text, or dict) and flatten it into `Operation` records with dereferenced schemas, parameter wire locations, OAuth scopes, deprecation flags and success response schemas.

### load_spec

::: promptise.mcpcast.load_spec
    options:
      show_source: false
      heading_level: 4

### check_document

::: promptise.mcpcast.check_document
    options:
      show_source: false
      heading_level: 4

### scrub_strings

::: promptise.mcpcast.scrub_strings
    options:
      show_source: false
      heading_level: 4

### public_url

::: promptise.mcpcast.public_url
    options:
      show_source: false
      heading_level: 4

### expanded_nodes

::: promptise.mcpcast.expanded_nodes
    options:
      show_source: false
      heading_level: 4

### MAX_DOCUMENT_NODES

::: promptise.mcpcast.parse.MAX_DOCUMENT_NODES
    options:
      show_source: false
      heading_level: 4

### is_url

::: promptise.mcpcast.is_url
    options:
      show_source: false
      heading_level: 4

### extract_operations

::: promptise.mcpcast.extract_operations
    options:
      show_source: false
      heading_level: 4

### Operation

::: promptise.mcpcast.Operation
    options:
      show_source: false
      heading_level: 4

### ParamSpec

::: promptise.mcpcast.ParamSpec
    options:
      show_source: false
      heading_level: 4

### SecurityScheme

::: promptise.mcpcast.SecurityScheme
    options:
      show_source: false
      heading_level: 4

### api_name_from_spec

::: promptise.mcpcast.api_name_from_spec
    options:
      show_source: false
      heading_level: 4

### spec_base_url

::: promptise.mcpcast.spec_base_url
    options:
      show_source: false
      heading_level: 4

### spec_title

::: promptise.mcpcast.spec_title
    options:
      show_source: false
      heading_level: 4

### spec_description

::: promptise.mcpcast.spec_description
    options:
      show_source: false
      heading_level: 4

### spec_summary_line

::: promptise.mcpcast.spec_summary_line
    options:
      show_source: false
      heading_level: 4

---

## Classification

Deterministic, ordered risk rules over the operation id, path and summary (method → destructive verbs → money words → `POST` led by a query verb with no mutating verb → write), then one escalation step per signal: an OAuth scope containing `admin`/`root`/`superuser` (a `write:` scope does not escalate), an `admin`/`internal`/`sudo`/`impersonate` path segment, `deprecated: true`, or a `GET` that names a destructive verb. The free-form description is never matched. The curator may escalate a class, never relax it.

### classify

::: promptise.mcpcast.classify.classify
    options:
      show_source: false
      heading_level: 4

### classify_operation

::: promptise.mcpcast.classify_operation
    options:
      show_source: false
      heading_level: 4

### risk_floor

::: promptise.mcpcast.risk_floor
    options:
      show_source: false
      heading_level: 4

### Classification

::: promptise.mcpcast.Classification
    options:
      show_source: false
      heading_level: 4

---

## Planning

The `--no-curate` path: one tool per operation, filtered by the safety profile, with everything not generated recorded in `plan.dropped` with a reason.

### build_plan

::: promptise.mcpcast.build_plan
    options:
      show_source: false
      heading_level: 4

### route_base_url

::: promptise.mcpcast.plan.route_base_url
    options:
      show_source: false
      heading_level: 4

### derive_tool_name

::: promptise.mcpcast.derive_tool_name
    options:
      show_source: false
      heading_level: 4

### example_value

::: promptise.mcpcast.example_value
    options:
      show_source: false
      heading_level: 4

### resolve_base_url

::: promptise.mcpcast.resolve_base_url
    options:
      show_source: false
      heading_level: 4

### unmappable_reason

::: promptise.mcpcast.unmappable_reason
    options:
      show_source: false
      heading_level: 4

### credential_slot

::: promptise.mcpcast.credential_slot
    options:
      show_source: false
      heading_level: 4

### CredentialSlot

::: promptise.mcpcast.CredentialSlot
    options:
      show_source: false
      heading_level: 4

### example_mismatch

::: promptise.mcpcast.example_mismatch
    options:
      show_source: false
      heading_level: 4

### make_example

::: promptise.mcpcast.make_example
    options:
      show_source: false
      heading_level: 4

---

## Curation

The model proposes a budgeted, collapsed, renamed, LLM-described tool surface; `check_postconditions()` rejects any proposal that exceeds the budget, references unknown operations, downgrades risk, hides a required parameter without a default, or keeps a deprecated operation. Violations are fed back to the model up to `max_attempts` times, then `curate()` raises `MCPcastError` — there is no silent fallback.

### CurationResult

::: promptise.mcpcast.CurationResult
    options:
      show_source: false
      heading_level: 4

### CuratedTool

::: promptise.mcpcast.CuratedTool
    options:
      show_source: false
      heading_level: 4

### CuratedParam

::: promptise.mcpcast.CuratedParam
    options:
      show_source: false
      heading_level: 4

### CurationViolation

::: promptise.mcpcast.CurationViolation
    options:
      show_source: false
      heading_level: 4

### apply_curation

::: promptise.mcpcast.apply_curation
    options:
      show_source: false
      heading_level: 4

### check_postconditions

::: promptise.mcpcast.check_postconditions
    options:
      show_source: false
      heading_level: 4

### Completer

`Completer` is the type of the `complete=` hook accepted by `curate()`, `generate_tasks()` and `evaluate()`: an `async (system_prompt, user_prompt) -> str` callable. Tests inject a scripted one; production code always goes through `build_agent()`.

### render_curation_prompt

::: promptise.mcpcast.render_curation_prompt
    options:
      show_source: false
      heading_level: 4

---

## Code generation

`write_project()` emits a real project: the `<name>_mcp/` package (config, HTTP client, approval gate, `build_server()`, one tools module per resource), the `server.py` launcher (exposes `build_server`, a module-level `server` and `main(argv)`; runs without installing), `tests/` (a pytest suite through the full pipeline), `README.md`, and — written once — `pyproject.toml`, `Dockerfile`, `.env.example` and `.gitignore` (`SCAFFOLD_ONCE`). The package depends only on `promptise` and `httpx`. `render_project()` returns every file as `{path: text}`; `load_generated_server()` imports a written project through its launcher, dropping any previously imported copy of the package first.

### render_project

::: promptise.mcpcast.render_project
    options:
      show_source: false
      heading_level: 4

### load_generated_server

::: promptise.mcpcast.load_generated_server
    options:
      show_source: false
      heading_level: 4

### SCAFFOLD_ONCE

::: promptise.mcpcast.SCAFFOLD_ONCE
    options:
      show_source: false
      heading_level: 4

### package_name

::: promptise.mcpcast.package_name
    options:
      show_source: false
      heading_level: 4

### tool_group

::: promptise.mcpcast.tool_group
    options:
      show_source: false
      heading_level: 4

### describe_written

::: promptise.mcpcast.describe_written
    options:
      show_source: false
      heading_level: 4

### render_readme

::: promptise.mcpcast.render_readme
    options:
      show_source: false
      heading_level: 4

---

## Agent Readiness

`evaluate()` generates tasks (one expected tool each), drives the generated server in-process with a real `build_agent()` through `TestClient`, and scores the run: `score = 0.6 × task success + 0.4 × correct-tool-selected-first`, graded A (≥ 0.9), B (≥ 0.75), C (≥ 0.6), D (≥ 0.4), else F. Reads may reach the live API; writes, destructive and financial calls hit spec-derived mocks behind an auto-approver, so an evaluation never changes real data.

### EvalReport

::: promptise.mcpcast.EvalReport
    options:
      show_source: false
      heading_level: 4

### EvalTask

::: promptise.mcpcast.EvalTask
    options:
      show_source: false
      heading_level: 4

### TaskResult

::: promptise.mcpcast.TaskResult
    options:
      show_source: false
      heading_level: 4

### ToolCall

::: promptise.mcpcast.ToolCall
    options:
      show_source: false
      heading_level: 4

### ConfusedPair

::: promptise.mcpcast.ConfusedPair
    options:
      show_source: false
      heading_level: 4

### CallRecorder

::: promptise.mcpcast.CallRecorder
    options:
      show_source: false
      heading_level: 4

### DEFAULT_EVAL_TASKS

::: promptise.mcpcast.readiness.DEFAULT_EVAL_TASKS
    options:
      show_source: false
      heading_level: 4

### generate_tasks

::: promptise.mcpcast.generate_tasks
    options:
      show_source: false
      heading_level: 4

### tools_from_server

::: promptise.mcpcast.tools_from_server
    options:
      show_source: false
      heading_level: 4

### mock_transport

::: promptise.mcpcast.mock_transport
    options:
      show_source: false
      heading_level: 4

### EvalTransport

::: promptise.mcpcast.readiness.EvalTransport
    options:
      show_source: false
      heading_level: 4

### base_url_override

::: promptise.mcpcast.readiness.base_url_override
    options:
      show_source: false
      heading_level: 4

### NO_MOCK_STATUS

::: promptise.mcpcast.readiness.NO_MOCK_STATUS
    options:
      show_source: false
      heading_level: 4

### credential_slot (readiness)

::: promptise.mcpcast.readiness.credential_slot
    options:
      show_source: false
      heading_level: 4

### score

::: promptise.mcpcast.score
    options:
      show_source: false
      heading_level: 4

### grade_for

::: promptise.mcpcast.grade_for
    options:
      show_source: false
      heading_level: 4

### write_eval

::: promptise.mcpcast.write_eval
    options:
      show_source: false
      heading_level: 4

---

## Guided setup

`promptise.mcpcast.wizard` is the terminal wizard behind `promptise mcpcast` with no arguments (see [Guided Setup](../mcpcast/guided-setup.md)). `run_wizard()` opens it; the plain helpers it is built from carry no UI state and are importable on their own.

### run_wizard

::: promptise.mcpcast.wizard.run_wizard
    options:
      show_source: false
      heading_level: 4

### MCPcastWizard

::: promptise.mcpcast.wizard.MCPcastWizard
    options:
      show_source: false
      heading_level: 4
      members: false

### WizardResult

::: promptise.mcpcast.wizard.WizardResult
    options:
      show_source: false
      heading_level: 4

### WizardSettings

::: promptise.mcpcast.wizard.WizardSettings
    options:
      show_source: false
      heading_level: 4

### ParsedSpec

::: promptise.mcpcast.wizard.ParsedSpec
    options:
      show_source: false
      heading_level: 4

### preview_profile

::: promptise.mcpcast.wizard.preview_profile
    options:
      show_source: false
      heading_level: 4

### detect_local_apis

::: promptise.mcpcast.wizard.detect_local_apis
    options:
      show_source: false
      heading_level: 4

### review_warnings

::: promptise.mcpcast.wizard.review_warnings
    options:
      show_source: false
      heading_level: 4

### recommended_auth

::: promptise.mcpcast.wizard.recommended_auth
    options:
      show_source: false
      heading_level: 4

### equivalent_command

::: promptise.mcpcast.wizard.equivalent_command
    options:
      show_source: false
      heading_level: 4

### quote_argument

::: promptise.mcpcast.wizard.quote_argument
    options:
      show_source: false
      heading_level: 4

### probe_local_apis

::: promptise.mcpcast.wizard.probe_local_apis
    options:
      show_source: false
      heading_level: 4

### public_source

::: promptise.mcpcast.wizard.public_source
    options:
      show_source: false
      heading_level: 4

### Candidate, Detection, ProfilePreview, Usage

::: promptise.mcpcast.wizard.Candidate
    options:
      show_source: false
      heading_level: 4

::: promptise.mcpcast.wizard.Detection
    options:
      show_source: false
      heading_level: 4

::: promptise.mcpcast.wizard.ProfilePreview
    options:
      show_source: false
      heading_level: 4

::: promptise.mcpcast.wizard.Usage
    options:
      show_source: false
      heading_level: 4

### Constants

`DEFAULT_MODEL` (the curation model when none is chosen), `STEP_NAMES` (the seven steps in order), `PROBE_PORTS` and `PROBE_PATHS` (the loopback ports and document paths local API detection looks at).

---

## Source files

| File | Purpose |
|---|---|
| `src/promptise/mcpcast/__init__.py` | Package exports and the deterministic `mcpcast()` entry point (load → extract → classify → plan) |
| `src/promptise/mcpcast/schema.py` | The plan schema: `MCPcastPlan` and its Pydantic models, the `RiskClass` / `SafetyProfile` / `AuthMode` / `ApprovalMode` enums, plan invariants, YAML round-trip, `MCPcastError` |
| `src/promptise/mcpcast/parse.py` | OpenAPI 3.x / Swagger 2 loading (`load_spec`, `is_url`), spec metadata (`spec_title`, `spec_description`, `spec_base_url`, `api_name_from_spec`) and operation extraction with local `$ref` inlining (`extract_operations`, `Operation`, `ParamSpec`) |
| `src/promptise/mcpcast/classify.py` | Deterministic, ordered risk classification with escalation signals (`classify`, `classify_operation`, `Classification`) |
| `src/promptise/mcpcast/plan.py` | Deterministic planning: one tool per operation, profile filtering, tool budget, snake_case naming and worked examples (`build_plan`, `derive_tool_name`, `example_value`, `make_example`) |
| `src/promptise/mcpcast/curate.py` | LLM-assisted tool design with every post-condition enforced in code (`curate`, `check_postconditions`, `apply_curation`, `render_curation_prompt`, `CurationResult`, `CuratedTool`, `CuratedParam`, `CurationViolation`) |
| `src/promptise/mcpcast/emit.py` | Code generation from the plan: the `<name>_mcp/` package, the `server.py` launcher, `tests/`, `README.md` and the scaffold files (`render_project`, `write_project`, `load_generated_server`, `package_name`, `tool_group`, `SCAFFOLD_ONCE`, `describe_written`) |
| `src/promptise/mcpcast/readiness.py` | Agent Readiness Score: task generation, in-process agent evaluation through `TestClient`, spec-derived upstream mocks, scoring, grading and the `eval/` report (`evaluate`, `generate_tasks`, `tools_from_server`, `mock_transport`, `EvalTransport`, `base_url_override`, `NO_MOCK_STATUS`, `credential_slot`, `score`, `grade_for`, `write_eval`, `EvalReport`, `EvalTask`, `TaskResult`, `ToolCall`, `ConfusedPair`) |
| `src/promptise/mcpcast/wizard.py` | The guided setup: the Textual terminal wizard behind `promptise mcpcast` with no arguments, plus the plain helpers it is built from (`run_wizard`, `MCPcastWizard`, `ParsedSpec`, `preview_profile`, `detect_local_apis`, `review_warnings`, `recommended_auth`, `equivalent_command`, `WizardResult`) |
| `src/promptise/mcpcast/_llm.py` | Private LLM plumbing shared by curation and readiness: every model call goes through `build_agent()`; a scripted `Completer` can be injected by tests (`complete=`), never by production code |
