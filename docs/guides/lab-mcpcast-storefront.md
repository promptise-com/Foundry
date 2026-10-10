---
title: Lab — Build an MCP Server for Your SaaS API so Customers Can Use Any AI With It
description: Turn an existing OpenAPI spec into a curated, safe MCP server with promptise mcpcast — risk-classified tools, server-side human approval for refunds, per-tenant upstream credentials, and an Agent Readiness Score that tells you exactly what to fix. A complete, runnable lab.
keywords: MCP server for your SaaS API, let customers use AI with your product, human approval AI agent, OpenAPI to MCP server, agent readiness score, four-eyes approval, MCP server from REST API
---

# Lab: MCPcast Your SaaS API

Your customers keep asking the same question: *"does it work with Claude?"*

You could build a chatbot into your product. Almost everyone does, and almost
nobody uses it. The other answer is better: **ship an MCP server for the API you
already have**, and let every customer bring their own AI — Claude Desktop,
Claude Code, Cursor, an internal agent, whatever they already run.

The hard part was never the protocol. It is that a REST API is not a tool
surface: 200 endpoints is not a set of tools, `POST /orders/{id}/refund` must
never fire without a human, and no one can tell you whether an agent can
actually *drive* what you shipped.

This lab does the whole thing for a fictional commerce SaaS, end to end, with
runnable code you can point at your own spec afterwards.

## What You'll Build

From one OpenAPI file:

- an **MCP server as editable Python** — not a black box, not a proxy: an
  installable package with its own tests that depends only on `promptise` and
  `httpx`
- a **risk-classified tool surface** where every read is open, every write is
  approval-gated, and every excluded operation is recorded *with a reason*
- **multi-tenant auth**: each customer's MCP key maps to their own upstream
  credential, and reviewers only ever see their own tenant's calls
- a **four-eyes approval gate** enforced server-side — a refund waits for a
  second human, for *any* MCP client, and is denied on timeout
- an **Agent Readiness Score**: a real agent, real tasks, a grade, and a list of
  specific tool-design fixes
- a measured **before/after** when you apply those fixes

Everything runs on your machine. The upstream API is faked in-process (the way
you would fake Stripe in a test suite); the agents, tools, approvals and scores
are real.

## Prerequisites

```bash
pip install promptise
export OPENAI_API_KEY=sk-...
```

The complete lab lives in `examples/mcp/mcpcast_storefront_lab/`:

| File | What it is |
|---|---|
| `storefront.yaml` | The API the company already ships — 13 operations |
| `fake_api.py` | The upstream, as an in-process `httpx.MockTransport` that records every request |
| `run.py` | The lab itself, in five printed sections |

```bash
.venv/bin/python examples/mcp/mcpcast_storefront_lab/run.py
```

## The API you already have

Nothing about `storefront.yaml` is unusual — that is the point. Customers,
orders, a refund endpoint, a subscription cancellation, one admin route, one
deprecated CSV export, one multipart upload:

```yaml
  /orders/{order_id}/refund:
    post:
      operationId: refundOrder
      tags: [billing]
      summary: Refund an order
      description: >
        Move money back to the customer's original payment method. Partial
        refunds are allowed; the sum of refunds may not exceed the order total.
      parameters:
        - $ref: "#/components/parameters/OrderId"
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [amount]
              properties:
                amount:
                  type: number
                  description: Amount to refund, in the order's currency.
```

## Step 1 — Generate the server, and read what it decided

One call does parse → classify → plan → emit. `write_project` puts three files
on disk: the plan, the server, and a README for whoever runs it.

```python
from promptise.mcpcast import AuthMode, SafetyProfile, mcpcast, write_project

plan = mcpcast(SPEC, profile=SafetyProfile.FULL, auth=AuthMode.API_KEY, name="storefront")
write_project(plan, OUT)          # the plan, the package, the launcher, tests, scaffold
```

The CLI does the same, and adds the LLM curation pass and the review screen:

```bash
promptise mcpcast storefront.yaml --profile full --auth api-key --out storefront-mcp
```

The interesting part is not that it generated something — it is that it can tell
you *why* it generated exactly this:

```text
  tool                  risk         approval  upstream operation
  get_health            read         -         GET /health
  list_customers        read         -         GET /customers
  search_customers      read         -         POST /customers/search
  get_customer          read         -         GET /customers/{customer_id}
  list_orders           read         -         GET /orders
  create_order          write        human     POST /orders
  get_order             read         -         GET /orders/{order_id}
  refund_order          financial    human     POST /orders/{order_id}/refund
  cancel_subscription   destructive  human     DELETE /subscriptions/{subscription_id}
  get_revenue_report    write        human     GET /reports/revenue
  get_customer_pii      write        human     GET /admin/customers/{customer_id}/pii

  why the classifier decided that (deterministic, no model):
    search_customers      POST that only queries ('search')
    refund_order          mentions money ('refund')
    cancel_subscription   DELETE is destructive
    get_revenue_report    GET is a read; escalated: requires scope 'admin:read'
    get_customer_pii      GET is a read; escalated: path is admin-only

  not exposed (nothing vanishes silently):
    uploadOrderAttachment: unsupported by mcpcast: unsupported request body media type multipart/form-data
    exportLegacyOrders: deprecated in spec

  the same spec under each safety profile:
    read-only     6 tools (0 approval-gated), 7 not exposed
    standard      9 tools (3 approval-gated), 4 not exposed
    full         11 tools (5 approval-gated), 2 not exposed

  auth: api-key   approval: pending (four-eyes)
```

Four things happened there that you would otherwise have hand-written and got
wrong:

- **`POST /customers/search` is a read.** Method is not intent. A POST whose
  leading verb is a query verb, with no mutating verb anywhere, is classified
  `read` — so it survives the `read-only` profile.
- **Two GETs were escalated.** `GET /reports/revenue` carries the OAuth scope
  `admin:read`; `GET /admin/customers/{id}/pii` sits under `/admin`. Both are
  HTTP reads that no one should hand an agent by default, so they come out as
  `write` — exposed only from `standard` up, and approval-gated.
- **Nothing disappeared quietly.** The multipart upload cannot be sent by the
  generated client and the CSV export is `deprecated: true` in the spec; both
  are in `plan.dropped` with the reason, not silently missing.
- **The profile is the dial.** Same spec, three surfaces. `read-only` is the
  default because it is the one you can ship to strangers.

!!! tip "The plan file is the source of truth"
    `mcpcast.plan.yaml` is what you edit — tool names, descriptions, examples,
    hidden parameters, the dropped list — and then regenerate with
    `promptise mcpcast mcpcast.plan.yaml`. Your edits survive; the package and
    `server.py` are always rebuilt from the plan. See
    [the plan file](../mcp/server/mcpcast.md#the-plan-file).

## Step 2 — Point a real agent at it

The generated server is a normal `MCPServer`, so an agent can drive it
in-process through `TestClient` — the full pipeline (validation, guards,
middleware, approval gate, handler) with no ports involved.
`tools_from_server()` does the bridging, and a `CallRecorder` notes every tool
the agent picks.

```python
async with api.client() as http:                      # the fake upstream
    server = module.build_server(http_client=http)
    recorder = CallRecorder()
    recorder.begin("lookup")
    tools = await tools_from_server(server, recorder=recorder,
                                    headers={"x-api-key": AGENT_KEY})
    agent = await build_agent(model="openai:gpt-5-mini", servers={},
                              extra_tools=tools, instructions=...)
    result = await agent.ainvoke({"messages": [HumanMessage(content=question)]})
```

```text
  question: A customer wrote in from ada@northwind.example. Who are they, and what are their two most recent orders?

  tools the agent chose:
    search_customers({"email": "ada@northwind.example"}) -> ok
    list_orders({"customer_id": "CUS-1001", "limit": 2}) -> ok

  answer: They’re Ada Lovelace (customer ID CUS-1001), email ada@northwind.example — on the Pro plan (created 2025-03-04).

Two most recent orders:
- ORD-1002
  - Status: shipped
  - Placed: 2026-02-11
  - Total: €49.90
  - Items: 1 × HUB-USBC-7 (unit price €49.90)
...

  requests the API received: ['POST /v1/customers/search', 'GET /v1/orders?customer_id=CUS-1001&limit=2']
  the credential each one carried: 'Bearer northwind-upstream-token'
```

No prompt engineering, no tool wiring, no glue: the agent chained a search into
a filtered list because the generated descriptions and examples told it how.

That last line is the multi-tenancy in action. Under `--auth api-key` the
generated server runs with `require_tenant=True`: the MCP client presents a key
from `MCPCAST_CLIENT_KEYS`, the key names a tenant, and the tenant selects the
upstream credential from `MCPCAST_UPSTREAM_TOKENS` — read per call, so rotating a
customer's token needs no restart.

```python
CLIENT_KEYS = {
    AGENT_KEY:        {"client_id": "support-agent", "tenant_id": "northwind"},
    REVIEWER_KEY:     {"client_id": "dana", "tenant_id": "northwind", "roles": ["approver"]},
    OTHER_TENANT_KEY: {"client_id": "gus",  "tenant_id": "globex",    "roles": ["approver"]},
}
UPSTREAM_TOKENS = {"northwind": "Bearer northwind-upstream-token", ...}
```

## Step 3 — The refund that waits for a human

Now the agent is asked to move money. Same server, same agent, one different
sentence:

```text
  request: Order ORD-1003 arrived damaged. Refund the customer 24 euros for it.

  the gate is holding the call:
    refund_order {"order_id": "ORD-1003", "amount": 24.0, "reason": "Item arrived damaged"}
    requested by client_id='support-agent' tenant='northwind'
    refund requests the API has received so far: 0
    API log so far: []
```

The agent called the tool. The call is *parked* — because this is what the
generator wrote for that operation, in `storefront_mcp/tools/billing.py`, for
you to read:

```python
    # -- refund_order (financial, requires approval) -----------------------------
    @server.tool(
        name="refund_order",
        description=(
            "Refund an order. Move money back to the customer's original payment method. Partial "
            "refunds are allowed; the sum of refunds may not exceed the order total.\n"
            "\n"
            "Parameters:\n"
            "  - order_id (string, required): The order identifier, e.g. `ORD-1002`.\n"
            "  - amount (number, required): Amount to refund, in the order's currency.\n"
            "  - reason (string): Why the refund was issued (kept for the audit trail).\n"
            "\n"
            'Example: {"order_id": "ORD-1002", "amount": 24}'
        ),
        tags=["billing"],
        open_world_hint=True,
        requires_approval=True,
    )
    async def refund_order(
        ctx: RequestContext,
        order_id: str,
        amount: float,
        reason: str | None = None,
    ) -> Any:
        "Refund an order. Move money back to the customer's original payment method.…"
        args: dict[str, Any] = {
            "order_id": order_id,
            "amount": amount,
            "reason": reason,
        }
        return await upstream.call(select_route(ROUTES["refund_order"], args), args, ctx)
```

`ApprovalGateMiddleware` holds a `requires_approval` call before the handler
runs. **The upstream API received nothing at all** — that
empty log is the whole point. If nobody decides within
`MCPCAST_APPROVAL_TIMEOUT`, the call is denied, not allowed.

Because the callers are identified (api-key auth), the approver is the `pending`
one: an independent human, reached through two generated, tenant-scoped tools —
`approvals_list` and `approvals_decide`, both guarded by the `approver` role.

Approving is itself a tool call, so a reviewer can be any MCP client — a
dashboard, a Slack bot, or a second Claude session:

```python
(reply,) = await reviewer.call_tool(
    "approvals_decide",
    {"request_id": pending["request_id"], "approve": True, "reason": "damaged goods"},
    headers={"x-api-key": REVIEWER_KEY},          # dana, tenant northwind, role approver
)
```

```text
  a reviewer from ANOTHER tenant:
    globex/gus approvals_list -> []   (sees nothing)
    globex/gus tries to approve -> NOT_FOUND

  the caller approving their own request (same principal, approver role):
    -> ACCESS_DENIED: A caller may not approve their own request

  dana (northwind, approver role) approves:
    approvals_decide -> {"request_id": "73c810f0...", "approved": true, "resolved": true}

  answer: Refund successful — REF-3001: €24 refunded to order ORD-1003 (reason: Item arrived damaged).

  refund requests the API received: 1
    POST /v1/orders/ORD-1003/refund body={"amount": 24.0, "reason": "Item arrived damaged"}
    authorization='Bearer northwind-upstream-token'  (northwind's upstream credential)
```

Four attempts and what the server did with each — enforced by the server, not by
the client's good manners:

| Attempt | Outcome |
|---|---|
| A reviewer of another tenant lists pending calls | sees nothing — reviewers never see other tenants' arguments |
| That reviewer decides by request id anyway | `NOT_FOUND` — the id does not exist *for them* |
| The caller approves their own request, holding the `approver` role | `ACCESS_DENIED` — four-eyes, always |
| A different human of the same tenant approves | the call resumes and hits the API, once |

This is why the gate belongs in the server. Your customer might connect with
Claude Desktop today and a homegrown agent tomorrow; neither can opt out of a
policy that lives on your side of the wire.

!!! note "Which approver you get"
    `--auth api-key` defaults to `pending` (four-eyes review, shown here).
    `passthrough`, `env-token` and `none` default to `elicitation`: the human
    behind the *calling* client confirms the action in their own UI — the right
    shape for a personal server launched by Claude Desktop over stdio. Both are
    fail-closed. See [Human approval](../mcp/server/mcpcast.md#human-approval).

## Step 4 — Measure: the Agent Readiness Score

You now have a server that is safe. Safe is not the same as *usable*: the real
question is whether an agent picks the right tool from your descriptions.
`evaluate()` answers it with evidence — a real agent, real tasks, the full
server pipeline, and mocks derived from your spec so nothing can touch real data.

```python
report = await evaluate(
    plan, module.build_server,
    model="openai:gpt-5-mini",
    tasks=tasks,                  # eight explicit EvalTasks; --eval writes them for you
    operations=operations,        # response schemas -> realistic mocks
    live_reads=False,             # this API is fictional; mock every route
    headers={"x-api-key": AGENT_KEY},
)
```

```text
  Agent Readiness: A   score 1.00
    tasks succeeded 8/8   correct tool first 100%   parameter errors 0%
    t1  expected search_customers   called search_customers                               ok
    t2  expected get_order          called get_order                                      ok
    t3  expected list_orders        called list_orders                                    ok
    t4  expected refund_order       called refund_order                                   ok
    t5  expected get_revenue_report called get_revenue_report                             ok
    t6  expected get_customer       called get_customer -> get_customer                   ok
    t7  expected list_customers     called list_customers                                 ok
    t8  expected create_order       called create_order                                   ok

  what the report says to fix:
    • 3 tools not covered by any task: `get_health`, `cancel_subscription`, `get_customer_pii` — raise --eval-tasks to score them
    • `list_customers` has no example — agents lean on examples heavily
    • `list_orders` has no example — agents lean on examples heavily

  wrote generated/v1/eval/tasks.yaml and generated/v1/eval/report.md
```

The score is `0.6 × task success + 0.4 × correct-tool-first`, graded A–F, and the
fixes are the payload: exact tool pairs the agent confuses, parameters that
produced validation errors, tools no task covered, tools a task needed and the
agent never reached for.

A grade of A on eight tasks is *not* the interesting part — the three lines under
it are. Two tools have no worked example, and three tools are not covered by any
task at all, which means the run says nothing about them. That is the honest
answer to "is my API agent-ready?": here is what I measured, here is what I did
not, here is what to fix first.

`--eval` writes the same thing to disk — `eval/tasks.yaml` (rerun the exact
tasks) and `eval/report.md` (the table above, in Markdown) — so a readiness score
can live in CI next to your other tests.

!!! warning "An evaluation never changes your data"
    The live/mock split is by **risk class**, not HTTP method: only routes of
    `read` tools may reach the real API (and only with `live_reads=True`).
    Everything else — writes, refunds, cancellations, and any GET the classifier
    escalated — is answered by a spec-derived mock behind an auto-approver.

## Step 5 — Apply the fixes, regenerate, re-measure

The report is advice, and the plan is code, so the fixes are a few lines. Here
the lab merges the two customer lookups into one *intent* tool with two routes —
the agent should be picking a customer, not an HTTP endpoint — adds the examples
the report asked for, drops the health check no task needs, and puts
`notify_customer` on a param diet so an agent can never email a customer:

```python
find_customer = ToolPlan(
    name="find_customer",
    description=(
        "Find ONE customer, by id or by email address. Pass customer_id when you "
        "know it (e.g. 'CUS-1001'), otherwise pass email. Use list_customers only "
        "to browse or filter the whole directory."
    ),
    risk=RiskClass.READ,
    routes=[get_customer.routes[0], search_customers.routes[0]],   # dispatch by what you pass
    params={"customer_id": ParamPlan(...), "email": ParamPlan(...)},
    example={"email": "ada@northwind.example"},
)

params["notify_customer"] = params["notify_customer"].model_copy(
    update={"hidden": True, "default": False}
)
```

A multi-route tool dispatches to the first route whose required parameters were
supplied, so `find_customer(customer_id=…)` hits `GET /customers/{id}` and
`find_customer(email=…)` hits `POST /customers/search`. The hidden parameter is
not hidden from *people*: its fixed value is printed in the tool description and
in the generated README, so a reviewer can see what actually runs.

In the regenerated `mcpcast.plan.yaml` the merge is one tool with two routes:

```yaml
- name: find_customer
  description: Find ONE customer, by id or by email address. Pass customer_id when you know it (e.g. 'CUS-1001'),
    otherwise pass email. Use list_customers only to browse or filter the whole directory.
  risk: read
  routes:
  - operation_id: getCustomer
    method: GET
    path: /customers/{customer_id}
    params:
      customer_id:
        location: path
        required: true
  - operation_id: searchCustomers
    method: POST
    path: /customers/search
    params:
      email:
        location: body
        required: true
  params:
    customer_id:
      description: The customer identifier, e.g. 'CUS-1001'.
    email:
      description: The customer's email address, matched exactly.
      json_schema:
        type: string
        format: email
  example:
    email: ada@northwind.example
  tags:
  - customers
```

Then the *same eight requests* run again against the regenerated server:

```text
  merged get_customer + search_customers -> find_customer (2 routes, dispatched by which id you pass)
  added the missing examples; dropped get_health
  create_order description now says: 'Always sends: notify_customer=false'
  surface: 11 -> 9 tools, 8 -> 9 of them with a worked example

  Agent Readiness: A   score 1.00
    tasks succeeded 8/8   correct tool first 100%   parameter errors 0%
    t1  expected find_customer      called find_customer                                  ok
    t2  expected get_order          called get_order                                      ok
    t3  expected list_orders        called list_orders                                    ok
    t4  expected refund_order       called refund_order                                   ok
    t5  expected get_revenue_report called get_revenue_report                             ok
    t6  expected find_customer      called find_customer                                  ok
    t7  expected list_customers     called list_customers                                 ok
    t8  expected create_order       called create_order                                   ok

  what the report says to fix:
    • 2 tools not covered by any task: `cancel_subscription`, `get_customer_pii` — raise --eval-tasks to score them

  before: A (1.00)   after: A (1.00)
  identical score on this run — the surface was already unambiguous for these eight
  requests, and the edits kept it that way with two fewer tools.
```

**The grade did not move, and the lab says so.** That is what an honest metric
looks like: this surface was already unambiguous for these eight requests, so the
score had nowhere to go. What *did* change is worth having anyway — two fewer
tools to choose from and to pay tokens for, every tool carrying a worked example,
and a fix list that shrank from three items to one. On a real API with a hundred
endpoints the first run is rarely an A, and the confusion pairs the report prints
are usually the difference between a demo and a product.

One row is worth reading twice. In the first run `t6` shows
`get_customer -> get_customer`: the agent asked again, because the mock answers
*every* customer id with the spec's canned `Customer` example, so the record that
came back did not carry the id it had asked for. That wobble is mock-shaped, not
tool-shaped — mocks are how an evaluation stays safe, and the exact row varies
from run to run because the model does. Run with `live_reads=True`
(and a real credential in `MCPCAST_EVAL_AUTHORIZATION`) when you want reads to hit
the real API and the numbers to reflect real payloads.

## Step 6 — Hand it to Claude Desktop, Claude Code or Cursor

For a personal server that a desktop client launches over stdio, generate it
with `--auth env-token` so it carries one credential — yours:

```bash
promptise mcpcast examples/mcp/mcpcast_storefront_lab/storefront.yaml \
  --profile standard --auth env-token --out storefront-mcp
```

```json
{
  "mcpServers": {
    "storefront": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["/absolute/path/to/storefront-mcp/server.py"],
      "env": {
        "MCPCAST_UPSTREAM_TOKEN": "Bearer <your Storefront API token>",
        "MCPCAST_BASE_URL": "https://api.storefront.example/v1"
      }
    }
  }
}
```

Claude Code takes the same command and the same token, passed with `-e` so it
lands in the server's environment — the client launches the server itself and
does not see your shell's exports:

```bash
claude mcp add storefront -e MCPCAST_UPSTREAM_TOKEN="Bearer <your Storefront API token>" \
  -- /absolute/path/to/.venv/bin/python /absolute/path/to/storefront-mcp/server.py
```

Cursor's `.cursor/mcp.json` uses the identical `command`/`args`/`env` shape. In
all three the write tools still stop and ask the human before they run — that
policy came with the server, not with the client.

For the multi-tenant deployment your customers connect to, keep `--auth api-key`
and serve the same file over HTTP — the generated module exposes a module-level
`server`, so `promptise serve` picks it up:

```bash
# Mint real keys: python -c 'import secrets; print("sk-" + secrets.token_urlsafe(32))'
# (the server refuses placeholders and the documentation's sample keys)
export MCPCAST_CLIENT_KEYS='{"sk-<agent key>": {"client_id": "acme-agent", "tenant_id": "acme"},
                            "sk-<reviewer key>": {"client_id": "dana", "tenant_id": "acme",
                                                  "roles": ["approver"]}}'
export MCPCAST_UPSTREAM_TOKENS='{"acme": "Bearer <acme upstream token>"}'
promptise serve server:server --transport http --port 8080
```

## What You've Built

- **An MCP server for an API you did not write a line of glue for** — 13 spec
  operations became 11 tools, with 5 of them behind a human.
- **A safety story you can explain to a security reviewer**: risk classes with
  reasons, three profiles, drops with reasons, and approval enforced at the tool.
- **Per-tenant isolation**: one server, many customers, each with their own
  upstream credential, and reviewers who only see their own tenant.
- **Evidence instead of vibes**: a grade for how well an agent drives your tool
  surface, and a list of what to fix — with a before/after when you fix it.
- **Editable output**: `mcpcast.plan.yaml` is yours, the package is regenerated
  from it, and all of it is plain, reviewable, diffable Python.

## The whole lab

```python
--8<-- "examples/mcp/mcpcast_storefront_lab/run.py"
```

The fake upstream it runs against is `fake_api.py` — a dict-backed store behind
an `httpx.MockTransport` that records every request, so the lab can *prove* what
did and did not reach the API.

## Next Steps

- [MCPcast an Existing API](mcpcast-existing-api.md) — the step-by-step guide to
  running the same pipeline against *your* spec, with curation and review
- [MCPcast reference](../mcp/server/mcpcast.md) — pipeline, curation, plan schema,
  auth modes, readiness scoring, limits
- [Approval Gates](../mcp/server/approval-gates.md) — elicitation, pending,
  webhook and callback approvers in depth
- [Multi-Tenancy](../mcp/server/multi-tenancy.md) — tenant claims, isolation
  keys, per-tenant rate limits and audit
- [Build a Secure Multi-Tenant Agent Platform](secure-multi-tenant-platform.md) —
  the same guarantees for a server you write by hand
