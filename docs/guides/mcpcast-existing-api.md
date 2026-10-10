---
title: MCPcast Your Existing API — turn an OpenAPI spec into an MCP server any AI can use
description: A step-by-step guide to turning an existing HTTP API into a curated, safe, agent-ready MCP server with `promptise mcpcast`. From one command to a reviewed tool plan, human-approved writes, a real AI connected over stdio, an Agent Readiness Score, and a shipped deployment.
keywords: MCPcast your API, OpenAPI to MCP server, make my API work with Claude, let customers use AI with our product, Swagger to MCP, MCP server from OpenAPI, Claude Desktop MCP server, agent-ready API, MCP server tutorial
---

# MCPcast an Existing API

You already have an API. Your customers already have AI assistants. This guide
connects the two — without writing an MCP server by hand and without handing a
language model a `DELETE` endpoint.

By the end you will have taken a real OpenAPI document to a running MCP server
that Claude Desktop, Claude Code, Cursor or a Promptise agent can drive, with a
tool surface you reviewed, writes a human has to approve, and a score that tells
you how well an agent actually copes with it.

!!! tip "This is the step-by-step path"
    Every feature used here has a deep reference on
    [MCPcast an Existing API](../mcp/server/mcpcast.md) — risk rules, curation
    post-conditions, the full plan schema, auth modes and the readiness metrics.
    Read this page first, that one when you need the detail. Every option is
    also listed in the [CLI reference](../core/cli.md), and every exported
    symbol in the [API reference](../api/mcpcast.md).

## What You'll Build

An MCP server for a storefront's Orders API, built in twelve steps:

- **Steps 1–2** — a working read-only server from one offline command, and the
  plan file that says exactly what was exposed and what wasn't
- **Step 3** — a real AI answering business questions through it
- **Steps 4–6** — writes, deletes and refunds opened up deliberately, each one
  held by an approval gate the *server* enforces, including tenant-scoped
  four-eyes review
- **Steps 7–8** — a model-designed tool surface, then your own hand edits on top
- **Steps 9–10** — an Agent Readiness Score, what it is telling you, and the
  grade after acting on it
- **Steps 11–12** — deployment, the right auth mode per shape, and a CI test
  that stops the tool surface from rotting

## Concepts

**An API surface is not a tool surface.** Your API was designed for a developer
with reference docs open, a debugger and a retry loop. An agent has one shot per
turn, a context window, and no way to ask what `include_deleted` does. Two
hundred endpoints do not become two hundred good tools: they become two hundred
near-duplicates the model picks between badly. What works is a small set of
*intent* tools — `find_orders`, `refund_order` — with descriptions written for a
model, parameters trimmed to the ones a caller should actually choose, and one
worked example each. `promptise mcpcast` is the machine that turns the first
thing into the second.

**Safe by default, because the downside is asymmetric.** A tool that reads the
wrong record wastes a turn. A tool that refunds the wrong order costs money and
trust. So generation is read-only unless you opt in, risk is classified by fixed
deterministic rules rather than by a model's opinion, and everything that
changes data is gated behind a human decision enforced in the server's
middleware — not as a courtesy of whichever client happens to be calling. A
model can never talk the gate down, because the model is not the one holding it.

**"MCPcast" is a commercial move, not just a technical one.** The instinct when
AI shows up on the roadmap is to build another agent inside your product. That
is a large bet on a feature your customers may already have solved: they are
running Claude, Cursor, ChatGPT and their own internal agents already. MCPcasting
your API sells the other side of that trade — *use any AI you like with our
product* — for a fraction of the work. One generated server, and every
MCP-capable assistant your customers already run becomes a client of your API.

**The plan file is the artifact you actually own.** `mcpcast` does not hide
behind a runtime that reinterprets your spec on every start. It emits
`mcpcast.plan.yaml` (what the surface *is*) and a real project around it — an
installable package of readable Promptise code, a launcher, a test suite, a
`Dockerfile` and a README with the install snippets. You edit the plan,
regenerate, and diff the result in code review like anything else you ship.

---

## The API we'll MCPcast

A small storefront API with one endpoint of each interesting shape: plain reads,
a search that happens to be a `POST`, writes, a delete, a refund, an internal
admin route, a deprecated route, and a file upload. Save it as `orders.yaml`:

```yaml
openapi: 3.0.3
info:
  title: Northwind Orders
  version: "2.4.0"
  description: Orders, customers and payments for the Northwind storefront.
servers: [{url: "http://127.0.0.1:8931"}]

# Reusable pieces, so the paths below stay readable.
components:
  schemas:
    Order: {type: object, properties: {id: {type: string}, customer_id: {type: string},
      status: {type: string}, total_cents: {type: integer}}}
    Customer: {type: object, properties: {id: {type: string}, name: {type: string},
      email: {type: string}}}
    OrderList: {type: object, properties: {orders: {type: array, items: {$ref: "#/components/schemas/Order"}}}}
  parameters:
    OrderId: {name: order_id, in: path, required: true, schema: {type: string}}
    CustomerId: {name: customer_id, in: path, required: true, schema: {type: string}}
  responses:
    Order: {description: An order, content: {application/json: {schema: {$ref: "#/components/schemas/Order"}}}}
    Orders: {description: A list of orders, content: {application/json: {schema: {$ref: "#/components/schemas/OrderList"}}}}
    Customer: {description: A customer, content: {application/json: {schema: {$ref: "#/components/schemas/Customer"}}}}

paths:
  /orders:
    get:
      operationId: listOrders
      summary: List orders
      parameters:
        - {name: status, in: query, schema: {type: string, enum: [open, shipped, refunded]}}
        - {name: limit, in: query, schema: {type: integer, default: 20, maximum: 100}}
      responses: {"200": {$ref: "#/components/responses/Orders"}}
    post:
      operationId: createOrder
      summary: Create an order
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [customer_id, items]
              properties:
                customer_id: {type: string}
                items: {type: array, items: {type: object, properties: {sku: {type: string}, qty: {type: integer}}}}
                notify_customer: {type: boolean, default: true}
      responses: {"201": {$ref: "#/components/responses/Order"}}

  /orders/{order_id}:
    get:
      operationId: getOrder
      summary: Get an order
      parameters: [{$ref: "#/components/parameters/OrderId"}]
      responses: {"200": {$ref: "#/components/responses/Order"}}
    patch:
      operationId: updateOrder
      summary: Update an order
      parameters: [{$ref: "#/components/parameters/OrderId"}]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                status: {type: string, enum: [open, shipped]}
                shipping_address: {type: string}
      responses: {"200": {$ref: "#/components/responses/Order"}}

  /orders/search:                     # a POST that is really a read
    post:
      operationId: searchOrders
      summary: Search orders by customer, SKU or date range
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [query]
              properties:
                query: {type: string}
                customer_id: {type: string}
                since: {type: string, format: date}
      responses: {"200": {$ref: "#/components/responses/Orders"}}

  /orders/{order_id}/cancel:          # destructive: 'cancel' is a destructive verb
    post:
      operationId: cancelOrder
      summary: Cancel an order
      parameters: [{$ref: "#/components/parameters/OrderId"}]
      requestBody:
        content:
          application/json:
            schema:
              type: object
              properties:
                reason: {type: string}
                notify_customer: {type: boolean, default: true}
      responses: {"200": {$ref: "#/components/responses/Order"}}

  /orders/{order_id}/refund:          # financial: 'refund' is a money word
    post:
      operationId: refundOrder
      summary: Refund an order to the original payment method
      parameters: [{$ref: "#/components/parameters/OrderId"}]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [amount_cents]
              properties:
                amount_cents: {type: integer}
                reason: {type: string}
      responses:
        "200":
          description: The refund
          content:
            application/json:
              schema: {type: object, properties: {refund_id: {type: string}, amount_cents: {type: integer}}}

  /orders/{order_id}/attachments:     # mcpcast cannot encode multipart bodies
    post:
      operationId: uploadOrderAttachment
      summary: Attach a document to an order
      parameters: [{$ref: "#/components/parameters/OrderId"}]
      requestBody:
        required: true
        content:
          multipart/form-data:
            schema: {type: object, properties: {file: {type: string, format: binary}}}
      responses: {"201": {description: Created}}

  /customers/{customer_id}:
    get:
      operationId: getCustomer
      summary: Get a customer
      parameters: [{$ref: "#/components/parameters/CustomerId"}]
      responses: {"200": {$ref: "#/components/responses/Customer"}}
    delete:
      operationId: deleteCustomer
      summary: Delete a customer and all their data
      parameters: [{$ref: "#/components/parameters/CustomerId"}]
      responses: {"204": {description: Deleted}}

  /admin/customers/{customer_id}/risk:   # a GET the 'admin' segment escalates
    get:
      operationId: getCustomerRiskProfile
      summary: Read the internal fraud-risk profile for a customer
      parameters: [{$ref: "#/components/parameters/CustomerId"}]
      responses:
        "200":
          description: Risk profile
          content:
            application/json:
              schema: {type: object, properties: {customer_id: {type: string}, score: {type: integer}}}

  /v1/orders:
    get:
      operationId: listOrdersV1
      summary: List orders (legacy)
      deprecated: true
      responses: {"200": {description: Orders}}
```

To follow along with your own API instead, keep the shape and pass
`--base-url https://api.yourcompany.com`. Every command and every block of
output on this page was run against the stand-in below, so the numbers are real
and you can reproduce them.

??? note "The stand-in API — stdlib only, run it with `python api_stub.py`"

    ```python
    # api_stub.py — run with: python api_stub.py
    import json, re
    from http.server import BaseHTTPRequestHandler, HTTPServer

    ORDERS = [
        {"id": "ORD-1001", "customer_id": "CUS-77", "status": "shipped", "total_cents": 4250},
        {"id": "ORD-1002", "customer_id": "CUS-77", "status": "open", "total_cents": 1899},
        {"id": "ORD-1003", "customer_id": "CUS-42", "status": "refunded", "total_cents": 9900},
    ]
    CUSTOMERS = {
        "CUS-77": {"id": "CUS-77", "name": "Ada Byron", "email": "ada@example.com"},
        "CUS-42": {"id": "CUS-42", "name": "Grace Hopper", "email": "grace@example.com"},
    }

    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload, status=200):
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/orders":
                return self._send({"orders": ORDERS})
            if m := re.fullmatch(r"/orders/([^/]+)", path):
                order = next((o for o in ORDERS if o["id"] == m.group(1)), None)
                return self._send(order or {"error": "not found"}, 200 if order else 404)
            if m := re.fullmatch(r"/customers/([^/]+)", path):
                cus = CUSTOMERS.get(m.group(1))
                return self._send(cus or {"error": "not found"}, 200 if cus else 404)
            self._send({"error": "not found"}, 404)

        def do_POST(self):
            if self.path == "/orders/search":
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                cid = json.loads(raw or b"{}").get("customer_id")
                hits = [o for o in ORDERS if not cid or o["customer_id"] == cid]
                return self._send({"orders": hits})
            self._send({"error": "not found"}, 404)

        def log_message(self, *args):
            pass

    HTTPServer(("127.0.0.1", 8931), Handler).serve_forever()
    ```

---

## Step 1 — From spec to server in one command

```bash
pip install promptise
promptise mcpcast orders.yaml --no-curate
```

```text
Parsed 12 operations from orders.yaml; profile=read-only auth=passthrough
╭──────────────────────── mcpcast ────────────────────────╮
│ northwind-orders → northwind-orders-mcp/               │
│   tools: 4  (0 require human approval)                 │
│   not exposed: 8 operations (with reasons in the plan) │
│   files: northwind_orders_mcp/ (8 modules), tests/ (2), server.py, README.md,    │
│   pyproject.toml, Dockerfile, .env.example, .gitignore                          │
╰──────────────────────────────────────────────────────────────────────────────────╯
```

Twelve operations in, four tools out, in about two and a half seconds — most of
which is importing Promptise. `--no-curate` is the fully offline path: no model,
no network beyond fetching the spec if it is a URL. A project lands in
`northwind-orders-mcp/`:

| Path | What it is | On regeneration |
|---|---|---|
| `mcpcast.plan.yaml` | The tool surface as data — the file you edit | never touched |
| `northwind_orders_mcp/` | A real Promptise `MCPServer` as a package: `config.py`, `upstream.py` (the HTTP client), `approval.py` (the gate), `server.py` (`build_server()`), `tools/<resource>.py`; depends only on `promptise` and `httpx` | rewritten |
| `server.py` | Launcher: `python server.py` runs the package without installing it; `promptise serve server:server` works | rewritten |
| `tests/` | `conftest.py` + `test_tools.py`: every tool listed, routed to the right operation on a fake upstream, gated when it changes data | rewritten |
| `README.md` | Install snippets for Claude Desktop / Claude Code / Cursor, the layout, the tool table, the not-exposed table, the env vars | rewritten |
| `pyproject.toml`, `Dockerfile`, `.env.example`, `.gitignore` | `pip install -e .` → the `northwind-orders-mcp` command; the image; the configuration template | written once, then yours |

Two defaults did the heavy lifting. **`--profile read-only`** means nothing that
changes data was generated at all — this is why four tools came out of twelve
operations. And the spec's own `servers:` entry became the upstream base URL; if
your spec omits one, `mcpcast` refuses to guess and asks for `--base-url`.

The package is code, not configuration. Read it, `git diff` it, step through it
in a debugger, run `pytest` in it. What it is *not* is a place to edit: every
module says so in its docstring, because regenerating from the plan overwrites
it. Your own tests go in another file under `tests/`; your packaging changes in
`pyproject.toml` stay.

---

## Step 2 — Read the plan

`mcpcast.plan.yaml` is the source of truth. Everything downstream — the server,
the README, the score — is derived from it.

```yaml
# mcpcast.plan.yaml — the editable source of truth for this MCP server.
# Edit tool names, descriptions, examples, hidden params or the dropped
# list, then regenerate (this file is left untouched):  promptise mcpcast mcpcast.plan.yaml
version: 1
api:
  name: northwind-orders
  base_url: http://127.0.0.1:8931
  auth: passthrough
  description: 'Northwind Orders: Orders, customers and payments for the Northwind storefront.'
  spec_source: orders.yaml
profile: read-only
tools:
- name: get_order
  description: Get an order
  risk: read
  routes:
  - operation_id: getOrder
    method: GET
    path: /orders/{order_id}
    params:
      order_id:
        location: path
        required: true
  params:
    order_id:
      required: true
  example:
    order_id: '123'
# … list_orders, search_orders, get_customer …
dropped:
- operation_id: createOrder
  reason: write operation excluded by profile 'read-only'
- operation_id: updateOrder
  reason: write operation excluded by profile 'read-only'
- operation_id: cancelOrder
  reason: destructive operation excluded by profile 'read-only'
- operation_id: refundOrder
  reason: financial operation excluded by profile 'read-only'
- operation_id: uploadOrderAttachment
  reason: 'unsupported by mcpcast: unsupported request body media type multipart/form-data'
- operation_id: deleteCustomer
  reason: destructive operation excluded by profile 'read-only'
- operation_id: getCustomerRiskProfile
  reason: write operation excluded by profile 'read-only'
- operation_id: listOrdersV1
  reason: deprecated in spec
```

**Nothing vanishes silently.** Every operation in the spec is either a tool or
an entry in `dropped` with a reason. The offline path writes five kinds of
reason — excluded by the safety profile, over the tool budget, deprecated in the
spec, not selected by curation, or unsupported by mcpcast — and curation (step 7)
adds its own, in its own words. The last of the five is the honest one:
`uploadOrderAttachment` uses `multipart/form-data`, which the generated upstream
client does not encode, so it is named rather than quietly skipped.

Two rows deserve a second look:

- `searchOrders` is a `POST`, and it is a **read** tool. The classifier looks at
  the leading verb of the operation id, summary or last path segment — `search`,
  `query`, `find`, `lookup`, `list`, `preview`, `validate`, `calculate` and
  friends — and a `POST` that leads with one of those and mentions no mutating
  verb is a read. Real APIs put search behind `POST` all the time; treating them
  as writes would gate half your read surface behind a human.
- `getCustomerRiskProfile` is a `GET`, and it was dropped as a **write**. Its
  path contains the segment `admin`, which escalates the risk one level. The
  same escalation applies to `internal`, `sudo` and `impersonate` segments, to
  OAuth scopes containing `admin`/`root`/`superuser`, to anything marked
  `deprecated`, and to a `GET` whose own text names a destructive verb — the
  classic side-effecting legacy read.

The rules are fixed, deterministic and matched only on the operation id, path
and summary — never the free-form description, so nobody can lower a tool's risk
by rewording their API docs. The full table is in the
[risk classification reference](../mcp/server/mcpcast.md#risk-classification).

---

## Step 3 — Connect a real AI

A read-only server is already worth shipping, so let's plug one into an actual
assistant. Desktop clients launch the server themselves over stdio, which means
they cannot send an `Authorization` header — so generate with `--auth env-token`
and give the server one credential from its environment:

```bash
promptise mcpcast orders.yaml --no-curate --auth env-token --output northwind-personal
export MCPCAST_UPSTREAM_TOKEN='Bearer <your API token>'
```

The generated `README.md` carries a ready-to-paste snippet per client, with the
credential already in the client's own environment block — a desktop client
launched from the dock never sees your shell's exports, so an `export` in a
terminal would not reach it:

=== "Claude Desktop"

    `claude_desktop_config.json`:

    ```json
    {
      "mcpServers": {
        "northwind-orders": {
          "command": "python",
          "args": ["/absolute/path/to/server.py"],
          "env": { "MCPCAST_UPSTREAM_TOKEN": "Bearer <your API token>" }
        }
      }
    }
    ```

=== "Claude Code"

    ```bash
    claude mcp add northwind-orders \
        -e MCPCAST_UPSTREAM_TOKEN='Bearer <your API token>' \
        -- python /absolute/path/to/server.py
    ```

=== "Cursor"

    `.cursor/mcp.json`:

    ```json
    {
      "mcpServers": {
        "northwind-orders": {
          "command": "python",
          "args": ["/absolute/path/to/server.py"],
          "env": { "MCPCAST_UPSTREAM_TOKEN": "Bearer <your API token>" }
        }
      }
    }
    ```

=== "Promptise agent"

    ```python
    from promptise import build_agent, StdioServerSpec

    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={"northwind": StdioServerSpec(
            command="python", args=["northwind-personal/server.py"],
            env={"MCPCAST_UPSTREAM_TOKEN": "Bearer <your API token>"},
        )},
        instructions="You are connected to the Northwind storefront over MCP. Use the tools.",
    )
    result = await agent.ainvoke({"messages": [
        {"role": "user", "content": "Which orders belong to customer CUS-77, and what is the total of the shipped one?"}
    ]})
    print(result["messages"][-1].content)
    ```

Run the agent version and the answer comes back from your API, through tools
nobody hand-wrote:

```text
Customer CUS-77 has orders ORD-1001 (shipped) and ORD-1002 (open).
The shipped order ORD-1001 totals $42.50.
```

Forget the credential and the failure is specific rather than mysterious — the
model even relays it:

```text
- UPSTREAM_AUTH_MISSING: MCPCAST_UPSTREAM_TOKEN is not set
```

**Which auth mode when:**

| Deployment shape | Mode | Why |
|---|---|---|
| A personal server your desktop client launches over stdio | `env-token` | stdio carries no headers; one credential from the environment |
| One shared HTTP server, each caller acting as themselves | `passthrough` (default) | the caller's `Authorization` header is forwarded unchanged, so your API's own permissions still apply |
| A multi-tenant service you host for customers | `api-key` | clients present `x-api-key`; each key names a tenant whose upstream credential never leaves the server |
| A local demo against a sandbox API | `none` | no credentials at all — and the generated server refuses to bind to a non-loopback address |

---

## Step 4 — Open up writes, and look before you leap

Reads are the safe half. To let an agent actually *do* something, raise the
profile — and pass `--review` so you see the plan before a line of code is
written:

```bash
promptise mcpcast orders.yaml --profile standard --no-curate --review \
    --output northwind-standard
```

```text
Parsed 12 operations from orders.yaml; profile=standard auth=passthrough
                                    Tools (7) — profile standard
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━┓
┃ Tool              ┃ Risk  ┃ Approval ┃ Operations        ┃ Params            ┃ Description       ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━┩
│ list_orders       │ read  │ —        │ GET /orders       │ status, limit     │ List orders       │
│ create_order      │ write │ required │ POST /orders      │ customer_id,      │ Create an order   │
│                   │       │          │                   │ items,            │                   │
│                   │       │          │                   │ notify_customer   │                   │
│ get_order         │ read  │ —        │ GET               │ order_id          │ Get an order      │
│                   │       │          │ /orders/{order_i… │                   │                   │
│ update_order      │ write │ required │ PATCH             │ order_id, status, │ Update an order   │
│                   │       │          │ /orders/{order_i… │ shipping_address  │                   │
│ search_orders     │ read  │ —        │ POST              │ query,            │ Search orders by  │
│                   │       │          │ /orders/search    │ customer_id,      │ customer, SKU or  │
│                   │       │          │                   │ since             │ date range        │
│ get_customer      │ read  │ —        │ GET               │ customer_id       │ Get a customer    │
│                   │       │          │ /customers/{cust… │                   │                   │
│ get_customer_ris… │ write │ required │ GET               │ customer_id       │ Read the internal │
│                   │       │          │ /admin/customers… │                   │ fraud-risk        │
│                   │       │          │                   │                   │ profile for a     │
│                   │       │          │                   │                   │ customer          │
└───────────────────┴───────┴──────────┴───────────────────┴───────────────────┴───────────────────┘
                                          Not exposed (5)
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Operation             ┃ Reason                                                                   ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ cancelOrder           │ destructive operation excluded by profile 'standard'                     │
│ refundOrder           │ financial operation excluded by profile 'standard'                       │
│ uploadOrderAttachment │ unsupported by mcpcast: unsupported request body media type               │
│                       │ multipart/form-data                                                      │
│ deleteCustomer        │ destructive operation excluded by profile 'standard'                     │
│ listOrdersV1          │ deprecated in spec                                                       │
└───────────────────────┴──────────────────────────────────────────────────────────────────────────┘
Write the project? [Y/n]:
```

Answer `n` and you get `Aborted — nothing written.` with exit code 1 and no
output directory at all. `--yes` skips the question for scripts.

Three things to notice. `standard` added **writes only** — the delete and the
refund are still excluded, by name, with the profile that excluded them.
`get_customer_risk_profile` shows up as `write` with approval required even
though it is a `GET`, because the `admin` segment escalated it; the review table
is exactly where you catch that and decide whether an agent should see your
fraud model at all. And every non-read tool says **required** in the Approval
column — that is not advice, it is what the generated code does.

The three profiles:

| Profile | Exposes | Approval |
|---|---|---|
| `read-only` (default) | reads | none needed |
| `standard` | reads + writes | every write |
| `full` | reads + writes + destructive + financial | every non-read |

The profile is enforced by the plan schema itself, not just by the generator: a
plan whose tool risk exceeds its profile, or whose write is not gated, **refuses
to load**. Hand-editing the YAML cannot smuggle a refund into a `standard`
server.

---

## Step 5 — What `requires_approval=True` actually does

In `northwind-standard/server.py`, each write carries the flag, and an
`ApprovalGateMiddleware` sits in the chain ahead of every handler:

```python
    @server.tool(
        name='create_order',
        description='Create an order\n\nParameters:\n  - customer_id (string, required)…',
        open_world_hint=True,
        requires_approval=True,
    )
```

The gate is server-side, so it holds for *any* MCP client — Claude Desktop,
Cursor, a curl script, someone else's agent framework. Call a gated tool with
nobody available to approve, and the call is denied rather than allowed:

```python
from promptise.mcp.server import TestClient
# import build_server from the generated northwind-standard/server.py

client = TestClient(build_server(), meta={"authorization": "Bearer demo-token"})
out = await client.call_tool("create_order",
                             {"customer_id": "CUS-77", "items": [{"sku": "TEA-1", "qty": 1}]})
```

```json
{
  "error": {
    "code": "APPROVAL_DENIED",
    "message": "Approval denied for tool 'create_order': no live MCP session for elicitation (client or transport does not support it) — denied fail-closed",
    "retryable": false,
    "details": {
      "approval_request_id": "59e359e0d6f86d21802ce2f51e7dc0e6",
      "reviewer_id": "elicitation"
    }
  }
}
```

Fail-closed is the whole point: a timeout denies, a handler crash denies, and a
reviewer who tries to *modify* the arguments denies, because the call was
already bound and silently rewriting it would execute something nobody
approved. The default timeout is 300s (`MCPCAST_APPROVAL_TIMEOUT`).

Now give it an approver. `build_server()` takes an `approval_handler` (and an
`http_client`, so this test never touches the real API):

```python
async def approve(request):
    print(f"APPROVAL ASKED: tool={request.tool_name} args={request.arguments}")
    return True

server = build_server(
    approval_handler=approve,
    http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake_upstream)),
)
client = TestClient(server, meta={"authorization": "Bearer demo-token"})
out = await client.call_tool("create_order",
                             {"customer_id": "CUS-77", "items": [{"sku": "TEA-1", "qty": 1}]})
```

```text
APPROVAL ASKED: tool=create_order args={'customer_id': 'CUS-77', 'items': [{'sku': 'TEA-1', 'qty': 1}], 'notify_customer': None}
{"id": "ORD-1004", "status": "open", "total_cents": 1899}
```

`TestClient` runs the **full** pipeline in-process — validation, guards,
middleware, the approval gate, the handler — with no network and no transport.
It is the right way to test a generated server; see
[Testing](../mcp/server/testing.md).

In production you do not pass a callable: the default approver for
`passthrough`, `env-token` and `none` is **elicitation**, which asks the human
behind the calling client through the MCP elicitation protocol. Details and
alternative handlers (webhooks, queues, callbacks) are in
[Approval Gates](../mcp/server/approval-gates.md).

---

## Step 6 — Destructive and financial, with four-eyes review

`--profile full` adds the delete and the refund. Those are exactly the calls you
do not want the *requesting* human to wave through, so pair them with an
approval mode that involves someone else. That needs identified callers, which
means `--auth api-key`:

```bash
promptise mcpcast orders.yaml --profile full --no-curate --auth api-key \
    --output northwind-full
```

```text
Parsed 12 operations from orders.yaml; profile=full auth=api-key
╭──────────────────────── mcpcast ────────────────────────╮
│ northwind-orders → northwind-full/                     │
│   tools: 10  (6 require human approval)                │
│   not exposed: 2 operations (with reasons in the plan) │
│   files: northwind_orders_mcp/ (8 modules), tests/ (2), server.py, README.md,    │
│   pyproject.toml, Dockerfile, .env.example, .gitignore                          │
╰──────────────────────────────────────────────────────────────────────────────────╯
```

Only the deprecated and the multipart operation remain unexposed. With
`api-key`, the default approval mode becomes **pending**: the call parks in a
store and a *different* human of the same tenant, holding the `approver` role,
releases it through two generated tools, `approvals_list` and
`approvals_decide`. Ask for `pending` without identified callers and the plan
refuses to exist:

```text
Error: could not build plan: 1 validation error for ApiPlan
  Value error, approval mode 'pending' (independent four-eyes review) needs identified
  callers so a caller can never approve their own request — use --auth api-key, or
  approval 'elicitation'
```

Two keys, two roles, and the whole flow runs in-process:

```python
# Any keys but the documentation's sample ones (`sk-acme`, `sk-reviewer`) and
# `<placeholders>`, which a generated server refuses at start-up.
os.environ["MCPCAST_CLIENT_KEYS"] = json.dumps({
    "sk-acme-agent-7f3k":  {"client_id": "acme-agent", "tenant_id": "acme", "roles": []},
    "sk-acme-dana-2q9m":   {"client_id": "dana",       "tenant_id": "acme", "roles": ["approver"]},
})
os.environ["MCPCAST_UPSTREAM_TOKENS"] = json.dumps({"acme": "Bearer acme-upstream-token"})

agent = TestClient(server, meta={"x-api-key": "sk-acme-agent-7f3k"})
dana  = TestClient(server, meta={"x-api-key": "sk-acme-dana-2q9m"})

call = asyncio.create_task(agent.call_tool("refund_order",
                                           {"order_id": "ORD-1001", "amount_cents": 4250}))
await asyncio.sleep(0.3)          # let the call reach the gate and park
pending = await dana.call_tool("approvals_list", {})
req_id = json.loads(pending[0].text)[0]["request_id"]
await dana.call_tool("approvals_decide",
                     {"request_id": req_id, "approve": True, "reason": "verified with the customer"})
print((await call)[0].text)
```

```text
PENDING: [{"request_id": "10cbdcc3dc721d98e4a8b3bf88c58185", "tool": "refund_order",
           "arguments": {"order_id": "ORD-1001", "amount_cents": 4250, "reason": null},
           "client_id": "acme-agent", "tenant_id": "acme", "age_seconds": 0.3}]
DECIDED: {"request_id": "10cbdcc3dc721d98e4a8b3bf88c58185", "approved": true, "resolved": true}
RESULT:  {"refund_id": "RFN-9", "amount_cents": 4250}
```

Separation of duties is enforced twice over. The requesting client cannot even
see the queue, because the approval tools are role-guarded; and a reviewer who
*is* the requester is refused with `ACCESS_DENIED: A caller may not approve their
own request`. The store is tenant-scoped too, so a reviewer who asks about
another tenant's request gets `NOT_FOUND` rather than a leak, and a caller
without the `approver` role never reaches the queue at all:

```json
{
  "error": {
    "code": "ACCESS_DENIED",
    "message": "Requires any of roles [approver], but client has [(none)]",
    "retryable": false,
    "details": {"guard": "HasRole", "tool": "approvals_list"}
  }
}
```

The generated server sets `require_tenant=True`, so a key without a tenant
cannot call anything at all. See [Multi-Tenancy](../mcp/server/multi-tenancy.md)
and [Auth & Security](../mcp/server/auth-security.md).

---

## Step 7 — Let the model design the tool surface

Everything so far was deterministic: one tool per operation, descriptions lifted
from the spec's `summary`. That is a faithful surface, not a good one. Drop
`--no-curate` and a model designs the surface instead — with every one of its
decisions checked by code:

```bash
promptise mcpcast orders.yaml --profile standard --output northwind-curated
```

Curation uses `openai:gpt-5-mini` unless you pass `--model`, which accepts any
provider string Promptise understands — `--model azure:chat-prod` for an Azure
OpenAI deployment, `--model foundry:Llama-3.3-70B-Instruct` for the Azure AI
Foundry catalog, `--model bedrock:...`, `--model gemini:...` — see
[Model Setup](../getting-started/model-setup.md); `promptise models check <string>`
tells you what the provider still needs before you run.

```text
Parsed 12 operations from orders.yaml; profile=standard auth=passthrough
Curating with openai:gpt-5-mini (budget 25 tools)…
╭──────────────────────── mcpcast ────────────────────────╮
│ northwind-orders → northwind-curated/                  │
│   tools: 4  (2 require human approval)                 │
│   not exposed: 6 operations (with reasons in the plan) │
│   files: northwind_orders_mcp/ (8 modules), tests/ (2), server.py, README.md,    │
│   pyproject.toml, Dockerfile, .env.example, .gitignore                          │
╰──────────────────────────────────────────────────────────────────────────────────╯
```

Seven tools became four. Compare what the model did with what step 4 produced:

| What curation does | In this run |
|---|---|
| **Collapse CRUD into intent tools** | `getOrder`, `searchOrders` and `listOrders` merged into one `find_orders` |
| **Rename for the caller, not the codebase** | `getCustomer` → `find_customer` |
| **Rewrite descriptions for an LLM audience** | "Do NOT use this to create or modify orders — use create_order or update_order for writes" |
| **Put the tool surface on a diet** | `since` hidden; `notify_customer` hidden with a fixed default; `limit` defaulted to 50 |
| **Add one worked example per tool** | `{"order_id": "ORD-1007"}` |
| **Write real reasons for what it drops** | `getCustomerRiskProfile`: "internal admin endpoint (fraud/risk profile) — not appropriate for end-user agents" |

The merged tool looks like this in the plan (routes abbreviated — each one also
lists its own `params`). Note the order:

```yaml
- name: find_orders
  description: Retrieve orders by id, list recent orders, or run a search across orders
    (by customer, SKU, or date). … Do NOT use this to create or modify orders — use
    create_order or update_order for writes …
  risk: read
  routes:
  - {operation_id: getOrder,     method: GET,  path: /orders/{order_id}}   # needs order_id
  - {operation_id: searchOrders, method: POST, path: /orders/search}       # needs query
  - {operation_id: listOrders,   method: GET,  path: /orders}              # needs nothing
```

and reaches the model as one description that tells it how to choose:

```text
Provide one of: order_id | query | (no parameters)

Example: {"order_id": "ORD-1007"}
```

**The model proposes; the code disposes.** Every proposal is checked against
post-conditions before it can become a plan: at most `--max-tools` tools; valid,
unique, non-reserved names; every referenced operation exists; an operation in
at most one tool; nothing both kept and dropped; a hidden required parameter must
have a default; deprecated operations must be dropped; routes must be orderable;
and — the important one — **risk is never downgraded below the classifier**.
Curation can escalate a tool's risk. It can never talk one down. The safety
profile is then applied *after* curation, which is why the model's own drop
reasons and the profile's sit side by side in `dropped`.

A violation is fed back to the model *with its previous proposal*, up to three
attempts. Then it stops, loudly — there is no silent fallback to a worse plan:

```text
Error: curation failed after 3 attempt(s): curation proposal rejected:
- tool 'manage_order': operation 'cancelOrder' can never be selected because 'updateOrder'
  (listed earlier) needs a subset of its parameters; list the more specific operation first
Re-run with --no-curate for a deterministic one-tool-per-operation plan.
```

That message is worth understanding, because it is the one post-condition about
*shape* rather than safety. A tool dispatches to the first of its operations
whose required parameters were all supplied, so merged operations have to be
tellable apart: `PATCH /orders/{id}` and `POST /orders/{id}/cancel` both require
exactly `order_id`, and no call could ever select the second. The prompt states
that rule and most proposals respect it — the `--max-tools 6` run above squeezed
twelve operations into three tools without tripping it. If you do hit the wall,
give the model room (the default budget is 25), narrow the spec, or accept the
deterministic plan.

One thing the code fixes rather than rejects: an example the model invents that
contradicts the spec. Asked for a `create_order` example against an item schema
of `{sku, qty}`, a model will cheerfully write `{sku, quantity, unit_price_cents}`
— which would teach every agent reading the tool the wrong shape. Values that do
not fit the declared schema are dropped and a spec-derived example is used
instead, so a hint is never worth failing a run over.

!!! note "Curation output varies between runs"
    Names, wording and merges will differ on your machine — the commands are
    identical, the plan is not. Everything from here on is shown against the
    plan this run produced.

---

## Step 8 — Edit the plan by hand, then regenerate

The model gets you 90% there. The last 10% is yours, and it is a YAML edit.
Rename a tool:

```yaml
- name: get_customer        # was: find_customer
```

Freeze a parameter the agent should never choose, and give it the value your
business actually wants:

```yaml
    notify_customer:
      description: If true, send an order confirmation to the customer.
      json_schema: {type: boolean, default: true}
      default: true
      hidden: true
```

A hidden parameter is invisible to the *agent*, never to your reviewers —
hiding what the server actually sends would defeat the point of the approval
gate. So the fixed value is printed in the tool's own description and in the
README, where whoever approves the call will read it:

```text
Always sends: notify_customer=true
```

Then regenerate. The command takes the **plan** as its argument:

```bash
cd northwind-curated && promptise mcpcast mcpcast.plan.yaml -o .
```

```text
Regenerating from plan mcpcast.plan.yaml (4 tools)
╭──────────────────────── mcpcast ────────────────────────╮
│ northwind-orders → ./                                  │
│   tools: 4  (2 require human approval)                 │
│   not exposed: 6 operations (with reasons in the plan) │
│   files: northwind_orders_mcp/ (8 modules), tests/ (2), server.py, README.md   │
╰────────────────────────────────────────────────────────────────────────────────╯
```

Only the derived files were written — **your plan file is never rewritten**,
comments and all, and neither are `pyproject.toml`, `Dockerfile`,
`.env.example` or `.gitignore` once they exist. Spec-only flags are rejected here rather than
silently ignored, because the plan already answers them:

```text
Invalid value: --profile cannot be combined with a plan file — edit mcpcast.plan.yaml
and regenerate instead
```

**Route order is load-bearing.** A multi-route tool dispatches to the first
route whose required visible parameters were supplied, so the most specific
route must come first. Move `listOrders` (which requires nothing) above the
others and the plan will not load:

```text
Error: invalid plan:
1 validation error for MCPcastPlan
tools.0
  Value error, tool 'find_orders': route 'searchOrders' can never be selected — route
  'listOrders' (listed earlier) is satisfied by any call that satisfies it; reorder the
  routes so the more specific one comes first
```

The other edits worth knowing: rewrite any `description` (the highest-leverage
change you can make), add or fix an `example`, move an operation into `dropped`
with your own reason, or raise a `risk` — for instance if `update_order` in
*your* business really is destructive. Lowering one below what the classifier
can read from the route itself is refused when the plan loads: set
`delete_order` to `risk: read` and regeneration stops with `tool 'delete_order'
is declared 'read' but its route 'deleteOrder' (DELETE /orders/{order-id}) is at
least 'destructive': a plan may raise a tool's risk, never lower it`. The floor
comes from the method, the path and the operation id (a `DELETE`, a `PUT`, a
destructive or money word, an admin-style path), so a `POST` whose only "search"
signal was in a summary the plan does not carry can still be a `read` — which is
also what the classifier concluded from the spec.

---

## Step 9 — Measure it: the Agent Readiness Score

You now have a tool surface. Whether an *agent* can use it is an empirical
question, so answer it empirically:

```bash
promptise mcpcast northwind-curated/mcpcast.plan.yaml --eval
```

A model writes tasks — one expected tool each — and a real `build_agent()` drives
the generated server in-process through `TestClient`, exercising the whole
pipeline including the approval gate. **An evaluation never changes real data:**
routes belonging to `read`-classified tools may hit your live API, and every
other route is answered by a mock derived from your spec's response schemas,
behind an auto-approver. The split is by risk class, not HTTP method, so the
`GET` that got escalated to `write` is mocked too.

```text
Hint: auth is passthrough — set MCPCAST_EVAL_AUTHORIZATION='Bearer <token>' so live reads reach
the API authenticated (a placeholder credential is used otherwise).
Evaluating with openai:gpt-5-mini (20 tasks)…
Agent Readiness: B  (16/20 tasks succeeded)
Report: northwind-curated/eval/report.md  Tasks: northwind-curated/eval/tasks.yaml
```

`eval/report.md` opens with the numbers:

```markdown
# Agent Readiness: B  (16/20 tasks succeeded)

- Score: 0.88
- Correct tool selected first: 100%
- Parameter error rate: 0%
- Tools never used: 0
- Tools not covered by any task: 0
```

| Metric | What it is telling you |
|---|---|
| **Task success** | The expected tool was called *and* returned ok. The end-to-end number, worth 60% of the score. |
| **Correct tool selected first** | The agent's *first* call was the right tool — a pure description-quality metric, worth the other 40%. Low here means your tools read alike. |
| **Parameter error rate** | The share of tool calls rejected with `VALIDATION_ERROR`. Usually a missing example, a misleading one, or a required parameter no description mentions. |
| **Tools never used** | A task needed this tool and the agent never called it. It is invisible — bad name, or a neighbour whose description sounds like it covers the job. |
| **Tools not covered** | No generated task exercised this tool at all. Raise `--eval-tasks`. |
| **Confused pairs** | The agent reached for A when the task needed B, with counts. These are the descriptions that need to say what they are *not* for. |

Anything that scores badly turns into a named fix under a `## Fixes` heading in
the report, phrased as an instruction rather than a statistic. The heading is
omitted entirely when there is nothing to fix — as on this run — so here is what
it looks like on a surface that does have problems:

```markdown
## Fixes

- ✗ `find_orders` vs `get_customer` are ambiguous — the agent picked `get_customer`
  in 3/5 runs that needed `find_orders` → merge them, or say in each description
  when NOT to use it
- • `refund_order` has no example — agents lean on examples heavily
```

`score = 0.6 × success + 0.4 × selection`, graded A ≥ 0.90, B ≥ 0.75, C ≥ 0.60,
D ≥ 0.40, else F. Two files land in `eval/`: `tasks.yaml` (every task, so you
can re-run the same set) and `report.md` (the numbers, the fixes, and a row per
task). This run listed no fixes at all — which is exactly why the next step
starts by reading the task table instead of the grade.

!!! tip "Authenticated live reads"
    With `passthrough` or `api-key` auth the evaluation warns that it is using a
    placeholder credential. Export `MCPCAST_EVAL_AUTHORIZATION='Bearer <token>'`
    to let the read half hit your API for real.

---

## Step 10 — Act on the score, then score again

A grade is a headline; the task table is the story. Four tasks failed, and in
every one the agent picked the *right* tool — the call itself came back not-ok:

```markdown
| # | Task                                                            | Expected       | Called          | Result |
|---|-----------------------------------------------------------------|----------------|-----------------|--------|
| t1 | Please fetch the order with ID ORD-1007 and show me its details. | `find_orders`  | find_orders ✗   | ✗ |
| t7 | Pull up order ORD-1015 so I can review its status and items.     | `find_orders`  | find_orders ✗   | ✗ |
| t19 | Retrieve the customer record for CUST-42 …                      | `get_customer` | get_customer ✗  | ✗ |
| t20 | Can you pull up the profile for customer CUST-99?               | `get_customer` | get_customer ✗  | ✗ |
```

`find_orders` and `get_customer` are reads, so those calls went to the **live**
API — which answered `404`, because `ORD-1007`, `CUST-42` and `CUST-99` do not
exist. Where did the task author get those ids? From the plan. The model's
examples invented an id format (`ORD-1007`, `CUST-42`) that does not match the
real one (`ORD-1001`, `CUS-77`), and everything downstream — the generated
tasks, and a real agent guessing at ids — inherited the invention.

So fix the examples, in the plan, to look like your actual data:

```yaml
  example:
    order_id: ORD-1001        # was: ORD-1007
```

```yaml
  example:
    customer_id: CUS-77       # was: CUST-42
```

While you are in there, the `create_order` example is worth a hard look too — it
had invented `quantity` and `unit_price_cents` fields the spec's `items` schema
does not have (`sku` and `qty`). The evaluation never caught that one, because
writes are answered by mocks that accept anything. A human reading the plan
catches it in ten seconds. **Read what the model wrote.**

```yaml
  example:
    customer_id: CUS-77
    items:
    - sku: NW-ALMOND
      qty: 3
```

Regenerate, then score the same way:

```bash
promptise mcpcast northwind-curated/mcpcast.plan.yaml --eval
```

```text
Agent Readiness: A  (20/20 tasks succeeded)
```

```markdown
# Agent Readiness: A  (20/20 tasks succeeded)

- Score: 1.00
- Correct tool selected first: 100%
- Parameter error rate: 0%
```

**B (0.88) → A (1.00)** from three example edits and no code at all. The
generated tasks changed with them — `Can you pull up order ORD-1001 and show me
its details?`, `List all orders for customer CUS-77` — which is the real lesson:
examples in the plan are not decoration, they are the strongest signal a model
has about what your identifiers look like.

One honest caveat. Both grades come from a single run each, with freshly
generated tasks and a sampling agent; run the same command twice and the number
will move. For a controlled before/after, pin the task set and hand it back to
`evaluate()` instead of a count:

```python
import yaml
from promptise.mcpcast import EvalTask, evaluate

tasks = [EvalTask(**t) for t in yaml.safe_load(open("eval/tasks.yaml"))["tasks"]]
report = await evaluate(plan, build_server, tasks=tasks, operations=operations)
```

Treat a one-grade swing on regenerated tasks as noise until a re-run confirms it.

---

## Step 11 — Ship it

To try the server immediately, add `--serve` to any generation:

```bash
promptise mcpcast orders.yaml --no-curate --auth env-token --output northwind-personal \
    --force --serve --transport http --port 8977
```

```text
Serving northwind-personal/server.py over http…
INFO:     Uvicorn running on http://127.0.0.1:8977 (Press CTRL+C to quit)
```

For anything longer-lived, the launcher exposes `server` at module level, so
the standard runner works from inside the project directory — with the
dashboard, hot reload and everything else in
[Deployment](../mcp/server/deployment.md). Or install the project and it is a
command, and build the generated `Dockerfile` and it is an image:

```bash
cd northwind-personal
promptise serve server:server --transport http --port 8080 --dashboard
pip install -e . && northwind-personal-mcp --transport http --port 8080
docker build -t northwind-personal . && docker run -i --env-file .env northwind-personal   # stdio
```

All `mcpcast` progress goes to **stderr**, because under `--serve` with the stdio
transport stdout is the MCP protocol stream. Piping stdout is safe.

The auth mode is not just configuration — it is enforced. This project is
`env-token`: every call carries *your* token and there is no MCP-level
authentication, so the server binds loopback only, whichever way you start
it — `python server.py`, `promptise serve`, `--serve`, the Docker image
(which therefore serves stdio). `--auth none` behaves the same way:

```text
promptise: error: auth mode 'env-token' refuses to bind to a non-loopback address
without --public: anyone reaching the port would act with MCPCAST_UPSTREAM_TOKEN.
Pass --public (or MCPCAST_PUBLIC=1) only behind an authenticating gateway, or
regenerate with --auth api-key or passthrough for a shared deployment
```

`--public` (on the generated command line and on `promptise mcpcast --serve`)
is the explicit opt-in for a gateway-fronted deployment. A `passthrough`
server is a relay for whatever `Authorization` header reaches it and belongs
behind an authenticating gateway too — its README says so; `api-key` is the
mode that authenticates callers itself.

Everything the server reads at runtime:

| Variable | Purpose | Default |
|---|---|---|
| `MCPCAST_BASE_URL` | Override the upstream base URL (staging vs production) — every route, including operations that declare their own server. One clean absolute URL: a query string or fragment, `user:password@` or whitespace is refused at start-up (without echoing the value) | from the plan |
| `MCPCAST_UPSTREAM_TOKEN` | `env-token`: the credential sent upstream, read per call — the full `Authorization` value by default, or the raw key when the spec's security scheme names a header or query parameter (`.env.example` and `config.py` say which); a `<placeholder>` left in place is refused with `UPSTREAM_AUTH_INVALID` | — |
| `MCPCAST_CLIENT_KEYS` | `api-key`: JSON map of key → `{client_id, tenant_id, roles}`, read at start-up. The server refuses to start with none, with a `<placeholder>` key or with a key from the documentation (`sk-acme`, `sk-reviewer`); `.env.example` ships no working key — mint one with `python -c 'import secrets; print("sk-" + secrets.token_urlsafe(32))'` | — |
| `MCPCAST_UPSTREAM_TOKENS` | `api-key`: JSON map of tenant → upstream credential (presented where the spec's security scheme puts it, like `MCPCAST_UPSTREAM_TOKEN`), read per call so rotation needs no restart | — |
| `MCPCAST_TIMEOUT` | Total time one upstream call may take, in seconds — connecting, sending, waiting and reading the body together (`UPSTREAM_TIMEOUT` past it) | `30` |
| `MCPCAST_APPROVAL_TIMEOUT` | How long a gated call waits before it is denied | `300` |
| `MCPCAST_MAX_PENDING` | `pending` approval: how many gated calls may wait for a reviewer at once, server-wide | `100` |
| `MCPCAST_MAX_PENDING_PER_TENANT` | `pending` approval: how many of those one tenant may hold, across all of its API keys | `40` |
| `MCPCAST_MAX_PENDING_PER_CLIENT` | `pending` approval: how many one client (API key) may hold | `20` |
| `MCPCAST_MAX_RESPONSE_BYTES` | Largest upstream response body passed on to the agent (`UPSTREAM_RESPONSE_TOO_LARGE` beyond it) | `1048576` |
| `MCPCAST_ERROR_EXCERPT_CHARS` | How much of an upstream error body the agent may see (`0` hides it); the credential the server sent is scrubbed to `[redacted]` first, so an upstream that echoes it cannot leak it | `500` |
| `MCPCAST_ALLOW_INSECURE_HTTP` | `1` to send the credential to a plain-`http://` host that is not loopback; pre-set in `.env.example`, with the hosts named, when the plan's upstream is plain http | unset |
| `MCPCAST_PUBLIC` | `env-token` / `none`: `1` to bind a non-loopback address — the `--public` flag | unset |

Errors the tools return are structured, so an agent can react to them rather
than parse prose: `UPSTREAM_AUTH_MISSING` (no credential),
`UPSTREAM_AUTH_INVALID` (a credential with a control or non-ASCII character —
the message names the variable, never the value), `UPSTREAM_INSECURE` (the
credential would travel over plain http), `UPSTREAM_TIMEOUT` (the call did not
finish within `MCPCAST_TIMEOUT`; retryable), `UPSTREAM_UNREACHABLE` (connection,
DNS or protocol failure; retryable), `UPSTREAM_ERROR` (with `details.status`, `details.retry_after`;
retryable for 408/425/429 and 5xx), `UPSTREAM_RESPONSE_TOO_LARGE`,
`VALIDATION_ERROR` (bad arguments, or no route of a multi-route tool matches)
and `APPROVAL_DENIED`.

---

## Step 12 — Keep the surface from rotting

Your API will change. The tool surface should fail loudly when it does, not
drift. The Python API is the same pipeline the CLI drives, so put it in CI:

```python
"""CI guard: the shipped tool surface matches the spec and still scores well."""

import os
from pathlib import Path

import pytest

from promptise.mcpcast import (
    MCPcastPlan, RiskClass, SafetyProfile, evaluate, extract_operations, load_spec, mcpcast,
)

HERE = Path(__file__).parent
PLAN = MCPcastPlan.load(HERE / "northwind-curated" / "mcpcast.plan.yaml")


def test_every_operation_is_kept_or_dropped_with_a_reason():
    """No endpoint may quietly disappear when the spec changes."""
    spec_ops = {op.operation_id for op in extract_operations(load_spec(HERE / "orders.yaml"))}
    accounted = {op for tool in PLAN.tools for op in tool.operations}
    accounted |= {d.operation_id for d in PLAN.dropped}
    assert spec_ops == accounted, f"unaccounted operations: {spec_ops - accounted}"


def test_no_tool_is_less_risky_than_the_classifier_says():
    """A hand edit can raise a tool's risk, never lower it."""
    baseline = mcpcast(HERE / "orders.yaml", profile=SafetyProfile.FULL)
    floor = {op: t.risk for t in baseline.tools for op in t.operations}
    for tool in PLAN.tools:
        for op in tool.operations:
            # `at_least` walks the severity ladder — RiskClass is a str enum, so a
            # plain `>=` would compare the *names* and quietly accept a downgrade.
            assert tool.risk.at_least(floor[op]), f"{tool.name} downgrades {op}"


def test_every_write_is_approval_gated():
    for tool in PLAN.tools:
        if tool.risk is not RiskClass.READ:
            assert tool.requires_approval, f"{tool.name} is {tool.risk.value} but ungated"


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs a model")
@pytest.mark.asyncio
async def test_agent_readiness_does_not_regress():
    from promptise.mcpcast import load_generated_server

    module = load_generated_server(HERE / "northwind-curated" / "server.py")

    operations = extract_operations(load_spec(HERE / "orders.yaml"))
    # A model writes the tasks and a model drives the tools, so the score moves
    # a little between runs. Assert a *floor* you never want to fall through,
    # not the number you saw once — and use enough tasks that one unlucky task
    # cannot swing the result.
    report = await evaluate(PLAN, module.build_server, tasks=20, operations=operations)
    assert report.score >= 0.6, report.render_summary()
```

```text
$ pytest test_tool_surface.py -q -k "not readiness"
3 passed, 1 deselected, 2 warnings in 1.82s
```

The first three tests are offline, free, fast and deterministic — run them on
every commit. The fourth needs `pytest-asyncio`, a model and about a minute, and
its score varies run to run; treat a failure as "look at `eval/report.md`", not
as a broken build. Add a nightly job that
regenerates from the live spec URL
(`mcpcast("https://api.yourcompany.com/openapi.json", profile=SafetyProfile.STANDARD)`)
and diffs the plan: a new endpoint shows up as an unaccounted operation, and a
newly destructive one shows up as a risk-floor failure. The readiness test costs
model calls, so keep it on the schedule that matches your budget.

---

## What You've Built

- A **read-only MCP server** from one offline command, and a **standard** and
  **full** variant of the same API, each generated as editable code
- A **plan file** in which every one of the twelve operations is either a tool
  or a dropped entry with a reason — nothing unaccounted for
- Writes, deletes and refunds behind a **server-side approval gate** that fails
  closed, with elicitation for personal servers and **tenant-scoped four-eyes
  review** for the multi-tenant one
- A **model-curated tool surface** — merged, renamed, described for an LLM, put
  on a parameter diet — with every proposal checked by code and risk that can
  only go up
- Hand edits on top of it, regenerated without ever losing the plan
- An **Agent Readiness Score** with named fixes, applied and re-measured
- A **CI test** that fails when the API changes underneath the tools

## Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `UPSTREAM_AUTH_MISSING: MCPCAST_UPSTREAM_TOKEN is not set` | `env-token` server with no credential in its environment | `export MCPCAST_UPSTREAM_TOKEN='Bearer …'` — for desktop clients, set it in the client's `env` block, not your shell |
| `UPSTREAM_AUTH_MISSING` on a `passthrough` server | The MCP client sent no `Authorization` header — stdio clients cannot | Regenerate with `--auth env-token` (personal) or `--auth api-key` (multi-tenant) |
| `APPROVAL_DENIED … no live MCP session for elicitation … denied fail-closed` | A gated tool was called by a client that cannot show an elicitation prompt (or by a test) | Expected behaviour. Use a client that supports elicitation, switch to `--approval pending` with `--auth api-key`, or inject an `approval_handler` in tests |
| `VALIDATION_ERROR: Invalid value for path parameter 'order_id': ''` | A path value that is empty, `.` or `..` — it would change which URL is called | Pass a real single path segment; the guard is deliberate and cannot be disabled |
| `UPSTREAM_UNREACHABLE` | The base URL is wrong, or the API is down | Check `api.base_url` in the plan, or override at runtime with `MCPCAST_BASE_URL` |
| `northwind-standard already contains an mcpcast project — pass --force to overwrite it, or regenerate from …/mcpcast.plan.yaml to keep your edits` | Re-running a *spec* into a directory that already holds a project would discard your plan edits | Regenerate from the plan (`promptise mcpcast …/mcpcast.plan.yaml`), or `--force` if you meant to start over |
| `spec has no 'paths' — is this an OpenAPI document?` | The file loaded but is not an OpenAPI 3.x / Swagger 2 document (a config file, an AsyncAPI spec, an HTML error page from a URL) | Point at the actual spec document |
| `base_url must start with http:// or https:// (got ''); pass --base-url if the spec does not declare a server` | The spec has no `servers:` entry, and mcpcast will not guess | `--base-url https://api.yourcompany.com` |
| `approval mode 'pending' … needs identified callers so a caller can never approve their own request` | `--approval pending` without `--auth api-key` | Add `--auth api-key`, or use `--approval elicitation` |
| `model 'foundry:…' could not be used for curation: ModelSetupError: Cannot use model … yet: - AZURE_INFERENCE_ENDPOINT is not set — …` | The `--model` provider's env vars are missing (the message lists each one, where to find it, and the `.env` / export / `Model(...)` ways to supply it) | Follow the listed fixes, or check any string first with `promptise models check <string>`; `--no-curate` needs no model at all |
| `curation failed after 3 attempt(s): curation proposal rejected: …` | The model's design broke a post-condition three times running — usually merges forced by a tight `--max-tools` | Raise the budget, or fall back to `--no-curate`. The message names the exact violation |
| `the curation prompt for N operations is … characters (limit 600,000); narrow the spec first` | A very large spec | Curate one tag or path prefix at a time, or run `--no-curate` with `--max-tools` |
| `--profile cannot be combined with a plan file — edit mcpcast.plan.yaml and regenerate instead` | Spec-only flags passed alongside a plan file | Change the value in the plan and regenerate |
| `auth mode 'none' refuses to bind to a non-loopback address` | `--auth none` served on `0.0.0.0` | Regenerate with a real auth mode; `none` is for local demos |

## Next Steps

- [MCPcast, end to end](../mcpcast/index.md) — how MCP works, the full real run, and the review checklist

- **[Make Your Python API MCP-Ready](mcpcast-python-app.md)** — you have an
  app rather than a spec: the OpenAPI URL per framework, a spec written for
  a model, generation from the running app, and shipping with a CI guard
- **[Lab: MCPcast a SaaS API](lab-mcpcast-storefront.md)** — the same journey as a
  copy-paste lab, with an agent completing a real business task and a refund
  held at the gate
- **[MCPcast reference](../mcp/server/mcpcast.md)** — the full risk table,
  curation contract, plan schema and readiness metrics
- **[Approval Gates](../mcp/server/approval-gates.md)** — elicitation, pending
  stores, webhooks and custom approval handlers
- **[Auth & Security](../mcp/server/auth-security.md)** and
  **[Multi-Tenancy](../mcp/server/multi-tenancy.md)** — what to layer on top of
  a generated server before it faces customers
- **[Testing](../mcp/server/testing.md)** — `TestClient` against the full
  pipeline
- **[CLI reference](../core/cli.md)** — every option and exit
  code
- **[API reference](../api/mcpcast.md)** — `mcpcast()`, `curate()`,
  `write_project()`, `evaluate()` and the plan model
- **[Examples](../resources/examples.md)** — `examples/mcp/mcpcast_petstore/`
  runs this end to end against the live Swagger Petstore
- **[Building Production MCP Servers](production-mcp-servers.md)** — when the
  generated server needs hand-written tools alongside the generated ones
