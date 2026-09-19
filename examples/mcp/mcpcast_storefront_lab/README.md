# Lab: MCPcast the Storefront API

You already have an API. Your customers want to use their AI with your product — Claude
Desktop, Claude Code, Cursor, whatever they run. This lab turns an OpenAPI spec into an MCP
server they can point at, with the dangerous half of the API behind a human approval gate,
and then **measures** how well a real agent can actually drive it.

```
storefront.yaml ──▶ parse ──▶ classify (risk) ──▶ plan ──▶ emit ──▶ generated/v1/
                                                                   ├── mcpcast.plan.yaml   # the source of truth
                                                                   ├── server.py           # launcher
                                                                   ├── storefront_mcp/     # the server as a package
                                                                   ├── tests/              # its own pytest suite
                                                                   ├── pyproject.toml  Dockerfile  .env.example
                                                                   └── README.md
                                                            ──▶ eval ──▶ Agent Readiness: A
```

## What it shows

| # | Step | What you see |
|---|------|--------------|
| 1 | **Generate** | 13 operations → 11 tools under the `full` profile, 5 of them approval-gated. Every excluded operation is listed with a reason, and the classifier prints *why* each tool got its risk class. The same spec is shown under all three safety profiles. |
| 2 | **Drive** | A real `build_agent("openai:gpt-5-mini")` answers *"who is ada@northwind.example and what are their last orders?"* through the generated server, in-process. You see which tools it chose and every HTTP request that reached the API. |
| 3 | **Govern** | The agent tries to refund an order. The server-side gate holds the call — **zero** refund requests reach the API. A reviewer from another tenant sees nothing and cannot decide; the caller cannot approve their own request even holding the `approver` role; a second human approves and the refund goes through, carrying that tenant's upstream credential. |
| 4 | **Measure** | The Agent Readiness Score: 8 tasks, a real agent, the full server pipeline, spec-derived mocks. A grade, a per-task table, and specific tool-design fixes. |
| 5 | **Improve** | The fixes are applied to the plan *in code* — two customer lookups merged into one `find_customer` tool with two routes, the missing examples added, the health check dropped, `notify_customer` put on a param diet — the project is regenerated and the same eight tasks are re-scored. The two grades are printed side by side, whichever way it goes. |

## Files

| File | What it is |
|---|---|
| `storefront.yaml` | The company's existing OpenAPI 3 spec: customers, orders, refunds, subscriptions, an admin endpoint, a deprecated CSV export and a multipart upload |
| `fake_api.py` | The upstream API as an in-process `httpx.MockTransport` — a dict-backed store that records every request, so the lab can prove what did and did not reach it |
| `run.py` | The lab, in five printed sections |
| `generated/` | Written by `run.py` (git-ignored): `v1/` and `v2/`, each a complete MCP server project plus its `eval/` report |

`fake_api.py` fakes only the **upstream HTTP API**, the way you would fake Stripe in your own
test suite. The agents, the tools, the approval gate and the scores are real.

## Run

The only requirement is an OpenAI API key. Step 1 runs offline; the script stops with a clear
message before step 2 if the key is missing.

```bash
export OPENAI_API_KEY=sk-...
.venv/bin/python examples/mcp/mcpcast_storefront_lab/run.py
```

It runs 18 agent tasks, each one or more model calls (two agent tasks plus 8 evaluation tasks per plan version) —
roughly three minutes on `gpt-5-mini`.

## Sample output

```text
1. Generate the MCP server from the OpenAPI spec
==============================================================================
  wrote generated/v1/mcpcast.plan.yaml
  wrote generated/v1/server.py
  wrote generated/v1/storefront_mcp/__init__.py
  …  (the package, tests/, README.md, pyproject.toml, Dockerfile, .env.example, .gitignore)

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
```

The full run — including the approval gate holding a refund and the two readiness grades — is
in the [lab write-up](../../../docs/guides/lab-mcpcast-storefront.md).

## Point an AI client at the generated server

`generated/v1/server.py` depends only on `promptise` and `httpx`, and is yours to edit (or to
regenerate from `mcpcast.plan.yaml` at any time). For a personal server that a desktop client
launches over stdio, generate it with `--auth env-token` so it carries **your** API token:

```bash
promptise mcpcast examples/mcp/mcpcast_storefront_lab/storefront.yaml \
  --profile standard --auth env-token --out storefront-mcp
```

**Claude Desktop** — `claude_desktop_config.json`:

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

**Claude Code** — the same command and the same token, passed with `-e` so it lands in
the server's environment (the client launches the server itself and does not see your
shell's exports):

```bash
claude mcp add storefront -e MCPCAST_UPSTREAM_TOKEN="Bearer <your Storefront API token>" \
  -- /absolute/path/to/.venv/bin/python /absolute/path/to/storefront-mcp/server.py
```

**Cursor** — `.cursor/mcp.json` takes the same `command`/`args`/`env` shape as Claude Desktop.

Every write, refund and cancellation still asks the human behind that client to confirm before
it runs — the gate lives in the server, not in the client.

## See also

- [MCPcast an Existing API](../../../docs/mcp/server/mcpcast.md) — the reference guide
- [Lab write-up](../../../docs/guides/lab-mcpcast-storefront.md) — this lab, step by step
- [`examples/mcp/mcpcast_petstore/`](../mcpcast_petstore/) — the same pipeline against a live public API
