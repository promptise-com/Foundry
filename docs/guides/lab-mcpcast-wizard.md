---
title: Lab — Turn a Running API into an MCP Server with the Guided Setup, Then Prove It Works
description: A hands-on lab for promptise mcpcast's terminal wizard. Start a real Helpdesk API, walk through the seven steps, review what the model decided (it names tools that do not exist), then watch a real agent drive the generated server, the approval gate deny a write, and the Agent Readiness Score tell you what to fix.
keywords: promptise mcpcast wizard lab, MCP server from FastAPI, guided setup MCP, review generated MCP tools, agent readiness score, approval gate MCP
---

# Lab: The Guided Setup

You have an API. You want to know, in twenty minutes, what it takes to hand it
to an AI assistant safely — and what the tool surface a model designs for it
actually looks like when you read it.

This lab is that twenty minutes. A small, realistic **Helpdesk API** stands in
for yours. You drive the wizard; the lab proves the result with a real agent
against the live app, shows the safety profile and the approval gate doing
their job, and scores the server.

Everything is in
[`examples/mcp/mcpcast_wizard_lab/`](https://github.com/promptise-com/foundry/tree/main/examples/mcp/mcpcast_wizard_lab).

## Prerequisites

```bash
pip install promptise fastapi              # fastapi is the sample app's own dependency
```

`OPENAI_API_KEY` in a `.env` file in the directory you run from, or exported —
[Configuration & Secrets](../getting-started/configuration.md) explains where
keys live. The wizard's step 2 shows whether it was found; without it, choose
*Offline* in step 2 and the lab stops before the agent steps with the reason.

## The API you already have

`app.py` is an ordinary FastAPI app behind a bearer token (`demo-token`).
Twelve operations, chosen so every risk class shows up:

| Operation | Method | What the classifier says | Why |
|---|---|---|---|
| `list_tickets`, `get_ticket`, `get_customer`, `health_check` | `GET` | `read` | reads |
| `search_tickets` | `POST /tickets/search` | `read` | the verb *search* wins over the method |
| `create_ticket`, `update_ticket`, `close_ticket` | `POST` / `PATCH` | `write` | changes state |
| `refund_ticket` | `POST /tickets/{id}/refund` | `financial` | money leaves the company |
| `delete_ticket` | `DELETE` | `destructive` | no undo |
| `purge_closed_tickets` | `POST /admin/purge-closed` | `destructive` | *purge*, under `/admin` |
| `export_tickets_csv` | `GET` | dropped | `deprecated: true` |

Nothing in the file knows about MCP. Its `summary=`, docstrings, `operation_id=`
and `Field(description=...)` are what become tool descriptions, tool names and
parameter descriptions — which is why the review step matters.

## Run it

```bash
.venv/bin/python examples/mcp/mcpcast_wizard_lab/run.py
```

The lab starts the app on a port the wizard probes and prints the card below.
Press Enter and the wizard opens in the same terminal.

```text
1 API spec   Enter on the detected "Helpdesk API" (http://127.0.0.1:8001/openapi.json)
             — 12 operations: 5 read · 4 write · 2 destructive · 1 financial
2 Model      "Design the tools with a model" if the key line is green,
             otherwise Offline (everything after still works)
3 Safety     standard  — reads open, writes approval-gated, refund/delete
             not generated. Compare the counts on the three rows.
4 Auth       Personal — a desktop client  (--auth env-token)
5 Project    keep the defaults (name helpdesk, folder generated/helpdesk-mcp)
6 Review     read every row. Look for a description that names refund_ticket
             or delete_ticket — tools this profile did not generate — and for
             hidden parameters. Fix them later in mcpcast.plan.yaml.
7 Write      Finish (Enter). The lab continues from here.
```

## Step 1 – 5 — the choices

Three things to notice on the way through:

- **Step 1** lists what it found on the usual local ports. If the
  [end-to-end lab's](lab-mcpcast-storefront.md) bookshelf app or your own
  service is also running, it is listed too — `↓` to the Helpdesk entry.
- **Step 3** is not a generic menu. The wizard ran the deterministic planner
  over *this* spec for each profile: `read-only — 5 tools · reads only`,
  `standard — 8 tools · 3 require human approval`, `full — 11 tools · 6 require
  human approval`. Pick `standard`.
- **Step 5** shows the folder pre-filled by the lab (`generated/helpdesk-mcp`)
  and, if you switch the evaluation on, whether `MCPCAST_EVAL_AUTHORIZATION` is
  set — the lab sets it, so the evaluation's live reads reach the app
  authenticated.

## Step 6 — read what the model decided

Curation runs (a real `openai:gpt-5-mini` call, about half a minute). Then the
review workspace. In the run this page was written from, the model:

- merged `get_ticket`, `search_tickets` and `list_tickets` into one
  `find_ticket` — a good call: one tool, three routes, the most specific first;
- wrote *"use create_ticket, update_ticket, delete_ticket or refund_ticket
  instead"* into three descriptions — but `delete_ticket` and `refund_ticket`
  were **excluded by the `standard` profile**. An agent that reads this will look
  for tools that do not exist;
- hid `priority` on `create_ticket` and `note` on `update_ticket`, sending fixed
  defaults instead — so no user can set them through the agent.

The review summary line shows the first of these as ⚠ warnings — they are
computed from the plan, not guessed — and the checklist on the right is the
rest of the job. Press *Write project*.

!!! danger "Never trust it blindly"
    Every one of the three findings above is a one-line fix in
    `mcpcast.plan.yaml` — after a human read it. The wizard can tell you that a
    description names a tool that is not in the plan; it cannot tell you
    whether your users need to set `priority`. Read every row.

## After the wizard — what the lab proves

### 3. What was written, as a reviewer reads it

Eighteen files — a project, not a script: `mcpcast.plan.yaml`, the
`helpdesk_mcp/` package (config, HTTP client, approval gate, `build_server()`,
`tools/tickets.py`, `tools/customers.py`), the `server.py` launcher, `tests/`,
`README.md`, `pyproject.toml`, `Dockerfile`, `.env.example`, `.gitignore`.
Then the plan, the way a reviewer reads it:

```text
  tool                    risk         approval  upstream operation(s)
  find_ticket             read         -         GET /tickets/{ticket_id}, POST /tickets/search, GET /tickets
  create_ticket           write        required  POST /tickets
  update_ticket           write        required  PATCH /tickets/{ticket_id}
  close_ticket            write        required  POST /tickets/{ticket_id}/close
  find_customer           read         -         GET /customers/{customer_id}

  not exposed (each with its reason, recorded in the plan):
    health_check: Liveness probe for the load balancer; not useful to an end-user agent.
    export_tickets_csv: Deprecated legacy CSV export; documentation says use the reporting API instead.
    purge_closed_tickets: Admin-only destructive maintenance operation (purges all closed tickets); not appropriate for day-to-day agent use.
    delete_ticket: destructive operation excluded by profile 'standard'
    refund_ticket: financial operation excluded by profile 'standard'

  computed review warnings:
    ! find_ticket: the description names delete_ticket, refund_ticket — not exposed by this plan, so the agent will look for a tool that does not exist
    ! find_ticket: hides limit (sent as defaults) — fine unless your users need to set them
    ! create_ticket: hides priority (sent as defaults) — fine unless your users need to set them
    ! update_ticket: the description names delete_ticket — not exposed by this plan, so the agent will look for a tool that does not exist
    ! update_ticket: hides note (sent as defaults) — fine unless your users need to set them
    ! close_ticket: the description names refund_ticket — not exposed by this plan, so the agent will look for a tool that does not exist
```

The same `review_warnings()` the wizard uses, over the whole plan. Every line
is a place to edit.

### 4. A real agent drives it

`build_agent("openai:gpt-5-mini")` launches `generated/helpdesk-mcp/server.py`
over MCP stdio — the transport Claude Desktop uses — with the upstream token
in the server's environment, and asks a question the live app answers:

```text
  question: Which open tickets does customer cus_ada (Ada Lovelace) have, and what is each about?
  tools the agent chose:
    find_ticket({"customer_id": "cus_ada", "status": "open"})
  answer: Ada (cus_ada) has two open tickets:
- #5 — "Cannot add a team member" (opened 2026-03-15, priority: normal, unassigned) ...
- #1 — "Invoice charged twice in March" (opened 2026-03-04, priority: high, assignee: sam) ...
```

### 5. The profile and the gate

```text
  a) Close ticket 2 with the resolution 'export fixed in release 4.3'.
     tools the agent chose:
    close_ticket({"ticket_id": 2, "resolution": "export fixed in release 4.3"})
      -> APPROVAL_DENIED: Approval denied for tool 'close_ticket': client declined or returned an invalid elicitation response
     live app: ticket 2 is still 'pending' — nothing reached the API

  b) Refund 29.90 EUR on ticket 1; the customer was charged twice.
     (profile standard: no refund tool exists; the agent cannot even try)
     live app: ticket 1 refunded total is 0.00 EUR — nothing reached the API
```

Over stdio there is no human to ask, so the server-side gate denies the write
fail-closed — the app's own state confirms nothing changed. The refund is not
denied; it does not exist: `standard` never generated it. Run the wizard again
with `full` to see `refund_ticket` generated *and* gated.

### 6. Measure

```text
  Agent Readiness: B  (6/8 tasks succeeded)
    ✗ `find_customer`, `find_ticket` called the API with identifiers it does not recognise — the example in the plan teaches both the task writer and the agent, so replace those example values with ones that exist
```

Two tasks failed for one reason: the model's examples use invented identifiers
(`cust_123`, a ticket number that does not exist); the task writer copied them, the agent used them,
the API said 404. Replace them with ids that exist (`cus_ada`, `1`),
regenerate, re-measure.

## What you've built

- A reviewed, editable MCP server for an API you started ten minutes ago —
  `mcpcast.plan.yaml` is yours, `server.py` is derived from it.
- Evidence, not hope: a real agent answered from the live app; a write was
  denied by the server, not by a prompt; a refund tool did not exist because
  the profile said so; the evaluation named the exact examples to fix.
- The non-interactive command for next time, printed by the wizard:

```text
promptise mcpcast http://127.0.0.1:8001/openapi.json --profile standard --auth env-token --out generated/helpdesk-mcp
```

## Next steps

- Fix the six warnings in `generated/helpdesk-mcp/mcpcast.plan.yaml`, then
  re-measure with the app running and the API's token for the live reads —
  `MCPCAST_EVAL_AUTHORIZATION='Bearer demo-token' promptise mcpcast generated/helpdesk-mcp/mcpcast.plan.yaml --eval`.
  `--eval` writes a fresh task set from the fixed plan and a new grade; without
  the variable every read task fails on the missing credential, not the tools.
- Add it to Claude Code with the app running:
  `claude mcp add helpdesk -e MCPCAST_UPSTREAM_TOKEN='Bearer demo-token' -- python generated/helpdesk-mcp/server.py`
- Point the wizard at your own API: `promptise mcpcast`, with it running.
- [The guided setup](../mcpcast/guided-setup.md) — every step, with screenshots.
- [MCPcast, end to end](../mcpcast/index.md) — how MCP works, the full review
  checklist, shipping.
