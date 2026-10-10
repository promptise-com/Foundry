# Lab: the guided setup — turn a running API into an MCP server, then prove it

You drive the wizard; the lab proves what it wrote. `app.py` is a small, realistic Helpdesk
API (customers, tickets, a refund endpoint, an admin corner, one deprecated route) standing
in for yours. `run.py` starts it, opens `promptise mcpcast` in your terminal, and — once you
have written the project — points a real agent at it, shows the safety profile and the
approval gate doing their job against the live app, and scores the result.

```
app.py ──uvicorn──▶ http://127.0.0.1:<port>/openapi.json
                              │
                    promptise mcpcast  (the guided setup — you, at the keyboard)
                              │
                              ▼
                    generated/helpdesk-mcp/
                    ├── mcpcast.plan.yaml   # the editable source of truth
                    ├── server.py           # launcher: runs the package without installing it
                    ├── helpdesk_mcp/       # the server as a package: config, upstream, approval, tools/
                    ├── tests/              # its own pytest suite
                    ├── pyproject.toml  Dockerfile  .env.example  .gitignore
                    ├── README.md           # Claude / Cursor / Claude Code snippets
                    └── eval/               # the Agent Readiness report
                              │
   build_agent("openai:gpt-5-mini") ──MCP stdio──▶ server.py ──HTTP──▶ app.py
```

## What it shows

| # | Step | What you see |
|---|------|--------------|
| 1 | **Start** | The Helpdesk API on a local port the wizard's detection probes. 12 operations: 5 read, 4 write, 2 destructive (`DELETE`, admin purge), 1 financial (`refund`), one deprecated. |
| 2 | **Wizard** | Step 1 finds the running API. Step 3 shows the exact counts for *this* spec: read-only 5 tools · standard 8 (3 approval-gated) · full 11 (6 gated). Step 6 is the review workspace. |
| 3 | **Inspect** | The plan as a reviewer reads it, plus the *computed* warnings: descriptions that name `refund_ticket` or `delete_ticket` — tools the `standard` profile never generated — and parameters hidden from the agent. |
| 4 | **Drive** | A real `build_agent("openai:gpt-5-mini")` launches `generated/helpdesk-mcp/server.py` over MCP stdio, exactly as Claude Desktop would, and answers *"Which open tickets does customer cus_ada have?"* from the live app. |
| 5 | **Govern** | The agent tries to close a ticket: `APPROVAL_DENIED`, ticket still `pending`. It tries to refund: under `standard` there is no refund tool at all — the live app shows 0.00 EUR refunded. |
| 6 | **Measure** | The Agent Readiness Score: 8 generated tasks, a real agent, the full server pipeline, a grade and the fixes to make. |
| 7 | **Next** | The exact non-interactive command, how to regenerate and re-measure after editing the plan (the re-measure line carries `MCPCAST_EVAL_AUTHORIZATION`, the token the live reads need — without it the grade measures the missing credential), and the `claude mcp add` line. |

## Files

| File | What it is |
|---|---|
| `app.py` | The Helpdesk API — an ordinary FastAPI app behind a bearer token (`demo-token`). Its `summary=`, docstrings, `operation_id=` and `Field(description=...)` become the tool descriptions, names and parameter descriptions. |
| `run.py` | The lab, in seven printed sections |
| `generated/` | Written by the wizard (git-ignored): `helpdesk-mcp/` — the plan, the `helpdesk_mcp/` package, the `server.py` launcher, `tests/`, `pyproject.toml`, `Dockerfile`, `.env.example`, the README and the `eval/` report |

## Run

The app is *your* app, so its web framework is its own: `fastapi` is not a Promptise
dependency (`uvicorn` ships with Promptise; the `dev` extra includes `fastapi` for these
examples). The agent steps need `OPENAI_API_KEY` in `.env` or exported — the wizard's step 2
shows whether it was picked up.

```bash
.venv/bin/python -m pip install fastapi
.venv/bin/python examples/mcp/mcpcast_wizard_lab/run.py
```

`--no-eval` skips step 6. A second run finds `generated/helpdesk-mcp/` in place: switch
on *overwrite* in step 5, or regenerate from the plan instead of re-running the wizard.

## What to choose

The lab prints this card before the wizard opens:

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

If another API is running on one of the probed ports it is listed too — pick the Helpdesk
entry with `↓`.

## What to look for in step 6

This is the part the wizard cannot do for you. In the run below `openai:gpt-5-mini`:

- merged `get_ticket`, `search_tickets` and `list_tickets` into one `find_ticket` — a
  good call, three routes behind one tool;
- wrote *"use create_ticket, update_ticket, delete_ticket or refund_ticket instead"* into
  three descriptions — but `delete_ticket` and `refund_ticket` were excluded by the
  `standard` profile. An agent reading that will look for tools that do not exist;
- hid `priority` on `create_ticket` and `note` on `update_ticket` — sent as fixed defaults,
  so a user can never set them through the agent.

The lab's *computed review warnings* flag all of these after the wizard exits (the wizard
shows the first three under the review summary). They are the first lines to fix in
`generated/helpdesk-mcp/mcpcast.plan.yaml`; then `promptise mcpcast
generated/helpdesk-mcp/mcpcast.plan.yaml` regenerates the server without re-curating.

## Sample output

The sections after the wizard, from a real run (`standard`, model curation):

```text
  Next time, without the wizard:
    promptise mcpcast http://127.0.0.1:8001/openapi.json --profile standard --auth env-token --out generated/helpdesk-mcp

==============================================================================
3. What the wizard wrote — read it like a reviewer
==============================================================================
  wrote generated/helpdesk-mcp/mcpcast.plan.yaml
  wrote generated/helpdesk-mcp/server.py
  wrote generated/helpdesk-mcp/helpdesk_mcp/__init__.py
  wrote generated/helpdesk-mcp/helpdesk_mcp/__main__.py
  wrote generated/helpdesk-mcp/helpdesk_mcp/config.py
  wrote generated/helpdesk-mcp/helpdesk_mcp/upstream.py
  wrote generated/helpdesk-mcp/helpdesk_mcp/approval.py
  wrote generated/helpdesk-mcp/helpdesk_mcp/server.py
  wrote generated/helpdesk-mcp/helpdesk_mcp/tools/__init__.py
  wrote generated/helpdesk-mcp/helpdesk_mcp/tools/tickets.py
  wrote generated/helpdesk-mcp/helpdesk_mcp/tools/customers.py
  wrote generated/helpdesk-mcp/tests/conftest.py
  wrote generated/helpdesk-mcp/tests/test_tools.py
  wrote generated/helpdesk-mcp/README.md
  wrote generated/helpdesk-mcp/pyproject.toml
  wrote generated/helpdesk-mcp/Dockerfile
  wrote generated/helpdesk-mcp/.env.example
  wrote generated/helpdesk-mcp/.gitignore

  profile standard · auth env-token · base_url http://127.0.0.1:8001

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
    -> fix in generated/helpdesk-mcp/mcpcast.plan.yaml, then regenerate

==============================================================================
4. A real agent over MCP stdio -> generated server.py -> the live app
==============================================================================
  question: Which open tickets does customer cus_ada (Ada Lovelace) have, and what is each about?
  tools the agent chose:
    find_ticket({"customer_id": "cus_ada", "status": "open"})
  answer: Ada (cus_ada) has two open tickets:

- #5 — "Cannot add a team member" (opened 2026-03-15, priority: normal, unassigned)
  - Issue: Inviting a colleague fails with "seat limit reached" even though there are 3 seats available.

- #1 — "Invoice charged twice in March" (opened 2026-03-04, priority: high, assignee: sam)
  - Issue: Card shows two charges of €29.90 for March; billing confirmed a duplicate charge.

==============================================================================
5. Writes and money: what the safety profile and the gate do
==============================================================================
  profile standard: write tools ['create_ticket', 'update_ticket', 'close_ticket'] · financial tools none

  a) Close ticket 2 with the resolution 'export fixed in release 4.3'.
     (a write tool exists; over stdio no human can be asked, so the gate denies)
     tools the agent chose:
    close_ticket({"ticket_id": 2, "resolution": "export fixed in release 4.3"})
      -> APPROVAL_DENIED: Approval denied for tool 'close_ticket': client declined or returned an invalid elicitation response
     answer: I couldn't close ticket #2 — the system denied the close_ticket request (approval denied). ...
     live app: ticket 2 is still 'pending' — nothing reached the API

  b) Refund 29.90 EUR on ticket 1; the customer was charged twice.
     (profile standard: no refund tool exists; the agent cannot even try)
     tools the agent chose:
     answer: I can’t issue refunds — there’s no refund function available in these tools. ...
     live app: ticket 1 refunded total is 0.00 EUR — nothing reached the API

==============================================================================
6. Measure: the Agent Readiness Score
==============================================================================
  8 generated tasks, a real agent (openai:gpt-5-mini), the full server pipeline …
  wrote generated/helpdesk-mcp/eval/report.md and generated/helpdesk-mcp/eval/tasks.yaml

  Agent Readiness: B  (6/8 tasks succeeded)
    ✗ `find_customer`, `find_ticket` called the API with identifiers it does not recognise — the example in the plan teaches both the task writer and the agent, so replace those example values with ones that exist
```

Two of eight tasks failed because the plan's examples use invented identifiers (`cust_123`,
a ticket number that does not exist) — the task writer copied them, the agent used them, the API said 404. Replace
the examples with ids that exist (`cus_ada`, ticket `1`), regenerate, re-measure.

## Why the wizard's evaluation switch has a warning

The Agent Readiness evaluation runs the *read* tools against your live API. In `env-token`
mode that credential comes from `MCPCAST_EVAL_AUTHORIZATION`; without it the reads carry a
placeholder token and every read task fails against an API that needs auth — which is a
statement about the credential, not the tools. The lab sets the variable before the wizard
opens, so both the wizard's own switch and step 6 evaluate for real.

## Notes

- Everything the wizard collects is a `promptise mcpcast` flag; the command it prints at
  the end reproduces the run without it. Pass `--interactive` with a spec to open it
  pre-filled.
- `generated/helpdesk-mcp/server.py` depends only on `promptise` and `httpx`; keep it next
  to your app, edit `mcpcast.plan.yaml` (never `server.py`), regenerate.
- The API here has no way to look a customer up by name — an agent asked about "Ada
  Lovelace" without her id has nothing to call. The eval cannot see that; your reviewers
  can. Tool surfaces expose gaps in the API underneath them.
