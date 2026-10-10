# MCPcast the Swagger Petstore

Point `promptise mcpcast` at a real, public third-party API spec and get an MCP server that
any AI can drive — with the destructive operations behind a human approval gate.

```
petstore.yaml ──▶ parse ──▶ classify (risk) ──▶ plan ──▶ emit ──▶ generated/
                                                                 ├── mcpcast.plan.yaml   # the source of truth
                                                                 ├── server.py           # launcher
                                                                 ├── petstore_mcp/       # the server as a package
                                                                 ├── tests/              # its own pytest suite
                                                                 ├── pyproject.toml  Dockerfile  .env.example
                                                                 └── README.md
```

`run.py` does it with the Python API (`promptise.mcpcast`) and then proves the result works:

1. **Generate** `generated/` from `petstore.yaml` under the `full` profile with `--auth none`:
   all 11 operations become tools, the 6 that write or delete are `requires_approval=True`,
   and the plan records every operation it leaves out with a reason.
2. **Real agent, real transport, real API** — `build_agent("openai:gpt-5-mini")` launches
   `generated/server.py` over MCP stdio and answers *"Which pets are currently available?"*
   by calling the live `petstore3.swagger.io` API through `find_pets_by_status`.
3. **Approval gate** — in-process with `TestClient`, `delete_pet` is **denied** when no
   approver can be reached (fail-closed), then runs once an approver says yes. The upstream is
   an `httpx.MockTransport`, so nothing is deleted anywhere.

## Files

| File | What it is |
|---|---|
| `petstore.yaml` | Trimmed copy of the public Petstore v3 spec: pet + store operations, real schemas |
| `run.py` | The runnable demo (about 160 lines) |
| `generated/` | Created by `run.py`, git-ignored — regenerate it any time |

## Run

Only `OPENAI_API_KEY` is needed. Step 1 runs offline; the script stops with a clear message
before step 2 if the key is missing.

```bash
OPENAI_API_KEY=... .venv/bin/python examples/mcp/mcpcast_petstore/run.py
```

The CLI equivalent of step 1 (add `--eval` to score it, drop `--no-curate` to let the model
design the tool surface):

```bash
promptise mcpcast examples/mcp/mcpcast_petstore/petstore.yaml --no-curate \
  --profile full --auth none --name petstore --out examples/mcp/mcpcast_petstore/generated
```

## What to expect

```
=== 1. Generate the MCP server project (profile=full, auth=none) ===
  wrote generated/mcpcast.plan.yaml
  wrote generated/server.py
  wrote generated/petstore_mcp/__init__.py
  wrote generated/petstore_mcp/__main__.py
  wrote generated/petstore_mcp/config.py
  wrote generated/petstore_mcp/upstream.py
  wrote generated/petstore_mcp/approval.py
  wrote generated/petstore_mcp/server.py
  wrote generated/petstore_mcp/tools/__init__.py
  wrote generated/petstore_mcp/tools/pet.py
  wrote generated/petstore_mcp/tools/store.py
  wrote generated/tests/conftest.py
  wrote generated/tests/test_tools.py
  wrote generated/README.md
  wrote generated/pyproject.toml
  wrote generated/Dockerfile
  wrote generated/.env.example
  wrote generated/.gitignore

  tool                    risk         approval  upstream operation
  add_pet                 write        required  POST /pet
  update_pet              write        required  PUT /pet
  find_pets_by_status     read         -         GET /pet/findByStatus
  find_pets_by_tags       read         -         GET /pet/findByTags
  get_pet_by_id           read         -         GET /pet/{petId}
  update_pet_with_form    write        required  POST /pet/{petId}
  delete_pet              destructive  required  DELETE /pet/{petId}
  get_inventory           read         -         GET /store/inventory
  place_order             write        required  POST /store/order
  get_order_by_id         read         -         GET /store/order/{orderId}
  delete_order            destructive  required  DELETE /store/order/{orderId}

  not exposed:
  (none — profile 'full' exposes every operation and gates every non-read)

  default read-only profile: 5 tools, 6 not exposed (e.g. addPet: write operation excluded by profile 'read-only')

=== 2. Real agent over MCP stdio -> live petstore API ===
  question: Which pets are currently available? List up to five names.
  answer:   Five available pets: doggie, Puppy, Max, Bella, Charlie

=== 3. Human approval gate on delete_pet (in-process, mock upstream) ===
  no approver reachable -> APPROVAL_DENIED: Approval denied for tool 'delete_pet': no live MCP session for elicitation (client or transport does not support it) — denied fail-closed
  DELETE requests that reached upstream: 0
  approver says yes     -> {"message": "Pet deleted"}
  upstream received     -> DELETE https://petstore3.swagger.io/api/v3/pet/10
  (upstream was an httpx.MockTransport — no real pet was deleted)

=== 4. Next steps ===
  edit generated/mcpcast.plan.yaml, then regenerate and score it with a real agent:
    promptise mcpcast generated/mcpcast.plan.yaml --out generated --eval
  connect Claude Desktop / Claude Code / Cursor: see generated/README.md
```

The pet names in step 2 are illustrative — the Petstore demo data is shared and changes
constantly. `petstore3.swagger.io` is also a public demo that is sometimes down; the generated
tool then returns a structured `UPSTREAM_ERROR` (HTTP 500, `retryable: true`) and the agent says
so instead of inventing pets. `run.py` prints the failure and continues with step 3, which
needs no network.

## Try the generated server in Claude Desktop or Claude Code

`generated/README.md` carries the install snippets for every client. Use the interpreter that
has `promptise` installed (this repo's `.venv/bin/python`) and the absolute path to `server.py`.

**Claude Desktop** — add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "petstore": {
      "command": "/absolute/path/to/AgentMCP/.venv/bin/python",
      "args": ["/absolute/path/to/AgentMCP/examples/mcp/mcpcast_petstore/generated/server.py"]
    }
  }
}
```

**Claude Code:**

```bash
claude mcp add petstore -- /absolute/path/to/AgentMCP/.venv/bin/python \
  /absolute/path/to/AgentMCP/examples/mcp/mcpcast_petstore/generated/server.py
```

Ask *"how many pets are sold?"* and the AI calls `get_inventory`. Ask it to delete a pet and
the server asks **you** to approve first (MCP elicitation) — a client that cannot elicit gets
`APPROVAL_DENIED`, never a silent delete.

## Notes

- `--auth none` is for local development: the server sends no credentials upstream and refuses
  to bind to a non-loopback address. For a real API that needs credentials, regenerate with
  `--auth env-token` and export `MCPCAST_UPSTREAM_TOKEN` (the standard setup for a personal
  server over stdio), or use `--auth passthrough` / `--auth api-key` for a shared HTTP/SSE
  deployment — those two read a request header (`Authorization` / `x-api-key`) that stdio
  clients cannot send.
- `petstore.yaml` drops the upstream spec's OAuth requirements on purpose: the demo API accepts
  unauthenticated calls, so the spec should not advertise credentials it never needs. Keeping
  them would not change the tool surface: the classifier only escalates on `admin`/`root`/`superuser`
  scopes, so the upstream spec's `write:pets` scope on its GET operations leaves them `read`.
- `generated/server.py` is regenerated from `generated/mcpcast.plan.yaml` — edit the plan, not
  the server. It depends only on `promptise` and `httpx`.
