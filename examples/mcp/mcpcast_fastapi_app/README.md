# Make your own FastAPI app MCP-ready

You have an app, not a spec file. This example is the exact sequence for turning *your*
Python API into an MCP server that Claude Desktop, Claude Code, Cursor and any other AI can
use — with a real FastAPI app standing in for yours.

```
app.py  ──uvicorn──▶  http://127.0.0.1:<port>/openapi.json
                              │
                       promptise.mcpcast  (profile standard, auth env-token)
                              │
                              ▼
                       generated/
                       ├── mcpcast.plan.yaml   # the editable source of truth
                       ├── server.py           # launcher: runs the package without installing it
                       ├── bookshelf_mcp/      # the server as a package: config, upstream, approval, tools/
                       ├── tests/              # its own pytest suite
                       ├── pyproject.toml  Dockerfile  .env.example  .gitignore
                       └── README.md           # Claude / Cursor / Claude Code snippets
                              │
        build_agent("openai:gpt-5-mini") ──MCP stdio──▶ server.py ──HTTP──▶ app.py
```

`run.py` does it end to end, with real calls everywhere:

1. **Start** `app.py` — the Bookshelf API: list / get / search (a `POST`) / create / update /
   delete, one `/admin` endpoint, a health check and a deprecated route, behind a bearer token
   (`demo-token`). Its `summary=`, docstrings, `operation_id=` and `Field(description=...)`
   are what become tool descriptions, tool names and parameter descriptions.
2. **Generate** `generated/` from the *running* app's `/openapi.json`. FastAPI emits no
   `servers` block, so the API base URL is resolved from the spec URL — the app's own origin.
   Reads become plain tools, the two writes are `requires_approval=True`, the delete and the
   admin reset are refused by the `standard` profile, the deprecated route is dropped, and the
   health check is moved to `dropped` in a two-line review edit — every exclusion with a reason.
3. **Drive** — `build_agent("openai:gpt-5-mini")` launches `generated/server.py` over the real
   MCP stdio transport, exactly as a desktop client would, with the upstream token in the
   server's environment, and answers *"Which books by Ursula K. Le Guin do we have?"* against
   the live app.
4. **Govern** — the same agent tries to add a note to a book. Over stdio there is no human to
   ask, so the server-side approval gate returns a structured `APPROVAL_DENIED` and nothing
   reaches the app. In-process with `TestClient` and an approving handler, the same
   `update_book` call runs and `get_book` confirms the change.
5. **Ship** — the commands for a shared HTTP deployment and the desktop-client config.

## Files

| File | What it is |
|---|---|
| `app.py` | The FastAPI app — the stand-in for your codebase. Needs `fastapi` and `uvicorn` |
| `run.py` | The runnable driver (about 260 lines) |
| `generated/` | Created by `run.py`, git-ignored — regenerate it any time |

## Run

The app is *your* app, so its web framework is its own: `fastapi` is not a Promptise
dependency (`uvicorn` ships with Promptise; the `dev` extra includes `fastapi` for these
examples).

```bash
.venv/bin/python -m pip install fastapi
OPENAI_API_KEY=... .venv/bin/python examples/mcp/mcpcast_fastapi_app/run.py
```

Steps 1–2 run offline; the script stops with a clear message before step 3 if the key is
missing. The CLI equivalent of step 2, against an app you started yourself:

```bash
uvicorn --app-dir examples/mcp/mcpcast_fastapi_app app:app --port 8000 &
promptise mcpcast http://127.0.0.1:8000/openapi.json --no-curate \
  --profile standard --auth env-token --name bookshelf --review -o bookshelf-mcp
```

Drop `--no-curate` to let a model design the tool surface, add `--eval` to score it with a
real agent. Or let the [guided setup](https://docs.promptise.com/mcpcast/guided-setup/) ask
the questions: with the app running, `promptise mcpcast` (no arguments) detects it on port
8000, walks through model / profile / auth / name, shows the plan for review, and prints the
command above when it is done.

## What to expect

```text
==============================================================================
1. Start the Bookshelf app (app.py) with uvicorn
==============================================================================
  serving http://127.0.0.1:56798  (bearer token: 'demo-token')

==============================================================================
2. Generate the MCP server from the app's /openapi.json
==============================================================================
  spec:     http://127.0.0.1:56798/openapi.json
  base_url: http://127.0.0.1:56798   (no servers block -> the spec URL's origin)
  wrote generated/mcpcast.plan.yaml
  wrote generated/server.py
  wrote generated/bookshelf_mcp/__init__.py
  wrote generated/bookshelf_mcp/__main__.py
  wrote generated/bookshelf_mcp/config.py
  wrote generated/bookshelf_mcp/upstream.py
  wrote generated/bookshelf_mcp/approval.py
  wrote generated/bookshelf_mcp/server.py
  wrote generated/bookshelf_mcp/tools/__init__.py
  wrote generated/bookshelf_mcp/tools/books.py
  wrote generated/tests/conftest.py
  wrote generated/tests/test_tools.py
  wrote generated/README.md
  wrote generated/pyproject.toml
  wrote generated/Dockerfile
  wrote generated/.env.example
  wrote generated/.gitignore

  tool            risk    approval  upstream operation        params
  list_books      read    -         GET /books                author, limit
  create_book     write   required  POST /books               title, author, year, tags
  search_books    read    -         POST /books/search        query, author, limit
  get_book        read    -         GET /books/{book_id}      book_id
  update_book     write   required  PATCH /books/{book_id}    book_id, notes, tags, year

  not exposed (each with its reason, recorded in the plan):
    find_books_legacy: deprecated in spec
    delete_book: destructive operation excluded by profile 'standard'
    reset_catalogue: destructive operation excluded by profile 'standard'
    health_check: operational endpoint — for the load balancer, not an agent

==============================================================================
3. A real agent over MCP stdio -> generated/server.py -> the live app
==============================================================================
  question: Which books by Ursula K. Le Guin do we have, and which of them is the oldest?
  tools the agent chose:
    list_books({"author": "Ursula K. Le Guin"})
  answer: We have these Ursula K. Le Guin books:
- A Wizard of Earthsea (1968)
- The Left Hand of Darkness (1969)
- The Dispossessed (1974)

The oldest is A Wizard of Earthsea (1968).

==============================================================================
4. Writes: denied fail-closed over stdio, executed once a human approves
==============================================================================
  a) this MCP client does not support elicitation, so nobody can be asked and the gate denies:
  request: Add the note 'signed first edition' to The Dispossessed.
  tools the agent chose:
    search_books({"query": "The Dispossessed", "limit": 10})
    update_book({"book_id": 3, "notes": "signed first edition"})
      -> APPROVAL_DENIED: Approval denied for tool 'update_book': client declined or returned an invalid elicitation response
  answer: I couldn't add the note — the update_book call was denied by the system (approval denied). The note was not added.

  b) in-process, with a human (here: a callback) who approves:
    get_book(3) before: notes=''  (the denied call changed nothing)
    approval requested: update_book {"book_id": 3, "notes": "signed first edition", "tags": null, "year": null}
    update_book -> {"id": 3, "title": "The Dispossessed", "author": "Ursula K. Le Guin", "year": 1974, "tags": ["science-fiction"], "notes": "signed first edition"}
    get_book(3) after:  notes='signed first edition'

==============================================================================
5. Ship it
==============================================================================
  generated/ is a real project: edit mcpcast.plan.yaml (never the package), regenerate.
    cd generated && pytest                    # its own tests: listing, routing, the gate
    pip install -e generated && bookshelf-mcp --transport http --port 8080
    cd generated && promptise serve server:server --transport http --port 8080
    docker build -t bookshelf-mcp generated && docker run -i --env-file generated/.env \
      -e MCPCAST_BASE_URL=https://api.yourcompany.com bookshelf-mcp
  desktop clients (stdio) get the token in their own config: see generated/README.md
  env-token = one shared credential and no caller authentication, so every HTTP form above binds loopback only
  and the image serves stdio; publish only behind an authenticating gateway (--host 0.0.0.0 --public)
```

The port, the agent's wording and the exact tools it picks vary between runs; the tool
table, the dropped list and both halves of step 4 do not.

## Why `search_books` is a read and `update_book` is gated

Risk is classified by fixed rules, not by HTTP method alone: `POST /books/search` leads with
the query verb `search` and mentions no mutating verb, so it is `read` and ungated;
`PATCH /books/{book_id}` is a `write` and carries `requires_approval=True`, which the
generated server enforces in middleware for **every** MCP client. `DELETE /books/{book_id}` is
`destructive` and `POST /admin/reset` matches the destructive verb `reset` — both are refused
under `standard` and would only appear (still gated) under `--profile full`.

## Try it in Claude Desktop or Claude Code

Start the app on a fixed port (from the repo root:
`uvicorn --app-dir examples/mcp/mcpcast_fastapi_app app:app --port 8000`), regenerate against
it, and use this repo's interpreter plus the absolute path to `server.py`. The token goes in the client's
own config — a desktop client launched from the dock does not see your shell's exports:

```bash
claude mcp add bookshelf -e MCPCAST_UPSTREAM_TOKEN="Bearer demo-token" \
  -- /absolute/path/to/AgentMCP/.venv/bin/python \
     /absolute/path/to/AgentMCP/examples/mcp/mcpcast_fastapi_app/generated/server.py
```

Ask *"what do we have by Le Guin?"* and the AI calls `list_books`. Ask it to add a note and
the server asks **you** to approve first — a client that cannot show that prompt gets
`APPROVAL_DENIED`, never a silent write.

## Notes

- `generated/mcpcast.plan.yaml` is the source of truth; `server.py` is regenerated from it
  (`promptise mcpcast generated/mcpcast.plan.yaml`). It depends only on `promptise` and
  `httpx`.
- The base URL baked into the plan is the port `run.py` happened to get. For a real deployment
  set `MCPCAST_BASE_URL` at runtime, or regenerate against the app's stable URL.
- The full walkthrough for your own app — spec URLs per framework, the before/after of a good
  spec, auth mapping, shipping and CI — is the guide
  [Make Your Python API MCP-Ready](../../../docs/guides/mcpcast-python-app.md).
