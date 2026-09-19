---
title: Make Your Python API MCP-Ready — FastAPI, Django, Flask or Litestar to an MCP server for Claude, Cursor and every AI
description: The exact steps to make your own Python API usable by Claude Desktop, Claude Code, Cursor and any MCP client — from the OpenAPI document your FastAPI, Django Ninja, DRF, Flask or Litestar app already serves, through `promptise mcpcast`, to a shipped MCP server with human-approved writes. Real commands, real output, a runnable example.
keywords: FastAPI MCP server, make my Python API MCP ready, Django REST MCP, Flask MCP, expose my API to Claude, Litestar MCP server, Django Ninja MCP, FastAPI Claude Desktop, OpenAPI to MCP Python, Cursor MCP server from FastAPI
---

# Make Your Python API MCP-Ready

You are looking at your own codebase — a FastAPI, Django, Flask or Litestar
service — and the question is concrete: *exactly what do I do so Claude, Cursor
and every other AI can use this API?*

Not a spec file in hand. An app in hand. This guide is that sequence, step by
step, on a small FastAPI app that stands in for yours. Every command was run and
every output below is what it printed.

!!! tip "You do not write an MCP server"
    Your framework already emits an OpenAPI document. `promptise mcpcast` turns
    that document into a curated, safe MCP server as editable code. This page
    is the *your own app* path; the general walkthrough from a spec file is
    [MCPcast an Existing API](mcpcast-existing-api.md), the deep reference is
    [MCPcast](../mcp/server/mcpcast.md), and the [recipes](../mcp/server/mcpcast-recipes.md)
    show what happens on Stripe, GitHub and legacy Swagger 2 documents.

## What You'll Build

- **Step 1** — the URL your framework already serves the spec at
- **Step 2** — a spec that is a good *tool surface*: names, descriptions and
  examples a model can act on, with a before/after
- **Step 3** — an MCP server generated from the *running* app, with reads open,
  writes gated, and the admin, deprecated and health endpoints kept out
- **Step 4** — your app's authentication mapped onto the right auth mode
- **Step 5** — a real agent using it over MCP stdio, then Claude Desktop, Claude
  Code and Cursor
- **Step 6** — a write denied fail-closed, then executed once a human approves
- **Step 7** — the server shipped next to your app: CI guard, `promptise serve`,
  a Dockerfile, the environment variables
- **Step 8** — "works with Claude" as a feature of your product, with a
  readiness score as the quality bar

## Concepts

**The spec you already have is the input.** FastAPI, Django Ninja and Litestar
generate OpenAPI from your route signatures; Django REST Framework and Flask do
it through `drf-spectacular` and `flask-smorest`. There is nothing to export by
hand and nothing to keep in sync: `mcpcast` fetches the document from the
running app.

**The spec's words are the model's only eyes.** An agent never reads your
code. It sees a tool *name*, a tool *description* and parameter descriptions —
which are your `operation_id`, your `summary` plus docstring, and your
`Field(description=...)`. A route without them becomes a tool called
`search_books_search_post` described as *"Search"*, and that is what the model
has to choose between. Step 2 is the highest-value twenty minutes on this page.

**Exposure is a policy, not a side effect.** Every operation is risk-classified
by fixed rules (`read`, `write`, `destructive`, `financial`). The safety profile
decides which classes become tools at all, and everything that changes data is
`requires_approval=True` — enforced by the generated server's middleware for
*every* MCP client, not just polite ones. Your `DELETE` never reaches an agent
unless you opt in, and then never runs without a human.

**Your app's auth maps onto one of four modes.** A personal bearer token, a
token per user forwarded over HTTP, or a key per customer — each is one
`--auth` flag, and the generated server carries the credential so the model
never sees it.

**The plan file is yours.** `mcpcast.plan.yaml` is the reviewable, diffable
description of the tool surface; the generated package is regenerated from it.
It lives in your repository next to the app, and CI checks it against the app
on every commit.

---

## The app

The runnable example at `examples/mcp/mcpcast_fastapi_app/` is a Bookshelf API
— a deliberately ordinary FastAPI service with the shapes every real app has:

| Route | `operation_id` | What it is |
|---|---|---|
| `GET /health` | `health_check` | liveness probe for the load balancer |
| `GET /books` | `list_books` | list, optional `author` filter |
| `GET /books/{book_id}` | `get_book` | one record |
| `POST /books/search` | `search_books` | a search that is a `POST` because it carries a body |
| `POST /books` | `create_book` | a write |
| `PATCH /books/{book_id}` | `update_book` | a write |
| `DELETE /books/{book_id}` | `delete_book` | destructive |
| `GET /books/find` | `find_books_legacy` | `deprecated=True` |
| `POST /admin/reset` | `reset_catalogue` | admin-only, destructive |

Everything under `/books` and `/admin` needs `Authorization: Bearer demo-token`,
checked by a FastAPI dependency. The store is in memory. `run.py` starts the app,
generates the server, drives it with a real agent and exercises the approval
gate — the [whole driver](#the-whole-example) is at the end of this page.

---

## Step 1 — Get your spec URL

Start your app and open the OpenAPI document in a browser. You should see a
JSON (or YAML) document with `"openapi": "3.x.x"` and a `paths` object. Where it
lives depends on the framework:

| Framework | Where the spec is | Notes |
|---|---|---|
| **FastAPI** | `/openapi.json` | `FastAPI(openapi_url=...)` moves it; `None` disables it. Behind a proxy with `root_path` set, the document gains a relative `servers` entry |
| **Django Ninja** | `/api/openapi.json` | `NinjaAPI(openapi_url="/openapi.json")` by default, relative to where you mounted `api.urls` — `/api/` in the usual `path("api/", api.urls)` |
| **Django REST Framework + drf-spectacular** | `/api/schema/` | if you routed `SpectacularAPIView` there, as the drf-spectacular quickstart does; YAML by default, `?format=json` for JSON — `mcpcast` reads both. Offline: `python manage.py spectacular --file schema.yml` |
| **Flask + flask-smorest** | `<OPENAPI_URL_PREFIX>/openapi.json` | only served when `OPENAPI_URL_PREFIX` is set — with `"/"` the spec is at `/openapi.json` |
| **Litestar** | `/schema/openapi.json` | `OpenAPIConfig(path=...)` moves the whole `/schema` group |
| **Starlette, aiohttp, plain WSGI** | — | no full OpenAPI generator built in (Starlette's `SchemaGenerator` is docstring-only). Write the document by hand, or skip the spec and build the tools directly with the [Server SDK](../mcp/server/building-servers.md) |

These are the defaults as of the versions current when this page was written;
your project may have moved them, so confirm by opening the URL.

For the Bookshelf app:

```bash
cd examples/mcp/mcpcast_fastapi_app
pip install fastapi                    # the app's own web framework (uvicorn ships with Promptise)
uvicorn app:app --port 8011
curl -s http://127.0.0.1:8011/openapi.json | head -c 160
```

```text
{"openapi":"3.1.0","info":{"title":"Bookshelf API","description":"A small library catalogue: browse, search and maintain books.","version":"1.0.0"},"paths":{"/h
```

!!! note "The spec URL is also the API URL"
    FastAPI emits no `servers` block. When `mcpcast` fetches the document over
    HTTP it resolves the API base against the URL it fetched from — the app's
    own origin — so nothing else is needed. The same resolution handles a
    *relative* `servers` entry (FastAPI with `root_path`, DRF's generators).
    Working from a saved file instead, pass `--base-url https://api.yourcompany.com`.

---

## Step 2 — Make the spec good enough to be a tool surface

This is where the quality of the result is decided, and it happens in *your*
code. The mapping is exact:

| In your app | In the spec | What the agent sees |
|---|---|---|
| `operation_id="search_books"` | `operationId` | the **tool name** |
| `summary="…"` + the docstring | `summary`, `description` | the **tool description** — the only thing the model reads when choosing |
| `Field(description=…)`, `Query(description=…)` | parameter `description` | each **parameter's description** |
| `Field(examples=[…])` | `examples` | the worked **example** in the description |
| `tags=[…]` | `tags` | tool tags, kept in the plan and on the generated `@server.tool(tags=…)` |
| `deprecated=True` | `deprecated` | always dropped, with the reason `deprecated in spec` |
| `include_in_schema=False` | absent | never seen — the right answer for internal endpoints |

**Before.** A route as most of us write it on day one:

```python
class SearchQuery(BaseModel):
    q: str
    limit: int = 10

@app.post("/books/search")
async def search(request: SearchQuery) -> list[dict]:
    ...
```

FastAPI derives the operation id from the function, path and method, and the
summary from the function name. `mcpcast` turns that into a tool named
`search_books_search_post` described as `Search`, with two undescribed
parameters. Exactly what the generated tools module registers:

```python
@server.tool(
    name='search_books_search_post',
    description='Search\n\nParameters:\n  - q (string, required)\n  - limit (integer)\n\nExample: {"q": "string"}',
    read_only_hint=True,
    open_world_hint=True,
)
```

A model choosing between `search_books_search_post`, `get_books_book_id_get`
and `list_books_books_get` is guessing.

**After.** The same route in the Bookshelf app:

```python
class SearchQuery(BaseModel):
    """A search request (POST, because it carries a body)."""

    query: str = Field(
        description="Case-insensitive text matched against title and author.",
        examples=["earthsea"],
    )
    author: str | None = Field(default=None, description="Only books by this exact author.")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results.")


@books.post("/search", operation_id="search_books", summary="Search books by title or author")
async def search_books(request: SearchQuery) -> list[Book]:
    """Full-text search over titles and authors. Read-only despite being a POST."""
```

which becomes, in the plan and in the server, a tool called `search_books`
(plan excerpt, JSON schemas trimmed):

```yaml
- name: search_books
  description: Search books by title or author. Full-text search over titles and authors. Read-only despite
    being a POST.
  risk: read
  params:
    query:
      description: Case-insensitive text matched against title and author.
      required: true
    author:
      description: Only books by this exact author.
    limit:
      description: Maximum number of results.
  example:
    query: earthsea
```

The checklist, in the order it pays off:

1. **`operation_id` on every route.** It is the tool name, and tool names are
   most of what the model has to go on. Use verbs in your product's language:
   `search_books`, not `search`, not `books_post`.
2. **A `summary` and a docstring** that say what the operation does *and when
   to use it* — "Full-text search … use `get_book` when you already have an
   id". Write them for a model that cannot ask follow-up questions.
3. **`Field(description=…)` on every request-model field** and
   `Query(description=…)` on every query parameter. A parameter called `q` with
   no description is a parameter error waiting to happen.
4. **`examples=[…]`** on required fields. `mcpcast` builds each tool's worked
   example from them; without them the example is `{"q": "string"}`.
5. **Response models** (`-> list[Book]`). They are how the agent knows what
   comes back, and how the readiness evaluation mocks writes safely.
6. **`deprecated=True`** on routes you are retiring — they are dropped
   automatically — and **`include_in_schema=False`** on anything that was
   never for API consumers.

!!! warning "Set `operation_id` before anything else"
    Renaming tools later in `mcpcast.plan.yaml` works, but every regeneration
    from the spec brings the derived names back. Fix the source. In FastAPI
    you can also set `generate_unique_id_function` once on the app so every
    route's id is its function name.

Other frameworks expose the same knobs: Django Ninja's route decorators take
`operation_id`, `summary`, `description`, `tags` and `deprecated`; Litestar's
`@get`/`@post` take the same names; drf-spectacular's `@extend_schema(operation_id=…,
summary=…, description=…)` annotates a DRF view; flask-smorest uses the
docstring's first line as the summary and `@blp.doc(operationId=…)` for the id.

---

## Step 3 — Generate the server from the running app

Point `mcpcast` at the URL. `--profile standard` opens reads and gated writes,
`--auth env-token` is the mode for a personal server (Step 4 explains why),
`--review` shows the plan before writing, and `--no-curate` keeps this run
deterministic and offline:

```bash
promptise mcpcast http://127.0.0.1:8011/openapi.json --no-curate \
    --profile standard --auth env-token --name bookshelf --review -o bookshelf-mcp
```

Trimmed to the columns that matter:

```text
Parsed 9 operations from http://127.0.0.1:8011/openapi.json; profile=standard auth=env-token
Tools (6) — profile standard
  health_check   read   —         GET /health             —
  list_books     read   —         GET /books              author, limit
  create_book    write  required  POST /books             title, author, year, tags
  search_books   read   —         POST /books/search      query, author, limit
  get_book       read   —         GET /books/{book_id}    book_id
  update_book    write  required  PATCH /books/{book_id}  book_id, notes, tags, year
Not exposed (3)
  find_books_legacy   deprecated in spec
  delete_book         destructive operation excluded by profile 'standard'
  reset_catalogue     destructive operation excluded by profile 'standard'
Write the project? [Y/n]:
╭─────────────────────── mcpcast ────────────────────────╮
│ bookshelf → bookshelf-mcp/                             │
│   tools: 6  (2 require human approval)                 │
│   not exposed: 3 operations (with reasons in the plan) │
│   files: bookshelf_mcp/ (8 modules), tests/ (2), server.py, README.md,           │
│   pyproject.toml, Dockerfile, .env.example, .gitignore                          │
╰──────────────────────────────────────────────────────────────────────────────────╯
```

Read that against your own endpoints:

- **`POST /books/search` is `read`.** The classifier looks at the leading verb
  of the id, summary and last path segment, not just the method: `search` with
  no mutating verb anywhere is a read, so the tool is ungated. `POST /books`
  (`create_book`) is a write.
- **`create_book` and `update_book` are `requires_approval=True`.** Under
  `standard` every write is generated *and* gated; the gate is middleware in
  the generated server, so it holds for Claude Desktop, Cursor and a script
  alike.
- **`delete_book` is not there.** `DELETE` is `destructive`, and `standard`
  refuses that class. It appears — still gated — only under `--profile full`.
- **`reset_catalogue` is not there either**, twice over: the verb `reset`
  makes it destructive, and a path segment `admin` escalates whatever class an
  operation has. An admin-only `GET` under `standard` would still be generated
  as a gated write rather than a free read.
- **`find_books_legacy` is dropped** because it is `deprecated: true`, under
  every profile.
- **`health_check` is a tool**, and it should not be: an agent has no task
  that needs a liveness probe. Deterministic generation keeps every read.
  Three fixes, pick one: `include_in_schema=False` in the app (best if no API
  consumer needs it in the docs either), move it to `dropped` in the plan
  (below), or let curation decide — it drops health checks on its own.

The plan file is the artifact. In `bookshelf-mcp/mcpcast.plan.yaml` the fix is
moving one entry:

```yaml
dropped:
- operation_id: find_books_legacy
  reason: deprecated in spec
- operation_id: delete_book
  reason: destructive operation excluded by profile 'standard'
- operation_id: reset_catalogue
  reason: destructive operation excluded by profile 'standard'
- operation_id: health_check
  reason: operational endpoint — for the load balancer, not an agent
```

then `promptise mcpcast bookshelf-mcp/mcpcast.plan.yaml` regenerates the
package, `server.py`, `tests/` and `README.md` from it, leaving the plan (and
your comments) untouched. The runnable example makes the same edit in code, so its output shows
five tools and four drops.

!!! note "With curation on"
    Drop `--no-curate` and a model designs the surface (it needs
    `OPENAI_API_KEY`; `--model` picks another provider). On this app one run
    produced two tools: `find_books` (three routes: by id, search, list —
    dispatched by which parameters you pass) and `save_book` (create + update),
    with `health_check`, the deprecated route and `reset_catalogue` dropped
    with written reasons. It also marked `notes` on `save_book` as hidden — a
    hidden parameter is never shown to the agent, and that is the one field
    `update_book` mostly exists for. Curation is a strong first draft that you
    review, which is what `--review` is for; every proposal is
    checked in code and [risk is never downgraded](../mcp/server/mcpcast.md#curation).
    Nine operations is small enough that the deterministic surface is already
    good; curation earns its keep on the 100-operation apps.

---

## Step 4 — Map your app's auth to an auth mode

The generated server calls your API with a credential. Which one, and where it
comes from, is the `--auth` mode — pick it from how your app authenticates
today:

| Your app today | Who will use the MCP server | `--auth` | What the server does |
|---|---|---|---|
| one bearer token / API key per developer or per team | you, in Claude Desktop, Claude Code, Cursor (stdio) | `env-token` | sends `MCPCAST_UPSTREAM_TOKEN` as `Authorization` on every call |
| a token per user (JWT, OAuth access token) | many users, over HTTP, each as themselves | `passthrough` (default) | forwards the caller's `Authorization` header unchanged — your API's own permissions still apply |
| a key per customer / tenant | your customers, on a server you host | `api-key` | clients present `x-api-key`; each key names a tenant whose upstream credential lives on the server, never on the client |
| nothing (a local sandbox) | you, locally | `none` | no credentials; the server refuses to bind to anything but loopback |

The Bookshelf app takes one bearer token, and the goal is *my* desktop client
using *my* API, so `env-token` is right. Desktop clients launch the server over
stdio and cannot send request headers, which is why `passthrough` would fail
every call there with `UPSTREAM_AUTH_MISSING`.

The generated `bookshelf-mcp/README.md` already carries the client
configuration with the credential in the client's *own* environment block:

```json
{
  "mcpServers": {
    "bookshelf": { "command": "python", "args": ["/absolute/path/to/server.py"], "env": {"MCPCAST_UPSTREAM_TOKEN": "Bearer <your API token>"} }
  }
}
```

and it says why it is there:

> The credential goes in the client's own config below — a desktop client
> launched from the GUI does not inherit your shell, so `export
> MCPCAST_UPSTREAM_TOKEN=…` in a terminal will not reach it.

The value is the whole header — `Bearer demo-token`, scheme included. Get it
wrong and the failure is specific: without the variable the tool returns
`UPSTREAM_AUTH_MISSING`; with the wrong token your API answers and the tool
relays it as `UPSTREAM_ERROR … HTTP 401` (both verbatim in
[Troubleshooting](#troubleshooting)).

For the other two shapes: `passthrough` is a shared HTTP deployment
(`promptise serve server:server -t http`) where every client sends its own
`Authorization` header — see [Auth & Security](../mcp/server/auth-security.md);
`api-key` adds per-tenant credentials and tenant-scoped four-eyes approval — see
[Multi-Tenancy](../mcp/server/multi-tenancy.md) and the
[storefront lab](lab-mcpcast-storefront.md), which runs it end to end.

---

## Step 5 — Try it with a real agent

`run.py` connects `build_agent("openai:gpt-5-mini")` to the generated server
over the real MCP stdio transport — the same way Claude Desktop launches it —
with the token in the server's environment, and asks a question the live app
has to answer:

```python
agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={
        "bookshelf": StdioServerSpec(
            command=sys.executable,
            args=[str(OUT / "server.py")],          # generated/server.py
            env={"MCPCAST_UPSTREAM_TOKEN": "Bearer demo-token"},
        )
    },
    instructions="You are the Bookshelf assistant. Answer from what the tools return. Be brief.",
)
result = await agent.ainvoke({"messages": [HumanMessage(content=QUESTION)]})
```

```text
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
```

One tool call, the right one, with the `author` filter the description
advertised — because Step 2 gave the model something to read.

The port, the model's wording and the exact arguments it picks (a `limit`
of 10 or 20, say) vary between runs; the tool table, the dropped list and
both halves of Step 6's write do not.

Now the clients you actually wanted. Use the interpreter that has `promptise`
installed and the absolute path to `server.py`:

=== "Claude Desktop"

    `claude_desktop_config.json`:

    ```json
    {
      "mcpServers": {
        "bookshelf": {
          "command": "/absolute/path/to/.venv/bin/python",
          "args": ["/absolute/path/to/bookshelf-mcp/server.py"],
          "env": { "MCPCAST_UPSTREAM_TOKEN": "Bearer demo-token" }
        }
      }
    }
    ```

=== "Claude Code"

    ```bash
    claude mcp add bookshelf -e MCPCAST_UPSTREAM_TOKEN="Bearer demo-token" \
        -- /absolute/path/to/.venv/bin/python /absolute/path/to/bookshelf-mcp/server.py
    ```

=== "Cursor"

    `.cursor/mcp.json`:

    ```json
    {
      "mcpServers": {
        "bookshelf": {
          "command": "/absolute/path/to/.venv/bin/python",
          "args": ["/absolute/path/to/bookshelf-mcp/server.py"],
          "env": { "MCPCAST_UPSTREAM_TOKEN": "Bearer demo-token" }
        }
      }
    }
    ```

=== "Any HTTP client"

    ```bash
    cd bookshelf-mcp
    MCPCAST_UPSTREAM_TOKEN='Bearer demo-token' promptise serve server:server -t http --port 8080
    # MCP endpoint: http://127.0.0.1:8080/mcp
    ```

Ask *"what do we have by Le Guin?"* and the client calls `list_books`. The app
must be reachable at the base URL baked into the plan; `MCPCAST_BASE_URL`
overrides it at runtime.

---

## Step 6 — Writes and approval

Reads are half the value. The other half — *"add a note to that book"* — is
where a raw API handed to a model becomes a headline. The generated server gates
it, and the gate is not advisory.

**Over stdio, with no human to ask.** `run.py` asks the same agent for a write.
The Promptise MCP client does not implement elicitation, so the server has
nobody to put the question to — and denies:

```text
4. Writes: denied fail-closed over stdio, executed once a human approves
==============================================================================
  a) this MCP client does not support elicitation, so nobody can be asked and the gate denies:
  request: Add the note 'signed first edition' to The Dispossessed.
  tools the agent chose:
    search_books({"query": "The Dispossessed", "limit": 10})
    update_book({"book_id": 3, "notes": "signed first edition"})
      -> APPROVAL_DENIED: Approval denied for tool 'update_book': client declined or returned an invalid elicitation response
  answer: I couldn't add the note — the update_book call was denied by the system (approval denied). The note was not added.
```

The agent found the book, chose the right tool with the right arguments, and
the `PATCH` never reached the app. The error is structured
(`{"error": {"code": "APPROVAL_DENIED", …}}`), so the model reports it instead
of retrying or inventing a result.

**With a human who says yes.** In a client that implements MCP elicitation
(Claude Code, for one) the same call shows *you* a prompt naming the tool and
its arguments, and runs only if you accept. `run.py` reproduces that
in-process: `build_server()` in the
generated module takes an `approval_handler`, and `TestClient` runs the full
pipeline — validation, middleware, gate, handler — without a port:

```python
def reviewer(request: ApprovalRequest) -> bool:
    print(f"    approval requested: {request.tool_name} {json.dumps(request.arguments)}")
    return True

client = TestClient(import_generated().build_server(approval_handler=reviewer))
(reply,) = await client.call_tool("update_book", {"book_id": 3, "notes": "signed first edition"})
```

```text
  b) in-process, with a human (here: a callback) who approves:
    get_book(3) before: notes=''  (the denied call changed nothing)
    approval requested: update_book {"book_id": 3, "notes": "signed first edition", "tags": null, "year": null}
    update_book -> {"id": 3, "title": "The Dispossessed", "author": "Ursula K. Le Guin", "year": 1974, "tags": ["science-fiction"], "notes": "signed first edition"}
    get_book(3) after:  notes='signed first edition'
```

The reviewer sees exactly what will run, the `PATCH` hits the live app once
approved, and the follow-up read proves it. A timeout denies; a handler crash
denies; nothing is ever silently allowed. Elicitation is "confirm your own
action"; for an independent reviewer — someone *other* than the person driving
the agent — use `--auth api-key` with `--approval pending`, covered in
[Approval Gates](../mcp/server/approval-gates.md).

---

## Step 7 — Ship it next to your app

**Keep the project in the repository**, beside the code it exposes:

```text
your-service/
├── app/                      # your FastAPI / Django / Flask code
├── bookshelf-mcp/
│   ├── mcpcast.plan.yaml     # the source of truth — reviewed like code
│   ├── server.py             # launcher — regenerated, never hand-edited
│   ├── bookshelf_mcp/        # the server as a package — regenerated
│   ├── tests/                # generated tests + your own files
│   ├── pyproject.toml        # yours after the first write
│   ├── Dockerfile  .env.example  .gitignore
│   └── README.md
└── tests/
    └── test_mcp_surface.py   # below
```

**Guard it in CI.** Your API will change; the tool surface should fail loudly
when it does, not drift. The pipeline is a Python API, and `app.openapi()` is
the spec — no server needed:

```python
"""CI guard: the shipped MCP tool surface matches the app. Offline, deterministic, fast."""

from pathlib import Path

from promptise.mcpcast import MCPcastPlan, RiskClass, mcpcast

from app import app  # your FastAPI app — app.openapi() is the spec, no server needed

PLAN = MCPcastPlan.load(Path(__file__).parent.parent / "bookshelf-mcp" / "mcpcast.plan.yaml")


def test_every_endpoint_is_a_tool_or_dropped_with_a_reason():
    """A new endpoint must be added to the plan or dropped deliberately — never ignored."""
    fresh = mcpcast(app.openapi(), profile=PLAN.profile, base_url=PLAN.api.base_url)
    in_app = fresh.kept_operations | {d.operation_id for d in fresh.dropped}
    in_plan = PLAN.kept_operations | {d.operation_id for d in PLAN.dropped}
    assert in_app == in_plan, f"unaccounted: {in_app ^ in_plan}"


def test_no_tool_is_less_risky_than_the_classifier_says():
    """Hand edits may raise a tool's risk, never lower it."""
    fresh = mcpcast(app.openapi(), profile=PLAN.profile, base_url=PLAN.api.base_url)
    floor = {op: t.risk for t in fresh.tools for op in t.operations}
    for tool in PLAN.tools:
        for op in tool.operations:
            assert tool.risk.at_least(floor[op]), f"{tool.name} downgrades {op}"


def test_everything_that_writes_is_approval_gated():
    for tool in PLAN.tools:
        assert tool.requires_approval == (tool.risk is not RiskClass.READ), tool.name


def test_the_tools_customers_rely_on_still_exist():
    assert {"list_books", "search_books", "get_book"} <= set(PLAN.tool_names)
```

```text
$ pytest tests/test_mcp_surface.py -q
4 passed, 2 warnings in 2.26s
```

When a colleague adds `POST /books/{book_id}/lend`, the first test fails with
`unaccounted: {'lend_book'}` and the pull request has to say what happens to it:
add it to the plan (regenerate from the spec into a scratch directory and copy
the new tool's entry over, or write it by hand) or drop it with a reason. Either way `promptise mcpcast bookshelf-mcp/mcpcast.plan.yaml`
regenerates the package, and the diff is reviewed like any other. The
[nightly readiness test](mcpcast-existing-api.md#step-12-keep-the-surface-from-rotting)
in the general guide adds a model-scored floor on top.

**Run it.** The launcher exposes `server`, so the standard runner works from
inside the project directory, with the dashboard and hot reload from
[Deployment](../mcp/server/deployment.md) — or install the project and run it
as a command:

```bash
cd bookshelf-mcp
promptise serve server:server --transport http --port 8080
pip install -e ".[dev]" && pytest && bookshelf-mcp --transport http --port 8080
```

```text
INFO:     Uvicorn running on http://127.0.0.1:8080 (Press CTRL+C to quit)
    Server      bookshelf v0.1.0
    Transport   Streamable HTTP
    Endpoint    http://127.0.0.1:8080/mcp
    Tools       5 registered
```

**Containerise it.** The project ships with a `Dockerfile` (written once —
edit it freely): a fully-qualified base image, a non-root user, no baked-in
configuration. The server depends only on `promptise` and `httpx` and talks
to your API over HTTP, so it can run in the same image as your app or in its
own:

```dockerfile
FROM python:3.12-slim-bookworm

WORKDIR /app
COPY pyproject.toml README.md ./
COPY bookshelf_mcp ./bookshelf_mcp
RUN pip install --no-cache-dir . \
    && useradd --system --create-home --shell /usr/sbin/nologin app
USER app

# The upstream base URL is compiled into bookshelf_mcp/config.py; MCPCAST_BASE_URL at run
# time points the image at another environment. Credentials come from the
# environment at run time too (see .env.example) — never bake them into the image.
EXPOSE 8080

# Auth mode 'env-token' executes every call with the operator's
# MCPCAST_UPSTREAM_TOKEN and has no MCP-level authentication, so the server
# binds loopback only and this image serves stdio. To publish it anyway — only
# behind an authenticating gateway — run it with
# `--transport http --host 0.0.0.0 --public` (or MCPCAST_PUBLIC=1). Regenerate
# with --auth api-key or passthrough for a shared deployment.
CMD ["bookshelf-mcp"]
```

```bash
docker build -t bookshelf-mcp . && docker run -i --env-file .env bookshelf-mcp
```

This is an `env-token` project — one credential, no caller authentication —
so the image serves stdio and the server refuses a non-loopback bind unless
you pass `--public` behind an authenticating gateway. Regenerate with
`--auth api-key` for a shared HTTP deployment that authenticates its callers.

**Everything the server reads at runtime:**

| Variable | Purpose | Default |
|---|---|---|
| `MCPCAST_BASE_URL` | Where your API is — staging vs production, or the port the app got this time. One clean absolute URL: a query string or fragment, `user:password@` or whitespace is refused at start-up | `api.base_url` from the plan |
| `MCPCAST_UPSTREAM_TOKEN` | `env-token`: the full `Authorization` value sent upstream, scheme included — or the raw key in the header / query parameter your app's security scheme names (`.env.example` says which); a `<placeholder>` left in place is refused | — |
| `MCPCAST_CLIENT_KEYS` | `api-key`: JSON map of client key → `{client_id, tenant_id, roles}`; the server refuses to start with none, with a `<placeholder>` key or with a documentation key (`sk-acme`, `sk-reviewer`) — `.env.example` ships no working key | — |
| `MCPCAST_UPSTREAM_TOKENS` | `api-key`: JSON map of tenant → upstream credential, read per call so rotation needs no restart | — |
| `MCPCAST_TIMEOUT` | Total time one upstream call may take, seconds — connecting, sending, waiting and reading the body together | `30` |
| `MCPCAST_APPROVAL_TIMEOUT` | How long a gated call waits for a decision before it is denied | `300` |
| `MCPCAST_MAX_RESPONSE_BYTES`, `MCPCAST_ERROR_EXCERPT_CHARS`, `MCPCAST_ALLOW_INSECURE_HTTP`, `MCPCAST_PUBLIC`, `MCPCAST_MAX_PENDING`, `MCPCAST_MAX_PENDING_PER_TENANT`, `MCPCAST_MAX_PENDING_PER_CLIENT` | Response cap, error-excerpt length (the credential is scrubbed from the body first), plain-http opt-in (pre-set in `.env.example` when the plan's upstream is plain http), non-loopback opt-in, pending-approval capacity (server-wide, per tenant, per API key) — see the generated README's configuration table | see README |

The `run.py` example bakes the ephemeral port it was given into the plan; for
anything that outlives one run, regenerate against the app's stable URL or set
`MCPCAST_BASE_URL`.

---

## Step 8 — Make it a feature

Once the server is in the repo, "works with Claude" is a line in your product's
README rather than a roadmap item. The block your users need is three lines:

```markdown
## Use Bookshelf with Claude, Cursor or any MCP client

pip install promptise
claude mcp add bookshelf -e MCPCAST_UPSTREAM_TOKEN="Bearer <your token>" \
    -- python /path/to/bookshelf-mcp/server.py

Reads work immediately. Anything that changes data asks you to approve it first.
```

**Make the readiness score your quality bar.** Tool design is usually a matter
of opinion; `--eval` makes it a measurement. A model writes realistic tasks for
your tools, a real agent attempts them against the generated server — reads
against your live API, writes against spec-derived mocks behind an
auto-approver, so nothing real changes — and you get a grade with named fixes:

```bash
export MCPCAST_EVAL_AUTHORIZATION='Bearer demo-token'   # so live reads are authenticated
promptise mcpcast bookshelf-mcp/mcpcast.plan.yaml --eval --eval-tasks 8
```

```text
Evaluating with openai:gpt-5-mini (8 tasks)…
Agent Readiness: A  (8/8 tasks succeeded)
  ✗ `update_book` vs `get_book` are ambiguous — the agent picked `get_book` in 1/2 runs that needed `update_book` → merge them, or say in each description when NOT to use it
  • `list_books` has no example — agents lean on examples heavily
Report: bookshelf-mcp/eval/report.md  Tasks: bookshelf-mcp/eval/tasks.yaml
```

(Report paths shortened; the CLI prints them absolute.) `eval/report.md`
records score 0.95, correct tool selected first 88%, parameter error rate 0%,
and a per-task table — every task landed, and the one selection miss was
`get_book` called before `update_book` on a task that said "update book id 1".

Both fixes are small. The generated example only covers *required*
parameters and `list_books` has none, so its example is one line in the plan:
`example: {author: Ursula K. Le Guin}`. The ambiguity is one sentence in
`update_book`'s docstring, in your code — "use `get_book` to read; this tool
only changes fields". Regenerate, re-score. An **A** before a release is a bar
a team can hold, and the score moves a little between runs, so assert a floor
rather than a number.

**When your API is big.** A nine-operation API reads fine as one tool per
operation; a two-hundred-operation one does not. That is what curation is for — a model that
merges routes serving one intent, renames into your domain language, puts
parameters on a diet and drops what no user task needs, with every proposal
checked in code. The [recipes](../mcp/server/mcpcast-recipes.md) show it on
Stripe and on GitHub's 1,225-operation spec, including how to narrow a spec by
tag before curating.

---

## What You've Built

- **The spec URL for your framework**, and a spec whose names, descriptions
  and examples are written for a model
- **An MCP server generated from the running app** — reads open, writes
  `requires_approval=True`, the delete and the admin reset refused by the
  profile, the deprecated route and the health check dropped with reasons
- **The right auth mode** for your app's authentication, with the credential
  in the client's config and never in the model's context
- **A real agent using it over MCP stdio**, and the Claude Desktop, Claude Code
  and Cursor configurations
- **A write denied fail-closed** when no human could be asked, and executed
  against the live app once one approved
- **The server shipped next to your app** with a CI guard, `promptise serve`,
  a Dockerfile and the runtime variables
- **A readiness grade** as the quality bar for "works with Claude"

## Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `could not fetch spec from http://127.0.0.1:8011/: Client error '404 Not Found'` | The URL is your app's root, not the spec | Use the path from the Step 1 table — `/openapi.json`, `/api/openapi.json`, `/api/schema/`, `/schema/openapi.json` |
| `http://127.0.0.1:8011/docs: not valid JSON or YAML: …` | You pointed at the Swagger UI page, not the document it renders | Same fix — the JSON/YAML URL, not `/docs` or `/redoc` |
| `spec has no 'paths' — is this an OpenAPI document?` | The URL returned JSON that is not an OpenAPI document (an endpoint's payload, an error body) | Open the URL in a browser and check for `"openapi"` and `"paths"` |
| `base_url must start with http:// or https:// (got '/api'); the spec declares a relative server URL '/api'; pass --base-url https://<api-host>/api` | A spec read from a *file* with a relative `servers` entry (FastAPI `root_path`, DRF generators) | Fetch it over HTTP so it resolves against the spec URL, or pass `--base-url` |
| Tools named `search_books_search_post`, `get_books_book_id_get` | No `operation_id` on the routes | Step 2 — set them in the app; renaming in the plan is undone by the next regeneration from the spec |
| `UPSTREAM_AUTH_MISSING: MCPCAST_UPSTREAM_TOKEN is not set. Auth mode 'env-token' presents that value as the Authorization header on every upstream call …` | The server has no credential in *its* environment | Put it in the client's `env` block (or `-e` for `claude mcp add`); a shell `export` does not reach a GUI-launched client |
| `UPSTREAM_AUTH_MISSING` on a `passthrough` server | The MCP client sent no `Authorization` header — stdio clients cannot | Regenerate with `--auth env-token` for a personal server, or run over HTTP with the header |
| `UPSTREAM_ERROR: GET /books/{book_id} returned HTTP 401: {"detail":"invalid token"}` | Your app rejected the token — `MCPCAST_UPSTREAM_TOKEN` is wrong | Set the token your app accepts, scheme included: `Bearer demo-token` |
| `UPSTREAM_ERROR: … HTTP 401: {"detail":"Not authenticated"}` | The variable holds the bare token without `Bearer ` | The value is the whole header value |
| `UPSTREAM_UNREACHABLE: GET /books/{book_id} failed: ConnectError` | The app is not running at `api.base_url` — for the example, the port changes every run | Start the app, or set `MCPCAST_BASE_URL` |
| `UPSTREAM_TIMEOUT: GET /books did not complete within 30s (MCPCAST_TIMEOUT)` | The app accepted the connection but did not answer in time — the whole call, headers and body, has a wall-clock deadline | Raise `MCPCAST_TIMEOUT` for slow endpoints; the error is marked retryable |
| `APPROVAL_DENIED: Approval denied for tool 'update_book': client declined or returned an invalid elicitation response` | A gated tool was called from a client that cannot show an approval prompt | Expected. Use a client that implements elicitation, `--approval pending` with `--auth api-key`, or an `approval_handler` in tests |
| `bookshelf-mcp already contains an mcpcast project — pass --force to overwrite it, or regenerate from bookshelf-mcp/mcpcast.plan.yaml to keep your edits` | Re-running against the *spec* would discard your plan edits | Regenerate from the plan to keep them; `--force` only if you mean to start over |
| `auth mode 'none' refuses to bind to a non-loopback address` | `--auth none` served on `0.0.0.0` | `none` is for local demos; regenerate with a real auth mode |

## The whole example

`examples/mcp/mcpcast_fastapi_app/run.py` — start the app, generate, drive with
a real agent, deny and approve a write, print the shipping commands. It needs
`fastapi`, `uvicorn` and `OPENAI_API_KEY`; the app it MCPcasts is `app.py` in
the same directory.

```python
--8<-- "examples/mcp/mcpcast_fastapi_app/run.py"
```

## Next Steps

- [MCPcast, end to end](../mcpcast/index.md) — how MCP works, the full real run, and the review checklist

- **[MCPcast an Existing API](mcpcast-existing-api.md)** — the general
  walkthrough: the `full` profile, four-eyes review, curation and hand edits,
  acting on a readiness score, the nightly CI job
- **[MCPcast Recipes](../mcp/server/mcpcast-recipes.md)** — Stripe, GitHub,
  Swagger 2, and what to do when the spec fights back
- **[MCPcast reference](../mcp/server/mcpcast.md)** — risk rules, curation
  post-conditions, the plan schema, auth modes, readiness metrics
- **[Approval Gates](../mcp/server/approval-gates.md)** — elicitation, pending
  stores, webhooks and custom handlers
- **[Deployment](../mcp/server/deployment.md)** — `promptise serve`, the
  dashboard, transports, CORS
- **[Building Production MCP Servers](production-mcp-servers.md)** — for the
  tools your spec cannot express, hand-written and mounted alongside the
  generated ones
- **[Examples gallery](../resources/examples.md)** — every runnable example,
  including this one
