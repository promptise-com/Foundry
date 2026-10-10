# Multi-Tenancy

First-class tenant isolation across the whole stack. When a caller carries a
`tenant_id`, it becomes part of **every isolation key** — semantic cache,
memory scoping, conversation ownership, rate-limit buckets, audit entries,
and tool access — so two tenants with the *same* `user_id` can never see
each other's data. Isolation is a structural invariant, not a naming
convention buried in `metadata`.

!!! tip "Why this matters"
    Cross-tenant data leakage is the worst incident class for any
    multi-customer platform. Conventions ("we always prefix keys with the
    org") fail silently the first time one call site forgets. Promptise
    bakes the tenant into the key derivation itself, in one place per
    surface — there is no code path that stores or reads tenant data
    without it.

## Server side: `ClientContext.tenant_id`

`AuthMiddleware` extracts the tenant from a configurable JWT claim
(default `tenant_id`) and attaches it to `ctx.client.tenant_id`:

```python
from promptise.mcp.server import AuthMiddleware, JWTAuth, MCPServer

server = MCPServer(name="api")
server.add_middleware(
    AuthMiddleware(
        JWTAuth(secret="...", audience="api://my-server"),
        tenant_claim="tenant_id",   # or "org", "org_id", ... — your IdP's claim
    )
)

@server.tool(auth=True)
async def whoami(ctx: RequestContext) -> dict:
    return {"client": ctx.client.client_id, "tenant": ctx.client.tenant_id}
```

Only **string** claim values are accepted; anything else leaves
`tenant_id` unset and tenant guards fail closed.

`audience` and `issuer` are checked on every token: a token whose `aud`
does not include `api://my-server`, or whose `iss` differs from `issuer`
when you set one, is rejected. Set `audience` whenever the signing secret
is shared by more than one service, so a token minted for one of them is
refused by the others. `AsymmetricJWTAuth` takes the same two arguments,
and `JwksAuth` requires `audience`.

For `APIKeyAuth`, the tenant comes from the key's config dict:

```python
APIKeyAuth(keys={
    "sk-acme-1":   {"client_id": "acme-agent",   "roles": ["analyst"], "tenant_id": "acme"},
    "sk-globex-1": {"client_id": "globex-agent", "roles": ["analyst"], "tenant_id": "globex"},
})
```

## Enforcing tenancy: guards and `require_tenant`

Two guards mirror the role/scope guards:

| Guard | Grants access when |
|-------|--------------------|
| `RequireTenant()` | The client has *any* tenant identity |
| `HasTenant("acme", "globex")` | The client belongs to one of the listed tenants |

```python
from promptise.mcp.server import HasTenant, RequireTenant

@server.tool(auth=True, guards=[RequireTenant()])
async def list_records() -> list: ...

@server.tool(auth=True, guards=[HasTenant("acme")])
async def acme_only_tool() -> str: ...
```

To make tenancy a **server-wide invariant**, build the server with
`require_tenant=True` — every tool (from decorators, routers, mounts, or
OpenAPI import) is forced to authenticate and carries a `RequireTenant`
guard. A client whose token lacks the tenant claim is denied on every call:

```python
server = MCPServer(name="api", require_tenant=True)  # implies require_auth
```

## What the tenant automatically isolates (server side)

| Surface | Behavior with a tenant present |
|---------|-------------------------------|
| Rate limiting | Bucket keys are tenant-qualified in both `RateLimitMiddleware` and declared per-tool limits — one tenant's traffic can never exhaust another's quota, even for identical `client_id` strings |
| Audit log | `AuditMiddleware` records `tenant_id` in each entry's identity descriptors — tenant-scoped forensics without joining external data |
| Result caching | `CacheMiddleware` and `@cached` key every entry on the caller (issuer, tenant and client id) by default, so a result computed for one tenant is never served to another. Widen with `scope="tenant"` or `scope="shared"` only for data that does not depend on the caller — see [Caching](caching-performance.md#who-shares-a-cached-result) |
| Tool access | `RequireTenant` / `HasTenant` guards, or the server-wide `require_tenant` invariant |
| Tool listing | Every tool is listed to every client unless you build the server with `hide_unauthorized_tools=True` — see below |
| Job queue | `MCPQueue` jobs record the submitting client and tenant; `queue_status`, `queue_result`, `queue_cancel` and `queue_list` only show a caller its own jobs, and the admin role widens that to its own tenant only ([Job ownership](queue.md#job-ownership)) |

`SessionState` needs no tenant prefix: it is keyed by the live transport
session, which is connection-scoped and therefore cannot be shared across
tenants.

## Hiding tools a tenant cannot call

Guards decide whether a tool may be **called**. By default, `tools/list`
still returns every registered tool to every client, so a tool guarded with
`HasTenant("acme")` shows its name, description and input schema to
Globex too. Calling it is refused, but the listing itself can reveal a
feature, a customer-specific integration, or an admin tool, and an agent
shown a tool it cannot use wastes turns on it.

Build the server with `hide_unauthorized_tools=True` to filter the list per
request:

```python
server = MCPServer(name="crm", require_tenant=True, hide_unauthorized_tools=True)
server.add_middleware(AuthMiddleware(JWTAuth(secret="...", audience="api://crm"), tenant_claim="org"))

@server.tool(guards=[HasTenant("acme")])
async def forecast_renewals(ctx: RequestContext) -> list[dict]: ...
```

For each `tools/list` request (and each read of the `docs://manifest`
resource) the server authenticates the request with its `AuthMiddleware`,
exactly as a tool call would, and evaluates every tool's guards against
that identity. Acme's agent sees `forecast_renewals`; Globex's does not.

`resources/list`, `resources/templates/list` and `prompts/list` are
filtered the same way, using each resource's and prompt's `auth`, `roles`
and `guards`.

- **It fails closed.** Credentials that do not verify hide every tool that
  needs authentication, and a guard that raises hides its tool.
- **Calls are still guarded.** Hiding is not the access control; a client
  that calls a hidden tool by name gets `ACCESS_DENIED` as before. Denials
  name the caller's own tenant, never the tenants that are allowed.
- **Guards see what `AuthMiddleware` sets.** Built-in guards read
  `ctx.client`, which `AuthMiddleware` and its `on_authenticate` hook fill
  in. A custom guard that depends on state another middleware sets cannot
  be evaluated at list time, so its tool is hidden.
- **The list follows the credential that asks for it.** An agent discovers
  its tools once, when it is built, with the credentials in its
  `HTTPServerSpec`. Give that credential access to every tool the agent
  should know about; each call is then made as the invoking user (see
  below) and checked against that user's guards.

## Agent side: `CallerContext.tenant_id`

The same invariant applies inside the agent. `CallerContext` gains
`tenant_id`, and one derivation — `CallerContext.isolation_key`
(`"{tenant_id}::{user_id}"`, or the plain `user_id` without a tenant) —
feeds every per-user isolation surface:

```python
from promptise import CallerContext

acme_alice   = CallerContext(user_id="alice", tenant_id="acme")
globex_alice = CallerContext(user_id="alice", tenant_id="globex")

# Same user_id, different tenants — fully isolated:
await agent.chat("...", session_id=sid, caller=acme_alice)
```

| Surface | Behavior |
|---------|----------|
| Semantic cache | Scope keys embed the tenant — cross-tenant cache hits are structurally impossible. `purge_user("alice", tenant_id="acme")` purges exactly that tenant's scope |
| Memory | Providers receive the isolation key as `user_id` — no provider changes needed, isolation guaranteed at the scoping layer |
| Conversations | Session ownership keys on the isolation key — a same-`user_id` caller from another tenant gets `SessionAccessDenied` |
| Cross-agent delegation | The full `CallerContext` (including tenant) is inherited by peers via caller-context continuity |
| MCP tool calls | `CallerContext.bearer_token` is sent on every call to an HTTP/SSE MCP server during the invocation, over a session opened for that token — so the server authenticates the user and derives *their* tenant |

### The user's token reaches the MCP server

One agent can serve every tenant. Build it once, and pass each user's token
on the invocation:

```python
agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={"crm": HTTPServerSpec(url=CRM_URL, bearer_token=SERVICE_TOKEN)},
)

alice = CallerContext(user_id="alice", tenant_id="acme", bearer_token=alice_jwt)
bob = CallerContext(user_id="bob", tenant_id="globex", bearer_token=bob_jwt)

# Concurrent invocations: each tool call carries its own caller's token.
await asyncio.gather(
    agent.ainvoke({"messages": [...]}, caller=alice),
    agent.ainvoke({"messages": [...]}, caller=bob),
)
```

Each distinct token gets its own MCP session, opened on first use and
reused by that caller's later calls; concurrent invocations never share a
session or a header. Idle sessions close after five minutes, and at most
256 idle sessions stay open. `SERVICE_TOKEN` is used to discover the tools
when the agent is built, and for invocations without a caller token.

Two things to check:

- **The server's tenant comes from the token, not from
  `CallerContext.tenant_id`.** A caller with a `tenant_id` but no
  `bearer_token` is sent with the spec's credential, and the server sees
  that identity. For tenant-scoped data, always pass the user's token.
- **Opt a server out** with `HTTPServerSpec(..., forward_caller_token=False)`
  when the user's token must not leave your trust boundary (a third-party
  MCP server) or is not meant for it. stdio servers have no request
  headers, never receive the token, and log a warning the first time a
  caller token cannot be sent to them.

!!! note "Memory providers see composite ids"
    With a tenant present, providers store owner ids like
    `"acme::alice"`. If you query a provider directly (outside the agent),
    use the same composite form.

!!! note "The isolation-key separator is reserved"
    `CallerContext` construction rejects (with a `ValueError`) a `tenant_id`
    containing **any** colon and a `user_id` containing the **`::`** sequence.
    That makes the `tenant::user` join unambiguous and injective, and keeps
    the tenanted keyspace (always containing `::`) provably disjoint from the
    untenanted one (a raw user_id, which can never contain `::`) — an
    untenanted `user_id="acme::alice"` cannot forge tenant `acme`'s user
    `alice`, it simply fails to construct. Single colons in `user_id` (SSO
    ids like `google:12345`, `auth0|abc`) remain fine; tenant ids are plain
    identifiers (`acme`, an org UUID) and so are colon-free.

## End-to-end: tenant flows from token to storage

```python
# 1. Your app authenticates the user and knows their org
caller = CallerContext(user_id="alice", tenant_id="acme")

# 2. Agent-side isolation is automatic
reply = await agent.chat("What did we discuss?", session_id=sid, caller=caller)

# 3. Server-side: the user's JWT (caller.bearer_token) carries the tenant
#    claim, AuthMiddleware extracts it, and guards, rate limits, caching
#    and audit key on it
caller = CallerContext(user_id="alice", tenant_id="acme", bearer_token=alice_jwt)
reply = await agent.chat("Which renewals are due?", session_id=sid, caller=caller)
```

## See Also

- [`examples/mcp/multi_tenant_agent.py`](https://github.com/promptise-com/foundry/blob/main/examples/mcp/multi_tenant_agent.py) — runnable: one agent serving two tenants, audience checks, caller-scoped caching, per-tenant tool lists
- [Authentication & Security](auth-security.md) — auth providers, guards, `ClientContext`
- [Multi-User Identity guide](../../guides/multi-user-identity.md) — end-to-end `CallerContext` flow
- [Approval Gates](approval-gates.md) — server-side human-in-the-loop
