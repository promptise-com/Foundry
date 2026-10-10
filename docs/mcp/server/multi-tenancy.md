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
| Tool access | `RequireTenant` / `HasTenant` guards, or the server-wide `require_tenant` invariant |
| Job queue | `MCPQueue` jobs record the submitting client and tenant; `queue_status`, `queue_result`, `queue_cancel` and `queue_list` only show a caller its own jobs, and the admin role widens that to its own tenant only ([Job ownership](queue.md#job-ownership)) |

`SessionState` needs no tenant prefix: it is keyed by the live transport
session, which is connection-scoped and therefore cannot be shared across
tenants.

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

# 3. Server-side: the agent's JWT carries the tenant claim,
#    AuthMiddleware extracts it, guards + rate limits + audit key on it
```

## See Also

- [Authentication & Security](auth-security.md) — auth providers, guards, `ClientContext`
- [Multi-User Identity guide](../../guides/multi-user-identity.md) — end-to-end `CallerContext` flow
- [Approval Gates](approval-gates.md) — server-side human-in-the-loop
