---
title: Build a Secure Multi-Tenant Agent Platform — Promptise Foundry
description: Build a multi-tenant SaaS AI platform with Promptise Foundry — per-tenant data isolation, role-based access, server-side human-in-the-loop approval for destructive tools, and a tamper-evident audit trail. The enterprise capstone, end to end.
keywords: multi-tenant AI platform, SaaS AI agent, tenant isolation, human-in-the-loop approval, four-eyes approval, MCP server security, enterprise AI agent, audit trail AI
---

# Build a Secure Multi-Tenant Agent Platform

This is the capstone guide: you'll build the AI backend for a **multi-tenant
SaaS product** — one server that many customer organizations share, where each
tenant's data is provably isolated, sensitive actions require a human's
sign-off, and every action lands in a tamper-evident audit log.

If you serve more than one customer from the same deployment, this is the shape
you need. It combines four capabilities that most frameworks leave to you to
bolt on (and get wrong): **tenant isolation**, **role-based access**,
**server-side approval gates**, and **tamper-evident audit** — enforced at the
tool boundary, so governance never depends on trusting the client.

## What You'll Build

A **billing-operations** MCP server for a SaaS company whose customers are
organizations (`acme`, `globex`, …). Support agents for each customer connect
through it to look up invoices and issue refunds. Requirements a real platform
has:

- **Tenant isolation** — Acme's agent can never see Globex's data, even if both
  agents share a service `client_id`.
- **Role-based access** — only the `billing` role may issue refunds.
- **Human-in-the-loop** — a refund blocks until a *different* human approves it
  (four-eyes), enforced by the server for any client.
- **Fair usage** — one tenant can't exhaust another's rate-limit quota.
- **Tamper-evident audit** — every call recorded with the acting tenant.

Everything below runs in-process with `TestClient` (no cloud, no keys) so you
can follow along; the [Deployment](../mcp/server/deployment.md) guide covers
serving it over HTTP.

## Concepts: three identities and two invariants

A multi-tenant agent system carries **three layers of identity**:

| Layer | Who | Carried by |
|-------|-----|-----------|
| **Tenant** | the customer *organization* | `ClientContext.tenant_id` (server) / `CallerContext.tenant_id` (agent) |
| **Principal** | the *user or service* acting | `client_id` (JWT `sub`) / roles + scopes |
| **Session** | the *conversation* | `session_id` |

And it rests on **two structural invariants** — properties enforced by the
framework, not conventions you have to remember:

1. **Tenant isolation is injective.** `tenant_id` enters *every* isolation key
   (cache, memory, conversations, rate limits, audit). Two tenants with the
   same user id land in disjoint keyspaces — cross-tenant leakage is
   structurally impossible, not merely avoided.
2. **Approval lives where the tool lives.** A tool declared
   `requires_approval=True` cannot execute — for *any* MCP client — until a
   human decides. Governance is a property of the tool, not a courtesy of the
   caller.

## Step 1 — A tenant-aware MCP server

Map each credential to a `(client_id, roles, tenant_id)`. Here we use API keys
for brevity; in production `JWTAuth` extracts the tenant from a claim
(`AuthMiddleware(JWTAuth(...), tenant_claim="org")`).

```python
from promptise.mcp.server import MCPServer, AuthMiddleware
from promptise.mcp.server._auth import APIKeyAuth

# require_tenant=True makes tenancy a server-wide INVARIANT: every tool
# authenticates AND must carry a tenant, or the call is denied.
server = MCPServer(name="billing-ops", require_tenant=True)

auth = APIKeyAuth(keys={
    # Two customer orgs deliberately share a service client_id — to prove
    # isolation is by TENANT, not by client_id.
    "sk-acme":     {"client_id": "svc-agent", "roles": ["billing"],  "tenant_id": "acme"},
    "sk-globex":   {"client_id": "svc-agent", "roles": ["billing"],  "tenant_id": "globex"},
    # A human reviewer for Acme (different principal than the caller).
    "sk-approver": {"client_id": "dana",      "roles": ["approver"], "tenant_id": "acme"},
})
server.add_middleware(AuthMiddleware(auth))
```

!!! tip "Why `require_tenant=True`"
    It turns "we always pass a tenant" from a hope into an invariant. Every
    tool — from decorators, routers, mounts, or OpenAPI import — is forced to
    authenticate and carries a `RequireTenant` guard. A token without the
    tenant claim is denied on *every* call. See
    [Multi-Tenancy](../mcp/server/multi-tenancy.md).

## Step 2 — Tenant-scoped tools with fair usage

Tools read `ctx.client.tenant_id` and scope their data access to it. A declared
`rate_limit` is enforced automatically, **per tenant** — so a noisy tenant
can't starve another's quota even when they share a `client_id`.

```python
from promptise.mcp.server import HasRole, RequestContext

@server.tool(rate_limit="60/min", guards=[HasRole("billing")])
async def get_invoice(invoice_id: str, ctx: RequestContext) -> dict:
    """Look up an invoice — scoped to the caller's tenant."""
    tenant = ctx.client.tenant_id
    return await db.get_invoice(invoice_id, tenant=tenant)
```

The `rate_limit="60/min"` bucket key is tenant-qualified and injective, so
Acme's traffic and Globex's traffic never share a bucket. (Declared limits are
enforced with no extra wiring — see [Caching &
Performance](../mcp/server/caching-performance.md).)

## Step 3 — Human approval on destructive actions

Refunds move money — they need a human. Declare the tool `requires_approval` and
install an `ApprovalGateMiddleware`. The gate is **fail-closed**: denied by
default on timeout, denied on a handler crash, and it evaluates the tool's
guards *before* it ever bothers a reviewer (so an unauthorized caller can't spam
approvers).

```python
from promptise.mcp.server import ApprovalGateMiddleware, PendingApprover

# PendingApprover blocks the call and exposes role-guarded admin tools
# (approvals_list / approvals_decide) for a human reviewer.
approver = PendingApprover(server, approver_role="approver")
server.add_middleware(ApprovalGateMiddleware(approver, timeout=300))

@server.tool(guards=[HasRole("billing")], requires_approval=True)
async def issue_refund(order_id: str, amount: float, ctx: RequestContext) -> dict:
    """Issue a refund — blocks until a human approves."""
    return await billing.refund(order_id, amount, tenant=ctx.client.tenant_id)
```

!!! warning "An ungated declaration refuses to build"
    If you declare `requires_approval=True` but forget the gate, the server
    **raises at build time** rather than silently letting the call through. A
    declared approval that doesn't enforce would be worse than none.

**Separation of duties is enforced.** `approvals_decide` rejects an *approval*
whose reviewer equals the original caller — you cannot approve your own refund,
even if you also hold `approver`. Denying your own is always allowed. See
[Approval Gates](../mcp/server/approval-gates.md) for the elicitation and
webhook approvers.

## Step 4 — Tamper-evident audit, tenant-stamped

Add `AuditMiddleware` for an HMAC-chained, append-only record. Each entry
carries the acting tenant, so per-customer forensics need no external join.

```python
from promptise.mcp.server import AuditMiddleware

server.add_middleware(AuditMiddleware(secret="${AUDIT_SECRET}"))
```

Approval outcomes flow through the same pipeline: a denial surfaces as a
structured `APPROVAL_DENIED` error (recorded like any error, with the approval
request id), and grants proceed to a normal audited call.

## Step 5 — The agent side: isolation follows the user

The server enforces isolation for *tools*. On the **agent** side, the same
`tenant_id` isolates memory, cache, and conversations — so a support agent
serving Acme's user `alice` can never surface Globex's data, even for a
same-named user. Each caller's `bearer_token` is what the billing server
authenticates, so one agent serves every tenant.

```python
from promptise import CallerContext, build_agent

agent = await build_agent(
    model="openai:gpt-5-mini",
    # The agent's own credential: opens the connection and discovers tools.
    # Tool calls are made with each caller's bearer_token instead.
    servers={"billing": {"url": "https://billing.internal/mcp",
                         "transport": "http", "bearer_token": service_jwt}},
    memory=ChromaProvider(persist_directory="./mem"),
    cache=SemanticCache(),
    conversation_store=SQLiteConversationStore("chat.db"),
)

acme_alice   = CallerContext(user_id="alice", tenant_id="acme", bearer_token=acme_alice_jwt)
globex_alice = CallerContext(user_id="alice", tenant_id="globex",
                             bearer_token=globex_alice_jwt)  # sees NONE of acme

# Same user_id, different tenants — fully isolated across cache/memory/sessions,
# and the billing server sees each caller's own token:
await agent.chat("What did we discuss?", session_id="s1", caller=acme_alice)
```

!!! warning "Never build the agent with one user's token"
    The server derives the tenant from the token it receives. A
    `CallerContext` without a `bearer_token` is sent with the spec's
    credential, so the server sees whoever that credential belongs to —
    not the caller. Give the spec a service credential and give every
    caller their own token. See
    [The user's token reaches the MCP server](../mcp/server/multi-tenancy.md#the-users-token-reaches-the-mcp-server).

One derivation — `CallerContext.isolation_key` — feeds every per-user surface,
so isolation is guaranteed at the scoping layer, not re-implemented per feature.
The `tenant_id` even rides through cross-agent delegation automatically.

## Step 6 — Run it end to end

Here's the whole platform in one runnable file, driven through `TestClient` (no
cloud, no key). It proves all four properties: per-tenant rate isolation, the
`require_tenant` invariant, four-eyes approval, and guards-before-approval.

```python
--8<-- "examples/mcp/tenancy_and_approval.py"
```

Run it:

```bash
python examples/mcp/tenancy_and_approval.py
```

You'll see: Acme and Globex get independent rate buckets despite a shared
`client_id`; an unauthenticated call is denied; a refund blocks until a
*different* approver releases it; and a wrong-role caller is rejected before any
approval is ever created.

## Security architecture summary

| Layer | Mechanism | What it guarantees |
|-------|-----------|--------------------|
| Transport | `JWTAuth` / `APIKeyAuth` | every caller is identified |
| Tenancy | `tenant_id` in every isolation key | provable cross-tenant isolation |
| Authorization | `RequireTenant`, `HasRole`, `HasScope`, `HasTenant` | who may call which tool |
| Human-in-the-loop | `requires_approval` + `ApprovalGateMiddleware` | destructive actions need a human (four-eyes) |
| Fair usage | tenant-qualified rate limits | no cross-tenant quota starvation |
| Audit | `AuditMiddleware` (HMAC-chained, tenant-stamped) | tamper-evident record of every action |
| Agent-side | `CallerContext.tenant_id` → `isolation_key` | per-tenant cache / memory / conversation isolation |

## What's Next

- [Multi-Tenancy](../mcp/server/multi-tenancy.md) — the tenancy model in depth
- [Approval Gates](../mcp/server/approval-gates.md) — elicitation, webhook, and pending approvers
- [Authentication & Security](../mcp/server/auth-security.md) — auth providers, guards, `on_authenticate`
- [Multi-User Systems](multi-user-systems.md) — per-user (not just per-tenant) identity flow
- [Deployment](../mcp/server/deployment.md) — serve it over HTTP with `promptise serve`
