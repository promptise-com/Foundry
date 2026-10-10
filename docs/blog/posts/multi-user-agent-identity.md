---
title: "Pass User Identity Through Agents to MCP Tools"
description: "The concrete implementation query for anyone building a real multi-user product: how does Alice's identity reach the tool server without the client asserting…"
keywords: "multi-user agent identity, CallerContext, propagate user identity to MCP server, per-user data isolation agents, tenant isolation MCP, bearer token propagation"
date: 2026-07-16
slug: multi-user-agent-identity
categories:
  - Identity
---

# Pass User Identity Through Agents to MCP Tools

Getting **multi-user agent identity** right is the difference between a demo and a product your auditors will sign off on. The moment two customers share one agent, you have to answer a hard question: when Alice asks "show my invoices," how does *her* identity reach the tool server so it returns *her* data and not Bob's — without the client simply asserting whatever roles it likes? This post traces the whole path in Promptise Foundry: from a per-request `CallerContext` in your app, to a bearer token on the wire, to server-side JWT extraction and role/tenant guards. By the end you'll be able to wire it end to end and know exactly where the trust boundary sits.

<!-- more -->

## Two identities: the human principal and the acting agent

Multi-user systems actually carry two distinct identities, and conflating them is where most designs go wrong.

- **The human principal** — Alice, the logged-in user whose request the agent is currently serving. This is what scopes data access, memory, and per-user isolation. In Promptise it's a `CallerContext`.
- **The acting agent** — the non-human service identity of the process itself, answering "which agent did this?" for attribution and least privilege. That's [Agent Identity](../../identity/overview.md), a separate, verifiable service-account-style identity you can attach with `AgentIdentity`.

They solve different problems. Agent Identity says *which agent acted*; `CallerContext` says *on whose behalf*. A refund tool might require both: a trusted `billing-bot` agent **and** a caller with the `finance` role. For the rest of this post we focus on the human principal, because that's the isolation guarantee SaaS teams are usually evaluating. If you're weighing the broader picture of service accounts, tokens, and delegation, the [AI Agent Identity & Authentication guide](ai-agent-identity.md) covers how the two layers fit together.

## How bearer token propagation carries a user to your tools

`CallerContext` is the object you attach to every request. It has five fields, and — this is the important part — only one of them crosses the network:

```python
from promptise import CallerContext

alice = CallerContext(
    user_id="user-alice-001",                    # agent-side identity
    bearer_token="eyJhbGciOiJIUzI1NiIs...",      # the JWT — the ONLY thing sent to MCP servers
    roles={"analyst", "viewer"},                 # agent-side logic only
    scopes={"read", "write"},                    # agent-side logic only
    tenant_id="acme",                            # agent-side; also for local isolation
    metadata={"team": "finance", "plan": "pro"},
)
```

When you pass `caller=alice` into an invocation, Promptise:

- stores the `CallerContext` in an async-safe contextvar for the life of the request,
- sends `bearer_token` as an `Authorization: Bearer <token>` header on **every** tool call the invocation makes to an HTTP MCP server, over an MCP session opened for that token and never shared with another user,
- scopes memory search, conversation history, and the semantic cache to `user_id`.

Note what does **not** cross the wire: `roles`, `scopes`, and `tenant_id`. Those drive *agent-side* decisions in your process. The server never trusts them, because a client asserting its own roles is not authentication — it's a suggestion. The server derives roles and tenant from the signed JWT instead. This is **bearer token propagation** done the boring, correct way.

## Enforce roles and tenant isolation MCP-side with guards

Because only the JWT crosses the boundary, the MCP server is where authorization actually happens. `AuthMiddleware` validates the token, extracts the subject, roles, scopes, and tenant claim, and builds a trusted `ClientContext`. Per-tool guards then decide who may call what.

```python
from promptise.mcp.server import (
    MCPServer, AuthMiddleware, JWTAuth,
    HasRole, RequireTenant, RequestContext,
)

# require_tenant makes "every request must resolve a tenant" a server-wide invariant
server = MCPServer("billing-api", require_tenant=True)

# Same secret as your token issuer; tenant is read from the "org" JWT claim
server.add_middleware(AuthMiddleware(JWTAuth(secret="your-jwt-secret"), tenant_claim="org"))

@server.tool(auth=True, guards=[HasRole("analyst"), RequireTenant()])
async def list_invoices(ctx: RequestContext) -> list[dict]:
    """Return invoices for the authenticated caller's tenant only."""
    tenant = ctx.client.tenant_id          # from the signed token, not the client's word
    user = ctx.client.client_id            # JWT 'sub'
    return await db.invoices(tenant_id=tenant, owner=user)
```

Two things make this **tenant isolation MCP**-side rather than a hope:

- `RequireTenant()` refuses any call that doesn't resolve a tenant, and `require_tenant=True` makes that a server invariant so you can't forget it on one tool.
- `ctx.client.tenant_id` comes from the `org` claim inside the validated JWT. A client that hand-edits its roles or tenant gets a signature failure, not elevated access.

The `roles=["analyst"]` shorthand on `@server.tool()` is just sugar for a `HasRole` guard, and when a guard denies, the error explains *why* — which roles were required versus what the client presented (a tenant denial names only the caller's own tenant, never the others that are allowed). The full guard catalog (`RequireAuth`, `HasRole`, `HasAllRoles`, `HasScope`, `RequireTenant`, `HasTenant`, and custom guards) and the enrichment hooks live in the [MCP auth & security reference](../../mcp/server/auth-security.md). If you haven't stood up token validation yet, the walkthrough in [JWT Authentication for MCP Servers](jwt-authentication-for-mcp-servers.md) is the fastest way to get a signed, verifiable token flowing.

## Wire CallerContext end to end

Here's the client half — a runnable agent that carries Alice's identity into a tool call. The agent discovers the MCP tools with its own service token, and because `caller=alice` is set, every tool call during that invocation carries `alice.bearer_token` instead.

```python
import asyncio
import os
from promptise import build_agent, CallerContext
from promptise.config import HTTPServerSpec

async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={
            "billing": HTTPServerSpec(
                url="https://mcp.internal/billing/mcp",
                # The agent's own credential: opens the connection and discovers tools.
                bearer_token=os.environ["BILLING_SERVICE_TOKEN"],
            ),
        },
        instructions="You are a billing assistant. Use tools to answer questions.",
    )

    alice = CallerContext(
        user_id="user-alice-001",
        bearer_token="eyJhbGciOiJIUzI1NiIs...",   # Alice's signed JWT
        roles={"analyst"},
        tenant_id="acme",
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "List my open invoices."}]},
        caller=alice,   # identity for THIS request only
    )
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

Serving a different user is just a different `CallerContext` on the next call — no re-wiring, no shared mutable state. Concurrent invocations for different users are safe on one agent: each token gets its own MCP session, so Alice's token never rides on Bob's request. Under the hood, the agent's `list_invoices` call arrives at the server with Alice's JWT, `AuthMiddleware` validates it, `RequireTenant` and `HasRole` pass, and the handler reads `acme` from the token. The end-to-end sequence — with a field-by-field table of what crosses the wire and what doesn't — is documented in the [Multi-User Identity guide](../../guides/multi-user-identity.md).

One detail worth knowing for orchestration: the caller **survives cross-agent delegation**. If an orchestrator running as Alice delegates to a peer with `ask_peer` or `broadcast`, the peer inherits the ambient `CallerContext` unless you pass an explicit one, so cache scoping, memory, and audit tags all stay attributed to Alice rather than running as some anonymous system principal.

## Per-user data isolation for agents

Server-side guards protect your tools, but the agent process itself also holds per-user state — conversation history, long-term memory, and cached responses. Attaching a `CallerContext` gives you **per-user data isolation agents** can rely on without bespoke plumbing:

```python
from promptise import build_agent, CallerContext
from promptise.conversations import SQLiteConversationStore
from promptise.memory import ChromaProvider
from promptise.cache import SemanticCache

agent = await build_agent(
    model="openai:gpt-5-mini",
    servers=srv,
    conversation_store=SQLiteConversationStore("chat.db"),
    memory=ChromaProvider(persist_directory="./memory"),
    cache=SemanticCache(),
)

# Same question, two users — completely separate history, memory, and cache
await agent.chat("What did I ask last time?", session_id="s-alice", caller=alice)
await agent.chat("What did I ask last time?", session_id="s-bob",   caller=bob)
```

With `caller` set, conversation stores enforce session ownership (Alice can't read Bob's sessions), memory search is scoped to `user_id`, and semantic-cache entries are keyed per user — so one tenant never gets served another tenant's cached answer. The semantic cache still delivers its usual 30–50% cost reduction; isolation and savings are not a trade-off here.

## When an API gateway is the better fit

Promptise's approach is right when the agent itself needs identity — to isolate memory and cache per user, to make role- and tenant-aware decisions in-process, and to propagate a principal across delegation hops. It is not the only place to enforce access.

If your tools are plain HTTP services and you already run a mature **API gateway or service mesh** (Kong, Envoy, an OAuth2 proxy) that terminates auth and enforces tenancy at the edge, keep doing that — it's a well-trodden path and your platform team already operates it. Reach for `CallerContext` and MCP guards when the *agent* is the thing that must reason about who it's acting for, or when the same identity has to flow through several agents and tool servers coherently. Many teams run both: the gateway for coarse network-level policy, and MCP guards for fine-grained, per-tool authorization the agent can't route around.

## Frequently asked questions

### Does the MCP server trust the roles in CallerContext?

No. The `roles`, `scopes`, and `tenant_id` on `CallerContext` are used only for agent-side logic in your own process. Across the wire, the server receives only the `bearer_token` and derives roles and tenant from the validated JWT claims. A client cannot elevate itself by editing those fields.

### How do I propagate user identity to an MCP server?

Attach a `CallerContext` with a valid `bearer_token` to each invocation via `caller=`. Promptise sends that token as an `Authorization: Bearer` header on every MCP call the agent makes. On the server, `AuthMiddleware(JWTAuth(...))` validates it and exposes the identity on `ctx.client` for your guards and handlers.

### What's the difference between CallerContext and AgentIdentity?

`CallerContext` is the human principal the agent is currently serving — it scopes data and per-user isolation. `AgentIdentity` is the agent's own service identity for attribution and least privilege. They're independent, and a sensitive tool can require both a trusted agent and an authorized caller.

## Next steps

Wire `CallerContext` end to end and ship per-user isolation your auditors will sign off on: give each request a caller, put a bearer token on the wire, and let `RequireTenant` and `HasRole` do the enforcing server-side. Start from the [Quick Start](../../getting-started/quickstart.md), then follow the field-by-field [Multi-User Identity guide](../../guides/multi-user-identity.md) to lock down the trust boundary.
