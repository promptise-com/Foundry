---
title: "Secure Agent-to-Agent Authentication in Practice"
description: "When one agent hands work to another, 'the LLM did it' breaks attribution entirely. This deep-dive shows delegation over HTTP+JWT where the peer verifies the…"
keywords: "secure agent-to-agent communication, agent delegation authentication, ask_peer JWT, multi-agent trust, who delegated attribution, cross-agent auth"
date: 2026-07-16
slug: secure-agent-to-agent-communication
categories:
  - Identity
---

# Secure Agent-to-Agent Authentication in Practice

Secure agent-to-agent communication is the part of every multi-agent demo that quietly gets skipped: one agent delegates a task to another, the second agent acts, and when you open the log the only actor recorded is "the assistant." That is not an audit trail — it is a shrug. If a planner agent asks a payments agent to issue a refund, you need to know *which* agent asked, whether the callee was allowed to trust it, and where that decision landed in a tamper-evident record. By the end of this article you will know how Promptise Foundry propagates a verifiable caller identity through a delegation call and lands the answer to "who delegated to whom" in an HMAC-chained audit log.

<!-- more -->

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## Why "the LLM did it" breaks attribution

A single agent is easy to reason about: one identity, one set of permissions, one line in the audit log per action. The moment you introduce delegation, that clarity collapses. The callee sees a message; it does not automatically see who is behind the message. Left unaddressed, this produces three concrete failure modes:

- **Blind trust.** The callee executes whatever it is asked because it has no way to distinguish a legitimate peer from a prompt-injected impersonator.
- **Lost attribution.** The action is logged under the callee's own identity, so "who initiated this refund?" has no honest answer.
- **Over-broad permissions.** Because you cannot authorize *by caller*, you end up granting the callee every permission any caller might need.

This is the difference between multi-agent choreography that looks impressive in a video and a system you would let touch production data. Secure agent-to-agent communication is precisely the layer that turns the second category into the first, and getting agent delegation authentication right is how you build it.

## How cross-agent delegation actually works

In Promptise, you expose a peer agent to a primary agent by passing it through `cross_agents`. Each peer becomes a standard tool the primary agent can call during planning: an `ask_agent_<name>` tool for a single peer and, with `include_broadcast=True`, a `broadcast_to_agents` tool for fan-out to several peers in parallel. No new message bus, no extra service — peers are ordinary agent graphs.

The identity piece is what makes it *secure* rather than merely convenient. When the primary agent carries an `AgentIdentity`, every delegation injects a system message announcing the delegating agent's verified claims to the peer — the cheap descriptors (`agent_id`, issuer, roles), never a credential token. The peer now knows who is asking and can attribute, or refuse, accordingly.

```python
import asyncio
from promptise import build_agent, CallerContext, AgentIdentity
from promptise.cross_agent import CrossAgent


async def main():
    # A specialist peer that only does research.
    researcher = await build_agent(
        model="openai:gpt-5-mini",
        instructions="You research questions and return a short, sourced summary.",
    )

    # The planner delegates to the peer and carries a verifiable identity.
    planner = await build_agent(
        model="openai:gpt-5-mini",
        instructions="Plan the work. Delegate research to your peer when useful.",
        identity=AgentIdentity.auto(),   # local id, or Entra/AWS/GCP/SPIFFE/OIDC
        cross_agents={
            "researcher": CrossAgent(agent=researcher, description="Web research peer"),
        },
        observe=True,                    # timeline of every delegation + tool call
    )

    result = await planner.ainvoke(
        {"messages": [{"role": "user", "content": "Summarize the EU AI Act's tiers."}]},
        caller=CallerContext(user_id="alice", roles=["analyst"], tenant_id="acme"),
    )
    print(result["messages"][-1].content)

    await planner.shutdown()
    await researcher.shutdown()


asyncio.run(main())
```

The planner now calls `ask_agent_researcher(...)` on its own initiative, and the peer receives both the question and a signed statement of who is behind it. The `CallerContext` you pass — user, roles, tenant — is the per-request identity that flows alongside the delegation, so multi-tenant boundaries survive the hand-off instead of dissolving at it.

## Carrying a verifiable identity across the trust boundary

In-process delegation is only half the story. The interesting security question arises when the work leaves the process — when a peer needs to call a real tool server over HTTP. That is where an `AgentIdentity` stops being a label and becomes a credential the other side can *check*.

`AgentIdentity.auto()` picks a provider from the environment: a local identity on your laptop, or a verifiable one backed by Microsoft Entra, AWS STS, GCP metadata, SPIFFE, or OIDC in production. When you attach that identity to an agent and point it at MCP servers, Promptise mints a resource-scoped JWT per server, so a token minted for the billing API cannot be replayed against the CRM. The [Agent Identity overview](../../identity/overview.md) walks through the two tiers — local versus verifiable — and the end-to-end [identity guide](../../identity/guide.md) shows the full outbound-then-inbound flow. You can also present a credential by hand:

```python
identity = AgentIdentity.auto()
headers = identity.auth_header("api://billing")   # {"Authorization": "Bearer <jwt>"}
```

This is the honest version of the "ask_peer JWT" idea. The JWT is not what one agent whispers to another in-process; it is what an agent presents when it crosses a real network boundary to a resource that demands proof. Local delegation carries claims for attribution; the token carries authority for access. Keeping those two concerns distinct is what keeps multi-agent trust from becoming a single over-powered shared secret.

## Recording who delegated to whom in the audit log

Attribution is only real if a resource *verifies* the identity and *records* the verified result. On the server side, the Promptise MCP Server SDK checks the agent's IdP token against the issuer's published keys and writes each call into a tamper-evident chain. The [authentication and security reference](../../mcp/server/auth-security.md) covers every provider; here is the shape that answers "who delegated this?":

```python
from promptise.mcp.server import (
    MCPServer, AuthMiddleware, JwksAuth, RequireClientId, AuditMiddleware,
)

server = MCPServer(name="billing")

# `audience` is required — it stops a token minted for another resource being replayed.
auth = JwksAuth.from_discovery(
    issuer="https://login.microsoftonline.com/<tenant>/v2.0",
    audience="api://billing",
)
server.add_middleware(AuthMiddleware(auth))

# Each entry records the VERIFIED identity (subject / issuer / audience / roles)
# inside an HMAC chain — edit one line and the chain no longer verifies.
server.add_middleware(AuditMiddleware(log_path="billing-audit.jsonl", signed=True))


@server.tool(auth=True, guards=[RequireClientId("planner-bot", "reporting-bot")])
async def issue_refund(ctx, invoice_id: str, amount: float) -> str:
    # ctx.client.subject -> the IdP id of the calling agent
    # ctx.client.issuer  -> the IdP that vouched for it
    return f"Refunded {amount} on {invoice_id}"
```

Two things make this trustworthy. First, `RequireClientId` authorizes *by caller*, so `issue_refund` runs only for named agents — cross-agent auth becomes a per-tool decision instead of a global grant. Second, `AuditMiddleware(signed=True)` HMAC-chains every entry: each record commits to the one before it, so a delegated refund cannot be quietly deleted or backdated without breaking the chain. The line that lands in the log is not "the assistant issued a refund" — it is the verified subject, issuer, audience, and roles of the agent that actually made the call. That is the accountability layer most multi-agent demos skip.

## When a message bus is the better fit

Promptise's cross-agent delegation is deliberately lightweight: peers are in-process graphs, and the trust boundary you harden is the MCP call each agent makes to a real resource. That is the right model when your agents are co-located and you want attribution to ride on verifiable identity.

It is *not* the right model for every topology. If you need durable, replayable messaging between agents on different hosts — retries, dead-letter queues, back-pressure, ordering guarantees — a dedicated broker like NATS, Kafka, or a task queue is the better fit, and Promptise agents can sit on either end of it. Reach for those when the delegation itself must survive a crash and be replayed, not just be attributed. For request/response delegation where the security question is "can I prove who asked," secure agent-to-agent communication built on the cross-agent tools plus verifiable identity is the leaner, more honest choice.

## Frequently asked questions

### How does the callee know which agent delegated the task?

When the calling agent carries an `AgentIdentity`, each delegation injects a system message with the caller's verified claims — `agent_id`, issuer, and roles — so the peer can attribute the request. It is a set of identity descriptors, never a credential token, so nothing sensitive is shared just to establish who is asking.

### Is a JWT sent between the two agents directly?

No. In-process delegation passes identity *claims* for attribution. The JWT comes into play when an agent crosses a real network boundary to an MCP server, which mints a resource-scoped token per audience and verifies it with `JwksAuth`. This keeps access authority separate from in-process attribution.

### What stops someone from editing the audit log after a refund?

`AuditMiddleware(signed=True)` HMAC-chains entries, so each record commits to the previous one. Altering, reordering, or deleting any entry breaks the chain and fails verification, which makes tampering detectable rather than silent.

## Next steps

Wire a verifiable identity through a delegation call and watch attribution land in the audit trail — start from the [Quick Start](../../getting-started/quickstart.md), then follow the end-to-end [identity guide](../../identity/guide.md) to take an agent from a bare label to an IdP-backed credential your servers can verify. For the wider picture of how identity, tokens, and audit fit together across a fleet, read [AI Agent Identity & Authentication: The Complete Guide](ai-agent-identity.md) and the step-by-step walkthrough in [JWT Authentication for MCP Servers](jwt-authentication-for-mcp-servers.md).
