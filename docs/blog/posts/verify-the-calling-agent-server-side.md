---
title: "Zero-trust: verify the calling agent, not its name"
description: "In a zero-trust mesh the receiving MCP server must verify the calling agent cryptographically, not trust a self-asserted name. This how-to verifies the…"
keywords: "verify the calling agent server-side, zero-trust agent authentication, RequireClientId guard, JwksAuth audience required, agent identity spoofing, per-agent authorization MCP"
date: 2026-07-16
slug: verify-the-calling-agent-server-side
categories:
  - Identity
---

# Zero-trust: verify the calling agent, not its name

In a zero-trust agent mesh you must **verify the calling agent server-side** — cryptographically, on the resource that receives the call — because a self-asserted agent name in a request body is trivial to forge. Agent A tells your billing server "I am `billing-bot`"; your server believes it; a compromised or misconfigured Agent B says exactly the same thing. If the only thing standing between a caller and `issue_refund` is a string it chose for itself, you have no security boundary at all — you have a naming convention.

<!-- more -->

Zero-trust has one rule: never trust, always verify, and verify at the resource. For AI agents that means the *receiving* MCP server does the work. It checks that the caller's credential was minted by an identity provider you trust, that it was minted for *this* resource, that it belongs to the *specific* agent you allow, and it records the verified subject so an auditor can prove who did what. This post wires that server-side half end to end. It is the counterpart to the outbound question — [how does an AI agent authenticate to an API?](how-does-an-ai-agent-authenticate-to-an-api.md) covers minting and presenting the credential; here we verify it.

## Why a self-asserted agent name is not identity

The failure mode is subtle because the happy path looks fine. Your agents each carry a name, they pass it along, tools log it, dashboards show it. Nothing is obviously broken — until you ask what stops a caller from claiming a name that isn't theirs.

- **A name in the payload is attacker-controlled.** Anything the client puts in the request body, the client can change. `agent_id="billing-bot"` is a suggestion, not a fact.
- **A shared secret proves membership, not identity.** If every agent in the fleet presents the same API key, a verified call tells you "someone in the fleet," never "which one." After an incident that distinction is everything.
- **Signature-only verification is replayable.** Even if you verify a real JWT signature, a token minted for the *analytics* service will validate at the *billing* service unless you also pin the audience. That is token substitution, and it is the single most common way a "we check the token" server is still spoofable. The [identity threat model](../../identity/security.md) treats it as a first-class attack.

The fix is to make identity a property the server *derives from cryptography*, never a property the caller *asserts*. That is three concrete checks: verify the token against the issuer's published keys with a required audience, authorize the specific agent, and audit the verified subject.

## What other frameworks do today

Be precise here, because the honest gap is narrower than "nobody else does this" — and more structural.

- **Agent orchestration frameworks — LangChain/LangGraph, CrewAI, AutoGen, LlamaIndex — ship no resource-server SDK for verifying a *remote* calling agent.** Their multi-agent handoffs address a peer by an in-process name or role; the "identity" of the agent you delegate to is a Python object or a routing key, not a verified credential. That is fine inside one trust boundary and offers nothing across one. AutoGen 0.4 does ship a distributed gRPC runtime that coordinates workers across processes and machines — but it routes messages between registered workers; it does not verify a calling agent's IdP token with a required audience, allow-list the specific agent per tool, and record the verified subject in a tamper-evident log.
- **General MCP server frameworks do verify bearer tokens.** FastMCP, for example, validates JWTs against a JWKS endpoint and can enforce issuer, audience, and scopes. Credit where due: that closes the signature-and-audience gap. What it does not bundle is the rest of the stack as one enforced layer — per-agent allow-listing and role checks as composable per-tool primitives, and a verified-subject, HMAC-chained audit as a drop-in middleware.

So the fair delta is not "they can't verify a token." It is that Promptise Foundry makes the *whole* server-side contract structural: audience is a **required** constructor argument (you cannot stand up signature-only verification by accident), per-agent authorization is a first-class `Guard`, and the tamper-evident audit of the verified subject is one line. Three checks that are easy to skip individually become one layer you compose deliberately. The [server auth & security reference](../../mcp/server/auth-security.md) documents each piece.

## Verify, authorize, audit — the three checks a resource server owes you

Here is the full server-side loop, and it is genuinely runnable on a laptop with no identity provider and no model key. It uses a local HS256 secret purely so the example is self-contained; the production swap to asymmetric IdP verification is the next section. What matters is the enforcement, and every check below is real:

```python
import asyncio

from promptise.mcp.server import (
    MCPServer,
    RequestContext,
    AuthMiddleware,
    JWTAuth,
    RequireClientId,
    HasRole,
    AuditMiddleware,
    TestClient,
)

auth = JWTAuth(secret="demo-signing-secret")
audit = AuditMiddleware(log_path="billing-audit.jsonl", signed=True, hmac_secret="demo-audit-key")

server = MCPServer(name="billing")
server.add_middleware(AuthMiddleware(auth))   # verify the credential
server.add_middleware(audit)                  # record the verified subject


# Authorize the SPECIFIC agent — not just "some authenticated caller".
@server.tool(auth=True, guards=[RequireClientId("billing-bot"), HasRole("refunder")])
async def issue_refund(ctx: RequestContext, invoice_id: str, amount: float) -> str:
    # ctx.client.subject is the cryptographically verified caller,
    # never a name lifted from the request body.
    return f"{ctx.client.subject} refunded {amount} on {invoice_id}"


async def main() -> None:
    # The authorised agent: valid token AND allow-listed id AND the right role.
    good = auth.create_token({"sub": "billing-bot", "roles": ["refunder"]})
    client = TestClient(server, meta={"authorization": f"Bearer {good}"})
    print(await client.call_tool("issue_refund", {"invoice_id": "INV-42", "amount": 9.99}))

    # A DIFFERENT agent with a perfectly valid token and the same role —
    # but not this agent. RequireClientId denies it: identity is verified
    # first, then authorised per-agent.
    other = auth.create_token({"sub": "scraper-bot", "roles": ["refunder"]})
    client2 = TestClient(server, meta={"authorization": f"Bearer {other}"})
    print(await client2.call_tool("issue_refund", {"invoice_id": "INV-42", "amount": 9.99}))

    # The audit recorded the VERIFIED subject of every call, in an HMAC chain.
    for e in audit.entries:
        print(e["status"], e.get("identity", {}).get("subject"), e["tool"])
    print("chain intact:", audit.verify_chain())


asyncio.run(main())
```

Run it and the point lands immediately. `billing-bot` succeeds. `scraper-bot` — which holds a **valid, correctly signed token** and even carries the `refunder` role — is denied:

```
ACCESS_DENIED  Client 'scraper-bot' is not allowed to call this tool
```

That denial is the entire thesis of this post. Authentication answered "is this a real, verified agent?" Authorization answered "is it *this* agent?" A framework that only does the first will happily let any authenticated caller invoke `issue_refund`. `RequireClientId` (and `HasRole`, `HasAllRoles`, `RequireTenant` for the coarser cuts) makes per-agent authorization a declaration on the tool, not a branch you remember to write. And the audit line records `subject="billing-bot"` and `subject="scraper-bot"` against each call — so the tamper-evident log answers "which agent did this?" from the verified credential, not a client-supplied string.

## In production: verify against your IdP with JwksAuth

The only thing that changes for production is the provider. Swap `JWTAuth` (shared secret) for `JwksAuth`, which verifies the agent's token against your identity provider's *published* keys — no shared secret, and IdP key rotation needs no redeploy because the key set is re-fetched on demand:

```python
from promptise.mcp.server import MCPServer, AuthMiddleware, JwksAuth, RequireClientId, AuditMiddleware

server = MCPServer(name="billing")

# Verify tokens THIS IdP issued for THIS resource. `audience` is mandatory —
# omit it and JwksAuth raises, because signature-only verification would accept
# a token minted for any other resource of the same IdP (token substitution).
auth = JwksAuth.from_discovery(
    issuer="https://login.microsoftonline.com/<tenant>/v2.0",
    audience="api://billing",
)
server.add_middleware(AuthMiddleware(auth))
server.add_middleware(AuditMiddleware(log_path="billing-audit.jsonl", signed=True))


@server.tool(auth=True, guards=[RequireClientId("billing-bot")])
async def issue_refund(ctx, invoice_id: str, amount: float) -> str:
    # ctx.client.subject / issuer / audience — all cryptographically verified.
    return f"Refunded {amount} on {invoice_id}"
```

That required `audience` is the anti-replay control. Because the billing server only accepts tokens whose `aud` claim is `api://billing`, a token the analytics agent was legitimately issued for `api://analytics` is worthless here even though the signature is valid and the issuer is the same. `JwksAuth` also rejects a token that claims `HS256` (algorithm confusion) and, with `issuer` pinned, tokens from any other issuer. The [end-to-end identity guide](../../identity/guide.md) wires exactly this inbound-verification section — mint, present, verify, audit — as a two-server walkthrough.

Now the loop is closed and zero-trust holds at the resource: the credential is checked against your directory, the *specific* agent is authorized per tool, and every call is written to an HMAC-chained log keyed on the verified subject — which is what makes fleet-wide attribution trustworthy after the fact.

## Frequently asked questions

### How do I verify the calling agent server-side instead of trusting its name?

Put an `AuthMiddleware` on the MCP server backed by `JwksAuth` (production) or `JWTAuth` (local). It verifies the caller's JWT against the issuer's keys with a required audience and populates `ctx.client.subject` from the token's `sub`. Your tool never reads an agent name from the request body — it reads the cryptographically verified subject. Then a `RequireClientId(...)` guard authorizes the specific agent.

### Isn't verifying the token signature enough?

No. A valid signature proves the token came from the issuer, not that it was minted for *your* resource. Without a required `audience`, a token issued for a different service of the same IdP will validate — that is token substitution. `JwksAuth` makes `audience` a required argument and raises if you omit it, so signature-only verification is not something you can ship by accident.

### What's the difference between authentication and per-agent authorization here?

Authentication (`AuthMiddleware` + `JwksAuth`) establishes that the caller is a real, verified agent. Per-agent authorization (`RequireClientId`, `HasRole`, `RequireTenant`) establishes that it is the *right* agent for this tool. In the runnable example, `scraper-bot` passes authentication with a valid token and even holds the required role, yet `RequireClientId` still denies it — verified identity and granted access are two separate decisions.

### Do agent orchestration frameworks do this for me?

Not at the resource layer. LangGraph, CrewAI, AutoGen, and LlamaIndex address peer agents by in-process name or role and ship no SDK to verify a *remote* calling agent's credential. General MCP server frameworks like FastMCP do verify bearer tokens (JWKS, issuer, audience, scopes); Promptise adds per-agent allow-list guards and a verified-subject, tamper-evident audit as one enforced layer on top of that verification.

## Next steps

Stand up server-side verification by walking the zero-trust inbound section of the [end-to-end identity guide](../../identity/guide.md) — it verifies the caller with `JwksAuth` (audience required), authorizes the specific agent with `RequireClientId`, and records the verified subject in a signed audit. Then read the [server auth & security reference](../../mcp/server/auth-security.md) for the full guard and middleware catalog, and the [identity threat model](../../identity/security.md) to see exactly which spoofing and replay attacks each check closes. Building the fleet these servers protect? Start by giving [each AI agent its own identity, not a shared key](give-each-ai-agent-its-own-identity.md) — verification on the server is only as meaningful as the identities on the callers.
