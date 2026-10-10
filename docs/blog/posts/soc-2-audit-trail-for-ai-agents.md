---
title: "How to Prove What Your AI Agent Did for a SOC 2 Audit"
description: "Step-by-step: add AuditMiddleware with a managed PROMPTISE_AUDIT_SECRET, capture attributable tool calls, keep PII out by default, and produce a verifiable…"
keywords: "soc 2 audit trail for ai agents, prove what an ai agent did for compliance, soc 2 evidence for llm agents, ai agent audit logging setup, agent access log soc 2"
date: 2026-07-16
slug: soc-2-audit-trail-for-ai-agents
categories:
  - Compliance & Audit
---

# How to Prove What Your AI Agent Did for a SOC 2 Audit

A **soc 2 audit trail for ai agents** is the artifact you reach for the moment a reviewer stops asking about your policies and asks the pointed question: "for this refund your agent issued, show me who was allowed to call that tool, who actually called it, and prove the record wasn't edited after the fact." Pointing at your LLM provider's SOC 2 badge doesn't answer it — that badge attests to *their* controls, not to evidence of *your* agent's actions. This post is the step-by-step setup: capability-based per-tool guards so only the right agent can take a sensitive action, `AuditMiddleware(signed=True)` with a managed `PROMPTISE_AUDIT_SECRET` so every attempt becomes an attributable, tamper-evident record, PII kept out by default, and a `verify_chain()` report an auditor will accept.

<!-- more -->

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## What a SOC 2 reviewer asks for — and what a vendor's badge can't answer

SOC 2 evidence for LLM agents comes down to two questions a reviewer will keep circling back to, and they map to two different controls:

- **Access control** (Common Criteria CC6.x): was this consequential action restricted to a principal that was *authorized* to take it? A reviewer wants to see least privilege enforced, not assumed.
- **Attribution and integrity** (CC7.x monitoring): for the action that happened, which verified identity performed it, and can you demonstrate the record of it wasn't altered or silently dropped?

The common mistake is to answer the second question with someone else's certification. Your model vendor's "SOC 2 Type II" report means an independent auditor examined *their* organization — their access management, their change control, their infrastructure. It is genuinely useful when you evaluate that vendor. It says nothing about the integrity of the specific log line your agent emitted when it called `refund`. When *your* auditor asks you to prove what your agent did, a third party's badge is evidence about the third party. You still have to produce evidence about your own system.

That is the gap this setup closes. You wire both halves — the access check and the verifiable record — into the same MCP server, so the proof is a byproduct of how the agent runs, not a spreadsheet you assemble after the incident.

## What other frameworks do today

Being precise here matters, because the popular stacks all produce *a* record of what an agent did. The honest gap isn't "they log nothing" — it's what that record is built to be.

- **LangChain / LangGraph** give you callbacks and LangSmith tracing for every run. That's rich, queryable telemetry — ideal for debugging and evals. The properties to state plainly: the trace store is a *mutable database*, and the identity attributes on a run are written by the emitting process, so the "who" is self-reported rather than checked against an identity provider. LangGraph checkpointers persist graph state so a run can resume or time-travel — genuinely valuable durable execution, but a checkpoint is *resumable state* in a mutable backend, not a cryptographically chained record whose edits are detectable.
- **CrewAI** can write execution logs (for example via `output_log_file`) and plugs into external observability tools. As shipped, it does not advertise a hash-chained, tamper-evident audit primitive with a verify step; integrity is left to whatever you point those logs at.
- **AutoGen** emits message and event logs for a run. Again useful for tracing behavior — and again, no built-in per-record signed chain you can independently verify.
- **FastMCP**, the closest MCP-server comparison, ships structured logging middleware for requests and responses. That's the right tool for operational visibility. What it does not ship today is a signed, hash-chained audit record with a `verify_chain()`-style primitive; tamper-evidence is bring-your-own — WORM storage, or hashing the records yourself.

None of that is wrong for its purpose. The shared shape is: they produce logs or traces, and leave *integrity* and *verified attribution* as something you bolt on with a second system. Promptise's edge isn't that others lack logging — it's that a per-action, tamper-evident, attributable trail is a **first-class, structural property of the record itself**, turned on as one middleware, rather than a metadata convention plus an integrity layer you assemble. And because access control lives in the same server as the audit, the "who was allowed" and "who did it" halves land in one place. For the long-form version of the traces-aren't-evidence argument, see [Why AI Agent Traces Aren't an Audit Trail (or SOC 2 Proof)](ai-agent-observability-vs-audit-trail.md).

## Set up least-privilege access and a signed audit trail

Here is the whole setup in one runnable file. It stands up a billing server, restricts `refund` to agents that hold a `billing-clerk` capability, lets an authorized clerk agent issue a refund, watches an analytics agent get **denied** for lacking the role, then shows both the allowed action *and* the rejected attempt landing in an HMAC-chained trail attributed to a verified principal — and catches an edit to that trail.

Every API here is real, and it runs in-process with `TestClient` — no network, no LLM key required:

```python
# soc2_audit.py — least-privilege access control + a verifiable audit trail
import asyncio
import json

from promptise.mcp.server import (
    MCPServer, AuthMiddleware, JWTAuth, AuditMiddleware, TestClient,
    RequestContext, HasRole,
)

SECRET = "rotate-me-in-prod"          # prod: PROMPTISE_AUDIT_SECRET from a vault

server = MCPServer(name="billing-api")

# 1. Verify the caller's JWT server-side. The principal is CHECKED, not asserted;
#    the tenant is read from the `tenant_id` claim onto ctx.client.tenant_id.
auth = JWTAuth(secret=SECRET)
server.add_middleware(AuthMiddleware(auth, tenant_claim="tenant_id"))

# 2. One HMAC-chained trail. include_args stays False, so amounts and order ids
#    (potential PII) never enter the evidence log.
audit = AuditMiddleware(log_path="billing-audit.jsonl", signed=True, hmac_secret=SECRET)
server.add_middleware(audit)


# Capability-based per-tool guard: only a billing-clerk agent may issue refunds.
@server.tool(auth=True, guards=[HasRole("billing-clerk")])
async def refund(order_id: str, amount: float, ctx: RequestContext) -> dict:
    """Issue a refund — a consequential action a SOC 2 reviewer scrutinizes."""
    return {"order_id": order_id, "refunded": amount, "by": ctx.client.subject}


async def main() -> None:
    # Two IdP-issued tokens in tenant "acme": one clerk, one read-only analyst.
    clerk = auth.create_token(
        {"sub": "billing-agent", "roles": ["billing-clerk"], "tenant_id": "acme"}
    )
    analyst = auth.create_token(
        {"sub": "analytics-agent", "roles": ["analyst"], "tenant_id": "acme"}
    )

    # Allowed: the clerk agent holds the required capability.
    ok = TestClient(server, meta={"authorization": f"Bearer {clerk}"})
    await ok.call_tool("refund", {"order_id": "A-1001", "amount": 49.0})

    # Denied: the analyst lacks billing-clerk. Least privilege, enforced
    # server-side — and the rejected attempt is recorded, not swallowed.
    denied = TestClient(server, meta={"authorization": f"Bearer {analyst}"})
    result = await denied.call_tool("refund", {"order_id": "A-1002", "amount": 5000.0})
    print("analyst refund ->", json.loads(result[0].text)["error"]["code"])   # ACCESS_DENIED

    # Two entries, both attributed to a VERIFIED principal (not a self-report):
    # the allowed action AND the denied attempt.
    for e in audit.entries:
        print(e["identity"]["subject"], e["tool"], "->", e["status"])
    print("chain valid:", audit.verify_chain())      # True

    # An insider edits the trail to hide who was refused...
    audit.entries[1]["identity"]["subject"] = "someone-else"
    print("chain valid:", audit.verify_chain())      # False — tamper detected


asyncio.run(main())
```

Running it prints exactly this:

```text
analyst refund -> ACCESS_DENIED
billing-agent refund -> ok
analytics-agent refund -> error
chain valid: True
chain valid: False
```

Two things make this a SOC 2 setup rather than a debug log. First, `HasRole("billing-clerk")` is a *capability-based per-tool guard*: access control is a property of the tool declaration, checked server-side before the handler runs, so no MCP client can talk its way past it. (The `roles=["billing-clerk"]` shorthand on `@server.tool()` builds the same guard.) Second, the denied attempt isn't dropped — it lands in the trail with `status: error`, the acting principal, and the reason. That is precisely the evidence a reviewer wants that your access control is *working*, including the calls it refused. The full guard catalog — `HasRole`, `HasScope`, `RequireAuth`, and friends — is on the [Authentication & Security](../../mcp/server/auth-security.md) page.

## Manage the secret with PROMPTISE_AUDIT_SECRET

The HMAC chain is only as trustworthy as the key behind it, so managing `PROMPTISE_AUDIT_SECRET` is the one operational step you can't skip. The `hmac_secret` resolves in a defined order: the constructor argument first, then the `PROMPTISE_AUDIT_SECRET` environment variable, and if neither is set, a signed log file refuses to start (`ValueError`), because a random per-process secret would leave a file nobody can verify. Only an in-memory chain, without `log_path`, falls back to a random secret, with a warning.

For an audit that has to hold up months later, that matters:

- **Set `PROMPTISE_AUDIT_SECRET` from your secrets manager**, not a literal in code. The demo's `"rotate-me-in-prod"` is a placeholder that exists so the example runs standalone.
- **Use the same key on every instance** so any node's log verifies anywhere — a reviewer shouldn't need to know which pod wrote a given line.
- **Keep the key out of reach of whoever can write the log file.** Tamper-evidence assumes the attacker can edit entries but *not* re-sign them. If the log-writer also holds the key, an insider can forge a clean chain, and the guarantee is gone.

That last point is the honest boundary of the primitive: it detects edits by anyone who lacks the secret. Pair it with write-once (WORM) or append-only storage and secret isolation, and you have the full control a reviewer expects.

## Keep PII out of the evidence by default

A SOC 2 (and adjacent HIPAA/GDPR) reviewer wants proof of *access*, not a second copy of the sensitive data — and an append-only, tamper-evident log is the worst place to stash erasable personal data, precisely because you can't quietly remove a line later. Promptise defaults to the safe posture:

- **`include_args` and `include_result` are `False` by default.** Tool arguments and results — the fields most likely to carry PII, like an order amount or a customer id — never enter the log unless you deliberately opt in. In the example, the refund amount stays out of the evidence entirely.
- **The `identity` block records descriptors only** — `subject`, `issuer`, `audience`, `roles`, and `tenant_id` — never the raw token or the full claim set. You get "which verified agent, in which tenant, called what" without leaking credentials into your logs.

Because the trail is PII-minimal by construction, a data-subject erasure request targets the *mutable* stores where personal data actually lives (memory, cache, the observability recorder), and you leave the tamper-evident access record intact and verifiable. That separation — erase the data, retain the proof of access — is the same pattern the cluster hub walks through in [One Audit Trail for SOC 2, HIPAA and the EU AI Act](ai-agent-compliance-audit-trail.md).

## Hand the auditor a verify_chain() report they accept

The payoff is that `verify_chain()` returns a plain boolean, so "prove it wasn't edited" becomes something the reviewer can watch you run — or run themselves with the secret. Each entry stores an HMAC-SHA256 over its own fields *plus* the previous entry's `prev_hash`, so verification is one deterministic pass. Map the fields straight onto what a reviewer asks:

| SOC 2 asks for | The field that answers it |
|---|---|
| **Integrity** — records weren't altered | `hmac` + `prev_hash` chain; `verify_chain()` returns `True`/`False` |
| **Attribution** — whose action was it | `identity` block: verified `subject`, `issuer`, `audience`, `roles`, `tenant_id` |
| **Completeness** — nothing silently dropped | Chain linkage — a deleted entry breaks the next entry's `prev_hash` |
| **Access control is enforced** — including refusals | The `status: error` entry for the denied attempt, attributed to its principal |

Because the check is a boolean, you can assert it in CI over a fixture trail, or wire it into a scheduled job that alarms the instant integrity breaks — and demonstrate a clean chain over the entire audit period on demand. That is the difference between "trust us, the log says so" and "here, check the math." Keep your mutable traces alongside it for debugging via the transporters on the [core observability](../../core/observability.md) page — you want both — and see the full audit field reference and recommended middleware ordering on the [MCP server observability](../../mcp/server/observability.md) page.

## Frequently asked questions

### Doesn't my LLM vendor's SOC 2 certification cover my agent?

No. A vendor's SOC 2 report attests that *that service organization* runs sound controls over a period. It is evidence about the vendor, not a per-record guarantee about the specific actions your agent took. When your own auditor — or a customer's — asks you to prove what your agent did, you need a verifiable record your system produced, which is exactly what `AuditMiddleware(signed=True)` plus `verify_chain()` gives you.

### How do I prove an agent's action log wasn't edited?

Turn on `AuditMiddleware(signed=True)` with a managed `PROMPTISE_AUDIT_SECRET`, then call `verify_chain()`. Each entry's HMAC covers its own fields plus the previous entry's hash, so a single deterministic pass returns `True` for an intact chain and `False` — localized to the break — if any entry was edited, deleted, or reordered. Without the secret, an attacker can't forge a valid chain.

### Does the trail capture attempts that were denied by a guard?

Yes. A capability-based guard like `HasRole("billing-clerk")` rejects an unauthorized call server-side, and that rejection is recorded as an entry with `status: error`, the acting principal, and the denial reason — not swallowed. Evidence that your access control refuses the wrong caller is as valuable to a reviewer as evidence of the allowed action.

### Do I have to log tool arguments to make the trail useful?

No, and you shouldn't by default. `include_args` and `include_result` are `False`, and the `identity` block records descriptors only — never the raw token. The trail answers "which verified principal called what, when, and with what outcome" without becoming a second copy of the sensitive payload. Enable argument capture only where the compliance benefit clearly outweighs the privacy cost.

### Where do I set the HMAC secret in production?

Set `PROMPTISE_AUDIT_SECRET` from your secrets manager, or pass `hmac_secret=` explicitly. Use the same key across every instance so any node's log verifies anywhere, and keep it out of reach of whoever can write the log file — otherwise a tamper *and* re-sign becomes possible and the guarantee is lost.

## Next steps

Follow the setup above to generate an auditor-ready trail today: add `AuthMiddleware(JWTAuth(...), tenant_claim=...)`, guard your sensitive tools with `HasRole(...)`, add `AuditMiddleware(signed=True)` with a managed `PROMPTISE_AUDIT_SECRET`, and run `verify_chain()` on a schedule. Start from the [MCP server observability](../../mcp/server/observability.md) reference for the full field list and middleware ordering, and the [Authentication & Security](../../mcp/server/auth-security.md) page for the complete guard catalog. To see one trail feed SOC 2, HIPAA, and the EU AI Act at once, read [One Audit Trail for SOC 2, HIPAA and the EU AI Act](ai-agent-compliance-audit-trail.md); for the deeper argument on why traces alone don't qualify, read [Why AI Agent Traces Aren't an Audit Trail (or SOC 2 Proof)](ai-agent-observability-vs-audit-trail.md).
