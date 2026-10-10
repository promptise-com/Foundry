---
title: "AI Agent Audit Logging: Tamper-Evident by Design"
description: "Plain JSON logs are trivial to edit after an incident; this compares naive logging against an HMAC-chained audit trail where any deletion or edit breaks the…"
keywords: "AI agent audit logging, tamper-evident audit log, HMAC chained audit, MCP tool call logging, compliance logging for agents, who called which tool"
date: 2026-07-16
slug: ai-agent-audit-logging
categories:
  - Production
---

# AI Agent Audit Logging: Tamper-Evident by Design

AI agent audit logging is the difference between telling an auditor "trust us, the log says so" and handing them a record they can independently verify wasn't edited after the fact. When an autonomous agent calls a `refund` tool, a `delete_user` tool, or anything that moves money or data, you need a durable answer to "who called which tool, when, and with what outcome" — and you need that answer to survive an insider with write access to the log file. By the end of this post you'll know why plain JSON logs fail that test, what an HMAC-chained audit trail buys you, and how to turn one on in a single line with Promptise Foundry's `AuditMiddleware`.

<!-- more -->

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## Why plain JSON logs fail an audit

Most agent stacks log tool calls the obvious way: append a JSON object per call to a file or ship it to a log aggregator. That's fine for debugging. It falls apart the moment the log itself is the evidence.

The problem is that a flat JSON log has no internal integrity. Any line can be edited, reordered, or deleted, and nothing about the surviving lines reveals the change. Consider the aftermath of a disputed refund:

- An insider opens `audit.jsonl` and changes `"tool": "refund"` to `"tool": "get_status"` on one line. The record now says a harmless read happened.
- A line is deleted entirely. There's no gap, no missing sequence number, nothing that says a call ever existed.
- Two entries are swapped so a denied approval appears to come *after* the action it was supposed to gate.

A reviewer looking at the edited file sees valid JSON. There is no cryptographic reason to believe it reflects what actually happened. "The log says so" is not an answer a SOC2 reviewer accepts, because the log is exactly the thing under question.

## What "tamper-evident" means: an HMAC-chained audit log

A tamper-evident audit log doesn't try to stop edits — file permissions and write-once storage do that. Instead it makes edits *detectable*. Promptise does this with an HMAC-chained audit: every entry carries a keyed hash (HMAC-SHA256) computed over its own fields **plus the hash of the entry before it**.

That chaining is the important part. Each entry commits to the entire history that preceded it:

- Edit any field in any entry and its HMAC no longer matches — the chain breaks at that point.
- Delete an entry and the `prev_hash` of the next entry no longer lines up.
- Reorder entries and the linkage is wrong from the swap onward.

Because the HMAC uses a secret key, an attacker can't recompute a valid chain without that secret. Verification is a single pass: recompute each entry's HMAC and check it against the stored value and the previous hash. If everything lines up, the log is intact. If one line was touched, verification fails and points you at the break.

This is the same idea behind git commit hashes and certificate transparency logs, applied to MCP tool call logging.

## Enable AuditMiddleware in one line

Here's the feature in practice. `AuditMiddleware` is a standard MCP server middleware — you add it to the chain, and from then on every tool call becomes one signed JSON line. The example below stands up a billing server, records two refund calls, verifies the chain, then simulates an insider rewriting history so a refund looks like a read.

Every API here is real, and it runs in-process with `TestClient` — no network, no LLM key required:

```python
# audit_demo.py — a tamper-evident audit trail for MCP tool calls
import asyncio
from promptise.mcp.server import MCPServer, AuditMiddleware, TestClient

server = MCPServer(name="billing-api")

# One line turns on an HMAC-chained JSONL audit trail.
audit = AuditMiddleware(
    log_path="audit.jsonl",
    signed=True,                       # HMAC chain: each entry hashes the previous one
    hmac_secret="rotate-me-in-prod",   # or set PROMPTISE_AUDIT_SECRET
)
server.add_middleware(audit)


@server.tool()
async def refund(order_id: str, amount: float) -> dict:
    """Issue a refund for an order."""
    return {"order_id": order_id, "refunded": amount}


async def main():
    client = TestClient(server)
    await client.call_tool("refund", {"order_id": "A-1001", "amount": 49.0})
    await client.call_tool("refund", {"order_id": "A-1002", "amount": 12.5})

    # Every call is now one signed JSON line, linked to the one before it.
    print("chain valid:", audit.verify_chain())   # True

    # Simulate an insider editing the log to hide that a refund happened.
    audit.entries[0]["tool"] = "get_status"
    print("chain valid:", audit.verify_chain())   # False — tamper detected


asyncio.run(main())
```

The `hmac_secret` resolves in a defined order: the constructor argument first, then the `PROMPTISE_AUDIT_SECRET` environment variable, and if neither is set, a signed log file refuses to start (`ValueError`), because a random per-process secret would leave a file nobody can verify. Only the in-memory chain, without `log_path`, falls back to a random secret, with a warning. In production, set `PROMPTISE_AUDIT_SECRET` from your secrets manager so the same key verifies logs from every instance.

You'll usually add `AuditMiddleware` near the top of the chain so it captures *everything*, including calls that later get rejected by auth or a guard. The [Production Features](../../mcp/server/production-features.md) page shows the recommended middleware ordering — audit logging sits just under the dashboard layer, above auth and rate limiting.

## What lands in every entry: caller identity, tenant, tool, and outcome

An audit trail is only useful if each entry answers *who did what*. Every `AuditMiddleware` entry records:

- **`tool`** — the tool that was called.
- **`client_id`** and **`request_id`** — the caller and a unique id for correlating with other logs.
- **`status`** (`ok` or `error`), **`error`** when one occurred, and **`duration_s`**.
- **`timestamp`** — when the call happened.
- **`prev_hash`** and **`hmac`** — the chain linkage and signature.

When the request was authenticated with a JWT or JWKS auth provider, the entry also carries an **`identity`** block with the *verified* descriptors of the acting agent: `subject`, `issuer`, `audience`, `roles`, and `tenant_id`. Crucially, it records identity descriptors only — never the raw token or the full claim set, which could leak sensitive data into your logs.

That `tenant_id` field is what makes the trail usable in a multi-tenant SaaS. Every audited call is attributable to a tenant, so you can prove one customer's agent never touched another customer's tools. Tenant identity comes from a configurable JWT claim — see [Multi-Tenancy](../../mcp/server/multi-tenancy.md) for how `tenant_id` is populated and enforced as a first-class server invariant. For the end-to-end picture — auth, tenant isolation, and audit wired together — the [Secure Multi-Tenant Agent Platform](../../guides/secure-multi-tenant-platform.md) guide builds a complete governed deployment.

Argument and result capture are opt-in via `include_args` and `include_result`, both off by default because tool arguments and results frequently contain PII. Turn them on deliberately, and only where the compliance benefit outweighs the privacy cost.

## What SOC2 reviewers actually ask for

Compliance logging for agents isn't about volume — it's about answering a specific set of questions on demand. In practice, auditors ask for:

1. **Attribution** — for any sensitive action, which identity and tenant performed it. The `identity` block covers this.
2. **Completeness** — evidence that records weren't silently dropped. The HMAC chain makes deletions detectable.
3. **Integrity** — proof the record wasn't edited after the incident. `verify_chain()` is that proof, and anyone with the secret can run it.
4. **Non-repudiation of the log itself** — the reviewer doesn't have to take your word for it; they check the math.

Because `verify_chain()` returns a plain boolean, you can wire it into a periodic job that alarms the moment integrity breaks, or run it during an audit to demonstrate a clean chain over the whole period. For a broader view of the controls reviewers expect from an agent platform, the [Production AI Agent Checklist](production-ai-agent-checklist.md) walks through auth, rate limiting, approval gates, and audit as one set.

## When plain logging is enough

Tamper-evidence has a cost: you carry a secret, you keep the chain in order, and verification is only meaningful if the secret stays out of the attacker's reach. That trade-off isn't always worth it.

Reach for a plain structured log or your existing aggregator when:

- You're logging for **debugging and metrics**, not for evidence. If no one will ever dispute the record, HMAC chaining adds ceremony for little gain — use `StructuredLoggingMiddleware` or `MetricsMiddleware` instead.
- Your tools are **read-only and low-stakes**. A weather lookup doesn't need a tamper-evident trail.
- You already have **write-once, immutable storage** (WORM, an append-only ledger service) enforcing integrity at the storage layer, and you don't need per-record cryptographic verification on top.

`AuditMiddleware` shines specifically where the log is evidence: refunds, deletions, privilege changes, anything an auditor or an incident response will scrutinize later. If that's not your situation, skip it.

## Frequently asked questions

### What makes an audit log tamper-evident?

Each entry stores a keyed hash (HMAC-SHA256) computed over its own contents plus the previous entry's hash, forming a chain. Editing, deleting, or reordering any entry breaks the hash linkage, and a single verification pass detects exactly where. Without the secret key, an attacker can't forge a valid chain.

### Does audit logging capture who called which tool?

Yes. Every entry records the tool name, `client_id`, `request_id`, status, and duration. When the request was authenticated with a JWT or JWKS provider, it also includes an `identity` block with the verified `subject`, `issuer`, `audience`, `roles`, and `tenant_id` — attributing the call to a specific agent and tenant.

### Where do I set the HMAC secret in production?

Set `PROMPTISE_AUDIT_SECRET` from your secrets manager, or pass `hmac_secret=` explicitly. Use the same key across all instances so any node's log can be verified anywhere. If you set neither, a signed log file refuses to start; only an in-memory chain (no `log_path`) falls back to a random per-process secret, with a warning. Check a log file with `promptise audit verify audit.jsonl` or `verify_audit_log()`.

## Next steps

See the audit section of the [Production Features](../../mcp/server/production-features.md) page and enable `AuditMiddleware` on your MCP server — it's one `add_middleware` call to go from "trust us" to "verify it." If you're just getting started, the [Quick Start](../../getting-started/quickstart.md) gets a server running in a few minutes, and [Multi-Tenant AI Agents: Architecture for SaaS](multi-tenant-ai-agent.md) shows how audit, tenant identity, and isolation fit together for a real platform.
