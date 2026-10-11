---
title: "Why AI Agent Traces Aren't an Audit Trail (or SOC 2 Proof)"
description: "Teams assume LangSmith/Logfire/OTel traces already serve as compliance evidence. They are mutable debugging telemetry, and the vendor's SOC 2 badge says…"
keywords: "ai agent observability vs audit trail, llm tracing vs audit log, langsmith soc 2 compliance, is observability enough for compliance, agent trace vs audit record"
date: 2026-07-16
slug: ai-agent-observability-vs-audit-trail
categories:
  - Compliance & Audit
---

# Why AI Agent Traces Aren't an Audit Trail (or SOC 2 Proof)

The question of **ai agent observability vs audit trail** usually starts with a comforting assumption: the traces your stack already emits — LangSmith runs, Logfire spans, OpenTelemetry exports — will double as compliance evidence when an auditor asks what your agent did. They won't. Those traces are mutable debugging telemetry, and a vendor's SOC 2 badge attests to that vendor's controls, not to the integrity of the log your own agent produces. This post draws the exact line between the two, names precisely what today's observability tools do and don't guarantee, and shows the one primitive a span export can't give you: a record whose integrity you can verify on demand.

<!-- more -->

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## Observability and an audit trail answer different questions

Observability and audit logging get conflated because they both "write down what happened." But they exist to answer different questions, and that difference decides whether either survives scrutiny.

Observability answers *why is my agent slow, expensive, or wrong?* It captures LLM turns, tool calls, token counts, latency, retries, and errors so you can debug and evaluate. It is designed to be mutable and lossy: you sample high-volume spans, drop or redact fields, retain data for 14 or 30 days, and re-tag runs during evaluation. Promptise ships exactly this — set `observe=True` and every turn, tool call, and token count is captured and routed to transporters like HTML, structured logs, Prometheus, or OTLP spans, as documented in [core observability](../../core/observability.md). It is genuinely useful. It is not evidence.

An audit trail answers a narrower, adversarial question: *for this sensitive action, which principal performed it, and can you prove the record wasn't edited after the fact?* When an agent calls `grant_admin`, `delete_records`, or a tool that moves money, the log itself becomes the disputed artifact. "Trust us, the trace says so" fails the moment the trace is exactly the thing under question — because anyone with write access can change it and nothing about the surviving data reveals the edit.

The defining property of an audit trail is integrity, not richness. That is the property observability tools were never built to provide.

## What a SOC 2 badge actually certifies

Teams reach for the vendor's SOC 2 badge to close this gap, and it's the wrong instrument for two reasons.

A SOC 2 report is an attestation by an independent auditor that a *service organization's* controls meet the Trust Services Criteria over a period — access control, change management, monitoring, and so on. When LangSmith, Logfire, or any tracing vendor says "SOC 2 Type II," an auditor examined **their** organization and infrastructure. That tells you the vendor runs a tidy shop. It says nothing about the integrity of the specific log line your agent emitted.

Second, even for data sitting inside that vendor's platform, SOC 2 is a statement about *controls and process*, not a per-record cryptographic guarantee you can independently reproduce. It does not let you hand an auditor a single record and prove that this exact entry is byte-for-byte what happened.

When *your* company goes through a SOC 2 audit — or a customer's auditor reviews you — they ask for evidence of *your* system's actions: "show me the log of privileged operations, and demonstrate it hasn't been altered." A vendor's badge cannot answer that question about your agent. The same gap shows up under the EU AI Act, whose Article 12 requires high-risk systems to *automatically* record events in a way that is tamper-resistant and attributable — see [EU AI Act Article 12: Logging Requirements for AI Agents](eu-ai-act-agent-logging.md). Debugging telemetry doesn't meet that bar either.

## What LangSmith, Logfire and OpenTelemetry do today

To be fair to these tools, here is precisely what they do — and the exact delta.

**OpenTelemetry** is an open standard for traces, metrics, and logs. A tool call becomes a span with attributes (`mcp.tool.name`, `mcp.request.id`, `mcp.client.id`, `mcp.status`) and timing, built to debug latency and errors across services. The spec defines no integrity mechanism: there is no per-span signature and no hash linking one span to the next. Spans are mutable in the SDK before export and then land in whatever backend you point them at, retained and sampled on your policy. Promptise emits OTel spans too — the `OTelMiddleware` and OTLP transporter on the [MCP server observability](../../mcp/server/observability.md) page — so this isn't a knock on OTel; it's the honest boundary of what telemetry is for. Partial overlap worth stating plainly: spans *do* carry attribution-like attributes such as client id and tool name. What's missing is any cryptographic guarantee that the stored span matches what actually happened.

**LangSmith** is LangChain's hosted tracing and evaluation platform. It captures runs — LLM calls, tool calls, chains — for debugging, evals, and monitoring, and supports feedback, datasets, and run annotation. Runs can be created, updated, and deleted through its API and UI, which is exactly right for an evaluation workflow. LangChain publishes a SOC 2 report for the service. As shipped today, LangSmith does not advertise a hash-chained, tamper-evident audit primitive; its value is queryable telemetry, not a verifiable record.

**Pydantic Logfire** is Pydantic's observability platform built on OpenTelemetry — traces, metrics, and logs with a Python-first developer experience. It sits in the same category: rich, queryable debugging telemetry over OTel semantics, not a per-record integrity mechanism.

The delta is consistent across all three. They give you excellent, searchable telemetry with some attribution baked into span attributes. None of them ships a keyed hash chain over the record, tamper detection, or a verify primitive you can run to prove a specific entry wasn't edited. Promptise's edge isn't "these tools lack logging" — it's that Promptise makes verifiable integrity a **first-class, structural property of the record itself**, rather than something you hope your backend enforces.

## The primitive a span can't give you: a verifiable hash chain

A tamper-evident audit log doesn't try to *prevent* edits — file permissions and write-once storage do that. It makes edits *detectable*. Promptise's `AuditMiddleware` does this by writing one JSON line per tool call in which every entry carries a keyed hash (HMAC-SHA256) computed over its own fields **plus the hash of the entry before it**. The chain starts from a fixed genesis hash (64 zeros) and each entry commits to the entire history preceding it.

That chaining is the whole point:

- Edit any field in any entry and its HMAC no longer matches — the chain breaks at that entry.
- Delete an entry and the next entry's `prev_hash` no longer lines up.
- Reorder two entries and the linkage is wrong from the swap onward.

Because the HMAC uses a secret key, an attacker can't recompute a valid chain without it. And because verification is a single deterministic pass, anyone with the secret can run `verify_chain()` and get a plain boolean back — the exact primitive a mutable span export cannot offer. When the caller authenticated with a JWT or JWKS provider, each entry also carries a verified `identity` block (`subject`, `issuer`, `audience`, `roles`, `tenant_id`) *inside* the chain, so "which principal did what" is both attributable and tamper-evident. The full behavior — configuration, what lands in each entry, and recommended middleware ordering — is on the [MCP server observability](../../mcp/server/observability.md) page.

## Turn the record into evidence in one line

Here is the difference made concrete. `AuditMiddleware` is a standard MCP server middleware: add it to the chain and every tool call becomes one signed JSON line. The example stands up an admin server, records two privileged calls, verifies the chain, simulates someone rewriting the record to hide who was escalated, then restores the exact original value to show the check is deterministic.

Every API here is real, and it runs in-process with `TestClient` — no network, no LLM key required:

```python
# trace_vs_audit.py — an editable trace vs. a verifiable audit record
import asyncio
from promptise.mcp.server import MCPServer, AuditMiddleware, TestClient

server = MCPServer(name="admin-api")

# AuditMiddleware writes one HMAC-chained JSON line per tool call.
audit = AuditMiddleware(
    log_path="audit.jsonl",
    signed=True,                       # each entry hashes the entry before it
    hmac_secret="rotate-me-in-prod",   # in prod: set PROMPTISE_AUDIT_SECRET
)
server.add_middleware(audit)


@server.tool()
async def grant_admin(user_id: str) -> dict:
    """Escalate a user to admin — exactly the action an auditor scrutinizes."""
    return {"user_id": user_id, "role": "admin"}


async def main():
    client = TestClient(server)
    await client.call_tool("grant_admin", {"user_id": "u-1001"})
    await client.call_tool("grant_admin", {"user_id": "u-2002"})

    # A trace is telemetry you can rewrite; this record proves it wasn't.
    print("chain valid:", audit.verify_chain())        # True

    # Someone edits the record to hide who was escalated.
    original = audit.entries[0]["tool"]
    audit.entries[0]["tool"] = "get_status"
    print("chain valid:", audit.verify_chain())        # False — tamper detected

    # Restore the exact original value and the chain verifies again.
    audit.entries[0]["tool"] = original
    print("chain valid:", audit.verify_chain())        # True


asyncio.run(main())
```

The `hmac_secret` resolves in a defined order: the constructor argument first, then the `PROMPTISE_AUDIT_SECRET` environment variable, and only if neither is set does it fall back to a random per-process secret (with a warning, because that can't verify the chain across restarts). In production, load it from your secrets manager so the same key verifies logs from every instance. Because `verify_chain()` returns a boolean, you can assert it in CI or wire it into a periodic job that alarms the instant integrity breaks — something you cannot do with a span you're free to edit.

## You need both, for different reasons

This isn't observability *or* audit — a serious agent platform runs both, because they serve different masters.

Keep your traces for what they're excellent at: debugging a slow tool, profiling token usage, running evals, and watching error rates. Reach for LangSmith, Logfire, or OTLP spans, and let Promptise feed them through the transporters on the [core observability](../../core/observability.md) page. Sample aggressively, retain briefly, redact freely — mutability is a feature here.

Add a tamper-evident audit trail specifically where the log is *evidence*: refunds, deletions, privilege changes, anything an auditor or an incident responder will scrutinize. There, the properties invert — you want completeness, attribution, and an integrity proof, not sampling. Turning one on is a single `add_middleware` call, and it captures everything, including calls later rejected by auth or a guard.

The mistake is using debugging telemetry as your compliance record. Draw the line deliberately, and you get the best of both: fast iteration from observability, and evidence you can defend from an audit trail.

## Frequently asked questions

### Is observability enough for compliance?

No. Observability tools like LangSmith, Logfire, and OpenTelemetry produce mutable debugging telemetry — spans and runs you can sample, edit, redact, or delete by design. Compliance evidence needs the opposite: a complete, attributable record whose integrity you can prove. A tamper-evident audit trail with a verify step covers what a trace export cannot.

### Doesn't my tracing vendor's SOC 2 certification cover this?

It covers the vendor's controls, not the integrity of your agent's log. A SOC 2 report attests that the service organization runs sound controls over a period; it isn't a per-record cryptographic guarantee, and it says nothing about the specific entries your agent emits. When your own auditor asks you to prove what your agent did, you need evidence about your system — a verifiable record — not a badge belonging to a third party.

### What's the difference between an LLM trace and an audit record?

A trace is telemetry for debugging and evaluation: LLM turns, tool calls, latency, and tokens, retained and mutable on your policy. An audit record is evidence: one signed entry per sensitive action, chained by an HMAC so any edit, deletion, or reorder is detectable, and attributed to a verified principal. Same raw event, opposite guarantees.

### How do I prove an agent log wasn't edited?

Use `AuditMiddleware(signed=True)` with a managed `PROMPTISE_AUDIT_SECRET`, then call `verify_chain()`. Each entry's HMAC covers its own fields plus the previous entry's hash, so verification is a single deterministic pass that returns `True` for an intact chain and `False` — pointing at the break — if any entry was touched.

## Next steps

Add `AuditMiddleware` to your MCP server and run `verify_chain()` — it's one `add_middleware` call to move a sensitive action from "the trace says so" to "verify it yourself." The [MCP server observability](../../mcp/server/observability.md) page shows the configuration and middleware ordering, and the [core observability](../../core/observability.md) page covers the traces you'll keep alongside it for debugging. To see how one verifiable trail feeds SOC 2, HIPAA, and the EU AI Act at once, read [One Audit Trail for SOC 2, HIPAA and the EU AI Act](ai-agent-compliance-audit-trail.md).
