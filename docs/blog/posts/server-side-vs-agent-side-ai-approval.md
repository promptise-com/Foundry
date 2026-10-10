---
title: "Approve in the Agent or the MCP Server? Or Both"
description: "Reconciles the two enforcement layers Promptise ships, which no current post puts side by side. Agent-side ApprovalPolicy on build_agent is fast, in-process…"
keywords: "server-side vs agent-side ai approval, mcp approval gate vs approvalpolicy, where to enforce agent tool approval, defense in depth agent approval, approvalpolicy build_agent"
date: 2026-07-16
slug: server-side-vs-agent-side-ai-approval
categories:
  - Approvals & HITL
---

# Approve in the Agent or the MCP Server? Or Both

The **server-side vs agent-side ai approval** decision is the one most teams get wrong by not knowing they had a decision to make — they wire a single human-in-the-loop hook, wherever the docs put it, and assume that one hook is doing two jobs it can't do at once. Promptise Foundry ships two distinct enforcement layers for a reason, and this post puts them side by side so you can map each sensitive tool to the layer it actually belongs in.

<!-- more -->

The short version: agent-side [`ApprovalPolicy`](../../core/approval.md) is a fast, in-process triage layer that lives with a specific agent and, paired with the [`AutoApprovalClassifier`](../../core/approval-classifier.md), keeps reviewers from drowning in low-risk calls. Server-side [`ApprovalGateMiddleware`](../../mcp/server/approval-gates.md) is a security boundary that sits on the tool itself and holds for *any* MCP client. They are not competitors. They share one `ApprovalHandler` protocol, both fail closed, and the interesting move is running both.

## "Should I approve this?" is two questions, not one

When you say a tool "needs approval," you are usually answering one of two different questions, and they have different owners.

**Question one is triage: is this call worth a human's attention at all?** Your agent might fire hundreds of tool calls an hour. The vast majority are reads, lookups, and idempotent fetches nobody needs to sign off on. If every one of them pages a human, you get alert fatigue — reviewers start rubber-stamping, and the one refund that mattered slips through in a batch of forty `get_balance` calls. This is a *volume* problem, and it belongs close to the agent, where the call is cheap to inspect and divert.

**Question two is security: is this tool safe to execute regardless of who called it?** A `refund` or `delete_records` tool is dangerous no matter which client reached it — your Promptise agent, someone's Claude Desktop, a teammate's script, a 3 a.m. cron job. This is a *boundary* problem, and it belongs on the tool, where it can't be routed around.

A single in-process hook forces you to answer both questions in one place, so you end up compromising on both: either you escalate everything (fatigue) or you move the check somewhere a second caller never passes through (a hole). The fix is to stop pretending it's one question.

## Two composable layers that share one handler

Promptise gives each question its own layer, and — this is the part that keeps the two from becoming two integrations — both speak the same `ApprovalHandler` protocol. A handler you write once (a Slack bot, a webhook, a review queue) plugs into either layer unchanged.

- **Agent-side** — [`ApprovalPolicy`](../../core/approval.md) on `build_agent()` wraps the agent's matching tool calls. Put the [`AutoApprovalClassifier`](../../core/approval-classifier.md) in front of your handler and it resolves the easy cases itself through an ordered hierarchy — deny rules, ask rules, allow rules, read-only auto-allow, an optional LLM classifier — and only escalates genuine judgement calls to the human. It supports reviewer edits (`modified_arguments`) because it re-binds arguments before dispatch.
- **Server-side** — [`ApprovalGateMiddleware`](../../mcp/server/approval-gates.md) sits in the MCP server pipeline in front of the handler, so the requirement is a property of the *tool*. Every client that ever calls it inherits the gate. It is approve-or-deny only (it won't run arguments a reviewer rewrote, because it can't guarantee the substitution is what executes).

The snippet below runs on nothing but `pip install promptise` — no API key, no network — and wires the *same* handler object into both layers at once:

```python
import asyncio
import secrets

from promptise import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalRule,
    AutoApprovalClassifier,
    CallbackApprovalHandler,
)
from promptise.mcp.server import (
    ApprovalGateMiddleware,
    AuthMiddleware,
    JWTAuth,
    MCPServer,
    TestClient,
)

# ONE handler, written once. It is the human-facing decision channel:
# approve small refunds, escalate anything bigger. In production this is your
# Slack bot, webhook, or review queue — the point is you reuse the same object
# in both enforcement layers below.
def review(request: ApprovalRequest) -> ApprovalDecision:
    amount = request.arguments.get("amount", 0)
    approved = amount <= 100
    return ApprovalDecision(
        approved=approved,
        reviewer_id="on-call",
        reason="under $100" if approved else "over limit — denied",
    )


shared_handler = CallbackApprovalHandler(review)  # implements ApprovalHandler

# ---- Layer 1: SERVER-SIDE security boundary -------------------------------
# The gate lives on the tool. It fires for every MCP client that ever calls
# refund — Promptise agent, Claude Desktop, a cron job — not just one agent.
auth = JWTAuth(secret="dev-secret")


def build_billing_server() -> MCPServer:
    server = MCPServer(name="billing")
    server.add_middleware(AuthMiddleware(auth))                 # identity first
    server.add_middleware(ApprovalGateMiddleware(shared_handler, timeout=5))

    @server.tool(auth=True, roles=["clerk"], requires_approval=True)
    async def refund(order_id: str, amount: float) -> dict:
        """Refund an order — the gate must approve before this body runs."""
        return {"order_id": order_id, "amount": amount, "status": "refunded"}

    return server


# ---- Layer 2: AGENT-SIDE triage in front of the SAME handler --------------
# The classifier answers the easy cases itself (deny/allow rules, read-only
# auto-allow) so a human only ever sees real judgement calls. Its fallback IS
# the shared_handler from above — one channel, reused.
classifier = AutoApprovalClassifier(
    deny_rules=[ApprovalRule(tool="drop_*", reason="never allowed")],
    allow_rules=[ApprovalRule(tool="get_*", reason="read-only lookup")],
    read_only_auto_allow=True,
    fallback=shared_handler,
)


async def main() -> None:
    # ---- Server-side: the boundary holds for any caller ----
    clerk = auth.create_token({"sub": "clerk-1", "roles": ["clerk"]})
    hdr = {"authorization": f"Bearer {clerk}"}
    client = TestClient(build_billing_server())

    ok = await client.call_tool("refund", {"order_id": "A-1", "amount": 50.0}, headers=hdr)
    print("server small refund:", ok[0].text)             # ran — under $100

    blocked = await client.call_tool("refund", {"order_id": "A-2", "amount": 5000.0}, headers=hdr)
    print("server big refund:  ", blocked[0].text[:44])   # APPROVAL_DENIED — fail closed

    # ---- Agent-side: triage diverts volume before it reaches a human ----
    read = await classifier.request_approval(
        ApprovalRequest(request_id=secrets.token_hex(8), tool_name="get_balance", arguments={})
    )
    print("agent get_balance:  ", read.approved, "via", read.trace.layer)

    listed = await classifier.request_approval(
        ApprovalRequest(request_id=secrets.token_hex(8), tool_name="list_orders", arguments={})
    )
    print("agent list_orders:  ", listed.approved, "via", listed.trace.layer)

    escalated = await classifier.request_approval(
        ApprovalRequest(
            request_id=secrets.token_hex(8), tool_name="refund", arguments={"amount": 5000.0}
        )
    )
    print("agent refund:       ", escalated.approved, "via", escalated.trace.layer)

    print(
        "triage stats:       ",
        f"read_only_allows={classifier.stats.read_only_allows}",
        f"fallback_denies={classifier.stats.fallback_denies}",
    )


asyncio.run(main())
```

It prints, in order: the small server-side refund runs; the big one comes back `APPROVAL_DENIED`; agent-side, `get_balance` is approved via the `allow_rule` layer, `list_orders` via `read_only`, and `refund` falls through to the shared handler which denies it — `read_only_allows=1 fallback_denies=1`. The same `shared_handler` made the human-facing decisions in both places, and neither layer ever silently allowed anything.

## Which layer does a given tool belong in?

Here is the decision, tool by tool. The rule is simple: **security at the server, triage in the agent, both when the tool is dangerous *and* high-volume.**

| If the tool is… | Put approval… | Why |
|---|---|---|
| Dangerous regardless of caller (`refund`, `delete_*`, `wire_transfer`) | **Server-side** gate | The requirement must survive being called by a second client, a script, or a cron job. It's a property of the tool. |
| High-volume, mostly low-risk, one agent | **Agent-side** policy + classifier | Triage the flood in-process; escalate only the calls a rule can't clear. Reviewers see signal, not noise. |
| Dangerous *and* high-volume | **Both** | Classifier trims the volume the human sees; the server gate guarantees the boundary for every caller. |
| Needs the reviewer to edit-then-run (change a recipient, cap an amount) | **Agent-side** policy | Only agent-side supports `modified_arguments`; the server gate is approve-or-deny and denies rewritten args by design. |
| Called only ever inside one app, never exposed as a shared MCP endpoint | **Agent-side** policy | An in-process control is complete here and cheaper — no need to gate the tool boundary. |

Two honest edges worth naming before you deploy: the server-side gate is approve-or-deny only (reviewer edits belong on the agent side), and the `PendingApprover` store is process-local today — pending calls that outlive the gate timeout are denied by default rather than surviving a restart. Both behaviors are documented in the [Approval Gates guide](../../mcp/server/approval-gates.md); neither is a silent surprise.

## What other frameworks do today

To be fair about the delta: human-in-the-loop is not a Promptise invention, and every major framework ships a real, working control. The precise difference is that each offers *one* in-process hook, which has to serve as both the triage layer and the boundary.

- **LangGraph** pauses with `interrupt()` inside a graph node; with a checkpointer configured it durably saves state and resumes when your app calls `Command(resume=...)`. Genuinely powerful — but the pause lives in the node that drives the tool, so it's one hook doing both jobs, and a second MCP client that reaches the tool another way never hits it.
- **CrewAI** has `human_input=True` on a `Task`, which prompts for review of that task's output on the console. Real and easy — but it's a flag on the task in the driver code, not a triage tier and not a property of the tool.
- **AutoGen** routes approval through a `UserProxyAgent` with `human_input_mode` (`ALWAYS`/`TERMINATE`/`NEVER`). It's approval as a property of the conversation topology you assemble, inside the agent process.

None of these frameworks *lacks* HITL, and with discipline you can wire any of them correctly. The exact delta is that they give you a single knob: turn it to `ALWAYS` and you get alert fatigue; turn it down and you narrow what a human ever sees, in the same place that also happens to be your only enforcement point. There's no separate, structural triage tier, and the hook lives with the process driving the agent rather than with the tool — so it doesn't travel when the tool becomes a shared endpoint. For a side-by-side of exactly where each framework's HITL executes, see [LangGraph vs CrewAI vs AutoGen: Where HITL Runs](langgraph-vs-crewai-vs-autogen-human-in-the-loop.md).

Promptise's edge isn't "we have approval and they don't." It's that the two jobs are *first-class and separate*: a caller-side triage layer and a tool-boundary security layer, each tunable on its own threat model, sharing one handler.

## Defense in depth: run both with the same handler

The question a single hook can't cleanly answer is this: *how do I cut reviewer fatigue on a high-volume agent **and** guarantee the tool is safe no matter which client calls it — through one decision channel?* Answer it with one hook and you must choose which half to sacrifice. Answer it with Promptise and you don't choose; you compose.

In the snippet, `shared_handler` is a single object. The `AutoApprovalClassifier` uses it as its `fallback`, so it only ever sees the calls the rules couldn't clear. The `ApprovalGateMiddleware` uses the *same object* as its enforcer on the `refund` tool, so any client — not just the triaging agent — is held to the boundary. Two enforcement points, one place to change your escalation logic.

And both fail closed, which is what makes the composition trustworthy rather than merely convenient. Agent-side, a handler that errors or a denial returns a `DENIED` result the LLM must adapt to; a timeout applies `on_timeout` (`"deny"` by default). Server-side, a denial raises a structured `APPROVAL_DENIED` error, no decision within `timeout` is denied by default, a reviewer's edited arguments are denied rather than run, and a crashed handler resolves to denial through the error pipeline. The server gate goes one step further into structural territory: if a tool declares `requires_approval=True` and no gate is installed, the server **refuses to build** — you cannot ship a declared approval that quietly never fires. That build-time invariant is its own story in [An MCP Approval Gate That Refuses to Ship Ungated](build-time-enforced-approval-gate-mcp.md).

Defense in depth here isn't two redundant checks; it's two *different* checks — triage for volume, boundary for safety — that you keep in sync by sharing one handler.

## Frequently asked questions

### What is the difference between an MCP approval gate and ApprovalPolicy?

`ApprovalGateMiddleware` is server-side: it enforces approval on the tool inside the MCP server pipeline, so the requirement holds for every client that calls the tool. [`ApprovalPolicy`](../../core/approval.md) is agent-side: it wraps a specific Promptise agent's own tool calls via `build_agent(approval=...)`. The gate is a security boundary on the tool; the policy is in-process triage for one agent. They share the same `ApprovalHandler` protocol, so a handler written for one works in the other.

### Where should I enforce agent tool approval — server or agent?

Enforce at the server when the tool must be safe regardless of caller (it's, or will be, a shared MCP endpoint). Enforce in the agent when you're triaging a high volume of one agent's calls, or when reviewers need to edit arguments before they run (`modified_arguments` is agent-side only). Enforce in both when a tool is dangerous *and* high-volume: the [`AutoApprovalClassifier`](../../core/approval-classifier.md) trims what a human sees, and the server gate guarantees the boundary for every caller.

### Do I have to write two handlers for defense in depth?

No — that's the point of the shared protocol. Write one `ApprovalHandler` (or wrap a callable in `CallbackApprovalHandler`), pass the same object to `ApprovalPolicy(handler=...)` / the classifier's `fallback` and to `ApprovalGateMiddleware(...)`. One decision channel, two enforcement points, one place to change escalation logic.

### Does the AutoApprovalClassifier replace the human reviewer?

No. It sits *in front* of your handler and resolves only the unambiguous cases — explicit allow/deny rules and read-only tools. Anything it can't clear (or an optional LLM classifier marks `"escalate"`) falls through to the human via the `fallback` handler. It reduces the volume a reviewer sees; it doesn't remove the reviewer from the decisions that matter.

### What happens on a timeout or a handler crash?

Both layers fail closed. Agent-side, a timeout triggers `on_timeout` (`"deny"` by default) and a handler exception is treated as denial. Server-side, no decision within `timeout` is denied by default (`on_timeout="allow"` is the explicit opt-out), and a crashed handler resolves to denial through the error pipeline. There is no configuration in which forgetting to respond silently allows a gated call.

## Next steps

Go through your sensitive tools and sort them: the ones that are dangerous no matter who calls them get a server-side gate; the high-volume, mostly-safe ones get an agent-side `ApprovalPolicy` with an [`AutoApprovalClassifier`](../../core/approval-classifier.md) to beat alert fatigue; the ones that are both get both. Then write **one** `ApprovalHandler` and share it across every layer. Start with the [agent-side Approval guide](../../core/approval.md) for the policy and classifier, wire the boundary with the [Approval Gates guide](../../mcp/server/approval-gates.md), and if some of those tools are shared MCP endpoints, read [LangGraph vs CrewAI vs AutoGen: Where HITL Runs](langgraph-vs-crewai-vs-autogen-human-in-the-loop.md) to see why the enforcement point — not the feature checkbox — is what decides your architecture.
