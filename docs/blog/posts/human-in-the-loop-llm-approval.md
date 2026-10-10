---
title: "Human-in-the-Loop Approval for AI Agent Tool Calls"
description: "Solves the real HITL problem: approving everything is unusable and approving nothing is unsafe. The layered AutoApprovalClassifier (deny rules, ask rules, allow rules…"
keywords: "human in the loop llm approval, ai agent approval workflow, auto-approve tool calls, llm tool call gating, hitl ai agents python"
date: 2026-07-16
slug: human-in-the-loop-llm-approval
categories:
  - Guardrails
---

# Human-in-the-Loop Approval for AI Agent Tool Calls

Human-in-the-loop LLM approval is the difference between an agent that can safely touch production and one that stays stuck in a demo. The hard part is not intercepting tool calls — it is deciding *which* ones actually need a human. Approve everything and your reviewers drown in `get_status` prompts until they rubber-stamp a `delete_database` by reflex. Approve nothing and one hallucinated argument wipes a table. By the end of this post you will have a six-layer classifier that auto-clears the safe calls, escalates only the risky ones, and fails closed when no one answers.

<!-- more -->

## Approving everything is unusable; approving nothing is unsafe

Most first attempts at an [AI agent approval workflow](../../core/approval.md) put a single gate in front of every tool. It works for a day. Then reality sets in:

- A research agent fires 40 read-only lookups per task. Each one pings a human.
- Reviewers learn that "approve" is almost always correct, so they stop reading.
- The one call that mattered — a refund, a deletion, a shell command — slides through on autopilot.

Alert fatigue is not a UX nitpick; it is the failure mode that makes HITL worse than useless. The fix is not a bigger queue. It is a *policy* that understands the vast majority of tool calls are boring and safe, and that human attention is a scarce resource you spend only on the calls that can hurt you.

## The six-layer AutoApprovalClassifier

Promptise Foundry ships that policy as the [`AutoApprovalClassifier`](../../core/approval-classifier.md). It wraps any approval handler you already have — a webhook, a queue, a callback — and runs each request through six ordered layers. The first layer to reach a verdict wins:

1. **Explicit deny rules.** Glob patterns, argument substrings, per-user filters, or async predicates that always deny. Put `delete_*` and `exec_shell` here. They run first, so no allow rule can override them.
2. **Ask rules.** The same matching machinery, but they always send the call to a human, whatever the layers below would say.
3. **Explicit allow rules.** The same again, but they always allow. First match wins.
4. **Read-only auto-allow.** Tools annotated `readOnlyHint=True`, or named `get_`, `list_`, `read_`, `search_`, `fetch_` and a dozen more prefixes, are cleared without prompting — unless the name also contains a destructive verb (`fetch_and_purge_cache`) or the server marks the tool destructive. This is where most of your volume disappears.
5. **Optional LLM classifier.** An async function that returns `("allow" | "deny" | "escalate", reason)`. Use it for the fuzzy middle — "does this argument look destructive?" — and only when the cheap rule layers didn't already decide.
6. **Human fallback.** Your existing handler. Reached only when nothing above it fired. This is the real human, and now they only see calls that genuinely earned their attention.

Because the classifier implements the same `ApprovalHandler` protocol as everything else, dropping it in front of your current approver is a one-line change. It also records every decision in `classifier.stats`, so you can see exactly which layer is doing the work and tune from evidence instead of guesswork.

## Auto-approve read-only tool calls, gate the destructive ones

Here is the whole thing wired into an agent. The classifier auto-approves reads, hard-denies deletions and dangerous shell arguments, and pages a human only for the ambiguous remainder. This is the pattern that turns HITL for AI agents in Python from a bottleneck into a background process.

```python
import asyncio
from promptise import (
    build_agent,
    ApprovalPolicy,
    ApprovalDecision,
    AutoApprovalClassifier,
    ApprovalRule,
    CallbackApprovalHandler,
)
from promptise.config import StdioServerSpec


async def human_reviewer(request):
    """Last-resort fallback: page a human. Here we deny for the demo."""
    print(f"REVIEW NEEDED: {request.tool_name}({request.arguments})")
    return ApprovalDecision(
        approved=False, reviewer_id="on-call", reason="no human confirmation"
    )


classifier = AutoApprovalClassifier(
    deny_rules=[
        ApprovalRule(tool="delete_*", reason="destructive"),
        ApprovalRule(tool="*", argument_contains="rm -rf", reason="dangerous shell"),
    ],
    allow_rules=[
        ApprovalRule(tool="get_*", reason="read-only report"),
    ],
    read_only_auto_allow=True,   # list_*, search_*, fetch_*, ... cleared automatically
    fallback=CallbackApprovalHandler(human_reviewer),
)


async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={"ops": StdioServerSpec(command="python", args=["ops_server.py"])},
        instructions="You are an ops assistant. Use tools to answer.",
        approval=ApprovalPolicy(
            tools=["*"],            # every tool call flows through the classifier
            handler=classifier,
            timeout=300,
            on_timeout="deny",      # fail closed
        ),
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "How many incidents are open?"}]}
    )
    print(result["messages"][-1].content)
    print(classifier.stats)         # which layer cleared each call
    await agent.shutdown()


asyncio.run(main())
```

The `ApprovalPolicy` decides *which* tools enter the gate (`tools=["*"]` means all of them); the `AutoApprovalClassifier` decides *what happens* once they are there. Keeping those two concerns separate is what makes the setup composable — you can widen or narrow the gate without touching your decision logic, and vice versa.

## Add an LLM classifier for the fuzzy middle

Rules cover the obvious cases. The interesting ones are calls like `update_config` or `run_query` whose *risk lives in the arguments*, not the name. That is what layer four is for. Give the classifier an async function and it runs only when the rule layers stay silent:

```python
async def is_destructive(request) -> tuple[str, str]:
    args = str(request.arguments).lower()
    if any(k in args for k in ("drop table", "truncate", "force")):
        return "deny", "argument looks destructive"
    if request.tool_name.startswith("update_"):
        return "escalate", "mutation — send to human"
    return "allow", "no risk signals"


classifier = AutoApprovalClassifier(
    deny_rules=[ApprovalRule(tool="delete_*")],
    allow_rules=[ApprovalRule(tool="get_*")],
    llm_classifier=is_destructive,
    fallback=CallbackApprovalHandler(human_reviewer),
)
```

An `"escalate"` verdict does not decide anything — it hands off to your human fallback, so the LLM never gets the final word on a risky call. You can back this function with a real model call for genuine judgment, or keep it as fast heuristics; the classifier does not care as long as it returns one of the three verdicts. Because this layer is a plain callable, it composes cleanly with the rest of your [guardrails](../../core/guardrails.md) — PII redaction, credential detection, and prompt-injection scanning still run on the request payload before a reviewer ever sees it.

## Failing closed and auditing every decision

Two properties make this safe to run unattended.

**It fails closed.** `on_timeout="deny"` means a request that no human answers within the timeout is rejected, not silently allowed. The default is deny for exactly this reason — an unanswered approval should never become an approval. Set it to `"allow"` only for genuinely low-stakes tools where availability matters more than caution.

**Every decision is auditable.** The `stats` object counts hits per layer (`allow_rule_hits`, `read_only_allows`, `llm_denies`, `fallback_denies`, and so on), and each decision carries `decision.trace`, which tells you exactly why that call was cleared or blocked; pass `ApprovalPolicy(on_decision=...)` to write every one to your audit log. If you see `fallback_allows` climbing, your rules are too tight and humans are picking up slack they shouldn't. If `read_only_allows` dominates, the read-only layer is earning its keep. You tune the policy from real traffic, not intuition. For the security background on why layered, deny-by-default gating beats a single checkpoint, the [LLM Guardrails in Python: The Complete Guide](llm-guardrails-python.md) walks through the broader defense-in-depth model this fits into.

## When a plain approval gate is the better fit

Be honest about scope. The classifier is the right tool when your agent calls many tools, most of them safe, and you need volume without losing the dangerous few. It is overkill in a few cases:

- **You have one or two sensitive tools and nothing else.** A bare `ApprovalPolicy(tools=["refund"], handler=...)` with no classifier is simpler and just as safe. Reach for the classifier when a plain allow-list of tool names stops scaling.
- **Every call is destructive by nature.** If a human genuinely must see all of them — say, a financial-transfer agent — auto-approval layers add complexity with nothing to auto-approve. Gate everything and skip the classifier.
- **Approval must happen inside the MCP server, for any client.** If the guarantee needs to hold regardless of which agent connects, enforce it server-side with the MCP server's `ApprovalGateMiddleware` and `requires_approval=True` declarations instead of (or in addition to) the agent-side policy shown here.

The classifier shines in the messy middle, which is where most real agents live.

## Frequently asked questions

### What is human-in-the-loop LLM approval?

It is a control that pauses an AI agent before it executes a chosen tool call and requires a decision — from a human, a rule, or a policy — before the call runs. In Promptise Foundry you attach an `ApprovalPolicy` to `build_agent()`, and the `AutoApprovalClassifier` decides per call whether to auto-clear it, hard-deny it, or escalate to a person.

### How do I auto-approve safe tool calls without approving risky ones?

Wrap your existing approval handler in an `AutoApprovalClassifier`. Its read-only layer auto-clears `get_*`, `list_*`, `search_*`, and similar tools, while explicit deny rules block `delete_*` or dangerous arguments outright. Only the calls that match neither reach a human, so reviewers spend attention where it counts.

### What happens if no one approves a request in time?

The request is denied. `ApprovalPolicy` defaults to `on_timeout="deny"`, so an approval that times out becomes a rejection rather than a silent allow. This "fail closed" behavior is why the gate is safe to leave running unattended.

## Next steps

Wrap your approver in an `AutoApprovalClassifier` to auto-clear read-only calls and gate only the destructive ones — it is a one-line change on top of the webhook, queue, or callback handler you already have. Start with the [Quick Start](../../getting-started/quickstart.md) to stand up an agent, then follow the [approval classifier guide](../../core/approval-classifier.md) to tune the six layers against your own tool set.
