---
title: "Single-Agent vs Multi-Agent: When to Actually Split"
description: "A deliberately honest decision guide: most teams reach for multiple agents too early and pay coordination and token overhead for it. Explains the concrete…"
keywords: "single agent vs multi-agent, when to use multi-agent systems, multi-agent vs single agent, agent delegation tradeoffs, do i need multiple agents"
date: 2026-07-16
slug: single-agent-vs-multi-agent
categories:
  - Use Cases
---

# Single-Agent vs Multi-Agent: When to Actually Split

The single agent vs multi-agent decision gets made too early and for the wrong reasons — usually a diagram in a slide deck rather than a signal in your workload. Splitting one agent into several feels like progress, but it buys you a coordination layer, extra token overhead, and a new class of failures before it buys you any capability. This guide gives you the concrete signals that actually justify a split, the cost governance you should put in place first, and runnable code for adding delegation to Promptise Foundry only when the signals fire.

<!-- more -->

## Start with one agent (the honest default)

A single, well-scoped `build_agent()` handles far more than most people expect. It has memory, guardrails, a sandbox, semantic caching, and tool discovery from any number of MCP servers — all behind one factory call. Before you reach for a team, ask whether your problem is really two problems or just one problem with a lot of tools.

The reflex to split usually comes from tool-list anxiety: "this agent has forty tools, it must be doing too much." But a long tool list is not a reason to split. Promptise's semantic tool optimization selects only the relevant tools per query, which the *Why Promptise* page documents as 40–70% fewer tokens exposed to the model. One agent with forty well-organized tools is often cheaper and more reliable than four agents passing messages, because every hop between agents is another LLM round-trip you pay for and another place the plan can drift.

Reach for one agent first. Make it prove it can't do the job before you add a second.

## When to use multi-agent systems: the three real signals

Multi-agent vs single agent is not a capability question — a single agent can call any tool a specialist could. It's a *boundary* question. Split only when your workload has a boundary that a single agent's shared context and shared permissions cannot honor. In practice there are three:

- **Divergent tool sets.** Two roles use tools that never overlap and would only confuse each other's tool selection — a web researcher and a Postgres migration runner have nothing to share. Keeping them separate keeps each agent's tool surface small and its prompt focused.
- **Independent budgets.** One role does cheap, high-volume work (search, summarize) and another does expensive, rate-limited, or irreversible work (charge a card, send email). You want to cap and meter them separately, not average their behavior into one limit.
- **Separate trust boundaries.** One role can touch production data or external customers and another must never. A researcher that can hit the open web should not share a context window — or a JWT — with an agent that can publish to your CMS.

If none of these apply, you have one agent with a big job, not a team. When they do apply, the [multi-agent teams guide](../../guides/multi-agent-teams.md) walks through the four coordination primitives (shared MCP servers, delegation, EventBus messaging, and shared state) with a full research-pipeline example.

## What a split actually costs

Every split adds three recurring taxes that a single agent never pays:

1. **Latency and tokens.** Delegation is an LLM turn on the caller *and* a full invocation on the peer. A three-agent chain can triple your round-trips for a task one agent would finish in a single reasoning loop.
2. **Coordination surface.** You now own how agents address each other, how failures propagate when a peer times out, and how partial results get reconciled. That is real code you didn't have to write with one agent.
3. **Governance fan-out.** Budgets, audit logging, and identity now have to be reasoned about per agent, not once.

None of this is a reason to never split — it's a reason to split *deliberately*, with the cost controls in place before the second agent exists. That order matters, and it's the part most teams skip.

## Delegation without the ceremony (agent delegation tradeoffs)

When the signals do fire, Promptise keeps delegation lightweight: you register peers on `build_agent()` through the `cross_agents` parameter, and the agent gets an `ask_agent_<name>` tool per peer automatically (plus a `broadcast_to_agents` fan-out tool with `include_broadcast=True`). Your calling code doesn't change; the coordinator simply decides, per query, whether to answer directly or hand off.

```python
import asyncio
from promptise import build_agent
from promptise.cross_agent import CrossAgent


async def main():
    # A focused specialist — divergent tools, its own trust boundary.
    researcher = await build_agent(
        model="openai:gpt-5-mini",
        instructions="You research the web and return sourced findings only.",
    )

    # The coordinator delegates to the researcher as a peer, not a subroutine.
    coordinator = await build_agent(
        model="openai:gpt-5-mini",
        instructions="Answer the user. Delegate research to your peer when it helps.",
        cross_agents={
            "researcher": CrossAgent(
                agent=researcher,
                description="Web research specialist. Returns sourced findings.",
            ),
        },
    )

    result = await coordinator.ainvoke(
        {"messages": [{"role": "user",
                       "content": "Summarize the latest on MCP adoption."}]}
    )
    print(result["messages"][-1].content)

    await coordinator.shutdown()
    await researcher.shutdown()


asyncio.run(main())
```

Peers can be in-process during development or reachable over HTTP with JWT in production without changing this code. The tradeoff to keep in mind: `ask_agent_researcher` is a tool call the coordinator's model chooses to make, so it costs an LLM turn on both sides. That's exactly why you want a budget wrapping it. The full delegation semantics — context propagation, timeouts, and the broadcast contract — are in the [cross-agent delegation reference](../../core/agents/cross-agent.md).

## Govern the split before you make it

This is the step that separates a resilient team from a runaway one. Before a coordinator can delegate freely, give it an explicit envelope with an `AgentRuntime` governance budget. `BudgetConfig` caps tool calls, LLM turns, and cost per run — and each `ask_agent_*` delegation counts against those limits, so a chatty coordinator can't fan out forever.

```python
from promptise.runtime import ProcessConfig, BudgetConfig, ToolCostAnnotation

# Give the coordinator a hard envelope before it can delegate.
config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Answer the user. Delegate research to your peer when it helps.",
    budget=BudgetConfig(
        enabled=True,
        max_tool_calls_per_run=20,     # each ask_agent_* call counts here
        max_llm_turns_per_run=10,
        max_cost_per_run=25.0,         # abstract cost units, not dollars
        tool_costs={
            "ask_agent_researcher": ToolCostAnnotation(cost_weight=3.0),
        },
        on_exceeded="escalate",         # or "pause" / "stop"
        inject_remaining=True,          # the agent sees its remaining budget
    ),
)
```

Because delegation is expensive, weighting `ask_agent_researcher` higher than a cheap local tool makes the coordinator spend its budget where it counts. When the envelope is hit, the runtime pauses, stops, or escalates — your choice — instead of quietly burning turns. This is the independent-budgets signal made real: the researcher and the coordinator each get their own envelope. The full set of limits, cost annotations, and escalation targets lives in the [autonomy budget docs](../../runtime/governance/budget.md), and you can see governed multi-agent setups in the [Promptise showcase](../../resources/showcase.md).

## When a single agent is still the better fit

Be honest about the cases where splitting is the wrong call:

- **Your "roles" share most of their tools.** If the researcher and the writer both need the same five tools, you've split cosmetically. Keep one agent and let tool optimization narrow the surface per query.
- **Your workflow is a fixed, deterministic pipeline.** If step order never changes and you need durable retries and human-approval gates on a known DAG, a dedicated workflow engine (a durable-execution runner) will serve you better than LLM-driven delegation. Cross-agent primitives shine for *dynamic* handoffs the model decides at runtime — not as a replacement for a deterministic scheduler.
- **Latency is your hard constraint.** Every hop adds a round-trip. If you're optimizing for sub-second responses, one agent with tight tools usually wins.

For a fuller treatment of topologies once you've decided a team is warranted, the pillar post [How to Build Multi-Agent Systems in Python: 2026 Guide](multi-agent-systems-python.md) covers all four coordination patterns end to end.

## Frequently asked questions

### Do I need multiple agents, or just more tools?

Almost always just more tools. Adding a second agent is justified by a *boundary* — divergent tool sets, independent budgets, or separate trust boundaries — not by tool count. If your two "roles" would share most of their tools and context, keep one agent and let semantic tool optimization keep the prompt lean.

### What does multi-agent delegation cost compared to a single agent?

Each delegation is a tool call on the caller plus a full invocation on the peer, so you pay extra LLM turns and latency per hop. That overhead is worth it when a peer has genuinely different tools or permissions, and wasteful when it doesn't. Wrapping delegation in a `BudgetConfig` keeps the fan-out bounded and visible.

### Can I start with one agent and split later without a rewrite?

Yes. You add peers through the `cross_agents` parameter on `build_agent()` — the coordinator's calling code and its other tools stay the same. Start with one agent, and introduce `ask_agent_*` delegation only when a split signal actually fires.

## Next steps

Read the split checklist above, then start with one agent and add `ask_agent_*` delegation only when the signals fire — divergent tools, independent budgets, or separate trust boundaries. Ship your first agent with the [Quick Start](../../getting-started/quickstart.md), and when a split is genuinely warranted, wire delegation and governance together using the [cross-agent delegation reference](../../core/agents/cross-agent.md).
