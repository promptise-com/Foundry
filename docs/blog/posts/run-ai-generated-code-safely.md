---
title: "How to Run AI-Generated Code Safely: The 2026 Field Guide"
description: "The cluster hub: contrasts the three ways teams run model-written code — in-process REPL (isolates nothing), a bolt-on third-party sandbox (e2b/Riza/Modal …"
keywords: "run AI-generated code safely, execute llm code in a sandbox, hardened agent sandbox, code interpreter agent isolated, governed tool bridge sandbox"
date: 2026-07-16
slug: run-ai-generated-code-safely
categories:
  - Sandboxing
---

# How to Run AI-Generated Code Safely: The 2026 Field Guide

To **run AI-generated code safely** you have to satisfy two requirements at once, and most stacks only meet one of them: *can the model's code escape the box?* and *is every tool that code reaches inside the box still governed the way the rest of your agent is?* Containment answers the first. Almost nothing answers the second. This field guide contrasts the three ways teams actually run model-written code today — an in-process REPL, a bolt-on third-party sandbox, and a hardened first-party sandbox with a governed tool bridge — and shows why Promptise Foundry treats *both* containment and per-call tool governance as first-class parts of one layer instead of two separate integrations you assemble yourself.

<!-- more -->

## The three ways to run model-written code

Every approach to executing LLM-written Python lands in one of three buckets. They differ on two independent axes — does it isolate the code, and does it govern the tool calls that code makes — and only the third gets both.

| Approach | Isolates the code? | Governs each in-sandbox tool call? | What it is |
|---|---|---|---|
| **In-process REPL** | No | No | `exec()` in your interpreter — the model's code inherits your process |
| **Bolt-on third-party sandbox** | Yes | No | e2b / Riza / Modal / `langchain-sandbox` — a real boundary, opt-in, generic code executor |
| **Hardened first-party sandbox + governed bridge** | Yes | Yes | Promptise's Docker/gVisor sandbox where each tool call routes back through host-side approval, budget, and audit |

The first option is a footgun and it's the frictionless default in most frameworks — the [in-process risks are their own post](risks-of-running-ai-generated-code.md). The second is genuinely good tooling that closes the containment gap but leaves a governance gap. The third is the one this guide argues for, because "safe" is not only "the code can't touch my host" — it's also "the code can't quietly call a mutating tool without the same approval gate, budget cap, and audit trail that every other agent action goes through."

## What other frameworks do today

To be fair and precise, hardened code execution *exists* across the ecosystem in 2026. The honest gaps are narrower and more specific than "they can't sandbox."

- **LangChain `PythonREPLTool` / `PythonAstREPLTool`** run the model's code in your host Python process via `exec` (the AST variant parses first, but still executes in-process). Both ship under `langchain_experimental` and warn in their own docs that they can execute arbitrary code. The delta here is not a missing warning — it's that isolation is *your* job the moment you accept the built-in tool.
- **e2b, Riza, and Modal** each give you a real, hardened boundary — e2b runs Firecracker microVMs, Riza executes in WASM isolates, Modal spins serverless containers. This is excellent isolation. The precise delta is two-fold. First, each is an opt-in dependency or hosted service you choose, wire up, and (for the hosted ones) send code to. Second — and this is the part teams miss — they isolate the *code*, but they are **generic code executors, not tool-aware bridges.** If you want the sandboxed program to call your database tool or your MCP tools, you expose those yourself over RPC, and any approval prompt, per-run budget decrement, or audit entry on those calls is integration work you build and maintain.
- **`langchain-sandbox`** (Pyodide/Deno-based) is the newer in-ecosystem answer and runs Python in a WASM boundary. Same shape as above: a good containment story, opt-in, and no built-in notion of routing each tool call the code makes back through host-side governance.

So the honest framing is not "other frameworks can't isolate model code." They can, and some do it well. The exact delta is that **the safe containment path is a second decision layered on later, and per-tool-call governance inside that sandbox is something you assemble by hand.** Promptise's edge is structural: it makes both the hardened container *and* the governed tool bridge the path of least resistance, in one layer, from the same `build_agent` call.

## The layer everyone forgets: governing each in-sandbox tool call

Here is the scenario a pure code sandbox does not cover. Your agent writes one program to reconcile a batch of orders. Inside that program it calls `list_orders()`, `get_order(id)` forty times, and then `issue_refund(id, amount)` on three of them. A generic sandbox contains all of that beautifully — the code can't read your `.env` or phone home. But `issue_refund` is a **mutating, money-moving action**, and from the sandbox's point of view it's just another function the code happened to call. There was no approval prompt, no budget check, no audit entry — the exact governance you'd demand if the agent had called `issue_refund` as a normal tool.

Promptise closes that gap with the **code-action governed bridge**. When you select `agent_pattern="code-action"`, the model writes one Python program over your tools and runs it in the hardened sandbox with `network="none"` auto-set — but each tool call inside that program does not execute in the container. The generated stub writes a request file, a concurrent **host-side loop** picks it up, runs the *real* `BaseTool` with all its hooks attached, and writes the response back. Because the actual tool runs on the host, its governance runs too:

- **Approval gates fire.** If you wrapped `issue_refund` with an approval gate, the bridged call triggers it exactly as a direct tool call would — human approval is enforced before the refund happens, deny-by-default on timeout.
- **Audit records every bridged call.** Each tool invocation the program makes is a real host-side tool execution, so it lands in your audit trail with the same identity and arguments as any other action.
- **Budget and health hooks apply under the Agent Runtime.** Run the agent as a governed `AgentProcess` and per-run limits (tool-call count, cost, irreversible actions) and behavioral anomaly detection apply to each bridged call.
- **A hard `max_tool_calls` cap is hook-independent.** Even a plain agent with no runtime attached enforces a per-run ceiling, so a generated program can never loop a tool unbounded.

That is the difference between "the code is isolated" and "the code is isolated *and* every action it takes is still governed." The full mechanics — the request/response file bridge over the writable tmpfs, the `.done` marker that prevents partial-read races, and the bounded stderr-driven self-repair — live in the [code-action guide](../../guides/code-action.md).

## Run a governed, sandboxed agent in Promptise Foundry

There is no isolation service to choose and no RPC to stand up. You point `build_agent` at the pattern, hand it your tools, and the sandbox provisions itself. The example below is runnable end-to-end — set `OPENAI_API_KEY` and have Docker running.

```python
import asyncio
from langchain_core.tools import tool
from promptise import build_agent

ORDERS = {
    "A-1001": {"region": "EU", "amount": 420.0},
    "A-1002": {"region": "US", "amount": 275.0},
    "A-1003": {"region": "EU", "amount": 158.5},
    "A-1004": {"region": "EU", "amount": 640.0},
}

@tool("list_orders")
def list_orders() -> list:
    """Return every order id."""
    return list(ORDERS)

@tool("get_order")
def get_order(order_id: str) -> dict:
    """Return {order_id, region, amount} for one order."""
    return {"order_id": order_id, **ORDERS[order_id]}

async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={},                       # or your MCP servers
        agent_pattern="code-action",      # sandbox auto-enabled; network="none" auto-set
        extra_tools=[list_orders, get_order],
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content":
        "What is the total amount of every EU order?"}]})
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

The model writes one program: list the ids, look each up, filter to `EU`, sum. Every `list_orders()` and `get_order()` call inside the container bridges back to the *real* tool on your host — which is precisely where a governance hook (an approval gate on a mutating tool, an audit entry, a budget decrement) would run. Swap `get_order` for a mutating `issue_refund` wrapped with `build_agent(..., approval=...)` and the bridged call prompts for approval before it executes. If Docker isn't available, `build_agent(agent_pattern="code-action")` raises a clear error rather than silently running model-written code on your host. That "no quiet fallback to the dangerous path" stance is deliberate.

## Harden the box — and the second kind of model code

The default profile is already locked down — read-only rootfs, roughly 40 dropped Linux capabilities, Docker's default seccomp profile, resource limits, and no network. For untrusted or multi-tenant input, tighten two knobs. Cut the network outright and move to kernel-level isolation:

```python
agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={},
    agent_pattern="code-action",
    extra_tools=[list_orders, get_order],
    sandbox={"backend": "gvisor", "network": "none"},   # gVisor kernel + no egress
)
```

Cutting the network is the single highest-leverage control, because it collapses the exfiltration path even if something inside the box goes wrong — the full argument is in [Can Your Agent Exfiltrate Data? Sandboxing Code With No Network](sandbox-agent-code-with-no-network.md). Every knob — `backend`, `network`, `memory_limit`, `cpu_limit`, `timeout`, and the gVisor `runsc` runtime — is documented layer by layer in the [sandbox reference](../../core/sandbox.md).

There is also a *second* kind of model-written code worth naming, because it uses the same containment layer. An **Open Mode** agent can write its own tools at runtime via the `create_tool` meta-tool. With `sandbox_custom_tools=True` (the default), that agent-authored Python executes with **restricted builtins** — `open`, `exec`, `eval`, `__import__`, file I/O, and network access are all blocked — so a self-modifying agent still can't reach your host through the code it invents. The full permission model, including `max_custom_tools` and the MCP-URL whitelist, is in the [meta-tools reference](../../runtime/meta-tools.md). Whether the code comes from a code-action program or a self-authored tool, the same principle holds: the box contains it, and the bridge (or restricted builtins) governs what it can reach.

## Frequently asked questions

### What is the safest way to run AI-generated code?

Run it in a hardened container, cut the network, and — the part most stacks miss — keep every tool the code calls under the same governance as the rest of your agent. In Promptise that is one argument: `agent_pattern="code-action"` provisions a Docker sandbox with a read-only rootfs, ~40 dropped capabilities, Docker's default seccomp profile, resource limits, and `network="none"`, while routing each in-sandbox tool call back through the real host-side tool so approval gates, audit, and budget hooks still apply.

### How is this different from e2b, Riza, or Modal?

Those are genuinely good, hardened code sandboxes — e2b uses microVMs, Riza uses WASM isolates, Modal uses serverless containers. The delta is not isolation. It's that they are generic code executors: you wire your tools into them yourself over RPC, and any approval prompt, budget cap, or audit entry on the calls the sandboxed code makes is integration work you build. Promptise ships that as a first-class tool bridge, so per-call governance is structural rather than assembled by hand.

### Does the sandbox make my tools safe, or just the code?

The containment stops the model's *code* from escaping the box; it does not, on its own, make your *tools* safe. That is exactly why the governed bridge matters: bridged calls invoke the real `BaseTool` with its hooks, so an approval gate on a mutating tool still fires. For untrusted or multi-tenant input, keep the tool set least-privilege, wrap mutating tools with approval, set `max_tool_calls` conservatively, and prefer the gVisor backend.

### Do I have to configure the sandbox myself?

No. Selecting `agent_pattern="code-action"` (or `sandbox=True`) auto-enables the hardened profile — you only need Docker running. If a sandbox can't be initialized, `build_agent` raises a clear error instead of falling back to running the code unsandboxed. You override individual knobs by passing a `sandbox={...}` dict when you need gVisor, a different network mode, or tighter resource limits.

## Next steps

Compare the three approaches against your own workload, then start with `build_agent(agent_pattern="code-action")` — the sandbox provisions itself and the tool bridge keeps every in-sandbox call governed, so there is nothing extra to stand up. Wire your own aggregation or reconciliation tools through the [code-action guide](../../guides/code-action.md), tune the containment layers in the [sandbox reference](../../core/sandbox.md), and if you're running self-modifying agents, review how agent-authored code is contained in the [meta-tools reference](../../runtime/meta-tools.md).
