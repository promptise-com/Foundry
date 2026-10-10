---
title: "Give a Code-Interpreter Agent Your Tools, Keep Approval Gates"
description: "A build guide scoped to the governance you keep, not the codeact pattern itself: enable agent_pattern='code-action', wrap mutating tools with an approval…"
keywords: "code interpreter agent with approval gates, give code interpreter my tools safely, max_tool_calls cap sandbox, human approval on generated code"
date: 2026-07-16
slug: code-interpreter-agent-with-approval-gates
categories:
  - Sandboxing
---

# Give a Code-Interpreter Agent Your Tools, Keep Approval Gates

Building a **code interpreter agent with approval gates** means letting the model write and run one program over *your* tools while every mutating call it makes still stops at a human — and gets counted against a hard per-run budget. That combination is the whole point of this guide. The code-interpreter (codeact) pattern itself is easy; the interesting engineering is the governance you refuse to give up when you hand a model an execution surface. This post is scoped to exactly that: enable `agent_pattern="code-action"`, wrap the tools that change state with an approval gate, keep a hard `max_tool_calls` cap on the generated program, and pick the gVisor backend when the input is untrusted.

<!-- more -->

If you want the case for the pattern in the first place — why one sandboxed program beats a 30-call tool loop — read [the code-action guide](../../guides/code-action.md) first. Here we assume you're sold on codeact and worried about the safety story. Good instinct.

## The governance you keep when the model writes code

A hosted code interpreter runs model-written Python in a box, computes, and returns a number. That's useful, but it runs over a *closed* toolset. The moment you want the program to call your database, your payments service, or your MCP servers, you've changed the risk profile entirely: the model can now trigger side effects, in bulk, from inside code you didn't review line by line.

Promptise Foundry's answer is a **governed tool bridge**. When you select `agent_pattern="code-action"`, the model writes one program in a single turn; that program runs inside a hardened, network-isolated sandbox; and every tool call the program makes is bridged back out to the *real* `BaseTool` on your host. Because the call executes your host tool — not a copy inside the box — three controls stay attached to generated code with no extra wiring:

1. **A human-approval gate on the bridged call.** Wrap a mutating tool with an `ApprovalPolicy` and the bridged call triggers it, exactly as a normal tool call would.
2. **A hard `max_tool_calls` cap per run.** The code-action node enforces its own hook-independent ceiling (default 50) so a generated loop can never hammer a tool unbounded.
3. **Kernel-level isolation on demand.** Swap the container backend to gVisor for untrusted input.

That is the delta this article is about: giving a code interpreter your tools *safely*, where "safely" is a policy you author, not a property of a vendor's closed sandbox.

## Build it: a code interpreter that still asks permission

The example below is runnable end-to-end — set `OPENAI_API_KEY` and have Docker running. It gives a code-action agent three tools: two read-only (`list_employees`, `get_employee`) that the program may call freely, and one mutating tool (`apply_raise`) that is gated. The model is asked to raise a whole department, so it will write a loop that calls `apply_raise` several times — and each of those calls passes through your approval handler.

```python
import asyncio
from langchain_core.tools import tool
from promptise import build_agent, ApprovalPolicy, ApprovalDecision

EMPLOYEES = {
    "Ada":   {"department": "Engineering", "salary": 210000},
    "Grace": {"department": "Engineering", "salary": 195000},
    "Linus": {"department": "Sales",       "salary": 130000},
}

@tool("list_employees")
def list_employees() -> list:
    """Return every employee name."""
    return list(EMPLOYEES)

@tool("get_employee")
def get_employee(name: str) -> dict:
    """Return {name, department, salary} for one employee."""
    return {"name": name, **EMPLOYEES[name]}

@tool("apply_raise")
def apply_raise(name: str, amount: int) -> str:
    """Give an employee a raise (mutating — writes payroll)."""
    EMPLOYEES[name]["salary"] += amount
    return f"{name} now earns {EMPLOYEES[name]['salary']}"

# The approval gate. It fires for EVERY bridged call to a mutating tool —
# including calls the model makes from inside its generated program.
async def review(request) -> ApprovalDecision:
    amount = request.arguments.get("amount", 0)
    if amount <= 5000:  # auto-approve small, routine changes
        return ApprovalDecision(approved=True, reviewer_id="policy-bot")
    # Anything larger is held for a human (here we deny to keep the demo non-interactive).
    print(f"HOLD: {request.tool_name}({request.arguments}) exceeds the auto-approve limit")
    return ApprovalDecision(approved=False, reason="over auto-approve limit")

async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={},                      # or your MCP servers
        agent_pattern="code-action",     # the model writes ONE program over your tools
        extra_tools=[list_employees, get_employee, apply_raise],
        approval=ApprovalPolicy(
            tools=["apply_raise", "delete_*", "payment_*"],  # glob patterns
            handler=review,
            on_timeout="deny",           # fail closed if no decision arrives
        ),
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content":
        "Give everyone in Engineering a 5000 raise, then report the new department total."}]})
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

Two things are worth calling out. First, the read-only tools run unimpeded — the approval policy's glob patterns only match `apply_raise` (and any future `delete_*` / `payment_*` tools), so there's zero overhead on the tools that don't mutate. Second, `on_timeout="deny"` makes the gate **fail closed**: if a human never answers, the tool call is rejected, not silently allowed. The full policy surface — argument redaction, `max_pending`, per-deny retry limits — is documented alongside the tool bridge in the [code-action guide](../../guides/code-action.md).

## The three controls, in detail

### 1. Approval on the bridged call

`build_agent(..., approval=...)` wraps matching tools *before* the code-action graph is built, so the code-action node only ever sees the wrapped tool. When the generated program calls `apply_raise("Ada", 5000)`, the bridge invokes the wrapped tool on the host, which pauses and asks your `handler` for an `ApprovalDecision`. This is what "human approval on generated code" actually looks like in practice: the model can write whatever loop it wants, but it cannot commit a mutating action you haven't authorized. The same `ApprovalHandler` protocol also has built-in webhook and pending-queue implementations if a Python callback isn't where your reviewers live.

### 2. The hard `max_tool_calls` cap

The approval gate governs *which* calls are allowed. The `max_tool_calls` cap governs *how many* calls the program can make at all — a `max_tool_calls` sandbox ceiling that is independent of any runtime hook. The code-action node ships a default of 50: once a single program has issued that many bridged calls, further calls return a budget-exceeded error instead of running. This is the backstop against a model that writes `while True:` around your tools. It is on by default and needs no configuration.

If you want a *tighter*, explicitly-set per-run cap — and per-call budget, health, and audit records on every bridged call — run the same agent under the [Agent Runtime](../../runtime/index.md). Its governance hooks apply to each bridged call, so a runtime budget bounds generated code too:

```python
from promptise.runtime import BudgetConfig

# Attach to a ProcessConfig; the runtime enforces this on every bridged call.
budget = BudgetConfig(
    enabled=True,
    max_tool_calls_per_run=20,   # your ceiling, below the built-in 50
    on_exceeded="stop",          # stop the run when the cap is hit
)
```

The distinction is honest and worth internalizing: the code-action node's built-in cap is a *hook-independent* safety bound that protects even a plain, standalone agent; the runtime `BudgetConfig` is a *governed* cap with escalation, daily limits, and an audit trail. Reach for the runtime when the agent is long-lived or the input is untrusted.

### 3. gVisor for untrusted input

By default, code-action provisions a hardened Docker sandbox: read-only rootfs, ~40 dropped Linux capabilities, Docker's default seccomp profile, resource limits, and `network="none"` (the sandbox default, and auto-set for this pattern, so the program's only reach to the outside world is your bridged tools). For untrusted or multi-tenant input, add kernel-level isolation by switching the backend to gVisor — a one-line change:

```python
agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={},
    agent_pattern="code-action",
    extra_tools=[list_employees, get_employee, apply_raise],
    approval=ApprovalPolicy(tools=["apply_raise"], handler=review, on_timeout="deny"),
    sandbox={"backend": "gvisor", "network": "none"},   # runsc runtime; kernel-level isolation
)
```

gVisor (`runsc`) interposes a user-space kernel between the container and the host kernel, shrinking the syscall attack surface for genuinely adversarial code. It needs `runsc` installed on the host; if you don't have it, drop the `sandbox=` line and code-action falls back to its auto-enabled hardened Docker sandbox. The full container security model — seccomp, capability dropping, `no-new-privileges`, network modes — is documented under [Sandbox](../../core/sandbox.md).

## What other frameworks do today

This capability is worth situating honestly against the alternatives, because each does part of the job well.

**Hosted code interpreters (e.g. OpenAI's `code_interpreter` tool).** These run model-written Python in a sandboxed, network-isolated container that the vendor hosts. The isolation is real and well-engineered. The delta is the *toolset*: the code runs over a closed environment — a Python runtime plus files you upload — not over your arbitrary host tools. In the same API you can define function tools, but those are called through the normal tool-call loop, separate from the interpreter's sandbox; the program the interpreter writes does not itself invoke your function tools, so there is no supported path to fire a per-call human-approval gate and a hard per-run tool-call budget *you* define on the code the model wrote. As of these APIs, human-in-the-loop over tool calls is client-side (you decide whether to submit outputs), not a server-enforced gate on generated code.

**DIY / bolt-on sandboxes (e2b, Riza, Modal, `langchain-sandbox`).** These give you a genuinely hardened boundary — e2b runs Firecracker microVMs, Riza uses WASM isolates, Modal spins serverless containers. Where they have a partial feature, say so: they isolate the *code* excellently. The precise delta is that they are **generic code executors, not tool-aware bridges.** If you want the sandboxed program to call your database or MCP tools, you expose those yourself over RPC, and any approval prompt, per-run cap, or audit entry on those bridged calls is integration work you build and maintain.

Promptise's edge isn't "we have a sandbox and they don't" — several of these have excellent sandboxes. It's that the code-action bridge makes per-call governance **structural**: because each tool call the generated program makes runs back through the real host `BaseTool`, your `ApprovalPolicy` and the hard `max_tool_calls` cap apply to generated code as a first-class property of the pattern, with no closed toolset to accept and no RPC layer to stand up. For the fuller three-way comparison of in-process REPLs, bolt-on sandboxes, and governed bridges, see [How to Run AI-Generated Code Safely: The 2026 Field Guide](run-ai-generated-code-safely.md); for the threat model that motivates isolating this code in the first place, see [The Real Risks of Running AI-Generated Code In-Process](risks-of-running-ai-generated-code.md).

## When to reach for this — and when not

Code-action with gates is a precise instrument, not a default. Be honest about the fit:

| Reach for a gated code-interpreter agent | Prefer a normal `react` loop |
|---|---|
| The task gathers many facts then computes (sums, joins, aggregation) | A single tool call answers it |
| Some tools mutate state and must not fire unsupervised | Everything is read-only |
| Input may be untrusted or multi-tenant | Input is fully trusted, internal |
| You can run Docker (and `runsc` for gVisor) | No container runtime available |

Two caveats stated plainly. First, there's a **latency floor** — each run spins a fresh container (~1–2 s), so for a one-shot lookup a plain `react` call is snappier; the wins dominate on real aggregation. Second, containment stops the model's *code* from escaping the box; it does not, on its own, make your *tools* safe. That's precisely why the approval gate, the tool-call cap, and a least-privilege tool set matter — they govern what the (contained) program is allowed to *do*, which is a different question from where it runs.

## Frequently asked questions

### Does the approval gate really fire on calls the model makes from inside its own program?

Yes. `build_agent(..., approval=...)` wraps matching tools before the code-action graph is constructed, and the bridge invokes those wrapped tools on the host. So a mutating call the model writes into its generated program pauses at your approval handler exactly like a top-level tool call would. With `on_timeout="deny"`, an unanswered request fails closed.

### How do I give a code interpreter my tools safely without a hosted sandbox service?

Point `build_agent` at `agent_pattern="code-action"` and pass your tools via `extra_tools` (or your MCP `servers`). The sandbox provisions itself — hardened Docker by default, with `network="none"` — and each tool the program calls is bridged back to the real host tool, so your approval policy and the `max_tool_calls` cap apply. There is no isolation service to sign up for and no RPC bridge to build.

### What is the `max_tool_calls` cap, and can I change it?

It's a hard, hook-independent ceiling on how many bridged tool calls one generated program may make (default 50); beyond it, calls return a budget-exceeded error. The built-in cap protects even a standalone agent. To set a tighter, explicit per-run cap — with escalation and an audit trail — run the agent under the Agent Runtime and configure `BudgetConfig(max_tool_calls_per_run=...)`, which the runtime enforces on every bridged call.

### When should I use the gVisor backend?

Use it for untrusted or multi-tenant input, where you want a user-space kernel between the container and the host kernel. Set `sandbox={"backend": "gvisor", "network": "none"}` (requires `runsc` installed). For trusted internal workloads, the default hardened Docker sandbox is sufficient.

## Next steps

Ship a code-interpreter agent that still honors your approval gates and per-run call caps. Start from the runnable example above, then wire your own mutating tools into the `ApprovalPolicy` globs. Read the [code-action guide](../../guides/code-action.md) for the tool-bridge mechanics, the [Sandbox](../../core/sandbox.md) reference for the container security layers and the gVisor backend, and the [Agent Runtime](../../runtime/index.md) docs to add a governed `BudgetConfig` cap, health checks, and audit on every bridged call.
