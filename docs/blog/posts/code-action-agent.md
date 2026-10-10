---
title: "Code-Action Agents: Write One Program, Not 30 Tool Calls"
description: "For data-heavy work (sums, joins, multi-hop aggregation), chaining tool calls is slow, lossy, and gets the arithmetic wrong. Code-action has the model write…"
keywords: "code-action agent, codeact pattern, code as action llm, agent writes code instead of tool calls, code-action reasoning"
date: 2026-07-16
slug: code-action-agent
categories:
  - Reasoning
---

# Code-Action Agents: Write One Program, Not 30 Tool Calls

A **code-action agent** flips the action space of an LLM: instead of chaining dozens of conversational tool calls to gather facts and then guessing the arithmetic in its head, the model writes **one Python program** over your tools and runs it. If you have ever watched an agent re-query the same records, blow its context window, and still return the wrong total, this pattern is for you. By the end of this post you will know when the codeact pattern wins, how to switch it on in Promptise Foundry with a single argument, and why the sandbox it runs in is locked down by default — not a bolt-on you have to remember to add.

<!-- more -->

## What a code-action agent actually does

The **codeact pattern** — sometimes written "code as action" — treats generated code as the agent's action. Rather than emitting a tool call, waiting for a result, emitting another, and so on, the model emits a whole program in a single turn. That program calls your tools as ordinary Python functions, loops over results, filters, joins, and computes an exact answer.

The difference matters most for a specific shape of task: **gather N facts, then compute**. Think "sum every salary in Engineering," "average a metric across a dependency graph," or "join orders to customers and count the repeat buyers." A normal tool loop handles each of those poorly because:

- The transcript grows on every call, so the model loses the thread and re-queries facts it already has.
- Aggregation happens in the model's head, where arithmetic over more than a handful of numbers is unreliable.
- Every round-trip is another LLM call — more latency, more tokens, more cost.

A program has none of those failure modes. Loops don't forget. Sums don't drift. And the whole thing is one turn.

## Why the codeact pattern beats chaining tool calls

Here is the concrete anatomy of the problem. Suppose you ask a `react` agent for the combined salary of a 40-person department. It calls `list_employees`, then `get_employee` forty times, each result appended to the conversation. By call thirty the context is a wall of JSON, the model starts summarizing instead of reading, and the final addition is a plausible-looking guess. You paid for 41 LLM round-trips to get a number that's off by a few thousand dollars.

**Code-action reasoning** removes the guesswork. The model writes:

```
names = list_employees()
total = sum(get_employee(n)["salary"] for n in names
            if get_employee(n)["department"] == "Engineering")
print(f"RESULT: {total}")
```

One turn produces the plan; the sandbox produces the exact number. This is the core idea behind [the ReAct agent pattern](react-agent-pattern.md) inverted — instead of *reason, act, observe* repeated in a loop, you *reason once, act as a whole program, observe once*. For a fuller tour of how code-action sits alongside the other nine built-in strategies, see [Agent Reasoning Patterns: The Complete Guide](agent-reasoning-patterns.md).

## Run a code-action agent in Promptise Foundry

Switching an agent to write code instead of tool calls is one argument: `agent_pattern="code-action"`. Point it at your MCP servers, or hand it plain tools via `extra_tools`. The example below is fully runnable — set `OPENAI_API_KEY` and have Docker running.

```python
import asyncio
from langchain_core.tools import tool
from promptise import build_agent

EMPLOYEES = {
    "Ada":  {"department": "Engineering", "salary": 210000},
    "Grace":{"department": "Engineering", "salary": 195000},
    "Linus":{"department": "Sales",       "salary": 130000},
    "Mabel":{"department": "Engineering", "salary": 205000},
}

@tool("list_employees")
def list_employees() -> list:
    """Return a list of every employee name."""
    return list(EMPLOYEES)

@tool("get_employee")
def get_employee(name: str) -> dict:
    """Return {name, department, salary} for an employee."""
    return {"name": name, **EMPLOYEES[name]}

async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={},                       # or your MCP servers
        agent_pattern="code-action",      # sandbox auto-enabled (Docker required)
        extra_tools=[list_employees, get_employee],
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content":
        "What is the combined salary of everyone in Engineering?"}]})
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

The model writes one program: list the names, look each up, filter to Engineering, sum. The `CodeActionNode` under the hood bridges every `list_employees()` and `get_employee()` call inside the program back to the *real* tool on your host, then feeds the printed `RESULT:` back as the answer. There is no manual wiring — pointing `build_agent` at the pattern is enough. The full mechanics, including the request/response file bridge, live in the [code-action guide](../../guides/code-action.md).

**Tip:** code-action is at its best when tools return **structured data** — lists, dicts, numbers — because the bridge preserves JSON-serializable values. A tool that returns a `dict` arrives in the program as a `dict`, so `emp["salary"]` just works. Prose-string tools still function (the model is told to parse them), but structured returns are more reliable.

## The sandbox is auto-enabled and locked down

Here is the differentiator that makes code-action safe to reach for, not a footgun: **the sandbox provisions itself and is hardened by default.** You do not pass `sandbox=True`; selecting the pattern turns it on. And it isn't a permissive scratch container — the model's program runs with:

- **`network="none"`** — auto-set for this pattern. The program's only reach to the outside world is your tools, via the bridge. It cannot phone home.
- **Read-only rootfs and dropped capabilities** — roughly 40 Linux capabilities stripped, plus `no-new-privileges` and Docker's default seccomp profile.
- **Resource limits** — CPU, memory, and an `exec_timeout` (default 120s) so a runaway program is killed, not left spinning.
- **A hard `max_tool_calls` cap per run** — hook-independent, so a generated program can never loop a tool unbounded, even without the Agent Runtime attached.

Crucially, **your tools keep their own protections.** Each bridged call invokes the real `BaseTool` on the host, so if you wrapped a mutating tool with an approval gate, the bridged call triggers it too. Under the Agent Runtime, budget, health, and audit hooks apply to each bridged call. The containment means the model's *code* can't escape the box; it does not, on its own, make your *tools* safe — so for untrusted or multi-tenant input, keep the tool set least-privilege, wrap mutating tools with approval, and consider the gVisor backend for kernel-level isolation. The full security model is documented in [the prebuilt patterns reference](../../core/engine-prebuilts.md).

If Docker isn't available, `build_agent(agent_pattern="code-action")` raises a clear error rather than silently running model-written code on the host. That "no silent fallback" stance is deliberate.

## When code-action is the right pattern — and when it isn't

Code-action is a pattern, not a replacement. Be honest about the fit:

| Reach for code-action | Prefer `react` / `managed` instead |
|---|---|
| Sums, averages, counts over a dataset | Open-ended or conversational Q&A |
| Multi-hop joins or graph traversal | A single tool call answers it |
| "Gather many facts, then compute" | Ambiguous tasks that need clarifying first |
| You can run Docker | No container runtime available |

Two caveats worth stating plainly. First, there is a **latency floor**: each run spins a fresh container (~1–2 seconds), so for a one-shot lookup a plain `react` call is snappier — the token and accuracy wins only dominate on real aggregation work. Second, code-action makes the *computation* exact; it does not make the model's *plan* smarter. If the task is ambiguous, clarify it before you reach for a program. The full comparison across all ten built-in patterns lives in the [reasoning patterns reference](../../core/agents/reasoning-patterns.md).

## Frequently asked questions

### What is a code-action agent?

A code-action agent is an LLM that responds to a task by writing a single program over your tools and executing it, instead of emitting a chain of individual tool calls. It's ideal for data-heavy work — sums, joins, multi-hop aggregation — where code computes an exact answer that a conversational loop would get wrong by re-querying facts and mis-adding them in the model's head.

### Do I need to configure the sandbox myself?

No. Selecting `agent_pattern="code-action"` auto-enables Promptise's hardened Docker sandbox with no network, a read-only rootfs, dropped capabilities, and resource limits. You only need Docker running. If a sandbox can't be initialized, `build_agent` raises a clear error rather than falling back to running the code unsandboxed.

### Is code-action safe for untrusted or multi-tenant input?

The containment stops the model's *code* from escaping the box, but it doesn't automatically make your *tools* safe. For untrusted or multi-tenant use, keep the tool set least-privilege, wrap mutating tools with approval gates, set a conservative `max_tool_calls`, and prefer the gVisor backend for kernel-level isolation.

## Next steps

Run `agent_pattern="code-action"` on a data task today — the sandbox provisions itself, so there's nothing extra to stand up. Start with the [Quick Start](../../getting-started/quickstart.md) to get a first agent running, then walk through the [code-action guide](../../guides/code-action.md) to wire your own aggregation tools into a single-program agent.
