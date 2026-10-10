---
title: "AutoGen Docker Executor vs a Hardened Agent Sandbox"
description: "AutoGen genuinely ships DockerCommandLineCodeExecutor, so this credits it up front. The honest delta: that executor gives you a container at Docker's defaults, not cap-drop…"
keywords: "autogen docker code executor isolation, hardened agent code sandbox, seccomp agent sandbox, gvisor agent code execution"
date: 2026-07-16
slug: autogen-docker-code-executor-isolation
categories:
  - Sandboxing
---

# AutoGen Docker Executor vs a Hardened Agent Sandbox

If you are evaluating **autogen docker code executor isolation** for running model-written code, start with the good news: AutoGen genuinely ships `DockerCommandLineCodeExecutor`, and it really does put generated code in a Docker container instead of your Python process. That is a meaningful boundary and a real step up from executing code in-process. This post credits that up front, then draws the honest line between "a container" and a *hardened agent code sandbox* — the difference between Docker's defaults and a profile purpose-built for untrusted model code, plus the piece AutoGen's executor does not do at all: bridging each tool call back through host governance.

<!-- more -->

## What AutoGen's Docker executor actually gives you

Let's be precise and fair, because AutoGen earns the credit. `DockerCommandLineCodeExecutor` lives in the `autogen-ext` package (`autogen_ext.code_executors.docker`). When the model emits a code block, the executor writes it to a file in a bind-mounted work directory and runs it inside a container — `python:3-slim` by default — with a `timeout`, and cleans the container up afterward (`auto_remove`, `stop_container`). Compared to `LocalCommandLineCodeExecutor`, which runs the same code directly on your host, this is the responsible option, and AutoGen documents it as such.

```python
# Illustrative ONLY — this is AutoGen, not Promptise.
from autogen_ext.code_executors.docker import DockerCommandLineCodeExecutor

executor = DockerCommandLineCodeExecutor(
    image="python:3-slim",   # real container isolation — a genuine step up from in-process
    timeout=60,
    work_dir="coding",
)
```

So the container itself is real. What matters for a security decision is everything the constructor above *doesn't* let you set — and what the container is doing that whole time.

## What other frameworks do today

To be fair and precise: a Docker container is not "wide open." Docker applies a default seccomp profile to every container and already drops a chunk of Linux capabilities. AutoGen inherits that baseline for free, and it is genuinely better than nothing. The honest delta is not "AutoGen has no isolation" — it is that AutoGen's executor stops at Docker's defaults and does not expose the knobs a hardened profile turns on. (Seccomp is not one of them: Promptise also runs under Docker's default seccomp profile, so on syscall filtering the two are level.)

- **No `no-new-privileges`.** `DockerCommandLineCodeExecutor` exposes no `security_opt` parameter, so a setuid binary in the image keeps its escalation path.
- **No extra capability dropping.** The container keeps Docker's default capability set; there is no `cap_drop` switch to strip it down to the minimum.
- **Writable root filesystem.** There is no `read_only` option, so the container's rootfs is writable — model code can overwrite system files or persist a foothold for the container's lifetime.
- **Network on by default.** The executor exposes no network switch, so the container runs on Docker's default bridge network with outbound access — the exfiltration path a computation task never needed.

None of that means AutoGen "can't" be hardened — you can always wrap or fork the executor and hand-roll `docker run` flags. The point is the delta: with AutoGen those controls are your job to research, wire, and maintain; the built-in path gives you Docker's defaults. LangChain's built-in `PythonREPLTool` sits a rung lower still, running code in your host process (we trace exactly what that reaches in [The Real Risks of Running AI-Generated Code In-Process](risks-of-running-ai-generated-code.md)). Promptise's edge is structural: the hardened profile is the *default* posture, not an upgrade you schedule for later.

## The bigger gap: standalone scripts vs a governed tool bridge

Isolation is only half the story, and the second half is the one that separates a code executor from an agent sandbox. AutoGen's executor runs the model's program as a **standalone script**. Whatever the code does, it does inside the container against the container's own world. That is safe precisely because it is *sealed* — but it also means the program can't reach your real, governed tools, and the calls it does make aren't individually accountable to your host.

Promptise Foundry takes a different shape with the [code-action pattern](../../guides/code-action.md): the model still writes one program in a sealed, network-isolated container, but each tool call in that program is **bridged back to the real tool on your host**. A concurrent host loop services the program's requests over the writable `/workspace` tmpfs, runs the actual `BaseTool`, and returns the value — so the program computes locally but every side-effecting call passes back through your governance. If a tool is wrapped with an approval gate, the bridged call still triggers it. Run the agent under the [Agent Runtime](../../guides/code-action.md) and its budget, health, and audit hooks apply to every bridged call. The container stays fully network-isolated because the bridge is a shared filesystem, not a socket.

That is the capability AutoGen's standalone executor structurally cannot offer: a program that is *both* sealed from the host and able to call the host's governed tools, one accountable call at a time.

## A hardened agent code sandbox you turn on with one argument

Here is the runnable version. Set `OPENAI_API_KEY`, have Docker running, and the model will write **one** Python program over your tools and run it in a hardened container — no in-process path, no manual `docker run` flags. This uses only real Promptise APIs.

```python
import asyncio
from promptise import build_agent
from langchain_core.tools import tool


@tool("list_invoices")
def list_invoices() -> list[str]:
    """Return every invoice id in the current period."""
    return ["INV-1001", "INV-1002", "INV-1003"]


@tool("get_invoice")
def get_invoice(invoice_id: str) -> dict:
    """Return {id, amount, status} for one invoice."""
    ledger = {
        "INV-1001": {"id": "INV-1001", "amount": 4200, "status": "paid"},
        "INV-1002": {"id": "INV-1002", "amount": 1800, "status": "open"},
        "INV-1003": {"id": "INV-1003", "amount": 900, "status": "paid"},
    }
    return ledger[invoice_id]


async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={},                       # or your MCP servers
        agent_pattern="code-action",      # sandbox auto-enabled; network auto-set to "none"
        extra_tools=[list_invoices, get_invoice],
        instructions="Answer aggregation questions by writing one program over the tools.",
    )

    result = await agent.ainvoke({"messages": [{"role": "user", "content":
        "What is the total amount of all PAID invoices this period?"}]})
    print(result["messages"][-1].content)

    await agent.shutdown()


asyncio.run(main())
```

Two things happen here that AutoGen's executor doesn't do on its own. First, `agent_pattern="code-action"` provisions the sandbox on Promptise's default-hardened profile *and* auto-sets `network="none"` — the program has no route off the box. Second, `list_invoices` and `get_invoice` run as real host tools through the governed bridge, so the program computes the sum exactly while every lookup stays accountable. If Docker can't be initialized, `build_agent` raises a clear error rather than silently running model code on your host — there is no quiet fallback to the unsafe path.

## Layer it up: seccomp, cap-drop, read-only rootfs, gVisor

The one argument above already gives you the hardened defaults; when you want to see or tighten the layers, they are first-class knobs, not a fork of someone else's executor. With `sandbox=True` (or code-action) Promptise runs the program on a profile that goes past Docker's baseline:

| Layer | What it closes |
|---|---|
| **Seccomp + no-new-privileges** | Docker's default `seccomp agent sandbox` profile blocks roughly 44 dangerous syscalls (`mount`, `reboot`, `kexec_load`, …) — the same baseline AutoGen gets — and `no-new-privileges` stops setuid escalation, which AutoGen's executor does not set. |
| **Capability dropping** | Roughly 40 Linux capabilities are stripped (`CAP_SYS_ADMIN`, `CAP_NET_ADMIN`, `CAP_SYS_PTRACE`, …), down to the minimum. |
| **Read-only rootfs** | The root filesystem is mounted read-only; only `/workspace` and `/tmp` are writable, as size-capped tmpfs mounts, so code can't overwrite system files or persist a foothold. |
| **Network isolation** | `network="none"`, the default, removes the outbound exfiltration path entirely; the opt-in `restricted` mode allows only DNS and HTTP/HTTPS, enforced with iptables. |
| **Resource limits** | CPU, memory, a PID cap (default 256), and an execution timeout — a timed-out command is killed inside the container — stop a runaway or fork-bomb program instead of your host. |

For untrusted input or an extra kernel-level boundary, switch the backend to **gVisor**, which intercepts syscalls in userspace so the container never talks to the host kernel directly — the strongest option for `gvisor agent code execution`:

```python
agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={},
    agent_pattern="code-action",
    extra_tools=[list_invoices, get_invoice],
    sandbox={
        "backend": "gvisor",   # gVisor (runsc) — kernel-level isolation
        "network": "none",     # no outbound; the bridge is the only channel
        "memory_limit": "256M",
        "cpu_limit": 1,
        "timeout": 30,
    },
)
```

Every one of these is documented layer by layer in the [sandbox reference](../../core/sandbox.md). The delta versus AutoGen is not that these controls are impossible elsewhere — it's that here they are the default and a single dict away, and they sit *on top of* the governed bridge rather than around a standalone script.

## Frequently asked questions

### Does AutoGen's Docker executor apply seccomp?

Indirectly — it inherits Docker's default seccomp profile, like any container, so it is not unprotected. Promptise uses that same default seccomp profile, so syscall filtering is not the difference. What `DockerCommandLineCodeExecutor` does not do is drop extra capabilities, set `no-new-privileges`, mount the rootfs read-only, or disable networking — it exposes no parameter for any of them. A `seccomp agent sandbox` in Promptise means Docker's default seccomp profile plus cap-drop, `no-new-privileges`, a read-only rootfs, resource limits, and `network="none"`, all turned on by default.

### What makes a hardened agent code sandbox different from "just a container"?

A container gives you isolation at Docker's defaults. A `hardened agent code sandbox` keeps Docker's default seccomp profile and layers on ~40 dropped capabilities, `no-new-privileges`, a read-only root filesystem, a configurable network mode (defaulting to none), and resource limits — before any model code runs. In Promptise that profile is the default posture of `sandbox=True`, not a set of `docker run` flags you assemble yourself.

### Do I need gVisor for agent code execution?

Not for most workloads — the default Docker profile with seccomp, cap-drop, read-only rootfs, and `network="none"` already closes the common reaches. Reach for `gvisor agent code execution` (`sandbox={"backend": "gvisor"}`) when you are running genuinely untrusted or multi-tenant input and want a kernel-level boundary, since gVisor services syscalls in userspace so the container never touches the host kernel directly. It requires installing the `runsc` runtime on the host.

### Can I migrate an AutoGen `DockerCommandLineCodeExecutor` workflow to Promptise?

Yes, and you usually simplify it. Where AutoGen runs the model's script standalone in a container, Promptise's code-action pattern runs the program in a hardened container *and* bridges each tool call back to your governed host tools. You define tools as normal `BaseTool`s, pass them via `extra_tools`, set `agent_pattern="code-action"`, and the sandbox is provisioned and network-isolated for you — no separate executor object to configure.

## Next steps

AutoGen's `DockerCommandLineCodeExecutor` gives you real container isolation; Promptise makes the hardening the default and adds the governed tool bridge on top of it. Turn on `agent_pattern="code-action"` (or `sandbox=True`) in `build_agent` and the model's code runs behind seccomp, dropped capabilities, a read-only rootfs, and `network="none"` — with each tool call still passing through your host governance. Start from the [sandbox reference](../../core/sandbox.md) for the full layer-by-layer configuration, wire an aggregation workload through the [code-action guide](../../guides/code-action.md) to get the governed bridge, and read [How to Run AI-Generated Code Safely: The 2026 Field Guide](run-ai-generated-code-safely.md) for the production checklist.
