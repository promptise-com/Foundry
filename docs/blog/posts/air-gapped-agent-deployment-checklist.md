---
title: "Air-Gapped Agent Deployment Checklist for Regulated Teams"
description: "A concrete pre-production checklist for finance, health, and gov teams shipping into an air-gapped or no-egress environment: models and embeddings staged…"
keywords: "air-gapped agent deployment checklist, on-prem agent compliance checklist, regulated LLM agent no egress, air-gapped agent audit logging, regulated industry on-prem AI agent"
date: 2026-07-16
slug: air-gapped-agent-deployment-checklist
categories:
  - Air-Gapped & Sovereign
---

# Air-Gapped Agent Deployment Checklist for Regulated Teams

This **air-gapped agent deployment checklist** is written for the teams who cannot hand-wave the offline story: banks, hospitals, insurers, and government programs shipping an agent into a network with no route to the public internet. A general production checklist tells you to add auth, rate limits, and health checks — all necessary, none of which prove that nothing leaves the perimeter. The offline dimension is its own discipline: every model staged on-box, every guardrail running on-device, code execution locked to no network, memory kept local and injection-scanned, and every sensitive action captured in a tamper-evident chain your auditor can verify. This is the pre-production list that closes the gap a standard checklist skips.

<!-- more -->

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## The offline dimension a production checklist skips

A normal go-live checklist assumes the internet is there and the question is whether you use it safely. In a `regulated LLM agent no egress` deployment, the question inverts: the internet is *gone*, and any layer that quietly expected it becomes a hard failure or — far worse — a silent data-exfiltration path that passes review because nobody noticed the outbound call.

An agent is not one dependency, it is a stack, and each layer has its own default network behavior:

| Layer | Air-gap failure mode | What you must verify |
|---|---|---|
| Model + embeddings | Weights fetch from a registry on first use | Staged locally, referenced by path |
| Guardrails | Moderation routed to a hosted API | Detection heads run on-device |
| Code execution | Sandbox assumes network for `pip`/DNS | Container pinned to no network |
| Memory | Embedding call leaves the host | Local embeddings, injection-scanned |
| Audit | Log is mutable and unsigned | Tamper-evident, verifiable offline |

The insight that makes an **on-prem agent compliance checklist** tractable is that these are not independent checkboxes. They share caller identity, they feed each other, and — critically — they all have to be verified *together* for a sign-off. When each is a separate tool you bolted on, "prove nothing egresses" becomes a per-seam audit you repeat every time a dependency updates. The companion hub [Air-Gapped AI Agent Framework: The On-Prem Guide](air-gapped-agent-framework.md) maps every layer to its local implementation; this checklist focuses on the three that regulated reviewers scrutinize hardest — sandbox, guardrails, and audit — and on proving nothing crosses the boundary.

## The air-gapped deployment checklist

Work top to bottom. Each item is a line you should be able to point at in code or config before go-live.

- **Stage models locally.** Pull the LLM (`ollama pull`) and pre-download the guardrail weights on a connected machine, copy the folders across the gap, and reference them by path. No component may reach a registry at runtime. The [model setup guide](../../getting-started/model-setup.md) documents the provider string and the local-endpoint path.
- **Run guardrails on-device.** Enable a `PromptiseSecurityScanner` whose injection, PII, credential, and NER heads all execute on the host. The [guardrails reference](../../core/guardrails.md) confirms every detection head runs locally by design; point each model-backed head at a pre-staged directory.
- **Lock the sandbox to no network.** Any agent-executed code runs in a container with the network cut, seccomp syscall filtering, ~40 dropped Linux capabilities, and a read-only root filesystem. The [sandbox reference](../../core/sandbox.md) documents `NetworkMode.NONE` and the hardening profile.
- **Keep memory local and scanned.** Use on-device embeddings for vector memory, and let the scanner sanitise retrieved content so a poisoned memory cannot inject instructions back into the prompt.
- **Turn on chained audit.** Every sensitive tool call is written to a tamper-evident, HMAC-chained JSONL log you can verify offline — the backbone of `air-gapped agent audit logging`.
- **Confirm no phone-home.** Promptise makes no external pricing or telemetry calls, and the default [observability](../../core/observability.md) transporters write to local files — so there is no outbound trace to disable in the first place.

The differentiator is the last three items shipping as one pre-integrated stack. In Promptise, a network-isolated hardened sandbox, an integrated local scanner, and a tamper-evident audit chain are enabled together and certified as a single unit — not three tools you wire and re-verify at every seam.

## What other frameworks do today

It would be dishonest to claim other frameworks can't sandbox code — they can, and a bare sandbox is not the gap. Here is the field, stated precisely.

- **AutoGen** ships a real Docker code executor (`DockerCommandLineCodeExecutor`) that runs generated code inside a container. This is a genuine sandbox, and it is the recommended posture over local execution.
- **CrewAI** runs tool code in a Docker-based interpreter and offers a `safe` code-execution mode. Also real. Worth flagging separately: CrewAI enables anonymous telemetry by default, which is an outbound flow you opt out of via an environment variable before a sovereign deployment.
- **Hugging Face smolagents** supports a Docker sandbox or the E2B *cloud* sandbox — the latter being a hosted service, a non-starter in an air gap unless you deliberately pick the local backend.

So the sandbox itself is table stakes, and this checklist won't pretend otherwise. The honest delta is threefold. First, **default hardening**: these executors isolate a process, but a no-network posture plus ~40 dropped capabilities, `no-new-privileges`, and a read-only rootfs as the *default* — rather than flags you assemble — is where the postures diverge. Second, **the audit chain**: a Docker executor gives you isolation, not a tamper-evident record of what ran; regulated teams add a separate audit tool and re-certify the seam. Third, **integration**: none of these bundles a network-isolated hardened sandbox *plus* an HMAC-chained audit log *plus* an on-device injection/PII/credential scanner into one offline-capable, certifiable unit. The accurate framing is not "competitors lack a sandbox" — it is that a `regulated industry on-prem AI agent` otherwise stitches sandbox, audit, and moderation from separate parts and re-verifies the joins on every dependency bump. Promptise's edge is structural: the three arrive pre-integrated and air-gap-capable, so you audit one stack once.

## One stack: local guardrails, no-network sandbox, chained audit

Here is the agent side of the checklist in a single call — local inference, on-device guardrail heads loaded from pre-staged directories, a no-egress sandbox, and a local-only observability report. Nothing in this path opens an outbound socket.

```python
import asyncio

from promptise import (
    build_agent,
    PromptiseSecurityScanner,
    InjectionDetector,
    PIIDetector,
    CredentialDetector,
    NERDetector,
)
from promptise.config import HTTPServerSpec


async def main():
    # Guardrails: every detection head runs on-device. Model-backed heads load
    # from folders you pre-staged across the air gap — nothing is fetched at
    # runtime.
    scanner = PromptiseSecurityScanner(
        detectors=[
            InjectionDetector(model="/opt/models/injection", threshold=0.9),
            PIIDetector(),         # 69 regex patterns, 0 MB, no network
            CredentialDetector(),  # 96 regex patterns, 0 MB, no network
            NERDetector(model="/opt/models/gliner-pii"),
        ],
    )
    scanner.warmup()  # load weights from disk now — fail loudly if any are missing

    agent = await build_agent(
        model="ollama:llama3",  # inference stays on-box via Ollama
        servers={"tools": HTTPServerSpec(url="http://localhost:8000/mcp")},
        instructions="You are an on-prem analyst. Assume no internet access.",
        guardrails=scanner,
        sandbox={
            "network": "none",  # NetworkMode.NONE (the default) — executed code cannot reach the network
            "memory_limit": "512M",
            "cpu_limit": 2,
            "timeout": 120,
        },
        observe=True,  # local HTML/JSONL report; nothing phones home
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Reconcile today's ledger and flag anomalies."}]}
    )
    print(result["messages"][-1].content)
    await agent.shutdown()


asyncio.run(main())
```

The third leg of the stack lives on the MCP tool server the agent calls. `AuditMiddleware` writes one JSONL line per tool call and links each entry to the hash of the previous one, so a single altered, deleted, or reordered line breaks the chain:

```python
import os

from promptise.mcp.server import MCPServer, AuditMiddleware

server = MCPServer(name="ledger-tools")

audit = AuditMiddleware(
    log_path="/var/log/promptise/audit.jsonl",
    signed=True,                                      # HMAC chain on (the default)
    hmac_secret=os.environ["PROMPTISE_AUDIT_SECRET"],  # stable secret => chain survives restarts
    include_args=True,
    include_result=False,                            # keep sensitive tool outputs out of the log
)
server.add_middleware(audit)
```

Read those two blocks as one certifiable unit. The `guardrails` scanner runs a local DeBERTa injection classifier and a local GLiNER NER head plus 165 pure-regex PII and credential patterns; the `sandbox` dict pins code execution to `NetworkMode.NONE`; and `AuditMiddleware` produces the tamper-evident record. Each verified-identity descriptor (subject, issuer, roles) is captured in the audit entry, never the token — so the log answers *which agent did what* without leaking a secret across the gap.

## Verify the perimeter before go-live

"Runs offline" is a claim you hand an auditor, not a vibe. Each checklist item has a specific thing you prove.

- **Models never fetch.** `scanner.warmup()` forces every model to load from disk at startup, so a missing or misplaced artifact fails immediately and loudly instead of on the first user request. Pre-load steps for each weight are in the [guardrails reference](../../core/guardrails.md) under *Local models (air-gapped / offline)*.
- **Code cannot reach the network.** `network: "none"` maps to `NetworkMode.NONE` (also the default, but pin it so the config documents the intent), so executed code physically cannot open a connection. The [sandbox reference](../../core/sandbox.md) also documents the seccomp filter, the ~40 dropped capabilities, and the read-only rootfs that harden the container beyond just the network cut; a locked-down host can go further with the gVisor `runsc` runtime.
- **The audit chain is intact.** Call `verify_chain()` to confirm no entry was altered, deleted, or reordered — it recomputes every HMAC from the stored `prev_hash` and returns `False` on any break:

```python
assert audit.verify_chain()  # False if any line was altered, deleted, or reordered
```

  Set `PROMPTISE_AUDIT_SECRET` to a stable value so the chain remains verifiable across restarts. `AuditMiddleware` is one of the middleware documented in the [MCP production features guide](../../mcp/server/production-features.md).
- **No trace leaves the host.** The default [observability](../../core/observability.md) transporters write an HTML report and JSONL log lines to a local directory; the Prometheus transporter exposes an endpoint your in-cluster scraper pulls. Promptise makes no external pricing or telemetry calls, so there is nothing outbound to switch off.

Because these are integrated defaults of one framework rather than five separate integrations, you audit the stack's egress posture once instead of re-certifying every seam each time a transitive dependency bumps a version.

## Frequently asked questions

### Isn't a Docker sandbox enough to pass a regulated review?

A sandbox is necessary but not sufficient, and this checklist is deliberate about saying so. AutoGen and CrewAI both ship real Docker executors, so isolation alone is not the differentiator. A regulated review also asks: is the sandbox network-off and hardened *by default* (seccomp, dropped capabilities, read-only rootfs), is there a tamper-evident record of what ran, and is dangerous input blocked before it reaches a tool? Promptise ships the no-network hardened sandbox, the HMAC-chained audit log, and the on-device scanner as one pre-integrated unit, which is what turns "we sandbox code" into a defensible sign-off.

### How does the audit log prove tamper-evidence offline?

Each entry stores `prev_hash` (the HMAC of the previous entry) and its own `hmac` over the sorted payload, forming a chain from a genesis hash. `verify_chain()` recomputes every HMAC from your `PROMPTISE_AUDIT_SECRET` and returns `False` if any line was altered, deleted, or reordered — no external service required. That is exactly the property `air-gapped agent audit logging` needs: verification runs entirely on-box.

### Which layers still need a one-time internet connection?

Only the initial download of the model weights: the LLM (via `ollama pull`), the DeBERTa injection classifier, the optional GLiNER NER head, and the embedding model. Do this once on a connected machine, copy the folders across the gap, and reference each by local path. After that, every request runs with zero outbound connections. [Why AI Agent Frameworks Fail in Air-Gapped Networks](air-gapped-ai-agent.md) walks through the hidden dependencies that catch teams here.

### Does the audit log leak sensitive data?

Not by default. `include_args` and `include_result` are both off unless you opt in, and even with identity capture enabled the log records only descriptors — subject, issuer, audience, roles — never the bearer token or the full claim set. For a `regulated LLM agent no egress` posture, keep `include_result=False` so tool outputs stay out of the log entirely.

## Next steps

Work through the checklist in order: stage the model and guardrail weights on a connected machine and reference them by path per the [model setup guide](../../getting-started/model-setup.md); enable the on-device scanner from the [guardrails reference](../../core/guardrails.md); lock code execution to no network with the [sandbox reference](../../core/sandbox.md); turn on `AuditMiddleware` and run `verify_chain()` before go-live using the [MCP production features guide](../../mcp/server/production-features.md); and confirm local-only tracing via the [observability reference](../../core/observability.md). If you are still mapping the hidden cloud dependencies in your current setup, start with [Why AI Agent Frameworks Fail in Air-Gapped Networks](air-gapped-ai-agent.md), then use [Air-Gapped AI Agent Framework: The On-Prem Guide](air-gapped-agent-framework.md) as the full layer-by-layer reference.
