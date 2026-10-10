---
title: "How to Deploy a Sovereign AI Agent for EU Data Residency"
description: "A step-by-step deployment where every byte stays in-region: local model inference, local embeddings and vector store, local guardrails, and a no-network…"
keywords: "sovereign AI agent EU data residency, EU data residency AI agent, no US cloud AI agent, GDPR-compliant agent deployment, data sovereignty LLM agent"
date: 2026-07-16
slug: sovereign-ai-agent-eu-data-residency
categories:
  - Air-Gapped & Sovereign
---

# How to Deploy a Sovereign AI Agent for EU Data Residency

When a European bank, hospital group, or public-sector team specifies a **sovereign AI agent EU data residency** deployment, they mean something exact and non-negotiable: every byte the agent touches — prompts, retrieved context, tool arguments, generated code — must stay on infrastructure inside the region, on components you operate, with no packet crossing to a US-hosted API. That is harder than pointing your model string at a European endpoint, because an agent framework has more channels to the internet than the model call. This post walks through a deployment where inference, embeddings, the vector store, the safety guardrails, and code execution are all bound to local, no-egress components — and, critically, where that binding lives in one declarative `.superagent` file an auditor can read top to bottom instead of trusting five separate integrations.

<!-- more -->

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## The five things a sovereign deployment has to keep in-region

"Data residency" for a normal web app is a database region setting. For an agent it is a checklist of five distinct channels, each of which can quietly reach a US cloud if you let a default stand:

- **Model inference.** The obvious one. If your model string is `openai:...` your prompts go to OpenAI in the US. A sovereign build points inference at a local model — Ollama, or an OpenAI-compatible server like vLLM running in your own DC.
- **Embeddings.** Every memory search and every semantic tool-selection pass turns text into a vector. If that embedding call goes to a hosted embeddings API, your text left the region *before* the LLM ever saw it. This is the channel teams miss most often.
- **The vector store.** Where memory persists. A hosted vector database is a copy of your data in someone else's cloud. Residency requires a local, on-disk store.
- **Guardrails / moderation.** Prompt-injection and content-safety checks that call a hosted moderation API ship the exact text you were trying to protect to a third party.
- **Code execution.** If the agent runs code, the sandbox must not be able to phone home — a container with network access is an egress path with a shell attached.

A `no US cloud AI agent` has to close all five. Miss one and your `EU data residency AI agent` has a leak that a determined auditor will find. The design question is not "can each piece run locally" — usually it can — but "is there one place I can look to confirm all five are pinned, and that a config change can't silently re-open one?"

## What other frameworks do today

A fair comparison names actual behavior. Here is where the popular frameworks genuinely stand, stated as precisely as I can.

**LangChain and LangGraph** are model-agnostic and fully capable of a local stack. You can point a chat model at Ollama or at a local OpenAI-compatible server via `base_url`, use `HuggingFaceEmbeddings` for local embeddings, and persist to a local Chroma or FAISS store. None of that is off-limits. Two honest deltas remain. First, the *default road* — the quickstarts, the most-copied tutorials — reaches for `ChatOpenAI` and OpenAI embeddings, so the safe local configuration is something you assemble by overriding defaults across several imports. Second, LangChain's core ships no bundled, local safety model; moderation is either the hosted OpenAI Moderation endpoint or a separate library (NeMo Guardrails, Guardrails AI) you add and wire yourself. The capability exists; it is spread across integrations you compose in imperative Python, and there is no single file that pins the whole stack.

**CrewAI** likewise supports local LLMs through LiteLLM, so you can route inference to Ollama. The nuance worth stating for a residency review: CrewAI's built-in memory uses OpenAI embeddings by default, and you have to configure a local embedder explicitly to change that — precisely the embeddings channel above, defaulting to a US API unless you intervene. CrewAI does have YAML for agents, tasks, and crews, but that YAML describes *roles and workflow*, not a binding of model-plus-embeddings-plus-moderation-plus-vector-store-plus-sandbox to no-egress components.

The honest summary: none of these frameworks *prevents* a sovereign deployment, and it would be wrong to say they can't do it. The delta is that "keep everything in-region" is an assembly you perform across multiple integrations, with hosted defaults in a few of them, and — most importantly — no single artifact that states the whole residency posture in one place. Proving data residency becomes a per-integration audit that has to be repeated every time a dependency updates or a teammate forgets an override. Promptise's edge is not a capability the others lack; it is making the *whole-stack, no-egress binding* structural and first-class — one file, one audit.

## One `.superagent` file that binds the whole stack in-region

Promptise's `.superagent` file is a single YAML document, validated against a strict schema, that declares an entire agent: model, memory, embeddings, guardrails, sandbox, and MCP servers. For a sovereign build, that means every one of the five channels is pinned in one readable place, and the schema uses `extra="forbid"`, so a typo'd field is a hard validation error rather than a silently-ignored setting.

Here is a `data sovereignty LLM agent` where every layer resolves to a local, in-region component:

```yaml
# sovereign.superagent — every channel pinned to a no-egress component
version: "1.0"

agent:
  model:
    provider: openai                    # OpenAI-compatible protocol...
    name: llama-3.1-8b-instruct
    api_key: "${LOCAL_LLM_KEY:-not-needed}"
    base_url: "http://10.0.0.5:8000/v1" # ...pointed at your in-region vLLM
  instructions: "Internal operations assistant. Answer only from provided tools."
  trace: true

memory:
  provider: chroma                      # local, on-disk vector store
  collection: eu_ops
  persist_directory: "/srv/promptise/chroma"

optimize_tools:
  level: semantic
  embedding_model: "/models/all-MiniLM-L6-v2"   # local path — no HF download
  top_k: 8

guardrails:
  detect_injection: true                # local DeBERTa classifier
  detect_pii: true                      # local regex + local GLiNER NER
  detect_credentials: true              # local pattern matching
  detect_toxicity: false                # Llama Guard is local, enable if needed

sandbox:
  backend: docker
  image: "python:3.11-slim"
  network: none                         # the container has no network at all
  cpu_limit: 2
  memory_limit: "2G"
  timeout: 120

servers:
  internal_tools:
    type: http
    url: "${TOOLS_URL:-http://10.0.0.6:9000/mcp}"
    transport: streamable-http
    headers:
      Authorization: "Bearer ${TOOLS_TOKEN}"
```

Read that file as an auditor would, channel by channel:

- **Inference** uses the `openai` protocol adapter but `base_url` sends it to a machine on `10.0.0.5` — no traffic to OpenAI. The [model setup guide](../../getting-started/model-setup.md) covers this pattern, plus the even simpler `ollama:llama3` string when you want the model itself managed locally with zero key.
- **Embeddings** for semantic tool selection load from a filesystem path (`/models/all-MiniLM-L6-v2`), so an air-gapped host that can't reach Hugging Face still starts.
- **The vector store** is `chroma` with a `persist_directory` on local disk — memory never leaves the box.
- **Guardrails** run entirely on local models. Injection detection is a local DeBERTa transformer, PII detection combines local regex with a local GLiNER NER model, and credential detection is local pattern matching — all documented in the [guardrails guide](../../core/guardrails.md), which is explicit that all detection runs locally with no data leaving your infrastructure. (The one cloud option to avoid for content safety is Azure AI Content Safety; the local Llama Guard provider keeps that channel in-region too.)
- **The sandbox** sets `network: none`, the mode that gives a container no network access whatsoever, per the [sandbox guide](../../core/sandbox.md). Even if the agent writes code that tries to exfiltrate, there is no route out.

That is the whole point of the artifact: five channels, five local bindings, one file. Change any binding to a hosted component and it shows up as a diff in code review, not as an unexplained outbound flow discovered months later in a packet capture.

## Deploy the offline agent

The file is only useful if it runs. Loading a `.superagent` is three real calls: parse and validate, convert to build kwargs, and hand those to `build_agent()`. The snippet below writes a minimal-but-real sovereign config (local Ollama model, no-network sandbox), loads it, invokes it, and shuts down. It uses only the public loader API and needs no cloud key — a fitting first step for a `GDPR-compliant agent deployment`:

```python
import asyncio
from pathlib import Path

from promptise import build_agent
from promptise.superagent import load_superagent_file

# A real, loadable .superagent pinned to local components. Inference is a
# local Ollama model (no API key, no egress); the sandbox has no network.
SUPERAGENT = """
version: "1.0"

agent:
  model: "ollama:llama3"          # local inference, nothing leaves the host
  instructions: "You are an internal, in-region operations assistant."
  trace: true

sandbox:
  backend: docker
  image: "python:3.11-slim"
  network: none                   # container cannot reach any network
  cpu_limit: 1
  memory_limit: "1G"
  timeout: 60
"""


async def main() -> None:
    Path("sovereign.superagent").write_text(SUPERAGENT)

    # 1) parse + validate + resolve env vars, 2) convert, 3) build.
    loader, _cross_agents = load_superagent_file("sovereign.superagent")
    config = loader.to_agent_config()
    agent = await build_agent(**config.to_build_kwargs())

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Summarize today's incidents."}]}
    )
    print(result["messages"][-1].content)

    await agent.shutdown()


asyncio.run(main())
```

Prerequisites are exactly what the file names: a local Ollama daemon with `llama3` pulled, and a Docker daemon for the no-network sandbox. Swap in the full `sovereign.superagent` from the previous section to add local memory, local embeddings, and local guardrails once those models are staged on the host. Nothing in the load path reaches out to a hosted service — the loader reads YAML, resolves `${VAR}` placeholders from your environment, and calls `build_agent()`.

## What an auditor actually reads

The value of collapsing the whole residency posture into one file is that the audit stops being a code-reading exercise and becomes a config-reading one:

- **One artifact, one review.** An auditor opens `sovereign.superagent` and checks five lines: `base_url` is an internal address, `memory.persist_directory` is a local path, `embedding_model` is a filesystem path, `guardrails.*` are on, and `sandbox.network` is `none`. There is no need to trace which embedding client a memory helper happened to instantiate three call-frames deep.
- **It is greppable in CI.** "Reject any `.superagent` whose model `base_url` is not on the internal allowlist, or whose `sandbox.network` is not `none`" is a policy you can enforce as a lint rule on every pull request. Residency becomes a check, not a promise.
- **It is diffable.** Because the whole stack is declared, re-opening an egress channel is a visible change to a version-controlled file — the kind of thing review catches — rather than a default that regresses silently on a dependency bump.
- **It fails loud.** With `extra="forbid"`, a mistyped `netowrk: none` is a validation error at load time, not a setting that silently defaults back to a networked container.

For the wider on-prem picture — pinning dependencies, staging local model weights, and offline installs behind the physical boundary — see [Air-Gapped AI Agent Framework: The On-Prem Guide](air-gapped-agent-framework.md). And for the specific failure modes that catch teams by surprise when they first move an agent into a disconnected network, [Why AI Agent Frameworks Fail in Air-Gapped Networks](air-gapped-ai-agent.md) walks through each one.

## Frequently asked questions

### Doesn't every framework let me use a local model, so what's actually different here?

Yes — LangChain, LangGraph, and CrewAI can all route inference to a local model, and it would be wrong to claim otherwise. The difference is scope and defaults. Inference is only one of five residency channels; embeddings, the vector store, moderation, and code-execution egress are the other four, and in those other frameworks each is a separate integration you configure — with a hosted default in a couple of them (CrewAI's memory embeddings default to OpenAI; LangChain's easiest moderation path is the hosted OpenAI endpoint). Promptise's difference is that all five are declared and pinned in one `.superagent` file, so "everything stays in-region" is a single reviewable artifact rather than a posture reassembled across imperative code.

### How do I keep embeddings from leaving the region?

Two channels use embeddings: semantic tool selection (`optimize_tools.embedding_model`) and memory search. Set `embedding_model` to a local filesystem path so the sentence-transformers weights load from disk with no Hugging Face download, and use the `chroma` memory provider with a `persist_directory` on local storage. Both then compute vectors on-host; the text is never sent to a hosted embeddings API.

### Can the code sandbox exfiltrate data?

Not when you set `sandbox.network: none`. That mode gives the container no network access whatsoever, so even agent-written code that tries to open a socket has no route off the host. `none` is also the default; `restricted` (DNS plus outbound HTTP/HTTPS only, enforced with iptables inside the container) and `full` exist for cases that genuinely need outbound calls, but a sovereign deployment should keep the sandbox at `none` and enforce it in CI. The [sandbox guide](../../core/sandbox.md) documents the network modes and the other isolation layers.

### Are the guardrail models really local, or do they call a moderation API?

Prompt-injection, PII, credential, and NER detection all run on local models — a DeBERTa classifier, local regex, and a local GLiNER model — with no data leaving your infrastructure. Content safety has two providers: local Llama Guard (via Ollama) and cloud Azure AI Content Safety. For a residency-sensitive deployment, use the local provider and leave the Azure one unconfigured, and your moderation channel stays in-region like the rest.

### Is a local model good enough for real tool-calling work?

For many internal operations tasks, yes — but be deliberate about it. Local models vary in how reliably they call tools, so validate your specific model against your tool set before production. The `base_url` pattern lets you run a stronger open-weights model on your own vLLM or GPU host rather than a small laptop model, which is usually the right trade for a production `data sovereignty LLM agent`. The [model setup guide](../../getting-started/model-setup.md) covers both the Ollama string and the custom-endpoint form.

## Next steps

Copy the `sovereign.superagent` template above, change `base_url` to your in-region model endpoint (or use `ollama:llama3` for a fully local start), point `persist_directory` and `embedding_model` at local paths, and confirm `sandbox.network` reads `none`. Then run the load-and-invoke snippet and, ideally, run it once under a network monitor so you can hand a reviewer an empty connection table alongside the file. Start from the [model setup guide](../../getting-started/model-setup.md) to pin inference, the [guardrails guide](../../core/guardrails.md) to keep detection local, and the [sandbox guide](../../core/sandbox.md) to lock the execution channel — then deploy a no-egress, in-region agent whose entire residency posture fits on one screen.
