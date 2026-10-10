---
title: "Air-Gapped AI Agent Framework: The On-Prem Guide"
description: "The definitive hub for building agents that run with zero outbound internet: every layer — model inference, embeddings, vector memory, guardrails…"
keywords: "air-gapped agent framework, on-prem AI agent framework, self-hosted AI agent, fully offline agent stack, local-first agent framework"
date: 2026-07-16
slug: air-gapped-agent-framework
categories:
  - Air-Gapped & Sovereign
---

# Air-Gapped AI Agent Framework: The On-Prem Guide

Choosing an **air-gapped agent framework** is not the same as choosing a framework that can talk to a local model — and that distinction is exactly where most on-prem projects stall. Swapping a hosted LLM for Ollama is the easy 20%. The hard 80% is every other layer of a modern agent: the embedding model behind your vector memory, the moderation call that scores toxic output, the injection classifier guarding your input, the code sandbox your agent executes in, and the telemetry exporter shipping traces somewhere for debugging. Each one is a potential outbound connection, and in an isolated network a single one breaks the whole system — or worse, quietly passes a compliance review it should have failed. This guide is the hub for the whole picture: it maps every layer of an agent to a local implementation, shows the exact configuration that keeps each layer on-box, and is honest about what a truly **fully offline agent stack** requires versus what "supports local models" actually buys you.

<!-- more -->

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## The five layers that must go local

An agent is not one dependency; it is a stack. To run one inside an air gap, every layer in the request path has to resolve to something on the host. Here is the map most teams only draw after their first failed deployment:

| Layer | What it does | Local implementation |
|---|---|---|
| Model inference | Generates the completion | Ollama or any OpenAI-compatible endpoint on-box |
| Embeddings | Turns memory + queries into vectors | `all-MiniLM-L6-v2` inside `ChromaProvider`, runs on CPU |
| Vector memory | Long-term recall and RAG retrieval | `ChromaProvider` with a local `persist_directory` |
| Guardrails | Injection detection, PII/credential redaction, content safety | `PromptiseSecurityScanner` — DeBERTa, GLiNER, Llama Guard, all on-device |
| Code execution | Runs agent-written or tool code | Docker sandbox with `NetworkMode.NONE` |
| Observability | Traces, metrics, debugging | Local HTML/JSON report or a Prometheus endpoint you scrape |

The insight behind a **local-first agent framework** is that these layers are not independent checkboxes — they share caller identity, they feed each other (retrieved memory becomes model input; model output becomes sandbox code), and they all have to be verified together for an air-gap sign-off. When each layer is a separate integration you bolted on yourself, "prove nothing egresses" becomes a per-seam audit you repeat every time a dependency updates. When the layers are integrated defaults of one framework, the proof is structural. That is the difference this guide is built around, and it is covered end to end in the companion piece [Why AI Agent Frameworks Fail in Air-Gapped Networks](air-gapped-ai-agent.md).

## What other frameworks do today

It would be dishonest to claim other frameworks can't run offline. They can run large parts of the stack locally, and it is worth being precise about exactly how far each gets, because the gap is narrower and sharper than the marketing on either side suggests.

- **Local model inference** — LangChain, LlamaIndex, CrewAI, and AutoGen can all point the LLM at a local Ollama or OpenAI-compatible endpoint. This layer is genuinely solved everywhere; nobody has an edge here.
- **Local embeddings and vector store** — LangChain and LlamaIndex expose local embedding classes (sentence-transformers/HuggingFace) and local vector stores (Chroma, FAISS, Qdrant). "Local memory" is table stakes, not a differentiator, and this guide won't pretend otherwise.
- **Guardrails** — this is where the honest gap opens. None of these frameworks ships an ML prompt-injection classifier plus PII/credential redaction plus content safety wired into the agent's input/output path *by default*. You add [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) or [Guardrails AI](https://github.com/guardrails-ai/guardrails) and supply and wire the models yourself, or you route to a cloud service — OpenAI's moderation endpoint and Azure AI Content Safety are both hosted APIs that require sending prompts off-host, which is a non-starter in an air gap.
- **Code sandbox** — several frameworks offer sandboxing, and it's a partial win, not a blank: CrewAI can run tool code in a Docker-based interpreter, and Hugging Face's smolagents supports Docker or the E2B cloud sandbox. The delta is the default posture. These sandboxes are opt-in add-ons that often assume network access; a no-egress configuration is something you assemble and verify separately.
- **Telemetry** — CrewAI enables anonymous telemetry by default (opt-out via an environment variable), and the LangChain ecosystem steers teams toward LangSmith, a hosted tracing service. Both are outbound data flows you have to notice and disable before a sovereign deployment.

So the accurate framing is not "competitors lack offline support." It is that **an on-prem AI agent framework is only as offline as its most cloud-shaped layer**, and for these frameworks that layer is almost always guardrails, moderation, or telemetry. You *can* assemble a fully local stack from parts — thousands of teams have — but you own the assembly and, critically, the re-verification every time a layer changes. Promptise's edge is making injection detection, PII redaction, content safety, embeddings, vector memory, and a no-network sandbox integrated, offline-capable defaults of a single agent, so the whole stack is offline because the framework is, not because you kept every seam honest by hand.

## Assemble the full stack offline in one build_agent()

Here is the entire local stack — local model, local guardrail heads, local-embedding vector memory, a no-network sandbox, and a local-only observability report — declared in one call. Every model-backed guardrail head points at a pre-downloaded directory so nothing is fetched at runtime.

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
from promptise.memory import ChromaProvider


async def main():
    # 1. Guardrails — every detection head runs on-device. The model-backed
    #    heads load from local directories you pre-populated on a connected
    #    machine, so nothing is fetched from HuggingFace at runtime.
    scanner = PromptiseSecurityScanner(
        detectors=[
            InjectionDetector(model="/opt/models/injection", threshold=0.9),
            PIIDetector(),         # 69 regex patterns, 0 MB, no network
            CredentialDetector(),  # 96 regex patterns, 0 MB, no network
            NERDetector(model="/opt/models/gliner-pii"),
        ],
    )
    scanner.warmup()  # load models from disk now, not on the first message

    # 2. Vector memory — ChromaProvider embeds locally with all-MiniLM-L6-v2
    #    and persists to disk. No embedding API call ever leaves the host.
    memory = ChromaProvider(
        collection_name="agent_memory",
        persist_directory=".promptise/chroma",
    )

    # 3. One agent: local inference, local guardrails, local memory,
    #    a no-egress sandbox, and a local HTML observability report.
    agent = await build_agent(
        model="ollama:llama3",  # inference stays on-box via Ollama
        servers={"tools": HTTPServerSpec(url="http://localhost:8000/mcp")},
        instructions="You are an on-prem analyst. Never assume internet access.",
        guardrails=scanner,
        memory=memory,
        sandbox={
            "network": "none",  # NetworkMode.NONE (the default) — code cannot reach the network
            "memory_limit": "512M",
            "cpu_limit": 2,
            "timeout": 120,
        },
        observe=True,  # writes a local HTML report to ./reports; nothing phones home
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Summarize today's pipeline health."}]}
    )
    print(result["messages"][-1].content)
    await agent.shutdown()


asyncio.run(main())
```

Read that block as a stack diagram. The `model` string keeps inference on-box. The `guardrails` scanner runs a local DeBERTa injection classifier and a local GLiNER NER model, plus 165 pure-regex PII and credential patterns that need no model at all. `ChromaProvider` embeds with a local CPU model and writes vectors to `.promptise/chroma`. The `sandbox` dict pins code execution to no network. And `observe=True` emits a self-contained HTML report to a local folder — the default transporter never opens a socket. Five layers, one declaration, no outbound connection anywhere in the path.

## Verifying zero egress

"Runs offline" is a claim you have to be able to hand an auditor, not a vibe. Each layer above has a specific thing you verify.

- **Model inference** — the provider string is `ollama:...` or a `BaseChatModel` pointed at an on-box endpoint. The [model setup guide](../../getting-started/model-setup.md) documents all three ways to pass a model, including a pre-configured LangChain client with a custom `base_url` if your local endpoint isn't Ollama.
- **Embeddings** — `ChromaProvider` defaults to `all-MiniLM-L6-v2`, which runs locally with no API key. No OpenAI/Cohere embedding key means no embedding egress by construction.
- **Guardrails** — every model-backed detector accepts a local directory in its `model=` parameter. The [guardrails reference](../../core/guardrails.md) has the exact two-step pattern: `save_pretrained(...)` the injection and GLiNER models on a connected machine, copy the folders across the gap, then reference `/opt/models/...`. The regex heads download nothing ever. For local content safety, `ContentSafetyDetector(provider="local")` uses Llama Guard through Ollama instead of Azure. Pre-loading these weights is covered step by step in [How to Pre-Load LLM Guardrail Models on an Air-Gapped Host](offline-guardrail-models.md).
- **Code sandbox** — `network: "none"` maps to `NetworkMode.NONE` (also the default), so agent-executed code physically cannot open a connection. The [sandbox reference](../../core/sandbox.md) also documents the seccomp syscall filter, the ~40 dropped Linux capabilities, and the read-only root filesystem that harden the container beyond just the network cut. For a locked-down host you can go further with the gVisor `runsc` runtime.
- **Observability** — the default HTML and JSON transporters write to a local directory; the Prometheus transporter exposes an endpoint your in-cluster scraper pulls from. Promptise makes no external pricing or telemetry calls, so there is no default outbound trace to disable in the first place.

The verification story is the whole point of choosing an integrated stack. Because these are defaults of one framework rather than five separate integrations, you audit the framework's egress posture once instead of re-certifying every seam each time a transitive dependency bumps a version.

## The on-prem quickstart in three steps

To stand up your first **self-hosted AI agent** with no internet in the request path:

1. **Install and pull a local model.** `pip install "promptise[all]"`, install [Ollama](https://ollama.com), then `ollama pull llama3`. Point Promptise at it with `model="ollama:llama3"` — that one string is the entire model-provider change, as shown in the [model setup guide](../../getting-started/model-setup.md).
2. **Pre-load the guardrail models.** On a connected machine, `save_pretrained` the injection and GLiNER weights, copy the folders across the air gap, and reference them with `InjectionDetector(model="/opt/models/injection")` and `NERDetector(model="/opt/models/gliner-pii")`. Call `scanner.warmup()` at startup so nothing loads lazily mid-request.
3. **Turn on the local defaults.** Pass `guardrails=scanner`, `memory=ChromaProvider(persist_directory=...)`, and `sandbox={"network": "none"}` to `build_agent()`. That is the full local stack — inference, embeddings, memory, injection/PII/content-safety guardrails, and no-egress code execution — running as integrated defaults rather than five hand-wired integrations.

## Frequently asked questions

### Does "air-gapped agent framework" just mean using a local LLM?

No, and that misconception is the single most common reason on-prem projects fail late. The model is one of at least five layers in the request path. Embeddings, guardrail models, content moderation, code execution, and telemetry each have their own default endpoints, and any one of them reaching for the network breaks the air gap. A genuinely **fully offline agent stack** requires every layer to resolve locally — which is exactly what the `build_agent()` call above assembles.

### Can't I build the same offline stack on LangChain or LlamaIndex myself?

Yes — this guide is deliberate about saying so. LangChain and LlamaIndex give you local models, local embeddings, and local vector stores, and you can add local guardrails (NeMo Guardrails or Guardrails AI) and a sandbox on top. The difference is ownership: you assemble those pieces and, more importantly, re-verify the seams every time a dependency changes. Promptise ships injection detection, PII redaction, content safety, embeddings, vector memory, and a no-network sandbox as integrated defaults of one agent, so the assembly and the re-verification are the framework's job, not yours.

### Which layers still need a one-time internet connection?

Only the initial download of the guardrail models (DeBERTa for injection, GLiNER for NER, optionally Llama Guard for content safety) and the embedding model. You do this once on a connected machine, copy the folders across the gap, and reference them by local path. After that, every request runs with zero outbound connections. The [offline guardrail models guide](offline-guardrail-models.md) walks through the pre-load and transfer step in detail.

### Is telemetry sent anywhere by default?

No. Promptise makes no external pricing or telemetry calls, and the default observability transporters write to local files. This is a deliberate contrast with frameworks that enable anonymous telemetry by default or steer you toward a hosted tracing service — both of which are outbound data flows a sovereign deployment has to justify or disable.

## Next steps

Start the on-prem quickstart: install Promptise, follow the [model setup guide](../../getting-started/model-setup.md) to point at a local Ollama model, and enable local guardrails from the [guardrails reference](../../core/guardrails.md) in three steps. Before go-live, lock code execution to no network with the [sandbox reference](../../core/sandbox.md), and pre-stage every guardrail weight using [How to Pre-Load LLM Guardrail Models on an Air-Gapped Host](offline-guardrail-models.md). If you're still mapping the hidden cloud dependencies in your current setup, [Why AI Agent Frameworks Fail in Air-Gapped Networks](air-gapped-ai-agent.md) is the layer-by-layer diagnosis to read first.
