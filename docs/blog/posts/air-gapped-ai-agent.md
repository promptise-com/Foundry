---
title: "Why AI Agent Frameworks Fail in Air-Gapped Networks"
description: "Most teams discover too late that 'model-agnostic' does not mean 'offline-capable': even when the LLM runs via Ollama, the default embedding calls…"
keywords: "air-gapped AI agent, run an AI agent fully offline, on-prem agent no internet, offline agent framework, no internet AI agent"
date: 2026-07-16
slug: air-gapped-ai-agent
categories:
  - Air-Gapped & Sovereign
---

# Why AI Agent Frameworks Fail in Air-Gapped Networks

Building an **air-gapped AI agent** looks deceptively simple: swap the hosted LLM for a local one, unplug the network cable, done. Then the first request fails — not because the model is missing, but because the embedding call, the moderation endpoint, or the tracing exporter tried to reach a host that no longer exists. Teams learn the hard way that "model-agnostic" is not the same as "offline-capable." The model is only one of five layers in a modern agent, and the other four usually assume an internet connection you don't have. This post maps the hidden cloud dependencies that break inside an isolated network, and shows which layers have to be local before you can honestly say you run an AI agent fully offline.

<!-- more -->

## The hidden cloud dependencies that break an air-gapped agent

When people say an agent framework is "local," they almost always mean the LLM. But an agent is a pipeline, and every stage can quietly reach outbound:

- **Embeddings.** Memory, RAG, and semantic tool selection all embed text. Many stacks default the embedding step to a hosted API (OpenAI `text-embedding-3`, Cohere, Voyage) even when the chat model is local. The chat model runs on your GPU; the embeddings still leave the building.
- **Moderation and injection defense.** Content safety is frequently sold as a hosted service — OpenAI's moderation endpoint, Azure AI Content Safety, or third parties like Lakera. Every prompt and completion you scan gets shipped off-host, which is a non-starter when the data can't leave the room.
- **Tracing and telemetry.** Hosted observability (LangSmith, Logfire, and friends) is the recommended debugging path in several ecosystems, and some frameworks emit anonymous usage telemetry by default. Both are outbound run data a security review will flag.
- **Cost and pricing lookups.** Some tooling pings a pricing service to estimate spend. On an isolated network that call just hangs.
- **Vector store.** Hosted vector databases (Pinecone, and managed variants of others) are the documented "for scale" option, so the easy path points at a SaaS endpoint.

In an `on-prem agent no internet` deployment, any one of these either fails hard or — worse — silently egresses data you promised would never leave. The failure is rarely the model. It's the plumbing around it.

## What "model-agnostic" actually gets you — and what it doesn't

Let's be precise and fair about the current field, because this is where the real differentiation lives.

**What other frameworks do today.** LangChain, LangGraph, CrewAI, AutoGen, LlamaIndex, and Pydantic AI can all point inference at a local model — typically Ollama or a self-hosted server — so local *inference* is genuinely solved everywhere. Most of them also support local embeddings as an option (LangChain's `HuggingFaceEmbeddings`, LlamaIndex's local embedding classes, sentence-transformers under the hood) and a local vector store like Chroma or FAISS. Those layers are not unique to anyone, and it would be dishonest to imply otherwise.

The gap is not "can a layer run locally" — it's "does the assembled stack run locally *by default*, or do you have to re-plumb each seam yourself." Two seams in particular stay cloud-shaped:

- **Security and moderation.** No mainstream agent framework bundles a local ML prompt-injection classifier plus content moderation into the agent's input/output path as a wired-in default. Detection is left to you, or the documented path is a hosted moderation API. NeMo Guardrails and Guardrails AI can run some rails locally, but you supply and host the models and wire them in yourself — they are separate libraries, not a default agent layer.
- **Telemetry.** CrewAI emits anonymous telemetry by default (you opt out via an environment variable). The LangChain/LangGraph ecosystem steers teams toward LangSmith, and Pydantic AI toward Logfire — both hosted tracing services. These are opt-in, but they are the paved road, so the "happy path" your team copies from the docs sends run data off-host unless someone remembers to disable it.

So the honest delta is not that competitors *lack* offline capability. It's that offline is an assembly project you perform per layer and re-verify at audit time. Promptise Foundry's edge is structural: local model, local embeddings, local guardrails, and local telemetry are the *wired-in default*, and the framework's "no fallbacks" design means nothing silently swaps to a hosted service behind your back. That is the difference between "possible to run offline" and being an **offline agent framework**.

## The layers that must be local — and how Promptise wires them

Here is a single agent with all four risky layers pinned local: the model via Ollama, memory embeddings via a local sentence-transformers model, a persistent local vector store, and a security scanner whose detection heads run on-device. Nothing in this pipeline makes an outbound request.

```python
import asyncio

from promptise import (
    build_agent,
    PromptiseSecurityScanner,
    InjectionDetector,
    PIIDetector,
    CredentialDetector,
)
from promptise.memory import ChromaProvider


async def main():
    # 1. Local guardrails. The DeBERTa injection model runs on-device;
    #    PII and credential heads are pure regex (0 MB, no network).
    scanner = PromptiseSecurityScanner(
        detectors=[
            InjectionDetector(),   # local transformer; swap in a local path for air-gap
            PIIDetector(),         # 69 regex patterns, no model download
            CredentialDetector(),  # 96 regex patterns, no model download
        ],
    )
    scanner.warmup()  # load models from disk now, not on the first message

    # 2. Local vector memory. Default embeddings (all-MiniLM-L6-v2)
    #    run locally with no API key; persisted to disk.
    memory = ChromaProvider(
        collection_name="agent_memory",
        persist_directory=".promptise/chroma",
    )

    # 3. Local model via Ollama — no API key, no outbound request.
    agent = await build_agent(
        servers={},  # or your own on-prem MCP servers
        model="ollama:llama3",
        instructions="You are an offline operations assistant.",
        guardrails=scanner,
        memory=memory,
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Summarize today's pipeline health."}]}
    )
    print(result["messages"][-1].content)
    await agent.shutdown()


asyncio.run(main())
```

Every choice here is deliberate. `model="ollama:llama3"` keeps inference on your hardware; the [model setup guide](../../getting-started/model-setup.md) documents the `provider:model-name` string and the Ollama path. `ChromaProvider` defaults to the `all-MiniLM-L6-v2` embedding model, which runs locally with no API key, and `persist_directory` keeps the vectors on disk — the [memory reference](../../core/memory.md) covers the provider protocol and per-user isolation if you need multi-tenant scoping. The `PromptiseSecurityScanner` gives you injection, PII, and credential detection in the agent loop itself; the [guardrails reference](../../core/guardrails.md) lists all six detection heads and confirms that every model runs locally by design.

The point is not that each of these *can* be made local — it is that they are local in one `build_agent()` call, with no seam left cloud-shaped.

## Pre-loading models so the host can run with the network unplugged

"Runs locally" still means the model weights have to arrive somehow. The injection classifier (~750 MB), the optional GLiNER NER head (~500 MB), the sentence-transformers embedding model, and the Ollama model all download from their registries on first use and cache on disk. For a `no internet AI agent`, you stage those artifacts once on a connected machine, copy them across the boundary, and point each component at a local directory.

Every model-backed detector accepts a local path instead of a registry ID:

```python
scanner = PromptiseSecurityScanner(
    detectors=[
        InjectionDetector(model="/opt/models/injection"),  # pre-staged DeBERTa dir
        PIIDetector(),
        CredentialDetector(),
    ],
)
scanner.warmup()  # verifies the models load from disk before any traffic
```

`warmup()` is what makes this safe: it forces every model to load at startup, so a missing or misplaced artifact fails immediately and loudly instead of on the first user request. Pre-stage the Ollama model the same way (`ollama pull` on the connected host, then transfer the model store), and the ChromaDB embedding model similarly. Our step-by-step walkthrough, [How to Pre-Load LLM Guardrail Models on an Air-Gapped Host](offline-guardrail-models.md), covers the transfer and verification workflow for each artifact so nothing reaches for HuggingFace at runtime.

## No phone-home, no pricing calls

The last outbound risk is the one teams forget because it is not a feature they turned on. Promptise ships **no default telemetry and makes no external cost or pricing calls** — a deliberate design decision, since the framework doesn't control external providers and won't estimate their prices. Observability is local-first: the built-in transporters write an HTML report, a JSON file, a structured log, the console, or a Prometheus endpoint you host yourself. Tracing your agent does not require sending run data to anyone.

That closes the loop for a truly `on-prem agent no internet` posture: local model, local embeddings, local vector store, local guardrails, local traces. Unplug the cable and the agent keeps working — because there was never a hosted dependency in the critical path to begin with.

## Frequently asked questions

### Can I run a Promptise agent with no internet at all?

Yes. Point `model` at a local Ollama model, use `ChromaProvider` for memory (its default embeddings run on-device), enable the `PromptiseSecurityScanner` with local model paths, and rely on the local observability transporters. Pre-stage all model weights on a connected machine first, run `warmup()` to confirm they load from disk, and the agent runs with the network unplugged.

### Isn't a local LLM enough to be "offline"?

No, and that is the trap. A local LLM only handles inference. Embeddings for memory and RAG, injection and content moderation, and tracing each have their own network defaults. If any of them points at a hosted endpoint, your "offline" agent still egresses. Offline means every layer is local, not just the model.

### Do other frameworks support offline deployment?

They support the pieces. LangChain, CrewAI, AutoGen, LlamaIndex, and Pydantic AI can all run a local model, and most support local embeddings and a local vector store as options. What none ships as a wired-in default is an integrated, on-device security and moderation layer plus local-only telemetry — those you assemble and re-verify per layer. Promptise makes them the default so there is no seam to re-plumb.

### Where do the guardrail models come from in an air-gapped setup?

Download them once on a connected host and reference the local directory: `InjectionDetector(model="/opt/models/injection")`, `NERDetector(model="/opt/models/gliner-pii")`. The regex-based PII and credential heads need no model at all. See the [guardrails reference](../../core/guardrails.md) for the full list of heads and their local-path options.

## Next steps

Start by pointing Promptise at a local model: follow the [model setup guide](../../getting-started/model-setup.md) to pull an Ollama model and run your first fully offline agent, then enable local guardrails and `ChromaProvider` memory from the code above. When you are ready to lock down the whole stack — sandboxed code execution, tamper-evident audit, and a `.superagent` deployment artifact your security review can point at — work through the [Air-Gapped AI Agent Framework: The On-Prem Guide](air-gapped-agent-framework.md).
