---
title: "How to Pre-Load LLM Guardrail Models on an Air-Gapped Host"
description: "An air-gapped host has no model-hub access at runtime, so a guardrail or embedding model that tries to auto-download on the first user message doesn't…"
keywords: "offline guardrail models, air-gapped model provisioning, load ML models without internet, local model path guardrails, pre-download transformer models offline"
date: 2026-07-16
slug: offline-guardrail-models
categories:
  - Air-Gapped & Sovereign
---

# How to Pre-Load LLM Guardrail Models on an Air-Gapped Host

Running **offline guardrail models** on an air-gapped host sounds like it should be a config flag, but the failure mode is subtle: a guardrail or embedding model that expects to auto-download from a model hub does not gracefully degrade when the network is gone — it hard-fails or blocks on the first user message, in production, at the exact moment you least want a surprise. The DeBERTa injection classifier, GLiNER NER model, Llama Guard, and the `all-MiniLM-L6-v2` embedding model that powers semantic tool selection all default to pulling weights from Hugging Face the first time they run. On a laptop that is fine. In a classified data center or a sovereign-cloud tenant with no egress, that first `from_pretrained()` call raises a connection error and takes your agent down with it.

<!-- more -->

This is the provisioning how-to that nobody writes. The fix is not a hack — Promptise Foundry lets you pass an explicit local path to every guardrail detector and to the embedding model, then load them all up front with a single `warmup()` call, so you can boot the process, prove it made zero outbound requests, and only then accept traffic.

## Why an offline guardrail model fails closed, not soft

The default provisioning story for local ML models is "download on first use, cache in `~/.cache/huggingface/`, run offline forever after." That works because the first run happens on a machine with internet. Air-gapped deployments break the assumption at step one: there is no first run with a network. So the model library reaches for the hub, the socket call times out or is refused, and the exception propagates up through your guardrail into the request path.

The reason this is worse than an ordinary missing dependency is *when* it surfaces. A guardrail model is loaded lazily — the injection classifier is not touched until the first message needs scanning. So the process starts cleanly, health checks pass, the deployment goes green, and the failure waits for the first real user. You get a 500 on a live request instead of a loud failure at boot. Worse, if the guardrail is your security boundary, a load failure that is caught and swallowed somewhere upstream can mean the message is processed *without* the scan that was supposed to protect it.

The whole point of pre-staging is to move that failure to the earliest, safest possible moment: to build time, on the connected machine where you provision, and to process startup, where a wrong path fails the container before it ever serves a request.

## The models to stage before any traffic

An air-gapped Promptise agent with full guardrails and semantic tool selection depends on four local models. Only the ones you actually enable need to be staged — a regex-only scanner needs zero downloads — but a full enterprise configuration touches all four.

| Model | Role | Detector / config | Size | How it loads |
|-------|------|-------------------|------|--------------|
| `protectai/deberta-v3-base-prompt-injection-v2` | Prompt-injection classification | `InjectionDetector(model=...)` | ~750 MB | `transformers` pipeline |
| `knowledgator/gliner-pii-edge-v1.0` | Zero-shot NER for names/addresses | `NERDetector(model=...)` | ~500 MB | `gliner` (separate install) |
| `all-MiniLM-L6-v2` | Embeddings for semantic tool selection | `ToolOptimizationConfig(embedding_model=...)` | ~90 MB | `sentence-transformers` |
| `llama-guard3` | 13-category content safety | `ContentSafetyDetector(provider="local")` | ~4.9 GB | Ollama model store |

Three of these — DeBERTa, GLiNER, and the MiniLM embedding model — are Hugging Face artifacts that live in a directory you can copy. Llama Guard is different: it runs through a local Ollama daemon, so provisioning it offline means seeding Ollama's model store rather than copying a `from_pretrained()` directory. That distinction matters for how you move the bytes, and the [guardrails reference](../../core/guardrails.md) documents each detector's model parameter in full. The [model setup guide](../../getting-started/model-setup.md) covers the separate question of your *LLM* provider — the same air-gap logic applies there if you serve the language model itself locally through Ollama.

## Pre-download the weights on a connected host

On a machine that still has internet, materialize each artifact into a directory you can carry across the air gap on approved media. Each of these commands writes real files you will later reference by path.

```bash
# 1. DeBERTa prompt-injection classifier → ./models/injection
python -c "
from transformers import pipeline
pipeline('text-classification',
         model='protectai/deberta-v3-base-prompt-injection-v2') \
    .save_pretrained('./models/injection')
"

# 2. GLiNER PII NER model → ./models/gliner-pii
python -c "
from huggingface_hub import snapshot_download
snapshot_download('knowledgator/gliner-pii-edge-v1.0',
                  local_dir='./models/gliner-pii')
"

# 3. sentence-transformers embedding model → ./models/all-MiniLM-L6-v2
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2').save('./models/all-MiniLM-L6-v2')
"

# 4. Llama Guard → seed the Ollama model store on the connected host,
#    then copy ~/.ollama/models to the air-gapped host's Ollama store.
ollama pull llama-guard3
```

Copy `./models/` to the air-gapped host — say, `/opt/models/` — and copy the Ollama model blobs into that host's Ollama store if you use content safety. Nothing here is Promptise-specific yet; this is the standard "download once, move the bytes" workflow. What Promptise adds is the wiring on the other side.

## Point every detector and the tool selector at a local path

This is the part that is usually left to environment variables and hope. In Promptise, every model-backed detector accepts a local directory in its `model` argument, the semantic tool optimizer's `embedding_model` accepts a local directory too (it defaults to `all-MiniLM-L6-v2`), and `scanner.warmup()` loads every model in the scanner before you accept a single request. One script provisions the whole security boundary from disk:

```python
import asyncio

from promptise import (
    build_agent,
    PromptiseSecurityScanner,
    InjectionDetector,
    PIIDetector,
    CredentialDetector,
    NERDetector,
    ToolOptimizationConfig,
    OptimizationLevel,
)
from promptise.config import HTTPServerSpec


async def main() -> None:
    # Every ML detector points at a directory on the air-gapped host.
    # PIIDetector and CredentialDetector are pure regex — no weights to load.
    scanner = PromptiseSecurityScanner(
        detectors=[
            InjectionDetector(model="/opt/models/injection", threshold=0.9),
            PIIDetector(),
            CredentialDetector(),
            NERDetector(model="/opt/models/gliner-pii"),
        ],
    )

    # Load EVERY model in the scanner now, at startup — not on the first message.
    # If a path is wrong or a file is missing, this raises here, before traffic.
    scanner.warmup()

    agent = await build_agent(
        servers={"tools": HTTPServerSpec(url="http://localhost:8000/mcp")},
        model="ollama:llama3",  # a locally served model — no LLM egress either
        guardrails=scanner,
        # Semantic tool selection reads its embedding model from disk.
        optimize_tools=ToolOptimizationConfig(
            level=OptimizationLevel.SEMANTIC,
            embedding_model="/opt/models/all-MiniLM-L6-v2",
        ),
    )

    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Summarize today's tickets."}]}
    )
    print(result["messages"][-1].content)
    await agent.shutdown()


asyncio.run(main())
```

Three things are worth calling out. First, `PIIDetector` and `CredentialDetector` are regex-only — 69 and 96 patterns respectively — so they add zero downloads and sub-millisecond scans even on an air-gapped box; you only stage weights for the ML detectors you enable. Second, `warmup()` is the single call that turns "lazy download on first message" into "eager load at startup": it walks the scanner's detectors and loads the injection and NER models immediately. Third, `embedding_model="/opt/models/all-MiniLM-L6-v2"` is the same knob the [tool optimization guide](../../core/tool-optimization.md) uses to cut 40–70% of tool-definition tokens — semantic selection stops being a cloud dependency and becomes a local file read. If you also enable `ContentSafetyDetector(provider="local")`, warmup confirms the detector is ready and the model itself is served by your local Ollama daemon rather than fetched from a hub.

## Prove there are no model-hub calls at runtime

Pre-staging is only convincing if you can *demonstrate* the process reaches out to nothing. The Hugging Face and `transformers` libraries honor offline flags — set them and any accidental hub lookup raises immediately instead of silently succeeding on a machine that happens to have a cache:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

With those exported, run the script above. Because `warmup()` loads every model at startup, a missing file, a typo'd path, or a model that still expects to phone home fails *at boot*, loudly, before the agent accepts traffic — exactly the property you want. Pair the flags with a locked-down network namespace (no default route, or an egress-deny policy) and the proof becomes airtight: if the process starts and answers, it did so with bytes that were already on the host. This is the difference between "we think it's offline" and "it demonstrably cannot reach the internet, and we verified it at startup." For the broader picture of running the whole stack this way, see the companion [air-gapped agent framework guide](air-gapped-agent-framework.md).

## What other frameworks do today

Most serious frameworks that use local ML models *can* run offline — the honest question is how much of the provisioning they make first-class versus how much they leave to you. Being precise about the delta matters more than scoring points.

- **Guardrails AI** lets you configure the model behind ML validators, and its validators run locally once their weights are present. Offline provisioning, though, largely rides on the Hugging Face cache and environment variables like `HF_HOME`/`HF_HUB_OFFLINE`: you warm the cache on a connected machine and carry it over. There is no single documented call that eagerly loads every validator's model across a guard before traffic — model loading is per-validator and generally lazy, so "did everything actually load?" is something you assemble yourself.
- **NVIDIA NeMo Guardrails** is explicit about models in `config.yml`, including the embedding model used for its rails (it defaults to `all-MiniLM-L6-v2` via sentence-transformers/fastembed), and you can point those at alternatives. Air-gapped operation is supported, but it again leans on the standard HF/sentence-transformers cache-and-offline-flag mechanism for weight provisioning rather than a one-call warm-up that guarantees every rail's model is resident before the first request.
- **Llama Guard and Prompt Guard used directly** are just Hugging Face models: `from_pretrained` with a local path works fine offline, but "load all my guardrail models now, from these paths" is glue you write around each `pipeline()` yourself.
- **LangChain** exposes local model integrations and honors the same offline environment flags, but guardrailing is composed from separate pieces (an injection model here, a PII library there), so there is no unified provisioning surface across them.

None of these frameworks *lack* offline support — that would be an unfair claim, and it is not the point. The delta is structural. In each case, air-gapped provisioning is spread across environment variables, per-component model arguments, and cache-warming rituals, and confirming that *every* guardrail model is loaded is left as an exercise. Promptise makes the capability first-class: an explicit local path on each detector and on the embedding model, and one `warmup()` that loads them all so a missing artifact fails at startup instead of on a live request. The offline path is a supported configuration you can verify, not a sequence of environment hacks you hope you got right. If you are weighing this against a cloud-first stack, the [why agent frameworks fail in air-gapped networks](air-gapped-ai-agent.md) post digs into the wider category of assumptions that break without egress.

## Frequently asked questions

**Do I have to stage all four models?**
No. You only stage weights for the ML detectors you enable. A scanner built from just `PIIDetector` and `CredentialDetector` is pure regex — 165 patterns, zero downloads — and runs fully offline with no provisioning at all. Add the injection or NER model only if you want ML-based detection, and the embedding model only if you turn on semantic tool selection.

**What exactly does `warmup()` load?**
It walks the scanner's configured detectors and eagerly loads their models — the injection classifier and the GLiNER NER model — logging each one. It is safe to call multiple times because loaded models are cached. Calling it at startup is what converts a lazy first-message download into an eager, verifiable startup load.

**Will `warmup()` accidentally reach the internet if a path is wrong?**
That is exactly what the offline flags are for. With `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` set, a wrong or missing local path raises at `warmup()` time instead of silently attempting a download. Run warmup behind those flags in a no-egress namespace and a successful boot is your proof that everything loaded from disk.

**Does the semantic tool optimizer download anything at runtime?**
Not when you pass a local directory. `ToolOptimizationConfig(embedding_model="/opt/models/all-MiniLM-L6-v2")` reads the model from disk. It defaults to `all-MiniLM-L6-v2`, which auto-downloads on a connected machine — so on an air-gapped host you pass the staged path instead, and embedding becomes a local file read.

**How do I provision Llama Guard offline — it isn't a Hugging Face directory?**
Correct. `ContentSafetyDetector(provider="local")` runs Llama Guard through a local Ollama daemon. You seed it by running `ollama pull llama-guard3` on a connected host and copying the Ollama model store to the air-gapped host, rather than copying a `from_pretrained()` folder. The detector then talks only to your local Ollama, never to a hub.

## Next steps

Pre-download the guardrail and embedding models onto approved media, copy them to your air-gapped host, point each detector and the tool selector at its local path, and run `warmup()` behind `HF_HUB_OFFLINE=1` to confirm the process makes no outbound requests before it serves a single message.

- Wire the detectors and read every model parameter in the [Security Guardrails reference](../../core/guardrails.md).
- Point semantic tool selection at a local embedding model in the [Tool Optimization guide](../../core/tool-optimization.md).
- Serve your language model locally too — see [Model Setup](../../getting-started/model-setup.md) for the Ollama path.
- Zoom out to the full offline stack in the [air-gapped agent framework guide](air-gapped-agent-framework.md) and the [why frameworks fail in air-gapped networks](air-gapped-ai-agent.md) deep dive.
