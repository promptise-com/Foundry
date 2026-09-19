---
title: Install Promptise Foundry — pip install with optional extras
description: Install Promptise Foundry with pip. Python 3.10+. Optional extras for memory backends (Chroma, Mem0), sandboxing (Docker, gVisor), observability (Prometheus, OpenTelemetry), and more. The base install ships a complete framework.
keywords: install promptise, pip install promptise, Promptise Foundry install, Python AI agent install
---

# Installation

Promptise ships a complete framework in the base install. You only need an extra when you want the full production stack.

## Two choices

```bash
pip install promptise              # core — start here
pip install "promptise[all]"       # production-ready
```

That's it.

## What you get

### `pip install promptise`

The base install. Everything you need to build, run, and deploy agents that call MCP tools.

- Core agent factory (`build_agent()`)
- Reasoning engine (graph + 20+ node types)
- MCP server SDK + native MCP client
- Agent runtime (triggers, journal, governance, hooks)
- Prompt engineering (blocks, flows, strategies, guards, ContextEngine)
- CLI (`promptise ...`)
- Orchestration REST API
- Every model provider: OpenAI, Azure OpenAI and Anthropic natively; Groq, Gemini, Vertex AI, Bedrock, Mistral, Ollama, the Azure AI Foundry catalog and more through their OpenAI-compatible endpoints — no per-provider packages (see [Model Setup](model-setup.md))

### `pip install "promptise[all]"`

Adds the heavy dependencies that unlock everything optional. Recommended for production and for evaluating the full feature set.

| Category | Package | Unlocks |
|---|---|---|
| **Vector memory** | `chromadb`, `mem0ai` | `ChromaProvider`, `Mem0Provider` |
| **Embeddings** | `sentence-transformers`, `numpy` | Semantic tool optimization, `SemanticCache` |
| **ML guardrails** | `transformers` | DeBERTa prompt-injection + GLiNER NER |
| **Infrastructure** | `redis`, `docker` | `RedisConversationStore`, `RedisCache`, Docker sandbox |
| **Observability** | `opentelemetry-*`, `prometheus_client` | OTel tracing, Prometheus `/metrics` |

## Contributors

```bash
pip install "promptise[dev]"
```

Everything in `[all]` plus test runners (`pytest`, `pytest-asyncio`, `pytest-cov`), lint (`mypy`, `ruff`, `types-PyYAML`), and docs tooling (`mkdocs`, `mkdocs-material`, `mkdocstrings`).

## Air-gapped deployments

For environments without internet access:

```bash
# On a connected machine
pip download "promptise[all]" -d ./wheels

# Transfer the wheels/ directory

# On the target machine
pip install --no-index --find-links=./wheels "promptise[all]"
```

For semantic tool optimization with local embedding models, set `embedding_model` to a local directory path in `ToolOptimizationConfig`.

## Migrating from older extras

Earlier versions had `[ml]`, `[infra]`, `[observability]`, `[mcp]`, `[deep]`, and `[docs]` as separate extras. All of their dependencies are now in `[all]`.

| Old | New |
|---|---|
| `pip install "promptise[ml]"` | `pip install "promptise[all]"` |
| `pip install "promptise[infra]"` | `pip install "promptise[all]"` |
| `pip install "promptise[observability]"` | `pip install "promptise[all]"` |
| `pip install "promptise[mcp]"` | `pip install promptise` (MCP is core) |
| `pip install "promptise[docs]"` | `pip install "promptise[dev]"` |
| `pip install "promptise[deep]"` | `pip install deepagents` (installed separately) |
