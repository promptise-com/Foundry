---
title: Model Setup — use any LLM with Promptise Foundry in two minutes (Azure AI Foundry, OpenAI, Bedrock, Gemini, Ollama)
description: Get a model working in two minutes — put the key in .env or pass it in code with Model(...), check what is missing with promptise models check, and ping it. Every provider works with the core install; the exact environment variable for each is in one table. The in-depth reference lives in Agent → Models & Providers.
keywords: Promptise model setup, Azure AI Foundry Python, Azure OpenAI deployment, use any LLM, model agnostic AI agent, promptise models check
---

# Model Setup

Every place Promptise takes a model — `build_agent(model=...)`, `.superagent`
and `.agent` files, `promptise mcpcast --model`, the CLI — takes it in one of
three forms:

| Form | Example |
|---|---|
| **`Model(...)` in code** — credentials explicit | `Model("gpt-4o", provider="azure", deployment="chat-prod", endpoint=..., api_key=..., api_version=...)` |
| **A `provider:model` string** — credentials from `.env` / environment | `"azure:chat-prod"`, `"openai:gpt-5-mini"` |
| **A `model:` block** in a `.superagent` / `.agent` file | `provider: azure` / `model: gpt-4o` / `deployment: chat-prod` |

**Nothing to install per provider** — `pip install promptise` reaches all of
them. This page gets you to a working model; the full reference — every word
of `Model`, Azure AI Foundry in depth, every provider's section, custom and
self-hosted endpoints, failover, per-node models — is
[**Agent → Models & Providers**](../core/agents/models.md).

## Two minutes to a working model

**1. Pick the provider and see what it needs.** Nothing set yet:

--8<-- "docs/.snippets/check-azure.txt"

**2. Give it the values — in `.env`, or in code.**

=== "In `.env` (loaded automatically)"

    `promptise models env <provider>` prints the lines, with a note on where
    each value is found in the provider's console:

    ```bash
    promptise models env azure >> .env    # then fill in the values
    ```

    --8<-- "docs/.snippets/env-azure.txt"

=== "In code with `Model(...)`"

    ```python
    from promptise import Model, build_agent

    agent = await build_agent(
        model=Model(
            "gpt-4o",                                         # the model
            provider="azure",
            deployment="chat-prod",                           # Foundry → Deployments → Name
            endpoint="https://my-resource.openai.azure.com/", # your resource → Overview → Endpoint
            api_key="...",                                    # your resource → Keys and Endpoint
            api_version="2024-10-21",
        ),
        servers=...,
    )
    ```

    A value given in code counts as provided — nothing needs to be in the
    environment. Every word is explained in
    [Models & Providers → In code: Model](../core/agents/models.md#in-code-model).

**3. Check again, then ping.** Without `--ping` the check covers configuration
only — the key is set, not that it works. `--ping` makes a real one-token call:

```text
$ promptise models check openai:gpt-5-mini --ping
…
Configuration OK.
Pinging… ok — replied 'ok'
```

For a local model (Ollama, or an `endpoint=` on `localhost`) the check also
connects to the server and fails with `nothing is listening at
http://localhost:11434 — is Ollama running?` when it is down.

## Every provider at a glance

--8<-- "docs/.snippets/providers-table.md"

## Where next

- [**Agent → Models & Providers**](../core/agents/models.md) — the in-depth reference: every `Model` word, [Azure AI Foundry](../core/agents/models.md#azure-ai-foundry), [every provider](../core/agents/models.md#every-provider), [custom and self-hosted endpoints](../core/agents/models.md#custom-self-hosted-and-inference-endpoints), failover, per-node models, troubleshooting
- [Configuration & Secrets](configuration.md) — where keys live and the precedence between `.env`, environment, code and config files
- [Quick Start](quickstart.md) — your first agent
