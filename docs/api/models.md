# Models API Reference

Bring your own model: `promptise.models` is the one place that turns a model
description into a LangChain chat model. Every entry point that accepts a
model -- `build_agent(model=...)`, `.superagent` and `.agent` files,
`promptise mcpcast --model`, the CLI's `--model-id` -- goes through
`resolve_model()`, so the friendly provider names (`azure`, `foundry`, `gemini`,
`vertex`, `bedrock`, `mistral`, `grok`, `hf`, ...) and the same actionable
errors apply everywhere. The `promptise models` CLI and the provider table in
the docs are generated from the same registry, so the code, the CLI and the
pages cannot disagree.

**Nothing to install per provider.** OpenAI, Azure OpenAI and Anthropic use
their native integrations, which are part of the core install; every other
provider is reached through its OpenAI-compatible endpoint with
`langchain-openai`, also core. `Model(..., native=True)` opts into a provider's
own LangChain package when you have installed it yourself.

There are three ways to say which model and how to reach it, and they share
one vocabulary:

1. **A string** -- `"provider:model"`; credentials come from the environment or a `.env` file: `build_agent(model="azure:chat-prod")`.
2. **A [`Model`](#model)** -- provider, model and credentials in code, in the shape of LangChain's `init_chat_model`: `Model("gpt-4o", provider="azure", deployment="chat-prod", endpoint=..., api_key=..., api_version=...)`. The words are the same for every provider.
3. **A config file** -- the same fields under `model:` in a [`.superagent` file](../core/agents/superagent-files.md#agent).

Any LangChain `BaseChatModel` instance is accepted too, for full control.

```python
import asyncio
from promptise import Model, build_agent, check_model, resolve_model, ModelSetupError

# Diagnose without calling anything
result = check_model("azure:chat-prod")
if not result.ok:
    print("\n".join(result.problems))   # each problem names its fix

# Or let resolution raise a ModelSetupError with the same text
try:
    llm = resolve_model("groq:llama-3.3-70b-versatile")
except ModelSetupError as exc:
    print(exc)   # GROQ_API_KEY is not set — console.groq.com → API Keys ... put it in .env, export it, or pass api_key=

async def main():
    # A string: build_agent() calls resolve_model() for you; .env or env vars supply the credentials
    agent = await build_agent(model="azure:chat-prod", instructions="Be concise.")
    await agent.shutdown()

    # A Model: provider, model and credentials in code
    agent = await build_agent(
        model=Model(
            "gpt-4o",
            provider="azure",
            deployment="chat-prod",
            endpoint="https://my-resource.openai.azure.com/",
            api_key="...",
            api_version="2024-10-21",
            temperature=0,
        ),
        instructions="Be concise.",
    )
    await agent.shutdown()

asyncio.run(main())
```

For the walkthrough -- Azure AI Foundry in depth, every provider, custom and
self-hosted endpoints -- see [Model Setup](../getting-started/model-setup.md);
for where keys live, [Configuration & Secrets](../getting-started/configuration.md).

---

## Resolution

### resolve_model

::: promptise.models.resolve_model
    options:
      show_source: false
      heading_level: 4

### Model

::: promptise.models.Model
    options:
      show_source: false
      heading_level: 4

How the words reach a provider: for the three native integrations they map to
that class's own arguments (`endpoint` → `azure_endpoint=` on Azure OpenAI,
`base_url=` on OpenAI and Anthropic; `deployment` → `azure_deployment=`). For
every other provider the words build the OpenAI-compatible request:
`endpoint` overrides the provider's default URL (or *is* the URL for Azure AI
Foundry and Ollama), `region` and `project` fill the URL template for Bedrock
and Vertex AI, `api_version` sets Azure AI Foundry's `api-version` query
parameter, and `api_key` becomes the bearer token (Bedrock: a Bedrock API
key; Vertex AI: an OAuth access token — minted for you when `google-auth` and
Application Default Credentials are available, as a token *provider* the
client calls before every request, so the ~1 h token is refreshed before it
expires; a token given by hand is used as given and expires). A word a
provider has no setting for raises `ModelSetupError` with what to use
instead; a word that is `None` or a blank string counts as not given (an
empty `api_key=""` never shadows the environment check). `extra` is merged
in last, verbatim -- a keyless Azure OpenAI setup is
`Model("gpt-4o", provider="azure", deployment=..., endpoint=..., api_version=..., extra={"azure_ad_token_provider": provider})`.

```text
>>> Model("m", provider="groq", deployment="d").resolve()
ModelSetupError: Groq has no 'deployment' setting: only Azure OpenAI addresses models by deployment name — put the name in model=.
```

### check_model

::: promptise.models.check_model
    options:
      show_source: false
      heading_level: 4

### ModelCheck

::: promptise.models.ModelCheck
    options:
      show_source: false
      heading_level: 4

### ModelSetupError

::: promptise.models.ModelSetupError
    options:
      show_source: false
      heading_level: 4

The message always says what to do. Missing credentials, nothing set:

```text
Cannot use model azure:'chat-prod' (Azure OpenAI (OpenAI models deployed in Azure AI Foundry)) yet:
  - AZURE_OPENAI_ENDPOINT is not set — Azure AI Foundry portal → your resource → Overview → Endpoint (https://<resource>.openai.azure.com/, no path) (e.g. https://my-resource.openai.azure.com/)
  - AZURE_OPENAI_API_KEY is not set — Azure AI Foundry portal → your resource → Keys and Endpoint → KEY 1 (or Entra ID: pass extra={'azure_ad_token_provider': ...})
  - OPENAI_API_VERSION is not set — the REST API version your deployment supports, e.g. 2024-10-21 (Azure docs → 'API version lifecycle') (e.g. 2024-10-21)
  - put AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, OPENAI_API_VERSION in a .env file next to your script (loaded automatically, never overrides a set variable), export it, or pass it in code: Model(..., endpoint=, api_key=, api_version=) — see promptise models env azure
  Model part means: in the string form, your DEPLOYMENT name (Foundry → Deployments → Name); with Model(...), the model name (gpt-4o) — the deployment goes in deployment=.
  Example: azure:chat-prod
  Diagnose any model string with: promptise models check azure:chat-prod
```

### load_dotenv_if_present

::: promptise.models.load_dotenv_if_present
    options:
      show_source: false
      heading_level: 4

The search is bounded and the file is checked before it is read: the walk
from the working directory stops after the project root (the first ancestor
holding `pyproject.toml` or `.git`), and a candidate that is not a regular
file or -- on POSIX -- is owned by another user or world-writable is skipped
with a `UserWarning` that names it. A file that cannot be read or is not UTF-8
raises `ModelSetupError` naming the path (`cannot read /path/.env: ... — fix
its permissions/encoding, move it, or set PROMPTISE_NO_DOTENV=1`) — the same
error `resolve_model()`, `build_agent()` and the CLI surface, never a raw
`PermissionError`. The `promptise` CLI loads the file once, in its global
callback, and prints `.env loaded from <path>` on stderr.

### dotenv_origin

::: promptise.models.dotenv_origin
    options:
      show_source: false
      heading_level: 4

```python
from promptise.models import dotenv_origin, load_dotenv_if_present

load_dotenv_if_present()
dotenv_origin("OPENAI_API_KEY")   # '/home/me/project/.env' — or None when the shell set it
```

`promptise models check` uses it to print `set (from /home/me/project/.env)`
next to every variable the file filled.

---

## Parsing

### parse_model

::: promptise.models.parse_model
    options:
      show_source: false
      heading_level: 4

### find_provider

::: promptise.models.find_provider
    options:
      show_source: false
      heading_level: 4

### env_template

::: promptise.models.env_template
    options:
      show_source: false
      heading_level: 4

```text
$ promptise models env azure
# Azure OpenAI (OpenAI models deployed in Azure AI Foundry) — model string example: azure:chat-prod
export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/
#   ↳ Azure AI Foundry portal → your resource → Overview → Endpoint (https://<resource>.openai.azure.com/, no path)
export AZURE_OPENAI_API_KEY=...
#   ↳ Azure AI Foundry portal → your resource → Keys and Endpoint → KEY 1 (or Entra ID: pass extra={'azure_ad_token_provider': ...})
export OPENAI_API_VERSION=2024-10-21
#   ↳ the REST API version your deployment supports, e.g. 2024-10-21 (Azure docs → 'API version lifecycle')
```

---

## Registry

### Provider

::: promptise.models.Provider
    options:
      show_source: false
      heading_level: 4

### EnvVar

::: promptise.models.EnvVar
    options:
      show_source: false
      heading_level: 4

### PROVIDERS

`PROVIDERS` is the tuple of every `Provider` Promptise knows, in the order the
CLI lists them. The table below is generated from it (and a test asserts it
stays exact):

--8<-- "docs/.snippets/providers-table.md"

---

## CLI module

### promptise.models_cli

The `promptise models` command group (`list`, `check [--ping]`, `env`) lives in
`promptise.models_cli` and reads only `PROVIDERS`, `check_model()`,
`resolve_model()` and `env_template()` -- see the
[CLI reference](../core/cli.md#promptise-models-model-providers).
