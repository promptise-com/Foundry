---
title: Models & Providers — configure any LLM for a Promptise agent (Azure AI Foundry, OpenAI, Bedrock, Gemini, Ollama, self-hosted)
description: The in-depth reference for the model an agent runs on. Three ways to specify it (string, Model object, config file), every configuration word, Azure AI Foundry deployments and catalog, every provider with its exact environment variables, custom and self-hosted endpoints, failover, per-node models, and troubleshooting.
keywords: build_agent model, Azure AI Foundry agent Python, Azure OpenAI deployment LangChain, Bedrock agent, Gemini agent, Ollama agent, vLLM agent, model agnostic agent framework, FallbackChain
---

# Models & Providers

`build_agent(model=...)` is the one parameter every agent needs. This page is
the full reference for it: how to say which model, how to give it its
credentials, how each provider is reached, and what the agent does with the
model once it has one.

**Source:** `src/promptise/models.py` (registry, `Model`, `resolve_model`),
`src/promptise/models_cli.py` (`promptise models`), `src/promptise/fallback.py`

## Three ways to specify a model

| Form | When | Example |
|---|---|---|
| **`Model(...)` in code** | You have the key, endpoint, deployment in hand and want them explicit — Azure especially | `Model("gpt-4o", provider="azure", deployment="chat-prod", endpoint=..., api_key=..., api_version=...)` |
| **A `provider:model` string** | Credentials live in `.env` or the environment | `"azure:chat-prod"`, `"openai:gpt-5-mini"` |
| **A `model:` block in a config file** | Declarative agents (`.superagent`, `.agent`) | `provider: azure` / `model: gpt-4o` / `deployment: chat-prod` |

They share one vocabulary, and one resolver: aliases, `.env` loading and the
actionable errors are identical whichever form you use. Whatever you type,
`promptise models check <model>` tells you what is still missing.

**Nothing to install per provider.** OpenAI, Azure OpenAI and Anthropic use
their native integrations; every other provider is reached through its
OpenAI-compatible endpoint with `langchain-openai` — all part of
`pip install promptise`.

## Every provider at a glance

--8<-- "docs/.snippets/providers-table.md"

`promptise models list` shows this table live, with whether each variable is
set. Where the credentials should live — `.env`, environment, in code, config
file — and the precedence between them is in
[Configuration & Secrets](../../getting-started/configuration.md).

## In code: `Model`

```python
from promptise import Model, build_agent

agent = await build_agent(
    model=Model(
        "gpt-4o",                                        # what the model is
        provider="azure",                                # where it runs
        deployment="chat-prod",                          # what you named it in Azure AI Foundry
        endpoint="https://my-resource.openai.azure.com/",
        api_key="...",
        api_version="2024-10-21",
        temperature=0,
    ),
    servers=...,
)
```

`Model` has the shape of LangChain's `init_chat_model`: `model` is positional,
`provider` names the service, and the rest are keywords:

| Word | Meaning | How it is used |
|---|---|---|
| `provider` | Any name or alias below (`azure`, `foundry`, `openai`, `bedrock`, `gemini`, …). Omit it and the provider is inferred from the model name, as LangChain does (`gpt-…` → OpenAI, `claude…` → Anthropic). | picks the route |
| `deployment` | Azure OpenAI only: the name you gave the model when you deployed it | `azure_deployment=`; defaults to `model` |
| `api_key` | The provider's key — Bedrock: a Bedrock API key; Vertex AI: an OAuth access token (minted for you with `google-auth` + Application Default Credentials, and refreshed before it expires); a local Ollama needs none | the bearer token of every request |
| `endpoint` | Where to send requests: the Azure OpenAI resource, the Azure AI Foundry inference endpoint, a remote Ollama, or the `/v1` URL of any self-hosted server — for a public provider this **overrides** its default URL | `azure_endpoint=` on Azure OpenAI, `base_url=` everywhere else |
| `api_version` | Azure OpenAI REST API version; Azure AI Foundry inference `api-version` | `api_version=` / the `api-version` query parameter |
| `region` | Bedrock region / Vertex AI location | fills the endpoint URL |
| `project` | Google Cloud project | fills the Vertex AI endpoint URL |
| `temperature`, `max_tokens`, `timeout` | Generation settings | passed through |
| `native` | Use the provider's own LangChain integration instead of its OpenAI-compatible endpoint (you install it: `pip install langchain-aws`, …) | `False` by default |
| `extra` | Anything provider-specific, verbatim | `{"azure_ad_token_provider": ...}`, `{"default_headers": {...}}` |

A value you give in code counts as provided — the matching environment
variable is not required. `None` or a blank string does **not** count:
`Model(..., api_key="")` — what a `${OPENAI_API_KEY}` reference in a
`.superagent` file yields when the variable is exported but empty — leaves
the key to the environment or `.env` exactly as if it had been left out, and
the error for a missing key names the provider's own variable
(`GROQ_API_KEY`, not `OPENAI_API_KEY`). A word that does not apply to a
provider is refused with what to use instead:

```text
Groq has no 'deployment' setting: only Azure OpenAI addresses models by deployment name — put the name in model=.
```

`Model("azure:chat-prod")` — the string form inside `Model` — is accepted too
and means `Model("chat-prod", provider="azure")`.

## Azure AI Foundry

Azure AI Foundry serves two kinds of model, and they are reached differently.

### OpenAI models deployed in Foundry — `provider="azure"`

Four things identify a deployment, and they are four different values:

| You need | Where it is in the portal | Word |
|---|---|---|
| The **model** (`gpt-4o`, `gpt-4.1-mini`, …) | Foundry → Deployments → the *Model* column | `model` |
| The **deployment name** — what *you* called it | Foundry → Deployments → the *Name* column | `deployment` |
| The resource **endpoint** | Foundry → your resource → Overview → Endpoint (`https://<resource>.openai.azure.com/`, no path) | `endpoint` |
| The **API version** | Azure OpenAI docs → *API version lifecycle*; `2024-10-21` is a current GA version | `api_version` |
| A **key** — or Entra ID | Foundry → your resource → Keys and Endpoint → KEY 1 | `api_key` |

```python
from promptise import Model, build_agent

azure = Model(
    "gpt-4o",
    provider="azure",
    deployment="chat-prod",
    endpoint="https://my-resource.openai.azure.com/",
    api_key="...",
    api_version="2024-10-21",
)
agent = await build_agent(model=azure, servers=...)
```

Requests go to `/openai/deployments/chat-prod/chat/completions` on your
endpoint; `model` is used for token accounting. If you leave `deployment` out
it defaults to the model name — correct only if you named the deployment after
the model.

=== "Environment variables"

    ```bash
    export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/
    export AZURE_OPENAI_API_KEY=...
    export OPENAI_API_VERSION=2024-10-21
    ```

    Then the string form names the deployment:

    ```python
    agent = await build_agent(model="azure:chat-prod", servers=...)
    ```

=== "`.superagent` file"

    ```yaml
    agent:
      model:
        provider: azure
        model: gpt-4o
        deployment: chat-prod
        endpoint: https://my-resource.openai.azure.com/
        api_key: ${AZURE_OPENAI_API_KEY}
        api_version: "2024-10-21"
    ```

=== "`promptise mcpcast`"

    The CLI reads the environment variables above:

    ```bash
    promptise mcpcast openapi.yaml --model azure:chat-prod
    ```

=== "Keyless (Entra ID)"

    Drop `api_key` and pass a token provider from `azure-identity` (not a
    Promptise extra — `pip install azure-identity`):

    ```python
    from azure.identity import DefaultAzureCredential, get_bearer_token_provider

    token = get_bearer_token_provider(
        DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default"
    )
    azure = Model(
        "gpt-4o",
        provider="azure",
        deployment="chat-prod",
        endpoint="https://my-resource.openai.azure.com/",
        api_version="2024-10-21",
        extra={"azure_ad_token_provider": token},
    )
    ```

    The signed-in identity needs the *Cognitive Services OpenAI User* role on
    the resource.

### Other models in the Foundry catalog — `provider="foundry"`

Llama, Mistral, DeepSeek, Phi, Cohere and the rest of the catalog are served
through the Azure AI inference endpoint, which is a different endpoint and
key from the OpenAI one:

```python
llama = Model(
    "Llama-3.3-70B-Instruct",                            # the deployment name in the catalog
    provider="foundry",
    endpoint="https://my-resource.services.ai.azure.com/models",
    api_key="...",
)
```

| You need | Where it is in the portal |
|---|---|
| The model / deployment name | Foundry → Models + endpoints → the deployment's *Name* |
| The inference endpoint | The deployment's *Endpoint* — `https://<resource>.services.ai.azure.com/models` |
| The key | The deployment's *Key* |

Environment-variable form: `AZURE_INFERENCE_ENDPOINT` and
`AZURE_INFERENCE_CREDENTIAL`, then `"foundry:Llama-3.3-70B-Instruct"`.

!!! tip "OpenAI-compatible route"
    If your Foundry resource exposes an OpenAI-compatible `/v1` endpoint for a
    model, it also works through the `openai` provider:
    `Model("my-model", provider="openai", endpoint="https://<resource>.services.ai.azure.com/openai/v1", api_key="...")`.
    Check the endpoint shown on your deployment page.

## How a provider is reached

| Route | Providers | What happens |
|---|---|---|
| **native** | OpenAI, Azure OpenAI, Anthropic | The provider's own LangChain class (`ChatOpenAI`, `AzureChatOpenAI`, `ChatAnthropic`), all core dependencies |
| **OpenAI-compatible** | everything else | `ChatOpenAI` pointed at the provider's OpenAI-compatible endpoint — Groq's `/openai/v1`, Gemini's `/v1beta/openai/`, Bedrock's `/openai/v1` (with a Bedrock API key), Vertex AI's `endpoints/openapi` (with an OAuth token), the Azure AI Foundry inference endpoint, Ollama's `/v1`, … |
| **native, opt-in** | any provider, `Model(..., native=True)` | The provider's own LangChain package (`langchain-aws`, `langchain-google-genai`, …) — **you** install it; use it for what the compatible route cannot do, such as IAM/SSO credentials on Bedrock |

`promptise models check <model>` prints the route and the exact URL a string
will call.

## Every provider

Nothing to install for any of them — `pip install promptise` is the whole
setup. Each section shows the in-code form; the string form is
`provider:model` with the listed environment variables set. OpenAI, Azure
OpenAI and Anthropic use their native integrations; every other provider is
reached through its OpenAI-compatible endpoint. Add `native=True` to
`Model(...)` to use a provider's own LangChain package instead when you have
it installed (for example `langchain-aws` for IAM or SSO credentials on
Bedrock) — Promptise never installs anything for you.

### OpenAI

**Provider:** `openai`, `gpt` · **Route:** its native LangChain integration (core) · **Model:** the model name.

```python
Model("gpt-5-mini", provider="openai", api_key="sk-...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `OPENAI_API_KEY` | yes | platform.openai.com → API keys |
| `OPENAI_BASE_URL` | no | only for an OpenAI-compatible server (vLLM, LM Studio, a gateway): its /v1 URL |

Also the provider for any self-hosted or third-party server that speaks the OpenAI chat API: set endpoint= (or OPENAI_BASE_URL) to its /v1 URL.

Provider docs: <https://platform.openai.com/docs/models>

### Azure OpenAI (OpenAI models deployed in Azure AI Foundry)

**Provider:** `azure_openai`, `azure`, `azure-openai`, `azureopenai`, `aoai` · **Route:** its native LangChain integration (core) · **Model:** in the string form, your DEPLOYMENT name (Foundry → Deployments → Name); with Model(...), the model name (gpt-4o) — the deployment goes in deployment=.

```python
Model("gpt-4o", provider="azure", deployment="chat-prod",
      endpoint="https://my-resource.openai.azure.com/", api_key="...", api_version="2024-10-21")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `AZURE_OPENAI_ENDPOINT` | yes | Azure AI Foundry portal → your resource → Overview → Endpoint (https://<resource>.openai.azure.com/, no path) |
| `AZURE_OPENAI_API_KEY` | yes | Azure AI Foundry portal → your resource → Keys and Endpoint → KEY 1 (or Entra ID: pass extra={'azure_ad_token_provider': ...}) |
| `OPENAI_API_VERSION` | yes | the REST API version your deployment supports, e.g. 2024-10-21 (Azure docs → 'API version lifecycle') |

Azure routes requests by deployment name, not model name: Model('gpt-4o', provider='azure', deployment='chat-prod', ...).

Provider docs: <https://learn.microsoft.com/azure/ai-services/openai/>

### Azure AI Foundry model catalog (Llama, Mistral, DeepSeek, Phi, Cohere, …)

**Provider:** `azure_ai`, `foundry`, `azure-ai`, `azureai`, `ai-foundry` · **Route:** its OpenAI-compatible endpoint, built from the words · **Model:** the deployment name shown in Models + endpoints.

```python
Model("Llama-3.3-70B-Instruct", provider="foundry",
      endpoint="https://my-resource.services.ai.azure.com/models", api_key="...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `AZURE_INFERENCE_ENDPOINT` | yes | Azure AI Foundry portal → Models + endpoints → your deployment → Endpoint (https://<resource>.services.ai.azure.com/models) |
| `AZURE_INFERENCE_CREDENTIAL` | yes | Azure AI Foundry portal → Models + endpoints → your deployment → Key |

For catalog models served by Azure AI Foundry's inference endpoint. OpenAI models deployed in Foundry use provider='azure' instead. The Azure AI Model Inference API is OpenAI-compatible; requests carry api-version=2024-05-01-preview unless api_version= says otherwise.

Provider docs: <https://learn.microsoft.com/azure/ai-foundry/>

### Anthropic Claude

**Provider:** `anthropic`, `claude` · **Route:** its native LangChain integration (core) · **Model:** the model name.

```python
Model("claude-sonnet-4-5", provider="anthropic", api_key="sk-ant-...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | console.anthropic.com → API keys |

Provider docs: <https://docs.anthropic.com/en/docs/about-claude/models>

### Google Gemini (AI Studio API key)

**Provider:** `google_genai`, `gemini`, `google`, `google-genai`, `genai` · **Route:** its OpenAI-compatible endpoint `https://generativelanguage.googleapis.com/v1beta/openai/` · **Model:** the model name.

```python
Model("gemini-2.5-pro", provider="gemini", api_key="AIza...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `GOOGLE_API_KEY` / `GEMINI_API_KEY` | yes | aistudio.google.com → Get API key |

For Gemini through Google Cloud (no API key, ADC) use provider='vertex'.

Provider docs: <https://ai.google.dev/gemini-api/docs/openai>

### Google Vertex AI (Gemini and Model Garden via Google Cloud)

**Provider:** `google_vertexai`, `vertex`, `vertexai`, `google-vertex`, `gcp` · **Route:** its OpenAI-compatible endpoint, built from the words · **Model:** the model name (a bare Gemini name is prefixed google/ for you).

```python
Model("gemini-2.5-pro", provider="vertex", project="my-project-123", region="us-central1",
      api_key="<gcloud auth print-access-token>")   # or leave api_key out with google-auth + ADC
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `GOOGLE_CLOUD_PROJECT` | yes | your GCP project id (console.cloud.google.com, top bar) |
| `GOOGLE_CLOUD_LOCATION` | no | the region of your Vertex AI endpoint |
| `GOOGLE_OAUTH_ACCESS_TOKEN` | no | an OAuth access token: `gcloud auth print-access-token` (valid ~1 h, never refreshed — for one-off runs); not needed when google-auth is installed and Application Default Credentials are configured — the token is minted for you and refreshed before it expires |

Authenticates with a Google OAuth access token. With google-auth installed (`pip install google-auth`) and `gcloud auth application-default login` done, the token is minted automatically and refreshed before it expires, so a long-running agent keeps working; a token set by hand (GOOGLE_OAUTH_ACCESS_TOKEN or api_key=) is used as given and expires after about an hour.

!!! note "Token lifetime"
    Google access tokens live about an hour. On the auto-minted path Promptise
    hands the OpenAI-compatible client a token *provider* rather than a
    string: the credentials are refreshed once up front (so a missing or
    revoked ADC fails immediately with a `ModelSetupError` naming the cause)
    and again, before the next request, whenever google-auth reports the
    token expired — an `AgentRuntime` process, `promptise serve` or a
    cross-agent server running for days never sees a bare 401 from Google.
    A token you set by hand is used as given and stops working after ~1 h;
    that path is for one-off runs. A refresh that fails later (revoked
    credentials, no network to the token endpoint) raises `ModelSetupError`
    from the request that needed it, with the `gcloud` command that fixes it.
    Automatic refresh needs `langchain-openai` 1.0.1 or later, whose
    `ChatOpenAI` accepts a callable `api_key`; with an older release the
    minted token is pinned for the life of the model and a warning says so.

Provider docs: <https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/call-vertex-using-openai-library>

### Amazon Bedrock

**Provider:** `bedrock`, `aws`, `amazon` · **Route:** its OpenAI-compatible endpoint, built from the words · **Model:** the Bedrock model id or an inference-profile id/ARN.

```python
Model("anthropic.claude-sonnet-4-20250514-v1:0", provider="bedrock", region="us-east-1", api_key="ABSK...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `AWS_DEFAULT_REGION` / `AWS_REGION` | yes | the region your Bedrock models are enabled in |
| `AWS_BEARER_TOKEN_BEDROCK` | yes | AWS console → Amazon Bedrock → API keys → Generate (a Bedrock API key; long-term keys start with ABSK) |

Uses Bedrock's OpenAI-compatible endpoint with a Bedrock API key — no AWS SDK needed. For IAM/SSO credentials install langchain-aws and pass native=True.

Provider docs: <https://docs.aws.amazon.com/bedrock/latest/userguide/inference-chat-completions.html>

### Ollama (local models)

**Provider:** `ollama`, `local` · **Route:** its OpenAI-compatible endpoint, built from the words · **Model:** a model you have pulled (`ollama pull llama3.1`).

```python
Model("llama3.1", provider="ollama")                      # endpoint="http://gpu-box:11434" for a remote one
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `OLLAMA_HOST` | no | only if Ollama is not on the default http://localhost:11434 |

No API key. The model must support tool calling to drive MCP tools.

Provider docs: <https://ollama.com/library>

### Cohere

**Provider:** `cohere` · **Route:** its OpenAI-compatible endpoint `https://api.cohere.ai/compatibility/v1` · **Model:** the model name.

```python
Model("command-a-03-2025", provider="cohere", api_key="...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `COHERE_API_KEY` | yes | dashboard.cohere.com → API keys |

Provider docs: <https://docs.cohere.com/docs/compatibility-api>

### DeepSeek

**Provider:** `deepseek` · **Route:** its OpenAI-compatible endpoint `https://api.deepseek.com/v1` · **Model:** the model name.

```python
Model("deepseek-chat", provider="deepseek", api_key="...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `DEEPSEEK_API_KEY` | yes | platform.deepseek.com → API keys |

Provider docs: <https://api-docs.deepseek.com/>

### Fireworks AI

**Provider:** `fireworks` · **Route:** its OpenAI-compatible endpoint `https://api.fireworks.ai/inference/v1` · **Model:** the full model path.

```python
Model("accounts/fireworks/models/llama-v3p3-70b-instruct", provider="fireworks", api_key="...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `FIREWORKS_API_KEY` | yes | fireworks.ai → API Keys |

Provider docs: <https://fireworks.ai/models>

### Groq

**Provider:** `groq` · **Route:** its OpenAI-compatible endpoint `https://api.groq.com/openai/v1` · **Model:** the model name.

```python
Model("llama-3.3-70b-versatile", provider="groq", api_key="...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `GROQ_API_KEY` | yes | console.groq.com → API Keys |

Provider docs: <https://console.groq.com/docs/models>

### Hugging Face Inference Providers

**Provider:** `huggingface`, `hf` · **Route:** its OpenAI-compatible endpoint `https://router.huggingface.co/v1` · **Model:** the repository id (add :provider to pin an inference provider).

```python
Model("meta-llama/Llama-3.3-70B-Instruct", provider="hf", api_key="hf_...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` | yes | huggingface.co → Settings → Access Tokens |

A dedicated Inference Endpoint: set endpoint= to its URL.

Provider docs: <https://huggingface.co/docs/inference-providers>

### Mistral AI

**Provider:** `mistralai`, `mistral` · **Route:** its OpenAI-compatible endpoint `https://api.mistral.ai/v1` · **Model:** the model name.

```python
Model("mistral-large-latest", provider="mistral", api_key="...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `MISTRAL_API_KEY` | yes | console.mistral.ai → API Keys |

Provider docs: <https://docs.mistral.ai/getting-started/models/>

### NVIDIA NIM

**Provider:** `nvidia`, `nim` · **Route:** its OpenAI-compatible endpoint `https://integrate.api.nvidia.com/v1` · **Model:** the model path.

```python
Model("meta/llama-3.3-70b-instruct", provider="nvidia", api_key="nvapi-...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `NVIDIA_API_KEY` | yes | build.nvidia.com → API key |

A NIM you run yourself: set endpoint= to its /v1 URL.

Provider docs: <https://build.nvidia.com/models>

### OpenRouter

**Provider:** `openrouter` · **Route:** its OpenAI-compatible endpoint `https://openrouter.ai/api/v1` · **Model:** the provider/model path as listed by OpenRouter.

```python
Model("anthropic/claude-sonnet-4.5", provider="openrouter", api_key="...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `OPENROUTER_API_KEY` | yes | openrouter.ai → Keys |

Provider docs: <https://openrouter.ai/models>

### Perplexity

**Provider:** `perplexity` · **Route:** its OpenAI-compatible endpoint `https://api.perplexity.ai` · **Model:** the model name.

```python
Model("sonar-pro", provider="perplexity", api_key="...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `PPLX_API_KEY` | yes | perplexity.ai → Settings → API |

Provider docs: <https://docs.perplexity.ai/guides/model-cards>

### Together AI

**Provider:** `together` · **Route:** its OpenAI-compatible endpoint `https://api.together.xyz/v1` · **Model:** the model path as listed by Together.

```python
Model("meta-llama/Llama-3.3-70B-Instruct-Turbo", provider="together", api_key="...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `TOGETHER_API_KEY` | yes | api.together.ai → Settings → API Keys |

Provider docs: <https://docs.together.ai/docs/serverless-models>

### xAI Grok

**Provider:** `xai`, `grok` · **Route:** its OpenAI-compatible endpoint `https://api.x.ai/v1` · **Model:** the model name.

```python
Model("grok-4", provider="xai", api_key="...")
```

| Environment variable | Required | Where to find it |
|---|---|---|
| `XAI_API_KEY` | yes | console.x.ai → API Keys |

Provider docs: <https://docs.x.ai/docs/models>

## Custom, self-hosted and inference endpoints

A model you host yourself, a fine-tune, or a managed inference endpoint is
configured with the same words — the only question is which provider speaks
its protocol:

| What you have | `provider=` | What to set | Example |
|---|---|---|---|
| **Any OpenAI-compatible server** — vLLM, LM Studio, llama.cpp server, TGI (`--openai`), SGLang, Ollama's `/v1`, a LiteLLM or corporate gateway | `openai` | `endpoint=` the server's `/v1` URL, `api_key=` whatever it expects (often anything) | `Model("qwen2.5-72b", provider="openai", endpoint="http://gpu-box:8000/v1", api_key="none")` |
| **Azure AI Foundry serverless / managed endpoint** (catalog models, your own fine-tune deployed there) | `foundry` | `endpoint=` the deployment's `…/models` URL, `api_key=` its key | `Model("my-finetune", provider="foundry", endpoint="https://r.services.ai.azure.com/models", api_key="...")` |
| **Azure OpenAI fine-tuned deployment** | `azure` | `deployment=` the fine-tune's deployment name; `model=` the base model | `Model("gpt-4o-mini", provider="azure", deployment="support-ft-v3", endpoint=..., api_key=..., api_version="2024-10-21")` |
| **Amazon Bedrock custom / imported model or inference profile** | `bedrock` | `model=` the model ARN or inference-profile ARN, `region=` | `Model("arn:aws:bedrock:us-east-1:123:inference-profile/…", provider="bedrock", region="us-east-1")` |
| **Vertex AI endpoint** (tuned Gemini) | `vertex` | `model=` the tuned model resource, `project=`, `region=`, and an access token (`api_key=` or ADC) | `Model("projects/p/locations/us-central1/endpoints/123", provider="vertex", project="p", region="us-central1")` |
| **Hugging Face Inference Endpoint** (dedicated) | `hf` | `endpoint=` the endpoint URL, `api_key=` your token | `Model("my-org/my-model", provider="hf", endpoint="https://xyz.endpoints.huggingface.cloud", api_key="hf_...")` |
| **NVIDIA NIM you run yourself** | `nvidia` | `endpoint=` the NIM's `/v1` URL, `api_key=` if it enforces one | `Model("meta/llama-3.3-70b-instruct", provider="nvidia", endpoint="http://nim:8000/v1", api_key="none")` |
| **Ollama on another machine** | `ollama` | `endpoint=` its URL | `Model("llama3.1", provider="ollama", endpoint="http://gpu-box:11434")` |
| **A LiteLLM proxy** (100+ providers behind one OpenAI-compatible URL) | `openai` | `endpoint=` the proxy's `/v1` URL, `api_key=` its virtual key, `model=` the proxy's model name | `Model("azure/my-deployment", provider="openai", endpoint="http://litellm:4000/v1", api_key="sk-litellm")` |
| **Bedrock with IAM / SSO instead of an API key**, Gemini-only features, or any provider's own client | the same provider, `native=True` | Install that provider's LangChain package yourself (`pip install langchain-aws`, `langchain-google-genai`, …) | `Model("anthropic.claude-sonnet-4-20250514-v1:0", provider="bedrock", region="us-east-1", native=True)` |
| **Anything else** — a provider LangChain integrates that is not listed, or a client with special auth | pass the instance | Build the LangChain chat model yourself and hand it to `build_agent(model=...)` | see [Passing a model instance](#passing-a-model-instance) |

Two things every custom endpoint must satisfy to drive MCP tools: it must
implement **tool calling** (function calling) for the model you pick, and it
must be reachable from where the agent runs. `promptise models check
<model> --ping` proves the first hop; a model that answers the ping but
ignores tools needs a tool-capable model or a server flag (vLLM:
`--enable-auto-tool-choice --tool-call-parser …`).

String form for the OpenAI-compatible case: `OPENAI_BASE_URL=http://gpu-box:8000/v1`
plus `OPENAI_API_KEY`, then `"openai:qwen2.5-72b"`.

## In config files

`.superagent` files take the same words under `model:`; `name:` and `model:`
are synonyms, `${ENV_VAR}` is resolved, and a value in the file counts as
provided:

```yaml
agent:
  model:
    provider: bedrock
    model: anthropic.claude-sonnet-4-20250514-v1:0
    region: us-east-1
    temperature: 0.2
    extra:
      aws_profile: prod
```

`.agent` runtime manifests take the string form (`model: azure:chat-prod`)
and read the environment.

## Passing a model instance

For anything the words above do not cover, build the LangChain chat model
yourself and pass it — every provider's class is accepted as-is:

```python
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(azure_deployment="chat-prod", model="gpt-4o", api_version="2024-10-21",
                      azure_endpoint="https://my-resource.openai.azure.com/", api_key="...")
agent = await build_agent(model=llm, servers=...)
```

Any `Runnable` works too, so a `FallbackChain` or a model wrapped with
retries is a valid `model=`.

## The model inside the agent

Everything above is about *reaching* a model. This is what the agent does
with it.

**Tool calling is required.** The agent drives MCP tools through the model's
function-calling API (`bind_tools`), so the model must support it. Every
hosted model listed on this page does; among local models pick one that
advertises tool use (`llama3.1`, `qwen2.5`, `mistral-nemo`, …). A model that
answers `promptise models check --ping` but ignores tools will loop or reply
in prose — `agent.ainvoke` still returns, with no tool calls in the trace.

**One model per agent, overridable per node.** `build_agent(model=...)` is
the agent's model. In a [custom reasoning graph](reasoning-patterns.md) any
node can carry its own with `model_override` — a string, a `Model`, or an
instance — for a cheap model on routing nodes and a strong one on the answer
node:

```python
from promptise import Model
from promptise.engine import PromptNode

PromptNode("draft", instructions="…", model_override="groq:llama-3.3-70b-versatile")
PromptNode("final", instructions="…", model_override=Model("gpt-4o", provider="azure",
           deployment="chat-prod", endpoint=..., api_key=..., api_version="2024-10-21"))
```

**Failover across providers.** [`FallbackChain`](../fallback.md) takes the
same strings and `Model` objects and tries them in order, with a per-model
timeout and a circuit breaker:

```python
from promptise import FallbackChain, Model, build_agent

agent = await build_agent(
    model=FallbackChain(
        [
            Model("gpt-4o", provider="azure", deployment="chat-prod", endpoint=..., api_key=..., api_version="2024-10-21"),
            "anthropic:claude-sonnet-4-5",
            "groq:llama-3.3-70b-versatile",
        ],
        timeout_per_model=20,
    ),
    servers=...,
)
```

**Where the string travels.** The same value is accepted by `.superagent`
files (`agent.model`), `.agent` runtime manifests (`model:`), cross-agent
peers (each peer is its own `build_agent` with its own model),
`promptise mcpcast --model`, and the adaptive strategy's `synthesis_model`.
All of them resolve through the registry, so an alias, a `.env` file or a
missing-key error behaves identically everywhere.

**Settings that matter for agents.** `temperature=0` for deterministic
tool selection; `timeout` (seconds) bounds one model call — pair it with
`build_agent(max_invocation_time=...)` to bound the whole run;
`max_tokens` caps the answer, not the tool calls. Streaming and structured
output (`with_structured_output`) go through the same model, so a provider
that supports them via its OpenAI-compatible endpoint supports them here.

**Observability.** With `observe=True` every model turn is recorded with
its token counts and latency under the model name (`Model.spec` or the
string you passed) — see [Observability](../observability.md).

**Testing without a provider.** Any `Runnable` is a valid `model=`, so tests
pass a fake: a LangChain `FakeListChatModel`, or a `MagicMock` with
`ainvoke`/`bind_tools` as the framework's own tests do
(`tests/test_engine_execution.py`). `promptise models check <model>` (no
`--ping`) is the CI preflight — it verifies configuration without a call.

## Troubleshooting

| You see | Cause | Fix |
|---|---|---|
| `Cannot use model … yet: … is not set — …` | A required environment variable is missing | Set it (the message says where the value is), or pass it in code with `Model(...)` |
| `native=True needs the langchain_… package → pip install langchain-…` | You opted into a provider's own integration without having it | Install it, or drop `native=True` — the default route needs nothing |
| `unknown provider 'x'` (from `Model(provider="x")` or `promptise models env x`), or `Unable to infer model provider for model='x:…'` followed by `Promptise understands these prefixes …` (from a string) | A prefix Promptise does not know | `promptise models list` for the prefixes and aliases |
| `Unable to infer model provider for model='…'` | A bare model name LangChain cannot classify | Add `provider=` / the prefix |
| `… has no 'deployment' setting` | `deployment` given for a provider other than Azure OpenAI | Put the name in `model` |
| `DeploymentNotFound` / HTTP 404 from Azure | `deployment` is not the name in Foundry → Deployments, or the endpoint is another resource | Copy the *Name* column, not the *Model* column |
| HTTP 401 / `invalid_api_key` on `--ping` | Wrong key or a key from another resource | Regenerate in the portal; check `endpoint` and key belong together |
| `tool calling is not supported` / tools ignored | A small or old model (many local ones) | Pick a tool-capable model; `--ping` succeeding does not prove tool support |

## See also

- [Configuration & Secrets](../../getting-started/configuration.md) — where keys live, `.env`, precedence
- [Model Setup](../../getting-started/model-setup.md) — the two-minute on-ramp
- [Building Agents](building-agents.md) — everything else `build_agent()` takes
- [SuperAgent Files](superagent-files.md) — the declarative form
- [Model Fallback](../fallback.md) — `FallbackChain` in depth
- [Custom Reasoning Patterns](reasoning-patterns.md) — per-node models
- [CLI reference](../cli.md#promptise-models-model-providers) — `promptise models`
- [API reference: models](../../api/models.md) — `Model`, `resolve_model`, `check_model`, `PROVIDERS`
