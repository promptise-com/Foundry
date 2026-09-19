| Provider | `provider=` | Environment variables (exact names) | `Model(...)` words | Route |
|---|---|---|---|---|
| OpenAI | `openai` <br><small>also `gpt`</small> | `OPENAI_API_KEY`<br>`OPENAI_BASE_URL` (optional) | `api_key=`, `endpoint=` | native (core) |
| Azure OpenAI | `azure` <br><small>also `azure_openai`, `azure-openai`, `azureopenai`, `aoai`</small> | `AZURE_OPENAI_ENDPOINT`<br>`AZURE_OPENAI_API_KEY`<br>`OPENAI_API_VERSION` | `deployment=`, `api_key=`, `endpoint=`, `api_version=` | native (core) |
| Azure AI Foundry model catalog | `foundry` <br><small>also `azure_ai`, `azure-ai`, `azureai`, `ai-foundry`</small> | `AZURE_INFERENCE_ENDPOINT`<br>`AZURE_INFERENCE_CREDENTIAL` | `api_key=`, `endpoint=`, `api_version=` | OpenAI-compatible, URL built from the words |
| Anthropic Claude | `anthropic` <br><small>also `claude`</small> | `ANTHROPIC_API_KEY` | `api_key=`, `endpoint=` | native (core) |
| Google Gemini | `gemini` <br><small>also `google_genai`, `google`, `google-genai`, `genai`</small> | `GOOGLE_API_KEY` / `GEMINI_API_KEY` | `api_key=`, `endpoint=` | OpenAI-compatible: `https://generativelanguage.googleapis.com/v1beta/openai/` |
| Google Vertex AI | `vertex` <br><small>also `google_vertexai`, `vertexai`, `google-vertex`, `gcp`</small> | `GOOGLE_CLOUD_PROJECT`<br>`GOOGLE_CLOUD_LOCATION` (optional)<br>`GOOGLE_OAUTH_ACCESS_TOKEN` (optional) | `api_key=`, `endpoint=`, `region=`, `project=` | OpenAI-compatible, URL built from the words |
| Amazon Bedrock | `bedrock` <br><small>also `aws`, `amazon`</small> | `AWS_DEFAULT_REGION` / `AWS_REGION`<br>`AWS_BEARER_TOKEN_BEDROCK` | `api_key=`, `endpoint=`, `region=` | OpenAI-compatible, URL built from the words |
| Ollama | `ollama` <br><small>also `local`</small> | `OLLAMA_HOST` (optional) | `endpoint=` | OpenAI-compatible, URL built from the words |
| Cohere | `cohere` | `COHERE_API_KEY` | `api_key=`, `endpoint=` | OpenAI-compatible: `https://api.cohere.ai/compatibility/v1` |
| DeepSeek | `deepseek` | `DEEPSEEK_API_KEY` | `api_key=`, `endpoint=` | OpenAI-compatible: `https://api.deepseek.com/v1` |
| Fireworks AI | `fireworks` | `FIREWORKS_API_KEY` | `api_key=`, `endpoint=` | OpenAI-compatible: `https://api.fireworks.ai/inference/v1` |
| Groq | `groq` | `GROQ_API_KEY` | `api_key=`, `endpoint=` | OpenAI-compatible: `https://api.groq.com/openai/v1` |
| Hugging Face Inference Providers | `huggingface` <br><small>also `hf`</small> | `HF_TOKEN` / `HUGGINGFACEHUB_API_TOKEN` | `api_key=`, `endpoint=` | OpenAI-compatible: `https://router.huggingface.co/v1` |
| Mistral AI | `mistral` <br><small>also `mistralai`</small> | `MISTRAL_API_KEY` | `api_key=`, `endpoint=` | OpenAI-compatible: `https://api.mistral.ai/v1` |
| NVIDIA NIM | `nvidia` <br><small>also `nim`</small> | `NVIDIA_API_KEY` | `api_key=`, `endpoint=` | OpenAI-compatible: `https://integrate.api.nvidia.com/v1` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` | `api_key=`, `endpoint=` | OpenAI-compatible: `https://openrouter.ai/api/v1` |
| Perplexity | `perplexity` | `PPLX_API_KEY` | `api_key=`, `endpoint=` | OpenAI-compatible: `https://api.perplexity.ai` |
| Together AI | `together` | `TOGETHER_API_KEY` | `api_key=`, `endpoint=` | OpenAI-compatible: `https://api.together.xyz/v1` |
| xAI Grok | `xai` <br><small>also `grok`</small> | `XAI_API_KEY` | `api_key=`, `endpoint=` | OpenAI-compatible: `https://api.x.ai/v1` |

Nothing to install for any row: `pip install promptise` covers every provider. `native=True` opts into a provider's own LangChain integration when you have it installed.
