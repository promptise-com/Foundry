---
title: Configuration & Secrets — where API keys live in Promptise Foundry
description: Where to put your LLM API keys and provider settings when using Promptise Foundry — a .env file (loaded automatically), environment variables, in code with Model(...), or a .superagent file with ${VAR} references. Precedence, what to commit, and how to check what is missing.
keywords: Promptise API key, where to put OpenAI API key Python, .env file AI agent, Azure OpenAI key configuration, secrets management AI agent, environment variables LLM
---

# Configuration & Secrets

Every model provider needs a credential, and some need an endpoint or a
region as well. There are four places you can keep them. Pick one; they all
end up in the same place.

| Where | Best for | Picked up by |
|---|---|---|
| **A `.env` file** in your project | Local development — one file, git-ignored | Everything: `python my_agent.py`, the `promptise` CLI, `.superagent` files |
| **Environment variables** | CI, containers, servers — whatever your platform injects | Everything |
| **In code, with `Model(...)`** | When the key comes from *your* secret store (Vault, Key Vault, a settings object) | `build_agent(model=Model(...))` |
| **A `.superagent` file** | Declarative agents; `${VAR}` references keep secrets out of the file | `promptise agent`, `load_superagent_file()` |

## Every provider's variables

The exact variable names each provider reads (put them in `.env` or export
them), the `provider=` value for `Model(...)` or the string prefix, and the
words that carry the same values in code. This table is generated from the
code, so the names are exact.

--8<-- "docs/.snippets/providers-table.md"

`promptise models env <provider>` prints the same variables as ready-to-fill
lines with a note on where each value is found in the provider's console.

## The `.env` file

Create a file named `.env` in your project directory:

```bash title=".env"
OPENAI_API_KEY=sk-...
```

That is all. Promptise loads it before it checks a provider's variables —
from the directory you run in or a parent, up to your project's root — so a
plain script, the CLI and config files all see the same keys, with no
`load_dotenv()` call in your code:

```python
import asyncio
from promptise import build_agent

async def main():
    agent = await build_agent(model="openai:gpt-5-mini", instructions="Be concise.")
    ...

asyncio.run(main())   # OPENAI_API_KEY comes from .env
```

Rules of the file:

- A variable that is **already set to a value in the environment wins** over
  the file — the file never overrides your shell or your platform. One that is
  exported but empty (`export OPENAI_API_KEY=`) counts as not set and takes the
  file's value; `promptise models check` says which happened (`MISSING
  (OPENAI_API_KEY is exported but empty — unset it or give it a value)` when
  the file has nothing for it either). An empty value *in the file* sets
  nothing.
- **Where it is looked for.** The directory you run in, then its parents,
  stopping after the project root — the first directory holding
  `pyproject.toml` or `.git`. A `.env` above your project (`/tmp/.env` on a
  shared host, another checkout, somebody's home directory) is never read.
  The first file found wins; nothing further up is merged in.
- **Which files are trusted.** Only a regular file and, on POSIX, one that
  you own and that nobody else can write (Windows has neither owners nor
  mode bits in this sense, so only the regular-file check applies there).
  A `.env` that is world-writable, owned by another user or not a regular
  file is skipped with a warning naming it
  (`ignoring /srv/.env: world-writable (mode 666) — run: chmod o-w
  /srv/.env`) and the search continues upward — a file a co-located user
  could plant or edit cannot redirect your endpoint or supply a key
  silently.
- **Where a value came from.** The CLI prints `.env loaded from <path>` on
  stderr, `promptise models check` shows `set (from <path>)` next to every
  variable the file filled, and `promptise.models.dotenv_origin("OPENAI_API_KEY")`
  returns the path in code (`None` when the shell set it).
- **A file that cannot be read** — a root-owned `0600` file up the tree, one
  saved as Latin-1 or cp1252 — is one clean error, not a traceback:
  `cannot read /app/.env: [Errno 13] Permission denied — fix its
  permissions/encoding, move it, or set PROMPTISE_NO_DOTENV=1`. It is a
  `ModelSetupError` from `build_agent()` / `resolve_model()` and an
  `Error:` line with exit code 2 from every `promptise` command;
  `promptise --version` and `promptise --help` never touch the file.
- It is loaded **once per process** (call `promptise.models.load_dotenv_if_present()`
  after an `os.chdir` if you need another one).
- Set `PROMPTISE_NO_DOTENV=1` to turn the loading off entirely (hermetic
  deployments, tests).
- **Never commit it.** Add `.env` to `.gitignore`; commit a `.env.example`
  with the variable names and no values instead.

`promptise models env <provider>` prints exactly the lines a provider needs,
with a note on where each value is found — paste them in:

```bash
promptise models env azure >> .env      # then fill in the values
```

```text
# Azure OpenAI (OpenAI models deployed in Azure AI Foundry) — model string example: azure:chat-prod
export AZURE_OPENAI_ENDPOINT=https://my-resource.openai.azure.com/
#   ↳ Azure AI Foundry portal → your resource → Overview → Endpoint (https://<resource>.openai.azure.com/, no path)
export AZURE_OPENAI_API_KEY=...
#   ↳ Azure AI Foundry portal → your resource → Keys and Endpoint → KEY 1 (or Entra ID: pass extra={'azure_ad_token_provider': ...})
export OPENAI_API_VERSION=2024-10-21
#   ↳ the REST API version your deployment supports, e.g. 2024-10-21 (Azure docs → 'API version lifecycle')
```

(`export` lines are valid `.env` syntax; python-dotenv accepts them.)

## Environment variables

Export them in your shell, or let your platform inject them (Docker `-e`,
Kubernetes secrets, GitHub Actions secrets, systemd `Environment=`):

```bash
export OPENAI_API_KEY=sk-...
```

The variable each provider reads is listed in [Model Setup](model-setup.md)
and printed by `promptise models env <provider>`.

## In code — `Model(...)`

When the key lives in your own secret store, hand it over in code. The same
words work for every provider; nothing needs to be in the environment:

```python
from promptise import Model, build_agent

secrets = my_vault.get("llm")   # wherever your secrets come from

agent = await build_agent(
    model=Model(
        "gpt-4o",
        provider="azure",
        deployment="chat-prod",
        endpoint=secrets["endpoint"],
        api_key=secrets["key"],
        api_version="2024-10-21",
    ),
    servers=...,
)
```

A value given in code counts as provided — the matching variable is not
looked for. `None` or a blank string does not count as given:
`Model(..., api_key="")` leaves the key to the environment and `.env`, so a
`${OPENAI_API_KEY}` reference that resolves to an empty export never shadows
the value in your file. See [Model Setup](../core/agents/models.md#in-code-model)
for every word and what it maps to per provider.

## In a `.superagent` file

Keep the secret out of the file with a `${VAR}` reference; the variable is
resolved from the environment (and therefore from `.env`) when the file is
loaded:

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

`${VAR:-default}` supplies a fallback. `promptise validate agent.superagent
--check-env` reports any reference that is not set. See
[SuperAgent Files](../core/agents/superagent-files.md).

## Precedence

When the same setting is available in more than one place:

1. A value **in code** (`Model(api_key=...)`, or a `.superagent` field) —
   always used, nothing else is consulted for that word. An empty or blank
   value is not a value: it falls through to the next two.
2. An **environment variable** already set when the process started.
3. The **`.env` file** — only fills in what is still missing. A variable that
   is exported but *empty* counts as missing, so the file's value is used;
   `promptise models check` says so when that happens, and names the file a
   value came from.

`Model(...)` never shows `api_key` or `extra` in its `repr()`/`str()` — the
object is safe to log, put in an error message or print in a traceback.

## Check what is missing

Before running anything:

```bash
promptise models check azure:chat-prod          # what resolves, what is missing, where to find it
promptise models check openai:gpt-5-mini --ping # plus a real one-token call
```

When the string is not usable the command lists every problem with its fix
— an unset variable, one exported but empty, a `native=True` package that is
not installed — and exits with `1`. Every error Promptise raises for a
missing credential names the variable, where its value lives in the
provider's console, and the three ways to supply it:

```text
Cannot use model openai:'gpt-5-mini' (OpenAI) yet:
  - OPENAI_API_KEY is not set — platform.openai.com → API keys (e.g. sk-...)
  - put OPENAI_API_KEY in a .env file next to your script (loaded automatically, never
    overrides a set variable), export it, or pass it in code: Model(..., api_key=) — see
    promptise models env openai
```

## Other secrets

The model key is the one every project needs. The same four places work for
the rest:

| Secret | Read by | Variable |
|---|---|---|
| Upstream API token for a generated MCPcast server | `server.py` from `promptise mcpcast --auth env-token` | `MCPCAST_UPSTREAM_TOKEN` (put it in the MCP client's own `env` block for desktop clients — see [MCPcast](../mcp/server/mcpcast.md#auth-modes)) |
| JWT signing key for an MCP server | `JWTAuth(secret=...)` | your choice — pass it from `os.environ` |
| Redis / Postgres URLs | conversation stores, caches | your choice |
| Agent identity credentials (Entra, AWS, GCP, SPIFFE) | `AgentIdentity` providers | the cloud's own variables — see [Agent Identity](../identity/overview.md) |

## See also

- [Model Setup](model-setup.md) — every provider, the `Model` object, Azure AI Foundry in depth
- [Quick Start](quickstart.md) — your first agent
- [CLI reference](../core/cli.md#promptise-models-model-providers) — `promptise models`
- [Environment Resolver](../core/env-resolver.md) — `${VAR}` syntax in config files
