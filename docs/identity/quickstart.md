# Quickstart

Give an agent a traceable identity in five minutes — no infrastructure
required. Then, optionally, make it verifiable.

## 1. A local identity (no key, no infrastructure)

Creating and inspecting an identity needs nothing — no cloud, no model API key:

```python
from promptise import AgentIdentity

identity = AgentIdentity(
    "billing-bot",
    name="Billing Bot",
    owner="payments-team",
    labels={"env": "prod"},
)

identity.agent_id          # "billing-bot"
identity.is_verifiable     # False — local identity (id only)
identity.claims()          # {"agent_id": "billing-bot", "verifiable": False,
                           #  "name": "Billing Bot", "owner": "payments-team", ...}
```

That id is what the framework stamps onto everything the agent does. Run it as a
script: [`examples/identity/local/app.py`](https://github.com/promptise-com/foundry/blob/main/examples/identity/local/app.py).

## 2. See it on the agent's timeline (needs a model key)

Attach the identity to an agent and enable observability — every tool call and
LLM turn is then tagged with `agent_id="billing-bot"`, so across a fleet you can
answer *which agent did what*:

```python
import asyncio

from promptise import AgentIdentity, build_agent


async def main() -> None:
    identity = AgentIdentity("billing-bot", name="Billing Bot", owner="payments-team")
    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={},
        identity=identity,
        observe=True,   # enable the timeline so attribution is visible
    )
    await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Summarize today's invoices."}]}
    )
    await agent.shutdown()


asyncio.run(main())
```

!!! info "You need an API key first"
    Every model provider needs a credential. Put it in a `.env` file in your project — Promptise loads it automatically, in scripts and in the CLI — or pass it in code with `Model(...)` — explained in [**Models & Providers → In code: Model**](../core/agents/models.md#in-code-model): `Model("gpt-4o-mini", provider="openai", api_key="...")`. The **exact variable name for every provider** (Azure AI Foundry, Bedrock, Gemini, Ollama, …) is in [**Configuration & Secrets → Every provider's variables**](../getting-started/configuration.md#every-providers-variables); self-hosted and inference endpoints are in [**Models & Providers → Custom, self-hosted and inference endpoints**](../core/agents/models.md#custom-self-hosted-and-inference-endpoints). `promptise models env <provider>` prints the lines a provider needs and `promptise models check <model>` tells you what is still missing.

## 3. Make it verifiable

When the agent calls resources that must *verify* who it is — an MCP
server, a protected API — back the identity with a credential provider.
The signed credential is what the agent presents; the resource validates
it and attributes the caller.

=== "Generic OIDC (works anywhere)"

    ```python
    identity = AgentIdentity.from_oidc(
        "billing-bot",
        issuer="https://gitlab.com",
        token_env_var="CI_JOB_JWT_V2",
    )
    ```

=== "Microsoft Entra"

    ```python
    identity = AgentIdentity.from_entra(
        "billing-bot", client_id="...", resource="api://my-mcp-server"
    )
    ```

=== "Auto-detect"

    ```python
    identity = AgentIdentity.auto("billing-bot")   # picks the platform
    ```

Then present it to the resources the agent calls:

```python
identity.get_credential()    # a short-lived, signed identity JWT
identity.auth_header()       # {"Authorization": "Bearer <jwt>"}
```

## 4. Present the identity to an MCP server

A verifiable identity plugs into Promptise's MCP auth **automatically**.
Pass it to `build_agent` and every MCP server that has no bearer of its
own receives the agent's identity credential, so the server can
authenticate and attribute the calling agent (via its JWT auth and
`RequireClientId` / role guards):

```python
from promptise.config import HTTPServerSpec

agent = await build_agent(
    model="anthropic:claude-sonnet-4-5",
    servers={"tools": HTTPServerSpec(url="https://tools.internal/mcp")},
    identity=identity,   # presented to "tools" as its bearer token
)
```

To override for a specific server, set its `bearer_token` explicitly —
an explicit per-server token always wins. You can also present the
credential by hand anywhere with `identity.get_credential()` /
`identity.auth_header()`.

## 5. One identity, several resources

When the agent calls more than one protected resource, give each server
its own `audience`. The same identity then mints a credential scoped to
each — no need for a second `AgentIdentity`:

```python
from promptise.config import HTTPServerSpec

agent = await build_agent(
    model="anthropic:claude-sonnet-4-5",
    identity=identity,
    servers={
        "billing": HTTPServerSpec(url="https://billing.internal/mcp",
                                  audience="api://billing"),
        "crm":     HTTPServerSpec(url="https://crm.internal/mcp",
                                  audience="api://crm"),
    },
)
```

Providers that can re-mint on demand (Entra IMDS, AWS STS, GCP metadata,
SPIFFE SDK) honour the per-server audience; fixed-audience sources
(projected-token files, OIDC file/env) use the audience the platform
issued. See [Per-resource credentials](architecture.md#per-resource-credentials).

## 6. Declarative — `.superagent` files

You don't have to wire identity in Python. Add an `identity:` block to a
[`.superagent` file](../core/agents/superagent-files.md#identity) and it is
applied automatically by `build_agent`, the CLI (`promptise agent`,
`promptise serve`), and `SuperAgentLoader`:

```yaml
version: "1.0"
agent:
  model: "anthropic:claude-sonnet-4-5"
identity:
  provider: entra                  # local|entra|aws|gcp|spiffe|oidc|auto
  agent_id: billing-bot
  owner: payments
  client_id: "${AZURE_CLIENT_ID}"
  resource: api://my-mcp-server
servers:
  tools:
    type: http
    url: "https://tools.internal/mcp"
```

All string fields support `${ENV_VAR}` resolution. For `provider: local`
only `agent_id` is needed; for `provider: oidc`, `issuer` plus exactly one
of `token_file` / `token_env_var`.

See the [provider pages](overview.md#credential-providers) for
per-platform setup, and [Architecture](architecture.md) for how the
identity flows through the framework.
