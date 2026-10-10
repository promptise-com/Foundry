<div align="center">

<a href="https://www.promptise.com">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/promptise-com/foundry/main/docs/assets/readme/hero-dark.svg">
  <img alt="Promptise Foundry: the foundation layer for agentic intelligence" src="https://raw.githubusercontent.com/promptise-com/foundry/main/docs/assets/readme/hero-light.svg" width="100%">
</picture>
</a>

<br/>

<a href="https://pypi.org/project/promptise/"><img alt="PyPI" src="https://img.shields.io/pypi/v/promptise?style=flat-square&color=1a1a1a&label=pypi"></a>
<a href="https://pypi.org/project/promptise/"><img alt="Python 3.10 to 3.13" src="https://img.shields.io/badge/python-3.10%20%E2%80%93%203.13-1a1a1a?style=flat-square"></a>
<a href="https://github.com/promptise-com/foundry/actions/workflows/test.yml"><img alt="Tests" src="https://github.com/promptise-com/foundry/actions/workflows/test.yml/badge.svg"></a>
<a href="https://github.com/promptise-com/foundry/blob/main/LICENSE"><img alt="License: Apache 2.0" src="https://img.shields.io/badge/license-Apache%202.0-1a1a1a?style=flat-square"></a>
<a href="https://docs.promptise.com"><img alt="Docs" src="https://img.shields.io/badge/docs-docs.promptise.com-1a1a1a?style=flat-square"></a>
<a href="https://github.com/promptise-com/foundry/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/promptise-com/foundry?style=flat-square&color=1a1a1a&label=stars"></a>

**[Website](https://www.promptise.com)** &nbsp;·&nbsp; **[Docs](https://docs.promptise.com/)** &nbsp;·&nbsp; **[Quick start](https://docs.promptise.com/getting-started/quickstart/)** &nbsp;·&nbsp; **[How it works](https://www.promptise.com/how-it-works)** &nbsp;·&nbsp; **[Examples](https://docs.promptise.com/resources/examples/)** &nbsp;·&nbsp; **[Discussions](https://github.com/promptise-com/foundry/discussions)**

</div>

<br/>

## What Promptise is

**The framework for agentic intelligence.**

Promptise gives you everything an agentic system needs, in one place and working together from the first line: agents that think, safe ways for them to act in your systems, and a harness that keeps them running and accountable. You decide what your agent should do; Promptise takes care of everything around it.

Start with one agent on your laptop and grow it into a fleet running real work, with the same framework and the same way of building all the way through. No rewrite in between, and nothing to stitch together yourself.

<br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/promptise-com/foundry/main/docs/assets/readme/stack-dark.svg">
  <img alt="The three parts of every agentic system: Agent, Interface and Harness" src="https://raw.githubusercontent.com/promptise-com/foundry/main/docs/assets/readme/stack-light.svg" width="100%">
</picture>

<br/>

| Layer | What it does for you |
|---|---|
| **Agent** · *it thinks* | Agents that work through a task step by step, check their own work and get it right. |
| **Interface** · *it acts* | Safe access to your tools, your data and the systems you already run. |
| **Harness** · *it operates* | Keeps your agents running around the clock, within budget, and accountable for everything they do. |

<br/>

## Get started in 30 seconds

```bash
pip install promptise
```

```python
import asyncio
from promptise import build_agent, PromptiseSecurityScanner, SemanticCache
from promptise.config import HTTPServerSpec
from promptise.memory import ChromaProvider

async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={
            "tools": HTTPServerSpec(url="http://localhost:8000/mcp"),
        },
        instructions="You are a helpful assistant.",
        memory=ChromaProvider(persist_directory="./memory"),  # remembers across calls
        guardrails=PromptiseSecurityScanner.default(),          # blocks injection, redacts PII
        cache=SemanticCache(),                                  # serves similar queries instantly
        observe=True,                                           # traces every step
    )

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What's the status of our pipeline?"}]
    })
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

One call. The agent discovers its tools on the MCP server by itself; memory, guardrails, cache and tracing are one argument each, and the ones you leave out cost nothing. Vector memory and the ML guardrails need the extras: `pip install "promptise[all]"`.

**Any model, one string.** `openai:gpt-5-mini` · `anthropic:claude-sonnet-4-5` · `azure:chat-prod` · `gemini:gemini-2.5-pro` · `bedrock:…` · `ollama:llama3.1`. The same string works in `build_agent()`, `.superagent` files and every CLI command, and `promptise models check <string>` tells you exactly what a provider still needs. → [Model setup](https://docs.promptise.com/getting-started/model-setup/)

<br/>

## One request, through the whole system

A customer writes in at three in the morning. The runtime wakes the agent, its identity is established, the input is checked, it gathers context, plans, calls a tool through MCP, a person approves the refund, it checks its own work, answers, and every step lands in the audit trail.

<br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/promptise-com/foundry/main/docs/assets/readme/flow-dark.svg">
  <img alt="One customer request handled across the Interface, Agent and Harness layers" src="https://raw.githubusercontent.com/promptise-com/foundry/main/docs/assets/readme/flow-light.svg" width="100%">
</picture>

<br/>

<p align="center"><a href="https://www.promptise.com/how-it-works"><b>Walk through all fourteen steps, with every module linked →</b></a></p>

<br/>

## Already have an API? MCPcast it.

```bash
promptise mcpcast openapi.yaml --profile standard --auth env-token
```

MCPcast reads an OpenAPI or Swagger document and writes a real, reviewed MCP server: a small set of tools an agent can actually use, chosen and described for agents. Reads by default; writes only when you ask, and every one of them **approval-gated on the server**, so a person signs off before anything changes, whichever MCP client calls it.

<br/>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/promptise-com/foundry/main/docs/assets/readme/mcpcast-dark.svg">
  <img alt="MCPcast turns an OpenAPI spec into a reviewed MCP server" src="https://raw.githubusercontent.com/promptise-com/foundry/main/docs/assets/readme/mcpcast-light.svg" width="100%">
</picture>

<br/>

What comes out is a project, not a script: an installable package, a launcher, a generated test suite, `pyproject.toml`, a `Dockerfile` and a README, all regenerated from `mcpcast.plan.yaml`, the one file you edit. `--eval` grades the result A to F with a real agent before you ship, and plain `promptise mcpcast` opens a [guided setup](https://docs.promptise.com/mcpcast/guided-setup/) in the terminal. Above: the real output for the Swagger Petstore spec. → [MCPcast, end to end](https://docs.promptise.com/mcpcast/)

<br/>

## Built for governance and structure

For people and teams who need their agentic systems to be accountable: who did what, on whose behalf, within which limits. These are part of the framework, not something you add later.

- **Multi-tenant, by construction.** Tag a request with a tenant, and every place data lives — [memory](https://docs.promptise.com/core/memory/), [cache](https://docs.promptise.com/core/cache/), [conversations](https://docs.promptise.com/core/conversations/), rate limits, [audit](https://docs.promptise.com/mcp/server/observability/) — stays separated per tenant. Two tenants who both have a user named `alice` can never see each other's data. It's a structural rule, not a filter you have to remember on every query. → [Multi-Tenant Platform guide](https://docs.promptise.com/guides/secure-multi-tenant-platform/)

- **Human approval, enforced on the server.** Mark a tool as needing sign-off and the approval is required no matter which app calls it — including one you didn't write. Denies on timeout, rejects self-approval, records who approved what. → [Approval Gates](https://docs.promptise.com/mcp/server/approval-gates/)

- **A real identity for each agent.** Agents authenticate as themselves to the APIs they call, backed by Microsoft Entra ID, AWS, Google Cloud, SPIFFE, or plain OIDC — so you can retire the shared API key, and every action traces to the person it acted for, even across agents calling agents. → [Agent Identity](https://docs.promptise.com/identity/overview/)

- **Audit you can hand to a reviewer.** Every action is written to a tamper-evident chain, tied to the tenant and the user. Delete one tenant's data with a single call when they ask. → [Auth & Security](https://docs.promptise.com/mcp/server/auth-security/)

- **Runs offline.** The security models, embeddings, and vector store can all run locally — so the whole stack works air-gapped, for on-premises and regulated environments where data can't leave. → [Guardrails](https://docs.promptise.com/core/guardrails/) · [Model Setup](https://docs.promptise.com/getting-started/model-setup/)

<br/>

## Everything Promptise ships

<table>
<tr><td colspan="2"><sub><b>AGENT</b> &nbsp;·&nbsp; it thinks</sub></td></tr>
<tr>
<td valign="top" width="24%">

**The Promptise Agent**

One function turns any model into a production agent.

[Explore →](https://docs.promptise.com/core/)

</td>
<td valign="top">

`Setup` &nbsp; [Build](https://docs.promptise.com/core/agents/building-agents/) · [Server config](https://docs.promptise.com/core/agents/server-specs/) · [Network server](https://docs.promptise.com/core/agents/network-server/) · [SuperAgent files](https://docs.promptise.com/core/agents/superagent-files/) · [Custom patterns](https://docs.promptise.com/core/agents/reasoning-patterns/) · [Cross-agent](https://docs.promptise.com/core/agents/cross-agent/)

`Memory & state` &nbsp; [Memory](https://docs.promptise.com/core/memory/) · [RAG](https://docs.promptise.com/core/rag/) · [Conversations](https://docs.promptise.com/core/conversations/) · [Semantic cache](https://docs.promptise.com/core/cache/) · [Context engine](https://docs.promptise.com/core/context-engine/)

`Security` &nbsp; [Guardrails](https://docs.promptise.com/core/guardrails/) · [Approval](https://docs.promptise.com/core/approval/) · [Auto-approval](https://docs.promptise.com/core/approval-classifier/) · [Sandbox](https://docs.promptise.com/core/sandbox/)

`Performance` &nbsp; [Tool optimization](https://docs.promptise.com/core/tool-optimization/) · [Fallback](https://docs.promptise.com/core/fallback/) · [Adaptive strategy](https://docs.promptise.com/core/adaptive-strategy/)

`Execution` &nbsp; [Streaming](https://docs.promptise.com/core/streaming/) · [Events](https://docs.promptise.com/core/events/) · [Observability](https://docs.promptise.com/core/observability/)

`Reference` &nbsp; [Config](https://docs.promptise.com/core/config/) · [Types](https://docs.promptise.com/core/types/) · [Default prompt](https://docs.promptise.com/core/default-prompt/) · [Callbacks](https://docs.promptise.com/core/callback-handler/) · [Tools](https://docs.promptise.com/core/tools/) · [Env resolver](https://docs.promptise.com/core/env-resolver/) · [Exceptions](https://docs.promptise.com/core/exceptions/) · [CLI](https://docs.promptise.com/core/cli/)

</td>
</tr>
<tr>
<td valign="top">

**Reasoning Engine**

Reasoning as a graph you can read and change.

[Explore →](https://docs.promptise.com/core/engine/)

</td>
<td valign="top">

`Graph` &nbsp; [Overview](https://docs.promptise.com/core/engine/) · [Nodes](https://docs.promptise.com/core/engine-nodes/) · [Edges](https://docs.promptise.com/core/engine-edges/) · [Flags](https://docs.promptise.com/core/engine-flags/) · [Internals](https://docs.promptise.com/core/engine-internals/)

`Patterns & skills` &nbsp; [Prebuilt patterns](https://docs.promptise.com/core/engine-prebuilts/) · [Skills](https://docs.promptise.com/core/engine-skills/) · [Skill registry](https://docs.promptise.com/core/skill-registry/) · [Custom reasoning](https://docs.promptise.com/guides/custom-reasoning/)

`Runtime` &nbsp; [Tool injection](https://docs.promptise.com/core/engine-tools/) · [Processors](https://docs.promptise.com/core/engine-processors/) · [Hooks](https://docs.promptise.com/core/engine-hooks/) · [Serialization](https://docs.promptise.com/core/engine-serialization/)

</td>
</tr>
<tr>
<td valign="top">

**Prompt Engineering**

Prompts built like software, versioned and tested.

[Explore →](https://docs.promptise.com/prompting/)

</td>
<td valign="top">

`Build` &nbsp; [PromptBlocks](https://docs.promptise.com/prompting/blocks/) · [ConversationFlow](https://docs.promptise.com/prompting/flows/) · [Builder](https://docs.promptise.com/prompting/builder/) · [Loader & templates](https://docs.promptise.com/prompting/loader-templates/) · [Shell injection](https://docs.promptise.com/prompting/shell-interpolation/)

`Strategies` &nbsp; [Strategies](https://docs.promptise.com/prompting/strategies/) · [Chaining](https://docs.promptise.com/prompting/chaining/) · [Context & variables](https://docs.promptise.com/prompting/context/)

`Quality` &nbsp; [Guards](https://docs.promptise.com/prompting/guards/) · [Inspector](https://docs.promptise.com/prompting/inspector/) · [Testing](https://docs.promptise.com/prompting/testing/) · [Suite & registry](https://docs.promptise.com/prompting/suite-registry/)

</td>
</tr>
<tr><td colspan="2"><sub><b>INTERFACE</b> &nbsp;·&nbsp; it acts</sub></td></tr>
<tr>
<td valign="top">

**MCP Server, Client &amp; MCPcast**

Build a tool once; every agent can use it.

[Explore →](https://docs.promptise.com/mcp/)

</td>
<td valign="top">

`Server` &nbsp; [Guide](https://docs.promptise.com/guides/production-mcp-servers/) · [MCPcast an existing API](https://docs.promptise.com/mcp/server/mcpcast/) · [Fundamentals](https://docs.promptise.com/mcp/server/building-servers/) · [Routers & middleware](https://docs.promptise.com/mcp/server/routers-middleware/) · [Auth & security](https://docs.promptise.com/mcp/server/auth-security/) · [Multi-tenancy](https://docs.promptise.com/mcp/server/multi-tenancy/) · [Approval gates](https://docs.promptise.com/mcp/server/approval-gates/) · [Production](https://docs.promptise.com/mcp/server/production-features/) · [Caching](https://docs.promptise.com/mcp/server/caching-performance/) · [Observability](https://docs.promptise.com/mcp/server/observability/) · [Resilience](https://docs.promptise.com/mcp/server/resilience-patterns/) · [Queue](https://docs.promptise.com/mcp/server/queue/) · [Advanced](https://docs.promptise.com/mcp/server/advanced-patterns/) · [Deployment](https://docs.promptise.com/mcp/server/deployment/) · [Testing](https://docs.promptise.com/mcp/server/testing/)

`Client` &nbsp; [Guide](https://docs.promptise.com/mcp/client/) · [Tool adapter](https://docs.promptise.com/mcp/client/tool-adapter/)

</td>
</tr>
<tr><td colspan="2"><sub><b>HARNESS</b> &nbsp;·&nbsp; it operates</sub></td></tr>
<tr>
<td valign="top">

**Agent Runtime**

Run agents unattended, on budget, recoverable.

[Explore →](https://docs.promptise.com/runtime/)

</td>
<td valign="top">

`Core` &nbsp; [Processes](https://docs.promptise.com/runtime/processes/) · [Orchestration API](https://docs.promptise.com/runtime/api/) · [Manager](https://docs.promptise.com/runtime/runtime-manager/) · [Context & state](https://docs.promptise.com/runtime/context/) · [Lifecycle](https://docs.promptise.com/runtime/lifecycle/) · [Hooks](https://docs.promptise.com/runtime/hooks/) · [Conversation](https://docs.promptise.com/runtime/conversation/)

`Governance` &nbsp; [Mission](https://docs.promptise.com/runtime/governance/mission/) · [Budget](https://docs.promptise.com/runtime/governance/budget/) · [Health](https://docs.promptise.com/runtime/governance/health/) · [Secrets](https://docs.promptise.com/runtime/governance/secrets/)

`Triggers` &nbsp; [Overview](https://docs.promptise.com/runtime/triggers/) · [Cron](https://docs.promptise.com/runtime/triggers/cron/) · [Event & webhook](https://docs.promptise.com/runtime/triggers/event-webhook/) · [File watch](https://docs.promptise.com/runtime/triggers/file-watch/)

`Journal & recovery` &nbsp; [Overview](https://docs.promptise.com/runtime/journal/) · [Backends](https://docs.promptise.com/runtime/journal/backends/) · [Replay](https://docs.promptise.com/runtime/journal/replay/) · [Rewind](https://docs.promptise.com/runtime/journal/rewind/)

`Config & scale` &nbsp; [Options](https://docs.promptise.com/runtime/configuration/) · [Manifests](https://docs.promptise.com/runtime/manifests/) · [Meta-tools](https://docs.promptise.com/runtime/meta-tools/) · [Coordinator](https://docs.promptise.com/runtime/distributed/coordinator/) · [Discovery](https://docs.promptise.com/runtime/distributed/discovery-transport/) · [Dashboard](https://docs.promptise.com/runtime/dashboard/) · [CLI](https://docs.promptise.com/runtime/cli/)

</td>
</tr>
<tr>
<td valign="top">

**Agent Identity**

An authenticated identity for every agent.

[Explore →](https://docs.promptise.com/identity/overview/)

</td>
<td valign="top">

`Core` &nbsp; [Overview](https://docs.promptise.com/identity/overview/) · [Quickstart](https://docs.promptise.com/identity/quickstart/) · [Guide](https://docs.promptise.com/identity/guide/) · [Architecture](https://docs.promptise.com/identity/architecture/) · [Security](https://docs.promptise.com/identity/security/) · [Migration](https://docs.promptise.com/identity/migration/)

`Providers` &nbsp; [Microsoft Entra ID](https://docs.promptise.com/identity/providers/entra/) · [AWS IAM](https://docs.promptise.com/identity/providers/aws/) · [Google Cloud](https://docs.promptise.com/identity/providers/gcp/) · [SPIFFE / SPIRE](https://docs.promptise.com/identity/providers/spiffe/) · [Generic OIDC](https://docs.promptise.com/identity/providers/oidc/)

</td>
</tr>
<tr><td colspan="2"><sub><b>ALSO IN THE DOCS</b></sub></td></tr>
<tr>
<td valign="top"><b>Guides &amp; labs</b></td>
<td valign="top">

[Building agents](https://docs.promptise.com/guides/building-agents/) · [Context lifecycle](https://docs.promptise.com/guides/context-lifecycle/) · [Code-action](https://docs.promptise.com/guides/code-action/) · [Production MCP servers](https://docs.promptise.com/guides/production-mcp-servers/) · [Agentic runtime](https://docs.promptise.com/guides/agentic-runtime/) · [Prompt engineering](https://docs.promptise.com/guides/prompt-engineering/) · [Multi-user systems](https://docs.promptise.com/guides/multi-user-systems/) · [Agent-to-MCP identity](https://docs.promptise.com/guides/multi-user-identity/) · [Secure multi-tenant platform](https://docs.promptise.com/guides/secure-multi-tenant-platform/) · [Multi-agent coordination](https://docs.promptise.com/guides/multi-agent-teams/) &nbsp;•&nbsp; **Labs:** [Customer support](https://docs.promptise.com/guides/lab-customer-support/) · [Data analysis](https://docs.promptise.com/guides/lab-data-analysis/) · [Code review](https://docs.promptise.com/guides/lab-code-review/) · [Pipeline observer](https://docs.promptise.com/guides/lab-pipeline-observer/)

</td>
</tr>
<tr>
<td valign="top"><b>API reference</b></td>
<td valign="top">

[Agent](https://docs.promptise.com/api/agent/) · [Config](https://docs.promptise.com/api/config/) · [Memory](https://docs.promptise.com/api/memory/) · [RAG](https://docs.promptise.com/api/rag/) · [Sandbox](https://docs.promptise.com/api/sandbox/) · [Observability](https://docs.promptise.com/api/observability/) · [Identity](https://docs.promptise.com/api/identity/) · [MCP server](https://docs.promptise.com/api/mcp-server/) · [MCP client](https://docs.promptise.com/api/mcp-client/) · [Prompts](https://docs.promptise.com/api/prompts/) · [Runtime](https://docs.promptise.com/api/runtime/) · [Cross-agent](https://docs.promptise.com/api/cross-agent/) · [SuperAgent](https://docs.promptise.com/api/superagent/) · [Utilities](https://docs.promptise.com/api/utilities/)

</td>
</tr>
<tr>
<td valign="top"><b>Start here</b></td>
<td valign="top">

[Installation](https://docs.promptise.com/) · [Extras](https://docs.promptise.com/getting-started/installation-extras/) · [Quick start](https://docs.promptise.com/getting-started/quickstart/) · [Cookbook](https://docs.promptise.com/getting-started/cookbook/) · [Why Promptise](https://docs.promptise.com/getting-started/why-promptise/) · [What is MCP?](https://docs.promptise.com/getting-started/what-is-mcp/) · [Model setup](https://docs.promptise.com/getting-started/model-setup/) · [Best LLMs](https://docs.promptise.com/getting-started/best-llms-for-agents/) · [Key concepts](https://docs.promptise.com/getting-started/concepts/) · [Glossary](https://docs.promptise.com/getting-started/glossary/) &nbsp;•&nbsp; **More:** [Blog](https://docs.promptise.com/blog/) · [Showcase](https://docs.promptise.com/resources/showcase/) · [Examples](https://docs.promptise.com/resources/examples/) · [Migration](https://docs.promptise.com/resources/migration/) · [Changelog](https://docs.promptise.com/resources/changelog/) · [FAQ](https://docs.promptise.com/faq/) · [Contributing](https://docs.promptise.com/resources/contributing/)

</td>
</tr>
</table>

<br/>

## Works with what you already run

| Area | Works with |
|---|---|
| **Models** | [OpenAI](https://openai.com) · [Anthropic](https://www.anthropic.com) · [Azure OpenAI & AI Foundry](https://ai.azure.com) · [Gemini & Vertex AI](https://ai.google.dev) · [Bedrock](https://aws.amazon.com/bedrock/) · [Mistral](https://mistral.ai) · [Groq](https://groq.com) · [Ollama](https://ollama.com) · [Hugging Face](https://huggingface.co) · any LangChain chat model · `FallbackChain` for failover → [Model setup](https://docs.promptise.com/getting-started/model-setup/) |
| **Memory & vectors** | [ChromaDB](https://www.trychroma.com) · [Mem0](https://mem0.ai) · [Sentence Transformers](https://www.sbert.net) · local embeddings for air-gapped installs → [Memory](https://docs.promptise.com/core/memory/) |
| **Conversations** | [PostgreSQL](https://www.postgresql.org) · [Redis](https://redis.io) · [SQLite](https://sqlite.org) · in-memory, with session ownership enforced → [Conversations](https://docs.promptise.com/core/conversations/) |
| **Identity & auth** | [Microsoft Entra ID](https://www.microsoft.com/security/business/identity-access/microsoft-entra-id) · [AWS IAM](https://aws.amazon.com/iam/) · [Google Cloud](https://cloud.google.com) · [SPIFFE / SPIRE](https://spiffe.io) · [OIDC](https://openid.net/connect/) · JWT · OAuth 2.0 → [Agent Identity](https://docs.promptise.com/identity/overview/) |
| **Observability** | [OpenTelemetry](https://opentelemetry.io) · [Prometheus](https://prometheus.io) · [Slack](https://slack.com) · [PagerDuty](https://www.pagerduty.com) · webhook · HTML · JSON · console → [Observability](https://docs.promptise.com/core/observability/) |
| **Sandbox & deploy** | [Docker](https://www.docker.com) · [gVisor](https://gvisor.dev) · seccomp · capability dropping · [Kubernetes](https://kubernetes.io) health probes → [Sandbox](https://docs.promptise.com/core/sandbox/) |
| **Protocols** | [Model Context Protocol](https://modelcontextprotocol.io) over stdio, streamable HTTP and SSE · [OpenAPI](https://www.openapis.org) · HMAC-chained audit logs |

<br/>

---

<div align="center">

**[Contributing](CONTRIBUTING.md)** &nbsp;·&nbsp; **[Security](SECURITY.md)** &nbsp;·&nbsp; **[Changelog](CHANGELOG.md)** &nbsp;·&nbsp; **[License: Apache 2.0](LICENSE)**

<sub>Built by <a href="https://www.promptise.com"><b>Promptise</b></a> · questions and ideas in <a href="https://github.com/promptise-com/foundry/discussions">Discussions</a> · bugs in <a href="https://github.com/promptise-com/foundry/issues/new/choose">Issues</a></sub>

<sub><sup>Formerly <a href="https://github.com/cryxnet/DeepMCPAgent">DeepMCPAgent</a>, a public preview of one part of this framework (MCP-native agent tooling).</sup></sub>

</div>
