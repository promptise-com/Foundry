<!-- Hero -->
<div align="center">
  <br/>
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-light.png">
    <img src="docs/assets/logo-dark.png" width="112" alt="Promptise Foundry"/>
  </picture>

  <h1>Promptise Foundry</h1>

  <p>
    <strong>The first full-stack Agentic Engineering framework.</strong>
  </p>

  <p>
    <em>Build agents and the tools they use. Design how they reason. Run them as autonomous, governed systems.</em><br/>
    <em>Ship them to real customers — multi-tenant, secure, and observable. One framework, not a dozen libraries.</em>
  </p>

  <br/>

  <p>
    <a href="https://github.com/promptise-com/foundry/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/promptise-com/foundry?style=flat&color=%23eab308&label=%E2%98%85%20stars"></a>
    <a href="https://pypi.org/project/promptise/"><img alt="PyPI" src="https://img.shields.io/pypi/v/promptise?color=%23a855f7&label=pypi&logo=pypi&logoColor=white"></a>
    <a href="https://pypi.org/project/promptise/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-%233b82f6?logo=python&logoColor=white"></a>
    <a href="https://pypi.org/project/promptise/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/promptise?color=%2322c55e&label=downloads"></a>
    <a href="https://github.com/promptise-com/foundry/actions/workflows/test.yml"><img alt="CI" src="https://github.com/promptise-com/foundry/actions/workflows/test.yml/badge.svg"></a>
    <a href="https://github.com/promptise-com/foundry/commits/main"><img alt="Last commit" src="https://img.shields.io/github/last-commit/promptise-com/foundry?color=%2306b6d4&label=last%20commit"></a>
    <a href="https://github.com/promptise-com/foundry/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-%23f59e0b"></a>
    <a href="https://docs.promptise.com"><img alt="Docs" src="https://img.shields.io/badge/docs-latest-%2306b6d4"></a>
  </p>

  <p>
    <img alt="Async" src="https://img.shields.io/badge/100%25-async-%230ea5e9">
    <img alt="Typed" src="https://img.shields.io/badge/mypy-strict-%238b5cf6">
    <img alt="Security" src="https://img.shields.io/badge/bandit-0%20HIGH-%2322c55e">
    <img alt="MCP" src="https://img.shields.io/badge/MCP-native-%23f97316">
    <img alt="Tests" src="https://img.shields.io/badge/tests-3493-%2306b6d4">
  </p>

  <br/>

  <p>
    <a href="https://www.promptise.com"><strong>Website</strong></a>
    &nbsp;·&nbsp;
    <a href="https://docs.promptise.com/"><strong>Documentation</strong></a>
    &nbsp;·&nbsp;
    <a href="https://docs.promptise.com/getting-started/quickstart/"><strong>Quick Start</strong></a>
    &nbsp;·&nbsp;
    <a href="https://docs.promptise.com/blog/"><strong>Blog</strong></a>
    &nbsp;·&nbsp;
    <a href="https://github.com/promptise-com/foundry/discussions"><strong>Discussions</strong></a>
  </p>

  <br/>
</div>

<hr/>

<br/>

## What Promptise is

Promptise is **one framework for the whole job of building with AI agents** — the agents, the tools they use, the reasoning behind them, the runtime that keeps them running, and the security and governance to put them in front of customers. Not a single feature, but the full stack you'd otherwise assemble from a dozen separate libraries.

Most agent stacks are assembled by hand: a model SDK, a tool layer, a vector database, auth, guardrails, a job runner, logging — glued together and kept alive by you. **Promptise pulls all of it into one framework.** `build_agent()` and a Python decorator give you the agent and its tools; memory, security, multi-tenancy, human approvals, a runtime, and observability are already inside, each switched on with a parameter.

The impact: you build what your agent *does*, not the ten libraries underneath it. A prototype becomes something you can put in front of paying customers without rebuilding the production layer each time — and the same install that runs one agent on your laptop runs a fleet serving real users.

<br/>

## &nbsp;

<br/>

<div align="center">
  <h2>Get started in 30 seconds</h2>
</div>

<br/>

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

<br/>

<div align="center">
<sub>
One call. The agent finds its tools on the MCP server on its own. Memory, guardrails, cache, and tracing are each a single line —<br/>
and the ones you don't pass cost you nothing. Works with OpenAI, Anthropic, Gemini, or a local model via Ollama.
</sub>
</div>

<br/>

**Any model, one string.** `azure:chat-prod` · `foundry:Llama-3.3-70B-Instruct` · `bedrock:anthropic.claude-sonnet-4-20250514-v1:0` · `gemini:gemini-2.5-pro` · `ollama:llama3.1` — the same string works in `build_agent()`, `.superagent` files, and every CLI command. Not sure what a provider needs? `promptise models check <string>` tells you exactly what is missing — each env var and where to find it — and nothing needs installing for any provider. → [Model setup](https://docs.promptise.com/getting-started/model-setup/)

<br/>

## &nbsp;

<br/>

<div align="center">
  <h2>The five parts of the framework</h2>
  <p><sub>Each replaces a stack of libraries you'd otherwise wire together yourself.</sub></p>
</div>

<br/>

<table align="center">
<tr>
<td width="80" align="center" valign="top">
<br/>
<h1>01</h1>
<sub>🤖</sub>
</td>
<td valign="top">

### Agent

**One function turns any model into a working agent.**

`build_agent()` connects to your tool servers, discovers the tools on its own, and gives the agent what it needs to be useful in practice: memory that's searched before every reply, a security scanner that blocks prompt injection and redacts PII, response caching, sandboxed code execution, and full tracing. Each is one parameter. Any model — OpenAI, Anthropic, Gemini, a local model, or anything built on LangChain.

[Agent docs →](https://docs.promptise.com/core/agents/building-agents/)

</td>
</tr>
<tr><td colspan="2"><br/></td></tr>
<tr>
<td width="80" align="center" valign="top">
<br/>
<h1>02</h1>
<sub>🧠</sub>
</td>
<td valign="top">

### Reasoning Engine

**Decide how the agent thinks — or use a preset.**

Most tasks run fine on the default tool loop. When you need more control, lay out the agent's reasoning as a graph you can read and change: think, use tools, check its own answer, then respond. Seven presets cover common shapes — research, debate, plan-act-reflect, one-shot self-verify, write-one-program — and you build your own when none fit. No black box.

[Reasoning docs →](https://docs.promptise.com/core/engine/)

</td>
</tr>
<tr><td colspan="2"><br/></td></tr>
<tr>
<td width="80" align="center" valign="top">
<br/>
<h1>03</h1>
<sub>🔧</sub>
</td>
<td valign="top">

### MCP Server SDK

**Build a tool once; every agent can use it.**

Write a Python function, add `@server.tool()`, and it becomes an MCP tool with a schema taken straight from your type hints. The same tool works with Promptise agents and with Claude Desktop, Cursor, and any other MCP client. It comes with authentication, per-tool permissions, rate limits, circuit breakers, tamper-evident audit logs, a background job queue, and a test client that runs the whole request path without a network.

**Already have an API? MCPcast it.**

```bash
promptise mcpcast openapi.yaml --profile standard --auth env-token
export MCPCAST_UPSTREAM_TOKEN="Bearer <your API token>"
claude mcp add myapi -- python myapi-mcp/server.py
```

What comes out is a real project, not a script: an installable `myapi_mcp/` package (config, HTTP client, approval gate, one tools module per resource), a `server.py` launcher, a generated `tests/` suite, `pyproject.toml`, a `Dockerfile` and a README — regenerated from `mcpcast.plan.yaml`, the only file you edit.

Or just `promptise mcpcast` — a full-screen terminal wizard that detects the API running on your machine, explains every choice, shows the plan for review, and prints the command it ran. [Guided setup →](https://docs.promptise.com/mcpcast/guided-setup/)

The safety profile decides what is generated — `read-only` (default), `standard` adds writes, `full` adds destructive and financial operations — and every non-read tool is approval-gated on the server, so a human signs off before it runs. [MCPcast, end to end →](https://docs.promptise.com/mcpcast/)

[MCP docs →](https://docs.promptise.com/mcp/)

</td>
</tr>
<tr><td colspan="2"><br/></td></tr>
<tr>
<td width="80" align="center" valign="top">
<br/>
<h1>04</h1>
<sub>⚡</sub>
</td>
<td valign="top">

### Agent Runtime

**Keep agents running, on budget, and recoverable.**

Turn an agent into a long-running process that wakes on a schedule, a webhook, or a file change. It writes down every step, so a crash resumes from where it stopped instead of starting over. Set limits on tool calls and spend, watch for stuck or looping behavior, and require a human when it hits something risky. Run one, or a fleet across machines.

[Runtime docs →](https://docs.promptise.com/runtime/)

</td>
</tr>
<tr><td colspan="2"><br/></td></tr>
<tr>
<td width="80" align="center" valign="top">
<br/>
<h1>05</h1>
<sub>✨</sub>
</td>
<td valign="top">

### Prompt Engineering

**Prompts you can version and test, not strings you paste.**

Assemble a system prompt from typed blocks with a token budget, let it change across the phases of a conversation, and check it with the same kind of tools you use for code. Version prompts, roll back a bad one, and trace exactly how each was built — so a prompt change is a reviewable diff, not a mystery.

[Prompts docs →](https://docs.promptise.com/prompting/)

</td>
</tr>
</table>

<br/>

## &nbsp;

<br/>

<div align="center">
  <h2>Built for putting agents in front of customers</h2>
  <p><sub>The parts most teams end up hand-building. Here they're in the framework, off by default, on with a parameter.</sub></p>
</div>

<br/>

- **Multi-tenant, by construction.** Tag a request with a tenant, and every place data lives — [memory](https://docs.promptise.com/core/memory/), [cache](https://docs.promptise.com/core/cache/), [conversations](https://docs.promptise.com/core/conversations/), rate limits, [audit](https://docs.promptise.com/mcp/server/observability/) — stays separated per customer. Two customers who both have a user named `alice` can never see each other's data. It's a structural rule, not a filter you have to remember on every query. → [Multi-Tenant Platform guide](https://docs.promptise.com/guides/secure-multi-tenant-platform/)

- **Human approval, enforced on the server.** Mark a tool as needing sign-off and the approval is required no matter which app calls it — including one you didn't write. Denies on timeout, rejects self-approval, records who approved what. → [Approval Gates](https://docs.promptise.com/mcp/server/approval-gates/)

- **A real identity for each agent.** Agents authenticate as themselves to the APIs they call, backed by Microsoft Entra ID, AWS, Google Cloud, SPIFFE, or plain OIDC — so you can retire the shared API key, and every action traces to the person it acted for, even across agents calling agents. → [Agent Identity](https://docs.promptise.com/identity/overview/)

- **Audit you can hand to a reviewer.** Every action is written to a tamper-evident chain, tied to the tenant and the user. Delete one customer's data with a single call when they ask. → [Auth & Security](https://docs.promptise.com/mcp/server/auth-security/)

- **Runs offline.** The security models, embeddings, and vector store can all run locally — so the whole stack works air-gapped, for on-prem or regulated customers who can't send data out. → [Guardrails](https://docs.promptise.com/core/guardrails/) · [Model Setup](https://docs.promptise.com/getting-started/model-setup/)

<br/>

## &nbsp;

<br/>

## Everything Promptise ships

<sub>Every capability, grouped by pillar and linked to its docs. Six parts, one framework.</sub>

<br/>

<table>
<tr>
<td valign="top" width="26%">

#### 🤖&nbsp; Agent
<sub>One function turns any model into a production agent.</sub>

**[Explore →](https://docs.promptise.com/core/)**

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

#### 🧠&nbsp; Reasoning Engine
<sub>Reasoning as a graph you can read and change.</sub>

**[Explore →](https://docs.promptise.com/core/engine/)**

</td>
<td valign="top">

`Graph` &nbsp; [Overview](https://docs.promptise.com/core/engine/) · [Nodes](https://docs.promptise.com/core/engine-nodes/) · [Edges](https://docs.promptise.com/core/engine-edges/) · [Flags](https://docs.promptise.com/core/engine-flags/) · [Internals](https://docs.promptise.com/core/engine-internals/)

`Patterns & skills` &nbsp; [Prebuilt patterns](https://docs.promptise.com/core/engine-prebuilts/) · [Skills](https://docs.promptise.com/core/engine-skills/) · [Skill registry](https://docs.promptise.com/core/skill-registry/) · [Custom reasoning](https://docs.promptise.com/guides/custom-reasoning/)

`Runtime` &nbsp; [Tool injection](https://docs.promptise.com/core/engine-tools/) · [Processors](https://docs.promptise.com/core/engine-processors/) · [Hooks](https://docs.promptise.com/core/engine-hooks/) · [Serialization](https://docs.promptise.com/core/engine-serialization/)

</td>
</tr>
<tr>
<td valign="top">

#### 🔧&nbsp; MCP Server & Client
<sub>Build a tool once; every agent can use it.</sub>

**[Explore →](https://docs.promptise.com/mcp/)**

</td>
<td valign="top">

`Server` &nbsp; [Guide](https://docs.promptise.com/guides/production-mcp-servers/) · [MCPcast an existing API](https://docs.promptise.com/mcp/server/mcpcast/) · [Fundamentals](https://docs.promptise.com/mcp/server/building-servers/) · [Routers & middleware](https://docs.promptise.com/mcp/server/routers-middleware/) · [Auth & security](https://docs.promptise.com/mcp/server/auth-security/) · [Multi-tenancy](https://docs.promptise.com/mcp/server/multi-tenancy/) · [Approval gates](https://docs.promptise.com/mcp/server/approval-gates/) · [Production](https://docs.promptise.com/mcp/server/production-features/) · [Caching](https://docs.promptise.com/mcp/server/caching-performance/) · [Observability](https://docs.promptise.com/mcp/server/observability/) · [Resilience](https://docs.promptise.com/mcp/server/resilience-patterns/) · [Queue](https://docs.promptise.com/mcp/server/queue/) · [Advanced](https://docs.promptise.com/mcp/server/advanced-patterns/) · [Deployment](https://docs.promptise.com/mcp/server/deployment/) · [Testing](https://docs.promptise.com/mcp/server/testing/)

`Client` &nbsp; [Guide](https://docs.promptise.com/mcp/client/) · [Tool adapter](https://docs.promptise.com/mcp/client/tool-adapter/)

</td>
</tr>
<tr>
<td valign="top">

#### ⚡&nbsp; Agent Runtime
<sub>Run agents unattended, on budget, recoverable.</sub>

**[Explore →](https://docs.promptise.com/runtime/)**

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

#### 🔐&nbsp; Agent Identity
<sub>An authenticated identity for every agent.</sub>

**[Explore →](https://docs.promptise.com/identity/overview/)**

</td>
<td valign="top">

`Core` &nbsp; [Overview](https://docs.promptise.com/identity/overview/) · [Quickstart](https://docs.promptise.com/identity/quickstart/) · [Guide](https://docs.promptise.com/identity/guide/) · [Architecture](https://docs.promptise.com/identity/architecture/) · [Security](https://docs.promptise.com/identity/security/) · [Migration](https://docs.promptise.com/identity/migration/)

`Providers` &nbsp; [Microsoft Entra ID](https://docs.promptise.com/identity/providers/entra/) · [AWS IAM](https://docs.promptise.com/identity/providers/aws/) · [Google Cloud](https://docs.promptise.com/identity/providers/gcp/) · [SPIFFE / SPIRE](https://docs.promptise.com/identity/providers/spiffe/) · [Generic OIDC](https://docs.promptise.com/identity/providers/oidc/)

</td>
</tr>
<tr>
<td valign="top">

#### ✨&nbsp; Prompt Engineering
<sub>Prompts built like software — versioned and tested.</sub>

**[Explore →](https://docs.promptise.com/prompting/)**

</td>
<td valign="top">

`Build` &nbsp; [PromptBlocks](https://docs.promptise.com/prompting/blocks/) · [ConversationFlow](https://docs.promptise.com/prompting/flows/) · [Builder](https://docs.promptise.com/prompting/builder/) · [Loader & templates](https://docs.promptise.com/prompting/loader-templates/) · [Shell injection](https://docs.promptise.com/prompting/shell-interpolation/)

`Strategies` &nbsp; [Strategies](https://docs.promptise.com/prompting/strategies/) · [Chaining](https://docs.promptise.com/prompting/chaining/) · [Context & variables](https://docs.promptise.com/prompting/context/)

`Quality` &nbsp; [Guards](https://docs.promptise.com/prompting/guards/) · [Inspector](https://docs.promptise.com/prompting/inspector/) · [Testing](https://docs.promptise.com/prompting/testing/) · [Suite & registry](https://docs.promptise.com/prompting/suite-registry/)

</td>
</tr>
</table>

<br/>

<div align="center"><sub><b>Also in the docs</b></sub></div>

<table>
<tr>
<td valign="top" width="26%"><sub><b>📚 &nbsp;Guides & Labs</b></sub></td>
<td valign="top">

[Building agents](https://docs.promptise.com/guides/building-agents/) · [Context lifecycle](https://docs.promptise.com/guides/context-lifecycle/) · [Code-action](https://docs.promptise.com/guides/code-action/) · [Production MCP servers](https://docs.promptise.com/guides/production-mcp-servers/) · [Agentic runtime](https://docs.promptise.com/guides/agentic-runtime/) · [Prompt engineering](https://docs.promptise.com/guides/prompt-engineering/) · [Multi-user systems](https://docs.promptise.com/guides/multi-user-systems/) · [Agent-to-MCP identity](https://docs.promptise.com/guides/multi-user-identity/) · [Secure multi-tenant platform](https://docs.promptise.com/guides/secure-multi-tenant-platform/) · [Multi-agent coordination](https://docs.promptise.com/guides/multi-agent-teams/) &nbsp;•&nbsp; **Labs:** [Customer support](https://docs.promptise.com/guides/lab-customer-support/) · [Data analysis](https://docs.promptise.com/guides/lab-data-analysis/) · [Code review](https://docs.promptise.com/guides/lab-code-review/) · [Pipeline observer](https://docs.promptise.com/guides/lab-pipeline-observer/)

</td>
</tr>
<tr>
<td valign="top"><sub><b>📖 &nbsp;API reference</b></sub></td>
<td valign="top">

[Agent](https://docs.promptise.com/api/agent/) · [Config](https://docs.promptise.com/api/config/) · [Memory](https://docs.promptise.com/api/memory/) · [RAG](https://docs.promptise.com/api/rag/) · [Sandbox](https://docs.promptise.com/api/sandbox/) · [Observability](https://docs.promptise.com/api/observability/) · [Identity](https://docs.promptise.com/api/identity/) · [MCP server](https://docs.promptise.com/api/mcp-server/) · [MCP client](https://docs.promptise.com/api/mcp-client/) · [Prompts](https://docs.promptise.com/api/prompts/) · [Runtime](https://docs.promptise.com/api/runtime/) · [Cross-agent](https://docs.promptise.com/api/cross-agent/) · [SuperAgent](https://docs.promptise.com/api/superagent/) · [Utilities](https://docs.promptise.com/api/utilities/)

</td>
</tr>
<tr>
<td valign="top"><sub><b>🚀 &nbsp;Start here</b></sub></td>
<td valign="top">

[Installation](https://docs.promptise.com/) · [Extras](https://docs.promptise.com/getting-started/installation-extras/) · [Quick start](https://docs.promptise.com/getting-started/quickstart/) · [Cookbook](https://docs.promptise.com/getting-started/cookbook/) · [Why Promptise](https://docs.promptise.com/getting-started/why-promptise/) · [What is MCP?](https://docs.promptise.com/getting-started/what-is-mcp/) · [Model setup](https://docs.promptise.com/getting-started/model-setup/) · [Best LLMs](https://docs.promptise.com/getting-started/best-llms-for-agents/) · [Key concepts](https://docs.promptise.com/getting-started/concepts/) · [Glossary](https://docs.promptise.com/getting-started/glossary/) &nbsp;•&nbsp; **More:** [Blog](https://docs.promptise.com/blog/) · [Showcase](https://docs.promptise.com/resources/showcase/) · [Examples](https://docs.promptise.com/resources/examples/) · [Migration](https://docs.promptise.com/resources/migration/) · [Changelog](https://docs.promptise.com/resources/changelog/) · [FAQ](https://docs.promptise.com/faq/) · [Contributing](https://docs.promptise.com/resources/contributing/)

</td>
</tr>
</table>

<br/>

## &nbsp;

<br/>

<div align="center">
  <h2>Ecosystem</h2>
  <p><sub>Promptise plugs into what your team already runs — and runs fully offline when it has to.</sub></p>
</div>

<br/>

<div align="center">

#### &nbsp;&nbsp;Models&nbsp;&nbsp;

<a href="https://openai.com"><img alt="OpenAI" src="https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white"></a>
<a href="https://www.anthropic.com"><img alt="Anthropic" src="https://img.shields.io/badge/Anthropic-D97757?style=for-the-badge&logo=anthropic&logoColor=white"></a>
<a href="https://ai.google.dev"><img alt="Gemini" src="https://img.shields.io/badge/Gemini-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white"></a>
<a href="https://ollama.com"><img alt="Ollama" src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white"></a>
<a href="https://mistral.ai"><img alt="Mistral" src="https://img.shields.io/badge/Mistral-FA520F?style=for-the-badge&logoColor=white"></a>
<a href="https://huggingface.co"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"></a>

<sub>+ any LangChain <code>BaseChatModel</code> · <code>FallbackChain</code> for automatic failover · <a href="https://docs.promptise.com/getting-started/model-setup/">Model setup →</a></sub>

<br/><br/>

#### &nbsp;&nbsp;Memory &amp; Vectors&nbsp;&nbsp;

<a href="https://www.trychroma.com"><img alt="ChromaDB" src="https://img.shields.io/badge/ChromaDB-FF6B6B?style=for-the-badge&logoColor=white"></a>
<a href="https://mem0.ai"><img alt="Mem0" src="https://img.shields.io/badge/Mem0-111111?style=for-the-badge&logoColor=white"></a>
<a href="https://www.sbert.net"><img alt="Sentence Transformers" src="https://img.shields.io/badge/Sentence--Transformers-EE4C2C?style=for-the-badge"></a>

<sub>Local embeddings · air-gapped model paths · per-tenant isolation · <a href="https://docs.promptise.com/core/memory/">Memory →</a></sub>

<br/><br/>

#### &nbsp;&nbsp;Conversation Storage&nbsp;&nbsp;

<a href="https://www.postgresql.org"><img alt="PostgreSQL" src="https://img.shields.io/badge/PostgreSQL-4169E1?style=for-the-badge&logo=postgresql&logoColor=white"></a>
<a href="https://redis.io"><img alt="Redis" src="https://img.shields.io/badge/Redis-DC382D?style=for-the-badge&logo=redis&logoColor=white"></a>
<a href="https://sqlite.org"><img alt="SQLite" src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white"></a>

<sub>Session ownership enforced · per-tenant isolation for cache and guardrails · <a href="https://docs.promptise.com/core/conversations/">Conversations →</a></sub>

<br/><br/>

#### &nbsp;&nbsp;Identity &amp; Auth&nbsp;&nbsp;

<a href="https://www.microsoft.com/security/business/identity-access/microsoft-entra-id"><img alt="Microsoft Entra ID" src="https://img.shields.io/badge/Entra%20ID-0078D4?style=for-the-badge&logo=microsoftazure&logoColor=white"></a>
<a href="https://aws.amazon.com/iam/"><img alt="AWS IAM" src="https://img.shields.io/badge/AWS%20IAM-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white"></a>
<a href="https://cloud.google.com"><img alt="Google Cloud" src="https://img.shields.io/badge/Google%20Cloud-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white"></a>
<a href="https://spiffe.io"><img alt="SPIFFE" src="https://img.shields.io/badge/SPIFFE%2FSPIRE-000000?style=for-the-badge"></a>
<a href="https://openid.net/connect/"><img alt="OIDC" src="https://img.shields.io/badge/OIDC-F78C40?style=for-the-badge&logo=openid&logoColor=white"></a>

<sub>A verifiable identity per agent — no shared keys · <a href="https://docs.promptise.com/identity/overview/">Agent Identity →</a></sub>

<br/><br/>

#### &nbsp;&nbsp;Observability&nbsp;&nbsp;

<a href="https://opentelemetry.io"><img alt="OpenTelemetry" src="https://img.shields.io/badge/OpenTelemetry-425CC7?style=for-the-badge&logo=opentelemetry&logoColor=white"></a>
<a href="https://prometheus.io"><img alt="Prometheus" src="https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white"></a>
<a href="https://slack.com"><img alt="Slack" src="https://img.shields.io/badge/Slack-4A154B?style=for-the-badge&logo=slack&logoColor=white"></a>
<a href="https://www.pagerduty.com"><img alt="PagerDuty" src="https://img.shields.io/badge/PagerDuty-06AC38?style=for-the-badge&logo=pagerduty&logoColor=white"></a>

<sub>8 transporters: OTel · Prometheus · Slack · PagerDuty · Webhook · HTML · JSON · Console · <a href="https://docs.promptise.com/core/observability/">Observability →</a></sub>

<br/><br/>

#### &nbsp;&nbsp;Sandbox &amp; Deployment&nbsp;&nbsp;

<a href="https://www.docker.com"><img alt="Docker" src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"></a>
<a href="https://gvisor.dev"><img alt="gVisor" src="https://img.shields.io/badge/gVisor-4285F4?style=for-the-badge&logoColor=white"></a>
<a href="https://en.wikipedia.org/wiki/Seccomp"><img alt="seccomp" src="https://img.shields.io/badge/seccomp-111111?style=for-the-badge"></a>
<a href="https://kubernetes.io"><img alt="Kubernetes" src="https://img.shields.io/badge/Kubernetes-326CE5?style=for-the-badge&logo=kubernetes&logoColor=white"></a>

<sub>Docker + seccomp + gVisor + capability dropping · Kubernetes health probes · <a href="https://docs.promptise.com/core/sandbox/">Sandbox →</a></sub>

<br/><br/>

#### &nbsp;&nbsp;Protocols&nbsp;&nbsp;

<a href="https://modelcontextprotocol.io"><img alt="MCP" src="https://img.shields.io/badge/MCP-native-F97316?style=for-the-badge"></a>
<a href="https://www.openapis.org"><img alt="OpenAPI" src="https://img.shields.io/badge/OpenAPI-6BA539?style=for-the-badge&logo=openapiinitiative&logoColor=white"></a>
<a href="https://datatracker.ietf.org/doc/html/rfc7519"><img alt="JWT" src="https://img.shields.io/badge/JWT-000000?style=for-the-badge&logo=jsonwebtokens&logoColor=white"></a>
<a href="https://oauth.net/2/"><img alt="OAuth 2.0" src="https://img.shields.io/badge/OAuth%202.0-1E78D4?style=for-the-badge"></a>

<sub>stdio · streamable HTTP · SSE · HMAC-chained audit logs</sub>

</div>

<br/>

## &nbsp;

<div align="center">
  <sub>
    Want to ship with us? See <a href="CONTRIBUTING.md">CONTRIBUTING.md</a> · join <a href="https://github.com/promptise-com/foundry/discussions">Discussions</a> · file an <a href="https://github.com/promptise-com/foundry/issues/new/choose">issue</a>.
  </sub>
</div>

<br/>

---

<br/>

<div align="center">

  [**Contributing**](CONTRIBUTING.md) &nbsp;·&nbsp; [**Security**](SECURITY.md) &nbsp;·&nbsp; [**License: Apache 2.0**](LICENSE)

  <br/>
  <br/>

  <sub>Built by <a href="https://www.promptise.com"><strong>Promptise</strong></a></sub>

  <br/>
  <br/>

  <sub><sup>Formerly <a href="https://github.com/cryxnet/DeepMCPAgent">DeepMCPAgent</a> — a public preview of one sliver of this framework (MCP-native agent tooling). Promptise Foundry is the full system it hinted at: reasoning engine, agent runtime, prompt engineering, sandboxed execution, governance, and observability.</sup></sub>

  <br/>
</div>
