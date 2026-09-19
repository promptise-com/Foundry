---
title: Quick Start — Build your first AI agent with Promptise Foundry in 5 minutes
description: Step-by-step guide to building your first production-ready AI agent in Python with Promptise Foundry. Includes installation, model setup, MCP tool discovery, and your first agent.run() call.
keywords: AI agent tutorial, Python agent quickstart, build AI agent, Promptise Foundry tutorial, getting started, MCP agent example
---

# Quick Start

Build your first agent in 5 minutes. No external servers needed — everything runs locally.

## Install

```bash
pip install promptise
```

## Add your API key

Put it in a file named `.env` in your project directory — Promptise loads it
automatically, in scripts and in the CLI alike:

```bash title=".env"
OPENAI_API_KEY=sk-...
```

Add `.env` to `.gitignore`. That is the whole setup for OpenAI; for any other
provider, one command prints exactly the lines it needs and where to find the
values:

```bash
promptise models env azure >> .env       # Azure OpenAI in Azure AI Foundry
promptise models check azure:chat-prod   # confirms nothing is missing
```

If you would rather keep the key somewhere else — your shell, your platform's
secret injection, or a vault you read in code with [`Model(...)`](../core/agents/models.md#in-code-model) —
every option is in [Configuration & Secrets](configuration.md), and the exact
variable name for every provider is in [one table](configuration.md#every-providers-variables).

## Your First Agent (30 seconds)

The simplest possible agent — just an LLM with instructions:

=== "Provider string"

    Credentials come from `.env` or the environment:

    ```python
    import asyncio
    from promptise import build_agent

    async def main():
        agent = await build_agent(
            model="openai:gpt-4o-mini",
            instructions="You are a helpful assistant. Be concise.",
        )

        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "What is 42 * 17?"}]
        })
        print(result["messages"][-1].content)  # "42 * 17 = 714"
        await agent.shutdown()

    asyncio.run(main())
    ```

=== "In code with `Model`"

    Credentials and settings explicit — the same words for every provider:

    ```python
    import asyncio
    from promptise import Model, build_agent

    async def main():
        agent = await build_agent(
            model=Model("gpt-4o-mini", provider="openai", api_key="sk-...", temperature=0),
            instructions="You are a helpful assistant. Be concise.",
        )

        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": "What is 42 * 17?"}]
        })
        print(result["messages"][-1].content)  # "42 * 17 = 714"
        await agent.shutdown()

    asyncio.run(main())
    ```

=== "Azure AI Foundry"

    Azure addresses a model by the *deployment* you created, so it is one
    more word:

    ```python
    from promptise import Model, build_agent

    agent = await build_agent(
        model=Model(
            "gpt-4o",
            provider="azure",
            deployment="chat-prod",                          # Foundry → Deployments → Name
            endpoint="https://my-resource.openai.azure.com/", # your resource → Overview → Endpoint
            api_key="...",                                    # your resource → Keys and Endpoint
            api_version="2024-10-21",
        ),
        instructions="You are a helpful assistant. Be concise.",
    )
    ```

That's it. `build_agent()` handles model initialization, message formatting, and execution.

!!! note "Any provider, one string or one object"
    `"azure:<deployment>"`, `"foundry:<model>"`, `"bedrock:<model-id>"`, `"gemini:gemini-2.5-pro"`, `"ollama:llama3.1"` — or the same as `Model("...", provider="...")` with the credentials in code. [Model Setup](model-setup.md) covers every provider; `promptise models check <string>` tells you what is missing before you run.

## Add Tools (2 minutes)

Agents become useful when they can call tools. Create an MCP server in the same file:

```python
import asyncio
import sys
from promptise import build_agent
from promptise.config import StdioServerSpec
from promptise.mcp.server import MCPServer

# ── Build a tool server ──
server = MCPServer("my-tools")

@server.tool()
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In production, call a real API
    return f"Sunny, 22°C in {city}"

@server.tool()
async def calculate(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))  # noqa: S307

# Save as tools.py, then:

async def main():
    agent = await build_agent(
        model="openai:gpt-4o-mini",
        servers={
            "tools": StdioServerSpec(
                command=sys.executable,
                args=["tools.py"],
            ),
        },
        instructions="You are a helpful assistant with access to tools.",
    )

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "What's the weather in Berlin?"}]
    })
    print(result["messages"][-1].content)
    # "It's sunny and 22°C in Berlin!"

    await agent.shutdown()

if __name__ == "__main__":
    # If run directly, start the MCP server
    if "--serve" in sys.argv:
        server.run(transport="stdio")
    else:
        asyncio.run(main())
```

The agent discovers `get_weather` and `calculate` automatically — no manual tool definitions.

### Already have an API?

Then don't hand-write those tools. `promptise mcpcast` turns an OpenAPI 3.x or Swagger 2 spec into a curated, editable MCP server — read-only by default, with every write gated by human approval the server enforces:

```bash
promptise mcpcast openapi.yaml --name myapi --no-curate --auth env-token
```

Point `servers=` at the generated `myapi-mcp/server.py` exactly like above. Full walkthrough: [MCPcast an Existing API](../guides/mcpcast-existing-api.md).

## Add a Custom Reasoning Pattern (3 minutes)

Instead of the default tool loop, define how your agent thinks:

```python
from promptise.engine import PromptGraph, PromptNode, NodeFlag
from promptise.engine.reasoning_nodes import ThinkNode, SynthesizeNode

agent = await build_agent(
    model="openai:gpt-4o-mini",
    servers=my_servers,
    agent_pattern=PromptGraph("analyst", nodes=[
        ThinkNode("think", is_entry=True),          # Analyze the question
        PromptNode("research", inject_tools=True),   # Use tools to gather data
        SynthesizeNode("answer", is_terminal=True),  # Produce final answer
    ]),
)
```

The agent now thinks before acting and synthesizes a structured answer — instead of jumping straight to tool calls.

**10 built-in patterns available:**

```python
agent = await build_agent(..., agent_pattern="react")       # Default tool loop
agent = await build_agent(..., agent_pattern="verify")      # Plan → Solve → Self-check (1 turn)
agent = await build_agent(..., agent_pattern="managed")     # Tool loop with facts-ledger context
agent = await build_agent(..., agent_pattern="code-action") # Writes ONE sandboxed program (1 turn)
agent = await build_agent(..., agent_pattern="peoatr")      # Plan → Act → Think → Reflect
agent = await build_agent(..., agent_pattern="research")    # Search → Verify → Synthesize
agent = await build_agent(..., agent_pattern="autonomous")  # Agent picks from node pool
agent = await build_agent(..., agent_pattern="deliberate")  # Think → Plan → Act → Observe → Reflect
agent = await build_agent(..., agent_pattern="debate")      # Proposer ↔ Critic → Judge
agent = await build_agent(..., agent_pattern="pipeline")    # Sequential chain
```

## Add Production Features (4 minutes)

Each capability is one parameter:

```python
from promptise import build_agent, CallerContext
from promptise.memory import ChromaProvider
from promptise.cache import SemanticCache
from promptise.conversations import SQLiteConversationStore

agent = await build_agent(
    model="openai:gpt-4o-mini",
    servers=my_servers,

    # Security: block injection attacks, detect PII
    guardrails=True,

    # Memory: remember context across conversations
    memory=ChromaProvider(persist_directory="./memory"),

    # Cache: serve similar queries instantly (30-50% cost savings)
    cache=SemanticCache(),

    # Conversations: persist chat history
    conversation_store=SQLiteConversationStore("conversations.db"),

    # Observability: trace every tool call, token, and decision
    observe=True,
)

# Use with per-user identity
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Analyze last quarter's revenue"}]},
    caller=CallerContext(user_id="analyst-42", roles=["analyst"]),
)
```

## What Happens Inside

When you call `ainvoke()`, this pipeline runs:

```
User message
    → Input guardrails (block injection, flag PII)
    → Memory search (inject relevant past context)
    → Cache check (return instantly if similar query cached)
    → Reasoning Engine (execute your reasoning pattern)
        → Tool discovery (auto-inject MCP tools)
        → LLM call (with system prompt, tools, context)
        → Tool execution (parallel when 2+ calls)
        → Loop until done (or budget exhausted)
    → Output guardrails (redact PII, credentials)
    → Cache store (save for future similar queries)
    → Conversation persist (store in SQLite/Postgres/Redis)
    → Return response
```

Every step is opt-in. Features you don't enable have zero overhead.

---

## Next Steps

| Want to... | Go to... |
|---|---|
| Grab a quick recipe (memory, cache, auth, approval…) | [Cookbook](cookbook.md) |
| Use Claude, Gemini, Ollama, or local models | [Model Setup](model-setup.md) |
| Understand the architecture | [Key Concepts](concepts.md) |
| Design custom reasoning patterns | [Reasoning Patterns](../core/agents/reasoning-patterns.md) |
| Build a complete production agent | [Building Agents Guide](../guides/building-agents.md) |
| Build MCP tool servers | [Building MCP Servers](../guides/production-mcp-servers.md) |
| Turn an API you already have into MCP tools | [MCPcast an Existing API](../guides/mcpcast-existing-api.md) |
| Build a customer support agent | [Lab: Customer Support](../guides/lab-customer-support.md) |
| Build a data analysis agent | [Lab: Data Analysis](../guides/lab-data-analysis.md) |
| Build a code review agent | [Lab: Code Review](../guides/lab-code-review.md) |
