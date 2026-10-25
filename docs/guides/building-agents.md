---
title: Build a Production AI Agent in Python, step by step with Promptise Foundry
seo_title: "Build a Production AI Agent in Python"
description: "Build an AI agent in Python step by step: @tool functions, MCP servers, conversations, memory, guardrails, approval, observability, sandbox and delegation."
keywords: build AI agent Python, AI agent tutorial, production AI agent, MCP agent guide, Python agent example, agentic AI tutorial
---

# Building AI Agents

Build a production agent in twelve small steps. Each step adds one capability to the
same agent: Python tools, MCP servers, conversations, memory, guardrails, human
approval, observability, a code sandbox, delegation to other agents, a reasoning
pattern and, at the end, the whole agent as a `.superagent` file.

!!! tip "Start here"
    Every step is a complete script you can run. For the full reference of each
    feature, follow the links at the end of each step, or start from
    [Building Agents](../core/agents/building-agents.md).

## What you need

```bash
pip install promptise
```

A `.env` file in your project with your model key (Promptise loads it automatically):

```bash title=".env"
OPENAI_API_KEY=sk-...
```

Any other provider works the same way: see [Model Setup](../getting-started/model-setup.md).
`promptise models check openai:gpt-5-mini --ping` tells you whether the key works.

## How a Promptise agent is put together

- **One function builds it.** `build_agent()` takes a model, tools and optional
  features, and returns a `PromptiseAgent`. Every feature is a keyword argument and
  off unless you pass it.
- **Tools come from two places.** Python functions you decorate with `@tool`, and the
  tools of the MCP servers you connect. The agent treats both the same way.
- **The model is a string or a `Model`.** `"openai:gpt-5-mini"`,
  `"anthropic:claude-sonnet-5-5"`, `"ollama:llama3.1"`, or
  `Model("gpt-5-mini", provider="openai", api_key=...)`. Changing it changes nothing else.

---

## Step 1: A minimal agent

```python
import asyncio
from promptise import build_agent

async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        instructions="You are a support assistant for an online shop. Be brief.",
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Say hello in five words."}]}
    )
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

`ainvoke()` takes the conversation as a list of messages and returns the state after
the run; the last message is the answer. Always `await agent.shutdown()` when you are
done: it closes MCP connections, stores and background tasks.

---

## Step 2: Give it Python tools

Put the shop's tools in their own module, so the next steps can import them:

```python title="shop_tools.py"
from promptise import CallerContext, ToolError, tool

ORDERS = {
    "A-1001": {"order_id": "A-1001", "status": "shipped", "total": 120.0, "customer": "alice"},
    "A-1002": {"order_id": "A-1002", "status": "processing", "total": 42.5, "customer": "bob"},
}

@tool
def get_order(order_id: str) -> dict:
    """Look up an order by its ID.

    Args:
        order_id: The order ID, for example "A-1001".
    """
    order = ORDERS.get(order_id)
    if order is None:
        raise ToolError(f"No order {order_id}. Ask the customer to check the ID.")
    return order

@tool
def refund_order(order_id: str, amount: float, caller: CallerContext) -> str:
    """Refund part or all of an order.

    Args:
        order_id: The order to refund.
        amount: Amount to refund, in EUR.
    """
    order = ORDERS.get(order_id)
    if order is None:
        raise ToolError(f"No order {order_id}.")
    if amount > order["total"]:
        raise ToolError(f"Refund {amount} is more than the order total {order['total']}.")
    return f"Refunded {amount:.2f} EUR on {order_id} (requested by {caller.user_id})."
```

Then pass them with `extra_tools`:

```python
import asyncio
from promptise import CallerContext, build_agent
from shop_tools import get_order, refund_order

async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        instructions="You are a support assistant for an online shop.",
        extra_tools=[get_order, refund_order],
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the status of order A-1001?"}]},
        caller=CallerContext(user_id="alice"),
    )
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

What happens:

- The tool's name is the function name, its description is the docstring, and its
  parameters come from the type hints and the `Args:` section.
- Arguments are validated before the function runs. Invalid arguments go back to the
  model as a tool error it can correct.
- `raise ToolError("...")` sends that message to the model instead of a stack trace.
- A parameter typed `CallerContext` is not shown to the model: the agent fills it in
  with the `caller=` you pass to `ainvoke()` or `chat()`.

See [Tools (@tool)](../core/tools.md) for every option (`name=`, `description=`,
`timeout=`, `Annotated` constraints, sync and async functions).

---

## Step 3: Connect an MCP server

Tools that other agents or apps should also use belong in an MCP server. Here the
order lookup moves into a server that the agent starts as a subprocess:

```python title="shop_server.py"
from promptise.mcp.server import MCPServer

server = MCPServer("shop")

@server.tool()
async def list_open_orders(customer: str) -> list[str]:
    """List the IDs of a customer's orders that have not shipped yet."""
    return ["A-1002"] if customer == "bob" else []

if __name__ == "__main__":
    server.run(transport="stdio")
```

```python
import asyncio
import sys
from promptise import build_agent
from promptise.config import StdioServerSpec
from shop_tools import get_order

async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={
            "shop": StdioServerSpec(command=sys.executable, args=["shop_server.py"]),
        },
        extra_tools=[get_order],
    )
    print(agent.tool_names)  # ['list_open_orders', 'get_order']
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Which orders of bob are still open?"}]}
    )
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

`build_agent()` connects to every server in `servers`, lists its tools and converts
their JSON Schemas. A remote server is an `HTTPServerSpec`:

```python
from promptise.config import HTTPServerSpec

servers = {
    "crm": HTTPServerSpec(url="https://crm.internal/mcp", bearer_token="..."),
}
```

!!! warning "Tool names are not namespaced"
    When two servers expose a tool with the same name, the server listed last in
    `servers` wins and a warning names both. Give the tools distinct names on the
    servers, for example with `MCPRouter(prefix="billing")`, which names a router's tools `billing_<name>`.

See [Server Configuration](../core/agents/server-specs.md) for every server option
(headers, API keys, per-caller tokens, transports) and
[MCP Server SDK](../mcp/index.md) for building servers.

---

## Step 4: Keep conversations

For a chat application, the agent should load a session's history, answer, and store
the new messages. Give it a conversation store and use `chat()`:

```python
import asyncio
from promptise import build_agent
from promptise.conversations import SQLiteConversationStore, generate_session_id
from shop_tools import get_order

async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        extra_tools=[get_order],
        conversation_store=SQLiteConversationStore("conversations.db"),
        conversation_max_messages=200,  # keep the last 200 messages per session; 0 keeps all
    )
    sid = generate_session_id()  # unpredictable "sess_..." ID

    print(await agent.chat("Where is my order A-1001?", session_id=sid, user_id="alice"))
    print(await agent.chat("And what was its total?", session_id=sid, user_id="alice"))

    sessions = await agent.list_sessions(user_id="alice")
    print([s.session_id for s in sessions])
    await agent.update_session(sid, calling_user_id="alice", title="Order A-1001")
    await agent.delete_session(sid, user_id="alice")
    await agent.shutdown()

asyncio.run(main())
```

The first `chat()` with a `user_id` makes that user the session's owner; later calls
from another user raise `SessionAccessDenied`. Generate session IDs on the server
(never take them from the client unchecked).

| Store | Backend | Use |
|---|---|---|
| `InMemoryConversationStore` | Python dict | Tests |
| `SQLiteConversationStore` | `aiosqlite` (installed with Promptise) | One process, local apps |
| `PostgresConversationStore` | `asyncpg` (`pip install asyncpg`) | Production, several processes |
| `RedisConversationStore` | `redis` | Sessions with a TTL |

See [Conversations](../core/conversations.md).

---

## Step 5: Add memory

A conversation store replays one session. Memory is different: before each run the
agent searches a memory provider for text related to the user's message and adds the
best matches to the system prompt.

```python
import asyncio
from promptise import InMemoryProvider, build_agent

async def main():
    memory = InMemoryProvider()
    await memory.add("Alice prefers refunds as store credit.", metadata={"user": "alice"})

    agent = await build_agent(
        model="openai:gpt-5-mini",
        memory=memory,
        memory_max_results=5,    # at most 5 memories per run (default 5)
        memory_auto_store=False, # True also stores each exchange (default False)
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "How does Alice like her refunds?"}]}
    )
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

| Provider | Search | Persistence | Extra |
|---|---|---|---|
| `InMemoryProvider` | Keyword match | Process only | none |
| `ChromaProvider` | Vector similarity | Disk with `persist_directory=` | `promptise[all]` |
| `Mem0Provider` | Vector and graph | Mem0 backend | `promptise[all]` |

!!! warning "Memory is shared by default"
    Providers are created with `scope=MemoryScope.SHARED`: every caller reads the same
    memories. For a multi-user agent create the provider with
    `scope=MemoryScope.PER_USER` and pass a `CallerContext` with a `user_id`. See
    [Memory scopes](../core/memory.md#memory-scopes-shared-vs-per-user).

---

## Step 6: Add guardrails

Guardrails scan the user's message before the model sees it and the answer before it
is returned. This scanner uses regular expressions only, so it downloads no model:

```python
import asyncio
from promptise import (
    Action, CredentialDetector, CustomRule, GuardrailViolation, PIICategory,
    PIIDetector, PromptiseSecurityScanner, build_agent,
)

async def main():
    scanner = PromptiseSecurityScanner(
        detectors=[
            PIIDetector(categories={PIICategory.CREDIT_CARDS, PIICategory.EMAIL}),
            CredentialDetector(),
        ],
        custom_rules=[
            CustomRule(name="internal_codename", pattern=r"\bPROJECT-ORION\b", action=Action.BLOCK),
        ],
        redact_input=True,  # also redact PII and keys in the user's message
    )
    agent = await build_agent(model="openai:gpt-5-mini", guardrails=scanner)

    # The key is replaced with [AWS_ACCESS_KEY] before the model sees it.
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Is AKIAIOSFODNN7EXAMPLE a valid key format?"}]}
    )
    print(result["messages"][-1].content)

    # A BLOCK finding stops the run before the model is called.
    try:
        await agent.ainvoke({"messages": [{"role": "user", "content": "What is PROJECT-ORION?"}]})
    except GuardrailViolation as exc:
        print("Blocked:", exc)
    await agent.shutdown()

asyncio.run(main())
```

What each part does:

- PII and credentials in the **answer** are always replaced with labels such as
  `[CREDIT_CARD]` or `[AWS_ACCESS_KEY]`.
- In the **input** they are only recorded unless `redact_input=True`.
- A finding whose action is `Action.BLOCK` (the prompt-injection classifier, or your
  own `CustomRule`) raises `GuardrailViolation` and the model is never called.

`guardrails=True` uses `PromptiseSecurityScanner.default()`, which adds the
prompt-injection classifier (a local model from Hugging Face, downloaded on first use).
See [Guardrails](../core/guardrails.md) for every detector and action.

---

## Step 7: Ask a human before sensitive tools run

An `ApprovalPolicy` pauses the run when the agent calls a matching tool and asks a
handler for a decision. A denied call returns a `DENIED` result to the model, which
then adapts.

```python
import asyncio
from promptise import ApprovalDecision, ApprovalPolicy, CallerContext, build_agent
from shop_tools import get_order, refund_order

async def approve(request):
    print(f"Approve {request.tool_name}({request.arguments})?")
    ok = request.arguments.get("amount", 0) <= 50  # your UI or rule goes here
    return ApprovalDecision(approved=ok, reason="" if ok else "Over 50 EUR needs a manager.")

async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        extra_tools=[get_order, refund_order],
        approval=ApprovalPolicy(tools=["refund_*"], handler=approve, timeout=300),
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Refund 100 EUR on order A-1001."}]},
        caller=CallerContext(user_id="alice"),
    )
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

`tools` takes glob patterns. Handlers can be an async function, a
`WebhookApprovalHandler` (HTTP with HMAC signatures) or a `QueueApprovalHandler`
(your own UI). See [Approval (HITL)](../core/approval.md); to enforce approval inside
an MCP server for every client, see
[Approval Gates](../mcp/server/approval-gates.md).

---

## Step 8: Observe every run

```python
import asyncio
from promptise import build_agent
from shop_tools import get_order

async def main():
    agent = await build_agent(model="openai:gpt-5-mini", extra_tools=[get_order], observe=True)
    await agent.ainvoke({"messages": [{"role": "user", "content": "Status of A-1002?"}]})

    stats = agent.get_stats()
    print(stats["total_tokens"], stats["llm_call_count"], stats["tool_call_count"], stats["error_count"])
    agent.generate_report("report.html", title="Support agent")
    await agent.shutdown()

asyncio.run(main())
```

`observe=True` records LLM turns, tool calls, tokens, latency and errors at the
`standard` level. For more control pass an `ObservabilityConfig`:

```python
from promptise import ObservabilityConfig, ObserveLevel, TransporterType

observe = ObservabilityConfig(
    level=ObserveLevel.FULL,          # OFF, BASIC, STANDARD (default), FULL
    session_name="support",
    transporters=[TransporterType.HTML, TransporterType.STRUCTURED_LOG],
    output_dir="./reports",
    log_file="./logs/agent.jsonl",
)
```

The eight transporters are `HTML`, `JSON`, `STRUCTURED_LOG`, `CONSOLE`, `PROMETHEUS`,
`OTLP`, `WEBHOOK` and `CALLBACK`. See [Observability](../core/observability.md).

---

## Step 9: Run code in a sandbox

When the agent should write and run code, give it a Docker sandbox. It needs Docker
running and `pip install "promptise[sandbox]"`.

```python
import asyncio
from promptise import build_agent

async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        sandbox={"memory_limit": "512M", "cpu_limit": 1, "timeout": 60},  # network stays "none"
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Use Python to compute the 30th Fibonacci number."}]}
    )
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

`sandbox=True` uses the defaults. The agent gets five tools: `sandbox_exec`,
`sandbox_read_file`, `sandbox_write_file`, `sandbox_list_files` and
`sandbox_install_package`. The container runs as an unprivileged user with a
read-only root file system, dropped capabilities and no network unless you set
`"network": "restricted"` or `"full"`. Unknown keys raise an error, and when the
sandbox cannot start `build_agent()` raises instead of building an agent without it.
See [Sandbox](../core/sandbox.md).

---

## Step 10: Delegate to other agents

A peer agent becomes a tool named `ask_agent_<name>`:

```python
import asyncio
from promptise import CrossAgent, build_agent
from shop_tools import get_order

async def main():
    policy_expert = await build_agent(
        model="openai:gpt-5-mini",
        instructions="You know the shop's refund policy: refunds within 30 days, store credit or card.",
    )
    agent = await build_agent(
        model="openai:gpt-5-mini",
        extra_tools=[get_order],
        cross_agents={
            "policy": CrossAgent(agent=policy_expert, description="Answers refund policy questions.", timeout=60),
        },
    )
    print(agent.tool_names)  # ['get_order', 'ask_agent_policy']
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Can I still get a refund after 3 weeks?"}]}
    )
    print(result["messages"][-1].content)
    await agent.shutdown()
    await policy_expert.shutdown()

asyncio.run(main())
```

Delegation runs in the same process. `include_broadcast=True` adds a
`broadcast_to_agents` tool that asks several peers at once, and
`max_delegation_depth` (default 3) limits chains of agents asking agents. See
[Cross-Agent Delegation](../core/agents/cross-agent.md).

---

## Step 11: Choose a reasoning pattern

The default is a tool loop (`"react"`). Other patterns change how the agent thinks:

```python
import asyncio
from promptise import build_agent
from shop_tools import get_order

async def main():
    agent = await build_agent(
        model="openai:gpt-5-mini",
        extra_tools=[get_order],
        agent_pattern="verify",  # plan, solve, then check the answer before replying
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Is order A-1001 shipped, and what did it cost?"}]}
    )
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

Pattern names: `react` (default), `verify`, `managed`, `code-action`, `peoatr`,
`research`, `autonomous`, `deliberate`, `debate`. Any other string raises
`ValueError`. You can also pass your own `PromptGraph`:

```python
from promptise.engine import PlanNode, PromptGraph, PromptNode, SynthesizeNode

graph = PromptGraph("support", nodes=[
    PlanNode("plan", is_entry=True),
    PromptNode("act", inject_tools=True),
    SynthesizeNode("answer", is_terminal=True),
])
# build_agent(..., agent_pattern=graph)
```

See [Reasoning Patterns](../core/agents/reasoning-patterns.md) and the
[Reasoning Graph](../core/engine.md).

---

## Step 12: Put the agent in a `.superagent` file

The same agent can live in a YAML file that you check in and run from the CLI:

```yaml title="support.superagent"
version: "1.0"
agent:
  model: openai:gpt-5-mini
  instructions: |
    You are a support assistant for an online shop. Be brief.
servers:
  shop:
    type: stdio
    command: python
    args: ["shop_server.py"]
memory:
  provider: in_memory
observability:
  level: standard
  transporters: [html]
  output_dir: ./reports
```

```bash
promptise validate support.superagent
promptise agent support.superagent
```

Or build it in Python:

```python
from promptise import build_superagent

agent = await build_superagent("support.superagent")
```

`${VAR}` and `${VAR:-default}` in the file are read from the environment (and `.env`).
See [SuperAgent Files](../core/agents/superagent-files.md) for every key.

---

## Limits and errors

```python
agent = await build_agent(
    model="openai:gpt-5-mini",
    max_agent_iterations=25,   # most model calls per run (default 25); raises GraphIterationLimitError
    max_invocation_time=120,   # seconds per run (default 0, no limit); raises TimeoutError
)
```

| What fails | What you see |
|---|---|
| An MCP server cannot be reached at build time | `build_agent()` raises `MCPClientError` naming the server |
| The model provider is missing a key | `ModelSetupError` with the variable to set |
| A guardrail blocks the input | `GuardrailViolation` from `ainvoke()` |
| A tool raises | The model gets the error as the tool result and continues |
| Too many model calls | `GraphIterationLimitError` |

`print(agent.tool_names)` shows the tools the agent actually has.

---

## Run it from the CLI

```bash
promptise agent support.superagent                                 # chat with a .superagent file
promptise validate support.superagent                              # check the file
promptise list-tools --stdio "name=shop command=python args='shop_server.py'"  # list a server's tools
promptise run agent.py:create_agent                                # chat with the agent a factory returns
promptise run agent.py:create_agent --devtools                     # the same, with live traces and cost
```

See the [CLI Reference](../core/cli.md) and [Projects & promptise run](../core/projects.md).

---

## What's next

| Feature | Reference |
|---|---|
| `build_agent()` and `PromptiseAgent` | [Building Agents](../core/agents/building-agents.md) |
| `@tool` functions | [Tools](../core/tools.md) |
| Models and providers | [Models & Providers](../core/agents/models.md) |
| Streaming answers and tool events | [Streaming](../core/streaming.md) |
| Semantic cache, model fallback, tool optimization | [Semantic Cache](../core/cache.md), [Model Fallback](../core/fallback.md), [Tool Optimization](../core/tool-optimization.md) |
| Long-running agents with triggers | [Agent Runtime](../runtime/index.md) |

Other guides:

- [Building Production MCP Servers](production-mcp-servers.md): build the tool servers your agents connect to
- [Building Agentic Runtime Systems](agentic-runtime.md): run agents on triggers with budgets and health checks
- [Prompt Engineering](prompt-engineering.md): build tested, versioned system prompts

!!! tip "Learn the concept"

    - [How to Build an AI Agent in Python](https://www.promptise.com/guides/view/how-to-build-ai-agent-python): the tool loop, MCP tools, fallback, approvals and limits.
    - [AI Agent Production Checklist](https://www.promptise.com/guides/view/ai-agent-production-checklist): what to check before an agent goes live.
