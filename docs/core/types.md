# Types & ModelLike

The `types` module provides the central type aliases and re-exports used across
the Promptise framework.  Its most important export is `ModelLike`, which
defines the three ways you can specify a language model.

**Source:** `src/promptise/types.py`

## Quick example

```python
import asyncio
from promptise import build_agent, StdioServerSpec

async def main():
    # ModelLike as a string -- the most common form
    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={
            "math": StdioServerSpec(command="python", args=["-m", "math_server"]),
        },
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is 7 + 3?"}]}
    )
    print(result["messages"][-1].content)

asyncio.run(main())
```

## Concepts

### ModelLike

`ModelLike` is a union type that accepts four forms for specifying a model — the in-depth reference is [Models & Providers](agents/models.md):

```python
ModelLike = str | Model | BaseChatModel | Runnable[Any, Any]
```

| Form | Type | Example | When to use |
|------|------|---------|-------------|
| Provider string | `str` | `"openai:gpt-5-mini"` | Most common.  Promptise resolves the string to the correct LangChain chat model class. |
| `Model` object | `promptise.Model` | `Model("gpt-4o", provider="azure", deployment="chat-prod", endpoint=..., api_key=..., api_version=...)` | Credentials and settings in code, the same words for every provider — see [Models & Providers](agents/models.md#in-code-model). |
| Chat model instance | `BaseChatModel` | `ChatOpenAI(model="gpt-5-mini")` | When you need fine-grained control over model parameters (temperature, max tokens, etc.). |
| Runnable | `Runnable[Any, Any]` | A custom LangChain runnable | When you have a pre-built chain or custom model wrapper. |

#### Using a provider string

Provider strings follow the pattern `provider:model_name`:

```python
import asyncio
from promptise import build_agent

async def main():
    # OpenAI
    agent = await build_agent(model="openai:gpt-5-mini", servers={})

    # Anthropic
    agent = await build_agent(model="anthropic:claude-sonnet-4-20250514", servers={})

asyncio.run(main())
```

#### Using a chat model instance

Pass a pre-configured LangChain `BaseChatModel` for full control:

```python
import asyncio
from langchain_openai import ChatOpenAI
from promptise import build_agent

async def main():
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0.0, max_tokens=2048)
    agent = await build_agent(model=llm, servers={})
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Summarise quantum computing"}]}
    )
    print(result["messages"][-1].content)

asyncio.run(main())
```

### ServerSpec

Re-exported from `config.py` for convenience:

```python
from promptise.types import ServerSpec  # same as: from promptise.config import ServerSpec
```

See [Config & Server Specs](config.md) for full documentation.

### CrossAgent

Re-exported from `cross_agent.py` for convenience:

```python
from promptise.types import CrossAgent
```

`CrossAgent` is a frozen dataclass that wraps a peer agent runnable with a
human-readable description.  It is used for cross-agent communication where one
agent can delegate tasks to another.

```python
import asyncio
from promptise import build_agent
from promptise.types import CrossAgent

async def main():
    # Build a specialist peer agent
    peer = await build_agent(model="openai:gpt-5-mini", servers={})

    # Attach it to a main agent as a cross-agent tool
    main_agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={},
        cross_agents={
            "researcher": CrossAgent(agent=peer, description="General research assistant"),
        },
    )
    result = await main_agent.ainvoke(
        {"messages": [{"role": "user", "content": "Ask the researcher about photosynthesis"}]}
    )
    print(result["messages"][-1].content)

asyncio.run(main())
```

## API summary

| Export | Type | Description |
|--------|------|-------------|
| `ModelLike` | Type alias | `str \| Model \| BaseChatModel \| Runnable[Any, Any]` -- accepted by `build_agent(model=...)` |
| `ServerSpec` | Type alias | `StdioServerSpec \| HTTPServerSpec` -- re-exported from `config.py` |
| `CrossAgent` | Frozen dataclass | Wraps a peer agent with a description for cross-agent delegation |

### CrossAgent attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `agent` | `Runnable[Any, Any]` | *required* | The peer agent runnable |
| `description` | `str` | `""` | One-line description used in tool docs |

## Tips and gotchas

!!! tip
    The provider string format `"openai:gpt-5-mini"` is the recommended approach
    for most use cases.  It keeps your code concise and lets Promptise manage
    the LangChain model instantiation.

!!! warning
    When passing a `BaseChatModel` instance, make sure the model supports
    tool calling (function calling).  Not all LangChain chat model classes
    support it.  Most OpenAI and Anthropic models do.

!!! tip
    The `CrossAgent` wrapper is descriptive only -- it does not modify the peer
    agent.  The actual tool behaviour (ask, broadcast) is implemented by
    `make_cross_agent_tools()` in the `cross_agent` module.

## What's next

- [Config & Server Specs](config.md) -- details on `StdioServerSpec` and `HTTPServerSpec`
- [Environment Resolver](env-resolver.md) -- resolve `${VAR}` placeholders
- [Building Agents](agents/building-agents.md) -- use `ModelLike` with `build_agent`
