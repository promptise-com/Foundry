# Runtime Tool Injection

When you use `build_agent(servers=...)`, MCP tools are discovered at startup. For the default ReAct pattern, these tools are automatically available. For custom graphs, you control which nodes get tools.

## How It Works

1. `build_agent()` discovers tools from MCP servers (plus `extra_tools`, cross-agent and sandbox tools)
2. Tools are converted to LangChain `BaseTool` instances
3. `build_agent()` hands them to the engine: `PromptGraphEngine(graph, model, tools=discovered)`
4. Nodes with `inject_tools=True` receive those tools at runtime, together with every tool another node of the graph declares
5. Injected tools merge with the node's own `tools` (no duplicates — the node's own tool wins on a name clash)

Nodes without `inject_tools=True` see only their own `tools`. `agent.tool_names` lists what was discovered; it does not mean every node can call them.

## Usage

```python
from promptise.engine import PromptGraph, PromptNode

graph = PromptGraph("my-agent")

# This node gets ALL MCP tools at runtime
graph.add_node(PromptNode("search",
    instructions="Search for information.",
    inject_tools=True,
))

# This node gets NO tools (pure reasoning)
graph.add_node(PromptNode("think",
    instructions="Analyze the results.",
    inject_tools=False,
))

# This node gets ONLY its explicit tools + MCP tools
graph.add_node(PromptNode("enhanced",
    tools=[my_custom_calculator],
    inject_tools=True,   # MCP tools ALSO added
))

graph.always("search", "think")
graph.always("think", "__end__")
graph.set_entry("search")

agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={"tools": HTTPServerSpec(url="http://localhost:8000/mcp")},
    agent_pattern=graph,
)
```

## When to Use

| Scenario | `inject_tools` | Why |
|----------|---------------|-----|
| Tool-calling node | `True` | Needs MCP tools to do work |
| Pure reasoning | `False` | No tools — just thinking |
| Routing decision | `False` | Lightweight, no tool overhead |
| Guard/validation | `False` | No LLM call, no tools needed |
| Mixed (custom + MCP) | `True` + explicit `tools` | Both available |

## Without build_agent

Running a graph directly, pass the tools to the engine yourself:

```python
engine = PromptGraphEngine(graph=graph, model=model, tools=my_tools)
```
