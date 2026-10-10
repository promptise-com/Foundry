# Cross-Agent Delegation

Enable agents to delegate tasks to peer agents using auto-generated tools like `ask_agent_researcher` (and, opt-in, `broadcast_to_agents`).

## Quick Example

```python
import asyncio
from promptise import build_agent
from promptise.config import HTTPServerSpec
from promptise.cross_agent import CrossAgent

async def main():
    # Build a specialist peer agent
    researcher = await build_agent(
        servers={"search": HTTPServerSpec(url="http://localhost:8001/mcp")},
        model="openai:gpt-5-mini",
        instructions="You are a web research specialist.",
    )

    # Build the main agent with delegation to the peer
    agent = await build_agent(
        servers={"files": HTTPServerSpec(url="http://localhost:8002/mcp")},
        model="openai:gpt-5-mini",
        cross_agents={
            "researcher": CrossAgent(agent=researcher, description="Web research", timeout=120),
        },
    )

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Research the latest trends in AI safety"}]
    })
    print(result["messages"][-1].content)

    await agent.shutdown()
    await researcher.shutdown()

asyncio.run(main())
```

## Concepts

Cross-agent delegation lets a primary agent call peer agents as if they were regular tools. When you pass the `cross_agents` parameter to `build_agent()`, Promptise generates:

1. **Per-peer ask tools** -- For each peer named `<name>`, a tool called `ask_agent_<name>` is created. The primary agent calls this tool to forward a message to that specific peer and receive its response.
2. **Broadcast tool (opt-in)** -- With `include_broadcast=True`, a single `broadcast_to_agents` tool that sends the same message to multiple peers in parallel and returns a mapping of peer name to response.

Peers are standard LangChain `Runnable` objects (typically `PromptiseAgent` instances returned by `build_agent()`). No new infrastructure is required -- delegation happens in-process via async calls.

```
Primary Agent
  |
  |-- ask_agent_researcher(message="...")  --> Researcher Agent --> response
  |-- ask_agent_analyst(message="...")     --> Analyst Agent   --> response
  |-- broadcast_to_agents(message="...")   --> [all peers]     --> {name: response}   (include_broadcast=True)
```

## The `CrossAgent` Dataclass

`CrossAgent` is a frozen dataclass that wraps a peer agent with metadata for tool generation.

```python
from promptise.cross_agent import CrossAgent

peer = CrossAgent(
    agent=researcher_agent,                 # any LangChain Runnable
    description="Searches the web and summarizes findings",
    timeout=120,                            # seconds; optional
)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `agent` | `Runnable[Any, Any]` | **required** | The peer agent. Must accept `{"messages": [...]}` input and return a result with extractable text. |
| `description` | `str` | `""` | One-line description used in the auto-generated tool's docstring. Helps the primary agent decide when to delegate. |
| `timeout` | `float \| None` | `None` | Seconds to wait for this peer's answer. `None` uses `delegation_timeout` from `build_agent()`; when both are `None` there is no limit beyond the peer's own `max_invocation_time`. |

`CrossAgent` is also importable as `from promptise import CrossAgent`.

!!! tip "Write good descriptions"
    The `description` field directly influences when the LLM chooses to delegate. Be specific: `"Accurate math calculations and equation solving"` is better than `"Math agent"`.

## Auto-Generated Tools

### `ask_agent_<name>`

One tool per peer. The primary agent uses this to delegate a specific task to a single peer.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `message` | `str` | Yes | The message to forward to the peer agent (becomes a user message). |
| `context` | `str \| None` | No | Optional caller context (constraints, partial results, style guide). Injected as a system message before the user message. |

The model cannot choose a timeout: it is set in Python (see [Timeouts](#timeouts)).

**Example tool call (as seen by the LLM):**

```json
{
  "name": "ask_agent_researcher",
  "arguments": {
    "message": "Find the top 3 papers on transformer architectures from 2025",
    "context": "Focus on efficiency improvements, not architecture changes"
  }
}
```

If the peer raises, the error is returned to the calling model as the tool's error, and the model decides what to do next.

### `broadcast_to_agents`

Added only with `build_agent(..., include_broadcast=True)`. A single tool that fans out a question to multiple peers concurrently. Each peer runs in parallel; timeouts and errors are captured per peer so one slow or failing peer does not block the others.

**Parameters:**

| Parameter | Type | Required | Description |
|---|---|---|---|
| `message` | `str` | Yes | The message sent to all selected peers. |
| `context` | `str \| None` | No | Caller context, sent to every peer as a system message before the message (as with `ask_agent_<name>`). |
| `peers` | `list[str] \| None` | No | Subset of peer names to consult. If omitted, all registered peers are queried. |

**Return value:** A `dict[str, str]` mapping each peer name to its response text (or `"Timed out"` / `"Error: <message>"`). Each peer uses its own timeout.

## Detailed Walkthrough

### Two Peers Delegating to Each Other

This example builds two specialist agents and a coordinator that can delegate to both:

```python
import asyncio
from promptise import build_agent
from promptise.config import HTTPServerSpec
from promptise.cross_agent import CrossAgent

async def main():
    # --- Build specialist agents ---
    researcher = await build_agent(
        servers={"search": HTTPServerSpec(url="http://localhost:8001/mcp")},
        model="openai:gpt-5-mini",
        instructions="You are a web research specialist. Find accurate information.",
    )

    analyst = await build_agent(
        servers={"data": HTTPServerSpec(url="http://localhost:8002/mcp")},
        model="openai:gpt-5-mini",
        instructions="You are a data analyst. Analyze data and produce insights.",
    )

    # --- Build coordinator with delegation to both ---
    coordinator = await build_agent(
        servers={},  # no direct MCP tools needed
        model="openai:gpt-5-mini",
        instructions=(
            "You coordinate research tasks. Delegate research to the researcher "
            "and data analysis to the analyst. Synthesize their results."
        ),
        cross_agents={
            "researcher": CrossAgent(
                agent=researcher,
                description="Searches the web for information and summarizes findings",
            ),
            "analyst": CrossAgent(
                agent=analyst,
                description="Analyzes datasets and produces statistical insights",
            ),
        },
    )

    # The coordinator can now call:
    #   ask_agent_researcher(message=..., context=...)
    #   ask_agent_analyst(message=..., context=...)

    result = await coordinator.ainvoke({
        "messages": [{
            "role": "user",
            "content": "Research recent AI safety papers and analyze their citation trends",
        }]
    })
    print(result["messages"][-1].content)

    await coordinator.shutdown()
    await researcher.shutdown()
    await analyst.shutdown()

asyncio.run(main())
```

### Timeouts

Timeouts prevent a slow peer from blocking the primary agent indefinitely. You set them in Python -- the model has no say:

```python
coordinator = await build_agent(
    servers={},
    model="openai:gpt-5-mini",
    cross_agents={
        "researcher": CrossAgent(agent=researcher, description="Web research", timeout=120),
        "analyst": CrossAgent(agent=analyst, description="Data analysis"),
    },
    delegation_timeout=30,   # for peers without a timeout of their own (the analyst)
)
```

When a timeout fires, the tool returns `"Timed out waiting for peer agent reply."` (in a broadcast, `"Timed out"` for that peer) instead of raising, and the model carries on. The timeout is enforced using `anyio.move_on_after`, which cancels the peer call cleanly without leaking resources.

A peer's own `max_invocation_time` still applies inside it; `max_invocation_time` on the coordinator caps the whole request, delegations included.

### Delegation limits

Delegation is bounded at run time, so an agent that keeps delegating -- to itself through a peer, or around a cycle of peers -- stops quickly instead of spending model calls:

- **Depth.** `max_delegation_depth` (default `3`) is the most nested delegations one request may make: coordinator → peer (1) → peer (2) → peer (3). A fourth level is refused.
- **Loops.** A call to a peer that is already working on the same request is refused, at any depth.

A refused call raises `DelegationError` (`from promptise import DelegationError`) inside the tool; the calling model sees it as the tool's error, for example:

```text
Delegation to 'helper' refused: 'helper' is already working on this request (helper → helper),
so the call would loop. Answer with the information you already have.
```

The depth is tracked in a context variable that follows the request through every nested agent, so it works across agents built separately. Every agent checks against its own `max_delegation_depth`:

```python
agent = await build_agent(..., cross_agents=peers, max_delegation_depth=1)  # peers may not delegate further
```

`promptise.cross_agent.get_delegation_chain()` returns the names of the peers working on the current request, outermost first.

### Tracing

With `trace_tools=True`, delegations print like any other tool call, and with `observe=True` they are recorded on the timeline:

```text
→ Invoking tool: ask_agent_billing with {'message': 'What plan is acme on?'}
✔ Tool result from ask_agent_billing: acme is on the Team plan, 12 seats.
```

### Using Context

The optional `context` parameter lets the caller inject constraints or partial results into the peer's conversation. It is inserted as a system message before the user message.

```python
# The LLM might produce a tool call like:
# ask_agent_analyst(
#     message="Analyze the correlation between paper length and citation count",
#     context="Use only papers from 2024-2025. The researcher already found 47 relevant papers."
# )
```

### Enabling the Broadcast Tool

`build_agent()` adds only the per-peer ask tools. To let the agent ask several peers at once, opt in:

```python
coordinator = await build_agent(..., cross_agents=peers, include_broadcast=True)
```

### Approval and identity

Delegation tools are ordinary tools, so `approval=` patterns apply to them: `ApprovalPolicy(tools=["ask_agent_payments"], ...)` asks a reviewer before the coordinator may ask the payments agent. The broadcast tool respects that gate: a peer whose `ask_agent_<name>` tool needs approval is not called by `broadcast_to_agents`, and its entry in the result says to use `ask_agent_<name>` instead. If the broadcast tool needs approval itself (for example `tools=["broadcast_*"]`), that approval covers every peer it reaches. A peer's own `approval=` applies inside the peer as usual.

The peer runs with the same `CallerContext` as the request that reached it, so caches, memory, guardrails and approval requests stay scoped to the original user. The delegating agent's `identity` is announced to the peer and recorded as `delegated_by` on the peer's observability events; each hop records its own delegator, and a hop made by an agent without an identity records none -- never the identity of an agent further up the chain.

### Building the tools yourself

`make_cross_agent_tools()` returns the tools so you can attach them elsewhere. It includes the broadcast tool unless you pass `include_broadcast=False`, and takes the same limits as `build_agent()`:

```python
from promptise.cross_agent import CrossAgent, make_cross_agent_tools

peers = {
    "researcher": CrossAgent(agent=researcher, description="Web research"),
}

tools = make_cross_agent_tools(
    peers,
    include_broadcast=False,     # only [ask_agent_researcher]
    timeout=60,                  # for peers without CrossAgent.timeout
    max_delegation_depth=3,
)
```

You can also customize the tool name prefix:

```python
tools = make_cross_agent_tools(
    peers,
    tool_name_prefix="delegate_to_",  # produces "delegate_to_researcher"
)
```

## API Summary

| Symbol | Import | Description |
|---|---|---|
| `CrossAgent` | `from promptise import CrossAgent` | Frozen dataclass wrapping a peer agent with `agent`, `description` and `timeout` fields. |
| `make_cross_agent_tools()` | `from promptise.cross_agent import make_cross_agent_tools` | Creates LangChain tools from a `Mapping[str, CrossAgent]`. Parameters: `peers`, `tool_name_prefix` (default `"ask_agent_"`), `include_broadcast` (default `True`), `caller_identity`, `timeout`, `max_delegation_depth` (default `3`), `on_before` / `on_after` / `on_error` trace callbacks, `requires_approval` (the agent's approval check; gated peers are left out of the broadcast). |
| `cross_agents` param | `build_agent(..., cross_agents={...})` | Pass a dict of `name -> CrossAgent` to automatically attach delegation tools to the agent. Related parameters: `include_broadcast=False`, `max_delegation_depth=3`, `delegation_timeout=None`. |
| `DelegationError` | `from promptise import DelegationError` | Raised inside an ask tool when a call would exceed `max_delegation_depth` or loop. |
| `build_superagent()` | `from promptise import build_superagent` | Builds an agent from a `.superagent` file together with every cross-agent it references. |

!!! tip "Peer agents are just Runnables"
    Any LangChain `Runnable` works as a peer -- it does not have to be a `PromptiseAgent`. A custom chain, a `PromptGraphEngine`, or even a mock runnable for testing all work as long as they accept `{"messages": [...]}` input.

!!! tip "In-process only"
    Cross-agent delegation is in-process. For remote agent delegation across machines, connect agents to shared MCP servers so they can exchange data through tools.

!!! warning "Shutdown order"
    Shut down the coordinator first, then the peers. If a peer is shut down while the coordinator is still running, delegation calls to that peer will fail. An agent built with `build_superagent()` shuts down its peers itself, in that order.

!!! warning "Circular delegation"
    Design your delegation graph as a DAG (directed acyclic graph). At run time a call back into a peer that is already working on the request is refused (see [Delegation limits](#delegation-limits)), and the `.superagent` file loader rejects circular *file references*, but a cycle still costs the model calls made before the loop is detected.

## What's Next?

- [Building Agents](building-agents.md) -- the `build_agent()` function reference.
- [SuperAgent Files](superagent-files.md) -- define cross-agent references declaratively in YAML and build the whole team with `build_superagent()`.
- [Multi-Agent Coordination](../../guides/multi-agent-teams.md) -- delegation, shared MCP servers and the runtime together.
- [Agent Runtime](../../runtime/index.md) -- long-running agents with triggers and lifecycle management.
