# Engine Internals

How the Reasoning Engine works under the hood — execution flow, transition resolution, autonomous path building, caching, flag processing, and the complete data flow from `build_agent()` to tool response.

## The Big Picture

```mermaid
graph TB
    subgraph "build_agent()"
        BA[build_agent] --> NM[Normalize Model]
        BA --> MC[Connect MCP Servers]
        BA --> BG[Build Graph]
        MC --> TD[Discover Tools]
        TD --> TA[Convert to LangChain Tools]
        BG --> |"pattern string"| PB[Prebuilt Pattern]
        BG --> |"PromptGraph"| CG[Custom Graph]
        BG --> |"node_pool"| AP[Autonomous Pool]
        BG --> |"None"| DR[Default ReAct]
        PB --> ENG[PromptGraphEngine]
        CG --> ENG
        AP --> ENG
        DR --> ENG
        NM --> ENG
        TA --> ENG
    end

    ENG --> |"ainvoke(input, caller)"| EXEC[Execution Loop]

    style BA fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style ENG fill:#0e4429,stroke:#4ade80,color:#fff
    style EXEC fill:#2d1b4e,stroke:#c084fc,color:#fff
```

## Execution Loop — What Happens on Every `ainvoke()`

```mermaid
graph TD
    START[ainvoke called] --> COPY[Copy graph per-invocation isolation]
    COPY --> TOOLS[Collect all tools from nodes]
    TOOLS --> STATE[Create GraphState with messages]
    STATE --> LOOP{current_node != __end__?}

    LOOP --> |yes| GET[Get node from graph]
    GET --> |not found| END_ERR[Log error, end]
    GET --> |found| PRE_HOOK[Run pre_node hooks]

    PRE_HOOK --> PRE_FLAG[Pre-execute flags]
    PRE_FLAG --> |SKIP_ON_ERROR + prev errored| SKIP[Return skip result]
    PRE_FLAG --> |CACHEABLE + cache hit| CACHE_HIT[Return cached result]
    PRE_FLAG --> |NO_HISTORY| STRIP[Strip messages, keep system only]
    PRE_FLAG --> |ISOLATED_CONTEXT| ISOLATE[Save context, isolate]
    PRE_FLAG --> |LIGHTWEIGHT| SWAP_MODEL[Swap to lightweight model]
    PRE_FLAG --> |none triggered| EXEC_NODE[Execute node]

    SKIP --> POST_FLAG
    CACHE_HIT --> POST_FLAG
    EXEC_NODE --> |RETRYABLE flag| RETRY[Execute with retry + backoff]
    EXEC_NODE --> |normal| NORMAL[node.execute state, config]
    RETRY --> POST_FLAG
    NORMAL --> POST_FLAG

    POST_FLAG[Post-execute flags] --> |CRITICAL + error| ABORT[Abort graph]
    POST_FLAG --> |NO_HISTORY| RESTORE_MSG[Restore messages]
    POST_FLAG --> |ISOLATED_CONTEXT| RESTORE_CTX[Restore context, merge output_key]
    POST_FLAG --> |CACHEABLE| CACHE_STORE[Store in cache]
    POST_FLAG --> |SUMMARIZE_OUTPUT| SUMMARIZE[LLM summarize if > 1000 chars]
    POST_FLAG --> |VALIDATE_OUTPUT| VALIDATE[Validate against schema]
    POST_FLAG --> POST_HOOK

    POST_HOOK[Run post_node hooks] --> |hook sets __end__| END
    POST_HOOK --> TRANSITION[Resolve transition]
    TRANSITION --> LOOP

    ABORT --> END[Build ExecutionReport]
    END_ERR --> END
    LOOP --> |no| END
    END --> |run ended on an unrecovered node failure| RAISE[raise GraphExecutionError]

    style START fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style EXEC_NODE fill:#0e4429,stroke:#4ade80,color:#fff
    style ABORT fill:#5c1a1a,stroke:#f87171,color:#fff
    style END fill:#1a1a1a,stroke:#666,color:#fff
    style CACHE_HIT fill:#1a2e3a,stroke:#22d3ee,color:#fff
```

## Transition Resolution — How the Engine Picks the Next Node

After every node executes, the engine decides where to go next. This is the 7-step priority chain:

```mermaid
graph TD
    NE[Node finished executing] --> TC{Tool calls in result?}
    TC --> |"yes, transition_reason = tool_calls_present"| SAME[Re-enter same node]
    TC --> |no| LLM{output.route or _next or goto set?}

    LLM --> |"yes, target exists in graph"| TARGET[Go to named node]
    LLM --> |no| NR{NodeResult.next_node set?}

    NR --> |"yes (reasoning nodes set this)"| TARGET2[Go to next_node]
    NR --> |no| EDGE{Conditional edges from this node?}

    EDGE --> |"match (checked by priority desc)"| TARGET3[Follow matched edge]
    EDGE --> |no match| TR{Node transitions dict match output?}

    TR --> |yes| TARGET4[Follow transition]
    TR --> |no| DEF{node.default_next set?}

    DEF --> |yes| TARGET5[Go to default]
    DEF --> |no| GRAPH_END[__end__]

    SAME --> |"LLM sees tool results, calls again or produces final answer"| NE

    style NE fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style SAME fill:#0e4429,stroke:#4ade80,color:#fff
    style GRAPH_END fill:#1a1a1a,stroke:#666,color:#fff
    style TARGET fill:#2d1b4e,stroke:#c084fc,color:#fff
    style TARGET2 fill:#2d1b4e,stroke:#c084fc,color:#fff
    style TARGET3 fill:#2d1b4e,stroke:#c084fc,color:#fff
```

**Key insight:** Tool-calling nodes re-enter themselves automatically. The LLM calls tools → engine adds ToolMessages to conversation → re-enters the node → LLM sees results → calls more tools or produces final answer. This loop continues until the LLM stops calling tools.

## Autonomous Mode — How the AI Builds Its Own Path

```mermaid
sequenceDiagram
    participant E as Engine
    participant A as AutonomousNode
    participant LLM as LLM
    participant P as Pool Nodes

    E->>A: execute(state, config)

    loop For each step (up to max_steps)
        A->>LLM: "Here are available nodes:<br/>- plan: Break task into subgoals<br/>- act: Execute tools<br/>- think: Analyze gaps<br/>- reflect: Self-evaluate<br/>- answer: Final synthesis (terminal)<br/><br/>Which should run next?"
        LLM-->>A: {"next_node": "act", "reason": "Need to search for data"}

        alt next_node == "__done__" or terminal node succeeded
            A-->>E: Return final result
        else Valid node selected
            A->>P: Execute chosen node (e.g. "act")
            P-->>A: NodeResult with output
            A->>A: Append result to state, continue loop
        else Invalid node name
            A->>A: Log warning, ask LLM again
        end
    end
```

In autonomous mode, the `AutonomousNode` wraps the pool and orchestrates:

1. Builds a routing prompt listing all available nodes with descriptions
2. Asks the LLM which node to run next
3. Extracts JSON from the response (balanced brace parser for nested objects)
4. Executes the chosen node
5. Records result in state
6. Loops back to step 2 until the LLM chooses `__done__` or a terminal node succeeds

The developer controls the pool composition — what nodes are available. The LLM controls the path — which nodes to use and in what order.

## PromptNode Execution — The 9-Step Pipeline

Every PromptNode runs this pipeline internally:

```mermaid
graph TD
    START[PromptNode.execute] --> PRE[1. Preprocessor]
    PRE --> CTX[2. Context Assembly]
    CTX --> |instructions + blocks + input_keys + inherited context| CTX2[+ observations + plan + reflections]
    CTX2 --> |"skip observations for tool nodes"| SCHEMA[+ Auto tool schema injection]
    SCHEMA --> STRAT[3. Strategy wrapping + perspective]
    STRAT --> BIND[4. Tool binding + structured output]
    BIND --> |"first call: build + cache"| CACHE_CHECK{Cached from previous tool loop?}
    CACHE_CHECK --> |yes| REUSE[Reuse cached sys msg + model]
    CACHE_CHECK --> |no| BUILD[Build SystemMessage + bind_tools]
    BUILD --> CACHE_STORE[Cache in config for re-entry]
    REUSE --> LLM[5. LLM call]
    CACHE_STORE --> LLM

    LLM --> RESP{6. Response type?}
    RESP --> |AIMessage with tool_calls| TOOL_EXEC[Execute tools]
    RESP --> |AIMessage text only| PARSE[Parse output]
    RESP --> |Structured output| STRUCT[Use as-is]

    TOOL_EXEC --> |"2+ calls"| PARALLEL[asyncio.gather parallel]
    TOOL_EXEC --> |"1 call"| SEQUENTIAL[Sequential execution]
    PARALLEL --> TOOL_MSG[Append ToolMessages to state]
    SEQUENTIAL --> TOOL_MSG
    TOOL_MSG --> |"transition_reason = tool_calls_present"| RETURN[Return result → engine re-enters node]

    PARSE --> GUARD[7. Guard checking sync + async]
    STRUCT --> GUARD
    GUARD --> STRAT_PARSE[8. Strategy parsing]
    STRAT_PARSE --> POST[9. Postprocessor]
    POST --> WRITE[Write output_key to state.context]
    WRITE --> DONE[Return NodeResult]

    style START fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style LLM fill:#2d1b4e,stroke:#c084fc,color:#fff
    style PARALLEL fill:#0e4429,stroke:#4ade80,color:#fff
    style RETURN fill:#3a2a0a,stroke:#fbbf24,color:#fff
    style DONE fill:#1a1a1a,stroke:#666,color:#fff
```

## Caching Strategy — What Gets Cached and When

```mermaid
graph LR
    subgraph "First execution (cold)"
        A1[Build system prompt] --> A2[Resolve tools]
        A2 --> A3[bind_tools on model]
        A3 --> A4[Create SystemMessage]
        A4 --> A5[Build tool_map dict]
        A5 --> A6["Cache all in config[_node_cache_{name}]"]
    end

    subgraph "Tool loop re-entry (hot)"
        B1["Read from config cache"] --> B2[Reuse SystemMessage]
        B2 --> B3[Reuse bound model]
        B3 --> B4[Reuse tool_map]
        B4 --> B5[Skip: prompt assembly, tool resolution, bind_tools]
    end

    A6 -.-> B1

    style A6 fill:#22d3ee,stroke:#0891b2,color:#000
    style B1 fill:#22d3ee,stroke:#0891b2,color:#000
```

**What's cached per node:**

| Item | Cache key | Rebuilt when? |
|------|-----------|---------------|
| SystemMessage (assembled prompt) | `config[_node_cache_{name}].sys_msg` | Never during same invocation |
| Bound model (with tools) | `config[_node_cache_{name}].model` | Never during same invocation |
| Active tools list | `config[_node_cache_{name}].tools` | Never during same invocation |
| Tool name→instance map | `config[_tool_map_{name}]` | Never during same invocation |
| Edge adjacency index | `graph._edge_index` | On edge add/remove (lazy dirty flag) |
| Node result (CACHEABLE flag) | `state.context._node_cache[hash]` | When input_keys change |

## Flag Processing Order

Flags are processed in two phases around node execution:

```mermaid
graph TD
    subgraph "Pre-execute (before node runs)"
        F1[SKIP_ON_ERROR] --> |"previous errored → skip"| F1R[Return skip result]
        F2[CACHEABLE] --> |"cache hit → return cached"| F2R[Return cached result]
        F3[NO_HISTORY] --> |"save messages, strip to system only"| F3R[Continue]
        F4[ISOLATED_CONTEXT] --> |"save context, give empty/input_keys only"| F4R[Continue]
        F5[LIGHTWEIGHT] --> |"swap config model to lightweight_model"| F5R[Continue]
        F6[REQUIRES_HUMAN] --> |"set flag in state, emit hook event"| F6R[Continue]
    end

    subgraph "Node executes"
        EXEC["node.execute(state, config)"]
        RETRY["If RETRYABLE: retry with exp backoff, max 8s cap"]
    end

    subgraph "Post-execute (after node runs)"
        P1[CRITICAL] --> |"error → abort graph immediately"| P1R[Break loop]
        P2[NO_HISTORY] --> |"restore original messages + new ones"| P2R[Continue]
        P3[ISOLATED_CONTEXT] --> |"restore context, merge output_key back"| P3R[Continue]
        P4[LIGHTWEIGHT] --> |"restore original model"| P4R[Continue]
        P5[CACHEABLE] --> |"store result in cache"| P5R[Continue]
        P6[VERBOSE] --> |"log full output at DEBUG"| P6R[Continue]
        P7[OBSERVABLE] --> |"emit metrics event to hooks"| P7R[Continue]
        P8[SUMMARIZE_OUTPUT] --> |"LLM summarize if >1000 chars"| P8R[Continue]
        P9[VALIDATE_OUTPUT] --> |"validate against output_schema"| P9R[Continue]
    end

    style F1R fill:#3a2a0a,stroke:#fbbf24,color:#fff
    style F2R fill:#22d3ee,stroke:#0891b2,color:#000
    style EXEC fill:#0e4429,stroke:#4ade80,color:#fff
    style P1R fill:#5c1a1a,stroke:#f87171,color:#fff
```

## Data Flow Between Nodes

```mermaid
graph LR
    subgraph "Node A"
        A_OUT["output_key='findings'"]
    end

    subgraph "state.context"
        CTX["findings: '...data...'<br/>A_output: '...raw...'"]
    end

    subgraph "Node B (reads findings)"
        B_IN["input_keys=['findings']"]
    end

    subgraph "Node C (inherits A)"
        C_IN["inherit_context_from='A'"]
    end

    A_OUT --> CTX
    CTX --> B_IN
    CTX --> C_IN
```

Three ways nodes pass data:

| Method | How it works | Use when |
|--------|-------------|----------|
| `output_key` → `input_keys` | Node A writes to `state.context[output_key]`. Node B reads from `state.context[input_keys[i]]`. | Structured data with named fields |
| `inherit_context_from` | Node B gets `state.context["{A_name}_output"]` auto-injected into its prompt. | One node directly continues another's work |
| `state.context` direct | Node's execute() reads/writes `state.context` directly. | Custom state management in BaseNode subclasses |

## CallerContext → MCP Server Identity Flow

```mermaid
sequenceDiagram
    participant App as Your App
    participant Agent as PromptiseAgent
    participant CV as ContextVar
    participant Client as MCPClient
    participant Transport as HTTP Transport
    participant Auth as AuthMiddleware
    participant Guard as HasRole Guard
    participant Handler as Tool Handler

    App->>Agent: ainvoke(input, caller=CallerContext(bearer_token="eyJ..."))
    Agent->>CV: Store caller in async contextvar

    Note over Agent: Reasoning Engine executes graph

    Agent->>Client: Tool call → MCPClient(bearer_token=caller.bearer_token)
    Client->>Transport: Authorization: Bearer eyJ...
    Transport->>Auth: Extract token from headers
    Auth->>Auth: Verify JWT signature (HMAC-SHA256)
    Auth->>Auth: Extract roles, scopes, claims
    Auth->>Auth: Build ClientContext
    Auth->>Guard: Check HasRole("analyst")
    Guard-->>Auth: Pass ✓
    Auth->>Handler: ctx.client = ClientContext(roles, scopes, claims)
    Handler-->>Transport: Result
    Transport-->>Client: Response
    Client-->>Agent: Tool result
    Agent->>CV: Reset contextvar
    Agent-->>App: Final response
```

## Hook Execution Points

```mermaid
graph TD
    subgraph "Per-node execution"
        H1["pre_node hooks (state modifiable)"] --> FLAGS_PRE["Pre-execute flags"]
        FLAGS_PRE --> EXEC["node.execute()"]
        EXEC --> FLAGS_POST["Post-execute flags"]
        FLAGS_POST --> H2["post_node hooks (result modifiable)"]
    end

    subgraph "Per-tool execution (inside PromptNode)"
        H3["pre_tool hooks (args modifiable)"] --> TOOL["tool.ainvoke()"]
        TOOL --> H4["post_tool hooks (result modifiable)"]
    end

    subgraph "Special hooks"
        H5["on_observable_event (OBSERVABLE nodes)"]
        H6["on_human_required (REQUIRES_HUMAN nodes)"]
        H7["on_graph_mutation (runtime mutations)"]
    end

    style H1 fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style H2 fill:#1e3a5f,stroke:#60a5fa,color:#fff
    style EXEC fill:#0e4429,stroke:#4ade80,color:#fff
    style TOOL fill:#2d1b4e,stroke:#c084fc,color:#fff
```

All hooks are wrapped in try/except — a failing hook never crashes the graph. Hooks run in registration order.

## Safety Mechanisms

```mermaid
graph TD
    subgraph "Iteration Guards"
        S1["max_iterations (default 50)"] --> |exceeded| S1R["Force graph end"]
        S2["max_node_iterations (default 25)"] --> |exceeded| S2R["_handle_stuck_node()"]
        S2R --> |"error transition exists"| S2A["Follow error edge"]
        S2R --> |"__error__ node exists"| S2B["Go to error handler"]
        S2R --> |"nothing"| S2C["Force __end__"]
    end

    subgraph "Budget & Health"
        S3["BudgetHook.max_tokens"] --> |exceeded| S3R["Set current_node = __end__"]
        S4["CycleDetectionHook"] --> |"pattern repeats N times"| S4R["Force __end__"]
        S5["TimingHook"] --> |"node exceeds budget"| S5R["Set error on result"]
    end

    subgraph "Error Recovery"
        S6["CRITICAL flag + error"] --> S6R["Abort immediately, report.error set"]
        S7["RETRYABLE flag + error"] --> S7R["Retry up to max_iterations times"]
        S7R --> |"backoff"| S7B["0.5s, 1s, 2s, 4s, 8s (capped)"]
        S8["SKIP_ON_ERROR"] --> S8R["Skip node, continue graph"]
    end

    subgraph "Graph Isolation"
        S9["graph.copy() per invocation"] --> S9R["Mutations don't affect original"]
        S10["config = dict(config)"] --> S10R["Concurrent calls don't interfere"]
        S11["max_mutations_per_run (default 10)"] --> S11R["Cap LLM graph modifications"]
    end

    style S6R fill:#5c1a1a,stroke:#f87171,color:#fff
    style S7B fill:#3a2a0a,stroke:#fbbf24,color:#fff
    style S9R fill:#0e4429,stroke:#4ade80,color:#fff
```

## Performance Architecture

```
Measurement: 5-node pipeline × 100 invocations

Per-node overhead:     0.009ms (9 microseconds)
Per-invocation:        0.04ms
PromptNode (mock LLM): 0.02ms

Where time goes:
├── Graph copy:           ~0.001ms (once per invocation)
├── Edge lookup:          ~0.0001ms (O(1) adjacency index)
├── Flag processing:      ~0.001ms (conditional checks)
├── Hook execution:       ~0.001ms (if hooks registered)
├── System prompt cache:  ~0.0001ms (dict lookup on re-entry)
├── Tool map cache:       ~0.0001ms (dict lookup)
└── LLM call:             99.99% of wall clock time
```

The engine is not the bottleneck. The LLM provider is.
