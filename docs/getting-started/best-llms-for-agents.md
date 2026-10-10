---
title: Best LLMs for AI Agents — Model picks for agentic Python use cases
description: A practical, opinionated guide to choosing the right LLM for your AI agent in 2026. Covers OpenAI GPT-5, Claude Sonnet, Llama, Mistral, and open-source options — with real-world ratings and which model fits tool use, reasoning, and long-context jobs.
keywords: best LLM for AI agents, best model for agents, GPT-5 vs Claude, agent LLM comparison, open source LLM agents, tool calling LLM, Claude Sonnet agent
---

# Best LLMs for Agentic Use Cases

A practical, opinionated guide to choosing the right model for your Promptise agent. Covers proprietary and open-source options, real-world ratings, and which model fits which job.

!!! info "Last updated: April 2026"
    The agentic LLM landscape moves fast. This page reflects the state of available models as of April 2026. We update it as new models ship.

---

## What makes a good agentic model?

Picking a chat model for an agent is different from picking one for a chatbot. An agent needs to:

1. **Call tools reliably** — invoke functions with the right arguments, in the right order, without hallucinating signatures
2. **Reason across steps** — chain multiple tool calls toward a goal, recover from errors, decide when to stop
3. **Follow instructions precisely** — respect role definitions, system prompts, output schemas
4. **Handle structured output** — return valid JSON when asked, match Pydantic schemas
5. **Stay within budget** — call only the tools that matter, not 40 in a row

Models that score well on chat benchmarks (MMLU, HellaSwag) often **fail** on agentic ones. The benchmarks that matter for agents are:

| Benchmark | What it measures |
|-----------|------------------|
| **Berkeley Function Calling Leaderboard (BFCL)** | Tool-call accuracy, parameter extraction, multi-step composition |
| **τ²-Bench** | Realistic multi-turn agent tasks across customer service domains |
| **Terminal-Bench** | Real terminal/coding agent tasks end-to-end |
| **IFBench** | Instruction-following reliability under structured constraints |
| **ToolComp** | Composing dependent tool calls toward a goal |

---

## Quick recommendations

If you don't want to read the whole page, here's where to start:

| Your situation | Use this model |
|----------------|----------------|
| **Just starting, want it to work** | `openai:gpt-5-mini` — best balance of cost, speed, reliability |
| **Production agent, need maximum reliability** | `anthropic:claude-sonnet-4-6` — best tool calling in 2026 |
| **Maximum reasoning quality, cost no object** | `anthropic:claude-opus-4-6` or `openai:gpt-5` |
| **Multi-million token context** | `google:gemini-3-pro` — 2M token window |
| **Self-hosted, no data leaves your infra** | `ollama:qwen3-coder` or `ollama:deepseek-v3.2` |
| **Local laptop dev** | `ollama:phi-4-mini` or `ollama:qwen3-coder-30b-a3b` |
| **Hybrid: cheap for simple steps, strong for hard ones** | Use Promptise's per-node `model_override` in the Reasoning Engine |

---

## Proprietary models

### Anthropic Claude 4.6 family

The current state-of-the-art for agentic workloads. Claude 4.6 leads tool-calling benchmarks and is the most reliable choice for production agents that orchestrate complex multi-tool workflows.

| Model | Provider String | Input $/M | Output $/M | Speed | Context | Best For |
|-------|----------------|-----------|------------|-------|---------|----------|
| **Claude Opus 4.6** | `anthropic:claude-opus-4-6` | $15 | $75 | 20-30 t/s | 1M | Maximum quality, hardest tasks |
| **Claude Sonnet 4.6** | `anthropic:claude-sonnet-4-6` | $3 | $15 | 40-60 t/s | 1M | Production agents, best value |
| **Claude Haiku 4.5** | `anthropic:claude-haiku-4-5` | $1 | $5 | 80+ t/s | 200K | Fast, simple agent tasks |

| Rating | Score |
|--------|-------|
| Tool calling reliability | ★★★★★ |
| Reasoning quality | ★★★★★ (Opus) / ★★★★☆ (Sonnet) |
| Speed | ★★★☆☆ (Opus) / ★★★★☆ (Sonnet) / ★★★★★ (Haiku) |
| Cost efficiency | ★★☆☆☆ (Opus) / ★★★★☆ (Sonnet) / ★★★★★ (Haiku) |
| Setup complexity | ★☆☆☆☆ (just an API key) |

**When to use each:**

- **Opus 4.6** — Long-horizon agents, complex multi-agent orchestration, agents that need to debug their own mistakes. Use when getting the right answer matters more than the cost per call.
- **Sonnet 4.6** — The default choice for production. Near-Opus quality at 5x lower cost. Best tool-calling/cost ratio in the market right now.
- **Haiku 4.5** — Fast classification, routing, simple tool calls. Pair it with Sonnet via the Reasoning Engine for cost-optimized pipelines.

```python
from promptise import build_agent

agent = await build_agent(
    model="anthropic:claude-sonnet-4-6",
    servers={"tools": HTTPServerSpec(url="http://localhost:8080/mcp")},
    instructions="You are a senior analyst.",
)
```

!!! info "You need an API key first"
    Every model provider needs a credential. Put it in a `.env` file in your project — Promptise loads it automatically, in scripts and in the CLI — or pass it in code with `Model(...)` — explained in [**Models & Providers → In code: Model**](../core/agents/models.md#in-code-model): `Model("gpt-4o-mini", provider="openai", api_key="...")`. The **exact variable name for every provider** (Azure AI Foundry, Bedrock, Gemini, Ollama, …) is in [**Configuration & Secrets → Every provider's variables**](../getting-started/configuration.md#every-providers-variables); self-hosted and inference endpoints are in [**Models & Providers → Custom, self-hosted and inference endpoints**](../core/agents/models.md#custom-self-hosted-and-inference-endpoints). `promptise models env <provider>` prints the lines a provider needs and `promptise models check <model>` tells you what is still missing.

---

### OpenAI GPT-5 family

GPT-5.2 currently leads several agentic benchmarks alongside Claude. GPT-5-mini is the best entry-level model for cost-conscious agentic work.

| Model | Provider String | Input $/M | Output $/M | Speed | Context | Best For |
|-------|----------------|-----------|------------|-------|---------|----------|
| **GPT-5** | `openai:gpt-5` | $5 | $20 | 30-50 t/s | 1M | Top-tier reasoning, parity with Claude Opus |
| **GPT-5-mini** | `openai:gpt-5-mini` | $1.25 | $10 | 60-80 t/s | 400K | Best entry point — fast, cheap, capable |
| **GPT-5-nano** | `openai:gpt-5-nano` | $0.30 | $1.20 | 100+ t/s | 128K | High-volume, simple tasks |

| Rating | Score |
|--------|-------|
| Tool calling reliability | ★★★★★ |
| Reasoning quality | ★★★★★ (GPT-5) / ★★★★☆ (mini) |
| Speed | ★★★★☆ |
| Cost efficiency | ★★★★☆ (mini is exceptional value) |
| Setup complexity | ★☆☆☆☆ |

**When to use each:**

- **GPT-5** — When you need the absolute best on coding and reasoning. Strong native function calling. Excellent for complex agentic workflows.
- **GPT-5-mini** — Promptise's recommended default for getting started. Great tool calling, fast, cheap enough to iterate freely.
- **GPT-5-nano** — Use for high-volume agent fleets where each call is simple (classification, routing, simple lookups).

```python
agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={"tools": HTTPServerSpec(url="...")},
)
```

---

### Google Gemini 3 family

Gemini 3 Pro's massive 2M token context window is the killer feature. If your agent needs to read entire codebases, full document libraries, or hours of conversation history, Gemini is the only option.

| Model | Provider String | Input $/M | Output $/M | Speed | Context | Best For |
|-------|----------------|-----------|------------|-------|---------|----------|
| **Gemini 3 Pro** | `google:gemini-3-pro` | $2 / $4 | $12 / $24 | 40-60 t/s | 2M | Massive context, multimodal |
| **Gemini 3 Flash** | `google:gemini-3-flash` | $0.30 | $2.50 | 100+ t/s | 1M | Fast, very cheap |

*Gemini 3 Pro pricing doubles beyond 200K tokens (input) and 1M tokens (output)*

| Rating | Score |
|--------|-------|
| Tool calling reliability | ★★★★☆ |
| Reasoning quality | ★★★★★ (Pro) |
| Speed | ★★★★☆ |
| Cost efficiency | ★★★★☆ (Flash is excellent value) |
| Setup complexity | ★★☆☆☆ |

**When to use:**

- **Gemini 3 Pro** — Document analysis agents, codebase navigation, long-running conversations where the full history matters.
- **Gemini 3 Flash** — High-throughput agent workers, real-time agents that need fast responses, multimodal tasks (image + text).

---

## Open-source models

The gap between proprietary and open-source has narrowed dramatically in 2026. For self-hosted production deployments, open-source models are now genuinely competitive.

### Qwen 3 family

**The current leader** in open-source agentic performance. Qwen3 models top the Berkeley Function Calling Leaderboard among open weights and ship under Apache 2.0 (commercially permissive).

| Model | Provider String | Parameters | Hardware | Best For |
|-------|----------------|------------|----------|----------|
| **Qwen3-Coder Next (80B MoE)** | `ollama:qwen3-coder` | 80B total / 3B active | 1× A100 80GB or 2× RTX 4090 | Coding agents, long-horizon reasoning |
| **Qwen3-Coder-30B-A3B** | `ollama:qwen3-coder-30b` | 30B total / 3B active | 1× RTX 4090 | Best laptop/workstation option |
| **Qwen3.5-397B-A17B** | `ollama:qwen3-397b` | 397B total / 17B active | 4× A100 80GB | Top-tier self-hosted reasoning |

| Rating | Score |
|--------|-------|
| Tool calling reliability | ★★★★★ (best open-source) |
| Reasoning quality | ★★★★☆ |
| Speed | ★★★★☆ (MoE = fast inference) |
| Cost efficiency | ★★★★★ (free + low compute via MoE) |
| Setup complexity | ★★★☆☆ (requires GPU + Ollama/vLLM) |

```python
agent = await build_agent(
    model="ollama:qwen3-coder",  # local Ollama
    servers={...},
)
```

---

### DeepSeek V3.2

The first open model to integrate "thinking mode" directly into tool use. Strong general reasoning, very good for autonomous agent workloads.

| Model | Provider String | Parameters | Hardware | Best For |
|-------|----------------|------------|----------|----------|
| **DeepSeek V3.2** | `ollama:deepseek-v3.2` | 671B total / 37B active | Multi-GPU cluster | Top open-source reasoning + tool use |

| Rating | Score |
|--------|-------|
| Tool calling reliability | ★★★★★ |
| Reasoning quality | ★★★★★ (rivals GPT-4 Turbo) |
| Speed | ★★★☆☆ |
| Cost efficiency | ★★★★☆ (free, but high compute) |
| Setup complexity | ★★★★☆ (large model, complex serving) |

---

### GLM 4.7 / GLM 5

The strongest open-source competitor to Claude on agentic terminal tasks. Hybrid reasoning modes (think/no-think) are unique among open models.

| Model | Provider String | Parameters | Best For |
|-------|----------------|------------|----------|
| **GLM-4.7 Flash** | `ollama:glm-4.7-flash` | 30B MoE | Lightweight agents, fast serving |
| **GLM-5** | `ollama:glm-5` | Larger MoE | Reasoning + coding + agent triple threat |

| Rating | Score |
|--------|-------|
| Tool calling reliability | ★★★★★ (90.6% on tool use benchmarks) |
| Reasoning quality | ★★★★★ |
| Speed | ★★★★☆ |
| Cost efficiency | ★★★★☆ |
| Setup complexity | ★★★☆☆ |

---

### Llama 4 family

Meta's open model. Strong general performance and a 10M token context window on Scout. Lower tool-calling reliability than Qwen/DeepSeek but the most ecosystem support.

| Model | Provider String | Parameters | Context | Best For |
|-------|----------------|------------|---------|----------|
| **Llama 4 Scout** | `ollama:llama4-scout` | 17B active | 10M | Massive context, document processing |
| **Llama 4 Maverick** | `ollama:llama4-maverick` | 400B total | 1M | General-purpose, multilingual |

| Rating | Score |
|--------|-------|
| Tool calling reliability | ★★★☆☆ (improving but not best-in-class) |
| Reasoning quality | ★★★★☆ |
| Speed | ★★★★☆ |
| Cost efficiency | ★★★★☆ |
| Setup complexity | ★★☆☆☆ (excellent ecosystem) |

---

### Mistral Large 3 / Medium 3.1

Apache 2.0, European, strong general capabilities. Mistral Medium 3.1 specifically is positioned as "Claude Sonnet quality at 8x lower cost" — true for many tasks.

| Model | Provider String | Parameters | Best For |
|-------|----------------|------------|----------|
| **Mistral Large 3** | `mistral:mistral-large-3` | 675B / 41B active | Top-tier API-based open model |
| **Mistral Medium 3.1** | `mistral:mistral-medium-3.1` | ~70B | Best price/performance in Mistral lineup |
| **Mistral Small 4** | `mistral:mistral-small-4` | ~24B | 25x cheaper than GPT-4o, capable |

| Rating | Score |
|--------|-------|
| Tool calling reliability | ★★★★☆ |
| Reasoning quality | ★★★★☆ |
| Speed | ★★★★☆ |
| Cost efficiency | ★★★★★ |
| Setup complexity | ★☆☆☆☆ (API-based) |

---

### Lightweight local models

For laptop development, edge deployments, or privacy-sensitive workflows. Don't expect frontier performance — but they support real function calling on consumer hardware.

| Model | Provider String | Parameters | Hardware | Best For |
|-------|----------------|------------|----------|----------|
| **Phi-4-Mini** | `ollama:phi-4-mini` | 14B | M2 Mac, RTX 3060 | Surprisingly strong tool calling on a laptop |
| **Falcon 3** | `ollama:falcon3` | 7B / 10B | Consumer GPU | Native function calling, fast |
| **Qwen3-Coder-30B-A3B** | `ollama:qwen3-coder-30b` | 30B / 3B active | M3 Max, RTX 4090 | Best laptop option for coding agents |

| Rating | Score |
|--------|-------|
| Tool calling reliability | ★★★☆☆ |
| Reasoning quality | ★★★☆☆ |
| Speed | ★★★★★ (small + local) |
| Cost efficiency | ★★★★★ (zero per-token cost) |
| Setup complexity | ★★☆☆☆ (Ollama install) |

```python
# Local development with Ollama
agent = await build_agent(
    model="ollama:phi-4-mini",
    servers={...},
)
```

---

## Cost vs. quality cheat sheet

Per-million-token costs as of April 2026, sorted by output cost:

| Model | Input $/M | Output $/M | Tier |
|-------|-----------|------------|------|
| Claude Opus 4.6 | $15 | $75 | Premium |
| GPT-5 | $5 | $20 | Premium |
| Claude Sonnet 4.6 | $3 | $15 | High |
| Gemini 3 Pro | $2 | $12 | High |
| GPT-5-mini | $1.25 | $10 | Mid |
| Claude Haiku 4.5 | $1 | $5 | Mid |
| Mistral Medium 3.1 | $0.40 | $2 | Low |
| Gemini 3 Flash | $0.30 | $2.50 | Low |
| GPT-5-nano | $0.30 | $1.20 | Low |
| Self-hosted (any open model) | $0 | $0 | Free + compute |

---

## Mixing models in one agent

This is where Promptise's Reasoning Engine shines. Use a cheap model for simple steps and an expensive one for hard reasoning — all in the same agent.

```python
from promptise import build_agent
from promptise.engine import PromptNode, ThinkNode, SynthesizeNode

agent = await build_agent(
    model="openai:gpt-5-mini",  # Default model for most nodes
    servers={"tools": HTTPServerSpec(url="...")},
    node_pool=[
        ThinkNode("think"),  # Uses default model
        PromptNode("research",
            inject_tools=True,
            model_override="anthropic:claude-haiku-4-5",  # Fast/cheap for tool calls
        ),
        PromptNode("deep_analysis",
            input_keys=["research_results"],
            model_override="anthropic:claude-opus-4-6",  # Strong model for hard reasoning
            max_tokens=8192,
        ),
        SynthesizeNode("final",
            model_override="openai:gpt-5-mini",  # Cheap for formatting
        ),
    ],
)
```

This pattern can cut LLM costs by 60-80% on complex agent workflows while maintaining quality where it matters.

---

## How to evaluate a model for your use case

Don't trust benchmarks alone. The best test is your own workflow on a realistic task. Here's a quick eval template:

```python
import asyncio
from promptise import build_agent
from promptise.config import HTTPServerSpec

async def eval_model(model_id: str, task: str) -> dict:
    agent = await build_agent(
        model=model_id,
        servers={"tools": HTTPServerSpec(url="http://localhost:8080/mcp")},
        instructions="You are a research assistant.",
    )

    import time
    start = time.time()
    result = await agent.ainvoke({"messages": [{"role": "user", "content": task}]})
    elapsed = time.time() - start

    tool_calls = [m for m in result["messages"] if getattr(m, "type", "") == "tool"]
    response = next(
        (m.content for m in reversed(result["messages"])
         if getattr(m, "type", "") == "ai" and m.content),
        ""
    )

    await agent.shutdown()
    return {
        "model": model_id,
        "time_seconds": elapsed,
        "tool_calls": len(tool_calls),
        "response_length": len(response),
        "response": response,
    }

# Test the same task across models
async def main():
    task = "Find the weather in Berlin, Tokyo, and NYC. Calculate the average temperature."
    for model in ["openai:gpt-5-mini", "anthropic:claude-sonnet-4-6", "google:gemini-3-flash"]:
        result = await eval_model(model, task)
        print(f"{model}: {result['time_seconds']:.1f}s, {result['tool_calls']} tools")

asyncio.run(main())
```

Run this with your real MCP servers. The model that uses fewer tool calls and produces a correct answer fastest is the right one for your workload.

---

## Tips and gotchas

!!! tip "Default to Claude Sonnet 4.6 for production"
    If you're shipping a real product, Claude Sonnet 4.6 is the safest choice in 2026. Best tool-calling reliability, near-Opus quality, predictable costs.

!!! tip "Default to GPT-5-mini for prototyping"
    For local development and iteration, GPT-5-mini is hard to beat. Fast, cheap, reliable enough to debug your agent logic without thinking about per-token costs.

!!! warning "Open-source isn't free"
    The model itself is free, but GPU hosting isn't. A 70B+ model needs a serious GPU (A100/H100). Factor in serving infrastructure, monitoring, and model updates before going self-hosted.

!!! warning "Tool calling support varies by model"
    Some open-source models support tool calling natively (Qwen, DeepSeek, GLM, Phi-4). Others need prompt engineering to fake it. Check the model's docs before assuming it works with `inject_tools=True`.

!!! tip "Use semantic tool optimization for big tool sets"
    If your agent has 30+ tools, enable `optimize_tools=True` in `build_agent()`. This uses local embeddings to send only the relevant tools per query, cutting input tokens by 40-70% regardless of model.

---

## Sources

- [Best Agentic AI Models January 2026](https://whatllm.org/blog/best-agentic-models-january-2026)
- [Best Open Source LLM For Agent Workflow 2026](https://www.siliconflow.com/articles/en/best-open-source-LLM-for-Agent-Workflow)
- [Scale Labs Tool Use Leaderboard](https://labs.scale.com/leaderboard/tool_use_enterprise)
- [Best Open-Source LLMs in 2026 (BentoML)](https://www.bentoml.com/blog/navigating-the-world-of-open-source-large-language-models)
- [Local LLMs Tool Calling Eval 2026](https://www.jdhodges.com/blog/local-llms-on-tool-calling-2026-pt1-local-lm/)
- [Best Open-Source LLMs for Agents 2026](https://fast.io/resources/top-open-source-llms-agents/)
- [Claude API Pricing 2026](https://platform.claude.com/docs/en/about-claude/pricing)
- [Gemini API Pricing 2026](https://ai.google.dev/gemini-api/docs/pricing)
- [Qwen 3.5 vs Llama vs Mistral 2026](https://www.aimagicx.com/blog/qwen-3-5-vs-llama-vs-mistral-china-open-source-ai-2026)
- [Open-Source AI Landscape April 2026](https://www.digitalapplied.com/blog/open-source-ai-landscape-april-2026-gemma-qwen-llama)
