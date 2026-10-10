# Model Fallback

Automatic failover across multiple LLM providers. If the primary model fails, the next one in the chain handles the request — no downtime, no manual switching.

```python
from promptise import build_agent, FallbackChain

agent = await build_agent(
    model=FallbackChain([
        "openai:gpt-5-mini",           # Primary
        "anthropic:claude-sonnet-4-20250514",    # Fallback 1
        "ollama:llama3",                # Fallback 2 (local, always up)
    ]),
    servers=servers,
)
# If OpenAI is down → Claude handles it. If both down → local Llama.
```

---

## Why

Single-provider agents are a single point of failure. When OpenAI has an outage at 3am, your agent goes down with it. With a fallback chain, the next provider picks up seamlessly. With 3 providers at 99.9% uptime each, compound availability reaches 99.9999%.

---

## How It Works

`FallbackChain` is a `BaseChatModel` — LangChain and Promptise treat it like any other model. Under the hood:

1. Primary model receives the request
2. If it fails (error, timeout, rate limit), the next model is tried
3. Each model has an independent **circuit breaker** — after N consecutive failures, the model is skipped entirely for a recovery period (a request the provider rejects as invalid does not count; see [Which Errors Fall Back](#which-errors-fall-back))
4. When the recovery period elapses, one test request is sent (half-open state)
5. If the test succeeds, the circuit closes and the model resumes normal traffic

---

## Which Errors Fall Back

Every exception from a model moves on to the next one: timeouts (`timeout_per_model`), connection errors, rate limits, server errors, authentication errors, and also requests the provider rejects. What differs is whether the failure counts toward the model's circuit breaker:

| Error | Next model tried | Counts toward the circuit breaker |
|-------|------------------|-----------------------------------|
| Timeout, connection error, 429, 5xx, 401/403, anything without a status | Yes | Yes |
| The request was rejected: HTTP 400, 413, 422 (malformed, too large, context window exceeded) | Yes | **No** |

A rejected request says nothing about the provider's health, so one oversized prompt cannot open the primary's circuit and push every other user to the fallback for `recovery_timeout` seconds. The status is read from the exception's `status_code` (OpenAI and Anthropic SDK errors) or `response.status_code` (`httpx`); `promptise.fallback.is_request_error(exc)` shows the verdict.

When every model fails, `FallbackChain` raises `RuntimeError` listing each model's error, with the last one as its `__cause__`.

---

## Tools

`build_agent()` binds the agent's tools with `bind_tools()`, and `FallbackChain.bind_tools()` binds them to **every** model in the chain, so a fallback model can call tools too. Every model must support tool calling; one that doesn't is named in the `NotImplementedError`. The bound chain shares the original chain's circuit breakers, and `chain.model_name` / `chain.get_chain_status()` reflect the requests it served.

Each answer records the model that wrote it in `AIMessage.response_metadata["fallback_model"]`.

---

## Streaming

`FallbackChain` streams: `astream()`, `astream_events()` and `agent.astream_with_tools()` get the answer token by token from whichever model serves it.

- A model that fails before its first chunk is skipped for the next, as with `ainvoke()`. `timeout_per_model` and `global_timeout` bound the wait for that first chunk.
- Once a model has sent a chunk, the answer is committed to it. If its stream then fails, the error is raised (and counted toward its circuit breaker): the chunks already delivered cannot be taken back, so the chain does not splice another model's answer onto them.
- A model in the chain that cannot stream answers in one chunk.

---

## With the Semantic Cache

A [`SemanticCache`](cache.md) keys answers by the model that wrote them: an answer from a fallback model is stored under that model's id, and lookups use the first model whose circuit is not open. A fallback's answer is never served as the primary's.

---

## Configuration

```python
FallbackChain(
    models=["openai:gpt-5-mini", "anthropic:claude-sonnet-4-20250514", "ollama:llama3"],
    timeout_per_model=15.0,   # 15s max per model attempt (0 = no limit)
    global_timeout=30.0,      # 30s max across ALL attempts (0 = no limit)
    failure_threshold=3,      # Circuit opens after 3 consecutive failures
    recovery_timeout=60.0,    # 60s before testing a tripped circuit
    on_fallback=my_callback,  # Optional: called on each fallback activation
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `models` | `list[str \| BaseChatModel]` | **required** | Ordered list of models. First is primary. |
| `timeout_per_model` | `float` | `0` | Max seconds per model attempt. `0` = provider default. |
| `global_timeout` | `float` | `0` | Max seconds across all attempts combined. |
| `failure_threshold` | `int` | `3` | Consecutive failures before circuit opens. |
| `recovery_timeout` | `float` | `60.0` | Seconds before a tripped circuit allows a test. |
| `on_fallback` | `callable \| None` | `None` | `(primary_id, fallback_id, error)` callback. |

---

## Circuit Breaker States

| State | Behavior |
|-------|----------|
| **Closed** | Normal operation. Requests go to this model. |
| **Open** | Model is skipped entirely. Too many recent failures. |
| **Half-open** | Recovery period elapsed. One test request is allowed. If it succeeds → closed. If it fails → open again. |

---

## Monitoring

```python
chain = FallbackChain([...])
status = chain.get_chain_status()
# [
#     {"model_id": "openai:gpt-5-mini", "state": "closed", "failures": 0, "is_primary": True},
#     {"model_id": "claude-sonnet", "state": "open", "failures": 5, "is_primary": False},
# ]

chain.active_model  # "openai:gpt-5-mini" (first non-skipped model)
```

---

## Fallback Notifications

Combine with the [Events](events.md) system to get notified when fallbacks activate:

```python
from promptise import FallbackChain, EventNotifier, WebhookSink

def on_fallback(primary, fallback, error):
    print(f"Switched from {primary} to {fallback}: {error}")

agent = await build_agent(
    model=FallbackChain(
        ["openai:gpt-5-mini", "anthropic:claude-sonnet-4-20250514"],
        on_fallback=on_fallback,
    ),
    events=EventNotifier(sinks=[
        WebhookSink("https://hooks.slack.com/...", events=["invocation.error"]),
    ]),
    servers=servers,
)
```

---

## What's Next?

- [Events](events.md) — get notified when models fail or fallbacks activate
- [Observability](observability.md) — track which model served each request
- [Building Agents](agents/building-agents.md) — full `build_agent()` parameter reference
