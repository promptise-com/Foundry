# Autonomy Budget

Explicit, configured limits on what an agent can do autonomously — tool calls, LLM turns, cost, and irreversible actions. The agent is aware of its budget and the runtime enforces limits with configurable actions.

## When you need it

Without a budget, an autonomous agent can loop indefinitely, call expensive APIs without limit, or take irreversible actions at 3am with no guardrails. The autonomy budget defines the envelope within which the agent operates freely. Anything outside that envelope requires human approval.

## Configuration

```python
from promptise.runtime import ProcessConfig, BudgetConfig, EscalationTarget, ToolCostAnnotation

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Process customer support tickets.",
    budget=BudgetConfig(
        enabled=True,
        max_tool_calls_per_run=20,
        max_llm_turns_per_run=10,
        max_cost_per_run=25.0,
        max_tool_calls_per_day=500,
        max_cost_per_day=100.0,
        tool_costs={
            "stripe_charge": ToolCostAnnotation(cost_weight=5.0, irreversible=True),
            "send_email": ToolCostAnnotation(cost_weight=2.0, irreversible=True),
            "search": ToolCostAnnotation(cost_weight=0.5),
        },
        on_exceeded="escalate",  # "pause", "stop", or "escalate"
        inject_remaining=True,    # Show budget in agent context
        escalation=EscalationTarget(
            webhook_url="https://hooks.slack.com/...",
        ),
    ),
)
```

Or in a `.agent` manifest:

```yaml
name: support-agent
model: openai:gpt-5-mini
budget:
  enabled: true
  max_tool_calls_per_run: 20
  max_cost_per_day: 100.0
  on_exceeded: escalate
  inject_remaining: true
  tool_costs:
    stripe_charge:
      cost_weight: 5.0
      irreversible: true
    search:
      cost_weight: 0.5
  escalation:
    webhook_url: "https://hooks.slack.com/..."
```

## How it works

### Per-run and daily tracking

The budget system tracks two scopes:

- **Per-run**: Resets at the start of every invocation. Limits how much the agent can do in a single trigger-driven execution.
- **Per-day**: Resets at `daily_reset_hour_utc` (default: midnight UTC). Limits total daily activity.

---

## Understanding cost_weight

!!! danger "Critical: cost_weight is NOT real money"
    Budget cost is measured in **abstract weight units that you define**, not dollars. A `cost_weight=5.0` means "5 budget units" — it does NOT mean $5. The framework does not connect to any LLM provider's pricing API, does not know your token rates, and does not track actual monetary spend.

    **You** are responsible for defining what the numbers mean and keeping them aligned with reality.

### What the budget tracks and what it does NOT

| Tracked | Not tracked |
|---------|-------------|
| Tool calls (counted and weighted by `cost_weight`) | LLM API costs (token pricing from OpenAI, Anthropic, etc.) |
| LLM turns (counted, but with zero cost) | Actual dollar spend on any provider |
| Irreversible actions (counted) | Infrastructure costs (compute, network, storage) |
| Abstract cost units (sum of `cost_weight` values) | Real-time exchange rates or dynamic pricing |

**This means:** An agent with `max_cost_per_day=100.0` and zero tool cost annotations can make unlimited LLM API calls consuming thousands of dollars in tokens — the budget system will not stop it, because LLM turns have no cost weight.

The budget system controls **what the agent does** (which tools it calls, how many times), not **how much it costs the provider**.

### Default behavior for unannotated tools

Any tool without a `ToolCostAnnotation` entry defaults to `cost_weight=1.0`. This means:

```python
# These two are identical:
tool_costs={"search": ToolCostAnnotation(cost_weight=1.0)}
# vs. not listing "search" at all (it defaults to 1.0)
```

If you want a tool to be "free" (not consume budget), set `cost_weight` to a very small value:

```python
tool_costs={"health_check": ToolCostAnnotation(cost_weight=0.01)}
```

---

## Designing your cost scale

### Approach 1: Relative risk weighting (recommended)

Assign weights based on how expensive or risky the tool is **relative to a baseline read operation** (weight 1.0). This is the simplest and most robust approach — you don't need to know actual dollar costs.

```python
tool_costs={
    # Baseline: internal reads — cheap, low risk
    "search_docs": ToolCostAnnotation(cost_weight=0.5),
    "get_user": ToolCostAnnotation(cost_weight=1.0),       # 1.0 = baseline

    # Write operations — moderate risk
    "update_record": ToolCostAnnotation(cost_weight=2.0),
    "send_email": ToolCostAnnotation(cost_weight=3.0, irreversible=True),

    # External API calls — expensive, high risk
    "stripe_charge": ToolCostAnnotation(cost_weight=10.0, irreversible=True),
    "deploy_production": ToolCostAnnotation(cost_weight=20.0, irreversible=True),
}
```

**How to set limits with this approach:**

Think in terms of operations, not dollars:

- `max_cost_per_run=50.0` → the agent can do 50 searches, or 5 Stripe charges, or 25 email sends, or any mix
- `max_cost_per_day=500.0` → daily limit of ~500 baseline operations

This approach works because you're limiting **agent behavior**, not monetary spend. You don't need to update weights when provider prices change.

### Approach 2: Dollar-based approximation

If you want cost units to roughly map to real money, assign weights based on the actual cost you incur per tool call:

```python
tool_costs={
    # ~$0.01 per call (internal DB query, negligible infrastructure cost)
    "search": ToolCostAnnotation(cost_weight=0.01),

    # ~$0.10 per call (third-party geocoding API with per-request pricing)
    "geocode_address": ToolCostAnnotation(cost_weight=0.10),

    # ~$2.50 per call (Stripe processing fee)
    "stripe_charge": ToolCostAnnotation(cost_weight=2.50, irreversible=True),

    # ~$0.50 per call (email delivery service fee)
    "send_email": ToolCostAnnotation(cost_weight=0.50, irreversible=True),
}
# max_cost_per_day=50.0 → approximately $50/day in tool-related costs
```

!!! warning "Dollar-based weights require manual maintenance"
    The framework does NOT validate these weights against real pricing. If Stripe raises their fee to $3.00, you must manually update `cost_weight=3.00`. If your email provider changes pricing tiers, the weights become inaccurate silently.

    **Dollar-based weights also do NOT include LLM API costs.** An agent making 100 GPT-5 calls at $0.015/1K tokens could spend $15 in LLM costs alone — none of this is tracked by the budget.

### Approach 3: Combined with external cost tracking

For production systems that need real monetary limits, combine the budget system with external cost tracking:

```python
from promptise.runtime import ProcessConfig, BudgetConfig, ToolCostAnnotation

config = ProcessConfig(
    model="openai:gpt-5-mini",
    budget=BudgetConfig(
        enabled=True,

        # Use relative weights for tool budget
        max_cost_per_day=500.0,
        tool_costs={
            "stripe_charge": ToolCostAnnotation(cost_weight=10.0, irreversible=True),
            "search": ToolCostAnnotation(cost_weight=0.5),
        },

        # Use quantity limits for LLM cost control
        max_llm_turns_per_run=15,     # Cap LLM turns per invocation
        max_tool_calls_per_day=1000,  # Cap total daily tool calls

        on_exceeded="pause",
    ),
)
```

For actual dollar-level cost control, track LLM spending through:

- **Your LLM provider's dashboard** (OpenAI usage page, Anthropic console, etc.)
- **OpenAI/Anthropic API usage headers** (`x-ratelimit-remaining-tokens`, usage metadata)
- **Promptise observability** — token counts are tracked per invocation via `get_stats()` and can be exported to Prometheus/Grafana for alerting on real spend

The budget system handles **behavioral limits** (what the agent is allowed to do). External systems handle **monetary limits** (how much the infrastructure costs).

---

## Context injection

When `inject_remaining=True`, the agent sees remaining budget in its system prompt before every invocation:

```
[Budget Remaining] {"tool_calls_this_run": 15, "cost_this_run": 18.5, "tool_calls_today": 340}
```

This lets the agent prioritize — if it's running low on budget, it can choose cheaper tools or skip non-essential operations. In open mode, the agent can also call `check_budget` to inspect remaining limits programmatically.

---

## Per-call enforcement

Tool-call, cost and irreversible-action limits are checked **before each tool call runs**. A call that would push a counter past its limit (or any call once `max_llm_turns_per_run` is exceeded) is refused: the tool body never executes, the call is not counted, and the agent receives a `BudgetExceededError` as the tool's result, telling it to stop calling tools and answer. Parallel tool calls in one turn are checked one by one, so a limit of 2 lets exactly 2 of 3 parallel calls through.

```python
from promptise.runtime import BudgetExceededError  # error the agent sees as the tool result
```

After the invocation finishes, the first violation of the run triggers the action configured in `on_exceeded` (below). `max_runs_per_day` is checked when an invocation starts; a run over that limit does not start.

!!! note "Changed in 1.3.0"
    Earlier versions counted tool calls and applied `on_exceeded` only after the invocation had finished, so an agent could exceed `max_irreversible_per_run` (or any per-run limit) within a single run.

---

## Enforcement actions

When a limit is exceeded (after the invocation in which it happened):

| Action | Behavior |
|--------|----------|
| `"pause"` | Suspend the process. Can be resumed by an operator or the runtime. |
| `"stop"` | Stop the process entirely. Requires manual restart. |
| `"escalate"` | Fire escalation notification (webhook + EventBus), then suspend. |

Escalation sends a POST to the configured webhook URL with violation details:

```json
{
  "type": "budget_exceeded",
  "process_name": "support-agent",
  "violation": "max_cost_per_day exceeded (102.5 / 100.0)",
  "timestamp": "2026-03-23T14:30:00Z"
}
```

---

## Irreversible action tracking

Tools marked `irreversible=True` are tracked separately. Use `max_irreversible_per_run` to limit how many destructive actions the agent can take per invocation:

```python
budget=BudgetConfig(
    max_irreversible_per_run=2,  # Max 2 irreversible actions per invocation
    tool_costs={
        "delete_record": ToolCostAnnotation(cost_weight=3.0, irreversible=True),
        "send_email": ToolCostAnnotation(cost_weight=1.0, irreversible=True),
        "search": ToolCostAnnotation(cost_weight=0.5),  # Not irreversible
    },
)
```

The agent can call `search` as many times as the cost budget allows, but it can only call `delete_record` and/or `send_email` a combined total of 2 times per invocation.

---

## API reference

### `BudgetConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | `bool` | `False` | Enable budget tracking |
| `max_tool_calls_per_run` | `int \| None` | `None` | Per-invocation tool call limit |
| `max_llm_turns_per_run` | `int \| None` | `None` | Per-invocation LLM turn limit |
| `max_cost_per_run` | `float \| None` | `None` | Per-invocation cost limit (abstract units) |
| `max_irreversible_per_run` | `int \| None` | `None` | Per-invocation irreversible action limit |
| `max_tool_calls_per_day` | `int \| None` | `None` | Daily tool call limit |
| `max_runs_per_day` | `int \| None` | `None` | Daily invocation limit |
| `max_cost_per_day` | `float \| None` | `None` | Daily cost limit (abstract units) |
| `tool_costs` | `dict[str, ToolCostAnnotation]` | `{}` | Per-tool cost annotations |
| `on_exceeded` | `str` | `"pause"` | Action on violation: `"pause"`, `"stop"`, or `"escalate"` |
| `inject_remaining` | `bool` | `True` | Show remaining budget in agent context |
| `daily_reset_hour_utc` | `int` | `0` | Hour (0-23 UTC) when daily counters reset |
| `escalation` | `EscalationTarget \| None` | `None` | Where to send violation notifications |

### `ToolCostAnnotation`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `cost_weight` | `float` | `1.0` | Abstract cost units consumed per call. Must be > 0. |
| `irreversible` | `bool` | `False` | Whether this tool performs an irreversible action (delete, send, charge). |

### `BudgetState`

| Method | Description |
|--------|-------------|
| `.record_tool_call(tool_name, *, enforce=False)` | Record a tool call and add its `cost_weight` to run and daily totals. Returns a `BudgetViolation` if any limit is exceeded, or `None`. With `enforce=True` the call is checked first: if it would exceed a limit it is **not** counted and the violation has `blocked=True` (the caller must not run the tool). |
| `.record_tool_call_sync(tool_name, *, enforce=False)` | Thread-safe synchronous form, used by the runtime's budget guard (synchronous tools run on worker threads). |
| `.record_llm_turn()` | Record an LLM turn (counts toward `max_llm_turns_per_run` only, no cost). Returns a `BudgetViolation` if limit exceeded. |
| `.remaining()` | Dict of all remaining limits (tool calls, cost, LLM turns, irreversible actions). |
| `.reset_run()` | Reset per-run counters. Called automatically at the start of each invocation. |

### `BudgetEnforcer`

| Method | Description |
|--------|-------------|
| `.handle_violation(violation, process)` | Execute the configured enforcement action (pause, stop, or escalate). |
