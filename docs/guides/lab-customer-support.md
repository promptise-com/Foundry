# Lab: Customer Support Agent

Build a production customer support agent with multi-turn conversations, knowledge base integration, conversation phases, quality validation, and escalation rules. Complete, runnable code at every step.

## What You'll Build

An agent that:

- Greets the customer and classifies their issue
- Searches a knowledge base for relevant articles
- Drafts a response following company policy
- Validates the response against quality criteria
- Escalates to a human when the issue is too complex
- Persists conversation history across sessions

## Prerequisites

```bash
pip install promptise
export OPENAI_API_KEY=sk-...
```

## Step 1 — Build the Knowledge Base Server

Create an MCP server that the agent connects to for customer data and KB articles.

```python
# support_server.py
from promptise.mcp.server import MCPServer, MCPRouter

server = MCPServer("support-tools")
kb = MCPRouter(prefix="kb")
crm = MCPRouter(prefix="crm")

# ── Knowledge Base ──

ARTICLES = {
    "shipping": "Standard shipping: 5-7 business days. Express: 2-3 days. "
                "Tracking available 24h after dispatch. International: 10-14 days.",
    "returns": "30-day return policy. Item must be unused and in original packaging. "
               "Refund processed within 5 business days. Return shipping is free.",
    "billing": "Charges appear within 24h. Disputes: contact support within 60 days. "
               "Duplicate charges are auto-detected and refunded.",
    "account": "Password reset via email link. 2FA available in Settings > Security. "
               "Account deletion: contact support, processed within 72h.",
    "outage": "Check status.example.com for live system status. "
              "Compensation policy: 1 day credit per hour of downtime.",
}

@kb.tool()
async def search_articles(query: str) -> str:
    """Search the knowledge base for articles matching a topic."""
    results = []
    for topic, content in ARTICLES.items():
        if topic in query.lower() or any(w in content.lower() for w in query.lower().split()):
            results.append(f"[{topic.upper()}] {content}")
    return "\n\n".join(results) if results else "No articles found."

@kb.tool()
async def get_article(topic: str) -> str:
    """Get a specific knowledge base article by topic name."""
    return ARTICLES.get(topic.lower(), f"No article found for: {topic}")

# ── CRM ──

CUSTOMERS = {
    "C-1001": {"name": "Alice Chen", "plan": "pro", "since": "2024-01", "tickets": 3},
    "C-1002": {"name": "Bob Martinez", "plan": "enterprise", "since": "2023-06", "tickets": 12},
    "C-1003": {"name": "Carol Williams", "plan": "starter", "since": "2025-01", "tickets": 1},
}

ORDERS = {
    "ORD-5001": {"customer": "C-1001", "status": "shipped", "item": "Widget Pro", "tracking": "TRK-88421"},
    "ORD-5002": {"customer": "C-1002", "status": "processing", "item": "Enterprise Suite", "tracking": None},
    "ORD-5003": {"customer": "C-1001", "status": "delivered", "item": "Adapter Kit", "tracking": "TRK-77312"},
}

@crm.tool()
async def lookup_customer(customer_id: str) -> str:
    """Look up customer profile by ID."""
    c = CUSTOMERS.get(customer_id)
    if not c:
        return f"Customer {customer_id} not found."
    return f"Name: {c['name']}, Plan: {c['plan']}, Since: {c['since']}, Past tickets: {c['tickets']}"

@crm.tool()
async def lookup_order(order_id: str) -> str:
    """Look up order details by order ID."""
    o = ORDERS.get(order_id.upper())
    if not o:
        return f"Order {order_id} not found."
    tracking = o['tracking'] or "Not yet available"
    return f"Item: {o['item']}, Status: {o['status']}, Tracking: {tracking}"

@crm.tool()
async def create_ticket(customer_id: str, subject: str, priority: str = "medium") -> str:
    """Create a support ticket for escalation."""
    return f"Ticket created: #{len(CUSTOMERS)*100+1} for {customer_id} — '{subject}' [{priority}]"

server.include_router(kb)
server.include_router(crm)

if __name__ == "__main__":
    server.run(transport="stdio")
```

## Step 2 — Build the Agent with a Custom Reasoning Pattern

Instead of generic ReAct, use a specialized support pattern:
**Classify → Search KB → Draft → Validate → Respond**

```python
# support_agent.py
import asyncio
import sys
from promptise import build_agent
from promptise.config import StdioServerSpec
from promptise.engine import PromptGraph, PromptNode, NodeFlag
from promptise.engine.reasoning_nodes import (
    ThinkNode, PlanNode, ValidateNode, SynthesizeNode,
)

async def main():
    # Build the custom reasoning pattern
    graph = PromptGraph("support-agent", nodes=[
        # Step 1: Classify the issue
        ThinkNode("classify",
            is_entry=True,
            focus_areas=["issue type", "urgency", "customer sentiment"],
        ),

        # Step 2: Search KB + look up customer data
        PromptNode("investigate",
            instructions=(
                "You have the issue classification. Now:\n"
                "1. Search the knowledge base for relevant articles\n"
                "2. Look up the customer if they provided an ID\n"
                "3. Look up orders if they mentioned one\n"
                "Gather ALL relevant information before responding."
            ),
            inject_tools=True,
            flags={NodeFlag.RETRYABLE},
        ),

        # Step 3: Draft the response
        PlanNode("draft",
            instructions=(
                "Draft a customer support response. Rules:\n"
                "- Be empathetic and professional\n"
                "- Reference specific KB articles\n"
                "- Include order tracking numbers if relevant\n"
                "- Never promise what you can't deliver\n"
                "- If the issue needs escalation, say so explicitly"
            ),
        ),

        # Step 4: Validate against company policy
        ValidateNode("policy_check",
            criteria=[
                "Response is empathetic and professional",
                "Includes relevant KB article references",
                "Does not make false promises",
                "Includes next steps for the customer",
            ],
            on_pass="respond",
            on_fail="draft",  # Loop back to re-draft
        ),

        # Step 5: Final response
        SynthesizeNode("respond", is_terminal=True),
    ])

    # Build the agent
    agent = await build_agent(
        model="openai:gpt-4o-mini",
        servers={
            "support": StdioServerSpec(
                command=sys.executable,
                args=["support_server.py"],
            ),
        },
        agent_pattern=graph,
        instructions=(
            "You are a customer support agent for a SaaS company. "
            "Be helpful, empathetic, and precise. Always use your tools "
            "to look up real data — never guess customer information."
        ),
    )

    # Run a sample conversation
    result = await agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": (
                "Hi, I'm customer C-1001. My order ORD-5001 was supposed "
                "to arrive last week but I haven't received it. Can you help?"
            ),
        }]
    })

    # Print the response
    for msg in reversed(result["messages"]):
        if getattr(msg, "type", "") == "ai" and msg.content:
            print("\n=== Agent Response ===")
            print(msg.content)
            break

    await agent.shutdown()

asyncio.run(main())
```

## Step 3 — Add Conversation Persistence

Make the agent remember past interactions:

```python
from promptise.conversations import SQLiteConversationStore

agent = await build_agent(
    ...,
    conversation_store=SQLiteConversationStore("support.db"),
)

# Use chat() for persistent sessions
response = await agent.chat(
    session_id="session-alice-001",
    user_message="Hi, I'm customer C-1001. My order ORD-5001 is late.",
    caller=CallerContext(user_id="C-1001"),
)

# Second message — agent remembers the first
response = await agent.chat(
    session_id="session-alice-001",
    user_message="Can you check the tracking number?",
    caller=CallerContext(user_id="C-1001"),
)
```

## Step 4 — Add Escalation Rules

Use approval workflows for sensitive actions:

```python
from promptise.agent import ApprovalConfig

agent = await build_agent(
    ...,
    approval=ApprovalConfig(
        # Require human approval before creating tickets
        tools_requiring_approval=["crm_create_ticket"],
        handler="callback",
    ),
)
```

## Step 5 — Add Guardrails

Prevent the agent from leaking PII or making unauthorized promises:

```python
agent = await build_agent(
    ...,
    guardrails=True,  # PromptiseSecurityScanner.default(): injection + PII + credentials
)
```

The guardrails will:

- Block prompt injection attempts in customer messages
- Detect and redact PII in agent responses (credit card numbers, SSNs)
- Flag credential patterns
- Run content safety checks

## What You've Built

A production customer support agent with:

- **Custom reasoning pattern**: Classify → Investigate → Draft → Validate → Respond
- **Knowledge base integration**: Articles searched automatically
- **CRM lookup**: Customer and order data retrieved from tools
- **Quality validation**: Every response checked against company policy before sending
- **Conversation persistence**: SQLite-backed session history
- **Escalation rules**: Human approval required for ticket creation
- **Security guardrails**: PII detection, injection prevention

## Next Steps

- [Multi-User Systems Guide](multi-user-systems.md) — Add JWT auth and per-user isolation
- [Reasoning Patterns](../core/agents/reasoning-patterns.md) — Customize the reasoning flow
- [Prompt Engineering Guide](prompt-engineering.md) — Add ConversationFlow for phase-based prompts
