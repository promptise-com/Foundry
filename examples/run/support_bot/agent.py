"""The support bot's agent: the factory and three local tools.

The tools read and write an in-memory shop, so the example needs no network
apart from the model itself.  ``main.py`` calls ``create_agent()``;
``promptise run`` calls ``create_agent(ctx)`` for ``--serve``, ``-i`` and chat
mode.
"""

from __future__ import annotations

import json
import os

from langchain_core.tools import tool
from promptise import PromptiseAgent, build_agent
from promptise.run import RunContext

DEFAULT_MODEL = "openai:gpt-5-mini"

INSTRUCTIONS = """\
You are the support agent of Trailhead, a small online outdoor shop.

- Before you say anything about an order, look it up with lookup_order.
- Answer policy questions (returns, refunds, shipping) from search_help_center,
  never from memory.
- Refunds and damaged items need a person: open a ticket with open_ticket and
  give the customer the ticket ID.
- Never invent order details. If an order ID is missing, ask for it.
- Be brief and friendly. Sign off as "Trailhead Support".
"""

ORDERS: dict[str, dict[str, str]] = {
    "A1001": {"item": "Rain jacket", "status": "delivered", "delivered_on": "2026-10-12"},
    "A1002": {
        "item": "Hiking boots",
        "status": "in transit",
        "carrier": "DHL",
        "expected_on": "2026-10-19",
    },
    "A1003": {"item": "Steel water bottle", "status": "delivered", "delivered_on": "2026-10-15"},
}

ARTICLES: dict[str, str] = {
    "Returns": (
        "Unworn items can be returned within 30 days of delivery. Start a return at "
        "trailhead.example.com/returns; the return label is free."
    ),
    "Refunds": (
        "Refunds go to the original payment method 3 to 5 business days after the "
        "returned item arrives. Damaged items are refunded without a return."
    ),
    "Shipping": (
        "Standard shipping takes 3 to 5 business days. The tracking link is in the "
        "shipping confirmation email."
    ),
}

TICKETS: list[dict[str, str]] = []
"""Tickets opened for a person (in memory; a real bot would call a help desk API)."""


@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by its ID, for example A1002. Returns the item, status and dates."""
    key = order_id.strip().upper()
    order = ORDERS.get(key)
    if order is None:
        return f"No order {order_id!r}. Order IDs look like A1002."
    return json.dumps({"order_id": key, **order})


@tool
def search_help_center(query: str) -> str:
    """Search the help center articles (returns, refunds, shipping). Returns the best matches."""
    words = [w.strip(".,?!").lower() for w in query.split() if len(w) > 3]
    scored = []
    for title, text in ARTICLES.items():
        haystack = f"{title} {text}".lower()
        score = sum(word in haystack for word in words)
        if score:
            scored.append((score, title, text))
    if not scored:
        return "No article matches. Try other words, or open a ticket for a person."
    scored.sort(reverse=True)
    return "\n".join(f"{title}: {text}" for _, title, text in scored[:2])


@tool
def open_ticket(customer_email: str, summary: str, priority: str = "normal") -> str:
    """Hand a case to a person on the support team. Use it for refunds, damaged items and
    anything you cannot solve. priority is low, normal or high."""
    ticket_id = f"T-{len(TICKETS) + 1:04d}"
    TICKETS.append(
        {"id": ticket_id, "email": customer_email, "summary": summary, "priority": priority}
    )
    return (
        f"Opened {ticket_id} ({priority}) for {customer_email}. "
        "A person replies within one business day."
    )


async def create_agent(ctx: RunContext | None = None) -> PromptiseAgent:
    """Build the support agent.

    The model is the ``--model`` override when ``promptise run`` passes one,
    else ``SUPPORT_BOT_MODEL`` (a default in ``promptise.toml``), else
    ``openai:gpt-5-mini``.
    """
    model = (
        (ctx.model if ctx is not None else None)
        or os.environ.get("SUPPORT_BOT_MODEL")
        or DEFAULT_MODEL
    )
    return await build_agent(
        model=model,
        instructions=INSTRUCTIONS,
        extra_tools=[lookup_order, search_help_center, open_ticket],
    )
