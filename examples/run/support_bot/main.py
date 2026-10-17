"""The support bot's program: answer the tickets in the inbox, or the one you pass.

A normal Python script::

    python main.py                                  # the three sample tickets
    python main.py "Where is my order A1002?"       # one ticket

``promptise run`` runs it the same way and adds the project's environment,
dev tools, the terminal UI or production behavior, with no changes here::

    promptise run --devtools
    promptise run main.py "Where is my order A1002?"
"""

from __future__ import annotations

import asyncio
import sys

from agent import TICKETS, create_agent

INBOX = [
    "Hi, where is my order A1002? It was supposed to arrive by now.",
    "How do I return a jacket that doesn't fit?",
    "My order A1003 arrived dented. I'd like a refund please. I'm lee@example.com.",
]


async def main() -> None:
    tickets = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else INBOX
    agent = await create_agent()
    try:
        for number, ticket in enumerate(tickets, 1):
            print(f"--- Ticket {number}: {ticket}")
            result = await agent.ainvoke({"messages": [{"role": "user", "content": ticket}]})
            print(result["messages"][-1].content)
            print()
    finally:
        await agent.shutdown()
    if TICKETS:
        print(f"Handed to a person: {', '.join(t['id'] for t in TICKETS)}")


if __name__ == "__main__":
    asyncio.run(main())
