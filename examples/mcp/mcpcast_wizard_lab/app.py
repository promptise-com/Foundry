"""Helpdesk — the API you are about to turn into an MCP server with the guided setup.

An ordinary FastAPI service behind a bearer token: customers, support tickets,
a refund endpoint, an admin corner, and one deprecated route. Nothing in it
knows about MCP or Promptise. It exists so that every step of the wizard has
something real to show:

- **step 1** detects it on a local port and reads its ``/openapi.json``;
- **step 3** counts what each safety profile would generate from *these*
  operations — reads, writes, one financial call (``refund``) and two
  destructive ones (``DELETE``, ``purge``);
- **step 6** shows what the model made of the summaries and docstrings below.

Three things in this file decide how good the resulting tools are: ``summary=``
and the docstring become the tool description an LLM reads; ``operation_id=``
becomes the tool name; ``Field(description=...)`` and ``Query(description=...)``
become each parameter's description.

Run it on its own from the repo root with
``uvicorn --app-dir examples/mcp/mcpcast_wizard_lab app:app --port 8001``
(``uvicorn app:app --port 8001`` from this directory; 8001 is a port the
wizard's detection probes) and open ``http://127.0.0.1:8001/openapi.json`` —
that URL is all the wizard needs.
"""

from __future__ import annotations

import secrets
from datetime import date
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# The one token this demo accepts. A generated `--auth env-token` server presents
# it upstream as `Authorization: Bearer demo-token` on every call.
DEMO_TOKEN = "demo-token"

Status = Literal["open", "pending", "closed"]
Priority = Literal["low", "normal", "high", "urgent"]

# ---------------------------------------------------------------------------
# Models — descriptions here reach the agent as parameter descriptions
# ---------------------------------------------------------------------------


class Customer(BaseModel):
    """A customer of the helpdesk."""

    id: str = Field(description="Customer identifier, e.g. 'cus_ada'.")
    name: str = Field(description="Full name.")
    email: str = Field(description="Primary email address.")
    plan: str = Field(description="Subscription plan: 'free', 'team' or 'enterprise'.")


class Ticket(BaseModel):
    """A support ticket."""

    id: int = Field(description="Stable numeric ticket number.")
    customer_id: str = Field(description="The customer who opened it.")
    subject: str = Field(description="One-line subject.")
    body: str = Field(description="What the customer wrote.")
    status: Status = Field(description="'open', 'pending' (waiting on the customer) or 'closed'.")
    priority: Priority = Field(description="'low', 'normal', 'high' or 'urgent'.")
    assignee: str | None = Field(default=None, description="Agent handle, e.g. 'sam'.")
    opened_on: date = Field(description="Date the ticket was opened.")
    notes: list[str] = Field(default_factory=list, description="Internal notes, oldest first.")
    refunded: float = Field(default=0.0, description="Total refunded on this ticket, in EUR.")


class TicketCreate(BaseModel):
    """Fields needed to open a ticket."""

    customer_id: str = Field(description="The customer opening it.", examples=["cus_ada"])
    subject: str = Field(description="One-line subject.", examples=["Cannot export report"])
    body: str = Field(description="The customer's message.")
    priority: Priority = Field(default="normal", description="Initial priority.")


class TicketUpdate(BaseModel):
    """Fields that may change on a ticket; omitted fields are left alone."""

    priority: Priority | None = Field(default=None, description="New priority.")
    assignee: str | None = Field(default=None, description="Agent handle to assign it to.")
    note: str | None = Field(
        default=None, description="An internal note to append.", examples=["Called the customer"]
    )


class SearchQuery(BaseModel):
    """A search request (POST, because it carries a body)."""

    query: str = Field(
        description="Case-insensitive text matched against subject and body.",
        examples=["invoice"],
    )
    status: Status | None = Field(default=None, description="Only tickets in this status.")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results.")


class RefundRequest(BaseModel):
    """A refund issued against a ticket."""

    amount: float = Field(gt=0, description="Amount in EUR.", examples=[29.9])
    reason: str = Field(description="Why the refund is issued.", examples=["double charge"])


# ---------------------------------------------------------------------------
# Auth — a bearer token, checked by a dependency
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(description="Bearer token issued by the helpdesk admin.")


def require_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
) -> None:
    """Reject any request whose bearer token is not the demo token."""
    # Compare bytes, not str: `compare_digest` raises on non-ASCII text, and a
    # client can send any bytes in the header — that must be a 401, not a 500.
    presented = credentials.credentials.encode("utf-8")
    if not secrets.compare_digest(presented, DEMO_TOKEN.encode("utf-8")):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid token")


# ---------------------------------------------------------------------------
# State — an in-memory store seeded with a few customers and tickets
# ---------------------------------------------------------------------------

CUSTOMERS: dict[str, Customer] = {
    c.id: c
    for c in (
        Customer(id="cus_ada", name="Ada Lovelace", email="ada@analytical.example", plan="team"),
        Customer(
            id="cus_grace", name="Grace Hopper", email="grace@cobol.example", plan="enterprise"
        ),
        Customer(id="cus_alan", name="Alan Turing", email="alan@bletchley.example", plan="free"),
    )
}

SEED: list[Ticket] = [
    Ticket(
        id=1,
        customer_id="cus_ada",
        subject="Invoice charged twice in March",
        body="My card shows two charges of 29.90 EUR for the March invoice. Please refund one.",
        status="open",
        priority="high",
        assignee="sam",
        opened_on=date(2026, 3, 4),
        notes=["Confirmed duplicate charge in the billing system."],
    ),
    Ticket(
        id=2,
        customer_id="cus_ada",
        subject="Export to CSV times out",
        body="Exporting the analytics report for Q1 hangs at 90% and then fails.",
        status="pending",
        priority="normal",
        assignee=None,
        opened_on=date(2026, 3, 11),
    ),
    Ticket(
        id=3,
        customer_id="cus_grace",
        subject="SSO login loop",
        body="After the SAML update our users are bounced back to the login page.",
        status="open",
        priority="urgent",
        assignee="lin",
        opened_on=date(2026, 3, 12),
        notes=["Escalated to the identity team."],
    ),
    Ticket(
        id=4,
        customer_id="cus_alan",
        subject="Feature request: dark mode",
        body="It would be nice to have a dark theme for the dashboard.",
        status="closed",
        priority="low",
        assignee="sam",
        opened_on=date(2026, 2, 20),
        notes=["Added to the roadmap.", "Closed: shipped in 4.2."],
    ),
    Ticket(
        id=5,
        customer_id="cus_ada",
        subject="Cannot add a team member",
        body="Inviting a colleague fails with 'seat limit reached' although we have 3 seats free.",
        status="open",
        priority="normal",
        assignee=None,
        opened_on=date(2026, 3, 15),
    ),
]
TICKETS: dict[int, Ticket] = {}


def reset_store() -> None:
    """Restore the seed data (used at start-up and by the admin purge)."""
    TICKETS.clear()
    TICKETS.update({t.id: t.model_copy(deep=True) for t in SEED})


reset_store()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Helpdesk API",
    version="2.3.0",
    description="Customers, support tickets, refunds and the admin corner of a small helpdesk.",
)
tickets = APIRouter(prefix="/tickets", tags=["tickets"], dependencies=[Depends(require_token)])
customers = APIRouter(
    prefix="/customers", tags=["customers"], dependencies=[Depends(require_token)]
)
admin = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_token)])


@app.get("/health", operation_id="health_check", summary="Liveness probe", tags=["ops"])
async def health_check() -> dict[str, str]:
    """Returns ok when the service is up. Used by the load balancer, not by people."""
    return {"status": "ok"}


@tickets.get("", operation_id="list_tickets", summary="List tickets, newest first")
async def list_tickets(
    status_filter: Annotated[
        Status | None, Query(alias="status", description="Only tickets in this status.")
    ] = None,
    customer_id: Annotated[
        str | None, Query(description="Only tickets opened by this customer id.")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum number of tickets.")] = 20,
) -> list[Ticket]:
    """Every ticket, newest first, optionally filtered by status and customer."""
    found = sorted(TICKETS.values(), key=lambda t: t.opened_on, reverse=True)
    if status_filter is not None:
        found = [t for t in found if t.status == status_filter]
    if customer_id is not None:
        found = [t for t in found if t.customer_id == customer_id]
    return found[:limit]


@tickets.get(
    "/export",
    operation_id="export_tickets_csv",
    summary="Legacy CSV export",
    deprecated=True,
)
async def export_tickets_csv() -> str:
    """Deprecated: use the reporting API instead."""
    return "id,subject\n" + "\n".join(f"{t.id},{t.subject}" for t in TICKETS.values())


@tickets.post("/search", operation_id="search_tickets", summary="Search tickets by text")
async def search_tickets(request: SearchQuery) -> list[Ticket]:
    """Full-text search over subjects and bodies. Read-only despite being a POST."""
    needle = request.query.lower()
    found = [t for t in TICKETS.values() if needle in t.subject.lower() or needle in t.body.lower()]
    if request.status is not None:
        found = [t for t in found if t.status == request.status]
    return found[: request.limit]


@tickets.get("/{ticket_id}", operation_id="get_ticket", summary="Get one ticket by number")
async def get_ticket(ticket_id: int) -> Ticket:
    """The full ticket, including internal notes and the refunded total."""
    if ticket_id not in TICKETS:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such ticket")
    return TICKETS[ticket_id]


@tickets.post(
    "",
    operation_id="create_ticket",
    summary="Open a new ticket",
    status_code=status.HTTP_201_CREATED,
)
async def create_ticket(request: TicketCreate) -> Ticket:
    """Open a ticket for a customer; returns it with its new number."""
    if request.customer_id not in CUSTOMERS:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such customer")
    ticket = Ticket(
        id=max(TICKETS, default=0) + 1,
        opened_on=date.today(),
        status="open",
        **request.model_dump(),
    )
    TICKETS[ticket.id] = ticket
    return ticket


@tickets.patch("/{ticket_id}", operation_id="update_ticket", summary="Update a ticket")
async def update_ticket(ticket_id: int, request: TicketUpdate) -> Ticket:
    """Change priority or assignee, or append an internal note. Only sent fields change."""
    ticket = await get_ticket(ticket_id)
    if request.priority is not None:
        ticket.priority = request.priority
    if request.assignee is not None:
        ticket.assignee = request.assignee
    if request.note is not None:
        ticket.notes.append(request.note)
    return ticket


@tickets.post("/{ticket_id}/close", operation_id="close_ticket", summary="Close a ticket")
async def close_ticket(
    ticket_id: int,
    resolution: Annotated[str, Query(description="One line on how it was resolved.")],
) -> Ticket:
    """Mark a ticket closed and record the resolution as its last note."""
    ticket = await get_ticket(ticket_id)
    ticket.status = "closed"
    ticket.notes.append(f"Closed: {resolution}")
    return ticket


@tickets.post("/{ticket_id}/refund", operation_id="refund_ticket", summary="Refund a customer")
async def refund_ticket(ticket_id: int, request: RefundRequest) -> Ticket:
    """Issue a refund against a ticket. Money leaves the company; the amount is recorded."""
    ticket = await get_ticket(ticket_id)
    ticket.refunded += request.amount
    ticket.notes.append(f"Refunded {request.amount:.2f} EUR: {request.reason}")
    return ticket


@tickets.delete(
    "/{ticket_id}",
    operation_id="delete_ticket",
    summary="Delete a ticket permanently",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_ticket(ticket_id: int) -> None:
    """Remove a ticket and its notes. There is no undo."""
    await get_ticket(ticket_id)
    del TICKETS[ticket_id]


@customers.get("/{customer_id}", operation_id="get_customer", summary="Get one customer")
async def get_customer(customer_id: str) -> Customer:
    """Name, email and plan of one customer."""
    if customer_id not in CUSTOMERS:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such customer")
    return CUSTOMERS[customer_id]


@admin.post("/purge-closed", operation_id="purge_closed_tickets", summary="Purge closed tickets")
async def purge_closed_tickets() -> dict[str, int]:
    """Delete every closed ticket. Operator maintenance, not for day-to-day use."""
    closed = [t.id for t in TICKETS.values() if t.status == "closed"]
    for ticket_id in closed:
        del TICKETS[ticket_id]
    return {"purged": len(closed)}


app.include_router(tickets)
app.include_router(customers)
app.include_router(admin)
