"""An in-process fake of the Storefront API, as an ``httpx.MockTransport``.

The lab needs an upstream it can *prove things about*: that a refund never
reached the API before a human approved it, and that the call the server
finally made carried the right tenant's credential.  A dict-backed fake gives
exactly that — no third-party service, no network, and a complete request log.

This is not an LLM mock: the agent in ``run.py`` is a real
:func:`~promptise.agent.build_agent` calling a real model.  Only the upstream
HTTP API is faked, the way you would fake Stripe in your own test suite.

Example::

    api = FakeStorefront()
    async with api.client() as http:
        server = module.build_server(http_client=http)
    print(api.log)  # ['GET /v1/customers/CUS-1001', ...]
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

__all__ = ["Call", "FakeStorefront"]

PREFIX = "/v1"


@dataclass(frozen=True)
class Call:
    """One request the fake API received."""

    method: str
    path: str
    query: dict[str, str]
    body: Any
    authorization: str | None

    def __str__(self) -> str:
        query = f"?{'&'.join(f'{k}={v}' for k, v in self.query.items())}" if self.query else ""
        return f"{self.method} {self.path}{query}"


def _customer(
    customer_id: str, name: str, email: str, plan: str, created_at: str
) -> dict[str, Any]:
    return {
        "customer_id": customer_id,
        "name": name,
        "email": email,
        "plan": plan,
        "created_at": created_at,
    }


def _order(
    order_id: str, customer_id: str, status: str, total: float, placed_at: str, sku: str
) -> dict[str, Any]:
    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "status": status,
        "total": total,
        "currency": "eur",
        "placed_at": placed_at,
        "items": [{"sku": sku, "quantity": 1, "unit_price": total}],
    }


@dataclass
class FakeStorefront:
    """A tiny, mutable stand-in for the Storefront API.

    Attributes:
        calls: Every request received, in order — the lab's evidence log.
        customers: Customer records by id.
        orders: Order records by id.
        refunds: Refund receipts, in the order they were issued.
    """

    calls: list[Call] = field(default_factory=list)
    customers: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "CUS-1001": _customer(
                "CUS-1001", "Ada Lovelace", "ada@northwind.example", "pro", "2025-03-04"
            ),
            "CUS-1002": _customer(
                "CUS-1002", "Grace Hopper", "grace@northwind.example", "enterprise", "2024-11-19"
            ),
            "CUS-1003": _customer(
                "CUS-1003", "Alan Turing", "alan@northwind.example", "starter", "2026-01-22"
            ),
        }
    )
    orders: dict[str, dict[str, Any]] = field(
        default_factory=lambda: {
            "ORD-1001": _order("ORD-1001", "CUS-1001", "paid", 129.0, "2026-01-28", "KB-PRO-87"),
            "ORD-1002": _order("ORD-1002", "CUS-1001", "shipped", 49.9, "2026-02-11", "HUB-USBC-7"),
            "ORD-1003": _order("ORD-1003", "CUS-1002", "paid", 24.0, "2026-02-14", "STK-PACK-1"),
            "ORD-1004": _order("ORD-1004", "CUS-1003", "open", 89.0, "2026-02-20", "MAT-DESK-90"),
        }
    )
    refunds: list[dict[str, Any]] = field(default_factory=list)

    # -- wiring ---------------------------------------------------------

    def transport(self) -> httpx.MockTransport:
        """An ``httpx`` transport that answers this API."""
        return httpx.MockTransport(self._handle)

    def client(self) -> httpx.AsyncClient:
        """An ``httpx.AsyncClient`` bound to this API (pass to ``build_server``)."""
        return httpx.AsyncClient(transport=self.transport(), timeout=10)

    @property
    def log(self) -> list[str]:
        """Every request received, as ``"METHOD /path?query"`` strings."""
        return [str(c) for c in self.calls]

    def received(self, method: str, contains: str) -> list[Call]:
        """Recorded calls with this method whose path contains *contains*."""
        return [c for c in self.calls if c.method == method and contains in c.path]

    # -- request handling -----------------------------------------------

    async def _handle(self, request: httpx.Request) -> httpx.Response:
        body: Any = None
        raw = request.content
        if raw:
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                body = raw.decode("utf-8", "replace")
        call = Call(
            method=request.method,
            path=request.url.path,
            query=dict(request.url.params),
            body=body,
            authorization=request.headers.get("authorization"),
        )
        self.calls.append(call)

        if not call.authorization:
            return self._json(request, 401, {"code": "unauthorized", "message": "Missing token."})
        route = call.path.removeprefix(PREFIX)
        for method, pattern, handler in self._routes():
            match = re.fullmatch(pattern, route)
            if match and method == call.method:
                return self._json(request, *handler(call, *match.groups()))
        return self._json(
            request, 404, {"code": "not_found", "message": f"No route for {call.method} {route}."}
        )

    def _routes(self) -> list[tuple[str, str, Any]]:
        return [
            ("GET", r"/health", self._health),
            ("GET", r"/customers", self._list_customers),
            ("POST", r"/customers/search", self._search_customers),
            ("GET", r"/customers/([^/]+)", self._get_customer),
            ("GET", r"/orders", self._list_orders),
            ("POST", r"/orders", self._create_order),
            ("GET", r"/orders/([^/]+)", self._get_order),
            ("POST", r"/orders/([^/]+)/refund", self._refund_order),
            ("DELETE", r"/subscriptions/([^/]+)", self._cancel_subscription),
            ("GET", r"/reports/revenue", self._revenue_report),
            ("GET", r"/admin/customers/([^/]+)/pii", self._customer_pii),
        ]

    @staticmethod
    def _json(request: httpx.Request, status: int, payload: Any) -> httpx.Response:
        return httpx.Response(status, json=payload, request=request)

    # -- endpoints ------------------------------------------------------

    def _health(self, _call: Call) -> tuple[int, Any]:
        return 200, {"status": "ok", "region": "eu-central-1"}

    def _list_customers(self, call: Call) -> tuple[int, Any]:
        found = list(self.customers.values())
        if q := call.query.get("q", "").lower():
            found = [c for c in found if q in c["name"].lower() or q in c["email"].lower()]
        if plan := call.query.get("plan"):
            found = [c for c in found if c["plan"] == plan]
        limit = int(call.query.get("limit", 20))
        return 200, {"data": found[:limit], "has_more": len(found) > limit}

    def _search_customers(self, call: Call) -> tuple[int, Any]:
        email = (call.body or {}).get("email", "").lower()
        return 200, {"data": [c for c in self.customers.values() if c["email"].lower() == email]}

    def _get_customer(self, _call: Call, customer_id: str) -> tuple[int, Any]:
        customer = self.customers.get(customer_id)
        if customer is None:
            return 404, {"code": "not_found", "message": f"No customer {customer_id}."}
        return 200, customer

    def _list_orders(self, call: Call) -> tuple[int, Any]:
        found = list(self.orders.values())
        if customer_id := call.query.get("customer_id"):
            found = [o for o in found if o["customer_id"] == customer_id]
        if status := call.query.get("status"):
            found = [o for o in found if o["status"] == status]
        found.sort(key=lambda o: o["placed_at"], reverse=True)
        limit = int(call.query.get("limit", 20))
        return 200, {"data": found[:limit], "has_more": len(found) > limit}

    def _create_order(self, call: Call) -> tuple[int, Any]:
        body = call.body or {}
        order_id = f"ORD-{1001 + len(self.orders)}"
        items = body.get("items") or []
        total = sum(float(i.get("unit_price", 0)) * int(i.get("quantity", 1)) for i in items)
        order = {
            "order_id": order_id,
            "customer_id": body.get("customer_id", ""),
            "status": "open",
            "total": round(total, 2),
            "currency": body.get("currency", "eur"),
            "placed_at": "2026-02-24",
            "items": items,
        }
        self.orders[order_id] = order
        return 201, order

    def _get_order(self, _call: Call, order_id: str) -> tuple[int, Any]:
        order = self.orders.get(order_id)
        if order is None:
            return 404, {"code": "not_found", "message": f"No order {order_id}."}
        return 200, order

    def _refund_order(self, call: Call, order_id: str) -> tuple[int, Any]:
        order = self.orders.get(order_id)
        if order is None:
            return 404, {"code": "not_found", "message": f"No order {order_id}."}
        amount = float((call.body or {}).get("amount", order["total"]))
        receipt = {
            "refund_id": f"REF-{3001 + len(self.refunds)}",
            "order_id": order_id,
            "amount": amount,
            "status": "succeeded",
        }
        self.refunds.append(receipt)
        order["status"] = "refunded"
        return 200, receipt

    def _cancel_subscription(self, call: Call, subscription_id: str) -> tuple[int, Any]:
        at_period_end = call.query.get("at_period_end", "true") == "true"
        return 200, {
            "subscription_id": subscription_id,
            "status": "cancelled",
            "ends_at": "2026-03-31" if at_period_end else "2026-02-24",
        }

    def _revenue_report(self, call: Call) -> tuple[int, Any]:
        return 200, {
            "month": call.query.get("month", "2026-02"),
            "gross": 18420.5,
            "net": 17103.9,
            "currency": "eur",
        }

    def _customer_pii(self, _call: Call, customer_id: str) -> tuple[int, Any]:
        return 200, {
            "customer_id": customer_id,
            "address": "12 Analytical Engine Way, London",
            "phone": "+44 20 7946 0000",
            "tax_id": "GB123456789",
        }
