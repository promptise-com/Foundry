"""Table-driven tests for the deterministic risk classifier."""

from __future__ import annotations

import pytest

from promptise.mcpcast.classify import classify, classify_operation, risk_floor, tokens
from promptise.mcpcast.parse import Operation
from promptise.mcpcast.schema import RiskClass


def op(method: str, path: str, summary: str = "", op_id: str = "", **kw) -> Operation:
    return Operation(
        operation_id=op_id or f"{method.lower()}_{path.strip('/').replace('/', '_') or 'root'}",
        method=method,  # type: ignore[arg-type]
        path=path,
        summary=summary,
        **kw,
    )


CASES: list[tuple[Operation, RiskClass]] = [
    # -- reads -------------------------------------------------------------
    (op("GET", "/pets"), RiskClass.READ),
    (op("HEAD", "/pets/{id}"), RiskClass.READ),
    (op("OPTIONS", "/pets"), RiskClass.READ),
    (op("GET", "/invoices"), RiskClass.READ),  # listing money is still a read
    (op("GET", "/deleted-items", "List deleted items"), RiskClass.READ),
    (op("GET", "/subscriptions/{id}"), RiskClass.READ),
    # -- read-like POSTs (the traps) ----------------------------------------
    (op("POST", "/search"), RiskClass.READ),
    (op("POST", "/customers/search", "Search customers"), RiskClass.READ),
    # GraphQL accepts mutations as readily as queries — never a read.
    (op("POST", "/graphql"), RiskClass.WRITE),
    (op("POST", "/addresses/validate", "Validate an address"), RiskClass.READ),
    (op("POST", "/shipping/estimate"), RiskClass.READ),
    (op("POST", "/search", "Save a search and delete old ones"), RiskClass.DESTRUCTIVE),
    (op("POST", "/user/createWithList", op_id="createUsersWithListInput"), RiskClass.WRITE),
    (op("POST", "/reports/export", "Create a report from a query"), RiskClass.WRITE),
    (op("POST", "/files/upload", "Upload and validate a file"), RiskClass.WRITE),
    (op("POST", "/user/login", "Logs user into the system"), RiskClass.WRITE),
    # -- writes ------------------------------------------------------------
    (op("POST", "/pets", "Add a new pet"), RiskClass.WRITE),
    (op("PUT", "/pets/{id}", "Update a pet"), RiskClass.WRITE),
    (op("PATCH", "/pets/{id}"), RiskClass.WRITE),
    (op("POST", "/users/{id}/reactivate", "Reactivate user"), RiskClass.WRITE),
    (op("POST", "/customers", op_id="createCustomer"), RiskClass.WRITE),
    (op("POST", "/uploads", "Upload payload"), RiskClass.WRITE),  # 'payload' is not 'pay'
    # -- destructive -------------------------------------------------------
    (op("DELETE", "/pets/{id}"), RiskClass.DESTRUCTIVE),
    (op("DELETE", "/cache"), RiskClass.DESTRUCTIVE),
    (op("POST", "/pets/{id}/remove"), RiskClass.DESTRUCTIVE),
    (op("POST", "/tokens/{id}/revoke", "Revoke a token"), RiskClass.DESTRUCTIVE),
    (op("POST", "/instances/{id}:terminate"), RiskClass.DESTRUCTIVE),
    (
        op("POST", "/customers/{id}/subscriptions:cancel", "Cancel subscription"),
        RiskClass.DESTRUCTIVE,
    ),
    (op("POST", "/users/{id}/deactivate"), RiskClass.DESTRUCTIVE),
    (op("POST", "/purge", op_id="purgeCache"), RiskClass.DESTRUCTIVE),
    (op("PUT", "/x", op_id="removeMember"), RiskClass.DESTRUCTIVE),  # verb in operationId
    (op("POST", "/passwords/reset", "Reset password"), RiskClass.DESTRUCTIVE),
    # -- financial ---------------------------------------------------------
    (op("POST", "/charges"), RiskClass.FINANCIAL),
    (op("POST", "/v1/payment_intents", "Create a PaymentIntent"), RiskClass.FINANCIAL),
    (op("POST", "/orders/{id}/refund"), RiskClass.FINANCIAL),
    (op("POST", "/invoices", "Create an invoice"), RiskClass.FINANCIAL),
    (op("POST", "/accounts/{id}/transfers"), RiskClass.FINANCIAL),
    (op("POST", "/payouts"), RiskClass.FINANCIAL),
    (op("POST", "/subscriptions", "Start a subscription"), RiskClass.FINANCIAL),
    (op("POST", "/billing/update"), RiskClass.FINANCIAL),
    (op("POST", "/checkout/sessions"), RiskClass.FINANCIAL),
    (op("POST", "/pay"), RiskClass.FINANCIAL),
    (op("POST", "/wallet/withdraw"), RiskClass.FINANCIAL),
    # -- escalation --------------------------------------------------------
    (op("GET", "/pets", scopes=["read:pets"]), RiskClass.READ),
    (op("GET", "/pets", scopes=["write:pets"]), RiskClass.READ),  # a scoped read is still a read
    (op("POST", "/pets", scopes=["write:pets"]), RiskClass.WRITE),  # ...and a scoped write a write
    (op("GET", "/users", scopes=["admin"]), RiskClass.WRITE),
    (op("GET", "/users", scopes=["repo:admin"]), RiskClass.WRITE),
    (op("POST", "/pets", scopes=["superuser"]), RiskClass.DESTRUCTIVE),
    (op("GET", "/admin/users"), RiskClass.WRITE),
    (op("GET", "/internal/metrics"), RiskClass.WRITE),
    (op("GET", "/administrators"), RiskClass.READ),  # 'administrators' is not 'admin'
    (op("GET", "/legacy", deprecated=True), RiskClass.WRITE),
    (op("POST", "/pets", deprecated=True), RiskClass.DESTRUCTIVE),
    (op("DELETE", "/pets/{id}", deprecated=True, scopes=["admin"]), RiskClass.DESTRUCTIVE),
    (op("POST", "/charges", deprecated=True), RiskClass.FINANCIAL),  # top of ladder stays
]


@pytest.mark.parametrize(
    "operation,expected", CASES, ids=[f"{o.method} {o.path}" for o, _ in CASES]
)
def test_classification_table(operation: Operation, expected: RiskClass) -> None:
    assert classify_operation(operation) is expected


def test_reasons_explain_base_and_escalation() -> None:
    c = classify(op("GET", "/admin/x", deprecated=True, scopes=["admin"]))
    assert c.base is RiskClass.READ
    assert (
        c.risk is RiskClass.DESTRUCTIVE
    )  # read → write (scope) → destructive (path); deprecated stays
    assert c.reasons[0] == "GET is a read"
    assert any("scope" in r for r in c.reasons)
    assert any("admin-only" in r for r in c.reasons)
    assert any("deprecated" in r for r in c.reasons)


def test_destructive_beats_financial_when_both_match() -> None:
    assert classify_operation(op("POST", "/subscriptions/{id}/cancel")) is RiskClass.DESTRUCTIVE


def test_tokens_split_camel_case_and_separators() -> None:
    assert tokens("cancelSubscription /v2/customers-{id}:cancel") == [
        "cancel",
        "subscription",
        "v2",
        "customers",
        "id",
        "cancel",
    ]


# ---------------------------------------------------------------------------
# risk_floor: what a plan file's wire mapping alone can prove
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "operation,expected", CASES, ids=[f"{o.method} {o.path}" for o, _ in CASES]
)
def test_floor_never_exceeds_the_full_classification(operation: Operation, expected: RiskClass):
    """The summary, scopes and deprecation can only raise the class, so the
    floor computed without them is a lower bound — a freshly generated plan
    always reloads."""
    floor = risk_floor(
        operation_id=operation.operation_id, method=operation.method, path=operation.path
    )
    assert expected.at_least(floor), (operation, expected, floor)


FLOORS = [
    ("deleteOrder", "DELETE", "/orders/{id}", RiskClass.DESTRUCTIVE),
    ("getPet", "GET", "/pets/{id}", RiskClass.READ),
    ("listInvoices", "GET", "/invoices", RiskClass.READ),
    ("searchBooks", "POST", "/books/search", RiskClass.READ),
    ("find_customer", "POST", "/customers/find", RiskClass.READ),
    ("validateAddress", "POST", "/addresses/validate", RiskClass.READ),
    # no mutating verb in the id or path: a summary the plan lacks may have
    # led with a query verb, so the floor stays at read
    ("postCustomers", "POST", "/customers", RiskClass.READ),
    ("post_orders", "POST", "/orders", RiskClass.READ),
    ("createOrder", "POST", "/orders", RiskClass.WRITE),
    ("post_users_create_with_list", "POST", "/users/createWithList", RiskClass.WRITE),
    ("updateOrder", "PUT", "/orders/{id}", RiskClass.WRITE),
    ("patchOrder", "PATCH", "/orders/{id}", RiskClass.WRITE),
    ("refundOrder", "POST", "/orders/{id}/refund", RiskClass.FINANCIAL),
    ("cancelSub", "POST", "/subs/{id}:cancel", RiskClass.DESTRUCTIVE),
    ("removeMember", "PUT", "/x", RiskClass.DESTRUCTIVE),
    ("adminUsers", "GET", "/admin/users", RiskClass.WRITE),
    ("purgeAll", "GET", "/purge", RiskClass.WRITE),  # a GET that says it mutates
    ("charge", "POST", "/charges", RiskClass.FINANCIAL),
]


@pytest.mark.parametrize("op_id,method,path,expected", FLOORS, ids=[f[0] for f in FLOORS])
def test_floor_table(op_id: str, method: str, path: str, expected: RiskClass) -> None:
    assert risk_floor(operation_id=op_id, method=method, path=path) is expected


def test_floor_matches_classification_without_summary_except_for_query_posts() -> None:
    for op_id, method, path, floor in FLOORS:
        bare = classify_operation(Operation(operation_id=op_id, method=method, path=path))  # type: ignore[arg-type]
        assert bare.at_least(floor)
        if not (method == "POST" and floor is RiskClass.READ):
            assert bare is floor
