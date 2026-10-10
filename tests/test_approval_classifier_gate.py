"""AutoApprovalClassifier behind the agent's approval gate (guide 25 regressions).

1. With ``redact_sensitive=True`` (the default) the classifier saw redacted
   arguments, so a deny rule on ``@competitor.example`` never fired.
2. The ``max_pending`` and retry-limit short-circuits produced no decision
   anyone could record, and classifier rule denials counted towards
   ``max_retries_after_deny``.
3. The classifier's trace is per decision, also through the gate.
4. MCP tool annotations reach the classifier's read-only layer.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from unittest.mock import patch

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from promptise import CallerContext
from promptise.agent import _caller_ctx_var, _session_ctx_var
from promptise.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    wrap_tools_with_approval,
)
from promptise.approval_classifier import ApprovalRule, AutoApprovalClassifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class EmailArgs(BaseModel):
    to: str
    subject: str
    body: str


class SendEmail(BaseTool):
    name: str = "send_email"
    description: str = "Send an email."
    args_schema: type[BaseModel] = EmailArgs

    async def _arun(self, **kwargs: Any) -> str:
        return f"SENT to {kwargs['to']}"

    def _run(self, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError


class RefundArgs(BaseModel):
    order_id: str
    amount: float


class IssueRefund(BaseTool):
    name: str = "issue_refund"
    description: str = "Refund an order."
    args_schema: type[BaseModel] = RefundArgs

    async def _arun(self, **kwargs: Any) -> str:
        return f"REFUNDED {kwargs['amount']} on {kwargs['order_id']}"

    def _run(self, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError


class Human:
    """Fallback reviewer answering from a script (default: approve)."""

    def __init__(self, *answers: bool, wait: asyncio.Event | None = None) -> None:
        self.requests: list[ApprovalRequest] = []
        self._answers = list(answers)
        self._wait = wait

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        if self._wait is not None:
            await self._wait.wait()
        approved = self._answers.pop(0) if self._answers else True
        return ApprovalDecision(approved=approved, reviewer_id="lead", reason="lead decided")


class Audit:
    """``on_decision`` sink."""

    def __init__(self) -> None:
        self.entries: list[tuple[ApprovalRequest, ApprovalDecision]] = []

    def __call__(self, request: ApprovalRequest, decision: ApprovalDecision) -> None:
        self.entries.append((request, decision))

    @property
    def summary(self) -> list[tuple[str, bool, str, str | None]]:
        return [
            (
                req.tool_name,
                dec.approved,
                dec.decided_by,
                dec.trace.layer if dec.trace is not None else None,
            )
            for req, dec in self.entries
        ]


def gate(tool_: BaseTool, handler: Any, **policy_kwargs: Any) -> BaseTool:
    policy_kwargs.setdefault("tools", ["*"])
    event_notifier = policy_kwargs.pop("event_notifier", None)
    [wrapped] = wrap_tools_with_approval(
        [tool_],
        ApprovalPolicy(handler=handler, **policy_kwargs),
        event_notifier=event_notifier,
    )
    return wrapped


async def as_caller(caller: CallerContext | None, coro: Any, *, session: str | None = None) -> Any:
    caller_token = _caller_ctx_var.set(caller)
    session_token = _session_ctx_var.set(session)
    try:
        return await coro
    finally:
        _session_ctx_var.reset(session_token)
        _caller_ctx_var.reset(caller_token)


COMPETITOR_EMAIL = {"to": "sam@competitor.example", "subject": "Hi", "body": "Prices attached"}


# ---------------------------------------------------------------------------
# 1. Rules see the real arguments; reviewers see the redacted copy
# ---------------------------------------------------------------------------


class TestRulesSeeRealArguments:
    async def test_deny_rule_on_an_email_domain_fires_with_default_redaction(self):
        human, audit = Human(), Audit()
        classifier = AutoApprovalClassifier(
            deny_rules=[
                ApprovalRule(
                    tool="send_*",
                    argument_contains="@competitor.example",
                    reason="never email competitors",
                )
            ],
            fallback=human,
        )
        wrapped = gate(SendEmail(), classifier, on_decision=audit)  # redact_sensitive=True

        result = await wrapped.ainvoke(COMPETITOR_EMAIL)

        assert result == "DENIED: never email competitors"
        assert human.requests == []
        [(request, decision)] = audit.entries
        assert decision.decided_by == "classifier"
        assert decision.trace is not None and decision.trace.layer == "deny_rule"
        # The audit copy is redacted
        assert request.arguments["to"] == "[EMAIL]"

    async def test_the_human_still_gets_the_redacted_copy(self):
        human = Human()
        classifier = AutoApprovalClassifier(
            deny_rules=[ApprovalRule(tool="send_*", argument_contains="@competitor.example")],
            fallback=human,
        )
        wrapped = gate(SendEmail(), classifier)

        result = await wrapped.ainvoke({**COMPETITOR_EMAIL, "to": "dana@example.com"})

        assert result == "SENT to dana@example.com"
        [shown] = human.requests
        assert shown.arguments["to"] == "[EMAIL]"
        assert shown.raw_arguments is None
        assert "dana@example.com" not in json.dumps(shown.to_dict())

    async def test_predicates_see_real_order_ids(self):
        seen: list[str] = []

        async def small(request: ApprovalRequest) -> bool:
            seen.append(request.arguments["order_id"])
            return request.arguments["amount"] < 20

        classifier = AutoApprovalClassifier(
            allow_rules=[ApprovalRule(tool="issue_refund", predicate=small)],
            fallback=Human(False),
        )
        wrapped = gate(IssueRefund(), classifier)
        assert "REFUNDED" in await wrapped.ainvoke({"order_id": "A-1003", "amount": 9.0})
        assert seen == ["A-1003"]

    async def test_rules_work_when_arguments_are_hidden_from_reviewers(self):
        human = Human()
        classifier = AutoApprovalClassifier(
            deny_rules=[ApprovalRule(tool="send_*", argument_contains="@competitor.example")],
            fallback=human,
        )
        wrapped = gate(SendEmail(), classifier, include_arguments=False)
        assert "DENIED" in await wrapped.ainvoke(COMPETITOR_EMAIL)
        assert "SENT" in await wrapped.ainvoke({**COMPETITOR_EMAIL, "to": "dana@example.com"})
        assert human.requests[0].arguments == {}


# ---------------------------------------------------------------------------
# 2. Every decision is recorded; only reviewer denials count
# ---------------------------------------------------------------------------


def _refund_classifier(human: Human) -> AutoApprovalClassifier:
    async def small(request: ApprovalRequest) -> bool:
        return request.arguments["amount"] < 20

    async def finance_sized(request: ApprovalRequest) -> bool:
        return request.arguments["amount"] > 500

    return AutoApprovalClassifier(
        deny_rules=[
            ApprovalRule(tool="issue_refund", predicate=finance_sized, reason="over the limit")
        ],
        allow_rules=[ApprovalRule(tool="issue_refund", predicate=small, reason="under 20")],
        fallback=human,
    )


class TestRetryLimitAndRuleDenials:
    async def test_rule_denials_do_not_use_up_the_retry_limit(self):
        human, audit = Human(), Audit()
        wrapped = gate(IssueRefund(), _refund_classifier(human), on_decision=audit)
        for amount in (900, 800, 700, 600):
            result = await wrapped.ainvoke({"order_id": "A-1001", "amount": amount})
            assert result == "DENIED: over the limit"
        assert "REFUNDED 5.0" in await wrapped.ainvoke({"order_id": "A-1001", "amount": 5})
        # A mid-sized refund still goes to the human
        assert "REFUNDED 240.0" in await wrapped.ainvoke({"order_id": "A-1001", "amount": 240})
        assert len(human.requests) == 1
        assert [layer for *_, layer in audit.summary] == [
            "deny_rule",
            "deny_rule",
            "deny_rule",
            "deny_rule",
            "allow_rule",
            "fallback",
        ]

    async def test_reviewer_denials_count_and_rule_approvals_do_not_reset_them(self):
        human, audit = Human(False, False), Audit()
        wrapped = gate(
            IssueRefund(),
            _refund_classifier(human),
            on_decision=audit,
            max_retries_after_deny=2,
        )
        alice = CallerContext(user_id="alice")
        for amount in (240, 5, 250):
            await as_caller(alice, wrapped.ainvoke({"order_id": "A-1", "amount": amount}))
        result = await as_caller(alice, wrapped.ainvoke({"order_id": "A-1", "amount": 260}))
        assert "already denied 2 times" in result
        assert len(human.requests) == 2
        assert audit.summary == [
            ("issue_refund", False, "reviewer", "fallback"),
            ("issue_refund", True, "classifier", "allow_rule"),
            ("issue_refund", False, "reviewer", "fallback"),
            ("issue_refund", False, "gate", None),
        ]
        # Another session of the same user is still asked
        assert "REFUNDED" in await as_caller(
            alice, wrapped.ainvoke({"order_id": "A-1", "amount": 260}), session="other"
        )

    async def test_retry_limit_short_circuit_is_recorded_and_emitted(self):
        audit = Audit()
        events: list[tuple[str, dict[str, Any]]] = []
        wrapped = gate(
            IssueRefund(),
            Human(False),
            on_decision=audit,
            max_retries_after_deny=1,
            event_notifier=object(),
        )
        with patch(
            "promptise.events.emit_event",
            lambda notifier, event_type, severity, data, **kw: events.append((event_type, data)),
        ):
            await wrapped.ainvoke({"order_id": "A-1", "amount": 50})
            result = await wrapped.ainvoke({"order_id": "A-1", "amount": 50})

        assert "already denied 1 times" in result
        request, decision = audit.entries[-1]
        assert decision.decided_by == "gate" and not decision.approved
        assert request.tool_name == "issue_refund" and request.request_id
        denied = [data for name, data in events if name == "approval.denied"]
        assert [d["decided_by"] for d in denied] == ["reviewer", "gate"]

    async def test_max_pending_short_circuit_is_recorded(self):
        release = asyncio.Event()
        human, audit = Human(wait=release), Audit()
        wrapped = gate(IssueRefund(), human, on_decision=audit, max_pending=1)

        first = asyncio.create_task(wrapped.ainvoke({"order_id": "A-1", "amount": 50}))
        while not human.requests:
            await asyncio.sleep(0)
        second = await wrapped.ainvoke({"order_id": "A-2", "amount": 60})
        release.set()
        assert "REFUNDED" in await first

        assert "Too many pending approval requests" in second
        assert [(req.arguments["order_id"], dec.decided_by) for req, dec in audit.entries] == [
            ("A-2", "gate"),
            ("A-1", "reviewer"),
        ]

    async def test_timeout_and_handler_error_are_recorded(self):
        audit = Audit()

        class Silent:
            async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
                await asyncio.sleep(10)
                raise AssertionError("unreachable")

        class Broken:
            async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
                raise RuntimeError("queue down")

        await gate(IssueRefund(), Silent(), on_decision=audit, timeout=0.05).ainvoke(
            {"order_id": "A-1", "amount": 50}
        )
        await gate(IssueRefund(), Broken(), on_decision=audit).ainvoke(
            {"order_id": "A-1", "amount": 50}
        )
        assert [(dec.decided_by, dec.reason) for _, dec in audit.entries] == [
            ("gate", "Approval timed out after 0.05s"),
            ("gate", "Approval handler error: RuntimeError"),
        ]

    async def test_async_on_decision_and_errors_in_it(self, caplog):
        recorded: list[str] = []

        async def async_audit(request: ApprovalRequest, decision: ApprovalDecision) -> None:
            recorded.append(request.tool_name)

        assert "REFUNDED" in await gate(IssueRefund(), Human(), on_decision=async_audit).ainvoke(
            {"order_id": "A-1", "amount": 5}
        )
        assert recorded == ["issue_refund"]

        def broken_audit(request: ApprovalRequest, decision: ApprovalDecision) -> None:
            raise OSError("disk full")

        result = await gate(IssueRefund(), Human(), on_decision=broken_audit).ainvoke(
            {"order_id": "A-1", "amount": 5}
        )
        assert "REFUNDED" in result
        assert "on_decision raised" in caplog.text

    def test_on_decision_must_be_callable(self):
        with pytest.raises(TypeError, match="on_decision"):
            ApprovalPolicy(tools=["*"], handler=Human(), on_decision="audit.jsonl")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Concurrent gated calls keep their own trace
# ---------------------------------------------------------------------------


class TestConcurrentTraces:
    async def test_escalated_call_is_recorded_with_its_own_layer(self):
        release = asyncio.Event()
        human, audit = Human(wait=release), Audit()

        async def llm(request: ApprovalRequest) -> tuple[str, str]:
            return "escalate", "unsure about this one"

        classifier = AutoApprovalClassifier(llm_classifier=llm, fallback=human)
        [refund, get_order] = wrap_tools_with_approval(
            [IssueRefund(), _GetOrder()],
            ApprovalPolicy(tools=["*"], handler=classifier, on_decision=audit),
        )
        escalated = asyncio.create_task(refund.ainvoke({"order_id": "A-1", "amount": 50}))
        while not human.requests:
            await asyncio.sleep(0)
        await get_order.ainvoke({"order_id": "A-1"})
        release.set()
        await escalated

        by_tool = {req.tool_name: dec for req, dec in audit.entries}
        assert by_tool["get_order"].trace.layer == "read_only"
        assert by_tool["issue_refund"].trace.layer == "llm_escalate_then_fallback"
        assert by_tool["issue_refund"].trace.rule_reason == "unsure about this one"


class OrderArgs(BaseModel):
    order_id: str


class _GetOrder(BaseTool):
    name: str = "get_order"
    description: str = "Look up an order."
    args_schema: type[BaseModel] = OrderArgs

    async def _arun(self, **kwargs: Any) -> str:
        return f"order {kwargs['order_id']}"

    def _run(self, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 4. MCP tool annotations reach the read-only layer
# ---------------------------------------------------------------------------


class TestToolAnnotations:
    async def _mcp_tools(self) -> dict[str, BaseTool]:
        from promptise.mcp.client import MCPToolAdapter
        from promptise.mcp.server import MCPServer
        from promptise.mcp.server._testing import TestClient

        server = MCPServer(name="shop")

        @server.tool(read_only_hint=True)
        async def order_status(order_id: str) -> str:
            """Status of an order."""
            return f"{order_id}: shipped"

        @server.tool(read_only_hint=False, destructive_hint=True)
        async def get_and_reset_counter(name: str) -> str:
            """Read a counter and reset it."""
            return "0"

        @server.tool(destructive_hint=True)
        async def get_session(session_id: str) -> str:
            """Get a session (and end it)."""
            return "ended"

        client = TestClient(server)

        class _Multi:
            tool_to_server: dict[str, str] = {}

            async def list_tools(self) -> Any:
                return await client.list_tools()

            async def call_tool(self, name: str, arguments: dict[str, Any], **_: Any) -> Any:
                from mcp.types import CallToolResult

                content = await client.call_tool(name, arguments)
                return CallToolResult(content=content)

        tools = await MCPToolAdapter(_Multi()).as_langchain_tools()  # type: ignore[arg-type]
        return {t.name: t for t in tools}

    async def test_adapter_keeps_annotations_in_metadata(self):
        tools = await self._mcp_tools()
        assert tools["order_status"].metadata == {"readOnlyHint": True}
        assert tools["get_session"].metadata == {"destructiveHint": True}

    async def test_classifier_uses_annotations_through_the_gate(self):
        tools = await self._mcp_tools()
        human, audit = Human(False, False, False), Audit()
        classifier = AutoApprovalClassifier(fallback=human)
        wrapped = {
            t.name: t
            for t in wrap_tools_with_approval(
                list(tools.values()),
                ApprovalPolicy(tools=["*"], handler=classifier, on_decision=audit),
            )
        }

        assert "shipped" in await wrapped["order_status"].ainvoke({"order_id": "A-1"})
        assert "DENIED" in await wrapped["get_and_reset_counter"].ainvoke({"name": "x"})
        assert "DENIED" in await wrapped["get_session"].ainvoke({"session_id": "s"})

        assert [(name, layer) for name, _, _, layer in audit.summary] == [
            ("order_status", "read_only"),
            ("get_and_reset_counter", "fallback"),
            ("get_session", "fallback"),
        ]
        assert human.requests[1].tool_annotations == {"destructiveHint": True}
        assert human.requests[1].to_dict()["tool_annotations"] == {"destructiveHint": True}
