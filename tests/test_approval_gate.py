"""Server-side approval gates — HITL enforced where the tool lives.

Covers: pass-through for ungated tools, approve/deny via callback, timeout
denied-by-default (and the explicit allow opt-out), modified-arguments
fail-closed, the build-time no-gate configuration error, the PendingApprover
block → list → decide flow with role-guarded admin tools, the per-client and
per-tenant pending caps (one caller, or one tenant with many keys, cannot deny
everyone else by filling the store), and the ElicitationApprover: fail-closed without a live MCP session, mocked at
the real SDK boundary (``ServerSession.elicit`` → ``ElicitResult``), a
contract test pinning that SDK method, and real in-process round trips over
the SDK's in-memory transport (accept, decline, no elicitation support).
"""

from __future__ import annotations

import asyncio

import pytest

from promptise.approval import ApprovalDecision, ApprovalRequest
from promptise.mcp.server import (
    ApprovalGateMiddleware,
    AuthMiddleware,
    ElicitationApprover,
    MCPServer,
    PendingApprover,
    TestClient,
)
from promptise.mcp.server._auth import APIKeyAuth
from promptise.mcp.server._context import ClientContext, RequestContext


def _server_with_gate(handler, **gate_kwargs) -> MCPServer:
    server = MCPServer(name="gated")
    server.add_middleware(ApprovalGateMiddleware(handler, **gate_kwargs))

    @server.tool(requires_approval=True)
    async def refund(order_id: str) -> str:
        """Refund an order."""
        return f"refunded {order_id}"

    @server.tool()
    async def lookup(order_id: str) -> str:
        """Look up an order."""
        return f"order {order_id}"

    return server


class TestGateCore:
    @pytest.mark.asyncio
    async def test_ungated_tool_passes_through(self):
        async def never_called(request):
            raise AssertionError("gate must not fire for ungated tools")

        client = TestClient(_server_with_gate(never_called))
        result = await client.call_tool("lookup", {"order_id": "o1"})
        assert result[0].text == "order o1"

    @pytest.mark.asyncio
    async def test_approved_call_proceeds(self):
        seen = {}

        async def approve(request):
            seen["request"] = request
            return ApprovalDecision(approved=True, reviewer_id="alice")

        client = TestClient(_server_with_gate(approve))
        result = await client.call_tool("refund", {"order_id": "o1"})
        assert result[0].text == "refunded o1"
        # The request carried the tool and its validated arguments
        assert seen["request"].tool_name == "refund"
        assert seen["request"].arguments == {"order_id": "o1"}

    @pytest.mark.asyncio
    async def test_denied_call_is_blocked(self):
        async def deny(request):
            return ApprovalDecision(approved=False, reviewer_id="bob", reason="not today")

        client = TestClient(_server_with_gate(deny))
        result = await client.call_tool("refund", {"order_id": "o1"})
        assert "APPROVAL_DENIED" in result[0].text
        assert "not today" in result[0].text
        assert "refunded" not in result[0].text

    @pytest.mark.asyncio
    async def test_bare_bool_callback_is_wrapped(self):
        client = TestClient(_server_with_gate(lambda request: True))
        result = await client.call_tool("refund", {"order_id": "o1"})
        assert result[0].text == "refunded o1"

    @pytest.mark.asyncio
    async def test_timeout_denies_by_default(self):
        async def stall(request):
            await asyncio.sleep(30)

        client = TestClient(_server_with_gate(stall, timeout=0.2))
        result = await client.call_tool("refund", {"order_id": "o1"})
        assert "APPROVAL_DENIED" in result[0].text
        assert "timed out" in result[0].text

    @pytest.mark.asyncio
    async def test_on_timeout_allow_proceeds(self):
        async def stall(request):
            await asyncio.sleep(30)

        client = TestClient(_server_with_gate(stall, timeout=0.2, on_timeout="allow"))
        result = await client.call_tool("refund", {"order_id": "o1"})
        assert result[0].text == "refunded o1"

    @pytest.mark.asyncio
    async def test_modified_arguments_fail_closed(self):
        async def modify(request):
            return ApprovalDecision(
                approved=True, modified_arguments={"order_id": "SOMETHING-ELSE"}
            )

        client = TestClient(_server_with_gate(modify))
        result = await client.call_tool("refund", {"order_id": "o1"})
        assert "APPROVAL_DENIED" in result[0].text
        assert "modified" in result[0].text
        assert "refunded" not in result[0].text

    @pytest.mark.asyncio
    async def test_request_metadata_carries_identity(self):
        seen = {}

        async def approve(request):
            seen["meta"] = request.metadata
            return ApprovalDecision(approved=True)

        server = MCPServer(name="gated")
        server.add_middleware(
            AuthMiddleware(APIKeyAuth(keys={"sk-1": {"client_id": "agent-1", "tenant_id": "acme"}}))
        )
        server.add_middleware(ApprovalGateMiddleware(approve))

        @server.tool(auth=True, requires_approval=True)
        async def refund(order_id: str) -> str:
            """Refund."""
            return "ok"

        client = TestClient(server)
        result = await client.call_tool("refund", {"order_id": "o1"}, headers={"x-api-key": "sk-1"})
        assert result[0].text == "ok"
        assert seen["meta"]["client_id"] == "agent-1"
        assert seen["meta"]["tenant_id"] == "acme"

    @pytest.mark.asyncio
    async def test_classifier_rules_see_hidden_arguments_and_annotations(self):
        from promptise.approval import CallbackApprovalHandler
        from promptise.approval_classifier import ApprovalRule, AutoApprovalClassifier

        asked: list[ApprovalRequest] = []

        async def human(request):
            asked.append(request)
            return ApprovalDecision(approved=True)

        classifier = AutoApprovalClassifier(
            deny_rules=[ApprovalRule(tool="refund", argument_contains='"order_id": "VIP-')],
            fallback=CallbackApprovalHandler(human),
        )
        server = MCPServer(name="gated")
        server.add_middleware(ApprovalGateMiddleware(classifier, include_arguments=False))

        @server.tool(requires_approval=True, destructive_hint=True)
        async def refund(order_id: str) -> str:
            """Refund an order."""
            return f"refunded {order_id}"

        client = TestClient(server)
        denied = await client.call_tool("refund", {"order_id": "VIP-1"})
        assert "APPROVAL_DENIED" in denied[0].text
        ok = await client.call_tool("refund", {"order_id": "o1"})
        assert ok[0].text == "refunded o1"
        # The human saw neither the arguments nor the raw copy
        [request] = asked
        assert request.arguments == {}
        assert request.raw_arguments is None
        assert request.tool_annotations == {"destructiveHint": True}

    def test_config_validation(self):
        with pytest.raises(ValueError, match="on_timeout"):
            ApprovalGateMiddleware(lambda r: True, on_timeout="shrug")
        with pytest.raises(ValueError, match="timeout"):
            ApprovalGateMiddleware(lambda r: True, timeout=0)


class TestUngatedDeclarationFailsLoudly:
    def test_build_raises_without_gate(self):
        server = MCPServer(name="misconfigured")

        @server.tool(requires_approval=True)
        async def refund(order_id: str) -> str:
            """Refund."""
            return "ok"

        with pytest.raises(RuntimeError, match="ApprovalGateMiddleware"):
            server._build_lowlevel_server()

    @pytest.mark.asyncio
    async def test_testclient_raises_without_gate(self):
        server = MCPServer(name="misconfigured")

        @server.tool(requires_approval=True)
        async def refund(order_id: str) -> str:
            """Refund."""
            return "ok"

        with pytest.raises(RuntimeError, match="ApprovalGateMiddleware"):
            await TestClient(server).call_tool("refund", {"order_id": "o1"})


class TestPendingApprover:
    def _build(self) -> tuple[MCPServer, PendingApprover]:
        server = MCPServer(name="pending")
        server.add_middleware(
            AuthMiddleware(
                APIKeyAuth(
                    keys={
                        "sk-caller": {"client_id": "caller-1"},
                        "sk-approver": {"client_id": "human-1", "roles": ["approver"]},
                    }
                )
            )
        )
        approver = PendingApprover(server)
        server.add_middleware(ApprovalGateMiddleware(approver, timeout=5.0))

        @server.tool(auth=True, requires_approval=True)
        async def refund(order_id: str) -> str:
            """Refund."""
            return f"refunded {order_id}"

        return server, approver

    @pytest.mark.asyncio
    async def test_block_list_approve_flow(self):
        server, approver = self._build()
        client = TestClient(server)

        call = asyncio.create_task(
            client.call_tool("refund", {"order_id": "o1"}, headers={"x-api-key": "sk-caller"})
        )
        # Wait until the request lands in the pending store
        for _ in range(100):
            if approver.pending():
                break
            await asyncio.sleep(0.01)
        pending = approver.pending()
        assert len(pending) == 1
        assert pending[0]["tool"] == "refund"
        assert pending[0]["client_id"] == "caller-1"

        # A human with the approver role releases it via the admin tool
        decided = await client.call_tool(
            "approvals_decide",
            {"request_id": pending[0]["request_id"], "approve": True},
            headers={"x-api-key": "sk-approver"},
        )
        assert "true" in decided[0].text.lower() or "resolved" in decided[0].text

        result = await asyncio.wait_for(call, timeout=5)
        assert result[0].text == "refunded o1"
        assert approver.pending() == []

    @pytest.mark.asyncio
    async def test_deny_flow(self):
        server, approver = self._build()
        client = TestClient(server)

        call = asyncio.create_task(
            client.call_tool("refund", {"order_id": "o2"}, headers={"x-api-key": "sk-caller"})
        )
        for _ in range(100):
            if approver.pending():
                break
            await asyncio.sleep(0.01)
        rid = approver.pending()[0]["request_id"]
        assert approver.decide(rid, False, reviewer_id="human-1", reason="fraud check")

        result = await asyncio.wait_for(call, timeout=5)
        assert "APPROVAL_DENIED" in result[0].text
        assert "fraud check" in result[0].text

    @pytest.mark.asyncio
    async def test_admin_tools_are_role_guarded(self):
        server, _ = self._build()
        client = TestClient(server)

        # The caller key lacks the approver role → guard denies
        result = await client.call_tool("approvals_list", {}, headers={"x-api-key": "sk-caller"})
        assert "[]" not in result[0].text  # not an empty-list success
        assert "approver" in result[0].text or "denied" in result[0].text.lower()

        # The approver key lists fine (empty)
        ok = await client.call_tool("approvals_list", {}, headers={"x-api-key": "sk-approver"})
        assert ok[0].text == "[]"

    @pytest.mark.asyncio
    async def test_decide_unknown_request(self):
        server, approver = self._build()
        assert approver.decide("nope", True) is False

    @pytest.mark.asyncio
    async def test_queue_full_denies_immediately(self):
        from promptise.approval import ApprovalRequest

        approver = PendingApprover(max_pending=1)
        req1 = ApprovalRequest(request_id="r1", tool_name="t", arguments={})
        req2 = ApprovalRequest(request_id="r2", tool_name="t", arguments={})

        waiting = asyncio.create_task(approver.request_approval(req1))
        for _ in range(100):
            if approver.pending():
                break
            await asyncio.sleep(0.01)

        decision = await approver.request_approval(req2)
        assert decision.approved is False
        assert "full" in (decision.reason or "")

        approver.decide("r1", True)
        assert (await asyncio.wait_for(waiting, timeout=5)).approved is True


class TestPendingApproverPerClientCap:
    """One client or tenant must not be able to fill the shared pending store
    and deny every other caller's gated call (a fail-closed DoS)."""

    @staticmethod
    def _request(rid: str, client: str | None, **metadata) -> ApprovalRequest:
        return ApprovalRequest(
            request_id=rid,
            tool_name="refund",
            arguments={},
            caller_user_id=client,
            metadata=metadata,
        )

    @staticmethod
    async def _park(approver: PendingApprover, request) -> asyncio.Task:
        task = asyncio.create_task(approver.request_approval(request))
        for _ in range(200):
            if request.request_id in approver._pending:
                break
            await asyncio.sleep(0.005)
        assert request.request_id in approver._pending
        return task

    def test_validation(self):
        with pytest.raises(ValueError, match="max_pending_per_client"):
            PendingApprover(max_pending=10, max_pending_per_client=0)
        with pytest.raises(ValueError, match="max_pending_per_client"):
            PendingApprover(max_pending=10, max_pending_per_client=11)
        # unlimited per client is the default
        assert PendingApprover()._max_pending_per_client is None

    @pytest.mark.asyncio
    async def test_one_client_at_cap_does_not_block_another(self):
        approver = PendingApprover(max_pending=100, max_pending_per_client=2)

        parked = [
            await self._park(approver, self._request("a1", "alice")),
            await self._park(approver, self._request("a2", "alice")),
        ]
        # alice is at her cap → denied immediately, naming the per-client cap
        denied = await approver.request_approval(self._request("a3", "alice"))
        assert denied.approved is False
        assert "per-client" in (denied.reason or "")
        assert "'alice'" in (denied.reason or "")
        assert "2 per client" in (denied.reason or "")
        assert len(approver.pending()) == 2

        # bob is unaffected and his request is parked in the store
        bob = await self._park(approver, self._request("b1", "bob"))
        assert approver.pending_for("bob") == 1
        assert approver.decide("b1", True, reviewer_id="human")
        assert (await asyncio.wait_for(bob, timeout=5)).approved is True

        # once one of alice's requests is decided her slot frees up
        assert approver.decide("a1", False, reviewer_id="human")
        await asyncio.wait_for(parked[0], timeout=5)
        again = await self._park(approver, self._request("a4", "alice"))
        assert approver.pending_for("alice") == 2
        for rid in ("a2", "a4"):
            approver.decide(rid, True, reviewer_id="human")
        await asyncio.gather(parked[1], again)

    @pytest.mark.asyncio
    async def test_global_cap_still_applies(self):
        approver = PendingApprover(max_pending=2, max_pending_per_client=2)
        parked = [
            await self._park(approver, self._request("a1", "alice")),
            await self._park(approver, self._request("b1", "bob")),
        ]
        denied = await approver.request_approval(self._request("c1", "carol"))
        assert denied.approved is False
        assert "full" in (denied.reason or "")
        for rid in ("a1", "b1"):
            approver.decide(rid, True)
        await asyncio.gather(*parked)

    @pytest.mark.asyncio
    async def test_anonymous_callers_share_one_bucket(self):
        approver = PendingApprover(max_pending=100, max_pending_per_client=1)
        parked = await self._park(approver, self._request("x1", None))
        denied = await approver.request_approval(self._request("x2", None))
        assert denied.approved is False
        assert "anonymous" in (denied.reason or "")
        # an identified caller is still served
        named = await self._park(approver, self._request("y1", "yuki"))
        approver.decide("x1", True)
        approver.decide("y1", True)
        await asyncio.gather(parked, named)

    def test_client_key_fallbacks(self):
        from promptise.approval import ApprovalRequest

        key = PendingApprover.client_key
        assert key(self._request("r", "alice", tenant_id="acme")) == "alice"
        assert key(self._request("r", None, client_id="svc", tenant_id="acme")) == "svc"
        assert key(self._request("r", None, tenant_id="acme")) == "acme"
        assert (
            key(ApprovalRequest(request_id="r", tool_name="t", arguments={}, agent_id="ag")) == "ag"
        )
        assert key(ApprovalRequest(request_id="r", tool_name="t", arguments={})) == ""

    @pytest.mark.asyncio
    async def test_cap_enforced_on_live_server_per_authenticated_client(self):
        server = MCPServer(name="pending-cap")
        server.add_middleware(
            AuthMiddleware(
                APIKeyAuth(
                    keys={
                        "sk-alice": {"client_id": "alice"},
                        "sk-bob": {"client_id": "bob"},
                    }
                )
            )
        )
        approver = PendingApprover(server, max_pending=100, max_pending_per_client=1)
        server.add_middleware(ApprovalGateMiddleware(approver, timeout=5.0))

        @server.tool(auth=True, requires_approval=True)
        async def refund(order_id: str) -> str:
            """Refund."""
            return f"refunded {order_id}"

        client = TestClient(server)
        first = asyncio.create_task(
            client.call_tool("refund", {"order_id": "o1"}, headers={"x-api-key": "sk-alice"})
        )
        for _ in range(200):
            if approver.pending():
                break
            await asyncio.sleep(0.005)

        # alice's second call is denied at once; bob's is parked normally
        second = await client.call_tool(
            "refund", {"order_id": "o2"}, headers={"x-api-key": "sk-alice"}
        )
        assert "APPROVAL_DENIED" in second[0].text
        assert "per-client" in second[0].text
        bob = asyncio.create_task(
            client.call_tool("refund", {"order_id": "o3"}, headers={"x-api-key": "sk-bob"})
        )
        for _ in range(200):
            if approver.pending_for("bob"):
                break
            await asyncio.sleep(0.005)
        assert approver.pending_for("bob") == 1

        for entry in approver.pending():
            approver.decide(entry["request_id"], True, reviewer_id="human")
        assert (await asyncio.wait_for(first, 5))[0].text == "refunded o1"
        assert (await asyncio.wait_for(bob, 5))[0].text == "refunded o3"


class TestPendingApproverPerTenantCap:
    """The per-client cap counts client ids; a tenant holding several API keys
    could still take the whole store. The per-tenant cap counts the tenant."""

    _request = staticmethod(TestPendingApproverPerClientCap._request)
    _park = staticmethod(TestPendingApproverPerClientCap._park)

    def test_validation(self):
        with pytest.raises(ValueError, match="max_pending_per_tenant"):
            PendingApprover(max_pending=10, max_pending_per_tenant=0)
        with pytest.raises(ValueError, match="max_pending_per_tenant"):
            PendingApprover(max_pending=10, max_pending_per_tenant=11)
        # the caps nest: client <= tenant <= store
        with pytest.raises(ValueError, match=r"max_pending_per_tenant \(5\), got 6"):
            PendingApprover(max_pending=10, max_pending_per_tenant=5, max_pending_per_client=6)
        approver = PendingApprover(
            max_pending=10, max_pending_per_tenant=5, max_pending_per_client=5
        )
        assert approver._max_pending_per_tenant == 5
        assert PendingApprover()._max_pending_per_tenant is None  # unlimited by default

    def test_tenant_key(self):
        key = PendingApprover.tenant_key
        assert key(self._request("r", "alice", tenant_id="acme")) == "acme"
        assert key(self._request("r", "alice", client_id="svc")) == ""  # no tenant: one bucket
        assert key(ApprovalRequest(request_id="r", tool_name="t", arguments={})) == ""

    @pytest.mark.asyncio
    async def test_one_tenant_at_cap_across_several_clients_does_not_block_another(self):
        approver = PendingApprover(max_pending=100, max_pending_per_tenant=2)

        parked = [
            await self._park(approver, self._request("a1", "alice", tenant_id="acme")),
            await self._park(approver, self._request("a2", "amir", tenant_id="acme")),
        ]
        # a third client id of the same tenant is denied at once, naming the tenant cap
        denied = await approver.request_approval(self._request("a3", "anna", tenant_id="acme"))
        assert denied.approved is False
        assert "per-tenant" in (denied.reason or "")
        assert "'acme'" in (denied.reason or "") and "2 per tenant" in (denied.reason or "")
        assert approver.pending_for_tenant("acme") == 2 and len(approver.pending()) == 2

        # another tenant is unaffected and parked normally
        bob = await self._park(approver, self._request("b1", "bob", tenant_id="globex"))
        assert approver.pending_for_tenant("globex") == 1
        assert approver.decide("b1", True, reviewer_id="human")
        assert (await asyncio.wait_for(bob, timeout=5)).approved is True

        # a decided request frees the tenant's slot
        assert approver.decide("a1", False, reviewer_id="human")
        await asyncio.wait_for(parked[0], timeout=5)
        again = await self._park(approver, self._request("a4", "anna", tenant_id="acme"))
        assert approver.pending_for_tenant("acme") == 2
        for rid in ("a2", "a4"):
            approver.decide(rid, True, reviewer_id="human")
        await asyncio.gather(parked[1], again)

    @pytest.mark.asyncio
    async def test_global_cap_still_applies(self):
        approver = PendingApprover(max_pending=2, max_pending_per_tenant=2)
        parked = [
            await self._park(approver, self._request("a1", "alice", tenant_id="acme")),
            await self._park(approver, self._request("b1", "bob", tenant_id="globex")),
        ]
        denied = await approver.request_approval(self._request("c1", "carol", tenant_id="initech"))
        assert denied.approved is False
        assert "full" in (denied.reason or "")
        for rid in ("a1", "b1"):
            approver.decide(rid, True)
        await asyncio.gather(*parked)

    @pytest.mark.asyncio
    async def test_requests_without_a_tenant_share_one_bucket(self):
        approver = PendingApprover(max_pending=100, max_pending_per_tenant=1)
        parked = await self._park(approver, self._request("x1", "xena"))
        denied = await approver.request_approval(self._request("x2", "yuki"))
        assert denied.approved is False
        assert "without a tenant" in (denied.reason or "")
        named = await self._park(approver, self._request("z1", "zoe", tenant_id="acme"))
        approver.decide("x1", True)
        approver.decide("z1", True)
        await asyncio.gather(parked, named)

    @pytest.mark.asyncio
    async def test_cap_enforced_on_live_server_per_tenant(self):
        """The gate stamps ``tenant_id`` from the authenticated client; the cap counts it."""
        server = MCPServer(name="pending-tenant-cap")
        server.add_middleware(
            AuthMiddleware(
                APIKeyAuth(
                    keys={
                        "sk-a1": {"client_id": "acme-1", "tenant_id": "acme"},
                        "sk-a2": {"client_id": "acme-2", "tenant_id": "acme"},
                        "sk-b": {"client_id": "globex-1", "tenant_id": "globex"},
                    }
                )
            )
        )
        approver = PendingApprover(server, max_pending=100, max_pending_per_tenant=1)
        server.add_middleware(ApprovalGateMiddleware(approver, timeout=5.0))

        @server.tool(auth=True, requires_approval=True)
        async def refund(order_id: str) -> str:
            """Refund."""
            return f"refunded {order_id}"

        client = TestClient(server)
        first = asyncio.create_task(
            client.call_tool("refund", {"order_id": "o1"}, headers={"x-api-key": "sk-a1"})
        )
        for _ in range(200):
            if approver.pending():
                break
            await asyncio.sleep(0.005)

        # acme's second key is denied at once; globex is parked normally
        second = await client.call_tool(
            "refund", {"order_id": "o2"}, headers={"x-api-key": "sk-a2"}
        )
        assert "APPROVAL_DENIED" in second[0].text and "per-tenant" in second[0].text
        other = asyncio.create_task(
            client.call_tool("refund", {"order_id": "o3"}, headers={"x-api-key": "sk-b"})
        )
        for _ in range(200):
            if approver.pending_for_tenant("globex"):
                break
            await asyncio.sleep(0.005)
        assert approver.pending_for_tenant("globex") == 1

        for entry in approver.pending():
            approver.decide(entry["request_id"], True, reviewer_id="human")
        assert (await asyncio.wait_for(first, 5))[0].text == "refunded o1"
        assert (await asyncio.wait_for(other, 5))[0].text == "refunded o3"


class TestElicitationApprover:
    @pytest.mark.asyncio
    async def test_no_session_fails_closed(self):
        # TestClient has no MCP session → the approver must deny, not allow
        client = TestClient(_server_with_gate(ElicitationApprover()))
        result = await client.call_tool("refund", {"order_id": "o1"})
        assert "APPROVAL_DENIED" in result[0].text
        assert "refunded" not in result[0].text

    @staticmethod
    def _ctx_with_session(session) -> RequestContext:
        ctx = RequestContext(server_name="s", tool_name="refund")
        ctx.client = ClientContext(client_id="c1")
        ctx.state["_mcp_session"] = session
        return ctx

    @pytest.mark.asyncio
    async def test_live_session_approve_and_deny(self):
        from unittest.mock import AsyncMock

        from mcp.types import ElicitResult

        from promptise.approval import ApprovalRequest

        approver = ElicitationApprover()
        req = ApprovalRequest(request_id="r1", tool_name="refund", arguments={"o": 1})

        session = AsyncMock()
        session.elicit = AsyncMock(
            return_value=ElicitResult(action="accept", content={"approve": True, "reason": "ok"})
        )
        decision = await approver.request_approval_ctx(req, self._ctx_with_session(session))
        assert decision.approved is True
        assert decision.reviewer_id == "elicitation:client-user"
        assert decision.reason == "ok"
        # The SDK's form-mode API was called with its real keyword names
        session.elicit.assert_awaited_once()
        kwargs = session.elicit.await_args.kwargs
        assert "refund" in kwargs["message"]
        assert kwargs["requestedSchema"]["required"] == ["approve"]

        session.elicit = AsyncMock(
            return_value=ElicitResult(action="accept", content={"approve": False, "reason": "no"})
        )
        decision = await approver.request_approval_ctx(req, self._ctx_with_session(session))
        assert decision.approved is False
        assert decision.reason == "no"

    @pytest.mark.asyncio
    async def test_decline_cancel_and_failure_deny(self):
        from unittest.mock import AsyncMock

        from mcp.shared.exceptions import McpError
        from mcp.types import INVALID_REQUEST, ElicitResult, ErrorData

        from promptise.approval import ApprovalRequest

        approver = ElicitationApprover()
        req = ApprovalRequest(request_id="r1", tool_name="refund", arguments={})

        outcomes = [
            ElicitResult(action="decline"),
            ElicitResult(action="cancel"),
            # a client that accepts but sends the approval flag as a string
            ElicitResult(action="accept", content={"approve": "yes"}),
        ]
        for outcome in outcomes:
            session = AsyncMock()
            session.elicit = AsyncMock(return_value=outcome)
            decision = await approver.request_approval_ctx(req, self._ctx_with_session(session))
            assert decision.approved is False, outcome
            assert decision.reviewer_id == "elicitation"

        # the client rejected the request (no elicitation capability)
        session = AsyncMock()
        session.elicit = AsyncMock(
            side_effect=McpError(
                ErrorData(code=INVALID_REQUEST, message="Elicitation not supported")
            )
        )
        decision = await approver.request_approval_ctx(req, self._ctx_with_session(session))
        assert decision.approved is False

    def test_sdk_contract_elicit_method_exists(self):
        """A renamed SDK method must fail this test, not silently deny every
        gated call (the bug: the approver called a method that never existed
        on ServerSession and swallowed the AttributeError)."""
        import inspect

        from mcp.server.session import ServerSession

        assert hasattr(ServerSession, "elicit")
        signature = inspect.signature(ServerSession.elicit)
        # Elicitor.ask() passes exactly these keyword arguments
        signature.bind(
            None,
            message="Approve?",
            requestedSchema={"type": "object", "properties": {}},
            related_request_id="req-1",
        )


class TestElicitationRoundTrip:
    """Drive the real MCP SDK client ↔ server in-process over its in-memory
    transport: the gated tool must only run when the human behind the
    client accepts the elicitation."""

    @staticmethod
    def _server(timeout: float = 5.0) -> MCPServer:
        server = MCPServer(name="gated-elicit")
        server.add_middleware(ApprovalGateMiddleware(ElicitationApprover(), timeout=timeout))

        @server.tool(requires_approval=True)
        async def refund(order_id: str) -> str:
            """Refund an order."""
            return f"refunded {order_id}"

        return server

    @staticmethod
    async def _call(server: MCPServer, elicitation_callback=None) -> str:
        from mcp.shared.memory import create_connected_server_and_client_session

        lowlevel = server._build_lowlevel_server()
        async with create_connected_server_and_client_session(
            lowlevel, elicitation_callback=elicitation_callback
        ) as client:
            result = await client.call_tool("refund", {"order_id": "o1"})
            return result.content[0].text

    @pytest.mark.asyncio
    async def test_client_accepts_then_tool_runs(self):
        from mcp.types import ElicitResult

        prompts = []

        async def accept(context, params):
            prompts.append(params)
            return ElicitResult(action="accept", content={"approve": True, "reason": "sure"})

        assert await self._call(self._server(), accept) == "refunded o1"
        assert len(prompts) == 1
        assert "refund" in prompts[0].message
        assert prompts[0].requestedSchema["required"] == ["approve"]

    @pytest.mark.asyncio
    async def test_client_declines_then_denied(self):
        from mcp.types import ElicitResult

        async def decline(context, params):
            return ElicitResult(action="decline")

        text = await self._call(self._server(), decline)
        assert "APPROVAL_DENIED" in text
        assert "refunded" not in text

    @pytest.mark.asyncio
    async def test_client_without_elicitation_support_is_denied(self):
        # No callback → the SDK client answers "Elicitation not supported"
        text = await self._call(self._server(), None)
        assert "APPROVAL_DENIED" in text
        assert "refunded" not in text


class TestApprovalRunsAfterGuards:
    """The gate must reject a caller the tool's guards deny BEFORE requesting
    approval — else unauthorized callers spam reviewers and fill the queue."""

    def _build(self):
        from promptise.mcp.server._guards import HasRole

        calls = {"approvals": 0}

        async def counting_handler(request):
            calls["approvals"] += 1
            return ApprovalDecision(approved=True)

        server = MCPServer(name="guarded-gate")
        server.add_middleware(
            AuthMiddleware(
                APIKeyAuth(
                    keys={
                        "sk-viewer": {"client_id": "v1", "roles": ["viewer"]},
                        "sk-admin": {"client_id": "a1", "roles": ["admin"]},
                    }
                )
            )
        )
        server.add_middleware(ApprovalGateMiddleware(counting_handler, timeout=5.0))

        @server.tool(auth=True, guards=[HasRole("admin")], requires_approval=True)
        async def refund(order_id: str) -> str:
            """Refund — admin only."""
            return f"refunded {order_id}"

        return server, calls

    @pytest.mark.asyncio
    async def test_unauthorized_caller_never_triggers_approval(self):
        server, calls = self._build()
        client = TestClient(server)

        # viewer lacks the admin role → denied BEFORE any approval is requested
        denied = await client.call_tool(
            "refund", {"order_id": "o1"}, headers={"x-api-key": "sk-viewer"}
        )
        assert "ACCESS_DENIED" in denied[0].text
        assert "refunded" not in denied[0].text
        assert calls["approvals"] == 0  # the human was never bothered

    @pytest.mark.asyncio
    async def test_authorized_caller_still_gated(self):
        server, calls = self._build()
        client = TestClient(server)

        ok = await client.call_tool("refund", {"order_id": "o1"}, headers={"x-api-key": "sk-admin"})
        assert ok[0].text == "refunded o1"
        assert calls["approvals"] == 1


class TestApprovalSurvivesComposition:
    """requires_approval must not be silently dropped by include_router/mount —
    the build-time invariant has to fire for composed servers too."""

    def test_included_router_gated_tool_requires_gate(self):
        from promptise.mcp.server import MCPRouter

        router = MCPRouter(prefix="billing")

        @router.tool(requires_approval=True)
        async def refund(order_id: str) -> str:
            """Refund."""
            return "ok"

        server = MCPServer(name="composed")
        server.include_router(router)
        # The flag survived composition → the ungated-build invariant fires
        assert server._tool_registry.get("billing_refund").requires_approval is True
        with pytest.raises(RuntimeError, match="ApprovalGateMiddleware"):
            server._build_lowlevel_server()

    @pytest.mark.asyncio
    async def test_included_router_gated_tool_enforced_with_gate(self):
        from promptise.mcp.server import MCPRouter

        seen = {"n": 0}

        async def deny(request):
            seen["n"] += 1
            return ApprovalDecision(approved=False, reason="no")

        router = MCPRouter(prefix="billing")

        @router.tool(requires_approval=True)
        async def refund(order_id: str) -> str:
            """Refund."""
            return "refunded"

        server = MCPServer(name="composed")
        server.include_router(router)
        server.add_middleware(ApprovalGateMiddleware(deny))

        # ... and actually enforces at call time
        result = await TestClient(server).call_tool("billing_refund", {"order_id": "o1"})
        assert "APPROVAL_DENIED" in result[0].text
        assert seen["n"] == 1

    def test_mounted_gated_tool_requires_gate(self):
        from promptise.mcp.server import mount

        child = MCPServer(name="ops")

        @child.tool(requires_approval=True)
        async def delete_all() -> str:
            """Delete everything."""
            return "gone"

        parent = MCPServer(name="parent")
        mount(parent, child, prefix="ops")
        assert parent._tool_registry.get("ops_delete_all").requires_approval is True
        with pytest.raises(RuntimeError, match="ApprovalGateMiddleware"):
            parent._build_lowlevel_server()


class TestSeparationOfDuties:
    """approvals_decide must reject self-approval (four-eyes)."""

    @pytest.mark.asyncio
    async def test_caller_cannot_approve_own_request(self):
        server = MCPServer(name="sod")
        server.add_middleware(
            AuthMiddleware(
                APIKeyAuth(
                    keys={
                        # caller ALSO holds the approver role
                        "sk-dual": {"client_id": "dana", "roles": ["approver"]},
                        "sk-other": {"client_id": "eve", "roles": ["approver"]},
                    }
                )
            )
        )
        approver = PendingApprover(server)
        server.add_middleware(ApprovalGateMiddleware(approver, timeout=5.0))

        @server.tool(auth=True, requires_approval=True)
        async def refund(order_id: str) -> str:
            """Refund."""
            return f"refunded {order_id}"

        client = TestClient(server)
        call = asyncio.create_task(
            client.call_tool("refund", {"order_id": "o1"}, headers={"x-api-key": "sk-dual"})
        )
        for _ in range(100):
            if approver.pending():
                break
            await asyncio.sleep(0.01)
        rid = approver.pending()[0]["request_id"]

        # dana (the caller) tries to approve her own request → refused
        self_decide = await client.call_tool(
            "approvals_decide",
            {"request_id": rid, "approve": True},
            headers={"x-api-key": "sk-dual"},
        )
        assert "four-eyes" in self_decide[0].text or "your own" in self_decide[0].text
        assert approver.pending()  # still pending — not resolved

        # a DIFFERENT approver can release it
        other = await client.call_tool(
            "approvals_decide",
            {"request_id": rid, "approve": True},
            headers={"x-api-key": "sk-other"},
        )
        assert "resolved" in other[0].text.lower() or "true" in other[0].text.lower()
        assert (await asyncio.wait_for(call, timeout=5))[0].text == "refunded o1"


class TestReviewerIdentityOnLivePath:
    """Regression for the test/prod divergence: SoD + reviewer attribution
    must work on the REAL transport closure, not only under TestClient.
    Previously approvals_decide read reviewer from an un-injected ctx param,
    so on live it was always 'unknown-reviewer' and SoD was bypassed."""

    async def _live_call(self, server, name, args, api_key):
        import mcp.types as t

        from promptise.mcp.server._context import set_request_headers

        ll = server._build_lowlevel_server()
        handler = ll.request_handlers[t.CallToolRequest]
        set_request_headers({"x-api-key": api_key})
        try:
            req = t.CallToolRequest(
                method="tools/call",
                params=t.CallToolRequestParams(name=name, arguments=args),
            )
            res = await handler(req)
            return res.root.content[0].text
        finally:
            set_request_headers({})

    @pytest.mark.asyncio
    async def test_self_approval_rejected_on_live_transport(self):
        server = MCPServer(name="live-sod")
        server.add_middleware(
            AuthMiddleware(
                APIKeyAuth(
                    keys={
                        "sk-dana": {"client_id": "dana", "roles": ["approver"]},
                        "sk-eve": {"client_id": "eve", "roles": ["approver"]},
                    }
                )
            )
        )
        approver = PendingApprover(server)
        server.add_middleware(ApprovalGateMiddleware(approver, timeout=5.0))

        @server.tool(auth=True, requires_approval=True)
        async def refund(order_id: str) -> str:
            """Refund."""
            return f"refunded {order_id}"

        call = asyncio.create_task(self._live_call(server, "refund", {"order_id": "o1"}, "sk-dana"))
        for _ in range(200):
            if approver.pending():
                break
            await asyncio.sleep(0.01)
        rid = approver.pending()[0]["request_id"]

        # dana (the caller) tries to self-approve on the LIVE path
        self_decide = await self._live_call(
            server, "approvals_decide", {"request_id": rid, "approve": True}, "sk-dana"
        )
        assert "four-eyes" in self_decide or "your own" in self_decide
        assert approver.pending()  # still pending

        # a different approver releases it, and the reviewer is correctly 'eve'
        pend_before = approver._pending[rid][1]
        await self._live_call(
            server, "approvals_decide", {"request_id": rid, "approve": True}, "sk-eve"
        )
        decision = pend_before.result()
        assert decision.reviewer_id == "eve"  # NOT 'unknown-reviewer'
        assert (await asyncio.wait_for(call, timeout=5)) == "refunded o1"


class TestHandlerCrashFailsClosed:
    @pytest.mark.asyncio
    async def test_raising_handler_denies(self):
        async def boom(request):
            raise RuntimeError("approval backend down")

        client = TestClient(_server_with_gate(boom))
        result = await client.call_tool("refund", {"order_id": "o1"})
        # Fail closed: the handler crash must NOT let the call through
        assert "refunded" not in result[0].text


class TestLiveContextParamInjection:
    """Regression for the framework gap that caused #7: a ctx: RequestContext
    parameter must be populated on the live transport path, not only under
    TestClient."""

    @pytest.mark.asyncio
    async def test_ctx_param_injected_on_live_path(self):
        import mcp.types as t

        server = MCPServer(name="ctx-probe")

        @server.tool()
        async def whoami(x: int, ctx: RequestContext) -> str:
            """Return whether ctx was injected."""
            return "injected" if ctx is not None else "NONE"

        ll = server._build_lowlevel_server()
        handler = ll.request_handlers[t.CallToolRequest]
        req = t.CallToolRequest(
            method="tools/call",
            params=t.CallToolRequestParams(name="whoami", arguments={"x": 1}),
        )
        res = await handler(req)
        assert res.root.content[0].text == "injected"


class TestRouterLevelGate:
    """A gate installed at ROUTER level covers its gated tools — the build
    invariant must not falsely reject a validly-gated composition."""

    def test_router_level_gate_builds(self):
        from promptise.mcp.server import MCPRouter

        async def handler(request):
            return ApprovalDecision(approved=False, reason="no")

        router = MCPRouter(prefix="r", middleware=[ApprovalGateMiddleware(handler)])

        @router.tool(requires_approval=True)
        async def wipe(x: int) -> int:
            """Wipe."""
            return x

        server = MCPServer(name="rr")
        server.include_router(router)
        # Must NOT raise — the gate is in router_middleware, compiled per-tool
        server._build_lowlevel_server()

    @pytest.mark.asyncio
    async def test_router_level_gate_enforces_via_testclient(self):
        from promptise.mcp.server import MCPRouter

        async def deny(request):
            return ApprovalDecision(approved=False, reason="nope")

        router = MCPRouter(prefix="r", middleware=[ApprovalGateMiddleware(deny)])

        @router.tool(requires_approval=True)
        async def wipe(x: int) -> int:
            """Wipe."""
            return x

        server = MCPServer(name="rr")
        server.include_router(router)
        result = await TestClient(server).call_tool("r_wipe", {"x": 1})
        assert "APPROVAL_DENIED" in result[0].text


class TestOptionalRequestContextInjection:
    """`ctx: RequestContext = None` gets implicit-Optional'd by Python 3.10's
    get_type_hints (removed in 3.11). Injection must recognise the Optional /
    union / string forms on every supported Python, else the param stays None
    on 3.10 and handlers reading ctx.client crash (regression: a tenancy test
    failed only on the 3.10 CI matrix)."""

    def test_wants_request_context_forms(self):
        from typing import Optional

        from promptise.mcp.server._context import RequestContext as RC
        from promptise.mcp.server._context import _wants_request_context

        assert _wants_request_context(RC)
        assert _wants_request_context(Optional[RC])  # 3.10 implicit-Optional shape
        assert _wants_request_context(RC | None)
        assert _wants_request_context("RequestContext")
        assert not _wants_request_context(int)

    @pytest.mark.asyncio
    async def test_optional_ctx_param_injected_on_live_path(self):
        import mcp.types as mt

        # `RequestContext` is a MODULE-level import (test handlers resolve type
        # hints against module globals, not local aliases). `... | None = None`
        # is the exact shape 3.10 produced implicitly for `ctx: RequestContext
        # = None` — inject_context must populate it, else ctx is None on 3.10.
        server = MCPServer(name="opt-ctx")

        @server.tool()
        async def whoami(x: int, ctx: RequestContext | None = None) -> str:
            """Return whether ctx was injected."""
            return "injected" if ctx is not None else "NONE"

        ll = server._build_lowlevel_server()
        handler = ll.request_handlers[mt.CallToolRequest]
        res = await handler(
            mt.CallToolRequest(
                method="tools/call",
                params=mt.CallToolRequestParams(name="whoami", arguments={"x": 1}),
            )
        )
        assert res.root.content[0].text == "injected"
