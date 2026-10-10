"""Tests for promptise.approval — Human-in-the-Loop approval system."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from promptise.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    CallbackApprovalHandler,
    QueueApprovalHandler,
    _ApprovalToolWrapper,
    _GateState,
    wrap_tools_with_approval,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class FakeTool:
    """Minimal tool mock for testing."""

    def __init__(self, name: str = "test_tool"):
        self.name = name
        self.description = f"A test tool called {name}"
        self.args_schema = None
        self._call_count = 0

    async def _arun(self, **kwargs):
        self._call_count += 1
        return f"Result from {self.name}: {kwargs}"


def make_policy(
    tools=None,
    handler=None,
    timeout=5.0,
    on_timeout="deny",
    **kwargs,
):
    """Create a policy with reasonable defaults for testing."""
    if handler is None:
        handler = CallbackApprovalHandler(AsyncMock(return_value=ApprovalDecision(approved=True)))
    return ApprovalPolicy(
        tools=tools or ["test_*"],
        handler=handler,
        timeout=timeout,
        on_timeout=on_timeout,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# ApprovalRequest
# ---------------------------------------------------------------------------


class TestApprovalRequest:
    def test_construction(self):
        req = ApprovalRequest(
            request_id="abc123",
            tool_name="send_email",
            arguments={"to": "alice@example.com"},
        )
        assert req.request_id == "abc123"
        assert req.tool_name == "send_email"
        assert req.arguments == {"to": "alice@example.com"}
        assert req.timeout == 300.0

    def test_to_dict(self):
        req = ApprovalRequest(
            request_id="abc",
            tool_name="test",
            arguments={"key": "value"},
        )
        d = req.to_dict()
        assert d["request_id"] == "abc"
        assert d["tool_name"] == "test"
        assert isinstance(d["timestamp"], float)

    def test_hmac_signature(self):
        req = ApprovalRequest(
            request_id="abc",
            tool_name="test",
            arguments={},
        )
        sig1 = req.compute_hmac("secret1")
        sig2 = req.compute_hmac("secret2")
        sig3 = req.compute_hmac("secret1")
        # Same secret → same signature
        assert sig1 == sig3
        # Different secret → different signature
        assert sig1 != sig2


# ---------------------------------------------------------------------------
# ApprovalDecision
# ---------------------------------------------------------------------------


class TestApprovalDecision:
    def test_approved(self):
        d = ApprovalDecision(approved=True, reviewer_id="alice")
        assert d.approved is True
        assert d.reviewer_id == "alice"

    def test_denied(self):
        d = ApprovalDecision(approved=False, reason="Not authorized for this action")
        assert d.approved is False
        assert d.reason == "Not authorized for this action"


# ---------------------------------------------------------------------------
# ApprovalPolicy
# ---------------------------------------------------------------------------


class TestApprovalPolicy:
    def test_glob_matching(self):
        policy = make_policy(tools=["send_*", "delete_*"])
        assert policy.requires_approval("send_email") is True
        assert policy.requires_approval("send_notification") is True
        assert policy.requires_approval("delete_user") is True
        assert policy.requires_approval("search_database") is False

    def test_exact_matching(self):
        policy = make_policy(tools=["send_email"])
        assert policy.requires_approval("send_email") is True
        assert policy.requires_approval("send_notification") is False

    def test_wildcard_all(self):
        policy = make_policy(tools=["*"])
        assert policy.requires_approval("anything") is True

    def test_no_match(self):
        policy = make_policy(tools=["payment_*"])
        assert policy.requires_approval("search") is False

    def test_validation_empty_tools(self):
        with pytest.raises(ValueError, match="at least one"):
            ApprovalPolicy(
                tools=[],
                handler=CallbackApprovalHandler(
                    AsyncMock(return_value=ApprovalDecision(approved=True))
                ),
            )

    def test_validation_negative_timeout(self):
        with pytest.raises(ValueError, match="positive"):
            make_policy(timeout=-1)

    def test_validation_excessive_timeout(self):
        with pytest.raises(ValueError, match="86400"):
            make_policy(timeout=100000)

    def test_callable_handler_wrapped(self):
        """A plain callable is wrapped in CallbackApprovalHandler."""

        async def my_handler(req):
            return ApprovalDecision(approved=True)

        policy = make_policy(handler=my_handler)
        assert isinstance(policy.handler, CallbackApprovalHandler)


# ---------------------------------------------------------------------------
# CallbackApprovalHandler
# ---------------------------------------------------------------------------


class TestCallbackHandler:
    @pytest.mark.asyncio
    async def test_async_callback(self):
        decision = ApprovalDecision(approved=True, reviewer_id="alice")

        async def handler(req):
            return decision

        h = CallbackApprovalHandler(handler)
        req = ApprovalRequest(request_id="abc", tool_name="test", arguments={})
        result = await h.request_approval(req)
        assert result.approved is True
        assert result.reviewer_id == "alice"

    @pytest.mark.asyncio
    async def test_bool_return(self):
        """Callback can return a plain bool for convenience."""

        async def handler(req):
            return True

        h = CallbackApprovalHandler(handler)
        req = ApprovalRequest(request_id="abc", tool_name="test", arguments={})
        result = await h.request_approval(req)
        assert result.approved is True

    @pytest.mark.asyncio
    async def test_invalid_return_type(self):
        async def handler(req):
            return "invalid"

        h = CallbackApprovalHandler(handler)
        req = ApprovalRequest(request_id="abc", tool_name="test", arguments={})
        with pytest.raises(TypeError, match="ApprovalDecision or bool"):
            await h.request_approval(req)


# ---------------------------------------------------------------------------
# QueueApprovalHandler
# ---------------------------------------------------------------------------


class TestQueueHandler:
    @pytest.mark.asyncio
    async def test_submit_and_resolve(self):
        handler = QueueApprovalHandler()
        req = ApprovalRequest(request_id="abc", tool_name="test", arguments={}, timeout=5.0)

        async def approver():
            # Wait for request to appear
            queued_req = await handler.request_queue.get()
            assert queued_req.request_id == "abc"
            # Submit decision
            handler.submit_decision("abc", ApprovalDecision(approved=True, reviewer_id="bob"))

        # Run both concurrently
        _, result = await asyncio.gather(
            approver(),
            handler.request_approval(req),
        )
        assert result.approved is True
        assert result.reviewer_id == "bob"

    @pytest.mark.asyncio
    async def test_timeout(self):
        handler = QueueApprovalHandler()
        req = ApprovalRequest(request_id="abc", tool_name="test", arguments={}, timeout=0.1)
        with pytest.raises(asyncio.TimeoutError):
            await handler.request_approval(req)

    @pytest.mark.asyncio
    async def test_invalid_request_id(self):
        handler = QueueApprovalHandler()
        with pytest.raises(KeyError, match="nonexistent"):
            handler.submit_decision("nonexistent", ApprovalDecision(approved=True))


# ---------------------------------------------------------------------------
# ApprovalToolWrapper
# ---------------------------------------------------------------------------


class TestApprovalToolWrapper:
    @pytest.mark.asyncio
    async def test_approved_executes(self):
        tool = FakeTool("send_email")
        policy = make_policy(tools=["send_*"])

        wrapper = _ApprovalToolWrapper(
            inner=tool,
            policy=policy,
        )
        result = await wrapper._arun(config={}, to="alice@example.com")
        assert "Result from send_email" in result
        assert tool._call_count == 1

    @pytest.mark.asyncio
    async def test_denied_returns_message(self):
        async def deny_handler(req):
            return ApprovalDecision(approved=False, reason="Not authorized")

        tool = FakeTool("send_email")
        policy = make_policy(
            tools=["send_*"],
            handler=CallbackApprovalHandler(deny_handler),
        )
        wrapper = _ApprovalToolWrapper(
            inner=tool,
            policy=policy,
        )
        result = await wrapper._arun(config={}, to="alice@example.com")
        assert "DENIED" in result
        assert "Not authorized" in result
        assert tool._call_count == 0

    @pytest.mark.asyncio
    async def test_timeout_deny(self):
        async def slow_handler(req):
            await asyncio.sleep(10)  # Will timeout
            return ApprovalDecision(approved=True)

        tool = FakeTool("send_email")
        policy = make_policy(
            tools=["send_*"],
            handler=CallbackApprovalHandler(slow_handler),
            timeout=0.1,
            on_timeout="deny",
        )
        wrapper = _ApprovalToolWrapper(
            inner=tool,
            policy=policy,
        )
        result = await wrapper._arun(config={})
        assert "DENIED" in result
        assert tool._call_count == 0

    @pytest.mark.asyncio
    async def test_timeout_allow(self):
        async def slow_handler(req):
            await asyncio.sleep(10)
            return ApprovalDecision(approved=True)

        tool = FakeTool("send_email")
        policy = make_policy(
            tools=["send_*"],
            handler=CallbackApprovalHandler(slow_handler),
            timeout=0.1,
            on_timeout="allow",
        )
        wrapper = _ApprovalToolWrapper(
            inner=tool,
            policy=policy,
        )
        result = await wrapper._arun(config={})
        assert "Result from send_email" in result
        assert tool._call_count == 1

    @pytest.mark.asyncio
    async def test_modified_arguments(self):
        async def modify_handler(req):
            return ApprovalDecision(
                approved=True,
                modified_arguments={"to": "bob@example.com"},
            )

        tool = FakeTool("send_email")
        policy = make_policy(
            tools=["send_*"],
            handler=CallbackApprovalHandler(modify_handler),
        )
        wrapper = _ApprovalToolWrapper(
            inner=tool,
            policy=policy,
        )
        result = await wrapper._arun(config={}, to="alice@example.com")
        assert "bob@example.com" in result  # Modified args used

    @pytest.mark.asyncio
    async def test_retry_after_deny_limit(self):
        async def deny(req):
            return ApprovalDecision(approved=False)

        tool = FakeTool("send_email")
        policy = make_policy(
            tools=["send_*"],
            handler=CallbackApprovalHandler(deny),
            max_retries_after_deny=2,
        )
        wrapper = _ApprovalToolWrapper(
            inner=tool,
            policy=policy,
        )

        # First denial — count becomes 1
        r1 = await wrapper._arun(config={})
        assert "DENIED" in r1
        assert "permanently" not in r1

        # Second denial — count becomes 2 (hits limit)
        r2 = await wrapper._arun(config={})
        assert "DENIED" in r2

        # Third attempt — count >= limit, denied without calling handler
        r3 = await wrapper._arun(config={})
        assert "already denied 2 times" in r3

    @pytest.mark.asyncio
    async def test_max_pending(self):
        tool = FakeTool("send_email")
        policy = make_policy(tools=["send_*"], max_pending=1)
        wrapper = _ApprovalToolWrapper(
            inner=tool,
            policy=policy,
            state=_GateState(policy, pending=1),  # Already at max
        )
        result = await wrapper._arun(config={})
        assert "Too many pending" in result

    @pytest.mark.asyncio
    async def test_handler_error(self):
        async def error_handler(req):
            raise RuntimeError("Connection failed")

        tool = FakeTool("send_email")
        policy = make_policy(
            tools=["send_*"],
            handler=CallbackApprovalHandler(error_handler),
        )
        wrapper = _ApprovalToolWrapper(
            inner=tool,
            policy=policy,
        )
        result = await wrapper._arun(config={})
        assert "DENIED" in result
        assert tool._call_count == 0


# ---------------------------------------------------------------------------
# wrap_tools_with_approval
# ---------------------------------------------------------------------------


class TestWrapToolsWithApproval:
    def test_wraps_matching_tools(self):
        tools = [FakeTool("send_email"), FakeTool("search_db"), FakeTool("delete_user")]
        policy = make_policy(tools=["send_*", "delete_*"])
        wrapped = wrap_tools_with_approval(tools, policy)

        assert len(wrapped) == 3
        assert isinstance(wrapped[0], _ApprovalToolWrapper)  # send_email
        assert not isinstance(wrapped[1], _ApprovalToolWrapper)  # search_db
        assert isinstance(wrapped[2], _ApprovalToolWrapper)  # delete_user

    def test_preserves_tool_names(self):
        tools = [FakeTool("send_email")]
        policy = make_policy(tools=["send_*"])
        wrapped = wrap_tools_with_approval(tools, policy)
        assert wrapped[0].name == "send_email"

    def test_no_matching_tools(self):
        tools = [FakeTool("search_db")]
        policy = make_policy(tools=["send_*"])
        wrapped = wrap_tools_with_approval(tools, policy)
        assert wrapped[0] is tools[0]  # Not wrapped

    def test_shared_state(self):
        """All wrappers share the pending count and denial counts."""
        tools = [FakeTool("send_a"), FakeTool("send_b")]
        policy = make_policy(tools=["send_*"])
        wrapped = wrap_tools_with_approval(tools, policy)

        w1, w2 = wrapped
        assert w1._state is w2._state


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


class TestRedaction:
    @pytest.mark.asyncio
    async def test_redact_disabled(self):
        policy = make_policy(redact_sensitive=False)
        args = {"password": "secret123"}
        result = await policy.redact_arguments(args)
        assert result["password"] == "secret123"

    @pytest.mark.asyncio
    async def test_redact_preserves_non_sensitive(self):
        policy = make_policy(redact_sensitive=True)
        args = {"query": "hello world"}
        result = await policy.redact_arguments(args)
        assert result["query"] == "hello world"


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocolCompliance:
    def test_callback_handler_is_approval_handler(self):
        from promptise.approval import ApprovalHandler

        handler = CallbackApprovalHandler(lambda r: True)
        assert isinstance(handler, ApprovalHandler)

    def test_queue_handler_is_approval_handler(self):
        from promptise.approval import ApprovalHandler

        handler = QueueApprovalHandler()
        assert isinstance(handler, ApprovalHandler)

    def test_custom_handler_satisfies_protocol(self):
        from promptise.approval import ApprovalHandler

        class MyHandler:
            async def request_approval(self, request):
                return ApprovalDecision(approved=True)

        assert isinstance(MyHandler(), ApprovalHandler)


# ---------------------------------------------------------------------------
# Import / export
# ---------------------------------------------------------------------------


class TestExports:
    def test_imports_from_promptise(self):
        from promptise import (
            ApprovalDecision,
            ApprovalPolicy,
            ApprovalRequest,
        )

        # All importable
        assert ApprovalPolicy is not None
        assert ApprovalRequest is not None
        assert ApprovalDecision is not None
