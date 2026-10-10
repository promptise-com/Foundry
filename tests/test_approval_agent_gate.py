"""Regression tests for the agent-side approval gate (``build_agent(approval=ApprovalPolicy(...))``).

Each class pins one reported defect:

1. approved plain LangChain tools (``@tool`` / ``StructuredTool``) crashed
   because the gate called ``_arun`` without ``config``;
2. redaction garbled order IDs (``A-1001`` → ``[MEDICAL]1001``) and destroyed
   the argument structure when a phone number was present;
3. denials were counted per tool name for the agent's lifetime, so three
   denials auto-denied the tool for every user forever;
4. ``agent_id`` / ``context_summary`` / ``metadata`` were always empty;
5. several approvals in one turn could not be serialized;
6. the webhook signature had no verification helper;
7. ``WebhookApprovalHandler`` refused private networks with no override;
8. ``modified_arguments`` replaced every argument;
9. the model was not told that a reviewer changed its arguments.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, PrivateAttr

from promptise import CallerContext, build_agent
from promptise.agent import _caller_ctx_var, _Invocation, _invocation_ctx_var, _session_ctx_var
from promptise.approval import (
    SIGNED_FIELDS,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    WebhookApprovalHandler,
    verify_webhook_signature,
    wrap_tools_with_approval,
)
from promptise.guardrails import PromptiseSecurityScanner

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@tool
async def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a customer."""
    return f"sent to {to}: {subject} / {body}"


@tool
def sync_send_email(to: str, subject: str) -> str:
    """Send an email (a synchronous @tool)."""
    return f"sync sent to {to}: {subject}"


class RefundArgs(BaseModel):
    order_id: str
    amount: float
    reason: str


class IssueRefund(BaseTool):
    """Shaped like the tools Promptise builds from MCP servers."""

    name: str = "issue_refund"
    description: str = "Refund an order."
    args_schema: type[BaseModel] = RefundArgs
    calls: list[dict[str, Any]] = []

    async def _arun(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return json.dumps({"refund_id": "R-0001", **kwargs, "status": "refunded"})

    def _run(self, **kwargs: Any) -> str:  # pragma: no cover
        raise NotImplementedError


class Recorder:
    """Approval handler that records requests and answers from a script."""

    def __init__(self, *decisions: ApprovalDecision | bool, delay: float = 0.0) -> None:
        self.requests: list[ApprovalRequest] = []
        self._decisions = list(decisions)
        self._delay = delay
        self.active = 0
        self.max_active = 0

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self._delay:
                await asyncio.sleep(self._delay)
            decision = self._decisions.pop(0) if self._decisions else True
        finally:
            self.active -= 1
        return decision if isinstance(decision, ApprovalDecision) else ApprovalDecision(decision)


def gate(tool_: BaseTool, handler: Any, **policy_kwargs: Any) -> BaseTool:
    policy_kwargs.setdefault("tools", ["*"])
    [wrapped] = wrap_tools_with_approval([tool_], ApprovalPolicy(handler=handler, **policy_kwargs))
    return wrapped


async def as_caller(caller: CallerContext | None, coro: Any, *, session: str | None = None) -> Any:
    """Run *coro* with a CallerContext (and chat session) set, as ainvoke/chat would."""
    caller_token = _caller_ctx_var.set(caller)
    session_token = _session_ctx_var.set(session)
    try:
        return await coro
    finally:
        _session_ctx_var.reset(session_token)
        _caller_ctx_var.reset(caller_token)


REFUND = {"order_id": "A-1001", "amount": 18.5, "reason": "Item arrived broken"}
EMAIL = {"to": "dana@example.com", "subject": "Hi", "body": "Hello"}


# ---------------------------------------------------------------------------
# 1. Plain LangChain tools behind approval
# ---------------------------------------------------------------------------


class TestLangChainTools:
    async def test_approved_structured_tool_runs(self):
        wrapped = gate(send_email, Recorder(True))
        assert await wrapped.ainvoke(EMAIL) == "sent to dana@example.com: Hi / Hello"

    async def test_approved_sync_tool_runs(self):
        wrapped = gate(sync_send_email, Recorder(True))
        result = await wrapped.ainvoke({"to": "dana@example.com", "subject": "Hi"})
        assert result == "sync sent to dana@example.com: Hi"

    async def test_config_reaches_the_inner_tool(self):
        seen: list[Any] = []

        @tool
        async def configured(to: str, config: RunnableConfig) -> str:
            """Reads the run config."""
            seen.append(config.get("configurable", {}).get("tenant"))
            return f"ok {to}"

        wrapped = gate(configured, Recorder(True))
        result = await wrapped.ainvoke({"to": "x"}, config={"configurable": {"tenant": "acme"}})
        assert result == "ok x"
        assert seen == ["acme"]

    async def test_one_tool_run_per_call(self):
        """The inner tool runs inside the gate's run — no second on_tool_start."""

        class Counter(AsyncCallbackHandler):
            def __init__(self) -> None:
                self.starts: list[str] = []

            async def on_tool_start(self, serialized, input_str, **kwargs):  # type: ignore[override]
                self.starts.append(serialized["name"])

        counter = Counter()
        wrapped = gate(send_email, Recorder(True))
        await wrapped.ainvoke(EMAIL, config={"callbacks": [counter]})
        assert counter.starts == ["send_email"]

    async def test_denied_structured_tool_does_not_run(self):
        wrapped = gate(send_email, Recorder(ApprovalDecision(False, reason="No.")))
        assert await wrapped.ainvoke(EMAIL) == "DENIED: No."


# ---------------------------------------------------------------------------
# 2. Redaction
# ---------------------------------------------------------------------------


class TestRedaction:
    async def test_order_ids_are_not_blood_types(self):
        policy = ApprovalPolicy(tools=["*"], handler=lambda r: True)
        assert await policy.redact_arguments(REFUND) == REFUND

    async def test_structure_survives_a_phone_number(self):
        policy = ApprovalPolicy(tools=["*"], handler=lambda r: True)
        args = {"to": "dana@example.com", "subject": "Your refund", "body": "Call +1 415 555 0100."}
        assert await policy.redact_arguments(args) == {
            "to": "[EMAIL]",
            "subject": "Your refund",
            "body": "Call [PHONE].",
        }

    async def test_nested_values_are_redacted_in_place(self):
        policy = ApprovalPolicy(tools=["*"], handler=lambda r: True)
        args = {
            "recipients": ["dana@example.com", "lee@example.com"],
            "meta": {"note": "ping dana@example.com", "count": 2, "urgent": True},
            "amount": 18.5,
        }
        assert await policy.redact_arguments(args) == {
            "recipients": ["[EMAIL]", "[EMAIL]"],
            "meta": {"note": "ping [EMAIL]", "count": 2, "urgent": True},
            "amount": 18.5,
        }

    async def test_reviewer_sees_the_order_id(self):
        recorder = Recorder(True)
        await gate(IssueRefund(), recorder).ainvoke(REFUND)
        assert recorder.requests[0].arguments == REFUND

    @pytest.mark.parametrize(
        "text",
        ["Blood type: O-", "blood group AB+", "Patient blood type is A positive", "bloodtype B-"],
    )
    async def test_blood_types_in_context_are_redacted(self, text):
        scanner = PromptiseSecurityScanner(detect_injection=False, detect_toxicity=False)
        assert "[MEDICAL]" in await scanner.check_output(text)

    @pytest.mark.parametrize("text", ["order A-1001", "grade B+ student", "A+ rating", "AB-12"])
    async def test_bare_letter_and_sign_is_not_a_blood_type(self, text):
        scanner = PromptiseSecurityScanner(detect_injection=False, detect_toxicity=False)
        assert await scanner.check_output(text) == text

    async def test_overlapping_findings_do_not_eat_the_following_text(self):
        """Two phone patterns match the same span; it is replaced once."""
        scanner = PromptiseSecurityScanner(detect_injection=False, detect_toxicity=False)
        text = '{"body": "Call us on +1 415 555 0100.", "x": 1}'
        assert await scanner.check_output(text) == '{"body": "Call us on [PHONE].", "x": 1}'


# ---------------------------------------------------------------------------
# 3. Repeated denials: per caller/session, within a window
# ---------------------------------------------------------------------------


class TestRepeatedDenials:
    async def test_limit_applies_within_one_caller(self):
        recorder = Recorder(False, False, False)
        wrapped = gate(IssueRefund(), recorder, redact_sensitive=False)
        alice = CallerContext(user_id="alice")
        for _ in range(3):
            assert "DENIED" in await as_caller(alice, wrapped.ainvoke(REFUND))
        fourth = await as_caller(alice, wrapped.ainvoke(REFUND))
        assert "already denied 3 times in the last 10 minutes" in fourth
        assert len(recorder.requests) == 3

    async def test_other_users_are_still_asked(self):
        recorder = Recorder(False, False, False, True)
        wrapped = gate(IssueRefund(), recorder, redact_sensitive=False)
        alice, bob = CallerContext(user_id="alice"), CallerContext(user_id="bob")
        for _ in range(3):
            await as_caller(alice, wrapped.ainvoke(REFUND))
        result = await as_caller(bob, wrapped.ainvoke(REFUND))
        assert "refunded" in result
        assert len(recorder.requests) == 4

    async def test_tenants_do_not_share_counts(self):
        recorder = Recorder(False, True)
        wrapped = gate(IssueRefund(), recorder, max_retries_after_deny=1)
        await as_caller(CallerContext(user_id="u1", tenant_id="t1"), wrapped.ainvoke(REFUND))
        result = await as_caller(
            CallerContext(user_id="u1", tenant_id="t2"), wrapped.ainvoke(REFUND)
        )
        assert "refunded" in result

    async def test_sessions_are_separate_by_default(self):
        recorder = Recorder(False, True)
        wrapped = gate(IssueRefund(), recorder, max_retries_after_deny=1)
        alice = CallerContext(user_id="alice")
        await as_caller(alice, wrapped.ainvoke(REFUND), session="s1")
        assert "already denied" in await as_caller(alice, wrapped.ainvoke(REFUND), session="s1")
        assert "refunded" in await as_caller(alice, wrapped.ainvoke(REFUND), session="s2")

    async def test_session_id_from_caller_metadata(self):
        recorder = Recorder(False, True)
        wrapped = gate(IssueRefund(), recorder, max_retries_after_deny=1)
        s1 = CallerContext(user_id="alice", metadata={"session_id": "s1"})
        s2 = CallerContext(user_id="alice", metadata={"session_id": "s2"})
        await as_caller(s1, wrapped.ainvoke(REFUND))
        assert "refunded" in await as_caller(s2, wrapped.ainvoke(REFUND))

    async def test_user_scope_spans_sessions(self):
        recorder = Recorder(False)
        wrapped = gate(IssueRefund(), recorder, max_retries_after_deny=1, deny_scope="user")
        alice = CallerContext(user_id="alice")
        await as_caller(alice, wrapped.ainvoke(REFUND), session="s1")
        assert "already denied" in await as_caller(alice, wrapped.ainvoke(REFUND), session="s2")

    async def test_agent_scope_spans_users(self):
        recorder = Recorder(False)
        wrapped = gate(IssueRefund(), recorder, max_retries_after_deny=1, deny_scope="agent")
        await as_caller(CallerContext(user_id="alice"), wrapped.ainvoke(REFUND))
        result = await as_caller(CallerContext(user_id="bob"), wrapped.ainvoke(REFUND))
        assert "already denied" in result

    async def test_denials_expire_after_the_window(self):
        recorder = Recorder(False, True)
        wrapped = gate(IssueRefund(), recorder, max_retries_after_deny=1, deny_window=60)
        await wrapped.ainvoke(REFUND)
        assert "already denied 1 times in the last 1 minute" in await wrapped.ainvoke(REFUND)
        later = time.monotonic() + 61
        with patch("promptise.approval.time.monotonic", return_value=later):
            assert "refunded" in await wrapped.ainvoke(REFUND)

    async def test_an_approval_resets_the_count(self):
        recorder = Recorder(False, True, False, True)
        wrapped = gate(IssueRefund(), recorder, max_retries_after_deny=2)
        for _ in range(4):
            await wrapped.ainvoke(REFUND)
        assert len(recorder.requests) == 4  # never hit the limit: 1 denial, reset, 1 denial

    async def test_limit_can_be_disabled(self):
        recorder = Recorder(*([False] * 5))
        wrapped = gate(IssueRefund(), recorder, max_retries_after_deny=None)
        for _ in range(5):
            await wrapped.ainvoke(REFUND)
        assert len(recorder.requests) == 5

    async def test_handler_errors_are_not_denials(self):
        calls = 0

        async def flaky(request):
            nonlocal calls
            calls += 1
            raise ConnectionError("down")

        wrapped = gate(IssueRefund(), flaky, max_retries_after_deny=1)
        for _ in range(3):
            assert "handler error" in await wrapped.ainvoke(REFUND)
        assert calls == 3

    def test_validation(self):
        handler = Recorder()
        with pytest.raises(ValueError, match="max_retries_after_deny"):
            ApprovalPolicy(tools=["*"], handler=handler, max_retries_after_deny=0)
        with pytest.raises(ValueError, match="deny_window"):
            ApprovalPolicy(tools=["*"], handler=handler, deny_window=0)
        with pytest.raises(ValueError, match="deny_scope"):
            ApprovalPolicy(tools=["*"], handler=handler, deny_scope="tenant")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="context_messages"):
            ApprovalPolicy(tools=["*"], handler=handler, context_messages=-1)


# ---------------------------------------------------------------------------
# 4. agent_id, context_summary, metadata
# ---------------------------------------------------------------------------


def _with_invocation(messages: list[Any], coro: Any) -> Any:
    async def run() -> Any:
        token = _invocation_ctx_var.set(_Invocation("inv-1", tuple(messages)))
        try:
            return await coro
        finally:
            _invocation_ctx_var.reset(token)

    return run()


class TestRequestFields:
    async def test_agent_id_and_metadata(self):
        recorder = Recorder(True)
        [wrapped] = wrap_tools_with_approval(
            [IssueRefund()],
            ApprovalPolicy(tools=["*"], handler=recorder, metadata={"env": "prod"}),
            agent_id="support-bot",
        )
        caller = CallerContext(user_id="alice", tenant_id="acme")
        await as_caller(caller, wrapped.ainvoke(REFUND), session="s-9")
        [request] = recorder.requests
        assert request.agent_id == "support-bot"
        assert request.caller_user_id == "alice"
        assert request.metadata == {
            "source": "agent",
            "session_id": "s-9",
            "tenant_id": "acme",
            "env": "prod",
        }

    async def test_metadata_callable_gets_raw_arguments(self):
        seen: list[tuple[str, dict[str, Any]]] = []

        async def tier(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            seen.append((tool_name, arguments))
            return {"tier": "high" if arguments["amount"] > 100 else "low"}

        recorder = Recorder(True)
        await gate(send_email, recorder, metadata=lambda name, args: {"to": args["to"]}).ainvoke(
            EMAIL
        )
        assert recorder.requests[0].metadata["to"] == "dana@example.com"
        await gate(IssueRefund(), recorder, metadata=tier).ainvoke(REFUND)
        assert recorder.requests[1].metadata["tier"] == "low"
        assert seen == [("issue_refund", REFUND)]

    async def test_context_summary_is_the_last_messages(self):
        recorder = Recorder(True)
        messages = [
            {"role": "system", "content": "You are a support bot."},
            HumanMessage(content="Hi"),
            AIMessage(content="Hello! How can I help?"),
            HumanMessage(
                content="Refund order A-1001, the mug arrived broken. Mail dana@example.com"
            ),
            AIMessage(content="", tool_calls=[{"name": "issue_refund", "args": {}, "id": "c1"}]),
            ToolMessage(content="{}", tool_call_id="c1"),
        ]
        await _with_invocation(messages, gate(IssueRefund(), recorder).ainvoke(REFUND))
        assert recorder.requests[0].context_summary == (
            "user: Hi\n"
            "assistant: Hello! How can I help?\n"
            "user: Refund order A-1001, the mug arrived broken. Mail [EMAIL]"
        )

    async def test_context_summary_can_be_disabled(self):
        recorder = Recorder(True)
        wrapped = gate(IssueRefund(), recorder, context_messages=0)
        await _with_invocation([HumanMessage(content="Refund it")], wrapped.ainvoke(REFUND))
        assert recorder.requests[0].context_summary == ""

    async def test_build_agent_fills_the_request(self):
        """End to end: build_agent + a scripted model + a plain @tool."""
        recorder = Recorder(True)
        model = _ScriptedModel([[{"name": "send_email", "args": EMAIL, "id": "c1"}]])
        agent = await build_agent(
            model=model,
            servers={},
            extra_tools=[send_email],
            approval=ApprovalPolicy(tools=["send_*"], handler=recorder),
            observer_agent_id="support-bot",
        )
        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content="Email Dana to say hi")]},
                caller=CallerContext(user_id="alice"),
            )
        finally:
            await agent.shutdown()
        [request] = recorder.requests
        assert request.agent_id == "support-bot"
        assert request.caller_user_id == "alice"
        assert request.context_summary == "user: Email Dana to say hi"
        assert request.metadata["source"] == "agent"
        [tool_message] = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert tool_message.content == "sent to dana@example.com: Hi / Hello"

    async def test_runtime_process_attributes_requests_to_the_process(self):
        from promptise.runtime import AgentProcess, ProcessConfig

        agent = AsyncMock()
        with patch("promptise.agent.build_agent", new=AsyncMock(return_value=agent)) as build:
            process = AgentProcess("refund-bot", ProcessConfig(model="openai:gpt-5-mini"))
            await process._build_agent()
        assert build.call_args.kwargs["observer_agent_id"] == "refund-bot"


class _ScriptedModel(BaseChatModel):
    """Emits the scripted tool calls, one turn each, then answers ``done``."""

    _script: list[list[dict[str, Any]]] = PrivateAttr()

    def __init__(self, script: list[list[dict[str, Any]]]) -> None:
        super().__init__()
        self._script = list(script)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> _ScriptedModel:
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self._script:
            message = AIMessage(content="", tool_calls=self._script.pop(0))
        else:
            message = AIMessage(content="done")
        return ChatResult(generations=[ChatGeneration(message=message)])


# ---------------------------------------------------------------------------
# 5. Several approvals in one turn
# ---------------------------------------------------------------------------


class TestConcurrentApprovals:
    async def _two_refunds(self, sequential: bool) -> Recorder:
        recorder = Recorder(True, True, delay=0.05)
        model = _ScriptedModel(
            [
                [
                    {"name": "issue_refund", "args": REFUND, "id": "c1"},
                    {"name": "issue_refund", "args": {**REFUND, "order_id": "A-1002"}, "id": "c2"},
                ]
            ]
        )
        agent = await build_agent(
            model=model,
            servers={},
            extra_tools=[IssueRefund()],
            approval=ApprovalPolicy(tools=["issue_*"], handler=recorder, sequential=sequential),
        )
        try:
            await agent.ainvoke({"messages": [HumanMessage(content="Refund both")]})
        finally:
            await agent.shutdown()
        assert len(recorder.requests) == 2
        return recorder

    async def test_concurrent_by_default(self):
        assert (await self._two_refunds(sequential=False)).max_active == 2

    async def test_sequential_asks_one_at_a_time(self):
        assert (await self._two_refunds(sequential=True)).max_active == 1

    async def test_sequential_does_not_serialize_other_invocations(self):
        recorder = Recorder(True, True, delay=0.05)
        wrapped = gate(IssueRefund(), recorder, sequential=True)
        await asyncio.gather(
            _with_invocation([], wrapped.ainvoke(REFUND)),
            _invocation_scope("inv-2", wrapped.ainvoke(REFUND)),
        )
        assert recorder.max_active == 2


async def _invocation_scope(invocation_id: str, coro: Any) -> Any:
    token = _invocation_ctx_var.set(_Invocation(invocation_id, ()))
    try:
        return await coro
    finally:
        _invocation_ctx_var.reset(token)


# ---------------------------------------------------------------------------
# 6. Webhook signature verification
# ---------------------------------------------------------------------------


class TestVerifyWebhookSignature:
    def _signed(self) -> tuple[bytes, str]:
        request = ApprovalRequest(
            request_id="r1",
            tool_name="send_email",
            arguments={"to": "[EMAIL]", "amount": 18.5},
            agent_id="bot",
            caller_user_id="alice",
            context_summary="user: hi",
            metadata={"source": "agent"},
        )
        return json.dumps(request.to_dict()).encode(), request.compute_hmac("s3cret")

    def test_accepts_the_handlers_signature(self):
        body, signature = self._signed()
        assert verify_webhook_signature(body, signature, "s3cret")
        assert verify_webhook_signature(body.decode(), signature, "s3cret")
        assert verify_webhook_signature(json.loads(body), signature, "s3cret", max_age=60)

    def test_signed_fields(self):
        assert SIGNED_FIELDS == (
            "request_id",
            "tool_name",
            "arguments",
            "agent_id",
            "caller_user_id",
            "timestamp",
        )

    def test_rejects_tampering_and_garbage(self):
        body, signature = self._signed()
        data = json.loads(body)
        assert not verify_webhook_signature(body, signature, "wrong-secret")
        assert not verify_webhook_signature({**data, "arguments": {"to": "x"}}, signature, "s3cret")
        assert not verify_webhook_signature({**data, "agent_id": None}, signature, "s3cret")
        missing = {k: v for k, v in data.items() if k != "timestamp"}
        assert not verify_webhook_signature(missing, signature, "s3cret")
        assert not verify_webhook_signature(b"not json", signature, "s3cret")
        assert not verify_webhook_signature(b"[1, 2]", signature, "s3cret")
        assert not verify_webhook_signature(body, None, "s3cret")
        assert not verify_webhook_signature(body, "", "s3cret")

    def test_max_age_rejects_stale_requests(self):
        body, signature = self._signed()
        with patch("promptise.approval.time.time", return_value=time.time() + 3600):
            assert verify_webhook_signature(body, signature, "s3cret")
            assert not verify_webhook_signature(body, signature, "s3cret", max_age=300)

    async def test_round_trip_through_the_handler(self):
        import httpx

        received: list[bool] = []

        def service(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                received.append(
                    verify_webhook_signature(
                        request.content, request.headers["X-Promptise-Signature"], "s3cret"
                    )
                )
                return httpx.Response(202)
            return httpx.Response(200, json={"approved": True})

        handler = WebhookApprovalHandler(
            url="https://approvals.example.com/requests",
            secret="s3cret",
            poll_interval=0.5,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(service)),
        )
        assert "refunded" in await gate(IssueRefund(), handler).ainvoke(REFUND)
        assert received == [True]


# ---------------------------------------------------------------------------
# 7. Private networks
# ---------------------------------------------------------------------------


class TestWebhookPrivateNetworks:
    @pytest.mark.parametrize(
        "url",
        ["http://localhost:8140/approvals", "http://127.0.0.1/approvals", "http://10.0.0.5/a"],
    )
    def test_refused_by_default_with_a_useful_message(self, url):
        with pytest.raises(ValueError, match="allow_private_networks=True") as excinfo:
            WebhookApprovalHandler(url=url)
        assert "base_url" not in str(excinfo.value)

    def test_poll_url_is_checked_too(self):
        with pytest.raises(ValueError, match="allow_private_networks"):
            WebhookApprovalHandler(
                url="https://approvals.example.com/requests",
                poll_url="http://169.254.169.254/latest",
            )

    @pytest.mark.parametrize("url", ["http://localhost:8140/approvals", "http://10.0.0.5/a"])
    def test_opt_in(self, url):
        handler = WebhookApprovalHandler(
            url=url, poll_url="http://10.0.0.5/decisions", allow_private_networks=True
        )
        assert isinstance(handler, WebhookApprovalHandler)


# ---------------------------------------------------------------------------
# 8. modified_arguments merges onto the original call
# ---------------------------------------------------------------------------


class TestModifiedArguments:
    async def test_partial_edit_keeps_the_other_arguments(self):
        refund = IssueRefund()
        refund.calls = []
        wrapped = gate(
            refund, Recorder(ApprovalDecision(True, modified_arguments={"amount": 9.25}))
        )
        await wrapped.ainvoke(REFUND)
        assert refund.calls == [{**REFUND, "amount": 9.25}]

    async def test_echoed_redacted_values_keep_the_originals(self):
        """``{**request.arguments, ...}`` must not send "[EMAIL]" to the tool."""

        async def edit(request: ApprovalRequest) -> ApprovalDecision:
            assert request.arguments["to"] == "[EMAIL]"
            return ApprovalDecision(True, modified_arguments={**request.arguments, "subject": "Re"})

        result = await gate(send_email, edit).ainvoke(EMAIL)
        assert result.endswith("sent to dana@example.com: Re / Hello")

    async def test_invalid_edit_fails_without_running_the_tool(self):
        refund = IssueRefund()
        refund.calls = []
        decision = ApprovalDecision(True, modified_arguments={"amount": "half"})
        with pytest.raises(Exception, match="amount"):
            await gate(refund, Recorder(decision)).ainvoke(REFUND)
        assert refund.calls == []


# ---------------------------------------------------------------------------
# 9. The model is told about the reviewer's changes
# ---------------------------------------------------------------------------


class TestModificationNote:
    async def test_note_names_the_change(self):
        wrapped = gate(
            IssueRefund(), Recorder(ApprovalDecision(True, modified_arguments={"amount": 9.25}))
        )
        result = await wrapped.ainvoke(REFUND)
        note, _, payload = result.partition("\n\n")
        assert note.startswith(
            "[Approved with modified arguments] A reviewer changed amount: 18.5 -> 9.25."
        )
        assert json.loads(payload)["amount"] == 9.25

    async def test_no_note_when_nothing_changed(self):
        decision = ApprovalDecision(True, modified_arguments=dict(REFUND))
        result = await gate(IssueRefund(), Recorder(decision)).ainvoke(REFUND)
        assert not result.startswith("[Approved")

    async def test_note_in_front_of_content_blocks(self):
        @tool
        async def blocks(to: str) -> list[dict[str, Any]]:
            """Returns content blocks."""
            return [{"type": "text", "text": f"sent to {to}"}]

        decision = ApprovalDecision(True, modified_arguments={"to": "team@x.com"})
        result = await gate(blocks, Recorder(decision)).ainvoke({"to": "all@x.com"})
        assert result[0]["type"] == "text" and result[0]["text"].startswith("[Approved")
        assert result[1] == {"type": "text", "text": "sent to team@x.com"}

    async def test_the_model_sees_the_note(self):
        decision = ApprovalDecision(True, modified_arguments={"amount": 9.25})
        model = _ScriptedModel([[{"name": "issue_refund", "args": REFUND, "id": "c1"}]])
        agent = await build_agent(
            model=model,
            servers={},
            extra_tools=[IssueRefund()],
            approval=ApprovalPolicy(tools=["issue_*"], handler=Recorder(decision)),
        )
        try:
            result = await agent.ainvoke({"messages": [HumanMessage(content="Refund A-1001")]})
        finally:
            await agent.shutdown()
        [tool_message] = [m for m in result["messages"] if isinstance(m, ToolMessage)]
        assert "A reviewer changed amount: 18.5 -> 9.25" in str(tool_message.content)


class TestSuperAgentYaml:
    def test_private_webhook_and_denial_options(self, tmp_path):
        from promptise.superagent import SuperAgentLoader

        path = tmp_path / "ops.superagent"
        path.write_text(
            """version: "1.0"
agent:
  model: "openai:gpt-5-mini"
servers:
  support:
    type: http
    url: "https://support.example.com/mcp"
approval:
  tools: ["issue_refund"]
  handler: webhook
  webhook_url: "http://10.0.0.5/approvals"
  webhook_allow_private_networks: true
  deny_window: 120
  deny_scope: user
  sequential: true
  context_messages: 5
"""
        )
        kwargs = SuperAgentLoader.from_file(path).to_agent_config().to_build_kwargs()
        policy = kwargs["approval"]
        assert isinstance(policy.handler, WebhookApprovalHandler)
        assert (policy.deny_window, policy.deny_scope, policy.sequential) == (120, "user", True)
        assert policy.context_messages == 5

    def test_private_webhook_refused_without_opt_in(self, tmp_path):
        from promptise.superagent import SuperAgentLoader

        path = tmp_path / "ops.superagent"
        path.write_text(
            """version: "1.0"
agent:
  model: "openai:gpt-5-mini"
servers:
  support:
    type: http
    url: "https://support.example.com/mcp"
approval:
  tools: ["issue_refund"]
  webhook_url: "http://10.0.0.5/approvals"
"""
        )
        config = SuperAgentLoader.from_file(path).to_agent_config()
        with pytest.raises(ValueError, match="allow_private_networks"):
            config.to_build_kwargs()
