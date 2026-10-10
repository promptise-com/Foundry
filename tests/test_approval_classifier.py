"""Tests for AutoApprovalClassifier — explicit decision hierarchy."""

from __future__ import annotations

import asyncio
import json

import pytest

from promptise.approval import ApprovalDecision, ApprovalRequest
from promptise.approval_classifier import (
    DEFAULT_READ_ONLY_PREFIXES,
    ApprovalRule,
    AutoApprovalClassifier,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _RecordingFallback:
    """Fake handler — records every request and returns a configured outcome."""

    def __init__(self, *, approved: bool = True, reason: str = "fallback") -> None:
        self.requests: list[ApprovalRequest] = []
        self._approved = approved
        self._reason = reason

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        return ApprovalDecision(
            approved=self._approved,
            reviewer_id="fallback",
            reason=self._reason,
        )


def _make_request(tool: str, **kwargs) -> ApprovalRequest:
    return ApprovalRequest(
        request_id=f"req-{tool}",
        tool_name=tool,
        arguments=kwargs.get("arguments", {}),
        agent_id=kwargs.get("agent_id"),
        caller_user_id=kwargs.get("user"),
    )


# ---------------------------------------------------------------------------
# ApprovalRule
# ---------------------------------------------------------------------------


class TestApprovalRule:
    @pytest.mark.asyncio
    async def test_glob_tool_match(self):
        rule = ApprovalRule(tool="get_*")
        assert await rule.matches(_make_request("get_users"))
        assert not await rule.matches(_make_request("delete_users"))

    @pytest.mark.asyncio
    async def test_user_filter(self):
        rule = ApprovalRule(tool="*", user="alice")
        assert await rule.matches(_make_request("anything", user="alice"))
        assert not await rule.matches(_make_request("anything", user="bob"))

    @pytest.mark.asyncio
    async def test_argument_substring(self):
        rule = ApprovalRule(tool="*", argument_contains="rm -rf")
        bad = _make_request("shell", arguments={"cmd": "sudo rm -rf /"})
        good = _make_request("shell", arguments={"cmd": "ls"})
        assert await rule.matches(bad)
        assert not await rule.matches(good)

    @pytest.mark.asyncio
    async def test_predicate(self):
        async def is_alice(req: ApprovalRequest) -> bool:
            return req.caller_user_id == "alice"

        rule = ApprovalRule(predicate=is_alice)
        assert await rule.matches(_make_request("x", user="alice"))
        assert not await rule.matches(_make_request("x", user="bob"))

    @pytest.mark.asyncio
    async def test_predicate_exception_treats_as_no_match(self):
        async def boom(req: ApprovalRequest) -> bool:
            raise RuntimeError("kaboom")

        rule = ApprovalRule(predicate=boom)
        assert await rule.matches(_make_request("x")) is False


# ---------------------------------------------------------------------------
# Decision hierarchy
# ---------------------------------------------------------------------------


class TestDecisionHierarchy:
    @pytest.mark.asyncio
    async def test_layer1_allow_rule_short_circuits(self):
        fb = _RecordingFallback(approved=False)
        clf = AutoApprovalClassifier(
            allow_rules=[ApprovalRule(tool="send_email", reason="trusted")],
            fallback=fb,
        )
        decision = await clf.request_approval(_make_request("send_email"))
        assert decision.approved is True
        assert decision.reason == "trusted"
        assert fb.requests == []
        assert clf.stats.allow_rule_hits == 1
        assert clf.last_trace.layer == "allow_rule"

    @pytest.mark.asyncio
    async def test_layer2_deny_rule_short_circuits(self):
        fb = _RecordingFallback(approved=True)
        clf = AutoApprovalClassifier(
            deny_rules=[ApprovalRule(tool="exec_*", reason="too risky")],
            fallback=fb,
        )
        decision = await clf.request_approval(_make_request("exec_shell"))
        assert decision.approved is False
        assert decision.reason == "too risky"
        assert fb.requests == []
        assert clf.stats.deny_rule_hits == 1

    @pytest.mark.asyncio
    async def test_deny_rules_take_precedence_over_allow(self):
        fb = _RecordingFallback()
        clf = AutoApprovalClassifier(
            allow_rules=[ApprovalRule(tool="exec_safe", reason="vetted")],
            deny_rules=[ApprovalRule(tool="exec_*", reason="risky")],
            fallback=fb,
        )
        decision = await clf.request_approval(_make_request("exec_safe"))
        assert decision.approved is False
        assert decision.reason == "risky"
        assert decision.trace is not None and decision.trace.layer == "deny_rule"

    @pytest.mark.asyncio
    async def test_layer3_read_only_auto_allow(self):
        fb = _RecordingFallback(approved=False)
        clf = AutoApprovalClassifier(fallback=fb)

        for tool in ["get_users", "list_files", "read_config", "search_docs"]:
            decision = await clf.request_approval(_make_request(tool))
            assert decision.approved is True
            assert "read-only" in decision.reason

        assert clf.stats.read_only_allows == 4
        assert fb.requests == []

    @pytest.mark.asyncio
    async def test_layer3_disabled_falls_through(self):
        fb = _RecordingFallback(approved=True)
        clf = AutoApprovalClassifier(read_only_auto_allow=False, fallback=fb)
        await clf.request_approval(_make_request("get_users"))
        # Read-only disabled → goes to fallback
        assert len(fb.requests) == 1
        assert clf.stats.read_only_allows == 0

    @pytest.mark.asyncio
    async def test_layer4_llm_classifier_allow(self):
        fb = _RecordingFallback(approved=False)

        async def llm(req):
            return "allow", "looks safe"

        clf = AutoApprovalClassifier(
            llm_classifier=llm,
            read_only_auto_allow=False,
            fallback=fb,
        )
        decision = await clf.request_approval(_make_request("custom_tool"))
        assert decision.approved is True
        assert decision.reason == "looks safe"
        assert clf.stats.llm_allows == 1
        assert fb.requests == []

    @pytest.mark.asyncio
    async def test_layer4_llm_classifier_deny(self):
        fb = _RecordingFallback(approved=True)

        async def llm(req):
            return "deny", "looks dangerous"

        clf = AutoApprovalClassifier(
            llm_classifier=llm,
            read_only_auto_allow=False,
            fallback=fb,
        )
        decision = await clf.request_approval(_make_request("custom_tool"))
        assert decision.approved is False
        assert decision.reason == "looks dangerous"
        assert clf.stats.llm_denies == 1

    @pytest.mark.asyncio
    async def test_layer4_llm_escalate_falls_to_fallback(self):
        fb = _RecordingFallback(approved=True)

        async def llm(req):
            return "escalate", "not sure"

        clf = AutoApprovalClassifier(
            llm_classifier=llm,
            read_only_auto_allow=False,
            fallback=fb,
        )
        decision = await clf.request_approval(_make_request("custom_tool"))
        assert decision.approved is True  # comes from fallback
        assert len(fb.requests) == 1
        assert clf.stats.llm_escalations == 1
        assert clf.stats.fallback_allows == 1

    @pytest.mark.asyncio
    async def test_layer5_fallback_when_no_rules_match(self):
        fb = _RecordingFallback(approved=False, reason="human said no")
        clf = AutoApprovalClassifier(read_only_auto_allow=False, fallback=fb)
        decision = await clf.request_approval(_make_request("modify_data"))
        assert decision.approved is False
        assert decision.reason == "human said no"
        assert clf.stats.fallback_denies == 1

    @pytest.mark.asyncio
    async def test_classifier_exception_falls_back(self):
        fb = _RecordingFallback(approved=True)

        async def boom(req):
            raise RuntimeError("classifier broken")

        clf = AutoApprovalClassifier(
            llm_classifier=boom,
            read_only_auto_allow=False,
            fallback=fb,
        )
        decision = await clf.request_approval(_make_request("custom_tool"))
        # Falls back; doesn't raise
        assert decision.approved is True
        assert clf.stats.errors == 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestClassifierStats:
    @pytest.mark.asyncio
    async def test_reset_stats(self):
        fb = _RecordingFallback()
        clf = AutoApprovalClassifier(fallback=fb)
        await clf.request_approval(_make_request("get_x"))
        assert clf.stats.read_only_allows == 1
        clf.reset_stats()
        assert clf.stats.read_only_allows == 0


class TestDefaultReadOnlyPrefixes:
    def test_includes_common_prefixes(self):
        for p in ("get_", "list_", "read_", "search_"):
            assert p in DEFAULT_READ_ONLY_PREFIXES


# ---------------------------------------------------------------------------
# Guide 25 regressions
# ---------------------------------------------------------------------------


class TestDenyBeforeAllow:
    """A broad allow rule auto-approved ``delete_customer`` despite a deny rule."""

    @pytest.mark.asyncio
    async def test_admin_bypass_does_not_beat_a_deny_rule(self):
        fb = _RecordingFallback()
        clf = AutoApprovalClassifier(
            allow_rules=[ApprovalRule(user="admin@shop.example", reason="admin bypass")],
            deny_rules=[ApprovalRule(tool="delete_*", reason="never auto-delete")],
            fallback=fb,
        )
        decision = await clf.request_approval(
            _make_request("delete_customer", user="admin@shop.example")
        )
        assert decision.approved is False
        assert decision.reason == "never auto-delete"
        assert decision.decided_by == "classifier"
        # The allow rule still applies to everything the deny rule doesn't cover.
        allowed = await clf.request_approval(
            _make_request("refund_order", user="admin@shop.example")
        )
        assert allowed.approved is True
        assert allowed.trace is not None and allowed.trace.layer == "allow_rule"


class TestAskRules:
    @pytest.mark.asyncio
    async def test_ask_rule_beats_allow_rule_read_only_and_llm(self):
        llm_calls: list[str] = []

        async def llm(req):
            llm_calls.append(req.tool_name)
            return "allow", "looks fine"

        fb = _RecordingFallback(approved=False, reason="lead said no")
        clf = AutoApprovalClassifier(
            ask_rules=[ApprovalRule(tool="get_payroll", reason="payroll needs a person")],
            allow_rules=[ApprovalRule(tool="get_*", reason="reads are fine")],
            llm_classifier=llm,
            fallback=fb,
        )
        decision = await clf.request_approval(_make_request("get_payroll"))
        assert decision.approved is False
        assert decision.reason == "lead said no"
        assert decision.decided_by == "reviewer"
        assert decision.trace is not None
        assert decision.trace.layer == "ask_rule"
        assert decision.trace.rule_reason == "payroll needs a person"
        assert [r.tool_name for r in fb.requests] == ["get_payroll"]
        assert llm_calls == []
        assert clf.stats.ask_rule_hits == 1
        assert clf.stats.fallback_denies == 1
        assert clf.stats.allow_rule_hits == 0

    @pytest.mark.asyncio
    async def test_deny_rule_beats_ask_rule(self):
        fb = _RecordingFallback()
        clf = AutoApprovalClassifier(
            deny_rules=[ApprovalRule(tool="wire_*", argument_contains="offshore", reason="no")],
            ask_rules=[ApprovalRule(tool="wire_*", reason="a person decides")],
            fallback=fb,
        )
        denied = await clf.request_approval(
            _make_request("wire_funds", arguments={"to": "offshore-1"})
        )
        assert denied.approved is False and denied.trace.layer == "deny_rule"
        asked = await clf.request_approval(_make_request("wire_funds", arguments={"to": "acme"}))
        assert asked.approved is True and asked.trace.layer == "ask_rule"
        assert len(fb.requests) == 1


class TestArgumentContainsIsJson:
    @pytest.mark.asyncio
    async def test_json_style_matches_and_python_repr_does_not(self):
        req = _make_request("delete_customer", arguments={"customer_id": "C-7", "force": True})
        assert await ApprovalRule(argument_contains='"force": true').matches(req)
        assert not await ApprovalRule(argument_contains="'force': True").matches(req)

    @pytest.mark.asyncio
    async def test_keys_are_sorted_and_non_ascii_kept(self):
        req = _make_request("note", arguments={"z": 1, "a": "Zürich"})
        assert await ApprovalRule(argument_contains='{"a": "Zürich", "z": 1}').matches(req)


class TestRawArguments:
    """Redaction for the reviewer must not hide arguments from the rules."""

    @pytest.mark.asyncio
    async def test_rules_match_raw_arguments_and_fallback_gets_the_redacted_copy(self):
        fb = _RecordingFallback()
        seen: list[dict] = []

        async def predicate(req):
            seen.append(dict(req.arguments))
            return False

        clf = AutoApprovalClassifier(
            deny_rules=[
                ApprovalRule(tool="send_*", predicate=predicate),
                ApprovalRule(tool="send_*", argument_contains="@competitor.example"),
            ],
            fallback=fb,
        )
        raw = {"to": "sam@competitor.example"}
        request = ApprovalRequest(
            request_id="r1",
            tool_name="send_email",
            arguments={"to": "[EMAIL]"},
            raw_arguments=raw,
        )
        decision = await clf.request_approval(request)
        assert decision.approved is False
        assert seen == [raw]

        other = ApprovalRequest(
            request_id="r2",
            tool_name="send_email",
            arguments={"to": "[EMAIL]"},
            raw_arguments={"to": "dana@example.com"},
        )
        await clf.request_approval(other)
        [shown] = fb.requests
        assert shown.arguments == {"to": "[EMAIL]"}
        assert shown.raw_arguments is None

    def test_raw_arguments_never_serialized_or_shown(self):
        request = ApprovalRequest(
            request_id="r1",
            tool_name="send_email",
            arguments={"to": "[EMAIL]"},
            raw_arguments={"to": "sam@competitor.example"},
        )
        assert "competitor" not in repr(request)
        assert "raw_arguments" not in request.to_dict()
        assert "competitor" not in json.dumps(request.to_dict())


class TestReadOnlyLayer:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "name",
        ["fetch_and_purge_cache", "show_and_delete", "getAndDeleteUser", "list_then_drop_tables"],
    )
    async def test_destructive_verbs_in_the_name_are_not_read_only(self, name):
        fb = _RecordingFallback(approved=False)
        clf = AutoApprovalClassifier(fallback=fb)
        decision = await clf.request_approval(_make_request(name))
        assert decision.approved is False
        assert decision.trace.layer == "fallback"
        assert len(fb.requests) == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", ["get_order", "get_settings", "list_addresses", "show_cart"])
    async def test_plain_read_names_are_still_read_only(self, name):
        clf = AutoApprovalClassifier(fallback=_RecordingFallback(approved=False))
        decision = await clf.request_approval(_make_request(name))
        assert decision.approved is True
        assert decision.trace.layer == "read_only"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "annotations",
        [
            {"readOnlyHint": False},
            {"destructiveHint": True},
            {"readOnlyHint": True, "destructiveHint": True},
        ],
    )
    async def test_annotations_veto_a_read_only_name(self, annotations):
        clf = AutoApprovalClassifier(fallback=_RecordingFallback(approved=False))
        request = ApprovalRequest(
            request_id="r", tool_name="get_order", arguments={}, tool_annotations=annotations
        )
        decision = await clf.request_approval(request)
        assert decision.approved is False
        assert decision.trace.layer == "fallback"

    @pytest.mark.asyncio
    async def test_read_only_hint_allows_a_name_without_a_prefix(self):
        clf = AutoApprovalClassifier(fallback=_RecordingFallback(approved=False))
        request = ApprovalRequest(
            request_id="r",
            tool_name="order_status",
            arguments={},
            tool_annotations={"readOnlyHint": True},
        )
        assert (await clf.request_approval(request)).trace.layer == "read_only"

    @pytest.mark.asyncio
    async def test_read_only_hint_does_not_override_a_destructive_verb(self):
        clf = AutoApprovalClassifier(fallback=_RecordingFallback(approved=False))
        request = ApprovalRequest(
            request_id="r",
            tool_name="purge_cache",
            arguments={},
            tool_annotations={"readOnlyHint": True},
        )
        assert (await clf.request_approval(request)).approved is False

    @pytest.mark.asyncio
    async def test_annotations_can_be_ignored(self):
        clf = AutoApprovalClassifier(
            fallback=_RecordingFallback(approved=False), use_tool_annotations=False
        )
        request = ApprovalRequest(
            request_id="r",
            tool_name="order_status",
            arguments={},
            tool_annotations={"readOnlyHint": True},
        )
        assert (await clf.request_approval(request)).approved is False

    @pytest.mark.asyncio
    async def test_custom_destructive_verbs(self):
        clf = AutoApprovalClassifier(
            fallback=_RecordingFallback(approved=False), destructive_verbs=("archive",)
        )
        assert (await clf.request_approval(_make_request("get_and_archive"))).approved is False
        # The default list is replaced, not extended.
        assert (await clf.request_approval(_make_request("show_and_delete"))).approved is True

    def test_default_destructive_verbs_exported(self):
        from promptise import DEFAULT_DESTRUCTIVE_VERBS

        assert {"delete", "purge", "drop", "update", "send"} <= set(DEFAULT_DESTRUCTIVE_VERBS)


class TestTracePerDecision:
    """``last_trace`` was shared, so concurrent decisions overwrote each other's layer."""

    @pytest.mark.asyncio
    async def test_concurrent_escalation_keeps_its_own_trace(self):
        gate = asyncio.Event()

        class SlowHuman:
            async def request_approval(self, request):
                await gate.wait()
                return ApprovalDecision(approved=True, reviewer_id="lead")

        async def llm(req):
            return "escalate", "unsure about this one"

        clf = AutoApprovalClassifier(llm_classifier=llm, fallback=SlowHuman())
        escalated = asyncio.create_task(
            clf.request_approval(_make_request("issue_refund", arguments={"amount": 50}))
        )
        await asyncio.sleep(0)
        quick = await clf.request_approval(_make_request("get_order"))
        gate.set()
        decision = await escalated

        assert quick.trace.layer == "read_only"
        assert decision.trace.layer == "llm_escalate_then_fallback"
        assert decision.trace.rule_reason == "unsure about this one"
        assert decision.reviewer_id == "lead"
        assert decision.decided_by == "reviewer"
        # last_trace is the most recently *finished* decision.
        assert clf.last_trace is decision.trace

    @pytest.mark.asyncio
    async def test_fallback_decision_object_is_not_mutated(self):
        shared = ApprovalDecision(approved=True, reviewer_id="lead")

        class Constant:
            async def request_approval(self, request):
                return shared

        clf = AutoApprovalClassifier(fallback=Constant(), read_only_auto_allow=False)
        decision = await clf.request_approval(_make_request("refund"))
        assert decision.trace.layer == "fallback"
        assert shared.trace is None

    @pytest.mark.asyncio
    async def test_error_layer_trace(self):
        async def broken_llm(req):
            raise RuntimeError("model down")

        clf = AutoApprovalClassifier(
            llm_classifier=broken_llm,
            fallback=_RecordingFallback(approved=False),
            read_only_auto_allow=False,
        )
        decision = await clf.request_approval(_make_request("refund"))
        assert decision.trace.layer == "error"
        assert clf.stats.errors == 1
