"""Tests for LLM curation (``promptise.mcpcast.curate``).

The model is replaced by scripted completions (tests may mock; production
never does).  Every post-condition is asserted both as a rejection and as
the retry-with-feedback loop that follows it.
"""

from __future__ import annotations

import json

import pytest

from promptise.mcpcast._llm import extract_json_object, final_text
from promptise.mcpcast.classify import classify
from promptise.mcpcast.curate import (
    CURATION_SYSTEM,
    CurationResult,
    CurationViolation,
    apply_curation,
    check_postconditions,
    curate,
    render_curation_prompt,
)
from promptise.mcpcast.parse import Operation, ParamSpec
from promptise.mcpcast.schema import MCPcastError, RiskClass, SafetyProfile

BASE = "https://api.example.com"


def op(op_id, method, path, params=(), **kw):
    return Operation(
        operation_id=op_id, method=method, path=path, params=list(params), base_url=BASE, **kw
    )


OPS = [
    op(
        "getCustomerById",
        "GET",
        "/customers/{id}",
        [ParamSpec(name="id", location="path", required=True)],
        summary="Get customer",
    ),
    op(
        "searchCustomers",
        "GET",
        "/customers",
        [
            ParamSpec(name="email", location="query"),
            ParamSpec(name="limit", location="query", json_schema={"type": "integer"}),
        ],
        summary="Search customers",
    ),
    op(
        "cancelSubscription",
        "POST",
        "/customers/{id}/subscriptions:cancel",
        [
            ParamSpec(name="id", location="path", required=True),
            ParamSpec(name="reason", location="body"),
        ],
    ),
    op("healthCheck", "GET", "/health"),
    op("legacyExport", "GET", "/export", deprecated=True),
]
CLASSES = {o.operation_id: classify(o) for o in OPS}

GOOD = {
    "tools": [
        {
            "name": "find_customer",
            "description": "Look up a customer by id or email. Use before any operation that needs a customer id.",
            "risk": "read",
            "operations": ["getCustomerById", "searchCustomers"],
            "params": {
                "email": {"description": "Customer email address"},
                "limit": {"hidden": True, "default": 5},
            },
            "example": {"email": "ada@example.com"},
            "tags": ["customers"],
        },
        {
            "name": "cancel_subscription",
            "description": "Cancel a customer's subscription. Irreversible; confirm with find_customer first.",
            "risk": "destructive",
            "operations": ["cancelSubscription"],
            "example": {"id": "cus_1"},
        },
    ],
    "dropped": [
        {"operation_id": "healthCheck", "reason": "not useful to an agent"},
        {"operation_id": "legacyExport", "reason": "deprecated"},
    ],
}


def _script(*responses: object):
    """A Completer that replays responses and records prompts."""
    queue = [r if isinstance(r, str) else json.dumps(r) for r in responses]
    seen: list[tuple[str, str]] = []

    async def complete(system: str, user: str) -> str:
        seen.append((system, user))
        if not queue:
            raise AssertionError("model called more times than scripted")
        return queue.pop(0)

    complete.seen = seen  # type: ignore[attr-defined]
    return complete


def _violations(proposal: dict, max_tools: int = 25) -> list[str]:
    return check_postconditions(
        CurationResult.model_validate(proposal), OPS, CLASSES, max_tools=max_tools
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestApplyCuration:
    def test_good_proposal_becomes_plan(self):
        plan = apply_curation(
            CurationResult.model_validate(GOOD),
            OPS,
            CLASSES,
            profile=SafetyProfile.FULL,
            max_tools=25,
            name="crm",
        )
        assert plan.tool_names == ["find_customer", "cancel_subscription"]
        find = plan.tool("find_customer")
        assert find.operations == ["getCustomerById", "searchCustomers"]
        assert find.risk is RiskClass.READ and not find.requires_approval
        # union of params; id is optional on the tool because one route does not need it
        assert set(find.params) == {"id", "email", "limit"}
        assert find.params["id"].required is False
        assert find.params["email"].description == "Customer email address"
        assert find.params["limit"].hidden is True and find.params["limit"].default == 5
        assert find.example == {"email": "ada@example.com"}
        assert find.tags == ["customers"]
        cancel = plan.tool("cancel_subscription")
        assert cancel.requires_approval and cancel.risk is RiskClass.DESTRUCTIVE
        assert cancel.example == {"id": "cus_1"}
        assert {d.operation_id: d.reason for d in plan.dropped} == {
            "healthCheck": "not useful to an agent",
            "legacyExport": "deprecated",
        }

    def test_profile_applied_after_curation(self):
        plan = apply_curation(
            CurationResult.model_validate(GOOD),
            OPS,
            CLASSES,
            profile=SafetyProfile.READ_ONLY,
            max_tools=25,
        )
        assert plan.tool_names == ["find_customer"]
        reasons = {d.operation_id: d.reason for d in plan.dropped}
        assert (
            reasons["cancelSubscription"] == "destructive operation excluded by profile 'read-only'"
        )

    def test_unmentioned_operations_are_dropped_transparently(self):
        proposal = {"tools": [GOOD["tools"][0]], "dropped": []}
        plan = apply_curation(
            CurationResult.model_validate(proposal),
            OPS,
            CLASSES,
            profile=SafetyProfile.FULL,
            max_tools=25,
        )
        reasons = {d.operation_id: d.reason for d in plan.dropped}
        assert reasons == {
            "cancelSubscription": "not selected by curation",
            "healthCheck": "not selected by curation",
            "legacyExport": "not selected by curation",
        }

    def test_example_generated_when_missing(self):
        tool = {**GOOD["tools"][1]}
        tool.pop("example")
        plan = apply_curation(
            CurationResult.model_validate({"tools": [tool], "dropped": []}),
            OPS,
            CLASSES,
            profile=SafetyProfile.FULL,
            max_tools=25,
        )
        assert plan.tool("cancel_subscription").example == {"id": "123"}

    def test_escalation_allowed(self):
        tool = {**GOOD["tools"][0], "risk": "write"}
        plan = apply_curation(
            CurationResult.model_validate({"tools": [tool], "dropped": []}),
            OPS,
            CLASSES,
            profile=SafetyProfile.FULL,
            max_tools=25,
        )
        assert plan.tool("find_customer").risk is RiskClass.WRITE
        assert plan.tool("find_customer").requires_approval is True


# ---------------------------------------------------------------------------
# Post-conditions
# ---------------------------------------------------------------------------


class TestPostConditions:
    def test_good_has_no_violations(self):
        assert _violations(GOOD) == []

    def test_budget(self):
        assert any("exceed the budget" in v for v in _violations(GOOD, max_tools=1))

    def test_unknown_operation(self):
        bad = {"tools": [{**GOOD["tools"][1], "operations": ["nukeEverything"]}], "dropped": []}
        assert any("unknown operation 'nukeEverything'" in v for v in _violations(bad))

    def test_operation_in_two_tools_and_kept_and_dropped(self):
        two = {
            "tools": [GOOD["tools"][0], {**GOOD["tools"][1], "operations": ["searchCustomers"]}],
            "dropped": [],
        }
        assert any("used by both" in v for v in _violations(two))
        both = {
            "tools": GOOD["tools"],
            "dropped": [{"operation_id": "searchCustomers", "reason": "x"}],
        }
        assert any("both kept" in v for v in _violations(both))

    def test_risk_never_downgraded(self):
        bad = {"tools": [{**GOOD["tools"][1], "risk": "read"}], "dropped": []}
        (v,) = [v for v in _violations(bad) if "downgrades" in v]
        assert "at least 'destructive'" in v
        # destructive <-> financial is not a downgrade
        assert not any(
            "downgrades" in v
            for v in _violations(
                {"tools": [{**GOOD["tools"][1], "risk": "financial"}], "dropped": []}
            )
        )

    def test_deprecated_must_be_dropped(self):
        bad = {
            "tools": [
                {
                    "name": "export",
                    "description": "x",
                    "risk": "read",
                    "operations": ["legacyExport"],
                }
            ],
            "dropped": [],
        }
        assert any("deprecated" in v for v in _violations(bad))

    def test_names(self):
        bad = {"tools": [{**GOOD["tools"][0], "name": "FindCustomer"}], "dropped": []}
        assert any("snake_case" in v for v in _violations(bad))
        dup = {
            "tools": [GOOD["tools"][0], {**GOOD["tools"][1], "name": "find_customer"}],
            "dropped": [],
        }
        assert any("duplicate tool name" in v for v in _violations(dup))
        blank = {"tools": [{**GOOD["tools"][0], "description": " "}], "dropped": []}
        assert any("empty description" in v for v in _violations(blank))

    def test_param_rules(self):
        unknown = {"tools": [{**GOOD["tools"][0], "params": {"ghost": {}}}], "dropped": []}
        assert any("unknown parameter 'ghost'" in v for v in _violations(unknown))
        hidden_required = {
            "tools": [{**GOOD["tools"][1], "params": {"id": {"hidden": True}}}],
            "dropped": [],
        }
        assert any("hides required parameter 'id'" in v for v in _violations(hidden_required))
        bad_example = {"tools": [{**GOOD["tools"][0], "example": {"limit": 3}}], "dropped": []}
        assert any("example uses unknown or hidden" in v for v in _violations(bad_example))

    def test_dropped_rules(self):
        bad = {
            "tools": [],
            "dropped": [
                {"operation_id": "nope", "reason": "x"},
                {"operation_id": "healthCheck", "reason": "a"},
                {"operation_id": "healthCheck", "reason": "b"},
            ],
        }
        vs = _violations(bad)
        assert any("dropped unknown operation" in v for v in vs)
        assert any("dropped twice" in v for v in vs)

    def test_apply_raises_with_all_violations(self):
        bad = {"tools": [{**GOOD["tools"][1], "risk": "read", "name": "Bad"}], "dropped": []}
        with pytest.raises(CurationViolation) as exc:
            apply_curation(
                CurationResult.model_validate(bad),
                OPS,
                CLASSES,
                profile=SafetyProfile.FULL,
                max_tools=25,
            )
        assert len(exc.value.violations) == 2


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


class TestCurateLoop:
    @pytest.mark.asyncio
    async def test_first_try(self):
        script = _script(GOOD)
        plan = await curate(
            OPS, profile=SafetyProfile.FULL, complete=script, name="crm", description="CRM"
        )
        assert plan.tool_names == ["find_customer", "cancel_subscription"]
        system, user = script.seen[0]
        assert system == CURATION_SYSTEM
        assert "Tool budget: at most 25 tools" in user and "getCustomerById" in user
        assert '"risk": "destructive"' in user  # classifier hints are in the catalogue

    @pytest.mark.asyncio
    async def test_violations_fed_back_then_accepted(self):
        bad = {"tools": [{**GOOD["tools"][1], "risk": "read"}], "dropped": []}
        script = _script(bad, GOOD)
        plan = await curate(OPS, profile=SafetyProfile.FULL, complete=script)
        assert len(script.seen) == 2
        assert "downgrades risk" in script.seen[1][1]
        assert plan.tool_names == ["find_customer", "cancel_subscription"]

    @pytest.mark.asyncio
    async def test_unparseable_then_accepted(self):
        script = _script(
            "Sure! Here is my plan: it's great.", "```json\n" + json.dumps(GOOD) + "\n```"
        )
        plan = await curate(OPS, profile=SafetyProfile.FULL, complete=script)
        assert "could not be parsed" in script.seen[1][1]
        assert len(plan.tools) == 2

    @pytest.mark.asyncio
    async def test_gives_up_loudly(self):
        bad = {"tools": [{**GOOD["tools"][1], "operations": ["nope"]}], "dropped": []}
        script = _script(bad, bad, bad)
        with pytest.raises(MCPcastError, match="failed after 3 attempt") as exc:
            await curate(OPS, profile=SafetyProfile.FULL, complete=script, max_attempts=3)
        assert "--no-curate" in str(exc.value)
        assert len(script.seen) == 3

    @pytest.mark.asyncio
    async def test_empty_ops_and_bad_budget(self):
        with pytest.raises(MCPcastError, match="no operations"):
            await curate([], complete=_script())
        with pytest.raises(ValueError, match="max_tools"):
            await curate(OPS, max_tools=0, complete=_script())

    def test_prompt_lists_every_operation_once(self):
        text = render_curation_prompt(
            OPS, CLASSES, max_tools=3, profile=SafetyProfile.STANDARD, api_name="crm"
        )
        for o in OPS:
            assert text.count(f'"operation_id": "{o.operation_id}"') == 1
        assert "Operations in the spec: 5" in text


# ---------------------------------------------------------------------------
# LLM plumbing
# ---------------------------------------------------------------------------


class TestLLMHelpers:
    def test_extract_json_object(self):
        assert extract_json_object('{"a": 1}') == {"a": 1}
        assert extract_json_object('text\n```json\n{"a": [1]}\n```\nmore') == {"a": [1]}
        assert extract_json_object('prose {"a": {"b": 2}} trailing') == {"a": {"b": 2}}
        with pytest.raises(MCPcastError, match="JSON object"):
            extract_json_object("[1, 2]")
        with pytest.raises(MCPcastError):
            extract_json_object("nothing here")

    def test_final_text_shapes(self):
        from langchain_core.messages import AIMessage

        assert final_text({"messages": [AIMessage(content="hi")]}) == "hi"
        assert (
            final_text(
                {
                    "messages": [
                        AIMessage(
                            content=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
                        )
                    ]
                }
            )
            == "ab"
        )
        assert final_text("raw") == "raw"

    @pytest.mark.asyncio
    async def test_complete_dogfoods_build_agent(self):
        """The production completer runs through build_agent with any chat model."""
        from unittest.mock import AsyncMock, MagicMock

        from langchain_core.messages import AIMessage

        from promptise.mcpcast._llm import complete

        model = MagicMock(spec=["ainvoke", "bind_tools", "with_structured_output"])
        model.ainvoke = AsyncMock(return_value=AIMessage(content='{"ok": true}'))
        model.bind_tools = MagicMock(return_value=model)
        assert await complete(model, "sys", "user") == '{"ok": true}'
        sent = model.ainvoke.call_args.args[0]
        assert [type(m).__name__ for m in sent] == ["SystemMessage", "HumanMessage"]
        assert sent[0].content == "sys" and sent[1].content == "user"


class TestUnmappableOperations:
    @pytest.mark.asyncio
    async def test_unmappable_ops_are_dropped_before_the_model_sees_them(self):
        broken = op("broken", "GET", "/things/{id}")
        script = _script(GOOD)
        plan = await curate([*OPS, broken], profile=SafetyProfile.FULL, complete=script)
        assert "broken" not in script.seen[0][1]
        reasons = {d.operation_id: d.reason for d in plan.dropped}
        assert reasons["broken"].startswith("unsupported by mcpcast:")
        assert len(plan.tools) == 2

    @pytest.mark.asyncio
    async def test_all_unmappable_is_an_error(self):
        with pytest.raises(MCPcastError, match="no operation in the spec can be mapped"):
            await curate([op("broken", "GET", "/things/{id}")], complete=_script())


class TestHardenedPostConditions:
    def test_reserved_and_keyword_names_rejected(self):
        for bad in ("import", "server", "upstream", "approvals_list", "str"):
            proposal = {"tools": [{**GOOD["tools"][0], "name": bad}], "dropped": []}
            assert any("keyword" in v or "reserved" in v for v in _violations(proposal)), bad

    def test_unreachable_operation_order_rejected(self):
        # searchCustomers needs nothing, so getCustomerById listed after it can never run
        proposal = {
            "tools": [{**GOOD["tools"][0], "operations": ["searchCustomers", "getCustomerById"]}],
            "dropped": [],
        }
        (v,) = [v for v in _violations(proposal) if "can never be selected" in v]
        assert "getCustomerById" in v and "searchCustomers" in v

    def test_hidden_required_anywhere(self):
        # id is required by getCustomerById even though optional on searchCustomers
        proposal = {
            "tools": [{**GOOD["tools"][0], "params": {"id": {"hidden": True}}}],
            "dropped": [],
        }
        assert any("hides required parameter 'id'" in v for v in _violations(proposal))
        ok = {
            "tools": [{**GOOD["tools"][0], "params": {"id": {"hidden": True, "default": "me"}}}],
            "dropped": [],
        }
        # hidden id with a default satisfies both routes → searchCustomers becomes unreachable
        assert any("can never be selected" in v for v in _violations(ok))


class TestCurateLoopHardening:
    @pytest.mark.asyncio
    async def test_plan_level_errors_never_reach_the_model(self):
        ops = [op("a", "GET", "/a")]
        ops[0].base_url = ""
        script = _script(GOOD)
        with pytest.raises(MCPcastError, match="--base-url"):
            await curate(ops, complete=script)
        assert script.seen == []  # no model call was made

    @pytest.mark.asyncio
    async def test_feedback_carries_previous_proposal(self):
        bad = {"tools": [{**GOOD["tools"][1], "risk": "read"}], "dropped": []}
        script = _script(bad, GOOD)
        await curate(OPS, profile=SafetyProfile.FULL, complete=script)
        assert "Your previous proposal:" in script.seen[1][1]
        assert '"cancel_subscription"' in script.seen[1][1]

    @pytest.mark.asyncio
    async def test_max_attempts_and_prompt_size(self):
        with pytest.raises(ValueError, match="max_attempts"):
            await curate(OPS, complete=_script(), max_attempts=0)
        from promptise.mcpcast.curate import MAX_PROMPT_CHARS

        huge = [op(f"op{i}", "GET", f"/p{i}", summary="x" * 4000) for i in range(200)]
        script = _script()
        with pytest.raises(MCPcastError, match="narrow the spec"):
            await curate(huge, complete=script)
        assert script.seen == []
        assert MAX_PROMPT_CHARS == 600_000

    @pytest.mark.asyncio
    async def test_multi_route_example_prefers_first_route(self):
        tool = {**GOOD["tools"][0]}
        tool.pop("example")
        plan = await curate(
            OPS,
            profile=SafetyProfile.FULL,
            complete=_script({"tools": [tool], "dropped": GOOD["dropped"]}),
        )
        assert plan.tool("find_customer").example == {"id": "123"}


def test_final_text_skips_tool_messages():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    assert (
        final_text(
            {
                "messages": [
                    HumanMessage(content="q"),
                    AIMessage(content="a"),
                    ToolMessage(content="echo:3", tool_call_id="1"),
                ]
            }
        )
        == "a"
    )
    assert (
        final_text(
            {
                "messages": [
                    HumanMessage(content="q"),
                    ToolMessage(content="echo:3", tool_call_id="1"),
                ]
            }
        )
        == ""
    )
    assert (
        final_text({"messages": [{"role": "assistant", "content": "dict-style"}]}) == "dict-style"
    )


class TestExampleShapeIsRepaired:
    """A curated example that contradicts the API is replaced, not fatal.

    An example is a hint to the agent, so a wrong one is worth discarding —
    failing the whole run over it would strand the user (the model reliably
    re-invents the same plausible-but-wrong shape on every retry).
    """

    OPS = [
        op(
            "createOrder",
            "POST",
            "/orders",
            [
                ParamSpec(
                    name="items",
                    location="body",
                    required=True,
                    json_schema={
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string"},
                                "qty": {"type": "integer"},
                            },
                        },
                    },
                ),
                ParamSpec(name="rush", location="body", json_schema={"type": "boolean"}),
                ParamSpec(
                    name="channel",
                    location="body",
                    json_schema={"type": "string", "enum": ["web", "pos"]},
                ),
            ],
        )
    ]

    def _plan_with_example(self, example):
        proposal = {
            "tools": [
                {
                    "name": "create_order",
                    "description": "Create an order.",
                    "risk": "write",
                    "operations": ["createOrder"],
                    "example": example,
                }
            ],
            "dropped": [],
        }
        classes = {o.operation_id: classify(o) for o in self.OPS}
        return apply_curation(
            CurationResult.model_validate(proposal),
            self.OPS,
            classes,
            profile=SafetyProfile.FULL,
            max_tools=25,
        ).tool("create_order")

    def test_invented_item_fields_are_replaced_with_a_spec_derived_example(self):
        # The real item schema is {sku, qty}; the model wrote unit_price_cents.
        tool = self._plan_with_example(
            {"items": [{"sku": "A1", "quantity": 2, "unit_price_cents": 500}]}
        )
        assert tool.example == {"items": [{"sku": "string", "qty": 1}]}

    def test_wrong_types_and_enums_are_dropped_but_good_keys_survive(self):
        tool = self._plan_with_example({"items": [{"sku": "A1", "qty": 2}], "rush": "yes"})
        assert tool.example == {"items": [{"sku": "A1", "qty": 2}]}
        assert self._plan_with_example({"channel": "carrier-pigeon"}).example == {
            "items": [{"sku": "string", "qty": 1}]
        }

    def test_faithful_example_is_kept_verbatim(self):
        example = {"items": [{"sku": "A1", "qty": 2}], "rush": True, "channel": "web"}
        assert self._plan_with_example(example).example == example

    def test_freeform_object_is_not_second_guessed(self):
        ops = [
            op(
                "freeform",
                "POST",
                "/f",
                [
                    ParamSpec(
                        name="meta",
                        location="body",
                        required=True,
                        json_schema={"type": "object", "additionalProperties": True},
                    )
                ],
            )
        ]
        proposal = {
            "tools": [
                {
                    "name": "send_meta",
                    "description": "Send.",
                    "risk": "write",
                    "operations": ["freeform"],
                    "example": {"meta": {"anything": 1}},
                }
            ],
            "dropped": [],
        }
        plan = apply_curation(
            CurationResult.model_validate(proposal),
            ops,
            {o.operation_id: classify(o) for o in ops},
            profile=SafetyProfile.FULL,
            max_tools=25,
        )
        assert plan.tool("send_meta").example == {"meta": {"anything": 1}}

    def test_a_wrong_example_never_fails_the_run(self):
        from promptise.mcpcast.curate import example_mismatch

        assert example_mismatch({"quantity": 1}, {"properties": {"qty": {}}, "type": "object"})
        assert example_mismatch("yes", {"type": "boolean"})
        assert example_mismatch("x", {"type": "string"}) is None


def test_curation_prompt_states_the_dispatch_rule():
    """Tight budgets used to fail because the model merged indistinguishable routes."""
    assert "DISPATCH RULE" in CURATION_SYSTEM
    assert "required parameters" in CURATION_SYSTEM and "listed last" in CURATION_SYSTEM


# ---------------------------------------------------------------------------
# Refusals on the curated path never echo a value
# ---------------------------------------------------------------------------


class TestCuratedRefusalsNeverEchoTheValue:
    @pytest.mark.asyncio
    async def test_base_url_with_a_credential_is_refused_before_the_model_is_called(self):
        script = _script(GOOD)
        with pytest.raises(MCPcastError) as info:
            await curate(OPS, base_url="https://alice:S3CRET@api.acme.test/v1", complete=script)
        message = str(info.value)
        assert message.startswith(
            "could not build plan:\n1 validation error for ApiPlan\nbase_url\n"
        )
        assert "must not carry credentials" in message and "'api.acme.test'" in message
        assert "S3CRET" not in message and "alice" not in message
        assert "input_value" not in message
        assert script.seen == []

    @pytest.mark.asyncio
    async def test_query_string_in_base_url_is_refused_the_same_way(self):
        with pytest.raises(MCPcastError) as info:
            await curate(
                OPS, base_url="https://api.acme.test/v1?api_key=QSECRET", complete=_script()
            )
        assert "must not carry a query string or fragment" in str(info.value)
        assert "QSECRET" not in str(info.value) and "input_value" not in str(info.value)

    def test_plan_level_violation_is_rendered_without_input_value(self):
        """apply_curation's own refusal is fed back to the model on retry."""
        ops = [op("a", "GET", "/a")]
        ops[0].base_url = "https://svc:S3CRET@api.acme.test"
        ops[0].doc_base_url = ops[0].base_url
        proposal = {
            "tools": [{"name": "a", "description": "A.", "risk": "read", "operations": ["a"]}],
            "dropped": [],
        }
        with pytest.raises(CurationViolation) as info:
            apply_curation(
                CurationResult.model_validate(proposal),
                ops,
                {o.operation_id: classify(o) for o in ops},
                profile=SafetyProfile.FULL,
                max_tools=5,
            )
        (violation,) = info.value.violations
        assert violation.startswith("1 validation error for ApiPlan\nbase_url\n")
        assert "S3CRET" not in violation and "input_value" not in violation

    @pytest.mark.asyncio
    async def test_operation_server_with_a_credential_is_dropped_before_the_model_sees_it(self):
        from promptise.mcpcast.parse import extract_operations

        spec = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://api.acme.test/v1"}],
            "paths": {
                "/customers/{id}": {
                    "get": {
                        "operationId": "getCustomerById",
                        "parameters": [{"name": "id", "in": "path", "required": True}],
                    }
                },
                "/customers": {
                    "get": {
                        "operationId": "searchCustomers",
                        "parameters": [
                            {"name": "email", "in": "query"},
                            {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                        ],
                    }
                },
                "/customers/{id}/subscriptions:cancel": {
                    "post": {
                        "operationId": "cancelSubscription",
                        "parameters": [{"name": "id", "in": "path", "required": True}],
                    }
                },
                "/health": {"get": {"operationId": "healthCheck"}},
                "/export": {"get": {"operationId": "legacyExport", "deprecated": True}},
                "/reports": {
                    "get": {
                        "operationId": "reports",
                        "servers": [{"url": "https://bob:ROUTESECRET@reports.acme.test/v2"}],
                    }
                },
            },
        }
        ops = extract_operations(spec)
        script = _script(GOOD)
        plan = await curate(ops, profile=SafetyProfile.FULL, complete=script)
        assert "ROUTESECRET" not in script.seen[0][1] and "reports" not in script.seen[0][1]
        reasons = {d.operation_id: d.reason for d in plan.dropped}
        assert reasons["reports"].startswith(
            "unsupported by mcpcast: route base_url must not carry"
        )
        assert "ROUTESECRET" not in plan.to_yaml()


# ---------------------------------------------------------------------------
# Credential slot on the curated path
# ---------------------------------------------------------------------------


def _keyed(ops_security=None):
    from promptise.mcpcast.parse import extract_operations

    spec = {
        "openapi": "3.0.0",
        "servers": [{"url": "https://k.example"}],
        "components": {
            "securitySchemes": {"key": {"type": "apiKey", "in": "header", "name": "X-API-Key"}}
        },
        "security": [{"key": []}],
        "paths": {
            "/customers/{id}": {
                "get": {
                    "operationId": "getCustomerById",
                    "parameters": [{"name": "id", "in": "path", "required": True}],
                }
            },
            "/customers": {
                "get": {
                    "operationId": "searchCustomers",
                    "parameters": [
                        {"name": "email", "in": "query"},
                        {"name": "limit", "in": "query", "schema": {"type": "integer"}},
                    ],
                }
            },
            "/customers/{id}/subscriptions:cancel": {
                "post": {
                    "operationId": "cancelSubscription",
                    "parameters": [{"name": "id", "in": "path", "required": True}],
                }
            },
            "/health": {"get": {"operationId": "healthCheck"}},
            "/export": {"get": {"operationId": "legacyExport", "deprecated": True}},
        },
    }
    return extract_operations(spec)


class TestCuratedCredentialSlot:
    @pytest.mark.asyncio
    async def test_passthrough_is_refused_before_the_model_is_called(self):
        script = _script(GOOD)
        with pytest.raises(MCPcastError, match="header 'X-API-Key'.*passthrough cannot relay"):
            await curate(_keyed(), profile=SafetyProfile.FULL, complete=script)
        assert script.seen == []

    @pytest.mark.asyncio
    async def test_env_token_records_the_slot(self):
        from promptise.mcpcast.schema import AuthMode

        plan = await curate(
            _keyed(), profile=SafetyProfile.FULL, auth=AuthMode.ENV_TOKEN, complete=_script(GOOD)
        )
        assert (plan.api.credential_location, plan.api.credential_name) == ("header", "X-API-Key")
        assert plan.tool_names == ["find_customer", "cancel_subscription"]

    def test_apply_curation_refuses_passthrough_too(self):
        ops = _keyed()
        with pytest.raises(MCPcastError, match="passthrough cannot relay"):
            apply_curation(
                CurationResult.model_validate(GOOD),
                ops,
                {o.operation_id: classify(o) for o in ops},
                profile=SafetyProfile.FULL,
                max_tools=25,
            )


# ---------------------------------------------------------------------------
# Descriptions never name a tool the server does not have
# ---------------------------------------------------------------------------

PET_ID = ParamSpec(name="petId", location="path", required=True)
PETS = [
    op("getPetById", "GET", "/pet/{petId}", [PET_ID], summary="Find pet by ID"),
    op(
        "findPetsByStatus",
        "GET",
        "/pet/findByStatus",
        [ParamSpec(name="status", location="query", description="Status to filter by")],
        summary="Finds pets by status",
    ),
    op(
        "updatePetWithForm",
        "POST",
        "/pet/{petId}",
        [PET_ID, ParamSpec(name="name", location="query")],
        summary="Updates a pet with form data",
    ),
    op("deletePet", "DELETE", "/pet/{petId}", [PET_ID], summary="Deletes a pet"),
]
FIND_PETS = {
    "name": "find_pets",
    "description": (
        "Look up pets by id or by status. Do not use this tool to change pets — "
        "use update_pet_form or delete_pet for those actions. Returns pet objects."
    ),
    "risk": "read",
    "operations": ["getPetById", "findPetsByStatus"],
}
UPDATE_PET_FORM = {
    "name": "update_pet_form",
    "description": "Rename a pet. Related tools: find_pets.",
    "risk": "write",
    "operations": ["updatePetWithForm"],
}
DELETE_PET = {
    "name": "delete_pet",
    "description": "Delete a pet permanently.",
    "risk": "destructive",
    "operations": ["deletePet"],
}
PHANTOM = {"tools": [FIND_PETS, UPDATE_PET_FORM, DELETE_PET], "dropped": []}
FIXED = {
    "tools": [
        {
            **FIND_PETS,
            "description": "Look up pets by id or by status. Use update_pet_form to rename one.",
        },
        UPDATE_PET_FORM,
        DELETE_PET,
    ],
    "dropped": [],
}


def _mentions(plan, name: str) -> bool:
    texts = [t.description for t in plan.tools]
    texts += [p.description for t in plan.tools for p in t.params.values()]
    return any(name in text for text in texts)


class TestDanglingToolReferences:
    """The petstore repro: ``delete_pet`` is excluded by ``standard`` but named elsewhere."""

    @pytest.mark.asyncio
    async def test_reference_to_a_profile_excluded_tool_is_fed_back(self):
        script = _script(PHANTOM, FIXED)
        plan = await curate(PETS, profile=SafetyProfile.STANDARD, complete=script)
        assert len(script.seen) == 2
        feedback = script.seen[1][1]
        assert "tool 'find_pets' description names delete_pet" in feedback
        assert "excluded by profile 'standard'" in feedback
        assert "find_pets, update_pet_form" in feedback  # the tools it may name instead
        assert plan.tool_names == ["find_pets", "update_pet_form"]
        assert "deletePet" in {d.operation_id for d in plan.dropped}
        assert not _mentions(plan, "delete_pet")

    @pytest.mark.asyncio
    async def test_last_attempt_strips_the_sentence_instead_of_failing(self):
        script = _script(PHANTOM, PHANTOM, PHANTOM)
        plan = await curate(PETS, profile=SafetyProfile.STANDARD, complete=script)
        assert len(script.seen) == 3
        assert all("names delete_pet" in user for _, user in script.seen[1:])
        find = plan.tool("find_pets")
        assert find.description == "Look up pets by id or by status. Returns pet objects."
        assert not _mentions(plan, "delete_pet")
        # Nothing else is touched: references to exposed tools stay.
        assert plan.tool("update_pet_form").description == "Rename a pet. Related tools: find_pets."

    @pytest.mark.asyncio
    async def test_parameter_descriptions_are_checked_and_repaired(self):
        bad_param = {
            **UPDATE_PET_FORM,
            "params": {"name": {"description": "New name. To remove the pet use delete_pet."}},
        }
        proposal = {"tools": [FIXED["tools"][0], bad_param, DELETE_PET], "dropped": []}
        script = _script(proposal, proposal)
        plan = await curate(PETS, profile=SafetyProfile.STANDARD, complete=script, max_attempts=2)
        assert (
            "tool 'update_pet_form' parameter 'name' description names delete_pet"
            in (script.seen[1][1])
        )
        assert plan.tool("update_pet_form").params["name"].description == "New name."

    def test_merged_operation_name_points_at_the_tool_that_serves_it(self):
        proposal = {
            "tools": [
                FIXED["tools"][0],
                {**UPDATE_PET_FORM, "description": "Rename a pet; get_pet_by_id shows it."},
            ],
            "dropped": [{"operation_id": "deletePet", "reason": "too dangerous"}],
        }
        with pytest.raises(CurationViolation) as exc:
            apply_curation(
                CurationResult.model_validate(proposal),
                PETS,
                {o.operation_id: classify(o) for o in PETS},
                profile=SafetyProfile.FULL,
                max_tools=25,
            )
        (violation,) = exc.value.violations
        assert "names get_pet_by_id" in violation
        assert "'getPetById' is served by 'find_pets'" in violation

    def test_unmappable_operations_count_through_spec_operations(self):
        broken = op("exportPets", "GET", "/pets/export/{format}")  # no path parameter
        proposal = {
            "tools": [
                {**FIXED["tools"][0], "description": "Find pets; for a dump use export_pets."}
            ],
            "dropped": [
                {"operation_id": "updatePetWithForm", "reason": "x"},
                {"operation_id": "deletePet", "reason": "x"},
            ],
        }
        classes = {o.operation_id: classify(o) for o in PETS}
        args = (CurationResult.model_validate(proposal), PETS, classes)
        kw = {"profile": SafetyProfile.FULL, "max_tools": 25}
        apply_curation(*args, **kw)  # not part of the plan's operations: unknown here
        with pytest.raises(CurationViolation, match="names export_pets"):
            apply_curation(*args, **kw, spec_operations=[*PETS, broken])

    def test_parameter_names_and_single_words_are_not_tools(self):
        ops = [
            op("search", "GET", "/pets", [ParamSpec(name="pet_status", location="query")]),
            op("petStatus", "GET", "/status"),
        ]
        proposal = {
            "tools": [
                {
                    "name": "find_pets",
                    "description": "Search pets by pet_status; use search terms freely.",
                    "risk": "read",
                    "operations": ["search"],
                }
            ],
            "dropped": [{"operation_id": "petStatus", "reason": "internal"}],
        }
        plan = apply_curation(
            CurationResult.model_validate(proposal),
            ops,
            {o.operation_id: classify(o) for o in ops},
            profile=SafetyProfile.READ_ONLY,
            max_tools=25,
        )
        assert plan.tools[0].description.startswith("Search pets by pet_status")

    def test_a_description_left_empty_falls_back_to_the_spec(self):
        proposal = {
            "tools": [{**FIND_PETS, "description": "Use delete_pet to remove one."}, DELETE_PET],
            "dropped": [{"operation_id": "updatePetWithForm", "reason": "x"}],
        }
        plan = apply_curation(
            CurationResult.model_validate(proposal),
            PETS,
            {o.operation_id: classify(o) for o in PETS},
            profile=SafetyProfile.READ_ONLY,
            max_tools=25,
            repair_references=True,
        )
        assert plan.tool("find_pets").description == "Find pet by ID"

    def test_prompt_names_the_classes_the_profile_exposes(self):
        text = render_curation_prompt(
            PETS,
            {o.operation_id: classify(o) for o in PETS},
            max_tools=5,
            profile=SafetyProfile.STANDARD,
        )
        assert "exposes only these risk classes: read, write;" in text
        assert "ONLY tools in your" in CURATION_SYSTEM
