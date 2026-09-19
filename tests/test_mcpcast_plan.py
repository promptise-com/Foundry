"""Tests for the deterministic planner (``promptise.mcpcast.plan``)."""

from __future__ import annotations

import pytest

from promptise.mcpcast import mcpcast
from promptise.mcpcast.parse import Operation, ParamSpec, extract_operations
from promptise.mcpcast.plan import (
    build_plan,
    derive_tool_name,
    example_value,
    make_example,
    route_base_url,
)
from promptise.mcpcast.schema import (
    AuthMode,
    MCPcastError,
    ParamPlan,
    RiskClass,
    SafetyProfile,
    valid_tool_name,
)

BASE = "https://api.example.com"


def op(op_id, method, path, params=(), **kw):
    return Operation(
        operation_id=op_id, method=method, path=path, params=list(params), base_url=BASE, **kw
    )


OPS = [
    op(
        "listPets",
        "GET",
        "/pets",
        [ParamSpec(name="limit", location="query", json_schema={"type": "integer"})],
    ),
    op(
        "getPet",
        "GET",
        "/pets/{id}",
        [
            ParamSpec(name="id", location="path", required=True, json_schema={"type": "integer"}),
            ParamSpec(name="X-Trace", location="header"),
        ],
    ),
    op("createPet", "POST", "/pets", [ParamSpec(name="name", location="body", required=True)]),
    op("deletePet", "DELETE", "/pets/{id}", [ParamSpec(name="id", location="path", required=True)]),
    op("charge", "POST", "/charges"),
    op("oldThing", "GET", "/old", deprecated=True),
]


class TestNaming:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("getPetById", "get_pet_by_id"),
            ("get-pet", "get_pet"),
            ("GET_users", "get_users"),
            ("HTTPRequest", "httprequest"),
            ("import", "import_op"),
            ("123abc", "op_123abc"),
            ("", "operation"),
            ("a" * 80, "a" * 64),
            # soft keywords and the names the generated server reserves are
            # suffixed, never dropped
            ("list", "list_op"),
            ("set", "set_op"),
            ("type", "type_op"),
            ("match", "match_op"),
            ("case", "case_op"),
            ("main", "main_op"),
            ("server", "server_op"),
            ("selectRoute", "select_route_op"),
            ("register", "register_op"),
            ("approvals_list", "approvals_list_op"),
            ("_", "operation"),
        ],
    )
    def test_derive(self, raw, expected):
        assert derive_tool_name(raw) == expected
        assert valid_tool_name(expected) is None

    def test_unique_and_reserved(self):
        taken: set[str] = set()
        assert derive_tool_name("approvals_list", taken) == "approvals_list_op"
        assert derive_tool_name("approvals_list", taken) == "approvals_list_op_2"
        assert derive_tool_name("getPet", taken) == "get_pet"
        assert derive_tool_name("get_pet", taken) == "get_pet_2"
        assert derive_tool_name("get-pet", taken) == "get_pet_3"
        assert derive_tool_name("x" * 64, taken) == "x" * 64
        assert derive_tool_name("x" * 64, taken) == "x" * 62 + "_2"

    def test_reserved_operation_ids_become_tools_not_drops(self):
        """``list``, ``type``, ``match``… are common operationIds: renamed, not "unsupported"."""
        ids = ["list", "set", "main", "type", "match", "case", "ctx", "args", "str", "object"]
        plan = build_plan([op(i, "GET", f"/{i}") for i in ids])
        assert plan.dropped == []
        assert plan.tool_names == [f"{i}_op" for i in ids]


class TestExamples:
    @pytest.mark.parametrize(
        ("schema", "name", "expected"),
        [
            ({"example": 5}, "x", 5),
            ({"default": "d", "example": None}, "x", "d"),
            ({"examples": ["e"]}, "x", "e"),
            ({"enum": ["a", "b"]}, "x", "a"),
            ({"type": "integer", "minimum": 10}, "x", 10),
            ({"type": "number"}, "x", 1.0),
            ({"type": "boolean"}, "x", True),
            ({"type": ["null", "string"], "format": "date"}, "x", "2026-01-15"),
            ({"type": "array", "items": {"type": "integer"}}, "x", [1]),
            ({"type": "object", "properties": {"n": {"type": "string"}}}, "x", {"n": "string"}),
            ({"anyOf": [{"type": "integer"}, {"type": "string"}]}, "x", 1),
            ({"type": "string", "format": "email"}, "x", "ada@example.com"),
            ({"type": "string"}, "customerId", "123"),
            ({"type": "string"}, "displayName", "example"),
            ({}, "whatever", "string"),
        ],
    )
    def test_example_value(self, schema, name, expected):
        assert example_value(schema, name) == expected

    def test_make_example_only_required_visible(self):
        params = {
            "id": ParamPlan(required=True, json_schema={"type": "integer"}),
            "opt": ParamPlan(),
            "hid": ParamPlan(required=True, hidden=True, default="x"),
        }
        assert make_example(params) == {"id": 1}
        assert make_example({"opt": ParamPlan()}) is None


class TestBuildPlan:
    def test_read_only_default(self):
        plan = build_plan(OPS, name="pets")
        assert plan.tool_names == ["list_pets", "get_pet"]
        assert plan.api.base_url == BASE and plan.api.auth is AuthMode.PASSTHROUGH
        reasons = {d.operation_id: d.reason for d in plan.dropped}
        assert reasons["createPet"] == "write operation excluded by profile 'read-only'"
        assert reasons["deletePet"].startswith("destructive operation excluded")
        assert reasons["charge"].startswith("financial operation excluded")
        assert reasons["oldThing"] == "deprecated in spec"
        assert not any(t.requires_approval for t in plan.tools)

    def test_standard_gates_writes(self):
        plan = build_plan(OPS, profile=SafetyProfile.STANDARD)
        assert plan.tool("create_pet").requires_approval is True
        assert plan.tool("create_pet").risk is RiskClass.WRITE
        assert "deletePet" in {d.operation_id for d in plan.dropped}

    def test_full_exposes_everything_gated(self):
        plan = build_plan(OPS, profile=SafetyProfile.FULL)
        assert {t.name for t in plan.gated_tools} == {"create_pet", "delete_pet", "charge"}
        assert plan.tool("charge").risk is RiskClass.FINANCIAL
        assert [d.operation_id for d in plan.dropped] == ["oldThing"]

    def test_header_params_not_exposed(self):
        plan = build_plan(OPS)
        get = plan.tool("get_pet")
        assert list(get.params) == ["id"]
        assert get.routes[0].params["id"].location == "path"
        assert get.example == {"id": 1}

    def test_budget_keeps_reads_first_with_reason(self):
        plan = build_plan(OPS, profile=SafetyProfile.FULL, max_tools=3)
        assert plan.tool_names == ["list_pets", "get_pet", "create_pet"]
        over = [d for d in plan.dropped if "over tool budget" in d.reason]
        assert {d.operation_id for d in over} == {"deletePet", "charge"}
        with pytest.raises(ValueError, match="max_tools"):
            build_plan(OPS, max_tools=0)

    def test_missing_base_url_is_actionable(self):
        ops = [op("a", "GET", "/a")]
        ops[0].base_url = ""
        with pytest.raises(MCPcastError, match="--base-url"):
            build_plan(ops)
        assert (
            build_plan(ops, base_url="http://localhost:8000/").api.base_url
            == "http://localhost:8000"
        )

    def test_annotations_and_descriptions(self):
        ops = [op("x", "GET", "/x", summary="Sum", description="Long text")]
        plan = build_plan(ops)
        assert plan.tools[0].description == "Sum. Long text"
        assert build_plan([op("y", "GET", "/y")]).tools[0].description == "GET /y"


class TestRouteBaseUrl:
    """Only an operation with its *own* ``servers`` carries a host on its route."""

    SPEC = {  # a FastAPI app: no servers block at all
        "openapi": "3.1.0",
        "info": {"title": "Shop"},
        "paths": {
            "/items": {"get": {"operationId": "list_items"}},
            "/items/{id}": {
                "get": {
                    "operationId": "get_item",
                    "parameters": [
                        {"name": "id", "in": "path", "required": True, "schema": {"type": "string"}}
                    ],
                }
            },
            "/reports": {
                "get": {
                    "operationId": "reports",
                    "servers": [{"url": "https://reports.example/v2"}],
                }
            },
        },
    }

    def test_spec_url_origin_is_not_frozen_into_routes(self):
        ops = extract_operations(self.SPEC, spec_url="http://127.0.0.1:8000/openapi.json")
        by_id = {o.operation_id: o for o in ops}
        assert by_id["list_items"].base_url == "http://127.0.0.1:8000"
        assert route_base_url(by_id["list_items"], "http://127.0.0.1:8000") is None
        # the wizard re-plans with a base the user typed: still no per-route host
        assert route_base_url(by_id["list_items"], "https://shop.example/api") is None
        # an operation that declares its own server keeps it
        assert route_base_url(by_id["reports"], "https://shop.example/api") == (
            "https://reports.example/v2"
        )
        plan = build_plan(ops, base_url="https://shop.example/api")
        assert plan.api.base_url == "https://shop.example/api"
        assert plan.tool("list_items").routes[0].base_url is None
        assert plan.tool("get_item").routes[0].base_url is None
        assert plan.tool("reports").routes[0].base_url == "https://reports.example/v2"

    def test_own_server_equal_to_plan_base_is_redundant(self):
        ops = extract_operations(self.SPEC)  # no spec URL either
        by_id = {o.operation_id: o for o in ops}
        assert by_id["list_items"].base_url == ""
        assert route_base_url(by_id["list_items"], "https://x") is None
        assert route_base_url(by_id["reports"], "https://reports.example/v2") is None
        assert route_base_url(by_id["reports"], "https://x") == "https://reports.example/v2"
        assert route_base_url(by_id["reports"], None) == "https://reports.example/v2"

    def test_forced_base_url_wins_everywhere(self):
        ops = extract_operations(self.SPEC, base_url="https://forced.example")
        assert all(route_base_url(o, "https://forced.example") is None for o in ops)


def test_mcpcast_end_to_end_from_dict():
    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Widgets API", "description": "Widgets API\nLine two"},
        "servers": [{"url": "https://w.example.com"}],
        "paths": {
            "/widgets": {
                "get": {"operationId": "listWidgets"},
                "post": {"operationId": "createWidget"},
            },
        },
    }
    plan = mcpcast(spec, profile=SafetyProfile.STANDARD)
    assert plan.api.name == "widgets"
    assert plan.api.description == "Widgets API"
    assert plan.api.spec_source is None
    assert plan.tool_names == ["list_widgets", "create_widget"]
    assert extract_operations(spec)[0].operation_id == "listWidgets"


class TestUnmappable:
    def test_bad_placeholder_is_dropped_with_reason_not_fatal(self):
        from promptise.mcpcast.plan import unmappable_reason

        bad = op("broken", "GET", "/things/{id}")  # placeholder, no path param
        assert unmappable_reason(bad) is not None and "placeholders" in unmappable_reason(bad)
        assert unmappable_reason(OPS[1]) is None
        plan = build_plan([bad, OPS[1]])
        assert plan.tool_names == ["get_pet"]
        (dropped,) = plan.dropped
        assert dropped.operation_id == "broken"
        assert dropped.reason.startswith("unsupported by mcpcast:")

    def test_collision_route_keeps_wire_names(self):
        spec = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://x"}],
            "paths": {
                "/user/{username}": {
                    "put": {
                        "operationId": "updateUser",
                        "parameters": [
                            {
                                "name": "username",
                                "in": "path",
                                "required": True,
                                "schema": {"type": "string"},
                            }
                        ],
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"username": {"type": "string"}},
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }
        plan = mcpcast(spec, profile=SafetyProfile.STANDARD)
        route = plan.tool("update_user").routes[0]
        assert set(route.params) == {"username", "body_username"}
        assert route.params["body_username"].wire_name == "username"
        assert route.wire("body_username") == "username" and route.wire("username") == "username"


def test_inline_document_is_not_recorded_as_source():
    import json

    spec = {
        "openapi": "3.0.0",
        "info": {"title": "Inline"},
        "servers": [{"url": "https://i.example"}],
        "paths": {"/x": {"get": {"operationId": "x"}}},
    }
    plan = mcpcast(json.dumps(spec))
    assert plan.api.spec_source == "<inline>" and plan.api.name == "inline"


# ---------------------------------------------------------------------------
# Route-level base URLs go through the same validator as api.base_url
# ---------------------------------------------------------------------------


def _spec_with_route_server(url: str) -> dict:
    return {
        "openapi": "3.0.0",
        "info": {"title": "Acme"},
        "servers": [{"url": "https://api.acme.test/v1"}],
        "paths": {
            "/things": {"get": {"operationId": "listThings"}},
            "/reports": {
                "get": {"operationId": "reports", "servers": [{"url": url}]},
            },
        },
    }


class TestRouteBaseUrlIsValidated:
    @pytest.mark.parametrize(
        "url",
        [
            "https://bob:ROUTESECRET@reports.acme.test/v2",
            "https://ROUTESECRET@reports.acme.test/v2",
            "https://bob:ROUTESECRET@reports.acme.test/v 2",
        ],
    )
    def test_credential_in_an_operation_server_url_is_dropped_not_copied(self, url, tmp_path):
        from promptise.mcpcast import write_project
        from promptise.mcpcast.plan import unmappable_reason

        ops = extract_operations(_spec_with_route_server(url))
        by_id = {o.operation_id: o for o in ops}
        reason = unmappable_reason(by_id["reports"], base_url="https://api.acme.test/v1")
        assert reason is not None and reason.startswith("unsupported by mcpcast: route base_url")
        assert "must not carry credentials" in reason and "'reports.acme.test'" in reason
        assert "ROUTESECRET" not in reason and "bob" not in reason

        plan = build_plan(ops)
        assert plan.tool_names == ["list_things"]
        (dropped,) = plan.dropped
        assert dropped.operation_id == "reports" and dropped.reason == reason
        assert "ROUTESECRET" not in plan.to_yaml()

        out = tmp_path / "acme-mcp"
        write_project(plan, out)
        for path in out.rglob("*"):
            if path.is_file():
                assert "ROUTESECRET" not in path.read_text(), path

    def test_query_string_in_an_operation_server_url_is_dropped(self):
        ops = extract_operations(
            _spec_with_route_server("https://reports.acme.test/v2?key=RSECRET")
        )
        plan = build_plan(ops)
        (dropped,) = plan.dropped
        assert "must not carry a query string or fragment" in dropped.reason
        assert "RSECRET" not in dropped.reason

    def test_clean_operation_server_is_kept_on_the_route(self):
        ops = extract_operations(_spec_with_route_server("https://reports.acme.test/v2/"))
        plan = build_plan(ops)
        assert plan.dropped == []
        assert plan.tool("reports").routes[0].base_url == "https://reports.acme.test/v2"
        assert plan.tool("list_things").routes[0].base_url is None

    def test_route_from_operation_takes_the_plan_base(self):
        from promptise.mcpcast.plan import route_from_operation

        ops = {
            o.operation_id: o for o in extract_operations(_spec_with_route_server("https://r.test"))
        }
        assert route_from_operation(ops["reports"]).base_url == "https://r.test"
        assert route_from_operation(ops["reports"], base_url="https://r.test").base_url is None
        assert (
            route_from_operation(ops["listThings"], base_url="https://api.acme.test/v1").base_url
            is None
        )


# ---------------------------------------------------------------------------
# Required header / cookie parameters
# ---------------------------------------------------------------------------


class TestRequiredHeaderParameters:
    def _op(self, name, location="header", required=True):
        return op(
            "listThings",
            "GET",
            "/things",
            [
                ParamSpec(name="q", location="query"),
                ParamSpec(name=name, location=location, required=required),
            ],
        )

    def test_required_custom_header_drops_the_operation(self):
        from promptise.mcpcast.plan import unmappable_reason

        reason = unmappable_reason(self._op("X-Tenant"))
        assert (
            reason == "unsupported by mcpcast: required header parameter 'X-Tenant' cannot be sent"
        )
        plan = build_plan([self._op("X-Tenant")])
        assert plan.tools == [] and plan.dropped[0].reason == reason

    def test_required_cookie_drops_the_operation(self):
        from promptise.mcpcast.plan import unmappable_reason

        assert unmappable_reason(self._op("session", "cookie")) == (
            "unsupported by mcpcast: required cookie parameter 'session' cannot be sent"
        )

    @pytest.mark.parametrize(
        "name", ["Authorization", "authorization", "Content-Type", "Accept", "Host", "User-Agent"]
    )
    def test_headers_the_runtime_sends_are_fine(self, name):
        from promptise.mcpcast.plan import unmappable_reason

        assert unmappable_reason(self._op(name)) is None
        assert build_plan([self._op(name)]).tool_names == ["list_things"]

    def test_optional_header_is_fine(self):
        from promptise.mcpcast.plan import unmappable_reason

        assert unmappable_reason(self._op("X-Tenant", required=False)) is None

    def test_the_plans_credential_header_counts_as_sent(self):
        from promptise.mcpcast.plan import CredentialSlot, unmappable_reason

        operation = self._op("X-API-Key")
        assert unmappable_reason(operation) is not None
        assert (
            unmappable_reason(operation, credential=CredentialSlot("header", "x-api-key")) is None
        )
        assert (
            unmappable_reason(operation, credential=CredentialSlot("query", "X-API-Key"))
            is not None
        )


# ---------------------------------------------------------------------------
# Credential slot from the spec's security schemes
# ---------------------------------------------------------------------------


def _keyed_spec(scheme: dict, *, security=None, paths=None) -> dict:
    return {
        "openapi": "3.0.0",
        "info": {"title": "Keyed"},
        "servers": [{"url": "https://k.example"}],
        "components": {
            "securitySchemes": {"main": scheme, "bearer": {"type": "http", "scheme": "bearer"}}
        },
        "security": security if security is not None else [{"main": []}],
        "paths": paths or {"/things": {"get": {"operationId": "listThings"}}},
    }


class TestCredentialSlot:
    def test_api_key_header_lands_on_the_plan(self):
        spec = _keyed_spec({"type": "apiKey", "in": "header", "name": "X-API-Key"})
        plan = build_plan(extract_operations(spec), auth=AuthMode.ENV_TOKEN)
        assert plan.api.credential_location == "header"
        assert plan.api.credential_name == "X-API-Key"
        assert "credential_name: X-API-Key" in plan.to_yaml()
        assert plan.tool_names == ["list_things"]

    def test_api_key_query_lands_on_the_plan(self):
        spec = _keyed_spec({"type": "apiKey", "in": "query", "name": "api_key"})
        plan = build_plan(extract_operations(spec), auth=AuthMode.ENV_TOKEN)
        assert (plan.api.credential_location, plan.api.credential_name) == ("query", "api_key")
        text = plan.to_yaml()
        assert "credential_location: query" in text and "credential_name: api_key" in text

    def test_swagger2_definitions(self):
        spec = {
            "swagger": "2.0",
            "host": "k.example",
            "securityDefinitions": {"key": {"type": "apiKey", "in": "header", "name": "api_key"}},
            "security": [{"key": []}],
            "paths": {"/pets": {"get": {"operationId": "listPets"}}},
        }
        plan = build_plan(extract_operations(spec), auth=AuthMode.ENV_TOKEN)
        assert (plan.api.credential_location, plan.api.credential_name) == ("header", "api_key")

    def test_bearer_oauth_or_nothing_means_the_authorization_header(self):
        bearer = _keyed_spec({"type": "http", "scheme": "bearer"})
        plan = build_plan(extract_operations(bearer))
        assert plan.api.credential_is_authorization_header
        assert "credential_" not in plan.to_yaml()
        unsecured = _keyed_spec({"type": "http", "scheme": "bearer"}, security=[])
        assert build_plan(extract_operations(unsecured)).api.credential_name == "Authorization"

    def test_passthrough_refuses_a_credential_it_cannot_relay(self):
        from promptise.mcpcast import mcpcast

        spec = _keyed_spec({"type": "apiKey", "in": "header", "name": "X-API-Key"})
        with pytest.raises(MCPcastError) as info:
            build_plan(extract_operations(spec))  # passthrough is the default
        message = str(info.value)
        assert "header 'X-API-Key'" in message and "--auth passthrough cannot relay" in message
        assert "--auth env-token" in message and "--auth api-key" in message
        with pytest.raises(MCPcastError, match="query parameter 'key'"):
            mcpcast(_keyed_spec({"type": "apiKey", "in": "query", "name": "key"}))
        # every other mode presents the credential where the spec says
        for auth in (AuthMode.ENV_TOKEN, AuthMode.API_KEY, AuthMode.NONE):
            assert (
                build_plan(extract_operations(spec), auth=auth).api.credential_name == "X-API-Key"
            )

    def test_majority_wins_and_the_minority_is_dropped_with_a_reason(self):
        paths = {
            "/a": {"get": {"operationId": "a"}},
            "/b": {"get": {"operationId": "b"}},
            "/c": {"get": {"operationId": "c", "security": [{"bearer": []}]}},
            "/d": {"get": {"operationId": "d", "security": []}},
        }
        spec = _keyed_spec({"type": "apiKey", "in": "header", "name": "X-Key"}, paths=paths)
        plan = build_plan(extract_operations(spec), auth=AuthMode.ENV_TOKEN)
        assert plan.api.credential_name == "X-Key"
        assert plan.tool_names == ["a", "b", "d"]
        (dropped,) = plan.dropped
        assert dropped.operation_id == "c"
        assert dropped.reason == (
            "unsupported by mcpcast: requires HTTP bearer authentication, but this server "
            "presents its credential in header 'X-Key'"
        )

    def test_cookie_key_cannot_be_presented(self):
        spec = _keyed_spec({"type": "apiKey", "in": "cookie", "name": "session"})
        plan = build_plan(extract_operations(spec), auth=AuthMode.ENV_TOKEN)
        assert plan.api.credential_name == "Authorization"  # nothing presentable was declared
        (dropped,) = plan.dropped
        assert dropped.reason == (
            "unsupported by mcpcast: requires an API key in cookie 'session', which the "
            "generated server cannot present"
        )

    def test_credential_slot_helper(self):
        from promptise.mcpcast.plan import AUTHORIZATION_HEADER, CredentialSlot, credential_slot

        assert credential_slot([]) == AUTHORIZATION_HEADER
        assert credential_slot(OPS) == AUTHORIZATION_HEADER
        assert CredentialSlot("header", "authorization").is_authorization_header
        assert CredentialSlot("header", "X-Key").matches(CredentialSlot("header", "x-key"))
        assert not CredentialSlot("query", "key").matches(CredentialSlot("query", "Key"))
        assert CredentialSlot("query", "api_key").describe() == "query parameter 'api_key'"
        spec = _keyed_spec({"type": "apiKey", "in": "header", "name": "x-api-key"})
        ops = extract_operations(spec)
        assert credential_slot(ops) == CredentialSlot("header", "x-api-key")


# ---------------------------------------------------------------------------
# Examples fit their declared type
# ---------------------------------------------------------------------------


class TestExamplesFitTheirType:
    @pytest.mark.parametrize(
        ("schema", "expected"),
        [
            ({"type": "string", "example": 2019}, "2019"),
            ({"type": "string", "example": 1.5}, "1.5"),
            ({"type": "string", "example": True}, "true"),
            ({"type": "string", "default": False}, "false"),
            ({"type": "integer", "example": "42"}, 42),
            ({"type": "integer", "example": " -7 "}, -7),
            ({"type": "integer", "example": 2.0}, 2),
            ({"type": "number", "example": "2.5"}, 2.5),
            ({"type": "number", "example": "3"}, 3.0),
            ({"type": "number", "examples": ["1e3"]}, 1000.0),
            ({"type": ["null", "integer"], "example": "5"}, 5),
        ],
    )
    def test_obvious_slips_are_coerced(self, schema, expected):
        value = example_value(schema, "x")
        assert value == expected and type(value) is type(expected)

    @pytest.mark.parametrize(
        ("schema", "expected"),
        [
            ({"type": "number", "example": float("inf")}, 1.0),
            ({"type": "number", "example": "Infinity"}, 1.0),
            ({"type": "number", "default": float("nan")}, 1.0),
            ({"type": "number", "example": "nan"}, 1.0),
            ({"type": "integer", "example": "abc"}, 1),
            ({"type": "integer", "example": "2.5"}, 1),
            ({"type": "integer", "example": True}, 1),
            ({"type": "boolean", "example": "yes"}, True),
            ({"type": "string", "example": {"a": 1}}, "string"),
            ({"type": "string", "example": ["a"]}, "string"),
            ({"type": "array", "items": {"type": "integer"}, "example": "1,2"}, [1]),
            (
                {"type": "object", "properties": {"n": {"type": "string"}}, "example": "x"},
                {"n": "string"},
            ),
            ({"type": "string", "enum": ["a", "b"], "example": 1}, "a"),
            ({"type": "string", "enum": ["a", "b"], "example": "zzz"}, "a"),
        ],
    )
    def test_the_rest_is_dropped_for_a_synthesised_example(self, schema, expected):
        assert example_value(schema, "x") == expected

    def test_untyped_and_fitting_examples_are_untouched(self):
        assert example_value({"example": 5}, "x") == 5
        assert example_value({"example": "5"}, "x") == "5"
        assert example_value({"type": "integer", "example": 5}, "x") == 5
        assert example_value({"type": "string", "example": "2019"}, "x") == "2019"
        assert example_value({"type": "number", "example": 2}, "x") == 2
        assert example_value(
            {"type": "array", "items": {"type": "integer"}, "example": [3]}, "x"
        ) == [3]

    def test_example_mismatch_lives_in_plan_and_is_re_exported(self):
        from promptise.mcpcast import example_mismatch as exported
        from promptise.mcpcast.curate import example_mismatch as from_curate
        from promptise.mcpcast.plan import example_mismatch

        assert exported is example_mismatch and from_curate is example_mismatch
        assert (
            example_mismatch(True, {"type": "integer"})
            == "is a boolean but the API declares integer"
        )
        assert example_mismatch(2019, {"type": "string"}) == "is a int but the API declares string"
        assert example_mismatch("2019", {"type": "string"}) is None

    def test_plan_and_tool_examples_match_their_signatures(self):
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "Gauge"},
            "servers": [{"url": "https://g.example"}],
            "paths": {
                "/gauge": {
                    "post": {
                        "operationId": "setGauge",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "required": ["value", "year", "count"],
                                        "properties": {
                                            "value": {"type": "number", "example": float("inf")},
                                            "year": {"type": "string", "example": 2019},
                                            "count": {"type": "integer", "example": "abc"},
                                            "ratio": {"type": "number", "default": float("nan")},
                                        },
                                    }
                                }
                            }
                        },
                    }
                }
            },
        }
        plan = mcpcast(spec, profile=SafetyProfile.STANDARD)
        tool = plan.tool("set_gauge")
        assert tool.example == {"value": 1.0, "year": "2019", "count": 1}
        assert tool.params["ratio"].json_schema["default"] == "NaN"  # JSON-native, and unfit
        assert example_value(tool.params["ratio"].json_schema, "ratio") == 1.0

    def test_non_mapping_property_schemas_have_no_example(self):
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": "not a schema", "c": None, "d": 5},
        }
        assert example_value(schema, "x") == {"a": "string", "c": "string"}
        assert example_value({"type": "object", "properties": "nope"}, "x") == {}
        assert example_value({"type": "array", "items": "nope"}, "x") == []
        assert example_value({"anyOf": ["nope"], "type": "integer"}, "x") == 1
