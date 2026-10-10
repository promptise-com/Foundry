"""MCPcast schema fidelity: what an OpenAPI spec declares about a parameter reaches the
generated tool's input schema — enums, patterns, bounds, lengths, formats, nullability,
unions, nested objects, required lists and descriptions — and what cannot be carried in
full is reported.

Each project is generated from a small spec, imported, and driven through
``TestClient`` against a recording ``httpx.MockTransport`` upstream (no sockets).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from promptise.mcp.server import TestClient
from promptise.mcpcast import load_generated_server, mcpcast, write_project
from promptise.mcpcast.emit import _clean_schema, render_project, trimmed_schemas
from promptise.mcpcast.readiness import score
from promptise.mcpcast.schema import SafetyProfile
from promptise.mcpcast.wizard import review_warnings

SPEC: dict[str, Any] = {
    "openapi": "3.0.3",
    "info": {"title": "Shop", "version": "1"},
    "servers": [{"url": "https://shop.example"}],
    "paths": {
        "/items": {
            "get": {
                "operationId": "listItems",
                "summary": "List items",
                "parameters": [
                    {
                        "name": "status",
                        "in": "query",
                        "description": "Filter by status",
                        "schema": {"type": "string", "enum": ["open", "closed"], "default": "open"},
                    },
                    {
                        "name": "limit",
                        "in": "query",
                        "schema": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 100,
                            "exclusiveMaximum": True,
                        },
                    },
                    {
                        "name": "sku",
                        "in": "query",
                        "schema": {
                            "type": "string",
                            "pattern": "^[A-Z]{3}-\\d+$",
                            "minLength": 5,
                            "maxLength": 12,
                        },
                    },
                    {
                        "name": "since",
                        "in": "query",
                        "schema": {"type": "string", "format": "date-time", "nullable": True},
                    },
                    {
                        "name": "tags",
                        "in": "query",
                        "schema": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["a", "b"]},
                            "maxItems": 3,
                            "uniqueItems": True,
                        },
                    },
                    {
                        "name": "region",
                        "in": "query",
                        "example": "eu-west",
                        "schema": {"type": "string"},
                    },
                    {
                        "name": "filter",
                        "in": "query",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"color": {"type": "string"}},
                                }
                            }
                        },
                    },
                ],
            },
            "post": {
                "operationId": "createItem",
                "summary": "Create an item",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/Item"}}
                    },
                },
            },
        }
    },
    "components": {
        "schemas": {
            "Kind": {"type": "string", "enum": ["x", "y"], "description": "Kind of item"},
            "Item": {
                "type": "object",
                "required": ["id", "name", "price", "parent"],
                "properties": {
                    "id": {"type": "integer", "readOnly": True},
                    "name": {"type": "string", "minLength": 1, "description": "Display name"},
                    "price": {
                        "type": "number",
                        "minimum": 0,
                        "exclusiveMinimum": True,
                        "multipleOf": 0.01,
                    },
                    "parent": {"type": "integer", "nullable": True, "description": "Parent id"},
                    "dims": {
                        "type": "object",
                        "required": ["w"],
                        "properties": {
                            "w": {"type": "integer", "description": "Width in mm"},
                            "h": {"type": "integer"},
                            "measured": {"type": "string", "readOnly": True},
                        },
                    },
                    "pay": {
                        "oneOf": [
                            {"type": "object", "properties": {"card": {"type": "string"}}},
                            {"type": "object", "properties": {"iban": {"type": "string"}}},
                        ]
                    },
                    "note": {"anyOf": [{"type": "string"}, {"type": "integer"}], "nullable": True},
                    "kind": {"allOf": [{"$ref": "#/components/schemas/Kind"}], "nullable": True},
                },
            },
        }
    },
}


class _Upstream:
    """Records every request and answers ``{"ok": true}``."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    async def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(200, json={"ok": True}, request=request)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def _plan(spec: dict[str, Any] = SPEC):
    return mcpcast(spec, name="shop", profile=SafetyProfile.STANDARD, auth="none")  # type: ignore[arg-type]


def _generate(tmp_path: Path, spec: dict[str, Any] = SPEC):
    plan = _plan(spec)
    out = tmp_path / "proj"
    write_project(plan, out)
    return plan, load_generated_server(out / "server.py")


async def _schemas(module, http: httpx.AsyncClient) -> dict[str, dict[str, Any]]:
    client = TestClient(
        module.build_server(approval_handler=lambda request: True, http_client=http)
    )
    return {t.name: t.inputSchema for t in await client.list_tools()}


class TestAdvertisedSchema:
    """``tools/list`` carries the spec's schema, not the bare Python type."""

    @pytest.mark.asyncio
    async def test_query_parameters_keep_their_constraints(self, tmp_path: Path) -> None:
        _, module = _generate(tmp_path)
        async with _Upstream().client() as http:
            schema = (await _schemas(module, http))["list_items"]
        props = schema["properties"]
        assert props["status"]["enum"] == ["open", "closed"]
        assert props["status"]["default"] == "open"
        assert props["status"]["description"] == "Filter by status"
        # OpenAPI 3.0's boolean exclusiveMaximum becomes JSON Schema's number.
        assert props["limit"]["minimum"] == 1
        assert props["limit"]["exclusiveMaximum"] == 100
        assert "maximum" not in props["limit"]
        assert props["sku"]["pattern"] == "^[A-Z]{3}-\\d+$"
        assert (props["sku"]["minLength"], props["sku"]["maxLength"]) == (5, 12)
        assert props["since"]["format"] == "date-time"
        assert props["since"]["type"] == ["string", "null"]  # nullable: true
        assert props["tags"]["items"] == {"type": "string", "enum": ["a", "b"]}
        assert props["tags"]["maxItems"] == 3 and props["tags"]["uniqueItems"] is True
        # A parameter declared through ``content`` keeps its schema (was a bare string).
        assert props["filter"]["type"] == "object"
        assert props["filter"]["properties"] == {"color": {"type": "string"}}

    @pytest.mark.asyncio
    async def test_body_properties_keep_nesting_unions_and_nullability(
        self, tmp_path: Path
    ) -> None:
        _, module = _generate(tmp_path)
        async with _Upstream().client() as http:
            schema = (await _schemas(module, http))["create_item"]
        props = schema["properties"]
        assert sorted(schema["required"]) == ["name", "parent", "price"]
        assert props["name"]["minLength"] == 1
        assert props["price"]["exclusiveMinimum"] == 0 and props["price"]["multipleOf"] == 0.01
        assert props["parent"]["type"] == ["integer", "null"]
        assert props["dims"]["required"] == ["w"]
        assert props["dims"]["properties"]["w"]["description"] == "Width in mm"
        assert [m["properties"] for m in props["pay"]["oneOf"]] == [
            {"card": {"type": "string"}},
            {"iban": {"type": "string"}},
        ]
        assert props["note"]["anyOf"] == [
            {"type": "string"},
            {"type": "integer"},
            {"type": "null"},
        ]
        # OpenAPI 3.0's nullable $ref (allOf + nullable) admits null as JSON Schema says it.
        assert props["kind"]["anyOf"][1] == {"type": "null"}
        assert props["kind"]["anyOf"][0]["allOf"][0]["enum"] == ["x", "y"]

    @pytest.mark.asyncio
    async def test_read_only_properties_are_not_inputs(self, tmp_path: Path) -> None:
        plan, module = _generate(tmp_path)
        assert "id" not in plan.tool("create_item").params
        async with _Upstream().client() as http:
            schema = (await _schemas(module, http))["create_item"]
        assert "id" not in schema["properties"]
        assert "id" not in schema["required"]
        assert "measured" not in schema["properties"]["dims"]["properties"]

    def test_a_parameter_level_example_feeds_the_tool_example(self) -> None:
        plan = _plan()
        assert plan.tool("list_items").params["region"].json_schema["example"] == "eu-west"

    def test_every_generated_module_carries_the_schemas(self) -> None:
        files = render_project(_plan())
        items = next(src for path, src in files.items() if path.endswith("tools/items.py"))
        assert "PARAM_SCHEMAS" in items and '@advertise(PARAM_SCHEMAS["create_item"])' in items


class TestCallingTheGeneratedTools:
    @pytest.mark.asyncio
    async def test_a_constrained_call_reaches_the_api_as_sent(self, tmp_path: Path) -> None:
        _, module = _generate(tmp_path)
        upstream = _Upstream()
        async with upstream.client() as http:
            server = module.build_server(approval_handler=lambda request: True, http_client=http)
            client = TestClient(server)
            (r,) = await client.call_tool(
                "list_items", {"status": "closed", "limit": 5, "sku": "ABC-12", "tags": ["a"]}
            )
            assert json.loads(r.text) == {"ok": True}
            (r,) = await client.call_tool(
                "create_item",
                {"name": "lamp", "price": 9.99, "parent": 7, "dims": {"w": 3}, "kind": "x"},
            )
            assert json.loads(r.text) == {"ok": True}
        get, post = upstream.requests
        assert dict(get.url.params) == {
            "status": "closed",
            "limit": "5",
            "sku": "ABC-12",
            "tags": "a",
        }
        assert json.loads(post.content) == {
            "name": "lamp",
            "price": 9.99,
            "parent": 7,
            "dims": {"w": 3},
            "kind": "x",
        }

    @pytest.mark.asyncio
    async def test_a_required_nullable_body_property_sends_null(self, tmp_path: Path) -> None:
        """``parent`` is required and nullable: an explicit null is a value the API
        must receive (it was rejected before — the handler took ``int`` only)."""
        _, module = _generate(tmp_path)
        upstream = _Upstream()
        async with upstream.client() as http:
            server = module.build_server(approval_handler=lambda request: True, http_client=http)
            (r,) = await TestClient(server).call_tool(
                "create_item", {"name": "lamp", "price": 1.5, "parent": None}
            )
        assert json.loads(r.text) == {"ok": True}
        (post,) = upstream.requests
        assert json.loads(post.content) == {"name": "lamp", "price": 1.5, "parent": None}

    @pytest.mark.asyncio
    async def test_an_omitted_optional_property_is_not_sent(self, tmp_path: Path) -> None:
        _, module = _generate(tmp_path)
        upstream = _Upstream()
        async with upstream.client() as http:
            server = module.build_server(approval_handler=lambda request: True, http_client=http)
            await TestClient(server).call_tool(
                "create_item", {"name": "lamp", "price": 1.5, "parent": 2, "note": None}
            )
        (post,) = upstream.requests
        assert "note" not in json.loads(post.content)


class TestSwagger2Parameters:
    SPEC: dict[str, Any] = {
        "swagger": "2.0",
        "info": {"title": "Legacy", "version": "1"},
        "host": "legacy.example",
        "schemes": ["https"],
        "paths": {
            "/search": {
                "get": {
                    "operationId": "search",
                    "parameters": [
                        {
                            "name": "q",
                            "in": "query",
                            "type": "string",
                            "minLength": 2,
                            "maxLength": 50,
                            "x-nullable": True,
                        },
                        {
                            "name": "page",
                            "in": "query",
                            "type": "integer",
                            "minimum": 0,
                            "exclusiveMinimum": True,
                            "multipleOf": 1,
                        },
                        {
                            "name": "ids",
                            "in": "query",
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 1,
                            "maxItems": 10,
                            "uniqueItems": True,
                        },
                    ],
                    "responses": {"200": {"description": "ok"}},
                }
            }
        },
    }

    @pytest.mark.asyncio
    async def test_inline_parameter_constraints_survive(self, tmp_path: Path) -> None:
        _, module = _generate(tmp_path, self.SPEC)
        async with _Upstream().client() as http:
            props = (await _schemas(module, http))["search"]["properties"]
        assert (props["q"]["minLength"], props["q"]["maxLength"]) == (2, 50)
        assert props["q"]["type"] == ["string", "null"]  # x-nullable
        assert props["page"]["exclusiveMinimum"] == 0 and props["page"]["multipleOf"] == 1
        assert (props["ids"]["minItems"], props["ids"]["maxItems"]) == (1, 10)
        assert props["ids"]["uniqueItems"] is True


class TestCleanSchema:
    """OpenAPI dialect → JSON Schema, one rule at a time."""

    def test_nullable_enum_lists_null(self) -> None:
        cleaned = _clean_schema({"type": "string", "enum": ["a"], "nullable": True})
        assert cleaned == {"type": ["string", "null"], "enum": ["a", None]}

    def test_nullable_type_list_and_union(self) -> None:
        assert _clean_schema({"type": ["string", "integer"], "nullable": True}) == {
            "type": ["string", "integer", "null"]
        }
        assert _clean_schema({"oneOf": [{"type": "string"}], "nullable": True}) == {
            "oneOf": [{"type": "string"}, {"type": "null"}]
        }

    def test_exclusive_bound_false_keeps_the_inclusive_bound(self) -> None:
        assert _clean_schema({"type": "integer", "minimum": 1, "exclusiveMinimum": False}) == {
            "type": "integer",
            "minimum": 1,
        }

    def test_openapi_only_keywords_and_extensions_are_dropped(self) -> None:
        cleaned = _clean_schema(
            {
                "type": "object",
                "x-internal": True,
                "discriminator": {"propertyName": "kind"},
                "xml": {"name": "item"},
                "example": {"kind": "a"},
                "properties": {"kind": {"type": "string"}},
            }
        )
        assert cleaned == {
            "type": "object",
            "examples": [{"kind": "a"}],
            "properties": {"kind": {"type": "string"}},
        }


def _huge_enum_spec(entries: int) -> dict[str, Any]:
    return {
        "openapi": "3.0.3",
        "info": {"title": "Codes", "version": "1"},
        "servers": [{"url": "https://codes.example"}],
        "paths": {
            "/codes": {
                "get": {
                    "operationId": "lookupCode",
                    "parameters": [
                        {
                            "name": "code",
                            "in": "query",
                            "schema": {
                                "type": "string",
                                "enum": [f"CODE-{i:05d}" for i in range(entries)],
                            },
                        }
                    ],
                }
            }
        },
    }


class TestWhatCannotBeAdvertisedIsReported:
    def test_a_schema_too_large_to_list_is_trimmed_and_named(self) -> None:
        plan = _plan(_huge_enum_spec(2000))
        assert trimmed_schemas(plan) == ["lookup_code"]
        fixes = "\n".join(score(plan, []).fixes)
        assert "`lookup_code` advertises a trimmed input schema" in fixes
        assert any(
            line.startswith("lookup_code: input schema trimmed") for line in review_warnings(plan)
        )

    @pytest.mark.asyncio
    async def test_the_trimmed_schema_keeps_its_type(self, tmp_path: Path) -> None:
        _, module = _generate(tmp_path, _huge_enum_spec(2000))
        async with _Upstream().client() as http:
            code = (await _schemas(module, http))["lookup_code"]["properties"]["code"]
        assert code["type"] == "string" and "enum" not in code

    def test_a_schema_that_fits_is_not_reported(self) -> None:
        plan = _plan(_huge_enum_spec(20))
        assert trimmed_schemas(plan) == []
        assert not any("trimmed" in fix for fix in score(plan, []).fixes)
        assert not any("trimmed" in line for line in review_warnings(plan))


class TestAnAgentSeesTheSchema:
    """The Promptise agent side turns each listed input schema into a typed model
    (``promptise.tools``); a nullable parameter's ``["string", "null"]`` type used to
    crash that conversion, so no agent — nor the readiness evaluation — could use
    a generated server whose spec declares ``nullable``."""

    @pytest.mark.asyncio
    async def test_the_agent_tools_keep_the_constraints(self, tmp_path: Path) -> None:
        from promptise.mcpcast.readiness import tools_from_server

        _, module = _generate(tmp_path)
        upstream = _Upstream()
        async with upstream.client() as http:
            server = module.build_server(approval_handler=lambda request: True, http_client=http)
            tools = {t.name: t for t in await tools_from_server(server)}
            listed = tools["list_items"].args_schema.model_json_schema()["properties"]  # type: ignore[union-attr]
            assert listed["status"]["enum"] == ["open", "closed"]
            assert listed["sku"]["pattern"] == "^[A-Z]{3}-\\d+$"
            assert {"type": "string", "format": "date-time"} in listed["since"]["anyOf"]
            assert listed["tags"]["items"]["enum"] == ["a", "b"]
            created = tools["create_item"].args_schema.model_json_schema()  # type: ignore[union-attr]
            (dims,) = [d for d in created["$defs"].values() if "w" in d.get("properties", {})]
            assert dims["required"] == ["w"]
            assert dims["properties"]["w"]["description"] == "Width in mm"

            await tools["create_item"].ainvoke({"name": "lamp", "price": 2.5, "parent": None})
        (post,) = upstream.requests
        assert json.loads(post.content) == {"name": "lamp", "price": 2.5, "parent": None}
