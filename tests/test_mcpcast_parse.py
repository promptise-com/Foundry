"""Tests for OpenAPI parsing in ``promptise.mcpcast.parse``."""

from __future__ import annotations

import json
import time
from pathlib import Path

import httpx
import pytest

from promptise.mcpcast.parse import (
    api_name_from_spec,
    extract_operations,
    load_spec,
    spec_base_url,
)
from promptise.mcpcast.schema import MCPcastError

PET = {
    "type": "object",
    "required": ["name"],
    "properties": {
        "id": {"type": "integer"},
        "name": {"type": "string", "description": "Pet name"},
        "owner": {"$ref": "#/components/schemas/Owner"},
    },
}
OWNER = {"type": "object", "properties": {"email": {"type": "string", "format": "email"}}}

SPEC = {
    "openapi": "3.0.3",
    "info": {"title": "Petstore API", "description": "Pets.\nMore."},
    "servers": [{"url": "https://{env}.example.com/v3", "variables": {"env": {"default": "api"}}}],
    "security": [{"oauth": ["read:pets"]}],
    "components": {
        "schemas": {"Pet": PET, "Owner": OWNER},
        "parameters": {"Trace": {"name": "X-Trace", "in": "header", "schema": {"type": "string"}}},
    },
    "paths": {
        "/pet/{petId}": {
            "parameters": [
                {"name": "petId", "in": "path", "schema": {"type": "integer"}, "description": "ID"},
                {"$ref": "#/components/parameters/Trace"},
            ],
            "get": {
                "operationId": "getPetById",
                "summary": "Find pet by ID",
                "tags": ["pet"],
                "parameters": [
                    {"name": "petId", "in": "path", "required": True, "schema": {"type": "string"}},
                    {"name": "verbose", "in": "query", "schema": {"type": "boolean"}},
                ],
                "responses": {
                    "404": {"description": "nope"},
                    "200": {
                        "content": {
                            "application/json": {"schema": {"$ref": "#/components/schemas/Pet"}}
                        }
                    },
                },
            },
            "delete": {
                "operationId": "delete-pet",
                "security": [{"oauth": ["write:pets", "admin"]}],
                "deprecated": True,
            },
        },
        "/pet": {
            "post": {
                "operationId": "addPet",
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {"schema": {"$ref": "#/components/schemas/Pet"}}
                    },
                },
            },
            "summary": "ignored path-level key",
        },
        "/import": {
            "post": {
                "operationId": "importPets",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "array",
                                "items": {"$ref": "#/components/schemas/Pet"},
                            }
                        }
                    }
                },
            }
        },
        "/form": {
            "post": {
                "operationId": "postForm",
                "requestBody": {
                    "content": {
                        "application/x-www-form-urlencoded": {
                            "schema": {"type": "object", "properties": {"a": {"type": "string"}}}
                        }
                    }
                },
            }
        },
        "/no-id/{x}": {"get": {"parameters": [{"name": "x", "in": "path", "required": True}]}},
        "/no-id/{x}/": {"get": {}},
    },
}


_REAL_CLIENT = httpx.Client


def _by_id(ops):
    return {o.operation_id: o for o in ops}


def _mock_http(monkeypatch, handler) -> None:
    """Route every ``httpx.Client`` the parser opens through *handler*."""

    def client(**kwargs):
        return _REAL_CLIENT(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)


class TestLoadSpec:
    def test_dict_passthrough(self):
        assert load_spec(SPEC) == SPEC

    def test_json_and_yaml_files(self, tmp_path):
        j = tmp_path / "s.json"
        j.write_text(json.dumps(SPEC), encoding="utf-8")
        y = tmp_path / "s.yaml"
        y.write_text("openapi: 3.0.0\ninfo: {title: Y}\npaths: {}\n", encoding="utf-8")
        assert load_spec(str(j)) == SPEC
        assert load_spec(y)["info"]["title"] == "Y"

    def test_inline_text(self):
        assert load_spec(json.dumps(SPEC))["openapi"] == "3.0.3"
        assert load_spec("openapi: 3.1.0\npaths: {}\n")["openapi"] == "3.1.0"

    def test_missing_and_invalid(self, tmp_path):
        with pytest.raises(MCPcastError, match="spec not found"):
            load_spec("does/not/exist.yaml")
        bad = tmp_path / "bad.yaml"
        bad.write_text("- just\n- a list\n", encoding="utf-8")
        with pytest.raises(MCPcastError, match="not a mapping"):
            load_spec(str(bad))
        broken = tmp_path / "broken.json"
        broken.write_text("{not json", encoding="utf-8")
        with pytest.raises(MCPcastError, match="not valid JSON or YAML"):
            load_spec(str(broken))

    def test_url_fetch_failure_is_clean(self, monkeypatch):
        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        _mock_http(monkeypatch, down)
        with pytest.raises(MCPcastError, match="could not fetch.*ConnectError"):
            load_spec("https://example.invalid/openapi.json")

    def test_url_fetch_is_streamed_and_capped(self, monkeypatch):
        """A redirect can land anywhere: the body is never buffered past the cap."""
        body = json.dumps(SPEC).encode()
        monkeypatch.setenv("MCPCAST_MAX_SPEC_BYTES", str(len(body)))
        _mock_http(monkeypatch, lambda r: httpx.Response(200, content=body))
        assert load_spec("https://h.example/openapi.json") == SPEC

        monkeypatch.setenv("MCPCAST_MAX_SPEC_BYTES", str(len(body) - 1))
        with pytest.raises(MCPcastError, match="larger than .* MCPCAST_MAX_SPEC_BYTES"):
            load_spec("https://h.example/openapi.json")

        # a Content-Length over the cap is refused before a single byte is read
        monkeypatch.setenv("MCPCAST_MAX_SPEC_BYTES", "10")
        served: list[str] = []

        def big(request: httpx.Request) -> httpx.Response:
            served.append(str(request.url))
            return httpx.Response(200, headers={"content-length": "999"}, content=b"{}")

        _mock_http(monkeypatch, big)
        with pytest.raises(MCPcastError, match="larger than 10 bytes"):
            load_spec("https://h.example/openapi.json")
        monkeypatch.setenv("MCPCAST_MAX_SPEC_BYTES", "zero")
        with pytest.raises(MCPcastError, match="must be an integer"):
            load_spec("https://h.example/openapi.json")

    def test_local_documents_are_capped_too(self, tmp_path, monkeypatch):
        big = tmp_path / "big.json"
        big.write_text(json.dumps(SPEC), encoding="utf-8")
        monkeypatch.setenv("MCPCAST_MAX_SPEC_BYTES", "100")
        with pytest.raises(MCPcastError, match="larger than MCPCAST_MAX_SPEC_BYTES"):
            load_spec(str(big))
        with pytest.raises(MCPcastError, match="larger than MCPCAST_MAX_SPEC_BYTES"):
            load_spec(json.dumps(SPEC))
        monkeypatch.delenv("MCPCAST_MAX_SPEC_BYTES")
        assert load_spec(str(big)) == SPEC

    def test_lone_surrogates_are_scrubbed(self):
        """``json.loads`` accepts ``"\\ud800"``; nothing could write it as UTF-8 later."""
        text = '{"openapi": "3.0.0", "info": {"title": "A\\ud800B"}, "paths": {"/\\udc00": {}}}'
        spec = load_spec(text)
        spec["info"]["title"].encode("utf-8")  # would raise on a lone surrogate
        assert spec["info"]["title"] == "A?B"
        assert list(spec["paths"]) == ["/?"]


class TestSpecMetadata:
    def test_base_url_variables_and_override(self):
        assert spec_base_url(SPEC) == "https://api.example.com/v3"
        assert spec_base_url(SPEC, "http://localhost:8000/") == "http://localhost:8000"

    def test_swagger2_host(self):
        assert (
            spec_base_url({"host": "x.io", "basePath": "/v1", "schemes": ["http"]})
            == "http://x.io/v1"
        )
        assert spec_base_url({"paths": {}}) == ""

    @pytest.mark.parametrize(
        ("title", "expected"),
        [
            ("Petstore API", "petstore"),
            ("Swagger Petstore - OpenAPI 3.0", "petstore"),
            ("My Cool  API", "my-cool"),
            ("GitHub v3 REST API", "github"),
            ("Stripe API", "stripe"),
            ("API", "api"),
            ("v2", "v2"),
        ],
    )
    def test_api_name(self, title, expected):
        assert api_name_from_spec({"info": {"title": title}}) == expected

    def test_api_name_fallbacks(self):
        assert api_name_from_spec({}, "/tmp/billing-openapi.yaml") == "billing"
        assert api_name_from_spec({}) == "api"


class TestExtractOperations:
    def test_requires_paths(self):
        with pytest.raises(MCPcastError, match="no 'paths'"):
            extract_operations({"openapi": "3.0.0"})

    def test_ids_sanitised_and_unique(self):
        ids = [o.operation_id for o in extract_operations(SPEC)]
        assert "delete_pet" in ids
        assert "get_no_id_x" in ids and "get_no_id_x_2" in ids
        assert len(ids) == len(set(ids))

    def test_operation_params_override_shared_and_headers_kept(self):
        get = _by_id(extract_operations(SPEC))["getPetById"]
        pet_id = get.param("petId")
        assert pet_id is not None and pet_id.json_schema == {"type": "string"}  # op-level wins
        assert pet_id.required and pet_id.location == "path"
        assert get.param("verbose").location == "query"
        trace = get.param("X-Trace")
        assert trace is not None and trace.location == "header"  # $ref param resolved

    def test_path_params_always_required(self):
        ops = _by_id(extract_operations(SPEC))
        x = ops["get_no_id_x"].param("x")
        assert x.required and x.json_schema == {"type": "string"}

    def test_body_properties_become_params_with_nested_refs_inlined(self):
        add = _by_id(extract_operations(SPEC))["addPet"]
        names = {p.name: p for p in add.params}
        assert set(names) == {"id", "name", "owner"}
        assert names["name"].required and names["name"].location == "body"
        assert names["owner"].json_schema["properties"]["email"]["format"] == "email"
        assert add.body_encoding == "json"

    def test_non_object_body_is_raw_body(self):
        imp = _by_id(extract_operations(SPEC))["importPets"]
        (body,) = imp.params
        assert body.name == "body" and body.location == "raw_body" and not body.required
        assert body.json_schema["items"]["properties"]["name"]["type"] == "string"

    def test_form_encoding(self):
        form = _by_id(extract_operations(SPEC))["postForm"]
        assert form.body_encoding == "form"
        assert form.param("a").location == "body"

    def test_scopes_global_vs_operation_and_deprecated(self):
        ops = _by_id(extract_operations(SPEC))
        assert ops["getPetById"].scopes == ["read:pets"]  # inherited
        assert ops["delete_pet"].scopes == ["admin", "write:pets"]
        assert ops["delete_pet"].deprecated is True
        assert ops["getPetById"].deprecated is False

    def test_success_response_schema_prefers_2xx(self):
        get = _by_id(extract_operations(SPEC))["getPetById"]
        assert get.response_schema["properties"]["owner"]["properties"]["email"]["type"] == "string"
        assert _by_id(extract_operations(SPEC))["addPet"].response_schema is None

    def test_base_url_and_metadata(self):
        ops = extract_operations(SPEC, base_url="http://localhost:9000")
        assert all(o.base_url == "http://localhost:9000" for o in ops)
        get = _by_id(ops)["getPetById"]
        assert get.tags == ["pet"] and get.summary == "Find pet by ID"
        assert "getPetById" in get.text and "/pet/{petId}" in get.text

    def test_swagger2_body_and_form_data(self):
        spec = {
            "swagger": "2.0",
            "host": "x.io",
            "paths": {
                "/a": {
                    "post": {
                        "operationId": "a",
                        "parameters": [
                            {
                                "name": "body",
                                "in": "body",
                                "required": True,
                                "schema": {
                                    "type": "object",
                                    "properties": {"k": {"type": "string"}},
                                    "required": ["k"],
                                },
                            },
                        ],
                        "responses": {"200": {"schema": {"type": "object"}}},
                    }
                },
                "/b": {
                    "post": {
                        "operationId": "b",
                        "parameters": [
                            {"name": "f", "in": "formData", "type": "integer", "enum": [1, 2]}
                        ],
                    }
                },
            },
        }
        ops = _by_id(extract_operations(spec))
        assert ops["a"].param("k").required and ops["a"].param("k").location == "body"
        assert ops["a"].response_schema == {"type": "object"}
        assert ops["b"].body_encoding == "form"
        assert ops["b"].param("f").json_schema == {"type": "integer", "enum": [1, 2]}

    def test_cyclic_refs_do_not_recurse_forever(self):
        spec = {
            "components": {
                "schemas": {
                    "Node": {
                        "type": "object",
                        "properties": {"next": {"$ref": "#/components/schemas/Node"}},
                    }
                }
            },
            "paths": {
                "/n": {
                    "post": {
                        "operationId": "n",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Node"}
                                }
                            }
                        },
                    }
                }
            },
        }
        (op,) = extract_operations(spec)
        assert op.param("next") is not None  # resolved to a bounded depth


class TestUnresolvedRefs:
    """An external or dangling ``$ref`` never silently becomes ``{}``."""

    def _spec(self, **paths):
        return {
            "openapi": "3.0.0",
            "servers": [{"url": "https://x"}],
            "components": {"schemas": {"Order": {"type": "object"}}},
            "paths": paths,
        }

    def test_external_body_ref_makes_the_operation_unsupported(self):
        spec = self._spec(
            **{
                "/orders": {
                    "post": {
                        "operationId": "createOrder",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "schemas/order.yaml#/Order"}
                                }
                            }
                        },
                    }
                }
            }
        )
        (op,) = extract_operations(spec)
        assert op.params == []
        assert op.unsupported_body == (
            "external $ref 'schemas/order.yaml#/Order' is not supported; bundle the spec "
            "into one document"
        )
        from promptise.mcpcast.plan import build_plan
        from promptise.mcpcast.schema import SafetyProfile

        plan = build_plan([op], profile=SafetyProfile.STANDARD)
        assert plan.tools == []
        (dropped,) = plan.dropped
        assert dropped.operation_id == "createOrder"
        assert "external $ref 'schemas/order.yaml#/Order' is not supported" in dropped.reason

    def test_parameter_refs_external_dangling_and_shared(self):
        spec = self._spec(
            **{
                "/a/{id}": {
                    "get": {
                        "operationId": "a",
                        "parameters": [{"$ref": "params.yaml#/OrderId"}],
                    }
                },
                "/b": {
                    "get": {
                        "operationId": "b",
                        "parameters": [{"$ref": "#/components/parameters/Missing"}],
                    }
                },
                "/c": {
                    "parameters": [{"$ref": "shared.yaml#/Tenant"}],
                    "get": {"operationId": "c1"},
                    "delete": {"operationId": "c2"},
                },
                "/d": {"get": {"operationId": "d"}},
            }
        )
        ops = _by_id(extract_operations(spec))
        assert "external $ref 'params.yaml#/OrderId'" in ops["a"].unsupported_body
        assert "dangling $ref '#/components/parameters/Missing'" in ops["b"].unsupported_body
        assert "external $ref 'shared.yaml#/Tenant'" in ops["c1"].unsupported_body
        assert ops["c2"].unsupported_body == ops["c1"].unsupported_body
        assert ops["d"].unsupported_body is None

    def test_response_ref_only_loses_the_mock_shape(self):
        spec = self._spec(
            **{
                "/e": {
                    "get": {
                        "operationId": "e",
                        "responses": {
                            "200": {
                                "content": {
                                    "application/json": {"schema": {"$ref": "models.yaml#/Thing"}}
                                }
                            }
                        },
                    }
                }
            }
        )
        (op,) = extract_operations(spec)
        assert op.unsupported_body is None and op.response_schema is None

    def test_external_path_item_ref_is_an_error(self):
        spec = self._spec(**{"/f": {"$ref": "paths/f.yaml"}})
        with pytest.raises(MCPcastError, match=r"path item '/f'.*external \$ref 'paths/f.yaml'"):
            extract_operations(spec)

    def test_local_refs_still_resolve(self):
        spec = self._spec(
            **{
                "/g": {
                    "post": {
                        "operationId": "g",
                        "requestBody": {
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/Order"}
                                }
                            }
                        },
                    }
                }
            }
        )
        (op,) = extract_operations(spec)
        assert op.unsupported_body is None
        (body,) = op.params
        assert body.location == "raw_body" and body.json_schema == {"type": "object"}


class TestNameCollisions:
    """The same name may appear in several locations — tool names must stay unique."""

    SPEC = {
        "openapi": "3.0.0",
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
                        },
                        {"name": "username", "in": "query", "schema": {"type": "string"}},
                        {"name": "username", "in": "header", "schema": {"type": "string"}},
                    ],
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "username": {"type": "string"},
                                        "email": {"type": "string"},
                                    },
                                }
                            }
                        }
                    },
                }
            }
        },
    }

    def test_path_wins_then_query_then_body(self):
        (op,) = extract_operations(self.SPEC, base_url="https://x")
        exposed = [(p.name, p.location, p.wire) for p in op.params if p.location != "header"]
        assert exposed == [
            ("username", "path", "username"),
            ("query_username", "query", "username"),
            ("body_username", "body", "username"),
            ("email", "body", "email"),
        ]
        header = op.param("username")
        assert header is not None and header.location == "path"
        assert [p.name for p in op.params if p.location == "header"] == ["username"]  # untouched


class TestRelativeServerUrls:
    def test_resolved_against_spec_url(self):
        spec = {"servers": [{"url": "/api/v3"}], "paths": {"/p": {"get": {}}}}
        assert spec_base_url(spec) == "/api/v3"  # nothing to resolve against
        assert spec_base_url(spec, spec_url="https://petstore3.swagger.io/api/v3/openapi.json") == (
            "https://petstore3.swagger.io/api/v3"
        )
        assert (
            spec_base_url({"paths": {}}, spec_url="https://h.io/docs/openapi.json")
            == "https://h.io"
        )
        assert spec_base_url(spec, "https://override", spec_url="https://x") == "https://override"
        (op,) = extract_operations(spec, spec_url="https://h.io/x/openapi.json")
        assert op.base_url == "https://h.io/api/v3"

    def test_relative_url_error_is_actionable(self):
        from promptise.mcpcast.plan import build_plan

        spec = {"servers": [{"url": "/api/v3"}], "paths": {"/p": {"get": {"operationId": "p"}}}}
        with pytest.raises(MCPcastError, match=r"--base-url https://<api-host>/api/v3"):
            build_plan(extract_operations(spec))

    def test_is_url(self):
        from promptise.mcpcast.parse import is_url

        assert (
            is_url("https://x") and is_url(" http://x ") and not is_url("/x") and not is_url(None)
        )


def test_swagger2_file_upload_is_unsupported_like_openapi3_multipart():
    """`type: file` is the Swagger 2 spelling of a multipart body."""
    spec = {
        "swagger": "2.0",
        "host": "x.io",
        "paths": {
            "/pet/{petId}/uploadImage": {
                "post": {
                    "operationId": "uploadFile",
                    "consumes": ["multipart/form-data"],
                    "parameters": [
                        {"name": "petId", "in": "path", "required": True, "type": "integer"},
                        {"name": "additionalMetadata", "in": "formData", "type": "string"},
                        {"name": "file", "in": "formData", "type": "file"},
                    ],
                }
            },
            "/pet/{petId}": {
                "post": {
                    "operationId": "updatePetWithForm",
                    "parameters": [
                        {"name": "petId", "in": "path", "required": True, "type": "integer"},
                        {"name": "name", "in": "formData", "type": "string"},
                    ],
                }
            },
        },
    }
    ops = {o.operation_id: o for o in extract_operations(spec)}
    assert (
        ops["uploadFile"].unsupported_body
        == "unsupported request body media type multipart/form-data"
    )
    assert ops["uploadFile"].param("file") is None
    assert ops["updatePetWithForm"].unsupported_body is None  # ordinary form data still works

    from promptise.mcpcast.plan import build_plan
    from promptise.mcpcast.schema import SafetyProfile

    plan = build_plan(ops.values(), profile=SafetyProfile.STANDARD, base_url="https://x.io")
    assert plan.tool_names == ["update_pet_with_form"]
    (dropped,) = plan.dropped
    assert dropped.operation_id == "uploadFile" and "multipart/form-data" in dropped.reason


# ---------------------------------------------------------------------------
# Credentials in the spec URL (audit finding: never derived from, never echoed)
# ---------------------------------------------------------------------------

SECRET_URL = "http://svc-user:S3CRET-TOKEN@127.0.0.1:8099/docs/openapi.json?api_key=QUERY-SECRET#f"
SERVERLESS_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Ledger", "description": "Ledger entries."},
    "paths": {"/entries": {"get": {"operationId": "listEntries", "summary": "List entries"}}},
}


class TestPublicUrl:
    def test_strips_userinfo_query_and_fragment(self):
        from promptise.mcpcast.parse import public_url

        assert public_url(SECRET_URL) == "http://127.0.0.1:8099/docs/openapi.json"
        assert public_url(" https://u:p@h.example/x.json ") == "https://h.example/x.json"
        assert public_url("HTTPS://u:p@[::1]:9/x.json?k=v") == "https://[::1]:9/x.json"
        assert public_url("http://u:p@ss@h.example/x") == "http://h.example/x"  # last '@' wins

    def test_uppercase_scheme_is_still_a_url(self, monkeypatch):
        """RFC 3986 schemes are case-insensitive; a mis-cased one must not fall through to
        the "spec not found" error, which would echo the credential."""
        from promptise.mcpcast.parse import is_url

        assert is_url("HTTP://u:p@h/x") and is_url(" HTTPS://h ")
        _mock_http(monkeypatch, lambda r: httpx.Response(200, json=SERVERLESS_SPEC))
        assert load_spec("HTTP://u:S3CRET@h.example/openapi.json?k=v") == SERVERLESS_SPEC
        _mock_http(monkeypatch, lambda r: httpx.Response(500))
        with pytest.raises(MCPcastError) as info:
            load_spec("HTTP://u:S3CRET@h.example/openapi.json?k=v")
        assert "S3CRET" not in str(info.value) and "http://h.example/openapi.json" in str(
            info.value
        )

    def test_non_urls_are_returned_unchanged(self):
        from promptise.mcpcast.parse import public_url

        for value in ("/tmp/openapi.yaml", "openapi: 3.0.0\npaths: {}", '{"a": 1}', ""):
            assert public_url(value) == value

    def test_exported_from_package(self):
        import promptise.mcpcast as pkg

        assert pkg.public_url is not None and "public_url" in pkg.__all__
        assert "expanded_nodes" in pkg.__all__ and "MAX_DOCUMENT_NODES" in pkg.__all__


class TestCredentialsInSpecUrl:
    def test_credential_is_used_for_the_fetch_only(self, monkeypatch, tmp_path):
        """The userinfo authenticates the download (Basic auth) and goes nowhere else."""
        from promptise.mcpcast import mcpcast, write_project

        requests: list[httpx.Request] = []

        def serve(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(200, json=SERVERLESS_SPEC)

        _mock_http(monkeypatch, serve)
        plan = mcpcast(SECRET_URL)
        assert requests[0].headers["authorization"].startswith("Basic ")
        assert "api_key=QUERY-SECRET" in str(requests[0].url)

        assert plan.api.base_url == "http://127.0.0.1:8099"
        assert "@" not in plan.api.base_url
        assert plan.api.spec_source == "http://127.0.0.1:8099/docs/openapi.json"
        assert plan.api.name == "ledger"

        out = tmp_path / "ledger-mcp"
        write_project(plan, out)
        written = {p: p.read_text(encoding="utf-8") for p in out.rglob("*") if p.is_file()}
        assert written
        for path, text in written.items():
            assert "S3CRET-TOKEN" not in text, path
            assert "svc-user" not in text, path
            assert "QUERY-SECRET" not in text, path
            assert "openapi.json?" not in text, path  # the spec URL's query string

    def test_relative_servers_resolve_against_the_public_url(self, monkeypatch):
        spec = {**SERVERLESS_SPEC, "servers": [{"url": "/v2"}]}
        _mock_http(monkeypatch, lambda r: httpx.Response(200, json=spec))
        (op,) = extract_operations(load_spec(SECRET_URL), spec_url=SECRET_URL)
        assert op.base_url == "http://127.0.0.1:8099/v2" and op.doc_base_url == op.base_url
        assert spec_base_url(spec, spec_url=SECRET_URL) == "http://127.0.0.1:8099/v2"
        assert spec_base_url(SERVERLESS_SPEC, spec_url=SECRET_URL) == "http://127.0.0.1:8099"

    def test_name_from_url_ignores_the_query(self, monkeypatch):
        spec = {"openapi": "3.0.0", "paths": SERVERLESS_SPEC["paths"]}  # no title → file name
        assert api_name_from_spec(spec, "http://u:p@h/ledger-v2.json?api_key=x.y") == "ledger"

    def test_fetch_errors_never_echo_the_credential(self, monkeypatch):
        _mock_http(monkeypatch, lambda r: httpx.Response(401, text="nope"))
        with pytest.raises(MCPcastError) as info:
            load_spec(SECRET_URL)
        message = str(info.value)
        assert "could not fetch spec from http://127.0.0.1:8099/docs/openapi.json" in message
        assert "401" in message
        for secret in ("S3CRET-TOKEN", "svc-user", "QUERY-SECRET", "api_key"):
            assert secret not in message

        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError(f"cannot reach {request.url}", request=request)

        _mock_http(monkeypatch, down)
        with pytest.raises(MCPcastError) as info:
            load_spec(SECRET_URL)
        assert "S3CRET-TOKEN" not in str(info.value) and "ConnectError" in str(info.value)

        body = "x" * 50
        monkeypatch.setenv("MCPCAST_MAX_SPEC_BYTES", "10")
        _mock_http(monkeypatch, lambda r: httpx.Response(200, text=body))
        with pytest.raises(MCPcastError, match="spec at http://127.0.0.1:8099/docs/openapi.json"):
            load_spec(SECRET_URL)


# ---------------------------------------------------------------------------
# Download limits: wall-clock deadline and cancellation
# ---------------------------------------------------------------------------


class _Trickle(httpx.SyncByteStream):
    """A response body that arrives one chunk every *delay* seconds."""

    def __init__(self, chunks: list[bytes], delay: float) -> None:
        self._chunks = chunks
        self._delay = delay

    def __iter__(self):
        for chunk in self._chunks:
            time.sleep(self._delay)
            yield chunk


def _trickling(monkeypatch, *, chunks: int, delay: float) -> None:
    body = [b'{"openapi": "3.0.0", "paths": {}', *([b" "] * chunks), b"}"]
    _mock_http(monkeypatch, lambda r: httpx.Response(200, stream=_Trickle(body, delay)))


class TestFetchDeadline:
    def test_slow_server_hits_the_wall_clock_deadline(self, monkeypatch):
        """httpx's timeout is per read; a trickling server must not hold the CLI forever."""
        _trickling(monkeypatch, chunks=200, delay=0.02)
        monkeypatch.setenv("MCPCAST_FETCH_SECONDS", "0.25")
        started = time.monotonic()
        with pytest.raises(MCPcastError, match="longer than 0.25s.*MCPCAST_FETCH_SECONDS"):
            load_spec("https://slow.example/openapi.json")
        assert time.monotonic() - started < 2

    def test_fast_enough_download_is_unaffected(self, monkeypatch):
        _trickling(monkeypatch, chunks=3, delay=0.01)
        monkeypatch.setenv("MCPCAST_FETCH_SECONDS", "5")
        assert load_spec("https://ok.example/openapi.json") == {"openapi": "3.0.0", "paths": {}}

    def test_env_validation(self, monkeypatch):
        _trickling(monkeypatch, chunks=1, delay=0)
        monkeypatch.setenv("MCPCAST_FETCH_SECONDS", "soon")
        with pytest.raises(MCPcastError, match="MCPCAST_FETCH_SECONDS must be a number"):
            load_spec("https://ok.example/openapi.json")
        monkeypatch.setenv("MCPCAST_FETCH_SECONDS", "0")
        with pytest.raises(MCPcastError, match="MCPCAST_FETCH_SECONDS must be positive"):
            load_spec("https://ok.example/openapi.json")

    def test_cancelled_callback_aborts_the_download(self, monkeypatch):
        _trickling(monkeypatch, chunks=200, delay=0.01)
        polls: list[int] = []

        def cancelled() -> bool:
            polls.append(1)
            return len(polls) >= 3

        started = time.monotonic()
        with pytest.raises(MCPcastError, match="spec download cancelled"):
            load_spec("https://slow.example/openapi.json", cancelled=cancelled)
        assert len(polls) == 3 and time.monotonic() - started < 2

    def test_cancelled_is_keyword_only_and_optional(self, monkeypatch):
        _trickling(monkeypatch, chunks=2, delay=0)
        assert load_spec("https://ok.example/openapi.json", cancelled=lambda: False)["openapi"]
        assert load_spec("https://ok.example/openapi.json")["openapi"] == "3.0.0"
        assert load_spec(SPEC, cancelled=lambda: True) == SPEC  # only a download is cancellable


# ---------------------------------------------------------------------------
# Malformed but OpenAPI-shaped documents (audit finding: no tracebacks)
# ---------------------------------------------------------------------------

SOUND_OP = {"operationId": "listThings", "summary": "List things"}


def _spec_with(**paths) -> dict:
    return {
        "openapi": "3.0.0",
        "info": {"title": "Things"},
        "paths": {"/things": {"get": SOUND_OP}, **paths},
    }


class TestWholeDocumentDefects:
    @pytest.mark.parametrize(
        ("document", "message"),
        [
            (
                {"openapi": "3.0.0", "info": "oops", "paths": {"/x": {"get": {}}}},
                "'info' must be a mapping, got string",
            ),
            (
                {"openapi": "3.0.0", "info": ["oops"], "paths": {"/x": {"get": {}}}},
                "'info' must be a mapping, got list",
            ),
            ({"openapi": "3.0.0", "paths": "nope"}, "'paths' must be a mapping, got string"),
            ({"openapi": "3.0.0", "paths": [{"/x": {}}]}, "'paths' must be a mapping, got list"),
        ],
    )
    def test_load_spec_refuses_them(self, document, message, tmp_path):
        with pytest.raises(MCPcastError, match=f"<inline>: {message}"):
            load_spec(json.dumps(document))
        path = tmp_path / "bad.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        with pytest.raises(MCPcastError, match=f"bad.json: {message}"):
            load_spec(str(path))
        with pytest.raises(MCPcastError, match=f"<mapping>: {message}"):
            load_spec(document)
        with pytest.raises(MCPcastError, match=message):
            extract_operations(document)  # a dict that never went through load_spec

    def test_metadata_getters_stay_total(self):
        """spec_title & co. are used for labels before the document is validated."""
        from promptise.mcpcast.parse import spec_description, spec_summary_line, spec_title

        broken = {"openapi": "3.0.0", "info": "oops"}
        assert spec_title(broken) == "" and spec_description(broken) == ""
        assert spec_summary_line(broken) == "" and api_name_from_spec(broken) == "api"

    def test_too_deep_to_parse(self):
        deep = '{"a":' * 100_000 + "1" + "}" * 100_000
        with pytest.raises(MCPcastError, match="<inline>: document is nested too deeply to parse"):
            load_spec(deep)
        deep_yaml = "openapi: 3.0.0\npaths: {}\nx: " + "{a: " * 100_000 + "1" + "}" * 100_000
        with pytest.raises(MCPcastError, match="nested too deeply"):
            load_spec(deep_yaml)

    def test_depth_cap_is_deterministic(self):
        """A document the parser survives can still be too deep for the walks that follow."""
        from promptise.mcpcast.parse import _MAX_DOCUMENT_DEPTH

        fine = '{"openapi": "3.0.0", "paths": {}, "x": ' + '{"a":' * 200 + "1" + "}" * 200 + "}"
        assert load_spec(fine)["openapi"] == "3.0.0"
        deep = '{"openapi": "3.0.0", "paths": {}, "x": ' + '{"a":' * 300 + "1" + "}" * 300 + "}"
        with pytest.raises(MCPcastError, match=f"nested more than {_MAX_DOCUMENT_DEPTH} levels"):
            load_spec(deep)


def _alias_bomb(levels: int = 9, width: int = 10) -> str:
    """~400 bytes of YAML that expand to ``width ** levels`` nodes."""
    lines = ["a: &a [" + ", ".join(["x"] * width) + "]"]
    prev = "a"
    for i in range(1, levels):
        cur = chr(ord("a") + i)
        lines.append(f"{cur}: &{cur} [" + ", ".join([f"*{prev}"] * width) + "]")
        prev = cur
    return (
        "openapi: 3.0.0\ninfo: {title: bomb}\npaths:\n  /x: {get: {}}\n" + "\n".join(lines) + "\n"
    )


class TestNodeBudget:
    def test_alias_bomb_is_refused_quickly_with_bounded_memory(self, tmp_path):
        import tracemalloc

        bomb = _alias_bomb()
        assert len(bomb) < 1024
        tracemalloc.start()
        try:
            started = time.monotonic()
            with pytest.raises(
                MCPcastError, match="expands to more than 2000000 nodes.*MCPCAST_MAX_SPEC_NODES"
            ):
                load_spec(bomb)
            elapsed = time.monotonic() - started
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        # Refusal is a bounded reference walk (well under a second here; a few seconds
        # on a loaded CI runner under tracemalloc) — expansion would take minutes and GiBs.
        assert elapsed < 30, elapsed
        assert peak < 32 * 1024 * 1024, peak  # the count walks references; nothing is copied
        # the same document through a file and a URL
        path = tmp_path / "bomb.yaml"
        path.write_text(bomb, encoding="utf-8")
        with pytest.raises(MCPcastError, match="bomb.yaml: document expands to more than"):
            load_spec(path)

    def test_budget_is_overridable(self, monkeypatch):
        from promptise.mcpcast.parse import MAX_DOCUMENT_NODES, expanded_nodes

        assert MAX_DOCUMENT_NODES == 2_000_000
        small = "openapi: 3.0.0\npaths: {}\na: &a [x, x, x]\nb: [*a, *a, *a]\n"
        # root + 4 values + a[3] + b[3] + three aliased copies of a[3]: aliases count per use
        assert expanded_nodes(yaml_load(small)) == 20
        monkeypatch.setenv("MCPCAST_MAX_SPEC_NODES", "19")
        with pytest.raises(MCPcastError, match="more than 19 nodes"):
            load_spec(small)
        assert expanded_nodes(yaml_load(small)) > 19  # stops early, still reports "over"
        assert expanded_nodes(yaml_load(small), limit=1000) == 20
        monkeypatch.setenv("MCPCAST_MAX_SPEC_NODES", "0")
        with pytest.raises(MCPcastError, match="MCPCAST_MAX_SPEC_NODES must be positive"):
            load_spec(small)
        monkeypatch.setenv("MCPCAST_MAX_SPEC_NODES", "lots")
        with pytest.raises(MCPcastError, match="MCPCAST_MAX_SPEC_NODES must be an integer"):
            load_spec(small)

    def test_walk_is_iterative(self):
        from promptise.mcpcast.parse import expanded_nodes

        node: dict = {}
        for _ in range(5000):  # far past the recursion limit
            node = {"a": node}
        assert expanded_nodes(node) == 5001


class TestPerOperationDefects:
    """One malformed operation is dropped with a reason; the sound ones survive."""

    @pytest.mark.parametrize(
        ("operation", "reason"),
        [
            (
                {"parameters": "nope"},
                "malformed parameters: expected a list of mappings, got string",
            ),
            (
                {"parameters": ["nope"]},
                "malformed parameters entry: expected a mapping, got string",
            ),
            (
                {"parameters": [{"name": "q", "in": "query", "schema": "nope"}]},
                "malformed parameter 'q' schema: expected a mapping, got string",
            ),
            (
                {"parameters": [{"name": "b", "in": "body", "schema": 5}]},
                "malformed parameter 'b' schema: expected a mapping, got int",
            ),
            ({"requestBody": "nope"}, "malformed requestBody: expected a mapping, got string"),
            (
                {"requestBody": {"content": "nope"}},
                "malformed requestBody.content: expected a mapping, got string",
            ),
            (
                {"requestBody": {"content": {"application/json": {"schema": "nope"}}}},
                "malformed requestBody.content['application/json'].schema: expected a mapping, got string",
            ),
            (
                {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"type": "object", "properties": {"a": "nope"}}
                            }
                        }
                    }
                },
                "malformed body property 'a' schema: expected a mapping, got string",
            ),
        ],
    )
    def test_dropped_with_reason(self, operation, reason):
        spec = _spec_with(**{"/broken": {"post": {"operationId": "broken", **operation}}})
        ops = _by_id(extract_operations(load_spec(json.dumps(spec))))
        assert ops["broken"].unsupported_body == reason and ops["broken"].params == []
        assert ops["listThings"].unsupported_body is None

        from promptise.mcpcast.plan import build_plan
        from promptise.mcpcast.schema import SafetyProfile

        plan = build_plan(ops.values(), profile=SafetyProfile.FULL, base_url="https://t.io")
        assert plan.tool_names == ["list_things"]
        (dropped,) = plan.dropped
        assert dropped.operation_id == "broken" and reason in dropped.reason

    def test_path_level_parameters_apply_to_every_operation_under_the_path(self):
        spec = _spec_with(
            **{
                "/p": {
                    "parameters": "nope",
                    "get": {"operationId": "a"},
                    "post": {"operationId": "b"},
                }
            }
        )
        ops = _by_id(extract_operations(spec))
        reason = "malformed path parameters: expected a list of mappings, got string"
        assert ops["a"].unsupported_body == reason and ops["b"].unsupported_body == reason
        assert ops["listThings"].unsupported_body is None

    def test_scalars_are_coerced(self):
        spec = _spec_with(
            **{
                "/n": {"get": {"operationId": 5, "tags": 5, "deprecated": "yes"}},
                "/m": {
                    "get": {"operationId": {"x": 1}, "tags": ["a", 1, 2.5, True, None, {"b": 1}]}
                },
                "/z": {"get": {"operationId": 0, "tags": "pets"}},
            }
        )
        ops = _by_id(extract_operations(spec))
        assert ops["5"].tags == [] and ops["5"].deprecated is True
        assert ops["x_1"].tags == ["a", "1", "2.5"]
        assert (
            "get_z" in ops and ops["get_z"].tags == []
        )  # a falsy id falls back like a missing one

    def test_malformed_metadata_never_fails_an_operation(self):
        """servers / variables / responses / security only lose their contribution."""
        spec = _spec_with(
            **{
                "/r": {
                    "servers": 5,
                    "get": {
                        "operationId": "r",
                        "servers": "nope",
                        "responses": "nope",
                        "security": "nope",
                    },
                },
                "/s": {"get": {"operationId": "s", "responses": {"200": {"content": "nope"}}}},
                "/t": {
                    "get": {
                        "operationId": "t",
                        "responses": {"200": {"content": {"application/json": {"schema": "nope"}}}},
                    }
                },
                "/u": {"get": {"operationId": "u", "responses": {"200": {"schema": "nope"}}}},
            }
        )
        spec["servers"] = 5
        spec["info"]["x"] = 1
        ops = _by_id(extract_operations(spec, spec_url="https://u:p@h.io/openapi.json"))
        for op_id in ("r", "s", "t", "u"):
            assert ops[op_id].unsupported_body is None and ops[op_id].response_schema is None
        assert ops["r"].base_url == "https://h.io" and ops["r"].scopes == []
        variables = {
            "servers": [{"url": "https://{h}/v1", "variables": "nope"}],
            "paths": {"/x": {"get": {}}},
        }
        with pytest.raises(MCPcastError, match="has a variable with no default"):
            extract_operations(variables)
        variables["servers"][0]["variables"] = {"h": "nope"}
        with pytest.raises(MCPcastError, match="has a variable with no default"):
            extract_operations(variables)


def yaml_load(text: str):
    import yaml

    return yaml.safe_load(text)


# ---------------------------------------------------------------------------
# Control characters are removed at the data boundary
# ---------------------------------------------------------------------------


ESC = "\x1b[2K"


def _hostile_spec() -> dict:
    return {
        "openapi": "3.0.0",
        "info": {"title": f"Acme{ESC}", "description": f"Wipe\x9b2K all\x07 data{ESC}"},
        "servers": [{"url": "https://api.acme.test"}],
        "paths": {
            f"/b{ESC}": {
                "delete": {
                    "operationId": "wipe",
                    "summary": f"Wipe all data{ESC}",
                    "tags": [f"danger{ESC}"],
                    "parameters": [
                        {
                            "name": f"scope{ESC}",
                            "in": "query",
                            "schema": {"type": "string", "example": f"all{ESC}"},
                            "description": f"What\x85 to wipe{ESC}",
                        }
                    ],
                }
            },
            "/upload": {
                "post": {
                    "operationId": "upload",
                    "requestBody": {"content": {f"multipart/form-data{ESC}": {"schema": {}}}},
                }
            },
        },
    }


def _walk_strings(node):
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _walk_strings(k)
            yield from _walk_strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_strings(v)


class TestControlCharactersAreScrubbed:
    @pytest.mark.parametrize("form", ["dict", "json", "yaml", "file"])
    def test_load_spec_removes_them_everywhere(self, form, tmp_path):
        import yaml

        source = _hostile_spec()
        if form == "json":
            source = json.dumps(source)
        elif form == "yaml":
            source = yaml.safe_dump(source, allow_unicode=True, sort_keys=False)
        elif form == "file":
            path = tmp_path / "hostile.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            source = str(path)
        spec = load_spec(source)
        for text in _walk_strings(spec):
            assert "\x1b" not in text and "\x9b" not in text and "\x07" not in text
            assert "\x85" not in text
        assert spec["info"]["title"] == "Acme[2K"
        assert spec["info"]["description"] == "Wipe2K all data[2K"
        assert list(spec["paths"]) == ["/b[2K", "/upload"]
        assert spec["paths"]["/b[2K"]["delete"]["parameters"][0]["name"] == "scope[2K"
        assert list(spec["paths"]["/upload"]["post"]["requestBody"]["content"]) == [
            "multipart/form-data[2K"
        ]

    def test_operations_and_plan_are_clean(self):
        from promptise.mcpcast.plan import build_plan
        from promptise.mcpcast.schema import SafetyProfile

        ops = extract_operations(load_spec(_hostile_spec()))
        for op in ops:
            for text in _walk_strings(op.model_dump()):
                assert "\x1b" not in text
        plan = build_plan(ops, profile=SafetyProfile.FULL)
        assert "\x1b" not in plan.to_yaml()
        wipe = plan.tool("wipe")
        assert wipe.description == "Wipe all data[2K" and wipe.routes[0].path == "/b[2K"
        assert list(wipe.params) == ["scope[2K"] and wipe.tags == ["danger[2K"]
        (dropped,) = plan.dropped
        assert dropped.reason == (
            "unsupported by mcpcast: unsupported request body media type multipart/form-data[2K"
        )

    def test_newlines_and_tabs_survive(self):
        spec = load_spec({"openapi": "3.0.0", "info": {"title": "A\nB\tC"}, "paths": {}})
        assert spec["info"]["title"] == "A\nB\tC"

    def test_public_helpers(self):
        """The wizard parses probe bodies itself and routes them through these two."""
        from promptise.mcpcast import check_document, scrub_strings
        from promptise.mcpcast.parse import check_document as check
        from promptise.mcpcast.parse import scrub_strings as scrub

        assert check is check_document and scrub is scrub_strings
        document = {"openapi": "3.0.0", "info": {"title": f"T{ESC}"}, "paths": {}}
        assert check_document(document, hint="probe") is document
        assert scrub_strings(document) == {
            "openapi": "3.0.0",
            "info": {"title": "T[2K"},
            "paths": {},
        }
        assert scrub_strings([f"a{ESC}", 1, None, {"k\x00": "\ud800"}]) == [
            "a[2K",
            1,
            None,
            {"k": "?"},
        ]
        with pytest.raises(MCPcastError, match="probe: 'paths' must be a mapping"):
            check_document({"paths": "nope"}, hint="probe")


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------


class TestYamlLoader:
    def test_libyaml_parses_but_python_composes(self):
        """libyaml's C composer overflows the C stack (a segfault, not an
        exception) on a few thousand nesting levels; the loader keeps its
        scanner and parser and PyYAML's Python composer, so depth is a
        ``RecursionError`` reported like any other."""
        import yaml

        from promptise.mcpcast.parse import _Loader

        if hasattr(yaml, "CSafeLoader"):
            from yaml.cyaml import CParser

            assert issubclass(_Loader, CParser)
            assert issubclass(_Loader, yaml.composer.Composer)
            assert _Loader.get_single_node is yaml.composer.Composer.get_single_node
        else:
            assert _Loader is yaml.SafeLoader
        assert load_spec("openapi: 3.0.0\ninfo: {title: L}\npaths: {}\n")["info"] == {"title": "L"}
        nested = "openapi: 3.0.0\npaths: {}\nx: " + "{a: " * 5000 + "1" + "}" * 5000
        with pytest.raises(MCPcastError, match="nested too deeply"):
            load_spec(nested)
        deep_lists = "openapi: 3.0.0\npaths: {}\nx: " + "[" * 20000 + "]" * 20000
        with pytest.raises(MCPcastError, match="nested too deeply"):
            load_spec(deep_lists)

    def test_loader_is_safe(self):
        """The hybrid loader constructs with ``SafeConstructor``: no Python objects."""
        import yaml

        from promptise.mcpcast.parse import _Loader

        assert issubclass(_Loader, yaml.constructor.SafeConstructor)
        assert not issubclass(_Loader, yaml.constructor.FullConstructor)
        with pytest.raises(yaml.constructor.ConstructorError):
            yaml.load("a: !!python/object/apply:os.system ['true']", _Loader)  # nosec B506
        with pytest.raises(MCPcastError, match="not valid JSON or YAML"):
            load_spec("openapi: 3.0.0\ninfo: !!python/object/apply:os.system ['true']\npaths: {}\n")

    def test_yaml_errors_and_control_characters_are_reported(self):
        with pytest.raises(MCPcastError, match="not valid JSON or YAML"):
            load_spec("openapi: 3.0.0\npaths: [\n")
        with pytest.raises(MCPcastError, match="not valid JSON or YAML"):
            load_spec("openapi: 3.0.0\ninfo: {title: a\x1bb}\npaths: {}\n")  # YAML forbids raw ESC


# ---------------------------------------------------------------------------
# Names, paths
# ---------------------------------------------------------------------------


class TestInlineAndHome:
    @pytest.mark.parametrize(
        "source", ["<inline>", "<mapping>", '{"openapi": "3.0.0"}', "openapi: 3.0.0\n"]
    )
    def test_untitled_inline_spec_is_named_api(self, source):
        """The wizard records ``<inline>``; the CLI passes the text; mcpcast()
        gets a dict — all four must derive the same name."""
        assert api_name_from_spec({"openapi": "3.0.0"}, source) == "api"
        assert api_name_from_spec({"info": {"title": "Ledger"}}, source) == "ledger"

    def test_untitled_inline_spec_end_to_end(self):
        from promptise.mcpcast import mcpcast

        spec = {
            "openapi": "3.0.0",
            "servers": [{"url": "https://i.example"}],
            "paths": {"/x": {"get": {}}},
        }
        assert mcpcast(spec).api.name == "api"
        assert mcpcast(json.dumps(spec)).api.name == "api"
        assert mcpcast(json.dumps(spec)).api.spec_source == "<inline>"

    def test_file_sources_still_name_the_api(self):
        assert api_name_from_spec({}, "/tmp/billing-openapi.yaml") == "billing"
        assert api_name_from_spec({}, "https://h/ledger.json?key=1") == "ledger"

    def test_tilde_is_expanded(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))  # what ~ means on Windows
        (tmp_path / "api").mkdir()
        (tmp_path / "api" / "openapi.json").write_text(json.dumps(SPEC), encoding="utf-8")
        assert load_spec("~/api/openapi.json") == SPEC
        assert load_spec(Path("~/api/openapi.json")) == SPEC
        with pytest.raises(MCPcastError, match="spec not found"):
            load_spec("~/api/missing.json")


# ---------------------------------------------------------------------------
# Security schemes
# ---------------------------------------------------------------------------


def _secured(security_schemes: dict, global_security, **paths) -> dict:
    spec = {
        "openapi": "3.0.0",
        "servers": [{"url": "https://s.example"}],
        "components": {"securitySchemes": security_schemes},
        "paths": paths or {"/things": {"get": {"operationId": "listThings"}}},
    }
    if global_security is not None:
        spec["security"] = global_security
    return spec


class TestSecuritySchemes:
    def test_api_key_in_header(self):
        spec = _secured(
            {"ApiKeyAuth": {"type": "apiKey", "in": "header", "name": "X-API-Key"}},
            [{"ApiKeyAuth": []}],
        )
        (op,) = extract_operations(spec)
        scheme = op.security_scheme
        assert scheme is not None
        assert (scheme.key, scheme.type, scheme.location, scheme.name) == (
            "ApiKeyAuth",
            "apiKey",
            "header",
            "X-API-Key",
        )
        assert scheme.credential == ("header", "X-API-Key")
        assert scheme.describe() == "an API key in header 'X-API-Key'"

    def test_api_key_in_query(self):
        spec = _secured(
            {"key": {"type": "apiKey", "in": "query", "name": "api_key"}}, [{"key": []}]
        )
        (op,) = extract_operations(spec)
        assert op.security_scheme is not None
        assert op.security_scheme.credential == ("query", "api_key")

    def test_swagger2_security_definitions(self):
        spec = {
            "swagger": "2.0",
            "host": "s.example",
            "securityDefinitions": {
                "api_key": {"type": "apiKey", "in": "header", "name": "api_key"},
                "basic": {"type": "basic"},
            },
            "security": [{"api_key": []}],
            "paths": {
                "/pets": {"get": {"operationId": "listPets"}},
                "/store": {"get": {"operationId": "store", "security": [{"basic": []}]}},
            },
        }
        ops = _by_id(extract_operations(spec))
        assert ops["listPets"].security_scheme.credential == ("header", "api_key")
        basic = ops["store"].security_scheme
        assert basic.type == "basic" and basic.credential == ("header", "Authorization")
        assert basic.describe() == "HTTP basic authentication"

    def test_http_oauth2_and_openid_use_the_authorization_header(self):
        schemes = {
            "bearer": {"type": "http", "scheme": "Bearer"},
            "oauth": {"type": "oauth2", "flows": {}},
            "oidc": {"type": "openIdConnect", "openIdConnectUrl": "https://x/.well-known"},
        }
        spec = _secured(
            schemes,
            [{"bearer": []}],
            **{
                "/a": {"get": {"operationId": "a"}},
                "/b": {"get": {"operationId": "b", "security": [{"oauth": ["read"]}]}},
                "/c": {"get": {"operationId": "c", "security": [{"oidc": []}]}},
            },
        )
        ops = _by_id(extract_operations(spec))
        assert ops["a"].security_scheme.scheme == "bearer"
        assert ops["a"].security_scheme.describe() == "HTTP bearer authentication"
        for oid in ("a", "b", "c"):
            assert ops[oid].security_scheme.credential == ("header", "Authorization")
        assert ops["b"].scopes == ["read"]  # scopes are still collected

    def test_operation_override_and_explicit_none(self):
        spec = _secured(
            {
                "key": {"type": "apiKey", "in": "header", "name": "X-Key"},
                "bearer": {"type": "http", "scheme": "bearer"},
            },
            [{"key": []}],
            **{
                "/a": {"get": {"operationId": "a"}},
                "/b": {"get": {"operationId": "b", "security": [{"bearer": []}]}},
                "/c": {"get": {"operationId": "c", "security": []}},
            },
        )
        ops = _by_id(extract_operations(spec))
        assert ops["a"].security_scheme.credential == ("header", "X-Key")
        assert ops["b"].security_scheme.credential == ("header", "Authorization")
        assert ops["c"].security_scheme is None

    def test_first_presentable_alternative_wins(self):
        """``[{cookie}, {bearer}]``: the cookie key cannot be presented, bearer can."""
        spec = _secured(
            {
                "cookie": {"type": "apiKey", "in": "cookie", "name": "session"},
                "bearer": {"type": "http", "scheme": "bearer"},
                "two": {"type": "apiKey", "in": "header", "name": "X-Id"},
            },
            [{"cookie": []}, {"two": [], "bearer": []}, {"bearer": []}],
        )
        (op,) = extract_operations(spec)
        assert op.security_scheme.key == "bearer"

    def test_unpresentable_scheme_is_still_recorded(self):
        spec = _secured(
            {"cookie": {"type": "apiKey", "in": "cookie", "name": "session"}}, [{"cookie": []}]
        )
        (op,) = extract_operations(spec)
        assert op.security_scheme.credential is None
        assert op.security_scheme.describe() == "an API key in cookie 'session'"
        mtls = _secured({"m": {"type": "mutualTLS"}}, [{"m": []}])
        (op,) = extract_operations(mtls)
        assert op.security_scheme.type == "mutualTLS" and op.security_scheme.credential is None

    def test_dangling_malformed_and_ref_schemes(self):
        spec = _secured(
            {
                "real": {"$ref": "#/components/securitySchemes/target"},
                "target": {"type": "apiKey", "in": "header", "name": "X-T"},
                "broken": "nope",
                "weird": {"type": "hmac"},
                "external": {"$ref": "other.yaml#/S"},
            },
            [{"missing": []}, {"broken": []}, {"weird": []}, {"external": []}, {"real": []}],
        )
        (op,) = extract_operations(spec)
        assert op.security_scheme.key == "real" and op.security_scheme.name == "X-T"
        nothing = _secured({}, [{"missing": []}])
        (op,) = extract_operations(nothing)
        assert op.security_scheme is None
        (op,) = extract_operations(
            _secured({"k": {"type": "apiKey", "in": "header", "name": "X"}}, "nope")
        )
        assert op.security_scheme is None


# ---------------------------------------------------------------------------
# $ref is a reference only when its value is a string
# ---------------------------------------------------------------------------


def _ref_spec(body_schema: dict, **components) -> dict:
    return {
        "openapi": "3.0.0",
        "info": {"title": "Registry", "version": "1"},
        "servers": [{"url": "https://r.example"}],
        "components": {"schemas": components},
        "paths": {
            "/schemas": {
                "post": {
                    "operationId": "registerSchema",
                    "requestBody": {"content": {"application/json": {"schema": body_schema}}},
                }
            }
        },
    }


class TestRefNamedProperties:
    def test_property_named_ref_at_hop_zero(self):
        spec = _ref_spec(
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "$ref": {"type": "string", "description": "A JSON pointer"},
                    "$schema": {"type": "string"},
                },
            }
        )
        (op,) = extract_operations(spec)
        assert op.unsupported_body is None
        assert [p.name for p in op.params] == ["name", "$ref", "$schema"]
        assert op.param("$ref").json_schema == {"type": "string", "description": "A JSON pointer"}

    def test_property_named_ref_behind_two_hops(self):
        spec = _ref_spec(
            {"$ref": "#/components/schemas/Envelope"},
            Envelope={
                "type": "object",
                "properties": {"schema": {"$ref": "#/components/schemas/JsonSchema"}},
            },
            JsonSchema={
                "type": "object",
                "properties": {
                    "$ref": {"type": "string"},
                    "$schema": {"type": "string"},
                    "$defs": {"type": "object"},
                },
            },
        )
        (op,) = extract_operations(spec)
        assert op.unsupported_body is None
        (schema_param,) = op.params
        assert schema_param.name == "schema"
        assert set(schema_param.json_schema["properties"]) == {"$ref", "$schema", "$defs"}
        assert schema_param.json_schema["properties"]["$ref"] == {"type": "string"}
        from promptise.mcpcast.plan import build_plan, example_value
        from promptise.mcpcast.schema import SafetyProfile

        assert example_value(schema_param.json_schema, "schema") == {
            "$ref": "string",
            "$schema": "string",
            "$defs": {},
        }
        plan = build_plan([op], profile=SafetyProfile.STANDARD)
        assert plan.tool_names == ["register_schema"]

    def test_example_object_with_ref_key_is_preserved_verbatim(self):
        example = {"$ref": "#/info", "title": "t", "nested": {"$ref": "#/components"}}
        spec = _ref_spec(
            {
                "type": "object",
                "properties": {
                    "doc": {"type": "object", "additionalProperties": True, "example": example},
                    "kind": {"type": "string", "enum": ["$ref", "plain"], "default": "$ref"},
                    "pinned": {"type": "object", "const": {"$ref": "x"}},
                    "many": {
                        "type": "array",
                        "items": {"type": "object"},
                        "examples": [[{"$ref": "y"}]],
                    },
                },
            }
        )
        (op,) = extract_operations(spec)
        assert op.param("doc").json_schema["example"] == example
        assert op.param("kind").json_schema == {
            "type": "string",
            "enum": ["$ref", "plain"],
            "default": "$ref",
        }
        assert op.param("pinned").json_schema["const"] == {"$ref": "x"}
        assert op.param("many").json_schema["examples"] == [[{"$ref": "y"}]]
        from promptise.mcpcast.plan import example_value

        assert example_value(op.param("doc").json_schema, "doc") == example

    def test_string_refs_are_still_followed_and_refused(self):
        spec = _ref_spec({"$ref": "#/components/schemas/Missing"})
        (op,) = extract_operations(spec)
        assert "dangling $ref '#/components/schemas/Missing'" in op.unsupported_body
        spec = _ref_spec({"$ref": "schemas.yaml#/Thing"})
        (op,) = extract_operations(spec)
        assert "external $ref 'schemas.yaml#/Thing'" in op.unsupported_body
        spec = _ref_spec(
            {"$ref": "#/components/schemas/Thing"},
            Thing={"type": "object", "properties": {"id": {"type": "integer"}}},
        )
        (op,) = extract_operations(spec)
        assert [p.name for p in op.params] == ["id"]

    def test_non_string_ref_value_is_a_plain_key(self):
        spec = _ref_spec({"type": "object", "properties": {"x": {"$ref": {"type": "string"}}}})
        (op,) = extract_operations(spec)
        assert op.unsupported_body is None
        assert op.param("x").json_schema == {"$ref": {"type": "string"}}

    def test_path_item_ref_must_be_a_string(self):
        spec = _ref_spec({"type": "object"})
        spec["paths"]["/odd"] = {"$ref": {"not": "a pointer"}, "get": {"operationId": "odd"}}
        ids = [o.operation_id for o in extract_operations(spec)]
        assert ids == ["registerSchema", "odd"]
