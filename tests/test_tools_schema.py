"""Tests for _jsonschema_to_pydantic — recursive JSON Schema → Pydantic model.

Covers: primitives, nested objects, arrays-of-objects, $ref/$defs,
anyOf/oneOf (Optional), allOf, enums, descriptions, defaults.
"""

from __future__ import annotations

import json

from pydantic import BaseModel

from promptise.tools import _jsonschema_to_pydantic  # type: ignore

# =====================================================================
# Basic primitives (existing)
# =====================================================================


class TestBasicTypes:
    def test_all_primitives(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "s": {"type": "string", "description": "a string"},
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
            },
            "required": ["s", "i"],
        }
        model = _jsonschema_to_pydantic(schema, model_name="Primitives")
        fields = model.model_fields
        assert fields["s"].is_required()
        assert fields["i"].is_required()
        assert fields["n"].is_required() is False
        assert fields["b"].is_required() is False
        assert fields["s"].description == "a string"

    def test_empty_schema(self) -> None:
        # A tool without parameters gets an empty object schema, not a
        # made-up ``payload`` parameter.
        model = _jsonschema_to_pydantic({}, model_name="Empty")
        assert model.model_fields == {}
        assert model.model_json_schema()["properties"] == {}

    def test_free_form_object_passes_keys_through(self) -> None:
        model = _jsonschema_to_pydantic(
            {"type": "object", "additionalProperties": True}, model_name="FreeForm"
        )
        assert model.model_fields == {}
        assert model(a=1, b="x").model_dump() == {"a": 1, "b": "x"}

    def test_nested_object_without_properties_is_dict(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "cfg": {"type": "object", "properties": {}},
                "rows": {"type": "array", "items": {"type": "object"}},
            },
        }
        model = _jsonschema_to_pydantic(schema, model_name="Loose")
        instance = model(cfg={"k": 1}, rows=[{"a": 1}])
        assert instance.cfg == {"k": 1}
        assert instance.rows == [{"a": 1}]

    def test_defaults_preserved(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "default": 10, "description": "Max results"},
                "active": {"type": "boolean", "default": True},
            },
        }
        model = _jsonschema_to_pydantic(schema, model_name="Defaults")
        inst = model()
        assert inst.limit == 10  # type: ignore[attr-defined]
        assert inst.active is True  # type: ignore[attr-defined]

    def test_array_of_strings(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["tags"],
        }
        model = _jsonschema_to_pydantic(schema, model_name="Tags")
        # Should accept list of strings
        inst = model.model_validate({"tags": ["a", "b"]})
        assert inst.tags == ["a", "b"]  # type: ignore[attr-defined]


# =====================================================================
# Nested objects
# =====================================================================


class TestNestedObjects:
    def test_nested_object_creates_submodel(self) -> None:
        """Object with properties should create a Pydantic sub-model, NOT dict."""
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string", "description": "Street name"},
                        "city": {"type": "string"},
                        "zip_code": {"type": "string", "default": "00000"},
                    },
                    "required": ["street", "city"],
                },
            },
            "required": ["address"],
        }
        model = _jsonschema_to_pydantic(schema, model_name="WithAddr")
        # The address field annotation should be a Pydantic model, NOT dict
        addr_field = model.model_fields["address"]
        addr_type = addr_field.annotation
        assert isinstance(addr_type, type)
        assert issubclass(addr_type, BaseModel)
        # Sub-model should have proper fields
        assert "street" in addr_type.model_fields
        assert "city" in addr_type.model_fields
        assert addr_type.model_fields["street"].is_required()

    def test_nested_validation_works(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "employee": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "city": {"type": "string"},
                            },
                            "required": ["city"],
                        },
                    },
                    "required": ["name", "address"],
                },
            },
            "required": ["employee"],
        }
        model = _jsonschema_to_pydantic(schema, model_name="DeepNest")
        inst = model.model_validate(
            {
                "employee": {
                    "name": "Alice",
                    "address": {"city": "SF"},
                }
            }
        )
        assert inst.employee.name == "Alice"  # type: ignore[attr-defined]
        assert inst.employee.address.city == "SF"  # type: ignore[attr-defined]

    def test_object_without_properties_stays_dict(self) -> None:
        """An object type without properties should remain `dict`."""
        schema = {
            "type": "object",
            "properties": {
                "metadata": {"type": "object"},
            },
        }
        model = _jsonschema_to_pydantic(schema, model_name="Meta")
        ann = model.model_fields["metadata"].annotation
        assert ann is dict


# =====================================================================
# Arrays of objects
# =====================================================================


class TestArrayOfObjects:
    def test_array_of_objects(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "color": {"type": "string", "default": "blue"},
                        },
                        "required": ["label"],
                    },
                },
            },
            "required": ["tags"],
        }
        model = _jsonschema_to_pydantic(schema, model_name="WithTags")
        inst = model.model_validate(
            {
                "tags": [
                    {"label": "urgent", "color": "red"},
                    {"label": "review"},
                ]
            }
        )
        assert len(inst.tags) == 2  # type: ignore[attr-defined]
        assert inst.tags[0].label == "urgent"  # type: ignore[attr-defined]
        assert inst.tags[1].color == "blue"  # type: ignore[attr-defined]


# =====================================================================
# $ref / $defs
# =====================================================================


class TestRefDefs:
    def test_simple_ref(self) -> None:
        schema = {
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
            "type": "object",
            "properties": {
                "home": {"$ref": "#/$defs/Address"},
            },
            "required": ["home"],
        }
        model = _jsonschema_to_pydantic(schema, model_name="RefTest")
        inst = model.model_validate({"home": {"city": "NYC", "zip": "10001"}})
        assert inst.home.city == "NYC"  # type: ignore[attr-defined]

    def test_nested_refs(self) -> None:
        """$ref pointing to a model that itself has a $ref."""
        schema = {
            "$defs": {
                "Zip": {"type": "string"},
                "Address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip_code": {"$ref": "#/$defs/Zip"},
                    },
                    "required": ["city"],
                },
            },
            "type": "object",
            "properties": {
                "address": {"$ref": "#/$defs/Address"},
            },
            "required": ["address"],
        }
        model = _jsonschema_to_pydantic(schema, model_name="NestedRef")
        inst = model.model_validate({"address": {"city": "LA", "zip_code": "90001"}})
        assert inst.address.city == "LA"  # type: ignore[attr-defined]

    def test_array_items_ref(self) -> None:
        schema = {
            "$defs": {
                "Tag": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                    },
                    "required": ["label"],
                },
            },
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Tag"},
                },
            },
        }
        model = _jsonschema_to_pydantic(schema, model_name="ArrayRef")
        inst = model.model_validate({"tags": [{"label": "x"}]})
        assert inst.tags[0].label == "x"  # type: ignore[attr-defined]

    def test_multiple_refs_same_def(self) -> None:
        schema = {
            "$defs": {
                "Addr": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
            "type": "object",
            "properties": {
                "home": {"$ref": "#/$defs/Addr"},
                "work": {"$ref": "#/$defs/Addr"},
            },
            "required": ["home", "work"],
        }
        model = _jsonschema_to_pydantic(schema, model_name="MultiRef")
        inst = model.model_validate({"home": {"city": "A"}, "work": {"city": "B"}})
        assert inst.home.city == "A"  # type: ignore[attr-defined]
        assert inst.work.city == "B"  # type: ignore[attr-defined]


# =====================================================================
# anyOf / oneOf (Optional types)
# =====================================================================


class TestUnionTypes:
    def test_optional_via_anyof(self) -> None:
        """Pydantic generates anyOf for Optional fields."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "nickname": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                },
            },
            "required": ["name"],
        }
        model = _jsonschema_to_pydantic(schema, model_name="OptTest")
        # Should accept None
        inst = model.model_validate({"name": "Alice", "nickname": None})
        assert inst.nickname is None  # type: ignore[attr-defined]
        # Should accept string
        inst2 = model.model_validate({"name": "Bob", "nickname": "Bobby"})
        assert inst2.nickname == "Bobby"  # type: ignore[attr-defined]

    def test_optional_object_via_anyof(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                        {"type": "null"},
                    ],
                },
            },
        }
        model = _jsonschema_to_pydantic(schema, model_name="OptObj")
        inst = model.model_validate({"address": {"city": "NYC"}})
        assert inst.address.city == "NYC"  # type: ignore[attr-defined]
        inst2 = model.model_validate({"address": None})
        assert inst2.address is None  # type: ignore[attr-defined]


# =====================================================================
# allOf
# =====================================================================


class TestAllOf:
    def test_allof_merges_properties(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "record": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"id": {"type": "integer"}},
                            "required": ["id"],
                        },
                        {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                    ],
                },
            },
            "required": ["record"],
        }
        model = _jsonschema_to_pydantic(schema, model_name="AllOfTest")
        inst = model.model_validate({"record": {"id": 1, "name": "X"}})
        assert inst.record.id == 1  # type: ignore[attr-defined]
        assert inst.record.name == "X"  # type: ignore[attr-defined]


# =====================================================================
# Enum
# =====================================================================


class TestEnum:
    def test_string_enum(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive", "pending"],
                    "description": "Account status",
                },
            },
            "required": ["status"],
        }
        model = _jsonschema_to_pydantic(schema, model_name="EnumTest")
        inst = model.model_validate({"status": "active"})
        assert inst.status == "active"  # type: ignore[attr-defined]


# =====================================================================
# Real-world: Pydantic server schema round-trip
# =====================================================================


class TestRealWorldSchemas:
    def test_employee_schema(self) -> None:
        """Schema matching our example server's create_employee tool."""
        schema = {
            "type": "object",
            "properties": {
                "employee": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1, "description": "Full name"},
                        "email": {"type": "string", "description": "Work email"},
                        "department": {"type": "string"},
                        "salary": {"type": "number", "description": "Annual salary"},
                        "address": {
                            "type": "object",
                            "properties": {
                                "street": {"type": "string"},
                                "city": {"type": "string"},
                                "country": {"type": "string", "default": "US"},
                                "zip_code": {"type": "string"},
                            },
                            "required": ["street", "city", "zip_code"],
                        },
                        "skills": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": [],
                        },
                        "is_remote": {"type": "boolean", "default": False},
                    },
                    "required": ["name", "email", "department", "salary", "address"],
                },
            },
            "required": ["employee"],
        }
        model = _jsonschema_to_pydantic(schema, model_name="Args_create_employee")

        # The employee field should be a Pydantic model
        emp_type = model.model_fields["employee"].annotation
        assert issubclass(emp_type, BaseModel)

        # The address field inside employee should also be a model
        addr_type = emp_type.model_fields["address"].annotation
        assert issubclass(addr_type, BaseModel)
        assert "street" in addr_type.model_fields
        assert "city" in addr_type.model_fields

        # Full validation works
        inst = model.model_validate(
            {
                "employee": {
                    "name": "Alice",
                    "email": "alice@co.com",
                    "department": "Eng",
                    "salary": 150000,
                    "address": {
                        "street": "123 Main",
                        "city": "SF",
                        "zip_code": "94105",
                    },
                    "skills": ["python", "go"],
                    "is_remote": True,
                }
            }
        )
        assert inst.employee.name == "Alice"  # type: ignore[attr-defined]
        assert inst.employee.address.city == "SF"  # type: ignore[attr-defined]
        assert inst.employee.skills == ["python", "go"]  # type: ignore[attr-defined]

    def test_schema_json_roundtrip(self) -> None:
        """Generated model's own schema should be valid JSON Schema."""
        schema = {
            "type": "object",
            "properties": {
                "item": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "tags": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "color": {"type": "string", "default": "blue"},
                                },
                                "required": ["label"],
                            },
                        },
                    },
                    "required": ["title"],
                },
            },
            "required": ["item"],
        }
        model = _jsonschema_to_pydantic(schema, model_name="RoundTrip")
        # Should produce valid JSON
        generated = model.model_json_schema()
        assert isinstance(json.dumps(generated), str)
        assert "title" in str(generated)

    def test_description_propagation(self) -> None:
        """Descriptions from JSON Schema should propagate to Pydantic fields."""
        schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search text to match against names",
                },
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Max results to return",
                },
            },
            "required": ["query"],
        }
        model = _jsonschema_to_pydantic(schema, model_name="DescTest")
        assert model.model_fields["query"].description == "Search text to match against names"
        assert model.model_fields["limit"].description == "Max results to return"


class TestSchemaFidelity:
    """What a server's input schema says about a parameter reaches the model the agent sees."""

    @staticmethod
    def _props(schema: dict) -> dict:
        model = _jsonschema_to_pydantic(schema, model_name="Fidelity")
        return model.model_json_schema()["properties"]

    def test_a_type_list_is_a_union_not_a_crash(self) -> None:
        """``type: ["string", "null"]`` (JSON Schema, OpenAPI 3.1, MCPcast's nullable)
        raised ``TypeError: unhashable type: 'list'`` and broke the whole tool list."""
        model = _jsonschema_to_pydantic(
            {
                "type": "object",
                "required": ["since"],
                "properties": {
                    "since": {"type": ["string", "null"], "format": "date-time"},
                    "id": {"type": ["string", "integer"]},
                },
            },
            model_name="TypeList",
        )
        assert model(since=None).since is None
        assert model(since="2026-01-01T00:00:00Z", id=7).id == 7
        props = model.model_json_schema()["properties"]
        assert {"type": "string", "format": "date-time"} in props["since"]["anyOf"]
        assert {"type": "null"} in props["since"]["anyOf"]
        assert {m["type"] for m in props["id"]["anyOf"] if "type" in m} >= {"string", "integer"}

    def test_constraints_are_shown_to_the_model(self) -> None:
        props = self._props(
            {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "pattern": "^[A-Z]{3}$", "minLength": 3},
                    "limit": {"type": "integer", "minimum": 1, "exclusiveMaximum": 100},
                    "price": {"type": "number", "multipleOf": 0.01},
                    "email": {"type": "string", "format": "email"},
                    "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
                },
            }
        )
        assert props["sku"]["pattern"] == "^[A-Z]{3}$" and props["sku"]["minLength"] == 3
        assert props["limit"]["minimum"] == 1 and props["limit"]["exclusiveMaximum"] == 100
        assert props["price"]["multipleOf"] == 0.01
        assert props["email"]["format"] == "email"
        assert props["tags"]["maxItems"] == 3

    def test_constraints_are_not_enforced_on_the_agent_side(self) -> None:
        """A JavaScript-only pattern (lookbehind) must not break or reject anything here."""
        model = _jsonschema_to_pydantic(
            {"type": "object", "properties": {"code": {"type": "string", "pattern": "(?<=x)y"}}},
            model_name="Unenforced",
        )
        assert model(code="anything").code == "anything"

    def test_integer_and_nullable_enums_are_kept(self) -> None:
        props = self._props(
            {
                "type": "object",
                "properties": {
                    "level": {"type": "integer", "enum": [1, 2, 3]},
                    "state": {"type": "string", "enum": ["on", "off", None]},
                    "ratio": {"type": "number", "enum": [0.5, 1.5]},
                },
            }
        )
        assert props["level"]["enum"] == [1, 2, 3]
        (state,) = [m for m in props["state"]["anyOf"] if m.get("type") == "string"]
        assert state["enum"] == ["on", "off"]
        assert {"type": "null"} in props["state"]["anyOf"]
        assert props["ratio"]["enum"] == [0.5, 1.5]

    def test_array_items_keep_their_enum_and_constraints(self) -> None:
        props = self._props(
            {
                "type": "object",
                "properties": {
                    "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
                    "codes": {"type": "array", "items": {"type": "string", "maxLength": 4}},
                },
            }
        )
        assert props["tags"]["items"]["enum"] == ["a", "b"]
        assert props["codes"]["items"]["maxLength"] == 4

    def test_openapi_nullable_is_optional(self) -> None:
        model = _jsonschema_to_pydantic(
            {
                "type": "object",
                "required": ["parent"],
                "properties": {"parent": {"type": "integer", "nullable": True}},
            },
            model_name="Nullable",
        )
        assert model(parent=None).parent is None
        assert model.model_fields["parent"].is_required()

    def test_a_nullable_union_of_several_types(self) -> None:
        """FastAPI's ``Union[str, int, None]`` raised ``TypeError: typing.Optional
        requires a single type`` and broke the whole tool list."""
        model = _jsonschema_to_pydantic(
            {
                "type": "object",
                "properties": {
                    "note": {"anyOf": [{"type": "string"}, {"type": "integer"}, {"type": "null"}]}
                },
            },
            model_name="NullableUnion",
        )
        assert model(note=None).note is None and model(note=3).note == 3
        types = [m.get("type") for m in model.model_json_schema()["properties"]["note"]["anyOf"]]
        assert types == ["string", "integer", "null"]
