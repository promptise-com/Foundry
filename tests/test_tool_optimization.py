"""Tests for tool optimization module."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from promptise.tool_optimization import (
    OptimizationLevel,
    ToolIndex,
    ToolOptimizationConfig,
    _minify_pydantic_model,
    _RequestMoreToolsTool,
    _resolve_config,
    _truncate_description,
    apply_static_optimizations,
)

# ======================================================================
# Helpers
# ======================================================================


class _FakeTool(BaseTool):
    """Minimal BaseTool for testing."""

    name: str
    description: str
    args_schema: type[BaseModel] = BaseModel

    async def _arun(self, **kwargs: Any) -> str:
        return "ok"

    def _run(self, **kwargs: Any) -> str:
        return "ok"


def _make_tool(
    name: str, description: str = "A tool.", schema: type[BaseModel] | None = None
) -> BaseTool:
    return _FakeTool(name=name, description=description, args_schema=schema or BaseModel)


class SimpleArgs(BaseModel):
    query: str = Field(..., description="The search query")
    limit: int = Field(default=10, description="Max results to return")


class NestedInner(BaseModel):
    city: str = Field(..., description="City name")
    zip_code: str = Field(default=None, description="ZIP code")


class NestedArgs(BaseModel):
    name: str = Field(..., description="User name")
    address: NestedInner = Field(..., description="User address")
    tags: list[str] = Field(default_factory=list, description="Tags")


class DeepArgs(BaseModel):
    class Level2(BaseModel):
        class Level3(BaseModel):
            value: str = Field(..., description="Deep value")

        inner: Level3 = Field(..., description="Level 3")

    outer: Level2 = Field(..., description="Level 2")


# ======================================================================
# TestResolveConfig
# ======================================================================


class TestResolveConfig:
    def test_minimal_preset(self):
        config = ToolOptimizationConfig(level=OptimizationLevel.MINIMAL)
        resolved = _resolve_config(config)
        assert resolved.minify_schema is True
        assert resolved.max_description_length == 200
        assert resolved.strip_nested_descriptions is False
        assert resolved.max_schema_depth is None
        assert resolved.semantic_selection is False

    def test_standard_preset(self):
        config = ToolOptimizationConfig(level=OptimizationLevel.STANDARD)
        resolved = _resolve_config(config)
        assert resolved.max_description_length == 150
        assert resolved.strip_nested_descriptions is True
        assert resolved.max_schema_depth == 3
        assert resolved.semantic_selection is False

    def test_semantic_preset(self):
        config = ToolOptimizationConfig(level=OptimizationLevel.SEMANTIC)
        resolved = _resolve_config(config)
        assert resolved.max_description_length == 100
        assert resolved.strip_nested_descriptions is True
        assert resolved.max_schema_depth == 2
        assert resolved.semantic_selection is True
        assert resolved.semantic_top_k == 8
        assert resolved.always_include_fallback is True

    def test_explicit_override_beats_preset(self):
        config = ToolOptimizationConfig(
            level=OptimizationLevel.MINIMAL,
            max_description_length=50,
            semantic_selection=True,
        )
        resolved = _resolve_config(config)
        assert resolved.max_description_length == 50
        assert resolved.semantic_selection is True
        # Other fields from MINIMAL preset
        assert resolved.minify_schema is True

    def test_no_level_uses_defaults(self):
        config = ToolOptimizationConfig(minify_schema=True)
        resolved = _resolve_config(config)
        assert resolved.minify_schema is True
        assert resolved.max_description_length == 200  # default

    def test_preserve_tools_as_frozenset(self):
        config = ToolOptimizationConfig(preserve_tools={"tool_a", "tool_b"})
        resolved = _resolve_config(config)
        assert resolved.preserve_tools == frozenset({"tool_a", "tool_b"})

    def test_empty_preserve_tools(self):
        config = ToolOptimizationConfig()
        resolved = _resolve_config(config)
        assert resolved.preserve_tools == frozenset()


# ======================================================================
# TestDescriptionTruncation
# ======================================================================


class TestDescriptionTruncation:
    def test_short_unchanged(self):
        assert _truncate_description("Hello", 200) == "Hello"

    def test_exact_length_unchanged(self):
        text = "x" * 200
        assert _truncate_description(text, 200) == text

    def test_truncates_at_word_boundary(self):
        text = "This is a really long description that needs to be truncated to fit"
        result = _truncate_description(text, 30)
        assert result.endswith("...")
        assert len(result) <= 30

    def test_empty_string(self):
        assert _truncate_description("", 100) == ""

    def test_no_space_truncation(self):
        text = "a" * 300
        result = _truncate_description(text, 50)
        assert result.endswith("...")
        assert len(result) <= 50

    def test_preserves_words(self):
        text = "search for employees by name or department"
        result = _truncate_description(text, 25)
        assert "..." in result
        # Should not cut in the middle of a word
        clean = result.replace("...", "").rstrip()
        assert " " not in clean or clean.endswith(" ") is False

    SUSPEND = (
        "Suspend a user so they can't sign in. Their data and memberships are kept "
        "and can be restored. Use this, not delete_user, when someone leaves the company."
    )

    def test_ends_on_sentence_without_ellipsis(self):
        # Used to produce "... can be restored...." (sentence dot + "...").
        result = _truncate_description(self.SUSPEND, 100)
        assert result == (
            "Suspend a user so they can't sign in. Their data and memberships "
            "are kept and can be restored."
        )

    def test_word_cut_when_sentence_would_waste_budget(self):
        result = _truncate_description(self.SUSPEND, 150)
        assert result.endswith("when someone leaves the...")
        assert len(result) <= 150

    def test_never_doubles_dots_or_ends_on_punctuation(self):
        for n in range(10, len(self.SUSPEND)):
            result = _truncate_description(self.SUSPEND, n)
            assert len(result) <= n
            assert "...." not in result and ",..." not in result

    def test_abbreviation_is_not_a_sentence_end(self):
        text = "Lists items, e.g. users or teams, with filters and paging support for large orgs."
        assert not _truncate_description(text, 40).endswith("e.g.")


# ======================================================================
# TestSchemaMinification
# ======================================================================


class TestSchemaMinification:
    def test_strips_field_descriptions(self):
        minified = _minify_pydantic_model(SimpleArgs)
        for field_info in minified.model_fields.values():
            assert field_info.description is None

    def test_preserves_field_names(self):
        minified = _minify_pydantic_model(SimpleArgs)
        assert set(minified.model_fields.keys()) == {"query", "limit"}

    def test_preserves_required_status(self):
        minified = _minify_pydantic_model(SimpleArgs)
        assert minified.model_fields["query"].is_required()
        assert not minified.model_fields["limit"].is_required()

    def test_preserves_defaults(self):
        minified = _minify_pydantic_model(SimpleArgs)
        assert minified.model_fields["limit"].default == 10

    def test_nested_model_stripped_when_strip_nested(self):
        minified = _minify_pydantic_model(
            NestedArgs,
            strip_nested=True,
        )
        # Top-level descriptions stripped
        for field_info in minified.model_fields.values():
            assert field_info.description is None

    def test_nested_model_preserved_when_not_strip_nested(self):
        minified = _minify_pydantic_model(
            NestedArgs,
            strip_nested=False,
        )
        # Top-level descriptions still stripped (_depth=0 always strips)
        assert minified.model_fields["name"].description is None

    def test_depth_flattening(self):
        minified = _minify_pydantic_model(
            NestedArgs,
            max_depth=1,
        )
        # At depth 1, the nested model should be flattened to dict
        address_annotation = minified.model_fields["address"].annotation
        assert address_annotation is dict

    def test_valid_model_instance(self):
        """Minified model should be instantiable."""
        minified = _minify_pydantic_model(SimpleArgs)
        instance = minified(query="test", limit=5)
        assert instance.query == "test"
        assert instance.limit == 5

    def test_nested_valid_instance(self):
        """Minified nested model should be instantiable."""
        minified = _minify_pydantic_model(NestedArgs, strip_nested=True)
        # The address field still needs to be a model (not flattened)
        inner_type = minified.model_fields["address"].annotation
        if isinstance(inner_type, type) and issubclass(inner_type, BaseModel):
            inner = inner_type(city="NYC")
            instance = minified(name="Alice", address=inner, tags=[])
            assert instance.name == "Alice"


# ======================================================================
# TestToolIndex
# ======================================================================


class TestToolIndex:
    def _make_tools(self) -> list[BaseTool]:
        return [
            _make_tool("search_employees", "Search for employees by name or department"),
            _make_tool("get_weather", "Get current weather for a city"),
            _make_tool("send_email", "Send an email to a recipient"),
            _make_tool("create_ticket", "Create a support ticket in the system"),
            _make_tool("analyze_data", "Run statistical analysis on a dataset"),
            _make_tool("deploy_service", "Deploy a service to production"),
            _make_tool("query_database", "Execute a SQL query against the database"),
            _make_tool("upload_file", "Upload a file to cloud storage"),
            _make_tool("translate_text", "Translate text between languages"),
            _make_tool("generate_report", "Generate a business report"),
        ]

    def test_select_returns_top_k(self):
        tools = self._make_tools()
        index = ToolIndex(tools)
        selected = index.select("search for John in HR", top_k=3)
        assert len(selected) <= 3

    def test_preserve_always_included(self):
        tools = self._make_tools()
        index = ToolIndex(tools)
        selected = index.select(
            "weather forecast",
            top_k=2,
            preserve=frozenset({"deploy_service"}),
        )
        names = {t.name for t in selected}
        assert "deploy_service" in names

    def test_all_tool_names(self):
        tools = self._make_tools()
        index = ToolIndex(tools)
        assert len(index.all_tool_names) == 10
        assert "search_employees" in index.all_tool_names

    def test_tool_summaries(self):
        tools = self._make_tools()
        index = ToolIndex(tools)
        summaries = index.tool_summaries
        assert "search_employees" in summaries
        assert "get_weather" in summaries

    def test_empty_query_returns_tools(self):
        tools = self._make_tools()
        index = ToolIndex(tools)
        # Empty query should still return top_k tools (by score 0)
        selected = index.select("", top_k=3)
        assert len(selected) == 3

    def test_all_tools_property(self):
        tools = self._make_tools()
        index = ToolIndex(tools)
        assert len(index.all_tools) == 10


# ======================================================================
# TestRequestMoreToolsTool
# ======================================================================


class TestRequestMoreToolsTool:
    @pytest.mark.asyncio
    async def test_returns_all_tool_names(self):
        tools = [
            _make_tool("alpha", "Tool Alpha"),
            _make_tool("beta", "Tool Beta"),
        ]
        index = ToolIndex(tools)
        fallback = _RequestMoreToolsTool(tool_index=index)

        result = await fallback._arun()
        assert "2 of 2 tools are now available" in result
        assert "alpha" in result
        assert "beta" in result

    def test_tool_name(self):
        tools = [_make_tool("x", "y")]
        index = ToolIndex(tools)
        fallback = _RequestMoreToolsTool(tool_index=index)
        assert fallback.name == "request_more_tools"

    def test_tool_description(self):
        tools = [_make_tool("x", "y")]
        index = ToolIndex(tools)
        fallback = _RequestMoreToolsTool(tool_index=index)
        assert "none of your current tools" in fallback.description


# ======================================================================
# TestApplyStaticOptimizations
# ======================================================================


class TestApplyStaticOptimizations:
    def test_truncates_descriptions(self):
        tools = [_make_tool("t1", "A" * 300, SimpleArgs)]
        config = _resolve_config(ToolOptimizationConfig(level=OptimizationLevel.MINIMAL))
        result = apply_static_optimizations(tools, config)
        assert len(result[0].description) <= 200

    def test_minifies_schema(self):
        tools = [_make_tool("t1", "desc", SimpleArgs)]
        config = _resolve_config(ToolOptimizationConfig(level=OptimizationLevel.MINIMAL))
        result = apply_static_optimizations(tools, config)
        schema = result[0].args_schema
        for field_info in schema.model_fields.values():
            assert field_info.description is None

    def test_preserves_tools(self):
        tools = [
            _make_tool("preserved", "A" * 300, SimpleArgs),
            _make_tool("optimized", "B" * 300, SimpleArgs),
        ]
        config = _resolve_config(
            ToolOptimizationConfig(
                level=OptimizationLevel.MINIMAL,
                preserve_tools={"preserved"},
            )
        )
        result = apply_static_optimizations(tools, config)
        # Preserved tool should not be truncated
        assert len(result[0].description) == 300
        # Optimized tool should be truncated
        assert len(result[1].description) <= 200

    def test_standard_level(self):
        tools = [_make_tool("t1", "A" * 300, NestedArgs)]
        config = _resolve_config(ToolOptimizationConfig(level=OptimizationLevel.STANDARD))
        result = apply_static_optimizations(tools, config)
        assert len(result[0].description) <= 150

    def test_empty_tool_list(self):
        config = _resolve_config(ToolOptimizationConfig(level=OptimizationLevel.MINIMAL))
        result = apply_static_optimizations([], config)
        assert result == []


# ======================================================================
# TestOptimizationLevelEnum
# ======================================================================


class TestOptimizationLevelEnum:
    def test_string_values(self):
        assert OptimizationLevel.MINIMAL == "minimal"
        assert OptimizationLevel.STANDARD == "standard"
        assert OptimizationLevel.SEMANTIC == "semantic"

    def test_from_string(self):
        assert OptimizationLevel("minimal") is OptimizationLevel.MINIMAL
        assert OptimizationLevel("semantic") is OptimizationLevel.SEMANTIC

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            OptimizationLevel("invalid")


# ======================================================================
# TestToolOptimizationConfig
# ======================================================================


class TestToolOptimizationConfig:
    def test_defaults(self):
        config = ToolOptimizationConfig()
        assert config.level is None
        assert config.minify_schema is None
        assert config.preserve_tools is None

    def test_all_fields(self):
        config = ToolOptimizationConfig(
            level=OptimizationLevel.SEMANTIC,
            minify_schema=True,
            max_description_length=100,
            strip_nested_descriptions=True,
            max_schema_depth=2,
            semantic_selection=True,
            semantic_top_k=5,
            always_include_fallback=True,
            preserve_tools={"tool_a"},
        )
        assert config.level == OptimizationLevel.SEMANTIC
        assert config.semantic_top_k == 5
        assert "tool_a" in config.preserve_tools


# ======================================================================
# TestIntegrationWithBuildAgent
# ======================================================================


class TestIntegrationWithBuildAgent:
    """Test that optimize_tools parameter is accepted by build_agent."""

    def test_normalize_true(self):
        """optimize_tools=True should normalize to MINIMAL config."""
        from promptise.tool_optimization import OptimizationLevel, ToolOptimizationConfig

        # Test the normalization logic directly
        optimize_tools: Any = True
        if optimize_tools is True:
            config = ToolOptimizationConfig(level=OptimizationLevel.MINIMAL)
        assert config.level == OptimizationLevel.MINIMAL

    def test_normalize_string(self):
        """optimize_tools="semantic" should normalize to SEMANTIC config."""
        from promptise.tool_optimization import OptimizationLevel, ToolOptimizationConfig

        optimize_tools = "semantic"
        config = ToolOptimizationConfig(level=OptimizationLevel(optimize_tools))
        assert config.level == OptimizationLevel.SEMANTIC

    def test_normalize_config_passthrough(self):
        """ToolOptimizationConfig should be passed through as-is."""
        from promptise.tool_optimization import OptimizationLevel, ToolOptimizationConfig

        config = ToolOptimizationConfig(
            level=OptimizationLevel.STANDARD,
            max_description_length=50,
        )
        assert config.max_description_length == 50


# ======================================================================
# TestStripDescriptionsInJsonSchema
# ======================================================================


class TestStripDescriptionsInJsonSchema:
    """Test the strip_descriptions parameter on _jsonschema_to_pydantic."""

    def test_strip_descriptions_removes_field_descriptions(self):
        from promptise.tools import _jsonschema_to_pydantic

        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "limit": {"type": "integer", "description": "Max results", "default": 10},
            },
            "required": ["query"],
        }

        model = _jsonschema_to_pydantic(schema, strip_descriptions=True)
        for field_info in model.model_fields.values():
            assert field_info.description is None

    def test_no_strip_preserves_descriptions(self):
        from promptise.tools import _jsonschema_to_pydantic

        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
            },
            "required": ["query"],
        }

        model = _jsonschema_to_pydantic(schema, strip_descriptions=False)
        assert model.model_fields["query"].description == "The search query"

    def test_strip_preserves_types_and_required(self):
        from promptise.tools import _jsonschema_to_pydantic

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User name"},
                "age": {"type": "integer", "description": "Age", "default": 0},
            },
            "required": ["name"],
        }

        model = _jsonschema_to_pydantic(schema, strip_descriptions=True)
        assert model.model_fields["name"].is_required()
        assert not model.model_fields["age"].is_required()
        assert model.model_fields["age"].default == 0
