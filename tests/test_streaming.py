"""Tests for promptise.streaming — Streaming with Tool Visibility."""

from __future__ import annotations

import json

from promptise.streaming import (
    DoneEvent,
    ErrorEvent,
    StreamEvent,
    TokenEvent,
    ToolEndEvent,
    ToolStartEvent,
    content_text,
    tool_display_name,
    tool_error_summary,
    tool_summary,
)

# ---------------------------------------------------------------------------
# StreamEvent base
# ---------------------------------------------------------------------------


class TestStreamEvent:
    def test_construction(self):
        e = StreamEvent(type="test")
        assert e.type == "test"
        assert isinstance(e.timestamp, float)

    def test_to_dict(self):
        e = StreamEvent(type="test")
        d = e.to_dict()
        assert d["type"] == "test"
        assert "timestamp" in d

    def test_to_json(self):
        e = StreamEvent(type="test")
        j = e.to_json()
        parsed = json.loads(j)
        assert parsed["type"] == "test"


# ---------------------------------------------------------------------------
# ToolStartEvent
# ---------------------------------------------------------------------------


class TestToolStartEvent:
    def test_defaults(self):
        e = ToolStartEvent()
        assert e.type == "tool_start"
        assert e.tool_name == ""
        assert e.tool_display_name == ""
        assert e.arguments == {}
        assert e.tool_index == 0

    def test_full_construction(self):
        e = ToolStartEvent(
            tool_name="search_customers",
            tool_display_name="Searching customers",
            arguments={"query": "Alice"},
            tool_index=2,
        )
        assert e.tool_name == "search_customers"
        assert e.tool_display_name == "Searching customers"
        assert e.arguments == {"query": "Alice"}
        assert e.tool_index == 2

    def test_serialization(self):
        e = ToolStartEvent(tool_name="test", arguments={"a": 1})
        d = e.to_dict()
        assert d["type"] == "tool_start"
        assert d["tool_name"] == "test"
        assert d["arguments"] == {"a": 1}
        # JSON roundtrip
        j = json.loads(e.to_json())
        assert j["type"] == "tool_start"


# ---------------------------------------------------------------------------
# ToolEndEvent
# ---------------------------------------------------------------------------


class TestToolEndEvent:
    def test_defaults(self):
        e = ToolEndEvent()
        assert e.type == "tool_end"
        assert e.success is True

    def test_failure(self):
        e = ToolEndEvent(
            tool_name="send_email",
            tool_summary="Connection refused",
            success=False,
            duration_ms=1234.5,
        )
        assert e.success is False
        assert e.duration_ms == 1234.5


# ---------------------------------------------------------------------------
# TokenEvent
# ---------------------------------------------------------------------------


class TestTokenEvent:
    def test_construction(self):
        e = TokenEvent(text="Hello", cumulative_text="Hello")
        assert e.type == "token"
        assert e.text == "Hello"
        assert e.cumulative_text == "Hello"

    def test_accumulation(self):
        TokenEvent(text="He", cumulative_text="He")
        e2 = TokenEvent(text="llo", cumulative_text="Hello")
        assert e2.cumulative_text == "Hello"


# ---------------------------------------------------------------------------
# DoneEvent
# ---------------------------------------------------------------------------


class TestDoneEvent:
    def test_defaults(self):
        e = DoneEvent()
        assert e.type == "done"
        assert e.full_response == ""
        assert e.tool_calls == []
        assert e.cache_hit is False

    def test_full(self):
        e = DoneEvent(
            full_response="Here is your answer.",
            tool_calls=[{"name": "search", "summary": "Found 3 results"}],
            duration_ms=2500,
            cache_hit=True,
        )
        assert e.cache_hit is True
        assert len(e.tool_calls) == 1


# ---------------------------------------------------------------------------
# ErrorEvent
# ---------------------------------------------------------------------------


class TestErrorEvent:
    def test_defaults(self):
        e = ErrorEvent()
        assert e.type == "error"
        assert e.message == ""
        assert e.recoverable is False

    def test_construction(self):
        e = ErrorEvent(message="Input blocked by safety policy.", recoverable=False)
        assert "blocked" in e.message


# ---------------------------------------------------------------------------
# tool_display_name
# ---------------------------------------------------------------------------


class TestToolDisplayName:
    def test_search_verb(self):
        assert tool_display_name("search_customers") == "Searching customers"

    def test_get_verb(self):
        assert tool_display_name("get_order_status") == "Getting order status"

    def test_create_verb(self):
        assert tool_display_name("create_ticket") == "Creating ticket"

    def test_delete_verb(self):
        assert tool_display_name("delete_user") == "Deleting user"

    def test_list_verb(self):
        assert tool_display_name("list_all_items") == "Listing all items"

    def test_send_verb(self):
        assert tool_display_name("send_email") == "Sending email"

    def test_deploy_verb(self):
        assert tool_display_name("deploy_service") == "Deploying service"

    def test_namespace_prefix_skipped(self):
        """Namespace prefix (hr_, finance_) should be skipped to find the verb."""
        assert tool_display_name("hr_list_employees") == "Listing employees"
        assert tool_display_name("finance_create_invoice") == "Creating invoice"

    def test_unknown_verb_capitalizes(self):
        assert tool_display_name("foobar_baz") == "Foobar Baz"

    def test_single_word(self):
        assert tool_display_name("search") == "Searching"

    def test_developer_override(self):
        overrides = {"my_tool": "Doing something special"}
        assert tool_display_name("my_tool", overrides) == "Doing something special"

    def test_override_takes_priority(self):
        overrides = {"search_db": "Looking up records"}
        assert tool_display_name("search_db", overrides) == "Looking up records"

    def test_empty_name(self):
        assert tool_display_name("") == ""

    def test_hyphens(self):
        assert tool_display_name("get-user-info") == "Getting user info"


# ---------------------------------------------------------------------------
# tool_summary
# ---------------------------------------------------------------------------


class TestToolSummary:
    def test_json_dict_short(self):
        result = json.dumps({"name": "Alice", "status": "active"})
        summary = tool_summary(result)
        assert "Alice" in summary
        assert "active" in summary

    def test_json_dict_long(self):
        data = {f"key_{i}": f"value_{i}" for i in range(10)}
        result = json.dumps(data)
        summary = tool_summary(result)
        assert "+7 more" in summary

    def test_json_list(self):
        result = json.dumps([1, 2, 3, 4, 5])
        assert tool_summary(result) == "Found 5 result(s)"

    def test_json_empty_list(self):
        assert tool_summary("[]") == "Found 0 result(s)"

    def test_plain_text_short(self):
        assert tool_summary("Hello world") == "Hello world"

    def test_plain_text_long(self):
        text = "A" * 200
        summary = tool_summary(text, max_length=50)
        assert len(summary) <= 50
        assert summary.endswith("...")

    def test_empty_result(self):
        assert tool_summary("") == "Done"
        assert tool_summary(None) == "Done"

    def test_whitespace_only(self):
        assert tool_summary("   ") == "Done"


# ---------------------------------------------------------------------------
# tool_error_summary / content_text
# ---------------------------------------------------------------------------


class TestToolErrorSummary:
    def test_mcp_tool_error_message(self):
        text = json.dumps(
            {"error": {"code": "TOOL_ERROR", "message": "No order A-9.", "retryable": False}},
            indent=2,
        )
        assert tool_error_summary(text) == "No order A-9."

    def test_long_message_truncated(self):
        text = json.dumps({"error": {"message": "x" * 300}})
        assert len(tool_error_summary(text)) == 120

    def test_plain_text(self):
        assert tool_error_summary("Error: Unknown tool 'x'") == "Error: Unknown tool 'x'"

    def test_json_without_message(self):
        assert tool_error_summary('{"error": "boom"}') == "error: boom"

    def test_empty(self):
        assert tool_error_summary("") == "Tool call failed"
        assert tool_error_summary(None) == "Tool call failed"


class TestContentText:
    def test_string(self):
        assert content_text("hi") == "hi"

    def test_content_blocks(self):
        blocks = [
            {"type": "text", "text": "Hello "},
            {"type": "tool_use", "id": "t1", "input": {}},
            {"type": "text", "text": "world"},
        ]
        assert content_text(blocks) == "Hello world"

    def test_none_and_other(self):
        assert content_text(None) == ""
        assert content_text(42) == ""


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_imports_from_promptise(self):
        from promptise import (
            DoneEvent,
            StreamEvent,
            ToolStartEvent,
        )

        assert StreamEvent is not None
        assert ToolStartEvent is not None
        assert DoneEvent is not None
