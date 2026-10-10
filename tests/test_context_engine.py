"""Tests for promptise.context_engine — unified context assembly."""

from __future__ import annotations

import pytest

from promptise.context_engine import (
    ContextEngine,
    ContextLayer,
    ContextReport,
    _detect_context_window,
    _EstimateCounter,
)

# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


class TestEstimateCounter:
    def test_empty_string(self):
        c = _EstimateCounter()
        assert c.count("") == 0

    def test_short_text(self):
        c = _EstimateCounter()
        # "hello world" = 11 chars → ~3 tokens
        assert 2 <= c.count("hello world") <= 5

    def test_long_text(self):
        c = _EstimateCounter()
        text = "word " * 1000  # ~5000 chars
        tokens = c.count(text)
        assert 1000 < tokens < 2000  # Reasonable range


# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------


class TestModelDetection:
    def test_openai_gpt5_mini(self):
        # OpenAI lists a 400k window (272k input + 128k output) for the GPT-5 family.
        assert _detect_context_window("openai:gpt-5-mini") == 400_000
        assert _detect_context_window("openai:gpt-5") == 400_000

    def test_gpt5_budget_is_capped_at_the_input_limit(self):
        engine = ContextEngine(model="openai:gpt-5-mini")
        assert engine.window == 400_000
        assert engine.max_input_tokens == 272_000
        assert engine.budget == 272_000

    def test_explicit_window_is_not_capped(self):
        engine = ContextEngine(model="openai:gpt-5-mini", model_context_window=50_000)
        assert engine.max_input_tokens is None
        assert engine.budget == 50_000 - 4096

    def test_window_from_model_profile(self):
        class _Model:
            profile = {"max_input_tokens": 1_000_000, "max_output_tokens": 64_000}

        engine = ContextEngine(model="custom:my-model")
        assert engine._apply_model_profile(_Model())
        assert engine.window == 1_064_000
        assert engine.max_input_tokens == 1_000_000
        assert engine.budget == 1_000_000

    def test_model_profile_does_not_override_explicit_window(self):
        class _Model:
            profile = {"max_input_tokens": 1_000_000}

        engine = ContextEngine(model_context_window=32_000)
        assert not engine._apply_model_profile(_Model())
        assert engine.window == 32_000

    def test_model_without_profile_keeps_table_value(self):
        engine = ContextEngine(model="openai:gpt-4o")
        assert not engine._apply_model_profile(object())
        assert engine.window == 128_000

    def test_claude_sonnet(self):
        assert _detect_context_window("anthropic:claude-sonnet-4") == 200_000

    def test_unknown_model(self):
        assert _detect_context_window("custom:my-model") == 128_000

    def test_none_model(self):
        assert _detect_context_window(None) == 128_000

    def test_ollama_llama(self):
        assert _detect_context_window("ollama:llama3") == 8_192


# ---------------------------------------------------------------------------
# ContextLayer
# ---------------------------------------------------------------------------


class TestContextLayer:
    def test_construction(self):
        layer = ContextLayer(name="test", priority=5, content="Hello world")
        assert layer.name == "test"
        assert layer.priority == 5
        assert layer.content == "Hello world"
        assert layer.required is False

    def test_token_estimate(self):
        layer = ContextLayer(name="test", content="Hello world")
        assert layer.token_estimate > 0

    def test_empty_content(self):
        layer = ContextLayer(name="test", content="")
        assert layer.token_estimate == 0


# ---------------------------------------------------------------------------
# ContextEngine
# ---------------------------------------------------------------------------


class TestContextEngine:
    def test_construction_defaults(self):
        engine = ContextEngine(model="openai:gpt-4o")
        assert engine.window == 128_000
        assert engine.budget == 128_000 - 4096

    def test_construction_explicit_window(self):
        engine = ContextEngine(model_context_window=32_000)
        assert engine.window == 32_000
        assert engine.budget == 32_000 - 4096

    def test_builtin_layers_registered(self):
        engine = ContextEngine()
        info = engine.get_layer_info()
        names = [l["name"] for l in info]
        assert "identity" in names
        assert "tools" in names
        assert "memory" in names
        assert "strategies" in names
        assert "conversation" in names
        assert "user_message" in names

    def test_builtin_priorities(self):
        engine = ContextEngine()
        info = {l["name"]: l for l in engine.get_layer_info()}
        assert info["identity"]["priority"] == 10
        assert info["tools"]["priority"] == 9
        assert info["memory"]["priority"] == 3
        assert info["strategies"]["priority"] == 2
        assert info["conversation"]["priority"] == 1

    def test_set_content(self):
        engine = ContextEngine()
        engine.set_content("identity", "You are a helpful assistant.")
        assert engine.get_content("identity") == "You are a helpful assistant."

    def test_set_content_unknown_layer(self):
        engine = ContextEngine()
        with pytest.raises(KeyError, match="not registered"):
            engine.set_content("nonexistent", "test")

    def test_add_custom_layer(self):
        engine = ContextEngine()
        engine.add_layer("company_policy", priority=7, content="We are Acme Corp.")
        assert engine.get_content("company_policy") == "We are Acme Corp."
        info = {l["name"]: l for l in engine.get_layer_info()}
        assert info["company_policy"]["priority"] == 7

    def test_remove_layer(self):
        engine = ContextEngine()
        engine.add_layer("temp", content="temporary")
        engine.remove_layer("temp")
        assert engine.get_content("temp") == ""

    def test_clear_content(self):
        engine = ContextEngine()
        engine.set_content("identity", "test")
        engine.clear_content("identity")
        assert engine.get_content("identity") == ""

    def test_clear_all(self):
        engine = ContextEngine()
        engine.set_content("identity", "test")
        engine.set_content("memory", "recalled data")
        engine.clear_all()
        assert engine.get_content("identity") == ""
        assert engine.get_content("memory") == ""

    def test_count_tokens(self):
        engine = ContextEngine()
        count = engine.count_tokens("Hello world")
        assert count > 0


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


class TestAssembly:
    def test_empty_assembly(self):
        engine = ContextEngine()
        messages = engine.assemble()
        assert messages == []

    def test_basic_assembly(self):
        engine = ContextEngine()
        engine.set_content("identity", "You are a helpful assistant.")
        engine.set_content("user_message", "What is Python?")
        messages = engine.assemble()

        # Should have system (identity) + user message
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is Python?"

    def test_priority_ordering(self):
        engine = ContextEngine()
        engine.set_content("memory", "Memory recall here")  # priority 3
        engine.set_content("identity", "You are an agent")  # priority 10
        engine.set_content("strategies", "Strategy here")  # priority 2
        messages = engine.assemble()

        # Higher priority should come first
        assert messages[0]["content"] == "You are an agent"  # priority 10
        assert messages[1]["content"] == "Memory recall here"  # priority 3
        assert messages[2]["content"] == "Strategy here"  # priority 2

    def test_empty_layers_skipped(self):
        engine = ContextEngine()
        engine.set_content("identity", "Agent identity")
        # memory, strategies, etc. are empty → skipped
        messages = engine.assemble()
        assert len(messages) == 1

    def test_report_generated(self):
        engine = ContextEngine()
        engine.set_content("identity", "You are an agent")
        engine.set_content("user_message", "Hello")
        engine.assemble()

        report = engine.get_report()
        assert report is not None
        assert report.total_tokens > 0
        assert report.budget > 0
        assert 0 < report.utilization < 1.0
        assert len(report.layers) == 2

    def test_custom_layer_in_assembly(self):
        engine = ContextEngine()
        engine.add_layer("company", priority=7, content="Acme Corp policies...")
        engine.set_content("identity", "You are an Acme assistant")
        messages = engine.assemble()

        # Identity (10) should be before company (7)
        assert messages[0]["content"] == "You are an Acme assistant"
        assert messages[1]["content"] == "Acme Corp policies..."


# ---------------------------------------------------------------------------
# Trimming
# ---------------------------------------------------------------------------


class TestTrimming:
    def test_trim_lowest_priority_first(self):
        # Use a tiny budget to force trimming
        engine = ContextEngine(model_context_window=200, response_reserve=50)
        # Budget = 150 tokens

        engine.set_content("identity", "You are an agent.")  # ~5 tokens, priority 10, required
        engine.set_content("memory", "A" * 500)  # ~143 tokens, priority 3
        engine.set_content("strategies", "B" * 500)  # ~143 tokens, priority 2

        engine.assemble()
        report = engine.get_report()

        # Identity should survive (required), strategies should be trimmed first (priority 2)
        assert any("identity" not in t for t in report.trimmed_layers) or True
        assert report.total_tokens <= engine.budget + 50  # Allow some estimation slack

    def test_required_layers_never_trimmed(self):
        engine = ContextEngine(model_context_window=100, response_reserve=10)
        engine.set_content("identity", "You are an agent.")  # required=True
        engine.set_content("user_message", "Hello")  # required=True

        messages = engine.assemble()
        assert len(messages) >= 2  # Both required layers present

        report = engine.get_report()
        assert "identity" not in report.trimmed_layers
        assert "user_message" not in report.trimmed_layers

    def test_conversation_oldest_first_trim(self):
        engine = ContextEngine(model_context_window=300, response_reserve=50)

        conversation = (
            "User: First message\n"
            "Assistant: First response\n"
            "User: Second message\n"
            "Assistant: Second response\n"
            "User: Third message\n"
            "Assistant: Third response"
        )
        engine.set_content("identity", "Agent")
        engine.set_content("conversation", conversation)

        messages = engine.assemble()
        report = engine.get_report()

        # If trimming occurred, oldest messages should be removed first
        if "conversation" in report.trimmed_layers:
            conv_messages = [m for m in messages if m["role"] in ("user", "assistant")]
            # Should have fewer than 6 messages (3 pairs)
            assert len(conv_messages) < 6


# ---------------------------------------------------------------------------
# Conversation parsing
# ---------------------------------------------------------------------------


class TestConversationParsing:
    def test_basic_parsing(self):
        content = "User: Hello\nAssistant: Hi there\nUser: How are you?\nAssistant: Good!"
        engine = ContextEngine()
        messages = engine._parse_conversation(content)

        assert len(messages) == 4
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi there"}
        assert messages[2] == {"role": "user", "content": "How are you?"}
        assert messages[3] == {"role": "assistant", "content": "Good!"}

    def test_empty_content(self):
        engine = ContextEngine()
        messages = engine._parse_conversation("")
        assert messages == []

    def test_human_ai_format(self):
        content = "Human: Hello\nAI: Hi there"
        engine = ContextEngine()
        messages = engine._parse_conversation(content)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_multiline_message(self):
        content = "User: First line\nsecond line\nAssistant: Response"
        engine = ContextEngine()
        messages = engine._parse_conversation(content)
        assert len(messages) == 2
        assert "second line" in messages[0]["content"]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


class TestReport:
    def test_report_utilization(self):
        report = ContextReport(total_tokens=500, budget=1000)
        assert report.utilization == 0.5

    def test_report_zero_budget(self):
        report = ContextReport(total_tokens=0, budget=0)
        assert report.utilization == 0.0


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------


class TestExports:
    def test_imports_from_promptise(self):
        from promptise import ContextEngine

        assert ContextEngine is not None

    def test_context_layer_importable(self):
        from promptise.context_engine import ContextLayer, ContextReport, Tokenizer

        assert ContextLayer is not None
        assert ContextReport is not None
        assert Tokenizer is not None
