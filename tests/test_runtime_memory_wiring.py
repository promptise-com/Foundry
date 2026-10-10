"""Tests for runtime memory wiring and conversation buffer.

Tests that:
- ConversationBuffer correctly manages messages
- Memory providers are created from ContextConfig
- AgentProcess wires memory into AgentContext and build_agent
- Conversation buffer is used in _invoke_agent
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from promptise.runtime.config import ContextConfig, ProcessConfig
from promptise.runtime.conversation import ConversationBuffer
from promptise.runtime.lifecycle import ProcessState
from promptise.runtime.process import AgentProcess, _create_memory_provider

# ===========================================================================
# ConversationBuffer tests
# ===========================================================================


class TestConversationBuffer:
    """Tests for ConversationBuffer."""

    def test_creation_defaults(self):
        buf = ConversationBuffer()
        assert buf.max_messages == 100
        assert len(buf) == 0
        assert buf.get_messages() == []

    def test_creation_custom_max(self):
        buf = ConversationBuffer(max_messages=10)
        assert buf.max_messages == 10

    def test_append_single(self):
        buf = ConversationBuffer()
        msg = {"role": "user", "content": "hello"}
        buf.append(msg)
        assert len(buf) == 1
        assert buf.get_messages() == [msg]

    def test_append_multiple(self):
        buf = ConversationBuffer()
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        for m in msgs:
            buf.append(m)
        assert len(buf) == 2
        assert buf.get_messages() == msgs

    def test_extend(self):
        buf = ConversationBuffer()
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        buf.extend(msgs)
        assert len(buf) == 2
        assert buf.get_messages() == msgs

    def test_eviction_on_append(self):
        buf = ConversationBuffer(max_messages=3)
        for i in range(5):
            buf.append({"role": "user", "content": f"msg-{i}"})
        assert len(buf) == 3
        # Should keep the last 3
        msgs = buf.get_messages()
        assert msgs[0]["content"] == "msg-2"
        assert msgs[1]["content"] == "msg-3"
        assert msgs[2]["content"] == "msg-4"

    def test_eviction_on_extend(self):
        buf = ConversationBuffer(max_messages=2)
        buf.extend(
            [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "user", "content": "c"},
            ]
        )
        assert len(buf) == 2
        msgs = buf.get_messages()
        assert msgs[0]["content"] == "b"
        assert msgs[1]["content"] == "c"

    def test_clear(self):
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": "hello"})
        assert len(buf) == 1
        buf.clear()
        assert len(buf) == 0
        assert buf.get_messages() == []

    def test_set_messages(self):
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": "old"})
        new_msgs = [
            {"role": "user", "content": "new1"},
            {"role": "assistant", "content": "new2"},
        ]
        buf.set_messages(new_msgs)
        assert len(buf) == 2
        assert buf.get_messages() == new_msgs

    def test_set_messages_with_eviction(self):
        buf = ConversationBuffer(max_messages=2)
        buf.set_messages(
            [
                {"role": "user", "content": "a"},
                {"role": "user", "content": "b"},
                {"role": "user", "content": "c"},
            ]
        )
        assert len(buf) == 2
        msgs = buf.get_messages()
        assert msgs[0]["content"] == "b"

    def test_get_messages_returns_copy(self):
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": "hello"})
        msgs1 = buf.get_messages()
        msgs2 = buf.get_messages()
        assert msgs1 == msgs2
        assert msgs1 is not msgs2  # Different list objects

    def test_to_dict(self):
        buf = ConversationBuffer(max_messages=50)
        buf.append({"role": "user", "content": "hello"})
        d = buf.to_dict()
        assert d["max_messages"] == 50
        assert len(d["messages"]) == 1
        assert d["messages"][0]["content"] == "hello"

    def test_from_dict(self):
        d = {
            "max_messages": 50,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ],
        }
        buf = ConversationBuffer.from_dict(d)
        assert buf.max_messages == 50
        assert len(buf) == 2
        assert buf.get_messages()[0]["content"] == "hello"

    def test_from_dict_defaults(self):
        buf = ConversationBuffer.from_dict({})
        assert buf.max_messages == 100
        assert len(buf) == 0

    def test_roundtrip(self):
        buf1 = ConversationBuffer(max_messages=30)
        buf1.append({"role": "user", "content": "test"})
        buf1.append({"role": "assistant", "content": "response"})

        d = buf1.to_dict()
        buf2 = ConversationBuffer.from_dict(d)

        assert buf2.max_messages == buf1.max_messages
        assert buf2.get_messages() == buf1.get_messages()

    def test_repr(self):
        buf = ConversationBuffer(max_messages=50)
        buf.append({"role": "user", "content": "hello"})
        r = repr(buf)
        assert "messages=1" in r
        assert "max=50" in r

    def test_zero_max_messages_disabled(self):
        """When max_messages=0, buffer is effectively disabled."""
        buf = ConversationBuffer(max_messages=0)
        buf.append({"role": "user", "content": "hello"})
        # 0 means no eviction (infinite buffer for this edge case)
        assert len(buf) == 1


# ===========================================================================
# Memory provider factory tests
# ===========================================================================


class TestCreateMemoryProvider:
    """Tests for _create_memory_provider factory."""

    def test_none_when_no_provider(self):
        config = ContextConfig()
        result = _create_memory_provider(config)
        assert result is None

    def test_in_memory_provider(self):
        config = ContextConfig(memory_provider="in_memory")
        result = _create_memory_provider(config)
        assert result is not None
        from promptise.memory import InMemoryProvider

        assert isinstance(result, InMemoryProvider)

    def test_unknown_provider_returns_none(self):
        config = ContextConfig()
        # Manually set to trigger the warning path
        config.memory_provider = "unknown"  # type: ignore
        result = _create_memory_provider(config)
        assert result is None


# ===========================================================================
# AgentProcess memory wiring tests
# ===========================================================================


class TestAgentProcessMemoryWiring:
    """Tests for memory wiring in AgentProcess."""

    def test_no_memory_by_default(self):
        process = AgentProcess(
            name="test",
            config=ProcessConfig(),
        )
        assert process._long_term_memory is None
        assert process._context.memory is None
        assert len(process._conversation_buffer) == 0

    def test_memory_wired_when_configured(self):
        process = AgentProcess(
            name="test",
            config=ProcessConfig(
                context=ContextConfig(memory_provider="in_memory"),
            ),
        )
        assert process._long_term_memory is not None
        assert process._context.memory is not None
        assert process._context.memory is process._long_term_memory

    def test_conversation_buffer_created(self):
        process = AgentProcess(
            name="test",
            config=ProcessConfig(
                context=ContextConfig(conversation_max_messages=50),
            ),
        )
        assert process._conversation_buffer.max_messages == 50

    def test_conversation_buffer_default(self):
        process = AgentProcess(
            name="test",
            config=ProcessConfig(),
        )
        assert process._conversation_buffer.max_messages == 100

    def test_dynamic_state_initialized(self):
        process = AgentProcess(
            name="test",
            config=ProcessConfig(),
        )
        assert process._dynamic_instructions is None
        assert process._dynamic_servers == {}
        assert process._custom_tools == []
        assert process._dynamic_triggers == []
        assert process._rebuild_count == 0


class TestAgentProcessStatus:
    """Tests for enhanced status() method."""

    def test_status_includes_new_fields(self):
        process = AgentProcess(
            name="test",
            config=ProcessConfig(),
        )
        status = process.status()
        assert "execution_mode" in status
        assert status["execution_mode"] == "strict"
        assert "dynamic_trigger_count" in status
        assert status["dynamic_trigger_count"] == 0
        assert "custom_tool_count" in status
        assert status["custom_tool_count"] == 0
        assert "rebuild_count" in status
        assert status["rebuild_count"] == 0
        assert "conversation_messages" in status
        assert status["conversation_messages"] == 0
        assert "has_memory" in status
        assert status["has_memory"] is False

    def test_status_with_memory(self):
        process = AgentProcess(
            name="test",
            config=ProcessConfig(
                context=ContextConfig(memory_provider="in_memory"),
            ),
        )
        status = process.status()
        assert status["has_memory"] is True

    def test_status_open_mode(self):
        from promptise.runtime.config import ExecutionMode

        process = AgentProcess(
            name="test",
            config=ProcessConfig(execution_mode=ExecutionMode.OPEN),
        )
        status = process.status()
        assert status["execution_mode"] == "open"


class TestAgentProcessInvokeWithConversation:
    """Tests for _invoke_agent using conversation buffer."""

    @pytest.mark.asyncio
    async def test_invoke_appends_to_conversation(self):
        """Verify that _invoke_agent appends messages to the buffer."""
        process = AgentProcess(
            name="test",
            config=ProcessConfig(),
        )

        # Mock agent
        mock_result = {"messages": [{"role": "assistant", "content": "I processed the event."}]}
        process._agent = AsyncMock()
        process._agent.ainvoke = AsyncMock(return_value=mock_result)

        from promptise.runtime.triggers.base import TriggerEvent

        event = TriggerEvent(
            trigger_id="t1",
            trigger_type="cron",
            payload={"message": "tick"},
        )

        await process._invoke_agent(event)

        assert len(process._conversation_buffer) == 2  # user + assistant
        msgs = process._conversation_buffer.get_messages()
        assert msgs[0]["role"] == "user"
        assert "tick" in msgs[0]["content"]
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "I processed the event."

    @pytest.mark.asyncio
    async def test_invoke_sends_conversation_history(self):
        """Verify that previous conversation is included in messages."""
        process = AgentProcess(
            name="test",
            config=ProcessConfig(),
        )

        # Pre-fill conversation buffer
        process._conversation_buffer.append({"role": "user", "content": "previous question"})
        process._conversation_buffer.append({"role": "assistant", "content": "previous answer"})

        process._agent = AsyncMock()
        process._agent.ainvoke = AsyncMock(return_value={"messages": []})

        from promptise.runtime.triggers.base import TriggerEvent

        event = TriggerEvent(
            trigger_id="t1",
            trigger_type="manual",
            payload={"data": "new"},
        )

        await process._invoke_agent(event)

        # Check what was sent to agent
        call_args = process._agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]

        # Should have: prev_user, prev_assistant, new_user
        assert len(messages) >= 3
        assert messages[0]["content"] == "previous question"
        assert messages[1]["content"] == "previous answer"
        assert "new" in messages[2]["content"]

    @pytest.mark.asyncio
    async def test_invoke_includes_context_state(self):
        """Verify context state is injected as system message."""
        process = AgentProcess(
            name="test",
            config=ProcessConfig(),
        )
        process._context.put("counter", 42, source="system")

        process._agent = AsyncMock()
        process._agent.ainvoke = AsyncMock(return_value={"messages": []})

        from promptise.runtime.triggers.base import TriggerEvent

        event = TriggerEvent(
            trigger_id="t1",
            trigger_type="manual",
            payload={},
        )

        await process._invoke_agent(event)

        call_args = process._agent.ainvoke.call_args
        messages = call_args[0][0]["messages"]
        # First message should be system with context state
        assert messages[0]["role"] == "system"
        assert "counter" in messages[0]["content"]


class TestAgentProcessStopCleansUp:
    """Tests for cleanup during stop()."""

    @pytest.mark.asyncio
    async def test_stop_clears_conversation(self):
        process = AgentProcess(
            name="test",
            config=ProcessConfig(),
        )
        process._conversation_buffer.append({"role": "user", "content": "hello"})
        assert len(process._conversation_buffer) == 1

        # Transition to a stoppable state first
        await process._lifecycle.transition(ProcessState.STARTING, reason="test")
        await process._lifecycle.transition(ProcessState.RUNNING, reason="test")

        await process.stop()
        assert len(process._conversation_buffer) == 0


# ===========================================================================
# ContextConfig memory fields tests
# ===========================================================================


class TestContextConfigMemoryFields:
    """Tests for new memory fields on ContextConfig."""

    def test_defaults(self):
        config = ContextConfig()
        assert config.memory_provider is None
        assert config.memory_max == 5
        assert config.memory_min_score == 0.0
        assert config.memory_auto_store is False
        assert config.memory_collection == "agent_memory"
        assert config.memory_persist_directory is None
        assert config.memory_user_id == "default"
        assert config.conversation_max_messages == 100

    def test_custom_values(self):
        config = ContextConfig(
            memory_provider="chroma",
            memory_auto_store=True,
            memory_collection="custom",
            memory_persist_directory="/data/chroma",
            memory_user_id="user-42",
            conversation_max_messages=200,
        )
        assert config.memory_provider == "chroma"
        assert config.memory_auto_store is True
        assert config.memory_collection == "custom"
        assert config.memory_persist_directory == "/data/chroma"
        assert config.memory_user_id == "user-42"
        assert config.conversation_max_messages == 200

    def test_serialization_roundtrip(self):
        config = ContextConfig(
            memory_provider="in_memory",
            memory_auto_store=True,
            conversation_max_messages=50,
        )
        d = config.model_dump()
        restored = ContextConfig.model_validate(d)
        assert restored.memory_provider == "in_memory"
        assert restored.memory_auto_store is True
        assert restored.conversation_max_messages == 50


# ===========================================================================
# build_agent extra_tools parameter tests
# ===========================================================================


class TestBuildDeepAgentExtraTools:
    """Tests for extra_tools parameter on build_agent."""

    @pytest.mark.asyncio
    async def test_extra_tools_parameter_accepted(self):
        """Verify build_agent accepts extra_tools without error."""
        import inspect

        from promptise.agent import build_agent

        sig = inspect.signature(build_agent)
        assert "extra_tools" in sig.parameters
        param = sig.parameters["extra_tools"]
        assert param.default is None


# ---------------------------------------------------------------------------
# Memory settings reach build_agent()
# ---------------------------------------------------------------------------


class TestMemorySettingsReachBuildAgent:
    """``ContextConfig.memory_max`` / ``memory_min_score`` / ``memory_timeout``
    are passed to ``build_agent()`` (they used to be accepted and ignored)."""

    @pytest.mark.asyncio
    async def test_process_passes_memory_settings(self) -> None:
        from unittest.mock import patch

        config = ProcessConfig(
            model="openai:gpt-5-mini",
            context=ContextConfig(
                memory_provider="in_memory",
                memory_max=3,
                memory_min_score=0.4,
                memory_timeout=20.0,
            ),
        )
        process = AgentProcess(name="mem", config=config)
        with patch("promptise.agent.build_agent", new=AsyncMock()) as mock_build:
            await process._build_agent()
        kwargs = mock_build.await_args.kwargs
        assert kwargs["memory_max_results"] == 3
        assert kwargs["memory_min_score"] == 0.4
        assert kwargs["memory_timeout"] == 20.0

    def test_manifest_memory_timeout(self) -> None:
        from promptise.runtime.manifest import AgentManifestSchema, manifest_to_process_config

        manifest = AgentManifestSchema(
            name="m",
            memory={"provider": "in_memory", "max": 2, "min_score": 0.5, "timeout": 12},
        )
        ctx = manifest_to_process_config(manifest).context
        assert (ctx.memory_max, ctx.memory_min_score, ctx.memory_timeout) == (2, 0.5, 12)
