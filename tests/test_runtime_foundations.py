"""Tests for runtime Phase 1: exceptions, lifecycle, config, context."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from promptise.runtime import (
    DEFAULT_DEVELOPMENT_CONFIG,
    DEFAULT_PRODUCTION_CONFIG,
    VALID_TRANSITIONS,
    AgentContext,
    DistributedConfig,
    JournalConfig,
    JournalError,
    ManifestError,
    ManifestValidationError,
    ProcessConfig,
    ProcessLifecycle,
    ProcessState,
    ProcessStateError,
    ProcessTransition,
    RuntimeBaseError,
    RuntimeConfig,
    StateEntry,
    StateError,
    TriggerConfig,
    TriggerError,
)

# =========================================================================
# Exceptions
# =========================================================================


class TestExceptions:
    """Verify exception hierarchy and attribute storage."""

    def test_runtime_base_error_is_runtime_error(self) -> None:
        assert issubclass(RuntimeBaseError, RuntimeError)

    def test_process_state_error_attributes(self) -> None:
        err = ProcessStateError("proc-1", "running", "created")
        assert err.process_id == "proc-1"
        assert err.current_state == "running"
        assert err.attempted_state == "created"
        assert "proc-1" in str(err)
        assert "running" in str(err)
        assert "created" in str(err)

    def test_manifest_validation_error_with_details(self) -> None:
        err = ManifestValidationError(
            "bad manifest",
            errors=[{"loc": ("name",), "msg": "required"}],
            file_path="/tmp/bad.agent",
        )
        assert err.errors == [{"loc": ("name",), "msg": "required"}]
        assert err.file_path == "/tmp/bad.agent"

    def test_all_errors_are_runtime_base(self) -> None:
        for cls in (
            ProcessStateError,
            ManifestError,
            ManifestValidationError,
            TriggerError,
            JournalError,
        ):
            assert issubclass(cls, RuntimeBaseError)


# =========================================================================
# ProcessState
# =========================================================================


class TestProcessState:
    """Verify enum values and string representations."""

    def test_all_states_exist(self) -> None:
        expected = {
            "created",
            "starting",
            "running",
            "suspended",
            "stopping",
            "stopped",
            "failed",
            "awaiting",
        }
        assert {s.value for s in ProcessState} == expected

    def test_string_enum(self) -> None:
        assert ProcessState.RUNNING == "running"
        assert str(ProcessState.RUNNING) == "ProcessState.RUNNING"

    def test_from_value(self) -> None:
        assert ProcessState("created") is ProcessState.CREATED

    def test_invalid_value_raises(self) -> None:
        with pytest.raises(ValueError):
            ProcessState("nonexistent")


# =========================================================================
# VALID_TRANSITIONS
# =========================================================================


class TestValidTransitions:
    """Verify the transition table is structurally sound."""

    def test_all_states_have_transitions(self) -> None:
        for state in ProcessState:
            assert state in VALID_TRANSITIONS, f"Missing transitions for {state}"

    def test_stopped_can_restart(self) -> None:
        assert ProcessState.STARTING in VALID_TRANSITIONS[ProcessState.STOPPED]

    def test_failed_can_restart(self) -> None:
        assert ProcessState.STARTING in VALID_TRANSITIONS[ProcessState.FAILED]

    def test_running_can_suspend(self) -> None:
        assert ProcessState.SUSPENDED in VALID_TRANSITIONS[ProcessState.RUNNING]

    def test_running_can_await(self) -> None:
        assert ProcessState.AWAITING in VALID_TRANSITIONS[ProcessState.RUNNING]

    def test_no_self_transitions(self) -> None:
        for state, targets in VALID_TRANSITIONS.items():
            assert state not in targets, f"{state} allows self-transition"


# =========================================================================
# ProcessTransition
# =========================================================================


class TestProcessTransition:
    """Verify transition dataclass serialization."""

    def test_create(self) -> None:
        t = ProcessTransition(
            from_state=ProcessState.CREATED,
            to_state=ProcessState.STARTING,
            reason="user requested",
        )
        assert t.from_state == ProcessState.CREATED
        assert t.to_state == ProcessState.STARTING
        assert t.reason == "user requested"

    def test_to_dict_from_dict_roundtrip(self) -> None:
        t = ProcessTransition(
            from_state=ProcessState.RUNNING,
            to_state=ProcessState.SUSPENDED,
            reason="idle",
            metadata={"idle_seconds": 300},
        )
        data = t.to_dict()
        restored = ProcessTransition.from_dict(data)
        assert restored.from_state == t.from_state
        assert restored.to_state == t.to_state
        assert restored.reason == t.reason
        assert restored.metadata == t.metadata
        assert restored.timestamp.isoformat() == t.timestamp.isoformat()


# =========================================================================
# ProcessLifecycle
# =========================================================================


class TestProcessLifecycle:
    """Verify the state machine."""

    @pytest.fixture()
    def lc(self) -> ProcessLifecycle:
        return ProcessLifecycle()

    async def test_initial_state_is_created(self, lc: ProcessLifecycle) -> None:
        assert lc.state == ProcessState.CREATED

    async def test_valid_transition(self, lc: ProcessLifecycle) -> None:
        t = await lc.transition(ProcessState.STARTING, reason="start")
        assert lc.state == ProcessState.STARTING
        assert t.from_state == ProcessState.CREATED
        assert t.to_state == ProcessState.STARTING
        assert t.reason == "start"

    async def test_invalid_transition_raises(self, lc: ProcessLifecycle) -> None:
        with pytest.raises(StateError) as exc_info:
            await lc.transition(ProcessState.RUNNING)
        assert exc_info.value.current == ProcessState.CREATED
        assert exc_info.value.target == ProcessState.RUNNING

    async def test_history_tracks_transitions(self, lc: ProcessLifecycle) -> None:
        await lc.transition(ProcessState.STARTING)
        await lc.transition(ProcessState.RUNNING)
        assert len(lc.history) == 2
        assert lc.history[0].to_state == ProcessState.STARTING
        assert lc.history[1].to_state == ProcessState.RUNNING

    async def test_can_transition(self, lc: ProcessLifecycle) -> None:
        assert lc.can_transition(ProcessState.STARTING) is True
        assert lc.can_transition(ProcessState.RUNNING) is False

    async def test_full_lifecycle(self, lc: ProcessLifecycle) -> None:
        await lc.transition(ProcessState.STARTING)
        await lc.transition(ProcessState.RUNNING)
        await lc.transition(ProcessState.STOPPING)
        await lc.transition(ProcessState.STOPPED)
        assert lc.state == ProcessState.STOPPED
        assert len(lc.history) == 4

    async def test_restart_from_stopped(self, lc: ProcessLifecycle) -> None:
        await lc.transition(ProcessState.STARTING)
        await lc.transition(ProcessState.RUNNING)
        await lc.transition(ProcessState.STOPPING)
        await lc.transition(ProcessState.STOPPED)
        # Restart
        await lc.transition(ProcessState.STARTING, reason="restart")
        assert lc.state == ProcessState.STARTING

    async def test_restart_from_failed(self, lc: ProcessLifecycle) -> None:
        await lc.transition(ProcessState.STARTING)
        await lc.transition(ProcessState.FAILED, reason="crash")
        await lc.transition(ProcessState.STARTING, reason="retry")
        assert lc.state == ProcessState.STARTING

    async def test_suspend_resume(self, lc: ProcessLifecycle) -> None:
        await lc.transition(ProcessState.STARTING)
        await lc.transition(ProcessState.RUNNING)
        await lc.transition(ProcessState.SUSPENDED)
        await lc.transition(ProcessState.RUNNING, reason="resume")
        assert lc.state == ProcessState.RUNNING

    async def test_awaiting_to_running(self, lc: ProcessLifecycle) -> None:
        await lc.transition(ProcessState.STARTING)
        await lc.transition(ProcessState.RUNNING)
        await lc.transition(ProcessState.AWAITING, reason="waiting for trigger")
        await lc.transition(ProcessState.RUNNING, reason="trigger fired")
        assert lc.state == ProcessState.RUNNING

    async def test_snapshot_and_restore(self, lc: ProcessLifecycle) -> None:
        await lc.transition(ProcessState.STARTING)
        await lc.transition(ProcessState.RUNNING)
        snap = lc.snapshot()
        assert snap["state"] == "running"
        assert len(snap["history"]) == 2

        restored = ProcessLifecycle.from_snapshot(snap)
        assert restored.state == ProcessState.RUNNING
        assert len(restored.history) == 2

    def test_repr(self, lc: ProcessLifecycle) -> None:
        assert "created" in repr(lc)
        assert "transitions=0" in repr(lc)


# =========================================================================
# TriggerConfig
# =========================================================================


class TestTriggerConfig:
    """Verify trigger config validation."""

    def test_cron_valid(self) -> None:
        cfg = TriggerConfig(type="cron", cron_expression="*/5 * * * *")
        assert cfg.cron_expression == "*/5 * * * *"

    def test_cron_missing_expression_raises(self) -> None:
        with pytest.raises(ValidationError):
            TriggerConfig(type="cron")

    def test_webhook_defaults(self) -> None:
        cfg = TriggerConfig(type="webhook")
        assert cfg.webhook_path == "/webhook"
        assert cfg.webhook_port == 9090

    def test_file_watch_valid(self) -> None:
        cfg = TriggerConfig(
            type="file_watch",
            watch_path="/data",
            watch_patterns=["*.csv"],
        )
        assert cfg.watch_path == "/data"

    def test_file_watch_missing_path_raises(self) -> None:
        with pytest.raises(ValidationError):
            TriggerConfig(type="file_watch")

    def test_event_valid(self) -> None:
        cfg = TriggerConfig(type="event", event_type="task.completed")
        assert cfg.event_type == "task.completed"

    def test_event_missing_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            TriggerConfig(type="event")

    def test_message_valid(self) -> None:
        cfg = TriggerConfig(type="message", topic="reports.*")
        assert cfg.topic == "reports.*"

    def test_message_missing_topic_raises(self) -> None:
        with pytest.raises(ValidationError):
            TriggerConfig(type="message")

    def test_custom_type_accepted(self) -> None:
        # TriggerConfig.type is now str to support custom trigger types.
        # Validation is deferred to the trigger factory at creation time.
        cfg = TriggerConfig(type="my_custom_trigger")
        assert cfg.type == "my_custom_trigger"

    def test_filter_expression_optional(self) -> None:
        cfg = TriggerConfig(
            type="cron",
            cron_expression="0 9 * * *",
            filter_expression="data_changed",
        )
        assert cfg.filter_expression == "data_changed"


# =========================================================================
# ProcessConfig
# =========================================================================


class TestProcessConfig:
    """Verify composite process config."""

    def test_defaults(self) -> None:
        cfg = ProcessConfig()
        assert cfg.model == "openai:gpt-5-mini"
        assert cfg.concurrency == 1
        assert cfg.heartbeat_interval == 10.0
        assert cfg.max_consecutive_failures == 3
        assert cfg.restart_policy == "never"

    def test_with_triggers(self) -> None:
        cfg = ProcessConfig(
            triggers=[
                TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
                TriggerConfig(type="webhook"),
            ]
        )
        assert len(cfg.triggers) == 2

    def test_nested_configs_have_defaults(self) -> None:
        cfg = ProcessConfig()
        # Journaling is opt-in per process; a JournalConfig itself
        # defaults to checkpoint level.
        assert cfg.journal.level == "none"
        assert JournalConfig().level == "checkpoint"
        assert cfg.context.env_prefix == "AGENT_"


# =========================================================================
# RuntimeConfig
# =========================================================================


class TestRuntimeConfig:
    """Verify top-level runtime config + presets."""

    def test_defaults(self) -> None:
        cfg = RuntimeConfig()
        assert len(cfg.processes) == 0
        assert cfg.distributed.enabled is False

    def test_to_dict_from_dict_roundtrip(self) -> None:
        cfg = RuntimeConfig(
            processes={
                "watcher": ProcessConfig(model="openai:gpt-4o"),
            },
            distributed=DistributedConfig(enabled=True, transport_port=9200),
        )
        data = cfg.to_dict()
        restored = RuntimeConfig.from_dict(data)
        assert restored.processes["watcher"].model == "openai:gpt-4o"
        assert restored.distributed.enabled is True
        assert restored.distributed.transport_port == 9200

    def test_development_preset(self) -> None:
        cfg = DEFAULT_DEVELOPMENT_CONFIG
        assert cfg.distributed.enabled is False

    def test_production_preset(self) -> None:
        cfg = DEFAULT_PRODUCTION_CONFIG
        assert cfg.distributed.enabled is True


# =========================================================================
# StateEntry
# =========================================================================


class TestStateEntry:
    """Verify state entry serialization."""

    def test_create(self) -> None:
        entry = StateEntry(key="counter", value=42, source="agent")
        assert entry.key == "counter"
        assert entry.value == 42
        assert entry.source == "agent"
        assert entry.timestamp > 0

    def test_to_dict_from_dict_roundtrip(self) -> None:
        entry = StateEntry(key="k", value={"nested": True}, source="trigger")
        data = entry.to_dict()
        restored = StateEntry.from_dict(data)
        assert restored.key == entry.key
        assert restored.value == entry.value
        assert restored.source == entry.source


# =========================================================================
# AgentContext — State
# =========================================================================


class TestAgentContextState:
    """Verify state CRUD and audit trail."""

    def test_put_and_get(self) -> None:
        ctx = AgentContext()
        ctx.put("key1", "value1")
        assert ctx.get("key1") == "value1"

    def test_get_default(self) -> None:
        ctx = AgentContext()
        assert ctx.get("missing") is None
        assert ctx.get("missing", 42) == 42

    def test_initial_state(self) -> None:
        ctx = AgentContext(initial_state={"counter": 0, "name": "test"})
        assert ctx.get("counter") == 0
        assert ctx.get("name") == "test"

    def test_writable_keys_enforcement(self) -> None:
        ctx = AgentContext(
            initial_state={"counter": 0, "name": "test"},
            writable_keys=["counter"],
        )
        ctx.put("counter", 1)  # allowed
        assert ctx.get("counter") == 1

        with pytest.raises(KeyError, match="not writable"):
            ctx.put("name", "changed")

    def test_state_history(self) -> None:
        ctx = AgentContext(initial_state={"x": 0})
        ctx.put("x", 1, source="agent")
        ctx.put("x", 2, source="trigger")
        hist = ctx.state_history("x")
        assert len(hist) == 3  # initial + 2 puts
        assert hist[0].source == "system"
        assert hist[1].source == "agent"
        assert hist[2].source == "trigger"
        assert [e.value for e in hist] == [0, 1, 2]

    def test_state_snapshot(self) -> None:
        ctx = AgentContext(initial_state={"a": 1, "b": 2})
        snap = ctx.state_snapshot()
        assert snap == {"a": 1, "b": 2}
        # Snapshot is a copy, not a reference
        snap["a"] = 999
        assert ctx.get("a") == 1

    def test_state_keys(self) -> None:
        ctx = AgentContext(initial_state={"x": 1, "y": 2})
        assert sorted(ctx.state_keys()) == ["x", "y"]

    def test_clear_state(self) -> None:
        ctx = AgentContext(initial_state={"x": 1})
        ctx.clear_state()
        assert ctx.get("x") is None
        assert ctx.state_history("x") == []

    def test_empty_writable_keys_means_all_writable(self) -> None:
        ctx = AgentContext()
        ctx.put("anything", "allowed")
        assert ctx.get("anything") == "allowed"


# =========================================================================
# AgentContext — Memory
# =========================================================================


class TestAgentContextMemory:
    """Verify memory provider integration."""

    def test_no_provider_returns_none(self) -> None:
        ctx = AgentContext()
        assert ctx.memory is None

    async def test_search_without_provider_returns_empty(self) -> None:
        ctx = AgentContext()
        results = await ctx.search_memory("test query")
        assert results == []

    async def test_add_without_provider_returns_none(self) -> None:
        ctx = AgentContext()
        result = await ctx.add_memory("test content")
        assert result is None

    async def test_delete_without_provider_returns_false(self) -> None:
        ctx = AgentContext()
        assert await ctx.delete_memory("id-123") is False

    async def test_search_with_provider(self) -> None:
        mock_result = MagicMock(score=0.9)
        provider = AsyncMock()
        provider.search = AsyncMock(return_value=[mock_result])

        ctx = AgentContext(memory_provider=provider)
        results = await ctx.search_memory("query", limit=3)
        assert len(results) == 1
        provider.search.assert_awaited_once_with("query", limit=3)

    async def test_search_with_min_score_filters(self) -> None:
        low = MagicMock(score=0.3)
        high = MagicMock(score=0.9)
        provider = AsyncMock()
        provider.search = AsyncMock(return_value=[low, high])

        ctx = AgentContext(memory_provider=provider)
        results = await ctx.search_memory("query", min_score=0.5)
        assert len(results) == 1
        assert results[0].score == 0.9

    async def test_add_with_provider(self) -> None:
        provider = AsyncMock()
        provider.add = AsyncMock(return_value="mem-123")

        ctx = AgentContext(memory_provider=provider)
        mid = await ctx.add_memory("content", metadata={"tag": "test"})
        assert mid == "mem-123"
        provider.add.assert_awaited_once()


# =========================================================================
# AgentContext — Env
# =========================================================================


class TestAgentContextEnv:
    """Verify environment variable filtering."""

    def test_filters_by_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_FOO", "bar")
        monkeypatch.setenv("AGENT_BAZ", "qux")
        monkeypatch.setenv("OTHER_VAR", "ignored")

        ctx = AgentContext(env_prefix="AGENT_")
        env = ctx.env
        assert env["AGENT_FOO"] == "bar"
        assert env["AGENT_BAZ"] == "qux"
        assert "OTHER_VAR" not in env

    def test_custom_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_APP_KEY", "value")
        ctx = AgentContext(env_prefix="MY_APP_")
        assert "MY_APP_KEY" in ctx.env


# =========================================================================
# AgentContext — Files
# =========================================================================


class TestAgentContextFiles:
    """Verify file mount mapping."""

    def test_file_mounts(self) -> None:
        ctx = AgentContext(file_mounts={"data": "/data/input", "config": "/etc/app"})
        assert ctx.files == {"data": "/data/input", "config": "/etc/app"}

    def test_files_is_copy(self) -> None:
        ctx = AgentContext(file_mounts={"x": "/x"})
        files = ctx.files
        files["y"] = "/y"
        assert "y" not in ctx.files


# =========================================================================
# AgentContext — Serialization
# =========================================================================


class TestAgentContextSerialization:
    """Verify to_dict / from_dict roundtrip."""

    def test_roundtrip(self) -> None:
        ctx = AgentContext(
            initial_state={"counter": 0, "name": "test"},
            writable_keys=["counter"],
            file_mounts={"data": "/data"},
            env_prefix="MYAPP_",
        )
        ctx.put("counter", 1)

        data = ctx.to_dict()
        restored = AgentContext.from_dict(data)

        assert restored.get("counter") == 1
        assert restored.get("name") == "test"
        assert restored.files == {"data": "/data"}
        assert len(restored.state_history("counter")) == 2

    def test_roundtrip_preserves_writable_keys(self) -> None:
        ctx = AgentContext(writable_keys=["a", "b"])
        data = ctx.to_dict()
        AgentContext.from_dict(data)
        # writable_keys should be restored
        assert data["writable_keys"] == ["a", "b"]

    def test_memory_not_serialized(self) -> None:
        provider = AsyncMock()
        ctx = AgentContext(memory_provider=provider)
        data = ctx.to_dict()
        assert "memory_provider" not in data

        # Restored without provider
        restored = AgentContext.from_dict(data)
        assert restored.memory is None

        # Restored with provider
        restored2 = AgentContext.from_dict(data, memory_provider=provider)
        assert restored2.memory is provider

    def test_repr(self) -> None:
        ctx = AgentContext(initial_state={"x": 1}, file_mounts={"a": "/a"})
        r = repr(ctx)
        assert "keys=1" in r
        assert "memory=no" in r
        assert "mounts=1" in r
