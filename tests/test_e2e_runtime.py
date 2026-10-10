"""End-to-end tests for the runtime subsystem.

Covers: processes, triggers, journals, manifests, meta-tools, distributed
coordination, context, conversation buffer, and lifecycle.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from promptise.runtime import (
    AgentContext,
    AgentProcess,
    AgentRuntime,
    ConversationBuffer,
    CronTrigger,
    EventTrigger,
    MessageTrigger,
    ProcessConfig,
    ProcessLifecycle,
    ProcessState,
    RuntimeConfig,
    StateError,
    TriggerConfig,
)
from promptise.runtime.config import (
    DistributedConfig,
    ExecutionMode,
)
from promptise.runtime.distributed.coordinator import NodeInfo, RuntimeCoordinator
from promptise.runtime.exceptions import RuntimeBaseError
from promptise.runtime.journal import InMemoryJournal, ReplayEngine
from promptise.runtime.journal.base import JournalEntry
from promptise.runtime.journal.file import FileJournal
from promptise.runtime.manifest import (
    AgentManifestSchema,
    load_manifest,
    manifest_to_process_config,
    save_manifest,
)
from promptise.runtime.triggers import (
    create_trigger,
    register_trigger_type,
    registered_trigger_types,
    unregister_trigger_type,
)
from promptise.runtime.triggers.base import TriggerEvent

BUILD_TARGET = "promptise.agent.build_agent"


def _make_mock_agent() -> AsyncMock:
    """Create a mock PromptiseAgent."""
    agent = AsyncMock()
    agent.ainvoke = AsyncMock(return_value={"messages": [{"role": "assistant", "content": "Done"}]})
    agent.shutdown = AsyncMock()
    return agent


def _patch_build(agent: AsyncMock | None = None):
    """Patch build_agent to return a mock agent."""
    if agent is None:
        agent = _make_mock_agent()
    return patch(
        BUILD_TARGET,
        new_callable=lambda: AsyncMock(return_value=agent),
    )


# =========================================================================
# 1. TestProcessConfig (4 tests)
# =========================================================================


class TestProcessConfig:
    """Verify ProcessConfig defaults, validation, serialization, and composition."""

    def test_defaults(self) -> None:
        cfg = ProcessConfig()
        assert cfg.model == "openai:gpt-5-mini"
        assert cfg.concurrency == 1
        assert cfg.heartbeat_interval == 10.0
        assert cfg.idle_timeout == 0.0
        assert cfg.max_lifetime == 0.0
        assert cfg.max_consecutive_failures == 3
        assert cfg.restart_policy == "never"
        assert cfg.max_restarts == 3
        assert cfg.execution_mode == ExecutionMode.STRICT

    def test_validation_rejects_negative_concurrency(self) -> None:
        with pytest.raises(ValidationError):
            ProcessConfig(concurrency=0)

    def test_serialization_roundtrip(self) -> None:
        cfg = ProcessConfig(
            model="openai:gpt-4o",
            instructions="Be helpful.",
            concurrency=4,
            heartbeat_interval=30.0,
            triggers=[
                TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
            ],
        )
        data = cfg.model_dump(mode="json")
        restored = ProcessConfig.model_validate(data)
        assert restored.model == "openai:gpt-4o"
        assert restored.instructions == "Be helpful."
        assert restored.concurrency == 4
        assert restored.heartbeat_interval == 30.0
        assert len(restored.triggers) == 1
        assert restored.triggers[0].cron_expression == "*/5 * * * *"

    def test_runtime_config_composition(self) -> None:
        rt = RuntimeConfig(
            processes={
                "watcher": ProcessConfig(model="openai:gpt-4o"),
                "analyst": ProcessConfig(model="anthropic:claude-3-haiku"),
            },
            distributed=DistributedConfig(enabled=True, transport_port=9200),
        )
        assert len(rt.processes) == 2
        assert rt.processes["watcher"].model == "openai:gpt-4o"
        assert rt.distributed.enabled is True
        assert rt.distributed.transport_port == 9200
        data = rt.to_dict()
        restored = RuntimeConfig.from_dict(data)
        assert restored.processes["analyst"].model == "anthropic:claude-3-haiku"


# =========================================================================
# 2. TestProcessLifecycle (4 tests)
# =========================================================================


class TestProcessLifecycle:
    """Verify the process lifecycle state machine."""

    async def test_full_forward_lifecycle(self) -> None:
        lc = ProcessLifecycle()
        assert lc.state == ProcessState.CREATED
        await lc.transition(ProcessState.STARTING, reason="start")
        await lc.transition(ProcessState.RUNNING, reason="agent built")
        await lc.transition(ProcessState.STOPPING, reason="user stop")
        await lc.transition(ProcessState.STOPPED, reason="clean shutdown")
        assert lc.state == ProcessState.STOPPED
        assert len(lc.history) == 4

    async def test_suspend_and_resume(self) -> None:
        lc = ProcessLifecycle()
        await lc.transition(ProcessState.STARTING)
        await lc.transition(ProcessState.RUNNING)
        await lc.transition(ProcessState.SUSPENDED, reason="idle timeout")
        assert lc.state == ProcessState.SUSPENDED
        await lc.transition(ProcessState.RUNNING, reason="resume")
        assert lc.state == ProcessState.RUNNING

    async def test_invalid_transition_raises_state_error(self) -> None:
        lc = ProcessLifecycle()
        with pytest.raises(StateError) as exc_info:
            await lc.transition(ProcessState.RUNNING)
        assert exc_info.value.current == ProcessState.CREATED
        assert exc_info.value.target == ProcessState.RUNNING

    def test_process_state_enum_values(self) -> None:
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


# =========================================================================
# 3. TestAgentProcess (4 tests)
# =========================================================================


class TestAgentProcess:
    """Verify AgentProcess creation, lifecycle, status, and spawn tracking."""

    def test_create_with_process_config(self) -> None:
        cfg = ProcessConfig(
            model="openai:gpt-4o",
            instructions="Monitor data.",
            concurrency=2,
        )
        process = AgentProcess("data-monitor", cfg)
        assert process.name == "data-monitor"
        assert process.config.model == "openai:gpt-4o"
        assert process.state == ProcessState.CREATED
        assert process.process_id  # auto-generated UUID

    async def test_start_stop_lifecycle(self) -> None:
        with _patch_build():
            process = AgentProcess("test-proc", ProcessConfig())
            assert process.state == ProcessState.CREATED
            await process.start()
            assert process.state == ProcessState.RUNNING
            await process.stop()
            assert process.state == ProcessState.STOPPED

    async def test_status_returns_dict(self) -> None:
        with _patch_build():
            process = AgentProcess("status-proc", ProcessConfig())
            await process.start()
            status = process.status()
            assert isinstance(status, dict)
            assert status["name"] == "status-proc"
            assert status["state"] == "running"
            assert status["invocation_count"] == 0
            assert status["consecutive_failures"] == 0
            assert status["spawned_process_count"] == 0
            assert status["uptime_seconds"] is not None
            assert status["uptime_seconds"] >= 0
            await process.stop()

    def test_spawned_process_count_tracking(self) -> None:
        process = AgentProcess("parent", ProcessConfig())
        assert process.status()["spawned_process_count"] == 0
        process._spawned_processes.append("child-1")
        process._spawned_processes.append("child-2")
        assert process.status()["spawned_process_count"] == 2


# =========================================================================
# 4. TestTriggers (6 tests)
# =========================================================================


class TestTriggers:
    """Verify trigger creation, start/stop, and event production."""

    def test_cron_trigger_creation_and_trigger_id(self) -> None:
        trigger = CronTrigger("*/5 * * * *")
        assert trigger.trigger_id.startswith("cron-")
        trigger2 = CronTrigger("*/10 * * * *", trigger_id="my-cron-id")
        assert trigger2.trigger_id == "my-cron-id"

    async def test_cron_trigger_start_stop(self) -> None:
        trigger = CronTrigger("*/5 * * * *")
        assert trigger._running is False
        await trigger.start()
        assert trigger._running is True
        await trigger.stop()
        assert trigger._running is False

    async def test_event_trigger_with_mock_bus(self) -> None:
        bus = AsyncMock()
        bus.subscribe = AsyncMock()
        bus.unsubscribe = AsyncMock()
        trigger = EventTrigger(bus, "task.done")
        await trigger.start()
        bus.subscribe.assert_awaited_once_with("task.done", trigger._handler)

        mock_event = MagicMock()
        mock_event.event_id = "ev-1"
        mock_event.source = "agent-1"
        mock_event.data = {"result": "ok"}
        await trigger._handler(mock_event)
        event = await trigger.wait_for_next()
        assert event.trigger_type == "event"
        assert event.payload["event_type"] == "task.done"
        await trigger.stop()

    async def test_message_trigger_with_mock_broker(self) -> None:
        broker = AsyncMock()
        broker.subscribe = AsyncMock(return_value="sub-abc")
        broker.unsubscribe = AsyncMock()
        trigger = MessageTrigger(broker, "reports.*")
        await trigger.start()
        broker.subscribe.assert_awaited_once()

        msg = MagicMock()
        msg.message_id = "m-1"
        msg.sender = "sender-1"
        msg.content = "report ready"
        await trigger._handler(msg)
        event = await trigger.wait_for_next()
        assert event.trigger_type == "message"
        assert event.payload["topic"] == "reports.*"
        await trigger.stop()
        broker.unsubscribe.assert_awaited_once_with("sub-abc")

    def test_create_trigger_factory_for_each_type(self) -> None:
        cron_cfg = TriggerConfig(type="cron", cron_expression="* * * * *")
        cron_t = create_trigger(cron_cfg)
        assert isinstance(cron_t, CronTrigger)

        bus = AsyncMock()
        event_cfg = TriggerConfig(type="event", event_type="task.completed")
        event_t = create_trigger(event_cfg, event_bus=bus)
        assert isinstance(event_t, EventTrigger)

        broker = AsyncMock()
        msg_cfg = TriggerConfig(type="message", topic="reports.*")
        msg_t = create_trigger(msg_cfg, broker=broker)
        assert isinstance(msg_t, MessageTrigger)

    def test_trigger_event_creation_and_serialization(self) -> None:
        event = TriggerEvent(
            trigger_id="t-1",
            trigger_type="cron",
            payload={"scheduled_time": "2026-03-01T10:00:00"},
            metadata={"source": "test"},
        )
        assert event.trigger_id == "t-1"
        assert event.trigger_type == "cron"
        assert event.event_id  # auto-generated
        data = event.to_dict()
        restored = TriggerEvent.from_dict(data)
        assert restored.trigger_id == event.trigger_id
        assert restored.trigger_type == event.trigger_type
        assert restored.payload == event.payload
        assert restored.metadata == event.metadata


# =========================================================================
# 5. TestCustomTriggerRegistry (4 tests)
# =========================================================================


class TestCustomTriggerRegistry:
    """Verify custom trigger type registration and lifecycle."""

    def test_register_trigger_type(self) -> None:
        def sqs_factory(config, *, event_bus=None, broker=None):
            return CronTrigger("* * * * *")

        register_trigger_type("_e2e_sqs", sqs_factory)
        try:
            assert "_e2e_sqs" in registered_trigger_types()
        finally:
            unregister_trigger_type("_e2e_sqs")

    def test_registered_trigger_types_lists_all(self) -> None:
        types = registered_trigger_types()
        for builtin in ("cron", "event", "message", "webhook", "file_watch"):
            assert builtin in types

    def test_create_trigger_with_custom_type(self) -> None:
        def custom_factory(config, *, event_bus=None, broker=None):
            expr = config.custom_config.get("expression", "* * * * *")
            return CronTrigger(expr, trigger_id="custom-trigger")

        register_trigger_type("_e2e_custom", custom_factory)
        try:
            cfg = TriggerConfig(
                type="_e2e_custom",
                custom_config={"expression": "*/3 * * * *"},
            )
            trigger = create_trigger(cfg)
            assert trigger.trigger_id == "custom-trigger"
        finally:
            unregister_trigger_type("_e2e_custom")

    def test_unregister_trigger_type_cleanup(self) -> None:
        def factory(c, **kw):
            return CronTrigger("* * * * *")

        register_trigger_type("_e2e_cleanup", factory)
        assert "_e2e_cleanup" in registered_trigger_types()
        unregister_trigger_type("_e2e_cleanup")
        assert "_e2e_cleanup" not in registered_trigger_types()
        # Unregistering unknown type is a no-op
        unregister_trigger_type("_e2e_cleanup")


# =========================================================================
# 6. TestJournals (5 tests)
# =========================================================================


class TestJournals:
    """Verify InMemoryJournal, FileJournal, and ReplayEngine."""

    async def test_in_memory_journal_append_and_read(self) -> None:
        journal = InMemoryJournal()
        entry = JournalEntry(
            process_id="proc-1",
            entry_type="trigger_event",
            data={"trigger_type": "cron"},
        )
        await journal.append(entry)
        entries = await journal.read("proc-1")
        assert len(entries) == 1
        assert entries[0].process_id == "proc-1"
        assert entries[0].entry_type == "trigger_event"

    async def test_in_memory_journal_filter_by_entry_type(self) -> None:
        journal = InMemoryJournal()
        await journal.append(JournalEntry(process_id="p-1", entry_type="trigger_event"))
        await journal.append(JournalEntry(process_id="p-1", entry_type="state_transition"))
        await journal.append(JournalEntry(process_id="p-1", entry_type="trigger_event"))
        results = await journal.read("p-1", entry_type="trigger_event")
        assert len(results) == 2
        for r in results:
            assert r.entry_type == "trigger_event"

    async def test_in_memory_journal_checkpoint_and_restore(self) -> None:
        journal = InMemoryJournal()
        state = {"counter": 42, "name": "test"}
        await journal.checkpoint("proc-1", state)
        restored = await journal.last_checkpoint("proc-1")
        assert restored == state
        # Non-existent process returns None
        assert await journal.last_checkpoint("unknown") is None

    async def test_file_journal_persistence(self, tmp_path: Path) -> None:
        journal = FileJournal(str(tmp_path / "journals"))
        entry1 = JournalEntry(
            process_id="file-proc",
            entry_type="invocation_result",
            data={"result": "ok"},
        )
        entry2 = JournalEntry(
            process_id="file-proc",
            entry_type="state_transition",
            data={"to_state": "running"},
        )
        await journal.append(entry1)
        await journal.append(entry2)

        entries = await journal.read("file-proc")
        assert len(entries) == 2

        # Checkpoint
        await journal.checkpoint("file-proc", {"counter": 10})
        cp = await journal.last_checkpoint("file-proc")
        assert cp == {"counter": 10}

        # Filter by entry type
        transitions = await journal.read("file-proc", entry_type="state_transition")
        assert len(transitions) == 1

    async def test_replay_engine_replays_in_order(self) -> None:
        journal = InMemoryJournal()
        pid = "replay-proc"

        # Simulate a lifecycle: entries before checkpoint
        await journal.append(
            JournalEntry(
                process_id=pid,
                entry_type="state_transition",
                data={"to_state": "starting"},
            )
        )
        await journal.checkpoint(
            pid,
            {
                "context_state": {"counter": 5},
                "lifecycle_state": "running",
            },
        )
        # Entries after checkpoint
        await journal.append(
            JournalEntry(
                process_id=pid,
                entry_type="context_update",
                data={"key": "counter", "value": 10},
            )
        )
        await journal.append(
            JournalEntry(
                process_id=pid,
                entry_type="state_transition",
                data={"to_state": "suspended"},
            )
        )

        engine = ReplayEngine(journal)
        recovered = await engine.recover(pid)
        assert recovered["context_state"]["counter"] == 10
        assert recovered["lifecycle_state"] == "suspended"
        assert recovered["entries_replayed"] == 2


# =========================================================================
# 7. TestAgentContext (3 tests)
# =========================================================================


class TestAgentContext:
    """Verify AgentContext state operations."""

    def test_state_get_set(self) -> None:
        ctx = AgentContext(initial_state={"counter": 0})
        assert ctx.get("counter") == 0
        ctx.put("counter", 42, source="agent")
        assert ctx.get("counter") == 42
        assert ctx.get("missing") is None
        assert ctx.get("missing", "default") == "default"
        history = ctx.state_history("counter")
        assert len(history) == 2
        assert history[0].source == "system"
        assert history[1].source == "agent"

    def test_writable_keys_enforcement(self) -> None:
        ctx = AgentContext(
            initial_state={"allowed": 0, "locked": "secret"},
            writable_keys=["allowed"],
        )
        ctx.put("allowed", 1)
        assert ctx.get("allowed") == 1
        with pytest.raises(KeyError, match="not writable"):
            ctx.put("locked", "hacked")

    def test_environment_variable_access(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_TEST_KEY", "test_value")
        monkeypatch.setenv("OTHER_KEY", "ignored")
        ctx = AgentContext(env_prefix="AGENT_")
        env = ctx.env
        assert env["AGENT_TEST_KEY"] == "test_value"
        assert "OTHER_KEY" not in env


# =========================================================================
# 8. TestConversationBuffer (2 tests)
# =========================================================================


class TestConversationBuffer:
    """Verify conversation buffer add/get and window truncation."""

    def test_add_and_get_messages(self) -> None:
        buf = ConversationBuffer(max_messages=100)
        buf.append({"role": "user", "content": "Hello!"})
        buf.append({"role": "assistant", "content": "Hi there!"})
        messages = buf.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert len(buf) == 2

    def test_window_limit_truncation(self) -> None:
        buf = ConversationBuffer(max_messages=3)
        for i in range(5):
            buf.append({"role": "user", "content": f"msg-{i}"})
        messages = buf.get_messages()
        assert len(messages) == 3
        assert messages[0]["content"] == "msg-2"
        assert messages[1]["content"] == "msg-3"
        assert messages[2]["content"] == "msg-4"


# =========================================================================
# 9. TestAgentRuntime (4 tests)
# =========================================================================


class TestAgentRuntime:
    """Verify AgentRuntime process management."""

    async def test_add_process_registers(self) -> None:
        runtime = AgentRuntime()
        process = await runtime.add_process("proc-a", ProcessConfig())
        assert "proc-a" in runtime.processes
        assert process.name == "proc-a"

    async def test_duplicate_name_raises_error(self) -> None:
        runtime = AgentRuntime()
        await runtime.add_process("proc-a", ProcessConfig())
        with pytest.raises(RuntimeBaseError, match="already exists"):
            await runtime.add_process("proc-a", ProcessConfig())

    async def test_start_all_stop_all(self) -> None:
        with _patch_build():
            runtime = AgentRuntime()
            await runtime.add_process("p1", ProcessConfig())
            await runtime.add_process("p2", ProcessConfig())
            await runtime.start_all()
            assert runtime.get_process("p1").state == ProcessState.RUNNING
            assert runtime.get_process("p2").state == ProcessState.RUNNING
            await runtime.stop_all()
            assert runtime.get_process("p1").state == ProcessState.STOPPED
            assert runtime.get_process("p2").state == ProcessState.STOPPED

    async def test_context_manager_lifecycle(self) -> None:
        with _patch_build():
            async with AgentRuntime() as runtime:
                await runtime.add_process("ctx-proc", ProcessConfig())
                await runtime.start_all()
                assert runtime.get_process("ctx-proc").state == ProcessState.RUNNING
            # After exiting, stop_all should have been called
            assert runtime.get_process("ctx-proc").state == ProcessState.STOPPED


# =========================================================================
# 10. TestManifest (3 tests)
# =========================================================================


class TestManifest:
    """Verify manifest schema, load/save, and conversion to ProcessConfig."""

    def test_schema_validation_with_valid_data(self) -> None:
        manifest = AgentManifestSchema(
            name="test-agent",
            model="openai:gpt-5-mini",
            instructions="You are a helpful assistant.",
            triggers=[{"type": "cron", "cron_expression": "*/5 * * * *"}],
            world={"counter": 0},
        )
        assert manifest.name == "test-agent"
        assert manifest.version == "1.0"
        assert len(manifest.triggers) == 1

    def test_load_save_roundtrip(self, tmp_path: Path) -> None:
        manifest = AgentManifestSchema(
            name="roundtrip-agent",
            model="openai:gpt-4o",
            instructions="Watch things.",
            triggers=[{"type": "cron", "cron_expression": "0 * * * *"}],
            world={"status": "active"},
        )
        file_path = tmp_path / "test.agent"
        save_manifest(manifest, file_path)
        assert file_path.exists()
        loaded = load_manifest(file_path)
        assert loaded.name == "roundtrip-agent"
        assert loaded.model == "openai:gpt-4o"
        assert loaded.instructions == "Watch things."
        assert loaded.world == {"status": "active"}
        assert len(loaded.triggers) == 1

    def test_manifest_to_process_config_mapping(self) -> None:
        manifest = AgentManifestSchema(
            name="config-agent",
            model="openai:gpt-4o",
            instructions="Be smart.",
            triggers=[{"type": "cron", "cron_expression": "*/10 * * * *"}],
            world={"pipeline": "healthy"},
            config={"concurrency": 4, "heartbeat_interval": 30.0},
        )
        cfg = manifest_to_process_config(manifest)
        assert cfg.model == "openai:gpt-4o"
        assert cfg.instructions == "Be smart."
        assert cfg.concurrency == 4
        assert cfg.heartbeat_interval == 30.0
        assert len(cfg.triggers) == 1
        assert cfg.triggers[0].cron_expression == "*/10 * * * *"
        assert cfg.context.initial_state == {"pipeline": "healthy"}


# =========================================================================
# 11. TestDistributed (2 tests)
# =========================================================================


class TestDistributed:
    """Verify RuntimeCoordinator and NodeInfo."""

    def test_coordinator_register_node(self) -> None:
        coordinator = RuntimeCoordinator()
        node = coordinator.register_node(
            "node-1",
            "http://host1:9100",
            metadata={"region": "us-east"},
        )
        assert node.node_id == "node-1"
        assert node.url == "http://host1:9100"
        assert node.metadata == {"region": "us-east"}
        assert "node-1" in coordinator.nodes
        retrieved = coordinator.get_node("node-1")
        assert retrieved.node_id == "node-1"

    def test_node_info_serialization(self) -> None:
        node = NodeInfo(
            node_id="n-1",
            url="http://host:9100",
            status="healthy",
            process_count=3,
            metadata={"env": "prod"},
        )
        data = node.to_dict()
        assert data["node_id"] == "n-1"
        assert data["url"] == "http://host:9100"
        assert data["status"] == "healthy"
        assert data["process_count"] == 3
        assert data["metadata"] == {"env": "prod"}
        assert node.is_healthy is True


# =========================================================================
# 12. TestExecutionMode (1 test)
# =========================================================================


class TestExecutionMode:
    """Verify ExecutionMode enum values."""

    def test_enum_values(self) -> None:
        assert ExecutionMode.STRICT.value == "strict"
        assert ExecutionMode.OPEN.value == "open"
        assert ExecutionMode("strict") is ExecutionMode.STRICT
        assert ExecutionMode("open") is ExecutionMode.OPEN


# =========================================================================
# 13. TestProcessIdentity (identity wired into the runtime build)
# =========================================================================


class TestProcessIdentity:
    """A process identity is attributed and presented to its MCP servers."""

    def _patch_build(self, captured_bearers: list, captured_build: dict) -> list:
        from promptise.config import HTTPServerSpec  # noqa: F401

        def _fake_client(**kwargs: object) -> MagicMock:
            # The identity is presented through a provider asked on every
            # request (so the credential is renewed); record what it yields.
            provider = kwargs.get("bearer_token_provider")
            captured_bearers.append(
                provider(False) if callable(provider) else kwargs.get("bearer_token")
            )
            return MagicMock()

        multi = MagicMock()
        multi.__aenter__ = AsyncMock(return_value=multi)
        multi.__aexit__ = AsyncMock(return_value=None)
        adapter = MagicMock()
        adapter.as_langchain_tools = AsyncMock(return_value=[])

        async def _fake_build(**kwargs: object) -> MagicMock:
            captured_build.update(kwargs)
            return MagicMock()

        return [
            patch("promptise.mcp.client.MCPClient", side_effect=_fake_client),
            patch("promptise.mcp.client.MCPMultiClient", return_value=multi),
            patch("promptise.mcp.client.MCPToolAdapter", return_value=adapter),
            patch("promptise.agent.build_agent", side_effect=_fake_build),
        ]

    @pytest.mark.asyncio
    async def test_verifiable_identity_presented_per_server_audience(self) -> None:
        from contextlib import ExitStack

        from promptise.config import HTTPServerSpec
        from promptise.identity import AgentIdentity, CallableTokenProvider

        seen_audiences: list = []

        def mint(audience: str | None = None) -> str:
            seen_audiences.append(audience)
            return f"token-for-{audience}"

        identity = AgentIdentity("watcher-bot", credential=CallableTokenProvider(token_fn=mint))
        cfg = ProcessConfig(
            model="openai:gpt-5-mini",
            identity=identity,
            servers={
                "data": HTTPServerSpec(url="https://data.internal/mcp", audience="api://data")
            },
        )
        process = AgentProcess("watcher", cfg)

        bearers: list = []
        build_kwargs: dict = {}
        with ExitStack() as stack:
            for cm in self._patch_build(bearers, build_kwargs):
                stack.enter_context(cm)
            await process._build_agent()

        # Outbound: the per-audience credential reached the MCP client.
        assert "token-for-api://data" in bearers
        assert "api://data" in seen_audiences
        # Attribution: the identity was handed to build_agent.
        assert build_kwargs["identity"] is identity

    @pytest.mark.asyncio
    async def test_local_identity_presents_no_credential(self) -> None:
        from contextlib import ExitStack

        from promptise.config import HTTPServerSpec
        from promptise.identity import AgentIdentity

        cfg = ProcessConfig(
            model="openai:gpt-5-mini",
            identity=AgentIdentity("local-bot"),
            servers={"data": HTTPServerSpec(url="https://data.internal/mcp")},
        )
        process = AgentProcess("watcher", cfg)

        bearers: list = []
        build_kwargs: dict = {}
        with ExitStack() as stack:
            for cm in self._patch_build(bearers, build_kwargs):
                stack.enter_context(cm)
            await process._build_agent()

        assert bearers == [None]  # local identity → no credential presented
        assert build_kwargs["identity"].agent_id == "local-bot"  # still attributed

    @pytest.mark.asyncio
    async def test_unreachable_idp_fails_closed(self) -> None:
        """The process never presents its MCP servers an unauthenticated
        request when its identity cannot acquire a credential."""
        from contextlib import ExitStack

        from promptise.config import HTTPServerSpec
        from promptise.identity import (
            AgentIdentity,
            CallableTokenProvider,
            CredentialAcquisitionError,
        )

        def mint(audience: str | None = None) -> str:
            raise CredentialAcquisitionError("metadata server unreachable")

        cfg = ProcessConfig(
            model="openai:gpt-5-mini",
            identity=AgentIdentity("watcher-bot", credential=CallableTokenProvider(token_fn=mint)),
            servers={
                "data": HTTPServerSpec(url="https://data.internal/mcp", audience="api://data")
            },
        )
        process = AgentProcess("watcher", cfg)
        providers: list = []

        def _fake_client(**kwargs: object) -> MagicMock:
            providers.append(kwargs.get("bearer_token_provider"))
            return MagicMock()

        patches = self._patch_build([], {})
        patches[0] = patch("promptise.mcp.client.MCPClient", side_effect=_fake_client)
        with ExitStack() as stack:
            for cm in patches:
                stack.enter_context(cm)
            await process._build_agent()

        (provider,) = providers
        with pytest.raises(CredentialAcquisitionError):
            provider(False)
