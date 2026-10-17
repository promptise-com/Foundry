"""`promptise run` building blocks: event bus, stats, prices, entries, the harness, IPC."""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
import threading
from pathlib import Path
from typing import Any

import pytest
from _scripted_model import Scripted
from langchain_core.tools import tool

from promptise import _harness, build_agent
from promptise.events import AgentEvent, CallbackSink, EventNotifier
from promptise.models import ModelSetupError, load_env_file
from promptise.observability_config import ObservabilityConfig, ObserveLevel
from promptise.run.context import RunContext
from promptise.run.entry import Entry, EntryError, detect_kind, load_factory, parse_ref
from promptise.run.events import EventBus
from promptise.run.harness import RunHarness
from promptise.run.ipc import IpcClient, IpcServer
from promptise.run.manifest import Price
from promptise.run.pricing import PriceTable
from promptise.run.stats import RunStats
from promptise.run.watch import FileWatcher


@pytest.fixture(autouse=True)
def _no_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")


@tool
def get_account(account_id: str) -> str:
    """Look up an account."""
    return f"account {account_id}: active"


def _write(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


def test_bus_publish_subscribe_replay_history() -> None:
    bus = EventBus(history=3)
    seen: list[dict[str, Any]] = []
    bus.publish("a", x=1)
    bus.subscribe(seen.append, replay=True)
    bus.publish("b", x=2)
    assert [e["kind"] for e in seen] == ["a", "b"]
    assert seen[1]["seq"] == 2 and seen[1]["v"] == 1 and seen[1]["x"] == 2
    for i in range(5):
        bus.publish("c", i=i)
    assert len(bus.history()) == 3
    assert [e["kind"] for e in bus.history(["b"])] == []


def test_bus_drops_a_raising_subscriber_and_is_thread_safe() -> None:
    bus = EventBus()

    def bad(event: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    good: list[int] = []
    bus.subscribe(bad)
    bus.subscribe(lambda e: good.append(e["seq"]))
    threads = [
        threading.Thread(target=lambda: [bus.publish("t") for _ in range(200)]) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert bus.dropped_subscribers == 1
    assert sorted(good) == list(range(1, 801))


# ---------------------------------------------------------------------------
# prices and stats
# ---------------------------------------------------------------------------


def test_price_table_matching() -> None:
    table = PriceTable(
        {"gpt-5-mini": Price(input=1, output=2), "claude-*": Price(input=3, output=4)}
    )
    assert table.price("gpt-5-mini") is not None
    assert table.price("openai:gpt-5-mini") is not None  # provider prefix stripped
    assert table.price("anthropic:claude-sonnet-4-5").input == 3  # glob
    assert table.price("unknown") is None and table.price(None) is None
    assert table.cost("gpt-5-mini", 1_000_000, 500_000) == pytest.approx(2.0)
    assert table.cost("unknown", 10, 10) is None


def test_stats_fold_timeline_entries() -> None:
    stats = RunStats(PriceTable({"m-1": Price(input=1, output=1)}))
    stats.set_model("a", "m-1")
    stats.add_entry(
        "a",
        {
            "event_type": "llm.end",
            "parent_id": "run1",
            "duration": 0.2,
            "metadata": {"model": "m-1", "prompt_tokens": 600_000, "completion_tokens": 400_000},
        },
    )
    stats.add_entry("a", {"event_type": "tool.call", "metadata": {"tool_name": "t"}})
    stats.add_entry(
        "a",
        {"event_type": "tool.result", "metadata": {"tool_name": "t", "latency_ms": 30.0}},
    )
    stats.add_entry("b", {"event_type": "tool.call", "metadata": {"tool_name": "t"}})
    stats.add_entry(
        "b", {"event_type": "tool.error", "metadata": {"tool_name": "t", "latency_ms": 10.0}}
    )
    stats.add_entry("b", {"event_type": "llm.retry", "metadata": {}})
    # A BASIC-level run carries its totals on agent.output instead (an unpriced model).
    stats.add_entry(
        "b",
        {
            "event_type": "agent.output",
            "parent_id": "run2",
            "metadata": {"llm_call_count": 2, "prompt_tokens": 10, "completion_tokens": 5},
        },
    )
    # ...but not again for a run that recorded llm.end.
    stats.add_entry(
        "a",
        {
            "event_type": "agent.output",
            "parent_id": "run1",
            "metadata": {"llm_call_count": 1, "prompt_tokens": 600_000, "completion_tokens": 1},
        },
    )
    snap = stats.snapshot()
    totals = snap["totals"]
    assert totals["llm_calls"] == 3
    assert totals["prompt_tokens"] == 600_010 and totals["total_tokens"] == 1_000_015
    assert totals["cost_usd"] == pytest.approx(1.0)
    assert totals["unpriced_models"] == ["unknown"]
    assert totals["tool_calls"] == 2 and totals["tool_errors"] == 1 and totals["errors"] == 1
    assert totals["retries"] == 1 and totals["latency_p50_ms"] == 200.0
    assert snap["tools"]["t"] == {"calls": 2, "errors": 1, "total_ms": 40.0, "avg_ms": 20.0}
    assert snap["agents"]["a"]["model"] == "m-1" and snap["agents"]["a"]["cost_usd"] == 1.0
    assert snap["models"]["m-1"]["llm_calls"] == 1


# ---------------------------------------------------------------------------
# entries
# ---------------------------------------------------------------------------


def test_parse_ref_forms(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    _write(tmp_path / "main.py", "print(1)\n")
    _write(tmp_path / "agent.py", "def create_agent(): ...\n")
    assert parse_ref("main.py", tmp_path).kind == "script"
    ref = parse_ref("agent.py:create_agent", tmp_path)
    assert ref.kind == "factory" and ref.attr == "create_agent" and ref.path is not None
    mod = parse_ref("pkg.agents:make", tmp_path)
    assert mod.module == "pkg.agents" and mod.attr == "make" and mod.path is None
    with pytest.raises(EntryError, match="no such file"):
        parse_ref("missing.py", tmp_path)
    with pytest.raises(EntryError, match="is not a script name"):
        parse_ref("what", tmp_path)


def test_detect_kind(tmp_path: Path) -> None:
    script = _write(
        tmp_path / "s.py",
        """
        import asyncio
        def create_agent(): ...
        if __name__ == "__main__":
            asyncio.run(main())
        """,
    )
    factory = _write(
        tmp_path / "f.py",
        """
        import sys
        sys.path.insert(0, ".")
        async def create_agent(ctx):
            ...
        """,
    )
    plain = _write(tmp_path / "p.py", "print('hi')\n")
    broken = _write(tmp_path / "b.py", "def (:\n")
    assert detect_kind(script) == ("script", None)
    assert detect_kind(factory) == ("factory", "create_agent")
    assert detect_kind(plain) == ("script", None)
    assert detect_kind(broken) == ("script", None)


async def test_load_factory_variants(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    _write(
        tmp_path / "factories_mod.py",
        """
        from _scripted_model import Scripted
        from promptise import build_agent

        async def no_args():
            return await build_agent(model=Scripted(mode="answer"))

        def with_ctx(ctx):
            async def go():
                return await build_agent(model=Scripted(mode="answer"), instructions=ctx.project)
            return go()

        def two(a, b):
            ...

        def not_agent():
            return 42
        """,
    )
    ctx = RunContext(project="p", root=tmp_path, mode="chat")
    monkeypatch.delitem(sys.modules, "factories_mod", raising=False)
    try:
        for attr in ("no_args", "with_ctx"):
            agent = await load_factory(Entry("x", tmp_path / "factories_mod.py", None, attr), ctx)
            try:
                result = await agent.ainvoke({"messages": [{"role": "user", "content": "hi"}]})
                assert result["messages"][-1].content == "ok"
            finally:
                await agent.shutdown()
        with pytest.raises(EntryError, match="no argument or one"):
            await load_factory(Entry("x", tmp_path / "factories_mod.py", None, "two"), ctx)
        with pytest.raises(EntryError, match="not an agent"):
            await load_factory(Entry("x", tmp_path / "factories_mod.py", None, "not_agent"), ctx)
        with pytest.raises(EntryError, match="no 'missing'"):
            await load_factory(Entry("x", tmp_path / "factories_mod.py", None, "missing"), ctx)
    finally:
        sys.modules.pop("factories_mod", None)


async def test_module_level_agent_is_rejected(tmp_path: Path) -> None:
    class FakeAgent:
        async def ainvoke(self, *a: Any) -> None: ...
        async def shutdown(self) -> None: ...

    mod_path = _write(tmp_path / "module_level_mod.py", "agent = None\n")
    import importlib.util

    spec = importlib.util.spec_from_file_location("module_level_mod", mod_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.create_agent = FakeAgent()  # type: ignore[attr-defined]
    sys.modules["module_level_mod"] = module
    try:
        with pytest.raises(EntryError, match="built at import time"):
            await load_factory(
                Entry("x", mod_path.resolve(), None, "create_agent"),
                RunContext(project="p", root=tmp_path, mode="chat"),
            )
    finally:
        sys.modules.pop("module_level_mod", None)


# ---------------------------------------------------------------------------
# the harness hook in build_agent
# ---------------------------------------------------------------------------


@pytest.fixture
def harness() -> Any:
    bus = EventBus()
    h = RunHarness(bus, RunStats(), observe="full", events=True)
    _harness.install(h)
    yield h
    _harness.uninstall(h)


async def test_harness_instruments_agents(harness: RunHarness) -> None:
    agent = await build_agent(
        model=Scripted(mode="tool:get_account", args={"account_id": "42"}),
        extra_tools=[get_account],
    )
    assert agent._run_harness is harness
    result = await agent.ainvoke({"messages": [{"role": "user", "content": "hi"}]})
    assert result["messages"][-1].content == "ANSWER: account 42: active"
    await agent.shutdown()
    harness.stop()
    bus = harness.bus
    built = bus.history(["agent.built"])[0]
    assert built["agent"] == "agent" and built["observe_level"] == "full"
    assert [t["name"] for t in built["tools"]] == ["get_account"]
    timeline = [e["entry"]["event_type"] for e in bus.history(["timeline"])]
    assert timeline[0] == "agent.input" and "tool.call" in timeline and "llm.end" in timeline
    assert {e["event"]["event_type"] for e in bus.history(["event"])} >= {"invocation.start"}
    assert bus.history(["agent.closed"])[0]["agent"] == "agent"
    totals = bus.history(["stats"])[-1]["totals"]
    assert totals["llm_calls"] == 2 and totals["tool_calls"] == 1
    # The injected notifier was the agent's own: shutdown stopped it.
    assert not agent._event_notifier.is_running


async def test_harness_respects_user_values(harness: RunHarness) -> None:
    received: list[str] = []

    async def collect(event: AgentEvent) -> None:
        received.append(event.event_type)

    notifier = EventNotifier([CallbackSink(collect)])
    off = await build_agent(model=Scripted(mode="answer"), observe=False, observer_agent_id="quiet")
    own = await build_agent(
        model=Scripted(mode="answer"),
        observe=ObservabilityConfig(level=ObserveLevel.BASIC, transporters=[]),
        events=notifier,
    )
    try:
        assert off.collector is None
        assert own._observe_config.level == ObserveLevel.BASIC
        assert own._event_notifier is notifier and len(notifier.sinks) == 2
        await own.ainvoke({"messages": [{"role": "user", "content": "hi"}]})
        await notifier.flush(5)
    finally:
        await off.shutdown()
        await own.shutdown()
    labels = [e["agent"] for e in harness.bus.history(["agent.built"])]
    assert labels == ["quiet", "agent"]
    # The user's own sink still gets its events, and so does the run.
    assert "invocation.start" in received
    assert any(e["agent"] == "agent" for e in harness.bus.history(["event"]))


async def test_harness_model_override() -> None:
    bus = EventBus()
    h = RunHarness(bus, RunStats(), model=None)
    h.model = Scripted(mode="answer")  # type: ignore[assignment]
    _harness.install(h)
    try:
        agent = await build_agent(
            model=Scripted(mode="tool:get_account"), extra_tools=[get_account]
        )
        result = await agent.ainvoke({"messages": [{"role": "user", "content": "hi"}]})
        assert result["messages"][-1].content == "ok"
        await agent.shutdown()
    finally:
        _harness.uninstall(h)


async def test_no_harness_changes_nothing() -> None:
    assert _harness.current() is None
    agent = await build_agent(model=Scripted(mode="answer"))
    try:
        assert agent._run_harness is None
        assert agent.collector is None and agent._event_notifier is None
    finally:
        await agent.shutdown()


def test_install_twice_is_refused() -> None:
    a = RunHarness(EventBus(), RunStats())
    b = RunHarness(EventBus(), RunStats())
    _harness.install(a)
    try:
        _harness.install(a)  # the same one again is fine
        with pytest.raises(RuntimeError):
            _harness.install(b)
    finally:
        _harness.uninstall()
    assert _harness.current() is None


# ---------------------------------------------------------------------------
# EventNotifier.add_sink
# ---------------------------------------------------------------------------


async def test_add_sink_to_a_running_notifier() -> None:
    first: list[str] = []
    second: list[str] = []

    async def one(e: AgentEvent) -> None:
        first.append(e.event_type)

    async def two(e: AgentEvent) -> None:
        second.append(e.event_type)

    notifier = EventNotifier([CallbackSink(one)])
    await notifier.start()
    sink = CallbackSink(two)
    notifier.add_sink(sink)
    notifier.add_sink(sink)  # idempotent
    assert len(notifier.sinks) == 2
    notifier.emit_sync(AgentEvent(event_type="x.y", severity="info"))
    assert await notifier.flush(5)
    await notifier.stop()
    assert first == ["x.y"] and second == ["x.y"]


async def test_add_sink_before_start() -> None:
    got: list[str] = []

    async def cb(e: AgentEvent) -> None:
        got.append(e.event_type)

    async def noop(e: AgentEvent) -> None:
        return None

    notifier = EventNotifier([CallbackSink(noop)])
    notifier.add_sink(CallbackSink(cb))
    await notifier.start()
    notifier.emit_sync(AgentEvent(event_type="a.b", severity="info"))
    assert await notifier.flush(5)
    await notifier.stop()
    assert got == ["a.b"]


# ---------------------------------------------------------------------------
# load_env_file
# ---------------------------------------------------------------------------


def test_load_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PROMPTISE_NO_DOTENV")
    monkeypatch.setenv("RUN_KEEP", "mine")
    monkeypatch.delenv("RUN_NEW", raising=False)
    path = _write(tmp_path / ".env", "RUN_KEEP=theirs\nRUN_NEW=value\n")
    assert load_env_file(path) == ["RUN_NEW"]
    assert os.environ["RUN_KEEP"] == "mine" and os.environ["RUN_NEW"] == "value"
    monkeypatch.delenv("RUN_NEW")
    assert load_env_file(tmp_path / "missing.env") == []
    monkeypatch.setenv("PROMPTISE_NO_DOTENV", "1")
    assert load_env_file(path) == [] and "RUN_NEW" not in os.environ


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_load_env_file_refuses_a_world_writable_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PROMPTISE_NO_DOTENV")
    path = _write(tmp_path / ".env", "X=1\n")
    path.chmod(0o666)
    with pytest.raises(ModelSetupError, match="world-writable"):
        load_env_file(path)


# ---------------------------------------------------------------------------
# IPC
# ---------------------------------------------------------------------------


async def test_ipc_roundtrip_and_token() -> None:
    received: list[dict[str, Any]] = []
    server = IpcServer(received.append)
    await server.start()
    commands: list[dict[str, Any]] = []
    client = await asyncio.to_thread(IpcClient, server.address)
    client.on_command = commands.append
    try:
        client({"kind": "hello", "seq": 1})
        for _ in range(100):
            if received and server.connected:
                break
            await asyncio.sleep(0.02)
        assert received == [{"kind": "hello", "seq": 1}]
        assert await server.send({"cmd": "chat", "text": "hi"})
        for _ in range(100):
            if commands:
                break
            await asyncio.sleep(0.02)
        assert commands == [{"cmd": "chat", "text": "hi"}]

        # A client with the wrong token is dropped without a word.
        intruder = await asyncio.to_thread(IpcClient, server.address.rsplit(":", 1)[0] + ":wrong")
        intruder({"kind": "evil"})
        await asyncio.sleep(0.2)
        assert {"kind": "evil"} not in received
        intruder.close()
    finally:
        client.close()
        await server.close()


# ---------------------------------------------------------------------------
# watch
# ---------------------------------------------------------------------------


def test_watch_relevance(tmp_path: Path) -> None:
    w = FileWatcher([tmp_path], lambda paths: None, ignore=["data/*"], root=tmp_path)
    assert w.relevant(str(tmp_path / "agent.py"))
    assert w.relevant(str(tmp_path / "promptise.toml"))
    assert w.relevant(str(tmp_path / ".env.local"))
    assert not w.relevant(str(tmp_path / "__pycache__" / "agent.cpython-312.pyc"))
    assert not w.relevant(str(tmp_path / ".venv" / "lib" / "x.py"))
    assert not w.relevant(str(tmp_path / "image.png"))
    assert not w.relevant(str(tmp_path / "data" / "x.json"))
    assert not w.relevant(str(tmp_path / ".agent.py.swp"))


def test_watch_debounces(tmp_path: Path) -> None:
    fired: list[list[str]] = []
    done = threading.Event()

    def on_change(paths: list[str]) -> None:
        fired.append(paths)
        done.set()

    w = FileWatcher([tmp_path], on_change, root=tmp_path, debounce=0.05)
    for name in ("a.py", "b.py", "a.py"):
        w._changed(str(tmp_path / name))
    assert done.wait(2)
    assert fired == [[str(tmp_path / "a.py"), str(tmp_path / "b.py")]]
