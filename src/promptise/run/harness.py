"""The ``promptise run`` harness: instruments every agent the program builds.

Installed with :func:`promptise._harness.install` before the program starts,
:class:`RunHarness` is asked by :func:`~promptise.agent.build_agent` for
argument overrides and told about every agent it builds.  It then:

- turns observability on (at the run's level) for agents that set no
  ``observe`` of their own, without writing report files;
- gives every agent an id (``agent``, ``agent-2``, ...) when it has none;
- adds a sink for agent events (``EventNotifier``), creating a notifier when
  the agent has none;
- replaces every agent's model with the ``--model`` override;
- publishes ``agent.built``, ``timeline``, ``event``, ``stats`` and
  ``agent.closed`` run events on the run's :class:`~promptise.run.events.EventBus`.

User values always win: an agent built with ``observe=False`` stays
unobserved, one with its own ``observe`` config keeps it.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from .events import EventBus
from .stats import RunStats

__all__ = ["RunHarness", "describe_agent"]

logger = logging.getLogger("promptise.run")

_STATS_INTERVAL = 0.5


class _TimelineBridge:
    """A collector transporter that forwards timeline entries to the run."""

    def __init__(self, harness: RunHarness, label: str) -> None:
        self._harness = harness
        self._label = label

    def on_event(self, entry: Any) -> None:
        data = entry.to_dict()
        agent = data.get("agent_id") or self._label
        self._harness.stats.add_entry(agent, data)
        self._harness.bus.publish("timeline", agent=agent, entry=data)

    def flush(self) -> None:  # transporter protocol
        return None

    def close(self) -> None:  # transporter protocol
        return None


class _EventBridge:
    """An :class:`~promptise.events.EventSink` that forwards agent events to the run."""

    def __init__(self, harness: RunHarness, label: str | None) -> None:
        self._harness = harness
        self.label = label

    async def emit(self, event: Any) -> None:
        data = event.to_dict()
        agent = self.label or data.get("agent_id") or "agent"
        self._harness.bus.publish("event", agent=agent, event=data)


def _schema(tool: Any) -> dict[str, Any] | None:
    schema = getattr(tool, "args_schema", None)
    if schema is None:
        return None
    try:
        if isinstance(schema, dict):
            return schema
        return dict(schema.model_json_schema())
    except Exception:
        return None


_SECRET_WORDS = ("key", "token", "secret", "password", "passwd", "auth", "credential", "bearer")
_LONG_OPAQUE = re.compile(r"^[A-Za-z0-9_\-.~+/=]{24,}$")


def _secretish(word: str) -> bool:
    return any(w in word.lower() for w in _SECRET_WORDS)


def redact_target(transport: str, target: str) -> str:
    """An MCP server's URL or command line, safe to log and show.

    Hosted MCP endpoints often carry the key in the URL (``user:pass@``, a
    query parameter, an opaque path segment), and stdio servers take keys as
    arguments.  URLs keep their scheme, host, port and path, with long opaque
    path segments masked; command lines keep every argument except the value
    after a secret-looking flag, a secret-looking ``NAME=value`` and long
    opaque strings.
    """
    if transport != "stdio":
        from urllib.parse import urlsplit, urlunsplit

        try:
            parts = urlsplit(target)
            host = parts.hostname or ""
            if parts.port:
                host = f"{host}:{parts.port}"
            path = "/".join(
                "***" if _LONG_OPAQUE.match(seg) and any(c.isdigit() for c in seg) else seg
                for seg in parts.path.split("/")
            )
            query = "***" if parts.query else ""
            return urlunsplit((parts.scheme, host, path, query, ""))
        except ValueError:
            return "***"
    out: list[str] = []
    hide_next = False
    for arg in target.split():
        if hide_next:
            out.append("***")
            hide_next = False
            continue
        name, eq, _ = arg.partition("=")
        if eq and _secretish(name):
            out.append(f"{name}=***")
        elif arg.startswith("-") and _secretish(arg) and not eq:
            out.append(arg)
            hide_next = True
        elif _LONG_OPAQUE.match(arg) and any(c.isdigit() for c in arg) and "/" not in arg:
            out.append("***")
        else:
            out.append(arg)
    return " ".join(out)


def _with_stream_usage(model: Any) -> Any:
    """An OpenAI chat model that reports token usage when it streams.

    A ``ChatOpenAI`` whose ``stream_usage`` is off or unset (``None``, which
    depends on the langchain-openai version and the client options) may
    stream answers without usage, so the run would count them as 0 tokens.  Only
    OpenAI's own classes are changed (OpenAI-compatible servers may reject
    the option), and only a copy: the program's model object is untouched.
    """
    if type(model).__name__ not in ("ChatOpenAI", "AzureChatOpenAI"):
        return model
    if getattr(model, "stream_usage", None) is True:
        return model
    if getattr(model, "openai_api_base", None) not in (None, "", "https://api.openai.com/v1"):
        if type(model).__name__ == "ChatOpenAI":
            return model
    try:
        return model.model_copy(update={"stream_usage": True})
    except Exception:
        return model


def _type_name(value: Any) -> str | bool:
    if value is None or value is False:
        return False
    if value is True:
        return True
    return type(value).__name__


def describe_agent(agent: Any, label: str) -> dict[str, Any]:
    """The ``agent.built`` payload for *agent* (best effort, never raises)."""
    info: dict[str, Any] = {"agent": label}
    try:
        info["model"] = getattr(agent, "model_name", None)
        cfg = getattr(agent, "_observe_config", None)
        level = getattr(cfg, "level", None)
        info["observe_level"] = getattr(level, "value", None) if cfg is not None else "off"

        multi = getattr(agent, "_mcp_multi", None)
        owners: dict[str, str] = dict(getattr(multi, "_tool_to_server", None) or {})
        tools = []
        for tool in list(getattr(agent, "tools", None) or []):
            name = str(getattr(tool, "name", "?"))
            tools.append(
                {
                    "name": name,
                    "description": str(getattr(tool, "description", "") or "")[:500],
                    "server": owners.get(name),
                    "args_schema": _schema(tool),
                }
            )
        info["tools"] = tools

        servers = []
        clients = getattr(multi, "_clients", None) or {}
        for name, client in clients.items():
            transport = getattr(client, "_transport", None) or "?"
            if transport == "stdio":
                command = getattr(client, "_command", "") or ""
                args = " ".join(str(a) for a in (getattr(client, "_args", None) or []))
                target = f"{command} {args}".strip()
            else:
                target = str(getattr(client, "_url", "") or "")
            servers.append(
                {
                    "name": name,
                    "transport": transport,
                    "target": redact_target(str(transport), target),
                    "tools": sum(1 for owner in owners.values() if owner == name),
                }
            )
        info["servers"] = servers

        info["features"] = {
            "memory": _type_name(getattr(agent, "provider", None)),
            "conversation_store": _type_name(getattr(agent, "_conversation_store", None)),
            "cache": _type_name(getattr(agent, "_cache", None)),
            "guardrails": _type_name(getattr(agent, "_guardrails", None)),
            "approval": _type_name(getattr(agent, "_approval", None)),
            "sandbox": _type_name(agent.__dict__.get("_sandbox_session")),
            "context_engine": _type_name(agent.__dict__.get("_context_engine")),
            "adaptive": _type_name(agent.__dict__.get("_strategy_manager")),
            "flow": _type_name(agent.__dict__.get("_flows")),
        }
        info["instructions"] = str(agent.__dict__.get("_raw_instructions") or "")[:2000]
    except Exception:  # introspection must never break the program
        logger.debug("could not describe agent %s", label, exc_info=True)
    return info


class RunHarness:
    """Instruments agents for one run (see the module docstring).

    Args:
        bus: Where run events are published.
        stats: The run's totals.
        observe: Observability level for agents with no ``observe`` of their
            own (``basic``, ``standard``, ``full``), or ``None`` to inject none.
        events: Forward agent events (``EventNotifier``) to the run.
        model: The ``--model`` override for every agent, or ``None``.
    """

    def __init__(
        self,
        bus: EventBus,
        stats: RunStats,
        *,
        observe: str | None = None,
        events: bool = False,
        model: str | None = None,
    ) -> None:
        self.bus = bus
        self.stats = stats
        self.observe = observe if observe not in (None, "off") else None
        self.events = events
        self.model = model
        self._lock = threading.Lock()
        self._count = 0
        self._labels: dict[int, str] = {}
        self._bridged: set[int] = set()
        self._pending_sinks: dict[int, _EventBridge] = {}
        self.agents: dict[str, Any] = {}
        """Live agents by id."""
        self.closed: set[str] = set()
        self.approvals: Any = None
        """An :class:`~promptise.run.approvals.ApprovalBridge` (terminal UI only)."""
        self._stats_version = -1
        self._stop = threading.Event()
        self._ticker: threading.Thread | None = None

    # -- harness protocol (called by build_agent / shutdown) -------------

    def _next_label(self) -> str:
        with self._lock:
            self._count += 1
            return "agent" if self._count == 1 else f"agent-{self._count}"

    def build_overrides(self, args: dict[str, Any]) -> dict[str, Any]:
        out = dict(args)
        if self.model:
            out["model"] = self.model
        out["model"] = _with_stream_usage(out["model"])
        if self.observe and args.get("observe") is None:
            from ..observability_config import ObservabilityConfig, ObserveLevel

            out["observe"] = ObservabilityConfig(
                level=ObserveLevel(self.observe), transporters=[], record_tool_io=True
            )
        if args.get("observer_agent_id") is None and args.get("identity") is None:
            out["observer_agent_id"] = self._next_label()
        if self.events and args.get("events") is None:
            from ..events import EventNotifier

            sink = _EventBridge(self, out.get("observer_agent_id"))
            notifier = EventNotifier([sink])
            self._pending_sinks[id(notifier)] = sink
            out["events"] = notifier
        return out

    def agent_built(self, agent: Any) -> None:
        label = getattr(agent, "_actor_id", None) or self._next_label()
        with self._lock:
            base, n = label, 1
            while label in self.agents:
                n += 1
                label = f"{base}-{n}"
            self.agents[label] = agent
            self._labels[id(agent)] = label
        collector = getattr(agent, "collector", None)
        if collector is not None and id(collector) not in self._bridged:
            self._bridged.add(id(collector))
            collector.add_transporter(_TimelineBridge(self, label))
        notifier = getattr(agent, "_event_notifier", None)
        if self.events and notifier is not None:
            sink = self._pending_sinks.pop(id(notifier), None)
            if sink is not None:
                sink.label = label
            elif hasattr(notifier, "add_sink"):
                notifier.add_sink(_EventBridge(self, label))
        info = describe_agent(agent, label)
        self.stats.set_model(label, info.get("model"))
        self.bus.publish("agent.built", **info)
        if self.approvals is not None:
            try:
                self.approvals.watch(agent, label)
            except Exception:
                logger.debug("could not watch %s's approvals", label, exc_info=True)

    def agent_closed(self, agent: Any) -> None:
        label = self._labels.get(id(agent))
        if label is None:
            return
        self.closed.add(label)
        self.flush_stats()
        self.bus.publish("agent.closed", agent=label)

    # -- stats ------------------------------------------------------------

    def label_of(self, agent: Any) -> str | None:
        """The id the run gave *agent*."""
        return self._labels.get(id(agent))

    def flush_stats(self, *, force: bool = False) -> None:
        """Publish a ``stats`` event when the totals changed since the last one."""
        version = self.stats.version
        if force or version != self._stats_version:
            self._stats_version = version
            self.bus.publish("stats", **self.stats.snapshot())

    def start(self) -> None:
        """Start publishing ``stats`` events twice a second while they change."""
        if self._ticker is not None:
            return

        def _tick() -> None:
            while not self._stop.wait(_STATS_INTERVAL):
                try:
                    self.flush_stats()
                except Exception:
                    logger.debug("stats publish failed", exc_info=True)

        self._ticker = threading.Thread(target=_tick, name="promptise-run-stats", daemon=True)
        self._ticker.start()

    def stop(self) -> None:
        """Stop the stats ticker and publish the final totals."""
        self._stop.set()
        if self._ticker is not None:
            self._ticker.join(timeout=2)
            self._ticker = None
        self.flush_stats()
