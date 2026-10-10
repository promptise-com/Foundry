"""Trigger system for the agent runtime.

Triggers produce :class:`TriggerEvent` objects that wake an
:class:`~promptise.runtime.process.AgentProcess` and cause it to
invoke its agent.

Built-in trigger types:

* :class:`CronTrigger` — fires on a cron schedule
* :class:`EventTrigger` — fires on EventBus events
* :class:`MessageTrigger` — fires on MessageBroker messages

Additional triggers (webhook, file_watch) are available when their
optional dependencies are installed.

Custom trigger types can be registered via
:func:`register_trigger_type` and used in :class:`TriggerConfig`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config import TriggerConfig
from ..exceptions import TriggerError
from .base import BaseTrigger, TriggerEvent
from .cron import CronTrigger
from .event import EventTrigger, MessageTrigger
from .filters import FilterExpressionError, compile_filter

__all__ = [
    "BaseTrigger",
    "TriggerEvent",
    "CronTrigger",
    "EventTrigger",
    "MessageTrigger",
    "create_trigger",
    "compile_filter",
    "FilterExpressionError",
    "register_trigger_type",
    "unregister_trigger_type",
    "registered_trigger_types",
]


# ---------------------------------------------------------------------------
# Trigger registry
# ---------------------------------------------------------------------------

#: Factory callable signature:
#:   (config: TriggerConfig, *, event_bus=None, broker=None) -> BaseTrigger
TriggerFactory = Callable[..., BaseTrigger]

_TRIGGER_REGISTRY: dict[str, TriggerFactory] = {}


def register_trigger_type(
    type_name: str,
    factory: TriggerFactory,
    *,
    overwrite: bool = False,
) -> None:
    """Register a custom trigger type.

    After registration, the type name can be used in
    :class:`~promptise.runtime.config.TriggerConfig` and
    :func:`create_trigger`.

    Args:
        type_name: The type string used in ``TriggerConfig.type``.
        factory: Callable with signature
            ``(config, *, event_bus=None, broker=None) -> BaseTrigger``.
        overwrite: If ``True``, allow overwriting an existing
            registration.

    Raises:
        ValueError: If *type_name* is already registered and
            *overwrite* is ``False``.

    Example::

        from promptise.runtime.triggers import register_trigger_type

        def sqs_factory(config, *, event_bus=None, broker=None):
            return SQSTrigger(config.custom_config["queue_url"])

        register_trigger_type("sqs", sqs_factory)
    """
    if type_name in _TRIGGER_REGISTRY and not overwrite:
        raise ValueError(
            f"Trigger type {type_name!r} is already registered. Pass overwrite=True to replace it."
        )
    _TRIGGER_REGISTRY[type_name] = factory


def unregister_trigger_type(type_name: str) -> None:
    """Remove a registered trigger type.

    Silently ignores unknown type names.

    Args:
        type_name: The type string to remove.
    """
    _TRIGGER_REGISTRY.pop(type_name, None)


def registered_trigger_types() -> list[str]:
    """Return the names of all registered trigger types.

    Returns:
        Sorted list of registered type names (built-in + custom).
    """
    return sorted(_TRIGGER_REGISTRY)


# ---------------------------------------------------------------------------
# Built-in trigger factories
# ---------------------------------------------------------------------------


def _cron_factory(
    config: TriggerConfig,
    *,
    event_bus: Any | None = None,
    broker: Any | None = None,
) -> BaseTrigger:
    """Factory for cron triggers."""
    if not config.cron_expression:
        raise TriggerError("Cron trigger requires cron_expression")
    return CronTrigger(config.cron_expression, timezone=config.cron_timezone)


def _event_factory(
    config: TriggerConfig,
    *,
    event_bus: Any | None = None,
    broker: Any | None = None,
) -> BaseTrigger:
    """Factory for event triggers."""
    if event_bus is None:
        raise TriggerError("Event trigger requires an EventBus instance")
    if not config.event_type:
        raise TriggerError("Event trigger requires event_type")
    return EventTrigger(
        event_bus,
        config.event_type,
        source_filter=config.event_source,
    )


def _message_factory(
    config: TriggerConfig,
    *,
    event_bus: Any | None = None,
    broker: Any | None = None,
) -> BaseTrigger:
    """Factory for message triggers."""
    if broker is None:
        raise TriggerError("Message trigger requires a MessageBroker instance")
    if not config.topic:
        raise TriggerError("Message trigger requires topic")
    return MessageTrigger(broker, config.topic)


def _webhook_factory(
    config: TriggerConfig,
    *,
    event_bus: Any | None = None,
    broker: Any | None = None,
) -> BaseTrigger:
    """Factory for webhook triggers (lazy import)."""
    try:
        from .webhook import WebhookTrigger
    except ImportError as exc:
        raise TriggerError(
            "Webhook trigger requires aiohttp. Reinstall with: pip install --upgrade promptise"
        ) from exc
    return WebhookTrigger(
        path=config.webhook_path,
        port=config.webhook_port,
        host=config.webhook_host,
        hmac_secret=config.hmac_secret.get_secret_value() if config.hmac_secret else None,
        signature_scheme=config.signature_scheme,
        signature_header=config.signature_header,
        signature_tolerance=config.signature_tolerance,
        allowed_sources=config.allowed_sources,
        event_filter=compile_filter(config.filter_expression)
        if config.filter_expression is not None
        else None,
    )


def _file_watch_factory(
    config: TriggerConfig,
    *,
    event_bus: Any | None = None,
    broker: Any | None = None,
) -> BaseTrigger:
    """Factory for file watch triggers (lazy import)."""
    try:
        from .file_watch import FileWatchTrigger
    except ImportError as exc:
        raise TriggerError(
            "File watch trigger requires watchdog. Reinstall with: pip install --upgrade promptise"
        ) from exc
    if not config.watch_path:
        raise TriggerError("File watch trigger requires watch_path")
    return FileWatchTrigger(
        watch_path=config.watch_path,
        patterns=config.watch_patterns,
        events=config.watch_events,
        debounce_seconds=config.watch_debounce_seconds,
    )


# Register all built-in types
_TRIGGER_REGISTRY["cron"] = _cron_factory
_TRIGGER_REGISTRY["event"] = _event_factory
_TRIGGER_REGISTRY["message"] = _message_factory
_TRIGGER_REGISTRY["webhook"] = _webhook_factory
_TRIGGER_REGISTRY["file_watch"] = _file_watch_factory


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def create_trigger(
    config: TriggerConfig,
    *,
    event_bus: Any | None = None,
    broker: Any | None = None,
) -> BaseTrigger:
    """Factory: create a trigger from a :class:`TriggerConfig`.

    Looks up the trigger type in the registry (built-in and custom
    types registered via :func:`register_trigger_type`).

    Args:
        config: Trigger configuration.
        event_bus: Required for ``event`` type triggers.
        broker: Required for ``message`` type triggers.

    Returns:
        A trigger instance satisfying :class:`BaseTrigger`.

    Raises:
        TriggerError: If the trigger type is unknown or dependencies
            are missing.
    """
    factory = _TRIGGER_REGISTRY.get(config.type)
    if factory is None:
        registered = ", ".join(sorted(_TRIGGER_REGISTRY)) or "(none)"
        raise TriggerError(f"Unknown trigger type: {config.type!r}. Registered types: {registered}")

    return factory(config, event_bus=event_bus, broker=broker)
