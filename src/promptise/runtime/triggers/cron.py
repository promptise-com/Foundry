"""Cron-based trigger.

Fires at scheduled intervals defined by a cron expression, evaluated with
``croniter`` (installed with promptise).  Expressions have five fields, or
six with a trailing **seconds** field for sub-minute schedules
(``"* * * * * */10"`` = every 10 seconds).  Schedules are read in UTC
unless a ``timezone`` is given.  If ``croniter`` is missing, a simple
fallback handles ``*/N * * * *``, ``* * * * *`` and single-minute
expressions.

Expressions and time zones are validated when the trigger (or its
:class:`~promptise.runtime.config.TriggerConfig`) is created.

Example::

    trigger = CronTrigger("0 9 * * 1-5", timezone="Europe/Zurich")
    await trigger.start()
    event = await trigger.wait_for_next()  # blocks until 09:00 Zurich time
    print(event.payload)  # {"scheduled_time": "2026-03-02T09:00:00+01:00", ...}
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timedelta, timezone, tzinfo
from uuid import uuid4

from ..exceptions import TriggerError
from .base import TriggerEvent

logger = logging.getLogger(__name__)

try:
    from croniter import croniter  # type: ignore[import-untyped]

    CRONITER_AVAILABLE = True
except ImportError:
    CRONITER_AVAILABLE = False


def validate_cron_expression(expression: str) -> None:
    """Raise :class:`TriggerError` unless *expression* is a usable cron expression.

    With ``croniter`` installed this accepts 5 fields, or 6 with a trailing
    seconds field.  Without it, only the fallback's simple forms pass.
    """
    if CRONITER_AVAILABLE:
        fields = len(expression.split())
        if fields not in (5, 6) or not croniter.is_valid(expression):
            raise TriggerError(
                f"Invalid cron expression: {expression!r} (expected 5 fields "
                "'minute hour day month weekday', or 6 with a trailing seconds field)"
            )
        return
    CronTrigger._simple_next_fire_for(expression, datetime.now(timezone.utc))


def validate_timezone(name: str) -> tzinfo:
    """Return the :class:`~zoneinfo.ZoneInfo` for *name* or raise :class:`TriggerError`."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise TriggerError(
            f"Unknown time zone {name!r} (use an IANA name such as 'Europe/Zurich'; "
            "on Windows install the 'tzdata' package)"
        ) from exc


class CronTrigger:
    """Fires at scheduled intervals defined by a cron expression.

    Args:
        cron_expression: Cron expression (e.g. ``*/5 * * * *``), with an
            optional sixth seconds field.
        trigger_id: Unique identifier (auto-generated if not provided).
        timezone: IANA time zone the expression is read in (default UTC).

    Raises:
        TriggerError: If the expression or time zone is invalid.
    """

    def __init__(
        self,
        cron_expression: str,
        *,
        trigger_id: str | None = None,
        timezone: str | None = None,
    ) -> None:
        validate_cron_expression(cron_expression)
        self._tz: tzinfo | None = validate_timezone(timezone) if timezone else None
        self.trigger_id = trigger_id or f"cron-{uuid4().hex[:8]}"
        self._cron_expression = cron_expression
        self._running = False
        self._event: asyncio.Event = asyncio.Event()

    async def start(self) -> None:
        """Mark the trigger as active."""
        self._running = True
        logger.info(
            "CronTrigger %s started: %s",
            self.trigger_id,
            self._cron_expression,
        )

    async def stop(self) -> None:
        """Mark the trigger as inactive and unblock waiters."""
        self._running = False
        self._event.set()
        logger.info("CronTrigger %s stopped", self.trigger_id)

    async def wait_for_next(self) -> TriggerEvent:
        """Block until the next scheduled time.

        Uses ``self._event`` to allow :meth:`stop` to unblock the wait
        immediately instead of sleeping for the full delay.

        Returns:
            A :class:`TriggerEvent` with the scheduled time in the payload.

        Raises:
            asyncio.CancelledError: If stopped while waiting.
            TriggerError: If the cron expression is invalid.
        """
        self._event.clear()
        next_fire = self._compute_next_fire()
        now = datetime.now(timezone.utc)
        delay = max(0, (next_fire - now).total_seconds())

        if delay > 0:
            # Wait for either the delay or a stop signal
            try:
                await asyncio.wait_for(self._event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass  # Delay elapsed — fire the trigger
            except asyncio.CancelledError:
                raise

        if not self._running:
            raise asyncio.CancelledError("Trigger stopped")

        return TriggerEvent(
            trigger_id=self.trigger_id,
            trigger_type="cron",
            payload={
                "scheduled_time": next_fire.isoformat(),
                "cron_expression": self._cron_expression,
                "timezone": str(self._tz) if self._tz else "UTC",
            },
        )

    def _compute_next_fire(self) -> datetime:
        """Calculate the next fire time from now (in the trigger's time zone)."""
        now = datetime.now(self._tz or timezone.utc)

        if CRONITER_AVAILABLE:
            try:
                cron = croniter(self._cron_expression, now)
                next_dt = cron.get_next(datetime)
                if next_dt.tzinfo is None:
                    next_dt = next_dt.replace(tzinfo=self._tz or timezone.utc)
                return next_dt
            except (ValueError, KeyError) as exc:
                raise TriggerError(f"Invalid cron expression: {self._cron_expression!r}") from exc

        # Fallback: parse simple interval expressions like "*/N * * * *"
        return self._simple_next_fire(now)

    def _simple_next_fire(self, now: datetime) -> datetime:
        """Parse simple ``*/N * * * *`` expressions without croniter."""
        return self._simple_next_fire_for(self._cron_expression, now)

    @staticmethod
    def _simple_next_fire_for(expression: str, now: datetime) -> datetime:
        parts = expression.strip().split()
        if len(parts) < 5:
            raise TriggerError(f"Invalid cron expression (need 5 fields): {expression!r}")

        minute_field = parts[0]
        match = re.match(r"^\*/(\d+)$", minute_field)
        if match:
            interval = int(match.group(1))
            return now + timedelta(minutes=interval)

        # Every minute
        if minute_field == "*":
            return now + timedelta(minutes=1)

        # Specific minute
        try:
            target_minute = int(minute_field)
            result = now.replace(second=0, microsecond=0)
            if result.minute >= target_minute:
                result += timedelta(hours=1)
            result = result.replace(minute=target_minute)
            return result
        except ValueError:
            pass

        raise TriggerError(
            f"Cannot parse cron expression without croniter: "
            f"{expression!r}. Install croniter for full support."
        )

    def __repr__(self) -> str:
        return (
            f"CronTrigger(id={self.trigger_id!r}, "
            f"cron={self._cron_expression!r}, "
            f"running={self._running})"
        )
