# Cron Trigger

The `CronTrigger` fires at scheduled intervals defined by a standard cron expression. It is the most common trigger type for periodic monitoring, data pipeline checks, and scheduled reporting.

```python
from promptise.runtime.triggers.cron import CronTrigger

trigger = CronTrigger("*/5 * * * *")
await trigger.start()

event = await trigger.wait_for_next()  # Blocks up to 5 minutes
print(event.payload)
# {"scheduled_time": "2026-03-04T10:05:00+00:00", "cron_expression": "*/5 * * * *", "timezone": "UTC"}

await trigger.stop()
```

---

## Concepts

The `CronTrigger` calculates the next fire time from the current moment, sleeps until that time, and then produces a `TriggerEvent`. This cycle repeats for as long as the trigger is active.

Expressions are evaluated with **`croniter`**, which ships with `pip install promptise`: ranges, lists, steps, day-of-week names and an optional seconds field all work. If `croniter` has been removed from the environment, a built-in fallback handles only `*/N * * * *`, `* * * * *` and single-minute expressions like `30 * * * *`.

**Schedules run in UTC** unless you set a time zone (see [Time zones](#time-zones)). `0 9 * * 1-5` therefore means 09:00 UTC on weekdays by default, which is 10:00 or 11:00 in Zurich depending on daylight saving time.

The expression (and time zone) is validated when the `TriggerConfig` or `CronTrigger` is created. A bad expression raises a `ValidationError` / `TriggerError` straight away, so a process with a broken schedule never starts.

---

## Configuration

### Via TriggerConfig

```python
from promptise.runtime import ProcessConfig, TriggerConfig

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Check data pipelines every 5 minutes.",
    triggers=[
        TriggerConfig(type="cron", cron_expression="*/5 * * * *"),
        # 09:00 Zurich time on weekdays, DST-aware
        TriggerConfig(
            type="cron",
            cron_expression="0 9 * * 1-5",
            cron_timezone="Europe/Zurich",
        ),
    ],
)
```

| Field | Default | Description |
|---|---|---|
| `cron_expression` | required | 5-field cron expression, or 6 fields with a trailing seconds field |
| `cron_timezone` | `None` (UTC) | IANA time zone name the expression is read in |

### Direct instantiation

```python
from promptise.runtime.triggers.cron import CronTrigger

# Every 5 minutes
trigger = CronTrigger("*/5 * * * *")

# Every hour at minute 0
trigger = CronTrigger("0 * * * *")

# Every day at 9:00 AM
trigger = CronTrigger("0 9 * * *")

# Every day at 9:00 AM New York time
trigger = CronTrigger("0 9 * * *", timezone="America/New_York")

# Every 10 seconds (sixth field = seconds)
trigger = CronTrigger("* * * * * */10")

# Custom trigger ID
trigger = CronTrigger("*/10 * * * *", trigger_id="pipeline-check")
```

---

## Cron Expression Reference

Standard 5-field cron format, with an optional sixth field for seconds:

```
┌───────────── minute (0-59)
│ ┌───────────── hour (0-23)
│ │ ┌───────────── day of month (1-31)
│ │ │ ┌───────────── month (1-12)
│ │ │ │ ┌───────────── day of week (0-7, 0 and 7 are Sunday)
│ │ │ │ │ ┌───────────── second (0-59, optional, croniter's convention: last)
│ │ │ │ │ │
* * * * * *
```

Common patterns:

| Expression | Schedule |
|---|---|
| `*/5 * * * *` | Every 5 minutes |
| `*/15 * * * *` | Every 15 minutes |
| `0 * * * *` | Every hour |
| `0 */2 * * *` | Every 2 hours |
| `0 9 * * *` | Daily at 9:00 AM |
| `0 9 * * 1` | Every Monday at 9:00 AM |
| `0 0 1 * *` | First day of every month |
| `* * * * *` | Every minute |
| `* * * * * */10` | Every 10 seconds |
| `* * * * * 30` | Every minute at second 30 |

## Time zones

By default the expression is evaluated in **UTC**. Set `cron_timezone` (or `timezone=` on `CronTrigger`) to an IANA name to schedule in local time:

```python
TriggerConfig(type="cron", cron_expression="0 9 * * 1-5", cron_timezone="Europe/Zurich")
```

Daylight-saving changes are handled by `croniter`: the trigger fires at 09:00 local time all year, and `scheduled_time` in the payload carries the local offset (`2026-03-02T09:00:00+01:00`). On Windows, install the `tzdata` package so time-zone names resolve.

---

## How It Works

### Wait mechanism

`CronTrigger` uses `asyncio.wait_for` with an `asyncio.Event` to implement cancellable sleeping:

1. Calculate the next fire time from the cron expression.
2. Compute the delay in seconds from now.
3. Sleep for the delay using `wait_for(event.wait(), timeout=delay)`.
4. If the event is set (by `stop()`), raise `CancelledError`.
5. If the timeout expires naturally, produce a `TriggerEvent`.

This design allows `stop()` to immediately unblock a waiting trigger rather than sleeping for the full delay.

### Event payload

```python
{
    "scheduled_time": "2026-03-04T10:05:00+00:00",
    "cron_expression": "*/5 * * * *",
    "timezone": "UTC"
}
```

---

## Lifecycle

```python
trigger = CronTrigger("*/5 * * * *")

# Start the trigger
await trigger.start()

# Wait for events in a loop
while True:
    try:
        event = await trigger.wait_for_next()
        print(f"Fired at {event.payload['scheduled_time']}")
    except asyncio.CancelledError:
        break

# Stop the trigger
await trigger.stop()
```

---

## API Summary

| Method / Property | Description |
|---|---|
| `CronTrigger(cron_expression, *, trigger_id, timezone)` | Create a cron trigger (raises `TriggerError` on a bad expression or time zone) |
| `trigger_id` | Unique identifier (auto-generated: `cron-XXXXXXXX`) |
| `await start()` | Mark the trigger as active |
| `await stop()` | Stop and unblock any waiters |
| `await wait_for_next()` | Block until the next scheduled time |

---

## Tips and Gotchas

!!! tip "Sub-minute scheduling"
    Add a sixth field for seconds: `* * * * * */10` fires every 10 seconds. Each firing is a full agent run, so keep an eye on cost.

!!! warning "UTC unless told otherwise"
    `0 9 * * *` is 09:00 **UTC**, not server-local time. Set `cron_timezone` when the schedule follows office hours.

!!! warning "Clock drift"
    The trigger computes the next fire time from the system clock. On systems with significant clock drift, scheduled times may shift. Use NTP synchronization in production.

!!! info "Validated up front"
    Invalid expressions and unknown time zones are rejected when the config is created, so `promptise runtime validate` and process start-up both catch them; a running process never ends up retrying a broken schedule.

---

## What's Next

- [Triggers Overview](index.md) -- all trigger types and the base protocol
- [Event and Webhook Triggers](event-webhook.md) -- event-driven activation
- [File Watch Trigger](file-watch.md) -- filesystem change detection
