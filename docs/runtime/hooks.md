# Runtime Lifecycle Hooks

React to high-level agent process events without modifying the agent itself. Runtime hooks are separate from [reasoning graph hooks](../core/engine-hooks.md) — they fire at **process-level** moments, not at every node or tool call.

```python
from promptise.runtime import HookManager, HookEvent

hooks = HookManager()

@hooks.on(HookEvent.USER_PROMPT_SUBMIT)
async def audit(ctx):
    await audit_log.write(ctx.data["prompt"])

@hooks.on(HookEvent.SESSION_START, once=True)
async def welcome(ctx):
    print("First session of the run!")
```

---

## HookEvent — 14 lifecycle events

| Event | When it fires |
|---|---|
| `SESSION_START` | Process enters RUNNING for the first time |
| `SESSION_END` | Process is about to stop |
| `USER_PROMPT_SUBMIT` | New prompt submitted by user or trigger |
| `PERMISSION_REQUEST` | Tool requiring approval is about to execute |
| `PERMISSION_DENIED` | Approval request was denied |
| `SUBAGENT_START` | Agent spawns a subagent |
| `SUBAGENT_STOP` | Spawned subagent finishes |
| `PRE_COMPACT` | Before conversation buffer compaction |
| `POST_COMPACT` | After compaction completes |
| `FILE_CHANGED` | Watched file on disk changes |
| `CONFIG_CHANGE` | Process config changes at runtime |
| `TASK_CREATED` | New background task enqueued |
| `TASK_COMPLETED` | Background task finishes |

---

## HookContext

Every hook callback receives a single `HookContext`:

```python
@dataclass
class HookContext:
    event: HookEvent
    process_id: str
    timestamp: datetime
    data: dict[str, Any]      # event payload — hooks may mutate this
    metadata: dict[str, Any]
```

For `USER_PROMPT_SUBMIT`, `data["prompt"]` contains the prompt text. Hooks can mutate `data` to alter downstream behavior.

---

## once: true — fire-and-forget hooks

Pass `once=True` to auto-deregister after the first invocation:

```python
@hooks.on(HookEvent.SESSION_START, once=True)
async def first_run_setup(ctx):
    await initialize_database()
```

The hook runs exactly once, then is removed. Useful for first-run setup, one-shot notifications, or "next time X happens, do Y" patterns.

---

## Priority ordering

Higher-priority hooks run first:

```python
hooks.register(HookEvent.USER_PROMPT_SUBMIT, audit_hook, priority=10)
hooks.register(HookEvent.USER_PROMPT_SUBMIT, transform_hook, priority=5)
hooks.register(HookEvent.USER_PROMPT_SUBMIT, log_hook, priority=0)
# Runs: audit → transform → log
```

---

## Blocking an action

Raise `HookBlocked` to cancel the action that triggered the hook:

```python
from promptise.runtime import HookBlocked

async def content_filter(ctx):
    if "DROP TABLE" in ctx.data.get("prompt", ""):
        raise HookBlocked("SQL injection attempt blocked")
```

The `HookManager.dispatch()` returns a `DispatchResult` with `blocked=True` so the caller can abort.

---

## Exception isolation

One broken hook never takes down the runtime. Exceptions (other than `HookBlocked`) are caught, logged, and recorded in `DispatchResult.errors`. The remaining hooks still run.

---

## Tag-based deregistration

Group hooks by tag for bulk removal:

```python
hooks.register(HookEvent.SESSION_START, h1, tag="audit")
hooks.register(HookEvent.SESSION_END, h2, tag="audit")

await hooks.remove_by_tag("audit")  # removes both
```

---

## ShellHook — delegate to an external command

`ShellHook` runs an external script with the event as JSON on stdin and parses JSON from stdout:

```python
from promptise.runtime import ShellHook

hook = ShellHook(command="./hooks/audit_prompt.sh", timeout=5.0)
hooks.register(HookEvent.USER_PROMPT_SUBMIT, hook.callback)
```

**JSON contract — stdin (sent to script):**
```json
{
  "event": "user_prompt_submit",
  "process_id": "support-bot",
  "timestamp": "2026-04-08T10:30:00+00:00",
  "data": {"prompt": "..."}
}
```

**JSON contract — stdout (from script, optional):**
```json
{
  "block": false,
  "reason": "",
  "data": {"prompt": "REWRITTEN"},
  "log": "anything"
}
```

- Return `{"block": true, "reason": "..."}` to block the action
- Return `{"data": {...}}` to replace the event's data
- Return `{}` or nothing for a no-op
- Non-zero exit or timeout = broken hook (logged, not fatal)

This lets you write hooks in Bash, Node, Go, or any language — and hot-reload them by editing the script.

The command is tokenized like a terminal would (`shlex.split`) and its
executable resolved on `PATH`; unless `cwd` is set, the process starts without
`fork()` (CPython's `posix_spawn` path). That matters on macOS: a forked child
of a multithreaded process that has used system frameworks — any `httpx`
client does — can die with SIGSEGV before `exec` (`exited -11`). Prefer an
absolute script path over `cwd` when you can.

---

## Related

- [Reasoning Graph Hooks](../core/engine-hooks.md) — per-node and per-tool hooks for the reasoning engine
- [Approval](../core/approval.md) — human-in-the-loop for tool calls
- [Events](../core/events.md) — structured event notifications
