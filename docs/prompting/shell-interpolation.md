# Shell Context Injection in Templates

Inject live system context into prompt templates by running shell commands at render time. The `!`cmd`` syntax executes a command and replaces itself with stdout.

```python
from promptise.prompts import TemplateEngine, SubprocessShellExecutor

engine = TemplateEngine(
    shell_executor=SubprocessShellExecutor(timeout=2.0),
)

out = engine.render(
    "Today is !`date +%Y-%m-%d`. Branch: !`git rev-parse --abbrev-ref HEAD`.",
    {},
)
# "Today is 2026-04-11. Branch: main."
```

---

## Disabled by default

Without an explicit `shell_executor`, the `!`cmd`` syntax is left as **literal text** — there is no implicit shell access, ever. This is a deliberate security choice.

```python
# No executor → syntax stays literal
engine = TemplateEngine()
out = engine.render("Today is !`date`", {})
# "Today is !`date`"
```

---

## SubprocessShellExecutor

The built-in executor runs commands via the system shell with a configurable timeout:

```python
from promptise.prompts import SubprocessShellExecutor

executor = SubprocessShellExecutor(
    timeout=5.0,          # max seconds per command (default 5)
    shell=True,           # run via /bin/sh -c (default True)
    cwd="/path/to/repo",  # working directory (default: inherited)
    env={"API_KEY": "x"}, # extra env vars (merged with os.environ)
    allowlist={"echo", "date", "git"},  # restrict allowed commands
)
```

With `shell=False` the command line is tokenized like a terminal would
(`shlex.split`) and its executable resolved on `PATH`. Unless `cwd` is set,
the process is started without `fork()` (CPython's `posix_spawn` path): a
forked child of a multithreaded process that has used macOS system frameworks
— any `httpx` client does — can die with SIGSEGV before `exec`, which surfaces
as `exited -11`. Setting `cwd` forces the fork path, so prefer running the
command with an absolute path over changing directory.

### Allowlist

When `allowlist` is set, only commands whose first token matches are permitted:

```python
executor = SubprocessShellExecutor(allowlist={"echo", "git"})

# Works
engine = TemplateEngine(shell_executor=executor)
engine.render("!`echo hello`", {})

# Raises ShellExecutionError
engine.render("!`rm -rf /`", {})
```

---

## Custom executors

Any callable `(str) -> str` works as a shell executor:

```python
# Dry-run executor for testing
engine = TemplateEngine(shell_executor=lambda cmd: f"[would run: {cmd}]")

# Sandbox executor
async def sandboxed(cmd: str) -> str:
    return await sandbox.execute(cmd)
```

---

## Processing order

Shell interpolation runs **after** template includes and **before** conditionals and variable substitution. This means:

1. `{% include "name" %}` — resolved first (included templates can contain `!`cmd``)
2. `!`cmd`` — executed, stdout replaces the block
3. `{% if %}`, `{% for %}` — conditionals and loops
4. `{variable}` — variable interpolation

---

## Error handling

- **Non-zero exit code** → raises `ShellExecutionError`
- **Timeout** → raises `ShellExecutionError`
- **Command not in allowlist** → raises `ShellExecutionError`

All errors propagate to the caller — there is no silent fallback.

---

## Related

- [Loader & Templates](loader-templates.md) — the full template engine reference
- [Context & Variables](context.md) — dynamic context injection via providers
