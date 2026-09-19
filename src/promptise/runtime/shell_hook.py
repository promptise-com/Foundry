"""ShellHook — runtime hook backed by an external command.

A :class:`ShellHook` is a runtime lifecycle hook that delegates the
work to an external program. The dispatcher serializes the
:class:`HookContext` to JSON, writes it to the program's stdin, runs
the program with a configurable timeout, and parses one line of JSON
back from stdout.

Use this when:

* You want to write a hook in a non-Python language (Bash, Node, Go).
* The hook depends on external state that's easier to manage in a
  separate process.
* You want hot-reloadable hooks: just edit the script.

The JSON contract:

**Stdin (sent to the script):**

::

    {
      "event": "user_prompt_submit",
      "process_id": "support-bot",
      "timestamp": "2026-04-08T10:30:00+00:00",
      "data": {"prompt": "..."},
      "metadata": {}
    }

**Stdout (read from the script, optional):**

::

    {
      "block": false,            // optional, default false
      "reason": "",              // optional, used if block=true
      "data": {"prompt": "..."}, // optional, replaces ctx.data
      "log": "anything"          // optional, free-form
    }

If the script writes nothing, the hook is treated as a no-op success.
If it exits non-zero or times out, the hook is treated as a *broken
hook*: the runtime logs the error and continues. (Hook errors never
take down a runtime — see :class:`HookManager.dispatch`.)

To block an action via shell hook, output ``{"block": true,
"reason": "..."}``. The runtime will raise :class:`HookBlocked`
internally and the hook manager returns a ``blocked=True`` dispatch
result.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import shlex
from dataclasses import dataclass
from typing import Any

from .._spawn import resolve_executable, spawn_options
from .hooks import HookBlocked, HookContext

# Transient fork/spawn failures (EAGAIN, etc.) under heavy load are worth a
# brief retry instead of failing the hook outright.
_TRANSIENT_SPAWN_ERRNOS = {errno.EAGAIN, errno.EWOULDBLOCK, errno.ENOMEM}


logger = logging.getLogger(__name__)


@dataclass
class ShellHookResult:
    """Parsed result returned by an external shell hook."""

    block: bool = False
    reason: str = ""
    data: dict[str, Any] | None = None
    log: str = ""
    raw: str = ""


@dataclass
class ShellHook:
    """Runtime hook that shells out to an external command.

    The command receives a JSON event on stdin and may write a JSON
    response to stdout. Use ``ShellHook.callback`` as the
    ``callback`` argument to :meth:`HookManager.register`.

    Attributes:
        command: Shell command line. Tokenized with :func:`shlex.split`
            when ``shell=False`` (default), so quoting works as in a
            terminal. Pass ``shell=True`` to run via ``/bin/sh -c``
            (only do this if you trust the command source).
        timeout: Max seconds the command may run before being killed.
            Default 5.0.
        cwd: Working directory for the subprocess. Default current
            directory.
        env: Optional environment variables; merged on top of the
            inherited environment.
        shell: If True, run via the shell instead of an exec.
        encoding: stdin/stdout encoding. Default "utf-8".

    Example::

        hook = ShellHook(command="./hooks/audit_prompt.sh")
        hooks.register(HookEvent.USER_PROMPT_SUBMIT, hook.callback)

    The script ``audit_prompt.sh`` might look like::

        #!/usr/bin/env bash
        # Read JSON from stdin, log the prompt, allow it through.
        prompt=$(jq -r .data.prompt)
        echo "[$(date)] $prompt" >> /var/log/agent_prompts.log
        echo '{}'
    """

    command: str
    timeout: float = 5.0
    cwd: str | None = None
    env: dict[str, str] | None = None
    shell: bool = False
    encoding: str = "utf-8"

    async def callback(self, ctx: HookContext) -> None:
        """Async hook callback — pass this to :meth:`HookManager.register`.

        Serializes the context, runs the command, parses any JSON
        reply, and applies its directives (mutate ``ctx.data``, raise
        :class:`HookBlocked`).
        """
        result = await self.run(ctx)

        if result.data is not None:
            # Replace ctx.data in place so downstream code sees mutations
            ctx.data.clear()
            ctx.data.update(result.data)

        if result.log:
            logger.debug("shell hook for %s logged: %s", ctx.event.value, result.log)

        if result.block:
            raise HookBlocked(result.reason or "blocked by shell hook")

    async def run(self, ctx: HookContext) -> ShellHookResult:
        """Execute the underlying command and return the parsed result.

        Public so it can be unit-tested or invoked manually.
        """
        payload = json.dumps(
            {
                "event": ctx.event.value,
                "process_id": ctx.process_id,
                "timestamp": ctx.timestamp.isoformat(),
                "data": ctx.data,
                "metadata": ctx.metadata,
            }
        ).encode(self.encoding)

        args: list[str] | None = None
        if not self.shell:
            args = shlex.split(self.command)
            if not args:
                raise ValueError("ShellHook.command is empty")
            args = resolve_executable(args)
        options = spawn_options(self.cwd)

        # Spawn the subprocess, retrying a transient fork/spawn failure (e.g.
        # EAGAIN, "Resource temporarily unavailable") that can occur under heavy
        # concurrent load rather than failing the hook outright.
        proc = None
        for attempt in range(3):
            try:
                if self.shell:
                    proc = await asyncio.create_subprocess_shell(
                        self.command,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=self.cwd,
                        env=self._merged_env(),
                        **options,
                    )
                else:
                    proc = await asyncio.create_subprocess_exec(
                        *args,  # type: ignore[misc]
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        cwd=self.cwd,
                        env=self._merged_env(),
                        **options,
                    )
                break
            except OSError as exc:
                if exc.errno in _TRANSIENT_SPAWN_ERRNOS and attempt < 2:
                    await asyncio.sleep(0.2 * (attempt + 1))
                    continue
                raise
        assert proc is not None  # nosec B101 - loop either sets proc or raises

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(input=payload), timeout=self.timeout
            )
        except asyncio.TimeoutError as exc:
            try:
                proc.kill()
            finally:
                await proc.wait()
            raise TimeoutError(
                f"shell hook '{self.command}' exceeded timeout of {self.timeout}s"
            ) from exc

        if proc.returncode != 0:
            stderr_text = stderr_bytes.decode(self.encoding, errors="replace")
            raise RuntimeError(
                f"shell hook '{self.command}' exited {proc.returncode}: {stderr_text.strip()[:500]}"
            )

        stdout_text = stdout_bytes.decode(self.encoding, errors="replace").strip()
        if not stdout_text:
            return ShellHookResult(raw="")

        try:
            payload_out = json.loads(stdout_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"shell hook '{self.command}' produced invalid JSON: {stdout_text[:200]}"
            ) from exc

        if not isinstance(payload_out, dict):
            raise ValueError(
                f"shell hook '{self.command}' must return a JSON object, got "
                f"{type(payload_out).__name__}"
            )

        return ShellHookResult(
            block=bool(payload_out.get("block", False)),
            reason=str(payload_out.get("reason", "")),
            data=payload_out.get("data"),
            log=str(payload_out.get("log", "")),
            raw=stdout_text,
        )

    def _merged_env(self) -> dict[str, str] | None:
        if self.env is None:
            return None
        import os

        merged = dict(os.environ)
        merged.update(self.env)
        return merged


__all__ = ["ShellHook", "ShellHookResult"]
