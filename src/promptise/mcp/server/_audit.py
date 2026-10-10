"""Audit logging middleware for MCP servers.

Produces a tamper-evident HMAC-chained audit log of tool calls.
Each entry includes a hash of the previous entry, forming an
integrity chain.  A server that restarts on the same file continues
the chain from the file's last entry, so one key verifies the whole
file with :func:`~promptise.mcp.server.verify_audit_log` (or
``promptise audit verify FILE``).

Example::

    from promptise.mcp.server import MCPServer, AuditMiddleware

    server = MCPServer(name="api")
    server.add_middleware(AuditMiddleware(
        log_path="audit.jsonl",
        signed=True,            # key from PROMPTISE_AUDIT_SECRET
        include_args=True,
        include_result=False,
    ))
"""

from __future__ import annotations

import asyncio
import collections
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from collections.abc import Callable
from typing import Any

from ._context import RequestContext

logger = logging.getLogger("promptise.server")

#: ``prev_hash`` of the first entry of a chain.
GENESIS_HASH = "0" * 64

_TAIL_CHUNK = 64 * 1024


def canonical_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Return *entry* exactly as it reads back from the JSONL file.

    Values JSON cannot hold (sets, models, datetimes) become strings, tuples
    become lists and dict keys become strings.  Signing this form means the
    HMAC is computed over the same data a verifier reads from the file — a
    tool argument such as ``{1: "a", 10: "b"}`` used to be signed with its
    keys sorted as integers and verified with them sorted as strings.
    """
    result: dict[str, Any] = json.loads(json.dumps(entry, default=str))
    return result


def entry_mac(secret: bytes, entry: dict[str, Any]) -> str:
    """HMAC-SHA256 of *entry* without its ``hmac`` field (hex digest)."""
    unsigned = {k: v for k, v in entry.items() if k != "hmac"}
    payload = json.dumps(unsigned, sort_keys=True)
    return hmac.new(secret, payload.encode(), hashlib.sha256).hexdigest()


def _read_tail(path: str) -> tuple[bytes | None, bool]:
    """Read the last complete line of *path*.

    Returns ``(line, ends_with_newline)``: *line* is the last line that has
    its newline (without it), or ``None`` when the file is empty or holds
    only an unterminated fragment.  ``ends_with_newline`` is ``False`` when
    the file ends in a partial line, e.g. after a crash during a write.
    """
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size == 0:
            return None, True
        f.seek(size - 1)
        ends_with_newline = f.read(1) == b"\n"
        data = b""
        pos = size
        while pos > 0:
            step = min(_TAIL_CHUNK, pos)
            pos -= step
            f.seek(pos)
            data = f.read(step) + data
            end = data.rfind(b"\n")
            if end == -1:
                continue
            start = data.rfind(b"\n", 0, end)
            if start != -1 or pos == 0:
                return data[start + 1 : end], ends_with_newline
        return None, ends_with_newline


class AuditMiddleware:
    """HMAC-chained audit log middleware.

    Writes one JSON line per tool call to ``log_path`` with optional
    argument/result capture and HMAC chain integrity.

    Each signed entry carries ``seq`` (its position in the chain),
    ``prev_hash`` (the previous entry's ``hmac``, or 64 zeros for the
    first entry) and ``hmac``.  When ``log_path`` already holds entries,
    the first entry this instance writes continues the chain from the
    file's last entry and is marked ``"resumed": true``.  Run one writer
    per file: two processes appending to the same file fork the chain.

    Args:
        log_path: File path for audit log (JSONL format).  If ``None``,
            entries are only kept in memory and emitted via Python logging.
        signed: Enable HMAC chain (default ``True``).
        hmac_secret: Secret key for HMAC chain.  Resolved from (in order):
            1. This parameter
            2. ``PROMPTISE_AUDIT_SECRET`` env var
            3. Without ``log_path`` only: an auto-generated random secret
               (logged as warning).  A signed log *file* needs a key you
               keep, or nobody could ever verify it, so ``log_path`` with
               ``signed=True`` and no key raises ``ValueError``.
        include_args: Log tool arguments (default ``False`` — may
            contain PII).
        include_result: Log tool results (default ``False``).
        max_memory_entries: How many entries :attr:`entries` keeps in
            memory (default 10 000; ``None`` keeps all).  The file keeps
            everything.

    Raises:
        ValueError: ``log_path`` is set with ``signed=True`` and no key is
            configured.
    """

    def __init__(
        self,
        log_path: str | None = None,
        *,
        signed: bool = True,
        hmac_secret: str | None = None,
        include_args: bool = False,
        include_result: bool = False,
        max_memory_entries: int | None = 10_000,
    ) -> None:
        self._log_path = log_path
        self._signed = signed
        resolved_secret = hmac_secret or os.environ.get("PROMPTISE_AUDIT_SECRET")
        if not resolved_secret:
            if signed and log_path is not None:
                raise ValueError(
                    "AuditMiddleware: a signed audit log file needs a key you keep. "
                    "Set PROMPTISE_AUDIT_SECRET or pass hmac_secret= (or signed=False "
                    "for an unsigned log). A random key would make the file "
                    "impossible to verify."
                )
            resolved_secret = secrets.token_hex(32)
            if signed:
                logger.warning(
                    "AuditMiddleware: no hmac_secret or PROMPTISE_AUDIT_SECRET set. "
                    "Using an auto-generated secret for the in-memory chain; it "
                    "cannot be verified after this process exits."
                )
        self._secret = resolved_secret.encode()
        self._include_args = include_args
        self._include_result = include_result
        self._prev_hash: str = GENESIS_HASH
        self._seq = 0
        self._resume_pending = log_path is not None
        self._mark_resumed = False
        self._needs_newline = False
        self._entries: collections.deque[dict[str, Any]] = collections.deque(
            maxlen=max_memory_entries
        )
        # prev_hash of the oldest entry still in memory (verify_chain anchor)
        self._memory_anchor = GENESIS_HASH
        self._chain_lock = asyncio.Lock()

    async def __call__(self, ctx: RequestContext, call_next: Callable[..., Any]) -> Any:
        start = time.perf_counter()
        error: str | None = None
        result: Any = None

        try:
            result = await call_next(ctx)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # Build, sign and write under one lock: the file holds entries in
            # chain order, and the chain only advances once an entry is
            # written (a failed write would otherwise leave a gap that looks
            # exactly like a deleted entry).
            async with self._chain_lock:
                loop = asyncio.get_running_loop()
                if self._resume_pending:
                    try:
                        await loop.run_in_executor(None, self._resume_from_file)
                    except OSError as exc:
                        logger.error("Audit log %s unreadable: %s", self._log_path, exc)
                entry = self._build_entry(ctx, start, error, result)
                written = True
                if self._log_path:
                    line = json.dumps(entry) + "\n"
                    if self._needs_newline:
                        # Close the partial line a crash left behind, so it
                        # stays a line of its own (the verifier reports it).
                        line = "\n" + line
                    try:
                        await loop.run_in_executor(None, self._write_log_line, line)
                        self._needs_newline = False
                    except OSError as exc:
                        written = False
                        logger.error(
                            "Audit log write failed, entry not recorded "
                            "(tool=%s client=%s request_id=%s): %s",
                            ctx.tool_name,
                            ctx.client_id,
                            ctx.request_id,
                            exc,
                        )
                if written:
                    self._commit(entry)

            logger.info(
                "AUDIT: tool=%s client=%s status=%s duration=%.3fs",
                ctx.tool_name,
                ctx.client_id,
                "error" if error else "ok",
                entry["duration_s"],
            )

    def _commit(self, entry: dict[str, Any]) -> None:
        """Advance the chain past *entry* and keep it in memory."""
        if self._signed:
            self._prev_hash = entry["hmac"]
            self._seq += 1
            self._mark_resumed = False
        maxlen = self._entries.maxlen
        if maxlen is not None and len(self._entries) == maxlen:
            if maxlen == 0:
                return
            self._memory_anchor = self._entries[0].get("hmac", GENESIS_HASH)
        self._entries.append(entry)

    def _resume_from_file(self) -> None:
        """Continue the chain from the last entry already in ``log_path``.

        Runs once, before the first write.  An unterminated last line (a
        crash during a write) is kept: the next write starts on a new line,
        and the chain continues from the last complete entry.
        """
        assert self._log_path is not None
        self._resume_pending = False
        try:
            last_line, ends_with_newline = _read_tail(self._log_path)
        except FileNotFoundError:
            return
        self._needs_newline = not ends_with_newline
        # The first entry of this run says the server restarted on this file.
        self._mark_resumed = self._signed and (last_line is not None or not ends_with_newline)
        if not ends_with_newline:
            logger.warning(
                "Audit log %s ends in an incomplete line (crash during a write?). "
                "It is kept; the chain continues from the last complete entry.",
                self._log_path,
            )
        if last_line is None or not self._signed:
            return
        try:
            last = json.loads(last_line)
        except ValueError:
            last = None
        if not isinstance(last, dict) or not isinstance(last.get("hmac"), str):
            logger.warning(
                "Audit log %s: the last line is not a signed entry, so the chain "
                "cannot be continued; starting a new chain (the verifier reports "
                "it as a chain reset).",
                self._log_path,
            )
            return
        if not hmac.compare_digest(entry_mac(self._secret, last), last["hmac"]):
            logger.warning(
                "Audit log %s: its last entry was not signed with the current key "
                "(key rotated?). The chain continues, and verifying the file needs "
                "both keys (verify_audit_log(path, [new_key, old_key])).",
                self._log_path,
            )
        self._prev_hash = last["hmac"]
        seq = last.get("seq")
        self._seq = seq + 1 if isinstance(seq, int) and not isinstance(seq, bool) else 0
        self._memory_anchor = self._prev_hash

    def _write_log_line(self, line: str) -> None:
        """Append one line to the audit log file (runs in executor).

        One ``O_APPEND`` write per entry, so concurrent appenders never land
        inside each other's line.
        """
        assert self._log_path is not None
        data = line.encode()
        fd = os.open(self._log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
        finally:
            os.close(fd)

    def _build_entry(
        self,
        ctx: RequestContext,
        start: float,
        error: str | None,
        result: Any,
    ) -> dict[str, Any]:
        duration = time.perf_counter() - start
        entry: dict[str, Any] = {
            "timestamp": time.time(),
            "tool": ctx.tool_name,
            "client_id": ctx.client_id,
            "request_id": ctx.request_id,
            "status": "error" if error else "ok",
            "duration_s": round(duration, 4),
        }

        # Resource reads and prompt requests are audited too; say which.
        request_type = getattr(ctx, "request_type", "tool")
        if request_type != "tool":
            entry["request_type"] = request_type
            if ctx.state.get("resource_uri"):
                entry["uri"] = ctx.state["resource_uri"]

        if error:
            entry["error"] = error

        if self._include_args:
            # The validated arguments the caller sent (set by the server
            # before the middleware chain runs) — not ctx.state, which holds
            # the tool definition and whatever other middleware stored there.
            arguments = ctx.state.get("_tool_arguments")
            entry["args"] = dict(arguments) if isinstance(arguments, dict) else {}

        if self._include_result and result is not None and not error:
            try:
                entry["result"] = str(result)[:1000]  # Truncate large results
            except Exception:
                entry["result"] = "<unserializable>"

        # Record the verified identity of the acting agent (from a JWT/JWKS
        # auth provider), so the tamper-evident log answers "which agent did
        # what" — not just a client_id string. Only identity *descriptors* are
        # included (subject, issuer, audience, roles); never the token or the
        # full claim set, which may carry sensitive data.
        identity = self._identity_fields(ctx)
        if identity:
            entry["identity"] = identity

        entry = canonical_entry(entry)
        if self._signed:
            entry["seq"] = self._seq
            if self._mark_resumed:
                entry["resumed"] = True
            entry["prev_hash"] = self._prev_hash
            entry["hmac"] = entry_mac(self._secret, entry)

        return entry

    @staticmethod
    def _identity_fields(ctx: RequestContext) -> dict[str, Any]:
        """Extract verified-identity descriptors from ``ctx.client``.

        Returns the acting agent's ``subject`` / ``issuer`` / ``audience`` /
        ``roles`` when present (JWT or JWKS auth), or an empty dict for
        unauthenticated or API-key calls. Never includes the token or the
        full claim set.
        """
        client = getattr(ctx, "client", None)
        if client is None:
            return {}
        identity: dict[str, Any] = {}
        if getattr(client, "subject", None):
            identity["subject"] = client.subject
        if getattr(client, "issuer", None):
            identity["issuer"] = client.issuer
        if getattr(client, "audience", None):
            identity["audience"] = client.audience
        if getattr(client, "tenant_id", None):
            identity["tenant_id"] = client.tenant_id
        roles = getattr(client, "roles", None)
        if roles:
            identity["roles"] = sorted(roles)
        return identity

    @property
    def entries(self) -> list[dict[str, Any]]:
        """The most recent audit entries kept in memory (useful for testing).

        At most ``max_memory_entries``; the log file keeps every entry.
        """
        return list(self._entries)

    def verify_chain(self) -> bool:
        """Verify the HMAC chain of the entries kept in memory.

        Checks the entries this instance wrote (the most recent
        ``max_memory_entries``).  To verify a log *file* — every entry,
        across restarts, with the reason and line of the first break — use
        :func:`~promptise.mcp.server.verify_audit_log`.

        Returns:
            ``True`` if the chain is valid, ``False`` if tampered.
        """
        if not self._signed:
            return True
        prev_hash = self._memory_anchor
        for entry in self._entries:
            stored_hmac = entry.get("hmac")
            if not isinstance(stored_hmac, str) or entry.get("prev_hash") != prev_hash:
                return False
            if not hmac.compare_digest(entry_mac(self._secret, entry), stored_hmac):
                return False
            prev_hash = stored_hmac
        return True
