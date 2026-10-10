"""Audit logging middleware for MCP servers.

Produces a tamper-evident HMAC-chained audit log of tool calls.
Each entry includes a hash of the previous entry, forming an
integrity chain.

Example::

    from promptise.mcp.server import MCPServer, AuditMiddleware

    server = MCPServer(name="api")
    server.add_middleware(AuditMiddleware(
        log_path="audit.jsonl",
        signed=True,
        include_args=True,
        include_result=False,
    ))
"""

from __future__ import annotations

import asyncio
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


class AuditMiddleware:
    """HMAC-chained audit log middleware.

    Writes one JSON line per tool call to ``log_path`` with optional
    argument/result capture and HMAC chain integrity.

    Args:
        log_path: File path for audit log (JSONL format).  If ``None``,
            entries are only emitted via Python logging.
        signed: Enable HMAC chain (default ``True``).
        hmac_secret: Secret key for HMAC chain.  Resolved from (in order):
            1. This parameter
            2. ``PROMPTISE_AUDIT_SECRET`` env var
            3. Auto-generated random secret (logged as warning)
        include_args: Log tool arguments (default ``False`` — may
            contain PII).
        include_result: Log tool results (default ``False``).
    """

    def __init__(
        self,
        log_path: str | None = None,
        *,
        signed: bool = True,
        hmac_secret: str | None = None,
        include_args: bool = False,
        include_result: bool = False,
    ) -> None:
        self._log_path = log_path
        self._signed = signed
        resolved_secret = hmac_secret or os.environ.get("PROMPTISE_AUDIT_SECRET")
        if resolved_secret is None:
            resolved_secret = secrets.token_hex(32)
            logger.warning(
                "AuditMiddleware: no hmac_secret or PROMPTISE_AUDIT_SECRET set. "
                "Using auto-generated secret. Audit chain cannot be verified "
                "across restarts. Set PROMPTISE_AUDIT_SECRET for production."
            )
        self._secret = resolved_secret.encode()
        self._include_args = include_args
        self._include_result = include_result
        self._prev_hash: str = "0" * 64  # genesis hash
        self._entries: list[dict[str, Any]] = []  # in-memory buffer for testing
        self._chain_lock = asyncio.Lock()

    async def __call__(self, ctx: RequestContext, call_next: Callable[..., Any]) -> Any:
        start = time.time()
        error: str | None = None
        result: Any = None

        try:
            result = await call_next(ctx)
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            # Serialize chain updates to prevent concurrent corruption
            async with self._chain_lock:
                entry = self._build_entry(ctx, start, error, result)
                self._entries.append(entry)

            if self._log_path:
                try:
                    line = json.dumps(entry, default=str) + "\n"
                    await asyncio.get_running_loop().run_in_executor(
                        None, self._write_log_line, line
                    )
                except OSError as exc:
                    logger.warning("Audit log write failed: %s", exc)

            logger.info(
                "AUDIT: tool=%s client=%s status=%s duration=%.3fs",
                ctx.tool_name,
                ctx.client_id,
                "error" if error else "ok",
                entry["duration_s"],
            )

    def _write_log_line(self, line: str) -> None:
        """Write a single line to the audit log file (runs in executor).

        Caller guarantees ``self._log_path`` is set before scheduling this task
        (see the ``if self._log_path:`` guard at the call site).
        """
        assert self._log_path is not None
        with open(self._log_path, "a") as f:
            f.write(line)

    def _build_entry(
        self,
        ctx: RequestContext,
        start: float,
        error: str | None,
        result: Any,
    ) -> dict[str, Any]:
        duration = time.time() - start
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
            entry["args"] = {k: v for k, v in ctx.state.items() if not k.startswith("_")}

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

        if self._signed:
            entry["prev_hash"] = self._prev_hash
            payload = json.dumps(entry, sort_keys=True, default=str)
            entry["hmac"] = hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()
            self._prev_hash = entry["hmac"]

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
        """Access in-memory audit entries (useful for testing)."""
        return self._entries

    def verify_chain(self) -> bool:
        """Verify the HMAC chain integrity of logged entries.

        Returns:
            ``True`` if the chain is valid, ``False`` if tampered.
        """
        prev_hash = "0" * 64
        for entry in self._entries:
            if not self._signed:
                continue
            stored_hmac = entry.get("hmac")
            stored_prev = entry.get("prev_hash")
            if stored_prev != prev_hash:
                return False
            # Reconstruct payload without hmac field
            check_entry = {k: v for k, v in entry.items() if k != "hmac"}
            payload = json.dumps(check_entry, sort_keys=True, default=str)
            expected = hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()
            if expected != stored_hmac:
                return False
            prev_hash = stored_hmac
        return True
