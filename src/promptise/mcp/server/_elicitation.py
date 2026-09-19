"""Elicitation support for MCP servers.

Allows tools to request structured input from the user during
execution via the MCP elicitation protocol.

Example::

    from promptise.mcp.server import MCPServer, Elicitor, Depends

    server = MCPServer(name="deploy")

    @server.tool()
    async def deploy(
        env: str,
        elicit: Elicitor = Depends(Elicitor),
    ) -> str:
        answer = await elicit.ask(
            message=f"Deploy to {env}?",
            schema={"type": "object", "properties": {"confirm": {"type": "boolean"}}},
        )
        if not answer or not answer.get("confirm"):
            return "Cancelled"
        return f"Deployed to {env}"
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger("promptise.server")


class Elicitor:
    """Request structured input from the user mid-execution.

    Bound to the MCP session by the framework's DI wiring.  Sends an
    ``elicitation/create`` request through ``ServerSession.elicit`` and
    resolves the client's :class:`mcp.types.ElicitResult`.

    ``ask()`` returns ``None`` — never raises — when the answer is not a
    usable acceptance: the client declined or cancelled, did not declare
    the elicitation capability (the request is rejected by the client),
    the request timed out, or the transport failed.  Every one of these
    is logged at WARNING/INFO level so an operator can tell *why* a gated
    call was denied instead of guessing.

    Args:
        timeout: Default timeout in seconds for elicitation requests.
    """

    def __init__(self, timeout: float = 60.0) -> None:
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got {timeout}")
        self._session: Any = None
        self._request_id: str | int | None = None
        self._timeout = timeout

    def _bind(self, session: Any, request_id: str | int | None = None) -> None:
        """Bind to the current MCP session (called by framework)."""
        self._session = session
        self._request_id = request_id

    async def ask(
        self,
        message: str,
        schema: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any] | None:
        """Ask the user for structured input.

        Args:
            message: Human-readable prompt.
            schema: JSON Schema for the expected response.  Defaults to an
                empty object schema (a bare confirmation).
            timeout: Override the default timeout, in seconds.

        Returns:
            The submitted form values when the user accepted, or ``None``
            when the elicitor is unbound, the client declined or cancelled,
            the client does not support elicitation, the request timed
            out, or the transport failed.
        """
        if self._session is None:
            return None

        effective_timeout = self._timeout if timeout is None else timeout
        requested_schema = schema or {"type": "object", "properties": {}}
        try:
            # ``ServerSession.elicit`` is the SDK entry point for form-mode
            # elicitation across the pinned mcp range; the contract test in
            # tests/test_approval_gate.py fails loudly if the SDK renames it.
            result = await asyncio.wait_for(
                self._session.elicit(
                    message=message,
                    requestedSchema=requested_schema,
                    related_request_id=self._request_id,
                ),
                timeout=effective_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Elicitation timed out after %.0fs (request_id=%s) — treated as declined",
                effective_timeout,
                self._request_id,
            )
            return None
        except Exception as exc:
            # McpError when the client rejects the request (no elicitation
            # capability), or a transport failure. Either way there is no
            # answer — callers treat None as a decline.
            logger.warning(
                "Elicitation request failed (%s: %s) — treated as declined",
                type(exc).__name__,
                exc,
            )
            return None

        action = getattr(result, "action", None)
        if action != "accept":
            logger.info("Elicitation not accepted by client (action=%r)", action)
            return None
        content = getattr(result, "content", None)
        if not isinstance(content, dict):
            logger.warning(
                "Elicitation accepted without form content (%s) — treated as declined",
                type(content).__name__,
            )
            return None
        return dict(content)
