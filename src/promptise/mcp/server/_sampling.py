"""Sampling support for MCP servers.

Allows tools to request LLM completions from the client's model
via the MCP sampling protocol.

Example::

    from promptise.mcp.server import MCPServer, Sampler, Depends

    server = MCPServer(name="assistant")

    @server.tool()
    async def summarize(
        text: str,
        sampler: Sampler = Depends(Sampler),
    ) -> str:
        result = await sampler.create_message(
            messages=[{"role": "user", "content": f"Summarize: {text}"}],
            max_tokens=500,
        )
        return result or "Sampling unavailable"
"""

from __future__ import annotations

import logging
from typing import Any, Literal

logger = logging.getLogger("promptise.server")


class Sampler:
    """Request LLM completions from the client.

    Bound to the MCP session by the framework's DI wiring.
    If the client does not support sampling, methods return
    ``None`` silently.
    """

    def __init__(self) -> None:
        self._session: Any = None
        self._request_id: str | int | None = None

    def _bind(self, session: Any, request_id: str | int | None = None) -> None:
        """Bind to the current MCP session (called by framework)."""
        self._session = session
        self._request_id = request_id

    async def create_message(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        model: str | None = None,
        system: str | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
    ) -> str | None:
        """Request an LLM completion from the client.

        Args:
            messages: List of message dicts with ``role`` and ``content``.
            max_tokens: Maximum tokens to generate.
            model: Model hint (client may ignore).
            system: System prompt.
            temperature: Sampling temperature.
            stop_sequences: Stop sequences.

        Returns:
            Generated text, or ``None`` if unavailable.
        """
        if self._session is None:
            return None

        try:
            # Build MCP sampling message format
            from mcp.types import SamplingMessage, TextContent

            mcp_messages = []
            for msg in messages:
                _role = msg.get("role", "user")
                # MCP protocol only permits "user" or "assistant" roles.
                role: Literal["user", "assistant"] = "assistant" if _role == "assistant" else "user"
                mcp_messages.append(
                    SamplingMessage(
                        role=role,
                        content=TextContent(type="text", text=msg.get("content", "")),
                    )
                )

            # The SDK's ServerSession.create_message signature (mcp>=1.9): the model
            # hint travels as ModelPreferences and the system prompt as system_prompt.
            from mcp.types import ModelHint, ModelPreferences

            result = await self._session.create_message(
                mcp_messages,
                max_tokens=max_tokens,
                system_prompt=system,
                temperature=temperature,
                stop_sequences=stop_sequences,
                model_preferences=(
                    ModelPreferences(hints=[ModelHint(name=model)]) if model else None
                ),
            )

            if result is None:
                return None
            if hasattr(result, "content"):
                content = result.content
                if isinstance(content, str):
                    return content
                if hasattr(content, "text"):
                    return content.text
                return str(content)
            return str(result)
        except (ImportError, AttributeError) as exc:
            logger.warning("Sampling not supported by client session: %s", exc)
            return None
        except Exception as exc:
            logger.warning("Sampling failed: %s: %s", type(exc).__name__, exc)
            return None
