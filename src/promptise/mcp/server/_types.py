"""Server-specific types and enums for the MCP Server Framework."""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from mcp.types import (
    EmbeddedResource as MCPEmbeddedResource,
)
from mcp.types import (
    ImageContent as MCPImageContent,
)
from mcp.types import (
    TextContent as MCPTextContent,
)


class TransportType(str, Enum):
    """Supported MCP transport protocols."""

    STDIO = "stdio"
    HTTP = "http"
    SSE = "sse"


# ------------------------------------------------------------------
# Structured content helpers
# ------------------------------------------------------------------


class ImageContent:
    """Helper for returning image content from tool handlers.

    Wraps binary image data for automatic serialization to MCP
    ``ImageContent`` (base64-encoded).

    Args:
        data: Raw image bytes.
        mime_type: MIME type (e.g. ``"image/png"``, ``"image/jpeg"``).

    Example::

        @server.tool()
        async def chart(values: list[float]) -> ImageContent:
            png = render_chart(values)
            return ImageContent(data=png, mime_type="image/png")
    """

    __slots__ = ("data", "mime_type")

    def __init__(self, data: bytes, *, mime_type: str = "image/png") -> None:
        self.data = data
        self.mime_type = mime_type

    def to_mcp(self) -> MCPImageContent:
        """Convert to an MCP ``ImageContent`` message."""
        return MCPImageContent(
            type="image",
            data=base64.b64encode(self.data).decode(),
            mimeType=self.mime_type,
        )


#: Union of content types that tools can return directly.
#: Tools may return a single content item or a list of them for
#: mixed-content responses (e.g. text + image).
Content = MCPTextContent | MCPImageContent | MCPEmbeddedResource | ImageContent


# ------------------------------------------------------------------
# Tool annotations
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ToolAnnotations:
    """MCP tool annotations describing tool behaviour hints.

    All fields are optional hints — they are **not** enforced by the
    framework but communicated to MCP clients so they can make informed
    decisions (e.g. skipping confirmation for read-only tools).

    Attributes:
        title: Human-readable title for the tool.
        read_only_hint: Tool does not modify any state.
        destructive_hint: Tool may perform destructive operations.
        idempotent_hint: Calling the tool multiple times with the same
            arguments produces the same result.
        open_world_hint: Tool may interact with external systems
            beyond the server's control.
    """

    title: str | None = None
    read_only_hint: bool | None = None
    destructive_hint: bool | None = None
    idempotent_hint: bool | None = None
    open_world_hint: bool | None = None


@dataclass(frozen=True)
class ToolDef:
    """Internal definition of a registered tool."""

    name: str
    description: str
    handler: Any  # Callable
    input_schema: dict[str, Any]
    tags: list[str] = field(default_factory=list)
    auth: bool = False
    rate_limit: str | None = None
    timeout: float | None = None
    guards: list[Any] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    router_middleware: list[Any] = field(default_factory=list)
    annotations: ToolAnnotations | None = None
    max_concurrent: int | None = None
    requires_approval: bool = False


@dataclass(frozen=True)
class ResourceDef:
    """Internal definition of a registered resource or resource template.

    The access-control fields mirror :class:`ToolDef`: a read runs through
    the server middleware chain with this definition in
    ``ctx.state["tool_def"]``, so ``AuthMiddleware``, guards, rate limits
    and timeouts apply to resources exactly as they do to tools.
    """

    uri: str
    name: str
    description: str
    handler: Any  # Callable
    mime_type: str = "text/plain"
    is_template: bool = False
    tags: list[str] = field(default_factory=list)
    auth: bool = False
    rate_limit: str | None = None
    timeout: float | None = None
    guards: list[Any] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    router_middleware: list[Any] = field(default_factory=list)
    # Pydantic model that coerces template parameters (always strings in
    # the URI) to the handler's type hints.  ``None`` for a static resource.
    input_model: Any = None


@dataclass(frozen=True)
class PromptDef:
    """Internal definition of a registered prompt.

    The access-control fields mirror :class:`ToolDef` (see
    :class:`ResourceDef`).
    """

    name: str
    description: str
    handler: Any  # Callable
    arguments: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    auth: bool = False
    rate_limit: str | None = None
    timeout: float | None = None
    guards: list[Any] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    router_middleware: list[Any] = field(default_factory=list)
    # Pydantic model that coerces the string-valued MCP prompt arguments to
    # the handler's type hints.  ``None`` passes arguments through unchanged.
    input_model: Any = None
