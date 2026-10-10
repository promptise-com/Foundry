"""Structured MCP errors that LLMs can understand.

When raised inside a tool handler, the server answers the call with a
``CallToolResult`` that has ``isError=True`` and structured
``{"error": {...}}`` text content that helps the LLM recover.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ErrorDetail:
    """Structured error payload serialised into MCP content."""

    code: str
    message: str
    suggestion: str | None = None
    retryable: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }
        if self.suggestion:
            d["suggestion"] = self.suggestion
        if self.details:
            d["details"] = self.details
        return d

    def to_json(self) -> str:
        return json.dumps({"error": self.to_dict()}, indent=2)


class MCPError(Exception):
    """Base class for structured MCP server errors.

    Attributes:
        code: Machine-readable error code (e.g. ``"VALIDATION_ERROR"``).
        message: Human-readable message.
        suggestion: What the caller should try instead.
        retryable: Whether re-trying the same call might succeed.
        details: Additional structured context.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "INTERNAL_ERROR",
        suggestion: str | None = None,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.suggestion = suggestion
        self.retryable = retryable
        self.details = details or {}
        super().__init__(message)

    @property
    def detail(self) -> ErrorDetail:
        return ErrorDetail(
            code=self.code,
            message=str(self),
            suggestion=self.suggestion,
            retryable=self.retryable,
            details=self.details,
        )

    def to_text(self) -> str:
        """Serialise to a JSON text block suitable for MCP ``TextContent``."""
        return self.detail.to_json()


class ToolError(MCPError):
    """Error raised inside a tool handler."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("code", "TOOL_ERROR")
        super().__init__(message, **kwargs)


class ResourceError(MCPError):
    """Error raised inside a resource handler."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("code", "RESOURCE_ERROR")
        super().__init__(message, **kwargs)


class PromptError(MCPError):
    """Error raised inside a prompt handler."""

    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("code", "PROMPT_ERROR")
        super().__init__(message, **kwargs)


class AuthenticationError(MCPError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication required", **kwargs: Any) -> None:
        kwargs.setdefault("code", "AUTHENTICATION_ERROR")
        kwargs.setdefault("retryable", False)
        super().__init__(message, **kwargs)


class RateLimitError(MCPError):
    """Raised when rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        retry_after: float | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("code", "RATE_LIMIT_EXCEEDED")
        kwargs.setdefault("retryable", True)
        if retry_after is not None:
            kwargs.setdefault("details", {})
            kwargs["details"]["retry_after_seconds"] = retry_after
            kwargs.setdefault("suggestion", f"Wait {retry_after:.0f} seconds before retrying")
        super().__init__(message, **kwargs)


class ApprovalDeniedError(MCPError):
    """Raised when a server-side approval gate denies (or times out on) a call.

    Emitted by ``ApprovalGateMiddleware`` for tools declared with
    ``requires_approval=True``.  Not retryable by default: retrying would
    re-request the same approval a human just denied.
    """

    def __init__(
        self,
        message: str = "Approval denied",
        *,
        request_id: str | None = None,
        reviewer_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("code", "APPROVAL_DENIED")
        kwargs.setdefault("retryable", False)
        details = kwargs.setdefault("details", {})
        if request_id is not None:
            details.setdefault("approval_request_id", request_id)
        if reviewer_id is not None:
            details.setdefault("reviewer_id", reviewer_id)
        super().__init__(message, **kwargs)


class ValidationError(MCPError):
    """Raised when input validation fails."""

    def __init__(
        self,
        message: str,
        *,
        field_errors: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("code", "VALIDATION_ERROR")
        kwargs.setdefault("retryable", True)
        if field_errors:
            kwargs.setdefault("details", {})
            kwargs["details"]["field_errors"] = field_errors
        super().__init__(message, **kwargs)
