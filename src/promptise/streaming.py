"""Streaming with tool visibility for Promptise agents.

Yields structured events during agent execution so chat UIs can show
real-time tool activity: "Searching database..." → "Found 3 results"
→ "Generating answer..."

Example::

    async for event in agent.astream_with_tools(
        {"messages": [{"role": "user", "content": "Check my order"}]},
        caller=CallerContext(user_id="user-42"),
    ):
        match event.type:
            case "tool_start":
                print(f"🔧 {event.tool_display_name}...")
            case "tool_end":
                print(f"   → {event.tool_summary}")
            case "token":
                print(event.text, end="", flush=True)
            case "done":
                print()
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "StreamEvent",
    "ToolStartEvent",
    "ToolEndEvent",
    "TokenEvent",
    "DoneEvent",
    "ErrorEvent",
]


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


@dataclass
class StreamEvent:
    """Base event yielded by ``astream_with_tools()``.

    All events have a ``type`` string and a ``timestamp``.
    Use ``to_dict()`` for JSON serialization or ``to_json()`` for SSE.

    Attributes:
        type: Event type identifier.
        timestamp: Monotonic time when the event was created.
    """

    type: str = ""
    timestamp: float = field(default_factory=time.monotonic)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def to_json(self) -> str:
        """Serialize to a JSON string (for SSE ``data:`` lines)."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class ToolStartEvent(StreamEvent):
    """A tool has started executing.

    Attributes:
        tool_name: Raw MCP tool name (e.g. ``"search_customers"``).
        tool_display_name: Human-readable name (e.g. ``"Searching customers"``).
        arguments: Tool arguments (redacted if guardrails active).
        tool_index: 0-based index of tool calls in this invocation.
    """

    type: str = "tool_start"
    tool_name: str = ""
    tool_display_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    tool_index: int = 0


@dataclass
class ToolEndEvent(StreamEvent):
    """A tool has finished executing.

    Attributes:
        tool_name: Raw MCP tool name.
        tool_summary: One-line summary of the result.  For a failed call,
            the error: the server's message for a tool that reported an
            error (an MCP ``ToolError``), or ``"Tool call failed"`` for one
            that raised (the exception text is not exposed).
        duration_ms: Execution time of this call in milliseconds.
        success: ``False`` when the tool reported an error or raised.
        tool_index: 0-based index of this tool call; its ``tool_start``
            event has the same index.
    """

    type: str = "tool_end"
    tool_name: str = ""
    tool_summary: str = ""
    duration_ms: float = 0
    success: bool = True
    tool_index: int = 0


@dataclass
class TokenEvent(StreamEvent):
    """An LLM token has been generated.

    Attributes:
        text: The token text.
        cumulative_text: All text generated so far in this invocation.
    """

    type: str = "token"
    text: str = ""
    cumulative_text: str = ""


@dataclass
class DoneEvent(StreamEvent):
    """The agent has finished processing.

    Attributes:
        full_response: The final answer (post-guardrail redaction): the text
            of the run's last model call, the same text ``ainvoke()``
            returns as the last message.  Text a model wrote before calling
            a tool was streamed as tokens but is not part of the answer.
        tool_calls: Summary of all tool calls made.
        duration_ms: Total invocation time in milliseconds.
        cache_hit: Whether the response came from cache.
    """

    type: str = "done"
    full_response: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    duration_ms: float = 0
    cache_hit: bool = False


@dataclass
class ErrorEvent(StreamEvent):
    """An error occurred during processing.

    The ``message`` is always generic — internal details are never exposed.

    Attributes:
        message: Human-readable error description.
        recoverable: Whether the agent might retry.
    """

    type: str = "error"
    message: str = ""
    recoverable: bool = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Common verb prefixes → present participle for display names
_VERB_MAP: dict[str, str] = {
    "get": "Getting",
    "search": "Searching",
    "create": "Creating",
    "update": "Updating",
    "delete": "Deleting",
    "list": "Listing",
    "send": "Sending",
    "check": "Checking",
    "find": "Finding",
    "fetch": "Fetching",
    "query": "Querying",
    "analyze": "Analyzing",
    "generate": "Generating",
    "process": "Processing",
    "calculate": "Calculating",
    "validate": "Validating",
    "read": "Reading",
    "write": "Writing",
    "run": "Running",
    "execute": "Executing",
    "submit": "Submitting",
    "cancel": "Cancelling",
    "close": "Closing",
    "open": "Opening",
    "load": "Loading",
    "save": "Saving",
    "export": "Exporting",
    "import": "Importing",
    "upload": "Uploading",
    "download": "Downloading",
    "deploy": "Deploying",
    "start": "Starting",
    "stop": "Stopping",
    "restart": "Restarting",
    "monitor": "Monitoring",
    "schedule": "Scheduling",
    "assign": "Assigning",
    "escalate": "Escalating",
    "resolve": "Resolving",
    "count": "Counting",
    "compare": "Comparing",
    "summarize": "Summarizing",
    "extract": "Extracting",
    "parse": "Parsing",
    "convert": "Converting",
    "transform": "Transforming",
    "filter": "Filtering",
    "sort": "Sorting",
    "merge": "Merging",
    "split": "Splitting",
    "connect": "Connecting",
    "disconnect": "Disconnecting",
    "authenticate": "Authenticating",
    "authorize": "Authorizing",
    "verify": "Verifying",
    "encrypt": "Encrypting",
    "decrypt": "Decrypting",
    "scan": "Scanning",
    "detect": "Detecting",
    "notify": "Notifying",
    "publish": "Publishing",
    "subscribe": "Subscribing",
    "install": "Installing",
    "uninstall": "Uninstalling",
    "configure": "Configuring",
    "reset": "Resetting",
    "backup": "Backing up",
    "restore": "Restoring",
    "migrate": "Migrating",
    "test": "Testing",
    "debug": "Debugging",
    "profile": "Profiling",
    "optimize": "Optimizing",
    "refactor": "Refactoring",
    "review": "Reviewing",
    "approve": "Approving",
    "reject": "Rejecting",
}


def tool_display_name(
    tool_name: str,
    overrides: dict[str, str] | None = None,
) -> str:
    """Convert a tool name to a human-readable display string.

    Examples:
        ``"search_customers"`` → ``"Searching customers"``
        ``"get_order_status"`` → ``"Getting order status"``
        ``"hr_list_employees"`` → ``"Listing employees"``

    Args:
        tool_name: Raw tool name (underscore-separated).
        overrides: Developer-provided display name overrides.

    Returns:
        Human-readable display string.
    """
    # Developer override takes priority
    if overrides and tool_name in overrides:
        return overrides[tool_name]

    # Split on underscores
    words = tool_name.replace("-", "_").split("_")
    if not words:
        return tool_name

    # Try each word as a verb (some tools have namespace prefixes like "hr_list_employees")
    for i, word in enumerate(words):
        lower = word.lower()
        if lower in _VERB_MAP:
            # Found the verb — use it as the start, rest follows
            rest = " ".join(w for w in words[i + 1 :])
            return f"{_VERB_MAP[lower]} {rest}".strip()

    # No known verb found — just capitalize
    return " ".join(w.capitalize() for w in words)


def tool_summary(result: str | None, max_length: int = 120) -> str:
    """Summarize a tool result for display.

    - JSON dicts: shows first 3 key-value pairs
    - JSON lists: shows item count
    - Plain text: truncates to max_length

    Args:
        result: Raw tool result string.
        max_length: Maximum summary length.

    Returns:
        One-line summary string.
    """
    if not result or not result.strip():
        return "Done"

    # Try to parse as JSON
    try:
        data = json.loads(result)
        if isinstance(data, dict):
            items = list(data.items())[:3]
            preview = ", ".join(f"{k}: {v}" for k, v in items)
            if len(data) > 3:
                preview += f" (+{len(data) - 3} more)"
            return preview[:max_length] if len(preview) > max_length else preview
        elif isinstance(data, list):
            return f"Found {len(data)} result(s)"
    except (json.JSONDecodeError, TypeError):
        pass

    # Plain text truncation
    result = result.strip()
    if len(result) <= max_length:
        return result
    return result[: max_length - 3] + "..."


def tool_error_summary(output: str | None, max_length: int = 120) -> str:
    """Summarize a failed tool call's output for display.

    An MCP ``ToolError`` arrives as ``{"error": {"code", "message", ...}}``;
    its ``message`` is the summary.  Any other output is summarized like
    :func:`tool_summary`.

    Examples:
        ``'{"error": {"code": "TOOL_ERROR", "message": "No order A-9."}}'``
        → ``"No order A-9."``

    Args:
        output: Raw tool output.
        max_length: Maximum summary length.

    Returns:
        One-line summary string.
    """
    if output:
        try:
            data = json.loads(output)
        except (json.JSONDecodeError, TypeError):
            data = None
        error = data.get("error") if isinstance(data, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        if isinstance(message, str) and message.strip():
            return tool_summary(message, max_length)
    return tool_summary(output, max_length) if output else "Tool call failed"


def content_text(content: Any) -> str:
    """The text in a message's (or chunk's) ``content``.

    ``content`` is a string, or a list of content blocks (Anthropic,
    Gemini…) of which only the ``"text"`` blocks are text.

    Args:
        content: A ``BaseMessage.content`` value.

    Returns:
        The concatenated text, ``""`` when there is none.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


async def redact_tool_args(
    args: dict[str, Any],
    guardrails: Any | None,
) -> dict[str, Any]:
    """Redact PII/credentials from tool arguments before streaming.

    Args:
        args: Tool arguments dict.
        guardrails: A :class:`PromptiseSecurityScanner` instance, or None.

    Returns:
        Redacted arguments dict.
    """
    if guardrails is None:
        return dict(args)

    try:
        text = json.dumps(args, default=str)
        redacted = await guardrails.check_output(text)
        if isinstance(redacted, str) and redacted != text:
            try:
                return json.loads(redacted)
            except json.JSONDecodeError:
                return {"_redacted": True}
        return dict(args)
    except Exception as exc:
        # If guardrails raise a violation (blocked content in args),
        # err on the side of hiding the args entirely
        exc_name = type(exc).__name__
        if "Violation" in exc_name or "Guardrail" in exc_name:
            return {"_redacted": True}
        return dict(args)
