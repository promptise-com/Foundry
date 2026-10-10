"""Agent tools that expose MCP resources and prompts.

``build_agent(expose_resources=True)`` adds ``list_resources`` and
``read_resource``; ``build_agent(expose_prompts=True)`` adds ``get_prompt``.
They let the model pull what the connected MCP servers publish besides
tools: documents and data (resources) and ready-made instructions
(prompts).  The catalogue is discovered once at build time and written
into the tool descriptions; ``list_resources`` re-reads it on demand.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from mcp.types import GetPromptResult, ReadResourceResult
from pydantic import BaseModel, Field

from ._multi import MCPMultiClient
from ._tool_adapter import OnAfter, OnBefore, OnError

logger = logging.getLogger("promptise.mcp.client")

# Catalogue entries written into a tool description; the rest is reachable
# through ``list_resources``.
_MAX_LISTED = 40


class _ReadResourceArgs(BaseModel):
    uri: str = Field(description="URI of the resource to read, e.g. 'docs://pages/refunds'.")
    server: str | None = Field(
        default=None,
        description="Name of the MCP server to read from. Only needed when two servers "
        "serve the same URI.",
    )


class _GetPromptArgs(BaseModel):
    name: str = Field(description="Name of the prompt.")
    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="The prompt's arguments, by name.",
    )
    server: str | None = Field(
        default=None,
        description="Name of the MCP server to get the prompt from. Only needed when two "
        "servers serve a prompt with this name.",
    )


class _NoArgs(BaseModel):
    pass


def _unique_name(name: str, taken: set[str]) -> str:
    """*name*, or ``mcp_<name>`` when an MCP server already has a tool called *name*."""
    if name not in taken:
        return name
    alt = f"mcp_{name}"
    logger.warning(
        "An MCP server already exposes a tool named %r; the generated tool is named %r.",
        name,
        alt,
    )
    return alt


def _line(text: str | None) -> str:
    return " ".join((text or "").split())


async def _resource_catalogue(multi: MCPMultiClient) -> list[str]:
    """One line per resource and resource template, with the serving server."""
    multi_server = len(multi.servers) > 1
    resources = await multi.list_resources()
    templates = await multi.list_resource_templates()
    by_uri = multi.resource_to_server
    by_template = multi.template_to_server

    lines: list[str] = []
    for r in resources:
        where = f" [server: {by_uri.get(str(r.uri), '?')}]" if multi_server else ""
        mime = f" ({r.mimeType})" if r.mimeType else ""
        lines.append(f"- {r.uri}{mime}{where}: {_line(r.description) or r.name}")
    for t in templates:
        where = f" [server: {by_template.get(t.uriTemplate, '?')}]" if multi_server else ""
        mime = f" ({t.mimeType})" if t.mimeType else ""
        lines.append(
            f"- {t.uriTemplate}{mime}{where}: {_line(t.description) or t.name} "
            "(a template: replace each {placeholder})"
        )
    return lines


def _resource_text(result: ReadResourceResult) -> str:
    parts: list[str] = []
    for item in result.contents:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
        else:
            blob = getattr(item, "blob", "") or ""
            size = len(blob) * 3 // 4  # base64 → approximate byte count
            parts.append(
                f"[binary resource {item.uri}: {item.mimeType or 'application/octet-stream'}, "
                f"about {size} bytes — not shown]"
            )
    return "\n\n".join(parts)


def _prompt_text(result: GetPromptResult) -> str:
    rendered: list[tuple[str, str]] = []
    for message in result.messages:
        content = message.content
        text = getattr(content, "text", None)
        if text is None:
            text = f"[{getattr(content, 'type', 'non-text')} content — not shown]"
        rendered.append((message.role, text))
    if len(rendered) == 1 and rendered[0][0] == "user":
        return rendered[0][1]
    return "\n\n".join(f"[{role}]\n{text}" for role, text in rendered)


def _wrap(
    name: str,
    run: Callable[..., Awaitable[str]],
    *,
    on_before: OnBefore | None,
    on_after: OnAfter | None,
    on_error: OnError | None,
) -> Callable[..., Awaitable[str]]:
    """Fire the agent's tool callbacks; return failures to the model as text."""

    async def _call(**kwargs: Any) -> str:
        if on_before:
            with contextlib.suppress(Exception):
                on_before(name, kwargs)
        try:
            out = await run(**kwargs)
        except Exception as exc:
            if on_error:
                with contextlib.suppress(Exception):
                    on_error(name, exc)
            return f"Error: {exc}"
        if on_after:
            with contextlib.suppress(Exception):
                on_after(name, out)
        return out

    return _call


async def make_resource_tools(
    multi: MCPMultiClient,
    *,
    taken: set[str],
    on_before: OnBefore | None = None,
    on_after: OnAfter | None = None,
    on_error: OnError | None = None,
) -> list[BaseTool]:
    """Build ``list_resources`` and ``read_resource`` over *multi*.

    Returns an empty list when no connected server serves resources.

    Args:
        multi: A connected multi-client.
        taken: Names of the tools the agent already has (a generated tool
            is renamed ``mcp_<name>`` on a clash).
    """
    lines = await _resource_catalogue(multi)
    if not lines:
        return []

    listed = "\n".join(lines[:_MAX_LISTED])
    list_name = _unique_name("list_resources", taken)
    if len(lines) > _MAX_LISTED:
        listed += f"\n- ... and {len(lines) - _MAX_LISTED} more: call {list_name} to see them."

    async def _list() -> str:
        return "\n".join(await _resource_catalogue(multi)) or "No resources available."

    async def _read(uri: str, server: str | None = None) -> str:
        return _resource_text(await multi.read_resource(uri, server=server))

    read_name = _unique_name("read_resource", taken)
    hooks: dict[str, Any] = {"on_before": on_before, "on_after": on_after, "on_error": on_error}
    return [
        StructuredTool.from_function(
            coroutine=_wrap(list_name, _list, **hooks),
            name=list_name,
            description=(
                "List the resources (documents and data) the connected MCP servers "
                f"publish, with their URIs. Read one with {read_name}."
            ),
            args_schema=_NoArgs,
        ),
        StructuredTool.from_function(
            coroutine=_wrap(read_name, _read, **hooks),
            name=read_name,
            description=(
                "Read a resource (a document or data) that the connected MCP servers "
                "publish, by its URI. Use it to look things up instead of guessing.\n"
                f"Available resources:\n{listed}"
            ),
            args_schema=_ReadResourceArgs,
        ),
    ]


async def make_prompt_tools(
    multi: MCPMultiClient,
    *,
    taken: set[str],
    on_before: OnBefore | None = None,
    on_after: OnAfter | None = None,
    on_error: OnError | None = None,
) -> list[BaseTool]:
    """Build ``get_prompt`` over *multi*.

    Returns an empty list when no connected server serves prompts.

    Args:
        multi: A connected multi-client.
        taken: Names of the tools the agent already has.
    """
    prompts = await multi.list_prompts()
    if not prompts:
        return []
    multi_server = len(multi.servers) > 1
    by_name = multi.prompt_to_server

    lines: list[str] = []
    for p in prompts[:_MAX_LISTED]:
        args = ", ".join(
            f"{a.name}{'' if a.required else '?'}: {_line(a.description) or a.name}"
            for a in (p.arguments or [])
        )
        where = f" [server: {by_name.get(p.name, '?')}]" if multi_server else ""
        lines.append(
            f"- {p.name}{where}: {_line(p.description) or p.name}"
            + (f" Arguments — {args}" if args else "")
        )
    if len(prompts) > _MAX_LISTED:
        lines.append(f"- ... and {len(prompts) - _MAX_LISTED} more.")

    async def _get(
        name: str, arguments: dict[str, Any] | None = None, server: str | None = None
    ) -> str:
        return _prompt_text(await multi.get_prompt(name, arguments or {}, server=server))

    tool_name = _unique_name("get_prompt", taken)
    return [
        StructuredTool.from_function(
            coroutine=_wrap(
                tool_name, _get, on_before=on_before, on_after=on_after, on_error=on_error
            ),
            name=tool_name,
            description=(
                "Get a ready-made prompt from the connected MCP servers, filled in with "
                "its arguments (optional ones end in '?'), and follow the instructions it "
                "returns.\nAvailable prompts:\n" + "\n".join(lines)
            ),
            args_schema=_GetPromptArgs,
        )
    ]
