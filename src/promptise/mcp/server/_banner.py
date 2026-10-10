"""Startup banner for the MCP server.

Prints a visually rich, informative banner to the console when the
server starts — similar to FastAPI.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


# =====================================================================
# ANSI helpers (auto-disabled when not a TTY)
# =====================================================================

_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Wrap *text* in ANSI escape if stdout is a TTY."""
    if not _IS_TTY:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(text: str) -> str:
    return _c("1", text)


def _dim(text: str) -> str:
    return _c("2", text)


def _cyan(text: str) -> str:
    return _c("36", text)


def _green(text: str) -> str:
    return _c("32", text)


def _yellow(text: str) -> str:
    return _c("33", text)


def _magenta(text: str) -> str:
    return _c("35", text)


def _blue(text: str) -> str:
    return _c("34", text)


# =====================================================================
# Banner
# =====================================================================

_ASCII_BOX = str.maketrans("╔═╗║╚╝", "+=+|++")


def _emit(text: str) -> None:
    """Print *text* in a form stdout's encoding can represent.

    A redirected stdout on Windows (a service, ``> server.log``, a parent
    process piping output) uses the ANSI code page, cp1252, which has no
    box-drawing characters: printing the banner raised
    ``UnicodeEncodeError`` and the server never started.  The box falls back
    to ASCII and anything else unencodable (a server name, say) to ``?``.
    """
    stream = sys.stdout
    if stream is None:  # pythonw.exe
        return
    encoding = getattr(stream, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
    except UnicodeEncodeError:
        text = text.translate(_ASCII_BOX).encode(encoding, errors="replace").decode(encoding)
    except LookupError:
        pass
    print(text, file=stream)


def print_banner(
    *,
    server_name: str,
    version: str,
    transport: str,
    host: str,
    port: int,
    tool_count: int,
    auth_tool_count: int,
    resource_count: int,
    prompt_count: int,
    middleware_count: int,
) -> None:
    """Print the server startup banner to stdout."""
    # Endpoint URL
    if transport == "stdio":
        endpoint = "stdin/stdout"
    elif transport == "sse":
        endpoint = f"http://{host}:{port}/sse"
    else:
        endpoint = f"http://{host}:{port}/mcp"

    # Transport label
    transport_label = {"stdio": "Stdio", "http": "Streamable HTTP", "sse": "SSE"}.get(
        transport, transport.upper()
    )

    # Build the banner
    lines: list[str] = []

    lines.append("")
    lines.append(_bold(_cyan("    ╔═══════════════════════════════════════════════════════════╗")))
    lines.append(
        _bold(_cyan("    ║"))
        + _bold("          P R O M P T I S E   M C P   S E R V E R          ")
        + _bold(_cyan("║"))
    )
    lines.append(_bold(_cyan("    ╚═══════════════════════════════════════════════════════════╝")))
    lines.append("")

    # Server info
    lines.append(f"    {_dim('Server')}      {_bold(server_name)} {_dim('v')}{version}")
    lines.append(f"    {_dim('Transport')}   {_bold(transport_label)}")

    if transport != "stdio":
        lines.append(f"    {_dim('Endpoint')}    {_green(endpoint)}")

    lines.append("")

    # Registrations
    auth_suffix = ""
    if auth_tool_count > 0:
        auth_suffix = f"  {_dim(f'({auth_tool_count} require auth)')}"
    lines.append(f"    {_dim('Tools')}       {_bold(str(tool_count))} registered{auth_suffix}")
    lines.append(f"    {_dim('Resources')}   {_bold(str(resource_count))} registered")
    lines.append(f"    {_dim('Prompts')}     {_bold(str(prompt_count))} registered")

    if middleware_count > 0:
        lines.append(f"    {_dim('Middleware')}   {_bold(str(middleware_count))} active")

    lines.append("")

    # Docs hint
    if transport != "stdio":
        lines.append(f"    {_dim('Docs')}        {_cyan('docs://manifest')}")
        lines.append("")

    # Ready message
    lines.append(f"    {_green('Ready!')} {_dim('Press Ctrl+C to stop.')}")
    lines.append("")

    _emit("\n".join(lines))
