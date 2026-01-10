"""
CLI for deepmcpagent: list tools and run an interactive agent session.

Notes:
    - The CLI path uses provider id strings for models (e.g., "openai:gpt-4.1"),
      which `init_chat_model` handles. In code, you can pass a model instance.
    - Model is REQUIRED (no fallback).
    - Usage for repeated server specs:
        --stdio "name=echo command=python args='-m mypkg.server --port 3333' env.API_KEY=xyz keep_alive=false"
        --stdio "name=tool2 command=/usr/local/bin/tool2"
        --http  "name=remote url=http://127.0.0.1:8000/mcp transport=http"

      (Repeat --stdio/--http for multiple servers.)
"""

from __future__ import annotations

import asyncio
import json
import shlex
from importlib.metadata import version as get_version
from typing import Annotated, Any, Literal, cast
from urllib.parse import urlparse

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import build_deep_agent
from .config import HTTPServerSpec, ServerSpec, StdioServerSpec

load_dotenv()

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


@app.callback(invoke_without_command=True)
def _version_callback(
    version: Annotated[
        bool | None,
        typer.Option("--version", help="Show version and exit", is_eager=True),
    ] = None,
) -> None:
    """Global callback to support --version printing."""
    if version:
        console.print(get_version("deepmcpagent"))
        raise typer.Exit()


def _parse_kv(opts: list[str]) -> dict[str, str]:
    """Parse ['k=v', 'x=y', ...] into a dict. Values may contain spaces."""
    out: dict[str, str] = {}
    for it in opts:
        if "=" not in it:
            raise typer.BadParameter(f"Expected key=value, got: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _merge_servers(stdios: list[str], https: list[str], *, require_https: bool = False) -> dict[str, ServerSpec]:
    """
    Convert flat lists of block strings into server specs.

    Each entry in `stdios` / `https` is a single quoted string like:
      "name=echo command=python args='-m mymod --port 3333' env.API_KEY=xyz cwd=/tmp keep_alive=false"
      "name=remote url=http://127.0.0.1:8000/mcp transport=http"

    We first shlex-split the string into key=value tokens, then parse.
    """
    servers: dict[str, ServerSpec] = {}

    # stdio (kept for completeness)
    for block_str in stdios:
        tokens = shlex.split(block_str)
        kv = _parse_kv(tokens)

        name = kv.pop("name", None)
        if not name:
            raise typer.BadParameter("Missing required key: name (in --stdio block)")

        command = kv.pop("command", None)
        if not command:
            raise typer.BadParameter("Missing required key: command (in --stdio block)")

        args_value = kv.pop("args", "")
        args_list = shlex.split(args_value) if args_value else []

        env = {k.split(".", 1)[1]: v for k, v in list(kv.items()) if k.startswith("env.")}
        cwd = kv.get("cwd")
        keep_alive = kv.get("keep_alive", "true").lower() != "false"

        stdio_spec: ServerSpec = StdioServerSpec(
            command=command,
            args=args_list,
            env=env,
            cwd=cwd,
            keep_alive=keep_alive,
        )
        servers[name] = stdio_spec

    # http
    for block_str in https:
        tokens = shlex.split(block_str)
        kv = _parse_kv(tokens)

        name = kv.pop("name", None)
        if not name:
            raise typer.BadParameter("Missing required key: name (in --http block)")

        url = kv.pop("url", None)
        if not url:
            raise typer.BadParameter("Missing required key: url (in --http block)")

        parsed = urlparse(url)
        if require_https and parsed.scheme != "https":
            raise typer.BadParameter("HTTPS is required for --http blocks when https-only is enabled")

        transport_str = kv.pop("transport", "http")  # "http", "streamable-http", or "sse"
        transport = cast(Literal["http", "streamable-http", "sse"], transport_str)

        headers = {k.split(".", 1)[1]: v for k, v in list(kv.items()) if k.startswith("header.")}
        auth = kv.get("auth")

        if "Authorization" in headers and parsed.scheme != "https":
            console.print(
                "[yellow]Warning:[/] Authorization header over non-HTTPS may leak credentials; use HTTPS or omit auth.",
                highlight=False,
            )

        http_spec: ServerSpec = HTTPServerSpec(
            url=url,
            transport=transport,
            headers=headers,
            auth=auth,
        )
        servers[name] = http_spec

    return servers


def _extract_final_answer(result: Any) -> str:
    """Best-effort extraction of the final text from various executors."""
    try:
        # LangGraph prebuilt returns {"messages": [ ... ]}
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            last = result["messages"][-1]
            content = getattr(last, "content", None)
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list) and content and isinstance(content[0], dict):
                return content[0].get("text") or str(content)
            return str(last)
        return str(result)
    except Exception:
        return str(result)


@app.command(name="list-tools")
def list_tools(
    model_id: Annotated[
        str,
        typer.Option("--model-id", help="REQUIRED model provider id (e.g., 'openai:gpt-4.1')."),
    ],
    stdio: Annotated[
        list[str] | None,
        typer.Option(
            "--stdio",
            help=(
                "Block string: \"name=... command=... args='...' "
                '[env.X=Y] [cwd=...] [keep_alive=true|false]\". Repeatable.'
            ),
        ),
    ] = None,
    http: Annotated[
        list[str] | None,
        typer.Option(
            "--http",
            help=(
                'Block string: "name=... url=... [transport=http|streamable-http|sse] '
                '[header.X=Y] [auth=...]". Repeatable.'
            ),
        ),
    ] = None,
    instructions: Annotated[
        str,
        typer.Option("--instructions", help="Optional system prompt override."),
    ] = "",
    https_only: Annotated[
        bool,
        typer.Option("--https-only", help="Require HTTPS URLs for MCP servers."),
    ] = False,
) -> None:

    """List all MCP tools discovered using the provided server specs."""
    servers = _merge_servers(stdio or [], http or [], require_https=https_only)

    async def _run() -> None:
        _, loader = await build_deep_agent(
            servers=servers,
            model=model_id,
            instructions=instructions or None,
        )
        infos = await loader.list_tool_info()
        infos = list(infos or [])

        table = Table(title="MCP Tools", show_lines=True)
        table.add_column("Tool", style="cyan", no_wrap=True)
        table.add_column("Description", style="green")
        table.add_column("Input Schema", style="white")
        for i in infos:
            schema_str = json.dumps(i.input_schema, ensure_ascii=False)
            if len(schema_str) > 120:
                schema_str = schema_str[:117] + "..."
            table.add_row(i.name, i.description or "-", schema_str)
        console.print(table)

    asyncio.run(_run())


@app.command()
def run(
    model_id: Annotated[
        str,
        typer.Option(..., help="REQUIRED model provider id (e.g., 'openai:gpt-4.1')."),
    ],
    stdio: Annotated[
        list[str] | None,
        typer.Option(
            "--stdio",
            help=(
                "Block string: \"name=... command=... args='...' "
                '[env.X=Y] [cwd=...] [keep_alive=true|false]\". Repeatable.'
            ),
        ),
    ] = None,
    http: Annotated[
        list[str] | None,
        typer.Option(
            "--http",
            help=(
                'Block string: "name=... url=... [transport=http|streamable-http|sse] '
                '[header.X=Y] [auth=...]". Repeatable.'
            ),
        ),
    ] = None,
    instructions: Annotated[
        str,
        typer.Option("--instructions", help="Optional system prompt override."),
    ] = "",
    # IMPORTANT: don't duplicate defaults in Option() and the parameter!
    trace: Annotated[
        bool,
        typer.Option("--trace/--no-trace", help="Print tool invocations & results."),
    ] = True,
    raw: Annotated[
        bool,
        typer.Option("--raw/--no-raw", help="Also print raw result object."),
    ] = False,
    https_only: Annotated[
        bool,
        typer.Option("--https-only", help="Require HTTPS URLs for MCP servers."),
    ] = False,
) -> None:

    """Start an interactive agent that uses only MCP tools."""
    servers = _merge_servers(stdio or [], http or [], require_https=https_only)

    async def _chat() -> None:
        graph, _ = await build_deep_agent(
            servers=servers,
            model=model_id,
            instructions=instructions or None,
            trace_tools=trace,  # <- enable deepmcpagent tool tracing
        )
        console.print("[bold]DeepMCPAgent is ready. Type 'exit' to quit.[/bold]")
        while True:
            try:
                user = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\nExiting.")
                break
            if user.lower() in {"exit", "quit"}:
                break
            if not user:
                continue
            try:
                result = await graph.ainvoke({"messages": [{"role": "user", "content": user}]})
            except Exception as exc:
                console.print(f"[red]Error during run:[/red] {exc}")
                continue

            final_text = _extract_final_answer(result)
            console.print(
                Panel(final_text or "(no content)", title="Final LLM Answer", style="bold green")
            )
            if raw:
                console.print(result)

    asyncio.run(_chat())
