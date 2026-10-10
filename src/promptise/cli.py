"""
CLI for promptise: list tools and run an interactive agent session.

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
import re
import shlex
from importlib.metadata import version as get_version
from typing import Annotated, Any, Literal, cast

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from .agent import build_agent
from .config import HTTPServerSpec, ServerSpec, StdioServerSpec
from .exceptions import SuperAgentError, SuperAgentValidationError
from .models import ModelSetupError, load_dotenv_if_present

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

# Mount runtime sub-app
from .models_cli import models_app
from .runtime.cli import runtime_app

app.add_typer(runtime_app, name="runtime")
app.add_typer(models_app, name="models")


@app.callback(invoke_without_command=True)
def _version_callback(
    version: Annotated[
        bool | None,
        typer.Option("--version", help="Show version and exit", is_eager=True),
    ] = None,
) -> None:
    """Global callback: ``--version``, then the ``.env`` file every command shares."""
    if version:
        console.print(get_version("promptise"))
        raise typer.Exit()
    # One rule for scripts and the CLI: .env from the working directory (or a
    # parent, up to the project root), never overriding a set variable, off
    # with PROMPTISE_NO_DOTENV=1. Loaded here rather than at import so that
    # --version and --help work from any directory, and an unreadable file
    # is one clean error instead of a traceback before typer even starts.
    err = Console(stderr=True)
    try:
        loaded = load_dotenv_if_present()
    except ModelSetupError as exc:
        err.print(f"[red]Error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=2)
    if loaded:
        err.print(f"[dim].env loaded from {escape(loaded)}[/dim]")


def _parse_kv(opts: list[str]) -> dict[str, str]:
    """Parse ['k=v', 'x=y', ...] into a dict. Values may contain spaces."""
    out: dict[str, str] = {}
    for it in opts:
        if "=" not in it:
            raise typer.BadParameter(f"Expected key=value, got: {it}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _merge_servers(stdios: list[str], https: list[str]) -> dict[str, ServerSpec]:
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

        transport_str = kv.pop("transport", "http")  # "http", "streamable-http", or "sse"
        transport = cast(Literal["http", "streamable-http", "sse"], transport_str)

        headers = {k.split(".", 1)[1]: v for k, v in list(kv.items()) if k.startswith("header.")}
        if "auth" in kv:
            raise typer.BadParameter(
                "auth= is not supported (it was never sent to the server); use "
                "bearer_token=<token> or api_key=<key> (in --http block)"
            )

        http_spec: ServerSpec = HTTPServerSpec.model_validate(
            {
                "url": url,
                "transport": transport,
                "headers": headers,
                "bearer_token": kv.get("bearer_token"),
                "api_key": kv.get("api_key"),
            }
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
                '[env.X=Y] [cwd=...] [keep_alive=true|false]". Repeatable.'
            ),
        ),
    ] = None,
    http: Annotated[
        list[str] | None,
        typer.Option(
            "--http",
            help=(
                'Block string: "name=... url=... [transport=http|streamable-http|sse] '
                '[header.X=Y] [bearer_token=...] [api_key=...]". Repeatable.'
            ),
        ),
    ] = None,
    instructions: Annotated[
        str,
        typer.Option("--instructions", help="Optional system prompt override."),
    ] = "",
) -> None:
    """List all MCP tools discovered using the provided server specs."""
    servers = _merge_servers(stdio or [], http or [])

    async def _run() -> None:
        agent = await build_agent(
            servers=servers,
            model=model_id,
            instructions=instructions or None,
        )

        table = Table(title="MCP Tools", show_lines=True)
        table.add_column("Tool", style="cyan", no_wrap=True)
        table.add_column("Description", style="green")
        table.add_column("Input Schema", style="white")

        try:
            for tool in agent.tools:
                schema_str = ""
                schema: Any = getattr(tool, "args_schema", None)
                if schema is not None:
                    try:
                        import json

                        raw = schema if isinstance(schema, dict) else schema.model_json_schema()
                        schema_str = json.dumps(raw, indent=2)
                    except Exception:
                        schema_str = str(schema)
                table.add_row(tool.name, tool.description or "", schema_str)

            console.print(table)
        finally:
            # Close MCP sessions in the task that opened them; tearing them
            # down at loop close would cross an anyio cancel scope.
            await agent.shutdown()

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
                '[env.X=Y] [cwd=...] [keep_alive=true|false]". Repeatable.'
            ),
        ),
    ] = None,
    http: Annotated[
        list[str] | None,
        typer.Option(
            "--http",
            help=(
                'Block string: "name=... url=... [transport=http|streamable-http|sse] '
                '[header.X=Y] [bearer_token=...] [api_key=...]". Repeatable.'
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
) -> None:
    """Start an interactive agent that uses only MCP tools."""
    servers = _merge_servers(stdio or [], http or [])

    async def _chat() -> None:
        graph = await build_agent(
            servers=servers,
            model=model_id,
            instructions=instructions or None,
            trace_tools=trace,  # <- enable promptise tool tracing
        )
        console.print("[bold]Promptise Foundry is ready. Type 'exit' to quit.[/bold]")
        try:
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
                    Panel(
                        final_text or "(no content)", title="Final LLM Answer", style="bold green"
                    )
                )
                if raw:
                    console.print(result)
        finally:
            # Close MCP sessions in the task that opened them — tearing them
            # down at interpreter exit crosses an anyio cancel scope.
            await graph.shutdown()

    asyncio.run(_chat())


# =============================================================================
# New Commands for .superagent File Support
# =============================================================================


@app.command()
def agent(
    file: Annotated[
        str,
        typer.Argument(help="Path to .superagent configuration file"),
    ],
    model_id: Annotated[
        str | None,
        typer.Option("--model-id", help="Override model from config file"),
    ] = None,
    instructions: Annotated[
        str | None,
        typer.Option("--instructions", help="Override instructions from config file"),
    ] = None,
    trace: Annotated[
        bool | None,
        typer.Option("--trace/--no-trace", help="Override trace setting from config file"),
    ] = None,
    stdio: Annotated[
        list[str] | None,
        typer.Option(
            "--stdio",
            help="Additional stdio server (merged with config file servers)",
        ),
    ] = None,
    http: Annotated[
        list[str] | None,
        typer.Option(
            "--http",
            help="Additional http server (merged with config file servers)",
        ),
    ] = None,
    raw: Annotated[
        bool,
        typer.Option("--raw/--no-raw", help="Also print raw result object."),
    ] = False,
) -> None:
    """Run an agent from a .superagent configuration file.

    This command loads agent configuration from a .superagent file and
    optionally overrides specific settings via CLI flags. CLI flags always
    take precedence over file configuration (they apply to the top agent;
    cross-agents keep their own files' settings).

    Every agent under ``cross_agents:`` is built with its whole file, at any
    depth, in the same event loop as the chat. When an agent's approval
    handler is ``queue`` and you are at a terminal, each approval request is
    asked here as a y/N question.

    Examples:
        promptise agent my_agent.superagent
        promptise agent my_agent.superagent --model-id "openai:gpt-4o"
        promptise agent my_agent.superagent --no-trace
    """
    from .superagent import SuperAgentLoader, build_superagent

    # Load configuration file (and every cross-agent file it references)
    try:
        loader = SuperAgentLoader.from_file(file)
        loader.resolve_env_vars()
        loader.resolve_cross_agents(recursive=True)
    except SuperAgentValidationError as exc:
        console.print("[red]Configuration validation failed:[/red]")
        console.print(str(exc))
        raise typer.Exit(1)
    except SuperAgentError as exc:
        console.print(f"[red]Failed to load configuration:[/red] {escape(str(exc))}")
        raise typer.Exit(1)

    # Report CLI overrides
    if model_id:
        console.print(f"[yellow]Overriding model:[/yellow] {model_id}")
    if instructions:
        console.print("[yellow]Overriding instructions[/yellow]")
    if trace is not None:
        console.print(f"[yellow]Overriding trace:[/yellow] {trace}")

    # Additional servers from CLI
    extra_servers: dict[str, ServerSpec] = {}
    if stdio or http:
        extra_servers = _merge_servers(stdio or [], http or [])
        console.print(f"[yellow]Added {len(extra_servers)} server(s) from CLI[/yellow]")

    # Build and run the whole team in ONE event loop: MCP sessions (stdio
    # servers above all) belong to the loop and task that opened them.
    async def _chat() -> int:
        """Build the team, then start an interactive chat session."""
        try:
            graph = await build_superagent(
                loader,
                model=model_id,
                instructions=instructions,
                trace=trace,
                extra_servers=extra_servers,
            )
        except Exception as exc:
            # One line, not a traceback: the cause (e.g. a stdio server that
            # exits at startup) is in the message and the server's stderr.
            console.print(
                f"[red]Failed to start the agent:[/red] {escape(str(exc))}", soft_wrap=True
            )
            return 1

        team = _count_cross_agents(loader)
        if team:
            console.print(f"[green]Loaded {team} cross-agent(s)[/green]")

        reader = _LineReader()
        prompters = _start_approval_prompts(graph, reader)

        console.print(f"[bold]Promptise Foundry loaded from {file}. Type 'exit' to quit.[/bold]")

        try:
            while True:
                try:
                    user = (await reader.readline("> ")).strip()
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
                    console.print(f"[red]Error during run:[/red] {escape(str(exc))}")
                    continue

                final_text = _extract_final_answer(result)
                console.print(
                    Panel(
                        final_text or "(no content)", title="Final LLM Answer", style="bold green"
                    )
                )
                if raw:
                    console.print(result)
        finally:
            for task in prompters:
                task.cancel()
            await graph.shutdown()  # same task that opened the MCP sessions; whole team
        return 0

    try:
        code = asyncio.run(_chat())
    except KeyboardInterrupt:
        console.print("\nExiting.")
        code = 130
    if code:
        raise typer.Exit(code)


def _count_cross_agents(loader: Any) -> int:
    """Number of cross-agents under *loader*, at every depth."""
    children = loader.cross_loaders or {}
    return len(children) + sum(_count_cross_agents(c) for c in children.values())


class _LineReader:
    """The single reader of stdin, shared by the chat prompt and approval prompts.

    Each line is read on a daemon thread (so exiting never waits for a line).
    A read that its caller stopped waiting for — an approval that timed out —
    stays pending and is handed to the next caller instead of starting a
    second, competing read.
    """

    def __init__(self) -> None:
        self._pending: asyncio.Future[str] | None = None

    async def readline(self, prompt: str) -> str:
        """Print *prompt* and return the next line. Raises EOFError at end of input."""
        import threading

        console.print(prompt, end="", markup=False, highlight=False)
        if self._pending is None:
            loop = asyncio.get_running_loop()
            future: asyncio.Future[str] = loop.create_future()

            def _fail(exc: BaseException) -> None:
                if not future.done():
                    future.set_exception(exc)

            def _done(line: str) -> None:
                if not future.done():
                    future.set_result(line)

            def _read() -> None:
                try:
                    line = input()
                except BaseException as exc:  # EOFError, KeyboardInterrupt
                    loop.call_soon_threadsafe(_fail, exc)
                else:
                    loop.call_soon_threadsafe(_done, line)

            threading.Thread(target=_read, daemon=True).start()
            self._pending = future
        line = await asyncio.shield(self._pending)
        self._pending = None
        return line


def _start_approval_prompts(agent: Any, reader: _LineReader) -> list[asyncio.Task[None]]:
    """Answer ``queue`` approval requests of every agent in the team at the terminal."""
    import sys

    from .approval import QueueApprovalHandler

    handlers: list[tuple[QueueApprovalHandler, float, str]] = []
    pending = [agent]
    while pending:
        current = pending.pop()
        pending.extend(getattr(current, "_owned_agents", []))
        policy: Any = getattr(current, "_approval", None)
        handler = getattr(policy, "handler", None)
        if isinstance(handler, QueueApprovalHandler):
            handlers.append((handler, policy.timeout, policy.on_timeout))
    if not handlers:
        return []

    if not sys.stdin.isatty():
        for _, timeout, on_timeout in handlers:
            console.print(
                "[yellow]approval.handler is 'queue' but input is not a terminal: approval "
                f"requests cannot be answered here and time out after {timeout:g}s "
                f"(on_timeout: {on_timeout}).[/yellow]"
            )
        return []

    lock = asyncio.Lock()  # one question at a time across the team
    return [
        asyncio.ensure_future(_answer_approvals(handler, reader, lock))
        for handler, _, _ in handlers
    ]


async def _answer_approvals(handler: Any, reader: _LineReader, lock: asyncio.Lock) -> None:
    """Ask y/N for each request in *handler*'s queue until cancelled."""
    import json
    import time

    from .approval import ApprovalDecision

    while True:
        request = await handler.request_queue.get()
        async with lock:
            remaining = request.timestamp + request.timeout - time.time()
            if remaining <= 0:
                continue
            args = json.dumps(request.arguments, default=str)
            console.print(
                Panel(
                    escape(f"{request.tool_name}({args})"),
                    title=f"Approval needed — answer within {remaining:.0f}s",
                    style="yellow",
                )
            )
            answer = asyncio.ensure_future(reader.readline(f"Allow {request.tool_name}? [y/N] "))
            done, _ = await asyncio.wait({answer}, timeout=remaining)
            if not done:
                answer.cancel()
                console.print("\n[yellow]No answer in time: the request timed out.[/yellow]")
                continue
            try:
                approved = answer.result().strip().lower() in {"y", "yes"}
            except (EOFError, KeyboardInterrupt):
                approved = False
            decision = ApprovalDecision(
                approved=approved,
                reviewer_id="terminal",
                reason=None if approved else "Denied at the terminal.",
            )
            try:
                handler.submit_decision(request.request_id, decision)
            except KeyError:
                console.print("[yellow]Too late: the request already timed out.[/yellow]")


@app.command()
def validate(
    file: Annotated[
        str,
        typer.Argument(help="Path to .superagent configuration file to validate"),
    ],
    check_env: Annotated[
        bool,
        typer.Option("--check-env/--no-check-env", help="Check environment variables"),
    ] = True,
    allow_missing_env: Annotated[
        bool,
        typer.Option(
            "--allow-missing-env",
            help="Report missing environment variables as a warning instead of failing",
        ),
    ] = False,
    check_refs: Annotated[
        bool,
        typer.Option("--check-refs/--no-check-refs", help="Validate cross-agent references"),
    ] = True,
) -> None:
    """Validate a .superagent configuration file.

    Performs dry-run validation without building the agent:
    - YAML syntax check
    - Schema validation
    - Cross-agent reference validation, at every depth (optional)
    - Environment variable availability check, for the file and every
      cross-agent file (optional)

    Exits 1 when anything fails, including a missing environment variable
    (pass --allow-missing-env to only warn, or --no-check-env to skip the
    check, e.g. in CI where the secrets are not set).

    Examples:
        promptise validate my_agent.superagent
        promptise validate my_agent.superagent --no-check-env
        promptise validate my_agent.superagent --allow-missing-env
    """
    from .superagent import SuperAgentLoader

    console.print(f"[bold]Validating {file}...[/bold]")

    # Load and parse file
    try:
        loader = SuperAgentLoader.from_file(file)
        console.print("[green]✓[/green] File format and schema valid")
    except SuperAgentValidationError as exc:
        console.print("[red]✗ Schema validation failed:[/red]")
        console.print(str(exc))
        raise typer.Exit(1)
    except SuperAgentError as exc:
        console.print(f"[red]✗ Failed to load file:[/red] {escape(str(exc))}")
        raise typer.Exit(1)

    # Check cross-agent references (the whole tree, without resolving env vars)
    loaders = [loader]
    if check_refs and loader.schema.cross_agents:
        try:
            loader.resolve_cross_agents(recursive=True, resolve_env=False)
        except SuperAgentError as exc:
            console.print(f"[red]✗ Cross-agent reference error:[/red] {escape(str(exc))}")
            raise typer.Exit(1)
        stack = list((loader.cross_loaders or {}).values())
        while stack:
            child = stack.pop()
            loaders.append(child)
            stack.extend((child.cross_loaders or {}).values())
        console.print(f"[green]✓[/green] All {len(loaders) - 1} cross-agent reference(s) valid")

    # Check environment variables
    if check_env:
        try:
            missing = {str(ld.file_path): ld.validate_env_vars() for ld in loaders}
        except SuperAgentError as exc:
            console.print(f"[red]✗ {escape(str(exc))}[/red]")
            raise typer.Exit(1)
        missing = {path: names for path, names in missing.items() if names}
        if missing:
            colour, mark = ("yellow", "⚠") if allow_missing_env else ("red", "✗")
            console.print(f"[{colour}]{mark} Missing environment variables:[/{colour}]")
            for path, names in missing.items():
                where = "" if path == str(loader.file_path) else f"  (in {path})"
                for var in names:
                    console.print(f"  - {var}{escape(where)}")
            console.print(
                f"[{colour}]Set these variables (or put them in .env), or use defaults "
                "(${VAR:-default}).[/" + colour + "]"
            )
            if not allow_missing_env:
                console.print(
                    "[dim]Pass --allow-missing-env to only warn, or --no-check-env "
                    "to skip this check.[/dim]"
                )
                raise typer.Exit(1)
        else:
            console.print("[green]✓[/green] All environment variables available")

    console.print("[bold green]✓ Validation complete![/bold green]")


@app.command()
def init(
    output: Annotated[
        str,
        typer.Option("--output", "-o", help="Output file path"),
    ] = "agent.superagent",
    template: Annotated[
        str,
        typer.Option(
            "--template",
            "-t",
            help="Template type: basic, http, stdio, cross-agent, advanced",
        ),
    ] = "basic",
    force: Annotated[
        bool,
        typer.Option("--force/--no-force", help="Overwrite existing file"),
    ] = False,
) -> None:
    """Generate a template .superagent configuration file.

    Creates a starter .superagent file with common patterns and best practices.

    Templates:
      basic       - Minimal HTTP server configuration
      http        - HTTP server with auth headers
      stdio       - Local stdio server configuration
      cross-agent - Multi-agent setup with cross-agent communication
      advanced    - Full-featured example with all options

    Examples:
        promptise init
        promptise init --output my_agent.superagent --template http
        promptise init -o advanced.superagent -t advanced --force
    """
    from pathlib import Path

    output_path = Path(output)

    # Check if file exists
    if output_path.exists() and not force:
        console.print(f"[yellow]File already exists: {output}[/yellow]")
        console.print("Use --force to overwrite")
        raise typer.Exit(1)

    # Template content
    templates = {
        "basic": """version: "1.0"

agent:
  model: "openai:gpt-4.1"
  instructions: "You are a helpful assistant."
  trace: true

servers:
  example:
    type: http
    url: "http://127.0.0.1:8000/mcp"
    transport: http
""",
        "http": """version: "1.0"

agent:
  model: "openai:gpt-4.1"
  instructions: "You are a helpful assistant with access to external tools."
  trace: true

servers:
  api_server:
    type: http
    url: "https://api.example.com/mcp"
    transport: http
    # Sent as "Authorization: Bearer <token>". For a pre-shared key use
    # api_key: "${API_KEY}" instead (sent as "x-api-key: <key>").
    bearer_token: "${API_TOKEN}"
    headers:
      Content-Type: "application/json"
""",
        "stdio": """version: "1.0"

agent:
  model: "openai:gpt-4.1"
  instructions: "You are a helpful assistant with local tools."
  trace: true

servers:
  local_tools:
    type: stdio
    command: python
    args:
      - "-m"
      - "mypackage.server"
    env:
      API_KEY: "${MY_API_KEY}"
      DEBUG: "false"
    cwd: null
    keep_alive: true
""",
        "cross-agent": """version: "1.0"

agent:
  model: "openai:gpt-4.1"
  instructions: "You are a coordinator agent that can delegate to specialists."
  trace: true

servers:
  general_tools:
    type: http
    url: "http://127.0.0.1:8000/mcp"
    transport: http

cross_agents:
  math_specialist:
    file: ./agents/math_agent.superagent
    description: "Specialized agent for mathematical calculations and analysis"

  research_specialist:
    file: ./agents/research_agent.superagent
    description: "Specialized agent for web research and fact-checking"
""",
        "advanced": """version: "1.0"

agent:
  # Detailed model configuration
  model:
    provider: openai
    name: gpt-4.1
    api_key: ${OPENAI_API_KEY}
    temperature: 0.7
    max_tokens: 4096
    timeout: 60

  instructions: |
    You are an advanced AI assistant with access to multiple tools and
    specialist agents. Use available tools to gather information and
    delegate complex tasks to specialist agents when appropriate.

  trace: true

servers:
  # HTTP server with authentication
  remote_api:
    type: http
    url: "https://api.example.com/mcp"
    transport: http
    bearer_token: ${API_TOKEN}
    headers:
      X-Custom-Header: "value"

  # Local stdio server
  local_tools:
    type: stdio
    command: python
    args: ["-m", "mytools.server", "--port", "3000"]
    env:
      API_KEY: ${TOOL_API_KEY}
      LOG_LEVEL: "info"
    cwd: /tmp
    keep_alive: true

cross_agents:
  specialist_a:
    file: ./agents/specialist_a.superagent
    description: "Domain expert for task A"

  specialist_b:
    file: ./agents/specialist_b.superagent
    description: "Domain expert for task B"
""",
    }

    if template not in templates:
        console.print(f"[red]Unknown template: {template}[/red]")
        console.print(f"Available: {', '.join(templates.keys())}")
        raise typer.Exit(1)

    # Write template
    try:
        output_path.write_text(templates[template], encoding="utf-8")
        console.print(f"[green]✓ Created {output}[/green]")
        console.print(f"[dim]Template: {template}[/dim]")
        console.print("\n[bold]Next steps:[/bold]")
        console.print(f"  1. Edit {output} to customize configuration")
        console.print("  2. Set required environment variables")
        console.print(f"  3. Run: promptise agent {output}")
    except Exception as exc:
        console.print(f"[red]Failed to write file:[/red] {exc}")
        raise typer.Exit(1)


_PROFILES = ("read-only", "standard", "full")
_AUTH_MODES = ("passthrough", "env-token", "api-key", "none")
_APPROVAL_MODES = ("elicitation", "pending")
_MCPCAST_MODEL = "openai:gpt-5-mini"
_MCPCAST_EVAL_TASKS = (
    20  # == promptise.mcpcast.readiness.DEFAULT_EVAL_TASKS (kept lazy-import free)
)
_MCPCAST_TRANSPORT, _MCPCAST_HOST, _MCPCAST_PORT = "stdio", "127.0.0.1", 8080


def _interactive_terminal() -> bool:
    """Whether stdin and stdout are a terminal the guided setup can take over."""
    import sys

    return sys.stdin.isatty() and sys.stdout.isatty()


def _is_file(candidate: str) -> bool:
    """Whether *candidate* names an existing regular file.

    ``Path.is_file()`` swallows only "not found"-class errors; a string
    longer than the filesystem's name limit (an inline plan document)
    raises ``ENAMETOOLONG``, which is just as much "not a file".
    """
    from pathlib import Path

    try:
        return Path(candidate).is_file()
    except (OSError, ValueError):
        return False


def _eval_credential_configured(auth: Any) -> bool:
    """Whether ``--eval`` live reads will carry a real upstream credential.

    Mirrors what the evaluation honours: ``MCPCAST_EVAL_HEADERS`` or
    ``MCPCAST_EVAL_AUTHORIZATION`` for any mode; an existing
    ``MCPCAST_UPSTREAM_TOKEN`` for env-token; for api-key, an
    ``MCPCAST_UPSTREAM_TOKENS`` entry for the evaluation tenant (any other
    entry is not used by the evaluation's own key).
    """
    import json
    import os

    from .mcpcast.readiness import EVAL_TENANT
    from .mcpcast.schema import AuthMode

    if os.environ.get("MCPCAST_EVAL_HEADERS") or os.environ.get("MCPCAST_EVAL_AUTHORIZATION"):
        return True
    if auth is AuthMode.ENV_TOKEN:
        return bool(os.environ.get("MCPCAST_UPSTREAM_TOKEN"))
    if auth is AuthMode.API_KEY:
        try:
            tokens = json.loads(os.environ.get("MCPCAST_UPSTREAM_TOKENS") or "{}")
        except json.JSONDecodeError:
            return False  # the evaluation reports the malformed value itself
        return isinstance(tokens, dict) and bool(tokens.get(EVAL_TENANT))
    return False


_TERMINAL_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
"""C0 controls except tab and newline, DEL, and the C1 range (0x9b is a one-byte CSI).

Rich strips only a handful of them; ESC and C1 CSI reach the terminal, where
``\\x1b[2K`` blanks the row being printed and a cursor move forges the next
cells — enough to hide a destructive tool from the human review table.
"""


def _console_safe(text: str) -> str:
    """*text* as an inert Rich cell: control characters become spaces, markup is escaped.

    For every string that comes from the spec, a plan file or the model —
    descriptions, paths, parameter names, operation ids, drop reasons.
    """
    return escape(_TERMINAL_CONTROL.sub(" ", text))


def _print_plan_review(plan: Any, out: Console) -> None:
    """Render the plan as kept / dropped tables for ``--review``.

    Plan text (descriptions, reasons, paths, names) comes from the spec or
    the model and goes through :func:`_console_safe`: square brackets must
    never be read as Rich markup, and a terminal escape sequence must never
    blank or rewrite a row of the one table a reviewer relies on.
    """
    from .mcpcast.plan import derive_tool_name

    kept = Table(title=f"Tools ({len(plan.tools)}) — profile {plan.profile.value}")
    kept.add_column("Tool", style="bold")
    kept.add_column("Risk")
    kept.add_column("Approval")
    kept.add_column("Operations")
    kept.add_column("Params")
    kept.add_column("Description")
    for tool in plan.tools:
        risk_style = {"read": "green", "write": "yellow"}.get(tool.risk.value, "red")
        notes = []
        if len(tool.routes) > 1:
            notes.append(f"merged {len(tool.routes)} ops")
        derived = derive_tool_name(tool.routes[0].operation_id)
        if tool.name != derived:
            notes.append(f"renamed from {derived}")
        # Scrub before truncating: a sequence cut in half is still a sequence.
        lines = _TERMINAL_CONTROL.sub(" ", tool.description).strip().splitlines()
        first_line = lines[0] if lines else ""
        desc = _console_safe(first_line[:80] + ("…" if len(first_line) > 80 else ""))
        if notes:
            desc += f"\n[dim]{_console_safe('; '.join(notes))}[/dim]"
        kept.add_row(
            _console_safe(tool.name),
            f"[{risk_style}]{tool.risk.value}[/{risk_style}]",
            "[red]required[/red]" if tool.requires_approval else "—",
            _console_safe(", ".join(f"{r.method} {r.path}" for r in tool.routes)),
            _console_safe(", ".join(tool.visible_params) or "—"),
            desc,
        )
    out.print(kept)
    if plan.dropped:
        dropped = Table(title=f"Not exposed ({len(plan.dropped)})")
        dropped.add_column("Operation", style="dim")
        dropped.add_column("Reason")
        for d in plan.dropped:
            dropped.add_row(_console_safe(d.operation_id), _console_safe(d.reason))
        out.print(dropped)


@app.command()
def mcpcast(
    spec: Annotated[
        str | None,
        typer.Argument(
            help=(
                "OpenAPI spec (file path, URL, or JSON) — or an existing mcpcast.plan.yaml "
                "to regenerate the server from an edited plan. Omit it to open the guided setup."
            ),
            show_default=False,
        ),
    ] = None,
    interactive: Annotated[
        bool,
        typer.Option(
            "--interactive",
            "-i",
            help=(
                "Open the guided setup (the default when SPEC is omitted), pre-filled with "
                "SPEC, --base-url, --out, --model, --eval-tasks and --force"
            ),
        ),
    ] = False,
    out: Annotated[
        str | None,
        typer.Option(
            "--output",
            "--out",
            "-o",
            help=(
                "Output directory (default: ./<name>-mcp, or the plan's own directory when "
                "SPEC is a mcpcast.plan.yaml). server.py and README.md are regenerated; a "
                "directory that already has files in it is not written to without --force"
            ),
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help=(
                "Write into the output directory although it already has files in it (an "
                "mcpcast project, or anything else: README.md, server.py and tests/ are replaced)"
            ),
        ),
    ] = False,
    base_url: Annotated[
        str | None,
        typer.Option("--base-url", help="API base URL (if the spec does not declare one)"),
    ] = None,
    profile: Annotated[
        str | None,
        typer.Option("--profile", help="Safety profile: read-only (default), standard, full"),
    ] = None,
    auth: Annotated[
        str | None,
        typer.Option(
            "--auth",
            help=(
                "Upstream auth: passthrough (default; callers forward their own token over "
                "HTTP), env-token (one token from MCPCAST_UPSTREAM_TOKEN — for stdio/desktop "
                "clients), api-key (per-tenant), none"
            ),
        ),
    ] = None,
    approval: Annotated[
        str | None,
        typer.Option(
            "--approval",
            help="Who approves gated calls: elicitation, pending (default depends on --auth)",
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option("--name", help="Server name (default: derived from the spec title)"),
    ] = None,
    curate: Annotated[
        bool,
        typer.Option(
            "--curate/--no-curate",
            help="LLM-assisted tool design (default on); --no-curate is fully offline",
        ),
    ] = True,
    model: Annotated[
        str,
        typer.Option("--model", help="Model for curation and --eval"),
    ] = _MCPCAST_MODEL,
    max_tools: Annotated[
        int | None,
        typer.Option("--max-tools", help="Tool budget (curation default 25; unlimited otherwise)"),
    ] = None,
    review: Annotated[
        bool,
        typer.Option("--review", help="Show the kept/dropped plan and confirm before writing"),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="With --review: write without asking"),
    ] = False,
    evaluate: Annotated[
        bool,
        typer.Option("--eval", help="Score the result with a real agent (Agent Readiness)"),
    ] = False,
    eval_tasks: Annotated[
        int,
        typer.Option("--eval-tasks", help="Number of tasks to generate for --eval"),
    ] = _MCPCAST_EVAL_TASKS,
    serve: Annotated[
        bool,
        typer.Option("--serve", help="Run the generated server immediately"),
    ] = False,
    transport: Annotated[
        str,
        typer.Option("--transport", "-t", help="With --serve: stdio, http, or sse"),
    ] = _MCPCAST_TRANSPORT,
    host: Annotated[
        str,
        typer.Option("--host", help="With --serve: bind host for HTTP/SSE"),
    ] = _MCPCAST_HOST,
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="With --serve: bind port for HTTP/SSE"),
    ] = _MCPCAST_PORT,
    public: Annotated[
        bool,
        typer.Option(
            "--public",
            help=(
                "With --serve: let an env-token or none server bind a non-loopback address. "
                "Such a server has no MCP-level authentication and acts with the operator's "
                "credential — only behind an authenticating gateway"
            ),
        ),
    ] = False,
) -> None:
    """Turn an existing API into a curated, safe, agent-ready MCP server.

    Parses the OpenAPI spec, classifies every operation's risk, designs the
    tool surface (with a model, or deterministically with --no-curate),
    and emits an editable Promptise MCPServer project: server.py,
    mcpcast.plan.yaml and a README with Claude / Cursor / Claude Code install
    snippets. Read-only by default; write, destructive and financial tools
    are only generated under --profile standard / full and always demand
    human approval, enforced server-side.

    Examples:
        promptise mcpcast                                  # guided setup in the terminal
        promptise mcpcast openapi.yaml --no-curate
        promptise mcpcast https://api.example.com/openapi.json --profile standard --review
        promptise mcpcast openapi.yaml --profile full --auth api-key --eval
        promptise mcpcast mcpcast.plan.yaml --out .        # regenerate after editing the plan
        promptise mcpcast openapi.yaml --no-curate --serve -t http
    """
    import contextlib
    import sys
    from pathlib import Path

    from .mcpcast import (
        MCPcastError,
        MCPcastPlan,
        api_name_from_spec,
        build_plan,
        describe_written,
        extract_operations,
        is_plan_document,
        is_url,
        load_generated_server,
        load_spec,
        plain_http_hosts,
        public_url,
        spec_summary_line,
        write_project,
    )
    from .mcpcast import curate as run_curation
    from .mcpcast.readiness import evaluate as run_evaluation
    from .mcpcast.readiness import write_eval
    from .mcpcast.schema import ApprovalMode, AuthMode, SafetyProfile

    # All progress goes to stderr: with --serve over stdio, stdout is the
    # MCP protocol stream. Model calls (curation, eval) are wrapped the same
    # way — build_agent() prints its discovery notice to stdout.
    err = Console(stderr=True)
    quiet_stdout = contextlib.redirect_stdout(sys.stderr)

    if profile is not None and profile not in _PROFILES:
        raise typer.BadParameter(f"--profile must be one of {', '.join(_PROFILES)}")
    if auth is not None and auth not in _AUTH_MODES:
        raise typer.BadParameter(f"--auth must be one of {', '.join(_AUTH_MODES)}")
    if approval is not None and approval not in _APPROVAL_MODES:
        raise typer.BadParameter(f"--approval must be one of {', '.join(_APPROVAL_MODES)}")
    if transport not in ("stdio", "http", "sse"):
        raise typer.BadParameter("--transport must be one of stdio, http, sse")
    if max_tools is not None and max_tools < 1:
        raise typer.BadParameter("--max-tools must be >= 1")
    if eval_tasks < 1:
        raise typer.BadParameter("--eval-tasks must be >= 1")
    if public and not serve:
        raise typer.BadParameter("--public only applies with --serve")

    if spec is None or interactive:
        # The guided setup collects every other option itself; SPEC, --base-url,
        # --out, --model, --eval-tasks and --force only pre-fill it.
        for flag, conflicts in (
            ("--profile", profile is not None),
            ("--auth", auth is not None),
            ("--approval", approval is not None),
            ("--name", name is not None),
            ("--max-tools", max_tools is not None),
            ("--no-curate", not curate),
            ("--review", review),
            ("--yes", yes),
            ("--eval", evaluate),
            ("--serve", serve),
            ("--transport", transport != _MCPCAST_TRANSPORT),
            ("--host", host != _MCPCAST_HOST),
            ("--public", public),
            ("--port", port != _MCPCAST_PORT),
        ):
            if conflicts:
                raise typer.BadParameter(
                    f"{flag} cannot be combined with the guided setup — choose it in the "
                    "wizard, or pass SPEC to run non-interactively"
                )
        if not _interactive_terminal():
            typer.echo(
                "Error: the guided setup needs an interactive terminal. Pass the spec to run "
                "non-interactively, e.g.  promptise mcpcast openapi.json --no-curate  "
                "(promptise mcpcast --help lists every option).",
                err=True,
            )
            raise typer.Exit(code=1)
        from .mcpcast.wizard import run_wizard

        result = run_wizard(
            spec,
            base_url=base_url,
            out_dir=out,
            model=model if model != _MCPCAST_MODEL else None,
            eval_tasks=eval_tasks,
            force=force,
        )
        if result is None:
            err.print("Nothing written.")
            raise typer.Exit(code=1)
        plan = result.plan
        shown_dir = result.out_dir
        with contextlib.suppress(ValueError):
            shown_dir = result.out_dir.relative_to(Path.cwd())
        if result.report:
            readiness = (
                f"\n  Agent Readiness: {result.report.grade} "
                f"({result.report.tasks_succeeded}/{result.report.tasks_total} tasks)"
            )
        elif result.eval_requested:
            readiness = (
                "\n  Agent Readiness: not completed — run it with\n"
                f"    promptise mcpcast {escape(str(shown_dir / 'mcpcast.plan.yaml'))} --eval"
            )
        else:
            readiness = ""
        err.print(
            Panel.fit(
                f"[bold]{escape(plan.api.name)}[/bold] → {escape(str(shown_dir))}/\n"
                f"  tools: {len(plan.tools)}  ({len(plan.gated_tools)} require human approval)\n"
                f"  not exposed: {len(plan.dropped)} operations (with reasons in the plan)\n"
                f"  files: {escape(describe_written(result.written, result.out_dir))}"
                f"{readiness}",
                title="mcpcast",
            )
        )
        err.print(f"[dim]Next time, without the wizard:[/dim]\n  {escape(result.command)}")
        return

    try:
        # Inline text (a JSON document or a YAML one starting with its
        # version key) is never a path: a plan longer than the filesystem's
        # name limit would otherwise crash the path checks below.
        is_inline = spec.lstrip().startswith(("{", "openapi:", "swagger:"))
        document = load_spec(spec)
        operations = None
        from_plan = is_plan_document(document)
        if from_plan:
            for flag, value in (
                ("--profile", profile),
                ("--auth", auth),
                ("--approval", approval),
                ("--base-url", base_url),
                ("--name", name),
                ("--max-tools", max_tools),
            ):
                if value is not None:
                    raise typer.BadParameter(
                        f"{flag} cannot be combined with a plan file — edit mcpcast.plan.yaml "
                        "and regenerate instead"
                    )
            # The document is already loaded (file, URL or inline text alike).
            plan = MCPcastPlan.from_document(document)
            plan_label = "<inline>" if is_inline else public_url(spec)
            err.print(
                f"[dim]Regenerating from plan {escape(plan_label)} ({len(plan.tools)} tools)[/dim]"
            )
            if evaluate and plan.api.spec_source:
                # Spec-derived mock responses need the original spec; the plan
                # records where it came from.
                try:
                    source_doc = load_spec(plan.api.spec_source)
                except MCPcastError as exc:
                    raise MCPcastError(
                        f"--eval needs the spec recorded in the plan (spec_source="
                        f"{plan.api.spec_source!r}) for realistic mocks, but it could not "
                        f"be loaded: {exc}. Fix api.spec_source in the plan or run --eval "
                        "against the spec instead."
                    ) from exc
                operations = extract_operations(
                    source_doc,
                    base_url=plan.api.base_url,
                    spec_url=plan.api.spec_source if is_url(plan.api.spec_source) else None,
                )
        else:
            safety = SafetyProfile(profile or "read-only")
            auth_mode = AuthMode(auth or "passthrough")
            approval_mode = ApprovalMode(approval) if approval else None
            # A spec URL may carry a credential so the *fetch* is authenticated
            # (load_spec above used it); nothing derived from the URL — the
            # base URL, the api name, the plan's spec_source, a log line —
            # may keep it.
            shown_spec = public_url(spec)
            spec_label = "<inline>" if is_inline else shown_spec
            operations = extract_operations(
                document, base_url=base_url, spec_url=shown_spec if is_url(spec) else None
            )
            api_name = name or api_name_from_spec(document, shown_spec)
            description = spec_summary_line(document)
            err.print(
                f"[dim]Parsed {len(operations)} operations from {escape(spec_label)}; "
                f"profile={safety.value} auth={auth_mode.value}[/dim]"
            )
            if curate:
                err.print(
                    f"[dim]Curating with {escape(model)} (budget {max_tools or 25} tools)…[/dim]"
                )
                try:
                    with quiet_stdout:
                        plan = asyncio.run(
                            run_curation(
                                operations,
                                model=model,
                                max_tools=max_tools or 25,
                                profile=safety,
                                base_url=base_url,
                                auth=auth_mode,
                                approval=approval_mode,
                                name=api_name,
                                description=description,
                                spec_source=spec_label,
                            )
                        )
                except MCPcastError:
                    raise
                except Exception as exc:  # provider/credential failures, not plan problems
                    raise MCPcastError(
                        f"model {model!r} could not be used for curation: "
                        f"{type(exc).__name__}: {exc}\nSet the provider API key, or re-run "
                        "with --no-curate for the offline path."
                    ) from exc
            else:
                plan = build_plan(
                    operations,
                    profile=safety,
                    base_url=base_url,
                    auth=auth_mode,
                    approval=approval_mode,
                    name=api_name,
                    description=description,
                    spec_source=spec_label,
                    max_tools=max_tools,
                )

        if public and plan.api.auth not in (AuthMode.ENV_TOKEN, AuthMode.NONE):
            # Only the loopback-only modes have a --public switch; the others
            # authenticate every caller and bind any host as a matter of course.
            raise typer.BadParameter(
                f"--public only applies to --auth env-token / none; this project uses "
                f"{plan.api.auth.value}, which already binds any host"
            )

        if review:
            _print_plan_review(plan, err)
            if not yes and not typer.confirm("Write the project?", default=True, err=True):
                err.print("Aborted — nothing written.")
                raise typer.Exit(code=1)

        plan_file = (
            Path(spec)
            if from_plan and not is_url(spec) and not is_inline and _is_file(spec)
            else None
        )
        if out:
            out_dir = Path(out)
        elif plan_file is not None:
            out_dir = plan_file.resolve().parent
        else:
            # A plan fetched from a URL or given inline has no directory of its own.
            out_dir = Path(f"{plan.api.name}-mcp")
        existing_plan = out_dir / "mcpcast.plan.yaml"
        regenerating_in_place = (
            plan_file is not None and existing_plan.resolve() == plan_file.resolve()
        )
        if existing_plan.exists() and not regenerating_in_place and not force:
            raise MCPcastError(
                f"{out_dir} already contains an mcpcast project — pass --force to overwrite "
                f"it, or regenerate from {existing_plan} to keep your edits"
            )
        if out_dir.is_dir() and not existing_plan.exists() and not force and any(out_dir.iterdir()):
            # Somebody else's folder: README.md, server.py and tests/ would be replaced.
            raise MCPcastError(
                f"{out_dir} has files in it and is not an mcpcast project — choose another "
                "--out, or pass --force to write into it anyway"
            )
        try:
            written = write_project(
                plan, out_dir, write_plan=not regenerating_in_place, force=force
            )
        except MCPcastError:
            raise
        except Exception as exc:  # permissions, a character the encoder refuses, …
            raise MCPcastError(f"could not write {out_dir}: {type(exc).__name__}: {exc}") from exc
        gated = len(plan.gated_tools)
        err.print(
            Panel.fit(
                f"[bold]{escape(plan.api.name)}[/bold] → {escape(str(out_dir))}/\n"
                f"  tools: {len(plan.tools)}  ({gated} require human approval)\n"
                f"  not exposed: {len(plan.dropped)} operations (with reasons in the plan)\n"
                f"  files: {escape(describe_written(written, out_dir))}",
                title="mcpcast",
            )
        )
        insecure_hosts = plain_http_hosts(plan)
        if insecure_hosts:
            err.print(
                f"[yellow]Warning:[/yellow] the upstream credential would travel over plain "
                f"http:// to {escape(', '.join(insecure_hosts))}. The generated server refuses "
                "every call to such a host with UPSTREAM_INSECURE until "
                "MCPCAST_ALLOW_INSECURE_HTTP=1 accepts the risk (an intranet API, a staging "
                "box) — set it where the server runs, or point the plan at an https:// base_url."
            )

        module: Any = None
        if evaluate or serve:
            try:
                module = load_generated_server(out_dir / "server.py")
            except (ImportError, RuntimeError, SyntaxError, ValueError) as exc:
                raise MCPcastError(f"could not import {out_dir / 'server.py'}: {exc}") from exc

        if evaluate:
            if not plan.tools:
                raise MCPcastError("nothing to evaluate: the plan exposes no tools")
            if plan.api.auth is not AuthMode.NONE and not _eval_credential_configured(
                plan.api.auth
            ):
                err.print(
                    f"[yellow]Hint:[/yellow] auth is {plan.api.auth.value} — set "
                    "MCPCAST_EVAL_AUTHORIZATION='Bearer <token>' so live reads reach the API "
                    "authenticated (a placeholder credential is used otherwise)."
                )
            from .mcpcast.readiness import base_url_override

            override = base_url_override()
            if override and override != plan.api.base_url:
                err.print(
                    f"[dim]Live reads go to MCPCAST_BASE_URL={escape(override)} "
                    f"(the plan says {escape(plan.api.base_url)})[/dim]"
                )
            err.print(f"[dim]Evaluating with {escape(model)} ({eval_tasks} tasks)…[/dim]")
            try:
                with quiet_stdout:
                    report = asyncio.run(
                        run_evaluation(
                            plan,
                            module.build_server,
                            model=model,
                            tasks=eval_tasks,
                            operations=operations,
                        )
                    )
            except MCPcastError:
                raise
            except Exception as exc:
                raise MCPcastError(
                    f"model {model!r} could not be used for the evaluation: "
                    f"{type(exc).__name__}: {exc}\nSet the provider API key and retry."
                ) from exc
            tasks_path, report_path = write_eval(report, [r.task for r in report.results], out_dir)
            err.print(escape(report.render_summary()))
            err.print(
                f"[dim]Report: {escape(str(report_path))}  Tasks: {escape(str(tasks_path))}[/dim]"
            )

        if serve:
            err.print(f"[dim]Serving {escape(str(out_dir / 'server.py'))} over {transport}…[/dim]")
            # The generated entry point carries the auth-mode bind guard; its
            # configuration (MCPCAST_* variables) is validated when it starts.
            argv = ["--transport", transport, "--host", host, "--port", str(port)]
            if public:
                argv.append("--public")
            try:
                module.main(argv)
            except (RuntimeError, ValueError) as exc:
                raise MCPcastError(f"could not start {out_dir / 'server.py'}: {exc}") from exc
    except MCPcastError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)


@app.command()
def serve(
    target: Annotated[
        str,
        typer.Argument(help="Server target as 'module.path:attribute' (e.g. myapp.server:server)"),
    ],
    transport: Annotated[
        str,
        typer.Option("--transport", "-t", help="Transport type: stdio, http, or sse"),
    ] = "stdio",
    host: Annotated[
        str,
        typer.Option("--host", help="Bind host for HTTP/SSE"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Bind port for HTTP/SSE"),
    ] = 8080,
    dashboard: Annotated[
        bool,
        typer.Option("--dashboard", help="Enable the live terminal dashboard (http/sse only)"),
    ] = False,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Hot-reload on source changes (development only)"),
    ] = False,
    allowed_host: Annotated[
        list[str] | None,
        typer.Option(
            "--allowed-host",
            metavar="HOST",
            help=(
                "Host header value to accept on HTTP/SSE, e.g. api.example.com or "
                "api.example.com:* (repeatable). A loopback bind validates Host and Origin "
                "against the loopback names by default; a non-loopback bind validates only "
                "when this is given."
            ),
        ),
    ] = None,
    allowed_origin: Annotated[
        list[str] | None,
        typer.Option(
            "--allowed-origin",
            metavar="ORIGIN",
            help=(
                "Origin header value to accept for browser clients, e.g. "
                "https://app.example.com (repeatable). Requires --allowed-host on a "
                "non-loopback bind."
            ),
        ),
    ] = None,
) -> None:
    """Run an MCP server from a Python module.

    Imports the ``module:attribute`` target, validates it is an
    ``MCPServer``, and serves it over the chosen transport — the MCP
    equivalent of ``uvicorn myapp:app``.  ``--allowed-host`` and
    ``--allowed-origin`` are the transport's ``Host``/``Origin`` allow-lists:
    a loopback bind validates against the loopback names on its own and the
    lists add to them (a reverse proxy forwarding a public ``Host`` to
    ``127.0.0.1``); a non-loopback bind is unrestricted until
    ``--allowed-host`` names the hosts it serves.

    Examples:
        promptise serve myapp.server:server
        promptise serve myapp.server:server --transport http --port 8080
        promptise serve myapp.server:server -t http --dashboard
        promptise serve myapp.server:server -t http --reload
        promptise serve myapp.server:server -t http --host 0.0.0.0 \\
            --allowed-host api.example.com --allowed-origin https://app.example.com
    """
    import argparse as _argparse

    from .mcp.server import MCPServer
    from .mcp.server._serve_cli import resolve_server, run_serve

    if transport not in ("stdio", "http", "sse"):
        raise typer.BadParameter(
            f"Invalid transport {transport!r}: must be one of stdio, http, sse"
        )

    try:
        server = resolve_server(target)
    except (ValueError, ImportError, AttributeError) as exc:
        # stderr: with the stdio transport, stdout is the MCP protocol stream
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    if not isinstance(server, MCPServer):
        typer.echo(
            f"Error: {target!r} resolved to {type(server).__name__}, not an MCPServer instance",
            err=True,
        )
        raise typer.Exit(code=1)

    if dashboard and transport == "stdio":
        typer.echo(
            "Warning: --dashboard has no effect with the stdio transport "
            "(the terminal is the protocol stream); use --transport http or sse.",
            err=True,
        )

    # The same names build_serve_parser() produces: run_serve() forwards the
    # allow-lists to server.run() / hot_reload() only when they are given, so
    # an empty list here means the bind-derived default policy.
    args = _argparse.Namespace(
        target=target,
        transport=transport,
        host=host,
        port=port,
        dashboard=dashboard,
        reload=reload,
        allowed_hosts=list(allowed_host or []),
        allowed_origins=list(allowed_origin or []),
    )
    run_serve(args, server=server)
