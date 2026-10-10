"""``promptise models`` — see, check and configure model providers.

::

    promptise models list                      # every provider, route, env status
    promptise models check azure:my-deployment # what is missing for this string
    promptise models check openai:gpt-5-mini --ping   # plus a real one-token call
    promptise models env azure                 # export lines to paste into .env
"""

from __future__ import annotations

import asyncio
import os
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .models import (
    PROVIDERS,
    EnvVar,
    ModelSetupError,
    check_model,
    dotenv_origin,
    env_template,
    find_provider,
)

models_app = typer.Typer(
    no_args_is_help=True,
    help="Bring your own model: list providers, check a model string, print env templates.",
)
_console = Console()


@models_app.command("list")
def list_providers() -> None:
    """List every supported provider with its prefixes, route and env status."""
    table = Table(
        title="Model providers — use as  provider:model  or  Model(model, provider=...)  "
        "— nothing to install for any of them"
    )
    table.add_column("provider= (aliases)", style="bold")
    table.add_column("Provider")
    table.add_column("Route")
    table.add_column("Env")
    table.add_column("Example")
    for p in PROVIDERS:
        prefixes = p.display + (
            f"  ({', '.join(a for a in p.prefixes if a != p.display)})"
            if len(p.prefixes) > 1
            else ""
        )
        route = "native" if p.base_url is None else "OpenAI-compatible"
        required = [v for v in p.env if v.required]
        missing = p.missing(set())
        if not required:
            env = "[dim]none required[/dim]"
        elif not missing:
            env = "[green]set[/green]"
        else:
            env = "[red]missing:[/red] " + escape(", ".join(v.name for v in missing))
        table.add_row(escape(prefixes), escape(p.title), route, env, escape(p.example))
    _console.print(table)
    _console.print(
        "[dim]Diagnose one string:  promptise models check <provider:model> [--ping]   ·   "
        "env template:  promptise models env <provider>[/dim]"
    )


def _env_state(var: EnvVar) -> str:
    """The state of one provider variable as Rich markup: set (and from where), missing, unset.

    A variable that is exported but empty is called out as such — it counts
    as not set, and the fix is to unset it or give it a value, not to look
    for a typo in the name.  A value that came from a ``.env`` file names
    the file, so a redirected endpoint or an unexpected key is traceable.
    """
    names = (var.name, *var.aliases)
    for name in names:
        if os.environ.get(name):
            origin = dotenv_origin(name)
            return "[green]set[/green]" + (f" (from {escape(origin)})" if origin else "")
    empty = [n for n in names if os.environ.get(n) == ""]
    if var.required:
        if empty:
            return (
                f"[red]MISSING[/red] ({empty[0]} is exported but empty — unset it or give it "
                "a value)"
            )
        return "[red]MISSING[/red]"
    if empty:
        return f"[dim]unset (optional; {empty[0]} is exported but empty)[/dim]"
    return "[dim]unset (optional)[/dim]"


@models_app.command("check")
def check(
    model: Annotated[str, typer.Argument(help="Model string, e.g. azure:my-deployment")],
    ping: Annotated[
        bool, typer.Option("--ping", help="Also make a real one-token call to the model")
    ] = False,
) -> None:
    """Explain what a model string resolves to and what is missing to use it."""
    result = check_model(model)
    if result.provider is None:
        _console.print(
            f"[yellow]No Promptise provider prefix in {escape(model)!s}[/yellow] — the string is "
            "handed to LangChain as-is, which infers the provider from the model name "
            "(gpt-… → openai, claude… → anthropic). Prefer an explicit prefix; run "
            "`promptise models list`."
        )
    else:
        p = result.provider
        _console.print(
            f"[bold]{escape(model)}[/bold] → {escape(result.canonical)}  ({escape(p.title)})"
        )
        _console.print(f"  model part: {escape(result.model)!s} — {escape(p.model_hint)}")
        route = (
            "native integration (core)"
            if p.base_url is None
            else f"OpenAI-compatible endpoint {escape(p.base_url)} (core, nothing to install)"
        )
        _console.print(f"  route: {route}")
        for var in p.env:
            _console.print(f"  {var.name}: {_env_state(var)} — {escape(var.where)}")
        if p.notes:
            _console.print(f"  [dim]{escape(p.notes)}[/dim]")
        if not result.ok:
            _console.print("[red]Not usable yet.[/red]")
            for problem in result.problems:
                _console.print(f"  - {escape(problem)}")
            raise typer.Exit(code=1)
        _console.print("[green]Usable.[/green]")

    if ping:
        _console.print("Pinging…", end=" ")
        try:
            reply = asyncio.run(_ping(model))
        except ModelSetupError as exc:
            _console.print(f"\n[red]{escape(str(exc))}[/red]")
            raise typer.Exit(code=1)
        except Exception as exc:  # provider/network errors: show them, do not hide them
            _console.print(f"\n[red]{type(exc).__name__}: {escape(str(exc)[:600])}[/red]")
            raise typer.Exit(code=1)
        _console.print(f"[green]ok[/green] — replied {escape(reply)!s}")


async def _ping(model: str) -> str:
    """One cheap round trip through the real model."""
    from langchain_core.messages import HumanMessage

    from .models import resolve_model

    llm = resolve_model(model)
    reply = await llm.ainvoke([HumanMessage(content="Reply with the single word: ok")])
    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    return repr(text.strip()[:40])


@models_app.command("env")
def env(
    provider: Annotated[str, typer.Argument(help="Provider prefix or alias, e.g. azure, bedrock")],
) -> None:
    """Print the environment variables a provider needs, as export lines with hints."""
    if find_provider(provider) is None:
        typer.echo(f"Error: unknown provider {provider!r}. Run `promptise models list`.", err=True)
        raise typer.Exit(code=2)
    typer.echo(env_template(provider))
