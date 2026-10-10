"""``promptise models`` — see, check and configure model providers.

::

    promptise models list                      # every provider, route, env status
    promptise models check azure:my-deployment # what is missing for this string
    promptise models check openai:gpt-5-mini --ping   # plus a real one-token call
    promptise models env azure                 # export lines to paste into .env
"""

from __future__ import annotations

import asyncio
import ipaddress
import os
import socket
from collections.abc import Iterator
from typing import Annotated
from urllib.parse import urlsplit

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .models import (
    PROVIDERS,
    EnvVar,
    ModelSetupError,
    Provider,
    check_model,
    dotenv_origin,
    env_template,
    find_provider,
    route_url,
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
        url = route_url(p)
        route = (
            "native integration (core)"
            if url is None
            else f"OpenAI-compatible endpoint {escape(url)} (core, nothing to install)"
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
        target = _target_url(p)
        if ping:
            _console.print("[green]Configuration OK.[/green]")
        elif target is not None and (p.key_optional or _is_local(target)):
            # A keyless or local server: configuration alone proves nothing, and
            # a TCP connect is cheap — say whether anything is listening.
            origin = _origin(target)
            if not _reachable(target):
                _console.print(
                    f"[red]Not reachable.[/red] The configuration is complete, but nothing is "
                    f"listening at {escape(origin)}{escape(_server_hint(p))}"
                )
                raise typer.Exit(code=1)
            _console.print(
                f"[green]Configuration OK[/green] — {escape(origin)} accepts connections. "
                "Add --ping to make a real one-token call to the model."
            )
        else:
            _console.print(
                "[green]Configuration OK[/green] — every required setting is in place; nothing "
                "was called. Add --ping to make a real one-token call to the model."
            )

    if ping:
        _console.print("Pinging…", end=" ")
        try:
            reply = asyncio.run(_ping(model))
        except ModelSetupError as exc:
            _console.print(f"\n[red]{escape(str(exc))}[/red]")
            raise typer.Exit(code=1)
        except Exception as exc:  # provider/network errors: show them, do not hide them
            _console.print("[red]failed[/red]")
            _console.print(f"[red]{type(exc).__name__}: {escape(str(exc)[:600])}[/red]")
            hint = _ping_hint(exc, result.provider, result.model)
            if hint:
                _console.print(f"  → {escape(hint)}")
            raise typer.Exit(code=1)
        _console.print(f"[green]ok[/green] — replied {escape(reply)!s}")


_PROBE_TIMEOUT = 1.0
"""Seconds `models check` waits for a local or keyless server to accept a connection."""


def _target_url(p: Provider) -> str | None:
    """The URL requests go to: the OpenAI-compatible route, or a native
    provider's endpoint override (``OPENAI_BASE_URL``, ``AZURE_OPENAI_ENDPOINT``)."""
    url = route_url(p)
    if url is not None:
        return url
    for var in p.env:
        if var.word == "endpoint" and var.value():
            return var.value()
    return None


def _origin(url: str) -> str:
    """``scheme://host:port`` of *url* — what "nothing is listening at" names."""
    parts = urlsplit(url if "://" in url else f"http://{url}")
    return f"{parts.scheme}://{parts.netloc}"


def _is_local(url: str) -> bool:
    """Whether *url* points at this machine (loopback, ``localhost``, ``0.0.0.0``)."""
    host = urlsplit(url if "://" in url else f"http://{url}").hostname or ""
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def _reachable(url: str) -> bool:
    """Whether a TCP connection to *url*'s host and port succeeds within
    :data:`_PROBE_TIMEOUT` — nothing is sent."""
    parts = urlsplit(url if "://" in url else f"http://{url}")
    if not parts.hostname:
        return False
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:  # a malformed port: nothing can listen there
        return False
    try:
        with socket.create_connection((parts.hostname, port), timeout=_PROBE_TIMEOUT):
            return True
    except OSError:
        return False


def _server_hint(p: Provider) -> str:
    """What to try when a provider's server is not listening, as a ``" — ..."`` suffix."""
    if p.key == "ollama":
        if os.environ.get("OLLAMA_HOST"):
            return " — is Ollama running? (ollama serve) The address comes from OLLAMA_HOST."
        return (
            " — is Ollama running? (ollama serve) If it runs on another host or port, "
            "set OLLAMA_HOST."
        )
    names = [v.name for v in p.env if v.word == "endpoint" and v.value()]
    if names:
        return f" — is the server running? The address comes from {names[0]}."
    return " — is the server running?"


def _causes(exc: BaseException) -> Iterator[BaseException]:
    """*exc* and every exception it was raised from or during."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _ping_hint(exc: BaseException, p: Provider | None, model: str) -> str | None:
    """One line saying what a failed ping most likely means and what to do.

    Works from the exception chain without importing any provider SDK: the
    HTTP status (``status_code``, set by the OpenAI and Anthropic clients and
    LangChain's wrappers) when the server answered, else the class names
    (``...ConnectionError``, ``...Timeout...``) and the built-in socket errors.
    """
    chain = list(_causes(exc))
    names = " ".join(type(e).__name__ for e in chain)
    statuses = [getattr(e, "status_code", None) for e in chain]
    status = next((s for s in statuses if isinstance(s, int)), None)
    target = _target_url(p) if p is not None else None
    where = f" at {_origin(target)}" if target else ""
    key = next((v for v in p.env if v.word == "api_key"), None) if p is not None else None

    if status is None:
        if "Timeout" in names or any(isinstance(e, TimeoutError) for e in chain):
            if target and _is_local(target):
                return (
                    f"the server{where} accepted the connection but did not answer in time — "
                    "a model loading for the first time can take a while; try again"
                )
            return (
                f"no answer{where} in time — check the network, a proxy (HTTPS_PROXY), and "
                "that the endpoint is right"
            )
        if "Connect" in names or any(isinstance(e, ConnectionError) for e in chain):
            if target and (_is_local(target) or (p is not None and p.key_optional)):
                assert p is not None
                return f"nothing is listening{where}{_server_hint(p)}"
            return (
                f"could not connect to {_origin(target) if target else 'the provider'} — check "
                "the network, a proxy (HTTPS_PROXY), and that the endpoint is right"
            )
        return None

    if status == 401:
        if key is not None:
            return f"the provider rejected the credential — check {key.name} ({key.where})"
        return "the server rejected the request as unauthenticated — it expects a credential"
    if status == 403:
        if p is not None and p.key == "bedrock":
            return (
                f"the key is valid but may not use {model} — enable model access for it in "
                "the Bedrock console, in this region"
            )
        return f"the credential is valid but not allowed to use {model} — check its permissions"
    if status == 404:
        if p is not None and p.key == "ollama":
            return f"this Ollama has no model {model!r} — run: ollama pull {model}"
        if p is not None and p.key == "azure_openai":
            return (
                f"no deployment named {model!r}{where} — use the Name column of "
                "Foundry → Deployments, and check the endpoint belongs to that resource"
            )
        docs = f" ({p.docs})" if p is not None and p.docs else ""
        return f"the provider does not know model {model!r} — check the name{docs}"
    if status == 429:
        if any(word in str(exc).lower() for word in ("quota", "credit", "billing")):
            return "the account is out of credits or quota — add credits or raise its limits"
        return "rate limited — wait and retry, or check the account's limits"
    if status >= 500:
        return "the provider had a server error — usually temporary; try again"
    return None


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
