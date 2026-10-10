"""``promptise audit`` — check an audit log written by ``AuditMiddleware``.

::

    promptise audit verify audit.jsonl                       # key from PROMPTISE_AUDIT_SECRET
    promptise audit verify audit.jsonl --key-env AUDIT_KEY   # key from another variable
    promptise audit verify audit.1.jsonl audit.jsonl         # rotated files, one chain
    promptise audit verify audit.jsonl --json                # machine-readable report

Exit codes: 0 intact, 1 tampered (or warnings with ``--strict``), 2 the log
or the key cannot be read.  The key is read from the environment, never from
the command line, and is never printed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape

audit_app = typer.Typer(
    no_args_is_help=True,
    help="Check the HMAC-chained audit logs written by AuditMiddleware.",
)
_console = Console()
_err = Console(stderr=True)


@audit_app.command("verify")
def verify(
    files: Annotated[
        list[Path],
        typer.Argument(
            help="Audit log file(s); several files are verified as one chain, in order."
        ),
    ],
    key_env: Annotated[
        list[str] | None,
        typer.Option(
            "--key-env",
            help="Environment variable holding the HMAC key (default PROMPTISE_AUDIT_SECRET). "
            "Repeat it after a key rotation: each entry must verify with one of the keys.",
        ),
    ] = None,
    anchor: Annotated[
        str | None,
        typer.Option(
            "--anchor",
            help="An entry hash recorded earlier (the last_hash of a previous run); "
            "fails if it is no longer in the log, i.e. entries were cut from the end.",
        ),
    ] = None,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Also fail on warnings: a truncated last line, a crash fragment, a chain reset.",
        ),
    ] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Print the report as JSON.")] = False,
) -> None:
    """Verify an audit log's HMAC chain; exit 1 if it was tampered with."""
    from .mcp.server._audit_verify import verify_audit_log

    names = key_env or ["PROMPTISE_AUDIT_SECRET"]
    keys: list[str] = []
    for name in names:
        value = os.environ.get(name)
        if not value:
            _err.print(
                f"[red]Error:[/red] environment variable {escape(name)} is not set "
                "(it must hold the audit log's HMAC key)."
            )
            raise typer.Exit(code=2)
        keys.append(value)

    try:
        report = verify_audit_log([str(f) for f in files], keys, anchor=anchor)
    except OSError as exc:
        _err.print(f"[red]Error:[/red] cannot read the audit log: {escape(str(exc))}")
        raise typer.Exit(code=2)

    failed = not report.ok or (strict and bool(report.warnings))
    if as_json:
        print(json.dumps(report.to_dict(), indent=2))
        raise typer.Exit(code=1 if failed else 0)

    for issue in report.problems:
        _console.print(f"[red]TAMPERED[/red] {escape(str(issue))}")
    for issue in report.warnings:
        _console.print(f"[yellow]warning[/yellow] {escape(str(issue))}")
    restarts = len(report.restarts)
    summary = f"{report.entries} entr{'y' if report.entries == 1 else 'ies'}"
    if restarts:
        summary += f", chain continued across {restarts} restart{'s' if restarts != 1 else ''}"
    if report.ok:
        state = "intact" if report.continuous else "intact, but not one continuous chain"
        _console.print(f"[green]OK[/green] {summary}: {state}.")
        if report.last_hash:
            _console.print(f"[dim]last_hash {report.last_hash} (seq {report.last_seq})[/dim]")
    else:
        _console.print(
            f"[red]FAILED[/red] {summary}: {len(report.problems)} problem"
            f"{'s' if len(report.problems) != 1 else ''}; the first is at "
            f"{escape(str(report.first_problem).split(': ', 1)[0])}."
        )
    raise typer.Exit(code=1 if failed else 0)
