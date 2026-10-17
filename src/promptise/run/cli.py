"""``promptise run``: resolve the project, the target and the mode, then run.

:func:`run_command` is what the ``promptise run`` CLI command calls (the
legacy ``--model-id`` REPL is handled before it, in :mod:`promptise.cli`).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import platform
import shlex
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field, fields, replace
from importlib.metadata import version as _version
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .context import RunContext
from .entry import Entry, EntryError, detect_kind, load_factory, parse_ref
from .manifest import ManifestError, ProjectManifest, default_manifest, find_manifest, load_manifest

__all__ = ["RunOptions", "run_command"]

err = Console(stderr=True, highlight=False)

_DEPTH_ENV = "PROMPTISE_RUN_DEPTH"
_MAX_DEPTH = 8
_MODES = ("auto", "script", "chat", "once", "serve")


@dataclass
class RunOptions:
    """Everything ``promptise run`` was asked to do."""

    target: str | None = None
    args: list[str] = field(default_factory=list)
    manifest: str | None = None
    devtools: bool = False
    tui: bool = False
    prod: bool = False
    watch: bool = False
    mode: str = "auto"
    serve: bool = False
    input: str | None = None
    json_output: bool = False
    host: str | None = None
    port: int | None = None
    env_files: list[str] = field(default_factory=list)
    env: list[str] = field(default_factory=list)
    model: str | None = None
    observe: str | None = None
    report: str | None = None
    check: bool = False
    list_scripts: bool = False

    def to_args(self, *, exclude: frozenset[str] = frozenset()) -> list[str]:
        """Command-line arguments that reproduce these options (minus *exclude*)."""
        out: list[str] = []
        flags = {
            "devtools": "--devtools",
            "tui": "--tui",
            "prod": "--prod",
            "watch": "--watch",
            "serve": "--serve",
            "json_output": "--json",
            "check": "--check",
            "list_scripts": "--list",
        }
        values = {
            "manifest": "--manifest",
            "input": "--input",
            "host": "--host",
            "port": "--port",
            "model": "--model",
            "observe": "--observe",
            "report": "--report",
        }
        for f in fields(self):
            name = f.name
            if name in exclude or name in ("target", "args"):
                continue
            value = getattr(self, name)
            if name in flags:
                if value:
                    out.append(flags[name])
            elif name in values:
                if value is not None:
                    out += [values[name], str(value)]
            elif name == "mode":
                if value != "auto":
                    out += ["--mode", value]
            elif name == "env_files":
                for item in value:
                    out += ["--env-file", item]
            elif name == "env":
                for item in value:
                    out += ["--env", item]
        if self.target is not None and "target" not in exclude:
            out.append(self.target)
        if self.args and "args" not in exclude:
            out += ["--", *self.args]
        return out


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _problem(message: str) -> int:
    err.print(f"[bold red]promptise run:[/bold red] {escape(message)}")
    return 2


def _load_project(opts: RunOptions) -> ProjectManifest:
    if opts.manifest:
        path = Path(opts.manifest)
        if path.is_dir():
            found = find_manifest(path)
            if found is None:
                raise ManifestError(path, ["no promptise.toml or [tool.promptise] here"])
            path = found
        if not path.is_file():
            raise ManifestError(path, ["no such file"])
        return load_manifest(path)
    found = find_manifest()
    if found is None:
        return default_manifest(Path.cwd())
    return load_manifest(found)


def _list_scripts(manifest: ProjectManifest) -> int:
    if not manifest.scripts:
        where = manifest.path or "this project"
        err.print(f"[dim]no [scripts] in {escape(str(where))}[/dim]")
        return 0
    table = Table(show_edge=False, box=None, header_style="bold", pad_edge=False)
    table.add_column("script", style="cyan")
    table.add_column("command")
    for name, cmd in manifest.scripts.items():
        table.add_row(escape(name), escape(cmd))
    Console(highlight=False).print(table)
    return 0


def _forwarded_flags(opts: RunOptions) -> list[str]:
    """Our flags given with a script name, passed on when the script is a ``promptise`` command."""
    return replace(opts, target=None, args=[], manifest=None, list_scripts=False).to_args()


_CMD_SPECIAL = ("&", "|", "<", ">", "^", "%", "!", '"', "\n", "\r")


def _masked(args: list[str]) -> list[str]:
    """*args* for display: the value of every ``--env KEY=VALUE`` hidden."""
    out: list[str] = []
    hide = False
    for arg in args:
        if hide:
            key = arg.partition("=")[0]
            out.append(f"{key}=***")
            hide = False
            continue
        if arg in ("--env", "-e"):
            hide = True
        elif arg.startswith("--env="):
            out.append("--env=" + arg[len("--env=") :].partition("=")[0] + "=***")
            continue
        out.append(arg)
    return out


def _run_named_script(manifest: ProjectManifest, name: str, opts: RunOptions) -> int:
    """Run a ``[scripts]`` entry through the shell, like ``npm run``."""
    depth = int(os.environ.get(_DEPTH_ENV, "0") or 0)
    if depth >= _MAX_DEPTH:
        return _problem(f"script {name!r} calls promptise run recursively (depth {depth})")
    command = manifest.scripts[name]
    extra: list[str] = []
    ours = _forwarded_flags(opts)
    if ours:
        first = shlex.split(command, posix=os.name == "posix")[:1]
        if first and Path(first[0]).name.removesuffix(".exe") == "promptise":
            extra += ours
        else:
            err.print(
                f"[yellow]ignoring {escape(' '.join(_masked(ours)))}: script {escape(name)!r} "
                "is not a promptise command[/yellow]"
            )
    extra += opts.args
    shown = command
    if extra:
        quote: Callable[[str], str]
        if os.name == "posix":
            quote = shlex.quote
        else:
            # cmd.exe has no quoting that is safe for every character: refuse
            # the ones it would act on rather than run something else.
            unsafe = [a for a in extra if any(c in a for c in _CMD_SPECIAL)]
            if unsafe:
                return _problem(
                    f"argument {unsafe[0]!r} has a character cmd.exe would interpret "
                    f"({' '.join(_CMD_SPECIAL)}); run the command directly instead"
                )

            def quote(arg: str) -> str:  # type: ignore[misc]  # the Windows variant
                return subprocess.list2cmdline([arg])

        command = command + " " + " ".join(quote(a) for a in extra)
        shown = shown + " " + " ".join(quote(a) for a in _masked(extra))
    env = dict(os.environ)
    # Like node_modules/.bin: this Python's directory and its scripts directory
    # (where `promptise` and `pytest` live; a separate `Scripts` on Windows) first.
    import sysconfig

    dirs = [str(Path(sys.executable).parent)]
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir and scripts_dir not in dirs:
        dirs.append(scripts_dir)
    env["PATH"] = os.pathsep.join([*dirs, env.get("PATH", "")])
    env[_DEPTH_ENV] = str(depth + 1)
    env["PROMPTISE_RUN_SCRIPT"] = name
    env["PROMPTISE_PROJECT_ROOT"] = str(manifest.root)
    env.setdefault("PROMPTISE_INIT_CWD", os.getcwd())
    err.print(f"[dim]> {escape(manifest.project_name)} {escape(name)}: {escape(shown)}[/dim]")
    try:
        # A [scripts] entry is a shell command from the project's own manifest,
        # run through the shell on purpose (like npm); appended arguments are
        # quoted above, and refused on Windows when cmd.exe would act on them.
        proc = subprocess.run(command, shell=True, cwd=str(manifest.root), env=env)  # nosec B602
    except KeyboardInterrupt:
        return 130
    return proc.returncode


def _apply_env(manifest: ProjectManifest, opts: RunOptions) -> list[str]:
    """Load the env files and variables; return the problems found."""
    from ..models import ModelSetupError, load_env_file

    problems: list[str] = []
    # -e KEY=VALUE wins over everything; files and defaults never override.
    for item in opts.env:
        key, sep, value = item.partition("=")
        if not sep or not key:
            problems.append(f"--env {item!r}: expected KEY=VALUE")
            continue
        os.environ[key] = value
    # Files given on the command line load first, so they win over the manifest's.
    for item in opts.env_files:
        if not Path(item).is_file():
            problems.append(f"--env-file {item}: no such file")
    files = [Path(p) for p in opts.env_files if Path(p).is_file()]
    files += [manifest.root / p for p in manifest.env.files]
    for path in files:
        try:
            load_env_file(path)
        except ModelSetupError as exc:
            problems.append(str(exc))
    for key, value in manifest.env.vars.items():
        if not os.environ.get(key):
            os.environ[key] = value
    return problems


def _resolve_entry(manifest: ProjectManifest, opts: RunOptions, mode: str) -> tuple[Entry, str]:
    """The entry to run and the concrete mode (script, chat, once, serve)."""
    wants_agent = mode in ("chat", "once", "serve")
    root = manifest.root
    if opts.target:
        entry = parse_ref(opts.target, root)
    elif wants_agent and manifest.agent:
        entry = parse_ref(manifest.agent, root)
    elif manifest.main and ":" in manifest.main and parse_ref(manifest.main, root).attr:
        entry = parse_ref(manifest.main, root)
    else:
        main = manifest.resolve_main()
        if main is None:
            if manifest.agent:
                entry = parse_ref(manifest.agent, root)
                return entry, ("chat" if mode in ("auto", "script") else mode)
            hint = (
                "set main in promptise.toml"
                if manifest.path
                else "pass a file (promptise run main.py) or create promptise.toml (promptise init --project)"
            )
            raise EntryError(f"nothing to run: no main.py, agent.py or app.py; {hint}")
        if not main.is_file():
            raise EntryError(f"main {manifest.main!r}: no such file ({main})")
        entry = Entry(raw=str(manifest.main or main.name), path=main, module=None, attr=None)

    if entry.attr is not None:
        if mode == "script":
            raise EntryError(
                f"{entry.describe(manifest.root)} is a factory; --mode script runs a file"
            )
        return entry, ("chat" if mode == "auto" else mode)

    # A plain file.
    assert entry.path is not None
    if wants_agent:
        kind, attr = detect_kind(entry.path)
        if attr is None:
            # A script with a create_agent() of its own still works as a factory module.
            attr = "create_agent" if _defines(entry.path, "create_agent") else None
        if attr is None:
            if manifest.agent and not opts.target:
                return parse_ref(manifest.agent, root), mode
            raise EntryError(
                f"{entry.path.name} has no create_agent(); --{mode} needs an agent factory "
                '(set agent = "agent.py:create_agent" in promptise.toml)'
            )
        return replace(entry, attr=attr), mode
    if mode == "script":
        return entry, "script"
    kind, attr = detect_kind(entry.path)
    if kind == "factory":
        return replace(entry, attr=attr), "chat"
    return entry, "script"


def _defines(path: Path, name: str) -> bool:
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in tree.body
    )


def _check_entry(entry: Entry, manifest: ProjectManifest) -> list[str]:
    """``--check``: the entry compiles (script) or imports and has its factory."""
    problems: list[str] = []
    if entry.attr is None:
        assert entry.path is not None
        try:
            compile(entry.path.read_text(encoding="utf-8"), str(entry.path), "exec")
        except SyntaxError as exc:
            problems.append(f"{entry.path}: syntax error on line {exc.lineno}: {exc.msg}")
        return problems
    from .entry import _import_entry

    try:
        module = _import_entry(entry, manifest.root)
    except EntryError as exc:
        return [str(exc)]
    except Exception as exc:
        return [f"importing {entry.describe(manifest.root)} failed: {type(exc).__name__}: {exc}"]
    if not callable(getattr(module, entry.attr, None)):
        problems.append(f"{entry.describe(manifest.root)}: no callable {entry.attr!r}")
    return problems


def _env_report(manifest: ProjectManifest) -> list[dict[str, Any]]:
    from ..models import dotenv_origin

    names = list(dict.fromkeys([*manifest.env.required, *manifest.env.vars]))
    return [
        {
            "name": n,
            "set": bool(os.environ.get(n)),
            "required": n in manifest.env.required,
            "source": dotenv_origin(n) or ("environment" if os.environ.get(n) else None),
        }
        for n in names
    ]


# ---------------------------------------------------------------------------
# the command
# ---------------------------------------------------------------------------


def _tolerant_streams() -> None:
    """Never fail on a character the console cannot encode (a Windows code page).

    The dev tools draw with symbols (●, ⚙, ✓) and agents answer in any
    language; on a legacy console encoding those are replaced instead of
    raising ``UnicodeEncodeError``.  Only the error handler changes.
    """
    for stream in (sys.stdout, sys.stderr):
        encoding = (getattr(stream, "encoding", None) or "").lower().replace("-", "")
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None and encoding not in ("utf8", "utf8sig"):
            with contextlib.suppress(Exception):
                reconfigure(errors="replace")


def run_command(opts: RunOptions) -> int:
    """Run ``promptise run`` with *opts*; return the process exit code."""
    from .supervisor import CHILD_ENV

    _tolerant_streams()

    is_child = os.environ.get(CHILD_ENV) == "1"
    if opts.mode not in _MODES:
        return _problem(f"--mode must be one of {', '.join(_MODES)}")
    if opts.observe is not None and opts.observe not in ("off", "basic", "standard", "full"):
        return _problem("--observe must be off, basic, standard or full")
    if opts.tui and opts.prod:
        return _problem("--tui is a development view; it cannot be combined with --prod")
    if opts.watch and opts.prod:
        return _problem("--watch is for development; it cannot be combined with --prod")

    try:
        manifest = _load_project(opts)
    except ManifestError as exc:
        return _problem(str(exc))

    if opts.list_scripts:
        return _list_scripts(manifest)

    if opts.target and opts.target in manifest.scripts:
        return _run_named_script(manifest, opts.target, opts)
    if opts.target and not (
        opts.target.endswith(".py") or ":" in opts.target or Path(opts.target).exists()
    ):
        known = ", ".join(sorted(manifest.scripts)) or "none"
        import difflib

        close = difflib.get_close_matches(opts.target, list(manifest.scripts), n=1)
        hint = f" — did you mean {close[0]!r}?" if close else ""
        return _problem(
            f"no script or file named {opts.target!r}{hint} (scripts: {known}). "
            "To pass arguments to the main file, name it: promptise run main.py ARGS"
        )

    mode = opts.mode
    if opts.serve:
        mode = "serve"
    elif opts.input is not None and mode == "auto":
        mode = "once"
    if opts.input is not None and mode not in ("once", "auto"):
        return _problem("--input asks one question; it cannot be combined with --mode " + mode)

    problems = _apply_env(manifest, opts)
    try:
        entry, mode = _resolve_entry(manifest, opts, mode)
    except EntryError as exc:
        problems.append(str(exc))
        entry = None  # type: ignore[assignment]
    from .prod import missing_env

    missing = missing_env(manifest.env.required)
    if missing:
        problems.append(
            "missing required environment variable(s): "
            + ", ".join(missing)
            + " (set them, or add them to "
            + ", ".join(manifest.env.files or [".env"])
            + ")"
        )
    if opts.check and entry is not None:
        problems += _check_entry(entry, manifest)
    if problems:
        if opts.prod and not opts.check:
            from .prod import configure_logging

            configure_logging(manifest.prod.log_level, manifest.prod.log_format)
            logging.getLogger("promptise.run").error(
                "preflight failed", extra={"problems": problems}
            )
            return 2
        err.print("[bold red]promptise run: cannot start[/bold red]")
        for p in problems:
            err.print(f"  [red]✗[/red] {escape(p)}")
        return 2
    assert entry is not None
    if opts.check:
        err.print(
            f"[green]✓[/green] {escape(manifest.project_name)}: {escape(entry.describe(manifest.root))} "
            f"({mode}) is ready"
        )
        return 0

    if (opts.tui or opts.watch) and not is_child:
        return _supervise(manifest, opts)

    return _run_here(manifest, opts, entry, mode, is_child=is_child)


def _supervise(manifest: ProjectManifest, opts: RunOptions) -> int:
    from .supervisor import ChildRunSource, supervise_watch

    child_opts = replace(opts, tui=False, watch=False)
    if opts.tui:
        # The child records at the dev level; its trace goes to the UI, not the console.
        child_opts = replace(child_opts, devtools=False)
    args = child_opts.to_args()
    watch_paths = [manifest.root / p for p in manifest.dev.watch]
    cwd = Path.cwd()

    if not opts.tui:
        try:
            return asyncio.run(
                supervise_watch(
                    args,
                    cwd=cwd,
                    watch_paths=watch_paths,
                    ignore=manifest.dev.ignore,
                    root=manifest.root,
                )
            )
        except KeyboardInterrupt:
            return 130

    async def _tui() -> int:
        from .tui import run_tui

        source = ChildRunSource(
            args,
            cwd=cwd,
            watch=(watch_paths, manifest.dev.ignore, manifest.root) if opts.watch else None,
        )
        os.environ["PROMPTISE_RUN_TUI"] = "1"
        await source.start()
        try:
            return await run_tui(source, title=manifest.project_name)
        finally:
            await source.close()

    try:
        return asyncio.run(_tui())
    except KeyboardInterrupt:
        return 130


def _run_here(
    manifest: ProjectManifest, opts: RunOptions, entry: Entry, mode: str, *, is_child: bool
) -> int:
    """Run the program in this process, with the requested instrumentation."""
    from .. import _harness
    from .devtools import DevTracer, print_summary, write_report
    from .events import EventBus
    from .harness import RunHarness
    from .ipc import IpcClient, LogBridge
    from .pricing import PriceTable
    from .prod import JsonEventLog, SigtermHandler, configure_logging
    from .stats import RunStats

    ipc = IpcClient.from_env() if is_child else None
    tui = ipc is not None
    devtools = opts.devtools and not tui
    observe: str | None
    if devtools or tui:
        observe = opts.observe or manifest.dev.observe
    elif opts.prod:
        observe = opts.observe or manifest.prod.observe
    else:
        observe = opts.observe
    if observe is None and mode != "script" and not opts.prod:
        # Chat and once modes show usage (/stats, --json): record LLM turns.
        observe = "standard"
    if observe == "off":
        observe = None
    # --report is relative to where you typed it; [dev] report to the project root.
    report: Path | None = None
    if opts.report:
        report = Path(os.getcwd()) / opts.report
    elif (devtools or tui) and manifest.dev.report:
        report = manifest.root / manifest.dev.report
    instrument = bool(devtools or tui or observe or opts.model or report or mode != "script")

    # Like npm: run from the project root; the original directory stays known.
    os.environ.setdefault("PROMPTISE_INIT_CWD", os.getcwd())
    if manifest.path is not None and Path.cwd() != manifest.root:
        os.chdir(manifest.root)
    for extra in reversed(manifest.python_path):
        candidate = str((manifest.root / extra).resolve())
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    prices = PriceTable(manifest.pricing)
    bus = EventBus()
    stats = RunStats(prices)
    harness: RunHarness | None = None
    json_log: JsonEventLog | None = None
    log_bridge: LogBridge | None = None
    if instrument:
        harness = RunHarness(
            bus,
            stats,
            observe=observe,
            events=bool(devtools or tui or (opts.prod and observe)),
            model=opts.model,
        )
        _harness.install(harness)
        harness.start()
    if devtools:
        DevTracer(prices).attach(bus)
    if opts.prod:
        configure_logging(manifest.prod.log_level, manifest.prod.log_format)
        json_log = JsonEventLog(timeline=bool(observe))
        bus.subscribe(json_log)
    if ipc is not None:
        from .approvals import ApprovalBridge

        bridge = ApprovalBridge(bus)
        if harness is not None:
            harness.approvals = bridge
        router = _CommandRouter(bridge)
        ipc.on_command = router
        ipc.on_close = router.closed
        bus.subscribe(ipc)
        log_bridge = LogBridge(bus)
        root_logger = logging.getLogger()
        root_logger.addHandler(log_bridge)
        if root_logger.level > logging.INFO or root_logger.level == logging.NOTSET:
            root_logger.setLevel(logging.INFO)
        for noisy in ("httpx", "httpcore", "watchdog", "asyncio"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    ctx = RunContext(
        project=manifest.project_name,
        root=manifest.root,
        mode=mode,  # type: ignore[arg-type]
        devtools=devtools or tui,
        prod=opts.prod,
        tui=tui,
        model=opts.model,
        manifest=manifest,
        args=tuple(opts.args),
    )
    started = time.time()
    bus.publish(
        "run.start",
        project=manifest.project_name,
        project_version=manifest.version,
        main=entry.describe(manifest.root),
        mode=mode,
        entry_kind=entry.kind,
        pid=os.getpid(),
        python=platform.python_version(),
        promptise_version=_safe_version(),
        cwd=os.getcwd(),
        root=str(manifest.root),
        argv=[entry.describe(manifest.root), *opts.args],
        flags={"devtools": devtools or tui, "prod": opts.prod, "tui": tui, "watch": opts.watch},
        model_override=opts.model,
        config=manifest.to_public_dict(),
        env=_env_report(manifest),
        scripts=dict(manifest.scripts),
    )
    bus.publish("state", state="running", detail="")
    if devtools:
        err.print(
            f"[bold cyan]promptise run[/bold cyan] {escape(manifest.project_name)} · "
            f"{escape(entry.describe(manifest.root))} · {mode} · devtools on (observe {observe})"
        )

    sigterm = SigtermHandler()
    if opts.prod or is_child:
        sigterm.install()
    code, error, tb = 0, None, None
    try:
        if mode == "script":
            from .script import run_script

            assert entry.path is not None
            code = run_script(entry.path, opts.args)
        else:
            code = asyncio.run(_agent_main(entry, ctx, mode, opts, manifest, bus, harness, ipc))
    except KeyboardInterrupt:
        code = 143 if sigterm.received else 130
    except EntryError as exc:
        code, error = 2, str(exc)
        err.print(f"[bold red]promptise run:[/bold red] {escape(str(exc))}")
    except SystemExit as exc:
        from .script import exit_code

        code = exit_code(exc)
    except BaseException as exc:
        code, error = 1, f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()
        if opts.prod:
            logging.getLogger("promptise.run").error("program failed", exc_info=True)
        elif devtools:
            from rich.traceback import Traceback

            err.print(
                Traceback.from_exception(type(exc), exc, exc.__traceback__, show_locals=False)
            )
        else:
            traceback.print_exc()
    finally:
        sigterm.uninstall()
        duration = time.time() - started
        with contextlib.suppress(Exception):
            sys.stdout.flush()
        if harness is not None:
            harness.stop()
            _harness.uninstall(harness)
        bus.publish("state", state="exited", detail=f"exit code {code}")
        bus.publish("run.exit", code=code, error=error, traceback=tb, duration_s=round(duration, 3))
        if json_log is not None and harness is not None:
            json_log.final_stats(stats.snapshot())
        if devtools:
            print_summary(stats.snapshot(), duration_s=duration, console=err, code=code)
        if report is not None:
            path = report
            try:
                written = write_report(path, bus, extra={"project": manifest.project_name})
                if devtools:
                    err.print(f"[dim]report: {escape(str(written))}[/dim]")
            except OSError as exc:
                err.print(f"[yellow]could not write the report {escape(str(path))}: {exc}[/yellow]")
        if log_bridge is not None:
            logging.getLogger().removeHandler(log_bridge)
        if ipc is not None:
            ipc.close()
    return code


def _safe_version() -> str:
    try:
        return _version("promptise")
    except Exception:
        return "unknown"


async def _agent_main(
    entry: Entry,
    ctx: RunContext,
    mode: str,
    opts: RunOptions,
    manifest: ProjectManifest,
    bus: Any,
    harness: Any,
    ipc: Any,
) -> int:
    from .chat import Conversation, ask_once, repl

    agent = await load_factory(entry, ctx)
    try:
        if mode == "once":
            question = opts.input
            if question == "-" or question is None:
                question = sys.stdin.read()
            return await ask_once(
                agent, question, bus=bus, harness=harness, as_json=opts.json_output
            )
        if mode == "serve":
            from .serve import run_serve

            info = {
                "project": manifest.project_name,
                "version": manifest.version,
                "model": getattr(agent, "model_name", None),
            }

            def started(host: str, port: int) -> None:
                shown = f"[{host}]" if ":" in host else host
                url = f"http://{shown}:{port}"
                if opts.prod:
                    logging.getLogger("promptise.run").info(
                        "serving", extra={"url": url, "host": host, "port": port}
                    )
                else:
                    err.print(
                        f"[bold green]serving[/bold green] {escape(manifest.project_name)} on "
                        f"{url}  [dim]POST /v1/invoke · POST /v1/stream · GET /v1/info · "
                        "Ctrl+C to stop[/dim]"
                    )
                    token_env = manifest.serve.auth_token_env
                    if not (token_env and os.environ.get(token_env)):
                        err.print(
                            "[yellow]no auth token: any process on this machine can call the "
                            f"agent. Set {escape(token_env or '[serve] auth_token_env')} to "
                            "require one.[/yellow]"
                        )
                bus.publish("state", state="idle", detail=f"serving on {url}")

            return await run_serve(
                agent,
                settings=manifest.serve,
                bus=bus,
                info=info,
                shutdown_timeout=manifest.prod.shutdown_timeout,
                host=opts.host,
                port=opts.port,
                on_started=started,
            )
        if ipc is not None:
            return await _remote_chat(Conversation(agent, bus, harness), ipc.on_command)
        return await repl(
            agent,
            bus=bus,
            harness=harness,
            title=manifest.project_name,
            devtools=opts.devtools,
        )
    finally:
        await agent.shutdown()


class _CommandRouter:
    """Routes the terminal UI's commands in the child: approvals here, chat to the chat loop."""

    def __init__(self, approvals: Any) -> None:
        self.approvals = approvals
        self.chat: Any = None
        self.close: Any = None

    def __call__(self, command: dict[str, Any]) -> None:
        if self.approvals is not None and self.approvals.handle_command(command):
            return
        if self.chat is not None:
            self.chat(command)

    def closed(self) -> None:
        if self.close is not None:
            self.close()


async def _remote_chat(convo: Any, router: _CommandRouter) -> int:
    """The chat for the terminal UI: turns are asked over the IPC channel."""
    loop = asyncio.get_running_loop()
    commands: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    router.chat = lambda cmd: loop.call_soon_threadsafe(commands.put_nowait, cmd)
    router.close = lambda: loop.call_soon_threadsafe(commands.put_nowait, None)
    running: dict[str, asyncio.Task[Any]] = {}
    convo.bus.publish("state", state="idle", detail="ready")

    async def do_turn(text: str, turn_id: str) -> None:
        convo.bus.publish("state", state="busy", detail=turn_id)
        try:
            await convo.turn(text, turn_id=turn_id)
        finally:
            running.pop(turn_id, None)
            if not running:
                convo.bus.publish("state", state="idle", detail="")

    while True:
        cmd = await commands.get()
        if cmd is None:
            break
        name = cmd.get("cmd")
        if name == "chat":
            turn_id = str(cmd.get("turn_id") or os.urandom(8).hex())
            running[turn_id] = asyncio.ensure_future(do_turn(str(cmd.get("text", "")), turn_id))
        elif name == "cancel":
            task = running.get(str(cmd.get("turn_id")))
            if task is not None:
                task.cancel()
        elif name == "reset":
            for task in list(running.values()):
                task.cancel()
            convo.reset()
            convo.bus.publish("state", state="idle", detail="new conversation")
    return 0
