"""CLI ``serve`` command for running MCP servers from the command line.

Allows running any Python module that exposes an ``MCPServer`` instance
without writing boilerplate.

Usage::

    # Run a server from a module path
    promptise serve myapp.server:server --transport http --port 8080

    # With dashboard
    promptise serve myapp.server:server --dashboard

    # With hot reload (development)
    promptise serve myapp.server:server --reload
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import Any


def build_serve_parser(subparsers: Any = None) -> argparse.ArgumentParser:
    """Build the ``serve`` subcommand parser.

    Args:
        subparsers: Parent subparser group (from argparse).
            If ``None``, creates a standalone parser.
    """
    if subparsers is not None:
        parser = subparsers.add_parser(
            "serve",
            help="Run an MCP server from a Python module",
        )
    else:
        parser = argparse.ArgumentParser(
            prog="promptise serve",
            description="Run an MCP server from a Python module",
        )

    parser.add_argument(
        "target",
        help="Server target in module:attribute format (e.g. myapp.server:server)",
    )
    parser.add_argument(
        "--transport",
        "-t",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help="Transport type (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host for HTTP/SSE (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        default=8080,
        help="Bind port for HTTP/SSE (default: 8080)",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Enable live terminal dashboard",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot reload (development only)",
    )
    parser.add_argument(
        "--allowed-host",
        dest="allowed_hosts",
        action="append",
        metavar="HOST",
        help=(
            "Host header value to accept on HTTP/SSE, e.g. api.example.com or "
            "api.example.com:* (repeatable). A loopback bind validates Host and Origin "
            "against the loopback names by default; a non-loopback bind validates only "
            "when this is given."
        ),
    )
    parser.add_argument(
        "--allowed-origin",
        dest="allowed_origins",
        action="append",
        metavar="ORIGIN",
        help=(
            "Origin header value to accept for browser clients, e.g. "
            "https://app.example.com (repeatable). Requires --allowed-host on a "
            "non-loopback bind."
        ),
    )
    return parser


def resolve_server(target: str) -> Any:
    """Import and return the server instance from a target string.

    Args:
        target: ``module_path:attribute`` (e.g. ``"myapp.server:server"``).

    Returns:
        The resolved ``MCPServer`` instance.
    """
    if ":" not in target:
        raise ValueError(
            f"Invalid target format: {target!r}. "
            f"Expected 'module.path:attribute' (e.g. 'myapp.server:server')"
        )

    module_path, attr_name = target.rsplit(":", 1)

    # Add CWD to sys.path for local module imports
    import os

    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Cannot import module {module_path!r}: {e}") from e

    try:
        server = getattr(module, attr_name)
    except AttributeError:
        raise AttributeError(f"Module {module_path!r} has no attribute {attr_name!r}")

    return server


def run_serve(args: argparse.Namespace, server: Any | None = None) -> None:
    """Execute the ``serve`` command.

    Args:
        args: Parsed command-line arguments.
        server: Optional pre-resolved ``MCPServer`` instance. When ``None``,
            the server is resolved from ``args.target`` (lets callers
            validate the resolved object before dispatch).
    """
    if server is None:
        server = resolve_server(args.target)

    options: dict[str, Any] = {
        "transport": args.transport,
        "host": args.host,
        "port": args.port,
        "dashboard": args.dashboard,
    }
    # Host/Origin allow-lists are optional on the namespace (front-ends that
    # do not expose them keep the bind-derived default policy).
    for name in ("allowed_hosts", "allowed_origins"):
        values = getattr(args, name, None)
        if values:
            options[name] = list(values)

    if args.reload:
        from ._hot_reload import hot_reload

        hot_reload(server, **options)
    else:
        server.run(**options)
