"""``promptise run``: run a project's program with dev tools, a terminal UI, or for production.

A project is described by ``promptise.toml`` (or ``[tool.promptise]`` in
``pyproject.toml``), like ``package.json`` for Node: ``main`` names the program,
``[scripts]`` holds named commands, ``[env]`` the variables it needs.
``promptise run`` runs ``main`` as a normal Python program and instruments every
agent it builds, without changes to your code::

    promptise run                 # like `python main.py`, with the project's env
    promptise run --devtools      # live traces, tool calls, tokens and cost
    promptise run --tui           # the full terminal UI
    promptise run --prod          # JSON logs, preflight checks, graceful SIGTERM
    promptise run dev             # a script from [scripts]
"""

from __future__ import annotations

from .context import RunContext
from .manifest import ManifestError, ProjectManifest, find_manifest, load_manifest

__all__ = ["ManifestError", "ProjectManifest", "RunContext", "find_manifest", "load_manifest"]
