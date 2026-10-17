"""The context ``promptise run`` passes to an agent factory."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .manifest import ProjectManifest

__all__ = ["RunContext"]

RunMode = Literal["script", "chat", "once", "serve"]


@dataclass(frozen=True)
class RunContext:
    """What an agent factory may want to know about the run.

    A factory takes either no argument or this one::

        async def create_agent(ctx: RunContext) -> PromptiseAgent:
            return await build_agent(model=ctx.model or "openai:gpt-5-mini", ...)

    Attributes:
        project: The project's name.
        root: The project root (the manifest's directory).
        mode: ``script``, ``chat``, ``once`` or ``serve``.
        devtools: ``--devtools`` (or ``--tui``) is on.
        prod: ``--prod`` is on.
        tui: The run is shown in the terminal UI.
        model: The ``--model`` override, or ``None``.  ``promptise run`` also
            applies it to every agent the program builds, so a factory only
            needs it to pick a model-specific setting.
        manifest: The project manifest.
        args: The arguments after the target (``sys.argv[1:]`` of a script).
    """

    project: str
    root: Path
    mode: RunMode
    devtools: bool = False
    prod: bool = False
    tui: bool = False
    model: str | None = None
    manifest: ProjectManifest | None = None
    args: tuple[str, ...] = field(default_factory=tuple)
