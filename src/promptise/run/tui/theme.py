"""Colours of the ``promptise run`` terminal UI (the Promptise brand palette)."""

from __future__ import annotations

from textual.theme import Theme

__all__ = [
    "ACCENT",
    "DIM",
    "ERROR",
    "MUTED",
    "PRIMARY",
    "SECONDARY",
    "SUCCESS",
    "TEXT",
    "THEME",
    "WARNING",
    "level_style",
    "severity_style",
    "status_style",
]

PRIMARY = "#818cf8"
SECONDARY = "#6366f1"
ACCENT = "#a5b4fc"
SUCCESS = "#34d399"
WARNING = "#fbbf24"
ERROR = "#f87171"
TEXT = "#e5e7eb"
DIM = "#9ca3af"
MUTED = "#6b7280"

THEME = Theme(
    name="promptise",
    primary=PRIMARY,
    secondary=SECONDARY,
    accent=ACCENT,
    success=SUCCESS,
    warning=WARNING,
    error=ERROR,
    background="#0f1117",
    surface="#161a26",
    panel="#1e2333",
    dark=True,
)
"""The same palette as ``promptise mcpcast``'s guided setup."""


def level_style(level: str) -> str:
    """The Rich style of a log level (``stderr``/``stdout`` lines included)."""
    level = level.upper()
    if level in ("ERROR", "CRITICAL", "FATAL"):
        return f"bold {ERROR}"
    if level in ("WARNING", "WARN", "STDERR"):
        return WARNING
    if level == "DEBUG":
        return MUTED
    return DIM


def severity_style(severity: str) -> str:
    """The Rich style of an agent event severity."""
    severity = severity.lower()
    if severity in ("error", "critical"):
        return f"bold {ERROR}"
    if severity == "warning":
        return WARNING
    if severity == "debug":
        return MUTED
    return DIM


def status_style(status: str) -> str:
    """The Rich style of a tool call status."""
    return {
        "ok": SUCCESS,
        "error": f"bold {ERROR}",
        "aborted": WARNING,
        "running": ACCENT,
    }.get(status, DIM)
