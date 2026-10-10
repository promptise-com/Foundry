"""Generate the provider reference table from ``promptise.models.PROVIDERS``.

The table is included in the docs with ``--8<--`` so every page shows the same,
exact variable names.  ``tests/test_models.py`` asserts the committed table
matches this output; regenerate with::

    .venv/bin/python docs/.snippets/gen_providers_table.py
"""

from __future__ import annotations

from pathlib import Path

from promptise.models import PROVIDERS, Provider

OUT = Path(__file__).with_name("providers-table.md")

_ORDER = [
    "openai",
    "azure_openai",
    "azure_ai",
    "anthropic",
    "google_genai",
    "google_vertexai",
    "bedrock",
    "ollama",
]


def _env_cell(p: Provider) -> str:
    if not p.env:
        return "_none_"
    parts = []
    for v in p.env:
        names = " / ".join(f"`{n}`" for n in (v.name, *v.aliases))
        parts.append(names if v.required else f"{names} (optional)")
    return "<br>".join(parts)


def _words_cell(p: Provider) -> str:
    return ", ".join(f"`{w}=`" for w in p.words()) or "_none_"


def _route_cell(p: Provider) -> str:
    if p.base_url is None:
        return "native (core)"
    if "{" in p.base_url:
        return "OpenAI-compatible, URL built from the words"
    return f"OpenAI-compatible: `{p.base_url}`"


def render() -> str:
    by = {p.key: p for p in PROVIDERS}
    keys = _ORDER + sorted(k for k in by if k not in _ORDER)
    lines = [
        "| Provider | `provider=` | Environment variables (exact names) | `Model(...)` words | Route |",
        "|---|---|---|---|---|",
    ]
    for key in keys:
        p = by[key]
        aliases = [a for a in p.prefixes if a != p.display]
        provider_cell = f"`{p.display}`" + (
            f" <br><small>also {', '.join(f'`{a}`' for a in aliases)}</small>" if aliases else ""
        )
        title = p.title.split(" (")[0]
        lines.append(
            f"| {title} | {provider_cell} | {_env_cell(p)} | {_words_cell(p)} | {_route_cell(p)} |"
        )
    lines.append("")
    lines.append(
        "Nothing to install for any row: `pip install promptise` covers every provider. "
        "`native=True` opts into a provider's own LangChain integration when you have it installed."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUT.write_text(render(), encoding="utf-8")
    print(f"wrote {OUT} ({len(PROVIDERS)} providers)")
