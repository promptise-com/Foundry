"""Regenerate the guided-setup screenshots in docs/assets/mcpcast/.

Drives ``promptise mcpcast`` (the Textual wizard) headlessly over the
bookshelf example API from ``examples/mcp/mcpcast_fastapi_app/app.py`` and
saves one SVG per step.  Nothing is fetched and no model is called: local API
detection is answered from the app's own ``openapi()`` document, and curation
replays the proposal a real ``openai:gpt-5-mini`` run produced for this API
(kept verbatim — including the descriptions that mention ``delete_book``, a
tool the standard profile excludes, which is what the review step is for).

Run from the repository root::

    .venv/bin/python docs/.snippets/gen_wizard_screens.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples" / "mcp" / "mcpcast_fastapi_app"))

from app import app as bookshelf  # noqa: E402  (the example FastAPI app)

from promptise.mcpcast.wizard import MCPcastWizard  # noqa: E402

OUT = ROOT / "docs" / "assets" / "mcpcast"
SPEC_URL = "http://127.0.0.1:8765/openapi.json"
SIZE = (110, 36)
WORKDIR = Path("/tmp/bookshelf-api")  # a neutral working directory for the screenshots

# The proposal openai:gpt-5-mini returned for this API under --profile standard
# (see docs/mcpcast/guided-setup.md): three descriptions refer to delete_book,
# which the profile excluded, and `tags` is hidden on create_book.
PROPOSAL = {
    "tools": [
        {
            "name": "find_books",
            "description": (
                "Search or browse the catalogue. If a full-text query is supplied this will run the "
                "POST /books/search route; if no query is provided it falls back to listing books "
                "(GET /books) and can filter by exact author. Use this to discover books; do not use "
                "it to modify records. For fetching a single book use get_book; to change data use "
                "create_book, update_book or delete_book."
            ),
            "risk": "read",
            "operations": ["search_books", "list_books"],
            "params": {
                "query": {
                    "description": (
                        "Full-text search matched against title and author; when present the "
                        "search route is used. If omitted the API returns the book list."
                    )
                },
                "author": {"description": "Exact author name to filter results."},
                "limit": {
                    "hidden": True,
                    "default": 20,
                    "description": "Maximum number of results. Hidden: the default suits most callers.",
                },
            },
            "example": {"query": "hobbit", "author": "J.R.R. Tolkien"},
        },
        {
            "name": "get_book",
            "description": (
                "Retrieve the full record for a single book by id, including librarian notes. Use "
                "this when you need the canonical details for one book. For searching use "
                "find_books; to change the record use update_book or delete_book."
            ),
            "risk": "read",
            "operations": ["get_book"],
            "params": {"book_id": {"description": "The numeric id of the book to fetch."}},
            "example": {"book_id": 42},
        },
        {
            "name": "create_book",
            "description": (
                "Add a new book to the catalogue; returns the created record with its assigned id. "
                "Use when ingesting new items; do not use this to modify existing books. For edits "
                "to existing records use update_book."
            ),
            "risk": "write",
            "operations": ["create_book"],
            "params": {
                "title": {"description": "Full book title (as printed on the cover)."},
                "author": {"description": "Author name exactly as printed on the cover."},
                "year": {"description": "Year of first publication (integer)."},
                "tags": {
                    "hidden": True,
                    "default": [],
                    "description": "Optional list of tags. Hidden by default for simplicity.",
                },
            },
            "example": {"title": "Neuromancer", "author": "William Gibson", "year": 1984},
        },
        {
            "name": "update_book",
            "description": (
                "Modify one or more fields on an existing book; only the fields you send will "
                "change. Useful for correcting year, updating librarian notes, or replacing the tag "
                "list. For creating or removing books use create_book or delete_book respectively; "
                "to view the record first use get_book."
            ),
            "risk": "write",
            "operations": ["update_book"],
            "params": {
                "book_id": {"description": "The numeric id of the book to update."},
                "notes": {"description": "Replace or set the librarian's notes for the book."},
                "tags": {"description": "Replace the book's tag list (provide an array)."},
                "year": {"description": "Corrected publication year (integer)."},
            },
            "example": {"book_id": 42, "notes": "First edition, slightly worn."},
        },
    ],
    "dropped": [
        {
            "operation_id": "health_check",
            "reason": "Liveness probe for infrastructure; not useful to end-user agents.",
        },
        {
            "operation_id": "find_books_legacy",
            "reason": "Deprecated legacy endpoint; callers should use find_books.",
        },
        {
            "operation_id": "reset_catalogue",
            "reason": "Admin-only destructive operator action; not appropriate for general agents.",
        },
    ],
}


async def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    body = json.dumps(bookshelf.openapi())

    def fetch(url: str) -> str | None:
        return body if url == SPEC_URL else None

    async def complete(system: str, user: str) -> str:
        return json.dumps(PROPOSAL)

    shutil.rmtree(WORKDIR, ignore_errors=True)
    WORKDIR.mkdir(parents=True)
    os.environ["PROMPTISE_NO_DOTENV"] = "1"
    os.environ["OPENAI_API_KEY"] = "sk-docs-placeholder"  # never called: curation is scripted
    app = MCPcastWizard(cwd=WORKDIR, fetch=fetch, completer=complete)

    def shot(name: str) -> None:
        path = OUT / f"wizard-{name}.svg"
        svg = app.export_screenshot(title="promptise mcpcast")
        # Rich emits only a viewBox; an <img> then has no intrinsic size and renders
        # at the browser default. Give it real dimensions so the docs can scale it.
        match = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
        if match:
            width, height = match.groups()
            svg = svg.replace(
                '<svg class="rich-terminal" ',
                f'<svg class="rich-terminal" width="{width}" height="{height}" ',
                1,
            )
        path.write_text(svg)
        print("wrote", path.relative_to(ROOT))

    async with app.run_test(size=SIZE) as pilot:
        await pilot.pause()
        shot("welcome")
        await pilot.press("enter")  # Start → detection finds the bookshelf API
        await pilot.pause(0.5)
        await pilot.pause()
        shot("detect")
        await pilot.press("enter")  # pick it → loads
        await pilot.pause(0.5)
        await pilot.pause()
        shot("spec")
        await pilot.press("enter")  # Continue → model
        await pilot.pause()
        shot("model")
        await pilot.press("enter")  # design with a model → safety
        await pilot.pause()
        shot("safety")
        await pilot.press("down", "enter")  # standard → auth
        await pilot.pause()
        shot("auth")
        await pilot.press("enter")  # personal → project
        await pilot.pause()
        shot("project")
        await pilot.press("enter")  # → review (scripted curation)
        await pilot.pause(0.8)
        await pilot.pause()
        shot("review")
        await pilot.press("enter")  # write
        await pilot.pause(0.5)
        await pilot.pause()
        shot("write")
        await pilot.press("f1")
        await pilot.pause()
        shot("help")
        await pilot.press("escape")
        await pilot.press("ctrl+q")
    shutil.rmtree(WORKDIR, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
