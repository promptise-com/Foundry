"""Bookshelf — a small, realistic FastAPI app about to become an MCP server.

This file stands in for *your* codebase: an ordinary HTTP API with a bearer
token, a few resources and an admin corner. Nothing in it knows about MCP or
Promptise. What ``promptise mcpcast`` works from is the OpenAPI document FastAPI
already serves at ``/openapi.json`` — and three things in this file decide how
good the resulting tools are:

- ``summary=`` and the docstring become the **tool description** an LLM reads
  when it decides which tool to call. Write them for a model, not for a human
  with the reference docs open.
- ``operation_id=`` becomes the **tool name**. Without it FastAPI derives
  ``list_books_books_get`` from the function, path and method — and that is
  the name the model would have to pick from.
- ``Field(description=...)`` on the Pydantic models and ``Query(description=...)``
  on query parameters become each **parameter's description**.

Run it on its own from the repo root with
``uvicorn --app-dir examples/mcp/mcpcast_fastapi_app app:app --port 8000``
(``uvicorn app:app --port 8000`` from this directory) and open
``http://127.0.0.1:8000/openapi.json`` — that URL is all ``mcpcast`` needs.
"""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

# The one token this demo accepts. A generated `--auth env-token` server presents
# it upstream as `Authorization: Bearer demo-token` on every call.
DEMO_TOKEN = "demo-token"

# ---------------------------------------------------------------------------
# Models — descriptions here reach the agent as parameter descriptions
# ---------------------------------------------------------------------------


class Book(BaseModel):
    """A book on the shelf."""

    id: int = Field(description="Stable numeric identifier of the book.")
    title: str = Field(description="Full title.")
    author: str = Field(description="Author as printed on the cover, e.g. 'Ursula K. Le Guin'.")
    year: int = Field(description="Year of first publication.")
    tags: list[str] = Field(default_factory=list, description="Free-form tags such as 'fantasy'.")
    notes: str = Field(default="", description="Librarian's notes about this copy.")


class BookCreate(BaseModel):
    """Fields needed to add a book."""

    title: str = Field(description="Full title.", examples=["The Lathe of Heaven"])
    author: str = Field(
        description="Author as printed on the cover.", examples=["Ursula K. Le Guin"]
    )
    year: int = Field(description="Year of first publication.", ge=0, examples=[1971])
    tags: list[str] = Field(
        default_factory=list, description="Optional tags.", examples=[["science-fiction"]]
    )


class BookUpdate(BaseModel):
    """Fields that may be changed on an existing book; omitted fields are left alone."""

    notes: str | None = Field(
        default=None,
        description="Replace the librarian's notes.",
        examples=["signed first edition"],
    )
    tags: list[str] | None = Field(default=None, description="Replace the tag list.")
    year: int | None = Field(default=None, description="Correct the publication year.", ge=0)


class SearchQuery(BaseModel):
    """A search request (POST, because it carries a body)."""

    query: str = Field(
        description="Case-insensitive text matched against title and author.",
        examples=["earthsea"],
    )
    author: str | None = Field(default=None, description="Only books by this exact author.")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results.")


# ---------------------------------------------------------------------------
# Auth — a bearer token, checked by a dependency
# ---------------------------------------------------------------------------

_bearer = HTTPBearer(description="Bearer token issued by the Bookshelf admin.")


def require_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(_bearer)],
) -> None:
    """Reject any request whose bearer token is not the demo token."""
    # Compare bytes, not str: `compare_digest` raises on non-ASCII text, and a
    # client can send any bytes in the header — that must be a 401, not a 500.
    presented = credentials.credentials.encode("utf-8")
    if not secrets.compare_digest(presented, DEMO_TOKEN.encode("utf-8")):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="invalid token")


# ---------------------------------------------------------------------------
# State — an in-memory store seeded with a few books
# ---------------------------------------------------------------------------

SEED: list[Book] = [
    Book(
        id=1, title="A Wizard of Earthsea", author="Ursula K. Le Guin", year=1968, tags=["fantasy"]
    ),
    Book(
        id=2,
        title="The Left Hand of Darkness",
        author="Ursula K. Le Guin",
        year=1969,
        tags=["science-fiction"],
    ),
    Book(
        id=3,
        title="The Dispossessed",
        author="Ursula K. Le Guin",
        year=1974,
        tags=["science-fiction"],
    ),
    Book(id=4, title="Dune", author="Frank Herbert", year=1965, tags=["science-fiction"]),
    Book(id=5, title="Neuromancer", author="William Gibson", year=1984, tags=["cyberpunk"]),
    Book(
        id=6, title="Parable of the Sower", author="Octavia E. Butler", year=1993, tags=["dystopia"]
    ),
]
BOOKS: dict[int, Book] = {}


def reset_store() -> None:
    """Restore the seed data (used by the admin endpoint and at start-up)."""
    BOOKS.clear()
    BOOKS.update({book.id: book.model_copy(deep=True) for book in SEED})


reset_store()

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Bookshelf API",
    version="1.0.0",
    description="A small library catalogue: browse, search and maintain books.",
)
books = APIRouter(prefix="/books", tags=["books"], dependencies=[Depends(require_token)])
admin = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_token)])


@app.get("/health", operation_id="health_check", summary="Liveness probe", tags=["ops"])
async def health_check() -> dict[str, str]:
    """Returns ok when the service is up. Used by the load balancer, not by people."""
    return {"status": "ok"}


@books.get("", operation_id="list_books", summary="List books on the shelf")
async def list_books(
    author: Annotated[str | None, Query(description="Only books by this exact author.")] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Maximum number of books.")] = 20,
) -> list[Book]:
    """Every book, oldest first, optionally filtered to one author."""
    found = sorted(BOOKS.values(), key=lambda b: b.year)
    if author is not None:
        found = [b for b in found if b.author == author]
    return found[:limit]


@books.get(
    "/find",
    operation_id="find_books_legacy",
    summary="Legacy title lookup",
    deprecated=True,
)
async def find_books_legacy(title: str) -> list[Book]:
    """Deprecated: use POST /books/search instead."""
    return [b for b in BOOKS.values() if title.lower() in b.title.lower()]


@books.post("/search", operation_id="search_books", summary="Search books by title or author")
async def search_books(request: SearchQuery) -> list[Book]:
    """Full-text search over titles and authors. Read-only despite being a POST."""
    needle = request.query.lower()
    found = [b for b in BOOKS.values() if needle in b.title.lower() or needle in b.author.lower()]
    if request.author is not None:
        found = [b for b in found if b.author == request.author]
    return found[: request.limit]


@books.get("/{book_id}", operation_id="get_book", summary="Get one book by id")
async def get_book(book_id: int) -> Book:
    """The full record for one book, including the librarian's notes."""
    if book_id not in BOOKS:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such book")
    return BOOKS[book_id]


@books.post(
    "",
    operation_id="create_book",
    summary="Add a book to the shelf",
    status_code=status.HTTP_201_CREATED,
)
async def create_book(book: BookCreate) -> Book:
    """Adds a new book and returns it with its assigned id."""
    new_id = max(BOOKS, default=0) + 1
    BOOKS[new_id] = Book(id=new_id, **book.model_dump())
    return BOOKS[new_id]


@books.patch(
    "/{book_id}", operation_id="update_book", summary="Update a book's notes, tags or year"
)
async def update_book(book_id: int, changes: BookUpdate) -> Book:
    """Changes only the fields you send; everything else is kept."""
    if book_id not in BOOKS:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such book")
    updated = BOOKS[book_id].model_copy(update=changes.model_dump(exclude_none=True))
    BOOKS[book_id] = updated
    return updated


@books.delete(
    "/{book_id}",
    operation_id="delete_book",
    summary="Remove a book from the shelf",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_book(book_id: int) -> None:
    """Permanently removes the book."""
    if BOOKS.pop(book_id, None) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="no such book")


@admin.post("/reset", operation_id="reset_catalogue", summary="Reset the catalogue to seed data")
async def reset_catalogue() -> dict[str, int]:
    """Drops every change and restores the seed books. Operators only."""
    reset_store()
    return {"books": len(BOOKS)}


app.include_router(books)
app.include_router(admin)
