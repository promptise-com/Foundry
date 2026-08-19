"""Examples gallery contract tests."""

from __future__ import annotations

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
GALLERY_PATH = REPOSITORY_ROOT / "docs" / "resources" / "examples.md"
PYTHON_EXAMPLE_PATTERN = re.compile(r"`(examples/[A-Za-z0-9_./-]+\.py)`")


def test_literal_python_example_paths_exist() -> None:
    """Keep runnable gallery paths aligned with the repository."""
    gallery = GALLERY_PATH.read_text(encoding="utf-8")
    paths = sorted(set(PYTHON_EXAMPLE_PATTERN.findall(gallery)))

    assert paths, "The examples gallery must reference Python examples."
    assert [path for path in paths if not (REPOSITORY_ROOT / path).is_file()] == []
