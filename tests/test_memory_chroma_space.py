"""ChromaProvider distance function and scores, against a real ChromaDB.

Chroma creates collections with squared-L2 distance unless told otherwise.
ChromaProvider used to create collections that way but score results as
``max(0, 1 - distance)``: typical L2 distances are 1.2–1.5, so every score
was 0.00 and any ``min_score`` filter dropped every memory.  New collections
now use cosine distance; existing collections keep their distance function
and are scored for it.

Uses a deterministic bag-of-words embedding so no model is downloaded.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Any

import pytest

chromadb = pytest.importorskip("chromadb")

from chromadb.api.types import EmbeddingFunction  # noqa: E402

from promptise.memory import (  # noqa: E402
    ChromaProvider,
    MemoryScope,
    _chroma_score,
    _chroma_space,
)


class _BagOfWords(EmbeddingFunction):  # type: ignore[type-arg]
    """Unit-length hashed bag-of-words vectors (64 dims)."""

    def __init__(self) -> None:
        pass

    def __call__(self, input: Any) -> Any:
        vectors = []
        for text in input:
            vec = [0.0] * 64
            for word in text.lower().replace(".", " ").replace(",", " ").split():
                vec[int(hashlib.sha256(word.encode()).hexdigest(), 16) % 64] += 1.0
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors

    @staticmethod
    def name() -> str:
        return "promptise-test-bow"

    def get_config(self) -> dict[str, Any]:
        return {}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> _BagOfWords:  # noqa: ARG004
        return _BagOfWords()


FACT = "The user is vegetarian and never flies before nine"
#: Related to FACT, cosine similarity 0.447 under _BagOfWords.  Its squared
#: L2 distance is 1.106, which the old ``1 - distance`` scoring turned into 0.
RELATED = "vegetarian meals for the team"


def _cosine(a: str, b: str) -> float:
    va, vb = _BagOfWords()([a, b])
    return sum(x * y for x, y in zip(va, vb, strict=True))


class TestNewCollections:
    @pytest.mark.asyncio
    async def test_new_collection_uses_cosine(self, tmp_path) -> None:
        p = ChromaProvider(persist_directory=str(tmp_path), embedding_function=_BagOfWords())
        assert p._space == "cosine"
        assert _chroma_space(p._collection) == "cosine"

    @pytest.mark.asyncio
    async def test_scores_are_cosine_similarity(self, tmp_path) -> None:
        p = ChromaProvider(persist_directory=str(tmp_path), embedding_function=_BagOfWords())
        await p.add(FACT)
        await p.add("Quarterly revenue grew in the Lisbon office")
        exact = await p.search(FACT, limit=1)
        assert exact[0].content == FACT
        assert exact[0].score == pytest.approx(1.0, abs=1e-3)

        related = await p.search(RELATED, limit=2)
        assert related[0].content == FACT
        assert related[0].score == pytest.approx(_cosine(RELATED, FACT), abs=1e-3)
        assert related[1].score < related[0].score

    @pytest.mark.asyncio
    async def test_min_score_keeps_relevant_memories(self, tmp_path) -> None:
        """With l2 scoring every result was 0.00, so any min_score dropped everything."""
        p = ChromaProvider(
            persist_directory=str(tmp_path),
            embedding_function=_BagOfWords(),
            scope=MemoryScope.PER_USER,
        )
        await p.add(FACT, user_id="mara")
        results = [r for r in await p.search(RELATED, user_id="mara") if r.score >= 0.3]
        assert [r.content for r in results] == [FACT]

    @pytest.mark.asyncio
    async def test_reopening_keeps_the_collection(self, tmp_path) -> None:
        first = ChromaProvider(persist_directory=str(tmp_path), embedding_function=_BagOfWords())
        await first.add(FACT)
        second = ChromaProvider(persist_directory=str(tmp_path), embedding_function=_BagOfWords())
        assert second._space == "cosine"
        assert [r.content for r in await second.search(FACT)] == [FACT]


class TestExistingCollections:
    @pytest.mark.asyncio
    async def test_existing_l2_collection_is_scored_for_l2(self, tmp_path, caplog) -> None:
        """A collection created by Promptise 1.2.1 (Chroma's default l2) keeps working."""
        client = chromadb.PersistentClient(path=str(tmp_path))
        legacy = client.create_collection("agent_memory", embedding_function=_BagOfWords())
        assert _chroma_space(legacy) == "l2"
        legacy.add(ids=["m1"], documents=[FACT])
        del client, legacy

        with caplog.at_level(logging.WARNING, logger="promptise.memory"):
            p = ChromaProvider(persist_directory=str(tmp_path), embedding_function=_BagOfWords())
        assert p._space == "l2"
        assert "uses l2 distance" in caplog.text

        exact = await p.search(FACT, limit=1)
        assert exact[0].score == pytest.approx(1.0, abs=1e-3)
        related = await p.search(RELATED, limit=1)
        # Same ranking and the same score the cosine collection gives
        # (unit-length embeddings: 1 - d/2 == cosine similarity).
        assert related[0].content == FACT
        assert related[0].score == pytest.approx(_cosine(RELATED, FACT), abs=1e-3)

    @pytest.mark.asyncio
    async def test_existing_collection_metadata_is_not_rewritten(self, tmp_path) -> None:
        client = chromadb.PersistentClient(path=str(tmp_path))
        client.create_collection(
            "agent_memory", embedding_function=_BagOfWords(), metadata={"team": "travel"}
        )
        del client
        p = ChromaProvider(persist_directory=str(tmp_path), embedding_function=_BagOfWords())
        assert p._space == "l2"
        assert (p._collection.metadata or {}).get("team") == "travel"


class TestScoreConversion:
    @pytest.mark.parametrize(
        ("distance", "space", "expected"),
        [
            (0.0, "cosine", 1.0),
            (0.25, "cosine", 0.75),
            (1.0, "cosine", 0.0),
            (1.6, "cosine", 0.0),  # opposite direction clamps to 0
            (0.25, "ip", 0.75),
            (0.0, "l2", 1.0),
            (1.4682, "l2", 1 - 1.4682 / 2),  # distance seen in the guide's probe
            (2.0, "l2", 0.0),
            (3.0, "l2", 0.0),
            (None, "cosine", 0.0),
        ],
    )
    def test_chroma_score(self, distance: float | None, space: str, expected: float) -> None:
        assert _chroma_score(distance, space) == pytest.approx(expected)

    def test_space_from_legacy_metadata(self) -> None:
        class _Old:
            configuration_json = None
            metadata = {"hnsw:space": "ip"}

        assert _chroma_space(_Old()) == "ip"

    def test_space_defaults_to_l2(self) -> None:
        class _Bare:
            metadata = None

        assert _chroma_space(_Bare()) == "l2"
