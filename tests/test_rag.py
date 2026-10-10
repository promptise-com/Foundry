"""Unit tests for the RAG base foundation (promptise.rag)."""

from __future__ import annotations

import pytest

from promptise.rag import (
    Chunk,
    Chunker,
    Document,
    DocumentLoader,
    Embedder,
    InMemoryVectorStore,
    RAGPipeline,
    RecursiveTextChunker,
    RetrievalResult,
    VectorStore,
    content_hash,
    rag_to_tool,
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class TestDocument:
    def test_minimal(self):
        doc = Document(id="doc-1", text="hello world")
        assert doc.id == "doc-1"
        assert doc.text == "hello world"
        assert doc.metadata == {}

    def test_with_metadata(self):
        doc = Document(
            id="doc-1",
            text="hello",
            metadata={"source": "test.md", "title": "Test"},
        )
        assert doc.metadata["source"] == "test.md"
        assert doc.metadata["title"] == "Test"


class TestChunk:
    def test_minimal(self):
        c = Chunk(id="c-1", document_id="doc-1", text="chunk text")
        assert c.id == "c-1"
        assert c.document_id == "doc-1"
        assert c.text == "chunk text"
        assert c.embedding is None
        assert c.metadata == {}

    def test_with_embedding(self):
        c = Chunk(
            id="c-1",
            document_id="doc-1",
            text="chunk text",
            embedding=[0.1, 0.2, 0.3],
        )
        assert c.embedding == [0.1, 0.2, 0.3]


class TestRetrievalResult:
    def test_score_clamped_to_valid_range(self):
        chunk = Chunk(id="c-1", document_id="doc-1", text="hi")
        r1 = RetrievalResult(chunk=chunk, score=1.5)
        assert r1.score == 1.0
        r2 = RetrievalResult(chunk=chunk, score=-0.5)
        assert r2.score == 0.0

    def test_shortcut_properties(self):
        chunk = Chunk(
            id="c-1",
            document_id="doc-1",
            text="hello",
            metadata={"source": "test"},
        )
        r = RetrievalResult(chunk=chunk, score=0.8)
        assert r.text == "hello"
        assert r.metadata == {"source": "test"}


# ---------------------------------------------------------------------------
# Base class default behavior — must raise NotImplementedError
# ---------------------------------------------------------------------------


class TestBaseClasses:
    @pytest.mark.asyncio
    async def test_document_loader_raises(self):
        loader = DocumentLoader()
        with pytest.raises(NotImplementedError):
            await loader.load()

    @pytest.mark.asyncio
    async def test_chunker_raises(self):
        chunker = Chunker()
        with pytest.raises(NotImplementedError):
            await chunker.chunk(Document(id="d", text="t"))

    @pytest.mark.asyncio
    async def test_embedder_raises(self):
        embedder = Embedder()
        with pytest.raises(NotImplementedError):
            await embedder.embed(["text"])
        with pytest.raises(NotImplementedError):
            _ = embedder.dimension

    @pytest.mark.asyncio
    async def test_vector_store_raises(self):
        store = VectorStore()
        with pytest.raises(NotImplementedError):
            await store.add([])
        with pytest.raises(NotImplementedError):
            await store.search([0.1, 0.2])
        with pytest.raises(NotImplementedError):
            await store.delete(["id"])


# ---------------------------------------------------------------------------
# RecursiveTextChunker
# ---------------------------------------------------------------------------


class TestRecursiveTextChunker:
    def test_validation(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            RecursiveTextChunker(chunk_size=0)
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            RecursiveTextChunker(chunk_size=100, overlap=-1)
        with pytest.raises(ValueError, match="overlap must be less than chunk_size"):
            RecursiveTextChunker(chunk_size=50, overlap=50)

    @pytest.mark.asyncio
    async def test_short_text_returns_single_chunk(self):
        chunker = RecursiveTextChunker(chunk_size=500)
        doc = Document(id="d", text="Just a short sentence.")
        chunks = await chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Just a short sentence."
        assert chunks[0].document_id == "d"
        assert chunks[0].metadata["chunk_index"] == 0

    @pytest.mark.asyncio
    async def test_empty_text_returns_no_chunks(self):
        chunker = RecursiveTextChunker()
        doc = Document(id="d", text="")
        chunks = await chunker.chunk(doc)
        assert chunks == []

    @pytest.mark.asyncio
    async def test_long_text_splits_into_multiple_chunks(self):
        # Build text with clear paragraph breaks
        paragraphs = [f"Paragraph number {i}. " * 20 for i in range(10)]
        text = "\n\n".join(paragraphs)

        chunker = RecursiveTextChunker(chunk_size=500, overlap=50)
        doc = Document(id="d", text=text)
        chunks = await chunker.chunk(doc)

        assert len(chunks) > 1
        # Every chunk should fit within chunk_size
        for c in chunks:
            assert len(c.text) <= 500
        # Chunk indexes should be sequential
        for i, c in enumerate(chunks):
            assert c.metadata["chunk_index"] == i

    @pytest.mark.asyncio
    async def test_metadata_inherited_from_document(self):
        chunker = RecursiveTextChunker(chunk_size=100)
        doc = Document(
            id="d",
            text="First paragraph.\n\nSecond paragraph.",
            metadata={"source": "test.md", "author": "alice"},
        )
        chunks = await chunker.chunk(doc)
        for c in chunks:
            assert c.metadata["source"] == "test.md"
            assert c.metadata["author"] == "alice"
            # Chunk-level metadata added
            assert "chunk_index" in c.metadata
            assert "char_start" in c.metadata
            assert "char_end" in c.metadata

    @pytest.mark.asyncio
    async def test_chunk_ids_are_deterministic(self):
        chunker = RecursiveTextChunker(chunk_size=100)
        doc = Document(id="doc-1", text="First.\n\nSecond.\n\nThird.")
        chunks = await chunker.chunk(doc)
        assert chunks[0].id == "doc-1:chunk-0"
        if len(chunks) > 1:
            assert chunks[1].id == "doc-1:chunk-1"


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------


class TestInMemoryVectorStore:
    @pytest.mark.asyncio
    async def test_add_and_search(self):
        store = InMemoryVectorStore()
        chunks = [
            Chunk(id="c1", document_id="d1", text="cat", embedding=[1.0, 0.0, 0.0]),
            Chunk(id="c2", document_id="d1", text="dog", embedding=[0.0, 1.0, 0.0]),
            Chunk(id="c3", document_id="d1", text="bird", embedding=[0.0, 0.0, 1.0]),
        ]
        await store.add(chunks)
        assert await store.count() == 3

        # Query for cat
        results = await store.search([1.0, 0.0, 0.0], limit=2)
        assert len(results) == 2
        assert results[0].chunk.id == "c1"  # best match
        assert results[0].score > results[1].score  # sorted by score

    @pytest.mark.asyncio
    async def test_rejects_chunk_without_embedding(self):
        store = InMemoryVectorStore()
        with pytest.raises(ValueError, match="no embedding"):
            await store.add([Chunk(id="c1", document_id="d1", text="hi")])

    @pytest.mark.asyncio
    async def test_dimension_validation(self):
        store = InMemoryVectorStore(dimension=3)
        with pytest.raises(ValueError, match="dimension"):
            await store.add([Chunk(id="c1", document_id="d1", text="hi", embedding=[0.1, 0.2])])

    @pytest.mark.asyncio
    async def test_delete(self):
        store = InMemoryVectorStore()
        await store.add(
            [
                Chunk(id="c1", document_id="d1", text="a", embedding=[1.0, 0.0]),
                Chunk(id="c2", document_id="d1", text="b", embedding=[0.0, 1.0]),
            ]
        )
        await store.delete(["c1"])
        assert await store.count() == 1
        results = await store.search([1.0, 0.0])
        assert all(r.chunk.id != "c1" for r in results)

    @pytest.mark.asyncio
    async def test_delete_by_document(self):
        store = InMemoryVectorStore()
        await store.add(
            [
                Chunk(id="c1", document_id="d1", text="a", embedding=[1.0, 0.0]),
                Chunk(id="c2", document_id="d1", text="b", embedding=[0.5, 0.5]),
                Chunk(id="c3", document_id="d2", text="c", embedding=[0.0, 1.0]),
            ]
        )
        removed = await store.delete_by_document("d1")
        assert removed == 2
        assert await store.count() == 1

    @pytest.mark.asyncio
    async def test_metadata_filter(self):
        store = InMemoryVectorStore()
        await store.add(
            [
                Chunk(
                    id="c1",
                    document_id="d1",
                    text="a",
                    embedding=[1.0, 0.0],
                    metadata={"category": "tech"},
                ),
                Chunk(
                    id="c2",
                    document_id="d1",
                    text="b",
                    embedding=[1.0, 0.0],
                    metadata={"category": "food"},
                ),
            ]
        )
        results = await store.search([1.0, 0.0], filter={"category": "tech"})
        assert len(results) == 1
        assert results[0].chunk.metadata["category"] == "tech"


# ---------------------------------------------------------------------------
# RAGPipeline end-to-end
# ---------------------------------------------------------------------------


class _FakeLoader(DocumentLoader):
    """Returns a fixed list of documents."""

    def __init__(self, docs: list[Document]) -> None:
        self._docs = docs

    async def load(self) -> list[Document]:
        return self._docs


class _FakeEmbedder(Embedder):
    """Deterministic embedder: returns a vector based on text length."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Simple vector based on text characteristics
        return [
            [
                float(len(t) % 10) / 10,
                float(t.count("a")) / max(len(t), 1),
                float(t.count("e")) / max(len(t), 1),
            ]
            for t in texts
        ]

    @property
    def dimension(self) -> int:
        return 3


class TestRAGPipeline:
    @pytest.mark.asyncio
    async def test_index_and_retrieve(self):
        docs = [
            Document(id="d1", text="The quick brown fox jumps over the lazy dog."),
            Document(id="d2", text="A journey of a thousand miles begins with a single step."),
            Document(id="d3", text="To be or not to be, that is the question."),
        ]
        pipeline = RAGPipeline(
            loader=_FakeLoader(docs),
            chunker=RecursiveTextChunker(chunk_size=200, overlap=20),
            embedder=_FakeEmbedder(),
            store=InMemoryVectorStore(dimension=3),
        )

        report = await pipeline.index()
        assert report.documents_loaded == 3
        assert report.chunks_created >= 3
        assert report.chunks_stored == report.chunks_created
        assert report.errors == []
        assert report.duration_seconds >= 0

        # Retrieve
        results = await pipeline.retrieve("question about life", limit=3)
        assert len(results) > 0
        # All results should have valid scores
        for r in results:
            assert 0.0 <= r.score <= 1.0

    @pytest.mark.asyncio
    async def test_index_with_explicit_documents(self):
        pipeline = RAGPipeline(
            embedder=_FakeEmbedder(),
            store=InMemoryVectorStore(),
        )
        docs = [Document(id="d1", text="hello world")]
        report = await pipeline.index(documents=docs)
        assert report.documents_loaded == 1
        assert report.chunks_stored >= 1

    @pytest.mark.asyncio
    async def test_index_without_loader_or_docs_raises(self):
        pipeline = RAGPipeline(
            embedder=_FakeEmbedder(),
            store=InMemoryVectorStore(),
        )
        with pytest.raises(ValueError, match="no loader"):
            await pipeline.index()

    @pytest.mark.asyncio
    async def test_delete_document(self):
        pipeline = RAGPipeline(
            loader=_FakeLoader(
                [
                    Document(id="d1", text="document one content"),
                    Document(id="d2", text="document two content"),
                ]
            ),
            embedder=_FakeEmbedder(),
            store=InMemoryVectorStore(),
        )
        await pipeline.index()
        count_before = await pipeline._store.count()

        removed = await pipeline.delete_document("d1")
        assert removed >= 1
        count_after = await pipeline._store.count()
        assert count_after < count_before

    @pytest.mark.asyncio
    async def test_close(self):
        pipeline = RAGPipeline(
            embedder=_FakeEmbedder(),
            store=InMemoryVectorStore(),
        )
        await pipeline.close()  # should not raise


# ---------------------------------------------------------------------------
# rag_to_tool adapter
# ---------------------------------------------------------------------------


class TestRagToTool:
    @pytest.mark.asyncio
    async def test_creates_langchain_tool(self):
        pipeline = RAGPipeline(
            loader=_FakeLoader(
                [
                    Document(id="d1", text="The capital of France is Paris."),
                ]
            ),
            embedder=_FakeEmbedder(),
            store=InMemoryVectorStore(),
        )
        await pipeline.index()

        tool = rag_to_tool(
            pipeline,
            name="search_docs",
            description="Search internal docs.",
        )
        assert tool.name == "search_docs"
        assert "internal docs" in tool.description.lower()

        # Invoke the tool
        result = await tool.ainvoke({"query": "capital"})
        assert isinstance(result, str)
        assert "Paris" in result or "result" in result.lower()

    @pytest.mark.asyncio
    async def test_tool_handles_no_results(self):
        pipeline = RAGPipeline(
            embedder=_FakeEmbedder(),
            store=InMemoryVectorStore(),
        )
        # Pipeline not indexed — should return nothing
        tool = rag_to_tool(pipeline, name="search", description="x")
        result = await tool.ainvoke({"query": "anything"})
        assert "no relevant" in result.lower() or result == "No relevant results found."

    @pytest.mark.asyncio
    async def test_json_format(self):
        pipeline = RAGPipeline(
            loader=_FakeLoader([Document(id="d1", text="hello world")]),
            embedder=_FakeEmbedder(),
            store=InMemoryVectorStore(),
        )
        await pipeline.index()

        tool = rag_to_tool(pipeline, format="json")
        result = await tool.ainvoke({"query": "hello"})
        import json

        # Should be valid JSON: an untrusted-data notice plus the results
        parsed = json.loads(result)
        assert "untrusted" in parsed["notice"]
        assert isinstance(parsed["results"], list)
        assert parsed["results"][0]["text"] == "hello world"


# ---------------------------------------------------------------------------
# content_hash utility
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_deterministic(self):
        a = content_hash("hello world")
        b = content_hash("hello world")
        assert a == b

    def test_different_inputs_different_hashes(self):
        a = content_hash("hello")
        b = content_hash("world")
        assert a != b

    def test_fixed_length(self):
        for text in ["", "a", "much longer input text" * 100]:
            assert len(content_hash(text)) == 12
