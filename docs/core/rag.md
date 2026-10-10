# RAG (Retrieval-Augmented Generation)

A small, pluggable foundation for adding document retrieval to Promptise agents. The framework ships with base classes you subclass for your own loader, chunker, embedder, and vector store — plus a batteries-included reference implementation (in-memory store + recursive text chunker) you can use immediately or swap out piece by piece as you productionise.

```python
from promptise import (
    build_agent,
    RAGPipeline,
    RecursiveTextChunker,
    InMemoryVectorStore,
    rag_to_tool,
)
from promptise.config import HTTPServerSpec

# Your code: plug in a loader + embedder
pipeline = RAGPipeline(
    loader=MyMarkdownLoader("./knowledge_base"),
    chunker=RecursiveTextChunker(chunk_size=800, overlap=100),
    embedder=MyOpenAIEmbedder(),
    store=InMemoryVectorStore(),
)
await pipeline.index()

# Expose retrieval as a tool the agent can call
docs_tool = rag_to_tool(
    pipeline,
    name="search_internal_docs",
    description="Search Acme Corp's engineering wiki.",
)

agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={"tools": HTTPServerSpec(url="http://localhost:8000/mcp")},
    extra_tools=[docs_tool],
)
```

---

## Architecture

```
┌──────────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐
│ DocumentLoader│-->│ Chunker │-->│ Embedder │-->│ VectorStore │
└──────────────┘   └──────────┘   └──────────┘   └──────────────┘
                                                        │
                                                        v
                                                  rag_to_tool()
                                                        │
                                                        v
                                                  LangChain Tool
                                                        │
                                                        v
                                                    Your Agent
```

Four base classes, a pipeline orchestrator, and a tool adapter. That's it. Each base class has one or two methods — override them, ship your own implementation, and pass it to `RAGPipeline`.

---

## Core Data Types

| Type | Purpose |
|---|---|
| `Document` | A raw text document with `id`, `text`, and metadata. Whatever your loader produces. |
| `Chunk` | A piece of a document with `id`, `document_id`, `text`, optional `embedding`, and metadata. Produced by the chunker. |
| `RetrievalResult` | A chunk plus a `score` in `[0.0, 1.0]`. Returned by `search()` and `retrieve()`. |
| `IndexReport` | Summary of `pipeline.index()`: documents loaded, chunks created, chunks stored, errors, duration. |

All four are `@dataclass`-based — cheap to construct, easy to pickle, introspectable.

---

## Base Classes (override these)

### DocumentLoader

Load raw documents from wherever they live: filesystem, S3, Notion, Confluence, Postgres, a REST API. Override one of two methods:

```python
class DocumentLoader:
    async def load(self) -> list[Document]:
        raise NotImplementedError

    async def iter_load(self) -> AsyncIterator[Document]:
        for doc in await self.load():   # default: yields from load()
            yield doc
```

Override `load()` for sources that fit in memory, or `iter_load()` to stream a large source one document at a time.

**Example — loading markdown files from disk:**

```python
from pathlib import Path
from promptise.rag import Document, DocumentLoader

class MarkdownLoader(DocumentLoader):
    def __init__(self, root: str) -> None:
        self.root = Path(root)

    async def load(self) -> list[Document]:
        return [
            Document(
                id=str(p.relative_to(self.root)),
                text=p.read_text(encoding="utf-8"),
                metadata={"source": str(p), "title": p.stem},
            )
            for p in self.root.rglob("*.md")
        ]
```

That's it. During `index()` the pipeline reads the loader through `iter_load()` (which calls `load()` unless you override it) and feeds the documents to the chunker, about `batch_size` chunks at a time.

### Chunker

Split documents into retrievable chunks. The base class:

```python
class Chunker:
    async def chunk(self, document: Document) -> list[Chunk]:
        raise NotImplementedError
```

**Built-in: `RecursiveTextChunker`** — splits on natural separators (`\n\n`, `\n`, `. `, ` `) with configurable `chunk_size` and `overlap`. Good default for prose, markdown, and code.

```python
from promptise import RecursiveTextChunker

chunker = RecursiveTextChunker(
    chunk_size=800,       # target chars per chunk
    overlap=100,          # overlap between consecutive chunks
    separators=None,      # defaults to ["\n\n", "\n", ". ", " "]
)
```

When no separator splits a piece small enough, the piece is cut into `chunk_size`-character windows that share `overlap` characters. An empty string `""` in `separators` means that character-level cut, so LangChain-style lists such as `["\n\n", "\n", " ", ""]` work. `overlap` must be smaller than `chunk_size`. Empty and whitespace-only documents produce no chunks.

Chunk IDs are deterministic (`{document_id}:chunk-{i}`) so re-indexing the same document produces stable IDs — useful for incremental updates (see [Updating documents](#updating-documents)).

**Rolling your own** is ~20 lines. Subclass `Chunker` and return a list of `Chunk` objects with `document_id`, `text`, and any metadata you want to pass through.

### Embedder

Turn text into dense vectors. The base class has one method plus one property:

```python
class Embedder:
    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError
```

**Example — OpenAI embeddings:**

```python
from openai import AsyncOpenAI
from promptise.rag import Embedder

class OpenAIEmbedder(Embedder):
    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self.client = AsyncOpenAI()
        self.model = model
        self._dim = 1536 if "small" in model else 3072

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        return [d.embedding for d in resp.data]

    @property
    def dimension(self) -> int:
        return self._dim
```

Swap for Cohere, VoyageAI, sentence-transformers, Ollama, or an air-gapped local model by changing this one class.

### VectorStore

Persist chunks and serve similarity queries. The base class:

```python
class VectorStore:
    async def add(self, chunks: list[Chunk]) -> None: ...
    async def search(
        self,
        vector: list[float],
        *,
        limit: int = 5,
        filter: dict | None = None,
    ) -> list[RetrievalResult]: ...
    async def delete(self, chunk_ids: list[str]) -> None: ...
    async def delete_by_document(self, document_id: str) -> int: ...
    async def count(self) -> int: ...
    async def close(self) -> None: ...
```

**Built-in: `InMemoryVectorStore`** — cosine-similarity over an in-process list. Zero external dependencies, no persistence. Great for tests, notebooks, and small corpora (< ~10k chunks). When you outgrow it, subclass and wire up Pinecone, Qdrant, Weaviate, pgvector, Milvus, or Chroma.

```python
from promptise import InMemoryVectorStore

store = InMemoryVectorStore(dimension=1536)  # optional dimension enforcement
```

Scores are cosine similarity mapped from `[-1, 1]` onto `[0, 1]` as `(cosine + 1) / 2`: an identical direction scores `1.0` and an unrelated (orthogonal) chunk scores `0.5`. A query vector whose length differs from the stored vectors (or from `dimension`) raises `ValueError` — the query and the index must come from the same embedder. A non-positive `limit` returns no results. Results are copies, so changing one never changes the stored chunk.

**Rolling your own** means implementing `add`, `search`, and `delete` against your backend. ~50-100 lines for most vector DBs.

---

## RAGPipeline

The orchestrator. Ties a loader + chunker + embedder + store together and exposes two operations: `index()` (build the index) and `retrieve()` (query it).

```python
pipeline = RAGPipeline(
    loader=MyLoader(),
    chunker=RecursiveTextChunker(chunk_size=800, overlap=100),
    embedder=MyEmbedder(),
    store=InMemoryVectorStore(),
)

# Ingest everything from the loader
report = await pipeline.index()
print(f"Indexed {report.documents_loaded} docs -> {report.chunks_stored} chunks")
print(f"Duration: {report.duration_seconds:.2f}s")

# Retrieve
results = await pipeline.retrieve("how do I configure memory?", limit=5)
for r in results:
    print(f"[{r.score:.2f}] {r.metadata.get('source')}: {r.text[:80]}")
```

### Incremental indexing

Pass explicit documents to skip the loader for ad-hoc ingestion:

```python
from promptise import Document

await pipeline.index(documents=[
    Document(id="note-1", text="Meeting notes: ..."),
])
```

### Updating documents

Indexing a document whose `id` is already in the store replaces it: the pipeline removes the document's old chunks, then stores the new ones, so text you removed from a document can no longer be retrieved. If the new version's chunks fail to embed, the old version stays and the failure is in `report.errors`. When the same `id` appears twice in one `index()` call, the last one wins.

Removing old chunks uses `VectorStore.delete_by_document()`. If your store doesn't implement it, the pipeline deletes the chunk ids it stored for that document itself — which only covers documents indexed by the same `RAGPipeline` instance, so implement `delete_by_document()` on a persistent store.

### Deletion

Remove a document and all its chunks:

```python
removed = await pipeline.delete_document("note-1")
```

This uses the same removal as an update, and returns `0` (with a warning) when the store has no `delete_by_document()` and the pipeline never indexed the document.

### Defaults

`Chunker` defaults to `RecursiveTextChunker(chunk_size=500, overlap=50)` if you don't supply one, and `batch_size` (chunks per `embed()` call) defaults to `64`. `embedder` and `store` are required; `loader` is only needed when you call `index()` without documents.

---

## rag_to_tool — expose retrieval to an agent

The glue that turns a `RAGPipeline` into a LangChain tool the agent can call via `extra_tools`. The agent sees a single tool with a `query` argument; calling it runs retrieval and returns formatted results.

```python
from promptise import rag_to_tool

docs_tool = rag_to_tool(
    pipeline,
    name="search_product_docs",
    description="Search Acme Corp's product documentation. Use for how-to questions.",
    limit=5,
    format="markdown",  # "markdown" (default), "json", or "text"
)
```

| Parameter | Default | Purpose |
|---|---|---|
| `name` | `"search_knowledge_base"` | Tool name the LLM sees. Make it specific. |
| `description` | Generic | What's in the knowledge base. The LLM uses this to decide when to call the tool. |
| `limit` | `5` | Default result count (at least 1). The LLM can override it with any value of 1 or more. |
| `format` | `"markdown"` | How results are returned to the LLM. Any other value raises `ValueError`. |

**Format cheat sheet:**

- `"markdown"` — human + LLM friendly, includes source and title headers
- `"json"` — structured: `{"notice": ..., "results": [{score, text, metadata}, ...]}`, best for downstream processing
- `"text"` — plain text with source prefix, minimal token overhead

**Retrieved text is untrusted.** Anyone who can put text into an indexed document — a wiki editor, a customer filing a ticket — can write instructions aimed at the model. The tool output says that the results are data, not instructions to follow. In `"markdown"` and `"text"` the results sit inside a block delimited by a random tag generated for each call (`<retrieved-documents-…>`), so a document can't close the block early and pose as the tool's own output; in `"json"` the `notice` field carries the same statement. This lowers the risk but doesn't remove it: keep side-effecting tools behind [approval](approval.md) when the agent also reads documents that outsiders can write.

---

## Production patterns

### Content hashing for dedup

Use `content_hash()` as part of your document ID to detect unchanged documents and skip re-embedding:

```python
from promptise.rag import content_hash

doc_id = f"{source_path}:{content_hash(text)}"
```

The hash is a stable 12-character string derived from the text — same text always hashes to the same value.

### Metadata filtering

`InMemoryVectorStore.search()` (and `pipeline.retrieve(..., filter=...)`) supports exact-match metadata filters: every key must be present on the chunk with an equal value, so `{"tenant_id": None}` does not match chunks that have no `tenant_id`. Your custom stores should do the same:

```python
results = await store.search(
    query_embedding,
    limit=5,
    filter={"category": "support", "status": "published"},
)
```

### Who can retrieve what

A `RAGPipeline` is one shared corpus, and `rag_to_tool()` does not scope results to the caller: every caller of an agent that has the tool can retrieve every document in the pipeline. Index only documents that every caller may read. For per-user facts, use a [memory provider](memory.md) with `MemoryScope.PER_USER`, which is scoped to the caller automatically.

To keep several tenants' documents in one store, tag each document with its owner and write the tool yourself, filtering on the caller of the current request. `get_current_caller()` returns the `CallerContext` passed to `ainvoke()` / `chat()`, and the tool must refuse when there is none:

```python
from langchain_core.tools import StructuredTool
from promptise import get_current_caller

await pipeline.index(documents=[
    Document(id="acme-sla", text="Our SLA is 4 hours.", metadata={"tenant_id": "acme"}),
])

async def search_my_docs(query: str) -> str:
    caller = get_current_caller()
    if caller is None or caller.tenant_id is None:
        return "No tenant for this request; search refused."
    hits = await pipeline.retrieve(query, limit=5, filter={"tenant_id": caller.tenant_id})
    return "\n\n".join(h.text for h in hits) or "No relevant results found."

docs_tool = StructuredTool.from_function(
    coroutine=search_my_docs,
    name="search_my_docs",
    description="Search the documents of the caller's organisation.",
)
```

A tool you write yourself doesn't get `rag_to_tool()`'s untrusted-data block; add the same kind of notice to its output.

### Hybrid search

`VectorStore` is just a protocol. Subclass to add BM25, reranking, recency boosts, or any hybrid strategy. Return `RetrievalResult` with your combined score.

### Multi-store composition

Run multiple pipelines for different corpora and expose each as its own tool. The agent picks the right one based on the description:

```python
docs_tool = rag_to_tool(docs_pipeline, name="search_docs", description="Product docs.")
tickets_tool = rag_to_tool(tickets_pipeline, name="search_tickets", description="Support tickets.")

agent = await build_agent(
    model="openai:gpt-5-mini",
    servers={},
    extra_tools=[docs_tool, tickets_tool],
)
```

---

## When to use RAG vs. Memory

Both inject external context into the LLM. Different lifecycles:

| | RAG | Memory |
|---|---|---|
| **Source** | Your documents (filesystem, wiki, tickets) | Conversation history, facts the agent observed |
| **Write path** | Offline indexing | Live during agent runs |
| **Trigger** | LLM explicitly calls the tool | Auto-injected before every invocation |
| **Scale** | Millions of chunks | Thousands of memories per user |
| **Typical use** | "What's our refund policy?" | "User's preferred deployment target is GKE" |

Use both together: memory for who the user is, RAG for what the knowledge base says.

---

## Testing

The in-memory components are designed for tests — zero dependencies, deterministic behavior. Wire them up with a fake embedder and you've got an end-to-end RAG test in ~20 lines:

```python
from promptise.rag import (
    Document,
    Embedder,
    InMemoryVectorStore,
    RAGPipeline,
    RecursiveTextChunker,
)

class FakeEmbedder(Embedder):
    async def embed(self, texts):
        return [[float(len(t) % 10) / 10, float(t.count("a")) / max(len(t), 1)] for t in texts]

    @property
    def dimension(self):
        return 2

async def test_retrieval():
    pipeline = RAGPipeline(
        chunker=RecursiveTextChunker(chunk_size=200),
        embedder=FakeEmbedder(),
        store=InMemoryVectorStore(),
    )
    await pipeline.index(documents=[
        Document(id="d1", text="The capital of France is Paris."),
    ])
    results = await pipeline.retrieve("France capital")
    assert any("Paris" in r.text for r in results)
```

See `tests/test_rag.py` in the repo for the full test suite.

---

## Related

- [Memory](memory.md) — auto-injected context from conversation history
- [Tool Optimization](tool-optimization.md) — semantic tool selection for agents with large tool sets
- [Building Agents](agents/building-agents.md) — full `build_agent()` reference
