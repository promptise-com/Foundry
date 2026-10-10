"""Semantic Cache — cache LLM responses by query similarity.

Reduces LLM API costs by 30-50% by serving cached responses for
semantically similar queries.  All embedding runs locally by default.

Example::

    from promptise import build_agent, SemanticCache

    agent = await build_agent(
        ...,
        cache=SemanticCache(),
    )

    # First call → LLM, result cached
    # Second similar call → cache hit, no LLM call
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import UUID

from langchain_core.callbacks import AsyncCallbackHandler

if TYPE_CHECKING:
    pass  # type: ignore[import-not-found]

logger = logging.getLogger("promptise.cache")

# Lazy numpy: imported on first SemanticCache use, not at module load time.
# This keeps the base install free of a heavy numpy dependency for users
# who don't enable semantic caching. Install with `pip install "promptise[all]"`
# or `pip install numpy` if you want to use SemanticCache.
_np: Any = None


def _get_np() -> Any:
    """Lazy import numpy. Raises clear error if missing."""
    global _np
    if _np is None:
        try:
            import numpy as np_module

            _np = np_module
        except ImportError as e:
            raise ImportError(
                "SemanticCache requires numpy. Install with: "
                'pip install numpy  (or pip install "promptise[all]")'
            ) from e
    return _np


__all__ = [
    "SemanticCache",
    "EmbeddingProvider",
    "LocalEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "InMemoryCacheBackend",
    "RedisCacheBackend",
    "CacheEntry",
    "CacheStats",
]


# ═══════════════════════════════════════════════════════════════════════
# Protocols & Data Types
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers.

    Implement this to plug in any embedding model or API::

        class MyProvider:
            async def embed(self, texts: list[str]) -> list[list[float]]:
                return my_model.encode(texts)
    """

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class CacheEntry:
    """A single cached response.

    Attributes:
        query_text: The original user query.
        response_text: The extracted response text.
        output: The output to replay.  The agent stores ``{"messages": [answer]}``
            (the final assistant message only) and, on a hit, returns it after
            the current request's own messages.
        embedding: The query embedding vector.
        scope_key: Isolation scope (e.g. ``"user:user-42"``).
        context_fingerprint: Hash of memory + history + prompt context.
        model_id: LLM model that generated this response.
        instruction_hash: Hash of the system instructions.
        checksum: SHA-256 of response_text for corruption detection.
        created_at: Wall-clock (``time.time()``) timestamp of creation.
        ttl: Time-to-live in seconds.
        metadata: Extra info (tools_used, token count, etc.).
        similarity: Cosine similarity between the stored query and the
            query that found this entry.  Set on the entries a search
            returns; ``None`` on stored entries.
    """

    query_text: str
    response_text: str
    output: Any
    embedding: list[float]
    scope_key: str
    context_fingerprint: str
    model_id: str
    instruction_hash: str
    checksum: str
    created_at: float
    ttl: int
    metadata: dict[str, Any] = field(default_factory=dict)
    similarity: float | None = None

    @property
    def age(self) -> float:
        """Seconds since this entry was stored."""
        return max(0.0, time.time() - self.created_at)

    @property
    def expired(self) -> bool:
        """Check if this entry has expired.

        Uses ``time.time()`` (wall clock) consistently across all backends.
        Both InMemory and Redis backends set ``created_at`` with ``time.time()``.
        """
        return (time.time() - self.created_at) > self.ttl

    def verify_checksum(self) -> bool:
        """Verify response integrity."""
        return self.checksum == hashlib.sha256(self.response_text.encode()).hexdigest()


@dataclass
class CacheStats:
    """Cache performance statistics.

    Attributes:
        hits: Number of cache hits.
        misses: Number of cache misses.
        stores: Number of entries stored.
        evictions: Number of entries evicted.
        hit_rate: Proportion of requests served from cache.
    """

    hits: int = 0
    misses: int = 0
    stores: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════
# Embedding Providers
# ═══════════════════════════════════════════════════════════════════════

# Global embedding model cache — per cache.py module.
# Note: tool_optimization.py has its own model loading; they are NOT
# shared.  This avoids coupling between modules while keeping models
# cached within each module's lifetime.
_embedding_model_cache: dict[str, Any] = {}


class LocalEmbeddingProvider:
    """Local embedding via sentence-transformers.

    Uses the same model loading pattern as tool optimization.
    If the same model is already loaded (e.g. for semantic tool
    selection), the instance is shared — no duplicate memory.

    Args:
        model: Model name or local directory path.

    Example::

        provider = LocalEmbeddingProvider()  # all-MiniLM-L6-v2
        provider = LocalEmbeddingProvider(model="BAAI/bge-small-en-v1.5")
        provider = LocalEmbeddingProvider(model="/models/local/embeddings")
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self._model_name = model
        self._encode_fn: Any | None = None

    def warmup(self) -> None:
        """Pre-load the embedding model."""
        self._get_encode_fn()

    def _get_encode_fn(self) -> Any:
        if self._encode_fn is not None:
            return self._encode_fn

        if self._model_name in _embedding_model_cache:
            model = _embedding_model_cache[self._model_name]
            self._encode_fn = model.encode
            return self._encode_fn

        try:
            import warnings

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*resume_download.*")
                warnings.filterwarnings("ignore", message=".*UNEXPECTED.*")
                from sentence_transformers import SentenceTransformer

                model = SentenceTransformer(self._model_name)
            _embedding_model_cache[self._model_name] = model
            self._encode_fn = model.encode
            logger.info("Loaded embedding model: %s", self._model_name)
            return self._encode_fn
        except ImportError:
            raise ImportError(
                "sentence-transformers required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the local model."""
        encode_fn = self._get_encode_fn()
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: encode_fn(texts, normalize_embeddings=True).tolist()
        )
        return embeddings


class OpenAIEmbeddingProvider:
    """Embedding via OpenAI or Azure OpenAI API.

    Args:
        model: Model name (e.g. ``"text-embedding-3-small"``).
        api_key: OpenAI API key.
        base_url: Custom base URL (for Azure or proxies).
        azure_endpoint: Azure OpenAI endpoint URL.
        azure_deployment: Azure deployment name.

    Example::

        # OpenAI
        provider = OpenAIEmbeddingProvider(
            model="text-embedding-3-small",
            api_key="${OPENAI_API_KEY}",
        )

        # Azure OpenAI
        provider = OpenAIEmbeddingProvider(
            model="text-embedding-3-small",
            azure_endpoint="https://xxx.openai.azure.com",
            azure_deployment="my-embedding",
            api_key="${AZURE_OPENAI_KEY}",
        )
    """

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._azure_endpoint = azure_endpoint
        self._azure_deployment = azure_deployment

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts via OpenAI API."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required for OpenAI embeddings: pip install httpx")

        # Resolve API key from env if needed
        api_key = self._api_key
        if api_key and "${" in api_key:
            from .env_resolver import resolve_env_var

            api_key = resolve_env_var(api_key)
        if not api_key:
            raise ValueError(
                "OpenAIEmbeddingProvider: api_key is empty. "
                "Set the environment variable or pass the key directly."
            )

        # Build URL
        if self._azure_endpoint:
            url = (
                f"{self._azure_endpoint}/openai/deployments/"
                f"{self._azure_deployment or self._model}/embeddings"
                f"?api-version=2024-02-01"
            )
            headers = {"api-key": api_key or ""}
        else:
            url = f"{self._base_url or 'https://api.openai.com'}/v1/embeddings"
            headers = {"Authorization": f"Bearer {api_key}"}

        headers["Content-Type"] = "application/json"

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                url,
                headers=headers,
                json={"input": texts, "model": self._model},
            )
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]


# ═══════════════════════════════════════════════════════════════════════
# Cache Backends
# ═══════════════════════════════════════════════════════════════════════


class InMemoryCacheBackend:
    """In-memory cache with numpy-based similarity search.

    Stores entries per scope with LRU eviction. Thread-safe via asyncio
    (single event loop). No persistence — lost on restart.

    Args:
        max_entries_per_scope: Max entries per scope partition.
        max_total_entries: Max entries across all scopes.
    """

    def __init__(
        self,
        *,
        max_entries_per_scope: int = 1000,
        max_total_entries: int = 100_000,
    ) -> None:
        self._max_per_scope = max_entries_per_scope
        self._max_total = max_total_entries
        # scope_key → list of CacheEntry
        self._entries: dict[str, list[CacheEntry]] = {}
        # scope_key → numpy array of embeddings (N x D)
        self._embeddings: dict[str, Any] = {}
        self._stats = CacheStats()
        self._total_entries = 0

    async def search(
        self,
        scope_key: str,
        embedding: list[float],
        threshold: float,
        *,
        match: Callable[[CacheEntry], bool] | None = None,
    ) -> CacheEntry | None:
        """Find the most similar entry above ``threshold``.

        Only entries for which ``match`` returns ``True`` are considered,
        so a closer entry stored under another context, model or
        instruction set does not hide the one that applies.  The returned
        entry is a copy with :attr:`CacheEntry.similarity` set.  Hits and
        misses are counted by :class:`SemanticCache`, which knows whether
        the request was served.
        """
        entries = self._entries.get(scope_key, [])
        if not entries:
            return None

        emb_matrix = self._embeddings.get(scope_key)
        if emb_matrix is None or len(emb_matrix) == 0:
            return None

        # Remove expired entries first
        self._evict_expired(scope_key)
        entries = self._entries.get(scope_key, [])
        if not entries:
            return None

        np = _get_np()
        emb_matrix = self._embeddings[scope_key]
        query_vec = np.array(embedding, dtype=np.float32)
        scores = np.dot(emb_matrix, query_vec)

        corrupted: list[int] = []
        found: CacheEntry | None = None
        for idx in np.argsort(-scores, kind="stable"):
            score = float(scores[idx])
            if score < threshold:
                break
            entry = entries[int(idx)]
            if match is not None and not match(entry):
                continue
            if not entry.verify_checksum():
                logger.warning("Cache: checksum mismatch, treating as miss")
                corrupted.append(int(idx))
                continue
            found = replace(entry, similarity=score)
            break

        for idx in sorted(corrupted, reverse=True):
            self._remove_entry(scope_key, idx)
        return found

    async def store(self, scope_key: str, entry: CacheEntry) -> None:
        """Store an entry, evicting LRU if at capacity."""
        np = _get_np()
        if scope_key not in self._entries:
            self._entries[scope_key] = []
            self._embeddings[scope_key] = np.empty((0, len(entry.embedding)), dtype=np.float32)

        entries = self._entries[scope_key]

        # Evict oldest if at per-scope limit
        while len(entries) >= self._max_per_scope:
            self._remove_entry(scope_key, 0)
            self._stats.evictions += 1

        # Evict oldest globally if at total limit
        while self._total_entries >= self._max_total:
            self._evict_oldest_global()
            self._stats.evictions += 1

        entries.append(entry)
        emb_vec = np.array([entry.embedding], dtype=np.float32)
        self._embeddings[scope_key] = np.vstack([self._embeddings[scope_key], emb_vec])
        self._total_entries += 1
        self._stats.stores += 1

    async def invalidate(self, scope_key: str, pattern: str | None = None) -> int:
        """Evict entries matching a pattern (or all for scope)."""
        entries = self._entries.get(scope_key, [])
        if not entries:
            return 0

        if pattern is None:
            count = len(entries)
            self._total_entries -= count
            del self._entries[scope_key]
            del self._embeddings[scope_key]
            return count

        # Pattern match against tool names in metadata
        # Escape regex metacharacters first, then convert glob * to .*
        safe_pattern = re.escape(pattern).replace(r"\*", ".*")
        regex = re.compile(f"^{safe_pattern}$")
        to_remove = []
        for i, e in enumerate(entries):
            tools = e.metadata.get("tools_used", [])
            if any(regex.match(t) for t in tools):
                to_remove.append(i)

        for idx in reversed(to_remove):
            self._remove_entry(scope_key, idx)
        return len(to_remove)

    async def purge_user(self, user_id: str) -> int:
        """Remove all entries for a user (GDPR compliance)."""
        # Exact match — not prefix! "user:12" must NOT match "user:123"
        exact_key = f"user:{user_id}"
        count = 0
        keys_to_remove = [k for k in self._entries if k == exact_key]
        for key in keys_to_remove:
            count += len(self._entries[key])
            self._total_entries -= len(self._entries[key])
            del self._entries[key]
            del self._embeddings[key]
        return count

    async def stats(self) -> CacheStats:
        return self._stats

    def _remove_entry(self, scope_key: str, idx: int) -> None:
        np = _get_np()
        entries = self._entries[scope_key]
        entries.pop(idx)
        emb = self._embeddings[scope_key]
        self._embeddings[scope_key] = np.delete(emb, idx, axis=0)
        self._total_entries -= 1

    def _evict_expired(self, scope_key: str) -> None:
        entries = self._entries.get(scope_key, [])
        to_remove = [i for i, e in enumerate(entries) if e.expired]
        for idx in reversed(to_remove):
            self._remove_entry(scope_key, idx)

    def _evict_oldest_global(self) -> None:
        """Evict the oldest entry across all scopes."""
        oldest_key = None
        oldest_time = float("inf")
        for key, entries in self._entries.items():
            if entries and entries[0].created_at < oldest_time:
                oldest_time = entries[0].created_at
                oldest_key = key
        if oldest_key:
            self._remove_entry(oldest_key, 0)


# ═══════════════════════════════════════════════════════════════════════
# SemanticCache
# ═══════════════════════════════════════════════════════════════════════


class RedisCacheBackend:
    """Redis-backed cache with vector similarity search.

    Stores cache entries as JSON in Redis hashes. Embeddings are stored
    per scope and similarity is computed by fetching all embeddings for
    a scope and running numpy dot product locally. This avoids requiring
    the RediSearch module while keeping similarity search functional.

    Optional AES encryption at rest via ``encrypt_values=True``.
    Encryption key is read from ``PROMPTISE_CACHE_KEY`` env var or
    auto-generated per process.

    Args:
        redis_url: Redis connection URL (e.g. ``redis://localhost:6379``).
        max_entries_per_scope: Max entries per scope partition.
        max_total_entries: Max entries across all scopes.
        encrypt_values: Encrypt cached response values at rest.
    """

    def __init__(
        self,
        *,
        redis_url: str = "redis://localhost:6379",
        max_entries_per_scope: int = 1000,
        max_total_entries: int = 100_000,
        encrypt_values: bool = False,
    ) -> None:
        self._max_per_scope = max_entries_per_scope
        self._max_total = max_total_entries
        self._stats = CacheStats()
        self._encrypt = encrypt_values
        self._fernet: Any | None = None
        self._redis: Any | None = None
        self._redis_url = redis_url

        if encrypt_values:
            self._init_encryption()

    def _init_encryption(self) -> None:
        """Initialize Fernet encryption from env or auto-generate."""
        try:
            from cryptography.fernet import Fernet
        except ImportError:
            raise ImportError("cryptography required for encrypted cache: pip install cryptography")
        import os

        key = os.environ.get("PROMPTISE_CACHE_KEY")
        if not key:
            key = Fernet.generate_key().decode()
            logger.warning(
                "PROMPTISE_CACHE_KEY not set — auto-generated encryption key. "
                "Cache will not survive restarts. Set the env var for persistence."
            )
        self._fernet = Fernet(key if isinstance(key, bytes) else key.encode())

    async def _get_redis(self) -> Any:
        """Lazy Redis connection with reconnection on failure."""
        if self._redis is not None:
            # Verify the connection is alive (with timeout to prevent hangs)
            try:
                await asyncio.wait_for(self._redis.ping(), timeout=5.0)
                return self._redis
            except Exception:
                logger.warning("Redis cache: connection lost, reconnecting")
                try:
                    await self._redis.aclose()
                except Exception:
                    pass
                self._redis = None

        try:
            import redis.asyncio as aioredis
        except ImportError:
            raise ImportError("redis[asyncio] required for Redis cache backend: pip install redis")
        self._redis = aioredis.from_url(self._redis_url, decode_responses=False)
        return self._redis

    def _scope_entries_key(self, scope_key: str) -> str:
        """Redis key for the hash storing entries for a scope."""
        return f"promptise:cache:entries:{scope_key}"

    def _scope_embeddings_key(self, scope_key: str) -> str:
        """Redis key for the list storing embeddings for a scope."""
        return f"promptise:cache:embeddings:{scope_key}"

    def _scope_order_key(self, scope_key: str) -> str:
        """Redis key for the sorted set tracking insertion order (LRU)."""
        return f"promptise:cache:order:{scope_key}"

    def _encrypt_value(self, data: bytes) -> bytes:
        if self._fernet:
            return self._fernet.encrypt(data)
        return data

    def _decrypt_value(self, data: bytes) -> bytes:
        if self._fernet:
            return self._fernet.decrypt(data)
        return data

    async def search(
        self,
        scope_key: str,
        embedding: list[float],
        threshold: float,
        *,
        match: Callable[[CacheEntry], bool] | None = None,
    ) -> CacheEntry | None:
        """Find the most similar entry above ``threshold``.

        Same contract as :meth:`InMemoryCacheBackend.search`: candidates
        are tried from most to least similar and the first one that is
        live, intact and accepted by ``match`` is returned, with
        :attr:`CacheEntry.similarity` set.
        """
        r = await self._get_redis()

        # Get all entry IDs for this scope
        entries_key = self._scope_entries_key(scope_key)
        entry_ids = await r.hkeys(entries_key)
        if not entry_ids:
            return None

        # Get all embeddings
        emb_key = self._scope_embeddings_key(scope_key)
        raw_embeddings = await r.hgetall(emb_key)
        if not raw_embeddings:
            return None

        # Score every entry locally, keep those above the threshold
        np = _get_np()
        query_vec = np.array(embedding, dtype=np.float32)
        candidates: list[tuple[float, bytes]] = []
        for eid in entry_ids:
            raw_emb = raw_embeddings.get(eid)
            if raw_emb is None:
                continue
            emb_vec = np.frombuffer(raw_emb, dtype=np.float32)
            score = float(np.dot(emb_vec, query_vec))
            if score >= threshold:
                candidates.append((score, eid))
        candidates.sort(key=lambda c: c[0], reverse=True)

        for score, eid in candidates:
            entry = await self._load_entry(r, scope_key, eid, embedding)
            if entry is None or (match is not None and not match(entry)):
                continue
            return replace(entry, similarity=score)
        return None

    async def _load_entry(
        self, r: Any, scope_key: str, entry_id: bytes, embedding: list[float]
    ) -> CacheEntry | None:
        """Load one entry; evict it and return ``None`` if expired or corrupt."""
        entries_key = self._scope_entries_key(scope_key)
        emb_key = self._scope_embeddings_key(scope_key)
        raw_entry = await r.hget(entries_key, entry_id)
        if raw_entry is None:
            return None

        try:
            decrypted = self._decrypt_value(raw_entry)
            entry_data = json.loads(decrypted)
        except Exception:
            logger.warning("Redis cache: failed to deserialize entry, treating as miss")
            await r.hdel(entries_key, entry_id)
            return None

        # Deserialize LangGraph output
        raw_output = entry_data["output"]
        output: Any
        if isinstance(raw_output, dict) and raw_output.get("_fallback"):
            output = {"messages": []}  # Minimal output on fallback
        else:
            try:
                from langchain_core.load import load

                output = load(raw_output)
            except Exception:
                output = raw_output  # Use as-is if load fails

        entry = CacheEntry(
            query_text=entry_data["query_text"],
            response_text=entry_data["response_text"],
            output=output,
            embedding=embedding,  # Use the query embedding (we matched)
            scope_key=scope_key,
            context_fingerprint=entry_data["context_fingerprint"],
            model_id=entry_data["model_id"],
            instruction_hash=entry_data["instruction_hash"],
            checksum=entry_data["checksum"],
            created_at=entry_data["created_at"],
            ttl=entry_data["ttl"],
            metadata=entry_data.get("metadata", {}),
        )

        # Check TTL (use wall clock for Redis — persists across restarts)
        if (time.time() - entry.created_at) > entry.ttl:
            await r.hdel(entries_key, entry_id)
            await r.hdel(emb_key, entry_id)
            await r.zrem(self._scope_order_key(scope_key), entry_id)
            return None

        # Verify checksum
        if not entry.verify_checksum():
            logger.warning("Redis cache: checksum mismatch, evicting corrupted entry")
            await r.hdel(entries_key, entry_id)
            await r.hdel(emb_key, entry_id)
            return None

        return entry

    async def store(self, scope_key: str, entry: CacheEntry) -> None:
        """Store an entry in Redis."""
        r = await self._get_redis()
        import secrets

        entry_id = secrets.token_hex(8).encode()

        entries_key = self._scope_entries_key(scope_key)
        emb_key = self._scope_embeddings_key(scope_key)
        order_key = self._scope_order_key(scope_key)

        # Evict oldest if at per-scope limit
        count = await r.hlen(entries_key)
        while count >= self._max_per_scope:
            oldest = await r.zrange(order_key, 0, 0)
            if not oldest:
                break
            await r.hdel(entries_key, oldest[0])
            await r.hdel(emb_key, oldest[0])
            await r.zrem(order_key, oldest[0])
            count -= 1
            self._stats.evictions += 1

        # Serialize entry (use wall clock for Redis)
        # LangGraph output contains AIMessage/ToolMessage objects that aren't
        # JSON-serializable. Use LangChain's dumpd() for safe serialization.
        try:
            from langchain_core.load import dumpd

            serialized_output = dumpd(entry.output)
        except Exception as exc:
            # Fallback: store response_text only (lose tool call details)
            logger.warning(
                "Cache: output serialization failed (%s), storing text-only fallback",
                type(exc).__name__,
            )
            serialized_output = {"_fallback": True, "response_text": entry.response_text}

        entry_data = {
            "query_text": entry.query_text,
            "response_text": entry.response_text,
            "output": serialized_output,
            "context_fingerprint": entry.context_fingerprint,
            "model_id": entry.model_id,
            "instruction_hash": entry.instruction_hash,
            "checksum": entry.checksum,
            "created_at": time.time(),  # Wall clock for Redis persistence
            "ttl": entry.ttl,
            "metadata": entry.metadata,
        }

        raw_entry = json.dumps(entry_data).encode()
        encrypted = self._encrypt_value(raw_entry)

        # Store entry + embedding + order
        await r.hset(entries_key, entry_id, encrypted)
        np = _get_np()
        emb_bytes = np.array(entry.embedding, dtype=np.float32).tobytes()
        await r.hset(emb_key, entry_id, emb_bytes)
        await r.zadd(order_key, {entry_id: time.time()})

        # NOTE: We do NOT set expire() on the shared hash/sorted-set keys.
        # These hold ALL entries for a scope — setting TTL would clobber
        # earlier entries with longer TTLs. TTL is enforced per-entry
        # during search() via created_at + ttl check.

        self._stats.stores += 1

    async def invalidate(self, scope_key: str, pattern: str | None = None) -> int:
        """Evict entries for a scope."""
        r = await self._get_redis()
        entries_key = self._scope_entries_key(scope_key)

        if pattern is None:
            count = await r.hlen(entries_key)
            await r.delete(entries_key)
            await r.delete(self._scope_embeddings_key(scope_key))
            await r.delete(self._scope_order_key(scope_key))
            return count

        # Pattern-based invalidation: load all entries, match, delete
        all_entries = await r.hgetall(entries_key)
        safe_pattern = re.escape(pattern).replace(r"\*", ".*")
        regex = re.compile(f"^{safe_pattern}$")
        to_remove = []

        for eid, raw in all_entries.items():
            try:
                decrypted = self._decrypt_value(raw)
                data = json.loads(decrypted)
                tools = data.get("metadata", {}).get("tools_used", [])
                if any(regex.match(t) for t in tools):
                    to_remove.append(eid)
            except Exception:
                to_remove.append(eid)  # Remove corrupted entries

        emb_key = self._scope_embeddings_key(scope_key)
        order_key = self._scope_order_key(scope_key)
        for eid in to_remove:
            await r.hdel(entries_key, eid)
            await r.hdel(emb_key, eid)
            await r.zrem(order_key, eid)

        return len(to_remove)

    async def purge_user(self, user_id: str) -> int:
        """Remove all entries for a user (GDPR compliance)."""
        r = await self._get_redis()
        exact_key = f"user:{user_id}"
        entries_key = self._scope_entries_key(exact_key)
        count = await r.hlen(entries_key)
        if count > 0:
            await r.delete(entries_key)
            await r.delete(self._scope_embeddings_key(exact_key))
            await r.delete(self._scope_order_key(exact_key))
        return count

    async def stats(self) -> CacheStats:
        return self._stats

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None


class SemanticCache:
    """Semantic cache for agent responses.

    Caches LLM responses by query similarity using local or cloud
    embeddings.  Reduces API costs by 30-50% for workloads with
    repetitive queries.

    **Security:** Default scope is ``per_user`` — each user gets an
    isolated cache partition.  No ``CallerContext`` = no caching.
    Cached responses always pass through output guardrails.

    Args:
        backend: ``"memory"`` (default) or ``"redis"``.
        redis_url: Redis connection URL (when backend is ``"redis"``).
        embedding: An :class:`EmbeddingProvider`, a model name string,
            or ``None`` for the default local model.
        similarity_threshold: Minimum cosine similarity for a cache hit.
        default_ttl: Default time-to-live in seconds.
        scope: Cache isolation: ``"per_user"`` (default),
            ``"per_session"``, or ``"shared"``.
        max_entries_per_user: Max entries per scope partition.
        max_total_entries: Max entries across all scopes.
        encrypt_values: Encrypt cached values at rest (Redis only).
        ttl_patterns: Regex → TTL overrides for time-sensitive queries.
        invalidate_on_write: Evict the caller's cached responses when the
            agent calls a write tool (see :meth:`is_write_tool`).
        cache_multi_turn: Cache requests that carry earlier turns.  Off by
            default: a follow-up such as "What river runs through it?"
            means something different in every conversation, so only
            single-message requests are cached.  When on, the earlier
            messages are hashed into the cache key, so a follow-up only
            hits for the same conversation history.
        cache_tool_turns: Cache answers for turns in which the agent called
            tools.  Off by default, so a cached answer never stands in for
            a tool call.  When on, a turn is cached only if every tool it
            called is read-only; a turn that called a write tool is never
            cached, because replaying it would skip the write.
        write_tools: Tool names (``*`` wildcards allowed) always treated as
            writes, whatever their annotations say.
        read_only_tools: Tool names (``*`` wildcards allowed) treated as
            read-only, for tools without MCP annotations (for example
            ``extra_tools``) or with wrong ones.  ``write_tools`` wins when
            a name matches both.
        shared_data_acknowledged: Required when scope is ``"shared"``.

    Example::

        # One-liner
        cache = SemanticCache()

        # Full config
        cache = SemanticCache(
            backend="redis",
            redis_url="redis://localhost:6379",
            similarity_threshold=0.92,
            scope="per_user",
            ttl_patterns={r"current|now|today": 60},
        )

        agent = await build_agent(..., cache=cache)
    """

    def __init__(
        self,
        *,
        backend: str = "memory",
        redis_url: str | None = None,
        embedding: EmbeddingProvider | str | None = None,
        similarity_threshold: float = 0.92,
        default_ttl: int = 3600,
        scope: str = "per_user",
        max_entries_per_user: int = 1000,
        max_total_entries: int = 100_000,
        encrypt_values: bool = False,
        ttl_patterns: dict[str, int] | None = None,
        invalidate_on_write: bool = True,
        cache_multi_turn: bool = False,
        cache_tool_turns: bool = False,
        write_tools: Sequence[str] | None = None,
        read_only_tools: Sequence[str] | None = None,
        shared_data_acknowledged: bool = False,
    ) -> None:
        self._threshold = similarity_threshold
        self._default_ttl = default_ttl
        self._scope = scope
        self._ttl_patterns = {re.compile(k): v for k, v in (ttl_patterns or {}).items()}
        self._invalidate_on_write = invalidate_on_write
        self._cache_multi_turn = cache_multi_turn
        self._cache_tool_turns = cache_tool_turns
        self._write_tools = [_glob_regex(p) for p in (write_tools or [])]
        self._read_only_tools = [_glob_regex(p) for p in (read_only_tools or [])]
        # scope_key → number of write invalidations so far.  A request
        # that started before a write must not store what it read.
        self._write_generation: dict[str, int] = {}

        # Warn if shared scope without acknowledgment
        if scope == "shared" and not shared_data_acknowledged:
            logger.warning(
                "SemanticCache: scope='shared' without shared_data_acknowledged=True. "
                "Cached responses will be shared across ALL users. Ensure no "
                "personalized data is cached. Set shared_data_acknowledged=True "
                "to suppress this warning."
            )

        # Resolve embedding provider
        self._embedding: EmbeddingProvider
        if embedding is None:
            self._embedding = LocalEmbeddingProvider()
        elif isinstance(embedding, str):
            self._embedding = LocalEmbeddingProvider(model=embedding)
        else:
            self._embedding = embedding

        # Resolve backend
        self._backend: InMemoryCacheBackend | RedisCacheBackend
        if backend == "memory":
            self._backend = InMemoryCacheBackend(
                max_entries_per_scope=max_entries_per_user,
                max_total_entries=max_total_entries,
            )
        elif backend == "redis":
            if not redis_url:
                raise ValueError("redis_url is required when backend='redis'")
            self._backend = RedisCacheBackend(
                redis_url=redis_url,
                max_entries_per_scope=max_entries_per_user,
                max_total_entries=max_total_entries,
                encrypt_values=encrypt_values,
            )
        else:
            raise ValueError(f"Unknown cache backend: {backend!r}")

    def warmup(self) -> None:
        """Pre-load the embedding model.

        Call at startup to avoid download/load latency on first cache check.
        """
        if isinstance(self._embedding, LocalEmbeddingProvider):
            self._embedding.warmup()

    def check_dependencies(self) -> None:
        """Raise :class:`ImportError` if this cache cannot run here.

        ``build_agent(cache=...)`` calls this, so a missing ``numpy`` or
        ``sentence-transformers`` fails the build with an install hint
        instead of silently disabling the cache on every request.  Only
        checks that the packages are installed; :meth:`warmup` loads the
        model.
        """
        import importlib.util

        missing = []
        if importlib.util.find_spec("numpy") is None:
            missing.append("numpy")
        if (
            isinstance(self._embedding, LocalEmbeddingProvider)
            and importlib.util.find_spec("sentence_transformers") is None
        ):
            missing.append("sentence-transformers")
        if missing:
            raise ImportError(
                f"SemanticCache needs {' and '.join(missing)}, which "
                f"{'is' if len(missing) == 1 else 'are'} not installed. Install with: "
                f'pip install {" ".join(missing)}  (or pip install "promptise[all]"). '
                "To embed through an API instead of a local model, pass "
                "embedding=OpenAIEmbeddingProvider(...)."
            )

    # ── Policy ───────────────────────────────────────────────────────

    def allows_conversation(self, messages: Sequence[Any]) -> bool:
        """Whether a request with these input messages may use the cache.

        A request carrying earlier turns (any user, assistant or tool
        message before the last one) is only cached when
        ``cache_multi_turn=True``.  System messages do not count.
        """
        return self._cache_multi_turn or _conversation_turns(messages) <= 1

    def is_write_tool(self, name: str, annotations: Mapping[str, Any] | None = None) -> bool:
        """Whether calling this tool may change data.

        Decided in this order: a ``write_tools`` match is a write; a
        ``read_only_tools`` match is not; otherwise the tool's MCP
        ``readOnlyHint`` annotation decides.  A tool with no annotation
        counts as a write -- the MCP default for ``readOnlyHint`` is false.
        """
        if any(p.match(name) for p in self._write_tools):
            return True
        if any(p.match(name) for p in self._read_only_tools):
            return False
        if name in _BUILTIN_READ_ONLY_TOOLS:
            return False
        return not (annotations or {}).get("readOnlyHint", False)

    def write_generation(self, caller: Any | None = None) -> int:
        """Number of write invalidations so far in ``caller``'s scope.

        Pass the value read when a request starts to :meth:`store`; the
        entry is dropped if a write invalidated the scope in between.
        """
        scope_key = self._build_scope_key(caller)
        return self._write_generation.get(scope_key or "", 0)

    # ── Core API ─────────────────────────────────────────────────────

    async def check(
        self,
        query_text: str,
        *,
        context_fingerprint: str = "",
        caller: Any | None = None,
        model_id: str | None = None,
        instruction_hash: str = "",
    ) -> CacheEntry | None:
        """Check for a cached response.

        Returns the cached entry if a semantically similar query was
        previously cached with the same context, model, and instructions.
        Returns ``None`` on cache miss.

        Args:
            query_text: The user's query.
            context_fingerprint: Hash of memory + history context.
            caller: :class:`CallerContext` for scope isolation.
            model_id: LLM model identifier.
            instruction_hash: Hash of the system instructions.
        """
        scope_key = self._build_scope_key(caller)
        if scope_key is None:
            self._backend._stats.misses += 1
            logger.debug(
                "Cache: no CallerContext or user_id provided — caching disabled "
                "for this request. Pass caller=CallerContext(user_id=...) to "
                "enable caching, or use scope='shared' for public data."
            )
            return None

        if not query_text.strip():
            self._backend._stats.misses += 1
            return None

        # Embed the query
        query_emb = await self._embed_one(query_text, "skipping cache check")
        if query_emb is None:
            self._backend._stats.misses += 1
            return None

        # Only entries stored under the same context, model and instructions
        # apply -- a different context means the stored answer may be stale.
        wanted_model = model_id or ""

        def _applies(entry: CacheEntry) -> bool:
            return (
                entry.context_fingerprint == context_fingerprint
                and entry.model_id == wanted_model
                and entry.instruction_hash == instruction_hash
            )

        entry = await self._backend.search(scope_key, query_emb, self._threshold, match=_applies)
        if entry is None:
            self._backend._stats.misses += 1
            return None

        self._backend._stats.hits += 1
        logger.debug(
            "Cache hit for query %r (scope=%s, similarity=%.3f, age=%.0fs)",
            query_text[:50],
            scope_key,
            entry.similarity or 0.0,
            entry.age,
        )
        return entry

    async def store(
        self,
        query_text: str,
        response_text: str,
        output: Any,
        *,
        context_fingerprint: str = "",
        caller: Any | None = None,
        model_id: str | None = None,
        instruction_hash: str = "",
        tools_used: list[str] | None = None,
        write_generation: int | None = None,
    ) -> None:
        """Store a response in the cache.

        Args:
            query_text: The user's query.
            response_text: The extracted response text.
            output: The full LangGraph output dict.
            context_fingerprint: Hash of memory + history context.
            caller: :class:`CallerContext` for scope isolation.
            model_id: LLM model identifier.
            instruction_hash: Hash of the system instructions.
            tools_used: List of tool names called during this invocation.
            write_generation: :meth:`write_generation` read when the request
                started.  If a write invalidated the scope since, the
                response may predate the write and is not stored.
        """
        scope_key = self._build_scope_key(caller)
        if scope_key is None:
            return  # No identity → don't cache

        if not query_text.strip() or not response_text.strip():
            return

        if (
            write_generation is not None
            and self._write_generation.get(scope_key, 0) != write_generation
        ):
            logger.debug(
                "Cache: scope %s was written to during this request, not storing", scope_key
            )
            return

        # Embed the query
        query_emb = await self._embed_one(query_text, "skipping store")
        if query_emb is None:
            return
        # The embedding call yields: re-check that no write landed meanwhile.
        if (
            write_generation is not None
            and self._write_generation.get(scope_key, 0) != write_generation
        ):
            return

        # Compute TTL
        ttl = self._resolve_ttl(query_text)

        # Build entry
        entry = CacheEntry(
            query_text=query_text,
            response_text=response_text,
            output=output,
            embedding=query_emb,
            scope_key=scope_key,
            context_fingerprint=context_fingerprint,
            model_id=model_id or "",
            instruction_hash=instruction_hash,
            checksum=hashlib.sha256(response_text.encode()).hexdigest(),
            created_at=time.time(),  # Wall clock — consistent with Redis backend and .expired
            ttl=ttl,
            metadata={"tools_used": tools_used or []},
        )

        await self._backend.store(scope_key, entry)
        logger.debug(
            "Cache stored for query %r (scope=%s, ttl=%ds)",
            query_text[:50],
            scope_key,
            ttl,
        )

    async def invalidate_for_write(self, tool_name: str, caller: Any | None = None) -> None:
        """Evict ``caller``'s cached responses after a write.

        The agent calls this when a write tool (see :meth:`is_write_tool`)
        finishes or fails, so answers computed before the write are not
        served after it.  Responses still in flight from before the write
        are not stored either.  Does nothing with
        ``invalidate_on_write=False``.
        """
        if not self._invalidate_on_write:
            return

        scope_key = self._build_scope_key(caller)
        if scope_key is None:
            return

        self._write_generation[scope_key] = self._write_generation.get(scope_key, 0) + 1
        count = await self._backend.invalidate(scope_key)
        if count > 0:
            logger.debug(
                "Cache invalidated %d entries for scope %s (write tool: %s)",
                count,
                scope_key,
                tool_name,
            )

    async def purge_user(self, user_id: str, *, tenant_id: str | None = None) -> int:
        """Remove all cached entries for a user.

        Use for GDPR right-to-erasure compliance.

        Args:
            user_id: The user whose cache to purge.
            tenant_id: The user's tenant, when tenant-scoped callers were
                used.  Must match the ``CallerContext.tenant_id`` the
                entries were stored under.

        Note:
            This purges the **per-user** scope (``user:<id>``), which is the
            default and the GDPR-relevant one.  A cache created with
            ``scope="per_session"`` keys entries by the user's session, so
            ``purge_user`` does not remove them; they expire with their TTL.

        Returns:
            Number of entries removed.
        """
        scoped = self._scoped_user_id(tenant_id, user_id)
        if not scoped:
            return 0
        count = await self._backend.purge_user(scoped)
        logger.info("Cache purged %d entries for user %s", count, scoped)
        # Emit cache.purged event if notifier is available
        notifier = getattr(self, "_event_notifier", None)
        if notifier is not None:
            try:
                from promptise.events import emit_event

                emit_event(
                    notifier,
                    "cache.purged",
                    "info",
                    {"user_id": user_id, "entries_removed": count},
                )
            except Exception:
                pass
        return count

    async def stats(self) -> CacheStats:
        """Get cache performance statistics."""
        return await self._backend.stats()

    async def close(self) -> None:
        """Close the cache backend (Redis connections, etc.).

        Called automatically by ``agent.shutdown()``.
        """
        if hasattr(self._backend, "close"):
            await self._backend.close()

    # ── Internals ────────────────────────────────────────────────────

    async def _embed_one(self, text: str, action: str) -> list[float] | None:
        """Embed one text, or log why not and return ``None``."""
        try:
            embeddings = await self._embedding.embed([text])
        except Exception:
            logger.warning("Cache: embedding failed, %s", action, exc_info=True)
            return None
        if not embeddings:
            logger.warning("Cache: embedding returned empty list, %s", action)
            return None
        return embeddings[0]

    @staticmethod
    def _sanitize_scope_id(value: str) -> str:
        """Sanitize a scope identifier — reject empty, replace delimiters."""
        if not value or not value.strip():
            return ""
        # Replace colons and slashes to prevent scope key confusion
        return value.strip().replace(":", "_").replace("/", "_")

    @classmethod
    def _scoped_user_id(cls, tenant_id: str | None, user_id: str) -> str:
        """Derive an **injective**, delimiter-free scope id for ``(tenant, id)``.

        Untenanted → the sanitized id (unchanged, backward compatible).
        Tenanted → ``"t:" + sha256(f"{len(tenant)}:{tenant}:{id}")[:40]``.
        This is injective in ``(tenant, id)`` because the hash input is
        **length-prefixed**: the leading ``len(tenant)`` fixes exactly how
        many following characters are the tenant, so no two distinct
        ``(tenant, id)`` pairs can produce the same material string no matter
        where colons fall.  The two keyspaces are also **disjoint**: the
        sanitizer strips ``:`` from untenanted ids, so an untenanted scope id
        can never contain a colon, while every tenanted key begins with
        ``"t:"`` — no untenanted user_id (not even one literally shaped like
        ``"t:<hex>"``, which sanitizes its colon away) can collide with a
        tenanted hash.

        A plain ``__`` join was not injective (``("acme","corp__alice")`` and
        ``("acme__corp","alice")`` both produced ``acme__corp__alice``); a
        bare ``f"{tenant}::{id}"`` was likewise ambiguous under adversarial
        colons (``("a", ":b")`` and ``("a:", "b")`` collide); and a ``"t."``
        prefix still overlapped the untenanted namespace (a user named
        ``"t.<hex>"`` collided) — hence the length-prefixed material and the
        ``"t:"`` colon prefix.

        One derivation shared by scope-key construction and
        :meth:`purge_user`, so a tenant-scoped purge matches exactly what
        was stored.  Empty string when the id sanitizes to nothing.
        """
        if not user_id or not user_id.strip():
            return ""
        if not tenant_id:
            return cls._sanitize_scope_id(user_id)
        # Length-prefix the tenant so the hash INPUT decodes unambiguously
        # regardless of colon placement — ``f"{tenant}::{user}"`` alone is not
        # injective ("a"+"::"+":b" == "a:"+"::"+"b"), but ``len:tenant:user``
        # is (read the length, then exactly that many chars are the tenant).
        material = f"{len(tenant_id)}:{tenant_id}:{user_id}"
        digest = hashlib.sha256(material.encode()).hexdigest()
        return f"t:{digest[:40]}"

    @staticmethod
    def _session_scope_id(scoped_user: str, session_id: str) -> str:
        """Injective, delimiter-free id for one user's session.

        ``scoped_user`` comes from :meth:`_scoped_user_id`; the hash input
        is length-prefixed, so no other ``(user, session)`` pair produces
        the same id.
        """
        material = f"{len(scoped_user)}:{scoped_user}:{session_id}"
        return hashlib.sha256(material.encode()).hexdigest()[:40]

    def _build_scope_key(self, caller: Any | None) -> str | None:
        """Build the scope isolation key from caller context.

        Returns None if caching should be disabled for this request.
        When the caller carries a ``tenant_id``, it is baked into the
        scope key — tenants with identical user ids never share entries.
        """
        if self._scope == "shared":
            logger.debug("Cache: using shared scope (no per-user isolation)")
            return "shared"

        if caller is None:
            return None

        user_id = getattr(caller, "user_id", None)
        tenant_id = getattr(caller, "tenant_id", None)

        if self._scope == "per_user":
            if user_id is None:
                return None
            safe_id = self._scoped_user_id(tenant_id, user_id)
            if not safe_id:
                return None
            return f"user:{safe_id}"

        if self._scope == "per_session":
            # A session partition always belongs to one user: session ids
            # come from the application (often short or guessable), so two
            # users naming the same session must not share answers.
            if user_id is None:
                return None
            safe_id = self._scoped_user_id(tenant_id, user_id)
            if not safe_id:
                return None
            session_id = (getattr(caller, "metadata", None) or {}).get("session_id")
            if session_id and str(session_id).strip():
                return f"session:{self._session_scope_id(safe_id, str(session_id))}"
            # No session id: fall back to the user's partition
            return f"user:{safe_id}"

        return None

    def _resolve_ttl(self, query_text: str) -> int:
        """Resolve TTL using pattern overrides, then default."""
        query_lower = query_text.lower()
        for pattern, ttl in self._ttl_patterns.items():
            if pattern.search(query_lower):
                return ttl
        return self._default_ttl


# ═══════════════════════════════════════════════════════════════════════
# Helper functions for agent integration
# ═══════════════════════════════════════════════════════════════════════


# Promptise's own tools that never change data.
_BUILTIN_READ_ONLY_TOOLS = frozenset({"request_more_tools"})

_ROLE_ALIASES = {"user": "human", "assistant": "ai"}


def _glob_regex(pattern: str) -> re.Pattern[str]:
    """Compile a tool-name pattern where ``*`` matches anything."""
    return re.compile("^" + re.escape(pattern).replace(r"\*", ".*") + "$")


def _message_role(message: Any) -> str:
    """Normalized role of a dict, tuple, string or LangChain message."""
    if isinstance(message, dict):
        role = str(message.get("role") or message.get("type") or "human")
    elif isinstance(message, tuple) and message:
        role = str(message[0])
    elif isinstance(message, str):
        role = "human"
    else:
        role = str(getattr(message, "type", "") or type(message).__name__)
    return _ROLE_ALIASES.get(role, role)


def _conversation_turns(messages: Sequence[Any]) -> int:
    """Number of non-system messages (user, assistant and tool turns)."""
    return sum(1 for m in messages if _message_role(m) != "system")


def _message_digest(message: Any) -> str:
    """Stable digest of one message: role, content and tool calls.

    Tool-call ids are left out -- they are random per run, so including
    them would make every conversation look new.
    """
    if isinstance(message, dict):
        content: Any = message.get("content", "")
        tool_calls: Any = message.get("tool_calls") or []
        name = message.get("name")
    elif isinstance(message, tuple):
        content, tool_calls, name = (message[1] if len(message) > 1 else ""), [], None
    elif isinstance(message, str):
        content, tool_calls, name = message, [], None
    else:
        content = getattr(message, "content", "")
        tool_calls = [
            {"name": tc.get("name"), "args": tc.get("args")}
            for tc in getattr(message, "tool_calls", None) or []
        ]
        name = getattr(message, "name", None)
    material = json.dumps(
        [_message_role(message), name, content, tool_calls], sort_keys=True, default=str
    )
    return hashlib.sha256(material.encode()).hexdigest()


def compute_context_fingerprint(
    *,
    memory_results: list[Any] | None = None,
    conversation_length: int = 0,
    instruction_hash: str = "",
    tool_set_hash: str = "",
    history: Sequence[Any] | None = None,
) -> str:
    """Compute a fingerprint of the current context.

    Used as part of the cache key to ensure stale context doesn't
    produce stale cache hits.  Hashes the actual memory content —
    not just the count — so new memories invalidate stale cache entries.

    Args:
        memory_results: Memory search results injected for this request.
        conversation_length: Number of input messages.
        instruction_hash: Hash of the system instructions.
        tool_set_hash: Hash of the tools available.
        history: The messages before the query (earlier turns, system
            messages).  Their role, content and tool calls are hashed, so
            the same follow-up question in two different conversations
            gets two different fingerprints.
    """
    # Hash actual memory content, not just count
    mem_hash = "none"
    if memory_results:
        mem_texts = []
        for r in memory_results:
            if hasattr(r, "content"):
                mem_texts.append(str(r.content)[:200])
            elif hasattr(r, "text"):
                mem_texts.append(str(r.text)[:200])
            else:
                mem_texts.append(str(r)[:200])
        mem_hash = hashlib.sha256("|".join(mem_texts).encode()).hexdigest()[:16]

    hist_hash = "none"
    if history:
        joined = "|".join(_message_digest(m) for m in history)
        hist_hash = hashlib.sha256(joined.encode()).hexdigest()[:32]

    parts = [
        f"mem:{mem_hash}",
        f"conv:{conversation_length}",
        f"hist:{hist_hash}",
        f"inst:{instruction_hash[:16]}",
        f"tools:{tool_set_hash[:16]}",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def turn_tool_calls(output: Any) -> list[str]:
    """Names of the tools called in the last turn of a graph output.

    The last turn is everything after the final user message, so tool
    calls from earlier turns in the history are not counted.
    """
    names: list[str] = []
    messages = output.get("messages", []) if isinstance(output, dict) else []
    for msg in reversed(messages):
        if _message_role(msg) == "human":
            break
        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
            if name:
                names.append(name)
    names.reverse()
    return names


def served_model(output: Any) -> str | None:
    """Model that wrote the final answer, when a :class:`FallbackChain` recorded it."""
    messages = output.get("messages", []) if isinstance(output, dict) else []
    for msg in reversed(messages):
        if _message_role(msg) == "ai":
            metadata = getattr(msg, "response_metadata", None) or {}
            model = metadata.get("fallback_model")
            return str(model) if model else None
    return None


def answer_messages(output: Any) -> list[Any]:
    """The final answer of a graph output: its last assistant message.

    This is what the cache stores and replays.  The rest of the output --
    the asker's own messages, injected context, tool calls and results --
    belongs to the request that produced it, not to the one a hit serves.
    """
    messages = output.get("messages", []) if isinstance(output, dict) else []
    for msg in reversed(messages):
        if _message_role(msg) == "ai" and not (getattr(msg, "tool_calls", None) or []):
            return [msg]
    return []


def replay_output(input_messages: Sequence[Any], cached_output: Any) -> dict[str, Any]:
    """Graph output for a cache hit: this request's messages plus the cached answer."""
    from langchain_core.messages import convert_to_messages

    return {
        "messages": [*convert_to_messages(list(input_messages)), *answer_messages(cached_output)]
    }


def tool_annotations(tool: Any) -> Mapping[str, Any] | None:
    """MCP annotations recorded on a LangChain tool, if any."""
    metadata = getattr(tool, "metadata", None) or {}
    annotations = metadata.get("mcp_annotations")
    return annotations if isinstance(annotations, Mapping) else None


class ToolCallWatcher(AsyncCallbackHandler):
    """Watch one request's tool calls for :class:`SemanticCache`.

    Added to the run's callbacks by the agent.  When a write tool (see
    :meth:`SemanticCache.is_write_tool`) finishes or fails, the caller's
    cached responses are evicted at once -- before the agent writes its
    answer, and even if the run fails afterwards.

    Args:
        cache: The agent's cache.
        caller: The request's :class:`CallerContext`.
        annotations: Tool name → MCP annotations for the agent's tools.
        approval_gated: Names of the tools behind an approval gate.  A
            turn that called one is never cached: the approval is a
            decision about that one call, and a replayed answer would
            skip asking.
    """

    def __init__(
        self,
        cache: SemanticCache,
        caller: Any | None,
        annotations: Mapping[str, Mapping[str, Any] | None],
        approval_gated: Collection[str] = (),
    ) -> None:
        super().__init__()
        self._cache = cache
        self._caller = caller
        self._annotations = annotations
        self._approval_gated = frozenset(approval_gated)
        self._running: dict[UUID, str] = {}
        self.tools_called: list[str] = []
        self.write_generation = cache.write_generation(caller)

    def is_write(self, name: str) -> bool:
        return self._cache.is_write_tool(name, self._annotations.get(name))

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        name = str((serialized or {}).get("name") or kwargs.get("name") or "")
        self._running[run_id] = name
        self.tools_called.append(name)

    async def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        await self._finished(run_id)

    async def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        await self._finished(run_id)

    async def _finished(self, run_id: UUID) -> None:
        name = self._running.pop(run_id, None)
        if name is not None and self.is_write(name):
            await self._cache.invalidate_for_write(name, caller=self._caller)

    async def settle(self, tools_in_output: Sequence[str]) -> bool:
        """Finish the turn; return whether its answer may be cached.

        ``tools_in_output`` are the tool calls found in the graph output,
        which covers tools run without callbacks.  A write among them that
        the watcher did not see evicts the cache now.
        """
        unseen_writes = [
            n for n in tools_in_output if n not in self.tools_called and self.is_write(n)
        ]
        for name in unseen_writes:
            await self._cache.invalidate_for_write(name, caller=self._caller)
        called = [*self.tools_called, *tools_in_output]
        if any(self.is_write(n) or n in self._approval_gated for n in called):
            return False
        return not called or self._cache._cache_tool_turns


def compute_instruction_hash(instructions: str | None) -> str:
    """Hash the system instructions for cache key inclusion."""
    if not instructions:
        return "default"
    return hashlib.sha256(instructions.encode()).hexdigest()[:16]
