"""Tests for SemanticCache — embedding, backends, isolation, integration."""

from __future__ import annotations

import hashlib
import time

import numpy as np
import pytest

from promptise.agent import CallerContext
from promptise.cache import (
    CacheEntry,
    InMemoryCacheBackend,
    SemanticCache,
    compute_context_fingerprint,
    compute_instruction_hash,
)

# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _make_entry(
    query: str = "test query",
    response: str = "test response",
    scope_key: str = "user:test",
    embedding: list[float] | None = None,
    ttl: int = 3600,
) -> CacheEntry:
    emb = embedding or [0.1] * 384
    return CacheEntry(
        query_text=query,
        response_text=response,
        output={"messages": [{"content": response}]},
        embedding=emb,
        scope_key=scope_key,
        context_fingerprint="fp1",
        model_id="openai:gpt-5-mini",
        instruction_hash="inst1",
        checksum=hashlib.sha256(response.encode()).hexdigest(),
        created_at=time.time(),
        ttl=ttl,
    )


class FakeEmbeddingProvider:
    """Deterministic embedding for tests — returns normalized random vectors."""

    def __init__(self, dim: int = 384):
        self._dim = dim
        self._cache: dict[str, list[float]] = {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            if text not in self._cache:
                # Deterministic: hash text to seed random
                seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
                rng = np.random.RandomState(seed)
                vec = rng.randn(self._dim).astype(np.float32)
                vec = vec / np.linalg.norm(vec)
                self._cache[text] = vec.tolist()
            results.append(self._cache[text])
        return results


# ═══════════════════════════════════════════════════════════════════════
# InMemoryCacheBackend
# ═══════════════════════════════════════════════════════════════════════


class TestInMemoryBackend:
    @pytest.mark.asyncio
    async def test_store_and_search(self):
        backend = InMemoryCacheBackend()
        entry = _make_entry()
        await backend.store("user:a", entry)
        result = await backend.search("user:a", entry.embedding, 0.9)
        assert result is not None
        assert result.response_text == "test response"

    @pytest.mark.asyncio
    async def test_miss_below_threshold(self):
        backend = InMemoryCacheBackend()
        entry = _make_entry(embedding=[1.0] + [0.0] * 383)
        await backend.store("user:a", entry)
        # Orthogonal vector → similarity ~0
        result = await backend.search("user:a", [0.0] + [1.0] + [0.0] * 382, 0.9)
        assert result is None

    @pytest.mark.asyncio
    async def test_per_user_isolation(self):
        backend = InMemoryCacheBackend()
        entry_a = _make_entry(scope_key="user:a")
        entry_b = _make_entry(scope_key="user:b")
        await backend.store("user:a", entry_a)
        await backend.store("user:b", entry_b)
        # User A can't see user B's entries
        result = await backend.search("user:a", entry_b.embedding, 0.9)
        assert result is not None  # Same embedding, but user A has their own copy

    @pytest.mark.asyncio
    async def test_lru_eviction(self):
        backend = InMemoryCacheBackend(max_entries_per_scope=2)
        e1 = _make_entry(query="q1", embedding=[1.0] + [0.0] * 383)
        e2 = _make_entry(query="q2", embedding=[0.0, 1.0] + [0.0] * 382)
        e3 = _make_entry(query="q3", embedding=[0.0, 0.0, 1.0] + [0.0] * 381)
        await backend.store("user:a", e1)
        await backend.store("user:a", e2)
        await backend.store("user:a", e3)
        # e1 should be evicted (oldest)
        result = await backend.search("user:a", e1.embedding, 0.9)
        assert result is None

    @pytest.mark.asyncio
    async def test_purge_user(self):
        backend = InMemoryCacheBackend()
        await backend.store("user:alice", _make_entry())
        await backend.store("user:alice", _make_entry(query="q2"))
        await backend.store("user:bob", _make_entry())
        count = await backend.purge_user("alice")
        assert count == 2
        # Alice gone, Bob still there
        await backend.stats()
        assert backend._total_entries == 1

    @pytest.mark.asyncio
    async def test_invalidate_all(self):
        backend = InMemoryCacheBackend()
        await backend.store("user:a", _make_entry())
        await backend.store("user:a", _make_entry(query="q2"))
        count = await backend.invalidate("user:a")
        assert count == 2

    @pytest.mark.asyncio
    async def test_checksum_corruption(self):
        backend = InMemoryCacheBackend()
        entry = _make_entry()
        entry.checksum = "corrupted"
        await backend.store("user:a", entry)
        result = await backend.search("user:a", entry.embedding, 0.5)
        assert result is None  # Corrupted → treated as miss

    @pytest.mark.asyncio
    async def test_expired_entry_evicted(self):
        backend = InMemoryCacheBackend()
        entry = _make_entry(ttl=0)  # Immediately expires
        entry.created_at = time.time() - 10  # Created 10s ago
        await backend.store("user:a", entry)
        result = await backend.search("user:a", entry.embedding, 0.5)
        assert result is None  # Expired

    @pytest.mark.asyncio
    async def test_stats(self):
        backend = InMemoryCacheBackend()
        entry = _make_entry()
        await backend.store("user:a", entry)
        await backend.search("user:a", entry.embedding, 0.9)  # found
        await backend.search("user:a", [0.0] * 384, 0.99)  # not found
        stats = await backend.stats()
        # Hits and misses are counted by SemanticCache, which knows whether
        # a found entry was actually served.
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.stores == 1


# ═══════════════════════════════════════════════════════════════════════
# SemanticCache
# ═══════════════════════════════════════════════════════════════════════


class TestSemanticCache:
    @pytest.mark.asyncio
    async def test_cache_hit_on_identical_query(self):
        cache = SemanticCache(
            embedding=FakeEmbeddingProvider(),
            similarity_threshold=0.9,
        )
        caller = CallerContext(user_id="user-1")

        # Store
        await cache.store(
            "What is Python?",
            "Python is a programming language.",
            {"messages": []},
            caller=caller,
            model_id="gpt",
            instruction_hash="h1",
        )

        # Check — identical query
        result = await cache.check(
            "What is Python?",
            caller=caller,
            model_id="gpt",
            instruction_hash="h1",
        )
        assert result is not None
        assert result.response_text == "Python is a programming language."

    @pytest.mark.asyncio
    async def test_cache_miss_different_model(self):
        cache = SemanticCache(
            embedding=FakeEmbeddingProvider(),
            similarity_threshold=0.9,
        )
        caller = CallerContext(user_id="user-1")

        await cache.store(
            "test",
            "response",
            {"messages": []},
            caller=caller,
            model_id="gpt-4",
            instruction_hash="h1",
        )

        # Different model → miss
        result = await cache.check(
            "test",
            caller=caller,
            model_id="claude",
            instruction_hash="h1",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_miss_different_instructions(self):
        cache = SemanticCache(
            embedding=FakeEmbeddingProvider(),
            similarity_threshold=0.9,
        )
        caller = CallerContext(user_id="user-1")

        await cache.store(
            "test",
            "response",
            {"messages": []},
            caller=caller,
            model_id="gpt",
            instruction_hash="v1",
        )

        # Different instructions → miss
        result = await cache.check(
            "test",
            caller=caller,
            model_id="gpt",
            instruction_hash="v2",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_no_caller_no_cache(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider())

        # No caller → store is no-op
        await cache.store("test", "response", {"messages": []})

        # No caller → check returns None
        result = await cache.check("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_no_user_id_no_cache(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider())
        caller = CallerContext()  # user_id is None

        await cache.store("test", "response", {"messages": []}, caller=caller)
        result = await cache.check("test", caller=caller)
        assert result is None

    @pytest.mark.asyncio
    async def test_per_user_isolation(self):
        cache = SemanticCache(
            embedding=FakeEmbeddingProvider(),
            similarity_threshold=0.5,
        )
        alice = CallerContext(user_id="alice")
        bob = CallerContext(user_id="bob")

        await cache.store(
            "secret",
            "alice's secret data",
            {"messages": []},
            caller=alice,
            model_id="gpt",
            instruction_hash="h",
        )

        # Bob can't see Alice's cache
        result = await cache.check(
            "secret",
            caller=bob,
            model_id="gpt",
            instruction_hash="h",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_shared_scope(self):
        cache = SemanticCache(
            embedding=FakeEmbeddingProvider(),
            scope="shared",
            shared_data_acknowledged=True,
            similarity_threshold=0.9,
        )
        alice = CallerContext(user_id="alice")
        bob = CallerContext(user_id="bob")

        await cache.store(
            "weather",
            "sunny",
            {"messages": []},
            caller=alice,
            model_id="gpt",
            instruction_hash="h",
        )

        # Bob CAN see shared cache
        result = await cache.check(
            "weather",
            caller=bob,
            model_id="gpt",
            instruction_hash="h",
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_ttl_pattern_override(self):
        cache = SemanticCache(
            embedding=FakeEmbeddingProvider(),
            default_ttl=3600,
            ttl_patterns={r"current|now|today": 60},
        )
        # "current" matches pattern → TTL should be 60
        ttl = cache._resolve_ttl("What is the current price?")
        assert ttl == 60

        # No pattern match → default TTL
        ttl = cache._resolve_ttl("What is Python?")
        assert ttl == 3600

    @pytest.mark.asyncio
    async def test_invalidate_for_write(self):
        cache = SemanticCache(
            embedding=FakeEmbeddingProvider(),
            similarity_threshold=0.5,
        )
        caller = CallerContext(user_id="user-1")

        await cache.store(
            "count tickets",
            "47 tickets",
            {"messages": []},
            caller=caller,
            model_id="gpt",
            instruction_hash="h",
        )

        # Write tool fires → invalidate
        await cache.invalidate_for_write("create_ticket", caller=caller)

        # Cache should be empty now
        result = await cache.check(
            "count tickets",
            caller=caller,
            model_id="gpt",
            instruction_hash="h",
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_purge_user_gdpr(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider())
        caller = CallerContext(user_id="user-to-delete")

        await cache.store(
            "q1",
            "r1",
            {"messages": []},
            caller=caller,
            model_id="gpt",
            instruction_hash="h",
        )
        await cache.store(
            "q2",
            "r2",
            {"messages": []},
            caller=caller,
            model_id="gpt",
            instruction_hash="h",
        )

        count = await cache.purge_user("user-to-delete")
        assert count == 2

    @pytest.mark.asyncio
    async def test_stats(self):
        cache = SemanticCache(
            embedding=FakeEmbeddingProvider(),
            similarity_threshold=0.9,
        )
        caller = CallerContext(user_id="u")

        await cache.store("q", "r", {}, caller=caller, model_id="m", instruction_hash="h")
        await cache.check("q", caller=caller, model_id="m", instruction_hash="h")
        await cache.check("different", caller=caller, model_id="m", instruction_hash="h")

        stats = await cache.stats()
        assert stats.stores == 1
        assert stats.hits >= 1

    @pytest.mark.asyncio
    async def test_empty_query_skipped(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider())
        caller = CallerContext(user_id="u")
        result = await cache.check("", caller=caller)
        assert result is None

    @pytest.mark.asyncio
    async def test_warmup(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider())
        cache.warmup()  # Should not raise


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════


class TestHelpers:
    def test_compute_context_fingerprint(self):
        fp1 = compute_context_fingerprint(conversation_length=5)
        fp2 = compute_context_fingerprint(conversation_length=10)
        assert fp1 != fp2  # Different context → different fingerprint

    def test_compute_context_fingerprint_deterministic(self):
        fp1 = compute_context_fingerprint(conversation_length=5, instruction_hash="abc")
        fp2 = compute_context_fingerprint(conversation_length=5, instruction_hash="abc")
        assert fp1 == fp2

    def test_compute_instruction_hash(self):
        h1 = compute_instruction_hash("You are helpful.")
        h2 = compute_instruction_hash("You are a data analyst.")
        assert h1 != h2

    def test_compute_instruction_hash_none(self):
        h = compute_instruction_hash(None)
        assert h == "default"


# ═══════════════════════════════════════════════════════════════════════
# CacheEntry
# ═══════════════════════════════════════════════════════════════════════


class TestCacheEntry:
    def test_verify_checksum_valid(self):
        entry = _make_entry()
        assert entry.verify_checksum() is True

    def test_verify_checksum_invalid(self):
        entry = _make_entry()
        entry.checksum = "wrong"
        assert entry.verify_checksum() is False

    def test_expired(self):
        entry = _make_entry(ttl=0)
        entry.created_at = time.time() - 10
        assert entry.expired is True

    def test_not_expired(self):
        entry = _make_entry(ttl=3600)
        assert entry.expired is False


# ═══════════════════════════════════════════════════════════════════════
# Conversation context: multi-turn requests and history fingerprints
# ═══════════════════════════════════════════════════════════════════════

PARIS = [
    {"role": "user", "content": "Tell me about the city of Paris."},
    {"role": "assistant", "content": "Paris is the capital of France."},
    {"role": "user", "content": "What river runs through it?"},
]
LONDON = [
    {"role": "user", "content": "Tell me about the city of London."},
    {"role": "assistant", "content": "London is the capital of the United Kingdom."},
    {"role": "user", "content": "What river runs through it?"},
]


class TestConversationContext:
    def test_history_content_changes_the_fingerprint(self):
        # Same length, same question, different earlier turns.
        paris = compute_context_fingerprint(conversation_length=3, history=PARIS[:-1])
        london = compute_context_fingerprint(conversation_length=3, history=LONDON[:-1])
        assert paris != london

    def test_dict_and_langchain_messages_fingerprint_alike(self):
        from langchain_core.messages import AIMessage, HumanMessage

        as_objects = [
            HumanMessage(content="Tell me about the city of Paris."),
            AIMessage(content="Paris is the capital of France."),
        ]
        assert compute_context_fingerprint(history=PARIS[:-1]) == compute_context_fingerprint(
            history=as_objects
        )

    def test_tool_call_ids_do_not_change_the_fingerprint(self):
        from langchain_core.messages import AIMessage

        def turn(call_id: str) -> list[AIMessage]:
            return [AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": call_id}])]

        assert compute_context_fingerprint(history=turn("a")) == compute_context_fingerprint(
            history=turn("b")
        )

    def test_multi_turn_requests_bypass_the_cache_by_default(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider())
        assert cache.allows_conversation([PARIS[-1]])
        assert not cache.allows_conversation(PARIS)
        # System messages are not turns
        assert cache.allows_conversation([{"role": "system", "content": "Be brief."}, PARIS[-1]])

    def test_cache_multi_turn_allows_conversations(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider(), cache_multi_turn=True)
        assert cache.allows_conversation(PARIS)

    @pytest.mark.asyncio
    async def test_matching_context_found_behind_a_closer_mismatch(self):
        # Paris and London store the same follow-up under different contexts;
        # London's entry must be found although Paris's was stored first.
        cache = SemanticCache(embedding=FakeEmbeddingProvider(), cache_multi_turn=True)
        caller = CallerContext(user_id="alice")
        question = "What river runs through it?"
        await cache.store(question, "The Seine.", {}, caller=caller, context_fingerprint="paris")
        await cache.store(question, "The Thames.", {}, caller=caller, context_fingerprint="london")

        hit = await cache.check(question, caller=caller, context_fingerprint="london")
        assert hit is not None and hit.response_text == "The Thames."
        hit = await cache.check(question, caller=caller, context_fingerprint="paris")
        assert hit is not None and hit.response_text == "The Seine."
        assert await cache.check(question, caller=caller, context_fingerprint="rome") is None


# ═══════════════════════════════════════════════════════════════════════
# Stats, similarity and age
# ═══════════════════════════════════════════════════════════════════════


class TestHitAccounting:
    @pytest.mark.asyncio
    async def test_context_mismatch_is_one_miss_not_a_hit_and_a_miss(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider())
        caller = CallerContext(user_id="alice")
        await cache.store("q", "r", {}, caller=caller, context_fingerprint="paris")

        assert await cache.check("q", caller=caller, context_fingerprint="london") is None
        stats = await cache.stats()
        assert (stats.hits, stats.misses) == (0, 1)

    @pytest.mark.asyncio
    async def test_hit_is_counted_once(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider())
        caller = CallerContext(user_id="alice")
        await cache.store("q", "r", {}, caller=caller)

        assert await cache.check("q", caller=caller) is not None
        stats = await cache.stats()
        assert (stats.hits, stats.misses) == (1, 0)

    @pytest.mark.asyncio
    async def test_hit_carries_similarity_and_age(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider())
        caller = CallerContext(user_id="alice")
        await cache.store("q", "r", {}, caller=caller)

        hit = await cache.check("q", caller=caller)
        assert hit is not None
        assert hit.similarity == pytest.approx(1.0, abs=1e-4)
        assert 0 <= hit.age < 5

    @pytest.mark.asyncio
    async def test_backend_returns_a_copy_with_the_score(self):
        backend = InMemoryCacheBackend()
        entry = _make_entry(embedding=[1.0] + [0.0] * 383)
        await backend.store("user:a", entry)

        found = await backend.search("user:a", entry.embedding, 0.9)
        assert found is not None and found is not entry
        assert found.similarity == pytest.approx(1.0, abs=1e-4)
        assert entry.similarity is None


# ═══════════════════════════════════════════════════════════════════════
# Write tools and invalidation
# ═══════════════════════════════════════════════════════════════════════


class TestWriteTools:
    def test_annotations_decide_by_default(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider())
        assert not cache.is_write_tool("count", {"readOnlyHint": True})
        assert cache.is_write_tool("open", {"readOnlyHint": False})
        # The MCP default for readOnlyHint is false: unannotated tools may write
        assert cache.is_write_tool("open", None)
        assert cache.is_write_tool("open", {"destructiveHint": False})

    def test_lists_override_annotations(self):
        cache = SemanticCache(
            embedding=FakeEmbeddingProvider(),
            write_tools=["count_*"],
            read_only_tools=["lookup_*", "count_tickets"],
        )
        assert cache.is_write_tool("count_tickets", {"readOnlyHint": True})  # write wins
        assert not cache.is_write_tool("lookup_order", None)
        assert cache.is_write_tool("lookupx", None)

    def test_glob_is_not_a_regex(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider(), read_only_tools=["get.item"])
        assert not cache.is_write_tool("get.item", None)
        assert cache.is_write_tool("getXitem", None)

    def test_promptise_meta_tool_is_read_only(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider())
        assert not cache.is_write_tool("request_more_tools", None)

    @pytest.mark.asyncio
    async def test_response_from_before_a_write_is_not_stored(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider())
        caller = CallerContext(user_id="alice")
        generation = cache.write_generation(caller)
        # A concurrent request writes while this one is still running
        await cache.invalidate_for_write("open_ticket", caller=caller)
        await cache.store("count", "2 tickets", {}, caller=caller, write_generation=generation)
        assert (await cache.stats()).stores == 0

        # A request that started after the write stores normally
        await cache.store(
            "count", "3 tickets", {}, caller=caller, write_generation=cache.write_generation(caller)
        )
        assert (await cache.stats()).stores == 1

    @pytest.mark.asyncio
    async def test_write_generation_is_per_scope(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider())
        alice, bob = CallerContext(user_id="alice"), CallerContext(user_id="bob")
        generation = cache.write_generation(bob)
        await cache.invalidate_for_write("open_ticket", caller=alice)
        await cache.store("q", "r", {}, caller=bob, write_generation=generation)
        assert (await cache.stats()).stores == 1

    @pytest.mark.asyncio
    async def test_invalidate_on_write_false_keeps_entries(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider(), invalidate_on_write=False)
        caller = CallerContext(user_id="alice")
        await cache.store("q", "r", {}, caller=caller)
        await cache.invalidate_for_write("open_ticket", caller=caller)
        assert await cache.check("q", caller=caller) is not None
        assert cache.write_generation(caller) == 0


class TestToolCallWatcher:
    @staticmethod
    def _watcher(cache: SemanticCache, caller: CallerContext):
        from promptise.cache import ToolCallWatcher

        annotations = {"count": {"readOnlyHint": True}, "open": {"readOnlyHint": False}}
        return ToolCallWatcher(cache, caller, annotations)

    @pytest.mark.asyncio
    async def test_write_tool_evicts_when_it_finishes(self):
        from uuid import uuid4

        cache = SemanticCache(embedding=FakeEmbeddingProvider())
        caller = CallerContext(user_id="alice")
        await cache.store("q", "r", {}, caller=caller)
        watcher = self._watcher(cache, caller)

        read, write = uuid4(), uuid4()
        await watcher.on_tool_start({"name": "count"}, "{}", run_id=read)
        await watcher.on_tool_end("2", run_id=read)
        assert await cache.check("q", caller=caller) is not None

        await watcher.on_tool_start({"name": "open"}, "{}", run_id=write)
        await watcher.on_tool_error(RuntimeError("half done"), run_id=write)
        assert await cache.check("q", caller=caller) is None
        assert await watcher.settle([]) is False

    @pytest.mark.asyncio
    async def test_settle_policy(self):
        from uuid import uuid4

        caller = CallerContext(user_id="alice")
        plain = SemanticCache(embedding=FakeEmbeddingProvider())
        assert await self._watcher(plain, caller).settle([]) is True  # no tools
        assert await self._watcher(plain, caller).settle(["count"]) is False  # tool turn

        tool_turns = SemanticCache(embedding=FakeEmbeddingProvider(), cache_tool_turns=True)
        watcher = self._watcher(tool_turns, caller)
        run = uuid4()
        await watcher.on_tool_start({"name": "count"}, "{}", run_id=run)
        await watcher.on_tool_end("2", run_id=run)
        assert await watcher.settle(["count"]) is True
        assert await self._watcher(tool_turns, caller).settle(["count", "open"]) is False

    @pytest.mark.asyncio
    async def test_write_seen_only_in_output_still_evicts(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider())
        caller = CallerContext(user_id="alice")
        await cache.store("q", "r", {}, caller=caller)
        assert await self._watcher(cache, caller).settle(["open"]) is False
        assert await cache.check("q", caller=caller) is None


class TestTurnHelpers:
    def test_turn_tool_calls_ignores_earlier_turns(self):
        from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

        from promptise.cache import turn_tool_calls

        def call(name: str) -> AIMessage:
            return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": name}])

        output = {
            "messages": [
                HumanMessage(content="open a ticket"),
                call("open_ticket"),
                ToolMessage(content="T-1", tool_call_id="open_ticket"),
                AIMessage(content="Opened T-1."),
                HumanMessage(content="how many?"),
                call("count"),
                ToolMessage(content="1", tool_call_id="count"),
                AIMessage(content="One."),
            ]
        }
        assert turn_tool_calls(output) == ["count"]

    def test_served_model_reads_the_final_answer(self):
        from langchain_core.messages import AIMessage

        from promptise.cache import served_model

        answer = AIMessage(content="hi", response_metadata={"fallback_model": "backup"})
        assert served_model({"messages": [answer]}) == "backup"
        assert served_model({"messages": [AIMessage(content="hi")]}) is None


# ═══════════════════════════════════════════════════════════════════════
# Dependencies
# ═══════════════════════════════════════════════════════════════════════


class TestDependencies:
    @staticmethod
    def _hide(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
        import importlib.util

        real = importlib.util.find_spec
        monkeypatch.setattr(
            importlib.util,
            "find_spec",
            lambda name, *a, **k: None if name in names else real(name, *a, **k),
        )

    def test_missing_sentence_transformers_is_named(self, monkeypatch):
        self._hide(monkeypatch, "sentence_transformers")
        with pytest.raises(ImportError, match="pip install sentence-transformers"):
            SemanticCache().check_dependencies()

    def test_api_embeddings_do_not_need_sentence_transformers(self, monkeypatch):
        from promptise.cache import OpenAIEmbeddingProvider

        self._hide(monkeypatch, "sentence_transformers")
        SemanticCache(embedding=OpenAIEmbeddingProvider(api_key="k")).check_dependencies()


# ═══════════════════════════════════════════════════════════════════════
# per_session scope
# ═══════════════════════════════════════════════════════════════════════


class TestPerSessionScope:
    @pytest.mark.asyncio
    async def test_same_session_id_does_not_cross_users(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider(), scope="per_session")
        alice = CallerContext(user_id="alice", metadata={"session_id": "1"})
        bob = CallerContext(user_id="bob", metadata={"session_id": "1"})
        await cache.store("What is my balance?", "Alice has $9,120.", {}, caller=alice)

        assert await cache.check("What is my balance?", caller=bob) is None
        assert await cache.check("What is my balance?", caller=alice) is not None

    @pytest.mark.asyncio
    async def test_session_without_a_user_is_not_cached(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider(), scope="per_session")
        anonymous = CallerContext(user_id=None, metadata={"session_id": "1"})
        await cache.store("q", "r", {}, caller=anonymous)
        assert (await cache.stats()).stores == 0

    @pytest.mark.asyncio
    async def test_sessions_of_one_user_stay_apart(self):
        cache = SemanticCache(embedding=FakeEmbeddingProvider(), scope="per_session")
        one = CallerContext(user_id="alice", metadata={"session_id": "1"})
        two = CallerContext(user_id="alice", metadata={"session_id": "2"})
        await cache.store("q", "r", {}, caller=one)
        assert await cache.check("q", caller=two) is None
        assert await cache.check("q", caller=one) is not None
