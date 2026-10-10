"""Mem0Provider against fake Mem0 clients with the real ``mem0.Memory`` signatures.

``Memory.add()`` took ``data=`` only in early 0.1 releases; 0.1.118, 1.x and
2.x take the text as ``messages``.  2.0 also moved the entity ids of
``search()`` / ``get_all()`` into ``filters=`` and renamed ``limit`` to
``top_k`` — and rejects the old keyword arguments.  The previous tests used a
bare ``MagicMock``, which accepts any keyword, so ``add(data=...)`` passed
them while failing on every real Mem0 release.

The fakes below copy the keyword-only signatures of mem0ai 1.0.11 (identical
to 0.1.118 apart from ``rerank``) and mem0ai 2.2.1, and keep a real in-memory
store so search, isolation, delete and purge are exercised end to end.  When
mem0ai is installed, ``TestAgainstInstalledMem0`` also binds every call
Promptise makes to the installed release's real signature.
"""

from __future__ import annotations

import inspect
import re
import sys
import types
from typing import Any
from uuid import uuid4

import pytest

from promptise.memory import Mem0Provider, MemoryIsolationError, MemoryScope

# ---------------------------------------------------------------------------
# Fake Mem0 clients
# ---------------------------------------------------------------------------


class _Store:
    """Shared state of a fake Mem0 client: id → entry dict."""

    def __init__(self) -> None:
        self.entries: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def add(self, text: str, user_id: str | None, agent_id: str | None, metadata: Any) -> dict:
        mid = str(uuid4())
        self.entries[mid] = {
            "id": mid,
            "memory": text,
            "user_id": user_id,
            "agent_id": agent_id,
            "metadata": metadata,
        }
        return {"results": [{"id": mid, "memory": text, "event": "ADD"}]}

    def matching(self, entities: dict[str, Any]) -> list[dict[str, Any]]:
        return [e for e in self.entries.values() if all(e.get(k) == v for k, v in entities.items())]

    def search(self, query: str, entities: dict[str, Any], limit: int) -> dict:
        words = set(re.findall(r"\w+", query.lower()))
        scored = []
        for e in self.matching(entities):
            overlap = len(words & set(re.findall(r"\w+", e["memory"].lower())))
            if overlap:
                scored.append({**e, "score": min(1.0, overlap / max(len(words), 1))})
        scored.sort(key=lambda e: e["score"], reverse=True)
        return {"results": scored[:limit]}


def _require_entity(user_id: Any, agent_id: Any, run_id: Any) -> None:
    if not (user_id or agent_id or run_id):
        raise ValueError("One of the filters: user_id, agent_id or run_id is required!")


class FakeMem0V1:
    """mem0ai 0.1.118 / 1.x ``Memory``: entity ids and ``limit`` are keyword arguments."""

    def __init__(self) -> None:
        self.store = _Store()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FakeMem0V1:  # noqa: ARG003
        return cls()

    def add(
        self,
        messages: Any,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        infer: bool = True,
        memory_type: str | None = None,
        prompt: str | None = None,
    ) -> dict:
        _require_entity(user_id, agent_id, run_id)
        self.store.calls.append(("add", {"messages": messages, "user_id": user_id}))
        return self.store.add(messages, user_id, agent_id, metadata)

    def search(
        self,
        query: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
        filters: dict[str, Any] | None = None,
        threshold: float | None = None,
        rerank: bool = True,
    ) -> dict:
        _require_entity(user_id, agent_id, run_id)
        entities = {k: v for k, v in (("user_id", user_id), ("agent_id", agent_id)) if v}
        self.store.calls.append(("search", {"limit": limit, **entities}))
        return self.store.search(query, entities, limit)

    def get_all(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        filters: dict[str, Any] | None = None,
        limit: int = 100,
    ) -> dict:
        _require_entity(user_id, agent_id, run_id)
        entities = {k: v for k, v in (("user_id", user_id), ("agent_id", agent_id)) if v}
        return {"results": self.store.matching(entities)[:limit]}

    def get(self, memory_id: str) -> dict | None:
        return self.store.entries.get(memory_id)

    def delete(self, memory_id: str) -> dict:
        self.store.entries.pop(memory_id, None)
        return {"message": "Memory deleted successfully!"}

    def delete_all(
        self,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
    ) -> dict:
        _require_entity(user_id, agent_id, run_id)
        for e in self.store.matching({"user_id": user_id} if user_id else {"agent_id": agent_id}):
            self.store.entries.pop(e["id"])
        return {"message": "Memories deleted successfully!"}

    def reset(self) -> None:
        self.store.entries.clear()


class FakeMem0V2(FakeMem0V1):
    """mem0ai 2.x ``Memory``: ``search``/``get_all`` take ``filters=`` and ``top_k=``."""

    def add(  # type: ignore[override]
        self,
        messages: Any,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        timestamp: Any | None = None,
        expiration_date: Any | None = None,
        infer: bool = True,
        memory_type: str | None = None,
        prompt: str | None = None,
    ) -> dict:
        return super().add(messages, user_id=user_id, agent_id=agent_id, metadata=metadata)

    def search(  # type: ignore[override]
        self,
        query: str,
        *,
        top_k: int = 20,
        filters: dict[str, Any] | None = None,
        threshold: float = 0.1,
        rerank: bool = False,
        explain: bool = False,
        reference_date: Any | None = None,
        show_expired: bool = False,
        **kwargs: Any,
    ) -> dict:
        _reject_top_level_entity_params(kwargs, "search")
        filters = dict(filters or {})
        if not any(k in filters for k in ("user_id", "agent_id", "run_id")):
            raise ValueError("filters must contain at least one of: user_id, agent_id, run_id.")
        self.store.calls.append(("search", {"top_k": top_k, **filters}))
        return self.store.search(query, filters, top_k)

    def get_all(  # type: ignore[override]
        self,
        *,
        filters: dict[str, Any] | None = None,
        top_k: int = 20,
        show_expired: bool = False,
        **kwargs: Any,
    ) -> dict:
        _reject_top_level_entity_params(kwargs, "get_all")
        filters = dict(filters or {})
        if not any(k in filters for k in ("user_id", "agent_id", "run_id")):
            raise ValueError("filters must contain at least one of: user_id, agent_id, run_id.")
        return {"results": self.store.matching(filters)[:top_k]}


def _reject_top_level_entity_params(kwargs: dict[str, Any], method: str) -> None:
    """mem0ai 2.x raises when entity ids are passed outside ``filters``."""
    bad = {"user_id", "agent_id", "run_id", "limit"} & set(kwargs)
    if bad:
        raise ValueError(
            f"Top-level entity parameters {sorted(bad)} are not supported in {method}(). "
            "Use filters={...} instead."
        )
    if kwargs:
        raise TypeError(f"{method}() got unexpected keyword arguments {sorted(kwargs)}")


@pytest.fixture(params=[FakeMem0V1, FakeMem0V2], ids=["mem0-0.1-1.x", "mem0-2.x"])
def fake_mem0(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> type:
    """Install a fake ``mem0`` module whose ``Memory`` is one of the fakes."""
    module = types.ModuleType("mem0")
    module.Memory = request.param  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mem0", module)
    return request.param


# ---------------------------------------------------------------------------
# The guide's mem0_check.py flow, against both API generations
# ---------------------------------------------------------------------------


class TestMem0Flow:
    @pytest.mark.asyncio
    async def test_mem0_check_flow(self, fake_mem0: type) -> None:
        """The repro from the memory guide: add, search, and a second user sees nothing."""
        memory = Mem0Provider(scope=MemoryScope.PER_USER, config={"vector_store": {}})
        await memory.add(
            "I'm Mara. I'm vegetarian, and I never take flights that leave before 9 am.",
            user_id="mara",
        )
        results = await memory.search("is she vegetarian", user_id="mara")
        assert [r.content for r in results] == [
            "I'm Mara. I'm vegetarian, and I never take flights that leave before 9 am."
        ]
        assert 0.0 < results[0].score <= 1.0
        assert await memory.search("is she vegetarian", user_id="bob") == []

    @pytest.mark.asyncio
    async def test_add_passes_text_as_messages(self, fake_mem0: type) -> None:
        memory = Mem0Provider(user_id="u1")
        await memory.add("likes tea", metadata={"source": "test"})
        name, call = memory._client.store.calls[-1]
        assert name == "add"
        assert call == {"messages": "likes tea", "user_id": "u1"}

    @pytest.mark.asyncio
    async def test_add_returns_mem0_id(self, fake_mem0: type) -> None:
        memory = Mem0Provider(user_id="u1")
        mid = await memory.add("likes tea")
        assert mid in memory._client.store.entries

    @pytest.mark.asyncio
    async def test_search_uses_the_api_of_the_installed_generation(self, fake_mem0: type) -> None:
        memory = Mem0Provider(user_id="u1", agent_id="travel")
        await memory.add("likes tea")
        await memory.search("tea", limit=3)
        _, call = memory._client.store.calls[-1]
        if fake_mem0 is FakeMem0V2:
            assert memory._filters_api is True
            assert call == {"top_k": 3, "user_id": "u1", "agent_id": "travel"}
        else:
            assert memory._filters_api is False
            assert call == {"limit": 3, "user_id": "u1", "agent_id": "travel"}

    @pytest.mark.asyncio
    async def test_shared_scope_uses_default_user(self, fake_mem0: type) -> None:
        memory = Mem0Provider(user_id="org")
        await memory.add("office is in Lisbon")
        assert [r.content for r in await memory.search("Lisbon office")] == ["office is in Lisbon"]
        assert memory._client.store.calls[0][1]["user_id"] == "org"

    @pytest.mark.asyncio
    async def test_per_user_scope_requires_user_id(self, fake_mem0: type) -> None:
        memory = Mem0Provider(scope=MemoryScope.PER_USER)
        with pytest.raises(MemoryIsolationError):
            await memory.add("x")
        with pytest.raises(MemoryIsolationError):
            await memory.search("x")

    @pytest.mark.asyncio
    async def test_delete_refuses_another_users_entry(self, fake_mem0: type) -> None:
        memory = Mem0Provider(scope=MemoryScope.PER_USER)
        mid = await memory.add("mara's secret", user_id="mara")
        assert await memory.delete(mid, user_id="bob") is False
        assert mid in memory._client.store.entries
        assert await memory.delete(mid, user_id="mara") is True
        assert mid not in memory._client.store.entries

    @pytest.mark.asyncio
    async def test_purge_user_counts_and_removes_only_that_user(self, fake_mem0: type) -> None:
        memory = Mem0Provider(scope=MemoryScope.PER_USER)
        await memory.add("one", user_id="mara")
        await memory.add("two", user_id="mara")
        await memory.add("three", user_id="bob")
        assert await memory.purge_user("mara") == 2
        assert [e["user_id"] for e in memory._client.store.entries.values()] == ["bob"]

    @pytest.mark.asyncio
    async def test_list_entries_uses_the_api_of_the_installed_generation(
        self, fake_mem0: type
    ) -> None:
        """The adaptive strategy lists its lessons with list_entries().

        Its first version called ``get_all(user_id=...)`` and retried with
        ``filters=`` on ``TypeError``, but mem0ai 2.x raises ``ValueError``,
        so every list came back empty.
        """
        memory = Mem0Provider(scope=MemoryScope.PER_USER)
        await memory.add("prefer the EU endpoint", user_id="mara", metadata={"type": "strategy"})
        await memory.add("likes tea", user_id="mara")
        await memory.add("bob's lesson", user_id="bob", metadata={"type": "strategy"})
        entries = await memory.list_entries(user_id="mara", metadata={"type": "strategy"})
        assert [e.content for e in entries] == ["prefer the EU endpoint"]
        assert entries[0].metadata["type"] == "strategy"

    @pytest.mark.asyncio
    async def test_search_errors_propagate(self, fake_mem0: type, caplog) -> None:
        """An incompatible client or a broken store must not look like 'no memories'."""
        memory = Mem0Provider(user_id="u1")

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise ConnectionError("vector store down")

        memory._client.search = boom
        with pytest.raises(ConnectionError):
            await memory.search("anything")
        assert "Mem0Provider.search failed" in caplog.text


# ---------------------------------------------------------------------------
# Guard: Promptise's calls bind to the installed mem0ai's real signatures
# ---------------------------------------------------------------------------


class TestAgainstInstalledMem0:
    """Bind the exact calls Promptise makes to the installed ``mem0.Memory``.

    Skipped when mem0ai is not installed.  Never constructs a real client
    (that would need an LLM and a vector store); it only inspects signatures.
    """

    @pytest.fixture
    def real_memory_cls(self) -> type:
        mem0 = pytest.importorskip("mem0")
        return mem0.Memory

    def _provider_for(self, real_memory_cls: type) -> Mem0Provider:
        provider = object.__new__(Mem0Provider)
        provider._user_id = "u1"
        provider._agent_id = "a1"
        provider._closed = False
        provider.scope = MemoryScope.SHARED

        class _SignatureOnly:
            search = real_memory_cls.search

        from promptise.memory import _mem0_uses_filters_api

        # Detection reads the bound method's signature, as on a real instance.
        provider._filters_api = _mem0_uses_filters_api(_SignatureOnly())
        return provider

    def test_search_and_get_all_bind(self, real_memory_cls: type) -> None:
        provider = self._provider_for(real_memory_cls)
        kwargs = provider._entity_kwargs("u1", 5)
        search = inspect.signature(real_memory_cls.search)
        search.bind(None, "query", **kwargs)
        get_all = inspect.signature(real_memory_cls.get_all)
        get_all.bind(None, **provider._entity_kwargs("u1", 10_000))
        named = set(search.parameters)
        if provider._filters_api:
            # 2.x: entity ids must not be top-level keywords.
            assert not {"user_id", "agent_id", "limit"} & set(kwargs)
            assert "filters" in named and "top_k" in named
        else:
            assert {"user_id", "agent_id", "limit"} <= named

    def test_add_binds(self, real_memory_cls: type) -> None:
        sig = inspect.signature(real_memory_cls.add)
        sig.bind(None, messages="text", user_id="u1", agent_id="a1", metadata={"k": "v"})
        assert "data" not in sig.parameters

    def test_delete_get_and_delete_all_bind(self, real_memory_cls: type) -> None:
        inspect.signature(real_memory_cls.get).bind(None, "mid")
        inspect.signature(real_memory_cls.delete).bind(None, "mid")
        inspect.signature(real_memory_cls.delete_all).bind(None, user_id="u1")

    def test_fake_matches_installed_generation(self, real_memory_cls: type) -> None:
        """The fake for the installed generation accepts every real ``search`` keyword."""
        provider = self._provider_for(real_memory_cls)
        fake = FakeMem0V2 if provider._filters_api else FakeMem0V1
        real_params = {
            n
            for n, p in inspect.signature(real_memory_cls.search).parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        }
        fake_params = set(inspect.signature(fake.search).parameters)
        missing = {"top_k", "filters", "user_id", "limit"} & real_params - fake_params
        assert not missing
