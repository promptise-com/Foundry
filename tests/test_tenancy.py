"""First-class tenancy — the isolation invariant, proven surface by surface.

The claim under test: two tenants with the SAME ``user_id`` can never see
each other's data.  Agent side, every isolation surface keys on
``CallerContext.isolation_key`` (``tenant::user``); server side,
``ClientContext.tenant_id`` enters rate-limit keys, audit entries, and the
tenant guards.
"""

from __future__ import annotations

import hashlib
import random
from unittest.mock import AsyncMock, MagicMock

import pytest

from promptise.agent import CallerContext
from promptise.mcp.server import (
    AuthMiddleware,
    HasTenant,
    MCPServer,
    RequireTenant,
    TestClient,
)
from promptise.mcp.server._auth import APIKeyAuth, _build_client_context_from_jwt
from promptise.mcp.server._context import ClientContext, RequestContext


class FakeEmbeddingProvider:
    """Deterministic embedding for tests (same shape as test_cache.py's)."""

    def __init__(self, dim: int = 384):
        self._dim = dim
        self._cache: dict[str, list[float]] = {}

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results = []
        for text in texts:
            if text not in self._cache:
                seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
                rng = random.Random(seed)
                vec = [rng.gauss(0, 1) for _ in range(self._dim)]
                norm = sum(v * v for v in vec) ** 0.5
                self._cache[text] = [v / norm for v in vec]
            results.append(self._cache[text])
        return results


# ---------------------------------------------------------------------------
# CallerContext.isolation_key — the single derivation
# ---------------------------------------------------------------------------


class TestIsolationKey:
    def test_tenant_qualifies_user(self):
        c = CallerContext(user_id="alice", tenant_id="acme")
        assert c.isolation_key == "acme::alice"

    def test_no_tenant_is_plain_user(self):
        assert CallerContext(user_id="alice").isolation_key == "alice"

    def test_no_user_is_none_even_with_tenant(self):
        assert CallerContext(tenant_id="acme").isolation_key is None

    def test_two_tenants_same_user_differ(self):
        a = CallerContext(user_id="alice", tenant_id="acme")
        b = CallerContext(user_id="alice", tenant_id="globex")
        assert a.isolation_key != b.isolation_key


# ---------------------------------------------------------------------------
# Semantic cache — same user id, different tenants → no cross-tenant hits
# ---------------------------------------------------------------------------


class TestCacheTenantIsolation:
    @pytest.mark.asyncio
    async def test_cross_tenant_miss_same_tenant_hit(self):
        from promptise.cache import SemanticCache

        cache = SemanticCache(embedding=FakeEmbeddingProvider(), similarity_threshold=0.9)
        acme_alice = CallerContext(user_id="alice", tenant_id="acme")
        globex_alice = CallerContext(user_id="alice", tenant_id="globex")

        await cache.store("what is our revenue?", "acme numbers", {}, caller=acme_alice)

        # Same tenant, same user → hit
        hit = await cache.check("what is our revenue?", caller=acme_alice)
        assert hit is not None and hit.response_text == "acme numbers"

        # Different tenant, SAME user id → structurally impossible to hit
        assert await cache.check("what is our revenue?", caller=globex_alice) is None

    @pytest.mark.asyncio
    async def test_purge_targets_exactly_the_tenant_scope(self):
        from promptise.cache import SemanticCache

        cache = SemanticCache(embedding=FakeEmbeddingProvider(), similarity_threshold=0.9)
        acme_alice = CallerContext(user_id="alice", tenant_id="acme")
        globex_alice = CallerContext(user_id="alice", tenant_id="globex")
        await cache.store("q", "acme answer", {}, caller=acme_alice)
        await cache.store("q", "globex answer", {}, caller=globex_alice)

        removed = await cache.purge_user("alice", tenant_id="acme")
        assert removed == 1
        assert await cache.check("q", caller=acme_alice) is None
        # The other tenant's entry survives
        globex_hit = await cache.check("q", caller=globex_alice)
        assert globex_hit is not None and globex_hit.response_text == "globex answer"

    def test_scope_key_derivation_is_injective_and_disjoint(self):
        from promptise.cache import SemanticCache

        sid = SemanticCache._scoped_user_id
        # Untenanted keys are unchanged (backward compatible) and colon-free
        assert sid(None, "alice") == "alice"
        assert ":" not in sid(None, "alice")
        # Deterministic
        assert sid("acme", "alice") == sid("acme", "alice")
        # INJECTIVE — the __ collisions the review found are gone:
        assert sid("acme", "corp__alice") != sid("acme__corp", "alice")
        assert sid("a:b", "x") != sid("a_b", "x")
        # DISJOINT namespaces: the tenanted key carries a ':' (colon), which a
        # sanitized untenanted id can never contain — so no untenanted user_id,
        # not even one literally shaped like the tenanted key, can collide.
        assert ":" in sid("acme", "alice")
        assert sid("acme", "alice") != sid(None, sid("acme", "alice"))
        # slash-free either way (safe for Redis namespacing)
        assert "/" not in sid("acme", "alice")

    @pytest.mark.asyncio
    async def test_colliding_ids_do_not_leak_across_tenants(self):
        from promptise.cache import SemanticCache

        cache = SemanticCache(embedding=FakeEmbeddingProvider(), similarity_threshold=0.9)
        # The exact pair the review used to demonstrate the leak
        a = CallerContext(user_id="corp__alice", tenant_id="acme")
        b = CallerContext(user_id="alice", tenant_id="acme__corp")
        await cache.store("q", "A's secret", {}, caller=a)
        assert await cache.check("q", caller=b) is None  # no cross-tenant hit


# ---------------------------------------------------------------------------
# Memory scoping — the wrapper passes the isolation key as user_id
# ---------------------------------------------------------------------------


class TestMemoryTenantIsolation:
    @pytest.mark.asyncio
    async def test_memory_agent_scopes_by_isolation_key(self):
        from promptise.agent import _caller_ctx_var
        from promptise.memory import InMemoryProvider, MemoryAgent, MemoryScope

        provider = InMemoryProvider(scope=MemoryScope.PER_USER)
        memory_agent = MemoryAgent(inner=None, provider=provider)

        # The derivation MemoryAgent uses for all provider calls
        token = _caller_ctx_var.set(CallerContext(user_id="alice", tenant_id="acme"))
        try:
            assert memory_agent._caller_user_id() == "acme::alice"
        finally:
            _caller_ctx_var.reset(token)

        token = _caller_ctx_var.set(CallerContext(user_id="alice", tenant_id="globex"))
        try:
            assert memory_agent._caller_user_id() == "globex::alice"
        finally:
            _caller_ctx_var.reset(token)

        # And the provider enforces isolation on those derived owners:
        # acme::alice's memory is invisible to globex::alice
        await provider.add("acme's secret roadmap", user_id="acme::alice")
        assert await provider.search("roadmap", user_id="acme::alice") != []
        assert await provider.search("roadmap", user_id="globex::alice") == []


# ---------------------------------------------------------------------------
# Conversation ownership — cross-tenant session access is denied
# ---------------------------------------------------------------------------


class TestConversationTenantIsolation:
    @pytest.mark.asyncio
    async def test_same_user_id_other_tenant_denied(self):
        from promptise.agent import PromptiseAgent
        from promptise.conversations import InMemoryConversationStore, SessionAccessDenied

        inner = AsyncMock()
        inner.ainvoke = AsyncMock(return_value={"messages": [MagicMock(content="ok", type="ai")]})
        agent = PromptiseAgent(inner=inner, conversation_store=InMemoryConversationStore())

        acme_alice = CallerContext(user_id="alice", tenant_id="acme")
        globex_alice = CallerContext(user_id="alice", tenant_id="globex")

        await agent.chat("hello", session_id="s1", caller=acme_alice)

        # Identical user_id, different tenant → ownership check denies
        with pytest.raises(SessionAccessDenied):
            await agent.chat("let me in", session_id="s1", caller=globex_alice)

        # The rightful tenant continues fine
        assert await agent.chat("again", session_id="s1", caller=acme_alice) == "ok"


# ---------------------------------------------------------------------------
# Server side — JWT claim, API-key config, guards, rate limits, audit
# ---------------------------------------------------------------------------


class TestServerTenantExtraction:
    def test_jwt_tenant_claim_default_name(self):
        ctx = _build_client_context_from_jwt({"sub": "c1", "tenant_id": "acme"}, "c1")
        assert ctx.tenant_id == "acme"

    def test_jwt_tenant_claim_custom_name(self):
        ctx = _build_client_context_from_jwt({"sub": "c1", "org": "acme"}, "c1", tenant_claim="org")
        assert ctx.tenant_id == "acme"

    def test_non_string_tenant_claim_is_ignored(self):
        ctx = _build_client_context_from_jwt({"sub": "c1", "tenant_id": 42}, "c1")
        assert ctx.tenant_id is None

    @pytest.mark.asyncio
    async def test_api_key_tenant_flows_to_client_context(self):
        from promptise.mcp.server._types import ToolDef

        auth = APIKeyAuth(
            keys={"sk-1": {"client_id": "agent-1", "roles": ["analyst"], "tenant_id": "acme"}}
        )
        middleware = AuthMiddleware(auth)
        tool_def = ToolDef(
            name="t", description="", handler=lambda: None, input_schema={}, auth=True
        )
        ctx = RequestContext(server_name="test", tool_name="t", meta={"x-api-key": "sk-1"})
        ctx.state["tool_def"] = tool_def

        await middleware(ctx, AsyncMock(return_value="r"))
        assert ctx.client.tenant_id == "acme"


class TestTenantGuards:
    def _ctx(self, tenant: str | None) -> RequestContext:
        ctx = RequestContext(server_name="s", tool_name="t")
        ctx.client = ClientContext(client_id="c1", tenant_id=tenant)
        return ctx

    @pytest.mark.asyncio
    async def test_require_tenant(self):
        assert await RequireTenant().check(self._ctx("acme")) is True
        assert await RequireTenant().check(self._ctx(None)) is False

    @pytest.mark.asyncio
    async def test_has_tenant(self):
        guard = HasTenant("acme", "globex")
        assert await guard.check(self._ctx("acme")) is True
        assert await guard.check(self._ctx("initech")) is False
        assert await guard.check(self._ctx(None)) is False
        assert "presented none" in guard.describe_denial(self._ctx(None))
        # A refused tenant is never told which tenants are allowed.
        denial = guard.describe_denial(self._ctx("initech"))
        assert "initech" in denial
        assert "acme" not in denial and "globex" not in denial


class TestRequireTenantServer:
    @pytest.mark.asyncio
    async def test_invariant_denies_tenantless_and_allows_tenanted(self):
        server = MCPServer(name="tenanted", require_tenant=True)
        server.add_middleware(
            AuthMiddleware(
                APIKeyAuth(
                    keys={
                        "sk-acme": {"client_id": "a1", "tenant_id": "acme"},
                        "sk-none": {"client_id": "a2"},  # no tenant configured
                    }
                )
            )
        )

        @server.tool()  # note: no auth=..., no guards — the invariant adds them
        async def whoami(ctx: RequestContext = None) -> str:  # type: ignore[assignment]
            """Return the caller's tenant."""
            return ctx.client.tenant_id or "?"

        client = TestClient(server)

        # Tenanted key passes and sees its tenant
        ok = await client.call_tool("whoami", {}, headers={"x-api-key": "sk-acme"})
        assert ok[0].text == "acme"

        # Authenticated but tenantless → RequireTenant denies
        denied = await client.call_tool("whoami", {}, headers={"x-api-key": "sk-none"})
        assert "tenant" in denied[0].text.lower()
        assert "acme" not in denied[0].text

        # Unauthenticated → denied before the handler
        anon = await client.call_tool("whoami", {})
        assert "acme" not in anon[0].text


class TestTenantRateLimitKeying:
    @pytest.mark.asyncio
    async def test_declared_buckets_do_not_span_tenants(self):
        from types import SimpleNamespace

        from promptise.mcp.server import DeclaredRateLimitMiddleware
        from promptise.mcp.server._errors import RateLimitError

        mw = DeclaredRateLimitMiddleware()
        tdef = SimpleNamespace(rate_limit="1/min")

        async def _next(ctx):
            return "ok"

        def _ctx(tenant: str) -> RequestContext:
            ctx = RequestContext(server_name="s", tool_name="t")
            ctx.state["tool_def"] = tdef
            ctx.client_id = "same-client-id"
            ctx.client = ClientContext(client_id="same-client-id", tenant_id=tenant)
            return ctx

        # acme consumes its only token; acme is limited …
        assert await mw(_ctx("acme"), _next) == "ok"
        with pytest.raises(RateLimitError):
            await mw(_ctx("acme"), _next)
        # … but globex — same client_id string — has its own bucket
        assert await mw(_ctx("globex"), _next) == "ok"


class TestTenantInAudit:
    def test_identity_fields_include_tenant(self):
        from promptise.mcp.server._audit import AuditMiddleware

        ctx = RequestContext(server_name="s", tool_name="t")
        ctx.client = ClientContext(client_id="c1", tenant_id="acme", subject="agent-1")
        fields = AuditMiddleware._identity_fields(ctx)
        assert fields["tenant_id"] == "acme"

    def test_no_tenant_no_field(self):
        from promptise.mcp.server._audit import AuditMiddleware

        ctx = RequestContext(server_name="s", tool_name="t")
        ctx.client = ClientContext(client_id="c1", subject="agent-1")
        assert "tenant_id" not in AuditMiddleware._identity_fields(ctx)


class TestTenantIdValidation:
    def test_colon_in_tenant_rejected(self):
        # Any colon in the tenant is rejected (not just '::') so the
        # ``tenant::user`` join is unambiguous and cannot collide at the
        # separator boundary — e.g. ('a:', 'b') would else equal ('a', ':b').
        with pytest.raises(ValueError, match="':'"):
            CallerContext(user_id="alice", tenant_id="acme::evil")
        with pytest.raises(ValueError, match="':'"):
            CallerContext(user_id="alice", tenant_id="a:")

    def test_boundary_colon_pairs_cannot_collide(self):
        # ('a', ':b') is valid & unique; the colliding ('a:', 'b') is rejected
        assert CallerContext(tenant_id="a", user_id=":b").isolation_key == "a:::b"
        with pytest.raises(ValueError):
            CallerContext(tenant_id="a:", user_id="b")
        # cache derivation is injective even called directly (length-prefixed)
        from promptise.cache import SemanticCache

        assert SemanticCache._scoped_user_id("a", ":b") != SemanticCache._scoped_user_id("a:", "b")

    def test_blank_tenant_rejected(self):
        with pytest.raises(ValueError, match="non-empty"):
            CallerContext(user_id="alice", tenant_id="   ")

    def test_plain_tenant_fine(self):
        assert CallerContext(user_id="alice", tenant_id="acme").tenant_id == "acme"


class TestIsolationKeyInjectivity:
    """The isolation_key map must be injective AND its tenanted keyspace
    disjoint from untenanted — the review found user_id was unvalidated,
    letting an untenanted 'acme::alice' forge tenant acme / user alice."""

    def test_user_id_with_separator_rejected(self):
        # This is the exact forge vector: untenanted user_id == a tenanted key
        with pytest.raises(ValueError, match="separator"):
            CallerContext(user_id="acme::alice")

    def test_boundary_merge_forge_rejected(self):
        # ('a', '::b') would merge with ('a:', ':b') at the separator boundary;
        # rejecting '::' in user_id removes the ambiguous input entirely.
        with pytest.raises(ValueError, match="separator"):
            CallerContext(tenant_id="a", user_id="::b")

    def test_normal_ids_unaffected(self):
        # Colons and SSO-style ids without '::' are still fine
        assert CallerContext(user_id="google:12345", tenant_id="acme").isolation_key == (
            "acme::google:12345"
        )
        assert CallerContext(user_id="auth0|abc").isolation_key == "auth0|abc"

    @pytest.mark.asyncio
    async def test_memory_conversation_no_cross_tenant_forge(self):
        # With user_id '::' rejected, the tenanted keyspace (always contains
        # '::') is disjoint from every untenanted user_id (never can) —
        # conversation ownership can't be forged across the boundary.
        from promptise.agent import PromptiseAgent
        from promptise.conversations import InMemoryConversationStore

        inner = AsyncMock()
        inner.ainvoke = AsyncMock(return_value={"messages": [MagicMock(content="ok", type="ai")]})
        agent = PromptiseAgent(inner=inner, conversation_store=InMemoryConversationStore())
        # tenant acme / user alice owns the session
        await agent.chat(
            "hi", session_id="s1", caller=CallerContext(user_id="alice", tenant_id="acme")
        )
        # The forging caller can no longer even be constructed
        with pytest.raises(ValueError):
            CallerContext(user_id="acme::alice")


class TestCacheNamespaceDisjoint:
    @pytest.mark.asyncio
    async def test_untenanted_lookalike_does_not_hit_tenanted(self):
        from promptise.cache import SemanticCache

        cache = SemanticCache(embedding=FakeEmbeddingProvider(), similarity_threshold=0.9)
        victim = CallerContext(user_id="alice", tenant_id="acme")
        await cache.store("q", "acme secret", {}, caller=victim)

        # Derive the lookalikes from the REAL stored key (not a hand-rolled
        # hash) so the test actually pins the disjointness: the untenanted
        # sanitizer turns the ':' of the real 't:<hash>' key into '_', which
        # is exactly what a colon-free prefix mutation ('t_<hash>') would
        # produce — so setting an untenanted user_id to the real key (and its
        # '.'/'_' variants) catches any prefix that reopens the collision.
        real = SemanticCache._scoped_user_id("acme", "alice")  # 't:<hash>'
        assert real.startswith("t:")
        lookalikes = [real, real.replace("t:", "t_", 1), real.replace("t:", "t.", 1)]
        for lookalike in lookalikes:
            attacker = CallerContext(user_id=lookalike)  # untenanted
            assert await cache.check("q", caller=attacker) is None


class TestUserInstalledRateLimitTenantKeying:
    """The user-installed RateLimitMiddleware (not just the auto-inserted
    DeclaredRateLimitMiddleware) must tenant-qualify its keys — the review
    found this security-relevant path had zero coverage."""

    @pytest.mark.asyncio
    async def test_two_tenants_same_client_id_have_separate_buckets(self):
        from promptise.mcp.server import RateLimitMiddleware, TokenBucketLimiter

        mw = RateLimitMiddleware(TokenBucketLimiter(rate_per_minute=60, burst=1))

        async def _next(ctx):
            return "ok"

        def _ctx(tenant: str) -> RequestContext:
            ctx = RequestContext(server_name="s", tool_name="t")
            ctx.client_id = "same-client-id"
            ctx.client = ClientContext(client_id="same-client-id", tenant_id=tenant)
            return ctx

        # acme exhausts its burst; acme is then limited …
        assert await mw(_ctx("acme"), _next) == "ok"
        from promptise.mcp.server._errors import RateLimitError

        with pytest.raises(RateLimitError):
            await mw(_ctx("acme"), _next)
        # … but globex — same client_id — has its own bucket
        assert await mw(_ctx("globex"), _next) == "ok"

    def test_key_is_tenant_qualified(self):
        from promptise.mcp.server import RateLimitMiddleware

        mw = RateLimitMiddleware()
        a = RequestContext(server_name="s", tool_name="t")
        a.client_id = "c1"
        a.client = ClientContext(client_id="c1", tenant_id="acme")
        b = RequestContext(server_name="s", tool_name="t")
        b.client_id = "c1"
        b.client = ClientContext(client_id="c1", tenant_id="globex")
        assert mw._get_key(a) != mw._get_key(b)
        assert "tenant=acme" in mw._get_key(a)


class TestRateLimitKeyInjectivityWithColons:
    """Server-side tenant_id/client_id come from JWT/API-key config and are
    NOT colon-validated, so the rate-limit key join must be injective even
    for URN-style ids ('org:acme') — else distinct tenants share a bucket."""

    def test_colon_ids_do_not_collide_user_installed(self):
        from promptise.mcp.server import RateLimitMiddleware

        mw = RateLimitMiddleware()

        def key(tenant, client):
            ctx = RequestContext(server_name="s", tool_name="t")
            ctx.client_id = client
            ctx.client = ClientContext(client_id=client, tenant_id=tenant)
            return mw._get_key(ctx)

        # ('a','b:c') and ('a:b','c') would both be 'tenant=a:b:c' under a bare join
        assert key("a", "b:c") != key("a:b", "c")

    @pytest.mark.asyncio
    async def test_colon_ids_do_not_share_bucket_declared(self):
        from promptise.mcp.server import DeclaredRateLimitMiddleware

        mw = DeclaredRateLimitMiddleware()
        from types import SimpleNamespace

        tdef = SimpleNamespace(rate_limit="1/min")

        async def _next(ctx):
            return "ok"

        def ctx(tenant, client):
            c = RequestContext(server_name="s", tool_name="t")
            c.state["tool_def"] = tdef
            c.client_id = client
            c.client = ClientContext(client_id=client, tenant_id=tenant)
            return c

        assert await mw(ctx("a", "b:c"), _next) == "ok"  # tenant a spends its token
        # distinct tenant 'a:b' / client 'c' must NOT be throttled by tenant 'a'
        assert await mw(ctx("a:b", "c"), _next) == "ok"


class TestDeclaredRateLimitUntenantedForge:
    """An untenanted caller whose client_id is literally shaped like a
    tenanted composed key must NOT share the tenant's declared-limit bucket
    (round-5 finding: the untenanted branch was not length-prefixed)."""

    @pytest.mark.asyncio
    async def test_untenanted_lookalike_client_id_isolated(self):
        from types import SimpleNamespace

        from promptise.mcp.server import DeclaredRateLimitMiddleware

        mw = DeclaredRateLimitMiddleware()
        tdef = SimpleNamespace(rate_limit="2/min")

        async def _next(ctx):
            return "ok"

        def ctx(tenant, client):
            c = RequestContext(server_name="s", tool_name="t")
            c.state["tool_def"] = tdef
            c.client_id = client
            c.client = (
                ClientContext(client_id=client, tenant_id=tenant)
                if tenant
                else ClientContext(client_id=client)
            )
            return c

        forge = "8:tenant=a|1:b"  # the composed key for tenant 'a' / client 'b'
        assert await mw(ctx(None, forge), _next) == "ok"
        assert await mw(ctx(None, forge), _next) == "ok"  # attacker drains its own bucket
        # legit tenant 'a'/'b' still has a full, separate bucket
        assert await mw(ctx("a", "b"), _next) == "ok"
