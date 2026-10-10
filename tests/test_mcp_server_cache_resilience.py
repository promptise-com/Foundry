"""Caching and resilience regressions (guide 24).

* ``CacheMiddleware`` keyed on ``ctx.state["arguments"]`` (never set), so
  every call to a tool got the first cached answer; its documented
  ``tdef.cache = False`` opt-out did not exist.
* ``@cached`` on a tool with ``ctx: RequestContext`` never hit (the context,
  with a random request id, was part of the key).
* The auto-added per-tool limiter sat inside the circuit breaker, so
  "at capacity" refusals opened the circuit.
* An open circuit reached the model as a non-retryable ``INTERNAL_ERROR``.
* The breaker counted every error — three bad lookups paused the tool.
* Half-open let every waiting call through instead of one probe.
* ``@server.tool(timeout=...)`` did nothing without ``TimeoutMiddleware``.
* ``TestClient`` ignored ``max_concurrent`` and leaked exception text.
* ``InMemoryCache`` evicted the oldest *insert*, not the least recently
  used; ``RedisCache`` rejected sub-second TTLs; rate-limit suggestions
  said "Wait 0 seconds"; concurrent identical calls each hit upstream.
"""

from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock

import pytest

from promptise.mcp.server import (
    APIKeyAuth,
    AuthMiddleware,
    CacheMiddleware,
    CircuitBreakerMiddleware,
    CircuitOpenError,
    CircuitState,
    ConcurrencyLimitError,
    Depends,
    InMemoryCache,
    MCPRouter,
    MCPServer,
    PerToolConcurrencyLimiter,
    RateLimitError,
    RedisCache,
    RequestContext,
    TestClient,
    TimeoutMiddleware,
    ToolError,
    cached,
    get_context,
    is_upstream_failure,
)

KEYS = {"acme-key": {"client_id": "acme"}, "globex-key": {"client_id": "globex"}}


def _authed(name: str) -> MCPServer:
    server = MCPServer(name, require_auth=True)
    server.add_middleware(AuthMiddleware(APIKeyAuth(keys=KEYS)))
    return server


async def _call(client: TestClient, tool: str, args: dict | None = None):
    """The tool's result: decoded JSON, or the raw text."""
    text = (await client.call_tool(tool, args or {}))[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _code(result) -> str | None:
    return result["error"]["code"] if isinstance(result, dict) and "error" in result else None


async def _live_call(server: MCPServer, name: str, args: dict | None = None):
    """Call through the real lowlevel handler (not TestClient)."""
    import mcp.types as t

    ll = server._build_lowlevel_server()
    handler = ll.request_handlers[t.CallToolRequest]
    req = t.CallToolRequest(
        method="tools/call", params=t.CallToolRequestParams(name=name, arguments=args or {})
    )
    text = (await handler(req)).root.content[0].text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


# =====================================================================
# CacheMiddleware
# =====================================================================


class TestCacheMiddleware:
    async def test_arguments_and_caller_are_part_of_the_key(self):
        server = _authed("inventory")
        server.add_middleware(CacheMiddleware(ttl=60))
        runs = {"n": 0}

        @server.tool()
        async def stock(sku: str) -> str:
            runs["n"] += 1
            return f"stock of {sku} for {get_context().client_id}"

        acme = TestClient(server, meta={"x-api-key": "acme-key"})
        globex = TestClient(server, meta={"x-api-key": "globex-key"})
        assert await _call(acme, "stock", {"sku": "SKU-7"}) == "stock of SKU-7 for acme"
        assert await _call(globex, "stock", {"sku": "SKU-9"}) == "stock of SKU-9 for globex"
        assert await _call(globex, "stock", {"sku": "SKU-7"}) == "stock of SKU-7 for globex"
        assert await _call(acme, "stock", {"sku": "SKU-7"}) == "stock of SKU-7 for acme"
        assert runs["n"] == 3  # the last call was a hit

    async def test_arguments_are_part_of_the_key_on_the_live_path(self):
        server = MCPServer("weather")
        server.add_middleware(CacheMiddleware(ttl=60))

        @server.tool()
        async def forecast(city: str) -> str:
            return f"weather for {city}"

        assert await _live_call(server, "forecast", {"city": "Paris"}) == "weather for Paris"
        assert await _live_call(server, "forecast", {"city": "Tokyo"}) == "weather for Tokyo"

    async def test_cache_false_opts_a_tool_out(self):
        server = MCPServer("opt-out")
        server.add_middleware(CacheMiddleware(ttl=60))
        runs = {"n": 0}

        @server.tool(cache=False)
        async def next_ticket() -> int:
            runs["n"] += 1
            return runs["n"]

        client = TestClient(server)
        assert [await _call(client, "next_ticket") for _ in range(3)] == [1, 2, 3]

    async def test_destructive_tools_are_never_cached(self):
        server = MCPServer("destructive")
        server.add_middleware(CacheMiddleware(ttl=60))
        runs = {"n": 0}

        @server.tool(destructive_hint=True)
        async def delete_record(record_id: str) -> int:
            runs["n"] += 1
            return runs["n"]

        client = TestClient(server)
        await _call(client, "delete_record", {"record_id": "a"})
        await _call(client, "delete_record", {"record_id": "a"})
        assert runs["n"] == 2

    async def test_router_tools_accept_cache_false(self):
        server = MCPServer("router-opt-out")
        server.add_middleware(CacheMiddleware(ttl=60))
        router = MCPRouter()
        runs = {"n": 0}

        @router.tool(cache=False)
        async def tick() -> int:
            runs["n"] += 1
            return runs["n"]

        server.include_router(router)
        client = TestClient(server)
        assert [await _call(client, "tick") for _ in range(2)] == [1, 2]

    async def test_other_tools_are_still_cached(self):
        server = MCPServer("still-cached")
        server.add_middleware(CacheMiddleware(ttl=60))
        runs = {"n": 0}

        @server.tool()
        async def lookup(q: str) -> int:
            runs["n"] += 1
            return runs["n"]

        client = TestClient(server)
        assert await _call(client, "lookup", {"q": "x"}) == 1
        assert await _call(client, "lookup", {"q": "x"}) == 1


# =====================================================================
# @cached
# =====================================================================


class TestCachedDecorator:
    async def test_ctx_parameter_does_not_defeat_the_cache(self):
        server = MCPServer("with-ctx")
        runs = {"n": 0}

        @server.tool()
        @cached(ttl=60, backend=InMemoryCache(cleanup_interval=0))
        async def stock(ctx: RequestContext, sku: str) -> str:
            runs["n"] += 1
            return f"{sku} run {runs['n']}"

        client = TestClient(server)
        assert [await _call(client, "stock", {"sku": "S"}) for _ in range(3)] == ["S run 1"] * 3

    async def test_depends_parameter_is_not_part_of_the_key(self):
        server = MCPServer("with-depends")
        runs = {"n": 0}

        class Conn:
            pass

        def get_conn() -> Conn:
            return Conn()  # a fresh object per request

        @server.tool()
        @cached(ttl=60, backend=InMemoryCache(cleanup_interval=0))
        async def query(q: str, conn: Conn = Depends(get_conn)) -> str:
            runs["n"] += 1
            return q

        client = TestClient(server)
        await _call(client, "query", {"q": "a"})
        await _call(client, "query", {"q": "a"})
        assert runs["n"] == 1

    async def test_results_are_per_caller_by_default(self):
        server = _authed("prices")

        @server.tool()
        @cached(ttl=60, backend=InMemoryCache(cleanup_interval=0))
        async def my_price(sku: str) -> str:
            return f"{sku} price for {get_context().client_id}"

        acme = TestClient(server, meta={"x-api-key": "acme-key"})
        globex = TestClient(server, meta={"x-api-key": "globex-key"})
        assert await _call(acme, "my_price", {"sku": "S"}) == "S price for acme"
        assert await _call(globex, "my_price", {"sku": "S"}) == "S price for globex"


class TestCoalescing:
    async def test_identical_concurrent_calls_share_one_upstream_call(self):
        server = MCPServer("stampede")
        runs = {"n": 0}

        @server.tool()
        @cached(ttl=60, backend=InMemoryCache(cleanup_interval=0))
        async def get_stock(sku: str) -> dict:
            runs["n"] += 1
            await asyncio.sleep(0.05)
            return {"sku": sku}

        client = TestClient(server)
        results = await asyncio.gather(
            *(_call(client, "get_stock", {"sku": "S"}) for _ in range(5))
        )
        assert results == [{"sku": "S"}] * 5
        assert runs["n"] == 1

    async def test_different_arguments_are_not_coalesced(self):
        server = MCPServer("distinct")
        runs = {"n": 0}

        @server.tool()
        @cached(ttl=60, backend=InMemoryCache(cleanup_interval=0))
        async def get_stock(sku: str) -> str:
            runs["n"] += 1
            await asyncio.sleep(0.02)
            return sku

        client = TestClient(server)
        assert await asyncio.gather(
            _call(client, "get_stock", {"sku": "A"}), _call(client, "get_stock", {"sku": "B"})
        ) == ["A", "B"]
        assert runs["n"] == 2

    async def test_concurrent_callers_from_different_tenants_get_their_own_result(self):
        """Coalescing keys on the scoped cache key — never shares across callers."""
        server = _authed("coalesce-scope")
        release = asyncio.Event()
        runs = {"n": 0}

        @server.tool()
        @cached(ttl=60, backend=InMemoryCache(cleanup_interval=0))
        async def balance(account: str) -> str:
            runs["n"] += 1
            caller = get_context().client_id
            await release.wait()
            return f"{account} for {caller}"

        acme = TestClient(server, meta={"x-api-key": "acme-key"})
        globex = TestClient(server, meta={"x-api-key": "globex-key"})
        pending = asyncio.gather(
            _call(acme, "balance", {"account": "main"}),
            _call(globex, "balance", {"account": "main"}),
        )
        await asyncio.sleep(0.01)
        release.set()
        assert await pending == ["main for acme", "main for globex"]
        assert runs["n"] == 2

    async def test_coalesce_false_calls_upstream_each_time(self):
        server = MCPServer("no-coalesce")
        runs = {"n": 0}

        @server.tool()
        @cached(ttl=60, backend=InMemoryCache(cleanup_interval=0), coalesce=False)
        async def get_stock(sku: str) -> str:
            runs["n"] += 1
            await asyncio.sleep(0.02)
            return sku

        client = TestClient(server)
        await asyncio.gather(*(_call(client, "get_stock", {"sku": "S"}) for _ in range(3)))
        assert runs["n"] == 3

    async def test_concurrent_callers_share_an_error_which_is_not_cached(self):
        runs = {"n": 0}

        @cached(ttl=60, backend=InMemoryCache(cleanup_interval=0))
        async def flaky(sku: str) -> str:
            runs["n"] += 1
            await asyncio.sleep(0.02)
            if runs["n"] == 1:
                raise ConnectionError("upstream down")
            return sku

        results = await asyncio.gather(*(flaky(sku="S") for _ in range(3)), return_exceptions=True)
        assert all(isinstance(r, ConnectionError) for r in results)
        assert runs["n"] == 1
        assert await flaky(sku="S") == "S"  # the failure was not cached
        assert runs["n"] == 2

    async def test_a_waiter_takes_over_when_the_first_call_is_cancelled(self):
        started = asyncio.Event()
        runs = {"n": 0}

        @cached(ttl=60, backend=InMemoryCache(cleanup_interval=0))
        async def slow(sku: str) -> str:
            runs["n"] += 1
            started.set()
            await asyncio.sleep(0.05)
            return sku

        leader = asyncio.create_task(slow(sku="S"))
        await started.wait()
        waiter = asyncio.create_task(slow(sku="S"))
        await asyncio.sleep(0)
        leader.cancel()
        assert await waiter == "S"
        assert runs["n"] == 2


# =====================================================================
# Backends
# =====================================================================


class TestInMemoryCacheLRU:
    async def test_evicts_the_least_recently_used_entry(self):
        cache = InMemoryCache(max_size=2, cleanup_interval=0)
        await cache.set("a", 1, 60)
        await cache.set("b", 2, 60)
        assert await cache.get("a") == 1  # "a" is now the most recently used
        await cache.set("c", 3, 60)
        assert await cache.get("b") is None
        assert await cache.get("a") == 1
        assert await cache.get("c") == 3

    async def test_overwriting_a_key_at_capacity_evicts_nothing(self):
        cache = InMemoryCache(max_size=2, cleanup_interval=0)
        await cache.set("a", 1, 60)
        await cache.set("b", 2, 60)
        await cache.set("a", 10, 60)
        assert (await cache.get("a"), await cache.get("b")) == (10, 2)


class TestRedisSubSecondTTL:
    async def test_sub_second_ttl_uses_milliseconds(self):
        client = AsyncMock()
        await RedisCache(client=client).set("k", {"v": 1}, 0.5)
        client.set.assert_awaited_once()
        assert client.set.call_args.kwargs["px"] == 500
        client.setex.assert_not_called()

    async def test_non_positive_ttl_is_not_stored(self):
        client = AsyncMock()
        await RedisCache(client=client).set("k", 1, 0)
        client.set.assert_not_called()


# =====================================================================
# Circuit breaker
# =====================================================================


def _breaker_server(threshold: int = 2, recovery: float = 60.0, **kw):
    server = MCPServer("breaker")
    breaker = CircuitBreakerMiddleware(failure_threshold=threshold, recovery_timeout=recovery, **kw)
    server.add_middleware(breaker)
    return server, breaker


class TestBreakerClassification:
    @pytest.mark.parametrize(
        ("exc", "counts"),
        [
            (RuntimeError("boom"), True),
            (ConnectionError("refused"), True),
            (ToolError("No product BAD"), False),
            (ToolError("upstream 503", retryable=True), True),
            (ToolError("slow", code="TIMEOUT", retryable=True), True),
            (RateLimitError(retry_after=1), False),
            (ConcurrencyLimitError("busy"), False),
            (CircuitOpenError("t", 1), False),
            (asyncio.CancelledError(), False),
        ],
    )
    def test_default_classifier(self, exc, counts):
        assert is_upstream_failure(exc) is counts

    async def test_bad_input_errors_never_open_the_circuit(self):
        server, breaker = _breaker_server(threshold=2)

        @server.tool()
        async def lookup(sku: str) -> str:
            raise ToolError(f"No product {sku}")

        client = TestClient(server)
        for _ in range(5):
            assert _code(await _call(client, "lookup", {"sku": "BAD"})) == "TOOL_ERROR"
        assert breaker.get_state("lookup") == CircuitState.CLOSED

    async def test_upstream_errors_open_the_circuit(self):
        server, breaker = _breaker_server(threshold=2)

        @server.tool()
        async def lookup(sku: str) -> str:
            raise ConnectionError("db down")

        client = TestClient(server)
        await _call(client, "lookup", {"sku": "A"})
        await _call(client, "lookup", {"sku": "A"})
        assert breaker.get_state("lookup") == CircuitState.OPEN

    async def test_custom_classifier(self):
        server, breaker = _breaker_server(threshold=1, is_failure=lambda exc: True)

        @server.tool()
        async def lookup(sku: str) -> str:
            raise ToolError("No product")

        await _call(TestClient(server), "lookup", {"sku": "A"})
        assert breaker.get_state("lookup") == CircuitState.OPEN


class TestCircuitOpenResponse:
    async def _open(self, server):
        @server.tool()
        async def boom() -> str:
            raise RuntimeError("upstream exploded")

    async def test_testclient_returns_retryable_circuit_open(self):
        server, _ = _breaker_server(threshold=1)
        await self._open(server)
        client = TestClient(server)
        first = await _call(client, "boom")
        assert first["error"] == {
            "code": "INTERNAL_ERROR",
            "message": "An internal error occurred.",
            "retryable": False,
        }
        err = (await _call(client, "boom"))["error"]
        assert err["code"] == "CIRCUIT_OPEN"
        assert err["retryable"] is True
        assert 59 < err["details"]["retry_after_seconds"] <= 60
        assert "boom" in err["suggestion"]

    async def test_live_path_returns_retryable_circuit_open(self):
        server, _ = _breaker_server(threshold=1)
        await self._open(server)
        await _live_call(server, "boom")
        err = (await _live_call(server, "boom"))["error"]
        assert (err["code"], err["retryable"]) == ("CIRCUIT_OPEN", True)
        assert err["details"]["retry_after_seconds"] > 0

    async def test_a_registered_handler_still_customises_it(self):
        server, _ = _breaker_server(threshold=1)
        await self._open(server)

        @server.exception_handler(CircuitOpenError)
        async def paused(ctx, exc):
            return ToolError("Paused, try later", code="PAUSED", retryable=True)

        client = TestClient(server)
        await _call(client, "boom")
        assert _code(await _call(client, "boom")) == "PAUSED"
        assert _code(await _live_call(server, "boom")) == "PAUSED"

    async def test_a_catch_all_handler_does_not_swallow_structured_errors(self):
        server = MCPServer("catch-all")

        @server.exception_handler(Exception)
        async def anything(ctx, exc):
            return ToolError("swallowed", code="SWALLOWED")

        @server.tool()
        async def lookup() -> str:
            raise ToolError("No product", code="NOT_FOUND")

        assert _code(await _call(TestClient(server), "lookup")) == "NOT_FOUND"
        assert _code(await _live_call(server, "lookup")) == "NOT_FOUND"


class TestHalfOpen:
    async def _tripped(self, recovery: float = 0.05):
        server, breaker = _breaker_server(threshold=1, recovery=recovery)
        state = {"fail": True, "reached": 0}
        release = asyncio.Event()

        @server.tool()
        async def upstream(bad: bool = False) -> str:
            state["reached"] += 1
            if bad:
                raise ToolError("bad input")
            if state["fail"]:
                raise RuntimeError("down")
            await release.wait()
            return "ok"

        client = TestClient(server)
        await _call(client, "upstream")
        assert breaker.get_state("upstream") == CircuitState.OPEN
        await asyncio.sleep(recovery + 0.02)
        state.update(fail=False, reached=0)
        return client, breaker, state, release

    async def test_exactly_one_probe_reaches_the_tool(self):
        client, breaker, state, release = await self._tripped()
        calls = [asyncio.create_task(_call(client, "upstream")) for _ in range(5)]
        await asyncio.sleep(0.02)
        assert state["reached"] == 1
        release.set()
        results = await asyncio.gather(*calls)
        assert results.count("ok") == 1
        assert sorted(_code(r) for r in results if r != "ok") == ["CIRCUIT_OPEN"] * 4
        assert breaker.get_state("upstream") == CircuitState.CLOSED

    async def test_a_failed_probe_reopens(self):
        client, breaker, state, release = await self._tripped()
        state["fail"] = True
        await _call(client, "upstream")
        assert breaker.get_state("upstream") == CircuitState.OPEN
        assert _code(await _call(client, "upstream")) == "CIRCUIT_OPEN"

    async def test_a_probe_without_a_verdict_lets_the_next_call_probe(self):
        client, breaker, state, release = await self._tripped()
        assert _code(await _call(client, "upstream", {"bad": True})) == "TOOL_ERROR"
        assert breaker.get_state("upstream") == CircuitState.HALF_OPEN
        release.set()
        assert await _call(client, "upstream") == "ok"
        assert breaker.get_state("upstream") == CircuitState.CLOSED

    async def test_a_cancelled_probe_releases_the_slot(self):
        client, breaker, state, release = await self._tripped()
        probe = asyncio.create_task(_call(client, "upstream"))
        await asyncio.sleep(0.01)
        probe.cancel()
        with pytest.raises(asyncio.CancelledError):
            await probe
        release.set()
        assert await _call(client, "upstream") == "ok"


# =====================================================================
# Concurrency limits
# =====================================================================


class TestConcurrencyLimits:
    async def test_capacity_refusals_never_open_the_circuit(self):
        server, breaker = _breaker_server(threshold=3)
        release = asyncio.Event()

        @server.tool(max_concurrent=2)
        async def slow(n: int) -> str:
            await release.wait()
            return f"done {n}"

        client = TestClient(server)
        calls = [asyncio.create_task(_call(client, "slow", {"n": i})) for i in range(8)]
        await asyncio.sleep(0.02)
        release.set()
        results = await asyncio.gather(*calls)
        assert sum(isinstance(r, str) for r in results) == 2
        assert [_code(r) for r in results if not isinstance(r, str)] == [
            "CONCURRENCY_LIMIT_EXCEEDED"
        ] * 6
        assert breaker.get_state("slow") == CircuitState.CLOSED
        assert await _call(client, "slow", {"n": 99}) == "done 99"

    def test_auto_added_limiter_sits_outside_the_breaker(self):
        server, breaker = _breaker_server()

        @server.tool(max_concurrent=2)
        async def slow() -> str:
            return "ok"

        server._build_lowlevel_server()
        kinds = [type(m) for m in server._middlewares]
        assert kinds == [PerToolConcurrencyLimiter, CircuitBreakerMiddleware]

    def test_an_installed_limiter_is_not_duplicated(self):
        server, _ = _breaker_server()
        server.add_middleware(PerToolConcurrencyLimiter())

        @server.tool(max_concurrent=2)
        async def slow() -> str:
            return "ok"

        server._build_lowlevel_server()
        assert sum(isinstance(m, PerToolConcurrencyLimiter) for m in server._middlewares) == 1

    async def test_testclient_enforces_max_concurrent(self):
        server = MCPServer("busy")
        release = asyncio.Event()

        @server.tool(max_concurrent=1)
        async def busy() -> str:
            await release.wait()
            return "done"

        client = TestClient(server)
        calls = [asyncio.create_task(_call(client, "busy")) for _ in range(3)]
        await asyncio.sleep(0.02)
        release.set()
        results = await asyncio.gather(*calls)
        assert results.count("done") == 1
        err = next(r for r in results if r != "done")["error"]
        assert err["code"] == "CONCURRENCY_LIMIT_EXCEEDED"
        assert err["retryable"] is True

    def test_concurrency_error_is_still_a_rate_limit_error(self):
        exc = ConcurrencyLimitError("busy")
        assert isinstance(exc, RateLimitError)
        assert exc.code == "CONCURRENCY_LIMIT_EXCEEDED"


class TestRateLimitSuggestion:
    @pytest.mark.parametrize(
        ("wait", "text"),
        [(0.3, "Wait 1 second"), (1.0, "Wait 1 second"), (42.2, "Wait 43 seconds")],
    )
    def test_never_says_wait_zero_seconds(self, wait, text):
        exc = RateLimitError(retry_after=wait)
        assert exc.suggestion.startswith(text)
        assert exc.details["retry_after_seconds"] == pytest.approx(wait)


# =====================================================================
# Timeouts
# =====================================================================


class TestToolTimeout:
    async def test_enforced_without_timeout_middleware(self):
        server = MCPServer("timeouts")

        @server.tool(timeout=0.05)
        async def hang() -> str:
            await asyncio.sleep(5)
            return "finished"

        start = time.perf_counter()
        err = (await _call(TestClient(server), "hang"))["error"]
        assert time.perf_counter() - start < 1
        assert (err["code"], err["retryable"]) == ("TIMEOUT", True)
        assert "simpler input" not in err["suggestion"]

    async def test_enforced_on_the_live_path(self):
        server = MCPServer("timeouts-live")

        @server.tool(timeout=0.05)
        async def hang() -> str:
            await asyncio.sleep(5)
            return "finished"

        assert _code(await _live_call(server, "hang")) == "TIMEOUT"

    async def test_fast_tools_are_unaffected(self):
        server = MCPServer("fast")
        server.add_middleware(TimeoutMiddleware(default_timeout=5))

        @server.tool(timeout=1)
        async def quick() -> str:
            return "ok"

        assert await _call(TestClient(server), "quick") == "ok"

    async def test_timeouts_count_as_breaker_failures(self):
        server, breaker = _breaker_server(threshold=1)

        @server.tool(timeout=0.02)
        async def hang() -> str:
            await asyncio.sleep(5)
            return "finished"

        await _call(TestClient(server), "hang")
        assert breaker.get_state("hang") == CircuitState.OPEN
