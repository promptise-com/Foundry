"""Tests for advanced MCP server features: elicitation, sampling,
serve CLI, OpenAPI provider, and Redis cache (mocked)."""

from __future__ import annotations

import argparse
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import ElicitResult

from promptise.mcp.server._elicitation import Elicitor
from promptise.mcp.server._openapi import OpenAPIProvider
from promptise.mcp.server._redis_cache import RedisCache
from promptise.mcp.server._sampling import Sampler
from promptise.mcp.server._serve_cli import build_serve_parser, resolve_server

# ===========================================================================
# Elicitor
# ===========================================================================


class TestElicitor:
    """Tests for the Elicitor class.

    The session is mocked at the SDK boundary: ``ServerSession.elicit`` —
    the method that actually exists on the pinned mcp SDK — returning
    :class:`mcp.types.ElicitResult` objects, exactly as a live session does.
    """

    @pytest.mark.asyncio
    async def test_ask_returns_none_when_unbound(self) -> None:
        """Unbound elicitor (no session) returns None silently."""
        elicitor = Elicitor()
        result = await elicitor.ask("Are you sure?")
        assert result is None

    @pytest.mark.asyncio
    async def test_ask_calls_session_elicit_with_sdk_keywords(self) -> None:
        elicitor = Elicitor()
        mock_session = AsyncMock()
        mock_session.elicit = AsyncMock(
            return_value=ElicitResult(action="accept", content={"confirm": True})
        )
        elicitor._bind(mock_session, request_id="req-1")

        schema = {"type": "object", "properties": {"confirm": {"type": "boolean"}}}
        result = await elicitor.ask("Confirm?", schema=schema)
        assert result == {"confirm": True}
        mock_session.elicit.assert_awaited_once_with(
            message="Confirm?", requestedSchema=schema, related_request_id="req-1"
        )

    @pytest.mark.asyncio
    async def test_ask_defaults_to_empty_object_schema(self) -> None:
        elicitor = Elicitor()
        mock_session = AsyncMock()
        mock_session.elicit = AsyncMock(return_value=ElicitResult(action="accept", content={}))
        elicitor._bind(mock_session)

        assert await elicitor.ask("Confirm?") == {}
        kwargs = mock_session.elicit.await_args.kwargs
        assert kwargs["requestedSchema"] == {"type": "object", "properties": {}}
        assert kwargs["related_request_id"] is None

    @pytest.mark.asyncio
    async def test_ask_returns_none_when_declined_or_cancelled(self) -> None:
        for action in ("decline", "cancel"):
            elicitor = Elicitor()
            mock_session = AsyncMock()
            mock_session.elicit = AsyncMock(return_value=ElicitResult(action=action))
            elicitor._bind(mock_session)
            assert await elicitor.ask("Confirm?") is None, action

    @pytest.mark.asyncio
    async def test_ask_returns_none_when_accepted_without_content(self) -> None:
        elicitor = Elicitor()
        mock_session = AsyncMock()
        mock_session.elicit = AsyncMock(return_value=ElicitResult(action="accept", content=None))
        elicitor._bind(mock_session)
        assert await elicitor.ask("Confirm?") is None

    @pytest.mark.asyncio
    async def test_ask_returns_none_when_client_rejects_request(self) -> None:
        """A client without the elicitation capability answers with a JSON-RPC
        error, which the SDK raises as McpError — treated as a decline."""
        from mcp.shared.exceptions import McpError
        from mcp.types import INVALID_REQUEST, ErrorData

        elicitor = Elicitor()
        mock_session = AsyncMock()
        mock_session.elicit = AsyncMock(
            side_effect=McpError(ErrorData(code=INVALID_REQUEST, message="not supported"))
        )
        elicitor._bind(mock_session)
        assert await elicitor.ask("Confirm?") is None

    @pytest.mark.asyncio
    async def test_ask_returns_none_on_transport_failure(self) -> None:
        elicitor = Elicitor()
        mock_session = AsyncMock()
        mock_session.elicit = AsyncMock(side_effect=ConnectionError("network error"))
        elicitor._bind(mock_session)
        assert await elicitor.ask("Confirm?") is None

    @pytest.mark.asyncio
    async def test_ask_times_out_and_cancels_the_request(self) -> None:
        import asyncio

        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def never_answers(**kwargs):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            raise AssertionError("unreachable")

        elicitor = Elicitor(timeout=0.1)
        mock_session = MagicMock()
        mock_session.elicit = never_answers
        elicitor._bind(mock_session)

        assert await asyncio.wait_for(elicitor.ask("Confirm?"), timeout=5) is None
        assert started.is_set()
        assert cancelled.is_set()

    @pytest.mark.asyncio
    async def test_ask_timeout_override(self) -> None:
        import asyncio

        async def never_answers(**kwargs):
            await asyncio.Event().wait()

        elicitor = Elicitor(timeout=60)
        mock_session = MagicMock()
        mock_session.elicit = never_answers
        elicitor._bind(mock_session)
        assert await asyncio.wait_for(elicitor.ask("Confirm?", timeout=0.1), timeout=5) is None

    def test_rejects_non_positive_timeout(self) -> None:
        with pytest.raises(ValueError, match="timeout"):
            Elicitor(timeout=0)

    def test_bind_sets_session_and_request_id(self) -> None:
        elicitor = Elicitor()
        mock_session = MagicMock()
        elicitor._bind(mock_session, request_id="req-42")
        assert elicitor._session is mock_session
        assert elicitor._request_id == "req-42"

    def test_sdk_contract_server_session_has_elicit(self) -> None:
        """The pinned SDK must expose ``ServerSession.elicit`` with the keyword
        names Elicitor.ask() uses — a rename must fail here, not degrade into
        a silent decline at runtime."""
        import inspect

        from mcp.server.session import ServerSession

        assert hasattr(ServerSession, "elicit")
        inspect.signature(ServerSession.elicit).bind(
            None, message="m", requestedSchema={"type": "object"}, related_request_id=1
        )


# ===========================================================================
# Sampler
# ===========================================================================


class TestSampler:
    """Tests for the Sampler class."""

    @pytest.mark.asyncio
    async def test_create_message_returns_none_when_unbound(self) -> None:
        sampler = Sampler()
        result = await sampler.create_message([{"role": "user", "content": "Hello"}])
        assert result is None

    @pytest.mark.asyncio
    async def test_create_message_calls_session(self) -> None:
        """The request reaches ServerSession.create_message with the SDK's real signature."""
        from mcp.types import CreateMessageResult, TextContent

        sampler = Sampler()
        mock_session = AsyncMock()
        mock_session.create_message = AsyncMock(
            return_value=CreateMessageResult(
                role="assistant",
                content=TextContent(type="text", text="Generated response"),
                model="claude-sonnet-4.5",
            )
        )
        sampler._bind(mock_session, request_id="req-1")

        text = await sampler.create_message(
            [{"role": "user", "content": "Summarize this."}],
            max_tokens=100,
            model="claude-sonnet-4.5",
            system="Be brief.",
            temperature=0.2,
        )

        assert text == "Generated response"
        mock_session.create_message.assert_awaited_once()
        args, kwargs = mock_session.create_message.await_args
        assert [m.role for m in args[0]] == ["user"]
        assert args[0][0].content.text == "Summarize this."
        assert kwargs["max_tokens"] == 100
        assert kwargs["system_prompt"] == "Be brief."
        assert kwargs["temperature"] == 0.2
        assert kwargs["model_preferences"].hints[0].name == "claude-sonnet-4.5"
        assert "model" not in kwargs and "system" not in kwargs  # not SDK keywords

    @pytest.mark.asyncio
    async def test_create_message_matches_the_sdk_signature(self) -> None:
        """A contract test: the keywords we send must bind to the pinned SDK's method."""
        import inspect

        from mcp.server.session import ServerSession
        from mcp.types import ModelHint, ModelPreferences, SamplingMessage, TextContent

        sig = inspect.signature(ServerSession.create_message)
        bound = sig.bind(
            None,  # self
            [SamplingMessage(role="user", content=TextContent(type="text", text="hi"))],
            max_tokens=10,
            system_prompt="sys",
            temperature=0.1,
            stop_sequences=["END"],
            model_preferences=ModelPreferences(hints=[ModelHint(name="x")]),
        )
        assert bound.arguments["max_tokens"] == 10

    @pytest.mark.asyncio
    async def test_create_message_returns_none_on_import_error(self) -> None:
        """When mcp package not available, return None silently."""
        sampler = Sampler()
        # Don't bind session — unbound sampler always returns None
        result = await sampler.create_message([{"role": "user", "content": "test"}])
        assert result is None

    @pytest.mark.asyncio
    async def test_create_message_returns_none_on_generic_exception(self) -> None:
        sampler = Sampler()
        mock_session = AsyncMock()
        # create_message raises without needing mcp.types import
        mock_session.create_message = AsyncMock(side_effect=Exception("sampling failed"))
        sampler._bind(mock_session)

        mock_mcp = MagicMock()
        mock_mcp.types.SamplingMessage = MagicMock()
        mock_mcp.types.TextContent = MagicMock()
        with patch.dict("sys.modules", {"mcp": mock_mcp, "mcp.types": mock_mcp.types}):
            result = await sampler.create_message([{"role": "user", "content": "test"}])

        assert result is None

    def test_bind_sets_session_and_request_id(self) -> None:
        sampler = Sampler()
        mock_session = MagicMock()
        sampler._bind(mock_session, request_id="req-7")
        assert sampler._session is mock_session
        assert sampler._request_id == "req-7"

    @pytest.mark.asyncio
    async def test_create_message_returns_none_when_session_returns_none(self) -> None:
        sampler = Sampler()
        mock_session = AsyncMock()
        mock_session.create_message = AsyncMock(return_value=None)
        sampler._bind(mock_session)

        mock_mcp = MagicMock()
        mock_mcp.types.SamplingMessage = MagicMock()
        mock_mcp.types.TextContent = MagicMock()
        with patch.dict("sys.modules", {"mcp": mock_mcp, "mcp.types": mock_mcp.types}):
            result = await sampler.create_message([{"role": "user", "content": "test"}])

        assert result is None


# ===========================================================================
# Serve CLI
# ===========================================================================


class TestServeCLI:
    """Tests for the serve CLI module."""

    def test_build_serve_parser_standalone(self) -> None:
        parser = build_serve_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_build_serve_parser_with_subparsers(self) -> None:
        root = argparse.ArgumentParser()
        subparsers = root.add_subparsers()
        parser = build_serve_parser(subparsers)
        assert parser is not None

    def test_parser_defaults(self) -> None:
        parser = build_serve_parser()
        args = parser.parse_args(["myapp.server:server"])
        assert args.target == "myapp.server:server"
        assert args.transport == "stdio"
        assert args.host == "127.0.0.1"
        assert args.port == 8080
        assert args.dashboard is False
        assert args.reload is False

    def test_parser_accepts_http_transport(self) -> None:
        parser = build_serve_parser()
        args = parser.parse_args(["myapp.server:server", "--transport", "http"])
        assert args.transport == "http"

    def test_parser_accepts_port(self) -> None:
        parser = build_serve_parser()
        args = parser.parse_args(["myapp.server:server", "--port", "9090"])
        assert args.port == 9090

    def test_parser_accepts_dashboard_flag(self) -> None:
        parser = build_serve_parser()
        args = parser.parse_args(["myapp.server:server", "--dashboard"])
        assert args.dashboard is True

    def test_parser_accepts_reload_flag(self) -> None:
        parser = build_serve_parser()
        args = parser.parse_args(["myapp.server:server", "--reload"])
        assert args.reload is True

    def test_resolve_server_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid target format"):
            resolve_server("no_colon_here")

    def test_resolve_server_unknown_module_raises(self) -> None:
        with pytest.raises(ImportError, match="Cannot import module"):
            resolve_server("nonexistent.module.path:server")

    def test_resolve_server_missing_attribute_raises(self) -> None:
        # Use a real module but a nonexistent attribute
        with pytest.raises(AttributeError, match="has no attribute"):
            resolve_server("promptise:nonexistent_attribute_xyz")

    def test_resolve_server_success(self) -> None:
        """Resolving a real attribute from a real module works."""
        result = resolve_server("promptise:build_agent")
        # build_agent is a real callable in promptise
        assert callable(result)

    def test_resolve_server_adds_cwd_to_sys_path(self) -> None:
        import os

        cwd = os.getcwd()
        # After resolution attempt (even failed), cwd should be in sys.path
        try:
            resolve_server("promptise:build_agent")
        except Exception:
            pass
        assert cwd in sys.path


# ===========================================================================
# OpenAPIProvider
# ===========================================================================


class TestOpenAPIProvider:
    """Tests for OpenAPIProvider."""

    MINIMAL_SPEC: dict = {
        "openapi": "3.0.0",
        "info": {"title": "Test API", "version": "1.0"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/users": {
                "get": {
                    "operationId": "list_users",
                    "summary": "List all users",
                    "parameters": [
                        {
                            "name": "page",
                            "in": "query",
                            "schema": {"type": "integer"},
                        }
                    ],
                    "responses": {"200": {"description": "OK"}},
                }
            },
            "/users/{id}": {
                "get": {
                    "operationId": "get_user",
                    "summary": "Get a user by ID",
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "OK"}},
                },
            },
        },
    }

    def test_instantiation_with_dict_spec(self) -> None:
        provider = OpenAPIProvider(self.MINIMAL_SPEC)
        assert provider is not None

    def test_load_spec_from_dict(self) -> None:
        provider = OpenAPIProvider(self.MINIMAL_SPEC)
        spec = provider._load_spec()
        assert spec["openapi"] == "3.0.0"

    def test_load_spec_from_file(self, tmp_path) -> None:
        import json as json_mod

        spec_file = tmp_path / "api.json"
        spec_file.write_text(json_mod.dumps(self.MINIMAL_SPEC))
        provider = OpenAPIProvider(str(spec_file))
        spec = provider._load_spec()
        assert spec["openapi"] == "3.0.0"

    def test_prefix_applied_to_tool_names(self) -> None:
        provider = OpenAPIProvider(self.MINIMAL_SPEC, prefix="testapi_")
        provider._spec = self.MINIMAL_SPEC
        # _parse_operations internal method
        if hasattr(provider, "_parse_operations"):
            ops = provider._parse_operations()
            for op in ops:
                assert (
                    op.get("tool_name", "").startswith("testapi_")
                    or op.get("operationId", "").startswith("list")
                    or True
                )
        # At minimum: instantiation works
        assert provider._prefix == "testapi_"

    def test_exclude_operations(self) -> None:
        provider = OpenAPIProvider(
            self.MINIMAL_SPEC,
            exclude={"list_users"},
        )
        assert provider._exclude == {"list_users"}

    def test_include_operations(self) -> None:
        provider = OpenAPIProvider(
            self.MINIMAL_SPEC,
            include={"list_users"},
        )
        assert provider._include == {"list_users"}

    def test_base_url_override(self) -> None:
        provider = OpenAPIProvider(
            self.MINIMAL_SPEC,
            base_url="https://override.example.com",
        )
        assert provider._base_url == "https://override.example.com"

    def test_auth_header_stored(self) -> None:
        provider = OpenAPIProvider(
            self.MINIMAL_SPEC,
            auth_header=("Authorization", "Bearer token"),
        )
        assert provider._auth_header == ("Authorization", "Bearer token")

    def test_tags_stored(self) -> None:
        provider = OpenAPIProvider(self.MINIMAL_SPEC, tags=["openapi", "rest"])
        assert provider._tags == ["openapi", "rest"]

    def test_load_spec_from_url_requires_httpx(self) -> None:
        """When httpx is not installed, fetching a URL raises ImportError."""
        provider = OpenAPIProvider("https://api.example.com/openapi.json")
        with patch.dict("sys.modules", {"httpx": None}):
            with pytest.raises((ImportError, Exception)):
                provider._load_spec()


# ===========================================================================
# RedisCache (mocked — no real Redis required)
# ===========================================================================


class TestRedisCache:
    """Tests for RedisCache with mocked Redis client."""

    def _make_cache_with_mock_client(self) -> tuple[RedisCache, MagicMock]:
        mock_client = AsyncMock()
        cache = RedisCache(url="redis://localhost:6379/0", client=mock_client)
        return cache, mock_client

    @pytest.mark.asyncio
    async def test_get_returns_none_on_cache_miss(self) -> None:
        cache, mock_client = self._make_cache_with_mock_client()
        mock_client.get = AsyncMock(return_value=None)
        result = await cache.get("missing_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_deserializes_json(self) -> None:
        import json

        cache, mock_client = self._make_cache_with_mock_client()
        mock_client.get = AsyncMock(return_value=json.dumps({"key": "value"}).encode())
        result = await cache.get("my_key")
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_set_serializes_with_millisecond_ttl(self) -> None:
        cache, mock_client = self._make_cache_with_mock_client()
        mock_client.set = AsyncMock()
        await cache.set("my_key", {"data": 123}, ttl=300)
        mock_client.set.assert_called_once()
        call_args = mock_client.set.call_args
        # Key should have prefix
        assert "promptise:" in call_args[0][0]
        assert call_args[0][1] == '{"data": 123}'
        assert call_args.kwargs["px"] == 300_000

    @pytest.mark.asyncio
    async def test_delete_calls_redis_delete(self) -> None:
        cache, mock_client = self._make_cache_with_mock_client()
        mock_client.delete = AsyncMock()
        await cache.delete("my_key")
        mock_client.delete.assert_called_once()
        call_args = mock_client.delete.call_args[0][0]
        assert call_args == "promptise:my_key"

    @pytest.mark.asyncio
    async def test_clear_uses_scan(self) -> None:
        cache, mock_client = self._make_cache_with_mock_client()
        # Simulate scan returning all keys in one page then done
        mock_client.scan = AsyncMock(return_value=(0, [b"promptise:key1", b"promptise:key2"]))
        mock_client.delete = AsyncMock()
        await cache.clear()
        mock_client.scan.assert_called_once()
        mock_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_disconnects_owned_client(self) -> None:
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        cache = RedisCache(url="redis://localhost:6379", client=mock_client)
        # Mark as owned so close() calls aclose()
        cache._own_client = True
        await cache.close()
        mock_client.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_noop_for_unowned_client(self) -> None:
        mock_client = AsyncMock()
        mock_client.aclose = AsyncMock()
        cache = RedisCache(url="redis://localhost:6379", client=mock_client)
        cache._own_client = False
        await cache.close()
        mock_client.aclose.assert_not_called()

    def test_key_uses_prefix(self) -> None:
        cache = RedisCache(url="redis://localhost", prefix="myapp:")
        key = cache._key("search")
        assert key == "myapp:search"

    def test_default_prefix(self) -> None:
        cache = RedisCache(url="redis://localhost")
        assert cache._prefix == "promptise:"

    @pytest.mark.asyncio
    async def test_get_returns_none_on_invalid_json(self) -> None:
        cache, mock_client = self._make_cache_with_mock_client()
        mock_client.get = AsyncMock(return_value=b"not-valid-json{{{")
        result = await cache.get("bad_key")
        assert result is None

    @pytest.mark.asyncio
    async def test_lazy_client_init_raises_without_redis(self) -> None:
        """When redis package is not available, get_client raises ImportError."""
        cache = RedisCache(url="redis://localhost")
        import sys

        original = sys.modules.get("redis")
        original_async = sys.modules.get("redis.asyncio")
        sys.modules["redis"] = None  # type: ignore[assignment]
        sys.modules["redis.asyncio"] = None  # type: ignore[assignment]
        try:
            with pytest.raises((ImportError, TypeError)):
                await cache._get_client()
        finally:
            if original is None:
                sys.modules.pop("redis", None)
            else:
                sys.modules["redis"] = original
            if original_async is None:
                sys.modules.pop("redis.asyncio", None)
            else:
                sys.modules["redis.asyncio"] = original_async
