"""Single-server MCP client with token-based authentication.

Wraps the official MCP SDK's transport and session APIs into a clean
async context manager that handles:

- Transport selection (Streamable HTTP, SSE, stdio)
- Bearer token injection (from IdP or server token endpoint)
- API key injection (simple pre-shared secret)
- Custom header injection on every HTTP request
- Proper session lifecycle (initialize → use → close)
- Clear, typed errors when a server refuses the connection
- Transparent re-initialisation when an HTTP server forgets the session
  (a restart or redeploy answers the old ``mcp-session-id`` with ``404``)

The transport and session live in a task owned by the client.  The MCP
SDK's transports run their HTTP traffic in an anyio task group; owning
that task group in a dedicated task means a transport failure (an HTTP
401 during ``initialize``, a dropped connection) is entered, unwound and
reported in one task, and never cancels the caller's task.

The client **never** generates JWTs.  Tokens are obtained externally
(from an Identity Provider or the server's built-in token endpoint)
and passed in via ``bearer_token``, ``api_key``, or ``headers``.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from typing import Any, TypeVar

from mcp.client.session import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ListToolsResult, Tool

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# The MCP SDK's Streamable HTTP client turns an HTTP 404 answer to a request
# into a JSON-RPC error with this message: the server does not know the
# session (it restarted, was redeployed, or the request reached another
# replica), or — during ``initialize`` — the URL is not an MCP endpoint.
_SESSION_TERMINATED = "Session terminated"

_NETWORK_TRANSPORTS = ("http", "streamable-http", "sse")


class _ForgottenSessionDeleteFilter(logging.Filter):
    """Drop the SDK's warning for a ``DELETE`` the server answered ``404``.

    Closing a session the server has already forgotten (a restart, which is
    exactly when the client re-initialises) sends a ``DELETE`` that gets
    ``404``.  The session is gone either way, so the warning is noise.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return record.getMessage() != "Session termination failed: 404"


logging.getLogger("mcp.client.streamable_http").addFilter(_ForgottenSessionDeleteFilter())


class MCPClientError(RuntimeError):
    """Raised when an MCP client operation fails."""


class MCPConnectionRejectedError(MCPClientError):
    """Raised when an HTTP MCP server rejects the session with a 4xx status.

    Typically ``401``/``403`` (missing or wrong ``bearer_token``/``api_key``)
    or ``404`` (wrong endpoint URL).  Retrying with the same configuration
    will fail the same way.

    Attributes:
        status_code: The HTTP status the server answered ``initialize`` with.
        reason: The HTTP reason phrase (e.g. ``"Unauthorized"``).
        url: The endpoint that was contacted.
        server_name: The server's name when connected through
            :class:`~promptise.mcp.client.MCPMultiClient` or ``build_agent``.
    """

    def __init__(
        self,
        *,
        status_code: int,
        reason: str,
        url: str,
        server_name: str | None = None,
    ) -> None:
        self.status_code = status_code
        self.reason = reason
        self.url = url
        self.server_name = server_name
        who = f"Server '{server_name}'" if server_name else f"Server at {url}"
        status = f"{status_code} {reason}" if reason else str(status_code)
        message = f"{who} rejected the connection: {status}."
        if status_code in (401, 403):
            message += " Check the bearer_token/api_key configured for it."
        elif status_code == 404:
            message += f" Check the URL ({url}); Promptise servers serve MCP at /mcp."
        super().__init__(message)

    def for_server(self, server_name: str) -> MCPConnectionRejectedError:
        """Return a copy of this error that names *server_name*."""
        return MCPConnectionRejectedError(
            status_code=self.status_code,
            reason=self.reason,
            url=self.url,
            server_name=server_name,
        )


class _SessionReplaced(Exception):
    """The session a request waited on was closed to open a new one."""


def _is_session_terminated(exc: BaseException) -> bool:
    """True when *exc* is the SDK's report of an HTTP 404 for a request."""
    return isinstance(exc, McpError) and exc.error.message == _SESSION_TERMINATED


def _leaf_exceptions(exc: BaseException) -> list[BaseException]:
    """Flatten (possibly nested) exception groups into their leaf exceptions."""
    nested = getattr(exc, "exceptions", None)
    if isinstance(nested, (list, tuple)):
        leaves: list[BaseException] = []
        for inner in nested:
            leaves.extend(_leaf_exceptions(inner))
        return leaves
    return [exc]


class MCPClient:
    """Production-grade MCP client for a single server.

    Supports HTTP (Streamable HTTP), SSE, and stdio transports with
    Bearer token or API key authentication.

    Args:
        url: Server endpoint URL (for HTTP/SSE).  Mutually exclusive
            with ``command``.
        transport: ``"http"`` (default), ``"sse"``, or ``"stdio"``.
        headers: Extra HTTP headers sent on every request.
        bearer_token: Pre-issued Bearer token.  When provided, an
            ``Authorization: Bearer <token>`` header is injected
            automatically.  Obtain tokens from your Identity Provider
            or the server's token endpoint — the client never
            generates tokens itself.
        api_key: Pre-shared API key.  When provided, an ``x-api-key``
            header is injected automatically.  Use this for simple
            secret-based auth when JWT is overkill.
        command: Executable for stdio transport (e.g. ``"python"``).
        args: Arguments for the stdio command.
        env: Environment variables for the stdio process.
        cwd: Working directory for the stdio subprocess.  When ``None``
            the subprocess inherits the parent process's working directory.
        timeout: HTTP request timeout in seconds.
        auto_reconnect: Re-open the MCP session when an HTTP or SSE server
            loses it (default ``True``).  When a request is answered
            ``404`` because the server no longer knows the session — a
            restart, a redeploy, a replica without the session — the
            client opens a new session (``initialize``) and retries that
            request once, as the MCP specification requires.  The server
            never ran the refused request, so the retry is safe.  A
            connection that drops *during* a call is not retried (the tool
            may have run); the next call opens a new session instead.
            Ignored for stdio.

    Example — unauthenticated::

        async with MCPClient(url="http://localhost:8080/mcp") as client:
            tools = await client.list_tools()

    Example — with Bearer token::

        async with MCPClient(
            url="http://localhost:8080/mcp",
            bearer_token="eyJhbGciOiJIUzI1NiIs...",
        ) as client:
            result = await client.call_tool("search", {"query": "python"})

    Example — with API key::

        async with MCPClient(
            url="http://localhost:8080/mcp",
            api_key="my-secret-key",
        ) as client:
            tools = await client.list_tools()

    Example — fetch token from server endpoint::

        token = await MCPClient.fetch_token(
            "http://localhost:8080/auth/token",
            client_id="my-agent",
            client_secret="agent-secret",
        )
        async with MCPClient(
            url="http://localhost:8080/mcp",
            bearer_token=token,
        ) as client:
            tools = await client.list_tools()
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        transport: str = "http",
        headers: dict[str, str] | None = None,
        bearer_token: str | None = None,
        api_key: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = 30.0,
        auto_reconnect: bool = True,
    ) -> None:
        self._url = url
        self._transport = transport
        self._headers = dict(headers or {})
        self._command = command
        self._args = args or []
        self._env = env or {}
        self._cwd = cwd
        self._timeout = timeout
        self._auto_reconnect = auto_reconnect and transport in _NETWORK_TRANSPORTS

        # Session state (set on __aenter__).  The session is owned by
        # ``_runner``; ``_closing`` asks it to shut down, and ``_failure``
        # records why it ended if the connection dropped on its own.
        # ``_active`` is True between a successful ``__aenter__`` and
        # ``__aexit__`` — while the caller wants a connection, a lost
        # session may be re-opened.  ``_generation`` counts the sessions
        # opened so far; ``_reconnect_lock`` lets concurrent calls that hit
        # the same lost session share one re-initialisation.
        self._session: ClientSession | None = None
        self._runner: asyncio.Task[None] | None = None
        self._closing: asyncio.Event | None = None
        self._failure: MCPClientError | None = None
        self._active = False
        self._generation = 0
        self._reconnect_lock = asyncio.Lock()

        # Inject Bearer token as Authorization header
        if bearer_token:
            self._headers["authorization"] = f"Bearer {bearer_token}"

        # Inject API key as x-api-key header
        if api_key:
            self._headers["x-api-key"] = api_key

    # ------------------------------------------------------------------
    # Token acquisition helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def fetch_token(
        token_url: str,
        client_id: str,
        client_secret: str,
        *,
        timeout: float = 10.0,
    ) -> str:
        """Fetch a Bearer token from a server's token endpoint.

        This is a convenience for development/testing when the MCP server
        has a built-in token issuer enabled via
        :meth:`~promptise.mcp.server.MCPServer.enable_token_endpoint`.

        For production, obtain tokens from your Identity Provider
        (Auth0, Keycloak, Okta, etc.) and pass them directly via
        ``bearer_token``.

        Args:
            token_url: Full URL of the token endpoint
                (e.g. ``http://localhost:8080/auth/token``).
            client_id: Client identifier registered on the server.
            client_secret: Client secret for authentication.
            timeout: HTTP request timeout in seconds.

        Returns:
            The access token string (ready to pass as ``bearer_token``).

        Raises:
            MCPClientError: On network errors or authentication failure.
        """
        try:
            import httpx
        except ImportError as exc:
            raise MCPClientError(
                "httpx is required for fetch_token(). Install it with: pip install httpx"
            ) from exc

        try:
            async with httpx.AsyncClient(timeout=timeout) as http:
                resp = await http.post(
                    token_url,
                    json={
                        "client_id": client_id,
                        "client_secret": client_secret,
                    },
                )
                if resp.status_code != 200:
                    body = resp.text
                    raise MCPClientError(f"Token request failed (HTTP {resp.status_code}): {body}")
                data = resp.json()
                token = data.get("access_token")
                if not token:
                    raise MCPClientError(f"Token response missing 'access_token': {data}")
                return token
        except httpx.HTTPError as exc:
            raise MCPClientError(f"Token request failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> MCPClient:
        """Connect to the server and initialise the session.

        Raises:
            MCPConnectionRejectedError: The HTTP server answered the
                handshake with a 4xx status (e.g. 401 Unauthorized, or
                404 for a URL that is not an MCP endpoint).
            MCPClientError: Any other connection or handshake failure.
        """
        if self._runner is not None:
            raise MCPClientError("MCPClient is already connected")
        await self._open()
        self._active = True
        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Close the session and transport.

        Cleanup is best-effort and never raises: the session is closed in
        the task that opened it, so it is safe to call from any task.
        """
        self._active = False
        await self._shutdown()

    @property
    def session_generation(self) -> int:
        """How many MCP sessions this client has opened.

        ``1`` after connecting; each transparent re-initialisation after a
        lost session adds one.  Compare the value before and after a call to
        tell whether the server's session — and so possibly its tool list —
        changed underneath it.
        """
        return self._generation

    async def _open(self) -> None:
        """Start the runner and wait for the ``initialize`` handshake."""
        open_transport = self._transport_opener()
        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._closing = asyncio.Event()
        self._failure = None
        self._runner = asyncio.create_task(
            self._run_session(open_transport, ready, self._closing),
            name=f"promptise-mcp-client:{self._target}",
        )
        try:
            await ready
        except BaseException:
            # Handshake failed (the runner has already unwound in its own
            # task) or the caller was cancelled mid-handshake (stop it now).
            self._runner.cancel()
            await self._shutdown()
            raise
        self._generation += 1

    @property
    def _target(self) -> str:
        """Human-readable connection target for errors and logs."""
        if self._transport == "stdio":
            return " ".join([self._command or "", *self._args]).strip()
        return self._url or ""

    def _transport_opener(self) -> Any:
        """Validate the configuration and return a transport factory.

        The factory returns an async context manager yielding
        ``(read_stream, write_stream, ...)``.  Validation happens here, in
        the caller's task, so configuration errors raise immediately.
        """
        if self._transport in ("http", "streamable-http"):
            if not self._url:
                raise MCPClientError("url is required for HTTP transport")
            from mcp.client.streamable_http import streamablehttp_client

            return lambda: streamablehttp_client(
                url=self._url,
                headers=self._headers or None,
                timeout=self._timeout,
            )
        if self._transport == "sse":
            if not self._url:
                raise MCPClientError("url is required for SSE transport")
            from mcp.client.sse import sse_client

            return lambda: sse_client(
                url=self._url,
                headers=self._headers or None,
                timeout=self._timeout,
            )
        if self._transport == "stdio":
            from mcp.client.stdio import stdio_client

            params = self._stdio_params()
            return lambda: stdio_client(params)
        raise MCPClientError(f"Unknown transport: {self._transport!r}")

    async def _run_session(
        self,
        open_transport: Any,
        ready: asyncio.Future[None],
        closing: asyncio.Event,
    ) -> None:
        """Own the transport and session for the lifetime of the connection.

        Every anyio scope the SDK opens is entered and exited here, so a
        failure inside the transport's task group cancels only this task.
        The outcome of the handshake is reported through *ready*; setting
        *closing* ends the session.
        """
        try:
            async with AsyncExitStack() as stack:
                streams = await stack.enter_async_context(open_transport())
                session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
                await session.initialize()
                self._session = session
                if not ready.done():
                    ready.set_result(None)
                await closing.wait()
        except (Exception, asyncio.CancelledError) as exc:
            if not ready.done():
                ready.set_exception(self._connect_error(exc))
            elif not closing.is_set():
                self._failure = self._connect_error(exc, connected=True)
                logger.warning("%s", self._failure)
            else:
                logger.debug("MCP session cleanup error", exc_info=True)
        finally:
            self._session = None
            if not ready.done():
                ready.set_exception(MCPClientError(f"Connection to {self._target} closed"))

    def _connect_error(self, exc: BaseException, *, connected: bool = False) -> MCPClientError:
        """Translate a transport failure into a typed, readable error."""
        import httpx

        leaves = _leaf_exceptions(exc)
        if not connected and any(_is_session_terminated(leaf) for leaf in leaves):
            # The SDK reports a 404 answer to ``initialize`` as a terminated
            # session; with no session yet, it means nothing serves MCP here.
            not_found = MCPConnectionRejectedError(
                status_code=404, reason="Not Found", url=str(self._url)
            )
            not_found.__cause__ = exc
            return not_found
        for leaf in leaves:
            if isinstance(leaf, httpx.HTTPStatusError):
                response = leaf.response
                if not connected and 400 <= response.status_code < 500:
                    error: MCPClientError = MCPConnectionRejectedError(
                        status_code=response.status_code,
                        reason=response.reason_phrase,
                        url=str(self._url),
                    )
                else:
                    error = MCPClientError(
                        f"Server at {self._target} answered HTTP "
                        f"{response.status_code} {response.reason_phrase}".rstrip()
                    )
                error.__cause__ = leaf
                return error
        meaningful = [leaf for leaf in leaves if not isinstance(leaf, asyncio.CancelledError)]
        cause = meaningful[0] if meaningful else exc
        if isinstance(cause, MCPClientError):
            return cause
        detail = f"{type(cause).__name__}: {cause}" if str(cause) else type(cause).__name__
        if connected:
            error = MCPClientError(f"Connection to {self._target} was lost ({detail})")
        else:
            error = MCPClientError(f"Failed to connect to {self._target} ({detail})")
        error.__cause__ = cause
        return error

    async def _shutdown(self) -> None:
        """Ask the runner to close the session and wait for it to finish."""
        runner, self._runner = self._runner, None
        if runner is None:
            return
        if self._closing is not None:
            self._closing.set()
        done, _ = await asyncio.wait({runner}, timeout=self._timeout)
        if not done:
            logger.debug("MCP session did not close within %ss; cancelling", self._timeout)
            runner.cancel()
            await asyncio.wait({runner})
        self._session = None

    def _stdio_params(self) -> Any:
        """Build the stdio launch parameters, including the working dir.

        Factored out so the parameter mapping — notably ``cwd`` — can be
        unit-tested without launching a subprocess.

        Raises:
            MCPClientError: If no command is configured for stdio transport.
        """
        if not self._command:
            raise MCPClientError("command is required for stdio transport")

        from mcp.client.stdio import StdioServerParameters

        return StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env or None,
            cwd=self._cwd,
        )

    # ------------------------------------------------------------------
    # MCP operations
    # ------------------------------------------------------------------

    def _require_session(self) -> ClientSession:
        if self._session is None:
            if self._failure is not None:
                raise self._failure
            raise MCPClientError("Not connected. Use 'async with MCPClient(...) as client:'")
        return self._session

    async def _live_session(self) -> ClientSession:
        """The current session, re-opening it first if it dropped on its own."""
        if self._session is None and self._active and self._auto_reconnect:
            await self._reconnect(self._generation, reason=str(self._failure or "session closed"))
        return self._require_session()

    async def _reconnect(self, seen_generation: int, *, reason: str) -> None:
        """Replace the session the caller saw (*seen_generation*) with a new one.

        Concurrent callers that lost the same session share one
        re-initialisation: whoever takes the lock second finds a newer
        generation and returns.
        """
        async with self._reconnect_lock:
            if self._generation != seen_generation and self._session is not None:
                return
            logger.info("Re-opening the MCP session to %s (%s)", self._target, reason)
            await self._shutdown()
            try:
                await self._open()
            except MCPClientError as exc:
                # Stay active: the next call tries again (the server may be
                # still starting).
                self._failure = exc
                raise

    async def _with_session(self, operation: Callable[[ClientSession], Awaitable[_T]]) -> _T:
        """Run *operation*, re-initialising once if the server lost the session."""
        session = await self._live_session()
        generation = self._generation
        try:
            return await self._on_current_session(operation(session))
        except _SessionReplaced:
            # Another call found the session gone (404) and opened a new one
            # while this request waited on the old one, which the server had
            # forgotten too — the request was never processed.
            pass
        except Exception as exc:
            if not (self._auto_reconnect and self._active and _is_session_terminated(exc)):
                raise
            # The server answered 404 for the session: it never processed the
            # request, so it is safe to send it again on a new session.
            await self._reconnect(generation, reason="the server no longer knows the session (404)")
        async with self._reconnect_lock:
            pass  # let a re-initialisation in progress finish
        session = await self._live_session()
        try:
            return await self._on_current_session(operation(session))
        except _SessionReplaced:
            raise MCPClientError(f"The MCP session to {self._target} was replaced again") from None

    async def _on_current_session(self, request: Awaitable[_T]) -> _T:
        """Await *request*, failing fast if the session's runner ends first.

        The SDK leaves a request pending forever when its transport dies
        or its session is closed underneath it; racing the runner turns
        that into an error.
        """
        runner, closing = self._runner, self._closing
        pending = asyncio.ensure_future(request)
        if runner is None:
            return await pending
        try:
            await asyncio.wait({pending, runner}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            pending.cancel()
            raise
        if pending.done():
            return pending.result()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        if self._active and closing is not None and closing.is_set():
            raise _SessionReplaced
        raise self._failure or MCPClientError(f"Connection to {self._target} was closed")

    async def list_tools(self) -> list[Tool]:
        """List all tools from the connected server.

        Returns:
            List of MCP ``Tool`` objects with name, description, inputSchema.
        """
        try:
            result: ListToolsResult = await self._with_session(lambda s: s.list_tools())
            return list(result.tools)
        except MCPClientError:
            raise
        except Exception as exc:
            raise MCPClientError(f"Failed to list tools: {self._describe(exc)}") from exc

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> CallToolResult:
        """Call a tool on the connected server.

        Args:
            name: Tool name.
            arguments: Tool arguments dict.

        Returns:
            MCP ``CallToolResult`` with content list.
        """
        try:
            return await self._with_session(lambda s: s.call_tool(name, arguments))
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise MCPClientError(f"Timeout calling tool '{name}': {exc}") from exc
        except ConnectionError as exc:
            raise MCPClientError(f"Connection lost calling tool '{name}': {exc}") from exc
        except MCPClientError:
            raise  # Don't double-wrap
        except Exception as exc:
            raise MCPClientError(f"Failed to call tool '{name}': {self._describe(exc)}") from exc

    def _describe(self, exc: Exception) -> str:
        """Readable detail for a failed request."""
        if not _is_session_terminated(exc):
            return str(exc)
        if not self._auto_reconnect:
            return f"the server at {self._target} no longer knows this MCP session (HTTP 404)"
        return (
            f"the server at {self._target} answered HTTP 404 for a freshly opened MCP "
            "session. If it runs as several replicas or workers, route each "
            "mcp-session-id to the same one (sticky sessions) or serve it stateless."
        )

    @property
    def session(self) -> ClientSession | None:
        """The underlying MCP ``ClientSession``, or ``None`` if not connected."""
        return self._session

    @property
    def headers(self) -> dict[str, str]:
        """Current HTTP headers (read-only copy)."""
        return dict(self._headers)
