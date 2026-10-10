"""Single-server MCP client with token-based authentication.

Wraps the official MCP SDK's transport and session APIs into a clean
async context manager that handles:

- Transport selection (Streamable HTTP, SSE, stdio)
- Bearer token injection (from IdP or server token endpoint)
- API key injection (simple pre-shared secret)
- Custom header injection on every HTTP request
- Short-lived credentials: a ``bearer_token_provider`` is asked for the
  current token on every request, refreshed once on ``401``, and the
  session is reopened when the token changes
- Proper session lifecycle (initialize → use → close)
- Clear, typed errors when a server refuses the connection, also mid-session

The transport and session live in a task owned by the client.  The MCP
SDK's transports run their HTTP traffic in an anyio task group; owning
that task group in a dedicated task means a transport failure (an HTTP
401 during ``initialize``, a dropped connection) is entered, unwound and
reported in one task, and never cancels the caller's task.  Every
operation races the task that owns its session, so a session that dies
mid-call (an HTTP 401 once a token has expired) fails the call at once
instead of leaving it waiting for a response that can never arrive.

The client **never** generates JWTs.  Tokens are obtained externally
(from an Identity Provider or the server's built-in token endpoint)
and passed in via ``bearer_token``, ``api_key``, or ``headers``.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import AsyncExitStack
from typing import Any, TypeVar, Union

import httpx
from mcp.client.session import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ListToolsResult, Tool

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: A callable returning the bearer token to present, or ``None`` for none.
#: It receives ``force_refresh``: ``False`` for a normal request (a cached
#: token that is still valid is fine), ``True`` after the server answered
#: ``401`` (bypass any cache).  Sync callables run in a worker thread, so
#: they may block on an identity provider; async callables are awaited.
BearerTokenProvider = Callable[[bool], Union[str, None, Awaitable[Union[str, None]]]]


class MCPClientError(RuntimeError):
    """Raised when an MCP client operation fails."""


class MCPConnectionRejectedError(MCPClientError):
    """Raised when an HTTP MCP server rejects the session with a 4xx status.

    Typically ``401``/``403`` (missing or wrong ``bearer_token``/``api_key``)
    or ``404`` (wrong endpoint URL).  Retrying with the same configuration
    will fail the same way.

    Also raised when a server rejects the credential of an established
    session (``mid_session=True``), typically because a static
    ``bearer_token`` expired.  The call that hit it fails at once.

    Attributes:
        status_code: The HTTP status the server answered with.
        reason: The HTTP reason phrase (e.g. ``"Unauthorized"``).
        url: The endpoint that was contacted.
        server_name: The server's name when connected through
            :class:`~promptise.mcp.client.MCPMultiClient` or ``build_agent``.
        mid_session: ``True`` when an established session was rejected,
            ``False`` when the handshake was.
    """

    def __init__(
        self,
        *,
        status_code: int,
        reason: str,
        url: str,
        server_name: str | None = None,
        mid_session: bool = False,
        refreshable: bool = False,
    ) -> None:
        self.status_code = status_code
        self.reason = reason
        self.url = url
        self.server_name = server_name
        self.mid_session = mid_session
        self._refreshable = refreshable
        who = f"Server '{server_name}'" if server_name else f"Server at {url}"
        status = f"{status_code} {reason}" if reason else str(status_code)
        if not mid_session:
            message = f"{who} rejected the connection: {status}."
            if status_code in (401, 403):
                message += " Check the bearer_token/api_key configured for it."
            elif status_code == 404:
                message += f" Check the URL ({url}); Promptise servers serve MCP at /mcp."
        else:
            message = f"{who} rejected the session's credential mid-session: {status}."
            if status_code in (401, 403) and refreshable:
                message += (
                    " A freshly acquired token was rejected too: check that the"
                    " server trusts its issuer and audience."
                )
            elif status_code in (401, 403):
                message += (
                    " The bearer token most likely expired; pass"
                    " bearer_token_provider (or build the agent with an identity)"
                    " so the client presents a fresh one."
                )
        super().__init__(message)

    def for_server(self, server_name: str) -> MCPConnectionRejectedError:
        """Return a copy of this error that names *server_name*."""
        return MCPConnectionRejectedError(
            status_code=self.status_code,
            reason=self.reason,
            url=self.url,
            server_name=server_name,
            mid_session=self.mid_session,
            refreshable=self._refreshable,
        )


class MCPCredentialError(MCPClientError):
    """Raised when ``bearer_token_provider`` cannot supply a credential.

    The request that needed it is **not sent**: a client configured with a
    provider never falls back to an unauthenticated request.  The provider's
    own exception is attached as ``__cause__``; the message names only its
    type, so it can never echo a credential.
    """


def _leaf_exceptions(exc: BaseException) -> list[BaseException]:
    """Flatten (possibly nested) exception groups into their leaf exceptions."""
    nested = getattr(exc, "exceptions", None)
    if isinstance(nested, (list, tuple)):
        leaves: list[BaseException] = []
        for inner in nested:
            leaves.extend(_leaf_exceptions(inner))
        return leaves
    return [exc]


def _session_terminated(exc: BaseException) -> bool:
    """Whether *exc* is the SDK's report that the server forgot the session.

    The Streamable HTTP transport answers a request whose POST got HTTP 404
    (unknown session) with this error.  The server never ran the request,
    so it is safe to send it again on a new session.  Promptise servers
    answer 404 when a request presents a different credential than the one
    that opened the session, which is what a refreshed token looks like.
    """
    return isinstance(exc, McpError) and exc.error.message == "Session terminated"


class _TokenSource:
    """Asks a :data:`BearerTokenProvider` for tokens, collapsing refreshes."""

    def __init__(self, provider: BearerTokenProvider) -> None:
        self._provider = provider
        self._latest: str | None = None
        self._refresh_lock: asyncio.Lock | None = None

    async def _ask(self, force_refresh: bool) -> str | None:
        """Ask the provider; any failure fails closed as :class:`MCPCredentialError`."""
        try:
            if inspect.iscoroutinefunction(self._provider):
                token = await self._provider(force_refresh)
            else:
                import anyio.to_thread

                token = await anyio.to_thread.run_sync(self._provider, force_refresh)
                if inspect.isawaitable(token):
                    token = await token
        except Exception as exc:
            raise MCPCredentialError(
                f"bearer_token_provider failed ({type(exc).__name__}); the request "
                "was not sent without a credential"
            ) from exc
        if token is not None and not isinstance(token, str):
            raise MCPCredentialError(
                f"bearer_token_provider returned {type(token).__name__}, expected str or None"
            )
        return token or None

    async def current(self) -> str | None:
        """The token to present now (the provider may serve it from cache)."""
        self._latest = await self._ask(False)
        return self._latest

    async def refresh(self, rejected: str | None) -> str | None:
        """A fresh token to replace *rejected*, which the server refused.

        Concurrent requests rejected with the same token share a single
        forced refresh.
        """
        if self._refresh_lock is None:
            self._refresh_lock = asyncio.Lock()
        async with self._refresh_lock:
            if self._latest is not None and self._latest != rejected:
                return self._latest  # another request already refreshed it
            self._latest = await self._ask(True)
            return self._latest


class _BearerAuth(httpx.Auth):
    """Per-connection httpx auth hook: a current token on every request.

    On ``401`` the token is refreshed once and the request re-sent.  The
    token that opened the session is remembered so the client can reopen
    the session when the token changes; once the connection is retired
    (replaced by a newer one), its remaining requests — in-flight calls
    and the closing ``DELETE`` — keep presenting that token, so they
    address the session they belong to.
    """

    def __init__(self, source: _TokenSource) -> None:
        self._source = source
        self.opened_with: str | None = None
        self.retired = False

    def sync_auth_flow(self, request: httpx.Request) -> Any:
        raise RuntimeError("MCPClient's bearer_token_provider only supports async transports")

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        if self.retired:
            token = self.opened_with
        else:
            token = await self._source.current()
        _set_bearer(request, token)
        response = yield request
        if response.status_code != 401 or self.retired:
            self._note_opened(token, response)
            return
        fresh = await self._source.refresh(rejected=token)
        if fresh is None or fresh == token:
            return  # nothing better to offer: the 401 stands
        logger.info("Server at %s answered 401; retrying once with a refreshed token", request.url)
        _set_bearer(request, fresh)
        response = yield request
        self._note_opened(fresh, response)

    def _note_opened(self, token: str | None, response: httpx.Response) -> None:
        """Remember the token the server accepted to open the session.

        Only the first accepted request (``initialize``) counts, so a
        handshake that succeeded on the 401 retry records the refreshed
        token rather than the rejected one.
        """
        if self.opened_with is None and response.status_code < 400:
            self.opened_with = token


def _set_bearer(request: httpx.Request, token: str | None) -> None:
    if token:
        request.headers["Authorization"] = f"Bearer {token}"
    else:
        request.headers.pop("Authorization", None)


def _jwt_expired(token: str | None) -> bool:
    """Whether *token* is a JWT whose ``exp`` has passed (read, not verified)."""
    if not token or token.count(".") < 2:
        return False
    import base64
    import json
    import time

    segment = token.split(".")[1]
    try:
        claims = json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))
    except (ValueError, TypeError):
        return False
    exp = claims.get("exp") if isinstance(claims, dict) else None
    return isinstance(exp, (int, float)) and exp <= time.time()


class _RetiredSessionTransport(httpx.AsyncBaseTransport):
    """Skips the closing ``DELETE`` of a replaced session whose token expired.

    The server would refuse it (and the SDK would log a warning on every
    token renewal); an abandoned session expires on the server by itself.
    """

    def __init__(self, auth: _BearerAuth) -> None:
        self._auth = auth
        self._inner = httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if (
            request.method == "DELETE"
            and self._auth.retired
            and _jwt_expired(self._auth.opened_with)
        ):
            return httpx.Response(204, request=request)
        return await self._inner.handle_async_request(request)

    async def aclose(self) -> None:
        await self._inner.aclose()


class _Connection:
    """One MCP session and the task that owns its transport."""

    def __init__(self, auth: _BearerAuth | None) -> None:
        self.auth = auth
        self.session: ClientSession | None = None
        self.runner: asyncio.Task[None] | None = None
        self.closing = asyncio.Event()
        # Why the session ended, if it ended on its own.
        self.failure: MCPClientError | None = None
        self.inflight = 0
        self.retired = False

    @property
    def alive(self) -> bool:
        return self.runner is not None and not self.runner.done() and self.session is not None


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
        bearer_token_provider: For short-lived tokens, instead of
            ``bearer_token``: a callable returning the current token (or
            ``None``), asked on every HTTP request.  It receives
            ``force_refresh`` — ``True`` after the server answered ``401``,
            when the request is re-sent once with the refreshed token.  A
            session opened with a token the provider has since replaced is
            reopened before the next call, and a call the server rejected
            for an unknown session (it never ran) is re-sent once on a new
            session.  Sync callables run in a worker thread.  HTTP/SSE only.

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

    Example — with short-lived tokens from an identity::

        async with MCPClient(
            url="http://localhost:8080/mcp",
            bearer_token_provider=lambda force: identity.get_credential(
                "api://my-server", force_refresh=force
            ),
        ) as client:
            result = await client.call_tool("search", {"query": "python"})

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
        bearer_token_provider: BearerTokenProvider | None = None,
    ) -> None:
        if bearer_token and bearer_token_provider is not None:
            raise MCPClientError("Pass either bearer_token or bearer_token_provider, not both")
        if bearer_token_provider is not None and transport == "stdio":
            raise MCPClientError("bearer_token_provider needs an HTTP or SSE transport")
        self._url = url
        self._transport = transport
        self._headers = dict(headers or {})
        self._command = command
        self._args = args or []
        self._env = env or {}
        self._cwd = cwd
        self._timeout = timeout

        # A provider replaces any static Authorization header: the auth hook
        # sets the header on each request.  It is never copied into
        # ``_headers``, so a copy of this client's headers never carries it.
        self._token_source: _TokenSource | None = None
        if bearer_token_provider is not None:
            self._token_source = _TokenSource(bearer_token_provider)
            self._headers = {k: v for k, v in self._headers.items() if k.lower() != "authorization"}

        # Connection state (set on __aenter__).  ``_conn`` is the current
        # session; a connection replaced while calls were still using it is
        # retired and closed once they finish (``_closers``).
        self._entered = False
        self._conn: _Connection | None = None
        self._reconnect_lock: asyncio.Lock | None = None
        self._closers: set[asyncio.Task[None]] = set()

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
                handshake with a 4xx status (e.g. 401 Unauthorized).
            MCPClientError: Any other connection or handshake failure.
        """
        self._transport_opener(None)  # validate the configuration first
        if self._entered:
            raise MCPClientError("MCPClient is already connected")
        self._conn = await self._open_connection()
        self._entered = True
        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Close the session and transport.

        Cleanup is best-effort and never raises: the session is closed in
        the task that opened it, so it is safe to call from any task.
        """
        self._entered = False
        conn, self._conn = self._conn, None
        if conn is not None:
            await self._close_connection(conn)
        while self._closers:
            await asyncio.wait(set(self._closers))

    @property
    def _target(self) -> str:
        """Human-readable connection target for errors and logs."""
        if self._transport == "stdio":
            return " ".join([self._command or "", *self._args]).strip()
        return self._url or ""

    @property
    def _session(self) -> ClientSession | None:
        return self._conn.session if self._conn is not None else None

    def _transport_opener(self, auth: httpx.Auth | None) -> Any:
        """Validate the configuration and return a transport factory.

        The factory returns an async context manager yielding
        ``(read_stream, write_stream, ...)``.  Validation happens here, in
        the caller's task, so configuration errors raise immediately.
        """
        if self._transport in ("http", "streamable-http"):
            if not self._url:
                raise MCPClientError("url is required for HTTP transport")
            from mcp.client.streamable_http import streamablehttp_client

            factory: dict[str, Any] = {}
            if isinstance(auth, _BearerAuth):
                bearer = auth
                factory["auth"] = auth

                def _client_factory(
                    headers: dict[str, str] | None = None,
                    timeout: httpx.Timeout | None = None,
                    auth: httpx.Auth | None = None,
                ) -> httpx.AsyncClient:
                    return httpx.AsyncClient(
                        headers=headers,
                        timeout=timeout,
                        auth=auth,
                        transport=_RetiredSessionTransport(bearer),
                    )

                factory["httpx_client_factory"] = _client_factory
            return lambda: streamablehttp_client(
                url=self._url,
                headers=self._headers or None,
                timeout=self._timeout,
                **factory,
            )
        if self._transport == "sse":
            if not self._url:
                raise MCPClientError("url is required for SSE transport")
            from mcp.client.sse import sse_client

            sse_auth: dict[str, Any] = {"auth": auth} if auth is not None else {}
            return lambda: sse_client(
                url=self._url,
                headers=self._headers or None,
                timeout=self._timeout,
                **sse_auth,
            )
        if self._transport == "stdio":
            from mcp.client.stdio import stdio_client

            params = self._stdio_params()
            return lambda: stdio_client(params)
        raise MCPClientError(f"Unknown transport: {self._transport!r}")

    async def _open_connection(self) -> _Connection:
        """Open a session in a new runner task and wait for its handshake."""
        auth = _BearerAuth(self._token_source) if self._token_source is not None else None
        open_transport = self._transport_opener(auth)
        conn = _Connection(auth)
        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        conn.runner = asyncio.create_task(
            self._run_session(conn, open_transport, ready),
            name=f"promptise-mcp-client:{self._target}",
        )
        try:
            await ready
        except BaseException:
            # Handshake failed (the runner has already unwound in its own
            # task) or the caller was cancelled mid-handshake (stop it now).
            conn.runner.cancel()
            await self._close_connection(conn)
            raise
        return conn

    async def _run_session(
        self,
        conn: _Connection,
        open_transport: Any,
        ready: asyncio.Future[None],
    ) -> None:
        """Own the transport and session for the lifetime of the connection.

        Every anyio scope the SDK opens is entered and exited here, so a
        failure inside the transport's task group cancels only this task.
        The outcome of the handshake is reported through *ready*; setting
        ``conn.closing`` ends the session.
        """
        try:
            async with AsyncExitStack() as stack:
                streams = await stack.enter_async_context(open_transport())
                session = await stack.enter_async_context(ClientSession(streams[0], streams[1]))
                await session.initialize()
                conn.session = session
                if not ready.done():
                    ready.set_result(None)
                await conn.closing.wait()
        except (Exception, asyncio.CancelledError) as exc:
            if not ready.done():
                ready.set_exception(self._connect_error(exc))
            elif not conn.closing.is_set():
                conn.failure = self._connect_error(exc, connected=True)
                if conn.retired:
                    logger.debug("Retired MCP session ended: %s", conn.failure)
                else:
                    logger.warning("%s", conn.failure)
            else:
                logger.debug("MCP session cleanup error", exc_info=True)
        finally:
            conn.session = None
            if not ready.done():
                ready.set_exception(MCPClientError(f"Connection to {self._target} closed"))

    def _connect_error(self, exc: BaseException, *, connected: bool = False) -> MCPClientError:
        """Translate a transport failure into a typed, readable error."""
        leaves = _leaf_exceptions(exc)
        for leaf in leaves:
            if isinstance(leaf, httpx.HTTPStatusError):
                response = leaf.response
                if not connected and 400 <= response.status_code < 500:
                    error: MCPClientError = MCPConnectionRejectedError(
                        status_code=response.status_code,
                        reason=response.reason_phrase,
                        url=str(self._url),
                    )
                elif response.status_code in (401, 403):
                    error = MCPConnectionRejectedError(
                        status_code=response.status_code,
                        reason=response.reason_phrase,
                        url=str(self._url),
                        mid_session=True,
                        refreshable=self._token_source is not None,
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

    async def _close_connection(self, conn: _Connection) -> None:
        """Ask the runner to close the session and wait for it to finish."""
        runner, conn.runner = conn.runner, None
        if runner is None:
            return
        conn.closing.set()
        done, _ = await asyncio.wait({runner}, timeout=self._timeout)
        if not done:
            logger.debug("MCP session did not close within %ss; cancelling", self._timeout)
            runner.cancel()
            await asyncio.wait({runner})
        conn.session = None

    def _retire(self, conn: _Connection) -> None:
        """Close a replaced connection once no call is using it any more."""
        conn.retired = True
        if conn.auth is not None:
            conn.auth.retired = True
        if conn.inflight == 0:
            task = asyncio.create_task(self._close_connection(conn))
            self._closers.add(task)
            task.add_done_callback(self._closers.discard)

    async def _reconnect(self, stale: _Connection | None, reason: str) -> None:
        """Replace *stale* with a new session, unless another call already did."""
        if self._reconnect_lock is None:
            self._reconnect_lock = asyncio.Lock()
        async with self._reconnect_lock:
            if not self._entered or self._conn is not stale:
                return
            logger.info("Reopening the MCP session with %s: %s", self._target, reason)
            fresh = await self._open_connection()
            if not self._entered:  # closed while we were connecting
                await self._close_connection(fresh)
                return
            self._conn = fresh
            if stale is not None:
                self._retire(stale)

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
        conn = self._conn
        if conn is None or conn.session is None:
            if conn is not None and conn.failure is not None:
                raise conn.failure
            raise MCPClientError("Not connected. Use 'async with MCPClient(...) as client:'")
        return conn.session

    async def _acquire(self) -> _Connection:
        """Return the connection the next operation should use.

        With a ``bearer_token_provider``, a session that ended (say, after a
        rejected credential) is reopened, and so is one opened with a token
        the provider has since renewed: servers that bind a session to the
        credential that opened it (Promptise servers do) would not accept
        the new token on the old session.
        """
        if not self._entered:
            raise MCPClientError("Not connected. Use 'async with MCPClient(...) as client:'")
        conn = self._conn
        if self._token_source is not None:
            # Ask for the credential before anything is sent: if it cannot be
            # acquired, the call fails here (MCPCredentialError) and the
            # session stays open for later calls.
            token = await self._token_source.current()
            if conn is None or not conn.alive:
                await self._reconnect(conn, "the previous session ended")
            elif conn.auth is not None and token != conn.auth.opened_with:
                await self._reconnect(conn, "the bearer token was renewed")
            conn = self._conn
        self._require_session()
        assert conn is not None
        return conn

    async def _call(self, op: Callable[[ClientSession], Awaitable[_T]]) -> _T:
        """Run *op* on the current session; reopen it once if the server lost it.

        A request the server answered "unknown session" (HTTP 404) was never
        run, so it is re-sent on a new session.  Any other failure, including
        a credential rejected mid-session, fails the call at once.
        """
        retried = False
        while True:
            conn = await self._acquire()
            conn.inflight += 1
            try:
                return await self._race(conn, op)
            except McpError as exc:
                if retried or not _session_terminated(exc):
                    raise
                retried = True
                await self._reconnect(conn, "the server no longer knows the session")
            finally:
                conn.inflight -= 1
                if conn.retired and conn.inflight == 0:
                    self._retire(conn)

    async def _race(self, conn: _Connection, op: Callable[[ClientSession], Awaitable[_T]]) -> _T:
        """Await *op*, failing it as soon as the session's runner ends.

        The SDK reports a transport failure (such as an HTTP 401 on a
        request) by tearing down the runner's task group; a request already
        waiting for its response would otherwise wait forever.
        """
        session, runner = conn.session, conn.runner
        if session is None or runner is None:
            raise conn.failure or MCPClientError(f"Connection to {self._target} closed")
        task = asyncio.ensure_future(op(session))
        try:
            await asyncio.wait({task, runner}, return_when=asyncio.FIRST_COMPLETED)
        except BaseException:
            task.cancel()
            raise
        if task.done():
            return task.result()
        task.cancel()
        await asyncio.wait({task})
        raise conn.failure or MCPClientError(f"Connection to {self._target} closed")

    async def list_tools(self) -> list[Tool]:
        """List all tools from the connected server.

        Returns:
            List of MCP ``Tool`` objects with name, description, inputSchema.
        """
        try:
            result: ListToolsResult = await self._call(lambda session: session.list_tools())
            return list(result.tools)
        except MCPClientError:
            raise
        except Exception as exc:
            raise MCPClientError(f"Failed to list tools: {exc}") from exc

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

        Raises:
            MCPConnectionRejectedError: The server rejected the session's
                credential (``mid_session=True``).
            MCPClientError: Any other failure, including a session that
                ended while the call was in flight.
        """
        try:
            return await self._call(lambda session: session.call_tool(name, arguments))
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise MCPClientError(f"Timeout calling tool '{name}': {exc}") from exc
        except ConnectionError as exc:
            raise MCPClientError(f"Connection lost calling tool '{name}': {exc}") from exc
        except MCPClientError:
            raise  # Don't double-wrap
        except Exception as exc:
            raise MCPClientError(f"Failed to call tool '{name}': {exc}") from exc

    @property
    def session(self) -> ClientSession | None:
        """The underlying MCP ``ClientSession``, or ``None`` if not connected."""
        return self._session

    @property
    def headers(self) -> dict[str, str]:
        """Current static HTTP headers (read-only copy).

        A token from ``bearer_token_provider`` is set per request and is not
        included.
        """
        return dict(self._headers)
