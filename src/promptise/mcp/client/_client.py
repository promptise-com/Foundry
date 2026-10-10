"""Single-server MCP client with token-based authentication.

Wraps the official MCP SDK's transport and session APIs into a clean
async context manager that handles:

- Transport selection (Streamable HTTP, SSE, stdio)
- Bearer token injection (from IdP or server token endpoint)
- API key injection (simple pre-shared secret)
- Custom header injection on every HTTP request
- Proper session lifecycle (initialize → use → close)
- Clear, typed errors when a server refuses the connection

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
from contextlib import AsyncExitStack
from typing import Any

from mcp.client.session import ClientSession
from mcp.types import CallToolResult, ListToolsResult, Tool

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._url = url
        self._transport = transport
        self._headers = dict(headers or {})
        self._command = command
        self._args = args or []
        self._env = env or {}
        self._cwd = cwd
        self._timeout = timeout

        # Session state (set on __aenter__).  The session is owned by
        # ``_runner``; ``_closing`` asks it to shut down, and ``_failure``
        # records why it ended if the connection dropped on its own.
        self._session: ClientSession | None = None
        self._runner: asyncio.Task[None] | None = None
        self._closing: asyncio.Event | None = None
        self._failure: MCPClientError | None = None

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
        open_transport = self._transport_opener()
        if self._runner is not None:
            raise MCPClientError("MCPClient is already connected")

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
        return self

    async def __aexit__(self, *exc: Any) -> None:
        """Close the session and transport.

        Cleanup is best-effort and never raises: the session is closed in
        the task that opened it, so it is safe to call from any task.
        """
        await self._shutdown()

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

    async def list_tools(self) -> list[Tool]:
        """List all tools from the connected server.

        Returns:
            List of MCP ``Tool`` objects with name, description, inputSchema.
        """
        session = self._require_session()
        try:
            result: ListToolsResult = await session.list_tools()
            return list(result.tools)
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
        """
        session = self._require_session()
        try:
            return await session.call_tool(name, arguments)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise MCPClientError(f"Timeout calling tool '{name}': {exc}") from exc
        except ConnectionError as exc:
            raise MCPClientError(f"Connection lost calling tool '{name}': {exc}") from exc
        except MCPClientError:
            raise  # Don't double-wrap
        except Exception as exc:
            raise MCPClientError(f"Failed to call tool '{name}': {exc}") from exc

    @property
    def transport(self) -> str:
        """The transport this client connects with (``"http"``, ``"sse"``, ``"stdio"``)."""
        return self._transport

    @property
    def supports_bearer_token(self) -> bool:
        """Whether a per-caller bearer token can be sent to this server.

        ``True`` for HTTP and SSE.  ``False`` for stdio, which has no
        request headers: a stdio server runs with the agent's own
        privileges for every caller.
        """
        return self._transport != "stdio"

    def with_bearer_token(self, bearer_token: str) -> MCPClient:
        """Return a new, unconnected client that authenticates as *bearer_token*.

        The copy keeps this client's URL, transport, timeout and headers
        (including ``x-api-key``) and replaces any ``Authorization`` header,
        whatever its casing, with ``Bearer <bearer_token>``.  Used to open a
        session per caller, so one caller's token is never sent on another
        caller's requests.

        Raises:
            MCPClientError: For a stdio client, which cannot carry headers.
        """
        if not self.supports_bearer_token:
            raise MCPClientError(
                f"Cannot send a bearer token over the {self._transport} transport; "
                "only HTTP and SSE servers receive request headers."
            )
        headers = {k: v for k, v in self._headers.items() if k.lower() != "authorization"}
        return MCPClient(
            url=self._url,
            transport=self._transport,
            headers=headers,
            bearer_token=bearer_token,
            timeout=self._timeout,
        )

    @property
    def session(self) -> ClientSession | None:
        """The underlying MCP ``ClientSession``, or ``None`` if not connected."""
        return self._session

    @property
    def headers(self) -> dict[str, str]:
        """Current HTTP headers (read-only copy)."""
        return dict(self._headers)
