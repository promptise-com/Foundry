"""Webhook trigger: HTTP endpoint that fires trigger events.

An aiohttp server listens on a configurable host, port and path. When a POST
request arrives, a :class:`TriggerEvent` is produced with the request
body as payload.

Requests pass these checks in order:

1. ``allowed_sources`` — client IP must be in one of the listed IPs/CIDR
   ranges (403 otherwise).
2. Signature — when ``hmac_secret`` is set, the request must carry a valid
   HMAC-SHA256 signature in the scheme's header (401 otherwise):

   * ``generic``: ``X-Webhook-Signature: sha256=<hex of HMAC(body)>``
   * ``github``:  ``X-Hub-Signature-256: sha256=<hex of HMAC(body)>``
   * ``stripe``:  ``Stripe-Signature: t=<unix ts>,v1=<hex of HMAC("<ts>.<body>")>``,
     with the timestamp no older than ``signature_tolerance`` seconds.

3. Availability — when the owning process can't run events (failed,
   stopping) or its queue is full, the webhook answers 503 with a
   ``Retry-After`` header so the sender retries later.
4. ``event_filter`` — events that don't match get ``200 {"status": "ignored"}``
   and never reach the agent.

Accepted events get ``202 {"status": "accepted", "event_id": ...}``.

Uses ``aiohttp`` (ships with the base ``pip install promptise``).

Example::

    from promptise.runtime.triggers.webhook import WebhookTrigger

    trigger = WebhookTrigger(
        path="/github", port=9090,
        hmac_secret=os.environ["GITHUB_WEBHOOK_SECRET"],
        signature_scheme="github",
    )
    await trigger.start()

    # In another task:
    event = await trigger.wait_for_next()
    print(event.payload)  # {"action": "opened", ...} from the POST body

    await trigger.stop()
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac as hmac_mod
import ipaddress
import json
import logging
import time
from collections.abc import Callable
from typing import Any, Literal

from aiohttp import web

from ..config import SIGNATURE_HEADERS
from .base import TriggerEvent

logger = logging.getLogger(__name__)

_REDACTED_HEADERS = ("authorization", "cookie", "set-cookie", "proxy-authorization")


def _digest_equal(expected_hex: str, received: str) -> bool:
    """Timing-safe comparison that tolerates non-ASCII header values.

    ``hmac.compare_digest`` raises ``TypeError`` for ``str`` arguments with
    non-ASCII characters, which turned a garbage signature into a 500.
    """
    return hmac_mod.compare_digest(
        expected_hex.encode("ascii"), received.encode("utf-8", "surrogateescape")
    )


class WebhookTrigger:
    """HTTP webhook trigger.

    Starts an aiohttp server that listens for incoming POST requests
    and converts them to :class:`TriggerEvent` objects.

    Args:
        path: URL path to listen on (e.g. ``"/webhook"``).
        port: TCP port to bind to.
        host: Host/IP to bind to.  Defaults to ``"127.0.0.1"``
            (loopback only).  Set to ``"0.0.0.0"`` when external services
            or other nodes need to reach this webhook.
        hmac_secret: Optional HMAC secret for signature verification.
            When set, requests without a valid signature get 401.
            Verification uses timing-safe comparison.  **Strongly
            recommended** when ``host`` is not loopback.
        signature_scheme: ``"generic"``, ``"github"`` or ``"stripe"`` (see
            the module docs).
        signature_header: Header carrying the signature (defaults to the
            scheme's standard header).
        signature_tolerance: Max age in seconds of a ``stripe`` timestamp.
        allowed_sources: Client IPs / CIDR ranges allowed to call the
            webhook.  Empty means any.  Checked against the TCP peer
            address, so behind a reverse proxy list the proxy.
        event_filter: Optional predicate; events it rejects are answered
            with ``200 {"status": "ignored"}`` and not queued.
    """

    def __init__(
        self,
        path: str = "/webhook",
        port: int = 9090,
        host: str = "127.0.0.1",
        hmac_secret: str | None = None,
        *,
        signature_scheme: Literal["generic", "github", "stripe"] = "generic",
        signature_header: str | None = None,
        signature_tolerance: int = 300,
        allowed_sources: list[str] | None = None,
        event_filter: Callable[[TriggerEvent], bool] | None = None,
    ) -> None:
        if signature_scheme not in SIGNATURE_HEADERS:
            raise ValueError(f"Unknown signature_scheme {signature_scheme!r}")
        self._path = path
        self._port = port
        self._host = host
        self._hmac_secret = hmac_secret.encode() if hmac_secret else None
        self._signature_scheme = signature_scheme
        self._signature_header = signature_header or SIGNATURE_HEADERS[signature_scheme]
        self._signature_tolerance = signature_tolerance
        # Stripe signatures already accepted -> expiry time (replay protection)
        self._seen_signatures: dict[str, float] = {}
        self._allowed_networks = [
            ipaddress.ip_network(src, strict=False) for src in (allowed_sources or [])
        ]
        self.event_filter = event_filter
        self._availability_check: Callable[[], str | None] | None = None
        if self._hmac_secret is None:
            logger.warning(
                "WebhookTrigger on port %d has no HMAC secret — "
                "any HTTP client can trigger this webhook. "
                "Set hmac_secret for production use.",
                port,
            )

        self.trigger_id: str = f"webhook-{port}{path}"
        self._queue: asyncio.Queue[TriggerEvent] = asyncio.Queue(maxsize=1000)
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._stopped = False

    def set_availability_check(self, check: Callable[[], str | None] | None) -> None:
        """Install a callback that says whether events can be accepted.

        The callback returns ``None`` when events can be processed, or a
        short reason (``"process failed"``, ``"queue full"``) when they
        can't; the webhook then answers 503.  :class:`AgentProcess`
        installs this automatically.
        """
        self._availability_check = check

    async def start(self) -> None:
        """Start the webhook HTTP server."""
        self._stopped = False
        self._app = web.Application()
        self._app.router.add_post(self._path, self._handle_request)

        # Also add a health check endpoint
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self._host, self._port)
        await self._site.start()
        logger.info(
            "WebhookTrigger started on %s:%d%s",
            self._host,
            self._port,
            self._path,
        )

    async def stop(self) -> None:
        """Stop the webhook HTTP server."""
        self._stopped = True

        # Unblock any waiters
        sentinel = TriggerEvent(
            trigger_id=self.trigger_id,
            trigger_type="webhook",
            payload=None,
            metadata={"_stop": True},
        )
        try:
            self._queue.put_nowait(sentinel)
        except asyncio.QueueFull:
            pass

        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._site = None
        self._app = None
        logger.info("WebhookTrigger stopped")

    async def wait_for_next(self) -> TriggerEvent:
        """Wait for the next webhook event.

        Returns:
            A :class:`TriggerEvent` with the request body as payload.

        Raises:
            asyncio.CancelledError: If the wait is cancelled.
        """
        while True:
            event = await self._queue.get()
            # Skip stop sentinels
            if event.metadata and event.metadata.get("_stop"):
                if self._stopped:
                    raise asyncio.CancelledError("Webhook trigger stopped")
                continue
            return event

    # ------------------------------------------------------------------
    # Request checks
    # ------------------------------------------------------------------

    def _source_allowed(self, remote: str | None) -> bool:
        if not self._allowed_networks:
            return True
        if not remote:
            return False
        try:
            addr = ipaddress.ip_address(remote)
        except ValueError:
            return False
        return any(addr in net for net in self._allowed_networks)

    def _unavailable_reason(self) -> str | None:
        if self._stopped:
            return "webhook stopped"
        if self._queue.full():
            return "queue full"
        if self._availability_check is not None:
            try:
                return self._availability_check()
            except Exception:  # pragma: no cover - defensive
                logger.exception("WebhookTrigger: availability check failed")
                return "availability check failed"
        return None

    def _signature_error(self, headers: Any, raw_body: bytes) -> tuple[str | None, str | None]:
        """Check the request signature.

        Returns:
            ``(problem, replay_key)``: ``problem`` says why the signature is
            invalid (``None`` when valid); ``replay_key`` identifies a
            timestamped (``stripe``) signature so it can be accepted once.
        """
        assert self._hmac_secret is not None
        header = headers.get(self._signature_header, "")
        if not header:
            return f"missing {self._signature_header} header", None

        if self._signature_scheme == "stripe":
            parts: dict[str, list[str]] = {}
            for item in header.split(","):
                key, _, value = item.strip().partition("=")
                parts.setdefault(key, []).append(value)
            timestamps = parts.get("t", [])
            signatures = parts.get("v1", [])
            if len(timestamps) != 1 or not timestamps[0].isdigit() or not signatures:
                return "malformed signature header", None
            if abs(time.time() - int(timestamps[0])) > self._signature_tolerance:
                return "signature timestamp outside tolerance", None
            signed = timestamps[0].encode() + b"." + raw_body
            expected = hmac_mod.new(self._hmac_secret, signed, hashlib.sha256).hexdigest()
            if not any(_digest_equal(expected, sig) for sig in signatures):
                return "signature mismatch", None
            # The timestamp is signed, so a captured request stays valid for
            # the whole tolerance window: each one is accepted only once.
            self._prune_replay_cache()
            replay_key = f"{timestamps[0]}:{expected}"
            if replay_key in self._seen_signatures:
                return "signature already used (replay)", None
            return None, replay_key

        if not header.startswith("sha256="):
            return "malformed signature header", None
        expected = hmac_mod.new(self._hmac_secret, raw_body, hashlib.sha256).hexdigest()
        if _digest_equal(expected, header[len("sha256=") :]):
            return None, None
        return "signature mismatch", None

    def _prune_replay_cache(self) -> None:
        now = time.time()
        for key in [k for k, expires in self._seen_signatures.items() if expires < now]:
            del self._seen_signatures[key]

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    async def _handle_request(self, request: web.Request) -> web.Response:
        """Handle incoming POST request."""
        try:
            if not self._source_allowed(request.remote):
                logger.warning("WebhookTrigger: rejected request from %s", request.remote)
                return web.json_response(
                    {"status": "error", "message": "Source not allowed"}, status=403
                )

            raw_body = await request.read()

            # -- HMAC signature verification (if configured) --
            replay_key: str | None = None
            if self._hmac_secret is not None:
                problem, replay_key = self._signature_error(request.headers, raw_body)
                if problem is not None:
                    logger.warning(
                        "WebhookTrigger: rejected request from %s (%s)", request.remote, problem
                    )
                    return web.json_response(
                        {"status": "error", "message": "Missing or invalid signature"},
                        status=401,
                    )

            # -- Can the owning process take the event? --
            reason = self._unavailable_reason()
            if reason is not None:
                return web.json_response(
                    {"status": "unavailable", "message": reason},
                    status=503,
                    headers={"Retry-After": "30"},
                )
            if replay_key is not None:
                # Recorded only once the event is taken, so a sender retrying
                # after a 503 isn't mistaken for a replay.
                self._seen_signatures[replay_key] = time.time() + 2 * self._signature_tolerance

            # Try to parse JSON body
            try:
                body = json.loads(raw_body)
            except (json.JSONDecodeError, Exception):
                body = raw_body.decode("utf-8", errors="replace")

            # Extract headers for metadata
            headers = dict(request.headers)

            event = TriggerEvent(
                trigger_id=self.trigger_id,
                trigger_type="webhook",
                payload=body,
                metadata={
                    "method": request.method,
                    "path": str(request.path),
                    "query": dict(request.query),
                    "headers": {
                        k: v
                        for k, v in headers.items()
                        if k.lower() not in _REDACTED_HEADERS
                        and k.lower() != self._signature_header.lower()
                    },
                    "remote": request.remote,
                    "signature_verified": self._hmac_secret is not None,
                },
            )

            if self.event_filter is not None and not self.event_filter(event):
                return web.json_response(
                    {"status": "ignored", "event_id": event.event_id}, status=200
                )

            try:
                self._queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("WebhookTrigger: queue full, dropping event")
                if replay_key is not None:
                    self._seen_signatures.pop(replay_key, None)
                return web.json_response(
                    {"status": "unavailable", "message": "queue full"},
                    status=503,
                    headers={"Retry-After": "30"},
                )

            return web.json_response(
                {"status": "accepted", "event_id": event.event_id},
                status=202,
            )

        except Exception:
            logger.exception("WebhookTrigger: error handling request")
            return web.json_response(
                {"status": "error", "message": "internal error"},
                status=500,
            )

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint (503 while events can't be accepted)."""
        reason = self._unavailable_reason()
        return web.json_response(
            {
                "status": "healthy" if reason is None else "unavailable",
                "reason": reason,
                "trigger_id": self.trigger_id,
                "queue_size": self._queue.qsize(),
            },
            status=200 if reason is None else 503,
        )

    def __repr__(self) -> str:
        return (
            f"WebhookTrigger(path={self._path!r}, "
            f"port={self._port}, "
            f"queue_size={self._queue.qsize()})"
        )
