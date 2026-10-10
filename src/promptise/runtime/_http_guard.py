"""Shared request checks for the runtime's local HTTP APIs.

Used by :class:`~promptise.runtime.distributed.transport.RuntimeTransport`
and :class:`~promptise.runtime.api.OrchestrationAPI`.  A server bound to a
loopback address is reachable by any web page open in a browser on the
same machine (a cross-site ``fetch``/form post) and, through DNS
rebinding, by pages that make their own host name resolve to
``127.0.0.1``.  Both arrive with tell-tale headers:

* DNS rebinding: the ``Host`` header carries the attacker's host name,
  not a loopback name.  Refused with ``421 Misdirected Request``.
* Cross-site requests: the browser adds an ``Origin`` header naming the
  page's site.  Anything but a loopback origin is refused with ``403``.

Clients that aren't browsers (curl, ``httpx``, the Promptise clients)
send a loopback ``Host`` and no ``Origin``, so they are unaffected.
"""

from __future__ import annotations

import hmac
import ipaddress
import re

#: ``Origin`` values a loopback-bound server accepts.
LOOPBACK_ORIGIN = re.compile(r"^https?://(localhost|127(\.\d{1,3}){3}|\[::1\])(:\d+)?$", re.I)


def is_loopback(host: str) -> bool:
    """True for a loopback bind address or host name."""
    name = host.strip().strip("[]").lower()
    if name == "localhost":
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def host_name(host_header: str) -> str:
    """The host part of a ``Host`` header (port removed, IPv6 brackets kept off)."""
    value = host_header.strip()
    if value.startswith("["):
        return value[1:].split("]", 1)[0]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def loopback_request_problem(host_header: str, origin: str | None) -> tuple[int, str] | None:
    """Check a request to a loopback-bound server.

    Returns:
        ``(status, message)`` when the request must be refused (``421``
        for a non-loopback ``Host``, ``403`` for a cross-site ``Origin``),
        otherwise ``None``.
    """
    if not is_loopback(host_name(host_header)):
        return 421, "Misdirected request"
    if origin is not None and not LOOPBACK_ORIGIN.match(origin):
        return 403, "Cross-origin request refused"
    return None


def bearer_token_matches(authorization: str, token: str) -> bool:
    """Timing-safe check of an ``Authorization: Bearer <token>`` header.

    Compares bytes, so a header with non-ASCII characters is a mismatch
    rather than a ``TypeError``.
    """
    if not authorization.startswith("Bearer "):
        return False
    return hmac.compare_digest(
        authorization[7:].encode("utf-8", "surrogateescape"), token.encode("utf-8")
    )
