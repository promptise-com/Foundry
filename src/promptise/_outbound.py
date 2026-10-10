"""Outbound HTTP to user-configured URLs, safe against DNS rebinding.

Webhooks (event sinks, approval handlers, runtime escalation) refuse
private and internal addresses as SSRF protection.  Checking the URL once,
when the webhook is configured, is not enough: DNS can answer differently
later (DNS rebinding, a short TTL, a host that did not resolve at startup),
and httpx resolves the name again when it connects.

:func:`pin_target` closes that window.  Call it right before every request:
it resolves the host, refuses the request if *any* answer is non-public,
and returns a URL that names the checked IP, plus the original ``Host``
header and the TLS server name (SNI, and so certificate verification) as
httpx request extensions — httpx never resolves the name itself.  Send the
request with ``follow_redirects=False``: a redirect could point anywhere.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

__all__ = ["BlockedTarget", "PinnedTarget", "pin_target"]


class BlockedTarget(ValueError):
    """The URL's host resolved to a private or internal address."""


@dataclass(frozen=True)
class PinnedTarget:
    """Where to send one request: pass all three fields to httpx.

    Attributes:
        url: The URL to request (the host replaced by the checked IP, unless
            private networks are allowed).
        headers: Extra headers (``Host``) to merge into the request's.
        extensions: httpx request extensions (``sni_hostname`` for https).
    """

    url: str
    headers: dict[str, str] = field(default_factory=dict)
    extensions: dict[str, Any] = field(default_factory=dict)


async def pin_target(url: str, *, allow_private_networks: bool = False) -> PinnedTarget:
    """Resolve *url*'s host now and pin the request to the address checked.

    Args:
        url: An ``http(s)`` URL.
        allow_private_networks: Skip the check and leave the URL as is
            (the caller opted in to private and internal hosts).

    Returns:
        The :class:`PinnedTarget` to send the request to.

    Raises:
        BlockedTarget: The host resolves to a private, loopback, link-local,
            reserved or otherwise non-public address.
        OSError: The host did not resolve.
    """
    if allow_private_networks:
        return PinnedTarget(url)

    # Looked up through the module so tests can treat a local address as public.
    from promptise.mcp.server import _openapi

    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, type=socket.SOCK_STREAM)
    addresses = [ipaddress.ip_address(str(info[4][0]).split("%", 1)[0]) for info in infos]
    if not addresses:
        raise OSError(f"{host!r} did not resolve")
    for ip in addresses:
        if _openapi._is_private_ip(ip):
            raise BlockedTarget(f"{host!r} resolves to private/internal IP {ip}")

    ip = addresses[0]
    netloc = f"[{ip}]" if ip.version == 6 else str(ip)
    if parsed.port:
        netloc += f":{parsed.port}"
    userinfo, at, host_header = parsed.netloc.rpartition("@")
    if at:
        netloc = f"{userinfo}@{netloc}"
    extensions: dict[str, Any] = {"sni_hostname": host} if parsed.scheme == "https" else {}
    return PinnedTarget(parsed._replace(netloc=netloc).geturl(), {"Host": host_header}, extensions)
