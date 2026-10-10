"""Which tools a caller may see, for ``MCPServer(hide_unauthorized_tools=True)``.

``tools/list`` normally returns every registered tool: guards are enforced
when a tool is *called*, so listing a tool grants nothing.  But names,
descriptions and schemas can themselves be sensitive — a pilot feature
switched on for one customer, an admin tool — and an agent shown a tool it
cannot use wastes turns on it.  With hiding enabled, the list (and the
``docs://manifest`` resource) is filtered per request: the request is
authenticated with the server's ``AuthMiddleware`` exactly as a tool call
would be, and each tool's guards are evaluated against that identity.

Filtering fails closed: a guard that raises, or credentials that do not
verify, hide the tool.  Calling a hidden tool is still refused by its
guards as before.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from ._context import RequestContext, clear_context, set_context

logger = logging.getLogger("promptise.server")


def _auth_middleware_for(tdef: Any, server_middlewares: list[Any]) -> Any | None:
    """The ``AuthMiddleware`` that would authenticate a call to *tdef*."""
    from ._auth import AuthMiddleware

    for mw in [*server_middlewares, *getattr(tdef, "router_middleware", [])]:
        if isinstance(mw, AuthMiddleware):
            return mw
    return None


async def visible_tools(
    tools: Iterable[Any],
    server_middlewares: list[Any],
    *,
    server_name: str,
    meta: dict[str, Any],
) -> list[Any]:
    """Return the tool definitions the caller identified by *meta* may call.

    Args:
        tools: Registered ``ToolDef`` objects.
        server_middlewares: The server's middleware list (searched, with each
            tool's router middleware, for the ``AuthMiddleware`` to use).
        server_name: Server name for the request contexts.
        meta: The request's HTTP headers (lower-cased names).
    """
    # One authenticated context per AuthMiddleware: a router can bring its own.
    contexts: dict[int, RequestContext] = {}

    async def context_for(auth_mw: Any | None) -> RequestContext:
        key = id(auth_mw)
        if key not in contexts:
            ctx = RequestContext(server_name=server_name, meta=dict(meta))
            if auth_mw is not None:
                set_context(ctx)
                try:
                    await auth_mw.authenticate(ctx)
                except Exception:
                    # Unauthenticated: auth-only and guarded tools stay hidden.
                    logger.debug("tools/list caller did not authenticate", exc_info=True)
                finally:
                    clear_context()
            contexts[key] = ctx
        return contexts[key]

    visible: list[Any] = []
    for tdef in tools:
        if not tdef.auth and not tdef.guards:
            visible.append(tdef)
            continue
        auth_mw = _auth_middleware_for(tdef, server_middlewares)
        ctx = await context_for(auth_mw)
        if tdef.auth and auth_mw is not None and ctx.client_id is None:
            continue
        ctx.tool_name = tdef.name
        ctx.state["tool_def"] = tdef
        if await _guards_allow(tdef.guards, ctx):
            visible.append(tdef)
    return visible


async def _guards_allow(guards: list[Any], ctx: RequestContext) -> bool:
    for guard in guards:
        try:
            if not await guard.check(ctx):
                return False
        except Exception:
            logger.debug(
                "Guard %s raised while filtering tools/list; hiding '%s'",
                type(guard).__name__,
                ctx.tool_name,
                exc_info=True,
            )
            return False
    return True
