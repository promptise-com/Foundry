# promptise/cross_agent.py
"""
Cross-agent communication utilities for Foundry.

Expose other in-process agents (“peers”) as standard LangChain tools so a
primary (caller) agent can *delegate* to them during planning/execution.

Tools provided
    - Per-peer ask tool  →  ``ask_agent_<name>``
      Forward one message (plus optional caller context) to a single peer and
      return the peer’s final text.

    - Broadcast tool     →  ``broadcast_to_agents`` (opt-in)
      Send the same message to multiple peers in parallel and return a mapping
      of peer → final text. Timeouts/errors are captured per peer so one slow
      or failing peer does not fail the whole call.

Notes:
    - No new infrastructure is required. Peers are just in-process LangChain
      ``Runnable`` graphs (e.g., a DeepAgents loop or a LangGraph prebuilt
      executor returned by :func:`promptise.agent.build_agent`).
    - Both tool classes implement async and sync execution paths (``_arun`` and
      ``_run``) to satisfy ``BaseTool``’s interface.
    - Timeouts are set in Python — per peer (:attr:`CrossAgent.timeout`) or for
      all peers (``timeout=`` on :func:`make_cross_agent_tools`) — never by the
      model.
    - Delegation is bounded: every hop is recorded in a context variable, a
      call deeper than ``max_delegation_depth`` or back into a peer that is
      already working on the request raises :class:`DelegationError`.
    - The “final text” is extracted from common agent result shapes. If your
      peer returns a custom structure, adapt upstream or post-process the
      returned string.

Examples:
    Build a peer agent and attach it to a main agent as a tool:

    >>> from promptise.agent import build_agent
    >>> from promptise.cross_agent import CrossAgent
    >>>
    >>> peer_graph = await build_agent(servers=..., model="openai:gpt-5-mini")
    >>> main_graph = await build_agent(
    ...     servers=...,
    ...     model="openai:gpt-4.1",
    ...     cross_agents={"researcher": CrossAgent(agent=peer_graph, description="Web research")}
    ... )
    >>> # Now the main agent can call:
    >>> #   - ask_agent_researcher(message=..., context?=...)
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from .identity import AgentIdentity


DEFAULT_MAX_DELEGATION_DEPTH = 3
"""Default limit on nested delegations (coordinator → peer → peer → peer)."""

TIMEOUT_TEXT = "Timed out waiting for peer agent reply."

# The peers working on the current request, outermost first, as
# ``(peer name, id(peer runnable))``. Each delegation pushes one entry for the
# duration of the peer call; tasks spawned inside the peer copy the context,
# so the chain follows the request through every nested agent.
_delegation_chain: contextvars.ContextVar[tuple[tuple[str, int], ...]] = contextvars.ContextVar(
    "promptise_delegation_chain", default=()
)


class DelegationError(RuntimeError):
    """A cross-agent call was refused before it reached the peer.

    Raised by the ``ask_agent_<name>`` tool when the call would exceed
    ``max_delegation_depth`` or would delegate back into a peer that is
    already working on the same request (a loop). The agent sees the message
    as the tool's error and can answer with what it has.
    """


def get_delegation_chain() -> tuple[str, ...]:
    """Return the names of the peers working on the current request.

    Outermost first; empty outside a cross-agent call. A peer reads its own
    name last.
    """
    return tuple(name for name, _ in _delegation_chain.get())


def _caller_identity_message(
    identity: AgentIdentity | None,
) -> dict[str, str] | None:
    """Build a system message announcing the delegating agent's identity.

    Lets a peer agent know *who is asking* so it can attribute (and, if it
    chooses, authorize) the delegation. Uses the caller's cheap identity
    descriptors (``claims()``) — never a credential token.
    """
    if identity is None:
        return None
    claims = identity.claims()
    who = claims.get("agent_id")
    if who is None and claims.get("credential_provider"):
        who = f"verifiable via {claims['credential_provider']}"
    return {
        "role": "system",
        "content": f"Delegated by agent: {who or 'unknown'}. Caller identity: {claims}",
    }


# -----------------------------
# Public API surface
# -----------------------------


@dataclass(frozen=True)
class CrossAgent:
    """Metadata wrapper for a peer agent to be exposed as a tool.

    The wrapper is descriptive only. Behavior is implemented by tools produced
    via :func:`make_cross_agent_tools`.

    Attributes:
        agent: A runnable agent (e.g., LangGraph or DeepAgents) that accepts
            ``{"messages": [...]}`` and returns a result consumable by the
            built-in “best final text” extractor.
        description: One-line human description used in tool docs to help the
            calling agent decide when to delegate.
        timeout: Seconds to wait for this peer's answer. ``None`` uses the
            ``timeout`` given to :func:`make_cross_agent_tools` (or
            ``delegation_timeout`` on :func:`~promptise.agent.build_agent`);
            when both are ``None`` the call has no limit of its own.

    Examples:
        >>> cross = CrossAgent(agent=peer_graph, description="Accurate math", timeout=60)
    """

    agent: Runnable[Any, Any]
    description: str = ""
    timeout: float | None = None


ToolHook = Callable[..., Any]


def make_cross_agent_tools(
    peers: Mapping[str, CrossAgent],
    *,
    tool_name_prefix: str = "ask_agent_",
    include_broadcast: bool = True,
    caller_identity: AgentIdentity | None = None,
    timeout: float | None = None,
    max_delegation_depth: int = DEFAULT_MAX_DELEGATION_DEPTH,
    on_before: ToolHook | None = None,
    on_after: ToolHook | None = None,
    on_error: ToolHook | None = None,
    requires_approval: Callable[[str], bool] | None = None,
) -> list[BaseTool]:
    """Create LangChain tools for cross-agent communication.

    For each peer, a tool named ``f"{tool_name_prefix}{peer_name}"`` is created.
    Optionally, a ``broadcast_to_agents`` tool is added to fan-out questions to
    multiple peers concurrently.

    Args:
        peers: Mapping of peer name → :class:`CrossAgent`. The name becomes part
            of the tool id (e.g., ``ask_agent_mathpeer``).
        tool_name_prefix: Prefix used for each per-peer ask tool. Defaults to
            ``"ask_agent_"``.
        include_broadcast: When ``True`` (default here; ``build_agent`` defaults
            to ``False``), also include the group fan-out tool
            ``broadcast_to_agents``.
        caller_identity: Identity of the delegating agent, announced to peers.
        timeout: Seconds to wait for a peer that sets no
            :attr:`CrossAgent.timeout` of its own. ``None`` = no limit.
        max_delegation_depth: Most nested delegations allowed in one request
            (``1`` = this agent may ask a peer, but that peer may not
            delegate further). Must be at least 1.
        on_before: ``(tool_name, arguments)`` called before each peer call
            (used for ``trace_tools`` and observability).
        on_after: ``(tool_name, result_text)`` called after a successful call.
        on_error: ``(tool_name, exception)`` called when a call fails or is
            refused.
        requires_approval: The approval check of the agent that gets these
            tools (``ApprovalPolicy.requires_approval``), called with a tool
            name. ``broadcast_to_agents`` does not reach a peer whose
            ``ask_agent_<name>`` tool needs approval -- unless the broadcast
            tool needs approval itself -- because the broadcast would reach
            that peer without one. ``build_agent`` passes its ``approval``
            policy here.

    Returns:
        list[BaseTool]: A list of fully constructed tools ready to be appended
        to the caller agent’s toolset.

    Raises:
        ValueError: If ``max_delegation_depth`` is less than 1 or a timeout is
            not positive.

    Notes:
        Construction does not contact peers; errors (e.g., network) surface at
        call time during execution of the generated tools.

    Examples:
        >>> tools = make_cross_agent_tools({
        ...     "researcher": CrossAgent(agent=peer_graph, description="Web research")
        ... })
        >>> # Attach `tools` alongside your MCP-discovered tools when building the agent.
    """
    if max_delegation_depth < 1:
        raise ValueError(f"max_delegation_depth must be at least 1, got {max_delegation_depth}")
    if timeout is not None and timeout <= 0:
        raise ValueError(f"timeout must be positive, got {timeout}")
    for name, spec in peers.items():
        if spec.timeout is not None and spec.timeout <= 0:
            raise ValueError(f"CrossAgent '{name}': timeout must be positive, got {spec.timeout}")
    if not peers:
        return []

    def _best_text(result: Any) -> str:
        """Extract a final text answer from common agent result shapes.

        Looks for a LangGraph-like ``{"messages": [...]}`` structure; otherwise
        falls back to ``str(result)``.

        Args:
            result: The raw result returned by a peer agent.

        Returns:
            str: Best-effort final text response.
        """
        try:
            if isinstance(result, dict) and "messages" in result and result["messages"]:
                last = result["messages"][-1]
                content = getattr(last, "content", None)
                if content is None and isinstance(last, dict):
                    content = last.get("content")
                if isinstance(content, str) and content:
                    return content
                if isinstance(content, list) and content and isinstance(content[0], dict):
                    return cast(str, content[0].get("text") or str(content))
                return str(last)
            return str(result)
        except Exception:
            return str(result)

    delegate = _Delegator(
        extract=_best_text,
        caller_identity=caller_identity,
        default_timeout=timeout,
        max_depth=max_delegation_depth,
    )
    hooks = _Hooks(on_before, on_after, on_error)

    out: list[BaseTool] = []

    # Per-agent ask tools
    for name, spec in peers.items():
        out.append(
            _AskAgentTool(
                name=f"{tool_name_prefix}{name}",
                description=(
                    f"Ask peer agent '{name}' for help. " + (spec.description or "")
                ).strip(),
                peer_name=name,
                peer=spec,
                delegate=delegate,
                hooks=hooks,
            )
        )

    # Optional broadcast tool
    if include_broadcast:
        # Peers gated by approval stay reachable only through their own,
        # approval-wrapped ask tool.
        gated: dict[str, str] = {}
        if requires_approval is not None and not requires_approval("broadcast_to_agents"):
            for name in peers:
                ask_name = f"{tool_name_prefix}{name}"
                if requires_approval(ask_name):
                    gated[name] = ask_name
        out.append(
            _BroadcastTool(
                name="broadcast_to_agents",
                description=(
                    "Ask multiple peer agents the same question in parallel and "
                    "return each peer's final answer."
                ),
                peers=peers,
                delegate=delegate,
                hooks=hooks,
                gated=gated,
            )
        )

    return out


# -----------------------------
# Shared delegation logic
# -----------------------------


@dataclass(frozen=True)
class _Hooks:
    on_before: ToolHook | None = None
    on_after: ToolHook | None = None
    on_error: ToolHook | None = None

    @staticmethod
    def _call(hook: ToolHook | None, *args: Any) -> None:
        # Tracing must never break delegation.
        if hook is not None:
            try:
                hook(*args)
            except Exception:  # pragma: no cover - defensive
                pass

    def before(self, tool: str, args: dict[str, Any]) -> None:
        self._call(self.on_before, tool, args)

    def after(self, tool: str, result: Any) -> None:
        self._call(self.on_after, tool, result)

    def error(self, tool: str, exc: Exception) -> None:
        self._call(self.on_error, tool, exc)


@dataclass(frozen=True)
class _Delegator:
    """Forward one message to one peer, with the limits every call shares."""

    extract: Callable[[Any], str]
    caller_identity: AgentIdentity | None
    default_timeout: float | None
    max_depth: int

    def check(self, name: str, peer: CrossAgent) -> None:
        """Raise :class:`DelegationError` if calling *peer* now is not allowed."""
        chain = _delegation_chain.get()
        path = " → ".join([*(n for n, _ in chain), name])
        if len(chain) >= self.max_depth:
            raise DelegationError(
                f"Delegation to '{name}' refused: it would be delegation level "
                f"{len(chain) + 1} ({path}), over max_delegation_depth={self.max_depth}. "
                "Answer with the information you already have."
            )
        if any(pid == id(peer.agent) for _, pid in chain):
            raise DelegationError(
                f"Delegation to '{name}' refused: '{name}' is already working on this "
                f"request ({path}), so the call would loop. "
                "Answer with the information you already have."
            )

    async def __call__(self, name: str, peer: CrossAgent, messages: list[dict[str, Any]]) -> str:
        """Check the limits, call *peer* and return its final text.

        Returns :data:`TIMEOUT_TEXT` when the timeout elapses. Exceptions from
        the peer propagate.
        """
        self.check(name, peer)

        payload: list[dict[str, Any]] = []
        # Announce the delegating agent's identity so the peer knows who asks.
        identity_msg = _caller_identity_message(self.caller_identity)
        if identity_msg:
            payload.append(identity_msg)
        payload.extend(messages)

        timeout = peer.timeout if peer.timeout is not None else self.default_timeout

        # Carry the delegating agent's identity into the peer's run so the
        # peer's observability records who delegated (structured, beyond the
        # LLM-visible system message above).
        # Local import to avoid an import cycle (observability imports back into
        # this module's chain). Resolved at call time, after both modules load.
        from .observability import _delegation_ctx_var

        chain_token = _delegation_chain.set((*_delegation_chain.get(), (name, id(peer.agent))))
        # Always set it -- to None when this agent has no identity -- so a
        # nested hop never reports an outer agent as the one delegating.
        token = _delegation_ctx_var.set(
            self.caller_identity.claims() if self.caller_identity is not None else None
        )
        try:
            res = None
            if timeout:
                import anyio

                with anyio.move_on_after(timeout) as scope:
                    res = await peer.agent.ainvoke({"messages": payload})
                if scope.cancelled_caught:
                    return TIMEOUT_TEXT
            else:
                res = await peer.agent.ainvoke({"messages": payload})
        finally:
            _delegation_ctx_var.reset(token)
            _delegation_chain.reset(chain_token)

        if res is None:
            return "No response from peer agent."
        return self.extract(res)


def _messages(message: str, context: str | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # Put context first to bias some executors that read system first
    if context:
        out.append({"role": "system", "content": f"Caller context: {context}"})
    out.append({"role": "user", "content": message})
    return out


# -----------------------------
# Tool implementations
# -----------------------------


class _AskArgs(BaseModel):
    """Arguments for per-peer ask tools (``ask_agent_<name>``).

    Attributes:
        message: The user-level message to forward to the peer agent.
        context: Optional caller context (constraints, partial results, style
            guide). If provided, it is inserted first as a *system* message to
            bias many executors.
    """

    message: str = Field(..., description="Message to send to the peer agent.")
    context: str | None = Field(
        None,
        description=(
            "Optional additional context from the caller (e.g., hints, partial "
            "results, or constraints)."
        ),
    )


class _AskAgentTool(BaseTool):
    """Tool that forwards a question to a specific peer agent.

    This tool wraps a peer :class:`~langchain_core.runnables.Runnable` and
    returns the peer’s *final text* using a best-effort extractor.

    Attributes:
        name: Tool identifier (e.g., ``ask_agent_researcher``).
        description: Human description to guide the caller agent’s planning.
        args_schema: Pydantic model describing accepted keyword args.

    Notes:
        - Async-first: prefer ``_arun``; a sync shim (``_run``) is provided to
          satisfy the abstract base class and support sync-only executors.
        - The peer is invoked with a ChatML-like payload:
          ``{"messages": [{"role": "...", "content": "..."}]}``.

    """

    name: str
    description: str
    # Pydantic v2 requires a type annotation for field overrides.
    args_schema: type[BaseModel] = _AskArgs

    _peer_name: str = PrivateAttr()
    _peer: CrossAgent = PrivateAttr()
    _delegate: _Delegator = PrivateAttr()
    _hooks: _Hooks = PrivateAttr()

    def __init__(
        self,
        *,
        name: str,
        description: str,
        peer_name: str,
        peer: CrossAgent,
        delegate: _Delegator,
        hooks: _Hooks,
    ) -> None:
        """Initialize the ask tool.

        Args:
            name: Tool identifier.
            description: Human description for planning.
            peer_name: The peer's name in the ``cross_agents`` mapping.
            peer: The peer to call.
            delegate: Shared limits, identity and result extraction.
            hooks: Trace/observability callbacks.
        """
        super().__init__(name=name, description=description)
        self._peer_name = peer_name
        self._peer = peer
        self._delegate = delegate
        self._hooks = hooks

    async def _arun(self, *, message: str, context: str | None = None) -> str:
        """Asynchronously forward a message to the peer agent.

        Args:
            message: The message to forward (becomes a user message).
            context: Optional caller context, sent first as a system message.

        Returns:
            str: The peer agent’s best-effort final text answer, or a timeout
            message if the peer's timeout elapses.

        Raises:
            DelegationError: If the call would exceed ``max_delegation_depth``
                or loop back into a peer already working on the request.
            Exception: Propagates exceptions raised by the peer call.
        """
        args: dict[str, Any] = {"message": message}
        if context:
            args["context"] = context
        self._hooks.before(self.name, args)
        try:
            text = await self._delegate(self._peer_name, self._peer, _messages(message, context))
        except Exception as exc:
            self._hooks.error(self.name, exc)
            raise
        self._hooks.after(self.name, text)
        return text

    def _run(
        self, *, message: str, context: str | None = None
    ) -> str:  # pragma: no cover (usually unused in async apps)
        """Synchronous shim that delegates to :meth:`_arun`.

        Args:
            message: The message to forward (becomes a user message).
            context: Optional caller context, sent first as a system message.

        Returns:
            str: The peer agent’s best-effort final text answer (or timeout text).
        """
        import anyio

        return anyio.run(lambda: self._arun(message=message, context=context))


class _BroadcastArgs(BaseModel):
    """Arguments for the broadcast tool (``broadcast_to_agents``).

    Attributes:
        message: The shared message sent to all (or a subset of) peers.
        context: Optional caller context, sent to every peer as a system
            message before the message.
        peers: Optional subset of peer names to consult. If omitted, all
            registered peers are consulted.
    """

    message: str = Field(..., description="Message to send to all/selected peers.")
    context: str | None = Field(
        None,
        description=(
            "Optional additional context from the caller (e.g., hints, partial "
            "results, or constraints), sent to every peer."
        ),
    )
    peers: Sequence[str] | None = Field(
        None, description="Optional subset of peer names. If omitted, use all peers."
    )


class _BroadcastTool(BaseTool):
    """Ask multiple peer agents in parallel and return a mapping of answers.

    Each selected peer is invoked concurrently. Timeouts, refusals and
    exceptions are captured **per peer** so the overall call remains resilient.

    Attributes:
        name: Tool identifier (``broadcast_to_agents``).
        description: Human description for planning.
        args_schema: Pydantic model describing accepted keyword args.

    Notes:
        Uses ``anyio.create_task_group`` for compatibility across anyio versions.
    """

    name: str
    description: str
    # Pydantic v2 requires a type annotation for field overrides.
    args_schema: type[BaseModel] = _BroadcastArgs

    _peers: Mapping[str, CrossAgent] = PrivateAttr()
    _delegate: _Delegator = PrivateAttr()
    _hooks: _Hooks = PrivateAttr()
    _gated: Mapping[str, str] = PrivateAttr()

    def __init__(
        self,
        *,
        name: str,
        description: str,
        peers: Mapping[str, CrossAgent],
        delegate: _Delegator,
        hooks: _Hooks,
        gated: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize the broadcast tool.

        Args:
            name: Tool identifier.
            description: Human description for planning.
            peers: Mapping of peer name → :class:`CrossAgent`.
            delegate: Shared limits, identity and result extraction.
            hooks: Trace/observability callbacks.
            gated: Peer name → its ask tool, for peers that need approval:
                the broadcast does not call them.
        """
        super().__init__(name=name, description=description)
        self._peers = peers
        self._delegate = delegate
        self._hooks = hooks
        self._gated = dict(gated or {})

    async def _arun(
        self,
        *,
        message: str,
        context: str | None = None,
        peers: Sequence[str] | None = None,
    ) -> dict[str, str]:
        """Asynchronously consult multiple peers in parallel.

        Args:
            message: The message forwarded to each selected peer.
            context: Optional caller context, sent to every peer.
            peers: Optional subset of peer names to target. If ``None``, uses all.

        Returns:
            dict[str, str]: Mapping of ``peer_name`` → final text. Peers that
            exceed their timeout return ``"Timed out"``. Peers that raise (or
            are refused by the depth/loop limits, or need approval) return
            ``"Error: <message>"``.

        Raises:
            ValueError: If any requested peer name is unknown.
        """
        selected: Iterable[tuple[str, CrossAgent]]
        if peers:
            missing = [p for p in peers if p not in self._peers]
            if missing:
                raise ValueError(f"Unknown peer(s): {', '.join(missing)}")
            selected = [(p, self._peers[p]) for p in peers]
        else:
            selected = list(self._peers.items())

        import anyio

        args: dict[str, Any] = {"message": message}
        if context:
            args["context"] = context
        if peers:
            args["peers"] = list(peers)
        self._hooks.before(self.name, args)

        results: dict[str, str] = {}
        msgs = _messages(message, context)

        async def _one(name: str, spec: CrossAgent) -> None:
            if name in self._gated:
                results[name] = (
                    f"Error: '{name}' needs approval, so it cannot be asked in a "
                    f"broadcast. Use {self._gated[name]}."
                )
                return
            try:
                text = await self._delegate(name, spec, msgs)
            except Exception as exc:  # keep broadcast resilient
                results[name] = f"Error: {exc}"
                return
            results[name] = "Timed out" if text == TIMEOUT_TEXT else text

        # Using TaskGroup for compatibility across anyio versions. Each task
        # copies the current context, so each peer sees the delegation chain
        # and the caller's identity.
        async with anyio.create_task_group() as tg:
            for n, s in selected:
                tg.start_soon(_one, n, s)

        self._hooks.after(self.name, results)
        return results

    def _run(
        self,
        *,
        message: str,
        context: str | None = None,
        peers: Sequence[str] | None = None,
    ) -> dict[str, str]:  # pragma: no cover (usually unused in async apps)
        """Synchronous shim that delegates to :meth:`_arun`.

        Args:
            message: The message forwarded to each selected peer.
            context: Optional caller context, sent to every peer.
            peers: Optional subset of peer names to target. If ``None``, uses all.

        Returns:
            dict[str, str]: Mapping of ``peer_name`` → final text (or timeout/error text).
        """
        import anyio

        return anyio.run(lambda: self._arun(message=message, context=context, peers=peers))
