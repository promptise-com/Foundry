"""Code-action reasoning node — the model writes ONE program, not a tool chain.

For aggregation / data-traversal tasks, chaining dozens of conversational tool
calls is slow, expensive, and error-prone: the transcript grows, the model
loses the thread, and it aggregates facts unreliably in its head. ``code-action``
changes the *action space* — in a single LLM turn the model writes one Python
program that calls the available tools (via a sandbox bridge) and computes the
answer deterministically. Validated on agentic tasks: large accuracy gain at a
fraction of the tokens and latency, in one turn.

Security: the model-written code runs inside Promptise's hardened Docker sandbox
(read-only rootfs, seccomp, dropped capabilities, no network, resource limits).
It can only reach the outside world through the *bridge*, which invokes the real
:class:`~langchain_core.tools.BaseTool` instances on the host. Those host tools
keep their normal protections — if ``build_agent`` wrapped them with an
**approval** gate, bridged calls trigger it too; and when the Agent Runtime has
attached governance **hooks** (budget/health/audit), each bridged call passes
through them. Independently of any hooks, the node enforces a hard
``max_tool_calls`` cap per run so a program cannot loop a tool unbounded, and
validates every bridge request id to keep file I/O inside the RPC directory.

The bridge is a filesystem rendezvous over the sandbox's writable ``/workspace``
tmpfs: the in-container program writes a request file and blocks; a concurrent
host loop services it by running the real tool and writing the response file.
This avoids granting the sandbox any network access.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import shlex
import time
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .base import BaseNode
from .nodes import TOOL_SELECTOR_KEY
from .state import GraphState, NodeResult

logger = logging.getLogger("promptise.engine.code_action")

# Only hex/uuid-style request ids are serviced (defense-in-depth: a request id
# can never contain a path separator or traversal sequence).
_SAFE_RID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Directory inside the sandbox used for the host<->program tool bridge.
_RPC_DIR = "/workspace/_rpc"
_PROGRAM_PATH = "/workspace/program.py"
_TOOLS_PATH = "/workspace/promptise_tools.py"

# The in-sandbox RPC client, prepended to the generated tools module. It writes a
# request atomically (tmp + rename) and waits for the host to write a ``.done``
# marker (written AFTER the response body, so reads never race a partial write).
_RPC_CLIENT = """
import json as _json, os as _os, time as _time, uuid as _uuid
_RPC = "%s"
_os.makedirs(_RPC, exist_ok=True)

def _call(_tool, _args):
    _rid = _uuid.uuid4().hex
    _req, _tmp = _RPC + "/req_" + _rid + ".json", _RPC + "/req_" + _rid + ".json.tmp"
    _resp, _done = _RPC + "/resp_" + _rid + ".json", _RPC + "/resp_" + _rid + ".done"
    with open(_tmp, "w") as _f:
        _json.dump({"tool": _tool, "args": _args}, _f)
    _os.rename(_tmp, _req)
    _deadline = _time.time() + %d
    while not _os.path.exists(_done):
        if _time.time() > _deadline:
            raise TimeoutError("tool call timed out: " + _tool)
        _time.sleep(0.02)
    with open(_resp) as _f:
        _data = _json.load(_f)
    if "error" in _data:
        raise RuntimeError(_data["error"])
    return _data["result"]
"""


# A factory that produces a ready sandbox session. The node owns the session
# lifecycle (creates one per run, cleans it up afterwards).
SandboxFactory = Callable[[], Awaitable[Any]]


def _tool_signature(tool: Any) -> tuple[str, list[str]]:
    """Return (name, [arg_names]) for a LangChain BaseTool."""
    name = str(getattr(tool, "name", None) or getattr(tool, "__name__", None) or "tool")
    args = getattr(tool, "args", None)
    arg_names: list[str] = [str(k) for k in args] if isinstance(args, dict) else []
    return name, arg_names


def render_api_spec(tools: list[Any]) -> str:
    """Render the available tools as a Python API the model can call."""
    if not tools:
        return "(no data functions available — compute the answer directly)"
    lines: list[str] = []
    for tool in tools:
        name, arg_names = _tool_signature(tool)
        sig = ", ".join(arg_names)
        desc = (getattr(tool, "description", "") or "").strip().splitlines()
        doc = desc[0] if desc else ""
        lines.append(f"def {name}({sig}):  # {doc}".rstrip())
    return "\n".join(lines)


def _render_tools_module(tools: list[Any], call_timeout: int) -> str:
    """Generate the in-sandbox ``promptise_tools.py`` (RPC stubs)."""
    parts = [_RPC_CLIENT % (_RPC_DIR, call_timeout)]
    for tool in tools:
        name, arg_names = _tool_signature(tool)
        sig = ", ".join(arg_names)
        arg_dict = "{" + ", ".join(f"{a!r}: {a}" for a in arg_names) + "}"
        parts.append(f"def {name}({sig}):\n    return _call({name!r}, {arg_dict})\n")
    return "\n".join(parts)


def extract_code(text: str) -> str:
    """Pull a Python program out of an LLM response (```python block or raw)."""
    if not text:
        return ""
    fence = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1).strip()
    return text.strip()


def parse_result(stdout: str, marker: str) -> str:
    """Extract the final answer from program stdout (last ``marker`` line)."""
    answer = ""
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped.startswith(marker):
            answer = stripped[len(marker) :].strip()
    if answer:
        return answer
    # Fallback: last non-empty line.
    for line in reversed(stdout.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def _extract_task(state: GraphState) -> str:
    """Get the user's latest question from the message list (objects or dicts)."""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            return content if isinstance(content, str) else str(content)
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    # Fallback: first message content.
    if state.messages:
        first = state.messages[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(getattr(first, "content", first))
    return ""


async def write_text(session: Any, path: str, content: str) -> None:
    """Write a file from *inside* the container (read-only-rootfs safe).

    Uses ``base64 -d`` so arbitrary content (quotes, newlines, shell metachars)
    can never be interpreted by the shell.
    """
    b64 = base64.b64encode(content.encode()).decode()
    cmd = f"printf %s {shlex.quote(b64)} | base64 -d > {shlex.quote(path)}"
    res = await session.execute(cmd)
    if not res.success:
        raise RuntimeError(f"sandbox write failed ({path}): {res.stderr[:200]}")


async def _read_text(session: Any, path: str) -> str | None:
    res = await session.execute(f"cat {shlex.quote(path)}")
    return res.stdout if res.success else None


class CodeActionNode(BaseNode):
    """A node that solves a task by writing and running one sandboxed program.

    Args:
        name: Node identifier.
        tools: Tools the generated program may call (bridged to the host). If
            empty, falls back to the engine's runtime tools (``_engine_tools``).
        system_prompt: Base system prompt prepended to the code-writing prompt.
        blocks: Accepted for signature parity with other prebuilts (unused in
            the program prompt — ``system_prompt`` carries the instructions).
        sandbox_factory: ``async () -> SandboxSession``. Required at run time;
            the node creates one session per run and cleans it up.
        model_override: Optional per-node model.
        max_repairs: How many times to feed a crash's stderr back for a fix.
        result_marker: stdout prefix that marks the final answer line.
        exec_timeout: Max seconds the program may run inside the sandbox.
        max_tool_calls: Hard upper bound on bridged tool calls per run. Once
            reached, further calls return an error to the program instead of
            executing — a hook-independent safety bound so a generated program
            cannot loop a tool unbounded. Set ``0`` to disable (not advised).
    """

    def __init__(
        self,
        name: str,
        *,
        tools: list[Any] | None = None,
        system_prompt: str = "",
        blocks: list[Any] | None = None,
        sandbox_factory: SandboxFactory | None = None,
        model_override: Any | None = None,
        max_repairs: int = 1,
        result_marker: str = "RESULT:",
        exec_timeout: int = 120,
        max_tool_calls: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(name, **kwargs)
        self.tools = list(tools) if tools else []
        self.system_prompt = system_prompt
        self.blocks = list(blocks) if blocks else []
        self.sandbox_factory = sandbox_factory
        self.model_override = model_override
        self.max_repairs = max_repairs
        self.result_marker = result_marker
        self.exec_timeout = exec_timeout
        self.max_tool_calls = max_tool_calls

    # ── prompt assembly ──────────────────────────────────────────────────────
    def _build_prompt(self, task: str, tools: list[Any]) -> list[Any]:
        api = render_api_spec(tools)
        system = (f"{self.system_prompt}\n\n" if self.system_prompt else "") + (
            "You answer the user's question by writing ONE Python 3 program.\n"
            "The following functions are already imported and available to call:\n\n"
            f"{api}\n\n"
            "Rules:\n"
            "- Use ONLY those functions to obtain data; never invent data values.\n"
            "- Each function returns a plain Python value (str, int, float, list, "
            "or dict) — use the returned value directly. If a return value is a "
            "string, inspect it before assuming a structure (don't call .get on a "
            "str). When unsure of the shape, handle both dict and string.\n"
            "- Do the work in Python — loops, sums, filtering, etc.\n"
            f"- Print the final answer on the last line as: {self.result_marker} <answer>\n"
            "- Output ONLY the program inside a single ```python code block, no prose.\n"
            "- Do not call input(); do not access the network."
        )
        return [SystemMessage(content=system), HumanMessage(content=f"Question: {task}")]

    # ── tool bridge ──────────────────────────────────────────────────────────
    async def _invoke_tool(
        self,
        tool_map: dict[str, Any],
        state: GraphState,
        config: dict[str, Any],
        name: str,
        args: dict[str, Any],
        result: NodeResult,
    ) -> dict[str, Any]:
        """Run one real tool on the host, through the engine's hooks."""
        # Hook-independent safety bound: a generated program cannot loop a tool
        # unbounded, regardless of whether budget governance hooks are attached.
        if self.max_tool_calls and len(result.tool_calls) >= self.max_tool_calls:
            return {"error": f"tool-call budget exceeded (max_tool_calls={self.max_tool_calls})"}
        hooks = config.get("_engine_hooks", [])
        tool = tool_map.get(name)
        if tool is None:
            return {"error": f"unknown tool: {name}"}
        try:
            for hook in hooks:
                if hasattr(hook, "pre_tool"):
                    args = await hook.pre_tool(name, args, state)
            output = await tool.ainvoke(args)
            # Preserve JSON-serializable structure so the program receives real
            # Python objects (dict/list/number/str), not a stringified blob.
            try:
                json.dumps(output)
                value: Any = output
            except (TypeError, ValueError):
                value = str(output)
            text = value if isinstance(value, str) else json.dumps(value, default=str)
            obs = {"tool": name, "args": args, "result": text, "success": True}
            state.observations.append(obs)
            result.tool_calls.append({"name": name, "args": args, "result": text})
            for hook in hooks:
                if hasattr(hook, "post_tool"):
                    await hook.post_tool(name, args, text, state)
            return {"result": value}
        except Exception as exc:  # noqa: BLE001 — surface error to the program
            result.tool_calls_failed += 1
            state.observations.append(
                {"tool": name, "args": args, "result": str(exc), "success": False}
            )
            return {"error": f"{type(exc).__name__}: {exc}"}

    async def _bridge_loop(
        self,
        session: Any,
        tool_map: dict[str, Any],
        state: GraphState,
        config: dict[str, Any],
        result: NodeResult,
        stop: asyncio.Event,
    ) -> None:
        """Service the program's tool requests until it finishes."""
        served: set[str] = set()
        while not stop.is_set():
            try:
                files = await session.list_files(_RPC_DIR)
            except Exception:  # noqa: BLE001 — dir may not exist yet
                files = []
            for fn in files:
                if not (fn.startswith("req_") and fn.endswith(".json")):
                    continue
                rid = fn[len("req_") : -len(".json")]
                # Defense-in-depth: only service safe request ids so the id can
                # never steer host file I/O outside the RPC directory.
                if not _SAFE_RID.match(rid):
                    continue
                if rid in served:
                    continue
                served.add(rid)
                raw = await _read_text(session, f"{_RPC_DIR}/{fn}")
                if raw is None:
                    served.discard(rid)
                    continue
                try:
                    req = json.loads(raw)
                except json.JSONDecodeError:
                    served.discard(rid)
                    continue
                resp = await self._invoke_tool(
                    tool_map, state, config, req.get("tool", ""), req.get("args", {}), result
                )
                await write_text(session, f"{_RPC_DIR}/resp_{rid}.json", json.dumps(resp))
                await write_text(session, f"{_RPC_DIR}/resp_{rid}.done", "")
            await asyncio.sleep(0.03)

    # ── main pipeline ────────────────────────────────────────────────────────
    async def execute(self, state: GraphState, config: dict[str, Any]) -> NodeResult:
        start = time.monotonic()
        result = NodeResult(node_name=self.name, node_type="CodeActionNode")

        model = self.model_override or config.get("_engine_model")
        if model is None:
            result.error = "No model available (_engine_model missing)"
            result.duration_ms = (time.monotonic() - start) * 1000
            return result
        if self.sandbox_factory is None:
            result.error = (
                "code-action requires a sandbox. Build the agent with sandbox=True "
                "(Docker must be installed and running)."
            )
            result.duration_ms = (time.monotonic() - start) * 1000
            return result

        tools = self.tools or config.get("_engine_tools", []) or []
        tool_map = {getattr(t, "name", ""): t for t in tools}
        # The run's tool selector (semantic tool selection) narrows the API the
        # prompt describes; the bridge still serves every tool.
        offered = tools
        selector = config.get(TOOL_SELECTOR_KEY)
        if selector is not None and tools:
            try:
                offered = list(selector(tools, state))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Tool selector failed in node %r: %s", self.name, exc)
        task = _extract_task(state)
        messages = self._build_prompt(task, offered)

        # 1. Generate the program (1 LLM turn; +1 per repair).
        code = ""
        for attempt in range(self.max_repairs + 1):
            llm_start = time.monotonic()
            try:
                response = await model.ainvoke(messages, config=config)
            except Exception as exc:  # noqa: BLE001
                result.error = f"LLM call failed: {type(exc).__name__}: {exc}"
                result.duration_ms = (time.monotonic() - start) * 1000
                return result
            result.llm_duration_ms += (time.monotonic() - llm_start) * 1000
            usage = getattr(response, "usage_metadata", None)
            if usage:
                result.prompt_tokens += getattr(usage, "input_tokens", 0) or 0
                result.completion_tokens += getattr(usage, "output_tokens", 0) or 0
            raw = response.content if isinstance(response.content, str) else str(response.content)
            code = extract_code(raw)

            # 2. Run it in the sandbox (bridging tool calls back to the host).
            run = await self._run_program(code, tool_map, state, config, result)
            if run.success:
                answer = parse_result(run.stdout, self.result_marker)
                result.total_tokens = result.prompt_tokens + result.completion_tokens
                result.raw_output = answer
                result.output = answer
                ai = AIMessage(content=answer)
                state.messages.append(ai)
                result.messages_added.append(ai)
                result.transition_reason = f"code-action solved in {attempt + 1} attempt(s)"
                result.duration_ms = (time.monotonic() - start) * 1000
                return result

            # 3. Repair: feed the failure back and try again.
            if attempt < self.max_repairs:
                messages = messages + [
                    AIMessage(content=raw),
                    HumanMessage(
                        content=(
                            "Your program failed with this error — fix it and output "
                            f"only the corrected program:\n\n{run.stderr[-800:]}"
                        )
                    ),
                ]

        # All attempts failed.
        result.total_tokens = result.prompt_tokens + result.completion_tokens
        result.error = f"program failed after {self.max_repairs + 1} attempt(s)"
        result.raw_output = code
        result.output = ""
        result.duration_ms = (time.monotonic() - start) * 1000
        return result

    async def _run_program(
        self,
        code: str,
        tool_map: dict[str, Any],
        state: GraphState,
        config: dict[str, Any],
        result: NodeResult,
    ) -> Any:
        """Write + execute the program in a fresh sandbox session, bridging tools."""
        session = await self.sandbox_factory()  # type: ignore[misc]
        try:
            await session.execute(f"mkdir -p {_RPC_DIR}")
            await write_text(
                session,
                _TOOLS_PATH,
                _render_tools_module(list(tool_map.values()), self.exec_timeout),
            )
            program = "from promptise_tools import *  # noqa\n\n" + code
            await write_text(session, _PROGRAM_PATH, program)

            stop = asyncio.Event()
            prog_task = asyncio.create_task(
                session.execute(f"python3 {_PROGRAM_PATH}", timeout=self.exec_timeout)
            )
            bridge_task = asyncio.create_task(
                self._bridge_loop(session, tool_map, state, config, result, stop)
            )
            tool_start = time.monotonic()
            cmd = await prog_task
            stop.set()
            await bridge_task
            result.tool_duration_ms += (time.monotonic() - tool_start) * 1000
            logger.debug("code-action program:\n%s", program)
            logger.debug(
                "code-action exit=%s stdout=%r stderr=%r",
                cmd.exit_code,
                cmd.stdout[:600],
                cmd.stderr[:600],
            )
            return cmd
        finally:
            try:
                await session.cleanup()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                logger.warning("sandbox cleanup failed", exc_info=True)
