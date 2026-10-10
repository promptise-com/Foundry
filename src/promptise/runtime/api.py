"""Agent Orchestration API — REST API for managing AgentRuntime.

Exposes lifecycle management, configuration updates, human communication,
observability, and batch operations as HTTP endpoints.  Sits on top of
an existing :class:`AgentRuntime` and translates HTTP requests to
Python method calls.

Example::

    from promptise.runtime import AgentRuntime
    from promptise.runtime.api import OrchestrationAPI

    runtime = AgentRuntime()
    api = OrchestrationAPI(
        runtime,
        host="0.0.0.0",
        port=9100,
        auth_token="${ORCHESTRATION_API_TOKEN}",
    )
    await api.start()

Security model:

* Every route except ``GET /api/v1/health`` requires
  ``Authorization: Bearer <auth_token>`` when ``auth_token`` is set.  A
  non-loopback ``host`` without ``auth_token`` is refused at construction.
* On a loopback bind, requests whose ``Host`` header is not a loopback
  name are refused with ``421`` (DNS rebinding), and requests carrying a
  cross-site ``Origin`` with ``403``, so a web page open in a browser on
  the same machine can't drive the runtime.
* Request bodies must be JSON objects (``400`` otherwise).
* ``PATCH`` on ``budget``, ``health`` and ``mission`` validates the patched
  section against its config model (limits, strict JSON types, no unknown
  fields) and answers ``422`` without changing anything when it fails.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import types
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Union, get_args, get_origin

from aiohttp import web
from pydantic import BaseModel, ValidationError

from ._http_guard import bearer_token_matches, is_loopback, loopback_request_problem

logger = logging.getLogger("promptise.runtime.api")

__all__ = ["OrchestrationAPI"]

# Process name validation: alphanumeric + hyphens + underscores, max 64 chars
_PROCESS_NAME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9_-]{0,63}$")

#: Routes that answer without a bearer token (load-balancer probes).
_PUBLIC_PATHS = frozenset({"/api/v1/health"})

#: Methods whose request body is checked to be a JSON object.
_BODY_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})


def _json_dumps(obj: Any) -> str:
    """JSON serializer that handles non-standard types."""
    return json.dumps(obj, default=str, ensure_ascii=False)


def _json_response(data: Any, status: int = 200) -> web.Response:
    """Create a JSON response."""
    return web.Response(
        text=_json_dumps(data),
        content_type="application/json",
        status=status,
    )


def _error_response(code: str, message: str, status: int = 400) -> web.Response:
    """Create a structured error response."""
    return _json_response(
        {"error": {"code": code, "message": message}},
        status=status,
    )


def _unknown_fields(
    model_cls: type[BaseModel], data: Mapping[str, Any], path: str = ""
) -> list[str]:
    """Dotted names of keys in *data* that *model_cls* does not declare.

    Recurses into nested models (``escalation``) and dicts of models
    (``tool_costs``), which pydantic would otherwise silently drop.
    """
    unknown: list[str] = []
    for key, value in data.items():
        field = model_cls.model_fields.get(key)
        if field is None:
            unknown.append(f"{path}{key}")
            continue
        if not isinstance(value, Mapping):
            continue
        annotation = field.annotation
        is_union = get_origin(annotation) in (Union, types.UnionType)
        for kind in get_args(annotation) if is_union else (annotation,):
            if isinstance(kind, type) and issubclass(kind, BaseModel):
                unknown += _unknown_fields(kind, value, f"{path}{key}.")
            elif get_origin(kind) is dict:
                item_type = get_args(kind)[1]
                if isinstance(item_type, type) and issubclass(item_type, BaseModel):
                    for name, item in value.items():
                        if isinstance(item, Mapping):
                            unknown += _unknown_fields(item_type, item, f"{path}{key}.{name}.")
    return unknown


def _apply_patch(section: BaseModel, body: dict[str, Any]) -> web.Response | None:
    """Validate *body* against *section*'s model, then apply it in place.

    The patched section is validated as a whole (strict JSON types, the
    model's limits) before anything changes, so a bad value leaves the
    live config untouched.  Fields are set on the existing object, which
    the running process keeps reading.

    Returns:
        A ``422`` error response, or ``None`` once the patch is applied.
    """
    model_cls = type(section)
    unknown = _unknown_fields(model_cls, body)
    if unknown:
        return _error_response(
            "UNKNOWN_FIELD",
            f"Unknown field(s): {', '.join(unknown)}. Allowed: {', '.join(model_cls.model_fields)}",
            422,
        )
    merged = {**section.model_dump(mode="json"), **body}
    try:
        validated = model_cls.model_validate_json(json.dumps(merged), strict=True)
    except ValidationError as exc:
        detail = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        return _error_response("INVALID_CONFIG", f"Config validation failed: {detail}", 422)
    for key in body:
        setattr(section, key, getattr(validated, key))
    return None


class OrchestrationAPI:
    """REST API server for managing an AgentRuntime.

    Args:
        runtime: The :class:`AgentRuntime` to manage.
        host: Bind address (default ``127.0.0.1``).
        port: Port to listen on (default ``9100``).
        auth_token: Bearer token required on every route except
            ``GET /api/v1/health``. **Required** when ``host`` is not a
            loopback address.

    Raises:
        ValueError: ``auth_token`` is empty, or ``host`` is not a loopback
            address and there is no ``auth_token``.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        host: str = "127.0.0.1",
        port: int = 9100,
        auth_token: str | None = None,
    ) -> None:
        # Resolve env var in auth_token
        if auth_token and auth_token.startswith("${") and auth_token.endswith("}"):
            import os

            var_name = auth_token[2:-1].split(":-")[0]
            auth_token = os.environ.get(var_name, "")
            if not auth_token:
                raise ValueError(f"Environment variable '{var_name}' not set for auth_token")

        if auth_token is not None and not auth_token.strip():
            raise ValueError("OrchestrationAPI: auth_token must be a non-empty string (or None)")
        self._loopback = is_loopback(host)
        if not self._loopback and not auth_token:
            raise ValueError(
                "OrchestrationAPI: auth_token is required when binding to "
                f"non-localhost address '{host}'. Set auth_token to secure "
                "the management API."
            )

        self._runtime = runtime
        self._host = host
        self._port = port
        self._auth_token = auth_token

        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None

    @property
    def port(self) -> int:
        """The port the API listens on.

        With ``port=0`` the OS picks a free port; after :meth:`start` this
        returns the port actually bound.
        """
        if self._port == 0 and self._runner is not None:
            for address in self._runner.addresses:
                if isinstance(address, tuple) and len(address) >= 2:
                    return int(address[1])
        return self._port

    def _check_auth(self, request: web.Request) -> bool:
        """Verify Bearer token (timing-safe)."""
        if self._auth_token is None:
            return True  # Loopback bind without a token
        return bearer_token_matches(request.headers.get("Authorization", ""), self._auth_token)

    @web.middleware
    async def _guard(
        self,
        request: web.Request,
        handler: Callable[[web.Request], Awaitable[web.StreamResponse]],
    ) -> web.StreamResponse:
        """Host/Origin checks (loopback binds), auth and JSON-object bodies."""
        if self._loopback:
            problem = loopback_request_problem(
                request.headers.get("Host", ""), request.headers.get("Origin")
            )
            if problem is not None:
                status, message = problem
                code = "MISDIRECTED_REQUEST" if status == 421 else "CROSS_ORIGIN_REFUSED"
                return _error_response(code, message, status)
        if request.path not in _PUBLIC_PATHS and not self._check_auth(request):
            return _error_response("UNAUTHORIZED", "Invalid or missing auth token", 401)
        if request.method in _BODY_METHODS and request.can_read_body:
            raw = await request.read()
            if raw.strip():
                try:
                    body = json.loads(raw)
                except ValueError:
                    return _error_response("INVALID_JSON", "Request body must be valid JSON", 400)
                if not isinstance(body, dict):
                    return _error_response(
                        "INVALID_JSON", "Request body must be a JSON object", 400
                    )
        return await handler(request)

    async def _json_body(self, request: web.Request) -> dict[str, Any]:
        """The request's JSON object body (``{}`` when the body is empty).

        :meth:`_guard` has already refused bodies that aren't JSON objects.
        """
        raw = await request.read()
        if not raw.strip():
            return {}
        body = json.loads(raw)
        return body if isinstance(body, dict) else {}

    def _require_auth(self, request: web.Request) -> web.Response | None:
        """Return error response if auth fails, None if OK."""
        if not self._check_auth(request):
            return _error_response("UNAUTHORIZED", "Invalid or missing auth token", 401)
        return None

    def _get_process(self, name: str) -> Any:
        """Get a process by name, raise KeyError if not found."""
        return self._runtime.get_process(name)

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the API server."""
        self._app = web.Application(middlewares=[self._guard])
        self._setup_routes()

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self._host, self._port)
        await site.start()
        logger.info(
            "OrchestrationAPI started on %s:%d (%s)",
            self._host,
            self.port,
            "authenticated" if self._auth_token else "no auth (localhost)",
        )

    async def stop(self) -> None:
        """Stop the API server."""
        if self._runner:
            await self._runner.cleanup()
            self._runner = None
        self._app = None
        logger.info("OrchestrationAPI stopped")

    def _setup_routes(self) -> None:
        """Register all API routes."""
        assert self._app is not None
        r = self._app.router

        # Health (no auth — for load balancers)
        r.add_get("/api/v1/health", self._handle_health)

        # Runtime management
        r.add_get("/api/v1/runtime/status", self._handle_runtime_status)
        r.add_post("/api/v1/runtime/start-all", self._handle_start_all)
        r.add_post("/api/v1/runtime/stop-all", self._handle_stop_all)

        # Process lifecycle
        r.add_get("/api/v1/processes", self._handle_list_processes)
        r.add_post("/api/v1/processes", self._handle_deploy)
        r.add_get("/api/v1/processes/{name}", self._handle_get_process)
        r.add_delete("/api/v1/processes/{name}", self._handle_remove_process)
        r.add_post("/api/v1/processes/{name}/start", self._handle_start)
        r.add_post("/api/v1/processes/{name}/stop", self._handle_stop)
        r.add_post("/api/v1/processes/{name}/restart", self._handle_restart)

        # Config updates
        r.add_patch("/api/v1/processes/{name}/instructions", self._handle_update_instructions)
        r.add_patch("/api/v1/processes/{name}/budget", self._handle_update_budget)
        r.add_patch("/api/v1/processes/{name}/health", self._handle_update_health)
        r.add_patch("/api/v1/processes/{name}/mission", self._handle_update_mission)

        # Human communication
        r.add_post("/api/v1/processes/{name}/messages", self._handle_send_message)
        r.add_post("/api/v1/processes/{name}/ask", self._handle_ask)
        r.add_get("/api/v1/processes/{name}/inbox", self._handle_get_inbox)
        r.add_delete("/api/v1/processes/{name}/inbox", self._handle_clear_inbox)

        # Trigger management
        r.add_get("/api/v1/processes/{name}/triggers", self._handle_list_triggers)
        r.add_post("/api/v1/processes/{name}/triggers", self._handle_add_trigger)
        r.add_delete("/api/v1/processes/{name}/triggers/{trigger_id}", self._handle_remove_trigger)

        # Secret management
        r.add_get("/api/v1/processes/{name}/secrets", self._handle_list_secrets)
        r.add_patch("/api/v1/processes/{name}/secrets/{secret_name}", self._handle_rotate_secret)
        r.add_delete("/api/v1/processes/{name}/secrets", self._handle_revoke_secrets)

        # Journal
        r.add_get("/api/v1/processes/{name}/journal", self._handle_get_journal)

        # Mission control
        r.add_get("/api/v1/processes/{name}/mission", self._handle_get_mission)
        r.add_post("/api/v1/processes/{name}/mission/fail", self._handle_fail_mission)
        r.add_post("/api/v1/processes/{name}/mission/pause", self._handle_pause_mission)
        r.add_post("/api/v1/processes/{name}/mission/resume", self._handle_resume_mission)

        # Health details
        r.add_get("/api/v1/processes/{name}/health/anomalies", self._handle_get_anomalies)
        r.add_delete("/api/v1/processes/{name}/health/anomalies", self._handle_clear_anomalies)

        # Process suspend/resume
        r.add_post("/api/v1/processes/{name}/suspend", self._handle_suspend)
        r.add_post("/api/v1/processes/{name}/resume", self._handle_resume)

        # Observability
        r.add_get("/api/v1/processes/{name}/metrics", self._handle_get_metrics)
        r.add_get("/api/v1/processes/{name}/context", self._handle_get_context)
        r.add_patch("/api/v1/processes/{name}/context", self._handle_update_context)

    # ------------------------------------------------------------------
    # Health (no auth)
    # ------------------------------------------------------------------

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health probe for load balancers."""
        return _json_response(
            {
                "status": "healthy",
                "process_count": len(self._runtime.processes),
            }
        )

    # ------------------------------------------------------------------
    # Runtime management
    # ------------------------------------------------------------------

    async def _handle_runtime_status(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        return _json_response(self._runtime.status())

    async def _handle_start_all(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        await self._runtime.start_all()
        return _json_response({"status": "started"})

    async def _handle_stop_all(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        await self._runtime.stop_all()
        return _json_response({"status": "stopped"})

    # ------------------------------------------------------------------
    # Process lifecycle
    # ------------------------------------------------------------------

    async def _handle_list_processes(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        processes = self._runtime.list_processes()
        return _json_response({"processes": processes, "total": len(processes)})

    async def _handle_get_process(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            status = self._runtime.process_status(name)
            return _json_response(status)
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_deploy(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err

        try:
            body = await self._json_body(request)
        except Exception:
            return _error_response("INVALID_JSON", "Request body must be valid JSON", 400)

        name = body.get("name")
        if not name:
            return _error_response("MISSING_FIELD", "'name' is required", 422)
        if not _PROCESS_NAME_RE.match(name):
            return _error_response(
                "INVALID_NAME",
                "Process name must be alphanumeric with hyphens/underscores, max 64 chars",
                422,
            )

        config_data = body.get("config", {})
        try:
            from .config import ProcessConfig

            config = ProcessConfig(**config_data)
        except Exception as exc:
            # Extract field-level errors from Pydantic if available
            detail = str(exc)
            if hasattr(exc, "errors"):
                try:
                    field_errors = [
                        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
                    ]
                    detail = "; ".join(field_errors)
                except Exception:
                    pass
            return _error_response("INVALID_CONFIG", f"Config validation failed: {detail}", 422)

        try:
            process = await self._runtime.add_process(name, config)
        except Exception as exc:
            return _error_response("DEPLOY_FAILED", str(exc), 409)

        if body.get("start", False):
            try:
                await self._runtime.start_process(name)
            except Exception as exc:
                return _error_response("START_FAILED", str(exc), 500)

        return _json_response(
            {
                "name": name,
                "process_id": process.process_id,
                "state": process.state.value
                if hasattr(process.state, "value")
                else str(process.state),
                "created_at": time.time(),
            },
            status=201,
        )

    async def _handle_remove_process(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            await self._runtime.remove_process(name)
            return web.Response(status=204)
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_start(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            await self._runtime.start_process(name)
            process = self._get_process(name)
            return _json_response(
                {
                    "name": name,
                    "state": process.state.value
                    if hasattr(process.state, "value")
                    else str(process.state),
                }
            )
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)
        except Exception as exc:
            return _error_response("START_FAILED", str(exc), 500)

    async def _handle_stop(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            await self._runtime.stop_process(name)
            process = self._get_process(name)
            return _json_response(
                {
                    "name": name,
                    "state": process.state.value
                    if hasattr(process.state, "value")
                    else str(process.state),
                }
            )
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)
        except Exception as exc:
            return _error_response("STOP_FAILED", str(exc), 500)

    async def _handle_restart(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            await self._runtime.restart_process(name)
            process = self._get_process(name)
            return _json_response(
                {
                    "name": name,
                    "state": process.state.value
                    if hasattr(process.state, "value")
                    else str(process.state),
                }
            )
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)
        except Exception as exc:
            return _error_response("RESTART_FAILED", str(exc), 500)

    # ------------------------------------------------------------------
    # Config updates
    # ------------------------------------------------------------------

    async def _handle_update_instructions(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            body = await self._json_body(request)
        except Exception:
            return _error_response("INVALID_JSON", "Request body must be valid JSON", 400)

        try:
            instructions = body.get("instructions")
            if not instructions:
                return _error_response("MISSING_FIELD", "'instructions' is required", 422)

            process = self._get_process(name)
            # Update config + dynamic instructions. The agent reads
            # _dynamic_instructions on each invocation cycle (set during
            # _build_agent and checked in _invoke_agent context injection).
            process.config.instructions = instructions
            if hasattr(process, "_dynamic_instructions"):
                process._dynamic_instructions = instructions

            return _json_response(
                {
                    "name": name,
                    "instructions_updated": True,
                    "effective_from": "next_invocation",
                }
            )
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_update_budget(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            body = await self._json_body(request)
        except Exception:
            return _error_response("INVALID_JSON", "Request body must be valid JSON", 400)

        try:
            process = self._get_process(name)
            error = _apply_patch(process.config.budget, body)
            if error is not None:
                return error

            return _json_response(
                {
                    "name": name,
                    "budget_updated": True,
                    "effective_from": "next_budget_check",
                }
            )
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_update_health(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            body = await self._json_body(request)
        except Exception:
            return _error_response("INVALID_JSON", "Request body must be valid JSON", 400)

        try:
            process = self._get_process(name)
            error = _apply_patch(process.config.health, body)
            if error is not None:
                return error

            return _json_response(
                {
                    "name": name,
                    "health_updated": True,
                    "effective_from": "immediate",
                }
            )
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_update_mission(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            body = await self._json_body(request)
        except Exception:
            return _error_response("INVALID_JSON", "Request body must be valid JSON", 400)

        try:
            process = self._get_process(name)
            error = _apply_patch(process.config.mission, body)
            if error is not None:
                return error

            return _json_response(
                {
                    "name": name,
                    "mission_updated": True,
                    "effective_from": "immediate",
                }
            )
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    # ------------------------------------------------------------------
    # Human communication
    # ------------------------------------------------------------------

    async def _handle_send_message(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            body = await self._json_body(request)
            process = self._get_process(name)

            if not hasattr(process, "_inbox") or process._inbox is None:
                return _error_response(
                    "INBOX_DISABLED",
                    "Message inbox is not enabled for this process. Set inbox.enabled=true in config.",
                    400,
                )

            from .inbox import InboxMessage, MessageType

            content = body.get("content")
            if not content:
                return _error_response("MISSING_FIELD", "'content' is required", 422)

            msg_type = body.get("message_type", "context")
            try:
                message_type = MessageType(msg_type)
            except ValueError:
                return _error_response(
                    "INVALID_TYPE",
                    f"Invalid message_type: {msg_type!r}. Must be: directive, context, question, correction",
                    422,
                )

            ttl = body.get("ttl")
            expires_at = None
            if ttl and ttl > 0:
                expires_at = time.time() + ttl

            message = InboxMessage(
                content=content,
                message_type=message_type,
                sender_id=body.get("sender_id"),
                priority=body.get("priority", "normal"),
                expires_at=expires_at,
                metadata=body.get("metadata", {}),
            )

            try:
                msg_id = await process._inbox.add(message)
            except ValueError as exc:
                return _error_response("RATE_LIMITED", str(exc), 429)

            return _json_response(
                {
                    "message_id": msg_id,
                    "status": "queued",
                    "expires_at": message.expires_at,
                },
                status=201,
            )
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_ask(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            body = await self._json_body(request)
            process = self._get_process(name)

            if not hasattr(process, "_inbox") or process._inbox is None:
                return _error_response("INBOX_DISABLED", "Message inbox not enabled", 400)

            from .inbox import InboxMessage, MessageType

            content = body.get("content")
            if not content:
                return _error_response("MISSING_FIELD", "'content' is required", 422)

            timeout = body.get("timeout", 120)

            message = InboxMessage(
                content=content,
                message_type=MessageType.QUESTION,
                sender_id=body.get("sender_id"),
                priority=body.get("priority", "normal"),
                expires_at=time.time() + timeout + 60,  # Extra buffer beyond poll timeout
            )

            try:
                msg_id = await process._inbox.add(message)
            except ValueError as exc:
                return _error_response("RATE_LIMITED", str(exc), 429)

            # Long-poll: wait for the agent to respond
            try:
                response = await process._inbox.wait_for_response(msg_id, timeout=timeout)
                return _json_response(
                    {
                        "question_id": msg_id,
                        "status": "answered",
                        "response": response.content,
                        "answered_at": response.answered_at,
                        "invocation_id": response.invocation_id,
                    }
                )
            except asyncio.TimeoutError:
                return _json_response(
                    {
                        "question_id": msg_id,
                        "status": "timeout",
                        "response": None,
                    }
                )
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_get_inbox(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            process = self._get_process(name)
            if not hasattr(process, "_inbox") or process._inbox is None:
                return _json_response(
                    {"pending_messages": 0, "pending_questions": 0, "messages": []}
                )
            status = await process._inbox.status()
            return _json_response(status)
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_clear_inbox(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            process = self._get_process(name)
            if hasattr(process, "_inbox") and process._inbox is not None:
                await process._inbox.clear()
            return web.Response(status=204)
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    async def _handle_get_metrics(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            status = self._runtime.process_status(name)
            return _json_response(status)
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_get_context(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            process = self._get_process(name)
            ctx = process.context
            return _json_response(
                {
                    "state": ctx.state_snapshot() if hasattr(ctx, "state_snapshot") else {},
                    "writable_keys": list(ctx._writable_keys)
                    if hasattr(ctx, "_writable_keys") and ctx._writable_keys
                    else [],
                }
            )
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_update_context(self, request: web.Request) -> web.Response:
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            body = await self._json_body(request)
        except Exception:
            return _error_response("INVALID_JSON", "Request body must be valid JSON", 400)
        try:
            process = self._get_process(name)
            ctx = process.context

            state_updates = body.get("state", {})
            updated_keys = []
            for key, value in state_updates.items():
                try:
                    ctx.put(key, value, source="api")
                    updated_keys.append(key)
                except (ValueError, PermissionError, KeyError) as exc:
                    return _error_response(
                        "WRITE_DENIED",
                        f"Cannot write key '{key}': {exc}",
                        403,
                    )

            return _json_response(
                {
                    "name": name,
                    "updated_keys": updated_keys,
                }
            )
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    # ------------------------------------------------------------------
    # Trigger management
    # ------------------------------------------------------------------

    async def _handle_list_triggers(self, request: web.Request) -> web.Response:
        """List all triggers (static + dynamic) for a process."""
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            process = self._get_process(name)
            triggers = []
            for i, tc in enumerate(process.config.triggers):
                triggers.append(
                    {
                        "id": f"static_{i}",
                        "type": tc.type,
                        "source": "config",
                        "config": tc.model_dump()
                        if hasattr(tc, "model_dump")
                        else {"type": tc.type},
                    }
                )
            for t in getattr(process, "_dynamic_triggers", []):
                tid = getattr(t, "trigger_id", None) or str(id(t))
                triggers.append({"id": tid, "type": type(t).__name__, "source": "dynamic"})
            return _json_response({"name": name, "triggers": triggers, "count": len(triggers)})
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_add_trigger(self, request: web.Request) -> web.Response:
        """Add a trigger to a process (effective on next restart)."""
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            body = await self._json_body(request)
        except Exception:
            return _error_response("INVALID_JSON", "Request body must be valid JSON", 400)
        try:
            process = self._get_process(name)
            if not body.get("type"):
                return _error_response("MISSING_FIELD", "'type' is required", 422)
            from .config import TriggerConfig

            tc = TriggerConfig(**body)
            process.config.triggers.append(tc)
            return _json_response({"name": name, "trigger_added": True, "trigger_type": tc.type})
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)
        except Exception as exc:
            return _error_response("INVALID_CONFIG", f"Invalid trigger config: {exc}", 422)

    async def _handle_remove_trigger(self, request: web.Request) -> web.Response:
        """Remove a trigger from a process."""
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        trigger_id = request.match_info["trigger_id"]
        try:
            process = self._get_process(name)
            # Dynamic triggers are a list, not a dict
            dynamic = getattr(process, "_dynamic_triggers", [])
            for i, t in enumerate(dynamic):
                tid = getattr(t, "trigger_id", None) or str(id(t))
                if tid == trigger_id:
                    removed = dynamic.pop(i)
                    if hasattr(removed, "stop"):
                        await removed.stop()
                    return _json_response({"name": name, "trigger_removed": trigger_id})
            if trigger_id.startswith("static_"):
                try:
                    idx = int(trigger_id.split("_")[1])
                    if 0 <= idx < len(process.config.triggers):
                        process.config.triggers.pop(idx)
                        return _json_response({"name": name, "trigger_removed": trigger_id})
                except (ValueError, IndexError):
                    pass
            return _error_response("TRIGGER_NOT_FOUND", f"Trigger '{trigger_id}' not found", 404)
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    # ------------------------------------------------------------------
    # Secret management
    # ------------------------------------------------------------------

    async def _handle_list_secrets(self, request: web.Request) -> web.Response:
        """List secret names and TTL status (never values)."""
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            process = self._get_process(name)
            secrets = getattr(process, "_secrets", None)
            if secrets is None:
                return _json_response({"name": name, "secrets": [], "enabled": False})
            names = secrets.list_names() if hasattr(secrets, "list_names") else []
            info = []
            for sn in names:
                # SecretScope stores entries in _secrets dict, not _entries
                entry = secrets._secrets.get(sn) if hasattr(secrets, "_secrets") else None
                if entry:
                    is_expired = (
                        secrets._is_expired(entry) if hasattr(secrets, "_is_expired") else False
                    )
                    info.append(
                        {
                            "name": sn,
                            "active": not is_expired,
                        }
                    )
                else:
                    info.append({"name": sn, "active": True})
            return _json_response({"name": name, "secrets": info, "enabled": True})
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_rotate_secret(self, request: web.Request) -> web.Response:
        """Rotate a secret value (immediate effect)."""
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        secret_name = request.match_info["secret_name"]
        try:
            body = await self._json_body(request)
        except Exception:
            return _error_response("INVALID_JSON", "Request body must be valid JSON", 400)
        try:
            process = self._get_process(name)
            secrets = getattr(process, "_secrets", None)
            if secrets is None:
                return _error_response("SECRETS_DISABLED", "Secret scoping not enabled", 400)
            new_value = body.get("value")
            if not new_value:
                return _error_response("MISSING_FIELD", "'value' is required", 422)
            await secrets.rotate(secret_name, new_value)
            return _json_response({"name": name, "secret_rotated": secret_name})
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)
        except Exception as exc:
            return _error_response("ROTATION_FAILED", f"Rotation failed: {exc}", 400)

    async def _handle_revoke_secrets(self, request: web.Request) -> web.Response:
        """Revoke all secrets (zero-fill, immediate)."""
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            process = self._get_process(name)
            secrets = getattr(process, "_secrets", None)
            if secrets is None:
                return _error_response("SECRETS_DISABLED", "Secret scoping not enabled", 400)
            await secrets.revoke_all()
            return _json_response({"name": name, "secrets_revoked": True})
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    # ------------------------------------------------------------------
    # Journal
    # ------------------------------------------------------------------

    async def _handle_get_journal(self, request: web.Request) -> web.Response:
        """Read journal entries for a process."""
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            process = self._get_process(name)
            journal = getattr(process, "_journal", None)
            if journal is None:
                return _json_response({"name": name, "entries": [], "journal_enabled": False})
            limit = min(int(request.query.get("limit", "50")), 500)
            entry_type = request.query.get("type")
            entries = await journal.read(process.process_id) if hasattr(journal, "read") else []
            if entry_type:
                entries = [e for e in entries if getattr(e, "entry_type", "") == entry_type]
            entries = entries[-limit:]
            serialized = []
            for e in entries:
                serialized.append(
                    {
                        "entry_type": getattr(e, "entry_type", "unknown"),
                        "timestamp": getattr(e, "timestamp", 0),
                        "data": getattr(e, "data", {}),
                    }
                )
            return _json_response({"name": name, "entries": serialized, "count": len(serialized)})
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    # ------------------------------------------------------------------
    # Mission control
    # ------------------------------------------------------------------

    async def _handle_get_mission(self, request: web.Request) -> web.Response:
        """Get mission state and evaluation history."""
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            process = self._get_process(name)
            mission = getattr(process, "_mission", None)
            if mission is None:
                return _json_response({"name": name, "mission_enabled": False})
            state = {
                "state": mission.state.value
                if hasattr(mission.state, "value")
                else str(mission.state),
                "objective": getattr(getattr(mission, "_config", None), "objective", ""),
                "invocation_count": getattr(mission, "invocation_count", 0),
            }
            evals = []
            for ev in getattr(mission, "evaluations", []):
                evals.append(
                    {
                        "achieved": getattr(ev, "achieved", False),
                        "confidence": getattr(ev, "confidence", 0),
                        "reasoning": getattr(ev, "reasoning", ""),
                    }
                )
            state["evaluations"] = evals
            return _json_response({"name": name, "mission_enabled": True, "mission": state})
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_fail_mission(self, request: web.Request) -> web.Response:
        """Manually fail a mission."""
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            body = await self._json_body(request)
        except Exception:
            body = {}
        try:
            process = self._get_process(name)
            mission = getattr(process, "_mission", None)
            if mission is None:
                return _error_response("MISSION_DISABLED", "Mission not configured", 400)
            reason = body.get("reason", "Manually failed via API")
            mission.fail(reason)
            return _json_response({"name": name, "mission_failed": True, "reason": reason})
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_pause_mission(self, request: web.Request) -> web.Response:
        """Pause mission evaluation."""
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            process = self._get_process(name)
            mission = getattr(process, "_mission", None)
            if mission is None:
                return _error_response("MISSION_DISABLED", "Mission not configured", 400)
            if hasattr(mission, "pause"):
                mission.pause()
            return _json_response({"name": name, "mission_paused": True})
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_resume_mission(self, request: web.Request) -> web.Response:
        """Resume mission evaluation."""
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            process = self._get_process(name)
            mission = getattr(process, "_mission", None)
            if mission is None:
                return _error_response("MISSION_DISABLED", "Mission not configured", 400)
            if hasattr(mission, "resume"):
                mission.resume()
            return _json_response({"name": name, "mission_resumed": True})
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    # ------------------------------------------------------------------
    # Health details
    # ------------------------------------------------------------------

    async def _handle_get_anomalies(self, request: web.Request) -> web.Response:
        """Get anomaly history for a process."""
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            process = self._get_process(name)
            health = getattr(process, "_health", None)
            if health is None:
                return _json_response({"name": name, "health_enabled": False, "anomalies": []})
            anomalies = []
            for a in getattr(health, "anomalies", []):
                anomalies.append(
                    {
                        "type": getattr(a, "anomaly_type", "unknown"),
                        "description": getattr(a, "description", ""),
                        "timestamp": getattr(a, "timestamp", 0),
                    }
                )
            return _json_response({"name": name, "anomalies": anomalies, "count": len(anomalies)})
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    async def _handle_clear_anomalies(self, request: web.Request) -> web.Response:
        """Clear anomaly history."""
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            process = self._get_process(name)
            health = getattr(process, "_health", None)
            if health is None:
                return _error_response("HEALTH_DISABLED", "Health monitoring not configured", 400)
            cleared = len(getattr(health, "anomalies", []))
            if hasattr(health, "anomalies"):
                # anomalies is a property that returns a copy — clear the internal list
                health._anomalies.clear()
            return _json_response({"name": name, "anomalies_cleared": cleared})
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)

    # ------------------------------------------------------------------
    # Process suspend/resume
    # ------------------------------------------------------------------

    async def _handle_suspend(self, request: web.Request) -> web.Response:
        """Suspend a process (pause trigger processing)."""
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            process = self._get_process(name)
            if hasattr(process, "suspend"):
                await process.suspend()
            return _json_response({"name": name, "suspended": True})
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)
        except Exception as exc:
            return _error_response("SUSPEND_FAILED", f"Suspend failed: {exc}", 400)

    async def _handle_resume(self, request: web.Request) -> web.Response:
        """Resume a suspended process."""
        auth_err = self._require_auth(request)
        if auth_err:
            return auth_err
        name = request.match_info["name"]
        try:
            process = self._get_process(name)
            if hasattr(process, "resume"):
                await process.resume()
            return _json_response({"name": name, "resumed": True})
        except KeyError:
            return _error_response("PROCESS_NOT_FOUND", f"Process '{name}' not found", 404)
        except Exception as exc:
            return _error_response("RESUME_FAILED", f"Resume failed: {exc}", 400)
