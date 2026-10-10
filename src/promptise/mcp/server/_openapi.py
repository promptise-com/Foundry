"""OpenAPI-to-MCP tool generator.

Automatically generates MCP tools from an OpenAPI specification.
Each endpoint in the spec becomes a tool that makes HTTP requests
to the target API.

Example::

    from promptise.mcp.server import MCPServer, OpenAPIProvider

    server = MCPServer(name="api-bridge")
    provider = OpenAPIProvider("https://petstore.swagger.io/v2/swagger.json")
    provider.register(server)
"""

from __future__ import annotations

import ipaddress
import json
import logging
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger("promptise.server")


def _validate_url_not_private(
    url: str, *, hint: str = "Use base_url override for internal APIs."
) -> None:
    """Reject URLs that resolve to private/internal IP ranges (SSRF protection).

    Args:
        url: The URL to check.
        hint: Appended to the error: how the caller can reach an internal
            host on purpose.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError(f"Invalid URL: {url!r}")

    # Block well-known internal hostnames
    if hostname in ("localhost", "metadata.google.internal"):
        raise ValueError(f"URL targets a private/internal host: {hostname!r}. {hint}")

    # Resolve hostname and check IP ranges
    import socket

    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC)
    except socket.gaierror:
        return  # Cannot resolve — let httpx handle the error

    for _family, _type, _proto, _canonname, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            raise ValueError(f"URL {url!r} resolves to private/internal IP {ip}. {hint}")


class OpenAPIProvider:
    """Generate MCP tools from an OpenAPI specification.

    Parses an OpenAPI 3.x (or Swagger 2.x) spec and registers one
    MCP tool per operation.  Each tool makes an HTTP request to the
    target API when called.

    Args:
        spec: Either a URL to fetch the spec from, a file path, or
            a pre-parsed dict.
        base_url: Override the server base URL from the spec.
        prefix: Prefix for generated tool names (e.g. ``"petstore_"``).
        include: Only include operations matching these operation IDs.
        exclude: Exclude operations matching these operation IDs.
        auth_header: Default auth header for requests (e.g.
            ``("Authorization", "Bearer tok-123")``).
        tags: Tags to apply to all generated tools.
    """

    def __init__(
        self,
        spec: str | dict[str, Any],
        *,
        base_url: str | None = None,
        prefix: str = "",
        include: set[str] | None = None,
        exclude: set[str] | None = None,
        auth_header: tuple[str, str] | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self._raw_spec = spec
        self._base_url = base_url
        self._prefix = prefix
        self._include = include
        self._exclude = exclude or set()
        self._auth_header = auth_header
        self._tags = tags or []
        self._spec: dict[str, Any] | None = None
        self._operations: list[dict[str, Any]] = []

    def _load_spec(self) -> dict[str, Any]:
        """Load and parse the OpenAPI spec."""
        if isinstance(self._raw_spec, dict):
            return self._raw_spec

        raw = self._raw_spec
        # If it looks like a URL, try fetching it
        if raw.startswith("http://") or raw.startswith("https://"):
            try:
                import httpx

                _validate_url_not_private(raw)
                resp = httpx.get(raw, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except ImportError:
                raise ImportError(
                    "httpx is required to fetch OpenAPI specs from URLs. "
                    "Install with: pip install httpx"
                )

        # Try as file path
        import pathlib

        path = pathlib.Path(raw)
        if path.exists():
            text = path.read_text()
            if path.suffix in (".yaml", ".yml"):
                try:
                    import yaml

                    return yaml.safe_load(text)
                except ImportError:
                    raise ImportError(
                        "PyYAML is required to parse YAML specs. Install with: pip install pyyaml"
                    )
            return json.loads(text)

        # Try as raw JSON string
        return json.loads(raw)

    def _extract_operations(self, spec: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract operations from the spec."""
        ops: list[dict[str, Any]] = []
        paths = spec.get("paths", {})

        # Determine base URL
        base = self._base_url
        if base is None:
            servers = spec.get("servers", [])
            if servers:
                base = servers[0].get("url", "")
            elif "host" in spec:
                # Swagger 2.x
                scheme = (spec.get("schemes") or ["https"])[0]
                base_path = spec.get("basePath", "")
                base = f"{scheme}://{spec['host']}{base_path}"

        # SSRF protection: validate base_url doesn't target private networks
        if base and (base.startswith("http://") or base.startswith("https://")):
            _validate_url_not_private(base)

        for path, methods in paths.items():
            for method, operation in methods.items():
                if method in ("parameters", "summary", "description", "$ref"):
                    continue

                op_id = operation.get("operationId", f"{method}_{path}")
                op_id = op_id.replace("-", "_").replace("/", "_").strip("_")

                # Filter
                if self._include and op_id not in self._include:
                    continue
                if op_id in self._exclude:
                    continue

                # Build input schema from parameters and request body
                input_schema = self._build_input_schema(operation, spec)

                ops.append(
                    {
                        "operation_id": op_id,
                        "method": method.upper(),
                        "path": path,
                        "base_url": base or "",
                        "summary": operation.get("summary", ""),
                        "description": operation.get("description", ""),
                        "input_schema": input_schema,
                        "parameters": operation.get("parameters", []),
                    }
                )

        return ops

    def _build_input_schema(
        self, operation: dict[str, Any], spec: dict[str, Any]
    ) -> dict[str, Any]:
        """Build a JSON Schema for the operation's inputs."""
        properties: dict[str, Any] = {}
        required: list[str] = []

        # Path + query + header params
        for param in operation.get("parameters", []):
            if "$ref" in param:
                param = self._resolve_ref(param["$ref"], spec)
            name = param.get("name", "")
            schema = param.get("schema", {"type": "string"})
            properties[name] = {
                **schema,
                "description": param.get("description", name),
            }
            if param.get("required", False):
                required.append(name)

        # Request body
        body = operation.get("requestBody", {})
        content = body.get("content", {})
        json_content = content.get("application/json", {})
        body_schema = json_content.get("schema", {})
        if "$ref" in body_schema:
            body_schema = self._resolve_ref(body_schema["$ref"], spec)

        if body_schema.get("properties"):
            for k, v in body_schema["properties"].items():
                properties[k] = v
            if body_schema.get("required"):
                required.extend(body_schema["required"])
        elif body_schema:
            properties["body"] = body_schema
            if body.get("required", False):
                required.append("body")

        result: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            result["required"] = required
        return result

    def _resolve_ref(self, ref: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Resolve a $ref pointer."""
        parts = ref.lstrip("#/").split("/")
        node: Any = spec
        for part in parts:
            node = node.get(part, {})
        return dict(node) if isinstance(node, dict) else {}

    def parse(self) -> list[dict[str, Any]]:
        """Parse the spec and return operations (does not register tools).

        Returns:
            List of operation dicts with ``operation_id``, ``method``,
            ``path``, ``summary``, and ``input_schema``.
        """
        self._spec = self._load_spec()
        self._operations = self._extract_operations(self._spec)
        return self._operations

    def register(self, server: Any) -> int:
        """Parse the spec and register tools on the server.

        Returns:
            Number of tools registered.
        """
        if not self._operations:
            self.parse()

        count = 0
        for op in self._operations:
            self._register_tool(server, op)
            count += 1

        logger.info(
            "OpenAPI: registered %d tools from spec (prefix=%r)",
            count,
            self._prefix,
        )
        return count

    def _register_tool(self, server: Any, op: dict[str, Any]) -> None:
        """Register a single operation as an MCP tool."""
        tool_name = f"{self._prefix}{op['operation_id']}"
        description = op.get("summary") or op.get("description") or tool_name
        method = op["method"]
        path = op["path"]
        base_url = op["base_url"]
        auth_header = self._auth_header

        from ._types import ToolAnnotations

        _annotations = ToolAnnotations(  # noqa: F841 — reserved for future use
            title=description[:50] if len(description) > 50 else description,
            read_only_hint=method == "GET",
            destructive_hint=method == "DELETE",
            idempotent_hint=method in ("GET", "PUT", "DELETE"),
        )

        # Create a handler that makes the HTTP request
        async def handler(
            _method=method,
            _base=base_url,
            _path=path,
            _auth=auth_header,
            **kwargs: Any,
        ) -> dict[str, Any]:
            try:
                import httpx
            except ImportError:
                raise ImportError(
                    "httpx is required for OpenAPI tools. Install with: pip install httpx"
                )

            # Separate path params from body/query
            url = f"{_base}{_path}"
            for key, value in list(kwargs.items()):
                placeholder = f"{{{key}}}"
                if placeholder in url:
                    url = url.replace(placeholder, str(value))
                    del kwargs[key]

            headers = {}
            if _auth:
                headers[_auth[0]] = _auth[1]

            async with httpx.AsyncClient(timeout=30) as client:
                if _method in ("GET", "HEAD", "DELETE"):
                    resp = await client.request(_method, url, params=kwargs, headers=headers)
                else:
                    resp = await client.request(_method, url, json=kwargs, headers=headers)

                try:
                    return resp.json()
                except Exception:
                    return {"status": resp.status_code, "body": resp.text}

        # Use a factory to create unique handler with proper name
        handler.__name__ = tool_name
        handler.__doc__ = description

        @server.tool(
            name=tool_name,
            description=description,
            tags=self._tags,
        )
        async def _tool(**kwargs: Any) -> dict[str, Any]:
            return await handler(**kwargs)

        # Override the handler name for the registered tool
        _tool.__name__ = tool_name
