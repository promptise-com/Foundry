# Service Discovery and Transport

The distributed runtime uses two complementary systems: **service discovery** for nodes to find each other, and **transport** for remote management via HTTP. Together, they enable multi-node agent deployments where processes can be started, stopped, monitored, and triggered across machines.

```python
from promptise.runtime.distributed.discovery import StaticDiscovery, RegistryDiscovery
from promptise.runtime.distributed.transport import RuntimeTransport
from promptise.runtime import AgentRuntime

# Discovery: nodes find each other
discovery = StaticDiscovery(nodes={
    "node-1": "http://host1:9100",
    "node-2": "http://host2:9100",
})
nodes = await discovery.discover()

# Transport: expose a node for remote management
runtime = AgentRuntime()
async with RuntimeTransport(runtime, port=9100) as transport:
    # HTTP API now available at http://127.0.0.1:9100/ (this machine only)
    ...
```

---

## Service Discovery

Service discovery provides mechanisms for runtime nodes to find each other. Two implementations are provided:

### ProcessDiscovery Protocol

All discovery implementations satisfy this protocol:

```python
from promptise.runtime.distributed.discovery import ProcessDiscovery

class ProcessDiscovery(Protocol):
    async def discover(self) -> list[DiscoveredNode]:
        """Discover available runtime nodes."""
        ...

    async def register(self, node_id: str, url: str, metadata: dict | None = None) -> None:
        """Register this node for discovery by others."""
        ...

    async def unregister(self, node_id: str) -> None:
        """Remove this node from discovery."""
        ...
```

### DiscoveredNode

Each discovered node is represented as a `DiscoveredNode` dataclass:

| Field | Type | Description |
|---|---|---|
| `node_id` | `str` | Unique node identifier |
| `url` | `str` | Base URL for the node's transport API |
| `discovered_at` | `float` | Timestamp when the node was discovered |
| `metadata` | `dict[str, Any]` | Additional node metadata |

---

## StaticDiscovery

A simple discovery mechanism for fixed-topology deployments where all node addresses are known at configuration time.

```python
from promptise.runtime.distributed.discovery import StaticDiscovery

discovery = StaticDiscovery(nodes={
    "node-1": "http://host1:9100",
    "node-2": "http://host2:9100",
    "node-3": "http://host3:9100",
})

# Discover all nodes
nodes = await discovery.discover()
for node in nodes:
    print(f"{node.node_id}: {node.url}")

# Add a node dynamically
await discovery.register("node-4", "http://host4:9100")

# Remove a node
await discovery.unregister("node-2")
```

Best for:

- Development and testing environments
- Small, fixed-size clusters
- Deployments where node addresses are known at startup

---

## RegistryDiscovery

A dynamic, in-process registry where nodes register themselves and discover each other. Stale registrations are automatically pruned after a configurable TTL.

```python
from promptise.runtime.distributed.discovery import RegistryDiscovery

registry = RegistryDiscovery(ttl=60.0)  # 60-second TTL

# Nodes register themselves at startup
await registry.register("node-1", "http://host1:9100", metadata={"region": "us-east"})
await registry.register("node-2", "http://host2:9100", metadata={"region": "eu-west"})

# Discover all active nodes
nodes = await registry.discover()  # Prunes stale entries first

# Nodes send periodic heartbeats to stay registered
await registry.heartbeat("node-1")

# Unregister on shutdown
await registry.unregister("node-1")
```

### TTL and stale pruning

Nodes that do not refresh their registration within the TTL are automatically removed:

```python
registry = RegistryDiscovery(ttl=30.0)

await registry.register("node-1", "http://host1:9100")
# ... 30+ seconds pass without heartbeat ...
nodes = await registry.discover()  # node-1 is pruned
```

### Thread safety

`RegistryDiscovery` uses an `asyncio.Lock` for all operations, making it safe for concurrent use.

Best for:

- Dynamic clusters where nodes come and go
- Single-process testing with multiple logical nodes
- Coordinator-hosted registry exposed via HTTP API

---

## RuntimeTransport

The `RuntimeTransport` exposes an `AgentRuntime` as an HTTP API for remote management. It runs an `aiohttp` server that handles process control, status queries, and event injection.

### Creating a transport

```python
import os

from promptise.runtime import AgentRuntime
from promptise.runtime.distributed.transport import RuntimeTransport

runtime = AgentRuntime()
transport = RuntimeTransport(
    runtime,
    host="0.0.0.0",                     # reachable from other machines
    port=9100,
    node_id="node-1",
    auth_token=os.environ["PROMPTISE_NODE_TOKEN"],  # required off loopback
)

await transport.start()
# HTTP API available on port 9100 of every interface
await transport.stop()
```

The default `host` is `127.0.0.1` (this machine only). Pass `port=0` to let the OS pick a free port; `transport.port` reports it after `start()`.

### Security

The transport can start, stop and drive your agents, so it is locked down by default:

| Rule | Behaviour |
|---|---|
| Bearer token | With `auth_token` set, every endpoint except `GET /health` requires `Authorization: Bearer <token>` (compared in constant time). Missing or wrong token: `401`. |
| Public bind needs a token | `RuntimeTransport(host="0.0.0.0")` (or any non-loopback address) without `auth_token` raises `ValueError`. If an authenticating proxy or a private network you control already protects the port, opt out explicitly with `allow_unauthenticated=True` (a warning is logged at start). |
| Loopback bind blocks browsers | On `127.0.0.1` / `localhost` / `::1`, a request whose `Host` header is not a loopback name gets `421` (DNS rebinding), and a request with an `Origin` header from a non-loopback origin gets `403`. A web page you visit cannot drive the local runtime. Non-browser clients (the coordinator, `curl`, scripts) send no `Origin` and are unaffected. |
| Empty token | `auth_token=""` raises `ValueError`. |

Use TLS (a reverse proxy) when the token crosses an untrusted network: the transport itself speaks plain HTTP.

!!! warning "Changed in 1.3.0"
    A non-loopback bind without `auth_token` used to log a warning and serve every endpoint unauthenticated. It now raises `ValueError` unless you pass `allow_unauthenticated=True`.

### Context manager

```python
async with RuntimeTransport(runtime, port=9100, auth_token="node-token") as transport:
    # Server running
    ...
# Server stopped automatically
```

---

## HTTP API Endpoints

The transport exposes the following REST endpoints:

### Health check

```
GET /health
```

Response:

```json
{
    "status": "healthy",
    "node_id": "node-1",
    "process_count": 3
}
```

### Runtime status

```
GET /status
```

Returns the full `runtime.status()` dict including per-process status, plus the `node_id`.

### List processes

```
GET /processes
```

Response:

```json
{
    "node_id": "node-1",
    "processes": [
        {"name": "watcher", "state": "running", "process_id": "abc123"},
        {"name": "handler", "state": "stopped", "process_id": "def456"}
    ]
}
```

### Process status

```
GET /processes/{name}/status
```

Returns the status dict for a single process. Returns 404 if the process does not exist.

### Start a process

```
POST /processes/{name}/start
```

Response (200):

```json
{"status": "started", "name": "watcher"}
```

### Stop a process

```
POST /processes/{name}/stop
```

Response (200):

```json
{"status": "stopped", "name": "watcher"}
```

### Restart a process

```
POST /processes/{name}/restart
```

Response (200):

```json
{"status": "restarted", "name": "watcher"}
```

### Inject event

```
POST /processes/{name}/event
```

Request body (a JSON object; `payload` and `metadata` must be objects when given and default to `{}`):

```json
{
    "trigger_id": "remote",
    "trigger_type": "remote",
    "payload": {"alert": "high_error_rate"},
    "metadata": {}
}
```

Response (202):

```json
{
    "status": "injected",
    "event_id": "...",
    "process": "watcher"
}
```

### Error responses

All endpoints return appropriate HTTP error codes:

| Code | Meaning |
|---|---|
| 200 | Success |
| 202 | Accepted (async operations like event injection) |
| 400 | Bad request (invalid JSON, body or `payload`/`metadata` not an object) |
| 401 | Missing or wrong bearer token |
| 403 | Cross-origin browser request (loopback bind) |
| 404 | Process not found |
| 421 | `Host` header is not a loopback name (loopback bind) |
| 500 | Internal server error |

---

## Putting It Together

A complete distributed deployment:

```python
import asyncio
import os

from promptise.runtime import AgentRuntime, ProcessConfig, TriggerConfig
from promptise.runtime.distributed.transport import RuntimeTransport
from promptise.runtime.distributed.coordinator import RuntimeCoordinator
from promptise.runtime.distributed.discovery import RegistryDiscovery

TOKEN = os.environ["PROMPTISE_NODE_TOKEN"]

async def run_node(node_id: str, port: int):
    """Run a single runtime node."""
    runtime = AgentRuntime()
    await runtime.add_process("watcher", ProcessConfig(
        model="openai:gpt-5-mini",
        instructions="Monitor data.",
        triggers=[TriggerConfig(type="cron", cron_expression="*/5 * * * *")],
    ))
    await runtime.start_all()

    async with RuntimeTransport(runtime, port=port, node_id=node_id, auth_token=TOKEN):
        # Node is now discoverable and remotely manageable
        try:
            while True:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    await runtime.stop_all()

async def run_coordinator():
    """Run the cluster coordinator."""
    async with RuntimeCoordinator(auth_token=TOKEN) as coordinator:
        coordinator.register_node("node-1", "http://localhost:9100")
        coordinator.register_node("node-2", "http://localhost:9101")

        # Monitor cluster health
        while True:
            status = await coordinator.cluster_status()
            print(f"Nodes: {status['total_nodes']}, "
                  f"Healthy: {status['healthy_nodes']}, "
                  f"Processes: {status['total_processes']}")
            await asyncio.sleep(15)
```

---

## API Summary

### Discovery

| Class | Description |
|---|---|
| `ProcessDiscovery` | Protocol for discovery implementations |
| `DiscoveredNode` | Dataclass for discovered nodes |
| `StaticDiscovery(nodes)` | Fixed-topology discovery |
| `RegistryDiscovery(ttl)` | Dynamic registry with TTL |

### Transport

| Method / Property | Description |
|---|---|
| `RuntimeTransport(runtime, host, port, node_id, auth_token, allow_unauthenticated)` | Create a transport server |
| `node_id` | This node's unique identifier |
| `port` | The port being listened on (the OS-assigned one after `start()` with `port=0`) |
| `await start()` | Start the HTTP server |
| `await stop()` | Stop the HTTP server |

---

## Tips and Gotchas

!!! tip "Use RegistryDiscovery with the coordinator"
    The coordinator can host a `RegistryDiscovery` instance and expose it via its HTTP API. Nodes register at startup and send periodic heartbeats. Stale nodes are automatically pruned.

!!! tip "Event injection for cross-node coordination"
    Use the `POST /processes/{name}/event` endpoint to trigger processes on remote nodes. This enables cross-node agent coordination without shared message brokers.

!!! info "aiohttp shipped with base install"
    The `RuntimeTransport` uses `aiohttp`, which is included in the base `pip install promptise`.

!!! warning "One token per node, plain HTTP"
    `auth_token` is a single shared secret with full control of the node, and the transport does not terminate TLS. In production, put nodes behind a TLS reverse proxy or on a private network, and give each node its own token (`RuntimeCoordinator.register_node(..., auth_token=...)`).

!!! warning "RegistryDiscovery is in-process only"
    The `RegistryDiscovery` lives in memory within a single Python process. For multi-machine discovery, host it within the coordinator and expose registration/discovery via the coordinator's HTTP API.

!!! warning "Trailing slashes are stripped"
    Both `StaticDiscovery` and `RegistryDiscovery` strip trailing slashes from URLs to ensure consistent URL construction.

---

## What's Next

- [Coordinator](coordinator.md) -- cluster coordination and health monitoring
- [Configuration](../configuration.md) -- `DistributedConfig` reference
- [Runtime Manager](../runtime-manager.md) -- the `AgentRuntime` that runs on each node
- [Triggers Overview](../triggers/index.md) -- trigger events that can be injected remotely
