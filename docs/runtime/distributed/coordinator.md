# Distributed Coordinator

The `RuntimeCoordinator` is the central brain of a distributed agent runtime deployment. It tracks runtime nodes across the cluster, monitors their health with periodic checks, aggregates status information, and provides remote operations for starting, stopping, and injecting events into processes on remote nodes. It does not move processes on its own: when a node fails, your code picks a healthy node (`healthy_nodes`) and starts the process there (`start_process_on_node`).

```python
from promptise.runtime.distributed.coordinator import RuntimeCoordinator

async with RuntimeCoordinator(
    health_check_interval=15.0,
    node_timeout=45.0,
    auth_token="cluster-token",  # bearer token for the nodes' RuntimeTransport
) as coordinator:
    coordinator.register_node("node-1", "http://host1:9100")
    coordinator.register_node("node-2", "http://host2:9100")

    # Check cluster health
    health = await coordinator.check_health()
    print(health)
    # {"node-1": {"status": "healthy"}, "node-2": {"status": "healthy"}}

    # Get aggregated cluster status
    status = await coordinator.cluster_status()
```

---

## Concepts

In a distributed deployment, agent processes run across multiple machines (nodes). Each node runs an `AgentRuntime` with a `RuntimeTransport` HTTP server. The coordinator sits at the center, providing:

- **Node registry** -- track which nodes exist and their URLs.
- **Health monitoring** -- periodic HTTP health checks to detect node failures.
- **Cluster status** -- aggregate process counts and health across all nodes.
- **Remote operations** -- start/stop processes and inject events on remote nodes.

```
         ┌──────────────┐
         │  Coordinator  │
         └──────┬───────┘
          ┌─────┼─────┐
          │     │     │
     ┌────▼──┐ │  ┌──▼────┐
     │Node 1 │ │  │Node 3 │
     │Runtime │ │  │Runtime │
     └───────┘ │  └───────┘
          ┌────▼──┐
          │Node 2 │
          │Runtime │
          └───────┘
```

---

## Node Management

### Registering nodes

```python
from promptise.runtime.distributed.coordinator import RuntimeCoordinator

coordinator = RuntimeCoordinator()

# Register nodes with their transport URLs
node1 = coordinator.register_node("node-1", "http://host1:9100")
node2 = coordinator.register_node(
    "node-2",
    "http://host2:9100",
    metadata={"region": "us-east", "capacity": "high"},
    auth_token="node-2-token",  # overrides the coordinator's auth_token
)
```

### Authentication

Nodes that bind a non-loopback address must run `RuntimeTransport(auth_token=...)`. The coordinator sends `Authorization: Bearer <token>` on every request: the node's own `auth_token` from `register_node()`, or else the coordinator's `auth_token`. Tokens are never included in `NodeInfo.to_dict()` or `cluster_status()`.

### Unregistering nodes

```python
coordinator.unregister_node("node-1")
```

Raises `KeyError` if the node is not registered.

### Querying nodes

```python
# Get a specific node
node = coordinator.get_node("node-1")
print(node.url)       # "http://host1:9100"
print(node.status)    # "healthy" or "unhealthy" or "unknown"
print(node.is_healthy)  # True/False

# All registered nodes
all_nodes = coordinator.nodes  # dict[str, NodeInfo]

# Only healthy nodes
healthy = coordinator.healthy_nodes  # list[NodeInfo]
```

---

## NodeInfo

Information about a runtime node in the cluster:

| Field | Type | Description |
|---|---|---|
| `node_id` | `str` | Unique node identifier |
| `url` | `str` | Base URL for the node's transport API |
| `last_heartbeat` | `float` | Timestamp of last successful health check |
| `status` | `str` | `"healthy"`, `"unhealthy"`, or `"unknown"` |
| `process_count` | `int` | Number of processes on this node |
| `metadata` | `dict[str, Any]` | Additional node metadata |

```python
from promptise.runtime.distributed.coordinator import NodeInfo

node = NodeInfo(
    node_id="node-1",
    url="http://host1:9100",
    metadata={"region": "us-east"},
)
print(node.is_healthy)  # False (status defaults to "unknown")
```

---

## Health Monitoring

### Automatic monitoring

The coordinator runs a background health check loop when used as a context manager:

```python
async with RuntimeCoordinator(
    health_check_interval=15.0,  # Check every 15 seconds
    node_timeout=45.0,           # Consider unhealthy after 45s
) as coordinator:
    coordinator.register_node("node-1", "http://host1:9100")
    # Health checks run automatically in the background
```

### Manual health checks

```python
# Check all nodes now
health = await coordinator.check_health()
# {
#     "node-1": {"status": "healthy", "process_count": 3},
#     "node-2": {"status": "unhealthy", "error": "Connection refused"},
# }
```

Health checks make HTTP GET requests to each node's `/health` endpoint. Nodes that respond with status 200 are marked healthy. A node that has never answered, or has failed checks for longer than `node_timeout` since its last successful one, is marked unhealthy. A healthy node that misses a check within `node_timeout` stays healthy (its entry in the result carries `"missed_check": true` and the error), so one dropped request does not flap the cluster view.

A failing health round is logged and the background monitor keeps running.

### Starting and stopping the monitor

```python
coordinator = RuntimeCoordinator()
await coordinator.start_health_monitor()
# ... health checks run in background ...
await coordinator.stop_health_monitor()
```

---

## Cluster Status

Aggregate status across all nodes:

```python
status = await coordinator.cluster_status()
# {
#     "total_nodes": 3,
#     "healthy_nodes": 2,
#     "unhealthy_nodes": 1,
#     "total_processes": 8,
#     "nodes": {
#         "node-1": {
#             "node_id": "node-1",
#             "url": "http://host1:9100",
#             "status": "healthy",
#             "process_count": 3,
#             "health": {"status": "healthy"},
#             ...
#         },
#         ...
#     },
# }
```

### Single node status

Get detailed status from a specific node via HTTP:

```python
node_status = await coordinator.get_node_status("node-1")
# Full runtime.status() response from the remote node
```

---

## Remote Operations

Remote operations raise `RuntimeError` when the node answers with an error status (`401` wrong token, `404` unknown process, `500` failed start, ...), so a failure is never mistaken for success.

### Start a process on a remote node

```python
result = await coordinator.start_process_on_node("node-1", "data-watcher")
# {"status": "started", "name": "data-watcher"}
```

### Stop a process on a remote node

```python
result = await coordinator.stop_process_on_node("node-1", "data-watcher")
# {"status": "stopped", "name": "data-watcher"}
```

### Inject a trigger event on a remote node

Send a trigger event to a specific process on a remote node:

```python
result = await coordinator.inject_event_on_node(
    "node-1",
    "data-watcher",
    payload={"alert": "high_error_rate", "rate": 0.15},
    trigger_type="remote",
)
# {"status": "injected", "event_id": "...", "process": "data-watcher"}
```

---

## API Summary

| Method / Property | Description |
|---|---|
| `RuntimeCoordinator(health_check_interval, node_timeout, auth_token)` | Create a coordinator |
| `register_node(node_id, url, metadata, *, auth_token)` | Register a runtime node |
| `unregister_node(node_id)` | Remove a node |
| `get_node(node_id)` | Get `NodeInfo` for a node |
| `nodes` | Read-only dict of all nodes |
| `healthy_nodes` | List of healthy nodes |
| `await start_health_monitor()` | Start background health checks |
| `await stop_health_monitor()` | Stop health checks |
| `await check_health()` | Check all nodes now |
| `await cluster_status()` | Aggregate status across all nodes |
| `await get_node_status(node_id)` | Detailed status from a remote node |
| `await start_process_on_node(node_id, name)` | Start a remote process |
| `await stop_process_on_node(node_id, name)` | Stop a remote process |
| `await inject_event_on_node(node_id, name, payload)` | Inject event into remote process |

---

## Tips and Gotchas

!!! tip "Use the context manager"
    `async with RuntimeCoordinator() as coordinator:` automatically starts the health monitor on entry and stops it on exit. This is the recommended pattern.

!!! tip "Set node_timeout to 3x health_check_interval"
    A node that misses three consecutive health checks (45s with 15s intervals) is likely truly down, not just experiencing a transient network issue.

!!! info "aiohttp shipped with base install"
    All HTTP-based operations (health checks, remote commands, status queries) use `aiohttp`, which is included in the base `pip install promptise`.

!!! warning "Plain HTTP"
    The coordinator authenticates with bearer tokens but talks to nodes over plain HTTP unless the node URLs are `https://`. Across an untrusted network, put the nodes behind a TLS reverse proxy.

!!! warning "Coordinator is a single point of failure"
    The coordinator itself is not replicated. For high availability, run the coordinator behind a load balancer or implement coordinator election.

---

## What's Next

- [Discovery and Transport](discovery-transport.md) -- service discovery and HTTP transport API
- [Configuration](../configuration.md) -- `DistributedConfig` reference
- [Runtime Manager](../runtime-manager.md) -- the `AgentRuntime` that runs on each node
