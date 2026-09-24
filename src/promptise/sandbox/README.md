# Sandbox Module

This module provides a secure, isolated sandbox environment for agent code execution.

## Features

- **Zero-configuration**: Enable with `sandbox=True`
- **Full CLI access**: Execute any shell command safely
- **Multiple backends**: Docker, gVisor
- **7-layer security**: gVisor, seccomp, AppArmor, capabilities, read-only FS, resource limits
- **Resource management**: CPU, memory, disk, network quotas
- **Network isolation**: Three modes (none, restricted, full)

## Architecture

```
sandbox/
├── __init__.py          # Public API
├── config.py            # Configuration schemas and defaults
├── backends.py          # Backend implementations (Docker, gVisor, etc.)
├── manager.py           # Lifecycle management
├── session.py           # Command execution session
└── tools.py             # LangChain tools for MCP integration
```

## Components

### `SandboxConfig`

Configuration schema with validation:

```python
from promptise.sandbox import SandboxConfig

config = SandboxConfig(backend="gvisor", cpu_limit=4, memory_limit="8G", network="restricted")
```

### `SandboxBackend`

Abstract backend interface with implementations:

- **`DockerBackend`**: Docker-based backend with optional gVisor runtime

### `SandboxManager`

Manages sandbox lifecycle:

```python
from promptise.sandbox import SandboxManager

async with SandboxManager(config) as manager:
    session = await manager.create_session()
    # Use session...
```

### `SandboxSession`

Executes commands in the sandbox:

```python
async with await manager.create_session() as session:
    result = await session.execute("python script.py")
    print(result.stdout)
```

### MCP Tools

Five LangChain tools for agent interaction:

1. **`sandbox_exec`** - Execute shell commands
2. **`sandbox_read_file`** - Read files
3. **`sandbox_write_file`** - Write files
4. **`sandbox_list_files`** - List directory contents
5. **`sandbox_install_package`** - Install packages

## Security Architecture

### 7 Layers of Defense

1. **gVisor Runtime** (optional): User-space kernel with syscall interception
2. **Seccomp**: Syscall filtering (~200 allowed syscalls)
3. **AppArmor**: Mandatory Access Control
4. **Capability Dropping**: Remove ~40 dangerous capabilities
5. **Read-only Root FS**: Prevent system file modification
6. **No Privileged Mode**: Non-privileged container
7. **Resource Limits**: CPU, memory, disk quotas via cgroups

### Security Profiles

**Default Seccomp Profile** (`config.py`):
- Whitelist of essential syscalls
- Blocks dangerous operations (reboot, kernel module loading, etc.)
- ~200 allowed syscalls for normal operation

**Default AppArmor Profile** (`config.py`):
- Restricts file system access
- Allows workspace modifications only
- Denies access to sensitive host files

**Default Capability Drop** (`config.py`):
- Drops all unnecessary Linux capabilities
- Prevents privilege escalation
- Maintains minimal capabilities for operation

## Usage

### Simple (with defaults)

```python
from promptise import build_agent

agent = await build_agent(
    model="anthropic:claude-sonnet-4.5",
    instructions="You are a coding assistant.",
    servers={},
    sandbox=True,  # ✨ Zero-config
)
```

### Advanced (custom configuration)

```python
agent = await build_agent(
    model="anthropic:claude-opus-4.5",
    instructions="You are a coding assistant.",
    servers={},
    sandbox={
        "backend": "gvisor",
        "cpu_limit": 4,
        "memory_limit": "8G",
        "network": "restricted",
        "tools": ["python", "node", "rust"],
    },
)
```

### Direct API Usage

```python
from promptise.sandbox import SandboxManager, SandboxConfig

# Create config
config = SandboxConfig(backend="gvisor", cpu_limit=2)

# Create manager
async with SandboxManager(config) as manager:
    # Create session
    async with await manager.create_session() as session:
        # Execute command
        result = await session.execute("echo 'Hello, sandbox!'")
        print(result.stdout)

        # Write file
        await session.write_file("/workspace/test.py", "print('test')")

        # Read file
        content = await session.read_file("/workspace/test.py")

        # Install package
        await session.install_package("requests", tool="python")
```

## Network Modes

### None (`network="none"`)
- Complete isolation
- No network access
- Best for: Pure computation

### Restricted (`network="restricted"`) - **Default**
- Limited network
- DNS, HTTP, HTTPS allowed
- Other ports blocked
- Best for: Package installation, API calls

### Full (`network="full"`)
- Full network access
- All protocols allowed
- Best for: Web scraping, downloads

## Backends

### Docker (Default)

**Requirements:**
- Docker installed and running
- User in `docker` group (Linux)

**Configuration:**
```python
sandbox = {"backend": "docker"}
```

### gVisor (Recommended for Production)

**Requirements:**
- Docker installed
- gVisor runtime (`runsc`) installed

**Configuration:**
```python
sandbox = {"backend": "gvisor"}
```

**Installation:**
```bash
# Ubuntu/Debian
curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" | sudo tee /etc/apt/sources.list.d/gvisor.list
sudo apt-get update && sudo apt-get install -y runsc
sudo runsc install
sudo systemctl restart docker
```

## Testing

Run tests:

```bash
pytest tests/test_sandbox.py -v
```

Tests include:
- Configuration validation
- Manager lifecycle
- Session operations
- Tool creation
- Backend selection

## Dependencies

**Required:**
- `pydantic>=2.8` - Configuration validation
- `langchain_core` - Tool interfaces

**Optional:**
- `docker>=7.0.0` - Docker backend (install with `pip install "promptise[all]"`)

## Contributing

To add a new backend:

1. Subclass `SandboxBackend` in `backends.py`
2. Implement all abstract methods
3. Add backend selection in `SandboxManager._create_backend()`
4. Add tests for the new backend
5. Update documentation

## License

Apache License 2.0
