"""Container backends for sandbox execution.

This module provides abstract backend interface and concrete implementations
for different container runtimes (Docker, gVisor).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import posixpath
import shlex
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

from .config import DEFAULT_CAP_DROP, NetworkMode, SandboxConfig, parse_size
from .session import CommandResult

logger = logging.getLogger(__name__)


class SandboxBackend(ABC):
    """Abstract base class for sandbox backends.

    Defines the interface that all backend implementations must provide.
    """

    def __init__(self, config: SandboxConfig):
        """Initialize backend with configuration.

        Args:
            config: Sandbox configuration
        """
        self.config = config

    @abstractmethod
    async def create_container(self) -> str:
        """Create and start a new container.

        Returns:
            Container ID
        """
        pass

    @abstractmethod
    async def execute_command(
        self, container_id: str, command: str, timeout: int, workdir: str
    ) -> CommandResult:
        """Execute a command in the container.

        Args:
            container_id: Container ID
            command: Shell command to execute
            timeout: Timeout in seconds
            workdir: Working directory

        Returns:
            CommandResult with execution details
        """
        pass

    @abstractmethod
    async def read_file(self, container_id: str, file_path: str) -> str:
        """Read a file from the container.

        Args:
            container_id: Container ID
            file_path: Path to file inside container

        Returns:
            File contents
        """
        pass

    @abstractmethod
    async def write_file(self, container_id: str, file_path: str, content: str) -> None:
        """Write a file to the container.

        Args:
            container_id: Container ID
            file_path: Path to file inside container
            content: File contents
        """
        pass

    @abstractmethod
    async def stop_container(self, container_id: str) -> None:
        """Stop a running container.

        Args:
            container_id: Container ID
        """
        pass

    @abstractmethod
    async def remove_container(self, container_id: str) -> None:
        """Remove a container.

        Args:
            container_id: Container ID
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is available and working.

        Returns:
            True if backend is healthy
        """
        pass

    async def ensure_available(self) -> None:
        """Raise with the reason when the backend cannot run containers.

        The default implementation relies on :meth:`health_check`. Backends
        that know *why* they are unavailable override it to say so.

        Raises:
            RuntimeError: If the backend is not available.
        """
        if not await self.health_check():
            raise RuntimeError("health check failed")


# Every exec carries this variable so a timed-out command's processes can be
# found (and killed) inside the container.
EXEC_MARKER_ENV = "PROMPTISE_EXEC_ID"

# Largest base64 slice sent in one exec. A single argv string is capped at
# 128 KiB on Linux (MAX_ARG_STRLEN); a multiple of 4 keeps every slice
# independently decodable.
_WRITE_CHUNK = 96 * 1024

# Kills every process of one exec: the processes that carry its marker in
# their environment, plus all of their descendants. Processes are frozen
# (SIGSTOP) before they are killed so a fork loop cannot outrun the scan.
# POSIX sh, so it runs in any image. Exit 0 when nothing is left.
_KILL_SCRIPT = r"""
id="$1"
self="$$"
marked=""
scan() {
  for d in /proc/[0-9]*; do
    p="${d#/proc/}"
    [ "$p" = "$self" ] && continue
    case " $marked " in *" $p "*) continue ;; esac
    if tr '\0' '\n' < "$d/environ" 2>/dev/null | grep -qx "PROMPTISE_EXEC_ID=$id"; then
      marked="$marked $p"
    fi
  done
  changed=1
  while [ "$changed" = 1 ]; do
    changed=0
    for d in /proc/[0-9]*; do
      p="${d#/proc/}"
      [ "$p" = "$self" ] && continue
      case " $marked " in *" $p "*) continue ;; esac
      stat=$(cat "$d/stat" 2>/dev/null) || continue
      rest="${stat##*) }"
      set -- $rest
      case " $marked " in *" $2 "*) marked="$marked $p"; changed=1 ;; esac
    done
  done
}
for round in 1 2 3 4 5; do
  marked=""
  scan
  [ -z "$marked" ] && exit 0
  kill -STOP $marked 2>/dev/null
  scan
  kill -KILL $marked 2>/dev/null
  sleep 0.1
done
marked=""
scan
[ -z "$marked" ] && exit 0
echo "still running:$marked" >&2
exit 1
"""

# Applied by a privileged exec right after the container starts, before any
# sandboxed code runs. The sandboxed processes have no CAP_NET_ADMIN, so they
# cannot change the rules afterwards. Both IPv4 and IPv6 are filtered; any
# failure (missing binary, kernel without netfilter) aborts with a non-zero
# exit so the container is not used.
_RESTRICTED_NETWORK_SCRIPT = r"""
set -e
for t in iptables ip6tables; do
  command -v "$t" >/dev/null 2>&1 || { echo "$t is not installed in the image" >&2; exit 3; }
done
for t in iptables ip6tables; do
  "$t" -F OUTPUT
  "$t" -A OUTPUT -o lo -j ACCEPT
  "$t" -A OUTPUT -p udp --dport 53 -j ACCEPT
  "$t" -A OUTPUT -p tcp --dport 53 -j ACCEPT
  "$t" -A OUTPUT -p tcp --dport 80 -j ACCEPT
  "$t" -A OUTPUT -p tcp --dport 443 -j ACCEPT
  "$t" -P OUTPUT DROP
  "$t" -S OUTPUT | grep -qx -- "-P OUTPUT DROP" || { echo "$t policy not applied" >&2; exit 4; }
done
"""


class DockerBackend(SandboxBackend):
    """Docker-based sandbox backend with gVisor support.

    This backend uses Docker as the container runtime, with optional gVisor
    for enhanced security through user-space kernel implementation.
    """

    def __init__(self, config: SandboxConfig):
        """Initialize Docker backend.

        Args:
            config: Sandbox configuration

        Raises:
            ImportError: If the ``docker`` package is not installed.
        """
        try:
            import docker as _docker  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'docker' package is required for the sandbox. "
                "Install it with: pip install 'promptise[sandbox]'"
            ) from None
        super().__init__(config)
        self._docker_client: Any = None

    async def _get_client(self) -> Any:
        """Get or create Docker client.

        Returns:
            Docker client instance

        Raises:
            RuntimeError: If the Docker daemon cannot be reached.
        """
        if self._docker_client is None:
            import docker

            try:
                self._docker_client = docker.from_env()
            except Exception as e:
                raise RuntimeError(
                    f"Cannot connect to the Docker daemon ({e}). "
                    "Make sure Docker is installed and running."
                ) from e

        return self._docker_client

    @property
    def _runtime(self) -> str | None:
        """The Docker runtime the container runs with (``None`` = daemon default)."""
        if self.config.backend == "gvisor":
            return "runsc"
        return self.config.runtime

    def _environment(self) -> dict[str, str]:
        """Environment for the container and every exec.

        ``HOME`` points at the writable workspace so tools that write to the
        home directory (``pip`` user installs, caches) work on a read-only
        root filesystem. Values from ``config.env`` take precedence.
        """
        return {"HOME": self.config.workdir, **self.config.env}

    def _build_container_config(self) -> dict[str, Any]:
        """Build container configuration with security hardening.

        Returns:
            Docker container configuration dict
        """
        from datetime import datetime, timezone

        # Base configuration
        config: dict[str, Any] = {
            "image": self.config.image,
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "working_dir": self.config.workdir,
            "environment": self._environment(),
            "command": "/bin/bash",
            "labels": {
                "promptise.sandbox": "true",
                "promptise.sandbox.created": datetime.now(timezone.utc).isoformat(),
                "promptise.sandbox.backend": self.config.backend,
            },
        }

        # Network configuration. "restricted" starts on the bridge network;
        # create_container() then installs the egress filter or refuses to
        # hand out the container.
        if self.config.network == NetworkMode.NONE:
            config["network_mode"] = "none"
        else:
            config["network_mode"] = "bridge"

        # Docker applies its default seccomp profile (an allowlist that
        # blocks ~44 syscalls such as mount, reboot and kexec_load) and, on
        # AppArmor hosts, its docker-default AppArmor profile. Neither is
        # overridden here. no-new-privileges is always enforced, even with
        # allow_sudo.
        config["security_opt"] = ["no-new-privileges"]

        # Host config (resource limits and capabilities)
        host_config: dict[str, Any] = {
            "privileged": False,
            # Read-only rootfs is always enforced when configured,
            # regardless of allow_sudo.
            "read_only": self.config.read_only_rootfs,
        }

        # CPU limits (in nano CPUs: 1 CPU = 1e9)
        host_config["nano_cpus"] = int(self.config.cpu_limit * 1e9)

        # Memory limit
        memory_bytes = parse_size(self.config.memory_limit)
        host_config["mem_limit"] = memory_bytes
        host_config["memswap_limit"] = memory_bytes  # Disable swap

        # Process limit (contains fork bombs)
        host_config["pids_limit"] = self.config.pids_limit

        # Always drop dangerous capabilities.  If allow_sudo is True,
        # add back only CAP_SETUID/CAP_SETGID for sudo functionality.
        # CAP_SYS_ADMIN is NEVER re-added — it enables mount, ptrace,
        # BPF, and namespace manipulation which are container escape vectors.
        host_config["cap_drop"] = list(DEFAULT_CAP_DROP)
        if self.config.allow_sudo:
            for cap in ("CAP_SETUID", "CAP_SETGID"):
                if cap in host_config["cap_drop"]:
                    host_config["cap_drop"].remove(cap)

        if self._runtime:
            host_config["runtime"] = self._runtime

        # Writable areas are size-capped tmpfs mounts. The workspace is sized
        # by disk_limit; /tmp and /var/tmp never exceed it either.
        disk_bytes = parse_size(self.config.disk_limit)
        tmp_bytes = min(disk_bytes, 1024**3)
        var_tmp_bytes = min(disk_bytes, 512 * 1024**2)
        host_config["tmpfs"] = {
            "/tmp": f"rw,noexec,nosuid,size={tmp_bytes}",  # nosec B108 - path inside sandboxed container, not host
            "/var/tmp": f"rw,noexec,nosuid,size={var_tmp_bytes}",  # nosec B108 - path inside sandboxed container, not host
            self.config.workdir: f"rw,size={disk_bytes}",
        }

        config["host_config"] = host_config

        return config

    def _parse_size(self, size_str: str) -> int:
        """Parse size string to bytes.

        Args:
            size_str: Size string (e.g., "4G", "512M")

        Returns:
            Size in bytes
        """
        return parse_size(size_str)

    async def create_container(self) -> str:
        """Create and start a new Docker container.

        Returns:
            Container ID

        Raises:
            RuntimeError: If container creation fails, or if
                ``network="restricted"`` cannot be enforced (the container is
                removed and never used).
        """
        client = await self._get_client()
        config = self._build_container_config()

        try:
            # Pull image if not present
            try:
                client.images.get(self.config.image)
            except Exception:
                logger.info("Pulling image %s...", self.config.image)
                client.images.pull(self.config.image)

            # Modern Docker SDK expects parameters flattened (not nested in host_config)
            host_config_dict = config.pop("host_config", {})
            config.update(host_config_dict)

            container = client.containers.create(**config)
            container.start()
        except Exception as e:
            raise RuntimeError(f"Failed to create container: {e}") from e

        if self.config.network == NetworkMode.RESTRICTED:
            try:
                await self._setup_network_restrictions(container)
            except Exception:
                try:
                    container.remove(force=True)
                except Exception:
                    logger.warning("Failed to remove container %s", container.id, exc_info=True)
                raise

        return str(container.id)

    async def _setup_network_restrictions(self, container: Any) -> None:
        """Install the egress filter for ``network="restricted"``.

        Allows loopback, DNS (port 53) and TCP 80/443 to any host, and drops
        every other outbound packet, for IPv4 and IPv6.

        Args:
            container: The started Docker container.

        Raises:
            RuntimeError: If the filter cannot be installed. The sandbox
                fails closed instead of running with an open network.
        """

        def _apply() -> Any:
            return container.exec_run(
                cmd=["sh", "-c", _RESTRICTED_NETWORK_SCRIPT],
                user="root",
                privileged=True,
                demux=True,
            )

        try:
            result = await asyncio.get_running_loop().run_in_executor(None, _apply)
            exit_code = result.exit_code
            _, stderr_bytes = result.output
            detail = (stderr_bytes or b"").decode("utf-8", errors="replace").strip()
        except Exception as e:
            exit_code, detail = -1, str(e)

        if exit_code != 0:
            raise RuntimeError(
                "network='restricted' could not be enforced, so the sandbox refused "
                f"to start ({detail or f'exit code {exit_code}'}). The filter needs "
                "iptables and ip6tables in the image. Use network='none' (the "
                "default), an image that ships iptables, or network='full'."
            )
        logger.info("[sandbox] Network restrictions applied to %s", container.id)

    async def execute_command(
        self, container_id: str, command: str, timeout: int, workdir: str
    ) -> CommandResult:
        """Execute a command in Docker container with proper timeout enforcement.

        When the timeout expires, every process the command started is killed
        inside the container (see :meth:`_kill_exec`), so a runaway program
        does not keep burning CPU after the call returns.

        Args:
            container_id: Container ID
            command: Shell command to execute
            timeout: Timeout in seconds (actually enforced)
            workdir: Working directory

        Returns:
            CommandResult with execution details
        """
        from concurrent.futures import ThreadPoolExecutor

        try:
            client = await self._get_client()
            container = client.containers.get(container_id)
        except Exception as e:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=f"Execution failed: {e}",
                timeout=False,
                duration=0.0,
            )

        exec_id = uuid.uuid4().hex
        environment = {**self._environment(), EXEC_MARKER_ENV: exec_id}

        def _exec() -> Any:
            return container.exec_run(
                cmd=["bash", "-c", command],
                workdir=workdir,
                demux=True,  # Separate stdout/stderr
                environment=environment,
            )

        start_time = time.time()
        executor = ThreadPoolExecutor(max_workers=1)
        future = asyncio.get_running_loop().run_in_executor(executor, _exec)
        timed_out = False
        try:
            try:
                exec_result = await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
            except asyncio.TimeoutError:
                timed_out = True
                note = f"Command timed out after {timeout} seconds"
                if await self._kill_exec(container, exec_id):
                    note += "; its processes were killed"
                elif await self._restart_container(container):
                    # Last resort (e.g. a fork bomb at the pids limit leaves
                    # no room for the kill script): restarting kills every
                    # process in the container.
                    note += (
                        "; its processes could not be killed individually, so the "
                        "sandbox container was restarted and the workspace was reset"
                    )
                else:
                    note += (
                        "; its processes could not be killed and the sandbox "
                        "container could not be restarted"
                    )
                # Once its processes are gone the exec returns; collect
                # whatever it printed before the deadline.
                try:
                    exec_result = await asyncio.wait_for(future, timeout=10)
                except Exception:
                    exec_result = None
                stdout, stderr = self._decode(exec_result)
                return CommandResult(
                    exit_code=-1,
                    stdout=stdout,
                    stderr=(stderr + "\n" if stderr else "") + note,
                    timeout=True,
                    duration=time.time() - start_time,
                )
        except Exception as e:
            return CommandResult(
                exit_code=-1,
                stdout="",
                stderr=f"Execution failed: {e}",
                timeout=timed_out,
                duration=time.time() - start_time,
            )
        finally:
            executor.shutdown(wait=False)

        stdout, stderr = self._decode(exec_result)
        return CommandResult(
            exit_code=exec_result.exit_code,
            stdout=stdout,
            stderr=stderr,
            timeout=False,
            duration=time.time() - start_time,
        )

    @staticmethod
    def _decode(exec_result: Any) -> tuple[str, str]:
        """Decode a demuxed ``exec_run`` result into (stdout, stderr)."""
        if exec_result is None or exec_result.output is None:
            return "", ""
        stdout_bytes, stderr_bytes = exec_result.output
        stdout = stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else ""
        stderr = stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else ""
        return stdout, stderr

    async def _kill_exec(self, container: Any, exec_id: str) -> bool:
        """Kill every process started by one exec.

        Finds the processes that carry the exec's marker variable plus all of
        their descendants, freezes and kills them. A process that both clears
        its environment and detaches from its parent can escape this; it
        still lives inside the container's CPU, memory and pids limits and
        dies when the session is cleaned up.

        Args:
            container: Docker container object.
            exec_id: The marker value of the exec to kill.

        Returns:
            True if no process of the exec is left.
        """

        def _kill() -> Any:
            return container.exec_run(
                cmd=["sh", "-c", _KILL_SCRIPT, "sh", exec_id],
                user="root",
                privileged=True,
                demux=True,
            )

        try:
            result = await asyncio.get_running_loop().run_in_executor(None, _kill)
        except Exception:
            logger.error("[sandbox] Failed to kill timed-out command", exc_info=True)
            return False
        if result.exit_code != 0:
            _, stderr = self._decode(result)
            logger.error("[sandbox] Timed-out command not fully killed: %s", stderr.strip())
            return False
        return True

    async def _restart_container(self, container: Any) -> bool:
        """Restart the container after a timed-out command could not be killed.

        Restarting kills every process in the container (and empties the
        tmpfs workspace). For ``network="restricted"`` the egress filter is
        installed again; if that fails the container is stopped, so it never
        runs with an open network.

        Args:
            container: Docker container object.

        Returns:
            True if the container is running again and safe to use.
        """
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, lambda: container.restart(timeout=1))
        except Exception:
            logger.error("[sandbox] Failed to restart container %s", container.id, exc_info=True)
            return False
        logger.warning("[sandbox] Restarted container %s to stop a timed-out command", container.id)
        if self.config.network == NetworkMode.RESTRICTED:
            try:
                await self._setup_network_restrictions(container)
            except Exception:
                logger.error(
                    "[sandbox] Network filter not re-applied after restart; stopping %s",
                    container.id,
                    exc_info=True,
                )
                try:
                    await loop.run_in_executor(None, lambda: container.stop(timeout=1))
                except Exception:
                    logger.error("[sandbox] Failed to stop container %s", container.id)
                return False
        return True

    async def read_file(self, container_id: str, file_path: str) -> str:
        """Read a file from Docker container.

        Args:
            container_id: Container ID
            file_path: Path to file

        Returns:
            File contents

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        result = await self.execute_command(
            container_id,
            f"cat -- {shlex.quote(file_path)}",
            timeout=30,
            workdir="/tmp",  # nosec B108 - container path, not host
        )

        if not result.success:
            raise FileNotFoundError(f"File not found or not readable: {file_path}")

        return result.stdout

    async def write_file(self, container_id: str, file_path: str, content: str) -> None:
        """Write a file into the container from inside it.

        Docker's archive API cannot write to a container with a read-only
        root filesystem (nor into its tmpfs mounts), so the content is
        streamed in base64 slices through ``exec`` and decoded in the
        container. Arbitrary content (quotes, newlines, shell metacharacters)
        is never interpreted by the shell. The file is written to a temporary
        name and renamed into place, and missing parent directories are
        created.

        Args:
            container_id: Container ID
            file_path: Absolute path to the file inside the container
            content: File contents

        Raises:
            RuntimeError: If the file cannot be written.
        """
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        chunks = [encoded[i : i + _WRITE_CHUNK] for i in range(0, len(encoded), _WRITE_CHUNK)] or [
            ""
        ]
        target = shlex.quote(file_path)
        tmp = shlex.quote(f"{file_path}.{uuid.uuid4().hex}.tmp")
        parent = shlex.quote(posixpath.dirname(file_path) or "/")
        workdir = "/tmp"  # nosec B108 - container path, not host

        commands = []
        for index, chunk in enumerate(chunks):
            redirect = ">" if index == 0 else ">>"
            cmd = f"printf %s {shlex.quote(chunk)} | base64 -d {redirect} {tmp}"
            if index == 0:
                cmd = f"mkdir -p -- {parent} && {cmd}"
            if index == len(chunks) - 1:
                cmd = f"{cmd} && mv -f -- {tmp} {target}"
            commands.append(cmd)

        for cmd in commands:
            result = await self.execute_command(container_id, cmd, timeout=60, workdir=workdir)
            if not result.success:
                await self.execute_command(
                    container_id, f"rm -f -- {tmp}", timeout=10, workdir=workdir
                )
                detail = result.stderr.strip() or f"exit code {result.exit_code}"
                raise RuntimeError(f"Failed to write file {file_path}: {detail}")

    async def stop_container(self, container_id: str) -> None:
        """Stop Docker container.

        Args:
            container_id: Container ID
        """
        client = await self._get_client()

        try:
            container = client.containers.get(container_id)
            container.stop(timeout=5)
        except Exception as e:
            logger.warning("[sandbox] Failed to stop container %s: %s", container_id, e)

    async def remove_container(self, container_id: str) -> None:
        """Remove Docker container.

        Args:
            container_id: Container ID
        """
        client = await self._get_client()

        try:
            container = client.containers.get(container_id)
            container.remove(force=True)
        except Exception as e:
            logger.warning("[sandbox] Failed to remove container %s: %s", container_id, e)

    async def ensure_available(self) -> None:
        """Check that Docker answers and the requested runtime is installed.

        Raises:
            RuntimeError: With the actual reason: the daemon is unreachable,
                or the configured runtime (``runsc`` for gVisor) is not
                registered with Docker.
        """
        client = await self._get_client()
        try:
            client.ping()
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to the Docker daemon ({e}). "
                "Make sure Docker is installed and running."
            ) from e

        runtime = self._runtime
        if runtime:
            await self._verify_runtime_available(client, runtime)

    async def health_check(self) -> bool:
        """Check if Docker is available and the requested runtime is configured.

        Returns:
            True if Docker is healthy (and the runtime is available if requested)
        """
        try:
            await self.ensure_available()
            return True
        except Exception:
            return False

    async def _verify_runtime_available(self, client: Any, runtime: str) -> None:
        """Verify a Docker runtime (e.g. gVisor's ``runsc``) is registered.

        The framework does not silently fall back to the default runtime.

        Args:
            client: Docker client.
            runtime: Runtime name.

        Raises:
            RuntimeError: If the runtime is not registered with Docker.
        """
        try:
            info = client.info()
        except Exception as exc:
            raise RuntimeError(f"Cannot verify Docker runtime {runtime!r}: {exc}") from exc

        runtimes = info.get("Runtimes", {}) or {}
        if runtime in runtimes:
            return
        available = ", ".join(sorted(runtimes)) or "none reported"
        if runtime == "runsc":
            raise RuntimeError(
                "gVisor runtime 'runsc' is not registered with Docker "
                f"(available runtimes: {available}). Install gVisor from "
                "https://gvisor.dev/docs/user_guide/install/ and restart Docker, "
                "or set backend='docker' to use the default runtime."
            )
        raise RuntimeError(
            f"Docker runtime {runtime!r} is not registered with Docker "
            f"(available runtimes: {available})."
        )
