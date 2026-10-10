"""MCP tools for sandbox interaction.

This module provides LangChain-compatible tools that agents can use to
interact with the sandbox environment.
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from .session import SandboxSession


class SandboxExecInput(BaseModel):
    """Input schema for sandbox_exec tool."""

    command: str = Field(..., description="Shell command to execute in the sandbox")
    timeout: int | None = Field(None, description="Optional timeout in seconds (default: 300)")


class SandboxReadFileInput(BaseModel):
    """Input schema for sandbox_read_file tool."""

    file_path: str = Field(..., description="Path to file inside sandbox")


class SandboxWriteFileInput(BaseModel):
    """Input schema for sandbox_write_file tool."""

    file_path: str = Field(..., description="Path to file inside sandbox")
    content: str = Field(..., description="File contents to write")


class SandboxListFilesInput(BaseModel):
    """Input schema for sandbox_list_files tool."""

    directory: str = Field("/workspace", description="Directory path inside sandbox")


class SandboxInstallPackageInput(BaseModel):
    """Input schema for sandbox_install_package tool."""

    package: str = Field(..., description="Package name to install")
    tool: str = Field("python", description="Tool ecosystem (python, node, rust, go)")


class SandboxExecTool(BaseTool):
    """Execute command in sandbox environment.

    This tool allows the agent to run any shell command in the secure sandbox.
    The sandbox provides full CLI access with isolation from the host system.
    """

    name: str = "sandbox_exec"
    description: str = """Execute a shell command in the secure sandbox environment.

The sandbox is an isolated Linux container where you can safely:
- Run the interpreters and tools installed in its image
- Create and modify files in /workspace (the only writable directory besides /tmp)
- Run tests and experiments
- Execute any CLI commands

Network access depends on how the sandbox is configured; by default it has
none, so downloads and package installs fail.

Examples:
- sandbox_exec(command="python --version")
- sandbox_exec(command="python /workspace/script.py", timeout=60)
- sandbox_exec(command="ls -la /workspace")

The command runs in /workspace by default. Output includes stdout, stderr, and exit code.
A command that exceeds its timeout is killed.
"""
    args_schema: type[BaseModel] = SandboxExecInput

    session: SandboxSession

    def __init__(self, session: SandboxSession, **kwargs: Any):
        """Initialize tool with sandbox session."""
        super().__init__(session=session, **kwargs)

    async def _arun(self, command: str, timeout: int | None = None) -> str:
        """Execute command asynchronously."""
        try:
            result = await self.session.execute(command, timeout=timeout)

            if result.timeout:
                return f"ERROR: Command timed out after {timeout or self.session.config.timeout} seconds"

            output_parts = []

            if result.stdout:
                output_parts.append(f"STDOUT:\n{result.stdout}")

            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr}")

            output_parts.append(f"\nExit code: {result.exit_code}")
            output_parts.append(f"Duration: {result.duration:.2f}s")

            return "\n".join(output_parts)

        except Exception as e:
            return f"ERROR: Failed to execute command: {str(e)}"

    def _run(self, command: str, timeout: int | None = None) -> str:
        """Sync execution (not supported)."""
        raise NotImplementedError("Use async execution with _arun")


class SandboxReadFileTool(BaseTool):
    """Read file from sandbox environment."""

    name: str = "sandbox_read_file"
    description: str = """Read a file from the sandbox environment.

Use this to read files you've created or downloaded in the sandbox.

Examples:
- sandbox_read_file(file_path="/workspace/script.py")
- sandbox_read_file(file_path="/workspace/data/results.json")

Returns the file contents as a string.
"""
    args_schema: type[BaseModel] = SandboxReadFileInput

    session: SandboxSession

    def __init__(self, session: SandboxSession, **kwargs: Any):
        """Initialize tool with sandbox session."""
        super().__init__(session=session, **kwargs)

    async def _arun(self, file_path: str) -> str:
        """Read file asynchronously."""
        try:
            content = await self.session.read_file(file_path)
            return content
        except FileNotFoundError:
            return f"ERROR: File not found: {file_path}"
        except Exception as e:
            return f"ERROR: Failed to read file: {str(e)}"

    def _run(self, file_path: str) -> str:
        """Sync execution (not supported)."""
        raise NotImplementedError("Use async execution with _arun")


class SandboxWriteFileTool(BaseTool):
    """Write file to sandbox environment."""

    name: str = "sandbox_write_file"
    description: str = """Write a file to the sandbox environment.

Use this to create or overwrite files in the sandbox.

Examples:
- sandbox_write_file(file_path="/workspace/script.py", content="print('hello')")
- sandbox_write_file(file_path="/workspace/config.json", content='{"key": "value"}')

Returns success message or error.
"""
    args_schema: type[BaseModel] = SandboxWriteFileInput

    session: SandboxSession

    def __init__(self, session: SandboxSession, **kwargs: Any):
        """Initialize tool with sandbox session."""
        super().__init__(session=session, **kwargs)

    async def _arun(self, file_path: str, content: str) -> str:
        """Write file asynchronously."""
        try:
            await self.session.write_file(file_path, content)
            return f"Successfully wrote {len(content)} bytes to {file_path}"
        except Exception as e:
            return f"ERROR: Failed to write file: {str(e)}"

    def _run(self, file_path: str, content: str) -> str:
        """Sync execution (not supported)."""
        raise NotImplementedError("Use async execution with _arun")


class SandboxListFilesTool(BaseTool):
    """List files in sandbox directory."""

    name: str = "sandbox_list_files"
    description: str = """List files in a directory inside the sandbox.

Use this to see what files exist in the sandbox workspace.

Examples:
- sandbox_list_files(directory="/workspace")
- sandbox_list_files(directory="/workspace/data")

Returns list of file and directory names.
"""
    args_schema: type[BaseModel] = SandboxListFilesInput

    session: SandboxSession

    def __init__(self, session: SandboxSession, **kwargs: Any):
        """Initialize tool with sandbox session."""
        super().__init__(session=session, **kwargs)

    async def _arun(self, directory: str = "/workspace") -> str:
        """List files asynchronously."""
        try:
            files = await self.session.list_files(directory)
            if not files:
                return f"Directory {directory} is empty or does not exist"
            return "\n".join(files)
        except Exception as e:
            return f"ERROR: Failed to list files: {str(e)}"

    def _run(self, directory: str = "/workspace") -> str:
        """Sync execution (not supported)."""
        raise NotImplementedError("Use async execution with _arun")


class SandboxInstallPackageTool(BaseTool):
    """Install package in sandbox environment."""

    name: str = "sandbox_install_package"
    description: str = """Install a package in the sandbox environment.

Supports multiple package managers (packages go into /workspace):
- Python: pip install
- Node.js: npm install (into /workspace/node_modules)
- Rust: cargo install
- Go: go install

Downloading packages needs network access, which the sandbox does not have
unless it was configured with one.

Examples:
- sandbox_install_package(package="requests", tool="python")
- sandbox_install_package(package="express", tool="node")
- sandbox_install_package(package="ripgrep", tool="rust")

Returns installation output or error.
"""
    args_schema: type[BaseModel] = SandboxInstallPackageInput

    session: SandboxSession

    def __init__(self, session: SandboxSession, **kwargs: Any):
        """Initialize tool with sandbox session."""
        super().__init__(session=session, **kwargs)

    async def _arun(self, package: str, tool: str = "python") -> str:
        """Install package asynchronously."""
        try:
            result = await self.session.install_package(package, tool)

            if result.success:
                return f"Successfully installed {package} using {tool}\n\n{result.stdout}"
            else:
                return (
                    f"Failed to install {package} using {tool}\n\n"
                    f"Exit code: {result.exit_code}\n"
                    f"STDERR:\n{result.stderr}"
                )

        except ValueError as e:
            return f"ERROR: {str(e)}"
        except Exception as e:
            return f"ERROR: Failed to install package: {str(e)}"

    def _run(self, package: str, tool: str = "python") -> str:
        """Sync execution (not supported)."""
        raise NotImplementedError("Use async execution with _arun")


def create_sandbox_tools(session: SandboxSession) -> list[BaseTool]:
    """Create all sandbox tools for a session.

    Args:
        session: Active sandbox session

    Returns:
        List of sandbox tools
    """
    return [
        SandboxExecTool(session=session),
        SandboxReadFileTool(session=session),
        SandboxWriteFileTool(session=session),
        SandboxListFilesTool(session=session),
        SandboxInstallPackageTool(session=session),
    ]
