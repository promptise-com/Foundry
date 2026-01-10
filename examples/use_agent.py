"""
Example: Using DeepMCP with a custom model over HTTP.

Console output:
- Discovered tools (from your MCP servers)
- Each tool invocation + result (via deepmcpagent trace hooks)
- Final LLM answer
"""

import asyncio
from typing import Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from deepmcpagent import HTTPServerSpec, build_deep_agent


def _extract_text_from_message(msg: Any) -> str | None:
    """Best-effort extraction of text from a single LangGraph message."""
    try:
        content = getattr(msg, "content", None)
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list) and content and isinstance(content[0], dict):
            return content[0].get("text") or str(content)
        return None
    except Exception:
        return None


def _extract_final_answer(result: Any) -> str:
    """Best-effort extraction of the final text from different executors."""
    try:
        # LangGraph prebuilt typically returns {"messages": [...]}
        if isinstance(result, dict) and "messages" in result and result["messages"]:
            last = result["messages"][-1]
            text = _extract_text_from_message(last)
            return text or str(last)
        return str(result)
    except Exception:
        return str(result)


async def main() -> None:
    console = Console()
    load_dotenv()

    # Ensure your MCP server (e.g., math_server.py) is running in another terminal:
    #   python math_server.py
    servers = {
        "math": HTTPServerSpec(
            url="http://127.0.0.1:8000/mcp",
            transport="http",
        ),
    }

    # Any LangChain-compatible chat model (or init string) works here.
    model = ChatOpenAI(model="gpt-4.1")

    # Build the agent using your package. `trace_tools=True` prints tool calls/results.
    graph, loader = await build_deep_agent(
        servers=servers,
        model=model,
        instructions="You are a helpful agent. Use MCP math tools to solve problems.",
        trace_tools=True,
    )

    # Show discovered tools
    infos = await loader.list_tool_info()
    infos = list(infos) if infos else []

    table = Table(title="Discovered MCP Tools", show_lines=True)
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Description", style="green")
    if infos:
        for t in infos:
            table.add_row(t.name, t.description or "-")
    else:
        table.add_row("— none —", "No tools discovered (is your MCP server running?)")
    console.print(table)

    # Run a single-turn query with streaming. Tool traces will be printed automatically.
    query = "What is (3 + 5) * 7 using math tools?"
    console.print(Panel.fit(query, title="User Query", style="bold magenta"))

    console.print("\n[bold yellow]Agent Trace (Streaming)[/bold yellow]")
    final_text: str | None = None
    # Use astream_log() to get RunLogPatch.ops; astream() yields dicts without .ops
    async for patch in graph.astream_log({"messages": [{"role": "user", "content": query}]}, include_state=False):
        for op in patch.ops:
            # Look for new messages added to the conversation
            path = op.get("path") if isinstance(op, dict) else getattr(op, "path", None)
            is_add = (op.get("op") if isinstance(op, dict) else getattr(op, "op", None)) == "add"
            if not (is_add and isinstance(path, str) and path.startswith("/messages/")):
                continue
            msg = op.get("value") if isinstance(op, dict) else getattr(op, "value", None)
            if not msg:
                continue
            text = _extract_text_from_message(msg)
            if text:
                console.print(f"[cyan]{text}")
                role = getattr(msg, "role", None) or (msg.get("role") if isinstance(msg, dict) else None)
                if role == "assistant":
                    final_text = text

    console.print(Panel(final_text or "(no content)", title="Final LLM Answer", style="bold green"))


if __name__ == "__main__":
    asyncio.run(main())
