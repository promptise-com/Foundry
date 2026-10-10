"""Multi-User Chat — Isolated memory and conversations per user.

Demonstrates:
- SQLiteConversationStore for persistent chat history, owned per user
- ChromaProvider (PER_USER scope) for long-term vector memory across sessions
- chat(user_id=...) — scopes session ownership *and* memory to that user
- Memory settings on build_agent(): max results, minimum cosine score, timeout
- CLI with user switching to show complete isolation

Run:
    python examples/memory/multiuser_chat.py

Requires OPENAI_API_KEY and the [all] extra (chromadb).
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# ═══════════════════════════════════════════════════════════════════════════════
# Colors
# ═══════════════════════════════════════════════════════════════════════════════

DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
GREEN = "\033[32m"
BLUE = "\033[34m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"

USER_COLORS = {"alice": BLUE, "bob": MAGENTA}

# ═══════════════════════════════════════════════════════════════════════════════
# Knowledge tools
# ═══════════════════════════════════════════════════════════════════════════════

from promptise.mcp.server import MCPServer

server = MCPServer("knowledge")


@server.tool()
async def search_docs(query: str) -> str:
    """Search the company documentation."""
    docs = {
        "pricing": "Starter: $29/mo, Pro: $99/mo, Enterprise: $499/mo. All plans include core features. Enterprise adds SSO, audit logs, and dedicated support.",
        "api": "REST API at api.example.com. Rate limit: 100 req/min (Starter), 1000 (Pro), unlimited (Enterprise). Auth via Bearer token.",
        "security": "SOC2 Type II certified. Data encrypted at rest (AES-256) and in transit (TLS 1.3). GDPR compliant. Data residency: US, EU, APAC.",
        "integrations": "Native: Slack, GitHub, Jira, Linear. API: webhooks, REST, GraphQL. SDKs: Python, Node.js, Go.",
    }
    for key, content in docs.items():
        if key in query.lower():
            return content
    return "No documentation found for that topic. Try: pricing, api, security, integrations."


@server.tool()
async def get_account_info(user_id: str) -> str:
    """Get account information for a user."""
    accounts = {
        "alice": "Alice Chen | Plan: Pro ($99/mo) | Since: 2024-01 | Usage: 450/1000 API calls this month",
        "bob": "Bob Martinez | Plan: Enterprise ($499/mo) | Since: 2023-06 | Usage: 2,100 API calls this month",
    }
    return accounts.get(user_id, f"No account found for {user_id}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


async def main():
    from promptise import build_agent
    from promptise.config import StdioServerSpec
    from promptise.conversations import SQLiteConversationStore
    from promptise.memory import ChromaProvider, MemoryScope

    print(f"""
{BOLD}╔════════════════════════════════════════════════════════╗
║         Multi-User Chat — Memory Isolation Demo          ║
║        SQLite conversations · Per-user vector memory     ║
╚════════════════════════════════════════════════════════╝{RESET}

{DIM}Commands:{RESET}
  {CYAN}/user alice|bob{RESET}    — Switch user identity
  {CYAN}/history{RESET}           — Show conversation history for current user
  {CYAN}/memory{RESET}            — Show what the agent remembers about current user
  {CYAN}/whoami{RESET}            — Show current identity
  {CYAN}/quit{RESET}              — Exit

{DIM}Try: as Alice, say "I only use the Python SDK". Switch to Bob and ask which SDK
he uses — the agent does not know. Switch back to Alice and ask again — it remembers,
even in a new session, because long-term memory is kept per user.{RESET}
""")

    # Save server to temp file
    server_code = """
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from examples.memory.multiuser_chat import server
server.run(transport="stdio")
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir=".") as f:
        f.write(server_code)
        tmp_server = f.name

    db_path = "./multiuser_chat.db"
    memory_dir = tempfile.mkdtemp(prefix="multiuser-chat-memory-")

    try:
        conversation_store = SQLiteConversationStore(db_path)
        # PER_USER: every search and write is scoped to the caller's user id,
        # so Alice's memories are never injected into Bob's prompts.
        memory = ChromaProvider(persist_directory=memory_dir, scope=MemoryScope.PER_USER)
        # Load Chroma's embedding model now, so the first message does not
        # spend its memory timeout on it.
        await memory.search("warm-up", limit=1, user_id="alice")

        agent = await build_agent(
            model="openai:gpt-5-mini",
            servers={
                "docs": StdioServerSpec(command=sys.executable, args=[tmp_server]),
            },
            instructions=(
                "You are a helpful product support assistant. Use search_docs to answer questions "
                "about pricing, API, security, and integrations. Use get_account_info to look up "
                "the current user's account details. Be concise and helpful."
            ),
            conversation_store=conversation_store,
            memory=memory,
            memory_auto_store=True,
            memory_max_results=3,
            memory_min_score=0.2,  # cosine similarity: keep related memories, drop noise
            memory_timeout=15.0,
            max_agent_iterations=25,
        )

        current_user = "alice"
        sessions: dict[str, str] = {
            "alice": "session-alice-001",
            "bob": "session-bob-001",
        }

        print(f"  {USER_COLORS['alice']}{BOLD}● Logged in as: alice{RESET}\n")

        while True:
            try:
                color = USER_COLORS.get(current_user, DIM)
                user_input = input(f"  {color}{current_user}{RESET} > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not user_input:
                continue

            # Commands
            if user_input.startswith("/"):
                cmd = user_input.lower().split()

                if cmd[0] == "/quit":
                    break

                elif cmd[0] == "/user" and len(cmd) > 1:
                    new_user = cmd[1]
                    if new_user not in ("alice", "bob"):
                        print(f"  {DIM}Unknown user. Use alice or bob.{RESET}")
                        continue
                    current_user = new_user
                    color = USER_COLORS.get(current_user, DIM)
                    print(f"\n  {color}{BOLD}● Switched to: {current_user}{RESET}")
                    print(f"  {DIM}Session: {sessions[current_user]}{RESET}\n")

                elif cmd[0] == "/history":
                    session_id = sessions[current_user]
                    try:
                        messages = await conversation_store.load_messages(session_id)
                        if not messages:
                            print(f"  {DIM}No conversation history for {current_user}.{RESET}")
                        else:
                            print(
                                f"\n  {BOLD}History for {current_user} ({len(messages)} messages):{RESET}"
                            )
                            for msg in messages[-6:]:
                                role = getattr(msg, "role", "?")
                                content = getattr(msg, "content", str(msg))[:80]
                                icon = "👤" if role == "user" else "🤖"
                                print(f"    {icon} {content}")
                            print()
                    except Exception:
                        print(f"  {DIM}No history yet.{RESET}")

                elif cmd[0] == "/memory":
                    results = await memory.search(
                        "What do I know about this user?", limit=5, user_id=current_user
                    )
                    if not results:
                        print(f"  {DIM}No memories stored for {current_user}.{RESET}")
                    else:
                        print(
                            f"\n  {BOLD}Memory for {current_user} ({len(results)} entries):{RESET}"
                        )
                        for r in results:
                            print(f"    {DIM}• {r.score:.2f}  {r.content[:80]}{RESET}")
                        print()

                elif cmd[0] == "/whoami":
                    color = USER_COLORS.get(current_user, "")
                    print(f"\n  {color}{BOLD}User: {current_user}{RESET}")
                    print(f"  {DIM}Session: {sessions[current_user]}{RESET}\n")

                else:
                    print(f"  {DIM}Commands: /user, /history, /memory, /whoami, /quit{RESET}")
                continue

            # Chat with the agent. user_id= is shorthand for
            # caller=CallerContext(user_id=...): it checks that the session
            # belongs to this user and scopes memory search and auto-store to them.
            session_id = sessions[current_user]

            print(f"  {DIM}Thinking...{RESET}")
            try:
                response = await agent.chat(
                    user_input,
                    session_id=session_id,
                    user_id=current_user,
                )
                color = USER_COLORS.get(current_user, GREEN)
                print(f"\n  {color}{response}{RESET}\n")
            except Exception as exc:
                print(f"  {DIM}Error: {exc}{RESET}")

        await agent.shutdown()

    finally:
        os.unlink(tmp_server)
        if os.path.exists(db_path):
            os.unlink(db_path)
        shutil.rmtree(memory_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
