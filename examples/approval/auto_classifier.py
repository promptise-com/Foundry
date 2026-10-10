"""AutoApprovalClassifier — 6-layer decision hierarchy demo.

Demonstrates:
  - Deny rules (argument inspection) — checked first, so deny always wins
  - Ask rules (always send to a human)
  - Allow rules (glob patterns)
  - Read-only auto-allow (and a destructive verb that blocks it)
  - LLM classifier (simulated)
  - Fallback to a human handler
  - Stats inspection

Run:
    .venv/bin/python examples/approval/auto_classifier.py
"""

from __future__ import annotations

import asyncio
import time

from promptise.approval import ApprovalDecision, ApprovalRequest
from promptise.approval_classifier import (
    ApprovalRule,
    AutoApprovalClassifier,
)

# ---------------------------------------------------------------------------
# A simple fallback handler (simulates a human saying "yes")
# ---------------------------------------------------------------------------


class FakeHumanHandler:
    """Simulates a human reviewer who always approves (for demo purposes)."""

    def __init__(self) -> None:
        self.requests: list[ApprovalRequest] = []

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        print(f"    [human] Reviewing {request.tool_name}... approved")
        return ApprovalDecision(
            approved=True,
            reviewer_id="human-reviewer",
            reason="human approved",
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def main() -> None:
    human = FakeHumanHandler()

    # Simulated LLM classifier
    async def llm_check(request: ApprovalRequest) -> tuple[str, str]:
        """Simulate an LLM safety check."""
        if "DROP" in str(request.arguments):
            return "deny", "LLM detected destructive SQL"
        return "escalate", "LLM unsure — sending to human"

    classifier = AutoApprovalClassifier(
        deny_rules=[
            ApprovalRule(tool="exec_shell", reason="shell access is never allowed"),
            ApprovalRule(
                tool="*",
                argument_contains="rm -rf",
                reason="destructive filesystem command",
            ),
        ],
        ask_rules=[
            ApprovalRule(tool="get_payroll", reason="salary data needs a person"),
        ],
        allow_rules=[
            ApprovalRule(
                tool="update_profile",
                user="admin@acme.com",
                reason="admin can always update profiles",
            ),
        ],
        read_only_auto_allow=True,
        llm_classifier=llm_check,
        fallback=human,
    )

    # -----------------------------------------------------------------------
    # Test various scenarios
    # -----------------------------------------------------------------------

    scenarios = [
        ("get_user_profile", {}, None),
        ("search_documents", {"query": "quarterly report"}, None),
        ("get_payroll", {"month": "2026-09"}, None),
        ("fetch_and_purge_cache", {}, None),
        ("exec_shell", {"cmd": "ls -la"}, None),
        ("deploy_service", {"cmd": "rm -rf /tmp/old"}, None),
        ("run_query", {"sql": "DROP TABLE users"}, None),
        ("send_email", {"to": "boss@acme.com", "body": "Hi"}, None),
        ("update_profile", {"name": "Alice"}, "admin@acme.com"),
    ]

    for tool, args, user in scenarios:
        request = ApprovalRequest(
            request_id=f"req-{tool}",
            tool_name=tool,
            arguments=args,
            caller_user_id=user,
            timestamp=time.time(),
        )

        decision = await classifier.request_approval(request)
        trace = decision.trace  # per decision: safe with concurrent requests

        status = "APPROVED" if decision.approved else "DENIED"
        print(f"\n  {tool}({args})")
        print(f"    → {status} via layer={trace.layer}, reason={decision.reason!r}")

    # -----------------------------------------------------------------------
    # Stats summary
    # -----------------------------------------------------------------------

    s = classifier.stats
    print("\n--- Stats ---")
    print(f"  Allow rule hits:   {s.allow_rule_hits}")
    print(f"  Deny rule hits:    {s.deny_rule_hits}")
    print(f"  Ask rule hits:     {s.ask_rule_hits}")
    print(f"  Read-only allows:  {s.read_only_allows}")
    print(f"  LLM allows:        {s.llm_allows}")
    print(f"  LLM denies:        {s.llm_denies}")
    print(f"  LLM escalations:   {s.llm_escalations}")
    print(f"  Fallback allows:   {s.fallback_allows}")
    print(f"  Fallback denies:   {s.fallback_denies}")
    print(f"  Human requests:    {len(human.requests)}")


if __name__ == "__main__":
    asyncio.run(main())
