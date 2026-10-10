# AutoApprovalClassifier

An explicit decision hierarchy that sits in front of your approval handler. Instead of sending every tool call to a human, the classifier evaluates six ordered layers — deny rules, ask rules, allow rules, read-only detection, an optional LLM classifier, and finally the fallback handler — and returns the first definitive answer.

```python
from promptise import (
    ApprovalPolicy,
    AutoApprovalClassifier,
    ApprovalRule,
    WebhookApprovalHandler,
)

classifier = AutoApprovalClassifier(
    deny_rules=[ApprovalRule(tool="exec_shell", reason="too risky")],
    ask_rules=[ApprovalRule(tool="delete_*", reason="a person decides")],
    allow_rules=[ApprovalRule(tool="add_ticket_note", reason="internal note")],
    read_only_auto_allow=True,
    fallback=WebhookApprovalHandler(url="https://approvals.internal/api"),
)

policy = ApprovalPolicy(
    tools=["*"],
    handler=classifier,  # drop-in replacement
)
```

---

## The 6-layer hierarchy

Evaluated in order. First definitive answer wins.

| Layer | What it does | Result |
|---|---|---|
| **1. Deny rules** | Pattern/predicate matching. First match → deny. | `approved=False` |
| **2. Ask rules** | Pattern/predicate matching. First match → straight to the fallback handler, skipping layers 3–5. | whatever the human says |
| **3. Allow rules** | Pattern/predicate matching. First match → approve. | `approved=True` |
| **4. Read-only auto-allow** | The tool is read-only: annotated `readOnlyHint=True`, or named `get_…`, `list_…`, `read_…`, etc. See [Read-only auto-allow](#read-only-auto-allow). | `approved=True` |
| **5. LLM classifier** | Optional async function returns `"allow"`, `"deny"`, or `"escalate"`. | allow/deny or fall through |
| **6. Fallback handler** | Your existing `ApprovalHandler` (webhook, queue, callback). | whatever the human says |

**Deny always wins.** Deny rules run before allow rules, so a broad allow rule can't approve a call a deny rule forbids. An admin bypass (`ApprovalRule(user="admin@acme.com")`) in `allow_rules` still won't auto-approve `delete_customer` when `deny_rules` has `ApprovalRule(tool="delete_*")`.

**Ask rules always reach a person.** Use them for calls that must never be auto-approved, even when an allow rule, the read-only check or the LLM classifier would let them through:

```python
classifier = AutoApprovalClassifier(
    ask_rules=[
        ApprovalRule(tool="get_payroll", reason="salary data needs a person"),
        ApprovalRule(tool="send_*", argument_contains="@press.", reason="press contact"),
    ],
    allow_rules=[ApprovalRule(tool="send_*", reason="routine email")],
    fallback=human_handler,
)
```

!!! note "Upgrading from 1.2.x"
    Up to 1.2.1, allow rules were checked *before* deny rules, so the first matching allow rule won. If you relied on an allow rule to carve an exception out of a broader deny rule, narrow the deny rule (or use a predicate) instead.

---

## ApprovalRule

Rules match by tool glob, user ID, argument substring, or async predicate. All non-empty filters must match (AND logic). The same rule type is used for deny, ask and allow rules.

```python
# Simple glob match
ApprovalRule(tool="delete_*", reason="destructive")

# User-scoped
ApprovalRule(tool="*", user="admin@acme.com", reason="admin bypass")

# Argument inspection
ApprovalRule(tool="shell", argument_contains="rm -rf", reason="dangerous command")

# Custom async predicate
async def is_small_refund(req):
    return req.arguments.get("amount", 0) < 20

ApprovalRule(tool="issue_refund", predicate=is_small_refund, reason="small refund")
```

| Field | Matches when |
|---|---|
| `tool` | `fnmatch(request.tool_name, tool)` |
| `user` | `request.caller_user_id == user` (from the [`CallerContext`](approval.md#multi-user)) |
| `argument_contains` | the substring appears in `json.dumps(arguments, sort_keys=True, ensure_ascii=False)` |
| `predicate` | `await predicate(request)` is truthy. A predicate that raises counts as *no match* (and is logged). |

### `argument_contains` matches JSON

The arguments are serialized as JSON with sorted keys and the default separators, so write the substring the way JSON prints it:

```python
# {"customer_id": "C-7", "force": True}  →  '{"customer_id": "C-7", "force": true}'
ApprovalRule(tool="delete_*", argument_contains='"force": true')   # matches
ApprovalRule(tool="delete_*", argument_contains="'force': True")   # does not
```

Matching is case-sensitive, and JSON escapes quotes and backslashes (`C:\temp` is `C:\\temp` in the JSON). For anything beyond a plain substring — case-insensitive email domains, numeric limits — use a `predicate`.

### Rules see the real arguments

`ApprovalPolicy(redact_sensitive=True)` (the default) replaces emails, phone numbers and credentials with labels such as `[EMAIL]` in the request a reviewer sees. The classifier doesn't match against that copy: the agent's gate also passes the unredacted arguments (`ApprovalRequest.raw_arguments`), and deny/ask/allow rules, predicates, the read-only check and the LLM classifier all see the real values. So this rule fires even with redaction on:

```python
ApprovalRule(tool="send_*", argument_contains="@competitor.example", reason="never email competitors")
```

Inside a predicate or the LLM classifier, `request.arguments` holds the real arguments. The fallback handler receives the reviewer's copy: `arguments` redacted as configured, and no `raw_arguments`. Rules also still work with `ApprovalPolicy(include_arguments=False)`, which hides the arguments from reviewers only.

`raw_arguments` is never part of `to_dict()` (webhook payloads), the webhook signature or `repr()`. Your LLM classifier gets the real arguments — the same values the agent's model produced — so redact them yourself if it sends them somewhere the agent's model doesn't.

---

## Read-only auto-allow

Enabled by default. A tool is auto-approved as read-only when:

1. no word of its name is a destructive verb — `fetch_and_purge_cache`, `show_and_delete` and `getAndResetCounter` are never read-only (see `DEFAULT_DESTRUCTIVE_VERBS`: `delete`, `purge`, `drop`, `update`, `send`, `reset`, …), **and**
2. its MCP annotations don't say otherwise — `readOnlyHint=False` or `destructiveHint=True` rule it out, **and**
3. either it is annotated `readOnlyHint=True`, or (without a `readOnlyHint`) its name starts with one of the read-only prefixes:

`get_`, `list_`, `read_`, `search_`, `find_`, `fetch_`, `describe_`, `show_`, `view_`, `lookup_`, `query_`, `head_`, `stat_`, `exists_`, `count_`

Annotations come from the MCP server — `@server.tool(read_only_hint=True)` on a Promptise server, or any server that sets `ToolAnnotations` — and reach the classifier as `ApprovalRequest.tool_annotations`. Promptise's MCP client keeps them on each tool's `metadata` (as `langchain-mcp-adapters` does), and the gate copies them from there, so a LangChain tool you build yourself can declare `metadata={"readOnlyHint": True}`. Annotations are hints from the server, not guarantees, so a destructive verb in the name always wins over `readOnlyHint=True`.

Words are split on `_`, `-` and camelCase and compared whole: `get_settings` is fine (`settings` is not `set`), `get_and_set_flag` is not.

Override the prefixes or the verbs:

```python
classifier = AutoApprovalClassifier(
    read_only_prefixes=("get_", "list_", "count_"),
    destructive_verbs=(*DEFAULT_DESTRUCTIVE_VERBS, "archive"),  # replaces the list
    fallback=my_handler,
)
```

Ignore annotations (`use_tool_annotations=False`; only the name decides), or disable the layer entirely:

```python
classifier = AutoApprovalClassifier(
    read_only_auto_allow=False,
    fallback=my_handler,
)
```

`classifier.is_read_only(request)` tells you what the layer would decide for a request.

---

## LLM classifier (optional)

For fuzzy decisions that rules can't capture:

```python
async def safety_check(request):
    # Call your LLM / safety model
    response = await llm.classify(
        f"Is this tool call safe? {request.tool_name}({request.arguments})"
    )
    if response.safe:
        return "allow", "LLM classified as safe"
    if response.dangerous:
        return "deny", "LLM classified as dangerous"
    return "escalate", "LLM unsure — send to human"

classifier = AutoApprovalClassifier(
    llm_classifier=safety_check,
    fallback=my_handler,
)
```

Returning `"escalate"` defers to the fallback handler (layer 6). It runs only when no rule decided and the tool isn't read-only.

---

## Audit: which layer decided

Every decision the classifier returns carries its trace in `decision.trace`, and says what decided it in `decision.decided_by`:

| `decision.decided_by` | `decision.trace.layer` |
|---|---|
| `"classifier"` | `"deny_rule"`, `"allow_rule"`, `"read_only"`, `"llm_allow"`, `"llm_deny"` |
| `"reviewer"` (the fallback's decision) | `"ask_rule"`, `"llm_escalate_then_fallback"`, `"fallback"`, `"error"` (the classifier raised and fell back) |

`trace.rule_reason` holds the matched rule's `reason` or the LLM's reason, and `trace.matched_rule` the rule itself.

To record every decision, pass `on_decision` to the policy. It is called for the classifier's decisions and also for those the gate makes without asking the handler at all — `max_pending`, the [repeated-denial limit](approval.md#repeated-denials), a timeout or a handler error (`decided_by="gate"`, `trace=None`):

```python
import json, time

def write_audit_line(request, decision):
    entry = {
        "time": time.strftime("%H:%M:%S"),
        "request_id": request.request_id,
        "user": request.caller_user_id,
        "tool": request.tool_name,
        "arguments": request.arguments,          # the redacted copy
        "approved": decision.approved,
        "decided_by": decision.decided_by,       # reviewer | classifier | gate
        "layer": decision.trace.layer if decision.trace else None,
        "rule": decision.trace.rule_reason if decision.trace else None,
        "reviewer": decision.reviewer_id,
        "reason": decision.reason,
    }
    with open("approvals.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

policy = ApprovalPolicy(tools=["*"], handler=classifier, on_decision=write_audit_line)
```

`on_decision` may be async. If it raises, the error is logged and the decision stands. A handler that wraps the classifier sees only the decisions that reach the handler, so it misses the gate's own denials; use `on_decision` for the audit log.

!!! warning "`classifier.last_trace` is shared"
    `last_trace` is the trace of the most recently *finished* decision. When several requests are in flight — several gated calls in one turn, or one call waiting on a human — another request can overwrite it before you read it. Read `decision.trace` instead.

### Stats

Every decision increments a counter in `classifier.stats`:

```python
print(classifier.stats.deny_rule_hits)     # 3
print(classifier.stats.ask_rule_hits)      # 2  (also counted in fallback_*)
print(classifier.stats.allow_rule_hits)    # 42
print(classifier.stats.read_only_allows)   # 189
print(classifier.stats.llm_allows)         # 7
print(classifier.stats.fallback_denies)    # 1
```

### Classifier denials and the retry limit

`ApprovalPolicy(max_retries_after_deny=3)` stops the agent from asking a *person* over and over. Only the reviewer's denials (`decided_by="reviewer"`, including a fallback reached through an ask rule or an LLM escalation) and timeouts count towards it. A deny rule or an LLM `"deny"` costs no one any time and doesn't count, and an automatic approval doesn't reset what the reviewer denied. So after a deny rule refuses three over-the-limit refunds, a small refund the allow rule covers still goes through.

Once the limit is reached for a tool (per [`deny_scope`](approval.md#repeated-denials)), later calls are denied by the gate without consulting the handler, including the classifier's rules, until the denials age out of `deny_window`.

---

## Drop-in replacement

`AutoApprovalClassifier` implements the `ApprovalHandler` protocol. Swap it into any existing `ApprovalPolicy` without changing anything else:

```python
# Before
policy = ApprovalPolicy(tools=["*"], handler=webhook_handler)

# After
policy = ApprovalPolicy(tools=["*"], handler=AutoApprovalClassifier(
    deny_rules=[...],
    allow_rules=[...],
    fallback=webhook_handler,
))
```

On the agent side, the classifier only applies to the agent's own gate. [Server-side approval gates](approval.md#server-side-approval-gates) reached through MCP elicitation go straight to its `fallback` (and `on_decision` records those decisions too, with `request.metadata["source"] == "mcp_elicitation"`).

You can also use the classifier on an MCP server, as the handler of `ApprovalGateMiddleware`. There its rules see the validated arguments (also with `include_arguments=False`, which hides them from the fallback only) and the read-only layer sees the tool's own annotations (`@server.tool(read_only_hint=..., destructive_hint=...)`).

---

## Related

- [Approval (HITL)](approval.md) — the underlying approval system
- [Runtime Hooks](../runtime/hooks.md) — react to PERMISSION_REQUEST / PERMISSION_DENIED events
- [Guardrails](guardrails.md) — input/output security scanning
