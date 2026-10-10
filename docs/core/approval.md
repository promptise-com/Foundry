# Human-in-the-Loop Approval

Require human approval before executing sensitive tool calls. The agent pauses, sends an approval request to a human reviewer, and waits for a decision before proceeding or adapting.

!!! info "Server-side approval gates"
    This page covers **agent-side** approval — governance the agent applies to
    its own tool calls. To enforce approval **where the tool lives**, for any
    MCP client (declared with `@server.tool(requires_approval=True)` and
    enforced by `ApprovalGateMiddleware`), see
    [Approval Gates (Server-Side HITL)](../mcp/server/approval-gates.md).
    Both share the same `ApprovalRequest` / `ApprovalDecision` /
    `ApprovalHandler` protocol, so handlers written for one work with the other.

```python
from promptise import build_agent, ApprovalPolicy, CallbackApprovalHandler, ApprovalDecision

async def my_approver(request):
    print(f"Agent wants to call: {request.tool_name}({request.arguments})")
    answer = input("Approve? [y/n]: ").strip().lower()
    return ApprovalDecision(approved=(answer == "y"))

agent = await build_agent(
    servers=servers,
    model="openai:gpt-5-mini",
    approval=ApprovalPolicy(
        tools=["send_email", "delete_*", "payment_*"],
        handler=CallbackApprovalHandler(my_approver),
    ),
)
```

---

## How It Works

```
User sends message
    ↓
Agent reasons and decides to call send_email(to="alice@acme.com", subject="Invoice")
    ↓
┌─ ApprovalPolicy checks: does "send_email" match any pattern in tools=["send_*"]?
│
├─ NO MATCH → tool executes immediately (zero overhead)
│
└─ MATCH → Approval flow begins:
       ↓
   1. Build ApprovalRequest (unique ID, tool name, redacted arguments, caller identity)
       ↓
   2. Send to ApprovalHandler (webhook POST / async callback / UI queue)
       ↓
   3. Agent execution PAUSES (event loop is free — other agents/requests continue)
       ↓
   4. Human reviews:
      "Agent 'billing-bot' wants to send_email(to='[EMAIL]', subject='Invoice')"
      [Approve] [Deny] [Approve with edits]
       ↓
   5. Decision received (or timeout triggers on_timeout action)
       ↓
   ┌─ APPROVED → tool executes with original (or modified) arguments → agent continues
   ├─ DENIED → tool returns "DENIED: reason" as result → agent adapts
   └─ TIMEOUT → on_timeout="deny" returns denial / on_timeout="allow" proceeds
```

**Key principle:** The LLM doesn't know approval exists. It calls tools normally. The approval wrapper intercepts matching tool calls transparently. On approval, the tool runs. On denial, the tool returns a "DENIED" message as its result — the LLM sees this and adapts (tries alternatives, asks for help, or reports it can't proceed).

---

## Setup Guide

### Step 1: Choose which tools need approval

Use glob patterns to match tool names:

```python
tools=["send_email"]              # Exact match — only send_email
tools=["send_*"]                  # Wildcard — send_email, send_notification, send_sms
tools=["delete_*", "payment_*"]   # Multiple patterns
tools=["*"]                       # Every tool requires approval (for maximum control)
```

Patterns use Python's `fnmatch` — `*` matches any characters, `?` matches one character.

### Step 2: Choose a handler

Three built-in handlers + custom protocol:

| Handler | Best for | How it works |
|---|---|---|
| `CallbackApprovalHandler` | Quick integration, scripts, Slack bots | Call your async function, return decision |
| `WebhookApprovalHandler` | External approval systems, REST APIs | POST request + poll for decision |
| `QueueApprovalHandler` | In-process UIs (Gradio, Streamlit) | asyncio.Queue — UI reads, user clicks, decision flows back |
| Custom (protocol) | Anything else | Implement one async method |

### Step 3: Configure the policy

```python
from promptise import ApprovalPolicy

policy = ApprovalPolicy(
    tools=["send_email", "delete_*"],
    handler=my_handler,
    timeout=300,                  # 5 minutes to decide
    on_timeout="deny",            # Deny if no response
    include_arguments=True,       # Show tool args to reviewer
    redact_sensitive=True,        # Redact PII/credentials in args
    max_pending=10,               # Max 10 concurrent pending approvals
    max_retries_after_deny=3,     # Permanent deny after 3 retries
)
```

### Step 4: Pass to build_agent

```python
agent = await build_agent(
    servers=servers,
    model="openai:gpt-5-mini",
    approval=policy,
)
```

That's it. Every tool call matching your patterns now requires human approval.

---

## ApprovalPolicy — All Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tools` | `list[str]` | **required** | Glob patterns for tool names. `"send_*"` matches `send_email`, `send_notification`. |
| `handler` | `ApprovalHandler \| Callable` | **required** | How to send and receive approval decisions. |
| `timeout` | `float` | `300` | Seconds to wait for a decision before `on_timeout` triggers. Min: > 0. Max: 86,400 (24 hours). |
| `on_timeout` | `"deny" \| "allow"` | `"deny"` | Action when timeout expires. `"deny"` is the safe default. Use `"allow"` only for non-critical, low-risk tools. |
| `include_arguments` | `bool` | `True` | Include tool arguments in the approval request. Set to `False` to hide arguments from reviewers (e.g., when arguments contain data the reviewer shouldn't see). |
| `redact_sensitive` | `bool` | `True` | Run arguments through PII/credential detection before sending to the reviewer. Requires the guardrails module. Falls back to raw arguments if guardrails are not installed. |
| `max_pending` | `int` | `10` | Maximum concurrent pending approvals per agent. When reached, additional tool calls are auto-denied with "Too many pending approval requests." |
| `max_retries_after_deny` | `int` | `3` | After this many denials of the same tool name, return a permanent denial message. Prevents the LLM from retrying the same denied tool in an infinite loop. |

---

## ApprovalRequest — What the Handler Receives

Every handler receives an `ApprovalRequest` with these fields:

| Field | Type | Description |
|---|---|---|
| `request_id` | `str` | Cryptographically random unique ID (`secrets.token_hex(16)` — 32 hex chars). Use this to match requests with decisions. |
| `tool_name` | `str` | The name of the tool the agent wants to call (e.g., `"send_email"`). |
| `arguments` | `dict` | The tool arguments. Redacted if `redact_sensitive=True` — e.g., `{"to": "[EMAIL]", "body": "..."}`. Empty dict if `include_arguments=False`. |
| `agent_id` | `str \| None` | Agent or process identifier (set automatically in runtime). |
| `caller_user_id` | `str \| None` | The `user_id` from `CallerContext` — identifies which user triggered the action. |
| `context_summary` | `str` | Brief context for the reviewer (can be customized). |
| `timestamp` | `float` | When the request was created (`time.time()`). |
| `timeout` | `float` | How long the handler has to respond before the default action triggers. |
| `metadata` | `dict` | Developer-provided custom data (passed through unchanged). |

**HMAC signature:**

```python
# Compute a signature for tamper-proof webhook delivery
signature = request.compute_hmac("your-secret-key")
# Returns a hex string — include in X-Promptise-Signature header
```

**Serialization:**

```python
# Convert to JSON-safe dict (for webhook payloads)
payload = request.to_dict()
```

---

## ApprovalDecision — What the Handler Returns

| Field | Type | Default | Description |
|---|---|---|---|
| `approved` | `bool` | **required** | Whether the tool call is approved. |
| `modified_arguments` | `dict \| None` | `None` | If the reviewer edited the arguments. When set, the tool executes with these arguments instead of the originals. |
| `reviewer_id` | `str \| None` | `None` | Who made the decision (for audit trail). |
| `reason` | `str \| None` | `None` | Optional explanation shown to the agent on denial. |
| `timestamp` | `float` | `time.time()` | When the decision was made. |

---

## Handlers — Detailed Setup

### CallbackApprovalHandler

Wraps any Python async function. The simplest way to integrate.

```python
from promptise import CallbackApprovalHandler, ApprovalDecision

# Full control — return an ApprovalDecision
async def slack_approver(request):
    message = (
        f"*Approval needed*\n"
        f"Agent wants to call `{request.tool_name}`\n"
        f"Arguments: `{request.arguments}`\n"
        f"User: {request.caller_user_id or 'anonymous'}"
    )
    channel_id = await post_to_slack(message)
    reaction = await wait_for_reaction(channel_id, timeout=request.timeout)
    return ApprovalDecision(
        approved=(reaction.emoji == "thumbsup"),
        reviewer_id=reaction.user_id,
        reason="Approved via Slack" if reaction.emoji == "thumbsup" else "Denied via Slack",
    )

handler = CallbackApprovalHandler(slack_approver)
```

**Shortcut — return a plain bool:**

```python
# Auto-approve everything (for testing)
handler = CallbackApprovalHandler(lambda req: True)

# Auto-deny everything (for dry runs)
handler = CallbackApprovalHandler(lambda req: False)

# Approve based on tool name
handler = CallbackApprovalHandler(
    lambda req: req.tool_name != "delete_production_database"
)
```

### WebhookApprovalHandler

For external approval systems. POSTs the request as JSON, then polls for the decision.

```python
from promptise import WebhookApprovalHandler

handler = WebhookApprovalHandler(
    url="https://your-approval-api.com/requests",
    secret="your-hmac-secret",
    poll_url="https://your-approval-api.com/decisions",  # Optional — defaults to {url}/{request_id}
    poll_interval=2.0,
    headers={"Authorization": "Bearer your-api-token"},
)
```

**All parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `url` | `str` | **required** | Webhook URL to POST approval requests to. Validated against private IP blocklist (SSRF protection). |
| `secret` | `str \| None` | Auto-generated | HMAC secret for signing requests. If not provided, a random secret is generated per process (not useful for cross-process verification — set your own for production). |
| `poll_url` | `str \| None` | `{url}/{request_id}` | URL to poll for the decision. Handler GETs this URL every `poll_interval` seconds. |
| `poll_interval` | `float` | `2.0` | Seconds between poll attempts. Min: 0.5. |
| `headers` | `dict[str, str] \| None` | `None` | Custom HTTP headers (e.g., auth tokens, API keys). |
| `http_client` | `httpx.AsyncClient \| None` | `None` | Pre-configured HTTP client. Use for proxy, mTLS, custom CA certificates, or any httpx configuration. |

**Webhook flow:**

```
1. Handler POSTs to url:
   POST https://your-api.com/requests
   Headers:
     Content-Type: application/json
     X-Promptise-Signature: <hmac-sha256>
     X-Promptise-Request-Id: <request_id>
   Body: {"request_id": "...", "tool_name": "...", "arguments": {...}, ...}

2. Your API responds: 202 Accepted

3. Handler polls poll_url every poll_interval seconds:
   GET https://your-api.com/requests/<request_id>
   Headers:
     X-Promptise-Request-Id: <request_id>

4. Your API responds with 200 + decision when ready:
   {"approved": true, "reviewer_id": "alice", "reason": "Looks good"}

   Or 202 if still pending (handler continues polling)

5. On timeout: handler raises TimeoutError → on_timeout action triggers
```

**HMAC verification (server-side):**

```python
import hmac, hashlib, json

def verify_signature(request_id, tool_name, received_signature, secret):
    payload = json.dumps({"request_id": request_id, "tool_name": tool_name}, sort_keys=True)
    expected = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, received_signature)
```

**Enterprise proxy / mTLS configuration:**

```python
import httpx

# Corporate proxy
client = httpx.AsyncClient(
    proxies="http://corporate-proxy:8080",
    verify="/path/to/corporate-ca-bundle.crt",
    timeout=60,
)

handler = WebhookApprovalHandler(
    url="https://internal-approval-api.corp.com/requests",
    http_client=client,
)
```

### QueueApprovalHandler

For in-process UIs where the human reviewer is in the same Python process (Gradio, Streamlit, terminal prompts).

```python
from promptise import QueueApprovalHandler, ApprovalDecision

handler = QueueApprovalHandler(maxsize=100)

# Your UI reads pending requests from the queue
async def approval_ui_loop():
    while True:
        request = await handler.request_queue.get()

        # Display to user
        print(f"\n--- Approval Required ---")
        print(f"Tool: {request.tool_name}")
        print(f"Arguments: {request.arguments}")
        print(f"User: {request.caller_user_id}")

        # Collect decision
        answer = input("Approve? [y/n/edit]: ").strip().lower()

        if answer == "y":
            handler.submit_decision(request.request_id, ApprovalDecision(approved=True))
        elif answer == "edit":
            new_to = input("New recipient: ")
            handler.submit_decision(request.request_id, ApprovalDecision(
                approved=True,
                modified_arguments={"to": new_to},
            ))
        else:
            handler.submit_decision(request.request_id, ApprovalDecision(
                approved=False, reason="Denied by operator"
            ))
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `maxsize` | `int` | `100` | Maximum queue size. Prevents unbounded memory growth if no reviewer is consuming requests. |

| Method | Description |
|---|---|
| `handler.request_queue` | `asyncio.Queue[ApprovalRequest]` — read pending requests from here |
| `handler.submit_decision(request_id, decision)` | Submit a decision for a pending request. Raises `KeyError` if the request_id is unknown or already resolved. |

### Custom Handler

Implement the `ApprovalHandler` protocol — one async method:

```python
from promptise import ApprovalHandler, ApprovalRequest, ApprovalDecision

class JiraApprovalHandler:
    """Create a Jira ticket for each approval, wait for resolution."""

    def __init__(self, jira_url: str, project: str, api_token: str):
        self.jira_url = jira_url
        self.project = project
        self.api_token = api_token

    async def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        # Create Jira ticket
        ticket = await self._create_ticket(
            summary=f"Agent approval: {request.tool_name}",
            description=f"Arguments: {request.arguments}\nUser: {request.caller_user_id}",
        )
        # Poll ticket status
        while True:
            status = await self._get_ticket_status(ticket.id)
            if status == "approved":
                return ApprovalDecision(approved=True, reviewer_id=ticket.assignee)
            if status == "rejected":
                return ApprovalDecision(approved=False, reason=ticket.comment)
            await asyncio.sleep(5)
```

The protocol is runtime-checkable:

```python
handler = JiraApprovalHandler(...)
assert isinstance(handler, ApprovalHandler)  # True
```

---

## What the Agent Sees

The approval system is transparent to the LLM. It doesn't know approval is happening — it just calls tools and receives results.

### Approved

The tool executes normally. The agent sees the tool's actual result. It has no idea approval was involved.

### Denied

The tool returns a denial message as its result:

```
DENIED: Not authorized for this action.
```

The agent sees this as the tool output and must adapt. Typical agent behaviors:
- Try an alternative approach that doesn't require the denied tool
- Ask the user for clarification or permission
- Report that it cannot complete the requested action

### Denied with reason

```
DENIED: Customer has opted out of email communications.
```

The reviewer's reason is included, helping the agent understand why and adapt more intelligently.

### Modified arguments

The reviewer approves but changes the arguments. The tool executes with the modified arguments. The agent sees the result of the modified call.

Example: Agent calls `send_email(to="all-staff@acme.com")`. Reviewer changes to `to="marketing-team@acme.com"`. Agent sees the result of sending to marketing-team.

### Permanent denial

After `max_retries_after_deny` denials of the same tool:

```
DENIED: This action was permanently denied after 3 attempts. Do not retry this tool.
```

This stops the agent from wasting reviewer time by retrying the same denied action.

---

## Security

### HMAC Request Signing

`WebhookApprovalHandler` signs every request with HMAC-SHA256:

- Header: `X-Promptise-Signature`
- Payload: JSON of `{"request_id": "...", "tool_name": "..."}` (sorted keys)
- Secret: Your `secret` parameter

Verify this in your approval API to reject spoofed requests.

### Single-Use Request IDs

Every `request_id` is generated with `secrets.token_hex(16)` — 128 bits of cryptographic randomness. Request IDs are tracked per-agent to prevent replay attacks.

### Argument Redaction

When `redact_sensitive=True` (default), tool arguments are processed through the guardrails system before being sent to the reviewer:

- PII patterns: email addresses, phone numbers, SSNs → `[EMAIL]`, `[PHONE]`, `[SSN]`
- Credential patterns: API keys, tokens → `[AWS_ACCESS_KEY]`, `[OPENAI_KEY]`

The reviewer sees enough to make a decision without seeing raw sensitive data.

### SSRF Protection

`WebhookApprovalHandler` validates the webhook URL at construction time:

- Blocks private IP ranges (10.x, 172.16-31.x, 192.168.x)
- Blocks loopback (127.x)
- Blocks link-local (169.254.x — cloud metadata endpoints)
- Blocks `localhost` and known internal hostnames

### Timeout Enforcement

Timeouts are enforced server-side with `asyncio.wait_for`. A misbehaving handler cannot keep the agent suspended indefinitely. Maximum configurable timeout: 24 hours.

---

## Multi-User

`CallerContext.user_id` is automatically extracted and included in every `ApprovalRequest.caller_user_id`. The reviewer knows which user's action they're approving.

```python
# User identity flows through to the approval request
result = await agent.ainvoke(
    {"messages": [{"role": "user", "content": "Send the invoice"}]},
    caller=CallerContext(user_id="user-42"),
)
# ApprovalRequest will include caller_user_id="user-42"
```

Different users' pending approvals are completely independent. User A waiting for approval doesn't block User B's agent from executing.

---

## YAML Configuration (.superagent)

```yaml
approval:
  tools:
    - send_email
    - "delete_*"
    - "payment_*"
  handler: webhook
  webhook_url: https://your-api.com/approvals
  webhook_secret: "${APPROVAL_WEBHOOK_SECRET}"   # signs X-Promptise-Signature
  timeout: 300
  on_timeout: deny
  max_pending: 10
  redact_sensitive: true
  max_retries_after_deny: 3
```

Supported handler types in YAML:

| `handler` | Needs | Who answers |
|---|---|---|
| `webhook` | `webhook_url`; set `webhook_secret` so your service can verify `X-Promptise-Signature` (without it a random per-process secret is used) | Your approval service |
| `queue` | — | Code in the same process reading `handler.request_queue`. `promptise agent` asks each request as a y/N question when you run it at a terminal; with piped input nobody can answer, so requests time out and `on_timeout` applies. |

The file is checked when it is loaded (and by `promptise validate`): `handler: webhook` without `webhook_url`, `webhook_secret` with another handler, and `handler: callback` are validation errors.

!!! note "Callback handler not available in YAML"
    `CallbackApprovalHandler` requires a Python function, so a `.superagent` file cannot configure it. Use `webhook` or `queue` in YAML, or build the agent in Python with `approval=ApprovalPolicy(handler=CallbackApprovalHandler(fn), ...)`.

---

## Integration with Agent Runtime

Approval works seamlessly with autonomous agents in the runtime:

```python
from promptise.runtime import ProcessConfig, TriggerConfig
from promptise import ApprovalPolicy, WebhookApprovalHandler

config = ProcessConfig(
    model="openai:gpt-5-mini",
    instructions="Process customer refunds. Amounts over $500 need manager approval.",
    triggers=[TriggerConfig(type="cron", cron_expression="*/5 * * * *")],
    approval=ApprovalPolicy(
        tools=["process_refund", "delete_account", "escalate_to_legal"],
        handler=WebhookApprovalHandler(url="https://ops.internal/approvals"),
        timeout=600,
    ),
)
```

When the autonomous agent hits a tool requiring approval at 3am, the webhook fires, the ops team gets notified, and the agent waits.

---

## Integration with Guardrails

Approval and guardrails work together:

1. **Input guardrails** run first — if the input is blocked, the agent never reaches tool calling
2. **Approval** intercepts tool calls that match patterns — after the agent decides to call a tool but before execution
3. **Argument redaction** uses guardrail detectors to sanitize arguments sent to the reviewer
4. **Output guardrails** run on the tool result after execution (if approved)

---

## Integration with Budget

- Pending approvals count as in-progress tool calls for budget tracking
- If the budget is exhausted while an approval is pending, the pending tool call is not affected (it was already counted when the request was created)
- The `max_irreversible_per_run` limit is checked before sending the approval request — if the limit is already reached, the tool is denied without asking the reviewer

---

## Integration with Semantic Cache

Cached responses bypass approval entirely — tools aren't called on cache hits, so there's nothing to approve. This is correct behavior: the original response was already approved (or didn't need approval) when it was first generated and cached.

---

## Edge Cases

| Scenario | Behavior |
|---|---|
| **Tool doesn't match any pattern** | Executes immediately, zero overhead. |
| **Approval arrives after timeout** | Discarded. Agent already received the timeout decision. Late approvals are logged. |
| **Agent crash during pending approval** | On restart, the pending approval is gone. The agent starts fresh. |
| **Multiple tools need approval in one invocation** | Each is handled independently, sequentially. The second approval starts after the first resolves. |
| **Agent retries denied tool** | Tracked per tool name. After `max_retries_after_deny`, returns permanent denial without asking reviewer again. |
| **max_pending reached** | Additional tool calls are auto-denied: "Too many pending approval requests." |
| **Handler throws an exception** | Treated as denial: "Approval handler error: {error type}." Agent continues. |
| **Reviewer modifies arguments to invalid values** | The tool executes with modified arguments. If the tool validates internally (Pydantic), it will fail with a tool error. |
| **No CallerContext provided** | Approval still works. `caller_user_id` is `None` in the request. |
| **Delegation to a peer agent** | `ask_agent_<name>` tools match patterns like any tool. `broadcast_to_agents` does not call a peer whose `ask_agent_<name>` needs approval (unless the broadcast tool needs approval itself). A peer's own policy applies inside the peer -- also for the cross-agents of a `.superagent` file, which `build_superagent()` and `promptise agent` build with their whole file. See [Cross-Agent Delegation](agents/cross-agent.md#approval-and-identity). |

---

## API Summary

| Symbol | Import | Description |
|---|---|---|
| `ApprovalPolicy` | `from promptise import ApprovalPolicy` | Main configuration — tools, handler, timeout, behavior |
| `ApprovalRequest` | `from promptise import ApprovalRequest` | Data sent to the handler (tool name, args, caller, etc.) |
| `ApprovalDecision` | `from promptise import ApprovalDecision` | Human's decision (approved, modified args, reason) |
| `ApprovalHandler` | `from promptise import ApprovalHandler` | Runtime-checkable protocol for custom handlers |
| `CallbackApprovalHandler` | `from promptise import CallbackApprovalHandler` | Wrap an async callable or plain function |
| `WebhookApprovalHandler` | `from promptise import WebhookApprovalHandler` | POST to URL + poll for decision |
| `QueueApprovalHandler` | `from promptise import QueueApprovalHandler` | asyncio.Queue for in-process UIs |

---

## Manual Integration

If you're not using `build_agent()`, you can wrap tools manually:

```python
from promptise import wrap_tools_with_approval, ApprovalPolicy, CallbackApprovalHandler

policy = ApprovalPolicy(
    tools=["send_*", "delete_*"],
    handler=CallbackApprovalHandler(my_handler),
)

# Wrap a list of LangChain BaseTool instances
wrapped_tools = wrap_tools_with_approval(tools, policy, event_notifier=notifier)
# Tools matching patterns are wrapped; others pass through unchanged
```

| Parameter | Type | Description |
|---|---|---|
| `tools` | `list[BaseTool]` | Tools to potentially wrap |
| `policy` | `ApprovalPolicy` | Which tools need approval and how |
| `event_notifier` | `EventNotifier \| None` | Optional notifier for approval events |
| **Returns** | `list[BaseTool]` | New list with matching tools wrapped |

---

## What's Next?

- [Guardrails](guardrails.md) -- input/output security scanning
- [Semantic Cache](cache.md) -- reduce costs with response caching
- [Conversations](conversations.md) -- multi-user session persistence
- [Building Agents](agents/building-agents.md) -- full parameter reference
