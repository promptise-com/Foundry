# Streaming with Tool Visibility

Stream agent execution with real-time tool activity — see which tools are being called, their results, and the LLM generating text token by token.

```python
async for event in agent.astream_with_tools(
    {"messages": [{"role": "user", "content": "Check my order status"}]},
    caller=CallerContext(user_id="user-42"),
):
    match event.type:
        case "tool_start":
            print(f"🔧 {event.tool_display_name}...")
        case "tool_end":
            print(f"   → {event.tool_summary}")
        case "token":
            print(event.text, end="", flush=True)
        case "done":
            print(f"\n✅ Done in {event.duration_ms:.0f}ms")
```

Output:
```
🔧 Searching orders...
   → Found: Order #4521, shipped March 20
🔧 Getting tracking info...
   → Status: In transit, ETA March 25
Your order #4521 shipped on March 20 and is currently in transit.
Expected delivery: March 25.
✅ Done in 2340ms
```

---

## Event Types

| Type | Class | When | Key Fields |
|------|-------|------|-----------|
| `tool_start` | `ToolStartEvent` | Tool begins executing | `tool_name`, `tool_display_name`, `arguments`, `tool_index` |
| `tool_end` | `ToolEndEvent` | Tool finishes | `tool_name`, `tool_summary`, `duration_ms`, `success`, `tool_index` |
| `token` | `TokenEvent` | LLM generates a token | `text`, `cumulative_text` |
| `done` | `DoneEvent` | Agent finished | `full_response`, `tool_calls`, `duration_ms`, `cache_hit` |
| `error` | `ErrorEvent` | Something went wrong | `message`, `recoverable` |

`done` and `error` are always the last event of a run.

---

## What a Streamed Run Does

A streamed run executes exactly the steps `ainvoke()` would — the stream is a
view of that run, not a second one:

- **One model call per step.** Each step's model call is streamed as it is
  generated. A question answered without tools is one model call; with a tool
  it is two (the call that asks for the tool, and the answer).
- **The answer after a tool streams too.** Its tokens arrive after the
  `tool_end` events, and `full_response` is that answer.
- **`full_response` is the final answer** — the text of the run's last model
  call, the same text `ainvoke()` returns as the last message. Text a model
  writes *before* calling a tool ("Let me look that up.") streams as tokens and
  is in `cumulative_text`, but not in `full_response`.
- **Tool calls run in parallel**, as in `ainvoke()`. When the model asks for
  several tools at once, every `tool_start` comes first, then each `tool_end`
  as that call finishes — possibly out of order.
- **`tool_index` and `duration_ms` belong to the call.** A `tool_end` has the
  same `tool_index` as its `tool_start`, and `duration_ms` is how long that
  call ran.
- **Failed calls are `success: false`.** A tool that reports an error — an MCP
  server's `ToolError`, which the server returns with `isError` set — ends with
  `success: false` and the error's message as `tool_summary`. A tool that
  raises ends with `success: false` and `tool_summary` `"Tool call failed"`
  (the exception text is not streamed). The model sees the error either way
  and answers from it.

`examples/mcp/stream_tool_calls.py` shows both against a real model:

```
>>> Where are my orders A-1001 and A-1002?
  1.5s  tool_start #0 Getting order status {'order_id': 'A-1001'}
  1.5s  tool_start #1 Getting order status {'order_id': 'A-1002'}
  2.5s  tool_end   #0 ok in 1006 ms: order_id: A-1001, status: shipped, carrier: DHL (+1 more)
  2.5s  tool_end   #1 ok in 1006 ms: order_id: A-1002, status: processing, carrier: None (+1 more)
  6.2s  answer streaming: Here's the current status: ...
  6.8s  done (308 characters in full_response)

>>> Where is my order Z-9999?
  1.3s  tool_start #0 Getting order status {'order_id': 'Z-9999'}
  2.3s  tool_end   #0 FAILED in 1006 ms: No order found with ID Z-9999.
  4.9s  answer streaming: I couldn't find an order with ID Z-9999 ...
  5.9s  done (816 characters in full_response)
```

---

## Server-Sent Events (FastAPI + EventSource)

The browser's `EventSource` only sends **GET** requests, so the endpoint is a
GET that takes the question as a query parameter:

```python
from fastapi import Depends, FastAPI
from fastapi.responses import StreamingResponse

from promptise import CallerContext

app = FastAPI()


@app.get("/chat")
async def chat(q: str, user=Depends(current_user)):
    # current_user is your auth dependency. EventSource sends the page's
    # cookies but cannot set headers, so authenticate with the session cookie.
    async def event_stream():
        async for event in agent.astream_with_tools(
            {"messages": [{"role": "user", "content": q}]},
            caller=CallerContext(user_id=user.id),
        ):
            yield f"data: {event.to_json()}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

### Frontend (JavaScript)

```javascript
const source = new EventSource(`/chat?q=${encodeURIComponent(question)}`);

source.onmessage = (e) => {
  const event = JSON.parse(e.data);
  switch (event.type) {
    case "tool_start":
      showToolIndicator(event.tool_index, event.tool_display_name);
      break;
    case "tool_end":
      updateToolResult(event.tool_index, event.tool_summary, event.success);
      break;
    case "token":
      appendText(event.text);
      break;
    case "done":
      source.close();  // the run is over: stop EventSource from reconnecting
      finishResponse(event.full_response);
      break;
    case "error":
      source.close();
      showError(event.message);
      break;
  }
};

source.onerror = () => {
  // A dropped connection: don't let EventSource silently re-ask the question.
  source.close();
  showError("Connection lost.");
};
```

!!! warning "Always close the EventSource on `done` and `error`"
    When the server ends the response, `EventSource` treats it as a dropped
    connection and **reconnects about 3 seconds later** — the same GET, so the
    agent runs the question again (more model calls, more tool calls) and the
    page receives a second answer. Call `source.close()` when the `done` or
    `error` event arrives, and in `onerror`.

Use the event's `tool_index` to match a `tool_end` to its `tool_start`: parallel
calls can finish in any order.

### POST with `fetch()`

To send the question in a request body (long prompts, or a bearer token in a
header), use a POST endpoint — the same handler with `@app.post` and a request
model — and read the stream with `fetch()` instead of `EventSource`. A fetch
stream does not reconnect on its own:

```javascript
const response = await fetch("/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
  body: JSON.stringify({ message: question }),
});
const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
let buffer = "";
for (;;) {
  const { value, done } = await reader.read();
  if (done) break;
  buffer += value;
  const frames = buffer.split("\n\n");
  buffer = frames.pop();  // keep an incomplete frame for the next chunk
  for (const frame of frames) {
    if (frame.startsWith("data: ")) handleEvent(JSON.parse(frame.slice(6)));
  }
}
```

---

## Tool Display Names

Tool names are automatically converted to human-readable strings:

| Raw Name | Display Name |
|----------|-------------|
| `search_customers` | Searching customers |
| `get_order_status` | Getting order status |
| `hr_list_employees` | Listing employees |
| `create_ticket` | Creating ticket |
| `deploy_service` | Deploying service |

Override with custom names:

```python
async for event in agent.astream_with_tools(
    input,
    tool_display_names={
        "pg_query": "Querying database",
        "s3_upload": "Uploading to cloud storage",
    },
)
```

---

## Tool Result Summaries

Tool results are automatically summarized for display:

- JSON dicts: `"name: Alice, status: active (+3 more)"`
- JSON lists: `"Found 5 result(s)"`
- Plain text: truncated to 120 characters
- Failed calls: the error message (`"No order found with ID Z-9999."`), or
  `"Tool call failed"` when the tool raised

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `dict` | required | Agent input (same as `ainvoke`) |
| `caller` | `CallerContext` | `None` | Per-request identity |
| `include_arguments` | `bool` | `True` | Include tool arguments in events |
| `tool_display_names` | `dict[str, str]` | `None` | Custom display name overrides |

---

## Security

- **Argument redaction**: Tool arguments are run through guardrails before streaming. PII and credentials are replaced with labels.
- **Error sanitization**: `ErrorEvent.message` is always generic — no internal stack traces, file paths, or database URLs.
- **Stream cancellation**: Closing the SSE connection cancels the agent's async task. No wasted LLM calls.
- **No cumulative arguments**: `TokenEvent.cumulative_text` only contains LLM text, never tool arguments.

---

## Integration

`astream_with_tools()` runs the full agent pipeline — same guarantees as `ainvoke()`:

- ✅ Input guardrails (block injection → ErrorEvent)
- ✅ Memory injection (relevant context before LLM call)
- ✅ Output guardrails (PII redaction on `full_response`; tokens are streamed as the model generates them, before the check)
- ✅ Observability (callback handler records timeline)
- ✅ Event notifications (invocation.start/complete emitted)
- ✅ CallerContext propagation (multi-user safe)

---

## Helper Functions

### tool_display_name()

Convert raw MCP tool names to human-readable labels:

```python
from promptise.streaming import tool_display_name

tool_display_name("search_customers")      # "Searching customers"
tool_display_name("get_order_status")      # "Getting order status"
tool_display_name("deploy_production")     # "Deploying production"
```

| Parameter | Type | Description |
|---|---|---|
| `tool_name` | `str` | Raw tool name (e.g. `"search_customers"`) |
| `overrides` | `dict[str, str] \| None` | Custom name mappings (tool_name → display text) |
| **Returns** | `str` | Human-readable display name |

### tool_summary()

Summarize tool output for the stream:

```python
from promptise.streaming import tool_summary

tool_summary('{"results": [...100 items...]}')  # '{"results": [...100 items...]}'[:120]
tool_summary(None)                               # "Done"
tool_summary("")                                 # "Done"
```

| Parameter | Type | Description |
|---|---|---|
| `result` | `str \| None` | Raw tool output |
| `max_length` | `int` | Truncation limit (default: 120) |
| **Returns** | `str` | Summarized output |

### tool_error_summary()

Summarize a failed tool call — the message of an MCP `ToolError`:

```python
from promptise.streaming import tool_error_summary

tool_error_summary('{"error": {"code": "TOOL_ERROR", "message": "No order A-9."}}')  # "No order A-9."
tool_error_summary("Error: Unknown tool 'x'")                                      # "Error: Unknown tool 'x'"
```

### Event Serialization

All stream events support serialization:

```python
event.to_dict()  # Returns dict
event.to_json()  # Returns JSON string (for SSE)
```

---

## Streaming the Conversation: `astream()`

`agent.astream()` streams at a coarser grain: one chunk per step of the agent's
graph, each the whole conversation so far — `{"messages": [...]}`. The last
chunk is what `ainvoke()` returns. Use it to process intermediate messages
(tool calls and their results) in code; use `astream_with_tools()` for a chat
UI.

```python
async for chunk in agent.astream({"messages": [{"role": "user", "content": "Where is A-1001?"}]}):
    last = chunk["messages"][-1]
    print(type(last).__name__, last.content[:60])
# ToolMessage {"order_id": "A-1001", "status": "shipped", ...}   step 1: tool call + result
# AIMessage Your order A-1001 has shipped with DHL ...           step 2: the answer
```

---

## What's Next?

- [Events & Notifications](events.md) -- webhook alerts for invocation, tool, and guardrail events
- [Guardrails](guardrails.md) -- security scanning that redacts tool arguments in the stream
- [Observability](observability.md) -- detailed execution traces alongside streaming
