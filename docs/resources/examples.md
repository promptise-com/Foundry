# Examples Gallery

Runnable examples for Promptise Foundry. Some examples make live model or API
calls. Others demonstrate local framework behavior without external services.

Every literal Python path on this page is checked by the test suite.

---

## Running Examples

Install Promptise from the repository:

```bash
pip install -e ".[dev]"
```

Start with a local example that needs no API key:

```bash
python examples/identity/local/app.py
```

Then run a model-backed example after setting its documented provider key:

```bash
export OPENAI_API_KEY="..."
python examples/prompts/conversation_flow.py
```

Read each file's module docstring for its exact requirements.

---

## MCP Server & Client

Build production MCP servers, connect agents, enforce tenant boundaries, and
use remote MCP services.

| File | Description | Difficulty |
| --- | --- | --- |
| `examples/mcp/enterprise_server.py` | A 30-tool server with JWT auth, routers, roles, rate limits, and audit logging | Advanced |
| `examples/mcp/enterprise_cli.py` | An interactive agent that connects to the enterprise server and switches roles | Intermediate |
| `examples/mcp/tenancy_and_approval.py` | An in-process demo of tenant isolation and four-eyes approval gates | Intermediate |
| `examples/mcp/xquik_agent.py` | An X research agent using Xquik's credential-scoped remote MCP catalog | Beginner |

Run the Xquik remote MCP example:

```bash
export OPENAI_API_KEY="..."
export XQUIK_API_KEY="..."
python examples/mcp/xquik_agent.py "Find 5 recent posts about agentic engineering."
```

Use a least-privilege Xquik credential with read access for this research
example. The remote server enforces the credential's catalog and permissions.

Run the local enterprise pair:

```bash
# Terminal 1
python examples/mcp/enterprise_server.py

# Terminal 2
python examples/mcp/enterprise_cli.py
```

Run the self-contained tenancy and approval demo:

```bash
python examples/mcp/tenancy_and_approval.py
```

These examples cover:

- `MCPServer`, `MCPRouter`, and typed `@server.tool()` handlers
- JWT authentication, role guards, rate limits, and audit logging
- Server-side tenant isolation and human approval
- `HTTPServerSpec` with environment-backed bearer-token authentication
- Remote Streamable HTTP tool discovery and agent invocation

Xquik is an independent third-party service. Not affiliated with X Corp.
"Twitter" and "X" are trademarks of X Corp.

---

## Agent Identity

Give an agent a stable, traceable identity. The first two examples run locally
without a cloud account or model key.

| Example | What it demonstrates | Level |
| --- | --- | --- |
| `examples/identity/local/app.py` | A local identity with attribution and claims | Beginner |
| `examples/identity/verifiable_mcp/app.py` | End-to-end signed identity verification through MCP | Intermediate |
| `examples/identity/github_actions/script.py` | A verifiable identity from a GitHub Actions OIDC token | Intermediate |
| `examples/identity/aws_lambda/handler.py` | AWS IAM-backed identity in Lambda | Advanced |
| `examples/identity/gke_pod/app.py` | Google Cloud-backed identity in GKE | Advanced |
| `examples/identity/aks_workload/app.py` | Microsoft Entra-backed identity in AKS | Advanced |
| `examples/identity/spire/app.py` | SPIFFE/SPIRE-backed workload identity | Advanced |

```bash
python examples/identity/local/app.py
python examples/identity/verifiable_mcp/app.py
```

See [`examples/identity/README.md`](https://github.com/promptise-com/foundry/blob/main/examples/identity/README.md)
for the provider decision guide.

---

## Prompt Engineering

Compose prompts from blocks, evolve them across turns, and inspect the result.

| File | Description | Difficulty |
| --- | --- | --- |
| `examples/prompts/conversation_flow.py` | Conversation phases, prompt blocks, strategies, inspection, and version rollback | Intermediate |

```bash
python examples/prompts/conversation_flow.py
```

---

## Agent Runtime

Run agents as event-driven processes and coordinate multi-agent workflows.

| File | Description | Difficulty |
| --- | --- | --- |
| `examples/runtime/pipeline_monitor.py` | Event triggers, escalation, journaling, and budgets | Intermediate |
| `examples/runtime/multi_agent_pipeline.py` | A 3-agent research, analysis, and writing pipeline | Advanced |

```bash
python examples/runtime/pipeline_monitor.py
python examples/runtime/multi_agent_pipeline.py
```

---

## More Framework Examples

| File | Description |
| --- | --- |
| `examples/adaptive/learning_agent.py` | Adaptive failure classification and strategy learning |
| `examples/approval/auto_classifier.py` | Layered automatic approval decisions |
| `examples/hooks/lifecycle_hooks.py` | Prioritized runtime lifecycle hooks |
| `examples/memory/multiuser_chat.py` | Isolated memory, conversations, and cache |
| `examples/production/demo.py` | Full-stack production agent walkthrough |
| `examples/rag/basic_rag.py` | A custom RAG pipeline exposed as an agent tool |
| `examples/reasoning/code_action_agent.py` | Sandboxed code-action reasoning |
| `examples/reasoning/research_agent.py` | A custom multi-node research graph |
| `examples/reasoning/verify_and_managed.py` | Verify and managed reasoning patterns |
| `examples/sandbox/code_executor.py` | Iterative code execution in a Docker sandbox |
| `examples/security/financial_agent.py` | Guardrails and approval in a financial workflow |
