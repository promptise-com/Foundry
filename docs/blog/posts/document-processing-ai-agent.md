---
title: "Build a PII-Safe Document Processing AI Agent"
description: "Document pipelines leak sensitive data and run untrusted parsing code — the two failure modes this build closes. Every extracted output is scanned by 69 PII…"
keywords: "document processing ai agent, ai document extraction agent, pii redaction agent, intelligent document processing llm, secure document ai"
date: 2026-07-16
slug: document-processing-ai-agent
categories:
  - Use Cases
---

# Build a PII-Safe Document Processing AI Agent

A document processing ai agent is one of the highest-value things you can build with an LLM — invoices, contracts, lab results, and onboarding forms are exactly the unstructured text that models are good at, and exactly the data your compliance team loses sleep over. Two failure modes break these pipelines in production: sensitive data leaks out in an extracted response, and generated parsing code runs with full access to your host. This build closes both. By the end you'll have a working pipeline where every extracted output is scanned for PII before it leaves the agent, and any code the agent writes to parse a file runs inside a locked-down Docker sandbox with no network.

<!-- more -->

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## The two failure modes that break document pipelines

Most tutorials show you the happy path: read a PDF, stuff the text into a prompt, ask the model for JSON. That demo works. Then it meets reality:

- **Data exfiltration.** A contract contains a Social Security number, a customer's home address, and an API key someone pasted into a comments field. The model faithfully echoes all of it into the "summary" your app logs, caches, and ships downstream. Now that PII lives in a dozen places it should never have touched.
- **Untrusted code execution.** For messy inputs — a weird CSV, a nested table, an oddly encoded date — the most capable pattern is to let the agent *write* a small parser and run it. But "run code the model just generated" against your host filesystem and network is how a document pipeline becomes a remote code execution vector.

Promptise Foundry treats both as first-class concerns rather than afterthoughts. The security scanner is an output boundary; the sandbox is an execution boundary. You turn each on with one parameter on `build_agent()`.

## Expose documents to an ai document extraction agent via MCP

Before an ai document extraction agent can do anything, it needs a way to reach your files. In Promptise you don't hand-wire that — you expose it as tools on an MCP server, and the agent discovers and calls them automatically. Tool schemas are generated from your type hints, so there's no manual JSON-schema wiring.

```python
# doc_server.py
from pathlib import Path
from promptise.mcp.server import MCPServer

server = MCPServer("doc-tools")
INBOX = Path("./inbox")

@server.tool()
async def list_documents() -> list[str]:
    """List the filenames of documents waiting to be processed."""
    return [p.name for p in INBOX.glob("*.txt")]

@server.tool()
async def read_document(name: str) -> str:
    """Read the raw text of a single document by filename."""
    path = (INBOX / name).resolve()
    if INBOX.resolve() not in path.parents:   # block path traversal
        raise ValueError("Path outside inbox")
    return path.read_text(encoding="utf-8")

if __name__ == "__main__":
    server.run(transport="stdio")
```

That's the whole extraction surface: the agent can enumerate the inbox and read a document's text. Everything else — deciding what fields to pull, handling odd formats — is the model's job, governed by the two boundaries we add next.

## Add a PII redaction agent layer with the security scanner

This is the feature that makes the pipeline safe to ship. Promptise's `PromptiseSecurityScanner` is a composable set of local detection heads. For a pii redaction agent you want two of them working together: the `PIIDetector` (69 regex patterns plus Luhn validation, covering credit cards, SSNs, government IDs from 22+ countries, emails, and phone numbers) and the `NERDetector`, a GLiNER zero-shot NER model that catches the free-form entities regex misses — person names, physical addresses, and organizations.

```python
from promptise import (
    PromptiseSecurityScanner, PIIDetector, NERDetector, CredentialDetector,
)

scanner = PromptiseSecurityScanner(
    detectors=[
        PIIDetector(),        # 69 regex patterns + Luhn
        NERDetector(),        # GLiNER NER: names, addresses, orgs
        CredentialDetector(), # 96 patterns: API keys, tokens, DB URLs
    ],
)
scanner.warmup()   # load models once, up front
```

Every detector runs locally — no document text is sent to a third-party classification service. Input is scanned **before** it reaches the agent (so an injected "ignore your instructions" buried in a PDF is blocked), and output is scanned **after** the agent responds. PII and credentials are redacted from the output; the caller never sees the leaked data, and the agent never sees the injected instruction. Adding `CredentialDetector` is deliberate here — documents are a common place for someone to have pasted a live secret.

## Run untrusted parsing code in a no-network sandbox

The second boundary handles the messy-input case. When you build an intelligent document processing llm pipeline that lets the agent write and run parsing code, that code must never touch your host directly. Set `sandbox=True` and Promptise auto-injects five sandbox tools (execute code, read/write file, list files, install package) that run inside a hardened Docker container:

- **seccomp syscall filtering** (Docker's default profile), `no-new-privileges`, and roughly 40 dropped Linux capabilities
- a **read-only root filesystem** and CPU/memory/process/time resource limits
- **network isolation** — `none` by default, so exfiltration over the wire is off the table
- path-traversal and shell-injection prevention on the file tools

So even if the model writes a parser that tries to `curl` a customer record to an external host, there's no network for it to use. The [What you can build gallery](../../resources/showcase.md) walks through more of the sandbox's build ideas, and the [lab-data-analysis guide](../../guides/lab-data-analysis.md) shows the same sandboxed-code pattern applied to number-crunching over extracted data.

## Put it together: a secure document ai pipeline

Here's the full secure document ai pipeline. Both boundaries are single parameters on `build_agent()`, which keeps the security-critical wiring impossible to forget.

```python
import asyncio
from promptise import (
    build_agent, PromptiseSecurityScanner, PIIDetector, NERDetector, CredentialDetector,
)
from promptise.config import StdioServerSpec

async def main():
    scanner = PromptiseSecurityScanner(
        detectors=[PIIDetector(), NERDetector(), CredentialDetector()],
    )
    scanner.warmup()

    agent = await build_agent(
        model="openai:gpt-5-mini",
        servers={"docs": StdioServerSpec(command="python", args=["doc_server.py"])},
        instructions=(
            "You extract structured fields from documents. "
            "List the inbox, read each document, and return a JSON object "
            "with the vendor, total amount, and due date. If a field is missing, "
            "use null. When a format is irregular, write a small Python parser "
            "and run it in the sandbox rather than guessing."
        ),
        guardrails=scanner,   # scan input + output, redact PII/secrets
        sandbox=True,         # run any generated parser in hardened Docker
        observe=True,         # timeline of every LLM turn and tool call
    )

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": "Process every document in the inbox."}]
    })
    print(result["messages"][-1].content)
    await agent.shutdown()

asyncio.run(main())
```

Run it with your own `OPENAI_API_KEY` and a `.txt` file in `./inbox`. The agent discovers the two document tools, reads each file, extracts the fields, and — for anything irregular — writes a parser that executes with no network access. Whatever it produces passes through the scanner on the way out, so a stray SSN or address is redacted before it hits your logs. Because `observe=True`, you also get a full timeline you can inspect to confirm redaction actually fired.

From here the pattern composes cleanly with the rest of Promptise. Point the same agent at additional MCP servers and it becomes one node in a larger system — the [multi-agent systems guide](multi-agent-systems-python.md) shows how a document extractor can hand off to a review agent, and the same guardrails scanner protects a conversational front end like the one in the [customer support agent build](customer-support-ai-agent.md).

## When a plain script is the better fit

Be honest about the tradeoff. If your documents are uniform — the same invoice template from the same vendor, or a fixed CSV schema — you don't need an LLM at all. A deterministic parser with `pdfplumber` and a few regexes is faster, cheaper, and 100% reproducible, and you can bolt Promptise's `PromptiseSecurityScanner` onto its output on its own if you only want the redaction layer. Reach for a document processing ai agent when inputs are *heterogeneous*: many formats, unpredictable layouts, fields that require reading comprehension rather than a fixed offset. That's where the model earns its cost — and where the guardrails and sandbox stop earning their keep from being optional.

## Frequently asked questions

### Does the PII scanning send my documents to a third-party API?

No. Every detector in `PromptiseSecurityScanner` runs locally on your infrastructure. The `PIIDetector` and `CredentialDetector` are pure regex plus Luhn validation, and the `NERDetector` uses a local GLiNER model. Document text never leaves your machine for classification, which is what makes the pipeline suitable for regulated data.

### What stops generated parsing code from harming my host?

The Docker sandbox. With `sandbox=True`, any code the agent runs executes inside a container with seccomp syscall filtering, ~40 dropped capabilities, a read-only root filesystem, resource limits, and network isolation set to `none` by default. The file tools also block path traversal, so generated code can't read or write outside its working directory.

### Can I redact only some PII types and keep others?

Yes. `PromptiseSecurityScanner` is composable — you pass exactly the detectors you want. Include `PIIDetector` for structured identifiers and `NERDetector` for names and addresses, drop `CredentialDetector` if your documents never contain secrets, or add a `CustomRule` with your own regex for a domain-specific identifier like a policy number.

## Next steps

`pip install "promptise[sandbox]"` (the extra brings the Docker client), enable the security scanner, and run your first documents through the sandboxed pipeline. Start with the [Quick Start](../../getting-started/quickstart.md) to get an agent running in a few minutes, then browse the [What you can build gallery](../../resources/showcase.md) for the next document-processing pattern to add.
