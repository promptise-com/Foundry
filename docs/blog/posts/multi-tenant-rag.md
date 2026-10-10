---
title: "Multi-Tenant RAG: Isolate Customer Data in a Shared Store"
description: "RAG multi-tenancy usually means one shared vector store and a hand-written metadata filter on every query, and the first query that forgets the filter leaks…"
keywords: "multi-tenant rag, isolate rag by customer, prevent cross-tenant data leak in a shared vector store, per-tenant vector store isolation, multi-tenant retrieval augmented generation, shared vector store tenant leak"
date: 2026-07-16
slug: multi-tenant-rag
categories:
  - Multi-Tenancy
---

# Multi-Tenant RAG: Isolate Customer Data in a Shared Store

**Multi-tenant RAG** almost always ships as one shared vector store plus a hand-written metadata filter bolted onto every query — and the first query that forgets the filter quietly leaks Acme's documents into Globex's answer. There is no exception, no stack trace, no failing test. The retrieval just returns a neighbor from the wrong customer, the model summarizes it, and you find out when a support ticket arrives. This post is about the retrieval path specifically: why per-query filters are the wrong place to enforce a security boundary, what other frameworks hand you today, and how Promptise Foundry threads the tenant into retrieval automatically so a forgotten filter is structurally impossible.

<!-- more -->

!!! warning "Not legal or compliance advice"
    The information here is general technical information, not legal, regulatory, or compliance advice. Descriptions of any law, regulation, or standard (such as the GDPR, the EU AI Act, HIPAA, SOC 2, or PCI DSS) are simplified and may be incomplete, out of date, or inaccurate, and requirements vary by jurisdiction and situation. Promptise Foundry makes no warranty as to the accuracy or completeness of this content and is not responsible for how you use or rely on it. Using Promptise does not by itself make you or your product compliant with any law or standard. Consult a qualified lawyer or compliance professional before acting on anything here.


## Why the shared vector store is the dangerous part of RAG

Retrieval-augmented generation is attractive precisely because it is shared machinery: one embedding model, one index, one similarity search. In a multi-tenant retrieval augmented generation system that same efficiency is the risk. Every tenant's chunks live in the same collection, ranked by the same distance metric, and the only thing keeping Acme's Alice from retrieving Globex's contract is a `where` clause you remembered to attach.

That clause is a **convention**, and conventions fail silently. Consider the shape of the bug:

- A new endpoint calls `retriever.get_relevant_documents(query)` and someone forgets the filter kwarg. Every tenant now retrieves from the global pool.
- A refactor moves retrieval into a helper that takes a `query` but not a `tenant_id`. The filter is dropped at the call site nobody reviewed.
- A caching layer keys on the query text but not the tenant, so Acme's cached neighbors are served to Globex.

None of these raise. The classic failure is a single query that forgets the filter and leaks across customers, and because retrieval returns *plausible* text, the model happily grounds its answer in it. The boundary you actually need is not "remember to filter" — it is "make it impossible to run an unfiltered retrieval." That is a property of *where* the tenant lives, not *how carefully* each query is written.

## What other frameworks do today

To be fair, the ingredients for isolation exist across the ecosystem — they just leave the boundary as your responsibility on every call:

- **LangChain** gives you `vectorstore.as_retriever(search_kwargs={"filter": {"tenant_id": "acme"}})`. Real, supported metadata filtering — but you pass the filter per query, and nothing stops a retriever built without one.
- **LlamaIndex** has `MetadataFilters` / `ExactMatchFilter` on a `VectorStoreIndex`, applied per query. Same story: the filter is a query argument you must supply each time, not an invariant of the index.
- **Pinecone** offers `namespace="acme"` — genuine store-level partitioning, which is stronger than a metadata filter. But the namespace is a parameter you pass on every `query` and `upsert`; forget it and you hit the default namespace shared by everyone.
- **pgvector on Postgres** lets you isolate with a `WHERE tenant_id = $1` column predicate, or go further with Row-Level Security policies. RLS is genuinely fail-closed *at the database* — but it enforces at the DB layer, requires a per-connection session variable (`SET app.tenant_id = …`) that your pooler must set correctly, and lives outside your agent framework's retrieval call.

The honest summary: every one of these can be made secure. What none of them do is **thread the tenant from the caller's identity into the retrieval automatically**, with a framework-level guard that refuses to run when the owner is unset. Store-level namespaces and DB-level RLS are good primitives; they just sit below the framework and still depend on someone passing the right namespace or setting the right session variable at the right layer. The gap Promptise closes is making per-tenant vector store isolation a property of the *retrieval call itself*, independent of which store you use.

## How Promptise threads the tenant into retrieval automatically

Promptise treats retrieval scope as part of identity, not a query argument. A memory/RAG provider is created in `MemoryScope.PER_USER`, and the "user" it keys on is `CallerContext.isolation_key` — `"{tenant_id}::{user_id}"` when a tenant is present, the plain `user_id` otherwise. That key is derived in exactly one place and auto-propagated from an async contextvar: you pass a `CallerContext` to `ainvoke()`, and the framework reads it back inside the retrieval layer. Your handler never touches the owner id, so there is no call site that can forget it.

!!! note "Memory providers, not `RAGPipeline`"
    The automatic per-caller scoping described here belongs to the memory providers (`InMemoryProvider`, `ChromaProvider`, `Mem0Provider`). A [`RAGPipeline`](../../core/rag.md) exposed with `rag_to_tool()` is one shared corpus: every caller of the agent can retrieve every document in it. To keep several tenants' documents in one RAG store, filter on the caller yourself, as shown in [Who can retrieve what](../../core/rag.md#who-can-retrieve-what).

Here is the whole thing end to end — one user id, two tenants, one shared store, zero query changes:

```python
import asyncio

from promptise import build_agent, CallerContext, InMemoryProvider
from promptise.memory import MemoryScope


async def main() -> None:
    # One shared store, isolation keyed on tenant::user
    knowledge = InMemoryProvider(scope=MemoryScope.PER_USER)

    agent = await build_agent(
        servers={},  # no MCP tools needed for this demo
        model="openai:gpt-5-mini",
        instructions="You are a per-tenant knowledge assistant. Answer only from retrieved context.",
        memory=knowledge,
        memory_auto_store=True,  # store each exchange under the caller's isolation key
    )

    # Same user id "alice", two different tenants -> two disjoint retrieval scopes
    acme = CallerContext(user_id="alice", tenant_id="acme")
    globex = CallerContext(user_id="alice", tenant_id="globex")

    # Acme ingests a private fact
    await agent.ainvoke(
        {"messages": [{"role": "user", "content": "Our production SLA is 4 hours."}]},
        caller=acme,
    )

    # Globex asks the same question — retrieval is scoped to globex::alice
    reply = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is our production SLA?"}]},
        caller=globex,
    )
    print(reply)  # Globex never retrieves Acme's SLA

    await agent.shutdown()


asyncio.run(main())
```

The retrieval that runs for Globex searches with `user_id="globex::alice"`, so it can only match chunks stored under that exact owner. Acme's SLA was stored under `acme::alice`, a provably disjoint key: `CallerContext` construction forbids `::` inside a `user_id` and `:` inside a `tenant_id`, so an attacker cannot forge `acme::alice` by naming their user `acme::alice`. Single colons survive, so SSO subject ids like `google:12345` still work. The same isolation key drives the semantic cache and conversation ownership too, so a leak cannot sneak in through a caching side channel — the mechanics of why a bare `user_id` is not enough are covered in [Same user_id, Two Tenants: Why That Isn't Isolation](same-user-id-across-two-tenants.md).

## The fail-closed guard: no owner, no retrieval

The part competitors leave to discipline, Promptise makes an error. A `PER_USER` provider that is asked to search or store without a resolved owner does not silently fall back to the global pool — it raises `MemoryIsolationError` before any query touches the store:

```python
from promptise import InMemoryProvider
from promptise.memory import MemoryScope, MemoryIsolationError

store = InMemoryProvider(scope=MemoryScope.PER_USER)

try:
    await store.search("production SLA")  # no user_id -> refuses to run
except MemoryIsolationError as exc:
    print("blocked:", exc)  # "…requires a user_id when scope=PER_USER"
```

This is the exact delta versus a forgotten metadata filter. With a per-query `filter` kwarg, "no filter" means "search everything" — the most dangerous default in RAG. With `MemoryScope.PER_USER`, "no owner" means "raise and stop." When a `CallerContext` *is* active, the framework fills the owner in from the contextvar automatically, so the guard only ever fires on the code paths that genuinely lost the identity — a bug you want loud, not silent. This holds identically for the persistent `ChromaProvider` and the enterprise `Mem0Provider`; the guarantee lives at the scoping layer, so it does not depend on any single vector store's own isolation features.

## Wiring it into a real, persistent RAG pipeline

Swapping the in-memory store for a persistent local vector store is a one-line change — the isolation semantics are identical because they come from the scope, not the backend:

```python
from promptise import build_agent, ChromaProvider
from promptise.memory import MemoryScope

knowledge = ChromaProvider(
    collection_name="customer_kb",
    persist_directory=".promptise/chroma",
    scope=MemoryScope.PER_USER,  # tenant::user isolation on a shared collection
)

agent = await build_agent(
    servers={},
    model="openai:gpt-5-mini",
    memory=knowledge,
)
```

All tenants share one Chroma collection on disk, and every retrieval is still partitioned by the caller's isolation key. Because the whole customer's data hangs off one owner id, GDPR erasure is a single call — `await knowledge.purge_user("acme::alice")` deletes exactly that tenant-scoped Alice and nothing else. Where the tenant portion of that key actually comes from in production — a JWT claim on the server or an API-key binding — is covered in [Where Does tenant_id Come From? JWT Claim vs API Key](tenant-id-from-jwt-claim.md), and the full retrieval-and-memory reference lives in the [Memory guide](../../core/memory.md) alongside the broader [RAG for agents](../../core/rag.md) walkthrough.

## When a per-query filter is genuinely fine

Structural isolation is worth reaching for when a single deployment serves customers who must never see each other's data. It is honestly overkill when:

- **You run one store per customer.** If each tenant has its own collection or database, the store boundary already is the tenant boundary; `MemoryScope.SHARED` plus separate providers is simpler.
- **There is one trust domain.** An internal knowledge assistant where every user is in the same organization needs per-user recall, not cross-tenant isolation — `CallerContext(user_id=…)` with no tenant is enough.
- **You are prototyping.** Add isolation before your second paying customer onboards, not before your first demo.

If any of those describe you, LangChain's or LlamaIndex's per-query filters are perfectly reasonable. The moment "one store, many customers" is true, though, a filter you have to remember becomes a filter you will eventually forget — and that is exactly the case a structural invariant is built for. The end-to-end pattern, from token through retrieval to audit, is laid out in the [Secure Multi-Tenant Platform guide](../../guides/secure-multi-tenant-platform.md).

## Frequently asked questions

### How do you prevent a cross-tenant data leak in a shared vector store?

Stop treating the tenant as a per-query filter and make it part of the retrieval's owner key. In Promptise, a provider in `MemoryScope.PER_USER` keys every search and write on `CallerContext.isolation_key` (`tenant::user`), which the framework auto-fills from the caller contextvar. There is no call site that can drop the filter, and a `PER_USER` provider with no resolved owner raises `MemoryIsolationError` instead of searching the global pool.

### Can I isolate RAG by customer without changing every query?

Yes — that is the point. You set `scope=MemoryScope.PER_USER` once when you build the provider and pass a `CallerContext` with a `tenant_id` to `ainvoke()`. Retrieval, storage, cache, and conversation scope all key on the derived isolation key automatically, so your query code stays identical whether it runs for one tenant or a thousand.

### Do I still need Pinecone namespaces or pgvector row-level security?

They remain solid store-level primitives, and you can use them underneath Promptise. The difference is that Promptise enforces per-tenant vector store isolation at the framework's retrieval call, threaded from the caller identity and fail-closed when the owner is missing — so isolation holds the same way across `InMemoryProvider`, `ChromaProvider`, and `Mem0Provider` without depending on each backend's own mechanism or on someone setting the right namespace or session variable per call.

## Next steps

Attach a `ChromaProvider` (or `Mem0Provider`) in `MemoryScope.PER_USER`, pass a `CallerContext` with a `tenant_id`, and watch retrieval isolate per customer with zero query changes. Start with the [Memory guide](../../core/memory.md) for the provider reference, read [RAG for agents](../../core/rag.md) for the retrieval pipeline, then run through the [Secure Multi-Tenant Platform guide](../../guides/secure-multi-tenant-platform.md) to thread the same `tenant_id` from your JWT all the way to storage. `pip install promptise` and make a forgotten filter impossible.
