"""MCPQueue hardening: job ownership, hard cancellation, submit-time
argument validation, advertised job schemas and non-blocking retries.

Regressions covered (all reproduced against v1.2.1):

- Any caller could list, read and cancel every other caller's jobs.
- A cancelled job that never called ``cancel.check()`` kept running and
  its status flipped from ``cancelled`` to ``completed``.
- Wrong job arguments were only detected inside the worker, then retried
  as if the failure were transient.
- ``queue_submit`` did not tell clients which job types exist or what
  arguments they take.
- ``error`` stayed set after a retry succeeded.
- A worker slept through each retry backoff, blocking other jobs.
"""

from __future__ import annotations

import asyncio
import copy
import json
from typing import Any

import pytest

from promptise.mcp.server import (
    APIKeyAuth,
    AuthMiddleware,
    CancellationToken,
    Depends,
    MCPServer,
    ProgressReporter,
    TestClient,
    ToolError,
)
from promptise.mcp.server._queue import (
    InMemoryQueueBackend,
    Job,
    JobStatus,
    MCPQueue,
    QueueCaller,
)

TIMEOUT = 5.0


async def _wait_for_status(queue: MCPQueue, job_id: str, *statuses: str) -> dict[str, Any]:
    """Poll until the job reaches one of *statuses* (fails after TIMEOUT)."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + TIMEOUT
    while True:
        status = await queue.status(job_id)
        if status["status"] in statuses:
            return status
        if loop.time() > deadline:
            raise AssertionError(f"job {job_id} stuck in {status}")
        await asyncio.sleep(0.01)


def _data(result: list[Any]) -> dict[str, Any]:
    return json.loads(result[0].text)


# ---------------------------------------------------------------------------
# Ownership
# ---------------------------------------------------------------------------

KEYS = {
    "key-alice": {"client_id": "alice", "roles": ["analyst"]},
    "key-bob": {"client_id": "bob", "roles": ["analyst"]},
    "key-carol": {"client_id": "carol", "roles": ["admin"]},
    "key-dan": {"client_id": "dan", "roles": ["admin"], "tenant_id": "acme"},
    "key-erin": {"client_id": "erin", "roles": ["analyst"], "tenant_id": "acme"},
}


def _owned_server(**queue_kwargs: Any) -> tuple[MCPServer, MCPQueue]:
    server = MCPServer("owned", require_auth=True)
    server.add_middleware(AuthMiddleware(APIKeyAuth(keys=KEYS)))
    queue = MCPQueue(server, **queue_kwargs)

    @queue.job(name="add")
    async def add(a: int, b: int) -> int:
        return a + b

    return server, queue


def _as(server: MCPServer, key: str) -> TestClient:
    return TestClient(server, meta={"x-api-key": key})


async def _submit(client: TestClient) -> str:
    data = _data(
        await client.call_tool("queue_submit", {"job_type": "add", "args": {"a": 2, "b": 3}})
    )
    return data["job_id"]


class TestJobOwnership:
    async def test_other_client_cannot_read_or_cancel(self):
        server, queue = _owned_server()
        alice, bob = _as(server, "key-alice"), _as(server, "key-bob")
        job_id = await _submit(alice)

        for tool in ("queue_status", "queue_result", "queue_cancel"):
            err = _data(await bob.call_tool(tool, {"job_id": job_id}))["error"]
            # Indistinguishable from a job that does not exist
            assert err["code"] == "JOB_NOT_FOUND", tool
            assert err["message"] == f"Job not found: {job_id}"

        # Bob's cancel attempt did nothing
        assert _data(await alice.call_tool("queue_status", {"job_id": job_id}))["status"] == (
            "pending"
        )

    async def test_list_shows_only_own_jobs(self):
        server, _ = _owned_server()
        alice, bob = _as(server, "key-alice"), _as(server, "key-bob")
        alice_jobs = {await _submit(alice), await _submit(alice)}
        bob_job = await _submit(bob)

        listed = _data(await alice.call_tool("queue_list", {}))
        assert {j["job_id"] for j in listed["jobs"]} == alice_jobs
        assert listed["total"] == 2

        listed = _data(await bob.call_tool("queue_list", {}))
        assert [j["job_id"] for j in listed["jobs"]] == [bob_job]
        assert listed["total"] == 1

    async def test_list_pagination_counts_only_visible_jobs(self):
        server, _ = _owned_server()
        alice, bob = _as(server, "key-alice"), _as(server, "key-bob")
        for _ in range(3):
            await _submit(alice)
            await _submit(bob)

        page = _data(await alice.call_tool("queue_list", {"limit": 2, "offset": 1}))
        assert page["total"] == 3
        assert len(page["jobs"]) == 2

    async def test_owner_can_cancel_own_job(self):
        server, _ = _owned_server()
        alice = _as(server, "key-alice")
        job_id = await _submit(alice)
        assert _data(await alice.call_tool("queue_cancel", {"job_id": job_id}))["status"] == (
            "cancelled"
        )

    async def test_admin_sees_and_cancels_jobs_of_own_tenant(self):
        server, _ = _owned_server()
        alice, bob, carol = (_as(server, k) for k in ("key-alice", "key-bob", "key-carol"))
        alice_job, bob_job = await _submit(alice), await _submit(bob)

        listed = _data(await carol.call_tool("queue_list", {}))
        assert {j["job_id"] for j in listed["jobs"]} == {alice_job, bob_job}
        assert _data(await carol.call_tool("queue_status", {"job_id": alice_job}))["status"] == (
            "pending"
        )
        assert _data(await carol.call_tool("queue_cancel", {"job_id": alice_job}))["status"] == (
            "cancelled"
        )

    async def test_admin_role_never_crosses_tenants(self):
        server, _ = _owned_server()
        erin, dan, carol, alice = (
            _as(server, k) for k in ("key-erin", "key-dan", "key-carol", "key-alice")
        )
        erin_job = await _submit(erin)
        alice_job = await _submit(alice)

        # Admin of tenant "acme" sees the acme job, not the tenant-less one
        listed = _data(await dan.call_tool("queue_list", {}))
        assert [j["job_id"] for j in listed["jobs"]] == [erin_job]
        err = _data(await dan.call_tool("queue_status", {"job_id": alice_job}))["error"]
        assert err["code"] == "JOB_NOT_FOUND"

        # An admin without a tenant does not see acme's job either
        err = _data(await carol.call_tool("queue_cancel", {"job_id": erin_job}))["error"]
        assert err["code"] == "JOB_NOT_FOUND"
        err = _data(await alice.call_tool("queue_status", {"job_id": erin_job}))["error"]
        assert err["code"] == "JOB_NOT_FOUND"

    async def test_admin_override_can_be_disabled(self):
        server, _ = _owned_server(admin_role=None)
        alice, carol = _as(server, "key-alice"), _as(server, "key-carol")
        job_id = await _submit(alice)
        err = _data(await carol.call_tool("queue_status", {"job_id": job_id}))["error"]
        assert err["code"] == "JOB_NOT_FOUND"

    async def test_custom_admin_role(self):
        server, _ = _owned_server(admin_role="analyst")
        alice, bob = _as(server, "key-alice"), _as(server, "key-bob")
        job_id = await _submit(alice)
        assert _data(await bob.call_tool("queue_status", {"job_id": job_id}))["job_id"] == job_id

    async def test_auth_flag_authenticates_queue_tools(self):
        server = MCPServer("opt-in")  # no require_auth
        server.add_middleware(AuthMiddleware(APIKeyAuth(keys=KEYS)))
        queue = MCPQueue(server, auth=True)

        @queue.job(name="add")
        async def add(a: int, b: int) -> int:
            return a + b

        anonymous = _data(
            await TestClient(server).call_tool(
                "queue_submit", {"job_type": "add", "args": {"a": 1, "b": 1}}
            )
        )
        assert anonymous["error"]["code"] == "AUTHENTICATION_ERROR"

        alice, bob = _as(server, "key-alice"), _as(server, "key-bob")
        job_id = await _submit(alice)
        err = _data(await bob.call_tool("queue_status", {"job_id": job_id}))["error"]
        assert err["code"] == "JOB_NOT_FOUND"

    async def test_owner_is_recorded_on_the_job(self):
        server, queue = _owned_server()
        job_id = await _submit(_as(server, "key-erin"))
        job = await queue.backend.get(job_id)
        assert job is not None
        assert (job.owner_client_id, job.owner_tenant_id) == ("erin", "acme")

    async def test_python_api_is_unrestricted(self):
        server, queue = _owned_server()
        job_id = await _submit(_as(server, "key-alice"))
        assert (await queue.status(job_id))["status"] == "pending"
        assert (await queue.list_jobs())["total"] == 1
        # ...and can be scoped explicitly
        with pytest.raises(ToolError) as info:
            await queue.status(job_id, caller=QueueCaller(client_id="bob"))
        assert info.value.code == "JOB_NOT_FOUND"


class TestQueueCaller:
    def test_access_rules(self):
        job = Job(id="j", job_type="t", args={}, owner_client_id="a", owner_tenant_id="t1")
        assert QueueCaller("a", "t1").can_access(job)
        assert not QueueCaller("b", "t1").can_access(job)
        assert QueueCaller("b", "t1", is_admin=True).can_access(job)
        assert not QueueCaller("a", "t2").can_access(job)
        assert not QueueCaller("a", None, is_admin=True).can_access(job)
        anonymous_job = Job(id="k", job_type="t", args={})
        assert QueueCaller().can_access(anonymous_job)


# ---------------------------------------------------------------------------
# Cancellation of running jobs
# ---------------------------------------------------------------------------


class TestCancelRunningJob:
    async def test_job_ignoring_the_token_is_stopped_and_stays_cancelled(self):
        queue = MCPQueue(max_workers=1)
        started = asyncio.Event()
        outcome: list[str] = []

        @queue.job(name="stubborn")
        async def stubborn() -> dict:
            started.set()
            try:
                await asyncio.Event().wait()  # never checks cancel
            except asyncio.CancelledError:
                outcome.append("task cancelled")
                raise
            outcome.append("ran to the end")
            return {"done": True}

        job_id = (await queue.submit("stubborn", {}))["job_id"]
        await queue.start()
        try:
            await asyncio.wait_for(started.wait(), TIMEOUT)
            assert (await queue.cancel(job_id))["status"] == "cancelled"
            for _ in range(20):
                await asyncio.sleep(0.01)
            status = await queue.status(job_id)
            assert status["status"] == "cancelled"
            assert outcome == ["task cancelled"]
            assert "result" not in await queue.get_result(job_id)
        finally:
            await queue.stop()

    async def test_result_returned_after_cancel_is_discarded(self):
        queue = MCPQueue(max_workers=1)
        started = asyncio.Event()
        returned = asyncio.Event()

        @queue.job(name="swallows_cancel")
        async def swallows_cancel() -> dict:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                returned.set()
                return {"late": True}  # swallows the cancellation
            return {"never": True}

        job_id = (await queue.submit("swallows_cancel", {}))["job_id"]
        await queue.start()
        try:
            await asyncio.wait_for(started.wait(), TIMEOUT)
            await queue.cancel(job_id)
            await asyncio.wait_for(returned.wait(), TIMEOUT)
            for _ in range(10):
                await asyncio.sleep(0.01)
            result = await queue.get_result(job_id)
            assert result["status"] == "cancelled"
            assert "result" not in result
            job = await queue.backend.get(job_id)
            assert job is not None and job.result is None
        finally:
            await queue.stop()

    async def test_token_is_set_when_the_task_is_cancelled(self):
        queue = MCPQueue(max_workers=1)
        started = asyncio.Event()
        seen: list[bool] = []

        @queue.job(name="watcher")
        async def watcher(cancel: CancellationToken) -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                seen.append(cancel.is_cancelled)

        job_id = (await queue.submit("watcher", {}))["job_id"]
        await queue.start()
        try:
            await asyncio.wait_for(started.wait(), TIMEOUT)
            await queue.cancel(job_id)
            for _ in range(10):
                await asyncio.sleep(0.01)
            assert seen == [True]
        finally:
            await queue.stop()

    async def test_grace_period_lets_a_cooperative_job_stop_itself(self):
        queue = MCPQueue(max_workers=1, cancel_grace_period=TIMEOUT)
        started = asyncio.Event()
        seen: list[str | None] = []

        @queue.job(name="polite")
        async def polite(cancel: CancellationToken) -> str:
            started.set()
            await cancel.wait()
            seen.append(cancel.reason)
            cancel.check()
            return "unreachable"

        job_id = (await queue.submit("polite", {}))["job_id"]
        await queue.start()
        try:
            await asyncio.wait_for(started.wait(), TIMEOUT)
            await queue.cancel(job_id)
            while job_id in queue._job_tasks:  # until the worker has let go
                await asyncio.sleep(0.01)
            assert seen == ["Cancelled by user"]
            assert (await queue.status(job_id))["status"] == "cancelled"
        finally:
            await queue.stop()

    async def test_worker_is_free_after_cancel(self):
        queue = MCPQueue(max_workers=1)
        started = asyncio.Event()

        @queue.job(name="stubborn")
        async def stubborn() -> None:
            started.set()
            await asyncio.Event().wait()

        @queue.job(name="quick")
        async def quick() -> str:
            return "ok"

        stuck = (await queue.submit("stubborn", {}))["job_id"]
        await queue.start()
        try:
            await asyncio.wait_for(started.wait(), TIMEOUT)
            await queue.cancel(stuck)
            quick_id = (await queue.submit("quick", {}))["job_id"]
            assert (await _wait_for_status(queue, quick_id, "completed"))["status"] == "completed"
        finally:
            await queue.stop()

    async def test_stop_cancels_running_jobs(self):
        queue = MCPQueue(max_workers=1)
        started = asyncio.Event()

        @queue.job(name="endless")
        async def endless() -> None:
            started.set()
            await asyncio.Event().wait()

        job_id = (await queue.submit("endless", {}))["job_id"]
        await queue.start()
        await asyncio.wait_for(started.wait(), TIMEOUT)
        await queue.stop()
        status = await queue.status(job_id)
        assert status["status"] == "cancelled"
        assert status["error"] == "Queue stopped before the job finished"


# ---------------------------------------------------------------------------
# Argument validation on submit
# ---------------------------------------------------------------------------


def _report_queue() -> MCPQueue:
    queue = MCPQueue(max_workers=1)

    @queue.job(name="report", max_retries=2, backoff_base=0.01)
    async def report(region: str, quarter: str, rows: int = 10) -> dict:
        """Build a quarterly report."""
        return {"region": region, "quarter": quarter, "rows": rows}

    return queue


class TestSubmitValidation:
    @pytest.mark.parametrize(
        ("args", "detail"),
        [
            ({"region": "EMEA"}, "quarter: Field required"),
            ({"region": "EMEA", "quarter": "Q3", "currency": "EUR"}, "currency: Extra inputs"),
            ({"region": "EMEA", "quarter": "Q3", "rows": "many"}, "rows: Input should be"),
        ],
    )
    async def test_bad_arguments_are_rejected_without_a_job(self, args, detail):
        queue = _report_queue()
        with pytest.raises(ToolError) as info:
            await queue.submit("report", args)
        err = info.value
        assert err.code == "INVALID_JOB_ARGUMENTS"
        assert err.retryable is False
        assert detail in str(err)
        assert '"required":["region","quarter"]' in (err.suggestion or "")
        assert (await queue.list_jobs())["total"] == 0

    async def test_rejection_reaches_the_client_as_a_tool_error(self):
        server = MCPServer("v")
        queue = MCPQueue(server)

        @queue.job(name="report")
        async def report(region: str, quarter: str) -> dict:
            return {}

        data = _data(
            await TestClient(server).call_tool(
                "queue_submit", {"job_type": "report", "args": {"region": "EMEA"}}
            )
        )
        assert data["error"]["code"] == "INVALID_JOB_ARGUMENTS"
        assert data["error"]["retryable"] is False

    async def test_valid_arguments_are_coerced_and_run_once(self):
        queue = _report_queue()
        job_id = (await queue.submit("report", {"region": "EMEA", "quarter": "Q3", "rows": "7"}))[
            "job_id"
        ]
        await queue.start()
        try:
            await _wait_for_status(queue, job_id, "completed")
            result = await queue.get_result(job_id)
            assert result["result"] == {"region": "EMEA", "quarter": "Q3", "rows": 7}
            assert result["attempts"] == 1
        finally:
            await queue.stop()

    async def test_injected_parameters_are_not_arguments(self):
        queue = MCPQueue(max_workers=1)

        @queue.job(name="crawl")
        async def crawl(
            pages: int,
            progress: ProgressReporter,
            cancel: CancellationToken = Depends(CancellationToken),
        ) -> int:
            cancel.check()
            await progress.report(pages, total=pages, message="done crawling")
            return pages

        assert set(queue._job_defs["crawl"].input_schema["properties"]) == {"pages"}
        job_id = (await queue.submit("crawl", {"pages": 3}))["job_id"]
        await queue.start()
        try:
            status = await _wait_for_status(queue, job_id, "completed")
            assert status["progress_message"] == "done crawling"
        finally:
            await queue.stop()

    async def test_kwargs_handler_accepts_extra_arguments(self):
        queue = MCPQueue(max_workers=1)

        @queue.job(name="flexible")
        async def flexible(name: str, **options: Any) -> dict:
            return {"name": name, **options}

        assert "additionalProperties" not in queue._job_defs["flexible"].input_schema
        job_id = (await queue.submit("flexible", {"name": "x", "mode": "fast"}))["job_id"]
        await queue.start()
        try:
            await _wait_for_status(queue, job_id, "completed")
            assert (await queue.get_result(job_id))["result"] == {"name": "x", "mode": "fast"}
        finally:
            await queue.stop()

    async def test_untyped_parameters_accept_any_value(self):
        queue = MCPQueue(max_workers=1)

        @queue.job(name="echo")
        async def echo(value):  # type: ignore[no-untyped-def]
            return value

        job_id = (await queue.submit("echo", {"value": [1, {"a": 2}]}))["job_id"]
        await queue.start()
        try:
            await _wait_for_status(queue, job_id, "completed")
            assert (await queue.get_result(job_id))["result"] == [1, {"a": 2}]
        finally:
            await queue.stop()


# ---------------------------------------------------------------------------
# Tool list: job types, schemas, descriptions
# ---------------------------------------------------------------------------


class TestAdvertisedTools:
    async def _tools(self, server: MCPServer) -> dict[str, Any]:
        return {t.name: t for t in await TestClient(server).list_tools()}

    async def test_submit_lists_job_types_and_argument_schemas(self):
        server = MCPServer("listed")
        queue = MCPQueue(server)

        @queue.job(name="generate_report")
        async def generate_report(
            region: str, quarter: str, progress: ProgressReporter, title: str = "Q"
        ) -> dict:
            """Generate a quarterly report."""
            return {}

        @queue.job(name="ping")
        async def ping() -> str:
            return "pong"

        submit = (await self._tools(server))["queue_submit"]
        assert "- generate_report: Generate a quarterly report. args schema: " in (
            submit.description
        )
        schema_line = next(
            line for line in submit.description.splitlines() if "generate_report" in line
        )
        schema = json.loads(schema_line.split("args schema: ", 1)[1])
        assert schema == {
            "additionalProperties": False,
            "properties": {
                "quarter": {"type": "string"},
                "region": {"type": "string"},
                "title": {"default": "Q", "type": "string"},
            },
            "required": ["region", "quarter"],
            "type": "object",
        }
        assert "- ping: ping args schema: " in submit.description
        props = submit.inputSchema["properties"]
        assert props["job_type"]["enum"] == ["generate_report", "ping"]
        assert props["priority"]["enum"] == ["low", "normal", "high", "critical"]

    async def test_custom_prefix_is_advertised(self):
        server = MCPServer("prefixed")
        queue = MCPQueue(server, tool_prefix="jobs")

        @queue.job(name="ping")
        async def ping() -> str:
            return "pong"

        tools = await self._tools(server)
        assert tools["jobs_submit"].inputSchema["properties"]["job_type"]["enum"] == ["ping"]
        assert "jobs_status" in tools["jobs_submit"].description

    async def test_status_describes_progress_as_a_fraction(self):
        server = MCPServer("described")
        MCPQueue(server)
        description = (await self._tools(server))["queue_status"].description
        assert "fraction from 0.0 to 1.0" in description
        assert "percentage" not in description

    async def test_bad_priority_and_status_are_validation_errors(self):
        server = MCPServer("checked")
        queue = MCPQueue(server)

        @queue.job(name="ping")
        async def ping() -> str:
            return "pong"

        client = TestClient(server)
        bad = _data(
            await client.call_tool("queue_submit", {"job_type": "ping", "priority": "urgent"})
        )
        assert bad["error"]["code"] == "VALIDATION_ERROR"
        bad = _data(await client.call_tool("queue_list", {"status": "done"}))
        assert bad["error"]["code"] == "VALIDATION_ERROR"


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------


class TestRetries:
    async def test_error_is_cleared_after_a_successful_retry(self):
        queue = MCPQueue(max_workers=1)
        calls: list[int] = []

        @queue.job(name="flaky", max_retries=2, backoff_base=0.01)
        async def flaky() -> str:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("upstream hiccup")
            return "ok"

        job_id = (await queue.submit("flaky", {}))["job_id"]
        await queue.start()
        try:
            status = await _wait_for_status(queue, job_id, "completed")
            assert status["attempts"] == 2
            assert "error" not in status
        finally:
            await queue.stop()

    async def test_backoff_does_not_occupy_a_worker(self):
        queue = MCPQueue(max_workers=1)
        failed_once = asyncio.Event()

        @queue.job(name="flaky", max_retries=1, backoff_base=60)
        async def flaky() -> None:
            failed_once.set()
            raise RuntimeError("try again later")

        @queue.job(name="quick")
        async def quick() -> str:
            return "ok"

        flaky_id = (await queue.submit("flaky", {}))["job_id"]
        await queue.start()
        try:
            await asyncio.wait_for(failed_once.wait(), TIMEOUT)
            quick_id = (await queue.submit("quick", {}))["job_id"]
            # The only worker is free while flaky waits out its 60s backoff
            await _wait_for_status(queue, quick_id, "completed")
            status = await queue.status(flaky_id)
            assert status["status"] == "pending"
            assert "Retrying in 60" in status["error"]

            # Cancelling during the backoff drops the retry
            await queue.cancel(flaky_id)
            assert queue._retry_timers == {}
            assert (await queue.status(flaky_id))["status"] == "cancelled"
        finally:
            await queue.stop()

    async def test_retry_runs_after_backoff(self):
        queue = MCPQueue(max_workers=1)
        attempts: list[int] = []

        @queue.job(name="flaky", max_retries=1, backoff_base=0.05)
        async def flaky() -> str:
            attempts.append(1)
            if len(attempts) == 1:
                raise RuntimeError("once")
            return "ok"

        job_id = (await queue.submit("flaky", {}))["job_id"]
        await queue.start()
        try:
            await _wait_for_status(queue, job_id, "completed")
            assert len(attempts) == 2
        finally:
            await queue.stop()

    async def test_stop_requeues_jobs_waiting_on_backoff(self):
        queue = MCPQueue(max_workers=1)
        failed_once = asyncio.Event()

        @queue.job(name="flaky", max_retries=1, backoff_base=60)
        async def flaky() -> None:
            failed_once.set()
            raise RuntimeError("later")

        job_id = (await queue.submit("flaky", {}))["job_id"]
        await queue.start()
        await asyncio.wait_for(failed_once.wait(), TIMEOUT)
        while job_id not in queue._retry_timers:
            await asyncio.sleep(0.01)
        await queue.stop()

        job = await queue.backend.dequeue()
        assert job is not None and job.id == job_id
        assert job.status is JobStatus.PENDING


# ---------------------------------------------------------------------------
# The documented TestClient pattern
# ---------------------------------------------------------------------------


@pytest.fixture
def documented_queue():
    server = MCPServer(name="test")
    queue = MCPQueue(server, max_workers=2)

    @queue.job(name="add")
    async def add(a: int, b: int) -> int:
        return a + b

    return server, queue


async def test_documented_testclient_example(documented_queue):
    """Mirrors the 'Testing with TestClient' example in docs/mcp/server/queue.md."""
    server, queue = documented_queue
    client = TestClient(server)
    await queue.start()  # TestClient does not run startup hooks
    try:
        resp = await client.call_tool("queue_submit", {"job_type": "add", "args": {"a": 2, "b": 3}})
        submitted = json.loads(resp[0].text)
        assert submitted["status"] == "pending"
        job_id = submitted["job_id"]

        for _ in range(50):
            status = json.loads(
                (await client.call_tool("queue_status", {"job_id": job_id}))[0].text
            )
            if status["status"] == "completed":
                break
            await asyncio.sleep(0.05)

        result = json.loads((await client.call_tool("queue_result", {"job_id": job_id}))[0].text)
        assert result["result"] == 5
    finally:
        await queue.stop()


# ---------------------------------------------------------------------------
# Shared backends (records are copies, as with any external store)
# ---------------------------------------------------------------------------


class _CopyingBackend(InMemoryQueueBackend):
    """Hands out and stores copies, like a database-backed QueueBackend."""

    async def enqueue(self, job: Job) -> None:
        await super().enqueue(copy.deepcopy(job))

    async def dequeue(self) -> Job | None:
        job = await super().dequeue()
        return copy.deepcopy(job) if job is not None else None

    async def get(self, job_id: str) -> Job | None:
        job = await super().get(job_id)
        return copy.deepcopy(job) if job is not None else None

    async def update(self, job: Job) -> None:
        await super().update(copy.deepcopy(job))


async def test_cancel_from_another_replica_is_not_overwritten():
    backend = _CopyingBackend()
    release = asyncio.Event()
    started = asyncio.Event()

    def make_queue() -> MCPQueue:
        queue = MCPQueue(backend=backend, max_workers=1)

        @queue.job(name="report")
        async def report() -> str:
            started.set()
            await release.wait()
            return "done"

        return queue

    replica_a, replica_b = make_queue(), make_queue()
    job_id = (await replica_a.submit("report", {}))["job_id"]
    await replica_a.start()
    try:
        await asyncio.wait_for(started.wait(), TIMEOUT)
        # Replica B has no handle on A's task; it can only mark the record.
        assert (await replica_b.cancel(job_id))["status"] == "cancelled"
        release.set()
        while job_id in replica_a._job_tasks:
            await asyncio.sleep(0.01)
        result = await replica_b.get_result(job_id)
        assert result["status"] == "cancelled"
        assert "result" not in result
    finally:
        await replica_a.stop()
