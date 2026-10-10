"""Audit log files: what AuditMiddleware writes, and verify_audit_log / ``promptise audit verify``.

Logs are written through the real middleware pipeline (``TestClient``), then
tampered with the way an attacker (or a crash) would, and the verifier must
name the kind of change and the line it is on.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from promptise.cli import app
from promptise.mcp.server import (
    AuditMiddleware,
    MCPServer,
    TestClient,
    verify_audit_log,
)
from promptise.mcp.server._context import RequestContext

KEY = "audit-test-key-0123456789abcdef"


def _server(audit: AuditMiddleware) -> MCPServer:
    server = MCPServer(name="records")
    server.add_middleware(audit)

    @server.tool()
    async def view_record(patient_id: str) -> str:
        await asyncio.sleep(0)
        return f"record {patient_id}"

    @server.tool()
    async def fail(reason: str) -> str:
        raise RuntimeError(reason)

    return server


async def _write(path: Path, calls: int, *, key: str = KEY, start: int = 0) -> AuditMiddleware:
    """One server process: ``calls`` tool calls audited into ``path``."""
    audit = AuditMiddleware(log_path=str(path), hmac_secret=key, include_args=True)
    client = TestClient(_server(audit))
    for i in range(start, start + calls):
        await client.call_tool("view_record", {"patient_id": f"P-{i}"})
    return audit


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines()


def _rewrite(path: Path, lines: list[str], *, newline: bool = True) -> None:
    path.write_text("\n".join(lines) + ("\n" if newline else ""))


@pytest.fixture
def log(tmp_path: Path) -> Path:
    path = tmp_path / "audit.jsonl"
    asyncio.run(_write(path, 6))
    return path


# ---------------------------------------------------------------------------
# A real log verifies
# ---------------------------------------------------------------------------


class TestIntactLog:
    def test_log_written_through_the_middleware_verifies(self, log: Path) -> None:
        report = verify_audit_log(log, KEY)
        assert report.ok and report.continuous and not report.truncated
        assert report.entries == 6
        assert report.problems == () and report.warnings == ()
        assert report.first_problem is None
        last = json.loads(_lines(log)[-1])
        assert report.last_hash == last["hmac"] and report.last_seq == 5 == last["seq"]

    def test_entries_record_the_tool_arguments(self, log: Path) -> None:
        first = json.loads(_lines(log)[0])
        assert first["args"] == {"patient_id": "P-0"}
        assert first["seq"] == 0 and first["prev_hash"] == "0" * 64

    def test_bytes_key_and_pathlike_accepted(self, log: Path) -> None:
        assert verify_audit_log(str(log), KEY.encode()).ok

    async def test_existing_empty_file_starts_a_plain_chain(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        path.write_text("")
        await _write(path, 2)
        assert "resumed" not in json.loads(_lines(path)[0])
        assert verify_audit_log(path, KEY).restarts == ()

    def test_empty_log_is_ok(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        report = verify_audit_log(path, KEY)
        assert report.ok and report.entries == 0 and report.last_hash is None

    def test_empty_key_rejected(self, log: Path) -> None:
        with pytest.raises(ValueError):
            verify_audit_log(log, "")

    def test_wrong_key_is_one_clear_problem(self, log: Path) -> None:
        report = verify_audit_log(log, "not-the-key")
        assert not report.ok
        assert [p.kind for p in report.problems] == ["wrong_key"]
        assert report.first_problem is not None and report.first_problem.line == 1

    async def test_concurrent_calls_verify(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        audit = AuditMiddleware(log_path=str(path), hmac_secret=KEY, include_args=True)
        client = TestClient(_server(audit))
        await asyncio.gather(
            *[client.call_tool("view_record", {"patient_id": f"P-{i}"}) for i in range(25)],
            client.call_tool("fail", {"reason": "boom"}),
        )
        report = verify_audit_log(path, KEY)
        assert report.ok and report.entries == 26
        errors = [json.loads(line) for line in _lines(path) if '"status": "error"' in line]
        assert errors[0]["error"] == "RuntimeError: boom"


# ---------------------------------------------------------------------------
# Tampering, each kind reported precisely
# ---------------------------------------------------------------------------


class TestTampering:
    def test_edited_entry(self, log: Path) -> None:
        lines = _lines(log)
        entry = json.loads(lines[2])
        entry["args"]["patient_id"] = "P-999"
        lines[2] = json.dumps(entry)
        _rewrite(log, lines)
        report = verify_audit_log(log, KEY)
        assert not report.ok
        problem = report.first_problem
        assert problem is not None
        assert (problem.kind, problem.line) == ("modified", 3)
        assert len(report.problems) == 1  # the chain resumes after the edited entry

    def test_edited_entry_with_recomputed_hash_without_the_key(self, log: Path) -> None:
        lines = _lines(log)
        entry = json.loads(lines[2])
        entry["status"] = "error"
        entry["hmac"] = "f" * 64
        lines[2] = json.dumps(entry)
        _rewrite(log, lines)
        problem = verify_audit_log(log, KEY).first_problem
        assert problem is not None and problem.line == 3
        assert problem.kind in {"bad_signature", "modified", "inserted"}

    def test_deleted_entries(self, log: Path) -> None:
        lines = _lines(log)
        _rewrite(log, lines[:2] + lines[4:])  # drop seq 2 and 3
        report = verify_audit_log(log, KEY)
        problem = report.first_problem
        assert problem is not None
        assert (problem.kind, problem.line) == ("deleted", 3)
        assert "2 entries deleted" in problem.message and "seq 1 -> 4" in problem.message

    def test_deleted_first_entries(self, log: Path) -> None:
        _rewrite(log, _lines(log)[2:])
        problem = verify_audit_log(log, KEY).first_problem
        assert problem is not None
        assert (problem.kind, problem.line) == ("missing_start", 1)
        assert "first 2 entries" in problem.message

    def test_reordered_entries(self, log: Path) -> None:
        lines = _lines(log)
        lines[2], lines[3] = lines[3], lines[2]
        _rewrite(log, lines)
        report = verify_audit_log(log, KEY)
        problem = report.first_problem
        assert problem is not None
        assert (problem.kind, problem.line) == ("reordered", 3)
        assert "line 4" in problem.message
        assert {p.kind for p in report.problems} == {"reordered"}

    def test_inserted_forged_entry(self, log: Path) -> None:
        lines = _lines(log)
        forged = json.loads(lines[2])
        forged.update(request_id="forged", seq=3, prev_hash=forged["hmac"], hmac="a" * 64)
        _rewrite(log, lines[:3] + [json.dumps(forged)] + lines[3:])
        report = verify_audit_log(log, KEY)
        assert [(p.kind, p.line) for p in report.problems] == [("inserted", 4)]

    def test_inserted_copy_of_a_real_entry(self, log: Path) -> None:
        lines = _lines(log)
        _rewrite(log, lines[:4] + [lines[1]] + lines[4:])
        report = verify_audit_log(log, KEY)
        assert [(p.kind, p.line) for p in report.problems] == [("duplicate", 5)]
        assert "line 2" in report.problems[0].message

    async def test_inserted_entry_from_another_log_with_the_same_key(self, tmp_path: Path) -> None:
        mine, other = tmp_path / "mine.jsonl", tmp_path / "other.jsonl"
        await _write(mine, 4)
        await _write(other, 4, start=100)
        lines = _lines(mine)
        _rewrite(mine, lines[:2] + [_lines(other)[2]] + lines[2:])
        report = verify_audit_log(mine, KEY)
        assert [(p.kind, p.line) for p in report.problems] == [("inserted", 3)]

    def test_inserted_garbage_line(self, log: Path) -> None:
        lines = _lines(log)
        _rewrite(log, lines[:3] + ["not json at all"] + lines[3:])
        report = verify_audit_log(log, KEY)
        assert [(p.kind, p.line) for p in report.problems] == [("malformed", 4)]

    def test_unsigned_entry(self, log: Path) -> None:
        lines = _lines(log)
        entry = json.loads(lines[1])
        del entry["hmac"]
        lines[1] = json.dumps(entry)
        _rewrite(log, lines)
        problem = verify_audit_log(log, KEY).first_problem
        assert problem is not None and (problem.kind, problem.line) == ("unsigned", 2)

    def test_cut_tail_is_caught_by_an_anchor(self, log: Path) -> None:
        anchor = verify_audit_log(log, KEY).last_hash
        _rewrite(log, _lines(log)[:-2])
        # A hash chain cannot show entries removed from its end on its own...
        assert verify_audit_log(log, KEY).ok
        # ...a hash recorded earlier can.
        report = verify_audit_log(log, KEY, anchor=anchor)
        assert [p.kind for p in report.problems] == ["anchor_missing"]
        assert verify_audit_log(log, KEY, anchor=json.loads(_lines(log)[1])["hmac"]).ok


# ---------------------------------------------------------------------------
# Crashes and restarts
# ---------------------------------------------------------------------------


class TestCrashAndRestart:
    def test_truncated_last_line_is_a_crash_not_tampering(self, log: Path) -> None:
        lines = _lines(log)
        _rewrite(log, lines[:-1] + [lines[-1][:57]], newline=False)
        report = verify_audit_log(log, KEY)
        assert report.ok and report.truncated
        assert [(w.kind, w.line) for w in report.warnings] == [("truncated", 6)]
        assert report.entries == 5

    def test_truncated_last_line_that_is_not_an_entry_is_tampering(self, log: Path) -> None:
        log.write_text(log.read_text() + "garbage")
        report = verify_audit_log(log, KEY)
        assert [(p.kind, p.line) for p in report.problems] == [("malformed", 7)]

    def test_complete_last_line_cut_short_is_tampering(self, log: Path) -> None:
        # A crash never leaves a newline after a partial entry.
        lines = _lines(log)
        _rewrite(log, lines[:-1] + [lines[-1][:57]])
        assert [p.kind for p in verify_audit_log(log, KEY).problems] == ["malformed"]

    async def test_restart_continues_the_chain(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        first = await _write(path, 3)
        second = await _write(path, 2, start=3)
        lines = [json.loads(line) for line in _lines(path)]
        assert lines[3]["prev_hash"] == lines[2]["hmac"]
        assert lines[3]["resumed"] is True and lines[3]["seq"] == 3
        assert "resumed" not in lines[4]
        assert first.verify_chain() and second.verify_chain()
        report = verify_audit_log(path, KEY)
        assert report.ok and report.continuous and report.entries == 5
        assert report.restarts == ((str(path), 4),)

    async def test_restart_after_a_crash_mid_write(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        await _write(path, 3)
        partial = _lines(path)[1][:80]
        with open(path, "a") as f:
            f.write(partial)  # the process died during this write
        await _write(path, 2, start=3)
        lines = _lines(path)
        assert lines[3] == partial  # kept, on a line of its own
        report = verify_audit_log(path, KEY)
        assert report.ok and report.continuous
        assert [(w.kind, w.line) for w in report.warnings] == [("crash_fragment", 4)]
        assert report.restarts == ((str(path), 5),)
        assert json.loads(lines[4])["prev_hash"] == json.loads(lines[2])["hmac"]

    async def test_crash_during_the_very_first_write(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        path.write_text('{"timestamp": 1791')
        await _write(path, 2)
        report = verify_audit_log(path, KEY)
        assert report.ok and report.continuous and report.entries == 2
        assert [(w.kind, w.line) for w in report.warnings] == [("crash_fragment", 1)]

    async def test_garbage_before_a_resumed_entry_is_still_tampering(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        await _write(path, 2)
        await _write(path, 2, start=2)
        lines = _lines(path)
        _rewrite(path, lines[:2] + ["rm -rf /"] + lines[2:])
        assert [(p.kind, p.line) for p in verify_audit_log(path, KEY).problems] == [
            ("malformed", 3)
        ]

    async def test_chain_reset_is_a_warning(self, tmp_path: Path) -> None:
        # Two chains in one file, as servers before 1.3.0 wrote on restart.
        a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
        await _write(a, 3)
        await _write(b, 2, start=3)
        joined = tmp_path / "joined.jsonl"
        joined.write_text(a.read_text() + b.read_text())
        report = verify_audit_log(joined, KEY)
        assert report.ok and not report.continuous
        assert [(w.kind, w.line) for w in report.warnings] == [("chain_reset", 4)]

    async def test_key_rotation_on_the_same_file(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "audit.jsonl"
        await _write(path, 3, key="old-key-old-key")
        with caplog.at_level(logging.WARNING, logger="promptise.server"):
            await _write(path, 2, key="new-key-new-key", start=3)
        assert "not signed with the current key" in caplog.text
        assert verify_audit_log(path, ["new-key-new-key", "old-key-old-key"]).ok
        report = verify_audit_log(path, "new-key-new-key")
        assert not report.ok and report.first_problem is not None
        assert report.first_problem.line == 1

    async def test_file_rotated_while_running(self, tmp_path: Path) -> None:
        path, rotated = tmp_path / "audit.jsonl", tmp_path / "audit.1.jsonl"
        audit = AuditMiddleware(log_path=str(path), hmac_secret=KEY, include_args=True)
        client = TestClient(_server(audit))
        for i in range(3):
            await client.call_tool("view_record", {"patient_id": f"P-{i}"})
        os.rename(path, rotated)  # logrotate (create mode)
        for i in range(3, 5):
            await client.call_tool("view_record", {"patient_id": f"P-{i}"})
        alone = verify_audit_log(path, KEY)
        assert [p.kind for p in alone.problems] == ["missing_start"]
        together = verify_audit_log([rotated, path], KEY)
        assert together.ok and together.continuous and together.entries == 5


# ---------------------------------------------------------------------------
# promptise audit verify
# ---------------------------------------------------------------------------


def _cli(*args: str, env: dict[str, Any] | None = None) -> Any:
    runner = CliRunner(env={"COLUMNS": "200", "PROMPTISE_NO_DOTENV": "1", **(env or {})})
    return runner.invoke(app, ["audit", "verify", *args])


class TestCli:
    def test_intact_log_exits_zero_and_never_prints_the_key(self, log: Path) -> None:
        result = _cli(str(log), env={"PROMPTISE_AUDIT_SECRET": KEY})
        assert result.exit_code == 0, result.output
        assert "OK" in result.output and "6 entries" in result.output
        assert KEY not in result.output

    def test_tampered_log_exits_one_and_names_the_line(self, log: Path) -> None:
        lines = _lines(log)
        _rewrite(log, lines[:2] + lines[3:])
        result = _cli(str(log), "--key-env", "MY_AUDIT_KEY", env={"MY_AUDIT_KEY": KEY})
        assert result.exit_code == 1
        assert f"{log}:3: deleted" in result.output
        assert KEY not in result.output

    def test_wrong_key_exits_one(self, log: Path) -> None:
        result = _cli(str(log), env={"PROMPTISE_AUDIT_SECRET": "some-other-key"})
        assert result.exit_code == 1
        assert "wrong_key" in result.output and "some-other-key" not in result.output

    def test_missing_key_variable_exits_two(self, log: Path) -> None:
        result = _cli(str(log), "--key-env", "NOT_SET_ANYWHERE_XYZ")
        assert result.exit_code == 2
        assert "NOT_SET_ANYWHERE_XYZ" in result.output

    def test_missing_file_exits_two(self, tmp_path: Path) -> None:
        result = _cli(str(tmp_path / "nope.jsonl"), env={"PROMPTISE_AUDIT_SECRET": KEY})
        assert result.exit_code == 2

    def test_truncated_tail_warns_and_strict_fails(self, log: Path) -> None:
        log.write_text(log.read_text() + '{"timestamp": 17')
        env = {"PROMPTISE_AUDIT_SECRET": KEY}
        result = _cli(str(log), env=env)
        assert result.exit_code == 0 and "truncated" in result.output
        assert _cli(str(log), "--strict", env=env).exit_code == 1

    def test_json_report(self, log: Path) -> None:
        result = _cli(str(log), "--json", env={"PROMPTISE_AUDIT_SECRET": KEY})
        assert result.exit_code == 0
        report = json.loads(result.output)
        assert report["ok"] is True and report["entries"] == 6 and report["continuous"]
        assert KEY not in result.output

    def test_anchor_option(self, log: Path) -> None:
        env = {"PROMPTISE_AUDIT_SECRET": KEY}
        anchor = verify_audit_log(log, KEY).last_hash
        assert anchor is not None
        _rewrite(log, _lines(log)[:-1])
        result = _cli(str(log), "--anchor", anchor, env=env)
        assert result.exit_code == 1 and "anchor_missing" in result.output

    def test_rotated_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "audit.jsonl"
        asyncio.run(_write(path, 2, key="old-key-old-key"))
        asyncio.run(_write(path, 2, key="new-key-new-key", start=2))
        env = {"K_NEW": "new-key-new-key", "K_OLD": "old-key-old-key"}
        assert _cli(str(path), "--key-env", "K_NEW", env=env).exit_code == 1
        result = _cli(str(path), "--key-env", "K_NEW", "--key-env", "K_OLD", env=env)
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Unit-level: entries signed in the form they are read back
# ---------------------------------------------------------------------------


async def test_non_string_keys_in_arguments_verify_from_the_file(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    mw = AuditMiddleware(log_path=str(path), hmac_secret=KEY, include_args=True)
    ctx = RequestContext(server_name="s", tool_name="t", client_id="c")
    # Sorted as ints (2, 10) when signed, as strings ("10", "2") when read back.
    args = {"buckets": {2: "a", 10: "b"}, "tags": {"x"}}
    ctx.state["_tool_arguments"] = args
    ctx.state.update(args)  # where releases before 1.3.0 read "args" from

    async def call_next(c: RequestContext) -> str:
        return "ok"

    await mw(ctx, call_next)
    assert mw.verify_chain()
    assert verify_audit_log(path, KEY).ok
    [entry] = mw.entries
    assert entry["args"]["buckets"] == {"2": "a", "10": "b"}  # as read back
