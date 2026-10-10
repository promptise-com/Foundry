"""Verify an audit log file written by :class:`AuditMiddleware`.

::

    from promptise.mcp.server import verify_audit_log

    report = verify_audit_log("audit.jsonl", os.environ["PROMPTISE_AUDIT_SECRET"])
    if not report.ok:
        print(report.first_problem)   # audit.jsonl:7: deleted: ...

Also on the command line: ``promptise audit verify audit.jsonl``.
"""

from __future__ import annotations

import hmac
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from ._audit import GENESIS_HASH, entry_mac

__all__ = ["AuditIssue", "AuditVerification", "verify_audit_log"]

#: Kinds of :class:`AuditIssue` that mean the log was altered (or is not a
#: log this key signed).
ProblemKind = Literal[
    "wrong_key",
    "malformed",
    "unsigned",
    "modified",
    "inserted",
    "bad_signature",
    "duplicate",
    "reordered",
    "deleted",
    "missing_start",
    "anchor_missing",
]

#: Kinds of :class:`AuditIssue` that are not evidence of tampering.
WarningKind = Literal["truncated", "crash_fragment", "chain_reset"]


@dataclass(frozen=True)
class AuditIssue:
    """One finding of :func:`verify_audit_log`.

    Attributes:
        kind: What was found (see :class:`AuditVerification`).
        line: 1-based line number in ``path`` (``None`` for findings about
            the log as a whole, such as a missing anchor).
        path: The file the line is in.
        message: A human-readable explanation.
    """

    kind: str
    line: int | None
    path: str
    message: str

    def __str__(self) -> str:
        where = f"{self.path}:{self.line}" if self.line is not None else self.path
        return f"{where}: {self.kind}: {self.message}"


@dataclass(frozen=True)
class AuditVerification:
    """Result of :func:`verify_audit_log`.

    Problems (``ok`` is ``False``):

    * ``wrong_key`` — no entry verifies with the key(s) given.
    * ``malformed`` — a line that is not a JSON object.
    * ``unsigned`` — an entry without ``hmac`` / ``prev_hash``.
    * ``modified`` — an entry edited after it was signed (the next entry
      still links to it).
    * ``inserted`` — an entry that does not belong to the chain where it
      stands: forged, or copied from another log signed with the same key.
    * ``bad_signature`` — an entry whose signature fails where neither of
      the two above can be told apart (e.g. the last entry).
    * ``duplicate`` — a second copy of an entry earlier in the log.
    * ``reordered`` — an entry out of chain order.
    * ``deleted`` — entries missing before this one.
    * ``missing_start`` — the file does not begin a chain: its first entries
      were deleted (or the file continues a rotated one: verify the files
      together, in order).
    * ``anchor_missing`` — the ``anchor`` hash is not in the log, so entries
      at or after it were removed.

    Warnings (``ok`` stays ``True``):

    * ``truncated`` — the last line is incomplete and has no newline: the
      process stopped during a write.
    * ``crash_fragment`` — such an incomplete line inside the log, after
      which the restarted server continued the chain from the entry before
      it.
    * ``chain_reset`` — a new chain starts inside the log (a restart that
      did not continue the chain, as servers before 1.3.0 did, or a log
      whose last line was unreadable).  The entries before it may have lost
      their tail undetectably.

    Attributes:
        paths: The files verified, in order.
        ok: ``True`` when no problem was found.
        entries: Number of entries (JSON object lines) read.
        problems: Every problem, in file order.
        warnings: Every warning, in file order.
        restarts: Lines where a restarted server continued the chain
            (``"resumed": true``) as ``(path, line)``.
        last_hash: ``hmac`` of the last entry in the chain; store it
            elsewhere and pass it as ``anchor`` later to detect deletion of
            entries at the end of the log, which a hash chain alone cannot
            show.
        last_seq: ``seq`` of that entry, when it has one.
    """

    paths: tuple[str, ...]
    ok: bool
    entries: int
    problems: tuple[AuditIssue, ...] = ()
    warnings: tuple[AuditIssue, ...] = ()
    restarts: tuple[tuple[str, int], ...] = ()
    last_hash: str | None = None
    last_seq: int | None = None

    @property
    def first_problem(self) -> AuditIssue | None:
        """The first problem in file order, or ``None``."""
        return self.problems[0] if self.problems else None

    @property
    def truncated(self) -> bool:
        """The log ends in an incomplete line (a crash during a write)."""
        return any(w.kind == "truncated" for w in self.warnings)

    @property
    def continuous(self) -> bool:
        """One unbroken chain from the first entry to the last, across restarts."""
        return self.ok and not any(w.kind == "chain_reset" for w in self.warnings)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (what ``promptise audit verify --json`` prints)."""

        def issue(i: AuditIssue) -> dict[str, Any]:
            return {"kind": i.kind, "path": i.path, "line": i.line, "message": i.message}

        return {
            "paths": list(self.paths),
            "ok": self.ok,
            "continuous": self.continuous,
            "truncated": self.truncated,
            "entries": self.entries,
            "problems": [issue(i) for i in self.problems],
            "warnings": [issue(i) for i in self.warnings],
            "restarts": [{"path": p, "line": n} for p, n in self.restarts],
            "last_hash": self.last_hash,
            "last_seq": self.last_seq,
        }


@dataclass
class _Line:
    path: str
    line: int
    unterminated: bool
    entry: dict[str, Any] | None = None
    error: str = ""
    hmac: str | None = None
    prev_hash: str | None = None
    seq: int | None = None
    resumed: bool = False
    sig_ok: bool = False
    raw: bytes = b""

    @property
    def signed(self) -> bool:
        return self.hmac is not None and self.prev_hash is not None


def _as_keys(key: str | bytes | Sequence[str | bytes]) -> list[bytes]:
    raw: list[str | bytes] = [key] if isinstance(key, (str, bytes)) else list(key)
    keys = [k.encode() if isinstance(k, str) else bytes(k) for k in raw]
    if not keys or any(not k for k in keys):
        raise ValueError("verify_audit_log: the key must not be empty")
    return keys


# Every entry AuditMiddleware writes starts with this (``timestamp`` is the
# first key), so a fragment cut off by a crash is a prefix of it or starts
# with it.
_ENTRY_START = b'{"timestamp": '


def _is_entry_prefix(raw: bytes) -> bool:
    return raw.startswith(_ENTRY_START) or (bool(raw) and _ENTRY_START.startswith(raw))


def _read_lines(path: str, keys: list[bytes]) -> list[_Line]:
    with open(path, "rb") as f:
        data = f.read()
    if not data:
        return []
    raw_lines = data.split(b"\n")
    ends_with_newline = raw_lines[-1] == b""
    if ends_with_newline:
        raw_lines.pop()
    out: list[_Line] = []
    for idx, raw in enumerate(raw_lines):
        last = idx == len(raw_lines) - 1
        rec = _Line(
            path=path,
            line=idx + 1,
            unterminated=last and not ends_with_newline,
            raw=raw,
        )
        try:
            obj = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            rec.error = f"not valid JSON ({exc})" if raw.strip() else "empty line"
            out.append(rec)
            continue
        if not isinstance(obj, dict):
            rec.error = f"not a JSON object (a {type(obj).__name__})"
            out.append(rec)
            continue
        rec.entry = obj
        mac, prev = obj.get("hmac"), obj.get("prev_hash")
        rec.hmac = mac if isinstance(mac, str) else None
        rec.prev_hash = prev if isinstance(prev, str) else None
        seq = obj.get("seq")
        rec.seq = seq if isinstance(seq, int) and not isinstance(seq, bool) else None
        rec.resumed = obj.get("resumed") is True
        if rec.hmac is not None:
            expected = [entry_mac(k, obj) for k in keys]
            rec.sig_ok = any(hmac.compare_digest(e, rec.hmac) for e in expected)
        out.append(rec)
    return out


def verify_audit_log(
    path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    key: str | bytes | Sequence[str | bytes],
    *,
    anchor: str | None = None,
) -> AuditVerification:
    """Verify the HMAC chain of an audit log file written by :class:`AuditMiddleware`.

    Checks every entry's signature and its link to the entry before it, and
    reports the first break with its line number and what kind of change it
    is (see :class:`AuditVerification` for the kinds).  A last line cut off
    by a crash is reported as the warning ``truncated``, not as tampering.

    Args:
        path: The log file, or several files verified as one chain in order
            (a log rotated while the server ran).
        key: The ``PROMPTISE_AUDIT_SECRET`` / ``hmac_secret`` the log was
            signed with.  After a key rotation that kept the same file, pass
            every key (``[new_key, old_key]``): each entry must verify with
            one of them.
        anchor: An ``hmac`` recorded earlier (e.g. a previous
            :attr:`AuditVerification.last_hash`).  If it is not in the log,
            entries were deleted from its end — which a hash chain cannot
            show on its own.

    Returns:
        An :class:`AuditVerification`; ``ok`` is ``False`` if the log was
        tampered with.

    Raises:
        OSError: A file cannot be read.
        ValueError: The key is empty.
    """
    keys = _as_keys(key)
    if isinstance(path, (str, os.PathLike)):
        paths = [os.fspath(path)]
    else:
        paths = [os.fspath(p) for p in path]
    records: list[_Line] = []
    for p in paths:
        records.extend(_read_lines(p, keys))

    entries = [r for r in records if r.entry is not None]
    problems: list[AuditIssue] = []
    warnings: list[AuditIssue] = []
    restarts: list[tuple[str, int]] = []

    def problem(kind: ProblemKind, rec: _Line, message: str) -> None:
        problems.append(AuditIssue(kind, rec.line, rec.path, message))

    def warn(kind: WarningKind, rec: _Line, message: str) -> None:
        warnings.append(AuditIssue(kind, rec.line, rec.path, message))

    def where(rec: _Line) -> str:
        return f"line {rec.line}" if rec.path == records[-1].path else f"{rec.path}:{rec.line}"

    signed_entries = [r for r in entries if r.signed]
    if signed_entries and not any(r.sig_ok for r in signed_entries):
        first = signed_entries[0]
        problem(
            "wrong_key",
            first,
            "no entry verifies with the key given: wrong key",
        )
        return AuditVerification(
            paths=tuple(paths),
            ok=False,
            entries=len(entries),
            problems=tuple(problems),
        )

    # First position of every hmac, for telling reorders from deletions.
    position: dict[str, int] = {}
    for i, r in enumerate(records):
        if r.hmac is not None and r.sig_ok:
            position.setdefault(r.hmac, i)

    prev_hash = GENESIS_HASH
    prev: _Line | None = None
    chained: set[str] = set()

    for i, rec in enumerate(records):
        nxt = records[i + 1] if i + 1 < len(records) else None

        if rec.entry is None:
            crash_shaped = rec.error.startswith("not valid JSON") and _is_entry_prefix(rec.raw)
            if crash_shaped and rec.unterminated and nxt is None:
                warn(
                    "truncated",
                    rec,
                    "the last line is incomplete (no newline): the process stopped "
                    "during a write. Not evidence of tampering.",
                )
            elif (
                crash_shaped
                and nxt is not None
                and nxt.sig_ok
                and nxt.resumed
                and nxt.prev_hash == prev_hash
            ):
                # The restarted server's first write closed the fragment with
                # a newline and continued the chain from the entry before it.
                warn(
                    "crash_fragment",
                    rec,
                    "an incomplete line left by a crash during a write; the restarted "
                    "server continued the chain from the entry before it.",
                )
            else:
                problem("malformed", rec, f"{rec.error}: a line was edited or inserted")
            continue

        if not rec.signed:
            problem("unsigned", rec, "entry has no hmac/prev_hash: it was not written signed")
            continue

        assert rec.hmac is not None
        if not rec.sig_ok:
            if nxt is not None and nxt.prev_hash == rec.hmac:
                problem(
                    "modified",
                    rec,
                    "the entry was changed after it was signed (its signature no "
                    "longer matches, the next entry still links to it)",
                )
                prev_hash, prev = rec.hmac, rec
            elif nxt is not None and nxt.prev_hash == prev_hash:
                problem(
                    "inserted",
                    rec,
                    "the entry does not verify and the chain skips it: it was inserted (forged)",
                )
            else:
                problem(
                    "bad_signature",
                    rec,
                    "the entry's signature does not match: it was modified or forged",
                )
                prev_hash, prev = rec.hmac, rec
            continue

        if rec.hmac in chained:
            dup_of = records[position[rec.hmac]]
            problem("duplicate", rec, f"a second copy of the entry at {where(dup_of)}")
            continue

        if rec.prev_hash == prev_hash:
            if rec.resumed:
                restarts.append((rec.path, rec.line))
        elif rec.prev_hash == GENESIS_HASH and prev is not None:
            warn(
                "chain_reset",
                rec,
                "a new chain starts here (a restart that did not continue the chain); "
                "deleting entries just before this line would not be detected",
            )
        elif rec.prev_hash is not None and position.get(rec.prev_hash, -1) > i:
            problem(
                "reordered",
                rec,
                f"out of order: the entry it follows is at {where(records[position[rec.prev_hash]])}",
            )
        elif rec.prev_hash is not None and rec.prev_hash in position:
            after = records[position[rec.prev_hash]]
            expected = f"{where(prev)}" if prev is not None else "the start of the chain"
            problem(
                "reordered",
                rec,
                f"out of order: it follows the entry at {where(after)}, not {expected}",
            )
        elif nxt is not None and nxt.sig_ok and nxt.prev_hash == prev_hash and prev is not None:
            problem(
                "inserted",
                rec,
                "the entry belongs to another chain (copied from another log signed "
                "with the same key); the chain skips it",
            )
            continue
        elif prev is None:
            count = f"{rec.seq} " if rec.seq else ""
            problem(
                "missing_start",
                rec,
                f"the log does not start a chain: the first {count}entries are missing "
                "(deleted, or this file continues a rotated one — verify the files "
                "together, in order)",
            )
        else:
            if rec.seq is not None and prev.seq is not None and rec.seq > prev.seq + 1:
                n = rec.seq - prev.seq - 1
                detail = (
                    f"{n} entr{'y' if n == 1 else 'ies'} deleted between "
                    f"{where(prev)} and this line (seq {prev.seq} -> {rec.seq})"
                )
            else:
                detail = f"entries deleted between {where(prev)} and this line"
            problem("deleted", rec, detail)

        chained.add(rec.hmac)
        prev_hash, prev = rec.hmac, rec

    if anchor is not None and anchor not in chained:
        problems.append(
            AuditIssue(
                "anchor_missing",
                None,
                paths[-1],
                "the anchor hash is not in the log: entries at or after it were removed",
            )
        )

    return AuditVerification(
        paths=tuple(paths),
        ok=not problems,
        entries=len(entries),
        problems=tuple(problems),
        warnings=tuple(warnings),
        restarts=tuple(restarts),
        last_hash=prev.hmac if prev is not None else None,
        last_seq=prev.seq if prev is not None else None,
    )
