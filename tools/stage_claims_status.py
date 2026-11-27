"""Read-only status report for single-use research stage claims.

The report lists which pre-registered stages were claimed or consumed, their
decision labels and due dates. It never emits performance data (P&L, returns,
Sharpe, drawdown, equity, positions): continuous performance between interim
stages would be peeking. Result files are read through an explicit allowlist;
every other field is dropped before any output is built.

This is orientation only. It is not an authorization gate and never writes to a
ledger directory.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import stat
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from trade_rl.artifacts.stage_claim import StageClaim

SCHEMA_VERSION = "stage_claims_status_v1"
NO_PERFORMANCE_NOTICE = (
    "This report contains no performance data. It lists stage claim state, "
    "decision labels, digests and dates only."
)

CLAIMED = "CLAIMED"
RESULT_RECORDED = "RESULT_RECORDED"
INVALID = "INVALID"
NOT_CLAIMED = "NOT_CLAIMED"

# Result-file fields that may be read. Everything else is discarded unread,
# including any metric-like key at any nesting depth.
RESULT_ALLOWLIST = ("stage", "label", "protocol_digest", "result_sha256")

_STAGE = re.compile(r"[A-Z][A-Z0-9_]{0,63}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_LABEL = re.compile(r"[A-Za-z0-9_.:-]{1,80}")
_LEGACY_CLAIM = re.compile(r"([A-Z][A-Z0-9_]{0,63})\.claim")
_LEGACY_RESULT = re.compile(r"((?:stage-)?[A-Z][A-Z0-9_]{0,63})\.result\.json")
_V1_CLAIM = re.compile(r"stage-([A-Z][A-Z0-9_]{0,63})\.claim")
_COMMENT_CLAIM = re.compile(r"comment-([1-9][0-9]{0,18})-([1-9][0-9]{0,18})\.claim")
_V1_SCHEMA = "single_use_stage_claim_v1"
_MAX_CLAIM_BYTES = 64 * 1024
_MAX_RESULT_BYTES = 8 * 1024 * 1024
_MAX_DEPTH = 4
_MAX_ENTRIES = 5000


@dataclass(slots=True)
class _Claim:
    stage: str
    protocol_digest: str
    commit: str
    comment_id: int
    claimed_at: str
    schema: str


@dataclass(slots=True)
class _Result:
    stage: str
    label: str
    protocol_digest: str
    result_sha256: str
    recorded_at: str


@dataclass(slots=True)
class _Group:
    location: str
    stage: str
    claim: _Claim | None = None
    result: _Result | None = None
    comments: list[tuple[str, int]] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _read_regular(path: Path, limit: int) -> bytes:
    """Read a regular, non-symlink file with a bounded size."""
    info = os.lstat(path)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("not a regular file")
    if info.st_size > limit:
        raise ValueError("file exceeds the size limit")
    with open(path, "rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        raise ValueError("file exceeds the size limit")
    return data


def _json_object(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError) as exc:
        raise ValueError("not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON document is not an object")
    return value


def _parse_claim(path: Path, expected_stage: str | None) -> _Claim:
    """Validate a claim marker, reusing StageClaim's field validation."""
    body = _json_object(_read_regular(path, _MAX_CLAIM_BYTES))
    schema = body.get("schema_version")
    if schema is None:
        # Frozen legacy format: authorization_comment_id and no repository id.
        keys = {
            "stage",
            "protocol_digest",
            "commit",
            "authorization_comment_id",
            "claimed_at",
        }
        comment_id = body.get("authorization_comment_id")
        repository_id = 1  # placeholder: legacy markers carry no repository id
        label = "legacy"
    elif schema == _V1_SCHEMA:
        keys = {
            "schema_version",
            "stage",
            "protocol_digest",
            "commit",
            "repository_id",
            "comment_id",
            "claimed_at",
        }
        comment_id = body.get("comment_id")
        repository_id = body.get("repository_id")  # type: ignore[assignment]
        label = _V1_SCHEMA
    else:
        raise ValueError("unknown claim schema")
    if set(body) != keys:
        raise ValueError("claim fields differ from the schema")
    stamp = body["claimed_at"]
    if type(stamp) is not str:
        raise ValueError("claim timestamp is not a string")
    try:
        claimed_at = datetime.fromisoformat(stamp)
        validated = StageClaim(
            stage=body["stage"],
            protocol_digest=body["protocol_digest"],
            commit=body["commit"],
            repository_id=repository_id,
            comment_id=comment_id,  # type: ignore[arg-type]
            claimed_at=claimed_at,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid claim field: {exc}") from exc
    if expected_stage is not None and validated.stage != expected_stage:
        raise ValueError("claim stage differs from its file name")
    return _Claim(
        validated.stage,
        validated.protocol_digest,
        validated.commit,
        validated.comment_id,
        _iso(validated.claimed_at),
        label,
    )


def _parse_result(path: Path, expected_stage: str) -> _Result:
    """Extract only allowlisted fields; every other key is never read out."""
    body = _json_object(_read_regular(path, _MAX_RESULT_BYTES))
    picked = {key: body.get(key) for key in RESULT_ALLOWLIST}
    stage, label = picked["stage"], picked["label"]
    protocol, digest = picked["protocol_digest"], picked["result_sha256"]
    if type(stage) is not str or not _STAGE.fullmatch(stage):
        raise ValueError("result stage is missing or malformed")
    if stage != expected_stage:
        raise ValueError("result stage differs from its file name")
    if type(label) is not str or not _LABEL.fullmatch(label):
        raise ValueError("result label is missing or malformed")
    if type(protocol) is not str or not _HEX64.fullmatch(protocol):
        raise ValueError("result protocol digest is missing or malformed")
    if type(digest) is not str or not _HEX64.fullmatch(digest):
        raise ValueError("result digest is missing or malformed")
    recorded = datetime.fromtimestamp(os.lstat(path).st_mtime, UTC)
    return _Result(stage, label, protocol, digest, _iso(recorded))


def _scan(root: Path) -> tuple[list[_Group], list[dict[str, Any]]]:
    """Return per-(directory, stage) groups and unattributable invalid files."""
    groups: dict[tuple[str, str], _Group] = {}
    loose: list[dict[str, Any]] = []
    seen = 0

    def group(location: str, stage: str) -> _Group:
        key = (location, stage)
        if key not in groups:
            groups[key] = _Group(location, stage)
        return groups[key]

    def walk(directory: Path, location: str, depth: int) -> None:
        nonlocal seen
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            loose.append(_loose(location, "", "directory cannot be listed"))
            return
        for name in names:
            seen += 1
            if seen > _MAX_ENTRIES:
                if seen == _MAX_ENTRIES + 1:
                    loose.append(_loose(location, "", "entry limit exceeded"))
                return
            path = directory / name
            try:
                info = os.lstat(path)
            except OSError:
                loose.append(_loose(location, name, "file cannot be inspected"))
                continue
            if stat.S_ISLNK(info.st_mode):
                loose.append(_loose(location, name, "symlink is not followed"))
            elif stat.S_ISDIR(info.st_mode):
                if depth >= _MAX_DEPTH:
                    loose.append(_loose(location, name, "directory depth limit"))
                else:
                    child = f"{location}/{name}" if location else name
                    walk(path, child, depth + 1)
            elif stat.S_ISREG(info.st_mode):
                _classify(path, name, location)
            else:
                loose.append(_loose(location, name, "not a regular file"))

    def _classify(path: Path, name: str, location: str) -> None:
        v1 = _V1_CLAIM.fullmatch(name)
        legacy = _LEGACY_CLAIM.fullmatch(name)
        comment = _COMMENT_CLAIM.fullmatch(name)
        result = _LEGACY_RESULT.fullmatch(name)
        try:
            if v1 or legacy:
                stage = (v1 or legacy).group(1)  # type: ignore[union-attr]
                entry = group(location, stage)
                entry.files.append(name)
                claim = _parse_claim(path, stage)
                if (v1 is not None) != (claim.schema == _V1_SCHEMA):
                    raise ValueError("claim schema differs from its file name")
                entry.claim = claim
            elif comment:
                claim = _parse_claim(path, None)
                if claim.schema != _V1_SCHEMA:
                    raise ValueError("comment marker has the wrong schema")
                if f"{comment.group(2)}" != str(claim.comment_id):
                    raise ValueError("comment marker id differs from its name")
                entry = group(location, claim.stage)
                entry.files.append(name)
                entry.comments.append((name, claim.comment_id))
            elif result:
                raw = result.group(1)
                stage = raw.removeprefix("stage-")
                entry = group(location, stage)
                entry.files.append(name)
                entry.result = _parse_result(path, stage)
            else:
                loose.append(_loose(location, name, "foreign file"))
        except (OSError, ValueError) as exc:
            reason = str(exc) or "unreadable file"
            stage_name = _stage_from_name(name)
            if stage_name is None:
                loose.append(_loose(location, name, reason))
            else:
                group(location, stage_name).problems.append(f"{name}: {reason}")

    walk(root, "", 0)
    return [groups[key] for key in sorted(groups)], loose


def _stage_from_name(name: str) -> str | None:
    for pattern in (_V1_CLAIM, _LEGACY_CLAIM):
        match = pattern.fullmatch(name)
        if match:
            return match.group(1)
    match = _LEGACY_RESULT.fullmatch(name)
    if match:
        return match.group(1).removeprefix("stage-")
    return None


def _loose(location: str, name: str, reason: str) -> dict[str, Any]:
    return {
        "location": location,
        "stage": None,
        "state": INVALID,
        "file": name,
        "reason": reason,
    }


def _entry_for(item: _Group) -> dict[str, Any]:
    claim, result = item.claim, item.result
    problems = list(item.problems)
    namespace = item.location.rsplit("/", 1)[-1]
    if claim is None and not problems:
        problems.append("result or comment marker without a stage claim")
    if claim is not None and result is not None:
        if result.protocol_digest != claim.protocol_digest:
            problems.append("result protocol digest differs from the claim")
    if claim is not None and _HEX64.fullmatch(namespace):
        if claim.protocol_digest != namespace:
            problems.append("claim digest differs from its digest-named directory")
    if claim is not None:
        for name, comment_id in item.comments:
            if comment_id != claim.comment_id:
                problems.append(f"{name}: comment id differs from the stage claim")
    entry: dict[str, Any] = {
        "location": item.location,
        "stage": item.stage,
        "state": INVALID,
        "protocol_digest": claim.protocol_digest if claim else None,
        "commit": claim.commit if claim else None,
        "comment_id": claim.comment_id if claim else None,
        "claimed_at": claim.claimed_at if claim else None,
        "result_recorded_at": None,
        "result_label": None,
        "result_sha256": None,
        "files": sorted(item.files),
        "reason": "; ".join(problems) if problems else None,
    }
    if problems:
        return entry
    entry["state"] = CLAIMED if result is None else RESULT_RECORDED
    if result is not None:
        entry["result_recorded_at"] = result.recorded_at
        entry["result_label"] = result.label
        entry["result_sha256"] = result.result_sha256
    return entry


def _apply_schedule(
    entries: list[dict[str, Any]],
    schedule: dict[str, dict[str, date]],
    today: date,
) -> list[str]:
    """Annotate entries with due dates; add NOT_CLAIMED rows. Return matched keys."""
    matched: set[str] = set()
    claimed: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in entries:
        digest = entry.get("protocol_digest")
        if digest in schedule:
            matched.add(digest)
            claimed[(digest, entry["stage"])] = entry
    for entry in entries:
        entry.setdefault("earliest_allowed", None)
        entry.setdefault("days_until_due", None)
        entry.setdefault("schedule_status", None)
    for digest in sorted(matched):
        locations = sorted(
            {e["location"] for e in entries if e.get("protocol_digest") == digest}
        )
        for stage, due in sorted(schedule[digest].items()):
            existing = claimed.get((digest, stage))
            if existing is not None:
                existing["earliest_allowed"] = due.isoformat()
                continue
            days = (due - today).days
            entries.append(
                {
                    "location": locations[0],
                    "stage": stage,
                    "state": NOT_CLAIMED,
                    "protocol_digest": digest,
                    "commit": None,
                    "comment_id": None,
                    "claimed_at": None,
                    "result_recorded_at": None,
                    "result_label": None,
                    "result_sha256": None,
                    "files": [],
                    "reason": None,
                    "earliest_allowed": due.isoformat(),
                    "days_until_due": days,
                    "schedule_status": (
                        "NOT_YET_DUE"
                        if days > 0
                        else "DUE_TODAY"
                        if days == 0
                        else "OVERDUE"
                    ),
                }
            )
    return sorted(matched)


def _sort_key(entry: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        entry["location"],
        entry["stage"] or "",
        entry.get("file") or "",
        entry["state"],
    )


def build_report(
    ledgers: Sequence[Path],
    schedule: dict[str, dict[str, date]] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    """Build the allowlisted status report. `today` is injectable for tests."""
    scheduled = schedule or {}
    reference = today if today is not None else datetime.now(UTC).date()
    out: list[dict[str, Any]] = []
    matched: set[str] = set()
    for ledger in ledgers:
        try:
            ok = stat.S_ISDIR(os.lstat(ledger).st_mode)
        except OSError:
            ok = False
        if not ok:
            out.append(
                {
                    "path": str(ledger),
                    "error": "ledger is not a readable real directory",
                    "entries": [],
                }
            )
            continue
        groups, loose = _scan(ledger)
        entries = [_entry_for(item) for item in groups]
        entries.extend(loose)
        if scheduled:
            matched.update(_apply_schedule(entries, scheduled, reference))
        for entry in entries:
            entry.setdefault("earliest_allowed", None)
            entry.setdefault("days_until_due", None)
            entry.setdefault("schedule_status", None)
            entry.setdefault("protocol_digest", None)
            entry.setdefault("commit", None)
            entry.setdefault("comment_id", None)
            entry.setdefault("claimed_at", None)
            entry.setdefault("result_recorded_at", None)
            entry.setdefault("result_label", None)
            entry.setdefault("result_sha256", None)
            entry.setdefault("files", [])
            entry.setdefault("reason", None)
            entry.setdefault("file", None)
        entries.sort(key=_sort_key)
        out.append({"path": str(ledger), "error": None, "entries": entries})
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "notice": NO_PERFORMANCE_NOTICE,
        "as_of": reference.isoformat() if scheduled else None,
        "ledgers": out,
        "schedule_unmatched_protocols": sorted(set(scheduled) - matched),
    }
    return report


def render_json(report: dict[str, Any]) -> str:
    return json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def _short(value: str | None) -> str:
    return value[:12] if value else "-"


def _due_text(entry: dict[str, Any]) -> str:
    if entry["earliest_allowed"] is None:
        return "-"
    days = entry["days_until_due"]
    if days is None:
        return entry["earliest_allowed"]
    if days < 0:
        return f"{entry['earliest_allowed']} (overdue {-days}d)"
    return f"{entry['earliest_allowed']} (in {days}d)"


_COLUMNS = (
    ("STAGE", lambda e: e["stage"] or "-"),
    ("STATE", lambda e: e["state"]),
    ("PROTOCOL", lambda e: _short(e["protocol_digest"])),
    ("CLAIMED_AT", lambda e: e["claimed_at"] or "-"),
    ("RESULT_AT", lambda e: e["result_recorded_at"] or "-"),
    ("LABEL", lambda e: e["result_label"] or "-"),
    ("DUE", _due_text),
)


def _cells(entry: dict[str, Any]) -> list[str]:
    cells = []
    for _, getter in _COLUMNS:
        cells.append(_clean(getter(entry)))
    return cells


def _clean(text: str) -> str:
    return "".join(ch if ch.isprintable() else "?" for ch in str(text))


def render_text(report: dict[str, Any]) -> str:
    lines = [f"NOTICE: {report['notice']}"]
    for ledger in report["ledgers"]:
        lines.append("")
        lines.append(f"Ledger: {_clean(ledger['path'])}")
        if ledger["error"]:
            lines.append(f"  ERROR: {ledger['error']}")
            continue
        entries = ledger["entries"]
        if not entries:
            lines.append("  (no stage claim files found)")
            continue
        header = ["LOCATION", *[name for name, _ in _COLUMNS]]
        rows = [[_clean(e["location"] or ".")] + _cells(e) for e in entries]
        widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
        for row in [header, *rows]:
            lines.append(
                "  " + "  ".join(cell.ljust(width) for cell, width in zip(row, widths))
            )
        for entry in entries:
            if entry["state"] == INVALID:
                name = entry["stage"] or entry["file"] or "?"
                where = entry["location"] or "."
                lines.append(
                    f"  INVALID {_clean(where)}/{_clean(name)}: "
                    f"{_clean(entry['reason'] or '')}"
                )
        digests = sorted(
            {e["protocol_digest"] for e in entries if e["protocol_digest"]}
        )
        for digest in digests:
            lines.append(f"  protocol {digest}")
        for entry in entries:
            if entry["result_sha256"]:
                lines.append(
                    f"  result {_clean(entry['stage'])} sha256 {entry['result_sha256']}"
                )
    if report["schedule_unmatched_protocols"]:
        lines.append("")
        lines.append(
            "Schedule protocols not found in any ledger: "
            + ", ".join(report["schedule_unmatched_protocols"])
        )
    lines.append("")
    lines.append(f"NOTICE: {report['notice']}")
    return "\n".join(lines) + "\n"


_CSS = (
    "body{font-family:system-ui,sans-serif;margin:1.5rem;color:#111;background:#fff}"
    "table{border-collapse:collapse;margin:.5rem 0 1.5rem}"
    "th,td{border:1px solid #888;padding:.25rem .5rem;text-align:left;"
    "font-size:.9rem}th{background:#eee}code{font-size:.85rem}"
    ".notice{border:2px solid #333;padding:.5rem;font-weight:bold}"
    ".INVALID{background:#fdd}.CLAIMED{background:#ffd}"
    ".RESULT_RECORDED{background:#dfd}.NOT_CLAIMED{background:#eef}"
)


def render_html(report: dict[str, Any]) -> str:
    esc = html.escape
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta http-equiv="Content-Security-Policy" '
        "content=\"default-src 'none'; style-src 'unsafe-inline'\">",
        "<title>Stage claims status</title>",
        f"<style>{_CSS}</style></head><body>",
        "<h1>Stage claims status</h1>",
        f'<p class="notice">{esc(report["notice"])}</p>',
    ]
    for ledger in report["ledgers"]:
        parts.append(f"<h2>Ledger: <code>{esc(_clean(ledger['path']))}</code></h2>")
        if ledger["error"]:
            parts.append(f"<p>ERROR: {esc(ledger['error'])}</p>")
            continue
        if not ledger["entries"]:
            parts.append("<p>No stage claim files found.</p>")
            continue
        parts.append("<table><tr><th>LOCATION</th>")
        parts.extend(f"<th>{esc(name)}</th>" for name, _ in _COLUMNS)
        parts.append("<th>RESULT_SHA256</th><th>NOTE</th></tr>")
        for entry in ledger["entries"]:
            cells = [_clean(entry["location"] or "."), *_cells(entry)]
            cells.append(_clean(entry["result_sha256"] or "-"))
            note = entry["reason"] or ""
            if entry["file"]:
                note = f"{entry['file']}: {note}"
            cells.append(_clean(note))
            body = "".join(f"<td>{esc(c)}</td>" for c in cells)
            parts.append(f'<tr class="{esc(entry["state"])}">{body}</tr>')
        parts.append("</table>")
        digests = sorted(
            {e["protocol_digest"] for e in ledger["entries"] if e["protocol_digest"]}
        )
        for digest in digests:
            parts.append(f"<p>Protocol digest: <code>{esc(digest)}</code></p>")
    if report["schedule_unmatched_protocols"]:
        parts.append(
            "<p>Schedule protocols not found in any ledger: "
            + esc(", ".join(report["schedule_unmatched_protocols"]))
            + "</p>"
        )
    parts.append(f"<p>{esc(report['notice'])}</p></body></html>")
    return "\n".join(parts) + "\n"


def write_html(path: Path, content: str, *, replace: bool) -> None:
    """Write the page with exclusive create; `replace` swaps atomically."""
    data = content.encode("utf-8")
    if not replace:
        with open(path, "xb") as handle:
            handle.write(data)
        return
    directory = path.resolve().parent
    fd, temporary = tempfile.mkstemp(prefix=".stage-status-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def load_schedule(path: Path) -> dict[str, dict[str, date]]:
    """Parse {protocol_digest: {STAGE: 'YYYY-MM-DD'}}; reject anything else."""
    body = _json_object(_read_regular(path, _MAX_CLAIM_BYTES))
    schedule: dict[str, dict[str, date]] = {}
    for digest, stages in body.items():
        if not _HEX64.fullmatch(digest) or not isinstance(stages, dict):
            raise ValueError("schedule protocol key must be a 64-hex digest object")
        schedule[digest] = {}
        for stage, due in stages.items():
            if not _STAGE.fullmatch(stage) or type(due) is not str:
                raise ValueError("schedule stage must map a stage name to a date")
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", due):
                raise ValueError("schedule dates must be YYYY-MM-DD")
            schedule[digest][stage] = date.fromisoformat(due)
    return schedule


def main(
    argv: Sequence[str] | None = None,
    *,
    today: Callable[[], date] | None = None,
) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.stage_claims_status",
        description="Read-only stage claim status. " + NO_PERFORMANCE_NOTICE,
    )
    parser.add_argument("ledgers", nargs="+", type=Path, metavar="LEDGER_DIR")
    parser.add_argument("--schedule", type=Path, help="JSON: protocol -> stage -> date")
    parser.add_argument("--json", action="store_true", help="deterministic JSON")
    parser.add_argument("--html", type=Path, help="write a static HTML page")
    parser.add_argument("--replace", action="store_true", help="overwrite --html")
    parser.add_argument("--as-of", help="override today's date (YYYY-MM-DD)")
    args = parser.parse_args(argv)
    schedule: dict[str, dict[str, date]] | None = None
    try:
        if args.schedule is not None:
            schedule = load_schedule(args.schedule)
        if args.as_of is not None:
            reference = date.fromisoformat(args.as_of)
        else:
            reference = (today or (lambda: datetime.now(UTC).date()))()
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = build_report(args.ledgers, schedule, reference)
    if args.html is not None:
        try:
            write_html(args.html, render_html(report), replace=args.replace)
        except OSError as exc:
            print(f"error: cannot write HTML: {exc}", file=sys.stderr)
            return 2
    sys.stdout.write(render_json(report) if args.json else render_text(report))
    return 1 if any(item["error"] for item in report["ledgers"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
