"""Partition one complete pytest collection and verify all CI shard receipts.

Opt-in plugin only: ordinary local pytest remains unpartitioned. Successful
receipt publication requires every selected node to finish and pytest to pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

_SCHEMA = "ci_test_shard_v2"
_MAX_SHARDS = 16
# Four-way placement from #1416 run 45542520830 and #1420 run 45553392392:
# keep the costly module-scoped fixtures together, without changing the
# assignment or execution contract of other test modules or shard counts.
# The first affinity run concentrated three expensive modules on shard 3;
# move study and FS1 to shards 0 and 2 to reduce its observed critical path.
# Run 45744299892 then measured shard 1 at 1309s versus shard 3 at 588s;
# move the intact phase3 module (at least 378s of reported durations) to 3.
_MEASURED_MODULE_AFFINITY = {
    "tests/tools/tier_d_nonrl/test_task18_native_artifact.py": 3,
}


# Eight-way placement fitted to the 200 slow-duration entries in runs
# 45744299892 and 45744743567, cross-checked against 45746974291. Keep
# expensive shared fixtures together; preserve the established four-way map.
_MEASURED_EIGHT_WAY_AFFINITY = {
    "tests/tools/tier_d_nonrl/test_task18_native_recovery.py": 2,
    "tests/tools/tier_d_nonrl/test_task18_native_artifact.py": 1,
}


def _validate_count(count: int) -> None:
    if type(count) is not int or not 1 <= count <= _MAX_SHARDS:
        raise ValueError("invalid shard count")


def _validate_head(head: str) -> None:
    if (
        not isinstance(head, str)
        or not re.fullmatch(r"[0-9a-f]{40}", head)
        or head == "0" * 40
    ):
        raise ValueError("invalid exact HEAD")


def _canonical_items(items: Sequence[str]) -> list[str]:
    if any(type(item) is not str or not item for item in items):
        raise ValueError("invalid pytest node ID")
    if len(set(items)) != len(items):
        raise ValueError("duplicate pytest node ID")
    return sorted(items)


def collection_digest(items: Sequence[str]) -> str:
    # JSON preserves node-ID boundaries even for parameter IDs containing newlines.
    payload = json.dumps(
        _canonical_items(items), ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def shard_index(item: str, count: int) -> int:
    _validate_count(count)
    if type(item) is not str or not item:
        raise ValueError("invalid pytest node ID")
    if count in (4, 8):
        module, _, _ = item.partition("::")
        placement = (
            _MEASURED_MODULE_AFFINITY if count == 4 else _MEASURED_EIGHT_WAY_AFFINITY
        )
        affinity = placement.get(module)
        if affinity is not None:
            return affinity
    return (
        int.from_bytes(
            hashlib.sha256(item.encode("utf-8", errors="surrogatepass")).digest()[:8],
            "big",
        )
        % count
    )


def _unique_receipt_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate shard receipt key")
        result[key] = value
    return result


def _skip_reason(report: pytest.TestReport | pytest.CollectReport) -> str:
    # Pytest's skip representation carries location and reason separately.
    # Do not persist arbitrary tracebacks or captured output.
    value = report.longrepr
    if not isinstance(value, tuple) or len(value) != 3 or type(value[2]) is not str:
        raise ValueError("missing pytest skip reason")
    return value[2]


def _phase_report(report: pytest.TestReport) -> tuple[str, str, str | None]:
    status: str = report.outcome
    reason: str | None = None
    xfail = getattr(report, "wasxfail", None)
    if xfail is not None and status in ("passed", "skipped"):
        status = "xpassed" if status == "passed" else "xfailed"
        reason = xfail
    elif status == "skipped":
        reason = _skip_reason(report)
    return report.when, status, reason


def _complete_phases(value: object) -> bool:
    # A finish notification alone is not execution. Preserve actual skip,
    # xfail and non-strict xpass outcomes rather than describing them as passes.
    if type(value) is not list or len(value) not in (2, 3):
        return False
    phases: list[tuple[str, str]] = []
    for row in value:
        if type(row) not in (list, tuple) or len(row) != 3:
            return False
        phase, status, reason = row
        if type(phase) is not str or type(status) is not str:
            return False
        if status == "passed":
            if reason is not None:
                return False
        elif status not in ("skipped", "xfailed", "xpassed") or type(reason) is not str:
            return False
        phases.append(
            (phase, {"xfailed": "skipped", "xpassed": "passed"}.get(status, status))
        )
    if phases[-1] not in (("teardown", "passed"), ("teardown", "skipped")):
        return False
    if phases[0] == ("setup", "skipped"):
        return len(phases) == 2
    return (
        len(phases) == 3
        and phases[0] == ("setup", "passed")
        and phases[1] in (("call", "passed"), ("call", "skipped"))
    )


def _collection_skip_roster(value: object) -> list[tuple[str, str]]:
    if type(value) is not list:
        raise ValueError("invalid collection skip roster")
    rows: list[tuple[str, str]] = []
    for row in value:
        if (
            type(row) not in (tuple, list)
            or len(row) != 2
            or type(row[0]) is not str
            or not row[0]
            or type(row[1]) is not str
        ):
            raise ValueError("invalid collection skip entry")
        rows.append((row[0], row[1]))
    if len({nodeid for nodeid, _ in rows}) != len(rows):
        raise ValueError("duplicate collection skip node ID")
    return sorted(rows)


def verify_reports(
    directory: Path,
    *,
    count: int,
    head: str,
    expected_items: Sequence[str],
    expected_collection_skips: Sequence[tuple[str, str]] = (),
) -> int:
    """Verify disjoint completion, phase evidence and independent collection skips."""
    _validate_count(count)
    _validate_head(head)
    files = sorted(directory.glob("*/core-shard.json"))
    if len(files) != count:
        raise ValueError("missing or extra shard receipts")
    indices: set[int] = set()
    all_items: list[str] = []
    fingerprints: set[tuple[int, str]] = set()
    expected_skips = _collection_skip_roster(list(expected_collection_skips))
    for file in files:
        try:
            record = json.loads(
                file.read_text(encoding="utf-8"),
                object_pairs_hook=_unique_receipt_object,
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError("unreadable shard receipt") from error
        if type(record) is not dict or set(record) != {
            "schema",
            "head",
            "index",
            "count",
            "collection_count",
            "collection_digest",
            "completed",
            "phase_reports",
            "collection_skips",
        }:
            raise ValueError("invalid shard receipt schema")
        index, declared_count = record["index"], record["count"]
        size, digest, completed = (
            record["collection_count"],
            record["collection_digest"],
            record["completed"],
        )
        if (
            record["schema"] != _SCHEMA
            or record["head"] != head
            or type(declared_count) is not int
            or declared_count != count
            or type(index) is not int
            or not 0 <= index < count
            or index in indices
            or type(size) is not int
            or size <= 0
            or type(digest) is not str
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or type(completed) is not list
            or not completed
        ):
            raise ValueError("invalid shard identity or collection")
        items = _canonical_items(completed)
        if any(shard_index(item, count) != index for item in items):
            raise ValueError("item assigned to wrong shard")
        reports = record["phase_reports"]
        if (
            type(reports) is not dict
            or set(reports) != set(items)
            or any(not _complete_phases(phases) for phases in reports.values())
        ):
            raise ValueError("invalid or incomplete phase reports")
        if _collection_skip_roster(record["collection_skips"]) != expected_skips:
            raise ValueError("collection skip roster differs from independent collection")
        indices.add(index)
        fingerprints.add((size, digest))
        all_items.extend(items)
    if indices != set(range(count)) or len(fingerprints) != 1:
        raise ValueError("shard roster or full collection differs")
    if fingerprints != {(len(all_items), collection_digest(all_items))}:
        raise ValueError("completed items do not cover the full collection")
    if _canonical_items(all_items) != _canonical_items(expected_items):
        raise ValueError("shards differ from independent full collection")
    return len(all_items)


class _ShardRun:
    def __init__(self, index: int, count: int, head: str, report: Path) -> None:
        self.index, self.count, self.head, self.report = index, count, head, report
        self.full: list[str] = []
        self.selected: list[str] = []
        self.completed: list[str] = []
        self.phases: dict[str, list[tuple[str, str, str | None]]] = {}
        self.collection_skips: list[tuple[str, str]] = []

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.skipped:
            self.collection_skips.append((report.nodeid, _skip_reason(report)))

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(
        self, config: pytest.Config, items: list[pytest.Item]
    ) -> None:
        self.full = _canonical_items([item.nodeid for item in items])
        keep: list[pytest.Item] = []
        drop: list[pytest.Item] = []
        for item in items:
            (
                keep if shard_index(item.nodeid, self.count) == self.index else drop
            ).append(item)
        self.selected = sorted(item.nodeid for item in keep)
        items[:] = keep
        config.hook.pytest_deselected(items=drop)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        self.phases.setdefault(report.nodeid, []).append(_phase_report(report))

    def pytest_runtest_logfinish(self, nodeid: str) -> None:
        self.completed.append(nodeid)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if exitstatus != pytest.ExitCode.OK or session.exitstatus != pytest.ExitCode.OK:
            return
        if (
            not self.selected
            or sorted(self.completed) != self.selected
            or sorted(self.phases) != self.selected
            or any(not _complete_phases(phases) for phases in self.phases.values())
        ):
            print(
                "CI shard failed: not every selected test completed its execution phases",
                file=sys.stderr,
            )
            session.exitstatus = pytest.ExitCode.TESTS_FAILED
            return
        record = {
            "schema": _SCHEMA,
            "head": self.head,
            "index": self.index,
            "count": self.count,
            "collection_count": len(self.full),
            "collection_digest": collection_digest(self.full),
            "completed": self.selected,
            "phase_reports": self.phases,
            "collection_skips": _collection_skip_roster(self.collection_skips),
        }
        # Never overwrite a receipt from another execution.
        with self.report.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, ensure_ascii=True, separators=(",", ":"))


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("ci-test-shard")
    group.addoption("--ci-shard-index", type=int)
    group.addoption("--ci-shard-count", type=int)
    group.addoption("--ci-shard-head")
    group.addoption("--ci-shard-report")


def pytest_configure(config: pytest.Config) -> None:
    index, count = (
        config.getoption("ci_shard_index"),
        config.getoption("ci_shard_count"),
    )
    head, report = (
        config.getoption("ci_shard_head"),
        config.getoption("ci_shard_report"),
    )
    try:
        _validate_count(count)
        _validate_head(head)
        if type(index) is not int or not 0 <= index < count or not report:
            raise ValueError("invalid shard index or report")
        if (
            config.getoption("collectonly")
            or config.getoption("setuponly")
            or config.getoption("setupplan")
            or config.getoption("keyword")
            or config.getoption("markexpr")
        ):
            raise ValueError(
                "shard execution cannot filter or use nonexecuting collection/setup modes"
            )
        path = Path(report)
        if path.exists():
            raise ValueError("shard report already exists")
    except ValueError as error:
        raise pytest.UsageError(str(error)) from error
    config.pluginmanager.register(
        _ShardRun(index, count, head, path), "ci-test-shard-run"
    )


class _FullCollection:
    def __init__(self) -> None:
        self.items: list[str] = []
        self.collection_skips: list[tuple[str, str]] = []

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.skipped:
            self.collection_skips.append((report.nodeid, _skip_reason(report)))

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.items = [item.nodeid for item in session.items]


def collect_full_suite() -> tuple[list[str], list[tuple[str, str]]]:
    """Use ordinary pytest collection, without the shard selection plugin."""
    collector = _FullCollection()
    result = pytest.main(["--collect-only", "-qq", "tests"], plugins=[collector])
    if result != pytest.ExitCode.OK or not collector.items:
        raise ValueError("independent full collection failed or is empty")
    return _canonical_items(collector.items), _collection_skip_roster(
        collector.collection_skips
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--head", required=True)
    args = parser.parse_args(argv)
    try:
        items, collection_skips = collect_full_suite()
        total = verify_reports(
            args.directory,
            count=args.count,
            head=args.head,
            expected_items=items,
            expected_collection_skips=collection_skips,
        )
    except ValueError as error:
        parser.exit(1, f"CI shard verification failed: {error}\n")
    print(
        f"Complete exact-HEAD collection verified: {total} test items in {args.count} shards"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
