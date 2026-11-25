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

_SCHEMA = "ci_test_shard_v1"
_MAX_SHARDS = 16
# Candidate four-way placement from measured #1416 run 45542520830:
# keep the costly module-scoped fixtures together, without changing the
# assignment or execution contract of other test modules or shard counts.
_MEASURED_MODULE_AFFINITY = {
    "tests/evaluation/test_dev_realdata_phase3.py": 1,
    "tests/evaluation/test_dev_realdata_study.py": 3,
    "tests/evaluation/test_forward_study_fs1.py": 3,
    "tests/evaluation/test_signal_diagnosis_run_scenarios.py": 0,
    "tests/tools/tier_d_nonrl/test_task18_native_artifact.py": 3,
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
    if count == 4:
        module, _, _ = item.partition("::")
        affinity = _MEASURED_MODULE_AFFINITY.get(module)
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


def verify_reports(
    directory: Path, *, count: int, head: str, expected_items: Sequence[str]
) -> int:
    """Require all disjoint completed IDs to reproduce every full collection."""
    _validate_count(count)
    _validate_head(head)
    files = sorted(directory.glob("*/core-shard.json"))
    if len(files) != count:
        raise ValueError("missing or extra shard receipts")
    indices: set[int] = set()
    all_items: list[str] = []
    fingerprints: set[tuple[int, str]] = set()
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
        self.phases: dict[str, list[tuple[str, str]]] = {}

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
        self.phases.setdefault(report.nodeid, []).append((report.when, report.outcome))

    def pytest_runtest_logfinish(self, nodeid: str) -> None:
        self.completed.append(nodeid)

    @staticmethod
    def _complete_phases(phases: list[tuple[str, str]]) -> bool:
        # A finish notification alone is not execution: setup-only/plan or
        # another protocol hook can finish a node without a call report.
        # Preserve legitimate pytest setup skips (including xfail run=False)
        # and call skips/xfails, while requiring actual teardown reports.
        if len(phases) not in (2, 3):
            return False
        if phases[-1] not in (("teardown", "passed"), ("teardown", "skipped")):
            return False
        if phases[0] == ("setup", "skipped"):
            return len(phases) == 2
        return (
            len(phases) == 3
            and phases[0] == ("setup", "passed")
            and phases[1] in (("call", "passed"), ("call", "skipped"))
        )

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if exitstatus != pytest.ExitCode.OK or session.exitstatus != pytest.ExitCode.OK:
            return
        if (
            not self.selected
            or sorted(self.completed) != self.selected
            or sorted(self.phases) != self.selected
            or any(not self._complete_phases(phases) for phases in self.phases.values())
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

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.items = [item.nodeid for item in session.items]


def collect_full_suite() -> list[str]:
    """Use ordinary pytest collection, without the shard selection plugin."""
    collector = _FullCollection()
    result = pytest.main(["--collect-only", "-qq", "tests"], plugins=[collector])
    if result != pytest.ExitCode.OK or not collector.items:
        raise ValueError("independent full collection failed or is empty")
    return _canonical_items(collector.items)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--head", required=True)
    args = parser.parse_args(argv)
    try:
        total = verify_reports(
            args.directory,
            count=args.count,
            head=args.head,
            expected_items=collect_full_suite(),
        )
    except ValueError as error:
        parser.exit(1, f"CI shard verification failed: {error}\n")
    print(
        f"Complete exact-HEAD collection verified: {total} test items in {args.count} shards"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
