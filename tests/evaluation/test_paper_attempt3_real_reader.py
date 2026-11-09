"""Attempt-3 gates against the real v1 journal reader (a synthetic predecessor root).

The fake readers of ``test_paper_attempt3.py`` pin the interface; this file uses the
real read-only journal reader over a tiny synthetic root built in a tmp dir, so the
identity read, the failure-marker check and the collector-entry re-verification are
exercised end to end. No attempt-1 or attempt-2 root is read.
"""

from __future__ import annotations

from collections import namedtuple
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trade_rl.evaluation.forward_study.seal import CodeIdentity
from trade_rl.evaluation.paper import predecessor as pred
from trade_rl.evaluation.paper.attempt3 import (
    Attempt3Refusal,
    launch_attempt3,
    plan_attempt3,
    resolve_lineage,
    seal_attempt3,
)
from trade_rl.evaluation.paper.store import PaperJournal

AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-4242424242"
NOW = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
START = NOW + timedelta(hours=2)
OBSERVED = "2026-11-07T00:59:00.123456+00:00"
CLEAN = CodeIdentity("a" * 40, True)
Usage = namedtuple("Usage", "total used free")


def synthetic_predecessor(tmp_path: Path, *, events: int = 3) -> tuple[Path, str, str]:
    root = tmp_path / "attempt2"
    digest = PaperJournal.initialize(root, {"engine": "synthetic"})
    journal = PaperJournal(root, expected_protocol_sha256=digest)
    parent = digest
    for index in range(events):
        record = journal.append(
            f"k{index}",
            "decide",
            {"request": {}, "result": {"observed_at": OBSERVED}},
            expected_parent=parent,
        )
        parent = record["sha256"]
    (root / "collection-failure.json").write_text("{}")
    return root, digest, parent


def snapshot(root: Path) -> dict[str, tuple[int, int, bytes]]:
    return {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns, path.read_bytes())
        for path in root.iterdir()
        if path.is_file()
    }


def test_the_real_reader_builds_lineage_v4_without_touching_the_root(tmp_path):
    root, digest, tip = synthetic_predecessor(tmp_path)
    before = snapshot(root)

    lineage, reasons = resolve_lineage(root, digest)

    assert reasons == []
    assert lineage is not None
    proof = lineage["predecessor_root"]["verification"]
    assert proof["event_count"] == 3 and proof["final_tip_sha256"] == tip
    assert proof["last_observed_at"] == OBSERVED
    assert lineage["predecessor"]["protocol_sha256"] == digest
    assert pred.lineage_structure_problems(lineage) == []
    assert snapshot(root) == before  # neither written nor created


def test_the_real_reader_refuses_a_wrong_digest_a_missing_marker_and_a_drifted_root(
    tmp_path,
):
    root, digest, _ = synthetic_predecessor(tmp_path)
    lineage, _ = resolve_lineage(root, digest)
    assert lineage is not None

    wrong, reasons = resolve_lineage(root, "f" * 64)
    assert wrong is None and reasons == ["predecessor_root_unreadable:ValueError"]

    # Collector-entry re-verification: a root that gained an event no longer matches.
    journal = PaperJournal(root, expected_protocol_sha256=digest)
    tip = journal.events()[-1]["sha256"]
    journal.append(
        "late",
        "decide",
        {"request": {}, "result": {"observed_at": OBSERVED}},
        expected_parent=tip,
    )
    drifted = pred.available_root_problems(lineage)
    assert any("event_count_differs" in item for item in drifted)

    (root / "collection-failure.json").unlink()
    missing = pred.available_root_problems(lineage)
    assert any("failure_marker_missing" in item for item in missing)


def test_plan_seal_and_launch_use_the_real_reader_end_to_end(tmp_path):
    root, digest, _ = synthetic_predecessor(tmp_path)
    target = tmp_path / "attempt3"
    options = dict(
        start_at=START,
        authorization=AUTH,
        predecessor_root=root,
        predecessor_protocol_sha256=digest,
        clock=lambda: NOW,
        code_identity=lambda: CLEAN,
        disk_usage=lambda _p: Usage(10**13, 0, 10**13),
    )
    draft = plan_attempt3(target, bytes_per_observation=2_000_000, **options)
    assert draft["seal_blockers"] == []
    sealed = seal_attempt3(target, bytes_per_observation=2_000_000, **options)
    assert sealed["protocol_sha256"] not in {digest, pred.ATTEMPT_ONE_PROTOCOL_SHA256}

    calls: list[str] = []

    def launch(**overrides):
        values = dict(
            protocol_sha256=sealed["protocol_sha256"],
            start_at=START,
            authorization=AUTH,
            allow_network=True,
            log_file=tmp_path / "child.log",
            clock=lambda: NOW,
            code_identity=lambda: CLEAN,
            launcher=lambda *a, **k: lambda: pytest.fail("no child"),
            supervise=lambda *a, **k: calls.append("supervise") or {"exit_code": 0},
        )
        values.update(overrides)
        return launch_attempt3(target, **values)

    assert launch()["exit_code"] == 0 and calls == ["supervise"]
    # The predecessor root changes after the seal: the next launch refuses.
    (root / "collection-failure.json").unlink()
    with pytest.raises(Attempt3Refusal) as refused:
        launch()
    assert any("failure_marker_missing" in item for item in refused.value.reasons)
    assert calls == ["supervise"]
