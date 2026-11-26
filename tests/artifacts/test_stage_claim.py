"""Synthetic single-use stage claim contracts. No actual study ledger is opened."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path

import pytest

from trade_rl.artifacts.stage_claim import (
    StageClaim,
    inspect_stage_ledger_root,
    reserve_stage_claim,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="strict dir-fd POSIX contract"
)


def claim(stage="SYNTHETIC", comment_id=42, digest="a" * 64):
    return StageClaim(
        stage, digest, "b" * 40, 123, comment_id, datetime(2026, 1, 1, tzinfo=UTC)
    )


def test_absolute_root_across_two_cwds_and_duplicate_refusal(tmp_path, monkeypatch):
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    pin = inspect_stage_ledger_root(ledger)
    monkeypatch.chdir(a)
    receipt = reserve_stage_claim(pin, claim())
    assert receipt.stage_file == "stage-SYNTHETIC.claim"
    assert (ledger / receipt.stage_file).is_file()
    monkeypatch.chdir(b)
    with pytest.raises(FileExistsError):
        reserve_stage_claim(pin, claim())
    assert list(a.iterdir()) == list(b.iterdir()) == []


@pytest.mark.parametrize(
    "root", [Path("relative"), Path(r"C:\trade_rl_runs\ledger"), Path("//tmp/ledger")]
)
def test_relative_and_ambiguous_absolute_root_refused(root):
    with pytest.raises(ValueError):
        inspect_stage_ledger_root(root)


def test_replaced_root_rejected_before_claim(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    pin = inspect_stage_ledger_root(root)
    root.rename(tmp_path / "old")
    root.mkdir()
    with pytest.raises(ValueError, match="identity"):
        reserve_stage_claim(pin, claim())
    assert list(root.iterdir()) == []


def test_symlinked_root_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises((ValueError, OSError)):
        inspect_stage_ledger_root(alias)
    assert list(root.iterdir()) == []


def test_same_comment_cannot_be_reused_for_other_stage_or_digest(tmp_path):
    pin = inspect_stage_ledger_root(tmp_path)
    reserve_stage_claim(pin, claim())
    with pytest.raises(FileExistsError):
        reserve_stage_claim(pin, claim("SECOND", digest="c" * 64))
    assert not (tmp_path / "stage-SECOND.claim").exists()


def test_stage_cannot_be_reused_with_different_comment_or_digest(tmp_path):
    pin = inspect_stage_ledger_root(tmp_path)
    reserve_stage_claim(pin, claim())
    with pytest.raises(FileExistsError):
        reserve_stage_claim(pin, claim(comment_id=43, digest="c" * 64))


def test_concurrent_duplicate_only_one_winner(tmp_path):
    pin = inspect_stage_ledger_root(tmp_path)

    def attempt(_):
        try:
            reserve_stage_claim(pin, claim())
            return True
        except FileExistsError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(attempt, range(16))) == 1


def test_publication_failure_leaves_consumed_marker(tmp_path, monkeypatch):
    pin = inspect_stage_ledger_root(tmp_path)
    real = os.fsync

    def fail(_):
        raise OSError("synthetic flush failure")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError):
        reserve_stage_claim(pin, claim())
    monkeypatch.setattr(os, "fsync", real)
    assert (tmp_path / "comment-123-42.claim").exists()
    with pytest.raises(FileExistsError):
        reserve_stage_claim(pin, claim())


def test_validation_before_writes(tmp_path):
    inspect_stage_ledger_root(tmp_path)
    for args in [
        ("../SYNTHETIC", "a" * 64, "b" * 40, 123, 42, datetime(2026, 1, 1, tzinfo=UTC)),
        ("SYNTHETIC", "a" * 64, "b" * 40, True, 42, datetime(2026, 1, 1, tzinfo=UTC)),
        ("SYNTHETIC", "a" * 64, "b" * 40, 123, 42, datetime(2026, 1, 1)),
    ]:
        with pytest.raises(ValueError):
            StageClaim(*args)
    assert list(tmp_path.iterdir()) == []


def test_claim_normalizes_utc_with_one_offset_sample():
    class ChangingOffset(tzinfo):
        calls = 0

        def utcoffset(self, dt):
            self.calls += 1
            return timedelta(hours=self.calls)

    zone = ChangingOffset()
    now = datetime(2026, 1, 1, 1, tzinfo=zone)
    record = StageClaim("SYNTHETIC", "a" * 64, "b" * 40, 123, 42, now)
    assert record.claimed_at == datetime(2026, 1, 1, tzinfo=UTC)
    assert zone.calls == 1
