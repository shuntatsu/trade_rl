from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trade_rl.evaluation.paper_reconciliation import (
    PaperReconciliationEvidence,
    load_paper_reconciliation_evidence,
    write_paper_reconciliation_evidence,
)

START = datetime(2026, 7, 2, tzinfo=UTC)


def _evidence(**overrides: object) -> PaperReconciliationEvidence:
    values: dict[str, object] = {
        "dataset_id": "1" * 64,
        "environment_digest": "2" * 64,
        "policy_digest": "3" * 64,
        "training_run_digest": "4" * 64,
        "start_time": START,
        "end_time": START + timedelta(days=30),
        "created_at": START + timedelta(days=30, minutes=1),
        "order_log_digest": "5" * 64,
        "fill_log_digest": "6" * 64,
        "submitted_order_count": 120,
        "terminal_order_count": 120,
        "observed_fill_count": 100,
        "matched_fill_count": 100,
        "unknown_order_fill_count": 0,
        "duplicate_fill_count": 0,
        "open_order_count": 0,
        "maximum_position_notional_difference_fraction": 0.0,
        "maximum_cash_difference_fraction": 0.0,
        "maximum_equity_difference_fraction": 0.0,
        "position_notional_tolerance_fraction": 1e-8,
        "cash_tolerance_fraction": 1e-8,
        "equity_tolerance_fraction": 1e-8,
    }
    values.update(overrides)
    return PaperReconciliationEvidence.create(**values)


def test_paper_reconciliation_round_trips_derived_pass_state(tmp_path: Path) -> None:
    evidence = _evidence()
    path = write_paper_reconciliation_evidence(
        tmp_path / "paper-reconciliation.json", evidence
    )

    assert evidence.passed is True
    assert load_paper_reconciliation_evidence(path) == evidence


def test_paper_reconciliation_rejects_incomplete_fill_matching() -> None:
    evidence = _evidence(matched_fill_count=99, unknown_order_fill_count=1)

    assert evidence.passed is False
    with pytest.raises(ValueError, match="did not pass"):
        evidence.require_promotable()


def test_paper_reconciliation_rejects_release_tolerance_above_cap() -> None:
    evidence = _evidence(
        position_notional_tolerance_fraction=1e-4,
        cash_tolerance_fraction=1e-4,
        equity_tolerance_fraction=1e-4,
    )

    assert evidence.passed is True
    with pytest.raises(ValueError, match="release tolerance"):
        evidence.require_promotable()


def test_paper_reconciliation_load_rejects_tampered_observation(tmp_path: Path) -> None:
    path = write_paper_reconciliation_evidence(
        tmp_path / "paper-reconciliation.json", _evidence()
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["maximum_cash_difference_fraction"] = 0.5
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        load_paper_reconciliation_evidence(path)


def test_paper_reconciliation_write_is_immutable(tmp_path: Path) -> None:
    path = write_paper_reconciliation_evidence(
        tmp_path / "paper-reconciliation.json", _evidence()
    )

    with pytest.raises(FileExistsError, match="immutable"):
        write_paper_reconciliation_evidence(
            path,
            _evidence(maximum_cash_difference_fraction=1e-9),
        )
