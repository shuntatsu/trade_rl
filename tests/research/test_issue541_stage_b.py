from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.simulation.execution import MarketExecutor
from tools.issue541_stage_b import (
    StageBCapacityAudit,
    StageBCapacityAuditError,
    audited_market_executor,
    stage_b_suite_decision,
)

DATASET_ID = "6c0b040d317a1bb73a9273f4135879b31691634aa837f30f0eec005ac7531518"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
CAPS = (
    0.0021629560553901974,
    0.002044685341258238,
    0.002184898995567895,
    0.0020480213652913385,
    0.002346308308284808,
)


def _audit() -> StageBCapacityAudit:
    return StageBCapacityAudit(
        dataset_id=DATASET_ID,
        symbols=SYMBOLS,
        capacity_caps=CAPS,
    )


def test_capacity_audit_accepts_causal_zero_overlay_and_tracks_maxima() -> None:
    audit = _audit()
    audit.record_interval(
        dataset_id=DATASET_ID,
        processing_bar_volume_capacity=False,
        runtime_max_participation_rate=1.0,
        participation_by_symbol=np.asarray(
            [CAPS[0] * 0.5, 0.0, CAPS[2], CAPS[3] * 0.25, 0.0],
            dtype=np.float64,
        ),
    )
    audit.record_interval(
        dataset_id=DATASET_ID,
        processing_bar_volume_capacity=False,
        runtime_max_participation_rate=1.0,
        participation_by_symbol=np.asarray(
            [CAPS[0], CAPS[1] * 0.75, 0.0, 0.0, CAPS[4] * 0.5],
            dtype=np.float64,
        ),
    )

    report = audit.to_payload()

    assert report["status"] == "PASS"
    assert report["execution_intervals_checked"] == 2
    assert report["capacity_violation_count"] == 0
    assert report["processing_bar_capacity_violation_count"] == 0
    assert report["runtime_overlay_violation_count"] == 0
    assert report["max_participation_by_symbol"][0] == CAPS[0]
    assert report["max_participation_by_symbol"][2] == CAPS[2]
    assert report["previous_completed_bar_capacity"] is True


def test_capacity_audit_rejects_processing_bar_capacity_mode() -> None:
    audit = _audit()
    with pytest.raises(StageBCapacityAuditError, match="previous-completed-bar"):
        audit.record_interval(
            dataset_id=DATASET_ID,
            processing_bar_volume_capacity=True,
            runtime_max_participation_rate=1.0,
            participation_by_symbol=np.zeros(5, dtype=np.float64),
        )


def test_capacity_audit_rejects_runtime_participation_overlay() -> None:
    audit = _audit()
    with pytest.raises(StageBCapacityAuditError, match="runtime participation overlay"):
        audit.record_interval(
            dataset_id=DATASET_ID,
            processing_bar_volume_capacity=False,
            runtime_max_participation_rate=0.05,
            participation_by_symbol=np.zeros(5, dtype=np.float64),
        )


def test_capacity_audit_rejects_any_fill_participation_above_sealed_cap() -> None:
    audit = _audit()
    bad = np.zeros(5, dtype=np.float64)
    bad[1] = CAPS[1] + 1e-8
    with pytest.raises(
        StageBCapacityAuditError, match="participation exceeds sealed cap"
    ):
        audit.record_interval(
            dataset_id=DATASET_ID,
            processing_bar_volume_capacity=False,
            runtime_max_participation_rate=1.0,
            participation_by_symbol=bad,
        )


def test_capacity_audit_rejects_dataset_identity_drift() -> None:
    audit = _audit()
    with pytest.raises(StageBCapacityAuditError, match="Dataset identity"):
        audit.record_interval(
            dataset_id="0" * 64,
            processing_bar_volume_capacity=False,
            runtime_max_participation_rate=1.0,
            participation_by_symbol=np.zeros(5, dtype=np.float64),
        )


def test_audited_market_executor_observes_result_and_restores_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _audit()
    original_calls: list[tuple[int, int]] = []

    def fake_execute(
        _self: object,
        _book: object,
        _target: object,
        *,
        start_index: int,
        bars: int,
    ) -> object:
        original_calls.append((start_index, bars))
        return SimpleNamespace(
            participation_by_symbol=np.asarray(
                [CAPS[0] * 0.25, 0.0, 0.0, 0.0, CAPS[4] * 0.5],
                dtype=np.float64,
            )
        )

    monkeypatch.setattr(MarketExecutor, "execute_interval", fake_execute)
    executor = SimpleNamespace(
        dataset=SimpleNamespace(dataset_id=DATASET_ID),
        cost=SimpleNamespace(
            processing_bar_volume_capacity=False,
            max_participation_rate=1.0,
        ),
    )

    with audited_market_executor(audit):
        result = MarketExecutor.execute_interval(
            executor, object(), object(), start_index=3, bars=1
        )
        assert result.participation_by_symbol[0] == CAPS[0] * 0.25

    assert MarketExecutor.execute_interval is fake_execute
    assert original_calls == [(3, 1)]
    assert audit.execution_intervals_checked == 1


def test_audited_market_executor_restores_method_after_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit = _audit()

    def fake_execute(
        _self: object,
        _book: object,
        _target: object,
        *,
        start_index: int,
        bars: int,
    ) -> object:
        del start_index, bars
        raise RuntimeError("boom")

    monkeypatch.setattr(MarketExecutor, "execute_interval", fake_execute)
    executor = SimpleNamespace(
        dataset=SimpleNamespace(dataset_id=DATASET_ID),
        cost=SimpleNamespace(
            processing_bar_volume_capacity=False,
            max_participation_rate=1.0,
        ),
    )

    with pytest.raises(RuntimeError, match="boom"):
        with audited_market_executor(audit):
            MarketExecutor.execute_interval(
                executor, object(), object(), start_index=0, bars=1
            )

    assert MarketExecutor.execute_interval is fake_execute
    assert audit.execution_intervals_checked == 0


def test_stage_b_suite_decision_is_frozen_by_empty_source_eligible_roster() -> None:
    decision = stage_b_suite_decision(source_profitable_core_strategies=())
    assert decision == "NO_PROFITABLE_CORE_BASELINE"

    with pytest.raises(StageBCapacityAuditError, match="source-profitable roster"):
        stage_b_suite_decision(source_profitable_core_strategies=("mean_reversion",))
