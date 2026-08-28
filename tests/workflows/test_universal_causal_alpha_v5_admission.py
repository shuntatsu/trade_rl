from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from trade_rl.workflows.universal_causal_alpha_v5_admission import (
    evaluate_causal_alpha_v5_admission,
)
from trade_rl.workflows.universal_causal_alpha_v5_replay import (
    CausalAlphaV5ReplayMetric,
)
from trade_rl.workflows.universal_causal_alpha_v5_selection import (
    CausalAlphaV5SelectionEvidence,
)
from trade_rl.workflows.universal_causal_alpha_v5_signal import (
    CausalAlphaV5SignalEvidence,
)


def _d(char: str) -> str:
    return char * 64


def _record(symbol: str) -> CausalAlphaV5ReplayMetric:
    return CausalAlphaV5ReplayMetric(
        run_manifest_digest=_d("1"),
        v4_context_manifest_digest=_d("2"),
        config_digest=_d("3"),
        symbol=symbol,
        episode_index=0,
        contract_digest=_d("4"),
        fit_digest=_d("5"),
        forecast_digest=_d("6"),
        calibration_fit_digest=_d("7"),
        target_path_digest=_d("8"),
        gross_return=0.02,
        net_return=0.01,
        turnover_per_day=0.1,
        total_execution_cost=0.001,
        submitted_change_count=1,
        downstream_no_trade_suppression_count=0,
        executed_change_count=1,
        closed_trade_count=0,
        sign_flip_count=0,
        maximum_drawdown=0.01,
        active_coverage=0.5,
        flat_time_fraction=0.5,
        time_weighted_absolute_exposure=0.1,
        completed_holding_durations_hours=(),
        has_unclosed_position=True,
        execution_rejection_reason_counts=(),
        risk_projection_reason_counts=(),
        target_reason_counts=(("entry", 1),),
        hard_risk_violation=False,
        has_meaningful_execution=True,
    )


def _upstream(*, signal_passed: bool = True, selection_passed: bool = True):
    signal = Mock(spec=CausalAlphaV5SignalEvidence)
    signal.passed = signal_passed
    signal.digest = _d("a")
    signal.slow = SimpleNamespace(
        run_manifest_digest=_d("1"),
        calibration_config_digest=_d("3"),
        metrics=(SimpleNamespace(calibration_fit_digest=_d("7")),),
    )
    selection = Mock(spec=CausalAlphaV5SelectionEvidence)
    selection.passed = selection_passed
    selection.digest = _d("b")
    selection.run_manifest_digest = _d("1")
    selection.v4_context_manifest_digest = _d("2")
    selection.config_digest = _d("3")
    return signal, selection


def test_v5_admission_reuses_untouched_holdout_economics_and_binds_cutoff() -> None:
    signal, selection = _upstream()
    evidence = evaluate_causal_alpha_v5_admission(
        (_record("BTCUSDT"), _record("ETHUSDT")),
        signal_evidence=signal,
        selection_evidence=selection,
        fit_knowledge_cutoff=100,
        holdout_start=100,
    )
    assert evidence.passed
    assert evidence.aggregate_net_return == 0.02
    assert evidence.calibration_fit_digest == _d("7")


def test_v5_admission_blocks_failed_upstream_cutoff_drift_and_duplicate_symbols() -> (
    None
):
    signal, selection = _upstream()
    with pytest.raises(ValueError, match="cutoff"):
        evaluate_causal_alpha_v5_admission(
            (_record("BTCUSDT"),),
            signal_evidence=signal,
            selection_evidence=selection,
            fit_knowledge_cutoff=99,
            holdout_start=100,
        )
    failed_signal, _ = _upstream(signal_passed=False)
    with pytest.raises(ValueError, match="Signal"):
        evaluate_causal_alpha_v5_admission(
            (_record("BTCUSDT"),),
            signal_evidence=failed_signal,
            selection_evidence=selection,
            fit_knowledge_cutoff=100,
            holdout_start=100,
        )
    _, failed_selection = _upstream(selection_passed=False)
    with pytest.raises(ValueError, match="Selection"):
        evaluate_causal_alpha_v5_admission(
            (_record("BTCUSDT"),),
            signal_evidence=signal,
            selection_evidence=failed_selection,
            fit_knowledge_cutoff=100,
            holdout_start=100,
        )
    with pytest.raises(ValueError, match="unique"):
        evaluate_causal_alpha_v5_admission(
            (_record("BTCUSDT"), _record("BTCUSDT")),
            signal_evidence=signal,
            selection_evidence=selection,
            fit_knowledge_cutoff=100,
            holdout_start=100,
        )
