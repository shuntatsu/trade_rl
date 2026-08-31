from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v5 import (
    CausalAlphaV5CalibrationConfig,
    CausalAlphaV5SelectiveForecast,
    V5SelectiveState,
)
from trade_rl.workflows.universal_causal_alpha_v5_signal import (
    CausalAlphaV5SignalEvidence,
    CausalAlphaV5SignalScopeMetric,
    build_causal_alpha_v5_signal_scope_metric,
    causal_alpha_v5_signal_diagnostic_payload,
    evaluate_causal_alpha_v5_signal_gate,
)

_DIGEST = "1" * 64
_SYMBOLS = tuple(f"S{index}" for index in range(9))
_CONFIG = CausalAlphaV5CalibrationConfig()


def _forecast() -> CausalAlphaV5SelectiveForecast:
    prediction = np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
    states = (
        V5SelectiveState.CONFIDENCE_ABSTAIN,
        V5SelectiveState.ACTIVE,
        V5SelectiveState.DIRECTION_DISAGREEMENT,
        V5SelectiveState.ACTIVE,
        V5SelectiveState.EDGE_BELOW_HURDLE,
        V5SelectiveState.ACTIVE,
    )
    active = np.asarray([state is V5SelectiveState.ACTIVE for state in states])
    return CausalAlphaV5SelectiveForecast(
        symbol="S0",
        decision_indices=np.arange(6) * 10,
        slow_return_raw=prediction,
        slow_direction_raw=prediction,
        slow_uncertainty_raw=np.ones(6),
        slow_return_calibrated=prediction,
        slow_uncertainty_calibrated=np.ones(6),
        return_confidence=np.ones(6),
        direction_confidence=np.ones(6),
        selective_confidence=np.ones(6),
        execution_hurdle=np.zeros(6),
        actionable_mask=np.ones(6, dtype=np.bool_),
        active_mask=active,
        states=states,
        v4_forecast_digest="2" * 64,
        calibration_fit_digest="3" * 64,
    )


def test_v5_scope_uses_all_raw_rows_but_active_rows_for_selective_direction() -> None:
    metric = build_causal_alpha_v5_signal_scope_metric(
        run_manifest_digest=_DIGEST,
        calibration_config_digest=_CONFIG.digest,
        symbol="S0",
        episode_index=0,
        contract_start=0,
        contract_stop=100,
        contract_digest="5" * 64,
        selective_forecast=_forecast(),
        labels_24h=np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]),
        label_end_indices_24h=np.arange(6) * 10 + 1,
        labels_72h=np.zeros(6),
        label_end_indices_72h=np.arange(6) * 10 + 2,
    )
    assert metric.raw_sample_count == 6
    assert metric.raw_direction_sample_count == 6
    assert metric.active_direction_sample_count == 3
    assert metric.active_coverage == 0.5
    assert metric.unconditional_rank_correlation == 1.0
    assert metric.unconditional_direction_accuracy == 1.0
    assert metric.selective_direction_accuracy == 1.0
    assert dict(metric.inactive_reason_counts) == {
        "confidence_abstain": 1,
        "direction_disagreement": 1,
        "edge_below_hurdle": 1,
    }


def test_v5_scope_excludes_zero_realized_direction() -> None:
    metric = build_causal_alpha_v5_signal_scope_metric(
        run_manifest_digest=_DIGEST,
        calibration_config_digest=_CONFIG.digest,
        symbol="S0",
        episode_index=0,
        contract_start=0,
        contract_stop=100,
        contract_digest="5" * 64,
        selective_forecast=_forecast(),
        labels_24h=np.asarray([-3.0, -2.0, -1.0, 1.0, 2.0, 0.0]),
        label_end_indices_24h=np.arange(6) * 10 + 1,
        labels_72h=np.zeros(6),
        label_end_indices_72h=np.arange(6) * 10 + 2,
    )
    assert metric.raw_direction_sample_count == 5
    assert metric.active_direction_sample_count == 2


def _metric(
    symbol: str, episode: int, *, active: int = 5
) -> CausalAlphaV5SignalScopeMetric:
    raw = 10
    inactive = raw - active
    return CausalAlphaV5SignalScopeMetric(
        run_manifest_digest=_DIGEST,
        calibration_config_digest=_CONFIG.digest,
        symbol=symbol,
        episode_index=episode,
        contract_start=episode * 100,
        contract_stop=(episode + 1) * 100,
        contract_digest="5" * 64,
        calibration_fit_digest="6" * 64,
        selective_forecast_digest=f"{episode + 1:x}" * 64,
        raw_sample_count=raw,
        raw_direction_sample_count=raw,
        active_sample_count=active,
        active_direction_sample_count=active,
        active_coverage=active / raw,
        unconditional_rank_correlation=0.2,
        unconditional_direction_accuracy=0.75,
        selective_direction_accuracy=1.0,
        unconditional_top_bottom_realized_spread=0.1,
        raw_cohort_indices=tuple(range(raw)),
        active_cohort_indices=tuple(range(active)),
        inactive_reason_counts=(("confidence_abstain", inactive),) if inactive else (),
    )


def _metrics() -> tuple[CausalAlphaV5SignalScopeMetric, ...]:
    return tuple(
        _metric(symbol, episode) for episode in range(8) for symbol in _SYMBOLS
    )


def _gate(
    metrics: tuple[CausalAlphaV5SignalScopeMetric, ...],
) -> CausalAlphaV5SignalEvidence:
    return evaluate_causal_alpha_v5_signal_gate(
        metrics,
        expected_symbols=_SYMBOLS,
        v4_fast_lane_digest="7" * 64,
        v4_fast_lane_passed=True,
        config=_CONFIG,
    )


def test_v5_gate_passes_exact_authored_scope_and_binds_fast_lane() -> None:
    evidence = _gate(_metrics())
    assert evidence.passed
    assert evidence.slow.raw_scope_count == 72
    assert evidence.slow.independent_episode_count == 8
    assert evidence.v4_fast_lane_digest == "7" * 64


def test_v5_signal_diagnostic_payload_preserves_scalar_and_scope_evidence() -> None:
    evidence = _gate(_metrics())

    payload = causal_alpha_v5_signal_diagnostic_payload(evidence)

    assert payload["signal_evidence_digest"] == evidence.digest
    assert payload["overall_active_coverage"] == 0.5
    assert (
        payload["unconditional_rank_ic"]
        == evidence.slow.unconditional_rank_ic.to_payload()
    )
    metrics = payload["metrics"]
    assert isinstance(metrics, tuple)
    assert len(metrics) == 72
    assert metrics[0]["symbol"] == "S0"
    assert metrics[0]["active_coverage"] == 0.5


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda values: values[:-1], "raw_scope_count"),
        (
            lambda values: tuple(
                metric for metric in values if metric.episode_index != 7
            ),
            "independent_episode_count",
        ),
        (
            lambda values: tuple(metric for metric in values if metric.symbol != "S8"),
            "symbol_coverage",
        ),
        (
            lambda values: (
                replace(
                    values[0],
                    active_sample_count=2,
                    active_direction_sample_count=2,
                    active_coverage=0.2,
                    active_cohort_indices=(0, 1),
                    inactive_reason_counts=(("confidence_abstain", 8),),
                    digest="",
                ),
                *values[1:],
            ),
            "scope_active_support",
        ),
    ],
)
def test_v5_gate_rejects_missing_scope_episode_symbol_or_support(
    mutation: Callable[
        [tuple[CausalAlphaV5SignalScopeMetric, ...]],
        tuple[CausalAlphaV5SignalScopeMetric, ...],
    ],
    reason: str,
) -> None:
    assert reason in _gate(mutation(_metrics())).rejection_reasons


def test_v5_gate_rejects_low_overall_coverage_and_unaccounted_abstention() -> None:
    low = tuple(
        _metric(symbol, episode, active=2)
        for episode in range(8)
        for symbol in _SYMBOLS
    )
    evidence = _gate(low)
    assert "active_coverage" in evidence.rejection_reasons
    assert "scope_active_support" in evidence.rejection_reasons
    with pytest.raises(ValueError, match="account"):
        replace(
            _metrics()[0],
            inactive_reason_counts=(("confidence_abstain", 4),),
            digest="",
        )


def test_v5_one_active_row_cannot_pass_biased_zero_point() -> None:
    sparse = tuple(
        _metric(symbol, episode, active=1)
        for episode in range(8)
        for symbol in _SYMBOLS
    )
    evidence = _gate(sparse)
    assert not evidence.passed
    assert "scope_active_support" in evidence.rejection_reasons
