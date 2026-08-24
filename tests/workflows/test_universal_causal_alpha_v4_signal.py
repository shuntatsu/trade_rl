from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trade_rl.learning.causal_alpha_v4 import build_causal_alpha_v4_forecast
from trade_rl.workflows.universal_causal_alpha_v4_signal import (
    CausalAlphaV4SignalGateConfig,
    CausalAlphaV4SignalLane,
    CausalAlphaV4SignalScopeMetric,
    build_causal_alpha_v4_signal_scope_metrics,
    evaluate_causal_alpha_v4_signal_gate,
)


def _digest(char: str) -> str:
    return char * 64


def _forecast(*, rows: int, direction_sign: float = 1.0):
    index = np.arange(rows, dtype=np.float64)
    market = {
        "4h": 0.001 + index * 0.0001,
        "24h": 0.003 + index * 0.0002,
        "72h": 0.009 + index * 0.0006,
    }
    residual = {
        "4h": index * 0.00001,
        "24h": index * 0.00002,
        "72h": index * 0.00006,
    }
    direction = {
        horizon: np.full(rows, direction_sign, dtype=np.float64)
        for horizon in ("4h", "24h", "72h")
    }
    return build_causal_alpha_v4_forecast(
        symbol="ETHUSDT",
        decision_indices=np.arange(100, 100 + rows, dtype=np.int64),
        beta=np.ones(rows, dtype=np.float64),
        beta_available=np.ones(rows, dtype=np.bool_),
        market_predictions=market,
        residual_predictions=residual,
        direction_scores=direction,
        market_model_digests={horizon: _digest("a") for horizon in market},
        residual_model_digests={horizon: _digest("b") for horizon in market},
        direction_model_digests={horizon: _digest("c") for horizon in market},
        fit_digest=_digest("d"),
    )


def _scope_metrics(*, rows: int = 12, direction_sign: float = 1.0):
    forecast = _forecast(rows=rows, direction_sign=direction_sign)
    decisions = forecast.decision_indices
    labels_4h = np.linspace(0.001, 0.012, rows, dtype=np.float64)
    labels_24h = np.linspace(0.003, 0.036, rows, dtype=np.float64)
    labels_72h = np.linspace(0.009, 0.108, rows, dtype=np.float64)
    return build_causal_alpha_v4_signal_scope_metrics(
        run_manifest_digest=_digest("e"),
        fit_config_digest=_digest("f"),
        symbol="ETHUSDT",
        episode_index=0,
        contract_start=100,
        contract_stop=200,
        contract_digest=_digest("1"),
        fit_digest=_digest("d"),
        forecast=forecast,
        liveness_digests={
            "fast_4h": _digest("2"),
            "slow_fused": _digest("3"),
        },
        actionable_mask=np.ones(rows, dtype=np.bool_),
        labels_4h=labels_4h,
        label_end_indices_4h=decisions + 1,
        labels_24h=labels_24h,
        label_end_indices_24h=decisions + 3,
        labels_72h=labels_72h,
        label_end_indices_72h=decisions + 5,
    )


def test_v4_research_json_freezes_first_signal_hypothesis() -> None:
    payload = json.loads(
        Path("examples/binance/universal-causal-alpha-v4-research.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload == {
        "schema_version": "universal_causal_alpha_v4_research_config_v1",
        "fit": {
            "market_ridge_strength": 1.0,
            "residual_ridge_strength": 0.1,
            "direction_ridge_strength": 0.1,
        },
        "target": {
            "slow_target_magnitudes": [0.0, 0.025, 0.05, 0.1, 0.25],
            "fast_deviation_magnitudes": [0.0, 0.025, 0.05],
            "uncertainty_multiplier": 1.0,
            "execution_cost_multiplier": 1.5,
            "edge_margin": 0.001,
            "slow_rebalance_decisions": 16,
            "fast_rebalance_decisions": 4,
            "maximum_final_target_delta": 0.125,
            "maximum_fast_absolute_deviation": 0.05,
        },
        "signal_gate": {
            "independent_episode_count": 8,
            "minimum_rank_ic_lower_ci": 0.0,
            "minimum_top_bottom_spread_lower_ci": 0.0,
            "minimum_direction_accuracy_excess_lower_ci": 0.0,
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 20260823,
            "bootstrap_block_size": 2,
        },
    }
    assert CausalAlphaV4SignalGateConfig.from_mapping(payload["signal_gate"]).digest


def test_fast_signal_cohort_uses_4h_label_ends_not_slow_label_ends() -> None:
    metrics = _scope_metrics(rows=12)
    fast = metrics[CausalAlphaV4SignalLane.FAST_4H]
    slow = metrics[CausalAlphaV4SignalLane.SLOW_FUSED]

    assert fast.cohort_indices == (100, 102, 104, 106, 108, 110)
    assert slow.cohort_indices == (100, 106)
    assert fast.sample_count > slow.sample_count


def test_slow_signal_uses_24h_equivalent_return_fusion() -> None:
    rows = 12
    forecast = _forecast(rows=rows, direction_sign=1.0)
    decisions = forecast.decision_indices
    labels_24h = np.linspace(-0.03, 0.03, rows, dtype=np.float64)
    labels_72h = np.linspace(-0.09, 0.09, rows, dtype=np.float64)
    metrics = build_causal_alpha_v4_signal_scope_metrics(
        run_manifest_digest=_digest("e"),
        fit_config_digest=_digest("f"),
        symbol="ETHUSDT",
        episode_index=0,
        contract_start=100,
        contract_stop=200,
        contract_digest=_digest("1"),
        fit_digest=_digest("d"),
        forecast=forecast,
        liveness_digests={
            "fast_4h": _digest("2"),
            "slow_fused": _digest("3"),
        },
        actionable_mask=np.ones(rows, dtype=np.bool_),
        labels_4h=np.linspace(-0.01, 0.01, rows, dtype=np.float64),
        label_end_indices_4h=decisions + 1,
        labels_24h=labels_24h,
        label_end_indices_24h=decisions + 3,
        labels_72h=labels_72h,
        label_end_indices_72h=decisions + 5,
    )
    slow = metrics[CausalAlphaV4SignalLane.SLOW_FUSED]

    cohort_positions = np.asarray([0, 6], dtype=np.int64)
    expected_prediction = 0.5 * (
        forecast.final_predictions["24h"][cohort_positions]
        + forecast.final_predictions["72h"][cohort_positions] / 3.0
    )
    expected_realized = 0.5 * (
        labels_24h[cohort_positions] + labels_72h[cohort_positions] / 3.0
    )
    assert np.sign(expected_prediction[-1]) == np.sign(expected_realized[-1])
    assert slow.direction_sample_count == 2
    assert slow.direction_accuracy == 0.5


def test_direction_accuracy_uses_independent_direction_head_not_return_sign() -> None:
    rows = 12
    metrics = _scope_metrics(rows=rows, direction_sign=-1.0)
    fast = metrics[CausalAlphaV4SignalLane.FAST_4H]

    assert fast.rank_correlation > 0.0
    assert fast.direction_accuracy == 0.0


def test_slow_direction_accuracy_uses_deployed_fused_return_sign() -> None:
    metrics = _scope_metrics(rows=12, direction_sign=-1.0)
    slow = metrics[CausalAlphaV4SignalLane.SLOW_FUSED]

    assert slow.rank_correlation > 0.0
    assert slow.direction_accuracy == 1.0


def test_exact_zero_realized_direction_is_excluded_from_direction_support() -> None:
    rows = 12
    forecast = _forecast(rows=rows)
    decisions = forecast.decision_indices
    labels_4h = np.linspace(0.001, 0.012, rows, dtype=np.float64)
    labels_4h[0] = 0.0
    metrics = build_causal_alpha_v4_signal_scope_metrics(
        run_manifest_digest=_digest("e"),
        fit_config_digest=_digest("f"),
        symbol="ETHUSDT",
        episode_index=0,
        contract_start=100,
        contract_stop=200,
        contract_digest=_digest("1"),
        fit_digest=_digest("d"),
        forecast=forecast,
        liveness_digests={
            "fast_4h": _digest("2"),
            "slow_fused": _digest("3"),
        },
        actionable_mask=np.ones(rows, dtype=np.bool_),
        labels_4h=labels_4h,
        label_end_indices_4h=decisions + 1,
        labels_24h=np.linspace(0.003, 0.036, rows),
        label_end_indices_24h=decisions + 3,
        labels_72h=np.linspace(0.009, 0.108, rows),
        label_end_indices_72h=decisions + 5,
    )
    fast = metrics[CausalAlphaV4SignalLane.FAST_4H]

    assert fast.sample_count == 6
    assert fast.direction_sample_count == 5


def test_liveness_digest_is_bound_to_signal_scope_identity() -> None:
    first = _scope_metrics()[CausalAlphaV4SignalLane.FAST_4H]
    second = CausalAlphaV4SignalScopeMetric(
        run_manifest_digest=first.run_manifest_digest,
        fit_config_digest=first.fit_config_digest,
        lane=first.lane,
        symbol=first.symbol,
        episode_index=first.episode_index,
        contract_start=first.contract_start,
        contract_stop=first.contract_stop,
        contract_digest=first.contract_digest,
        fit_digest=first.fit_digest,
        forecast_digest=first.forecast_digest,
        liveness_digest=_digest("9"),
        sample_count=first.sample_count,
        direction_sample_count=first.direction_sample_count,
        rank_correlation=first.rank_correlation,
        direction_accuracy=first.direction_accuracy,
        top_bottom_realized_spread=first.top_bottom_realized_spread,
        cohort_indices=first.cohort_indices,
    )

    assert second.digest != first.digest


def _metric(
    *, lane: CausalAlphaV4SignalLane, episode: int, good: bool
) -> CausalAlphaV4SignalScopeMetric:
    return CausalAlphaV4SignalScopeMetric(
        run_manifest_digest=_digest("e"),
        fit_config_digest=_digest("f"),
        lane=lane,
        symbol="ETHUSDT",
        episode_index=episode,
        contract_start=episode * 100,
        contract_stop=episode * 100 + 50,
        contract_digest=(_digest(str((episode % 8) + 1))),
        fit_digest=_digest("d"),
        forecast_digest=_digest("4"),
        liveness_digest=_digest("5"),
        sample_count=10,
        direction_sample_count=10,
        rank_correlation=1.0,
        direction_accuracy=1.0 if good else 0.0,
        top_bottom_realized_spread=0.01,
        cohort_indices=tuple(range(episode * 100, episode * 100 + 10)),
    )


def test_both_fast_and_slow_lanes_must_pass_gate() -> None:
    gate = CausalAlphaV4SignalGateConfig()
    fast = tuple(
        _metric(lane=CausalAlphaV4SignalLane.FAST_4H, episode=index, good=True)
        for index in range(8)
    )
    slow = tuple(
        _metric(lane=CausalAlphaV4SignalLane.SLOW_FUSED, episode=index, good=False)
        for index in range(8)
    )

    evidence = evaluate_causal_alpha_v4_signal_gate(
        fast + slow,
        expected_raw_scope_count_per_lane=8,
        gate=gate,
    )

    assert evidence.fast_4h.passed is True
    assert evidence.slow_fused.passed is False
    assert evidence.passed is False
    assert "slow_fused:direction_accuracy_excess_lower_ci" in evidence.rejection_reasons


def test_incomplete_lane_scope_cannot_pass() -> None:
    gate = CausalAlphaV4SignalGateConfig()
    metrics = tuple(
        _metric(lane=CausalAlphaV4SignalLane.FAST_4H, episode=index, good=True)
        for index in range(7)
    ) + tuple(
        _metric(lane=CausalAlphaV4SignalLane.SLOW_FUSED, episode=index, good=True)
        for index in range(8)
    )

    evidence = evaluate_causal_alpha_v4_signal_gate(
        metrics,
        expected_raw_scope_count_per_lane=8,
        gate=gate,
    )

    assert evidence.fast_4h.passed is False
    assert "fast_4h:raw_scope_count" in evidence.rejection_reasons
    assert evidence.passed is False
