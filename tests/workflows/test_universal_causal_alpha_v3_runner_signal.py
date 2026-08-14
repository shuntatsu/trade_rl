from __future__ import annotations

import numpy as np
import pytest

from trade_rl.workflows.universal_causal_alpha_v3_config import CausalAlphaV3SignalGate
from trade_rl.workflows.universal_causal_alpha_v3_signal import (
    CausalAlphaV3SignalScopeMetric,
    evaluate_causal_alpha_v3_signal_gate,
    non_overlapping_causal_alpha_v3_rows,
)


def _metric(*, rank: float, spread: float, direction: float, episode: int) -> CausalAlphaV3SignalScopeMetric:
    return CausalAlphaV3SignalScopeMetric(
        fit_config_digest="1" * 64,
        symbol="BTCUSDT",
        episode_index=episode,
        contract_digest=(f"{episode + 2:x}" * 64)[:64],
        fit_digest="3" * 64,
        forecast_digest="4" * 64,
        sample_count=8,
        rank_correlation=rank,
        direction_accuracy=direction,
        top_bottom_realized_spread=spread,
        cohort_indices=tuple(range(8)),
    )


def test_non_overlapping_signal_rows_use_label_interval_endpoints() -> None:
    selected = non_overlapping_causal_alpha_v3_rows(
        decision_indices=np.asarray([0, 1, 2, 3, 6], dtype=np.int64),
        label_end_indices=np.asarray([2, 3, 4, 5, 8], dtype=np.int64),
        eligible_mask=np.asarray([True, True, True, True, True]),
    )

    assert selected.tolist() == [0, 3, 4]


def test_signal_gate_is_based_on_scope_level_bootstrap_lower_bounds() -> None:
    gate = CausalAlphaV3SignalGate(
        minimum_scope_count=2,
        minimum_scope_coverage=1.0,
        minimum_rank_ic_lower_ci=0.01,
        minimum_top_bottom_spread_lower_ci=0.001,
        minimum_direction_accuracy_excess_lower_ci=0.01,
        bootstrap_resamples=200,
        bootstrap_seed=5,
        bootstrap_block_size=1,
    )
    metrics = (
        _metric(rank=0.20, spread=0.010, direction=0.65, episode=0),
        _metric(rank=0.25, spread=0.012, direction=0.62, episode=1),
        _metric(rank=0.18, spread=0.008, direction=0.60, episode=2),
    )

    evidence = evaluate_causal_alpha_v3_signal_gate(
        metrics,
        expected_scope_count=3,
        gate=gate,
    )

    assert evidence.passed is True
    assert evidence.scope_coverage == pytest.approx(1.0)
    assert evidence.rank_ic.lower_ci > 0.01
    assert evidence.top_bottom_spread.lower_ci > 0.001
    assert evidence.direction_accuracy_excess.lower_ci > 0.01
    assert evidence.promotion_eligible is False


def test_signal_gate_fails_closed_on_missing_scope_or_undefined_rank_ic() -> None:
    gate = CausalAlphaV3SignalGate(
        minimum_scope_count=2,
        minimum_scope_coverage=1.0,
        minimum_rank_ic_lower_ci=0.0,
        minimum_top_bottom_spread_lower_ci=0.0,
        minimum_direction_accuracy_excess_lower_ci=0.0,
        bootstrap_resamples=100,
        bootstrap_seed=0,
        bootstrap_block_size=1,
    )

    metric = _metric(rank=0.2, spread=0.01, direction=0.6, episode=0)
    evidence = evaluate_causal_alpha_v3_signal_gate(
        (metric,), expected_scope_count=2, gate=gate
    )
    assert evidence.passed is False
    assert "scope_coverage" in evidence.rejection_reasons

    with pytest.raises(ValueError, match="rank"):
        CausalAlphaV3SignalScopeMetric(
            fit_config_digest="1" * 64,
            symbol="BTCUSDT",
            episode_index=0,
            contract_digest="2" * 64,
            fit_digest="3" * 64,
            forecast_digest="4" * 64,
            sample_count=8,
            rank_correlation=None,
            direction_accuracy=0.6,
            top_bottom_realized_spread=0.01,
            cohort_indices=tuple(range(8)),
        )
