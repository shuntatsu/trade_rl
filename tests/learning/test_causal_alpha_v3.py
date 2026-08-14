from __future__ import annotations

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v3 import (
    CausalAlphaV3TargetConfig,
    causal_alpha_overlap_uniqueness_weights,
    causal_alpha_v3_forecast,
    causal_alpha_v3_target_path,
)


def test_overlap_uniqueness_downweights_more_concurrent_labels() -> None:
    decisions = np.asarray([0, 1, 2], dtype=np.int64)
    label_ends = np.asarray([4, 4, 3], dtype=np.int64)

    weights = causal_alpha_overlap_uniqueness_weights(
        decisions,
        label_ends,
        knowledge_cutoff=5,
    )

    assert weights.shape == decisions.shape
    assert np.all(weights > 0.0)
    assert weights[1] < weights[0]


def test_overlap_uniqueness_excludes_labels_not_realized_before_cutoff() -> None:
    decisions = np.asarray([0, 1, 2], dtype=np.int64)
    label_ends = np.asarray([2, 4, 5], dtype=np.int64)

    weights = causal_alpha_overlap_uniqueness_weights(
        decisions,
        label_ends,
        knowledge_cutoff=5,
    )

    assert weights[0] > 0.0
    assert weights[1] > 0.0
    assert weights[2] == 0.0


def test_v3_forecast_converts_72h_prediction_to_24h_equivalent_units() -> None:
    forecast = causal_alpha_v3_forecast(
        np.asarray([0.03, -0.03]),
        np.asarray([0.09, -0.09]),
        residual_rmse_24h=0.01,
        residual_rmse_72h=0.03,
    )

    assert forecast.expected_return_24h_equivalent == pytest.approx([0.03, -0.03])
    assert np.all(forecast.uncertainty_24h_equivalent > 0.0)


def test_v3_forecast_uncertainty_increases_when_horizons_disagree() -> None:
    agreement = causal_alpha_v3_forecast(
        np.asarray([0.03]),
        np.asarray([0.09]),
        residual_rmse_24h=0.01,
        residual_rmse_72h=0.03,
    )
    disagreement = causal_alpha_v3_forecast(
        np.asarray([0.03]),
        np.asarray([-0.09]),
        residual_rmse_24h=0.01,
        residual_rmse_72h=0.03,
    )

    assert disagreement.uncertainty_24h_equivalent[0] > (
        agreement.uncertainty_24h_equivalent[0]
    )


def _target_config(**overrides: object) -> CausalAlphaV3TargetConfig:
    values: dict[str, object] = {
        "target_magnitudes": (0.0, 0.1, 0.25, 0.5),
        "uncertainty_multiplier": 1.0,
        "execution_cost_multiplier": 1.5,
        "edge_margin": 0.001,
        "alpha_rebalance_decisions": 1,
        "strong_reversal_threshold": 0.05,
        "max_target_delta": 1.0,
    }
    values.update(overrides)
    return CausalAlphaV3TargetConfig(**values)  # type: ignore[arg-type]


def test_target_compiler_holds_when_conservative_edge_does_not_clear_cost() -> None:
    path = causal_alpha_v3_target_path(
        np.asarray([0.01]),
        uncertainties=np.asarray([0.02]),
        one_way_cost_rates=np.asarray([0.001]),
        liquidity_weight_caps=np.asarray([0.5]),
        config=_target_config(),
        initial_weight=0.25,
    )

    assert path.targets.tolist() == pytest.approx([0.25])
    assert path.reasons == ("hold",)
    assert path.submitted_change_count == 0


def test_target_compiler_selects_confident_long_and_short_targets() -> None:
    path = causal_alpha_v3_target_path(
        np.asarray([0.20, -0.20]),
        uncertainties=np.asarray([0.01, 0.01]),
        one_way_cost_rates=np.asarray([0.0005, 0.0005]),
        liquidity_weight_caps=np.asarray([0.5, 0.5]),
        config=_target_config(),
        initial_weight=0.0,
    )

    assert path.targets[0] > 0.0
    assert path.targets[1] < 0.0
    assert path.submitted_change_count == 2


def test_target_compiler_blocks_non_emergency_changes_between_rebalances() -> None:
    path = causal_alpha_v3_target_path(
        np.asarray([0.20, 0.20, 0.20, 0.20, 0.20]),
        uncertainties=np.full(5, 0.001),
        one_way_cost_rates=np.full(5, 0.0001),
        liquidity_weight_caps=np.full(5, 0.5),
        config=_target_config(
            alpha_rebalance_decisions=4,
            max_target_delta=0.1,
        ),
        initial_weight=0.0,
    )

    assert path.targets[0] == pytest.approx(0.1)
    assert path.targets[1:4].tolist() == pytest.approx([0.1, 0.1, 0.1])
    assert path.targets[4] == pytest.approx(0.2)
    assert path.reasons[1:4] == ("cadence_hold", "cadence_hold", "cadence_hold")


def test_target_compiler_deleverages_immediately_when_liquidity_cap_contracts() -> (
    None
):
    path = causal_alpha_v3_target_path(
        np.asarray([0.20, 0.20]),
        uncertainties=np.asarray([0.001, 0.001]),
        one_way_cost_rates=np.asarray([0.0001, 0.0001]),
        liquidity_weight_caps=np.asarray([0.5, 0.15]),
        config=_target_config(alpha_rebalance_decisions=8),
        initial_weight=0.4,
    )

    assert path.targets[0] == pytest.approx(0.5)
    assert path.targets[1] == pytest.approx(0.15)
    assert path.reasons[1] == "liquidity_deleverage"
    assert path.liquidity_deleveraging_count == 1


def test_target_compiler_respects_max_target_delta_for_alpha_rebalances() -> None:
    path = causal_alpha_v3_target_path(
        np.asarray([0.50]),
        uncertainties=np.asarray([0.0]),
        one_way_cost_rates=np.asarray([0.0]),
        liquidity_weight_caps=np.asarray([0.5]),
        config=_target_config(max_target_delta=0.1),
        initial_weight=0.0,
    )

    assert path.targets.tolist() == pytest.approx([0.1])


def test_target_compiler_tie_breaks_toward_lower_turnover_then_lower_exposure() -> None:
    path = causal_alpha_v3_target_path(
        np.asarray([0.0]),
        uncertainties=np.asarray([0.0]),
        one_way_cost_rates=np.asarray([0.0]),
        liquidity_weight_caps=np.asarray([0.5]),
        config=_target_config(edge_margin=0.0),
        initial_weight=0.1,
    )

    assert path.targets.tolist() == pytest.approx([0.1])
    assert path.reasons == ("hold",)
