from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaCostAwareConfig,
    CausalAlphaHorizonMix,
    CausalAlphaRidgeConfig,
    causal_alpha_cost_aware_target_path,
    causal_alpha_target_path,
    combine_causal_alpha_predictions,
    fit_causal_alpha_ridge,
    forward_log_return_label,
)


class _RegularDataset:
    def __init__(self, open_price: np.ndarray, close_price: np.ndarray) -> None:
        self.open = np.asarray(open_price, dtype=np.float64).reshape(-1, 1)
        self.close = np.asarray(close_price, dtype=np.float64).reshape(-1, 1)
        self.n_bars = int(self.open.shape[0])
        self.regular_cadence = True
        self.bar_hours = 0.25

    def bars_for_hours(self, hours: float) -> int:
        bars = hours / self.bar_hours
        if not float(bars).is_integer():
            raise ValueError("hours do not resolve to exact bars")
        return int(bars)


def test_forward_label_starts_at_first_executable_bar_and_respects_delay() -> None:
    prices = np.arange(1.0, 300.0, dtype=np.float64)
    dataset = _RegularDataset(prices, prices + 0.5)

    immediate = forward_log_return_label(
        dataset,
        decision_index=10,
        horizon_hours=24.0,
        signal_delay_decisions=0,
        decision_bars=1,
    )
    delayed = forward_log_return_label(
        dataset,
        decision_index=10,
        horizon_hours=24.0,
        signal_delay_decisions=1,
        decision_bars=1,
    )

    assert immediate.execution_start_index == 11
    assert immediate.label_end_index == 106
    assert immediate.value == pytest.approx(
        np.log(dataset.close[106, 0] / dataset.open[11, 0])
    )
    assert delayed.execution_start_index == 12
    assert delayed.label_end_index == 107
    assert delayed.value == pytest.approx(
        np.log(dataset.close[107, 0] / dataset.open[12, 0])
    )


def test_forward_label_rejects_incomplete_horizon() -> None:
    prices = np.arange(1.0, 110.0, dtype=np.float64)
    dataset = _RegularDataset(prices, prices + 0.5)
    with pytest.raises(ValueError, match="horizon"):
        forward_log_return_label(
            dataset,
            decision_index=20,
            horizon_hours=24.0,
            signal_delay_decisions=1,
            decision_bars=1,
        )


def test_pooled_ridge_fits_prefix_only_and_zeroes_constant_columns() -> None:
    features = np.asarray(
        [
            [1.0, 7.0],
            [2.0, 7.0],
            [3.0, 7.0],
            [4.0, 7.0],
            [5.0, 7.0],
            [6.0, 7.0],
        ],
        dtype=np.float64,
    )
    labels = 2.0 * features[:, 0] + 1.0
    label_end = np.asarray([2, 3, 4, 5, 10, 11], dtype=np.int64)
    available = np.ones_like(features, dtype=np.bool_)

    model = fit_causal_alpha_ridge(
        features=features,
        labels=labels,
        feature_available=available,
        label_end_indices=label_end,
        knowledge_cutoff=10,
        feature_names=("signal", "constant"),
        config=CausalAlphaRidgeConfig(ridge_strength=1e-9),
    )

    assert model.sample_count == 4
    assert model.eligible_indices.tolist() == [0, 1, 2, 3]
    assert model.knowledge_cutoff == 10
    assert model.constant_mask.tolist() == [False, True]
    scaled = model.transform(features[model.eligible_indices])
    assert scaled[:, 1].tolist() == [0.0, 0.0, 0.0, 0.0]
    prediction = model.predict(np.asarray([[7.0, 7.0]], dtype=np.float64))[0]
    assert prediction == pytest.approx(15.0, rel=1e-6)


def test_pooled_ridge_serialization_is_stable_and_identity_sensitive() -> None:
    features = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]])
    labels = np.asarray([1.0, 2.0, 3.0, 4.0])
    kwargs = dict(
        features=features,
        labels=labels,
        feature_available=np.ones_like(features, dtype=np.bool_),
        label_end_indices=np.asarray([1, 2, 3, 4], dtype=np.int64),
        knowledge_cutoff=5,
        feature_names=("a", "b"),
        config=CausalAlphaRidgeConfig(ridge_strength=0.5),
    )
    first = fit_causal_alpha_ridge(**kwargs)
    second = fit_causal_alpha_ridge(**kwargs)

    assert canonical_json_bytes(first.to_payload()) == canonical_json_bytes(
        second.to_payload()
    )
    assert first.digest == second.digest

    changed = fit_causal_alpha_ridge(
        **{**kwargs, "config": replace(kwargs["config"], ridge_strength=1.0)}
    )
    assert changed.digest != first.digest


def test_prediction_mix_is_declared_and_deterministic() -> None:
    p24 = np.asarray([0.1, -0.2])
    p72 = np.asarray([0.3, 0.4])
    assert combine_causal_alpha_predictions(
        p24, p72, CausalAlphaHorizonMix.H24
    ).tolist() == pytest.approx([0.1, -0.2])
    assert combine_causal_alpha_predictions(
        p24, p72, CausalAlphaHorizonMix.H72
    ).tolist() == pytest.approx([0.3, 0.4])
    assert combine_causal_alpha_predictions(
        p24, p72, CausalAlphaHorizonMix.EQUAL
    ).tolist() == pytest.approx([0.2, 0.1])


def test_controller_hysteresis_has_no_holding_lock_and_respects_initial_weight() -> (
    None
):
    config = CausalAlphaControllerConfig(
        horizon_mix=CausalAlphaHorizonMix.EQUAL,
        score_scale=10.0,
        entry_threshold=0.10,
        exit_threshold=0.03,
        no_trade_band=0.0,
        max_target_delta=2.0,
    )
    path = causal_alpha_target_path(
        np.asarray([0.20, -0.20, -0.01, 0.20], dtype=np.float64),
        config=config,
        initial_weight=0.25,
    )

    assert path.initial_weight == pytest.approx(0.25)
    assert path.targets[0] > 0.0
    assert path.targets[1] < 0.0  # immediate next-decision reversal is allowed
    assert path.targets[2] == pytest.approx(0.0)
    assert path.targets[3] > 0.0
    assert path.sign_flip_count >= 1


def test_controller_no_trade_band_and_delta_cap_reduce_target_changes() -> None:
    config = CausalAlphaControllerConfig(
        horizon_mix=CausalAlphaHorizonMix.H24,
        score_scale=10.0,
        entry_threshold=0.05,
        exit_threshold=0.01,
        no_trade_band=0.10,
        max_target_delta=0.20,
    )
    path = causal_alpha_target_path(
        np.asarray([0.5, 0.51, -0.5], dtype=np.float64),
        config=config,
        initial_weight=0.0,
    )

    assert path.targets[0] == pytest.approx(0.20)
    assert path.targets[1] == pytest.approx(0.40)
    assert path.targets[2] == pytest.approx(0.20)
    changes = np.diff(np.concatenate(([0.0], path.targets)))
    assert np.max(np.abs(changes)) <= 0.20 + 1e-12


def test_controller_config_rejects_invalid_hysteresis() -> None:
    with pytest.raises(ValueError, match="exit_threshold"):
        CausalAlphaControllerConfig(
            horizon_mix=CausalAlphaHorizonMix.H24,
            score_scale=1.0,
            entry_threshold=0.05,
            exit_threshold=0.05,
            no_trade_band=0.0,
            max_target_delta=1.0,
        )


def test_cost_aware_config_has_stable_digest() -> None:
    config = CausalAlphaCostAwareConfig(
        execution_cost_multiplier=1.5,
        edge_margin=0.001,
        confirmation_count=2,
        strong_reversal_threshold=0.02,
        max_abs_target=0.5,
    )

    assert config.digest == CausalAlphaCostAwareConfig(**asdict(config)).digest


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("execution_cost_multiplier", 0.0),
        ("edge_margin", -0.001),
        ("confirmation_count", 0),
        ("strong_reversal_threshold", 0.0),
        ("max_abs_target", 0.0),
        ("max_abs_target", 1.1),
    ),
)
def test_cost_aware_config_rejects_invalid_economic_limits(
    field: str, value: float | int
) -> None:
    kwargs: dict[str, float | int] = {
        "execution_cost_multiplier": 1.5,
        "edge_margin": 0.001,
        "confirmation_count": 2,
        "strong_reversal_threshold": 0.02,
        "max_abs_target": 0.5,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=field):
        CausalAlphaCostAwareConfig(**kwargs)  # type: ignore[arg-type]


def test_cost_aware_path_confirms_entries_and_allows_immediate_strong_reversal() -> (
    None
):
    controller = CausalAlphaControllerConfig(
        horizon_mix=CausalAlphaHorizonMix.EQUAL,
        score_scale=25.0,
        entry_threshold=0.001,
        exit_threshold=0.0005,
        no_trade_band=0.0,
        max_target_delta=1.0,
    )
    economic = CausalAlphaCostAwareConfig(
        execution_cost_multiplier=1.5,
        edge_margin=0.001,
        confirmation_count=2,
        strong_reversal_threshold=0.02,
        max_abs_target=0.5,
    )

    path = causal_alpha_cost_aware_target_path(
        np.asarray([0.003, 0.003, -0.004, -0.03]),
        one_way_cost_rates=np.full(4, 0.0009),
        controller=controller,
        economic=economic,
        initial_weight=0.0,
    )

    assert path.targets[0] == pytest.approx(0.0)
    assert path.targets[1] > 0.0
    assert path.targets[2] == pytest.approx(path.targets[1])
    assert path.targets[3] < path.targets[2]
    assert path.confirmation_state.tolist() == [1, 2, -1, -1]
    assert path.strong_reversal_count == 1
    assert path.sign_flip_count == 1


def test_cost_aware_path_suppresses_marginal_edge_and_preserves_inactive_state() -> (
    None
):
    controller = CausalAlphaControllerConfig(
        horizon_mix=CausalAlphaHorizonMix.H24,
        score_scale=25.0,
        entry_threshold=0.0001,
        exit_threshold=0.00005,
        no_trade_band=0.0,
        max_target_delta=1.0,
    )
    economic = CausalAlphaCostAwareConfig(
        execution_cost_multiplier=1.5,
        edge_margin=0.001,
        confirmation_count=1,
        strong_reversal_threshold=0.02,
        max_abs_target=0.05,
    )

    path = causal_alpha_cost_aware_target_path(
        np.asarray([0.002, 0.01, -0.03]),
        one_way_cost_rates=np.full(3, 0.0009),
        controller=controller,
        economic=economic,
        initial_weight=0.0,
        actionable_mask=np.asarray([True, True, False]),
    )

    assert path.targets.tolist() == pytest.approx([0.0, 0.05, 0.05])
    assert path.predicted_incremental_edge[0] <= path.estimated_cost_hurdle[0]
    assert path.predicted_incremental_edge[1] > path.estimated_cost_hurdle[1]
    assert path.proposed_turnover.tolist() == pytest.approx(
        [0.04995837495787998, 0.05, 0.0]
    )
    assert path.cost_suppressed_change_count == 1
    assert path.submitted_change_count == 1
    assert np.max(np.abs(path.targets)) <= economic.max_abs_target
    assert not path.targets.flags.writeable
    assert not path.actionable_mask.flags.writeable


def test_cost_aware_path_deleverages_initial_state_above_target_cap() -> None:
    controller = CausalAlphaControllerConfig(
        horizon_mix=CausalAlphaHorizonMix.H24,
        score_scale=25.0,
        entry_threshold=0.001,
        exit_threshold=0.0005,
        no_trade_band=0.05,
        max_target_delta=0.125,
    )
    economic = CausalAlphaCostAwareConfig(
        execution_cost_multiplier=1.5,
        edge_margin=0.001,
        confirmation_count=2,
        strong_reversal_threshold=0.02,
        max_abs_target=0.5,
    )

    path = causal_alpha_cost_aware_target_path(
        np.asarray([0.0, 0.0]),
        one_way_cost_rates=np.full(2, 0.001),
        controller=controller,
        economic=economic,
        initial_weight=0.8,
        actionable_mask=np.asarray([False, True]),
    )

    assert path.initial_weight == pytest.approx(0.8)
    assert path.targets.tolist() == pytest.approx([0.5, 0.5])
    assert path.proposed_turnover[0] == pytest.approx(0.3)
    assert path.estimated_cost_hurdle[0] == pytest.approx(0.00075)
    assert path.submitted_change_count == 1
