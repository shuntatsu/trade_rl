from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.learning.causal_alpha_teacher import (
    CausalAlphaControllerConfig,
    CausalAlphaHorizonMix,
    CausalAlphaRidgeConfig,
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
    available[1, 0] = False

    model = fit_causal_alpha_ridge(
        features=features,
        labels=labels,
        feature_available=available,
        label_end_indices=label_end,
        knowledge_cutoff=10,
        feature_names=("signal", "constant"),
        config=CausalAlphaRidgeConfig(ridge_strength=1e-9),
    )

    assert model.sample_count == 3
    assert model.knowledge_cutoff == 10
    assert model.constant_mask.tolist() == [False, True]
    scaled = model.transform(features[model.eligible_indices])
    assert scaled[:, 1].tolist() == [0.0, 0.0, 0.0]
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
