from __future__ import annotations

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


def test_forward_label_starts_at_first_executable_bar_and_ends_exact_horizon() -> None:
    close = np.asarray([100.0, 101.0, 102.0, 104.0, 108.0, 116.0], dtype=np.float64)
    label = forward_log_return_label(
        close,
        decision_index=1,
        horizon_bars=2,
        decision_bars=1,
    )

    assert label.decision_index == 1
    assert label.execution_start_index == 3
    assert label.label_end_index == 4
    assert label.value == pytest.approx(np.log(108.0 / 104.0))


def test_forward_label_rejects_missing_future_horizon() -> None:
    close = np.asarray([100.0, 101.0, 102.0], dtype=np.float64)
    with pytest.raises(ValueError, match="fully realized"):
        forward_log_return_label(
            close,
            decision_index=1,
            horizon_bars=2,
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
        **{**kwargs, "config": CausalAlphaRidgeConfig(ridge_strength=1.0)}
    )
    assert changed.digest != first.digest


def test_horizon_combinations_are_explicit_and_deterministic() -> None:
    short = np.asarray([0.1, -0.2], dtype=np.float64)
    long = np.asarray([0.3, 0.4], dtype=np.float64)

    assert combine_causal_alpha_predictions(
        short, long, CausalAlphaHorizonMix.H24
    ).tolist() == pytest.approx(short.tolist())
    assert combine_causal_alpha_predictions(
        short, long, CausalAlphaHorizonMix.H72
    ).tolist() == pytest.approx(long.tolist())
    assert combine_causal_alpha_predictions(
        short, long, CausalAlphaHorizonMix.EQUAL
    ).tolist() == pytest.approx([0.2, 0.1])


def _controller(**overrides: float | CausalAlphaHorizonMix) -> CausalAlphaControllerConfig:
    values: dict[str, float | CausalAlphaHorizonMix] = {
        "horizon_mix": CausalAlphaHorizonMix.H24,
        "score_scale": 2.0,
        "entry_threshold": 0.5,
        "exit_threshold": 0.2,
        "no_trade_band": 0.0,
        "max_target_delta": 2.0,
    }
    values.update(overrides)
    return CausalAlphaControllerConfig(**values)  # type: ignore[arg-type]


def test_controller_uses_entry_exit_and_sign_hysteresis_without_holding_lock() -> None:
    path = causal_alpha_target_path(
        np.asarray([0.3, 0.6, 0.3, 0.1, -0.3, -0.7]),
        config=_controller(),
        initial_weight=0.0,
    )

    expected_long = np.tanh(1.2)
    expected_held = np.tanh(0.6)
    expected_short = np.tanh(-1.4)
    assert path.targets.tolist() == pytest.approx(
        [0.0, expected_long, expected_held, 0.0, 0.0, expected_short]
    )
    assert path.sign_flip_count == 0
    assert path.submitted_change_count == 4


def test_controller_can_reverse_immediately_when_new_direction_clears_entry() -> None:
    path = causal_alpha_target_path(
        np.asarray([0.8, -0.8]),
        config=_controller(),
        initial_weight=0.0,
    )

    assert path.targets[0] > 0.0
    assert path.targets[1] < 0.0
    assert path.sign_flip_count == 1


def test_no_trade_and_delta_limit_reduce_target_changes() -> None:
    path = causal_alpha_target_path(
        np.asarray([1.0, 1.0, 1.0, 0.95]),
        config=_controller(no_trade_band=0.15, max_target_delta=0.25),
        initial_weight=0.0,
    )

    assert path.targets[0] == pytest.approx(0.25)
    assert path.targets[1] == pytest.approx(0.5)
    assert path.targets[2] == pytest.approx(0.75)
    assert path.targets[3] == pytest.approx(0.75)
    assert path.submitted_change_count == 3
    assert path.suppressed_change_count == 1


def test_controller_respects_non_zero_initial_weight() -> None:
    path = causal_alpha_target_path(
        np.asarray([0.1, 0.7]),
        config=_controller(),
        initial_weight=0.4,
    )

    assert path.initial_weight == pytest.approx(0.4)
    assert path.targets[0] == pytest.approx(0.0)
    assert path.targets[1] == pytest.approx(np.tanh(1.4))
