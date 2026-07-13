from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from trade_rl.rl.experiments import ActionAblation, ActionExperimentSpec
from trade_rl.rl.normalization import ObservationNormalizer
from trade_rl.rl.policies import AssetSetFeatureExtractor
from trade_rl.rl.rewards import RewardConfig, RewardTracker, relative_interval_reward


def test_asset_set_extractor_is_permutation_invariant_and_handles_no_active_assets() -> (
    None
):
    torch.manual_seed(0)
    box = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
    extractor = AssetSetFeatureExtractor(
        box,
        n_symbols=2,
        per_symbol_width=2,
        global_width=3,
        active_column=1,
        asset_embedding_dim=4,
        global_embedding_dim=3,
    )
    extractor.eval()
    first = torch.tensor([[1.0, 1.0, 2.0, 1.0, 0.1, 0.2, 0.3]])
    swapped = torch.tensor([[2.0, 1.0, 1.0, 1.0, 0.1, 0.2, 0.3]])
    with torch.no_grad():
        output = extractor(first)
        permuted = extractor(swapped)
        inactive = extractor(torch.tensor([[1.0, 0.0, 2.0, 0.0, 0.1, 0.2, 0.3]]))
    assert output.shape == (1, 7)
    torch.testing.assert_close(output, permuted)
    torch.testing.assert_close(inactive[:, :4], torch.zeros((1, 4)))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n_symbols": 0}, "positive"),
        ({"active_column": 2}, "outside"),
        ({"asset_embedding_dim": 0}, "positive"),
    ],
)
def test_asset_set_extractor_validates_dimensions(
    kwargs: dict[str, int],
    message: str,
) -> None:
    values = {
        "n_symbols": 2,
        "per_symbol_width": 2,
        "global_width": 3,
        "active_column": 1,
        "asset_embedding_dim": 4,
        "global_embedding_dim": 3,
    }
    values.update(kwargs)
    box = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        AssetSetFeatureExtractor(box, **values)
    if kwargs == {"n_symbols": 0}:
        wrong = spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32)
        with pytest.raises(ValueError, match="does not match"):
            AssetSetFeatureExtractor(
                wrong,
                n_symbols=2,
                per_symbol_width=2,
                global_width=3,
                active_column=1,
            )


def test_action_ablation_contracts_cover_all_variants() -> None:
    baseline = ActionExperimentSpec(ActionAblation.BASELINE_ONLY, 3)
    assert not baseline.policy_enabled
    assert baseline.action_spec.size == 3
    assert not baseline.accept_legacy_actions
    assert baseline.direct_symbol_basis() is None

    legacy = ActionExperimentSpec(ActionAblation.TREND_LEGACY, 3)
    assert legacy.policy_enabled and legacy.accept_legacy_actions
    alpha_legacy = ActionExperimentSpec(ActionAblation.TREND_ALPHA_LEGACY, 3)
    assert alpha_legacy.action_spec.alpha_enabled
    assert alpha_legacy.accept_legacy_actions

    factor4 = ActionExperimentSpec(ActionAblation.FACTORIZED_4, 3)
    factor8 = ActionExperimentSpec(ActionAblation.FACTORIZED_8, 3)
    direct = ActionExperimentSpec(ActionAblation.DIRECT_SYMBOL, 3)
    assert factor4.action_spec.n_factors == 4
    assert factor8.action_spec.n_factors == 8
    assert direct.action_spec.n_factors == 3
    np.testing.assert_array_equal(direct.direct_symbol_basis(), np.eye(3))
    with pytest.raises(ValueError, match="positive"):
        ActionExperimentSpec(ActionAblation.FACTORIZED, 0)


def test_normalizer_fits_train_only_and_preserves_passthrough() -> None:
    observations = np.array(
        [
            [1.0, 0.0, 10.0],
            [3.0, 1.0, 20.0],
            [1000.0, 0.0, 30.0],
        ],
        dtype=np.float64,
    )
    normalizer = ObservationNormalizer.fit(
        observations,
        train_start=0,
        train_end=2,
        passthrough_indices=(1,),
        dataset_id="a" * 64,
    )
    transformed = normalizer.transform(observations[2])
    batch = normalizer.transform_batch(observations)
    assert normalizer.size == 3
    assert transformed[1] == 0.0
    np.testing.assert_array_equal(batch[:, 1], observations[:, 1])
    assert normalizer.digest

    clone = ObservationNormalizer(
        mean=normalizer.mean,
        scale=normalizer.scale,
        train_start=normalizer.train_start,
        train_end=normalizer.train_end,
        passthrough_indices=normalizer.passthrough_indices,
        dataset_id=normalizer.dataset_id,
        digest=normalizer.digest,
    )
    assert clone.digest == normalizer.digest


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: ObservationNormalizer(np.array([]), np.array([]), 0, 1), "non-empty"),
        (
            lambda: ObservationNormalizer(np.ones(2), np.ones(3), 0, 1),
            "identical",
        ),
        (
            lambda: ObservationNormalizer(np.ones(2), np.array([1.0, 0.0]), 0, 1),
            "positive",
        ),
        (
            lambda: ObservationNormalizer(np.ones(2), np.ones(2), 1, 1),
            "non-empty index range",
        ),
        (
            lambda: ObservationNormalizer(
                np.ones(2), np.ones(2), 0, 1, passthrough_indices=(2,)
            ),
            "outside",
        ),
    ],
)
def test_normalizer_rejects_invalid_contracts(factory: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        factory()  # type: ignore[operator]


def test_normalizer_rejects_bad_fit_and_transform_inputs() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        ObservationNormalizer.fit(np.ones(3), train_start=0, train_end=1)
    with pytest.raises(ValueError, match="finite"):
        ObservationNormalizer.fit(np.array([[np.nan]]), train_start=0, train_end=1)
    with pytest.raises(ValueError, match="outside"):
        ObservationNormalizer.fit(np.ones((2, 2)), train_start=0, train_end=3)
    normalizer = ObservationNormalizer.fit(np.ones((2, 2)), train_start=0, train_end=2)
    with pytest.raises(ValueError, match="does not match"):
        normalizer.transform(np.ones(3))
    with pytest.raises(ValueError, match="does not match"):
        normalizer.transform_batch(np.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        normalizer.transform_batch(np.array([[np.nan, 1.0]]))


def test_reward_validation_and_legacy_compatibility_paths() -> None:
    with pytest.raises(ValueError, match="positive"):
        RewardConfig(scale=0.0)
    with pytest.raises(ValueError, match="non-negative"):
        RewardConfig(margin_deficit_weight=-1.0)
    with pytest.raises(ValueError, match="at least one"):
        RewardConfig(baseline_progressive_power=0.5)
    with pytest.raises(ValueError, match="positive"):
        RewardTracker(RewardConfig(), decision_hours=0.0)

    tracker = RewardTracker(RewardConfig())
    with pytest.raises(ValueError, match="non-negative"):
        tracker.step(
            hybrid_log_return=0.0,
            shadow_log_return=0.0,
            hybrid_drawdown=0.0,
            shadow_drawdown=0.0,
            hybrid_margin_deficit_fraction=-0.1,
        )
    assert relative_interval_reward(
        hybrid_log_return=0.01,
        shadow_log_return=0.0,
        scale=100.0,
        hybrid_terminated=False,
        shadow_terminated=False,
        hybrid_drawdown=0.0,
        shadow_drawdown=0.0,
    ) == pytest.approx(1.0)
    assert (
        relative_interval_reward(
            hybrid_log_return=0.0,
            shadow_log_return=0.0,
            scale=100.0,
            hybrid_terminated=True,
            shadow_terminated=False,
            hybrid_drawdown=0.0,
            shadow_drawdown=0.0,
        )
        == -100.0
    )
    assert (
        relative_interval_reward(
            hybrid_log_return=0.0,
            shadow_log_return=0.0,
            scale=100.0,
            hybrid_terminated=False,
            shadow_terminated=True,
            hybrid_drawdown=0.0,
            shadow_drawdown=0.0,
        )
        == 100.0
    )
