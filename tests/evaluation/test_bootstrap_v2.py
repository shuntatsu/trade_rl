from __future__ import annotations

import numpy as np
import pytest

from trade_rl.evaluation.comparison.bootstrap import (
    BootstrapResult,
    moving_block_mean_test,
)


def test_bootstrap_p_value_has_finite_sample_floor() -> None:
    result = moving_block_mean_test((0.2,) * 20, n_bootstrap=99, seed=3)
    assert result.p_value >= 1 / 100


def test_bootstrap_circular_blocks_are_deterministic() -> None:
    values = (0.1, -0.05, 0.08, -0.02, 0.03, 0.01)
    assert moving_block_mean_test(
        values, n_bootstrap=100, seed=4
    ) == moving_block_mean_test(values, n_bootstrap=100, seed=4)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_short_sample_cannot_hide_nonfinite_difference(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        moving_block_mean_test((value,))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_bootstrap": True},
        {"n_bootstrap": np.bool_(True)},
        {"n_bootstrap": 1.5},
        {"n_bootstrap": 0},
        {"seed": True},
        {"seed": np.bool_(True)},
        {"seed": 0.5},
        {"seed": -1},
        {"block_size": True},
        {"block_size": np.bool_(True)},
        {"block_size": 0},
        {"block_size": 1.5},
    ],
)
def test_degenerate_sample_still_requires_integral_resampling_contract(kwargs) -> None:
    with pytest.raises(ValueError, match="integer"):
        moving_block_mean_test((0.0,), **kwargs)


def test_finite_inputs_cannot_publish_nonfinite_bootstrap_interval() -> None:
    with pytest.raises(ValueError, match="finite"):
        moving_block_mean_test(
            (1e308, -1e308, 1e308, -1e308),
            block_size=1,
            n_bootstrap=99,
            seed=3,
        )


def test_legacy_valid_bootstrap_vector_and_degenerate_policy_are_unchanged() -> None:
    assert moving_block_mean_test(
        (0.1, -0.05, 0.08, -0.02, 0.03, 0.01),
        n_bootstrap=np.int64(100),
        seed=np.int64(4),
    ) == BootstrapResult(
        p_value=0.009900990099009901,
        lower_ci=0.004124999999999999,
        upper_ci=0.045000000000000005,
        block_size=3,
    )
    assert moving_block_mean_test((0.25,)) == BootstrapResult(1.0, 0.0, 0.0, 1)
    assert moving_block_mean_test((0.0, 0.0), block_size=99) == BootstrapResult(
        1.0, 0.0, 0.0, 2
    )
    assert moving_block_mean_test(()) == BootstrapResult(1.0, 0.0, 0.0, 1)
    assert moving_block_mean_test((1e-16, -1e-16)) == BootstrapResult(1.0, 0.0, 0.0, 1)


def test_bootstrap_rejects_overflowed_observed_mean() -> None:
    with pytest.raises(ValueError, match="observed bootstrap mean must be finite"):
        moving_block_mean_test((1e308, 1e308))


def test_finite_resamples_cannot_produce_false_significance_from_centering_overflow() -> (
    None
):
    # Both the observed and every sampled mean are finite. The ordinary
    # reduction of three such means overflows; never count -inf as no extremes.
    with pytest.raises(ValueError, match="centered bootstrap means must be finite"):
        moving_block_mean_test((8e307, 8e307), n_bootstrap=3, seed=0)


def test_bootstrap_rejects_nonnative_rank_before_short_sample_fallback() -> None:
    with pytest.raises(ValueError, match="finite 1-D"):
        moving_block_mean_test(((0.1, 0.2),))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_quantile_failure_never_publishes_an_interval(monkeypatch, bad: float) -> None:
    # Fault injection at the quantile boundary, separate from resample overflow.
    def broken_quantile(means, probabilities):
        assert means.shape == (5,) and np.isfinite(means).all()
        assert probabilities == [0.025, 0.975]
        return np.array([bad, 0.1])

    monkeypatch.setattr(np, "quantile", broken_quantile)
    with pytest.raises(ValueError, match="bootstrap interval must be finite"):
        moving_block_mean_test((0.01, -0.01), n_bootstrap=5)
