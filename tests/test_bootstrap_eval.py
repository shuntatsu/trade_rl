import numpy as np

from mars_lite.eval.bootstrap_eval import bootstrap_sharpe_difference


def test_bootstrap_sharpe_difference_outputs_ci_and_p_value():
    candidate = np.array([0.03, 0.02, 0.01, 0.02, 0.03, 0.01])
    baseline = np.array([0.01, 0.01, 0.0, 0.01, 0.01, 0.0])

    result = bootstrap_sharpe_difference(
        candidate,
        baseline,
        n_bootstrap=300,
        ci=0.90,
        seed=7,
    )

    assert {
        "mean",
        "lower_ci",
        "upper_ci",
        "p_value",
        "observed_diff",
        "n_bootstrap",
        "ci",
        "block_size",
        "method",
    }.issubset(set(result))
    assert result["n_bootstrap"] == 300
    assert result["ci"] == 0.90
    assert result["method"] == "moving_block"
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["lower_ci"] <= result["mean"] <= result["upper_ci"]
    assert result["observed_diff"] > 0.0


def test_bootstrap_sharpe_difference_discriminates_null_from_real_effect():
    # Identical series: no true difference, so the CI must straddle zero and
    # the p-value must indicate no significance (not just "a number in [0,1]").
    same = np.array([0.01, 0.02, -0.01, 0.015, 0.005, -0.005, 0.02, 0.01])
    null_result = bootstrap_sharpe_difference(same, same, n_bootstrap=500, seed=1)
    assert null_result["observed_diff"] == 0.0
    assert null_result["p_value"] == 1.0
    assert null_result["lower_ci"] <= 0.0 <= null_result["upper_ci"]

    # Clearly separated distributions: CI should exclude zero and the
    # p-value should reject the null.
    rng = np.random.default_rng(0)
    candidate = rng.normal(0.01, 0.01, 200)
    baseline = rng.normal(0.0, 0.01, 200)
    effect_result = bootstrap_sharpe_difference(
        candidate, baseline, n_bootstrap=500, seed=3
    )
    assert effect_result["observed_diff"] > 0.0
    assert effect_result["lower_ci"] > 0.0
    assert effect_result["p_value"] < 0.05


def test_bootstrap_sharpe_difference_invalid_inputs():
    candidate = np.array([0.03, 0.02, 0.01])
    baseline = np.array([0.01, 0.01, 0.0])

    # Test annualization_factor <= 0
    import pytest

    with pytest.raises(ValueError, match="annualization_factor"):
        bootstrap_sharpe_difference(candidate, baseline, annualization_factor=0)
    with pytest.raises(ValueError, match="annualization_factor"):
        bootstrap_sharpe_difference(candidate, baseline, annualization_factor=-252)

    # Test mismatched shapes
    with pytest.raises(
        ValueError, match="candidate_returns and baseline_returns must match"
    ):
        bootstrap_sharpe_difference(candidate, np.array([0.01, 0.01]))

    # Test non-1D arrays
    with pytest.raises(ValueError, match="returns must be 1D arrays"):
        bootstrap_sharpe_difference(candidate.reshape(-1, 1), baseline.reshape(-1, 1))

    # Test negative/zero bootstrap
    with pytest.raises(ValueError, match="n_bootstrap must be positive"):
        bootstrap_sharpe_difference(candidate, baseline, n_bootstrap=0)

    # Test invalid CI
    with pytest.raises(ValueError, match="ci must be between 0 and 1"):
        bootstrap_sharpe_difference(candidate, baseline, ci=1.0)


def test_bootstrap_stationary_and_sensitivity():
    from mars_lite.eval.bootstrap_eval import analyze_block_size_sensitivity

    rng = np.random.default_rng(42)
    candidate = rng.normal(0.01, 0.01, 100)
    baseline = rng.normal(0.00, 0.01, 100)

    # Stationary Bootstrap のテスト
    stat_res = bootstrap_sharpe_difference(
        candidate,
        baseline,
        n_bootstrap=200,
        method="stationary",
        block_size=10,
        seed=123,
    )
    assert stat_res["method"] == "stationary"
    assert stat_res["block_size"] == 10

    # 感度分析のテスト
    sens = analyze_block_size_sensitivity(
        candidate, baseline, block_sizes=[5, 10], n_bootstrap=100, seed=123
    )
    assert set(sens.keys()) == {5, 10}
    assert sens[5]["block_size"] == 5
    assert sens[10]["block_size"] == 10
