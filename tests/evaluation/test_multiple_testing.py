from __future__ import annotations

import math

import pytest

from trade_rl.evaluation.comparison.multiple_testing import (
    SharpeMoments,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_moments_from_returns,
)

EULER_GAMMA = 0.5772156649015329


def _phi(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _phi_inverse(p: float) -> float:
    """Independent quantile: bisection on the erfc-based CDF."""

    low, high = -40.0, 40.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if _phi(mid) < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2.0


def _oracle_psr(
    sr: float, n: int, skew: float, kurt: float, benchmark: float = 0.0
) -> float:
    radicand = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr
    return _phi((sr - benchmark) * math.sqrt(n - 1) / math.sqrt(radicand))


def _oracle_emax(n_trials: int, variance: float) -> float:
    if n_trials == 1:
        return 0.0
    return math.sqrt(variance) * (
        (1.0 - EULER_GAMMA) * _phi_inverse(1.0 - 1.0 / n_trials)
        + EULER_GAMMA * _phi_inverse(1.0 - 1.0 / (n_trials * math.e))
    )


# --- literal oracles -------------------------------------------------------
# Values below were produced once with plain math (erfc + bisection quantile),
# not with the code under test. The published worked example of Bailey and
# Lopez de Prado (2014) could NOT be reproduced exactly from memory, so no
# published constant is asserted; see test_paper_style_scenario_is_in_range.


def test_probabilistic_sharpe_ratio_literal_oracles() -> None:
    assert probabilistic_sharpe_ratio(0.2, 100, 0.0, 3.0) == pytest.approx(
        0.9756019368039752, abs=1e-12
    )
    assert probabilistic_sharpe_ratio(0.1, 250, -0.5, 4.0) == pytest.approx(
        0.9375434364545412, abs=1e-12
    )
    assert probabilistic_sharpe_ratio(0.1, 250, -0.5, 4.0, 0.05) == pytest.approx(
        0.778529665488404, abs=1e-12
    )


def test_expected_maximum_sharpe_literal_oracles() -> None:
    assert expected_maximum_sharpe(2, 1.0) == pytest.approx(
        0.5197553442805937, abs=1e-12
    )
    assert expected_maximum_sharpe(10, 0.04) == pytest.approx(
        0.31491966026915, abs=1e-12
    )
    assert expected_maximum_sharpe(100, 0.01) == pytest.approx(
        0.2530602893201682, abs=1e-12
    )


def test_deflated_sharpe_ratio_literal_oracle() -> None:
    assert deflated_sharpe_ratio(0.1, 250, -0.5, 4.0, 10, 0.04) == pytest.approx(
        0.0004870784890680868, abs=1e-12
    )


def test_formulas_match_in_test_plain_math_oracle() -> None:
    for sr, n, skew, kurt, bench in (
        (0.05, 60, 0.3, 5.0, 0.0),
        (-0.08, 500, -1.2, 9.0, 0.02),
        (0.3, 30, 0.0, 3.0, 0.1),
    ):
        assert probabilistic_sharpe_ratio(sr, n, skew, kurt, bench) == pytest.approx(
            _oracle_psr(sr, n, skew, kurt, bench), abs=1e-12
        )
    for n_trials, variance in ((2, 0.5), (7, 0.02), (1000, 1e-4), (10**9, 3.0)):
        assert expected_maximum_sharpe(n_trials, variance) == pytest.approx(
            _oracle_emax(n_trials, variance), rel=1e-9, abs=1e-12
        )
    assert deflated_sharpe_ratio(0.12, 400, -0.4, 6.0, 25, 0.01) == pytest.approx(
        _oracle_psr(0.12, 400, -0.4, 6.0, _oracle_emax(25, 0.01)), abs=1e-12
    )


def test_paper_style_scenario_is_in_range() -> None:
    # Daily-frequency version of the paper's illustrative setting: annualised
    # SR 2.5, 1250 observations, skew -3, kurtosis 10, 100 trials, annualised
    # trial SR variance 0.5. Only a loose range is asserted; the paper's
    # exact figure was not reproduced.
    scale = math.sqrt(252.0)
    dsr = deflated_sharpe_ratio(2.5 / scale, 1250, -3.0, 10.0, 100, 0.5 / 252.0)
    assert 0.85 < dsr < 0.95
    assert expected_maximum_sharpe(100, 0.5 / 252.0) * scale == pytest.approx(
        1.7894, abs=1e-3
    )


# --- identities and properties --------------------------------------------


def test_single_trial_dsr_equals_psr_against_zero() -> None:
    for variance in (0.0, 0.01, 4.0):
        assert expected_maximum_sharpe(1, variance) == 0.0
        assert deflated_sharpe_ratio(
            0.1, 250, -0.5, 4.0, 1, variance
        ) == probabilistic_sharpe_ratio(0.1, 250, -0.5, 4.0, 0.0)


def test_psr_is_one_half_at_the_benchmark() -> None:
    assert probabilistic_sharpe_ratio(0.07, 120, 0.2, 4.0, 0.07) == 0.5


def test_psr_rises_with_sharpe_and_with_sample_size() -> None:
    by_sharpe = [
        probabilistic_sharpe_ratio(sr, 200, -0.3, 5.0) for sr in (0.01, 0.05, 0.1, 0.2)
    ]
    assert by_sharpe == sorted(by_sharpe)
    assert len(set(by_sharpe)) == len(by_sharpe)
    by_n = [probabilistic_sharpe_ratio(0.05, n, -0.3, 5.0) for n in (10, 50, 250, 2000)]
    assert by_n == sorted(by_n)
    assert len(set(by_n)) == len(by_n)


def test_psr_falls_as_the_benchmark_rises() -> None:
    values = [
        probabilistic_sharpe_ratio(0.1, 300, 0.0, 3.0, bench)
        for bench in (-0.05, 0.0, 0.05, 0.1)
    ]
    assert values == sorted(values, reverse=True)


def test_deflated_sharpe_falls_with_trial_count_and_variance() -> None:
    by_trials = [
        deflated_sharpe_ratio(0.15, 300, -0.2, 4.0, n, 0.01)
        for n in (1, 2, 10, 100, 10_000)
    ]
    assert by_trials == sorted(by_trials, reverse=True)
    assert len(set(by_trials)) == len(by_trials)
    by_variance = [
        deflated_sharpe_ratio(0.15, 300, -0.2, 4.0, 50, v) for v in (0.0, 0.005, 0.02)
    ]
    assert by_variance == sorted(by_variance, reverse=True)


def test_expected_maximum_is_nondecreasing_in_trials_and_scales_with_sqrt_v() -> None:
    values = [expected_maximum_sharpe(n, 0.04) for n in (1, 2, 3, 10, 1000, 10**9)]
    assert values == sorted(values)
    assert expected_maximum_sharpe(50, 0.16) == pytest.approx(
        2.0 * expected_maximum_sharpe(50, 0.04), rel=1e-12
    )
    assert expected_maximum_sharpe(50, 0.0) == 0.0


def test_return_negation_symmetry() -> None:
    sr, skew, kurt = 0.12, 0.7, 5.0
    # Negating returns flips SR and skew and leaves kurtosis unchanged.
    assert probabilistic_sharpe_ratio(-sr, 180, -skew, kurt) == pytest.approx(
        1.0 - probabilistic_sharpe_ratio(sr, 180, skew, kurt), abs=1e-12
    )


def test_return_series_scale_shift_and_order_metamorphics() -> None:
    returns = [0.01, -0.02, 0.015, 0.0, 0.03, -0.01, 0.005, 0.02]
    base = sharpe_moments_from_returns(returns)
    scaled = sharpe_moments_from_returns([3.0 * item for item in returns])
    assert scaled.sharpe == pytest.approx(base.sharpe, rel=1e-12)
    assert scaled.skew == pytest.approx(base.skew, rel=1e-12)
    assert scaled.kurtosis == pytest.approx(base.kurtosis, rel=1e-12)
    permuted = sharpe_moments_from_returns(list(reversed(returns)))
    assert permuted.sharpe == pytest.approx(base.sharpe, rel=1e-12)
    assert permuted.skew == pytest.approx(base.skew, rel=1e-12)
    negated = sharpe_moments_from_returns([-item for item in returns])
    assert negated.sharpe == pytest.approx(-base.sharpe, rel=1e-12)
    assert negated.skew == pytest.approx(-base.skew, rel=1e-12)
    assert negated.kurtosis == pytest.approx(base.kurtosis, rel=1e-12)
    assert negated.n_obs == base.n_obs == len(returns)


def test_moments_from_returns_literal_oracles() -> None:
    symmetric = sharpe_moments_from_returns((-1.0, 0.0, 1.0))
    assert symmetric == SharpeMoments(sharpe=0.0, n_obs=3, skew=0.0, kurtosis=1.5)
    # (0, 0, 0, 1): mean 1/4, population m2 3/16, sample std exactly 1/2.
    skewed = sharpe_moments_from_returns((0.0, 0.0, 0.0, 1.0))
    assert skewed.sharpe == pytest.approx(0.5, abs=1e-12)
    assert skewed.skew == pytest.approx(2.0 / math.sqrt(3.0), abs=1e-12)
    assert skewed.kurtosis == pytest.approx(7.0 / 3.0, abs=1e-12)
    assert skewed.n_obs == 4


# --- refusal paths ---------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sr": math.nan}, "sharpe"),
        ({"sr": math.inf}, "sharpe"),
        ({"sr": True}, "sharpe"),
        ({"sr": "0.1"}, "sharpe"),
        ({"n_obs": 1}, "n_obs"),
        ({"n_obs": 0}, "n_obs"),
        ({"n_obs": 10.0}, "n_obs"),
        ({"n_obs": True}, "n_obs"),
        ({"skew": math.nan}, "skew"),
        ({"kurtosis": math.inf}, "kurtosis"),
        ({"kurtosis": 0.0}, "non-excess"),
        ({"kurtosis": 0.99}, "non-excess"),
        ({"skew": 2.0, "kurtosis": 4.5}, "non-excess"),
        ({"benchmark_sr": math.nan}, "benchmark_sr"),
        ({"benchmark_sr": None}, "benchmark_sr"),
    ],
)
def test_psr_refuses_invalid_inputs(kwargs: dict[str, object], message: str) -> None:
    base: dict[str, object] = {
        "sr": 0.1,
        "n_obs": 100,
        "skew": 0.0,
        "kurtosis": 3.0,
        "benchmark_sr": 0.0,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        probabilistic_sharpe_ratio(**base)  # type: ignore[arg-type]


def test_psr_refuses_non_positive_variance_term() -> None:
    # At the Pearson floor kurt = 1 + skew**2 with skew=2 the radicand is
    # (1 - sr)**2, so sr=1 gives exactly 0.
    with pytest.raises(ValueError, match="not positive"):
        probabilistic_sharpe_ratio(1.0, 50, 2.0, 5.0)
    # Slightly off the singular point the term is positive and accepted.
    assert 0.0 <= probabilistic_sharpe_ratio(1.1, 50, 2.0, 5.0) <= 1.0


@pytest.mark.parametrize(
    ("n_trials", "variance", "message"),
    [
        (0, 0.1, "n_trials"),
        (-3, 0.1, "n_trials"),
        (2.0, 0.1, "n_trials"),
        (True, 0.1, "n_trials"),
        (10**16, 0.1, "n_trials"),
        (5, -0.1, "non-negative"),
        (5, math.nan, "trial_sr_variance"),
        (5, math.inf, "trial_sr_variance"),
        (5, None, "trial_sr_variance"),
    ],
)
def test_expected_maximum_refuses_invalid_inputs(
    n_trials: object, variance: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        expected_maximum_sharpe(n_trials, variance)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "returns",
    [
        (),
        (0.01,),
        (0.0, 0.0, 0.0),
        (0.01, 0.01, 0.01, 0.01),
        (0.01, math.nan, 0.02),
        (0.01, math.inf, 0.02),
        (0.01, True, 0.02),
        (0.01, "0.02", 0.03),
        "0.01,0.02",
        b"abc",
        0.5,
    ],
)
def test_moments_from_returns_refuses_unusable_series(returns: object) -> None:
    with pytest.raises(ValueError):
        sharpe_moments_from_returns(returns)  # type: ignore[arg-type]


def test_sharpe_moments_validates_on_construction() -> None:
    with pytest.raises(ValueError, match="kurtosis"):
        SharpeMoments(sharpe=0.1, n_obs=10, skew=0.0, kurtosis=0.0)
    with pytest.raises(ValueError, match="n_obs"):
        SharpeMoments(sharpe=0.1, n_obs=1, skew=0.0, kurtosis=3.0)
    assert SharpeMoments(sharpe=1, n_obs=10, skew=0, kurtosis=3).sharpe == 1.0
