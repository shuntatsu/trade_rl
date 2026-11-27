from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from trade_rl.strategies.allocation.sleeves import (
    SleeveCombination,
    SleeveCombinerConfig,
    SleeveCombinerError,
    SleeveMethod,
    SleeveTarget,
    VolatilityTarget,
    ZeroVariancePolicy,
    combine_sleeves,
)

SYMBOLS = ("AAA", "BBB", "CCC")


def _cov(vols: tuple[float, ...], corr: np.ndarray) -> np.ndarray:
    scale = np.diag(vols)
    return scale @ corr @ scale


def _history(
    cov: np.ndarray, *, n: int = 60, ddof: int = 1, seed: int = 0
) -> np.ndarray:
    """Rows whose sample covariance (``ddof``) is exactly ``cov``."""

    k = cov.shape[0]
    z = np.random.default_rng(seed).standard_normal((n, k))
    z -= z.mean(axis=0)
    sample = z.T @ z / (n - ddof)
    z = z @ np.linalg.inv(np.linalg.cholesky(sample)).T
    return z @ np.linalg.cholesky(cov).T + np.linspace(0.001, 0.002, k)


def _sleeves(
    n: int, targets: list[tuple[float, ...]] | None = None
) -> list[SleeveTarget]:
    if targets is None:
        targets = [
            tuple(float(v) for v in np.eye(len(SYMBOLS))[i % len(SYMBOLS)])
            for i in range(n)
        ]
    return [
        SleeveTarget(name=f"s{i}", symbols=SYMBOLS, weights=target)
        for i, target in enumerate(targets)
    ]


def _config(method: SleeveMethod, **overrides: Any) -> SleeveCombinerConfig:
    kwargs: dict[str, Any] = {
        "method": method,
        "lookback": 60,
        "ddof": 1,
        "shrinkage": 0.0,
    }
    kwargs.update(overrides)
    return SleeveCombinerConfig(**kwargs)


def _run(
    sleeves: list[SleeveTarget],
    history: np.ndarray,
    config: SleeveCombinerConfig,
) -> SleeveCombination:
    return combine_sleeves(sleeves, symbols=SYMBOLS, history=history, config=config)


def _fractions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    contribution = weights * (cov @ weights)
    return contribution / contribution.sum()


# ---------------------------------------------------------------- oracles


@pytest.mark.parametrize("rho", [-0.6, 0.0, 0.4, 0.95])
def test_two_sleeve_inverse_vol_is_inverse_sigma_for_any_correlation(
    rho: float,
) -> None:
    cov = _cov((0.01, 0.03), np.array([[1.0, rho], [rho, 1.0]]))
    result = _run(_sleeves(2), _history(cov), _config(SleeveMethod.INVERSE_VOL))
    # 1/0.01 : 1/0.03 = 3 : 1
    np.testing.assert_allclose(result.sleeve_weights, (0.75, 0.25), atol=1e-12)
    assert result.scale == 1.0
    assert result.diagnostics.iterations == 0
    assert result.diagnostics.converged


@pytest.mark.parametrize("rho", [-0.7, -0.2, 0.0, 0.5, 0.9])
def test_two_asset_erc_equals_inverse_vol_for_any_correlation(rho: float) -> None:
    # Equal contributions: s1^2 w1^2 + rho s1 s2 w1 w2 = s2^2 w2^2 + rho s1 s2 w1 w2
    # => s1 w1 = s2 w2 => w proportional to 1/sigma, independent of rho.
    cov = _cov((0.02, 0.05), np.array([[1.0, rho], [rho, 1.0]]))
    result = _run(
        _sleeves(2), _history(cov), _config(SleeveMethod.EQUAL_RISK_CONTRIBUTION)
    )
    np.testing.assert_allclose(result.sleeve_weights, (5.0 / 7.0, 2.0 / 7.0), atol=1e-9)
    fractions = result.risk_contribution_fractions
    assert fractions is not None
    np.testing.assert_allclose(fractions, (0.5, 0.5), atol=1e-9)


def test_two_asset_erc_with_unequal_budgets_has_closed_form() -> None:
    # Uncorrelated: w_i (s_i^2 w_i) / sum = b_i => w_i proportional to sqrt(b_i)/s_i.
    cov = np.diag([0.02**2, 0.05**2])
    budgets = (0.8, 0.2)
    result = _run(
        _sleeves(2),
        _history(cov),
        _config(SleeveMethod.EQUAL_RISK_CONTRIBUTION, risk_budgets=budgets),
    )
    raw = np.array([math.sqrt(0.8) / 0.02, math.sqrt(0.2) / 0.05])
    np.testing.assert_allclose(result.sleeve_weights, raw / raw.sum(), atol=1e-9)


def test_three_asset_erc_has_the_analytic_solution() -> None:
    # Unit vols; only assets 2 and 3 correlated (rho = 1/2). Asset 1 solves
    # x1^2 = 1/3; assets 2 and 3 are symmetric: y (y + y/2) = 1/3 => y^2 = 2/9.
    corr = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.5], [0.0, 0.5, 1.0]])
    cov = _cov((0.02, 0.02, 0.02), corr)
    result = _run(
        _sleeves(3), _history(cov), _config(SleeveMethod.EQUAL_RISK_CONTRIBUTION)
    )
    x1 = 1.0 / math.sqrt(3.0)
    y = math.sqrt(2.0) / 3.0
    total = x1 + 2.0 * y
    np.testing.assert_allclose(
        result.sleeve_weights, (x1 / total, y / total, y / total), atol=1e-9
    )


def test_three_asset_erc_contributions_are_equal_for_a_dense_covariance() -> None:
    corr = np.array([[1.0, 0.6, -0.2], [0.6, 1.0, 0.3], [-0.2, 0.3, 1.0]])
    cov = _cov((0.01, 0.025, 0.06), corr)
    history = _history(cov)
    result = _run(_sleeves(3), history, _config(SleeveMethod.EQUAL_RISK_CONTRIBUTION))
    weights = np.asarray(result.sleeve_weights)
    sample = np.cov(history, rowvar=False, ddof=1)
    np.testing.assert_allclose(_fractions(weights, sample), 1.0 / 3.0, atol=1e-9)
    assert np.all(weights > 0.0)
    assert math.isclose(weights.sum(), 1.0, abs_tol=1e-12)
    assert result.diagnostics.converged
    assert result.diagnostics.iterations > 0
    assert result.diagnostics.max_budget_deviation is not None
    assert result.diagnostics.max_budget_deviation <= 1e-10
    # Reported contributions are the independent w_k (Sigma w)_k.
    np.testing.assert_allclose(
        result.risk_contributions, weights * (sample @ weights), atol=1e-15
    )


def test_three_asset_erc_with_unequal_budgets_meets_the_budgets() -> None:
    corr = np.array([[1.0, 0.6, -0.2], [0.6, 1.0, 0.3], [-0.2, 0.3, 1.0]])
    cov = _cov((0.01, 0.025, 0.06), corr)
    history = _history(cov)
    budgets = (0.5, 0.3, 0.2)
    result = _run(
        _sleeves(3),
        history,
        _config(SleeveMethod.EQUAL_RISK_CONTRIBUTION, risk_budgets=budgets),
    )
    sample = np.cov(history, rowvar=False, ddof=1)
    np.testing.assert_allclose(
        _fractions(np.asarray(result.sleeve_weights), sample), budgets, atol=1e-9
    )


def test_fixed_weights_are_returned_exactly() -> None:
    cov = _cov((0.01, 0.02, 0.03), np.eye(3))
    result = _run(
        _sleeves(3),
        _history(cov),
        _config(SleeveMethod.FIXED, fixed_weights=(0.5, 0.25, 0.25)),
    )
    assert result.sleeve_weights == (0.5, 0.25, 0.25)


def _fixed_equal_pair(
    vol: float = 0.1, *, shrinkage: float = 0.0, **overrides: Any
) -> tuple[np.ndarray, SleeveCombinerConfig]:
    cov = _cov((vol, vol), np.eye(2))
    return _history(cov), _config(
        SleeveMethod.FIXED,
        fixed_weights=(0.5, 0.5),
        shrinkage=shrinkage,
        **overrides,
    )


def test_vol_target_scale_when_the_cap_does_not_bind() -> None:
    # w = (1/2, 1/2), unit-free vols 0.1, uncorrelated:
    # var = 0.25 * (0.01 + 0.01) = 0.005, vol = sqrt(0.005).
    history, base = _fixed_equal_pair()
    config = _config(
        SleeveMethod.FIXED,
        fixed_weights=(0.5, 0.5),
        vol_target=VolatilityTarget(target_vol=0.1, max_leverage=3.0),
    )
    result = _run(_sleeves(2), history, config)
    expected = 0.1 / math.sqrt(0.005)  # = sqrt(2)
    assert math.isclose(result.scale, expected, rel_tol=1e-12)
    assert math.isclose(result.scale, math.sqrt(2.0), rel_tol=1e-12)
    assert math.isclose(
        result.diagnostics.predicted_vol, math.sqrt(0.005), rel_tol=1e-12
    )
    assert not result.diagnostics.leverage_cap_binding
    assert base.vol_target is None


def test_vol_target_scale_when_the_cap_binds() -> None:
    history, _ = _fixed_equal_pair()
    config = _config(
        SleeveMethod.FIXED,
        fixed_weights=(0.5, 0.5),
        vol_target=VolatilityTarget(target_vol=0.1, max_leverage=1.2),
    )
    result = _run(_sleeves(2), history, config)
    assert result.scale == 1.2
    assert result.diagnostics.leverage_cap_binding


def test_vol_target_deleverages_when_predicted_vol_exceeds_target() -> None:
    history, _ = _fixed_equal_pair()
    config = _config(
        SleeveMethod.FIXED,
        fixed_weights=(0.5, 0.5),
        vol_target=VolatilityTarget(target_vol=0.01, max_leverage=3.0),
    )
    result = _run(_sleeves(2), history, config)
    assert math.isclose(result.scale, 0.01 / math.sqrt(0.005), rel_tol=1e-12)
    assert result.scale < 1.0


def test_annualization_scales_predicted_vol_by_sqrt_periods() -> None:
    history, _ = _fixed_equal_pair()
    config = _config(
        SleeveMethod.FIXED,
        fixed_weights=(0.5, 0.5),
        vol_target=VolatilityTarget(
            target_vol=0.5, max_leverage=10.0, periods_per_year=252.0
        ),
    )
    result = _run(_sleeves(2), history, config)
    expected_vol = math.sqrt(0.005 * 252.0)
    assert math.isclose(result.diagnostics.predicted_vol, expected_vol, rel_tol=1e-12)
    assert math.isclose(result.scale, 0.5 / expected_vol, rel_tol=1e-12)


def test_shrinkage_and_ddof_enter_the_covariance_exactly() -> None:
    # Two sleeves, vol 0.1, correlation 0.6, weights (1/2, 1/2).
    cov = _cov((0.1, 0.1), np.array([[1.0, 0.6], [0.6, 1.0]]))
    history = _history(cov, n=40, ddof=1)
    base = {"fixed_weights": (0.5, 0.5), "lookback": 40}
    vol_target = VolatilityTarget(target_vol=1.0, max_leverage=100.0)

    def predicted(**kw: Any) -> float:
        config = _config(SleeveMethod.FIXED, vol_target=vol_target, **base, **kw)
        return _run(_sleeves(2), history, config).diagnostics.predicted_vol

    # var = 0.25 * (0.01 + 0.01 + 2 * rho_eff * 0.01)
    assert math.isclose(
        predicted(ddof=1, shrinkage=0.0), math.sqrt(0.008), rel_tol=1e-9
    )
    assert math.isclose(
        predicted(ddof=1, shrinkage=0.5), math.sqrt(0.0065), rel_tol=1e-9
    )
    assert math.isclose(
        predicted(ddof=1, shrinkage=1.0), math.sqrt(0.005), rel_tol=1e-9
    )
    # ddof=0 divides by n instead of n-1: variance * (n-1)/n.
    assert math.isclose(
        predicted(ddof=0, shrinkage=0.0), math.sqrt(0.008 * 39.0 / 40.0), rel_tol=1e-9
    )


def test_erc_with_full_shrinkage_is_inverse_vol() -> None:
    corr = np.array([[1.0, 0.8, 0.5], [0.8, 1.0, 0.6], [0.5, 0.6, 1.0]])
    cov = _cov((0.01, 0.02, 0.04), corr)
    result = _run(
        _sleeves(3),
        _history(cov),
        _config(SleeveMethod.EQUAL_RISK_CONTRIBUTION, shrinkage=1.0),
    )
    inverse = np.array([100.0, 50.0, 25.0])
    np.testing.assert_allclose(
        result.sleeve_weights, inverse / inverse.sum(), atol=1e-9
    )


def _cap_fixture() -> tuple[list[SleeveTarget], np.ndarray]:
    sleeves = _sleeves(2, [(0.8, 0.4, 0.0), (0.6, -0.4, 0.8)])
    cov = _cov((0.1, 0.1), np.eye(2))
    return sleeves, _history(cov)


def test_caps_are_applied_scale_then_symbol_cap_then_gross_cap() -> None:
    sleeves, history = _cap_fixture()
    config = _config(
        SleeveMethod.FIXED,
        fixed_weights=(0.5, 0.5),
        vol_target=VolatilityTarget(target_vol=0.1, max_leverage=3.0),
        max_abs_weight=0.8,
        max_gross=1.0,
    )
    result = _run(sleeves, history, config)
    root2 = math.sqrt(2.0)
    # netted = 0.5 * ((0.8, 0.4, 0) + (0.6, -0.4, 0.8)) = (0.7, 0.0, 0.4)
    np.testing.assert_allclose(result.netted_weights, (0.7, 0.0, 0.4), atol=1e-12)
    assert math.isclose(result.scale, root2, rel_tol=1e-12)
    # scaled = (0.7 sqrt2, 0, 0.4 sqrt2) = (0.98995, 0, 0.56569)
    # symbol cap 0.8 -> (0.8, 0, 0.4 sqrt2); gross = 0.8 + 0.4 sqrt2 = 1.36569
    gross = 0.8 + 0.4 * root2
    expected = (0.8 / gross, 0.0, 0.4 * root2 / gross)
    np.testing.assert_allclose(result.combined_weights, expected, atol=1e-12)
    assert math.isclose(sum(abs(v) for v in result.combined_weights), 1.0)
    assert result.diagnostics.symbol_cap_binding
    assert result.diagnostics.gross_cap_binding
    # The documented order differs from gross-cap-first (which would give a
    # different answer), so this literal oracle discriminates the order.
    scaled = np.array([0.7 * root2, 0.0, 0.4 * root2])
    gross_first = np.clip(scaled / np.abs(scaled).sum(), -0.8, 0.8)
    assert not np.allclose(gross_first, expected)


def test_per_symbol_caps_and_no_binding_report_no_binding() -> None:
    sleeves, history = _cap_fixture()
    config = _config(
        SleeveMethod.FIXED,
        fixed_weights=(0.5, 0.5),
        max_abs_weight=(0.5, 0.5, 0.3),
        max_gross=5.0,
    )
    result = _run(sleeves, history, config)
    np.testing.assert_allclose(result.combined_weights, (0.5, 0.0, 0.3), atol=1e-15)
    assert result.diagnostics.symbol_cap_binding
    assert not result.diagnostics.gross_cap_binding
    loose = _run(
        sleeves,
        history,
        _config(SleeveMethod.FIXED, fixed_weights=(0.5, 0.5), max_abs_weight=5.0),
    )
    assert loose.combined_weights == loose.netted_weights
    assert not loose.diagnostics.symbol_cap_binding


def test_symbol_caps_clip_short_weights_symmetrically() -> None:
    sleeves = _sleeves(2, [(-1.0, 0.0, 0.0), (-0.6, 0.0, 0.0)])
    history = _history(_cov((0.1, 0.1), np.eye(2)))
    result = _run(
        sleeves,
        history,
        _config(SleeveMethod.FIXED, fixed_weights=(0.5, 0.5), max_abs_weight=0.5),
    )
    assert result.netted_weights[0] == pytest.approx(-0.8)
    assert result.combined_weights[0] == -0.5


def test_zero_variance_exclude_policy_renormalizes_the_rest() -> None:
    cov = _cov((0.01, 0.02), np.eye(2))
    history = np.column_stack([_history(cov), np.full(60, 0.003)])
    sleeves = _sleeves(3)
    iv = _run(
        sleeves,
        history,
        _config(SleeveMethod.INVERSE_VOL, zero_variance=ZeroVariancePolicy.EXCLUDE),
    )
    np.testing.assert_allclose(iv.sleeve_weights, (2 / 3, 1 / 3, 0.0), atol=1e-12)
    assert iv.diagnostics.excluded_sleeves == ("s2",)
    fixed = _run(
        sleeves,
        history,
        _config(
            SleeveMethod.FIXED,
            zero_variance=ZeroVariancePolicy.EXCLUDE,
            fixed_weights=(0.2, 0.3, 0.5),
        ),
    )
    np.testing.assert_allclose(fixed.sleeve_weights, (0.4, 0.6, 0.0), atol=1e-12)
    erc = _run(
        sleeves,
        history,
        _config(
            SleeveMethod.EQUAL_RISK_CONTRIBUTION,
            zero_variance=ZeroVariancePolicy.EXCLUDE,
        ),
    )
    np.testing.assert_allclose(erc.sleeve_weights, (2 / 3, 1 / 3, 0.0), atol=1e-9)
    assert erc.risk_contributions[2] == 0.0


def test_single_sleeve_gets_full_weight_and_scales() -> None:
    history = _history(np.array([[0.0004]]), n=60)
    sleeves = _sleeves(1, [(0.5, -0.5, 0.0)])
    for method in (SleeveMethod.INVERSE_VOL, SleeveMethod.EQUAL_RISK_CONTRIBUTION):
        result = _run(
            sleeves,
            history,
            _config(method, vol_target=VolatilityTarget(0.01, 5.0)),
        )
        assert result.sleeve_weights == (1.0,)
        assert math.isclose(result.scale, 0.5, rel_tol=1e-9)
    assert result.combined_weights == pytest.approx((0.25, -0.25, 0.0))


def test_zero_predicted_vol_is_allowed_without_a_target() -> None:
    cov = _cov((0.1, 0.1), np.eye(2))
    base = _history(cov)[:, :1]
    history = np.column_stack([base[:, 0], -base[:, 0]])
    config = _config(SleeveMethod.FIXED, fixed_weights=(0.5, 0.5))
    result = _run(_sleeves(2), history, config)
    assert result.diagnostics.predicted_vol == 0.0
    assert result.scale == 1.0
    assert result.risk_contribution_fractions is None


# --------------------------------------------------------- metamorphic


def _method_cases() -> list[dict[str, Any]]:
    return [
        {"method": SleeveMethod.FIXED, "fixed_weights": (0.5, 0.3, 0.2)},
        {"method": SleeveMethod.INVERSE_VOL},
        {"method": SleeveMethod.EQUAL_RISK_CONTRIBUTION},
        {
            "method": SleeveMethod.EQUAL_RISK_CONTRIBUTION,
            "risk_budgets": (0.5, 0.3, 0.2),
        },
    ]


def _dense() -> tuple[np.ndarray, list[SleeveTarget]]:
    corr = np.array([[1.0, 0.6, -0.2], [0.6, 1.0, 0.3], [-0.2, 0.3, 1.0]])
    cov = _cov((0.01, 0.025, 0.06), corr)
    targets = [(1.0, 0.0, 0.5), (0.0, -1.0, 0.25), (0.3, 0.3, -0.9)]
    return _history(cov), _sleeves(3, targets)


@pytest.mark.parametrize("case", _method_cases())
def test_permuting_sleeves_permutes_the_result(case: dict[str, Any]) -> None:
    history, sleeves = _dense()
    perm = [2, 0, 1]
    target = VolatilityTarget(0.02, 4.0)
    base_kwargs = dict(case)
    original = _run(sleeves, history, _config(vol_target=target, **base_kwargs))
    permuted_kwargs = dict(base_kwargs)
    for key in ("fixed_weights", "risk_budgets"):
        if key in permuted_kwargs:
            permuted_kwargs[key] = tuple(permuted_kwargs[key][i] for i in perm)
    permuted = _run(
        [sleeves[i] for i in perm],
        history[:, perm],
        _config(vol_target=target, **permuted_kwargs),
    )
    np.testing.assert_allclose(
        permuted.sleeve_weights,
        [original.sleeve_weights[i] for i in perm],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        permuted.risk_contributions,
        [original.risk_contributions[i] for i in perm],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        permuted.combined_weights, original.combined_weights, atol=1e-9
    )
    assert math.isclose(permuted.scale, original.scale, rel_tol=1e-9)
    assert permuted.sleeve_names == tuple(sleeves[i].name for i in perm)


@pytest.mark.parametrize("case", _method_cases())
@pytest.mark.parametrize("c", [0.001, 7.3, 1000.0])
def test_scaling_returns_keeps_weights_and_inverts_the_vol_scale(
    case: dict[str, Any], c: float
) -> None:
    history, sleeves = _dense()
    config = _config(vol_target=VolatilityTarget(0.5, 1e9), **case)
    base = _run(sleeves, history, config)
    scaled = _run(sleeves, history * c, config)
    np.testing.assert_allclose(scaled.sleeve_weights, base.sleeve_weights, atol=1e-9)
    assert math.isclose(scaled.scale, base.scale / c, rel_tol=1e-9)
    np.testing.assert_allclose(
        scaled.risk_contributions,
        np.asarray(base.risk_contributions) * c * c,
        rtol=1e-9,
    )


def test_data_the_caller_does_not_pass_cannot_change_the_result() -> None:
    rng = np.random.default_rng(3)
    series = rng.normal(0.0005, 0.01, size=(200, 3))
    sleeves = _sleeves(3)
    config = _config(
        SleeveMethod.EQUAL_RISK_CONTRIBUTION,
        lookback=50,
        shrinkage=0.1,
        vol_target=VolatilityTarget(0.01, 3.0),
    )
    t = 120
    baseline = _run(sleeves, series[:t], config)
    mutated = series.copy()
    mutated[t:] = rng.normal(5.0, 50.0, size=mutated[t:].shape)
    assert _run(sleeves, mutated[:t], config) == baseline
    # Rows older than the lookback window are never read either.
    older = series[:t].copy()
    older[: t - 50] = np.nan
    assert _run(sleeves, older, config) == baseline
    # ...while the newest row in the window is.
    newer = series[:t].copy()
    newer[-1, 0] += 0.05
    assert _run(sleeves, newer, config) != baseline


def test_combination_is_pure_and_does_not_mutate_inputs() -> None:
    history, sleeves = _dense()
    snapshot = history.copy()
    config = _config(SleeveMethod.EQUAL_RISK_CONTRIBUTION)
    first = _run(sleeves, history, config)
    second = _run(sleeves, history, config)
    assert first == second
    np.testing.assert_array_equal(history, snapshot)
    with pytest.raises(AttributeError):
        first.scale = 2.0  # type: ignore[misc]


# ------------------------------------------------------------- refusals


def _refuses(
    reason: str,
    sleeves: list[SleeveTarget],
    history: Any,
    config: SleeveCombinerConfig,
    symbols: tuple[str, ...] = SYMBOLS,
) -> None:
    with pytest.raises(SleeveCombinerError) as info:
        combine_sleeves(sleeves, symbols=symbols, history=history, config=config)
    assert info.value.reason == reason


def test_refuses_non_finite_history_and_weights() -> None:
    history, sleeves = _dense()
    config = _config(SleeveMethod.INVERSE_VOL)
    for bad in (np.nan, np.inf, -np.inf):
        corrupted = history.copy()
        corrupted[-1, 1] = bad
        _refuses("non_finite", sleeves, corrupted, config)
        with pytest.raises(SleeveCombinerError) as info:
            SleeveTarget("x", SYMBOLS, (0.0, bad, 0.0))
        assert info.value.reason == "non_finite"


def test_refuses_mismatched_symbols_shapes_and_sleeve_sets() -> None:
    history, sleeves = _dense()
    config = _config(SleeveMethod.INVERSE_VOL)
    other = SleeveTarget("s0", ("AAA", "BBB", "ZZZ"), (0.0, 0.0, 0.0))
    _refuses("symbol_mismatch", [other, *sleeves[1:]], history, config)
    reordered = SleeveTarget("s0", ("BBB", "AAA", "CCC"), (0.0, 0.0, 0.0))
    _refuses("symbol_mismatch", [reordered, *sleeves[1:]], history, config)
    _refuses("shape", sleeves, history[:, :2], config)
    _refuses("shape", sleeves, history[:, 0], config)
    _refuses("no_sleeves", [], history, config)
    _refuses("sleeves", [sleeves[0], sleeves[0], sleeves[2]], history, config)
    _refuses("symbols", sleeves, history, config, symbols=("AAA", "AAA", "CCC"))
    with pytest.raises(SleeveCombinerError):
        SleeveTarget("s0", SYMBOLS, (0.0, 0.0))
    with pytest.raises(TypeError):
        combine_sleeves(sleeves, symbols=list(SYMBOLS), history=history, config=config)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        combine_sleeves(sleeves, symbols=SYMBOLS, history=history, config="x")  # type: ignore[arg-type]
    _refuses("history", sleeves, [["a", "b", "c"]] * 60, config)


def test_refuses_short_history() -> None:
    history, sleeves = _dense()
    config = _config(SleeveMethod.INVERSE_VOL, lookback=60)
    _refuses("short_history", sleeves, history[:59], config)
    _refuses("short_history", sleeves, history[:0], config)
    # Exactly the lookback is enough.
    assert _run(sleeves, history[:60], config).diagnostics.n_observations == 60


def test_refuses_zero_variance_by_default() -> None:
    history, sleeves = _dense()
    flat = history.copy()
    flat[:, 1] = 0.0
    for method in SleeveMethod:
        extra = (
            {"fixed_weights": (0.5, 0.3, 0.2)} if method is SleeveMethod.FIXED else {}
        )
        _refuses("zero_variance", sleeves, flat, _config(method, **extra))
    constant = history.copy()
    constant[:, 2] = 0.1  # a non-zero constant is also zero variance
    _refuses("zero_variance", sleeves, constant, _config(SleeveMethod.INVERSE_VOL))
    everything = np.zeros_like(history)
    _refuses(
        "zero_variance",
        sleeves,
        everything,
        _config(SleeveMethod.INVERSE_VOL, zero_variance=ZeroVariancePolicy.EXCLUDE),
    )
    _refuses(
        "zero_variance",
        sleeves,
        flat,
        _config(
            SleeveMethod.FIXED,
            zero_variance=ZeroVariancePolicy.EXCLUDE,
            fixed_weights=(0.0, 1.0, 0.0),
        ),
    )


def test_refuses_zero_predicted_vol_only_when_targeting() -> None:
    cov = _cov((0.1, 0.1), np.eye(2))
    base = _history(cov)[:, 0]
    history = np.column_stack([base, -base])
    config = _config(
        SleeveMethod.FIXED,
        fixed_weights=(0.5, 0.5),
        vol_target=VolatilityTarget(0.1, 2.0),
    )
    _refuses("zero_predicted_vol", _sleeves(2), history, config)


def test_refuses_singular_covariance_for_erc_unless_shrunk() -> None:
    cov = _cov((0.01, 0.02), np.eye(2))
    base = _history(cov)
    history = np.column_stack([base[:, 0], base[:, 0] * 2.0])  # perfectly collinear
    config = _config(SleeveMethod.EQUAL_RISK_CONTRIBUTION)
    _refuses("singular_covariance", _sleeves(2), history, config)
    # Fewer observations than sleeves is also singular without shrinkage.
    wide = np.random.default_rng(1).normal(size=(3, 5))
    _refuses(
        "singular_covariance",
        _sleeves(5),
        wide,
        _config(SleeveMethod.EQUAL_RISK_CONTRIBUTION, lookback=3),
    )
    shrunk = _config(SleeveMethod.EQUAL_RISK_CONTRIBUTION, shrinkage=0.5)
    result = _run(_sleeves(2), history, shrunk)
    assert result.diagnostics.converged
    # Inverse vol never needs the off-diagonals, so it accepts the same history.
    assert _run(_sleeves(2), history, _config(SleeveMethod.INVERSE_VOL))


def test_refuses_to_return_an_unconverged_erc_solution() -> None:
    corr = np.array([[1.0, 0.9, 0.8], [0.9, 1.0, 0.85], [0.8, 0.85, 1.0]])
    cov = _cov((0.005, 0.02, 0.09), corr)
    history = _history(cov)
    sleeves = _sleeves(3)
    with pytest.raises(SleeveCombinerError, match="did not converge") as info:
        _run(
            sleeves,
            history,
            _config(SleeveMethod.EQUAL_RISK_CONTRIBUTION, erc_max_iterations=1),
        )
    assert info.value.reason == "non_convergence"
    # The same problem converges when given room, proving the cap was the cause.
    ok = _run(sleeves, history, _config(SleeveMethod.EQUAL_RISK_CONTRIBUTION))
    assert ok.diagnostics.converged
    assert ok.diagnostics.iterations > 1


def test_refuses_variance_contributions_that_cannot_be_represented() -> None:
    history, sleeves = _dense()
    # Weights and volatility are representable, but the variance contributions
    # (1e400) are not: refuse instead of returning inf.
    _refuses(
        "non_representable",
        sleeves,
        history * 1e200,
        _config(SleeveMethod.INVERSE_VOL),
    )
    _refuses(
        "non_representable",
        sleeves,
        history * 1e-200,
        _config(SleeveMethod.INVERSE_VOL),
    )


def _single(
    returns: list[list[float]],
    target: tuple[float, float],
    **overrides: Any,
) -> SleeveCombination:
    sleeve = SleeveTarget("only", ("A", "B"), target)
    config = SleeveCombinerConfig(
        method=SleeveMethod.FIXED,
        lookback=2,
        ddof=0,
        shrinkage=0.0,
        fixed_weights=(1.0,),
        **overrides,
    )
    return combine_sleeves(
        [sleeve], symbols=("A", "B"), history=np.array(returns), config=config
    )


def test_witness_gross_cap_does_not_overflow_on_huge_weights() -> None:
    result = _single([[-1.0], [1.0]], (1e308, 1e308), max_gross=1.0)
    assert result.combined_weights == (0.5, 0.5)
    assert result.diagnostics.gross_cap_binding
    # A gross of 1.5e308 under a 1.6e308 cap is untouched, not rescaled.
    loose = _single([[-1.0], [1.0]], (1e308, -0.5e308), max_gross=1.6e308)
    assert loose.combined_weights == (1e308, -0.5e308)
    assert not loose.diagnostics.gross_cap_binding
    # A gross above the float range (2e308) still shrinks proportionally.
    over = _single([[-1.0], [1.0]], (1e308, -1e308), max_gross=1.5e308)
    assert over.combined_weights == (0.75e308, -0.75e308)


def test_witness_huge_annualization_keeps_the_exact_finite_vol() -> None:
    result = _single(
        [[-2.0], [2.0]], (1.0, 1.0), vol_target=VolatilityTarget(1.0, 2.0, 1e308)
    )
    assert math.isclose(result.diagnostics.predicted_vol, 2e154, rel_tol=1e-12)
    assert math.isclose(result.scale, 5e-155, rel_tol=1e-12)
    assert result.combined_weights == pytest.approx((5e-155, 5e-155), rel=1e-12)
    assert result.risk_contributions == pytest.approx((4.0,), rel=1e-12)


def test_witness_tiny_returns_do_not_divide_by_zero() -> None:
    result = _single(
        [[-1e-100], [1e-100]],
        (1.0, 1.0),
        vol_target=VolatilityTarget(1.0, 2.0, 1e-200),
    )
    assert math.isclose(result.diagnostics.predicted_vol, 1e-200, rel_tol=1e-12)
    assert result.scale == 2.0  # 1 / 1e-200 is capped
    assert result.diagnostics.leverage_cap_binding
    assert result.combined_weights == (2.0, 2.0)
    assert result.risk_contributions == pytest.approx((1e-200,), rel=1e-12)


def test_truly_unrepresentable_results_are_refused_not_returned() -> None:
    sleeve = SleeveTarget("only", ("A", "B"), (1e308, 0.0))
    history = np.array([[-1.0], [1.0]])

    def run(target: VolatilityTarget) -> SleeveCombination:
        config = SleeveCombinerConfig(
            method=SleeveMethod.FIXED,
            lookback=2,
            ddof=0,
            shrinkage=0.0,
            fixed_weights=(1.0,),
            vol_target=target,
        )
        return combine_sleeves(
            [sleeve], symbols=("A", "B"), history=history, config=config
        )

    assert run(VolatilityTarget(1.0, 1.0)).combined_weights == (1e308, 0.0)
    # Leverage 10 on a 1e308 weight leaves the float range: refuse, never inf.
    with pytest.raises(SleeveCombinerError) as info:
        run(VolatilityTarget(10.0, 10.0))
    assert info.value.reason == "non_finite"
    # A volatility scale below the float range is refused, never returned as 0.
    with pytest.raises(SleeveCombinerError) as info:
        run(VolatilityTarget(1e-300, 1.0, 1e300))
    assert info.value.reason == "non_representable"
    # Sleeve volatilities that differ by more than the float range are refused.
    wide = np.array([[-1e200, -1e-200], [1e200, 1e-200]])
    _refuses(
        "non_representable",
        _sleeves(2),
        wide,
        _config(SleeveMethod.INVERSE_VOL, lookback=2),
    )


@pytest.mark.parametrize("case", _method_cases())
@pytest.mark.parametrize("c", [1e-150, 1e-50, 1.0, 1e50, 1e150])
def test_scale_metamorphic_across_extreme_magnitudes(
    case: dict[str, Any], c: float
) -> None:
    history, sleeves = _dense()
    config = _config(vol_target=VolatilityTarget(0.5, 1e300), **case)
    base = _run(sleeves, history, config)
    scaled = _run(sleeves, history * c, config)
    np.testing.assert_allclose(scaled.sleeve_weights, base.sleeve_weights, atol=1e-9)
    assert math.isclose(scaled.scale * c, base.scale, rel_tol=1e-9)
    assert math.isclose(
        scaled.diagnostics.predicted_vol,
        base.diagnostics.predicted_vol * c,
        rel_tol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(scaled.risk_contributions) / (c * c),
        base.risk_contributions,
        rtol=1e-9,
    )
    fractions = scaled.risk_contribution_fractions
    assert fractions is not None
    np.testing.assert_allclose(fractions, base.risk_contribution_fractions, rtol=1e-9)
    for value in (*scaled.combined_weights, scaled.scale):
        assert math.isfinite(value)


def test_mixed_sleeve_magnitudes_keep_inverse_vol_exact() -> None:
    cov = np.diag([1e-8, 1e-12])  # vols 1e-4 and 1e-6
    result = _run(_sleeves(2), _history(cov), _config(SleeveMethod.INVERSE_VOL))
    np.testing.assert_allclose(
        result.sleeve_weights, (1e-2 / 1.01, 1.0 / 1.01), rtol=1e-9
    )


@pytest.mark.parametrize(
    "bad",
    [
        np.full((60, 3), 0.01, dtype=complex),
        np.zeros((60, 3), dtype=bool),
        np.full((60, 3), "0.01"),
        np.full((60, 3), "0.01", dtype=object),
        np.full((60, 3), 0.01, dtype=object),
        [["0.01", "0.02", "0.03"]] * 60,
        [[True, False, True]] * 60,
        [[1 + 2j, 1.0, 1.0]] * 60,
    ],
)
def test_refuses_non_real_history_dtypes(bad: Any) -> None:
    _, sleeves = _dense()
    _refuses("history", sleeves, bad, _config(SleeveMethod.INVERSE_VOL))


def test_integer_history_is_accepted_as_real() -> None:
    history = np.tile(np.array([[1, 2, 3], [3, 1, 2]]), (30, 1))
    result = _run(_sleeves(3), history, _config(SleeveMethod.INVERSE_VOL))
    assert math.isclose(sum(result.sleeve_weights), 1.0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"lookback": 1},
        {"lookback": 0},
        {"lookback": 60.0},
        {"lookback": True},
        {"ddof": 2},
        {"ddof": -1},
        {"ddof": 1.0},
        {"shrinkage": -0.01},
        {"shrinkage": 1.01},
        {"shrinkage": math.nan},
        {"fixed_weights": (0.5, 0.5, 0.0)},  # only allowed with FIXED
        {"risk_budgets": (0.5, 0.5, 0.0)},  # only allowed with ERC
        {"erc_tolerance": 0.0},
        {"erc_tolerance": 1.0},
        {"erc_max_iterations": 0},
        {"max_abs_weight": 0.0},
        {"max_abs_weight": -1.0},
        {"max_abs_weight": (0.5, -0.5)},
        {"max_abs_weight": (0.5, math.inf)},
        {"max_gross": 0.0},
        {"max_gross": math.nan},
    ],
)
def test_refuses_bad_configs(overrides: dict[str, Any]) -> None:
    with pytest.raises((SleeveCombinerError, TypeError)):
        _config(SleeveMethod.INVERSE_VOL, **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"fixed_weights": (0.6, 0.6)},
        {"fixed_weights": (-0.1, 1.1)},
        {"fixed_weights": (0.5, math.nan)},
        {"fixed_weights": [0.5, 0.5]},
        {"fixed_weights": (True, False)},
    ],
)
def test_refuses_bad_fixed_configs(overrides: dict[str, Any]) -> None:
    with pytest.raises((SleeveCombinerError, TypeError)):
        _config(SleeveMethod.FIXED, **overrides)


@pytest.mark.parametrize(
    "budgets", [(0.5, 0.6), (0.0, 1.0), (-0.1, 1.1), (0.5, math.inf), [0.5, 0.5]]
)
def test_refuses_bad_risk_budgets(budgets: Any) -> None:
    with pytest.raises((SleeveCombinerError, TypeError)):
        _config(SleeveMethod.EQUAL_RISK_CONTRIBUTION, risk_budgets=budgets)


@pytest.mark.parametrize(
    "args",
    [
        (0.0, 1.0),
        (-0.1, 1.0),
        (0.1, 0.0),
        (0.1, -1.0),
        (math.nan, 1.0),
        (0.1, math.inf),
        (0.1, 1.0, 0.0),
        (True, 1.0),
    ],
)
def test_refuses_bad_vol_targets(args: tuple[Any, ...]) -> None:
    with pytest.raises((SleeveCombinerError, TypeError)):
        VolatilityTarget(*args)


def test_refuses_config_enums_and_target_types() -> None:
    with pytest.raises(TypeError):
        SleeveCombinerConfig(method="fixed", lookback=5, ddof=1, shrinkage=0.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        _config(SleeveMethod.INVERSE_VOL, zero_variance="refuse")
    with pytest.raises(TypeError):
        _config(SleeveMethod.INVERSE_VOL, vol_target=0.1)


def test_refuses_config_vectors_that_do_not_match_the_sleeves_or_symbols() -> None:
    history, sleeves = _dense()
    _refuses(
        "config",
        sleeves,
        history,
        _config(SleeveMethod.FIXED, fixed_weights=(0.5, 0.5)),
    )
    _refuses(
        "config",
        sleeves,
        history,
        _config(SleeveMethod.EQUAL_RISK_CONTRIBUTION, risk_budgets=(0.5, 0.5)),
    )
    _refuses(
        "config",
        sleeves,
        history,
        _config(SleeveMethod.INVERSE_VOL, max_abs_weight=(0.5, 0.5)),
    )


def test_refuses_bad_sleeve_targets() -> None:
    for kwargs in (
        {"name": "", "symbols": SYMBOLS, "weights": (0.0, 0.0, 0.0)},
        {"name": "x", "symbols": (), "weights": ()},
        {"name": "x", "symbols": ["A"], "weights": (0.0,)},
        {"name": "x", "symbols": ("A",), "weights": [0.0]},
        {"name": "x", "symbols": ("A", ""), "weights": (0.0, 0.0)},
        {"name": "x", "symbols": ("A",), "weights": ("0.0",)},
    ):
        with pytest.raises((SleeveCombinerError, TypeError)):
            SleeveTarget(**kwargs)
