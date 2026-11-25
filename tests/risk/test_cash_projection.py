"""Core-owned decision-mark spot/perpetual budget projections."""

from __future__ import annotations

import math
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.risk.cash_projection import (
    project_mixed_cash_budget,
    snapshot_drawdown_scale,
)
from trade_rl.risk.pretrade import PreTradeRisk, PreTradeRiskConfig
from trade_rl.risk.spot_cash import spot_cash_target_factor


def project(weights, spot=(), **changes):
    args = dict(
        spot_indices=spot,
        initial_margin_rate=0.4,
        cash_limit=1.0,
        net_exposure_limit=0.97,
        tolerance=1e-9,
    )
    args.update(changes)
    return project_mixed_cash_budget(tuple(weights), **args)


def test_mixed_carry_uses_spot_principal_plus_absolute_perp_margin():
    result = project((0.5, -0.5, 0.1, 0.25), (0, 2))
    assert result.margin_measure == pytest.approx(0.6 + 0.4 * 0.75)
    assert result.net_exposure == pytest.approx(0.35)
    assert result.factor == 1.0
    assert result.margin_factor == 1.0 and result.net_factor == 1.0


def test_margin_and_net_are_distinct_and_tighter_factor_wins():
    margin = project((0.75, -1.25), (0,))
    assert margin.margin_measure == 1.25
    assert margin.net_exposure == -0.5
    assert margin.margin_factor == 0.8 and margin.net_factor == 1.0
    assert margin.factor == 0.8
    net = project((0.5, 0.5), (0,))
    assert net.margin_measure == 0.7
    assert net.factor == net.net_factor == 0.97
    assert net.margin_factor == 1.0
    both = project((1.0, 1.0), (0,))
    assert both.margin_factor == pytest.approx(1 / 1.4)
    assert both.net_factor == 0.485 and both.factor == 0.485


# Frozen historical formulas, deliberately independent of the retired study
# packages. The original implementation-to-implementation comparison remains
# in the parent Git revision. Keeping these oracles literal permits deletion
# of those legacy study modules without weakening this public API test.
def _legacy_mixed_measure(weights):
    return float(np.maximum(weights[0::2], 0.0).sum()) + 0.4 * float(
        np.abs(weights[1::2]).sum()
    )


def _legacy_perp_factors(weights):
    measure = 0.4 * float(np.abs(weights).sum())
    net = float(weights.sum())
    margin_factor = 1.0 / measure if measure > 1.0 + 1e-9 else 1.0
    net_factor = 0.97 / net if net > 0.97 + 1e-9 else 1.0
    return min(margin_factor, net_factor), margin_factor, net_factor


def test_fixed_legacy_rosters_have_exact_projection_parity():
    rng = np.random.default_rng(819)
    for n in (1, 2, 6, 22):
        for _ in range(100):
            w = rng.uniform(-1.0, 1.0, n)
            got = project(w)
            assert (
                got.factor,
                got.margin_factor,
                got.net_factor,
            ) == _legacy_perp_factors(w)
            assert got.margin_measure == 0.4 * float(np.abs(w).sum())
            assert got.net_exposure == float(w.sum())
            if n % 2 == 0:
                w[0::2] = np.abs(w[0::2])
                mixed = project(w, tuple(range(0, n, 2)))
                assert mixed.margin_measure == _legacy_mixed_measure(w)
                m = _legacy_mixed_measure(w)
                net = float(w.sum())
                assert mixed.factor == min(
                    1.0 / m if m > 1.0 + 1e-9 else 1.0,
                    0.97 / net if net > 0.97 + 1e-9 else 1.0,
                )
                # Signed short perpetuals are not spot shorts.
                assert mixed.factor >= 0.0


def test_explicit_roster_permutation_has_same_economic_projection():
    a = project((0.6, -0.8, 0.3, 0.5), (0, 2))
    b = project((0.5, 0.3, -0.8, 0.6), (1, 3))
    assert a.margin_measure == pytest.approx(b.margin_measure)
    assert a.net_exposure == pytest.approx(b.net_exposure)
    assert a.factor == pytest.approx(b.factor)


def test_all_spot_and_zero_weights():
    assert project((0.0, 0.0), (0, 1)).factor == 1.0
    result = project((0.6, 0.6), (0, 1))
    assert result.margin_measure == pytest.approx(1.2)
    assert result.factor == pytest.approx(0.97 / 1.2)


@pytest.mark.parametrize(
    "weights,spot,changes",
    [
        ((), (), {}),
        ((1.0,), (0, 0), {}),
        ((1.0,), (1,), {}),
        ((1.0,), (True,), {}),
        ((1.0,), (-1,), {}),
        ((1.0, 2.0), (1, 0), {}),
        ((-0.1,), (0,), {}),
        ((float("nan"),), (), {}),
        ((float("inf"),), (), {}),
        ((True,), (), {}),
        ((1.0,), (), {"initial_margin_rate": 0.0}),
        ((1.0,), (), {"initial_margin_rate": 1.1}),
        ((1.0,), (), {"cash_limit": 0.0}),
        ((1.0,), (), {"net_exposure_limit": float("nan")}),
        ((1.0,), (), {"tolerance": -1.0}),
        ((1.0,), (), {"tolerance": 1.0}),
        ((1.7e308, 1.7e308), (), {}),
    ],
)
def test_invalid_weights_rosters_and_limits_fail_closed(weights, spot, changes):
    with pytest.raises((ValueError, TypeError)):
        project(weights, spot, **changes)


def test_subnormal_margin_cannot_disappear():
    with pytest.raises(ValueError, match="represent"):
        project((math.ulp(0.0),), initial_margin_rate=0.4)


def test_drawdown_scale_matches_canonical_risk_and_uses_observed_peak():
    config = PreTradeRiskConfig(drawdown_start=0.1, drawdown_stop=0.2)
    risk = PreTradeRisk(config)
    for equity, peak, saved in (
        (100.0, 100.0, 0.0),
        (85.0, 100.0, 0.0),
        (100.0, 100.0, 0.15),
        (70.0, 100.0, 0.1),
        (1e-13, 1e-13, 0.0),
        (-1.0, 100.0, 1.4),
    ):
        observed = min(1.0, max(0.0, 1.0 - equity / max(peak, 1e-12), saved))
        assert snapshot_drawdown_scale(
            portfolio_value=equity,
            peak_value=peak,
            max_drawdown=saved,
            drawdown_start=0.1,
            drawdown_stop=0.2,
        ) == risk.risk_scale(observed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("portfolio_value", float("nan")),
        ("peak_value", -1.0),
        ("max_drawdown", -1.0),
        ("drawdown_start", True),
        ("drawdown_stop", 0.05),
    ],
)
def test_invalid_drawdown_snapshot_fails_closed(field, value):
    args = dict(
        portfolio_value=100.0,
        peak_value=100.0,
        max_drawdown=0.0,
        drawdown_start=0.1,
        drawdown_stop=0.2,
    )
    args[field] = value
    with pytest.raises((ValueError, TypeError)):
        snapshot_drawdown_scale(**args)


@pytest.mark.parametrize(
    "weight",
    [
        Fraction(-1, 10**400),
        Fraction(1, 3),
        2**53 + 1,
        np.uint64(2**53 + 1),
        np.timedelta64(1, "ns"),
        np.bool_(True),
    ],
)
def test_non_native_or_lossy_weights_refused(weight):
    with pytest.raises((ValueError, TypeError)):
        project((weight,), (0,))


def test_wide_real_cannot_round_negative_spot_or_over_limit_to_valid():
    if np.dtype(np.longdouble).itemsize <= 8:
        pytest.skip("platform longdouble is binary64")
    tiny_negative = -np.nextafter(np.longdouble(0), np.longdouble(1))
    above_one = np.nextafter(np.longdouble(1), np.longdouble(2))
    with pytest.raises((ValueError, TypeError)):
        project((tiny_negative,), (0,))
    with pytest.raises((ValueError, TypeError)):
        project((0.5,), initial_margin_rate=above_one)
    with pytest.raises((ValueError, TypeError)):
        snapshot_drawdown_scale(
            portfolio_value=1.0,
            peak_value=1.0,
            max_drawdown=tiny_negative,
            drawdown_start=0.1,
            drawdown_stop=0.2,
        )


def test_mixed_projection_refuses_scalar_subclass_with_overridden_conversion():
    class LossyFloat(np.float64):
        def __float__(self):
            return 0.0

    with pytest.raises(TypeError, match="native real scalar"):
        project((LossyFloat(-0.5),), (0,))


def test_supported_native_scalar_types_keep_exact_binary64_value():
    result = project((np.float32(0.5), np.float64(-0.5), np.int32(0)), (0, 2))
    assert result.margin_measure == 0.7
    assert result.net_exposure == 0.0


@pytest.mark.parametrize("api", ["mixed", "spot", "drawdown"])
@pytest.mark.parametrize("spoof", ["dtype", "class", "metaclass"])
def test_cash_boundaries_reject_foreign_scalar_getters(api, spoof):
    accesses = []

    class LossyFloat(np.float64):
        @property
        def dtype(self):
            accesses.append("dtype")
            return SimpleNamespace(type=type(self), itemsize=8)

        def __float__(self):
            accesses.append("float")
            return 0.0

    class Foreign:
        @property
        def __class__(self):
            accesses.append("class")
            raise RuntimeError("foreign getter must not run")

    class ForeignMeta(type):
        def __eq__(self, other):
            accesses.append("type equality")
            raise RuntimeError("foreign metaclass comparison must not run")

    class ForeignType(metaclass=ForeignMeta):
        pass

    if spoof == "dtype":
        value = LossyFloat(-0.5)
    elif spoof == "class":
        value = Foreign()
    else:
        value = ForeignType()
    with pytest.raises(TypeError, match="native real scalar"):
        if api == "mixed":
            project((value,), (0,))
        elif api == "spot":
            spot_cash_target_factor(
                (0.0,), (value,), cash_fraction=1.0, gross_limit=1.0, reserve_rate=0.0
            )
        else:
            snapshot_drawdown_scale(
                portfolio_value=1.0,
                peak_value=1.0,
                max_drawdown=value,
                drawdown_start=0.1,
                drawdown_stop=0.2,
            )
    assert not accesses
