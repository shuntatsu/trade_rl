from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.actions import ActionSpec, ActionValidationMode, BaselineResidualComposer
from trade_rl.strategies.trend import TrendTargets


def trends() -> TrendTargets:
    return TrendTargets(
        fast=np.array([0.8, -0.2]),
        base=np.array([0.5, -0.5]),
        slow=np.array([0.2, -0.8]),
    )


def test_alpha_disabled_has_no_dead_action_dimension() -> None:
    assert ActionSpec(alpha_enabled=False).names == ("fast_tilt", "slow_tilt", "risk_tilt")
    assert ActionSpec(alpha_enabled=True).size == 4


def test_zero_v2_action_is_exact_baseline_identity() -> None:
    spec = ActionSpec(alpha_enabled=True, n_factors=2)
    action = spec.parse(np.zeros(spec.size))
    result = BaselineResidualComposer().compose(
        action,
        trends(),
        np.array([0.4, -0.4]),
        alpha_enabled=True,
        factor_basis=np.array([[1.0, -1.0], [-1.0, 1.0]]),
    )
    np.testing.assert_array_equal(result.proposal, trends().base)


def test_independent_fast_slow_alpha_factor_and_risk_controls_are_composed() -> None:
    spec = ActionSpec(alpha_enabled=True, n_factors=1)
    action = spec.parse(np.array([1.0, 0.5, -0.5, 0.5, 1.0]))
    result = BaselineResidualComposer().compose(
        action,
        trends(),
        np.array([0.2, -0.2]),
        alpha_enabled=True,
        factor_basis=np.array([[0.4, -0.4]]),
    )
    assert result.target_gross <= result.raw_gross + 1e-12
    assert np.any(result.factor_component != 0.0)
    assert np.any(result.alpha_component != 0.0)


def test_strict_and_fail_closed_modes_reject_out_of_bounds_actions() -> None:
    spec = ActionSpec(validation_mode=ActionValidationMode.STRICT)
    with pytest.raises(ValueError, match="outside"):
        spec.parse(np.array([2.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="failed closed"):
        spec.parse(np.array([2.0, 0.0, 0.0]), mode=ActionValidationMode.FAIL_CLOSED)
