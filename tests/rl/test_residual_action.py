from __future__ import annotations

import numpy as np
import pytest

from trade_rl.rl.actions import BaselineResidualComposer, ResidualAction
from trade_rl.strategies.trend import TrendTargets


def trends() -> TrendTargets:
    return TrendTargets(
        fast=np.array([0.8, -0.2], dtype=np.float64),
        base=np.array([0.5, -0.5], dtype=np.float64),
        slow=np.array([0.2, -0.8], dtype=np.float64),
    )


def test_zero_action_is_exact_baseline_identity() -> None:
    result = BaselineResidualComposer().compose(
        ResidualAction.from_array(np.zeros(2)),
        trends(),
        alpha=np.array([0.4, -0.4]),
        alpha_enabled=True,
    )

    np.testing.assert_array_equal(result.proposal, trends().base)
    assert result.action.trend_mix == 0.0
    assert result.action.alpha_budget == 0.0


def test_positive_trend_mix_moves_toward_fast_target() -> None:
    result = BaselineResidualComposer().compose(
        ResidualAction.from_array(np.array([1.0, 0.0])),
        trends(),
        alpha=np.zeros(2),
        alpha_enabled=False,
    )

    np.testing.assert_allclose(result.proposal, trends().fast)
    assert np.abs(result.proposal).sum() <= 1.0 + 1e-12


def test_negative_trend_mix_moves_toward_slow_target() -> None:
    result = BaselineResidualComposer().compose(
        ResidualAction.from_array(np.array([-1.0, 0.0])),
        trends(),
        alpha=np.zeros(2),
        alpha_enabled=False,
    )

    np.testing.assert_allclose(result.proposal, trends().slow)


def test_alpha_budget_is_ignored_when_alpha_is_disabled() -> None:
    result = BaselineResidualComposer().compose(
        ResidualAction.from_array(np.array([0.0, 1.0])),
        trends(),
        alpha=np.array([-0.5, 0.5]),
        alpha_enabled=False,
    )

    np.testing.assert_array_equal(result.proposal, trends().base)


def test_action_vector_requires_two_finite_values() -> None:
    with pytest.raises(ValueError, match="two"):
        ResidualAction.from_array(np.array([0.0]))
    with pytest.raises(ValueError, match="finite"):
        ResidualAction.from_array(np.array([0.0, np.nan]))


def test_action_vector_is_clipped_to_schema_bounds() -> None:
    action = ResidualAction.from_array(np.array([2.0, -3.0]))

    assert action.trend_mix == 1.0
    assert action.alpha_budget == -1.0
