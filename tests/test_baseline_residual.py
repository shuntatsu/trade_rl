import numpy as np
import pytest

from mars_lite.trading.baseline_residual import BaselineResidualComposer
from mars_lite.trading.trend_family import TrendTargets


def _trends() -> TrendTargets:
    return TrendTargets(
        fast=np.array([0.7, -0.3]),
        base=np.array([0.5, -0.5]),
        slow=np.array([0.3, -0.7]),
    )


def test_identity_action_returns_base_trend() -> None:
    result = BaselineResidualComposer().compose(
        np.array([0.0, 0.0]), _trends(), np.array([0.5, -0.5])
    )

    np.testing.assert_allclose(result.proposal, _trends().base, atol=1e-12)
    assert result.alpha_budget == 0.0


def test_trend_action_endpoints_select_fast_and_slow() -> None:
    composer = BaselineResidualComposer()

    fast = composer.compose(np.array([1.0, 0.0]), _trends(), np.zeros(2))
    slow = composer.compose(np.array([-1.0, 0.0]), _trends(), np.zeros(2))

    np.testing.assert_allclose(fast.trend_weights, _trends().fast)
    np.testing.assert_allclose(slow.trend_weights, _trends().slow)


def test_alpha_action_uses_at_most_thirty_percent_budget() -> None:
    composer = BaselineResidualComposer(alpha_budget_max=0.30)
    alpha = np.array([-0.5, 0.5])

    positive = composer.compose(np.array([0.0, 1.0]), _trends(), alpha)
    negative = composer.compose(np.array([0.0, -1.0]), _trends(), alpha)

    assert positive.alpha_budget == pytest.approx(0.30)
    assert negative.alpha_budget == pytest.approx(-0.30)
    assert np.abs(positive.proposal).sum() <= 1.0 + 1e-12
    assert np.abs(negative.proposal).sum() <= 1.0 + 1e-12


def test_disabled_alpha_forces_zero_budget() -> None:
    result = BaselineResidualComposer().compose(
        np.array([0.0, 1.0]),
        _trends(),
        np.array([-0.5, 0.5]),
        alpha_enabled=False,
    )

    assert result.alpha_budget == 0.0
    np.testing.assert_allclose(result.proposal, _trends().base)


def test_invalid_action_is_rejected() -> None:
    composer = BaselineResidualComposer()
    with pytest.raises(ValueError, match="shape"):
        composer.compose(np.array([0.0]), _trends(), np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        composer.compose(np.array([np.nan, 0.0]), _trends(), np.zeros(2))
