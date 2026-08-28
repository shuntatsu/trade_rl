from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v4 import (
    CAUSAL_ALPHA_V4_HORIZONS,
    CausalAlphaV4Forecast,
)
from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6SlowState,
    CausalAlphaV6TargetConfig,
)
from trade_rl.learning.causal_alpha_v6_target import (
    causal_alpha_v6_slow_state,
    causal_alpha_v6_target_path,
)


def _values(value: float | Sequence[float], rows: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return np.full(rows, float(array)) if array.ndim == 0 else array


def _forecast(
    *,
    p4: float | Sequence[float],
    p24: float | Sequence[float] = 0.0,
    p72: float | Sequence[float] = 0.0,
    d4: float | Sequence[float] = 1.0,
) -> CausalAlphaV4Forecast:
    candidates = (p4, p24, p72, d4)
    rows = max(np.asarray(value).size for value in candidates)
    zeros = np.zeros(rows)
    predictions = {
        "4h": _values(p4, rows),
        "24h": _values(p24, rows),
        "72h": _values(p72, rows),
    }
    directions = {
        "4h": _values(d4, rows),
        "24h": np.sign(predictions["24h"]),
        "72h": np.sign(predictions["72h"]),
    }
    digests = {horizon: str(index + 1) * 64 for index, horizon in enumerate(CAUSAL_ALPHA_V4_HORIZONS)}
    return CausalAlphaV4Forecast(
        symbol="BTCUSDT",
        decision_indices=np.arange(rows),
        beta=zeros,
        beta_available=np.ones(rows, dtype=np.bool_),
        market_predictions={horizon: zeros for horizon in CAUSAL_ALPHA_V4_HORIZONS},
        residual_predictions=predictions,
        direction_scores=directions,
        market_model_digests=digests,
        residual_model_digests=digests,
        direction_model_digests=digests,
        fit_digest="f" * 64,
    )


def _path(
    *,
    p4: float | Sequence[float],
    p24: float | Sequence[float] = 0.0,
    p72: float | Sequence[float] = 0.0,
    d4: float | Sequence[float] = 1.0,
    uncertainty: float = 0.0,
    cost: float = 0.0,
    liquidity_cap: float = 0.25,
    risk_cap: float = 0.25,
    actionable: bool = True,
    initial: float = 0.0,
    candidate: CausalAlphaV6Candidate = CausalAlphaV6Candidate.FAST_ONLY,
):
    forecast = _forecast(p4=p4, p24=p24, p72=p72, d4=d4)
    rows = forecast.decision_indices.size
    return causal_alpha_v6_target_path(
        forecast,
        uncertainty={horizon: np.full(rows, uncertainty) for horizon in CAUSAL_ALPHA_V4_HORIZONS},
        one_way_cost_rates=np.full(rows, cost),
        liquidity_weight_caps=np.full(rows, liquidity_cap),
        risk_weight_caps=np.full(rows, risk_cap),
        actionable_mask=np.full(rows, actionable),
        candidate=candidate,
        config=CausalAlphaV6TargetConfig(),
        initial_weight=initial,
    )


def test_v6_flat_entry_requires_two_consecutive_fast_confirmations() -> None:
    fast = np.full(5, 0.03)
    long_path = _path(p4=fast, d4=1.0)
    short_path = _path(p4=-fast, d4=-1.0)
    assert long_path.targets[0] == 0.0
    assert long_path.reasons[0] == "confirmation_hold"
    assert long_path.targets[-1] > 0.0
    assert short_path.targets[-1] == -long_path.targets[-1]
    assert long_path.confirmation_counts[[0, 4]].tolist() == [1, 2]


def test_v6_candidates_are_identical_while_flat() -> None:
    fast = np.full(5, 0.03)
    baseline = _path(p4=fast, p24=-0.04, p72=-0.05)
    retention = _path(
        p4=fast,
        p24=-0.04,
        p72=-0.05,
        candidate=CausalAlphaV6Candidate.FAST_SLOW_RETENTION,
    )
    np.testing.assert_array_equal(retention.targets, baseline.targets)


def test_v6_supportive_slow_context_holds_a_weak_fast_reduction() -> None:
    baseline = _path(p4=-0.005, p24=0.02, p72=0.03, d4=-1.0, initial=0.10)
    retention = _path(
        p4=-0.005,
        p24=0.02,
        p72=0.03,
        d4=-1.0,
        initial=0.10,
        candidate=CausalAlphaV6Candidate.FAST_SLOW_RETENTION,
    )
    assert baseline.targets[0] < 0.10
    assert retention.targets[0] == 0.10
    assert retention.reasons[0] == "slow_support_hold"


def test_v6_opposed_slow_context_suppresses_add_but_not_reduction() -> None:
    fast = np.full(5, 0.03)
    baseline = _path(p4=fast, p24=-0.02, p72=-0.03, initial=0.10)
    retention = _path(
        p4=fast,
        p24=-0.02,
        p72=-0.03,
        initial=0.10,
        candidate=CausalAlphaV6Candidate.FAST_SLOW_RETENTION,
    )
    assert baseline.targets[-1] > 0.10
    assert retention.targets[-1] == 0.10
    assert retention.reasons[-1] == "slow_add_suppressed"
    reducing = _path(
        p4=-0.01,
        p24=-0.02,
        p72=-0.03,
        d4=-1.0,
        initial=0.10,
        candidate=CausalAlphaV6Candidate.FAST_SLOW_RETENTION,
    )
    assert reducing.targets[0] < 0.10


def test_v6_strong_reversal_can_bypass_second_confirmation() -> None:
    path = _path(
        p4=-0.03,
        p24=0.02,
        p72=0.03,
        d4=-1.0,
        initial=0.10,
        candidate=CausalAlphaV6Candidate.FAST_SLOW_RETENTION,
    )
    assert path.targets[0] < 0.0
    assert path.reasons[0] == "flip"


def test_v6_operational_overrides_precede_proposals() -> None:
    zero_liquidity = _path(p4=0.03, initial=0.10, liquidity_cap=0.0)
    assert zero_liquidity.targets[0] == 0.0
    assert zero_liquidity.reasons[0] == "liquidity_deleverage"
    risk = _path(p4=0.03, initial=0.10, liquidity_cap=0.25, risk_cap=0.05)
    assert risk.targets[0] == 0.05
    assert risk.reasons[0] == "risk_projection"
    untradeable = _path(p4=0.03, actionable=False)
    assert untradeable.reasons[0] == "unactionable_hold"


def test_v6_direction_cost_uncertainty_and_cadence_holds_are_explicit() -> None:
    disagreement = _path(p4=0.03, d4=-1.0)
    assert disagreement.reasons[0] == "direction_disagreement_hold"
    expensive = _path(p4=0.003, cost=0.01)
    assert expensive.reasons[0] == "cost_or_uncertainty_hold"
    uncertain = _path(p4=0.01, uncertainty=0.02)
    assert uncertain.reasons[0] == "cost_or_uncertainty_hold"
    cadence = _path(p4=np.full(2, 0.03))
    assert cadence.reasons[1] == "cadence_hold"


def test_v6_target_delta_is_bounded_for_economic_proposals() -> None:
    path = _path(p4=-0.03, d4=-1.0, initial=0.10)
    assert abs(path.targets[0] - 0.10) <= 0.125


@pytest.mark.parametrize(
    ("previous", "p24", "p72", "expected"),
    [
        (0.0, 1.0, 1.0, CausalAlphaV6SlowState.FLAT),
        (0.1, 1.0, 1.0, CausalAlphaV6SlowState.SUPPORTIVE),
        (-0.1, -1.0, -1.0, CausalAlphaV6SlowState.SUPPORTIVE),
        (0.1, -1.0, -1.0, CausalAlphaV6SlowState.OPPOSED),
        (0.1, 1.0, -1.0, CausalAlphaV6SlowState.MIXED),
    ],
)
def test_v6_slow_state_is_position_relative(
    previous: float,
    p24: float,
    p72: float,
    expected: CausalAlphaV6SlowState,
) -> None:
    assert causal_alpha_v6_slow_state(previous, p24, p72) is expected


def test_v6_rejects_missing_uncertainty_horizon() -> None:
    forecast = _forecast(p4=0.03)
    with pytest.raises(ValueError, match="uncertainty horizons"):
        causal_alpha_v6_target_path(
            forecast,
            uncertainty={"4h": np.zeros(1)},
            one_way_cost_rates=np.zeros(1),
            liquidity_weight_caps=np.ones(1),
            actionable_mask=np.ones(1),
            candidate=CausalAlphaV6Candidate.FAST_ONLY,
            config=CausalAlphaV6TargetConfig(),
            initial_weight=0.0,
        )
