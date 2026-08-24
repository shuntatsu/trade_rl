from __future__ import annotations

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v4 import CausalAlphaV4TargetConfig
from trade_rl.learning.causal_alpha_v5 import (
    CausalAlphaV5SelectiveForecast,
    V5SelectiveState,
    _v5_transition_reason,
    causal_alpha_v5_target_path,
)


def _forecast(state: V5SelectiveState, rows: int = 1) -> CausalAlphaV5SelectiveForecast:
    values = np.full(rows, 0.20)
    return CausalAlphaV5SelectiveForecast(
        symbol="BTCUSDT",
        decision_indices=np.arange(rows),
        slow_return_raw=values,
        slow_direction_raw=values,
        slow_uncertainty_raw=np.zeros(rows),
        slow_return_calibrated=values,
        slow_uncertainty_calibrated=np.zeros(rows),
        return_confidence=np.ones(rows),
        direction_confidence=np.ones(rows),
        selective_confidence=np.ones(rows),
        execution_hurdle=np.zeros(rows),
        actionable_mask=np.full(rows, state is not V5SelectiveState.UNACTIONABLE),
        active_mask=np.full(rows, state is V5SelectiveState.ACTIVE),
        states=(state,) * rows,
        v4_forecast_digest="1" * 64,
        calibration_fit_digest="2" * 64,
    )


def _path(
    state: V5SelectiveState,
    *,
    initial: float = 0.0,
    p4: float = 0.20,
    d4: float = 1.0,
    liquidity_cap: float = 1.0,
    risk_cap: float = 1.0,
    rows: int = 1,
):
    return causal_alpha_v5_target_path(
        _forecast(state, rows),
        np.full(rows, p4),
        direction_score_4h=np.full(rows, d4),
        uncertainty_4h=np.zeros(rows),
        one_way_cost_rates=np.zeros(rows),
        liquidity_weight_caps=np.full(rows, liquidity_cap),
        risk_weight_caps=np.full(rows, risk_cap),
        config=CausalAlphaV4TargetConfig(),
        initial_weight=initial,
    )


@pytest.mark.parametrize(
    ("previous", "selected", "reason"),
    [
        (0.0, 0.0, "hold_flat"),
        (0.1, 0.1, "hold_position"),
        (0.0, 0.1, "entry"),
        (0.1, 0.2, "add"),
        (0.2, 0.1, "reduce"),
        (0.1, 0.0, "exit"),
        (0.1, -0.1, "flip"),
    ],
)
def test_v5_transition_reasons(previous: float, selected: float, reason: str) -> None:
    assert _v5_transition_reason(previous, selected) == reason


@pytest.mark.parametrize(
    ("state", "reason"),
    [
        (V5SelectiveState.UNACTIONABLE, "unactionable_hold"),
        (V5SelectiveState.CONFIDENCE_ABSTAIN, "confidence_abstain"),
        (V5SelectiveState.DIRECTION_DISAGREEMENT, "direction_disagreement_hold"),
        (V5SelectiveState.EDGE_BELOW_HURDLE, "edge_below_hurdle_hold"),
    ],
)
def test_v5_inactive_from_flat_cannot_enter(
    state: V5SelectiveState, reason: str
) -> None:
    path = _path(state)
    assert path.targets[0] == 0.0
    assert path.reasons == (reason,)


def test_v5_inactive_position_cannot_add_or_flip_but_can_reduce() -> None:
    path = _path(
        V5SelectiveState.CONFIDENCE_ABSTAIN,
        initial=0.10,
        p4=-0.20,
        d4=-1.0,
    )
    assert 0.0 <= path.targets[0] <= 0.10
    assert path.reasons[0] in {"reduce", "exit", "confidence_abstain"}


def test_v5_active_entry_uses_v4_fast_bound_and_counts_reason() -> None:
    path = _path(V5SelectiveState.ACTIVE)
    assert path.targets[0] > 0.0
    assert path.reasons[0] == "entry"
    assert abs(path.fast_deviations[0]) <= 0.05
    assert dict(path.reason_counts) == {"entry": 1}


def test_v5_cadence_hold_is_an_operational_override() -> None:
    path = _path(V5SelectiveState.ACTIVE, rows=2)
    assert path.reasons[1] == "cadence_hold"


def test_v5_liquidity_and_risk_projection_override_inactivity() -> None:
    liquidity = _path(
        V5SelectiveState.CONFIDENCE_ABSTAIN,
        initial=0.8,
        liquidity_cap=0.2,
        risk_cap=0.1,
    )
    assert liquidity.targets[0] == 0.2
    assert liquidity.reasons[0] == "liquidity_deleverage"
    risk = _path(V5SelectiveState.CONFIDENCE_ABSTAIN, initial=0.8, risk_cap=0.2)
    assert risk.targets[0] == 0.2
    assert risk.reasons[0] == "risk_projection"


def test_v5_target_rejects_malformed_reason_evidence() -> None:
    path = _path(V5SelectiveState.ACTIVE)
    values = {
        name: getattr(path, name)
        for name in path.__dataclass_fields__
        if name not in {"reason_counts", "digest"}
    }
    with pytest.raises(ValueError, match="reason counts"):
        type(path)(**values, reason_counts=(("entry", 2),))
