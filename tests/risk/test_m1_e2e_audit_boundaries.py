"""Result-blind M1 safety oracles for metadata and emergency risk admission."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.risk.pretrade import (
    PreTradeRisk,
    PreTradeRiskConfig,
    RiskConstrainedTarget,
)


@pytest.mark.parametrize(
    ("target", "current"),
    [
        ([1e308], [-1e308]),
        ([1e308, 1e308], [0.0, 0.0]),
        ([-1e308, -1e308], [0.0, 0.0]),
    ],
)
def test_finite_inputs_must_not_publish_infinite_turnover(
    target: list[float], current: list[float]
) -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(max_gross=1.0, max_abs_weight=1.0, max_turnover=None)
    )
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="turnover|finite"):
            risk.constrain(
                np.asarray(target), current=np.asarray(current), drawdown=0.0
            )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("requested_turnover", math.inf),
        ("requested_turnover", math.nan),
        ("requested_turnover", -1.0),
        ("constrained_turnover", math.inf),
        ("constrained_turnover", math.nan),
        ("projection_l1", math.inf),
        ("projection_l1", -1.0),
        ("risk_scale", math.nan),
        ("risk_scale", -0.01),
        ("risk_scale", 1.01),
    ],
)
def test_direct_risk_metadata_cannot_bypass_finite_contract(
    field: str, invalid: float
) -> None:
    payload: dict[str, object] = {
        "weights": np.array([0.25]),
        "requested_turnover": 0.25,
        "constrained_turnover": 0.25,
        "was_constrained": False,
        "reasons": (),
        "risk_scale": 1.0,
    }
    payload[field] = invalid
    with pytest.raises(ValueError, match=field):
        RiskConstrainedTarget(**payload)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_mask",
    [np.array([0, 2]), np.array([0.0, 1.0]), np.array(["", "true"])],
)
def test_emergency_mask_rejects_nonboolean_truthiness(
    invalid_mask: np.ndarray,
) -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(max_gross=1.0, max_abs_weight=1.0, max_turnover=0.0)
    )
    with pytest.raises(TypeError, match="emergency_flatten_mask.*boolean"):
        risk.constrain(
            np.array([0.5, 0.25]),
            current=np.array([0.5, 0.25]),
            drawdown=0.0,
            emergency_flatten_mask=invalid_mask,
        )


def test_legitimate_emergency_and_turnover_metadata_remain_finite() -> None:
    risk = PreTradeRisk(
        PreTradeRiskConfig(max_gross=1.0, max_abs_weight=1.0, max_turnover=0.0)
    )
    result = risk.constrain(
        np.array([0.5, 0.25]),
        current=np.array([0.5, 0.25]),
        drawdown=0.0,
        emergency_flatten_mask=np.array([True, False]),
    )
    np.testing.assert_array_equal(result.weights, [0.0, 0.25])
    assert result.requested_turnover == 0.0
    assert result.constrained_turnover == 0.5
    assert result.projection_l1 == 0.5
    assert result.turnover_overridden
