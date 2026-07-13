from __future__ import annotations

import math

import pytest

from trade_rl.evaluation.capacity import (
    CapacityCurve,
    CapacityPoint,
    evaluate_capacity_grid,
)


def point(capital: float, *, fill: float = 1.0, excess: float = 0.01) -> CapacityPoint:
    return CapacityPoint(
        initial_capital=capital,
        total_return=0.02,
        excess_total_return=excess,
        total_cost_fraction=0.001,
        fill_ratio=fill,
        unfilled_turnover=max(0.0, 1.0 - fill),
    )


def test_capacity_grid_orders_points_and_selects_maximum_viable() -> None:
    curve = evaluate_capacity_grid(
        [1_000_000.0, 100_000.0, 500_000.0],
        lambda capital: point(
            capital,
            fill=1.0 if capital < 1_000_000.0 else 0.80,
        ),
    )
    assert tuple(item.initial_capital for item in curve.points) == (
        100_000.0,
        500_000.0,
        1_000_000.0,
    )
    assert curve.maximum_viable_capital() == 500_000.0
    assert curve.maximum_viable_capital(minimum_excess_return=0.02) is None


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"initial_capital": 0.0}, "positive"),
        ({"fill_ratio": 1.1}, "fill_ratio"),
        ({"total_cost_fraction": -0.1}, "non-negative"),
        ({"unfilled_turnover": -0.1}, "non-negative"),
        ({"total_return": math.nan}, "finite"),
    ],
)
def test_capacity_point_rejects_invalid_values(
    kwargs: dict[str, float],
    message: str,
) -> None:
    values = {
        "initial_capital": 1.0,
        "total_return": 0.0,
        "excess_total_return": 0.0,
        "total_cost_fraction": 0.0,
        "fill_ratio": 1.0,
        "unfilled_turnover": 0.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        CapacityPoint(**values)


def test_capacity_curve_and_grid_validate_order_and_identity() -> None:
    with pytest.raises(ValueError, match="contain points"):
        CapacityCurve(())
    with pytest.raises(ValueError, match="ascending"):
        CapacityCurve((point(2.0), point(1.0)))
    with pytest.raises(ValueError, match="unique"):
        evaluate_capacity_grid([1.0, 1.0], point)
    with pytest.raises(ValueError, match="finite and positive"):
        evaluate_capacity_grid([0.0], point)
    with pytest.raises(ValueError, match="mismatched"):
        evaluate_capacity_grid([1.0], lambda capital: point(capital + 1.0))
