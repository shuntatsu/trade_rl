from __future__ import annotations

import pytest

from trade_rl.strategies.position_intent import PositionIntent, target_weight_for_intent


def test_position_intent_maps_to_one_symmetric_budget() -> None:
    assert target_weight_for_intent(PositionIntent.SHORT, gross_budget=0.10) == pytest.approx(-0.10)
    assert target_weight_for_intent(PositionIntent.FLAT, gross_budget=0.10) == 0.0
    assert target_weight_for_intent(PositionIntent.LONG, gross_budget=0.10) == pytest.approx(0.10)


def test_position_intent_rejects_invalid_budget() -> None:
    with pytest.raises(ValueError, match="gross_budget"):
        target_weight_for_intent(PositionIntent.LONG, gross_budget=0.0)
    with pytest.raises(ValueError, match="gross_budget"):
        target_weight_for_intent(PositionIntent.LONG, gross_budget=float("nan"))
