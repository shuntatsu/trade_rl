from __future__ import annotations

import pytest

from trade_rl.strategies.position_duration import (
    constrain_intent_for_minimum_hold,
    next_position_age_bars,
)
from trade_rl.strategies.position_intent import PositionIntent


def test_position_age_starts_at_first_fill_and_survives_partial_reductions() -> None:
    assert (
        next_position_age_bars(
            0,
            previous_quantity=0.0,
            filled_quantity=0.0,
        )
        == 0
    )
    age = next_position_age_bars(0, previous_quantity=0.0, filled_quantity=0.5)

    assert age == 1
    assert (
        next_position_age_bars(
            age,
            previous_quantity=0.5,
            filled_quantity=1.0,
        )
        == 2
    )
    assert (
        next_position_age_bars(
            2,
            previous_quantity=1.0,
            filled_quantity=0.75,
        )
        == 3
    )
    assert (
        next_position_age_bars(
            3,
            previous_quantity=0.75,
            filled_quantity=0.75,
        )
        == 4
    )


@pytest.mark.parametrize(
    ("previous_quantity", "filled_quantity", "expected_age"),
    [
        (1.0, 0.0, 0),
        (1.0, -0.25, 1),
        (-1.0, 0.25, 1),
        (0.0, 0.0, 0),
    ],
)
def test_position_age_resets_on_flat_and_restarts_after_reversal(
    previous_quantity: float,
    filled_quantity: float,
    expected_age: int,
) -> None:
    assert (
        next_position_age_bars(
            12,
            previous_quantity=previous_quantity,
            filled_quantity=filled_quantity,
        )
        == expected_age
    )


def test_minimum_hold_freezes_filled_quantity_and_cancels_unfilled_remainder() -> None:
    decision = constrain_intent_for_minimum_hold(
        PositionIntent.FLAT,
        current_quantity=0.375,
        position_age_bars=23,
        minimum_hold_bars=24,
    )

    assert decision.requested_intent is PositionIntent.FLAT
    assert decision.effective_intent is PositionIntent.LONG
    assert decision.target_quantity_override == 0.375
    assert decision.suppressed is True


def test_minimum_hold_suppresses_same_side_target_changes_until_boundary() -> None:
    decision = constrain_intent_for_minimum_hold(
        PositionIntent.LONG,
        current_quantity=-0.125,
        position_age_bars=23,
        minimum_hold_bars=24,
    )

    assert decision.effective_intent is PositionIntent.SHORT
    assert decision.target_quantity_override == -0.125
    assert decision.suppressed is True


def test_minimum_hold_unlocks_at_exact_age_boundary() -> None:
    decision = constrain_intent_for_minimum_hold(
        PositionIntent.FLAT,
        current_quantity=0.375,
        position_age_bars=24,
        minimum_hold_bars=24,
    )

    assert decision.effective_intent is PositionIntent.FLAT
    assert decision.target_quantity_override is None
    assert decision.suppressed is False


def test_flat_position_does_not_suppress_entry_action() -> None:
    decision = constrain_intent_for_minimum_hold(
        PositionIntent.LONG,
        current_quantity=0.0,
        position_age_bars=0,
        minimum_hold_bars=24,
    )

    assert decision.effective_intent is PositionIntent.LONG
    assert decision.target_quantity_override is None
    assert decision.suppressed is False
