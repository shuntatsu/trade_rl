"""Synthetic M1 margin-state atomicity across corrupted public BookState values."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState


@pytest.mark.parametrize("poisoned_state", ["unrepresentable_position", "nan_cash"])
@pytest.mark.parametrize("explicit_requirement", [False, True])
def test_set_margin_rejects_invalid_book_before_mutating_margin_state(
    poisoned_state: str, explicit_requirement: bool
) -> None:
    """An invalid existing account must not leave partial margin mutations."""
    book = BookState(
        quantities=np.array([1.0]),
        cash=1_000.0,
        mark_prices=np.array([1e-200]),
        peak_value=1_000.0,
        contract_multipliers=np.array([1e-120]),
    )
    if poisoned_state == "unrepresentable_position":
        book.contract_multipliers[:] = 1e-200
    else:
        book.cash = math.nan
    before = (
        book.margin_used,
        book.maintenance_margin,
        book.maintenance_requirement,
        book.margin_deficit,
        book.insolvent,
        book.termination_reason,
    )
    kwargs = {"maintenance_requirement": 2.0} if explicit_requirement else {}
    with pytest.raises(
        ValueError, match="non-finite|position value cannot be represented"
    ):
        book.set_margin(margin_used=3.0, maintenance_margin=0.3, **kwargs)
    after = (
        book.margin_used,
        book.maintenance_margin,
        book.maintenance_requirement,
        book.margin_deficit,
        book.insolvent,
        book.termination_reason,
    )
    assert after == before


def test_explicit_margin_requirement_keeps_valid_large_signed_netting() -> None:
    """Validation must not outlaw representable equity from offsetting legs."""
    book = BookState(
        quantities=np.array([1.0, -1.0]),
        cash=100.0,
        mark_prices=np.array([1e308, 1e308]),
        peak_value=100.0,
    )
    book.set_margin(
        margin_used=1.0, maintenance_margin=0.25, maintenance_requirement=10.0
    )
    assert book.portfolio_value == 100.0
    assert (book.margin_used, book.maintenance_margin) == (1.0, 0.25)
    assert (book.maintenance_requirement, book.margin_deficit) == (10.0, 0.0)
    assert book.insolvent is False
    assert book.termination_reason is None
