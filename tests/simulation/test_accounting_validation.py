from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.simulation.accounting import BookState, EconomicTerminationReason


def _book() -> BookState:
    return BookState.zero(2, 1_000.0, np.array([100.0, 50.0]))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"quantities": np.array([])}, "non-empty finite"),
        ({"quantities": np.array([0.0])}, "identical shapes"),
        ({"mark_prices": np.array([100.0, 0.0])}, "strictly positive"),
        ({"cash": math.inf}, "cash must be finite"),
        ({"peak_value": 0.0}, "peak_value must be positive"),
        ({"fill_count": -1}, "execution counters"),
        ({"borrow_cost": -1.0}, "must be non-negative"),
        ({"maintenance_margin": 1.1}, r"within \[0, 1\]"),
        ({"insolvent": 1}, "insolvent must be a boolean"),
        ({"termination_reason": "unknown"}, "not supported"),
    ],
)
def test_book_state_rejects_invalid_constructor_state(
    kwargs: dict[str, object], match: str
) -> None:
    values: dict[str, object] = {
        "quantities": np.zeros(2),
        "cash": 1_000.0,
        "mark_prices": np.array([100.0, 50.0]),
        "peak_value": 1_000.0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=match):
        BookState(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: BookState.zero(0, 1.0),
        lambda: BookState.zero(1, 0.0),
        lambda: BookState.zero(2, 1.0, np.array([1.0])),
        lambda: BookState.from_weights(
            weights=np.array([0.0]), capital=1.0, prices=np.array([1.0, 2.0])
        ),
        lambda: BookState.from_weights(
            weights=np.array([0.0]), capital=0.0, prices=np.array([1.0])
        ),
        lambda: BookState.from_weights(
            weights=np.array([0.0]), capital=1.0, prices=np.array([1.0]), max_gross=0.0
        ),
        lambda: BookState.from_weights(
            weights=np.array([1.1]), capital=1.0, prices=np.array([1.0])
        ),
        lambda: BookState.from_weights(
            weights=np.array([0.0]),
            capital=1.0,
            prices=np.array([1.0]),
            peak_value=0.5,
        ),
    ],
)
def test_book_factories_reject_invalid_inputs(factory: object) -> None:
    with pytest.raises(ValueError):
        factory()  # type: ignore[operator]


def test_accounting_operation_validation_and_edge_properties() -> None:
    book = _book()
    with pytest.raises(ValueError):
        book.apply_split(np.array([1.0]))
    with pytest.raises(ValueError):
        book.apply_dividend(np.array([0.0]))
    with pytest.raises(ValueError):
        book.apply_cash_interest(math.inf, periods_per_year=1)
    with pytest.raises(ValueError):
        book.apply_cash_interest(0.01, periods_per_year=0)
    with pytest.raises(ValueError):
        book.settle_positions(
            mask=np.array([True]),
            prices=np.array([100.0, 50.0]),
            recovery=np.ones(2),
        )
    with pytest.raises(ValueError):
        book.settle_positions(
            mask=np.array([True, False]),
            prices=np.array([100.0, 50.0]),
            recovery=np.array([1.1, 1.0]),
        )
    with pytest.raises(ValueError):
        book.set_margin(margin_used=-1.0, maintenance_margin=0.2)
    with pytest.raises(ValueError):
        book.set_margin(margin_used=0.0, maintenance_margin=1.1)
    with pytest.raises(ValueError):
        book.set_margin(
            margin_used=0.0,
            maintenance_margin=0.2,
            maintenance_requirement=-1.0,
        )
    with pytest.raises(ValueError):
        book.revalue(np.array([100.0]))
    with pytest.raises(ValueError):
        book.execute(
            fill_prices=np.array([100.0]),
            target_quantities=np.zeros(2),
            cost_amount=0.0,
            turnover=0.0,
        )
    with pytest.raises(ValueError):
        book.execute(
            fill_prices=np.array([100.0, 50.0]),
            target_quantities=np.zeros(2),
            cost_amount=-1.0,
            turnover=0.0,
        )
    with pytest.raises(ValueError):
        book.execute(
            fill_prices=np.array([100.0, 50.0]),
            target_quantities=np.zeros(2),
            cost_amount=0.0,
            turnover=-1.0,
        )
    with pytest.raises(ValueError):
        book.charge_borrow(-1.0)
    with pytest.raises(ValueError):
        book.mark_to_market(
            mark_prices=np.array([100.0, 50.0]), funding_amount=math.inf
        )
    with pytest.raises(ValueError):
        book.mark_to_market(
            mark_prices=np.array([100.0, 50.0]),
            funding_amount=0.0,
            period_start_value=0.0,
        )


def test_negative_value_properties_and_termination_are_fail_closed() -> None:
    book = BookState(
        quantities=np.array([1.0]),
        cash=-200.0,
        mark_prices=np.array([100.0]),
        peak_value=100.0,
        insolvent=True,
        termination_reason=EconomicTerminationReason.INSOLVENCY,
    )
    np.testing.assert_array_equal(book.weights, np.zeros(1))
    assert book.cash_weight == 0.0
    assert book.margin_utilization == 1.0
    clone = book.clone()
    assert clone.termination_reason is EconomicTerminationReason.INSOLVENCY
