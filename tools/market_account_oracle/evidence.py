"""Closed comparison against independent values, never a native P&L oracle."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction

from tools.market_account_oracle.numbers import arithmetic_context, number, quantity


@dataclass(frozen=True, slots=True)
class ReferenceSnapshot:
    cash: Decimal
    wealth: Decimal
    exact_quantities: tuple[Fraction, ...]
    total_cost: Decimal
    funding_pnl: Decimal
    borrow_cost: Decimal
    cash_interest: Decimal
    dividend: Decimal
    settlement_proceeds: Decimal
    max_signed_drawdown: Decimal
    max_native_drawdown: Decimal

    def comparison_values(self) -> dict[str, object]:
        return {
            "cash": self.cash,
            "wealth": self.wealth,
            "exact_quantities": tuple(str(q) for q in self.exact_quantities),
            "total_cost": self.total_cost,
            "funding_pnl": self.funding_pnl,
            "borrow_cost": self.borrow_cost,
            "cash_interest": self.cash_interest,
            "dividend": self.dividend,
            "settlement_proceeds": self.settlement_proceeds,
            "max_native_drawdown": self.max_native_drawdown,
        }


def assert_observed_account(
    reference: ReferenceSnapshot, observed: Mapping[str, object]
) -> None:
    expected = reference.comparison_values()
    if set(observed) != set(expected):
        raise AssertionError("observed account field closure mismatch")
    raw_quantities = observed["exact_quantities"]
    if not isinstance(raw_quantities, (list, tuple)):
        raise AssertionError("observed exact quantities are required")
    try:
        parsed = tuple(quantity(value) for value in raw_quantities)
    except ValueError as error:
        raise AssertionError("invalid observed exact quantities") from error
    if parsed != reference.exact_quantities:
        raise AssertionError("observed exact quantities differ")
    with localcontext(arithmetic_context()):
        for key, value in expected.items():
            if key == "exact_quantities":
                continue
            try:
                wanted, actual = number(value), number(observed[key])
            except ValueError as error:
                raise AssertionError(f"invalid observed {key}") from error
            tolerance = Decimal("1e-8") + Decimal("1e-12") * abs(wanted)
            if abs(actual - wanted) > tolerance:
                raise AssertionError(
                    f"observed {key} differs from independent calculation"
                )
