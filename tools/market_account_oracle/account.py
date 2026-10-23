"""Primitive account arithmetic independent of the native simulation ledger.

Callers own transition ordering and authenticate prices, accepted fills and clocks.
This calculator alone is not a replay validator or B4 admission check.
"""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction

from tools.market_account_oracle.costs import (
    FillCalculation,
    FillEconomics,
    calculate_fill,
)
from tools.market_account_oracle.evidence import ReferenceSnapshot
from tools.market_account_oracle.numbers import (
    Number,
    arithmetic_context,
    number,
    projection,
    quantity,
    vector,
)

_COMPONENTS = (
    "total_cost",
    "funding_pnl",
    "borrow_cost",
    "cash_interest",
    "dividend",
    "settlement_proceeds",
)


class ReferenceAccount:
    """Own exact inventory and separately projected monetary quantities."""

    def __init__(
        self,
        *,
        cash: Number,
        quantities: tuple[str, ...],
        marks: tuple[Number, ...],
        multipliers: tuple[Number, ...],
        peak_value: Number | None = None,
    ) -> None:
        if not isinstance(quantities, tuple) or not quantities:
            raise ValueError("requires a nonempty immutable quantity vector")
        self._quantities = tuple(quantity(value) for value in quantities)
        self._marks = vector(marks, len(quantities), positive=True)
        self._multipliers = vector(multipliers, len(quantities), positive=True)
        self._cash = number(cash)
        self._totals = dict.fromkeys(_COMPONENTS, Decimal(0))
        wealth = self._wealth()
        self._peak = wealth if peak_value is None else number(peak_value)
        if self._peak <= 0 or self._peak < wealth:
            raise ValueError(
                "requires a positive declared peak at least opening wealth"
            )
        self._signed_drawdown = Decimal(0)
        self._native_drawdown = Decimal(0)
        self._termination: str | None = None

    @property
    def quantities(self) -> tuple[Fraction, ...]:
        return self._quantities

    @property
    def cash(self) -> Decimal:
        return self._cash

    def _require_open(self) -> None:
        if self._termination is not None:
            raise ValueError("terminal account cannot consume later transitions")

    def _values(self) -> tuple[Decimal, ...]:
        return tuple(
            projection(q) * price * multiplier
            for q, price, multiplier in zip(
                self._quantities, self._marks, self._multipliers, strict=True
            )
        )

    def _wealth(self) -> Decimal:
        with localcontext(arithmetic_context()):
            return self._cash + sum(self._values(), Decimal(0))

    def _credit(self, component: str, amount: Decimal) -> Decimal:
        self._cash += amount
        self._totals[component] += amount
        return amount

    def apply_fill(
        self,
        *,
        symbol_index: int,
        reported_quantity: Number,
        execution_price: Number,
        economics: FillEconomics,
        market_notional: Number,
        quantity_capacity: Number | None = None,
        lot_count: int | None = None,
        lot_size: Number = 0,
    ) -> FillCalculation:
        self._require_open()
        if type(symbol_index) is not int or not 0 <= symbol_index < len(
            self._quantities
        ):
            raise ValueError("fill symbol is outside the account")
        reported, step, price = map(
            number, (reported_quantity, lot_size, execution_price)
        )
        if not reported or price <= 0:
            raise ValueError("fill quantity must be nonzero and price positive")
        if lot_count is None:
            if step != 0:
                raise ValueError("quantized fill requires its signed integer lots")
            accepted = Fraction(reported)
            if reported == projection(-self._quantities[symbol_index]):
                accepted = -self._quantities[symbol_index]
        else:
            if type(lot_count) is not int or not lot_count or step <= 0:
                raise ValueError(
                    "requires nonzero signed integer lots and positive step"
                )
            accepted = lot_count * Fraction(step)
        if projection(accepted) != reported:
            raise ValueError("reported fill differs from exact accepted lot evidence")
        with localcontext(arithmetic_context()):
            fill = calculate_fill(
                reported_quantity=projection(accepted),
                price=price,
                contract_multiplier=self._multipliers[symbol_index],
                economics=economics,
                market_notional=market_notional,
                quantity_capacity=quantity_capacity,
            )
            debit = projection(accepted) * price * self._multipliers[symbol_index]
            self._cash -= debit + fill.cost
            self._totals["total_cost"] += fill.cost
            quantities = list(self._quantities)
            quantities[symbol_index] += accepted
            self._quantities = tuple(quantities)
            marks = list(self._marks)
            marks[symbol_index] = price
            self._marks = tuple(marks)
            return fill

    def revalue(self, prices: tuple[Number, ...]) -> None:
        self._require_open()
        self._marks = vector(prices, len(self._quantities), positive=True)

    def apply_funding(
        self, *, price_rates: tuple[Number, ...], due: tuple[bool, ...]
    ) -> Decimal:
        self._require_open()
        rates = vector(price_rates, len(self._quantities))
        self._mask(due)
        with localcontext(arithmetic_context()):
            amount = -sum(
                (
                    projection(q) * multiplier * rate
                    for q, multiplier, rate, apply in zip(
                        self._quantities, self._multipliers, rates, due, strict=True
                    )
                    if apply
                ),
                Decimal(0),
            )
            return self._credit("funding_pnl", amount)

    def apply_borrow(
        self,
        *,
        annual_rates: tuple[Number, ...],
        year_fraction: Number,
        rate_multiplier: Number = 1,
    ) -> Decimal:
        self._require_open()
        rates = vector(annual_rates, len(self._quantities))
        fraction, multiplier = number(year_fraction), number(rate_multiplier)
        if min((*rates, fraction, multiplier)) < 0:
            raise ValueError(
                "borrow rates, year fraction and multiplier must be nonnegative"
            )
        with localcontext(arithmetic_context()):
            amount = (
                sum(
                    (
                        max(-value, Decimal(0)) * rate
                        for value, rate in zip(self._values(), rates, strict=True)
                    ),
                    Decimal(0),
                )
                * fraction
                * multiplier
            )
            self._cash -= amount
            self._totals["borrow_cost"] += amount
            return amount

    def apply_cash_interest(
        self,
        *,
        annual_rate: Number,
        year_fraction: Number,
        cash_basis: Number | None = None,
    ) -> Decimal:
        self._require_open()
        rate, fraction = number(annual_rate), number(year_fraction)
        basis = self._cash if cash_basis is None else number(cash_basis)
        if fraction < 0:
            raise ValueError("year fraction must be nonnegative")
        with localcontext(arithmetic_context()):
            return self._credit("cash_interest", basis * rate * fraction)

    def apply_split(self, factors: tuple[Number, ...]) -> None:
        self._require_open()
        split = vector(factors, len(self._quantities), positive=True)
        with localcontext(arithmetic_context()):
            self._quantities = tuple(
                q * Fraction(factor)
                for q, factor in zip(self._quantities, split, strict=True)
            )
            self._marks = tuple(
                price / factor for price, factor in zip(self._marks, split, strict=True)
            )

    def apply_dividend(self, per_unit: tuple[Number, ...]) -> Decimal:
        self._require_open()
        dividends = vector(per_unit, len(self._quantities))
        with localcontext(arithmetic_context()):
            amount = sum(
                (
                    projection(q) * multiplier * dividend
                    for q, multiplier, dividend in zip(
                        self._quantities, self._multipliers, dividends, strict=True
                    )
                ),
                Decimal(0),
            )
            return self._credit("dividend", amount)

    def _mask(self, values: tuple[bool, ...]) -> None:
        if (
            not isinstance(values, tuple)
            or len(values) != len(self._quantities)
            or any(type(value) is not bool for value in values)
        ):
            raise ValueError("requires a matching boolean mask")

    def settle(
        self,
        *,
        mask: tuple[bool, ...],
        prices: tuple[Number, ...],
        recovery: tuple[Number, ...],
    ) -> Decimal:
        self._require_open()
        self._mask(mask)
        marks = vector(prices, len(self._quantities), positive=True)
        recovered = vector(recovery, len(self._quantities))
        if any(not 0 <= item <= 1 for item in recovered):
            raise ValueError("recovery must be between zero and one")
        with localcontext(arithmetic_context()):
            exact_proceeds = sum(
                (
                    q * Fraction(price) * Fraction(multiplier) * Fraction(fraction)
                    for q, price, multiplier, fraction, apply in zip(
                        self._quantities,
                        marks,
                        self._multipliers,
                        recovered,
                        mask,
                        strict=True,
                    )
                    if apply
                ),
                Fraction(0),
            )
            amount = Decimal(exact_proceeds.numerator) / Decimal(
                exact_proceeds.denominator
            )
            self._quantities = tuple(
                Fraction(0) if apply else q
                for q, apply in zip(self._quantities, mask, strict=True)
            )
            self._marks = marks
            return self._credit("settlement_proceeds", amount)

    def terminate(self, *, mode: str) -> None:
        if mode not in {"floor_zero", "retain_debt"}:
            raise ValueError("unsupported terminal valuation mode")
        if self._termination is not None:
            if self._termination != mode:
                raise ValueError("cannot change the terminal valuation mode")
            return
        value = self._wealth()
        self._cash = max(value, Decimal(0)) if mode == "floor_zero" else value
        self._quantities = tuple(Fraction(0) for _ in self._quantities)
        self._termination = mode

    def observe(self) -> ReferenceSnapshot:
        with localcontext(arithmetic_context()):
            wealth = self._wealth()
            self._peak = max(self._peak, wealth)
            self._signed_drawdown = max(self._signed_drawdown, 1 - wealth / self._peak)
            self._native_drawdown = max(
                self._native_drawdown, 1 - max(wealth, Decimal(0)) / self._peak
            )
            return self.snapshot()

    def snapshot(self) -> ReferenceSnapshot:
        return ReferenceSnapshot(
            cash=self._cash,
            wealth=self._wealth(),
            exact_quantities=self._quantities,
            total_cost=self._totals["total_cost"],
            funding_pnl=self._totals["funding_pnl"],
            borrow_cost=self._totals["borrow_cost"],
            cash_interest=self._totals["cash_interest"],
            dividend=self._totals["dividend"],
            settlement_proceeds=self._totals["settlement_proceeds"],
            max_signed_drawdown=self._signed_drawdown,
            max_native_drawdown=self._native_drawdown,
        )
