"""Owned binary64 account arithmetic for explicit synthetic transitions only.

This projection is independent of the native BookState. It starts from caller-
owned opening cash, exact quantities, marks, and multipliers, then consumes only
explicit account events. It is not a receipt reader, execution replay, or B4 gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import TypeAlias

import numpy as np

_SYMBOLS = 2
_TOLERANCE = 1e-12
_EXACT_TOLERANCE = Fraction("1e-12")
_MIN_EQUITY = float(np.finfo(np.float64).tiny)
_TERMINATIONS = frozenset(
    {
        "minimum_equity",
        "execution_cost_exhaustion",
        "margin_call",
        "liquidation",
        "drawdown_stop",
        "insolvency",
    }
)


def _float(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite binary64 value")
    return value


def _pair(values: object, field: str) -> tuple[object, object]:
    if type(values) is not tuple or len(values) != _SYMBOLS:
        raise ValueError(f"{field} must be an immutable two-symbol tuple")
    return values


def _float_pair(
    values: object, field: str, *, positive: bool = False
) -> tuple[float, float]:
    first, second = _pair(values, field)
    result = (_float(first, field), _float(second, field))
    if positive and any(value <= 0.0 for value in result):
        raise ValueError(f"{field} must be strictly positive")
    return result


def _fraction_pair(values: object, field: str) -> tuple[Fraction, Fraction]:
    first, second = _pair(values, field)
    if type(first) is not Fraction or type(second) is not Fraction:
        raise ValueError(f"{field} must contain exact Fractions")
    _project(first)
    _project(second)
    return first, second


def _project(quantity: Fraction) -> float:
    """Match the inward binary64 projection of the native exact-quantity view."""
    try:
        reported = float(quantity)
    except OverflowError as error:
        raise ValueError("quantity has no finite binary64 projection") from error
    if not math.isfinite(reported):
        raise ValueError("quantity has no finite binary64 projection")
    while abs(Fraction(str(reported))) > abs(quantity):
        reported = math.nextafter(reported, 0.0)
    return reported


@dataclass(frozen=True, slots=True)
class ExecutionEvent:
    deltas: tuple[Fraction, Fraction]
    fill_prices: tuple[float, float]
    cost_amount: float
    turnover: float
    valuation_prices: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class BorrowEvent:
    annual_rates: tuple[float, float]
    year_fraction: float
    rate_multiplier: float = 1.0


@dataclass(frozen=True, slots=True)
class CashInterestEvent:
    annual_rate: float
    year_fraction: float
    cash_basis: float | None = None


@dataclass(frozen=True, slots=True)
class FundingMarkEvent:
    price_rates: tuple[float, float]
    due: tuple[bool, bool]
    mark_prices: tuple[float, float]
    period_start_value: float


@dataclass(frozen=True, slots=True)
class DividendEvent:
    per_unit: tuple[float, float]


@dataclass(frozen=True, slots=True)
class SettlementEvent:
    mask: tuple[bool, bool]
    prices: tuple[float, float]
    recovery: tuple[float, float]


@dataclass(frozen=True, slots=True)
class RevaluationEvent:
    mark_prices: tuple[float, float]


@dataclass(frozen=True, slots=True)
class DrawdownObservation:
    pass


@dataclass(frozen=True, slots=True)
class TerminationEvent:
    reason: str


@dataclass(frozen=True, slots=True)
class MarginEvent:
    max_leverage: float
    maintenance_margin_rate: float
    collateral_haircut: float
    insolvency_valuation: str = "floor_zero"
    margin_mode: str = "cross"


AccountEvent: TypeAlias = (
    ExecutionEvent
    | BorrowEvent
    | CashInterestEvent
    | FundingMarkEvent
    | DividendEvent
    | SettlementEvent
    | RevaluationEvent
    | DrawdownObservation
    | TerminationEvent
    | MarginEvent
)


@dataclass(frozen=True, slots=True)
class AccountSnapshot:
    cash: float
    wealth: float
    exact_quantities: tuple[Fraction, Fraction]
    projected_quantities: tuple[float, float]
    marks: tuple[float, float]
    multipliers: tuple[float, float]
    peak_value: float
    max_drawdown: float
    total_cost: float
    funding_pnl: float
    borrow_cost: float
    cash_interest: float
    dividend: float
    settlement_proceeds: float
    turnover_total: float
    fill_count: int
    rebalance_events: int
    returns_history: tuple[float, ...]
    insolvent: bool
    termination_reason: str | None
    margin_used: float
    maintenance_margin: float
    maintenance_requirement: float
    margin_deficit: float


class UnsupportedTransition(ValueError):
    """The scalar account prototype does not implement this transition kind."""


class OwnedBinary64Account:
    """Independent two-slot binary64 view over exact caller-owned inventory.

    No native BookState or observed receipt value is accepted as an input. The
    caller owns event order and must provide the exact accepted quantity evidence.
    Unsupported event classes fail closed; this class has no bar or phase driver.
    """

    def __init__(
        self,
        *,
        cash: float,
        exact_quantities: tuple[Fraction, Fraction],
        marks: tuple[float, float],
        multipliers: tuple[float, float],
        peak_value: float | None = None,
    ) -> None:
        self._cash = _float(cash, "cash")
        self._quantities = _fraction_pair(exact_quantities, "exact_quantities")
        self._marks = _float_pair(marks, "marks", positive=True)
        self._multipliers = _float_pair(multipliers, "multipliers", positive=True)
        self._peak_value: float
        self._max_drawdown = 0.0
        self._total_cost = 0.0
        self._funding_pnl = 0.0
        self._borrow_cost = 0.0
        self._cash_interest = 0.0
        self._dividend = 0.0
        self._settlement_proceeds = 0.0
        self._turnover_total = 0.0
        self._fill_count = 0
        self._rebalance_events = 0
        self._returns_history: list[float] = []
        self._insolvent = False
        self._termination_reason: str | None = None
        self._margin_used = 0.0
        self._maintenance_margin = 0.0
        self._maintenance_requirement = 0.0
        self._margin_deficit = 0.0

        opening_wealth = self.wealth
        if peak_value is None:
            if opening_wealth <= 0.0:
                raise ValueError("non-positive opening wealth requires a positive peak")
            self._peak_value = opening_wealth
        else:
            peak = _float(peak_value, "peak_value")
            if peak <= 0.0:
                raise ValueError("peak_value must be positive")
            self._peak_value = peak
        self._refresh_economic_state()
        tolerance = max(
            _TOLERANCE, abs(opening_wealth) * 1e-12, abs(self._peak_value) * 1e-12
        )
        if not self._insolvent and self._peak_value + tolerance < opening_wealth:
            raise ValueError("peak_value cannot be below opening wealth")

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def exact_quantities(self) -> tuple[Fraction, Fraction]:
        return self._quantities

    @property
    def projected_quantities(self) -> tuple[float, float]:
        return _project(self._quantities[0]), _project(self._quantities[1])

    @property
    def position_values(self) -> np.ndarray:
        quantities = np.asarray(self.projected_quantities, dtype=np.float64)
        marks = np.asarray(self._marks, dtype=np.float64)
        multipliers = np.asarray(self._multipliers, dtype=np.float64)
        return quantities * marks * multipliers

    @property
    def wealth(self) -> float:
        # Preserve H's full two-slot ndarray multiply and reduction shape.
        return float(self._cash + self.position_values.sum())

    @property
    def max_drawdown(self) -> float:
        return self._max_drawdown

    @property
    def insolvent(self) -> bool:
        return self._insolvent

    @property
    def termination_reason(self) -> str | None:
        return self._termination_reason

    def snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            cash=self._cash,
            wealth=self.wealth,
            exact_quantities=self._quantities,
            projected_quantities=self.projected_quantities,
            marks=self._marks,
            multipliers=self._multipliers,
            peak_value=self._peak_value,
            max_drawdown=self._max_drawdown,
            total_cost=self._total_cost,
            funding_pnl=self._funding_pnl,
            borrow_cost=self._borrow_cost,
            cash_interest=self._cash_interest,
            dividend=self._dividend,
            settlement_proceeds=self._settlement_proceeds,
            turnover_total=self._turnover_total,
            fill_count=self._fill_count,
            rebalance_events=self._rebalance_events,
            returns_history=tuple(self._returns_history),
            insolvent=self._insolvent,
            termination_reason=self._termination_reason,
            margin_used=self._margin_used,
            maintenance_margin=self._maintenance_margin,
            maintenance_requirement=self._maintenance_requirement,
            margin_deficit=self._margin_deficit,
        )

    def apply(self, event: AccountEvent | object) -> float | None:
        if type(event) is ExecutionEvent:
            self._execute(event)
            return None
        if type(event) is BorrowEvent:
            return self._borrow(event)
        if type(event) is CashInterestEvent:
            return self._cash_interest_event(event)
        if type(event) is FundingMarkEvent:
            return self._funding_mark(event)
        if type(event) is DividendEvent:
            return self._dividend_event(event)
        if type(event) is SettlementEvent:
            return self._settle(event)
        if type(event) is RevaluationEvent:
            self._revalue(_float_pair(event.mark_prices, "mark_prices", positive=True))
            return None
        if type(event) is DrawdownObservation:
            self._update_drawdown()
            return None
        if type(event) is TerminationEvent:
            if type(event.reason) is not str or event.reason not in _TERMINATIONS:
                raise ValueError("unsupported economic termination reason")
            self._insolvent = True
            self._termination_reason = event.reason
            return None
        if type(event) is MarginEvent:
            self._margin(event)
            return None
        raise UnsupportedTransition("unsupported account event type")

    def _margin(self, event: MarginEvent) -> None:
        leverage = _float(event.max_leverage, "max_leverage")
        rate = _float(event.maintenance_margin_rate, "maintenance_margin_rate")
        haircut = _float(event.collateral_haircut, "collateral_haircut")
        if leverage <= 0.0 or not 0.0 <= rate <= 1.0 or not 0.0 <= haircut <= 1.0:
            raise ValueError("invalid margin configuration")
        if event.margin_mode != "cross":
            raise UnsupportedTransition("only two-symbol cross margin is supported")
        if event.insolvency_valuation not in {"floor_zero", "retain_debt"}:
            raise ValueError("unsupported insolvency valuation")

        positions = self.position_values
        value = self.wealth
        gross = float(np.abs(positions).sum())
        used = gross / leverage
        required = rate * gross
        collateral = float(
            self._cash
            + np.minimum(positions, 0.0).sum()
            + haircut * np.maximum(positions, 0.0).sum()
        )
        if not all(math.isfinite(item) for item in (gross, used, required, collateral)):
            raise ValueError("margin calculation must preserve finite values")

        # The book-level margin check precedes the executor's haircut check.
        # Reset the deficit at this check, rather than retaining an earlier peak.
        self._margin_used = used
        self._maintenance_margin = rate
        self._maintenance_requirement = required
        self._margin_deficit = max(0.0, required - max(value, 0.0))
        if value > 0.0 and required > 0.0 and value < required - _TOLERANCE:
            self._terminate("margin_call")
        self._margin_deficit = max(self._margin_deficit, required - collateral, 0.0)
        if collateral + _TOLERANCE < required:
            self._terminate("margin_call")
        if self._insolvent:
            self._cash = (
                max(value, 0.0) if event.insolvency_valuation == "floor_zero" else value
            )
            self._quantities = (Fraction(0), Fraction(0))
            self._margin_used = 0.0
            self._maintenance_margin = 0.0
            self._maintenance_requirement = 0.0

    def _execute(self, event: ExecutionEvent) -> None:
        deltas = _fraction_pair(event.deltas, "deltas")
        prices = _float_pair(event.fill_prices, "fill_prices", positive=True)
        cost = _float(event.cost_amount, "cost_amount")
        turnover = _float(event.turnover, "turnover")
        if cost < 0.0 or turnover < 0.0:
            raise ValueError("cost and turnover must be non-negative")
        marks = (
            prices
            if event.valuation_prices is None
            else _float_pair(event.valuation_prices, "valuation_prices", positive=True)
        )
        targets = (
            self._quantities[0] + deltas[0],
            self._quantities[1] + deltas[1],
        )
        target_projection = np.asarray(
            (_project(targets[0]), _project(targets[1])), dtype=np.float64
        )
        mark_array = np.asarray(marks, dtype=np.float64)
        multiplier_array = np.asarray(self._multipliers, dtype=np.float64)
        projected_deltas = (_project(deltas[0]), _project(deltas[1]))
        signed_notional = math.fsum(
            quantity * price * multiplier
            for quantity, price, multiplier in zip(
                projected_deltas, prices, self._multipliers, strict=True
            )
        )
        cash_after = self._cash - (signed_notional + cost)
        turnover_after = self._turnover_total + turnover
        cost_after = self._total_cost + cost
        with np.errstate(over="ignore", invalid="ignore"):
            equity_after = float(
                cash_after + (target_projection * mark_array * multiplier_array).sum()
            )
        if not all(
            math.isfinite(value)
            for value in (cash_after, turnover_after, cost_after, equity_after)
        ):
            raise ValueError("execution must preserve finite account values")

        filled_count = sum(abs(value) > _EXACT_TOLERANCE for value in deltas)
        value_before = self.wealth
        self._cash = cash_after
        self._quantities = targets
        self._marks = marks
        self._turnover_total = turnover_after
        self._total_cost = cost_after
        self._fill_count += filled_count
        if filled_count:
            self._rebalance_events += 1
        if cost >= max(value_before, 0.0) - _TOLERANCE:
            self._terminate("execution_cost_exhaustion")
        self._refresh_economic_state()
        self._update_drawdown()
        return None

    def _borrow(self, event: BorrowEvent) -> float:
        rates = _float_pair(event.annual_rates, "annual_rates")
        year_fraction = _float(event.year_fraction, "year_fraction")
        multiplier = _float(event.rate_multiplier, "rate_multiplier")
        if any(rate < 0.0 for rate in rates) or year_fraction < 0.0 or multiplier < 0.0:
            raise ValueError(
                "borrow rates, year fraction and multiplier must be non-negative"
            )
        amount = float(
            np.sum(
                np.maximum(-self.position_values, 0.0)
                * np.asarray(rates, dtype=np.float64)
            )
            * year_fraction
            * multiplier
        )
        if not math.isfinite(amount) or amount < 0.0:
            raise ValueError("borrow amount must be finite and non-negative")
        if amount > 0.0:
            cash_after = self._cash - amount
            borrow_after = self._borrow_cost + amount
            wealth_after = float(cash_after + self.position_values.sum())
            if not all(
                math.isfinite(value)
                for value in (cash_after, borrow_after, wealth_after)
            ):
                raise ValueError("borrow would make account non-finite")
            self._cash = cash_after
            self._borrow_cost = borrow_after
            self._refresh_economic_state()
        return amount

    def _cash_interest_event(self, event: CashInterestEvent) -> float:
        rate = _float(event.annual_rate, "annual_rate")
        fraction = _float(event.year_fraction, "year_fraction")
        if fraction < 0.0:
            raise ValueError("year_fraction must be non-negative")
        basis = (
            self._cash
            if event.cash_basis is None
            else _float(event.cash_basis, "cash_basis")
        )
        amount = float(basis * rate * fraction)
        cash_after = self._cash + amount
        if not math.isfinite(amount) or not math.isfinite(cash_after):
            raise ValueError("cash interest must preserve finite account values")
        self._cash = cash_after
        self._cash_interest += amount
        self._refresh_economic_state()
        return amount

    def _funding_mark(self, event: FundingMarkEvent) -> float:
        rates = _float_pair(event.price_rates, "price_rates")
        if (
            type(event.due) is not tuple
            or len(event.due) != _SYMBOLS
            or any(type(item) is not bool for item in event.due)
        ):
            raise ValueError("due must be a two-symbol boolean tuple")
        marks = _float_pair(event.mark_prices, "mark_prices", positive=True)
        starting = _float(event.period_start_value, "period_start_value")
        if starting <= 0.0:
            raise ValueError("period_start_value must be positive")
        quantities = np.asarray(self.projected_quantities, dtype=np.float64)
        multipliers = np.asarray(self._multipliers, dtype=np.float64)
        rate_array = np.asarray(rates, dtype=np.float64)
        due_array = np.asarray(event.due, dtype=np.float64)
        funding = -float(np.dot(quantities * multipliers, rate_array * due_array))
        if not math.isfinite(funding):
            raise ValueError("funding amount must remain finite")
        self._mark_to_market(marks, funding, starting)
        return funding

    def _mark_to_market(
        self, marks: tuple[float, float], funding: float, starting: float
    ) -> float:
        cash_after = self._cash + float(funding)
        if not math.isfinite(cash_after):
            raise ValueError("funding would make cash non-finite")
        funding_total_after = self._funding_pnl + float(funding)
        if not math.isfinite(funding_total_after):
            raise ValueError("funding cumulative total would become non-finite")
        self._revalue(marks, cash_after=cash_after, period_start_value=starting)
        value = self.wealth
        if not math.isfinite(value):
            raise ValueError("portfolio value became non-finite")
        net_return = max(value / starting - 1.0, -1.0 + 1e-12)
        self._update_drawdown()
        self._funding_pnl = funding_total_after
        self._returns_history.append(float(net_return))
        return float(net_return)

    def _dividend_event(self, event: DividendEvent) -> float:
        dividends = _float_pair(event.per_unit, "per_unit")
        quantities = np.asarray(self.projected_quantities, dtype=np.float64)
        multipliers = np.asarray(self._multipliers, dtype=np.float64)
        amount = float(
            np.dot(quantities * multipliers, np.asarray(dividends, dtype=np.float64))
        )
        cash_after = self._cash + amount
        if not math.isfinite(amount) or not math.isfinite(cash_after):
            raise ValueError("dividend must preserve finite account values")
        self._cash = cash_after
        self._dividend += amount
        self._refresh_economic_state()
        return amount

    def _settle(self, event: SettlementEvent) -> float:
        if (
            type(event.mask) is not tuple
            or len(event.mask) != _SYMBOLS
            or any(type(item) is not bool for item in event.mask)
        ):
            raise ValueError("mask must be a two-symbol boolean tuple")
        prices = _float_pair(event.prices, "prices", positive=True)
        recovery = _float_pair(event.recovery, "recovery")
        if any(value < 0.0 or value > 1.0 for value in recovery):
            raise ValueError("recovery must be within [0, 1]")
        products = []
        for quantity, price, multiplier, recovered, selected in zip(
            self._quantities,
            prices,
            self._multipliers,
            recovery,
            event.mask,
            strict=True,
        ):
            if selected:
                exact = (
                    quantity
                    * Fraction(str(price))
                    * Fraction(str(multiplier))
                    * Fraction(str(recovered))
                )
                try:
                    products.append(float(exact))
                except OverflowError as error:
                    raise ValueError(
                        "settlement proceeds exceed binary64 range"
                    ) from error
        try:
            proceeds = math.fsum(products)
        except OverflowError as error:
            raise ValueError("settlement proceeds exceed binary64 range") from error
        cash_after = self._cash + proceeds
        if not math.isfinite(proceeds) or not math.isfinite(cash_after):
            raise ValueError("settlement must preserve finite account values")
        self._cash = cash_after
        self._quantities = (
            Fraction(0) if event.mask[0] else self._quantities[0],
            Fraction(0) if event.mask[1] else self._quantities[1],
        )
        self._marks = prices
        self._settlement_proceeds += proceeds
        self._refresh_economic_state()
        self._update_drawdown()
        return proceeds

    def _revalue(
        self,
        marks: tuple[float, float],
        *,
        cash_after: float | None = None,
        period_start_value: float | None = None,
    ) -> None:
        proposed_cash = self._cash if cash_after is None else cash_after
        # Project the next two-slot valuation before changing either cash or
        # marks. A finite input price and quantity can still overflow on multiply.
        with np.errstate(over="ignore", invalid="ignore"):
            projected = (
                np.asarray(self.projected_quantities, dtype=np.float64)
                * np.asarray(marks, dtype=np.float64)
                * np.asarray(self._multipliers, dtype=np.float64)
            )
            prospective_wealth = float(proposed_cash + projected.sum())
        if not math.isfinite(prospective_wealth):
            raise ValueError("revaluation would make portfolio value non-finite")
        if period_start_value is not None:
            # Match the native lower-return floor before checking finiteness.
            prospective_return = max(
                prospective_wealth / period_start_value - 1.0, -1.0 + 1e-12
            )
            if not math.isfinite(prospective_return):
                raise ValueError("funding would make net return non-finite")
        self._cash = proposed_cash
        self._marks = marks
        self._refresh_economic_state()

    def _refresh_economic_state(self) -> None:
        value = self.wealth
        if not math.isfinite(value):
            raise ValueError("portfolio value became non-finite")
        if value <= 0.0 and not self._insolvent:
            self._terminate("insolvency")

    def _update_drawdown(self) -> None:
        value = max(self.wealth, 0.0)
        self._peak_value = max(self._peak_value, value)
        self._max_drawdown = max(
            self._max_drawdown,
            1.0 - value / max(self._peak_value, _MIN_EQUITY),
        )

    def _terminate(self, reason: str) -> None:
        if reason not in _TERMINATIONS:
            raise ValueError("unsupported economic termination reason")
        self._insolvent = True
        self._termination_reason = reason
