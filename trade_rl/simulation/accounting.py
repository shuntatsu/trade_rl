"""Self-financing portfolio book state and base-bar accounting."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

_TOLERANCE = 1e-12
_MIN_EQUITY: float = float(np.finfo(np.float64).tiny)


class EconomicTerminationReason(str, Enum):
    """Economic outcomes that terminate an episode without a code error."""

    MINIMUM_EQUITY = "minimum_equity"
    EXECUTION_COST_EXHAUSTION = "execution_cost_exhaustion"
    MARGIN_CALL = "margin_call"
    LIQUIDATION = "liquidation"
    INSOLVENCY = "insolvency"


def _finite_vector(value: np.ndarray, *, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{field_name} must be a non-empty finite vector")
    return vector


@dataclass(slots=True)
class BookState:
    """Mutable signed-quantity account isolated from policy and workflow logic."""

    quantities: np.ndarray
    cash: float
    mark_prices: np.ndarray
    peak_value: float
    max_drawdown: float = 0.0
    turnover_total: float = 0.0
    total_cost: float = 0.0
    funding_pnl: float = 0.0
    fill_count: int = 0
    rebalance_events: int = 0
    returns_history: list[float] = field(default_factory=list)
    borrow_cost: float = 0.0
    margin_used: float = 0.0
    maintenance_margin: float = 0.0
    maintenance_requirement: float = 0.0
    margin_deficit: float = 0.0
    insolvent: bool = False
    termination_reason: EconomicTerminationReason | str | None = None

    def __post_init__(self) -> None:
        quantities = _finite_vector(self.quantities, field_name="quantities")
        marks = _finite_vector(self.mark_prices, field_name="mark_prices")
        if quantities.shape != marks.shape:
            raise ValueError("quantities and mark_prices must have identical shapes")
        if np.any(marks <= 0.0):
            raise ValueError("mark_prices must be strictly positive")
        for field_name, value in (
            ("cash", self.cash),
            ("peak_value", self.peak_value),
            ("max_drawdown", self.max_drawdown),
            ("turnover_total", self.turnover_total),
            ("total_cost", self.total_cost),
            ("funding_pnl", self.funding_pnl),
            ("borrow_cost", self.borrow_cost),
            ("margin_used", self.margin_used),
            ("maintenance_margin", self.maintenance_margin),
            ("maintenance_requirement", self.maintenance_requirement),
            ("margin_deficit", self.margin_deficit),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.peak_value <= 0.0:
            raise ValueError("peak_value must be positive")
        if self.fill_count < 0 or self.rebalance_events < 0:
            raise ValueError("execution counters must be non-negative")
        if (
            self.borrow_cost < 0.0
            or self.margin_used < 0.0
            or self.maintenance_requirement < 0.0
            or self.margin_deficit < 0.0
        ):
            raise ValueError(
                "borrow_cost, margin_used and maintenance_requirement must be non-negative"
            )
        if not 0.0 <= self.maintenance_margin <= 1.0:
            raise ValueError("maintenance_margin must be within [0, 1]")
        if not isinstance(self.insolvent, bool):
            raise ValueError("insolvent must be a boolean")
        reason = self.termination_reason
        if reason is not None:
            try:
                reason = EconomicTerminationReason(reason)
            except ValueError as error:
                raise ValueError("termination_reason is not supported") from error
        object.__setattr__(self, "quantities", quantities)
        object.__setattr__(self, "mark_prices", marks)
        object.__setattr__(self, "termination_reason", reason)
        self._refresh_economic_state()
        if not self.insolvent and self.peak_value + _TOLERANCE < self.portfolio_value:
            raise ValueError("peak_value cannot be below portfolio_value")

    @classmethod
    def zero(
        cls,
        n_symbols: int,
        initial_capital: float,
        initial_prices: np.ndarray | None = None,
    ) -> BookState:
        if n_symbols <= 0:
            raise ValueError("n_symbols must be positive")
        if not np.isfinite(initial_capital) or initial_capital <= 0.0:
            raise ValueError("initial_capital must be finite and positive")
        prices = (
            np.ones(n_symbols, dtype=np.float64)
            if initial_prices is None
            else _finite_vector(initial_prices, field_name="initial_prices")
        )
        if prices.shape != (n_symbols,) or np.any(prices <= 0.0):
            raise ValueError("initial_prices must match symbols and be positive")
        return cls(
            quantities=np.zeros(n_symbols, dtype=np.float64),
            cash=float(initial_capital),
            mark_prices=prices,
            peak_value=float(initial_capital),
        )

    @classmethod
    def from_weights(
        cls,
        *,
        weights: np.ndarray,
        capital: float,
        prices: np.ndarray,
        peak_value: float | None = None,
        max_gross: float = 1.0,
    ) -> BookState:
        weight_vector = _finite_vector(weights, field_name="weights")
        price_vector = _finite_vector(prices, field_name="prices")
        if weight_vector.shape != price_vector.shape or np.any(price_vector <= 0.0):
            raise ValueError("weights and prices must have identical valid shapes")
        if not np.isfinite(capital) or capital <= 0.0:
            raise ValueError("capital must be finite and positive")
        if not np.isfinite(max_gross) or max_gross <= 0.0:
            raise ValueError("max_gross must be finite and positive")
        gross = float(np.abs(weight_vector).sum())
        if gross > max_gross + _TOLERANCE:
            raise ValueError("initial gross exposure exceeds max_gross")
        quantities = weight_vector * capital / price_vector
        cash = capital - float(np.dot(quantities, price_vector))
        resolved_peak = capital if peak_value is None else float(peak_value)
        if resolved_peak < capital:
            raise ValueError("peak_value cannot be below capital")
        return cls(
            quantities=quantities,
            cash=cash,
            mark_prices=price_vector,
            peak_value=resolved_peak,
        )

    @property
    def position_values(self) -> np.ndarray:
        return self.quantities * self.mark_prices

    @property
    def portfolio_value(self) -> float:
        return float(self.cash + self.position_values.sum())

    @property
    def effective_portfolio_value(self) -> float:
        return max(self.portfolio_value, _MIN_EQUITY)

    @property
    def weights(self) -> np.ndarray:
        value = self.portfolio_value
        if value <= 0.0 or not np.isfinite(value):
            return np.zeros_like(self.quantities)
        return self.position_values / value

    @property
    def cash_weight(self) -> float:
        value = self.portfolio_value
        return 0.0 if value <= 0.0 else float(self.cash / value)

    @property
    def gross_exposure(self) -> float:
        return float(np.abs(self.weights).sum())

    @property
    def net_exposure(self) -> float:
        return float(self.weights.sum())

    @property
    def margin_utilization(self) -> float:
        value = self.portfolio_value
        if value <= 0.0:
            return 1.0
        return float(self.margin_used / value)

    @property
    def n_trades(self) -> int:
        return self.fill_count

    def clone(self) -> BookState:
        return BookState(
            quantities=self.quantities.copy(),
            cash=self.cash,
            mark_prices=self.mark_prices.copy(),
            peak_value=self.peak_value,
            max_drawdown=self.max_drawdown,
            turnover_total=self.turnover_total,
            total_cost=self.total_cost,
            funding_pnl=self.funding_pnl,
            fill_count=self.fill_count,
            rebalance_events=self.rebalance_events,
            returns_history=list(self.returns_history),
            borrow_cost=self.borrow_cost,
            margin_used=self.margin_used,
            maintenance_margin=self.maintenance_margin,
            maintenance_requirement=self.maintenance_requirement,
            margin_deficit=self.margin_deficit,
            insolvent=self.insolvent,
            termination_reason=self.termination_reason,
        )

    def apply_split(self, split_factor: np.ndarray) -> None:
        factors = _finite_vector(split_factor, field_name="split_factor")
        if factors.shape != self.quantities.shape or np.any(factors <= 0.0):
            raise ValueError("split_factor must match the book and be positive")
        self.quantities *= factors
        self.mark_prices /= factors
        self._refresh_economic_state()

    def apply_dividend(self, dividend_per_unit: np.ndarray) -> float:
        dividend = _finite_vector(dividend_per_unit, field_name="dividend_per_unit")
        if dividend.shape != self.quantities.shape:
            raise ValueError("dividend_per_unit must match the book")
        amount = float(np.dot(self.quantities, dividend))
        self.cash += amount
        self._refresh_economic_state()
        return amount

    def apply_cash_interest(
        self, annual_rate: float, *, periods_per_year: int
    ) -> float:
        if not np.isfinite(annual_rate):
            raise ValueError("annual_rate must be finite")
        if periods_per_year <= 0:
            raise ValueError("periods_per_year must be positive")
        amount = float(self.cash * annual_rate / periods_per_year)
        self.cash += amount
        self._refresh_economic_state()
        return amount

    def settle_positions(
        self,
        *,
        mask: np.ndarray,
        prices: np.ndarray,
        recovery: np.ndarray,
    ) -> float:
        settle_mask = np.asarray(mask, dtype=np.bool_).reshape(-1)
        price_vector = _finite_vector(prices, field_name="settlement_prices")
        recovery_vector = _finite_vector(recovery, field_name="recovery")
        if any(
            vector.shape != self.quantities.shape
            for vector in (settle_mask, price_vector, recovery_vector)
        ):
            raise ValueError("settlement vectors must match the book")
        if (
            np.any(price_vector <= 0.0)
            or np.any(recovery_vector < 0.0)
            or np.any(recovery_vector > 1.0)
        ):
            raise ValueError("settlement prices or recovery are invalid")
        proceeds = float(
            np.sum(
                self.quantities[settle_mask]
                * price_vector[settle_mask]
                * recovery_vector[settle_mask]
            )
        )
        self.cash += proceeds
        self.quantities[settle_mask] = 0.0
        self.mark_prices = price_vector.copy()
        self._refresh_economic_state()
        self._update_drawdown()
        return proceeds

    def set_margin(
        self,
        *,
        margin_used: float,
        maintenance_margin: float,
        maintenance_requirement: float | None = None,
    ) -> None:
        if not np.isfinite(margin_used) or margin_used < 0.0:
            raise ValueError("margin_used must be finite and non-negative")
        if not np.isfinite(maintenance_margin) or not 0.0 <= maintenance_margin <= 1.0:
            raise ValueError("maintenance_margin must be within [0, 1]")
        requirement = (
            float(maintenance_margin) * float(np.abs(self.position_values).sum())
            if maintenance_requirement is None
            else float(maintenance_requirement)
        )
        if not np.isfinite(requirement) or requirement < 0.0:
            raise ValueError("maintenance_requirement must be finite and non-negative")
        self.margin_used = float(margin_used)
        self.maintenance_margin = float(maintenance_margin)
        self.maintenance_requirement = requirement
        self.margin_deficit = max(
            0.0,
            requirement - max(self.portfolio_value, 0.0),
        )
        if (
            self.portfolio_value > 0.0
            and requirement > 0.0
            and self.portfolio_value < requirement - _TOLERANCE
        ):
            self.terminate(EconomicTerminationReason.MARGIN_CALL)

    def terminate(self, reason: EconomicTerminationReason | str) -> None:
        resolved = EconomicTerminationReason(reason)
        self.insolvent = True
        self.termination_reason = resolved

    def revalue(self, mark_prices: np.ndarray) -> None:
        marks = _finite_vector(mark_prices, field_name="mark_prices")
        if marks.shape != self.quantities.shape or np.any(marks <= 0.0):
            raise ValueError("mark_prices must match the book and be positive")
        self.mark_prices = marks
        self._refresh_economic_state()

    def execute(
        self,
        *,
        fill_prices: np.ndarray,
        target_quantities: np.ndarray,
        cost_amount: float,
        turnover: float,
    ) -> None:
        prices = _finite_vector(fill_prices, field_name="fill_prices")
        targets = _finite_vector(target_quantities, field_name="target_quantities")
        if (
            prices.shape != self.quantities.shape
            or targets.shape != self.quantities.shape
        ):
            raise ValueError("execution vectors must match the book")
        if np.any(prices <= 0.0):
            raise ValueError("fill_prices must be strictly positive")
        if not np.isfinite(cost_amount) or cost_amount < 0.0:
            raise ValueError("cost_amount must be finite and non-negative")
        if not np.isfinite(turnover) or turnover < 0.0:
            raise ValueError("turnover must be finite and non-negative")

        delta = targets - self.quantities
        filled = np.abs(delta) > _TOLERANCE
        value_before = self.portfolio_value
        self.cash -= float(np.dot(delta, prices)) + float(cost_amount)
        self.quantities = targets
        self.mark_prices = prices
        self.turnover_total += float(turnover)
        self.total_cost += float(cost_amount)
        fill_count = int(np.count_nonzero(filled))
        self.fill_count += fill_count
        if fill_count:
            self.rebalance_events += 1
        if cost_amount >= max(value_before, 0.0) - _TOLERANCE:
            self.terminate(EconomicTerminationReason.EXECUTION_COST_EXHAUSTION)
        self._refresh_economic_state()
        self._update_drawdown()

    def charge_borrow(self, amount: float) -> None:
        if not np.isfinite(amount) or amount < 0.0:
            raise ValueError("borrow amount must be finite and non-negative")
        self.cash -= float(amount)
        self.borrow_cost += float(amount)
        self._refresh_economic_state()

    def mark_to_market(
        self,
        *,
        mark_prices: np.ndarray,
        funding_amount: float,
        period_start_value: float | None = None,
    ) -> float:
        if not np.isfinite(funding_amount):
            raise ValueError("funding_amount must be finite")
        starting_value = (
            self.portfolio_value
            if period_start_value is None
            else float(period_start_value)
        )
        if not np.isfinite(starting_value) or starting_value <= 0.0:
            raise ValueError("period_start_value must be finite and positive")

        self.cash += float(funding_amount)
        self.revalue(mark_prices)
        value = self.portfolio_value
        if not np.isfinite(value):
            raise ValueError("base-bar portfolio value became non-finite")
        net_return = max(value / starting_value - 1.0, -1.0 + 1e-12)
        self._update_drawdown()
        self.funding_pnl += float(funding_amount)
        self.returns_history.append(float(net_return))
        return float(net_return)

    def _update_drawdown(self) -> None:
        value = max(self.portfolio_value, 0.0)
        self.peak_value = max(self.peak_value, value)
        self.max_drawdown = max(
            self.max_drawdown,
            1.0 - value / max(self.peak_value, _MIN_EQUITY),
        )

    def _refresh_economic_state(self) -> None:
        value = self.portfolio_value
        if not np.isfinite(value):
            raise ValueError("portfolio value became non-finite")
        if value <= 0.0 and not self.insolvent:
            self.terminate(EconomicTerminationReason.INSOLVENCY)
