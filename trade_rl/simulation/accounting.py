"""Self-financing portfolio book state and base-bar accounting."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

_TOLERANCE = 1e-12


def _finite_vector(value: np.ndarray, *, field_name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError(f"{field_name} must be a non-empty finite vector")
    return vector


@dataclass(frozen=True, slots=True)
class BarSettlement:
    net_return: float
    liquidated: bool
    bankrupt: bool
    liquidation_cost: float
    liquidation_turnover: float


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
    liquidation_count: int = 0
    bankrupt: bool = False
    returns_history: list[float] = field(default_factory=list)

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
        ):
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if self.peak_value <= 0.0:
            raise ValueError("peak_value must be positive")
        if min(self.fill_count, self.rebalance_events, self.liquidation_count) < 0:
            raise ValueError("execution counters must be non-negative")
        object.__setattr__(self, "quantities", quantities)
        object.__setattr__(self, "mark_prices", marks)
        self._validate_equity()
        if self.peak_value + _TOLERANCE < self.portfolio_value:
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
    ) -> BookState:
        resolved_weights = _finite_vector(weights, field_name="weights")
        resolved_prices = _finite_vector(prices, field_name="prices")
        if resolved_weights.shape != resolved_prices.shape:
            raise ValueError("weights and prices must have identical shapes")
        if np.any(resolved_prices <= 0.0):
            raise ValueError("prices must be positive")
        if not np.isfinite(capital) or capital <= 0.0:
            raise ValueError("capital must be finite and positive")
        if float(np.abs(resolved_weights).sum()) > 1.0 + _TOLERANCE:
            raise ValueError("initial gross exposure cannot exceed one")
        quantities = resolved_weights * capital / resolved_prices
        cash = capital - float(np.dot(quantities, resolved_prices))
        return cls(
            quantities=quantities,
            cash=cash,
            mark_prices=resolved_prices,
            peak_value=capital,
        )

    @property
    def position_values(self) -> np.ndarray:
        return self.quantities * self.mark_prices

    @property
    def portfolio_value(self) -> float:
        return float(self.cash + self.position_values.sum())

    @property
    def weights(self) -> np.ndarray:
        value = self.portfolio_value
        if value <= 0.0:
            return np.zeros_like(self.quantities)
        return self.position_values / value

    @property
    def n_trades(self) -> int:
        """Compatibility alias: one trade means one filled symbol order."""

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
            liquidation_count=self.liquidation_count,
            bankrupt=self.bankrupt,
            returns_history=list(self.returns_history),
        )

    def state_digest(self) -> str:
        digest = hashlib.sha256()
        for array_value in (self.quantities, self.mark_prices):
            digest.update(np.ascontiguousarray(array_value).tobytes())
        for float_value in (
            self.cash,
            self.peak_value,
            self.max_drawdown,
            self.turnover_total,
            self.total_cost,
            self.funding_pnl,
        ):
            digest.update(np.float64(float_value).tobytes())
        for integer_value in (
            self.fill_count,
            self.rebalance_events,
            self.liquidation_count,
            int(self.bankrupt),
        ):
            digest.update(int(integer_value).to_bytes(8, "little", signed=True))
        return digest.hexdigest()

    def revalue(self, mark_prices: np.ndarray) -> None:
        marks = _finite_vector(mark_prices, field_name="mark_prices")
        if marks.shape != self.quantities.shape or np.any(marks <= 0.0):
            raise ValueError("mark_prices must match the book and be positive")
        self.mark_prices = marks
        self._validate_equity()

    def maintenance_margin(
        self,
        rates: np.ndarray,
        *,
        prices: np.ndarray | None = None,
    ) -> float:
        resolved_rates = _finite_vector(rates, field_name="maintenance_margin_rates")
        resolved_prices = (
            self.mark_prices
            if prices is None
            else _finite_vector(prices, field_name="maintenance_margin_prices")
        )
        if (
            resolved_rates.shape != self.quantities.shape
            or resolved_prices.shape != self.quantities.shape
        ):
            raise ValueError("maintenance margin vectors must match the book")
        if np.any(resolved_rates < 0.0) or np.any(resolved_rates >= 1.0):
            raise ValueError("maintenance margin rates must be within [0, 1)")
        if np.any(resolved_prices <= 0.0):
            raise ValueError("maintenance margin prices must be positive")
        return float(np.sum(np.abs(self.quantities * resolved_prices) * resolved_rates))

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
        self.cash -= float(np.dot(delta, prices)) + float(cost_amount)
        self.quantities = targets
        self.mark_prices = prices
        self.turnover_total += float(turnover)
        self.total_cost += float(cost_amount)
        fill_count = int(np.count_nonzero(filled))
        self.fill_count += fill_count
        if fill_count:
            self.rebalance_events += 1
        self._validate_equity()
        self._update_drawdown()

    def force_liquidate(
        self,
        *,
        prices: np.ndarray,
        liquidation_fee_rate: float,
        turnover_denominator: float,
    ) -> tuple[float, float]:
        resolved_prices = _finite_vector(prices, field_name="liquidation_prices")
        if resolved_prices.shape != self.quantities.shape:
            raise ValueError("liquidation prices must match the book")
        if np.any(resolved_prices <= 0.0):
            raise ValueError("liquidation prices must be positive")
        if (
            not np.isfinite(liquidation_fee_rate)
            or not 0.0 <= liquidation_fee_rate < 1.0
        ):
            raise ValueError("liquidation_fee_rate must be within [0, 1)")
        if not np.isfinite(turnover_denominator) or turnover_denominator <= 0.0:
            raise ValueError("turnover_denominator must be finite and positive")

        nonzero = np.abs(self.quantities) > _TOLERANCE
        gross_notional = float(np.abs(self.quantities * resolved_prices).sum())
        closing_equity = float(self.cash + np.dot(self.quantities, resolved_prices))
        requested_cost = gross_notional * liquidation_fee_rate
        minimum_equity = max(self.peak_value * 1e-12, 1e-12)
        ending_equity = closing_equity - requested_cost
        actual_cost = min(requested_cost, max(0.0, closing_equity - minimum_equity))
        self.cash = max(ending_equity, minimum_equity)
        self.quantities = np.zeros_like(self.quantities)
        self.mark_prices = resolved_prices
        fill_count = int(np.count_nonzero(nonzero))
        self.fill_count += fill_count
        if fill_count:
            self.rebalance_events += 1
        self.liquidation_count += 1
        liquidation_turnover = gross_notional / turnover_denominator
        self.turnover_total += liquidation_turnover
        self.total_cost += actual_cost
        self.bankrupt = ending_equity <= minimum_equity
        self._update_drawdown()
        return actual_cost, liquidation_turnover

    def settle_bar(
        self,
        *,
        mark_prices: np.ndarray,
        funding_amount: float,
        period_start_value: float,
        maintenance_margin_rates: np.ndarray,
        liquidation_fee_rate: float,
    ) -> BarSettlement:
        marks = _finite_vector(mark_prices, field_name="mark_prices")
        if marks.shape != self.quantities.shape or np.any(marks <= 0.0):
            raise ValueError("mark_prices must match the book and be positive")
        if not np.isfinite(funding_amount):
            raise ValueError("funding_amount must be finite")
        if not np.isfinite(period_start_value) or period_start_value <= 0.0:
            raise ValueError("period_start_value must be finite and positive")

        self.cash += float(funding_amount)
        self.mark_prices = marks
        projected_equity = self.portfolio_value
        maintenance = self.maintenance_margin(maintenance_margin_rates)
        liquidated = projected_equity <= maintenance
        liquidation_cost = 0.0
        liquidation_turnover = 0.0
        if liquidated:
            liquidation_cost, liquidation_turnover = self.force_liquidate(
                prices=marks,
                liquidation_fee_rate=liquidation_fee_rate,
                turnover_denominator=period_start_value,
            )
        else:
            self._validate_equity()

        value = self.portfolio_value
        net_return = value / period_start_value - 1.0
        if not np.isfinite(net_return) or net_return <= -1.0:
            raise ValueError("base-bar net return must be finite and greater than -1")
        self._update_drawdown()
        self.funding_pnl += float(funding_amount)
        self.returns_history.append(float(net_return))
        return BarSettlement(
            net_return=float(net_return),
            liquidated=liquidated,
            bankrupt=self.bankrupt,
            liquidation_cost=liquidation_cost,
            liquidation_turnover=liquidation_turnover,
        )

    def mark_to_market(
        self,
        *,
        mark_prices: np.ndarray,
        funding_amount: float,
        period_start_value: float | None = None,
    ) -> float:
        starting_value = (
            self.portfolio_value
            if period_start_value is None
            else float(period_start_value)
        )
        settlement = self.settle_bar(
            mark_prices=mark_prices,
            funding_amount=funding_amount,
            period_start_value=starting_value,
            maintenance_margin_rates=np.zeros_like(self.quantities),
            liquidation_fee_rate=0.0,
        )
        return settlement.net_return

    def _update_drawdown(self) -> None:
        value = self.portfolio_value
        self.peak_value = max(self.peak_value, value)
        self.max_drawdown = max(
            self.max_drawdown,
            1.0 - value / self.peak_value,
        )

    def _validate_equity(self) -> None:
        value = self.portfolio_value
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("portfolio value became non-positive or non-finite")
