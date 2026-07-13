"""Next-open execution, liquidity costs, funding, and self-financing marks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState

_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class ExecutionCostConfig:
    fee_rate: float = 0.0005
    spread_rate: float = 0.0002
    impact_rate: float = 0.0001
    multiplier: float = 1.0
    max_participation_rate: float = 0.05
    slippage_std: float = 0.0
    tail_slippage_probability: float = 0.0
    tail_slippage_multiplier: float = 5.0
    random_seed: int = 0

    def __post_init__(self) -> None:
        for field_name, value in (
            ("fee_rate", self.fee_rate),
            ("spread_rate", self.spread_rate),
            ("impact_rate", self.impact_rate),
            ("multiplier", self.multiplier),
            ("max_participation_rate", self.max_participation_rate),
            ("slippage_std", self.slippage_std),
            ("tail_slippage_probability", self.tail_slippage_probability),
            ("tail_slippage_multiplier", self.tail_slippage_multiplier),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if (
            min(
                self.fee_rate,
                self.spread_rate,
                self.impact_rate,
                self.multiplier,
                self.slippage_std,
                self.tail_slippage_multiplier,
            )
            < 0.0
        ):
            raise ValueError("execution rates and multipliers must be non-negative")
        if not 0.0 < self.max_participation_rate <= 1.0:
            raise ValueError("max_participation_rate must be within (0, 1]")
        if not 0.0 <= self.tail_slippage_probability <= 1.0:
            raise ValueError("tail_slippage_probability must be within [0, 1]")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be a non-negative integer")
        if self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")

    @property
    def rate_per_turnover(self) -> float:
        """Nominal rate excluding participation-dependent and random costs."""

        return self.multiplier * (self.fee_rate + self.spread_rate)

    @classmethod
    def zero(cls) -> ExecutionCostConfig:
        return cls(
            fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            multiplier=1.0,
            max_participation_rate=1.0,
            slippage_std=0.0,
            tail_slippage_probability=0.0,
            tail_slippage_multiplier=0.0,
        )


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    book: BookState
    next_index: int
    bars_advanced: int
    interval_gross_return: float
    interval_cost: float
    interval_funding: float
    interval_net_return: float
    interval_log_return: float
    requested_turnover: float
    filled_turnover: float
    unfilled_turnover: float
    fill_count: int
    rebalance_events: int


@dataclass(frozen=True, slots=True)
class _FillResult:
    requested_notional: float
    filled_notional: float
    cost_amount: float
    fill_count: int
    rebalance_events: int


def _target_weights(value: np.ndarray, *, n_symbols: int) -> np.ndarray:
    target = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if target.shape != (n_symbols,):
        raise ValueError("target weights shape does not match market symbols")
    if not np.isfinite(target).all():
        raise ValueError("target weights must be finite")
    gross = float(np.abs(target).sum())
    if gross > 1.0 + _TOLERANCE:
        raise ValueError("target gross exposure cannot exceed one")
    return target


class MarketExecutor:
    """Execute one decision target while holding filled quantities until the next decision."""

    def __init__(
        self,
        dataset: MarketDataset,
        cost: ExecutionCostConfig | None = None,
    ) -> None:
        self.dataset = dataset
        self.cost = cost or ExecutionCostConfig()
        self._rng = np.random.default_rng(self.cost.random_seed)

    def reset_random_state(self, seed: int | None = None) -> None:
        resolved = self.cost.random_seed if seed is None else seed
        if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved < 0:
            raise ValueError("seed must be a non-negative integer")
        self._rng = np.random.default_rng(resolved)

    def _slippage_rates(self, size: int) -> np.ndarray:
        if self.cost.slippage_std == 0.0 or size == 0:
            return np.zeros(size, dtype=np.float64)
        rates = np.abs(self._rng.normal(0.0, self.cost.slippage_std, size=size))
        if self.cost.tail_slippage_probability > 0.0:
            tails = self._rng.random(size) < self.cost.tail_slippage_probability
            rates[tails] *= self.cost.tail_slippage_multiplier
        return rates

    def _fill_toward_quantities(
        self,
        book: BookState,
        desired_quantities: np.ndarray,
        *,
        prices: np.ndarray,
        market_notional: np.ndarray,
        tradable: np.ndarray,
        turnover_denominator: float,
    ) -> _FillResult:
        price_vector = np.asarray(prices, dtype=np.float64).reshape(-1)
        market_notional_vector = np.asarray(
            market_notional, dtype=np.float64
        ).reshape(-1)
        trade_mask = np.asarray(tradable, dtype=np.bool_).reshape(-1)
        desired = np.asarray(desired_quantities, dtype=np.float64).reshape(-1)
        expected_shape = (self.dataset.n_symbols,)
        if any(
            vector.shape != expected_shape
            for vector in (price_vector, market_notional_vector, trade_mask, desired)
        ):
            raise ValueError("execution market vectors do not match symbols")
        if not np.isfinite(desired).all():
            raise ValueError("desired quantities must be finite")
        if np.any(price_vector <= 0.0) or np.any(market_notional_vector < 0.0):
            raise ValueError("execution prices and market notional are invalid")
        if not math.isfinite(turnover_denominator) or turnover_denominator <= 0.0:
            raise ValueError("turnover denominator must be finite and positive")

        requested_delta = desired - book.quantities
        requested_notional_vector = requested_delta * price_vector
        capacity = self.cost.max_participation_rate * market_notional_vector
        capacity = np.where(trade_mask, capacity, 0.0)
        filled_notional_vector = np.sign(requested_notional_vector) * np.minimum(
            np.abs(requested_notional_vector),
            capacity,
        )
        filled_delta = filled_notional_vector / price_vector
        next_quantities = book.quantities + filled_delta

        positive_market = market_notional_vector > 0.0
        participation = np.zeros_like(market_notional_vector)
        participation[positive_market] = (
            np.abs(filled_notional_vector[positive_market])
            / market_notional_vector[positive_market]
        )
        impact = self.cost.impact_rate * np.sqrt(participation)
        slippage = self._slippage_rates(len(price_vector))
        unit_cost = self.cost.multiplier * (
            self.cost.fee_rate + self.cost.spread_rate + impact + slippage
        )
        cost_amount = float(np.sum(np.abs(filled_notional_vector) * unit_cost))
        if not math.isfinite(cost_amount) or cost_amount >= book.portfolio_value:
            raise ValueError("execution cost would exhaust the portfolio")

        fills_before = book.fill_count
        events_before = book.rebalance_events
        filled_notional = float(np.abs(filled_notional_vector).sum())
        book.execute(
            fill_prices=price_vector,
            target_quantities=next_quantities,
            cost_amount=cost_amount,
            turnover=filled_notional / turnover_denominator,
        )
        return _FillResult(
            requested_notional=float(np.abs(requested_notional_vector).sum()),
            filled_notional=filled_notional,
            cost_amount=cost_amount,
            fill_count=book.fill_count - fills_before,
            rebalance_events=book.rebalance_events - events_before,
        )

    def execute_interval(
        self,
        book: BookState,
        target: np.ndarray,
        *,
        start_index: int,
        bars: int,
    ) -> ExecutionResult:
        if bars <= 0:
            raise ValueError("bars must be positive")
        if start_index < 0 or start_index + bars >= self.dataset.n_bars:
            raise ValueError("execution interval is outside the dataset")
        if book.weights.shape != (self.dataset.n_symbols,):
            raise ValueError("book weights shape does not match market symbols")

        resolved_target = _target_weights(target, n_symbols=self.dataset.n_symbols)
        result_book = book.clone()
        starting_value = result_book.portfolio_value
        decision_equity = 0.0
        initial_requested_notional = 0.0
        filled_notional_total = 0.0
        total_cost = 0.0
        total_funding = 0.0
        total_fills = 0
        total_events = 0
        gross_factor = 1.0
        desired_quantities: np.ndarray | None = None

        for offset in range(bars):
            close_index = start_index + offset
            next_index = close_index + 1
            period_start_value = result_book.portfolio_value

            result_book.revalue(self.dataset.open[next_index])
            value_at_open = result_book.portfolio_value
            gap_return = value_at_open / period_start_value - 1.0
            if desired_quantities is None:
                decision_equity = value_at_open
                desired_quantities = (
                    resolved_target * decision_equity / self.dataset.open[next_index]
                )
                initial_requested_notional = float(
                    np.abs(
                        (desired_quantities - result_book.quantities)
                        * self.dataset.open[next_index]
                    ).sum()
                )

            fill = self._fill_toward_quantities(
                result_book,
                desired_quantities,
                prices=self.dataset.open[next_index],
                market_notional=self.dataset.market_notional(
                    next_index, self.dataset.open[next_index]
                ),
                tradable=self.dataset.tradable[next_index],
                turnover_denominator=decision_equity,
            )
            filled_notional_total += fill.filled_notional
            total_cost += fill.cost_amount
            total_fills += fill.fill_count
            total_events += fill.rebalance_events

            intrabar_asset_returns = (
                self.dataset.close[next_index] / self.dataset.open[next_index] - 1.0
            )
            intrabar_return = float(np.dot(result_book.weights, intrabar_asset_returns))
            gross_factor *= (1.0 + gap_return) * (1.0 + intrabar_return)

            funding_return = -float(
                np.dot(result_book.weights, self.dataset.funding_rate[next_index])
            )
            funding_amount = result_book.portfolio_value * funding_return
            total_funding += funding_amount
            result_book.mark_to_market(
                mark_prices=self.dataset.close[next_index],
                funding_amount=funding_amount,
                period_start_value=period_start_value,
            )

        interval_net_return = result_book.portfolio_value / starting_value - 1.0
        if interval_net_return <= -1.0 or not math.isfinite(interval_net_return):
            raise ValueError("execution produced an invalid interval return")
        requested_turnover = initial_requested_notional / decision_equity
        filled_turnover = filled_notional_total / decision_equity
        unfilled_turnover = max(0.0, requested_turnover - filled_turnover)
        return ExecutionResult(
            book=result_book,
            next_index=start_index + bars,
            bars_advanced=bars,
            interval_gross_return=gross_factor - 1.0,
            interval_cost=total_cost,
            interval_funding=total_funding,
            interval_net_return=interval_net_return,
            interval_log_return=math.log1p(interval_net_return),
            requested_turnover=requested_turnover,
            filled_turnover=filled_turnover,
            unfilled_turnover=unfilled_turnover,
            fill_count=total_fills,
            rebalance_events=total_events,
        )

    def liquidate_at_close(self, book: BookState, *, index: int) -> ExecutionResult:
        if not 0 <= index < self.dataset.n_bars:
            raise ValueError("liquidation index is outside the dataset")
        result_book = book.clone()
        result_book.revalue(self.dataset.close[index])
        starting_value = result_book.portfolio_value
        desired_quantities = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        fill = self._fill_toward_quantities(
            result_book,
            desired_quantities,
            prices=self.dataset.close[index],
            market_notional=self.dataset.market_notional(
                index, self.dataset.close[index]
            ),
            tradable=self.dataset.tradable[index],
            turnover_denominator=starting_value,
        )
        interval_net_return = result_book.portfolio_value / starting_value - 1.0
        requested_turnover = fill.requested_notional / starting_value
        filled_turnover = fill.filled_notional / starting_value
        return ExecutionResult(
            book=result_book,
            next_index=index,
            bars_advanced=0,
            interval_gross_return=0.0,
            interval_cost=fill.cost_amount,
            interval_funding=0.0,
            interval_net_return=interval_net_return,
            interval_log_return=math.log1p(interval_net_return),
            requested_turnover=requested_turnover,
            filled_turnover=filled_turnover,
            unfilled_turnover=max(0.0, requested_turnover - filled_turnover),
            fill_count=fill.fill_count,
            rebalance_events=fill.rebalance_events,
        )
