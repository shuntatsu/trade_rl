"""Next-open execution, liquidity costs, funding, and self-financing marks."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState

_TOLERANCE = 1e-12


def _validate_multiplier_range(
    value: tuple[float, float],
    *,
    field_name: str,
) -> None:
    if len(value) != 2:
        raise ValueError(f"{field_name} must contain lower and upper bounds")
    lower, upper = value
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"{field_name} bounds must be finite")
    if lower < 0.0 or upper < lower:
        raise ValueError(f"{field_name} must satisfy 0 <= lower <= upper")


@dataclass(frozen=True, slots=True)
class ExecutionCostConfig:
    fee_rate: float = 0.0005
    spread_rate: float = 0.0002
    impact_rate: float = 0.0001
    liquidation_fee_rate: float = 0.005
    multiplier: float = 1.0
    max_participation_rate: float = 0.05
    slippage_std: float = 0.0
    tail_slippage_probability: float = 0.0
    tail_slippage_multiplier: float = 5.0
    fee_multiplier_range: tuple[float, float] = (1.0, 1.0)
    spread_multiplier_range: tuple[float, float] = (1.0, 1.0)
    impact_multiplier_range: tuple[float, float] = (1.0, 1.0)
    random_seed: int = 0

    def __post_init__(self) -> None:
        for field_name, value in (
            ("fee_rate", self.fee_rate),
            ("spread_rate", self.spread_rate),
            ("impact_rate", self.impact_rate),
            ("liquidation_fee_rate", self.liquidation_fee_rate),
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
                self.liquidation_fee_rate,
                self.multiplier,
                self.slippage_std,
                self.tail_slippage_multiplier,
            )
            < 0.0
        ):
            raise ValueError("execution rates and multipliers must be non-negative")
        if self.liquidation_fee_rate >= 1.0:
            raise ValueError("liquidation_fee_rate must be below one")
        if not 0.0 < self.max_participation_rate <= 1.0:
            raise ValueError("max_participation_rate must be within (0, 1]")
        if not 0.0 <= self.tail_slippage_probability <= 1.0:
            raise ValueError("tail_slippage_probability must be within [0, 1]")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be a non-negative integer")
        if self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        for field_name, bounds in (
            ("fee_multiplier_range", self.fee_multiplier_range),
            ("spread_multiplier_range", self.spread_multiplier_range),
            ("impact_multiplier_range", self.impact_multiplier_range),
        ):
            _validate_multiplier_range(bounds, field_name=field_name)

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
            liquidation_fee_rate=0.0,
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
    liquidation_count: int
    bankrupt: bool


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
        self._runtime_cost_multipliers = (1.0, 1.0, 1.0)
        self.reset_random_state(self.cost.random_seed)

    @property
    def runtime_cost_multipliers(self) -> tuple[float, float, float]:
        return self._runtime_cost_multipliers

    def reset_random_state(self, seed: int | None = None) -> None:
        resolved = self.cost.random_seed if seed is None else seed
        if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved < 0:
            raise ValueError("seed must be a non-negative integer")
        self._rng = np.random.default_rng(resolved)
        draws = [
            float(self._rng.uniform(lower, upper)) if upper > lower else float(lower)
            for lower, upper in (
                self.cost.fee_multiplier_range,
                self.cost.spread_multiplier_range,
                self.cost.impact_multiplier_range,
            )
        ]
        self._runtime_cost_multipliers = (draws[0], draws[1], draws[2])

    def _slippage_rates(self, size: int) -> np.ndarray:
        if self.cost.slippage_std == 0.0 or size == 0:
            return np.zeros(size, dtype=np.float64)
        rates = np.abs(self._rng.normal(0.0, self.cost.slippage_std, size=size))
        if self.cost.tail_slippage_probability > 0.0:
            tails = self._rng.random(size) < self.cost.tail_slippage_probability
            rates[tails] *= self.cost.tail_slippage_multiplier
        return rates

    def _round_and_filter_delta(
        self,
        *,
        current: np.ndarray,
        desired: np.ndarray,
        raw_delta: np.ndarray,
        prices: np.ndarray,
    ) -> np.ndarray:
        step = self.dataset.quantity_steps
        rounded = raw_delta.copy()
        positive_step = step > 0.0
        rounded[positive_step] = (
            np.trunc(rounded[positive_step] / step[positive_step]) * step[positive_step]
        )
        next_quantities = current + rounded
        notional = np.abs(rounded * prices)
        reducing = np.abs(next_quantities) < np.abs(current) - _TOLERANCE
        below_minimum = (
            notional + _TOLERANCE < self.dataset.minimum_notionals
        ) & ~reducing
        rounded[below_minimum] = 0.0
        would_cross = np.sign(current) != np.sign(current + rounded)
        overshoots_desired = np.abs(current + rounded) > np.abs(desired) + _TOLERANCE
        rounded[would_cross & overshoots_desired] = -current[
            would_cross & overshoots_desired
        ]
        return rounded

    def _execution_rates(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        fee_multiplier, spread_multiplier, _ = self._runtime_cost_multipliers
        fee = (
            self.dataset.taker_fee_rate[index]
            if self.dataset.taker_fee_rate is not None
            else np.full(self.dataset.n_symbols, self.cost.fee_rate)
        )
        spread = (
            self.dataset.spread_rate[index]
            if self.dataset.spread_rate is not None
            else np.full(self.dataset.n_symbols, self.cost.spread_rate)
        )
        return fee * fee_multiplier, spread * spread_multiplier

    def _fill_toward_quantities(
        self,
        book: BookState,
        desired_quantities: np.ndarray,
        *,
        prices: np.ndarray,
        volume: np.ndarray,
        tradable: np.ndarray,
        market_index: int,
        turnover_denominator: float,
    ) -> _FillResult:
        price_vector = np.asarray(prices, dtype=np.float64).reshape(-1)
        volume_vector = np.asarray(volume, dtype=np.float64).reshape(-1)
        trade_mask = np.asarray(tradable, dtype=np.bool_).reshape(-1)
        desired = np.asarray(desired_quantities, dtype=np.float64).reshape(-1)
        expected_shape = (self.dataset.n_symbols,)
        if any(
            vector.shape != expected_shape
            for vector in (price_vector, volume_vector, trade_mask, desired)
        ):
            raise ValueError("execution market vectors do not match symbols")
        if not np.isfinite(desired).all():
            raise ValueError("desired quantities must be finite")
        if np.any(price_vector <= 0.0) or np.any(volume_vector < 0.0):
            raise ValueError("execution prices and volume are invalid")
        if not math.isfinite(turnover_denominator) or turnover_denominator <= 0.0:
            raise ValueError("turnover denominator must be finite and positive")

        requested_delta = desired - book.quantities
        requested_notional_vector = requested_delta * price_vector
        market_notional = price_vector * volume_vector
        capacity = self.cost.max_participation_rate * market_notional
        capacity = np.where(trade_mask, capacity, 0.0)
        raw_filled_notional = np.sign(requested_notional_vector) * np.minimum(
            np.abs(requested_notional_vector),
            capacity,
        )
        raw_filled_delta = raw_filled_notional / price_vector
        filled_delta = self._round_and_filter_delta(
            current=book.quantities,
            desired=desired,
            raw_delta=raw_filled_delta,
            prices=price_vector,
        )
        filled_notional_vector = filled_delta * price_vector
        next_quantities = book.quantities + filled_delta

        positive_market = market_notional > 0.0
        participation = np.zeros_like(market_notional)
        participation[positive_market] = (
            np.abs(filled_notional_vector[positive_market])
            / market_notional[positive_market]
        )
        fee, spread = self._execution_rates(market_index)
        _, _, impact_multiplier = self._runtime_cost_multipliers
        impact = self.cost.impact_rate * impact_multiplier * np.sqrt(participation)
        slippage = self._slippage_rates(len(price_vector))
        unit_cost = self.cost.multiplier * (fee + spread + impact + slippage)
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
        liquidation_count_before = result_book.liquidation_count
        fill_count_before = result_book.fill_count
        rebalance_events_before = result_book.rebalance_events
        turnover_before = result_book.turnover_total
        liquidation_turnover_total = 0.0
        gross_factor = 1.0
        desired_quantities: np.ndarray | None = None
        bars_advanced = 0

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
                volume=self.dataset.volume[next_index],
                tradable=self.dataset.tradable[next_index],
                market_index=next_index,
                turnover_denominator=decision_equity,
            )
            filled_notional_total += fill.filled_notional
            total_cost += fill.cost_amount
            total_fills += fill.fill_count
            total_events += fill.rebalance_events

            intrabar_asset_returns = (
                self.dataset.mark_prices[next_index] / self.dataset.open[next_index]
                - 1.0
            )
            intrabar_return = float(np.dot(result_book.weights, intrabar_asset_returns))
            gross_factor *= (1.0 + gap_return) * (1.0 + intrabar_return)

            funding_return = -float(
                np.dot(result_book.weights, self.dataset.funding_rate[next_index])
            )
            funding_amount = result_book.portfolio_value * funding_return
            total_funding += funding_amount
            settlement = result_book.settle_bar(
                mark_prices=self.dataset.mark_prices[next_index],
                funding_amount=funding_amount,
                period_start_value=period_start_value,
                maintenance_margin_rates=self.dataset.maintenance_margin_rates,
                liquidation_fee_rate=self.cost.liquidation_fee_rate,
            )
            total_cost += settlement.liquidation_cost
            liquidation_turnover_total += settlement.liquidation_turnover
            bars_advanced += 1
            if settlement.liquidated:
                desired_quantities = np.zeros(
                    self.dataset.n_symbols,
                    dtype=np.float64,
                )

        interval_net_return = result_book.portfolio_value / starting_value - 1.0
        if interval_net_return <= -1.0 or not math.isfinite(interval_net_return):
            raise ValueError("execution produced an invalid interval return")
        requested_turnover = (
            initial_requested_notional / decision_equity + liquidation_turnover_total
        )
        filled_turnover = result_book.turnover_total - turnover_before
        unfilled_turnover = max(
            0.0,
            initial_requested_notional / decision_equity
            - filled_notional_total / decision_equity,
        )
        return ExecutionResult(
            book=result_book,
            next_index=start_index + bars_advanced,
            bars_advanced=bars_advanced,
            interval_gross_return=gross_factor - 1.0,
            interval_cost=total_cost,
            interval_funding=total_funding,
            interval_net_return=interval_net_return,
            interval_log_return=math.log1p(interval_net_return),
            requested_turnover=requested_turnover,
            filled_turnover=filled_turnover,
            unfilled_turnover=unfilled_turnover,
            fill_count=result_book.fill_count - fill_count_before,
            rebalance_events=(result_book.rebalance_events - rebalance_events_before),
            liquidation_count=result_book.liquidation_count - liquidation_count_before,
            bankrupt=result_book.bankrupt,
        )

    def liquidate_at_close(self, book: BookState, *, index: int) -> ExecutionResult:
        if not 0 <= index < self.dataset.n_bars:
            raise ValueError("liquidation index is outside the dataset")
        result_book = book.clone()
        result_book.revalue(self.dataset.mark_prices[index])
        starting_value = result_book.portfolio_value
        desired_quantities = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        fill = self._fill_toward_quantities(
            result_book,
            desired_quantities,
            prices=self.dataset.close[index],
            volume=self.dataset.volume[index],
            tradable=self.dataset.tradable[index],
            market_index=index,
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
            liquidation_count=0,
            bankrupt=result_book.bankrupt,
        )
