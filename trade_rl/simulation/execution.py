"""Next-open execution, liquidity costs, funding, borrow and margin accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import (
    BookState,
    EconomicTerminationReason,
)

_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class ExecutionRuleStress:
    """Evaluation-only multiplicative overlay for execution-rule sensitivity."""

    name: str = "nominal"
    tick_size_factor: float = 1.0
    lot_size_factor: float = 1.0
    minimum_notional_factor: float = 1.0
    adverse_tick_rounding: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("execution-rule stress name must be non-empty")
        for field_name, value in (
            ("tick_size_factor", self.tick_size_factor),
            ("lot_size_factor", self.lot_size_factor),
            ("minimum_notional_factor", self.minimum_notional_factor),
        ):
            if not math.isfinite(value) or value < 1.0:
                raise ValueError(f"{field_name} must be finite and at least 1.0")
        if not isinstance(self.adverse_tick_rounding, bool):
            raise ValueError("adverse_tick_rounding must be a boolean")

    @property
    def enabled(self) -> bool:
        return (
            self.tick_size_factor > 1.0
            or self.lot_size_factor > 1.0
            or self.minimum_notional_factor > 1.0
            or self.adverse_tick_rounding
        )

    def digest_payload(self) -> dict[str, object]:
        return {
            "adverse_tick_rounding": self.adverse_tick_rounding,
            "lot_size_factor": self.lot_size_factor,
            "minimum_notional_factor": self.minimum_notional_factor,
            "name": self.name,
            "schema_version": "execution_rule_stress_v1",
            "tick_size_factor": self.tick_size_factor,
        }


@dataclass(frozen=True, slots=True)
class ExecutionCostConfig:
    fee_rate: float = 0.0005
    maker_fee_rate: float = 0.0
    taker_fee_rate: float = 0.0
    spread_rate: float = 0.0002
    impact_rate: float = 0.0001
    multiplier: float = 1.0
    max_participation_rate: float = 0.05
    slippage_std: float = 0.0
    tail_slippage_probability: float = 0.0
    tail_slippage_multiplier: float = 5.0
    random_seed: int = 0
    minimum_notional: float = 0.0
    lot_size: float = 0.0
    tick_size: float = 0.0
    allow_short: bool = True
    borrow_rate_multiplier: float = 1.0
    max_leverage: float = 1.0
    maintenance_margin_rate: float = 0.25
    collateral_haircut: float = 1.0
    margin_mode: str = "cross"
    order_latency_bars: int = 0
    order_type: str = "market"
    limit_offset_rate: float = 0.0005

    def __post_init__(self) -> None:
        for field_name, value in (
            ("fee_rate", self.fee_rate),
            ("maker_fee_rate", self.maker_fee_rate),
            ("taker_fee_rate", self.taker_fee_rate),
            ("spread_rate", self.spread_rate),
            ("impact_rate", self.impact_rate),
            ("multiplier", self.multiplier),
            ("max_participation_rate", self.max_participation_rate),
            ("slippage_std", self.slippage_std),
            ("tail_slippage_probability", self.tail_slippage_probability),
            ("tail_slippage_multiplier", self.tail_slippage_multiplier),
            ("minimum_notional", self.minimum_notional),
            ("lot_size", self.lot_size),
            ("tick_size", self.tick_size),
            ("borrow_rate_multiplier", self.borrow_rate_multiplier),
            ("max_leverage", self.max_leverage),
            ("maintenance_margin_rate", self.maintenance_margin_rate),
            ("collateral_haircut", self.collateral_haircut),
            ("limit_offset_rate", self.limit_offset_rate),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        if (
            min(
                self.fee_rate,
                self.maker_fee_rate,
                self.taker_fee_rate,
                self.spread_rate,
                self.impact_rate,
                self.multiplier,
                self.slippage_std,
                self.tail_slippage_multiplier,
                self.minimum_notional,
                self.lot_size,
                self.tick_size,
                self.borrow_rate_multiplier,
                self.limit_offset_rate,
            )
            < 0.0
        ):
            raise ValueError("execution rates and constraints must be non-negative")
        if not 0.0 < self.max_participation_rate <= 1.0:
            raise ValueError("max_participation_rate must be within (0, 1]")
        if not 0.0 <= self.tail_slippage_probability <= 1.0:
            raise ValueError("tail_slippage_probability must be within [0, 1]")
        if not 0.0 < self.max_leverage:
            raise ValueError("max_leverage must be positive")
        if not 0.0 <= self.maintenance_margin_rate <= 1.0:
            raise ValueError("maintenance_margin_rate must be within [0, 1]")
        if not 0.0 < self.collateral_haircut <= 1.0:
            raise ValueError("collateral_haircut must be within (0, 1]")
        if self.margin_mode not in {"cross", "isolated"}:
            raise ValueError("margin_mode must be 'cross' or 'isolated'")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be a non-negative integer")
        if self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        if (
            isinstance(self.order_latency_bars, bool)
            or not isinstance(self.order_latency_bars, int)
            or self.order_latency_bars < 0
        ):
            raise ValueError("order_latency_bars must be a non-negative integer")
        if self.order_type not in {"market", "limit"}:
            raise ValueError("order_type must be 'market' or 'limit'")
        if self.limit_offset_rate >= 1.0:
            raise ValueError("limit_offset_rate must be below one")
        if not isinstance(self.allow_short, bool):
            raise ValueError("allow_short must be a boolean")

    @property
    def rate_per_turnover(self) -> float:
        return self.multiplier * (self.fee_rate + self.spread_rate)

    @classmethod
    def zero(cls) -> ExecutionCostConfig:
        return cls(
            fee_rate=0.0,
            maker_fee_rate=0.0,
            taker_fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            multiplier=1.0,
            max_participation_rate=1.0,
            slippage_std=0.0,
            tail_slippage_probability=0.0,
            tail_slippage_multiplier=0.0,
            minimum_notional=0.0,
            lot_size=0.0,
            tick_size=0.0,
            borrow_rate_multiplier=0.0,
            max_leverage=1.0,
            maintenance_margin_rate=0.0,
            limit_offset_rate=0.0,
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
    fill_ratio: float = 1.0
    max_participation: float = 0.0
    interval_borrow_cost: float = 0.0
    interval_dividend: float = 0.0
    interval_cash_interest: float = 0.0
    margin_utilization: float = 0.0
    termination_reason: str | None = None
    requested_notional_by_symbol: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    filled_notional_by_symbol: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    participation_by_symbol: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    cost_by_symbol: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )


@dataclass(frozen=True, slots=True)
class _FillResult:
    requested_notional: float
    filled_notional: float
    cost_amount: float
    fill_count: int
    rebalance_events: int
    max_participation: float
    requested_notional_by_symbol: np.ndarray
    filled_notional_by_symbol: np.ndarray
    participation_by_symbol: np.ndarray
    cost_by_symbol: np.ndarray


def _target_weights(
    value: np.ndarray,
    *,
    n_symbols: int,
    maximum_gross: float,
) -> np.ndarray:
    if not math.isfinite(maximum_gross) or maximum_gross <= 0.0:
        raise ValueError("maximum_gross must be finite and positive")
    target = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if target.shape != (n_symbols,):
        raise ValueError("target weights shape does not match market symbols")
    if not np.isfinite(target).all():
        raise ValueError("target weights must be finite")
    gross = float(np.abs(target).sum())
    if gross > maximum_gross + _TOLERANCE:
        raise ValueError("target gross exposure exceeds configured leverage")
    return target


class MarketExecutor:
    """Execute one decision target while holding filled quantities."""

    def __init__(
        self,
        dataset: MarketDataset,
        cost: ExecutionCostConfig | None = None,
        *,
        rule_stress: ExecutionRuleStress | None = None,
    ) -> None:
        self.dataset = dataset
        self.cost = cost or ExecutionCostConfig()
        self.rule_stress = rule_stress or ExecutionRuleStress()
        self._validate_rule_stress()
        self._rng = np.random.default_rng(self.cost.random_seed)

    def _base_rule_array(self, field_name: str, *, floor: float) -> np.ndarray:
        return np.maximum(self.dataset.resolved_array(field_name), floor)

    def _validate_rule_stress(self) -> None:
        requirements = (
            (
                "tick_size",
                self.cost.tick_size,
                self.rule_stress.tick_size_factor > 1.0
                or self.rule_stress.adverse_tick_rounding,
            ),
            (
                "lot_size",
                self.cost.lot_size,
                self.rule_stress.lot_size_factor > 1.0,
            ),
            (
                "minimum_notional",
                self.cost.minimum_notional,
                self.rule_stress.minimum_notional_factor > 1.0,
            ),
        )
        for field_name, floor, required in requirements:
            if required and np.any(
                self._base_rule_array(field_name, floor=floor) <= 0.0
            ):
                raise ValueError(
                    f"{field_name} must be positive for execution-rule sensitivity"
                )

    def effective_rule_arrays(
        self, *, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not 0 <= index < self.dataset.n_bars:
            raise IndexError("execution-rule index is outside the dataset")
        tick = (
            self._base_rule_array("tick_size", floor=self.cost.tick_size)[index]
            * self.rule_stress.tick_size_factor
        )
        lot = (
            self._base_rule_array("lot_size", floor=self.cost.lot_size)[index]
            * self.rule_stress.lot_size_factor
        )
        minimum = (
            self._base_rule_array("minimum_notional", floor=self.cost.minimum_notional)[
                index
            ]
            * self.rule_stress.minimum_notional_factor
        )
        return tick, lot, minimum

    @staticmethod
    def _percentiles(values: np.ndarray) -> dict[str, float]:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("rule burden values must be finite and non-empty")
        return {
            "p50": float(np.percentile(vector, 50.0)),
            "p95": float(np.percentile(vector, 95.0)),
            "max": float(np.max(vector)),
        }

    def rule_burden_percentiles(self, *, start: int, stop: int) -> dict[str, object]:
        if not 0 <= start < stop <= self.dataset.n_bars:
            raise ValueError("rule burden range is outside the dataset")
        shape = self.dataset.resolved_array("tick_size")[start:stop].shape
        return {
            "adverse_tick_rounding": self.rule_stress.adverse_tick_rounding,
            "lot_size_ratio": self._percentiles(
                np.full(shape, self.rule_stress.lot_size_factor)
            ),
            "minimum_notional_ratio": self._percentiles(
                np.full(shape, self.rule_stress.minimum_notional_factor)
            ),
            "schema_version": "execution_rule_burden_v1",
            "tick_size_ratio": self._percentiles(
                np.full(shape, self.rule_stress.tick_size_factor)
            ),
        }

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

    def _round_prices(
        self,
        prices: np.ndarray,
        *,
        index: int,
        directions: np.ndarray | None = None,
    ) -> np.ndarray:
        tick, _, _ = self.effective_rule_arrays(index=index)
        rounded = prices.copy()
        mask = tick > 0.0
        scaled = np.zeros_like(rounded)
        scaled[mask] = rounded[mask] / tick[mask]
        if self.rule_stress.adverse_tick_rounding and directions is not None:
            direction = np.asarray(directions, dtype=np.float64).reshape(-1)
            if direction.shape != rounded.shape:
                raise ValueError("price-rounding directions do not match symbols")
            buy = mask & (direction > 0.0)
            sell = mask & (direction < 0.0)
            neutral = mask & ~(buy | sell)
            rounded[buy] = np.ceil(scaled[buy]) * tick[buy]
            rounded[sell] = np.floor(scaled[sell]) * tick[sell]
            rounded[neutral] = np.round(scaled[neutral]) * tick[neutral]
        else:
            rounded[mask] = np.round(scaled[mask]) * tick[mask]
        rounded[mask] = np.maximum(rounded[mask], tick[mask])
        if np.any(rounded <= 0.0) or not np.isfinite(rounded).all():
            raise ValueError("rounded execution prices must remain finite and positive")
        return rounded

    def _round_quantities(self, quantities: np.ndarray, *, index: int) -> np.ndarray:
        _, lot, _ = self.effective_rule_arrays(index=index)
        rounded = quantities.copy()
        mask = lot > 0.0
        rounded[mask] = np.trunc(rounded[mask] / lot[mask]) * lot[mask]
        return rounded

    def _capacity_notional(
        self,
        prices: np.ndarray,
        capacity_volume: np.ndarray,
    ) -> np.ndarray:
        result = np.empty_like(prices, dtype=np.float64)
        for index, unit in enumerate(self.dataset.volume_units):
            resolved = VolumeUnit(unit)
            if resolved is VolumeUnit.QUOTE_NOTIONAL:
                result[index] = capacity_volume[index]
            else:
                result[index] = prices[index] * capacity_volume[index]
        return result

    def _constrain_borrow(
        self,
        desired: np.ndarray,
        *,
        current: np.ndarray,
        index: int,
    ) -> np.ndarray:
        result = desired.copy()
        if not self.cost.allow_short:
            return np.maximum(result, 0.0)
        available = self.dataset.resolved_array("borrow_available")[index]
        lower_bound = np.minimum(current, 0.0)
        result[~available] = np.maximum(result[~available], lower_bound[~available])
        return result

    def _flatten_after_termination(self, book: BookState, prices: np.ndarray) -> None:
        value = max(book.portfolio_value, 0.0)
        book.quantities = np.zeros_like(book.quantities)
        book.mark_prices = prices.copy()
        book.cash = value
        book.margin_used = 0.0
        book.maintenance_margin = 0.0
        book.maintenance_requirement = 0.0

    def _update_margin(self, book: BookState) -> None:
        position_values = book.position_values
        gross_notional = float(np.abs(position_values).sum())
        margin_used = gross_notional / self.cost.max_leverage
        maintenance_required = self.cost.maintenance_margin_rate * gross_notional
        book.set_margin(
            margin_used=margin_used,
            maintenance_margin=self.cost.maintenance_margin_rate,
            maintenance_requirement=maintenance_required,
        )
        collateral_equity = float(
            book.cash
            + np.minimum(position_values, 0.0).sum()
            + self.cost.collateral_haircut * np.maximum(position_values, 0.0).sum()
        )
        maintenance_required = self.cost.maintenance_margin_rate * gross_notional
        book.margin_deficit = max(
            book.margin_deficit,
            maintenance_required - collateral_equity,
            0.0,
        )
        if self.cost.margin_mode == "isolated" and gross_notional > 0.0:
            allocations = np.abs(position_values) / gross_notional
            isolated_equity = collateral_equity * allocations
            isolated_required = self.cost.maintenance_margin_rate * np.abs(
                position_values
            )
            isolated_deficit = np.maximum(
                isolated_required - isolated_equity,
                0.0,
            )
            book.margin_deficit = max(
                book.margin_deficit,
                float(np.max(isolated_deficit, initial=0.0)),
            )
            if np.any(isolated_deficit > _TOLERANCE):
                book.terminate(EconomicTerminationReason.MARGIN_CALL)
        elif collateral_equity + _TOLERANCE < maintenance_required:
            book.terminate(EconomicTerminationReason.MARGIN_CALL)
        if book.insolvent:
            self._flatten_after_termination(book, book.mark_prices)

    def _fill_toward_quantities(
        self,
        book: BookState,
        desired_quantities: np.ndarray,
        *,
        prices: np.ndarray,
        capacity_volume: np.ndarray,
        tradable: np.ndarray,
        turnover_denominator: float,
        market_index: int,
        force_market: bool = False,
    ) -> _FillResult:
        reference_prices = np.asarray(prices, dtype=np.float64).reshape(-1)
        capacity_volume_vector = np.asarray(
            capacity_volume,
            dtype=np.float64,
        ).reshape(-1)
        trade_mask = np.asarray(tradable, dtype=np.bool_).reshape(-1)
        trade_mask = (
            trade_mask & self.dataset.resolved_array("asset_active")[market_index]
        )
        desired = np.asarray(desired_quantities, dtype=np.float64).reshape(-1)
        expected_shape = (self.dataset.n_symbols,)
        if any(
            vector.shape != expected_shape
            for vector in (
                reference_prices,
                capacity_volume_vector,
                trade_mask,
                desired,
            )
        ):
            raise ValueError("execution market vectors do not match symbols")
        if not np.isfinite(desired).all():
            raise ValueError("desired quantities must be finite")
        if np.any(reference_prices <= 0.0) or np.any(capacity_volume_vector < 0.0):
            raise ValueError("execution prices and capacity volume are invalid")
        if not math.isfinite(turnover_denominator) or turnover_denominator <= 0.0:
            raise ValueError("turnover denominator must be finite and positive")

        desired = self._constrain_borrow(
            desired,
            current=book.quantities,
            index=market_index,
        )
        desired = self._round_quantities(desired, index=market_index)
        requested_delta = desired - book.quantities
        price_vector = self._round_prices(
            reference_prices,
            index=market_index,
            directions=requested_delta,
        )
        if self.cost.order_type == "limit" and not force_market:
            buy = requested_delta > 0.0
            sell = requested_delta < 0.0
            limit_prices = np.where(
                buy,
                reference_prices * (1.0 - self.cost.limit_offset_rate),
                reference_prices * (1.0 + self.cost.limit_offset_rate),
            )
            limit_prices = self._round_prices(
                limit_prices,
                index=market_index,
                directions=requested_delta,
            )
            open_prices = self.dataset.open[market_index]
            touched = np.where(
                buy,
                self.dataset.low[market_index] <= limit_prices,
                np.where(
                    sell,
                    self.dataset.high[market_index] >= limit_prices,
                    False,
                ),
            )
            price_vector = np.where(
                buy,
                np.where(open_prices <= limit_prices, open_prices, limit_prices),
                np.where(
                    sell,
                    np.where(open_prices >= limit_prices, open_prices, limit_prices),
                    price_vector,
                ),
            )
            price_vector = self._round_prices(
                price_vector,
                index=market_index,
                directions=requested_delta,
            )
            trade_mask = trade_mask & touched
        requested_notional_vector = self.dataset.quantity_notional(
            market_index,
            requested_delta,
            price_vector,
        )
        direction_allowed = np.where(
            requested_notional_vector > 0.0,
            self.dataset.resolved_array("buy_allowed")[market_index],
            self.dataset.resolved_array("sell_allowed")[market_index],
        )
        trade_mask = trade_mask & direction_allowed
        _, _, minimum_notional = self.effective_rule_arrays(index=market_index)
        requested_notional_vector = np.where(
            np.abs(requested_notional_vector) >= minimum_notional,
            requested_notional_vector,
            0.0,
        )
        capacity_notional = self.dataset.market_notional(
            market_index,
            price_vector,
            volume=capacity_volume_vector,
        )
        participation_limit = np.minimum(
            self.dataset.resolved_array("max_participation_rate")[market_index],
            self.cost.max_participation_rate,
        )
        capacity = participation_limit * capacity_notional
        capacity = np.where(trade_mask, capacity, 0.0)
        filled_notional_vector = np.sign(requested_notional_vector) * np.minimum(
            np.abs(requested_notional_vector),
            capacity,
        )
        filled_delta = self.dataset.notional_to_quantity(
            market_index,
            filled_notional_vector,
            price_vector,
        )
        next_quantities = self._round_quantities(
            book.quantities + filled_delta,
            index=market_index,
        )
        filled_notional_vector = self.dataset.quantity_notional(
            market_index,
            next_quantities - book.quantities,
            price_vector,
        )

        positive_capacity = capacity_notional > 0.0
        participation = np.zeros_like(capacity_notional)
        participation[positive_capacity] = (
            np.abs(filled_notional_vector[positive_capacity])
            / capacity_notional[positive_capacity]
        )
        impact = self.cost.impact_rate * np.sqrt(participation)
        slippage = self._slippage_rates(len(price_vector))
        spread_multiplier = 0.5 if self.cost.order_type == "limit" else 1.0
        venue_fee = (
            self.cost.maker_fee_rate
            + self.dataset.resolved_array("maker_fee_rate")[market_index]
            if self.cost.order_type == "limit"
            else self.cost.taker_fee_rate
            + self.dataset.resolved_array("taker_fee_rate")[market_index]
        )
        unit_cost = self.cost.multiplier * (
            self.cost.fee_rate
            + self.dataset.resolved_array("fee_rate")[market_index]
            + venue_fee
            + spread_multiplier
            * (
                self.cost.spread_rate
                + self.dataset.resolved_array("spread_rate")[market_index]
            )
            + impact
            + slippage
        )
        cost_vector = np.abs(filled_notional_vector) * unit_cost
        cost_amount = float(np.sum(cost_vector))
        if not math.isfinite(cost_amount):
            raise ValueError("execution cost became non-finite")

        fills_before = book.fill_count
        events_before = book.rebalance_events
        filled_notional = float(np.abs(filled_notional_vector).sum())
        book.execute(
            fill_prices=price_vector,
            target_quantities=next_quantities,
            cost_amount=cost_amount,
            turnover=filled_notional / turnover_denominator,
        )
        self._update_margin(book)
        return _FillResult(
            requested_notional=float(np.abs(requested_notional_vector).sum()),
            filled_notional=filled_notional,
            cost_amount=cost_amount,
            fill_count=book.fill_count - fills_before,
            rebalance_events=book.rebalance_events - events_before,
            max_participation=float(np.max(participation, initial=0.0)),
            requested_notional_by_symbol=np.abs(requested_notional_vector),
            filled_notional_by_symbol=np.abs(filled_notional_vector),
            participation_by_symbol=participation,
            cost_by_symbol=cost_vector,
        )

    def _charge_carry(self, book: BookState, *, index: int) -> tuple[float, float]:
        funding_return = -float(
            np.dot(
                book.weights,
                self.dataset.funding_rate[index]
                * self.dataset.resolved_array("funding_due")[index].astype(np.float64),
            )
        )
        funding_amount = book.portfolio_value * funding_return
        short_values = np.maximum(-book.position_values, 0.0)
        previous_index = max(0, index - 1)
        year_fraction = self.dataset.elapsed_year_fraction(previous_index, index)
        borrow_amount = float(
            np.sum(short_values * self.dataset.resolved_array("borrow_rate")[index])
            * year_fraction
            * self.cost.borrow_rate_multiplier
        )
        if borrow_amount > 0.0:
            book.charge_borrow(borrow_amount)
        return funding_amount, borrow_amount

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
        if not np.array_equal(
            np.asarray(book.contract_multipliers),
            self.dataset.resolved_array("contract_multipliers"),
        ):
            raise ValueError("book contract multipliers do not match market dataset")

        resolved_target = _target_weights(
            target,
            n_symbols=self.dataset.n_symbols,
            maximum_gross=self.cost.max_leverage,
        )
        result_book = book.clone()
        starting_value = result_book.portfolio_value
        if starting_value <= 0.0:
            raise ValueError("execution requires positive starting equity")
        decision_equity = 0.0
        initial_requested_notional = 0.0
        filled_notional_total = 0.0
        total_cost = 0.0
        total_funding = 0.0
        total_borrow = 0.0
        total_dividend = 0.0
        total_cash_interest = 0.0
        total_fills = 0
        total_events = 0
        max_participation = 0.0
        requested_by_symbol = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        filled_by_symbol = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        participation_by_symbol = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        cost_by_symbol = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        gross_factor = 1.0
        desired_quantities: np.ndarray | None = None

        for offset in range(bars):
            close_index = start_index + offset
            next_index = close_index + 1
            period_start_value = max(result_book.portfolio_value, _TOLERANCE)
            result_book.apply_split(
                self.dataset.resolved_array("split_factor")[next_index]
            )
            inactive = ~self.dataset.resolved_array("asset_active")[next_index]
            if np.any(inactive & (np.abs(result_book.quantities) > _TOLERANCE)):
                result_book.settle_positions(
                    mask=inactive,
                    prices=self.dataset.open[next_index],
                    recovery=self.dataset.resolved_array("delisting_recovery")[
                        next_index
                    ],
                )
            result_book.revalue(self.dataset.open[next_index])
            value_at_open = max(result_book.portfolio_value, 0.0)
            gap_return = value_at_open / period_start_value - 1.0

            latency_elapsed = offset >= self.cost.order_latency_bars
            if desired_quantities is None and latency_elapsed:
                decision_equity = max(value_at_open, _TOLERANCE)
                desired_quantities = self.dataset.notional_to_quantity(
                    next_index,
                    resolved_target * decision_equity,
                    self.dataset.open[next_index],
                )
                requested_by_symbol = np.abs(
                    self.dataset.quantity_notional(
                        next_index,
                        desired_quantities - result_book.quantities,
                        self.dataset.open[next_index],
                    )
                )
                initial_requested_notional = float(requested_by_symbol.sum())

            if desired_quantities is not None and not result_book.insolvent:
                fill = self._fill_toward_quantities(
                    result_book,
                    desired_quantities,
                    prices=self.dataset.open[next_index],
                    capacity_volume=self.dataset.volume[close_index],
                    tradable=self.dataset.tradable[next_index],
                    turnover_denominator=decision_equity,
                    market_index=next_index,
                )
                filled_notional_total += fill.filled_notional
                total_cost += fill.cost_amount
                total_fills += fill.fill_count
                total_events += fill.rebalance_events
                max_participation = max(max_participation, fill.max_participation)
                filled_by_symbol += fill.filled_notional_by_symbol
                participation_by_symbol = np.maximum(
                    participation_by_symbol,
                    fill.participation_by_symbol,
                )
                cost_by_symbol += fill.cost_by_symbol

            if result_book.insolvent:
                self._flatten_after_termination(
                    result_book,
                    self.dataset.open[next_index],
                )

            intrabar_asset_returns = (
                self.dataset.resolved_array("mark_price")[next_index]
                / self.dataset.open[next_index]
                - 1.0
            )
            intrabar_return = float(np.dot(result_book.weights, intrabar_asset_returns))
            gross_factor *= max(
                (1.0 + gap_return) * (1.0 + intrabar_return),
                _TOLERANCE,
            )
            dividend_amount = result_book.apply_dividend(
                self.dataset.resolved_array("dividend")[next_index]
            )
            cash_interest = result_book.apply_cash_interest(
                float(self.dataset.resolved_array("cash_rate")[next_index]),
                year_fraction=self.dataset.elapsed_year_fraction(
                    close_index,
                    next_index,
                ),
            )
            total_dividend += dividend_amount
            total_cash_interest += cash_interest
            funding_amount, borrow_amount = self._charge_carry(
                result_book,
                index=next_index,
            )
            total_funding += funding_amount
            total_borrow += borrow_amount
            result_book.mark_to_market(
                mark_prices=self.dataset.resolved_array("mark_price")[next_index],
                funding_amount=funding_amount,
                period_start_value=period_start_value,
            )
            self._update_margin(result_book)
            if result_book.insolvent:
                self._flatten_after_termination(
                    result_book,
                    self.dataset.resolved_array("mark_price")[next_index],
                )

        ending_value = max(result_book.portfolio_value, 0.0)
        interval_net_return = max(
            ending_value / starting_value - 1.0,
            -1.0 + 1e-12,
        )
        denominator = max(decision_equity, starting_value, _TOLERANCE)
        requested_turnover = initial_requested_notional / denominator
        filled_turnover = filled_notional_total / denominator
        unfilled_turnover = max(0.0, requested_turnover - filled_turnover)
        fill_ratio = (
            1.0
            if initial_requested_notional <= _TOLERANCE
            else min(1.0, filled_notional_total / initial_requested_notional)
        )
        reason = (
            None
            if result_book.termination_reason is None
            else EconomicTerminationReason(result_book.termination_reason).value
        )
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
            fill_ratio=fill_ratio,
            max_participation=max_participation,
            interval_borrow_cost=total_borrow,
            interval_dividend=total_dividend,
            interval_cash_interest=total_cash_interest,
            margin_utilization=result_book.margin_utilization,
            termination_reason=reason,
            requested_notional_by_symbol=requested_by_symbol,
            filled_notional_by_symbol=filled_by_symbol,
            participation_by_symbol=participation_by_symbol,
            cost_by_symbol=cost_by_symbol,
        )

    def liquidate_at_close(self, book: BookState, *, index: int) -> ExecutionResult:
        if not 0 <= index < self.dataset.n_bars:
            raise ValueError("liquidation index is outside the dataset")
        result_book = book.clone()
        result_book.revalue(self.dataset.resolved_array("mark_price")[index])
        starting_value = max(result_book.portfolio_value, _TOLERANCE)
        desired_quantities = np.zeros(self.dataset.n_symbols, dtype=np.float64)
        fill = self._fill_toward_quantities(
            result_book,
            desired_quantities,
            prices=self.dataset.resolved_array("mark_price")[index],
            capacity_volume=self.dataset.volume[index],
            tradable=self.dataset.tradable[index],
            turnover_denominator=starting_value,
            market_index=index,
            force_market=True,
        )
        interval_net_return = max(
            max(result_book.portfolio_value, 0.0) / starting_value - 1.0,
            -1.0 + 1e-12,
        )
        requested_turnover = fill.requested_notional / starting_value
        filled_turnover = fill.filled_notional / starting_value
        if filled_turnover + _TOLERANCE >= requested_turnover:
            result_book.termination_reason = EconomicTerminationReason.LIQUIDATION
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
            fill_ratio=(
                1.0
                if fill.requested_notional <= _TOLERANCE
                else min(1.0, fill.filled_notional / fill.requested_notional)
            ),
            max_participation=fill.max_participation,
            margin_utilization=result_book.margin_utilization,
            termination_reason=(
                None
                if result_book.termination_reason is None
                else EconomicTerminationReason(result_book.termination_reason).value
            ),
            requested_notional_by_symbol=fill.requested_notional_by_symbol,
            filled_notional_by_symbol=fill.filled_notional_by_symbol,
            participation_by_symbol=fill.participation_by_symbol,
            cost_by_symbol=fill.cost_by_symbol,
        )
