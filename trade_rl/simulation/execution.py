"""Next-open execution, liquidity costs, funding, borrow and margin accounting."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from trade_rl._array_snapshot import checked_array_input
from trade_rl._validation import (
    _ordered_ns_durations,
    _require_nominal_session_seconds,
    require_sha256,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import MarketCalendarKind, VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.data.market_order_rules import (
    MarketOrderProfile,
    MarketOrderRule,
    joint_lot_size,
)
from trade_rl.simulation.accounting import (
    BookState,
    EconomicTerminationReason,
)
from trade_rl.simulation.diagnostics.fill_cash import NativeFillCashObservation
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    TerminalOrderArchive,
)
from trade_rl.simulation.orders.model import (
    execution_policy_digest as calculate_execution_policy_digest,
)
from trade_rl.simulation.quantities import exact_quantity, project_quantity
from trade_rl.simulation.stateful.execution import (
    StatefulExecutionObservation,
    StatefulExecutionResult,
    _require_native_execution_rows,
    execute_stateful_orders,
)
from trade_rl.simulation.targets.execution import execute_target_statefully

_TOLERANCE = 1e-12
_ATTEMPT_EVENT_TYPES = frozenset(
    {"eligible", "triggered", "no_fill", "partial_fill", "filled", "rejected"}
)


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
    path_mode: str = "conservative"
    processing_bar_volume_capacity: bool = False
    partial_fill_carry: bool = True
    trigger_volume_fractions: tuple[float, float, float, float] = (
        1.0,
        0.5,
        0.25,
        0.0,
    )

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
        if self.tail_slippage_probability > 0.0 and self.tail_slippage_multiplier < 1.0:
            raise ValueError(
                "tail_slippage_multiplier must be at least 1.0 when tail events are enabled"
            )
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
        if self.order_type not in {"market", "limit", "stop_market"}:
            raise ValueError("order_type must be 'market', 'limit' or 'stop_market'")
        if self.limit_offset_rate >= 1.0:
            raise ValueError("limit_offset_rate must be below one")
        if not isinstance(self.allow_short, bool):
            raise ValueError("allow_short must be a boolean")
        if self.path_mode not in {"optimistic", "neutral", "conservative"}:
            raise ValueError("path_mode must be optimistic, neutral or conservative")
        if not isinstance(self.processing_bar_volume_capacity, bool):
            raise ValueError("processing_bar_volume_capacity must be a boolean")
        if not isinstance(self.partial_fill_carry, bool):
            raise ValueError("partial_fill_carry must be a boolean")
        fractions = self.trigger_volume_fractions
        if (
            len(fractions) != 4
            or not all(math.isfinite(value) for value in fractions)
            or not all(0.0 <= value <= 1.0 for value in fractions)
            or not all(fractions[index] >= fractions[index + 1] for index in range(3))
        ):
            raise ValueError(
                "trigger_volume_fractions must be four non-increasing rates"
            )

    def execution_policy_payload(self) -> dict[str, object]:
        return {
            "allow_short": self.allow_short,
            "borrow_rate_multiplier": self.borrow_rate_multiplier,
            "collateral_haircut": self.collateral_haircut,
            "fee_rate": self.fee_rate,
            "impact_rate": self.impact_rate,
            "limit_offset_rate": self.limit_offset_rate,
            "lot_size": self.lot_size,
            "maintenance_margin_rate": self.maintenance_margin_rate,
            "maker_fee_rate": self.maker_fee_rate,
            "margin_mode": self.margin_mode,
            "max_leverage": self.max_leverage,
            "max_participation_rate": self.max_participation_rate,
            "minimum_notional": self.minimum_notional,
            "multiplier": self.multiplier,
            "order_latency_bars": self.order_latency_bars,
            "order_type": self.order_type,
            "partial_fill_carry": self.partial_fill_carry,
            "path_mode": self.path_mode,
            "processing_bar_volume_capacity": self.processing_bar_volume_capacity,
            "random_seed": self.random_seed,
            "schema_version": "execution_policy_v2",
            "slippage_std": self.slippage_std,
            "spread_rate": self.spread_rate,
            "tail_slippage_multiplier": self.tail_slippage_multiplier,
            "tail_slippage_probability": self.tail_slippage_probability,
            "taker_fee_rate": self.taker_fee_rate,
            "tick_size": self.tick_size,
            "trigger_volume_fractions": list(self.trigger_volume_fractions),
        }

    @property
    def execution_policy_digest(self) -> str:
        return calculate_execution_policy_digest(self.execution_policy_payload())

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


def _runtime_execution_policy_digest(
    cost: ExecutionCostConfig,
    *,
    rule_stress: ExecutionRuleStress,
    market_order_profile_digest: str | None,
    insolvency_valuation: Literal["floor_zero", "retain_debt"],
    funding_price_rate_present: bool,
    strict_spot_cash: bool,
    native_fixed_linear_carry: bool = False,
    native_calendar_month_linear_carry: bool = False,
    native_session_linear_carry: bool = False,
) -> str:
    """Single owner of the historical native digest composition."""
    if market_order_profile_digest is not None:
        digest = content_digest(
            {
                "schema_version": "profile_market_execution_v1",
                "base_policy_digest": cost.execution_policy_digest,
                "market_order_profile_digest": market_order_profile_digest,
                "rule_stress": rule_stress.digest_payload(),
            }
        )
    elif rule_stress.enabled:
        digest = content_digest(
            {
                "schema_version": "stressed_execution_policy_v1",
                "base_policy_digest": cost.execution_policy_digest,
                "rule_stress": rule_stress.digest_payload(),
            }
        )
    else:
        digest = cost.execution_policy_digest

    if insolvency_valuation == "retain_debt":
        digest = content_digest(
            {
                "schema_version": "insolvency_execution_policy_v2",
                "base_policy_digest": digest,
                "insolvency_valuation": insolvency_valuation,
                "loss_metric_semantics": "unbounded_simple_return_drawdown_v1",
            }
        )

    if funding_price_rate_present:
        digest = content_digest(
            {
                "schema_version": "funding_execution_policy_v1",
                "base_policy_digest": digest,
                "funding_semantics": "dataset_funding_price_rate_v1",
            }
        )

    if strict_spot_cash:
        digest = content_digest(
            {
                "schema_version": "strict_spot_cash_execution_v1",
                "base_policy_digest": digest,
                "unaffordable_fill": "no_fill",
            }
        )
    if (
        native_fixed_linear_carry
        or native_calendar_month_linear_carry
        or native_session_linear_carry
    ):
        digest = content_digest(
            {
                "schema_version": (
                    "native_session_linear_carry_execution_v1"
                    if native_session_linear_carry
                    else "native_calendar_month_linear_carry_execution_v1"
                    if native_calendar_month_linear_carry
                    else "native_fixed_linear_carry_execution_v1"
                ),
                "base_policy_digest": digest,
                "funding_semantics": "single_declared_close_event_post_open_fill",
                "borrow_semantics": (
                    "previous_close_gap_and_post_open_short_mark_actual_act365f"
                    if native_session_linear_carry
                    else "post_open_short_mark_actual_act365f"
                ),
            }
        )
    return digest


def _require_native_linear_carry_configuration(
    cost: ExecutionCostConfig,
    *,
    calendar_month: bool = False,
    session: bool = False,
    rule_stress: ExecutionRuleStress | None = None,
    market_order_profile_present: bool = False,
) -> None:
    if (
        type(cost) is not ExecutionCostConfig
        or cost.order_type != "market"
        or cost.margin_mode != "cross"
        or cost.processing_bar_volume_capacity
        or cost.max_leverage > 1.0
        or cost.collateral_haircut != 1.0
        or cost.slippage_std != 0.0
        or cost.tail_slippage_probability != 0.0
        or cost.order_latency_bars != 0
        or cost.partial_fill_carry
    ):
        raise ValueError(
            (
                "native session"
                if session
                else "native calendar month"
                if calendar_month
                else "native fixed"
            )
            + " linear carry requires deterministic unlevered cross-margin "
            "MARKET orders, previous-bar capacity, zero latency and no partial-fill carry"
        )
    if (calendar_month or session) and (
        market_order_profile_present
        or (
            rule_stress is not None
            and (type(rule_stress) is not ExecutionRuleStress or rule_stress.enabled)
        )
    ):
        raise ValueError(
            ("native session" if session else "native calendar month")
            + " linear carry excludes market profiles and rule stress"
        )


def _require_session_carry_mode(
    native_session_linear_carry: bool,
    *,
    native_fixed_linear_carry: bool,
    native_calendar_month_linear_carry: bool,
) -> None:
    if type(native_session_linear_carry) is not bool:
        raise ValueError("native_session_linear_carry must be boolean")
    if native_session_linear_carry and (
        native_fixed_linear_carry or native_calendar_month_linear_carry
    ):
        raise ValueError(
            "session and fixed/calendar month linear carry modes are exclusive"
        )


def runtime_execution_policy_digest(
    cost: ExecutionCostConfig,
    *,
    funding_price_rate_present: bool,
    rule_stress: ExecutionRuleStress | None = None,
    market_order_profile_digest: str | None = None,
    insolvency_valuation: Literal["floor_zero", "retain_debt"] = "floor_zero",
    strict_spot_cash: bool = False,
    native_fixed_linear_carry: bool = False,
    native_calendar_month_linear_carry: bool = False,
    native_session_linear_carry: bool = False,
) -> str:
    """Compute native execution identity from independently admitted metadata.

    The funding flag explicitly distinguishes legacy bar-mark funding from
    the optional event-price-rate array, even when every rate is zero. A
    profile digest is an independently verified identity, not source evidence
    merely because the caller supplies it. No dataset, arrays, execution,
    or policy side effects are required. This identifies the declared runtime
    configuration; it does not validate a profile against a dataset, validate
    rule arrays or processed funding rows, or authorize replay.
    """
    if type(cost) is not ExecutionCostConfig:
        raise ValueError("cost must be an ExecutionCostConfig")
    if rule_stress is not None and type(rule_stress) is not ExecutionRuleStress:
        raise ValueError("rule_stress must be an ExecutionRuleStress")
    if type(funding_price_rate_present) is not bool:
        raise ValueError("funding_price_rate_present must be boolean")
    if type(strict_spot_cash) is not bool:
        raise ValueError("strict_spot_cash must be boolean")
    if type(native_fixed_linear_carry) is not bool:
        raise ValueError("native_fixed_linear_carry must be boolean")
    if type(native_calendar_month_linear_carry) is not bool:
        raise ValueError("native_calendar_month_linear_carry must be boolean")
    if native_fixed_linear_carry and native_calendar_month_linear_carry:
        raise ValueError("fixed and calendar month linear carry modes are exclusive")
    _require_session_carry_mode(
        native_session_linear_carry,
        native_fixed_linear_carry=native_fixed_linear_carry,
        native_calendar_month_linear_carry=native_calendar_month_linear_carry,
    )
    if (
        native_fixed_linear_carry
        or native_calendar_month_linear_carry
        or native_session_linear_carry
    ):
        if strict_spot_cash or not funding_price_rate_present:
            raise ValueError(
                (
                    "native session"
                    if native_session_linear_carry
                    else "native calendar month"
                    if native_calendar_month_linear_carry
                    else "native fixed"
                )
                + " linear carry requires explicit event-price rates and excludes strict Spot"
            )
        _require_native_linear_carry_configuration(
            cost,
            calendar_month=native_calendar_month_linear_carry,
            session=native_session_linear_carry,
            rule_stress=rule_stress,
            market_order_profile_present=market_order_profile_digest is not None,
        )
    mode: object = insolvency_valuation
    if type(mode) is not str or mode not in (
        "floor_zero",
        "retain_debt",
    ):
        raise ValueError("insolvency_valuation must be floor_zero or retain_debt")
    if market_order_profile_digest is not None:
        if type(market_order_profile_digest) is not str:
            raise ValueError("market_order_profile_digest must be a digest")
        require_sha256(market_order_profile_digest, field="market_order_profile_digest")
        if market_order_profile_digest == "0" * 64:
            raise ValueError("market_order_profile_digest must not be null")
        if cost.order_type != "market":
            raise ValueError("market order profile requires MARKET configuration")
    if strict_spot_cash and (
        cost.allow_short
        or cost.max_leverage > 1.0
        or cost.collateral_haircut != 1.0
        or cost.slippage_std != 0.0
        or cost.tail_slippage_probability != 0.0
    ):
        raise ValueError(
            "strict spot cash requires long-only unlevered deterministic execution and full collateral"
        )
    return _runtime_execution_policy_digest(
        cost,
        rule_stress=rule_stress if rule_stress is not None else ExecutionRuleStress(),
        market_order_profile_digest=market_order_profile_digest,
        insolvency_valuation=insolvency_valuation,
        funding_price_rate_present=funding_price_rate_present,
        strict_spot_cash=strict_spot_cash,
        native_fixed_linear_carry=native_fixed_linear_carry,
        native_calendar_month_linear_carry=native_calendar_month_linear_carry,
        native_session_linear_carry=native_session_linear_carry,
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
    original_requested_notional: float
    original_requested_notional_by_symbol: np.ndarray
    filled_notional: float
    cost_amount: float
    fill_count: int
    rebalance_events: int
    max_participation: float
    requested_notional_by_symbol: np.ndarray
    filled_notional_by_symbol: np.ndarray
    participation_by_symbol: np.ndarray
    cost_by_symbol: np.ndarray


def _require_native_execution_cost(
    dataset: MarketDataset, cost: ExecutionCostConfig
) -> None:
    if (
        cost.order_type != "market"
        or cost.processing_bar_volume_capacity
        or cost.allow_short
        or cost.max_leverage > 1.0
        or cost.collateral_haircut != 1.0
        or cost.slippage_std != 0.0
        or cost.tail_slippage_probability != 0.0
    ):
        raise ValueError(
            (
                "native calendar month"
                if dataset.native_calendar_month
                else "native fixed"
            )
            + " execution requires deterministic unlevered long-only MARKET orders with previous-bar capacity"
        )


def _require_execution_margin_roster(
    dataset: MarketDataset, cost: ExecutionCostConfig
) -> None:
    if cost.margin_mode == "isolated" and dataset.n_symbols != 1:
        raise ValueError(
            "isolated margin requires a per-symbol collateral ledger; "
            "only single-asset execution is supported"
        )


def _require_calendar_month_spot_roster(dataset: MarketDataset) -> None:
    if dataset.native_calendar_month and (
        any(
            unit not in (VolumeUnit.BASE_ASSET, VolumeUnit.QUOTE_NOTIONAL)
            for unit in dataset.volume_units
        )
        or np.any(dataset.contract_multipliers != 1.0)
    ):
        raise ValueError(
            "calendar month execution requires unit-multiplier Spot volume"
        )


def _preflight_session_execution_window(
    dataset: MarketDataset,
    execution_cost: ExecutionCostConfig,
    *,
    start_index: int,
    stop_index: int,
    expected_boundaries_ns: tuple[int, ...],
    expected_nominal_bar_seconds: int,
    native_session_linear_carry: bool = False,
) -> tuple[int, ...]:
    """Match a caller-established session grid before entering execution."""
    if (
        type(dataset) is not MarketDataset
        or dataset.calendar_kind is not MarketCalendarKind.SESSION
        or dataset.native_clock is not None
        or not dataset.identity_verified
    ):
        raise ValueError("expected a verified declared session Dataset")
    if (
        type(start_index) is not int
        or type(stop_index) is not int
        or not 0 <= start_index < stop_index < dataset.n_bars
    ):
        raise ValueError("invalid session elapsed execution window")
    boundaries = tuple(
        int(x) for x in dataset.timestamps[start_index : stop_index + 1].view(np.int64)
    )
    durations = _ordered_ns_durations(expected_boundaries_ns)
    if boundaries != expected_boundaries_ns:
        raise ValueError("session Dataset differs from the expected boundary sequence")
    _require_nominal_session_seconds(expected_nominal_bar_seconds, durations)
    if dataset.bar_hours != expected_nominal_bar_seconds / 3600:
        raise ValueError("session nominal processing duration differs from the Dataset")
    if type(execution_cost) is not ExecutionCostConfig:
        raise ValueError("expected explicit session execution cost")
    if type(native_session_linear_carry) is not bool:
        raise ValueError("native_session_linear_carry must be boolean")
    if native_session_linear_carry:
        _require_session_linear_carry_dataset(dataset)
        _require_native_linear_carry_configuration(execution_cost, session=True)
        _require_native_execution_rows(
            dataset,
            start_index=start_index,
            stop_index=stop_index,
            native_linear_carry=True,
            native_session_linear_carry=True,
        )
        return boundaries
    try:
        _require_native_execution_cost(dataset, execution_cost)
    except ValueError as error:
        raise ValueError(
            "session elapsed replay requires deterministic unlevered long-only MARKET with previous-bar capacity"
        ) from error
    span = slice(start_index, stop_index + 1)
    processed = slice(start_index + 1, stop_index + 1)
    price_rate = dataset.funding_price_rate
    if (
        any(
            unit not in (VolumeUnit.BASE_ASSET, VolumeUnit.QUOTE_NOTIONAL)
            for unit in dataset.volume_units
        )
        or np.any(dataset.resolved_array("contract_multipliers") != 1.0)
        or np.any(~dataset.resolved_array("information_available")[span])
        or np.any(
            dataset.resolved_array("available_at")[span]
            > dataset.timestamps[span, None]
        )
        or np.any(~dataset.resolved_array("asset_active")[span])
        or np.any(~dataset.tradable[span])
        or np.any(dataset.resolved_array("funding_event_count")[processed])
        or np.any(dataset.funding_rate[processed] != 0.0)
        or (price_rate is not None and np.any(price_rate[processed] != 0.0))
        or np.any(dataset.resolved_array("dividend")[processed] != 0.0)
        or np.any(dataset.resolved_array("split_factor")[processed] != 1.0)
        or np.any(dataset.resolved_array("cash_rate")[processed] < 0.0)
    ):
        raise ValueError(
            "session elapsed replay requires complete available Spot rows without funding or corporate actions"
        )
    _require_execution_margin_roster(dataset, execution_cost)
    return boundaries


def _require_session_linear_carry_dataset(dataset: MarketDataset) -> None:
    if (
        type(dataset) is not MarketDataset
        or dataset.calendar_kind is not MarketCalendarKind.SESSION
        or dataset.native_clock is not None
        or not dataset.identity_verified
        or dataset.nominal_bar_hours is None
        or dataset.funding_price_rate is None
        or any(unit is not VolumeUnit.CONTRACTS for unit in dataset.volume_units)
    ):
        raise ValueError(
            "native session linear carry requires a verified session contract-unit Dataset "
            "with explicit nominal bar duration and event-price rates"
        )


def _require_native_linear_carry_dataset(
    dataset: MarketDataset, *, calendar_month: bool = False
) -> None:
    if (
        dataset.native_clock is None
        or dataset.native_calendar_month != calendar_month
        or dataset.funding_price_rate is None
        or any(unit is not VolumeUnit.CONTRACTS for unit in dataset.volume_units)
    ):
        raise ValueError(
            ("native calendar month" if calendar_month else "native fixed")
            + " linear carry requires an explicit matching contract-unit Dataset "
            "with event-price rates"
        )


def _preflight_native_learning_execution_window(
    dataset: MarketDataset,
    cost: ExecutionCostConfig,
    *,
    start_index: int,
    stop_index: int,
    native_calendar_month: bool = False,
    native_fixed_linear_carry: bool = False,
    native_calendar_month_linear_carry: bool = False,
    market_order_profile_present: bool = False,
) -> None:
    """Refuse unsupported native learning inputs before reserving work."""
    if (
        type(native_calendar_month) is not bool
        or native_calendar_month != dataset.native_calendar_month
    ):
        raise ValueError("calendar month execution is not supported by fixed-clock Gym")
    if type(native_fixed_linear_carry) is not bool:
        raise ValueError("native_fixed_linear_carry must be boolean")
    if type(native_calendar_month_linear_carry) is not bool:
        raise ValueError("native_calendar_month_linear_carry must be boolean")
    if native_fixed_linear_carry and native_calendar_month_linear_carry:
        raise ValueError("fixed and calendar month linear carry modes are exclusive")
    linear_carry = native_fixed_linear_carry or native_calendar_month_linear_carry
    if linear_carry:
        _require_native_linear_carry_dataset(
            dataset, calendar_month=native_calendar_month_linear_carry
        )
    if dataset.native_clock is None:
        return
    if not dataset.identity_verified:
        raise ValueError("native fixed execution requires a verified Dataset identity")
    if linear_carry:
        _require_native_linear_carry_configuration(
            cost,
            calendar_month=native_calendar_month_linear_carry,
            market_order_profile_present=market_order_profile_present,
        )
    else:
        _require_calendar_month_spot_roster(dataset)
        _require_native_execution_cost(dataset, cost)
    _require_execution_margin_roster(dataset, cost)
    _require_native_execution_rows(
        dataset,
        start_index=start_index,
        stop_index=stop_index,
        native_linear_carry=linear_carry,
    )


class MarketExecutor:
    """Execute one decision target while holding filled quantities.

    The public execution observer receives a cumulative terminal-order archive.
    The internal trace observer receives only the suffix appended this call.
    """

    def __init__(
        self,
        dataset: MarketDataset,
        cost: ExecutionCostConfig | None = None,
        *,
        rule_stress: ExecutionRuleStress | None = None,
        market_order_profile: MarketOrderProfile | None = None,
        execution_observer: Callable[[StatefulExecutionObservation], None]
        | None = None,
        execution_trace_observer: Callable[[StatefulExecutionObservation], None]
        | None = None,
        insolvency_valuation: Literal["floor_zero", "retain_debt"] = "floor_zero",
        strict_spot_cash: bool = False,
        native_calendar_month: bool = False,
        native_fixed_linear_carry: bool = False,
        native_calendar_month_linear_carry: bool = False,
        native_session_linear_carry: bool = False,
        fill_cash_observer: Callable[[NativeFillCashObservation], None] | None = None,
    ) -> None:
        if (
            type(native_calendar_month) is not bool
            or native_calendar_month != dataset.native_calendar_month
        ):
            raise ValueError(
                "calendar month execution requires explicit matching opt-in"
            )
        if type(native_fixed_linear_carry) is not bool:
            raise ValueError("native_fixed_linear_carry must be boolean")
        if type(native_calendar_month_linear_carry) is not bool:
            raise ValueError("native_calendar_month_linear_carry must be boolean")
        if native_fixed_linear_carry and native_calendar_month_linear_carry:
            raise ValueError(
                "fixed and calendar month linear carry modes are exclusive"
            )
        _require_session_carry_mode(
            native_session_linear_carry,
            native_fixed_linear_carry=native_fixed_linear_carry,
            native_calendar_month_linear_carry=native_calendar_month_linear_carry,
        )
        if native_session_linear_carry:
            if strict_spot_cash:
                raise ValueError("native session linear carry excludes strict Spot")
            _require_session_linear_carry_dataset(dataset)
        elif native_fixed_linear_carry or native_calendar_month_linear_carry:
            if strict_spot_cash:
                raise ValueError("native linear carry excludes strict Spot")
            _require_native_linear_carry_dataset(
                dataset, calendar_month=native_calendar_month_linear_carry
            )
        else:
            _require_calendar_month_spot_roster(dataset)
        self._native_fixed_linear_carry = native_fixed_linear_carry
        self._native_calendar_month_linear_carry = native_calendar_month_linear_carry
        self._native_session_linear_carry = native_session_linear_carry
        self.dataset = dataset
        self._native_fixed_dataset = (
            dataset
            if dataset.native_clock is not None or native_session_linear_carry
            else None
        )
        if self._native_fixed_dataset is not None and not dataset.identity_verified:
            raise ValueError(
                "native fixed execution requires a verified Dataset identity"
            )
        self.cost = cost or ExecutionCostConfig()
        self.rule_stress = (
            rule_stress if rule_stress is not None else ExecutionRuleStress()
        )
        self.market_order_profile = market_order_profile
        self._validate_native_fixed_configuration()
        if type(strict_spot_cash) is not bool:
            raise ValueError("strict_spot_cash must be boolean")
        self._strict_spot_cash = strict_spot_cash or (
            self._native_fixed_dataset is not None and not self._native_linear_carry
        )
        self._validate_strict_spot_configuration()
        self.insolvency_valuation = insolvency_valuation
        _require_execution_margin_roster(self.dataset, self.cost)
        if execution_observer is not None and not callable(execution_observer):
            raise TypeError("execution_observer must be callable")
        if execution_trace_observer is not None and not callable(
            execution_trace_observer
        ):
            raise TypeError("execution_trace_observer must be callable")
        if fill_cash_observer is not None and not callable(fill_cash_observer):
            raise TypeError("fill_cash_observer must be callable")
        self._fill_cash_observer = fill_cash_observer
        self._execution_observer = execution_observer
        self._execution_trace_observer = execution_trace_observer
        self._execution_policy_digest_cache: str | None = None
        self._execution_policy_digest_cache_inputs: (
            tuple[
                ExecutionCostConfig,
                ExecutionRuleStress,
                MarketOrderProfile | None,
                tuple[str, ...],
                str,
                str | None,
                bool,
                bool,
                bool,
                bool,
            ]
            | None
        ) = None
        self._effective_rule_array_cache_inputs: (
            tuple[
                MarketDataset,
                ExecutionCostConfig,
                ExecutionRuleStress,
                MarketOrderProfile | None,
            ]
            | None
        ) = None
        self._effective_rule_array_cache: dict[
            int, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        if market_order_profile is not None:
            if type(market_order_profile) is not MarketOrderProfile:
                raise ValueError(
                    "market order profile requires a verified factory value"
                )
            market_order_profile.validate_dataset(dataset)
            if self.cost.order_type != "market":
                raise ValueError("market order profile requires MARKET configuration")
        self._validate_rule_stress()
        self._rng = np.random.default_rng(self.cost.random_seed)
        self._compatibility_order_book = OrderBookState.empty()
        self._compatibility_last_book: BookState | None = None

    @property
    def strict_spot_cash(self) -> bool:
        """Opt-in cash-only native order admission (not venue spot attestation)."""
        return self._strict_spot_cash

    @property
    def native_fixed_linear_carry(self) -> bool:
        """Declared discrete fixed-grid contract-unit carry, not venue admission."""
        return self._native_fixed_linear_carry

    @property
    def native_calendar_month_linear_carry(self) -> bool:
        """Declared UTC-month close-event carry, not venue admission."""
        return self._native_calendar_month_linear_carry

    @property
    def native_session_linear_carry(self) -> bool:
        """Declared session gap and single close-event carry, not venue admission."""
        return self._native_session_linear_carry

    @property
    def _native_linear_carry(self) -> bool:
        return (
            self.native_fixed_linear_carry
            or self.native_calendar_month_linear_carry
            or self.native_session_linear_carry
        )

    def _validate_native_fixed_configuration(self) -> None:
        if (
            self._native_fixed_dataset is not None
            and self.dataset is not self._native_fixed_dataset
        ) or (
            self._native_fixed_dataset is None and self.dataset.native_clock is not None
        ):
            raise ValueError(
                "native fixed execution requires a fresh executor for a different Dataset"
            )
        if self.dataset.native_clock is not None or self.native_session_linear_carry:
            if self._native_linear_carry:
                _require_native_linear_carry_configuration(
                    self.cost,
                    calendar_month=self.native_calendar_month_linear_carry,
                    session=self.native_session_linear_carry,
                    rule_stress=self.rule_stress,
                    market_order_profile_present=self.market_order_profile is not None,
                )
            else:
                _require_native_execution_cost(self.dataset, self.cost)

    def _validate_strict_spot_configuration(self) -> None:
        if self.strict_spot_cash and (
            self.cost.allow_short
            or self.cost.max_leverage > 1.0
            or self.cost.collateral_haircut != 1.0
            or self.cost.slippage_std != 0.0
            or self.cost.tail_slippage_probability != 0.0
        ):
            raise ValueError(
                "strict spot cash requires long-only unlevered deterministic execution and full collateral"
            )

    def _validate_strict_spot_row(self, index: int) -> None:
        """Check only the processed row, never expose a future price to a policy."""
        if not self.strict_spot_cash:
            return
        due = self.dataset.resolved_array("funding_due")[index]
        rate = (
            self.dataset.funding_rate[index]
            if self.dataset.funding_price_rate is None
            else self.dataset.funding_price_rate[index]
        )
        if np.any(due & (rate != 0.0)):
            raise ValueError("strict spot cash refuses nonzero funding")
        if (
            np.any(self.dataset.resolved_array("dividend")[index] < 0.0)
            or self.dataset.resolved_array("cash_rate")[index] < 0.0
        ):
            raise ValueError("strict spot cash refuses negative non-trade cash flows")

    def _get_insolvency_valuation(self) -> Literal["floor_zero", "retain_debt"]:
        return self._insolvency_valuation

    def _set_insolvency_valuation(
        self, value: Literal["floor_zero", "retain_debt"]
    ) -> None:
        if not isinstance(value, str) or value not in ("floor_zero", "retain_debt"):
            raise ValueError("insolvency_valuation must be floor_zero or retain_debt")
        self._insolvency_valuation = value

    insolvency_valuation = property(
        _get_insolvency_valuation, _set_insolvency_valuation
    )

    def _base_rule_array(
        self,
        field_name: str,
        *,
        floor: float,
        index: int,
    ) -> np.ndarray:
        return np.maximum(self.dataset.resolved_array(field_name)[index], floor)

    def _validate_rule_stress(self) -> None:
        requirements = (
            (
                "tick_size",
                self.rule_stress.tick_size_factor > 1.0
                or self.rule_stress.adverse_tick_rounding,
            ),
            ("lot_size", self.rule_stress.lot_size_factor > 1.0),
            (
                "minimum_notional",
                self.rule_stress.minimum_notional_factor > 1.0,
            ),
        )
        for field_name, required in requirements:
            source_rules = self.dataset.resolved_array(field_name)
            if required and np.any(source_rules <= 0.0):
                raise ValueError(
                    f"{field_name} must be positive for execution-rule sensitivity"
                )

    def _calculate_effective_rule_arrays(
        self, *, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not 0 <= index < self.dataset.n_bars:
            raise IndexError("execution-rule index is outside the dataset")
        tick = (
            self._base_rule_array(
                "tick_size",
                floor=self.cost.tick_size,
                index=index,
            )
            * self.rule_stress.tick_size_factor
        )
        lot = (
            self._base_rule_array(
                "lot_size",
                floor=self.cost.lot_size,
                index=index,
            )
            * self.rule_stress.lot_size_factor
        )
        minimum = (
            self._base_rule_array(
                "minimum_notional",
                floor=self.cost.minimum_notional,
                index=index,
            )
            * self.rule_stress.minimum_notional_factor
        )
        if self.market_order_profile is not None:
            for rule in self.market_order_profile.rules:
                symbol = rule.symbol_index
                lot[symbol] = joint_lot_size(
                    (
                        float(self.dataset.resolved_array("lot_size")[index, symbol]),
                        self.cost.lot_size,
                        rule.lot_size,
                    ),
                    stress_factor=self.rule_stress.lot_size_factor,
                )
                minimum[symbol] = max(
                    minimum[symbol],
                    rule.minimum_notional * self.rule_stress.minimum_notional_factor,
                )
        return tick, lot, minimum

    def _effective_rule_array_views(
        self, *, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return read-only rule arrays cached for the internal execution path."""

        if not 0 <= index < self.dataset.n_bars:
            raise IndexError("execution-rule index is outside the dataset")
        cache_inputs = (
            self.dataset,
            self.cost,
            self.rule_stress,
            self.market_order_profile,
        )
        previous_inputs = self._effective_rule_array_cache_inputs
        if previous_inputs is None or any(
            previous is not current
            for previous, current in zip(previous_inputs, cache_inputs, strict=True)
        ):
            self._effective_rule_array_cache.clear()
            self._effective_rule_array_cache_inputs = cache_inputs

        cached = self._effective_rule_array_cache.pop(index, None)
        if cached is not None:
            self._effective_rule_array_cache[index] = cached
            return cached

        computed = self._calculate_effective_rule_arrays(index=index)
        for array in computed:
            array.setflags(write=False)
        self._effective_rule_array_cache[index] = computed
        if len(self._effective_rule_array_cache) > 2:
            oldest_index = next(iter(self._effective_rule_array_cache))
            del self._effective_rule_array_cache[oldest_index]
        return computed

    def effective_rule_arrays(
        self, *, index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return fresh writable arrays for the requested execution-rule row."""

        tick, lot, minimum = self._effective_rule_array_views(index=index)
        return tick.copy(), lot.copy(), minimum.copy()

    def market_order_rule(self, symbol_index: int) -> MarketOrderRule | None:
        profile = self.market_order_profile
        return None if profile is None else profile.rule_for(symbol_index)

    def order_minimum_notional(
        self, intent: OrderIntent, ordinary_minimum: float
    ) -> float:
        profile = self.market_order_profile
        if (
            profile is not None
            and profile.reduce_only_exits
            and intent.reduce_only
            and profile.rule_for(intent.symbol_index) is not None
        ):
            return self.cost.minimum_notional * self.rule_stress.minimum_notional_factor
        return ordinary_minimum

    @property
    def reduce_only_symbols(self) -> tuple[int, ...] | None:
        profile = self.market_order_profile
        if profile is None:
            return None
        return (
            tuple(r.symbol_index for r in profile.rules)
            if profile.reduce_only_exits
            else ()
        )

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
        lot_ratios = np.full(shape, self.rule_stress.lot_size_factor)
        if self.market_order_profile is not None:
            dataset_lots = self.dataset.resolved_array("lot_size")
            for rule in self.market_order_profile.rules:
                for index in range(start, stop):
                    steps = (
                        float(dataset_lots[index, rule.symbol_index]),
                        self.cost.lot_size,
                        rule.lot_size,
                    )
                    nominal = joint_lot_size(steps)
                    stressed = joint_lot_size(
                        steps, stress_factor=self.rule_stress.lot_size_factor
                    )
                    lot_ratios[index - start, rule.symbol_index] = float(
                        exact_quantity(stressed) / exact_quantity(nominal)
                    )
        return {
            "adverse_tick_rounding": self.rule_stress.adverse_tick_rounding,
            "lot_size_ratio": self._percentiles(lot_ratios),
            "minimum_notional_ratio": self._percentiles(
                np.full(shape, self.rule_stress.minimum_notional_factor)
            ),
            "schema_version": "execution_rule_burden_v1",
            "tick_size_ratio": self._percentiles(
                np.full(shape, self.rule_stress.tick_size_factor)
            ),
        }

    @property
    def compatibility_order_book(self) -> OrderBookState:
        return self._compatibility_order_book

    def _reset_compatibility_chain(self) -> None:
        self._compatibility_order_book = OrderBookState.empty()
        self._compatibility_last_book = None

    def reset_random_state(self, seed: int | None = None) -> None:
        resolved = self.cost.random_seed if seed is None else seed
        if isinstance(resolved, bool) or not isinstance(resolved, int) or resolved < 0:
            raise ValueError("seed must be a non-negative integer")
        self._rng = np.random.default_rng(resolved)
        self._reset_compatibility_chain()

    def _slippage_rates(
        self, size: int, *, chargeable_mask: np.ndarray | None = None
    ) -> np.ndarray:
        if chargeable_mask is not None and chargeable_mask.shape != (size,):
            raise ValueError("slippage chargeable mask must match symbols")
        if self.cost.slippage_std == 0.0 or size == 0:
            return np.zeros(size, dtype=np.float64)
        # Retain the unscaled draw so underflow cannot manufacture a free fill.
        # Standard-normal draws preserve the existing normal RNG stream.
        normal = self._rng.standard_normal(size)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            rates = np.abs(normal * self.cost.slippage_std)
            if self.cost.tail_slippage_probability > 0.0:
                tails = self._rng.random(size) < self.cost.tail_slippage_probability
                recoverable = tails & (normal != 0.0) & (rates == 0.0)
                rates[tails] *= self.cost.tail_slippage_multiplier
                if np.any(recoverable):
                    rates[recoverable] = np.abs(normal[recoverable]) * (
                        self.cost.slippage_std * self.cost.tail_slippage_multiplier
                    )
        if chargeable_mask is not None and np.any(
            chargeable_mask & (normal != 0.0) & (rates == 0.0)
        ):
            raise ValueError("nonzero execution cost rate cannot be represented")
        return rates

    def _round_prices(
        self,
        prices: np.ndarray,
        *,
        index: int,
        directions: np.ndarray | None = None,
    ) -> np.ndarray:
        tick, _, _ = self._effective_rule_array_views(index=index)
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
        from trade_rl.simulation.quantities import quantize_quantity

        _, lot, _ = self._effective_rule_array_views(index=index)
        return np.array(
            [
                quantize_quantity(float(quantity), float(step))[0]
                for quantity, step in zip(quantities, lot, strict=True)
            ]
        )

    def _capacity_bounded_legacy_targets(
        self,
        current: np.ndarray,
        accepted_delta: np.ndarray,
        *,
        market_index: int,
    ) -> np.ndarray:
        """Choose lot-grid targets without increasing the accepted fill budget.

        A pre-existing off-grid holding is legal and must remain unchanged when
        there is insufficient capacity to reach the next lot-grid target.
        """

        _, lot_sizes, _ = self._effective_rule_array_views(index=market_index)
        next_quantities = current.copy()
        for index, (position, change, lot_size) in enumerate(
            zip(current, accepted_delta, lot_sizes, strict=True)
        ):
            if change == 0.0:
                continue
            position_exact = exact_quantity(float(position))
            change_exact = exact_quantity(float(change))
            limit = position_exact + change_exact
            if lot_size > 0.0:
                step = exact_quantity(float(lot_size))
                count = (
                    math.floor(limit / step)
                    if change > 0.0
                    else math.ceil(limit / step)
                )
                candidate = project_quantity(count * step)
            else:
                candidate = project_quantity(limit)
            realized_delta = exact_quantity(candidate) - position_exact
            if (change_exact > 0 and 0 < realized_delta <= change_exact) or (
                change_exact < 0 and change_exact <= realized_delta < 0
            ):
                next_quantities[index] = candidate
        return next_quantities

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
        value = book.portfolio_value
        if self.insolvency_valuation == "floor_zero":
            value = max(value, 0.0)
        book.quantities = np.zeros_like(book.quantities)
        book.mark_prices = prices.copy()
        book.cash = value
        book.margin_used = 0.0
        book.maintenance_margin = 0.0
        book.maintenance_requirement = 0.0
        if self.insolvency_valuation == "retain_debt":
            book.refresh_drawdown(unbounded_loss=True)

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
        # Preserve the original close demand even when minimum-notional
        # admission later refuses every fill. Keep the same rounded price
        # basis used to calculate actual fill notional.
        raw_requested_notional_vector = requested_notional_vector.copy()
        _, _, minimum_notional = self._effective_rule_array_views(index=market_index)
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
        next_quantities = self._capacity_bounded_legacy_targets(
            book.quantities, filled_delta, market_index=market_index
        )
        filled_notional_vector = self.dataset.quantity_notional(
            market_index, next_quantities - book.quantities, price_vector
        )
        # Check the *actual* post-grid notional; a rounded position must never
        # consume more volume than the accepted capacity even by float error.
        over_capacity = np.abs(filled_notional_vector) > capacity
        if np.any(over_capacity):
            next_quantities[over_capacity] = book.quantities[over_capacity]
            filled_notional_vector = self.dataset.quantity_notional(
                market_index, next_quantities - book.quantities, price_vector
            )

        positive_capacity = capacity_notional > 0.0
        participation = np.zeros_like(capacity_notional)
        # A nonzero fill / large capacity can round the ratio to float zero.
        # Handle the NumPy underflow mode without claiming zero economic impact.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            participation[positive_capacity] = (
                np.abs(filled_notional_vector[positive_capacity])
                / capacity_notional[positive_capacity]
            )
        if (
            self.cost.impact_rate != 0.0
            and self.cost.multiplier != 0.0
            and np.any(
                positive_capacity
                & (filled_notional_vector != 0.0)
                & (participation == 0.0)
            )
        ):
            raise ValueError("nonzero execution cost rate cannot be represented")
        # Forced close bypasses limit resting/price rules and therefore takes
        # liquidity even when the configured ordinary order type is LIMIT.
        takes_liquidity = force_market or self.cost.order_type != "limit"
        spread_multiplier = 1.0 if takes_liquidity else 0.5
        chargeable = (filled_notional_vector != 0.0) & (self.cost.multiplier != 0.0)
        # NumPy strict underflow must not silently turn a nonzero booked fee
        # into a zero-cost fill. Validate only symbols that actually fill.
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            impact_scale = np.sqrt(participation)
            impact = self.cost.impact_rate * impact_scale
            slippage = self._slippage_rates(
                len(price_vector), chargeable_mask=chargeable
            )
            venue_fee = (
                self.cost.taker_fee_rate
                + self.dataset.resolved_array("taker_fee_rate")[market_index]
                if takes_liquidity
                else self.cost.maker_fee_rate
                + self.dataset.resolved_array("maker_fee_rate")[market_index]
            )
            spread_base = (
                self.cost.spread_rate
                + self.dataset.resolved_array("spread_rate")[market_index]
            )
            spread_component = spread_multiplier * spread_base
            base_rate = (
                self.cost.fee_rate
                + self.dataset.resolved_array("fee_rate")[market_index]
                + venue_fee
                + spread_component
                + impact
                + slippage
            )
            unit_cost = self.cost.multiplier * base_rate
            cost_vector = np.where(
                chargeable, np.abs(filled_notional_vector) * unit_cost, 0.0
            )
            cost_amount = float(np.sum(cost_vector))
        if (
            not np.isfinite(spread_component[chargeable]).all()
            or not np.isfinite(impact[chargeable]).all()
            or not np.isfinite(base_rate[chargeable]).all()
            or not np.isfinite(unit_cost[chargeable]).all()
            or not np.isfinite(cost_vector).all()
            or not math.isfinite(cost_amount)
        ):
            raise ValueError("execution cost became non-finite")
        if np.any(chargeable & (spread_base != 0.0) & (spread_component == 0.0)):
            raise ValueError("nonzero execution cost rate cannot be represented")
        if np.any(
            chargeable
            & (self.cost.impact_rate != 0.0)
            & (impact_scale != 0.0)
            & (impact == 0.0)
        ):
            raise ValueError("nonzero execution cost rate cannot be represented")
        if np.any(chargeable & (base_rate != 0.0) & (unit_cost == 0.0)):
            raise ValueError("nonzero execution cost rate cannot be represented")
        if np.any(chargeable & (unit_cost != 0.0) & (cost_vector == 0.0)):
            raise ValueError("nonzero execution cost cannot be represented")

        fills_before = book.fill_count
        events_before = book.rebalance_events
        filled_notional = float(np.abs(filled_notional_vector).sum())
        book.execute(
            fill_prices=price_vector,
            target_quantities=next_quantities,
            cost_amount=cost_amount,
            turnover=filled_notional / turnover_denominator,
            preserve_unchanged_exact=True,
        )
        self._update_margin(book)
        return _FillResult(
            requested_notional=float(np.abs(requested_notional_vector).sum()),
            original_requested_notional=float(
                np.abs(raw_requested_notional_vector).sum()
            ),
            original_requested_notional_by_symbol=np.abs(raw_requested_notional_vector),
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

    def _charge_borrow(
        self,
        book: BookState,
        *,
        index: int,
        year_fraction: float,
    ) -> float:
        if (
            isinstance(year_fraction, bool)
            or not math.isfinite(year_fraction)
            or year_fraction < 0.0
        ):
            raise ValueError("borrow year_fraction must be finite and non-negative")
        if year_fraction == 0.0 or self.cost.borrow_rate_multiplier == 0.0:
            return 0.0
        short_values = np.maximum(-book.position_values, 0.0)
        borrow_rates = self.dataset.resolved_array("borrow_rate")[index]
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            contributions = short_values * borrow_rates
            rate_total = float(np.sum(contributions))
        if not np.isfinite(contributions).all() or not math.isfinite(rate_total):
            raise ValueError("borrow amount became non-finite")
        active = (short_values != 0.0) & (borrow_rates != 0.0)
        if np.any(active & (contributions == 0.0)):
            raise ValueError("nonzero borrow amount cannot be represented")
        year_scaled = rate_total * year_fraction
        if not math.isfinite(year_scaled):
            raise ValueError("borrow amount became non-finite")
        if rate_total != 0.0 and year_scaled == 0.0:
            raise ValueError("nonzero borrow amount cannot be represented")
        borrow_amount = year_scaled * self.cost.borrow_rate_multiplier
        if not math.isfinite(borrow_amount):
            raise ValueError("borrow amount became non-finite")
        if year_scaled != 0.0 and borrow_amount == 0.0:
            raise ValueError("nonzero borrow amount cannot be represented")
        if borrow_amount > 0.0:
            book.charge_borrow(borrow_amount)
        return borrow_amount

    def _charge_carry(
        self,
        book: BookState,
        *,
        index: int,
        year_fraction: float | None = None,
    ) -> tuple[float, float]:
        funding_due = self.dataset.resolved_array("funding_due")[index].astype(
            np.float64
        )
        if self.dataset.funding_price_rate is None:
            funding_notional = self.dataset.quantity_notional(index, book.quantities)
            funding_rate = self.dataset.funding_rate[index]
            active_source = (funding_notional != 0.0) & (funding_rate != 0.0)
        else:
            multipliers = self.dataset.resolved_array("contract_multipliers")
            funding_rate = self.dataset.funding_price_rate[index]
            active_source = (
                (book.quantities != 0.0)
                & (multipliers != 0.0)
                & (funding_rate != 0.0)
                & (funding_due != 0.0)
            )
            # Zero effect must not evaluate an unrelated inf * 0 notional.
            funding_notional = np.zeros(book.quantities.shape, dtype=np.float64)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                funding_notional[active_source] = (
                    book.quantities[active_source] * multipliers[active_source]
                )
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            due_rates = funding_rate * funding_due
            contributions = funding_notional * due_rates
            # Preserve the original dot aggregation and cancellation semantics.
            funding_amount = -float(np.dot(funding_notional, due_rates))
        active_source &= funding_due != 0.0
        if not np.isfinite(contributions).all() or not math.isfinite(funding_amount):
            raise ValueError("funding amount became non-finite")
        if np.any(active_source & (contributions == 0.0)):
            raise ValueError("nonzero funding amount cannot be represented")
        previous_index = max(0, index - 1)
        resolved_year_fraction = (
            self.dataset.elapsed_year_fraction(previous_index, index)
            if year_fraction is None
            else year_fraction
        )
        borrow_amount = self._charge_borrow(
            book,
            index=index,
            year_fraction=resolved_year_fraction,
        )
        return funding_amount, borrow_amount

    @property
    def execution_policy_digest(self) -> str:
        if self.native_calendar_month_linear_carry or self.native_session_linear_carry:
            self._validate_native_fixed_configuration()
        trigger_fraction_snapshot = tuple(
            f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"
            for value in self.cost.trigger_volume_fractions
        )
        funding_semantics = (
            "dataset_funding_price_rate_v1"
            if self.dataset.funding_price_rate is not None
            else None
        )
        cache_inputs = (
            self.cost,
            self.rule_stress,
            self.market_order_profile,
            trigger_fraction_snapshot,
            self.insolvency_valuation,
            funding_semantics,
            self.strict_spot_cash,
            self.native_fixed_linear_carry,
            self.native_calendar_month_linear_carry,
            self.native_session_linear_carry,
        )
        previous_inputs = self._execution_policy_digest_cache_inputs
        cached_digest = self._execution_policy_digest_cache
        if (
            cached_digest is not None
            and previous_inputs is not None
            and previous_inputs[0] is self.cost
            and previous_inputs[1] is self.rule_stress
            and previous_inputs[2] is self.market_order_profile
            and previous_inputs[3] == trigger_fraction_snapshot
            and previous_inputs[4] == self.insolvency_valuation
            and previous_inputs[5] == funding_semantics
            and previous_inputs[6] == self.strict_spot_cash
            and previous_inputs[7] == self.native_fixed_linear_carry
            and previous_inputs[8] == self.native_calendar_month_linear_carry
            and previous_inputs[9] == self.native_session_linear_carry
        ):
            return cached_digest

        digest = _runtime_execution_policy_digest(
            self.cost,
            rule_stress=self.rule_stress,
            market_order_profile_digest=(
                self.market_order_profile.digest
                if self.market_order_profile is not None
                else None
            ),
            insolvency_valuation=self.insolvency_valuation,
            funding_price_rate_present=funding_semantics is not None,
            strict_spot_cash=self.strict_spot_cash,
            native_fixed_linear_carry=self.native_fixed_linear_carry,
            native_calendar_month_linear_carry=self.native_calendar_month_linear_carry,
            native_session_linear_carry=self.native_session_linear_carry,
        )

        self._execution_policy_digest_cache_inputs = cache_inputs
        self._execution_policy_digest_cache = digest
        return digest

    def execute_orders(
        self,
        book: BookState,
        order_book: OrderBookState,
        intents: Sequence[OrderIntent],
        *,
        start_index: int,
        bars: int,
    ) -> StatefulExecutionResult:
        return execute_stateful_orders(
            self,
            book,
            order_book,
            intents,
            start_index=start_index,
            bars=bars,
        )

    def _execute_interval_with_stateful_evidence(
        self,
        book: BookState,
        target: np.ndarray,
        *,
        start_index: int,
        bars: int,
        hold_current_quantity_symbols: tuple[int, ...] = (),
        continue_pending_orders: bool = False,
    ) -> tuple[
        ExecutionResult,
        StatefulExecutionResult,
        TerminalOrderArchive,
    ]:
        state = (
            self._compatibility_order_book
            if book is self._compatibility_last_book
            else OrderBookState.empty()
        )
        previous_terminal_orders = state.terminal_orders
        if not isinstance(previous_terminal_orders, TerminalOrderArchive):
            raise RuntimeError("terminal order archive was not normalized")
        if continue_pending_orders:
            # Only the risk-checked native no-new-intent transition may use
            # this path. Keep the exact active order identities and quantities.
            if book is not self._compatibility_last_book or not state.active_orders:
                raise ValueError(
                    "pending continuation requires the current executor book"
                )
            if hold_current_quantity_symbols:
                raise ValueError("pending continuation cannot also request exact HOLD")
            stateful = execute_stateful_orders(
                self, book, state, (), start_index=start_index, bars=bars
            )
        else:
            target_vector = np.asarray(
                checked_array_input(target, field_name="target weights"),
                dtype=np.float64,
            ).reshape(-1)
            target_identity = content_digest(
                {
                    "bars": bars,
                    "dataset_id": self.dataset.dataset_id,
                    "schema_version": "compatibility_target_execution_v1",
                    "start_index": start_index,
                    "target": tuple(float(value) for value in target_vector),
                }
            )
            if hold_current_quantity_symbols:
                stateful = execute_target_statefully(
                    self,
                    book,
                    state,
                    target_vector,
                    start_index=start_index,
                    bars=bars,
                    target_identity=target_identity,
                    hold_current_quantity_symbols=hold_current_quantity_symbols,
                )
            else:
                stateful = execute_target_statefully(
                    self,
                    book,
                    state,
                    target_vector,
                    start_index=start_index,
                    bars=bars,
                    target_identity=target_identity,
                )
        self._compatibility_order_book = stateful.order_book
        self._compatibility_last_book = stateful.book

        attempted = any(
            event.event_type in _ATTEMPT_EVENT_TYPES for event in stateful.order_events
        )
        requested_turnover = stateful.requested_turnover if attempted else 0.0
        unfilled_turnover = stateful.unfilled_turnover if attempted else 0.0
        requested_by_symbol = (
            stateful.requested_notional_by_symbol
            if attempted
            else np.zeros_like(stateful.requested_notional_by_symbol)
        )
        fill_ratio = stateful.fill_ratio if attempted else 1.0
        compatibility = ExecutionResult(
            book=stateful.book,
            next_index=stateful.next_index,
            bars_advanced=stateful.bars_advanced,
            interval_gross_return=stateful.interval_gross_return,
            interval_cost=stateful.interval_cost,
            interval_funding=stateful.interval_funding,
            interval_net_return=stateful.interval_net_return,
            interval_log_return=stateful.interval_log_return,
            requested_turnover=requested_turnover,
            filled_turnover=stateful.filled_turnover,
            unfilled_turnover=unfilled_turnover,
            fill_count=stateful.fill_count,
            rebalance_events=stateful.rebalance_events,
            fill_ratio=fill_ratio,
            max_participation=stateful.max_participation,
            interval_borrow_cost=stateful.interval_borrow_cost,
            interval_dividend=stateful.interval_dividend,
            interval_cash_interest=stateful.interval_cash_interest,
            margin_utilization=stateful.book.margin_utilization,
            termination_reason=stateful.termination_reason,
            requested_notional_by_symbol=requested_by_symbol,
            filled_notional_by_symbol=stateful.filled_notional_by_symbol,
            participation_by_symbol=stateful.participation_by_symbol,
            cost_by_symbol=stateful.cost_by_symbol,
        )
        return compatibility, stateful, previous_terminal_orders

    def execute_interval(
        self,
        book: BookState,
        target: np.ndarray,
        *,
        start_index: int,
        bars: int,
        hold_current_quantity_symbols: tuple[int, ...] = (),
    ) -> ExecutionResult:
        return self._observe_compatibility_execution(
            *self._execute_interval_with_stateful_evidence(
                book,
                target,
                start_index=start_index,
                bars=bars,
                hold_current_quantity_symbols=hold_current_quantity_symbols,
            )
        )

    def _continue_pending_interval(
        self, book: BookState, *, start_index: int
    ) -> ExecutionResult:
        return self._observe_compatibility_execution(
            *self._execute_interval_with_stateful_evidence(
                book,
                book.weights,
                start_index=start_index,
                bars=1,
                continue_pending_orders=True,
            )
        )

    def _observe_compatibility_execution(
        self,
        compatibility: ExecutionResult,
        stateful: StatefulExecutionResult,
        previous_terminal_orders: TerminalOrderArchive,
    ) -> ExecutionResult:
        if self._execution_observer is not None:
            self._execution_observer(StatefulExecutionObservation.from_result(stateful))
        if self._execution_trace_observer is not None:
            self._execution_trace_observer(
                StatefulExecutionObservation.from_result_since(
                    stateful, previous_terminal_orders
                )
            )
        return compatibility

    def liquidate_at_close(self, book: BookState, *, index: int) -> ExecutionResult:
        if self.dataset.native_clock is not None:
            raise ValueError(
                "native fixed execution requires next-open stateful settlement"
            )
        if self.strict_spot_cash:
            raise ValueError("strict spot cash requires stateful settlement orders")
        if self.market_order_profile is not None:
            raise ValueError(
                "market profile liquidation requires explicit stateful orders"
            )
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
        ending_value = (
            result_book.portfolio_value
            if self.insolvency_valuation == "retain_debt"
            else max(result_book.portfolio_value, 0.0)
        )
        interval_net_return = ending_value / starting_value - 1.0
        if self.insolvency_valuation != "retain_debt":
            interval_net_return = max(interval_net_return, -1.0 + 1e-12)
        interval_log_return = math.log1p(max(interval_net_return, -1.0 + 1e-12))
        requested_notional = fill.original_requested_notional
        requested_turnover = requested_notional / starting_value
        filled_turnover = fill.filled_notional / starting_value
        if result_book.termination_reason is None and all(
            quantity == 0 for quantity in result_book.exact_quantities
        ):
            result_book.termination_reason = EconomicTerminationReason.LIQUIDATION
        return ExecutionResult(
            book=result_book,
            next_index=index,
            bars_advanced=0,
            interval_gross_return=0.0,
            interval_cost=fill.cost_amount,
            interval_funding=0.0,
            interval_net_return=interval_net_return,
            interval_log_return=interval_log_return,
            requested_turnover=requested_turnover,
            filled_turnover=filled_turnover,
            unfilled_turnover=max(0.0, requested_turnover - filled_turnover),
            fill_count=fill.fill_count,
            rebalance_events=fill.rebalance_events,
            fill_ratio=(
                1.0
                if requested_notional == 0.0
                else min(1.0, fill.filled_notional / requested_notional)
            ),
            max_participation=fill.max_participation,
            margin_utilization=result_book.margin_utilization,
            termination_reason=(
                None
                if result_book.termination_reason is None
                else EconomicTerminationReason(result_book.termination_reason).value
            ),
            requested_notional_by_symbol=fill.original_requested_notional_by_symbol,
            filled_notional_by_symbol=fill.filled_notional_by_symbol,
            participation_by_symbol=fill.participation_by_symbol,
            cost_by_symbol=fill.cost_by_symbol,
        )
