"""Data-only contracts for one simultaneous portfolio strategy decision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Protocol


def _require_finite_real(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real numeric value")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class PortfolioRiskLimits:
    """Detached scalar view of the hard limits used for one replay decision."""

    max_gross: float
    max_abs_weight: float
    max_turnover: float | None
    drawdown_start: float
    drawdown_stop: float
    emergency_turnover_override: bool
    fail_closed_tolerance: float

    def __post_init__(self) -> None:
        for name in (
            "max_gross",
            "max_abs_weight",
            "drawdown_start",
            "drawdown_stop",
            "fail_closed_tolerance",
        ):
            _require_finite_real(getattr(self, name), name=name)
        if self.max_turnover is not None:
            _require_finite_real(self.max_turnover, name="max_turnover")
        if type(self.emergency_turnover_override) is not bool:
            raise TypeError("emergency_turnover_override must be a boolean")


@dataclass(frozen=True, slots=True)
class PortfolioAccountSnapshot:
    """Detached account facts; cash is not a free-cash or reservation estimate."""

    cash: float
    portfolio_value: float
    peak_value: float
    max_drawdown: float
    exact_quantities: tuple[str, ...]
    quantities: tuple[float, ...]
    weights: tuple[float, ...]
    mark_prices: tuple[float, ...]
    contract_multipliers: tuple[float, ...]
    turnover_total: float
    total_cost: float
    funding_pnl: float
    borrow_cost: float
    margin_used: float
    maintenance_margin: float
    maintenance_requirement: float
    margin_deficit: float
    insolvent: bool
    termination_reason: str | None

    def __post_init__(self) -> None:
        for name in (
            "cash",
            "portfolio_value",
            "peak_value",
            "max_drawdown",
            "turnover_total",
            "total_cost",
            "funding_pnl",
            "borrow_cost",
            "margin_used",
            "maintenance_margin",
            "maintenance_requirement",
            "margin_deficit",
        ):
            _require_finite_real(getattr(self, name), name=name)
        if type(self.insolvent) is not bool:
            raise TypeError("insolvent must be a boolean")
        if self.termination_reason is not None and not isinstance(
            self.termination_reason, str
        ):
            raise TypeError("termination_reason must be a string or null")
        n_symbols = len(self.exact_quantities)
        if type(self.exact_quantities) is not tuple or any(
            not isinstance(value, str) for value in self.exact_quantities
        ):
            raise TypeError("exact_quantities must be a tuple of strings")
        if n_symbols == 0 or any(
            type(value) is not tuple or len(value) != n_symbols
            for value in (
                self.quantities,
                self.weights,
                self.mark_prices,
                self.contract_multipliers,
            )
        ):
            raise ValueError("account vectors must share a non-empty symbol shape")
        for name in (
            "quantities",
            "weights",
            "mark_prices",
            "contract_multipliers",
        ):
            for value in getattr(self, name):
                _require_finite_real(value, name=name)


@dataclass(frozen=True, slots=True)
class PortfolioPendingOrderSnapshot:
    """Primitive view of one currently active order and its remaining quantity."""

    symbol_index: int
    order_type: str
    time_in_force: str
    requested_quantity: float
    remaining_quantity: float
    status: str
    submit_index: int
    eligible_index: int
    last_processed_index: int | None
    trigger_index: int | None
    expiry_index: int | None
    reduce_only: bool
    limit_price: float | None = None
    stop_price: float | None = None
    submission_reference_price: float | None = None

    def __post_init__(self) -> None:
        if type(self.symbol_index) is not int or self.symbol_index < 0:
            raise ValueError("symbol_index must be a non-negative integer")
        if type(self.submit_index) is not int or self.submit_index < 0:
            raise ValueError("submit_index must be a non-negative integer")
        if type(self.eligible_index) is not int or self.eligible_index < 0:
            raise ValueError("eligible_index must be a non-negative integer")
        for name in ("last_processed_index", "trigger_index", "expiry_index"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or null")
        _require_finite_real(self.requested_quantity, name="requested_quantity")
        _require_finite_real(self.remaining_quantity, name="remaining_quantity")
        if not all(
            isinstance(value, str) and value
            for value in (self.order_type, self.time_in_force, self.status)
        ):
            raise ValueError("order type, time in force and status must be non-empty")
        if type(self.reduce_only) is not bool:
            raise TypeError("reduce_only must be a boolean")
        prices = (
            ("limit_price", self.limit_price),
            ("stop_price", self.stop_price),
            ("submission_reference_price", self.submission_reference_price),
        )
        for name, value in prices:
            if value is not None:
                _require_finite_real(value, name=name)
                if value <= 0.0:
                    raise ValueError(f"{name} must be positive")
        if all(value is None for _, value in prices):
            return
        if self.submission_reference_price is None:
            raise ValueError(
                "submission_reference_price is required for complete price context"
            )
        if self.order_type == "market":
            if self.limit_price is not None or self.stop_price is not None:
                raise ValueError("market orders may not define limit or stop prices")
        elif self.order_type == "limit":
            if self.limit_price is None or self.stop_price is not None:
                raise ValueError("limit orders require only limit_price")
        elif self.order_type == "stop_market":
            if self.stop_price is None or self.limit_price is not None:
                raise ValueError("stop-market orders require only stop_price")
        else:
            raise ValueError("price context requires a supported order type")


@dataclass(frozen=True, slots=True)
class PortfolioDecisionSnapshot:
    """Immutable current-row information for one whole-portfolio decision."""

    index: int
    timestamp_ns: int
    agent_stop_index: int
    remaining_decisions: int
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    features: tuple[tuple[float, ...], ...]
    feature_available: tuple[tuple[bool, ...], ...]
    feature_staleness: tuple[tuple[float, ...], ...]
    global_features: tuple[float, ...]
    global_feature_available: tuple[bool, ...]
    account: PortfolioAccountSnapshot
    active_orders: tuple[PortfolioPendingOrderSnapshot, ...]
    risk: PortfolioRiskLimits

    def __post_init__(self) -> None:
        for name in (
            "index",
            "timestamp_ns",
            "agent_stop_index",
            "remaining_decisions",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be a native integer")
        if self.index < 0 or self.agent_stop_index <= self.index:
            raise ValueError("snapshot index must precede the agent stop index")
        if self.remaining_decisions != self.agent_stop_index - self.index:
            raise ValueError("remaining_decisions must match the agent stop index")
        if type(self.symbols) is not tuple:
            raise TypeError("symbols must be a tuple")
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("symbols must be a non-empty unique tuple")
        if any(not isinstance(value, str) or not value for value in self.symbols):
            raise TypeError("symbols must contain non-empty strings")
        for name in ("feature_names", "global_feature_names"):
            values = getattr(self, name)
            if type(values) is not tuple or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise TypeError(f"{name} must be a tuple of non-empty strings")
        n_symbols = len(self.symbols)
        n_features = len(self.feature_names)
        if any(
            type(value) is not tuple or len(value) != n_symbols
            for value in (self.features, self.feature_available, self.feature_staleness)
        ):
            raise ValueError("per-symbol feature rows must match the symbol roster")
        for name in ("features", "feature_available", "feature_staleness"):
            rows = getattr(self, name)
            if any(type(row) is not tuple or len(row) != n_features for row in rows):
                raise ValueError(f"{name} rows must match feature_names")
        if any(
            type(value) is not bool for row in self.feature_available for value in row
        ):
            raise TypeError("feature_available must contain booleans")
        if (
            type(self.global_features) is not tuple
            or type(self.global_feature_available) is not tuple
        ):
            raise TypeError("global feature values and mask must be tuples")
        if len(self.global_features) != len(self.global_feature_names) or len(
            self.global_feature_available
        ) != len(self.global_feature_names):
            raise ValueError("global feature values and mask must match their names")
        if any(type(value) is not bool for value in self.global_feature_available):
            raise TypeError("global_feature_available must contain booleans")
        for name in (
            "global_features",
            "features",
            "feature_staleness",
        ):
            values = getattr(self, name)
            if name in ("features", "feature_staleness"):
                values = tuple(item for row in values for item in row)
            for value in values:
                _require_finite_real(value, name=name)
        if not isinstance(self.account, PortfolioAccountSnapshot):
            raise TypeError("account must be a PortfolioAccountSnapshot")
        if not isinstance(self.risk, PortfolioRiskLimits):
            raise TypeError("risk must be PortfolioRiskLimits")
        if type(self.active_orders) is not tuple or any(
            not isinstance(order, PortfolioPendingOrderSnapshot)
            for order in self.active_orders
        ):
            raise TypeError("active_orders must be a tuple of pending-order snapshots")
        if len(self.account.exact_quantities) != n_symbols:
            raise ValueError("account vectors must match the symbol roster")
        if any(order.symbol_index >= n_symbols for order in self.active_orders):
            raise ValueError("active order symbol indexes must be in the symbol roster")


@dataclass(frozen=True, slots=True)
class PortfolioTargetProposal:
    """Complete numeric target vector plus explicit exact-quantity HOLD symbols."""

    target_weights: tuple[float, ...]
    hold_current_quantity_symbols: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.target_weights) is not tuple or not self.target_weights:
            raise TypeError("target_weights must be a non-empty tuple")
        weights: list[float] = []
        for value in self.target_weights:
            _require_finite_real(value, name="target_weights")
            weights.append(float(value))
        if type(self.hold_current_quantity_symbols) is not tuple:
            raise TypeError("hold_current_quantity_symbols must be a tuple")
        indexes = self.hold_current_quantity_symbols
        if any(type(index) is not int for index in indexes):
            raise TypeError("hold symbol indexes must be native integers")
        if len(set(indexes)) != len(indexes) or any(index < 0 for index in indexes):
            raise ValueError("hold symbol indexes must be unique and non-negative")
        object.__setattr__(self, "target_weights", tuple(weights))


class PortfolioPolicy(Protocol):
    """Policy called once with one simultaneous portfolio snapshot per bar."""

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal: ...


__all__ = [
    "PortfolioAccountSnapshot",
    "PortfolioDecisionSnapshot",
    "PortfolioPendingOrderSnapshot",
    "PortfolioPolicy",
    "PortfolioRiskLimits",
    "PortfolioTargetProposal",
]
