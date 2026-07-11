"""Pre-trade portfolio and order-risk validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np

OrderSide = Literal["buy", "sell"]


class PreTradeRejection(Exception):
    def __init__(self, reason: str, details: dict):
        super().__init__(f"PreTradeRejection: {reason} - {details}")
        self.reason = reason
        self.details = details


@dataclass(frozen=True)
class PendingOrder:
    symbol: str
    side: OrderSide
    notional: float
    reduce_only: bool = False


@dataclass
class PreTradeRiskConfig:
    max_leverage: float | None = None
    max_single_weight: float | None = None
    max_position_pct: float | None = None
    max_notional: float | None = None
    forbidden_symbols: set[str] | None = None
    max_net_exposure: float | None = None
    max_worst_case_notional: float | None = None
    symbol_liquidity_caps: dict[str, float] | None = None
    min_order_notional: float | None = None


class PreTradeRiskVerifier:
    def __init__(self, config: PreTradeRiskConfig):
        self.config = config

    def validate(
        self,
        target_weights: np.ndarray,
        portfolio_value: float,
        symbols: Iterable[str] | None = None,
        *,
        current_weights: np.ndarray | None = None,
        open_orders: Sequence[PendingOrder] = (),
    ) -> None:
        target = np.asarray(target_weights, dtype=np.float64)
        if target.ndim != 1:
            raise ValueError("target_weights must be one-dimensional")
        if not np.isfinite(target).all():
            raise PreTradeRejection(
                reason="nan_or_inf_in_weights",
                details={
                    "has_nan": bool(np.isnan(target).any()),
                    "has_inf": bool(np.isinf(target).any()),
                },
            )
        if not np.isfinite(portfolio_value) or portfolio_value <= 0:
            raise ValueError("portfolio_value must be finite and positive")

        # Legacy callers that only provide target weights retain target-level
        # validation without inventing a trade from zero. Delta/order checks are
        # activated only when the caller supplies the actual current portfolio.
        current = (
            target.copy()
            if current_weights is None
            else np.asarray(current_weights, dtype=np.float64)
        )
        if current.shape != target.shape:
            raise ValueError("current_weights must match target_weights shape")
        if not np.isfinite(current).all():
            raise PreTradeRejection(
                reason="nan_or_inf_in_current_weights",
                details={
                    "has_nan": bool(np.isnan(current).any()),
                    "has_inf": bool(np.isinf(current).any()),
                },
            )

        symbols_list = list(symbols) if symbols is not None else None
        if symbols_list is not None and len(symbols_list) != len(target):
            raise ValueError("symbols length must match target_weights length")
        if open_orders and symbols_list is None:
            raise ValueError("symbols are required when open_orders are supplied")

        self._validate_open_orders(open_orders, symbols_list)

        gross_leverage = float(np.abs(target).sum())
        net_exposure = float(abs(target.sum()))
        total_notional = portfolio_value * gross_leverage
        delta = target - current
        delta_notionals = np.abs(delta) * portfolio_value

        if self.config.forbidden_symbols and symbols_list is not None:
            self._validate_forbidden_symbols(
                target, current, symbols_list, self.config.forbidden_symbols
            )

        if self.config.max_leverage is not None and gross_leverage > self.config.max_leverage:
            raise PreTradeRejection(
                reason="leverage_limit_exceeded",
                details={"gross_leverage": gross_leverage, "max_leverage": self.config.max_leverage},
            )
        if self.config.max_net_exposure is not None and net_exposure > self.config.max_net_exposure:
            raise PreTradeRejection(
                reason="net_exposure_limit_exceeded",
                details={"net_exposure": net_exposure, "max_net_exposure": self.config.max_net_exposure},
            )

        max_single = float(np.abs(target).max()) if len(target) else 0.0
        if self.config.max_single_weight is not None and max_single > self.config.max_single_weight:
            raise PreTradeRejection(
                reason="single_weight_limit_exceeded",
                details={"max_single_weight_found": max_single, "max_single_weight_allowed": self.config.max_single_weight},
            )
        if self.config.max_position_pct is not None and max_single > self.config.max_position_pct:
            raise PreTradeRejection(
                reason="position_pct_limit_exceeded",
                details={"max_position_pct_found": max_single, "max_position_pct_allowed": self.config.max_position_pct},
            )
        if self.config.max_notional is not None and total_notional > self.config.max_notional:
            raise PreTradeRejection(
                reason="notional_limit_exceeded",
                details={"total_notional": total_notional, "max_notional": self.config.max_notional},
            )

        if self.config.max_worst_case_notional is not None:
            worst_case, per_symbol = self._worst_case_notional(
                target=target,
                current=current,
                portfolio_value=portfolio_value,
                symbols=symbols_list,
                open_orders=open_orders,
            )
            if worst_case > self.config.max_worst_case_notional:
                raise PreTradeRejection(
                    reason="worst_case_notional_exceeded",
                    details={
                        "worst_case_notional": worst_case,
                        "max_worst_case_notional": self.config.max_worst_case_notional,
                        "per_symbol_worst_case": per_symbol,
                    },
                )

        pending_by_symbol = self._pending_execution_notional(open_orders)
        if self.config.symbol_liquidity_caps and symbols_list is not None:
            for index, symbol in enumerate(symbols_list):
                cap = self.config.symbol_liquidity_caps.get(symbol)
                if cap is None:
                    continue
                execution_notional = float(delta_notionals[index]) + pending_by_symbol.get(symbol, 0.0)
                if execution_notional > cap:
                    raise PreTradeRejection(
                        reason="symbol_liquidity_cap_exceeded",
                        details={"symbol": symbol, "execution_notional": execution_notional, "liquidity_cap": cap},
                    )

        if self.config.min_order_notional is not None:
            for index, order_notional in enumerate(delta_notionals):
                if 0 < order_notional < self.config.min_order_notional:
                    symbol = symbols_list[index] if symbols_list is not None else f"index_{index}"
                    raise PreTradeRejection(
                        reason="min_order_notional_not_met",
                        details={"symbol": symbol, "order_notional": float(order_notional), "min_order_notional": self.config.min_order_notional},
                    )

    @staticmethod
    def _validate_open_orders(open_orders: Sequence[PendingOrder], symbols: list[str] | None) -> None:
        allowed = set(symbols) if symbols is not None else None
        for order in open_orders:
            if order.side not in ("buy", "sell"):
                raise ValueError(f"invalid pending order side: {order.side}")
            if not np.isfinite(order.notional) or order.notional <= 0:
                raise ValueError("pending order notional must be finite and positive")
            if allowed is not None and order.symbol not in allowed:
                raise ValueError(f"pending order symbol not in symbols: {order.symbol}")

    @staticmethod
    def _validate_forbidden_symbols(target: np.ndarray, current: np.ndarray, symbols: list[str], forbidden: set[str]) -> None:
        for symbol, target_weight, current_weight in zip(symbols, target, current):
            if symbol in forbidden and _increases_exposure(float(current_weight), float(target_weight)):
                raise PreTradeRejection(
                    reason="forbidden_symbol",
                    details={"symbol": symbol, "current_weight": float(current_weight), "target_weight": float(target_weight)},
                )

    @staticmethod
    def _pending_execution_notional(open_orders: Sequence[PendingOrder]) -> dict[str, float]:
        totals: dict[str, float] = {}
        for order in open_orders:
            totals[order.symbol] = totals.get(order.symbol, 0.0) + float(order.notional)
        return totals

    @staticmethod
    def _worst_case_notional(
        *, target: np.ndarray, current: np.ndarray, portfolio_value: float,
        symbols: list[str] | None, open_orders: Sequence[PendingOrder],
    ) -> tuple[float, dict[str, float]]:
        names = symbols or [f"index_{index}" for index in range(len(target))]
        buy_pending = {name: 0.0 for name in names}
        sell_pending = {name: 0.0 for name in names}
        for order in open_orders:
            if order.reduce_only:
                continue
            if order.symbol not in buy_pending:
                raise ValueError(f"pending order symbol not in symbols: {order.symbol}")
            if order.side == "buy":
                buy_pending[order.symbol] += float(order.notional)
            else:
                sell_pending[order.symbol] += float(order.notional)

        per_symbol: dict[str, float] = {}
        for index, symbol in enumerate(names):
            current_notional = float(current[index] * portfolio_value)
            proposed_delta = float((target[index] - current[index]) * portfolio_value)
            buy_scenario = current_notional + buy_pending[symbol] + max(proposed_delta, 0.0)
            sell_scenario = current_notional - sell_pending[symbol] + min(proposed_delta, 0.0)
            per_symbol[symbol] = max(abs(current_notional), abs(buy_scenario), abs(sell_scenario))
        return float(sum(per_symbol.values())), per_symbol


def _increases_exposure(current: float, target: float, epsilon: float = 1e-12) -> bool:
    if abs(target) <= epsilon:
        return False
    if abs(current) <= epsilon:
        return True
    if current * target < 0:
        return True
    return abs(target) > abs(current) + epsilon
