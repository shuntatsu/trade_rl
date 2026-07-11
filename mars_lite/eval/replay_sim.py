"""Replay execution simulator for one-minute aggregate trades."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal

import numpy as np
import pandas as pd

from mars_lite.utils.metrics import calc_sharpe_ratio

OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]


@dataclass(frozen=True)
class ExecutionOrder:
    timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = "market"
    limit_price: float | None = None
    max_delay_seconds: float = 3600.0
    latency_seconds: float = 0.0
    maker: bool = False


@dataclass(frozen=True)
class ReplayFill:
    order_timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    requested_quantity: float
    filled_quantity: float
    average_price: float
    fee_paid: float


@dataclass(frozen=True)
class ReplayResult:
    fills: list[ReplayFill]
    final_cash: float
    final_position: Dict[str, float]
    final_equity: float
    returns: list[float]
    sharpe: float


class LiquidityLedger:
    """Track unconsumed volume per trade row to prevent duplicate fills."""

    def __init__(self, trades: pd.DataFrame) -> None:
        self.remaining_volume: np.ndarray = trades["quantity"].to_numpy(dtype=float).copy()

    def consume(self, row_idx: int, amount: float) -> float:
        avail = self.remaining_volume[row_idx]
        take = min(avail, amount)
        self.remaining_volume[row_idx] -= take
        return take


class ReplaySimulator:
    """Deterministic simulator that fills orders against aggTrades rows with a LiquidityLedger."""

    def __init__(
        self,
        fee_rate: float = 0.0005,
        max_participation_rate: float = 0.1,
        slippage_bps: float = 0.0,
        maker_fee_rate: float | None = None,
    ) -> None:
        if max_participation_rate <= 0:
            raise ValueError("max_participation_rate must be positive")
        if fee_rate < 0:
            raise ValueError("fee_rate must be non-negative")
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        self.fee_rate = fee_rate
        self.max_participation_rate = max_participation_rate
        self.slippage_bps = slippage_bps
        self.maker_fee_rate = maker_fee_rate if maker_fee_rate is not None else fee_rate

    def simulate(
        self,
        agg_trades: pd.DataFrame,
        orders: Iterable[ExecutionOrder],
        initial_cash: float = 1_000_000.0,
    ) -> ReplayResult:
        self._validate_trades(agg_trades)
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        trades = agg_trades.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
        ledger = LiquidityLedger(trades)

        cash = float(initial_cash)
        positions: Dict[str, float] = {}
        fills: list[ReplayFill] = []
        equity_curve = [float(initial_cash)]

        for order in sorted(orders, key=lambda item: item.timestamp):
            fill = self._fill_order(trades, ledger, order)
            if fill.filled_quantity == 0:
                fills.append(fill)
                continue

            signed_qty = (
                fill.filled_quantity if fill.side == "buy" else -fill.filled_quantity
            )
            notional = fill.filled_quantity * fill.average_price
            cash -= signed_qty * fill.average_price
            cash -= fill.fee_paid
            positions[fill.symbol] = positions.get(fill.symbol, 0.0) + signed_qty
            fills.append(fill)
            equity_curve.append(
                self._mark_to_market(cash, positions, trades, order.timestamp)
            )

        last_ts = trades["timestamp"].max() if not trades.empty else None
        final_equity = self._mark_to_market(cash, positions, trades, last_ts)
        equity_curve.append(final_equity)
        returns = np.diff(np.asarray(equity_curve, dtype=float)) / np.asarray(
            equity_curve[:-1], dtype=float
        )
        sharpe = calc_sharpe_ratio(returns, annualization_factor=365 * 24 * 60)
        return ReplayResult(
            fills=fills,
            final_cash=cash,
            final_position=positions,
            final_equity=final_equity,
            returns=[float(value) for value in returns],
            sharpe=sharpe,
        )

    def _fill_order(
        self, trades: pd.DataFrame, ledger: LiquidityLedger, order: ExecutionOrder
    ) -> ReplayFill:
        remaining = float(order.quantity)
        if remaining <= 0:
            raise ValueError("order quantity must be positive")

        effective_start = order.timestamp + pd.Timedelta(seconds=order.latency_seconds)
        effective_end = order.timestamp + pd.Timedelta(seconds=order.max_delay_seconds)

        filled = 0.0
        notional = 0.0

        # 行をイテレーションして有効範囲内＆指値条件合致の行に対して台帳消費を行う
        for idx in range(len(trades)):
            row_symbol = trades.at[idx, "symbol"]
            if row_symbol != order.symbol:
                continue
            row_ts = trades.at[idx, "timestamp"]
            if row_ts < effective_start:
                continue
            if row_ts > effective_end:
                break

            row_price = float(trades.at[idx, "price"])
            if order.order_type == "limit" and order.limit_price is not None:
                if order.side == "buy" and row_price > order.limit_price:
                    continue
                if order.side == "sell" and row_price < order.limit_price:
                    continue

            # 台帳から利用可能な出来高を取得
            available_in_row = ledger.remaining_volume[idx] * self.max_participation_rate
            take_target = min(remaining, available_in_row)
            if take_target <= 0:
                continue

            take = ledger.consume(idx, take_target / self.max_participation_rate) * self.max_participation_rate
            if take <= 0:
                continue

            exec_price = self._execution_price(row_price, order.side)
            filled += take
            notional += take * exec_price
            remaining -= take
            if remaining <= 1e-12:
                break

        average_price = notional / filled if filled > 0 else 0.0
        fee_rate = self.maker_fee_rate if order.maker else self.fee_rate
        return ReplayFill(
            order_timestamp=pd.Timestamp(order.timestamp),
            symbol=order.symbol,
            side=order.side,
            requested_quantity=float(order.quantity),
            filled_quantity=float(filled),
            average_price=float(average_price),
            fee_paid=float(notional * fee_rate),
        )

    def _execution_price(self, price: float, side: OrderSide) -> float:
        slippage = self.slippage_bps / 10_000
        return price * (1 + slippage if side == "buy" else 1 - slippage)

    @staticmethod
    def _validate_trades(trades: pd.DataFrame) -> None:
        missing = {"timestamp", "symbol", "price", "quantity"} - set(trades.columns)
        if missing:
            raise ValueError(f"agg_trades is missing columns: {sorted(missing)}")

    @staticmethod
    def _mark_to_market(
        cash: float,
        positions: Dict[str, float],
        trades: pd.DataFrame,
        timestamp: pd.Timestamp | None = None,
    ) -> float:
        equity = cash
        for symbol, quantity in positions.items():
            symbol_trades = trades[trades["symbol"] == symbol]
            if timestamp is not None:
                symbol_trades = symbol_trades[symbol_trades["timestamp"] <= timestamp]
            if len(symbol_trades) == 0:
                continue
            equity += quantity * float(symbol_trades.iloc[-1]["price"])
        return float(equity)


def compare_bar_vs_replay(
    bar_returns: np.ndarray,
    replay_returns: np.ndarray,
    tolerance: float = 0.3,
    annualization_factor: float = 252,
) -> Dict[str, float | bool]:
    """Compare bar and replay simulations by Sharpe drift."""
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")

    bar_sharpe = calc_sharpe_ratio(
        np.asarray(bar_returns, dtype=float),
        annualization_factor=annualization_factor,
    )
    replay_sharpe = calc_sharpe_ratio(
        np.asarray(replay_returns, dtype=float),
        annualization_factor=annualization_factor,
    )
    sharpe_diff = replay_sharpe - bar_sharpe
    return {
        "bar_sharpe": bar_sharpe,
        "replay_sharpe": replay_sharpe,
        "sharpe_diff": sharpe_diff,
        "abs_sharpe_diff": abs(sharpe_diff),
        "within_tolerance": abs(sharpe_diff) <= tolerance,
    }
