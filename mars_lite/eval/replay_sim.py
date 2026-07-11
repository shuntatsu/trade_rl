"""Deterministic aggregate-trade replay with shared liquidity accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Literal

import numpy as np
import pandas as pd

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
class FillSlice:
    timestamp: pd.Timestamp
    quantity: float
    price: float
    fee_paid: float


@dataclass(frozen=True)
class ReplayFill:
    order_timestamp: pd.Timestamp
    symbol: str
    side: OrderSide
    requested_quantity: float
    filled_quantity: float
    average_price: float
    fee_paid: float
    first_fill_timestamp: pd.Timestamp | None
    last_fill_timestamp: pd.Timestamp | None
    slices: tuple[FillSlice, ...]


@dataclass(frozen=True)
class ReplayResult:
    fills: list[ReplayFill]
    final_cash: float
    final_position: Dict[str, float]
    final_equity: float
    equity_timestamps: list[pd.Timestamp]
    equity_curve: list[float]
    returns: list[float]
    sharpe: float
    annualization_factor: float


class LiquidityLedger:
    """Track participation-limited volume remaining on each trade row."""

    def __init__(self, trades: pd.DataFrame, max_participation_rate: float) -> None:
        self.remaining_volume = (
            trades["quantity"].to_numpy(dtype=float) * max_participation_rate
        )

    def consume(self, row_index: int, requested: float) -> float:
        available = float(self.remaining_volume[row_index])
        taken = min(available, requested)
        self.remaining_volume[row_index] = available - taken
        return taken


class ReplaySimulator:
    def __init__(
        self,
        fee_rate: float = 0.0005,
        max_participation_rate: float = 0.1,
        slippage_bps: float = 0.0,
        maker_fee_rate: float | None = None,
        equity_frequency: str = "1min",
    ) -> None:
        if not 0 < max_participation_rate <= 1:
            raise ValueError("max_participation_rate must be in (0, 1]")
        if fee_rate < 0:
            raise ValueError("fee_rate must be non-negative")
        if slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        if maker_fee_rate is not None and maker_fee_rate < 0:
            raise ValueError("maker_fee_rate must be non-negative")
        self.fee_rate = fee_rate
        self.max_participation_rate = max_participation_rate
        self.slippage_bps = slippage_bps
        self.maker_fee_rate = maker_fee_rate if maker_fee_rate is not None else fee_rate
        self.equity_frequency = equity_frequency
        self.annualization_factor = _annualization_factor(equity_frequency)

    def simulate(
        self,
        agg_trades: pd.DataFrame,
        orders: Iterable[ExecutionOrder],
        initial_cash: float = 1_000_000.0,
    ) -> ReplayResult:
        self._validate_trades(agg_trades)
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")

        trades = (
            agg_trades.copy()
            .assign(timestamp=lambda frame: pd.to_datetime(frame["timestamp"]))
            .sort_values(["timestamp", "symbol"], kind="mergesort")
            .reset_index(drop=True)
        )
        ledger = LiquidityLedger(trades, self.max_participation_rate)
        fills: list[ReplayFill] = []
        fill_events: list[tuple[pd.Timestamp, int, int, str, str, FillSlice]] = []

        indexed_orders = list(enumerate(orders))
        indexed_orders.sort(key=lambda item: (pd.Timestamp(item[1].timestamp), item[0]))
        for order_priority, order in indexed_orders:
            fill = self._fill_order(trades, ledger, order)
            fills.append(fill)
            for slice_index, fill_slice in enumerate(fill.slices):
                fill_events.append(
                    (
                        fill_slice.timestamp,
                        order_priority,
                        slice_index,
                        order.symbol,
                        order.side,
                        fill_slice,
                    )
                )

        equity_timestamps, equity_curve, final_cash, final_positions = (
            self._build_uniform_equity_curve(
                trades=trades,
                fill_events=fill_events,
                initial_cash=float(initial_cash),
            )
        )
        equity_array = np.asarray(equity_curve, dtype=float)
        returns = (
            np.diff(equity_array) / np.maximum(np.abs(equity_array[:-1]), 1e-12)
            if len(equity_array) > 1
            else np.asarray([], dtype=float)
        )
        sharpe = _calc_sharpe(returns, self.annualization_factor)
        final_equity = float(equity_curve[-1]) if equity_curve else float(initial_cash)

        return ReplayResult(
            fills=fills,
            final_cash=final_cash,
            final_position=final_positions,
            final_equity=final_equity,
            equity_timestamps=equity_timestamps,
            equity_curve=[float(value) for value in equity_curve],
            returns=[float(value) for value in returns],
            sharpe=sharpe,
            annualization_factor=self.annualization_factor,
        )

    def _fill_order(
        self, trades: pd.DataFrame, ledger: LiquidityLedger, order: ExecutionOrder
    ) -> ReplayFill:
        remaining = float(order.quantity)
        if remaining <= 0:
            raise ValueError("order quantity must be positive")
        if order.side not in ("buy", "sell"):
            raise ValueError(f"invalid order side: {order.side}")
        if order.order_type not in ("market", "limit"):
            raise ValueError(f"invalid order type: {order.order_type}")
        if order.order_type == "limit" and order.limit_price is None:
            raise ValueError("limit orders require limit_price")
        if order.latency_seconds < 0 or order.max_delay_seconds < 0:
            raise ValueError(
                "latency_seconds and max_delay_seconds must be non-negative"
            )

        effective_start = pd.Timestamp(order.timestamp) + pd.Timedelta(
            seconds=order.latency_seconds
        )
        effective_end = pd.Timestamp(order.timestamp) + pd.Timedelta(
            seconds=order.max_delay_seconds
        )
        slices: list[FillSlice] = []

        for row_index in range(len(trades)):
            row_timestamp = pd.Timestamp(trades.at[row_index, "timestamp"])
            if row_timestamp < effective_start:
                continue
            if row_timestamp > effective_end:
                break
            if trades.at[row_index, "symbol"] != order.symbol:
                continue

            row_price = float(trades.at[row_index, "price"])
            if order.order_type == "limit":
                assert order.limit_price is not None
                if order.side == "buy" and row_price > order.limit_price:
                    continue
                if order.side == "sell" and row_price < order.limit_price:
                    continue

            taken = ledger.consume(row_index, remaining)
            if taken <= 0:
                continue
            execution_price = self._execution_price(row_price, order.side)
            fee_rate = self.maker_fee_rate if order.maker else self.fee_rate
            fee_paid = taken * execution_price * fee_rate
            slices.append(
                FillSlice(
                    timestamp=row_timestamp,
                    quantity=float(taken),
                    price=float(execution_price),
                    fee_paid=float(fee_paid),
                )
            )
            remaining -= taken
            if remaining <= 1e-12:
                break

        filled_quantity = float(sum(item.quantity for item in slices))
        notional = float(sum(item.quantity * item.price for item in slices))
        average_price = notional / filled_quantity if filled_quantity else 0.0
        total_fee = float(sum(item.fee_paid for item in slices))
        return ReplayFill(
            order_timestamp=pd.Timestamp(order.timestamp),
            symbol=order.symbol,
            side=order.side,
            requested_quantity=float(order.quantity),
            filled_quantity=filled_quantity,
            average_price=float(average_price),
            fee_paid=total_fee,
            first_fill_timestamp=slices[0].timestamp if slices else None,
            last_fill_timestamp=slices[-1].timestamp if slices else None,
            slices=tuple(slices),
        )

    def _build_uniform_equity_curve(
        self,
        *,
        trades: pd.DataFrame,
        fill_events: list[tuple[pd.Timestamp, int, int, str, str, FillSlice]],
        initial_cash: float,
    ) -> tuple[list[pd.Timestamp], list[float], float, Dict[str, float]]:
        if trades.empty:
            return [], [initial_cash], initial_cash, {}

        start = pd.Timestamp(trades["timestamp"].min()).floor(self.equity_frequency)
        end = pd.Timestamp(trades["timestamp"].max()).ceil(self.equity_frequency)
        timeline = list(pd.date_range(start, end, freq=self.equity_frequency))
        fill_events.sort(key=lambda item: (item[0], item[1], item[2]))

        price_table = trades.pivot_table(
            index="timestamp", columns="symbol", values="price", aggfunc="last"
        ).sort_index()
        combined_index = price_table.index.union(
            pd.DatetimeIndex(timeline)
        ).sort_values()
        price_grid = price_table.reindex(combined_index).ffill().reindex(timeline)

        cash = float(initial_cash)
        positions: Dict[str, float] = {}
        event_index = 0
        equities: list[float] = []
        for timestamp_index, timestamp in enumerate(timeline):
            while (
                event_index < len(fill_events)
                and fill_events[event_index][0] <= timestamp
            ):
                _, _, _, symbol, side, fill_slice = fill_events[event_index]
                signed_quantity = (
                    fill_slice.quantity if side == "buy" else -fill_slice.quantity
                )
                cash -= signed_quantity * fill_slice.price
                cash -= fill_slice.fee_paid
                positions[symbol] = positions.get(symbol, 0.0) + signed_quantity
                event_index += 1

            equity = cash
            if timestamp_index < len(price_grid):
                prices = price_grid.iloc[timestamp_index]
                for symbol, quantity in positions.items():
                    price = prices.get(symbol)
                    if price is not None and not pd.isna(price):
                        equity += quantity * float(price)
            equities.append(float(equity))

        return timeline, equities, cash, positions

    def _execution_price(self, price: float, side: OrderSide) -> float:
        slippage = self.slippage_bps / 10_000
        return price * (1 + slippage if side == "buy" else 1 - slippage)

    @staticmethod
    def _validate_trades(trades: pd.DataFrame) -> None:
        missing = {"timestamp", "symbol", "price", "quantity"} - set(trades.columns)
        if missing:
            raise ValueError(f"agg_trades is missing columns: {sorted(missing)}")
        if len(trades) and (
            not np.isfinite(trades["price"].to_numpy(dtype=float)).all()
            or not np.isfinite(trades["quantity"].to_numpy(dtype=float)).all()
        ):
            raise ValueError("agg_trades contains non-finite price or quantity")
        if len(trades) and (trades["quantity"].to_numpy(dtype=float) < 0).any():
            raise ValueError("agg_trades quantity must be non-negative")


def compare_bar_vs_replay(
    bar_returns: np.ndarray,
    replay_returns: np.ndarray,
    tolerance: float = 0.3,
    annualization_factor: float = 252,
) -> Dict[str, float | bool]:
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if annualization_factor <= 0:
        raise ValueError("annualization_factor must be positive")
    bar_sharpe = _calc_sharpe(
        np.asarray(bar_returns, dtype=float), annualization_factor
    )
    replay_sharpe = _calc_sharpe(
        np.asarray(replay_returns, dtype=float), annualization_factor
    )
    sharpe_diff = replay_sharpe - bar_sharpe
    return {
        "bar_sharpe": bar_sharpe,
        "replay_sharpe": replay_sharpe,
        "sharpe_diff": sharpe_diff,
        "abs_sharpe_diff": abs(sharpe_diff),
        "within_tolerance": abs(sharpe_diff) <= tolerance,
    }


def _annualization_factor(frequency: str) -> float:
    try:
        offset = pd.tseries.frequencies.to_offset(frequency)
        nanos = offset.nanos
    except (ValueError, TypeError) as exc:
        raise ValueError("equity_frequency must be a fixed pandas frequency") from exc
    seconds = nanos / 1_000_000_000
    if seconds <= 0:
        raise ValueError("equity_frequency must be positive")
    return float(365.25 * 24 * 60 * 60 / seconds)


def _calc_sharpe(returns: np.ndarray, annualization_factor: float) -> float:
    values = np.asarray(returns, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return 0.0
    std = float(np.std(values, ddof=1))
    if std <= 1e-15:
        return 0.0
    return float(np.mean(values) / std * np.sqrt(annualization_factor))
