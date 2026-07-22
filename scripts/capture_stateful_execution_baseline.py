from __future__ import annotations

import json
from dataclasses import fields, is_dataclass, replace
from enum import Enum
from typing import Any

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.orders import (
    OrderBookState,
    OrderIntent,
    OrderType,
    TimeInForce,
)


def normalize(value: Any) -> Any:
    canonical_payload = getattr(value, "canonical_payload", None)
    if callable(canonical_payload):
        return normalize(canonical_payload())
    if is_dataclass(value):
        return {
            field.name: normalize(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return normalize(value.value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [normalize(item) for item in value]
    return value


def main() -> None:
    n_bars = 6
    shape = (n_bars, 1)
    open_price = np.full(shape, 100.0)
    high = np.full(shape, 110.0)
    low = np.full(shape, 90.0)
    close = np.array([[100.0], [102.0], [98.0], [104.0], [103.0], [105.0]])
    volume = np.full(shape, 2.0)
    dataset = MarketDataset(
        dataset_id="d" * 64,
        symbols=("S0",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )
    config = replace(
        ExecutionCostConfig.zero(),
        path_mode="conservative",
        processing_bar_volume_capacity=True,
        partial_fill_carry=True,
        max_participation_rate=1.0,
        lot_size=1.0,
        fee_rate=0.0005,
        spread_rate=0.001,
        impact_rate=0.0002,
        maker_fee_rate=0.0001,
        taker_fee_rate=0.0003,
    )
    executor = MarketExecutor(dataset, config)
    book = BookState.zero(
        1,
        1_000.0,
        dataset.close[0],
        dataset.resolved_array("contract_multipliers"),
    )

    def intent(
        quantity: float,
        *,
        target: str,
        order_type: OrderType,
        eligible_index: int,
        limit_price: float | None = None,
        stop_price: float | None = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
    ) -> OrderIntent:
        return OrderIntent.create(
            dataset_id=dataset.dataset_id,
            target_identity=target,
            execution_policy_digest=executor.execution_policy_digest,
            symbol_index=0,
            requested_quantity=quantity,
            order_type=order_type,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            submit_index=0,
            eligible_index=eligible_index,
            expiry_index=None,
            submission_reference_price=100.0,
            decision_equity=1_000.0,
        )

    result = executor.execute_orders(
        book,
        OrderBookState.empty(),
        (
            intent(
                2.0,
                target="market",
                order_type=OrderType.MARKET,
                eligible_index=1,
            ),
            intent(
                2.0,
                target="limit",
                order_type=OrderType.LIMIT,
                eligible_index=2,
                limit_price=95.0,
            ),
            intent(
                -1.0,
                target="stop",
                order_type=OrderType.STOP_MARKET,
                eligible_index=1,
                stop_price=95.0,
            ),
        ),
        start_index=0,
        bars=3,
    )
    payload = {
        "book": result.book,
        "order_book": result.order_book,
        "next_index": result.next_index,
        "bars_advanced": result.bars_advanced,
        "order_events": result.order_events,
        "capacity_evidence": result.capacity_evidence,
        "interval_cost": result.interval_cost,
        "interval_funding": result.interval_funding,
        "interval_borrow_cost": result.interval_borrow_cost,
        "interval_dividend": result.interval_dividend,
        "interval_cash_interest": result.interval_cash_interest,
        "interval_gross_return": result.interval_gross_return,
        "interval_net_return": result.interval_net_return,
        "interval_log_return": result.interval_log_return,
        "requested_notional": result.requested_notional,
        "filled_notional": result.filled_notional,
        "requested_turnover": result.requested_turnover,
        "filled_turnover": result.filled_turnover,
        "unfilled_turnover": result.unfilled_turnover,
        "fill_ratio": result.fill_ratio,
        "rebalance_events": result.rebalance_events,
        "completed_fill_count": result.completed_fill_count,
        "rejected_count": result.rejected_count,
        "expired_count": result.expired_count,
        "fill_count": result.fill_count,
        "max_participation": result.max_participation,
        "requested_notional_by_symbol": result.requested_notional_by_symbol,
        "filled_notional_by_symbol": result.filled_notional_by_symbol,
        "participation_by_symbol": result.participation_by_symbol,
        "cost_by_symbol": result.cost_by_symbol,
        "termination_reason": result.termination_reason,
    }
    print(json.dumps(normalize(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
