from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.orders.model import (
    OrderIntent,
    OrderType,
    PendingOrder,
    TimeInForce,
)
from trade_rl.strategies.portfolio import PortfolioPendingOrderSnapshot
from trade_rl.strategies.portfolio_observation import (
    build_portfolio_decision_snapshot,
)


def _dataset(*, dataset_id: str = "d" * 64, tail_price: float = 100.0) -> MarketDataset:
    close = np.full((3, 3), 100.0, dtype=np.float64)
    close[-1, :] = tail_price
    features = np.arange(9, dtype=np.float32).reshape(3, 3, 1)
    features[-1, :, 0] = tail_price
    return MarketDataset(
        dataset_id=dataset_id,
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(3) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.arange(3, dtype=np.float32).reshape(3, 1),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((3, 3), 1_000.0),
        funding_rate=np.zeros((3, 3), dtype=np.float64),
        tradable=np.ones((3, 3), dtype=np.bool_),
        feature_available=np.ones((3, 3, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def _source_orders(dataset: MarketDataset) -> tuple[PendingOrder, ...]:
    specs = (
        (OrderType.MARKET, None, None, 101.0),
        (OrderType.LIMIT, 95.0, None, 102.0),
        (OrderType.STOP_MARKET, None, 105.0, 103.0),
    )
    return tuple(
        PendingOrder.from_intent(
            OrderIntent.create(
                dataset_id=dataset.dataset_id,
                target_identity=f"target-{index}",
                execution_policy_digest="e" * 64,
                symbol_index=index,
                requested_quantity=1.0,
                order_type=order_type,
                time_in_force=TimeInForce.GTC,
                limit_price=limit_price,
                stop_price=stop_price,
                submit_index=0,
                eligible_index=1,
                expiry_index=None,
                submission_reference_price=reference_price,
                decision_equity=1_000.0,
            )
        )
        for index, (order_type, limit_price, stop_price, reference_price) in enumerate(
            specs
        )
    )


def _snapshot_with_orders(dataset: MarketDataset, orders: tuple[PendingOrder, ...]):
    book = BookState.zero(
        dataset.n_symbols,
        1_000.0,
        dataset.close[0],
        dataset.contract_multipliers,
    )
    return build_portfolio_decision_snapshot(
        dataset,
        index=0,
        agent_stop_index=2,
        book=book,
        active_orders=orders,
        risk_config=PreTradeRiskConfig(),
    )


def _legacy_snapshot(*, order_type: str = "market") -> PortfolioPendingOrderSnapshot:
    return PortfolioPendingOrderSnapshot(
        symbol_index=0,
        order_type=order_type,
        time_in_force="gtc",
        requested_quantity=1.0,
        remaining_quantity=1.0,
        status="submitted",
        submit_index=0,
        eligible_index=1,
        last_processed_index=None,
        trigger_index=None,
        expiry_index=None,
        reduce_only=False,
    )


def test_builder_projects_actual_pending_order_price_context() -> None:
    dataset = _dataset()
    snapshot = _snapshot_with_orders(dataset, _source_orders(dataset))

    assert [order.order_type for order in snapshot.active_orders] == [
        "market",
        "limit",
        "stop_market",
    ]
    assert [
        (order.limit_price, order.stop_price, order.submission_reference_price)
        for order in snapshot.active_orders
    ] == [
        (None, None, 101.0),
        (95.0, None, 102.0),
        (None, 105.0, 103.0),
    ]
    assert not hasattr(snapshot.active_orders[0], "order_id")
    assert not hasattr(snapshot.active_orders[0], "dataset_id")


def test_old_pending_snapshot_constructor_remains_valid_without_price_context() -> None:
    snapshot = _legacy_snapshot()

    assert snapshot.limit_price is None
    assert snapshot.stop_price is None
    assert snapshot.submission_reference_price is None


@pytest.mark.parametrize(
    ("order_type", "price_changes"),
    [
        ("market", {"limit_price": 99.0, "submission_reference_price": 100.0}),
        ("limit", {"submission_reference_price": 100.0}),
        (
            "limit",
            {
                "limit_price": 99.0,
                "stop_price": 101.0,
                "submission_reference_price": 100.0,
            },
        ),
        ("stop_market", {"submission_reference_price": 100.0}),
        (
            "stop_market",
            {
                "limit_price": 99.0,
                "stop_price": 101.0,
                "submission_reference_price": 100.0,
            },
        ),
        ("unknown", {"submission_reference_price": 100.0}),
    ],
)
def test_pending_snapshot_rejects_partial_or_wrong_order_price_context(
    order_type: str, price_changes: dict[str, object]
) -> None:
    with pytest.raises(ValueError):
        replace(_legacy_snapshot(order_type=order_type), **price_changes)


@pytest.mark.parametrize(
    ("order_type", "field_name", "value", "error"),
    [
        ("market", "submission_reference_price", True, TypeError),
        ("market", "submission_reference_price", float("nan"), ValueError),
        ("market", "submission_reference_price", 0.0, ValueError),
        ("limit", "limit_price", True, TypeError),
        ("limit", "limit_price", float("inf"), ValueError),
        ("limit", "limit_price", 0.0, ValueError),
        ("stop_market", "stop_price", True, TypeError),
        ("stop_market", "stop_price", float("nan"), ValueError),
        ("stop_market", "stop_price", 0.0, ValueError),
    ],
)
def test_pending_snapshot_price_values_must_be_positive_finite_nonbool_reals(
    order_type: str,
    field_name: str,
    value: object,
    error: type[Exception],
) -> None:
    values: dict[str, object] = {"submission_reference_price": 100.0}
    if order_type == "limit":
        values["limit_price"] = 99.0
    elif order_type == "stop_market":
        values["stop_price"] = 101.0
    values[field_name] = value

    with pytest.raises(error):
        replace(_legacy_snapshot(order_type=order_type), **values)


def test_pending_price_snapshot_is_detached_from_dataset_future_tail() -> None:
    original_dataset = _dataset()
    changed_tail_dataset = _dataset(dataset_id="f" * 64, tail_price=999.0)
    orders = _source_orders(original_dataset)

    original = _snapshot_with_orders(original_dataset, orders)
    changed_tail = _snapshot_with_orders(changed_tail_dataset, orders)

    assert original == changed_tail
    assert original.features == ((0.0,), (1.0,), (2.0,))
    assert original.active_orders[1].limit_price == 95.0
    with pytest.raises(FrozenInstanceError):
        original.active_orders[1].limit_price = 96.0  # type: ignore[misc]
