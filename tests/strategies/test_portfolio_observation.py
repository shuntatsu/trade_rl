from __future__ import annotations

import importlib
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


def _dataset(
    *,
    dataset_id: str = "d" * 64,
    tail_price: float = 100.0,
    symbol_names: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT"),
) -> MarketDataset:
    n_symbols = len(symbol_names)
    close = np.full((3, n_symbols), 100.0, dtype=np.float64)
    close[-1, :] = tail_price
    features = np.arange(3 * n_symbols, dtype=np.float32).reshape(3, n_symbols, 1)
    features[-1, :, 0] = tail_price
    return MarketDataset(
        dataset_id=dataset_id,
        symbols=symbol_names,
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(3) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.arange(3, dtype=np.float32).reshape(3, 1),
        open=close.copy(),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full((3, n_symbols), 1_000.0),
        funding_rate=np.zeros((3, n_symbols), dtype=np.float64),
        tradable=np.ones((3, n_symbols), dtype=np.bool_),
        feature_available=np.ones((3, n_symbols, 1), dtype=np.bool_),
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

    codec = importlib.import_module("trade_rl.strategies.portfolio_policy_codec")
    original_observation = codec.encode_portfolio_decision_snapshot(original)
    changed_tail_observation = codec.encode_portfolio_decision_snapshot(changed_tail)
    assert tuple(original_observation) == tuple(changed_tail_observation)
    for key in original_observation:
        np.testing.assert_array_equal(
            original_observation[key], changed_tail_observation[key]
        )


def test_portfolio_policy_codec_preserves_raw_v1_arrays_and_detaches_values() -> None:
    try:
        codec = importlib.import_module("trade_rl.strategies.portfolio_policy_codec")
    except ModuleNotFoundError:
        pytest.fail("the shared portfolio policy codec is not implemented")

    dataset = _dataset()
    snapshot = _snapshot_with_orders(dataset, _source_orders(dataset))
    encoded = codec.encode_portfolio_decision_snapshot(snapshot)

    assert tuple(encoded) == (
        "features",
        "feature_available",
        "feature_staleness",
        "global_features",
        "global_feature_available",
        "account",
        "portfolio",
        "risk",
        "active_orders",
        "clock",
        "terminal",
    )
    assert {key: value.dtype for key, value in encoded.items()} == {
        "features": np.dtype(np.float64),
        "feature_available": np.dtype(np.int8),
        "feature_staleness": np.dtype(np.float64),
        "global_features": np.dtype(np.float64),
        "global_feature_available": np.dtype(np.int8),
        "account": np.dtype(np.float64),
        "portfolio": np.dtype(np.float64),
        "risk": np.dtype(np.float64),
        "active_orders": np.dtype(np.float64),
        "clock": np.dtype(np.int64),
        "terminal": np.dtype(np.int8),
    }
    np.testing.assert_array_equal(encoded["features"], [[0.0], [1.0], [2.0]])
    np.testing.assert_array_equal(
        encoded["feature_available"], np.ones((3, 1), dtype=np.int8)
    )
    np.testing.assert_array_equal(
        encoded["account"],
        [[0.0, 0.0, 100.0, 1.0], [0.0, 0.0, 100.0, 1.0], [0.0, 0.0, 100.0, 1.0]],
    )
    assert codec.ACCOUNT_FIELDS == (
        "quantity",
        "weight",
        "mark_price",
        "contract_multiplier",
    )
    np.testing.assert_array_equal(
        encoded["active_orders"][:, [0, 1, 16, 17, 18, 19, 20]],
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 101.0],
            [1.0, 2.0, 95.0, 0.0, 0.0, 1.0, 102.0],
            [1.0, 3.0, 0.0, 1.0, 105.0, 1.0, 103.0],
        ],
    )
    np.testing.assert_array_equal(encoded["clock"], [0, snapshot.timestamp_ns, 2, 2])
    np.testing.assert_array_equal(encoded["terminal"], [0])
    encoded["features"][0, 0] = -99.0
    assert snapshot.features[0][0] == 0.0


def test_active_order_codec_encodes_absent_legacy_price_context_as_zero() -> None:
    codec = importlib.import_module("trade_rl.strategies.portfolio_policy_codec")

    encoded = codec.encode_portfolio_active_orders(1, (_legacy_snapshot(),))

    assert encoded.shape == (1, len(codec.ACTIVE_ORDER_FIELDS))
    np.testing.assert_array_equal(
        encoded[0, [0, 15, 16, 17, 18, 19, 20]], [1, 0, 0, 0, 0, 0, 0]
    )


@pytest.mark.parametrize("n_symbols", [0, -1, True, 1.0])
def test_active_order_codec_requires_a_positive_native_symbol_count(
    n_symbols: object,
) -> None:
    codec = importlib.import_module("trade_rl.strategies.portfolio_policy_codec")

    with pytest.raises(ValueError, match="positive native integer"):
        codec.encode_portfolio_active_orders(n_symbols, ())


def test_active_order_codec_rejects_symbol_index_outside_declared_shape() -> None:
    codec = importlib.import_module("trade_rl.strategies.portfolio_policy_codec")
    out_of_range = replace(_legacy_snapshot(), symbol_index=3)

    with pytest.raises(ValueError, match="symbol index.*roster"):
        codec.encode_portfolio_active_orders(3, (out_of_range,))


def test_portfolio_action_codec_maps_joint_codes_to_exact_hold_proposal() -> None:
    try:
        codec = importlib.import_module("trade_rl.strategies.portfolio_policy_codec")
    except ModuleNotFoundError:
        pytest.fail("the shared portfolio policy codec is not implemented")

    dataset = _dataset()
    snapshot = _snapshot_with_orders(dataset, ())
    proposal = codec.portfolio_action_to_proposal(
        np.asarray([0, 1, 2], dtype=np.int64),
        snapshot,
        (0.2, 0.3, 0.4),
    )

    assert proposal.target_weights == (0.0, -0.3, 0.0)
    assert proposal.hold_current_quantity_symbols == (0,)


def test_portfolio_action_codec_holds_nonzero_quantity_across_mixed_positions() -> None:
    try:
        codec = importlib.import_module("trade_rl.strategies.portfolio_policy_codec")
    except ModuleNotFoundError:
        pytest.fail("the shared portfolio policy codec is not implemented")

    dataset = _dataset(symbol_names=("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"))
    book = BookState.from_weights(
        weights=np.asarray([0.2, 0.0, -0.1, 0.0], dtype=np.float64),
        capital=1_000.0,
        prices=dataset.close[0],
        contract_multipliers=dataset.contract_multipliers,
        max_gross=1.0,
    )
    snapshot = build_portfolio_decision_snapshot(
        dataset,
        index=0,
        agent_stop_index=2,
        book=book,
        active_orders=(),
        risk_config=PreTradeRiskConfig(),
    )

    proposal = codec.portfolio_action_to_proposal(
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        snapshot,
        (0.4, 0.3, 0.2, 0.25),
    )

    assert snapshot.account.quantities == (2.0, 0.0, -1.0, 0.0)
    assert proposal.target_weights == (0.2, -0.3, 0.0, 0.25)
    assert proposal.hold_current_quantity_symbols == (0,)


@pytest.mark.parametrize(
    ("action", "error"),
    [
        ([True, 1, 2], TypeError),
        (np.asarray([0.0, 1.0, 2.0]), TypeError),
        (np.asarray([[0, 1, 2]], dtype=np.int64), ValueError),
        (np.asarray([0, 1, 4], dtype=np.int64), ValueError),
    ],
)
def test_portfolio_action_codec_rejects_coercion_shape_and_unknown_codes(
    action: object, error: type[Exception]
) -> None:
    try:
        codec = importlib.import_module("trade_rl.strategies.portfolio_policy_codec")
    except ModuleNotFoundError:
        pytest.fail("the shared portfolio policy codec is not implemented")

    dataset = _dataset()
    snapshot = _snapshot_with_orders(dataset, ())
    with pytest.raises(error):
        codec.portfolio_action_to_proposal(action, snapshot, (0.2, 0.3, 0.4))
