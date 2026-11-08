from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.simulation.accounting import BookState
from trade_rl.simulation.bar_path import TriggerSegment
from trade_rl.simulation.depth import DepthOrderRules, execute_depth_order
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.orders.model import (
    OrderBookState,
    OrderIntent,
    OrderType,
    TimeInForce,
)
from trade_rl.simulation.stateful.symbol_fills import _execution_cost


def _dataset(*, price: float, multiplier: float) -> MarketDataset:
    shape = (4, 1)
    prices = np.full(shape, price)
    return MarketDataset(
        dataset_id="e" * 64,
        symbols=("S0",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(4) * np.timedelta64(1, "h"),
        features=np.zeros((4, 1, 1), dtype=np.float32),
        global_features=np.zeros((4, 1), dtype=np.float32),
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        volume=np.ones(shape),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((4, 1, 1), dtype=np.bool_),
        feature_names=("x",),
        global_feature_names=("g",),
        periods_per_year=8_760,
        contract_multipliers=np.array([multiplier]),
        volume_units=(VolumeUnit.QUOTE_NOTIONAL,),
    )


def _intent(
    executor: MarketExecutor, *, quantity: float, reference_price: float
) -> OrderIntent:
    return OrderIntent.create(
        dataset_id=executor.dataset.dataset_id,
        target_identity="cost-underflow",
        execution_policy_digest=executor.execution_policy_digest,
        symbol_index=0,
        requested_quantity=quantity,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC,
        limit_price=None,
        stop_price=None,
        submit_index=0,
        eligible_index=1,
        expiry_index=None,
        submission_reference_price=reference_price,
        decision_equity=1_000.0,
    )


def test_stateful_rejects_nonzero_execution_cost_underflow_before_book_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(price=1e-100, multiplier=1e-100)
    executor = MarketExecutor(
        dataset, replace(ExecutionCostConfig.zero(), fee_rate=1e-200)
    )
    book = BookState.zero(1, 1_000.0, np.array([1e-100]), np.array([1e-100]))
    calls = 0
    original = BookState.execute_fill

    def observed(self: BookState, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        original(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(BookState, "execute_fill", observed)
    with pytest.raises(ValueError, match="execution cost.*represented"):
        executor.execute_orders(
            book,
            OrderBookState.empty(),
            (_intent(executor, quantity=1.0, reference_price=1e-100),),
            start_index=0,
            bars=1,
        )
    assert calls == 0
    assert book.quantities.tolist() == [0.0]
    assert book.cash == 1_000.0
    assert book.total_cost == 0.0


def test_stateful_rejects_unrepresentable_nonzero_unit_cost_before_book_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(price=1.0, multiplier=1.0)
    executor = MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            fee_rate=1e-200,
            multiplier=1e-200,
        ),
    )
    book = BookState.zero(1, 1_000.0, np.array([1.0]))
    calls = 0
    original = BookState.execute_fill

    def observed(self: BookState, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        original(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(BookState, "execute_fill", observed)
    with pytest.raises(ValueError, match="execution cost rate.*represented"):
        executor.execute_orders(
            book,
            OrderBookState.empty(),
            (_intent(executor, quantity=1.0, reference_price=1.0),),
            start_index=0,
            bars=1,
        )
    assert calls == 0
    assert book.quantities.tolist() == [0.0]
    assert book.total_cost == 0.0


def test_legacy_liquidation_rejects_cost_underflow_before_book_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(price=1e-100, multiplier=1e-100)
    executor = MarketExecutor(
        dataset, replace(ExecutionCostConfig.zero(), fee_rate=1e-200)
    )
    book = BookState(
        quantities=np.array([1.0]),
        cash=1_000.0,
        mark_prices=np.array([1e-100]),
        peak_value=1_000.0,
        contract_multipliers=np.array([1e-100]),
    )
    calls = 0
    original = BookState.execute

    def observed(self: BookState, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        original(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(BookState, "execute", observed)
    with pytest.raises(ValueError, match="execution cost.*represented"):
        executor.liquidate_at_close(book, index=1)
    assert calls == 0
    assert book.quantities.tolist() == [1.0]
    assert book.cash == 1_000.0
    assert book.total_cost == 0.0


def test_depth_rejects_nonzero_fee_underflow_before_book_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    book = BookState.zero(1, 1_000.0, np.array([1.0]), np.array([1e-100]))
    calls = 0
    original = BookState.execute_fill

    def observed(self: BookState, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        original(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(BookState, "execute_fill", observed)
    now = datetime(2026, 11, 6, tzinfo=UTC)
    with pytest.raises(ValueError, match="execution cost.*represented"):
        execute_depth_order(
            book=book,
            symbol_index=0,
            lot_count=1,
            rules=DepthOrderRules(1.0, 0.0, 10.0, 0.0, None),
            bids=((9e-101, 10.0),),
            asks=((1e-100, 10.0),),
            decision_at=now,
            quote_at=now + timedelta(seconds=1),
            execution_at=now + timedelta(seconds=2),
            fee_bps=1e-196,
            adverse_bps=0.0,
            depth_fraction=1.0,
        )
    assert calls == 0
    assert book.quantities.tolist() == [0.0]
    assert book.cash == 1_000.0
    assert book.total_cost == 0.0


def test_true_zero_execution_cost_still_allows_fill() -> None:
    dataset = _dataset(price=1.0, multiplier=1.0)
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState.zero(1, 1_000.0, np.array([1.0]))
    result = executor.execute_orders(
        book,
        OrderBookState.empty(),
        (_intent(executor, quantity=1.0, reference_price=1.0),),
        start_index=0,
        bars=1,
    )
    assert result.fill_count == 1
    assert result.interval_cost == 0.0
    assert result.book.total_cost == 0.0


def test_legacy_no_fill_does_not_charge_unrepresentable_spread() -> None:
    """A no-op has no execution cost even if configured spread would underflow."""
    dataset = _dataset(price=1.0, multiplier=1.0)
    executor = MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            order_type="limit",
            spread_rate=5e-324,
            fee_rate=1e-200,
        ),
    )
    book = BookState.zero(1, 1_000.0, np.array([1.0]))
    result = executor._fill_toward_quantities(
        book,
        np.zeros(1),
        prices=np.ones(1),
        capacity_volume=np.ones(1),
        tradable=np.ones(1, dtype=np.bool_),
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert result.fill_count == 0
    assert result.cost_amount == 0.0
    assert book.quantities.tolist() == [0.0]
    assert book.cash == 1_000.0


def test_explicit_zero_cost_multiplier_allows_fill_with_tiny_spread() -> None:
    """A zero global cost multiplier is a genuine configured zero cost."""
    dataset = _dataset(price=1.0, multiplier=1.0)
    executor = MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            order_type="limit",
            spread_rate=5e-324,
            fee_rate=1e-200,
            multiplier=0.0,
        ),
    )
    book = BookState.zero(1, 1_000.0, np.array([1.0]))
    result = executor._fill_toward_quantities(
        book,
        np.ones(1),
        prices=np.ones(1),
        capacity_volume=np.ones(1),
        tradable=np.ones(1, dtype=np.bool_),
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert result.fill_count == 1
    assert result.cost_amount == 0.0
    assert book.quantities.tolist() == [1.0]
    assert book.total_cost == 0.0


def test_stateful_maker_fee_zero_multiplier_allows_real_fill() -> None:
    """A genuine zero cost multiplier must not fail a valid resting limit fill."""
    executor = MarketExecutor(
        _dataset(price=1.0, multiplier=1.0),
        replace(
            ExecutionCostConfig.zero(),
            fee_rate=1e-200,
            spread_rate=5e-324,
            multiplier=0.0,
        ),
    )
    runtime = SimpleNamespace(executor=executor)
    order = SimpleNamespace(
        intent=SimpleNamespace(
            symbol_index=0,
            order_type=OrderType.LIMIT,
            eligible_index=0,
        )
    )
    trigger = SimpleNamespace(segment=TriggerSegment.FIRST_EXTREME)
    assert (
        _execution_cost(
            runtime,
            order,
            trigger=trigger,
            processing_index=1,
            filled_notional=1.0,
            participation_rate=1.0,
        )
        == 0.0
    )


@pytest.mark.parametrize("desired_quantity", [0.0, 1.0])
def test_legacy_nonchargeable_identity_ignores_nonfinite_intermediate_cost(
    desired_quantity: float,
) -> None:
    """Finite configured rates may overflow in an otherwise zero-cost identity."""
    executor = MarketExecutor(
        _dataset(price=1.0, multiplier=1.0),
        replace(
            ExecutionCostConfig.zero(),
            fee_rate=1e308,
            spread_rate=1e308,
            multiplier=1.0 if desired_quantity == 0.0 else 0.0,
        ),
    )
    book = BookState.zero(1, 1_000.0, np.array([1.0]))
    result = executor._fill_toward_quantities(
        book,
        np.array([desired_quantity]),
        prices=np.ones(1),
        capacity_volume=np.ones(1),
        tradable=np.ones(1, dtype=np.bool_),
        turnover_denominator=1_000.0,
        market_index=1,
    )
    assert result.fill_count == int(desired_quantity != 0.0)
    assert result.cost_amount == 0.0
    assert book.total_cost == 0.0
    assert book.quantities.tolist() == [desired_quantity]


def test_stateful_zero_multiplier_ignores_nonfinite_intermediate_cost() -> None:
    """Cost-disabled stateful fills should not reject a nonchargeable sum."""
    executor = MarketExecutor(
        _dataset(price=1.0, multiplier=1.0),
        replace(
            ExecutionCostConfig.zero(),
            fee_rate=1.5e308,
            spread_rate=1e308,
            multiplier=0.0,
        ),
    )
    runtime = SimpleNamespace(executor=executor)
    order = SimpleNamespace(
        intent=SimpleNamespace(
            symbol_index=0,
            order_type=OrderType.LIMIT,
            eligible_index=0,
        )
    )
    trigger = SimpleNamespace(segment=TriggerSegment.FIRST_EXTREME)
    assert (
        _execution_cost(
            runtime,
            order,
            trigger=trigger,
            processing_index=1,
            filled_notional=1.0,
            participation_rate=1.0,
        )
        == 0.0
    )


@pytest.mark.parametrize("quantity", [0.0, 1.0])
def test_legacy_strict_numpy_does_not_raise_on_nonchargeable_spread_sum(
    quantity: float,
) -> None:
    """Strict NumPy must not invalidate a no-fill or zero-multiplier identity."""
    dataset = replace(
        _dataset(price=1.0, multiplier=1.0),
        spread_rate=np.full((4, 1), 1e308),
    )
    executor = MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            spread_rate=1e308,
            multiplier=1.0 if quantity == 0.0 else 0.0,
        ),
    )
    book = BookState.zero(1, 1_000.0, np.array([1.0]))
    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = executor._fill_toward_quantities(
            book,
            np.array([quantity]),
            prices=np.ones(1),
            capacity_volume=np.ones(1),
            tradable=np.ones(1, dtype=np.bool_),
            turnover_denominator=1_000.0,
            market_index=1,
        )
    assert result.cost_amount == 0.0
    assert book.quantities.tolist() == [quantity]


def test_stateful_strict_numpy_zero_multiplier_ignores_spread_sum_overflow() -> None:
    dataset = replace(
        _dataset(price=1.0, multiplier=1.0),
        spread_rate=np.full((4, 1), 1e308),
    )
    executor = MarketExecutor(
        dataset,
        replace(ExecutionCostConfig.zero(), spread_rate=1e308, multiplier=0.0),
    )
    runtime = SimpleNamespace(executor=executor)
    order = SimpleNamespace(
        intent=SimpleNamespace(
            symbol_index=0,
            order_type=OrderType.LIMIT,
            eligible_index=0,
        )
    )
    trigger = SimpleNamespace(segment=TriggerSegment.FIRST_EXTREME)
    with np.errstate(over="raise", under="raise", invalid="raise"):
        amount = _execution_cost(
            runtime,
            order,
            trigger=trigger,
            processing_index=1,
            filled_notional=1.0,
            participation_rate=1.0,
        )
    assert amount == 0.0


def test_depth_finite_fee_avoids_avoidable_intermediate_overflow() -> None:
    """A finite 5bp fee must not be classified as an infinite cost."""
    notional = 1e308
    fee_bps = 5.0
    expected = notional * (fee_bps / 10_000)
    assert np.isfinite(expected)
    book = BookState.zero(1, notional, np.array([notional]), np.array([1.0]))
    now = datetime(2026, 11, 6, tzinfo=UTC)

    fill = execute_depth_order(
        book=book,
        symbol_index=0,
        lot_count=1,
        rules=DepthOrderRules(1.0, 0.0, 2.0, 0.0, None),
        bids=((9e307, 10.0),),
        asks=((notional, 10.0),),
        decision_at=now,
        quote_at=now + timedelta(seconds=1),
        execution_at=now + timedelta(seconds=2),
        fee_bps=fee_bps,
        adverse_bps=0.0,
        depth_fraction=1.0,
    )
    assert fill.filled_lots == 1
    assert fill.unfilled_lots == 0
    assert fill.fee == expected
    assert book.quantities.tolist() == [1.0]
    assert book.total_cost == expected
    assert np.isfinite(book.cash)
    assert np.isfinite(book.portfolio_value)


@pytest.mark.parametrize(
    ("desired_quantity", "cost_multiplier"),
    [(0.0, 1.0), (1.0, 0.0)],
)
def test_legacy_strict_numpy_slippage_overflow_preserves_nonchargeable_identity(
    desired_quantity: float, cost_multiplier: float
) -> None:
    executor = MarketExecutor(
        _dataset(price=1.0, multiplier=1.0),
        replace(
            ExecutionCostConfig.zero(),
            slippage_std=1e308,
            tail_slippage_probability=1.0,
            tail_slippage_multiplier=1e308,
            random_seed=0,
            multiplier=cost_multiplier,
        ),
    )
    book = BookState.zero(1, 1_000.0, np.array([1.0]))
    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = executor._fill_toward_quantities(
            book,
            np.array([desired_quantity]),
            prices=np.ones(1),
            capacity_volume=np.ones(1),
            tradable=np.ones(1, dtype=np.bool_),
            turnover_denominator=1_000.0,
            market_index=1,
        )
    assert result.fill_count == int(desired_quantity != 0.0)
    assert result.cost_amount == 0.0
    assert book.total_cost == 0.0
    assert book.quantities.tolist() == [desired_quantity]


def test_legacy_strict_numpy_slippage_overflow_is_rejected_before_book_mutation() -> (
    None
):
    executor = MarketExecutor(
        _dataset(price=1.0, multiplier=1.0),
        replace(
            ExecutionCostConfig.zero(),
            slippage_std=1e308,
            tail_slippage_probability=1.0,
            tail_slippage_multiplier=1e308,
            random_seed=0,
            multiplier=1.0,
        ),
    )
    book = BookState.zero(1, 1_000.0, np.array([1.0]))
    with pytest.raises(ValueError, match="execution cost became non-finite"):
        with np.errstate(over="raise", under="raise", invalid="raise"):
            executor._fill_toward_quantities(
                book,
                np.ones(1),
                prices=np.ones(1),
                capacity_volume=np.ones(1),
                tradable=np.ones(1, dtype=np.bool_),
                turnover_denominator=1_000.0,
                market_index=1,
            )
    assert book.quantities.tolist() == [0.0]
    assert book.cash == 1_000.0
    assert book.total_cost == 0.0
