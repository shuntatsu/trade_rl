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


@pytest.mark.parametrize("style", ["legacy", "stateful"])
def test_nonzero_sampled_slippage_underflow_is_rejected_before_accounting(
    style: str,
) -> None:
    """A nonzero standard-normal draw cannot become a free fill at the float floor."""
    tiny = float(np.nextafter(0.0, 1.0))
    assert np.random.default_rng(0).standard_normal(1)[0] > 0.0
    executor = MarketExecutor(
        _dataset(price=1.0, multiplier=1.0),
        replace(ExecutionCostConfig.zero(), slippage_std=tiny, random_seed=0),
    )
    if style == "stateful":
        runtime = SimpleNamespace(executor=executor)
        order = SimpleNamespace(
            intent=SimpleNamespace(
                symbol_index=0, order_type=OrderType.LIMIT, eligible_index=0
            )
        )
        trigger = SimpleNamespace(segment=TriggerSegment.FIRST_EXTREME)
        with pytest.raises(ValueError, match="nonzero execution cost rate"):
            _execution_cost(
                runtime,
                order,
                trigger=trigger,
                processing_index=1,
                filled_notional=1.0,
                participation_rate=1.0,
            )
        return

    book = BookState.zero(1, 1_000.0, np.array([1.0]))
    with pytest.raises(ValueError, match="nonzero execution cost rate"):
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


@pytest.mark.parametrize("std", [1e-200, 0.001, 1.0, 1e200])
def test_slippage_sampler_preserves_normal_draw_bits_and_rng_stream(std: float) -> None:
    executor = MarketExecutor(
        _dataset(price=1.0, multiplier=1.0),
        replace(
            ExecutionCostConfig.zero(),
            slippage_std=std,
            tail_slippage_probability=0.4,
            tail_slippage_multiplier=3.0,
            random_seed=7,
        ),
    )
    expected_rng = np.random.default_rng(7)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        expected = np.abs(expected_rng.normal(0.0, std, size=8))
        tail = expected_rng.random(8) < 0.4
        expected[tail] *= 3.0
        actual = executor._slippage_rates(8)
    assert np.array_equal(actual.view(np.uint64), expected.view(np.uint64))
    assert executor._rng.bit_generator.state == expected_rng.bit_generator.state


def test_tail_slippage_recovers_finite_cost_from_unrepresentable_intermediate() -> None:
    """Large permitted tail scaling can make a tiny sampled rate representable."""
    tiny = float(np.nextafter(0.0, 1.0))
    executor = MarketExecutor(
        _dataset(price=1.0, multiplier=1.0),
        replace(
            ExecutionCostConfig.zero(),
            slippage_std=tiny,
            tail_slippage_probability=1.0,
            tail_slippage_multiplier=1e308,
            random_seed=0,
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
    expected = abs(np.random.default_rng(0).standard_normal(1)[0]) * (tiny * 1e308)
    assert expected > 0.0
    assert result.fill_count == 1
    assert result.cost_amount == pytest.approx(expected, rel=1e-12, abs=0.0)
    assert book.total_cost == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_true_zero_standard_normal_draw_remains_valid_for_charged_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor = MarketExecutor(
        _dataset(price=1.0, multiplier=1.0),
        replace(ExecutionCostConfig.zero(), slippage_std=0.005),
    )
    monkeypatch.setattr(
        executor, "_rng", SimpleNamespace(standard_normal=lambda size: np.zeros(size))
    )
    actual = executor._slippage_rates(1, chargeable_mask=np.ones(1, dtype=np.bool_))
    assert actual.tolist() == [0.0]


def test_stateful_tail_slippage_recovers_finite_cost_after_intermediate_zero() -> None:
    tiny = float(np.nextafter(0.0, 1.0))
    executor = MarketExecutor(
        _dataset(price=1.0, multiplier=1.0),
        replace(
            ExecutionCostConfig.zero(),
            slippage_std=tiny,
            tail_slippage_probability=1.0,
            tail_slippage_multiplier=1e308,
            random_seed=0,
        ),
    )
    runtime = SimpleNamespace(executor=executor)
    order = SimpleNamespace(
        intent=SimpleNamespace(
            symbol_index=0, order_type=OrderType.LIMIT, eligible_index=0
        )
    )
    actual = _execution_cost(
        runtime,
        order,
        trigger=SimpleNamespace(segment=TriggerSegment.FIRST_EXTREME),
        processing_index=1,
        filled_notional=1.0,
        participation_rate=1.0,
    )
    expected = abs(np.random.default_rng(0).standard_normal(1)[0]) * (tiny * 1e308)
    assert actual == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_legacy_rejects_lost_half_spread_despite_positive_fee() -> None:
    tiny = float(np.nextafter(0.0, 1.0))
    executor = MarketExecutor(
        _dataset(price=1.0, multiplier=1.0),
        replace(
            ExecutionCostConfig.zero(),
            fee_rate=0.001,
            spread_rate=tiny,
            order_type="limit",
            limit_offset_rate=0.0,
        ),
    )
    book = BookState.zero(1, 1_000.0, np.array([1.0]))
    with pytest.raises(ValueError, match="execution cost rate.*represented"):
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
    assert book.total_cost == 0.0


def test_stateful_rejects_lost_spread_or_impact_despite_positive_fee() -> None:
    tiny = float(np.nextafter(0.0, 1.0))
    order = SimpleNamespace(
        intent=SimpleNamespace(
            symbol_index=0, order_type=OrderType.LIMIT, eligible_index=0
        )
    )
    for cost, participation in (
        (replace(ExecutionCostConfig.zero(), fee_rate=0.001, spread_rate=tiny), 0.5),
        (replace(ExecutionCostConfig.zero(), fee_rate=0.001, impact_rate=tiny), 0.1),
    ):
        executor = MarketExecutor(_dataset(price=1.0, multiplier=1.0), cost)
        with pytest.raises(ValueError, match="execution cost rate.*represented"):
            _execution_cost(
                SimpleNamespace(executor=executor),
                order,
                trigger=SimpleNamespace(segment=TriggerSegment.FIRST_EXTREME),
                processing_index=1,
                filled_notional=1.0,
                participation_rate=participation,
            )


def test_depth_refuses_nonzero_fill_with_unrepresentable_notional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The notional itself must not collapse to zero before fee validation."""
    book = BookState.zero(1, 1_000.0, np.array([1e-200]), np.array([1e-200]))
    reached_accounting = False

    def unexpected_fill(self: BookState, **kwargs: object) -> None:
        nonlocal reached_accounting
        reached_accounting = True

    monkeypatch.setattr(BookState, "execute_fill", unexpected_fill)
    now = datetime(2026, 11, 6, tzinfo=UTC)
    with pytest.raises(ValueError, match="execution notional.*represented"):
        execute_depth_order(
            book=book,
            symbol_index=0,
            lot_count=1,
            rules=DepthOrderRules(1.0, 0.0, 10.0, 0.0, None),
            bids=((9e-201, 10.0),),
            asks=((1e-200, 10.0),),
            decision_at=now,
            quote_at=now + timedelta(seconds=1),
            execution_at=now + timedelta(seconds=2),
            fee_bps=10.0,
            adverse_bps=0.0,
            depth_fraction=1.0,
        )
    assert not reached_accounting
    assert book.quantities.tolist() == [0.0]
    assert book.total_cost == 0.0


@pytest.mark.parametrize("cost_multiplier", (0.0, 1.0))
def test_stateful_strict_numpy_venue_fee_overflow_uses_cost_boundary(
    cost_multiplier: float,
) -> None:
    """Strict NumPy must not leak raw FloatingPointError on inactive costs."""
    dataset = replace(
        _dataset(price=1.0, multiplier=1.0),
        maker_fee_rate=np.full((4, 1), 1e308),
    )
    executor = MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            maker_fee_rate=1e308,
            multiplier=cost_multiplier,
        ),
    )
    runtime = SimpleNamespace(executor=executor)
    order = SimpleNamespace(
        intent=SimpleNamespace(
            symbol_index=0, order_type=OrderType.LIMIT, eligible_index=0
        )
    )
    with np.errstate(over="raise", invalid="raise"):
        if cost_multiplier == 0.0:
            assert (
                _execution_cost(
                    runtime,
                    order,
                    trigger=SimpleNamespace(segment=TriggerSegment.FIRST_EXTREME),
                    processing_index=1,
                    filled_notional=1.0,
                    participation_rate=0.5,
                )
                == 0.0
            )
        else:
            with pytest.raises(ValueError, match="non-finite"):
                _execution_cost(
                    runtime,
                    order,
                    trigger=SimpleNamespace(segment=TriggerSegment.FIRST_EXTREME),
                    processing_index=1,
                    filled_notional=1.0,
                    participation_rate=0.5,
                )


@pytest.mark.parametrize(
    ("impact_rate", "cost_multiplier", "must_reject"),
    ((1e100, 1.0, True), (0.0, 1.0, False), (1e100, 0.0, False)),
)
def test_legacy_impact_participation_underflow_is_never_free_or_raw_numpy_error(
    impact_rate: float, cost_multiplier: float, must_reject: bool
) -> None:
    """Tiny real fills against huge positive capacity cannot erase impact."""
    dataset = replace(
        _dataset(price=1.0, multiplier=1.0), volume=np.full((4, 1), 1e130)
    )
    executor = MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            impact_rate=impact_rate,
            multiplier=cost_multiplier,
            max_participation_rate=1.0,
        ),
    )
    book = BookState.zero(1, 1000.0, np.ones(1))

    def fill() -> object:
        return executor._fill_toward_quantities(
            book,
            np.array([1e-200]),
            prices=np.ones(1),
            capacity_volume=np.array([1e130]),
            tradable=np.array([True]),
            turnover_denominator=1000.0,
            market_index=1,
        )

    with np.errstate(over="raise", under="raise", invalid="raise"):
        if must_reject:
            with pytest.raises(ValueError, match="execution cost rate.*represented"):
                fill()
        else:
            result = fill()
            assert result.filled_notional == 1e-200
            assert book.quantities.tolist() == [1e-200]
            assert result.cost_amount == 0.0

    if must_reject:
        assert book.quantities.tolist() == [0.0]
        assert book.total_cost == 0.0


def test_legacy_impact_keeps_representable_subnormal_participation() -> None:
    """Reject only a lost ratio, not a valid tiny positive participation."""
    dataset = replace(
        _dataset(price=1.0, multiplier=1.0), volume=np.full((4, 1), 1e120)
    )
    executor = MarketExecutor(
        dataset,
        replace(
            ExecutionCostConfig.zero(),
            impact_rate=1e100,
            max_participation_rate=1.0,
        ),
    )
    book = BookState.zero(1, 1000.0, np.ones(1))
    with np.errstate(over="raise", under="raise", invalid="raise"):
        result = executor._fill_toward_quantities(
            book,
            np.array([1e-200]),
            prices=np.ones(1),
            capacity_volume=np.array([1e120]),
            tradable=np.array([True]),
            turnover_denominator=1000.0,
            market_index=1,
        )
    expected = 1e-200 * (1e100 * np.sqrt(1e-320))
    assert result.max_participation > 0.0
    assert result.filled_notional == 1e-200
    assert result.cost_amount == pytest.approx(expected, rel=1e-13)
    assert book.total_cost == pytest.approx(expected, rel=1e-13)
