from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.simulation import execution as execution_module
from trade_rl.simulation.accounting import BookState, EconomicTerminationReason
from trade_rl.simulation.execution import ExecutionCostConfig, MarketExecutor


def market(n_symbols: int = 1, **overrides: object) -> MarketDataset:
    n_bars = 5
    shape = (n_bars, n_symbols)
    close = np.full(shape, 100.0)
    values: dict[str, object] = {
        "dataset_id": "b" * 64,
        "symbols": tuple(f"S{index}" for index in range(n_symbols)),
        "timestamps": np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        "features": np.zeros((n_bars, n_symbols, 1), dtype=np.float32),
        "global_features": np.zeros((n_bars, 1), dtype=np.float32),
        "open": close.copy(),
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": np.full(shape, 1_000_000.0),
        "funding_rate": np.zeros(shape),
        "tradable": np.ones(shape, dtype=np.bool_),
        "feature_available": np.ones((n_bars, n_symbols, 1), dtype=np.bool_),
        "feature_names": ("ret",),
        "global_feature_names": ("regime",),
        "periods_per_year": 8_760,
    }
    values.update(overrides)
    return MarketDataset(**values)


@pytest.mark.parametrize(
    "override",
    [
        {"fee_rate": math.inf},
        {"fee_rate": -1.0},
        {"max_participation_rate": 0.0},
        {"tail_slippage_probability": 1.1},
        {"max_leverage": 0.0},
        {"maintenance_margin_rate": 1.1},
        {"collateral_haircut": 0.0},
        {"margin_mode": "portfolio"},
        {"random_seed": True},
        {"random_seed": -1},
        {"order_latency_bars": True},
        {"order_latency_bars": -1},
        {"order_type": "stop"},
        {"limit_offset_rate": 1.0},
        {"allow_short": 1},
    ],
)
def test_execution_cost_config_rejects_every_invalid_contract(
    override: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        ExecutionCostConfig(**override)


def test_accounting_explicit_multiplier_and_termination_paths() -> None:
    book = BookState(
        quantities=np.array([1.0]),
        cash=0.0,
        mark_prices=np.array([100.0]),
        peak_value=200.0,
        contract_multipliers=np.array([2.0]),
        insolvent=True,
        termination_reason="liquidation",
    )
    assert book.position_values.tolist() == [200.0]
    assert book.termination_reason is EconomicTerminationReason.LIQUIDATION

    weighted = BookState.from_weights(
        weights=np.array([0.5]),
        capital=1_000.0,
        prices=np.array([100.0]),
        contract_multipliers=np.array([5.0]),
    )
    assert weighted.quantities.tolist() == pytest.approx([1.0])


def test_target_weight_and_random_seed_validation() -> None:
    with pytest.raises(ValueError, match="maximum_gross"):
        execution_module._target_weights(np.array([0.0]), n_symbols=1, maximum_gross=0)
    with pytest.raises(ValueError, match="shape"):
        execution_module._target_weights(
            np.array([0.0, 0.0]), n_symbols=1, maximum_gross=1
        )
    with pytest.raises(ValueError, match="finite"):
        execution_module._target_weights(
            np.array([np.nan]), n_symbols=1, maximum_gross=1
        )
    with pytest.raises(ValueError, match="gross"):
        execution_module._target_weights(np.array([2.0]), n_symbols=1, maximum_gross=1)

    executor = MarketExecutor(market(), ExecutionCostConfig.zero())
    executor.reset_random_state()
    with pytest.raises(ValueError, match="seed"):
        executor.reset_random_state(True)
    with pytest.raises(ValueError, match="seed"):
        executor.reset_random_state(-1)


def test_execution_unit_rounding_capacity_and_borrow_branches() -> None:
    tick = np.full((5, 1), 0.5)
    base = MarketExecutor(
        market(tick_size=tick),
        ExecutionCostConfig.zero(),
    )
    rounded = base._round_prices(np.array([100.24]), index=1)
    assert rounded.tolist() == pytest.approx([100.0])
    assert base._capacity_notional(np.array([100.0]), np.array([2.0])).tolist() == [
        200.0
    ]

    quote = MarketExecutor(
        market(volume_units=(VolumeUnit.QUOTE_NOTIONAL,)),
        ExecutionCostConfig.zero(),
    )
    assert quote._capacity_notional(
        np.array([100.0]), np.array([2.0])
    ).tolist() == [2.0]

    no_short = MarketExecutor(market(), ExecutionCostConfig(allow_short=False))
    constrained = no_short._constrain_borrow(
        np.array([-1.0]), current=np.array([0.0]), index=1
    )
    assert constrained.tolist() == [0.0]


def test_fill_validation_and_nonfinite_cost_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = market()
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState.zero(1, 1_000.0, dataset.close[0])
    common = {
        "prices": np.array([100.0]),
        "capacity_volume": np.array([1_000.0]),
        "tradable": np.array([True]),
        "turnover_denominator": 1_000.0,
        "market_index": 1,
    }

    with pytest.raises(ValueError, match="vectors"):
        executor._fill_toward_quantities(book, np.array([1.0, 2.0]), **common)
    with pytest.raises(ValueError, match="finite"):
        executor._fill_toward_quantities(book, np.array([np.nan]), **common)
    with pytest.raises(ValueError, match="positive|invalid"):
        executor._fill_toward_quantities(
            book,
            np.array([1.0]),
            **{**common, "prices": np.array([-1.0])},
        )
    with pytest.raises(ValueError, match="invalid"):
        executor._fill_toward_quantities(
            book,
            np.array([1.0]),
            **{**common, "capacity_volume": np.array([-1.0])},
        )
    with pytest.raises(ValueError, match="denominator"):
        executor._fill_toward_quantities(
            book,
            np.array([1.0]),
            **{**common, "turnover_denominator": 0.0},
        )

    costly = MarketExecutor(
        dataset,
        ExecutionCostConfig(
            fee_rate=0.0,
            spread_rate=0.0,
            impact_rate=0.0,
            max_participation_rate=1.0,
            slippage_std=1.0,
        ),
    )
    monkeypatch.setattr(
        costly,
        "_slippage_rates",
        lambda size: np.full(size, np.inf),
    )
    with pytest.raises(ValueError, match="non-finite"):
        costly._fill_toward_quantities(
            BookState.zero(1, 1_000.0, dataset.close[0]),
            np.array([10.0]),
            **common,
        )


def test_cross_and_isolated_margin_calls_flatten_positions() -> None:
    for margin_mode in ("cross", "isolated"):
        executor = MarketExecutor(
            market(),
            ExecutionCostConfig(
                fee_rate=0.0,
                spread_rate=0.0,
                impact_rate=0.0,
                max_participation_rate=1.0,
                max_leverage=1.0,
                maintenance_margin_rate=1.0,
                collateral_haircut=0.1,
                margin_mode=margin_mode,
            ),
        )
        book = BookState.from_weights(
            weights=np.array([1.0]),
            capital=100.0,
            prices=np.array([100.0]),
        )
        executor._update_margin(book)
        assert book.insolvent
        assert book.termination_reason is EconomicTerminationReason.MARGIN_CALL
        assert book.quantities.tolist() == [0.0]


def test_borrow_carry_charges_only_short_positions() -> None:
    shape = (5, 1)
    dataset = market(borrow_rate=np.ones(shape))
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    short = BookState.from_weights(
        weights=np.array([-1.0]),
        capital=1_000.0,
        prices=np.array([100.0]),
    )
    _, charged = executor._charge_carry(short, index=1)
    assert charged > 0.0

    long = BookState.from_weights(
        weights=np.array([1.0]),
        capital=1_000.0,
        prices=np.array([100.0]),
    )
    _, uncharged = executor._charge_carry(long, index=1)
    assert uncharged == 0.0


def test_execute_interval_rejects_invalid_boundaries_and_book_identity() -> None:
    dataset = market()
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    book = BookState.zero(1, 1_000.0, dataset.close[0])

    with pytest.raises(ValueError, match="bars"):
        executor.execute_interval(book, np.array([0.0]), start_index=0, bars=0)
    with pytest.raises(ValueError, match="outside"):
        executor.execute_interval(book, np.array([0.0]), start_index=4, bars=1)
    with pytest.raises(ValueError, match="weights shape"):
        executor.execute_interval(
            BookState.zero(2, 1_000.0, np.array([100.0, 100.0])),
            np.array([0.0]),
            start_index=0,
            bars=1,
        )
    with pytest.raises(ValueError, match="multipliers"):
        executor.execute_interval(
            BookState.zero(
                1,
                1_000.0,
                dataset.close[0],
                contract_multipliers=np.array([2.0]),
            ),
            np.array([0.0]),
            start_index=0,
            bars=1,
        )
    with pytest.raises(ValueError, match="positive starting equity"):
        executor.execute_interval(
            BookState(
                quantities=np.array([0.0]),
                cash=0.0,
                mark_prices=np.array([100.0]),
                peak_value=1.0,
            ),
            np.array([0.0]),
            start_index=0,
            bars=1,
        )


def test_inactive_assets_settle_and_zero_target_has_identity_fill_ratio() -> None:
    active = np.ones((5, 1), dtype=np.bool_)
    active[1, 0] = False
    dataset = market(asset_active=active)
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    invested = BookState.from_weights(
        weights=np.array([1.0]),
        capital=1_000.0,
        prices=dataset.close[0],
    )
    settled = executor.execute_interval(
        invested,
        np.array([0.0]),
        start_index=0,
        bars=1,
    )
    assert settled.book.quantities.tolist() == [0.0]

    identity = executor.execute_interval(
        BookState.zero(1, 1_000.0, dataset.close[0]),
        np.array([0.0]),
        start_index=1,
        bars=1,
    )
    assert identity.fill_ratio == 1.0


def test_liquidation_rejects_out_of_range_index() -> None:
    dataset = market()
    executor = MarketExecutor(dataset, ExecutionCostConfig.zero())
    with pytest.raises(ValueError, match="outside"):
        executor.liquidate_at_close(
            BookState.zero(1, 1_000.0, dataset.close[0]),
            index=dataset.n_bars,
        )
