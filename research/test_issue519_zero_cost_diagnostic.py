from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import run_single_symbol_replay
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent

MODULE = "research.issue519_zero_cost_diagnostic"


def _module():
    spec = find_spec(MODULE)
    assert spec is not None, "zero-cost diagnostic helper is not implemented"
    return import_module(MODULE)


def _dataset() -> MarketDataset:
    n = 8
    timestamps = np.arange(n, dtype=np.int64).astype("datetime64[h]")
    close = np.asarray([100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0])[:, None]
    open_price = close.copy()
    high = close * 1.01
    low = close * 0.99
    features = np.zeros((n, 1, 1), dtype=np.float32)
    feature_available = np.ones_like(features, dtype=np.bool_)
    shape = (n, 1)
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=np.full(shape, 1_000_000.0),
        funding_rate=np.full(shape, 0.001),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=feature_available,
        feature_names=("signal",),
        global_feature_names=("global",),
        periods_per_year=8760,
        fee_rate=np.full(shape, 0.0005),
        maker_fee_rate=np.full(shape, 0.0001),
        taker_fee_rate=np.full(shape, 0.0002),
        spread_rate=np.full(shape, 0.0002),
        max_participation_rate=np.full(shape, 0.05),
        minimum_notional=np.full(shape, 10.0),
        lot_size=np.full(shape, 0.001),
        tick_size=np.full(shape, 0.01),
        borrow_rate=np.full(shape, 0.0003),
        cash_rate=np.full(n, 0.0001),
    )
    return dataset.with_content_identity({"fixture": "zero-cost-diagnostic"})


def _next_open_timing_dataset() -> MarketDataset:
    timestamps = np.arange(3, dtype=np.int64).astype("datetime64[h]")
    open_price = np.asarray([100.0, 100.0, 121.0], dtype=np.float64)[:, None]
    close = np.asarray([100.0, 110.0, 108.9], dtype=np.float64)[:, None]
    high = np.maximum(open_price, close)
    low = np.minimum(open_price, close)
    shape = (3, 1)
    dataset = MarketDataset(
        dataset_id="0" * 64,
        symbols=("BTCUSDT",),
        timestamps=timestamps,
        features=np.zeros((3, 1, 1), dtype=np.float32),
        global_features=np.zeros((3, 1), dtype=np.float32),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=np.full(shape, 1_000_000.0),
        funding_rate=np.zeros(shape),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones((3, 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("global",),
        periods_per_year=8760,
        fee_rate=np.zeros(shape),
        maker_fee_rate=np.zeros(shape),
        taker_fee_rate=np.zeros(shape),
        spread_rate=np.zeros(shape),
        max_participation_rate=np.ones(shape),
        minimum_notional=np.zeros(shape),
        lot_size=np.zeros(shape),
        tick_size=np.zeros(shape),
        borrow_rate=np.zeros(shape),
        cash_rate=np.zeros(3),
    )
    return dataset.with_content_identity({"fixture": "next-open-intent-attribution"})


class _LongThenShort:
    def decide(self, observation: StrategyObservation) -> PositionIntent:
        if observation.index == 0:
            return PositionIntent.LONG
        if observation.index == 1:
            return PositionIntent.SHORT
        return PositionIntent.FLAT


def test_zero_cost_dataset_changes_only_economic_cost_arrays() -> None:
    module = _module()
    source = _dataset()
    diagnostic = module.make_zero_cost_diagnostic_dataset(source)

    assert diagnostic.dataset_id != source.dataset_id
    assert diagnostic.identity_verified
    assert source.identity_verified
    for field in (
        "fee_rate",
        "maker_fee_rate",
        "taker_fee_rate",
        "spread_rate",
        "funding_rate",
        "borrow_rate",
        "cash_rate",
    ):
        assert np.all(diagnostic.resolved_array(field) == 0.0)
        assert np.any(source.resolved_array(field) != 0.0)

    for field in (
        "timestamps",
        "features",
        "feature_available",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "tradable",
        "max_participation_rate",
        "minimum_notional",
        "lot_size",
        "tick_size",
        "borrow_available",
        "buy_allowed",
        "sell_allowed",
        "mark_price",
        "index_price",
        "contract_multipliers",
    ):
        assert np.array_equal(diagnostic.resolved_array(field), source.resolved_array(field))


def _assert_direction_attribution(result: dict[str, object], returns: np.ndarray) -> None:
    expected = float(np.log1p(returns).sum())
    assert np.isclose(
        result["long_log_return_contribution"]
        + result["short_log_return_contribution"]
        + result["flat_log_return_contribution"],
        expected,
    )
    assert result["long_intervals"] == 2
    assert result["short_intervals"] == 1
    assert result["flat_intervals"] == 1


def test_direction_attribution_reconstructs_total_log_return() -> None:
    module = _module()
    returns = np.asarray([0.10, -0.05, 0.02, -0.01], dtype=np.float64)
    result = module.direction_log_return_attribution(
        returns,
        ["LONG", "SHORT", "LONG", "FLAT"],
    )
    _assert_direction_attribution(result, returns)


def test_direction_attribution_accepts_actual_position_intent_enum() -> None:
    module = _module()
    returns = np.asarray([0.10, -0.05, 0.02, -0.01], dtype=np.float64)
    result = module.direction_log_return_attribution(
        returns,
        [
            PositionIntent.LONG,
            PositionIntent.SHORT,
            PositionIntent.LONG,
            PositionIntent.FLAT,
        ],
    )
    _assert_direction_attribution(result, returns)


def test_direction_attribution_rejects_length_mismatch() -> None:
    module = _module()
    returns = np.asarray([0.01, 0.02], dtype=np.float64)
    try:
        module.direction_log_return_attribution(returns, ["LONG"])
    except ValueError as exc:
        assert "length" in str(exc)
    else:
        raise AssertionError("length mismatch must fail")


def test_segmented_replay_matches_canonical_and_assigns_gap_to_prior_intent() -> None:
    module = _module()
    dataset = _next_open_timing_dataset()
    canonical = run_single_symbol_replay(
        dataset,
        _LongThenShort(),
        symbol_index=0,
        start_index=0,
        stop_index=2,
        gross_budget=1.0,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )
    segmented = module.run_segmented_zero_cost_replay(
        dataset,
        _LongThenShort(),
        symbol_index=0,
        start_index=0,
        stop_index=2,
        gross_budget=1.0,
        initial_capital=1_000.0,
    )

    assert np.array_equal(
        np.asarray(segmented["returns"]),
        np.asarray(canonical.returns.values),
    )
    assert segmented["final_portfolio_value"] == canonical.book.portfolio_value
    attribution = segmented["exact_intent_log_attribution"]
    assert np.isclose(attribution["long_log_return_contribution"], np.log(1.21))
    assert np.isclose(attribution["short_log_return_contribution"], np.log(1.10))
    assert np.isclose(attribution["flat_log_return_contribution"], 0.0)
    assert np.isclose(
        attribution["total_log_return"],
        np.log(canonical.book.portfolio_value / 1_000.0),
    )
    assert attribution["gap_segments"] == 2
    assert attribution["post_open_segments"] == 2
