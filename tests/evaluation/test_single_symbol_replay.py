from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import trade_rl.evaluation as evaluation
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.position_intent import PositionIntent


@dataclass
class AlwaysLong:
    observations: list[object] = field(default_factory=list)

    def decide(self, observation: object) -> PositionIntent:
        self.observations.append(observation)
        return PositionIntent.LONG


@dataclass
class AlwaysShort:
    observations: list[object] = field(default_factory=list)

    def decide(self, observation: object) -> PositionIntent:
        self.observations.append(observation)
        return PositionIntent.SHORT


def _rising_market() -> MarketDataset:
    close = np.asarray([[100.0], [100.0], [110.0], [120.0], [130.0], [140.0]])
    open_price = np.vstack((close[0], close[:-1]))
    n_bars = close.shape[0]
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.arange(n_bars, dtype=np.float32).reshape(n_bars, 1, 1),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close),
        low=np.minimum(open_price, close),
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_repeated_long_intent_holds_quantity_instead_of_rebalancing_weight() -> None:
    run_replay = getattr(evaluation, "run_single_symbol_replay", None)
    assert callable(run_replay), "lean single-symbol replay is not implemented"

    strategy = AlwaysLong()
    result = run_replay(
        _rising_market(),
        strategy,
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1_000.0,
    )

    assert result.book.fill_count == 1
    assert result.book.rebalance_events == 1
    assert result.book.quantities[0] == 5.0
    assert result.book.portfolio_value == 1_200.0
    assert len(result.returns.values) == 5
    assert len(result.decisions) == 5
    assert result.decisions[0].intent is PositionIntent.LONG
    assert result.decisions[0].changed_intent is True
    assert all(not item.changed_intent for item in result.decisions[1:])

    first_observation = strategy.observations[0]
    assert getattr(first_observation, "symbol") == "BTCUSDT"
    assert getattr(first_observation, "index") == 0
    features = getattr(first_observation, "features")
    assert features.flags.writeable is False


def test_adverse_short_drift_is_hard_deleveraged_instead_of_crashing() -> None:
    result = evaluation.run_single_symbol_replay(
        _rising_market(),
        AlwaysShort(),
        start_index=0,
        stop_index=5,
        gross_budget=1.0,
        initial_capital=1_000.0,
    )

    assert len(result.returns.values) == 5
    assert result.book.fill_count > 1
    assert result.book.quantities[0] > -10.0
    assert all(abs(decision.target_weight) <= 1.0 + 1e-10 for decision in result.decisions)
