from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from trade_rl.data.artifacts import MarketDatasetView
from trade_rl.strategies.trend import TrendConfig, TrendStrategy
from trade_rl.workflows.causal_scenario.library import (
    build_causal_scenario_library,
    select_causal_scenarios,
)
from trade_rl.workflows.causal_scenario.replay import (
    CausalScenarioReplayIdentity,
    materialize_causal_scenario_dataset,
)


def _strategy() -> TrendStrategy:
    return TrendStrategy(TrendConfig(fast_hours=12, base_hours=48, slow_hours=96))


def _selection(dataset: Any) -> tuple[Any, Any]:
    library = build_causal_scenario_library(
        MarketDatasetView(dataset, 0, 960), _strategy()
    )
    query_index = 1_100
    selection = select_causal_scenarios(
        library,
        MarketDatasetView(dataset, 0, query_index + 1),
        query_index,
        _strategy(),
    )
    return library, selection


def test_replay_reconstructs_query_anchored_relative_paths(
    market_dataset_factory: Any,
) -> None:
    dataset = market_dataset_factory(n_bars=1_250, seed_offset=500.0)
    library, selection = _selection(dataset)
    replay = materialize_causal_scenario_dataset(
        library,
        selection,
        MarketDatasetView(dataset, 0, selection.query_index + 1),
        selected_rank=0,
    )
    block = selection.blocks[0]
    query = selection.query_index
    assert replay.n_bars == library.config.horizon_decisions + 1
    np.testing.assert_array_equal(replay.timestamps[0], dataset.timestamps[query])
    np.testing.assert_allclose(replay.open[0], dataset.open[query])
    np.testing.assert_allclose(replay.close[0], dataset.close[query])
    np.testing.assert_allclose(
        replay.open[1:],
        dataset.close[query] * block.price_relatives["open"],
    )
    np.testing.assert_allclose(
        replay.close[1:],
        dataset.close[query] * block.price_relatives["close"],
    )
    np.testing.assert_allclose(
        replay.mark_price[1:],
        dataset.mark_price[query] * block.price_relatives["mark_price"],
    )
    np.testing.assert_allclose(
        replay.volume[1:], dataset.volume[query] * block.volume_relative
    )
    assert np.all(replay.high >= np.maximum(replay.open, replay.close))
    assert np.all(replay.low <= np.minimum(replay.open, replay.close))

    base_notional = replay.market_notional(0, prices=replay.close[0])
    replay_relatives = (
        np.vstack(
            [
                replay.market_notional(index, prices=replay.close[index])
                for index in range(1, replay.n_bars)
            ]
        )
        / base_notional
    )
    np.testing.assert_allclose(replay_relatives, block.market_notional_relative)


def test_replay_preserves_execution_lifecycle_and_information_delays(
    market_dataset_factory: Any,
) -> None:
    dataset = market_dataset_factory(n_bars=1_250)
    library, selection = _selection(dataset)
    replay = materialize_causal_scenario_dataset(
        library,
        selection,
        MarketDatasetView(dataset, 0, selection.query_index + 1),
        selected_rank=3,
    )
    block = selection.blocks[3]
    for field in (
        "funding_rate",
        "tradable",
        "fee_rate",
        "maker_fee_rate",
        "taker_fee_rate",
        "spread_rate",
        "borrow_available",
        "borrow_rate",
        "funding_due",
        "asset_active",
        "buy_allowed",
        "sell_allowed",
        "split_factor",
        "delisting_recovery",
        "information_available",
    ):
        np.testing.assert_array_equal(
            replay.resolved_array(field)[1:], block.future_arrays[field]
        )
    replay_delay = (
        replay.resolved_array("available_at")[1:]
        .astype("datetime64[ns]")
        .astype(np.int64)
        - replay.timestamps[1:].astype("datetime64[ns]").astype(np.int64)[:, None]
    )
    np.testing.assert_array_equal(
        replay_delay, block.future_arrays["availability_delay_ns"]
    )


def test_replay_is_deterministic_and_does_not_alias_sources(
    market_dataset_factory: Any,
) -> None:
    dataset = market_dataset_factory(n_bars=1_250)
    library, selection = _selection(dataset)
    prefix = MarketDatasetView(dataset, 0, selection.query_index + 1)
    first = materialize_causal_scenario_dataset(
        library, selection, prefix, selected_rank=1
    )
    second = materialize_causal_scenario_dataset(
        library, selection, prefix, selected_rank=1
    )
    assert first.dataset_id == second.dataset_id
    assert first.identity_payload_json == second.identity_payload_json
    np.testing.assert_array_equal(first.close, second.close)
    assert not np.shares_memory(first.close, dataset.close)
    assert not np.shares_memory(
        first.close[1:], selection.blocks[1].price_relatives["close"]
    )
    assert first.close.flags.writeable is False


def test_replay_rejects_mismatches_and_bad_rank(market_dataset_factory: Any) -> None:
    dataset = market_dataset_factory(n_bars=1_250)
    library, selection = _selection(dataset)
    prefix = MarketDatasetView(dataset, 0, selection.query_index + 1)
    with pytest.raises(ValueError, match="rank"):
        materialize_causal_scenario_dataset(
            library, selection, prefix, selected_rank=64
        )
    with pytest.raises(ValueError, match="causal prefix"):
        materialize_causal_scenario_dataset(
            library,
            selection,
            MarketDatasetView(dataset, 0, selection.query_index),
            selected_rank=0,
        )
    with pytest.raises(ValueError, match="library"):
        materialize_causal_scenario_dataset(
            replace(library, dataset_id="f" * 64),
            selection,
            prefix,
            selected_rank=0,
        )


def test_replay_identity_contract_validation() -> None:
    with pytest.raises(ValueError):
        CausalScenarioReplayIdentity(
            query_dataset_id="bad",
            library_digest="a" * 64,
            selection_digest="b" * 64,
            block_digest="c" * 64,
            scenario_id="d" * 64,
            query_index=1,
            selected_rank=0,
        )
