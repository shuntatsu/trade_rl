from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from trade_rl.data.artifacts import MarketDatasetView
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.trend import TrendConfig, TrendStrategy
from trade_rl.workflows.causal_scenario.conditions import CausalConditionConfig
from trade_rl.workflows.causal_scenario.library import (
    CausalScenarioLibraryConfig,
    FrozenCausalScenarioLibrary,
    RelativeScenarioBlock,
    build_causal_scenario_library,
    select_causal_scenarios,
)


def _copy_dataset(dataset: MarketDataset, **updates: Any) -> MarketDataset:
    values = {
        name: getattr(dataset, name)
        for name, field in dataset.__dataclass_fields__.items()
        if field.init and not name.startswith("_")
    }
    values["identity_payload_json"] = None
    values.update(updates)
    return MarketDataset(**values)


def _strategy() -> TrendStrategy:
    return TrendStrategy(TrendConfig(fast_hours=12, base_hours=48, slow_hours=96))


def test_builder_uses_only_complete_train_blocks(market_dataset_factory: Any) -> None:
    dataset = market_dataset_factory(n_bars=1_100)
    train = MarketDatasetView(dataset, 0, 960)
    config = CausalScenarioLibraryConfig()
    library = build_causal_scenario_library(train, _strategy(), config)
    assert library.anchor_count >= config.scenario_count
    anchors = library.anchor_indices
    np.testing.assert_array_equal(anchors, np.sort(anchors))
    assert np.all(anchors >= train.start)
    assert np.all(anchors + 1 + config.horizon_decisions <= train.stop)
    assert library.raw_conditions.shape == (
        library.anchor_count,
        len(library.layout.feature_names),
    )
    assert library.normalized_conditions.shape == library.raw_conditions.shape
    assert anchors.flags.writeable is False
    assert library.raw_conditions.flags.writeable is False
    assert library.normalized_conditions.flags.writeable is False


def test_builder_is_invariant_to_rows_outside_train(
    market_dataset_factory: Any,
) -> None:
    dataset = market_dataset_factory(n_bars=1_100)
    train = MarketDatasetView(dataset, 0, 960)
    first = build_causal_scenario_library(train, _strategy())

    suffix = slice(train.stop, None)
    close = dataset.close.copy()
    open_price = dataset.open.copy()
    high = dataset.high.copy()
    low = dataset.low.copy()
    mark = dataset.mark_price.copy()
    index = dataset.index_price.copy()
    for array in (close, open_price, high, low, mark, index):
        array[suffix] *= 7.0
    funding = dataset.funding_rate.copy()
    funding[suffix] = 0.9
    changed = _copy_dataset(
        dataset,
        close=close,
        open=open_price,
        high=high,
        low=low,
        mark_price=mark,
        index_price=index,
        funding_rate=funding,
    )
    second = build_causal_scenario_library(
        MarketDatasetView(changed, train.start, train.stop), _strategy()
    )
    assert first.library_digest == second.library_digest
    np.testing.assert_array_equal(first.anchor_indices, second.anchor_indices)
    np.testing.assert_array_equal(first.raw_conditions, second.raw_conditions)
    np.testing.assert_array_equal(
        first.normalized_conditions, second.normalized_conditions
    )


def test_builder_does_not_read_before_nonzero_train_start(
    market_dataset_factory: Any,
) -> None:
    dataset = market_dataset_factory(n_bars=1_700)
    train = MarketDatasetView(dataset, 300, 1_500)
    first = build_causal_scenario_library(train, _strategy())

    prefix = slice(0, train.start)
    close = dataset.close.copy()
    open_price = dataset.open.copy()
    high = dataset.high.copy()
    low = dataset.low.copy()
    mark = dataset.mark_price.copy()
    index = dataset.index_price.copy()
    volume = dataset.volume.copy()
    features = dataset.features.copy()
    global_features = dataset.global_features.copy()
    funding = dataset.funding_rate.copy()
    for array in (close, open_price, high, low, mark, index):
        array[prefix] *= 9.0
    volume[prefix] *= 13.0
    features[prefix] = 77.0
    global_features[prefix] = -55.0
    funding[prefix] = 0.7
    changed = _copy_dataset(
        dataset,
        close=close,
        open=open_price,
        high=high,
        low=low,
        mark_price=mark,
        index_price=index,
        volume=volume,
        features=features,
        global_features=global_features,
        funding_rate=funding,
    )
    second = build_causal_scenario_library(
        MarketDatasetView(changed, train.start, train.stop), _strategy()
    )

    assert first.anchor_indices[0] >= train.start + 672
    assert first.library_digest == second.library_digest
    np.testing.assert_array_equal(first.anchor_indices, second.anchor_indices)
    np.testing.assert_array_equal(first.raw_conditions, second.raw_conditions)
    np.testing.assert_array_equal(
        first.normalized_conditions, second.normalized_conditions
    )


def test_relative_block_reconstructs_source_ratios(market_dataset_factory: Any) -> None:
    dataset = market_dataset_factory(n_bars=1_000)
    library = build_causal_scenario_library(
        MarketDatasetView(dataset, 0, 930), _strategy()
    )
    query_index = 950
    selection = select_causal_scenarios(
        library,
        MarketDatasetView(dataset, 0, query_index + 1),
        query_index,
        _strategy(),
    )
    block = selection.blocks[0]
    rows = slice(block.source_start, block.source_stop)
    anchor = block.anchor_index
    np.testing.assert_allclose(
        block.price_relatives["close"], dataset.close[rows] / dataset.close[anchor]
    )
    np.testing.assert_allclose(
        block.price_relatives["open"], dataset.open[rows] / dataset.close[anchor]
    )
    np.testing.assert_allclose(
        block.volume_relative, dataset.volume[rows] / dataset.volume[anchor]
    )
    expected_notional = np.vstack(
        [
            dataset.market_notional(index, prices=dataset.close[index])
            for index in range(block.source_start, block.source_stop)
        ]
    ) / dataset.market_notional(anchor, prices=dataset.close[anchor])
    np.testing.assert_allclose(block.market_notional_relative, expected_notional)


def test_selection_is_deterministic_uniform_and_query_past_only(
    market_dataset_factory: Any,
) -> None:
    dataset = market_dataset_factory(n_bars=1_250)
    train = MarketDatasetView(dataset, 0, 960)
    library = build_causal_scenario_library(train, _strategy())
    query_index = 1_100
    prefix = MarketDatasetView(dataset, 0, query_index + 1)
    selection = select_causal_scenarios(library, prefix, query_index, _strategy())
    assert len(selection.blocks) == library.config.scenario_count == 64
    assert selection.scenario_set.scenario_count == 64
    np.testing.assert_allclose(selection.scenario_set.probabilities, 1.0 / 64.0)
    assert all(block.source_stop <= query_index for block in selection.blocks)
    ordering = sorted(
        zip(
            selection.scenario_set.distances,
            selection.scenario_set.anchor_indices,
            strict=True,
        )
    )
    assert (
        list(
            zip(
                selection.scenario_set.distances,
                selection.scenario_set.anchor_indices,
                strict=True,
            )
        )
        == ordering
    )
    repeated = select_causal_scenarios(library, prefix, query_index, _strategy())
    assert selection.selection_digest == repeated.selection_digest


def test_train_query_applies_horizon_embargo(market_dataset_factory: Any) -> None:
    dataset = market_dataset_factory(n_bars=1_100)
    library = build_causal_scenario_library(
        MarketDatasetView(dataset, 0, 1_020), _strategy()
    )
    query_index = 930
    selection = select_causal_scenarios(
        library,
        MarketDatasetView(dataset, 0, query_index + 1),
        query_index,
        _strategy(),
    )
    assert all(
        block.source_stop <= query_index - library.config.horizon_decisions
        for block in selection.blocks
    )


def test_selection_rejects_extended_prefix_and_insufficient_candidates(
    market_dataset_factory: Any,
) -> None:
    dataset = market_dataset_factory(n_bars=1_100)
    library = build_causal_scenario_library(
        MarketDatasetView(dataset, 0, 960), _strategy()
    )
    with pytest.raises(ValueError, match="causal prefix"):
        select_causal_scenarios(
            library,
            MarketDatasetView(dataset, 0, 1_001),
            900,
            _strategy(),
        )
    with pytest.raises(ValueError, match="64 eligible"):
        select_causal_scenarios(
            library,
            MarketDatasetView(dataset, 0, 801),
            800,
            _strategy(),
        )


def test_library_contracts_reject_invalid_values() -> None:
    with pytest.raises(ValueError):
        CausalScenarioLibraryConfig(horizon_decisions=0)
    with pytest.raises(ValueError):
        CausalScenarioLibraryConfig(scenario_count=63)
    with pytest.raises(ValueError):
        RelativeScenarioBlock(
            anchor_index=10,
            source_start=12,
            source_stop=13,
            elapsed_ns=np.asarray([1]),
            raw_condition=np.asarray([0.0]),
            normalized_condition=np.asarray([0.0]),
            price_relatives={},
            volume_relative=np.ones((1, 1)),
            market_notional_relative=np.ones((1, 1)),
            future_arrays={},
        )


def test_library_digest_detects_mutation(market_dataset_factory: Any) -> None:
    dataset = market_dataset_factory(n_bars=1_000)
    library = build_causal_scenario_library(
        MarketDatasetView(dataset, 0, 930), _strategy()
    )
    with pytest.raises(ValueError, match="library_digest"):
        replace(library, library_digest="f" * 64)
    assert isinstance(library, FrozenCausalScenarioLibrary)


def test_first_anchor_supports_legacy_lookbacks_and_rejects_short_ranges(
    market_dataset_factory: Any,
) -> None:
    from trade_rl.workflows.causal_scenario import library as library_module

    dataset = market_dataset_factory(n_bars=400)
    config = CausalScenarioLibraryConfig(
        horizon_decisions=2,
        condition=CausalConditionConfig(
            volatility_hours=1,
            correlation_hours=2,
        ),
    )
    legacy = TrendStrategy(
        TrendConfig(
            fast_hours=12,
            base_hours=48,
            slow_hours=96,
            fast_lookback=4,
            base_lookback=8,
            slow_lookback=12,
        )
    )
    view = MarketDatasetView(dataset, 100, 300)
    assert library_module._first_train_anchor(view, legacy, config) == 112
    with pytest.raises(ValueError, match="too short"):
        library_module._minimum_history_index_from(
            dataset,
            start=dataset.n_bars - 2,
            hours=168,
        )


def test_library_stores_compact_anchor_index_not_overlapping_future_blocks(
    market_dataset_factory: Any,
) -> None:
    dataset = market_dataset_factory(n_bars=1_100)
    library = build_causal_scenario_library(
        MarketDatasetView(dataset, 0, 960), _strategy()
    )
    assert not hasattr(library, "blocks")
    assert library.anchor_indices.ndim == 1
    assert library.raw_conditions.shape == library.normalized_conditions.shape
    assert library.raw_conditions.shape[0] == library.anchor_indices.size
    assert library.anchor_indices.flags.writeable is False
    assert library.raw_conditions.flags.writeable is False
    assert library.normalized_conditions.flags.writeable is False
    compact_bytes = (
        library.anchor_indices.nbytes
        + library.raw_conditions.nbytes
        + library.normalized_conditions.nbytes
    )
    assert compact_bytes < 1_000_000
