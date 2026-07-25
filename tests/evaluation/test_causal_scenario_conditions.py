from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from trade_rl.data.artifacts import MarketDatasetView
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.trend import TrendConfig, TrendStrategy
from trade_rl.workflows.causal_scenario.conditions import (
    CausalConditionConfig,
    TrainRobustConditionNormalizer,
    build_causal_condition_layout,
    compute_raw_causal_condition,
    fit_train_condition_normalizer,
)


def _expected_names(symbols: tuple[str, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for group in ("trend_fast", "trend_base", "trend_slow", "realized_vol_24h"):
        names.extend(f"{group}:{symbol}" for symbol in symbols)
    for left_index, left in enumerate(symbols):
        for right in symbols[left_index + 1 :]:
            names.append(f"corr_7d:{left}|{right}")
    for group in (
        "spread_rate",
        "log_market_notional",
        "funding_rate",
        "funding_due",
        "tradable",
        "buy_allowed",
        "sell_allowed",
        "borrow_available",
        "asset_active",
    ):
        names.extend(f"{group}:{symbol}" for symbol in symbols)
    return tuple(names)


def _copy_dataset(dataset: MarketDataset, **updates: Any) -> MarketDataset:
    values = {
        field: getattr(dataset, field)
        for field in dataset.__dataclass_fields__
        if dataset.__dataclass_fields__[field].init and not field.startswith("_")
    }
    values["identity_payload_json"] = None
    values.update(updates)
    return MarketDataset(**values)


def test_layout_has_exact_order_and_binary_mask(
    market_dataset_factory: Any,
) -> None:
    dataset = market_dataset_factory()
    layout = build_causal_condition_layout(dataset.symbols)
    assert layout.feature_names == _expected_names(dataset.symbols)
    binary_names = {
        name
        for name in layout.feature_names
        if name.split(":", 1)[0]
        in {
            "funding_due",
            "tradable",
            "buy_allowed",
            "sell_allowed",
            "borrow_available",
            "asset_active",
        }
    }
    assert (
        set(np.asarray(layout.feature_names)[~layout.continuous_mask]) == binary_names
    )
    assert layout.continuous_mask.flags.writeable is False


def test_raw_condition_is_finite_readonly_and_prefix_causal(
    market_dataset_factory: Any,
) -> None:
    dataset = market_dataset_factory()
    strategy = TrendStrategy(TrendConfig(fast_hours=12, base_hours=48, slow_hours=96))
    index = 800
    before = compute_raw_causal_condition(dataset, index, strategy)

    suffix = slice(index + 1, None)
    close = dataset.close.copy()
    close[suffix] *= 9.0
    funding = dataset.funding_rate.copy()
    funding[suffix] = 0.99
    volume = dataset.volume.copy()
    volume[suffix] *= 100.0
    tradable = dataset.tradable.copy()
    tradable[suffix] = False
    open_price = dataset.open.copy()
    high = dataset.high.copy()
    low = dataset.low.copy()
    open_price[suffix] = close[suffix]
    high[suffix] = close[suffix]
    low[suffix] = close[suffix]
    changed = _copy_dataset(
        dataset,
        close=close,
        open=open_price,
        high=high,
        low=low,
        funding_rate=funding,
        volume=volume,
        tradable=tradable,
    )
    after = compute_raw_causal_condition(changed, index, strategy)
    np.testing.assert_array_equal(before, after)
    assert before.flags.writeable is False
    assert np.isfinite(before).all()


def test_condition_values_match_direct_calculation(
    market_dataset_factory: Any,
) -> None:
    dataset = market_dataset_factory()
    strategy = TrendStrategy(TrendConfig(fast_hours=12, base_hours=48, slow_hours=96))
    config = CausalConditionConfig()
    index = 800
    layout = build_causal_condition_layout(dataset.symbols)
    raw = compute_raw_causal_condition(dataset, index, strategy, config)
    by_name = dict(zip(layout.feature_names, raw, strict=True))
    targets = strategy.targets(dataset, index)
    for symbol_index, symbol in enumerate(dataset.symbols):
        assert by_name[f"trend_fast:{symbol}"] == pytest.approx(
            targets.fast[symbol_index]
        )
        assert by_name[f"trend_base:{symbol}"] == pytest.approx(
            targets.base[symbol_index]
        )
        assert by_name[f"trend_slow:{symbol}"] == pytest.approx(
            targets.slow[symbol_index]
        )
        assert by_name[f"spread_rate:{symbol}"] == pytest.approx(
            dataset.resolved_array("spread_rate")[index, symbol_index]
        )
        assert by_name[f"log_market_notional:{symbol}"] == pytest.approx(
            np.log(
                dataset.market_notional(index, prices=dataset.close[index])[
                    symbol_index
                ]
            )
        )


def test_train_normalizer_is_train_only_and_binary_passthrough(
    market_dataset_factory: Any,
) -> None:
    dataset = market_dataset_factory(n_bars=1_100)
    train = MarketDatasetView(dataset, 0, 950)
    strategy = TrendStrategy(TrendConfig(fast_hours=12, base_hours=48, slow_hours=96))
    anchors = np.arange(700, 900, 5)
    matrix = np.vstack(
        [
            compute_raw_causal_condition(dataset, int(index), strategy)
            for index in anchors
        ]
    )
    layout = build_causal_condition_layout(dataset.symbols)
    normalizer = fit_train_condition_normalizer(matrix, layout, train.identity)

    transformed = normalizer.transform(matrix[0])
    np.testing.assert_array_equal(
        transformed[~layout.continuous_mask], matrix[0, ~layout.continuous_mask]
    )
    assert transformed.flags.writeable is False
    assert normalizer.median.flags.writeable is False
    assert normalizer.scale.flags.writeable is False
    np.testing.assert_array_equal(normalizer.median[~layout.continuous_mask], 0.0)
    np.testing.assert_array_equal(normalizer.scale[~layout.continuous_mask], 1.0)

    changed_close = dataset.close.copy()
    changed_open = dataset.open.copy()
    changed_high = dataset.high.copy()
    changed_low = dataset.low.copy()
    changed_mark = dataset.mark_price.copy()
    changed_index = dataset.index_price.copy()
    for array in (
        changed_close,
        changed_open,
        changed_high,
        changed_low,
        changed_mark,
        changed_index,
    ):
        array[train.stop :] *= 11.0
    changed = _copy_dataset(
        dataset,
        close=changed_close,
        open=changed_open,
        high=changed_high,
        low=changed_low,
        mark_price=changed_mark,
        index_price=changed_index,
    )
    changed_train = MarketDatasetView(changed, train.start, train.stop)
    changed_matrix = np.vstack(
        [
            compute_raw_causal_condition(changed, int(index), strategy)
            for index in anchors
        ]
    )
    changed_normalizer = fit_train_condition_normalizer(
        changed_matrix, layout, changed_train.identity
    )
    np.testing.assert_array_equal(normalizer.median, changed_normalizer.median)
    np.testing.assert_array_equal(normalizer.scale, changed_normalizer.scale)
    assert normalizer.digest == changed_normalizer.digest


def test_normalizer_rejects_non_binary_mask_values() -> None:
    layout = build_causal_condition_layout(("BTCUSDT",))
    width = len(layout.feature_names)
    matrix = np.zeros((3, width), dtype=np.float64)
    normalizer = fit_train_condition_normalizer(matrix, layout, "b" * 64)
    bad = matrix[0].copy()
    bad[np.flatnonzero(~layout.continuous_mask)[0]] = 0.5
    with pytest.raises(ValueError, match="binary"):
        normalizer.transform(bad)


def test_contract_validation() -> None:
    with pytest.raises(ValueError):
        CausalConditionConfig(volatility_hours=0)
    layout = build_causal_condition_layout(("BTCUSDT",))
    with pytest.raises(ValueError):
        TrainRobustConditionNormalizer(
            feature_names=layout.feature_names,
            continuous_mask=layout.continuous_mask,
            median=np.zeros(len(layout.feature_names)),
            scale=np.zeros(len(layout.feature_names)),
            train_view_digest="c" * 64,
        )
    with pytest.raises(ValueError):
        replace(layout, feature_names=("duplicate",) * len(layout.feature_names))


def test_condition_history_start_and_legacy_lookbacks_are_enforced(
    market_dataset_factory: Any,
) -> None:
    dataset = market_dataset_factory(n_bars=1_024)
    hourly = TrendStrategy(TrendConfig(fast_hours=12, base_hours=48, slow_hours=96))
    with pytest.raises(ValueError, match="history_start"):
        compute_raw_causal_condition(dataset, 800, hourly, history_start=801)
    with pytest.raises(ValueError, match="escapes assigned range"):
        compute_raw_causal_condition(dataset, 800, hourly, history_start=200)

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
    vector = compute_raw_causal_condition(
        dataset,
        800,
        legacy,
        CausalConditionConfig(volatility_hours=1, correlation_hours=2),
        history_start=788,
    )
    assert np.isfinite(vector).all()
