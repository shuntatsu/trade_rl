from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.rl.sequence_observations import (
    SequenceObservationBuilder,
    SequenceWindowSpec,
)


def _dataset(*, mutate_future: bool = False) -> MarketDataset:
    n = 96
    symbols = ("BTCUSDT", "ETHUSDT")
    names = (
        "15m__ret",
        "15m__rsi",
        "1h__ret",
        "1h__rsi",
        "4h__ret",
        "4h__rsi",
        "1d__ret",
        "1d__rsi",
    )
    timestamps = np.datetime64("2026-01-01T00:15", "ns") + np.arange(
        n
    ) * np.timedelta64(15, "m")
    feature_values = np.zeros((n, len(symbols), len(names)), dtype=np.float32)
    for time_index in range(n):
        for symbol_index in range(len(symbols)):
            for feature_index in range(len(names)):
                feature_values[time_index, symbol_index, feature_index] = (
                    10_000 * feature_index + 100 * symbol_index + time_index
                )
    if mutate_future:
        feature_values[70:] += 1_000_000.0
    close = (
        100.0
        + np.arange(n, dtype=np.float64)[:, None]
        + np.arange(len(symbols))[None, :]
    )
    open_price = np.vstack((close[:1], close[:-1]))
    return MarketDataset(
        dataset_id=("b" if mutate_future else "a") * 64,
        symbols=symbols,
        timestamps=timestamps,
        features=feature_values,
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=open_price,
        high=np.maximum(open_price, close) + 1.0,
        low=np.minimum(open_price, close) - 1.0,
        close=close,
        volume=np.full_like(close, 1_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones_like(feature_values, dtype=np.bool_),
        feature_staleness_hours=np.zeros_like(feature_values, dtype=np.float32),
        feature_names=names,
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


def _builder() -> SequenceObservationBuilder:
    return SequenceObservationBuilder(
        windows=(
            SequenceWindowSpec("15m", 4),
            SequenceWindowSpec("1h", 3),
            SequenceWindowSpec("4h", 2),
            SequenceWindowSpec("1d", 1),
        )
    )


def test_sequence_builder_samples_each_native_clock_without_repeating_base_rows() -> (
    None
):
    dataset = _dataset()
    result = _builder().build(dataset, index=64)

    assert result.values["15m"].shape == (2, 4, 2)
    assert result.values["1h"].shape == (2, 3, 2)
    assert result.values["4h"].shape == (2, 2, 2)
    assert result.values["1d"].shape == (2, 1, 2)

    np.testing.assert_array_equal(result.source_indices["15m"], [61, 62, 63, 64])
    np.testing.assert_array_equal(result.source_indices["1h"], [56, 60, 64])
    np.testing.assert_array_equal(result.source_indices["4h"], [48, 64])
    np.testing.assert_array_equal(result.source_indices["1d"], [64])

    # Only channels belonging to the requested native clock are exposed.
    assert result.feature_names["1h"] == ("1h__ret", "1h__rsi")
    assert result.values["1h"][0, -1, 0] == 20_000 + 64


def test_sequence_builder_future_mutation_cannot_change_observation_prefix() -> None:
    baseline = _builder().build(_dataset(), index=64)
    mutated = _builder().build(_dataset(mutate_future=True), index=64)

    for timeframe in baseline.values:
        np.testing.assert_array_equal(
            baseline.values[timeframe], mutated.values[timeframe]
        )
        np.testing.assert_array_equal(
            baseline.available[timeframe], mutated.available[timeframe]
        )
        np.testing.assert_array_equal(
            baseline.staleness[timeframe], mutated.staleness[timeframe]
        )


def test_sequence_contract_digest_binds_window_and_feature_order() -> None:
    dataset = _dataset()
    builder = _builder()
    digest = builder.schema_digest(dataset)
    assert len(digest) == 64

    changed_window = SequenceObservationBuilder(
        windows=(
            SequenceWindowSpec("15m", 5),
            SequenceWindowSpec("1h", 3),
            SequenceWindowSpec("4h", 2),
            SequenceWindowSpec("1d", 1),
        )
    )
    assert changed_window.schema_digest(dataset) != digest

    reordered = replace(
        dataset,
        dataset_id="c" * 64,
        features=dataset.features[:, :, ::-1],
        feature_available=dataset.feature_available[:, :, ::-1],
        feature_staleness_hours=dataset.feature_staleness_hours[:, :, ::-1],
        feature_names=tuple(reversed(dataset.feature_names)),
        identity_payload_json=None,
    )
    assert builder.schema_digest(reordered) != digest


def test_structured_policy_observation_splits_current_state_and_sequences() -> None:
    from trade_rl.rl.observations import observation_layout
    from trade_rl.rl.sequence_observations import build_structured_policy_observation

    dataset = _dataset()
    sequence = _builder().build(dataset, index=64)
    layout = observation_layout(
        dataset, action_size=2, n_factors=0, finite_horizon=True
    )
    flat = np.arange(layout.size, dtype=np.float32)

    structured = build_structured_policy_observation(
        sequence=sequence,
        current_flat=flat,
        layout=layout,
        n_features=dataset.n_features,
    )

    assert structured["current_snapshot"].shape == (
        dataset.n_symbols,
        4 * dataset.n_features,
    )
    assert structured["asset_state"].shape == (
        dataset.n_symbols,
        layout.per_symbol_width - 4 * dataset.n_features,
    )
    assert structured["global_state"].shape == (layout.global_width,)
    assert structured["active"].shape == (dataset.n_symbols,)
    assert structured["sequence_1h_values"].shape == (dataset.n_symbols, 3, 2)
    assert structured["sequence_1h_available"].dtype == np.uint8
    assert structured["sequence_1h_staleness"].dtype == np.float16


def test_sequence_observation_schema_is_index_backed() -> None:
    from trade_rl.rl.sequence_observations import SEQUENCE_OBSERVATION_SCHEMA

    assert SEQUENCE_OBSERVATION_SCHEMA == "native_timeframe_sequence_observation_v2"
