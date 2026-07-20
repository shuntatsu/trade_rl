from __future__ import annotations

from dataclasses import replace

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.rl.sequence_normalization import SequenceFeatureNormalizer
from trade_rl.rl.sequence_observations import (
    SequenceObservationBuilder,
    SequenceWindowSpec,
)


def _dataset(*, future_scale: float = 1.0) -> MarketDataset:
    n = 140
    names = (
        "15m__ret",
        "1h__ret",
        "4h__ret",
        "1d__ret",
    )
    features = np.zeros((n, 2, 4), dtype=np.float32)
    available = np.ones_like(features, dtype=np.bool_)
    ages = np.zeros_like(features, dtype=np.float32)
    steps = (1, 4, 16, 32)
    for feature_index, step in enumerate(steps):
        for index in range(n):
            event = (index // step) + 10 * feature_index
            features[index, :, feature_index] = event + np.arange(2)
            ages[index, :, feature_index] = float(index % step) * 0.25
    features[120:] *= future_scale
    close = 100.0 + np.arange(n, dtype=np.float64)[:, None] + np.arange(2)[None, :]
    return MarketDataset(
        dataset_id=("a" if future_scale == 1.0 else "b") * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n) * np.timedelta64(15, "m"),
        features=features,
        global_features=np.zeros((n, 1), dtype=np.float32),
        open=np.vstack((close[:1], close[:-1])),
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=np.full_like(close, 1_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=available,
        feature_staleness_hours=ages,
        feature_names=names,
        global_feature_names=("regime",),
        periods_per_year=35_040,
    )


def _builder() -> SequenceObservationBuilder:
    return SequenceObservationBuilder(
        windows=(
            SequenceWindowSpec("15m", 4),
            SequenceWindowSpec("1h", 4),
            SequenceWindowSpec("4h", 3),
            SequenceWindowSpec("1d", 2),
        )
    )


def test_sequence_normalizer_uses_only_train_prefix() -> None:
    baseline = SequenceFeatureNormalizer.fit(
        _dataset(), _builder(), train_start=96, train_end=120
    )
    mutated = SequenceFeatureNormalizer.fit(
        _dataset(future_scale=1_000.0), _builder(), train_start=96, train_end=120
    )

    assert baseline.digest != mutated.digest  # dataset identities differ
    for timeframe in baseline.feature_names:
        np.testing.assert_allclose(
            baseline.center[timeframe], mutated.center[timeframe]
        )
        np.testing.assert_allclose(baseline.scale[timeframe], mutated.scale[timeframe])


def test_sequence_normalizer_masks_unavailable_values_after_scaling() -> None:
    dataset = _dataset()
    builder = _builder()
    normalizer = SequenceFeatureNormalizer.fit(
        dataset, builder, train_start=96, train_end=120
    )
    sequence = builder.build(dataset, index=128)
    values = sequence.values["1h"].copy()
    available = sequence.available["1h"].copy()
    available[:, -1, :] = False
    values[:, -1, :] = 1_000_000.0

    transformed = normalizer.transform(
        "1h",
        values,
        available,
        feature_names=sequence.feature_names["1h"],
    )

    assert np.count_nonzero(transformed[:, -1, :]) == 0
    assert np.isfinite(transformed).all()
    assert float(np.max(np.abs(transformed))) <= normalizer.clip


def test_sequence_normalizer_layout_contract_survives_dataset_view_identity() -> None:
    dataset = _dataset()
    view = replace(dataset, dataset_id="c" * 64)
    builder = _builder()
    normalizer = SequenceFeatureNormalizer.fit(
        view,
        builder,
        train_start=96,
        train_end=120,
        source_dataset_id=dataset.dataset_id,
    )

    assert builder.schema_digest(view) != builder.schema_digest(dataset)
    assert builder.layout_digest(view) == builder.layout_digest(dataset)
    assert normalizer.sequence_schema_digest == builder.layout_digest(dataset)


def test_sequence_normalizer_records_channel_sample_counts() -> None:
    normalizer = SequenceFeatureNormalizer.fit(
        _dataset(), _builder(), train_start=96, train_end=120
    )

    assert normalizer.minimum_samples_per_channel == 1
    for timeframe, counts in normalizer.sample_count.items():
        assert counts.shape == normalizer.center[timeframe].shape
        assert np.all(counts > 0)
        assert np.issubdtype(counts.dtype, np.integer)
    assert "sample_count" in normalizer.digest_payload()


def test_sequence_normalizer_fails_closed_when_a_required_channel_has_no_events() -> (
    None
):
    dataset = _dataset()
    available = dataset.feature_available.copy()
    available[:, :, 3] = False
    staleness = dataset.feature_staleness.copy()
    staleness[:, :, 3] = 1.0
    missing = replace(
        dataset,
        feature_available=available,
        feature_staleness=staleness,
        dataset_id="e" * 64,
    )

    with np.testing.assert_raises_regex(ValueError, "1d.*15m__|1d.*sample|1d"):
        SequenceFeatureNormalizer.fit(
            missing,
            _builder(),
            train_start=96,
            train_end=120,
            minimum_samples_per_channel=1,
        )


def test_precomputed_sequence_policy_plane_matches_legacy_window_normalization() -> (
    None
):
    import trade_rl.rl.sequence_observations as sequence_observations

    factory = getattr(sequence_observations, "build_sequence_policy_plane", None)
    assert callable(factory)
    dataset = _dataset()
    builder = _builder()
    normalizer = SequenceFeatureNormalizer.fit(
        dataset, builder, train_start=96, train_end=120
    )

    plane = factory(dataset, builder, normalizer)
    assert factory(dataset, builder, normalizer) is plane
    for decision_index in (128, 129):
        actual = plane.components(decision_index)
        sequence = builder.build(dataset, index=decision_index)
        for timeframe in sequence.values:
            expected_values = sequence_observations.sequence_policy_values(
                timeframe=timeframe,
                values=sequence.values[timeframe],
                available=sequence.available[timeframe],
                feature_names=sequence.feature_names[timeframe],
                sequence_normalizer=normalizer,
            )
            np.testing.assert_array_equal(
                actual[f"sequence_{timeframe}_values"], expected_values
            )
            np.testing.assert_array_equal(
                actual[f"sequence_{timeframe}_available"],
                sequence.available[timeframe].astype(np.uint8),
            )
            np.testing.assert_array_equal(
                actual[f"sequence_{timeframe}_staleness"],
                sequence.staleness[timeframe].astype(np.float16),
            )
