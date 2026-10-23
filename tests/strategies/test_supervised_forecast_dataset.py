from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data.contracts import FeatureKind, FeatureSpec, MarketBuildConfig
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecasts.supervised import build_causal_forecast_training_set


def market(*, future_multiplier: float = 1.0, n_bars: int = 12) -> MarketDataset:
    phase = np.arange(n_bars, dtype=np.float64)
    close = (100.0 * np.exp(0.01 * phase)).reshape(-1, 1)
    close[min(8, n_bars) :, 0] *= future_multiplier
    features = np.stack((phase, phase**2), axis=-1).reshape(n_bars, 1, 2)
    return MarketDataset(
        dataset_id="6" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=features.astype(np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 2), dtype=np.bool_),
        feature_names=("linear", "quadratic"),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def pooled_market() -> MarketDataset:
    n_bars = 12
    phase = np.arange(n_bars, dtype=np.float64)
    close = np.stack(
        (
            100.0 * np.exp(0.01 * phase),
            200.0 * np.exp(-0.005 * phase),
        ),
        axis=1,
    )
    features = np.empty((n_bars, 2, 2), dtype=np.float32)
    features[:, 0, 0] = phase
    features[:, 0, 1] = phase**2
    features[:, 1, 0] = 100.0 + phase
    features[:, 1, 1] = 10_000.0 + phase**2
    available = np.ones_like(features, dtype=np.bool_)
    available[:3, 1, :] = False
    return MarketDataset(
        dataset_id="7" * 64,
        symbols=("BTCUSDT", "ETHUSDT"),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n_bars, 2), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 2)),
        tradable=np.ones((n_bars, 2), dtype=np.bool_),
        feature_available=available,
        feature_names=("linear", "quadratic"),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def test_training_set_is_strictly_cutoff_causal_and_future_invariant() -> None:
    cutoff = np.datetime64("2026-01-01T08:00:00", "ns")
    baseline = build_causal_forecast_training_set(
        market(),
        feature_indices=(0, 1),
        fit_cutoff=cutoff,
        horizon_hours=2,
    )
    mutated = build_causal_forecast_training_set(
        market(future_multiplier=7.0),
        feature_indices=(0, 1),
        fit_cutoff=cutoff,
        horizon_hours=2,
    )

    assert baseline.n_samples == 6
    assert baseline.label_end_times.max() < cutoff
    np.testing.assert_array_equal(baseline.features, mutated.features)
    np.testing.assert_array_equal(baseline.labels, mutated.labels)
    np.testing.assert_array_equal(baseline.sample_weights, np.ones(6))
    assert baseline.features.flags.writeable is False
    assert baseline.labels.flags.writeable is False
    assert baseline.label_end_times.flags.writeable is False
    assert baseline.sample_weights.flags.writeable is False


def test_72_hour_training_labels_use_exact_elapsed_endpoint_and_maturity_cutoff() -> (
    None
):
    dataset = market(n_bars=80)
    cutoff = dataset.timestamps[74]

    training = build_causal_forecast_training_set(
        dataset,
        feature_indices=(0,),
        fit_cutoff=cutoff,
        horizon_hours=72,
    )

    np.testing.assert_array_equal(training.features[:, 0], np.asarray([0.0, 1.0]))
    np.testing.assert_array_equal(
        training.label_end_times,
        dataset.timestamps[[72, 73]],
    )
    assert np.all(training.label_end_times < cutoff)
    np.testing.assert_allclose(
        training.labels,
        np.log(dataset.close[[72, 73], 0] / dataset.close[[0, 1], 0]),
    )


@pytest.mark.parametrize(
    ("available_hour", "expected_included"),
    ((9, False), (8, True), (7, True)),
)
def test_training_label_endpoint_price_must_be_available_by_fit_cutoff(
    available_hour: int,
    expected_included: bool,
) -> None:
    dataset = market()
    timestamps = dataset.timestamps.astype("datetime64[ns]")
    available_at = dataset.resolved_array("available_at").copy()
    information_available = dataset.resolved_array("information_available").copy()
    available_at[4, 0] = timestamps[available_hour]
    information_available[4, 0] = False
    dataset = replace(
        dataset,
        available_at=available_at,
        information_available=information_available,
    )

    training = build_causal_forecast_training_set(
        dataset,
        feature_indices=(0, 1),
        fit_cutoff=timestamps[8],
        horizon_hours=2,
    )

    endpoint_row_present = bool(np.any(training.features[:, 0] == 2.0))
    assert endpoint_row_present is expected_included
    if expected_included:
        expected_return = np.log(dataset.close[4, 0] / dataset.close[2, 0])
        endpoint_position = int(np.flatnonzero(training.features[:, 0] == 2.0)[0])
        assert training.labels[endpoint_position] == pytest.approx(expected_return)


def test_training_label_start_price_must_be_available_by_fit_cutoff() -> None:
    dataset = market()
    timestamps = dataset.timestamps.astype("datetime64[ns]")
    available_at = dataset.resolved_array("available_at").copy()
    information_available = dataset.resolved_array("information_available").copy()
    available_at[3, 0] = timestamps[9]
    information_available[3, 0] = False
    dataset = replace(
        dataset,
        available_at=available_at,
        information_available=information_available,
    )
    assert dataset.feature_available[3, 0].all()

    training = build_causal_forecast_training_set(
        dataset,
        feature_indices=(0, 1),
        fit_cutoff=timestamps[8],
        horizon_hours=2,
    )

    # The synthetic feature mask remains available, so this checks the separate
    # source-price availability contract for the label's start endpoint.
    assert not np.any(training.features[:, 0] == 3.0)


def test_multi_symbol_rows_use_no_symbol_feature_and_equal_symbol_weight() -> None:
    training = build_causal_forecast_training_set(
        pooled_market(),
        feature_indices=(0, 1),
        fit_cutoff=np.datetime64("2026-01-01T08:00:00", "ns"),
        horizon_hours=2,
    )

    assert training.features.shape == (9, 2)
    assert training.n_samples == 9
    np.testing.assert_allclose(training.sample_weights[:6], np.full(6, 0.75))
    np.testing.assert_allclose(training.sample_weights[6:], np.full(3, 1.5))
    assert training.sample_weights[:6].sum() == 4.5
    assert training.sample_weights[6:].sum() == 4.5
    assert training.sample_weights.sum() == 9.0
    assert training.label_end_times.max() < training.fit_cutoff


def test_explicit_fit_symbol_scope_excludes_unseen_symbol_rows() -> None:
    training = build_causal_forecast_training_set(
        pooled_market(),
        feature_indices=(0, 1),
        fit_symbol_indices=(0,),
        fit_cutoff=np.datetime64("2026-01-01T08:00:00", "ns"),
        horizon_hours=2,
    )

    assert training.features.shape == (6, 2)
    assert training.n_samples == 6
    np.testing.assert_array_equal(training.features[:, 0], np.arange(6))
    np.testing.assert_array_equal(training.sample_weights, np.ones(6))


@pytest.mark.parametrize(
    "fit_symbol_indices",
    ((), (0, 0), (-1,), (2,)),
)
def test_fit_symbol_scope_rejects_empty_duplicate_or_out_of_range_indices(
    fit_symbol_indices: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="symbol"):
        build_causal_forecast_training_set(
            pooled_market(),
            feature_indices=(0, 1),
            fit_symbol_indices=fit_symbol_indices,
            fit_cutoff=np.datetime64("2026-01-01T08:00:00", "ns"),
            horizon_hours=2,
        )


def test_explicit_fit_scope_rejects_a_symbol_with_no_eligible_rows() -> None:
    with pytest.raises(ValueError, match="fit symbol.*eligible training rows"):
        build_causal_forecast_training_set(
            pooled_market(),
            feature_indices=(0, 1),
            fit_symbol_indices=(0, 1),
            fit_cutoff=np.datetime64("2026-01-01T04:00:00", "ns"),
            horizon_hours=2,
        )


def _with_feature_provenance(
    dataset: MarketDataset,
    *,
    first_kind: FeatureKind,
    reference_symbol: str = "BTCUSDT",
) -> MarketDataset:
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(
            FeatureSpec(name=dataset.feature_names[0], kind=first_kind),
            FeatureSpec(
                name=dataset.feature_names[1],
                kind=FeatureKind.LOG_RETURN,
            ),
        ),
        cross_asset_reference_symbol=reference_symbol,
    )
    return dataset.with_content_identity({"config": config.canonical_payload()})


@pytest.mark.parametrize(
    "kind",
    (
        FeatureKind.CROSS_ASSET_DISPERSION,
        FeatureKind.CROSS_SECTIONAL_MOMENTUM_RANK,
    ),
)
def test_subset_fit_rejects_universe_dependent_cross_asset_feature(
    kind: FeatureKind,
) -> None:
    dataset = _with_feature_provenance(pooled_market(), first_kind=kind)

    with pytest.raises(ValueError, match="fit.*scope|outside.*fit"):
        build_causal_forecast_training_set(
            dataset,
            feature_indices=(0,),
            fit_symbol_indices=(0,),
            fit_cutoff=np.datetime64("2026-01-01T08:00:00", "ns"),
            horizon_hours=2,
        )


def test_subset_fit_rejects_reference_feature_when_reference_is_holdout() -> None:
    dataset = _with_feature_provenance(
        pooled_market(),
        first_kind=FeatureKind.RELATIVE_RETURN_TO_BTC,
    )

    with pytest.raises(ValueError, match="reference.*fit|fit.*reference"):
        build_causal_forecast_training_set(
            dataset,
            feature_indices=(0,),
            fit_symbol_indices=(1,),
            fit_cutoff=np.datetime64("2026-01-01T08:00:00", "ns"),
            horizon_hours=2,
        )


def test_subset_fit_allows_reference_feature_when_reference_is_in_fit_scope() -> None:
    dataset = _with_feature_provenance(
        pooled_market(),
        first_kind=FeatureKind.RELATIVE_RETURN_TO_BTC,
    )

    training = build_causal_forecast_training_set(
        dataset,
        feature_indices=(0,),
        fit_symbol_indices=(0,),
        fit_cutoff=np.datetime64("2026-01-01T08:00:00", "ns"),
        horizon_hours=2,
    )

    assert training.n_samples == 6
