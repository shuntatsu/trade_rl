from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import trade_rl.strategies.forecasts.supervised as supervised_module
from trade_rl.data.contracts import FeatureKind, FeatureSpec, MarketBuildConfig
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecasts.supervised import (
    CausalForecastTrainingSet,
    build_causal_forecast_training_set,
)


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


def direct_training_set(label_end_times: np.ndarray) -> CausalForecastTrainingSet:
    return CausalForecastTrainingSet(
        feature_indices=(0,),
        features=np.asarray([[1.0], [2.0]]),
        labels=np.asarray([0.01, 0.02]),
        label_end_times=label_end_times,
        sample_weights=np.ones(2),
        fit_cutoff=np.datetime64("1970-01-02T00:00:00", "ns"),
        horizon_hours=1,
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


def test_direct_training_set_rejects_nat_label_end_times() -> None:
    label_end_times = np.asarray(
        ("1970-01-01T00:00:00", "1970-01-01T12:00:00"),
        dtype="datetime64[ns]",
    )
    label_end_times[0] = np.datetime64("NaT", "ns")

    with pytest.raises(ValueError, match="label_end_times must not contain NaT"):
        direct_training_set(label_end_times)


def test_direct_training_set_rejects_non_datetime_label_end_times() -> None:
    with pytest.raises(ValueError, match="label_end_times must use a datetime64 dtype"):
        direct_training_set(np.asarray((1, 2), dtype=np.int64))


@pytest.mark.parametrize(
    "label_end_times",
    (
        np.asarray(
            ("2500-01-01T00:00:00", "1970-01-01T12:00:00"),
            dtype="datetime64[s]",
        ),
        np.asarray(
            (
                "1970-01-01T00:00:00.000000000500",
                "1970-01-01T00:00:00.000000001000",
            ),
            dtype="datetime64[ps]",
        ),
    ),
)
def test_direct_training_set_rejects_lossy_label_end_timestamps(
    label_end_times: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="exactly representable in nanoseconds"):
        direct_training_set(label_end_times)


@pytest.mark.parametrize(
    ("unit", "label_ends"),
    (
        ("s", ("1970-01-01T00:00:00", "1970-01-01T12:00:00")),
        (
            "ps",
            (
                "1970-01-01T00:00:00.000000001000",
                "1970-01-01T00:00:00.000000002000",
            ),
        ),
    ),
)
def test_direct_training_set_accepts_lossless_label_end_timestamps(
    unit: str,
    label_ends: tuple[str, str],
) -> None:
    training = direct_training_set(np.asarray(label_ends, dtype=f"datetime64[{unit}]"))

    assert training.label_end_times.dtype == np.dtype("datetime64[ns]")
    np.testing.assert_array_equal(
        training.label_end_times,
        np.asarray(label_ends, dtype="datetime64[ns]"),
    )


def test_supervised_datetime_multiplier_overflow_is_rejected() -> None:
    dtype = np.dtype("datetime64[1000000W]")
    raw_ticks = np.asarray([-1, 0], dtype=np.int64)
    label_end_times = raw_ticks.view(dtype)
    exact_first_ns = int(raw_ticks[0]) * 1_000_000 * 7 * 24 * 60 * 60 * 1_000_000_000
    assert exact_first_ns == -604_800_000_000_000_000_000

    baseline = direct_training_set(
        np.asarray([0, 0], dtype=np.int64).view("datetime64[ns]")
    )
    wrapped_cutoff = np.datetime64(3_942_554_432_415_203_329, "ns")
    with pytest.raises(
        ValueError,
        match="label_end_times must be exactly representable in nanoseconds",
    ):
        replace(
            baseline,
            label_end_times=label_end_times,
            fit_cutoff=wrapped_cutoff,
        )

    with pytest.raises(ValueError, match="fit_cutoff must be exactly representable"):
        supervised_module._validated_fit_cutoff_ns(label_end_times[0])


def test_exact_large_multiplier_hours_are_lossless_for_labels_and_cutoff() -> None:
    dtype = np.dtype(">M8[1000000h]")
    raw_ticks = np.asarray([-2, -1], dtype=">i8")
    label_end_times = raw_ticks.view(dtype)
    expected_ns = np.asarray(
        [-7_200_000_000_000_000_000, -3_600_000_000_000_000_000],
        dtype=np.int64,
    ).view("datetime64[ns]")

    training = direct_training_set(label_end_times)
    np.testing.assert_array_equal(training.label_end_times, expected_ns)
    assert training.label_end_times.flags.writeable is False

    normalized_cutoff = supervised_module._validated_fit_cutoff_ns(label_end_times[0])
    assert int(normalized_cutoff.astype(np.int64)) == -7_200_000_000_000_000_000


@pytest.mark.parametrize(
    ("unit", "ticks", "expected_ns"),
    (
        (
            "Y",
            (-2, -1),
            (-63_158_400_000_000_000, -31_536_000_000_000_000),
        ),
        (
            "M",
            (-2, -1),
            (-5_270_400_000_000_000, -2_678_400_000_000_000),
        ),
        (
            "2M",
            (-2, -1),
            (-10_540_800_000_000_000, -5_270_400_000_000_000),
        ),
    ),
)
def test_exact_calendar_datetime_units_remain_supported(
    unit: str,
    ticks: tuple[int, int],
    expected_ns: tuple[int, int],
) -> None:
    label_end_times = np.asarray(ticks, dtype=np.int64).view(
        np.dtype(f"datetime64[{unit}]")
    )

    training = direct_training_set(label_end_times)

    np.testing.assert_array_equal(
        training.label_end_times.astype(np.int64),
        np.asarray(expected_ns, dtype=np.int64),
    )


@pytest.mark.parametrize(
    ("value", "expected_ns"),
    (
        (np.datetime64("1678", "Y"), -9_214_560_000_000_000_000),
        (np.datetime64("2262", "Y"), 9_214_646_400_000_000_000),
        (np.datetime64("1677-10", "M"), -9_222_508_800_000_000_000),
        (np.datetime64("2262-04", "M"), 9_222_422_400_000_000_000),
    ),
)
def test_calendar_cutoff_values_at_nanosecond_bounds_are_exact(
    value: np.datetime64,
    expected_ns: int,
) -> None:
    normalized = supervised_module._validated_fit_cutoff_ns(value)

    assert int(normalized.astype(np.int64)) == expected_ns


@pytest.mark.parametrize(
    "value",
    (
        np.datetime64("1677", "Y"),
        np.datetime64("2263", "Y"),
        np.datetime64("1677-09", "M"),
        np.datetime64("2262-05", "M"),
    ),
)
def test_calendar_cutoff_values_outside_nanosecond_bounds_are_rejected(
    value: np.datetime64,
) -> None:
    with pytest.raises(ValueError, match="fit_cutoff must be exactly representable"):
        supervised_module._validated_fit_cutoff_ns(value)


def test_exact_datetime_converter_preserves_empty_arrays() -> None:
    empty = np.asarray([], dtype=np.dtype(">M8[1000000h]"))

    normalized = supervised_module._datetime64_values_to_ns_exact(empty)

    assert normalized.shape == (0,)
    assert normalized.dtype == np.dtype("datetime64[ns]")


def test_direct_training_set_rejects_label_maturing_at_cutoff() -> None:
    cutoff = np.datetime64("1970-01-02T00:00:00", "ns")
    baseline = direct_training_set(
        np.asarray(
            [np.datetime64("1970-01-01", "ns")] * 2,
            dtype="datetime64[ns]",
        )
    )
    with pytest.raises(ValueError, match="strictly before fit_cutoff"):
        replace(
            baseline,
            label_end_times=np.asarray(
                [cutoff, np.datetime64("1970-01-01", "ns")],
                dtype="datetime64[ns]",
            ),
            fit_cutoff=cutoff,
        )


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
    "fit_cutoff",
    (
        np.datetime64("1677-01-01T00:00:00", "s"),
        np.datetime64("2263-01-01T00:00:00", "s"),
        np.datetime64("NaT", "ns"),
        np.datetime64("1969-12-01T00:00:00.000000000001", "ps"),
    ),
    ids=(
        "below-ns-range",
        "above-ns-range",
        "nat",
        "fractional-nanosecond",
    ),
)
def test_training_builder_rejects_lossy_cutoffs(fit_cutoff: np.datetime64) -> None:
    with pytest.raises(ValueError, match=r"fit_cutoff.*(NaT|datetime64\[ns\])"):
        build_causal_forecast_training_set(
            market(n_bars=5),
            feature_indices=(0,),
            fit_cutoff=fit_cutoff,
            horizon_hours=1,
        )


def test_training_builder_validates_cutoff_before_training_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_scope_is_checked(*args: object, **kwargs: object) -> None:
        pytest.fail("training scope was inspected before validating fit_cutoff")

    monkeypatch.setattr(
        supervised_module,
        "validated_training_scope",
        fail_if_scope_is_checked,
    )
    with pytest.raises(ValueError, match=r"fit_cutoff.*datetime64\[ns\]"):
        build_causal_forecast_training_set(
            market(n_bars=5),
            feature_indices=(0,),
            fit_cutoff=np.datetime64("1677-01-01T00:00:00", "s"),
            horizon_hours=1,
        )


def test_training_builder_preserves_exact_cutoffs_at_supported_precisions() -> None:
    dataset = market(n_bars=5)
    cutoffs = (
        np.datetime64("2026-01-01T04:00:00", "h"),
        np.datetime64("2026-01-01T04:00:00", "h").astype("datetime64[2h]"),
        np.datetime64("2026-01-01T04:00:00.000000000", "ns"),
    )

    training_sets = [
        build_causal_forecast_training_set(
            dataset,
            feature_indices=(0,),
            fit_cutoff=cutoff,
            horizon_hours=1,
        )
        for cutoff in cutoffs
    ]

    assert [training.n_samples for training in training_sets] == [3, 3, 3]
    assert all(
        training.fit_cutoff == np.datetime64("2026-01-01T04:00:00", "ns")
        for training in training_sets
    )


def test_training_builder_accepts_exact_datetime64_nanosecond_boundaries() -> None:
    dataset = market(n_bars=5)
    lowest = np.datetime64("1677-09-21T00:12:43.145224193", "ns")
    highest = np.datetime64("2262-04-11T23:47:16.854775807", "ns")

    with pytest.raises(ValueError, match="at least two eligible training rows"):
        build_causal_forecast_training_set(
            dataset,
            feature_indices=(0,),
            fit_cutoff=lowest,
            horizon_hours=1,
        )

    training = build_causal_forecast_training_set(
        dataset,
        feature_indices=(0,),
        fit_cutoff=highest,
        horizon_hours=1,
    )
    assert training.n_samples == 4


@pytest.mark.parametrize(
    "fit_cutoff",
    (
        np.datetime64("1677-01-01T00:00:00", "s"),
        np.datetime64("2263-01-01T00:00:00", "s"),
        np.datetime64("NaT", "ns"),
    ),
    ids=("below-ns-range", "above-ns-range", "nat"),
)
def test_training_set_rejects_lossy_cutoff_metadata(
    fit_cutoff: np.datetime64,
) -> None:
    training = build_causal_forecast_training_set(
        market(),
        feature_indices=(0,),
        fit_cutoff=np.datetime64("2026-01-01T08:00:00", "ns"),
        horizon_hours=2,
    )

    with pytest.raises(ValueError, match=r"fit_cutoff.*(NaT|datetime64\[ns\])"):
        replace(training, fit_cutoff=fit_cutoff)


def test_training_set_rejects_fractional_nanosecond_cutoff_metadata() -> None:
    training = build_causal_forecast_training_set(
        market(),
        feature_indices=(0,),
        fit_cutoff=np.datetime64("2026-01-01T08:00:00", "ns"),
        horizon_hours=2,
    )
    fractional_cutoff = np.datetime64(
        "1969-12-01T00:00:00.000000000001",
        "ps",
    )
    prior_labels = np.full(
        training.n_samples,
        np.datetime64("1969-11-30T00:00:00", "ns"),
        dtype="datetime64[ns]",
    )

    with pytest.raises(ValueError, match=r"fit_cutoff.*datetime64\[ns\]"):
        replace(
            training,
            label_end_times=prior_labels,
            fit_cutoff=fractional_cutoff,
        )


def test_training_set_accepts_subnanosecond_unit_when_value_is_exact() -> None:
    training = build_causal_forecast_training_set(
        market(),
        feature_indices=(0,),
        fit_cutoff=np.datetime64("2026-01-01T08:00:00", "ns"),
        horizon_hours=2,
    )
    exact_cutoff = np.datetime64(
        "1969-12-01T00:00:00.000000001000",
        "ps",
    )
    prior_labels = np.full(
        training.n_samples,
        np.datetime64("1969-11-30T00:00:00", "ns"),
        dtype="datetime64[ns]",
    )

    exact = replace(
        training,
        label_end_times=prior_labels,
        fit_cutoff=exact_cutoff,
    )

    assert exact.fit_cutoff == exact_cutoff.astype("datetime64[ns]")


def test_training_builder_rejects_horizon_that_wraps_to_one_hour() -> None:
    with pytest.raises(ValueError, match="representable.*nanosecond"):
        build_causal_forecast_training_set(
            market(n_bars=5),
            feature_indices=(0,),
            fit_cutoff=np.datetime64("2026-01-01T05:00:00", "ns"),
            horizon_hours=2**51 + 1,
        )


def test_training_builder_rejects_horizon_just_over_int64_nanosecond_limit() -> None:
    max_safe_hours = ((2**63) - 1) // 3_600_000_000_000

    with pytest.raises(ValueError, match="representable.*nanosecond"):
        build_causal_forecast_training_set(
            market(n_bars=5),
            feature_indices=(0,),
            fit_cutoff=np.datetime64("2026-01-01T05:00:00", "ns"),
            horizon_hours=max_safe_hours + 1,
        )


def test_training_builder_accepts_exact_maximum_horizon_without_fitting_rows() -> None:
    max_safe_hours = ((2**63) - 1) // 3_600_000_000_000

    with pytest.raises(ValueError, match="at least two eligible training rows"):
        build_causal_forecast_training_set(
            market(n_bars=5),
            feature_indices=(0,),
            fit_cutoff=np.datetime64("2026-01-01T05:00:00", "ns"),
            horizon_hours=max_safe_hours,
        )


def test_training_set_rejects_unrepresentable_horizon_metadata() -> None:
    max_safe_hours = ((2**63) - 1) // 3_600_000_000_000
    training = build_causal_forecast_training_set(
        market(),
        feature_indices=(0,),
        fit_cutoff=np.datetime64("2026-01-01T08:00:00", "ns"),
        horizon_hours=2,
    )

    with pytest.raises(ValueError, match="representable.*nanosecond"):
        replace(training, horizon_hours=max_safe_hours + 1)


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
