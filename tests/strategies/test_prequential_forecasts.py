from __future__ import annotations

import math
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecasts.prequential import (
    ForecastPacket,
    LightGBMForecastFitter,
    PrequentialForecastSource,
    PrequentialSchedule,
    RidgeForecastFitter,
)
from trade_rl.strategies.forecasts.supervised import (
    CausalForecastTrainingSet,
    build_causal_forecast_training_set,
)

START = np.datetime64("2026-01-01T00:00:00", "ns")
HOUR = np.timedelta64(1, "h")
DAY = np.timedelta64(1, "D")
N_DAYS = 100
N_BARS = N_DAYS * 24
SYMBOLS = ("AAAUSDT", "BBBUSDT", "CCCUSDT")


def _features_from_close(close: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_bars, n_symbols = close.shape
    features = np.zeros((n_bars, n_symbols, 2), dtype=np.float32)
    available = np.zeros((n_bars, n_symbols, 2), dtype=np.bool_)
    trailing = np.log(close[24:] / close[:-24])
    features[24:, :, 0] = trailing
    features[24:, :, 1] = trailing**2
    available[24:] = True
    return features, available


def market(
    *,
    mutate_after: int | None = None,
    available_lag_hours: int = 0,
    seed: int = 7,
) -> MarketDataset:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.004, size=(N_BARS, len(SYMBOLS)))
    steps[:, 0] += 0.0005
    log_close = np.cumsum(steps, axis=0)
    close = 100.0 * np.exp(log_close)
    features, feature_available = _features_from_close(close)
    if mutate_after is not None:
        later = np.random.default_rng(seed + 1)
        close[mutate_after + 1 :] *= np.exp(
            later.normal(0.0, 0.5, size=close[mutate_after + 1 :].shape)
        )
        features[mutate_after + 1 :] = later.normal(
            0.0, 1.0, size=features[mutate_after + 1 :].shape
        ).astype(np.float32)
    timestamps = START + np.arange(N_BARS) * HOUR
    available_at = None
    if available_lag_hours:
        available_at = np.repeat(
            (timestamps + available_lag_hours * HOUR).reshape(-1, 1),
            len(SYMBOLS),
            axis=1,
        )
    return MarketDataset(
        dataset_id="5" * 64,
        symbols=SYMBOLS,
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((N_BARS, 1), dtype=np.float32),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((N_BARS, len(SYMBOLS)), 1_000_000.0),
        funding_rate=np.zeros((N_BARS, len(SYMBOLS))),
        tradable=np.ones((N_BARS, len(SYMBOLS)), dtype=np.bool_),
        feature_available=feature_available,
        feature_names=("trailing", "trailing_sq"),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        available_at=available_at,
    )


def schedule(**overrides: object) -> PrequentialSchedule:
    base = {
        "first_fit_time": START + 20 * DAY,
        "refit_interval_days": 30,
    }
    base.update(overrides)
    return PrequentialSchedule(**base)  # type: ignore[arg-type]


def source(
    dataset: MarketDataset,
    *,
    horizon_hours: int = 24,
    sched: PrequentialSchedule | None = None,
    fitter: object | None = None,
) -> PrequentialForecastSource:
    return PrequentialForecastSource(
        dataset,
        schedule=sched or schedule(),
        fitter=fitter or RidgeForecastFitter(alpha=10.0),  # type: ignore[arg-type]
        feature_indices=(0, 1),
        horizon_hours=horizon_hours,
    )


def daily_times(first_day: int, last_day: int) -> list[np.datetime64]:
    return [START + day * DAY for day in range(first_day, last_day + 1)]


def packet_kwargs() -> dict[str, object]:
    fit = START + 20 * DAY
    return {
        "symbol": "AAAUSDT",
        "symbol_index": 0,
        "horizon_hours": 24,
        "decision_time": fit + 1 * DAY,
        "forecast_available_at": fit + 1 * DAY,
        "return_unit": "log",
        "point_forecast": 0.001,
        "residual_variance": 0.0004,
        "vintage_id": "a" * 64,
        "fit_time": fit,
        "fit_cutoff": fit,
        "max_training_label_available_at": fit - 1 * HOUR,
        "n_training_samples": 10,
        "fitter_identity": "ridge:alpha=10.0",
        "training_digest": "b" * 64,
        "model_digest": "c" * 64,
        "feature_digest": "d" * 64,
    }


def test_schedule_expanding_cutoffs_and_activation() -> None:
    sched = schedule()
    assert sched.cutoff_for(START + 20 * DAY) == START + 20 * DAY
    assert sched.cutoff_for(START + 49 * DAY + 23 * HOUR) == START + 20 * DAY
    assert sched.cutoff_for(START + 50 * DAY) == START + 50 * DAY
    assert sched.cutoffs(START + 80 * DAY) == (
        START + 20 * DAY,
        START + 50 * DAY,
        START + 80 * DAY,
    )
    with pytest.raises(ValueError, match="no vintage is usable"):
        sched.cutoff_for(START + 20 * DAY - HOUR)

    delayed = schedule(activation_delay_hours=2)
    assert delayed.fit_time_for(START + 20 * DAY) == START + 20 * DAY + 2 * HOUR
    assert delayed.cutoff_for(START + 20 * DAY + 2 * HOUR) == START + 20 * DAY
    with pytest.raises(ValueError, match="no vintage is usable"):
        delayed.cutoff_for(START + 20 * DAY + HOUR)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"refit_interval_days": 0}, "refit_interval_days"),
        ({"refit_interval_days": True}, "refit_interval_days"),
        ({"activation_delay_hours": -1}, "activation_delay_hours"),
        ({"first_fit_time": np.datetime64("NaT")}, "first_fit_time"),
    ],
)
def test_schedule_rejects_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        schedule(**kwargs)


def test_packet_accepts_a_valid_invariant_chain() -> None:
    packet = ForecastPacket(**packet_kwargs())  # type: ignore[arg-type]
    assert packet.return_unit == "log"
    assert packet.canonical_json() == ForecastPacket(**packet_kwargs()).canonical_json()  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "overrides, message",
    [
        # label available exactly at fit time (not strictly before)
        (
            {"max_training_label_available_at": START + 20 * DAY},
            "strictly before fit_time",
        ),
        (
            {"max_training_label_available_at": START + 20 * DAY + HOUR},
            "strictly before fit_time",
        ),
        # forecast available before the fit finished
        ({"forecast_available_at": START + 20 * DAY - HOUR}, "precede fit_time"),
        # forecast available after the decision
        (
            {"forecast_available_at": START + 21 * DAY + HOUR},
            "after decision_time",
        ),
        # cutoff later than fit time
        ({"fit_cutoff": START + 20 * DAY + HOUR}, "fit_cutoff"),
        ({"return_unit": "simple"}, "return_unit"),
        ({"point_forecast": float("nan")}, "point_forecast"),
        ({"residual_variance": -1e-9}, "residual_variance"),
        ({"residual_variance": float("inf")}, "residual_variance"),
        ({"horizon_hours": 0}, "horizon_hours"),
        ({"vintage_id": "A" * 64}, "vintage_id"),
        ({"decision_time": np.datetime64("NaT")}, "decision_time"),
        ({"n_training_samples": 0}, "n_training_samples"),
    ],
)
def test_packet_rejects_invariant_violations(
    overrides: dict[str, object], message: str
) -> None:
    kwargs = packet_kwargs()
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=message):
        ForecastPacket(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("horizon_hours", [24, 72])
def test_packets_respect_the_invariant_for_each_horizon(horizon_hours: int) -> None:
    dataset = market()
    src = source(dataset, horizon_hours=horizon_hours)
    seen_vintages: set[str] = set()
    for decision_time, packets in src.iter_packets(daily_times(21, 99)):
        assert len(packets) == len(SYMBOLS)
        for packet in packets:
            assert packet is not None
            assert packet.horizon_hours == horizon_hours
            assert packet.return_unit == "log"
            assert packet.decision_time == decision_time
            assert (
                packet.max_training_label_available_at
                < packet.fit_time
                <= packet.forecast_available_at
                <= packet.decision_time
            )
            assert packet.residual_variance > 0.0
            assert math.isfinite(packet.point_forecast)
            seen_vintages.add(packet.vintage_id)
    # Day 21..99 spans cutoffs at day 20, 50 and 80.
    assert len(seen_vintages) == 3


def test_vintage_is_frozen_between_refits_and_expands_at_refit() -> None:
    src = source(market())
    first = src.packet(0, START + 21 * DAY)
    last_before = src.packet(0, START + 49 * DAY + 23 * HOUR)
    at_refit = src.packet(0, START + 50 * DAY)
    assert first is not None and last_before is not None and at_refit is not None
    assert first.vintage_id == last_before.vintage_id
    assert first.model_digest == last_before.model_digest
    assert first.residual_variance == last_before.residual_variance
    assert first.fit_cutoff == START + 20 * DAY
    assert at_refit.fit_cutoff == START + 50 * DAY
    assert at_refit.vintage_id != first.vintage_id
    assert at_refit.n_training_samples > first.n_training_samples
    assert (
        at_refit.max_training_label_available_at > first.max_training_label_available_at
    )


def test_no_fit_uses_a_cutoff_after_the_requested_decision_time() -> None:
    calls: list[np.datetime64] = []
    inner = RidgeForecastFitter(alpha=10.0)

    class SpyFitter:
        identity = inner.identity

        def fit(self, training):  # type: ignore[no-untyped-def]
            calls.append(training.fit_cutoff)
            return inner.fit(training)

        def predict_rows(self, model, features):  # type: ignore[no-untyped-def]
            return inner.predict_rows(model, features)

    src = source(market(), fitter=SpyFitter())
    decision = START + 40 * DAY
    src.packets_at(decision)
    assert calls == [START + 20 * DAY]
    assert all(call <= decision for call in calls)
    src.packets_at(START + 60 * DAY)
    assert calls == [START + 20 * DAY, START + 50 * DAY]
    # Cached: asking again does not refit.
    src.packets_at(START + 41 * DAY)
    assert len(calls) == 2


@pytest.mark.parametrize("horizon_hours", [24, 72])
def test_future_perturbation_leaves_earlier_packets_byte_identical(
    horizon_hours: int,
) -> None:
    split_day = 65
    split_row = split_day * 24
    baseline = source(market(), horizon_hours=horizon_hours)
    mutated = source(market(mutate_after=split_row), horizon_hours=horizon_hours)
    times = daily_times(21, split_day)
    base_packets = [
        p.canonical_json() for _, ps in baseline.iter_packets(times) for p in ps if p
    ]
    mut_packets = [
        p.canonical_json() for _, ps in mutated.iter_packets(times) for p in ps if p
    ]
    assert len(base_packets) == len(times) * len(SYMBOLS)
    assert base_packets == mut_packets
    # Sanity: the perturbation does change packets after the split.
    later = START + (split_day + 2) * DAY
    assert baseline.packet(0, later) != mutated.packet(0, later)


def _weighted_standardization(
    training: CausalForecastTrainingSet,
) -> tuple[np.ndarray, np.ndarray]:
    x, w = training.features, training.sample_weights
    mean = (w[:, None] * x).sum(axis=0) / w.sum()
    scale = np.sqrt((w[:, None] * (x - mean) ** 2).sum(axis=0) / w.sum())
    return mean, scale


def _lstsq_ridge(
    training: CausalForecastTrainingSet, alpha: float
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Weighted ridge via an augmented least-squares system (independent oracle)."""

    y, w = training.labels, training.sample_weights
    mean, scale = _weighted_standardization(training)
    z = (training.features - mean) / scale
    y_mean = float((w * y).sum() / w.sum())
    root = np.sqrt(w)
    n_features = z.shape[1]
    design = np.vstack([z * root[:, None], np.sqrt(alpha) * np.eye(n_features)])
    target = np.concatenate([(y - y_mean) * root, np.zeros(n_features)])
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return y_mean, coefficients, mean, scale


def test_residual_variance_matches_an_independent_computation() -> None:
    dataset = market()
    cutoff = START + 50 * DAY
    src = source(dataset)
    packet = src.packet(1, cutoff)
    assert packet is not None
    training = build_causal_forecast_training_set(
        dataset, feature_indices=(0, 1), fit_cutoff=cutoff, horizon_hours=24
    )
    intercept, coefficients, mean, scale = _lstsq_ridge(training, alpha=10.0)
    fitted = intercept + ((training.features - mean) / scale) @ coefficients
    residuals = training.labels - fitted
    expected = float(
        np.dot(training.sample_weights, residuals**2) / training.sample_weights.sum()
    )
    assert packet.residual_variance == pytest.approx(expected, rel=1e-8)
    row = int((cutoff - START) / HOUR)
    decision_features = np.asarray(dataset.features[row, 1, :], dtype=np.float64)
    expected_forecast = float(
        intercept + ((decision_features - mean) / scale) @ coefficients
    )
    assert packet.point_forecast == pytest.approx(expected_forecast, rel=1e-8)
    assert packet.n_training_samples == training.n_samples


def test_custom_fitter_protocol_is_supported() -> None:
    class ConstantModel:
        def __init__(self, cutoff: np.datetime64, value: float) -> None:
            self.fit_cutoff = cutoff
            self.value = value

        def predict(self, features: np.ndarray) -> float:
            return self.value

    class ConstantFitter:
        identity = "constant-mean"

        def fit(self, training):  # type: ignore[no-untyped-def]
            mean = float(np.average(training.labels, weights=training.sample_weights))
            return ConstantModel(training.fit_cutoff, mean)

        def predict_rows(self, model, features):  # type: ignore[no-untyped-def]
            return np.full(features.shape[0], model.value)

    dataset = market()
    packet = source(dataset, fitter=ConstantFitter()).packet(2, START + 30 * DAY)
    assert packet is not None
    training = build_causal_forecast_training_set(
        dataset,
        feature_indices=(0, 1),
        fit_cutoff=START + 20 * DAY,
        horizon_hours=24,
    )
    mean = float(np.average(training.labels, weights=training.sample_weights))
    variance = float(
        np.average((training.labels - mean) ** 2, weights=training.sample_weights)
    )
    assert packet.point_forecast == pytest.approx(mean)
    assert packet.residual_variance == pytest.approx(variance)
    assert packet.fitter_identity == "constant-mean"


def test_fitter_with_wrong_cutoff_or_nonfinite_rows_is_rejected() -> None:
    inner = RidgeForecastFitter()

    class WrongCutoff:
        identity = "wrong"

        def fit(self, training):  # type: ignore[no-untyped-def]
            return replace(inner.fit(training), fit_cutoff=training.fit_cutoff - HOUR)

        def predict_rows(self, model, features):  # type: ignore[no-untyped-def]
            return inner.predict_rows(model, features)

    class NanRows:
        identity = "nan"

        def fit(self, training):  # type: ignore[no-untyped-def]
            return inner.fit(training)

        def predict_rows(self, model, features):  # type: ignore[no-untyped-def]
            return np.full(features.shape[0], np.nan)

    with pytest.raises(ValueError, match="cutoff does not match"):
        source(market(), fitter=WrongCutoff()).packet(0, START + 30 * DAY)
    with pytest.raises(ValueError, match="finite prediction"):
        source(market(), fitter=NanRows()).packet(0, START + 30 * DAY)


class RecordingFitter:
    """Ridge fitter that records exactly what the source hands it."""

    identity = "recording-ridge"

    def __init__(self) -> None:
        self.inner = RidgeForecastFitter(alpha=10.0)
        self.received: list[object] = []
        self.batches: list[np.ndarray] = []

    def fit(self, training):  # type: ignore[no-untyped-def]
        self.received.append(training)
        return self.inner.fit(training)

    def predict_rows(self, model, features):  # type: ignore[no-untyped-def]
        self.batches.append(np.array(features))
        return self.inner.predict_rows(model, features)


def test_fitter_receives_only_the_causal_training_set_never_the_dataset() -> None:
    dataset = market()
    recorder = RecordingFitter()
    source(dataset, fitter=recorder).vintage_for(START + 30 * DAY)
    (training,) = recorder.received
    assert isinstance(training, CausalForecastTrainingSet)
    expected = build_causal_forecast_training_set(
        dataset, feature_indices=(0, 1), fit_cutoff=START + 20 * DAY, horizon_hours=24
    )
    assert training.fit_cutoff == START + 20 * DAY
    np.testing.assert_array_equal(training.features, expected.features)
    np.testing.assert_array_equal(training.labels, expected.labels)
    np.testing.assert_array_equal(training.sample_weights, expected.sample_weights)
    assert np.all(training.label_end_times < training.fit_cutoff)
    # No attribute leads back to the dataset or to rows beyond the cutoff.
    assert not any(
        isinstance(getattr(training, name), MarketDataset)
        for name in training.__slots__
    )
    assert not hasattr(training, "dataset")


def test_peeking_fitter_inputs_are_invariant_to_post_cutoff_data() -> None:
    cutoff_row = 20 * 24
    clean, perturbed = RecordingFitter(), RecordingFitter()
    source(market(), fitter=clean).vintage_for(START + 20 * DAY)
    # Prices and features after the cutoff row are replaced wholesale.
    source(market(mutate_after=cutoff_row), fitter=perturbed).vintage_for(
        START + 20 * DAY
    )
    (a,) = clean.received
    (b,) = perturbed.received
    for name in ("features", "labels", "label_end_times", "sample_weights"):
        assert getattr(a, name).tobytes() == getattr(b, name).tobytes(), name


def test_legacy_dataset_taking_fitter_cannot_run() -> None:
    inner = RidgeForecastFitter(alpha=10.0)

    class DatasetFitter:
        identity = "legacy"

        def fit(  # type: ignore[no-untyped-def]
            self,
            dataset,
            *,
            feature_indices,
            fit_symbol_indices,
            fit_cutoff,
            horizon_hours,
        ):
            raise AssertionError("fitters must not be handed the dataset")

        def predict_rows(self, model, features):  # type: ignore[no-untyped-def]
            return inner.predict_rows(model, features)

    with pytest.raises(TypeError):
        source(market(), fitter=DatasetFitter()).packet(0, START + 30 * DAY)


def test_point_forecast_and_residuals_share_the_predict_rows_path() -> None:
    dataset = market()
    spy = RecordingFitter()
    decision = START + 30 * DAY
    src = source(dataset, fitter=spy)
    packet = src.packet(1, decision)
    assert packet is not None
    row = int((decision - START) / HOUR)
    decision_features = np.asarray(dataset.features[row, 1, :], dtype=np.float64)
    (training,) = spy.received
    assert isinstance(training, CausalForecastTrainingSet)
    # The residual batch and the decision row both go through predict_rows.
    np.testing.assert_array_equal(spy.batches[0], training.features)
    np.testing.assert_array_equal(spy.batches[-1], decision_features.reshape(1, -1))
    model = src.vintage_for(decision).model
    assert packet.point_forecast == float(
        spy.inner.predict_rows(model, decision_features.reshape(1, -1))[0]
    )
    assert packet.point_forecast == pytest.approx(
        model.predict(decision_features), rel=1e-12
    )


def test_model_predict_and_fitter_predict_rows_must_agree() -> None:
    inner = RidgeForecastFitter(alpha=10.0)

    class Inconsistent:
        identity = "inconsistent"

        def fit(self, training):  # type: ignore[no-untyped-def]
            return inner.fit(training)

        def predict_rows(self, model, features):  # type: ignore[no-untyped-def]
            return inner.predict_rows(model, features) + 1e-3

    with pytest.raises(ValueError, match="disagree"):
        source(market(), fitter=Inconsistent()).packet(0, START + 30 * DAY)


def test_unusable_decision_inputs_fail_closed() -> None:
    dataset = market()
    src = source(dataset)
    with pytest.raises(ValueError, match="no vintage is usable"):
        src.packet(0, START + 10 * DAY)
    with pytest.raises(ValueError, match="dataset timestamp"):
        src.packet(0, START + 30 * DAY + np.timedelta64(30, "m"))
    with pytest.raises(ValueError, match="symbol_index"):
        src.packet(3, START + 30 * DAY)
    with pytest.raises(ValueError, match="symbol_index"):
        src.packet(True, START + 30 * DAY)  # type: ignore[arg-type]

    unavailable = np.array(dataset.feature_available)
    row = 30 * 24
    unavailable[row, 1, 0] = False
    degraded = replace(
        dataset,
        feature_available=unavailable,
        feature_staleness=np.where(unavailable, 0.0, 1.0).astype(np.float32),
    )
    packets = source(degraded).packets_at(START + 30 * DAY)
    assert packets[0] is not None and packets[2] is not None
    assert packets[1] is None


def test_lagged_availability_activation_delay_only_covers_label_availability() -> None:
    dataset = market(available_lag_hours=1)
    decision = START + 30 * DAY
    # Labels become available exactly at the cutoff, so a zero delay fails closed.
    with pytest.raises(ValueError, match="strictly before the fit time"):
        source(dataset).vintage_for(decision)
    # A one-hour activation delay makes the vintage fit-able and servable...
    delayed = source(dataset, sched=schedule(activation_delay_hours=1))
    vintage = delayed.vintage_for(decision)
    assert vintage.fit_cutoff == START + 20 * DAY
    assert vintage.fit_time == START + 20 * DAY + HOUR
    assert vintage.max_training_label_available_at == START + 20 * DAY
    # ...but the delay does not make a decision row available earlier: a bar
    # whose availability lags its timestamp is never a decision row, and the
    # packet fails closed instead of serving look-ahead features.
    with pytest.raises(ValueError, match="decision row is not available"):
        delayed.packet(0, decision)


def test_lightgbm_fitter_goes_through_the_shared_training_rows(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    seen: dict[str, object] = {}

    class FakeRegressor:
        def __init__(self, **kwargs: object) -> None:
            seen["params"] = kwargs

        def fit(self, features, labels, *, sample_weight):  # type: ignore[no-untyped-def]
            self.mean = float(np.average(labels, weights=sample_weight))
            seen["rows"] = features.shape[0]
            return self

        def predict(self, features):  # type: ignore[no-untyped-def]
            return np.full(features.shape[0], self.mean)

    monkeypatch.setitem(
        sys.modules, "lightgbm", SimpleNamespace(LGBMRegressor=FakeRegressor)
    )
    dataset = market()
    packet = source(dataset, fitter=LightGBMForecastFitter(random_state=3)).packet(
        0, START + 30 * DAY
    )
    assert packet is not None
    assert packet.fitter_identity == "lightgbm:random_state=3"
    assert seen["rows"] == packet.n_training_samples
    assert packet.residual_variance > 0.0


def test_ridge_fitter_rejects_bad_alpha() -> None:
    for alpha in (0.0, -1.0, float("nan"), True):
        with pytest.raises(ValueError, match="alpha"):
            RidgeForecastFitter(alpha=alpha)  # type: ignore[arg-type]
