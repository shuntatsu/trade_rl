"""Forecast errors use an explicit, complete roster and literal price ratios."""

from __future__ import annotations

import importlib
import math
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import MarketCalendarKind
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecasts.simple_return import (
    SimpleReturnRidgeModel,
    SimpleReturnTrainingSet,
)
from trade_rl.strategies.forecasts.simple_stream import (
    FrozenSimpleReturnStream,
    SimpleReturnPacket,
    SimpleReturnVintage,
)
from trade_rl.strategies.forecasts.stream import ForecastBlock
from trade_rl.strategies.forecasts.supervised import CausalForecastTrainingSet
from trade_rl.strategies.forecasts.training_trace import ForecastTrainingTrace


def time(hour: float) -> np.datetime64:
    return np.datetime64("2026-01-01T00:00:00", "ns") + np.timedelta64(
        round(hour * 3_600_000_000_000), "ns"
    )


def market(*, symbols: tuple[str, ...] = ("ALPHA",)) -> MarketDataset:
    close = np.repeat(
        np.array([100, 100, 100, 100, 100, 110, 99, 99, 99.0]).reshape(-1, 1),
        len(symbols),
        axis=1,
    )
    features = np.ones((9, len(symbols), 1))
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=symbols,
        timestamps=np.array([time(h) for h in range(9)]),
        features=features,
        global_features=np.zeros((9, 1)),
        open=close,
        high=close,
        low=close,
        close=close,
        volume=np.ones_like(close),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=bool),
        feature_available=np.ones_like(features, dtype=bool),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8760,
    )


def vintage(*, start: int = 4, stop: int = 7, cutoff: int = 3) -> SimpleReturnVintage:
    # Native frozen declarations, constructed without fitting or prediction.
    train_start = 0 if cutoff == 3 else 3
    trace = ForecastTrainingTrace(
        row_symbols=("ALPHA",),
        start_times=np.array([time(train_start)]),
        end_times=np.array([time(train_start + 1)]),
        start_available_at=np.array([time(train_start)]),
        end_available_at=np.array([time(train_start + 1)]),
        start_close=np.array([100.0]),
        end_close=np.array([100.0]),
    )
    training = SimpleReturnTrainingSet(
        CausalForecastTrainingSet(
            feature_indices=(0,),
            feature_names=("signal",),
            features=np.ones((1, 1)),
            labels=np.zeros(1),
            label_end_times=trace.end_times,
            sample_weights=np.ones(1),
            fit_cutoff=time(cutoff),
            horizon_hours=1,
            trace=trace,
        )
    )
    model = SimpleReturnRidgeModel(
        feature_indices=(0,),
        feature_mean=np.ones(1),
        feature_scale=np.ones(1),
        coefficients=np.zeros(1),
        intercept=0.0,
        horizon_hours=1,
        alpha=1.0,
        n_samples=1,
        fit_cutoff=time(cutoff),
        fit_prefix_marginal_variance=0.0,
    )
    return SimpleReturnVintage(
        ForecastBlock(time(cutoff), time(cutoff + 0.5), time(start), time(stop)),
        training,
        model,
        ("ALPHA",),
    )


def packet(dataset: MarketDataset, owner: SimpleReturnVintage, row: int):
    return SimpleReturnPacket(
        symbol="ALPHA",
        as_of=time(row),
        source_available_at=time(row),
        forecast_available_at=time(row),
        horizon_end=time(row + 1),
        horizon_seconds=3600,
        expected_simple_return=0.0,
        fit_prefix_marginal_variance=0.0,
        vintage_digest=owner.digest,
        feature_values=(1.0,),
        decision_close=float(dataset.close[row, 0]),
    )


def declared_stream(dataset: MarketDataset | None = None, *, prequential=False):
    dataset = market() if dataset is None else dataset
    owners = (
        (vintage(stop=6), vintage(start=6, cutoff=5)) if prequential else (vintage(),)
    )
    # __post_init__ predicts: deliberately bypass it so evaluation cannot rely on
    # rebuilding a stream and tests also exercise incomplete supplied records.
    stream = object.__new__(FrozenSimpleReturnStream)
    object.__setattr__(stream, "dataset_id", dataset.dataset_id)
    object.__setattr__(stream, "vintages", owners)
    object.__setattr__(
        stream,
        "packets",
        tuple(
            packet(dataset, owners[-1] if prequential and r == 6 else owners[0], r)
            for r in (4, 5, 6)
        ),
    )
    return stream


def evaluate(dataset=None, stream=None, **kwargs):
    try:
        api = importlib.import_module("trade_rl.evaluation.forecast_diagnostics")
    except ModuleNotFoundError as error:
        if error.name != "trade_rl.evaluation.forecast_diagnostics":
            raise
        pytest.fail("Missing frozen forecast diagnostics API", pytrace=False)
    return api.evaluate_simple_return_forecasts(
        market() if dataset is None else dataset,
        declared_stream() if stream is None else stream,
        decision_indices=kwargs.get("decision_indices", (4, 5, 6)),
        evaluation_as_of=kwargs.get("evaluation_as_of", time(7)),
    )


def test_literal_residual_oracle_is_packet_weighted_and_does_not_predict(monkeypatch):
    stream = declared_stream()
    before = stream.payload()

    def forbidden(*args, **kwargs):
        pytest.fail("diagnostics must not fit or predict")

    monkeypatch.setattr(SimpleReturnRidgeModel, "predict_selected", forbidden)
    monkeypatch.setattr(SimpleReturnRidgeModel, "predict", forbidden)
    result = evaluate(stream=stream)
    metrics = result.payload()["metrics"]
    assert metrics["n"] == 3
    assert metrics["mean_prediction"] == 0
    assert metrics["mean_realized"] == pytest.approx(0, abs=1e-15)
    assert metrics["bias"] == pytest.approx(0, abs=1e-15)
    assert metrics["mae"] == pytest.approx(1 / 15)
    assert metrics["rmse"] == pytest.approx(math.sqrt(1 / 150))
    assert [r["residual"] for r in result.payload()["records"]] == pytest.approx(
        [-0.10, +0.10, 0]
    )
    assert stream.payload() == before
    assert all(
        v.payload()["calibration_state"] == "uncalibrated" for v in stream.vintages
    )


def test_missing_packet_cannot_select_a_favorable_subset():
    stream = declared_stream()
    object.__setattr__(stream, "packets", stream.packets[1:])
    with pytest.raises(ValueError, match="coverage"):
        evaluate(stream=stream)


def test_public_native_stream_constructor_is_compatible(monkeypatch):
    declared = declared_stream()
    # Only this compatibility setup invokes the existing frozen constructor's
    # projection checks; the diagnostic itself must make no prediction calls.
    stream = FrozenSimpleReturnStream(
        declared.dataset_id, declared.vintages, declared.packets
    )

    def forbidden(*args, **kwargs):
        pytest.fail("diagnostics must not predict")

    monkeypatch.setattr(SimpleReturnRidgeModel, "predict_selected", forbidden)
    assert evaluate(stream=stream).payload()["metrics"]["n"] == 3


@pytest.mark.parametrize("symbols", ((), ("ALPHA", "ALPHA")))
def test_vintage_symbol_roster_cannot_be_empty_or_duplicate(symbols):
    stream = declared_stream()
    owner = stream.vintages[0]
    object.__setattr__(owner, "prediction_symbols", symbols)
    object.__setattr__(owner, "_digest", content_digest(owner._content()))
    with pytest.raises(ValueError, match="prediction_symbols"):
        evaluate(stream=stream)


def test_each_block_uses_its_own_symbol_roster_instead_of_the_stream_union():
    dataset = market(symbols=("ALPHA", "BETA"))
    stream = declared_stream(dataset, prequential=True)
    first, second = stream.vintages
    second = replace(second, prediction_symbols=("ALPHA", "BETA"))
    object.__setattr__(stream, "vintages", (first, second))
    object.__setattr__(
        stream,
        "packets",
        stream.packets[:2]
        + (
            packet(dataset, second, 6),
            replace(packet(dataset, second, 6), symbol="BETA"),
        ),
    )
    result = evaluate(dataset=dataset, stream=stream).payload()
    assert result["metrics"]["n"] == 4
    assert [
        (s["decision_index"], s["symbol"]) for s in result["source_projection"]
    ] == [(4, "ALPHA"), (5, "ALPHA"), (6, "ALPHA"), (6, "BETA")]


@pytest.mark.parametrize(
    "roster", ((), [4, 5], (True,), (4.0,), (np.int64(4),), (5, 4), (4, 4), (-1,), (9,))
)
def test_roster_requires_explicit_sorted_unique_native_indices(roster):
    with pytest.raises(ValueError, match="decision_indices"):
        evaluate(decision_indices=roster)


@pytest.mark.parametrize(
    "cutoff", (np.datetime64("NaT"), np.datetime64("3000-01-01"), "2026-01-01", 7)
)
def test_cutoff_requires_an_exact_timestamp(cutoff):
    with pytest.raises(ValueError, match="evaluation_as_of"):
        evaluate(evaluation_as_of=cutoff)


def test_wrong_dataset_id_is_rejected():
    with pytest.raises(ValueError, match="identity"):
        evaluate(dataset=replace(market(), dataset_id="b" * 64))


@pytest.mark.parametrize("roster", ((3,), (7,)))
def test_requested_block_gaps_are_rejected(roster):
    with pytest.raises(ValueError, match="exactly one vintage"):
        evaluate(decision_indices=roster)


@pytest.mark.parametrize(
    "case", ("empty", "duplicate", "unknown_symbol", "unknown_vintage", "unknown_time")
)
def test_invalid_packet_coverage_is_rejected(case):
    stream = declared_stream()
    packets = stream.packets
    if case == "empty":
        packets = ()
    elif case == "duplicate":
        packets += (packets[0],)
    elif case == "unknown_symbol":
        packets = (replace(packets[0], symbol="BETA"),) + packets[1:]
    elif case == "unknown_vintage":
        packets = (replace(packets[0], vintage_digest="b" * 64),) + packets[1:]
    else:
        packets = (
            replace(
                packets[0],
                as_of=time(4.5),
                source_available_at=time(4.5),
                forecast_available_at=time(4.5),
                horizon_end=time(5.5),
            ),
        ) + packets[1:]
    object.__setattr__(stream, "packets", packets)
    with pytest.raises(ValueError, match="coverage"):
        evaluate(stream=stream)


def test_exact_endpoint_is_required_even_when_neighboring_prices_exist():
    dataset = market()
    times = dataset.timestamps.copy()
    times[7] = time(7.5)
    dataset = replace(
        dataset,
        timestamps=times,
        calendar_kind=MarketCalendarKind.SESSION,
        nominal_bar_hours=1,
        available_at=None,
        information_available=None,
    )
    with pytest.raises(ValueError, match="exact.*endpoint"):
        evaluate(dataset=dataset)


@pytest.mark.parametrize(
    "field",
    (
        "close",
        "features",
        "feature_names",
        "publication",
        "feature_available",
        "information_available",
    ),
)
def test_selected_snapshot_drift_or_unavailability_is_rejected(field):
    dataset = market()
    if field == "close":
        close = dataset.close.copy()
        close[4, 0] = 105
        dataset = replace(dataset, close=close, open=close, high=close, low=close)
    elif field == "features":
        features = dataset.features.copy()
        features[4, 0, 0] = 2
        dataset = replace(dataset, features=features)
    elif field == "feature_names":
        dataset = replace(dataset, feature_names=("different_signal",))
    elif field == "publication":
        available = dataset.timestamps[:, None].copy()
        available[4, 0] = time(4.5)
        dataset = replace(dataset, available_at=available, information_available=None)
    elif field == "feature_available":
        available = dataset.feature_available.copy()
        available[4, 0, 0] = False
        dataset = replace(dataset, feature_available=available, feature_staleness=None)
    else:
        information = np.ones_like(dataset.close, dtype=bool)
        information[4, 0] = False
        dataset = replace(dataset, information_available=information)
    with pytest.raises(ValueError, match="snapshot|feature names"):
        evaluate(dataset=dataset)


def test_delayed_forecast_cannot_be_used_at_its_earlier_decision():
    stream = declared_stream()
    owner = replace(
        stream.vintages[0],
        block=replace(stream.vintages[0].block, inference_delay_seconds=1),
    )
    object.__setattr__(stream, "vintages", (owner,))
    object.__setattr__(
        stream,
        "packets",
        tuple(
            replace(
                p,
                vintage_digest=owner.digest,
                forecast_available_at=p.as_of + np.timedelta64(1, "s"),
            )
            for p in stream.packets
        ),
    )
    with pytest.raises(ValueError, match="unavailable"):
        evaluate(stream=stream)


@pytest.mark.parametrize(
    "case",
    ("cutoff", "endpoint_information", "endpoint_publication", "nat_publication"),
)
def test_endpoint_maturity_and_publication_are_explicit(case):
    dataset = market()
    kwargs = {}
    if case == "cutoff":
        kwargs["evaluation_as_of"] = time(6)
    elif case == "endpoint_information":
        information = np.ones_like(dataset.close, dtype=bool)
        information[7, 0] = False
        dataset = replace(dataset, information_available=information)
    elif case == "endpoint_publication":
        available = dataset.timestamps[:, None].copy()
        available[7, 0] = time(7.5)
        dataset = replace(dataset, available_at=available, information_available=None)
    else:
        # A post-construction corruption must not be admitted by the reader.
        available = dataset.timestamps[:, None].copy()
        available[7, 0] = np.datetime64("NaT")
        object.__setattr__(dataset, "available_at", available)
    with pytest.raises(ValueError, match="endpoint|source clock"):
        evaluate(dataset=dataset, **kwargs)


def test_prequential_fit_inside_roster_is_allowed_and_pooled_rmse_is_not_group_mean():
    stream = declared_stream(prequential=True)
    assert stream.vintages[1].block.fit_cutoff > time(4)
    result = evaluate(stream=stream).payload()
    groups = {g["vintage_digest"]: g["metrics"] for g in result["groups"]}
    assert groups[stream.vintages[0].digest]["n"] == 2
    assert groups[stream.vintages[1].digest]["n"] == 1
    assert groups[stream.vintages[0].digest]["rmse"] == pytest.approx(0.1)
    assert groups[stream.vintages[1].digest]["rmse"] == 0
    assert result["metrics"]["rmse"] == pytest.approx(math.sqrt(1 / 150))
    assert result["metrics"]["rmse"] != pytest.approx(
        sum(g["rmse"] for g in groups.values()) / 2
    )


def test_cached_vintage_content_identity_cannot_hide_model_drift():
    stream = declared_stream()
    object.__setattr__(stream.vintages[0].model, "intercept", 0.25)
    with pytest.raises(ValueError, match="content"):
        evaluate(stream=stream)


def test_horizon_clock_overflow_is_not_silently_wrapped():
    stream = declared_stream()
    object.__setattr__(stream.packets[0], "horizon_seconds", 2**63)
    with pytest.raises(ValueError, match="nanosecond"):
        evaluate(stream=stream)


def test_vintage_labels_must_be_published_strictly_before_its_own_cutoff():
    stream = declared_stream()
    owner = stream.vintages[0]
    object.__setattr__(owner.training.trace, "end_available_at", np.array([time(3)]))
    object.__setattr__(owner, "_digest", content_digest(owner._content()))
    with pytest.raises(ValueError, match="causal fit"):
        evaluate(stream=stream)


def test_immutable_report_binds_observed_projection_roster_and_cutoff():
    result = evaluate()
    before = result.digest
    copy = result.payload()
    copy["records"][0]["packet"]["feature_values"][0] = 500
    copy["source_projection"][0]["endpoint_close"] = 500
    assert result.digest == before
    with pytest.raises(FrozenInstanceError):
        result._payload_json = "{}"
    assert evaluate(decision_indices=(4,)).digest != before
    assert evaluate(evaluation_as_of=time(8)).digest != before
    assert result.payload()["source_projection_digest"] == content_digest(
        result.payload()["source_projection"]
    )


def test_outside_roster_values_do_not_enter_error_arithmetic():
    dataset = market()
    close = dataset.close.copy()
    close[8, 0] = 1000
    changed = replace(
        dataset, dataset_id="b" * 64, close=close, open=close, high=close, low=close
    )
    original = evaluate()
    current = evaluate(dataset=changed, stream=declared_stream(changed))
    assert current.payload()["metrics"] == original.payload()["metrics"]
    assert (
        current.payload()["source_projection"]
        == original.payload()["source_projection"]
    )
    assert current.digest != original.digest  # Whole-artifact identity is separate.


@pytest.mark.parametrize("case", ("ratio", "squared_residual"))
def test_nonfinite_error_arithmetic_fails_closed(case):
    dataset = market()
    stream = declared_stream()
    if case == "ratio":
        close = dataset.close.copy()
        close[6, 0], close[7, 0] = 1e-308, 1e308
        dataset = replace(dataset, close=close, open=close, high=close, low=close)
        object.__setattr__(
            stream,
            "packets",
            tuple(
                replace(p, decision_close=float(close[r, 0]))
                for p, r in zip(stream.packets, (4, 5, 6), strict=True)
            ),
        )
    else:
        object.__setattr__(
            stream,
            "packets",
            tuple(replace(p, expected_simple_return=1e308) for p in stream.packets),
        )
    with pytest.raises(ValueError, match="finite"):
        evaluate(dataset=dataset, stream=stream)
