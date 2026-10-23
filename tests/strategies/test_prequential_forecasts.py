from __future__ import annotations

import importlib
import json
from dataclasses import replace
from types import ModuleType

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.strategies import forecasts


def time(hour: float) -> np.datetime64:
    return np.datetime64("2026-01-01T00:00:00", "ns") + np.timedelta64(
        round(hour * 3_600_000_000_000), "ns"
    )


def market(*, n_bars: int = 20, quadratic: bool = False) -> MarketDataset:
    phase = np.arange(n_bars, dtype=np.float64)
    log_price = (
        0.001 * phase**2
        if quadratic
        else np.where(phase <= 8, 0.01 * phase, 0.08 - 0.02 * (phase - 8))
    )
    close = (100.0 * np.exp(log_price)).reshape(-1, 1)
    features = (phase if quadratic else np.ones(n_bars)).reshape(-1, 1, 1)
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=("BTCUSDT",),
        timestamps=time(0) + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((n_bars, 1)),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros((n_bars, 1)),
        tradable=np.ones((n_bars, 1), dtype=np.bool_),
        feature_available=np.ones_like(features, dtype=np.bool_),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def load_api() -> ModuleType:
    try:
        return importlib.import_module("trade_rl.strategies.forecasts.prequential")
    except ModuleNotFoundError as error:
        if error.name != "trade_rl.strategies.forecasts.prequential":
            raise
        pytest.fail("Missing connected prequential forecast API", pytrace=False)


def blocks(api: ModuleType) -> tuple:
    return (
        api.ForecastBlock(
            fit_cutoff=time(8),
            model_fit_time=time(8.5),
            prediction_start=time(9),
            prediction_stop=time(14),
            inference_delay_seconds=1_800,
        ),
        api.ForecastBlock(
            fit_cutoff=time(14),
            model_fit_time=time(14.5),
            prediction_start=time(15),
            prediction_stop=time(20),
        ),
    )


def fit(api: ModuleType, dataset: MarketDataset | None = None):
    return api.fit_prequential_ridge(
        market() if dataset is None else dataset,
        blocks=blocks(api),
        feature_indices=(0,),
        horizon_hours=2,
        alpha=1.0,
    )


def test_two_real_fits_produce_next_block_log_forecasts_without_future_labels() -> None:
    api = load_api()
    stream = fit(api)
    assert len(stream.vintages) == 2
    assert [vintage.training.n_samples for vintage in stream.vintages] == [6, 12]
    # Constant features make each intercept the hand-calculated mean log label.
    # The second prefix contains seven +.02, one -.01, and four -.04 labels.
    for vintage, expected in zip(stream.vintages, (0.02, -0.0025), strict=True):
        assert isinstance(vintage.model, forecasts.RidgeForecastModel)
        assert vintage.model.intercept == pytest.approx(expected, abs=1e-12)
        np.testing.assert_allclose(vintage.model.coefficients, [0.0], atol=1e-12)
        assert vintage.training.label_end_times.max() < vintage.block.fit_cutoff
    assert [packet.as_of for packet in stream.packets] == [
        *(time(hour) for hour in range(9, 14)),
        *(time(hour) for hour in range(15, 20)),
    ]
    for packet in stream.packets:
        expected = 0.02 if packet.as_of < time(14) else -0.0025
        assert packet.forecast_log_return == pytest.approx(expected, abs=1e-12)
        assert packet.horizon_seconds == 7_200
        assert packet.horizon_end == packet.as_of + np.timedelta64(2, "h")
        assert packet.symbol == "BTCUSDT"
    assert stream.packets[-1].horizon_end == time(21)  # Dataset ends at hour 19.


def test_future_only_suffix_cannot_change_causal_stream() -> None:
    api = load_api()
    dataset = market(n_bars=24)
    changed_close = dataset.close.copy()
    changed_close[20:] *= 7.0
    changed_features = dataset.features.copy()
    changed_features[20:] = -123.0
    available = np.broadcast_to(dataset.timestamps[:, None], dataset.close.shape).copy()
    available[20:] += np.timedelta64(3, "h")
    changed = replace(
        dataset,
        dataset_id="b" * 64,
        open=changed_close.copy(),
        high=changed_close.copy(),
        low=changed_close.copy(),
        close=changed_close,
        features=changed_features,
        available_at=available,
        information_available=available <= dataset.timestamps[:, None],
    )
    baseline, mutated = fit(api, dataset), fit(api, changed)
    assert baseline.causal_scope_digest == mutated.causal_scope_digest
    assert [vintage.digest for vintage in baseline.vintages] == [
        vintage.digest for vintage in mutated.vintages
    ]
    assert [packet.digest for packet in baseline.packets] == [
        packet.digest for packet in mutated.packets
    ]
    assert baseline.payload()["dataset_id"] != mutated.payload()["dataset_id"]


def test_artifact_roundtrip_preserves_real_forecasts() -> None:
    api = load_api()
    original = fit(api)
    payload = json.loads(json.dumps(original.payload(), allow_nan=False))
    assert payload["schema"]
    assert payload["availability_semantics"] == "declared_simulation_v1"
    restored = api.FrozenForecastStream.from_payload(
        payload, expected_digest=original.digest
    )
    assert restored.payload() == original.payload()
    assert restored.digest == original.digest
    assert restored.causal_scope_digest == original.causal_scope_digest
    for exported in (
        "ForecastBlock",
        "FrozenForecastStream",
        "fit_prequential_ridge",
    ):
        assert getattr(forecasts, exported) is getattr(api, exported)
    for vintage in restored.vintages:
        for array in (vintage.model.coefficients, vintage.training.features):
            with pytest.raises(ValueError):
                array.setflags(write=True)
    payload["vintages"][0]["model"]["coefficients"][0] += 1.0
    assert restored.payload() == original.payload()


@pytest.mark.parametrize("component", ("model", "training", "block", "packet"))
def test_reader_rejects_tampering_even_with_rehashed_outer_payload(
    component: str,
) -> None:
    api = load_api()
    stream = fit(api)
    payload = json.loads(json.dumps(stream.payload()))
    if component == "model":
        payload["vintages"][0]["model"]["coefficients"][0] += 1.0
    elif component == "training":
        payload["vintages"][0]["training"]["labels"][0] += 1.0
    elif component == "block":
        payload["vintages"][0]["block"]["model_fit_time"] = "2026-01-02T00:00:00"
    else:
        payload["packets"][0]["forecast_available_at"] = "2025-12-31T00:00:00"
    with pytest.raises(ValueError):
        api.FrozenForecastStream.from_payload(payload, expected_digest=stream.digest)
    with pytest.raises(ValueError):
        api.FrozenForecastStream.from_payload(
            payload, expected_digest=content_digest(payload)
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("fit_cutoff", np.datetime64("NaT")),
        ("model_fit_time", time(7)),
        ("prediction_start", time(8)),
        ("prediction_stop", time(9)),
        ("inference_delay_seconds", -1),
        ("inference_delay_seconds", float("nan")),
    ),
)
def test_block_rejects_invalid_clock(field: str, value: object) -> None:
    api = load_api()
    arguments = {
        "fit_cutoff": time(8),
        "model_fit_time": time(8.5),
        "prediction_start": time(9),
        "prediction_stop": time(14),
    }
    arguments[field] = value
    with pytest.raises(ValueError):
        api.ForecastBlock(**arguments)


def test_schedule_rejects_overlapping_prediction_blocks() -> None:
    api = load_api()
    overlap = api.ForecastBlock(
        fit_cutoff=time(12),
        model_fit_time=time(12.5),
        prediction_start=time(13),
        prediction_stop=time(18),
    )
    with pytest.raises(ValueError):
        api.fit_prequential_ridge(
            market(),
            blocks=(blocks(api)[0], overlap),
            feature_indices=(0,),
            horizon_hours=2,
        )


def test_vintage_roster_and_feature_names_are_frozen_from_caller_lists() -> None:
    api = load_api()
    vintage = fit(api).vintages[0]
    symbols = list(vintage.prediction_symbols)
    names = list(vintage.training.feature_names)
    frozen = replace(
        vintage,
        prediction_symbols=symbols,
        training=replace(vintage.training, feature_names=names),
    )
    digest = frozen.digest
    symbols.append("ETHUSDT")
    names[0] = "different"
    assert frozen.prediction_symbols == ("BTCUSDT",)
    assert frozen.training.feature_names == ("signal",)
    assert frozen.digest == digest


def test_packet_validation_needs_only_selected_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = load_api()
    original = fit(api)
    vintage = original.vintages[0]
    sparse = replace(
        vintage,
        training=replace(vintage.training, feature_indices=(10**12,)),
        model=replace(vintage.model, feature_indices=(10**12,)),
    )
    packets = tuple(
        replace(packet, vintage_digest=sparse.digest)
        for packet in original.packets
        if packet.vintage_digest == vintage.digest
    )

    def no_dense_allocation(*args: object, **kwargs: object) -> None:
        pytest.fail("packet validation must not allocate unused feature columns")

    monkeypatch.setattr(api.np, "zeros", no_dense_allocation)
    stream = api.FrozenForecastStream(original.dataset_id, (sparse,), packets)
    restored = api.FrozenForecastStream.from_payload(
        stream.payload(), expected_digest=stream.digest
    )
    assert restored.packets[0].forecast_log_return == pytest.approx(0.02)
