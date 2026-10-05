from __future__ import annotations

import importlib
import json
import math
from dataclasses import replace
from types import ModuleType

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.strategies.forecasts.prequential import fit_prequential_ridge
from trade_rl.strategies.forecasts.stream import ForecastBlock


def time(hour: float) -> np.datetime64:
    return np.datetime64("2026-02-01T00:00:00", "ns") + np.timedelta64(
        round(hour * 3_600_000_000_000), "ns"
    )


def market(*, n_bars: int = 18) -> MarketDataset:
    close = np.array(
        [100.0, 200.0, 100.0, 200.0, 100.0]
        + [125.0 if hour % 2 else 100.0 for hour in range(5, n_bars)]
    ).reshape(-1, 1)
    features = np.ones((n_bars, 1, 1))
    return make_market(close, features)


def make_market(
    close: np.ndarray,
    features: np.ndarray,
    *,
    symbols: tuple[str, ...] = ("ALPHA",),
    feature_available: np.ndarray | None = None,
) -> MarketDataset:
    n_bars = close.shape[0]
    return MarketDataset(
        dataset_id="a" * 64,
        symbols=symbols,
        timestamps=time(0) + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=features,
        global_features=np.zeros((n_bars, 1)),
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        volume=np.full_like(close, 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=(
            np.ones_like(features, dtype=np.bool_)
            if feature_available is None
            else feature_available
        ),
        feature_names=("signal",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
    )


def blocks(*, delay_seconds: int = 0) -> tuple[ForecastBlock, ...]:
    return (
        ForecastBlock(time(5), time(5.25), time(6), time(8), delay_seconds),
        ForecastBlock(time(9), time(9.25), time(10), time(14)),
    )


def load_api() -> ModuleType:
    try:
        return importlib.import_module(
            "trade_rl.strategies.forecasts.simple_prequential"
        )
    except ModuleNotFoundError as error:
        if error.name != "trade_rl.strategies.forecasts.simple_prequential":
            raise
        pytest.fail("Missing causal simple-return prequential API", pytrace=False)


def fit(api: ModuleType, dataset: MarketDataset | None = None):
    return api.fit_prequential_simple_ridge(
        market() if dataset is None else dataset,
        blocks=blocks(),
        feature_indices=(0,),
        horizon_hours=1,
        alpha=1.0,
    )


@pytest.mark.parametrize("unit", ("h", "s", "ms"))
def test_valid_timestamp_storage_units_preserve_the_same_frozen_stream(
    unit: str,
) -> None:
    dataset = market()
    coarse = replace(
        dataset, timestamps=dataset.timestamps.astype(f"datetime64[{unit}]")
    )
    assert fit(load_api(), coarse).payload() == fit(load_api(), dataset).payload()


def test_direct_simple_return_labels_and_variance_use_only_mature_fit_rows() -> None:
    api = load_api()
    stream = fit(api)
    independent_labels = (
        np.array([1.0, -0.5, 1.0, -0.5]),
        np.array([1.0, -0.5, 1.0, -0.5, 0.25, -0.2, 0.25, -0.2]),
    )
    assert len(stream.vintages) == 2
    for vintage, labels in zip(stream.vintages, independent_labels, strict=True):
        np.testing.assert_allclose(vintage.training.labels, labels, atol=1e-15)
        assert vintage.training.n_samples == labels.size
        assert vintage.training.label_end_times.max() < vintage.block.fit_cutoff
        assert vintage.model.intercept == pytest.approx(float(np.mean(labels)))
        np.testing.assert_allclose(vintage.model.coefficients, [0.0], atol=1e-15)
        for packet in stream.packets:
            if packet.vintage_digest != vintage.digest:
                continue
            assert packet.expected_simple_return == pytest.approx(
                float(np.mean(labels))
            )
            assert packet.fit_prefix_marginal_variance == pytest.approx(
                float(np.var(labels))
            )
            assert packet.return_unit == "expected_simple_return"
            assert packet.valuation_basis == "same_close_price_return"
            assert packet.variance_kind == "fit_prefix_marginal_variance"
            assert packet.forecast_kind == "regularized_linear_projection"
            assert packet.forecast_available_at == packet.as_of
            assert packet.horizon_seconds == 3_600
            assert packet.horizon_end == packet.as_of + np.timedelta64(1, "h")
            assert (
                packet.decision_close
                == market().close[
                    int((packet.as_of - time(0)) / np.timedelta64(1, "h")), 0
                ]
            )
    assert [packet.as_of for packet in stream.packets] == [
        time(6),
        time(7),
        time(10),
        time(11),
        time(12),
        time(13),
    ]
    assert stream.vintages[0].model.intercept != pytest.approx(
        np.expm1(np.mean(np.log1p(independent_labels[0])))
    )


def test_selector_is_called_once_for_each_declared_mature_prefix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = load_api()
    supervised = importlib.import_module("trade_rl.strategies.forecasts.supervised")
    simple_stream = importlib.import_module(
        "trade_rl.strategies.forecasts.simple_stream"
    )
    original = supervised.build_causal_forecast_training_set
    selected_cutoffs: list[np.datetime64] = []

    def select(dataset: MarketDataset, **kwargs: object):
        selected_cutoffs.append(kwargs["fit_cutoff"])
        return original(dataset, **kwargs)

    # Patch the sole selector and any direct alias owned by the new capability.
    for module in (supervised, api, simple_stream):
        for name, value in list(vars(module).items()):
            if value is original:
                monkeypatch.setattr(module, name, select)
    stream = fit(api)
    assert selected_cutoffs == [time(5), time(9)]
    assert [vintage.training.n_samples for vintage in stream.vintages] == [4, 8]


def test_varying_features_predict_the_independent_regularized_simple_projection() -> (
    None
):
    api = load_api()
    dataset = market()
    features = np.arange(dataset.n_bars, dtype=np.float64).reshape(-1, 1, 1)
    features[6:8, 0, 0] = [4.0, 5.0]
    dataset = replace(dataset, features=features)
    stream = api.fit_prequential_simple_ridge(
        dataset,
        blocks=(blocks()[0],),
        feature_indices=(0,),
        horizon_hours=1,
        alpha=1.0,
    )
    # For x=0,1,2,3 and y=1,-.5,1,-.5: x mean=1.5, variance=1.25,
    # centered cross-product=-1.5, standardized Gram=4 and Ridge alpha=1.
    model = stream.vintages[0].model
    np.testing.assert_allclose(model.feature_mean, [1.5], atol=1e-15)
    np.testing.assert_allclose(model.feature_scale, [math.sqrt(1.25)], atol=1e-15)
    np.testing.assert_allclose(
        model.coefficients, [-1.5 / (5.0 * math.sqrt(1.25))], atol=1e-15
    )
    assert [p.expected_simple_return for p in stream.packets] == pytest.approx(
        [0.25 - (4.0 - 1.5) * 0.24, 0.25 - (5.0 - 1.5) * 0.24]
    )
    assert all(p.fit_prefix_marginal_variance == 0.5625 for p in stream.packets)


def test_direct_labels_preserve_legacy_log_training_and_endpoint_trace() -> None:
    api = load_api()
    dataset = market()
    before = fit_prequential_ridge(
        dataset, blocks=blocks(), feature_indices=(0,), horizon_hours=1
    )
    stream = fit(api, dataset)
    after = fit_prequential_ridge(
        dataset, blocks=blocks(), feature_indices=(0,), horizon_hours=1
    )
    assert before.payload() == after.payload()
    assert before.digest == after.digest
    for simple_vintage, log_vintage in zip(
        stream.vintages, before.vintages, strict=True
    ):
        log_training = simple_vintage.training.log_training
        assert log_training.scope_payload() == log_vintage.training.scope_payload()
        trace = log_training.trace
        assert trace is not None
        np.testing.assert_array_equal(
            simple_vintage.training.labels, trace.end_close / trace.start_close - 1.0
        )
        np.testing.assert_array_equal(
            log_training.labels,
            np.array(
                [
                    math.log(end / start)
                    for start, end in zip(
                        trace.start_close, trace.end_close, strict=True
                    )
                ]
            ),
        )
        assert log_vintage.digest != simple_vintage.digest
    assert stream.packets[0].return_unit != before.packets[0].return_unit
    assert stream.packets[0].expected_simple_return == pytest.approx(0.25)
    assert before.packets[0].forecast_log_return == pytest.approx(0.0, abs=1e-15)


def test_unpublished_endpoint_and_unavailable_feature_are_excluded() -> None:
    api = load_api()
    dataset = market()
    available = dataset.resolved_array("available_at").copy()
    available[4, 0] = time(5)
    information = dataset.resolved_array("information_available").copy()
    information[4, 0] = False
    feature_available = dataset.feature_available.copy()
    feature_available[2, 0, 0] = False
    changed = replace(
        dataset,
        dataset_id="b" * 64,
        available_at=available,
        information_available=information,
        feature_available=feature_available,
        feature_staleness=None,
    )
    stream = fit(api, changed)
    first = stream.vintages[0].training
    np.testing.assert_allclose(first.labels, [1.0, -0.5], atol=1e-15)
    assert first.n_samples == 2
    trace = first.log_training.trace
    assert trace is not None
    np.testing.assert_array_equal(trace.start_times, [time(0), time(1)])
    assert np.max(trace.label_available_times) < time(5)
    assert stream.packets[0].expected_simple_return == pytest.approx(0.25)


def test_unequal_symbol_rows_have_equal_total_weight_for_mean_and_variance() -> None:
    api = load_api()
    single = market()
    n_bars = single.n_bars
    close = np.column_stack((single.close[:, 0], 100.0 * 2.0 ** np.arange(n_bars)))
    features = np.ones((n_bars, 2, 1))
    available = np.ones_like(features, dtype=np.bool_)
    available[2:4, 1, 0] = False
    dataset = make_market(
        close,
        features,
        symbols=("ALPHA", "BETA"),
        feature_available=available,
    )
    stream = fit(api, dataset)
    first = stream.vintages[0].training
    trace = first.log_training.trace
    assert trace is not None
    assert trace.row_symbols.count("ALPHA") == 4
    assert trace.row_symbols.count("BETA") == 2
    weights = first.sample_weights
    alpha_mask = np.array([symbol == "ALPHA" for symbol in trace.row_symbols])
    assert weights[alpha_mask].sum() == pytest.approx(weights[~alpha_mask].sum())
    assert stream.vintages[0].model.intercept == pytest.approx(0.625)
    assert stream.vintages[0].model.intercept != pytest.approx(
        float(np.mean(first.labels))
    )
    for packet in stream.packets:
        if packet.vintage_digest == stream.vintages[0].digest:
            assert packet.expected_simple_return == pytest.approx(0.625)
            assert packet.fit_prefix_marginal_variance == pytest.approx(0.421875)
    subset = api.fit_prequential_simple_ridge(
        dataset,
        blocks=blocks(),
        feature_indices=(0,),
        fit_symbol_indices=(0,),
        horizon_hours=1,
    )
    assert subset.vintages[0].model.intercept == pytest.approx(0.25)
    assert subset.vintages[0].training.trace.row_symbols == ("ALPHA",) * 4
    assert {p.symbol for p in subset.packets} == {"ALPHA", "BETA"}


def test_future_suffix_cannot_change_published_causal_identities() -> None:
    api = load_api()
    dataset = market()
    close = dataset.close.copy()
    close[14:] *= 11.0
    features = dataset.features.copy()
    features[14:] = -9_999.0
    available = dataset.resolved_array("available_at").copy()
    available[14:] += np.timedelta64(7, "h")
    information = dataset.resolved_array("information_available").copy()
    information[14:] = False
    dividend = dataset.resolved_array("dividend").copy()
    dividend[14:] = 1_000.0
    split = dataset.resolved_array("split_factor").copy()
    split[14:] = 8.0
    changed = replace(
        dataset,
        dataset_id="b" * 64,
        open=close.copy(),
        high=close.copy(),
        low=close.copy(),
        close=close,
        features=features,
        available_at=available,
        information_available=information,
        dividend=dividend,
        split_factor=split,
    )
    original, poisoned = fit(api, dataset), fit(api, changed)
    assert original.causal_scope_digest == poisoned.causal_scope_digest
    assert [v.digest for v in original.vintages] == [
        v.digest for v in poisoned.vintages
    ]
    assert [p.snapshot_digest for p in original.packets] == [
        p.snapshot_digest for p in poisoned.packets
    ]
    assert [p.digest for p in original.packets] == [p.digest for p in poisoned.packets]
    assert original.digest != poisoned.digest  # The enclosing dataset identity changed.
    assert original.packets[-1].horizon_end == time(14)


def test_declared_delayed_simulation_packets_remain_valid_streams() -> None:
    api = load_api()
    stream = api.fit_prequential_simple_ridge(
        market(),
        blocks=blocks(delay_seconds=1_800),
        feature_indices=(0,),
        horizon_hours=1,
    )
    first = stream.packets[0]
    assert first.as_of == time(6)
    assert first.forecast_available_at == time(6.5)
    restored = api.FrozenSimpleReturnStream.from_payload(
        stream.payload(), expected_digest=stream.digest
    )
    assert restored.packets[0].forecast_available_at == time(6.5)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("expected_simple_return", -1.0001),
        ("expected_simple_return", float("nan")),
        ("expected_simple_return", float("inf")),
        ("expected_simple_return", True),
        ("fit_prefix_marginal_variance", -1e-15),
        ("fit_prefix_marginal_variance", float("nan")),
        ("fit_prefix_marginal_variance", float("inf")),
        ("fit_prefix_marginal_variance", True),
        ("decision_close", 0.0),
        ("decision_close", -1.0),
        ("decision_close", float("nan")),
        ("decision_close", float("inf")),
        ("decision_close", True),
        ("as_of", np.datetime64("NaT")),
        ("source_available_at", time(6.5)),
        ("forecast_available_at", time(5)),
        ("horizon_seconds", 0),
        ("horizon_seconds", True),
        ("horizon_end", time(8)),
        ("return_unit", "log_return"),
        ("valuation_basis", "total_return"),
        ("variance_kind", "conditional_variance"),
        ("forecast_kind", "conditional_mean"),
        ("feature_values", (float("nan"),)),
        ("feature_values", (True,)),
    ),
)
def test_packet_rejects_invalid_numbers_clocks_and_meanings(
    field: str,
    value: object,
) -> None:
    api = load_api()
    with pytest.raises(ValueError):
        replace(fit(api).packets[0], **{field: value})


@pytest.mark.parametrize(
    "component",
    (
        "prediction",
        "variance",
        "feature_count",
        "symbol",
        "owner",
        "block",
        "recipe",
        "horizon",
    ),
)
def test_stream_rejects_packet_that_disagrees_with_frozen_owner(component: str) -> None:
    api = load_api()
    stream = fit(api)
    packet = stream.packets[0]
    if component == "prediction":
        altered = replace(packet, expected_simple_return=0.3)
    elif component == "variance":
        altered = replace(packet, fit_prefix_marginal_variance=0.7)
    elif component == "feature_count":
        altered = replace(packet, feature_values=(1.0, 1.0))
    elif component == "symbol":
        altered = replace(packet, symbol="UNKNOWN")
    elif component == "owner":
        altered = replace(packet, vintage_digest="0" * 64)
    elif component == "block":
        altered = replace(
            packet,
            as_of=time(5),
            source_available_at=time(5),
            forecast_available_at=time(5),
            horizon_end=time(6),
        )
    elif component == "recipe":
        altered = replace(packet, forecast_available_at=time(6.5))
    else:
        altered = replace(packet, horizon_seconds=7_200, horizon_end=time(8))
    with pytest.raises(ValueError):
        replace(stream, packets=(altered, *stream.packets[1:]))


def test_stream_rejects_duplicate_sources_or_an_unused_vintage() -> None:
    api = load_api()
    stream = fit(api)
    with pytest.raises(ValueError):
        replace(stream, packets=(*stream.packets, stream.packets[0]))
    with pytest.raises(ValueError):
        replace(stream, packets=tuple(p for p in stream.packets if p.as_of < time(9)))


def test_json_roundtrip_requires_external_digest_and_preserves_frozen_arrays() -> None:
    api = load_api()
    stream = fit(api)
    payload = json.loads(json.dumps(stream.payload(), allow_nan=False))
    restored = api.FrozenSimpleReturnStream.from_payload(
        payload, expected_digest=stream.digest
    )
    assert restored.payload() == stream.payload()
    assert restored.digest == stream.digest
    assert restored.causal_scope_digest == stream.causal_scope_digest
    with pytest.raises(TypeError):
        api.FrozenSimpleReturnStream.from_payload(payload)
    with pytest.raises(ValueError):
        api.FrozenSimpleReturnStream.from_payload(payload, expected_digest="0" * 64)
    for vintage in restored.vintages:
        for array in (
            vintage.training.labels,
            vintage.training.features,
            vintage.training.sample_weights,
            vintage.training.log_training.labels,
            vintage.model.coefficients,
        ):
            with pytest.raises(ValueError):
                array.setflags(write=True)
    payload["packets"][0]["feature_values"][0] = -123.0
    assert restored.payload() == stream.payload()


@pytest.mark.parametrize(
    "component",
    (
        "stream_schema",
        "extra_stream_field",
        "extra_packet_field",
        "unit",
        "basis",
        "variance_kind",
        "forecast_kind",
        "horizon",
        "training_labels",
        "model",
        "variance",
        "prediction",
        "snapshot",
        "packet_digest",
        "causal_scope",
    ),
)
def test_reader_rejects_rehashed_outer_payload_with_inconsistent_nested_content(
    component: str,
) -> None:
    api = load_api()
    stream = fit(api)
    payload = json.loads(json.dumps(stream.payload()))
    packet = payload["packets"][0]
    if component == "stream_schema":
        payload["schema"] = "legacy_log_return_stream"
    elif component == "extra_stream_field":
        payload["unknown"] = "ignored content would escape the identity contract"
    elif component == "extra_packet_field":
        packet["unknown"] = 123
    elif component == "unit":
        packet["return_unit"] = "log_return"
    elif component == "basis":
        packet["valuation_basis"] = "total_return"
    elif component == "variance_kind":
        packet["variance_kind"] = "conditional_variance"
    elif component == "forecast_kind":
        packet["forecast_kind"] = "conditional_mean"
    elif component == "horizon":
        packet["horizon_seconds"] = 7_200
        packet["horizon_end"] = int(time(8).astype(np.int64))
    elif component == "training_labels":
        payload["vintages"][0]["training"]["labels"][0] += 0.1
    elif component == "model":
        payload["vintages"][0]["model"]["intercept"] += 0.1
    elif component == "variance":
        packet["fit_prefix_marginal_variance"] += 0.1
    elif component == "prediction":
        packet["expected_simple_return"] += 0.1
    elif component == "snapshot":
        packet["snapshot_digest"] = "0" * 64
    elif component == "packet_digest":
        packet["digest"] = "0" * 64
    else:
        payload["causal_scope_digest"] = "0" * 64
    with pytest.raises(ValueError):
        api.FrozenSimpleReturnStream.from_payload(
            payload, expected_digest=stream.digest
        )
    with pytest.raises(ValueError):
        api.FrozenSimpleReturnStream.from_payload(
            payload, expected_digest=content_digest(payload)
        )


@pytest.mark.parametrize(
    "field", ("expected_simple_return", "fit_prefix_marginal_variance")
)
def test_reader_rejects_wrong_owner_values_after_all_packet_hashes_are_recomputed(
    field: str,
) -> None:
    api = load_api()
    stream = fit(api)
    payload = json.loads(json.dumps(stream.payload()))
    packet = payload["packets"][0]
    packet[field] += 0.1
    packet["digest"] = content_digest(
        {key: value for key, value in packet.items() if key != "digest"}
    )
    payload["causal_scope_digest"] = content_digest(
        {"vintages": payload["vintages"], "packets": payload["packets"]}
    )
    with pytest.raises(ValueError):
        api.FrozenSimpleReturnStream.from_payload(
            payload, expected_digest=content_digest(payload)
        )
