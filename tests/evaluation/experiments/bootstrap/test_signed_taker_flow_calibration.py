from __future__ import annotations

import importlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.signed_taker_flow_prereg import (
    canonical_signed_taker_flow_protocol,
)


def _api() -> ModuleType:
    return importlib.import_module(
        "trade_rl.evaluation.experiments.bootstrap.signed_taker_flow_calibration"
    )


def _dataset(
    *,
    slopes: tuple[float, ...] = (0.60, 0.50, 0.40, 0.30, -0.20),
    n_bars: int = 9_000,
) -> MarketDataset:
    protocol = canonical_signed_taker_flow_protocol()
    n_symbols = len(protocol.symbols)
    timestamps = np.datetime64("2021-01-01T00:00:00", "ns") + np.arange(
        n_bars, dtype=np.int64
    ) * np.timedelta64(1, "h")
    time = np.arange(n_bars, dtype=np.float64)
    open_price = np.empty((n_bars, n_symbols), dtype=np.float64)
    for symbol_index in range(n_symbols):
        log_price = (
            5.0
            + 0.00001 * time
            + 0.001 * np.sin(2.0 * math.pi * time / 168.0 + 0.37 * symbol_index)
        )
        open_price[:, symbol_index] = np.exp(log_price)

    features = np.zeros((n_bars, n_symbols, 1), dtype=np.float32)
    for symbol_index, beta in enumerate(slopes):
        future = np.log(
            open_price[25:, symbol_index] / open_price[1 : n_bars - 24, symbol_index]
        )
        signal = future / beta
        assert np.max(np.abs(signal)) < 1.0
        features[: signal.size, symbol_index, 0] = signal.astype(np.float32)

    shape = (n_bars, n_symbols)
    return MarketDataset(
        dataset_id="1" * 64,
        symbols=protocol.symbols,
        timestamps=timestamps,
        features=features,
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=open_price,
        high=open_price.copy(),
        low=open_price.copy(),
        close=open_price.copy(),
        volume=np.full(shape, 1_000_000.0, dtype=np.float64),
        funding_rate=np.zeros(shape, dtype=np.float64),
        tradable=np.ones(shape, dtype=np.bool_),
        feature_available=np.ones_like(features, dtype=np.bool_),
        feature_names=(protocol.feature_name,),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        asset_active=np.ones(shape, dtype=np.bool_),
        information_available=np.ones(shape, dtype=np.bool_),
        available_at=np.broadcast_to(timestamps[:, None], shape).copy(),
    )


def _replace_array(
    dataset: MarketDataset, field: str, value: np.ndarray
) -> MarketDataset:
    return replace(dataset, **{field: value})


def _corrupt_array(
    dataset: MarketDataset, field: str, value: np.ndarray
) -> MarketDataset:
    object.__setattr__(dataset, field, np.asarray(value))
    return dataset


def test_calibration_is_genuinely_result_blind_until_module_exists() -> None:
    module = _api()
    assert hasattr(module, "calibrate_signed_taker_flow")
    assert hasattr(module, "load_signed_taker_flow_calibration_result")


def test_valid_gate_requires_at_least_four_strictly_positive_slopes() -> None:
    module = _api()
    protocol = canonical_signed_taker_flow_protocol()
    slopes = (0.60, 0.50, 0.40, 0.30, -0.20)

    result = module.calibrate_signed_taker_flow(_dataset(slopes=slopes), protocol)

    assert result.status == protocol.valid_status
    assert result.protocol_digest == protocol.digest
    assert result.symbols == protocol.symbols
    assert result.positive_slope_count == 4
    assert tuple(item.symbol for item in result.symbol_results) == protocol.symbols
    for item, expected in zip(result.symbol_results, slopes, strict=True):
        assert item.eligible_observations >= 8_760
        assert item.denominator > 0.0
        assert math.isfinite(item.numerator)
        assert item.beta == pytest.approx(expected, abs=5e-6)
        assert item.positive_slope is (expected > 0.0)

    assert result.training_relation_executed is True
    assert result.evaluation_pnl_inspected is False
    assert result.evaluation_execution_authorized is False
    assert result.final_test_authorized is False
    assert result.shared_cash_profitability_established is False
    assert result.production_eligible is False
    assert result.live_trading_authorized is False


def test_three_positive_slopes_reject_without_sign_flip_or_threshold_rescue() -> None:
    module = _api()
    protocol = canonical_signed_taker_flow_protocol()
    result = module.calibrate_signed_taker_flow(
        _dataset(slopes=(0.60, 0.50, 0.40, -0.30, -0.20)), protocol
    )

    assert result.status == protocol.reject_status
    assert result.positive_slope_count == 3
    assert result.failures == ()


def test_coverage_and_zero_denominator_fail_closed() -> None:
    module = _api()
    protocol = canonical_signed_taker_flow_protocol()
    dataset = _dataset()
    availability = np.asarray(dataset.feature_available).copy()
    availability[:500, 0, 0] = False
    insufficient = _replace_array(dataset, "feature_available", availability)

    result = module.calibrate_signed_taker_flow(insufficient, protocol)
    assert result.status == protocol.invalid_coverage_status
    assert result.symbol_results[0].eligible_observations < 8_760
    assert result.symbol_results[0].beta is None
    assert result.failures

    dataset = _dataset()
    features = np.asarray(dataset.features).copy()
    features[:, 0, 0] = 0.0
    zero_denominator = _replace_array(dataset, "features", features)
    result = module.calibrate_signed_taker_flow(zero_denominator, protocol)
    assert result.status == protocol.invalid_coverage_status
    assert result.symbol_results[0].denominator == 0.0
    assert result.symbol_results[0].beta is None


@pytest.mark.parametrize("field", ["information_available", "tradable", "asset_active"])
def test_label_and_decision_window_requires_information_active_and_tradable(
    field: str,
) -> None:
    module = _api()
    protocol = canonical_signed_taker_flow_protocol()
    baseline_dataset = _dataset()
    baseline = module.calibrate_signed_taker_flow(baseline_dataset, protocol)
    values = np.asarray(getattr(baseline_dataset, field)).copy()
    values[100, 0] = False
    changed = module.calibrate_signed_taker_flow(
        _replace_array(baseline_dataset, field, values), protocol
    )

    assert (
        baseline.symbol_results[0].eligible_observations
        - changed.symbol_results[0].eligible_observations
        == 26
    )


def test_feature_unavailable_at_decision_row_is_excluded() -> None:
    module = _api()
    protocol = canonical_signed_taker_flow_protocol()
    dataset = _dataset()
    baseline = module.calibrate_signed_taker_flow(dataset, protocol)
    available = np.asarray(dataset.feature_available).copy()
    available[100, 0, 0] = False
    changed = module.calibrate_signed_taker_flow(
        _replace_array(dataset, "feature_available", available), protocol
    )

    assert (
        baseline.symbol_results[0].eligible_observations
        - changed.symbol_results[0].eligible_observations
        == 1
    )


def test_execution_and_endpoint_are_exactly_t_plus_1_and_t_plus_25() -> None:
    module = _api()
    protocol = canonical_signed_taker_flow_protocol()
    dataset = _dataset()
    result = module.calibrate_signed_taker_flow(dataset, protocol)
    x = np.asarray(dataset.features[:, 0, 0], dtype=np.float64)
    opens = np.asarray(dataset.open[:, 0], dtype=np.float64)
    timestamps = np.asarray(dataset.timestamps, dtype="datetime64[ns]")
    fit_start = np.datetime64(protocol.fit_start.replace(tzinfo=None), "ns")
    cutoff = np.datetime64(protocol.fit_cutoff.replace(tzinfo=None), "ns")

    pairs: list[tuple[float, float]] = []
    for t in range(dataset.n_bars - 25):
        if timestamps[t] < fit_start or timestamps[t + 25] >= cutoff:
            continue
        label = math.log(float(opens[t + 25]) / float(opens[t + 1]))
        pairs.append((float(x[t]), label))
    expected_num = math.fsum(a * b for a, b in pairs)
    expected_den = math.fsum(a * a for a, _ in pairs)

    item = result.symbol_results[0]
    assert item.numerator == expected_num
    assert item.denominator == expected_den
    assert item.beta == pytest.approx(expected_num / expected_den, abs=1e-15)


def test_calibration_is_no_intercept_not_centered_ols() -> None:
    module = _api()
    protocol = canonical_signed_taker_flow_protocol()
    dataset = _dataset(slopes=(0.50, 0.50, 0.50, 0.50, 0.50))
    features = np.asarray(dataset.features).copy()
    features[:, :, 0] += np.float32(0.20)
    shifted = _replace_array(dataset, "features", features)
    result = module.calibrate_signed_taker_flow(shifted, protocol)

    timestamps = np.asarray(shifted.timestamps, dtype="datetime64[ns]")
    cutoff = np.datetime64(protocol.fit_cutoff.replace(tzinfo=None), "ns")
    x_values: list[float] = []
    y_values: list[float] = []
    for t in range(1, shifted.n_bars - 25):
        if timestamps[t + 25] >= cutoff:
            continue
        x = float(shifted.features[t, 0, 0])
        y = math.log(float(shifted.open[t + 25, 0]) / float(shifted.open[t + 1, 0]))
        x_values.append(x)
        y_values.append(y)
    expected = math.fsum(
        x * y for x, y in zip(x_values, y_values, strict=True)
    ) / math.fsum(x * x for x in x_values)
    mean_x = math.fsum(x_values) / len(x_values)
    mean_y = math.fsum(y_values) / len(y_values)
    centered = math.fsum(
        (x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values, strict=True)
    ) / math.fsum((x - mean_x) ** 2 for x in x_values)

    assert not math.isclose(expected, centered, rel_tol=1e-5, abs_tol=1e-8)
    assert result.symbol_results[0].beta == pytest.approx(expected, abs=1e-15)


def test_fixed_order_fsum_is_observable_against_vector_reduction() -> None:
    module = _api()
    protocol = canonical_signed_taker_flow_protocol()
    dataset = _dataset(slopes=(0.50,) * 5)
    n_bars = dataset.n_bars
    target = np.asarray(
        [(700.0, -700.0, 1.0e-13)[(t // 24) % 3] for t in range(n_bars - 25)],
        dtype=np.float64,
    )
    log_open = np.zeros(n_bars, dtype=np.float64)
    for t, value in enumerate(target):
        log_open[t + 25] = log_open[t + 1] + value
    opens_one = np.exp(log_open)
    opens = np.tile(opens_one[:, None], (1, len(protocol.symbols)))
    features = np.ones_like(dataset.features, dtype=np.float32)
    stressed = _corrupt_array(dataset, "open", opens)
    stressed = _corrupt_array(stressed, "features", features)

    result = module.calibrate_signed_taker_flow(stressed, protocol)
    actual_labels = [
        math.log(float(opens[t + 25, 0]) / float(opens[t + 1, 0]))
        for t in range(1, n_bars - 25)
    ]
    expected = math.fsum(actual_labels)
    vectorized = float(np.sum(np.asarray(actual_labels, dtype=np.float64)))
    assert expected != vectorized
    assert result.symbol_results[0].numerator == expected


def test_fit_cutoff_is_strict_and_post_2023_open_mutation_cannot_change_result() -> (
    None
):
    module = _api()
    protocol = canonical_signed_taker_flow_protocol()
    dataset = _dataset(n_bars=17_600)
    baseline = module.calibrate_signed_taker_flow(dataset, protocol)
    timestamps = np.asarray(dataset.timestamps, dtype="datetime64[ns]")
    cutoff = np.datetime64(protocol.fit_cutoff.replace(tzinfo=None), "ns")
    first_post = int(np.searchsorted(timestamps, cutoff, side="left"))
    opens = np.asarray(dataset.open).copy()
    opens[first_post:, :] *= 10.0
    changed = module.calibrate_signed_taker_flow(
        _corrupt_array(dataset, "open", opens), protocol
    )

    assert changed.to_payload() == baseline.to_payload()


def test_nonpositive_label_open_is_ineligible_not_silently_logged() -> None:
    module = _api()
    protocol = canonical_signed_taker_flow_protocol()
    dataset = _dataset()
    baseline = module.calibrate_signed_taker_flow(dataset, protocol)
    opens = np.asarray(dataset.open).copy()
    opens[100, 0] = 0.0
    changed = module.calibrate_signed_taker_flow(
        _corrupt_array(dataset, "open", opens), protocol
    )

    assert (
        baseline.symbol_results[0].eligible_observations
        - changed.symbol_results[0].eligible_observations
        == 2
    )


def test_calibration_rejects_non_hourly_clock_roster_and_feature_identity_drift() -> (
    None
):
    module = _api()
    protocol = canonical_signed_taker_flow_protocol()
    dataset = _dataset()

    timestamps = np.asarray(dataset.timestamps).copy()
    timestamps[100:] += np.timedelta64(30, "m")
    with pytest.raises(ValueError, match="1h|hour|contiguous"):
        module.calibrate_signed_taker_flow(
            _corrupt_array(dataset, "timestamps", timestamps), protocol
        )

    dataset = _dataset()
    with pytest.raises(ValueError, match="symbol|roster"):
        module.calibrate_signed_taker_flow(
            replace(dataset, symbols=tuple(reversed(dataset.symbols))), protocol
        )

    dataset = _dataset()
    with pytest.raises(ValueError, match="feature"):
        module.calibrate_signed_taker_flow(
            replace(dataset, feature_names=("wrong_feature",)), protocol
        )


def test_available_feature_outside_mathematical_bounds_fails_closed() -> None:
    module = _api()
    protocol = canonical_signed_taker_flow_protocol()
    dataset = _dataset()
    features = np.asarray(dataset.features).copy()
    features[100, 0, 0] = np.float32(1.01)
    with pytest.raises(ValueError, match=r"bound|\[-1, 1\]|feature"):
        module.calibrate_signed_taker_flow(
            _replace_array(dataset, "features", features), protocol
        )


def test_result_artifact_is_strict_content_addressed_and_forbids_evaluation_fields(
    tmp_path: Path,
) -> None:
    module = _api()
    protocol = canonical_signed_taker_flow_protocol()
    result = module.calibrate_signed_taker_flow(_dataset(), protocol)
    artifact = result.to_artifact_payload()

    assert artifact["protocol_digest"] == protocol.digest
    assert artifact["protocol_seal_run_id"] == 34844820406
    assert artifact["protocol_seal_artifact_id"] == 10347597245
    assert artifact["implementation_head"] == "0186a3a18ff14a5529de1f8869b1e45f461c036b"
    assert artifact["content_digest"] == result.digest
    assert len(result.digest) == 64

    path = tmp_path / "result.json"
    path.write_text(
        json.dumps(artifact, allow_nan=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    loaded = module.load_signed_taker_flow_calibration_result(path)
    assert loaded.to_artifact_payload() == artifact

    spoofed = dict(artifact)
    spoofed["evaluation_pnl_inspected"] = 0
    path.write_text(json.dumps(spoofed), encoding="utf-8")
    with pytest.raises(ValueError, match="boolean|malformed|canonical"):
        module.load_signed_taker_flow_calibration_result(path)

    forbidden = dict(artifact)
    forbidden["strategy_return"] = 1.23
    path.write_text(json.dumps(forbidden), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown|field|malformed|canonical"):
        module.load_signed_taker_flow_calibration_result(path)

    tampered = dict(artifact)
    tampered["positive_slope_count"] = 0
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="digest|content|canonical"):
        module.load_signed_taker_flow_calibration_result(path)
