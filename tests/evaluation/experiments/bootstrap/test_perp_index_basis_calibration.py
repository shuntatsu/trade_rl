from __future__ import annotations

import importlib
import json
import math
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.experiments.bootstrap.perp_index_basis_prereg import (
    canonical_perp_index_basis_protocol,
)


def _api() -> ModuleType:
    return importlib.import_module(
        "trade_rl.evaluation.experiments.bootstrap.perp_index_basis_calibration"
    )


def _dataset(
    *,
    slopes: tuple[float, ...] = (-0.60, -0.50, -0.40, -0.30, 0.20),
    n_bars: int = 9_000,
) -> MarketDataset:
    protocol = canonical_perp_index_basis_protocol()
    n_symbols = len(protocol.symbols)
    timestamps = np.datetime64("2021-01-01T01:00:00", "ns") + np.arange(
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
        row_present=np.ones(shape, dtype=np.bool_),
        available_at=np.broadcast_to(timestamps[:, None], shape).copy(),
    )


def _replace_array(
    dataset: MarketDataset, field: str, value: np.ndarray
) -> MarketDataset:
    changes: dict[str, object] = {field: value}
    if field == "feature_available":
        staleness = np.asarray(dataset.feature_staleness).copy()
        staleness[~np.asarray(value, dtype=np.bool_)] = 1.0
        changes["feature_staleness"] = staleness
    elif field == "asset_active":
        active = np.asarray(value, dtype=np.bool_)
        changes["symbol_active"] = active
        tradable = np.asarray(dataset.tradable).copy()
        tradable[~active] = False
        changes["tradable"] = tradable
        information = np.asarray(dataset.information_available).copy()
        information[~active] = False
        changes["information_available"] = information
    return replace(dataset, **changes)


def _corrupt_array(
    dataset: MarketDataset, field: str, value: np.ndarray
) -> MarketDataset:
    object.__setattr__(dataset, field, np.asarray(value))
    return dataset


def test_four_negative_slopes_validate_and_preserve_training_only_boundary() -> None:
    module = _api()
    protocol = canonical_perp_index_basis_protocol()
    slopes = (-0.60, -0.50, -0.40, -0.30, 0.20)
    result = module.calibrate_perp_index_basis(_dataset(slopes=slopes), protocol)

    assert result.status == protocol.valid_status
    assert result.protocol_digest == protocol.digest
    assert result.negative_slope_count == 4
    assert result.symbols == protocol.symbols
    for item, expected in zip(result.symbol_results, slopes, strict=True):
        assert item.eligible_observations >= 8_760
        assert item.denominator > 0.0
        assert math.isfinite(item.numerator)
        assert item.beta == pytest.approx(expected, abs=5e-6)
        assert item.negative_slope is (expected < 0.0)

    assert result.training_relation_executed is True
    assert result.evaluation_pnl_inspected is False
    assert result.evaluation_execution_authorized is False
    assert result.final_test_authorized is False
    assert result.shared_cash_profitability_established is False
    assert result.production_eligible is False
    assert result.live_trading_authorized is False


def test_three_negative_slopes_reject_without_rescue() -> None:
    module = _api()
    protocol = canonical_perp_index_basis_protocol()
    result = module.calibrate_perp_index_basis(
        _dataset(slopes=(-0.60, -0.50, -0.40, 0.30, 0.20)), protocol
    )
    assert result.status == protocol.reject_status
    assert result.negative_slope_count == 3
    assert result.failures == ()


def test_coverage_and_zero_denominator_fail_closed() -> None:
    module = _api()
    protocol = canonical_perp_index_basis_protocol()
    dataset = _dataset()
    availability = np.asarray(dataset.feature_available).copy()
    availability[:500, 0, 0] = False
    result = module.calibrate_perp_index_basis(
        _replace_array(dataset, "feature_available", availability), protocol
    )
    assert result.status == protocol.invalid_coverage_status
    assert result.symbol_results[0].eligible_observations < 8_760
    assert result.symbol_results[0].beta is None

    dataset = _dataset()
    features = np.asarray(dataset.features).copy()
    features[:, 0, 0] = 0.0
    result = module.calibrate_perp_index_basis(
        _replace_array(dataset, "features", features), protocol
    )
    assert result.status == protocol.invalid_coverage_status
    assert result.symbol_results[0].denominator == 0.0
    assert result.symbol_results[0].beta is None


@pytest.mark.parametrize(
    "field", ["row_present", "information_available", "tradable", "asset_active"]
)
def test_label_window_requires_present_information_active_and_tradable(
    field: str,
) -> None:
    module = _api()
    protocol = canonical_perp_index_basis_protocol()
    dataset = _dataset()
    baseline = module.calibrate_perp_index_basis(dataset, protocol)
    values = np.asarray(getattr(dataset, field)).copy()
    values[100, 0] = False
    changed = module.calibrate_perp_index_basis(
        _replace_array(dataset, field, values), protocol
    )
    assert (
        baseline.symbol_results[0].eligible_observations
        - changed.symbol_results[0].eligible_observations
        == 26
    )


def test_unavailable_feature_excludes_exact_decision() -> None:
    module = _api()
    protocol = canonical_perp_index_basis_protocol()
    dataset = _dataset()
    baseline = module.calibrate_perp_index_basis(dataset, protocol)
    available = np.asarray(dataset.feature_available).copy()
    available[100, 0, 0] = False
    changed = module.calibrate_perp_index_basis(
        _replace_array(dataset, "feature_available", available), protocol
    )
    assert (
        baseline.symbol_results[0].eligible_observations
        - changed.symbol_results[0].eligible_observations
        == 1
    )


def test_execution_and_endpoint_are_exactly_t_plus_1_and_t_plus_25() -> None:
    module = _api()
    protocol = canonical_perp_index_basis_protocol()
    dataset = _dataset()
    result = module.calibrate_perp_index_basis(dataset, protocol)
    x = np.asarray(dataset.features[:, 0, 0], dtype=np.float64)
    opens = np.asarray(dataset.open[:, 0], dtype=np.float64)
    timestamps = np.asarray(dataset.timestamps, dtype="datetime64[ns]")
    fit_start = np.datetime64(protocol.fit_start.replace(tzinfo=None), "ns")
    cutoff = np.datetime64(protocol.fit_cutoff.replace(tzinfo=None), "ns")
    pairs: list[tuple[float, float]] = []
    for t in range(dataset.n_bars - 25):
        if timestamps[t] < fit_start or timestamps[t + 25] >= cutoff:
            continue
        y = math.log(float(opens[t + 25]) / float(opens[t + 1]))
        pairs.append((float(x[t]), y))
    expected_num = math.fsum(a * b for a, b in pairs)
    expected_den = math.fsum(a * a for a, _ in pairs)
    item = result.symbol_results[0]
    assert item.numerator == expected_num
    assert item.denominator == expected_den
    assert item.beta == pytest.approx(expected_num / expected_den, abs=1e-15)


def test_cutoff_is_strict_and_post_2023_mutation_cannot_change_result() -> None:
    module = _api()
    protocol = canonical_perp_index_basis_protocol()
    dataset = _dataset(n_bars=17_600)
    baseline = module.calibrate_perp_index_basis(dataset, protocol)
    timestamps = np.asarray(dataset.timestamps, dtype="datetime64[ns]")
    cutoff = np.datetime64(protocol.fit_cutoff.replace(tzinfo=None), "ns")
    first_post = int(np.searchsorted(timestamps, cutoff, side="left"))
    opens = np.asarray(dataset.open).copy()
    opens[first_post:, :] *= 10.0
    features = np.asarray(dataset.features).copy()
    features[first_post:, :, :] *= -1000.0
    changed = module.calibrate_perp_index_basis(
        _corrupt_array(_corrupt_array(dataset, "open", opens), "features", features),
        protocol,
    )
    assert changed.to_payload() == baseline.to_payload()


def test_nonpositive_label_open_is_ineligible() -> None:
    module = _api()
    protocol = canonical_perp_index_basis_protocol()
    dataset = _dataset()
    baseline = module.calibrate_perp_index_basis(dataset, protocol)
    opens = np.asarray(dataset.open).copy()
    opens[100, 0] = 0.0
    changed = module.calibrate_perp_index_basis(
        _corrupt_array(dataset, "open", opens), protocol
    )
    assert (
        baseline.symbol_results[0].eligible_observations
        - changed.symbol_results[0].eligible_observations
        == 2
    )


def test_nonfinite_available_feature_fails_closed() -> None:
    module = _api()
    protocol = canonical_perp_index_basis_protocol()
    dataset = _dataset()
    features = np.asarray(dataset.features).copy()
    features[100, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite|feature"):
        module.calibrate_perp_index_basis(
            _corrupt_array(dataset, "features", features), protocol
        )


def test_no_intercept_differs_from_centered_ols() -> None:
    module = _api()
    protocol = canonical_perp_index_basis_protocol()
    dataset = _dataset(slopes=(-0.50,) * 5)
    features = np.asarray(dataset.features).copy()
    features[:, :, 0] += np.float32(0.20)
    shifted = _replace_array(dataset, "features", features)
    result = module.calibrate_perp_index_basis(shifted, protocol)

    timestamps = np.asarray(shifted.timestamps, dtype="datetime64[ns]")
    cutoff = np.datetime64(protocol.fit_cutoff.replace(tzinfo=None), "ns")
    xs: list[float] = []
    ys: list[float] = []
    for t in range(shifted.n_bars - 25):
        if timestamps[t + 25] >= cutoff:
            continue
        xs.append(float(shifted.features[t, 0, 0]))
        ys.append(
            math.log(
                float(shifted.open[t + 25, 0]) / float(shifted.open[t + 1, 0])
            )
        )
    expected = math.fsum(x * y for x, y in zip(xs, ys, strict=True)) / math.fsum(
        x * x for x in xs
    )
    mean_x = math.fsum(xs) / len(xs)
    mean_y = math.fsum(ys) / len(ys)
    centered = math.fsum(
        (x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)
    ) / math.fsum((x - mean_x) ** 2 for x in xs)
    assert not math.isclose(expected, centered, rel_tol=1e-5, abs_tol=1e-8)
    assert result.symbol_results[0].beta == pytest.approx(expected, abs=1e-15)


def test_fixed_order_fsum_is_observable_against_vector_reduction() -> None:
    module = _api()
    protocol = canonical_perp_index_basis_protocol()
    dataset = _dataset(slopes=(-0.50,) * 5)
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
    stressed = _corrupt_array(_corrupt_array(dataset, "open", opens), "features", features)
    result = module.calibrate_perp_index_basis(stressed, protocol)
    labels = [
        math.log(float(opens[t + 25, 0]) / float(opens[t + 1, 0]))
        for t in range(n_bars - 25)
    ]
    expected = math.fsum(labels)
    vectorized = float(np.sum(np.asarray(labels, dtype=np.float64)))
    assert expected != vectorized
    assert result.symbol_results[0].numerator == expected


def test_dataset_roster_feature_and_clock_drift_are_rejected() -> None:
    module = _api()
    protocol = canonical_perp_index_basis_protocol()
    dataset = _dataset()
    with pytest.raises(ValueError, match="symbol|roster"):
        module.calibrate_perp_index_basis(
            replace(dataset, symbols=tuple(reversed(dataset.symbols))), protocol
        )
    with pytest.raises(ValueError, match="feature"):
        module.calibrate_perp_index_basis(
            replace(dataset, feature_names=("wrong_feature",)), protocol
        )
    dataset = _dataset()
    timestamps = np.asarray(dataset.timestamps).copy()
    timestamps[100:] += np.timedelta64(30, "m")
    with pytest.raises(ValueError, match="1h|hour|contiguous"):
        module.calibrate_perp_index_basis(
            _corrupt_array(dataset, "timestamps", timestamps), protocol
        )


def test_result_artifact_is_strict_authority_bound_and_canonical(tmp_path: Path) -> None:
    module = _api()
    protocol = canonical_perp_index_basis_protocol()
    result = module.calibrate_perp_index_basis(
        _dataset(),
        protocol,
        calibration_head="4" * 40,
        source_manifest_digest="5" * 64,
    )
    artifact = result.to_artifact_payload()
    assert artifact["protocol_digest"] == protocol.digest
    assert artifact["prereg_head"] == "795b2e9efc2e7d9d65a7131a160c7da16ba02043"
    assert artifact["source_implementation_head"] == (
        "4911579874111bb6620481be8a4dbd7a8e664918"
    )
    assert artifact["source_preflight_content_digest"] == (
        "ccb22002ac28a0f2a1275e4c31fb5cd0cde59b72d14017e15d1f88141dcf0e65"
    )
    assert artifact["content_digest"] == result.digest

    path = tmp_path / "result.json"
    path.write_bytes(canonical_json_bytes(artifact))
    loaded = module.load_perp_index_basis_calibration_result(path)
    assert loaded.to_artifact_payload() == artifact

    unbound = module.calibrate_perp_index_basis(_dataset(), protocol)
    with pytest.raises(ValueError, match="calibration_head|source_manifest|publication"):
        unbound.to_artifact_payload()

    pretty = tmp_path / "pretty.json"
    pretty.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical|bytes"):
        module.load_perp_index_basis_calibration_result(pretty)

    spoofed = dict(artifact)
    spoofed["evaluation_pnl_inspected"] = 0
    path.write_bytes(canonical_json_bytes(spoofed))
    with pytest.raises(ValueError, match="boolean|canonical|malformed"):
        module.load_perp_index_basis_calibration_result(path)

    forbidden = dict(artifact)
    forbidden["pnl"] = 1.0
    path.write_bytes(canonical_json_bytes(forbidden))
    with pytest.raises(ValueError, match="unknown|field|keys"):
        module.load_perp_index_basis_calibration_result(path)

    tampered = dict(artifact)
    tampered["source_implementation_head"] = "6" * 40
    path.write_bytes(canonical_json_bytes(tampered))
    with pytest.raises(ValueError, match="canonical|authority|digest"):
        module.load_perp_index_basis_calibration_result(path)
