"""Causal multi-clock Signature integration and feature/strategy adapters."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import numpy as np
import pytest

from tests.data.test_native_cross_asset_alignment import _Source, _source
from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
    MarketCalendarKind,
    VolumeUnit,
)
from trade_rl.data.features import (
    with_multitimeframe_path_signatures,
    with_path_signatures,
)
from trade_rl.data.identity import DATASET_ID_ARRAY_FIELDS
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation import signature_comparison
from trade_rl.evaluation.signature_comparison import (
    _matched_baseline_fit_dataset,
    run_ridge_signature_comparison,
    validate_signature_pair,
)
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.forecasts.supervised import build_causal_forecast_training_set
from trade_rl.strategies.rl.ppo import PPOTradingEnv


def _dataset(source: _Source) -> MarketDataset:
    config = MarketBuildConfig(
        base_timeframe="1h",
        features=(FeatureSpec("log_return", kind=FeatureKind.LOG_RETURN),),
    )
    return MarketDatasetBuilder(config).build(
        source,
        (
            InstrumentContract("BTCUSDT", listed_at=datetime(2025, 1, 1, tzinfo=UTC)),
            InstrumentContract("ETHUSDT", listed_at=datetime(2025, 1, 1, tzinfo=UTC)),
        ),
    )


def test_native_15m_signature_uses_intrahour_completed_closes_and_masks() -> None:
    source = _source()
    base = _dataset(source)
    extended = with_multitimeframe_path_signatures(
        base,
        source,
        base_timeframe="1h",
        windows_by_timeframe={"15m": 5, "1h": 3},
        depth=2,
    )
    prefix = "mt_path_sig_v1_15m_w5_d2_tp"
    assert extended.n_features == base.n_features + 12
    tp = extended.feature_names.index(f"{prefix}_tp")
    pt = extended.feature_names.index(f"{prefix}_pt")
    assert not extended.feature_available[0, 0, tp]
    assert extended.feature_available[1, 0, tp]
    # Independent signed-area oracle for the four completed 15m segments
    raw = source.values["BTCUSDT", "15m"]
    price_deltas = np.diff(np.log(raw.close[:5]))
    time_deltas = np.full(4, 0.25)
    signed_area = sum(
        time_deltas[i] * price_deltas[j] - price_deltas[i] * time_deltas[j]
        for i in range(4)
        for j in range(i + 1, 4)
    )
    actual = extended.features[1, 0, tp] - extended.features[1, 0, pt]
    assert actual == pytest.approx(signed_area, abs=1e-6)
    assert extended.feature_available[2, 0, -12:].all()
    np.testing.assert_array_equal(extended.close, base.close)
    np.testing.assert_array_equal(extended.fee_rate, base.fee_rate)
    assert extended.dataset_id != base.dataset_id
    assert extended.identity_verified


def test_native_signature_late_publication_masks_early_decision() -> None:
    source = _source()
    raw = source.values["BTCUSDT", "15m"]
    delayed = raw.available_at.copy()
    delayed[4] = np.datetime64("2026-01-01T01:15", "ns")
    sources = dict(source.values)
    sources["BTCUSDT", "15m"] = replace(raw, available_at=delayed)
    slow = _Source(sources)
    base = _dataset(slow)
    result = with_multitimeframe_path_signatures(
        base, slow, base_timeframe="1h", windows_by_timeframe={"15m": 5}
    )
    index = result.feature_names.index("mt_path_sig_v1_15m_w5_d2_tp_tp")
    assert not result.feature_available[1, 0, index]
    assert result.feature_available[2, 0, index]


def test_native_signature_future_and_ohlc_modification_do_not_change_prefix() -> None:
    source = _source()
    base = _dataset(source)
    first = with_multitimeframe_path_signatures(
        base, source, base_timeframe="1h", windows_by_timeframe={"15m": 5}
    )
    raw = source.values["BTCUSDT", "15m"]
    altered = raw.close.copy()
    altered[21:] *= 1.5
    changed_sources = dict(source.values)
    changed_sources["BTCUSDT", "15m"] = replace(
        raw,
        close=altered,
        high=np.maximum(raw.high, altered) * 1.1,
        low=raw.low * 0.5,
    )
    changed = with_multitimeframe_path_signatures(
        base,
        _Source(changed_sources),
        base_timeframe="1h",
        windows_by_timeframe={"15m": 5},
    )
    np.testing.assert_array_equal(first.features[:5], changed.features[:5])
    np.testing.assert_array_equal(
        first.feature_available[:5], changed.feature_available[:5]
    )
    assert first.dataset_id != changed.dataset_id
    ohlc_sources = dict(source.values)
    ohlc_sources["BTCUSDT", "15m"] = replace(
        raw, high=raw.high * 1.4, low=raw.low * 0.6
    )
    same = with_multitimeframe_path_signatures(
        base,
        _Source(ohlc_sources),
        base_timeframe="1h",
        windows_by_timeframe={"15m": 5},
    )
    np.testing.assert_array_equal(first.features, same.features)


def test_opt_in_signature_ridge_and_ppo_observation_share_same_causal_dataset() -> None:
    source = _source()
    base = _dataset(source)
    extended = with_multitimeframe_path_signatures(
        base, source, base_timeframe="1h", windows_by_timeframe={"15m": 5}
    )
    name = "mt_path_sig_v1_15m_w5_d2_tp_tp"
    baseline, enhanced = validate_signature_pair(
        base,
        extended,
        baseline_feature_names=("log_return",),
        signature_feature_names=(name,),
    )
    assert baseline == (0,)
    assert enhanced == (0, extended.feature_names.index(name))
    # Same executed bars, capital and executor; comparison never promotes a winner.
    result = run_ridge_signature_comparison(
        base,
        extended,
        baseline_feature_names=("log_return",),
        signature_feature_names=(name,),
        fit_cutoff=base.timestamps[5],
        evaluation_start_index=5,
        evaluation_stop_index=7,
        symbol_index=0,
        horizon_hours=1,
        alpha=1.0,
        entry_threshold=0.00001,
        exit_threshold=0.0,
        gross_budget=0.05,
        initial_capital=10_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )
    assert len(result.baseline.returns.values) == 2
    assert len(result.signature.returns.values) == 2
    assert len(result.cash.returns.values) == 2
    assert result.source_dataset_id == base.dataset_id
    assert result.augmented_dataset_id == extended.dataset_id
    base_env = PPOTradingEnv(
        base,
        feature_indices=baseline,
        start_index=4,
        stop_index=7,
        gross_budget=0.05,
    )
    signature_env = PPOTradingEnv(
        extended,
        feature_indices=enhanced,
        start_index=4,
        stop_index=7,
        gross_budget=0.05,
    )
    baseline_obs, _ = base_env.reset(seed=1)
    signature_obs, _ = signature_env.reset(seed=1)
    assert len(baseline_obs) == 3 * len(baseline) + 2
    assert len(signature_obs) == 3 * len(enhanced) + 2
    assert np.isfinite(signature_obs).all()
    _, reward, terminated, truncated, _ = signature_env.step(1)
    assert np.isfinite(reward)
    assert not truncated
    assert isinstance(terminated, bool)


def test_signature_rejects_invalid_clock_and_unavailable_feature_pair() -> None:
    source = _source()
    base = _dataset(source)
    with pytest.raises(ValueError, match="base_timeframe"):
        with_multitimeframe_path_signatures(
            base, source, base_timeframe="4h", windows_by_timeframe={"15m": 5}
        )
    with pytest.raises(ValueError, match="depth"):
        with_multitimeframe_path_signatures(
            base, source, base_timeframe="1h", windows_by_timeframe={"15m": 5}, depth=4
        )
    with pytest.raises(ValueError, match="MultiTimeframeMarketDataSource"):
        with_multitimeframe_path_signatures(
            base, None, base_timeframe="1h", windows_by_timeframe={"15m": 5}
        )
    extended = with_multitimeframe_path_signatures(
        base, source, base_timeframe="1h", windows_by_timeframe={"15m": 5}
    )
    with pytest.raises(ValueError, match="rosters"):
        validate_signature_pair(
            base,
            extended,
            baseline_feature_names=("log_return",),
            signature_feature_names=("log_return",),
        )


def test_signature_comparison_matches_fit_rows_with_longer_native_warmup() -> None:
    source = _source()
    base = _dataset(source)
    augmented = with_multitimeframe_path_signatures(
        base, source, base_timeframe="1h", windows_by_timeframe={"1h": 5}
    )
    sig_name = "mt_path_sig_v1_1h_w5_d2_tp_tp"
    sig_index = augmented.feature_names.index(sig_name)
    fit_base = _matched_baseline_fit_dataset(base, augmented, (sig_index,))
    expected_mask = augmented.feature_available[:, :, sig_index]
    np.testing.assert_array_equal(
        fit_base.feature_available[:, :, 0],
        base.feature_available[:, :, 0] & expected_mask,
    )
    assert not fit_base.feature_available[:4, :, 0].any()
    assert fit_base.identity_verified
    assert fit_base.dataset_id != base.dataset_id
    baseline_rows = build_causal_forecast_training_set(
        fit_base,
        feature_indices=(0,),
        fit_cutoff=base.timestamps[7],
        horizon_hours=1,
    )
    signature_rows = build_causal_forecast_training_set(
        augmented,
        feature_indices=(0, sig_index),
        fit_cutoff=base.timestamps[7],
        horizon_hours=1,
    )
    np.testing.assert_array_equal(
        baseline_rows.label_end_times, signature_rows.label_end_times
    )
    np.testing.assert_array_equal(baseline_rows.labels, signature_rows.labels)
    np.testing.assert_array_equal(
        baseline_rows.sample_weights, signature_rows.sample_weights
    )
    assert baseline_rows.n_samples == signature_rows.n_samples
    # The original data/metrics are untouched by fit-only common eligibility.
    np.testing.assert_array_equal(fit_base.close, base.close)


def _funding_pair(
    *, session: bool = False, global_information: bool = False
) -> tuple[MarketDataset, MarketDataset]:
    base = _dataset(_source())
    rates = np.zeros_like(base.funding_rate)
    products = np.zeros_like(base.funding_rate)
    counts = np.zeros(base.funding_rate.shape, dtype=np.int32)
    rates[3, 0], products[3, 0], counts[3, 0] = 0.001, 0.12, 1
    changes: dict[str, object] = {
        "funding_rate": rates,
        "funding_price_rate": products,
        "funding_event_count": counts,
        "funding_due": counts > 0,
    }
    if session:
        changes.update(calendar_kind=MarketCalendarKind.SESSION, nominal_bar_hours=1.0)
    if global_information:
        changes.update(
            global_features=np.ones((base.n_bars, 1), dtype=np.float32),
            global_feature_names=("common_information",),
            global_feature_available=None,
            global_feature_staleness_hours=None,
            global_feature_missing_reason=None,
        )
    base = replace(base, identity_payload_json=None, **changes).with_content_identity(
        {"fixture": "paired-funding"}
    )
    augmented = with_path_signatures(base, window_bars=3, depth=2)
    return base, augmented


def _validate_pair(
    base: MarketDataset, augmented: MarketDataset
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return validate_signature_pair(
        base,
        augmented,
        baseline_feature_names=("log_return",),
        signature_feature_names=("path_sig_v1_w3_d2_tp_tp",),
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("funding_price_rate", 0.13),
        ("funding_event_count", 2),
        ("cash_rate", 0.005),
        ("contract_multipliers", 2),
    ],
)
def test_signature_pair_rejects_reidentified_financial_change(
    field: str, value: float
) -> None:
    base, augmented = _funding_pair()
    changed_array = augmented.resolved_array(field).copy()
    if field == "contract_multipliers":
        index = 0
    elif field == "cash_rate":
        index = 3
    else:
        index = (3, 0)
    changed_array[index] = value
    changed = replace(
        augmented, identity_payload_json=None, **{field: changed_array}
    ).with_content_identity({"counterexample": field})
    assert base.identity_verified and changed.identity_verified
    assert changed.dataset_id != augmented.dataset_id
    assert base.funding_rate[3, 0] == 0.001
    assert base.funding_price_rate[3, 0] == 0.12
    assert base.funding_event_count[3, 0] == 1
    assert changed.resolved_array(field)[index] == value
    for name in DATASET_ID_ARRAY_FIELDS:
        if name != field:
            np.testing.assert_array_equal(
                augmented.resolved_array(name), changed.resolved_array(name)
            )
    # A two-event count can retain aggregate rate/product; count alone is not P&L.
    with pytest.raises(ValueError, match=f"economic input: {field}$"):
        _validate_pair(base, changed)


@pytest.mark.parametrize(
    "field", ["volume_units", "calendar_kind", "nominal_bar_hours", "periods_per_year"]
)
def test_signature_pair_rejects_reidentified_economic_metadata(field: str) -> None:
    base, augmented = _funding_pair(
        session=field in {"nominal_bar_hours", "periods_per_year"}
    )
    changes: dict[str, object]
    if field == "volume_units":
        assert base.volume_units[0] is VolumeUnit.BASE_ASSET
        changes = {field: (VolumeUnit.QUOTE_NOTIONAL, *base.volume_units[1:])}
    elif field == "calendar_kind":
        changes = {field: MarketCalendarKind.SESSION, "nominal_bar_hours": 1.0}
    else:
        changes = {field: 2.0 if field == "nominal_bar_hours" else 252}
    changed = replace(
        augmented, identity_payload_json=None, **changes
    ).with_content_identity({"counterexample": field})
    assert base.identity_verified and changed.identity_verified
    for name in DATASET_ID_ARRAY_FIELDS:
        np.testing.assert_array_equal(
            augmented.resolved_array(name), changed.resolved_array(name)
        )
    original_metadata, changed_metadata = (
        base.identity_contract_payload(),
        changed.identity_contract_payload(),
    )
    assert original_metadata[field] != changed_metadata[field]
    for name in (
        "calendar_kind",
        "nominal_bar_hours",
        "periods_per_year",
        "volume_units",
    ):
        if name != field:
            assert original_metadata[name] == changed_metadata[name]
    with pytest.raises(ValueError, match=f"economic metadata: {field}$"):
        _validate_pair(base, changed)


@pytest.mark.parametrize(
    "field",
    [
        "feature_staleness",
        "feature_staleness_hours",
        "feature_missing_reason",
        "global_features",
        "global_feature_available",
        "global_feature_staleness_hours",
        "global_feature_missing_reason",
    ],
)
def test_signature_pair_preserves_original_information(field: str) -> None:
    base, augmented = _funding_pair(global_information=True)
    values = augmented.resolved_array(field).copy()
    if field == "feature_staleness":
        values[3, 0, 0] = 0.5
    elif field.startswith("feature_"):
        values[3, 0, 0] = 1
    else:
        values[3, 0] = False if field == "global_feature_available" else 2
    changed = replace(
        augmented, identity_payload_json=None, **{field: values}
    ).with_content_identity({"counterexample": field})
    assert base.identity_verified and changed.identity_verified
    for name in DATASET_ID_ARRAY_FIELDS:
        if name != field:
            np.testing.assert_array_equal(
                augmented.resolved_array(name), changed.resolved_array(name)
            )
    with pytest.raises(ValueError, match=f"economic input: {field}$"):
        _validate_pair(base, changed)


def test_signature_pair_accepts_normalized_continuous_clock_and_appended_features() -> (
    None
):
    base, augmented = _funding_pair()
    equivalent = replace(
        augmented, identity_payload_json=None, nominal_bar_hours=2.0
    ).with_content_identity({"fixture": "equivalent-clock"})
    assert (
        base.identity_contract_payload()["nominal_bar_hours"]
        == equivalent.identity_contract_payload()["nominal_bar_hours"]
        == 1.0
    )
    assert equivalent.identity_verified
    assert equivalent.n_features > base.n_features
    baseline, extended = _validate_pair(base, equivalent)
    assert baseline == (0,)
    assert extended == (0, equivalent.feature_names.index("path_sig_v1_w3_d2_tp_tp"))


def test_signature_comparison_refuses_settlement_mismatch_before_fit_or_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base, augmented = _funding_pair()
    products = augmented.funding_price_rate.copy()
    products[3, 0] = 0.13
    changed = replace(
        augmented, identity_payload_json=None, funding_price_rate=products
    ).with_content_identity({"counterexample": "before-fit"})
    assert base.identity_verified and changed.identity_verified

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("mismatched source reached fit or replay")

    monkeypatch.setattr(signature_comparison, "fit_ridge_forecast", forbidden)
    monkeypatch.setattr(signature_comparison, "run_single_symbol_replay", forbidden)
    with pytest.raises(ValueError, match="economic input: funding_price_rate$"):
        run_ridge_signature_comparison(
            base,
            changed,
            baseline_feature_names=("log_return",),
            signature_feature_names=("path_sig_v1_w3_d2_tp_tp",),
            fit_cutoff=base.timestamps[5],
            evaluation_start_index=5,
            evaluation_stop_index=7,
            symbol_index=0,
            horizon_hours=1,
            alpha=1.0,
            entry_threshold=0.00001,
            exit_threshold=0.0,
            gross_budget=0.05,
            initial_capital=10000,
            execution_cost=ExecutionCostConfig.zero(),
        )
