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
)
from trade_rl.data.features import with_multitimeframe_path_signatures
from trade_rl.evaluation.signature_comparison import (
    run_ridge_signature_comparison,
    validate_signature_pair,
)
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.rl.ppo import PPOTradingEnv


def _dataset(source: _Source) -> object:
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
