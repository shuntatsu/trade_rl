"""Explicit higher-clock feature columns on the existing execution clock."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from trade_rl.data import load_market_dataset_artifact, write_market_dataset_files
from trade_rl.data.build.builder import MarketDatasetBuilder
from trade_rl.data.contracts import (
    FeatureKind,
    FeatureSpec,
    InstrumentContract,
    MarketBuildConfig,
)
from trade_rl.data.features.clocked_overlay import (
    ClockedNativeSource,
    append_clocked_native_features,
    clocked_native_source_digest,
)
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.data.native_resampling import resample_complete_native_bars
from trade_rl.data.source import InMemoryMarketDataSource, RawMarketSeries
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk import PreTradeRisk, PreTradeRiskConfig
from trade_rl.simulation import ExecutionCostConfig
from trade_rl.strategies.dataset_scope import validated_training_scope
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)


def _raw(times, price, *, available=None):
    return RawMarketSeries(
        timestamps=times,
        available_at=times if available is None else available,
        open=price - 0.5,
        high=price + 1,
        low=price - 1,
        close=price,
        volume=np.full(len(price), 1000.0),
        funding_rate=np.zeros(len(price)),
        funding_available=np.zeros(len(price), dtype=bool),
        tradable=np.ones(len(price), dtype=bool),
    )


def _fixture(kind):
    if kind == "month":
        anchor = np.datetime64("2024-01-01T00", "ns")
        end = np.datetime64("2024-05-01T00", "ns")
        closes = np.arange("2024-02", "2024-06", dtype="datetime64[M]")
        clock = NativeBarClock(NativeClockKind.UTC_MONTH)
    else:
        anchor = np.datetime64("2024-01-01T00", "ns")  # Monday
        days = 3 if kind == "3d" else 7
        end = anchor + np.timedelta64(days * 4, "D")
        closes = anchor + np.arange(1, 5) * np.timedelta64(days, "D")
        clock = (
            NativeBarClock(NativeClockKind.FIXED_ELAPSED, 3 * 86400, anchor)
            if days == 3
            else NativeBarClock(NativeClockKind.UTC_WEEK)
        )
    hours = np.arange(
        anchor + np.timedelta64(1, "h"),
        end + np.timedelta64(1, "h"),
        np.timedelta64(1, "h"),
    ).astype("datetime64[ns]")
    i = np.arange(len(hours), dtype=np.float64)
    contracts = (InstrumentContract("SYNTH_A"), InstrumentContract("SYNTH_B"))
    hourly = {
        c.symbol: _raw(hours, 100 + i / 100 + 10 * j) for j, c in enumerate(contracts)
    }
    dataset = MarketDatasetBuilder(
        MarketBuildConfig("1h", (FeatureSpec("base_ret", FeatureKind.LOG_RETURN),))
    ).build(InMemoryMarketDataSource(hourly), contracts)
    sources = []
    expected_prices = []
    for c in contracts:
        series = hourly[c.symbol]
        # First target close delayed by two hours; later target bars are immediate.
        receipt = hours.copy()
        receipt[hours == closes[0]] += np.timedelta64(2, "h")
        result = resample_complete_native_bars(
            source_clock=NativeBarClock(NativeClockKind.FIXED_ELAPSED, 3600, anchor),
            source_closes=hours,
            source_available_at=receipt,
            source_ohlcv=np.column_stack(
                (
                    series.open,
                    series.high,
                    series.low,
                    series.close,
                    series.volume,
                    series.volume * series.close,
                )
            ),
            target_clock=clock,
            target_closes=closes,
            as_of=end,
        )
        native = RawMarketSeries(
            timestamps=result.closes,
            available_at=result.available_at,
            open=result.ohlcv[:, 0],
            high=result.ohlcv[:, 1],
            low=result.ohlcv[:, 2],
            close=result.ohlcv[:, 3],
            volume=result.ohlcv[:, 4],
            funding_rate=np.zeros(len(closes)),
            funding_available=np.zeros(len(closes), dtype=bool),
            tradable=np.ones(len(closes), dtype=bool),
        )
        sources.append(
            ClockedNativeSource(
                c, native, clocked_native_source_digest(native, contract=c, clock=clock)
            )
        )
        expected_prices.append(native)
    return dataset, clock, tuple(sources), tuple(expected_prices)


def _append(dataset, clock, sources, **kwargs):
    specs = kwargs.pop(
        "specs",
        (
            FeatureSpec("native_body", FeatureKind.BODY_RETURN, max_staleness_hours=48),
            FeatureSpec("native_ret", FeatureKind.LOG_RETURN, max_staleness_hours=48),
        ),
    )
    return append_clocked_native_features(
        dataset,
        clock=clock,
        sources=sources,
        specs=specs,
        **kwargs,
    )


@pytest.mark.parametrize("kind", ["3d", "week", "month"])
def test_higher_clock_overlay_literal_asof_and_unchanged_execution(kind, tmp_path):
    base, clock, sources, raw = _fixture(kind)
    result = _append(base, clock, sources)
    assert result.identity_verified and result.dataset_id != base.dataset_id
    assert result.feature_names == ("base_ret", "native_body", "native_ret")
    assert result.periods_per_year == base.periods_per_year == 8760
    assert result.bar_hours == base.bar_hours == 1
    for name, values in base.identity_arrays().items():
        if name.startswith("feature"):
            np.testing.assert_array_equal(
                result.identity_arrays()[name][:, :, :1], values
            )
        else:
            np.testing.assert_array_equal(result.identity_arrays()[name], values)
    for j, series in enumerate(raw):
        for i, decision in enumerate(base.timestamps):
            for feature, lookback in ((1, 0), (2, 1)):
                candidates = [
                    k
                    for k in range(lookback, len(series.close))
                    if all(
                        series.available_at[m] <= decision
                        for m in range(k - lookback, k + 1)
                    )
                ]
                valid = False
                expected = 0.0
                if candidates:
                    k = candidates[-1]
                    age = (decision - series.timestamps[k]) / np.timedelta64(1, "h")
                    valid = age <= 48
                    if valid:
                        expected = (
                            np.log(series.close[k] / series.open[k])
                            if feature == 1
                            else np.log(series.close[k] / series.close[k - 1])
                        )
                assert bool(result.feature_available[i, j, feature]) == valid
                assert result.features[i, j, feature] == pytest.approx(
                    expected, abs=2e-8
                )
    assert _append(base, clock, sources).dataset_id == result.dataset_id
    written = write_market_dataset_files(tmp_path / kind, result)
    restored = load_market_dataset_artifact(tmp_path / kind)
    assert restored.identity_verified and restored.dataset_id == result.dataset_id
    assert written.artifact_digest
    assert validated_training_scope(
        restored, feature_indices=(0, 1, 2), fit_symbol_indices=(0,)
    ) == ((0, 1, 2), (0,))
    identity = json.loads(restored.identity_payload_json)
    assert identity["source_dataset"]["dataset_id"] == base.dataset_id
    assert identity["clocked_native_overlay"]["group_source_digests"][0] == [
        {"symbol": s.contract.symbol, "sha256": s.expected_source_digest}
        for s in sources
    ]


def test_future_native_mutation_is_causal_but_changes_content_identity():
    base, clock, sources, raw = _fixture("3d")
    before = _append(base, clock, sources)
    old = raw[0]
    close = old.close.copy()
    close[-1] *= 1.01
    changed = replace(old, close=close, high=np.maximum(old.high, close))
    source = replace(
        sources[0],
        raw=changed,
        expected_source_digest=clocked_native_source_digest(
            changed, contract=sources[0].contract, clock=clock
        ),
    )
    after = _append(base, clock, (source, sources[1]))
    prefix = base.timestamps < old.timestamps[-1]
    np.testing.assert_array_equal(before.features[prefix], after.features[prefix])
    np.testing.assert_array_equal(
        before.feature_available[prefix], after.feature_available[prefix]
    )
    assert before.dataset_id != after.dataset_id


def test_refuses_wrong_source_pin_gap_clock_roster_and_unverified_base():
    base, clock, sources, raw = _fixture("3d")
    with pytest.raises(ValueError, match="digest"):
        _append(
            base,
            clock,
            (replace(sources[0], expected_source_digest="f" * 64), sources[1]),
        )
    with pytest.raises(ValueError, match="roster"):
        _append(base, clock, sources[::-1])
    with pytest.raises(ValueError, match="identity"):
        _append(replace(base, identity_payload_json=None), clock, sources)
    with pytest.raises(ValueError, match="feature"):
        _append(
            base,
            clock,
            sources,
            specs=(FeatureSpec("base_ret", FeatureKind.BODY_RETURN),),
        )
    with pytest.raises(ValueError):
        _append(base, NativeBarClock(NativeClockKind.UTC_WEEK), sources)
    s = raw[0]
    keep = [0, 2, 3]
    gap = RawMarketSeries(
        **{
            name: getattr(s, name)[keep]
            for name in (
                "timestamps",
                "available_at",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "funding_rate",
                "funding_available",
                "funding_event_count",
                "tradable",
            )
        }
    )
    with pytest.raises(ValueError, match="consecutive"):
        clocked_native_source_digest(gap, contract=sources[0].contract, clock=clock)
    active = s.tradable.copy()
    active[1] = False
    with pytest.raises(ValueError, match="tradable"):
        clocked_native_source_digest(
            replace(s, tradable=active), contract=sources[0].contract, clock=clock
        )


def test_native_overlay_drives_real_common_replay_with_delayed_publication():
    base, clock, sources, raw = _fixture("3d")
    dataset = _append(base, clock, sources)
    first_native_close = raw[0].timestamps[0]
    first_decision = int(np.flatnonzero(dataset.timestamps == first_native_close)[0])
    seen = []

    class Policy:
        def decide_portfolio(
            self, snapshot: PortfolioDecisionSnapshot
        ) -> PortfolioTargetProposal:
            ready = snapshot.feature_available[0][1]
            seen.append((snapshot.index, ready))
            return PortfolioTargetProposal((0.3 if ready else 0.0, 0.0))

    result = run_shared_portfolio_replay(
        dataset,
        Policy(),
        start_index=first_decision,
        stop_index=first_decision + 8,
        initial_capital=1000.0,
        execution_cost=replace(ExecutionCostConfig.zero(), fee_rate=0.001),
        risk=PreTradeRisk(PreTradeRiskConfig(max_gross=1.0, max_abs_weight=1.0)),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    assert seen[:3] == [
        (first_decision, False),
        (first_decision + 1, False),
        (first_decision + 2, True),
    ]
    ledger = result.ledger_evidence.intervals
    fills = [
        e
        for row in ledger.intervals
        for e in row.order_events
        if e.event_type in {"filled", "partial_fill"}
    ]
    assert fills and fills[0].processing_index == first_decision + 3
    assert fills[0].symbol_index == 0
    assert sum(row.interval_cost for row in ledger.intervals) > 0
    assert result.book.exact_quantities == (0, 0)


def test_overlay_rejects_instrument_economics_inconsistent_with_base_contract():
    from trade_rl.data.contracts import VolumeUnit

    base, clock, sources, _ = _fixture("3d")
    identity = json.loads(base.identity_payload_json)
    altered = replace(
        base,
        identity_payload_json=None,
        volume_units=(VolumeUnit.QUOTE_NOTIONAL, VolumeUnit.BASE_ASSET),
    ).with_content_identity(identity)
    assert altered.identity_verified
    with pytest.raises(ValueError, match="instrument"):
        _append(altered, clock, sources)


def test_overlay_preserves_cross_symbol_dependency_restrictions_and_v7():
    base, clock, sources, _ = _fixture("3d")
    raw = {
        s.contract.symbol: _raw(base.timestamps, base.close[:, j])
        for j, s in enumerate(sources)
    }
    specs = (
        FeatureSpec("base_ret", FeatureKind.LOG_RETURN),
        FeatureSpec("rank", FeatureKind.CROSS_SECTIONAL_MOMENTUM_RANK),
    )
    original = MarketDatasetBuilder(
        MarketBuildConfig("1h", specs, cross_asset_reference_symbol="SYNTH_A")
    ).build(
        InMemoryMarketDataSource(raw),
        tuple(s.contract for s in sources),
    )
    v7 = replace(
        original,
        identity_payload_json=None,
        funding_price_rate=np.zeros_like(original.close),
    ).with_content_identity(json.loads(original.identity_payload_json))
    result = _append(v7, clock, sources)
    assert (
        json.loads(result.identity_payload_json)["schema"]
        == "market_dataset_identity_v7"
    )
    np.testing.assert_array_equal(result.funding_price_rate, v7.funding_price_rate)
    assert validated_training_scope(
        result, feature_indices=(0, 2, 3), fit_symbol_indices=(0,)
    ) == ((0, 2, 3), (0,))
    with pytest.raises(ValueError, match="outside fit symbol scope"):
        validated_training_scope(result, feature_indices=(1,), fit_symbol_indices=(0,))


def test_overlay_refuses_foreign_contract_unsupported_kinds_and_clock_phase():
    base, clock, sources, _ = _fixture("3d")
    wrong = replace(sources[0].contract, contract_multiplier=2)
    with pytest.raises(ValueError, match="instrument"):
        _append(base, clock, (replace(sources[0], contract=wrong), sources[1]))
    for kind in (FeatureKind.FUNDING_BPS, FeatureKind.CROSS_SECTIONAL_MOMENTUM_RANK):
        with pytest.raises(ValueError, match="price/volume"):
            _append(base, clock, sources, specs=(FeatureSpec("bad", kind),))
    with pytest.raises(ValueError, match="larger"):
        same_clock = NativeBarClock(
            NativeClockKind.FIXED_ELAPSED, 3600, base.timestamps[0]
        )
        same = tuple(
            ClockedNativeSource(
                s.contract,
                _raw(base.timestamps, base.close[:, j]),
                "a" * 64,
            )
            for j, s in enumerate(sources)
        )
        _append(base, same_clock, same)
    with pytest.raises(ValueError, match="market_build_v3"):
        _append(
            _append(base, clock, sources),
            clock,
            sources,
            specs=(FeatureSpec("extra", FeatureKind.BODY_RETURN),),
        )


def test_feature_configuration_digest_is_not_a_data_or_normalization_digest():
    base, clock, sources, _ = _fixture("3d")
    first = _append(base, clock, sources)
    altered = replace(
        base,
        identity_payload_json=None,
        normalization_digest="c" * 64,
    ).with_content_identity(json.loads(base.identity_payload_json))
    second = _append(altered, clock, sources)
    assert first.feature_config_digest == second.feature_config_digest
    assert first.normalization_digest != second.normalization_digest
    assert first.dataset_id != second.dataset_id


def test_two_higher_clock_groups_share_one_execution_dataset():
    from trade_rl.data.features.clocked_overlay import (
        ClockedNativeFeatureGroup,
        append_clocked_native_feature_groups,
    )

    base, week, weekly, _ = _fixture("week")
    start = base.timestamps[0] - np.timedelta64(1, "h")
    three_day = NativeBarClock(NativeClockKind.FIXED_ELAPSED, 3 * 86400, start)
    closes = start + np.arange(1, 10) * np.timedelta64(3, "D")
    sources = []
    for j, s in enumerate(weekly):
        price = 100 + np.arange(len(closes), dtype=float) + j
        raw = _raw(closes, price)
        sources.append(
            ClockedNativeSource(
                s.contract,
                raw,
                clocked_native_source_digest(raw, contract=s.contract, clock=three_day),
            )
        )
    daily_spec = (
        FeatureSpec("three_day_body", FeatureKind.BODY_RETURN, max_staleness_hours=72),
    )
    week_spec = (
        FeatureSpec("week_body", FeatureKind.BODY_RETURN, max_staleness_hours=168),
    )
    groups = (
        ClockedNativeFeatureGroup(three_day, tuple(sources), daily_spec),
        ClockedNativeFeatureGroup(week, weekly, week_spec),
    )
    result = append_clocked_native_feature_groups(base, groups=groups)
    assert result.feature_names == ("base_ret", "three_day_body", "week_body")
    for k, group in enumerate(groups, 1):
        standalone = _append(base, group.clock, group.sources, specs=group.specs)
        np.testing.assert_array_equal(
            result.features[:, :, k], standalone.features[:, :, 1]
        )
        np.testing.assert_array_equal(
            result.feature_available[:, :, k], standalone.feature_available[:, :, 1]
        )
    assert result.identity_verified
    assert result.bar_hours == base.bar_hours
    assert validated_training_scope(
        result, feature_indices=(1, 2), fit_symbol_indices=(0,)
    ) == ((1, 2), (0,))
    with pytest.raises(ValueError, match="clock"):
        append_clocked_native_feature_groups(base, groups=(groups[0], groups[0]))
    with pytest.raises(ValueError, match="feature"):
        append_clocked_native_feature_groups(
            base, groups=(groups[0], replace(groups[1], specs=daily_spec))
        )
