"""Synthetic complete native-source construction, with no market-data reads."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction

import numpy as np
import pytest

from tests.data.test_native_execution_dataset import native_cost
from tests.evaluation.test_single_symbol_replay import AlwaysLong
from trade_rl.data.artifacts.codec import load_dataset_files, write_market_dataset_files
from trade_rl.data.build.native_spot import (
    build_native_fixed_spot_dataset,
    spot_raw_from_resampled,
)
from trade_rl.data.contracts import FeatureKind, FeatureSpec, InstrumentContract
from trade_rl.data.features.clocked_overlay import (
    ClockedNativeFeatureGroup,
    ClockedNativeSource,
    clocked_native_source_digest,
)
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.data.native_resampling import resample_complete_native_bars
from trade_rl.evaluation.replay import run_single_symbol_replay


def _pin(raw, contract, clock):
    return ClockedNativeSource(
        contract, raw, clocked_native_source_digest(raw, contract=contract, clock=clock)
    )


def _fixture(days=7, *, late=False, bump=0.0):
    anchor = np.datetime64("2024-01-01", "ns")
    clock = (
        NativeBarClock(NativeClockKind.UTC_WEEK)
        if days == 7
        else NativeBarClock(NativeClockKind.FIXED_ELAPSED, days * 86400, anchor)
    )
    fine = NativeBarClock(NativeClockKind.FIXED_ELAPSED, 3600, anchor)
    all_closes = anchor + np.arange(1, 9) * np.timedelta64(days, "D")
    fine_closes = anchor + np.arange(1, days * 24 * 8 + 1) * np.timedelta64(1, "h")
    x = 100.0 + np.arange(fine_closes.size) / 1000
    # A future-only perturbation for causal-prefix tests.
    x[fine_closes > all_closes[5]] += bump
    ohlcv = np.column_stack(
        (x, x + 0.5, x - 0.5, x + 0.25, np.full_like(x, 1000), x * 1000)
    )
    received = fine_closes.copy()
    if late:
        received[fine_closes == all_closes[3]] += np.timedelta64(1, "ns")
    as_of = all_closes[-1] + np.timedelta64(1, "s")
    bars = resample_complete_native_bars(
        source_clock=fine,
        source_closes=fine_closes,
        source_available_at=received,
        source_ohlcv=ohlcv,
        target_clock=clock,
        target_closes=all_closes,
        as_of=as_of,
    )
    raw = spot_raw_from_resampled(bars, clock=clock, as_of=as_of)
    fine_bars = resample_complete_native_bars(
        source_clock=fine,
        source_closes=fine_closes,
        source_available_at=received,
        source_ohlcv=ohlcv,
        target_clock=fine,
        target_closes=fine_closes,
        as_of=as_of,
    )
    fine_raw = spot_raw_from_resampled(fine_bars, clock=fine, as_of=as_of)
    contract = InstrumentContract(
        "SYNTH", tick_size=0.01, lot_size=0.001, minimum_notional=1.0
    )
    source = _pin(raw, contract, clock)
    groups = (
        ClockedNativeFeatureGroup(
            clock,
            (source,),
            (
                FeatureSpec("native_body", FeatureKind.BODY_RETURN),
                FeatureSpec("native_ret", FeatureKind.LOG_RETURN, lookback=2),
            ),
        ),
        ClockedNativeFeatureGroup(
            fine,
            (_pin(fine_raw, contract, fine),),
            (FeatureSpec("hour_body", FeatureKind.BODY_RETURN),),
        ),
    )
    return dict(
        clock=clock,
        closes=all_closes[2:],
        sources=(source,),
        feature_groups=groups,
        as_of=as_of,
    ), bars


@pytest.mark.parametrize("days", [3, 7])
def test_complete_resampling_to_artifact_and_replay(days, tmp_path):
    args, bars = _fixture(days)
    d = build_native_fixed_spot_dataset(**args)
    assert d.identity_verified and d.periods_per_year == Fraction(365, days)
    assert d.bar_hours == days * 24
    assert d.feature_names == ("native_body", "native_ret", "hour_body")
    assert d.feature_available.all()  # two complete native warm-up bars
    np.testing.assert_array_equal(d.close[:, 0], bars.ohlcv[2:, 3])
    np.testing.assert_array_equal(d.volume[:, 0], bars.ohlcv[2:, 4])
    np.testing.assert_array_equal(d.available_at[:, 0], bars.available_at[2:])
    np.testing.assert_allclose(
        d.features[:, 0, 0], np.log(bars.ohlcv[2:, 3] / bars.ohlcv[2:, 0]), rtol=1e-6
    )
    np.testing.assert_allclose(
        d.features[:, 0, 1], np.log(bars.ohlcv[2:, 3] / bars.ohlcv[:-2, 3]), rtol=1e-6
    )
    assert d.resolved_array("tick_size")[0, 0] == 0.01
    assert d.resolved_array("funding_event_count").sum() == 0
    assert d.global_feature_names == (
        "active_fraction",
        "tradable_fraction",
        "market_return_mean",
        "market_return_dispersion",
    )
    assert not d.resolved_array("global_feature_available")[0, 2]
    result = write_market_dataset_files(tmp_path, d)
    restored = load_dataset_files(tmp_path)
    assert result.schema_version == "market_dataset_artifact_v5"
    assert restored.dataset_id == d.dataset_id
    assert restored.identity_verified
    replay = run_single_symbol_replay(
        restored,
        AlwaysLong(),
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1000,
        execution_cost=native_cost(),
        settle_terminal_position=True,
    )
    assert replay.returns.periods_per_year == Fraction(365, days)
    assert replay.book.quantities[0] == 0
    assert replay.book.portfolio_value > 1000


def test_late_actual_receipt_is_not_rewritten_or_traded():
    args, bars = _fixture(late=True)
    d = build_native_fixed_spot_dataset(**args)
    assert d.available_at[1, 0] == bars.available_at[3] > d.timestamps[1]
    assert not d.resolved_array("information_available")[1, 0]
    assert not d.feature_available[1, 0].any()
    assert d.global_features[1, 1] == 0
    assert not d.resolved_array("global_feature_available")[1, 2]
    with pytest.raises(ValueError, match="unavailable"):
        run_single_symbol_replay(
            d,
            AlwaysLong(),
            start_index=0,
            stop_index=5,
            gross_budget=0.5,
            initial_capital=1000,
            execution_cost=native_cost(),
        )


def test_future_perturbation_does_not_change_prior_features():
    a = build_native_fixed_spot_dataset(**_fixture()[0])
    b = build_native_fixed_spot_dataset(**_fixture(bump=25)[0])
    assert a.dataset_id != b.dataset_id
    for key in (
        "features",
        "feature_available",
        "feature_staleness_hours",
        "global_features",
        "global_feature_available",
    ):
        np.testing.assert_array_equal(getattr(a, key)[:4], getattr(b, key)[:4])


def test_pin_and_clock_in_identity():
    args, _ = _fixture()
    d = build_native_fixed_spot_dataset(**args)
    assert d.dataset_id == build_native_fixed_spot_dataset(**args).dataset_id
    meta = json.loads(d.identity_payload_json)
    assert meta["config"]["schema_version"] == "native_fixed_spot_build_v1"
    assert meta["execution_source_digests"] == [
        args["sources"][0].expected_source_digest
    ]
    assert meta["feature_source_digests"][0] == [
        args["sources"][0].expected_source_digest
    ]
    assert meta["as_of_ns"] == int(args["as_of"].astype(np.int64))


@pytest.mark.parametrize(
    "mutation",
    [
        "pin",
        "gap",
        "cutoff",
        "future_range",
        "funding",
        "multiplier",
        "listing",
        "duplicate",
        "different_base_source",
    ],
)
def test_invalid_inputs_fail_closed(mutation):
    args, _ = _fixture()
    src = args["sources"][0]
    if mutation == "pin":
        args["sources"] = (replace(src, expected_source_digest="a" * 64),)
    elif mutation == "gap":
        args["closes"] = np.delete(args["closes"], 1)
    elif mutation == "cutoff":
        args["as_of"] = args["closes"][-2]
    elif mutation == "future_range":
        args["closes"] = args["closes"][:-1]
    elif mutation == "funding":
        raw = replace(
            src.raw,
            funding_rate=np.ones(src.raw.timestamps.shape),
            funding_available=np.ones(src.raw.timestamps.shape, dtype=bool),
            funding_event_count=np.ones(src.raw.timestamps.shape, dtype=np.int32),
        )
        args["sources"] = (_pin(raw, src.contract, args["clock"]),)
    elif mutation == "multiplier":
        args["sources"] = (
            _pin(src.raw, replace(src.contract, contract_multiplier=2), args["clock"]),
        )
    elif mutation == "listing":
        args["sources"] = (
            _pin(
                src.raw,
                replace(
                    src.contract, listed_at=datetime(2024, 1, 2, tzinfo=timezone.utc)
                ),
                args["clock"],
            ),
        )
    elif mutation == "duplicate":
        args["sources"] = (src, src)
    else:
        raw = replace(src.raw, volume=src.raw.volume * 2)
        group = replace(
            args["feature_groups"][0], sources=(_pin(raw, src.contract, args["clock"]),)
        )
        args["feature_groups"] = (group, args["feature_groups"][1])
    with pytest.raises(ValueError):
        build_native_fixed_spot_dataset(**args)


@pytest.mark.parametrize("kind", ["zero", "fractional", "masked", "bad_ohlcv"])
def test_resampled_adapter_rejects_invalid_members_or_values(kind):
    args, bars = _fixture()
    if kind == "zero":
        bars = replace(bars, member_counts=np.zeros_like(bars.member_counts))
    elif kind == "fractional":
        bars = replace(bars, member_counts=bars.member_counts.astype(float))
    elif kind == "masked":
        bars = replace(bars, member_counts=np.ma.array(bars.member_counts, mask=True))
    else:
        ohlcv = bars.ohlcv.copy()
        ohlcv[0, 5] = -1.0
        bars = replace(bars, ohlcv=ohlcv)
    with pytest.raises(ValueError):
        spot_raw_from_resampled(bars, clock=args["clock"], as_of=args["as_of"])


def test_multisymbol_roster_and_global_returns_are_causal():
    args, _ = _fixture()
    base = args["sources"][0]
    contract = replace(base.contract, symbol="SYNTH_B")
    raw = replace(
        base.raw,
        open=base.raw.open * 2,
        high=base.raw.high * 2,
        low=base.raw.low * 2,
        close=base.raw.close * 2,
    )
    second = _pin(raw, contract, args["clock"])
    groups = []
    for group in args["feature_groups"]:
        gsrc = group.sources[0]
        if group.clock == args["clock"]:
            other = second
        else:
            other_raw = replace(
                gsrc.raw,
                open=gsrc.raw.open * 2,
                high=gsrc.raw.high * 2,
                low=gsrc.raw.low * 2,
                close=gsrc.raw.close * 2,
            )
            other = _pin(other_raw, contract, group.clock)
        groups.append(replace(group, sources=(gsrc, other)))
    args.update(sources=(base, second), feature_groups=tuple(groups))
    dataset = build_native_fixed_spot_dataset(**args)
    assert dataset.symbols == ("SYNTH", "SYNTH_B")
    np.testing.assert_allclose(
        dataset.features[:, 0], dataset.features[:, 1], rtol=1e-6
    )
    expected = np.log(dataset.close[1:, 0] / dataset.close[:-1, 0])
    np.testing.assert_allclose(dataset.global_features[1:, 2], expected, rtol=1e-6)
    assert np.all(dataset.global_features[:, 3] == 0)
    with pytest.raises(ValueError, match="roster"):
        build_native_fixed_spot_dataset(**{**args, "sources": (second, base)})


def test_actual_capture_adapter_receipt_pipeline():
    from tests.integrations.test_binance_native_spot_capture import _encode, _message
    from trade_rl.data.contracts import VolumeUnit
    from trade_rl.integrations.binance.native_spot_capture import (
        resample_captured_spot_klines,
    )

    clock = NativeBarClock(NativeClockKind.UTC_WEEK)
    anchor = np.datetime64("2024-01-01", "ns")
    closes = anchor + np.arange(1, 5) * np.timedelta64(7, "D")
    receipts = closes + np.timedelta64(2, "s")
    as_of = receipts[-1]
    bars = resample_captured_spot_klines(
        captures=[
            (_encode(_message(t - np.timedelta64(7, "D"), t, interval="1w")), receipt)
            for t, receipt in zip(closes, receipts, strict=True)
        ],
        symbol="TESTUSDT",
        source_interval="1w",
        source_clock=clock,
        source_closes=closes,
        timestamp_unit="ms",
        target_clock=clock,
        target_closes=closes,
        as_of=as_of,
    )
    raw = spot_raw_from_resampled(
        bars, clock=clock, as_of=as_of, volume_unit=VolumeUnit.QUOTE_NOTIONAL
    )
    contract = InstrumentContract("TESTUSDT", volume_unit=VolumeUnit.QUOTE_NOTIONAL)
    source = _pin(raw, contract, clock)
    dataset = build_native_fixed_spot_dataset(
        clock=clock,
        closes=closes,
        sources=(source,),
        feature_groups=(
            ClockedNativeFeatureGroup(
                clock, (source,), (FeatureSpec("body", FeatureKind.BODY_RETURN),)
            ),
        ),
        as_of=as_of,
    )
    np.testing.assert_array_equal(dataset.volume[:, 0], np.full(4, 250.0))
    np.testing.assert_array_equal(dataset.available_at[:, 0], receipts)
    assert not dataset.resolved_array("information_available").any()
    assert not dataset.feature_available.any()
    assert dataset.identity_verified
    with pytest.raises(ValueError, match="unavailable"):
        run_single_symbol_replay(
            dataset,
            AlwaysLong(),
            start_index=0,
            stop_index=3,
            gross_budget=0.5,
            initial_capital=1000,
            execution_cost=native_cost(),
        )


@pytest.mark.parametrize("days,minutes", [(3, 60), (7, 15)])
def test_multiyear_native_cloud_warmup_and_literal_oracle(days, minutes):
    from trade_rl.data.features.indicators.ichimoku_cloud import visible_ichimoku_cloud

    anchor = np.datetime64("2024-01-01", "ns")
    n = 730 // days
    fine_count = n * days * 24 * 60 // minutes
    clock = (
        NativeBarClock(NativeClockKind.UTC_WEEK)
        if days == 7
        else NativeBarClock(NativeClockKind.FIXED_ELAPSED, days * 86400, anchor)
    )
    fine_clock = NativeBarClock(NativeClockKind.FIXED_ELAPSED, minutes * 60, anchor)
    fine_closes = anchor + np.arange(1, fine_count + 1) * np.timedelta64(minutes, "m")
    closes = anchor + np.arange(1, n + 1) * np.timedelta64(days, "D")
    index = np.arange(fine_count, dtype=np.float64)
    prices = 100 + index / 10000 + 4 * np.sin(index / 137)
    rows = np.column_stack(
        (prices, prices + 1, prices - 1, prices + 0.25, np.ones(fine_count), prices)
    )
    bars = resample_complete_native_bars(
        source_clock=fine_clock,
        source_closes=fine_closes,
        source_available_at=fine_closes,
        source_ohlcv=rows,
        target_clock=clock,
        target_closes=closes,
        as_of=closes[-1],
    )
    raw = spot_raw_from_resampled(bars, clock=clock, as_of=closes[-1])
    contract = InstrumentContract("SYNTH_LONG_HISTORY")
    source = _pin(raw, contract, clock)
    spec = FeatureSpec("unshifted", FeatureKind.ICHIMOKU_CLOUD_POSITION)
    dataset = build_native_fixed_spot_dataset(
        clock=clock,
        closes=closes,
        sources=(source,),
        feature_groups=(ClockedNativeFeatureGroup(clock, (source,), (spec,)),),
        as_of=closes[-1],
    )
    cloud = visible_ichimoku_cloud(raw)
    assert not dataset.feature_available[:51, 0, 0].any()
    assert dataset.feature_available[51:, 0, 0].all()
    assert not cloud.ready[:77].any()
    assert cloud.ready[77:].all()
    assert dataset.identity_verified
    for t in (77, n - 1):

        def mid(end, period):
            return (
                max(raw.high[end - period + 1 : end + 1])
                + min(raw.low[end - period + 1 : end + 1])
            ) * 0.5

        a = (mid(t, 9) + mid(t, 26)) * 0.5
        b = mid(t, 52)
        assert dataset.features[t, 0, 0] == pytest.approx(
            (raw.close[t] - (a + b) * 0.5) / raw.close[t], rel=1e-6
        )
        prior = t - 26
        assert cloud.span_a[t] == (mid(prior, 9) + mid(prior, 26)) * 0.5
        assert cloud.span_b[t] == mid(prior, 52)
    # The execution suffix receives the same warm-up, without inventing rows.
    suffix = build_native_fixed_spot_dataset(
        clock=clock,
        closes=closes[77:],
        sources=(source,),
        feature_groups=(ClockedNativeFeatureGroup(clock, (source,), (spec,)),),
        as_of=closes[-1],
    )
    np.testing.assert_array_equal(suffix.features, dataset.features[77:])
    np.testing.assert_array_equal(
        suffix.feature_available, dataset.feature_available[77:]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("tradable", 2),
        ("tradable", -1),
        ("tradable", np.nan),
        ("funding_available", 0),
        ("funding_available", np.nan),
        ("funding_event_count", 0.5),
        ("funding_event_count", -0.5),
        ("funding_event_count", 2**32),
        ("available_at", None),
        ("close", "101.0"),
        ("volume", 2**53 + 1),
    ],
)
def test_native_raw_adapter_refuses_coercion_before_pin(field, value):
    args, _ = _fixture(days=3)
    source = args["sources"][0]
    invalid = None if value is None else np.full(source.raw.timestamps.shape, value)
    with pytest.raises(ValueError, match="native raw"):
        replace(source.raw, **{field: invalid})


def test_native_builder_refuses_legacy_normalized_source_even_with_matching_pin():
    from dataclasses import fields

    from trade_rl.data.source import RawMarketSeries

    args, _ = _fixture(days=3)
    source = args["sources"][0]
    # The legacy constructor remains compatible; its normalized bytes/digest
    # cannot demonstrate that strict validation preceded conversion.
    legacy = RawMarketSeries(
        **{
            field.name: getattr(source.raw, field.name)
            for field in fields(RawMarketSeries)
        }
    )
    legacy_pin = _pin(legacy, source.contract, args["clock"])
    assert legacy_pin.expected_source_digest == source.expected_source_digest
    with pytest.raises(ValueError, match="strict native raw"):
        build_native_fixed_spot_dataset(**{**args, "sources": (legacy_pin,)})
    group = replace(args["feature_groups"][0], sources=(legacy_pin,))
    with pytest.raises(ValueError, match="strict native raw"):
        build_native_fixed_spot_dataset(
            **{**args, "feature_groups": (group, args["feature_groups"][1])}
        )
