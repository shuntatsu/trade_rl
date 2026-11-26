"""Opt-in native execution dataset/return identity: synthetic data only."""

from __future__ import annotations

import json
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.data.test_market_dataset_v2 import kwargs
from tests.evaluation.test_single_symbol_replay import AlwaysLong
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts.codec import load_dataset_files, write_market_dataset_files
from trade_rl.data.artifacts.verified_bytes import load_dataset_from_verified_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.replay import run_single_symbol_replay
from trade_rl.evaluation.series import ReturnKind, ReturnSeries
from trade_rl.simulation import ExecutionCostConfig, MarketExecutor


def native_values(days=7, *, weekly=True):
    values = kwargs(n_bars=6, n_symbols=1)
    anchor = np.datetime64("2026-01-05", "ns")
    clock = (
        NativeBarClock(NativeClockKind.UTC_WEEK)
        if weekly
        else NativeBarClock(NativeClockKind.FIXED_ELAPSED, days * 86400, anchor)
    )
    timestamps = anchor + np.arange(1, 7) * np.timedelta64(days, "D")
    values.update(
        timestamps=timestamps,
        periods_per_year=Fraction(365, days),
        native_clock=clock,
        available_at=timestamps[:, None].copy(),
        funding_price_rate=np.zeros((6, 1)),
        funding_due=np.zeros((6, 1), dtype=bool),
        funding_event_count=np.zeros((6, 1), dtype=np.int32),
    )
    return values


@pytest.mark.parametrize("days,weekly", [(3, False), (7, False), (7, True)])
def test_native_identity_and_artifact_roundtrip(tmp_path, days, weekly):
    values = native_values(days, weekly=weekly)
    d = MarketDataset(**values).with_content_identity({"source": "synthetic"})
    assert d.identity_verified
    assert d.regular_cadence and d.bar_hours == days * 24
    assert d.periods_per_year == Fraction(365, days)
    assert d.elapsed_year_fraction(1, 3) == pytest.approx(2 * days / 365)
    meta = json.loads(d.identity_payload_json)
    assert meta["schema"] == "market_dataset_identity_v8"
    assert meta["dataset_contract"]["periods_per_year"] == {
        "numerator": 365,
        "denominator": days,
    }
    assert meta["dataset_contract"]["native_clock"]["kind"] == d.native_clock.kind.value
    result = write_market_dataset_files(tmp_path, d)
    assert result.schema_version == "market_dataset_artifact_v5"
    restored = load_dataset_files(tmp_path)
    from_bytes, digest = load_dataset_from_verified_bytes(
        result.manifest_path.read_bytes(), result.arrays_path.read_bytes()
    )
    for x in (restored, from_bytes):
        assert x.dataset_id == d.dataset_id and x.identity_verified
        assert x.native_clock == d.native_clock
        assert type(x.periods_per_year) is Fraction
        assert x.periods_per_year == d.periods_per_year
    assert digest == result.artifact_digest


def test_explicit_clock_is_identity_not_calendar_approximation():
    weekly = MarketDataset(**native_values()).with_content_identity()
    fixed = MarketDataset(**native_values(7, weekly=False)).with_content_identity()
    assert weekly.dataset_id != fixed.dataset_id
    with pytest.raises(ValueError):
        replace(weekly, native_clock=fixed.native_clock)


@pytest.mark.parametrize(
    "mutation",
    [
        "rounded",
        "missing_clock",
        "month",
        "session",
        "gap",
        "phase",
        "missing_availability",
        "missing_funding",
        "inconsistent_funding",
        "missing_count",
    ],
)
def test_reject_invalid_native_admission(mutation):
    v = native_values()
    if mutation == "rounded":
        v["periods_per_year"] = 52
    elif mutation == "missing_clock":
        v.pop("native_clock")
    elif mutation == "month":
        v["native_clock"] = NativeBarClock(NativeClockKind.UTC_MONTH)
    elif mutation == "session":
        v["calendar_kind"] = "session_calendar"
    elif mutation == "gap":
        v["timestamps"][3:] += np.timedelta64(7, "D")
    elif mutation == "phase":
        v["timestamps"] += np.timedelta64(1, "D")
    elif mutation == "missing_availability":
        v.pop("available_at")
    elif mutation == "missing_funding":
        v.pop("funding_price_rate")
    elif mutation == "missing_count":
        v.pop("funding_event_count")
    else:
        v["funding_price_rate"][1, 0] = 0.1
    with pytest.raises(ValueError):
        MarketDataset(**v)


def native_cost(**kwargs):
    return replace(
        ExecutionCostConfig.zero(),
        allow_short=False,
        processing_bar_volume_capacity=False,
        **kwargs,
    )


def test_native_replay_and_metrics_use_exact_frequency():
    d = MarketDataset(**native_values()).with_content_identity()
    result = run_single_symbol_replay(
        d,
        AlwaysLong(),
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1000,
        execution_cost=native_cost(),
    )
    assert len(result.returns.values) == 5
    assert result.returns.periods_per_year == Fraction(365, 7)
    assert result.returns.annualization_periods_per_year == float(Fraction(365, 7))
    assert result.book.quantities[0] == 5
    assert result.book.portfolio_value == 1025
    metrics = evaluate_performance(result.returns)
    assert metrics.periods_per_year == Fraction(365, 7)
    assert type(metrics.periods_per_year) is Fraction


def test_native_return_series_exact_frequency():
    result = ReturnSeries((0.01, -0.02, 0.03), ReturnKind.BASE_BAR, Fraction(365, 3))
    assert result.annualization_periods_per_year == 365 / 3
    assert evaluate_performance(result).periods_per_year == Fraction(365, 3)


def test_legacy_artifact_keeps_old_fields(tmp_path):
    d = MarketDataset(**kwargs()).with_content_identity()
    result = write_market_dataset_files(tmp_path, d)
    assert result.schema_version == "market_dataset_artifact_v3"
    manifest = json.loads(result.manifest_path.read_bytes())
    assert "native_clock" not in manifest["dataset"]
    assert type(manifest["dataset"]["periods_per_year"]) is int
    assert load_dataset_files(tmp_path).dataset_id == d.dataset_id


def test_legacy_manifest_cannot_smuggle_native_clock(tmp_path):
    d = MarketDataset(**kwargs()).with_content_identity()
    result = write_market_dataset_files(tmp_path, d)
    manifest = json.loads(result.manifest_path.read_bytes())
    manifest["dataset"]["native_clock"] = None
    manifest.pop("artifact_digest")
    manifest["artifact_digest"] = content_digest(manifest)
    result.manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_dataset_files(tmp_path)


@pytest.mark.parametrize(
    "changes",
    [
        {"allow_short": True},
        {"processing_bar_volume_capacity": True},
        {"order_type": "limit"},
        {"max_leverage": 2.0},
        {"slippage_std": 0.01},
    ],
)
def test_native_rejects_unsafe_execution_configuration(changes):
    values = MarketDataset(**native_values()).with_content_identity()
    with pytest.raises(ValueError, match="native fixed execution requires"):
        MarketExecutor(values, replace(native_cost(), **changes))


def test_native_refuses_funding_without_intrabar_event_order():
    v = native_values()
    v["funding_event_count"][1, 0] = 1
    v["funding_due"][1, 0] = True
    v["funding_price_rate"][1, 0] = 0.05
    d = MarketDataset(**v).with_content_identity()
    with pytest.raises(ValueError, match="unsupported funding"):
        run_single_symbol_replay(
            d,
            AlwaysLong(),
            start_index=0,
            stop_index=5,
            gross_budget=0.5,
            initial_capital=1000,
            execution_cost=native_cost(),
        )


def test_native_refuses_late_publication():
    v = native_values()
    v["available_at"][1, 0] += np.timedelta64(1, "ns")
    d = MarketDataset(**v).with_content_identity()
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


@pytest.mark.parametrize("days,weekly", [(3, False), (7, True)])
def test_native_flat_cash_carry_exact_elapsed(days, weekly):
    from tests.evaluation.test_single_symbol_replay import IntentSequence
    from trade_rl.strategies.position_intent import PositionIntent

    v = native_values(days, weekly=weekly)
    v["cash_rate"] = np.full(6, 0.365)
    d = MarketDataset(**v).with_content_identity()
    result = run_single_symbol_replay(
        d,
        IntentSequence((PositionIntent.FLAT,)),
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1000,
        execution_cost=native_cost(),
    )
    assert result.book.portfolio_value == pytest.approx(
        1000 * (1 + 0.365 * days / 365) ** 5
    )
    assert result.returns.values == pytest.approx((0.365 * days / 365,) * 5)


def test_native_fee_and_terminal_settlement():
    d = MarketDataset(**native_values()).with_content_identity()
    result = run_single_symbol_replay(
        d,
        AlwaysLong(),
        start_index=0,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1000,
        execution_cost=native_cost(fee_rate=0.01),
        settle_terminal_position=True,
    )
    assert result.book.quantities[0] == 0
    # Buy five at 100 then close at the last bar's open (104).
    assert result.book.portfolio_value == pytest.approx(1000 + 5 * 4 - 5 - 5.2)
    assert result.diagnostics.total_cost == pytest.approx(10.2)


def test_native_execution_requires_verified_dataset_identity():
    with pytest.raises(ValueError, match="verified Dataset identity"):
        MarketExecutor(MarketDataset(**native_values()), native_cost())


def test_native_executor_rejects_dataset_swap():
    d = MarketDataset(**native_values()).with_content_identity()
    executor = MarketExecutor(d, native_cost())
    executor.dataset = MarketDataset(**kwargs(n_bars=6, n_symbols=1))
    with pytest.raises(ValueError, match="fresh executor"):
        executor._validate_native_fixed_configuration()


def test_native_return_payload_roundtrip_and_exact_interval():
    from trade_rl.evaluation.native_fixed_returns import (
        load_native_fixed_return_payload,
        native_fixed_return_payload,
    )

    d = MarketDataset(**native_values()).with_content_identity()
    replay = run_single_symbol_replay(
        d,
        AlwaysLong(),
        start_index=1,
        stop_index=5,
        gross_budget=0.5,
        initial_capital=1000,
        execution_cost=native_cost(),
    )
    payload = native_fixed_return_payload(d, replay.returns, start_index=1)
    assert payload["schema_version"] == "native_fixed_return_series_v1"
    assert payload["stop_index"] == 5
    assert payload["periods_per_year"] == {"numerator": 365, "denominator": 7}
    assert payload["elapsed_years"] == {"numerator": 28, "denominator": 365}
    restored = load_native_fixed_return_payload(d, json.loads(json.dumps(payload)))
    assert restored == replay.returns
    for key, bad in (
        ("dataset_id", "0" * 64),
        ("periods_per_year", 52),
        ("elapsed_years", 4 / 52),
        ("start_index", True),
        ("stop_index", 4),
        ("clock", {"kind": "utc_month"}),
        ("values", [True] * 4),
        ("schema_version", "native_fixed_return_series_v0"),
    ):
        with pytest.raises(ValueError):
            load_native_fixed_return_payload(d, {**payload, key: bad})
    with pytest.raises(ValueError):
        load_native_fixed_return_payload(d, {**payload, "unknown": 0})


@pytest.mark.parametrize("days,weekly", [(3, False), (7, True)])
def test_native_next_open_fill_does_not_use_processing_volume(days, weekly):
    outputs = []
    for processing_volume in (0.0, 1000000.0):
        v = native_values(days, weekly=weekly)
        v["volume"][0, 0] = 1000
        v["volume"][1, 0] = processing_volume
        d = MarketDataset(**v).with_content_identity()
        replay = run_single_symbol_replay(
            d,
            AlwaysLong(),
            start_index=0,
            stop_index=1,
            gross_budget=0.5,
            initial_capital=1000,
            execution_cost=native_cost(max_participation_rate=0.001),
        )
        outputs.append(
            (float(replay.book.quantities[0]), replay.diagnostics.total_cost)
        )
    assert outputs == [(1.0, 0.0), (1.0, 0.0)]
