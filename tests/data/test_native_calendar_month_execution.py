"""Actual complete UTC-month spot replay; no fabricated execution bars."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.data.test_long_native_alignment import _source
from tests.data.test_native_execution_dataset import native_cost
from trade_rl.data.artifacts.codec import load_dataset_files, write_market_dataset_files
from trade_rl.data.artifacts.verified_bytes import load_dataset_from_verified_bytes
from trade_rl.data.build import native_spot
from trade_rl.data.contracts import FeatureKind, FeatureSpec, InstrumentContract
from trade_rl.data.features.clocked_overlay import (
    ClockedNativeFeatureGroup,
    ClockedNativeSource,
    clocked_native_source_digest,
)
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.simulation import MarketExecutor
from trade_rl.strategies.portfolio import PortfolioTargetProposal


def _args(*, late=False):
    closes = np.arange("2024-01", "2024-08", dtype="datetime64[M]").astype(
        "datetime64[ns]"
    )
    clock = NativeBarClock(NativeClockKind.UTC_MONTH)
    receipts = closes.copy()
    if late:
        receipts[2] += np.timedelta64(1, "ns")
    raw = _source(closes, receipts)
    prices = np.arange(100.0, 107.0)
    raw = replace(
        raw,
        open=prices,
        high=prices + 1,
        low=prices - 1,
        close=prices,
        volume=np.full(7, 10000.0),
    )
    contract = InstrumentContract("SYNTH")
    source = ClockedNativeSource(
        contract, raw, clocked_native_source_digest(raw, contract=contract, clock=clock)
    )
    group = ClockedNativeFeatureGroup(
        clock, (source,), (FeatureSpec("body", FeatureKind.BODY_RETURN),)
    )
    return dict(
        clock=clock,
        closes=closes,
        sources=(source,),
        feature_groups=(group,),
        as_of=closes[-1] + np.timedelta64(1, "s"),
    )


def _dataset(**kwargs):
    return native_spot.build_native_calendar_month_spot_dataset(**_args(**kwargs))


class _Hold:
    def decide_portfolio(self, snapshot):
        return PortfolioTargetProposal(
            snapshot.account.weights if snapshot.index > 0 else (0.5,),
            hold_current_quantity_symbols=(0,) if snapshot.index > 0 else (),
        )


class _Flat:
    def decide_portfolio(self, snapshot):
        return PortfolioTargetProposal((0.0,))


def _replay(dataset, policy=None, **kwargs):
    return run_shared_portfolio_replay(
        dataset,
        _Hold() if policy is None else policy,
        start_index=0,
        stop_index=6,
        initial_capital=1000.0,
        execution_cost=native_cost(),
        native_calendar_month=True,
        **kwargs,
    )


def test_month_dataset_identity_and_roundtrip(tmp_path):
    d = _dataset()
    assert d.identity_verified and not d.regular_cadence
    assert d.periods_per_year == Fraction(7 * 365, 213)
    assert d.nominal_bar_hours is None
    with pytest.raises(ValueError, match="fixed"):
        _ = d.bar_hours
    with pytest.raises(ValueError, match="fixed"):
        d.bars_for_hours(720)
    assert json.loads(d.identity_payload_json)["schema"] == "market_dataset_identity_v9"
    result = write_market_dataset_files(tmp_path, d)
    assert result.schema_version == "market_dataset_artifact_v6"
    for restored in (
        load_dataset_files(tmp_path),
        load_dataset_from_verified_bytes(
            result.manifest_path.read_bytes(), result.arrays_path.read_bytes()
        )[0],
    ):
        assert restored.identity_verified and restored.dataset_id == d.dataset_id
        assert (
            restored.native_clock == d.native_clock
            and restored.periods_per_year == d.periods_per_year
        )
    with pytest.raises(ValueError, match="native execution"):
        native_spot.build_native_fixed_spot_dataset(**_args())


def test_actual_month_replay_matches_literal_cash_and_elapsed_returns():
    from trade_rl.evaluation.native_calendar_returns import (
        load_native_calendar_month_return_payload,
        native_calendar_month_return_payload,
    )
    from trade_rl.evaluation.series import CalendarMonthReturnSeries

    d = _dataset()
    result = _replay(d, settle_terminal_position=True, capture_ledger_evidence=True)
    assert type(result.returns) is CalendarMonthReturnSeries
    assert result.returns.closes_ns == tuple(
        int(x) for x in d.timestamps[1:].view(np.int64)
    )
    assert result.returns.periods_per_year == Fraction(6 * 365, 182)
    # Buy five units at actual next month's open 101, sell at the last open 106.
    expected_equity = [1000.0] + [1000.0 + 5 * (p - 101.0) for p in range(101, 107)]
    expected_returns = tuple(
        b / a - 1
        for a, b in zip(expected_equity[:-1], expected_equity[1:], strict=True)
    )
    assert result.returns.values == pytest.approx(expected_returns)
    assert result.book.portfolio_value == pytest.approx(1025.0)
    assert result.book.quantities[0] == 0
    assert result.ledger_evidence is not None
    days = (31, 29, 31, 30, 31, 30)
    vals = result.returns.values
    years = sum(days) / 365
    drift = sum(vals) / years
    vol = math.sqrt(
        sum((v - drift * t / 365) ** 2 for v, t in zip(vals, days, strict=True)) / years
    )
    metrics = evaluate_performance(result.returns)
    assert metrics.sharpe == pytest.approx(drift / vol)
    assert metrics.periods_per_year == Fraction(6 * 365, 182)
    assert metrics.total_return == pytest.approx(0.025)
    with pytest.raises(ValueError, match="calendar"):
        _ = result.returns.annualization_periods_per_year
    payload = native_calendar_month_return_payload(d, result.returns, start_index=0)
    assert load_native_calendar_month_return_payload(d, payload) == result.returns
    with pytest.raises(ValueError):
        load_native_calendar_month_return_payload(
            d, payload | {"elapsed_years": {"numerator": 1, "denominator": 2}}
        )


def test_calendar_cash_accrues_by_actual_month_duration():
    d = _dataset()
    d = replace(
        d,
        dataset_id="0" * 64,
        identity_payload_json=None,
        cash_rate=np.full(d.n_bars, 0.365),
    ).with_content_identity()
    result = _replay(d, _Flat())
    days = (31, 29, 31, 30, 31, 30)
    assert result.returns.values == pytest.approx(tuple(0.001 * x for x in days))
    assert result.book.cash == pytest.approx(
        1000.0 * math.prod(1 + 0.001 * x for x in days)
    )


def test_month_requires_explicit_replay_and_refuses_delayed_mark():
    d = _dataset()
    with pytest.raises(ValueError, match="calendar month"):
        MarketExecutor(d, native_cost())
    with pytest.raises(ValueError, match="calendar month"):
        run_shared_portfolio_replay(
            d,
            _Flat(),
            start_index=0,
            stop_index=6,
            initial_capital=1000.0,
            execution_cost=native_cost(),
        )
    with pytest.raises(ValueError, match="unavailable"):
        _replay(_dataset(late=True))
    with pytest.raises(ValueError, match="calendar month"):
        run_shared_portfolio_replay(
            d,
            _Flat(),
            start_index=0,
            stop_index=6,
            initial_capital=1000.0,
            execution_cost=replace(native_cost(), allow_short=True),
            native_calendar_month=True,
        )


def test_missing_month_and_forged_frequency_refused():
    args = _args()
    with pytest.raises(ValueError):
        native_spot.build_native_calendar_month_spot_dataset(
            **(args | {"closes": args["closes"][[0, 1, 3, 4, 5, 6]]})
        )
    d = _dataset()
    with pytest.raises(ValueError):
        replace(d, periods_per_year=Fraction(12))


def test_calendar_month_subset_returns_and_legacy_refusals():
    from trade_rl.evaluation.comparison.paired import compare_paired_returns
    from trade_rl.evaluation.native_calendar_returns import (
        load_native_calendar_month_return_payload,
        native_calendar_month_return_payload,
    )
    from trade_rl.evaluation.native_fixed_returns import native_fixed_return_payload
    from trade_rl.evaluation.robustness.walk_forward.stitching import (
        FoldOOSResult,
        stitch_oos,
    )
    from trade_rl.evaluation.series import CalendarMonthReturnSeries

    d = _dataset()
    result = run_shared_portfolio_replay(
        d,
        _Flat(),
        start_index=1,
        stop_index=3,
        execution_cost=native_cost(),
        native_calendar_month=True,
    )
    assert isinstance(result.returns, CalendarMonthReturnSeries)
    assert result.returns.exact_elapsed_years == Fraction(60, 365)
    assert result.returns.periods_per_year == Fraction(730, 60)
    payload = native_calendar_month_return_payload(d, result.returns, start_index=1)
    assert load_native_calendar_month_return_payload(d, payload) == result.returns
    with pytest.raises(ValueError, match="Dataset interval"):
        native_calendar_month_return_payload(d, result.returns, start_index=0)
    with pytest.raises(ValueError, match="Dataset interval"):
        load_native_calendar_month_return_payload(d, payload | {"dataset_id": "1" * 64})
    with pytest.raises(ValueError, match="native fixed"):
        native_fixed_return_payload(d, result.returns, start_index=1)
    with pytest.raises(ValueError, match="calendar month"):
        compare_paired_returns(result.returns, result.returns)
    with pytest.raises(ValueError, match="calendar month"):
        stitch_oos((FoldOOSResult(0, 1, 3, result.returns),))


def test_calendar_month_integer_observed_frequency_still_duration_bound():
    from trade_rl.evaluation.series import CalendarMonthReturnSeries

    closes = np.arange("2023-02", "2024-02", dtype="datetime64[M]")
    result = CalendarMonthReturnSeries.from_closes((0.01,) * 12, closes)
    assert result.periods_per_year == Fraction(12)
    assert result.exact_elapsed_years == Fraction(1)
    assert evaluate_performance(result).periods_per_year == Fraction(12)
    with pytest.raises(ValueError, match="duration-aware"):
        _ = result.annualization_periods_per_year


@pytest.mark.parametrize(
    "values,closes",
    [
        ((True,), ["2024-02-01"]),
        ((float("nan"),), ["2024-02-01"]),
        ((-1.0,), ["2024-02-01"]),
        ((0.0, 0.0), ["2024-02-01", "2024-04-01"]),
        ((0.0,), ["2024-02-02"]),
    ],
)
def test_calendar_month_return_admission(values, closes):
    from trade_rl.evaluation.series import CalendarMonthReturnSeries

    with pytest.raises(ValueError):
        CalendarMonthReturnSeries.from_closes(
            values, np.array(closes, dtype="datetime64[D]")
        )


def test_calendar_month_artifact_cannot_use_fixed_schema(tmp_path):
    from trade_rl.artifacts.canonical import canonical_json_bytes
    from trade_rl.artifacts.hashing import content_digest

    d = _dataset()
    result = write_market_dataset_files(tmp_path, d)
    manifest = json.loads(result.manifest_path.read_bytes())
    manifest["schema_version"] = "market_dataset_artifact_v5"
    manifest.pop("artifact_digest")
    manifest["artifact_digest"] = content_digest(manifest)
    raw = canonical_json_bytes(manifest)
    with pytest.raises(ValueError, match="schema"):
        load_dataset_from_verified_bytes(raw, result.arrays_path.read_bytes())
    result.manifest_path.write_bytes(raw)
    with pytest.raises(ValueError, match="schema"):
        load_dataset_files(tmp_path)


def test_month_risk_requires_native_clock():
    from trade_rl.risk.inputs import RollingPortfolioRiskInputsProvider

    with pytest.raises(ValueError, match="matching native clock"):
        RollingPortfolioRiskInputsProvider().inputs(_dataset(), index=3)


def test_month_rejects_default_session():
    from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession

    with pytest.raises(ValueError, match="calendar month"):
        SharedPortfolioSession(
            _dataset(), start_index=0, stop_index=3, execution_cost=native_cost()
        )


def test_month_rejects_non_spot_multipliers_and_contract_volume():
    from trade_rl.data.contracts import VolumeUnit

    dataset = _dataset()
    for changes in (
        {"contract_multipliers": np.array([2.0])},
        {"volume_units": (VolumeUnit.CONTRACTS,)},
    ):
        changed = replace(
            dataset, dataset_id="0" * 64, identity_payload_json=None, **changes
        ).with_content_identity()
        with pytest.raises(ValueError, match="unit-multiplier Spot"):
            MarketExecutor(changed, native_cost(), native_calendar_month=True)
