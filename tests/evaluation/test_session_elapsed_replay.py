"""Explicit session schedules and ACT/365F metrics, independent of venue claims."""

import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.data.test_market_dataset_v2 import kwargs
from tests.data.test_native_execution_dataset import native_cost
from trade_rl.data.artifacts.codec import load_dataset_files, write_market_dataset_files
from trade_rl.data.market import MarketCalendarKind, MarketDataset
from trade_rl.evaluation.comparison.paired import compare_paired_returns
from trade_rl.evaluation.metrics import evaluate_performance
from trade_rl.evaluation.native_annualization import (
    evaluate_session_elapsed_annualization,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.evaluation.robustness.walk_forward.stitching import (
    FoldOOSResult,
    stitch_oos,
)
from trade_rl.evaluation.series import ReturnKind, SessionElapsedReturnSeries
from trade_rl.evaluation.session_returns import (
    load_session_elapsed_return_payload,
    run_session_elapsed_portfolio_replay,
    session_elapsed_return_payload,
)
from trade_rl.strategies.portfolio import PortfolioTargetProposal


def _values():
    times = np.array(
        [
            "2026-01-02T16:00",
            "2026-01-05T09:00",
            "2026-01-05T10:00",
            "2026-01-05T11:00",
            "2026-01-06T09:00",
            "2026-01-06T10:00",
        ],
        dtype="datetime64[ns]",
    )
    prices = np.full((6, 1), 100.0)
    return kwargs(n_bars=6, n_symbols=1) | {
        "timestamps": times,
        "available_at": times[:, None].copy(),
        "calendar_kind": MarketCalendarKind.SESSION,
        "nominal_bar_hours": 1.0,
        "periods_per_year": 1638,
        "open": prices.copy(),
        "high": prices.copy(),
        "low": prices.copy(),
        "close": prices.copy(),
        "cash_rate": np.full(6, 0.365),
        "funding_due": np.zeros((6, 1), dtype=bool),
    }


def _dataset(**changes):
    return MarketDataset(**(_values() | changes)).with_content_identity(
        {"fixture": "session-elapsed"}
    )


def _boundaries(dataset):
    return tuple(int(x) for x in dataset.timestamps.view(np.int64))


class _Policy:
    def __init__(self, weight=0.0):
        self.weight = weight
        self.calls = []

    def decide_portfolio(self, snapshot):
        self.calls.append(snapshot.index)
        return PortfolioTargetProposal((self.weight,))


def _run(dataset, policy, **changes):
    return run_session_elapsed_portfolio_replay(
        dataset,
        policy,
        **(
            {
                "start_index": 0,
                "stop_index": 5,
                "expected_boundaries_ns": _boundaries(dataset),
                "expected_nominal_bar_seconds": 3600,
                "initial_capital": 1000.0,
                "execution_cost": native_cost(),
            }
            | changes
        ),
    )


def test_session_elapsed_replay_actual_weekend_cash_and_payload(tmp_path):
    dataset = _dataset()
    write_market_dataset_files(tmp_path, dataset)
    restored = load_dataset_files(tmp_path)
    policy = _Policy()
    result = _run(restored, policy)
    assert policy.calls == [0, 1, 2, 3]
    assert isinstance(result.returns, SessionElapsedReturnSeries)
    assert result.returns.boundaries_ns == _boundaries(restored)
    hours = (65, 1, 1, 22, 1)
    years = Fraction(sum(hours), 8760)
    wealth = Fraction(1000)
    expected = []
    for n in hours:
        rate = Fraction(n, 24000)
        expected.append(float(rate))
        wealth *= 1 + rate
    assert result.returns.exact_elapsed_years == years
    assert result.returns.periods_per_year == Fraction(5) / years
    assert result.returns.values == pytest.approx(expected, abs=1e-13)
    assert result.book.portfolio_value == pytest.approx(float(wealth), rel=1e-13)
    assert result.book.exact_quantities == (0,)
    payload = session_elapsed_return_payload(restored, result.returns, start_index=0)
    assert load_session_elapsed_return_payload(restored, payload) == result.returns
    assert payload["elapsed_years"] == {
        "numerator": years.numerator,
        "denominator": years.denominator,
    }
    for change in (
        {"dataset_id": "0" * 64},
        {"start_index": True},
        {"boundaries_ns": list(result.returns.boundaries_ns[:-1])},
        {"periods_per_year": {"numerator": 1638, "denominator": 1}},
        {"values": [True] * 5},
    ):
        with pytest.raises(ValueError):
            load_session_elapsed_return_payload(restored, payload | change)
    legacy = run_shared_portfolio_replay(
        restored,
        _Policy(),
        start_index=0,
        stop_index=5,
        initial_capital=1000.0,
        execution_cost=native_cost(),
        settle_terminal_position=True,
        strict_spot_cash=True,
        capture_ledger_evidence=True,
    )
    assert legacy.returns.values == result.returns.values
    assert legacy.book.portfolio_value == result.book.portfolio_value
    assert legacy.ledger_evidence == result.ledger_evidence
    assert legacy.returns.periods_per_year == 1638
    with pytest.raises(ValueError, match="session"):
        compare_paired_returns(result.returns, result.returns)
    with pytest.raises(ValueError, match="session"):
        stitch_oos((FoldOOSResult(0, 0, 5, result.returns),))


def test_session_duration_risk_independent_oracle():
    dataset = _dataset()
    boundaries = _boundaries(dataset)
    values = (0.01, -0.002, 0.003, -0.005, 0.001)
    actual = evaluate_session_elapsed_annualization(values, boundaries_ns=boundaries)
    years = 90 / 8760
    drift = sum(values) / years
    volatility = math.sqrt(
        sum(
            (r - drift * h / 8760) ** 2
            for r, h in zip(values, (65, 1, 1, 22, 1), strict=True)
        )
        / years
    )
    downside = math.sqrt(sum(min(r, 0) ** 2 for r in values) / years)
    assert actual.elapsed_years == Fraction(90, 8760)
    assert actual.annualized_arithmetic_return == pytest.approx(drift)
    assert actual.annualized_volatility == pytest.approx(volatility)
    assert actual.sharpe == pytest.approx(drift / volatility)
    assert actual.sortino == pytest.approx(drift / downside)
    result = SessionElapsedReturnSeries.from_boundaries(
        values, boundaries_ns=boundaries
    )
    metrics = evaluate_performance(result)
    assert metrics.sharpe == actual.sharpe
    assert metrics.sortino == actual.sortino
    assert metrics.periods_per_year == actual.periods_per_year


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_close",
        "wrong_close",
        "late_terminal",
        "unverified",
        "continuous",
        "funding",
        "short_cost",
        "too_short_interval",
        "spot_units",
    ],
)
def test_session_refuses_unsupported_scope_before_actor(mutation):
    dataset = _dataset()
    boundaries = _boundaries(dataset)
    options = {}
    if mutation == "missing_close":
        options["expected_boundaries_ns"] = boundaries[:-1]
    elif mutation == "wrong_close":
        options["expected_boundaries_ns"] = (
            boundaries[:2] + (boundaries[2] + 1,) + boundaries[3:]
        )
    elif mutation == "late_terminal":
        available = dataset.available_at.copy()
        available[5, 0] += np.timedelta64(1, "ns")
        dataset = _dataset(available_at=available)
    elif mutation == "unverified":
        dataset = replace(dataset, identity_payload_json=None)
    elif mutation == "continuous":
        times = np.datetime64("2026-01-01", "ns") + np.arange(6) * np.timedelta64(
            1, "h"
        )
        dataset = _dataset(
            timestamps=times,
            available_at=times[:, None],
            periods_per_year=8760,
            calendar_kind=MarketCalendarKind.CONTINUOUS,
        )
    elif mutation == "funding":
        count = np.zeros((6, 1), dtype=np.int32)
        count[5, 0] = 1
        dataset = _dataset(funding_event_count=count, funding_due=count > 0)
    elif mutation == "short_cost":
        options["execution_cost"] = replace(native_cost(), allow_short=True)
    elif mutation == "too_short_interval":
        dataset = _dataset(nominal_bar_hours=2.0)
    else:
        from trade_rl.data.contracts import VolumeUnit

        dataset = _dataset(volume_units=(VolumeUnit.CONTRACTS,))
    policy = _Policy()
    with pytest.raises(ValueError):
        _run(dataset, policy, **options)
    assert not policy.calls


@pytest.mark.parametrize(
    "boundaries",
    [
        (0, 0),
        (1, 0),
        (True, 1),
        (-(2**63), 0),
        (-(2**63) + 1, 2**63 - 1),
        (0, 2**63),
        [0, 1],
    ],
)
def test_session_rejects_invalid_exact_boundaries(boundaries):
    with pytest.raises(ValueError):
        SessionElapsedReturnSeries.from_boundaries((0.01,), boundaries_ns=boundaries)
    with pytest.raises(ValueError):
        evaluate_session_elapsed_annualization((0.01,), boundaries_ns=boundaries)


def test_integer_observed_frequency_remains_duration_bound():
    days = 86400 * 10**9
    series = SessionElapsedReturnSeries.from_boundaries(
        (0.01,), boundaries_ns=(0, days)
    )
    assert type(series.periods_per_year) is Fraction
    assert series.periods_per_year == 365
    assert series.kind is ReturnKind.BASE_BAR
    with pytest.raises(ValueError, match="session"):
        _ = series.annualization_periods_per_year


@pytest.mark.parametrize("nominal", [True, 0, -1, 3600.0, 60, 2**63])
def test_session_refuses_invalid_caller_nominal_clock(nominal):
    policy = _Policy()
    with pytest.raises(ValueError, match="nominal"):
        _run(_dataset(), policy, expected_nominal_bar_seconds=nominal)
    assert not policy.calls


def test_session_refuses_explicitly_unavailable_ontime_terminal():
    information = np.ones((6, 1), dtype=bool)
    information[5, 0] = False
    dataset = _dataset(information_available=information)
    policy = _Policy()
    with pytest.raises(ValueError, match="available"):
        _run(dataset, policy)
    assert not policy.calls


def test_session_does_not_publish_unsettled_terminal_result():
    sell = np.ones((6, 1), dtype=bool)
    sell[5, 0] = False
    dataset = _dataset(sell_allowed=sell)
    policy = _Policy(0.25)
    with pytest.raises(ValueError, match="settled window"):
        _run(dataset, policy)
    assert policy.calls == [0, 1, 2, 3]


def test_session_elapsed_signed_returns_and_long_history_proportional_drift():
    # Exact binary fractions proportional to 1h and 65h durations.
    durations = (1, 65, 1, 22) * 2048
    boundaries = [0]
    for hours in durations:
        boundaries.append(boundaries[-1] + hours * 3600 * 10**9)
    values = tuple(hours / 2**20 for hours in durations)
    actual = evaluate_session_elapsed_annualization(
        values, boundaries_ns=tuple(boundaries)
    )
    assert actual.elapsed_years == Fraction(sum(durations), 8760)
    assert actual.annualized_volatility == 0.0
    assert actual.sharpe == 0.0
    assert actual.annualized_arithmetic_return == pytest.approx(8760 / 2**20)


@pytest.mark.parametrize(
    "values", [(), (True,), (math.nan,), (math.inf,), (-1.0,), ("0.01",)]
)
def test_session_refuses_invalid_returns(values):
    boundaries = (0, 3600 * 10**9)
    with pytest.raises(ValueError):
        SessionElapsedReturnSeries.from_boundaries(values, boundaries_ns=boundaries)
    with pytest.raises(ValueError):
        evaluate_session_elapsed_annualization(values, boundaries_ns=boundaries)


def test_session_next_open_long_fee_and_gap_carry_literal_ledger():
    class HoldAfterEntry:
        def decide_portfolio(self, snapshot):
            if snapshot.index == 0:
                return PortfolioTargetProposal((0.25,))
            return PortfolioTargetProposal(
                snapshot.account.weights, hold_current_quantity_symbols=(0,)
            )

    dataset = _dataset()
    result = _run(dataset, HoldAfterEntry(), execution_cost=native_cost(fee_rate=0.001))
    assert result.ledger_evidence is not None
    cash = Fraction(1000)
    equity = cash
    for index, (row, hours) in enumerate(
        zip(result.ledger_evidence.intervals.intervals, (65, 1, 1, 22, 1), strict=True),
        1,
    ):
        fee = Fraction(1, 4) if index in (1, 5) else Fraction(0)
        trade = -250 if index == 1 else 250 if index == 5 else 0
        held = index < 5
        gap_interest = cash * Fraction(hours - 1, 24000)
        processing_basis = cash + trade - fee
        processing_interest = processing_basis / 24000
        cash = processing_basis + gap_interest + processing_interest
        after = cash + (250 if held else 0)
        assert row.interval_cost == pytest.approx(float(fee))
        assert row.interval_cash_interest == pytest.approx(
            float(gap_interest + processing_interest)
        )
        assert row.cash_after == pytest.approx(float(cash))
        assert row.portfolio_value_after == pytest.approx(float(after))
        assert tuple(Fraction(q) for q in row.exact_quantities_after) == (
            Fraction(5, 2) if held else Fraction(0),
        )
        assert row.interval_net_return == pytest.approx(float(after / equity - 1))
        equity = after
    assert result.book.exact_quantities == (0,)
    assert result.book.portfolio_value == pytest.approx(float(cash))
