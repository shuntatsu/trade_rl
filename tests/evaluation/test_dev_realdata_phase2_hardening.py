"""Regression tests for Phase 2 selection, isolation and diagnostic contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from tests.evaluation.test_dev_realdata_phase2_trials import (
    small_protocol,
    synthetic_market,
)
from trade_rl.evaluation.dev_realdata_study import runner
from trade_rl.evaluation.dev_realdata_study.trial_sources import DrawdownOverlayPolicy
from trade_rl.evaluation.replay import SharedCashPortfolioReplayResult
from trade_rl.strategies.allocation.cost_aware import AllocationDecisionRecord
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)


@pytest.mark.parametrize("equity, expected_scale", [(85.0, 0.5), (75.0, 0.0)])
def test_t7_preserves_fail_closed_hold_when_overlay_would_trim(
    equity: float, expected_scale: float
) -> None:
    proposal = PortfolioTargetProposal(
        target_weights=(0.5, -0.4),
        hold_current_quantity_symbols=(0, 1),
    )
    stamp = int(np.datetime64("2023-01-03T00:00:00", "ns").astype(np.int64))
    record = AllocationDecisionRecord(
        index=10,
        timestamp_ns=stamp,
        outcome="fail_closed",
        reason="invalid_forecast",
        iterations=0,
        objective=None,
        traded_symbols=(),
    )

    class Inner:
        records = (record,)

        def decide_portfolio(self, _snapshot: Any) -> PortfolioTargetProposal:
            return proposal

    policy = DrawdownOverlayPolicy(
        Inner(),
        gross_max=0.95,
        decision_interval_hours=24,
        decision_anchor_hour_utc=0,
    )
    snapshot = SimpleNamespace(
        timestamp_ns=stamp,
        account=SimpleNamespace(portfolio_value=equity, peak_value=100.0),
    )
    result = policy.decide_portfolio(cast(PortfolioDecisionSnapshot, snapshot))
    assert result is proposal
    summary = policy.overlay_summary()
    assert policy.scales[-1][1] == pytest.approx(expected_scale)
    assert summary["n_fail_closed_hold_preserved"] == 1
    assert summary["n_zero_scale_fail_closed_hold_preserved"] == int(
        expected_scale == 0.0
    )
    if expected_scale == 0.0:
        # An already-exposed book is not silently liquidated when the
        # allocator cannot safely issue a new target. Risk remains observable.
        assert summary["latched_at_end"] is True
        assert summary["n_decisions_at_zero"] == 1


@pytest.mark.parametrize(
    "hours, anchor",
    [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 8)],
)
def test_t7_rejects_schedule_that_can_never_fire(hours: int, anchor: int) -> None:
    class Inner:
        records: tuple[object, ...] = ()

    with pytest.raises(ValueError, match="interval|anchor"):
        DrawdownOverlayPolicy(
            Inner(),
            gross_max=0.95,
            decision_interval_hours=hours,
            decision_anchor_hour_utc=anchor,
        )


def test_pre_cut_trade_check_rejects_flat_no_fill_intervals() -> None:
    intervals = [
        SimpleNamespace(exact_quantities_after=("0", "0"), order_events=()),
        SimpleNamespace(exact_quantities_after=("0", "0"), order_events=()),
    ]
    result = SimpleNamespace(
        ledger_evidence=SimpleNamespace(intervals=SimpleNamespace(intervals=intervals))
    )
    assert (
        runner._pre_cut_has_trades(cast(SharedCashPortfolioReplayResult, result))
        is False
    )


def test_causality_trade_check_only_looks_at_actual_interval_rows() -> None:
    intervals = [
        SimpleNamespace(exact_quantities_after=("0", "0"), order_events=()),
        SimpleNamespace(exact_quantities_after=("1/5", "0"), order_events=()),
    ]
    # Decision count is intentionally unrelated to the interval count.
    result = SimpleNamespace(
        ledger_evidence=SimpleNamespace(intervals=SimpleNamespace(intervals=intervals)),
        decisions=[SimpleNamespace() for _ in range(20)],
    )
    assert (
        runner._pre_cut_has_trades(cast(SharedCashPortfolioReplayResult, result))
        is True
    )


def test_pre_cut_roundtrip_fill_counts_as_trading_even_if_flat() -> None:
    intervals = [
        SimpleNamespace(
            exact_quantities_after=("0",),
            order_events=(SimpleNamespace(filled_quantity=0.25),),
        )
    ]
    result = SimpleNamespace(
        ledger_evidence=SimpleNamespace(intervals=SimpleNamespace(intervals=intervals))
    )
    assert (
        runner._pre_cut_has_trades(cast(SharedCashPortfolioReplayResult, result))
        is True
    )


def test_runner_records_interval_units_and_all_cashflow_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = synthetic_market()
    protocol = small_protocol()
    adapter = SimpleNamespace(history=[(0.07, 0.14)], calls=1, null_packets=0)
    monkeypatch.setattr(
        runner,
        "build_arm",
        lambda *_: runner.BuiltArm(
            policy=SimpleNamespace(),
            adapter=adapter,
            forecast_interval_hours=168,
        ),
    )
    diagnostics = SimpleNamespace(
        total_cost=1.0,
        funding_pnl=-2.0,
        borrow_cost=3.0,
        turnover_total=0.5,
        n_trades=1,
        termination_reasons=(),
    )
    monkeypatch.setattr(
        runner,
        "run_shared_portfolio_replay",
        lambda *_a, **_kw: SimpleNamespace(diagnostics=diagnostics),
    )
    monkeypatch.setattr(
        runner,
        "equity_path",
        lambda *_: (np.asarray([10_000.0, 10_100.0]), np.asarray([1, 2])),
    )
    monkeypatch.setattr(
        runner,
        "performance_summary",
        lambda *_: {"years": 1.0},
    )
    monkeypatch.setattr(
        runner,
        "reconcile_ledger",
        lambda *_: {
            "passed": True,
            "dividend_pnl": 4.0,
            "cash_interest_pnl": 5.0,
        },
    )
    monkeypatch.setattr(runner, "exposure_summary", lambda *_: {})
    result = runner.run_replay(
        dataset,
        protocol,
        "t8_blend_weekly",
        "base",
        {},
        start_index=0,
        stop_index=200,
    )
    forecast = cast(dict[str, Any], result.summary["forecast"])
    assert forecast["decision_interval_hours"] == 168
    assert forecast["mean_abs_mu_per_interval"] == pytest.approx(0.105)
    assert forecast["mean_abs_mu_per_day"] == pytest.approx(0.015)
    assert result.summary["gross_price_pnl_usdt"] == pytest.approx(97.0)
    assert result.summary["dividend_pnl_usdt"] == 4.0
    assert result.summary["cash_interest_pnl_usdt"] == 5.0
    adapter.history = []
    empty = runner.run_replay(
        dataset,
        protocol,
        "t8_blend_weekly",
        "base",
        {},
        start_index=0,
        stop_index=200,
    )
    forecast_empty = cast(dict[str, Any], empty.summary["forecast"])
    for key in (
        "mean_abs_mu_per_interval",
        "median_abs_mu_per_interval",
        "mean_abs_mu_per_day",
        "median_abs_mu_per_day",
        "share_abs_mu_above_one_way_cost",
    ):
        assert forecast_empty[key] is None
    import json

    json.dumps(empty.summary, allow_nan=False)
