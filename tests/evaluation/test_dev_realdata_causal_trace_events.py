"""The pre-cut causality oracle must observe native events, not only account totals."""

from __future__ import annotations

from dataclasses import replace

import pytest

from tests.evaluation.test_dev_realdata_study import small_protocol, synthetic_market
from trade_rl.evaluation.dev_realdata_study import runner
from trade_rl.evaluation.dev_realdata_study.protocol import (
    ARM_BUY_AND_HOLD,
    SCENARIO_BASE,
)
from trade_rl.evaluation.replay import (
    SharedCashLedgerIntervalEvidence,
    SharedCashPortfolioReplayResult,
)


@pytest.fixture(scope="module")
def captured_result() -> SharedCashPortfolioReplayResult:
    """Small real native replay: avoid using historical research/results artifacts."""

    market = synthetic_market()
    protocol = small_protocol()
    start = runner._row_of(market, protocol.evaluation_start)
    policy = runner.build_arm(ARM_BUY_AND_HOLD, market, protocol, {}).policy
    return runner.run_shared_portfolio_replay(
        market,
        policy,
        start_index=start,
        stop_index=start + 48,
        initial_capital=protocol.initial_capital,
        execution_cost=runner.scenario_cost(protocol, SCENARIO_BASE),
        capture_ledger_evidence=True,
    )


def _with_interval(
    result: SharedCashPortfolioReplayResult,
    index: int,
    replacement: SharedCashLedgerIntervalEvidence,
) -> SharedCashPortfolioReplayResult:
    evidence = result.ledger_evidence
    assert evidence is not None
    original = evidence.intervals.intervals
    intervals = (*original[:index], replacement, *original[index + 1 :])
    return replace(
        result,
        ledger_evidence=replace(
            evidence, intervals=replace(evidence.intervals, intervals=intervals)
        ),
    )


def test_causal_trace_detects_changed_order_event_with_same_cash_and_position(
    captured_result: SharedCashPortfolioReplayResult,
) -> None:
    evidence = captured_result.ledger_evidence
    assert evidence is not None
    index, interval = next(
        (index, value)
        for index, value in enumerate(evidence.intervals.intervals)
        if value.order_events
    )
    event = interval.order_events[0]
    changed_event = replace(event, sequence=event.sequence + 10_000)
    changed = _with_interval(
        captured_result,
        index,
        replace(
            interval,
            order_events=(changed_event, *interval.order_events[1:]),
        ),
    )
    assert changed.book == captured_result.book
    assert changed.decisions == captured_result.decisions
    # Only an observer event has changed, not the previously compared totals.
    assert runner._trace(changed) != runner._trace(captured_result)


@pytest.mark.parametrize("field", ["interval_cost", "interval_funding"])
def test_causal_trace_detects_changed_interval_flow_with_same_equity(
    captured_result: SharedCashPortfolioReplayResult, field: str
) -> None:
    evidence = captured_result.ledger_evidence
    assert evidence is not None
    interval = evidence.intervals.intervals[0]
    changed = _with_interval(
        captured_result, 0, replace(interval, **{field: getattr(interval, field) + 0.01})
    )
    assert changed.book == captured_result.book
    assert runner._trace(changed) != runner._trace(captured_result)


def test_causal_trace_detects_decision_risk_reason_divergence(
    captured_result: SharedCashPortfolioReplayResult,
) -> None:
    assert captured_result.decisions
    old = captured_result.decisions[0]
    altered_decision = replace(
        old, risk_reasons=(*old.risk_reasons, "future_sensitive_reason")
    )
    changed = replace(
        captured_result, decisions=(altered_decision, *captured_result.decisions[1:])
    )
    assert changed.book == captured_result.book
    assert runner._trace(changed) != runner._trace(captured_result)


def test_causal_trace_keeps_original_intervals_and_decision_partition(
    captured_result: SharedCashPortfolioReplayResult,
) -> None:
    evidence = captured_result.ledger_evidence
    assert evidence is not None
    trace = runner._trace(captured_result)
    assert trace == runner._trace(captured_result)
    intervals, decisions = runner._split_trace(trace, captured_result)
    assert len(intervals) == len(evidence.intervals.intervals)
    assert len(decisions) == len(captured_result.decisions)
    assert intervals[0][1] == evidence.intervals.intervals[0].exact_quantities_after
