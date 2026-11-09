"""Future-perturbation oracles must compare native events, not just account totals."""

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
    """Capture a short synthetic native replay without historical research artifacts."""

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
    intervals = list(evidence.intervals.intervals)
    intervals[index] = replacement
    changed_ledger = replace(evidence.intervals, intervals=tuple(intervals))
    changed_evidence = replace(evidence, intervals=changed_ledger)
    return replace(result, ledger_evidence=changed_evidence)


def test_causal_trace_detects_changed_order_event_with_same_cash_and_position(
    captured_result: SharedCashPortfolioReplayResult,
) -> None:
    evidence = captured_result.ledger_evidence
    assert evidence is not None
    intervals = evidence.intervals.intervals
    index = next(i for i, interval in enumerate(intervals) if interval.order_events)
    original = intervals[index]
    events = list(original.order_events)
    events[0] = replace(events[0], sequence=events[0].sequence + 10_000)
    changed = _with_interval(
        captured_result,
        index,
        replace(original, order_events=tuple(events)),
    )
    assert changed.book is captured_result.book
    assert changed.decisions == captured_result.decisions
    assert runner._trace(changed) != runner._trace(captured_result)


@pytest.mark.parametrize("field", ["interval_cost", "interval_funding"])
def test_causal_trace_detects_changed_interval_flow_with_same_equity(
    captured_result: SharedCashPortfolioReplayResult, field: str
) -> None:
    evidence = captured_result.ledger_evidence
    assert evidence is not None
    original = evidence.intervals.intervals[0]
    changed_value = getattr(original, field) + 0.01
    changed_interval = replace(original, **{field: changed_value})
    changed = _with_interval(captured_result, 0, changed_interval)
    assert changed.book is captured_result.book
    assert runner._trace(changed) != runner._trace(captured_result)


def test_causal_trace_detects_decision_risk_reason_divergence(
    captured_result: SharedCashPortfolioReplayResult,
) -> None:
    assert captured_result.decisions
    original = captured_result.decisions[0]
    reasons = (*original.risk_reasons, "future_sensitive_reason")
    altered_decision = replace(original, risk_reasons=reasons)
    decisions = (altered_decision, *captured_result.decisions[1:])
    changed = replace(captured_result, decisions=decisions)
    assert changed.book is captured_result.book
    assert runner._trace(changed) != runner._trace(captured_result)


def test_causal_trace_preserves_original_interval_and_decision_partition(
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


def test_causality_probe_rejects_event_only_divergence(
    captured_result: SharedCashPortfolioReplayResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evidence = captured_result.ledger_evidence
    assert evidence is not None
    intervals = evidence.intervals.intervals
    index = next(i for i, interval in enumerate(intervals) if interval.order_events)
    original = intervals[index]
    events = list(original.order_events)
    events[0] = replace(events[0], sequence=events[0].sequence + 10_000)
    altered = _with_interval(
        captured_result,
        index,
        replace(original, order_events=tuple(events)),
    )
    responses = iter((captured_result, altered))
    calls = 0

    def fake_replay(*args: object, **kwargs: object) -> SharedCashPortfolioReplayResult:
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr(runner, "run_shared_portfolio_replay", fake_replay)
    market = synthetic_market()
    protocol = small_protocol()
    start = runner._row_of(market, protocol.evaluation_start)
    report = runner.causality_probe(
        market,
        protocol,
        {},
        start_index=start,
        arm=ARM_BUY_AND_HOLD,
        expect_trades=False,
        perturbed_pair=(market, {}),
    )
    assert calls == 2
    assert report["identical_pre_cut_trace"] is False
    assert report["passed"] is False
