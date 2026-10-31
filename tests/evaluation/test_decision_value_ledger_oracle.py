"""Source-backed money-reconstruction falsifiers for TD8 teacher and evaluation."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pytest

from tests.evaluation.test_paired_portfolio_action import (
    _CAPITAL,
    _COST,
    _RISK,
    _dataset,
    _EnterThenHold,
    _paired,
)
from trade_rl.evaluation import decision_value_reachable_labels as labels
from trade_rl.evaluation.decision_value_ledger_oracle import (
    verify_paired_economic_ledger,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_FLAT,
    ACTION_HOLD,
)


def test_td8_source_oracle_detects_forged_interior_cost_even_if_book_agrees() -> None:
    dataset = _dataset()
    pair = _paired(market=dataset)
    arm = pair.baseline_result
    assert verify_paired_economic_ledger(
        arm,
        dataset,
        expected_intervals=7,
        initial_capital=_CAPITAL,
        expected_start_index=0,
    )
    ledger = arm.ledger_evidence
    assert ledger is not None
    existing = ledger.intervals.intervals[2]
    corruption = replace(existing, interval_cost=existing.interval_cost + 13.0)
    fake = replace(
        arm,
        ledger_evidence=replace(
            ledger,
            intervals=replace(
                ledger.intervals,
                intervals=(
                    *ledger.intervals.intervals[:2],
                    corruption,
                    *ledger.intervals.intervals[3:],
                ),
            ),
        ),
    )
    # Terminal BookState and ledger summaries still agree; the old check passed.
    assert fake.book.portfolio_value == arm.book.portfolio_value
    assert (
        fake.ledger_evidence.intervals.final_portfolio_value
        == arm.ledger_evidence.intervals.final_portfolio_value
    )
    assert not verify_paired_economic_ledger(
        fake,
        dataset,
        expected_intervals=7,
        initial_capital=_CAPITAL,
        expected_start_index=0,
    )

    # This is the production evaluation path, not merely the standalone oracle.
    from trade_rl.evaluation.decision_value_dev_study import (
        _verify_ledger_reconciliation,
    )

    assert not _verify_ledger_reconciliation(
        fake,
        dataset=dataset,
        expected_intervals=7,
        expected_start_index=0,
        initial_capital=_CAPITAL,
    )


def test_td8_teacher_rejects_corrupted_economic_ledger_before_label_admission() -> None:
    dataset = _dataset()
    pair = _paired(market=dataset)
    ledger = pair.candidate_result.ledger_evidence
    assert ledger is not None
    iv = ledger.intervals.intervals[2]
    changed_arm = replace(
        pair.candidate_result,
        ledger_evidence=replace(
            ledger,
            intervals=replace(
                ledger.intervals,
                intervals=(
                    *ledger.intervals.intervals[:2],
                    replace(iv, interval_cost=iv.interval_cost + 12.0),
                    *ledger.intervals.intervals[3:],
                ),
            ),
        ),
    )
    fake_pair = replace(pair, candidate_result=changed_arm)
    with patch.object(labels, "run_paired_portfolio_action", return_value=fake_pair):
        with pytest.raises(ValueError, match="independent|ledger|accounting"):
            labels.collect_reachable_state_labels(
                dataset,
                reference_policy_provider=lambda _t, _start: _EnterThenHold,
                reference_policy_digest="d" * 64,
                branch_indices=(1,),
                train_start_index=0,
                prefix_bars=1,
                horizon_bars=6,
                candidate_actions=(
                    (ACTION_HOLD, ACTION_HOLD),
                    (ACTION_FLAT, ACTION_HOLD),
                ),
                target_abs_weights=(0.4, 0.4),
                initial_capital=_CAPITAL,
                execution_cost=_COST,
                risk_config=_RISK.config,
                fit_cutoff=np.datetime64("2026-02-01T00:00:00", "ns"),
                source_scope="synthetic_only",
            )


def test_td8_evaluation_oracle_rejects_price_and_market_identity_tamper() -> None:
    market = _dataset()
    result = _paired(market=market).baseline_result
    kwargs = {
        "expected_intervals": 7,
        "initial_capital": _CAPITAL,
        "expected_start_index": 0,
    }
    assert verify_paired_economic_ledger(result, market, **kwargs)
    market_changed = replace(market, dataset_id="f" * 64)
    assert not verify_paired_economic_ledger(result, market_changed, **kwargs)
    ledger = result.ledger_evidence
    assert ledger is not None
    interval = ledger.intervals.intervals[2]
    tampered = replace(
        result,
        ledger_evidence=replace(
            ledger,
            intervals=replace(
                ledger.intervals,
                intervals=(
                    *ledger.intervals.intervals[:2],
                    replace(
                        interval,
                        portfolio_value_after=interval.portfolio_value_after + 5.0,
                    ),
                    *ledger.intervals.intervals[3:],
                ),
            ),
        ),
    )
    assert not verify_paired_economic_ledger(tampered, market, **kwargs)


def test_td8_oracle_fails_closed_on_unmodeled_corporate_action_split() -> None:
    """The software oracle cannot silently bless a split it did not reconstruct."""
    market = _dataset()
    result = _paired(market=market).baseline_result
    split = np.ones_like(market.close)
    split[2, 1] = 2.0  # BBB is held flat: older oracle silently ignored the split.
    with_split = replace(market, split_factor=split)
    assert not verify_paired_economic_ledger(
        result,
        with_split,
        expected_intervals=7,
        initial_capital=_CAPITAL,
        expected_start_index=0,
    )


def test_td8_oracle_detects_non_fill_event_with_spurious_notional() -> None:
    market = _dataset()
    result = _paired(market=market).baseline_result
    ledger = result.ledger_evidence
    assert ledger is not None
    modified = None
    for i, iv in enumerate(ledger.intervals.intervals):
        for j, event in enumerate(iv.order_events):
            if event.filled_quantity == 0.0:
                events = list(iv.order_events)
                events[j] = replace(event, filled_notional=event.filled_notional + 10)
                ivs = list(ledger.intervals.intervals)
                ivs[i] = replace(iv, order_events=tuple(events))
                modified = replace(
                    result,
                    ledger_evidence=replace(
                        ledger,
                        intervals=replace(ledger.intervals, intervals=tuple(ivs)),
                    ),
                )
                break
        if modified is not None:
            break
    assert modified is not None
    assert not verify_paired_economic_ledger(
        modified,
        market,
        expected_intervals=7,
        initial_capital=_CAPITAL,
        expected_start_index=0,
    )


def test_td8_oracle_rejects_rewritten_interval_return_with_same_book() -> None:
    market = _dataset()
    result = _paired(market=market).baseline_result
    ledger = result.ledger_evidence
    assert ledger is not None
    iv = ledger.intervals.intervals[2]
    tampered = replace(
        result,
        ledger_evidence=replace(
            ledger,
            intervals=replace(
                ledger.intervals,
                intervals=(
                    *ledger.intervals.intervals[:2],
                    replace(iv, interval_net_return=iv.interval_net_return + 0.1),
                    *ledger.intervals.intervals[3:],
                ),
            ),
        ),
    )
    assert tampered.book.portfolio_value == result.book.portfolio_value
    assert not verify_paired_economic_ledger(
        tampered,
        market,
        expected_intervals=7,
        initial_capital=_CAPITAL,
        expected_start_index=0,
    )


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("dataset_id", "f" * 64),
        ("execution_policy_digest", "e" * 64),
        ("timestamp_ns", -1),
        ("processing_index", -1),
        ("symbol_index", -1),
    ],
)
def test_td8_oracle_rejects_forged_non_fill_event_identity(
    field: str, wrong: object
) -> None:
    """Source/clock/identity validation is required even with zero fills."""
    market = _dataset()
    result = _paired(market=market).baseline_result
    ledger = result.ledger_evidence
    assert ledger is not None
    first = ledger.intervals.intervals[0]
    event = next(e for e in first.order_events if e.filled_quantity == 0.0)
    # Keep each field syntactically valid but inconsistent with the real source.
    if field == "timestamp_ns":
        wrong = event.timestamp_ns + 7777
    if field == "processing_index":
        wrong = event.processing_index + 2
    if field == "symbol_index":
        wrong = len(market.symbols) + 1
    forged = replace(event, **{field: wrong})
    interval = replace(
        first,
        order_events=tuple(forged if e is event else e for e in first.order_events),
    )
    mutated = replace(
        result,
        ledger_evidence=replace(
            ledger,
            intervals=replace(
                ledger.intervals,
                intervals=(interval, *ledger.intervals.intervals[1:]),
            ),
        ),
    )
    assert verify_paired_economic_ledger(
        result,
        market,
        expected_intervals=7,
        initial_capital=_CAPITAL,
        expected_start_index=0,
    )
    assert not verify_paired_economic_ledger(
        mutated,
        market,
        expected_intervals=7,
        initial_capital=_CAPITAL,
        expected_start_index=0,
    )


def test_td8_oracle_rejects_coherent_out_of_market_price_forgery() -> None:
    """Consistent book/observer totals cannot bless impossible source fill prices."""
    market = _dataset()
    original = _paired(market=market).baseline_result
    wrapped = original.ledger_evidence
    assert wrapped is not None
    intervals = list(wrapped.intervals.intervals)
    last = intervals[-1]
    index = next(i for i, e in enumerate(last.order_events) if e.filled_quantity)
    event = last.order_events[index]
    assert market.high[last.next_index, event.symbol_index] == 110.0
    price = 1000.0
    multiplier = float(
        market.resolved_array("contract_multipliers")[event.symbol_index]
    )
    cash_delta = -event.filled_quantity * (price - event.execution_price) * multiplier
    revised_events = list(last.order_events)
    revised_events[index] = replace(
        event,
        execution_price=price,
        filled_notional=abs(event.filled_quantity) * price * multiplier,
    )
    revised_equity = last.portfolio_value_after + cash_delta
    intervals[-1] = replace(
        last,
        order_events=tuple(revised_events),
        cash_after=last.cash_after + cash_delta,
        portfolio_value_after=revised_equity,
        interval_net_return=max(
            max(revised_equity, 0.0) / last.portfolio_value_before - 1.0,
            -1.0 + 1e-12,
        ),
    )
    revised_ledger = replace(
        wrapped.intervals,
        intervals=tuple(intervals),
        final_cash=wrapped.intervals.final_cash + cash_delta,
        final_portfolio_value=wrapped.intervals.final_portfolio_value + cash_delta,
    )
    revised_book = replace(
        original.book,
        cash=original.book.cash + cash_delta,
        peak_value=max(original.book.peak_value, original.book.cash + cash_delta),
    )
    forged = replace(
        original,
        book=revised_book,
        ledger_evidence=replace(wrapped, intervals=revised_ledger),
    )
    kwargs = dict(
        expected_intervals=7,
        expected_start_index=0,
        initial_capital=_CAPITAL,
        execution_cost=_COST,
    )
    assert verify_paired_economic_ledger(original, market, **kwargs)
    assert not verify_paired_economic_ledger(forged, market, **kwargs)


def test_td8_oracle_recomputes_turnover_not_only_equal_book_and_final_ledger() -> None:
    market = _dataset()
    original = _paired(market=market).baseline_result
    evidence = original.ledger_evidence
    assert evidence is not None
    forged = replace(
        original,
        book=replace(original.book, turnover_total=0.0),
        ledger_evidence=replace(
            evidence,
            intervals=replace(evidence.intervals, final_turnover_total=0.0),
        ),
    )
    assert original.book.turnover_total > 0
    kwargs = dict(
        expected_intervals=7, initial_capital=_CAPITAL, expected_start_index=0
    )
    assert verify_paired_economic_ledger(original, market, **kwargs)
    assert not verify_paired_economic_ledger(forged, market, **kwargs)


def test_td8_oracle_detects_forged_low_drawdown_despite_native_equity_decline() -> None:
    market = _dataset(rise=-30.0)
    original = _paired(market=market).baseline_result
    wrapped = original.ledger_evidence
    assert wrapped is not None
    assert original.book.max_drawdown > 0.02
    forged = replace(
        original,
        book=replace(original.book, max_drawdown=0.0),
        ledger_evidence=replace(
            wrapped,
            intervals=replace(wrapped.intervals, final_max_drawdown=0.0),
        ),
    )
    kwargs = dict(
        expected_intervals=7, initial_capital=_CAPITAL, expected_start_index=0
    )
    assert verify_paired_economic_ledger(original, market, **kwargs)
    assert not verify_paired_economic_ledger(forged, market, **kwargs)


def test_td8_oracle_rejects_tampered_zero_fill_event_sequence() -> None:
    market = _dataset()
    result = _paired(market=market).baseline_result
    wrapped = result.ledger_evidence
    assert wrapped is not None
    first = wrapped.intervals.intervals[0]
    events = list(first.order_events)
    events[0] = replace(events[0], sequence=9)
    forged = replace(
        result,
        ledger_evidence=replace(
            wrapped,
            intervals=replace(
                wrapped.intervals,
                intervals=(
                    replace(first, order_events=tuple(events)),
                    *wrapped.intervals.intervals[1:],
                ),
            ),
        ),
    )
    kwargs = dict(
        expected_intervals=7, initial_capital=_CAPITAL, expected_start_index=0
    )
    assert verify_paired_economic_ledger(result, market, **kwargs)
    assert not verify_paired_economic_ledger(forged, market, **kwargs)
