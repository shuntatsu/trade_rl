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
