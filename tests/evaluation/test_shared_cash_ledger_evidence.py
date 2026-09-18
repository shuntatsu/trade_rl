from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from tests.evaluation.test_directional_market_profile import _data, _factory
from tests.evaluation.test_shared_cash_replay import FixedIntent, _market
from tests.integrations.test_binance_market_order_profile import _profile
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.directional import evaluate_directional_arm
from trade_rl.evaluation.replay import run_shared_cash_replay
from trade_rl.simulation import BookState, ExecutionCostConfig, MarketExecutor
from trade_rl.strategies.position_intent import PositionIntent


def _assert_same_economics(left, right) -> None:
    assert left.returns.values == pytest.approx(right.returns.values)
    np.testing.assert_allclose(left.book.quantities, right.book.quantities)
    assert left.book.exact_quantities == right.book.exact_quantities
    assert left.book.cash == pytest.approx(right.book.cash)
    assert left.book.portfolio_value == pytest.approx(right.book.portfolio_value)
    assert left.book.total_cost == pytest.approx(right.book.total_cost)
    assert left.book.turnover_total == pytest.approx(right.book.turnover_total)
    assert left.book.funding_pnl == pytest.approx(right.book.funding_pnl)
    assert left.book.borrow_cost == pytest.approx(right.book.borrow_cost)
    assert left.book.max_drawdown == pytest.approx(right.book.max_drawdown)
    assert left.book.termination_reason == right.book.termination_reason
    assert left.diagnostics == right.diagnostics
    assert left.decisions == right.decisions


def test_shared_cash_ledger_capture_is_observer_only_and_canonical() -> None:
    dataset = _market(
        np.asarray([[100.0], [100.0], [105.0], [110.0], [115.0], [120.0]])
    )
    kwargs = dict(
        start_index=0,
        stop_index=5,
        gross_budget=0.25,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
    )

    baseline = run_shared_cash_replay(
        dataset, (FixedIntent(PositionIntent.LONG),), **kwargs
    )
    observed = run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.LONG),),
        capture_ledger_evidence=True,
        **kwargs,
    )

    _assert_same_economics(baseline, observed)
    assert baseline.ledger_evidence is None
    ledger = observed.ledger_evidence
    assert ledger is not None
    assert ledger.schema_version == "shared_cash_replay_ledger_v1"
    assert ledger.dataset_id == dataset.dataset_id
    assert len(ledger.execution_policy_digest) == 64
    assert len(ledger.intervals) == len(observed.returns.values)

    first = ledger.intervals[0]
    assert first.start_index == 0
    assert first.next_index == 1
    assert first.exact_quantities_before == ("0",)
    assert first.exact_quantities_after != first.exact_quantities_before
    assert first.cash_before == pytest.approx(1_000.0)
    assert first.cash_after == pytest.approx(observed.book.cash, rel=0.0, abs=1_000.0)
    assert first.order_events
    assert first.interval_net_return == pytest.approx(observed.returns.values[0])

    payload = ledger.to_mapping()
    assert payload["schema_version"] == "shared_cash_replay_ledger_v1"
    assert payload["terminal_exact_quantities"] == tuple(
        str(value) for value in observed.book.exact_quantities
    )
    assert payload["final_cash"] == pytest.approx(observed.book.cash)
    assert payload["final_portfolio_value"] == pytest.approx(
        observed.book.portfolio_value
    )
    assert payload["final_max_drawdown"] == pytest.approx(observed.book.max_drawdown)
    assert canonical_json_bytes(payload)


@pytest.mark.parametrize("reduce_only_exits", [False, True])
def test_directional_ledger_capture_preserves_profile_economics(
    reduce_only_exits: bool,
) -> None:
    data = _data()
    profile = _profile(data, reduce_only_exits=reduce_only_exits)

    baseline = evaluate_directional_arm(
        data,
        _factory,
        start_index=0,
        stop_index=5,
        market_order_profile=profile,
    )
    observed = evaluate_directional_arm(
        data,
        _factory,
        start_index=0,
        stop_index=5,
        market_order_profile=profile,
        capture_ledger_evidence=True,
    )

    ledger = observed.pop("ledger_evidence")
    assert observed == baseline
    assert ledger["schema_version"] == "shared_cash_replay_ledger_v1"
    assert ledger["execution_policy_digest"] == baseline["execution_policy_digest"]
    assert ledger["terminal_exact_quantities"] == tuple(
        baseline["terminal_exact_quantities"]
    )
    assert ledger["intervals"]
    assert any(
        event["event_type"] in {"filled", "partial_fill"}
        for interval in ledger["intervals"]
        for event in interval["order_events"]
    )
    assert all("capacity_events" in interval for interval in ledger["intervals"])
    if reduce_only_exits:
        assert ledger["terminal_exact_quantities"] == ("0",)
    else:
        assert any(
            reason == "below_minimum_notional"
            for _, reason in ledger["terminal_order_reasons"]
        )


def test_ledger_payload_is_detached_from_replay_state() -> None:
    dataset = _market(np.full((5, 1), 100.0))
    result = run_shared_cash_replay(
        dataset,
        (FixedIntent(PositionIntent.LONG),),
        start_index=0,
        stop_index=4,
        gross_budget=0.2,
        initial_capital=1_000.0,
        execution_cost=ExecutionCostConfig.zero(),
        capture_ledger_evidence=True,
    )
    ledger = result.ledger_evidence
    assert ledger is not None
    payload = ledger.to_mapping()
    mutated = deepcopy(payload)
    mutated["intervals"][0]["cash_after"] = -123.0
    assert ledger.to_mapping() == payload
    assert result.book.cash != -123.0


def test_execution_observer_is_detached_and_policy_neutral() -> None:
    dataset = _market(np.full((4, 1), 100.0))
    observations: list[object] = []
    plain = MarketExecutor(dataset, ExecutionCostConfig.zero())
    observed = MarketExecutor(
        dataset,
        ExecutionCostConfig.zero(),
        execution_observer=observations.append,
    )
    assert observed.execution_policy_digest == plain.execution_policy_digest

    book = BookState.zero(1, 1_000.0, dataset.close[0])
    result = observed.execute_interval(
        book,
        np.array([0.2]),
        start_index=0,
        bars=1,
    )
    assert result.next_index == 1
    assert len(observations) == 1
    interval = observations[0]
    assert not hasattr(interval, "book")
    assert not hasattr(interval, "order_book")
    assert getattr(interval, "order_events")
    with pytest.raises(FrozenInstanceError):
        setattr(interval, "next_index", 99)
