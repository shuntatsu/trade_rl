"""Explicit terminal reserve software contracts, using only artificial data."""

from dataclasses import replace

import numpy as np
import pytest

from tools.tier_d_nonrl.task18_recipe import build_task18_protocol
from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.decision_value_ledger_oracle import (
    verify_paired_economic_ledger,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)


def _market(n_bars: int, *, low_volume: bool = False) -> MarketDataset:
    prices = np.full((n_bars, 1), 100.0, dtype=np.float64)
    volume = np.full_like(prices, 1.0 if low_volume else 1_000_000.0)
    return MarketDataset(
        dataset_id="0" * 64,
        symbols=("SYNTHETIC-ONLY",),
        timestamps=np.datetime64("2026-01-01T00:00:00", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 6), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        volume=volume,
        funding_rate=np.zeros_like(prices),
        tradable=np.ones_like(prices, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 6), dtype=np.bool_),
        feature_names=tuple(f"synthetic_{i}" for i in range(6)),
        global_feature_names=("synthetic_global",),
        periods_per_year=8760,
        fee_rate=np.zeros_like(prices),
        spread_rate=np.zeros_like(prices),
    ).with_content_identity({"fixture": "explicit-terminal-reserve-v1"})


class _Policy:
    def __init__(self, *, cash: bool = False):
        self.calls: list[int] = []
        self.cash = cash

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        self.calls.append(snapshot.index)
        if self.cash:
            return PortfolioTargetProposal((0.0,), hold_current_quantity_symbols=(0,))
        if len(self.calls) == 1:
            return PortfolioTargetProposal((0.8,))
        # Keep exact quantities through the remaining actor decisions.
        return PortfolioTargetProposal(
            snapshot.account.weights, hold_current_quantity_symbols=(0,)
        )


def _run(
    market,
    policy,
    *,
    start=0,
    stop=None,
    latency=0,
    reserve=None,
    explicit=True,
    settle=True,
    initial_capital=1000.0,
):
    kwargs = {"terminal_reserve_bars": reserve} if explicit else {}
    return run_shared_portfolio_replay(
        market,
        policy,
        start_index=start,
        stop_index=market.n_bars - 1 if stop is None else stop,
        initial_capital=initial_capital,
        execution_cost=replace(ExecutionCostConfig.zero(), order_latency_bars=latency),
        settle_terminal_position=settle,
        capture_ledger_evidence=True,
        **kwargs,
    )


@pytest.mark.parametrize("latency", (0, 1, 2))
def test_default_terminal_reserve_and_explicit_minimum_have_identical_ledger_bytes(
    latency,
):
    market = _market(12)
    a = _run(market, _Policy(), latency=latency, explicit=False)
    b = _run(market, _Policy(), latency=latency, reserve=latency + 1)
    assert canonical_json_bytes(a.ledger_evidence.to_mapping()) == canonical_json_bytes(
        b.ledger_evidence.to_mapping()
    )


@pytest.mark.parametrize("latency", (0, 2))
def test_explicit_reserve_ends_actor_at_exact_boundary_and_executes_every_terminal_interval(
    latency,
):
    market = _market(100)
    policy = _Policy()
    result = _run(market, policy, start=10, stop=95, latency=latency, reserve=72)
    ledger = result.ledger_evidence
    assert ledger is not None
    assert policy.calls == list(range(10, 23))
    assert [d.index for d in result.decisions] == policy.calls
    assert [i.start_index for i in ledger.intervals.intervals] == list(range(10, 95))
    assert ledger.intervals.intervals[-1].next_index == 95
    assert ledger.intervals.terminal_exact_quantities == ("0",)
    assert ledger.intervals.active_order_remainders == ()
    assert verify_paired_economic_ledger(
        result,
        market,
        expected_intervals=85,
        initial_capital=1000.0,
        expected_start_index=10,
    )


@pytest.mark.parametrize(
    ("reserve", "settle", "latency"),
    (
        (True, True, 0),
        (1.0, True, 0),
        ("2", True, 0),
        (0, True, 0),
        (-1, True, 0),
        (2, True, 2),
        (10, True, 0),
        (1, False, 0),
    ),
)
def test_reserve_is_refused_before_any_policy_call_or_session_creation(
    reserve, settle, latency
):
    market = _market(11)
    policy = _Policy()
    with pytest.raises(ValueError, match="terminal"):
        _run(market, policy, reserve=reserve, settle=settle, latency=latency)
    assert not policy.calls
    with pytest.raises(ValueError, match="terminal"):
        SharedPortfolioSession(
            market,
            start_index=0,
            stop_index=10,
            execution_cost=replace(
                ExecutionCostConfig.zero(), order_latency_bars=latency
            ),
            settle_terminal_position=settle,
            terminal_reserve_bars=reserve,
        )


def test_session_terminal_boundary_refuses_actor_and_does_not_finish_early():
    market = _market(100)
    session = SharedPortfolioSession(
        market,
        start_index=10,
        stop_index=95,
        initial_capital=1000.0,
        settle_terminal_position=True,
        terminal_reserve_bars=72,
    )
    assert session.agent_stop_index == 23
    for _ in range(13):
        assert session.can_decide
        session.advance((0.0,))
    assert session.current_index == 23
    assert session.can_settle_terminal
    with pytest.raises(RuntimeError, match="not accepting actor"):
        session.advance((1.0,))
    for index in range(23, 95):
        assert session.current_index == index
        assert session.can_settle_terminal
        session.settle_next_interval()
    assert session.finished and session.current_index == 95
    assert not session.can_decide and not session.can_settle_terminal
    with pytest.raises(RuntimeError, match="no reserved terminal"):
        session.settle_next_interval()


def test_longer_reserve_allows_native_partial_fill_close_without_bypassing_liquidity():
    market = _market(18, low_volume=True)

    # Native fills limited to one unit per interval. A position of four units
    # is built by repeatedly targeting 0.8 before the terminal reserve.
    class _Build:
        def decide_portfolio(self, snapshot):
            return PortfolioTargetProposal((0.8,))

    cost = replace(ExecutionCostConfig.zero(), max_participation_rate=1.0)
    kwargs = dict(
        start_index=0,
        initial_capital=1000.0,
        execution_cost=cost,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    short = run_shared_portfolio_replay(market, _Build(), stop_index=5, **kwargs)
    long = run_shared_portfolio_replay(
        market, _Build(), stop_index=9, terminal_reserve_bars=5, **kwargs
    )
    a = short.ledger_evidence.intervals
    b = long.ledger_evidence.intervals
    assert a.terminal_exact_quantities != ("0",)
    # Same actor prefix and same opening quantity at the first terminal step.
    assert [iv.to_mapping() for iv in a.intervals[:4]] == [
        iv.to_mapping() for iv in b.intervals[:4]
    ]
    assert (
        a.intervals[4].exact_quantities_before == b.intervals[4].exact_quantities_before
    )
    assert b.terminal_exact_quantities == ("0",)
    assert len(b.intervals) == 9
    assert [d.index for d in long.decisions] == list(range(4))
    assert all(
        abs(e.filled_quantity) <= 1.0 for i in b.intervals for e in i.order_events
    )
    assert verify_paired_economic_ledger(
        long, market, expected_intervals=9, initial_capital=1000.0
    )


def test_task18_full_size_native_cash_replay_matches_frozen_clock_counts():
    protocol = build_task18_protocol()
    plan = protocol.index_plan
    initial_capital = protocol.economics.initial_capital_per_symbol_usdt
    market = _market(protocol.dataset.row_count)
    policy = _Policy(cash=True)
    result = _run(
        market,
        policy,
        start=plan.actor_start_inclusive,
        stop=plan.terminal_stop_inclusive,
        latency=2,
        reserve=plan.terminal_advance_count,
        initial_capital=initial_capital,
    )
    assert policy.calls == list(
        range(plan.actor_start_inclusive, plan.actor_stop_exclusive)
    )
    ledger = result.ledger_evidence.intervals
    assert len(ledger.intervals) == plan.native_advances_per_arm == 6599
    assert (
        len(ledger.intervals) - len(result.decisions)
        == plan.terminal_advance_count
        == 72
    )
    assert ledger.intervals[-1].next_index == plan.terminal_stop_inclusive == 9502
    assert (
        ledger.final_cash == ledger.final_portfolio_value == initial_capital == 100000.0
    )
    assert (
        ledger.final_total_cost
        == ledger.final_funding_pnl
        == ledger.final_borrow_cost
        == 0.0
    )
    assert not any(i.order_events for i in ledger.intervals)
    assert verify_paired_economic_ledger(
        result,
        market,
        expected_intervals=6599,
        initial_capital=initial_capital,
        expected_start_index=plan.actor_start_inclusive,
    )


def test_native_advance_observer_reports_actor_and_reserved_intervals_without_ledger_drift():
    market = _market(12)
    before = _run(market, _Policy(), reserve=3)
    observed = []
    after = run_shared_portfolio_replay(
        market,
        _Policy(),
        start_index=0,
        stop_index=11,
        initial_capital=1000.0,
        execution_cost=ExecutionCostConfig.zero(),
        settle_terminal_position=True,
        terminal_reserve_bars=3,
        capture_ledger_evidence=True,
        on_native_advance=observed.append,
    )
    assert len(observed) == 11
    assert [o.next_index for o in observed] == list(range(1, 12))
    assert all(o.native_advances == o.next_index for o in observed)
    assert all(o.portfolio_value == 1000.0 and o.max_drawdown == 0.0 for o in observed)
    assert canonical_json_bytes(
        after.ledger_evidence.to_mapping()
    ) == canonical_json_bytes(before.ledger_evidence.to_mapping())


def test_native_advance_observation_precedes_fallible_replay_diagnostics(monkeypatch):
    from trade_rl.evaluation import replay

    market = _market(12)
    observed = []
    original = replay._ExecutedEntryPrices.ingest

    def fail_after_second_native_advance(self, *args, **kwargs):
        if len(observed) == 2:
            raise RuntimeError("injected diagnostic failure")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(
        replay._ExecutedEntryPrices, "ingest", fail_after_second_native_advance
    )
    with pytest.raises(RuntimeError, match="injected diagnostic failure"):
        run_shared_portfolio_replay(
            market,
            _Policy(cash=True),
            start_index=2,
            stop_index=11,
            initial_capital=1000.0,
            capture_ledger_evidence=True,
            on_native_advance=observed.append,
        )
    assert [(o.native_advances, o.next_index) for o in observed] == [(1, 3), (2, 4)]
    assert observed[-1].portfolio_value == 1000.0


def test_native_advance_observer_failure_aborts_dispatch_and_invalid_observer_fails_before_policy():
    market = _market(12)
    policy = _Policy(cash=True)

    def fail(observer):
        assert observer.native_advances == 1
        raise OSError("checkpoint write failure")

    with pytest.raises(OSError, match="checkpoint write failure"):
        run_shared_portfolio_replay(
            market, policy, start_index=0, stop_index=11, on_native_advance=fail
        )
    assert policy.calls == [0]
    policy = _Policy(cash=True)
    with pytest.raises(TypeError, match="on_native_advance"):
        run_shared_portfolio_replay(
            market, policy, start_index=0, stop_index=11, on_native_advance=1
        )
    assert policy.calls == []
