"""Cash account, stride/deadband, decision rule and runner tests of ``agy_anchor_realistic_v1``.

Synthetic data only: no market data, network or economic result is read.
"""

from __future__ import annotations

import itertools
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.evaluation.test_agy_anchor_oos import (
    BAR,
    assemble,
    leg_maps,
    small_window,
    truncate_leg,
)
from tests.evaluation.test_lowturn_screen import _constrain, _snapshot
from tests.evaluation.test_trend_oos import _slow_bootstrap
from trade_rl.evaluation.agy_anchor_oos import cash, data, decision, protocol, runner
from trade_rl.evaluation.agy_anchor_oos.decision import OosCell
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.strategies.agy_anchor import targets as anchor_targets
from trade_rl.strategies.agy_anchor.policy import AgyAnchorPolicy
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPolicy,
    PortfolioTargetProposal,
)

N_LEGS = 22
SPOT = np.arange(11) * 2
PERP = SPOT + 1
RATE = protocol.INITIAL_MARGIN_RATE
RUN_DATE = date(2026, 11, 20)


@pytest.fixture(autouse=True)
def _no_real_halts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(protocol, "DECLARED_HALTS", ())
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", None)


# ================================================================ the cash account


class _Fixed(PortfolioPolicy):
    """A policy proposing fixed weights (and optional exact-quantity holds)."""

    def __init__(self, weights: np.ndarray, holds: tuple[int, ...] = ()) -> None:
        self.weights = np.asarray(weights, dtype=np.float64)
        self.holds = holds

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        return PortfolioTargetProposal(
            target_weights=tuple(float(w) for w in self.weights),
            hold_current_quantity_symbols=self.holds,
        )


def _w(**legs: float) -> np.ndarray:
    """``s0=0.4`` is spot leg 0, ``p3=-0.5`` is the perp of pair 3."""
    out = np.zeros(N_LEGS)
    for name, value in legs.items():
        out[2 * int(name[1:]) + (name[0] == "p")] = value
    return out


def _wrapped(inner: PortfolioPolicy, start: int = 0) -> cash.AnchorCashAccountPolicy:
    return cash.AnchorCashAccountPolicy(inner, start_index=start)


def test_the_cash_measure_counts_spot_longs_plus_the_margin_rate_times_perp_gross() -> (
    None
):
    assert cash.cash_measure(_w(s0=0.4, s1=0.2)) == pytest.approx(0.6)
    assert cash.cash_measure(_w(p0=-0.5, p1=0.3)) == pytest.approx(RATE * 0.8)
    assert cash.cash_measure(_w(s0=0.5, p0=-0.5, s1=0.1, p4=0.25)) == pytest.approx(
        0.6 + RATE * 0.75
    )
    assert RATE == 0.4 and protocol.CASH_LIMIT == 1.0


def test_a_book_within_the_limit_is_untouched_and_counted_as_not_bound() -> None:
    inner = _Fixed(_w(s0=0.4, p0=-0.4, p1=0.5))  # 0.4 + 0.4 * 0.9 = 0.76
    policy = _wrapped(inner)
    proposal = policy.decide_portfolio(_snapshot(0))
    assert tuple(proposal.target_weights) == tuple(inner.weights)
    assert policy.stats.bound_decisions == 0
    stats = policy.stats.to_dict()
    assert stats["cap_bound_fraction"] == 0.0
    assert stats["mean_pre_cap_measure"] == pytest.approx(0.76)
    assert stats["initial_margin_rate"] == 0.4


def test_the_cap_scales_a_book_with_perps_to_exactly_the_limit_proportionally() -> None:
    inner = _Fixed(_w(s0=0.55, s1=0.55, p0=-0.55, p1=0.55, p2=-0.55, p3=0.55))
    raw = cash.cash_measure(inner.weights)
    assert raw == pytest.approx(1.1 + RATE * 2.2)  # 1.98
    policy = _wrapped(inner)
    proposal = policy.decide_portfolio(_snapshot(0))
    capped = np.asarray(proposal.target_weights)
    assert cash.cash_measure(capped) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(capped, inner.weights / raw, atol=1e-15)
    assert proposal.hold_current_quantity_symbols == ()
    assert policy.stats.bound_decisions == 1
    assert policy.stats.to_dict()["max_pre_cap_measure"] == pytest.approx(raw)
    # after risk (no drawdown) the executed book still satisfies the cap
    executed = _constrain(proposal, np.zeros(N_LEGS), 0.0)
    assert cash.cash_measure(executed) <= 1.0 + 1e-12


def test_perp_shorts_count_towards_the_margin_cap_like_perp_longs() -> None:
    long_book = _w(s0=0.5, p1=0.55, p2=0.55, p3=0.55)
    short_book = _w(s0=0.5, p1=-0.55, p2=-0.55, p3=-0.55)
    assert cash.cash_measure(long_book) == cash.cash_measure(short_book)
    # 0.5 + 0.4 * 1.65 = 1.16 > 1: the margin bound binds both books
    assert cash.cash_measure(long_book) == pytest.approx(0.5 + RATE * 1.65)
    # the short book has a negative net: only the margin bound acts and fills the limit
    capped = _wrapped(_Fixed(short_book)).decide_portfolio(_snapshot(0)).target_weights
    assert cash.cash_measure(np.asarray(capped)) == pytest.approx(1.0, abs=1e-12)
    # the long book is net long 2.15: the net bound is the tighter one (Amendment 3)
    capped_long = np.asarray(
        _wrapped(_Fixed(long_book)).decide_portfolio(_snapshot(0)).target_weights
    )
    assert cash.net_exposure(capped_long) == pytest.approx(0.97, abs=1e-12)
    assert cash.cash_measure(capped_long) < 1.0


def test_the_net_exposure_bound_alone_scales_a_net_long_book_to_the_limit() -> None:
    book = _w(s0=0.55, p0=0.55, p1=0.55)  # margin use 0.99 <= 1, net 1.65 > 1
    assert cash.cash_measure(book) <= 1.0 and cash.net_exposure(book) == pytest.approx(
        1.65
    )
    policy = _wrapped(_Fixed(book))
    capped = np.asarray(policy.decide_portfolio(_snapshot(0)).target_weights)
    assert protocol.NET_EXPOSURE_LIMIT == 0.97
    assert cash.net_exposure(capped) == pytest.approx(0.97, abs=1e-12)
    np.testing.assert_allclose(capped, book * 0.97 / 1.65, atol=1e-15)
    stats = policy.stats
    assert (
        stats.bound_by_net_exposure,
        stats.bound_by_margin,
        stats.bound_by_both,
    ) == (
        1,
        0,
        0,
    )
    assert stats.to_dict()["max_pre_cap_net"] == pytest.approx(1.65)
    assert stats.to_dict()["net_exposure_limit"] == 0.97


def test_a_carry_like_book_is_not_touched_by_the_net_bound() -> None:
    carry = _w(s0=0.4, p0=-0.4, s1=0.3, p1=-0.3)  # margin use 0.7 + 0.4 * 0.7 = 0.98
    assert cash.cash_measure(carry) < 1.0
    assert cash.net_exposure(carry) == pytest.approx(0.0)
    policy = _wrapped(_Fixed(carry))
    assert tuple(policy.decide_portfolio(_snapshot(0)).target_weights) == tuple(carry)
    assert policy.stats.bound_decisions == 0
    # three carry pairs are scaled by the margin bound only; the net stays zero
    three = _w(s0=0.55, p0=-0.55, s1=0.55, p1=-0.55, s2=0.55, p2=-0.55)
    policy = _wrapped(_Fixed(three))
    capped = np.asarray(policy.decide_portfolio(_snapshot(0)).target_weights)
    assert cash.net_exposure(capped) == pytest.approx(0.0, abs=1e-15)
    assert (policy.stats.bound_by_margin, policy.stats.bound_by_net_exposure) == (1, 0)


def test_the_binding_counts_distinguish_margin_net_and_both() -> None:
    inside = _w(s0=0.45, s1=0.45)  # net 0.9 and margin 0.9: neither binds
    policy = _wrapped(_Fixed(inside))
    policy.decide_portfolio(_snapshot(0))
    assert policy.stats.bound_decisions == 0
    # a spot-only book at net 1.0 is inside margin (1.0) but above the 0.97 net limit
    edge = _w(s0=0.5, s1=0.5)
    policy = _wrapped(_Fixed(edge))
    policy.decide_portfolio(_snapshot(0))
    assert (policy.stats.bound_by_net_exposure, policy.stats.bound_by_margin) == (1, 0)
    # a short perp lifts margin use and lowers net until the two factors are equal:
    # 1 / (1.5 + 0.4 p) = 0.97 / (1.5 - p)
    p = 0.045 / 1.388
    tie = _w(s0=0.55, s1=0.55, s2=0.4, p3=-p)
    policy = _wrapped(_Fixed(tie))
    policy.decide_portfolio(_snapshot(0))
    assert (
        policy.stats.bound_by_both,
        policy.stats.bound_by_margin,
        policy.stats.bound_by_net_exposure,
    ) == (1, 0, 0)
    # many short perps and some spot: margin binds, the net is negative
    margin = _w(s0=0.55, s1=0.55, p2=-0.55, p3=-0.55, p4=-0.55, p5=-0.55)
    policy = _wrapped(_Fixed(margin))
    policy.decide_portfolio(_snapshot(0))
    assert policy.stats.bound_by_margin == 1
    assert policy.stats.to_dict()["cap_bound_decisions"] == 1


def test_a_hedged_carry_book_is_charged_the_margin_not_the_notional() -> None:
    """One carry pair (spot 0.55, perp -0.55) uses 0.55 + 0.4 * 0.55 = 0.77 of the cash
    account and is untouched; three such pairs would use 2.31 and are scaled to 1.0."""
    one = _w(s0=0.55, p0=-0.55)
    assert cash.cash_measure(one) == pytest.approx(0.55 + RATE * 0.55)
    assert _wrapped(_Fixed(one)).decide_portfolio(_snapshot(0)).target_weights == tuple(
        one
    )
    three = _w(s0=0.55, p0=-0.55, s1=0.55, p1=-0.55, s2=0.55, p2=-0.55)
    capped = np.asarray(
        _wrapped(_Fixed(three)).decide_portfolio(_snapshot(0)).target_weights
    )
    assert cash.cash_measure(capped) == pytest.approx(1.0)
    np.testing.assert_allclose(capped, three * (1.0 / cash.cash_measure(three)))


def test_when_the_cap_binds_every_hold_is_dropped_so_the_deadband_cannot_creep() -> (
    None
):
    held_legs = (0, 2)
    inner = _Fixed(_w(s0=0.3, s1=0.3, s2=0.55, p2=-0.55, p3=0.55), holds=held_legs)
    current = _w(s0=0.3, s1=0.3)  # the held legs keep their current weight
    snapshot = _snapshot(0, weights=current)
    policy = _wrapped(inner)
    proposal = policy.decide_portfolio(snapshot)
    assert proposal.hold_current_quantity_symbols == ()
    executed = np.asarray(proposal.target_weights)
    assert cash.cash_measure(executed) == pytest.approx(1.0)
    # without the wrapper the mixture of held and re-targeted legs is above the limit
    raw = np.where(np.isin(np.arange(N_LEGS), held_legs), current, inner.weights)
    assert cash.cash_measure(raw) > 1.0


def test_a_pure_hold_and_a_no_trade_pre_image_are_never_capped() -> None:
    current = _w(
        s0=0.55, s1=0.55, p0=-0.55, p1=0.55, p2=0.55
    )  # measure above the limit
    assert cash.cash_measure(current) > 1.0
    hold = _Fixed(current, holds=tuple(range(N_LEGS)))
    policy = _wrapped(hold)
    proposal = policy.decide_portfolio(_snapshot(0, weights=current))
    assert proposal.hold_current_quantity_symbols == tuple(range(N_LEGS))
    assert policy.stats.bound_decisions == 0
    # the pre-image of an unchanged scale executes to the current weights: no trade
    pre = _Fixed(current / 0.5)
    snap = _snapshot(0, weights=current, value=85_000.0, peak=100_000.0, max_dd=0.15)
    assert _wrapped(pre).decide_portfolio(snap).target_weights == tuple(current / 0.5)


def test_risk_scale_is_applied_exactly_once_to_a_capped_proposal() -> None:
    inner = _Fixed(_w(s0=0.5, s1=0.5, s2=0.5, p0=-0.2))
    snapshot = _snapshot(0, value=88_000.0, peak=100_000.0, max_dd=0.12)  # scale 0.8
    proposal = _wrapped(inner).decide_portfolio(snapshot)
    executed = _constrain(proposal, np.zeros(N_LEGS), 0.12)
    assert cash.cash_measure(executed) == pytest.approx(1.0, abs=1e-9)
    # a stopped book (scale 0) is never rescaled
    stopped = _snapshot(0, value=75_000.0, peak=100_000.0, max_dd=0.25)
    out = _wrapped(inner).decide_portfolio(stopped)
    assert tuple(out.target_weights) == tuple(inner.weights)


def test_the_statistics_count_only_rebalance_bars() -> None:
    inner = _Fixed(_w(s0=0.55, s1=0.55, p0=-0.55, p1=0.55, p2=-0.55, p3=0.55))
    policy = cash.AnchorCashAccountPolicy(inner, start_index=100)
    for index in (100, 101, 117, 118, 119, 136):
        policy.decide_portfolio(_snapshot(index))
    assert policy.stats.rebalance_decisions == 3  # 100, 118, 136
    assert policy.stats.bound_decisions == 3
    assert policy.stats.to_dict()["cap_bound_fraction"] == 1.0


def test_the_wrapper_validates_its_parameters() -> None:
    inner = _Fixed(np.zeros(N_LEGS))
    for kwargs in (
        {"limit": 0.0},
        {"limit": 1.5},
        {"rate": 0.0},
        {"rate": 1.2},
        {"net_limit": 0.0},
        {"net_limit": 1.5},
        {"start_index": -1},
        {"rebalance_every_bars": 0},
    ):
        args = {"start_index": 0, **kwargs}
        with pytest.raises(ValueError):
            cash.AnchorCashAccountPolicy(inner, **args)


# ------------------------------------------------ the negative-cash INVALID cell (replay)


def _replay_cell(
    dataset: Any,
    weights: np.ndarray,
    *,
    start: int,
    steps: int = 40,
    capped: bool = True,
) -> OosCell:
    inner = _Fixed(weights)
    policy = _wrapped(inner, start) if capped else inner
    result = run_shared_portfolio_replay(
        dataset,
        policy,
        start_index=start,
        stop_index=start + steps,
        initial_capital=protocol.INITIAL_CAPITAL,
        execution_cost=protocol.scenario_cost("BASE"),
        risk=PreTradeRisk(protocol.risk_config()),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    return runner._build_cell(
        arm=protocol.CANDIDATE,
        scenario="BASE",
        result=result,
        dataset=dataset,
        causality_passed=True,
        probe_key="x",
        cap_statistics=_stats(policy),
    )


def _stats(policy: PortfolioPolicy) -> dict[str, object]:
    return runner._stats_of(policy)


@pytest.fixture(scope="module")
def _stage() -> Any:
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(protocol, "DECLARED_HALTS", ())
        mp.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", None)
        window = small_window(mp, protocol.STAGE_BACKCAST)
        yield window, assemble(window)
    finally:
        mp.undo()


def test_the_engine_books_perps_through_cash_so_an_unscaled_net_long_book_borrows(
    _stage: Any,
) -> None:
    """The mechanism behind Amendment 3: without the net bound this book sits inside the
    margin cap (0.99) yet its cash is about -0.65 x capital."""
    window, stage = _stage
    start = window.first_decision_index
    book = _w(s0=0.55, p0=0.55, p1=0.55)
    assert cash.cash_measure(book) <= 1.0
    cell = _replay_cell(stage.dataset, book, start=start, capped=False)
    assert protocol.REASON_NEGATIVE_CASH in cell.invalid_reasons
    assert not cell.valid
    assert cell.max_borrowing is not None and cell.max_borrowing > 10_000.0
    assert cell.min_cash is not None and cell.min_cash < -50_000.0


def _borrowing_of(
    stage: Any, window: Any, *, net_limit: float
) -> tuple[float, OosCell]:
    start = window.first_decision_index
    policy = cash.AnchorCashAccountPolicy(
        _Fixed(_w(s0=0.55, p0=0.55, p1=0.55)), start_index=start, net_limit=net_limit
    )
    result = run_shared_portfolio_replay(
        stage.dataset,
        policy,
        start_index=start,
        stop_index=start + 40,
        initial_capital=protocol.INITIAL_CAPITAL,
        execution_cost=protocol.scenario_cost("BASE"),
        risk=PreTradeRisk(protocol.risk_config()),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    cell = runner._build_cell(
        arm=protocol.CANDIDATE,
        scenario="BASE",
        result=result,
        dataset=stage.dataset,
        causality_passed=True,
        probe_key="x",
        cap_statistics=runner._stats_of(policy),
    )
    assert cell.max_borrowing is not None
    return cell.max_borrowing, cell


def test_the_net_bound_removes_the_borrowing_of_the_net_long_counterexample(
    _stage: Any,
) -> None:
    """Without the bound the book borrows about 65% of capital. At exactly 1.0 the only
    residual is the entry fill: the order is sized at the decision close and filled at the
    next bar's price, so a book at net 1.0 overshoots by the price gap (under 0.01% of
    capital, first interval only). The frozen limit 0.97 (Amendment 4) absorbs it: the cell
    is valid with zero borrowing."""
    window, stage = _stage
    unbound = _replay_cell(
        stage.dataset,
        _w(s0=0.55, p0=0.55, p1=0.55),
        start=window.first_decision_index,
        capped=False,
    )
    assert unbound.max_borrowing is not None and unbound.max_borrowing > 50_000.0
    exact, exact_cell = _borrowing_of(stage, window, net_limit=1.0)
    assert 0.0 < exact < 1e-3 * protocol.INITIAL_CAPITAL
    assert exact_cell.cap_statistics["bound_by_net_exposure"] >= 1
    assert exact_cell.cap_statistics["bound_by_margin"] == 0
    # cash never goes meaningfully negative: minus costs plus the entry-gap overshoot only
    assert exact_cell.min_cash is not None and exact_cell.min_cash > -1_000.0
    frozen, cell = _borrowing_of(stage, window, net_limit=protocol.NET_EXPOSURE_LIMIT)
    assert protocol.NET_EXPOSURE_LIMIT == 0.97
    assert frozen == 0.0
    assert cell.invalid_reasons == () and cell.valid
    assert cell.min_cash is not None and cell.min_cash > 0.0 - 1_000.0


def test_the_replay_runs_with_the_amended_drawdown_ramp(_stage: Any) -> None:
    window, stage = _stage
    seen: list[tuple[float, float]] = []

    class Recorder(_Fixed):
        def decide_portfolio(
            self, snapshot: PortfolioDecisionSnapshot
        ) -> PortfolioTargetProposal:
            seen.append((snapshot.risk.drawdown_start, snapshot.risk.drawdown_stop))
            return super().decide_portfolio(snapshot)

    start = window.first_decision_index
    run_shared_portfolio_replay(
        stage.dataset,
        Recorder(np.zeros(N_LEGS)),
        start_index=start,
        stop_index=start + 10,
        initial_capital=protocol.INITIAL_CAPITAL,
        execution_cost=protocol.scenario_cost("BASE"),
        risk=PreTradeRisk(protocol.risk_config()),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )
    assert seen and set(seen) == {(0.125, 0.25)}


def test_a_hedged_carry_book_does_not_borrow_and_is_valid(_stage: Any) -> None:
    window, stage = _stage
    start = window.first_decision_index
    cell = _replay_cell(
        stage.dataset, _w(s0=0.5, p0=-0.5, s1=0.4, p1=-0.4), start=start
    )
    assert cell.invalid_reasons == () and cell.valid
    assert cell.max_borrowing == 0.0
    assert cell.min_cash is not None and cell.min_cash > 0.0


def test_a_fully_invested_spot_book_is_cut_to_the_net_limit_and_keeps_a_cash_buffer(
    _stage: Any,
) -> None:
    window, stage = _stage
    start = window.first_decision_index
    spot_only = _w(
        s0=0.2, s1=0.2, s2=0.2, s3=0.2, s4=0.2
    )  # gross 1.0 > the 0.97 net limit
    cell = _replay_cell(stage.dataset, spot_only, start=start)
    assert cell.invalid_reasons == () and cell.valid
    assert cell.max_borrowing == 0.0
    assert cell.cap_statistics["bound_by_net_exposure"] >= 1
    # about 3% of capital stays in cash
    assert cell.min_cash is not None and cell.min_cash > 0.02 * protocol.INITIAL_CAPITAL


# ================================================================= stride and deadband


def _first_active_rebalance(signals: Any, start: int) -> int:
    for index in range(start, start + 18 * 6, 18):
        spot, perp = anchor_targets.synthesize_row(signals, index - 1)
        if float(np.abs(spot).sum() + np.abs(perp).sum()) > 0.2:
            return index
    raise AssertionError("no rebalance row with a meaningful target")


def test_the_anchor_rebalances_every_18_bars_and_holds_in_between(_stage: Any) -> None:
    window, stage = _stage
    start = window.first_decision_index
    signals = runner.anchor_signals_of(stage)
    assert protocol.REBALANCE_EVERY_BARS == 18 and protocol.DEADBAND == 0.04
    assert anchor_targets.DEADBAND == protocol.DEADBAND
    policy = runner.make_policy(
        protocol.CANDIDATE,
        anchor=signals,
        control=runner.build_oos_signals(stage.dataset),
        start_index=start,
        lag=1,
    )
    assert isinstance(policy, cash.AnchorCashAccountPolicy)
    inner = AgyAnchorPolicy(
        signals=signals,
        start_index=start,
        signal_lag_bars=1,
        rebalance_every_bars=protocol.REBALANCE_EVERY_BARS,
    )
    held_all = tuple(range(N_LEGS))
    rebalances: list[int] = []
    for index in range(start, start + 18 * 6):
        proposal = inner.decide_portfolio(_snapshot(index))
        if (index - start) % 18 != 0:
            assert proposal.hold_current_quantity_symbols == held_all, index
        elif proposal.hold_current_quantity_symbols != held_all:
            rebalances.append(index)
    assert rebalances, "the anchor never traded in six periods"
    assert all((i - start) % 18 == 0 for i in rebalances)


def test_the_deadband_is_the_labs_total_turnover_and_holds_the_whole_book(
    _stage: Any,
) -> None:
    window, stage = _stage
    start = window.first_decision_index
    signals = runner.anchor_signals_of(stage)
    index = _first_active_rebalance(signals, start)
    row = index - 1
    spot, perp = anchor_targets.synthesize_row(signals, row)
    target = anchor_targets.interleave_legs(spot, perp)
    valid = anchor_targets.interleave_legs(
        signals.spot_valid[index].astype(float), signals.perp_valid[index].astype(float)
    ).astype(bool)

    def decide(weights: np.ndarray) -> PortfolioTargetProposal:
        policy = AgyAnchorPolicy(
            signals=signals,
            start_index=start,
            signal_lag_bars=1,
            rebalance_every_bars=protocol.REBALANCE_EVERY_BARS,
        )
        # the policy needs a first rebalance bar: replay the starting bars flat
        for step in range(start, index):
            policy.decide_portfolio(_snapshot(step, weights=weights))
        return policy.decide_portfolio(_snapshot(index, weights=weights))

    held_all = tuple(range(N_LEGS))
    at_target = np.where(valid, target, 0.0)
    assert decide(at_target).hold_current_quantity_symbols == held_all  # turnover 0
    leg = int(np.flatnonzero(valid)[0])
    other = int(np.flatnonzero(valid)[1])
    small = at_target.copy()
    small[leg] += 0.015
    small[other] -= 0.015  # total turnover 0.03 < 0.04: the whole book is held
    assert decide(small).hold_current_quantity_symbols == held_all
    big = at_target.copy()
    big[leg] += 0.03
    big[other] -= 0.03  # total turnover 0.06 >= 0.04: re-target
    assert decide(big).hold_current_quantity_symbols != held_all
    far = np.zeros(
        N_LEGS
    )  # from flat the whole target is traded (invalid legs are held)
    proposal = decide(far)
    assert proposal.hold_current_quantity_symbols == tuple(
        int(i) for i in np.flatnonzero(~valid)
    )
    np.testing.assert_allclose(
        np.asarray(proposal.target_weights)[valid], target[valid], atol=1e-15
    )


# ====================================================================== decision rule


def _cell(arm: str, scenario: str, **over: Any) -> OosCell:
    values: dict[str, Any] = {
        "arm": arm,
        "scenario": scenario,
        "total_return": 0.10,
        "annual_sharpe": 1.0,
        "max_drawdown": 0.05,
        "total_cost": 10.0,
        "funding_pnl": 0.0,
        "turnover_total": 3.0,
        "complete": True,
        "terminal_flat": True,
        "terminal_dust_only": False,
        "execution_valid": True,
        "causality_passed": True,
        "ledger_reconciled": True,
        "stopped": False,
        "stop_reason": None,
        "invalid_reasons": (),
        "n_returns": 4,
        "calendar_year_returns": {"2020": 0.1},
        "returns": (0.01, -0.01, 0.02, 0.0),
    }
    values.update(over)
    return OosCell(**values)


def _grid(
    over: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> dict[Any, OosCell]:
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for scenario in protocol.SCENARIOS:
        cells[("AGY_ANCHOR", scenario)] = {
            "total_return": 0.2,
            "annual_sharpe": 0.8,
            "max_drawdown": 0.1,
        }
        cells[("EW_HOLD_SPOT", scenario)] = {
            "total_return": 0.5,
            "annual_sharpe": 3.0,  # far better than the anchor: it must not matter
            "max_drawdown": 0.6,
        }
        cells[("CASH", scenario)] = {
            "total_return": 0.0,
            "annual_sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_cost": 0.0,
            "turnover_total": 0.0,
        }
    for key, extra in (over or {}).items():
        cells[key] = {**cells[key], **extra}
    return {k: _cell(*k, **v) for k, v in cells.items()}


CAND = "AGY_ANCHOR"


def test_a_stage_with_every_condition_met_is_supported_whatever_the_control_does() -> (
    None
):
    cells = _grid()
    conds = decision.conditions(cells)
    assert len(conds) == 5 and all(conds.values())
    assert decision.stage_verdict(cells) == decision.VERDICT_SUPPORTED
    out = decision.decide_stage(cells, stage=protocol.STAGE_BACKCAST)
    assert out["supported"] is True and out["failed_conditions"] == []
    assert out["invalid_cells"] == []


@pytest.mark.parametrize(
    ("over", "failed"),
    [
        ({(CAND, "BASE"): {"total_return": 0.0}}, "3_candidate_base_return_above_zero"),
        (
            {(CAND, "BASE"): {"total_return": -0.01}},
            "3_candidate_base_return_above_zero",
        ),
        (
            {(CAND, "BASE"): {"annual_sharpe": 0.4999}},
            "4_candidate_base_sharpe_at_least_min",
        ),
        (
            {(CAND, "BASE"): {"max_drawdown": 0.25}},
            "2_no_candidate_cell_drawdown_at_least_limit",
        ),
        (
            {(CAND, "HIGH_COST"): {"max_drawdown": 0.30}},
            "2_no_candidate_cell_drawdown_at_least_limit",
        ),
        (
            {(CAND, "DELAY_1BAR"): {"max_drawdown": 0.35}},
            "2_no_candidate_cell_drawdown_at_least_limit",
        ),
        (
            {(CAND, "HIGH_COST"): {"total_return": 0.0}},
            "5_candidate_high_cost_and_delay_returns_above_zero",
        ),
        (
            {(CAND, "DELAY_1BAR"): {"total_return": -0.1}},
            "5_candidate_high_cost_and_delay_returns_above_zero",
        ),
    ],
)
def test_each_condition_flips_the_stage_to_not_supported(
    over: dict[Any, dict[str, Any]], failed: str
) -> None:
    cells = _grid(over)
    assert decision.stage_verdict(cells) == decision.VERDICT_NOT_SUPPORTED
    out = decision.decide_stage(cells, stage=protocol.STAGE_FORWARD)
    assert out["failed_conditions"] == [failed]
    assert out["invalid_cells"] == []


def test_the_boundaries_are_exactly_the_spec() -> None:
    assert decision.stage_verdict(_grid({(CAND, "BASE"): {"annual_sharpe": 0.5}})) == (
        decision.VERDICT_SUPPORTED  # at least 0.50
    )
    assert decision.stage_verdict(
        _grid({(CAND, "BASE"): {"max_drawdown": 0.2499}})
    ) == (
        decision.VERDICT_SUPPORTED  # below 0.25 (Amendment 2)
    )
    assert decision.stage_verdict(_grid({(CAND, "BASE"): {"max_drawdown": 0.22}})) == (
        decision.VERDICT_SUPPORTED  # 0.20 and above is allowed since the amendment
    )
    assert decision.stage_verdict(_grid({(CAND, "BASE"): {"total_return": 1e-12}})) == (
        decision.VERDICT_SUPPORTED  # above 0
    )


@pytest.mark.parametrize(
    "over",
    [
        {("CASH", "BASE"): {"causality_passed": False}},
        {("EW_HOLD_SPOT", "HIGH_COST"): {"ledger_reconciled": False}},
        {(CAND, "DELAY_1BAR"): {"invalid_reasons": ("negative_cash_borrowing",)}},
        {(CAND, "BASE"): {"terminal_flat": False}},
        {(CAND, "BASE"): {"total_cost": float("nan")}},
    ],
)
def test_any_invalid_cell_makes_the_stage_invalid_even_when_the_rest_passes(
    over: dict[Any, dict[str, Any]],
) -> None:
    cells = _grid(over)
    assert decision.stage_verdict(cells) == decision.VERDICT_INVALID
    out = decision.decide_stage(cells, stage=protocol.STAGE_BACKCAST)
    assert out["supported"] is False and out["invalid_cells"]


def test_the_label_truth_table_covers_every_pair_of_verdicts() -> None:
    SUP, NOT, INV = (
        decision.VERDICT_SUPPORTED,
        decision.VERDICT_NOT_SUPPORTED,
        decision.VERDICT_INVALID,
    )
    expected = {
        (SUP, SUP): "ANCHOR_SUPPORTED",
        (NOT, SUP): "ANCHOR_FORWARD_ONLY",
        (SUP, NOT): "ANCHOR_NOT_SUPPORTED",
        (NOT, NOT): "ANCHOR_NOT_SUPPORTED",
        (INV, SUP): "INVALID",
        (INV, NOT): "INVALID",
        (INV, INV): "INVALID",
        (SUP, INV): "INVALID",
        (NOT, INV): "INVALID",
    }  # (backcast, forward)
    assert set(expected) == set(itertools.product(decision.VERDICTS, repeat=2))
    for (backcast, forward), label in expected.items():
        assert decision.combine_label(backcast, forward) == label, (backcast, forward)
    assert set(expected.values()) == set(protocol.LABELS)
    with pytest.raises(ValueError, match="unknown backcast"):
        decision.combine_label("OOS_SUPPORTED", SUP)
    with pytest.raises(ValueError, match="unknown forward"):
        decision.combine_label(SUP, "x")


def test_the_grid_must_be_exactly_the_nine_cells() -> None:
    cells = _grid()
    short = {k: v for k, v in cells.items() if k != (CAND, "BASE")}
    with pytest.raises(ValueError, match="missing"):
        decision.conditions(short)
    with pytest.raises(ValueError, match="extra"):
        decision.conditions({**cells, ("X", "BASE"): _cell("X", "BASE")})
    swapped = {**cells, (CAND, "BASE"): _cell("CASH", "BASE")}
    with pytest.raises(ValueError, match="does not match"):
        decision.conditions(swapped)


def test_dev_never_gates_and_never_carries_a_label() -> None:
    report = decision.dev_report(_grid())
    assert report["label"] is None and report["gates"] is False
    assert report["status"] == decision.DEV_STATUS
    assert "conditions" not in report and "stage_verdict" not in report
    with pytest.raises(ValueError, match="DEV is diagnostic"):
        decision.decide_stage(_grid(), stage=protocol.STAGE_DEV)
    # an invalid cell is reported, not turned into a label
    bad = decision.dev_report(_grid({(CAND, "BASE"): {"causality_passed": False}}))
    assert bad["label"] is None and bad["invalid_cells"] == ["AGY_ANCHOR__BASE"]


def test_the_interim_report_carries_no_label_and_no_verdict() -> None:
    report = decision.interim_report(_grid())
    assert report["label"] is None and report["status"] == "INTERIM_NOT_A_DECISION"
    assert "conditions" not in report and "stage_verdict" not in report


def test_the_bootstrap_is_over_cash_with_the_frozen_seed_and_equals_the_definition() -> (
    None
):
    rng = np.random.default_rng(5)
    cand = tuple(float(x) for x in rng.normal(0.0004, 0.01, 41))
    ctrl = tuple(0.0 for _ in cand)
    cells = _grid(
        {
            (CAND, "BASE"): {"returns": cand, "n_returns": 41},
            ("CASH", "BASE"): {"returns": ctrl, "n_returns": 41},
        }
    )
    out = decision.bootstrap_excess(cells)
    assert list(out) == ["excess_over_CASH"]
    stats = out["excess_over_CASH"]
    assert stats["available"] and stats["seed"] == 20261121
    assert (stats["block_mean_bars"], stats["resamples"]) == (42, 10_000)
    assert stats["gates"] is False
    want = _slow_bootstrap(
        np.asarray(cand) - np.asarray(ctrl),
        block_mean=42,
        resamples=300,
        seed=20261121,
        chunk=250,
        confidence=0.95,
    )
    got = decision.stationary_bootstrap(
        np.asarray(cand), block_mean=42, resamples=300, seed=20261121, chunk=250
    )
    assert got["lower_bound"] == pytest.approx(want, abs=1e-15)
    assert decision.bootstrap_excess(_grid())["excess_over_CASH"]["available"] is True
    short = _grid({(CAND, "BASE"): {"returns": (0.0,), "n_returns": 1}})
    assert decision.bootstrap_excess(short)["excess_over_CASH"]["available"] is False
    # the bootstrap never gates: a hostile series changes no verdict
    hostile = _grid({(CAND, "BASE"): {"returns": (-0.5,) * 10, "n_returns": 10}})
    assert decision.stage_verdict(hostile) == decision.VERDICT_SUPPORTED


# ========================================================================== the runner


def _run(
    tmp_path: Path,
    stage_name: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    name: str = "out",
    mode: str = runner.MODE_DECISION,
    backcast_verdict: str | None = None,
    limit_bars_early: int = 0,
    authorization: Any = None,
) -> tuple[dict[str, Any], Path, data.StageWindow]:
    window = small_window(monkeypatch, stage_name)
    end = window.last_row_close - limit_bars_early * BAR
    spot, perp = leg_maps(window)
    stage = assemble(
        window,
        limit=end,
        spot={k: truncate_leg(v, end) for k, v in spot.items()},
        perp={k: truncate_leg(v, end) for k, v in perp.items()},
    )
    out = tmp_path / name
    result = runner.run_stage(
        stage,
        window=window,
        output_dir=out,
        expected_protocol_digest=protocol.protocol_digest(),
        mode=mode,
        backcast_verdict=backcast_verdict,
        code_commit="c" * 40,
        authorization=authorization,
        run_date=RUN_DATE,
    )
    return result, out, window


def _cell_files(out: Path) -> list[str]:
    return sorted(p.name for p in (out / runner.CELL_DIR).glob("*.json"))


def test_a_backcast_run_writes_nine_cells_a_manifest_and_a_stage_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, out, window = _run(tmp_path, protocol.STAGE_BACKCAST, monkeypatch)
    assert len(_cell_files(out)) == 9
    assert (out / runner.DECISION_NAME).exists() and (
        out / runner.MANIFEST_NAME
    ).exists()
    assert result["stage"] == "BACKCAST_2020" and result["mode"] == "decision"
    assert result["stage_verdict"] in decision.VERDICTS
    assert result["label"] in (None, "INVALID")
    assert result["protocol_digest"] == protocol.protocol_digest()
    assert result["port_source_digest"] == protocol.PORT_SOURCE_DIGEST
    assert result["parity_fixture_sha256"] == protocol.FIXTURE_SHA256
    assert len(result["taker_buy_quote_volume_sha256"]) == 64
    assert result["replay_start_index"] == window.first_decision_index
    assert result["replay_stop_index"] == window.last_row_index
    manifest = json.loads((out / runner.MANIFEST_NAME).read_text("utf-8"))
    assert sorted(manifest["causality_probes"]) == [
        "AGY_ANCHOR__lag1",
        "AGY_ANCHOR__lag2",
        "CASH__lag0",
        "CASH__lag1",
        "EW_HOLD_SPOT__lag0",
        "EW_HOLD_SPOT__lag1",
    ]
    assert all(p["passed"] for p in manifest["causality_probes"].values())
    cells = result["cells"]
    assert set(cells) == {
        f"{a}__{s}" for a in protocol.ARMS for s in protocol.SCENARIOS
    }
    anchor = cells["AGY_ANCHOR__BASE"]
    assert anchor["cell_valid"], anchor["invalid_reasons"]
    assert anchor["turnover_total"] > 0.0  # the anchor trades
    assert anchor["cap_statistics"]["initial_margin_rate"] == 0.4
    assert cells["CASH__BASE"]["total_return"] == 0.0
    assert cells["AGY_ANCHOR__BASE"]["total_cost"] > 0.0
    assert cells["AGY_ANCHOR__BASE"]["funding_pnl"] != 0.0  # perps carry funding
    assert result["bootstrap"]["excess_over_CASH"]["seed"] == 20261121
    # write-once
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        runner.run_stage(
            assemble(window),
            window=window,
            output_dir=out,
            expected_protocol_digest=protocol.protocol_digest(),
            run_date=RUN_DATE,
        )


def test_the_anchor_and_a_flat_book_differ_and_delay_costs_less_or_more_never_equal_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, _, _ = _run(tmp_path, protocol.STAGE_BACKCAST, monkeypatch)
    cells = result["cells"]
    base = cells["AGY_ANCHOR__BASE"]["total_return"]
    high = cells["AGY_ANCHOR__HIGH_COST"]["total_return"]
    delay = cells["AGY_ANCHOR__DELAY_1BAR"]["total_return"]
    assert high < base  # higher costs, same decisions
    assert delay != base  # a one-bar later signal is a different book


def test_a_dev_run_reports_but_never_labels_and_names_its_result_dev_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, out, _ = _run(tmp_path, protocol.STAGE_DEV, monkeypatch)
    assert (out / runner.DEV_NAME).exists() and not (
        out / runner.DECISION_NAME
    ).exists()
    assert result["label"] is None and result["status"] == decision.DEV_STATUS
    assert "stage_verdict" not in result and result["gates"] is False
    manifest = json.loads((out / runner.MANIFEST_NAME).read_text("utf-8"))
    assert manifest["label"] is None and manifest["result_file"] == "dev-result.json"


def test_a_forward_decision_needs_the_backcast_verdict_and_combines_the_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_FORWARD)
    stage = assemble(window, authorization_url=protocol.DATA_AUTHORIZATION_URL)
    with pytest.raises(ValueError, match="BACKCAST_2020 stage verdict"):
        runner.run_stage(
            stage,
            window=window,
            output_dir=tmp_path / "none",
            expected_protocol_digest=protocol.protocol_digest(),
            run_date=RUN_DATE,
        )
    assert not (tmp_path / "none").exists()
    result, out, _ = _run(
        tmp_path,
        protocol.STAGE_FORWARD,
        monkeypatch,
        backcast_verdict=decision.VERDICT_NOT_SUPPORTED,
    )
    verdict = result["stage_verdict"]
    assert result["label"] == decision.combine_label(
        decision.VERDICT_NOT_SUPPORTED, verdict
    )
    assert result["backcast_stage_verdict"] == decision.VERDICT_NOT_SUPPORTED
    assert (out / runner.DECISION_NAME).exists()
    assert "through_close" in result and result["through_close"].endswith("Z")


def test_a_forward_interim_writes_a_dated_report_and_never_a_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result, out, _ = _run(
        tmp_path,
        protocol.STAGE_FORWARD,
        monkeypatch,
        mode=runner.MODE_INTERIM,
        limit_bars_early=40,
    )
    name = "INTERIM_NOT_A_DECISION-20261120.json"
    assert (out / name).exists() and not (out / runner.DECISION_NAME).exists()
    assert result["label"] is None and result["status"] == "INTERIM_NOT_A_DECISION"
    assert "stage_verdict" not in result and "conditions" not in result
    assert len(_cell_files(out)) == 9


def test_a_decision_needs_the_whole_window_and_a_dev_or_backcast_has_no_interim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    spot, perp = leg_maps(window)
    end = window.last_row_close - 10 * BAR
    short = assemble(
        window,
        limit=end,
        spot={k: truncate_leg(v, end) for k, v in spot.items()},
        perp={k: truncate_leg(v, end) for k, v in perp.items()},
    )
    kwargs: dict[str, Any] = {
        "window": window,
        "output_dir": tmp_path / "x",
        "expected_protocol_digest": protocol.protocol_digest(),
        "run_date": RUN_DATE,
    }
    with pytest.raises(data.StageDataError, match="whole stage window"):
        runner.run_stage(short, **kwargs)
    with pytest.raises(ValueError, match="only FORWARD has an interim"):
        runner.run_stage(short, mode=runner.MODE_INTERIM, **kwargs)
    with pytest.raises(ValueError, match="unknown mode"):
        runner.run_stage(short, mode="final", **kwargs)
    with pytest.raises(ValueError, match="does not match the frozen"):
        runner.run_stage(short, **{**kwargs, "expected_protocol_digest": "0" * 64})
    assert not (tmp_path / "x").exists()


def test_every_cell_is_written_before_the_decision_is_computed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("decision computed")

    monkeypatch.setattr(runner.decision, "decide_stage", boom)
    seen: list[str] = []
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    with pytest.raises(RuntimeError, match="decision computed"):
        runner.run_stage(
            assemble(window),
            window=window,
            output_dir=tmp_path / "o",
            expected_protocol_digest=protocol.protocol_digest(),
            run_date=RUN_DATE,
            on_cell=lambda cell: seen.append(cell.arm + "__" + cell.scenario),
        )
    assert len(seen) == 9 and len(_cell_files(tmp_path / "o")) == 9
    assert not (tmp_path / "o" / runner.DECISION_NAME).exists()
    assert not (tmp_path / "o" / runner.MANIFEST_NAME).exists()


def test_a_policy_that_reads_the_future_fails_the_probe_and_the_cells_are_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Peeking(AgyAnchorPolicy):
        def decide_portfolio(
            self, snapshot: PortfolioDecisionSnapshot
        ) -> PortfolioTargetProposal:
            proposal = super().decide_portfolio(snapshot)
            ahead = min(snapshot.index + 1, self._signals.n_rows - 1)
            weights = list(proposal.target_weights)
            free = [
                i
                for i in range(len(weights))
                if i not in proposal.hold_current_quantity_symbols
            ]
            if free:  # a held leg must keep its current weight
                weights[free[0]] += 1e-2 * float(
                    self._signals.w_aw_bidir_75_25[ahead][0]
                )
            return PortfolioTargetProposal(
                target_weights=tuple(weights),
                hold_current_quantity_symbols=proposal.hold_current_quantity_symbols,
            )

    monkeypatch.setattr(runner, "AgyAnchorPolicy", Peeking)
    result, out, _ = _run(tmp_path, protocol.STAGE_BACKCAST, monkeypatch)
    manifest = json.loads((out / runner.MANIFEST_NAME).read_text("utf-8"))
    assert not manifest["causality_probes"]["AGY_ANCHOR__lag1"]["passed"]
    assert manifest["causality_probes"]["CASH__lag0"]["passed"]
    assert result["stage_verdict"] == decision.VERDICT_INVALID
    assert result["label"] == "INVALID"
    assert (
        "causality_probe_failed"
        in result["cells"]["AGY_ANCHOR__BASE"]["invalid_reasons"]
    )
    assert result["cells"]["CASH__BASE"]["cell_valid"]


def test_the_probe_runs_over_every_lag_of_every_arm_with_the_right_lags() -> None:
    assert runner._arm_lags(protocol.CANDIDATE) == [1, 2]
    assert runner._arm_lags(protocol.CONTROL_CASH) == [0, 1]
    assert runner._arm_lags(protocol.CONTROL_EW_SPOT) == [0, 1]
