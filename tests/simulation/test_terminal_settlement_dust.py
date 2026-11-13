"""Terminal dust below the venue minimum notional stays a position, valued honestly.

A holding can decay below the exchange minimum notional after it was opened (adverse
move, drawdown de-leveraging). The engine keeps enforcing the venue minimum
everywhere, including the terminal close, so such a residual cannot be traded and the
terminal book is not flat. The ledger reconciliation (``reconcile_ledger``) records it
as terminal dust, values it at mark minus the modelled exit cost, and accepts the run
only when every residual is such dust; a genuine stuck position still fails.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.trend_stop_support import synthetic_dataset
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study.protocol import PROTOCOL
from trade_rl.evaluation.dev_realdata_study.runner import (
    reconcile_ledger,
    terminal_dust_adjusted_equity,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.simulation import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.native_transition import execute_native_target_transition
from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession
from trade_rl.strategies.allocation.buy_and_hold import BuyAndHoldPortfolioPolicy

SYMBOLS = ("AAAUSDT", "BBBUSDT")
MINIMUM_NOTIONAL = 5.0
CAPITAL = 100.0
DROP_ROW = 8
COST = ExecutionCostConfig(max_participation_rate=0.5)
PROTO = dataclasses.replace(PROTOCOL, initial_capital=CAPITAL)


def _dataset(
    *,
    drop: bool,
    thin_last_volume: bool = False,
    terminal_open_rebounds: bool = False,
) -> MarketDataset:
    """Prices of 100 that halve at ``DROP_ROW`` when ``drop``; lot 0.01, minimum 5."""

    base = synthetic_dataset(
        "4h",
        SYMBOLS,
        first_close="2021-01-01T00:00:00",
        last_close="2021-01-05T00:00:00",
    )
    shape = (base.n_bars, len(SYMBOLS))
    price = np.full(shape, 100.0)
    if drop:
        price[DROP_ROW:] = 50.0
    opens = price.copy()
    highs = price.copy()
    if terminal_open_rebounds:
        assert drop, "the terminal open/mark counterexample needs a prior drop"
        opens[-1] = 100.0
        highs[-1] = 100.0
    volume = np.full(shape, 1.0e12)
    if thin_last_volume:
        volume[-1] = 1.0e-9  # no liquidity on the terminal bar
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.init}
    fields.update(
        open=opens,
        high=highs,
        low=price.copy(),
        close=price.copy(),
        mark_price=price.copy(),
        volume=volume,
        lot_size=np.full(shape, 0.01),
        minimum_notional=np.full(shape, MINIMUM_NOTIONAL),
    )
    return MarketDataset(**fields)


def _replay(dataset: MarketDataset):
    # Two positions of 6 USDT each pass the 5 USDT entry filter.
    return run_shared_portfolio_replay(
        dataset,
        BuyAndHoldPortfolioPolicy(symbols=SYMBOLS, gross_budget=0.12),
        start_index=1,
        stop_index=dataset.n_bars - 1,
        initial_capital=CAPITAL,
        execution_cost=COST,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )


def _unit_cost(notional: float, dataset: MarketDataset, symbol: int) -> float:
    """Independent fee + spread + impact rate of one market exit (zero slippage)."""

    market_notional = float(dataset.volume[-1, symbol]) * 50.0
    participation = notional / market_notional
    return COST.multiplier * (
        COST.fee_rate + COST.spread_rate + COST.impact_rate * math.sqrt(participation)
    )


def test_dust_case_reconciles_as_terminal_dust_only_with_exit_cost_haircut() -> None:
    dataset = _dataset(drop=True)
    result = _replay(dataset)
    report = reconcile_ledger(result, dataset, PROTO, execution_cost=COST)

    assert report["terminal_flat"] is False
    assert report["terminal_dust_only"] is True
    assert report["passed"] is True
    dust = report["terminal_dust"]
    assert [item["symbol"] for item in dust] == list(SYMBOLS)
    for symbol, item in enumerate(dust):
        assert item["quantity"] == "3/50"  # 0.06 units, exact
        assert item["notional_usdt"] == pytest.approx(3.0)
        assert item["minimum_notional_usdt"] == MINIMUM_NOTIONAL
        assert item["reason"] == "below_minimum_notional"
        expected_cost = 3.0 * _unit_cost(3.0, dataset, symbol)
        assert item["exit_cost_usdt"] == pytest.approx(expected_cost, rel=1e-9)
    total_cost = sum(item["exit_cost_usdt"] for item in dust)
    assert report["terminal_dust_exit_cost_usdt"] == pytest.approx(total_cost)
    # cash reconciles; equity is cash plus the dust at mark MINUS the exit cost.
    assert abs(float(report["cash_residual"])) <= PROTO.reconciliation_tolerance
    assert report["terminal_equity_dust_adjusted"] == pytest.approx(
        float(report["ledger_final_equity"]) - total_cost
    )
    # the equity path never credits the exit that could not happen
    raw = np.array([1.0, float(report["ledger_final_equity"])])
    adjusted = terminal_dust_adjusted_equity(raw, report)
    assert adjusted[-1] == pytest.approx(raw[-1] - total_cost)
    assert adjusted[0] == raw[0]


def test_dust_cannot_be_valued_without_the_cost_model() -> None:
    dataset = _dataset(drop=True)
    report = reconcile_ledger(_replay(dataset), dataset, PROTO)
    assert report["terminal_dust_only"] is False
    assert report["passed"] is False


def test_dust_requires_the_engine_to_have_rejected_the_close_for_the_minimum() -> None:
    # Sub-minimum at the open as well, with zero liquidity: admission rejects the
    # close for the minimum before capacity is consulted, so this is dust.
    dataset = _dataset(drop=True, thin_last_volume=True)
    result = _replay(dataset)
    ledger = result.ledger_evidence
    assert ledger is not None
    reasons = {
        event.reason
        for event in ledger.intervals.intervals[-1].order_events
        if event.event_type == "rejected"
    }
    assert reasons == {"below_minimum_notional"}
    report = reconcile_ledger(result, dataset, PROTO, execution_cost=COST)
    assert report["terminal_dust_only"] is True
    assert report["passed"] is True


def test_a_genuine_stuck_position_still_fails() -> None:
    # 6 USDT >= the 5 USDT minimum, but the terminal bar has no liquidity: the close
    # fails for a reason other than the minimum notional.
    dataset = _dataset(drop=False, thin_last_volume=True)
    report = reconcile_ledger(_replay(dataset), dataset, PROTO, execution_cost=COST)
    assert report["terminal_flat"] is False
    assert report["terminal_dust_only"] is False
    assert report["terminal_dust"] == []
    assert report["passed"] is False


def test_mark_dust_is_not_venue_dust_when_terminal_open_is_tradable() -> None:
    """A liquidity-blocked 6 USDT close must not pass as 3 USDT mark dust."""
    dataset = _dataset(
        drop=True,
        thin_last_volume=True,
        terminal_open_rebounds=True,
    )
    result = _replay(dataset)
    report = reconcile_ledger(result, dataset, PROTO, execution_cost=COST)
    residuals = report["terminal_residual_quantities"]
    assert set(residuals) == set(SYMBOLS)
    for index, symbol in enumerate(SYMBOLS):
        quantity = abs(float(Fraction(residuals[symbol])))
        mark = float(dataset.resolved_array("mark_price")[-1, index])
        opening = float(dataset.open[-1, index])
        assert quantity * opening >= MINIMUM_NOTIONAL
        assert quantity * mark < MINIMUM_NOTIONAL

    # The engine refuses the close for liquidity, not for the venue minimum.
    assert report["terminal_flat"] is False
    assert report["terminal_dust_only"] is False
    assert report["terminal_dust"] == []
    assert report["passed"] is False
    assert report["terminal_dust_exit_cost_usdt"] == 0.0


def test_classification_follows_the_engines_own_rejection_basis() -> None:
    """Admission tests the minimum at the unrounded open, whatever tick rounding does.

    Open 79 with a tick of 30: the 0.06 close is 4.74 USDT at the open (below the 5
    minimum) although the allocator's tick-rounded price would give 5.4. The engine
    rejects the close for ``below_minimum_notional``, so that rejection - not a
    re-derived price bound - decides that the residual is dust.
    """
    original = _dataset(drop=True, thin_last_volume=True)
    fields = {
        field.name: getattr(original, field.name)
        for field in dataclasses.fields(original)
        if field.init
    }
    opening = original.open.copy()
    highs = original.high.copy()
    ticks = original.resolved_array("tick_size").copy()
    opening[-1] = 79.0
    highs[-1] = 79.0
    ticks[-1] = 30.0
    dataset = MarketDataset(
        **{**fields, "open": opening, "high": highs, "tick_size": ticks}
    )
    result = _replay(dataset)
    ledger = result.ledger_evidence
    assert ledger is not None
    rejections = {
        event.reason
        for event in ledger.intervals.intervals[-1].order_events
        if event.event_type == "rejected"
    }
    assert rejections == {"below_minimum_notional"}
    report = reconcile_ledger(result, dataset, PROTO, execution_cost=COST)
    assert report["terminal_dust_only"] is True
    assert report["passed"] is True


def test_runs_that_end_flat_are_unchanged_by_the_dust_valuation() -> None:
    # No price drop: both 6 USDT positions are tradable at the terminal close. The
    # feature touches only the reconciliation report and the equity path; for a flat
    # run both must equal the plain reconciliation, key by key, with exact equality
    # (no platform-dependent digest: both sides are computed here at test time).
    dataset = _dataset(drop=False)
    result = _replay(dataset)
    ledger = result.ledger_evidence
    assert ledger is not None
    assert all(Fraction(v) == 0 for v in ledger.intervals.terminal_exact_quantities)

    plain = reconcile_ledger(result, dataset, PROTO)
    valued = reconcile_ledger(result, dataset, PROTO, execution_cost=COST)
    assert valued == plain
    assert valued["terminal_flat"] is True
    assert valued["terminal_dust_only"] is False
    assert valued["terminal_dust"] == []
    assert valued["terminal_dust_exit_cost_usdt"] == 0.0
    assert valued["passed"] is True
    # the engine-side numbers the checker re-derives are unchanged by the feature
    assert valued["equity_residual"] == valued["dust_adjusted_equity_residual"]
    raw = np.array([1.0, 2.0, 3.0])
    np.testing.assert_array_equal(terminal_dust_adjusted_equity(raw, valued), raw)


def _ordinary_dust_state():
    dataset = _dataset(drop=True)
    session = SharedPortfolioSession(
        dataset,
        start_index=1,
        stop_index=dataset.n_bars - 1,
        initial_capital=CAPITAL,
        execution_cost=COST,
        settle_terminal_position=True,
    )
    session.advance(np.array([0.06, 0.06]))
    while session.current_index <= DROP_ROW:
        session.advance(session.book.weights, hold_current_quantity_symbols=(0, 1))
    return dataset, MarketExecutor(dataset, COST), session.book, session.current_index


def test_no_public_path_changes_the_venue_minimum() -> None:
    # The engine enforces the venue minimum everywhere, including a terminal close:
    # no waiver keyword, override or entry point exists.
    forbidden = ("closeout", "terminal_closeout", "waive", "minimum_notional_override")
    for function in (
        MarketExecutor.execute_interval,
        MarketExecutor.order_minimum_notional,
        execute_native_target_transition,
    ):
        names = set(inspect.signature(function).parameters)
        assert not [n for n in names if any(f in n for f in forbidden)], function
    import trade_rl.simulation.native_transition as native

    assert not hasattr(native, "execute_terminal_settlement_transition")
    orders = Path(native.__file__).parent / "orders"
    assert not (orders / "terminal_closeout.py").exists()

    dataset, executor, book, index = _ordinary_dust_state()
    risk = PreTradeRisk.default_for_execution(max_leverage=executor.cost.max_leverage)
    _, result, _ = execute_native_target_transition(
        executor,
        book,
        risk,
        np.zeros(2),
        start_index=index,
        emergency_flatten_mask=np.array([True, True]),
    )
    assert [Fraction(v) for v in result.book.exact_quantities] == [
        Fraction(v) for v in book.exact_quantities
    ]
    assert result.fill_count == 0

    # ...and the reserved terminal interval keeps the dust as well.
    replayed = _replay(_dataset(drop=True))
    assert replayed.ledger_evidence is not None
    held = replayed.ledger_evidence.intervals.terminal_exact_quantities
    assert any(Fraction(v) != 0 for v in held)


def test_trend_stop_cell_with_dust_only_terminal_is_valid_and_haircut() -> None:
    from trade_rl.evaluation.trend_stop_study import runner as trend_runner
    from trade_rl.evaluation.trend_stop_study.protocol import (
        PROTOCOL as TREND_PROTOCOL,
    )

    protocol = dataclasses.replace(TREND_PROTOCOL, initial_capital=CAPITAL)
    dataset = _dataset(drop=True)
    result = trend_runner._replay(
        dataset,
        BuyAndHoldPortfolioPolicy(symbols=SYMBOLS, gross_budget=0.12),
        protocol,
        "base",
        start_index=1,
        stop_index=dataset.n_bars - 1,
    )
    cell = trend_runner.summarize_cell(
        arm="equal_weight_buy_and_hold",
        timeframe="4h",
        variant=None,
        scenario="base",
        result=result,
        dataset=dataset,
        protocol=protocol,
    )
    reconciliation = cell.summary["reconciliation"]
    assert reconciliation["terminal_dust_only"] is True
    assert trend_runner._cell_valid(cell.summary) is True
    cost = reconciliation["terminal_dust_exit_cost_usdt"]
    assert cost > 0.0
    assert cell.equity[-1] == pytest.approx(cell.engine_equity[-1] - cost)


def test_phase2_replay_cell_with_dust_only_terminal_is_reconciled_and_haircut() -> None:
    from trade_rl.evaluation.dev_realdata_study import runner as dev_runner

    dataset = _dataset(drop=True)

    def builder(arm, data, protocol, sources):  # type: ignore[no-untyped-def]
        return dev_runner.BuiltArm(
            policy=BuyAndHoldPortfolioPolicy(symbols=SYMBOLS, gross_budget=0.12),
            adapter=None,
        )

    protocol = dataclasses.replace(
        PROTO, fee_rate=COST.fee_rate, spread_rate=COST.spread_rate
    )
    outcome = dev_runner.run_replay(
        dataset,
        protocol,
        "buy_and_hold",
        "base",
        None,
        start_index=1,
        stop_index=dataset.n_bars - 1,
        arm_builder=builder,
    )
    reconciliation = outcome.summary["reconciliation"]
    assert reconciliation["terminal_dust_only"] is True
    assert reconciliation["passed"] is True
    cost = reconciliation["terminal_dust_exit_cost_usdt"]
    assert cost > 0.0
    assert outcome.equity[-1] == pytest.approx(
        reconciliation["ledger_final_equity"] - cost
    )


@pytest.mark.parametrize("last_volume", (0.0, 1e-320))
def test_unrepresentable_dust_exit_valuation_never_passes(last_volume: float) -> None:
    """Zero/overflowing denominator cannot authorize a dust-only economic PASS."""
    dataset = _dataset(drop=True)
    volume = np.array(dataset.volume, copy=True)
    volume[-1, :] = last_volume
    dataset = dataclasses.replace(dataset, volume=volume)
    replayed = _replay(dataset)
    report = reconcile_ledger(replayed, dataset, PROTO, execution_cost=COST)
    assert report["terminal_flat"] is False
    assert report["terminal_dust_only"] is False
    assert report["passed"] is False
    assert report["terminal_dust_exit_cost_usdt"] == 0.0
    assert report["terminal_dust"] == []
    assert set(report["terminal_residual_quantities"]) == set(SYMBOLS)


def test_finite_positive_volume_remains_valued_dust_control() -> None:
    """The fail-closed guard must not reject an ordinary finite dust valuation."""
    dataset = _dataset(drop=True)
    report = reconcile_ledger(_replay(dataset), dataset, PROTO, execution_cost=COST)
    assert report["passed"] is True
    assert report["terminal_dust_only"] is True
    assert len(report["terminal_dust"]) == len(SYMBOLS)
    assert math.isfinite(report["terminal_dust_exit_cost_usdt"])
