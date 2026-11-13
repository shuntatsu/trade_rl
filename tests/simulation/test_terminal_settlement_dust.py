"""Terminal settlement must close residuals below the venue minimum notional.

A position can decay below the exchange minimum notional after it was opened
(adverse price move, drawdown de-leveraging). The reserved terminal interval is a
risk-owned reduce-only close; a venue does not apply the minimum-notional filter to
reduce-only orders, so the book must end flat and the independent ledger
reconciliation (terminal equity from raw fills) must pass.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
from fractions import Fraction

import numpy as np
import pytest

from tests.evaluation.trend_stop_support import synthetic_dataset
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.dev_realdata_study.protocol import PROTOCOL
from trade_rl.evaluation.dev_realdata_study.runner import reconcile_ledger
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.risk.pretrade import PreTradeRisk
from trade_rl.simulation import ExecutionCostConfig, MarketExecutor
from trade_rl.simulation.native_transition import (
    execute_native_target_transition,
    execute_terminal_settlement_transition,
)
from trade_rl.simulation.orders.terminal_closeout import TerminalCloseoutAuthority
from trade_rl.simulation.stateful.portfolio_session import SharedPortfolioSession
from trade_rl.strategies.allocation.buy_and_hold import BuyAndHoldPortfolioPolicy

SYMBOLS = ("AAAUSDT", "BBBUSDT")
MINIMUM_NOTIONAL = 5.0
CAPITAL = 100.0
DROP_ROW = 8


def _dust_dataset() -> MarketDataset:
    """Flat 100.0 prices that halve at ``DROP_ROW``; lot 0.01, minimum notional 5."""

    base = synthetic_dataset(
        "4h",
        SYMBOLS,
        first_close="2021-01-01T00:00:00",
        last_close="2021-01-05T00:00:00",
    )
    shape = (base.n_bars, len(SYMBOLS))
    price = np.full(shape, 100.0)
    price[DROP_ROW:] = 50.0
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.init}
    fields.update(
        open=price.copy(),
        high=price.copy(),
        low=price.copy(),
        close=price.copy(),
        lot_size=np.full(shape, 0.01),
        minimum_notional=np.full(shape, MINIMUM_NOTIONAL),
    )
    return MarketDataset(**fields)


def _replay(dataset: MarketDataset):
    # Two positions of 6 USDT each pass the 5 USDT entry filter, then fall to 3 USDT.
    return run_shared_portfolio_replay(
        dataset,
        BuyAndHoldPortfolioPolicy(symbols=SYMBOLS, gross_budget=0.12),
        start_index=1,
        stop_index=dataset.n_bars - 1,
        initial_capital=CAPITAL,
        execution_cost=ExecutionCostConfig(max_participation_rate=0.5),
        settle_terminal_position=True,
        capture_ledger_evidence=True,
    )


def test_terminal_settlement_closes_positions_below_minimum_notional() -> None:
    dataset = _dust_dataset()
    result = _replay(dataset)
    ledger = result.ledger_evidence
    assert ledger is not None
    opened = [
        event
        for interval in ledger.intervals.intervals[:3]
        for event in interval.order_events
        if event.filled_quantity
    ]
    assert len(opened) == len(SYMBOLS)  # entries really filled at 6 USDT each
    assert all(
        Fraction(value) == 0 for value in ledger.intervals.terminal_exact_quantities
    )


def test_ledger_reconciliation_passes_when_terminal_notional_is_below_minimum() -> None:
    dataset = _dust_dataset()
    protocol = dataclasses.replace(PROTOCOL, initial_capital=CAPITAL)
    reconciliation = reconcile_ledger(_replay(dataset), dataset, protocol)
    assert reconciliation["terminal_flat"] is True
    assert reconciliation["terminal_residual_quantities"] == {}
    assert reconciliation["passed"] is True
    assert (
        abs(float(reconciliation["equity_residual"]))
        <= protocol.reconciliation_tolerance
    )


COST = ExecutionCostConfig(max_participation_rate=0.5)


def _ordinary_dust_state() -> tuple[MarketDataset, MarketExecutor, object, int]:
    """Two 3 USDT positions (below the 5 USDT minimum) at a non-terminal interval."""

    dataset = _dust_dataset()
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
    assert session.can_decide  # still an ordinary (non-terminal) interval
    book = session.book
    assert all(float(abs(v)) * 50.0 < MINIMUM_NOTIONAL for v in book.quantities)
    assert all(Fraction(v) != 0 for v in book.exact_quantities)
    return dataset, MarketExecutor(dataset, COST), book, session.current_index


def _held(book: object) -> list[Fraction]:
    return [Fraction(v) for v in book.exact_quantities]  # type: ignore[attr-defined]


def test_ordinary_zero_target_keeps_sub_minimum_position() -> None:
    dataset, executor, book, index = _ordinary_dust_state()
    result = executor.execute_interval(book, np.zeros(2), start_index=index, bars=1)
    assert _held(result.book) == _held(book)
    assert result.fill_count == 0


def test_public_closeout_symbols_keyword_is_gone() -> None:
    dataset, executor, book, index = _ordinary_dust_state()
    assert (
        "closeout_symbols"
        not in inspect.signature(MarketExecutor.execute_interval).parameters
    )
    with pytest.raises(TypeError):
        executor.execute_interval(
            book,
            np.zeros(2),
            start_index=index,
            bars=1,
            closeout_symbols=(0, 1),  # type: ignore[call-arg]
        )


def test_forged_terminal_authority_is_rejected() -> None:
    dataset, executor, book, index = _ordinary_dust_state()
    with pytest.raises(TypeError):
        TerminalCloseoutAuthority(symbols=(0, 1), start_index=index)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        executor.execute_interval(
            book,
            np.zeros(2),
            start_index=index,
            bars=1,
            terminal_closeout=(0, 1),  # type: ignore[arg-type]
        )


def test_caller_emergency_mask_at_an_ordinary_interval_keeps_the_minimum() -> None:
    dataset, executor, book, index = _ordinary_dust_state()
    _, result, _ = execute_native_target_transition(
        executor,
        book,
        PreTradeRisk.default_for_execution(max_leverage=executor.cost.max_leverage),
        np.zeros(2),
        start_index=index,
        emergency_flatten_mask=np.array([True, True]),
    )
    assert _held(result.book) == _held(book)
    assert result.fill_count == 0


def test_minimum_notional_override_is_not_a_public_argument() -> None:
    dataset, executor, book, index = _ordinary_dust_state()
    assert (
        "closeout"
        not in inspect.signature(MarketExecutor.order_minimum_notional).parameters
    )
    with pytest.raises(TypeError):
        executor.order_minimum_notional(
            object(),  # type: ignore[arg-type]
            MINIMUM_NOTIONAL,
            closeout=True,  # type: ignore[call-arg]
        )


def test_terminal_entry_point_rejects_a_non_terminal_interval() -> None:
    dataset, executor, book, index = _ordinary_dust_state()
    risk = PreTradeRisk.default_for_execution(max_leverage=executor.cost.max_leverage)
    stop = dataset.n_bars - 1
    for agent_stop in (
        stop - 1,
        index + 3,
    ):  # right window, wrong interval / bad window
        with pytest.raises(ValueError):
            execute_terminal_settlement_transition(
                executor,
                book,
                risk,
                start_index=index,
                agent_stop_index=agent_stop,
                stop_index=stop,
                flatten_mask=np.array([True, True]),
            )


def test_true_terminal_settlement_pays_costs_and_conserves_the_ledger() -> None:
    dataset = _dust_dataset()
    result = _replay(dataset)
    ledger = result.ledger_evidence
    assert ledger is not None
    last = ledger.intervals.intervals[-1]
    closes = [event for event in last.order_events if event.filled_quantity]
    assert len(closes) == len(SYMBOLS)
    assert last.interval_cost > 0.0
    protocol = dataclasses.replace(PROTOCOL, initial_capital=CAPITAL)
    reconciliation = reconcile_ledger(result, dataset, protocol)
    assert reconciliation["passed"] is True
    assert (
        abs(float(reconciliation["cash_residual"])) <= protocol.reconciliation_tolerance
    )


def test_runs_that_already_end_flat_are_byte_identical() -> None:
    # No price drop: both 6 USDT positions are still above the 5 USDT minimum, so
    # the terminal close never needed the waiver. The trace digest below was
    # computed on the unmodified engine (origin/main 24ad01497) and must not move.
    base = _dust_dataset()
    shape = (base.n_bars, len(SYMBOLS))
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base) if f.init}
    flat_price = np.full(shape, 100.0)
    fields.update(
        open=flat_price.copy(),
        high=flat_price.copy(),
        low=flat_price.copy(),
        close=flat_price.copy(),
    )
    result = _replay(MarketDataset(**fields))
    ledger = result.ledger_evidence
    assert ledger is not None
    digest = hashlib.sha256()
    for interval in ledger.intervals.intervals:
        digest.update(repr(sorted(interval.to_mapping().items())).encode())
    assert all(Fraction(v) == 0 for v in ledger.intervals.terminal_exact_quantities)
    assert digest.hexdigest() == FLAT_RUN_TRACE_DIGEST


FLAT_RUN_TRACE_DIGEST = (
    "c51bc995dbb51b88740a0e2eadb979e3962d463c1ad78e23ecee6b880f05669d"
)
