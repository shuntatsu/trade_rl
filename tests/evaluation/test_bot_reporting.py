from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from trade_rl.evaluation.bot import (
    BotConfig,
    BotReport,
    calculate_bot_report,
    generate_demo_dataset,
    print_report_table,
    run_trading_bot,
)
from trade_rl.evaluation.replay import SharedCashReplayResult
from trade_rl.simulation import EconomicTerminationReason, ExecutionCostConfig


@pytest.fixture
def cash_replay() -> SharedCashReplayResult:
    dataset = generate_demo_dataset(n_bars=4, n_symbols=1)
    result, _ = run_trading_bot(dataset, BotConfig(strategy_name="cash"))
    return result


def test_bot_report_without_ledger_evidence_does_not_claim_terminal_settlement(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = BotReport(
        strategy_name="manual",
        initial_capital=100.0,
        final_equity=100.0,
        net_pnl=0.0,
        total_return_pct=0.0,
        max_drawdown_pct=0.0,
        nonzero_return_intervals=0,
        positive_return_rate_pct=0.0,
        interval_profit_factor=0.0,
        sharpe_ratio=0.0,
        is_profitable=False,
    )

    assert report.terminal_settled is None
    print_report_table([report])
    assert "unknown" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("reason", "wire_value"),
    [
        (EconomicTerminationReason.MINIMUM_EQUITY, "minimum_equity"),
        (
            EconomicTerminationReason.EXECUTION_COST_EXHAUSTION,
            "execution_cost_exhaustion",
        ),
        (EconomicTerminationReason.MARGIN_CALL, "margin_call"),
        (EconomicTerminationReason.LIQUIDATION, "liquidation"),
        (EconomicTerminationReason.DRAWDOWN_STOP, "drawdown_stop"),
        (EconomicTerminationReason.INSOLVENCY, "insolvency"),
    ],
)
def test_bot_report_json_uses_canonical_economic_termination_values(
    cash_replay: SharedCashReplayResult,
    reason: EconomicTerminationReason,
    wire_value: str,
) -> None:
    book = cash_replay.book.clone()
    book.terminate(reason)
    assert book.termination_reason is reason
    ledger = cash_replay.ledger_evidence
    assert ledger is not None
    terminated = replace(
        cash_replay,
        book=book,
        ledger_evidence=replace(ledger, termination_reason=wire_value),
    )

    report = calculate_bot_report(terminated, "cash")
    payload = json.loads(json.dumps(asdict(report)))

    assert payload["termination_reason"] == wire_value
    assert not payload["terminal_settled"]


def test_flat_book_with_active_order_is_unsettled_and_preserves_remainder(
    cash_replay: SharedCashReplayResult,
) -> None:
    assert all(quantity == 0 for quantity in cash_replay.book.exact_quantities)
    ledger = cash_replay.ledger_evidence
    assert ledger is not None
    order_id = "a" * 64
    active_order = ((order_id, 0.5),)
    pending = replace(
        cash_replay,
        ledger_evidence=replace(ledger, active_order_remainders=active_order),
    )

    report = calculate_bot_report(pending, "cash")
    payload = json.loads(json.dumps(asdict(report)))

    assert not report.terminal_settled
    assert report.active_order_remainders == active_order
    assert payload["active_order_remainders"] == [[order_id, 0.5]]


def test_bot_cost_diagnostics_match_flat_price_round_trip() -> None:
    dataset = generate_demo_dataset(n_bars=3, n_symbols=1)
    prices = np.full_like(dataset.close, 100.0)
    dataset = replace(
        dataset,
        open=prices,
        high=prices,
        low=prices,
        close=prices,
        identity_payload_json=None,
    )
    _, report = run_trading_bot(
        dataset,
        BotConfig(
            strategy_name="constant_long",
            initial_capital=1000.0,
            execution_cost=ExecutionCostConfig(
                fee_rate=0.01,
                spread_rate=0.0,
                impact_rate=0.0,
            ),
        ),
    )
    payload = json.loads(json.dumps(asdict(report)))

    # Two units enter and exit at 100: each fill pays 2 in account currency.
    assert payload["total_execution_cost"] == pytest.approx(4.0)
    assert payload["funding_pnl"] == 0.0
    assert payload["borrow_cost"] == 0.0
    assert payload["fill_count"] == 2
    assert payload["rebalance_events"] == 2
    assert payload["net_pnl"] == pytest.approx(-4.0)
    assert 0.4 <= payload["turnover_total"] < 0.41
    assert payload["terminal_settled"]


def test_cash_cost_diagnostics_are_zero(cash_replay: SharedCashReplayResult) -> None:
    report = calculate_bot_report(cash_replay, "cash")
    assert (
        report.total_execution_cost == report.funding_pnl == report.borrow_cost == 0.0
    )
    assert report.turnover_total == 0.0
    assert report.fill_count == report.rebalance_events == 0
