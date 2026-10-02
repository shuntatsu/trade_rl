from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from trade_rl.evaluation.bot import (
    BotConfig,
    calculate_bot_report,
    generate_demo_dataset,
    run_trading_bot,
)
from trade_rl.evaluation.replay import SharedCashReplayResult
from trade_rl.simulation import EconomicTerminationReason


@pytest.fixture
def cash_replay() -> SharedCashReplayResult:
    dataset = generate_demo_dataset(n_bars=4, n_symbols=1)
    result, _ = run_trading_bot(dataset, BotConfig(strategy_name="cash"))
    return result


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
